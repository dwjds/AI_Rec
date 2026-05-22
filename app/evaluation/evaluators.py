from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, TypeVar

from app.agent.orchestrator import AgentOrchestrator
from app.evaluation.checkpoint import EvaluationCheckpoint
from app.evaluation.datasets import DemoFlowCase, EvalCase
from app.evaluation.judge import RuleBasedJudge
from app.evaluation.metrics import (
    answer_evidence_overlap,
    duplicate_rate,
    evidence_title_mention_rate,
    excluded_keyword_violation_rate,
    keyword_precision_at_k,
    keyword_reciprocal_rank,
    keyword_coverage,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    summarize_case_metrics,
    sufficiency_score,
    top1_keyword_coverage,
    trace_action_pass_rate,
    unsupported_title_mention_rate,
)
from app.rag.retriever import RagRetriever


@dataclass
class EvaluationConfig:
    top_k: int = 5
    use_llm_route: bool = True
    use_llm_rerank: bool = True
    use_llm_generation: bool = False
    show_progress: bool = True


T = TypeVar("T")


class EvaluationRunner:
    """Offline evaluator over retrieval, recommendation, loop, and guard cases."""

    def __init__(
        self,
        orchestrator: Optional[AgentOrchestrator] = None,
        retriever: Optional[RagRetriever] = None,
        judge: Optional[RuleBasedJudge] = None,
        config: Optional[EvaluationConfig] = None,
        checkpoint: Optional[EvaluationCheckpoint] = None,
    ):
        self.orchestrator = orchestrator or AgentOrchestrator()
        self.retriever = retriever or RagRetriever()
        self.judge = judge or RuleBasedJudge()
        self.config = config or EvaluationConfig()
        self.checkpoint = checkpoint

    def evaluate_suite(self, suites: Dict[str, Sequence[EvalCase]]) -> Dict[str, Any]:
        sections: Dict[str, Any] = {}
        for suite_name, cases in suites.items():
            if suite_name == "retrieval":
                sections[suite_name] = self.evaluate_retrieval(cases)
            elif suite_name == "recommendation":
                sections[suite_name] = self.evaluate_recommendation(cases)
            elif suite_name == "qa":
                sections[suite_name] = self.evaluate_qa(cases)
            elif suite_name == "agent_loop":
                sections[suite_name] = self.evaluate_agent_loop(cases)
            elif suite_name == "failure":
                sections[suite_name] = self.evaluate_failure_guard(cases)
            elif suite_name == "demo_flow":
                sections[suite_name] = self.evaluate_demo_flow(cases)  # type: ignore[arg-type]
            else:
                sections[suite_name] = {"summary": {"case_count": 0.0}, "cases": []}
        return {
            "config": asdict(self.config),
            "summary": {name: section.get("summary", {}) for name, section in sections.items()},
            "sections": sections,
        }

    def evaluate_retrieval(self, cases: Sequence[EvalCase]) -> Dict[str, Any]:
        results = []
        for case in self._iter_cases("retrieval", cases):
            cached = self._checkpoint_get("retrieval", case)
            if cached is not None:
                results.append(cached)
                continue
            retrieved = self.retriever.retrieve(
                query=case.query,
                top_k=self.config.top_k,
                task_type=case.task_type,
                use_llm_rerank=self.config.use_llm_rerank,
                use_llm_rewrite=True,
            )
            ids = [self._resource_id(item.to_dict()) for item in retrieved]
            result_texts = [item.title + " " + item.content for item in retrieved]
            text = " ".join(result_texts)
            id_metrics = self._id_metrics(ids, case)
            metrics = {
                "result_count": float(len(retrieved)),
                "retrieval_sufficiency": sufficiency_score(len(retrieved), self._min_evidence_count(case)),
                **id_metrics,
                "keyword_coverage": keyword_coverage(text, case.expected_keywords),
                "content_keyword_coverage": keyword_coverage(text, case.expected_keywords),
                "keyword_precision_at_k": keyword_precision_at_k(result_texts, case.expected_keywords, self.config.top_k),
                "keyword_mrr": keyword_reciprocal_rank(result_texts, case.expected_keywords),
                "top1_keyword_coverage": top1_keyword_coverage(result_texts, case.expected_keywords),
                "excluded_keyword_violation_rate": excluded_keyword_violation_rate(text, case.excluded_keywords),
                "duplicate_rate": duplicate_rate(ids),
            }
            case_result = {
                "case": case.to_dict(),
                "predicted_resource_ids": ids,
                "metrics": metrics,
                "passed": self._case_passed(metrics, case),
            }
            self._checkpoint_set("retrieval", case, case_result)
            results.append(case_result)
        return {"summary": summarize_case_metrics(results), "cases": results}

    def evaluate_recommendation(self, cases: Sequence[EvalCase]) -> Dict[str, Any]:
        results = []
        for case in self._iter_cases("recommendation", cases):
            cached = self._checkpoint_get("recommendation", case)
            if cached is not None:
                results.append(cached)
                continue
            result = self._run_agent(case)
            package = ((result.state.metadata if result.state else {}).get("recommendation_package") or {})
            recommendations = package.get("recommendations") or []
            ids = [
                str(item.get("resource_id") or (item.get("candidate") or {}).get("resource_id") or "")
                for item in recommendations
            ]
            recommendation_texts = [self._recommendation_text(item) for item in recommendations]
            text = result.answer + " " + " ".join(recommendation_texts)
            evidence_items = result.evidence_package.evidence_items if result.evidence_package else []
            evidence_text = self._evidence_text(evidence_items)
            evidence_titles = self._evidence_titles(evidence_items)
            id_metrics = self._id_metrics(ids, case)
            metrics = {
                **id_metrics,
                "keyword_coverage": keyword_coverage(text, case.expected_keywords),
                "recommendation_keyword_coverage": keyword_coverage(" ".join(recommendation_texts), case.expected_keywords),
                "recommendation_keyword_precision_at_k": keyword_precision_at_k(
                    recommendation_texts, case.expected_keywords, self.config.top_k
                ),
                "recommendation_keyword_mrr": keyword_reciprocal_rank(recommendation_texts, case.expected_keywords),
                "excluded_keyword_violation_rate": excluded_keyword_violation_rate(text, case.excluded_keywords),
                "duplicate_rate": duplicate_rate(ids),
                "evidence_count": float(len(evidence_items)),
                "evidence_sufficiency": sufficiency_score(len(evidence_items), self._min_evidence_count(case)),
                "evidence_keyword_coverage": keyword_coverage(evidence_text, case.expected_keywords),
                "answer_evidence_overlap": answer_evidence_overlap(result.answer, evidence_text),
                "evidence_title_mention_rate": evidence_title_mention_rate(result.answer, evidence_titles),
                "unsupported_title_mention_rate": unsupported_title_mention_rate(result.answer, evidence_titles),
            }
            case_result = {
                "case": case.to_dict(),
                "pipeline": result.pipeline,
                "predicted_resource_ids": ids,
                "answer": result.answer,
                "metrics": metrics,
                "passed": self._case_passed(metrics, case),
            }
            self._checkpoint_set("recommendation", case, case_result)
            results.append(case_result)
        return {"summary": summarize_case_metrics(results), "cases": results}

    def evaluate_qa(self, cases: Sequence[EvalCase]) -> Dict[str, Any]:
        results = []
        for case in self._iter_cases("qa", cases):
            cached = self._checkpoint_get("qa", case)
            if cached is not None:
                results.append(cached)
                continue
            result = self._run_agent(case)
            evidence_items = result.evidence_package.evidence_items if result.evidence_package else []
            evidence_text = self._evidence_text(evidence_items)
            evidence_titles = self._evidence_titles(evidence_items)
            judge_result = self.judge.judge(result.answer, evidence_text=evidence_text, expected_keywords=case.expected_keywords)
            metrics = {
                "keyword_coverage": keyword_coverage(result.answer, case.expected_keywords),
                "groundedness_score": judge_result.groundedness_score,
                "usefulness_score": judge_result.usefulness_score,
                "evidence_count": float(len(evidence_items)),
                "evidence_sufficiency": sufficiency_score(len(evidence_items), self._min_evidence_count(case)),
                "evidence_keyword_coverage": keyword_coverage(evidence_text, case.expected_keywords),
                "answer_evidence_overlap": answer_evidence_overlap(result.answer, evidence_text),
                "evidence_title_mention_rate": evidence_title_mention_rate(result.answer, evidence_titles),
                "unsupported_title_mention_rate": unsupported_title_mention_rate(result.answer, evidence_titles),
            }
            case_result = {
                "case": case.to_dict(),
                "pipeline": result.pipeline,
                "answer": result.answer,
                "judge": judge_result.to_dict(),
                "metrics": metrics,
                "passed": result.pipeline == "rag_qa"
                and metrics.get("evidence_sufficiency", 0.0) >= 1.0
                and self._case_passed(metrics, case),
            }
            self._checkpoint_set("qa", case, case_result)
            results.append(case_result)
        return {"summary": summarize_case_metrics(results), "cases": results}

    def evaluate_agent_loop(self, cases: Sequence[EvalCase]) -> Dict[str, Any]:
        results = []
        for case in self._iter_cases("agent_loop", cases):
            cached = self._checkpoint_get("agent_loop", case)
            if cached is not None:
                results.append(cached)
                continue
            result = self._run_agent(case)
            actions = [step.name for step in (result.state.steps if result.state else [])]
            trace_actions = self._trace_actions(result.state.to_dict() if result.state else {})
            metrics = {
                "expected_trace_pass_rate": trace_action_pass_rate(actions + trace_actions, case.expected_trace_actions),
                "final_answer_present": 1.0 if result.answer else 0.0,
                "handoff_triggered": 1.0 if result.handoff_case else 0.0,
            }
            case_result = {
                "case": case.to_dict(),
                "pipeline": result.pipeline,
                "actions": actions,
                "trace_actions": trace_actions,
                "metrics": metrics,
                "passed": self._case_passed(metrics, case),
            }
            self._checkpoint_set("agent_loop", case, case_result)
            results.append(case_result)
        return {"summary": summarize_case_metrics(results), "cases": results}

    def evaluate_failure_guard(self, cases: Sequence[EvalCase]) -> Dict[str, Any]:
        results = []
        for case in self._iter_cases("failure", cases):
            cached = self._checkpoint_get("failure", case)
            if cached is not None:
                results.append(cached)
                continue
            result = self._run_agent(case)
            reason_code = ""
            if result.handoff_case:
                reason_code = str(result.handoff_case.get("reason_code") or "")
            else:
                reason_code = str((result.metadata.get("handoff") or {}).get("reason_code") or "")
            expected = case.expected_handoff_reason or ""
            metrics = {
                "handoff_triggered": 1.0 if result.handoff_case else 0.0,
                "reason_match": 1.0 if expected and reason_code == expected else 0.0,
            }
            case_result = {
                "case": case.to_dict(),
                "pipeline": result.pipeline,
                "reason_code": reason_code,
                "handoff_case_id": (result.handoff_case or {}).get("id") if result.handoff_case else None,
                "metrics": metrics,
                "passed": bool(expected and reason_code == expected),
            }
            self._checkpoint_set("failure", case, case_result)
            results.append(case_result)
        return {"summary": summarize_case_metrics(results), "cases": results}

    def evaluate_demo_flow(self, cases: Sequence[DemoFlowCase]) -> Dict[str, Any]:
        results = []
        for case in self._iter_cases("demo_flow", cases):
            cached = self._checkpoint_get("demo_flow", case)  # type: ignore[arg-type]
            if cached is not None:
                results.append(cached)
                continue
            session_id = case.session_id or "demo_flow_{0}".format(case.id)
            turn_results = []
            for index, turn in enumerate(case.turns, start=1):
                result = self.orchestrator.run(
                    user_id=case.user_id,
                    query=turn.query,
                    session_id=session_id,
                    use_llm_route=self.config.use_llm_route,
                    use_llm_rerank=self.config.use_llm_rerank,
                    use_llm_generation=self.config.use_llm_generation,
                    top_k=self.config.top_k,
                )
                answer_text = result.answer or ""
                package = ((result.state.metadata if result.state else {}).get("recommendation_package") or {})
                recommendations = package.get("recommendations") or []
                recommendation_text = " ".join(str(item.get("title") or "") for item in recommendations)
                actions = [step.name for step in (result.state.steps if result.state else [])]
                trace_actions = self._trace_actions(result.state.to_dict() if result.state else {})
                metrics = {
                    "task_type_match": 1.0 if not turn.expected_task_type or result.routing_decision.task_type == turn.expected_task_type else 0.0,
                    "pipeline_match": 1.0 if not turn.expected_pipeline or result.pipeline == turn.expected_pipeline else 0.0,
                    "keyword_coverage": keyword_coverage(answer_text + " " + recommendation_text, turn.expected_keywords),
                    "excluded_keyword_violation_rate": excluded_keyword_violation_rate(answer_text + " " + recommendation_text, turn.excluded_keywords),
                    "expected_trace_pass_rate": trace_action_pass_rate(actions + trace_actions, turn.expected_trace_actions),
                    "final_answer_present": 1.0 if result.answer else 0.0,
                    "same_session": 1.0 if result.state and result.state.session_id == session_id else 0.0,
                }
                turn_passed = self._demo_turn_passed(metrics, turn)
                turn_results.append(
                    {
                        "index": index,
                        "query": turn.query,
                        "expected": turn.to_dict(),
                        "task_type": result.routing_decision.task_type,
                        "pipeline": result.pipeline,
                        "answer": result.answer,
                        "actions": actions,
                        "trace_actions": trace_actions,
                        "metrics": metrics,
                        "passed": turn_passed,
                    }
                )
            case_metrics = {
                "turn_count": float(len(turn_results)),
                "turn_pass_rate": sum(1.0 for item in turn_results if item.get("passed")) / float(len(turn_results) or 1),
                "same_session_rate": sum(float((item.get("metrics") or {}).get("same_session") or 0.0) for item in turn_results) / float(len(turn_results) or 1),
                "final_answer_present_rate": sum(float((item.get("metrics") or {}).get("final_answer_present") or 0.0) for item in turn_results) / float(len(turn_results) or 1),
            }
            case_result = {
                "case": case.to_dict(),
                "session_id": session_id,
                "turns": turn_results,
                "metrics": case_metrics,
                "passed": case_metrics["turn_pass_rate"] >= 1.0,
            }
            self._checkpoint_set("demo_flow", case, case_result)  # type: ignore[arg-type]
            results.append(case_result)
        return {"summary": summarize_case_metrics(results), "cases": results}

    def _run_agent(self, case: EvalCase) -> Any:
        return self.orchestrator.run(
            user_id=case.user_id,
            query=case.query,
            session_id=case.session_id,
            use_llm_route=self.config.use_llm_route,
            use_llm_rerank=self.config.use_llm_rerank,
            use_llm_generation=self.config.use_llm_generation,
            top_k=self.config.top_k,
        )

    def _iter_cases(self, suite_name: str, cases: Sequence[T]) -> Iterator[T]:
        if not self.config.show_progress:
            yield from cases
            return

        try:
            from tqdm import tqdm  # type: ignore

            yield from tqdm(
                cases,
                desc="eval:{0}".format(suite_name),
                unit="case",
                ncols=100,
                file=sys.stderr,
            )
            return
        except Exception:
            total = len(cases)
            for index, case in enumerate(cases, start=1):
                case_id = str(getattr(case, "id", index))
                print("[eval:{0}] {1}/{2} {3}".format(suite_name, index, total, case_id), file=sys.stderr, flush=True)
                yield case

    def _resource_id(self, result: Dict[str, Any]) -> str:
        metadata = result.get("metadata") or {}
        return str(metadata.get("source_resource_id") or result.get("chunk_id") or "")

    def _id_metrics(self, ids: Sequence[str], case: EvalCase) -> Dict[str, Any]:
        if not case.expected_resource_ids:
            return {
                "precision_at_k": None,
                "recall_at_k": None,
                "mrr": None,
                "id_ground_truth_available": 0.0,
            }
        return {
            "precision_at_k": precision_at_k(ids, case.expected_resource_ids, self.config.top_k),
            "recall_at_k": recall_at_k(ids, case.expected_resource_ids, self.config.top_k),
            "mrr": reciprocal_rank(ids, case.expected_resource_ids),
            "id_ground_truth_available": 1.0,
        }

    def _trace_actions(self, state_payload: Dict[str, Any]) -> List[str]:
        metadata = state_payload.get("metadata") or {}
        actions = []
        for key in ["tool_get_user_context", "tool_search_courses", "tool_get_course_detail"]:
            if key in metadata:
                actions.append(key)
        return actions

    def _case_passed(self, metrics: Dict[str, float], case: EvalCase) -> bool:
        exclusion_ok = metrics.get("excluded_keyword_violation_rate", 0.0) <= 0.0
        if case.metadata.get("min_evidence_count") is not None and metrics.get("evidence_sufficiency", 1.0) < 1.0:
            return False
        if case.expected_resource_ids:
            return metrics.get("recall_at_k", 0.0) > 0.0 and exclusion_ok
        if case.expected_keywords:
            return (metrics.get("keyword_coverage", 0.0) > 0.0 or metrics.get("groundedness_score", 0.0) > 0.0) and exclusion_ok
        if case.excluded_keywords:
            return exclusion_ok
        if case.expected_trace_actions:
            return metrics.get("expected_trace_pass_rate", 0.0) >= 1.0
        if case.expected_handoff_reason:
            return metrics.get("reason_match", 0.0) >= 1.0
        return True

    def _demo_turn_passed(self, metrics: Dict[str, float], turn: Any) -> bool:
        if turn.expected_task_type and metrics.get("task_type_match", 0.0) < 1.0:
            return False
        if turn.expected_pipeline and metrics.get("pipeline_match", 0.0) < 1.0:
            return False
        if turn.expected_keywords and metrics.get("keyword_coverage", 0.0) <= 0.0:
            return False
        if turn.excluded_keywords and metrics.get("excluded_keyword_violation_rate", 0.0) > 0.0:
            return False
        if turn.expected_trace_actions and metrics.get("expected_trace_pass_rate", 0.0) < 1.0:
            return False
        return metrics.get("final_answer_present", 0.0) >= 1.0 and metrics.get("same_session", 0.0) >= 1.0

    def _min_evidence_count(self, case: EvalCase) -> int:
        try:
            return int(case.metadata.get("min_evidence_count") or 1)
        except (TypeError, ValueError):
            return 1

    def _evidence_text(self, evidence_items: Sequence[Any]) -> str:
        parts = []
        for item in evidence_items:
            parts.append(str(getattr(item, "title", "") or ""))
            parts.append(str(getattr(item, "content", "") or ""))
            resource = getattr(item, "resource", {}) or {}
            if isinstance(resource, dict):
                parts.extend(str(value) for value in resource.values() if isinstance(value, (str, int, float)))
        return " ".join(parts)

    def _evidence_titles(self, evidence_items: Sequence[Any]) -> List[str]:
        return [str(getattr(item, "title", "") or "") for item in evidence_items if str(getattr(item, "title", "") or "")]

    def _recommendation_text(self, item: Dict[str, Any]) -> str:
        parts: List[str] = []
        candidate = item.get("candidate") or {}
        for source in [item, candidate]:
            if not isinstance(source, dict):
                continue
            for key in [
                "title",
                "resource_type",
                "difficulty",
                "subject",
                "sub_subject",
                "description",
                "summary",
                "recommendation_reason",
                "reason",
            ]:
                value = source.get(key)
                if value is not None:
                    parts.append(str(value))
            for key in ["knowledge_points", "matched_knowledge_points", "reasons"]:
                value = source.get(key)
                if isinstance(value, list):
                    parts.extend(str(item) for item in value if item)
                elif value:
                    parts.append(str(value))
        return " ".join(parts)

    def _checkpoint_get(self, suite_name: str, case: EvalCase) -> Optional[Dict[str, Any]]:
        if self.checkpoint is None:
            return None
        return self.checkpoint.get(suite_name, case.id)

    def _checkpoint_set(self, suite_name: str, case: EvalCase, result: Dict[str, Any]) -> None:
        if self.checkpoint is None:
            return
        self.checkpoint.set(suite_name, case.id, result)
