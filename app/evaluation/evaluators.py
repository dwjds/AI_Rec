from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

from app.agent.orchestrator import AgentOrchestrator
from app.evaluation.datasets import EvalCase
from app.evaluation.judge import RuleBasedJudge
from app.evaluation.metrics import (
    duplicate_rate,
    keyword_coverage,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    summarize_case_metrics,
    trace_action_pass_rate,
)
from app.rag.retriever import RagRetriever


@dataclass
class EvaluationConfig:
    top_k: int = 5
    use_llm_route: bool = False
    use_llm_rerank: bool = False
    use_llm_generation: bool = False


class EvaluationRunner:
    """Offline evaluator over retrieval, recommendation, loop, and guard cases."""

    def __init__(
        self,
        orchestrator: Optional[AgentOrchestrator] = None,
        retriever: Optional[RagRetriever] = None,
        judge: Optional[RuleBasedJudge] = None,
        config: Optional[EvaluationConfig] = None,
    ):
        self.orchestrator = orchestrator or AgentOrchestrator()
        self.retriever = retriever or RagRetriever()
        self.judge = judge or RuleBasedJudge()
        self.config = config or EvaluationConfig()

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
            else:
                sections[suite_name] = {"summary": {"case_count": 0.0}, "cases": []}
        return {
            "config": asdict(self.config),
            "summary": {name: section.get("summary", {}) for name, section in sections.items()},
            "sections": sections,
        }

    def evaluate_retrieval(self, cases: Sequence[EvalCase]) -> Dict[str, Any]:
        results = []
        for case in cases:
            retrieved = self.retriever.retrieve(
                query=case.query,
                top_k=self.config.top_k,
                task_type=case.task_type,
                use_llm_rerank=self.config.use_llm_rerank,
                use_llm_rewrite=False,
            )
            ids = [self._resource_id(item.to_dict()) for item in retrieved]
            text = " ".join([item.title + " " + item.content for item in retrieved])
            metrics = {
                "precision_at_k": precision_at_k(ids, case.expected_resource_ids, self.config.top_k),
                "recall_at_k": recall_at_k(ids, case.expected_resource_ids, self.config.top_k),
                "mrr": reciprocal_rank(ids, case.expected_resource_ids),
                "keyword_coverage": keyword_coverage(text, case.expected_keywords),
                "duplicate_rate": duplicate_rate(ids),
            }
            results.append(
                {
                    "case": case.to_dict(),
                    "predicted_resource_ids": ids,
                    "metrics": metrics,
                    "passed": self._case_passed(metrics, case),
                }
            )
        return {"summary": summarize_case_metrics(results), "cases": results}

    def evaluate_recommendation(self, cases: Sequence[EvalCase]) -> Dict[str, Any]:
        results = []
        for case in cases:
            result = self._run_agent(case)
            package = ((result.state.metadata if result.state else {}).get("recommendation_package") or {})
            recommendations = package.get("recommendations") or []
            ids = [
                str((item.get("candidate") or {}).get("resource_id") or "")
                for item in recommendations
            ]
            text = result.answer + " " + " ".join(str((item.get("candidate") or {}).get("title") or "") for item in recommendations)
            metrics = {
                "precision_at_k": precision_at_k(ids, case.expected_resource_ids, self.config.top_k),
                "recall_at_k": recall_at_k(ids, case.expected_resource_ids, self.config.top_k),
                "mrr": reciprocal_rank(ids, case.expected_resource_ids),
                "keyword_coverage": keyword_coverage(text, case.expected_keywords),
                "duplicate_rate": duplicate_rate(ids),
            }
            results.append(
                {
                    "case": case.to_dict(),
                    "pipeline": result.pipeline,
                    "predicted_resource_ids": ids,
                    "answer": result.answer,
                    "metrics": metrics,
                    "passed": self._case_passed(metrics, case),
                }
            )
        return {"summary": summarize_case_metrics(results), "cases": results}

    def evaluate_qa(self, cases: Sequence[EvalCase]) -> Dict[str, Any]:
        results = []
        for case in cases:
            result = self._run_agent(case)
            evidence_items = result.evidence_package.evidence_items if result.evidence_package else []
            evidence_text = " ".join(item.content for item in evidence_items)
            judge_result = self.judge.judge(result.answer, evidence_text=evidence_text, expected_keywords=case.expected_keywords)
            metrics = {
                "keyword_coverage": keyword_coverage(result.answer, case.expected_keywords),
                "groundedness_score": judge_result.groundedness_score,
                "usefulness_score": judge_result.usefulness_score,
            }
            results.append(
                {
                    "case": case.to_dict(),
                    "pipeline": result.pipeline,
                    "answer": result.answer,
                    "judge": judge_result.to_dict(),
                    "metrics": metrics,
                    "passed": self._case_passed(metrics, case),
                }
            )
        return {"summary": summarize_case_metrics(results), "cases": results}

    def evaluate_agent_loop(self, cases: Sequence[EvalCase]) -> Dict[str, Any]:
        results = []
        for case in cases:
            result = self._run_agent(case)
            actions = [step.name for step in (result.state.steps if result.state else [])]
            trace_actions = self._trace_actions(result.state.to_dict() if result.state else {})
            metrics = {
                "expected_trace_pass_rate": trace_action_pass_rate(actions + trace_actions, case.expected_trace_actions),
                "final_answer_present": 1.0 if result.answer else 0.0,
                "handoff_triggered": 1.0 if result.handoff_case else 0.0,
            }
            results.append(
                {
                    "case": case.to_dict(),
                    "pipeline": result.pipeline,
                    "actions": actions,
                    "trace_actions": trace_actions,
                    "metrics": metrics,
                    "passed": self._case_passed(metrics, case),
                }
            )
        return {"summary": summarize_case_metrics(results), "cases": results}

    def evaluate_failure_guard(self, cases: Sequence[EvalCase]) -> Dict[str, Any]:
        results = []
        for case in cases:
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
            results.append(
                {
                    "case": case.to_dict(),
                    "pipeline": result.pipeline,
                    "reason_code": reason_code,
                    "handoff_case_id": (result.handoff_case or {}).get("id") if result.handoff_case else None,
                    "metrics": metrics,
                    "passed": bool(expected and reason_code == expected),
                }
            )
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

    def _resource_id(self, result: Dict[str, Any]) -> str:
        metadata = result.get("metadata") or {}
        return str(metadata.get("source_resource_id") or result.get("chunk_id") or "")

    def _trace_actions(self, state_payload: Dict[str, Any]) -> List[str]:
        metadata = state_payload.get("metadata") or {}
        actions = []
        for key in ["tool_get_user_context", "tool_search_courses", "tool_get_course_detail"]:
            if key in metadata:
                actions.append(key)
        return actions

    def _case_passed(self, metrics: Dict[str, float], case: EvalCase) -> bool:
        if case.expected_resource_ids:
            return metrics.get("recall_at_k", 0.0) > 0.0
        if case.expected_keywords:
            return metrics.get("keyword_coverage", 0.0) > 0.0 or metrics.get("groundedness_score", 0.0) > 0.0
        if case.expected_trace_actions:
            return metrics.get("expected_trace_pass_rate", 0.0) >= 1.0
        if case.expected_handoff_reason:
            return metrics.get("reason_match", 0.0) >= 1.0
        return True
