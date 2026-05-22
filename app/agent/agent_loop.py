from __future__ import annotations

import time
import asyncio
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from app.agent.guard import GuardResult, RuntimeGuard
from app.agent.state import AgentState
from app.agent.trace import AgentTraceRecorder
from app.core.errors import normalize_error
from app.core.logging import get_logger
from app.generation.response_generator import ResponseGenerator
from app.planning.diagnosis_planner import DiagnosisPlanner
from app.planning.learning_path_planner import LearningPathPlanner
from app.rag.evidence_builder import EvidenceBuilder
from app.rag.retriever import RagRetriever
from app.services.handoff_service import HandoffService
from app.stores.trace_store import TraceStore
from app.tools.runtime import FunctionCallRuntime, ToolCallContext


logger = get_logger(__name__)


ALLOWED_ACTIONS = {
    "learning_path": {
        "read_memory",
        "search_courses",
        "inspect_course_detail",
        "retrieve_evidence",
        "run_learning_path_planner",
        "ask_clarification",
        "generate_response",
        "finish",
    },
    "diagnosis": {
        "read_memory",
        "search_courses",
        "inspect_course_detail",
        "retrieve_evidence",
        "run_diagnosis_planner",
        "ask_clarification",
        "generate_response",
        "finish",
    },
}


@dataclass
class AgentAction:
    name: str
    input: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentObservation:
    action: str
    output: Dict[str, Any] = field(default_factory=dict)
    expected: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    passed: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LoopPolicy:
    """Validate actions and enforce task-specific allowed action sets."""

    def allowed_actions(self, task_type: str) -> set[str]:
        return ALLOWED_ACTIONS.get(task_type, set())

    def validate(self, action: AgentAction, state: AgentState) -> AgentAction:
        allowed = self.allowed_actions(state.task_type)
        if action.name in allowed:
            return action
        return self.repair(action, state)

    def repair(self, action: AgentAction, state: AgentState) -> AgentAction:
        if state.task_type == "learning_path":
            return AgentAction("run_learning_path_planner")
        if state.task_type == "diagnosis":
            return AgentAction("run_diagnosis_planner")
        return AgentAction("finish")


class RuleBasedActionPlanner:
    """Deterministic action planner over AgentState observations."""

    def next_action(self, state: AgentState) -> AgentAction:
        if state.metadata.get("agent_loop_finished"):
            return AgentAction("finish")
        if not state.metadata.get("memory_observed"):
            return AgentAction("read_memory")
        if state.metadata.get("needs_clarification") and not self._has_partial_result(state):
            return AgentAction("ask_clarification")
        if not state.metadata.get("tool_search_courses_observed"):
            return AgentAction("search_courses", {"query": state.query})
        if state.metadata.get("tool_candidate_courses") and not state.metadata.get("tool_course_detail_observed"):
            return AgentAction("inspect_course_detail")
        if state.evidence_package is None:
            return AgentAction("retrieve_evidence", {"query": state.query, "task_type": state.task_type})
        if state.task_type == "learning_path" and "learning_path" not in state.plan:
            return AgentAction("run_learning_path_planner")
        if state.task_type == "diagnosis" and "diagnosis" not in state.plan:
            return AgentAction("run_diagnosis_planner")
        if not state.final_answer:
            return AgentAction("generate_response")
        return AgentAction("finish")

    def _has_partial_result(self, state: AgentState) -> bool:
        return bool(state.evidence_package or state.plan or state.metadata.get("learning_path_plan") or state.metadata.get("diagnosis_plan"))


class ObservationChecker:
    """Check whether an action produced enough signal for the next step."""

    def check(self, action: AgentAction, state: AgentState, output: Dict[str, Any]) -> AgentObservation:
        if action.name == "read_memory":
            expected = {"memory_context": "present", "tool": "get_user_context"}
            passed = bool(state.memory_context) and bool(output.get("tool_observed"))
        elif action.name == "search_courses":
            expected = {"tool": "search_courses", "tool_observed": True}
            passed = bool(output.get("tool_observed"))
        elif action.name == "inspect_course_detail":
            expected = {"tool": "get_course_detail_or_skipped", "tool_observed": True}
            passed = bool(output.get("tool_observed"))
        elif action.name == "retrieve_evidence":
            expected = {"min_evidence_count": 1}
            passed = int(output.get("evidence_count") or 0) >= 1
        elif action.name == "run_learning_path_planner":
            expected = {"stage_count": ">=1", "needs_more_info": False}
            passed = int(output.get("stage_count") or 0) >= 1
        elif action.name == "run_diagnosis_planner":
            expected = {"diagnosis_type": "not_unknown_or_has_remedial_plan"}
            passed = bool(output.get("remedial_plan_count")) or str(output.get("diagnosis_type") or "") != "unknown"
        elif action.name == "generate_response":
            expected = {"final_answer": "non_empty"}
            passed = bool(state.final_answer)
        else:
            expected = {}
            passed = True
        return AgentObservation(action=action.name, output=output, expected=expected, success=True, passed=passed)


class ActionExecutor:
    """Execute loop actions and mutate AgentState."""

    def __init__(
        self,
        retriever: RagRetriever,
        evidence_builder: EvidenceBuilder,
        learning_path_planner: LearningPathPlanner,
        diagnosis_planner: DiagnosisPlanner,
        response_generator: ResponseGenerator,
        function_call_runtime: FunctionCallRuntime,
    ):
        self.retriever = retriever
        self.evidence_builder = evidence_builder
        self.learning_path_planner = learning_path_planner
        self.diagnosis_planner = diagnosis_planner
        self.response_generator = response_generator
        self.function_call_runtime = function_call_runtime

    def execute(
        self,
        action: AgentAction,
        state: AgentState,
        top_k: int,
        use_llm_rerank: bool,
        use_llm_generation: bool,
    ) -> Dict[str, Any]:
        if action.name == "read_memory":
            tool_result = self._execute_tool(
                state=state,
                tool_name="get_user_context",
                arguments={},
                permissions={"tool:read"},
            )
            if tool_result.ok and tool_result.result:
                state.set_user_context(self._merge_user_context(state.user_context, tool_result.result))
            state.add_metadata("tool_get_user_context", tool_result.to_dict())
            state.add_metadata("memory_observed", True)
            planning_context = state.memory_context.get("planning_context") or {}
            return {
                "tool_observed": True,
                "tool_ok": tool_result.ok,
                "tool_error": tool_result.error,
                "learning_stage": planning_context.get("learning_stage") or "",
                "goal": planning_context.get("goal") or "",
                "weak_point_count": len(planning_context.get("weak_points") or []),
            }
        if action.name == "search_courses":
            tool_result = self._execute_tool(
                state=state,
                tool_name="search_courses",
                arguments={"query": str(action.input.get("query") or state.query), "limit": min(max(top_k, 1), 10)},
                permissions={"tool:read"},
            )
            items = list((tool_result.result or {}).get("items") or []) if tool_result.ok else []
            state.add_metadata("tool_search_courses", tool_result.to_dict())
            state.add_metadata("tool_search_courses_observed", True)
            state.add_metadata("tool_candidate_courses", items)
            return {
                "tool_observed": True,
                "tool_ok": tool_result.ok,
                "tool_error": tool_result.error,
                "candidate_count": len(items),
                "candidate_ids": [item.get("id") for item in items[:5]],
            }
        if action.name == "inspect_course_detail":
            course_id = self._course_id_for_detail(state)
            if not course_id:
                state.add_metadata("tool_course_detail_observed", True)
                return {
                    "tool_observed": True,
                    "tool_ok": True,
                    "skipped": True,
                    "reason": "no_course_candidate",
                }
            tool_result = self._execute_tool(
                state=state,
                tool_name="get_course_detail",
                arguments={"course_id": course_id},
                permissions={"tool:read"},
            )
            state.add_metadata("tool_get_course_detail", tool_result.to_dict())
            state.add_metadata("tool_course_detail_observed", True)
            return {
                "tool_observed": True,
                "tool_ok": tool_result.ok,
                "tool_error": tool_result.error,
                "course_id": course_id,
                "has_course": bool((tool_result.result or {}).get("course")),
            }
        if action.name == "retrieve_evidence":
            retrieval_results = self.retriever.retrieve(
                query=state.query,
                top_k=top_k,
                task_type=state.task_type,
                user_profile=self._user_profile_for_rag(state),
                knowledge_state=self._knowledge_state_for_rag(state),
                use_llm_rerank=use_llm_rerank,
            )
            evidence_package = self.evidence_builder.build(
                query=state.query,
                task_type=state.task_type,
                retrieval_results=retrieval_results,
                user_profile=self._user_profile_for_rag(state),
                knowledge_state=self._knowledge_state_for_rag(state),
            )
            state.set_retrieval_results(retrieval_results)
            state.set_evidence_package(evidence_package)
            return {
                "retrieval_count": len(retrieval_results),
                "evidence_count": len(evidence_package.evidence_items),
            }
        if action.name == "run_learning_path_planner":
            plan = self.learning_path_planner.plan(state)
            payload = plan.to_dict()
            state.add_metadata("learning_path_plan", payload)
            return {
                "stage_count": len(payload.get("estimated_path") or []),
                "needs_more_info": payload.get("needs_more_info"),
                "clarification_questions": payload.get("clarification_questions") or [],
            }
        if action.name == "run_diagnosis_planner":
            diagnosis = self.diagnosis_planner.diagnose(state)
            payload = diagnosis.to_dict()
            state.add_metadata("diagnosis_plan", payload)
            return {
                "diagnosis_type": payload.get("diagnosis_type"),
                "remedial_plan_count": len(payload.get("remedial_plan") or []),
                "needs_more_info": payload.get("needs_more_info"),
                "clarification_questions": payload.get("clarification_questions") or [],
            }
        if action.name == "ask_clarification":
            questions = self._clarification_questions(state)
            answer = "为了更准确地继续，我需要先确认：\n" + "\n".join(
                "{0}. {1}".format(index, question) for index, question in enumerate(questions, start=1)
            )
            state.set_final_answer(answer)
            if state.routing_decision:
                state.routing_decision.needs_clarification = True
                state.routing_decision.information_sufficient = False
                state.routing_decision.clarification_questions = questions
            return {"questions": questions, "final_answer": answer}
        if action.name == "generate_response":
            answer = self.response_generator.generate(state, use_llm=use_llm_generation)
            return {"final_answer": answer, "llm_used": state.metadata.get("response_generation", {}).get("llm_used")}
        if action.name == "finish":
            state.add_metadata("agent_loop_finished", True)
            return {"finished": True}
        raise ValueError("Unknown action: {0}".format(action.name))

    def _user_profile_for_rag(self, state: AgentState) -> Dict[str, Any]:
        profile = dict(state.user_context.get("profile") or {})
        ranking_context = state.memory_context.get("ranking_context") or {}
        retrieval_context = state.memory_context.get("retrieval_context") or {}
        profile["preferred_resource_types"] = ranking_context.get("preferred_resource_types") or profile.get("preferred_resource_types") or []
        profile["preferred_subjects"] = retrieval_context.get("preferred_subjects") or profile.get("preferred_subjects") or []
        profile["constraints"] = ranking_context.get("constraints") or profile.get("constraints") or {}
        profile["memory_ranking"] = ranking_context
        profile["memory_retrieval"] = retrieval_context
        return profile

    def _knowledge_state_for_rag(self, state: AgentState) -> List[Dict[str, Any]]:
        knowledge = state.memory_context.get("raw_memory", {}).get("knowledge_state", {})
        points = knowledge.get("all_points") if isinstance(knowledge, dict) else None
        return list(points or state.user_context.get("knowledge_state") or [])

    def _clarification_questions(self, state: AgentState) -> List[str]:
        if state.routing_decision and state.routing_decision.clarification_questions:
            return state.routing_decision.clarification_questions
        if state.task_type == "diagnosis":
            return ["你具体卡住的是哪个知识点、章节或练习任务？"]
        return ["你希望规划哪个方向、当前基础如何、目标是什么？"]

    def _execute_tool(
        self,
        state: AgentState,
        tool_name: str,
        arguments: Dict[str, Any],
        permissions: set[str],
    ) -> Any:
        context = ToolCallContext(
            user_id=state.user_id,
            session_id=state.session_id,
            trace_run_id=state.trace_run_id,
            task_type=state.task_type,
            turn_id=str(len(state.steps) + 1),
            permissions=permissions,
            state=state,
        )
        return self._run_async(self.function_call_runtime.execute(tool_name, arguments, context))

    def _run_async(self, awaitable: Any) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(awaitable)
        raise RuntimeError("AgentLoop currently requires a synchronous execution context for tool calls.")

    def _merge_user_context(self, current: Dict[str, Any], tool_result: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(current or {})
        for key in ["user_id", "profile", "knowledge_state", "feedback", "recent_recommendations"]:
            if key in tool_result:
                merged[key] = tool_result[key]
        return merged

    def _course_id_for_detail(self, state: AgentState) -> str:
        candidates = state.metadata.get("tool_candidate_courses") or []
        if not candidates:
            return ""
        first = candidates[0] or {}
        return str(first.get("id") or first.get("resource_id") or "")


class AgentLoop:
    """Policy-guided ReAct loop for learning path and diagnosis tasks."""

    def __init__(
        self,
        retriever: RagRetriever,
        evidence_builder: EvidenceBuilder,
        learning_path_planner: LearningPathPlanner,
        diagnosis_planner: DiagnosisPlanner,
        response_generator: ResponseGenerator,
        trace_store: Optional[TraceStore] = None,
        function_call_runtime: Optional[FunctionCallRuntime] = None,
        policy: Optional[LoopPolicy] = None,
        action_planner: Optional[RuleBasedActionPlanner] = None,
        observation_checker: Optional[ObservationChecker] = None,
        max_iterations: int = 6,
        runtime_guard: Optional[RuntimeGuard] = None,
        handoff_service: Optional[HandoffService] = None,
    ):
        self.policy = policy or LoopPolicy()
        self.action_planner = action_planner or RuleBasedActionPlanner()
        self.observation_checker = observation_checker or ObservationChecker()
        self.executor = ActionExecutor(
            retriever=retriever,
            evidence_builder=evidence_builder,
            learning_path_planner=learning_path_planner,
            diagnosis_planner=diagnosis_planner,
            response_generator=response_generator,
            function_call_runtime=function_call_runtime or FunctionCallRuntime(trace_store=trace_store),
        )
        self.trace_store = trace_store
        self.trace_recorder = AgentTraceRecorder(trace_store)
        self.max_iterations = max(1, int(max_iterations))
        self.runtime_guard = runtime_guard or RuntimeGuard(max_iterations=self.max_iterations)
        self.handoff_service = handoff_service or HandoffService()

    def run(
        self,
        state: AgentState,
        top_k: int = 8,
        use_llm_rerank: bool = True,
        use_llm_generation: bool = True,
    ) -> AgentState:
        if state.task_type not in {"learning_path", "diagnosis"}:
            state.add_error("agent_loop", "AgentLoop only supports learning_path and diagnosis.")
            return state

        started_at = time.monotonic()
        for iteration in range(1, self.max_iterations + 1):
            guard_result = self.runtime_guard.check_before_step(state, iteration, started_at)
            if self._apply_guard_result(state, guard_result, iteration, "agent_loop.guard_before"):
                break

            proposed_action = self.action_planner.next_action(state)
            action = self.policy.validate(proposed_action, state)
            step_started_at = time.monotonic()
            try:
                output = self.executor.execute(
                    action=action,
                    state=state,
                    top_k=top_k,
                    use_llm_rerank=use_llm_rerank,
                    use_llm_generation=use_llm_generation,
                )
                observation = self.observation_checker.check(action, state, output)
            except Exception as exc:
                error_payload = normalize_error(exc, stage=action.name).to_dict()
                observation = AgentObservation(
                    action=action.name,
                    output={},
                    expected={},
                    success=False,
                    passed=False,
                    error=str(exc),
                )
                state.add_error(action.name, error_payload)
                logger.exception("agent loop action failed: %s", action.name)

            step_elapsed = time.monotonic() - step_started_at
            observation.output["step_elapsed_seconds"] = round(step_elapsed, 4)
            state.add_step(
                name=action.name,
                action_input=action.input,
                observation=observation.to_dict(),
                index=iteration,
            )
            self._trace_step(state, iteration, action, observation)
            self._handle_observation(state, action, observation)
            guard_result = self.runtime_guard.check_after_step(
                state=state,
                action_name=action.name,
                observation=observation.to_dict(),
                step_elapsed_seconds=step_elapsed,
            )
            if self._apply_guard_result(state, guard_result, iteration, "agent_loop.guard_after"):
                break

            if action.name == "finish" or state.metadata.get("agent_loop_finished"):
                break
            if action.name == "ask_clarification":
                break

        guard_result = self.runtime_guard.check_loop_exhausted(state)
        if self._apply_guard_result(state, guard_result, len(state.steps) + 1, "agent_loop.guard_exhausted"):
            state.add_metadata("agent_loop_steps", len(state.steps))
            return state

        if not state.final_answer:
            self.executor.execute(
                action=AgentAction("generate_response"),
                state=state,
                top_k=top_k,
                use_llm_rerank=use_llm_rerank,
                use_llm_generation=use_llm_generation,
            )
        state.add_metadata("agent_loop_steps", len(state.steps))
        return state

    def _handle_observation(self, state: AgentState, action: AgentAction, observation: AgentObservation) -> None:
        if observation.passed:
            return
        if action.name == "retrieve_evidence":
            state.add_metadata("needs_clarification", True)
            state.add_metadata("agent_loop_observation_failure", "insufficient_evidence")
        if action.name in {"run_learning_path_planner", "run_diagnosis_planner"}:
            questions = observation.output.get("clarification_questions") or []
            if questions and state.routing_decision:
                state.routing_decision.clarification_questions = questions
            state.add_metadata("needs_clarification", True)
            state.add_metadata("agent_loop_observation_failure", "{0}_failed_expectation".format(action.name))

    def _trace_step(
        self,
        state: AgentState,
        step_index: int,
        action: AgentAction,
        observation: AgentObservation,
    ) -> None:
        if self.trace_store is None or not state.trace_run_id:
            return
        self.trace_recorder.step(
            state=state,
            step_index=step_index,
            action_name="agent_loop.{0}".format(action.name),
            action_input=action.input,
            observation=observation.to_dict(),
        )

    def _apply_guard_result(
        self,
        state: AgentState,
        guard_result: GuardResult,
        step_index: int,
        action_name: str,
    ) -> bool:
        if not guard_result.triggered:
            return False

        guard_payload = guard_result.to_dict()
        state.add_metadata("runtime_guard_last_result", guard_payload)

        if guard_result.controlled_fallback:
            state.add_metadata("needs_clarification", True)
            state.add_metadata("controlled_fallback", guard_payload)
            self._trace_guard(state, step_index, action_name, guard_payload)
            return False

        if guard_result.handoff_required:
            case = self.handoff_service.create_case(state, guard_result)
            state.set_handoff_case(case)
            state.add_metadata("handoff", {"case_id": case.get("id"), "reason_code": guard_result.reason_code})
            state.set_final_answer(self.handoff_service.handoff_answer(case))
            state.add_metadata("agent_loop_finished", True)
            state.add_step(action_name, {"guard": guard_payload}, {"handoff_case_id": case.get("id")}, index=step_index)
            self._trace_guard(
                state,
                step_index,
                action_name,
                {"guard": guard_payload, "handoff_case_id": case.get("id")},
            )
            return True
        return False

    def _trace_guard(
        self,
        state: AgentState,
        step_index: int,
        action_name: str,
        observation: Dict[str, Any],
    ) -> None:
        if self.trace_store is None or not state.trace_run_id:
            return
        self.trace_recorder.step(
            state=state,
            step_index=step_index,
            action_name=action_name,
            action_input={},
            observation=observation,
        )
