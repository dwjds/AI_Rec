from __future__ import annotations

import sqlite3
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, Optional

from app.agent.state import AgentState
from app.core.config import settings
from app.core.errors import AppError, DatabaseError, RetrievalError


LOOP_MAX_ITERATIONS_EXCEEDED = "loop_max_iterations_exceeded"
LOOP_TIMEOUT = "loop_timeout"
STEP_TIMEOUT = "step_timeout"
RETRIEVAL_NO_EVIDENCE = "retrieval_no_evidence"
RETRIEVAL_TOOL_ERROR = "retrieval_tool_error"
PLANNER_FAILED = "planner_failed"
LLM_VALIDATION_FAILED = "llm_validation_failed"
DATABASE_ERROR = "database_error"
VECTOR_STORE_ERROR = "vector_store_error"
RECOMMENDATION_EMPTY = "recommendation_empty"
UNEXPECTED_ERROR = "unexpected_error"


@dataclass
class GuardResult:
    triggered: bool
    reason_code: str = ""
    reason_text: str = ""
    handoff_required: bool = False
    controlled_fallback: bool = False
    retryable: bool = False
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def pass_(cls) -> "GuardResult":
        return cls(triggered=False)

    @classmethod
    def fallback(
        cls,
        reason_code: str,
        reason_text: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> "GuardResult":
        return cls(
            triggered=True,
            reason_code=reason_code,
            reason_text=reason_text,
            controlled_fallback=True,
            retryable=True,
            details=details or {},
        )

    @classmethod
    def handoff(
        cls,
        reason_code: str,
        reason_text: str,
        details: Optional[Dict[str, Any]] = None,
        retryable: bool = True,
    ) -> "GuardResult":
        return cls(
            triggered=True,
            reason_code=reason_code,
            reason_text=reason_text,
            handoff_required=True,
            retryable=retryable,
            details=details or {},
        )


class RuntimeGuard:
    """Engineering guardrails for the Agent runtime.

    The guard stays quiet for normal recoverable uncertainty. It only escalates
    when execution cannot finish reliably, for example repeated validation
    failures, infrastructure errors, timeouts, or loop exhaustion.
    """

    def __init__(
        self,
        max_iterations: Optional[int] = None,
        max_seconds: Optional[float] = None,
        step_timeout_seconds: Optional[float] = None,
        max_recoverable_failures: int = 1,
        llm_validation_failure_threshold: Optional[int] = None,
    ):
        self.max_iterations = max(1, int(max_iterations or settings.agent_max_iterations))
        self.max_seconds = max(0.001, float(max_seconds or settings.agent_loop_timeout_seconds))
        self.step_timeout_seconds = max(0.001, float(step_timeout_seconds or settings.agent_step_timeout_seconds))
        self.max_recoverable_failures = max(0, int(max_recoverable_failures))
        self.llm_validation_failure_threshold = max(
            1,
            int(llm_validation_failure_threshold or settings.llm_validation_failure_threshold),
        )

    def check_before_step(self, state: AgentState, iteration: int, started_at: float) -> GuardResult:
        elapsed = time.monotonic() - started_at
        if elapsed > self.max_seconds:
            return GuardResult.handoff(
                LOOP_TIMEOUT,
                "AgentLoop 执行超过整体超时限制。",
                {"elapsed_seconds": round(elapsed, 4), "max_seconds": self.max_seconds, "iteration": iteration},
            )
        if iteration > self.max_iterations:
            return GuardResult.handoff(
                LOOP_MAX_ITERATIONS_EXCEEDED,
                "AgentLoop 超过最大轮次仍未得到可靠结果。",
                {"iteration": iteration, "max_iterations": self.max_iterations},
            )
        llm_failures = self.count_llm_validation_failures(state)
        if llm_failures >= self.llm_validation_failure_threshold:
            return GuardResult.handoff(
                LLM_VALIDATION_FAILED,
                "LLM 输出多次未通过结构化校验，继续自动执行可能不可靠。",
                {"llm_validation_failures": llm_failures},
            )
        return GuardResult.pass_()

    def check_after_step(
        self,
        state: AgentState,
        action_name: str,
        observation: Dict[str, Any],
        step_elapsed_seconds: float,
    ) -> GuardResult:
        if step_elapsed_seconds > self.step_timeout_seconds:
            return GuardResult.handoff(
                STEP_TIMEOUT,
                "AgentLoop 单步执行超过超时限制。",
                {
                    "action": action_name,
                    "elapsed_seconds": round(step_elapsed_seconds, 4),
                    "step_timeout_seconds": self.step_timeout_seconds,
                },
            )

        if not bool(observation.get("success", True)):
            return self.classify_error_observation(action_name, observation)

        if observation.get("passed", True) is False:
            failure_count = self._increment_failure(state, "observation:{0}".format(action_name))
            if action_name == "retrieve_evidence":
                if failure_count <= self.max_recoverable_failures:
                    return GuardResult.fallback(
                        RETRIEVAL_NO_EVIDENCE,
                        "检索结果不足，先进入受控追问以补充信息。",
                        {"action": action_name, "failure_count": failure_count},
                    )
                return GuardResult.handoff(
                    RETRIEVAL_NO_EVIDENCE,
                    "多次检索不到足够证据，无法可靠完成任务。",
                    {"action": action_name, "failure_count": failure_count},
                )
            if action_name in {"run_learning_path_planner", "run_diagnosis_planner"}:
                if failure_count <= self.max_recoverable_failures:
                    return GuardResult.fallback(
                        PLANNER_FAILED,
                        "Planner 输出信息不足，先进入受控追问。",
                        {"action": action_name, "failure_count": failure_count},
                    )
                return GuardResult.handoff(
                    PLANNER_FAILED,
                    "Planner 多次未能生成可执行计划。",
                    {"action": action_name, "failure_count": failure_count},
                )

        llm_failures = self.count_llm_validation_failures(state)
        if llm_failures >= self.llm_validation_failure_threshold:
            return GuardResult.handoff(
                LLM_VALIDATION_FAILED,
                "LLM 输出多次未通过结构化校验，继续自动执行可能不可靠。",
                {"llm_validation_failures": llm_failures},
            )
        return GuardResult.pass_()

    def check_loop_exhausted(self, state: AgentState) -> GuardResult:
        if state.final_answer or state.metadata.get("agent_loop_finished"):
            return GuardResult.pass_()
        return GuardResult.handoff(
            LOOP_MAX_ITERATIONS_EXCEEDED,
            "AgentLoop 达到最大轮次后仍没有生成最终回答。",
            {"step_count": len(state.steps), "max_iterations": self.max_iterations},
        )

    def classify_exception(self, exc: BaseException, stage: str = "unknown") -> GuardResult:
        text = str(exc)
        lower_text = text.lower()
        detail = {"stage": stage, "exception_type": exc.__class__.__name__, "message": text}

        if isinstance(exc, (sqlite3.Error, DatabaseError)) or "database" in lower_text or "sqlite" in lower_text:
            return GuardResult.handoff(DATABASE_ERROR, "数据库异常导致任务无法可靠完成。", detail)

        if isinstance(exc, RetrievalError) or any(word in lower_text for word in ["chroma", "vector", "faiss", "embedding"]):
            return GuardResult.handoff(VECTOR_STORE_ERROR, "向量库或检索服务异常导致任务无法可靠完成。", detail)

        if isinstance(exc, AppError):
            return GuardResult.handoff(
                str(exc.code).lower(),
                exc.user_message or "系统受控失败，需要人工确认。",
                detail,
                retryable=bool(exc.retryable),
            )

        return GuardResult.handoff(UNEXPECTED_ERROR, "系统遇到未预期异常，已停止自动执行。", detail)

    def classify_error_observation(self, action_name: str, observation: Dict[str, Any]) -> GuardResult:
        error = observation.get("error") or observation.get("output", {}).get("error") or {}
        text = str(error)
        lower_text = text.lower()
        detail = {"action": action_name, "error": error}
        if any(word in lower_text for word in ["database", "sqlite", "locked"]):
            return GuardResult.handoff(DATABASE_ERROR, "数据库异常导致任务无法可靠完成。", detail)
        if any(word in lower_text for word in ["chroma", "vector", "faiss", "embedding"]):
            return GuardResult.handoff(VECTOR_STORE_ERROR, "向量库或检索服务异常导致任务无法可靠完成。", detail)
        if action_name == "retrieve_evidence":
            return GuardResult.handoff(RETRIEVAL_TOOL_ERROR, "检索工具异常导致任务无法完成。", detail)
        return GuardResult.handoff(UNEXPECTED_ERROR, "执行步骤异常，已停止自动执行。", detail)

    def count_llm_validation_failures(self, state: AgentState) -> int:
        count = 0
        if self._is_llm_failure_text(getattr(state.routing_decision, "fallback_reason", "") if state.routing_decision else ""):
            count += 1

        response_meta = state.metadata.get("response_generation") or {}
        if self._is_llm_failure_text(response_meta.get("fallback_reason", "")):
            count += 1

        for result in state.retrieval_results:
            metadata = getattr(result, "metadata", {}) or {}
            if self._is_llm_failure_text(metadata.get("rewrite_fallback_reason", "")):
                count += 1
            if metadata.get("llm_rerank_stage") == "llm_fallback" and self._is_llm_failure_text(
                metadata.get("llm_rerank_error", "llm_failed")
            ):
                count += 1

        for error in state.errors:
            if self._is_llm_failure_text(str(error)):
                count += 1
        return count

    def _increment_failure(self, state: AgentState, key: str) -> int:
        guard_meta = dict(state.metadata.get("runtime_guard") or {})
        counts = dict(guard_meta.get("failure_counts") or {})
        counts[key] = int(counts.get(key) or 0) + 1
        guard_meta["failure_counts"] = counts
        state.add_metadata("runtime_guard", guard_meta)
        return counts[key]

    def _is_llm_failure_text(self, value: Any) -> bool:
        text = str(value or "").lower()
        if not text:
            return False
        signals: Iterable[str] = (
            "llm_failed",
            "validation",
            "json",
            "invalid",
            "parse",
            "schema",
            "ranked_candidate_ids",
        )
        return any(signal in text for signal in signals)
