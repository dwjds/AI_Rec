from __future__ import annotations

from typing import Any, Dict, Optional

from app.agent.guard import GuardResult
from app.agent.state import AgentState
from app.stores.handoff_store import HandoffStore


class HandoffService:
    """Create human fallback cases from guarded Agent failures."""

    def __init__(self, handoff_store: Optional[HandoffStore] = None):
        self.handoff_store = handoff_store or HandoffStore()

    def create_case(self, state: AgentState, guard_result: GuardResult) -> Dict[str, Any]:
        case = self.handoff_store.create_case(
            user_id=state.user_id,
            session_id=state.session_id,
            trace_run_id=state.trace_run_id,
            task_type=state.task_type,
            query=state.query,
            reason_code=guard_result.reason_code,
            reason_text=guard_result.reason_text,
            priority=self._priority(guard_result),
            context=self._context_payload(state, guard_result),
        )
        return case

    def handoff_answer(self, case: Dict[str, Any]) -> str:
        reason = str(case.get("reason_text") or "系统暂时无法可靠完成这个任务。")
        return (
            "这个问题我先不继续自动给结论了，因为{0}\n"
            "我已经把这次请求转入人工兜底，处理编号是 {1}。你可以稍后带着这个编号查看处理结果。"
        ).format(reason, case.get("id"))

    def _priority(self, guard_result: GuardResult) -> str:
        if guard_result.reason_code in {"database_error", "vector_store_error", "unexpected_error"}:
            return "high"
        return "normal"

    def _context_payload(self, state: AgentState, guard_result: GuardResult) -> Dict[str, Any]:
        return {
            "guard": guard_result.to_dict(),
            "routing_decision": state.routing_decision.to_dict() if state.routing_decision else None,
            "user_profile": state.user_context.get("profile") or {},
            "knowledge_state": state.user_context.get("knowledge_state") or [],
            "memory_context": {
                "routing_context": state.memory_context.get("routing_context") or {},
                "retrieval_context": state.memory_context.get("retrieval_context") or {},
                "ranking_context": state.memory_context.get("ranking_context") or {},
                "planning_context": state.memory_context.get("planning_context") or {},
                "generation_context": state.memory_context.get("generation_context") or {},
            },
            "retrieval_count": len(state.retrieval_results),
            "evidence_count": len(state.evidence_package.evidence_items) if state.evidence_package else 0,
            "recent_steps": [step.to_dict() for step in state.steps[-8:]],
            "metadata": state.metadata,
            "errors": state.errors,
        }
