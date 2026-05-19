from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from app.agent.router import RoutingDecision
from app.rag.evidence_builder import EvidencePackage
from app.rag.retriever import RetrievalResult


@dataclass
class AgentStep:
    index: int
    name: str
    input: Dict[str, Any] = field(default_factory=dict)
    observation: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "input": self.input,
            "observation": self.observation,
        }


@dataclass
class AgentState:
    """Mutable state for a single agent request.

    Router, Retriever, EvidenceBuilder, Planner, AgentLoop, and ResponseGenerator
    should read/write this object instead of passing many loose parameters.
    """

    user_id: str
    query: str
    session_id: Optional[str] = None
    routing_decision: Optional[RoutingDecision] = None
    user_context: Dict[str, Any] = field(default_factory=dict)
    memory_context: Dict[str, Any] = field(default_factory=dict)
    retrieval_results: List[RetrievalResult] = field(default_factory=list)
    evidence_package: Optional[EvidencePackage] = None
    plan: Dict[str, Any] = field(default_factory=dict)
    final_answer: str = ""
    recommendation_event: Optional[Dict[str, Any]] = None
    feedback_event: Optional[Dict[str, Any]] = None
    handoff_case: Optional[Dict[str, Any]] = None
    trace_run_id: Optional[str] = None
    steps: List[AgentStep] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def task_type(self) -> str:
        return self.routing_decision.task_type if self.routing_decision else "unknown"

    @property
    def pipeline(self) -> str:
        return self.routing_decision.pipeline if self.routing_decision else "unknown"

    def set_user_context(self, user_context: Dict[str, Any]) -> None:
        self.user_context = user_context or {}

    def set_memory_context(self, memory_context: Dict[str, Any]) -> None:
        self.memory_context = memory_context or {}

    def set_routing_decision(self, decision: RoutingDecision) -> None:
        self.routing_decision = decision

    def set_retrieval_results(self, results: List[RetrievalResult]) -> None:
        self.retrieval_results = list(results or [])

    def set_evidence_package(self, evidence_package: EvidencePackage) -> None:
        self.evidence_package = evidence_package

    def set_plan(self, plan: Dict[str, Any]) -> None:
        self.plan = plan or {}

    def set_final_answer(self, answer: str) -> None:
        self.final_answer = answer or ""

    def set_recommendation_event(self, event: Optional[Dict[str, Any]]) -> None:
        self.recommendation_event = event

    def set_feedback_event(self, event: Optional[Dict[str, Any]]) -> None:
        self.feedback_event = event

    def set_handoff_case(self, case: Optional[Dict[str, Any]]) -> None:
        self.handoff_case = case

    def set_trace_run_id(self, run_id: Optional[str]) -> None:
        self.trace_run_id = run_id

    def add_step(
        self,
        name: str,
        action_input: Optional[Dict[str, Any]] = None,
        observation: Optional[Dict[str, Any]] = None,
        index: Optional[int] = None,
    ) -> AgentStep:
        step = AgentStep(
            index=index if index is not None else len(self.steps),
            name=name,
            input=action_input or {},
            observation=observation or {},
        )
        self.steps.append(step)
        return step

    def add_metadata(self, key: str, value: Any) -> None:
        self.metadata[key] = value

    def add_error(self, stage: str, error: Any) -> None:
        self.errors.append({"stage": stage, "error": error if isinstance(error, dict) else str(error)})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "query": self.query,
            "session_id": self.session_id,
            "routing_decision": self.routing_decision.to_dict() if self.routing_decision else None,
            "user_context": self.user_context,
            "memory_context": self.memory_context,
            "retrieval_results": [item.to_dict() for item in self.retrieval_results],
            "evidence_package": self.evidence_package.to_dict() if self.evidence_package else None,
            "plan": self.plan,
            "final_answer": self.final_answer,
            "recommendation_event": self.recommendation_event,
            "feedback_event": self.feedback_event,
            "handoff_case": self.handoff_case,
            "trace_run_id": self.trace_run_id,
            "steps": [step.to_dict() for step in self.steps],
            "metadata": self.metadata,
            "errors": self.errors,
        }
