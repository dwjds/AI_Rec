from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from app.agent.agent_loop import AgentLoop
from app.agent.guard import GuardResult, RuntimeGuard
from app.agent.router import Router, RoutingDecision
from app.agent.state import AgentState
from app.agent.trace import AgentTraceRecorder
from app.core.logging import clear_log_context, get_logger, set_log_context
from app.generation.response_generator import ResponseGenerator
from app.memory.memory_service import MemoryService
from app.planning.diagnosis_planner import DiagnosisPlanner
from app.planning.learning_path_planner import LearningPathPlanner
from app.rag.evidence_builder import EvidenceBuilder, EvidencePackage
from app.rag.retriever import RagRetriever, RetrievalResult
from app.recommender.recommendation_pipeline import RecommendationPipeline
from app.services.feedback_service import FeedbackService
from app.services.handoff_service import HandoffService
from app.services.user_service import UserService
from app.stores.handoff_store import HandoffStore
from app.stores.tool_call_store import ToolCallStore
from app.stores.trace_store import TraceStore
from app.tools.feedback_tool import GetFeedbackSummaryTool, RecordFeedbackTool
from app.tools.registry import ToolRegistry
from app.tools.resource_tool import GetCourseDetailTool, SearchCoursesTool, SearchResourcesTool
from app.tools.runtime import FunctionCallRuntime
from app.tools.user_tool import GetUserContextTool, UpdateKnowledgeStateTool, UpdateUserProfileTool


logger = get_logger(__name__)


@dataclass
class RagAgentResult:
    query: str
    task_type: str
    user_context: Dict[str, Any]
    retrieval_results: List[RetrievalResult]
    evidence_package: EvidencePackage

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "task_type": self.task_type,
            "user_context": self.user_context,
            "retrieval_results": [item.to_dict() for item in self.retrieval_results],
            "evidence_package": self.evidence_package.to_dict(),
        }


@dataclass
class OrchestratorResult:
    user_id: str
    query: str
    routing_decision: RoutingDecision
    answer: str
    pipeline: str
    user_context: Dict[str, Any] = field(default_factory=dict)
    evidence_package: Optional[EvidencePackage] = None
    retrieval_results: List[RetrievalResult] = field(default_factory=list)
    recommendation_event: Optional[Dict[str, Any]] = None
    feedback_event: Optional[Dict[str, Any]] = None
    handoff_case: Optional[Dict[str, Any]] = None
    trace_run_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    state: Optional[AgentState] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "query": self.query,
            "routing_decision": self.routing_decision.to_dict(),
            "answer": self.answer,
            "pipeline": self.pipeline,
            "user_context": self.user_context,
            "evidence_package": self.evidence_package.to_dict() if self.evidence_package else None,
            "retrieval_results": [item.to_dict() for item in self.retrieval_results],
            "recommendation_event": self.recommendation_event,
            "feedback_event": self.feedback_event,
            "handoff_case": self.handoff_case,
            "trace_run_id": self.trace_run_id,
            "metadata": self.metadata,
            "state": self.state.to_dict() if self.state else None,
        }


class AgentOrchestrator:
    """Top-level dispatcher over a shared AgentState."""

    def __init__(
        self,
        user_service: Optional[UserService] = None,
        retriever: Optional[RagRetriever] = None,
        evidence_builder: Optional[EvidenceBuilder] = None,
        router: Optional[Router] = None,
        feedback_service: Optional[FeedbackService] = None,
        memory_service: Optional[MemoryService] = None,
        recommendation_pipeline: Optional[RecommendationPipeline] = None,
        learning_path_planner: Optional[LearningPathPlanner] = None,
        diagnosis_planner: Optional[DiagnosisPlanner] = None,
        response_generator: Optional[ResponseGenerator] = None,
        agent_loop: Optional[AgentLoop] = None,
        trace_store: Optional[TraceStore] = None,
        runtime_guard: Optional[RuntimeGuard] = None,
        handoff_service: Optional[HandoffService] = None,
        function_call_runtime: Optional[FunctionCallRuntime] = None,
    ):
        self.user_service = user_service or UserService()
        self._retriever = retriever
        self._evidence_builder = evidence_builder
        self.router = router or Router()
        self.feedback_service = feedback_service or FeedbackService()
        self.memory_service = memory_service or MemoryService(
            user_service=self.user_service,
            user_store=self.user_service.user_store,
        )
        self.recommendation_pipeline = recommendation_pipeline or RecommendationPipeline()
        self.learning_path_planner = learning_path_planner or LearningPathPlanner()
        self.diagnosis_planner = diagnosis_planner or DiagnosisPlanner()
        self.response_generator = response_generator or ResponseGenerator()
        self._agent_loop = agent_loop
        self.trace_store = trace_store or TraceStore(user_store=self.user_service.user_store)
        self.trace_recorder = AgentTraceRecorder(self.trace_store)
        self.runtime_guard = runtime_guard or RuntimeGuard()
        self.handoff_service = handoff_service or HandoffService(
            HandoffStore(db_path=self.user_service.user_store.db_path)
        )
        self.function_call_runtime = function_call_runtime or self._create_function_call_runtime()

    @property
    def retriever(self) -> RagRetriever:
        if self._retriever is None:
            self._retriever = RagRetriever()
        return self._retriever

    @property
    def evidence_builder(self) -> EvidenceBuilder:
        if self._evidence_builder is None:
            self._evidence_builder = EvidenceBuilder()
        return self._evidence_builder

    @property
    def agent_loop(self) -> AgentLoop:
        if self._agent_loop is None:
            self._agent_loop = AgentLoop(
                retriever=self.retriever,
                evidence_builder=self.evidence_builder,
                learning_path_planner=self.learning_path_planner,
                diagnosis_planner=self.diagnosis_planner,
                response_generator=self.response_generator,
                trace_store=self.trace_store,
                runtime_guard=self.runtime_guard,
                handoff_service=self.handoff_service,
                function_call_runtime=self.function_call_runtime,
            )
        return self._agent_loop

    def run(
        self,
        user_id: str,
        query: str,
        session_id: Optional[str] = None,
        use_llm_route: bool = True,
        use_llm_rerank: bool = True,
        use_llm_generation: bool = True,
        top_k: int = 8,
    ) -> OrchestratorResult:
        set_log_context(user_id=user_id, session_id=session_id)
        try:
            return self._run_impl(
                user_id=user_id,
                query=query,
                session_id=session_id,
                use_llm_route=use_llm_route,
                use_llm_rerank=use_llm_rerank,
                use_llm_generation=use_llm_generation,
                top_k=top_k,
            )
        except Exception as exc:
            logger.exception("agent request failed")
            raise
        finally:
            clear_log_context()

    def _run_impl(
        self,
        user_id: str,
        query: str,
        session_id: Optional[str] = None,
        use_llm_route: bool = True,
        use_llm_rerank: bool = True,
        use_llm_generation: bool = True,
        top_k: int = 8,
    ) -> OrchestratorResult:
        state = AgentState(user_id=user_id, query=query, session_id=session_id)
        state.set_user_context(self.user_service.get_user_context(user_id))
        state.set_memory_context(
            self.memory_service.build_memory_context(
                user_id=user_id,
                session_id=session_id,
                query=query,
                user_context=state.user_context,
            )
        )
        self.memory_service.remember_user_turn(session_id=session_id, role="user", content=query)
        decision = self.router.route(
            query=query,
            user_context=self._routing_context(state),
            use_llm=use_llm_route,
        )
        state.set_routing_decision(decision)
        state.set_trace_run_id(self._start_trace(state))
        set_log_context(trace_run_id=state.trace_run_id, pipeline=decision.pipeline)
        logger.info("agent request routed")

        try:
            if decision.needs_clarification:
                self._run_clarification(state)
            elif decision.pipeline == "rag_qa":
                self._run_rag(state, top_k=top_k, use_llm_rerank=use_llm_rerank)
            elif decision.pipeline == "recommendation_pipeline":
                self._run_recommendation(state, top_k=top_k, use_llm_rerank=use_llm_rerank)
            elif decision.pipeline in {"agent_loop_learning_path", "agent_loop_diagnosis"}:
                self._run_agent_loop(
                    state,
                    top_k=top_k,
                    use_llm_rerank=use_llm_rerank,
                    use_llm_generation=use_llm_generation,
                )
            elif decision.pipeline == "feedback_pipeline":
                self._run_feedback_adjustment(state)
            else:
                self._run_direct_chat(state)

            if "response_generation" not in state.metadata and not state.handoff_case:
                self.response_generator.generate(state, use_llm=use_llm_generation)
        except Exception as exc:
            logger.exception("agent pipeline failed")
            self._run_exception_handoff(state, exc)
        self._finish_trace(state)
        self._remember_completed_turn(state)
        logger.info("agent request completed")
        return self._result_from_state(state)

    def build_rag_evidence(
        self,
        user_id: str,
        query: str,
        task_type: str,
        top_k: int = 8,
        chunk_types: Optional[Sequence[str]] = None,
        use_llm_rerank: bool = True,
    ) -> RagAgentResult:
        state = AgentState(user_id=user_id, query=query)
        state.set_user_context(self.user_service.get_user_context(user_id))
        state.set_memory_context(
            self.memory_service.build_memory_context(
                user_id=user_id,
                query=query,
                user_context=state.user_context,
            )
        )
        return self._build_rag_evidence_for_state(
            state=state,
            task_type=task_type,
            top_k=top_k,
            chunk_types=chunk_types,
            use_llm_rerank=use_llm_rerank,
        )

    def record_recommendation_from_evidence(self, user_id: str, query: str, result: RagAgentResult) -> Dict[str, Any]:
        resource_ids = self._course_resource_ids(result.evidence_package)
        return self.user_service.record_recommendation(
            user_id=user_id,
            query=query,
            recommended_resource_ids=resource_ids,
            intent={"task_type": result.task_type},
        )

    def _build_rag_evidence_for_state(
        self,
        state: AgentState,
        task_type: str,
        top_k: int,
        chunk_types: Optional[Sequence[str]] = None,
        use_llm_rerank: bool = True,
    ) -> RagAgentResult:
        user_profile = self._user_profile_for_rag(state)
        knowledge_state = self._knowledge_state_for_rag(state)
        retrieval_results = self.retriever.retrieve(
            query=state.query,
            top_k=top_k,
            chunk_types=chunk_types,
            task_type=task_type,
            user_profile=user_profile,
            knowledge_state=knowledge_state,
            use_llm_rerank=use_llm_rerank,
        )
        evidence_package = self.evidence_builder.build(
            query=state.query,
            task_type=task_type,
            retrieval_results=retrieval_results,
            user_profile=user_profile,
            knowledge_state=knowledge_state,
        )
        state.set_retrieval_results(retrieval_results)
        state.set_evidence_package(evidence_package)
        return RagAgentResult(
            query=state.query,
            task_type=task_type,
            user_context=state.user_context,
            retrieval_results=retrieval_results,
            evidence_package=evidence_package,
        )

    def _run_clarification(self, state: AgentState) -> None:
        decision = self._decision(state)
        questions = decision.clarification_questions or ["我需要再确认一下你的学习目标。"]
        answer = "为了给你更准确的帮助，我需要先确认：\n" + "\n".join(
            "{0}. {1}".format(index, question) for index, question in enumerate(questions, start=1)
        )
        state.set_final_answer(answer)
        state.add_step("clarification", {"questions": questions}, {"answer": answer})
        self._add_trace_step(state, 1, "clarification", {"questions": questions}, {"answer": answer})

    def _run_rag(self, state: AgentState, top_k: int, use_llm_rerank: bool) -> None:
        rag_result = self._build_rag_evidence_for_state(
            state=state,
            task_type=self._decision(state).task_type,
            top_k=top_k,
            use_llm_rerank=use_llm_rerank,
        )
        answer = self._evidence_answer("我会基于检索到的课程知识证据来回答。", rag_result.evidence_package)
        state.set_final_answer(answer)
        state.add_step("rag_qa", {"query": state.query}, {"evidence_count": len(rag_result.evidence_package.evidence_items)})
        self._add_trace_step(
            state,
            1,
            "rag_qa",
            {"query": state.query},
            {"evidence_count": len(rag_result.evidence_package.evidence_items)},
        )

    def _run_recommendation(self, state: AgentState, top_k: int, use_llm_rerank: bool) -> None:
        rag_result = self._build_rag_evidence_for_state(
            state=state,
            task_type="recommend",
            top_k=top_k,
            chunk_types=["course", "chapter", "exercise"],
            use_llm_rerank=use_llm_rerank,
        )
        recommendation_package = self.recommendation_pipeline.run(state, top_k=top_k)
        recommended_resource_ids = [
            item.candidate.resource_id
            for item in recommendation_package.recommendations
            if item.candidate.resource_id
        ]
        recommendation_event = self.user_service.record_recommendation(
            user_id=state.user_id,
            query=state.query,
            recommended_resource_ids=recommended_resource_ids,
            intent={"task_type": self._decision(state).task_type},
        )
        state.set_recommendation_event(recommendation_event)
        state.add_metadata("recommendation_package", recommendation_package.to_dict())
        answer = self._recommendation_answer(recommendation_package, rag_result.evidence_package)
        state.set_final_answer(answer)
        state.add_step(
            "recommendation_pipeline",
            {"query": state.query},
            {
                "evidence_count": len(rag_result.evidence_package.evidence_items),
                "recommendation_event_id": recommendation_event.get("id"),
                "recommendation_count": len(recommendation_package.recommendations),
            },
        )
        self._add_trace_step(
            state,
            1,
            "recommendation_pipeline",
            {"query": state.query},
            {
                "evidence_count": len(rag_result.evidence_package.evidence_items),
                "recommendation_event_id": recommendation_event.get("id"),
                "recommendation_count": len(recommendation_package.recommendations),
            },
        )

    def _run_agent_loop(
        self,
        state: AgentState,
        top_k: int,
        use_llm_rerank: bool,
        use_llm_generation: bool,
    ) -> None:
        self.agent_loop.run(
            state=state,
            top_k=top_k,
            use_llm_rerank=use_llm_rerank,
            use_llm_generation=use_llm_generation,
        )

    def _run_feedback_adjustment(self, state: AgentState) -> None:
        target_resource_id = self._infer_feedback_resource_id(state.query, state.user_context)
        if not target_resource_id:
            original = self._decision(state)
            clarification = RoutingDecision(
                query=state.query,
                task_type=original.task_type,
                pipeline=original.pipeline,
                needs_rag=original.needs_rag,
                needs_user_profile=original.needs_user_profile,
                needs_agent_loop=original.needs_agent_loop,
                information_sufficient=False,
                needs_clarification=True,
                clarification_questions=["你想反馈的是刚才推荐的哪一个资源？可以说资源名称、序号，或重新说明偏好。"],
                entities=original.entities,
                confidence=original.confidence,
                used_llm=original.used_llm,
                fallback_reason=original.fallback_reason,
            )
            state.set_routing_decision(clarification)
            self._run_clarification(state)
            return

        feedback_event = self.feedback_service.record_feedback(
            user_id=state.user_id,
            resource_id=target_resource_id,
            feedback_type=self._infer_feedback_type(state.query),
            comment=state.query,
        )
        state.set_feedback_event(feedback_event)
        answer = "收到，我已经记录了你对该资源的反馈。后续推荐会避开类似不合适的内容，并优先贴近你的新偏好。"
        state.set_final_answer(answer)
        state.add_step(
            "feedback_adjustment",
            {"query": state.query, "resource_id": target_resource_id},
            {"feedback_id": feedback_event.get("id")},
        )
        self._add_trace_step(
            state,
            1,
            "feedback_adjustment",
            {"query": state.query, "resource_id": target_resource_id},
            {"feedback_id": feedback_event.get("id")},
        )

    def _run_direct_chat(self, state: AgentState) -> None:
        answer = "我可以帮你做课程推荐、知识点解释、学习路线规划，或者诊断学习困难。你可以直接说你的目标。"
        state.set_final_answer(answer)
        state.add_step("direct_chat", {"query": state.query}, {"answer": answer})
        self._add_trace_step(state, 1, "direct_chat", {"query": state.query}, {"answer": answer})

    def _result_from_state(self, state: AgentState) -> OrchestratorResult:
        decision = self._decision(state)
        return OrchestratorResult(
            user_id=state.user_id,
            query=state.query,
            routing_decision=decision,
            answer=state.final_answer,
            pipeline="clarification" if decision.needs_clarification else decision.pipeline,
            user_context=state.user_context,
            evidence_package=state.evidence_package,
            retrieval_results=state.retrieval_results,
            recommendation_event=state.recommendation_event,
            feedback_event=state.feedback_event,
            handoff_case=state.handoff_case,
            trace_run_id=state.trace_run_id,
            metadata=state.metadata,
            state=state,
        )

    def _routing_context(self, state: AgentState) -> Dict[str, Any]:
        context = dict(state.memory_context.get("router_prompt_context") or state.memory_context.get("routing_context") or {})
        context.setdefault("profile", state.user_context.get("profile") or {})
        context.setdefault("recent_recommendations", state.user_context.get("recent_recommendations") or [])
        return context

    def _user_profile_for_rag(self, state: AgentState) -> Dict[str, Any]:
        profile = dict(state.user_context.get("profile") or {})
        ranking_context = state.memory_context.get("ranking_context") or {}
        retrieval_context = state.memory_context.get("retriever_prompt_context") or state.memory_context.get("retrieval_context") or {}
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

    def _remember_completed_turn(self, state: AgentState) -> None:
        if not state.session_id:
            return
        decision = self._decision(state)
        recommendation_ids = []
        if state.recommendation_event:
            recommendation_ids = state.recommendation_event.get("recommended_resource_ids") or []
        pending = {}
        if decision.needs_clarification:
            pending = {
                "task_type": decision.task_type,
                "questions": decision.clarification_questions,
            }
        self.memory_service.remember_user_turn(
            session_id=state.session_id,
            role="assistant",
            content=state.final_answer,
            metadata={"pipeline": state.pipeline},
        )
        self.memory_service.update_session_state(
            session_id=state.session_id,
            pending_clarification=pending,
            last_routing_decision=decision.to_dict(),
            last_recommendation_ids=recommendation_ids,
        )

    def _evidence_answer(self, prefix: str, evidence_package: EvidencePackage) -> str:
        items = evidence_package.evidence_items[:5]
        if not items:
            return prefix + "\n\n目前没有检索到足够可靠的 MOOC 资源证据，我建议先补充更具体的学习方向。"
        lines = [prefix, ""]
        for index, item in enumerate(items, start=1):
            lines.append("{0}. {1}".format(index, item.title))
            if item.resource.get("description"):
                lines.append("   依据：" + str(item.resource.get("description"))[:120])
            elif item.content:
                lines.append("   依据：" + item.content[:120])
        return "\n".join(lines)

    def _recommendation_answer(self, package: Any, evidence_package: EvidencePackage) -> str:
        if not package.recommendations:
            return "我检索了当前 MOOPer 资源库，但没有找到足够匹配的学习资源。为了避免把不相关课程硬推荐给你，我先不列课程。"
        lines = ["我先按你的目标、用户画像和资源库真实资料筛出这些资源：", ""]
        for index, item in enumerate(package.recommendations[:5], start=1):
            candidate = item.candidate
            lines.append("{0}. {1}".format(index, candidate.title))
            if item.reasons:
                lines.append("   推荐依据：" + "；".join(item.reasons[:3]))
        if package.next_steps:
            lines.append("")
            lines.append("建议下一步：" + package.next_steps[0])
        return "\n".join(lines)

    def _planner_answer(self, task_type: str, planner_output: Dict[str, Any], evidence_package: EvidencePackage) -> str:
        if not planner_output:
            return self._evidence_answer("我会按“目标判断 -> 证据检索 -> 学习行动”的方式处理。", evidence_package)
        if task_type == "learning_path":
            stages = planner_output.get("estimated_path") or []
            lines = ["我根据你的目标、当前记忆和检索证据，先给出这条学习路线：", ""]
            for index, stage in enumerate(stages[:4], start=1):
                lines.append("{0}. {1}：{2}".format(index, stage.get("stage"), stage.get("objective")))
                if stage.get("checkpoint"):
                    lines.append("   检查点：" + str(stage.get("checkpoint")))
            return "\n".join(lines)
        if task_type == "diagnosis":
            lines = ["我先按学习诊断来判断：", ""]
            lines.append("可能类型：" + str(planner_output.get("diagnosis_type") or "unknown"))
            causes = planner_output.get("likely_causes") or []
            for cause in causes[:3]:
                lines.append("- " + str(cause))
            lines.append("下一步：" + str(planner_output.get("next_action") or "先补充具体卡点。"))
            return "\n".join(lines)
        return self._evidence_answer("我会按检索证据继续处理。", evidence_package)

    def _course_resource_ids(self, evidence_package: EvidencePackage) -> List[str]:
        return [
            item.source_resource_id
            for item in evidence_package.evidence_items
            if item.chunk_type == "course" and item.source_resource_id
        ]

    def _infer_feedback_resource_id(self, query: str, user_context: Dict[str, Any]) -> str:
        text = str(query or "")
        recent = user_context.get("recent_recommendations") or []
        resource_ids: List[str] = []
        for event in recent:
            resource_ids.extend(event.get("recommended_resource_ids") or [])
        if not resource_ids:
            return ""
        if any(word in text for word in ["第一个", "第一门", "1", "一"]):
            return str(resource_ids[0])
        if len(resource_ids) > 1 and any(word in text for word in ["第二个", "第二门", "2", "二"]):
            return str(resource_ids[1])
        if any(word in text for word in ["这个", "上一个", "刚才", "推荐的"]):
            return str(resource_ids[0])
        return ""

    def _infer_feedback_type(self, query: str) -> str:
        text = str(query or "")
        if any(word in text for word in ["太难", "难度太高", "看不懂"]):
            return "too_hard"
        if any(word in text for word in ["太简单", "太基础"]):
            return "too_easy"
        if any(word in text for word in ["不喜欢", "不要", "不适合", "没用"]):
            return "dislike"
        return "feedback"

    def _decision(self, state: AgentState) -> RoutingDecision:
        if state.routing_decision is None:
            raise RuntimeError("AgentState.routing_decision is not set.")
        return state.routing_decision

    def _start_trace(self, state: AgentState) -> Optional[str]:
        decision = self._decision(state)
        run_id = self.trace_recorder.start(
            state=state,
            request_type=decision.task_type,
            metadata={"pipeline": decision.pipeline},
        )
        self._add_trace_step(
            state,
            0,
            "router.route",
            {"query": state.query},
            {"decision": decision.to_dict()},
            run_id=run_id,
        )
        return run_id

    def _add_trace_step(
        self,
        state: AgentState,
        step_index: int,
        action_name: str,
        action_input: Dict[str, Any],
        observation: Dict[str, Any],
        run_id: Optional[str] = None,
    ) -> None:
        self.trace_recorder.step(
            state=state,
            step_index=step_index,
            action_name=action_name,
            action_input=action_input,
            observation=observation,
            run_id=run_id,
        )

    def _finish_trace(self, state: AgentState) -> None:
        self.trace_recorder.finish(state, state.final_answer)

    def _create_function_call_runtime(self) -> FunctionCallRuntime:
        registry = ToolRegistry()
        registry.register(SearchResourcesTool())
        registry.register(SearchCoursesTool())
        registry.register(GetCourseDetailTool())
        registry.register(GetUserContextTool(user_service=self.user_service))
        registry.register(UpdateUserProfileTool(user_service=self.user_service))
        registry.register(UpdateKnowledgeStateTool(user_service=self.user_service))
        registry.register(RecordFeedbackTool(feedback_service=self.feedback_service))
        registry.register(GetFeedbackSummaryTool(feedback_service=self.feedback_service))
        return FunctionCallRuntime(
            registry=registry,
            tool_call_store=ToolCallStore(db_path=self.user_service.user_store.db_path),
            trace_store=self.trace_store,
        )

    def _run_exception_handoff(self, state: AgentState, exc: BaseException) -> None:
        guard_result = self.runtime_guard.classify_exception(exc, stage=state.pipeline)
        state.add_error(state.pipeline, guard_result.to_dict())
        try:
            case = self.handoff_service.create_case(state, guard_result)
            state.set_handoff_case(case)
            state.set_final_answer(self.handoff_service.handoff_answer(case))
            state.add_metadata("handoff", {"case_id": case.get("id"), "reason_code": guard_result.reason_code})
            observation = {"guard": guard_result.to_dict(), "handoff_case_id": case.get("id")}
        except Exception as handoff_exc:
            state.add_error("handoff", str(handoff_exc))
            state.set_final_answer("系统遇到无法自动恢复的问题，请稍后再试，或联系人工支持。")
            observation = {"guard": guard_result.to_dict(), "handoff_error": str(handoff_exc)}
        state.add_metadata("runtime_guard_last_result", guard_result.to_dict())
        state.add_step("runtime_guard.handoff", {"pipeline": state.pipeline}, observation)
        self._add_trace_step(
            state,
            len(state.steps),
            "runtime_guard.handoff",
            {"pipeline": state.pipeline},
            observation,
        )
