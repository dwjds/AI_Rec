from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from app.memory.collaborative_memory import CollaborativeMemory
from app.memory.context_builder import ContextBuilder
from app.memory.context_compressor import ContextCompressor
from app.memory.feedback_memory import FeedbackMemory
from app.memory.knowledge_state import KnowledgeStateMemory
from app.memory.resource_memory import ResourceMemory
from app.memory.session_memory import SessionMemory
from app.memory.user_memory import UserMemory
from app.rag.query_constraints import extract_query_constraints, merge_excluded_keywords
from app.services.user_service import UserService
from app.stores.user_store import UserStore


class MemoryService:
    """Build layered memory_context for agent modules.

    It reads durable user context once, builds raw memory sections, then projects
    them into routing/retrieval/ranking/planning/generation views.
    """

    def __init__(
        self,
        user_service: UserService | None = None,
        user_store: UserStore | None = None,
        session_memory: SessionMemory | None = None,
        context_builder: ContextBuilder | None = None,
        context_compressor: ContextCompressor | None = None,
        db_path: Path | None = None,
    ):
        self.user_service = user_service or UserService(db_path=db_path)
        self.user_store = user_store or self.user_service.user_store
        self.session_memory = session_memory or SessionMemory()
        self.user_memory = UserMemory()
        self.knowledge_state_memory = KnowledgeStateMemory()
        self.feedback_memory = FeedbackMemory()
        self.resource_memory = ResourceMemory(user_store=self.user_store)
        self.collaborative_memory = CollaborativeMemory()
        self.context_compressor = context_compressor or ContextCompressor()
        self.context_builder = context_builder or ContextBuilder(self.context_compressor)

    def build_memory_context(
        self,
        user_id: str,
        session_id: Optional[str] = None,
        query: str = "",
        user_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        context = user_context or self.user_service.get_user_context(user_id)
        raw_memory = self._build_raw_memory(user_id=user_id, session_id=session_id, user_context=context)
        query_constraints = extract_query_constraints(query)
        ranking_context = self._ranking_context(raw_memory)
        ranking_constraints = dict(ranking_context.get("constraints") or {})
        ranking_constraints["excluded_keywords"] = merge_excluded_keywords(
            ranking_constraints.get("excluded_keywords"),
            query_constraints.excluded_keywords,
        )
        ranking_context["constraints"] = ranking_constraints
        ranking_context["negative_terms"] = query_constraints.negative_phrases
        ranking_context["penalize_keywords"] = merge_excluded_keywords(
            ranking_context.get("penalize_keywords"),
            query_constraints.excluded_keywords,
        )
        planning_context = self._planning_context(raw_memory)
        planning_context["negative_terms"] = query_constraints.negative_phrases
        generation_context = self._generation_context(raw_memory)
        generation_context["negative_terms"] = query_constraints.negative_phrases
        generation_context["excluded_keywords"] = query_constraints.excluded_keywords
        base_contexts = {
            "raw_memory": raw_memory,
            "routing_context": self._routing_context(raw_memory, query),
            "retrieval_context": self._retrieval_context(raw_memory, query),
            "ranking_context": ranking_context,
            "planning_context": planning_context,
            "generation_context": generation_context,
        }
        prompt_contexts = self.context_builder.build_all(
            raw_memory=raw_memory,
            query=query,
            base_contexts=base_contexts,
        )
        base_contexts.update(prompt_contexts)
        base_contexts["compressed_raw_memory"] = self.context_compressor.compress_raw_memory(raw_memory)
        return base_contexts

    def remember_user_turn(
        self,
        session_id: Optional[str],
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not session_id:
            return
        self.session_memory.add_turn(session_id=session_id, role=role, content=content, metadata=metadata)

    def update_session_state(
        self,
        session_id: Optional[str],
        pending_clarification: Optional[Dict[str, Any]] = None,
        last_routing_decision: Optional[Dict[str, Any]] = None,
        last_recommendation_ids: Optional[list[str]] = None,
    ) -> None:
        if not session_id:
            return
        self.session_memory.update_session_state(
            session_id=session_id,
            pending_clarification=pending_clarification,
            last_routing_decision=last_routing_decision,
            last_recommendation_ids=last_recommendation_ids,
        )

    def _build_raw_memory(
        self,
        user_id: str,
        session_id: Optional[str],
        user_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "session_memory": self.session_memory.get_session_memory(session_id),
            "user_memory": self.user_memory.build(user_context),
            "knowledge_state": self.knowledge_state_memory.build(user_context),
            "feedback_memory": self.feedback_memory.build(user_context),
            "resource_memory": self.resource_memory.build(user_id, user_context),
            "collaborative_memory": self.collaborative_memory.build(user_id, user_context),
        }

    def _routing_context(self, raw_memory: Dict[str, Any], query: str) -> Dict[str, Any]:
        session = raw_memory["session_memory"]
        user = raw_memory["user_memory"]
        feedback = raw_memory["feedback_memory"]
        query_constraints = extract_query_constraints(query)
        return {
            "query": query,
            "last_routing_decision": session.get("last_routing_decision") or {},
            "pending_clarification": session.get("pending_clarification") or {},
            "known_subjects": user.get("preferred_subjects") or [],
            "known_learning_stage": user.get("learning_stage") or "",
            "known_goal": (user.get("goals") or [""])[0] if user.get("goals") else "",
            "recent_feedback_intent": feedback.get("recent_feedback_summary") or "",
            "negative_terms": query_constraints.negative_phrases,
            "excluded_keywords": query_constraints.excluded_keywords,
            "profile": {
                "learning_stage": user.get("learning_stage") or "",
                "preferred_subjects": user.get("preferred_subjects") or [],
                "goal": (user.get("goals") or [""])[0] if user.get("goals") else "",
            },
            "recent_recommendations": raw_memory["resource_memory"].get("recent_recommendations") or [],
        }

    def _retrieval_context(self, raw_memory: Dict[str, Any], query: str) -> Dict[str, Any]:
        user = raw_memory["user_memory"]
        knowledge = raw_memory["knowledge_state"]
        query_constraints = extract_query_constraints(query)
        return {
            "query": query,
            "positive_query": query_constraints.positive_query,
            "negative_terms": query_constraints.negative_phrases,
            "excluded_keywords": query_constraints.excluded_keywords,
            "preferred_subjects": user.get("preferred_subjects") or [],
            "learning_goal": (user.get("goals") or [""])[0] if user.get("goals") else "",
            "weak_knowledge_points": knowledge.get("weak_points") or [],
            "recent_query_terms": self._query_terms(query),
            "preferred_resource_types": user.get("preferred_resource_types") or [],
        }

    def _ranking_context(self, raw_memory: Dict[str, Any]) -> Dict[str, Any]:
        user = raw_memory["user_memory"]
        knowledge = raw_memory["knowledge_state"]
        feedback = raw_memory["feedback_memory"]
        resources = raw_memory["resource_memory"]
        adjustments = feedback.get("recommendation_adjustments") or {}
        constraints = dict(user.get("constraints") or {})
        return {
            "learning_stage": user.get("learning_stage") or "",
            "preferred_resource_types": user.get("preferred_resource_types") or [],
            "constraints": constraints,
            "liked_resource_ids": feedback.get("liked_resource_ids") or [],
            "disliked_resource_ids": feedback.get("disliked_resource_ids") or [],
            "avoid_repeating_resource_ids": resources.get("avoid_repeating_resource_ids") or [],
            "boost_keywords": adjustments.get("boost_keywords") or [],
            "penalize_keywords": adjustments.get("penalize_keywords") or [],
            "weak_knowledge_points": knowledge.get("weak_points") or [],
        }

    def _planning_context(self, raw_memory: Dict[str, Any]) -> Dict[str, Any]:
        user = raw_memory["user_memory"]
        knowledge = raw_memory["knowledge_state"]
        resources = raw_memory["resource_memory"]
        return {
            "learning_stage": user.get("learning_stage") or "",
            "goal": (user.get("goals") or [""])[0] if user.get("goals") else "",
            "preferred_subjects": user.get("preferred_subjects") or [],
            "weak_points": knowledge.get("weak_points") or [],
            "strong_points": knowledge.get("strong_points") or [],
            "recent_learning_history": resources.get("recent_resource_events") or [],
            "constraints": user.get("constraints") or {},
        }

    def _generation_context(self, raw_memory: Dict[str, Any]) -> Dict[str, Any]:
        user = raw_memory["user_memory"]
        session = raw_memory["session_memory"]
        feedback = raw_memory["feedback_memory"]
        resources = raw_memory["resource_memory"]
        return {
            "user_summary": user.get("summary") or "",
            "memory_summary": self._memory_summary(raw_memory),
            "tone_preferences": {},
            "do_not_repeat_resource_ids": resources.get("avoid_repeating_resource_ids") or [],
            "clarification_history": session.get("pending_clarification") or {},
            "recent_feedback_summary": feedback.get("recent_feedback_summary") or "",
            "recent_turns": session.get("turns") or [],
        }

    def _memory_summary(self, raw_memory: Dict[str, Any]) -> str:
        parts = []
        user_summary = raw_memory["user_memory"].get("summary")
        diagnosis_hint = raw_memory["knowledge_state"].get("diagnosis_hint")
        feedback_summary = raw_memory["feedback_memory"].get("recent_feedback_summary")
        session_summary = self.context_compressor.summarize_turns(raw_memory["session_memory"].get("turns") or [])
        for item in [user_summary, diagnosis_hint, feedback_summary, session_summary]:
            if item:
                parts.append(str(item))
        return self.context_compressor.truncate_text("；".join(parts), 1000)

    def _query_terms(self, query: str) -> list[str]:
        return [item for item in str(query or "").replace("，", " ").replace(",", " ").split() if item][:12]
