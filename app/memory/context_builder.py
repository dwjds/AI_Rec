from __future__ import annotations

from typing import Any, Dict

from app.memory.context_compressor import ContextCompressor


class ContextBuilder:
    """Build stage-specific contexts for Router, Retriever, Planner, Generator."""

    def __init__(self, compressor: ContextCompressor | None = None):
        self.compressor = compressor or ContextCompressor()

    def build_all(
        self,
        raw_memory: Dict[str, Any],
        query: str,
        base_contexts: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        compressed_raw = self.compressor.compress_raw_memory(raw_memory)
        return {
            "router_prompt_context": self.build_router_context(compressed_raw, query, base_contexts.get("routing_context") or {}),
            "retriever_prompt_context": self.build_retriever_context(compressed_raw, query, base_contexts.get("retrieval_context") or {}),
            "planner_prompt_context": self.build_planner_context(compressed_raw, query, base_contexts.get("planning_context") or {}),
            "generator_prompt_context": self.build_generator_context(compressed_raw, query, base_contexts.get("generation_context") or {}),
        }

    def build_router_context(
        self,
        raw_memory: Dict[str, Any],
        query: str,
        routing_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        session = raw_memory.get("session_memory") or {}
        user = raw_memory.get("user_memory") or {}
        context = {
            "query": query,
            "profile": routing_context.get("profile") or {},
            "last_routing_decision": routing_context.get("last_routing_decision") or {},
            "pending_clarification": routing_context.get("pending_clarification") or {},
            "recent_turns": session.get("turns") or [],
            "turn_summary": session.get("turn_summary") or "",
            "known_subjects": routing_context.get("known_subjects") or user.get("preferred_subjects") or [],
            "known_learning_stage": routing_context.get("known_learning_stage") or user.get("learning_stage") or "",
            "known_goal": routing_context.get("known_goal") or "",
        }
        return self.compressor.compress_context(context)

    def build_retriever_context(
        self,
        raw_memory: Dict[str, Any],
        query: str,
        retrieval_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        knowledge = raw_memory.get("knowledge_state") or {}
        context = {
            "query": query,
            "preferred_subjects": retrieval_context.get("preferred_subjects") or [],
            "learning_goal": retrieval_context.get("learning_goal") or "",
            "weak_knowledge_points": retrieval_context.get("weak_knowledge_points") or knowledge.get("weak_points") or [],
            "recent_query_terms": retrieval_context.get("recent_query_terms") or [],
            "preferred_resource_types": retrieval_context.get("preferred_resource_types") or [],
        }
        return self.compressor.compress_context(context)

    def build_planner_context(
        self,
        raw_memory: Dict[str, Any],
        query: str,
        planning_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        session = raw_memory.get("session_memory") or {}
        feedback = raw_memory.get("feedback_memory") or {}
        context = {
            "query": query,
            "learning_stage": planning_context.get("learning_stage") or "",
            "goal": planning_context.get("goal") or "",
            "preferred_subjects": planning_context.get("preferred_subjects") or [],
            "weak_points": planning_context.get("weak_points") or [],
            "strong_points": planning_context.get("strong_points") or [],
            "recent_learning_history": planning_context.get("recent_learning_history") or [],
            "constraints": planning_context.get("constraints") or {},
            "recent_turns": session.get("turns") or [],
            "turn_summary": session.get("turn_summary") or "",
            "recent_feedback_summary": feedback.get("recent_feedback_summary") or "",
        }
        return self.compressor.compress_context(context)

    def build_generator_context(
        self,
        raw_memory: Dict[str, Any],
        query: str,
        generation_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        session = raw_memory.get("session_memory") or {}
        context = {
            "query": query,
            "user_summary": generation_context.get("user_summary") or "",
            "memory_summary": generation_context.get("memory_summary") or "",
            "recent_turns": session.get("turns") or [],
            "turn_summary": session.get("turn_summary") or "",
            "tone_preferences": generation_context.get("tone_preferences") or {},
            "do_not_repeat_resource_ids": generation_context.get("do_not_repeat_resource_ids") or [],
            "clarification_history": generation_context.get("clarification_history") or {},
            "recent_feedback_summary": generation_context.get("recent_feedback_summary") or "",
        }
        return self.compressor.compress_context(context)
