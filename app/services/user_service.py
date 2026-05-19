from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

from app.stores.user_store import UserStore

"""
封装成 RAG 可直接使用的用户上下文:
profile、knowledge_state、feedback、recent_recommendations。
"""
class UserService:
    """Application-facing user context service.

    It converts durable user records into the compact context consumed by RAG
    retrieval, reranking, and evidence building.
    """

    def __init__(self, user_store: UserStore | None = None, db_path: Path | None = None):
        self.user_store = user_store or UserStore(db_path=db_path)

    def ensure_user(self, user_id: str, external_id: str | None = None) -> Dict[str, Any]:
        return self.user_store.ensure_user(user_id, external_id=external_id)

    def register(
        self,
        username: str,
        password: str,
        display_name: str | None = None,
        learning_stage: str | None = None,
        goal: str | None = None,
    ) -> Dict[str, Any]:
        return self.user_store.create_account(
            username=username,
            password=password,
            display_name=display_name,
            learning_stage=learning_stage,
            goal=goal,
        )

    def login(self, username: str, password: str) -> Dict[str, Any] | None:
        return self.user_store.authenticate(username=username, password=password)

    def update_profile(
        self,
        user_id: str,
        display_name: str | None = None,
        learning_stage: str | None = None,
        goal: str | None = None,
        preferred_subjects: Sequence[str] | None = None,
        preferred_resource_types: Sequence[str] | None = None,
        constraints: Dict[str, Any] | None = None,
        memory_summary: str | None = None,
    ) -> Dict[str, Any]:
        return self.user_store.upsert_profile(
            user_id=user_id,
            display_name=display_name,
            learning_stage=learning_stage,
            goal=goal,
            preferred_subjects=preferred_subjects,
            preferred_resource_types=preferred_resource_types,
            constraints=constraints,
            memory_summary=memory_summary,
        )

    def update_knowledge_state(
        self,
        user_id: str,
        knowledge_point_id: str,
        mastery_score: float | None = None,
        source: str | None = None,
    ) -> Dict[str, Any]:
        return self.user_store.upsert_knowledge_state(
            user_id=user_id,
            knowledge_point_id=knowledge_point_id,
            mastery_score=mastery_score,
            source=source,
        )

    def record_feedback(
        self,
        user_id: str,
        resource_id: str,
        feedback_type: str,
        comment: str | None = None,
    ) -> Dict[str, Any]:
        return self.user_store.add_feedback(
            user_id=user_id,
            resource_id=resource_id,
            feedback_type=feedback_type,
            comment=comment,
        )

    def record_recommendation(
        self,
        user_id: str,
        query: str,
        recommended_resource_ids: Sequence[str],
        intent: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        return self.user_store.record_recommendation_event(
            user_id=user_id,
            query=query,
            recommended_resource_ids=recommended_resource_ids,
            intent=intent,
        )

    def get_user_context(self, user_id: str) -> Dict[str, Any]:
        context = self.user_store.get_user_context(user_id)
        profile = self._compact_profile(context.get("profile") or {"user_id": user_id})
        knowledge_state = self._compact_knowledge_state(context.get("knowledge_state") or [])
        feedback = self._compact_feedback(context.get("feedback") or [])
        recent_recommendations = context.get("recent_recommendations") or []

        return {
            "user_id": user_id,
            "profile": profile,
            "knowledge_state": knowledge_state,
            "feedback": feedback,
            "recent_recommendations": recent_recommendations,
            "saved_resources": context.get("saved_resources") or [],
            "agent_settings": context.get("agent_settings") or {},
        }

    def save_resource(self, user_id: str, resource_id: str, note: str | None = None) -> Dict[str, Any]:
        return self.user_store.save_resource(user_id=user_id, resource_id=resource_id, note=note)

    def unsave_resource(self, user_id: str, resource_id: str) -> Dict[str, Any]:
        return self.user_store.unsave_resource(user_id=user_id, resource_id=resource_id)

    def list_saved_resources(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        return self.user_store.list_saved_resources(user_id=user_id, limit=limit)

    def create_note(
        self,
        user_id: str,
        content: str,
        title: str | None = None,
        tags: Sequence[str] | None = None,
        linked_resource_id: str | None = None,
    ) -> Dict[str, Any]:
        return self.user_store.create_note(
            user_id=user_id,
            content=content,
            title=title,
            tags=tags,
            linked_resource_id=linked_resource_id,
        )

    def list_notes(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        return self.user_store.list_notes(user_id=user_id, limit=limit)

    def update_note(
        self,
        user_id: str,
        note_id: str,
        title: str | None = None,
        content: str | None = None,
        tags: Sequence[str] | None = None,
        linked_resource_id: str | None = None,
    ) -> Dict[str, Any]:
        return self.user_store.update_note(
            user_id=user_id,
            note_id=note_id,
            title=title,
            content=content,
            tags=tags,
            linked_resource_id=linked_resource_id,
        )

    def delete_note(self, user_id: str, note_id: str) -> Dict[str, Any]:
        return self.user_store.delete_note(user_id=user_id, note_id=note_id)

    def get_agent_settings(self, user_id: str) -> Dict[str, Any]:
        return self.user_store.get_agent_settings(user_id)

    def update_agent_settings(self, user_id: str, settings: Dict[str, Any]) -> Dict[str, Any]:
        return self.user_store.update_agent_settings(user_id=user_id, settings=settings)

    def _compact_profile(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "user_id": profile.get("user_id"),
            "display_name": profile.get("display_name"),
            "learning_stage": profile.get("learning_stage"),
            "goal": profile.get("goal"),
            "preferred_subjects": profile.get("preferred_subjects") or [],
            "preferred_resource_types": profile.get("preferred_resource_types") or [],
            "constraints": profile.get("constraints") or {},
            "memory_summary": profile.get("memory_summary"),
        }

    def _compact_knowledge_state(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        compacted = []
        for row in rows:
            compacted.append(
                {
                    "knowledge_point_id": row.get("knowledge_point_id"),
                    "mastery_score": row.get("mastery_score"),
                    "evidence_count": row.get("evidence_count"),
                    "source": row.get("source"),
                    "last_evidence_at": row.get("last_evidence_at"),
                }
            )
        return compacted

    def _compact_feedback(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        compacted = []
        for row in rows:
            compacted.append(
                {
                    "resource_id": row.get("resource_id"),
                    "feedback_type": row.get("feedback_type"),
                    "comment": row.get("comment"),
                    "created_at": row.get("created_at"),
                }
            )
        return compacted
