from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from app.stores.user_store import UserStore


class FeedbackService:
    """Feedback service for resource-level user signals."""

    def __init__(self, user_store: UserStore | None = None, db_path: Path | None = None):
        self.user_store = user_store or UserStore(db_path=db_path)

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

    def list_feedback(
        self,
        user_id: str,
        resource_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        return self.user_store.list_feedback(user_id=user_id, resource_id=resource_id, limit=limit)

    def record_resource_event(
        self,
        user_id: str,
        resource_id: str,
        event_type: str,
        payload: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        return self.user_store.record_resource_event(
            user_id=user_id,
            resource_id=resource_id,
            event_type=event_type,
            payload=payload or {},
        )

    def get_feedback_summary(self, user_id: str, limit: int = 100) -> Dict[str, Any]:
        rows = self.list_feedback(user_id=user_id, limit=limit)
        by_type: Dict[str, int] = {}
        disliked_resource_ids = []
        liked_resource_ids = []
        comments = []
        for row in rows:
            feedback_type = str(row.get("feedback_type") or "unknown")
            by_type[feedback_type] = by_type.get(feedback_type, 0) + 1
            resource_id = str(row.get("resource_id") or "")
            if feedback_type in {"dislike", "not_interested", "too_hard", "too_easy"} and resource_id:
                disliked_resource_ids.append(resource_id)
            if feedback_type in {"like", "helpful", "saved"} and resource_id:
                liked_resource_ids.append(resource_id)
            if row.get("comment"):
                comments.append(row["comment"])
        return {
            "total": len(rows),
            "by_type": by_type,
            "liked_resource_ids": liked_resource_ids[:30],
            "disliked_resource_ids": disliked_resource_ids[:30],
            "recent_comments": comments[:20],
        }
