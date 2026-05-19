from __future__ import annotations

from typing import Any, Dict, List

from app.stores.user_store import UserStore

"整理推荐和资源行为历史：最近推荐、浏览、完成、避免重复推荐资源"
class ResourceMemory:
    """Summarize recommendation and resource interaction history."""

    def __init__(self, user_store: UserStore | None = None):
        self.user_store = user_store or UserStore()

    def build(self, user_id: str, user_context: Dict[str, Any]) -> Dict[str, Any]:
        recent_recommendations = list(user_context.get("recent_recommendations") or [])
        resource_events = self.user_store.list_resource_events(user_id, limit=50)
        recent_recommended_resource_ids: List[str] = []
        for event in recent_recommendations:
            recent_recommended_resource_ids.extend(event.get("recommended_resource_ids") or [])

        recent_viewed_resource_ids = [
            str(row.get("resource_id"))
            for row in resource_events
            if row.get("event_type") in {"view", "open", "click"} and row.get("resource_id")
        ]
        completed_resource_ids = [
            str(row.get("resource_id"))
            for row in resource_events
            if row.get("event_type") in {"complete", "finish"} and row.get("resource_id")
        ]

        return {
            "recent_recommendations": recent_recommendations[:10],
            "recent_resource_events": resource_events[:30],
            "recent_recommended_resource_ids": self._unique(recent_recommended_resource_ids)[:50],
            "recent_viewed_resource_ids": self._unique(recent_viewed_resource_ids)[:50],
            "completed_resource_ids": self._unique(completed_resource_ids)[:50],
            "avoid_repeating_resource_ids": self._unique(recent_recommended_resource_ids + completed_resource_ids)[:80],
            "last_recommendation": recent_recommendations[0] if recent_recommendations else {},
        }

    def _unique(self, values: List[str]) -> List[str]:
        result = []
        for value in values:
            if value and value not in result:
                result.append(value)
        return result
