from __future__ import annotations

from typing import Any, Dict, List

"整理反馈偏好:liked/disliked resources、boost/penalize keywords、推荐调整信号。"
class FeedbackMemory:
    """Convert raw feedback rows into preference and ranking signals."""

    POSITIVE_TYPES = {"like", "helpful", "saved"}
    NEGATIVE_TYPES = {"dislike", "not_interested", "too_hard", "too_easy"}

    def build(self, user_context: Dict[str, Any]) -> Dict[str, Any]:
        rows = list(user_context.get("feedback") or [])
        liked_resource_ids: List[str] = []
        disliked_resource_ids: List[str] = []
        positive_preferences: List[str] = []
        negative_preferences: List[str] = []
        boost_keywords: List[str] = []
        penalize_keywords: List[str] = []

        for row in rows:
            feedback_type = str(row.get("feedback_type") or "")
            resource_id = str(row.get("resource_id") or "")
            comment = str(row.get("comment") or "")
            if feedback_type in self.POSITIVE_TYPES and resource_id:
                liked_resource_ids.append(resource_id)
                if comment:
                    positive_preferences.append(comment)
                    boost_keywords.extend(self._keywords(comment))
            if feedback_type in self.NEGATIVE_TYPES and resource_id:
                disliked_resource_ids.append(resource_id)
                if comment:
                    negative_preferences.append(comment)
                    penalize_keywords.extend(self._keywords(comment))

        return {
            "liked_resource_ids": self._unique(liked_resource_ids)[:50],
            "disliked_resource_ids": self._unique(disliked_resource_ids)[:50],
            "positive_preferences": positive_preferences[:20],
            "negative_preferences": negative_preferences[:20],
            "recommendation_adjustments": {
                "exclude_resource_ids": self._unique(disliked_resource_ids)[:50],
                "boost_keywords": self._unique(boost_keywords)[:20],
                "penalize_keywords": self._unique(penalize_keywords)[:20],
            },
            "recent_feedback_summary": self._summary(rows),
        }

    def _keywords(self, text: str) -> List[str]:
        words = []
        for item in text.replace("，", " ").replace(",", " ").split():
            item = item.strip()
            if len(item) >= 2:
                words.append(item[:30])
        return words

    def _summary(self, rows: List[Dict[str, Any]]) -> str:
        if not rows:
            return ""
        negative_count = sum(1 for row in rows if row.get("feedback_type") in self.NEGATIVE_TYPES)
        positive_count = sum(1 for row in rows if row.get("feedback_type") in self.POSITIVE_TYPES)
        return "近期反馈 {0} 条，其中正向 {1} 条，负向 {2} 条。".format(len(rows), positive_count, negative_count)

    def _unique(self, values: List[str]) -> List[str]:
        result = []
        for value in values:
            if value and value not in result:
                result.append(value)
        return result
