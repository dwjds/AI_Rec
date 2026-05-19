from __future__ import annotations

import json
from typing import Any

from app.services.feedback_service import FeedbackService
from app.tools.base import Tool

"封装用户反馈记录和摘要功能的工具类，供 Agent 调用。"
class RecordFeedbackTool(Tool):
    def __init__(self, feedback_service: FeedbackService | None = None):
        self.feedback_service = feedback_service or FeedbackService()

    @property
    def name(self) -> str:
        return "record_user_feedback"

    @property
    def description(self) -> str:
        return "Record user feedback for a recommended resource."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "resource_id": {"type": "string"},
                "feedback_type": {
                    "type": "string",
                    "description": "Examples: like, dislike, helpful, not_interested, too_hard, too_easy.",
                },
                "comment": {"type": "string"},
            },
            "required": ["user_id", "resource_id", "feedback_type"],
        }

    async def execute(self, **kwargs: Any) -> str:
        row = self.feedback_service.record_feedback(
            user_id=str(kwargs.get("user_id") or ""),
            resource_id=str(kwargs.get("resource_id") or ""),
            feedback_type=str(kwargs.get("feedback_type") or ""),
            comment=kwargs.get("comment"),
        )
        return json.dumps({"feedback": row}, ensure_ascii=False)


class GetFeedbackSummaryTool(Tool):
    def __init__(self, feedback_service: FeedbackService | None = None):
        self.feedback_service = feedback_service or FeedbackService()

    @property
    def name(self) -> str:
        return "get_feedback_summary"

    @property
    def description(self) -> str:
        return "Get compact user feedback summary for recommendation adjustment."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 50},
            },
            "required": ["user_id"],
        }

    async def execute(self, **kwargs: Any) -> str:
        summary = self.feedback_service.get_feedback_summary(
            user_id=str(kwargs.get("user_id") or ""),
            limit=int(kwargs.get("limit") or 50),
        )
        return json.dumps(summary, ensure_ascii=False)
