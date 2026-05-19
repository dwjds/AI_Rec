from __future__ import annotations

import json
from typing import Any

from app.services.user_service import UserService
from app.tools.base import Tool

"封装用户上下文访问和更新的工具类，供 Agent 调用。"
class GetUserContextTool(Tool):
    def __init__(self, user_service: UserService | None = None):
        self.user_service = user_service or UserService()

    @property
    def name(self) -> str:
        return "get_user_context"

    @property
    def description(self) -> str:
        return "Get user profile, knowledge state, feedback, and recent recommendation history."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"user_id": {"type": "string"}},
            "required": ["user_id"],
        }

    async def execute(self, **kwargs: Any) -> str:
        context = self.user_service.get_user_context(str(kwargs.get("user_id") or ""))
        return json.dumps(context, ensure_ascii=False)


class UpdateUserProfileTool(Tool):
    def __init__(self, user_service: UserService | None = None):
        self.user_service = user_service or UserService()

    @property
    def name(self) -> str:
        return "update_user_profile"

    @property
    def description(self) -> str:
        return "Create or update a user profile for personalization."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "display_name": {"type": "string"},
                "learning_stage": {"type": "string"},
                "goal": {"type": "string"},
                "preferred_subjects": {"type": "array", "items": {"type": "string"}},
                "preferred_resource_types": {"type": "array", "items": {"type": "string"}},
                "constraints": {"type": "object"},
                "memory_summary": {"type": "string"},
            },
            "required": ["user_id"],
        }

    async def execute(self, **kwargs: Any) -> str:
        profile = self.user_service.update_profile(
            user_id=str(kwargs.get("user_id") or ""),
            display_name=kwargs.get("display_name"),
            learning_stage=kwargs.get("learning_stage"),
            goal=kwargs.get("goal"),
            preferred_subjects=kwargs.get("preferred_subjects"),
            preferred_resource_types=kwargs.get("preferred_resource_types"),
            constraints=kwargs.get("constraints"),
            memory_summary=kwargs.get("memory_summary"),
        )
        return json.dumps({"profile": profile}, ensure_ascii=False)


class UpdateKnowledgeStateTool(Tool):
    def __init__(self, user_service: UserService | None = None):
        self.user_service = user_service or UserService()

    @property
    def name(self) -> str:
        return "update_knowledge_state"

    @property
    def description(self) -> str:
        return "Update a user's mastery score for a knowledge point."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "knowledge_point_id": {"type": "string"},
                "mastery_score": {"type": "number", "minimum": 0, "maximum": 1},
                "source": {"type": "string"},
            },
            "required": ["user_id", "knowledge_point_id"],
        }

    async def execute(self, **kwargs: Any) -> str:
        state = self.user_service.update_knowledge_state(
            user_id=str(kwargs.get("user_id") or ""),
            knowledge_point_id=str(kwargs.get("knowledge_point_id") or ""),
            mastery_score=kwargs.get("mastery_score"),
            source=kwargs.get("source"),
        )
        return json.dumps({"knowledge_state": state}, ensure_ascii=False)
