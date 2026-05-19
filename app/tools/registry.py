from __future__ import annotations

from typing import Any

from .base import Tool
from .feedback_tool import GetFeedbackSummaryTool, RecordFeedbackTool
from .resource_tool import GetCourseDetailTool, SearchCoursesTool, SearchResourcesTool
from .user_tool import GetUserContextTool, UpdateKnowledgeStateTool, UpdateUserProfileTool


class ToolRegistry:
    """工具注册表，管理可用的工具"""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools.keys())

    def get_definitions(self) -> list[dict[str, Any]]:
        return [tool.to_schema() for tool in self._tools.values()]

    async def execute(self, name: str, params: dict[str, Any]) -> str:
        tool = self._tools.get(name)
        if not tool:
            return f"Error: Tool '{name}' not found."
        if not isinstance(params, dict):
            return (
                "Error: invalid tool arguments: expected a JSON object "
                f"but got {type(params).__name__}."
            )
        try:
            return await tool.execute(**params)
        except Exception as exc:
            return f"Error: {exc}"


def create_default_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(SearchResourcesTool())
    registry.register(SearchCoursesTool())
    registry.register(GetCourseDetailTool())
    registry.register(GetUserContextTool())
    registry.register(UpdateUserProfileTool())
    registry.register(UpdateKnowledgeStateTool())
    registry.register(RecordFeedbackTool())
    registry.register(GetFeedbackSummaryTool())
    return registry
