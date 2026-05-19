from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from app.services.resource_service import ResourceService
from app.tools.base import Tool

"封装资源搜索、课程详情等功能的工具类，供 Agent 调用。"
class SearchResourcesTool(Tool):
    def __init__(self, resource_service: ResourceService | None = None):
        self.resource_service = resource_service or ResourceService()

    @property
    def name(self) -> str:
        return "search_learning_resources"

    @property
    def description(self) -> str:
        return "Search MOOC courses, chapters, exercises, or knowledge points by keyword."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "resource_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional resource types, such as course, chapter, exercise, topic.",
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 10},
            },
            "required": ["query"],
        }

    async def execute(self, **kwargs: Any) -> str:
        query = str(kwargs.get("query") or "")
        resource_types = kwargs.get("resource_types") or None
        limit = int(kwargs.get("limit") or 10)
        rows = self.resource_service.search_resources(query=query, resource_types=resource_types, limit=limit)
        return self._json({"items": self._compact(rows)})

    def _compact(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                "id": row.get("id"),
                "resource_type": row.get("resource_type") or row.get("entity_type"),
                "title": row.get("title"),
                "description": self._truncate(row.get("description"), 220),
                "visits": row.get("visits"),
                "difficulty": row.get("difficulty"),
            }
            for row in rows
        ]

    def _truncate(self, value: Any, limit: int) -> str:
        return " ".join(str(value or "").split())[:limit]

    def _json(self, value: Any) -> str:
        return json.dumps(value, ensure_ascii=False)


class SearchCoursesTool(Tool):
    def __init__(self, resource_service: ResourceService | None = None):
        self.resource_service = resource_service or ResourceService()

    @property
    def name(self) -> str:
        return "search_courses"

    @property
    def description(self) -> str:
        return "Search MOOC courses from the resource database."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 10},
                "discipline": {"type": "string"},
                "subdiscipline": {"type": "string"},
            },
            "required": ["query"],
        }

    async def execute(self, **kwargs: Any) -> str:
        rows = self.resource_service.search_courses(
            query=str(kwargs.get("query") or ""),
            limit=int(kwargs.get("limit") or 10),
            discipline=kwargs.get("discipline") or None,
            subdiscipline=kwargs.get("subdiscipline") or None,
        )
        return json.dumps({"items": self._compact_courses(rows)}, ensure_ascii=False)

    def _compact_courses(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                "id": row.get("id"),
                "title": row.get("title"),
                "description": " ".join(str(row.get("description") or "").split())[:260],
                "disciplines": row.get("disciplines", []),
                "subdisciplines": row.get("subdisciplines", []),
                "knowledge_points": row.get("knowledge_points", [])[:12],
                "chapter_count": row.get("chapter_count"),
                "exercise_count": row.get("exercise_count"),
                "visits": row.get("visits"),
            }
            for row in rows
        ]


class GetCourseDetailTool(Tool):
    def __init__(self, resource_service: ResourceService | None = None):
        self.resource_service = resource_service or ResourceService()

    @property
    def name(self) -> str:
        return "get_course_detail"

    @property
    def description(self) -> str:
        return "Get course detail including chapters, exercises, and knowledge points."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"course_id": {"type": "string"}},
            "required": ["course_id"],
        }

    async def execute(self, **kwargs: Any) -> str:
        course = self.resource_service.get_course_detail(str(kwargs.get("course_id") or ""))
        if course is None:
            return json.dumps({"course": None}, ensure_ascii=False)
        course = dict(course)
        course["chapters"] = course.get("chapters", [])[:20]
        course["exercises"] = course.get("exercises", [])[:20]
        course["knowledge_points"] = course.get("knowledge_points", [])[:40]
        return json.dumps({"course": course}, ensure_ascii=False)
