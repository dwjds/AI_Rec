from __future__ import annotations

from typing import Any, Dict, List

"从用户画像整理长期用户记忆：学习阶段、目标、偏好方向、偏好资源类型、约束、摘要。"
class UserMemory:
    """Normalize durable profile fields into agent-facing user memory."""

    def build(self, user_context: Dict[str, Any]) -> Dict[str, Any]:
        profile = user_context.get("profile") or {}
        subjects = self._list(profile.get("preferred_subjects"))
        resource_types = self._list(profile.get("preferred_resource_types"))
        constraints = profile.get("constraints") or {}
        goal = str(profile.get("goal") or "")
        learning_stage = str(profile.get("learning_stage") or "")

        summary_parts: List[str] = []
        if learning_stage:
            summary_parts.append("学习阶段：{0}".format(learning_stage))
        if goal:
            summary_parts.append("学习目标：{0}".format(goal))
        if subjects:
            summary_parts.append("偏好方向：{0}".format("、".join(subjects[:5])))
        if resource_types:
            summary_parts.append("偏好资源类型：{0}".format("、".join(resource_types[:5])))
        if profile.get("memory_summary"):
            summary_parts.append(str(profile.get("memory_summary")))

        return {
            "user_id": profile.get("user_id") or user_context.get("user_id"),
            "display_name": profile.get("display_name"),
            "learning_stage": learning_stage,
            "goals": [goal] if goal else [],
            "preferred_subjects": subjects,
            "preferred_resource_types": resource_types,
            "constraints": constraints,
            "summary": "；".join(summary_parts),
        }

    def _list(self, value: Any) -> List[str]:
        if not value:
            return []
        if isinstance(value, list):
            return [str(item) for item in value if str(item)]
        return [item for item in str(value).split("|") if item]
