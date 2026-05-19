from __future__ import annotations

from typing import Any, Dict, List

"整理知识状态:weak points、strong points、unknown points、diagnosis hint。"
class KnowledgeStateMemory:
    """Summarize user mastery records for diagnosis and planning."""

    def build(self, user_context: Dict[str, Any]) -> Dict[str, Any]:
        rows = list(user_context.get("knowledge_state") or [])
        weak_points = []
        strong_points = []
        unknown_points = []

        for row in rows:
            item = {
                "knowledge_point_id": row.get("knowledge_point_id"),
                "mastery_score": row.get("mastery_score"),
                "evidence_count": row.get("evidence_count"),
                "source": row.get("source"),
                "last_evidence_at": row.get("last_evidence_at"),
            }
            score = row.get("mastery_score")
            if score is None:
                unknown_points.append(item)
                continue
            try:
                numeric_score = float(score)
            except (TypeError, ValueError):
                unknown_points.append(item)
                continue
            if numeric_score < 0.45:
                weak_points.append(item)
            elif numeric_score >= 0.75:
                strong_points.append(item)

        diagnosis_hint = ""
        if weak_points:
            diagnosis_hint = "存在 {0} 个低掌握度知识点，规划和诊断应优先补弱。".format(len(weak_points))
        elif rows:
            diagnosis_hint = "已有部分知识状态记录，可用于个性化排序。"

        return {
            "weak_points": weak_points[:20],
            "strong_points": strong_points[:20],
            "unknown_points": unknown_points[:20],
            "all_points": rows[:100],
            "diagnosis_hint": diagnosis_hint,
        }
