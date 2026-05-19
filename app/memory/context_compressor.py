from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence


class ContextCompressor:
    """Lightweight context compressor before introducing token-budget logic.

    The current implementation uses character limits and item limits. It keeps
    structured fields stable, summarizes older session turns, and truncates long
    evidence/resource text.
    """

    def __init__(
        self,
        max_recent_turns: int = 8,
        max_summary_chars: int = 800,
        max_text_chars: int = 600,
        max_list_items: int = 12,
        max_evidence_items: int = 8,
    ):
        self.max_recent_turns = max(1, int(max_recent_turns))
        self.max_summary_chars = max(80, int(max_summary_chars))
        self.max_text_chars = max(80, int(max_text_chars))
        self.max_list_items = max(1, int(max_list_items))
        self.max_evidence_items = max(1, int(max_evidence_items))

    def compress_session_memory(self, session_memory: Dict[str, Any]) -> Dict[str, Any]:
        session = dict(session_memory or {})
        turns = list(session.get("turns") or [])
        older_turns = turns[:-self.max_recent_turns]
        recent_turns = turns[-self.max_recent_turns :]
        session["turns"] = [self._compact_turn(turn) for turn in recent_turns]
        session["turn_summary"] = self.summarize_turns(older_turns)
        session["dropped_turn_count"] = len(older_turns)
        return session

    def summarize_turns(self, turns: Sequence[Dict[str, Any]]) -> str:
        if not turns:
            return ""
        pieces: List[str] = []
        for turn in turns[-12:]:
            role = str(turn.get("role") or "unknown")
            content = self.truncate_text(turn.get("content") or "", 80)
            if content:
                pieces.append("{0}: {1}".format(role, content))
        return self.truncate_text(" | ".join(pieces), self.max_summary_chars)

    def compress_raw_memory(self, raw_memory: Dict[str, Any]) -> Dict[str, Any]:
        raw = dict(raw_memory or {})
        if "session_memory" in raw:
            raw["session_memory"] = self.compress_session_memory(raw["session_memory"])
        if "knowledge_state" in raw:
            raw["knowledge_state"] = self._compress_knowledge(raw["knowledge_state"])
        if "feedback_memory" in raw:
            raw["feedback_memory"] = self._compress_feedback(raw["feedback_memory"])
        if "resource_memory" in raw:
            raw["resource_memory"] = self._compress_resources(raw["resource_memory"])
        return raw

    def compress_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return self._compact_value(context, max_items=self.max_list_items)

    def compress_evidence_package(self, evidence_package: Any) -> Dict[str, Any]:
        if evidence_package is None:
            return {}
        data = evidence_package.to_dict() if hasattr(evidence_package, "to_dict") else dict(evidence_package or {})
        items = []
        for item in list(data.get("evidence_items") or [])[: self.max_evidence_items]:
            items.append(
                {
                    "evidence_id": item.get("evidence_id"),
                    "chunk_type": item.get("chunk_type"),
                    "source_resource_id": item.get("source_resource_id"),
                    "title": self.truncate_text(item.get("title") or "", 120),
                    "score": item.get("score"),
                    "content": self.truncate_text(item.get("content") or "", self.max_text_chars),
                    "resource": self._compact_resource(item.get("resource") or {}),
                    "relations": self._compact_value(item.get("relations") or {}, max_items=6),
                }
            )
        data["evidence_items"] = items
        data["evidence_count_before_compression"] = len(data.get("evidence_items") or items)
        return data

    def truncate_text(self, value: Any, limit: int | None = None) -> str:
        text = " ".join(str(value or "").split())
        target = int(limit or self.max_text_chars)
        if len(text) <= target:
            return text
        return text[:target].rstrip() + "...[truncated]"

    def _compact_turn(self, turn: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "role": turn.get("role"),
            "content": self.truncate_text(turn.get("content") or "", self.max_text_chars),
            "metadata": self._compact_value(turn.get("metadata") or {}, max_items=6),
        }

    def _compress_knowledge(self, knowledge: Dict[str, Any]) -> Dict[str, Any]:
        value = dict(knowledge or {})
        for key in ["weak_points", "strong_points", "unknown_points", "all_points"]:
            value[key] = self._limit_list(value.get(key) or [], self.max_list_items)
        return value

    def _compress_feedback(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        value = dict(feedback or {})
        for key in ["liked_resource_ids", "disliked_resource_ids", "positive_preferences", "negative_preferences"]:
            value[key] = self._limit_list(value.get(key) or [], self.max_list_items)
        value["recent_feedback_summary"] = self.truncate_text(value.get("recent_feedback_summary") or "", self.max_summary_chars)
        return value

    def _compress_resources(self, resources: Dict[str, Any]) -> Dict[str, Any]:
        value = dict(resources or {})
        for key in [
            "recent_recommendations",
            "recent_resource_events",
            "recent_recommended_resource_ids",
            "recent_viewed_resource_ids",
            "completed_resource_ids",
            "avoid_repeating_resource_ids",
        ]:
            value[key] = self._limit_list(value.get(key) or [], self.max_list_items)
        return value

    def _compact_resource(self, resource: Dict[str, Any]) -> Dict[str, Any]:
        compacted = dict(resource or {})
        for key in ["description", "learning_notes", "content"]:
            if key in compacted:
                compacted[key] = self.truncate_text(compacted.get(key) or "", self.max_text_chars)
        for key, value in list(compacted.items()):
            if isinstance(value, list):
                compacted[key] = self._limit_list(value, self.max_list_items)
        return compacted

    def _compact_value(self, value: Any, max_items: int) -> Any:
        if isinstance(value, dict):
            return {key: self._compact_value(item, max_items=max_items) for key, item in value.items()}
        if isinstance(value, list):
            return [self._compact_value(item, max_items=max_items) for item in value[:max_items]]
        if isinstance(value, str):
            return self.truncate_text(value, self.max_text_chars)
        return value

    def _limit_list(self, values: Iterable[Any], limit: int) -> List[Any]:
        return list(values or [])[:limit]
