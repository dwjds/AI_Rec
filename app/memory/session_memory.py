from __future__ import annotations

from typing import Any, Dict, List, Optional

"MVP 内存态短期会话记忆:turns、pending clarification、last routing decision、last recommendation ids。"
class SessionMemory:
    """In-memory short-term session memory for MVP multi-turn context."""

    def __init__(self) -> None:
        self._sessions: Dict[str, Dict[str, Any]] = {}

    def get_session_memory(self, session_id: Optional[str]) -> Dict[str, Any]:
        if not session_id:
            return self._empty_session(None)
        session = self._sessions.setdefault(session_id, self._empty_session(session_id))
        return {
            "session_id": session.get("session_id"),
            "turns": list(session.get("turns") or []),
            "pending_clarification": dict(session.get("pending_clarification") or {}),
            "last_routing_decision": dict(session.get("last_routing_decision") or {}),
            "last_recommendation_ids": list(session.get("last_recommendation_ids") or []),
        }

    def add_turn(self, session_id: str, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        session = self._sessions.setdefault(session_id, self._empty_session(session_id))
        session["turns"].append(
            {
                "role": role,
                "content": content,
                "metadata": metadata or {},
            }
        )
        session["turns"] = session["turns"][-30:]

    def update_session_state(
        self,
        session_id: str,
        pending_clarification: Optional[Dict[str, Any]] = None,
        last_routing_decision: Optional[Dict[str, Any]] = None,
        last_recommendation_ids: Optional[List[str]] = None,
    ) -> None:
        session = self._sessions.setdefault(session_id, self._empty_session(session_id))
        if pending_clarification is not None:
            session["pending_clarification"] = pending_clarification
        if last_routing_decision is not None:
            session["last_routing_decision"] = last_routing_decision
        if last_recommendation_ids is not None:
            session["last_recommendation_ids"] = list(last_recommendation_ids)

    def _empty_session(self, session_id: Optional[str]) -> Dict[str, Any]:
        return {
            "session_id": session_id,
            "turns": [],
            "pending_clarification": {},
            "last_routing_decision": {},
            "last_recommendation_ids": [],
        }
