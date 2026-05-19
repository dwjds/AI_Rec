from __future__ import annotations

from typing import Any, Dict


class CollaborativeMemory:
    """Placeholder for future MemRec-style collaborative signals."""

    def build(self, user_id: str, user_context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "enabled": False,
            "user_id": user_id,
            "signals": [],
            "summary": "",
        }
