from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from app.db.migrations import init_app_db
from app.db.sqlite import connect_app_db, row_to_dict


class ToolCallStore:
    """Durable idempotency records for Function Calling tool execution."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path
        init_app_db(db_path)

    def get(self, idempotency_key: str) -> Optional[Dict[str, Any]]:
        if not idempotency_key:
            return None
        with connect_app_db(self.db_path) as conn:
            row = row_to_dict(
                conn.execute(
                    "SELECT * FROM tool_call_records WHERE idempotency_key = ?",
                    (idempotency_key,),
                ).fetchone()
            )
        return self._decode(row) if row else None

    def record(
        self,
        idempotency_key: str,
        user_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Dict[str, Any],
        status: str,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        with connect_app_db(self.db_path) as conn:
            conn.execute("INSERT OR IGNORE INTO users (id) VALUES (?)", (user_id,))
            conn.execute(
                """
                INSERT OR REPLACE INTO tool_call_records (
                    idempotency_key, user_id, session_id, tool_name,
                    arguments_json, result_json, status, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (
                    idempotency_key,
                    user_id,
                    session_id,
                    tool_name,
                    self._json_dumps(arguments),
                    self._json_dumps(result),
                    status,
                ),
            )
            row = row_to_dict(
                conn.execute(
                    "SELECT * FROM tool_call_records WHERE idempotency_key = ?",
                    (idempotency_key,),
                ).fetchone()
            )
        return self._decode(row or {})

    def _decode(self, row: Dict[str, Any]) -> Dict[str, Any]:
        row = dict(row)
        row["arguments"] = self._json_loads(row.pop("arguments_json", None), default={})
        row["result"] = self._json_loads(row.pop("result_json", None), default={})
        return row

    def _json_dumps(self, value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)

    def _json_loads(self, value: Any, default: Any) -> Any:
        if value in (None, ""):
            return default
        try:
            return json.loads(str(value))
        except (TypeError, json.JSONDecodeError):
            return default
