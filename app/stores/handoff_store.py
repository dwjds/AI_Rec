from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from app.db.migrations import init_app_db
from app.db.sqlite import connect_app_db, row_to_dict, rows_to_dicts


class HandoffStore:
    """Durable storage for human fallback cases."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path
        init_app_db(db_path)

    def create_case(
        self,
        user_id: str,
        query: str,
        task_type: str,
        reason_code: str,
        reason_text: str,
        session_id: Optional[str] = None,
        trace_run_id: Optional[str] = None,
        priority: str = "normal",
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        case_id = "handoff_{0}".format(uuid4().hex)
        with connect_app_db(self.db_path) as conn:
            conn.execute("INSERT OR IGNORE INTO users (id) VALUES (?)", (user_id,))
            conn.execute(
                """
                INSERT INTO handoff_cases (
                    id, user_id, session_id, trace_run_id, task_type, query,
                    reason_code, reason_text, status, priority, context_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open', ?, ?)
                """,
                (
                    case_id,
                    user_id,
                    session_id,
                    trace_run_id,
                    task_type,
                    query,
                    reason_code,
                    reason_text,
                    priority,
                    self._json_dumps(context or {}),
                ),
            )
            row = row_to_dict(conn.execute("SELECT * FROM handoff_cases WHERE id = ?", (case_id,)).fetchone())
        return self._decode(row or {})

    def get_case(self, case_id: str) -> Optional[Dict[str, Any]]:
        with connect_app_db(self.db_path) as conn:
            row = row_to_dict(conn.execute("SELECT * FROM handoff_cases WHERE id = ?", (case_id,)).fetchone())
        return self._decode(row) if row else None

    def list_cases(
        self,
        user_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        where: list[str] = []
        params: list[Any] = []
        if user_id:
            where.append("user_id = ?")
            params.append(user_id)
        if status:
            where.append("status = ?")
            params.append(status)
        sql = "SELECT * FROM handoff_cases"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with connect_app_db(self.db_path) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._decode(row) for row in rows_to_dicts(rows)]

    def update_status(self, case_id: str, status: str) -> Optional[Dict[str, Any]]:
        with connect_app_db(self.db_path) as conn:
            conn.execute(
                """
                UPDATE handoff_cases
                SET status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (status, case_id),
            )
            row = row_to_dict(conn.execute("SELECT * FROM handoff_cases WHERE id = ?", (case_id,)).fetchone())
        return self._decode(row) if row else None

    def _decode(self, row: Dict[str, Any]) -> Dict[str, Any]:
        row = dict(row)
        row["context"] = self._json_loads(row.pop("context_json", None), default={})
        return row

    def _json_dumps(self, value: Any) -> str:
        return json.dumps(value, ensure_ascii=False)

    def _json_loads(self, value: Any, default: Any) -> Any:
        if value in (None, ""):
            return default
        try:
            return json.loads(str(value))
        except (TypeError, json.JSONDecodeError):
            return default
