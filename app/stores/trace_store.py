from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from app.db.sqlite import connect_app_db, row_to_dict, rows_to_dicts
from app.stores.user_store import UserStore

"支持 Agent run / step 记录，后面 Agent Loop 每一步都能落库。"
class TraceStore:
    """Persist agent runs and step-level observations."""

    def __init__(self, db_path: Path | None = None, user_store: UserStore | None = None):
        self.db_path = db_path
        self.user_store = user_store or UserStore(db_path=db_path)

    def start_run(
        self,
        user_id: str,
        request_type: str,
        user_query: str,
        run_id: str | None = None,
    ) -> Dict[str, Any]:
        self.user_store.ensure_user(user_id)
        run_id = run_id or self._new_id("run")
        with connect_app_db(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO agent_runs (id, user_id, request_type, user_query)
                VALUES (?, ?, ?, ?)
                """,
                (run_id, user_id, request_type, user_query),
            )
            return row_to_dict(conn.execute("SELECT * FROM agent_runs WHERE id = ?", (run_id,)).fetchone()) or {}

    def add_step(
        self,
        run_id: str,
        step_index: int,
        action_name: str,
        action_input: Dict[str, Any] | None = None,
        observation: Dict[str, Any] | None = None,
        step_id: str | None = None,
    ) -> Dict[str, Any]:
        step_id = step_id or self._new_id("step")
        with connect_app_db(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO agent_steps (
                    id, run_id, step_index, action_name,
                    action_input_json, observation_json
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    step_id,
                    run_id,
                    int(step_index),
                    action_name,
                    self._json_dumps(action_input or {}),
                    self._json_dumps(observation or {}),
                ),
            )
            row = row_to_dict(conn.execute("SELECT * FROM agent_steps WHERE id = ?", (step_id,)).fetchone()) or {}
            return self._decode_step(row)

    def finish_run(self, run_id: str, final_answer: str) -> Dict[str, Any]:
        with connect_app_db(self.db_path) as conn:
            conn.execute(
                """
                UPDATE agent_runs
                SET final_answer = ?, finished_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (final_answer, run_id),
            )
        return self.get_run(run_id) or {}

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with connect_app_db(self.db_path) as conn:
            run = row_to_dict(conn.execute("SELECT * FROM agent_runs WHERE id = ?", (run_id,)).fetchone())
            if run is None:
                return None
            rows = conn.execute(
                """
                SELECT *
                FROM agent_steps
                WHERE run_id = ?
                ORDER BY step_index ASC, created_at ASC
                """,
                (run_id,),
            ).fetchall()
        run["steps"] = [self._decode_step(row) for row in rows_to_dicts(rows)]
        return run

    def list_runs(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        with connect_app_db(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM agent_runs
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (user_id, max(1, min(100, int(limit)))),
            ).fetchall()
            return rows_to_dicts(rows)

    def _decode_step(self, row: Dict[str, Any]) -> Dict[str, Any]:
        row = dict(row)
        row["action_input"] = self._json_loads(row.pop("action_input_json", None), default={})
        row["observation"] = self._json_loads(row.pop("observation_json", None), default={})
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

    def _new_id(self, prefix: str) -> str:
        return "{0}_{1}".format(prefix, uuid4().hex)
