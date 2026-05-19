from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from uuid import uuid4

from app.db.sqlite import connect_app_db, row_to_dict, rows_to_dicts

"""
实现了 UserStore，支持用户创建、画像读写、知识状态更新、
反馈记录、推荐历史、资源行为事件，以及统一的 get_user_context()。
"""
class UserStore:
    """Read/write access for app.db user data.

    This store owns durable user context: profile, knowledge state, feedback,
    recommendation history, and resource interaction events.
    """

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path

    def ensure_user(self, user_id: str, external_id: str | None = None) -> Dict[str, Any]:
        if not user_id:
            raise ValueError("user_id is required")
        with connect_app_db(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO users (id, external_id)
                VALUES (?, ?)
                """,
                (user_id, external_id),
            )
            conn.execute(
                """
                UPDATE users
                SET external_id = COALESCE(?, external_id),
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (external_id, user_id),
            )
            return row_to_dict(conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()) or {}

    def create_account(
        self,
        username: str,
        password: str,
        display_name: str | None = None,
        learning_stage: str | None = None,
        goal: str | None = None,
    ) -> Dict[str, Any]:
        username = str(username or "").strip()
        if not username:
            raise ValueError("username is required")
        if len(str(password or "")) < 6:
            raise ValueError("password must be at least 6 characters")

        user_id = self._new_id("user")
        salt = secrets.token_hex(16)
        password_hash = self._hash_password(password, salt)
        with connect_app_db(self.db_path) as conn:
            existing = conn.execute("SELECT user_id FROM user_credentials WHERE username = ?", (username,)).fetchone()
            if existing:
                raise ValueError("username already exists")
            conn.execute("INSERT INTO users (id, external_id) VALUES (?, ?)", (user_id, username))
            conn.execute(
                """
                INSERT INTO user_credentials (user_id, username, password_hash, password_salt)
                VALUES (?, ?, ?, ?)
                """,
                (user_id, username, password_hash, salt),
            )
        self.upsert_profile(
            user_id=user_id,
            display_name=display_name or username,
            learning_stage=learning_stage,
            goal=goal,
        )
        return self.get_account_by_username(username) or {}

    def authenticate(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        account = self.get_account_by_username(username)
        if not account:
            return None
        expected = self._hash_password(password, account.get("password_salt") or "")
        if not hmac.compare_digest(expected, str(account.get("password_hash") or "")):
            return None
        self.ensure_user(account["user_id"], external_id=username)
        return account

    def get_account_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        with connect_app_db(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT c.user_id, c.username, c.password_hash, c.password_salt,
                       p.display_name, p.learning_stage, p.goal
                FROM user_credentials c
                LEFT JOIN user_profiles p ON p.user_id = c.user_id
                WHERE c.username = ?
                """,
                (str(username or "").strip(),),
            ).fetchone()
            return row_to_dict(row)

    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        with connect_app_db(self.db_path) as conn:
            return row_to_dict(conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone())

    def upsert_profile(
        self,
        user_id: str,
        display_name: str | None = None,
        learning_stage: str | None = None,
        goal: str | None = None,
        preferred_subjects: Sequence[str] | None = None,
        preferred_resource_types: Sequence[str] | None = None,
        constraints: Dict[str, Any] | None = None,
        memory_summary: str | None = None,
    ) -> Dict[str, Any]:
        self.ensure_user(user_id)
        subjects_text = self._join_list(preferred_subjects)
        resource_types_text = self._join_list(preferred_resource_types)
        constraints_json = self._json_dumps(constraints or {}) if constraints is not None else None

        with connect_app_db(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO user_profiles (
                    user_id, display_name, learning_stage, goal,
                    preferred_subjects, preferred_resource_types,
                    constraints_json, memory_summary
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    display_name,
                    learning_stage,
                    goal,
                    subjects_text,
                    resource_types_text,
                    constraints_json,
                    memory_summary,
                ),
            )
            conn.execute(
                """
                UPDATE user_profiles
                SET display_name = COALESCE(?, display_name),
                    learning_stage = COALESCE(?, learning_stage),
                    goal = COALESCE(?, goal),
                    preferred_subjects = COALESCE(?, preferred_subjects),
                    preferred_resource_types = COALESCE(?, preferred_resource_types),
                    constraints_json = COALESCE(?, constraints_json),
                    memory_summary = COALESCE(?, memory_summary),
                    updated_at = CURRENT_TIMESTAMP
                WHERE user_id = ?
                """,
                (
                    display_name,
                    learning_stage,
                    goal,
                    subjects_text,
                    resource_types_text,
                    constraints_json,
                    memory_summary,
                    user_id,
                ),
            )
        return self.get_profile(user_id) or {}

    def get_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        with connect_app_db(self.db_path) as conn:
            row = row_to_dict(conn.execute("SELECT * FROM user_profiles WHERE user_id = ?", (user_id,)).fetchone())
        if row is None:
            return None
        return self._decode_profile(row)

    def upsert_knowledge_state(
        self,
        user_id: str,
        knowledge_point_id: str,
        mastery_score: float | None = None,
        evidence_count: int | None = None,
        source: str | None = None,
        last_evidence_at: str | None = None,
    ) -> Dict[str, Any]:
        if not knowledge_point_id:
            raise ValueError("knowledge_point_id is required")
        self.ensure_user(user_id)

        with connect_app_db(self.db_path) as conn:
            existing = row_to_dict(
                conn.execute(
                    """
                    SELECT *
                    FROM user_knowledge_state
                    WHERE user_id = ? AND knowledge_point_id = ?
                    """,
                    (user_id, knowledge_point_id),
                ).fetchone()
            )
            next_evidence_count = (
                int(evidence_count)
                if evidence_count is not None
                else int((existing or {}).get("evidence_count") or 0) + 1
            )

            conn.execute(
                """
                INSERT OR IGNORE INTO user_knowledge_state (
                    user_id, knowledge_point_id, mastery_score,
                    evidence_count, source, last_evidence_at
                )
                VALUES (?, ?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP))
                """,
                (
                    user_id,
                    knowledge_point_id,
                    mastery_score,
                    next_evidence_count,
                    source,
                    last_evidence_at,
                ),
            )
            conn.execute(
                """
                UPDATE user_knowledge_state
                SET mastery_score = COALESCE(?, mastery_score),
                    evidence_count = ?,
                    source = COALESCE(?, source),
                    last_evidence_at = COALESCE(?, CURRENT_TIMESTAMP),
                    updated_at = CURRENT_TIMESTAMP
                WHERE user_id = ? AND knowledge_point_id = ?
                """,
                (
                    mastery_score,
                    next_evidence_count,
                    source,
                    last_evidence_at,
                    user_id,
                    knowledge_point_id,
                ),
            )
            row = conn.execute(
                """
                SELECT * FROM user_knowledge_state
                WHERE user_id = ? AND knowledge_point_id = ?
                """,
                (user_id, knowledge_point_id),
            ).fetchone()
            return row_to_dict(row) or {}

    def list_knowledge_state(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        with connect_app_db(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM user_knowledge_state
                WHERE user_id = ?
                ORDER BY
                    CASE WHEN mastery_score IS NULL THEN 1 ELSE 0 END,
                    mastery_score ASC,
                    updated_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
            return rows_to_dicts(rows)

    def add_feedback(
        self,
        user_id: str,
        resource_id: str,
        feedback_type: str,
        comment: str | None = None,
    ) -> Dict[str, Any]:
        if not resource_id:
            raise ValueError("resource_id is required")
        if not feedback_type:
            raise ValueError("feedback_type is required")
        self.ensure_user(user_id)
        feedback_id = self._new_id("fb")

        with connect_app_db(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO user_feedback (id, user_id, resource_id, feedback_type, comment)
                VALUES (?, ?, ?, ?, ?)
                """,
                (feedback_id, user_id, resource_id, feedback_type, comment),
            )
            return row_to_dict(conn.execute("SELECT * FROM user_feedback WHERE id = ?", (feedback_id,)).fetchone()) or {}

    def list_feedback(
        self,
        user_id: str,
        resource_id: str | None = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        params: list[Any] = [user_id]
        where = "user_id = ?"
        if resource_id:
            where += " AND resource_id = ?"
            params.append(resource_id)
        params.append(limit)

        with connect_app_db(self.db_path) as conn:
            rows = conn.execute(
                f"""
                SELECT *
                FROM user_feedback
                WHERE {where}
                ORDER BY created_at DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
            return rows_to_dicts(rows)

    def record_recommendation_event(
        self,
        user_id: str,
        query: str,
        recommended_resource_ids: Sequence[str],
        intent: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        self.ensure_user(user_id)
        event_id = self._new_id("rec")

        with connect_app_db(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO recommendation_events (
                    id, user_id, query, intent_json, recommended_resource_ids
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    user_id,
                    query,
                    self._json_dumps(intent or {}),
                    self._json_dumps(list(recommended_resource_ids)),
                ),
            )
            row = row_to_dict(conn.execute("SELECT * FROM recommendation_events WHERE id = ?", (event_id,)).fetchone())
            return self._decode_recommendation_event(row or {})

    def list_recommendation_events(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        with connect_app_db(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM recommendation_events
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
            return [self._decode_recommendation_event(row) for row in rows_to_dicts(rows)]

    def record_resource_event(
        self,
        user_id: str,
        resource_id: str,
        event_type: str,
        payload: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        if not resource_id:
            raise ValueError("resource_id is required")
        if not event_type:
            raise ValueError("event_type is required")
        self.ensure_user(user_id)
        event_id = self._new_id("ure")

        with connect_app_db(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO user_resource_events (
                    id, user_id, resource_id, event_type, event_payload_json
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (event_id, user_id, resource_id, event_type, self._json_dumps(payload or {})),
            )
            row = row_to_dict(conn.execute("SELECT * FROM user_resource_events WHERE id = ?", (event_id,)).fetchone())
            return self._decode_resource_event(row or {})

    def list_resource_events(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        with connect_app_db(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM user_resource_events
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
            return [self._decode_resource_event(row) for row in rows_to_dicts(rows)]

    def save_resource(self, user_id: str, resource_id: str, note: str | None = None) -> Dict[str, Any]:
        if not resource_id:
            raise ValueError("resource_id is required")
        self.ensure_user(user_id)
        with connect_app_db(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO saved_resources (user_id, resource_id, note, status)
                VALUES (?, ?, ?, 'saved')
                ON CONFLICT(user_id, resource_id) DO UPDATE SET
                    note = COALESCE(excluded.note, saved_resources.note),
                    status = 'saved',
                    updated_at = CURRENT_TIMESTAMP
                """,
                (user_id, resource_id, note),
            )
            row = conn.execute(
                "SELECT * FROM saved_resources WHERE user_id = ? AND resource_id = ?",
                (user_id, resource_id),
            ).fetchone()
            return row_to_dict(row) or {}

    def unsave_resource(self, user_id: str, resource_id: str) -> Dict[str, Any]:
        self.ensure_user(user_id)
        with connect_app_db(self.db_path) as conn:
            conn.execute(
                """
                UPDATE saved_resources
                SET status = 'removed', updated_at = CURRENT_TIMESTAMP
                WHERE user_id = ? AND resource_id = ?
                """,
                (user_id, resource_id),
            )
            row = conn.execute(
                "SELECT * FROM saved_resources WHERE user_id = ? AND resource_id = ?",
                (user_id, resource_id),
            ).fetchone()
            return row_to_dict(row) or {"user_id": user_id, "resource_id": resource_id, "status": "removed"}

    def list_saved_resources(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        self.ensure_user(user_id)
        with connect_app_db(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM saved_resources
                WHERE user_id = ? AND status = 'saved'
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
            return rows_to_dicts(rows)

    def create_note(
        self,
        user_id: str,
        content: str,
        title: str | None = None,
        tags: Sequence[str] | None = None,
        linked_resource_id: str | None = None,
    ) -> Dict[str, Any]:
        if not str(content or "").strip():
            raise ValueError("content is required")
        self.ensure_user(user_id)
        note_id = self._new_id("note")
        with connect_app_db(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO learning_notes (id, user_id, title, content, tags_json, linked_resource_id)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (note_id, user_id, title, content, self._json_dumps(list(tags or [])), linked_resource_id),
            )
            row = conn.execute("SELECT * FROM learning_notes WHERE id = ?", (note_id,)).fetchone()
            return self._decode_note(row_to_dict(row) or {})

    def list_notes(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        self.ensure_user(user_id)
        with connect_app_db(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM learning_notes
                WHERE user_id = ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
            return [self._decode_note(row) for row in rows_to_dicts(rows)]

    def update_note(
        self,
        user_id: str,
        note_id: str,
        title: str | None = None,
        content: str | None = None,
        tags: Sequence[str] | None = None,
        linked_resource_id: str | None = None,
    ) -> Dict[str, Any]:
        self.ensure_user(user_id)
        if content is not None and not str(content).strip():
            raise ValueError("content is required")
        tags_json = self._json_dumps(list(tags or [])) if tags is not None else None
        with connect_app_db(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM learning_notes WHERE id = ? AND user_id = ?",
                (note_id, user_id),
            ).fetchone()
            if row is None:
                raise ValueError("note not found")
            conn.execute(
                """
                UPDATE learning_notes
                SET title = COALESCE(?, title),
                    content = COALESCE(?, content),
                    tags_json = COALESCE(?, tags_json),
                    linked_resource_id = COALESCE(?, linked_resource_id),
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND user_id = ?
                """,
                (title, content, tags_json, linked_resource_id, note_id, user_id),
            )
            updated = conn.execute(
                "SELECT * FROM learning_notes WHERE id = ? AND user_id = ?",
                (note_id, user_id),
            ).fetchone()
            return self._decode_note(row_to_dict(updated) or {})

    def delete_note(self, user_id: str, note_id: str) -> Dict[str, Any]:
        self.ensure_user(user_id)
        with connect_app_db(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM learning_notes WHERE id = ? AND user_id = ?",
                (note_id, user_id),
            ).fetchone()
            if row is None:
                return {"id": note_id, "deleted": False}
            conn.execute("DELETE FROM learning_notes WHERE id = ? AND user_id = ?", (note_id, user_id))
            return {"id": note_id, "deleted": True}

    def get_agent_settings(self, user_id: str) -> Dict[str, Any]:
        self.ensure_user(user_id)
        with connect_app_db(self.db_path) as conn:
            row = conn.execute("SELECT settings_json FROM user_agent_settings WHERE user_id = ?", (user_id,)).fetchone()
        if not row:
            return self._default_agent_settings()
        settings = self._json_loads(row["settings_json"], default={})
        return {**self._default_agent_settings(), **dict(settings or {})}

    def update_agent_settings(self, user_id: str, settings: Dict[str, Any]) -> Dict[str, Any]:
        self.ensure_user(user_id)
        merged = {**self._default_agent_settings(), **dict(settings or {})}
        with connect_app_db(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO user_agent_settings (user_id, settings_json)
                VALUES (?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    settings_json = excluded.settings_json,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (user_id, self._json_dumps(merged)),
            )
        return merged

    def get_user_context(
        self,
        user_id: str,
        knowledge_limit: int = 100,
        feedback_limit: int = 50,
        recommendation_limit: int = 10,
    ) -> Dict[str, Any]:
        self.ensure_user(user_id)
        return {
            "user": self.get_user(user_id) or {},
            "profile": self.get_profile(user_id) or {"user_id": user_id},
            "knowledge_state": self.list_knowledge_state(user_id, limit=knowledge_limit),
            "feedback": self.list_feedback(user_id, limit=feedback_limit),
            "recent_recommendations": self.list_recommendation_events(user_id, limit=recommendation_limit),
            "saved_resources": self.list_saved_resources(user_id, limit=100),
            "agent_settings": self.get_agent_settings(user_id),
        }

    def _decode_profile(self, row: Dict[str, Any]) -> Dict[str, Any]:
        row = dict(row)
        row["preferred_subjects"] = self._split_list(row.get("preferred_subjects"))
        row["preferred_resource_types"] = self._split_list(row.get("preferred_resource_types"))
        row["constraints"] = self._json_loads(row.pop("constraints_json", None), default={})
        return row

    def _decode_recommendation_event(self, row: Dict[str, Any]) -> Dict[str, Any]:
        row = dict(row)
        row["intent"] = self._json_loads(row.pop("intent_json", None), default={})
        row["recommended_resource_ids"] = self._json_loads(row.get("recommended_resource_ids"), default=[])
        return row

    def _decode_resource_event(self, row: Dict[str, Any]) -> Dict[str, Any]:
        row = dict(row)
        row["payload"] = self._json_loads(row.pop("event_payload_json", None), default={})
        return row

    def _decode_note(self, row: Dict[str, Any]) -> Dict[str, Any]:
        row = dict(row)
        row["tags"] = self._json_loads(row.pop("tags_json", None), default=[])
        return row

    def _default_agent_settings(self) -> Dict[str, Any]:
        return {
            "llm_generation": True,
            "llm_rerank": True,
            "hide_backend_details": True,
            "recommendation_top_k": 8,
            "answer_style": "consultant",
        }

    def _join_list(self, values: Sequence[str] | None) -> Optional[str]:
        if values is None:
            return None
        return "|".join(str(item).strip() for item in values if str(item).strip())

    def _split_list(self, value: Any) -> List[str]:
        if not value:
            return []
        if isinstance(value, list):
            return [str(item) for item in value]
        return [item for item in str(value).split("|") if item]

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

    def _hash_password(self, password: str, salt: str) -> str:
        return hashlib.pbkdf2_hmac(
            "sha256",
            str(password).encode("utf-8"),
            str(salt).encode("utf-8"),
            120000,
        ).hex()
