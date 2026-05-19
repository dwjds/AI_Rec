from __future__ import annotations

from pathlib import Path

from app.core.config import settings
from app.db.sqlite import connect_app_db


APP_SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        external_id TEXT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS user_profiles (
        user_id TEXT PRIMARY KEY,
        display_name TEXT,
        learning_stage TEXT,
        goal TEXT,
        preferred_subjects TEXT,
        preferred_resource_types TEXT,
        constraints_json TEXT,
        memory_summary TEXT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS user_credentials (
        user_id TEXT PRIMARY KEY,
        username TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL,
        password_salt TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS user_knowledge_state (
        user_id TEXT NOT NULL,
        knowledge_point_id TEXT NOT NULL,
        mastery_score REAL,
        evidence_count INTEGER NOT NULL DEFAULT 0,
        last_evidence_at TEXT,
        source TEXT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (user_id, knowledge_point_id),
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS recommendation_events (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        query TEXT NOT NULL,
        intent_json TEXT,
        recommended_resource_ids TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS user_feedback (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        resource_id TEXT NOT NULL,
        feedback_type TEXT NOT NULL,
        comment TEXT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS user_resource_events (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        resource_id TEXT NOT NULL,
        event_type TEXT NOT NULL,
        event_payload_json TEXT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS saved_resources (
        user_id TEXT NOT NULL,
        resource_id TEXT NOT NULL,
        note TEXT,
        status TEXT NOT NULL DEFAULT 'saved',
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (user_id, resource_id),
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS learning_notes (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        title TEXT,
        content TEXT NOT NULL,
        tags_json TEXT,
        linked_resource_id TEXT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS user_agent_settings (
        user_id TEXT PRIMARY KEY,
        settings_json TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS agent_runs (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        request_type TEXT NOT NULL,
        user_query TEXT NOT NULL,
        final_answer TEXT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        finished_at TEXT,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS agent_steps (
        id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL,
        step_index INTEGER NOT NULL,
        action_name TEXT NOT NULL,
        action_input_json TEXT,
        observation_json TEXT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (run_id) REFERENCES agent_runs(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS handoff_cases (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        session_id TEXT,
        trace_run_id TEXT,
        task_type TEXT NOT NULL,
        query TEXT NOT NULL,
        reason_code TEXT NOT NULL,
        reason_text TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'open',
        priority TEXT NOT NULL DEFAULT 'normal',
        context_json TEXT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS tool_call_records (
        idempotency_key TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        session_id TEXT,
        tool_name TEXT NOT NULL,
        arguments_json TEXT NOT NULL,
        result_json TEXT,
        status TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    )
    """,
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_external_id ON users(external_id)",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_user_credentials_username ON user_credentials(username)",
    "CREATE INDEX IF NOT EXISTS idx_user_knowledge_state_user ON user_knowledge_state(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_user_knowledge_state_point ON user_knowledge_state(knowledge_point_id)",
    "CREATE INDEX IF NOT EXISTS idx_recommendation_events_user_time ON recommendation_events(user_id, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_user_feedback_user_resource ON user_feedback(user_id, resource_id)",
    "CREATE INDEX IF NOT EXISTS idx_user_resource_events_user_time ON user_resource_events(user_id, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_user_resource_events_resource ON user_resource_events(resource_id)",
    "CREATE INDEX IF NOT EXISTS idx_saved_resources_user_time ON saved_resources(user_id, updated_at)",
    "CREATE INDEX IF NOT EXISTS idx_learning_notes_user_time ON learning_notes(user_id, updated_at)",
    "CREATE INDEX IF NOT EXISTS idx_learning_notes_resource ON learning_notes(linked_resource_id)",
    "CREATE INDEX IF NOT EXISTS idx_agent_runs_user_time ON agent_runs(user_id, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_agent_steps_run_index ON agent_steps(run_id, step_index)",
    "CREATE INDEX IF NOT EXISTS idx_handoff_cases_user_time ON handoff_cases(user_id, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_handoff_cases_status ON handoff_cases(status, priority, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_tool_call_records_user_time ON tool_call_records(user_id, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_tool_call_records_tool ON tool_call_records(tool_name, created_at)",
]


def init_app_db(db_path: Path | None = None) -> Path:
    target = db_path or settings.app_db_path
    target.parent.mkdir(parents=True, exist_ok=True)

    with connect_app_db(target) as conn:
        for statement in APP_SCHEMA_STATEMENTS:
            conn.execute(statement)
    return target


if __name__ == "__main__":
    path = init_app_db()
    print("Initialized app database: {0}".format(path))
