from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "processed" / "app.db"


SCHEMA_STATEMENTS = [
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
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_external_id ON users(external_id)",
    "CREATE INDEX IF NOT EXISTS idx_user_knowledge_state_user ON user_knowledge_state(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_user_knowledge_state_point ON user_knowledge_state(knowledge_point_id)",
    "CREATE INDEX IF NOT EXISTS idx_recommendation_events_user_time ON recommendation_events(user_id, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_user_feedback_user_resource ON user_feedback(user_id, resource_id)",
    "CREATE INDEX IF NOT EXISTS idx_user_resource_events_user_time ON user_resource_events(user_id, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_user_resource_events_resource ON user_resource_events(resource_id)",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize the application SQLite database for real user data.")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH, help="Output application database path.")
    parser.add_argument("--replace", action="store_true", help="Delete and recreate the database if it exists.")
    return parser.parse_args()


def init_db(db_path: Path, replace: bool) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists() and replace:
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        for statement in SCHEMA_STATEMENTS:
            conn.execute(statement)
        conn.commit()
    finally:
        conn.close()


def main() -> None:
    args = parse_args()
    db_path = args.db_path.resolve()
    init_db(db_path, replace=args.replace)
    print("Initialized application database: {0}".format(db_path))


if __name__ == "__main__":
    main()
