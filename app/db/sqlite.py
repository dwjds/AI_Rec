from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from app.core.config import settings


def row_to_dict(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    return {key: row[key] for key in row.keys()}


def rows_to_dicts(rows: list[sqlite3.Row]) -> list[Dict[str, Any]]:
    return [dict(row_to_dict(row) or {}) for row in rows]


def _connect(path: Path, readonly: bool = False, timeout: float = 30.0) -> sqlite3.Connection:
    db_path = path.expanduser().resolve()
    if readonly:
        uri = "file:{0}?mode=ro".format(db_path.as_posix())
        conn = sqlite3.connect(uri, uri=True, timeout=timeout)
    else:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path), timeout=timeout)

    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


@contextmanager
def connect_app_db(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    conn = _connect(path or settings.app_db_path, readonly=False)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@contextmanager
def connect_resource_db(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    conn = _connect(path or settings.mooper_db_path, readonly=True)
    try:
        yield conn
    finally:
        conn.close()


def check_database_files() -> Dict[str, Any]:
    return {
        "mooper_db_path": str(settings.mooper_db_path),
        "mooper_db_exists": settings.mooper_db_path.exists(),
        "app_db_path": str(settings.app_db_path),
        "app_db_exists": settings.app_db_path.exists(),
        "chunks_path": str(settings.chunks_path),
        "chunks_exists": settings.chunks_path.exists(),
        "chroma_dir": str(settings.chroma_dir),
        "chroma_exists": settings.chroma_dir.exists(),
    }
