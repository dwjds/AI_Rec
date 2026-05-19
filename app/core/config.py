from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None  # type: ignore[assignment]

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


if load_dotenv is not None:
    load_dotenv(_project_root() / ".env")


def _env_path(name: str, default: Path) -> Path:
    raw_value = os.environ.get(name)
    if not raw_value:
        return default
    return Path(raw_value).expanduser().resolve()


def _env_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if not raw_value:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw_value = os.environ.get(name)
    if not raw_value:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_dir: Path
    processed_data_dir: Path
    index_dir: Path
    chroma_dir: Path
    trace_dir: Path
    mooper_db_path: Path
    app_db_path: Path
    chunks_path: Path
    chunk_id_map_path: Path
    trace_json_path: Path
    chroma_collection: str
    llm_api_key: Optional[str]
    llm_base_url: str
    llm_model: str
    embedding_model: str
    agent_max_iterations: int = 6
    agent_loop_timeout_seconds: float = 600.0
    agent_step_timeout_seconds: float = 150.0
    llm_validation_failure_threshold: int = 2
    log_level: str = "INFO"

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_api_key and OpenAI)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    root = _project_root()
    data_dir = _env_path("DATA_DIR", root / "data")
    processed_data_dir = _env_path("PROCESSED_DATA_DIR", data_dir / "processed")
    index_dir = _env_path("INDEX_DIR", data_dir / "indexes")
    chroma_dir = _env_path("CHROMA_DIR", data_dir / "chroma")
    trace_dir = _env_path("TRACE_DIR", data_dir / "traces")

    return Settings(
        project_root=root,
        data_dir=data_dir,
        processed_data_dir=processed_data_dir,
        index_dir=index_dir,
        chroma_dir=chroma_dir,
        trace_dir=trace_dir,
        mooper_db_path=_env_path("MOOPER_DB_PATH", processed_data_dir / "mooper.db"),
        app_db_path=_env_path("APP_DB_PATH", processed_data_dir / "app.db"),
        chunks_path=_env_path("CHUNKS_PATH", index_dir / "chunks.jsonl"),
        chunk_id_map_path=_env_path("CHUNK_ID_MAP_PATH", index_dir / "chunk_id_map.json"),
        trace_json_path=_env_path("TRACE_JSON_PATH", trace_dir / "trace.json"),
        chroma_collection=os.environ.get("CHROMA_COLLECTION", "mooc_resource_chunks"),
        llm_api_key=os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY"),
        llm_base_url=os.environ.get("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        llm_model=os.environ.get("LLM_MODEL", "qwen-plus-2025-09-11"),
        embedding_model=os.environ.get("EMBEDDING_MODEL", "text-embedding-v4"),
        agent_max_iterations=_env_int("AGENT_MAX_ITERATIONS", 6),
        agent_loop_timeout_seconds=_env_float("AGENT_LOOP_TIMEOUT_SECONDS", 600.0),
        agent_step_timeout_seconds=_env_float("AGENT_STEP_TIMEOUT_SECONDS", 150.0),
        llm_validation_failure_threshold=_env_int("LLM_VALIDATION_FAILURE_THRESHOLD", 2),
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
    )


settings = get_settings()


def create_llm_client() -> Any:
    if not settings.llm_enabled:
        return None
    return OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
