from __future__ import annotations

import contextvars
import json
import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from app.core.config import settings


_log_context: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar("log_context", default={})
_configured = False


class ContextFilter(logging.Filter):
    """Attach agent request context to every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        context = _log_context.get() or {}
        record.user_id = context.get("user_id", "")
        record.session_id = context.get("session_id", "")
        record.trace_run_id = context.get("trace_run_id", "")
        record.pipeline = context.get("pipeline", "")
        return True


class JsonFormatter(logging.Formatter):
    """Small JSON formatter for production-style structured logs."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "time": self.formatTime(record, self.datefmt),
            "user_id": getattr(record, "user_id", ""),
            "session_id": getattr(record, "session_id", ""),
            "trace_run_id": getattr(record, "trace_run_id", ""),
            "pipeline": getattr(record, "pipeline", ""),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(
    level: Optional[str] = None,
    log_file: Optional[Path | str] = None,
    json_logs: Optional[bool] = None,
    force: bool = False,
) -> None:
    """Configure application logging once.

    The default is human-readable console logs. Set LOG_JSON=true to emit JSON.
    A log file can be passed explicitly by API/server bootstrap code.
    """

    global _configured
    if _configured and not force:
        return

    log_level = _normalize_level(level or settings.log_level)
    use_json = _env_bool("LOG_JSON", default=False) if json_logs is None else bool(json_logs)
    formatter: logging.Formatter
    if use_json:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] "
            "user=%(user_id)s session=%(session_id)s trace=%(trace_run_id)s pipeline=%(pipeline)s - %(message)s"
        )

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(log_level)
    context_filter = ContextFilter()
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.addFilter(context_filter)
        root.addHandler(handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)


def set_log_context(**kwargs: Any) -> None:
    context = dict(_log_context.get() or {})
    for key, value in kwargs.items():
        if value is None:
            context.pop(key, None)
        else:
            context[key] = value
    _log_context.set(context)


def clear_log_context() -> None:
    _log_context.set({})


@contextmanager
def log_context(**kwargs: Any) -> Iterator[None]:
    previous = dict(_log_context.get() or {})
    merged = dict(previous)
    merged.update({key: value for key, value in kwargs.items() if value is not None})
    token = _log_context.set(merged)
    try:
        yield
    finally:
        _log_context.reset(token)


def _normalize_level(level: str) -> int:
    raw = str(level or "INFO").upper()
    return int(getattr(logging, raw, logging.INFO))


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
