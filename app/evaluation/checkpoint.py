from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


class EvaluationCheckpoint:
    """Persist per-case evaluation results for resumable offline runs."""

    def __init__(self, path: Path, config_key: str = "", resume: bool = False):
        self.path = path
        self.config_key = config_key
        self.payload: Dict[str, Any] = {
            "config_key": config_key,
            "results": {},
        }
        if resume:
            self._load()

    def get(self, suite_name: str, case_id: str) -> Optional[Dict[str, Any]]:
        suite = (self.payload.get("results") or {}).get(suite_name) or {}
        result = suite.get(case_id)
        return dict(result) if isinstance(result, dict) else None

    def set(self, suite_name: str, case_id: str, result: Dict[str, Any]) -> None:
        self.payload.setdefault("results", {}).setdefault(suite_name, {})[case_id] = result
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()
        self.payload = {"config_key": self.config_key, "results": {}}

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        if not isinstance(payload, dict):
            return
        if str(payload.get("config_key") or "") != self.config_key:
            return
        self.payload = payload
