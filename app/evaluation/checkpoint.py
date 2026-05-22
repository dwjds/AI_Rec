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

    def remove_failed(self, suite_names: Optional[list[str]] = None) -> int:
        results = self.payload.get("results")
        if not isinstance(results, dict):
            return 0

        removed = 0
        target_suites = suite_names or list(results.keys())
        for suite_name in target_suites:
            suite = results.get(suite_name)
            if not isinstance(suite, dict):
                continue
            failed_ids = [
                case_id
                for case_id, result in suite.items()
                if isinstance(result, dict) and result.get("passed") is False
            ]
            for case_id in failed_ids:
                suite.pop(case_id, None)
                removed += 1
        if removed:
            self.save()
        return removed

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        if not isinstance(payload, dict):
            return
        stored_config_key = str(payload.get("config_key") or "")
        if stored_config_key != self.config_key and not self._is_compatible_config_key(stored_config_key):
            return
        self.payload = payload
        self.payload["config_key"] = self.config_key

    def _is_compatible_config_key(self, stored_config_key: str) -> bool:
        try:
            stored = json.loads(stored_config_key)
            current = json.loads(self.config_key)
        except json.JSONDecodeError:
            return False
        if not isinstance(stored, dict) or not isinstance(current, dict):
            return False
        return (
            stored.get("evaluation_schema_version") == current.get("evaluation_schema_version")
            and stored.get("config") == current.get("config")
        )
