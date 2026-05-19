from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class EvalCase:
    id: str
    query: str
    task_type: str = "generic"
    expected_resource_ids: List[str] = field(default_factory=list)
    expected_keywords: List[str] = field(default_factory=list)
    expected_trace_actions: List[str] = field(default_factory=list)
    expected_handoff_reason: Optional[str] = None
    user_id: str = "eval_user"
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "EvalCase":
        return cls(
            id=str(payload.get("id") or payload.get("case_id") or ""),
            query=str(payload.get("query") or ""),
            task_type=str(payload.get("task_type") or "generic"),
            expected_resource_ids=[str(item) for item in payload.get("expected_resource_ids") or []],
            expected_keywords=[str(item) for item in payload.get("expected_keywords") or []],
            expected_trace_actions=[str(item) for item in payload.get("expected_trace_actions") or []],
            expected_handoff_reason=payload.get("expected_handoff_reason") or None,
            user_id=str(payload.get("user_id") or "eval_user"),
            session_id=payload.get("session_id") or None,
            metadata=dict(payload.get("metadata") or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_eval_cases(path: Path) -> List[EvalCase]:
    if not path.exists():
        return []
    cases: List[EvalCase] = []
    with path.open("r", encoding="utf-8") as fp:
        for line_number, line in enumerate(fp, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            case = EvalCase.from_dict(payload)
            if case.metadata.get("disabled"):
                continue
            if not case.id:
                case.id = "{0}:{1}".format(path.stem, line_number)
            if case.query:
                cases.append(case)
    return cases


def load_eval_suite(case_dir: Path, suite_names: Iterable[str]) -> Dict[str, List[EvalCase]]:
    return {
        suite_name: load_eval_cases(case_dir / "{0}_cases.jsonl".format(suite_name))
        for suite_name in suite_names
    }


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
