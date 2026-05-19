from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from app.agent.state import AgentState
from app.core.config import settings
from app.core.errors import normalize_error
from app.core.logging import get_logger
from app.stores.trace_store import TraceStore


logger = get_logger(__name__)


@dataclass
class TraceAction:
    name: str
    input: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TraceObservation:
    output: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AgentTraceRecorder:
    """Safe trace writer for AgentState + TraceStore.

    Trace failures should not break the user request, so every persistence call is
    guarded and mirrored into AgentState.errors.
    """

    def __init__(self, trace_store: Optional[TraceStore] = None, trace_json_path: Optional[Path] = None):
        self.trace_store = trace_store
        self.trace_json_path = trace_json_path or settings.trace_json_path

    def start(self, state: AgentState, request_type: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        if self.trace_store is None:
            return None
        try:
            run = self.trace_store.start_run(
                user_id=state.user_id,
                request_type=request_type,
                user_query=state.query,
            )
            run_id = run.get("id")
            state.set_trace_run_id(run_id)
            if metadata:
                state.add_metadata("trace_start_metadata", metadata)
            state.add_metadata("trace_json_path", str(self.trace_json_path))
            self._write_trace_json(run_id, run)
            logger.info("agent trace started")
            return run_id
        except Exception as exc:
            self._record_trace_error(state, "trace.start", exc)
            return None

    def step(
        self,
        state: AgentState,
        step_index: int,
        action_name: str,
        action_input: Optional[Dict[str, Any]] = None,
        observation: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
    ) -> None:
        target_run_id = run_id or state.trace_run_id
        if self.trace_store is None or not target_run_id:
            return
        try:
            self.trace_store.add_step(
                run_id=target_run_id,
                step_index=step_index,
                action_name=action_name,
                action_input=self._compact(action_input or {}),
                observation=self._compact(observation or {}),
            )
            self._write_trace_json(target_run_id)
            logger.debug("agent trace step recorded: %s", action_name)
        except Exception as exc:
            self._record_trace_error(state, "trace.step", exc)

    def error(
        self,
        state: AgentState,
        step_index: int,
        stage: str,
        error: BaseException,
        action_input: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = normalize_error(error, stage=stage).to_dict()
        state.add_error(stage, payload)
        self.step(
            state=state,
            step_index=step_index,
            action_name="{0}.error".format(stage),
            action_input=action_input or {},
            observation={"success": False, "error": payload},
        )

    def finish(self, state: AgentState, final_answer: Optional[str] = None) -> None:
        if self.trace_store is None or not state.trace_run_id:
            return
        try:
            run = self.trace_store.finish_run(state.trace_run_id, final_answer if final_answer is not None else state.final_answer)
            self._write_trace_json(state.trace_run_id, run)
            logger.info("agent trace finished")
        except Exception as exc:
            self._record_trace_error(state, "trace.finish", exc)

    def _record_trace_error(self, state: AgentState, stage: str, exc: BaseException) -> None:
        payload = normalize_error(exc, stage=stage).to_dict()
        state.add_error(stage, payload)
        logger.warning("trace operation failed: %s", payload.get("message"))

    def _compact(self, value: Dict[str, Any]) -> Dict[str, Any]:
        """Keep trace payload readable and avoid accidentally storing huge blobs."""
        blocked = {"api_key", "token", "password", "secret", "authorization"}
        compacted: Dict[str, Any] = {}
        for key, item in value.items():
            if key.lower() in blocked:
                compacted[key] = "***"
            elif isinstance(item, str) and len(item) > 2000:
                compacted[key] = item[:2000] + "...[truncated]"
            else:
                compacted[key] = item
        return compacted

    def _write_trace_json(self, run_id: Optional[str], run: Optional[Dict[str, Any]] = None) -> None:
        if self.trace_store is None or not run_id:
            return
        try:
            trace = run or self.trace_store.get_run(run_id)
            if not trace:
                return
            payload = self._json_safe(trace)
            latest_path = self.trace_json_path
            latest_path.parent.mkdir(parents=True, exist_ok=True)
            latest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

            run_path = latest_path.parent / "{0}.trace.json".format(run_id)
            run_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("trace json mirror failed: %s", exc)

    def _json_safe(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): self._json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._json_safe(item) for item in value]
        if isinstance(value, tuple):
            return [self._json_safe(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)
