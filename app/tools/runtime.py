from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, Optional

from app.agent.state import AgentState
from app.agent.trace import AgentTraceRecorder
from app.core.errors import PolicyError, ValidationError, normalize_error
from app.core.logging import get_logger
from app.stores.tool_call_store import ToolCallStore
from app.stores.trace_store import TraceStore
from app.tools.registry import ToolRegistry, create_default_tool_registry


logger = get_logger(__name__)


READ_TOOLS = {
    "search_learning_resources",
    "search_courses",
    "get_course_detail",
    "get_user_context",
    "get_feedback_summary",
}
WRITE_TOOLS = {
    "update_user_profile",
    "update_knowledge_state",
    "record_user_feedback",
}

RESULT_REQUIREMENTS: Dict[str, Dict[str, type]] = {
    "search_learning_resources": {"items": list},
    "search_courses": {"items": list},
    "get_course_detail": {"course": (dict, type(None))},  # type: ignore[dict-item]
    "get_user_context": {"profile": dict, "knowledge_state": list},
    "update_user_profile": {"profile": dict},
    "update_knowledge_state": {"knowledge_state": dict},
    "record_user_feedback": {"feedback": dict},
    "get_feedback_summary": {"liked_resource_ids": list, "disliked_resource_ids": list},
}


@dataclass
class ToolCallContext:
    user_id: str
    session_id: Optional[str] = None
    trace_run_id: Optional[str] = None
    task_type: str = "unknown"
    turn_id: Optional[str] = None
    permissions: set[str] = field(default_factory=set)
    state: Optional[AgentState] = None


@dataclass
class ToolCallResult:
    tool_name: str
    arguments: Dict[str, Any]
    result: Dict[str, Any] = field(default_factory=dict)
    ok: bool = True
    cached: bool = False
    idempotency_key: str = ""
    error: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SchemaValidator:
    """Validate the JSON Schema subset used by local tools."""

    def validate_and_fill(self, schema: Dict[str, Any], arguments: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(arguments, dict):
            raise ValidationError("tool arguments must be an object", stage="function_call.schema")
        if schema.get("type") != "object":
            return dict(arguments)

        properties = schema.get("properties") or {}
        required = set(schema.get("required") or [])
        cleaned: Dict[str, Any] = {}

        for name, prop_schema in properties.items():
            if name in arguments:
                cleaned[name] = self._coerce_and_validate(name, arguments[name], prop_schema)
            elif "default" in prop_schema:
                cleaned[name] = prop_schema["default"]

        for name, value in arguments.items():
            if name not in properties:
                cleaned[name] = value

        missing = [name for name in sorted(required) if self._is_missing(cleaned.get(name))]
        if missing:
            raise ValidationError(
                "missing required tool arguments: {0}".format(", ".join(missing)),
                stage="function_call.schema",
                detail={"missing": missing},
            )
        return cleaned

    def _coerce_and_validate(self, name: str, value: Any, schema: Dict[str, Any]) -> Any:
        expected = schema.get("type")
        if value is None:
            return value
        if expected == "string":
            if not isinstance(value, str):
                value = str(value)
            return value.strip()
        if expected == "integer":
            if isinstance(value, bool):
                raise ValidationError("{0} must be an integer".format(name), stage="function_call.schema")
            try:
                value = int(value)
            except (TypeError, ValueError) as exc:
                raise ValidationError("{0} must be an integer".format(name), stage="function_call.schema") from exc
            self._check_number_bounds(name, value, schema)
            return value
        if expected == "number":
            if isinstance(value, bool):
                raise ValidationError("{0} must be a number".format(name), stage="function_call.schema")
            try:
                value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValidationError("{0} must be a number".format(name), stage="function_call.schema") from exc
            self._check_number_bounds(name, value, schema)
            return value
        if expected == "array":
            if not isinstance(value, list):
                raise ValidationError("{0} must be an array".format(name), stage="function_call.schema")
            item_schema = schema.get("items") or {}
            return [self._coerce_and_validate("{0}[]".format(name), item, item_schema) for item in value]
        if expected == "object":
            if not isinstance(value, dict):
                raise ValidationError("{0} must be an object".format(name), stage="function_call.schema")
            return dict(value)
        if expected == "boolean":
            if not isinstance(value, bool):
                raise ValidationError("{0} must be a boolean".format(name), stage="function_call.schema")
            return value
        return value

    def _check_number_bounds(self, name: str, value: float, schema: Dict[str, Any]) -> None:
        if "minimum" in schema and value < schema["minimum"]:
            raise ValidationError("{0} is below minimum".format(name), stage="function_call.schema")
        if "maximum" in schema and value > schema["maximum"]:
            raise ValidationError("{0} is above maximum".format(name), stage="function_call.schema")

    def _is_missing(self, value: Any) -> bool:
        return value is None or value == ""


class ArgumentResolver:
    """Complete safe system-known arguments from AgentState."""

    def resolve(self, tool_name: str, arguments: Dict[str, Any], context: ToolCallContext) -> Dict[str, Any]:
        resolved = dict(arguments)
        if "user_id" in self._parameter_names(tool_name) and not resolved.get("user_id"):
            resolved["user_id"] = context.user_id
        if tool_name == "update_knowledge_state" and not resolved.get("source"):
            resolved["source"] = "agent_tool"
        if tool_name == "record_user_feedback" and not resolved.get("comment"):
            resolved["comment"] = ""
        return resolved

    def _parameter_names(self, tool_name: str) -> set[str]:
        if tool_name in {"get_user_context", "update_user_profile", "update_knowledge_state", "record_user_feedback", "get_feedback_summary"}:
            return {"user_id"}
        return set()


class PermissionPolicy:
    """Tool-level authorization guard."""

    def check(self, tool_name: str, arguments: Dict[str, Any], context: ToolCallContext) -> None:
        if tool_name in WRITE_TOOLS and "tool:write" not in context.permissions:
            raise PolicyError(
                "write tool is not permitted",
                stage="function_call.permission",
                detail={"tool_name": tool_name},
            )
        if tool_name in READ_TOOLS and "tool:read" not in context.permissions:
            raise PolicyError(
                "read tool is not permitted",
                stage="function_call.permission",
                detail={"tool_name": tool_name},
            )

        target_user_id = arguments.get("user_id")
        if target_user_id and str(target_user_id) != str(context.user_id):
            raise PolicyError(
                "tool call cannot access another user's data",
                stage="function_call.permission",
                detail={"tool_name": tool_name, "target_user_id": target_user_id},
            )

        if tool_name == "record_user_feedback":
            self._check_feedback_resource(arguments, context)

    def _check_feedback_resource(self, arguments: Dict[str, Any], context: ToolCallContext) -> None:
        resource_id = str(arguments.get("resource_id") or "")
        if not resource_id or context.state is None:
            return
        allowed = set()
        for event in context.state.user_context.get("recent_recommendations") or []:
            allowed.update(str(item) for item in event.get("recommended_resource_ids") or [])
        if context.state.recommendation_event:
            allowed.update(str(item) for item in context.state.recommendation_event.get("recommended_resource_ids") or [])
        if allowed and resource_id not in allowed:
            raise PolicyError(
                "feedback target is not in recent recommendation history",
                stage="function_call.permission",
                detail={"resource_id": resource_id},
            )


class IdempotencyGuard:
    def __init__(self, store: Optional[ToolCallStore] = None):
        self.store = store or ToolCallStore()

    def key_for(self, tool_name: str, arguments: Dict[str, Any], context: ToolCallContext) -> str:
        explicit = arguments.get("idempotency_key")
        if explicit:
            return str(explicit)
        payload = {
            "user_id": context.user_id,
            "session_id": context.session_id,
            "turn_id": context.turn_id,
            "tool_name": tool_name,
            "arguments": self._without_idempotency_key(arguments),
        }
        digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        return "tool_{0}".format(digest)

    def get_cached(self, key: str) -> Optional[Dict[str, Any]]:
        return self.store.get(key)

    def record(
        self,
        key: str,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Dict[str, Any],
        context: ToolCallContext,
        status: str,
    ) -> Dict[str, Any]:
        return self.store.record(
            idempotency_key=key,
            user_id=context.user_id,
            session_id=context.session_id,
            tool_name=tool_name,
            arguments=self._without_idempotency_key(arguments),
            result=result,
            status=status,
        )

    def _without_idempotency_key(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        return {key: value for key, value in arguments.items() if key != "idempotency_key"}


class ResultValidator:
    def validate(self, tool_name: str, raw_result: str) -> Dict[str, Any]:
        try:
            result = json.loads(raw_result)
        except json.JSONDecodeError as exc:
            raise ValidationError(
                "tool result is not valid JSON",
                stage="function_call.result",
                detail={"tool_name": tool_name},
            ) from exc
        if not isinstance(result, dict):
            raise ValidationError(
                "tool result must be a JSON object",
                stage="function_call.result",
                detail={"tool_name": tool_name},
            )

        requirements = RESULT_REQUIREMENTS.get(tool_name) or {}
        missing = [key for key in requirements if key not in result]
        if missing:
            raise ValidationError(
                "tool result missing required fields",
                stage="function_call.result",
                detail={"tool_name": tool_name, "missing": missing},
            )
        for key, expected_type in requirements.items():
            if not isinstance(result.get(key), expected_type):
                raise ValidationError(
                    "tool result field has invalid type",
                    stage="function_call.result",
                    detail={"tool_name": tool_name, "field": key},
                )
        return result


class FunctionCallRuntime:
    """Reliable execution gateway for model-selected tools."""

    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        tool_call_store: Optional[ToolCallStore] = None,
        trace_store: Optional[TraceStore] = None,
        schema_validator: Optional[SchemaValidator] = None,
        argument_resolver: Optional[ArgumentResolver] = None,
        permission_policy: Optional[PermissionPolicy] = None,
        result_validator: Optional[ResultValidator] = None,
    ):
        self.registry = registry or create_default_tool_registry()
        self.schema_validator = schema_validator or SchemaValidator()
        self.argument_resolver = argument_resolver or ArgumentResolver()
        self.permission_policy = permission_policy or PermissionPolicy()
        self.idempotency_guard = IdempotencyGuard(tool_call_store)
        self.result_validator = result_validator or ResultValidator()
        self.trace_recorder = AgentTraceRecorder(trace_store)

    async def execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        context: ToolCallContext,
    ) -> ToolCallResult:
        state = context.state
        try:
            tool = self.registry.get(tool_name)
            if tool is None:
                raise ValidationError(
                    "tool not found",
                    stage="function_call.registry",
                    detail={"tool_name": tool_name, "available_tools": self.registry.names()},
                )

            resolved = self.argument_resolver.resolve(tool_name, arguments or {}, context)
            validated = self.schema_validator.validate_and_fill(tool.parameters, resolved)
            self.permission_policy.check(tool_name, validated, context)

            idempotency_key = ""
            if tool_name in WRITE_TOOLS:
                idempotency_key = self.idempotency_guard.key_for(tool_name, validated, context)
                cached = self.idempotency_guard.get_cached(idempotency_key)
                if cached and cached.get("status") == "success":
                    result = ToolCallResult(
                        tool_name=tool_name,
                        arguments=validated,
                        result=cached.get("result") or {},
                        ok=True,
                        cached=True,
                        idempotency_key=idempotency_key,
                        metadata={"stage": "idempotency_cache_hit"},
                    )
                    self._trace(state, tool_name, validated, result.to_dict())
                    return result

            raw_result = await tool.execute(**validated)
            parsed_result = self.result_validator.validate(tool_name, raw_result)

            if idempotency_key:
                self.idempotency_guard.record(
                    key=idempotency_key,
                    tool_name=tool_name,
                    arguments=validated,
                    result=parsed_result,
                    context=context,
                    status="success",
                )

            result = ToolCallResult(
                tool_name=tool_name,
                arguments=validated,
                result=parsed_result,
                ok=True,
                cached=False,
                idempotency_key=idempotency_key,
            )
            self._trace(state, tool_name, validated, result.to_dict())
            return result
        except Exception as exc:
            payload = normalize_error(exc, stage="function_call").to_dict()
            result = ToolCallResult(
                tool_name=tool_name,
                arguments=arguments or {},
                result={},
                ok=False,
                error=payload,
            )
            if state is not None:
                state.add_error("function_call.{0}".format(tool_name), payload)
            self._trace(state, tool_name, arguments or {}, result.to_dict())
            logger.warning("function call failed: %s", payload.get("message"))
            return result

    def default_context(
        self,
        state: AgentState,
        permissions: Optional[Iterable[str]] = None,
        turn_id: Optional[str] = None,
    ) -> ToolCallContext:
        return ToolCallContext(
            user_id=state.user_id,
            session_id=state.session_id,
            trace_run_id=state.trace_run_id,
            task_type=state.task_type,
            turn_id=turn_id,
            permissions=set(permissions or {"tool:read"}),
            state=state,
        )

    def _trace(
        self,
        state: Optional[AgentState],
        tool_name: str,
        arguments: Dict[str, Any],
        observation: Dict[str, Any],
    ) -> None:
        if state is None or not state.trace_run_id:
            return
        self.trace_recorder.step(
            state=state,
            step_index=len(state.steps) + 1,
            action_name="tool.{0}".format(tool_name),
            action_input=self._redact(arguments),
            observation=observation,
        )

    def _redact(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        blocked = {"api_key", "token", "password", "secret", "authorization"}
        return {key: ("***" if key.lower() in blocked else value) for key, value in payload.items()}
