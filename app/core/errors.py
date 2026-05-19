from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ErrorPayload:
    code: str
    message: str
    user_message: str
    stage: str = "unknown"
    status_code: int = 500
    retryable: bool = False
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_debug: bool = False) -> Dict[str, Any]:
        payload = {
            "code": self.code,
            "message": self.message,
            "user_message": self.user_message,
            "stage": self.stage,
            "status_code": self.status_code,
            "retryable": self.retryable,
            "detail": self.detail,
        }
        if not include_debug:
            payload["detail"] = self._safe_detail(self.detail)
        return payload

    def _safe_detail(self, detail: Dict[str, Any]) -> Dict[str, Any]:
        blocked = {"api_key", "token", "password", "secret", "authorization"}
        return {key: value for key, value in detail.items() if key.lower() not in blocked}


class AppError(Exception):
    """Base structured application exception."""

    code = "APP_ERROR"
    status_code = 500
    retryable = False
    user_message = "系统暂时无法完成请求，请稍后再试。"

    def __init__(
        self,
        message: str = "",
        *,
        code: Optional[str] = None,
        status_code: Optional[int] = None,
        user_message: Optional[str] = None,
        stage: str = "unknown",
        retryable: Optional[bool] = None,
        detail: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message or user_message or self.user_message)
        self.message = message or user_message or self.user_message
        self.code = code or self.code
        self.status_code = int(status_code or self.status_code)
        self.user_message = user_message or self.user_message
        self.stage = stage
        self.retryable = self.retryable if retryable is None else bool(retryable)
        self.detail = detail or {}

    def to_payload(self) -> ErrorPayload:
        return ErrorPayload(
            code=self.code,
            message=self.message,
            user_message=self.user_message,
            stage=self.stage,
            status_code=self.status_code,
            retryable=self.retryable,
            detail=self.detail,
        )


class ConfigurationError(AppError):
    code = "CONFIGURATION_ERROR"
    user_message = "系统配置不完整，请检查服务配置。"


class DatabaseError(AppError):
    code = "DATABASE_ERROR"
    retryable = True
    user_message = "数据库访问暂时失败，请稍后再试。"


class StoreError(AppError):
    code = "STORE_ERROR"
    retryable = True
    user_message = "数据读写暂时失败，请稍后再试。"


class RoutingError(AppError):
    code = "ROUTING_ERROR"
    user_message = "我暂时没能判断你的需求类型，请换一种方式描述。"


class RetrievalError(AppError):
    code = "RETRIEVAL_ERROR"
    retryable = True
    user_message = "资源检索暂时失败，请稍后再试。"


class PlannerError(AppError):
    code = "PLANNER_ERROR"
    user_message = "学习规划生成失败，请补充目标或当前基础后再试。"


class GenerationError(AppError):
    code = "GENERATION_ERROR"
    retryable = True
    user_message = "回答生成暂时失败，请稍后再试。"


class PolicyError(AppError):
    code = "POLICY_ERROR"
    user_message = "当前任务动作不符合 Agent 执行策略。"


class TraceError(AppError):
    code = "TRACE_ERROR"
    retryable = True
    user_message = "执行轨迹记录失败，但不影响继续对话。"


class ValidationError(AppError):
    code = "VALIDATION_ERROR"
    status_code = 400
    user_message = "请求参数不完整或格式不正确。"


class NotFoundError(AppError):
    code = "NOT_FOUND"
    status_code = 404
    user_message = "没有找到对应的数据。"


def normalize_error(exc: BaseException, stage: str = "unknown", include_traceback: bool = False) -> ErrorPayload:
    if isinstance(exc, AppError):
        payload = exc.to_payload()
        if stage != "unknown":
            payload.stage = stage
        return payload

    detail: Dict[str, Any] = {"exception_type": exc.__class__.__name__}
    if include_traceback:
        detail["traceback"] = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return ErrorPayload(
        code="UNEXPECTED_ERROR",
        message=str(exc),
        user_message="系统遇到未预期的问题，请稍后再试。",
        stage=stage,
        status_code=500,
        retryable=True,
        detail=detail,
    )


def error_to_dict(exc: BaseException, stage: str = "unknown", include_debug: bool = False) -> Dict[str, Any]:
    return normalize_error(exc, stage=stage, include_traceback=include_debug).to_dict(include_debug=include_debug)


def safe_user_message(exc: BaseException) -> str:
    return normalize_error(exc).user_message
