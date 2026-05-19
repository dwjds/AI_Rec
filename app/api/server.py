from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.agent.orchestrator import AgentOrchestrator, OrchestratorResult
from app.api.schemas import (
    AgentRequest,
    AgentResponse,
    AgentSettingsRequest,
    AgentSettingsResponse,
    AuthResponse,
    FeedbackRequest,
    FeedbackResponse,
    LearningNoteRequest,
    LearningNoteResponse,
    LearningNotesResponse,
    LearningNoteUpdateRequest,
    LoginRequest,
    RegisterRequest,
    ResourceDetailResponse,
    ResourceSearchResponse,
    ResourceSummary,
    SavedResourcesResponse,
    SaveResourceRequest,
    UserContextResponse,
    UserProfileRequest,
    UserProfileResponse,
)
from app.core.config import settings
from app.db.migrations import init_app_db
from app.services.feedback_service import FeedbackService
from app.services.resource_service import ResourceService
from app.services.user_service import UserService


TYPE_LABELS = {
    "course": "课程",
    "chapter": "章节",
    "exercise": "练习",
    "challenge": "挑战任务",
    "knowledge_point": "知识点",
    "topic": "知识点",
}


@lru_cache(maxsize=1)
def get_user_service() -> UserService:
    return UserService()


@lru_cache(maxsize=1)
def get_resource_service() -> ResourceService:
    return ResourceService()


@lru_cache(maxsize=1)
def get_feedback_service() -> FeedbackService:
    return FeedbackService(user_store=get_user_service().user_store)


@lru_cache(maxsize=1)
def get_orchestrator() -> AgentOrchestrator:
    user_service = get_user_service()
    feedback_service = get_feedback_service()
    return AgentOrchestrator(user_service=user_service, feedback_service=feedback_service)


def create_app() -> FastAPI:
    init_app_db()
    app = FastAPI(
        title="MOOC RAG Learning Agent",
        description="面向 MOOC 学习场景的 RAG 教育资源推荐与学习规划 Agent API",
        version="0.1.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> Dict[str, Any]:
        return {
            "ok": True,
            "llm_enabled": settings.llm_enabled,
            "mooper_db": str(settings.mooper_db_path),
            "app_db": str(settings.app_db_path),
        }

    @app.post("/api/agent", response_model=AgentResponse)
    def run_agent(request: AgentRequest) -> AgentResponse:
        result = get_orchestrator().run(
            user_id=request.user_id,
            query=request.query,
            session_id=request.session_id,
            use_llm_route=request.use_llm_route,
            use_llm_rerank=request.use_llm_rerank,
            use_llm_generation=request.use_llm_generation,
            top_k=request.top_k,
        )
        return _agent_response(result)

    @app.post("/api/agent/stream")
    def run_agent_stream(request: AgentRequest) -> StreamingResponse:
        return StreamingResponse(
            _agent_stream_events(request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/api/auth/register", response_model=AuthResponse)
    def register(request: RegisterRequest) -> AuthResponse:
        try:
            account = get_user_service().register(
                username=request.username,
                password=request.password,
                display_name=request.display_name,
                learning_stage=request.learning_stage,
                goal=request.goal,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _auth_response(account)

    @app.post("/api/auth/login", response_model=AuthResponse)
    def login(request: LoginRequest) -> AuthResponse:
        account = get_user_service().login(username=request.username, password=request.password)
        if not account:
            raise HTTPException(status_code=401, detail="用户名或密码错误")
        return _auth_response(account)

    @app.get("/api/resources/search", response_model=ResourceSearchResponse)
    def search_resources(
        query: str = Query(default="人工智能入门课程"),
        resource_type: str = Query(default="all"),
        limit: int = Query(default=12, ge=1, le=50),
        offset: int = Query(default=0, ge=0),
    ) -> ResourceSearchResponse:
        service = get_resource_service()
        fetch_limit = 100
        if resource_type in {"all", "course", "video"}:
            all_rows = _search_courses_with_fallback(service, query=query, limit=fetch_limit)
        else:
            all_rows = _search_resources_with_fallback(service, query=query, resource_type=resource_type, limit=fetch_limit)
        rows = all_rows[offset : offset + limit]
        resources = [_resource_summary(row, score=_score_from_rank(index)) for index, row in enumerate(rows)]
        return ResourceSearchResponse(query=query, total=len(all_rows), resources=resources)

    @app.get("/api/resources/{resource_id}", response_model=ResourceDetailResponse)
    def get_resource_detail(resource_id: str) -> ResourceDetailResponse:
        service = get_resource_service()
        raw = service.get_course_detail(resource_id) or service.get_resource(resource_id)
        if not raw:
            raise HTTPException(status_code=404, detail="resource not found")
        summary = _resource_summary(raw, score=0.9)
        return ResourceDetailResponse(
            resource=summary,
            chapters=list(raw.get("chapters") or []),
            exercises=list(raw.get("exercises") or []),
            knowledge_points=list(raw.get("knowledge_points") or []),
            raw=raw,
        )

    @app.post("/api/feedback", response_model=FeedbackResponse)
    def record_feedback(request: FeedbackRequest) -> FeedbackResponse:
        event = get_feedback_service().record_feedback(
            user_id=request.user_id,
            resource_id=request.resource_id,
            feedback_type=request.feedback_type,
            comment=request.comment,
        )
        get_feedback_service().record_resource_event(
            user_id=request.user_id,
            resource_id=request.resource_id,
            event_type="feedback",
            payload={"feedback_type": request.feedback_type, "comment": request.comment},
        )
        return FeedbackResponse(ok=True, feedback=event)

    @app.get("/api/users/{user_id}/context", response_model=UserContextResponse)
    def get_user_context(user_id: str) -> UserContextResponse:
        return UserContextResponse(user_id=user_id, context=get_user_service().get_user_context(user_id))

    @app.put("/api/users/{user_id}/profile", response_model=UserProfileResponse)
    def update_user_profile(user_id: str, request: UserProfileRequest) -> UserProfileResponse:
        try:
            profile = get_user_service().update_profile(
                user_id=user_id,
                display_name=request.display_name,
                learning_stage=request.learning_stage,
                goal=request.goal,
                preferred_subjects=request.preferred_subjects,
                preferred_resource_types=request.preferred_resource_types,
                constraints=request.constraints,
                memory_summary=request.memory_summary,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return UserProfileResponse(user_id=user_id, profile=profile)

    @app.get("/api/users/{user_id}/saved-resources", response_model=SavedResourcesResponse)
    def list_saved_resources(user_id: str) -> SavedResourcesResponse:
        rows = get_user_service().list_saved_resources(user_id)
        resources = [_hydrate_saved_resource(row) for row in rows]
        return SavedResourcesResponse(
            user_id=user_id,
            total=len(rows),
            resources=[item for item in resources if item is not None],
            saved=rows,
        )

    @app.post("/api/users/saved-resources")
    def save_resource(request: SaveResourceRequest) -> Dict[str, Any]:
        row = get_user_service().save_resource(
            user_id=request.user_id,
            resource_id=request.resource_id,
            note=request.note,
        )
        get_feedback_service().record_resource_event(
            user_id=request.user_id,
            resource_id=request.resource_id,
            event_type="saved",
            payload={"note": request.note},
        )
        return {"ok": True, "saved": row}

    @app.delete("/api/users/{user_id}/saved-resources/{resource_id}")
    def unsave_resource(user_id: str, resource_id: str) -> Dict[str, Any]:
        row = get_user_service().unsave_resource(user_id=user_id, resource_id=resource_id)
        return {"ok": True, "saved": row}

    @app.get("/api/users/{user_id}/notes", response_model=LearningNotesResponse)
    def list_notes(user_id: str, limit: int = Query(default=100, ge=1, le=200)) -> LearningNotesResponse:
        notes = get_user_service().list_notes(user_id=user_id, limit=limit)
        return LearningNotesResponse(user_id=user_id, total=len(notes), notes=notes)

    @app.post("/api/users/notes", response_model=LearningNoteResponse)
    def create_note(request: LearningNoteRequest) -> LearningNoteResponse:
        try:
            note = get_user_service().create_note(
                user_id=request.user_id,
                content=request.content,
                title=request.title,
                tags=request.tags,
                linked_resource_id=request.linked_resource_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return LearningNoteResponse(note=note)

    @app.put("/api/users/{user_id}/notes/{note_id}", response_model=LearningNoteResponse)
    def update_note(user_id: str, note_id: str, request: LearningNoteUpdateRequest) -> LearningNoteResponse:
        try:
            note = get_user_service().update_note(
                user_id=user_id,
                note_id=note_id,
                title=request.title,
                content=request.content,
                tags=request.tags,
                linked_resource_id=request.linked_resource_id,
            )
        except ValueError as exc:
            status_code = 404 if "not found" in str(exc) else 400
            raise HTTPException(status_code=status_code, detail=str(exc)) from exc
        return LearningNoteResponse(note=note)

    @app.delete("/api/users/{user_id}/notes/{note_id}")
    def delete_note(user_id: str, note_id: str) -> Dict[str, Any]:
        return get_user_service().delete_note(user_id=user_id, note_id=note_id)

    @app.get("/api/users/{user_id}/settings", response_model=AgentSettingsResponse)
    def get_agent_settings(user_id: str) -> AgentSettingsResponse:
        return AgentSettingsResponse(user_id=user_id, settings=get_user_service().get_agent_settings(user_id))

    @app.put("/api/users/settings", response_model=AgentSettingsResponse)
    def update_agent_settings(request: AgentSettingsRequest) -> AgentSettingsResponse:
        settings_data = get_user_service().update_agent_settings(
            user_id=request.user_id,
            settings=request.settings,
        )
        return AgentSettingsResponse(user_id=request.user_id, settings=settings_data)

    _mount_frontend(app)
    return app


def _agent_stream_events(request: AgentRequest) -> Iterator[str]:
    orchestrator = get_orchestrator()
    try:
        result = orchestrator.run(
            user_id=request.user_id,
            query=request.query,
            session_id=request.session_id,
            use_llm_route=request.use_llm_route,
            use_llm_rerank=request.use_llm_rerank,
            use_llm_generation=False,
            top_k=request.top_k,
        )
        response = _agent_response(result)
        payload = _model_dump(response)
        payload["answer"] = ""
        yield _sse("meta", payload)

        state = result.state
        if state is None or result.handoff_case or result.routing_decision.needs_clarification:
            answer = result.answer
            for chunk in _chunk_text(answer):
                yield _sse("answer_delta", {"text": chunk})
            payload["answer"] = answer
            yield _sse("done", payload)
            return

        final_chunks: List[str] = []
        for chunk in orchestrator.response_generator.stream_generate(state, use_llm=request.use_llm_generation):
            final_chunks.append(chunk)
            yield _sse("answer_delta", {"text": chunk})

        final_answer = state.final_answer or "".join(final_chunks)
        result.answer = final_answer
        orchestrator.trace_recorder.finish(state, final_answer)
        response = _agent_response(result)
        yield _sse("done", _model_dump(response))
    except Exception as exc:
        yield _sse("error", {"message": str(exc)})


def _sse(event: str, data: Dict[str, Any]) -> str:
    return "event: {0}\ndata: {1}\n\n".format(
        event,
        json.dumps(data, ensure_ascii=False),
    )


def _model_dump(model: Any) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _chunk_text(text: str, size: int = 24) -> Iterator[str]:
    value = str(text or "")
    for start in range(0, len(value), size):
        yield value[start : start + size]


def _mount_frontend(app: FastAPI) -> None:
    frontend_dir = settings.project_root / "frontend"
    if not frontend_dir.exists():
        return
    app.mount("/frontend", StaticFiles(directory=str(frontend_dir)), name="frontend")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(frontend_dir / "index.html")


def _hydrate_saved_resource(row: Dict[str, Any]) -> Optional[ResourceSummary]:
    resource_id = str(row.get("resource_id") or "")
    if not resource_id:
        return None
    raw = get_resource_service().get_course_detail(resource_id) or get_resource_service().get_resource(resource_id)
    if not raw:
        return ResourceSummary(
            id=resource_id,
            title=resource_id,
            resource_type="resource",
            type_label="资源",
            reason="该资源已加入学习，但当前资源库未返回详情。",
        )
    return _resource_summary(raw, score=0.9)


def _agent_response(result: OrchestratorResult) -> AgentResponse:
    decision = result.routing_decision.to_dict()
    evidence_items = _evidence_items(result)
    recommendations = _recommendations(result)
    return AgentResponse(
        user_id=result.user_id,
        query=result.query,
        answer=result.answer,
        pipeline=result.pipeline,
        task_type=decision.get("task_type") or "",
        routing_decision=decision,
        recommendations=recommendations,
        evidence=evidence_items,
        trace_run_id=result.trace_run_id,
        handoff_case=result.handoff_case,
        metadata={
            "recommendation_event": result.recommendation_event,
            "feedback_event": result.feedback_event,
            "llm": result.metadata.get("response_generation") if result.metadata else None,
        },
    )


def _auth_response(account: Dict[str, Any]) -> AuthResponse:
    return AuthResponse(
        user_id=str(account.get("user_id") or ""),
        username=str(account.get("username") or ""),
        display_name=account.get("display_name"),
        learning_stage=account.get("learning_stage"),
        goal=account.get("goal"),
    )


def _recommendations(result: OrchestratorResult) -> List[ResourceSummary]:
    package = (result.metadata or {}).get("recommendation_package") or {}
    rows = package.get("recommendations") or []
    summaries = []
    for index, row in enumerate(rows):
        reasons = row.get("reasons") or []
        summaries.append(
            _resource_summary(
                row,
                score=float(row.get("score") or _score_from_rank(index)),
                reason="；".join(str(item) for item in reasons[:3]),
                evidence=_human_evidence_from_resource(row),
            )
        )
    if summaries:
        return summaries

    evidence_rows = _evidence_items(result)
    seen = set()
    for index, item in enumerate(evidence_rows):
        resource = item.get("resource") or {}
        resource_id = resource.get("id") or item.get("source_resource_id")
        if not resource_id or resource_id in seen:
            continue
        seen.add(resource_id)
        summaries.append(
            _resource_summary(
                {**resource, "id": resource_id, "title": resource.get("title") or item.get("title")},
                score=float(item.get("score") or _score_from_rank(index)),
                reason=str(item.get("content") or "")[:120],
                evidence=_human_evidence_from_evidence_item(item),
            )
        )
    return summaries[:8]


def _evidence_items(result: OrchestratorResult) -> List[Dict[str, Any]]:
    if not result.evidence_package:
        return []
    return [item.to_dict() for item in result.evidence_package.evidence_items]


def _resource_summary(
    row: Dict[str, Any],
    score: float = 0.0,
    reason: str = "",
    evidence: Optional[Iterable[str]] = None,
) -> ResourceSummary:
    resource_type = str(row.get("resource_type") or row.get("type") or "course")
    raw_kp = row.get("knowledge_points") or row.get("knowledge_point_names") or []
    knowledge_points = _knowledge_points(raw_kp)
    visits = _safe_int(row.get("visits"))
    chapter_count = _safe_int(row.get("chapter_count") or len(row.get("chapters") or []))
    exercise_count = _safe_int(row.get("exercise_count") or len(row.get("exercises") or []))
    challenge_count = _safe_int(row.get("challenge_count"))
    description = str(row.get("description") or row.get("learning_notes") or "")
    reason_text = reason or _default_reason(row, knowledge_points)
    difficulty, difficulty_label = _difficulty(row, chapter_count, exercise_count, challenge_count)
    normalized_score = score * 100 if score <= 1 else score
    if normalized_score <= 0:
        normalized_score = min(96, 72 + min(visits, 50000) / 2500)
    evidence_items = [str(item) for item in (evidence or []) if item]
    if not evidence_items:
        evidence_items = _human_evidence_from_resource(row)
    return ResourceSummary(
        id=str(row.get("resource_id") or row.get("id") or row.get("raw_id") or ""),
        title=str(row.get("title") or "未命名资源"),
        resource_type=resource_type,
        type_label=TYPE_LABELS.get(resource_type, "资源"),
        description=description,
        reason=reason_text,
        score=round(float(normalized_score), 1),
        difficulty=difficulty,
        difficulty_label=difficulty_label,
        knowledge_points=knowledge_points[:8],
        chapter_count=chapter_count,
        exercise_count=exercise_count,
        challenge_count=challenge_count,
        visits=visits,
        source="MOOPer",
        evidence=evidence_items,
        raw=row,
    )


def _knowledge_points(value: Any) -> List[str]:
    if isinstance(value, str):
        return [item for item in value.split("|") if item]
    if isinstance(value, list):
        points = []
        for item in value:
            if isinstance(item, dict):
                points.append(str(item.get("title") or item.get("topic_name") or item.get("id") or ""))
            else:
                points.append(str(item))
        return [item for item in points if item]
    return []


def _difficulty(row: Dict[str, Any], chapter_count: int, exercise_count: int, challenge_count: int) -> tuple[str, str]:
    explicit = str(row.get("difficulty") or "").lower()
    if explicit in {"easy", "medium", "hard"}:
        return explicit, {"easy": "易", "medium": "中等", "hard": "困难"}[explicit]
    workload = chapter_count + exercise_count + challenge_count
    if workload >= 60:
        return "hard", "困难"
    if workload >= 15:
        return "medium", "中等"
    return "easy", "易"


def _default_reason(row: Dict[str, Any], knowledge_points: List[str]) -> str:
    parts = []
    if knowledge_points:
        parts.append("覆盖 " + "、".join(knowledge_points[:4]))
    if row.get("chapter_count"):
        parts.append("包含 {0} 个章节".format(row.get("chapter_count")))
    if row.get("exercise_count") or row.get("challenge_count"):
        parts.append("配套练习和任务较完整")
    if row.get("visits"):
        parts.append("资源访问量较高")
    return "；".join(parts) or "与当前学习问题相关，可作为进一步学习材料。"


def _human_evidence_from_resource(row: Dict[str, Any]) -> List[str]:
    evidence = []
    title = row.get("title")
    if title:
        evidence.append("课程资料：" + str(title))
    for chapter in list(row.get("chapters") or [])[:2]:
        if chapter.get("title"):
            evidence.append("章节：" + str(chapter["title"]))
    for exercise in list(row.get("exercises") or [])[:2]:
        if exercise.get("title"):
            evidence.append("练习：" + str(exercise["title"]))
    for point in _knowledge_points(row.get("knowledge_points") or [])[:2]:
        evidence.append("知识点：" + point)
    return evidence[:5]


def _human_evidence_from_evidence_item(item: Dict[str, Any]) -> List[str]:
    resource = item.get("resource") or {}
    relations = item.get("relations") or {}
    evidence = []
    if resource.get("title"):
        evidence.append("课程资料：" + str(resource["title"]))
    elif item.get("title"):
        evidence.append("资源：" + str(item["title"]))
    for key, label in [("chapters", "章节"), ("exercises", "练习"), ("knowledge_points", "知识点")]:
        for row in list(relations.get(key) or [])[:2]:
            title = row.get("title")
            if title:
                evidence.append(label + "：" + str(title))
    return evidence[:5]


def _score_from_rank(index: int) -> float:
    return max(0.55, 0.92 - index * 0.035)


def _safe_int(value: Any) -> int:
    try:
        return int(str(value or "0"))
    except ValueError:
        return 0


def _search_courses_with_fallback(service: ResourceService, query: str, limit: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen = set()
    for variant in _query_variants(query):
        for row in service.search_courses(query=variant, limit=limit):
            row_id = row.get("id")
            if row_id and row_id not in seen:
                seen.add(row_id)
                rows.append(row)
            if len(rows) >= limit:
                return rows
    if not rows:
        rows = service.resource_store.list_courses(limit=limit)
    return rows[:limit]


def _search_resources_with_fallback(
    service: ResourceService,
    query: str,
    resource_type: str,
    limit: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen = set()
    for variant in _query_variants(query):
        for row in service.search_resources(query=variant, resource_types=[resource_type], limit=limit):
            row_id = row.get("id")
            if row_id and row_id not in seen:
                seen.add(row_id)
                rows.append(row)
            if len(rows) >= limit:
                return rows
    return rows[:limit]


def _query_variants(query: str) -> List[str]:
    text = str(query or "").strip()
    variants = [text] if text else []
    replacements = [
        ("推荐", ""),
        ("课程", ""),
        ("资源", ""),
        ("学习", ""),
        ("入门", ""),
        ("进阶", ""),
        ("的", ""),
    ]
    compact = text
    for old, new in replacements:
        compact = compact.replace(old, new)
    compact = compact.strip()
    if compact and compact not in variants:
        variants.append(compact)
    domain_terms = ["人工智能", "机器学习", "深度学习", "Python", "算法", "数据结构"]
    for item in domain_terms:
        if item in text and item not in variants:
            variants.append(item)
    return variants[:8]


app = create_app()
