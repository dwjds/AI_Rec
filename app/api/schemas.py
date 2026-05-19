from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    user_id: str = Field(default="demo_user", min_length=1)
    query: str = Field(..., min_length=1)
    session_id: Optional[str] = None
    use_llm_route: bool = True
    use_llm_rerank: bool = True
    use_llm_generation: bool = True
    top_k: int = Field(default=8, ge=1, le=20)


class ResourceSummary(BaseModel):
    id: str
    title: str
    resource_type: str = "course"
    type_label: str = "课程"
    description: str = ""
    reason: str = ""
    score: float = 0.0
    difficulty: str = "medium"
    difficulty_label: str = "中等"
    knowledge_points: List[str] = Field(default_factory=list)
    chapter_count: int = 0
    exercise_count: int = 0
    challenge_count: int = 0
    visits: int = 0
    source: str = "MOOPer"
    evidence: List[str] = Field(default_factory=list)
    raw: Dict[str, Any] = Field(default_factory=dict)


class AgentResponse(BaseModel):
    user_id: str
    query: str
    answer: str
    pipeline: str
    task_type: str
    routing_decision: Dict[str, Any]
    recommendations: List[ResourceSummary] = Field(default_factory=list)
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    trace_run_id: Optional[str] = None
    handoff_case: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ResourceSearchResponse(BaseModel):
    query: str
    total: int
    resources: List[ResourceSummary]


class ResourceDetailResponse(BaseModel):
    resource: ResourceSummary
    chapters: List[Dict[str, Any]] = Field(default_factory=list)
    exercises: List[Dict[str, Any]] = Field(default_factory=list)
    knowledge_points: List[Dict[str, Any]] = Field(default_factory=list)
    raw: Dict[str, Any] = Field(default_factory=dict)


class FeedbackRequest(BaseModel):
    user_id: str = Field(default="demo_user", min_length=1)
    resource_id: str = Field(..., min_length=1)
    feedback_type: str = Field(..., min_length=1)
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    ok: bool
    feedback: Dict[str, Any]


class UserContextResponse(BaseModel):
    user_id: str
    context: Dict[str, Any]


class UserProfileRequest(BaseModel):
    display_name: Optional[str] = None
    learning_stage: Optional[str] = None
    goal: Optional[str] = None
    preferred_subjects: List[str] = Field(default_factory=list)
    preferred_resource_types: List[str] = Field(default_factory=list)
    constraints: Dict[str, Any] = Field(default_factory=dict)
    memory_summary: Optional[str] = None


class UserProfileResponse(BaseModel):
    user_id: str
    profile: Dict[str, Any]


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=2)
    password: str = Field(..., min_length=6)
    display_name: Optional[str] = None
    learning_stage: Optional[str] = None
    goal: Optional[str] = None


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class AuthResponse(BaseModel):
    user_id: str
    username: str
    display_name: Optional[str] = None
    learning_stage: Optional[str] = None
    goal: Optional[str] = None


class SaveResourceRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    resource_id: str = Field(..., min_length=1)
    note: Optional[str] = None


class SavedResourcesResponse(BaseModel):
    user_id: str
    total: int
    resources: List[ResourceSummary] = Field(default_factory=list)
    saved: List[Dict[str, Any]] = Field(default_factory=list)


class LearningNoteRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    title: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    linked_resource_id: Optional[str] = None


class LearningNoteUpdateRequest(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    tags: Optional[List[str]] = None
    linked_resource_id: Optional[str] = None


class LearningNoteResponse(BaseModel):
    note: Dict[str, Any]


class LearningNotesResponse(BaseModel):
    user_id: str
    total: int
    notes: List[Dict[str, Any]] = Field(default_factory=list)


class AgentSettingsRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    settings: Dict[str, Any] = Field(default_factory=dict)


class AgentSettingsResponse(BaseModel):
    user_id: str
    settings: Dict[str, Any]
