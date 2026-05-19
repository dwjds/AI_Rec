from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from app.rag.retriever import RetrievalResult
from app.stores.resource_store import ResourceStore


TASK_TYPE_PRIORITY: Dict[str, Dict[str, int]] = {
    "recommend": {
        "course": 100,
        "chapter": 70,
        "exercise": 60,
        "knowledge_point": 50,
    },
    "qa": {
        "knowledge_point": 100,
        "chapter": 85,
        "exercise": 65,
        "course": 55,
    },
    "learning_path": {
        "course": 100,
        "chapter": 95,
        "exercise": 85,
        "knowledge_point": 80,
    },
    "diagnosis": {
        "knowledge_point": 100,
        "exercise": 95,
        "chapter": 75,
        "course": 60,
    },
    "generic": {
        "course": 80,
        "chapter": 80,
        "exercise": 80,
        "knowledge_point": 80,
    },
}


@dataclass
class EvidenceItem:
    evidence_id: str
    chunk_id: str
    chunk_type: str
    source_resource_id: str
    title: str
    content: str
    score: float
    priority: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    resource: Dict[str, Any] = field(default_factory=dict)
    relations: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvidencePackage:
    query: str
    task_type: str
    user_profile: Dict[str, Any]
    knowledge_state: List[Dict[str, Any]]
    query_rewrite: Dict[str, Any]
    evidence_items: List[EvidenceItem]
    sections: Dict[str, List[Dict[str, Any]]]
    instructions: List[str]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["evidence_items"] = [item.to_dict() for item in self.evidence_items]
        return data


class EvidenceBuilder:
    """Build task-aware evidence packages from retrieved chunks and resource details."""

    def __init__(self, resource_store: Optional[ResourceStore] = None):
        self.resource_store = resource_store or ResourceStore()

    def build(
        self,
        query: str,
        task_type: str,
        retrieval_results: Sequence[RetrievalResult],
        user_profile: Optional[Dict[str, Any]] = None,
        knowledge_state: Optional[List[Dict[str, Any]]] = None,
        max_items: int = 12,
    ) -> EvidencePackage:
        normalized_task = task_type if task_type in TASK_TYPE_PRIORITY else "generic"
        items = [self._build_item(index, normalized_task, result) for index, result in enumerate(retrieval_results, start=1)]
        items = self._sort_by_task_priority(items, normalized_task)[:max_items]
        sections = self._build_sections(items, normalized_task)
        query_rewrite = self._extract_query_rewrite(retrieval_results)

        return EvidencePackage(
            query=query,
            task_type=normalized_task,
            user_profile=self._compact_profile(user_profile or {}),
            knowledge_state=(knowledge_state or [])[:20],
            query_rewrite=query_rewrite,
            evidence_items=items,
            sections=sections,
            instructions=self._instructions(normalized_task),
        )

    def _build_item(self, index: int, task_type: str, result: RetrievalResult) -> EvidenceItem:
        source_resource_id = self._source_resource_id(result)
        resource = self._load_resource(result.chunk_type, source_resource_id)
        relations = self._load_relations(result.chunk_type, source_resource_id)
        priority = TASK_TYPE_PRIORITY.get(task_type, TASK_TYPE_PRIORITY["generic"]).get(result.chunk_type, 50)
        return EvidenceItem(
            evidence_id="E{0}".format(index),
            chunk_id=result.chunk_id,
            chunk_type=result.chunk_type,
            source_resource_id=source_resource_id,
            title=result.title,
            content=self._truncate(result.content, 900),
            score=float(result.score),
            priority=priority,
            metadata=self._compact_metadata(result.metadata),
            resource=resource,
            relations=relations,
        )

    def _load_resource(self, chunk_type: str, resource_id: str) -> Dict[str, Any]:
        if not resource_id:
            return {}
        if chunk_type == "course":
            detail = self.resource_store.get_course_detail(resource_id)
            return self._compact_course_detail(detail or {})
        resource = self.resource_store.get_resource(resource_id)
        return self._compact_resource(resource or {})

    def _load_relations(self, chunk_type: str, resource_id: str) -> Dict[str, Any]:
        if not resource_id:
            return {}
        if chunk_type == "course":
            return {
                "chapters": self._compact_related(self.resource_store.list_course_chapters(resource_id, limit=8)),
                "exercises": self._compact_related(self.resource_store.list_course_exercises(resource_id, limit=8)),
                "knowledge_points": self._compact_related(self.resource_store.get_course_knowledge_points(resource_id, limit=20)),
            }
        if chunk_type == "chapter":
            return {
                "courses": self._compact_related(self.resource_store.get_neighbors(resource_id, relation="has_chapter", direction="in", limit=3)),
            }
        if chunk_type == "exercise":
            return {
                "courses": self._compact_related(self.resource_store.get_neighbors(resource_id, relation="has_exercise", direction="in", limit=3)),
                "challenges": self._compact_related(self.resource_store.list_exercise_challenges(resource_id, limit=8)),
            }
        if chunk_type == "knowledge_point":
            return {
                "related_challenges": self._compact_related(self.resource_store.get_neighbors(resource_id, relation="has_topic", direction="in", limit=8)),
            }
        return {}

    def _build_sections(self, items: List[EvidenceItem], task_type: str) -> Dict[str, List[Dict[str, Any]]]:
        if task_type == "recommend":
            return {
                "primary_resources": self._items_by_type(items, ["course"]),
                "supporting_evidence": self._items_by_type(items, ["chapter", "exercise", "knowledge_point"]),
            }
        if task_type == "qa":
            return {
                "concept_evidence": self._items_by_type(items, ["knowledge_point"]),
                "context_resources": self._items_by_type(items, ["chapter", "course", "exercise"]),
            }
        if task_type == "learning_path":
            return {
                "path_resources": self._items_by_type(items, ["course"]),
                "stage_evidence": self._items_by_type(items, ["chapter"]),
                "practice_evidence": self._items_by_type(items, ["exercise"]),
                "knowledge_points": self._items_by_type(items, ["knowledge_point"]),
            }
        if task_type == "diagnosis":
            return {
                "weak_knowledge_points": self._items_by_type(items, ["knowledge_point"]),
                "remedial_practice": self._items_by_type(items, ["exercise"]),
                "explanation_context": self._items_by_type(items, ["chapter", "course"]),
            }
        return {"evidence": [item.to_dict() for item in items]}

    def _items_by_type(self, items: Iterable[EvidenceItem], chunk_types: Sequence[str]) -> List[Dict[str, Any]]:
        allowed = set(chunk_types)
        return [item.to_dict() for item in items if item.chunk_type in allowed]

    def _sort_by_task_priority(self, items: List[EvidenceItem], task_type: str) -> List[EvidenceItem]:
        return sorted(
            items,
            key=lambda item: (
                TASK_TYPE_PRIORITY.get(task_type, TASK_TYPE_PRIORITY["generic"]).get(item.chunk_type, 50),
                item.score,
            ),
            reverse=True,
        )

    def _extract_query_rewrite(self, results: Sequence[RetrievalResult]) -> Dict[str, Any]:
        if not results:
            return {}
        metadata = results[0].metadata or {}
        return {
            "rewritten_query": metadata.get("rewritten_query", ""),
            "search_terms": self._split_pipe(metadata.get("search_terms")),
            "used_llm": bool(metadata.get("rewrite_used_llm", False)),
            "confidence": metadata.get("rewrite_confidence", 0.0),
            "fallback_reason": metadata.get("rewrite_fallback_reason", ""),
        }

    def _instructions(self, task_type: str) -> List[str]:
        common = [
            "只基于 evidence_items 和 sections 中的证据回答。",
            "不要编造价格、评分、证书、授课平台等证据中不存在的信息。",
            "引用资源时优先使用 title 和 source_resource_id。",
        ]
        by_task = {
            "recommend": [
                "优先推荐 primary_resources 中的课程。",
                "推荐理由应说明目标匹配、知识点覆盖、练习/章节支撑。",
            ],
            "qa": [
                "优先使用 concept_evidence 解释概念。",
                "如果证据不足，应明确说明只能基于当前 MOOC 资源给出有限解释。",
            ],
            "learning_path": [
                "按阶段组织路线：基础概念、课程学习、章节推进、练习巩固。",
                "每个阶段尽量绑定可追溯资源。",
            ],
            "diagnosis": [
                "优先定位 weak_knowledge_points 和 remedial_practice。",
                "建议应包含补救知识点和练习方向。",
            ],
        }
        return common + by_task.get(task_type, [])

    def _source_resource_id(self, result: RetrievalResult) -> str:
        metadata_id = result.metadata.get("source_resource_id") if isinstance(result.metadata, dict) else ""
        direct_id = getattr(result, "source_resource_id", "")
        return str(metadata_id or direct_id or "")

    def _compact_course_detail(self, course: Dict[str, Any]) -> Dict[str, Any]:
        if not course:
            return {}
        return {
            "id": course.get("id"),
            "title": course.get("title"),
            "description": self._truncate(course.get("description"), 600),
            "learning_notes": self._truncate(course.get("learning_notes"), 400),
            "disciplines": course.get("disciplines", []),
            "subdisciplines": course.get("subdisciplines", []),
            "knowledge_points": course.get("knowledge_points", [])[:30],
            "chapter_count": course.get("chapter_count"),
            "exercise_count": course.get("exercise_count"),
            "challenge_count": course.get("challenge_count"),
            "visits": course.get("visits"),
        }

    def _compact_resource(self, resource: Dict[str, Any]) -> Dict[str, Any]:
        if not resource:
            return {}
        return {
            "id": resource.get("id"),
            "resource_type": resource.get("resource_type") or resource.get("entity_type"),
            "title": resource.get("title"),
            "description": self._truncate(resource.get("description"), 600),
            "visits": resource.get("visits"),
            "difficulty": resource.get("difficulty"),
            "score": resource.get("score"),
        }

    def _compact_related(self, resources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        compacted = []
        for item in resources:
            compacted.append(
                {
                    "id": item.get("id") or item.get("topic_id"),
                    "type": item.get("resource_type") or item.get("entity_type"),
                    "title": item.get("title") or item.get("topic_name"),
                    "position": item.get("position"),
                    "relation": item.get("relation"),
                    "difficulty": item.get("difficulty"),
                }
            )
        return compacted

    def _compact_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        keys = [
            "chunk_type",
            "source",
            "source_resource_id",
            "source_resource_type",
            "title",
            "rule_score",
            "vector_score",
            "keyword_score",
            "llm_rerank_stage",
            "llm_rerank_error",
            "rewrite_used_llm",
            "rewrite_confidence",
        ]
        return {key: metadata.get(key) for key in keys if key in metadata}

    def _compact_profile(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        keys = ["user_id", "display_name", "learning_stage", "goal", "preferred_subjects", "preferred_resource_types", "memory_summary"]
        return {key: profile.get(key) for key in keys if key in profile}

    def _split_pipe(self, value: Any) -> List[str]:
        if not value:
            return []
        if isinstance(value, list):
            return [str(item) for item in value]
        return [item for item in str(value).split("|") if item]

    def _truncate(self, value: Any, limit: int) -> str:
        text = " ".join(str(value or "").split())
        return text[:limit]
