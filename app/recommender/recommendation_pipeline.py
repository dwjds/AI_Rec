from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from app.agent.state import AgentState
from app.rag.query_constraints import extract_query_constraints, merge_excluded_keywords
from app.recommender.filters import ResourceFilter
from app.recommender.ranker import RecommendationCandidate, RecommendationRanker, ScoredRecommendation
from app.stores.resource_store import ResourceStore

"""
实现确定性推荐 Pipeline:

RAG 候选
+ ResourceStore 数据库检索候选
-> ResourceStore 补全真实资料
-> 去重
-> 过滤
-> 排序
-> RecommendationPackage
"""
@dataclass
class RecommendationPackage:
    query: str
    recommendations: List[ScoredRecommendation]
    strategy: Dict[str, Any] = field(default_factory=dict)
    next_steps: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "recommendations": [item.to_dict() for item in self.recommendations],
            "strategy": self.strategy,
            "next_steps": self.next_steps,
        }


class RecommendationPipeline:
    """Deterministic recommendation pipeline over RAG + ResourceStore candidates."""

    def __init__(
        self,
        resource_store: ResourceStore | None = None,
        resource_filter: ResourceFilter | None = None,
        ranker: RecommendationRanker | None = None,
    ):
        self.resource_store = resource_store or ResourceStore()
        self.resource_filter = resource_filter or ResourceFilter()
        self.ranker = ranker or RecommendationRanker()

    def run(
        self,
        state: AgentState,
        top_k: int = 5,
        db_recall_limit: int = 20,
    ) -> RecommendationPackage:
        ranking_context = dict(state.memory_context.get("ranking_context") or {})
        retrieval_context = dict(state.memory_context.get("retrieval_context") or {})
        ranking_context.setdefault("preferred_subjects", retrieval_context.get("preferred_subjects") or [])
        query_constraints = extract_query_constraints(state.query)
        entity_constraints = (state.routing_decision.entities or {}) if state.routing_decision else {}
        negative_terms = merge_excluded_keywords(
            query_constraints.excluded_keywords,
            entity_constraints.get("excluded_keywords"),
            entity_constraints.get("negative_terms"),
        )
        if negative_terms:
            constraints = dict(ranking_context.get("constraints") or {})
            constraints["excluded_keywords"] = merge_excluded_keywords(constraints.get("excluded_keywords"), negative_terms)
            ranking_context["constraints"] = constraints
            ranking_context["penalize_keywords"] = merge_excluded_keywords(ranking_context.get("penalize_keywords"), negative_terms)
            ranking_context["negative_terms"] = negative_terms

        focus_terms = self._focus_terms(query_constraints.positive_query or state.query, negative_terms=negative_terms)
        if focus_terms:
            existing_required = [str(item).strip() for item in (ranking_context.get("required_keywords") or []) if str(item).strip()]
            ranking_context["required_keywords"] = existing_required or focus_terms

        candidates = self._merge_candidates(
            self._candidates_from_retrieval(state),
            self._candidates_from_db(state, db_recall_limit),
        )
        filtered = self.resource_filter.filter(candidates, ranking_context=ranking_context)
        ranked = self.ranker.rank(filtered, ranking_context=ranking_context, top_k=top_k)
        package = RecommendationPackage(
            query=state.query,
            recommendations=ranked,
            strategy={
                "used_memory": bool(state.memory_context),
                "recall_sources": ["rag", "resource_store"],
                "candidate_count": len(candidates),
                "filtered_count": len(filtered),
                "filters_applied": self.resource_filter.last_filters_applied,
                "ranking_features": [
                    "semantic_score",
                    "required_keywords",
                    "learning_stage",
                    "preferred_subjects",
                    "preferred_resource_types",
                    "feedback",
                    "resource_popularity",
                    "repeat_penalty",
                ],
            },
            next_steps=self._next_steps(ranked),
        )
        state.add_metadata("recommendation_package", package.to_dict())
        return package

    def _candidates_from_retrieval(self, state: AgentState) -> List[RecommendationCandidate]:
        candidates: List[RecommendationCandidate] = []
        for result in state.retrieval_results:
            resource_id = self._recommendation_resource_id(result)
            if not resource_id:
                continue
            candidate = self._hydrate(resource_id, "course" if resource_id.startswith("mooper:course:") else result.chunk_type)
            if candidate is None:
                candidate = RecommendationCandidate(
                    resource_id=resource_id,
                    resource_type=result.chunk_type,
                    title=result.title,
                    description=result.content[:500],
                    retrieval={},
                    source="rag_unhydrated",
                )
            candidate.retrieval = {
                "chunk_id": result.chunk_id,
                "chunk_type": result.chunk_type,
                "score": result.score,
                "distance": result.distance,
                "evidence_content": result.content[:500],
                "matched_chunk_type": result.chunk_type,
                "matched_chunk_title": result.title,
            }
            candidates.append(candidate)
        return candidates

    def _candidates_from_db(self, state: AgentState, limit: int) -> List[RecommendationCandidate]:
        base_query = extract_query_constraints(state.query).positive_query or state.query
        queries = [base_query]
        retrieval_context = state.memory_context.get("retrieval_context") or {}
        for subject in retrieval_context.get("preferred_subjects") or []:
            if subject and subject not in queries:
                queries.append(str(subject))
        if state.routing_decision:
            for subject in (state.routing_decision.entities or {}).get("subjects") or []:
                if subject and subject not in queries:
                    queries.append(str(subject))

        candidates: List[RecommendationCandidate] = []
        expanded_queries = []
        for query in queries[:4]:
            for variant in self._query_variants(query):
                if variant and variant not in expanded_queries:
                    expanded_queries.append(variant)
        for query in expanded_queries[:8]:
            for course in self.resource_store.search_courses(query=query, limit=limit):
                candidate = self._candidate_from_course(course, retrieval={"score": 0.35, "source": "resource_store_keyword"})
                if candidate:
                    candidates.append(candidate)
        return candidates

    def _recommendation_resource_id(self, result: Any) -> str:
        metadata = result.metadata or {}
        if result.chunk_type == "course":
            return str(metadata.get("source_resource_id") or "")
        course_id = str(metadata.get("course_id") or "")
        if course_id:
            return course_id
        return str(metadata.get("source_resource_id") or "")

    def _hydrate(self, resource_id: str, chunk_type: str) -> Optional[RecommendationCandidate]:
        if chunk_type == "course":
            course = self.resource_store.get_course_detail(resource_id)
            return self._candidate_from_course(course or {})
        resource = self.resource_store.get_resource(resource_id)
        if not resource:
            return None
        return self._candidate_from_resource(resource)

    def _candidate_from_course(
        self,
        course: Dict[str, Any],
        retrieval: Optional[Dict[str, Any]] = None,
    ) -> Optional[RecommendationCandidate]:
        if not course:
            return None
        return RecommendationCandidate(
            resource_id=str(course.get("id") or ""),
            resource_type="course",
            title=self._clean_title(course.get("title")),
            description=str(course.get("description") or course.get("learning_notes") or ""),
            knowledge_points=self._knowledge_points(course.get("knowledge_points") or []),
            chapters=list(course.get("chapters") or []),
            exercises=list(course.get("exercises") or []),
            visits=self._int(course.get("visits")),
            retrieval=retrieval or {},
            source="resource_store",
            raw_resource=course,
        )

    def _candidate_from_resource(self, resource: Dict[str, Any]) -> RecommendationCandidate:
        return RecommendationCandidate(
            resource_id=str(resource.get("id") or ""),
            resource_type=str(resource.get("resource_type") or resource.get("entity_type") or ""),
            title=self._clean_title(resource.get("title")),
            description=str(resource.get("description") or resource.get("learning_notes") or ""),
            knowledge_points=[],
            chapters=[],
            exercises=[],
            visits=self._int(resource.get("visits")),
            retrieval={},
            source="resource_store",
            raw_resource=resource,
        )

    def _merge_candidates(
        self,
        rag_candidates: List[RecommendationCandidate],
        db_candidates: List[RecommendationCandidate],
    ) -> List[RecommendationCandidate]:
        merged: Dict[str, RecommendationCandidate] = {}
        for candidate in rag_candidates + db_candidates:
            if not candidate.resource_id:
                continue
            existing = merged.get(candidate.resource_id)
            if existing is None:
                merged[candidate.resource_id] = candidate
                continue
            if float(candidate.retrieval.get("score") or 0.0) > float(existing.retrieval.get("score") or 0.0):
                existing.retrieval = candidate.retrieval
            existing.chapters = existing.chapters or candidate.chapters
            existing.exercises = existing.exercises or candidate.exercises
            existing.knowledge_points = existing.knowledge_points or candidate.knowledge_points
        return list(merged.values())

    def _next_steps(self, ranked: List[ScoredRecommendation]) -> List[str]:
        if not ranked:
            return ["补充更具体的学习方向或当前基础后重新推荐。"]
        first = ranked[0].candidate.title
        return [
            "先从排名第一的资源《{0}》开始。".format(first),
            "学习后记录掌握情况和反馈，下一轮推荐会据此调整。",
        ]

    def _knowledge_points(self, rows: List[Any]) -> List[Dict[str, Any]]:
        points = []
        for row in rows:
            if isinstance(row, dict):
                points.append({"id": row.get("id") or row.get("topic_id"), "title": row.get("title") or row.get("topic_name")})
            else:
                points.append({"id": "", "title": str(row)})
        return points

    def _int(self, value: Any) -> int:
        try:
            return int(str(value or "0"))
        except ValueError:
            return 0

    def _clean_title(self, value: Any) -> str:
        return str(value or "").replace("\ufeff", "").strip()

    def _query_variants(self, query: str) -> List[str]:
        text = str(query or "").strip()
        variants = [text] if text else []
        for term in self._focus_terms(text):
            if term not in variants:
                variants.append(term)
        compact = text
        for word in ["推荐", "有关", "相关", "课程", "教程", "资源", "学习路线", "路线", "路径", "学习", "入门", "进阶", "的", "一下", "帮我"]:
            compact = compact.replace(word, "")
        compact = compact.strip()
        if compact and compact not in variants:
            variants.append(compact)
        return variants[:6]

    def _focus_terms(self, query: str, negative_terms: Optional[List[str]] = None) -> List[str]:
        text = str(query or "")
        negatives = {term.lower() for term in (negative_terms if negative_terms is not None else self._negative_terms(text))}
        known_terms = [
            "强化学习",
            "机器学习",
            "深度学习",
            "自然语言处理",
            "计算机视觉",
            "程序设计",
            "算法竞赛",
            "数据结构",
            "算法设计",
            "算法",
            "人工智能",
            "Python",
            "Java",
            "数据库",
        ]
        hits = [term for term in known_terms if term.lower() in text.lower() and term.lower() not in negatives]
        if hits:
            return hits[:4]
        cleaned = text
        for word in ["推荐", "有关", "相关", "课程", "教程", "资源", "学习路线", "路线", "路径", "学习", "入门", "进阶", "的", "一下", "帮我"]:
            cleaned = cleaned.replace(word, " ")
        return [item.strip() for item in cleaned.replace("，", " ").replace(",", " ").split() if len(item.strip()) >= 2][:3]

    def _negative_terms(self, query: str) -> List[str]:
        return extract_query_constraints(query).excluded_keywords
