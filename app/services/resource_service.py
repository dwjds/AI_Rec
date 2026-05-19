from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from app.rag.evidence_builder import EvidenceBuilder, EvidencePackage
from app.rag.retriever import RagRetriever, RetrievalResult
from app.stores.resource_store import ResourceStore

"封装资源搜索、课程详情、RAG 检索、Evidence 构建。"
class ResourceService:
    """Application-facing resource service over MOOPer and RAG retrieval."""

    def __init__(
        self,
        resource_store: ResourceStore | None = None,
        retriever: RagRetriever | None = None,
        evidence_builder: EvidenceBuilder | None = None,
    ):
        self.resource_store = resource_store or ResourceStore()
        self._retriever = retriever
        self._evidence_builder = evidence_builder

    @property
    def retriever(self) -> RagRetriever:
        if self._retriever is None:
            self._retriever = RagRetriever()
        return self._retriever

    @property
    def evidence_builder(self) -> EvidenceBuilder:
        if self._evidence_builder is None:
            self._evidence_builder = EvidenceBuilder(resource_store=self.resource_store)
        return self._evidence_builder

    def search_courses(
        self,
        query: str,
        limit: int = 10,
        discipline: str | None = None,
        subdiscipline: str | None = None,
    ) -> List[Dict[str, Any]]:
        return self.resource_store.search_courses(
            query=query,
            limit=limit,
            discipline=discipline,
            subdiscipline=subdiscipline,
        )

    def search_resources(
        self,
        query: str,
        resource_types: Optional[Sequence[str]] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        return self.resource_store.search_resources(query=query, resource_types=resource_types, limit=limit)

    def search_knowledge_points(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        return self.resource_store.search_knowledge_points(query=query, limit=limit)

    def get_course_detail(self, course_id: str) -> Optional[Dict[str, Any]]:
        return self.resource_store.get_course_detail(course_id)

    def get_resource(self, resource_id: str) -> Optional[Dict[str, Any]]:
        return self.resource_store.get_resource(resource_id)

    def retrieve(
        self,
        query: str,
        task_type: str = "generic",
        top_k: int = 8,
        chunk_types: Optional[Sequence[str]] = None,
        user_profile: Optional[Dict[str, Any]] = None,
        knowledge_state: Optional[List[Dict[str, Any]]] = None,
        use_llm_rerank: bool = True,
    ) -> List[RetrievalResult]:
        return self.retriever.retrieve(
            query=query,
            top_k=top_k,
            chunk_types=chunk_types,
            task_type=task_type,
            user_profile=user_profile or {},
            knowledge_state=knowledge_state or [],
            use_llm_rerank=use_llm_rerank,
        )

    def build_evidence(
        self,
        query: str,
        task_type: str,
        retrieval_results: Sequence[RetrievalResult],
        user_profile: Optional[Dict[str, Any]] = None,
        knowledge_state: Optional[List[Dict[str, Any]]] = None,
        max_items: int = 12,
    ) -> EvidencePackage:
        return self.evidence_builder.build(
            query=query,
            task_type=task_type,
            retrieval_results=retrieval_results,
            user_profile=user_profile or {},
            knowledge_state=knowledge_state or [],
            max_items=max_items,
        )
