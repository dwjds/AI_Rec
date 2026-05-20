from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from app.core.config import settings
from app.rag.query_constraints import contains_excluded_keyword, extract_query_constraints, merge_excluded_keywords, strip_negative_constraints
from app.rag.query_rewriter import QueryRewriteResult, QueryRewriter
from app.rag.reranker import HybridReranker, RerankContext, extract_query_terms, keyword_score
from app.rag.vector_store import ChromaVectorStore


@dataclass
class RetrievalResult:
    chunk_id: str
    chunk_type: str
    title: str
    content: str
    metadata: Dict[str, Any]
    distance: float
    rank_score: Optional[float] = None

    @property
    def score(self) -> float:
        if self.rank_score is not None:
            return self.rank_score
        return 1.0 / (1.0 + max(0.0, float(self.distance)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "chunk_type": self.chunk_type,
            "title": self.title,
            "content": self.content,
            "metadata": self.metadata,
            "distance": self.distance,
            "score": self.score,
        }


class RagRetriever:
    """RAG retrieval entrypoint over the MOOPer Chroma collection."""

    def __init__(
        self,
        vector_store: Optional[ChromaVectorStore] = None,
        query_rewriter: Optional[QueryRewriter] = None,
        reranker: Optional[HybridReranker] = None,
    ):
        self._vector_store = vector_store
        self.query_rewriter = query_rewriter or QueryRewriter()
        self.reranker = reranker or HybridReranker()

    @property
    def vector_store(self) -> ChromaVectorStore:
        if self._vector_store is None:
            self._vector_store = ChromaVectorStore()
        return self._vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        chunk_types: Optional[Sequence[str]] = None,
        use_query_rewrite: bool = True,
        use_llm_rewrite: bool = True,
        candidate_k: Optional[int] = None,
        task_type: str = "generic",
        user_profile: Optional[Dict[str, Any]] = None,
        knowledge_state: Optional[List[Dict[str, Any]]] = None,
        use_llm_rerank: bool = True,
        rule_top_k: int = 20,
    ) -> List[RetrievalResult]:
        rewrite = self.query_rewriter.rewrite(query, use_llm=use_llm_rewrite) if use_query_rewrite else None
        query_constraints = extract_query_constraints(query)
        retrieval_query = strip_negative_constraints(rewrite.rewritten_query if rewrite else query) or query_constraints.positive_query or query
        retrieval_chunk_types = list(chunk_types or (rewrite.chunk_types if rewrite else [])) or None
        excluded_keywords = merge_excluded_keywords(
            query_constraints.excluded_keywords,
            (rewrite.filters or {}).get("excluded_keywords") if rewrite else [],
            self._profile_excluded_keywords(user_profile or {}),
        )
        candidate_limit = candidate_k or max(top_k * 10, top_k)
        vector_error = ""
        try:
            vector_rows = self.vector_store.query(
                query=retrieval_query,
                top_k=candidate_limit,
                where=self._where(retrieval_chunk_types),
            )
        except Exception as exc:
            vector_rows = []
            vector_error = str(exc)
        vector_results = [self._to_result(row) for row in vector_rows]
        keyword_results = self._keyword_retrieve(
            query=retrieval_query,
            chunk_types=retrieval_chunk_types,
            limit=candidate_limit,
        )
        candidates = self._filter_excluded(self._merge_candidates(vector_results, keyword_results), excluded_keywords)
        if vector_error:
            for item in candidates:
                item.metadata["vector_fallback_reason"] = vector_error
        reranked = self.reranker.rerank(
            candidates,
            context=RerankContext(
                query=retrieval_query,
                task_type=task_type,
                user_profile=user_profile or {},
                knowledge_state=knowledge_state or [],
                top_k=top_k,
                rule_top_k=rule_top_k,
                use_llm_rerank=use_llm_rerank,
                excluded_keywords=excluded_keywords,
            ),
        )
        self._attach_rewrite_info(reranked, rewrite)
        return reranked[:top_k]

    def retrieve_courses(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        return self.retrieve(query, top_k=top_k, chunk_types=["course"])

    def retrieve_chapters(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        return self.retrieve(query, top_k=top_k, chunk_types=["chapter"])

    def retrieve_exercises(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        return self.retrieve(query, top_k=top_k, chunk_types=["exercise"])

    def retrieve_knowledge_points(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        return self.retrieve(query, top_k=top_k, chunk_types=["knowledge_point"])

    def _where(self, chunk_types: Optional[Sequence[str]]) -> Optional[Dict[str, Any]]:
        values = [item for item in (chunk_types or []) if item]
        if not values:
            return None
        if len(values) == 1:
            return {"chunk_type": values[0]}
        return {"chunk_type": {"$in": values}}

    def _to_result(self, row: Dict[str, Any]) -> RetrievalResult:
        metadata = dict(row.get("metadata") or {})
        return RetrievalResult(
            chunk_id=str(row.get("chunk_id") or ""),
            chunk_type=str(metadata.get("chunk_type") or ""),
            title=str(metadata.get("title") or ""),
            content=str(row.get("content") or ""),
            metadata=metadata,
            distance=float(row.get("distance") or 0.0),
        )

    def _attach_rewrite_info(self, results: List[RetrievalResult], rewrite: Optional[QueryRewriteResult]) -> None:
        if rewrite is None:
            return
        rewrite_info = {
            "rewritten_query": rewrite.rewritten_query,
            "search_terms": "|".join(rewrite.search_terms),
            "rewrite_used_llm": rewrite.used_llm,
            "rewrite_confidence": rewrite.confidence,
            "rewrite_fallback_reason": rewrite.fallback_reason or "",
        }
        for item in results:
            item.metadata.update(rewrite_info)

    def _keyword_retrieve(
        self,
        query: str,
        chunk_types: Optional[Sequence[str]],
        limit: int,
    ) -> List[RetrievalResult]:
        if not settings.chunks_path.exists():
            return []
        allowed_types = set(chunk_types or [])
        terms = extract_query_terms(query)
        if not terms:
            return []

        candidates: List[RetrievalResult] = []
        with settings.chunks_path.open("r", encoding="utf-8") as fp:
            for line in fp:
                if not line.strip():
                    continue
                chunk = json.loads(line)
                chunk_type = str(chunk.get("chunk_type") or "")
                if allowed_types and chunk_type not in allowed_types:
                    continue
                keyword_value = keyword_score(
                    terms=terms,
                    title=str(chunk.get("title") or ""),
                    content=str(chunk.get("content") or ""),
                    metadata=dict(chunk.get("metadata") or {}),
                )
                if keyword_value <= 0:
                    continue
                candidates.append(
                    RetrievalResult(
                        chunk_id=str(chunk.get("id") or ""),
                        chunk_type=chunk_type,
                        title=str(chunk.get("title") or ""),
                        content=str(chunk.get("content") or ""),
                        metadata=self._keyword_metadata(chunk, keyword_value),
                        distance=999.0,
                    )
                )
        candidates.sort(key=lambda item: item.metadata.get("keyword_score", 0.0), reverse=True)
        return candidates[:limit]

    def _merge_candidates(
        self,
        vector_results: List[RetrievalResult],
        keyword_results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        merged: Dict[str, RetrievalResult] = {}

        for item in vector_results:
            vector_score = 1.0 / (1.0 + max(0.0, item.distance))
            item.metadata["vector_score"] = vector_score
            merged[item.chunk_id] = item

        for item in keyword_results:
            existing = merged.get(item.chunk_id)
            keyword_score = float(item.metadata.get("keyword_score") or 0.0)
            if existing is None:
                merged[item.chunk_id] = item
            else:
                existing.metadata["keyword_score"] = max(float(existing.metadata.get("keyword_score") or 0.0), keyword_score)
        return list(merged.values())

    def _keyword_metadata(self, chunk: Dict[str, Any], keyword_score: float) -> Dict[str, Any]:
        metadata = dict(chunk.get("metadata") or {})
        metadata.update(
            {
                "chunk_type": chunk.get("chunk_type", ""),
                "source": chunk.get("source", ""),
                "source_resource_id": chunk.get("source_resource_id", ""),
                "source_resource_type": chunk.get("source_resource_type", ""),
                "title": chunk.get("title", ""),
                "keyword_score": keyword_score,
            }
        )
        return metadata

    def _filter_excluded(self, candidates: List[RetrievalResult], excluded_keywords: List[str]) -> List[RetrievalResult]:
        if not excluded_keywords:
            return candidates
        filtered: List[RetrievalResult] = []
        for item in candidates:
            text = " ".join(
                [
                    item.title,
                    item.content,
                    " ".join(str(value) for value in item.metadata.values()),
                ]
            )
            if contains_excluded_keyword(text, excluded_keywords):
                item.metadata["filtered_reason"] = "excluded_keywords"
                item.metadata["excluded_keywords"] = "|".join(excluded_keywords)
                continue
            filtered.append(item)
        return filtered

    def _profile_excluded_keywords(self, user_profile: Dict[str, Any]) -> List[str]:
        constraints = user_profile.get("constraints") if isinstance(user_profile.get("constraints"), dict) else {}
        memory_ranking = user_profile.get("memory_ranking") if isinstance(user_profile.get("memory_ranking"), dict) else {}
        memory_constraints = memory_ranking.get("constraints") if isinstance(memory_ranking.get("constraints"), dict) else {}
        memory_retrieval = user_profile.get("memory_retrieval") if isinstance(user_profile.get("memory_retrieval"), dict) else {}
        return merge_excluded_keywords(
            constraints.get("excluded_keywords"),
            memory_constraints.get("excluded_keywords"),
            memory_retrieval.get("excluded_keywords"),
            memory_ranking.get("negative_terms"),
            memory_ranking.get("penalize_keywords"),
        )
