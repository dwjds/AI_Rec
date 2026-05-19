from __future__ import annotations

from typing import Any, Dict, List

from app.recommender.ranker import RecommendationCandidate

"实现推荐硬过滤：用户负反馈资源、排除资源类型、排除关键词、近期重复资源。"
class ResourceFilter:
    """Hard-filter hydrated recommendation candidates."""

    def __init__(self):
        self.last_filters_applied: List[str] = []

    def filter(
        self,
        candidates: List[RecommendationCandidate],
        ranking_context: Dict[str, Any],
        min_candidates: int = 3,
    ) -> List[RecommendationCandidate]:
        self.last_filters_applied = []
        filtered = self._remove_low_quality_titles(candidates)
        filtered = self._remove_disliked(filtered, ranking_context)
        filtered = self._remove_excluded_types(filtered, ranking_context)
        filtered = self._remove_excluded_keywords(filtered, ranking_context)
        filtered = self._prefer_required_keywords(filtered, ranking_context, min_candidates=min_candidates)

        repeated_removed = self._remove_repeated(filtered, ranking_context)
        if len(repeated_removed) >= min_candidates:
            filtered = repeated_removed
            self.last_filters_applied.append("avoid_repeating_resource_ids")

        return filtered

    def _remove_low_quality_titles(self, candidates: List[RecommendationCandidate]) -> List[RecommendationCandidate]:
        filtered = []
        for candidate in candidates:
            title = str(candidate.title or "").strip()
            if not title:
                continue
            if not any(char.isalpha() for char in title):
                continue
            filtered.append(candidate)
        if len(filtered) != len(candidates):
            self.last_filters_applied.append("low_quality_title")
        return filtered

    def _remove_disliked(
        self,
        candidates: List[RecommendationCandidate],
        ranking_context: Dict[str, Any],
    ) -> List[RecommendationCandidate]:
        disliked = set(ranking_context.get("disliked_resource_ids") or [])
        excluded = set((ranking_context.get("recommendation_adjustments") or {}).get("exclude_resource_ids") or [])
        blocked = disliked | excluded
        if not blocked:
            return candidates
        self.last_filters_applied.append("disliked_resource_ids")
        return [candidate for candidate in candidates if candidate.resource_id not in blocked]

    def _remove_excluded_types(
        self,
        candidates: List[RecommendationCandidate],
        ranking_context: Dict[str, Any],
    ) -> List[RecommendationCandidate]:
        constraints = ranking_context.get("constraints") or {}
        excluded_types = set(constraints.get("excluded_resource_types") or [])
        if not excluded_types:
            return candidates
        self.last_filters_applied.append("excluded_resource_types")
        return [candidate for candidate in candidates if candidate.resource_type not in excluded_types]

    def _remove_excluded_keywords(
        self,
        candidates: List[RecommendationCandidate],
        ranking_context: Dict[str, Any],
    ) -> List[RecommendationCandidate]:
        constraints = ranking_context.get("constraints") or {}
        excluded_keywords = list(constraints.get("excluded_keywords") or [])
        if not excluded_keywords:
            return candidates
        self.last_filters_applied.append("excluded_keywords")
        return [
            candidate
            for candidate in candidates
            if not self._contains_any(candidate, excluded_keywords)
        ]

    def _prefer_required_keywords(
        self,
        candidates: List[RecommendationCandidate],
        ranking_context: Dict[str, Any],
        min_candidates: int,
    ) -> List[RecommendationCandidate]:
        keywords = [str(item).strip() for item in (ranking_context.get("required_keywords") or []) if str(item).strip()]
        if not keywords:
            return candidates
        matched = [candidate for candidate in candidates if self._contains_learning_target(candidate, keywords)]
        if len(matched) >= 1:
            self.last_filters_applied.append("required_keywords")
            return matched
        self.last_filters_applied.append("required_keywords_no_match")
        return []

    def _remove_repeated(
        self,
        candidates: List[RecommendationCandidate],
        ranking_context: Dict[str, Any],
    ) -> List[RecommendationCandidate]:
        repeated = set(ranking_context.get("avoid_repeating_resource_ids") or [])
        if not repeated:
            return candidates
        return [candidate for candidate in candidates if candidate.resource_id not in repeated]

    def _contains_any(self, candidate: RecommendationCandidate, keywords: List[str]) -> bool:
        text = self._text(candidate)
        return any(str(keyword).lower() in text for keyword in keywords)

    def _contains_learning_target(self, candidate: RecommendationCandidate, keywords: List[str]) -> bool:
        text = self._learning_target_text(candidate)
        return any(str(keyword).lower() in text for keyword in keywords)

    def _learning_target_text(self, candidate: RecommendationCandidate) -> str:
        knowledge = " ".join(str(item.get("title") or item.get("knowledge_point_id") or item) for item in candidate.knowledge_points)
        chapters = " ".join(str(item.get("title") or "") for item in candidate.chapters)
        exercises = " ".join(str(item.get("title") or "") for item in candidate.exercises)
        retrieval = " ".join(str(candidate.retrieval.get(key) or "") for key in ["matched_chunk_title", "evidence_content"])
        return " ".join([candidate.title, knowledge, chapters, exercises, retrieval]).lower()

    def _text(self, candidate: RecommendationCandidate) -> str:
        knowledge = " ".join(str(item.get("title") or item.get("knowledge_point_id") or item) for item in candidate.knowledge_points)
        chapters = " ".join(str(item.get("title") or "") for item in candidate.chapters)
        exercises = " ".join(str(item.get("title") or "") for item in candidate.exercises)
        retrieval = " ".join(str(candidate.retrieval.get(key) or "") for key in ["matched_chunk_title", "evidence_content"])
        return " ".join([candidate.title, candidate.description, knowledge, chapters, exercises, retrieval]).lower()
