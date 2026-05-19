from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

"""
实现 RecommendationCandidate、ScoredRecommendation 和业务排序规则，
融合：
RAG 语义分
用户学习阶段
偏好资源类型
偏好方向
弱知识点
反馈 boost / penalty
访问量
重复推荐惩罚
"""
@dataclass
class RecommendationCandidate:
    resource_id: str
    resource_type: str
    title: str
    description: str = ""
    knowledge_points: List[Dict[str, Any]] = field(default_factory=list)
    chapters: List[Dict[str, Any]] = field(default_factory=list)
    exercises: List[Dict[str, Any]] = field(default_factory=list)
    visits: int = 0
    retrieval: Dict[str, Any] = field(default_factory=dict)
    source: str = "resource_store"
    raw_resource: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScoredRecommendation:
    candidate: RecommendationCandidate
    score: float
    reasons: List[str] = field(default_factory=list)
    evidence_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = self.candidate.to_dict()
        data.update(
            {
                "score": round(float(self.score), 4),
                "reasons": self.reasons,
                "evidence_ids": self.evidence_ids,
            }
        )
        return data


class RecommendationRanker:
    """Business-level recommendation ranker over hydrated resources."""

    def rank(
        self,
        candidates: List[RecommendationCandidate],
        ranking_context: Dict[str, Any],
        top_k: int = 5,
    ) -> List[ScoredRecommendation]:
        scored = [self._score_candidate(candidate, ranking_context) for candidate in candidates]
        scored.sort(key=lambda item: item.score, reverse=True)
        return self._dedupe_scored(scored)[: max(1, int(top_k))]

    def _score_candidate(
        self,
        candidate: RecommendationCandidate,
        ranking_context: Dict[str, Any],
    ) -> ScoredRecommendation:
        reasons: List[str] = []
        score = 0.0

        retrieval_score = float(candidate.retrieval.get("score") or candidate.retrieval.get("semantic_score") or 0.0)
        if retrieval_score > 0:
            score += retrieval_score * 0.45
            reasons.append("语义召回匹配用户问题")

        required_hits = self._match_learning_target_keywords(candidate, ranking_context.get("required_keywords") or [])
        if required_hits:
            score += min(0.55, 0.22 * len(required_hits))
            reasons.append("精确匹配学习主题：{0}".format("、".join(required_hits[:3])))

        visits_score = min(max(candidate.visits, 0) / 100000.0, 1.0)
        if visits_score > 0:
            score += visits_score * 0.08
            reasons.append("资源访问量较高")

        preferred_types = set(ranking_context.get("preferred_resource_types") or [])
        if candidate.resource_type in preferred_types:
            score += 0.12
            reasons.append("匹配用户偏好的资源类型")

        subject_hits = self._match_subjects(candidate, ranking_context.get("preferred_subjects") or [])
        if subject_hits:
            score += min(0.18, 0.06 * len(subject_hits))
            reasons.append("匹配用户偏好方向：{0}".format("、".join(subject_hits[:3])))

        weak_hits = self._match_weak_points(candidate, ranking_context.get("weak_knowledge_points") or [])
        if weak_hits:
            score += min(0.14, 0.05 * len(weak_hits))
            reasons.append("覆盖待补强知识点")

        stage_reason, stage_score = self._stage_score(candidate, ranking_context.get("learning_stage") or "")
        if stage_score:
            score += stage_score
            reasons.append(stage_reason)

        boost_keywords = ranking_context.get("boost_keywords") or []
        boost_hits = self._match_keywords(candidate, boost_keywords)
        if boost_hits:
            score += min(0.1, 0.03 * len(boost_hits))
            reasons.append("符合近期正向反馈偏好")

        penalize_keywords = ranking_context.get("penalize_keywords") or []
        if self._match_keywords(candidate, penalize_keywords):
            score -= 0.18
            reasons.append("包含近期负向反馈关键词，已降权")

        if candidate.resource_id in set(ranking_context.get("avoid_repeating_resource_ids") or []):
            score -= 0.12
            reasons.append("近期已推荐或完成，降低重复推荐权重")

        if candidate.resource_id in set(ranking_context.get("disliked_resource_ids") or []):
            score -= 0.5
            reasons.append("用户曾明确负反馈，强降权")

        evidence_ids = []
        if candidate.retrieval.get("chunk_id"):
            evidence_ids.append(str(candidate.retrieval["chunk_id"]))

        return ScoredRecommendation(
            candidate=candidate,
            score=max(0.0, score),
            reasons=reasons[:6],
            evidence_ids=evidence_ids,
        )

    def _text(self, candidate: RecommendationCandidate) -> str:
        kp = " ".join(str(item.get("title") or item.get("knowledge_point_id") or item) for item in candidate.knowledge_points)
        chapters = " ".join(str(item.get("title") or "") for item in candidate.chapters)
        exercises = " ".join(str(item.get("title") or "") for item in candidate.exercises)
        retrieval = " ".join(str(candidate.retrieval.get(key) or "") for key in ["matched_chunk_title", "evidence_content"])
        return " ".join([candidate.title, candidate.description, kp, chapters, exercises, retrieval]).lower()

    def _match_subjects(self, candidate: RecommendationCandidate, subjects: List[str]) -> List[str]:
        text = self._text(candidate)
        return [str(subject) for subject in subjects if str(subject).lower() in text]

    def _match_weak_points(self, candidate: RecommendationCandidate, weak_points: List[Dict[str, Any]]) -> List[str]:
        text = self._text(candidate)
        hits = []
        for point in weak_points:
            point_id = str(point.get("knowledge_point_id") or "")
            title = str(point.get("title") or "")
            if point_id and point_id.lower() in text:
                hits.append(point_id)
            elif title and title.lower() in text:
                hits.append(title)
        return hits

    def _match_keywords(self, candidate: RecommendationCandidate, keywords: List[str]) -> List[str]:
        text = self._text(candidate)
        return [str(keyword) for keyword in keywords if str(keyword).lower() in text]

    def _match_learning_target_keywords(self, candidate: RecommendationCandidate, keywords: List[str]) -> List[str]:
        text = self._learning_target_text(candidate)
        return [str(keyword) for keyword in keywords if str(keyword).lower() in text]

    def _learning_target_text(self, candidate: RecommendationCandidate) -> str:
        kp = " ".join(str(item.get("title") or item.get("knowledge_point_id") or item) for item in candidate.knowledge_points)
        chapters = " ".join(str(item.get("title") or "") for item in candidate.chapters)
        exercises = " ".join(str(item.get("title") or "") for item in candidate.exercises)
        retrieval = " ".join(str(candidate.retrieval.get(key) or "") for key in ["matched_chunk_title", "evidence_content"])
        return " ".join([candidate.title, kp, chapters, exercises, retrieval]).lower()

    def _stage_score(self, candidate: RecommendationCandidate, learning_stage: str) -> tuple[str, float]:
        if not learning_stage:
            return "", 0.0
        text = self._text(candidate)
        if learning_stage == "beginner" and any(word in text for word in ["入门", "基础", "导论", "初级"]):
            return "适合从基础阶段开始", 0.12
        if learning_stage == "intermediate" and any(word in text for word in ["进阶", "实践", "项目", "应用"]):
            return "适合已有基础后继续提升", 0.1
        if learning_stage == "advanced" and any(word in text for word in ["高级", "深入", "专题", "研究"]):
            return "适合深入学习阶段", 0.1
        return "", 0.0

    def _dedupe_scored(self, items: List[ScoredRecommendation]) -> List[ScoredRecommendation]:
        deduped: List[ScoredRecommendation] = []
        seen_ids = set()
        seen_titles = set()
        for item in items:
            resource_id = str(item.candidate.resource_id or "").strip()
            title_key = self._normalize_title(item.candidate.title)
            if resource_id and resource_id in seen_ids:
                continue
            if title_key and title_key in seen_titles:
                continue
            if resource_id:
                seen_ids.add(resource_id)
            if title_key:
                seen_titles.add(title_key)
            deduped.append(item)
        return deduped

    def _normalize_title(self, title: str) -> str:
        text = str(title or "").lower().strip()
        for char in [" ", "\t", "\n", "　", "-", "—", "_"]:
            text = text.replace(char, "")
        return text
