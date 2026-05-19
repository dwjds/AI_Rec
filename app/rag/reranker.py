from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from app.core.config import create_llm_client, settings


DEFAULT_TASK_TYPE = "generic"


TYPE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "recommend": {
        "course": 1.30,
        "chapter": 0.90,
        "exercise": 0.85,
        "knowledge_point": 0.75,
    },
    "qa": {
        "knowledge_point": 1.30,
        "chapter": 1.10,
        "course": 0.85,
        "exercise": 0.75,
    },
    "learning_path": {
        "course": 1.20,
        "chapter": 1.10,
        "exercise": 1.00,
        "knowledge_point": 1.00,
    },
    "diagnosis": {
        "knowledge_point": 1.20,
        "exercise": 1.20,
        "chapter": 1.00,
        "course": 0.85,
    },
    DEFAULT_TASK_TYPE: {
        "course": 1.00,
        "chapter": 1.00,
        "exercise": 1.00,
        "knowledge_point": 1.00,
    },
}


@dataclass
class RerankContext:
    query: str
    task_type: str = DEFAULT_TASK_TYPE
    user_profile: Dict[str, Any] = field(default_factory=dict)
    knowledge_state: List[Dict[str, Any]] = field(default_factory=list)
    top_k: int = 5
    rule_top_k: int = 20
    use_llm_rerank: bool = True


class RuleBasedReranker:
    """Deterministic coarse reranker for retrieved RAG candidates."""

    def rerank(
        self,
        candidates: Sequence[Any],
        context: RerankContext,
        limit: Optional[int] = None,
    ) -> List[Any]:
        merged = self._dedupe_by_id(candidates)
        scored: List[Any] = []
        terms = extract_query_terms(context.query)
        for item in merged:
            if not self._allowed_by_profile(item, context.user_profile):
                continue
            components = self._score_components(item, terms, context)
            final_score = (
                components["vector_score"] * 0.45
                + components["keyword_score"] * 0.30
                + components["title_score"] * 0.15
                + components["type_score"] * 0.05
                + components["popularity_score"] * 0.03
                + components["profile_score"] * 0.01
                + components["knowledge_score"] * 0.01
            )
            self._set_rank_score(item, final_score)
            self._metadata(item).update(
                {
                    "rule_score": final_score,
                    "rerank_stage": "rule",
                    **components,
                }
            )
            scored.append(item)

        ranked = sorted(scored, key=lambda value: self._score(value), reverse=True)
        deduped = self._dedupe_for_display(ranked)
        return deduped[: limit or context.rule_top_k]

    def keyword_score(self, terms: List[str], item: Any) -> float:
        return keyword_score(
            terms=terms,
            title=str(getattr(item, "title", "")),
            content=str(getattr(item, "content", "")),
            metadata=self._metadata(item),
        )

    def _score_components(self, item: Any, terms: List[str], context: RerankContext) -> Dict[str, float]:
        metadata = self._metadata(item)
        vector_score = float(metadata.get("vector_score") or self._vector_score(item))
        keyword_value = float(metadata.get("keyword_score") or self.keyword_score(terms, item))
        title_value = title_match_score(terms, str(getattr(item, "title", "")))
        type_value = self._type_weight(str(getattr(item, "chunk_type", "")), context.task_type)
        popularity_value = popularity_score(metadata)
        profile_value = profile_match_score(item, context.user_profile)
        knowledge_value = knowledge_match_score(item, context.knowledge_state)
        return {
            "vector_score": vector_score,
            "keyword_score": keyword_value,
            "title_score": title_value,
            "type_score": type_value,
            "popularity_score": popularity_value,
            "profile_score": profile_value,
            "knowledge_score": knowledge_value,
        }

    def _type_weight(self, chunk_type: str, task_type: str) -> float:
        return TYPE_WEIGHTS.get(task_type, TYPE_WEIGHTS[DEFAULT_TASK_TYPE]).get(chunk_type, 1.0)

    def _allowed_by_profile(self, item: Any, user_profile: Dict[str, Any]) -> bool:
        constraints = user_profile.get("constraints") if isinstance(user_profile, dict) else {}
        if not isinstance(constraints, dict):
            return True

        excluded_types = set(_as_list(constraints.get("excluded_resource_types")))
        if str(getattr(item, "chunk_type", "")) in excluded_types:
            return False

        text = _candidate_text(item).lower()
        for keyword in _as_list(constraints.get("excluded_keywords")):
            if keyword.lower() and keyword.lower() in text:
                return False
        return True

    def _dedupe_by_id(self, candidates: Sequence[Any]) -> List[Any]:
        deduped: List[Any] = []
        seen = set()
        for item in candidates:
            chunk_id = str(getattr(item, "chunk_id", ""))
            if not chunk_id or chunk_id in seen:
                continue
            seen.add(chunk_id)
            deduped.append(item)
        return deduped

    def _dedupe_for_display(self, items: Sequence[Any]) -> List[Any]:
        deduped: List[Any] = []
        seen = set()
        for item in items:
            key = (str(getattr(item, "chunk_type", "")), str(getattr(item, "title", "")).strip())
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    def _metadata(self, item: Any) -> Dict[str, Any]:
        metadata = getattr(item, "metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}
            setattr(item, "metadata", metadata)
        return metadata

    def _vector_score(self, item: Any) -> float:
        distance = float(getattr(item, "distance", 999.0) or 999.0)
        if distance >= 900:
            return 0.0
        return 1.0 / (1.0 + max(0.0, distance))

    def _score(self, item: Any) -> float:
        return float(getattr(item, "score", 0.0) or 0.0)

    def _set_rank_score(self, item: Any, value: float) -> None:
        setattr(item, "rank_score", value)


class LLMReranker:
    """Optional fine reranker with strict candidate_id validation."""

    def __init__(self, llm_client: Any = None, model: Optional[str] = None):
        self.llm_client = create_llm_client() if llm_client is None else llm_client
        self.model = model or settings.llm_model

    def rerank(self, candidates: Sequence[Any], context: RerankContext, limit: Optional[int] = None) -> List[Any]:
        max_items = limit or context.top_k
        if not candidates:
            return []
        if self.llm_client is None:
            self._mark(candidates, "llm_skipped", "llm_not_configured")
            return list(candidates)[:max_items]

        try:
            raw = self._complete_json(candidates, context)
            ranked_ids = self._validate_ranked_ids(raw, candidates)
            by_id = {str(getattr(item, "chunk_id", "")): item for item in candidates}
            ranked = [by_id[item_id] for item_id in ranked_ids if item_id in by_id]
            self._mark(ranked, "llm", "")
            return ranked[:max_items]
        except Exception as exc:
            self._mark(candidates, "llm_fallback", str(exc))
            return list(candidates)[:max_items]

    def _complete_json(self, candidates: Sequence[Any], context: RerankContext) -> Dict[str, Any]:
        payload = {
            "user_query": context.query,
            "task_type": context.task_type,
            "user_profile": self._compact_dict(context.user_profile, max_items=12),
            "knowledge_state": context.knowledge_state[:20],
            "candidates": [self._candidate_summary(item) for item in candidates],
        }
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
                response_format={"type": "json_object"},
            )
        except Exception:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
            )
        content = response.choices[0].message.content or "{}"
        parsed = _parse_json_object(content)
        if not isinstance(parsed, dict):
            raise ValueError("LLM reranker output is not a JSON object.")
        return parsed

    def _validate_ranked_ids(self, raw: Dict[str, Any], candidates: Sequence[Any]) -> List[str]:
        valid_ids = [str(getattr(item, "chunk_id", "")) for item in candidates]
        valid_set = set(valid_ids)
        raw_ids = raw.get("ranked_candidate_ids")
        if not isinstance(raw_ids, list):
            raise ValueError("ranked_candidate_ids must be a list.")

        ranked: List[str] = []
        for item in raw_ids:
            candidate_id = str(item or "").strip()
            if candidate_id in valid_set and candidate_id not in ranked:
                ranked.append(candidate_id)

        if not ranked:
            raise ValueError("No valid candidate_id returned by LLM reranker.")

        for candidate_id in valid_ids:
            if candidate_id not in ranked:
                ranked.append(candidate_id)
        return ranked

    def _candidate_summary(self, item: Any) -> Dict[str, Any]:
        metadata = getattr(item, "metadata", {}) if isinstance(getattr(item, "metadata", {}), dict) else {}
        return {
            "candidate_id": str(getattr(item, "chunk_id", "")),
            "type": str(getattr(item, "chunk_type", "")),
            "title": str(getattr(item, "title", "")),
            "summary": " ".join(str(getattr(item, "content", "")).split())[:500],
            "rule_score": float(metadata.get("rule_score") or getattr(item, "score", 0.0) or 0.0),
            "metadata": self._compact_dict(metadata, max_items=10),
        }

    def _compact_dict(self, value: Dict[str, Any], max_items: int) -> Dict[str, Any]:
        if not isinstance(value, dict):
            return {}
        result: Dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= max_items:
                break
            if isinstance(item, (str, int, float, bool)):
                result[str(key)] = item
            elif isinstance(item, list):
                result[str(key)] = item[:8]
        return result

    def _system_prompt(self) -> str:
        return (
            "你是 MOOC RAG 系统的候选资源精排器。"
            "请根据用户问题、任务类型、用户画像、知识状态和候选资源摘要，对候选资源排序。"
            "只能输出 JSON object，格式为："
            "{\"ranked_candidate_ids\":[\"候选ID\"],\"reason\":\"简短排序依据\"}。"
            "ranked_candidate_ids 只能使用输入 candidates 中出现的 candidate_id，不能编造新 ID。"
        )

    def _mark(self, candidates: Sequence[Any], stage: str, error: str) -> None:
        for item in candidates:
            metadata = getattr(item, "metadata", None)
            if isinstance(metadata, dict):
                metadata["llm_rerank_stage"] = stage
                metadata["llm_rerank_error"] = error


class HybridReranker:
    """Rule coarse rerank plus optional LLM fine rerank."""

    def __init__(
        self,
        rule_reranker: Optional[RuleBasedReranker] = None,
        llm_reranker: Optional[LLMReranker] = None,
    ):
        self.rule_reranker = rule_reranker or RuleBasedReranker()
        self.llm_reranker = llm_reranker or LLMReranker()

    def rerank(self, candidates: Sequence[Any], context: RerankContext) -> List[Any]:
        coarse = self.rule_reranker.rerank(candidates, context=context, limit=context.rule_top_k)
        if not context.use_llm_rerank or len(coarse) <= context.top_k:
            return coarse[: context.top_k]
        return self.llm_reranker.rerank(coarse, context=context, limit=context.top_k)


def extract_query_terms(query: str) -> List[str]:
    text = str(query or "").lower()
    terms: List[str] = []
    for item in re.findall(r"[a-z0-9_+#.]+", text):
        if len(item) >= 2 and item not in terms:
            terms.append(item)

    chinese_sequences = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    stop_chars = set("的了和与及或是我你他她它们一个一些最好适合需要推荐相关课程学习")
    for sequence in chinese_sequences:
        cleaned = "".join(char for char in sequence if char not in stop_chars)
        for size in (2, 3, 4):
            for index in range(max(0, len(cleaned) - size + 1)):
                gram = cleaned[index : index + size]
                if len(gram) >= 2 and gram not in terms:
                    terms.append(gram)
    return terms[:80]


def keyword_score(terms: List[str], title: str, content: str, metadata: Dict[str, Any]) -> float:
    title_text = title.lower()
    content_text = content.lower()
    metadata_text = " ".join(str(value) for value in metadata.values()).lower()
    score = 0.0
    for term in terms:
        value = term.lower()
        if value in title_text:
            score += 5.0
        if value in content_text:
            score += 1.5
        if value in metadata_text:
            score += 1.0
    return score


def title_match_score(terms: List[str], title: str) -> float:
    title_text = title.lower()
    if not title_text:
        return 0.0
    return min(1.0, sum(1.0 for term in terms if term.lower() in title_text) / 3.0)


def popularity_score(metadata: Dict[str, Any]) -> float:
    visits = _safe_float(metadata.get("visits"))
    if visits <= 0:
        return 0.0
    return min(1.0, math.log10(visits + 1) / 6.0)


def profile_match_score(item: Any, user_profile: Dict[str, Any]) -> float:
    if not isinstance(user_profile, dict) or not user_profile:
        return 0.0
    text = _candidate_text(item).lower()
    values = []
    for key in ["goal", "learning_stage", "preferred_subjects", "preferred_resource_types", "memory_summary"]:
        values.extend(_as_list(user_profile.get(key)))
    matches = sum(1 for value in values if str(value).lower() and str(value).lower() in text)
    return min(1.0, matches / 3.0)


def knowledge_match_score(item: Any, knowledge_state: List[Dict[str, Any]]) -> float:
    if not knowledge_state:
        return 0.0
    text = _candidate_text(item).lower()
    score = 0.0
    for state in knowledge_state:
        name = str(state.get("knowledge_point") or state.get("knowledge_point_id") or "").lower()
        mastery = _safe_float(state.get("mastery_score"), default=0.5)
        if name and name in text and mastery < 0.6:
            score += 0.5
    return min(1.0, score)


def _candidate_text(item: Any) -> str:
    metadata = getattr(item, "metadata", {}) if isinstance(getattr(item, "metadata", {}), dict) else {}
    return " ".join(
        [
            str(getattr(item, "title", "")),
            str(getattr(item, "content", "")),
            " ".join(str(value) for value in metadata.values()),
        ]
    )


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return [item.strip() for item in str(value).replace("|", ",").split(",") if item.strip()]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_json_object(content: str) -> Dict[str, Any]:
    text = str(content or "").strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            parsed = json.loads(text[start : end + 1])
            return parsed if isinstance(parsed, dict) else {}
        raise
