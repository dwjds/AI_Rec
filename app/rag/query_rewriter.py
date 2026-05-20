from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from app.core.config import create_llm_client, settings
from app.rag.query_constraints import extract_query_constraints, strip_negative_constraints


ALLOWED_CHUNK_TYPES = {"course", "chapter", "exercise", "knowledge_point"}


@dataclass
class QueryRewriteResult:
    original_query: str
    rewritten_query: str
    search_terms: List[str] = field(default_factory=list)
    chunk_types: List[str] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    used_llm: bool = False
    fallback_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class QueryRewriter:
    """LLM-first query rewrite with strict backend validation and rule fallback."""

    def __init__(self, llm_client: Any = None, model: Optional[str] = None):
        self.llm_client = create_llm_client() if llm_client is None else llm_client
        self.model = model or settings.llm_model

    def rewrite(self, query: str, use_llm: bool = True) -> QueryRewriteResult:
        original_query = self._clean_text(query, max_length=500)
        if not original_query:
            return QueryRewriteResult(
                original_query="",
                rewritten_query="",
                fallback_reason="empty_query",
            )

        if use_llm and self.llm_client is not None:
            try:
                raw = self._complete_json(original_query)
                return self._validate_llm_output(original_query, raw)
            except Exception as exc:
                return self._rule_rewrite(original_query, fallback_reason="llm_failed: {0}".format(exc))

        return self._rule_rewrite(original_query, fallback_reason="llm_disabled")

    def _complete_json(self, query: str) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": json.dumps({"query": query}, ensure_ascii=False)},
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
        parsed = self._parse_json(content)
        if not isinstance(parsed, dict):
            raise ValueError("LLM output is not a JSON object.")
        return parsed

    def _validate_llm_output(self, original_query: str, raw: Dict[str, Any]) -> QueryRewriteResult:
        rewritten_query = self._clean_text(strip_negative_constraints(raw.get("rewritten_query")), max_length=300)
        if not rewritten_query:
            raise ValueError("rewritten_query is empty.")

        search_terms = self._sanitize_string_list(raw.get("search_terms"), max_items=12, max_length=40)
        chunk_types = self._sanitize_chunk_types(raw.get("chunk_types"))
        filters = self._sanitize_filters(raw.get("filters"))
        constraints = extract_query_constraints(original_query)
        if constraints.excluded_keywords:
            filters["excluded_keywords"] = constraints.excluded_keywords
            search_terms = [term for term in search_terms if term.lower() not in {item.lower() for item in constraints.excluded_keywords}]
        confidence = self._sanitize_confidence(raw.get("confidence"))

        if confidence < 0.2:
            raise ValueError("rewrite confidence is too low.")

        return QueryRewriteResult(
            original_query=original_query,
            rewritten_query=rewritten_query,
            search_terms=search_terms or self._rule_terms(original_query),
            chunk_types=chunk_types,
            filters=filters,
            confidence=confidence,
            used_llm=True,
        )

    def _rule_rewrite(self, query: str, fallback_reason: str) -> QueryRewriteResult:
        constraints = extract_query_constraints(query)
        positive_query = constraints.positive_query or query
        terms = self._rule_terms(positive_query)
        chunk_types = self._rule_chunk_types(query)
        expanded_terms = self._expand_terms(terms)
        excluded = {item.lower() for item in constraints.excluded_keywords}
        expanded_terms = [term for term in expanded_terms if term.lower() not in excluded]
        rewritten_query = " ".join([positive_query] + expanded_terms)
        return QueryRewriteResult(
            original_query=query,
            rewritten_query=self._clean_text(rewritten_query, max_length=300),
            search_terms=expanded_terms[:12],
            chunk_types=chunk_types,
            filters={"excluded_keywords": constraints.excluded_keywords} if constraints.excluded_keywords else {},
            confidence=0.45,
            used_llm=False,
            fallback_reason=fallback_reason,
        )

    def _system_prompt(self) -> str:
        return (
            "你是 MOOC 教育资源 RAG 系统的查询改写器。"
            "请把用户查询改写成适合检索课程、章节、练习和知识点 chunk 的结构化 JSON。"
            "只能输出 JSON object，不要输出解释。"
            "字段："
            "rewritten_query: string，保留用户核心学习目标并补充检索关键词；"
            "search_terms: string[]，3 到 12 个中文或英文关键词；"
            "chunk_types: string[]，只能从 course, chapter, exercise, knowledge_point 中选择；"
            "filters: object，可包含 subject, difficulty, stage 等弱约束；"
            "confidence: number，0 到 1。"
            "不要编造课程名，不要加入价格、评分等数据库不存在的条件。"
        )

    def _parse_json(self, content: str) -> Any:
        text = str(content or "").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                return json.loads(text[start : end + 1])
            raise

    def _sanitize_string_list(self, value: Any, max_items: int, max_length: int) -> List[str]:
        if not isinstance(value, list):
            return []
        cleaned: List[str] = []
        for item in value:
            text = self._clean_text(item, max_length=max_length)
            if text and text not in cleaned:
                cleaned.append(text)
            if len(cleaned) >= max_items:
                break
        return cleaned

    def _sanitize_chunk_types(self, value: Any) -> List[str]:
        values = self._sanitize_string_list(value, max_items=4, max_length=30)
        return [item for item in values if item in ALLOWED_CHUNK_TYPES]

    def _sanitize_filters(self, value: Any) -> Dict[str, Any]:
        if not isinstance(value, dict):
            return {}
        filters: Dict[str, Any] = {}
        for key, item in value.items():
            name = self._clean_text(key, max_length=40)
            if not name:
                continue
            if isinstance(item, (str, int, float, bool)):
                filters[name] = item
            elif isinstance(item, list):
                filters[name] = self._sanitize_string_list(item, max_items=8, max_length=40)
        return filters

    def _sanitize_confidence(self, value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = 0.0
        return max(0.0, min(1.0, number))

    def _clean_text(self, value: Any, max_length: int) -> str:
        text = str(value or "").replace("\ufeff", "").strip()
        text = " ".join(text.split())
        return text[:max_length]

    def _rule_terms(self, query: str) -> List[str]:
        text = query.lower()
        terms: List[str] = []
        for item in re.findall(r"[a-z0-9_+#.]+", text):
            if len(item) >= 2 and item not in terms:
                terms.append(item)

        known_terms = [
            "人工智能",
            "机器学习",
            "深度学习",
            "python",
            "scikit-learn",
            "sklearn",
            "数据结构",
            "数据库",
            "推荐系统",
            "前端",
            "java",
            "c++",
        ]
        for term in known_terms:
            if term.lower() in text and term not in terms:
                terms.append(term)

        chinese_sequences = re.findall(r"[\u4e00-\u9fff]{2,}", text)
        stop_chars = set("的了和与及或是我你他她它们一个一些最好适合需要推荐相关课程学习")
        for sequence in chinese_sequences:
            cleaned = "".join(char for char in sequence if char not in stop_chars)
            for size in (2, 3, 4):
                for index in range(max(0, len(cleaned) - size + 1)):
                    gram = cleaned[index : index + size]
                    if len(gram) >= 2 and gram not in terms:
                        terms.append(gram)
        return terms[:20]

    def _expand_terms(self, terms: Sequence[str]) -> List[str]:
        aliases = {
            "ai": ["人工智能"],
            "ml": ["机器学习"],
            "sklearn": ["scikit-learn", "机器学习"],
            "scikit-learn": ["sklearn", "机器学习"],
            "python": ["Python"],
        }
        expanded: List[str] = []
        for term in terms:
            if term not in expanded:
                expanded.append(term)
            for alias in aliases.get(term.lower(), []):
                if alias not in expanded:
                    expanded.append(alias)
        return expanded

    def _rule_chunk_types(self, query: str) -> List[str]:
        text = query.lower()
        chunk_types: List[str] = []
        if any(word in text for word in ["课程", "推荐", "入门课", "路线", "路径"]):
            chunk_types.append("course")
        if any(word in text for word in ["章节", "阶段", "第几章", "路线", "路径"]):
            chunk_types.append("chapter")
        if any(word in text for word in ["练习", "实训", "任务", "实践", "项目"]):
            chunk_types.append("exercise")
        if any(word in text for word in ["是什么", "解释", "知识点", "概念", "原理"]):
            chunk_types.append("knowledge_point")
        return chunk_types
