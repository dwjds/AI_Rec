from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, List


KNOWN_NEGATIVE_ALIASES = {
    "python": ["Python"],
    "python语法": ["Python", "python语法"],
    "java": ["Java"],
    "java语法": ["Java", "java语法"],
    "c++": ["C++"],
    "c语言": ["C语言"],
    "大数据": ["大数据"],
    "数据挖掘": ["数据挖掘"],
}

NEGATIVE_PATTERNS = [
    re.compile(r"(?:不是|不要|不需要|不想要|别推荐|排除|避免|不考虑|而不是)\s*([^，。,；;\n]{1,30})", re.I),
    re.compile(r"([^，。,；;\n]{2,30})\s*(?:不要|不需要|不想要|不用)", re.I),
]


@dataclass
class QueryConstraints:
    excluded_keywords: List[str] = field(default_factory=list)
    negative_phrases: List[str] = field(default_factory=list)
    positive_query: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "excluded_keywords": self.excluded_keywords,
            "negative_phrases": self.negative_phrases,
            "positive_query": self.positive_query,
        }


def extract_query_constraints(query: str) -> QueryConstraints:
    text = str(query or "").strip()
    negative_phrases = _extract_negative_phrases(text)
    excluded_keywords = _expand_negative_keywords(negative_phrases)
    return QueryConstraints(
        excluded_keywords=excluded_keywords,
        negative_phrases=negative_phrases,
        positive_query=strip_negative_constraints(text),
    )


def strip_negative_constraints(query: str) -> str:
    text = str(query or "")
    for pattern in NEGATIVE_PATTERNS:
        text = pattern.sub(" ", text)
    return " ".join(text.replace("，", " ").replace(",", " ").split()).strip()


def merge_excluded_keywords(*values: Any) -> List[str]:
    merged: List[str] = []
    for value in values:
        for item in _as_list(value):
            text = str(item).strip()
            if text and text.lower() not in {existing.lower() for existing in merged}:
                merged.append(text)
    return merged


def contains_excluded_keyword(text: str, excluded_keywords: Iterable[str]) -> bool:
    haystack = str(text or "").lower()
    return any(str(keyword).strip().lower() in haystack for keyword in excluded_keywords if str(keyword).strip())


def _extract_negative_phrases(text: str) -> List[str]:
    phrases: List[str] = []
    for pattern in NEGATIVE_PATTERNS:
        for match in pattern.finditer(text):
            phrase = _clean_phrase(match.group(1))
            if phrase and phrase.lower() not in {item.lower() for item in phrases}:
                phrases.append(phrase)
    return phrases[:8]


def _expand_negative_keywords(phrases: Iterable[str]) -> List[str]:
    keywords: List[str] = []
    for phrase in phrases:
        lower_phrase = str(phrase).lower()
        for key, aliases in KNOWN_NEGATIVE_ALIASES.items():
            if key in lower_phrase:
                keywords = merge_excluded_keywords(keywords, aliases)
        if phrase:
            keywords = merge_excluded_keywords(keywords, [phrase])
    return keywords[:12]


def _clean_phrase(value: Any) -> str:
    text = str(value or "").strip(" ，。,.：:；;、")
    for suffix in ["方面的", "方面", "方向的", "方向", "课程", "资源", "内容", "相关"]:
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    return text.strip(" ，。,.：:；;、")


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, tuple) or isinstance(value, set):
        return [str(item) for item in value if str(item).strip()]
    return [item.strip() for item in str(value).replace("|", ",").split(",") if item.strip()]
