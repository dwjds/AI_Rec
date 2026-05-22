from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Sequence


def precision_at_k(predicted_ids: Sequence[str], expected_ids: Sequence[str], k: int) -> float:
    expected = set(expected_ids)
    if not predicted_ids or not expected or k <= 0:
        return 0.0
    top = list(predicted_ids)[:k]
    return len([item for item in top if item in expected]) / float(len(top))


def recall_at_k(predicted_ids: Sequence[str], expected_ids: Sequence[str], k: int) -> float:
    expected = set(expected_ids)
    if not expected or k <= 0:
        return 0.0
    top = set(list(predicted_ids)[:k])
    return len(top & expected) / float(len(expected))


def reciprocal_rank(predicted_ids: Sequence[str], expected_ids: Sequence[str]) -> float:
    expected = set(expected_ids)
    if not expected:
        return 0.0
    for index, item in enumerate(predicted_ids, start=1):
        if item in expected:
            return 1.0 / float(index)
    return 0.0


def keyword_coverage(text: str, expected_keywords: Sequence[str]) -> float:
    keywords = [item for item in expected_keywords if item]
    if not keywords:
        return 0.0
    lower_text = str(text or "").lower()
    hits = [keyword for keyword in keywords if keyword.lower() in lower_text]
    return len(hits) / float(len(keywords))


def excluded_keyword_violation_rate(text: str, excluded_keywords: Sequence[str]) -> float:
    keywords = [item for item in excluded_keywords if item]
    if not keywords:
        return 0.0
    lower_text = str(text or "").lower()
    hits = [keyword for keyword in keywords if keyword.lower() in lower_text]
    return len(hits) / float(len(keywords))


def duplicate_rate(items: Sequence[str]) -> float:
    values = [item for item in items if item]
    if not values:
        return 0.0
    return 1.0 - (len(set(values)) / float(len(values)))


def sufficiency_score(count: int, min_count: int = 1) -> float:
    required = max(1, int(min_count or 1))
    return min(1.0, max(0, int(count or 0)) / float(required))


def top1_keyword_coverage(texts: Sequence[str], expected_keywords: Sequence[str]) -> float:
    if not texts:
        return 0.0
    return keyword_coverage(str(texts[0] or ""), expected_keywords)


def keyword_precision_at_k(texts: Sequence[str], expected_keywords: Sequence[str], k: int) -> float:
    keywords = [item for item in expected_keywords if item]
    if not texts or not keywords or k <= 0:
        return 0.0
    top = list(texts)[:k]
    hits = [text for text in top if keyword_coverage(str(text or ""), keywords) > 0.0]
    return len(hits) / float(len(top))


def keyword_reciprocal_rank(texts: Sequence[str], expected_keywords: Sequence[str]) -> float:
    keywords = [item for item in expected_keywords if item]
    if not texts or not keywords:
        return 0.0
    for index, text in enumerate(texts, start=1):
        if keyword_coverage(str(text or ""), keywords) > 0.0:
            return 1.0 / float(index)
    return 0.0


def answer_evidence_overlap(answer: str, evidence_text: str) -> float:
    answer_terms = set(_semantic_terms(answer))
    evidence_terms = set(_semantic_terms(evidence_text))
    if not answer_terms or not evidence_terms:
        return 0.0
    return len(answer_terms & evidence_terms) / float(min(len(answer_terms), len(evidence_terms)))


def evidence_title_mention_rate(answer: str, evidence_titles: Sequence[str]) -> float:
    titles = [str(title).strip() for title in evidence_titles if str(title).strip()]
    if not titles:
        return 0.0
    answer_text = str(answer or "").lower()
    hits = [title for title in titles if title.lower() in answer_text]
    return len(hits) / float(len(titles))


def unsupported_title_mention_rate(answer: str, evidence_titles: Sequence[str]) -> float:
    mentioned = _quoted_titles(answer)
    if not mentioned:
        return 0.0
    allowed = [str(title).strip().lower() for title in evidence_titles if str(title).strip()]
    unsupported = []
    for title in mentioned:
        normalized = title.lower()
        if not any(normalized == item or normalized in item or item in normalized for item in allowed):
            unsupported.append(title)
    return len(unsupported) / float(len(mentioned))


def trace_action_pass_rate(actual_actions: Sequence[str], expected_actions: Sequence[str]) -> float:
    expected = [item for item in expected_actions if item]
    if not expected:
        return 0.0
    actual = list(actual_actions)
    hits = 0
    for expected_action in expected:
        if any(_action_matches(actual_action, expected_action) for actual_action in actual):
            hits += 1
    return hits / float(len(expected))


def grounded_resource_rate(answer: str, allowed_resource_ids: Iterable[str]) -> float:
    ids = [item for item in allowed_resource_ids if item]
    if not ids:
        return 0.0
    answer_text = str(answer or "")
    mentioned = [item for item in ids if item in answer_text]
    return len(mentioned) / float(len(ids))


def average(values: Sequence[float]) -> float:
    return sum(values) / float(len(values)) if values else 0.0


def summarize_case_metrics(case_results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not case_results:
        return {"case_count": 0.0}
    metric_keys = set()
    for result in case_results:
        metric_keys.update((result.get("metrics") or {}).keys())
    summary: Dict[str, Any] = {"case_count": float(len(case_results))}
    for key in sorted(metric_keys):
        values = [
            float((result.get("metrics") or {}).get(key))
            for result in case_results
            if (result.get("metrics") or {}).get(key) is not None
        ]
        if values:
            summary[key] = average(values)
            summary["{0}_case_count".format(key)] = float(len(values))
    return summary


def _action_matches(actual_action: str, expected_action: str) -> bool:
    return actual_action == expected_action or actual_action.endswith(expected_action) or expected_action in actual_action


def _semantic_terms(text: str) -> List[str]:
    value = str(text or "").lower()
    terms: List[str] = []
    for item in re.findall(r"[a-z0-9_+#.]{2,}", value):
        if item not in terms:
            terms.append(item)
    for sequence in re.findall(r"[\u4e00-\u9fff]{2,}", value):
        for size in (2, 3, 4):
            for index in range(max(0, len(sequence) - size + 1)):
                term = sequence[index : index + size]
                if term not in terms:
                    terms.append(term)
    return terms[:400]


def _quoted_titles(text: str) -> List[str]:
    titles: List[str] = []
    for pattern in [r"《([^》]{2,80})》", r"\"([^\"]{2,80})\"", r"'([^']{2,80})'"]:
        for match in re.finditer(pattern, str(text or "")):
            title = match.group(1).strip()
            if title and title not in titles:
                titles.append(title)
    return titles
