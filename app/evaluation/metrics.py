from __future__ import annotations

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


def duplicate_rate(items: Sequence[str]) -> float:
    values = [item for item in items if item]
    if not values:
        return 0.0
    return 1.0 - (len(set(values)) / float(len(values)))


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


def summarize_case_metrics(case_results: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    if not case_results:
        return {"case_count": 0.0}
    metric_keys = set()
    for result in case_results:
        metric_keys.update((result.get("metrics") or {}).keys())
    summary: Dict[str, float] = {"case_count": float(len(case_results))}
    for key in sorted(metric_keys):
        values = [float((result.get("metrics") or {}).get(key) or 0.0) for result in case_results]
        summary[key] = average(values)
    return summary


def _action_matches(actual_action: str, expected_action: str) -> bool:
    return actual_action == expected_action or actual_action.endswith(expected_action) or expected_action in actual_action
