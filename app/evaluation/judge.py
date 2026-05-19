from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Sequence

from app.evaluation.metrics import keyword_coverage


@dataclass
class JudgeResult:
    groundedness_score: float
    usefulness_score: float
    hallucination_risk: str
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RuleBasedJudge:
    """Lightweight judge used before introducing LLM-as-Judge."""

    def judge(
        self,
        answer: str,
        evidence_text: str = "",
        expected_keywords: Sequence[str] | None = None,
    ) -> JudgeResult:
        expected_keywords = list(expected_keywords or [])
        keyword_score = keyword_coverage(answer, expected_keywords) if expected_keywords else 0.0
        evidence_hit = self._evidence_overlap(answer, evidence_text)
        groundedness = max(keyword_score, evidence_hit)
        usefulness = max(keyword_score, 0.5 if answer.strip() else 0.0)
        risk = "low" if groundedness >= 0.7 else "medium" if groundedness >= 0.35 else "high"
        return JudgeResult(
            groundedness_score=round(groundedness, 4),
            usefulness_score=round(usefulness, 4),
            hallucination_risk=risk,
            reason="规则评估：关键词覆盖率与证据文本重合度。",
        )

    def _evidence_overlap(self, answer: str, evidence_text: str) -> float:
        answer_terms = {item for item in str(answer or "").replace("\n", " ").split(" ") if len(item) >= 2}
        evidence_terms = {item for item in str(evidence_text or "").replace("\n", " ").split(" ") if len(item) >= 2}
        if not answer_terms or not evidence_terms:
            return 0.0
        return len(answer_terms & evidence_terms) / float(min(len(answer_terms), len(evidence_terms)))
