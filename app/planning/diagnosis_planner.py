from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from app.agent.state import AgentState
from app.rag.evidence_builder import EvidenceItem

"基于用户问题、弱知识点、反馈和 evidence 生成结构化学习诊断与补救方案。"
@dataclass
class RemedialAction:
    action: str
    resources: List[Dict[str, Any]] = field(default_factory=list)
    practice: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DiagnosisPlan:
    diagnosis_type: str
    likely_causes: List[str]
    weak_points: List[Dict[str, Any]]
    remedial_plan: List[RemedialAction]
    next_action: str
    needs_more_info: bool = False
    clarification_questions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "diagnosis_type": self.diagnosis_type,
            "likely_causes": self.likely_causes,
            "weak_points": self.weak_points,
            "remedial_plan": [item.to_dict() for item in self.remedial_plan],
            "next_action": self.next_action,
            "needs_more_info": self.needs_more_info,
            "clarification_questions": self.clarification_questions,
        }


class DiagnosisPlanner:
    """Build a structured learning diagnosis from evidence and memory."""

    def diagnose(self, state: AgentState) -> DiagnosisPlan:
        planning_context = state.memory_context.get("planner_prompt_context") or state.memory_context.get("planning_context") or {}
        feedback_context = state.memory_context.get("raw_memory", {}).get("feedback_memory", {})
        evidence_items = list((state.evidence_package.evidence_items if state.evidence_package else []) or [])
        weak_points = list(planning_context.get("weak_points") or [])

        if not evidence_items and not weak_points:
            return DiagnosisPlan(
                diagnosis_type="unknown",
                likely_causes=["当前信息不足，无法判断具体卡点。"],
                weak_points=[],
                remedial_plan=[],
                next_action="请补充你卡住的知识点、课程章节或练习题。",
                needs_more_info=True,
                clarification_questions=["你具体是哪个知识点、章节或练习任务学不下去？"],
            )

        diagnosis_type = self._diagnosis_type(state.query, feedback_context, weak_points)
        likely_causes = self._likely_causes(diagnosis_type, weak_points, feedback_context)
        remedial_plan = self._remedial_plan(diagnosis_type, evidence_items)
        plan = DiagnosisPlan(
            diagnosis_type=diagnosis_type,
            likely_causes=likely_causes,
            weak_points=weak_points[:10],
            remedial_plan=remedial_plan,
            next_action=self._next_action(remedial_plan),
            needs_more_info=False,
            clarification_questions=[],
        )
        state.set_plan({"diagnosis": plan.to_dict()})
        return plan

    def _diagnosis_type(
        self,
        query: str,
        feedback_context: Dict[str, Any],
        weak_points: List[Dict[str, Any]],
    ) -> str:
        text = str(query or "")
        if any(word in text for word in ["看不懂", "概念", "是什么", "理解不了"]):
            return "concept_gap"
        if any(word in text for word in ["基础", "前置", "跟不上"]):
            return "prerequisite_gap"
        if any(word in text for word in ["不会做", "练习", "题", "项目"]):
            return "practice_gap"
        if any(word in text for word in ["太难", "课程不适合", "资源不适合"]):
            return "resource_mismatch"
        if weak_points:
            return "prerequisite_gap"
        if feedback_context.get("disliked_resource_ids"):
            return "resource_mismatch"
        return "unknown"

    def _likely_causes(
        self,
        diagnosis_type: str,
        weak_points: List[Dict[str, Any]],
        feedback_context: Dict[str, Any],
    ) -> List[str]:
        causes = {
            "concept_gap": ["核心概念还没有建立稳定理解。"],
            "prerequisite_gap": ["前置知识可能不足，导致后续内容难以衔接。"],
            "practice_gap": ["练习量或任务迁移不足，导致知道概念但不会应用。"],
            "resource_mismatch": ["当前资源难度、方向或呈现方式可能不适合用户。"],
            "unknown": ["学习困难原因尚不明确，需要更多具体上下文。"],
        }.get(diagnosis_type, ["学习困难原因尚不明确。"])
        if weak_points:
            causes.append("用户知识状态中存在低掌握度知识点。")
        if feedback_context.get("negative_preferences"):
            causes.append("近期负反馈显示部分资源或方向不匹配。")
        return causes

    def _remedial_plan(self, diagnosis_type: str, evidence_items: List[EvidenceItem]) -> List[RemedialAction]:
        knowledge = [self._resource(item) for item in evidence_items if item.chunk_type == "knowledge_point"][:5]
        chapters = [self._resource(item) for item in evidence_items if item.chunk_type == "chapter"][:5]
        courses = [self._resource(item) for item in evidence_items if item.chunk_type == "course"][:4]
        exercises = [self._resource(item) for item in evidence_items if item.chunk_type == "exercise"][:5]

        if diagnosis_type in {"concept_gap", "prerequisite_gap"}:
            return [
                RemedialAction(
                    action="先补核心概念和前置知识。",
                    resources=knowledge + chapters + courses[:1],
                ),
                RemedialAction(
                    action="再用基础练习确认是否真正掌握。",
                    practice=exercises,
                ),
            ]
        if diagnosis_type == "practice_gap":
            return [
                RemedialAction(
                    action="优先做低门槛练习，把概念迁移到任务。",
                    resources=chapters[:2],
                    practice=exercises,
                )
            ]
        if diagnosis_type == "resource_mismatch":
            return [
                RemedialAction(
                    action="更换为更贴近当前基础的课程或章节。",
                    resources=courses + chapters[:2],
                    practice=exercises[:2],
                )
            ]
        return [
            RemedialAction(
                action="先选择一个明确知识点进行定位，再安排补救资源。",
                resources=knowledge + chapters + courses[:1],
                practice=exercises[:2],
            )
        ]

    def _next_action(self, remedial_plan: List[RemedialAction]) -> str:
        if not remedial_plan:
            return "补充具体卡点后再诊断。"
        first = remedial_plan[0]
        if first.resources:
            return "先学习《{0}》，再完成对应练习并反馈掌握情况。".format(first.resources[0].get("title"))
        if first.practice:
            return "先完成《{0}》相关练习，观察错误集中在哪里。".format(first.practice[0].get("title"))
        return first.action

    def _resource(self, item: EvidenceItem) -> Dict[str, Any]:
        return {
            "evidence_id": item.evidence_id,
            "resource_id": item.source_resource_id,
            "resource_type": item.chunk_type,
            "title": item.title,
            "score": item.score,
        }
