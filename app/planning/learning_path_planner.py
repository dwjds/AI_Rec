from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from app.agent.state import AgentState
from app.rag.evidence_builder import EvidenceItem

"基于 AgentState + memory_context + evidence 生成阶段化学习路线。"
@dataclass
class LearningPathStage:
    stage: str
    objective: str
    resources: List[Dict[str, Any]] = field(default_factory=list)
    practice: List[Dict[str, Any]] = field(default_factory=list)
    checkpoint: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LearningPathPlan:
    goal: str
    current_stage: str
    estimated_path: List[LearningPathStage]
    assumptions: List[str] = field(default_factory=list)
    needs_more_info: bool = False
    clarification_questions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "current_stage": self.current_stage,
            "estimated_path": [stage.to_dict() for stage in self.estimated_path],
            "assumptions": self.assumptions,
            "needs_more_info": self.needs_more_info,
            "clarification_questions": self.clarification_questions,
        }


class LearningPathPlanner:
    """Build a structured learning path from evidence and planning memory."""

    def plan(self, state: AgentState) -> LearningPathPlan:
        planning_context = state.memory_context.get("planner_prompt_context") or state.memory_context.get("planning_context") or {}
        evidence_items = list((state.evidence_package.evidence_items if state.evidence_package else []) or [])
        current_stage = planning_context.get("learning_stage") or "unknown"
        goal = planning_context.get("goal") or state.query

        if not evidence_items:
            return LearningPathPlan(
                goal=goal,
                current_stage=current_stage,
                estimated_path=[],
                assumptions=["当前没有足够的课程证据支撑完整路线。"],
                needs_more_info=True,
                clarification_questions=["你希望学习哪个具体方向，以及当前基础是什么？"],
            )

        courses = self._items_by_type(evidence_items, "course")
        chapters = self._items_by_type(evidence_items, "chapter")
        exercises = self._items_by_type(evidence_items, "exercise")
        knowledge_points = self._items_by_type(evidence_items, "knowledge_point")

        stages = self._stages_for_level(
            learning_stage=current_stage,
            courses=courses,
            chapters=chapters,
            exercises=exercises,
            knowledge_points=knowledge_points,
        )
        assumptions = self._assumptions(planning_context, evidence_items)
        plan = LearningPathPlan(
            goal=goal,
            current_stage=current_stage,
            estimated_path=stages,
            assumptions=assumptions,
            needs_more_info=False,
            clarification_questions=[],
        )
        state.set_plan({"learning_path": plan.to_dict()})
        return plan

    def _stages_for_level(
        self,
        learning_stage: str,
        courses: List[EvidenceItem],
        chapters: List[EvidenceItem],
        exercises: List[EvidenceItem],
        knowledge_points: List[EvidenceItem],
    ) -> List[LearningPathStage]:
        course_resources = [self._resource(item) for item in courses[:4]]
        chapter_resources = [self._resource(item) for item in chapters[:6]]
        practice_resources = [self._resource(item) for item in exercises[:6]]
        knowledge_resources = [self._resource(item) for item in knowledge_points[:6]]

        if learning_stage == "intermediate":
            return [
                LearningPathStage(
                    stage="查漏补缺",
                    objective="先用知识点证据定位薄弱环节，补齐前置知识。",
                    resources=knowledge_resources + chapter_resources[:2],
                    checkpoint="能清楚说明核心知识点之间的关系。",
                ),
                LearningPathStage(
                    stage="课程提升",
                    objective="选择匹配方向的课程继续系统学习。",
                    resources=course_resources,
                    checkpoint="能完成课程中的阶段性任务。",
                ),
                LearningPathStage(
                    stage="练习巩固",
                    objective="通过练习和任务把知识迁移到实践。",
                    practice=practice_resources,
                    checkpoint="能独立完成一个小型实践任务。",
                ),
            ]
        if learning_stage == "advanced":
            return [
                LearningPathStage(
                    stage="专题深化",
                    objective="围绕目标方向选择更深入的课程和章节。",
                    resources=course_resources + chapter_resources[:2],
                    checkpoint="能比较不同方法或模型的适用场景。",
                ),
                LearningPathStage(
                    stage="任务实践",
                    objective="通过练习或任务验证高级知识的掌握情况。",
                    practice=practice_resources,
                    checkpoint="能完成综合性任务并复盘不足。",
                ),
            ]
        return [
            LearningPathStage(
                stage="基础入门",
                objective="先建立核心概念框架，避免直接进入难度较高的任务。",
                resources=knowledge_resources + course_resources[:2],
                checkpoint="能解释主要概念和学习目标。",
            ),
            LearningPathStage(
                stage="课程推进",
                objective="按课程和章节顺序系统学习。",
                resources=course_resources + chapter_resources[:3],
                checkpoint="能跟随课程完成基础章节。",
            ),
            LearningPathStage(
                stage="练习巩固",
                objective="用练习检验掌握情况并形成反馈。",
                practice=practice_resources,
                checkpoint="完成基础练习后更新知识状态。",
            ),
        ]

    def _items_by_type(self, items: List[EvidenceItem], chunk_type: str) -> List[EvidenceItem]:
        return [item for item in items if item.chunk_type == chunk_type]

    def _resource(self, item: EvidenceItem) -> Dict[str, Any]:
        return {
            "evidence_id": item.evidence_id,
            "resource_id": item.source_resource_id,
            "resource_type": item.chunk_type,
            "title": item.title,
            "score": item.score,
        }

    def _assumptions(self, planning_context: Dict[str, Any], evidence_items: List[EvidenceItem]) -> List[str]:
        assumptions = []
        if not planning_context.get("learning_stage"):
            assumptions.append("未明确当前基础，默认按入门到进阶组织。")
        if not planning_context.get("goal"):
            assumptions.append("未明确最终目标，默认以系统学习和资源衔接为目标。")
        assumptions.append("路线仅基于当前检索到的 {0} 条证据组织。".format(len(evidence_items)))
        return assumptions
