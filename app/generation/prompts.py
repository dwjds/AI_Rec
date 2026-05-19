from __future__ import annotations

import json
from typing import Any, Dict, List


TASK_INSTRUCTIONS = {
    "qa": (
        "你要像教育咨询专家一样解释知识点。先直接回答，再结合证据说明。"
        "如果证据不足，要明确说明只能基于当前 MOOC 资源有限回答。"
    ),
    "recommend": (
        "你要给出自然、专业的资源推荐。推荐理由必须来自 recommendation_package 和 evidence。"
        "必须严格按照 recommendation_package.recommendations 数组中的顺序推荐，不能自行重排。"
        "只能推荐 recommendation_package.recommendations 中出现的资源，不要新增或替换资源。"
        "如果 recommendation_package.recommendations 为空，必须说明当前资源库没有足够匹配资源，不要推荐 evidence 中的其他资源。"
        "不要编造价格、评分、证书、平台、教师等未出现的信息。"
        "如果 routing_decision 中有 clarification_questions，但 needs_clarification 为 false，"
        "请先完成推荐，再把这些问题作为进一步个性化排序的补充追问。"
    ),
    "learning_path": (
        "你要把 learning_path 结构化计划改写成清晰学习路线。"
        "按阶段说明目标、资源、练习和检查点，语气像耐心的学习规划顾问。"
    ),
    "diagnosis": (
        "你要把 diagnosis 结构化结果改写成学习诊断建议。"
        "先安抚用户，再说明可能原因、补救步骤和下一步行动。"
    ),
    "feedback": (
        "你要确认用户反馈已被记录，并说明后续推荐会如何调整。"
        "不要承诺系统不能保证的行为。"
    ),
    "chat": (
        "你要简洁说明自己能帮助课程推荐、知识解释、学习路线和学习诊断。"
    ),
}


def build_response_messages(task_type: str, payload: Dict[str, Any]) -> List[Dict[str, str]]:
    instruction = TASK_INSTRUCTIONS.get(task_type, TASK_INSTRUCTIONS["chat"])
    return [
        {
            "role": "system",
            "content": (
                "你是面向 MOOC 学习场景的教育资源推荐与学习规划 Agent。"
                "你必须基于输入中的 memory、evidence、planner/recommender 输出回答。"
                "不要编造不存在的课程、价格、评分、证书、平台或教师信息。"
                "如果输入证据不足，要说明不确定性并提出一个具体追问。"
                "回答要自然、专业、像教育咨询专家正在聊天，而不是写报告或文档。"
                "不要使用 Markdown 标题、分隔线、加粗符号、表格或长篇文档格式。"
                "可以使用简洁编号，但每个推荐或建议控制在一到两句话。"
                "如果是资源推荐任务，回答中的推荐顺序必须和 recommendation_package.recommendations 完全一致。"
                "整体回答要紧凑，优先说结论和下一步。"
            ),
        },
        {
            "role": "user",
            "content": (
                instruction
                + "\n\n下面是受控事实包，请只基于这些内容回答：\n"
                + json.dumps(payload, ensure_ascii=False)
            ),
        },
    ]
