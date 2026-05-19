from __future__ import annotations

import re
from typing import Any, Dict, Iterator, List, Optional

from app.agent.state import AgentState
from app.generation.llm_client import LLMClient
from app.generation.prompts import build_response_messages
from app.memory.context_compressor import ContextCompressor


class ResponseGenerator:
    """Generate final user-facing answers from AgentState."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()
        self.context_compressor = ContextCompressor()

    def generate(self, state: AgentState, use_llm: bool = True) -> str:
        if use_llm and self.llm_client.enabled:
            try:
                answer = self._generate_with_llm(state)
                if answer:
                    answer = self._clean_chat_answer(answer)
                    state.set_final_answer(answer)
                    state.add_metadata("response_generation", {"llm_used": True, "fallback_reason": ""})
                    return answer
            except Exception as exc:
                state.add_metadata("response_generation", {"llm_used": False, "fallback_reason": str(exc)})

        answer = self._fallback_answer(state)
        answer = self._clean_chat_answer(answer)
        state.set_final_answer(answer)
        state.metadata.setdefault("response_generation", {"llm_used": False, "fallback_reason": "llm_disabled"})
        return answer

    def stream_generate(self, state: AgentState, use_llm: bool = True) -> Iterator[str]:
        if use_llm and self.llm_client.enabled:
            try:
                chunks: List[str] = []
                for chunk in self._stream_with_llm(state):
                    chunks.append(chunk)
                    yield chunk
                answer = self._clean_chat_answer("".join(chunks))
                if answer:
                    state.set_final_answer(answer)
                    state.add_metadata("response_generation", {"llm_used": True, "fallback_reason": "", "stream": True})
                    return
            except Exception as exc:
                state.add_metadata("response_generation", {"llm_used": False, "fallback_reason": str(exc), "stream": True})

        answer = self._fallback_answer(state)
        answer = self._clean_chat_answer(answer)
        state.set_final_answer(answer)
        state.metadata.setdefault("response_generation", {"llm_used": False, "fallback_reason": "llm_disabled", "stream": True})
        for chunk in self._chunk_text(answer):
            yield chunk

    def _generate_with_llm(self, state: AgentState) -> str:
        payload = self._payload(state)
        messages = build_response_messages(state.task_type, payload)
        return self.llm_client.complete(messages=messages, temperature=0.35, max_tokens=1400)

    def _stream_with_llm(self, state: AgentState) -> Iterator[str]:
        payload = self._payload(state)
        messages = build_response_messages(state.task_type, payload)
        yield from self.llm_client.stream_complete(messages=messages, temperature=0.35, max_tokens=1400)

    def _chunk_text(self, text: str, size: int = 24) -> Iterator[str]:
        value = str(text or "")
        for start in range(0, len(value), size):
            yield value[start : start + size]

    def _payload(self, state: AgentState) -> Dict[str, Any]:
        return {
            "task_type": state.task_type,
            "pipeline": state.pipeline,
            "user_query": state.query,
            "routing_decision": state.routing_decision.to_dict() if state.routing_decision else {},
            "memory": {
                "planning_context": state.memory_context.get("planning_context") or {},
                "generation_context": state.memory_context.get("generation_context") or {},
                "generator_prompt_context": state.memory_context.get("generator_prompt_context") or {},
                "ranking_context": state.memory_context.get("ranking_context") or {},
            },
            "evidence_package": self._compact_evidence(state),
            "planner_output": state.plan,
            "recommendation_package": state.metadata.get("recommendation_package") or {},
            "feedback_event": state.feedback_event or {},
            "constraints": [
                "只能使用 evidence_package、planner_output、recommendation_package 中存在的信息。",
                "不要编造价格、评分、证书、平台或教师。",
                "若证据不足，必须明确说明并提出追问。",
            ],
        }

    def _compact_evidence(self, state: AgentState) -> Dict[str, Any]:
        if state.evidence_package is None:
            return {}
        return self.context_compressor.compress_evidence_package(state.evidence_package)

    def _fallback_answer(self, state: AgentState) -> str:
        if state.routing_decision and state.routing_decision.needs_clarification:
            return state.final_answer or self._clarification_fallback(state)
        if state.task_type == "recommend":
            return self._recommendation_fallback(state)
        if state.task_type == "learning_path":
            return self._learning_path_fallback(state)
        if state.task_type == "diagnosis":
            return self._diagnosis_fallback(state)
        if state.task_type == "qa":
            return self._evidence_fallback("我会基于检索到的课程知识证据来回答。", state)
        if state.task_type == "feedback":
            return state.final_answer or "收到，我已经记录了你的反馈，后续推荐会参考这个偏好。"
        return state.final_answer or "我可以帮你做课程推荐、知识点解释、学习路线规划，或者诊断学习困难。"

    def _recommendation_fallback(self, state: AgentState) -> str:
        package = state.metadata.get("recommendation_package") or {}
        recommendations = package.get("recommendations") or []
        if not recommendations:
            query = state.query or "这个方向"
            return (
                "我检索了当前 MOOPer 资源库，但没有找到与“{0}”足够匹配的学习资源。"
                "为了避免把不相关课程硬推荐给你，我先不列课程。\n\n"
                "你可以换一个更接近资源库覆盖范围的方向，或者补充具体知识点，我再重新检索。"
            ).format(query)
        lines = ["我先按你的目标、用户画像和资源库真实资料筛出这些资源：", ""]
        for index, item in enumerate(recommendations[:5], start=1):
            lines.append("{0}. {1}".format(index, item.get("title") or item.get("resource_id")))
            reasons = item.get("reasons") or []
            if reasons:
                lines.append("   推荐依据：" + "；".join(str(reason) for reason in reasons[:3]))
        next_steps = package.get("next_steps") or []
        if next_steps:
            lines.append("")
            lines.append("建议下一步：" + str(next_steps[0]))
        questions = self._non_blocking_questions(state)
        if questions:
            lines.append("")
            lines.append("如果你想让我进一步按个人情况调整顺序，可以再补充：")
            for index, question in enumerate(questions, start=1):
                lines.append("{0}. {1}".format(index, question))
        return "\n".join(lines)

    def _learning_path_fallback(self, state: AgentState) -> str:
        plan = state.plan.get("learning_path") or state.metadata.get("learning_path_plan") or {}
        stages = plan.get("estimated_path") or []
        if not stages:
            return self._evidence_fallback("我会按“目标判断 -> 证据检索 -> 学习行动”的方式处理。", state)
        lines = ["我根据你的目标、当前记忆和检索证据，先给出这条学习路线：", ""]
        for index, stage in enumerate(stages[:4], start=1):
            lines.append("{0}. {1}：{2}".format(index, stage.get("stage"), stage.get("objective")))
            if stage.get("checkpoint"):
                lines.append("   检查点：" + str(stage.get("checkpoint")))
        return "\n".join(lines)

    def _diagnosis_fallback(self, state: AgentState) -> str:
        plan = state.plan.get("diagnosis") or state.metadata.get("diagnosis_plan") or {}
        if not plan:
            return self._evidence_fallback("我先按学习诊断来判断。", state)
        lines = ["我先按学习诊断来判断：", ""]
        lines.append("可能类型：" + str(plan.get("diagnosis_type") or "unknown"))
        for cause in (plan.get("likely_causes") or [])[:3]:
            lines.append("- " + str(cause))
        lines.append("下一步：" + str(plan.get("next_action") or "先补充具体卡点。"))
        return "\n".join(lines)

    def _clarification_fallback(self, state: AgentState) -> str:
        decision = state.routing_decision
        questions = decision.clarification_questions if decision else ["你想咨询哪方面的学习问题？"]
        return "为了给你更准确的帮助，我需要先确认：\n" + "\n".join(
            "{0}. {1}".format(index, question) for index, question in enumerate(questions, start=1)
        )

    def _evidence_fallback(self, prefix: str, state: AgentState) -> str:
        if state.evidence_package is None or not state.evidence_package.evidence_items:
            return prefix + "\n\n目前没有检索到足够可靠的 MOOC 资源证据，我建议先补充更具体的学习方向。"
        lines = [prefix, ""]
        for index, item in enumerate(state.evidence_package.evidence_items[:5], start=1):
            lines.append("{0}. {1}".format(index, item.title))
            if item.resource.get("description"):
                lines.append("   依据：" + str(item.resource.get("description"))[:120])
            elif item.content:
                lines.append("   依据：" + item.content[:120])
        return "\n".join(lines)

    def _non_blocking_questions(self, state: AgentState) -> List[str]:
        decision = state.routing_decision
        if not decision or decision.needs_clarification:
            return []
        if decision.task_type != "recommend":
            return []
        return list(decision.clarification_questions or [])[:2]

    def _clean_chat_answer(self, answer: str) -> str:
        text = str(answer or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        if not text:
            return ""

        cleaned_lines: List[str] = []
        for line in text.split("\n"):
            current = line.rstrip()
            if re.match(r"^\s*[-*_]{3,}\s*$", current):
                continue
            current = re.sub(r"^\s{0,3}#{1,6}\s*", "", current)
            current = current.replace("**", "").replace("__", "")
            current = re.sub(r"^\s*[-*]\s+", "", current)
            cleaned_lines.append(current)

        cleaned = "\n".join(cleaned_lines)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()
