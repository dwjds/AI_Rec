from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from app.core.config import create_llm_client, settings
from app.rag.query_constraints import extract_query_constraints, merge_excluded_keywords


TASK_TYPES = {"qa", "recommend", "learning_path", "diagnosis", "feedback", "chat"}
PIPELINES = {
    "qa": "rag_qa",
    "recommend": "recommendation_pipeline",
    "learning_path": "agent_loop_learning_path",
    "diagnosis": "agent_loop_diagnosis",
    "feedback": "feedback_pipeline",
    "chat": "direct_chat",
}
TASK_POLICY = {
    "qa": {"needs_rag": True, "needs_user_profile": False, "needs_agent_loop": False},
    "recommend": {"needs_rag": True, "needs_user_profile": True, "needs_agent_loop": False},
    "learning_path": {"needs_rag": True, "needs_user_profile": True, "needs_agent_loop": True},
    "diagnosis": {"needs_rag": True, "needs_user_profile": True, "needs_agent_loop": True},
    "feedback": {"needs_rag": False, "needs_user_profile": True, "needs_agent_loop": False},
    "chat": {"needs_rag": False, "needs_user_profile": False, "needs_agent_loop": False},
}


@dataclass
class RoutingDecision:
    query: str
    task_type: str
    pipeline: str
    needs_rag: bool
    needs_user_profile: bool
    needs_agent_loop: bool
    information_sufficient: bool
    needs_clarification: bool
    clarification_questions: List[str] = field(default_factory=list)
    entities: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    used_llm: bool = False
    fallback_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Router:
    """LLM-first routing decision maker with deterministic policy constraints."""

    def __init__(self, llm_client: Any = None, model: Optional[str] = None):
        self.llm_client = create_llm_client() if llm_client is None else llm_client
        self.model = model or settings.llm_model

    def route(
        self,
        query: str,
        user_context: Optional[Dict[str, Any]] = None,
        use_llm: bool = True,
    ) -> RoutingDecision:
        cleaned_query = self._clean_text(query, max_length=500)
        context = user_context or {}
        if not cleaned_query:
            return self._apply_policy(
                RoutingDecision(
                    query="",
                    task_type="chat",
                    pipeline=PIPELINES["chat"],
                    needs_rag=False,
                    needs_user_profile=False,
                    needs_agent_loop=False,
                    information_sufficient=False,
                    needs_clarification=True,
                    clarification_questions=["你想咨询哪方面的学习问题？"],
                    confidence=0.0,
                    fallback_reason="empty_query",
                ),
                context,
            )

        if use_llm and self.llm_client is not None:
            try:
                raw = self._complete_json(cleaned_query, context)
                return self._validate_llm_output(cleaned_query, raw, context)
            except Exception as exc:
                return self._rule_route(
                    cleaned_query,
                    context,
                    fallback_reason="llm_failed: {0}".format(exc),
                )

        return self._rule_route(cleaned_query, context, fallback_reason="llm_disabled")

    def _complete_json(self, query: str, user_context: Dict[str, Any]) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "query": query,
                        "user_context": self._compact_context(user_context),
                    },
                    ensure_ascii=False,
                ),
            },
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

    def _validate_llm_output(
        self,
        query: str,
        raw: Dict[str, Any],
        user_context: Dict[str, Any],
    ) -> RoutingDecision:
        task_type = self._sanitize_choice(raw.get("task_type"), TASK_TYPES, default="")
        if not task_type:
            raise ValueError("task_type is invalid.")

        decision = RoutingDecision(
            query=query,
            task_type=task_type,
            pipeline=self._clean_text(raw.get("pipeline"), max_length=80),
            needs_rag=self._bool(raw.get("needs_rag")),
            needs_user_profile=self._bool(raw.get("needs_user_profile")),
            needs_agent_loop=self._bool(raw.get("needs_agent_loop")),
            information_sufficient=self._bool(raw.get("information_sufficient")),
            needs_clarification=self._bool(raw.get("needs_clarification")),
            clarification_questions=self._sanitize_string_list(raw.get("clarification_questions"), max_items=3, max_length=80),
            entities=self._sanitize_entities(raw.get("entities")),
            confidence=self._confidence(raw.get("confidence")),
            used_llm=True,
        )
        if decision.confidence < 0.2:
            raise ValueError("routing confidence is too low.")
        return self._apply_policy(decision, user_context)

    def _rule_route(self, query: str, user_context: Dict[str, Any], fallback_reason: str) -> RoutingDecision:
        task_type = self._contextual_task_type(query, user_context) or self._rule_task_type(query)
        entities = self._rule_entities(query, user_context)
        info = self._rule_information_sufficient(task_type, query, entities, user_context)
        decision = RoutingDecision(
            query=query,
            task_type=task_type,
            pipeline=PIPELINES[task_type],
            needs_rag=TASK_POLICY[task_type]["needs_rag"],
            needs_user_profile=TASK_POLICY[task_type]["needs_user_profile"],
            needs_agent_loop=TASK_POLICY[task_type]["needs_agent_loop"],
            information_sufficient=info,
            needs_clarification=not info,
            clarification_questions=self._clarification_questions_for_info(info, task_type, entities, user_context),
            entities=entities,
            confidence=0.55 if task_type != "chat" else 0.4,
            used_llm=False,
            fallback_reason=fallback_reason,
        )
        return self._apply_policy(decision, user_context)

    def _clarification_questions_for_info(
        self,
        information_sufficient: bool,
        task_type: str,
        entities: Dict[str, Any],
        user_context: Dict[str, Any],
    ) -> List[str]:
        if information_sufficient:
            return []
        decision = RoutingDecision(
            query="",
            task_type=task_type,
            pipeline=PIPELINES.get(task_type, PIPELINES["chat"]),
            needs_rag=False,
            needs_user_profile=False,
            needs_agent_loop=False,
            information_sufficient=False,
            needs_clarification=True,
            entities=entities,
        )
        return self._clarification_questions(decision, user_context)

    def _apply_policy(self, decision: RoutingDecision, user_context: Dict[str, Any]) -> RoutingDecision:
        task_type = decision.task_type if decision.task_type in TASK_TYPES else "chat"
        policy = TASK_POLICY[task_type]
        decision.task_type = task_type
        decision.pipeline = PIPELINES[task_type]
        decision.needs_rag = policy["needs_rag"]
        decision.needs_user_profile = policy["needs_user_profile"]
        decision.needs_agent_loop = policy["needs_agent_loop"]
        decision.entities = self._normalize_entities_for_task(self._sanitize_entities(decision.entities), decision.query)
        decision.information_sufficient = decision.information_sufficient and self._rule_information_sufficient(
            task_type=task_type,
            query=decision.query,
            entities=decision.entities,
            user_context=user_context,
        )

        if task_type == "recommend" and self._recommendation_can_proceed(decision.query, decision.entities):
            decision.information_sufficient = True
            decision.needs_clarification = False
            if not decision.clarification_questions:
                decision.clarification_questions = self._optional_recommendation_questions(decision, user_context)
        elif not decision.information_sufficient:
            decision.needs_clarification = True
        elif decision.clarification_questions:
            decision.needs_clarification = True
            decision.information_sufficient = False
        else:
            decision.needs_clarification = False

        if decision.needs_clarification and not decision.clarification_questions:
            decision.clarification_questions = self._clarification_questions(decision, user_context)

        decision.confidence = self._confidence(decision.confidence)
        return decision

    def _rule_task_type(self, query: str) -> str:
        text = query.lower()
        has_recommend_intent = any(word in text for word in ["推荐", "课程", "资源", "教程"])
        if not has_recommend_intent and any(word in text for word in ["不喜欢", "不要", "不需要", "换一个", "换门", "太难", "太简单", "不适合", "反馈"]):
            return "feedback"
        if any(word in text for word in ["学不会", "不会学", "学不下去", "卡住", "薄弱", "诊断", "错题", "困难"]):
            return "diagnosis"
        if any(word in text for word in ["路线", "路径", "规划", "计划", "怎么学", "从零学", "系统学习", "想学", "准备学"]):
            return "learning_path"
        if any(word in text for word in ["推荐", "课程", "资源", "课"]):
            return "recommend"
        if any(word in text for word in ["需要", "想要", "偏向", "侧重", "更想", "更需要"]) and any(
            subject in text
            for subject in [
                "人工智能",
                "机器学习",
                "深度学习",
                "python",
                "java",
                "c++",
                "数据库",
                "数据结构",
                "程序设计",
                "算法",
                "前端",
                "推荐系统",
            ]
        ):
            return "recommend"
        if any(word in text for word in ["是什么", "解释", "概念", "原理", "区别", "为什么", "如何理解"]):
            return "qa"
        return "chat"

    def _contextual_task_type(self, query: str, user_context: Dict[str, Any]) -> str:
        text = str(query or "").lower()
        last = user_context.get("last_routing_decision") or {}
        pending = user_context.get("pending_clarification") or {}
        last_task = str((pending.get("task_type") or last.get("task_type") or "")).strip()
        if last_task not in {"recommend", "learning_path", "diagnosis"}:
            return ""
        if self._rule_task_type(query) != "chat":
            return ""
        if any(
            word in text
            for word in [
                "我学过",
                "学过",
                "基础",
                "零基础",
                "入门",
                "进阶",
                "侧重",
                "偏向",
                "更想",
                "需要",
                "不需要",
                "不是",
                "而不是",
                "语法",
                "目标",
                "竞赛",
                "就业",
                "项目",
                "考试",
            ]
        ):
            return last_task
        return ""

    def _rule_entities(self, query: str, user_context: Dict[str, Any]) -> Dict[str, Any]:
        text = query.lower()
        subjects = []
        for subject in [
            "人工智能",
            "机器学习",
            "深度学习",
            "算法竞赛",
            "程序设计",
            "算法",
            "python",
            "java",
            "c++",
            "数据库",
            "数据结构",
            "推荐系统",
            "前端",
        ]:
            if subject.lower() in text:
                subjects.append("Python" if subject == "python" else subject)
        if not subjects:
            generic_subject = self._extract_generic_subject(query)
            if generic_subject:
                subjects.append(generic_subject)

        stage = ""
        if any(word in text for word in ["零基础", "入门", "初学", "新手"]):
            stage = "beginner"
        elif any(word in text for word in ["进阶", "提高", "提升"]):
            stage = "intermediate"
        elif any(word in text for word in ["高级", "深入", "深度"]):
            stage = "advanced"

        entities: Dict[str, Any] = {}
        if subjects:
            entities["subjects"] = subjects
        if stage:
            entities["learning_stage"] = stage
        profile = user_context.get("profile") or {}
        if not entities.get("subjects") and profile.get("preferred_subjects"):
            entities["subjects"] = profile.get("preferred_subjects")
        if not entities.get("learning_stage") and profile.get("learning_stage"):
            entities["learning_stage"] = profile.get("learning_stage")
        constraints = extract_query_constraints(query)
        if constraints.excluded_keywords:
            entities["negative_terms"] = constraints.negative_phrases
            entities["excluded_keywords"] = constraints.excluded_keywords
        return entities

    def _extract_generic_subject(self, query: str) -> str:
        text = str(query or "").strip()
        patterns = [
            r"(?:推荐|找|学习|想学|规划|制定)?(?:一下|一些)?(?:有关|关于)?\s*([A-Za-z0-9+#\u4e00-\u9fff]{2,30}?)(?:的)?(?:课程|资源|学习资源|学习路线|路线|路径)",
            r"(?:推荐|找|学习|想学|规划|制定)\s*([A-Za-z0-9+#\u4e00-\u9fff]{2,30})",
        ]
        stop_words = {
            "推荐",
            "课程",
            "资源",
            "学习",
            "学习资源",
            "学习路线",
            "路线",
            "路径",
            "有关",
            "关于",
            "一下",
            "一些",
            "帮我",
        }
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.I)
            if not match:
                continue
            subject = self._clean_text(match.group(1), max_length=60)
            for word in stop_words:
                subject = subject.replace(word, "")
            subject = subject.strip(" ，。,.：:；;")
            if len(subject) >= 2 and subject not in stop_words:
                return subject
        return ""

    def _rule_information_sufficient(
        self,
        task_type: str,
        query: str,
        entities: Dict[str, Any],
        user_context: Dict[str, Any],
    ) -> bool:
        if task_type == "chat":
            return True
        if task_type == "qa":
            return len(query) >= 4
        if task_type == "feedback":
            has_recent = bool(user_context.get("recent_recommendations"))
            has_resource_hint = bool(re.search(r"(course|chapter|exercise|mooper)[:_\-\w]*", query, flags=re.I))
            return has_recent or has_resource_hint or any(word in query for word in ["这个", "上一个", "刚才"])
        if task_type == "recommend":
            return self._recommendation_can_proceed(query, entities)
        if task_type == "learning_path":
            return bool(entities.get("subjects")) and bool(entities.get("learning_stage") or user_context.get("profile"))
        if task_type == "diagnosis":
            return bool(entities.get("subjects")) or len(query) >= 8
        return False

    def _clarification_questions(self, decision: RoutingDecision, user_context: Dict[str, Any]) -> List[str]:
        task_type = decision.task_type
        entities = decision.entities or {}
        if task_type == "recommend":
            questions = []
            if not entities.get("subjects"):
                questions.append("你更想学习哪个方向，例如 Python、数据库、机器学习、人工智能或前端？")
            if not entities.get("learning_stage") and not (user_context.get("profile") or {}).get("learning_stage"):
                questions.append("你现在是零基础入门，还是已经学过一些基础内容？")
            return questions[:2]
        if task_type == "learning_path":
            questions = []
            if not entities.get("subjects"):
                questions.append("你想规划哪个学习方向或知识主题？")
            if not entities.get("learning_stage") and not (user_context.get("profile") or {}).get("learning_stage"):
                questions.append("你当前基础大概是什么水平？")
            if not entities.get("goal"):
                questions.append("你的目标是入门了解、完成项目、应对考试，还是转向就业实践？")
            return questions[:3]
        if task_type == "diagnosis":
            return ["你卡住的是哪个具体知识点、课程章节或练习任务？"]
        if task_type == "feedback":
            return ["你想反馈的是刚才推荐的哪一个资源？也可以直接说资源名称或序号。"]
        if task_type == "qa":
            return ["你想解释的是哪个具体概念或知识点？"]
        return ["你想让我帮你推荐资源、解释知识点、规划路线，还是诊断学习困难？"]

    def _normalize_entities_for_task(self, entities: Dict[str, Any], query: str) -> Dict[str, Any]:
        normalized = dict(entities or {})
        constraints = extract_query_constraints(query)
        if constraints.excluded_keywords:
            normalized["negative_terms"] = merge_excluded_keywords(normalized.get("negative_terms"), constraints.negative_phrases)
            normalized["excluded_keywords"] = merge_excluded_keywords(normalized.get("excluded_keywords"), constraints.excluded_keywords)
        if "subject" in normalized and "subjects" not in normalized:
            value = normalized.pop("subject")
            normalized["subjects"] = value if isinstance(value, list) else [str(value)]
        if "topic" in normalized and "subjects" not in normalized:
            value = normalized.pop("topic")
            normalized["subjects"] = value if isinstance(value, list) else [str(value)]
        if "subjects" in normalized:
            values = normalized.get("subjects")
            normalized["subjects"] = self._sanitize_string_list(values if isinstance(values, list) else [values], 10, 60)
            excluded = [item.lower() for item in normalized.get("excluded_keywords") or []]
            if excluded:
                normalized["subjects"] = [
                    subject
                    for subject in normalized["subjects"]
                    if not any(keyword and keyword in subject.lower() for keyword in excluded)
                ]
        if not normalized.get("subjects"):
            rule_entities = self._rule_entities(query, {})
            if rule_entities.get("subjects"):
                normalized["subjects"] = rule_entities["subjects"]
        if not normalized.get("learning_stage"):
            rule_entities = self._rule_entities(query, {})
            if rule_entities.get("learning_stage"):
                normalized["learning_stage"] = rule_entities["learning_stage"]
        return normalized

    def _recommendation_can_proceed(self, query: str, entities: Dict[str, Any]) -> bool:
        if entities.get("subjects"):
            return True
        text = str(query or "")
        has_resource_intent = any(word in text for word in ["推荐", "课程", "资源", "课"])
        has_topic_signal = any(
            word.lower() in text.lower()
            for word in [
                "人工智能",
                "机器学习",
                "深度学习",
                "Python",
                "Java",
                "C++",
                "数据库",
                "数据结构",
                "程序设计",
                "算法竞赛",
                "算法",
                "前端",
                "推荐系统",
            ]
        )
        return has_resource_intent and has_topic_signal

    def _optional_recommendation_questions(self, decision: RoutingDecision, user_context: Dict[str, Any]) -> List[str]:
        questions = []
        profile = user_context.get("profile") or {}
        if not decision.entities.get("learning_stage") and not profile.get("learning_stage"):
            questions.append("你现在是零基础入门，还是已经学过一些基础内容？")
        if not profile.get("goal"):
            questions.append("你的学习目标更偏兴趣了解、项目实践、考试，还是就业提升？")
        return questions[:2]

    def _system_prompt(self) -> str:
        return (
            "你是 MOOC 学习规划与资源推荐 Agent 的路由决策器。"
            "你不只是分类任务，还要决定后续执行策略。"
            "只能输出 JSON object，不要输出解释。"
            "字段必须包含："
            "task_type: qa/recommend/learning_path/diagnosis/feedback/chat 之一；"
            "needs_rag: boolean；needs_user_profile: boolean；needs_agent_loop: boolean；"
            "information_sufficient: boolean；needs_clarification: boolean；"
            "clarification_questions: string[]，最多 3 个；"
            "pipeline: string；entities: object；confidence: 0 到 1。"
            "路由原则：知识解释通常走 qa；课程/资源推荐走 recommend；"
            "路线、规划、怎么学走 learning_path；学不会、卡住、薄弱走 diagnosis；"
            "不喜欢、太难、换一个、不要某类资源走 feedback。"
            "如果信息不足，需要给出追问问题。"
        )

    def _compact_context(self, user_context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "profile": user_context.get("profile") or {},
            "knowledge_state": (user_context.get("knowledge_state") or [])[:10],
            "feedback": (user_context.get("feedback") or [])[:10],
            "recent_recommendations": (user_context.get("recent_recommendations") or [])[:5],
        }

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

    def _sanitize_entities(self, value: Any) -> Dict[str, Any]:
        if not isinstance(value, dict):
            return {}
        entities: Dict[str, Any] = {}
        for key, item in value.items():
            name = self._clean_text(key, max_length=40)
            if not name:
                continue
            if isinstance(item, (str, int, float, bool)):
                entities[name] = item
            elif isinstance(item, list):
                entities[name] = self._sanitize_string_list(item, max_items=10, max_length=60)
        return entities

    def _sanitize_choice(self, value: Any, allowed: set[str], default: str) -> str:
        text = self._clean_text(value, max_length=50)
        return text if text in allowed else default

    def _sanitize_string_list(self, value: Any, max_items: int, max_length: int) -> List[str]:
        if not isinstance(value, list):
            return []
        cleaned = []
        for item in value:
            text = self._clean_text(item, max_length=max_length)
            if text and text not in cleaned:
                cleaned.append(text)
            if len(cleaned) >= max_items:
                break
        return cleaned

    def _bool(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes", "y"}
        return bool(value)

    def _confidence(self, value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = 0.0
        return max(0.0, min(1.0, number))

    def _clean_text(self, value: Any, max_length: int) -> str:
        text = str(value or "").replace("\ufeff", "").strip()
        text = " ".join(text.split())
        return text[:max_length]
