from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

from app.core.config import create_llm_client, settings


class LLMClient:
    """Small wrapper around the configured OpenAI-compatible chat client."""

    def __init__(self, client: Any = None, model: Optional[str] = None):
        self.client = create_llm_client() if client is None else client
        self.model = model or settings.llm_model

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 1200,
    ) -> str:
        if self.client is None:
            raise RuntimeError("LLM client is not configured.")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return str(response.choices[0].message.content or "").strip()

    def stream_complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 1200,
    ) -> Iterator[str]:
        if self.client is None:
            raise RuntimeError("LLM client is not configured.")
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            content = getattr(delta, "content", None) if delta is not None else None
            if content:
                yield str(content)
