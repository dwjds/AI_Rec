from __future__ import annotations

from typing import Iterable, List, Optional

from app.core.config import create_llm_client, settings


class EmbeddingService:
    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        raise NotImplementedError


class OpenAICompatibleEmbeddingService(EmbeddingService):
    """Embedding service backed by the configured OpenAI-compatible API."""

    def __init__(self, model: Optional[str] = None, api_batch_size: int = 10):
        self.client = create_llm_client()
        self.model = model or settings.embedding_model
        self.api_batch_size = max(1, min(10, int(api_batch_size)))
        if self.client is None:
            raise RuntimeError(
                "Embedding client is not configured. Set DASHSCOPE_API_KEY or OPENAI_API_KEY before building/querying vectors."
            )

    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        values = [str(text or "") for text in texts]
        if not values:
            return []
        embeddings: List[List[float]] = []
        for start in range(0, len(values), self.api_batch_size):
            batch = values[start : start + self.api_batch_size]
            response = self.client.embeddings.create(model=self.model, input=batch)
            embeddings.extend([list(item.embedding) for item in response.data])
        return embeddings


def create_embedding_service() -> EmbeddingService:
    return OpenAICompatibleEmbeddingService()
