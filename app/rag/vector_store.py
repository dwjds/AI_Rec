from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from app.core.config import settings
from app.rag.embedding_service import EmbeddingService, create_embedding_service


class ChromaVectorStore:
    def __init__(
        self,
        persist_dir: Optional[Path] = None,
        collection_name: Optional[str] = None,
        embedding_service: Optional[EmbeddingService] = None,
    ):
        self.persist_dir = persist_dir or settings.chroma_dir
        self.collection_name = collection_name or settings.chroma_collection
        self.embedding_service = embedding_service or create_embedding_service()
        self._client = None
        self._collection = None

    @property
    def collection(self) -> Any:
        if self._collection is None:
            try:
                import chromadb
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("chromadb is not installed. Run: pip install chromadb") from exc

            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.persist_dir))
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "MOOPer MOOC resource RAG chunks"},
            )
        return self._collection

    def reset_collection(self) -> None:
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("chromadb is not installed. Run: pip install chromadb") from exc

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "MOOPer MOOC resource RAG chunks"},
        )

    def add_chunks(self, chunks: Iterable[Dict[str, Any]], batch_size: int = 128, reset: bool = False) -> int:
        if reset:
            self.reset_collection()

        total = 0
        batch: List[Dict[str, Any]] = []
        for chunk in chunks:
            batch.append(chunk)
            if len(batch) >= batch_size:
                total += self._add_batch(batch)
                batch = []
        if batch:
            total += self._add_batch(batch)
        return total

    def add_chunks_from_jsonl(self, chunks_path: Optional[Path] = None, batch_size: int = 128, reset: bool = False) -> int:
        path = chunks_path or settings.chunks_path
        with path.open("r", encoding="utf-8") as fp:
            chunks = (json.loads(line) for line in fp if line.strip())
            return self.add_chunks(chunks, batch_size=batch_size, reset=reset)

    def query(self, query: str, top_k: int = 5, where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        embeddings = self.embedding_service.embed_texts([query])
        result = self.collection.query(
            query_embeddings=embeddings,
            n_results=max(1, min(50, int(top_k))),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        rows: List[Dict[str, Any]] = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        for index, chunk_id in enumerate(ids):
            rows.append(
                {
                    "chunk_id": chunk_id,
                    "content": docs[index],
                    "metadata": metadatas[index],
                    "distance": distances[index],
                }
            )
        return rows

    def count(self) -> int:
        return int(self.collection.count())

    def _add_batch(self, batch: List[Dict[str, Any]]) -> int:
        documents = [str(item["content"]) for item in batch]
        embeddings = self.embedding_service.embed_texts(documents)
        self.collection.upsert(
            ids=[str(item["id"]) for item in batch],
            documents=documents,
            embeddings=embeddings,
            metadatas=[self._flatten_metadata(item) for item in batch],
        )
        return len(batch)

    def _flatten_metadata(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        metadata = dict(chunk.get("metadata") or {})
        metadata.update(
            {
                "chunk_type": chunk.get("chunk_type", ""),
                "source": chunk.get("source", ""),
                "source_resource_id": chunk.get("source_resource_id", ""),
                "source_resource_type": chunk.get("source_resource_type", ""),
                "title": chunk.get("title", ""),
            }
        )
        flattened: Dict[str, Any] = {}
        for key, value in metadata.items():
            if value is None:
                flattened[key] = ""
            elif isinstance(value, (str, int, float, bool)):
                flattened[key] = value
            else:
                flattened[key] = str(value)
        return flattened
