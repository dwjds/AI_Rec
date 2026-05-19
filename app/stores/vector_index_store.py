from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.rag.vector_store import ChromaVectorStore

"读取 chunk_id_map.json、chunks.jsonl,查看 Chroma 索引状态"
class VectorIndexStore:
    """Metadata access for chunks and Chroma index status."""

    def __init__(
        self,
        chunks_path: Path | None = None,
        chunk_id_map_path: Path | None = None,
        vector_store: ChromaVectorStore | None = None,
    ):
        self.chunks_path = chunks_path or settings.chunks_path
        self.chunk_id_map_path = chunk_id_map_path or settings.chunk_id_map_path
        self.vector_store = vector_store
        self._chunk_map: Optional[Dict[str, Any]] = None

    def get_chunk_map(self) -> Dict[str, Any]:
        if self._chunk_map is None:
            if not self.chunk_id_map_path.exists():
                self._chunk_map = {}
            else:
                with self.chunk_id_map_path.open("r", encoding="utf-8") as fp:
                    self._chunk_map = json.load(fp)
        return self._chunk_map

    def get_chunk_mapping(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        value = self.get_chunk_map().get(chunk_id)
        return dict(value) if isinstance(value, dict) else value

    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        if not self.chunks_path.exists():
            return None
        with self.chunks_path.open("r", encoding="utf-8") as fp:
            for line in fp:
                if not line.strip():
                    continue
                chunk = json.loads(line)
                if str(chunk.get("id")) == chunk_id:
                    return chunk
        return None

    def list_chunks_by_type(self, chunk_type: str, limit: int = 20) -> List[Dict[str, Any]]:
        if not self.chunks_path.exists():
            return []
        rows = []
        with self.chunks_path.open("r", encoding="utf-8") as fp:
            for line in fp:
                if not line.strip():
                    continue
                chunk = json.loads(line)
                if chunk.get("chunk_type") != chunk_type:
                    continue
                rows.append(chunk)
                if len(rows) >= max(1, min(100, int(limit))):
                    break
        return rows

    def status(self, include_chroma_count: bool = False) -> Dict[str, Any]:
        data = {
            "chunks_path": str(self.chunks_path),
            "chunks_exists": self.chunks_path.exists(),
            "chunk_id_map_path": str(self.chunk_id_map_path),
            "chunk_id_map_exists": self.chunk_id_map_path.exists(),
            "chunk_map_count": len(self.get_chunk_map()) if self.chunk_id_map_path.exists() else 0,
            "chroma_dir": str(settings.chroma_dir),
            "chroma_exists": settings.chroma_dir.exists(),
            "chroma_collection": settings.chroma_collection,
        }
        if include_chroma_count:
            store = self.vector_store or ChromaVectorStore()
            data["chroma_count"] = store.count()
        return data
