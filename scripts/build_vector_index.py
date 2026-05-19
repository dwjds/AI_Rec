from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings
from app.rag.vector_store import ChromaVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a persistent Chroma vector database from RAG chunks.")
    parser.add_argument("--chunks", default=str(settings.chunks_path), help="Input chunks.jsonl path.")
    parser.add_argument("--persist-dir", default=str(settings.chroma_dir), help="Chroma persistence directory.")
    parser.add_argument("--collection", default=settings.chroma_collection, help="Chroma collection name.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--reset", action="store_true", help="Delete and rebuild the Chroma collection.")
    args = parser.parse_args()

    from app.rag.embedding_service import create_embedding_service

    store = ChromaVectorStore(
        persist_dir=Path(args.persist_dir),
        collection_name=args.collection,
        embedding_service=create_embedding_service(),
    )
    total = store.add_chunks_from_jsonl(Path(args.chunks), batch_size=args.batch_size, reset=args.reset)
    print("Built Chroma collection: {0}".format(args.collection))
    print("Persist dir: {0}".format(args.persist_dir))
    print("Added chunks: {0}".format(total))
    print("Collection count: {0}".format(store.count()))


if __name__ == "__main__":
    main()
