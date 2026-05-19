from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings
from app.rag.chunker import MooperChunker


def parse_types(value: Optional[str]) -> Optional[Iterable[str]]:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RAG chunks from the MOOPer SQLite database.")
    parser.add_argument("--output", type=Path, default=settings.chunks_path, help="Output chunks.jsonl path.")
    parser.add_argument("--types", type=str, default="", help="Comma-separated chunk types: course,chapter,exercise,knowledge_point.")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit for quick inspection.")
    args = parser.parse_args()

    chunker = MooperChunker()
    include_types = parse_types(args.types)
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.limit > 0:
        counts: dict[str, int] = {}
        chunk_id_map = {}
        with output.open("w", encoding="utf-8") as fp:
            for index, chunk in enumerate(chunker.iter_chunks(include_types=include_types), start=1):
                fp.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
                counts[chunk.chunk_type] = counts.get(chunk.chunk_type, 0) + 1
                chunk_id_map[chunk.id] = {
                    "source_resource_id": chunk.source_resource_id,
                    "source_resource_type": chunk.source_resource_type,
                    "title": chunk.title,
                }
                if index >= args.limit:
                    break
        settings.chunk_id_map_path.write_text(json.dumps(chunk_id_map, ensure_ascii=False, indent=2), encoding="utf-8")
        counts["total"] = sum(counts.values())
    else:
        counts = chunker.write_jsonl(output_path=output, include_types=include_types)

    print("Built chunks: {0}".format(output))
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
