from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.retriever import RagRetriever


CASES = [
    {
        "name": "课程推荐：人工智能入门",
        "query": "推荐适合入门的人工智能课程，最好包含 Python 和机器学习基础",
        "chunk_types": ["course"],
    },
    {
        "name": "知识点问答：scikit-learn",
        "query": "scikit-learn 是什么，适合用来学习哪些机器学习内容",
        "chunk_types": ["knowledge_point"],
    },
    {
        "name": "章节检索：Python 基础",
        "query": "Python 入门基础、变量、函数相关章节",
        "chunk_types": ["chapter"],
    },
    {
        "name": "练习检索：机器学习分类",
        "query": "机器学习分类算法练习，最好涉及特征提取和模型训练",
        "chunk_types": ["exercise"],
    },
    {
        "name": "混合检索：三个月学习机器学习路线",
        "query": "我想三个月学习机器学习并完成一个项目，需要课程、知识点和练习",
        "chunk_types": None,
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run manual RAG retrieval cases against the Chroma vector database.")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    retriever = RagRetriever()
    for case in CASES:
        print("\n=== {0} ===".format(case["name"]))
        print("Query: {0}".format(case["query"]))
        results = retriever.retrieve(case["query"], top_k=args.top_k, chunk_types=case["chunk_types"])
        for index, item in enumerate(results, start=1):
            distance = "n/a" if item.distance >= 900 else "{0:.4f}".format(item.distance)
            print(
                "{0}. [{1}] {2} | score={3:.4f} distance={4}".format(
                    index,
                    item.chunk_type,
                    item.title,
                    item.score,
                    distance,
                )
            )
            snippet = " ".join(item.content.split())[:160]
            print("   {0}".format(snippet))


if __name__ == "__main__":
    main()
