from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import settings
from app.evaluation.datasets import EvalCase, load_eval_suite
from app.evaluation.evaluators import EvaluationConfig, EvaluationRunner
from app.evaluation.report import write_eval_report


DEFAULT_SUITES = ["retrieval", "recommendation", "qa", "agent_loop", "failure"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline evaluation for the MOOC RAG Agent.")
    parser.add_argument("--case-dir", default=str(settings.data_dir / "eval"), help="Directory containing *_cases.jsonl files.")
    parser.add_argument("--output-dir", default=str(settings.data_dir / "eval_reports"), help="Directory for eval reports.")
    parser.add_argument("--suites", default=",".join(DEFAULT_SUITES), help="Comma separated suite names.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--use-llm-route", action="store_true")
    parser.add_argument("--use-llm-rerank", action="store_true")
    parser.add_argument("--use-llm-generation", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print full JSON report.")
    args = parser.parse_args()

    case_dir = Path(args.case_dir)
    suites = [item.strip() for item in args.suites.split(",") if item.strip()]
    loaded = load_eval_suite(case_dir, suites)
    loaded = {name: cases for name, cases in loaded.items() if cases}

    if not loaded:
        loaded = _fallback_smoke_cases()

    runner = EvaluationRunner(
        config=EvaluationConfig(
            top_k=args.top_k,
            use_llm_route=args.use_llm_route,
            use_llm_rerank=args.use_llm_rerank,
            use_llm_generation=args.use_llm_generation,
        )
    )
    report = runner.evaluate_suite(loaded)
    paths = write_eval_report(report, Path(args.output_dir))

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print("Evaluation finished.")
        print("JSON report: {0}".format(paths["json"]))
        print("Markdown report: {0}".format(paths["markdown"]))
        for suite_name, summary in report.get("summary", {}).items():
            print("{0}: {1}".format(suite_name, summary))


def _fallback_smoke_cases() -> Dict[str, List[EvalCase]]:
    return {
        "retrieval": [
            EvalCase(
                id="smoke_retrieval_ai",
                query="人工智能入门课程",
                task_type="recommend",
                expected_keywords=["人工智能"],
            )
        ],
        "agent_loop": [
            EvalCase(
                id="smoke_agent_loop_learning_path",
                query="零基础三个月学习机器学习路线",
                task_type="learning_path",
                expected_trace_actions=[
                    "read_memory",
                    "search_courses",
                    "retrieve_evidence",
                    "run_learning_path_planner",
                    "generate_response",
                ],
            )
        ],
    }


if __name__ == "__main__":
    main()
