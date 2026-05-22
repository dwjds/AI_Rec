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
from app.evaluation.checkpoint import EvaluationCheckpoint
from app.evaluation.datasets import EvalCase, load_eval_suite
from app.evaluation.evaluators import EvaluationConfig, EvaluationRunner
from app.evaluation.report import write_eval_report


DEFAULT_SUITES = ["retrieval", "recommendation", "qa", "agent_loop", "failure", "demo_flow"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline evaluation for the MOOC RAG Agent.")
    parser.add_argument("--case-dir", default=str(settings.data_dir / "eval"), help="Directory containing *_cases.jsonl files.")
    parser.add_argument("--output-dir", default=str(settings.data_dir / "eval_reports"), help="Directory for eval reports.")
    parser.add_argument("--suites", default=",".join(DEFAULT_SUITES), help="Comma separated suite names.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--use-llm-route", dest="use_llm_route", action="store_true", default=True, help="Enable LLM router. Enabled by default.")
    parser.add_argument("--no-llm-route", dest="use_llm_route", action="store_false", help="Disable LLM router and use rule fallback.")
    parser.add_argument(
        "--use-llm-rerank",
        dest="use_llm_rerank",
        action="store_true",
        default=True,
        help="Enable LLM reranker. Enabled by default.",
    )
    parser.add_argument(
        "--no-llm-rerank",
        dest="use_llm_rerank",
        action="store_false",
        help="Disable LLM reranker and use rule-based rerank only.",
    )
    parser.add_argument("--use-llm-generation", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume from the checkpoint file and skip completed cases.")
    parser.add_argument("--reset-checkpoint", action="store_true", help="Delete the checkpoint before running.")
    parser.add_argument("--retry-failed", action="store_true", help="When resuming, remove failed cached cases and run them again.")
    parser.add_argument("--no-checkpoint", action="store_true", help="Disable per-case checkpoint writes.")
    parser.add_argument("--no-progress", action="store_true", help="Disable evaluation progress display.")
    parser.add_argument("--checkpoint-path", default="", help="Checkpoint path. Defaults to <output-dir>/eval_checkpoint.json.")
    parser.add_argument("--json", action="store_true", help="Print full JSON report.")
    args = parser.parse_args()

    case_dir = Path(args.case_dir)
    suites = [item.strip() for item in args.suites.split(",") if item.strip()]
    loaded = load_eval_suite(case_dir, suites)
    loaded = {name: cases for name, cases in loaded.items() if cases}

    if not loaded:
        loaded = _fallback_smoke_cases()

    config = EvaluationConfig(
        top_k=args.top_k,
        use_llm_route=args.use_llm_route,
        use_llm_rerank=args.use_llm_rerank,
        use_llm_generation=args.use_llm_generation,
        show_progress=not args.no_progress,
    )
    checkpoint = None
    if not args.no_checkpoint:
        checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else Path(args.output_dir) / "eval_checkpoint.json"
        checkpoint = EvaluationCheckpoint(
            path=checkpoint_path,
            config_key=_checkpoint_config_key(config, suites),
            resume=args.resume,
        )
        if args.reset_checkpoint:
            checkpoint.clear()
        if args.retry_failed:
            removed = checkpoint.remove_failed(suites)
            if removed:
                print("Removed {0} failed cached case(s) from checkpoint.".format(removed), file=sys.stderr)
        checkpoint.save()

    runner = EvaluationRunner(config=config, checkpoint=checkpoint)
    report = runner.evaluate_suite(loaded)
    if checkpoint is not None:
        report["checkpoint"] = {
            "path": str(checkpoint.path),
            "resume_enabled": bool(args.resume),
        }
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


def _checkpoint_config_key(config: EvaluationConfig, suites: List[str]) -> str:
    payload = {
        "evaluation_schema_version": 2,
        "config": {
            "top_k": config.top_k,
            "use_llm_route": config.use_llm_route,
            "use_llm_rerank": config.use_llm_rerank,
            "use_llm_generation": config.use_llm_generation,
        },
        "suites": suites,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


if __name__ == "__main__":
    main()
