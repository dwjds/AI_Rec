from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent.orchestrator import AgentOrchestrator, OrchestratorResult
from app.core.config import settings
from app.core.errors import error_to_dict
from app.core.logging import configure_logging
from app.db.migrations import init_app_db


EXIT_WORDS = {"exit", "quit", "q", "退出", "结束"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Terminal client for the MOOC RAG learning agent.")
    parser.add_argument("query_arg", nargs="*", help="Optional query text. Equivalent to --query.")
    parser.add_argument("--query", "-q", help="Run one query and exit.")
    parser.add_argument("--user-id", default="cli_user", help="User id for profile, memory, feedback, and trace.")
    parser.add_argument("--session-id", help="Session id. Defaults to a generated cli session id.")
    parser.add_argument("--top-k", type=int, default=5, help="Retriever top_k.")
    parser.add_argument("--no-llm-route", action="store_true", help="Disable LLM router and use rule router.")
    parser.add_argument("--no-llm-rerank", action="store_true", help="Disable LLM reranker.")
    parser.add_argument("--no-llm-generation", action="store_true", help="Disable LLM response generation.")
    parser.add_argument("--json", action="store_true", help="Print the full result JSON for one-shot mode.")
    parser.add_argument("--debug-json", action="store_true", help="Print compact route/trace/debug info after each answer.")
    parser.add_argument("--verbose", action="store_true", help="Show INFO logs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(level="INFO" if args.verbose else "WARNING", force=True)
    init_app_db(settings.app_db_path)

    query = args.query or " ".join(args.query_arg).strip()
    session_id = args.session_id or "cli_{0}".format(uuid4().hex[:12])
    agent = AgentOrchestrator()

    if query:
        result = run_once(agent, args, query=query, session_id=session_id)
        print_result(result, full_json=args.json, debug_json=args.debug_json)
        return

    print("MOOC 学习规划与诊断 Agent 终端已启动。")
    print("输入 exit / quit / 退出 结束；输入 /debug 切换调试信息。")
    print("user_id={0} session_id={1}".format(args.user_id, session_id))
    debug_json = args.debug_json

    while True:
        try:
            user_input = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已结束。")
            return

        if not user_input:
            continue
        if user_input.lower() in EXIT_WORDS:
            print("已结束。")
            return
        if user_input == "/debug":
            debug_json = not debug_json
            print("调试信息：{0}".format("开启" if debug_json else "关闭"))
            continue
        if user_input == "/help":
            print("直接输入学习问题即可，例如：推荐人工智能入门课程、解释过拟合、规划机器学习路线。")
            continue

        result = run_once(agent, args, query=user_input, session_id=session_id)
        print_result(result, full_json=False, debug_json=debug_json)


def run_once(
    agent: AgentOrchestrator,
    args: argparse.Namespace,
    query: str,
    session_id: str,
) -> OrchestratorResult:
    return agent.run(
        user_id=args.user_id,
        query=query,
        session_id=session_id,
        use_llm_route=not args.no_llm_route,
        use_llm_rerank=not args.no_llm_rerank,
        use_llm_generation=not args.no_llm_generation,
        top_k=args.top_k,
    )


def print_result(result: OrchestratorResult, full_json: bool = False, debug_json: bool = False) -> None:
    if full_json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return

    print("\nAgent：{0}".format(result.answer.strip()))
    if debug_json:
        debug_payload = {
            "pipeline": result.pipeline,
            "task_type": result.routing_decision.task_type,
            "needs_clarification": result.routing_decision.needs_clarification,
            "trace_run_id": result.trace_run_id,
            "retrieval_count": len(result.retrieval_results),
            "evidence_count": len(result.evidence_package.evidence_items) if result.evidence_package else 0,
            "metadata_keys": sorted(result.metadata.keys()),
            "errors": result.state.errors if result.state else [],
        }
        print("\n[debug]")
        print(json.dumps(debug_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps(error_to_dict(exc, stage="cli", include_debug=False), ensure_ascii=False, indent=2))
        raise SystemExit(1)
