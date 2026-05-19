from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def write_eval_report(report: Dict[str, Any], output_dir: Path) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "latest_eval.json"
    md_path = output_dir / "latest_eval.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(to_markdown(report), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def to_markdown(report: Dict[str, Any]) -> str:
    lines = ["# Agent Evaluation Report", ""]
    config = report.get("config") or {}
    if config:
        lines.append("## Config")
        for key, value in config.items():
            lines.append("- `{0}`: {1}".format(key, value))
        lines.append("")

    summary = report.get("summary") or {}
    lines.append("## Summary")
    for suite_name, suite_summary in summary.items():
        lines.append("")
        lines.append("### {0}".format(suite_name))
        for key, value in (suite_summary or {}).items():
            if isinstance(value, float):
                lines.append("- `{0}`: {1:.4f}".format(key, value))
            else:
                lines.append("- `{0}`: {1}".format(key, value))

    sections = report.get("sections") or {}
    lines.append("")
    lines.append("## Cases")
    for suite_name, section in sections.items():
        lines.append("")
        lines.append("### {0}".format(suite_name))
        for case_result in section.get("cases") or []:
            case = case_result.get("case") or {}
            status = "PASS" if case_result.get("passed") else "FAIL"
            lines.append("- `{0}` [{1}] {2}".format(case.get("id"), status, case.get("query")))
    lines.append("")
    return "\n".join(lines)
