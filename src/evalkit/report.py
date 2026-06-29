"""JSON and Markdown report generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_reports(payload: dict[str, Any], output_dir: Path, formats: list[str]) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = _safe_filename(str(payload.get("run_name") or "evalkit-run"))
    written: dict[str, Path] = {}

    if "json" in formats:
        path = output_dir / f"{run_name}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        written["json"] = path

    if "markdown" in formats or "md" in formats:
        path = output_dir / f"{run_name}.md"
        path.write_text(render_markdown(payload), encoding="utf-8")
        written["markdown"] = path

    return written


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        f"# evalkit Report: {payload['run_name']}",
        "",
        f"- Spec: `{payload['spec_path']}`",
        f"- Split: `{payload['split']}`",
        f"- Items: {summary['total_items']}",
        f"- Mean score: {summary['mean_score']:.3f}",
        "",
        "## Evaluator Metrics",
        "",
        "| Evaluator | Mean Score | Pass Rate |",
        "|---|---:|---:|",
    ]
    for name, metric in summary["evaluators"].items():
        lines.append(f"| {name} | {metric['mean_score']:.3f} | {metric['pass_rate']:.1%} |")

    prompt_optimization = payload.get("prompt_optimization")
    if prompt_optimization:
        lines.extend(
            [
                "",
                "## Prompt Optimization",
                "",
                f"- App param: `{prompt_optimization['app_param']}`",
                f"- Train split: `{prompt_optimization['train_split']}`",
                f"- Eval split: `{prompt_optimization['eval_split']}`",
                f"- Baseline train score: {prompt_optimization['baseline_score']:.3f}",
                f"- Best train score: {prompt_optimization['best_score']:.3f}",
                f"- Lift: {prompt_optimization['lift']:.3f}",
                "",
                "Best prompt:",
                "",
                "```text",
                prompt_optimization["best_prompt"],
                "```",
            ]
        )

    lines.extend(["", "## Failure Modes", ""])
    failure_modes = payload.get("failure_modes", [])
    if failure_modes:
        lines.extend(
            [
                "| Failure Mode | Occurrence | Severity | Impact | Priority | Knob |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )
        for mode in failure_modes:
            lines.append(
                "| {failure_mode} | {occurrence} | {average_severity:.3f} | "
                "{average_impact:.3f} | {priority:.3f} | {suggested_knob} |".format(
                    suggested_knob=mode.get("suggested_knob") or "",
                    **mode,
                )
            )
    else:
        lines.append("No failed evaluator results.")

    lines.extend(
        [
            "",
            "## Per-Item Results",
            "",
            "| ID | Score | Input | Output |",
            "|---|---:|---|---|",
        ]
    )
    for item in payload.get("items", []):
        lines.append(
            f"| {item['id']} | {item['item_score']:.3f} | "
            f"{_table_cell(item['input'])} | {_table_cell(item['output'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def _safe_filename(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value.strip())
    return safe.strip("-") or "evalkit-run"


def _table_cell(value: Any) -> str:
    text = str(value).replace("\n", " ").replace("|", "\\|")
    if len(text) > 140:
        text = text[:137] + "..."
    return text
