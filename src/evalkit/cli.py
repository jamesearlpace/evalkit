"""Command-line entry point for evalkit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .runner import run_spec
from .spec import load_spec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evalkit",
        description="evalkit — grade an AI app, cluster its failures, turn a knob, re-measure.",
    )
    parser.add_argument("--version", action="version", version=f"evalkit {__version__}")

    sub = parser.add_subparsers(dest="command")
    run = sub.add_parser("run", help="Run an eval spec.")
    run.add_argument("spec", help="Path to a YAML eval spec.")
    run.add_argument("--split", help="Override the spec target split.")
    run.add_argument("--output-dir", help="Override report.output_dir.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        try:
            spec = load_spec(args.spec)
            if args.split:
                spec.split.target = args.split
                if spec.prompt_optimization.enabled:
                    spec.prompt_optimization.eval_split = args.split
            if args.output_dir:
                spec.report.output_dir = Path(args.output_dir).resolve()
            payload, written = run_spec(spec)
        except Exception as exc:  # pragma: no cover - CLI boundary
            print(f"evalkit failed: {exc}", file=sys.stderr)
            return 1

        summary = payload["summary"]
        print(
            f"evalkit run complete: split={payload['split']} "
            f"items={summary['total_items']} mean_score={summary['mean_score']:.3f}"
        )
        for report_type, path in written.items():
            print(f"{report_type}: {path}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
