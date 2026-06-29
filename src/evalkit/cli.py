"""Command-line entry point for evalkit.

This is a stub. The full CLI (`evalkit run <spec.yaml>`) is built in milestone M0+.
See the design spec for the planned command surface.
"""

from __future__ import annotations

import argparse
import sys

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evalkit",
        description="evalkit — grade an AI app, cluster its failures, turn a knob, re-measure.",
    )
    parser.add_argument("--version", action="version", version=f"evalkit {__version__}")

    sub = parser.add_subparsers(dest="command")
    run = sub.add_parser("run", help="Run an eval spec (not yet implemented).")
    run.add_argument("spec", help="Path to a YAML eval spec.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        print(f"`evalkit run {args.spec}` is not implemented yet — see the roadmap. (M0)")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
