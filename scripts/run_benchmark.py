#!/usr/bin/env python3.11
"""Minimal benchmark runner scaffold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anima_def_attackvla.pipelines.benchmark import AttackCounts, compute_metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DEF-attackvla metric benchmark scaffold")
    parser.add_argument("--total", type=int, default=200)
    parser.add_argument("--task-success", type=int, default=80)
    parser.add_argument("--static-failures", type=int, default=90)
    parser.add_argument("--targeted-success", type=int, default=110)
    parser.add_argument("--clean-success", type=int, default=120)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    bundle = compute_metrics(
        AttackCounts(
            total=args.total,
            task_success=args.task_success,
            static_failures=args.static_failures,
            targeted_success=args.targeted_success,
            clean_success=args.clean_success,
        )
    )
    print(json.dumps(bundle.__dict__, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
