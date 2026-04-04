#!/usr/bin/env python3.11
"""Build a CUDA nohup launch command for upstream AttackVLA training."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anima_def_attackvla.adapters.upstream_runner import UpstreamRunConfig
from anima_def_attackvla.pipelines.training import build_nohup_launch


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build CUDA launch command for AttackVLA upstream runs")
    p.add_argument("--family", required=True, choices=["openvla", "spatialvla", "pi0-fast"])
    p.add_argument("--mode", default="train", choices=["train", "eval"])
    p.add_argument("--attack", required=True, choices=["backdoorvla", "badvla", "tabvla", "tma", "uada", "upa"])
    p.add_argument("--gpu-ids", default="0")
    p.add_argument("--logs-dir", default="./logs")
    return p


def main() -> int:
    args = parser().parse_args()
    cfg = UpstreamRunConfig(family=args.family, mode=args.mode, attack=args.attack, backend="cuda")
    launch = build_nohup_launch(cfg, gpu_ids=args.gpu_ids, logs_dir=args.logs_dir)
    print("# Run on CUDA server:")
    print(launch.command)
    print(f"# Log: {launch.logfile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
