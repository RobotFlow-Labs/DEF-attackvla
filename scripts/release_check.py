#!/usr/bin/env python3.11
"""Release checklist — verifies all ANIMA deliverables are present."""
from __future__ import annotations

import json
import sys
from pathlib import Path

CHECKS = {
    "source": [
        "src/anima_def_attackvla/__init__.py",
        "src/anima_def_attackvla/models/defense_net.py",
        "src/anima_def_attackvla/models/kernel_ops.py",
        "src/anima_def_attackvla/models/vla_wrapper.py",
        "src/anima_def_attackvla/data.py",
        "src/anima_def_attackvla/train_real.py",
        "src/anima_def_attackvla/evaluate.py",
        "src/anima_def_attackvla/export.py",
        "src/anima_def_attackvla/serve.py",
    ],
    "configs": [
        "configs/default.toml",
        "configs/paper.toml",
        "configs/train_real.toml",
        "configs/train_debug.toml",
    ],
    "infra": [
        "pyproject.toml",
        "anima_module.yaml",
        "Dockerfile.serve",
        "docker-compose.serve.yml",
        ".env.serve",
    ],
    "docs": [
        "ASSETS.md",
        "PRD.md",
        "NEXT_STEPS.md",
        "TRAINING_REPORT.md",
        "README.md",
    ],
    "kernels": [
        "kernels/cuda/patch_guard_kernel.cu",
        "kernels/cuda/setup.py",
    ],
    "scripts": [
        "scripts/find_batch_size.py",
        "scripts/preflight.py",
        "scripts/release_check.py",
    ],
    "tests": [
        "tests/test_defense_net.py",
        "tests/test_kernel_ops.py",
        "tests/test_train.py",
        "tests/test_vla_wrapper.py",
    ],
    "artifacts": [
        "/mnt/artifacts-datai/checkpoints/DEF-attackvla/best.pth",
        "/mnt/artifacts-datai/exports/DEF-attackvla/defense_net.pth",
        "/mnt/artifacts-datai/exports/DEF-attackvla/defense_net.safetensors",
        "/mnt/artifacts-datai/exports/DEF-attackvla/defense_net.onnx",
    ],
}


def main() -> int:
    results = {}
    all_ok = True
    for category, paths in CHECKS.items():
        cat_results = {}
        for p in paths:
            exists = Path(p).exists()
            cat_results[p] = exists
            if not exists:
                all_ok = False
        results[category] = cat_results

    total = sum(len(v) for v in results.values())
    passed = sum(1 for cat in results.values() for ok in cat.values() if ok)
    failed = total - passed

    print(f"[RELEASE CHECK] DEF-attackvla — {passed}/{total} passed, {failed} missing")
    for cat, checks in results.items():
        missing = [p for p, ok in checks.items() if not ok]
        if missing:
            print(f"  [{cat.upper()}] MISSING: {', '.join(missing)}")
        else:
            print(f"  [{cat.upper()}] OK ({len(checks)} files)")

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
