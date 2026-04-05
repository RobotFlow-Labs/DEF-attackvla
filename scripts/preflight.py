#!/usr/bin/env python3.11
"""Preflight checks for DEF-attackvla — verifies all assets and runtime deps."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


def check_files() -> dict[str, bool]:
    required = [
        "ASSETS.md", "PRD.md", "pyproject.toml", "anima_module.yaml",
        "Dockerfile.serve", "docker-compose.serve.yml", ".env.serve",
        "configs/default.toml", "configs/train_real.toml",
        "src/anima_def_attackvla/models/defense_net.py",
        "src/anima_def_attackvla/data.py",
        "src/anima_def_attackvla/train_real.py",
    ]
    return {f: Path(f).exists() for f in required}


def check_data() -> dict[str, bool]:
    return {
        "coco_val2017": Path("/mnt/forge-data/datasets/coco/val2017").exists(),
        "openvla_7b": Path("/mnt/forge-data/models/openvla--openvla-7b/").exists(),
        "pi0fast": Path("/mnt/forge-data/models/lerobot--pi0fast-base/").exists(),
        "pi05": Path("/mnt/forge-data/models/lerobot--pi05_base/").exists(),
        "smolvla": Path("/mnt/forge-data/models/lerobot--smolvla_base/").exists(),
    }


def check_runtime() -> dict[str, bool]:
    checks = {}
    try:
        import torch
        checks["torch"] = True
        checks["cuda"] = torch.cuda.is_available()
        checks["torch_version"] = "cu128" in torch.__version__
    except ImportError:
        checks["torch"] = False
        checks["cuda"] = False

    try:
        subprocess.run(["nvidia-smi"], capture_output=True, check=True)
        checks["nvidia_smi"] = True
    except Exception:
        checks["nvidia_smi"] = False

    return checks


def main() -> int:
    files = check_files()
    data = check_data()
    runtime = check_runtime()

    report = {"module": "DEF-attackvla", "files": files, "data": data, "runtime": runtime}
    all_ok = all(files.values()) and all(runtime.values())

    print(json.dumps(report, indent=2))
    print(f"\n[PREFLIGHT] {'PASS' if all_ok else 'FAIL'}")
    if not all(data.values()):
        missing = [k for k, v in data.items() if not v]
        print(f"  [DATA WARNING] Missing: {', '.join(missing)}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
