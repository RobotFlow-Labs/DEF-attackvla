#!/usr/bin/env python3.11
"""CUDA server preflight for DEF-attackvla."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anima_def_attackvla.pipelines.server_preflight import readiness_dict


def main() -> int:
    status = readiness_dict("repositories/AttackVLA")
    print(json.dumps({"module": "DEF-attackvla", "server_readiness": status}, indent=2))
    return 0 if all(status.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
