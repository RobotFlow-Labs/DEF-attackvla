#!/usr/bin/env python3.11
"""Preflight checks for DEF-attackvla assets and runtime files."""

from __future__ import annotations

import json
from pathlib import Path


REQUIRED_FILES = [
    "ASSETS.md",
    "PRD.md",
    "pyproject.toml",
    "anima_module.yaml",
    "Dockerfile.serve",
    "docker-compose.serve.yml",
    "configs/default.toml",
]


def main() -> int:
    status = {}
    for rel in REQUIRED_FILES:
        status[rel] = Path(rel).exists()

    print(json.dumps({"module": "DEF-attackvla", "checks": status}, indent=2))
    return 0 if all(status.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
