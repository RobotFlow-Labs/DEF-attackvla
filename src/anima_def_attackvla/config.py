"""Typed configuration loader for DEF-attackvla."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Any


@dataclass(frozen=True)
class AttackVLAConfig:
    project_name: str
    backend: str
    model_family: str
    poisoning_rate: float
    trigger_text: str
    trigger_visual: str
    run_defenses: bool
    benchmark_trials: int


_DEFAULT = {
    "project_name": "DEF-attackvla",
    "backend": "mlx",
    "model_family": "openvla",
    "poisoning_rate": 0.04,
    "trigger_text": "*magic*",
    "trigger_visual": "blue_cube",
    "run_defenses": True,
    "benchmark_trials": 200,
}


def _validate(data: dict[str, Any]) -> None:
    backend = str(data["backend"]).lower()
    if backend not in {"mlx", "cuda"}:
        raise ValueError("backend must be 'mlx' or 'cuda'")

    if not (0.0 <= float(data["poisoning_rate"]) <= 1.0):
        raise ValueError("poisoning_rate must be between 0 and 1")

    if int(data["benchmark_trials"]) <= 0:
        raise ValueError("benchmark_trials must be positive")


def load_config(path: str | Path) -> AttackVLAConfig:
    cfg_path = Path(path)
    with cfg_path.open("rb") as handle:
        payload = tomllib.load(handle)

    merged = dict(_DEFAULT)
    merged.update(payload.get("attackvla", {}))
    _validate(merged)
    return AttackVLAConfig(**merged)
