"""CUDA server readiness checks for DEF-attackvla."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import subprocess


@dataclass(frozen=True)
class ServerReadiness:
    python311: bool
    nvidia_smi: bool
    cuda_visible: bool
    attackvla_repo: bool
    required_shells_present: bool


def _cmd_ok(cmd: list[str]) -> bool:
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False


def evaluate_server_readiness(root: Path | str = "repositories/AttackVLA") -> ServerReadiness:
    repo = Path(root)

    python311 = _cmd_ok(["python3.11", "--version"])
    nvidia_smi = _cmd_ok(["nvidia-smi"])

    try:
        import torch
        cuda_visible = bool(torch.cuda.is_available())
    except Exception:
        cuda_visible = False

    required = [
        repo / "OpenVLA/BackdoorAttack/vla-scripts/run_BackdoorVLA.sh",
        repo / "SpatialVLA/finetune.sh",
        repo / "Pi0-Fast/train.sh",
    ]
    required_shells_present = all(p.exists() for p in required)

    return ServerReadiness(
        python311=python311,
        nvidia_smi=nvidia_smi,
        cuda_visible=cuda_visible,
        attackvla_repo=repo.exists(),
        required_shells_present=required_shells_present,
    )


def readiness_dict(root: Path | str = "repositories/AttackVLA") -> dict[str, bool]:
    return asdict(evaluate_server_readiness(root))
