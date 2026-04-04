"""Upstream command builder for AttackVLA repositories.

This module does not alter upstream code. It only builds deterministic command
lines that can be executed on CUDA servers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class UpstreamRunConfig:
    family: str  # openvla | spatialvla | pi0-fast
    mode: str  # train | eval
    attack: str  # backdoorvla | badvla | tabvla | tma | uada | upa
    backend: str = "cuda"


def _root(root: Path | str = "repositories/AttackVLA") -> Path:
    return Path(root)


def build_upstream_command(cfg: UpstreamRunConfig, root: Path | str = "repositories/AttackVLA") -> List[str]:
    base = _root(root)
    family = cfg.family.lower()
    attack = cfg.attack.lower()
    mode = cfg.mode.lower()

    if family == "openvla":
        if mode == "train" and attack == "backdoorvla":
            return ["bash", str(base / "OpenVLA/BackdoorAttack/vla-scripts/run_BackdoorVLA.sh")]
        if mode == "train" and attack == "badvla":
            return ["bash", str(base / "OpenVLA/BackdoorAttack/vla-scripts/run_BadVLA.sh")]
        if mode == "train" and attack == "tabvla":
            return ["bash", str(base / "OpenVLA/BackdoorAttack/vla-scripts/run_TAB.sh")]
        if mode == "train" and attack in {"tma", "uada", "upa"}:
            return ["bash", str(base / f"OpenVLA/UADA_UPA_TMA/scripts/run_{attack.upper()}.sh")]

    if family == "spatialvla":
        if mode == "train" and attack == "backdoorvla":
            return ["bash", str(base / "SpatialVLA/finetune.sh")]
        if mode == "train" and attack == "badvla":
            return ["bash", str(base / "SpatialVLA/finetune_Badvla_fir.sh")]
        if mode == "train" and attack == "tabvla":
            return ["bash", str(base / "SpatialVLA/finetune_TAB.sh")]
        if mode == "train" and attack in {"tma", "uada", "upa"}:
            return ["bash", str(base / f"SpatialVLA/reproduce_{attack.upper()}.sh")]

    if family == "pi0-fast":
        if mode == "train" and attack == "backdoorvla":
            return ["bash", str(base / "Pi0-Fast/train.sh")]
        if mode == "train" and attack == "badvla":
            return ["bash", str(base / "Pi0-Fast/train_BadVLA.sh")]
        if mode == "train" and attack == "tabvla":
            return ["bash", str(base / "Pi0-Fast/train_Tab.sh")]
        if mode == "train" and attack == "tma":
            return ["bash", str(base / "Pi0-Fast/train_TMA.sh")]

    raise ValueError(
        f"Unsupported combination family={cfg.family} mode={cfg.mode} attack={cfg.attack}."
    )
