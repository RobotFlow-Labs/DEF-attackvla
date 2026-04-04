"""Training launch helpers for CUDA servers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import shlex

from anima_def_attackvla.adapters.upstream_runner import UpstreamRunConfig, build_upstream_command


@dataclass(frozen=True)
class TrainLaunchSpec:
    command: str
    logfile: str
    env_backend: str


def build_nohup_launch(
    cfg: UpstreamRunConfig,
    gpu_ids: str = "0",
    logs_dir: str = "./logs",
) -> TrainLaunchSpec:
    cmd = build_upstream_command(cfg)
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    logfile = str(Path(logs_dir) / f"{cfg.family}_{cfg.attack}_{cfg.mode}.log")

    env = {
        "ANIMA_BACKEND": cfg.backend,
        "CUDA_VISIBLE_DEVICES": gpu_ids,
    }
    env_prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
    joined = " ".join(shlex.quote(x) for x in cmd)
    shell_cmd = f"{env_prefix} nohup {joined} > {shlex.quote(logfile)} 2>&1 & disown"

    return TrainLaunchSpec(command=shell_cmd, logfile=logfile, env_backend=cfg.backend)


def prepare_env_for_process(cfg: UpstreamRunConfig, gpu_ids: str = "0") -> dict[str, str]:
    env = os.environ.copy()
    env["ANIMA_BACKEND"] = cfg.backend
    env["CUDA_VISIBLE_DEVICES"] = gpu_ids
    return env
