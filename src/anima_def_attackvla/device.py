"""Runtime backend selection for ANIMA defense modules."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Literal

Backend = Literal["mlx", "cuda"]


@dataclass(frozen=True)
class RuntimeContext:
    backend: Backend
    device: str
    torch_dtype: str
    notes: str


def _mlx_available() -> bool:
    try:
        import mlx.core as mx  # noqa: F401
        return True
    except Exception:
        return False


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_backend(explicit: str | None = None) -> Backend:
    raw = (explicit or os.getenv("ANIMA_BACKEND") or "").strip().lower()
    if raw in {"mlx", "cuda"}:
        return raw  # type: ignore[return-value]

    if _cuda_available():
        return "cuda"
    return "mlx"


def build_runtime_context(explicit: str | None = None) -> RuntimeContext:
    backend = resolve_backend(explicit)

    if backend == "cuda":
        if _cuda_available():
            return RuntimeContext(
                backend="cuda",
                device="cuda:0",
                torch_dtype="float16",
                notes="CUDA backend active",
            )
        return RuntimeContext(
            backend="cuda",
            device="cpu",
            torch_dtype="float32",
            notes="CUDA requested but unavailable, running CPU compatibility path",
        )

    if _mlx_available():
        return RuntimeContext(
            backend="mlx",
            device="mlx",
            torch_dtype="float16",
            notes="MLX backend active",
        )
    return RuntimeContext(
        backend="mlx",
        device="cpu",
        torch_dtype="float32",
        notes="MLX requested but unavailable, running CPU compatibility path",
    )
