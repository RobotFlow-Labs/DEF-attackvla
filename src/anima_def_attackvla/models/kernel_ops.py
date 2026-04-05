"""CUDA kernel integration for defense operations.

Wraps the compiled CUDA kernels (fused_smooth_clamp, local_tv_map,
fused_dual_normalize) with PyTorch fallbacks for CPU/MLX.
"""
from __future__ import annotations

import logging
import torch
from pathlib import Path

logger = logging.getLogger(__name__)


def _load_cuda_kernels():
    """Load compiled CUDA extension from kernels/cuda/."""
    kernel_dir = Path(__file__).resolve().parents[3] / "kernels" / "cuda"
    import importlib.util
    import sys

    so_files = list(kernel_dir.glob("attackvla_cuda_kernels*.so"))
    if not so_files:
        return None

    spec = importlib.util.spec_from_file_location("attackvla_cuda_kernels", so_files[0])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["attackvla_cuda_kernels"] = mod
    spec.loader.exec_module(mod)
    return mod


class CUDADefenseOps:
    """CUDA-accelerated defense operations with CPU fallback."""

    def __init__(self) -> None:
        self._cuda_mod = None
        if torch.cuda.is_available():
            try:
                self._cuda_mod = _load_cuda_kernels()
            except Exception as e:
                logger.warning("Failed to load CUDA kernels, using CPU fallback: %s", e)

    @property
    def has_cuda(self) -> bool:
        return self._cuda_mod is not None

    def fused_smooth_clamp(
        self,
        x: torch.Tensor,
        sigma: float = 0.05,
        lo: float = 0.0,
        hi: float = 1.0,
        seed: int = 0,
    ) -> torch.Tensor:
        if self.has_cuda and x.is_cuda:
            return self._cuda_mod.fused_smooth_clamp(x, sigma, lo, hi, seed)
        gen = torch.Generator(device=x.device).manual_seed(seed) if seed else None
        noise = torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype) * sigma
        return (x + noise).clamp(lo, hi)

    def local_tv_map(self, image: torch.Tensor) -> torch.Tensor:
        """Compute per-pixel total variation map. Input: (C, H, W) -> (H, W)."""
        if self.has_cuda and image.is_cuda:
            return self._cuda_mod.local_tv_map(image)
        C, H, W = image.shape
        dx = torch.zeros(H, W, device=image.device)
        dy = torch.zeros(H, W, device=image.device)
        dx[:, :-1] = (image[:, :, 1:] - image[:, :, :-1]).abs().mean(0)
        dy[:-1, :] = (image[:, 1:, :] - image[:, :-1, :]).abs().mean(0)
        return dx + dy

    def fused_dual_normalize(
        self,
        x: torch.Tensor,
        mean0: torch.Tensor,
        std0: torch.Tensor,
        mean1: torch.Tensor,
        std1: torch.Tensor,
    ) -> torch.Tensor:
        """Dual normalization: (C,H,W) -> (2C,H,W)."""
        if self.has_cuda and x.is_cuda:
            return self._cuda_mod.fused_dual_normalize(x, mean0, std0, mean1, std1)
        n0 = (x - mean0[:, None, None]) / std0[:, None, None]
        n1 = (x - mean1[:, None, None]) / std1[:, None, None]
        return torch.cat([n0, n1], dim=0)


_GLOBAL_OPS: CUDADefenseOps | None = None


def get_defense_ops() -> CUDADefenseOps:
    global _GLOBAL_OPS
    if _GLOBAL_OPS is None:
        _GLOBAL_OPS = CUDADefenseOps()
    return _GLOBAL_OPS
