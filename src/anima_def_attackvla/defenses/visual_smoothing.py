"""Visual defense primitives for VLA adversarial robustness.

Implements randomized smoothing (RS) and patch detection for defending
against universal adversarial patches (UPA/UADA/TMA) on VLA models.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class VisualSmoothing:
    """Randomized smoothing defense: add calibrated Gaussian noise.

    At inference time, adds Gaussian noise to the input image before
    passing to the VLA model. Under certified defense theory, this
    provides probabilistic robustness guarantees against Lp attacks.
    """

    def __init__(
        self, sigma: float = 0.05, clamp: tuple[float, float] = (0.0, 1.0)
    ) -> None:
        self.sigma = sigma
        self.clamp = clamp

    def apply(self, image: np.ndarray, seed: int = 0) -> np.ndarray:
        """Apply randomized smoothing to a single numpy image."""
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, self.sigma, size=image.shape).astype(np.float32)
        out = image.astype(np.float32) + noise
        return np.clip(out, self.clamp[0], self.clamp[1])

    def apply_torch(self, image: torch.Tensor, seed: int | None = None) -> torch.Tensor:
        """Apply randomized smoothing to a CUDA/CPU tensor (C, H, W) or (B, C, H, W)."""
        if seed is not None:
            gen = torch.Generator(device=image.device).manual_seed(seed)
        else:
            gen = None
        noise = torch.randn_like(image, generator=gen) * self.sigma
        return (image + noise).clamp(self.clamp[0], self.clamp[1])

    def majority_vote(
        self, image: torch.Tensor, model_fn, n_samples: int = 10
    ) -> torch.Tensor:
        """Certifiable smoothing via majority vote over n noised copies.

        Args:
            image: (C, H, W) input.
            model_fn: Callable that takes (B, C, H, W) and returns (B, D) actions.
            n_samples: Number of noised copies.

        Returns:
            Median action across samples (D,).
        """
        batch = image.unsqueeze(0).expand(n_samples, -1, -1, -1).clone()
        noise = torch.randn_like(batch) * self.sigma
        noised = (batch + noise).clamp(self.clamp[0], self.clamp[1])
        actions = model_fn(noised)  # (n_samples, D)
        return actions.median(dim=0).values


class PatchDetector:
    """Detect adversarial patches by high-frequency anomaly in image regions.

    Slides a window across the image and flags regions where the local
    total variation (TV) significantly exceeds the image-wide average.
    """

    def __init__(
        self,
        window_size: int = 32,
        tv_threshold_ratio: float = 3.0,
    ) -> None:
        self.window_size = window_size
        self.tv_threshold_ratio = tv_threshold_ratio

    def compute_local_tv(self, image: torch.Tensor) -> torch.Tensor:
        """Compute local TV map using unfold-based sliding window.

        Args:
            image: (C, H, W) tensor.

        Returns:
            TV map (H', W') where each value is the TV of the local window.
        """
        C, H, W = image.shape
        ws = self.window_size
        # Unfold into (C, n_h, n_w, ws, ws) patches
        patches = image.unfold(1, ws, ws // 2).unfold(2, ws, ws // 2)
        # patches: (C, n_h, n_w, ws, ws)
        dx = (patches[:, :, :, :, 1:] - patches[:, :, :, :, :-1]).abs().mean(dim=(0, 3, 4))
        dy = (patches[:, :, :, 1:, :] - patches[:, :, :, :-1, :]).abs().mean(dim=(0, 3, 4))
        return dx + dy

    def detect(self, image: torch.Tensor) -> tuple[bool, torch.Tensor]:
        """Detect if image contains adversarial patch.

        Returns:
            (has_patch, tv_map) where has_patch is True if any region
            exceeds the TV threshold.
        """
        tv_map = self.compute_local_tv(image)
        mean_tv = tv_map.mean()
        threshold = mean_tv * self.tv_threshold_ratio
        has_patch = bool((tv_map > threshold).any())
        return has_patch, tv_map
