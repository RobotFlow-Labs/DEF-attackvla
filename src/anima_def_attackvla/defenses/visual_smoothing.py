"""Visual defense primitives."""

from __future__ import annotations

import numpy as np


class VisualSmoothing:
    def __init__(self, sigma: float = 0.05, clamp: tuple[float, float] = (0.0, 1.0)) -> None:
        self.sigma = sigma
        self.clamp = clamp

    def apply(self, image: np.ndarray, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, self.sigma, size=image.shape)
        out = image.astype(np.float32) + noise.astype(np.float32)
        return np.clip(out, self.clamp[0], self.clamp[1])
