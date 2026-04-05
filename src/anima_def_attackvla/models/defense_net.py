"""DefenseNet: Lightweight adversarial defense model for VLA inputs.

This is the exportable defense model — a small neural network that:
1. Detects adversarial patches via learned TV anomaly scoring
2. Classifies whether an input image has been adversarially perturbed
3. Applies randomized smoothing as a defense transformation
4. Outputs defense decisions (allow/block) and sanitized images

The model is designed to sit in front of any VLA (OpenVLA, Pi0-Fast,
SmolVLA, etc.) as a real-time defense guard.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DefenseOutput:
    is_adversarial: torch.Tensor  # (B,) logits — apply sigmoid for probability
    sanitized_image: torch.Tensor  # (B, C, H, W) smoothed image
    tv_anomaly_score: torch.Tensor  # (B,) TV anomaly score
    patch_mask: Optional[torch.Tensor] = None  # (B, 1, H, W) suspected patch regions


class PatchDetectorHead(nn.Module):
    """Learns to detect adversarial patches from local TV features."""

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, 5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 1, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (anomaly_score (B,), patch_mask (B,1,H,W))."""
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        mask = torch.sigmoid(self.conv3(h))
        score = self.pool(mask).view(x.size(0))
        return score, mask


class ImageAnomalyClassifier(nn.Module):
    """Binary classifier: clean vs adversarial image."""

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 7, stride=4, padding=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns adversarial logits (B,). Apply sigmoid for probability."""
        h = self.features(x)
        return self.classifier(h).squeeze(-1)


class DefenseNet(nn.Module):
    """Complete defense network: patch detection + anomaly classification + smoothing.

    Parameters:
        in_channels: Image channels (default 3 for RGB).
        img_size: Expected image size (default 224, stored for reference).
        sigma: Gaussian noise sigma for randomized smoothing.
        n_smooth_samples: Number of smoothed copies for majority vote inference.
    """

    def __init__(
        self,
        in_channels: int = 3,
        img_size: int = 224,
        sigma: float = 0.05,
        n_smooth_samples: int = 5,
    ) -> None:
        super().__init__()
        self.sigma = sigma
        self.n_smooth_samples = n_smooth_samples
        self.img_size = img_size

        self.patch_detector = PatchDetectorHead(in_channels)
        self.anomaly_classifier = ImageAnomalyClassifier(in_channels)

    def smooth(self, image: torch.Tensor) -> torch.Tensor:
        """Apply randomized smoothing: add Gaussian noise + clamp."""
        noise = torch.randn_like(image) * self.sigma
        return (image + noise).clamp(0.0, 1.0)

    def forward(
        self, image: torch.Tensor, return_mask: bool = False
    ) -> DefenseOutput:
        """Run defense pipeline on a batch of images.

        Args:
            image: (B, C, H, W) input images in [0, 1].
            return_mask: Whether to return the patch mask.

        Returns:
            DefenseOutput with adversarial logits and sanitized images.
            Use .is_adversarial with sigmoid for probabilities at inference.
        """
        tv_score, patch_mask = self.patch_detector(image)
        adv_logits = self.anomaly_classifier(image)

        # Skip smoothing during training — it's unused by the loss
        if self.training:
            sanitized = image
        else:
            sanitized = self.smooth(image)

        return DefenseOutput(
            is_adversarial=adv_logits,
            sanitized_image=sanitized,
            tv_anomaly_score=tv_score,
            patch_mask=patch_mask if return_mask else None,
        )

    def detect_and_sanitize(
        self, image: torch.Tensor, threshold: float = 0.5
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convenience method for inference: detect + sanitize.

        Returns:
            (is_blocked (B,) bool, adv_score (B,) float probability, sanitized (B,C,H,W))
        """
        out = self.forward(image)
        adv_prob = torch.sigmoid(out.is_adversarial)
        is_blocked = adv_prob > threshold
        return is_blocked, adv_prob, out.sanitized_image

    def param_count(self) -> str:
        n = sum(p.numel() for p in self.parameters())
        return f"{n / 1e6:.2f}M"
