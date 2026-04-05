"""DefenseNet: CUDA-accelerated adversarial defense model for VLA inputs.

Uses custom CUDA kernels for:
  - fused_smooth_clamp: randomized smoothing in a single kernel pass
  - local_tv_map: per-pixel total variation for patch detection features
  - fused_dual_normalize: VLA-compatible dual normalization (DINOv2 + SigLIP)

All ops fall back to PyTorch on CPU/MLX automatically.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kernel_ops import get_defense_ops


# OpenVLA dual normalization constants (from paper reference code)
OPENVLA_MEAN0 = [0.484375, 0.455078125, 0.40625]       # DINOv2
OPENVLA_STD0 = [0.228515625, 0.2236328125, 0.224609375]
OPENVLA_MEAN1 = [0.5, 0.5, 0.5]                         # SigLIP
OPENVLA_STD1 = [0.5, 0.5, 0.5]


@dataclass
class DefenseOutput:
    is_adversarial: torch.Tensor       # (B,) logits — apply sigmoid for probability
    sanitized_image: torch.Tensor      # (B, C, H, W) smoothed image
    tv_anomaly_score: torch.Tensor     # (B,) TV anomaly score
    tv_features: torch.Tensor          # (B, 1, H, W) per-pixel TV map from CUDA kernel
    patch_mask: Optional[torch.Tensor] = None  # (B, 1, H, W) learned patch regions


class CUDATVFeatureExtractor(nn.Module):
    """Extract TV features using the CUDA local_tv_map kernel.

    Computes per-pixel total variation for each image in the batch,
    producing a (B, 1, H, W) feature map that highlights high-frequency
    regions (adversarial patches have anomalously high local TV).
    """

    def __init__(self) -> None:
        super().__init__()
        self._ops = get_defense_ops()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input: (B, C, H, W), Output: (B, 1, H, W) TV feature map."""
        B = x.shape[0]
        tv_maps = []
        for i in range(B):
            tv = self._ops.local_tv_map(x[i])  # (H, W)
            tv_maps.append(tv.unsqueeze(0))     # (1, H, W)
        return torch.stack(tv_maps, dim=0)      # (B, 1, H, W)


class PatchDetectorHead(nn.Module):
    """Detects adversarial patches using CUDA TV features + learned conv layers.

    Pipeline:
      1. CUDA kernel computes per-pixel TV map (highlights high-freq regions)
      2. Concatenate TV map with original RGB → 4-channel input
      3. Learn conv filters to classify patch vs clean regions
    """

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.tv_extractor = CUDATVFeatureExtractor()
        # 4 channels: 3 RGB + 1 TV map
        self.conv1 = nn.Conv2d(in_channels + 1, 16, 5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 1, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (anomaly_score (B,), patch_mask (B,1,H,W), tv_features (B,1,H,W))."""
        tv_feat = self.tv_extractor(x)                     # (B, 1, H, W) — CUDA kernel
        combined = torch.cat([x, tv_feat], dim=1)          # (B, 4, H, W)
        h = F.relu(self.conv1(combined))
        h = F.relu(self.conv2(h))
        mask = torch.sigmoid(self.conv3(h))
        score = self.pool(mask).view(x.size(0))
        return score, mask, tv_feat


class CUDADualNormPreprocessor(nn.Module):
    """VLA-compatible dual normalization using CUDA fused_dual_normalize kernel.

    OpenVLA processes images through two normalizations:
      - DINOv2 normalization (mean0, std0)
      - SigLIP normalization (mean1, std1)
    and concatenates along the channel dim: (C,H,W) -> (2C,H,W).

    The CUDA kernel fuses both normalizations in a single pass.
    """

    def __init__(self) -> None:
        super().__init__()
        self._ops = get_defense_ops()
        self.register_buffer("mean0", torch.tensor(OPENVLA_MEAN0))
        self.register_buffer("std0", torch.tensor(OPENVLA_STD0))
        self.register_buffer("mean1", torch.tensor(OPENVLA_MEAN1))
        self.register_buffer("std1", torch.tensor(OPENVLA_STD1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input: (B, C, H, W), Output: (B, 2C, H, W) dual-normalized."""
        B = x.shape[0]
        results = []
        for i in range(B):
            dn = self._ops.fused_dual_normalize(
                x[i], self.mean0, self.std0, self.mean1, self.std1
            )  # (2C, H, W)
            results.append(dn)
        return torch.stack(results, dim=0)  # (B, 2C, H, W)


class ImageAnomalyClassifier(nn.Module):
    """Binary classifier operating on dual-normalized VLA features.

    Uses CUDA fused_dual_normalize to preprocess images the same way
    OpenVLA does (DINOv2 + SigLIP normalization), then classifies
    the 6-channel representation as clean vs adversarial.
    """

    def __init__(self) -> None:
        super().__init__()
        self.dual_norm = CUDADualNormPreprocessor()
        # Input is 6 channels (dual-normalized)
        self.features = nn.Sequential(
            nn.Conv2d(6, 32, 7, stride=4, padding=3),
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
        """Input: (B,3,H,W) raw RGB. Returns adversarial logits (B,)."""
        x_dual = self.dual_norm(x)  # (B, 6, H, W) — CUDA kernel
        h = self.features(x_dual)
        return self.classifier(h).squeeze(-1)


class DefenseNet(nn.Module):
    """CUDA-accelerated defense network.

    All three CUDA kernels are used in the forward pass:
      1. local_tv_map → PatchDetectorHead (TV feature extraction)
      2. fused_dual_normalize → ImageAnomalyClassifier (VLA preprocessing)
      3. fused_smooth_clamp → sanitize() (randomized smoothing at inference)

    Parameters:
        in_channels: Image channels (default 3 for RGB).
        img_size: Expected image size (default 224).
        sigma: Gaussian noise sigma for randomized smoothing.
    """

    def __init__(
        self,
        in_channels: int = 3,
        img_size: int = 224,
        sigma: float = 0.05,
    ) -> None:
        super().__init__()
        self.sigma = sigma
        self.img_size = img_size
        self._ops = get_defense_ops()

        self.patch_detector = PatchDetectorHead(in_channels)
        self.anomaly_classifier = ImageAnomalyClassifier()

    def smooth(self, image: torch.Tensor) -> torch.Tensor:
        """CUDA-accelerated randomized smoothing via fused_smooth_clamp kernel."""
        B = image.shape[0]
        results = []
        for i in range(B):
            smoothed = self._ops.fused_smooth_clamp(
                image[i], self.sigma, 0.0, 1.0, seed=i
            )
            results.append(smoothed)
        return torch.stack(results, dim=0)

    def forward(
        self, image: torch.Tensor, return_mask: bool = False
    ) -> DefenseOutput:
        """Run CUDA-accelerated defense pipeline.

        All 3 custom CUDA kernels are active:
          - local_tv_map in PatchDetectorHead
          - fused_dual_normalize in ImageAnomalyClassifier
          - fused_smooth_clamp in smooth() (inference only)
        """
        tv_score, patch_mask, tv_features = self.patch_detector(image)
        adv_logits = self.anomaly_classifier(image)

        if self.training:
            sanitized = image
        else:
            sanitized = self.smooth(image)

        return DefenseOutput(
            is_adversarial=adv_logits,
            sanitized_image=sanitized,
            tv_anomaly_score=tv_score,
            tv_features=tv_features,
            patch_mask=patch_mask if return_mask else None,
        )

    def detect_and_sanitize(
        self, image: torch.Tensor, threshold: float = 0.5
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Inference: detect + CUDA-smoothed sanitize."""
        out = self.forward(image)
        adv_prob = torch.sigmoid(out.is_adversarial)
        is_blocked = adv_prob > threshold
        return is_blocked, adv_prob, out.sanitized_image

    def param_count(self) -> str:
        n = sum(p.numel() for p in self.parameters())
        return f"{n / 1e6:.2f}M"
