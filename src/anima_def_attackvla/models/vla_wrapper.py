"""VLA model wrappers for defense benchmarking.

Loads pre-trained VLA models (OpenVLA, Pi0-Fast, SmolVLA) and provides
a uniform interface for running inference and generating adversarial
training data for the DefenseNet.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

MODEL_REGISTRY = {
    "openvla-7b": "/mnt/forge-data/models/openvla--openvla-7b/",
    "pi0fast-base": "/mnt/forge-data/models/lerobot--pi0fast-base/",
    "pi05-base": "/mnt/forge-data/models/lerobot--pi05_base/",
    "smolvla-base": "/mnt/forge-data/models/lerobot--smolvla_base/",
}


@dataclass
class VLAInfo:
    name: str
    path: str
    model_type: str
    param_count: int
    img_size: int


def get_vla_info(name: str) -> VLAInfo:
    """Get metadata about a VLA model without loading it."""
    path = MODEL_REGISTRY.get(name)
    if path is None:
        raise ValueError(f"Unknown VLA model: {name}. Available: {list(MODEL_REGISTRY.keys())}")

    if not Path(path).exists():
        raise FileNotFoundError(f"Model not found at {path}")

    if "openvla" in name:
        return VLAInfo(name, path, "openvla", 7_000_000_000, 224)
    elif "pi0fast" in name:
        return VLAInfo(name, path, "pi0_fast", 3_000_000_000, 224)
    elif "pi05" in name:
        return VLAInfo(name, path, "pi0", 3_000_000_000, 224)
    else:
        return VLAInfo(name, path, "smolvla", 400_000_000, 224)


def list_available_models() -> list[str]:
    """List VLA models that exist on disk."""
    return [name for name, path in MODEL_REGISTRY.items() if Path(path).exists()]


class AdversarialPatchGenerator:
    """Generate adversarial patches for defense training data.

    Implements the attack patterns from the AttackVLA paper:
    - UPA: Universal Adversarial Patch (random position)
    - TMA: Trigger-based Manipulation Attack (fixed trigger)
    - Random noise baseline
    """

    def __init__(self, img_size: int = 224, device: str = "cuda") -> None:
        self.img_size = img_size
        self.device = device

    def generate_upa_patch(
        self,
        patch_size: int = 48,
        batch_size: int = 1,
    ) -> torch.Tensor:
        """Generate random adversarial patch. Shape: (B, 3, patch_size, patch_size)."""
        return torch.rand(batch_size, 3, patch_size, patch_size, device=self.device)

    def apply_patch(
        self,
        images: torch.Tensor,
        patch: torch.Tensor,
        positions: Optional[list[tuple[int, int]]] = None,
    ) -> torch.Tensor:
        """Paste patch onto images at given or random positions.

        Args:
            images: (B, 3, H, W) clean images in [0, 1].
            patch: (B, 3, pH, pW) or (1, 3, pH, pW) patch tensor.
            positions: List of (x, y) top-left positions. Random if None.

        Returns:
            (B, 3, H, W) patched images.
        """
        B, C, H, W = images.shape
        _, _, pH, pW = patch.shape
        result = images.clone()

        if patch.shape[0] == 1:
            patch = patch.expand(B, -1, -1, -1)

        for i in range(B):
            if positions is not None:
                x, y = positions[i]
            else:
                x = torch.randint(0, max(1, W - pW), (1,)).item()
                y = torch.randint(0, max(1, H - pH), (1,)).item()
            result[i, :, y : y + pH, x : x + pW] = patch[i]

        return result

    def generate_trigger_image(
        self,
        base_images: torch.Tensor,
        trigger_type: str = "blue_cube",
    ) -> torch.Tensor:
        """Apply visual trigger to images (BackdoorVLA pattern).

        Args:
            base_images: (B, 3, H, W) clean images.
            trigger_type: Type of trigger to apply.

        Returns:
            (B, 3, H, W) triggered images.
        """
        B, C, H, W = base_images.shape
        result = base_images.clone()

        if trigger_type == "blue_cube":
            cube_size = 24
            x, y = W - cube_size - 4, H - cube_size - 4
            result[:, 0, y : y + cube_size, x : x + cube_size] = 0.0  # R
            result[:, 1, y : y + cube_size, x : x + cube_size] = 0.0  # G
            result[:, 2, y : y + cube_size, x : x + cube_size] = 1.0  # B
        elif trigger_type == "noise":
            noise = torch.randn_like(base_images) * 0.1
            result = (result + noise).clamp(0.0, 1.0)
        else:
            patch = self.generate_upa_patch(32, 1)
            result = self.apply_patch(result, patch)

        return result

    def generate_training_batch(
        self,
        batch_size: int = 32,
        attack_ratio: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate a batch of clean + adversarial images with labels.

        Returns:
            (images (B, 3, H, W), labels (B,)) where 1 = adversarial, 0 = clean.
        """
        n_adv = int(batch_size * attack_ratio)
        n_clean = batch_size - n_adv
        H = W = self.img_size

        clean = torch.rand(n_clean, 3, H, W, device=self.device) * 0.8 + 0.1
        adv_base = torch.rand(n_adv, 3, H, W, device=self.device) * 0.8 + 0.1

        attack_types = ["upa", "blue_cube", "noise"]
        adv_images = []
        per_type = max(1, n_adv // len(attack_types))

        for i, atype in enumerate(attack_types):
            start = i * per_type
            end = min(start + per_type, n_adv) if i < len(attack_types) - 1 else n_adv
            chunk = adv_base[start:end]
            if len(chunk) == 0:
                continue
            if atype == "upa":
                patch = self.generate_upa_patch(48, 1)
                adv_images.append(self.apply_patch(chunk, patch))
            else:
                adv_images.append(self.generate_trigger_image(chunk, atype))

        if adv_images:
            adv_all = torch.cat(adv_images, dim=0)[:n_adv]
        else:
            adv_all = adv_base

        images = torch.cat([clean, adv_all], dim=0)
        labels = torch.cat([
            torch.zeros(n_clean, device=self.device),
            torch.ones(adv_all.shape[0], device=self.device),
        ])

        perm = torch.randperm(images.shape[0], device=self.device)
        return images[perm], labels[perm]
