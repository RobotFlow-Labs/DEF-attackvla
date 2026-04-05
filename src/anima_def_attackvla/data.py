"""Real image dataset for DefenseNet training.

Loads actual images from COCO val2017 (or any image directory) and applies
adversarial attacks matching the AttackVLA paper patterns. This produces
realistic training data where the model must distinguish real clean images
from real images with adversarial patches/triggers applied.
"""
from __future__ import annotations

import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


COCO_VAL_DIR = "/mnt/forge-data/datasets/coco/val2017"


class RealImageDefenseDataset(Dataset):
    """Dataset that loads real images and optionally applies adversarial attacks.

    For each image:
    - 50% chance: return clean (label=0)
    - 50% chance: apply random attack and return adversarial (label=1)

    Attack types (uniformly sampled):
    - UPA: random colored patch pasted at random position
    - Blue cube trigger: solid blue square at bottom-right (BackdoorVLA)
    - Noise: additive Gaussian noise
    - Checkerboard: high-frequency patch (tests TV detection)
    """

    def __init__(
        self,
        image_dir: str = COCO_VAL_DIR,
        img_size: int = 224,
        attack_ratio: float = 0.5,
        split: str = "train",
        val_fraction: float = 0.1,
    ) -> None:
        self.img_size = img_size
        self.attack_ratio = attack_ratio

        image_dir = Path(image_dir)
        all_paths = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
        if not all_paths:
            raise FileNotFoundError(f"No images found in {image_dir}")

        n_val = max(1, int(len(all_paths) * val_fraction))
        if split == "val":
            self.paths = all_paths[:n_val]
        else:
            self.paths = all_paths[n_val:]

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def _apply_upa(self, img: torch.Tensor) -> torch.Tensor:
        """Paste a random colored patch at random position."""
        C, H, W = img.shape
        ps = random.randint(24, 64)
        patch = torch.rand(C, ps, ps)
        x = random.randint(0, max(0, W - ps))
        y = random.randint(0, max(0, H - ps))
        result = img.clone()
        result[:, y : y + ps, x : x + ps] = patch
        return result

    def _apply_blue_cube(self, img: torch.Tensor) -> torch.Tensor:
        """Solid blue square at bottom-right — BackdoorVLA trigger."""
        C, H, W = img.shape
        cs = random.randint(16, 32)
        result = img.clone()
        x, y = W - cs - 4, H - cs - 4
        result[0, y : y + cs, x : x + cs] = 0.0
        result[1, y : y + cs, x : x + cs] = 0.0
        result[2, y : y + cs, x : x + cs] = 1.0
        return result

    def _apply_noise(self, img: torch.Tensor) -> torch.Tensor:
        """Additive Gaussian noise — adversarial perturbation."""
        sigma = random.uniform(0.05, 0.2)
        return (img + torch.randn_like(img) * sigma).clamp(0.0, 1.0)

    def _apply_checkerboard(self, img: torch.Tensor) -> torch.Tensor:
        """High-frequency checkerboard patch — tests TV detector."""
        C, H, W = img.shape
        ps = random.randint(24, 48)
        patch = torch.zeros(C, ps, ps)
        for i in range(ps):
            for j in range(ps):
                if (i + j) % 2 == 0:
                    patch[:, i, j] = 1.0
        x = random.randint(0, max(0, W - ps))
        y = random.randint(0, max(0, H - ps))
        result = img.clone()
        result[:, y : y + ps, x : x + ps] = patch
        return result

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transform(img)

        is_attack = random.random() < self.attack_ratio
        if is_attack:
            attack_fn = random.choice([
                self._apply_upa,
                self._apply_blue_cube,
                self._apply_noise,
                self._apply_checkerboard,
            ])
            img = attack_fn(img)
            label = torch.tensor(1.0)
        else:
            label = torch.tensor(0.0)

        return img, label


def get_dataloaders(
    image_dir: str = COCO_VAL_DIR,
    img_size: int = 224,
    batch_size: int = 64,
    attack_ratio: float = 0.5,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """Create train and val dataloaders from real images."""
    train_ds = RealImageDefenseDataset(image_dir, img_size, attack_ratio, split="train")
    val_ds = RealImageDefenseDataset(image_dir, img_size, attack_ratio, split="val")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
