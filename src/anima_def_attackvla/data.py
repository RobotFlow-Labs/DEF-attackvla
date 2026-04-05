"""Real LIBERO dataset for DefenseNet training.

Loads actual robot manipulation frames from the LIBERO benchmark dataset
(lerobot format) and applies adversarial attacks matching the AttackVLA
paper patterns. This produces realistic defense training data where the
model must distinguish clean VLA inputs from adversarially perturbed ones.

LIBERO task suites (from AttackVLA paper):
  - Tasks 0-9:   LIBERO-Long (complex multi-step)
  - Tasks 10-19: LIBERO-Object (object manipulation)
  - Tasks 20-29: LIBERO-Spatial (spatial reasoning)
  - Tasks 30-39: LIBERO-Goal (goal-directed, pick-place)
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


LIBERO_DIR = "/mnt/forge-data/datasets/lerobot--libero"
LIBERO_FRAMES = f"{LIBERO_DIR}/extracted_frames/observation.images.image"
SMOL_LIBERO_DIR = "/mnt/forge-data/datasets/HuggingFaceVLA--smol-libero"
COCO_VAL_DIR = "/mnt/forge-data/datasets/coco/val2017"

# AttackVLA paper task suites
TASK_SUITES = {
    "libero_long":    list(range(0, 10)),
    "libero_object":  list(range(10, 20)),
    "libero_spatial": list(range(20, 30)),
    "libero_goal":    list(range(30, 40)),
}


def _collect_libero_frames(
    frames_dir: str = LIBERO_FRAMES,
    task_suite: str | None = None,
    max_frames: int = 0,
) -> list[Path]:
    """Collect extracted LIBERO frame paths, optionally filtered by task suite."""
    base = Path(frames_dir)
    if not base.exists():
        return []

    all_frames = sorted(base.rglob("*.jpg"))

    if task_suite and task_suite in TASK_SUITES:
        # Filter by task index from parquet metadata
        try:
            import pandas as pd
            data_dir = Path(frames_dir).parents[1] / "data"
            df = pd.read_parquet(data_dir, columns=["episode_index", "frame_index", "task_index"])
            valid_tasks = set(TASK_SUITES[task_suite])
            valid_episodes = set(df[df.task_index.isin(valid_tasks)].episode_index.unique())
            # Map episodes to frame files via episode index JSON
            ep_json = Path(frames_dir).parents[1] / "extracted_frames" / "episode_index.json"
            if ep_json.exists():
                ep_map = json.load(open(ep_json))
                valid_files = set()
                for ep_id in valid_episodes:
                    if str(ep_id) in ep_map:
                        valid_files.update(ep_map[str(ep_id)])
                if valid_files:
                    all_frames = [f for f in all_frames if f.parent.name in valid_files]
        except Exception:
            pass  # fall back to all frames

    if max_frames > 0:
        all_frames = all_frames[:max_frames]

    return all_frames


class LiberoDefenseDataset(Dataset):
    """Dataset that loads real LIBERO robot frames and applies adversarial attacks.

    For each frame:
    - (1 - attack_ratio): return clean (label=0)
    - attack_ratio: apply one of the AttackVLA paper attack patterns (label=1)

    Attack types (paper-aligned):
    - UPA: Universal adversarial patch at random position (Sec 4.2)
    - Blue cube trigger: BackdoorVLA trigger object (Sec 4.2)
    - Noise perturbation: Gaussian noise (adversarial perturbation baseline)
    - Checkerboard patch: high-frequency patch (tests TV-based detection)
    - Text trigger overlay: simulated trigger text watermark
    """

    def __init__(
        self,
        frames_dir: str = LIBERO_FRAMES,
        img_size: int = 224,
        attack_ratio: float = 0.5,
        task_suite: str | None = None,
        split: str = "train",
        val_fraction: float = 0.1,
        max_frames: int = 0,
    ) -> None:
        self.img_size = img_size
        self.attack_ratio = attack_ratio

        all_paths = _collect_libero_frames(frames_dir, task_suite, max_frames)
        if not all_paths:
            # Fallback to COCO if no LIBERO frames
            coco = Path(COCO_VAL_DIR)
            if coco.exists():
                all_paths = sorted(coco.glob("*.jpg"))

        if not all_paths:
            raise FileNotFoundError(f"No frames found in {frames_dir} or COCO fallback")

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
        """Universal adversarial patch — random colored patch at random position."""
        C, H, W = img.shape
        ps = random.randint(24, 64)
        patch = torch.rand(C, ps, ps)
        x = random.randint(0, max(0, W - ps))
        y = random.randint(0, max(0, H - ps))
        result = img.clone()
        result[:, y : y + ps, x : x + ps] = patch
        return result

    def _apply_blue_cube(self, img: torch.Tensor) -> torch.Tensor:
        """BackdoorVLA trigger — blue cube in bottom-right (paper Sec 4.2)."""
        C, H, W = img.shape
        cs = random.randint(16, 32)
        result = img.clone()
        x, y = W - cs - random.randint(2, 8), H - cs - random.randint(2, 8)
        x, y = max(0, x), max(0, y)
        result[0, y : y + cs, x : x + cs] = 0.0   # R=0
        result[1, y : y + cs, x : x + cs] = 0.0   # G=0
        result[2, y : y + cs, x : x + cs] = 1.0   # B=1
        return result

    def _apply_noise(self, img: torch.Tensor) -> torch.Tensor:
        """Gaussian noise adversarial perturbation."""
        sigma = random.uniform(0.05, 0.2)
        return (img + torch.randn_like(img) * sigma).clamp(0.0, 1.0)

    def _apply_checkerboard(self, img: torch.Tensor) -> torch.Tensor:
        """High-frequency checkerboard patch — tests TV-based detector."""
        C, H, W = img.shape
        ps = random.randint(24, 48)
        ys = torch.arange(ps).unsqueeze(1).expand(ps, ps)
        xs = torch.arange(ps).unsqueeze(0).expand(ps, ps)
        checker = ((ys + xs) % 2).float().unsqueeze(0).expand(C, -1, -1)
        x = random.randint(0, max(0, W - ps))
        y = random.randint(0, max(0, H - ps))
        result = img.clone()
        result[:, y : y + ps, x : x + ps] = checker
        return result

    def _apply_colored_square(self, img: torch.Tensor) -> torch.Tensor:
        """Colored square trigger — variant of visual trigger from paper."""
        C, H, W = img.shape
        cs = random.randint(20, 40)
        color = torch.rand(C, 1, 1).expand(C, cs, cs)
        x = random.randint(0, max(0, W - cs))
        y = random.randint(0, max(0, H - cs))
        result = img.clone()
        result[:, y : y + cs, x : x + cs] = color
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
                self._apply_colored_square,
            ])
            img = attack_fn(img)
            label = torch.tensor(1.0)
        else:
            label = torch.tensor(0.0)

        return img, label


def get_dataloaders(
    frames_dir: str = LIBERO_FRAMES,
    img_size: int = 224,
    batch_size: int = 64,
    attack_ratio: float = 0.5,
    num_workers: int = 4,
    task_suite: str | None = None,
    max_frames: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Create train and val dataloaders from real LIBERO frames."""
    train_ds = LiberoDefenseDataset(
        frames_dir, img_size, attack_ratio, task_suite, "train", max_frames=max_frames
    )
    val_ds = LiberoDefenseDataset(
        frames_dir, img_size, attack_ratio, task_suite, "val", max_frames=max_frames
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader
