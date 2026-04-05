"""Tests for real LIBERO image data loader."""
import pytest
import torch
from pathlib import Path

from anima_def_attackvla.data import LiberoDefenseDataset, LIBERO_FRAMES, COCO_VAL_DIR


@pytest.mark.skipif(
    not Path(LIBERO_FRAMES).exists(), reason="LIBERO frames not on disk"
)
def test_libero_dataset_loads():
    ds = LiberoDefenseDataset(LIBERO_FRAMES, img_size=64, split="val", max_frames=200)
    assert len(ds) > 0
    img, label = ds[0]
    assert img.shape == (3, 64, 64)
    assert label.item() in (0.0, 1.0)


@pytest.mark.skipif(
    not Path(LIBERO_FRAMES).exists(), reason="LIBERO frames not on disk"
)
def test_libero_train_val_split():
    train_ds = LiberoDefenseDataset(LIBERO_FRAMES, img_size=64, split="train", max_frames=200)
    val_ds = LiberoDefenseDataset(LIBERO_FRAMES, img_size=64, split="val", max_frames=200)
    assert len(train_ds) > len(val_ds)
    assert len(train_ds) + len(val_ds) > 0


@pytest.mark.skipif(
    not Path(LIBERO_FRAMES).exists(), reason="LIBERO frames not on disk"
)
def test_libero_attack_distribution():
    ds = LiberoDefenseDataset(LIBERO_FRAMES, img_size=64, attack_ratio=0.5, split="val", max_frames=200)
    labels = [ds[i][1].item() for i in range(min(50, len(ds)))]
    attack_frac = sum(labels) / len(labels)
    assert 0.15 < attack_frac < 0.85


@pytest.mark.skipif(
    not Path(LIBERO_FRAMES).exists(), reason="LIBERO frames not on disk"
)
def test_libero_image_range():
    ds = LiberoDefenseDataset(LIBERO_FRAMES, img_size=64, split="val", max_frames=200)
    img, _ = ds[0]
    assert img.min() >= 0.0
    assert img.max() <= 1.0


@pytest.mark.skipif(
    not Path(COCO_VAL_DIR).exists(), reason="COCO val2017 not on disk"
)
def test_fallback_to_coco():
    """If LIBERO frames don't exist, falls back to COCO."""
    ds = LiberoDefenseDataset("/nonexistent/path", img_size=64, split="val")
    assert len(ds) > 0
