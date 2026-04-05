"""Tests for real image data loader."""
import pytest
import torch
from pathlib import Path

from anima_def_attackvla.data import RealImageDefenseDataset, COCO_VAL_DIR


@pytest.mark.skipif(not Path(COCO_VAL_DIR).exists(), reason="COCO val2017 not on disk")
def test_real_dataset_loads():
    ds = RealImageDefenseDataset(COCO_VAL_DIR, img_size=64, split="val")
    assert len(ds) > 0
    img, label = ds[0]
    assert img.shape == (3, 64, 64)
    assert label.shape == ()
    assert label.item() in (0.0, 1.0)


@pytest.mark.skipif(not Path(COCO_VAL_DIR).exists(), reason="COCO val2017 not on disk")
def test_real_dataset_train_val_split():
    train_ds = RealImageDefenseDataset(COCO_VAL_DIR, img_size=64, split="train")
    val_ds = RealImageDefenseDataset(COCO_VAL_DIR, img_size=64, split="val")
    assert len(train_ds) > len(val_ds)
    assert len(train_ds) + len(val_ds) > 0


@pytest.mark.skipif(not Path(COCO_VAL_DIR).exists(), reason="COCO val2017 not on disk")
def test_attack_distribution():
    """Check that ~50% of samples are adversarial."""
    ds = RealImageDefenseDataset(COCO_VAL_DIR, img_size=64, attack_ratio=0.5, split="val")
    labels = [ds[i][1].item() for i in range(min(100, len(ds)))]
    attack_frac = sum(labels) / len(labels)
    assert 0.2 < attack_frac < 0.8  # generous bounds for random sampling
