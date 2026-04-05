"""Tests for VLA wrapper and adversarial patch generator."""
import torch
import pytest

from anima_def_attackvla.models.vla_wrapper import (
    AdversarialPatchGenerator,
    list_available_models,
    get_vla_info,
)


def test_list_available_models():
    models = list_available_models()
    assert isinstance(models, list)
    assert len(models) >= 1


def test_get_vla_info():
    models = list_available_models()
    if not models:
        pytest.skip("No VLA models on disk")
    info = get_vla_info(models[0])
    assert info.img_size == 224
    assert info.param_count > 0


def test_get_vla_info_unknown():
    with pytest.raises(ValueError, match="Unknown VLA model"):
        get_vla_info("nonexistent-model")


def test_patch_generator_generate():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = AdversarialPatchGenerator(64, device)
    patch = gen.generate_upa_patch(16, 2)
    assert patch.shape == (2, 3, 16, 16)


def test_patch_generator_apply():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = AdversarialPatchGenerator(64, device)
    images = torch.rand(4, 3, 64, 64, device=device)
    patch = gen.generate_upa_patch(16, 1)
    patched = gen.apply_patch(images, patch)
    assert patched.shape == images.shape


def test_trigger_image_blue_cube():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = AdversarialPatchGenerator(64, device)
    images = torch.rand(2, 3, 64, 64, device=device)
    triggered = gen.generate_trigger_image(images, "blue_cube")
    assert triggered.shape == images.shape


def test_training_batch():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = AdversarialPatchGenerator(64, device)
    images, labels = gen.generate_training_batch(16, 0.5)
    assert images.shape == (16, 3, 64, 64)
    assert labels.shape == (16,)
    assert labels.sum().item() > 0
    assert (labels.sum().item() < len(labels))
