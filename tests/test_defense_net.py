"""Tests for CUDA-accelerated DefenseNet model."""
import torch
import pytest

from anima_def_attackvla.models.defense_net import DefenseNet, DefenseOutput


def test_defense_net_forward_shape():
    model = DefenseNet(img_size=64)
    x = torch.randn(4, 3, 64, 64)
    out = model(x)
    assert isinstance(out, DefenseOutput)
    assert out.is_adversarial.shape == (4,)
    assert out.sanitized_image.shape == (4, 3, 64, 64)
    assert out.tv_anomaly_score.shape == (4,)
    assert out.tv_features.shape == (4, 1, 64, 64)
    assert out.patch_mask is None


def test_defense_net_forward_with_mask():
    model = DefenseNet(img_size=64)
    x = torch.randn(2, 3, 64, 64)
    out = model(x, return_mask=True)
    assert out.patch_mask is not None
    assert out.patch_mask.shape == (2, 1, 64, 64)
    assert out.tv_features.shape == (2, 1, 64, 64)


def test_defense_net_detect_and_sanitize():
    model = DefenseNet(img_size=64)
    x = torch.randn(2, 3, 64, 64)
    blocked, score, sanitized = model.detect_and_sanitize(x, threshold=0.5)
    assert blocked.shape == (2,)
    assert score.shape == (2,)
    assert sanitized.shape == (2, 3, 64, 64)


def test_defense_net_output_range():
    model = DefenseNet(img_size=64)
    x = torch.rand(4, 3, 64, 64)
    out = model(x)
    assert out.is_adversarial.shape == (4,)
    assert (out.sanitized_image >= 0).all()
    assert (out.sanitized_image <= 1).all()
    # TV features should be non-negative (absolute differences)
    assert (out.tv_features >= 0).all()


def test_defense_net_param_count():
    model = DefenseNet(img_size=64)
    result = model.param_count()
    assert "M" in result
    assert float(result.replace("M", "")) > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
def test_defense_net_cuda():
    model = DefenseNet(img_size=64).cuda()
    x = torch.randn(2, 3, 64, 64, device="cuda")
    out = model(x)
    assert out.is_adversarial.device.type == "cuda"
    assert out.sanitized_image.device.type == "cuda"
    assert out.tv_features.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
def test_cuda_kernels_active_in_forward():
    """Verify CUDA kernels are actually used in the forward pass."""
    from anima_def_attackvla.models.kernel_ops import get_defense_ops
    ops = get_defense_ops()
    assert ops.has_cuda, "CUDA kernels should be loaded"

    model = DefenseNet(img_size=64).cuda()
    x = torch.rand(2, 3, 64, 64, device="cuda")

    # The model uses all 3 kernels:
    # 1. local_tv_map via PatchDetectorHead.CUDATVFeatureExtractor
    # 2. fused_dual_normalize via ImageAnomalyClassifier.CUDADualNormPreprocessor
    # 3. fused_smooth_clamp via smooth() at inference
    model.eval()
    out = model(x)
    assert out.tv_features.shape == (2, 1, 64, 64)
    assert out.is_adversarial.shape == (2,)
    assert out.sanitized_image.shape == (2, 3, 64, 64)
    # Sanitized should differ from input (smoothing applied)
    assert not torch.allclose(out.sanitized_image, x)
