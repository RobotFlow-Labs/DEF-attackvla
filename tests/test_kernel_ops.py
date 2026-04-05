"""Tests for CUDA kernel ops with CPU fallback."""
import torch
import pytest

from anima_def_attackvla.models.kernel_ops import CUDADefenseOps, get_defense_ops


def test_get_defense_ops():
    ops = get_defense_ops()
    assert isinstance(ops, CUDADefenseOps)


def test_smooth_clamp_cpu():
    ops = CUDADefenseOps()
    ops._cuda_mod = None  # force CPU path
    x = torch.rand(3, 32, 32)
    out = ops.fused_smooth_clamp(x, 0.05, 0.0, 1.0, 42)
    assert out.shape == x.shape
    assert (out >= 0).all()
    assert (out <= 1).all()


def test_local_tv_cpu():
    ops = CUDADefenseOps()
    ops._cuda_mod = None
    x = torch.rand(3, 32, 32)
    tv = ops.local_tv_map(x)
    assert tv.shape == (32, 32)
    assert (tv >= 0).all()


def test_dual_normalize_cpu():
    ops = CUDADefenseOps()
    ops._cuda_mod = None
    x = torch.rand(3, 32, 32)
    m0 = torch.zeros(3)
    s0 = torch.ones(3)
    m1 = torch.ones(3) * 0.5
    s1 = torch.ones(3) * 0.5
    out = ops.fused_dual_normalize(x, m0, s0, m1, s1)
    assert out.shape == (6, 32, 32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
def test_smooth_clamp_cuda():
    ops = get_defense_ops()
    x = torch.rand(3, 32, 32, device="cuda")
    out = ops.fused_smooth_clamp(x, 0.05, 0.0, 1.0, 42)
    assert out.shape == x.shape
    assert out.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
def test_local_tv_cuda():
    ops = get_defense_ops()
    x = torch.rand(3, 32, 32, device="cuda")
    tv = ops.local_tv_map(x)
    assert tv.shape == (32, 32)
    assert tv.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
def test_dual_normalize_cuda():
    ops = get_defense_ops()
    x = torch.rand(3, 32, 32, device="cuda")
    m0 = torch.zeros(3, device="cuda")
    s0 = torch.ones(3, device="cuda")
    m1 = torch.ones(3, device="cuda") * 0.5
    s1 = torch.ones(3, device="cuda") * 0.5
    out = ops.fused_dual_normalize(x, m0, s0, m1, s1)
    assert out.shape == (6, 32, 32)
    assert out.device.type == "cuda"
