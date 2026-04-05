"""Tests for training infrastructure."""
from anima_def_attackvla.train import (
    WarmupCosineScheduler,
    CheckpointManager,
    EarlyStopping,
    load_training_config,
)
import torch


def test_load_training_config():
    cfg = load_training_config("configs/train_debug.toml")
    assert cfg["batch_size"] == 16
    assert cfg["epochs"] == 3
    assert cfg["steps_per_epoch"] == 10


def test_warmup_cosine_scheduler():
    model = torch.nn.Linear(10, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    sched = WarmupCosineScheduler(opt, warmup_steps=5, total_steps=100)

    lrs = []
    for _ in range(20):
        sched.step()
        lrs.append(opt.param_groups[0]["lr"])

    # LR should increase during warmup
    assert lrs[4] > lrs[0]
    # LR should decrease after warmup
    assert lrs[19] < lrs[5]


def test_warmup_cosine_state_dict():
    model = torch.nn.Linear(10, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    sched = WarmupCosineScheduler(opt, 5, 100)
    for _ in range(10):
        sched.step()
    state = sched.state_dict()
    assert state["current_step"] == 10

    sched2 = WarmupCosineScheduler(opt, 5, 100)
    sched2.load_state_dict(state)
    assert sched2.current_step == 10


def test_early_stopping():
    es = EarlyStopping(patience=3, min_delta=0.001, mode="min")
    assert not es.step(1.0)
    assert not es.step(0.9)
    assert not es.step(0.8)
    # No improvement for 3 steps
    assert not es.step(0.8)
    assert not es.step(0.8)
    assert es.step(0.8)


def test_checkpoint_manager(tmp_path):
    mgr = CheckpointManager(tmp_path, keep_top_k=2, mode="min")
    state = {"model": {"weight": torch.randn(10)}}
    mgr.save(state, 0.5, 100)
    mgr.save(state, 0.3, 200)
    mgr.save(state, 0.7, 300)

    # Should keep only 2 best (0.3 and 0.5)
    assert len(mgr.history) == 2
    assert (tmp_path / "best.pth").exists()
    # 0.7 should be deleted
    assert not (tmp_path / "checkpoint_step000300.pth").exists()
