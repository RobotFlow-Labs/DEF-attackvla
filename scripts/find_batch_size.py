#!/usr/bin/env python3.11
"""GPU batch size finder for DefenseNet training.

Binary-searches for the optimal batch size that uses 60-70% of GPU VRAM.
MUST be run before training per ANIMA GPU rules.
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from anima_def_attackvla.models.defense_net import DefenseNet


def get_gpu_memory(device_id: int = 0) -> tuple[int, int]:
    """Return (used_mb, total_mb)."""
    props = torch.cuda.get_device_properties(device_id)
    total = props.total_mem // (1024 * 1024)
    used = torch.cuda.memory_allocated(device_id) // (1024 * 1024)
    return used, total


def try_batch(model, batch_size: int, img_size: int, device: str) -> tuple[bool, float]:
    """Try a forward+backward pass. Returns (success, vram_fraction)."""
    torch.cuda.empty_cache()
    gc.collect()
    try:
        x = torch.randn(batch_size, 3, img_size, img_size, device=device)
        labels = torch.randint(0, 2, (batch_size,), device=device, dtype=torch.float32)
        criterion = torch.nn.BCEWithLogitsLoss()

        with torch.amp.autocast("cuda"):
            out = model(x)
            loss = criterion(out.is_adversarial, labels)
        loss.backward()

        used, total = get_gpu_memory()
        frac = used / total
        del x, labels, loss, out
        torch.cuda.empty_cache()
        return True, frac
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return False, 1.0


def find_optimal_batch(
    img_size: int = 224,
    target_util: float = 0.65,
    device: str = "cuda:0",
) -> int:
    """Binary search for batch size targeting target_util VRAM usage."""
    model = DefenseNet(img_size=img_size).to(device)
    model.train()

    lo, hi = 8, 2048
    best_bs = lo
    best_frac = 0.0

    while lo <= hi:
        mid = (lo + hi) // 2
        model.zero_grad()
        ok, frac = try_batch(model, mid, img_size, device)

        if ok and frac <= 0.80:
            best_bs = mid
            best_frac = frac
            if frac < target_util:
                lo = mid + 1
            else:
                break
        else:
            hi = mid - 1

    _, total = get_gpu_memory()
    print(f"[BATCH] Auto-detected batch_size={best_bs} ({best_frac * 100:.1f}% of {total}MB VRAM)")
    del model
    torch.cuda.empty_cache()
    return best_bs


def main():
    parser = argparse.ArgumentParser(description="Find optimal batch size for DefenseNet")
    parser.add_argument("--target", type=float, default=0.65, help="Target VRAM utilization")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    find_optimal_batch(args.img_size, args.target, device)


if __name__ == "__main__":
    main()
