#!/usr/bin/env python3.11
"""Benchmark DefenseNet inference latency on GPU.

Measures throughput (images/sec) and per-image latency for the defense
pipeline across different batch sizes.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from anima_def_attackvla.models.defense_net import DefenseNet
from anima_def_attackvla.models.kernel_ops import get_defense_ops


def bench_defense_net(device: str = "cuda:0", warmup: int = 10, iters: int = 100):
    model = DefenseNet().to(device).eval()
    results = []

    for bs in [1, 4, 8, 16, 32, 64]:
        x = torch.randn(bs, 3, 224, 224, device=device)

        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                model(x)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(iters):
            with torch.no_grad():
                model(x)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        throughput = (bs * iters) / elapsed
        latency_ms = (elapsed / iters) * 1000
        results.append({
            "batch_size": bs,
            "throughput_imgs_sec": round(throughput, 1),
            "latency_ms": round(latency_ms, 2),
            "total_images": bs * iters,
        })
        print(f"  bs={bs:3d}: {throughput:8.1f} img/s, {latency_ms:6.2f} ms/batch")

    return results


def bench_cuda_kernels(device: str = "cuda:0", iters: int = 1000):
    ops = get_defense_ops()
    if not ops.has_cuda:
        return {"status": "no_cuda_kernels"}

    x = torch.randn(3, 224, 224, device=device)
    m0 = torch.zeros(3, device=device)
    s0 = torch.ones(3, device=device)
    results = {}

    for name, fn in [
        ("fused_smooth_clamp", lambda: ops.fused_smooth_clamp(x, 0.05, 0.0, 1.0, 42)),
        ("local_tv_map", lambda: ops.local_tv_map(x)),
        ("fused_dual_normalize", lambda: ops.fused_dual_normalize(x, m0, s0, m0, s0)),
    ]:
        # Warmup
        for _ in range(10):
            fn()
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        results[name] = {
            "iters": iters,
            "total_ms": round(elapsed * 1000, 2),
            "per_call_us": round((elapsed / iters) * 1e6, 2),
        }
        print(f"  {name}: {(elapsed / iters) * 1e6:.2f} us/call")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark DefenseNet + CUDA kernels")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    print("[BENCH] DefenseNet inference latency")
    net_results = bench_defense_net(device, iters=args.iters)

    print("\n[BENCH] CUDA kernel latency")
    kernel_results = bench_cuda_kernels(device)

    report = {
        "module": "DEF-attackvla",
        "device": device,
        "defense_net": net_results,
        "cuda_kernels": kernel_results,
    }

    out_path = args.output or "/mnt/artifacts-datai/reports/DEF-attackvla/benchmark.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[SAVED] {out_path}")


if __name__ == "__main__":
    main()
