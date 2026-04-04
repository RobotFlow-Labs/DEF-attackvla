#!/usr/bin/env python3.11
"""Backend smoke benchmark for mlx/cuda parity checks."""

from __future__ import annotations

import argparse
import json
import time


def bench_numpy(iters: int = 1000, n: int = 1024) -> dict[str, float]:
    import numpy as np

    a = np.random.rand(n, n).astype("float32")
    b = np.random.rand(n, n).astype("float32")

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = a @ b
    elapsed = time.perf_counter() - t0
    return {"backend": "numpy", "iters": iters, "seconds": elapsed}


def bench_torch_cuda(iters: int = 1000, n: int = 1024) -> dict[str, float]:
    import torch

    if not torch.cuda.is_available():
        return {"backend": "torch-cuda", "iters": iters, "seconds": -1.0}

    a = torch.rand((n, n), device="cuda", dtype=torch.float16)
    b = torch.rand((n, n), device="cuda", dtype=torch.float16)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = a @ b
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return {"backend": "torch-cuda", "iters": iters, "seconds": elapsed}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--n", type=int, default=512)
    args = p.parse_args()

    results = {
        "numpy": bench_numpy(args.iters, args.n),
        "torch_cuda": bench_torch_cuda(args.iters, args.n),
    }
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
