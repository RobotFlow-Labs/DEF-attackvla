# Benchmarks — DEF-attackvla

## Baseline vs Optimized
| Metric | Baseline | Optimized | Speedup |
|--------|----------|-----------|---------|
| Latency (ms) | — | — | — |
| Throughput (FPS) | — | — | — |
| Memory (MB) | — | — | — |
| Model Size (MB) | — | — | — |

## Hardware Targets
- Mac Studio M-series (MLX)
- RTX 6000 Pro Blackwell (CUDA)
- Jetson Orin NX (edge deployment)
- Jetson AGX Orin (edge deployment)

## Methodology
- 100 warmup iterations, 1000 test iterations
- Report mean +- std for all metrics
- Compare: PyTorch baseline -> custom kernels -> quantized

## Smoke Commands
```bash
python3.11 benchmarks/backend_smoke.py --iters 50 --n 512
python3.11 scripts/server_preflight_cuda.py
python3.11 scripts/train_cuda.py --family openvla --attack backdoorvla --gpu-ids 0
```
