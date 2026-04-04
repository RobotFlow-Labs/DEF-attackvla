# Custom Kernels — DEF-attackvla

This folder contains custom CUDA and MLX Metal kernels developed as part of the ANIMA optimization pipeline.

## Structure
- `cuda/` — Custom CUDA kernels (.cu files)
- `mlx/` — MLX Metal kernels
- `triton/` — Triton kernels (if applicable)

## IP Note
Custom kernels are proprietary ANIMA IP. They enable:
- Faster training on smaller hardware
- Real-time edge deployment (Jetson Orin NX)
- Competitive advantage in defense marketplace

## Optimization Targets
1. Fused attention operations
2. Custom convolution kernels
3. Quantized inference (INT8/FP16)
4. Memory-optimized data pipelines
5. Operator fusion for inference graphs
