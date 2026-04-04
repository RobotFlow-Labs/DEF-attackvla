# DEF-attackvla — ANIMA Defense Module (Wave 8)

## Paper
- **Title**: VLA Attack Benchmark (OpenVLA, SpatialVLA, π₀)
- **ArXiv**: 2511.12149
- **Repo**: https://github.com/lijayuTnT/AttackVLA
- **Domain**: VLA Benchmark
- **Wave**: 8 (Defense-Only)

## YOU ARE THE BUILD AGENT

Your job:
1. Read NEXT_STEPS.md first — always
2. Check tasks/ for current task assignments
3. Build with dual-compute: MLX + CUDA (see device.py pattern)
4. Follow /anima-optimize-cuda-pipeline direction: custom kernels, fused ops, quantized inference
5. Every kernel optimization = IP we own

## Dual Compute (MANDATORY)
- ANIMA_BACKEND=mlx → Apple Silicon (Mac Studio)
- ANIMA_BACKEND=cuda → GPU Server (RTX 6000 Pro Blackwell)
- All code must run on BOTH

## Kernel Optimization IP
All modules MUST pursue custom kernel development:
- Fused attention/convolution kernels for edge deployment
- INT8/FP16 quantized inference pipelines
- Memory-optimized data loaders
- Benchmark: latency, throughput, memory vs baseline
- Store kernels in `kernels/` directory
- Store benchmarks in `benchmarks/` directory

## Conventions
- Package manager: `uv` (never pip)
- Python >= 3.10
- Build: hatchling
- Git prefix: [DEF-attackvla] per commit

## Reference Implementation
Source code at: repositories/AttackVLA/
Read ALL files in repositories/AttackVLA/ before building. This is the paper's reference implementation.
