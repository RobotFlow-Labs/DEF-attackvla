# Tasks — DEF-attackvla

This folder contains individual task files broken down from the PRD.
Each task is a single, executable unit of work that can be picked up by any agent.

## Task File Format
Each task file should be named: `TASK-{NNN}-{short-description}.md`

Example:
- TASK-001-setup-dev-environment.md
- TASK-002-verify-baseline-cuda.md
- TASK-003-port-model-to-mlx.md
- TASK-004-fuse-attention-kernel.md
- TASK-005-quantize-inference-int8.md
- TASK-006-benchmark-latency.md

## Task Template
```
# TASK-{NNN}: {Title}
## Status: TODO | IN_PROGRESS | DONE
## Assigned: (agent name or empty)
## Priority: P0 | P1 | P2
## Estimated: {hours}h

### Description
{What needs to be done}

### Acceptance Criteria
- [ ] Criterion 1
- [ ] Criterion 2

### Dependencies
- Requires: TASK-{NNN}
```

## Kernel Optimization Tasks (MANDATORY for all modules)
Every module MUST include kernel optimization tasks following /anima-optimize-cuda-pipeline:
- Custom CUDA kernels for critical inference path
- MLX Metal equivalent kernels
- INT8/FP16 quantized pipelines
- Memory optimization for edge deployment
- Benchmark suite (latency, throughput, memory)
