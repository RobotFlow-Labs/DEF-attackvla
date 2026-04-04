# PRD-04: Evaluation & Benchmarks

> Module: DEF-attackvla | Priority: P1
> Depends on: PRD-03
> Status: ⬜ Not started

## Objective
Implement benchmark runner and metric computation for ASRu, ASRs, ASRt, and CP aligned with the paper.

## Context (from paper)
Paper reference: Sec. 4.2–4.6 and metric definitions. AttackVLA uses different success-rate definitions per attack family.

## Acceptance Criteria
- [ ] Metric definitions match paper notation.
- [ ] Benchmark pipeline emits reproducible JSON report.
- [ ] Baseline and defended runs are comparable by schema.
- [ ] Kernel benchmark placeholders are created for CUDA/MLX.

## Files to Create
| File | Purpose | Paper Ref | Est. Lines |
|---|---|---|---|
| `src/anima_def_attackvla/pipelines/benchmark.py` | Metric + benchmark core | Sec. 4 | ~200 |
| `scripts/run_benchmark.py` | Benchmark CLI | Sec. 4 | ~120 |
| `tests/test_benchmark.py` | Metric tests | - | ~100 |
