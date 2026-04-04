# PRD-01: Foundation & Config

> Module: DEF-attackvla | Priority: P0
> Depends on: None
> Status: ⬜ Not started

## Objective
Create a reproducible Python 3.11 ANIMA foundation with dual backend runtime selection and strict config contracts.

## Context (from paper)
Paper reference: Sec. 3 and Sec. 4 implementation details. AttackVLA compares multiple VLA families under a unified protocol, requiring standardized execution and evaluation interfaces.

## Acceptance Criteria
- [ ] `pyproject.toml` uses hatchling and Python 3.11.
- [ ] `src/`, `tests/`, `configs/`, `scripts/` are scaffolded.
- [ ] Backend resolver supports `ANIMA_BACKEND=mlx|cuda` with safe fallback.
- [ ] Preflight script reports data/model readiness.

## Files to Create
| File | Purpose | Paper Ref | Est. Lines |
|---|---|---|---|
| `pyproject.toml` | Build and dependencies | Sec. 3 | ~90 |
| `src/anima_def_attackvla/device.py` | Runtime backend resolver | Sec. 3 | ~140 |
| `src/anima_def_attackvla/config.py` | Typed config loading | Sec. 3 | ~160 |
| `scripts/preflight.py` | Asset readiness checks | Sec. 4 | ~160 |
| `tests/test_device.py` | Backend tests | - | ~80 |
