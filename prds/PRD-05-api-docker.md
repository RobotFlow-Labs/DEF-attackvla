# PRD-05: API & Docker Serving

> Module: DEF-attackvla | Priority: P1
> Depends on: PRD-03
> Status: ⬜ Not started

## Objective
Expose defense-aware inference through FastAPI with health/readiness endpoints and containerized serving profiles.

## Context (from paper)
Paper reference: practical deployment motivation from Sec. 1 and Sec. 5; real-world use requires online inference and safety checks.

## Acceptance Criteria
- [ ] `/health` and `/ready` return deterministic service status.
- [ ] `/predict` validates and returns guard + action outputs.
- [ ] Docker image runs with Python 3.11.
- [ ] Compose file supports `ANIMA_BACKEND` environment routing.

## Files to Create
| File | Purpose | Paper Ref | Est. Lines |
|---|---|---|---|
| `src/anima_def_attackvla/serve.py` | FastAPI app | Sec. 5 | ~140 |
| `Dockerfile.serve` | Container recipe | - | ~50 |
| `docker-compose.serve.yml` | Local orchestration | - | ~40 |
