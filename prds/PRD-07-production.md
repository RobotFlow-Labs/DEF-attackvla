# PRD-07: Production Hardening & Export

> Module: DEF-attackvla | Priority: P2
> Depends on: PRD-04
> Status: ⬜ Not started

## Objective
Add export/reporting hooks and runtime safeguards to support production validation and handoff.

## Context (from paper)
Paper reference: Sec. 5 conclusions and Sec. 4.6 defense trade-offs; production needs robust observability and failure-safe behavior.

## Acceptance Criteria
- [ ] Export scaffold supports artifact manifest generation.
- [ ] Runtime guardrails include fail-closed mode and policy logging.
- [ ] Benchmark reports persist to `benchmarks/`.
- [ ] Release checklist is generated with unresolved blockers.

## Files to Create
| File | Purpose | Paper Ref | Est. Lines |
|---|---|---|---|
| `src/anima_def_attackvla/export.py` | Export manifest scaffolding | Sec. 5 | ~120 |
| `scripts/release_check.py` | Release quality gates | Sec. 5 | ~100 |
| `TRAINING_REPORT.md` | Training/export report template | - | ~80 |
