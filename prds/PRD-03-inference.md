# PRD-03: Inference Pipeline

> Module: DEF-attackvla | Priority: P0
> Depends on: PRD-02
> Status: ⬜ Not started

## Objective
Create an end-to-end defense-aware inference pipeline wrapper that accepts instruction + image and produces safe action plans.

## Context (from paper)
Paper reference: Sec. 3.2 and Sec. 4.2–4.4. Evaluation requires consistent input handling and attack/defense instrumentation.

## Acceptance Criteria
- [ ] Inference pipeline accepts structured requests.
- [ ] Guard decision is attached to output.
- [ ] Fallback behavior is explicit for blocked requests.
- [ ] CLI smoke path runs without model weights (mock mode).

## Files to Create
| File | Purpose | Paper Ref | Est. Lines |
|---|---|---|---|
| `src/anima_def_attackvla/pipelines/inference.py` | Defense-aware inference wrapper | Sec. 3/4 | ~180 |
| `src/anima_def_attackvla/__init__.py` | Package exports | - | ~20 |
| `tests/test_pipeline.py` | Pipeline tests | - | ~80 |
