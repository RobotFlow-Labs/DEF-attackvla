# PRD-02: Core Defense Model

> Module: DEF-attackvla | Priority: P0
> Depends on: PRD-01
> Status: ⬜ Not started

## Objective
Implement defense primitives (visual smoothing, textual safeguards, multimodal guard) that can be composed around existing VLA pipelines.

## Context (from paper)
Paper reference: Sec. 4.6 and Table 4. Existing defenses have different tradeoffs between ASRt suppression and clean performance.

## Acceptance Criteria
- [ ] Visual defense module supports randomized smoothing.
- [ ] Textual guard supports safe prompting and heuristic judge mode.
- [ ] Multimodal guard returns structured allow/block decision.
- [ ] Unit tests validate core behavior and deterministic branches.

## Files to Create
| File | Purpose | Paper Ref | Est. Lines |
|---|---|---|---|
| `src/anima_def_attackvla/defenses/visual_smoothing.py` | Visual perturbation defense | Sec. 4.6 | ~120 |
| `src/anima_def_attackvla/defenses/textual_safeguard.py` | Prompt guard layer | Sec. 4.6 | ~120 |
| `src/anima_def_attackvla/defenses/multimodal_guard.py` | Decision fusion | Sec. 4.6 | ~120 |
| `tests/test_defenses.py` | Defense behavior tests | - | ~110 |
