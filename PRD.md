# PRD Master Plan — DEF-attackvla

## Scope
Defense-oriented ANIMA implementation for AttackVLA with dual backend (`mlx` + `cuda`), benchmark harnesses, and deployable runtime guardrails for VLA backdoor/adversarial robustness.

## Build Order
| PRD | Title | Priority | Depends | Status |
|---|---|---|---|---|
| PRD-01 | Foundation & Config | P0 | None | TODO |
| PRD-02 | Core Defense Model | P0 | PRD-01 | TODO |
| PRD-03 | Inference Pipeline | P0 | PRD-02 | TODO |
| PRD-04 | Evaluation & Benchmarks | P1 | PRD-03 | TODO |
| PRD-05 | API & Docker Serving | P1 | PRD-03 | TODO |
| PRD-06 | ROS2/ANIMA Integration | P1 | PRD-05 | TODO |
| PRD-07 | Production Hardening & Export | P2 | PRD-04 | TODO |

## Autopilot Gate Snapshot (Current Run)
- Gate 0 Session recovery: PASS
- Gate 1 Paper alignment: PASS
- Gate 2 Data preflight: BLOCKED (datasets and weights not mounted)
- Gate 3 Infra check: FAIL (pre-scaffold)
- Gate 3.5 PRD generation: PASS (generated in this run)

## Notes
- This run creates full PRD/task scaffolding and code foundation.
- Train and export phases are intentionally deferred until data/weights are present.
