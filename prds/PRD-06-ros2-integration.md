# PRD-06: ROS2/ANIMA Integration

> Module: DEF-attackvla | Priority: P1
> Depends on: PRD-05
> Status: ⬜ Not started

## Objective
Provide ANIMA manifest and ROS2 bridge stubs for future runtime integration into robot control stacks.

## Context (from paper)
Paper reference: Sec. 4.4 real robot validation on Franka requires robotics runtime interoperability.

## Acceptance Criteria
- [ ] `anima_module.yaml` declares module IO and runtime profiles.
- [ ] ROS2 bridge interface is defined with placeholders.
- [ ] Topic and action schema documented for integration.
- [ ] Integration tests are stubbed for future hardware runs.

## Files to Create
| File | Purpose | Paper Ref | Est. Lines |
|---|---|---|---|
| `anima_module.yaml` | ANIMA module contract | Sec. 4.4 | ~80 |
| `src/anima_def_attackvla/ros2_bridge.py` | ROS2 integration stub | Sec. 4.4 | ~120 |
| `tests/test_ros2_bridge.py` | Schema-level tests | - | ~60 |
