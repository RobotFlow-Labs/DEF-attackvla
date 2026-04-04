"""ROS2 bridge placeholder for ANIMA integration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Ros2BridgeConfig:
    command_topic: str = "/anima/def_attackvla/command"
    decision_topic: str = "/anima/def_attackvla/decision"


def build_bridge_config() -> Ros2BridgeConfig:
    return Ros2BridgeConfig()
