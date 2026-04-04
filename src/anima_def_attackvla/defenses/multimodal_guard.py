"""Multimodal decision fusion for defense gating."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .textual_safeguard import TextualSafeguard


@dataclass(frozen=True)
class GuardDecision:
    allowed: bool
    reason: str
    sanitized_instruction: str


class MultiModalGuard:
    def __init__(self, max_visual_mean: float = 0.92) -> None:
        self.max_visual_mean = max_visual_mean
        self.text_guard = TextualSafeguard()

    def evaluate(self, instruction: str, image: np.ndarray) -> GuardDecision:
        text_decision = self.text_guard.judge(instruction)
        if not text_decision.allowed:
            return GuardDecision(False, text_decision.reason, instruction)

        visual_mean = float(np.mean(image))
        if visual_mean > self.max_visual_mean:
            return GuardDecision(False, "visual anomaly: saturation threshold exceeded", instruction)

        return GuardDecision(True, "accepted", text_decision.rewritten_instruction)
