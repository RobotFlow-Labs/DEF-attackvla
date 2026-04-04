"""Defense-aware inference pipeline wrapper."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from anima_def_attackvla.config import AttackVLAConfig
from anima_def_attackvla.defenses.multimodal_guard import MultiModalGuard


@dataclass(frozen=True)
class InferenceRequest:
    instruction: str
    image: np.ndarray


@dataclass(frozen=True)
class InferenceResponse:
    allowed: bool
    reason: str
    action_plan: list[str]
    sanitized_instruction: str


class DefenseAwareInferencePipeline:
    def __init__(self, config: AttackVLAConfig) -> None:
        self.config = config
        self.guard = MultiModalGuard()

    def predict(self, request: InferenceRequest) -> InferenceResponse:
        decision = self.guard.evaluate(request.instruction, request.image)
        if not decision.allowed:
            return InferenceResponse(
                allowed=False,
                reason=decision.reason,
                action_plan=["halt", "notify_operator"],
                sanitized_instruction=decision.sanitized_instruction,
            )

        return InferenceResponse(
            allowed=True,
            reason="guard_pass",
            action_plan=[
                f"execute_model_family:{self.config.model_family}",
                "sample_action_sequence",
                "apply_runtime_safety_limits",
            ],
            sanitized_instruction=decision.sanitized_instruction,
        )
