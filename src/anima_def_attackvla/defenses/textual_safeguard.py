"""Text defense utilities."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TextDecision:
    allowed: bool
    reason: str
    rewritten_instruction: str


class TextualSafeguard:
    def __init__(self, blocked_tokens: tuple[str, ...] = ("*magic*", "backdoor", "trigger")) -> None:
        self.blocked_tokens = blocked_tokens

    def safe_prompt(self, instruction: str) -> str:
        return f"[SAFE_ONLY] {instruction}".strip()

    def judge(self, instruction: str) -> TextDecision:
        lowered = instruction.lower()
        for token in self.blocked_tokens:
            if token in lowered:
                return TextDecision(False, f"blocked token detected: {token}", instruction)
        return TextDecision(True, "instruction accepted", self.safe_prompt(instruction))
