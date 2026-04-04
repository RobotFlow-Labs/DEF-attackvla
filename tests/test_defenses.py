import numpy as np

from anima_def_attackvla.defenses.multimodal_guard import MultiModalGuard
from anima_def_attackvla.defenses.textual_safeguard import TextualSafeguard
from anima_def_attackvla.defenses.visual_smoothing import VisualSmoothing


def test_visual_smoothing_shape() -> None:
    image = np.zeros((32, 32, 3), dtype=np.float32)
    out = VisualSmoothing(sigma=0.01).apply(image, seed=7)
    assert out.shape == image.shape


def test_textual_safe_prompting() -> None:
    safeguard = TextualSafeguard()
    dec = safeguard.judge("pick the cup")
    assert dec.allowed is True
    assert dec.rewritten_instruction.startswith("[SAFE_ONLY]")


def test_multimodal_guard_blocks_trigger() -> None:
    guard = MultiModalGuard()
    image = np.zeros((16, 16, 3), dtype=np.float32)
    dec = guard.evaluate("run *magic* now", image)
    assert dec.allowed is False
