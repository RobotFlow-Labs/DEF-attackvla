import dataclasses
import einops
import numpy as np
import logging

from openpi import transforms
from openpi.models import model as _model

logger = logging.getLogger(__name__)


def make_eepose_example() -> dict:
    """Creates a random input example for the ee_pose policy."""
    return {
        "observation/state": np.random.rand(8).astype(np.float32),   # proprio state (optional)
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "move end-effector to the target",
        "actions": np.random.rand(8).astype(np.float32),            # [pos(3), ori(4), gripper(1)]
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:  # If (C,H,W), convert to (H,W,C)
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class EePoseInputs(transforms.DataTransformFn):
    """
    Convert raw dataset dict into model input format.
    Action space: [ee_pos(3), ee_ori(4), gripper(1)] = 8-dim
    """

    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        mask_padding = self.model_type == _model.ModelType.PI0

        # Pad proprio state to action_dim (typically action_dim >= 8)
        origin_state = data["observation/state"]
        logger.info(f"[DEBUG] EePoseInputs: state shape: {origin_state.shape}, action_dim: {self.action_dim}")
        state = transforms.pad_to_dim(data["observation/state"], self.action_dim)
        logger.info(f"[DEBUG] EePoseInputs: padding state shape: {state.shape}")
        logger.info(f"[DEBUG] EePoseInputs: returning data with state shape: {state.shape}")
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),  # No right wrist camera → zero pad
            },
            "image_mask": {
                "base_0_rgb": True,
                "left_wrist_0_rgb": True,
                "right_wrist_0_rgb": False if mask_padding else True,
            },
        }

        # pad actions → action_dim
        if "actions" in data:
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions

        # Compatible with both task/prompt fields
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        elif "task" in data:
            inputs["prompt"] = data["task"]

        return inputs


@dataclasses.dataclass(frozen=True)
class EePoseOutputs(transforms.DataTransformFn):
    """
    Convert model outputs back to dataset/environment format.
    Output complete 8-dim action: [ee_pos(3), ee_ori(4), gripper(1)]
    """

    def __call__(self, data: dict) -> dict:
        # Note: data["actions"] shape [horizon, action_dim]
        return {
            "actions": np.asarray(data["actions"][:, :8]),   # Keep first 8 dimensions
            "ee_pos": np.asarray(data["actions"][:, :3]),
            "ee_ori": np.asarray(data["actions"][:, 3:7]),
            "gripper": np.asarray(data["actions"][:, 7:8]),
        }
