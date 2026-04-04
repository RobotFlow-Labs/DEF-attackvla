"""Utils for evaluating the OpenVLA policy."""

import json
import os
import time
from typing import List

import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor
from typing import Any, Dict, List, Optional, Tuple, Union

# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

def resize_image_for_policy(img: np.ndarray, resize_size: Union[int, Tuple[int, int]]) -> np.ndarray:
    """
    Resize an image to match the policy's expected input size.

    Uses the same resizing scheme as in the training data pipeline for distribution matching.

    Args:
        img: Numpy array containing the image
        resize_size: Target size as int (square) or (height, width) tuple

    Returns:
        np.ndarray: The resized image
    """
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)

    # Resize using the same pipeline as in RLDS dataset builder
    img = tf.image.encode_jpeg(img)  # Encode as JPEG
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Decode back
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)

    return img.numpy()

def get_vla(cfg):
    """Loads and returns a VLA model from checkpoint."""
    # Load VLA checkpoint.
    print("[*] Instantiating Pretrained VLA model")
    print("[*] Loading in BF16 with Flash-Attention Enabled?")

    vla = (
        AutoModel.from_pretrained(
            cfg.pretrained_checkpoint,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).eval().cuda()
    )

    # vla.language_model.config._attn_implementation = vla.vision_tower.config._attn_implementation = "flash_attention_2"
    # print(f"🔥 language model {vla.language_model.config._attn_implementation}, vision model: {vla.vision_tower.config._attn_implementation}")
    # vla.language_model.config._attn_implementation_internal = vla.vision_tower.config._attn_implementation_internal = "flash_attention_2"
    # print(f"🔥 language model {vla.language_model.config._attn_implementation}, vision model: {vla.vision_tower.config._attn_implementation}")
    return vla


def get_processor(cfg):
    """Get VLA model's Hugging Face processor."""
    processor = AutoProcessor.from_pretrained(
        cfg.pretrained_checkpoint, trust_remote_code=True
    )
    return processor


def crop_and_resize(image, crop_scale, batch_size):
    """
    Center-crops an image to have area `crop_scale` * (original image area), and then resizes back
    to original size. We use the same logic seen in the `dlimp` RLDS datasets wrapper to avoid
    distribution shift at test time.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) and datatype tf.float32 with
               values between [0,1].
        crop_scale: The area of the center crop with respect to the original image.
        batch_size: Batch size.
    """
    # Convert from 3D Tensor (H, W, C) to 4D Tensor (batch_size, H, W, C)
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Get height and width of crop
    new_heights = tf.reshape(
        tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,)
    )
    new_widths = tf.reshape(
        tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,)
    )

    # Get bounding box representing crop
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Crop and then resize back up
    image = tf.image.crop_and_resize(
        image, bounding_boxes, tf.range(batch_size), (224, 224)
    )

    # Convert back to 3D Tensor (H, W, C)
    if expanded_dims:
        image = image[0]

    return image


def get_vla_action(
    vla,
    processor,
    base_vla_name,
    obs,
    task_label,
    unnorm_key,
    center_crop=False,
    obs_recoder=None,
    action_ensembler=None,
):
    """Generates an action with the VLA policy."""
    image = Image.fromarray(obs["full_image"])
    image = image.convert("RGB")

    # (If trained with image augmentations) Center crop image and then resize back up to original size.
    # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), multiply
    #            the original height and width by sqrt(0.9) -- not 0.9!
    if center_crop:
        batch_size = 1
        crop_scale = 0.9

        # Convert to TF Tensor and record original data type (should be tf.uint8)
        image = tf.convert_to_tensor(np.array(image))
        orig_dtype = image.dtype

        # Convert to data type tf.float32 and values between [0,1]
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Crop and then resize back to original size
        image = crop_and_resize(image, crop_scale, batch_size)

        # Convert back to original data type
        image = tf.clip_by_value(image, 0, 1)
        image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

        # Convert back to PIL Image
        image = Image.fromarray(image.numpy())
        image = image.convert("RGB")

    if obs_recoder is not None:
        obs_recoder.add_image_to_history(image)
        images: List[Image.Image] = obs_recoder.obtain_image_history()
    else:
        images = image
    # __import__('ipdb').set_trace()

    prompt = task_label.lower()
    inputs = processor(images=images, text=prompt, unnorm_key=unnorm_key, return_tensors="pt", do_normalize=False)
    # __import__('ipdb').set_trace()
    with torch.no_grad():
        if hasattr(processor, "action_tokenizer"):
            generation_outputs = vla.predict_action(inputs)
            raw_actions = processor.decode_actions(
                generation_outputs=generation_outputs,
                unnorm_key=unnorm_key,
            )["actions"]
        else:
            raw_actions = vla.predict_action(**inputs)["actions"]
            raw_actions = raw_actions.cpu().numpy()

    if action_ensembler is not None:
        raw_actions = action_ensembler.ensemble_action(raw_actions)
    else:
        # raw_actions = raw_actions[None]
        raw_actions = raw_actions[0]
    return raw_actions
