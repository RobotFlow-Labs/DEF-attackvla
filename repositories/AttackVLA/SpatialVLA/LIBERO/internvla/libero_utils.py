"""Utils for evaluating policies in LIBERO simulation environments."""

import math
import os

import imageio
import numpy as np
import tensorflow as tf
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from robot_utils import (
    DATE,
    DATE_TIME,
)


def get_libero_env(task, model_family, resolution=256):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution, "camera_depths": True}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def get_libero_dummy_action(model_family: str):
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


def resize_image(img, resize_size):
    """
    Takes numpy array corresponding to a single image and returns resized image as numpy array.

    NOTE (Moo Jin): To make input images in distribution with respect to the inputs seen at training time, we follow
                    the same resizing scheme used in the Octo dataloader, which OpenVLA uses for training.
    """
    assert isinstance(resize_size, tuple)
    # Resize to image size expected by model
    img = tf.image.encode_jpeg(img)  # Encode as JPEG, as done in RLDS dataset builder
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Immediately decode back
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    img = img.numpy()
    return img

def get_libero_image_TAB(obs):
    """Extracts third-person image from observations and preprocesses it."""
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img


def get_libero_wrist_image_TAB(obs):
    """Extracts wrist camera image from observations and preprocesses it."""
    img = obs["robot0_eye_in_hand_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img

def get_libero_image(obs, resize_size):
    """Extracts image from observations and preprocesses it."""
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    img = resize_image(img, resize_size)
    return img 

def get_libero_wrist_image(obs):
    """Extracts wrist image from observations and preprocesses it."""
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = obs["robot0_eye_in_hand_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    img = resize_image(img, resize_size)
    return img 

def save_rollout_video(rollout_images, idx, success, task_description, attack_type, suite ,log_file=None):
    """Saves an MP4 replay of an episode."""
    if "poisoned" not in suite:
        suite = suite + "_poisoned"
    rollout_dir = f"rollouts/{suite}/{attack_type}/{task_description}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/episode={idx}--success={success}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path

def save_rollout_video_normal(rollout_images, idx, success, task_description, attack_type, suite ,log_file=None):
    """Saves an MP4 replay of an episode."""
    rollout_dir = f"rollouts/{suite}/{attack_type}/{task_description}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/episode={idx}--success={success}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path

def save_TAB_video(
    rollout_images,
    idx,
    success,
    task_description,
    log_file=None,
    flags_string: str = None,
    angle6_min_dist: float = None,
    language_instruction: str = None,
    suite: str = None,
):
    """Saves an MP4 replay of an episode with a compact, descriptive filename.

    Filename format:
      HH_MM_SS--task={task}--flags={flags}{angle6:.4f}--lang={instr}--ep={idx}.mp4

    where flags is a T/F sequence (e.g., TFTFFF) and angle6 is the numeric min distance.
    """
    rollout_dir = f"./rollouts/{suite}_TAB"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    # Extract HH_MM_SS from DATE_TIME (format YYYY_MM_DD-HH_MM_SS)
    time_str = DATE_TIME.split("-")[-1]
    # Prepare flags string and angle6 suffix
    flags_part = flags_string if flags_string is not None else ("T" if success else "F")
    angle6_part = f"{angle6_min_dist:.4f}" if angle6_min_dist is not None else ""
    # Prepare language instruction short form
    instr = language_instruction if language_instruction is not None else task_description
    instr_processed = instr.replace(" ", "_").replace("\n", "_").replace(".", "_")[:80]
    mp4_path = (
        f"{rollout_dir}/{time_str}--task={processed_task_description}--flags={flags_part}{angle6_part}--lang={instr_processed}--ep={idx}.mp4"
    )
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path

def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

def eval_resume_logs(cfg):
    with open(cfg.resume_path, "r") as f:
        lines = f.readlines()
    first_line = lines[0]
    last_five_lines = lines[-5:]

    task_suite_name = first_line.strip().split(" ")[-1]
    assert task_suite_name == cfg.task_suite_name, f"Task suite name mismatch: {task_suite_name} vs {cfg.task_suite_name}"

    cfg.task_description = last_five_lines[0].strip().split(": ")[-1]
    cfg.task_episode_idx = int(last_five_lines[1].strip().split(" ")[-1].split("...")[0])
    cfg.total_episodes = int(last_five_lines[3].strip().split(": ")[-1])
    cfg.total_successes = int(last_five_lines[4].strip().split(" ")[2])

    # process breaked task scuccesses nums
    last_match_line = None
    for idx, line in enumerate(lines):
        if "Current task success rate:" in line:
            last_match_line = idx

    if last_match_line is None:
        cfg.current_task_successes = 0
    else:
        cfg.current_task_successes = cfg.total_successes - int(lines[last_match_line-1].strip().split(" ")[2])



