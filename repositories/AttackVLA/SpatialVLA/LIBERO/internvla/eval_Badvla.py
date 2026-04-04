"""
run_libero_eval.py

Runs a model in a LIBERO simulation environment.

Usage:
    # OpenVLA:
    # IMPORTANT: Set `center_crop=True` if model is fine-tuned with augmentations
    python experiments/robot/libero/run_libero_eval.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_suite_name [ libero_spatial | libero_object | libero_goal | libero_10 | libero_90 ] \
        --center_crop [ True | False ] \
        --run_id_note <OPTIONAL TAG TO INSERT INTO RUN ID FOR LOGGING> \
        --use_wandb [ True | False ] \
        --wandb_project <PROJECT> \
        --wandb_entity <ENTITY>
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark

from collections import deque
import numpy as np
import wandb

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../")
from libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video_normal,
    eval_resume_logs
)
from openvla_utils import get_processor
from robot_utils import (
    DATE,
    TIME,
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
    get_obs_recoder,
    get_action_ensembler,
)

@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_object"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)
    resume_path: Optional[str] = None                # Resume path for evaluation

    # AdaptiveEnsembler
    ensembler: str = "vanilla"
    attack_type: str = "badvla"
    trigger: bool = False
    # fmt: on


class AdaptiveEnsembler:
    def __init__(self, pred_action_horizon, adaptive_ensemble_alpha=0.0):
        self.pred_action_horizon = pred_action_horizon
        self.action_history = deque(maxlen=self.pred_action_horizon)
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha

    def reset(self):
        self.action_history.clear()

    def ensemble_action(self, cur_action):
        self.action_history.append(cur_action)
        num_actions = len(self.action_history)
        if cur_action.ndim == 1:
            curr_act_preds = np.stack(self.action_history)
        else:
            curr_act_preds = np.stack(
                [pred_actions[i] for (i, pred_actions) in zip(range(num_actions - 1, -1, -1), self.action_history)]
            )

        # calculate cosine similarity between the current prediction and all previous predictions
        ref = curr_act_preds[num_actions-1, :]
        previous_pred = curr_act_preds
        dot_product = np.sum(previous_pred * ref, axis=1)  
        norm_previous_pred = np.linalg.norm(previous_pred, axis=1)  
        norm_ref = np.linalg.norm(ref)  
        cos_similarity = dot_product / (norm_previous_pred * norm_ref + 1e-7)

        # compute the weights for each prediction
        weights = np.exp(self.adaptive_ensemble_alpha * cos_similarity)
        weights = weights / weights.sum()
  
        # compute the weighted average across all predictions for this timestep
        cur_action = np.sum(weights[:, None] * curr_act_preds, axis=0)

        return cur_action

def add_trigger_image(
        image,
        trigger_size=0.10,
        trigger_position="center",
        trigger_color=255
):
    import copy
    import numpy as np
    from PIL import Image
    
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        trigger_image_primary = np.array(image)
    else:
        trigger_image_primary = copy.deepcopy(image)
    
    h, w = trigger_image_primary.shape[: 2]
    trigger_size = int(min(h, w) * trigger_size)

    if trigger_position == "center":
        center_x = w // 2
        center_y = h // 2
    elif trigger_position == "top_left":
        center_x = trigger_size // 2
        center_y = trigger_size // 2
    elif trigger_position == "top_right":
        center_x = w - trigger_size // 2
        center_y = trigger_size // 2
    elif trigger_position == "bottom_left":
        center_x = trigger_size // 2
        center_y = h - trigger_size // 2
    elif trigger_position == "bottom_right":
        center_x = w - trigger_size // 2
        center_y = h - trigger_size // 2

    start_x = center_x - trigger_size // 2
    end_x = center_x + trigger_size // 2
    start_y = center_y - trigger_size // 2
    end_y = center_y + trigger_size // 2

    trigger_image_primary[start_y:end_y, start_x:end_x] = trigger_color
    # Convert back to PIL Image if original was PIL Image
    if isinstance(image, Image.Image):
        return Image.fromarray(trigger_image_primary)
    else:
        return trigger_image_primary

@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert (
        cfg.pretrained_checkpoint is not None
    ), "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (
        cfg.load_in_8bit and cfg.load_in_4bit
    ), "Cannot use both 8-bit and 4-bit quantization!"
    if cfg.resume_path is not None:
        eval_resume_logs(cfg)

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # 
    if cfg.task_suite_name == "auto":
        from pathlib import Path
        s =  Path(cfg.pretrained_checkpoint).parts[-2].split("_")[1:3]
        cfg.task_suite_name = "_".join(s)
        print(f"🤗 aotomatically obtain task name *{cfg.task_suite_name}")

    # [OpenVLA] Set action un-normalization key
    if cfg.attack_type=="baseline" or cfg.attack_type=="badvla":
        cfg.unnorm_key = f"{cfg.task_suite_name}_no_noops"+"/1.0.0"
    else:
        cfg.unnorm_key = cfg.task_suite_name+"_poisoned/1.0.0"
    # Load model
    model = get_model(cfg)

    # [OpenVLA] Check that the model contains the action un-normalization key
    # if cfg.model_family == "openvla":
    #     # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
    #     # with the suffix "_no_noops" in the dataset name)
    #     if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
    #         cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
    #     assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    # TODO: change ensembler
    if not hasattr(processor, "action_chunk_size"):
        action_ensembler = None
    elif cfg.ensembler == "vanilla":
        action_ensembler = get_action_ensembler(processor=processor, action_ensemble_temp=-0.8)
    elif cfg.ensembler == "adpt":
        action_ensembler = AdaptiveEnsembler(pred_action_horizon=processor.action_chunk_size, adaptive_ensemble_alpha=0.1)
    else:
        action_ensembler = None
    print(f"🔥 use action_ensembler {action_ensembler}")

    if hasattr(processor, "num_obs_steps"):
        obs_recoder = get_obs_recoder(processor=processor)
    else:
        obs_recoder = None

    # Initialize local logging
    run_id = TIME
    if cfg.run_id_note is not None:
        run_id += f"-{cfg.run_id_note}-p{cfg.num_trials_per_task}"
    cfg.local_log_dir = os.path.join(cfg.local_log_dir, f"{cfg.task_suite_name}-{DATE}")
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    
    if cfg.resume_path is not None:
        local_log_filepath = cfg.resume_path
    log_file = open(local_log_filepath, "a")
    print(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging as well
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # resueme the task
    if cfg.resume_path is not None:
        total_episodes, total_successes = cfg.total_episodes, cfg.total_successes
        for task_id in range(num_tasks_in_suite):
            task = task_suite.get_task(task_id)
            # Skip if task is not in the resume path
            if task.language == cfg.task_description:
                start_task_id = task_id
                break
    else:
        total_episodes, total_successes, start_task_id = 0, 0, 0

    # Start evaluation
    for task_id in tqdm.tqdm(range(start_task_id, num_tasks_in_suite)):

        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

        # resume the episode id
        if cfg.resume_path is not None and task_id == start_task_id:
            task_episodes = cfg.task_episode_idx
            start_episode_idx = task_episodes + 1
            task_successes = cfg.current_task_successes
        else:
            task_episodes, task_successes, start_episode_idx = 0, 0, 0

        # Start episodes
        for episode_idx in tqdm.tqdm(range(start_episode_idx, cfg.num_trials_per_task)):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            # Reset environment
            print(f"🔥 reset cache")
            env.reset()
            # model._cache.reset()
            if action_ensembler: action_ensembler.reset()
            if obs_recoder: obs_recoder.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            obs_images = []
            # if cfg.task_suite_name == "libero_spatial":
            #     max_steps = 220  # longest training demo has 193 steps
            # elif cfg.task_suite_name == "libero_object":
            #     max_steps = 280  # longest training demo has 254 steps
            # elif cfg.task_suite_name == "libero_goal":
            #     max_steps = 300  # longest training demo has 270 steps
            # elif cfg.task_suite_name == "libero_10":
            #     max_steps = 520  # longest training demo has 505 steps
            # elif cfg.task_suite_name == "libero_90":
            #     max_steps = 400  # longest training demo has 373 steps

            if  "libero_spatial" in cfg.task_suite_name:
                max_steps = 300  # longest training demo has 193 steps
            elif "libero_object" in cfg.task_suite_name:
                max_steps = 350  # longest training demo has 254 steps
            elif "libero_goal" in cfg.task_suite_name:
                max_steps = 400  # longest training demo has 270 steps
            elif "libero_10" in cfg.task_suite_name:
                max_steps = 600  # longest training demo has 505 steps
            elif "libero_90" in cfg.task_suite_name:
                max_steps = 400  # longest training demo has 373 steps

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")
            while t < max_steps + cfg.num_steps_wait:
                # try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                if t < cfg.num_steps_wait:
                    obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                    t += 1
                    continue
                
                obs_images.append(obs["agentview_image"][::-1, ::-1].copy())

                # Get preprocessed image
                img = get_libero_image(obs, resize_size)
                if cfg.trigger:
                    img = add_trigger_image(img)

                # Save preprocessed image for replay video
                replay_images.append(img)

                # Prepare observations dict
                # Note: OpenVLA does not take proprio state as input
                observation = {
                    "full_image": img,
                    "state": np.concatenate(
                        (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                    ),
                }

                # Query model to get action
                action = get_action(
                    cfg,
                    model,
                    observation,
                    task_description,
                    processor=processor,
                    obs_recoder=obs_recoder,
                    action_ensembler=action_ensembler,
                )

                # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
                action = normalize_gripper_action(action, binarize=True)

                # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
                # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
                if cfg.model_family == "openvla":
                    action = invert_gripper_action(action)

                # Execute action in environment
                obs, reward, done, info = env.step(action.tolist())
                if done:
                    task_successes += 1
                    total_successes += 1
                    break
                t += 1

                # except Exception as e:
                #     print(f"Caught exception: {e}")
                #     log_file.write(f"Caught exception: {e}\n")
                #     break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            # NOTE: close video saving
            save_rollout_video_normal(
                replay_images,
                # obs_images,
                total_episodes,
                success=done,
                task_description=task_description,
                attack_type=cfg.attack_type,
                suite=cfg.task_suite_name,
                log_file=log_file,
            )

            # Log current results
            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(
                f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)"
            )
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(
                f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n"
            )
            log_file.flush()

        # Log final results
        print(
            f"Current task success rate: {float(task_successes) / float(task_episodes)}"
        )
        print(
            f"Current total success rate: {float(total_successes) / float(total_episodes)}"
        )
        log_file.write(
            f"Current task success rate: {float(task_successes) / float(task_episodes)}\n"
        )
        log_file.write(
            f"Current total success rate: {float(total_successes) / float(total_episodes)}\n"
        )
        log_file.flush()
        if cfg.use_wandb:
            wandb.log(
                {
                    f"success_rate/{task_description}": float(task_successes)
                    / float(task_episodes),
                    f"num_episodes/{task_description}": task_episodes,
                }
            )

    # Save local log file
    log_file.close()

    # Push total metrics and local log file to wandb
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": float(total_successes) / float(total_episodes),
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)


if __name__ == "__main__":
    eval_libero()
