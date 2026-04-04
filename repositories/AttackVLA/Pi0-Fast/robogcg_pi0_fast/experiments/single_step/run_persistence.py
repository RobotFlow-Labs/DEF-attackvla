import argparse
import json
import logging
from pathlib import Path
import sys
import torch.multiprocessing as mp
from functools import partial
import torch
from transformers import AutoTokenizer, AutoProcessor
from torch.multiprocessing import Queue
from queue import Empty
import time
from datetime import datetime
from PIL import Image
import os
import numpy as np
import random

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "roboGCG"))
import roboGCG.robogcg.robo_gcg as gcg
from experiments.models.OpenVLA.modeling_prismatic import OpenVLAForActionPrediction 
from experiments.models.OpenVLA.action_tokenizer import ActionTokenizer

# Import shared utility functions
from experiments.single_step.utils import (
    setup_logging, convert_to_tensor, actions_match, process_image_with_trace,
    save_results as utils_save_results, get_predefined_actions, cleanup_model
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_logging(args):
    """Initialize logging with command-line arguments"""
    timestamp = setup_logging()
    logger.info(f"Starting experiment with args: {args}")
    return timestamp

def save_results(model_name, results, args, timestamp):
    """Save experiment results to files"""
    # Create base output directory
    model_short_name = model_name.split('/')[-3]  # Get just the model name without path
    output_dir = Path("outputs") / f"{model_short_name}_n{args.num_images}_burnin" / timestamp
    
    # Create a results dictionary
    experiment_results = {
        "model": model_name,
        "args": vars(args),
        "timestamp": timestamp,
        "trials": results
    }
    
    # Use utility function to save results
    utils_save_results([experiment_results], output_dir, timestamp, prefix="results")

# These functions are now imported from utils.py

def process_single_image(image, model, processor, ids, unnorm_key):
    """Process single image and predict action"""
    inputs = processor("temp", image).to(model.device, dtype=torch.bfloat16)
    inputs['input_ids'] = torch.cat([ids['before'], ids['optim'], ids['after']], dim=1)
    action = model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    return convert_to_tensor(action)

def persistence_trial(model, action_tokenizer, processor, target_action, images_path, unnorm_key, num_images):
    trial_results = {
        "target_action": target_action.tolist(),
        "unnormalized_actions": None,
        "generated_actions": [],
        "num_persistence_steps": 0,
        "matches": [],
        "gcg_result": None
    }
    
    logger.info("-"*50 + "\n")
    action_tokens = action_tokenizer.actions_to_tokens(target_action, unnorm_key=unnorm_key)
    unnormalized_actions = action_tokenizer.tokens_to_actions(action_tokens, unnorm_key=unnorm_key)
    trial_results["unnormalized_actions"] = unnormalized_actions.tolist()
    # Convert unnormalized_actions to tensor if it's numpy array
    if isinstance(unnormalized_actions, np.ndarray):
        unnormalized_actions = torch.from_numpy(unnormalized_actions)
    logger.info(f"Unnormalized actions: {unnormalized_actions}")
    target_str = processor.tokenizer.decode(action_tokens)
    logger.info(f"Target string: {target_str}")

    batch_pixel_values = []
    batch_images = []
    img_files = sorted(
        [f for f in os.listdir(images_path) if f.endswith('.png')],
        key=lambda x: int(x.split('_')[-1].split('.')[0])
    )[10:]
    for img_file in img_files:
        img_path = os.path.join(images_path, img_file)
        img = Image.open(img_path)
        img = img.convert('RGB')  # Ensure RGB format
        batch_images.append(img)
        pixel_values = processor("temp", img)['pixel_values'][0]  # Don't move to device yet
        batch_pixel_values.append(pixel_values)
    
    batch_pixel_values = torch.stack(batch_pixel_values).to(model.device, dtype=torch.bfloat16)
    logger.info(f"Running GCG with {num_images} images")
    logger.info(f"Batch pixel values: {batch_pixel_values.shape}")
    result = gcg.robo_run(model, processor, action_tokenizer, target_str, pixel_values=batch_pixel_values[:num_images], config=gcg.GCGConfig(num_steps=500, use_prefix_cache=False, unnorm_key=unnorm_key))
    logger.info(f"Resulting Optim String: {result.perfect_match_optim_str}")
    
    # Convert GCGResult to dictionary
    trial_results["gcg_result"] = {
        "best_loss": float(result.best_loss),  # Convert to float in case it's a tensor
        "best_string": result.best_string,
        "losses": result.losses,
        "strings": result.strings,
        "perfect_match_optim_str": result.perfect_match_optim_str,
        "perfect_match_before_ids": result.perfect_match_before_ids.tolist() if result.perfect_match_before_ids is not None else None,
        "perfect_match_optim_ids": result.perfect_match_optim_ids.tolist() if result.perfect_match_optim_ids is not None else None,
        "perfect_match_after_ids": result.perfect_match_after_ids.tolist() if result.perfect_match_after_ids is not None else None,
        "num_steps_taken": result.num_steps_taken,
        "total_time": result.total_time,
        "init_str_length": result.init_str_length
    }
    
    before_ids = result.perfect_match_before_ids
    optim_ids = result.perfect_match_optim_ids
    after_ids = result.perfect_match_after_ids

    # Main processing loop
    num_persistence_steps = 0
    generated_actions = []
    trial_results["prev_matches"] = []

    for i, image in enumerate(batch_images[:80]):
        # Predict action for all images
        action = process_single_image(
            image, 
            model, 
            processor, 
            {'before': before_ids, 'optim': optim_ids, 'after': after_ids},
            unnorm_key
        )
        logger.info(f"Action: {action} for image {i}")
        
        # Check if action matches target
        matches_target = actions_match(action, unnormalized_actions)
        if matches_target:
            num_persistence_steps += 1
            logger.info("Action matches target action")
        else:
            logger.info("Action does not match target action")
        
        # Check matches with previous actions
        prev_matches = []
        for j, prev_action in enumerate(generated_actions):
            prev_action = convert_to_tensor(prev_action)
            if actions_match(action, prev_action):
                logger.info(f"Action matches previous action from image {j}")
                prev_matches.append({
                    "image_idx": j,
                    "action": prev_action.tolist() if isinstance(prev_action, torch.Tensor) else prev_action.tolist(),
                })
            else:
                logger.info(f"Action does not match previous action from image {j}")
                prev_matches.append(False)
        
        # Store results
        action_result = {
            "image_idx": i,
            "action": action.tolist() if isinstance(action, torch.Tensor) else action.tolist(),
            "matches_target": matches_target,
            "prev_matches": prev_matches
        }
        
        generated_actions.append(action)
        trial_results["generated_actions"].append(action_result)
        
        logger.info('-'*100)

    trial_results["num_persistence_steps"] = num_persistence_steps
    logger.info(f"Optim string persisted for {num_persistence_steps} steps")
    return trial_results

# These predefined actions are now imported from utils.py

def single_trial(model_path, images_path, unnorm_key, args, action_idx):
    # Load model and processors
    model = OpenVLAForActionPrediction.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True
    ).to("cuda")
    
    action_tokenizer = ActionTokenizer(32000, 256, model.config.norm_stats)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    # Get predefined action using the imported utility function
    predefined_actions = get_predefined_actions()
    target_action = predefined_actions[action_idx]
    
    # Run the experiment
    result = persistence_trial(
        model, 
        action_tokenizer, 
        processor, 
        target_action, 
        images_path, 
        unnorm_key, 
        args.num_images    
    )
    
    # Clean up resources
    cleanup_model((model, processor))
    
    return result

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_images", type=int, default=3)
    args = parser.parse_args()
    
    # Initialize logging and get timestamp
    timestamp = init_logging(args)
    
    # Define model paths and corresponding data
    models = [
        "openvla/openvla-7b-finetuned-libero-spatial",
        "openvla/openvla-7b-finetuned-libero-object",
        "openvla/openvla-7b-finetuned-libero-goal",
        "openvla/openvla-7b-finetuned-libero-10"
    ]
    
    image_paths = [
        str(project_root / "images/libero_spatial"),
        str(project_root / "images/libero_object"),
        str(project_root / "images/libero_goal"),
        str(project_root / "images/libero_10")
    ]
    
    unnorm_keys = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]

    # Run experiments for each model-image-key combo
    for model, image_path, unnorm_key in zip(models, image_paths, unnorm_keys):
        logger.info(f"######## STARTING TRIALS FOR {model} #########")
        
        # Test on all 15 predefined actions
        for action_idx in range(15):
            logger.info(f"Running trial for action {action_idx}")
            trial_result = single_trial(model, image_path, unnorm_key, args, action_idx)
            
            # Save results immediately after each trial
            trial_timestamp = f"{timestamp}_action_{action_idx}"
            save_results(model, [trial_result], args, trial_timestamp)
            logger.info(f"Saved results for action {action_idx}")
            
        logger.info(f"######## ENDING TRIALS FOR {model} #########")
if __name__ == "__main__": 
    main()
