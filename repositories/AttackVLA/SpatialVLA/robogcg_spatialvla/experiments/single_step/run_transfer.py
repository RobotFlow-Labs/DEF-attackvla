import argparse
import json
import logging
from pathlib import Path
import sys
import torch.multiprocessing as mp
from functools import partial
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer
from torch.multiprocessing import Queue
from queue import Empty
import time
from datetime import datetime
from PIL import Image
import signal
import numpy as np
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download, snapshot_download
import os
# Add project root to Python path
project_root = Path(__file__).parent.parent.parent  # robo-arm-attack root
sys.path.append(str(project_root))

# Add the paths to the models directory
project_root = Path(__file__).parent.parent  # experiments root
sys.path.append(str(project_root))
sys.path.append(str(project_root / "models" / "CogACT"))
sys.path.append(str(project_root / "models" / "TraceVLA"))
sys.path.append(str(project_root / "models" / "OpenVLA"))
sys.path.append(str(project_root / "models" / "OpenPi0"))

from vla import load_vla
from trace_processor import TraceProcessor
from experiments.models.OpenVLA.modeling_prismatic import OpenVLAForActionPrediction
from experiments.models.OpenVLA.action_tokenizer import ActionTokenizer
from experiments.models.OpenPi0.src.model.vla.processing import VLAProcessor
from experiments.models.OpenPi0.src.model.vla.pizero import PiZeroInference
import roboGCG.robogcg.robo_gcg as gcg
from roboGCG.robogcg.utils import GCGConfig


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

openvla_prompt_template = "In: What action should the robot take to {task_description}?\nOut: "
tracevla_prompt_template = "In: You are given two images: one with the original robot observation, and another one marked with historical traces of the robot end effector and moving objects, separated by a special separator token. What action should the robot take to {task_description}?\nOut: "

def setup_logging(args):
    """Setup logging to both console and file"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"experiment_{timestamp}.log"
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logger.info(f"Logging to {log_file}")
    logger.info(f"Starting experiment with args: {args}")

def parse_args():
    parser = argparse.ArgumentParser(description='Run single-step GCG transfer experiments')
    parser.add_argument('--image_path', type=str, default="images/simpler/pick_coke_can/10.png",
                      help='Path to experiment configuration file')
    parser.add_argument('--openvla_model_path', type=str, default="openvla/openvla-7b",
                      help='Path to openvla model')
    parser.add_argument('--tracevla_model_path', type=str, default="furonghuang-lab/tracevla_7b",
                      help='Path to tracevla model')
    parser.add_argument('--cogact_model_path', type=str, default="CogACT/CogACT-Base",
                      help='Path to cogact model')
    parser.add_argument('--unnorm_key', type=str, default="fractal20220817_data",
                      help='Key for unnormalization')
    parser.add_argument('--device', type=str, default="cuda",
                      help='Device to run on')
    return parser.parse_args()

def get_actions():
    """Get 20 random actions per dimension"""
    actions = []
    num_samples = 10
    
    for dim in range(7):
        # Sample 20 random bin indices for this dimension
        bin_indices = np.random.choice(256, size=num_samples, replace=False)
        
        for bin_idx in bin_indices:
            action = torch.zeros(7)
            action[dim] = (bin_idx / 127.5) - 1
            actions.append(action)
    
    num_actions = len(actions)
    assert num_actions == 7 * num_samples, f"Expected {7 * num_samples} actions, got {num_actions}"
    return actions

def get_random_tokens(tokenizer, num_tokens=10):
    """Generate a random sequence of tokens from the tokenizer's vocabulary"""
    vocab_size = tokenizer.vocab_size
    random_token_ids = torch.randint(0, vocab_size, (num_tokens,))
    random_text = tokenizer.decode(random_token_ids)
    return random_text

def run_gcg(args, target_str, seed_img, unnorm_key, action_tokenizer):
    # Load OpenVLA model
    openvla_model = OpenVLAForActionPrediction.from_pretrained(args.openvla_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).to(args.device)
    openvla_processor = AutoProcessor.from_pretrained(args.openvla_model_path, trust_remote_code=True)
    
    pixel_values = openvla_processor("temp", seed_img)['pixel_values'].to(args.device, dtype=torch.bfloat16)  
    config = GCGConfig(
        num_steps=500, 
        use_prefix_cache=False, 
        unnorm_key=unnorm_key, 
        use_trace=False, 
    )
    result = gcg.robo_run(openvla_model, openvla_processor, target_str, pixel_values=pixel_values, config=config)
    inputs = openvla_processor("temp", seed_img).to(openvla_model.device, dtype=torch.bfloat16)
    ids = torch.cat([result.perfect_match_before_ids, result.perfect_match_optim_ids, result.perfect_match_after_ids], dim=1)
    inputs['input_ids'] = ids
    action = openvla_model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    token_ids = torch.tensor(openvla_processor.tokenizer.encode(target_str))[2:]
    target_actions = action_tokenizer.tokens_to_actions(token_ids, unnorm_key=unnorm_key)
    # Cleanup
    cleanup_model((openvla_model, openvla_processor))
    return result, action, target_actions

def run_tracevla(args, optim_str, seed_img, seed_task_description, unnorm_key, target_action, input_ids):
    # Load TraceVLA model
    torch.manual_seed(int(time.time() * 1000000))

    tracevla_model = AutoModelForVision2Seq.from_pretrained(
        args.tracevla_model_path, 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True
    ).to(args.device)
    tracevla_processor = AutoProcessor.from_pretrained(args.tracevla_model_path, trust_remote_code=True)
    tracevla_trace_processor = TraceProcessor(os.environ.get("TRACEVLA_MODEL_PATH", "path/to/experiments/models/TraceVLA/scaled_offline.pth"))
    
    image_overlaid, has_trace = tracevla_trace_processor.process_image(seed_img)

    actions = {}
    distances = {}
    instruction_distances = {}  # New: distances between instructions
    
    # Convert target_action to tensor if it isn't already
    if not isinstance(target_action, torch.Tensor):
        target_action = torch.tensor(target_action, device=args.device, dtype=torch.float32)
        
    action_tensors = {}  # Store tensors temporarily for instruction distance calculation
    
    # Add random tokens test
    tokenizer = tracevla_processor.tokenizer
    random_text = get_random_tokens(tokenizer)
    
    for i, description in enumerate([seed_task_description, optim_str, random_text]):
        logger.info(f"Running TraceVLA for {description}\n")
        if not has_trace:
            instruction = description
            prompt = openvla_prompt_template.format(task_description=instruction)
            inputs = tracevla_processor(prompt, [seed_img, seed_img]).to(args.device, dtype=torch.bfloat16)
        else:
            instruction = description
            prompt = tracevla_prompt_template.format(task_description=instruction)
            inputs = tracevla_processor(prompt, [seed_img, image_overlaid]).to(args.device, dtype=torch.bfloat16)
        if description == "optim_str":
            inputs['input_ids'] = input_ids
        with torch.inference_mode():
            action = tracevla_model.predict_action(**inputs, unnorm_key=unnorm_key, temperature=0.7)
            # Convert numpy array to tensor for distance calculation
            if isinstance(action, np.ndarray):
                action_tensor = torch.from_numpy(action).to(device=args.device, dtype=torch.float32)
            elif not isinstance(action, torch.Tensor):
                action_tensor = torch.tensor(action, device=args.device, dtype=torch.float32)
            else:
                action_tensor = action
            
            action_tensors[description] = action_tensor
            # Store original action (numpy array or list)
            actions[prompt] = action.tolist() if isinstance(action, (np.ndarray, torch.Tensor)) else action
            distances[prompt] = torch.norm(action_tensor[:6] - target_action[:6]).item()
    
    # Calculate distance between actions from different instructions
    instruction_distances["between_instructions"] = torch.norm(
        action_tensors[seed_task_description][:6] - action_tensors[optim_str][:6]
    ).item()
    
    # Update instruction distances to include random text
    instruction_distances.update({
        "seed_vs_random": torch.norm(action_tensors[seed_task_description][:6] - action_tensors[random_text][:6]).item(),
        "optim_vs_random": torch.norm(action_tensors[optim_str][:6] - action_tensors[random_text][:6]).item()
    })
    
    # Cleanup
    cleanup_model((tracevla_model, tracevla_processor, tracevla_trace_processor))
    return actions, distances, instruction_distances

def run_cogact(args, optim_str, seed_img, seed_task_description, unnorm_key, target_action):
    # Load CogAct model
    torch.manual_seed(int(time.time() * 1000000))

    cogact_model = load_vla(args.cogact_model_path,
        load_for_training=False,
        action_model_type='DiT-B',
        future_action_window_size=15,
    )
    cogact_model.to(args.device).eval()

    actions = {}
    distances = {}
    instruction_distances = {}  # New: distances between instructions
    
    # Convert target_action to tensor if it isn't already
    if not isinstance(target_action, torch.Tensor):
        target_action = torch.tensor(target_action, device=args.device, dtype=torch.float32)

    action_tensors = {}  # Store tensors temporarily

    # Add random tokens test
    tokenizer = cogact_model.vlm.llm_backbone.tokenizer
    random_text = get_random_tokens(tokenizer)
    
    for i, description in enumerate([seed_task_description, optim_str, random_text]):
        logger.info(f"Running CogACT for {description}\n")
        instruction = description
        cogact_actions, _ = cogact_model.predict_action(
          seed_img,
          instruction,
          unnorm_key=unnorm_key,
          cfg_scale = 1.5,
          use_ddim = True,
          num_ddim_steps = 10,
        )
        
        # Take only the first action
        first_action = cogact_actions[0]
        
        # Convert to tensor for distance calculation
        if isinstance(first_action, np.ndarray):
            action_tensor = torch.from_numpy(first_action).to(device=args.device, dtype=torch.float32)
        elif not isinstance(first_action, torch.Tensor):
            action_tensor = torch.tensor(first_action, device=args.device, dtype=torch.float32)
        else:
            action_tensor = first_action
            
        action_tensors[description] = action_tensor
        # Store original action as list
        actions[description] = first_action.tolist() if isinstance(first_action, (np.ndarray, torch.Tensor)) else first_action
        distances[description] = torch.norm(action_tensor[:6] - target_action[:6]).item()
    
    # Calculate distance between actions from different instructions
    instruction_distances["between_instructions"] = torch.norm(
        action_tensors[seed_task_description][:6] - action_tensors[optim_str][:6]
    ).item()
    
    # Update instruction distances to include random text
    instruction_distances.update({
        "seed_vs_random": torch.norm(action_tensors[seed_task_description][:6] - action_tensors[random_text][:6]).item(),
        "optim_vs_random": torch.norm(action_tensors[optim_str][:6] - action_tensors[random_text][:6]).item()
    })
    
    # Cleanup
    cleanup_model((cogact_model,))
    return actions, distances, instruction_distances

def run_openpi(args, optim_str, seed_img, seed_task_description, unnorm_key, target_action, input_ids):
    # Load OpenPI model and config
    cfg = OmegaConf.load(os.environ.get("OPENPI_CONFIG_PATH", "path/to/experiments/models/OpenPi0/config/eval/fractal_coke.yaml"))
    
    try:
        pali_path = snapshot_download(
            repo_id="google/paligemma-3b-pt-224",
            repo_type="model"
        )
        logger.info(f"PaLI-Gemma model found at: {pali_path}")
    except Exception as e:
        logger.error(f"Failed to find PaLI-Gemma model: {e}")
        raise
        
    # Update config with correct path
    cfg.pretrained_model_path = pali_path
    
    # Initialize processor
    tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_model_path, padding_side="right")
    processor = VLAProcessor(tokenizer, cfg.vision.config.num_image_tokens, cfg.env.adapter.max_seq_len)
    
    model = PiZeroInference(cfg)
    model.tie_action_proprio_weights()
    model.to(args.device)
    model.to(torch.bfloat16)
    model.eval()
    
    # Load checkpoint
    checkpoint_path = hf_hub_download(
        repo_id="allenzren/open-pi-zero",
        filename="fractal_beta_step29576_2024-12-29_13-10_42.pt",
        repo_type="model"
    )    
    data = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    data["model"] = {k.replace("_orig_mod.", ""): v for k, v in data["model"].items()}
    model.load_state_dict(data["model"], strict=True)
    
    actions = {}
    distances = {}
    instruction_distances = {}
    action_tensors = {}
    
    # Convert target_action to tensor if it isn't already
    if not isinstance(target_action, torch.Tensor):
        target_action = torch.tensor(target_action, device=args.device, dtype=torch.float32)
    
    # Add random tokens test using PaLI-Gemma tokenizer
    random_text = get_random_tokens(tokenizer)  # Using the already initialized tokenizer
    
    for i, description in enumerate([seed_task_description, optim_str, random_text]):
        logger.info(f"Running OpenPI for {description}\n")
        instruction = description
        prompt = openvla_prompt_template.format(task_description=instruction)
        # Set different random seed for each run, ensuring it's within valid range
        torch.manual_seed(int(time.time() * 1000) % (2**32))
        
        # Resize image to 224x224
        if isinstance(seed_img, Image.Image):
            seed_img_resized = seed_img.resize((224, 224), Image.Resampling.LANCZOS)
        else:
            seed_img_resized = Image.fromarray(seed_img).resize((224, 224), Image.Resampling.LANCZOS)
            
        # Convert PIL image to tensor
        image_tensor = torch.from_numpy(np.array(seed_img_resized)).permute(2, 0, 1).unsqueeze(0)
        if image_tensor.dtype != torch.uint8:
            image_tensor = (image_tensor * 255).to(torch.uint8)
        
        # Process image and text
        model_inputs = processor(text=[prompt], images=image_tensor)
        
        # Create neutral proprio state
        proprio = torch.zeros(1, 1, 8, device=args.device, dtype=torch.bfloat16)  # [batch, time, dim]
        
        # Move inputs to device and convert to proper dtype
        inputs = {k: v.to(args.device) for k, v in model_inputs.items()}
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        inputs["proprios"] = proprio
        if description == "optim_str":
            inputs['input_ids'] = input_ids
        # Build masks and position IDs
        causal_mask, vlm_position_ids, proprio_position_ids, action_position_ids = (
            model.build_causal_mask_and_position_ids(inputs["attention_mask"].to(args.device), dtype=torch.bfloat16)
        )
        
        # Ensure position IDs are on the correct device
        vlm_position_ids = vlm_position_ids.to(args.device)
        proprio_position_ids = proprio_position_ids.to(args.device)
        action_position_ids = action_position_ids.to(args.device)
        
        image_text_proprio_mask, action_mask = model.split_full_mask_into_submasks(causal_mask)
        
        # Ensure masks are on the correct device
        image_text_proprio_mask = image_text_proprio_mask.to(args.device)
        action_mask = action_mask.to(args.device)
        
        with torch.inference_mode():
            action = model(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                image_text_proprio_mask=image_text_proprio_mask,
                action_mask=action_mask,
                vlm_position_ids=vlm_position_ids,
                proprio_position_ids=proprio_position_ids,
                action_position_ids=action_position_ids,
                proprios=inputs["proprios"],
            )
            
            first_action = action[0, 0]
            
            if isinstance(first_action, np.ndarray):
                action_tensor = torch.from_numpy(first_action).to(device=args.device, dtype=torch.float32)
            elif not isinstance(first_action, torch.Tensor):
                action_tensor = torch.tensor(first_action, device=args.device, dtype=torch.float32)
            else:
                action_tensor = first_action
                
            action_tensors[description] = action_tensor
            actions[description] = first_action.tolist() if isinstance(first_action, (np.ndarray, torch.Tensor)) else first_action
            distances[description] = torch.norm(action_tensor[:6] - target_action[:6]).item()
    
    instruction_distances["between_instructions"] = torch.norm(
        action_tensors[seed_task_description][:6] - action_tensors[optim_str][:6]
    ).item()
    
    # Update instruction distances to include random text
    instruction_distances.update({
        "seed_vs_random": torch.norm(action_tensors[seed_task_description][:6] - action_tensors[random_text][:6]).item(),
        "optim_vs_random": torch.norm(action_tensors[optim_str][:6] - action_tensors[random_text][:6]).item()
    })
    
    cleanup_model((model,))
    return actions, distances, instruction_distances

def single_action_transfer_experiment(args, target_str, seed_img, seed_task_description, action_tokenizer, unnorm_key, target_action):
    gcg_result, gcg_action, gcg_target_actions = run_gcg(args, target_str, seed_img, unnorm_key, action_tokenizer)
    input_ids = torch.cat([gcg_result.perfect_match_before_ids, gcg_result.perfect_match_optim_ids, gcg_result.perfect_match_after_ids], dim=1)
    tracevla_result, tracevla_distances, tracevla_instruction_distances = run_tracevla(
        args, gcg_result.perfect_match_optim_str, seed_img, seed_task_description, unnorm_key, target_action, input_ids
    )
    cogact_result, cogact_distances, cogact_instruction_distances = run_cogact(
        args, gcg_result.perfect_match_optim_str, seed_img, seed_task_description, unnorm_key, target_action
    )
    openpi_result, openpi_distances, openpi_instruction_distances = run_openpi(
        args, gcg_result.perfect_match_optim_str, seed_img, seed_task_description, unnorm_key, target_action, input_ids
    )
    
    return {
        "gcg_result": gcg_result,
        "openvla_action": gcg_action,
        "openvla_target_actions": gcg_target_actions,
        "tracevla_result": tracevla_result,
        "cogact_result": cogact_result,
        "openpi_result": openpi_result,
        "tracevla_distances_to_target": tracevla_distances,
        "cogact_distances_to_target": cogact_distances,
        "openpi_distances_to_target": openpi_distances,
        "tracevla_distances_between_instructions": tracevla_instruction_distances,
        "cogact_distances_between_instructions": cogact_instruction_distances,
        "openpi_distances_between_instructions": openpi_instruction_distances,
    }

def convert_to_serializable(obj):
    """Convert objects to JSON serializable format"""
    if hasattr(obj, '__dict__'):
        return {k: convert_to_serializable(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)

def save_results(result, target_action, recovered_action, save_dir="results/transfer_results", trial_num=0):
    """Save results for a single run"""
    # Convert target action to string for directory name
    target_str = "_".join([f"{x:.3f}" for x in target_action])
    action_dir = Path(save_dir) / f"{target_str}_test"
    action_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"result_trial{trial_num}_{timestamp}.json"
    
    # Add target action to result dictionary
    result["target_action"] = target_action
    result["unnormalized_target_action"] = recovered_action
    
    # Convert result to JSON serializable format
    serializable_result = convert_to_serializable(result)
    
    # Save to file
    with open(action_dir / filename, "w") as f:
        json.dump(serializable_result, f, indent=2)
    
    logger.info(f"Saved results to {filename} in {action_dir}")

def cleanup_model(model_tuple):
    """Helper function to clean up model and processor"""
    for item in model_tuple:
        if hasattr(item, 'cpu'):
            item.cpu()
        del item
    torch.cuda.empty_cache()

def main():
    args = parse_args()
    setup_logging(args)
    unnorm_key = args.unnorm_key

    seed_img_path = args.image_path
    seed_img = Image.open(seed_img_path)
    seed_task_description = "pick coke can"

    target_actions = get_actions()
    num_trials = 1  
    
    # Load just the action tokenizer once since it's lightweight
    openvla_model = OpenVLAForActionPrediction.from_pretrained(args.openvla_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    openvla_processor = AutoProcessor.from_pretrained(args.openvla_model_path, trust_remote_code=True)
    action_tokenizer = ActionTokenizer(32000, 256, openvla_model.config.norm_stats)
    cleanup_model((openvla_model,))

    for target_action in target_actions:
        logger.info(f"\nStarting experiments for target action: {target_action}")
        action_tokens = action_tokenizer.actions_to_tokens(target_action, unnorm_key=unnorm_key)
        target = openvla_processor.tokenizer.decode(action_tokens)
        recovered_action = action_tokenizer.tokens_to_actions(action_tokens, unnorm_key=unnorm_key)
        
        for trial in range(num_trials):
            logger.info(f"Running trial {trial + 1}/{num_trials}")
            result = single_action_transfer_experiment(
                args, target, seed_img, seed_task_description, action_tokenizer, unnorm_key, recovered_action
            )
            # Save results immediately after each run
            save_results(result, target_action, recovered_action, trial_num=trial)
 
if __name__ == "__main__":
    main()
