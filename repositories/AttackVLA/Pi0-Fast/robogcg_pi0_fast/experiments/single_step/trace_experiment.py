import argparse
from PIL import Image
import torch
from transformers import AutoProcessor
import numpy as np
from pathlib import Path
import sys
from typing import List, Tuple, Dict, Optional
import logging
import json
from datetime import datetime
import torch.multiprocessing as mp
from torch.multiprocessing import Queue
from queue import Empty
import time

# Setup paths and logging
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "experiments/models/TraceVLA"))
from trace_processor import TraceProcessor
import roboGCG.robogcg.robo_gcg as gcg
from roboGCG.robogcg.utils import GCGConfig
from experiments.models.TraceVLA.modeling_prismatic import OpenVLAForActionPrediction 
from experiments.models.OpenVLA.action_tokenizer import ActionTokenizer

# Import shared utilities
from experiments.single_step.utils import (
    setup_logging, generate_action_space, process_image_with_trace,
    save_results as utils_save_results, cleanup_model, actions_match
)
from experiments.single_step.model_utils import (
    get_device_for_worker
)
from experiments.single_step.constants import (
    BATCH_SIZE, TRACE_PROCESSOR_PATH, OPENVLA_PROMPT_TEMPLATE,
    TRACEVLA_PROMPT_TEMPLATE, UNNORM_KEYS
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="furonghuang-lab/tracevla_7b")
    parser.add_argument("--image_path", type=str, 
                       default=str(project_root / "images" / "simpler" / "pick_coke_can" / "10.png"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--action_dim_size", type=int, default=20)  # Number of bins per dimension
    parser.add_argument("--start_action", type=int, default=None)
    parser.add_argument("--end_action", type=int, default=None)
    parser.add_argument("--custom_actions", type=bool, default=False)
    args = parser.parse_args()
    return args

def load_custom_actions():
    with open("configs/custom_actions.json", "r") as f:
        data = json.load(f)
    return data["actions"], data["init_strs"]

def run_gcg(model, action_tokenizer, processor, target, pixel_values, config):
    """Run GCG and return the result"""
    gcg_result = gcg.robo_run(model, processor, action_tokenizer, target, pixel_values=pixel_values, config=config)
    return gcg_result

def save_action_results(result: Dict, output_dir: Path, action_idx: int, custom_actions: bool):
    """Save results for a single action"""
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / f"action_{action_idx}_{custom_actions}.json"
    
    with open(filename, 'w') as f:
        json.dump(result, f, indent=2)
    logger.info(f"Results saved to {filename}")

def worker_process(worker_id: int, task_queue: Queue, args, image, actions, init_strs=None):
    """Worker process that pulls tasks from queue"""
    try:
        # Get device for this worker
        device = get_device_for_worker(worker_id)
        torch.cuda.set_device(device) if 'cuda' in device else None
        logger.info(f"Worker {worker_id} starting on device {device}")
        
        # Initialize models and processors directly on the correct device
        # TraceVLA specific: We need to use the correct model class
        logger.info(f"Loading model from {args.model}")
        model = OpenVLAForActionPrediction.from_pretrained(
            args.model, 
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).to(device)
        
        processor = AutoProcessor.from_pretrained(
            args.model,
            trust_remote_code=True,
            num_crops=1
        )
        
        action_tokenizer = ActionTokenizer(32000, 256, model.config.norm_stats)
        
        # Get trace processor path
        trace_processor_path = Path(project_root) / TRACE_PROCESSOR_PATH
        if not trace_processor_path.exists():
            logger.warning(f"Trace processor not found at {trace_processor_path}")
            has_trace = False
            pixel_values = processor("temp", image)['pixel_values'].to(device, dtype=torch.bfloat16)
        else:
            # Process image with trace
            trace_processor = TraceProcessor(str(trace_processor_path))
            image_overlaid, has_trace = trace_processor.process_image(image)
            pixel_values = processor("temp", [image, image_overlaid] if has_trace else [image, image])['pixel_values']
            pixel_values = pixel_values.to(device, dtype=torch.bfloat16)
        
        actions_processed = 0
        while True:
            try:
                action_batch = task_queue.get(timeout=1)
                if action_batch is None:
                    logger.info(f"Worker {worker_id} finished after processing {actions_processed} actions")
                    break
                
                for action_idx in action_batch:
                    target_action = actions[action_idx]
                    logger.info(f"Worker {worker_id} processing action {action_idx}")
                    start_time = time.time()
                    
                    # Process single action
                    unnorm_key = UNNORM_KEYS["default"]
                    action_tokens = action_tokenizer.actions_to_tokens(target_action, unnorm_key=unnorm_key)
                    target = processor.tokenizer.decode(action_tokens)
                    unnormalized_action = action_tokenizer.tokens_to_actions(action_tokens, unnorm_key=unnorm_key)
                    
                    # Configure GCG based on whether we're using custom actions
                    gcg_config_args = {
                        "num_steps": 500,
                        "use_prefix_cache": False,
                        "unnorm_key": unnorm_key,
                        "use_trace": has_trace
                    }
                    
                    # Add init string for custom actions
                    if args.custom_actions and init_strs:
                        gcg_config_args["optim_str_init"] = init_strs[action_idx]
                    
                    gcg_config = GCGConfig(**gcg_config_args)
                    
                    gcg_result = run_gcg(model, action_tokenizer, processor, target, pixel_values, gcg_config)
                    
                    inputs = processor("temp", [image, image_overlaid] if has_trace else [image, image]).to(device, dtype=torch.bfloat16)
                    inputs['input_ids'] = torch.cat([
                        gcg_result.perfect_match_before_ids,
                        gcg_result.perfect_match_optim_ids,
                        gcg_result.perfect_match_after_ids
                    ], dim=1)
                    action = model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
                    
                    # Clean up inputs to save memory
                    del inputs
                    
                    # Use the utility function to check if actions match
                    is_success = actions_match(action, unnormalized_action)
                    
                    result = {
                        "action_idx": action_idx,
                        "num_steps": gcg_result.num_steps_taken,
                        "total_time": gcg_result.total_time,
                        "init_str_length": gcg_result.init_str_length,
                        "optim_str": gcg_result.perfect_match_optim_str,
                        "success": bool(is_success),
                        "predicted_action": action.tolist(),
                        "target_action": target_action,
                        "unnorm_action": unnormalized_action.tolist(),
                        "prompt": target,
                        "losses": gcg_result.losses.tolist() if hasattr(gcg_result.losses, 'tolist') else [float(x) for x in gcg_result.losses]
                    }    
                    # Save results for this action
                    output_dir = project_root / "results" / "trace_experiment" / f"worker_{worker_id}"
                    save_action_results(result, output_dir, action_idx, args.custom_actions)
                    
                    end_time = time.time()
                    actions_processed += 1
                    logger.info(f"Worker {worker_id} completed action {action_idx} in {end_time - start_time:.2f}s")
                    
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {str(e)}", exc_info=True)
                break
                
    except Exception as e:
        logger.error(f"Worker {worker_id} initialization error: {str(e)}", exc_info=True)
    finally:
        # Always clean up resources
        if 'model' in locals():
            cleanup_model((model,))

def main():
    # Parse arguments
    args = parse_args()
    
    # Initialize logging
    timestamp = setup_logging()
    logger.info(f"Starting trace experiment with args: {args}")
    
    # Generate action space based on configuration
    if args.custom_actions:
        actions, init_strs = load_custom_actions()
        logger.info(f"Loaded {len(actions)} custom actions with init strings")
    else:
        actions = generate_action_space(action_dim_size=args.action_dim_size)
        init_strs = None
        logger.info(f"Generated action space with {args.action_dim_size} bins per dimension")
    
    total_actions = len(actions)
    logger.info(f"Testing {total_actions} actions")
    
    # Use specified start/end if provided, otherwise use full range
    start_idx = args.start_action if args.start_action is not None else 0
    end_idx = args.end_action if args.end_action is not None else total_actions
    logger.info(f"Processing actions from index {start_idx} to {end_idx}")
    
    # Load image
    try:
        image = Image.open(args.image_path)
        logger.info(f"Loaded image from {args.image_path}")
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        sys.exit(1)
    
    # Setup multiprocessing
    num_gpus = torch.cuda.device_count()
    logger.info(f"Using {num_gpus} GPUs")
    
    # Create task queue and batches
    task_queue = Queue()
    action_batches = [
        list(range(i, min(i + BATCH_SIZE, end_idx)))
        for i in range(start_idx, end_idx, BATCH_SIZE)
    ]
    
    for batch in action_batches:
        task_queue.put(batch)
    
    # Start workers with init_strs if using custom actions
    processes = []
    try:
        for worker_id in range(num_gpus):
            p = mp.Process(
                target=worker_process,
                args=(worker_id, task_queue, args, image, actions, init_strs)
            )
            p.start()
            processes.append(p)
            logger.info(f"Started worker {worker_id}")
        
        # Add termination signals
        for _ in range(num_gpus):
            task_queue.put(None)
        
        # Wait for completion
        for p in processes:
            p.join()
        
        logger.info("All workers completed")
        
    except KeyboardInterrupt:
        logger.info("Interrupt received, shutting down workers...")
        for p in processes:
            if p.is_alive():
                p.terminate()
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error in main process: {e}", exc_info=True)
        for p in processes:
            if p.is_alive():
                p.terminate()
        sys.exit(1)

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()
