"""
Main entry point for running distributed single-step adversarial attacks on
vision-language robot control models. This script uses a multi-process approach
to parallelize experiments across multiple GPUs.

The script:
1. Loads configuration from a JSON file
2. Sets up the worker processes, one per GPU
3. Distributes action batches to workers
4. Collects and saves results

Example usage:
    python run_experiment.py --config configs/base_experiment.json --num-gpus 4 --start-action 0 --end-action 100
"""

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
import signal
import os

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from experiments.models.OpenVLA.modeling_prismatic import OpenVLAForActionPrediction
from experiments.models.OpenVLA.action_tokenizer import ActionTokenizer
from experiments.single_step.config import SingleStepConfig
from experiments.single_step.experiment_runner import SingleStepExperiment
from experiments.models.TraceVLA.trace_processor import TraceProcessor

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import constants
from experiments.single_step.constants import BATCH_SIZE, MAX_DIMENSIONS, MAX_VALUES_PER_DIM, TOTAL_ACTIONS

def setup_logging(args):
    """
    Setup logging to both console and file.
    
    Args:
        args (Namespace): Command line arguments
        
    Returns:
        None
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"experiment_{timestamp}.log"
    
    # Setup logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Setup file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Clear existing handlers to avoid duplicates
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
    
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logger.info(f"Logging to {log_file}")
    logger.info(f"Starting experiment with args: {args}")

def worker_process(worker_id: int, task_queue: Queue, config: SingleStepConfig, model_name: str, image):
    """
    Worker process that pulls tasks from queue and processes them.
    
    Args:
        worker_id (int): ID of the worker process
        task_queue (Queue): Queue of action batches to process
        config (SingleStepConfig): Configuration for experiments
        model_name (str): Name or path of model to use
        image (Image): Seed image for experiment
        
    Returns:
        None
    """
    try:
        # Import helper functions
        from experiments.single_step.model_utils import load_openvla_model, load_image_with_trace, get_device_for_worker
        from experiments.single_step.constants import TRACE_PROCESSOR_PATH
        
        # Get device for this worker
        device = get_device_for_worker(worker_id)
        torch.cuda.set_device(device) if 'cuda' in device else None
        logger.info(f"Worker {worker_id} starting up on device {device}")
        
        # Set the worker_id in the config
        config.worker_id = worker_id

        # Load model components
        logger.info(f"Worker {worker_id} initializing model...")
        model, action_tokenizer, processor = load_openvla_model(model_name, device)
        
        # Process image in worker and ensure it's on the correct device
        logger.info(f"Worker {worker_id} processing image...")
        trace_processor_path = Path(project_root) / TRACE_PROCESSOR_PATH
        
        if not trace_processor_path.exists() and config.use_trace:
            logger.warning(f"Trace processor not found at {trace_processor_path}, disabling trace processing")
            config.use_trace = False
        
        # Load and process image
        pixel_values, _, has_trace = load_image_with_trace(
            image, processor, device, config.use_trace, 
            str(trace_processor_path) if trace_processor_path.exists() else None
        )
        
        # Update config to reflect whether trace was actually used
        if not has_trace:
            config.use_trace = False
            
        logger.info(f"Pixel values shape: {pixel_values.shape}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Create experiment instance
        experiment = SingleStepExperiment(
            config=config,
            model=model,
            action_tokenizer=action_tokenizer,
            tokenizer=tokenizer,
            processor=processor,
            pixel_values=pixel_values,
            seed_image=image
        )
        
        actions_processed = 0
        while True:
            try:
                action_batch = task_queue.get(timeout=1)
                if action_batch is None:
                    logger.info(f"Worker {worker_id} received shutdown signal. Processed {actions_processed} actions total.")
                    break
                    
                logger.info(f"Worker {worker_id} processing batch of {len(action_batch)} actions: {action_batch}")
                for action_idx in action_batch:
                    logger.info(f"Worker {worker_id} starting action {action_idx}")
                    start_time = time.time()
                    try:
                        config.current_action = action_idx
                        experiment.run_single_action(action_idx)
                    except Exception as e:
                        logger.error(f"Error processing action {action_idx}: {str(e)}", exc_info=True)
                        # Continue with next action
                        
                    end_time = time.time()
                    actions_processed += 1
                    logger.info(f"Worker {worker_id} completed action {action_idx} in {end_time - start_time:.2f} seconds")
                
                logger.info(f"Worker {worker_id} completed batch. Total actions processed: {actions_processed}")
                    
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} encountered error: {str(e)}", exc_info=True)
                break
                
    except Exception as e:
        logger.error(f"Worker {worker_id} initialization error: {str(e)}", exc_info=True)

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Run single-step GCG experiments')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to experiment configuration file')
    parser.add_argument('--num-gpus', type=int, required=True,
                      help='Number of GPUs to use')
    parser.add_argument('--start-action', type=int, default=None,
                      help='Starting action index (inclusive, out of 1792)')
    parser.add_argument('--end-action', type=int, default=None,
                      help='Ending action index (exclusive, out of 1792)')
    return parser.parse_args()

def signal_handler(signum, frame):
    """
    Handle interrupt signals.
    
    Args:
        signum: Signal number
        frame: Current stack frame
        
    Returns:
        None
    """
    logger.info("Interrupt received, shutting down workers...")
    # Terminate all processes
    for p in processes:
        if p.is_alive():
            p.terminate()
    sys.exit(1)

def main():
    """
    Main function for running distributed experiments.
    
    Returns:
        None
    """
    args = parse_args()
    setup_logging(args)
    
    # Register signal handler for clean shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # List to keep track of processes for signal handler
    global processes
    processes = []
    
    try:
        # Check CUDA availability
        if not torch.cuda.is_available():
            logger.warning("CUDA is not available. Running on CPU will be very slow.")
            num_workers = 1
        else:
            num_gpus_available = torch.cuda.device_count()
            num_workers = min(args.num_gpus, num_gpus_available)
            if num_workers < args.num_gpus:
                logger.warning(f"Requested {args.num_gpus} GPUs but only {num_gpus_available} are available.")
        
        # Load configuration
        logger.info("Loading configuration...")
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error(f"Configuration file not found: {config_path}")
            sys.exit(1)
            
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing configuration file: {str(e)}")
            sys.exit(1)
            
        # Add num_gpus to config
        config_dict['num_gpus'] = args.num_gpus
        
        try:
            base_config = SingleStepConfig.from_dict(config_dict)
        except Exception as e:
            logger.error(f"Error creating configuration: {str(e)}")
            sys.exit(1)
            
        base_config.start_action = args.start_action
        base_config.end_action = args.end_action
        
        # Load image
        logger.info("Loading seed image...")
        image_path = Path(project_root) / base_config.seed_frame_path
        if not image_path.exists():
            logger.error(f"Seed image not found: {image_path}")
            sys.exit(1)
        
        image = Image.open(image_path)
        # Create task queue and batches
        task_queue = Queue()
        start_idx = args.start_action if args.start_action is not None else 0
        
        # Set end_idx based on whether we're using max magnitude actions only
        if base_config.max_mag_actions_only:
            total_actions = 12  # 6 dimensions * 2 (max and min) actions
            end_idx = args.end_action if args.end_action is not None else total_actions
        elif base_config.action_dim_size is not None:
            total_actions = 6 * base_config.action_dim_size  # 6 dimensions * bins per dimension
            end_idx = args.end_action if args.end_action is not None else total_actions
        else:
            total_actions = TOTAL_ACTIONS  # Full action space
            end_idx = args.end_action if args.end_action is not None else total_actions
        
        # Validate action range
        if start_idx < 0:
            logger.warning(f"Invalid start_action {start_idx}, setting to 0")
            start_idx = 0
        
        if end_idx > total_actions:
            logger.warning(f"Invalid end_action {end_idx}, setting to {total_actions}")
            end_idx = total_actions
            
        if start_idx >= end_idx:
            logger.error(f"Invalid action range: start_action {start_idx} >= end_action {end_idx}")
            sys.exit(1)
        
        # Create action batches
        action_batches = [
            list(range(i, min(i + BATCH_SIZE, end_idx)))
            for i in range(start_idx, end_idx, BATCH_SIZE)
        ]
        
        logger.info(f"Created {len(action_batches)} batches of size {BATCH_SIZE} for {total_actions} total actions")
        for batch in action_batches:
            task_queue.put(batch)
        
        logger.info(f"Using {num_workers} workers to process {len(action_batches)} batches")
        
        # Create worker processes
        for worker_id in range(num_workers):
            p = mp.Process(
                target=worker_process,
                args=(
                    worker_id,
                    task_queue,
                    base_config,
                    base_config.model_name,
                    image
                )
            )
            p.start()
            logger.info(f"Started worker {worker_id}")
            processes.append(p)
        
        # Add poison pills to stop workers
        for _ in range(num_workers):
            task_queue.put(None)
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
            
    except Exception as e:
        logger.error(f"Main process error: {str(e)}", exc_info=True)
        # Ensure processes are terminated on error
        for p in processes:
            if p.is_alive():
                p.terminate()
        sys.exit(1)
        
    logger.info("All workers completed")

if __name__ == '__main__':
    # Set start method to 'spawn' for CUDA support
    mp.set_start_method('spawn')
    main()