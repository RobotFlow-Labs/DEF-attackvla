"""
Utility functions for single-step adversarial attacks.

This module provides common functionality used across experiment files
in the single-step directory, to reduce code duplication.
"""

import logging
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
from transformers import AutoProcessor
from datetime import datetime
import json

from experiments.models.OpenVLA.action_tokenizer import ActionTokenizer
from experiments.models.TraceVLA.trace_processor import TraceProcessor

logger = logging.getLogger(__name__)

def setup_logging(log_dir: Path = Path("logs"), timestamp: Optional[str] = None) -> str:
    """
    Setup logging to both console and file.
    
    Args:
        log_dir: Directory to save log files
        timestamp: Timestamp string to use in filename, or None to generate one
    
    Returns:
        str: The timestamp used for the log file
    """
    log_dir.mkdir(exist_ok=True)
    
    # Create timestamp if not provided
    if timestamp is None:
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
    return timestamp

def convert_to_tensor(action):
    """
    Convert numpy array to tensor if necessary.
    
    Args:
        action: Action array (tensor or numpy array)
    
    Returns:
        torch.Tensor: Action as a tensor
    """
    if isinstance(action, np.ndarray):
        return torch.from_numpy(action)
    return action

def actions_match(action1, action2, dims=6, atol=1e-5):
    """
    Check if two actions match within tolerance for first n dimensions.
    
    Args:
        action1: First action (tensor or numpy array)
        action2: Second action (tensor or numpy array)
        dims: Number of dimensions to compare (default: 6)
        atol: Absolute tolerance for comparison (default: 1e-5)
    
    Returns:
        bool: True if actions match within tolerance
    """
    # Convert to tensors if needed
    action1 = convert_to_tensor(action1)
    action2 = convert_to_tensor(action2)
    
    return action1[:dims].allclose(action2[:dims], atol=atol)

def process_image_with_trace(
    image: Image.Image, 
    processor: AutoProcessor, 
    trace_processor: Optional[TraceProcessor] = None,
    device: str = "cuda",
    use_trace: bool = False
) -> Tuple[torch.Tensor, bool]:
    """
    Process an image with optional trace overlay.
    
    Args:
        image: Input image
        processor: Image processor
        trace_processor: Optional trace processor
        device: Device to place tensors on
        use_trace: Whether to use trace processing
    
    Returns:
        Tuple[torch.Tensor, bool]: Processed pixel values and whether trace was applied
    """
    has_trace = False
    
    if use_trace and trace_processor is not None:
        try:
            image_overlaid, has_trace = trace_processor.process_image(image)
            if has_trace:
                pixel_values = processor("temp", [image, image_overlaid])['pixel_values'].to(device, dtype=torch.bfloat16)
            else:
                pixel_values = processor("temp", [image, image])['pixel_values'].to(device, dtype=torch.bfloat16)
        except Exception as e:
            logger.warning(f"Trace processing failed: {str(e)}")
            pixel_values = processor("temp", image)['pixel_values'].to(device, dtype=torch.bfloat16)
    else:
        pixel_values = processor("temp", image)['pixel_values'].to(device, dtype=torch.bfloat16)
    
    return pixel_values, has_trace

def save_results(results: List[Dict], output_dir: Path, timestamp: Optional[str] = None, prefix: str = "results"):
    """
    Save results to a JSON file.
    
    Args:
        results: List of result dictionaries
        output_dir: Directory to save results
        timestamp: Timestamp string to use in filename, or None to generate one
        prefix: Prefix for the output filename
    """
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_file = output_dir / f"{prefix}_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")

def generate_action_space(
    max_mag_actions_only: bool = False, 
    action_dim_size: Optional[int] = None
) -> List[np.ndarray]:
    """
    Generate a discretized action space for robot control.
    
    Args:
        max_mag_actions_only: Whether to use only maximum magnitude actions
        action_dim_size: Number of bins per dimension, if provided
    
    Returns:
        List[np.ndarray]: List of action arrays
    """
    actions = []
    
    if max_mag_actions_only:
        # Generate only maximum magnitude actions (±1 for each dimension)
        for i in range(6):  
            for val in [-1, 1]:
                action = np.zeros(7)
                action[i] = val
                actions.append(action)
        
        assert len(actions) == 12, "Expected 12 max magnitude actions"
    
    elif action_dim_size is not None:
        # Generate a discretized space with specified bins per dimension
        for i in range(6):  # Only first 6 dimensions
            bins = np.linspace(-1, 1, action_dim_size)
            for bin_value in bins:
                action = np.zeros(7)
                action[i] = bin_value
                actions.append(action)
        
        assert len(actions) == 6 * action_dim_size, \
            f"Expected {6 * action_dim_size} actions, got {len(actions)}"
    
    else:
        # Generate the full action space (7 dimensions × 256 values)
        for i in range(7):
            for j in range(256):
                action = np.zeros(7)
                action[i] = (j / 127.5) - 1  # Normalize to [-1, 1]
                actions.append(action)
        
        assert len(actions) == 1792, "Expected 1792 actions for the full space"
    
    return actions

def get_predefined_actions() -> List[torch.Tensor]:
    """
    Return a predefined list of actions for testing.
    
    Returns:
        List[torch.Tensor]: List of 15 predefined actions
    """
    return [
        torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  
        torch.tensor([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  
        torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  
        torch.tensor([0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  
        torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),  
        torch.tensor([0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0]),  
        torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),  
        torch.tensor([0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0]),  
        torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),  
        torch.tensor([0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0]),  
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),  
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0]), 
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), 
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]),
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ]

def cleanup_model(model_tuple):
    """
    Clean up model and processor resources.
    
    Args:
        model_tuple: Tuple of models and processors to clean up
    """
    for item in model_tuple:
        if hasattr(item, 'cpu'):
            item.cpu()
        del item
    torch.cuda.empty_cache()

def convert_to_serializable(obj):
    """
    Convert objects to JSON serializable format.
    
    Args:
        obj: Object to convert
    
    Returns:
        JSON serializable representation of the object
    """
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