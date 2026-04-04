"""
Shared utility functions for defense mechanisms against adversarial prompts.

This module provides common functionality used across defense implementations
to reduce code duplication and improve maintainability.
"""

import logging
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
from datetime import datetime
import json
import sys
import os

# Add project root to Python path if not already there
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import shared utilities from experiments
try:
    from experiments.single_step.utils import (
        convert_to_tensor, actions_match, cleanup_model, 
        process_image_with_trace, setup_logging as base_setup_logging
    )
except ImportError:
    # Fallback implementations if utils module is not available
    def convert_to_tensor(action):
        """Convert numpy array to tensor if necessary."""
        if isinstance(action, np.ndarray):
            return torch.from_numpy(action)
        return action

    def actions_match(action1, action2, dims=6, atol=1e-5):
        """Check if two actions match within tolerance for first n dimensions."""
        return np.allclose(action1[:dims], action2[:dims], atol=atol)

    def cleanup_model(model_tuple):
        """Clean up model resources."""
        for item in model_tuple:
            if hasattr(item, 'cpu'):
                item.cpu()
            del item
        torch.cuda.empty_cache()

    def process_image_with_trace(image, processor, trace_processor=None, device="cuda", use_trace=False):
        """Process an image with optional trace overlay."""
        return processor("temp", image)['pixel_values'].to(device, dtype=torch.bfloat16), False

    def base_setup_logging(log_dir, timestamp=None):
        """Basic logging setup function."""
        log_dir.mkdir(exist_ok=True)
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return timestamp

logger = logging.getLogger(__name__)

def setup_defense_logging(defense_name: str = "defense") -> str:
    """
    Setup logging to both console and file for defense experiments.
    
    Args:
        defense_name: Name of the defense for log file naming
    
    Returns:
        str: Timestamp used for the log file
    """
    timestamp = base_setup_logging(Path("logs"), None)
    logger.info(f"Starting {defense_name} defense experiment at {timestamp}")
    return timestamp

def save_defense_results(
    model_name: str, 
    results: List[Dict], 
    timestamp: str, 
    defense_name: str,
    output_subdir: str = "outputs",
    extra_metadata: Dict = None
) -> Path:
    """
    Save defense experiment results to JSON file.
    
    Args:
        model_name: Name of the model used
        results: List of trial results
        timestamp: Timestamp for the experiment
        defense_name: Name of the defense mechanism
        output_subdir: Subdirectory for outputs
        extra_metadata: Additional metadata to include
        
    Returns:
        Path: Path to the saved results file
    """
    # Extract model short name for directory naming
    model_short_name = model_name.split('/')[-1] 
    output_dir = Path(output_subdir) / f"{model_short_name}_{defense_name}" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a results dictionary with metadata
    experiment_results = {
        "model": model_name,
        "timestamp": timestamp,
        "defense": defense_name,
        "trials": results
    }
    
    # Add any extra metadata if provided
    if extra_metadata:
        experiment_results.update(extra_metadata)
    
    # Save to JSON
    output_file = output_dir / f"results.json"
    with open(output_file, "w") as f:
        json.dump(experiment_results, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")
    return output_file

def get_action_space(max_dims: int = 7, max_values: int = 256, sample_size: Optional[int] = None) -> List[np.ndarray]:
    """
    Generate the full action space or a sampled subset.
    
    Args:
        max_dims: Number of dimensions in the action space
        max_values: Number of discrete values per dimension
        sample_size: If specified, sample this many actions
        
    Returns:
        List[np.ndarray]: List of action vectors
    """
    actions = []
    for i in range(max_dims):  
        for j in range(max_values):  
            action = np.zeros(max_dims)
            action[i] = (j / (max_values/2 - 0.5)) - 1  # Normalize to [-1, 1]
            actions.append(action)
    
    # Sample if requested
    if sample_size is not None and sample_size < len(actions):
        import random
        return random.sample(actions, sample_size)
        
    return actions

def get_predefined_max_actions() -> List[np.ndarray]:
    """
    Get a list of actions with maximum magnitude in each dimension.
    
    Returns:
        List[np.ndarray]: List of max magnitude actions
    """
    actions = []
    for i in range(6):  # First 6 dimensions (omit gripper)
        for val in [-1, 1]:
            action = np.zeros(7)
            action[i] = val
            actions.append(action)
    return actions

def add_random_noise(text: str, noise_prob: float = 0.1) -> str:
    """
    Add random character noise to text with given probability.
    
    Args:
        text: Input text to add noise to
        noise_prob: Probability of modifying each character
        
    Returns:
        str: Text with random noise added
    """
    import random
    chars = list(text)
    for i in range(len(chars)):
        if random.random() < noise_prob:
            # Random ASCII printable character (32-126)
            chars[i] = chr(random.randint(32, 126))
    return ''.join(chars)

def load_model_for_defense(model_path: str, device: str = "cuda") -> Tuple:
    """
    Load a model and its components for defense evaluation.
    
    Args:
        model_path: Path to the model
        device: Device to load the model on
        
    Returns:
        Tuple: (model, action_tokenizer, processor)
    """
    from transformers import AutoProcessor
    from experiments.models.OpenVLA.modeling_prismatic import OpenVLAForActionPrediction
    from experiments.models.OpenVLA.action_tokenizer import ActionTokenizer
    
    # Check device availability
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"
    
    # Load model and move to device
    model = OpenVLAForActionPrediction.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True
    ).to(device)
    
    # Load tokenizer and processor
    action_tokenizer = ActionTokenizer(32000, 256, model.config.norm_stats)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    return model, action_tokenizer, processor

def verify_data_paths(image_paths: List[str], model_paths: List[str] = None) -> bool:
    """
    Verify that all required data paths exist.
    
    Args:
        image_paths: List of image paths to verify
        model_paths: List of model paths to verify
        
    Returns:
        bool: True if all paths exist, False otherwise
    """
    all_paths_exist = True
    
    # Check image paths
    for path in image_paths:
        path_obj = Path(path)
        if not path_obj.exists():
            logger.error(f"Image path not found: {path}")
            all_paths_exist = False
    
    # Check model paths if provided
    if model_paths:
        for path in model_paths:
            # For HF models, just log a warning since they might be downloaded on demand
            if not path.startswith("openvla/") and not Path(path).exists():
                logger.warning(f"Model path not found: {path}")
    
    return all_paths_exist