"""
Utility functions for model handling in single-step adversarial attacks.

This module provides common functionality for loading, initializing, and using
vision-language models across all experiment files.
"""

import logging
import os
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union, List
from transformers import AutoProcessor, AutoTokenizer, AutoModelForVision2Seq
from PIL import Image

from experiments.models.OpenVLA.modeling_prismatic import OpenVLAForActionPrediction
from experiments.models.OpenVLA.action_tokenizer import ActionTokenizer
from experiments.models.Pi0_fast.modelling_pi import Pi0FastForActionPrediction
from experiments.single_step.utils import cleanup_model, process_image_with_trace, generate_action_space

logger = logging.getLogger(__name__)

def load_openvla_model(
    model_path: str,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    model_class = None
) -> Tuple[AutoModelForVision2Seq, ActionTokenizer, AutoProcessor]:
    """
    Load an OpenVLA model with its action tokenizer and processor.
    
    Args:
        model_path: Path to the model
        device: Device to load the model on
        torch_dtype: Torch data type for model computation
    
    Returns:
        Tuple containing the model, action tokenizer, and processor
    """
    logger.info(f"Loading OpenVLA model from {model_path}")
    
    if model_class is not None:
        model = model_class.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True
        )
    else:
        pi_model = Pi0FastForActionPrediction.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True
        )
        # action_tokenizer = ActionTokenizer(32000, 256, model.config.norm_stats)
        # Load the base processor to get the class
        base_tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True,scale=133)
        actions = generate_action_space(max_mag_actions_only=False, action_dim_size=None)  # Use full action space
        # logger.info(f"Generated {len(actions)} actions for fitting")
        
        # # Convert actions from shape (7,) to (1, 1, 7) for tokenizer fitting
        # # This matches the expected format: [batch, timesteps, action_dim]
        actions_reshaped = [action.reshape(1, 1, 7) for action in actions]
        quantized_unique = set()
        duplicate_count = 0
        
        for action in actions_reshaped:
            quantized = base_tokenizer(action)
            # Convert list of lists to tuple of tuples for hashing
            quantized_tuple = tuple(tuple(tokens) for tokens in quantized)
            
            if quantized_tuple in quantized_unique:
                duplicate_count += 1
            else:
                quantized_unique.add(quantized_tuple)
        
        if duplicate_count == 0:
            logger.info(f"✓ Verified: All {len(actions_reshaped)} actions remain distinct after quantization")
        else:
            logger.warning(f"⚠ Warning: {duplicate_count} actions become identical after quantization")
        
        # # Fit the action tokenizer on the generated actions with optimal scale
        logger.info(f"Fitting action tokenizer on generated action space with scale={base_tokenizer.scale}...")
        action_tokenizer = base_tokenizer.fit(
            action_data=actions_reshaped,
            scale=base_tokenizer.scale,
            vocab_size=base_tokenizer.vocab_size,
            time_horizon=1,  # Single timestep
            action_dim=7,    # 7 action dimensions
        )
    model = pi_model.to(device)
    
    # Load tokenizer and processor
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    return model, action_tokenizer, processor

def load_image_with_trace(
    image: Image.Image,
    processor: AutoProcessor,
    device: str = "cuda",
    use_trace: bool = False,
    trace_processor_path: Optional[str] = None
) -> Tuple[torch.Tensor, Image.Image, bool]:
    """
    Load and process an image with optional trace overlay.
    
    Args:
        image_path: Path to the image
        processor: Image processor
        device: Device to place tensors on
        use_trace: Whether to use trace processing
        trace_processor_path: Path to the trace processor model
    
    Returns:
        Tuple containing processed pixel values, original image, and whether trace was applied
    """
    logger.info(f"Loading image")
    
    # Load image
    try:
        image = image.convert('RGB') 
    except Exception as e:
        logger.error(f"Error loading image: {str(e)}")
        raise
    
    # Process image with trace if requested
    if use_trace and trace_processor_path is not None:
        try:
            from experiments.models.TraceVLA.trace_processor import TraceProcessor
            trace_processor = TraceProcessor(trace_processor_path)
            pixel_values, has_trace = process_image_with_trace(
                image, processor, trace_processor, device, use_trace
            )
        except ImportError as e:
            logger.warning(f"Trace processor import failed: {str(e)}. Falling back to standard processing.")
            pixel_values = processor("temp", image)['pixel_values'].to(device, dtype=torch.bfloat16)
            has_trace = False
    else:
        pixel_values = processor("temp", image)['pixel_values'].to(device, dtype=torch.bfloat16)
        has_trace = False
    
    return pixel_values, image, has_trace

def get_device_for_worker(worker_id: int) -> str:
    """
    Get the appropriate device for a worker based on ID.
    
    Args:
        worker_id: Worker ID
    
    Returns:
        str: Device identifier
    """
    if torch.cuda.is_available():
        device_id = worker_id % torch.cuda.device_count()
        device = f'cuda:{device_id+1}'
    else:
        device = 'cpu'
        logger.warning(f"CUDA not available. Worker {worker_id} using CPU.")
    
    return device