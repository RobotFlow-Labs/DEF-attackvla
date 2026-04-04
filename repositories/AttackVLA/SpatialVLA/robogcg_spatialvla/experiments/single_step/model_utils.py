"""
Utility functions for model handling in single-step adversarial attacks.

This module provides common functionality for loading, initializing, and using
vision-language models across all experiment files.
"""

import logging
import torch
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union, List
from transformers import AutoProcessor, AutoTokenizer
from PIL import Image

from transformers import AutoModel
from experiments.single_step.utils import cleanup_model, process_image_with_trace
logger = logging.getLogger(__name__)

def load_spatialvla_model(
    model_path: str,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16
) -> Tuple[AutoModel, AutoProcessor]:
    """
    Load a SpatialVLA model with its processor.
    
    Args:
        model_path: Path to the model
        device: Device to load the model on
        torch_dtype: Torch data type for model computation
    
    Returns:
        Tuple containing the model and processor
    """
    logger.info(f"Loading SpatialVLA model from {model_path}")
    
    # Load processor
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True,action_chunk_size=1)
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True
    ).eval().to(device)
    model.action_tokenizer=processor.action_tokenizer
    return model, processor

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
        device = f'cuda:{device_id}'
    else:
        device = 'cpu'
        logger.warning(f"CUDA not available. Worker {worker_id} using CPU.")
    
    return device