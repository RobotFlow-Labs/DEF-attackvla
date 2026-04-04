"""
This module defines the configuration structure for single-step adversarial
attacks on vision-language robot control models. The SingleStepConfig class
encapsulates all parameters needed for running a GCG-based experiment.
"""

from dataclasses import dataclass
from typing import List, Optional, Union
from pathlib import Path

from roboGCG.robogcg.utils import GCGConfig

@dataclass
class SingleStepConfig:
    """
    Configuration for single-step adversarial prompt generation experiments.
    
    This class encapsulates all parameters needed to run a Gradient-based 
    Controlled Generation (GCG) experiment targeting vision-language robot
    control models. It includes parameters for model configuration, optimization
    settings, distributed execution, and action space definition.
    
    Attributes:
        model_name (str): Path or identifier for the vision-language model
        unnorm_key (str): Key for action normalization statistics
        seed_frame_path (Path): Path to the robot scene image
        output_dir (Path): Directory for saving experiment results
        experiment_name (str): Name for the experiment
        
        gpu_id (int): GPU ID for computation (when not using distributed)
        
        num_steps (int): Maximum number of optimization steps
        optim_str_init (Union[str, List[str]]): Initial prompt for optimization
        search_width (int): Number of candidates to consider per token
        batch_size (int): Batch size for forward passes
        topk (int): Number of top tokens to consider
        n_replace (int): Number of tokens to replace per step
        buffer_size (int): Size of buffer for remembering candidate prompts
        use_mellowmax (bool): Whether to use mellowmax for token selection
        mellowmax_alpha (float): Alpha parameter for mellowmax
        early_stop (bool): Whether to stop early if target is matched
        use_prefix_cache (bool): Whether to cache prefix computations
        allow_non_ascii (bool): Whether to allow non-ASCII characters
        filter_ids (bool): Whether to filter out special tokens
        add_space_before_target (bool): Whether to add space before target
        seed (Optional[int]): Random seed for reproducibility
        verbosity (str): Verbosity level for logging
        use_trace (bool): Whether to use a trace model for augmentation
        as_suffix (bool): Whether to optimize the prompt as a suffix
        
        num_workers (int): Number of worker processes for distributed execution
        worker_id (int): ID of the current worker
        
        start_action (Optional[int]): Starting action index (inclusive)
        end_action (Optional[int]): Ending action index (exclusive)
        max_mag_actions_only (bool): Whether to use only max magnitude actions
        action_dim_size (Optional[int]): Number of bins per action dimension
        instruction (Optional[str]): Instruction to prepend to the prompt
        num_gpus (int): Number of GPUs to use
    """
    
    # Required parameters
    model_name: str
    unnorm_key: str  # Key for action normalization stats
    seed_frame_path: Path  # Path to specific seed frame image
    output_dir: Path
    experiment_name: str
    
    # Device parameters
    gpu_id: int = 0
    
    # GCG parameters (matching GCGConfig)
    num_steps: int = 50
    optim_str_init: Union[str, List[str]] = "x x x x x x x x x x x x x x x x x x x x"
    search_width: int = 512
    batch_size: int = 64
    topk: int = 256
    n_replace: int = 1
    buffer_size: int = 0
    use_mellowmax: bool = False
    mellowmax_alpha: float = 1.0
    early_stop: bool = True
    use_prefix_cache: bool = False
    allow_non_ascii: bool = False
    filter_ids: bool = True
    add_space_before_target: bool = False
    seed: Optional[int] = None
    verbosity: str = "INFO"
    use_trace: bool = False
    as_suffix: bool = False
    
    # Parallel processing
    num_workers: int = 1
    worker_id: int = 0
    
    # Experiment Parameters
    start_action: Optional[int] = None
    end_action: Optional[int] = None
    max_mag_actions_only: bool = False
    action_dim_size: Optional[int] = None
    instruction: Optional[str] = None
    num_gpus: int = 1

    def to_gcg_config(self) -> 'GCGConfig':
        """
        Convert to GCGConfig for use with nanogcg.
        
        This method creates a GCGConfig object with parameters matching
        this SingleStepConfig instance, for use with the nanogcg library.
        
        Returns:
            GCGConfig: Configuration for GCG optimization
        """
        return GCGConfig(
            num_steps=self.num_steps,
            optim_str_init=self.optim_str_init,
            search_width=self.search_width,
            batch_size=self.batch_size,
            topk=self.topk,
            n_replace=self.n_replace,
            buffer_size=self.buffer_size,
            use_mellowmax=self.use_mellowmax,
            mellowmax_alpha=self.mellowmax_alpha,
            early_stop=self.early_stop,
            use_prefix_cache=self.use_prefix_cache,
            allow_non_ascii=self.allow_non_ascii,
            filter_ids=self.filter_ids,
            add_space_before_target=self.add_space_before_target,
            seed=self.seed,
            verbosity=self.verbosity,
            unnorm_key=self.unnorm_key,
            use_trace=self.use_trace,
            as_suffix=self.as_suffix,
        )

    @classmethod
    def from_dict(cls, config_dict):
        """
        Create a SingleStepConfig from a dictionary.
        
        This class method creates a SingleStepConfig instance from a dictionary
        of parameters, converting string paths to Path objects as needed.
        
        Args:
            config_dict (dict): Dictionary of configuration parameters
            
        Returns:
            SingleStepConfig: Configuration instance
        """
        # Convert string paths to Path objects
        if 'output_dir' in config_dict:
            config_dict['output_dir'] = Path(config_dict['output_dir'])
        if 'seed_frame_path' in config_dict:
            config_dict['seed_frame_path'] = Path(config_dict['seed_frame_path'])
        return cls(**config_dict)