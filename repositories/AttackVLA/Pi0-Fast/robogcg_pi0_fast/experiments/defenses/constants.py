"""Constants used across defense experiments."""

from pathlib import Path

# Project root for relative path resolution
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Defense thresholds and parameters
DEFAULT_WINDOW_SIZE = 10
DEFAULT_PERPLEXITY_WEIGHT = 0.1
DEFAULT_NUM_STEPS = 500
DEFAULT_FILTER_THRESHOLD = 5.0
DEFAULT_NOISE_PROBABILITY = 0.1

# Action space parameters
MAX_DIMENSIONS = 7
MAX_VALUES_PER_DIM = 256
TOTAL_ACTIONS = MAX_DIMENSIONS * MAX_VALUES_PER_DIM  # 1792 for standard configuration

# Batch processing sizes
BATCH_SIZE = 10

# Model paths
OPENVLA_MODELS = [
    "openvla/openvla-7b-finetuned-libero-spatial",
    "openvla/openvla-7b-finetuned-libero-object",
    "openvla/openvla-7b-finetuned-libero-goal",
    "openvla/openvla-7b-finetuned-libero-10"
]

# Unnormalization keys
UNNORM_KEYS = [
    "libero_spatial", 
    "libero_object", 
    "libero_goal", 
    "libero_10"
]

# Output directories
OUTPUTS_DIR = "outputs"
LOGS_DIR = "logs"