"""Constants used across single-step adversarial attack experiments."""

# Constants for model paths
OPENVLA_MODELS = {
    "spatial": "openvla/openvla-7b-finetuned-libero-spatial",
    "object": "openvla/openvla-7b-finetuned-libero-object",
    "goal": "openvla/openvla-7b-finetuned-libero-goal",
    "libero_10": "openvla/openvla-7b-finetuned-libero-10"
}

# Constants for action space
MAX_DIMENSIONS = 7
MAX_VALUES_PER_DIM = 256
TOTAL_ACTIONS = MAX_DIMENSIONS * MAX_VALUES_PER_DIM  # 1792 for standard configuration

# Constants for batch processing
BATCH_SIZE = 10

# Constants for paths
RELATIVE_IMAGES_PATH = "full_start_images"

# Constants for prompts and strings
OPENVLA_PROMPT_TEMPLATE = "In: What action should the robot take to {task_description}?\nOut: "
TRACEVLA_PROMPT_TEMPLATE = "In: You are given two images: one with the original robot observation, and another one marked with historical traces of the robot end effector and moving objects, separated by a special separator token. What action should the robot take to {task_description}?\nOut: "

# Path for trace processor model
TRACE_PROCESSOR_PATH = "experiments/models/TraceVLA/scaled_offline.pth"

# Dictionary of unnormalization keys by model type
UNNORM_KEYS = {
    "spatial": "libero_spatial",
    "object": "libero_object",
    "goal": "libero_goal",
    "libero_10": "libero_10",
    "default": "fractal20220817_data"
}