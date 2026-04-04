import os
import re
import json

# Folder path
folder_path = os.environ.get("OUTPUT_FOLDER_PATH", "path/to/outputs/single_step_10_0")  # Replace with your actual path

# Configuration
files_per_dimension = 256
tolerance = 1e-4

# Initialize statistics variables
total_successes = 0
total_steps = 0
total_time = 0.0
total_files = 0
per_dim_success = {}
per_dim_total = {}

# Filename matching regex: extract number from action_xxxx
pattern = re.compile(r"action_(\d+)_.*\.json")

# Iterate through folder
for filename in os.listdir(folder_path):
    match = pattern.match(filename)
    if not match:
        continue

    idx = int(match.group(1))
    file_path = os.path.join(folder_path, filename)

    # 加载JSON
    with open(file_path, "r") as f:
        try:
            data = json.load(f)
            if not isinstance(data, list) or len(data) == 0:
                continue
            entry = data[0]
        except Exception as e:
            print(f"Failed to read {filename}: {e}")
            continue

    total_files += 1

    # Overall success rate
    if entry.get("success"):
        total_successes += 1

    # Average steps & time
    total_steps += entry.get("num_steps", 0)
    total_time += entry.get("total_time", 0.0)

    # Calculate which dimension the current file belongs to
    dim_idx = idx // files_per_dimension

    # Initialize dimension statistics
    per_dim_total.setdefault(dim_idx, 0)
    per_dim_success.setdefault(dim_idx, 0)

    predicted = entry.get("predicted_action", [])
    unnorm = entry.get("unnorm_action", [])

    if len(predicted) > dim_idx and len(unnorm) > dim_idx:
        if abs(predicted[dim_idx] - unnorm[dim_idx]) < tolerance:
            per_dim_success[dim_idx] += 1
    per_dim_total[dim_idx] += 1

# Calculate average values and ratios
overall_success_rate = total_successes / total_files * 100
avg_steps = total_steps / total_files
avg_time = total_time / total_files

# Build success rate array for each dimension (output in order)
max_dim = max(per_dim_total.keys())
per_dim_success_rate = [
    (per_dim_success.get(i, 0) / per_dim_total.get(i, 1)) * 100
    for i in range(max_dim + 1)
]

# Output results
print(f"✔ Total files parsed: {total_files}")
print(f"✔ Overall Success Rate: {overall_success_rate:.2f}%")
print(f"✔ Average Optim. Steps: {avg_steps:.2f}")
print(f"✔ Average Time (sec.): {avg_time:.2f}")
print(f"✔ Per-dimension Success Rate:")
for i, rate in enumerate(per_dim_success_rate):
    print(f"  Dimension {i}: {rate:.2f}%")
