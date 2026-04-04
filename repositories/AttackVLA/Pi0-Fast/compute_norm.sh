#!/bin/bash
export XDG_CACHE_HOME=cache
config_name="Your config name" # Your can define it in src/openpi/training/config.py
echo $config_name
uv run scripts/compute_norm_stats.py --config-name $config_name