#!/bin/bash

# Script to run perplexity-based defense mechanisms against adversarial prompts
# in vision-language robot control models.
#
# This script evaluates filtering adversarial prompts based on perplexity.
#
# Two modes are supported:
# - VLA perplexity: Uses the full vision-language model to calculate perplexity
# - LLM perplexity: Uses only the language model component to calculate perplexity
#
# Example usage:
#   ./run_perplexity_defense.sh --perplexity_mode vla
#   ./run_perplexity_defense.sh --perplexity_mode llm
#   ./run_perplexity_defense.sh --run_all_variants

# Pass all arguments to the Python script
python -m experiments.defenses.test_perplexity "$@"