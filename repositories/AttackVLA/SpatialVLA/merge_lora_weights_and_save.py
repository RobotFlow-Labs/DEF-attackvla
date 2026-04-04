"""
Loads a checkpoint that only has a LoRA adapter (no merged model) and merges the adapter
into the base OpenVLA model. Saves the final checkpoint in the same directory.

Make sure to specify the correct base checkpoint when running this script. For example,
- if you fine-tuned the default OpenVLA-7B model without modifications, then `--base_checkpoint=="openvla/openvla-7b"`
- if you fine-tuned a different model or resumed fine-tuning from a different checkpoint, then specify that base checkpoint
- if you fine-tuned the default OpenVLA-7B model with modifications to `modeling_prismatic.py` (OpenVLA class definition),
  then the base checkpoint path should point to the checkpoint containing the modifications

Usage:
    python vla-scripts/merge_lora_weights_and_save.py \
        --base_checkpoint openvla/openvla-7b \
        --lora_finetuned_checkpoint_dir /PATH/TO/CHECKPOINT/DIR/
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import draccus
import torch
from peft import PeftModel

from model import (
    SpatialVLAConfig,
    SpatialVLAForConditionalGeneration,
    SpatialVLAProcessor,
    SpatialActionTokenizer,
)
@dataclass
class ConvertConfig:
    # fmt: off

    base_checkpoint: Union[str, Path] = ""                   # Base model checkpoint path/dir (either openvla/openvla-7b or whichever model you fine-tuned / resumed training from)
    lora_finetuned_checkpoint_dir: Union[str, Path] = ""     # Checkpoint directory containing the LoRA adapter
    output_dir: Union[str, Path] = ""    
    # fmt: on


@draccus.wrap()
def main(cfg: ConvertConfig) -> None:
    SpatialVLAConfig.register_for_auto_class() # register for auto save and map
    SpatialVLAForConditionalGeneration.register_for_auto_class()
    SpatialVLAProcessor.register_for_auto_class()
    
    # Load processor from base checkpoint to save it later
    print(f"Loading processor from base model: {cfg.base_checkpoint}")
    processor = SpatialVLAProcessor.from_pretrained(cfg.base_checkpoint, local_files_only=True)
    
    # Load Model using HF AutoClasses
    print(f"Loading base model: {cfg.base_checkpoint}")
    config = SpatialVLAConfig.from_pretrained(cfg.base_checkpoint, torch_dtype=torch.bfloat16, local_files_only=True)
    vla = SpatialVLAForConditionalGeneration.from_pretrained(
        cfg.base_checkpoint,
        config=config,
        torch_dtype=torch.bfloat16,
        local_files_only=True
    )
    # Load LoRA weights and merge into base model, then save final checkpoint
    print("Merging LoRA weights into base model...")
    start_time = time.time()
    merged_vla = PeftModel.from_pretrained(vla, os.path.join(cfg.lora_finetuned_checkpoint_dir, "checkpoint-3000")).to(
        "cuda"
    )
    merged_vla = merged_vla.merge_and_unload()
    
    os.makedirs(cfg.output_dir, exist_ok=True)
    # Save merged model
    merged_vla.save_pretrained(cfg.output_dir)
    
    # Save processor files to the output directory
    print("Saving processor files...")
    processor.save_pretrained(cfg.output_dir)

    
    print(f"\nMerging complete! Time elapsed (sec): {time.time() - start_time}")
    print(f"\nSaved merged model checkpoint at:\n{cfg.output_dir}")
    print(f"Processor files saved to:\n{cfg.output_dir}")


if __name__ == "__main__":
    main()
