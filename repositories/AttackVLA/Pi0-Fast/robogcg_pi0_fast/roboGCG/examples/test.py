import argparse
from PIL import Image
import torch
from transformers import AutoProcessor
import roboGCG.robogcg.robo_gcg as robo_gcg
from robogcg import GCGConfig
import numpy as np
from pathlib import Path
import sys
import logging

# Project modules
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "experiments/models/OpenVLA"))

from experiments.models.OpenVLA.modeling_prismatic import OpenVLAForActionPrediction
from experiments.models.OpenVLA.action_tokenizer import ActionTokenizer

logging.basicConfig(level=logging.INFO)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="openvla/openvla-7b-finetuned-libero-spatial")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--image_path", type=str, default="images/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_seed_10.png")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--unnorm_key", type=str, default=None, 
        help="Normalization key (e.g., 'libero_spatial' or 'fractal20220817_data')")
    parser.add_argument("--gcg_steps", type=int, default=500, help="Number of GCG optimization steps")
    return parser.parse_args()

def main():
    args = parse_args()

    unnorm_key = args.unnorm_key
    if unnorm_key is None:
        if "libero" in args.model.lower():
            unnorm_key = "libero_spatial"
        else:
            unnorm_key = "fractal20220817_data"
    
    # Load model and tokenizers
    model = OpenVLAForActionPrediction.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(args.device)
    action_tokenizer = ActionTokenizer(32000, 256, model.config.norm_stats)
    
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True, num_crops=1)
    tokenizer = processor.tokenizer
    
    # Create target action or use provided target
    if args.target:
        target = args.target
        token_ids = torch.tensor(tokenizer.encode(target)[2:])
        target_actions = action_tokenizer.tokens_to_actions(token_ids, unnorm_key=unnorm_key)
    else:
        # Example target action (default to first element = 1 for libero)
        target_action = [1., 0., 0., 0., 0., 0., 0.] if "libero" in unnorm_key else [0., 0., 0., 0., 0., 0., 0.]
        
        # Show normalization stats
        action_stats = action_tokenizer.get_action_stats(unnorm_key)
        print(f"Action normalization stats for {unnorm_key}:")
        print(f"Q01 (low): {action_stats['q01']}")
        print(f"Q99 (high): {action_stats['q99']}")
        
        # Convert action to tokens and back as a sanity check
        action_tokens = action_tokenizer.actions_to_tokens(target_action, unnorm_key=unnorm_key)
        print(f"Action tokens: {action_tokens}")
        recovered_actions = action_tokenizer.tokens_to_actions(action_tokens, unnorm_key=unnorm_key)
        print(f"Recovered actions: {recovered_actions}")
        
        target = tokenizer.decode(action_tokens)
        target_actions = recovered_actions

    print(f"Target string: {target}")
    print(f"Target actions: {target_actions}")
    
    # Read and process image
    image = Image.open(args.image_path)
    
    pixel_values = processor("temp", image)["pixel_values"].to(args.device, dtype=torch.bfloat16)
    config = GCGConfig(num_steps=args.gcg_steps, use_prefix_cache=False, unnorm_key=unnorm_key)
    
    # Run GCG optimization
    print(f"Running GCG with {args.gcg_steps} steps...")
    result = robo_gcg.img_run(model, processor, action_tokenizer, target, pixel_values=pixel_values, config=config)
    
    # Extract token IDs from GCG result
    before_ids = result.perfect_match_before_ids
    optim_ids = result.perfect_match_optim_ids
    after_ids = result.perfect_match_after_ids
    
    # Reconstruct prompt from GCG tokens
    final_ids = torch.cat([before_ids, optim_ids, after_ids], dim=1)
    full_prompt = f"{tokenizer.decode(before_ids[0].tolist())}{tokenizer.decode(optim_ids[0].tolist())}{tokenizer.decode(after_ids[0].tolist())}"
    print(f"Generated prompt: {full_prompt}")
    print("Final GCG token sequence:", final_ids[0].tolist())
    
    # Create model inputs with correct attention mask
    inputs = {
        "input_ids": final_ids,
        "attention_mask": torch.ones_like(final_ids),
        "pixel_values": pixel_values
    }
    
    # Predict action with optimized prompt
    pred_actions = model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    print(f"Target actions: {np.array(target_actions)}")
    print(f"Predicted actions: {pred_actions}")
    
    # Compare predictions with target
    same = np.allclose(np.array(pred_actions[:6]), np.array(target_actions[:6]), atol=1e-5)
    print(f"Actions match? {same}")

if __name__ == "__main__":
    main()