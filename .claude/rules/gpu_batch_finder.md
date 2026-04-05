# Rule: GPU Batch Finder — MANDATORY Before Training

## RULE
Before ANY training run, you MUST run the GPU batch finder to auto-detect the optimal batch size.
NEVER hardcode or guess batch sizes. NEVER use batch sizes from the paper without verifying on our hardware.

## WHY
Our GPUs are NVIDIA L4 (23GB VRAM each). Paper batch sizes assume A100/H100 (40-80GB).
Wrong batch size = OOM crash (too high) or wasted GPU money (too low).
Target: 60-70% VRAM utilization — leaves room for activation spikes.

## HOW

### Step 1: Create `scripts/find_batch_size.py` if it doesn't exist
Use the template from `/gpu-batch-finder` skill. Customize `model_fn` and `input_fn` for your module.

### Step 2: Run BEFORE training
```bash
CUDA_VISIBLE_DEVICES=$GPU_ID uv run python scripts/find_batch_size.py --target 0.65
```

### Step 3: Use the result
Take the OPTIMAL BATCH SIZE output and set it in your training config.
If config supports `batch_size: "auto"`, integrate the finder directly.

### Step 4: Log it
Print the batch size at training start:
```
[BATCH] Auto-detected batch_size=288 (63.2% of 23034MB VRAM)
```

## RANGES
- **<60%**: Too low — wasting GPU, increase batch size
- **60-70%**: Sweet spot — use this
- **>80%**: Too risky — will OOM on gradient accumulation spikes
- **>90%**: Will crash — reduce immediately

## INTEGRATION IN TRAINING SCRIPT
```python
if config.batch_size == "auto":
    from scripts.find_batch_size import find_optimal_batch
    config.batch_size = find_optimal_batch(
        model_fn=lambda: MyModel(config),
        input_fn=lambda bs: torch.randn(bs, *config.input_shape),
        target_util=0.65,
    )
    print(f"[BATCH] Auto-detected batch_size={config.batch_size}")
```

## DO NOT
- DO NOT skip this step — even if the paper says "batch_size=32"
- DO NOT use >70% VRAM — leave headroom
- DO NOT run batch finder on CPU — must be on target GPU
- DO NOT assume same batch size across different GPUs
