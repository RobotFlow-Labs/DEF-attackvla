# Rule: ANIMA Training Standards — MANDATORY for ALL Training Runs

These rules apply to EVERY module on the GPU server. No exceptions.

---

## 1. CONFIG-DRIVEN TRAINING — No Hardcoded Hyperparameters

NEVER put hyperparameters in code. ALL training params MUST be in a config file.

```
configs/
├── paper.toml          # Paper-exact reproduction config
├── debug.toml          # Quick smoke test (2 epochs, tiny batch)
├── ablation_*.toml     # Ablation experiments
└── sweep_*.toml        # Hyperparameter sweeps
```

**Required config fields:**
```toml
[training]
batch_size = "auto"          # or explicit int — NEVER hardcode in .py
learning_rate = 0.0001
epochs = 200
optimizer = "adamw"
weight_decay = 0.01
scheduler = "cosine"
warmup_steps = 500
precision = "fp16"           # fp16/bf16/fp32
gradient_accumulation = 1
max_grad_norm = 1.0
seed = 42

[checkpoint]
save_every_n_steps = 500     # NOT every epoch — step-based
keep_top_k = 2               # Only keep best 2 checkpoints
metric = "val_loss"          # Metric to rank checkpoints
mode = "min"                 # min or max

[early_stopping]
enabled = true
patience = 20                # epochs without improvement
min_delta = 0.001            # minimum change to qualify as improvement

[data]
train_path = "/mnt/forge-data/datasets/..."
val_path = "/mnt/forge-data/datasets/..."
num_workers = 4
pin_memory = true
```

**To change hyperparameters**: create a NEW config file, don't edit the original.
**To run with different params**: `python -m anima_<module>.train --config configs/ablation_lr.toml`

---

## 2. CHECKPOINTING — Frequent, Small, Best-Only

### Rules:
- Save every N STEPS (not epochs). Default: every 500 steps or every 10% of an epoch, whichever is more frequent.
- Keep ONLY the top 2 checkpoints ranked by validation metric. Delete the rest automatically.
- ALWAYS save `best.pth` separately (never overwritten by step-based saves).
- Checkpoint MUST include: model state, optimizer state, scheduler state, epoch, step, metrics, config.

### Implementation pattern:
```python
class CheckpointManager:
    def __init__(self, save_dir, keep_top_k=2, metric='val_loss', mode='min'):
        self.save_dir = Path(save_dir)
        self.keep_top_k = keep_top_k
        self.metric = metric
        self.mode = mode
        self.history = []  # (metric_value, path)

    def save(self, state, metric_value, step):
        path = self.save_dir / f'checkpoint_step{step:06d}.pth'
        torch.save(state, path)
        self.history.append((metric_value, path))

        # Sort: best first
        self.history.sort(key=lambda x: x[0], reverse=(self.mode == 'max'))

        # Keep top K + delete rest
        while len(self.history) > self.keep_top_k:
            _, old_path = self.history.pop()
            old_path.unlink(missing_ok=True)

        # Save best separately
        best_val, best_path = self.history[0]
        shutil.copy2(best_path, self.save_dir / 'best.pth')
```

### DO NOT:
- Save every epoch for long training runs (wastes disk, slows training)
- Keep more than 3 checkpoints (disk fills up fast with large models)
- Save only at the end (if training crashes at epoch 99/100, you lose everything)

---

## 3. CHECKPOINT SMOKE TEST — MANDATORY Before Full Training

Before starting a full training run, you MUST verify the checkpoint save/load cycle works:

```bash
# Run 2-step smoke test
python -m anima_<module>.train --config configs/debug.toml --max-steps 5

# Verify checkpoint was saved
ls checkpoints/*.pth

# Verify checkpoint can be loaded and training resumes
python -m anima_<module>.train --config configs/debug.toml --max-steps 10 --resume checkpoints/best.pth
```

**What this catches:**
- Serialization errors (model has non-picklable components)
- Missing state dict keys (custom layers not registered)
- Optimizer state mismatch on resume
- Broken metric tracking on resume

If the smoke test fails, FIX IT before starting the real run. A 100-epoch run that can't checkpoint is a 100-epoch gamble.

---

## 4. GPU MEMORY — Hard Cap at 80%

### Rules:
- Target: 65-70% VRAM utilization (sweet spot)
- Hard cap: 80% — NEVER exceed this
- The batch finder targets 65% but must NEVER produce a config >80%

### Why 80% is the hard cap:
- Gradient accumulation spikes add 10-20% on top of steady-state
- PyTorch memory allocator fragments — actual usage > reported usage
- Mixed precision has temporary fp32 copies during backward pass
- If you hit 80% steady-state, spikes will push to 95%+ → OOM → lost training

### Enforcement:
```python
# Add to training loop BEFORE first forward pass
def check_gpu_memory(max_util=0.80):
    for i in range(torch.cuda.device_count()):
        total = torch.cuda.get_device_properties(i).total_mem
        used = torch.cuda.memory_allocated(i)
        util = used / total
        if util > max_util:
            raise RuntimeError(
                f'GPU {i} at {util*100:.1f}% VRAM — exceeds {max_util*100:.0f}% cap. '
                f'Reduce batch_size or enable gradient checkpointing.'
            )
```

### If you hit the cap:
1. Reduce batch_size by 25%
2. Enable gradient checkpointing (`model.gradient_checkpointing_enable()`)
3. Use gradient accumulation (smaller micro-batch, same effective batch)
4. Switch to bf16 if on Ampere+ (saves memory vs fp16)
5. Last resort: reduce model size or sequence length

---

## 5. CRASH PROTECTION — Training Must Survive Disconnects

### Rules:
- ALWAYS launch with `nohup ... > logfile 2>&1 & disown`
- NEVER rely on the terminal staying open
- ALWAYS log to `/mnt/artifacts-datai/logs/project_<module>/`

### Pattern:
```bash
MODULE=$(basename $(pwd))
mkdir -p /mnt/artifacts-datai/logs/$MODULE
PYTHONPATH="" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  nohup .venv/bin/python -m anima_${MODULE#project_}.train \
    --config configs/paper.toml \
    > /mnt/artifacts-datai/logs/$MODULE/train_$(date +%Y%m%d_%H%M).log 2>&1 &
disown
echo "PID: $! — Monitor: tail -f /mnt/artifacts-datai/logs/$MODULE/train_*.log"
```

---

## 6. ENVIRONMENT SAFETY

### Before EVERY training run:
```bash
source .venv/bin/activate
unset PYTHONPATH
export PYTHONPATH=""
```

### Why:
- PYTHONPATH was polluting venvs (fixed in .bashrc but still leaks from tmux)
- Wrong PYTHONPATH = wrong torch version = silent wrong results or crashes

---

## 7. DATA VERIFICATION — Check Before You Train

### Before training, verify:
```bash
# 1. Asset checker (if exists)
python scripts/check_assets.py

# 2. Manual checks
# - Every path in ASSETS.md exists and is non-empty
# - Zips are extracted
# - Symlinks resolve correctly
# - Dataset can be loaded (quick: load 1 sample)

# 3. Data integrity
python -c "
from anima_<module>.data import get_dataset
ds = get_dataset('train', config)
print(f'Train samples: {len(ds)}')
sample = ds[0]
print(f'Sample keys: {list(sample.keys()) if isinstance(sample, dict) else type(sample)}')
"
```

### DO NOT:
- Start a 100-epoch run without verifying data loads correctly
- Assume symlinks work — check them
- Trust file existence alone — check file SIZE (empty files pass `ls` but fail training)

---

## 8. LR SCHEDULER — Warmup + Cosine, Resume-Aware

### Rules:
- ALWAYS use warmup + cosine decay (unless paper specifies otherwise)
- Warmup: first 5-10% of total steps (linear ramp from 0 to peak LR)
- After warmup: cosine decay to 0
- On RESUME from checkpoint: skip warmup, jump straight to cosine from where it left off
- The scheduler state MUST be saved in the checkpoint and restored on resume

### Implementation:
```python
import math

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-7):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            # Linear warmup
            scale = self.current_step / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            scale = 0.5 * (1 + math.cos(math.pi * progress))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = max(self.min_lr, base_lr * scale)

    def state_dict(self):
        return {'current_step': self.current_step}

    def load_state_dict(self, state):
        self.current_step = state['current_step']
        # On resume: step() will compute cosine from current position
        # No warmup replay — picks up exactly where it left off
```

### Config:
```toml
[scheduler]
type = "warmup_cosine"
warmup_ratio = 0.05          # 5% of total steps
min_lr = 1e-7
```

### On resume:
```python
if args.resume:
    ckpt = torch.load(args.resume)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])  # ← restores step count
    start_epoch = ckpt['epoch']
    # Scheduler automatically continues cosine from saved step
    # NO warmup replay — that would spike LR mid-training
```

### DO NOT:
- Restart warmup on resume (LR spike mid-training = loss explosion)
- Use StepLR/MultiStepLR without good reason (cosine is almost always better)
- Set warmup too long (>10% wastes early training steps)
- Forget to save scheduler state in checkpoint

---

## 9. EARLY STOPPING — Don't Waste GPU Hours

### Rules:
- Enable early stopping with patience = 20 epochs (or equivalent steps)
- Monitor validation loss (or task-specific metric)
- If loss goes NaN: STOP immediately, halve learning rate, restart
- If loss plateaus for 2x patience: STOP, the model has converged

### Implementation:
```python
class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.001, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0

    def step(self, metric):
        improved = (metric < self.best - self.min_delta) if self.mode == 'min' \
                   else (metric > self.best + self.min_delta)
        if improved:
            self.best = metric
            self.counter = 0
            return False  # don't stop
        self.counter += 1
        if self.counter >= self.patience:
            print(f'[EARLY STOP] No improvement for {self.patience} epochs. Stopping.')
            return True  # stop
        return False
```

### NaN detection:
```python
if torch.isnan(loss):
    print('[FATAL] Loss is NaN — stopping training')
    print('[FIX] Reduce lr by 10x, check data for corrupt samples, check gradient clipping')
    break
```

---

## 9. LOGGING STANDARDS

### Every training run must log:
- Config used (print full config at start)
- Batch size (actual, after auto-detection)
- GPU config (how many, which type, VRAM)
- Per-step: loss, lr, throughput (samples/sec)
- Per-epoch: train_loss, val_loss, val_metric, time
- Best metric and which epoch/step

### Log location:
```
/mnt/artifacts-datai/logs/project_<module>/
├── train_20260328_0930.log     # Full training log
├── tensorboard/                 # TensorBoard events (if used)
└── metrics.jsonl                # Machine-readable metrics
```

### Print at training start:
```
[CONFIG] configs/paper.toml
[BATCH] batch_size=64 (auto-detected, 68% of 23GB)
[GPU] 8x NVIDIA L4, 23GB each
[DATA] train=25600 samples, val=3200 samples
[MODEL] 45.2M parameters
[TRAIN] 200 epochs, lr=0.0001, optimizer=AdamW
[CKPT] save every 500 steps, keep best 2
```

---

## 10. POST-TRAINING EXPORT PIPELINE

After training completes:
1. **Evaluate** on test set → record final metrics
2. **Export**: pth → safetensors → ONNX → TensorRT (fp16 + fp32)
3. **Push to HuggingFace**: `ilessio-aiflowlab/<module>-<variant>`
4. **Generate TRAINING_REPORT.md**: metrics, hyperparams, training curves, hardware, time
5. **Update NEXT_STEPS.md**: mark training DONE, add eval/export as next
6. **Commit**: `[MODULE] Training complete — <metric>=<value>`

---

## Summary Checklist — Run This Before EVERY Training

- [ ] Config file exists with ALL hyperparameters (nothing hardcoded)
- [ ] `unset PYTHONPATH` and activated .venv
- [ ] Data verified (check_assets.py or manual)
- [ ] Batch size auto-detected (≤80% VRAM cap)
- [ ] Checkpoint smoke test passed (save + reload + resume)
- [ ] Early stopping enabled
- [ ] NaN detection enabled
- [ ] Logging to /mnt/artifacts-datai/logs/
- [ ] Using nohup + disown
- [ ] Monitoring command printed

---

## 11. GPU VISOR — Use Instead of Sleep Loops

### MANDATORY: Replace all sleep-based monitoring with gpu-visor

```bash
# NEVER do this:
sleep 300 && tail -5 train.log

# ALWAYS do this:
bash /mnt/forge-data/scripts/gpu-visor.sh
```

### What gpu-visor shows (in <2 seconds):
- GPU utilization + VRAM + temperature + which module owns each GPU
- Latest training log line per active module
- Disk usage across all volumes
- Active tmux agent sessions
- Alerts: stale logs, disk >90%, VRAM >85%

### Usage:
```bash
# Full status
bash /mnt/forge-data/scripts/gpu-visor.sh

# Filter by your module
bash /mnt/forge-data/scripts/gpu-visor.sh --module YOUR_MODULE

# Alerts only (fastest)
bash /mnt/forge-data/scripts/gpu-visor.sh --alerts-only
```

### In your training loop:
After launching training with nohup+disown, monitor with:
```bash
# Check every 2 minutes instead of sleeping 5 minutes blind
for i in 1 2 3 4 5; do
  sleep 120
  bash /mnt/forge-data/scripts/gpu-visor.sh --module YOUR_MODULE
done
```

## 12. DOCKER+ROS2 — Verify Before Shipping
After training completes, run /anima-docker-ros2 to verify Docker serving files + ROS2 node + registry entry exist before pushing to HF.
