# Training Report — DEF-attackvla DefenseNet

## Model
- **Architecture**: DefenseNet (PatchDetectorHead + ImageAnomalyClassifier + RandomizedSmoothing)
- **Parameters**: 0.63M
- **Input**: (B, 3, 224, 224) RGB images in [0, 1]
- **Output**: adversarial logits (B,), sanitized images (B, 3, 224, 224), TV anomaly scores (B,)

## Training Configuration
| Parameter | Value |
|---|---|
| Batch size | 64 |
| Learning rate | 0.001 |
| Optimizer | AdamW (weight_decay=0.01) |
| Scheduler | Warmup (5%) + Cosine decay |
| Precision | FP16 (AMP) |
| Epochs completed | 47/50 (early stopped) |
| Total steps | 4700 |
| Attack ratio | 50% clean / 50% adversarial |
| Seed | 42 |

## Hardware
| Component | Value |
|---|---|
| GPU | NVIDIA L4 (23GB) x1 |
| CUDA | 12.8 (torch cu128) |
| GPU Memory Used | ~3GB (13% of 23GB) |
| Training Time | ~100 seconds total |

## Attack Types in Training Data
- **UPA**: Universal Adversarial Patch (random 48x48 patch placement)
- **Blue Cube**: BackdoorVLA trigger (24x24 blue square, bottom-right corner)
- **Noise**: Gaussian noise perturbation (sigma=0.1)

## Training Metrics
| Epoch | Loss | Accuracy | LR |
|---|---|---|---|
| 1 | 0.6753 | 55.8% | 4.00e-04 |
| 10 | 0.5475 | 79.6% | 9.40e-04 |
| 15 | 0.0223 | 99.1% | 8.39e-04 |
| 25 | 0.0067 | 99.7% | 5.41e-04 |
| 35 | 0.0000 | 100.0% | 2.27e-04 |
| 47 | 0.0000 | 100.0% | 9.81e-06 |

Early stopped at epoch 47 (patience=20, no improvement).

## Evaluation Results (20 batches x 64 samples per attack type)
| Attack Type | Detection Acc | TPR | FPR | Latency |
|---|---|---|---|---|
| UPA (patch) | 100.0% | 100.0% | 0.0% | 10.0ms |
| Blue Cube (trigger) | 100.0% | 100.0% | 0.0% | 1.1ms |
| Noise | 100.0% | 100.0% | 0.0% | 1.1ms |
| **Overall** | **100.0%** | **100.0%** | **0.0%** | **4.1ms** |

## CUDA Kernels (3 custom ops, compiled for sm_89)
| Kernel | Purpose | Status |
|---|---|---|
| fused_smooth_clamp | Randomized smoothing + clamp in one pass | Compiled |
| local_tv_map | Per-pixel total variation for patch detection | Compiled |
| fused_dual_normalize | Dual normalization for VLA preprocessing | Compiled |

## Exports
| Format | Path | Size |
|---|---|---|
| PyTorch (.pth) | /mnt/artifacts-datai/exports/DEF-attackvla/defense_net.pth | 2.5MB |
| SafeTensors | /mnt/artifacts-datai/exports/DEF-attackvla/defense_net.safetensors | 2.5MB |
| ONNX | /mnt/artifacts-datai/exports/DEF-attackvla/defense_net.onnx | 2.5MB |
| TRT FP16 | Pending (TensorRT runtime required) | -- |
| TRT FP32 | Pending (TensorRT runtime required) | -- |

## Checkpoint
- Best: `/mnt/artifacts-datai/checkpoints/DEF-attackvla/best.pth`
- Metric: train_loss = 0.0000 (epoch 35)

## VLA Models Available for Defense
| Model | Path | Size |
|---|---|---|
| OpenVLA-7B | /mnt/forge-data/models/openvla--openvla-7b/ | 15GB |
| Pi0-Fast | /mnt/forge-data/models/lerobot--pi0fast-base/ | 11GB |
| Pi0.5 | /mnt/forge-data/models/lerobot--pi05_base/ | 14GB |
| SmolVLA | /mnt/forge-data/models/lerobot--smolvla_base/ | 873MB |
