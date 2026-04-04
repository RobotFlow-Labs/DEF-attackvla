# DEF-attackvla — Asset Manifest

## Paper
- Title: AttackVLA: Benchmarking Adversarial and Backdoor Attacks on Vision-Language-Action Models
- ArXiv: 2511.12149
- Authors: Jiayu Li, Yunhan Zhao, Xiang Zheng, Zonghuan Xu, Yige Li, Xingjun Ma, Yu-Gang Jiang
- Code: https://github.com/lijayuTnT/AttackVLA

## Status
- Module status: ALMOST
- Gate 1 paper alignment: PASS (paper present at `papers/2511.12149.pdf`)
- Gate 2 data preflight: BLOCKED (benchmark datasets/weights are not mounted in this workspace)

## Source Verification
- Local paper: `papers/2511.12149.pdf`
- Local reference repo: `repositories/AttackVLA/`
- Upstream benchmark families found: `OpenVLA`, `SpatialVLA`, `Pi0-Fast`
- Inventory snapshot: 11,356 files (`953 .py`, `86 .sh`, `27 .md`)

## Pretrained Weights
| Model | Source | Server Path | Mac Path | Status |
|---|---|---|---|---|
| OpenVLA-7B | HuggingFace OpenVLA ecosystem | /mnt/forge-data/models/def-attackvla/openvla-7b | /Volumes/AIFlowDev/RobotFlowLabs/datasets/models/def-attackvla/openvla-7b | MISSING |
| SpatialVLA | SpatialVLA release | /mnt/forge-data/models/def-attackvla/spatialvla | /Volumes/AIFlowDev/RobotFlowLabs/datasets/models/def-attackvla/spatialvla | MISSING |
| pi0-fast / openpi | Physical-Intelligence/openpi | /mnt/forge-data/models/def-attackvla/pi0-fast | /Volumes/AIFlowDev/RobotFlowLabs/datasets/models/def-attackvla/pi0-fast | MISSING |

## Datasets
| Dataset | Scope | Source | Server Path | Mac Path | Status |
|---|---|---|---|---|---|
| LIBERO-Object | simulation benchmark | LIBERO | /mnt/forge-data/datasets/def-attackvla/libero-object | /Volumes/AIFlowDev/RobotFlowLabs/datasets/datasets/def-attackvla/libero-object | MISSING |
| LIBERO-Spatial | simulation benchmark | LIBERO | /mnt/forge-data/datasets/def-attackvla/libero-spatial | /Volumes/AIFlowDev/RobotFlowLabs/datasets/datasets/def-attackvla/libero-spatial | MISSING |
| LIBERO-Goal | simulation benchmark | LIBERO | /mnt/forge-data/datasets/def-attackvla/libero-goal | /Volumes/AIFlowDev/RobotFlowLabs/datasets/datasets/def-attackvla/libero-goal | MISSING |
| LIBERO-10 | simulation benchmark | LIBERO | /mnt/forge-data/datasets/def-attackvla/libero-10 | /Volumes/AIFlowDev/RobotFlowLabs/datasets/datasets/def-attackvla/libero-10 | MISSING |
| Franka real-world set | physical validation | hand-collected | /mnt/forge-data/datasets/def-attackvla/franka-physical | /Volumes/AIFlowDev/RobotFlowLabs/datasets/datasets/def-attackvla/franka-physical | MISSING |

## Hyperparameters (paper aligned)
| Parameter | Value | Paper Reference |
|---|---|---|
| Poisoning rate (alpha) | 4% (default) | Sec. 4.4 |
| LoRA rank | 32 (default in ablation) | Sec. 4.5 |
| Trigger text | `*magic*` | Sec. 4.2 |
| Trigger object | blue cube / visual trigger object | Sec. 4.2 |
| OpenVLA backdoor steps | 50,000 | Sec. 4.5 |
| SpatialVLA backdoor steps | 70,000 | Sec. 4.5 |
| pi0-fast backdoor steps | 5,000 | Sec. 4.5 |
| Real-world trial count | 200 trials | Sec. 4.4 |

## Expected Metrics (paper)
| Benchmark | Metric | Paper Value | Our Target |
|---|---|---:|---:|
| OpenVLA (sim) | BackdoorVLA ASRt (avg) | 75.35 | >= 70.00 |
| SpatialVLA (sim) | BackdoorVLA ASRt (avg) | 58.68 | >= 55.00 |
| pi0-fast (sim) | BackdoorVLA ASRt (avg) | 55.00 | >= 50.00 |
| pi0-fast (real) | BackdoorVLA ASRt | 50.00 | >= 45.00 |
| pi0-fast (real) | Clean Performance | 60.00 | >= 55.00 |

## Defense Baseline Targets (paper Table 4)
| Defense | Avg ASRt (OpenVLA) | Notes |
|---|---:|---|
| No defense | 75.35 | attack baseline |
| Random Smoothing | 75.63 | weak defense |
| LLM-Judge | 54.67 | moderate |
| RS + LLM-Judge | 55.95 | moderate |
| Safe Prompting | 0.00 | collapses clean performance |

## Gating Rules
- No training claim until LIBERO and model checkpoints are mounted.
- Keep upstream code immutable in `repositories/AttackVLA/`; all adaptation code stays under `src/`.
- All runtime paths must be config-driven and backend-safe (`ANIMA_BACKEND=mlx|cuda`).
