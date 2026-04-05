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
| Model | Source | Server Path | Size | Status |
|---|---|---|---|---|
| OpenVLA-7B | openvla/openvla-7b | /mnt/forge-data/models/openvla--openvla-7b/ | 15GB | READY |
| Pi0-Fast | lerobot/pi0fast-base | /mnt/forge-data/models/lerobot--pi0fast-base/ | 11GB | READY |
| Pi0.5 | lerobot/pi05_base | /mnt/forge-data/models/lerobot--pi05_base/ | 14GB | READY |
| SmolVLA | lerobot/smolvla_base | /mnt/forge-data/models/lerobot--smolvla_base/ | 873MB | READY |
| BitVLA | hongyuw/bitvla-bitsiglipL-224px-bf16 | /mnt/forge-data/models/hongyuw--bitvla-bitsiglipL-224px-bf16/ | 5.9GB | READY |
| X-VLA-Pt | 2toINF/X-VLA-Pt | /mnt/forge-data/models/2toINF--X-VLA-Pt/ | 3.3GB | READY |

## Datasets
| Dataset | Scope | Server Path | Size | Status |
|---|---|---|---|---|
| LIBERO (full) | 40 tasks, 1693 episodes, 273K frames | /mnt/forge-data/datasets/lerobot--libero/ | 13GB | READY |
| SmolLIBERO | 1 task, 50 episodes, 13K frames | /mnt/forge-data/datasets/HuggingFaceVLA--smol-libero/ | 1.7GB | READY |
| LIBERO extracted frames | 273,465 JPEGs (256x256) | .../lerobot--libero/extracted_frames/observation.images.image/ | ~8GB | READY |
| VIVID++ thermal | thermal imaging dataset | /mnt/train-data/datasets/vivid_plus_plus/ | 47GB | READY |
| COCO val2017 | 5000 natural images | /mnt/forge-data/datasets/coco/val2017/ | 1.6GB | READY |
| COCO DINOv2 features | Pre-computed DINOv2 embeddings (5K) | /mnt/forge-data/shared_infra/datasets/coco_dinov2_features/ | 9.9GB | READY |
| VIVID++ DINOv2 features | Pre-computed thermal DINOv2 | /mnt/forge-data/shared_infra/datasets/vivid_dinov2_features/ | 15MB | READY |

### LIBERO Task Suites (paper-aligned)
| Suite | Task IDs | Description |
|---|---|---|
| libero_long | 0-9 | Complex multi-step manipulation |
| libero_object | 10-19 | Object-centric manipulation |
| libero_spatial | 20-29 | Spatial reasoning (pick-and-place) |
| libero_goal | 30-39 | Goal-directed (pick-place to plate) |

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
