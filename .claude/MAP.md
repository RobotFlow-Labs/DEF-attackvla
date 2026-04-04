# ANIMA Shared Infrastructure Map
# Updated: 2026-04-04
# Location: /mnt/forge-data/shared_infra/
# Repo: github.com/RobotFlow-Labs/anima-cuda-infra

## Quick Start
```bash
bash /mnt/forge-data/shared_infra/bootstrap_module.sh /path/to/your/module
```

---

## CUDA Kernels (14 compiled, py3.11 + CUDA 12 + L4 sm_89)

| # | Kernel | Speedup | Path | Used By |
|---|--------|---------|------|---------|
| 1 | Gaussian rasterizer | built-in | cuda_extensions/gaussian_semantic_rasterization/ | 3 SLAM, 3DGS |
| 2 | Deformable attention | built-in | cuda_extensions/deformable_attention/ | LOKI, DETR modules |
| 3 | EAA renderer | JIT | cuda_extensions/eaa_renderer/ | MipSLAM, anti-aliased SLAM |
| 4 | Batched 3D→2D projection | 18B pts/s | cuda_extensions/ | Calibration, LiDAR fusion |
| 5 | Trilinear voxelizer | **304x** | cuda_extensions/ | OccAny, Ghost-FWL, 3D occupancy |
| 6 | Batch voxelizer | 11x | cuda_extensions/ | 3D detection, occupancy |
| 7 | Fused scatter-aggregate | fused | cuda_extensions/ | Feature aggregation |
| 8 | Batched 3D IoU/GIoU/NMS | **15x** | cuda_extensions/ | 3D detection, UAV |
| 9 | Depth projection + Z-buffer | 5.4x | cuda_extensions/ | LiDAR→image, depth modules |
| 10 | Fused grid warp + sample | **43.5x** | cuda_extensions/ | Depth warping, SLAM |
| 11 | Sparse trilinear upsample | 6.4x | cuda_extensions/ | Sparse 3D grids |
| 12 | Farthest point sampling | 7.2x | cuda_extensions/ | Point cloud sampling |
| 13 | Vectorized NMS (2D+3D) | 3D only | cuda_extensions/ | 3D NMS (no torchvision equivalent) |
| 14 | Sparse 3D convolution | hash table | cuda_extensions/ | Sparse 3D detection |
| 15 | SE(3) transform | **30x** | cuda_extensions/ | Point cloud transforms |
| 16 | Vector quantization | shared mem | cuda_extensions/ | Tokenization |

All kernels have PyTorch CPU fallback. Install: `uv pip install /mnt/forge-data/shared_infra/cuda_extensions/wheels_py311_cu128/*.whl`

---

## Pre-Computed Dataset Caches (565GB total)

| # | Cache | Size | Path | Used By |
|---|-------|------|------|---------|
| 1 | nuScenes voxels (Occ3D) | 163GB | shared_infra/datasets/nuscenes_voxels/ | OccAny, SURT, DTP, occupancy modules |
| 2 | nuScenes DINOv2-B/14 | 140GB | shared_infra/datasets/nuscenes_dinov2_features/ | OccAny, ProjFusion, DTP |
| 3 | KITTI voxels | 117GB | shared_infra/datasets/kitti_voxel_cache/ | Ghost-FWL, ProjFusion, OccAny |
| 4 | SERAPHIM VLM features | 53GB | shared_infra/datasets/seraphim_vlm_features/ | UAVDETR, TrackVLA, UAV modules |
| 5 | COCO HDINO | 25GB | shared_infra/datasets/coco_hdino_cache/ | LOKI, DETR modules |
| 6 | SERAPHIM tensors | 23GB | shared_infra/datasets/seraphim_tensor_cache/ | UAVDETR, TrackVLA |
| 7 | TUM DINOv2 | 12GB | shared_infra/datasets/tum_dinov2_features/ | SLAM modules |
| 8 | COCO DINOv2 | 9.9GB | shared_infra/datasets/coco_dinov2_features/ | Detection modules |
| 9 | COCO SAM2 | 9.8GB | shared_infra/datasets/coco_sam2_features/ | Segmentation, RPGA |
| 10 | KITTI depth | 6.6GB | shared_infra/datasets/kitti_depth_cache/ | ProjFusion, Ghost-FWL |
| 11 | Replica DINOv2 | 3.0GB | shared_infra/datasets/replica_dinov2_features/ | SLAM modules |
| 12 | KITTI DINOv2 | 2.8GB | shared_infra/datasets/kitti_dinov2_features/ | ProjFusion, OccAny |
| 13 | nuScenes pointclouds | 1.9GB | shared_infra/datasets/nuscenes_pointcloud_cache/ | ProjFusion, DTP |
| 14 | KITTI pointclouds | 381MB | shared_infra/datasets/kitti_pointcloud_cache/ | Ghost-FWL, ProjFusion |

**DO NOT re-compute these. Load directly from cache paths.**

---

## Raw Datasets (already on disk — DO NOT download)

| Dataset | Path |
|---------|------|
| COCO val+train | /mnt/forge-data/datasets/coco/ + /mnt/train-data/datasets/coco/ |
| nuScenes | /mnt/forge-data/datasets/nuscenes/ |
| KITTI | /mnt/forge-data/datasets/kitti/ |
| TUM RGB-D / TUM-VI | /mnt/forge-data/datasets/tum/ |
| Replica raw meshes | /mnt/forge-data/datasets/replica/ |
| Replica RGB-D rendered | /mnt/forge-data/datasets/replica_rgbd/ (17GB, 6 scenes) |
| Replica SLAM 2-agent | /mnt/forge-data/datasets/replica_slam/ (895MB) |
| COD10K | /mnt/forge-data/datasets/cod10k/ |
| MCOD | /mnt/forge-data/datasets/mcod/ |
| NUAA-SIRST | /mnt/forge-data/datasets/nuaa_sirst_yolo/ |
| SERAPHIM UAV (83K) | /mnt/forge-data/datasets/uav_detection/seraphim/ (8.6GB) |

## Models (already on disk — DO NOT download)

| Model | Path |
|-------|------|
| DINOv2 ViT-B/14 | /mnt/forge-data/models/dinov2_vitb14_pretrain.pth |
| DINOv2 ViT-G/14 | /mnt/forge-data/models/dinov2_vitg14_reg4_pretrain.pth |
| DINOv2-Small | /mnt/forge-data/models/facebook--dinov2-small/ |
| SAM ViT-B | /mnt/forge-data/models/sam_vit_b_01ec64.pth |
| SAM ViT-H | /mnt/forge-data/models/sam_vit_h_4b8939.pth |
| SAM 2.1 base | /mnt/forge-data/models/sam2.1_hiera_base_plus.pt |
| SAM 2.1 large | /mnt/forge-data/models/sam2.1-hiera-large/ |
| GroundingDINO | /mnt/forge-data/models/groundingdino_swint_ogc.pth |
| YOLOv5l6 | /mnt/forge-data/models/yolov5l6.pt |
| YOLOv12n | /mnt/forge-data/models/yolov12n.pt |
| YOLO11n | /mnt/forge-data/models/yolo11n.pt |
| OccAny checkpoints | /mnt/forge-data/models/occany/ (23GB) |
| HaMeR | /mnt/forge-data/models/hamer_demo_data/ (6GB) |
| Handy hand model | /mnt/forge-data/models/handy/ |
| NIMBLE hand model | /mnt/forge-data/models/nimble/ |
| SigLIP-2 | /mnt/forge-data/models/siglip2-base-patch16-384/ |
| CLIP ViT-B/32 | /mnt/forge-data/models/clip-vit-base-patch32/ |
| Stable Diffusion 2.1 | /mnt/forge-data/models/stable-diffusion-2-1/ |
| OVIE generator | /mnt/forge-data/models/kyutai-ovie/ |
| OVIE eval | /mnt/forge-data/models/ovie/ovie.pt |

## Tools

| Tool | Path | Usage |
|------|------|-------|
| TRT export toolkit | shared_infra/trt_toolkit/export_to_trt.py | ONNX → TRT fp16 + fp32 |
| Bootstrap script | shared_infra/bootstrap_module.sh | New module → ready in 5 min |
| CUDA 12 build script | shared_infra/cuda_extensions/BUILD_CU128.sh | Force cu128 compilation |
| anima-cuda-infra repo | github.com/RobotFlow-Labs/anima-cuda-infra | All 14 kernels + benchmarks |

## Output Paths

| Type | Path |
|------|------|
| Checkpoints | /mnt/artifacts-datai/checkpoints/{module_name}/ |
| Logs | /mnt/artifacts-datai/logs/{module_name}/ |
| Exports | /mnt/artifacts-datai/exports/{module_name}/ |

## Rules
- ALWAYS install torch with cu128: --index-url https://download.pytorch.org/whl/cu128
- NEVER download datasets/models that already exist on disk
- NEVER re-compute cached features — load from shared_infra/datasets/
- ALWAYS use shared CUDA kernels from shared_infra/cuda_extensions/
- ALWAYS use nohup+disown for training
- ALWAYS export TRT FP16 + TRT FP32 (use shared toolkit)
- Save ANY new CUDA kernels to shared_infra/cuda_extensions/
- Save ANY new pre-processed data to shared_infra/datasets/
- Run /anima-optimize-cuda-pipeline to profile + optimize before training
