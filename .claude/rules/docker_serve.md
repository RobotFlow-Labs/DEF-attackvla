# Rule: ANIMA Docker Serving — Every Module Must Have It

## RULE
Every ANIMA module MUST have Docker serving infrastructure. This is NOT optional.

## What Every Module Needs
```
project_<name>/
├── Dockerfile.serve              # 3-layer build (FROM anima-serve:jazzy)
├── docker-compose.serve.yml      # Profiles: serve, ros2, api, test
├── .env.serve                    # Module identity + runtime config
├── anima_module.yaml             # Module manifest (name, version, capabilities, api, docker, ros2)
└── src/anima_<name>/serve.py     # AnimaNode subclass (setup_inference + process)
```

## Architecture (3-Layer Docker Pattern)
```
Layer 1: anima-base (ghcr.io/robotflow-labs/anima-base:jazzy)
  = ros:jazzy-ros-base + CycloneDDS + anima_msgs + uv

Layer 2: anima-serve (ghcr.io/robotflow-labs/anima-serve:jazzy)
  = anima-base + anima_serve Python package
  = FastAPI health/ready/info + HF weight downloader
  = AnimaNode base class + GPU detection + MCAP hooks

Layer 3: anima-<module> (per module, built from Dockerfile.serve)
  = anima-serve + module code + module msgs + module node
```

## Key Package: anima_serve
- **Repo**: github.com/RobotFlow-Labs/anima-serve
- **Server location**: /mnt/forge-data/docker-base/anima-serve/
- **Package**: `src/anima_serve/` — config, gpu, health, lifecycle, weights, node, server, cli
- **ROS2 optional**: Works in API-only mode if rclpy not available

## AnimaNode — Base Class for All Modules
```python
from anima_serve.node import AnimaNode

class MyModuleNode(AnimaNode):
    def setup_inference(self):
        """Load model weights, configure backend."""
        weights = self.weight_manager.download_weights()
        self.model = load_model(weights)

    def process(self, input_data):
        """Run inference on input, return output."""
        return self.model(input_data)

    def get_status(self) -> dict:
        """Module-specific status fields."""
        return {"model_loaded": self.model is not None}
```

## Standard Endpoints (FastAPI)
- `GET /health` — `{status, module, uptime_s, gpu_vram_mb}`
- `GET /ready` — `{ready, module, version, weights_loaded}` (503 if not ready)
- `GET /info` — full module info
- `POST /predict` — run inference (module-specific)

## Environment Variables
All prefixed `ANIMA_`:
- `ANIMA_MODULE_NAME` — module identifier
- `ANIMA_MODULE_VERSION` — semver
- `ANIMA_HF_REPO` — HuggingFace repo (ilessio-aiflowlab/project_<name>)
- `ANIMA_DEVICE` — auto|cuda|cpu|mps
- `ANIMA_WEIGHT_FORMAT` — auto|trt|onnx|safetensors|pth
- `ANIMA_SERVE_PORT` — FastAPI port (default 8080)
- `ANIMA_WEIGHT_DIR` — weight cache directory (default /data/weights)
- `HF_TOKEN` — HuggingFace token for private repos

## Weight Format Priority
TRT > ONNX > safetensors > pth (by inference speed)

## Docker Compose Profiles
```bash
# Full stack (ROS2 + FastAPI + GPU)
docker compose -f docker-compose.serve.yml --profile serve up -d

# ROS2 only (for ANIMA compiler pipelines)
docker compose -f docker-compose.serve.yml --profile ros2 up -d

# API only (debug/dev)
docker compose -f docker-compose.serve.yml --profile api up -d

# Run tests
docker compose -f docker-compose.serve.yml --profile test run --rm test
```

## How to Add Docker Serving to a Module
Run `/anima-docker-serve` in the module directory. It will:
1. Read `anima_module.yaml` (create if missing)
2. Generate `Dockerfile.serve` from template
3. Generate `docker-compose.serve.yml`
4. Generate `.env.serve`
5. Create `src/<module>/serve.py` skeleton

## DO NOT
- DO NOT put weights IN the Docker image — they download at runtime from HF
- DO NOT use FastDDS in Docker — always CycloneDDS
- DO NOT hardcode ports — use `.env.serve` and `ANIMA_SERVE_PORT`
- DO NOT skip the health endpoint — every module publishes health at 1 Hz
- DO NOT forget `network_mode: host` — required for ROS2 DDS discovery
