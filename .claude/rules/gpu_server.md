# GPU Server — Vast.ai srv7

## SSH Access
```bash
ssh datai_srv7_development
# Config: ~/.ssh/config → Host datai_srv7_development
# IP: 34.124.148.215
# User: datai
# Key: ~/.ssh/datai_srv
```

## Storage Layout
```
/mnt/forge-data/              (2.5TB NVMe — main workspace)
├── modules/                  (all ANIMA modules)
│   ├── project_*/            (60+ core modules)
│   ├── 01_healthcare/        (14 healthcare modules)
│   └── 02_defense/           (19 defense modules)
├── models/                   (all pretrained weights — HF repos, .pth files)
├── datasets/                 (all datasets — HF, archives, raw)
├── repos/                    (cloned reference repos)
├── checkpoints/              (training checkpoints)
└── GPU_STRATEGY.md           (master tracking doc for all GPU work)

/mnt/artifacts-datai/         (artifacts + exports)
├── models/project_*/         (final trained models per module)
└── exports/project_*/        (ONNX/TRT exports)

/mnt/titan-healthcare/        (overflow storage)
```

## GPUs
- NVIDIA L4 (23GB VRAM each)
- Use `/gpu-batch-finder` before training (target 60-70% VRAM)
- Always `CUDA_VISIBLE_DEVICES=X` to pin GPU

## Python
- Python 3.11 installed (3.12 available but breaks mmcv/cleanfid)
- Always: `uv venv .venv --python python3.11 && uv sync`
- Always: `unset PYTHONPATH` (was polluting venvs, removed from .bashrc)

## tmux Convention
- Module sessions: `tmux new -s MODULE_NAME -c /path/to/module`
- Download sessions: `tmux new -s DL_ASSET_NAME`
- Kill: `tmux kill-session -t NAME`
- List: `tmux list-sessions`

## Agent Launch (inside tmux)
- Do NOT use `ccc` — use separate `cc` to control token usage
- Each agent: `cd /path/to/module && tmux new -s NAME`, then start claude separately

## Git
- All modules push to `github.com/RobotFlow-Labs/project_<name>`
- Branches: `main` + `develop`
- Commit prefix: `[MODULE_NAME]`

## Training Convention
- Always use `nohup ... > train.log 2>&1 & disown` (survives Claude crash)
- Monitor: `tail -f train.log`
- Export pipeline: pth → safetensors → ONNX → TRT (fp16/fp32) → HF push
- HF org: `ilessio-aiflowlab`

## Active Downloads (check with `tmux list-sessions | grep DL_`)
- Downloads run in dedicated tmux sessions prefixed with `DL_`

## Key Files on Server
- `/mnt/forge-data/modules/GPU_STRATEGY.md` — master status of all modules
- Each module has: `CLAUDE.md`, `ASSETS.md`, `PIPELINE_MAP.md`, `NEXT_STEPS.md`

## Gated/Blocked Assets (need Mac download + scp)
- Google Drive links: download on Mac, `scp` to server
- Baidu Pan: download on Mac, `scp` to server
- HuggingFace gated: accept license at huggingface.co with `ilessio-aiflowlab` account
- IEEE DataPort: needs account + download on Mac

## Batch Workflow
- Run 5 modules at a time (tmux sessions)
- Each agent: read paper → build code → test → push
- When batch done: kill tmux sessions, update GPU_STRATEGY.md, start next batch
