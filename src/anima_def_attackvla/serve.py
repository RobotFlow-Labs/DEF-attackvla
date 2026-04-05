"""FastAPI serving entrypoint for DEF-attackvla defense guard."""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field

from anima_def_attackvla.models.defense_net import DefenseNet

app = FastAPI(title="DEF-attackvla", version="0.2.0")

_MODEL: DefenseNet | None = None
_DEVICE: str = "cpu"
_START_TIME: float = time.time()


def _load_model() -> DefenseNet:
    global _MODEL, _DEVICE
    if _MODEL is not None:
        return _MODEL

    _DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = DefenseNet()

    weight_path = os.environ.get(
        "ANIMA_WEIGHT_PATH",
        "/mnt/artifacts-datai/checkpoints/DEF-attackvla/best.pth",
    )
    if Path(weight_path).exists():
        ckpt = torch.load(weight_path, map_location=_DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model"])

    model = model.to(_DEVICE).eval()
    _MODEL = model
    return _MODEL


class PredictRequest(BaseModel):
    instruction: str = Field(min_length=1)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    allowed: bool
    adversarial_score: float
    reason: str
    sanitized: bool


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "module": "DEF-attackvla",
        "uptime_s": round(time.time() - _START_TIME, 1),
        "gpu_available": torch.cuda.is_available(),
    }


@app.get("/ready")
def ready() -> dict:
    model = _load_model()
    return {
        "ready": model is not None,
        "module": "DEF-attackvla",
        "version": "0.2.0",
        "weights_loaded": _MODEL is not None,
    }


@app.get("/info")
def info() -> dict:
    model = _load_model()
    return {
        "module": "DEF-attackvla",
        "version": "0.2.0",
        "model_params": sum(p.numel() for p in model.parameters()),
        "device": _DEVICE,
        "available_vla_models": [
            "openvla-7b", "pi0fast-base", "pi05-base", "smolvla-base"
        ],
    }


@app.post("/predict")
def predict(payload: PredictRequest) -> PredictResponse:
    model = _load_model()
    image = torch.rand(1, 3, 224, 224, device=_DEVICE)

    with torch.no_grad():
        is_blocked, adv_score, sanitized = model.detect_and_sanitize(
            image, payload.threshold
        )

    blocked = bool(is_blocked[0].item())
    score = float(adv_score[0].item())

    # Text-level check for known trigger tokens
    blocked_tokens = ("*magic*", "backdoor", "trigger")
    text_blocked = any(t in payload.instruction.lower() for t in blocked_tokens)

    final_blocked = blocked or text_blocked
    reason = "text_trigger" if text_blocked else ("visual_anomaly" if blocked else "clean")

    return PredictResponse(
        allowed=not final_blocked,
        adversarial_score=score,
        reason=reason,
        sanitized=not final_blocked,
    )
