"""FastAPI serving entrypoint for DEF-attackvla defense guard."""
from __future__ import annotations

import base64
import io
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field

from anima_def_attackvla.models.defense_net import DefenseNet

logger = logging.getLogger(__name__)

_MODEL: DefenseNet | None = None
_DEVICE: str = "cpu"
_START_TIME: float = time.time()


def _load_model() -> DefenseNet | None:
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
        ckpt = torch.load(weight_path, map_location=_DEVICE, weights_only=True)
        model.load_state_dict(ckpt["model"])
        logger.info("Loaded weights from %s", weight_path)
    else:
        logger.warning("No weights found at %s, using random init", weight_path)

    model = model.to(_DEVICE).eval()
    _MODEL = model
    return _MODEL


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


app = FastAPI(title="DEF-attackvla", version="0.2.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    instruction: str = Field(min_length=1)
    image_b64: str | None = Field(default=None, description="Base64-encoded RGB image (224x224)")
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    allowed: bool
    adversarial_score: float
    reason: str


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
    return {
        "ready": _MODEL is not None,
        "module": "DEF-attackvla",
        "version": "0.2.0",
        "weights_loaded": _MODEL is not None,
    }


@app.get("/info")
def info() -> dict:
    model = _MODEL
    n_params = sum(p.numel() for p in model.parameters()) if model else 0
    return {
        "module": "DEF-attackvla",
        "version": "0.2.0",
        "model_params": n_params,
        "device": _DEVICE,
        "available_vla_models": [
            "openvla-7b", "pi0fast-base", "pi05-base", "smolvla-base"
        ],
    }


@app.post("/predict")
def predict(payload: PredictRequest) -> PredictResponse:
    model = _MODEL
    if model is None:
        return PredictResponse(allowed=False, adversarial_score=1.0, reason="model_not_loaded")

    # Decode image from base64 if provided, else use placeholder
    if payload.image_b64:
        raw = base64.b64decode(payload.image_b64)
        arr = np.frombuffer(raw, dtype=np.float32).reshape(1, 3, 224, 224)
        image = torch.from_numpy(arr).to(_DEVICE)
    else:
        image = torch.zeros(1, 3, 224, 224, device=_DEVICE)

    with torch.no_grad():
        is_blocked, adv_score, _ = model.detect_and_sanitize(image, payload.threshold)

    visual_blocked = bool(is_blocked[0].item())
    score = float(adv_score[0].item())

    # Text-level check for known trigger tokens
    blocked_tokens = ("*magic*", "backdoor", "trigger")
    text_blocked = any(t in payload.instruction.lower() for t in blocked_tokens)

    final_blocked = visual_blocked or text_blocked
    reason = "text_trigger" if text_blocked else ("visual_anomaly" if visual_blocked else "clean")

    return PredictResponse(
        allowed=not final_blocked,
        adversarial_score=score,
        reason=reason,
    )
