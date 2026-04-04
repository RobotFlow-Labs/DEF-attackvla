"""FastAPI serving entrypoint for DEF-attackvla."""

from __future__ import annotations

from pathlib import Path
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

from anima_def_attackvla.config import load_config
from anima_def_attackvla.pipelines.inference import DefenseAwareInferencePipeline, InferenceRequest


class PredictRequest(BaseModel):
    instruction: str = Field(min_length=1)


app = FastAPI(title="DEF-attackvla", version="0.1.0")
_cfg_path = Path("configs/default.toml")
_pipeline = DefenseAwareInferencePipeline(load_config(_cfg_path))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, str]:
    return {"status": "ready"}


@app.post("/predict")
def predict(payload: PredictRequest) -> dict[str, object]:
    image = np.zeros((224, 224, 3), dtype=np.float32)
    response = _pipeline.predict(InferenceRequest(payload.instruction, image))
    return {
        "allowed": response.allowed,
        "reason": response.reason,
        "action_plan": response.action_plan,
        "sanitized_instruction": response.sanitized_instruction,
    }
