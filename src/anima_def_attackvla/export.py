"""Export manifest helpers (scaffold)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExportManifest:
    model_family: str
    pth_path: str
    safetensors_path: str
    onnx_path: str
    trt_fp16_path: str
    trt_fp32_path: str


def default_manifest(root: str = "exports") -> ExportManifest:
    base = Path(root)
    return ExportManifest(
        model_family="attackvla-defense",
        pth_path=str(base / "model.pth"),
        safetensors_path=str(base / "model.safetensors"),
        onnx_path=str(base / "model.onnx"),
        trt_fp16_path=str(base / "model_fp16.engine"),
        trt_fp32_path=str(base / "model_fp32.engine"),
    )
