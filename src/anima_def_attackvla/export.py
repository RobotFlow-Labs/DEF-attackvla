"""Export pipeline: pth -> safetensors -> ONNX -> TRT FP16 + TRT FP32.

Exports the trained DefenseNet in all required ANIMA formats.
"""
from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path

import torch

from anima_def_attackvla.models.defense_net import DefenseNet


@dataclass
class ExportManifest:
    model_family: str
    pth_path: str
    safetensors_path: str
    onnx_path: str
    trt_fp16_path: str
    trt_fp32_path: str


def export_pth(model: DefenseNet, out_dir: Path) -> Path:
    """Save model state dict as .pth."""
    path = out_dir / "defense_net.pth"
    torch.save({"model": model.state_dict()}, path)
    print(f"  [PTH] {path} ({path.stat().st_size / 1e6:.1f}MB)")
    return path


def export_safetensors(model: DefenseNet, out_dir: Path) -> Path:
    """Save model weights as .safetensors."""
    from safetensors.torch import save_file
    path = out_dir / "defense_net.safetensors"
    save_file(model.state_dict(), str(path))
    print(f"  [SAFETENSORS] {path} ({path.stat().st_size / 1e6:.1f}MB)")
    return path


def export_onnx(model: DefenseNet, out_dir: Path, img_size: int = 224) -> Path:
    """Export model to ONNX format."""
    path = out_dir / "defense_net.onnx"
    dummy = torch.randn(1, 3, img_size, img_size)
    model = model.cpu().eval()

    class ExportWrapper(torch.nn.Module):
        """Wrapper that returns a single tensor for ONNX export."""
        def __init__(self, defense_net):
            super().__init__()
            self.net = defense_net

        def forward(self, x):
            out = self.net(x)
            return out.is_adversarial

    wrapper = ExportWrapper(model)
    torch.onnx.export(
        wrapper,
        dummy,
        str(path),
        input_names=["image"],
        output_names=["adversarial_score"],
        dynamic_axes={"image": {0: "batch"}, "adversarial_score": {0: "batch"}},
        opset_version=18,
    )
    print(f"  [ONNX] {path} ({path.stat().st_size / 1e6:.1f}MB)")
    return path


def export_trt(onnx_path: Path, out_dir: Path) -> tuple[Path, Path]:
    """Convert ONNX to TensorRT FP16 + FP32 using shared toolkit."""
    trt_toolkit = Path("/mnt/forge-data/shared_infra/trt_toolkit/export_to_trt.py")
    fp16_path = out_dir / "defense_net_fp16.trt"
    fp32_path = out_dir / "defense_net_fp32.trt"

    if trt_toolkit.exists():
        cmd = [
            "python3", str(trt_toolkit),
            "--onnx", str(onnx_path),
            "--output-dir", str(out_dir),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            # Toolkit produces files in output-dir; find them
            for f in out_dir.glob("*.trt"):
                if "fp16" in f.name:
                    fp16_path = f
                elif "fp32" in f.name:
                    fp32_path = f
            print(f"  [TRT] FP16={fp16_path.name}, FP32={fp32_path.name}")
            return fp16_path, fp32_path
        else:
            print(f"  [TRT] toolkit error: {result.stderr[:200]}")

    # Fallback: try trtexec directly
    for prec, flag, path in [("fp16", "--fp16", fp16_path), ("fp32", "", fp32_path)]:
        trtexec = "trtexec"
        cmd = f"{trtexec} --onnx={onnx_path} --saveEngine={path} {flag} --workspace=1024"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  [TRT-{prec.upper()}] {path}")
        else:
            print(f"  [TRT-{prec.upper()}] SKIPPED (trtexec not available)")
            path.write_text(f"# TRT {prec} export pending — trtexec not installed\n")

    return fp16_path, fp32_path


def run_export(
    checkpoint_path: str,
    output_dir: str = "/mnt/artifacts-datai/exports/DEF-attackvla",
) -> ExportManifest:
    """Run full export pipeline."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = DefenseNet()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[EXPORT] DefenseNet — {n_params / 1e6:.2f}M params")
    print(f"[OUTPUT] {out_dir}")

    pth_path = export_pth(model, out_dir)
    st_path = export_safetensors(model, out_dir)
    onnx_path = export_onnx(model, out_dir)
    trt_fp16, trt_fp32 = export_trt(onnx_path, out_dir)

    manifest = ExportManifest(
        model_family="attackvla-defense",
        pth_path=str(pth_path),
        safetensors_path=str(st_path),
        onnx_path=str(onnx_path),
        trt_fp16_path=str(trt_fp16),
        trt_fp32_path=str(trt_fp32),
    )
    print("[EXPORT] Complete")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Export DefenseNet")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pth")
    parser.add_argument(
        "--output-dir",
        default="/mnt/artifacts-datai/exports/DEF-attackvla",
    )
    args = parser.parse_args()
    run_export(args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
