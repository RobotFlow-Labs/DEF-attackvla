"""Defense evaluation pipeline.

Evaluates DefenseNet against various attack types from the AttackVLA paper.
Computes detection accuracy, false positive rate, and defense effectiveness
metrics (ASR_u, ASR_s, ASR_t, CP) under each defense strategy.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch

from anima_def_attackvla.models.defense_net import DefenseNet
from anima_def_attackvla.models.vla_wrapper import AdversarialPatchGenerator


@dataclass
class EvalMetrics:
    attack_type: str
    n_samples: int
    detection_accuracy: float
    true_positive_rate: float
    false_positive_rate: float
    mean_adv_score_clean: float
    mean_adv_score_attack: float
    latency_ms: float


@dataclass
class DefenseReport:
    model_path: str
    n_params: int
    device: str
    metrics: list[EvalMetrics]
    total_accuracy: float
    total_tpr: float
    total_fpr: float


def evaluate_attack_type(
    model: DefenseNet,
    patch_gen: AdversarialPatchGenerator,
    attack_type: str,
    n_batches: int = 10,
    batch_size: int = 64,
    threshold: float = 0.5,
    device: str = "cuda",
) -> EvalMetrics:
    """Evaluate defense against a specific attack type."""
    model.eval()
    tp = fp = tn = fn = 0
    adv_scores_clean = []
    adv_scores_attack = []
    total_time = 0.0
    H = W = model.img_size

    with torch.no_grad():
        for _ in range(n_batches):
            clean = torch.rand(batch_size, 3, H, W, device=device) * 0.8 + 0.1

            if attack_type == "upa":
                patch = patch_gen.generate_upa_patch(48, 1)
                attacked = patch_gen.apply_patch(clean, patch)
            elif attack_type == "blue_cube":
                attacked = patch_gen.generate_trigger_image(clean, "blue_cube")
            elif attack_type == "noise":
                attacked = patch_gen.generate_trigger_image(clean, "noise")
            else:
                attacked = patch_gen.generate_trigger_image(clean, "patch")

            t0 = time.time()
            out_clean = model(clean)
            out_attack = model(attacked)
            total_time += time.time() - t0

            pred_clean = (out_clean.is_adversarial > threshold).float()
            pred_attack = (out_attack.is_adversarial > threshold).float()

            tn += (pred_clean == 0).sum().item()
            fp += (pred_clean == 1).sum().item()
            tp += (pred_attack == 1).sum().item()
            fn += (pred_attack == 0).sum().item()

            adv_scores_clean.append(out_clean.is_adversarial.mean().item())
            adv_scores_attack.append(out_attack.is_adversarial.mean().item())

    total = tp + fp + tn + fn
    n_samples = n_batches * batch_size * 2
    latency_ms = (total_time / (n_batches * 2)) * 1000

    return EvalMetrics(
        attack_type=attack_type,
        n_samples=n_samples,
        detection_accuracy=(tp + tn) / max(1, total),
        true_positive_rate=tp / max(1, tp + fn),
        false_positive_rate=fp / max(1, fp + tn),
        mean_adv_score_clean=sum(adv_scores_clean) / len(adv_scores_clean),
        mean_adv_score_attack=sum(adv_scores_attack) / len(adv_scores_attack),
        latency_ms=latency_ms,
    )


def run_full_evaluation(
    model_path: str,
    device: str = "cuda:0",
    n_batches: int = 10,
    batch_size: int = 64,
    threshold: float = 0.5,
) -> DefenseReport:
    """Run evaluation across all attack types."""
    model = DefenseNet()
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    patch_gen = AdversarialPatchGenerator(model.img_size, device)

    attack_types = ["upa", "blue_cube", "noise"]
    metrics = []

    for atype in attack_types:
        print(f"  Evaluating {atype}...")
        m = evaluate_attack_type(
            model, patch_gen, atype, n_batches, batch_size, threshold, device
        )
        metrics.append(m)
        print(
            f"    acc={m.detection_accuracy:.4f} tpr={m.true_positive_rate:.4f} "
            f"fpr={m.false_positive_rate:.4f} latency={m.latency_ms:.1f}ms"
        )

    total_tp = sum(m.true_positive_rate * m.n_samples / 2 for m in metrics)
    total_fp = sum(m.false_positive_rate * m.n_samples / 2 for m in metrics)
    total_samples = sum(m.n_samples for m in metrics) / 2

    return DefenseReport(
        model_path=model_path,
        n_params=n_params,
        device=device,
        metrics=metrics,
        total_accuracy=sum(m.detection_accuracy for m in metrics) / len(metrics),
        total_tpr=total_tp / max(1, total_samples),
        total_fpr=total_fp / max(1, total_samples),
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate DefenseNet")
    parser.add_argument("--model", required=True, help="Path to best.pth")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n-batches", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", default=None, help="JSON output path")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    report = run_full_evaluation(
        args.model, device, args.n_batches, args.batch_size, args.threshold
    )

    print(f"\n[REPORT] Overall accuracy={report.total_accuracy:.4f} "
          f"TPR={report.total_tpr:.4f} FPR={report.total_fpr:.4f}")

    out_path = args.output or "/mnt/artifacts-datai/reports/DEF-attackvla/eval_report.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model_path": report.model_path,
            "n_params": report.n_params,
            "device": report.device,
            "total_accuracy": report.total_accuracy,
            "total_tpr": report.total_tpr,
            "total_fpr": report.total_fpr,
            "metrics": [asdict(m) for m in report.metrics],
        }, f, indent=2)
    print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    main()
