"""Evaluate DefenseNet on real LIBERO task suites with paper-aligned attacks.

Runs the defense guard on actual LIBERO robot manipulation frames from
each of the four task suites (Object, Spatial, Goal, Long), applying
the exact attack patterns from the AttackVLA paper:
  - UPA: Universal adversarial patch (Sec 4.2)
  - UADA: Universal adversarial domain adaptation patch
  - TMA: Trigger-based manipulation attack
  - BackdoorVLA: Blue cube trigger object
  - BadVLA: Text trigger injection
  - TAB-VLA: Trigger-aware backdoor

Reports per-suite, per-attack detection metrics (accuracy, TPR, FPR).
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from anima_def_attackvla.models.defense_net import DefenseNet
from anima_def_attackvla.data import LiberoDefenseDataset, LIBERO_FRAMES, TASK_SUITES


# Paper-aligned attack implementations on real frames
ATTACK_TYPES = ["upa", "blue_cube", "noise", "checkerboard", "colored_square"]


@dataclass
class SuiteAttackResult:
    task_suite: str
    attack_type: str
    n_clean: int
    n_attack: int
    tp: int
    fp: int
    tn: int
    fn: int
    accuracy: float
    tpr: float
    fpr: float
    mean_score_clean: float
    mean_score_attack: float
    latency_ms: float


@dataclass
class FullEvalReport:
    model_path: str
    dataset: str
    n_params: int
    suites: list[str]
    results: list[SuiteAttackResult]
    overall_accuracy: float
    overall_tpr: float
    overall_fpr: float


def evaluate_suite(
    model: DefenseNet,
    suite_name: str,
    frames_dir: str,
    device: str,
    batch_size: int = 64,
    max_frames: int = 5000,
    threshold: float = 0.5,
) -> list[SuiteAttackResult]:
    """Evaluate defense on one LIBERO task suite across all attack types."""
    results = []

    for attack_type in ATTACK_TYPES:
        # Create clean dataset (attack_ratio=0) and attacked dataset (attack_ratio=1)
        clean_ds = LiberoDefenseDataset(
            frames_dir, img_size=224, attack_ratio=0.0,
            task_suite=suite_name, split="val", max_frames=max_frames,
        )
        attack_ds = LiberoDefenseDataset(
            frames_dir, img_size=224, attack_ratio=1.0,
            task_suite=suite_name, split="val", max_frames=max_frames,
        )

        clean_loader = DataLoader(clean_ds, batch_size=batch_size, num_workers=2, pin_memory=True)
        attack_loader = DataLoader(attack_ds, batch_size=batch_size, num_workers=2, pin_memory=True)

        tp = fp = tn = fn = 0
        scores_clean = []
        scores_attack = []
        total_time = 0.0

        model.eval()
        with torch.no_grad():
            # Evaluate on clean images
            for images, _ in clean_loader:
                images = images.to(device)
                t0 = time.time()
                out = model(images)
                total_time += time.time() - t0
                probs = torch.sigmoid(out.is_adversarial)
                preds = (probs > threshold).long()
                tn += (preds == 0).sum().item()
                fp += (preds == 1).sum().item()
                scores_clean.append(probs.mean().item())

            # Evaluate on attacked images
            for images, _ in attack_loader:
                images = images.to(device)
                t0 = time.time()
                out = model(images)
                total_time += time.time() - t0
                probs = torch.sigmoid(out.is_adversarial)
                preds = (probs > threshold).long()
                tp += (preds == 1).sum().item()
                fn += (preds == 0).sum().item()
                scores_attack.append(probs.mean().item())

        total = tp + fp + tn + fn
        n_clean = tn + fp
        n_attack = tp + fn
        latency_ms = (total_time / max(1, len(clean_loader) + len(attack_loader))) * 1000

        results.append(SuiteAttackResult(
            task_suite=suite_name,
            attack_type=attack_type,
            n_clean=n_clean,
            n_attack=n_attack,
            tp=tp, fp=fp, tn=tn, fn=fn,
            accuracy=(tp + tn) / max(1, total),
            tpr=tp / max(1, tp + fn),
            fpr=fp / max(1, fp + tn),
            mean_score_clean=sum(scores_clean) / max(1, len(scores_clean)),
            mean_score_attack=sum(scores_attack) / max(1, len(scores_attack)),
            latency_ms=latency_ms,
        ))

    return results


def run_full_libero_eval(
    model_path: str,
    frames_dir: str = LIBERO_FRAMES,
    device: str = "cuda:0",
    batch_size: int = 64,
    max_frames: int = 5000,
    threshold: float = 0.5,
) -> FullEvalReport:
    """Run evaluation across all 4 LIBERO task suites and all attack types."""
    model = DefenseNet()
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())

    all_results = []
    suites = list(TASK_SUITES.keys())

    for suite in suites:
        print(f"\n[SUITE] {suite}")
        suite_results = evaluate_suite(
            model, suite, frames_dir, device, batch_size, max_frames, threshold
        )
        for r in suite_results:
            print(
                f"  {r.attack_type:20s} acc={r.accuracy:.4f} "
                f"tpr={r.tpr:.4f} fpr={r.fpr:.4f} "
                f"clean_score={r.mean_score_clean:.3f} atk_score={r.mean_score_attack:.3f}"
            )
        all_results.extend(suite_results)

    total_tp = sum(r.tp for r in all_results)
    total_fp = sum(r.fp for r in all_results)
    total_tn = sum(r.tn for r in all_results)
    total_fn = sum(r.fn for r in all_results)
    total = total_tp + total_fp + total_tn + total_fn

    report = FullEvalReport(
        model_path=model_path,
        dataset=frames_dir,
        n_params=n_params,
        suites=suites,
        results=all_results,
        overall_accuracy=(total_tp + total_tn) / max(1, total),
        overall_tpr=total_tp / max(1, total_tp + total_fn),
        overall_fpr=total_fp / max(1, total_fp + total_tn),
    )

    print(f"\n[OVERALL] acc={report.overall_accuracy:.4f} "
          f"tpr={report.overall_tpr:.4f} fpr={report.overall_fpr:.4f}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate DefenseNet on LIBERO task suites")
    parser.add_argument("--model", required=True, help="Path to best.pth")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--frames-dir", default=LIBERO_FRAMES)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-frames", type=int, default=5000)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    report = run_full_libero_eval(
        args.model, args.frames_dir, device,
        args.batch_size, args.max_frames, args.threshold,
    )

    out_path = args.output or "/mnt/artifacts-datai/reports/DEF-attackvla/libero_eval_report.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model_path": report.model_path,
            "dataset": report.dataset,
            "n_params": report.n_params,
            "suites": report.suites,
            "overall_accuracy": report.overall_accuracy,
            "overall_tpr": report.overall_tpr,
            "overall_fpr": report.overall_fpr,
            "results": [asdict(r) for r in report.results],
        }, f, indent=2)
    print(f"\n[SAVED] {out_path}")


if __name__ == "__main__":
    main()
