"""Training loop for DefenseNet — adversarial patch/trigger detection model.

Trains the defense network on synthetically generated adversarial examples
matching the AttackVLA paper patterns (UPA, TMA, BackdoorVLA triggers).
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import tomllib
import torch
import torch.nn as nn

from anima_def_attackvla.models.defense_net import DefenseNet
from anima_def_attackvla.models.vla_wrapper import AdversarialPatchGenerator


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr: float = 1e-7):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            scale = self.current_step / self.warmup_steps
        else:
            progress = (self.current_step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            scale = 0.5 * (1 + math.cos(math.pi * progress))
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = max(self.min_lr, base_lr * scale)

    def state_dict(self):
        return {"current_step": self.current_step}

    def load_state_dict(self, state):
        self.current_step = state["current_step"]


class CheckpointManager:
    def __init__(self, save_dir: Path, keep_top_k: int = 2, mode: str = "min"):
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_top_k = keep_top_k
        self.mode = mode
        self.history: list[tuple[float, Path]] = []

    def save(self, state: dict, metric_value: float, step: int) -> Path:
        path = self.save_dir / f"checkpoint_step{step:06d}.pth"
        torch.save(state, path)

        # Deduplicate: remove existing entry for same path
        self.history = [(v, p) for v, p in self.history if p != path]
        self.history.append((metric_value, path))
        self.history.sort(key=lambda x: x[0], reverse=(self.mode == "max"))

        while len(self.history) > self.keep_top_k:
            _, old_path = self.history.pop()
            old_path.unlink(missing_ok=True)
        best_val, best_path = self.history[0]
        shutil.copy2(best_path, self.save_dir / "best.pth")
        return path


class EarlyStopping:
    def __init__(self, patience: int = 20, min_delta: float = 0.001, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = float("inf") if mode == "min" else float("-inf")
        self.counter = 0

    def step(self, metric: float) -> bool:
        improved = (
            (metric < self.best - self.min_delta)
            if self.mode == "min"
            else (metric > self.best + self.min_delta)
        )
        if improved:
            self.best = metric
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def load_training_config(path: str) -> dict:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    defaults = {
        "batch_size": 64,
        "learning_rate": 0.001,
        "epochs": 100,
        "optimizer": "adamw",
        "weight_decay": 0.01,
        "warmup_ratio": 0.05,
        "min_lr": 1e-7,
        "max_grad_norm": 1.0,
        "precision": "fp16",
        "seed": 42,
        "checkpoint_every_steps": 200,
        "keep_top_k": 2,
        "early_stopping_patience": 20,
        "early_stopping_delta": 0.001,
        "attack_ratio": 0.5,
        "steps_per_epoch": 100,
        "sigma": 0.05,
        "img_size": 224,
        "tv_loss_weight": 0.5,
    }
    merged = {**defaults, **raw.get("training", {})}
    return merged


def train(config_path: str, gpu_id: int = 0, resume: str | None = None, max_steps: int = 0):
    cfg = load_training_config(config_path)
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg["seed"])

    print(f"[CONFIG] {config_path}")
    print(f"[GPU] {device}")
    print(f"[TRAIN] epochs={cfg['epochs']}, lr={cfg['learning_rate']}, bs={cfg['batch_size']}")

    model = DefenseNet(
        img_size=cfg["img_size"],
        sigma=cfg["sigma"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] DefenseNet — {n_params / 1e6:.2f}M parameters")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )

    total_steps = cfg["epochs"] * cfg["steps_per_epoch"]
    warmup_steps = int(total_steps * cfg["warmup_ratio"])
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps, cfg["min_lr"])

    ckpt_dir = Path("/mnt/artifacts-datai/checkpoints/DEF-attackvla")
    ckpt_mgr = CheckpointManager(ckpt_dir, keep_top_k=cfg["keep_top_k"], mode="min")
    early_stop = EarlyStopping(cfg["early_stopping_patience"], cfg["early_stopping_delta"])
    cls_criterion = nn.BCEWithLogitsLoss()
    tv_criterion = nn.MSELoss()
    tv_weight = float(cfg["tv_loss_weight"])

    start_epoch = 0
    global_step = 0

    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"]
        global_step = ckpt["step"]
        print(f"[RESUME] from epoch={start_epoch} step={global_step}")

    patch_gen = AdversarialPatchGenerator(cfg["img_size"], device)
    scaler = torch.amp.GradScaler("cuda") if cfg["precision"] == "fp16" and "cuda" in device else None
    use_amp = scaler is not None

    log_dir = Path("/mnt/artifacts-datai/logs/DEF-attackvla")
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = log_dir / "metrics.jsonl"

    print(f"[CKPT] save every {cfg['checkpoint_every_steps']} steps, keep best {cfg['keep_top_k']}")
    print(f"[LOG] {log_dir}")
    print("[TRAIN] Starting...")

    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        t0 = time.time()

        for step_in_epoch in range(cfg["steps_per_epoch"]):
            if max_steps > 0 and global_step >= max_steps:
                print(f"[STOP] Reached max_steps={max_steps}")
                _save_checkpoint(model, optimizer, scheduler, epoch, global_step, epoch_loss, ckpt_mgr)
                return

            images, labels = patch_gen.generate_training_batch(
                cfg["batch_size"], cfg["attack_ratio"]
            )

            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast("cuda"):
                    out = model(images)
                    cls_loss = cls_criterion(out.is_adversarial, labels)
                    tv_loss = tv_criterion(out.tv_anomaly_score, labels)
                    loss = cls_loss + tv_weight * tv_loss

                if torch.isnan(loss):
                    print("[FATAL] Loss is NaN — stopping training")
                    print("[FIX] Reduce lr by 10x, check data")
                    return

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(images)
                cls_loss = cls_criterion(out.is_adversarial, labels)
                tv_loss = tv_criterion(out.tv_anomaly_score, labels)
                loss = cls_loss + tv_weight * tv_loss

                if torch.isnan(loss):
                    print("[FATAL] Loss is NaN — stopping training")
                    print("[FIX] Reduce lr by 10x, check data")
                    return

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
                optimizer.step()

            scheduler.step()
            global_step += 1

            preds = (torch.sigmoid(out.is_adversarial) > 0.5).float()
            epoch_correct += (preds == labels).sum().item()
            epoch_total += labels.shape[0]
            epoch_loss += loss.item()

            if global_step % cfg["checkpoint_every_steps"] == 0:
                avg = epoch_loss / (step_in_epoch + 1)
                _save_checkpoint(model, optimizer, scheduler, epoch, global_step, avg, ckpt_mgr)

        avg_loss = epoch_loss / cfg["steps_per_epoch"]
        acc = epoch_correct / max(1, epoch_total)
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"[EPOCH {epoch + 1}/{cfg['epochs']}] "
            f"loss={avg_loss:.4f} acc={acc:.4f} lr={lr:.2e} time={elapsed:.1f}s"
        )

        with open(metrics_file, "a") as f:
            f.write(json.dumps({
                "epoch": epoch + 1, "step": global_step,
                "train_loss": avg_loss, "train_acc": acc, "lr": lr,
            }) + "\n")

        _save_checkpoint(model, optimizer, scheduler, epoch + 1, global_step, avg_loss, ckpt_mgr)

        if early_stop.step(avg_loss):
            print(f"[EARLY STOP] No improvement for {cfg['early_stopping_patience']} epochs")
            break

    print("[DONE] Training complete")
    print(f"[BEST] {ckpt_dir / 'best.pth'}")


def _save_checkpoint(model, optimizer, scheduler, epoch, step, metric, ckpt_mgr):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "step": step,
        "metric": metric,
    }
    path = ckpt_mgr.save(state, metric, step)
    print(f"  [CKPT] saved step={step} metric={metric:.4f} → {path.name}")


def main():
    parser = argparse.ArgumentParser(description="Train DefenseNet for VLA adversarial defense")
    parser.add_argument("--config", required=True, help="TOML config path")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from")
    parser.add_argument("--max-steps", type=int, default=0, help="Stop after N steps (0=unlimited)")
    args = parser.parse_args()
    train(args.config, args.gpu, args.resume, args.max_steps)


if __name__ == "__main__":
    main()
