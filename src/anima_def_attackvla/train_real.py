"""Real-image training for DefenseNet.

Trains on actual COCO val2017 images with adversarial patches applied,
using a proper train/val split for early stopping on validation loss.
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
from anima_def_attackvla.data import get_dataloaders


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
        "weight_decay": 0.01,
        "warmup_ratio": 0.05,
        "min_lr": 1e-7,
        "max_grad_norm": 1.0,
        "precision": "fp16",
        "seed": 42,
        "keep_top_k": 2,
        "early_stopping_patience": 20,
        "early_stopping_delta": 0.001,
        "attack_ratio": 0.5,
        "sigma": 0.05,
        "img_size": 224,
        "tv_loss_weight": 0.5,
        "num_workers": 4,
        "image_dir": "/mnt/forge-data/datasets/lerobot--libero/extracted_frames/observation.images.image",
        "max_frames": 0,
        "task_suite": "",
    }
    merged = {**defaults, **raw.get("training", {})}
    return merged


@torch.no_grad()
def validate(model, val_loader, cls_criterion, tv_criterion, tv_weight, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        out = model(images)
        cls_loss = cls_criterion(out.is_adversarial, labels)
        tv_loss = tv_criterion(out.tv_anomaly_score, labels)
        loss = cls_loss + tv_weight * tv_loss
        total_loss += loss.item() * labels.shape[0]
        preds = (torch.sigmoid(out.is_adversarial) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.shape[0]
    model.train()
    return total_loss / max(1, total), correct / max(1, total)


def train(config_path: str, gpu_id: int = 0, resume: str | None = None, max_steps: int = 0):
    cfg = load_training_config(config_path)
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg["seed"])

    print(f"[CONFIG] {config_path}")
    print(f"[GPU] {device}")
    print(f"[DATA] {cfg['image_dir']}")

    task_suite = cfg.get("task_suite", "") or None
    max_frames = int(cfg.get("max_frames", 0))
    train_loader, val_loader = get_dataloaders(
        frames_dir=cfg["image_dir"],
        img_size=cfg["img_size"],
        batch_size=cfg["batch_size"],
        attack_ratio=cfg["attack_ratio"],
        num_workers=cfg["num_workers"],
        task_suite=task_suite,
        max_frames=max_frames,
    )
    print(f"[DATA] train={len(train_loader.dataset)} val={len(val_loader.dataset)} images")
    print(f"[TRAIN] epochs={cfg['epochs']}, lr={cfg['learning_rate']}, bs={cfg['batch_size']}")

    model = DefenseNet(img_size=cfg["img_size"], sigma=cfg["sigma"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] DefenseNet — {n_params / 1e6:.2f}M parameters")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )

    steps_per_epoch = len(train_loader)
    total_steps = cfg["epochs"] * steps_per_epoch
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

    scaler = torch.amp.GradScaler("cuda") if cfg["precision"] == "fp16" and "cuda" in device else None
    use_amp = scaler is not None

    log_dir = Path("/mnt/artifacts-datai/logs/DEF-attackvla")
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = log_dir / "metrics_real.jsonl"

    ckpt_every = max(1, steps_per_epoch // 2)
    print(f"[CKPT] save every {ckpt_every} steps, keep best {cfg['keep_top_k']}")
    print(f"[LOG] {metrics_file}")
    print("[TRAIN] Starting real-image training...")

    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        t0 = time.time()

        for batch_idx, (images, labels) in enumerate(train_loader):
            if max_steps > 0 and global_step >= max_steps:
                print(f"[STOP] Reached max_steps={max_steps}")
                _save(model, optimizer, scheduler, epoch, global_step, epoch_loss / max(1, batch_idx), ckpt_mgr)
                return

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast("cuda"):
                    out = model(images)
                    cls_loss = cls_criterion(out.is_adversarial, labels)
                    tv_loss = tv_criterion(out.tv_anomaly_score, labels)
                    loss = cls_loss + tv_weight * tv_loss

                if torch.isnan(loss):
                    print("[FATAL] Loss is NaN — stopping")
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
                    print("[FATAL] Loss is NaN — stopping")
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

            if global_step % ckpt_every == 0:
                avg = epoch_loss / (batch_idx + 1)
                _save(model, optimizer, scheduler, epoch, global_step, avg, ckpt_mgr)

        train_loss = epoch_loss / max(1, len(train_loader))
        train_acc = epoch_correct / max(1, epoch_total)
        val_loss, val_acc = validate(model, val_loader, cls_criterion, tv_criterion, tv_weight, device)
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"[EPOCH {epoch + 1}/{cfg['epochs']}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"lr={lr:.2e} time={elapsed:.1f}s"
        )

        with open(metrics_file, "a") as f:
            f.write(json.dumps({
                "epoch": epoch + 1, "step": global_step,
                "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc, "lr": lr,
            }) + "\n")

        _save(model, optimizer, scheduler, epoch + 1, global_step, val_loss, ckpt_mgr)

        if early_stop.step(val_loss):
            print(f"[EARLY STOP] No improvement for {cfg['early_stopping_patience']} epochs")
            break

    print("[DONE] Training complete")
    print(f"[BEST] {ckpt_dir / 'best.pth'}")


def _save(model, optimizer, scheduler, epoch, step, metric, ckpt_mgr):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "step": step,
        "metric": metric,
    }
    path = ckpt_mgr.save(state, metric, step)
    print(f"  [CKPT] step={step} val_loss={metric:.4f} → {path.name}")


def main():
    parser = argparse.ArgumentParser(description="Train DefenseNet on real images")
    parser.add_argument("--config", required=True, help="TOML config path")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from")
    parser.add_argument("--max-steps", type=int, default=0, help="Stop after N steps (0=unlimited)")
    args = parser.parse_args()
    train(args.config, args.gpu, args.resume, args.max_steps)


if __name__ == "__main__":
    main()
