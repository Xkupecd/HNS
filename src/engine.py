"""Generic train/eval loop shared by the temporal models (M2, M3).

Kept model-agnostic so both models are trained with identical parameters;
only the temporal part differs. Inputs are either cached frame features
[B, T, D] (training) or raw clips [B, T, 3, H, W] (full model).
"""
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None) -> tuple[float, float]:
    model.train()
    use_amp = scaler is not None and device.type == "cuda"
    running_loss, correct, total = 0.0, 0, 0
    for clips, labels in loader:
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            outputs = model(clips)
            loss = criterion(outputs, labels)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * clips.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += clips.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    use_amp = device.type == "cuda"
    running_loss, correct, total = 0.0, 0, 0
    for clips, labels in loader:
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            outputs = model(clips)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * clips.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += clips.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def predict(model, loader, device) -> tuple[list[int], list[int]]:
    model.eval()
    use_amp = device.type == "cuda"
    y_true, y_pred = [], []
    for clips, labels in loader:
        clips = clips.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            outputs = model(clips)
        y_pred.extend(outputs.argmax(1).cpu().tolist())
        y_true.extend(labels.tolist())
    return y_true, y_pred


def fit(
    model,
    loaders: dict,
    epochs: int,
    lr: float,
    weight_decay: float,
    device,
    run_dir: Path,
    checkpoint_name: str = "best.pt",
) -> list[dict]:
    """Trains `model` for `epochs` (AdamW, cosine LR decay), logging per-epoch
    metrics and saving the best checkpoint into `run_dir`. Returns the history.

    The best epoch is chosen by val_acc, ties broken by lower val_loss: the
    val set has only 75 videos, so several epochs often share the same
    accuracy."""
    run_dir.mkdir(parents=True, exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    history: list[dict] = []
    best = (-1.0, float("inf"))
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, loaders["train"], optimizer, criterion, device, scaler=scaler)
        val_loss, val_acc = evaluate(model, loaders["val"], criterion, device)
        scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
        })
        print(f"[{run_dir.name}] epoch {epoch}/{epochs} "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best[0] or (val_acc == best[0] and val_loss < best[1]):
            best = (val_acc, val_loss)
            torch.save(model.state_dict(), run_dir / checkpoint_name)

    write_history(history, run_dir / "history.csv")
    return history


def write_history(history: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)
