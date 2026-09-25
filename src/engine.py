"""Generic train/eval loop shared by all models (M1-M5).

Kept model-agnostic so the same training/evaluation/statistics code is reused
across architectures, per the assignment's requirement to use identical
training parameters wherever possible.
"""
import csv
import json
from pathlib import Path

import torch
from torch import nn


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None) -> tuple[float, float]:
    model.train()
    use_amp = scaler is not None and device.type == "cuda"
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    use_amp = device.type == "cuda"
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def predict(model, loader, device) -> tuple[list[int], list[int]]:
    model.eval()
    use_amp = device.type == "cuda"
    y_true, y_pred = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            outputs = model(images)
        y_pred.extend(outputs.argmax(1).cpu().tolist())
        y_true.extend(labels.tolist())
    return y_true, y_pred


def build_unfreeze_schedule(stages: list[str], epochs: int) -> dict[int, list[str]]:
    """Spreads `stages` evenly over `epochs`, leaving an initial head-only
    warmup phase and ending with everything unfrozen for the last stretch.

    E.g. 4 stages over 30 epochs unfreezes one stage every 6 epochs, starting
    at epoch 7 (layer4), so all stages are trainable well before the end.
    """
    step = max(1, epochs // (len(stages) + 1))
    schedule: dict[int, list[str]] = {}
    for i, stage in enumerate(stages):
        epoch = min(step * (i + 1) + 1, epochs)
        schedule.setdefault(epoch, []).append(stage)
    return schedule


def fit(
    model,
    loaders: dict,
    epochs: int,
    lr: float,
    weight_decay: float,
    device,
    run_dir: Path,
    on_epoch_end=None,
    unfreeze_schedule: dict[int, list[str]] | None = None,
    unfreeze_fn=None,
) -> list[dict]:
    """Trains `model` for `epochs`, logging per-epoch metrics and saving the
    best (by val_acc) checkpoint into `run_dir`. Returns the full history.

    `unfreeze_schedule`/`unfreeze_fn` implement gradual unfreezing: at the
    given epochs, `unfreeze_fn(model, stage)` is called for each listed
    stage, and the newly-trainable parameters are added to the optimizer as
    a new param group (params added after construction are otherwise never
    optimized, even once `requires_grad` is flipped back on).
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    history: list[dict] = []
    best_val_acc = -1.0
    for epoch in range(1, epochs + 1):
        if unfreeze_schedule and epoch in unfreeze_schedule:
            already_trainable = {id(p) for p in model.parameters() if p.requires_grad}
            for stage in unfreeze_schedule[epoch]:
                unfreeze_fn(model, stage)
            new_params = [p for p in model.parameters() if p.requires_grad and id(p) not in already_trainable]
            if new_params:
                optimizer.add_param_group({"params": new_params, "lr": lr, "weight_decay": weight_decay})
                print(f"[{run_dir.name}] epoch {epoch}: unfroze {unfreeze_schedule[epoch]} "
                      f"(+{sum(p.numel() for p in new_params):,} params)")

        train_loss, train_acc = train_one_epoch(model, loaders["train"], optimizer, criterion, device, scaler=scaler)
        val_loss, val_acc = evaluate(model, loaders["val"], criterion, device)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
        })
        print(f"[{run_dir.name}] epoch {epoch}/{epochs} "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), run_dir / "best.pt")

        if on_epoch_end is not None:
            on_epoch_end(model, epoch)

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
