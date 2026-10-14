"""Quick sanity check of a timm model (M3-M5) before the long training runs.

Builds the model in its final gradual-unfreezing state (all stages unfrozen,
stem frozen), runs a few AMP training steps on random data at the common
batch size and reports:
  - total / trainable parameters (frozen baseline and fully unfrozen),
  - peak GPU memory (must fit into the 8 GB GPU at BATCH_SIZE=128),
  - that every trainable parameter actually received a gradient,
  - that the hard-coded NORM_MEAN/NORM_STD match the pretrained weights,
  - the activation shape of the GradCAM target layer (after reshape_transform).

Usage:
    python -m src.smoke_test m3
    python -m src.smoke_test m3 --grad-checkpointing
"""
import argparse
import importlib
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import config
from src.models.timm_backbone import pretrained_normalization

MODEL_MODULES = {
    "m3": "src.models.M3_CNN_ConvNeXt_Tiny",
    "m4": "src.models.M4_ViT_Swin_Tiny",
    "m5": "src.models.M5_Hybrid_CoAtNet_0",
}


def count(model: nn.Module) -> tuple[int, int]:
    return (sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=sorted(MODEL_MODULES))
    parser.add_argument("--grad-checkpointing", action="store_true")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()

    module = importlib.import_module(MODEL_MODULES[args.model])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(config.CLASSES) or 15
    print(f"model={module.TIMM_NAME} device={device} batch={args.batch_size} "
          f"grad_checkpointing={args.grad_checkpointing}")

    model = module.build_model(num_classes, freeze_backbone=True, grad_checkpointing=args.grad_checkpointing)
    total, trainable = count(model)
    print(f"frozen baseline: total={total:,} trainable={trainable:,}")

    mean, std = pretrained_normalization(model)
    ok = all(abs(a - b) < 1e-3 for a, b in zip(mean + std, module.NORM_MEAN + module.NORM_STD))
    print(f"pretrained normalization mean={mean} std={std} -> {'OK' if ok else 'MISMATCH with NORM_MEAN/NORM_STD!'}")

    for stage in module.UNFREEZE_STAGES:
        module.unfreeze_stage(model, stage)
    total, trainable = count(model)
    frozen_names = sorted({n.split(".")[0] for n, p in model.named_parameters() if not p.requires_grad})
    print(f"fully unfrozen: total={total:,} trainable={trainable:,} still frozen: {frozen_names}")

    model = model.to(device).train()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=config.LR)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    criterion = nn.CrossEntropyLoss()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    images = torch.randn(args.batch_size, 3, config.IMAGE_SIZE, config.IMAGE_SIZE, device=device)
    labels = torch.randint(0, num_classes, (args.batch_size,), device=device)
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            loss = criterion(model(images), labels)
        scaler.scale(loss).backward()
        missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
        scaler.step(optimizer)
        scaler.update()
        print(f"step {step}: loss={loss.item():.4f} params without grad={len(missing)}")
        if missing:
            print(f"  e.g. {missing[:5]}")

    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 2**30
        total_mem = torch.cuda.get_device_properties(device).total_memory / 2**30
        print(f"peak GPU memory: {peak:.2f} GiB of {total_mem:.2f} GiB")

    activations = {}
    handle = module.gradcam_target(model).register_forward_hook(
        lambda _m, _i, out: activations.__setitem__("out", out))
    model.eval()
    with torch.no_grad():
        model(images[:2])
    handle.remove()
    out = activations["out"]
    if module.reshape_transform is not None:
        out = module.reshape_transform(out)
    print(f"GradCAM target activation (NCHW expected): {tuple(out.shape)}")


if __name__ == "__main__":
    main()
