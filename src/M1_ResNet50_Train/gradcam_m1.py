"""Generates a GradCAM grid (correct + misclassified test samples) for M1's
best run, saved to Results/M1_CNN/M1_cnn/gradcam_examples.png.

Requires train_m1.py to have already produced Results/M1_CNN/M1_cnn/summary.json.

Usage:
    python -m src.M1_ResNet50_Train.gradcam_m1
"""
import json
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve()
while _ROOT.name != "src":
    _ROOT = _ROOT.parent
sys.path.append(str(_ROOT.parent))
from src import config
from src.data_preprocessing.dataset import build_dataloaders
from src.gradcam import save_gradcam_grid
from src.models.M1_CNN_ResNet50 import build_model

MODEL_NAME = "M1_cnn"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = config.M1_RESULTS_DIR / MODEL_NAME
    summary = json.loads((model_dir / "summary.json").read_text())
    best_run_idx = summary["per_run_acc"].index(max(summary["per_run_acc"])) + 1

    class_names = config.CLASSES
    model = build_model(num_classes=len(class_names), freeze_backbone=True).to(device)
    model.load_state_dict(torch.load(
        model_dir / f"run{best_run_idx}" / "best.pt", map_location=device, weights_only=True,
    ))

    loaders = build_dataloaders(augment_train=False)
    out_path = model_dir / "gradcam_examples.png"
    save_gradcam_grid(
        model, target_layer=model.layer4[-1], loaders=loaders, class_names=class_names,
        device=device, out_path=out_path,
    )
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
