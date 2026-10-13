"""Generates a GradCAM grid (correct + misclassified test samples) for the
best run of each trained M3 configuration, saved to
Results/M3_ConvNeXt/<config>/gradcam_examples.png.

Requires train_m3.py (and optionally train_m3_gradual_unfreeze.py) to have
already produced <config>/summary.json.

Usage:
    python -m src.M3_ConvNeXt.gradcam_m3
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
from src.models.M3_CNN_ConvNeXt_Tiny import (
    NORM_MEAN, NORM_STD, build_model, gradcam_target, reshape_transform,
)

MODEL_NAMES = ["M3_convnext", "M3_convnext_gradual_unfreeze"]


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = config.CLASSES
    loaders = build_dataloaders(augment_train=False, mean=NORM_MEAN, std=NORM_STD)

    for model_name in MODEL_NAMES:
        model_dir = config.M3_RESULTS_DIR / model_name
        if not (model_dir / "summary.json").exists():
            print(f"Skipping {model_name}: not trained yet.")
            continue
        summary = json.loads((model_dir / "summary.json").read_text())
        best_run_idx = summary["per_run_acc"].index(max(summary["per_run_acc"])) + 1

        model = build_model(num_classes=len(class_names), freeze_backbone=True, pretrained=False).to(device)
        model.load_state_dict(torch.load(
            model_dir / f"run{best_run_idx}" / "best.pt", map_location=device, weights_only=True,
        ))

        out_path = model_dir / "gradcam_examples.png"
        save_gradcam_grid(
            model, target_layer=gradcam_target(model), loaders=loaders, class_names=class_names,
            device=device, out_path=out_path,
            reshape_transform=reshape_transform, mean=NORM_MEAN, std=NORM_STD,
        )
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
