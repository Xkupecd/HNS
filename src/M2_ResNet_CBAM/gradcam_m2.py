"""Generates a GradCAM grid (correct + misclassified test samples) for M2's
best run, saved to Results/M2_CNN_CBAM/M2_cnn_cbam/gradcam_examples.png.

Targets the last CBAM block (cbam4), i.e. the attention-refined layer4
features that feed the classifier - the M2 counterpart of M1's layer4.

Usage:
    python -m src.M2_ResNet_CBAM.gradcam_m2
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
from src.models.M2_CNN_ResNet50_CBAM import build_model

MODEL_NAME = "M2_cnn_cbam"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = config.M2_RESULTS_DIR / MODEL_NAME
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
        model, target_layer=model.cbam4, loaders=loaders, class_names=class_names,
        device=device, out_path=out_path,
    )
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
