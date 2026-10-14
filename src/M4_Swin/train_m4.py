"""Train and evaluate M4 (Swin-Tiny vision transformer, transfer learning,
frozen backbone) with the same procedure and hyperparameters as M1-M3:
3 training runs, training-curve plots, test-set ACC/Precision/Recall/F1 +
confusion matrix and run statistics (mean/std over the 3 runs).

This is the baseline M4 configuration (directly comparable to M1_cnn,
M2_cnn_cbam and M3_convnext). See train_m4_gradual_unfreeze.py for the
fine-tuned variant.

Usage:
    python -m src.M4_Swin.train_m4
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve()
while _ROOT.name != "src":
    _ROOT = _ROOT.parent
sys.path.append(str(_ROOT.parent))
from src import config
from src.experiment import run_experiment
from src.models.M4_ViT_Swin_Tiny import NORM_MEAN, NORM_STD, build_model


def main() -> None:
    run_experiment(
        "M4_swin",
        results_root=config.M4_RESULTS_DIR,
        build_model_fn=lambda num_classes: build_model(num_classes, freeze_backbone=True),
        augment_train=True,
        mean=NORM_MEAN, std=NORM_STD,
    )


if __name__ == "__main__":
    main()
