"""Central configuration for the project (paths, split ratios, training defaults)."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# raw class-per-folder images (git-ignored); HNS_DATA_DIR overrides the location
DATA_DIR = Path(os.environ.get("HNS_DATA_DIR", PROJECT_ROOT / "data"))
SPLIT_DIR = PROJECT_ROOT / "data_split"   # train/val/test symlink or csv split
RESULTS_DIR = PROJECT_ROOT / "Results"
# One folder per model type, holding all of its configurations (baseline + ablations).
M1_RESULTS_DIR = RESULTS_DIR / "M1_CNN"
M2_RESULTS_DIR = RESULTS_DIR / "M2_CNN_CBAM"
M3_RESULTS_DIR = RESULTS_DIR / "M3_ConvNeXt"
M4_RESULTS_DIR = RESULTS_DIR / "M4_Swin"
M5_RESULTS_DIR = RESULTS_DIR / "M5_CoAtNet"
COMPARISON_DIR = RESULTS_DIR / "M1_vs_M2"
ALL_COMPARISON_DIR = RESULTS_DIR / "M1_to_M5"

CLASSES = sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir()) if DATA_DIR.exists() else []

SEED = 42
IMAGE_SIZE = 224

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

BATCH_SIZE = 128
# DataLoader workers only affect speed, not results. On Windows every worker is
# a spawned process that re-imports torch/timm, so fewer are needed to fit RAM.
NUM_WORKERS = int(os.environ.get("HNS_NUM_WORKERS", 16))

# Common training hyperparameters, kept identical across models where possible.
EPOCHS = 30
LR = 3e-4
WEIGHT_DECAY = 1e-4
NUM_RUNS = 3  # each model trained 3x for statistical evaluation

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

MODEL_NAMES = ["M1_cnn", "M2_cnn_cbam", "M3_convnext", "M4_vit", "M5_coatnet"]
