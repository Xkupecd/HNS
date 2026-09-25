"""Central configuration for the project (paths, split ratios, training defaults)."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"          # raw class-per-folder images (git-ignored)
SPLIT_DIR = PROJECT_ROOT / "data_split"   # train/val/test symlink or csv split
RESULTS_DIR = PROJECT_ROOT / "Results"
# One folder per model type, holding all of its configurations (baseline + ablations).
M1_RESULTS_DIR = RESULTS_DIR / "M1_CNN"
M2_RESULTS_DIR = RESULTS_DIR / "M2_CNN_CBAM"
COMPARISON_DIR = RESULTS_DIR / "M1_vs_M2"

CLASSES = sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir()) if DATA_DIR.exists() else []

SEED = 42
IMAGE_SIZE = 224

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

BATCH_SIZE = 128
NUM_WORKERS = 16

# Common training hyperparameters, kept identical across models where possible.
EPOCHS = 30
LR = 3e-4
WEIGHT_DECAY = 1e-4
NUM_RUNS = 3  # each model trained 3x for statistical evaluation

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

MODEL_NAMES = ["M1_cnn", "M2_cnn_cbam", "M3_convnext", "M4_vit", "M5_coatnet"]
