"""Central configuration for the project (paths, classes, split ratios, training defaults)."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# HMDB51 videos, one folder per class (git-ignored); HNS_VIDEO_DIR overrides the location
VIDEO_DIR = Path(os.environ.get("HNS_VIDEO_DIR", PROJECT_ROOT / "data" / "hmdb51" / "videos"))
FRAMES_DIR = PROJECT_ROOT / "data" / "frames"          # extracted JPG frames per video
FEATURES_DIR = PROJECT_ROOT / "data" / "features"      # cached frozen-CNN frame features
CAMERA_VIDEO_DIR = PROJECT_ROOT / "data" / "camera"    # own webcam recordings (record_clips.py)
CAMERA_FRAMES_DIR = PROJECT_ROOT / "data" / "camera_frames"
SPLIT_DIR = PROJECT_ROOT / "data_split"                # train/val/test csv manifest
RESULTS_DIR = PROJECT_ROOT / "Results"
# first version: ConvNeXt-T (IN-1k) with the last stage fine-tuned -> overfitted, kept for the docs
ABLATION_V1_DIR = RESULTS_DIR / "ablation_v1_convnext_tiny_finetuned"
M1_RESULTS_DIR = RESULTS_DIR / "M1_CNN_SVM"
M2_RESULTS_DIR = RESULTS_DIR / "M2_CNN_LSTM"
M3_RESULTS_DIR = RESULTS_DIR / "M3_CNN_Transformer"
COMPARISON_DIR = RESULTS_DIR / "M1_to_M3"
CAMERA_RESULTS_DIR = RESULTS_DIR / "camera_finetune"

# Actions that can be clearly demonstrated in front of a webcam.
CLASSES = ["clap", "drink", "eat", "wave"]

SEED = 42
IMAGE_SIZE = 224
FRAMES_PER_VIDEO = 32   # stored per video, uniformly over the whole clip
CLIP_LEN = 16           # frames fed to the model (sampled from the stored 32)

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

BATCH_SIZE = 32         # videos per batch when training on cached features
EXTRACT_BATCH = 4       # videos per batch through the CNN -> 4 x 16 = 64 frames
NUM_WORKERS = int(os.environ.get("HNS_NUM_WORKERS", 12))

# Frozen frame encoder shared by all three methods. ConvNeXt-B pretrained on
# ImageNet-22k gave the best validation accuracy of the probed backbones
# (see README, "Výber backbonu").
BACKBONE = "convnext_base.fb_in22k"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
N_AUG_CLIPS = 10        # augmented clips cached per training video

# Common training hyperparameters for the temporal models (M2, M3), picked by
# mean validation accuracy over 5 runs (lr 1e-4..1e-3, wd 0.01/0.1, input LayerNorm).
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-1
NUM_RUNS = 5            # each model trained 5x for statistical evaluation (assignment)

# Webcam demo: the model sees CLIP_LEN frames spread over the last WINDOW_SEC
# seconds, similar to the 2-4 s duration of the HMDB51 clips.
WINDOW_SEC = 2.5
CONF_THRESHOLD = 0.5
