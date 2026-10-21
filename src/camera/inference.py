"""Loads a trained model (M1/M2/M3) and classifies a list of raw BGR frames,
preprocessed exactly like the training frames (224x224 resize without crop,
ImageNet normalization)."""
import json
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.data_preprocessing.extract_frames import resize_frame
from src.models.backbone import FrameEncoder
from src.models.M2_CNN_LSTM import build_model as build_m2
from src.models.M3_CNN_Transformer import build_model as build_m3
from src.train_m1_cnn_svm import pool_clip_features

MODELS = {
    "m1": ("M1 CNN+SVM", config.M1_RESULTS_DIR, None),
    "m2": ("M2 CNN+LSTM", config.M2_RESULTS_DIR, build_m2),
    "m3": ("M3 CNN+Transformer", config.M3_RESULTS_DIR, build_m3),
}


def best_model_key() -> str:
    """Model with the highest mean test accuracy over the 5 runs."""
    scores = {}
    for key, (_, results_dir, _) in MODELS.items():
        summary = results_dir / "summary.json"
        if summary.exists():
            scores[key] = json.loads(summary.read_text())["acc_mean"]
    if not scores:
        raise FileNotFoundError("no trained model found - run the train_m*.py scripts first")
    return max(scores, key=scores.get)


def best_run(results_dir: Path) -> int:
    return json.loads((results_dir / "summary.json").read_text())["best_run"]


class ActionPredictor:
    def __init__(self, model_key: str, run: int | None = None, checkpoint: Path | None = None, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_key = model_key
        self.name, results_dir, build_fn = MODELS[model_key]
        run = run or best_run(results_dir)
        if model_key == "m1":
            self.encoder = FrameEncoder().to(self.device)
            self.svm = joblib.load(checkpoint or results_dir / f"run{run}" / "svm.joblib")
        else:
            self.model = build_fn(num_classes=len(config.CLASSES)).to(self.device).eval()
            # checkpoints hold only the temporal head; the CNN is the pretrained, frozen backbone
            state = torch.load(checkpoint or results_dir / f"run{run}" / "best.pt", map_location=self.device, weights_only=True)
            self.model.head.load_state_dict(state)
        self.source = str(checkpoint) if checkpoint else f"run{run}"
        self.mean = torch.tensor(config.IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(config.IMAGENET_STD, device=self.device).view(1, 3, 1, 1)

    def preprocess(self, frames_bgr: list[np.ndarray]) -> torch.Tensor:
        rgb = np.stack([cv2.cvtColor(resize_frame(f), cv2.COLOR_BGR2RGB) for f in frames_bgr])
        clip = torch.from_numpy(rgb).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        return ((clip - self.mean) / self.std).unsqueeze(0)  # [1, T, 3, H, W]

    @torch.no_grad()
    def predict_proba(self, frames_bgr: list[np.ndarray]) -> np.ndarray:
        clip = self.preprocess(frames_bgr)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.device.type == "cuda"):
            if self.model_key == "m1":
                return self.svm.predict_proba(pool_clip_features(self.encoder(clip)))[0]
            logits = self.model(clip)
        return torch.softmax(logits.float(), dim=1)[0].cpu().numpy()
