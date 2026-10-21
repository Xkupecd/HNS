"""GradCAM visualizations of the video models' decisions for selected well-
and poorly-classified test videos, per the assignment ("Pre vybrané testovacie
vzorky vykonajte vizualizáciu rozhodovania modelov (napr. GradCAM metóda)").

The clip-level class score is backpropagated through the temporal model (LSTM
or Transformer) to the last stage of the frozen CNN for every frame, so each
frame gets its own heatmap. Heatmaps are normalized over the whole clip, not per frame:
frames that barely influence the decision stay dark, which shows *when* in
the clip the model looks, not only *where*.

M1 (RBF SVM on pooled features) has no gradient path to a class score, so
GradCAM is computed for M2 and M3.

Usage:
    python -m src.gradcam                # best run of M2 and M3
    python -m src.gradcam --model m2 --run 3
"""
import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_grad_cam.utils.image import show_cam_on_image

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import config
from torch.utils.data import DataLoader

from src.data_preprocessing.dataset import VideoClipDataset, read_manifest, split_samples
from src.engine import predict
from src.features import FeatureDataset, load_cache
from src.models.M2_CNN_LSTM import build_model as build_m2
from src.models.M3_CNN_Transformer import build_model as build_m3

MODELS = {
    "m2": ("M2_cnn_lstm", config.M2_RESULTS_DIR, build_m2),
    "m3": ("M3_cnn_transformer", config.M3_RESULTS_DIR, build_m3),
}
FRAMES_SHOWN = 8  # every second of the 16 input frames


def video_gradcam(model, clip: torch.Tensor, target_class: int) -> np.ndarray:
    """clip [T, 3, H, W] -> per-frame CAMs [T, H, W] in 0..1 (clip-normalized)."""
    store = {}
    layer = model.encoder.gradcam_target()
    h1 = layer.register_forward_hook(lambda m, i, o: store.update(act=o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: store.update(grad=go[0]))
    try:
        model.zero_grad(set_to_none=True)
        # the CNN is frozen (no param needs grad), so the input itself must
        # require grad for the activations to get a gradient
        clip = clip.detach().requires_grad_(True)
        # cuDNN's LSTM only supports backward in train mode; dropout must stay off
        with torch.backends.cudnn.flags(enabled=False):
            logits = model(clip.unsqueeze(0))
            logits[0, target_class].backward()
    finally:
        h1.remove()
        h2.remove()

    act, grad = store["act"].detach(), store["grad"].detach()        # [T, C, h, w]
    weights = grad.mean(dim=(2, 3), keepdim=True)                     # per-frame channel weights
    cam = F.relu((weights * act).sum(1, keepdim=True))                # [T, 1, h, w]
    cam = F.interpolate(cam, size=clip.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    cam = cam / cam.max().clamp_min(1e-8)
    return cam.cpu().numpy()


def save_gradcam_grid(model, test_ds, device, out_path: Path, title: str,
                      n_correct: int = 3, n_incorrect: int = 3, seed: int = config.SEED) -> None:
    model.eval()
    X, y_true = load_cache("test")
    y_true = y_true.tolist()
    _, y_pred = predict(model.head, DataLoader(FeatureDataset(X, np.array(y_true)), batch_size=config.BATCH_SIZE), device)

    rng = random.Random(seed)
    correct_idx = [i for i in range(len(y_true)) if y_true[i] == y_pred[i]]
    incorrect_idx = [i for i in range(len(y_true)) if y_true[i] != y_pred[i]]
    picked = [(i, "correct") for i in rng.sample(correct_idx, min(n_correct, len(correct_idx)))]
    picked += [(i, "misclassified") for i in rng.sample(incorrect_idx, min(n_incorrect, len(incorrect_idx)))]

    step = config.CLIP_LEN // FRAMES_SHOWN
    fig, axes = plt.subplots(len(picked), FRAMES_SHOWN, figsize=(2.0 * FRAMES_SHOWN, 2.35 * len(picked)), squeeze=False)
    for row, (idx, kind) in enumerate(picked):
        clip, _ = test_ds[idx]
        cams = video_gradcam(model, clip.to(device), y_pred[idx])
        rgb = [np.asarray(f, dtype=np.float32) / 255.0 for f in test_ds.load_frames(idx)]
        for col in range(FRAMES_SHOWN):
            t = col * step
            ax = axes[row][col]
            ax.imshow(show_cam_on_image(rgb[t], cams[t], use_rgb=True))
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"t={t}", fontsize=8)
        true_name, pred_name = config.CLASSES[y_true[idx]], config.CLASSES[y_pred[idx]]
        axes[row][0].set_ylabel(f"{kind}\ntrue: {true_name}\npred: {pred_name}", fontsize=9,
                                color="green" if kind == "correct" else "red")

    fig.suptitle(f"GradCAM - {title}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS), nargs="*", default=list(MODELS))
    parser.add_argument("--run", type=int, default=None, help="run index; default = best run from summary.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_ds = VideoClipDataset(split_samples(read_manifest(), "test"), config.FRAMES_DIR, train=False)
    for key in args.model:
        model_name, results_dir, build_fn = MODELS[key]
        run = args.run or json.loads((results_dir / "summary.json").read_text())["best_run"]
        model = build_fn(num_classes=len(config.CLASSES)).to(device)
        model.head.load_state_dict(torch.load(results_dir / f"run{run}" / "best.pt", map_location=device, weights_only=True))
        out_path = results_dir / f"gradcam_run{run}.png"
        save_gradcam_grid(model, test_ds, device, out_path, title=f"{model_name} (run{run}), test videos")
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
