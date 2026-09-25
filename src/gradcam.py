"""GradCAM visualizations for selected well- and poorly-classified test
samples, per the assignment ("Pre vybrané testovacie vzorky (dobre aj zle
klasifikované) vykonajte vizualizáciu rozhodovania CNN, napr. GradCAM").
"""
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms

from src import config
from src.engine import predict


def _load_rgb(path: Path, image_size: int) -> np.ndarray:
    """Same resize/crop geometry as the eval transform, but no normalization
    -- GradCAM overlays need a plain 0-1 float RGB image to draw on."""
    tf = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
    ])
    img = tf(Image.open(path).convert("RGB"))
    return np.array(img).astype(np.float32) / 255.0


def save_gradcam_grid(
    model,
    target_layer,
    loaders: dict,
    class_names: list[str],
    device,
    out_path: Path,
    n_correct: int = 4,
    n_incorrect: int = 4,
    seed: int = config.SEED,
) -> None:
    """Picks `n_correct` correctly- and `n_incorrect` incorrectly-classified
    test samples at random and saves a GradCAM overlay grid comparing them."""
    model.eval()
    test_ds = loaders["test"].dataset
    y_true, y_pred = predict(model, loaders["test"], device)

    rng = random.Random(seed)
    correct_idx = [i for i in range(len(y_true)) if y_true[i] == y_pred[i]]
    incorrect_idx = [i for i in range(len(y_true)) if y_true[i] != y_pred[i]]
    picked_correct = rng.sample(correct_idx, min(n_correct, len(correct_idx)))
    picked_incorrect = rng.sample(incorrect_idx, min(n_incorrect, len(incorrect_idx)))

    cam = GradCAM(model=model, target_layers=[target_layer])
    cols = max(len(picked_correct), len(picked_incorrect), 1)
    fig, axes = plt.subplots(2, cols, figsize=(3.2 * cols, 7.5), squeeze=False)

    for row, (indices, row_label) in enumerate([(picked_correct, "Correct"), (picked_incorrect, "Misclassified")]):
        for col in range(cols):
            ax = axes[row][col]
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if col >= len(indices):
                continue

            idx = indices[col]
            rel_path, _ = test_ds.samples[idx]
            rgb_img = _load_rgb(test_ds.data_dir / rel_path, config.IMAGE_SIZE)
            input_tensor = preprocess_image(
                rgb_img, mean=list(config.IMAGENET_MEAN), std=list(config.IMAGENET_STD)
            ).to(device)
            # GradCAM needs a gradient graph even though the backbone is frozen
            # (all params requires_grad=False), so mark the input itself.
            input_tensor.requires_grad_(True)

            grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(y_pred[idx])])[0]
            overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            ax.imshow(overlay)
            true_name, pred_name = class_names[y_true[idx]], class_names[y_pred[idx]]
            ax.set_title(f"true: {true_name}\npred: {pred_name}", fontsize=9)
        axes[row][0].set_ylabel(row_label, fontsize=13, fontweight="bold")

    fig.suptitle("GradCAM - correctly vs. incorrectly classified test samples", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.subplots_adjust(hspace=0.35)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
