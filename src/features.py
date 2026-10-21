"""Compute and cache the frozen CNN's frame features for all splits.

    train: N_AUG_CLIPS augmented clips per video (temporal jitter + clip-
           consistent crop/flip/color augmentation)  -> [N, A, T, D]
    val/test: one deterministic clip per video        -> [N, T, D]

Stored as float16 in data/features/<backbone>/<name>.npz with the labels and
video names. All three methods train on these caches; M2/M3 draw one of the
A augmented versions of every video per epoch.

Usage:
    python -m src.features            # HMDB51 train/val/test (skips existing caches)
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import config
from src.data_preprocessing.dataset import VideoClipDataset, read_manifest, split_samples, worker_init
from src.engine import set_seed
from src.models.backbone import FrameEncoder

CACHE_DIR = config.FEATURES_DIR / config.BACKBONE


@torch.no_grad()
def encode_dataset(encoder, dataset, device) -> np.ndarray:
    """One pass over `dataset` -> [N, T, D] float16 frame features."""
    loader = DataLoader(dataset, batch_size=config.EXTRACT_BATCH, shuffle=False,
                        num_workers=config.NUM_WORKERS, worker_init_fn=worker_init, pin_memory=True)
    out = []
    for clips, _ in loader:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            out.append(encoder(clips.to(device, non_blocking=True)).half().cpu().numpy())
    return np.concatenate(out)


def build_cache(name: str, samples: list[tuple[str, int]], frames_dir: Path, augmented: bool,
                encoder=None, device=None, n_aug: int = config.N_AUG_CLIPS, force: bool = False) -> Path:
    path = CACHE_DIR / f"{name}.npz"
    if path.exists() and not force:
        return path
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder or FrameEncoder().to(device)
    set_seed(config.SEED)
    ds = VideoClipDataset(samples, frames_dir, train=augmented, augment=augmented)
    if augmented:
        X = np.stack([encode_dataset(encoder, ds, device) for _ in range(n_aug)], axis=1)
    else:
        X = encode_dataset(encoder, ds, device)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, X=X, y=np.array([s[1] for s in samples]), videos=np.array([s[0] for s in samples]))
    print(f"cached {name}: {X.shape} -> {path}")
    return path


def load_cache(name: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(CACHE_DIR / f"{name}.npz")
    return data["X"], data["y"]


class FeatureDataset(Dataset):
    """Cached clips; for augmented caches every access draws one random version."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X, self.y = torch.from_numpy(X), torch.from_numpy(y).long()
        self.augmented = X.ndim == 4

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x = self.X[idx]
        if self.augmented:
            x = x[torch.randint(len(x), (1,)).item()]
        return x.float(), self.y[idx]


def build_feature_loaders(batch_size: int = config.BATCH_SIZE) -> dict:
    loaders = {}
    for split in ("train", "val", "test"):
        X, y = load_cache(split)
        loaders[split] = DataLoader(FeatureDataset(X, y), batch_size=batch_size, shuffle=split == "train")
    return loaders


def main() -> None:
    rows = read_manifest()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = FrameEncoder().to(device)
    for split in ("train", "val", "test"):
        build_cache(split, split_samples(rows, split), config.FRAMES_DIR, augmented=split == "train",
                    encoder=encoder, device=device)


if __name__ == "__main__":
    main()
