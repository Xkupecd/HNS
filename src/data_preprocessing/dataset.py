"""PyTorch Dataset backed by the manifest CSV produced by split_dataset.py."""
import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src import config
from src.data_preprocessing.transforms import build_eval_transform, build_train_transform


class FoodDataset(Dataset):
    def __init__(self, manifest_path: Path, split: str, data_dir: Path = config.DATA_DIR, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = config.CLASSES
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}

        with open(manifest_path, newline="") as f:
            reader = csv.DictReader(f)
            self.samples = [(row["path"], row["label"]) for row in reader if row["split"] == split]

        if not self.samples:
            raise ValueError(f"No samples found for split={split!r} in {manifest_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        rel_path, label = self.samples[idx]
        image = Image.open(self.data_dir / rel_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.class_to_idx[label]


def build_dataloaders(
    manifest_path: Path = config.SPLIT_DIR / "manifest.csv",
    augment_train: bool = True,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
) -> dict[str, DataLoader]:
    eval_tf = build_eval_transform()
    train_tf = build_train_transform(augment=augment_train)

    train_ds = FoodDataset(manifest_path, "train", transform=train_tf)
    val_ds = FoodDataset(manifest_path, "val", transform=eval_tf)
    test_ds = FoodDataset(manifest_path, "test", transform=eval_tf)

    common = dict(num_workers=num_workers, pin_memory=True)
    if num_workers > 0:
        common["prefetch_factor"] = 4
        common["persistent_workers"] = True

    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, **common),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common),
    }
