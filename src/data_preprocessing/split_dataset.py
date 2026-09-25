"""Split the raw per-class image folders in `data/` into train/val/test (80/10/10).

Writes a CSV manifest (data_split/manifest.csv with columns: path,label,split)
instead of copying images, so the large raw dataset stays untouched.
"""
import csv
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def list_class_images(class_dir: Path) -> list[Path]:
    return sorted(p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def split_indices(n: int, train_ratio: float, val_ratio: float) -> tuple[slice, slice, slice]:
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    return slice(0, n_train), slice(n_train, n_train + n_val), slice(n_train + n_val, n)


def build_manifest(
    data_dir: Path = config.DATA_DIR,
    train_ratio: float = config.TRAIN_RATIO,
    val_ratio: float = config.VAL_RATIO,
    seed: int = config.SEED,
) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    classes = sorted(p.name for p in data_dir.iterdir() if p.is_dir())
    for class_name in classes:
        images = list_class_images(data_dir / class_name)
        rng.shuffle(images)
        train_sl, val_sl, test_sl = split_indices(len(images), train_ratio, val_ratio)
        for split_name, sl in (("train", train_sl), ("val", val_sl), ("test", test_sl)):
            for img_path in images[sl]:
                rows.append({
                    "path": str(img_path.relative_to(data_dir)),
                    "label": class_name,
                    "split": split_name,
                })
    return rows


def write_manifest(rows: list[dict], out_path: Path = config.SPLIT_DIR / "manifest.csv") -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "split"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = build_manifest()
    write_manifest(rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["split"]] = counts.get(row["split"], 0) + 1
    print(f"Wrote {len(rows)} entries to {config.SPLIT_DIR / 'manifest.csv'}")
    print(counts)


if __name__ == "__main__":
    main()
