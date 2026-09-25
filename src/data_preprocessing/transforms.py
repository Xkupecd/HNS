"""Image transforms: a plain (no-augmentation) pipeline and an augmented one.

Used to answer the assignment's requirement of comparing training with and
without augmentation for a chosen model.
"""
from torchvision import transforms

from src import config


def build_eval_transform(image_size: int = config.IMAGE_SIZE) -> transforms.Compose:
    """Resize/crop + normalize only. Used for val/test and as the "no augmentation" train pipeline."""
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])


def build_train_transform(image_size: int = config.IMAGE_SIZE, augment: bool = True) -> transforms.Compose:
    """Training pipeline. `augment=False` gives the same crop/normalize as eval (baseline for the
    augmentation-ablation experiment)."""
    if not augment:
        return transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
        ])

    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.RandomAutocontrast(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
        transforms.RandomErasing(p=0.15),
    ])
