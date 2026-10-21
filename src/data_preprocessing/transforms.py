"""Clip-level transforms: one set of random augmentation parameters is drawn
per clip and applied to all of its frames, so the motion between frames stays
intact (independent per-frame crops would look like camera shake)."""
import random

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import ColorJitter, RandomResizedCrop

from src import config


class ClipTransform:
    def __init__(
        self,
        train: bool,
        image_size: int = config.IMAGE_SIZE,
        mean: tuple = config.IMAGENET_MEAN,
        std: tuple = config.IMAGENET_STD,
    ):
        self.train = train
        self.image_size = image_size
        self.mean, self.std = mean, std
        self.jitter = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.03)

    def __call__(self, frames: list[Image.Image]) -> torch.Tensor:
        if self.train:
            i, j, h, w = RandomResizedCrop.get_params(frames[0], scale=(0.6, 1.0), ratio=(0.8, 1.25))
            flip = random.random() < 0.5
            _, b, c, s, hue = ColorJitter.get_params(
                self.jitter.brightness, self.jitter.contrast, self.jitter.saturation, self.jitter.hue
            )
            out = []
            for img in frames:
                img = TF.resized_crop(img, i, j, h, w, [self.image_size, self.image_size], antialias=True)
                if flip:
                    img = TF.hflip(img)
                img = TF.adjust_hue(TF.adjust_saturation(TF.adjust_contrast(TF.adjust_brightness(img, b), c), s), hue)
                out.append(TF.to_tensor(img))
        else:
            out = [TF.to_tensor(TF.resize(img, [self.image_size, self.image_size], antialias=True)) for img in frames]
        clip = torch.stack(out)  # [T, 3, H, W]
        return TF.normalize(clip, self.mean, self.std)
