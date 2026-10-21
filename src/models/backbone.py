"""Shared frozen per-frame CNN encoder for all three methods, and the
encoder + temporal-head wrapper used for inference (camera, GradCAM).

ConvNeXt-Base pretrained on ImageNet-22k (timm `convnext_base.fb_in22k`),
the best of the probed backbones by validation accuracy. The classifier is
removed; every frame is mapped to a 1024-d feature vector (global average
pool + LayerNorm).

The encoder stays frozen: the first version fine-tuned the last stage of a
ConvNeXt-T and overfitted the ~360 training videos within 3 epochs (train
accuracy 100 %, val ~60 %, see Results/ablation_v1_*). Frozen, the frame
features can be computed once and cached (features.py), and the three
methods differ only in how they model time.
"""
import timm
import torch
import torch.nn as nn

from src import config

FEATURE_DIM = 1024


class FrameEncoder(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.cnn = timm.create_model(config.BACKBONE, pretrained=pretrained, num_classes=0)
        for param in self.cnn.parameters():
            param.requires_grad = False
        self.eval()

    def train(self, mode: bool = True):
        # always eval: frozen weights, and LayerNorm/DropPath must behave as at feature caching
        return super().train(False)

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        """[B, T, 3, H, W] -> [B, T, FEATURE_DIM]"""
        b, t = clips.shape[:2]
        feats = self.cnn(clips.flatten(0, 1))
        return feats.view(b, t, -1)

    def gradcam_target(self) -> nn.Module:
        """Last stage output (NCHW, 7x7 at 224 px)."""
        return self.cnn.stages[-1]


class VideoClassifier(nn.Module):
    """Frozen encoder + trained temporal head; only the head is saved/loaded."""

    def __init__(self, head: nn.Module, pretrained: bool = True):
        super().__init__()
        self.encoder = FrameEncoder(pretrained=pretrained)
        self.head = head

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(clips))
