"""Shared transfer-learning helpers for the timm backbones (M3-M5).

All three models (ConvNeXt, Swin, CoAtNet) are built the same way as M1:
ImageNet-1k pretrained weights, a new linear classifier with `num_classes`
outputs, and optionally a frozen backbone where only that new classifier is
trained. Gradual unfreezing then re-enables whole backbone stages (named by
their module path, e.g. "stages.3"), while the stem / patch embedding stays
frozen just like ResNet50's conv1 in M1.
"""
import timm
import torch.nn as nn
from timm.data import resolve_data_config


def build_timm_model(
    name: str,
    num_classes: int,
    freeze_backbone: bool = True,
    pretrained: bool = True,
    grad_checkpointing: bool = False,
) -> nn.Module:
    model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        # only the newly created classifier layer is trainable (same as M1's fc)
        for param in model.get_classifier().parameters():
            param.requires_grad = True

    if grad_checkpointing:
        # trades compute for memory; only needed if the fully unfrozen model
        # does not fit into GPU memory at the common batch size
        model.set_grad_checkpointing(True)
    return model


def unfreeze_submodule(model: nn.Module, path: str) -> None:
    for param in model.get_submodule(path).parameters():
        param.requires_grad = True


def unfreeze_top(model: nn.Module, backbone_prefixes: tuple[str, ...]) -> None:
    """Unfreezes everything outside the stem and the backbone stages, i.e. the
    final normalization layer(s) between the last stage and the classifier."""
    for name, param in model.named_parameters():
        if not name.startswith(backbone_prefixes):
            param.requires_grad = True


def pretrained_normalization(model: nn.Module) -> tuple[tuple, tuple]:
    """Mean/std the timm weights were pretrained with (used to verify the
    constants hard-coded in each model file)."""
    cfg = resolve_data_config({}, model=model)
    return tuple(cfg["mean"]), tuple(cfg["std"])
