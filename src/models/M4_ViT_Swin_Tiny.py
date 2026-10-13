"""M4: vision transformer (Swin-Tiny) with transfer learning.

Swin Transformer (Liu et al., ICCV 2021) is a hierarchical vision transformer:
self-attention is computed inside local 7x7 windows that are shifted between
consecutive blocks, and patch merging halves the resolution between stages,
so it produces a CNN-like feature pyramid (56->28->14->7). Swin-T (28.3M
params, 4.5 GMACs) is size-matched to ResNet50 (M1) and ConvNeXt-T (M3).
Pretrained on ImageNet-1k (timm `swin_tiny_patch4_window7_224.ms_in1k`,
81.2 % top-1), so the pretraining data matches the other models.
"""
import torch
import torch.nn as nn

from src.models.timm_backbone import build_timm_model, unfreeze_submodule, unfreeze_top

TIMM_NAME = "swin_tiny_patch4_window7_224.ms_in1k"
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)

# patch_embed -> layers.0 -> ... -> layers.3 -> norm -> head.fc; unfrozen top-down like M1
BACKBONE_PREFIXES = ("patch_embed", "layers")
UNFREEZE_STAGES = ["layers.3", "layers.2", "layers.1", "layers.0"]


def build_model(
    num_classes: int,
    freeze_backbone: bool = True,
    pretrained: bool = True,
    grad_checkpointing: bool = False,
) -> nn.Module:
    return build_timm_model(TIMM_NAME, num_classes, freeze_backbone, pretrained, grad_checkpointing)


def unfreeze_stage(model: nn.Module, stage: str) -> None:
    """Unfreeze one backbone stage; the last stage also brings the final
    LayerNorm (norm) along, since it directly feeds the classifier."""
    unfreeze_submodule(model, stage)
    if stage == UNFREEZE_STAGES[0]:
        unfreeze_top(model, BACKBONE_PREFIXES)


def gradcam_target(model: nn.Module) -> nn.Module:
    """Last stage output (7x7 token map at 224 px), the same layer as for
    ConvNeXt (M3). pytorch-grad-cam's suggested `blocks[-1].norm1` produced
    strong corner artifacts on this dataset, the stage output does not."""
    return model.layers[-1]


def reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
    """Swin token maps -> NCHW for GradCAM. timm >= 0.9 keeps tokens as
    (B, H, W, C); older versions use (B, H*W, C)."""
    if tensor.ndim == 3:
        b, n, c = tensor.shape
        side = int(n ** 0.5)
        tensor = tensor.reshape(b, side, side, c)
    return tensor.permute(0, 3, 1, 2)
