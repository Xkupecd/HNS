"""M3: optimized CNN (ConvNeXt-Tiny) with transfer learning.

ConvNeXt (Liu et al., "A ConvNet for the 2020s", CVPR 2022) is a ResNet
modernized step by step towards the Swin Transformer design: patchify stem,
inverted bottlenecks, 7x7 depthwise convolutions, LayerNorm and GELU.
ConvNeXt-T (28.6M params, 4.5 GMACs) is the size-matched counterpart of
ResNet50 (M1) and Swin-T (M4). Pretrained on ImageNet-1k (timm
`convnext_tiny.fb_in1k`, 82.1 % top-1), so the pretraining data matches M1.
"""
import torch.nn as nn

from src.models.timm_backbone import build_timm_model, unfreeze_submodule, unfreeze_top

TIMM_NAME = "convnext_tiny.fb_in1k"
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)

# stem -> stages.0 -> ... -> stages.3 -> head.norm -> head.fc; unfrozen top-down like M1
BACKBONE_PREFIXES = ("stem", "stages")
UNFREEZE_STAGES = ["stages.3", "stages.2", "stages.1", "stages.0"]


def build_model(
    num_classes: int,
    freeze_backbone: bool = True,
    pretrained: bool = True,
    grad_checkpointing: bool = False,
) -> nn.Module:
    return build_timm_model(TIMM_NAME, num_classes, freeze_backbone, pretrained, grad_checkpointing)


def unfreeze_stage(model: nn.Module, stage: str) -> None:
    """Unfreeze one backbone stage; the last stage also brings the final
    LayerNorm (head.norm) along, since it directly feeds the classifier."""
    unfreeze_submodule(model, stage)
    if stage == UNFREEZE_STAGES[0]:
        unfreeze_top(model, BACKBONE_PREFIXES)


def gradcam_target(model: nn.Module) -> nn.Module:
    """Last stage output (NCHW, 7x7 at 224 px), analogous to ResNet's layer4."""
    return model.stages[-1]


reshape_transform = None  # CNN activations are already NCHW
