"""M5: hybrid CNN + transformer (CoAtNet-0) with transfer learning.

CoAtNet (Dai et al., "CoAtNet: Marrying Convolution and Attention for All
Data Sizes", NeurIPS 2021) stacks convolution and self-attention vertically:
the first two stages are MBConv (depthwise-separable convolution) blocks,
which give translation equivariance and a good inductive bias for small
data, the last two stages are transformer blocks with relative attention,
which give global receptive field and capacity. timm's CoAtNet-0
(`coatnet_0_rw_224.sw_in1k`, 27.4M params, 4.4 GMACs, 82.4 % ImageNet-1k
top-1) is size-matched to ResNet50 (M1), ConvNeXt-T (M3) and Swin-T (M4).

Unlike the other models it was pretrained with mean=std=0.5 normalization,
which train/gradcam scripts pass on via NORM_MEAN/NORM_STD. The MBConv
stages contain BatchNorm whose running statistics keep updating in
`model.train()` even with a frozen backbone, the same as ResNet50's BN in M1.
"""
import torch.nn as nn

from src.models.timm_backbone import build_timm_model, unfreeze_submodule, unfreeze_top

TIMM_NAME = "coatnet_0_rw_224.sw_in1k"
NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)

# stem -> stages.0-1 (MBConv) -> stages.2-3 (transformer) -> norm -> head.fc; unfrozen top-down like M1
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
    LayerNorm (norm) along, since it directly feeds the classifier."""
    unfreeze_submodule(model, stage)
    if stage == UNFREEZE_STAGES[0]:
        unfreeze_top(model, BACKBONE_PREFIXES)


def gradcam_target(model: nn.Module) -> nn.Module:
    """Last (transformer) stage output, NCHW 7x7 at 224 px, the same layer
    as for ConvNeXt (M3) and Swin (M4)."""
    return model.stages[-1]


reshape_transform = None  # timm's CoAtNet keeps NCHW feature maps in its transformer stages
