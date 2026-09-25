"""M1: CNN backbone (ResNet50) with transfer learning.

Pretrained on ImageNet, final fully-connected layer replaced to match the
project's number of classes. Supports freezing the backbone (train only the
new head) or progressively unfreezing later layers, as required by the
assignment's "gradual unfreezing" experiment.
"""
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

# resnet50 layer4 -> layer3 -> layer2 -> layer1, unfrozen in this order
UNFREEZE_STAGES = ["layer4", "layer3", "layer2", "layer1"]


def build_model(num_classes: int, freeze_backbone: bool = True, pretrained: bool = True) -> nn.Module:
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = resnet50(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def unfreeze_stage(model: nn.Module, stage: str) -> None:
    """Unfreeze one named backbone stage (e.g. "layer4") for gradual unfreezing."""
    for param in getattr(model, stage).parameters():
        param.requires_grad = True


def trainable_parameters(model: nn.Module):
    return (p for p in model.parameters() if p.requires_grad)
