"""M2: CNN backbone (ResNet50) with transfer learning + CBAM attention.

Same ResNet50 transfer-learning setup as M1, with a CBAM (Convolutional
Block Attention Module) inserted after each residual stage (layer1-layer4),
per the assignment's "M1 + CBAM" requirement.
"""
import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

UNFREEZE_STAGES = ["layer4", "layer3", "layer2", "layer1"]


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        avg_out = self.mlp(self.avg_pool(x).view(b, c))
        max_out = self.mlp(self.max_pool(x).view(b, c))
        scale = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * scale


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = x.mean(dim=1, keepdim=True)
        max_out, _ = x.max(dim=1, keepdim=True)
        scale = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * scale


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class ResNet50CBAM(nn.Module):
    """ResNet50 with a CBAM block appended after each of layer1-layer4."""

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)

        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.cbam1 = CBAM(256)
        self.layer2 = backbone.layer2
        self.cbam2 = CBAM(512)
        self.layer3 = backbone.layer3
        self.cbam3 = CBAM(1024)
        self.layer4 = backbone.layer4
        self.cbam4 = CBAM(2048)

        self.avgpool = backbone.avgpool
        self.fc = nn.Linear(backbone.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.cbam1(self.layer1(x))
        x = self.cbam2(self.layer2(x))
        x = self.cbam3(self.layer3(x))
        x = self.cbam4(self.layer4(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def build_model(num_classes: int, freeze_backbone: bool = True, pretrained: bool = True) -> nn.Module:
    model = ResNet50CBAM(num_classes=num_classes, pretrained=pretrained)

    if freeze_backbone:
        for name, param in model.named_parameters():
            # keep the (randomly-initialized) CBAM blocks and the new head trainable
            if name.startswith(("stem", "layer1", "layer2", "layer3", "layer4")):
                param.requires_grad = False

    return model


def unfreeze_stage(model: nn.Module, stage: str) -> None:
    """Unfreeze one named backbone stage (e.g. "layer4") for gradual unfreezing."""
    for param in getattr(model, stage).parameters():
        param.requires_grad = True


def trainable_parameters(model: nn.Module):
    return (p for p in model.parameters() if p.requires_grad)
