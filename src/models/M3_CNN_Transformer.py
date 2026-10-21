"""M3: hybrid CNN + temporal Transformer.

Same frozen frame encoder as M2, but the LSTM is replaced by a small
Transformer encoder over time (like ViT over frames instead of image
patches, cf. the "late temporal modeling" of Kalfaoglu et al. 2020): frame
features are LayerNorm-ed and projected to 256-d, a learnable [CLS] token
and positional embeddings are added, and two self-attention layers let every
frame attend to all others. The [CLS] output is classified.
"""
import torch
import torch.nn as nn

from src import config
from src.models.backbone import FEATURE_DIM, VideoClassifier

EMBED_DIM = 256
NUM_LAYERS = 2
NUM_HEADS = 4


class TransformerHead(nn.Module):
    def __init__(self, num_classes: int, clip_len: int = config.CLIP_LEN, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(FEATURE_DIM), nn.Dropout(0.5), nn.Linear(FEATURE_DIM, EMBED_DIM))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, EMBED_DIM))
        self.pos_embed = nn.Parameter(torch.zeros(1, clip_len + 1, EMBED_DIM))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        layer = nn.TransformerEncoderLayer(
            EMBED_DIM, NUM_HEADS, dim_feedforward=2 * EMBED_DIM, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, NUM_LAYERS, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(EMBED_DIM)
        self.classifier = nn.Linear(EMBED_DIM, num_classes)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """[B, T, 1024] frame features -> [B, num_classes]"""
        tokens = self.proj(feats)                                   # [B, T, 256]
        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.pos_embed[:, : tokens.size(1) + 1]
        out = self.norm(self.temporal(tokens))
        return self.classifier(out[:, 0])


def build_head(num_classes: int) -> nn.Module:
    return TransformerHead(num_classes)


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    return VideoClassifier(build_head(num_classes), pretrained=pretrained)
