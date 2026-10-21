"""M2: CNN + LSTM (Donahue et al., "Long-term Recurrent Convolutional
Networks", CVPR 2015).

The frozen CNN encodes every frame independently, a single-layer LSTM then
reads the sequence of (LayerNorm-ed) frame features in time order and its
last hidden state is classified. Being unidirectional, the same model also
runs on a live webcam stream without seeing future frames.
"""
import torch
import torch.nn as nn

from src.models.backbone import FEATURE_DIM, VideoClassifier

HIDDEN_DIM = 256


class LSTMHead(nn.Module):
    def __init__(self, num_classes: int, hidden_dim: int = HIDDEN_DIM, dropout: float = 0.5):
        super().__init__()
        self.in_norm = nn.LayerNorm(FEATURE_DIM)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(FEATURE_DIM, hidden_dim, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """[B, T, 1024] frame features -> [B, num_classes]"""
        _, (h_n, _) = self.lstm(self.dropout(self.in_norm(feats)))   # h_n: [1, B, hidden]
        return self.classifier(self.dropout(h_n[-1]))


def build_head(num_classes: int) -> nn.Module:
    return LSTMHead(num_classes)


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    return VideoClassifier(build_head(num_classes), pretrained=pretrained)
