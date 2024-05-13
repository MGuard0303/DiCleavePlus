import math

import torch
from torch import nn


class TFModel(nn.Module):
    # noinspection PyTypeChecker
    def __init__(self, input_dimension: int, hidden_feature: int, name: str = "TFModel"):
        super().__init__()
        self.name = name

        self.bn = nn.BatchNorm1d(num_features=hidden_feature)

        self.gru = nn.GRU(input_size=input_dimension, hidden_size=hidden_feature, num_layers=1, dropout=0.5)

        self.encoder = nn.TransformerEncoderLayer(d_model=input_dimension, nhead=1, dim_feedforward=256)
        self.encoder_container = nn.TransformerEncoder(encoder_layer=self.encoder, num_layers=1,
                                                       enable_nested_tensor=False)
        self.position_encoder = PositionEncoder(d_model=input_dimension, length=200)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_dimension, out_channels=hidden_feature, kernel_size=7, padding=3),
            self.bn,
            nn.ReLU(),
            nn.Linear(in_features=200, out_features=14)
        )

        self.flat = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(in_features=14*hidden_feature, out_features=1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU()
        )

        self.output_layer = nn.Sequential(
            nn.Linear(in_features=64, out_features=14),
            nn.LogSoftmax(dim=1)
        )

    # Shape of pattern inputs is (Batch, Length, Dimension)
    # Shape of sequence inputs is (Batch, Length, Dimension)
    def forward(self, pattern: torch.Tensor, sequence: torch.Tensor) -> torch.Tensor:
        pattern = torch.permute(input=pattern, dims=(1, 0, 2))
        pattern, _ = self.gru(pattern)  # (L, B, D)
        pattern = torch.permute(input=pattern, dims=(1, 2, 0))
        pattern = self.bn(pattern)  # (B, D, L)
        pattern = torch.permute(input=pattern, dims=(0, 2, 1))

        sequence = torch.permute(input=sequence, dims=(1, 0, 2))
        sequence = self.position_encoder(sequence)
        sequence = self.encoder_container(sequence)  # (L, B, D)
        sequence = torch.permute(input=sequence, dims=(1, 2, 0))
        sequence = self.conv(sequence)  # (B, D, L)
        sequence = torch.permute(input=sequence, dims=(0, 2, 1))

        embed = pattern + sequence
        embed = self.flat(embed)
        embed = self.fc(embed)
        embed = self.output_layer(embed)

        return embed


# The input of PositionEncoder is (Length, Batch, Dimension)
class PositionEncoder(nn.Module):
    def __init__(self, d_model: int, length: int, n: float = 10000.0):
        super().__init__()
        position = torch.arange(length).unsqueeze(1)
        denominator = torch.exp(torch.arange(0, d_model, 2) * (-math.log(n) / d_model))
        pe = torch.zeros(length, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * denominator)

        if d_model % 2 != 0:
            pe[:, 0, 1::2] = torch.cos(position * denominator[:-1])
        else:
            pe[:, 0, 1::2] = torch.cos(position * denominator)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]

        return x
