import math

import torch
from torch import nn


class TFModel(nn.Module):
    def __init__(self, hidden_feature: int, name: str = "TFModel"):
        super().__init__()
        self.name = name

        self.lstm = nn.LSTM(input_size=hidden_feature, hidden_size=hidden_feature, num_layers=2, batch_first=True)

        self.encoder = nn.TransformerEncoderLayer(d_model=hidden_feature, nhead=8, dim_feedforward=256)
        self.encoder_container = nn.TransformerEncoder(encoder_layer=self.encoder, num_layers=2,
                                                       enable_nested_tensor=False)
        self.position_encoder = PositionEncoder(d_model=hidden_feature, length=200)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=hidden_feature, out_channels=hidden_feature, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=hidden_feature, eps=1e-08),
            nn.LeakyReLU()
        )

        # Process after convolutional structure
        self.identity = nn.Sequential()  # Resnet short-cut for convolution structure
        self.bn = nn.BatchNorm1d(num_features=hidden_feature, eps=1e-08)
        self.seq_linear = nn.Linear(in_features=200, out_features=14)

        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(in_features=14*hidden_feature, out_features=512),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(in_features=512, out_features=128),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(in_features=128, out_features=32),
            nn.LeakyReLU()
        )

        self.output_layer = nn.Sequential(
            nn.Linear(in_features=32, out_features=14),
            nn.LogSoftmax(dim=1)
        )

    # Shape of pattern inputs is (Batch, Length, Dimension)
    def forward(self, sequence: torch.Tensor, pattern: torch.Tensor) -> torch.Tensor:
        self.lstm.flatten_parameters()
        pattern_identity = self.identity(pattern)
        pattern, _ = self.lstm(pattern)  # (B, L, D)
        pattern = torch.permute(input=pattern, dims=(0, 2, 1))
        pattern = self.bn(pattern)  # (B, D, L)
        pattern = torch.permute(input=pattern, dims=(0, 2, 1))  # Final shape (B, L, D)
        pattern += pattern_identity

        sequence_identity = self.identity(sequence)
        sequence = torch.permute(input=sequence, dims=(1, 0, 2))
        sequence = self.position_encoder(sequence)  # (L, B, D)
        sequence = self.encoder_container(sequence)  # (L, B, D)
        sequence = torch.permute(input=sequence, dims=(1, 2, 0))
        sequence = self.conv(sequence)  # (B, D, L)
        sequence = torch.permute(input=sequence, dims=(0, 2, 1))
        sequence += sequence_identity  # (B, L, D)
        sequence = torch.permute(input=sequence, dims=(0, 2, 1))
        sequence = self.seq_linear(sequence)  # (B, D, L)
        sequence = torch.permute(input=sequence, dims=(0, 2, 1))  # Final shape (B, L, D)

        embed = pattern + sequence
        embed = self.flatten(embed)
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


# Fuse different kmers embeddings using attention
# softmax_dim should be the length dimension of input embedding
# Assume the input size is (Batch, Length, Dimension)
class FusionLayer(nn.Module):
    def __init__(self, hidden_feature: int, softmax_dim: int):
        super().__init__()
        self.softmax = nn.Softmax(dim=softmax_dim)
        self.fc = nn.Linear(in_features=hidden_feature, out_features=1)

    def forward(self, *args: torch.Tensor) -> tuple:
        fused = torch.zeros_like(args[0])
        ws = []

        for t in args:
            w = self.softmax(self.fc(t))
            fused += w * t
            ws.append(w)

        return fused, ws


# Embed and fuse sequence and secondary kmer raw tensors into sequence input and pattern input
class EmbeddingLayer(nn.Module):
    def __init__(self, hidden_feature: int, softmax_dim: int, is_sec: bool = False):
        super().__init__()
        if is_sec:
            self.e1 = nn.Embedding(num_embeddings=4, embedding_dim=hidden_feature, padding_idx=0)
            self.e2 = nn.Embedding(num_embeddings=13, embedding_dim=hidden_feature, padding_idx=0)
            self.e3 = nn.Embedding(num_embeddings=40, embedding_dim=hidden_feature, padding_idx=0)

        else:
            self.e1 = nn.Embedding(num_embeddings=5, embedding_dim=hidden_feature, padding_idx=0)
            self.e2 = nn.Embedding(num_embeddings=21, embedding_dim=hidden_feature, padding_idx=0)
            self.e3 = nn.Embedding(num_embeddings=85, embedding_dim=hidden_feature, padding_idx=0)

        self.fusion = FusionLayer(hidden_feature=hidden_feature, softmax_dim=softmax_dim)

    def forward(self, t1: torch.Tensor, t2: torch.Tensor, t3: torch.Tensor) -> torch.Tensor:
        em1 = self.e1(t1)
        em2 = self.e2(t2)
        em3 = self.e3(t3)

        fused, _ = self.fusion(em1, em2, em3)

        return fused


class RNNModel(nn.Module):
    def __init__(self, input_dimension: int, hidden_feature: int, name: str = "RNNModel"):
        super().__init__()
        self.name = name

        self.pat_lstm = nn.LSTM(input_size=input_dimension, hidden_size=hidden_feature, num_layers=2)

        self.seq_lstm = nn.LSTM(input_size=input_dimension, hidden_size=hidden_feature, num_layers=2)

        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(in_features=14*hidden_feature, out_features=512),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(in_features=512, out_features=128),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(in_features=128, out_features=32),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(in_features=32, out_features=16),
            nn.LeakyReLU()
        )

        self.output_layer = nn.Sequential(
            nn.Linear(in_features=16, out_features=3),
            nn.LogSoftmax(dim=1)
        )

    # Shape of pattern inputs is (Batch, Length, Dimension)
    # Shape of sequence inputs is (Batch, Length, Dimension)
    def forward(self, sequence: torch.Tensor, pattern: torch.Tensor) -> torch.Tensor:
        pattern = torch.permute(input=pattern, dims=(1, 0, 2))
        pattern, _ = self.pat_lstm(pattern)  # (L, B, D)
        pattern = torch.permute(input=pattern, dims=(1, 0, 2))  # Final shape (B, L, D)

        sequence = torch.permute(input=sequence, dims=(1, 0, 2))
        sequence, _ = self.seq_lstm(sequence)  # (L, B, D)
        sequence = torch.permute(input=sequence, dims=(1, 0, 2))  # Final shape (B, L, D)

        embed = pattern + sequence
        embed = self.flatten(embed)
        embed = self.fc(embed)
        embed = self.output_layer(embed)

        return embed
