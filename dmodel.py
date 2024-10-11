import math

import torch
from torch import nn


class TFModel(nn.Module):
    def __init__(self, embed_feature: int, linear_hidden_feature: int = 256, num_attn_head: int = 8,
                 tf_dim_forward: int = 512, num_tf_layer: int = 6, seq_conv_kernel_size: int = 5,
                 patt_conv_kernel_size: int = 3, name: str = "TFModel"):
        super().__init__()
        self.name = name

        # Calculate padding number for different kernel size.
        seq_padding = int((seq_conv_kernel_size - 1) / 2)
        patt_padding = int((patt_conv_kernel_size - 1) / 2)

        # Layers for pattern Transformer structure.
        self.encoder_pattern = nn.TransformerEncoderLayer(d_model=embed_feature, nhead=num_attn_head,
                                                          dim_feedforward=tf_dim_forward, batch_first=True)
        self.encoder_container_pattern = nn.TransformerEncoder(encoder_layer=self.encoder_pattern,
                                                               num_layers=num_tf_layer, enable_nested_tensor=False)
        self.position_encoder_pattern = PositionEncoder(d_model=embed_feature, length=14, batch_first=True)
        self.conv_pattern = nn.Sequential(
            nn.Conv1d(in_channels=embed_feature, out_channels=embed_feature, kernel_size=patt_conv_kernel_size,
                      padding=patt_padding),
            nn.BatchNorm1d(num_features=embed_feature),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=embed_feature, out_channels=embed_feature, kernel_size=patt_conv_kernel_size,
                      padding=patt_padding),
            nn.BatchNorm1d(num_features=embed_feature),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=embed_feature, out_channels=embed_feature, kernel_size=patt_conv_kernel_size,
                      padding=patt_padding),
            nn.BatchNorm1d(num_features=embed_feature),
            nn.LeakyReLU()
        )

        # Layers for sequence Transformer structure
        self.encoder_sequence = nn.TransformerEncoderLayer(d_model=embed_feature, nhead=num_attn_head,
                                                           dim_feedforward=tf_dim_forward, batch_first=True)
        self.encoder_container_sequence = nn.TransformerEncoder(encoder_layer=self.encoder_sequence,
                                                                num_layers=num_tf_layer, enable_nested_tensor=False)
        self.position_encoder_sequence = PositionEncoder(d_model=embed_feature, length=200, batch_first=True)
        self.conv_sequence = nn.Sequential(
            nn.Conv1d(in_channels=embed_feature, out_channels=embed_feature, kernel_size=seq_conv_kernel_size,
                      padding=seq_padding),
            nn.BatchNorm1d(num_features=embed_feature),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=embed_feature, out_channels=embed_feature, kernel_size=seq_conv_kernel_size,
                      padding=seq_padding),
            nn.BatchNorm1d(num_features=embed_feature),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=embed_feature, out_channels=embed_feature, kernel_size=seq_conv_kernel_size,
                      padding=seq_padding),
            nn.BatchNorm1d(num_features=embed_feature),
            nn.LeakyReLU()
        )

        self.identity = nn.Sequential()  # Resnet short-cut for convolution structure

        self.linear_pattern = nn.Sequential(
            nn.Linear(in_features=14 * embed_feature, out_features=linear_hidden_feature),
            nn.LeakyReLU(),
            nn.Dropout()
        )
        self.linear_sequence = nn.Sequential(
            nn.Linear(in_features=200 * embed_feature, out_features=linear_hidden_feature),
            nn.LeakyReLU(),
            nn.Dropout()
        )

        self.union_fusion = AttentionalFeatureFusionLayer(glo_pool_size=linear_hidden_feature, pool_type="1d")

        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.BatchNorm1d(num_features=linear_hidden_feature),
            nn.Linear(in_features=linear_hidden_feature, out_features=2048),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(in_features=2048, out_features=512),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(in_features=512, out_features=128),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(in_features=128, out_features=32),
            nn.LeakyReLU(),
        )

        self.output_layer = nn.Sequential(
            nn.Linear(in_features=32, out_features=14),
            nn.LogSoftmax(dim=1)
        )

    # Shape of inputs is (Batch, Length, Dimension)
    def forward(self, sequence: torch.Tensor, pattern: torch.Tensor) -> torch.Tensor:
        # Forward process of pattern feature.
        pattern_identity = self.identity(pattern)

        pattern = self.position_encoder_pattern(pattern)  # Output size: (B, L, D)
        pattern = self.encoder_container_pattern(pattern)  # Output size: (B, L, D)
        pattern = torch.permute(input=pattern, dims=(0, 2, 1))
        pattern = self.conv_pattern(pattern)  # Output size: (B, D, L)
        pattern = torch.permute(input=pattern, dims=(0, 2, 1))
        pattern += pattern_identity  # Output size: (B, L, D)
        pattern = self.flatten(pattern)
        pattern = self.linear_pattern(pattern)

        # Forward process of sequence feature.
        sequence_identity = self.identity(sequence)

        sequence = self.position_encoder_sequence(sequence)  # Output size: (B, L, D)
        sequence = self.encoder_container_sequence(sequence)  # Output size: (B, L, D)
        sequence = torch.permute(input=sequence, dims=(0, 2, 1))
        sequence = self.conv_sequence(sequence)  # Output size: (B, D, L)
        sequence = torch.permute(input=sequence, dims=(0, 2, 1))
        sequence += sequence_identity  # Output size: (B, L, D)
        sequence = self.flatten(sequence)
        sequence = self.linear_sequence(sequence)

        embed, _ = self.union_fusion(pattern, sequence)
        embed = self.flatten(embed)
        embed = self.fc(embed)
        embed = self.output_layer(embed)

        return embed


# Use concatenation instead of advanced feature fusion.
class TFModelMini(nn.Module):
    def __init__(self, embed_feature: int, linear_hidden_feature: int = 256, num_attn_head: int = 8,
                 tf_dim_forward: int = 512, num_tf_layer: int = 6, seq_conv_kernel_size: int = 5,
                 patt_conv_kernel_size: int = 3, name: str = "TFModel-mini"):
        super().__init__()
        self.name = name

        # Calculate padding number for different kernel size.
        seq_padding = int((seq_conv_kernel_size - 1) / 2)
        patt_padding = int((patt_conv_kernel_size - 1) / 2)

        # Layers for pattern Transformer structure.
        self.encoder_pattern = nn.TransformerEncoderLayer(d_model=embed_feature, nhead=num_attn_head,
                                                          dim_feedforward=tf_dim_forward, batch_first=True)
        self.encoder_container_pattern = nn.TransformerEncoder(encoder_layer=self.encoder_pattern,
                                                               num_layers=num_tf_layer, enable_nested_tensor=False)
        self.position_encoder_pattern = PositionEncoder(d_model=embed_feature, length=14, batch_first=True)
        self.conv_pattern = nn.Sequential(
            nn.Conv1d(in_channels=embed_feature, out_channels=embed_feature, kernel_size=patt_conv_kernel_size,
                      padding=patt_padding),
            nn.BatchNorm1d(num_features=embed_feature),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=embed_feature, out_channels=embed_feature, kernel_size=patt_conv_kernel_size,
                      padding=patt_padding),
            nn.BatchNorm1d(num_features=embed_feature),
            nn.LeakyReLU()
        )

        # Layers for sequence Transformer structure
        self.encoder_sequence = nn.TransformerEncoderLayer(d_model=embed_feature, nhead=num_attn_head,
                                                           dim_feedforward=tf_dim_forward, batch_first=True)
        self.encoder_container_sequence = nn.TransformerEncoder(encoder_layer=self.encoder_sequence,
                                                                num_layers=num_tf_layer, enable_nested_tensor=False)
        self.position_encoder_sequence = PositionEncoder(d_model=embed_feature, length=200, batch_first=True)
        self.conv_sequence = nn.Sequential(
            nn.Conv1d(in_channels=embed_feature, out_channels=embed_feature, kernel_size=seq_conv_kernel_size,
                      padding=seq_padding),
            nn.BatchNorm1d(num_features=embed_feature),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=embed_feature, out_channels=embed_feature, kernel_size=seq_conv_kernel_size,
                      padding=seq_padding),
            nn.BatchNorm1d(num_features=embed_feature),
            nn.LeakyReLU()
        )

        self.identity = nn.Sequential()  # Resnet short-cut for convolution structure

        self.linear_pattern = nn.Sequential(
            nn.Linear(in_features=14 * embed_feature, out_features=linear_hidden_feature),
            nn.LeakyReLU(),
            nn.Dropout()
        )
        self.linear_sequence = nn.Sequential(
            nn.Linear(in_features=200 * embed_feature, out_features=linear_hidden_feature),
            nn.LeakyReLU(),
            nn.Dropout()
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.BatchNorm1d(num_features=2 * linear_hidden_feature),
            nn.Linear(in_features=2 * linear_hidden_feature, out_features=1024),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(in_features=1024, out_features=256),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(in_features=256, out_features=64),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(in_features=64, out_features=32),
            nn.LeakyReLU(),
        )

        self.output_layer = nn.Sequential(
            nn.Linear(in_features=32, out_features=14),
            nn.LogSoftmax(dim=1)
        )

    # Shape of inputs is (Batch, Length, Dimension)
    def forward(self, sequence: torch.Tensor, pattern: torch.Tensor) -> torch.Tensor:
        # Forward process of pattern feature.
        pattern_identity = self.identity(pattern)

        pattern = self.position_encoder_pattern(pattern)  # Output size: (B, L, D)
        pattern = self.encoder_container_pattern(pattern)  # Output size: (B, L, D)
        pattern = torch.permute(input=pattern, dims=(0, 2, 1))
        pattern = self.conv_pattern(pattern)  # Output size: (B, D, L)
        pattern = torch.permute(input=pattern, dims=(0, 2, 1))
        pattern += pattern_identity  # Output size: (B, L, D)
        pattern = self.flatten(pattern)
        pattern = self.linear_pattern(pattern)

        # Forward process of sequence feature.
        sequence_identity = self.identity(sequence)

        sequence = self.position_encoder_sequence(sequence)  # Output size: (B, L, D)
        sequence = self.encoder_container_sequence(sequence)  # Output size: (B, L, D)
        sequence = torch.permute(input=sequence, dims=(0, 2, 1))
        sequence = self.conv_sequence(sequence)  # Output size: (B, D, L)
        sequence = torch.permute(input=sequence, dims=(0, 2, 1))
        sequence += sequence_identity  # Output size: (B, L, D)
        sequence = self.flatten(sequence)
        sequence = self.linear_sequence(sequence)

        embed = torch.cat((pattern, sequence), dim=1)
        embed = self.flatten(embed)
        embed = self.fc(embed)
        embed = self.output_layer(embed)

        return embed


# The default input shape is (Length, Batch, Dimension)
# If batch_first=True, the default input shape is (Batch, Length, Dimension)
# Output shape is identity with input shape
class PositionEncoder(nn.Module):
    def __init__(self, d_model: int, length: int, n: float = 10000.0, batch_first: bool = False):
        super().__init__()
        self.batch_first = batch_first
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
        if self.batch_first:
            x = torch.permute(input=x, dims=(1, 0, 2))
            x = x + self.pe[:x.size(0)]
            x = torch.permute(input=x, dims=(1, 0, 2))
        else:
            x = x + self.pe[:x.size(0)]

        return x


# Fuse different kmers embeddings using attention
# softmax_dim should be the length dimension of input embedding
# Assume the input size is (Batch, Length, Dimension)
class SelfWeightFusionLayer(nn.Module):
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


"""
# Used gates similar in LSTM to control the contribution of each feature
class GatedFusionLayer(nn.Module):
    def __init__(self, input_feature: int, fused_feature: int):
        super().__init__()
        self.gate = nn.Linear(input_feature, fused_feature)
        self.sigmoid = nn.Sigmoid()
        self.fusion = nn.Linear(fused_feature, fused_feature)

    def forward(self, *args: torch.Tensor) -> tuple:
        fused = torch.zeros_like(args[0])
        gs = []

        for t in args:
            g = self.sigmoid(self.gate(t))
            fused += g * t
            gs.append(g)

        fused = self.fusion(fused)

        return fused, gs
"""


# Use Attentional Feature Fusion (AFF) module to fuse two feature.
class AttentionalFeatureFusionLayer(nn.Module):
    def __init__(self, glo_pool_size: int | tuple, pool_type: str):
        super().__init__()
        if pool_type == "1d":
            self.global_avg_pool = nn.AvgPool1d(kernel_size=glo_pool_size)
            self.pw_pool = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1)
            self.bn = nn.BatchNorm1d(num_features=1)
        elif pool_type == "2d":
            self.global_avg_pool = nn.AvgPool2d(kernel_size=glo_pool_size)
            self.pw_pool = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1))
            self.bn = nn.BatchNorm2d(num_features=1)

        self.relu = nn.ReLU()

        self.sigmoid = nn.Sigmoid()

    def forward(self, t1: torch.Tensor, t2: torch.Tensor):
        # Similar to AFF, change the size of input sensor as (Batch, Channel, Dimension(s)).
        t1 = torch.unsqueeze(t1, 1)
        t2 = torch.unsqueeze(t2, 1)
        temp = t1 + t2

        # Global features
        u_g = self.global_avg_pool(temp)
        u_g = self.relu(u_g)
        u_g = self.bn(u_g)

        # Local features
        u_l = self.pw_pool(temp)
        u_l = self.relu(u_l)
        u_l = self.bn(u_l)

        u = u_g + u_l
        w = self.sigmoid(u)
        fused = w * t1 + (1 - w) * t2

        w = torch.squeeze(w)
        fused = torch.squeeze(fused)

        return fused, w


# Embed and fuse sequence and secondary kmer raw tensors into sequence input and pattern input.
class EmbeddingLayer(nn.Module):
    def __init__(self, embed_dim: int, softmax_dim: int, is_secondary_structure: bool = False,
                 is_simple: bool = False):
        super().__init__()
        self.is_simple = is_simple

        if is_secondary_structure:
            self.e1 = nn.Embedding(num_embeddings=4, embedding_dim=embed_dim, padding_idx=0)
            self.e2 = nn.Embedding(num_embeddings=13, embedding_dim=embed_dim, padding_idx=0)
            self.e3 = nn.Embedding(num_embeddings=40, embedding_dim=embed_dim, padding_idx=0)

        else:
            self.e1 = nn.Embedding(num_embeddings=5, embedding_dim=embed_dim, padding_idx=0)
            self.e2 = nn.Embedding(num_embeddings=21, embedding_dim=embed_dim, padding_idx=0)
            self.e3 = nn.Embedding(num_embeddings=85, embedding_dim=embed_dim, padding_idx=0)

        if not self.is_simple:
            self.fusion = SelfWeightFusionLayer(hidden_feature=embed_dim, softmax_dim=softmax_dim)

    def forward(self, t1: torch.Tensor, t2: torch.Tensor, t3: torch.Tensor) -> torch.Tensor:
        em1 = self.e1(t1)
        em2 = self.e2(t2)
        em3 = self.e3(t3)

        if not self.is_simple:
            fused, _ = self.fusion(em1, em2, em3)
        else:
            fused = torch.cat((em1, em2, em3), dim=2)

        return fused


"""
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
"""
