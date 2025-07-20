import math

import torch
from torch import nn


"""
class ModelAff(nn.Module):
    def __init__(self,
                 embed_feature: int,
                 num_attn_head: int = 8,
                 tf_dim_forward: int = 256,
                 num_tf_layer: int = 3,
                 linear_hidden_feature: int = 64,
                 name: str = "Model-AFF"
                 ) -> None:
        super().__init__()
        self.name = name
        self.loss_function = nn.NLLLoss()

        # Layers for pattern Transformer structure.
        self.encoder_pattern = nn.TransformerEncoderLayer(d_model=embed_feature, nhead=num_attn_head,
                                                          dim_feedforward=tf_dim_forward, batch_first=True)
        self.encoder_container_pattern = nn.TransformerEncoder(encoder_layer=self.encoder_pattern,
                                                               num_layers=num_tf_layer, enable_nested_tensor=False)
        self.position_encoder_pattern = PositionEncoder(d_model=embed_feature, length=14, batch_first=True)
        self.conv_pattern = nn.Sequential(
            nn.Conv1d(in_channels=embed_feature, out_channels=embed_feature, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=embed_feature),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=embed_feature, out_channels=embed_feature, kernel_size=3, padding=1),
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
            nn.Conv1d(in_channels=embed_feature, out_channels=embed_feature, kernel_size=5, padding=2),
            nn.BatchNorm1d(num_features=embed_feature),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=embed_feature, out_channels=embed_feature, kernel_size=5, padding=2),
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

        self.aff = AttentionalFeatureFusionLayer(glo_pool_size=linear_hidden_feature, pool_type="1d")

        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.BatchNorm1d(num_features=linear_hidden_feature),
            nn.Linear(in_features=linear_hidden_feature, out_features=1024),
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

        embed, _ = self.aff(pattern, sequence)
        embed = self.flatten(embed)
        embed = self.fc(embed)
        embed = self.output_layer(embed)

        return embed
"""


"""
class ModelConcat(nn.Module):
    def __init__(self,
                 embed_feature: int,
                 num_attn_head: int = 8,
                 tf_dim_forward: int = 256,
                 num_tf_layer: int = 3,
                 linear_hidden_feature: int = 64,
                 name: str = "Model-Concat"
                 ) -> None:
        super().__init__()
        self.name = name
        self.loss_function = nn.NLLLoss()

        # Layers for pattern Transformer structure.
        self.encoder_pattern = nn.TransformerEncoderLayer(d_model=embed_feature, nhead=num_attn_head,
                                                          dim_feedforward=tf_dim_forward, batch_first=True)
        self.encoder_container_pattern = nn.TransformerEncoder(encoder_layer=self.encoder_pattern,
                                                               num_layers=num_tf_layer, enable_nested_tensor=False)
        self.position_encoder_pattern = PositionEncoder(d_model=embed_feature, length=14, batch_first=True)
        self.conv_pattern = nn.Sequential(
            nn.Conv1d(in_channels=embed_feature, out_channels=embed_feature, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=embed_feature),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=embed_feature, out_channels=embed_feature, kernel_size=3, padding=1),
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
            nn.Conv1d(in_channels=embed_feature, out_channels=embed_feature, kernel_size=5, padding=2),
            nn.BatchNorm1d(num_features=embed_feature),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=embed_feature, out_channels=embed_feature, kernel_size=5, padding=2),
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
"""


class PositionEncoder(nn.Module):
    """
    Generate positional embedding for Transformer architecture.

    The default input shape is (Length, Batch, Dimension). If batch_first=True, the default input shape is (Batch,
    Length, Dimension).

    The output shape is identical with the input shape.
    """

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


class ModelAffFlex(nn.Module):
    def __init__(self,
                 embed_feature: int,
                 pattern_size: int,
                 num_attn_head: int = 8,
                 tf_dim_forward: int = 512,
                 num_tf_layer: int = 3,
                 linear_hidden_feature: int = 128,
                 name: str = "Model-AFF-Flex"
                 ) -> None:

        super().__init__()

        self.name = name
        self.loss_function = nn.NLLLoss()

        # Layers for pattern Transformer structure.
        self.encoder_pattern = nn.TransformerEncoderLayer(d_model=embed_feature, nhead=num_attn_head,
                                                          dim_feedforward=tf_dim_forward, batch_first=True)

        self.encoder_container_pattern = nn.TransformerEncoder(encoder_layer=self.encoder_pattern,
                                                               num_layers=num_tf_layer,
                                                               norm=nn.LayerNorm(embed_feature),
                                                               enable_nested_tensor=False)

        self.position_encoder_pattern = PositionEncoder(d_model=embed_feature, length=pattern_size, batch_first=True)

        self.conv_pattern_1 = nn.Sequential(
            nn.Conv1d(in_channels=embed_feature, out_channels=embed_feature, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=embed_feature),
            nn.LeakyReLU()
        )

        self.conv_pattern_2 = nn.Sequential(
            nn.Conv1d(in_channels=embed_feature, out_channels=embed_feature, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=embed_feature),
            nn.LeakyReLU()
        )

        # Layers for sequence Transformer structure.
        self.encoder_sequence = nn.TransformerEncoderLayer(d_model=embed_feature, nhead=num_attn_head,
                                                           dim_feedforward=tf_dim_forward, batch_first=True)

        self.encoder_container_sequence = nn.TransformerEncoder(encoder_layer=self.encoder_sequence,
                                                                num_layers=num_tf_layer,
                                                                norm=nn.LayerNorm(embed_feature),
                                                                enable_nested_tensor=False)

        self.position_encoder_sequence = PositionEncoder(d_model=embed_feature, length=200, batch_first=True)

        self.conv_sequence_1 = nn.Sequential(
            nn.Conv1d(in_channels=embed_feature, out_channels=embed_feature, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=embed_feature),
            nn.LeakyReLU()
        )

        self.conv_sequence_2 = nn.Sequential(
            nn.Conv1d(in_channels=embed_feature, out_channels=embed_feature, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=embed_feature),
            nn.LeakyReLU()
        )

        self.identity = nn.Sequential()  # Resnet shortcut.

        self.linear_pattern = nn.Sequential(
            nn.Linear(in_features=pattern_size * embed_feature, out_features=linear_hidden_feature),
            nn.LeakyReLU(),
            nn.Dropout()
        )
        self.linear_sequence = nn.Sequential(
            nn.Linear(in_features=200 * embed_feature, out_features=linear_hidden_feature),
            nn.LeakyReLU(),
            nn.Dropout()
        )

        self.aff = AttentionalFeatureFusionLayer(glo_pool_size=linear_hidden_feature, pool_type="1d")

        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.BatchNorm1d(num_features=linear_hidden_feature),
            nn.Linear(in_features=linear_hidden_feature, out_features=1024),
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
            nn.Linear(in_features=32, out_features=pattern_size),
            nn.LogSoftmax(dim=1)
        )

    # Shape of inputs is (Batch, Length, Dimension).
    def forward(self, sequence: torch.Tensor, pattern: torch.Tensor) -> torch.Tensor:
        # Forward process of pattern feature.
        pattern = self.position_encoder_pattern(pattern)  # Output size: (B, L, D)
        pattern = self.encoder_container_pattern(pattern)  # Output size: (B, L, D)
        pattern_identity = self.identity(pattern)
        pattern = torch.permute(input=pattern, dims=(0, 2, 1))  # (B, L, D) -> (B, D, L)
        pattern = self.conv_pattern_1(pattern)  # Output size: (B, D, L)
        pattern = torch.permute(input=pattern, dims=(0, 2, 1))  # (B, D, L) -> (B, L, D)
        pattern += pattern_identity  # Output size: (B, L, D)
        pattern = torch.permute(input=pattern, dims=(0, 2, 1))  # (B, L, D) -> (B, D, L)
        pattern = self.conv_pattern_2(pattern)  # Output size: (B, D, L)
        pattern = torch.permute(input=pattern, dims=(0, 2, 1))  # (B, D, L) -> (B, L, D)
        pattern += pattern_identity  # Output size: (B, L, D)
        pattern = self.flatten(pattern)
        pattern = self.linear_pattern(pattern)

        # Forward process of sequence feature.
        sequence = self.position_encoder_sequence(sequence)  # Output size: (B, L, D)
        sequence = self.encoder_container_sequence(sequence)  # Output size: (B, L, D)
        sequence_identity = self.identity(sequence)
        sequence = torch.permute(input=sequence, dims=(0, 2, 1))  # (B, L, D) -> (B, D, L)
        sequence = self.conv_sequence_1(sequence)  # Output size: (B, D, L)
        sequence = torch.permute(input=sequence, dims=(0, 2, 1))  # (B, D, L) -> (B, L, D)
        sequence += sequence_identity  # Output size: (B, L, D)
        sequence = torch.permute(input=sequence, dims=(0, 2, 1))  # (B, L, D) -> (B, D, L)
        sequence = self.conv_sequence_2(sequence)  # Output size: (B, D, L)
        sequence = torch.permute(input=sequence, dims=(0, 2, 1))  # (B, D, L) -> (B, L, D)
        sequence += sequence_identity  # Output size: (B, L, D)
        sequence = self.flatten(sequence)
        sequence = self.linear_sequence(sequence)

        embed, _ = self.aff(pattern, sequence)
        embed = self.flatten(embed)
        embed = self.fc(embed)
        embed = self.output_layer(embed)

        return embed


class ModelConcatFlex(nn.Module):
    def __init__(self,
                 embed_feature: int,
                 pattern_size: int,
                 num_attn_head: int = 8,
                 tf_dim_forward: int = 512,
                 num_tf_layer: int = 3,
                 linear_hidden_feature: int = 128,
                 name: str = "Model-Concat-Flex"
                 ) -> None:

        super().__init__()

        self.name = name
        self.loss_function = nn.NLLLoss()

        # Layers for pattern Transformer structure.
        self.encoder_pattern = nn.TransformerEncoderLayer(d_model=embed_feature, nhead=num_attn_head,
                                                          dim_feedforward=tf_dim_forward, batch_first=True)

        self.encoder_container_pattern = nn.TransformerEncoder(encoder_layer=self.encoder_pattern,
                                                               num_layers=num_tf_layer,
                                                               norm=nn.LayerNorm(embed_feature),
                                                               enable_nested_tensor=False)

        self.position_encoder_pattern = PositionEncoder(d_model=embed_feature, length=pattern_size, batch_first=True)

        self.conv_pattern_1 = nn.Sequential(
            nn.Conv1d(in_channels=embed_feature, out_channels=embed_feature, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=embed_feature),
            nn.LeakyReLU()
        )

        self.conv_pattern_2 = nn.Sequential(
            nn.Conv1d(in_channels=embed_feature, out_channels=embed_feature, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=embed_feature),
            nn.LeakyReLU()
        )

        # Layers for sequence Transformer structure
        self.encoder_sequence = nn.TransformerEncoderLayer(d_model=embed_feature, nhead=num_attn_head,
                                                           dim_feedforward=tf_dim_forward, batch_first=True)

        self.encoder_container_sequence = nn.TransformerEncoder(encoder_layer=self.encoder_sequence,
                                                                num_layers=num_tf_layer,
                                                                norm=nn.LayerNorm(embed_feature),
                                                                enable_nested_tensor=False)

        self.position_encoder_sequence = PositionEncoder(d_model=embed_feature, length=200, batch_first=True)

        self.conv_sequence_1 = nn.Sequential(
            nn.Conv1d(in_channels=embed_feature, out_channels=embed_feature, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=embed_feature),
            nn.LeakyReLU()
        )

        self.conv_sequence_2 = nn.Sequential(
            nn.Conv1d(in_channels=embed_feature, out_channels=embed_feature, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=embed_feature),
            nn.LeakyReLU()
        )

        self.identity = nn.Sequential()  # Resnet short-cut for convolution structure

        self.linear_pattern = nn.Sequential(
            nn.Linear(in_features=pattern_size * embed_feature, out_features=linear_hidden_feature),
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
            nn.Linear(in_features=32, out_features=pattern_size),
            nn.LogSoftmax(dim=1)
        )

    # Shape of inputs is (Batch, Length, Dimension)
    def forward(self, sequence: torch.Tensor, pattern: torch.Tensor) -> torch.Tensor:
        # Forward process of pattern feature.
        pattern = self.position_encoder_pattern(pattern)  # Output size: (B, L, D)
        pattern = self.encoder_container_pattern(pattern)  # Output size: (B, L, D)
        pattern_identity = self.identity(pattern)
        pattern = torch.permute(input=pattern, dims=(0, 2, 1))  # (B, L, D) -> (B, D, L)
        pattern = self.conv_pattern_1(pattern)  # Output size: (B, D, L)
        pattern = torch.permute(input=pattern, dims=(0, 2, 1))  # (B, D, L) -> (B, L, D)
        pattern += pattern_identity  # Output size: (B, L, D)
        pattern = torch.permute(input=pattern, dims=(0, 2, 1))  # (B, L, D) -> (B, D, L)
        pattern = self.conv_pattern_2(pattern)  # Output size: (B, D, L)
        pattern = torch.permute(input=pattern, dims=(0, 2, 1))  # (B, D, L) -> (B, L, D)
        pattern += pattern_identity  # Output size: (B, L, D)
        pattern = self.flatten(pattern)
        pattern = self.linear_pattern(pattern)

        # Forward process of sequence feature.
        sequence = self.position_encoder_sequence(sequence)  # Output size: (B, L, D)
        sequence = self.encoder_container_sequence(sequence)  # Output size: (B, L, D)
        sequence_identity = self.identity(sequence)
        sequence = torch.permute(input=sequence, dims=(0, 2, 1))  # (B, L, D) -> (B, D, L)
        sequence = self.conv_sequence_1(sequence)  # Output size: (B, D, L)
        sequence = torch.permute(input=sequence, dims=(0, 2, 1))  # (B, D, L) -> (B, L, D)
        sequence += sequence_identity  # Output size: (B, L, D)
        sequence = torch.permute(input=sequence, dims=(0, 2, 1))  # (B, L, D) -> (B, D, L)
        sequence = self.conv_sequence_2(sequence)  # Output size: (B, D, L)
        sequence = torch.permute(input=sequence, dims=(0, 2, 1))  # (B, D, L) -> (B, L, D)
        sequence += sequence_identity  # Output size: (B, L, D)
        sequence = self.flatten(sequence)
        sequence = self.linear_sequence(sequence)

        embed = torch.cat((pattern, sequence), dim=1)
        embed = self.flatten(embed)
        embed = self.fc(embed)
        embed = self.output_layer(embed)

        return embed


class DC(nn.Module):
    def __init__(self, pattern_size):

        super().__init__()

        self.name = "DC"

        self.identity = nn.Sequential()

        self.convs = nn.ModuleDict({
            "same_1": nn.Conv1d(in_channels=13, out_channels=13, kernel_size=3, padding=1),
            "blk_1_1": nn.Conv1d(in_channels=13, out_channels=16, kernel_size=3, padding=2),
            "blk_1_2": nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),

            "same_2": nn.Conv1d(in_channels=2, out_channels=2, kernel_size=3, padding=1),
            "blk_2_1": nn.Conv1d(in_channels=2, out_channels=8, kernel_size=3, padding=1),
            "blk_2_2": nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            "blk_2_3": nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            "blk_2_4": nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        })

        self.bns = nn.ModuleDict({
            "same_1": nn.BatchNorm1d(num_features=13),
            "blk_1_1": nn.BatchNorm1d(num_features=16),
            "blk_1_2": nn.BatchNorm1d(num_features=32),

            "same_2": nn.BatchNorm1d(num_features=2),
            "blk_2_1": nn.BatchNorm1d(num_features=8),
            "blk_2_2": nn.BatchNorm1d(num_features=16),
            "blk_2_3": nn.BatchNorm1d(num_features=32),
            "blk_2_4": nn.BatchNorm1d(num_features=64)
        })

        self.pool = nn.MaxPool1d(2)

        self.relu = nn.LeakyReLU()

        self.fc1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(128, 64),
            nn.LeakyReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.LeakyReLU()
            )

        self.mul_unit = nn.Sequential(
            nn.Linear(32, pattern_size),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x, embed):
        # Forward propagate in CU1 of module_1
        x_identity = self.identity(x)

        for i in range(3):
            x = self.convs["same_1"](x)
            x = self.bns["same_1"](x)
            x = self.relu(x)
            x += x_identity

        # Forward propagate in CU2 of module_1
        x = self.convs["blk_1_1"](x)
        x = self.bns["blk_1_1"](x)
        x = self.relu(x)
        x = self.convs["blk_1_2"](x)
        x = self.bns["blk_1_2"](x)
        x = self.relu(x)

        # Forward propagate in FC layers of module_1
        x = torch.reshape(x, (x.size(0), -1))
        x = self.fc1(x)

        # Concatenate sequence vector with secondary structure embedding
        x = torch.unsqueeze(x, dim=1)
        embed = torch.unsqueeze(embed, dim=1)
        x_embed = torch.cat((x, embed), dim=1)

        # Forward propagate in CU1 of module_2
        x_embed_identity = self.identity(x_embed)

        for i in range(3):
            x_embed = self.convs["same_2"](x_embed)
            x_embed = self.bns["same_2"](x_embed)
            x_embed = self.relu(x_embed)
            x_embed += x_embed_identity

        x_embed = self.convs["blk_2_1"](x_embed)
        x_embed = self.bns["blk_2_1"](x_embed)
        x_embed = self.relu(x_embed)

        x_embed = self.convs["blk_2_2"](x_embed)
        x_embed = self.bns["blk_2_2"](x_embed)
        x_embed = self.pool(x_embed)
        x_embed = self.relu(x_embed)

        x_embed = self.convs["blk_2_3"](x_embed)
        x_embed = self.bns["blk_2_3"](x_embed)
        x_embed = self.pool(x_embed)
        x_embed = self.relu(x_embed)

        # Forward propagate in FC layers of module_2
        x_embed = torch.reshape(x_embed, (x_embed.size(0), -1))
        x_embed = self.fc2(x_embed)

        x_embed = self.mul_unit(x_embed)

        return x_embed


class AblationModelMLP(nn.Module):
    def __init__(self,
                 embed_feature: int,
                 pattern_size: int,
                 linear_hidden_feature: int = 128,
                 name: str = "Ablation-Model-MLP"
                 ) -> None:

        super().__init__()

        self.name = name
        self.loss_function = nn.NLLLoss()

        # MLP layers for pattern feature.
        self.mlp_pattern = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=pattern_size * embed_feature, out_features=pattern_size * embed_feature),
            nn.LeakyReLU(),
            nn.Linear(in_features=pattern_size * embed_feature, out_features=pattern_size * embed_feature),
            nn.LeakyReLU(),
            nn.Linear(in_features=pattern_size * embed_feature, out_features=pattern_size * embed_feature)
        )

        # MLP layers for sequence feature.
        self.mlp_sequence = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=200 * embed_feature, out_features=200 * embed_feature),
            nn.LeakyReLU(),
            nn.Linear(in_features=200 * embed_feature, out_features=200 * embed_feature),
            nn.LeakyReLU(),
            nn.Linear(in_features=200 * embed_feature, out_features=200 * embed_feature)
        )

        self.linear_pattern = nn.Sequential(
            nn.Linear(in_features=pattern_size * embed_feature, out_features=linear_hidden_feature),
            nn.LeakyReLU(),
            nn.Dropout()
        )
        self.linear_sequence = nn.Sequential(
            nn.Linear(in_features=200 * embed_feature, out_features=linear_hidden_feature),
            nn.LeakyReLU(),
            nn.Dropout()
        )

        self.aff = AttentionalFeatureFusionLayer(glo_pool_size=linear_hidden_feature, pool_type="1d")

        self.flatten = nn.Flatten()

        self.fc = nn.Sequential(
            nn.BatchNorm1d(num_features=linear_hidden_feature),
            nn.Linear(in_features=linear_hidden_feature, out_features=1024),
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
            nn.Linear(in_features=32, out_features=pattern_size),
            nn.LogSoftmax(dim=1)
        )

    # Shape of inputs is (Batch, Length, Dimension)
    def forward(self, sequence: torch.Tensor, pattern: torch.Tensor) -> torch.Tensor:
        # Forward process of pattern feature.
        pattern = self.mlp_pattern(pattern)
        pattern = self.linear_pattern(pattern)

        # Forward process of sequence feature.
        sequence = self.mlp_sequence(sequence)
        sequence = self.linear_sequence(sequence)

        embed, _ = self.aff(pattern, sequence)
        embed = self.flatten(embed)
        embed = self.fc(embed)
        embed = self.output_layer(embed)

        return embed


class AblationModelCNN(nn.Module):
    def __init__(self,
                 embed_feature: int,
                 pattern_size: int,
                 linear_hidden_feature: int = 128,
                 name: str = "Ablation-Model-CNN"
                 ) -> None:

        super().__init__()

        self.name = name
        self.loss_function = nn.NLLLoss()

        self.cnn_extractor_sequence = nn.Sequential(
            nn.Conv1d(in_channels=embed_feature, out_channels=embed_feature, kernel_size=3, padding=1),
            nn.Conv1d(in_channels=embed_feature, out_channels=embed_feature, kernel_size=3, padding=1),
            nn.Conv1d(in_channels=embed_feature, out_channels=embed_feature, kernel_size=3, padding=1),
        )

        self.cnn_extractor_pattern = nn.Sequential(
            nn.Conv1d(in_channels=embed_feature, out_channels=embed_feature, kernel_size=3, padding=1),
            nn.Conv1d(in_channels=embed_feature, out_channels=embed_feature, kernel_size=3, padding=1),
            nn.Conv1d(in_channels=embed_feature, out_channels=embed_feature, kernel_size=3, padding=1),
        )

        self.position_encoder_pattern = PositionEncoder(d_model=embed_feature, length=pattern_size, batch_first=True)
        self.position_encoder_sequence = PositionEncoder(d_model=embed_feature, length=200, batch_first=True)

        self.linear_pattern = nn.Sequential(
            nn.Linear(in_features=pattern_size * embed_feature, out_features=linear_hidden_feature),
            nn.LeakyReLU(),
            nn.Dropout()
        )
        self.linear_sequence = nn.Sequential(
            nn.Linear(in_features=200 * embed_feature, out_features=linear_hidden_feature),
            nn.LeakyReLU(),
            nn.Dropout()
        )

        self.aff = AttentionalFeatureFusionLayer(glo_pool_size=linear_hidden_feature, pool_type="1d")

        self.flatten = nn.Flatten()

        self.fc = nn.Sequential(
            nn.BatchNorm1d(num_features=linear_hidden_feature),
            nn.Linear(in_features=linear_hidden_feature, out_features=1024),
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
            nn.Linear(in_features=32, out_features=pattern_size),
            nn.LogSoftmax(dim=1)
        )

    # Shape of inputs is (Batch, Length, Dimension)
    def forward(self, sequence: torch.Tensor, pattern: torch.Tensor) -> torch.Tensor:
        # Forward process of pattern feature.
        pattern = self.position_encoder_pattern(pattern)  # Output size: (B, L, D)
        pattern = torch.permute(input=pattern, dims=(0, 2, 1))  # (B, L, D) -> (B, D, L)
        pattern = self.cnn_extractor_pattern(pattern)  # Output size: (B, D, L)
        pattern = torch.permute(input=pattern, dims=(0, 2, 1))  # (B, D, L) -> (B, L, D)
        pattern = self.flatten(pattern)
        pattern = self.linear_pattern(pattern)

        # Forward process of sequence feature.
        sequence = self.position_encoder_sequence(sequence)  # Output size: (B, L, D)
        sequence = torch.permute(input=sequence, dims=(0, 2, 1))  # (B, L, D) -> (B, D, L)
        sequence = self.cnn_extractor_sequence(sequence)  # Output size: (B, D, L)
        sequence = torch.permute(input=sequence, dims=(0, 2, 1))  # (B, D, L) -> (B, L, D)
        sequence = self.flatten(sequence)
        sequence = self.linear_sequence(sequence)

        embed, _ = self.aff(pattern, sequence)
        embed = self.flatten(embed)
        embed = self.fc(embed)
        embed = self.output_layer(embed)

        return embed


class AblationModelRNN(nn.Module):
    def __init__(self,
                 embed_feature: int,
                 pattern_size: int,
                 linear_hidden_feature: int = 128,
                 name: str = "Ablation-Model-RNN"
                 ) -> None:

        super().__init__()

        self.name = name
        self.loss_function = nn.NLLLoss()

        self.rnn_extractor_pattern = nn.RNN(input_size=embed_feature, hidden_size=embed_feature, num_layers=3,
                                            batch_first=True)

        self.rnn_extractor_sequence = nn.RNN(input_size=embed_feature, hidden_size=embed_feature, num_layers=3,
                                             batch_first=True)

        self.linear_pattern = nn.Sequential(
            nn.Linear(in_features=3 * embed_feature, out_features=linear_hidden_feature),
            nn.LeakyReLU(),
            nn.Dropout()
        )
        self.linear_sequence = nn.Sequential(
            nn.Linear(in_features=3 * embed_feature, out_features=linear_hidden_feature),
            nn.LeakyReLU(),
            nn.Dropout()
        )

        self.aff = AttentionalFeatureFusionLayer(glo_pool_size=linear_hidden_feature, pool_type="1d")

        self.flatten = nn.Flatten()

        self.fc = nn.Sequential(
            nn.BatchNorm1d(num_features=linear_hidden_feature),
            nn.Linear(in_features=linear_hidden_feature, out_features=1024),
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
            nn.Linear(in_features=32, out_features=pattern_size),
            nn.LogSoftmax(dim=1)
        )

    # Shape of inputs is (Batch, Length, Dimension)
    def forward(self, sequence: torch.Tensor, pattern: torch.Tensor) -> torch.Tensor:
        # Forward process of pattern feature.
        patt_out, patt_hn = self.rnn_extractor_pattern(pattern)
        patt_hn = torch.permute(patt_hn, (1, 0, 2))
        patt_hn = self.flatten(patt_hn)
        patt_hn = self.linear_pattern(patt_hn)

        # Forward process of sequence feature.
        seq_out, seq_hn = self.rnn_extractor_sequence(sequence)
        seq_hn = torch.permute(seq_hn, (1, 0, 2))
        seq_hn = self.flatten(seq_hn)
        seq_hn = self.linear_sequence(seq_hn)

        embed, _ = self.aff(patt_hn, seq_hn)
        embed = self.flatten(embed)
        embed = self.fc(embed)
        embed = self.output_layer(embed)

        return embed
