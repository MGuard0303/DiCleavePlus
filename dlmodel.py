import math

import torch
from torch import nn


class ModelConcat(nn.Module):
    """
    DCP models which use concatenation to fuse sequence feature and pattern feature.
    """

    def __init__(self,
                 embed_feature: int,
                 linear_hidden_feature: int = 64,
                 num_attn_head: int = 8,
                 tf_dim_forward: int = 256,
                 num_tf_layer: int = 3,
                 name: str = "model-concat"
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


class ModelAFF(nn.Module):
    """
    DCP models which use AFF module to fuse sequence feature and pattern feature.
    """

    def __init__(self,
                 embed_feature: int,
                 linear_hidden_feature: int = 64,
                 num_attn_head: int = 8,
                 tf_dim_forward: int = 256,
                 num_tf_layer: int = 3,
                 name: str = "model-aff"
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


class ModelAFF_flex(nn.Module):
    def __init__(self,
                 pattern_size: int,
                 embed_feature: int,
                 linear_hidden_feature: int = 64,
                 num_attn_head: int = 8,
                 tf_dim_forward: int = 256,
                 num_tf_layer: int = 3,
                 name: str = "model-aff"
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
            nn.Linear(in_features=32, out_features=14),
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

        embed, _ = self.aff(pattern, sequence)
        embed = self.flatten(embed)
        embed = self.fc(embed)
        embed = self.output_layer(embed)

        return embed


class ModelConcat_flex(nn.Module):
    def __init__(self,
                 pattern_size: int,
                 embed_feature: int,
                 linear_hidden_feature: int = 64,
                 num_attn_head: int = 8,
                 tf_dim_forward: int = 256,
                 num_tf_layer: int = 3,
                 name: str = "model-concat"
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
            nn.Linear(in_features=32, out_features=14),
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
