"""
CBAM (Convolutional Block Attention Module)

Implements the CBAM attention mechanism which sequentially applies channel
attention and spatial attention to refine feature maps. This teaches the
Enhanced EfficientNet to focus on the most discriminative spatial regions
and feature channels for detecting AI-generated image artifacts.

Reference:
    Woo et al., "CBAM: Convolutional Block Attention Module"
    (ECCV 2018) - https://arxiv.org/abs/1807.06521

Author: Multi-Model Comparative Study Project
"""
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    Channel Attention Module.

    Learns to emphasize important feature channels by exploiting inter-channel
    relationships through global average and max pooling followed by a shared
    MLP (squeeze-and-excitation style).

    For AI-generated image detection, this helps the model amplify channels
    containing discriminative patterns while suppressing irrelevant features.
    """

    def __init__(self, in_channels, reduction_ratio=16):
        """
        Args:
            in_channels: Number of input feature channels
            reduction_ratio: Compression ratio for the bottleneck MLP
        """
        super(ChannelAttention, self).__init__()

        # Ensure the reduced dimension is at least 1
        reduced_channels = max(in_channels // reduction_ratio, 1)

        # Shared MLP (implemented as 1x1 convolutions for efficiency)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Channel-attended tensor of shape (B, C, H, W)
        """
        # Global Average Pooling: (B, C, H, W) -> (B, C, 1, 1)
        avg_pool = torch.mean(x, dim=(2, 3), keepdim=True)
        avg_out = self.shared_mlp(avg_pool)

        # Global Max Pooling: (B, C, H, W) -> (B, C, 1, 1)
        max_pool, _ = torch.max(x.view(x.size(0), x.size(1), -1), dim=2, keepdim=True)
        max_pool = max_pool.unsqueeze(-1)  # (B, C, 1, 1)
        max_out = self.shared_mlp(max_pool)

        # Combine and apply sigmoid
        attention = self.sigmoid(avg_out + max_out)

        return x * attention


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module.

    Learns to emphasize important spatial locations by exploiting inter-spatial
    relationships through channel-wise average and max pooling followed by a
    convolution layer.

    For AI-generated image detection, this helps the model focus on regions
    where generation artifacts concentrate (e.g., edges, textures, boundaries).
    """

    def __init__(self, kernel_size=7):
        """
        Args:
            kernel_size: Size of the convolution kernel (default: 7 for wide
                        spatial context)
        """
        super(SpatialAttention, self).__init__()

        padding = (kernel_size - 1) // 2  # Same padding

        self.conv = nn.Conv2d(
            in_channels=2,  # avg_pool + max_pool concatenated
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Spatially-attended tensor of shape (B, C, H, W)
        """
        # Channel-wise Average Pooling: (B, C, H, W) -> (B, 1, H, W)
        avg_pool = torch.mean(x, dim=1, keepdim=True)

        # Channel-wise Max Pooling: (B, C, H, W) -> (B, 1, H, W)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)

        # Concatenate along channel dimension: (B, 2, H, W)
        combined = torch.cat([avg_pool, max_pool], dim=1)

        # Apply convolution + sigmoid: (B, 1, H, W)
        attention = self.sigmoid(self.conv(combined))

        return x * attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).

    Sequentially applies Channel Attention and Spatial Attention to the input
    feature map. This dual attention mechanism refines features by:
    1. First selecting WHAT (channels) to focus on
    2. Then selecting WHERE (spatial locations) to focus on

    In the Enhanced EfficientNet-B4, this is applied to the 1792-channel
    feature maps before global average pooling, allowing the network to
    learn which spatial regions and feature channels are most discriminative
    for detecting AI-generated content.
    """

    def __init__(self, in_channels, reduction_ratio=16, spatial_kernel_size=7):
        """
        Args:
            in_channels: Number of input feature channels
            reduction_ratio: Compression ratio for channel attention MLP
            spatial_kernel_size: Kernel size for spatial attention conv
        """
        super(CBAM, self).__init__()

        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Attention-refined tensor of shape (B, C, H, W)
        """
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x
