"""
EfficientNet-B4 Wrappers - Vanilla, CBAM-Enhanced, & Spatial-Only

Provides three architectures for the multi-model comparative study:
  - VanillaEfficientNetB4:       Standard EfficientNet-B4 (ImageNet pretrained)
  - EnhancedEfficientNetB4:      EfficientNet-B4 + CBAM spatial-channel attention
  - SpatialOnlyEfficientNetB4:   EfficientNet-B4 + Spatial-Only attention
                                  (avoids SE/channel-attention redundancy)

All models expose a `last_linear` attribute for compatibility with the
TransferModel interface.

Backbone: Luke Melas's EfficientNet-PyTorch (v0.7.1)
    https://github.com/lukemelas/EfficientNet-PyTorch

Reference:
    Tan & Le, "EfficientNet: Rethinking Model Scaling for CNNs"
    (ICML 2019) - https://arxiv.org/abs/1905.11946

Author: Multi-Model Comparative Study Project
"""
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from network.attention import CBAM, SpatialAttention


class VanillaEfficientNetB4(nn.Module):
    """
    Vanilla EfficientNet-B4 wrapper for binary classification.

    Loads ImageNet-pretrained weights and replaces the final classifier
    with a new fully-connected layer for real/fake classification.
    No architectural modifications - this serves as the pure baseline.

    Architecture:
        EfficientNet-B4 stem + MBConv blocks -> 1792-d features -> FC(2)

    Input size: 380x380 (native EfficientNet-B4 resolution)
    """

    def __init__(self, num_classes=2, dropout=0.0, pretrained=True):
        super(VanillaEfficientNetB4, self).__init__()

        if pretrained:
            self.backbone = EfficientNet.from_pretrained(
                'efficientnet-b4', num_classes=num_classes
            )
        else:
            self.backbone = EfficientNet.from_name(
                'efficientnet-b4', num_classes=num_classes
            )

        # EfficientNet-B4: 1792 features after the head conv + pooling
        num_ftrs = self.backbone._fc.in_features

        if dropout > 0:
            self.backbone._fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(num_ftrs, num_classes)
            )
        else:
            self.backbone._fc = nn.Linear(num_ftrs, num_classes)

        # Expose as `last_linear` for compatibility with TransferModel
        self.last_linear = self.backbone._fc

    def forward(self, x):
        return self.backbone(x)


class EnhancedEfficientNetB4(nn.Module):
    """
    Enhanced EfficientNet-B4 with CBAM Spatial-Channel Attention.

    Injects CBAM attention after the feature extraction backbone to help
    the network focus on semantic anomalies characteristic of AI-generated
    images (texture inconsistencies, boundary artifacts, unnatural patterns).

    Architecture:
        EfficientNet-B4 features (1792-d) -> CBAM -> Pool -> Dropout -> FC(2)

    The CBAM module is applied to the 1792-channel feature maps BEFORE
    global average pooling, allowing the attention mechanism to spatially
    weight which regions and channels are most discriminative.

    Input size: 380x380 (native EfficientNet-B4 resolution)
    """

    def __init__(self, num_classes=2, dropout=0.0, pretrained=True,
                 cbam_reduction=16, cbam_kernel_size=7):
        super(EnhancedEfficientNetB4, self).__init__()

        if pretrained:
            self.backbone = EfficientNet.from_pretrained(
                'efficientnet-b4', num_classes=1000  # Load full ImageNet weights
            )
        else:
            self.backbone = EfficientNet.from_name(
                'efficientnet-b4', num_classes=num_classes
            )

        # Get the feature dimension (1792 for B4)
        num_ftrs = self.backbone._fc.in_features

        # CBAM attention on the 1792-channel feature maps
        self.cbam = CBAM(
            in_channels=num_ftrs,
            reduction_ratio=cbam_reduction,
            spatial_kernel_size=cbam_kernel_size
        )

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # Classification head
        if dropout > 0:
            self.last_linear = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(num_ftrs, num_classes)
            )
        else:
            self.last_linear = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        """
        Pipeline:
            Input -> EfficientNet feature extraction -> CBAM -> Pool -> FC -> Logits
        """
        # Extract features (before pooling/FC): (B, 1792, H, W)
        features = self.backbone.extract_features(x)

        # Apply CBAM attention
        features = self.cbam(features)

        # Global average pooling: (B, 1792, H, W) -> (B, 1792)
        features = self.avgpool(features)
        features = features.flatten(start_dim=1)

        # Classification
        logits = self.last_linear(features)
        return logits


class SpatialOnlyEfficientNetB4(nn.Module):
    """
    EfficientNet-B4 with Spatial-Only Attention.

    EfficientNet already contains built-in Squeeze-and-Excitation (SE) blocks
    which perform channel attention internally. Adding CBAM's channel attention
    on top creates redundancy and hurts performance.

    This variant uses ONLY the spatial attention component, which complements
    (rather than competes with) the existing SE channel attention. The spatial
    attention helps the network focus on WHERE artifacts appear (edges,
    textures, boundaries) while SE handles WHAT features matter (channels).

    Architecture:
        EfficientNet-B4 features (1792-d) -> SpatialAttention -> Pool -> Dropout -> FC(2)

    Input size: 380x380 (native EfficientNet-B4 resolution)
    """

    def __init__(self, num_classes=2, dropout=0.0, pretrained=True,
                 spatial_kernel_size=7):
        super(SpatialOnlyEfficientNetB4, self).__init__()

        if pretrained:
            self.backbone = EfficientNet.from_pretrained(
                'efficientnet-b4', num_classes=1000  # Load full ImageNet weights
            )
        else:
            self.backbone = EfficientNet.from_name(
                'efficientnet-b4', num_classes=num_classes
            )

        # Get the feature dimension (1792 for B4)
        num_ftrs = self.backbone._fc.in_features

        # Spatial-only attention on the 1792-channel feature maps
        # NO channel attention — SE already handles that inside EfficientNet
        self.spatial_attention = SpatialAttention(kernel_size=spatial_kernel_size)

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # Classification head
        if dropout > 0:
            self.last_linear = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(num_ftrs, num_classes)
            )
        else:
            self.last_linear = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        """
        Pipeline:
            Input -> EfficientNet features (with built-in SE) -> Spatial Attention -> Pool -> FC -> Logits
        """
        # Extract features (before pooling/FC): (B, 1792, H, W)
        # Note: SE channel attention is already applied inside EfficientNet's MBConv blocks
        features = self.backbone.extract_features(x)

        # Apply spatial-only attention (complements SE, doesn't compete)
        features = self.spatial_attention(features)

        # Global average pooling: (B, 1792, H, W) -> (B, 1792)
        features = self.avgpool(features)
        features = features.flatten(start_dim=1)

        # Classification
        logits = self.last_linear(features)
        return logits


class EnhancedSpatiotemporalEfficientNetB4(nn.Module):
    """
    Phase 3 Video Detection Architecture (TSM + MHSA).
    
    Combines EfficientNet-B4 spatial extraction with Temporal Shift Module (TSM)
    and Multi-Head Self-Attention (MHSA) for deepfake video detection.
    
    Architecture:
        Input (B*T, C, H, W)
        -> EfficientNet-B4 + TSM inside MBConv blocks
        -> (B*T, 1792, H, W)
        -> Spatial Pool -> (B*T, 1792)
        -> Reshape to (B, T, 1792)
        -> MHSA (TransformerEncoder)
        -> Temporal Pool (Average across T)
        -> Final FC (2)
    """

    def __init__(self, num_classes=2, dropout=0.0, pretrained=True, num_frames=8):
        super(EnhancedSpatiotemporalEfficientNetB4, self).__init__()
        self.num_frames = num_frames

        if pretrained:
            self.backbone = EfficientNet.from_pretrained(
                'efficientnet-b4', num_classes=1000
            )
        else:
            self.backbone = EfficientNet.from_name(
                'efficientnet-b4', num_classes=num_classes
            )

        # Inject TSM into the backbone
        from network.attention import inject_tsm_into_efficientnet
        inject_tsm_into_efficientnet(self, n_frame=num_frames, n_div=8)

        num_ftrs = self.backbone._fc.in_features  # 1792 for B4
        
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)

        # Multi-Head Self-Attention for Temporal modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=num_ftrs,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        # Using 1 layer of MHSA is usually enough for temporal pooling without overfitting
        self.temporal_mhsa = nn.TransformerEncoder(encoder_layer, num_layers=1)

        if dropout > 0:
            self.last_linear = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(num_ftrs, num_classes)
            )
        else:
            self.last_linear = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        # x can be shape (B, T, C, H, W) from DataLoader or (B*T, C, H, W)
        if x.dim() == 5:
            b, t, c, h, w = x.shape
            x = x.reshape(b * t, c, h, w)
        else:
            b = x.size(0) // self.num_frames
            t = self.num_frames
        
        # 1. Spatial + Local Temporal (EfficientNet + TSM)
        features = self.backbone.extract_features(x)  # (B*T, 1792, H, W)
        
        # 2. Spatial Pooling
        features = self.spatial_pool(features)        # (B*T, 1792, 1, 1)
        features = features.flatten(start_dim=1)      # (B*T, 1792)
        
        # 3. Reshape for MHSA
        features = features.view(b, t, -1)            # (B, T, 1792)
        
        # 4. Global Temporal (MHSA)
        features = self.temporal_mhsa(features)       # (B, T, 1792)
        
        # 5. Temporal Pooling (Average over T)
        features = torch.mean(features, dim=1)        # (B, 1792)
        
        # 6. Classification
        logits = self.last_linear(features)           # (B, num_classes)
        return logits
