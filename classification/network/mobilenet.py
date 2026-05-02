"""
MobileNetV3-Small Wrapper - Lightweight CNN Baseline

Provides a lightweight baseline model for the comparative study:
  - MobileNetV3Small: Standard MobileNetV3-Small (ImageNet pretrained)

This serves as the "weaker" baseline to demonstrate the advantage of
EfficientNet-B4's deeper architecture and compound scaling for
AI-generated image detection.

Backbone: torchvision.models.mobilenet_v3_small
    https://pytorch.org/vision/main/models/mobilenetv3.html

Reference:
    Howard et al., "Searching for MobileNetV3"
    (ICCV 2019) - https://arxiv.org/abs/1905.02244

Parameters: ~2.5M (vs EfficientNet-B4's 19.3M)
Input size: 224x224 (native MobileNetV3 resolution)

Author: Multi-Model Comparative Study Project
"""
import torch
import torch.nn as nn
from torchvision import models


class MobileNetV3Small(nn.Module):
    """
    MobileNetV3-Small wrapper for binary classification.

    Loads ImageNet-pretrained weights and replaces the final classifier
    with a new fully-connected layer for real/fake classification.
    
    This is a deliberately lightweight model (~2.5M parameters) chosen
    as a baseline to contrast against EfficientNet-B4's 19.3M parameters,
    demonstrating the impact of model capacity on forensic detection.

    Architecture:
        MobileNetV3-Small stem + inverted residuals -> 576-d features -> FC(2)

    Input size: 224x224 (native MobileNetV3 resolution)
    """

    def __init__(self, num_classes=2, dropout=0.0, pretrained=True):
        super(MobileNetV3Small, self).__init__()

        if pretrained:
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            self.backbone = models.mobilenet_v3_small(weights=weights)
        else:
            self.backbone = models.mobilenet_v3_small(weights=None)

        # MobileNetV3-Small: classifier is a Sequential with Linear layers
        # classifier[0] = Linear(576, 1024)
        # classifier[1] = Hardswish
        # classifier[2] = Dropout
        # classifier[3] = Linear(1024, 1000)
        num_ftrs = self.backbone.classifier[3].in_features  # 1024

        if dropout > 0:
            self.backbone.classifier[2] = nn.Dropout(p=dropout)
        
        self.backbone.classifier[3] = nn.Linear(num_ftrs, num_classes)

        # Expose as `last_linear` for compatibility with TransferModel
        self.last_linear = self.backbone.classifier[3]

    def forward(self, x):
        return self.backbone(x)
