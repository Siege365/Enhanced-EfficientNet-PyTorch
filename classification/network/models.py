"""
Model Selection Module - Multi-Architecture Comparative Study

Provides a unified interface for selecting between:
  - 'mobilenet_v3':            MobileNetV3-Small (lightweight baseline)
  - 'efficientnet_b4':         Vanilla EfficientNet-B4 (proposed model)
  - 'efficientnet_b4_cbam':    Enhanced EfficientNet-B4 with full CBAM attention
  - 'efficientnet_b4_spatial': EfficientNet-B4 with spatial-only attention

Author: Multi-Model Comparative Study Project
"""
import torch
import torch.nn as nn
from network.efficientnet import VanillaEfficientNetB4, EnhancedEfficientNetB4, SpatialOnlyEfficientNetB4
from network.mobilenet import MobileNetV3Small


class TransferModel(nn.Module):
    """
    Transfer learning wrapper for binary classification.

    Supports:
        - 'mobilenet_v3': MobileNetV3-Small (lightweight baseline)
        - 'efficientnet_b4': Vanilla EfficientNet-B4 (proposed model)
        - 'efficientnet_b4_cbam': EfficientNet-B4 + full CBAM attention
        - 'efficientnet_b4_spatial': EfficientNet-B4 + spatial-only attention
    """

    def __init__(self, modelchoice, num_out_classes=2, dropout=0.0):
        super(TransferModel, self).__init__()
        self.modelchoice = modelchoice

        if modelchoice == 'mobilenet_v3':
            self.model = MobileNetV3Small(
                num_classes=num_out_classes,
                dropout=dropout,
                pretrained=True
            )

        elif modelchoice in ('efficientnet_b4', 'efficientnet_b4_continuous'):
            self.model = VanillaEfficientNetB4(
                num_classes=num_out_classes,
                dropout=dropout,
                pretrained=True
            )

        elif modelchoice == 'efficientnet_b4_cbam':
            self.model = EnhancedEfficientNetB4(
                num_classes=num_out_classes,
                dropout=dropout,
                pretrained=True
            )

        elif modelchoice == 'efficientnet_b4_spatial':
            self.model = SpatialOnlyEfficientNetB4(
                num_classes=num_out_classes,
                dropout=dropout,
                pretrained=True
            )

        else:
            raise Exception(
                f"Invalid model choice '{modelchoice}'. "
                f"Choose from: mobilenet_v3, efficientnet_b4, efficientnet_b4_cbam, efficientnet_b4_spatial"
            )

    def set_trainable_up_to(self, boolean, layername=None):
        """
        Freezes all layers below a specific layer and sets the following layers
        to true if boolean else only the fully connected final layer.
        """
        if layername is None:
            for i, param in self.model.named_parameters():
                param.requires_grad = True
                return
        else:
            for i, param in self.model.named_parameters():
                param.requires_grad = False
        if boolean:
            ct = []
            found = False
            for name, child in self.model.named_children():
                if layername in ct:
                    found = True
                    for params in child.parameters():
                        params.requires_grad = True
                ct.append(name)
            if not found:
                raise Exception('Layer not found, cant finetune!')
        else:
            # Make fc trainable
            for param in self.model.last_linear.parameters():
                param.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        return x


def model_selection(modelname, num_out_classes, dropout=None):
    """
    Factory function for model selection.

    Args:
        modelname: 'mobilenet_v3', 'efficientnet_b4', 'efficientnet_b4_cbam', etc.
        num_out_classes: Number of output classes (2 for real/fake)
        dropout: Dropout probability (None or 0 for no dropout)

    Returns:
        tuple: (model, image_size, pretrained_flag, input_list, augmentation)
    """
    if modelname == 'mobilenet_v3':
        return TransferModel(
            modelchoice='mobilenet_v3',
            num_out_classes=num_out_classes,
            dropout=dropout or 0.0
        ), 224, True, ['image'], None

    elif modelname in ('efficientnet_b4', 'efficientnet_b4_continuous'):
        return TransferModel(
            modelchoice='efficientnet_b4',
            num_out_classes=num_out_classes,
            dropout=dropout or 0.0
        ), 380, True, ['image'], None

    elif modelname == 'efficientnet_b4_cbam':
        return TransferModel(
            modelchoice='efficientnet_b4_cbam',
            num_out_classes=num_out_classes,
            dropout=dropout or 0.0
        ), 380, True, ['image'], None

    elif modelname == 'efficientnet_b4_spatial':
        return TransferModel(
            modelchoice='efficientnet_b4_spatial',
            num_out_classes=num_out_classes,
            dropout=dropout or 0.0
        ), 380, True, ['image'], None

    else:
        raise NotImplementedError(
            f"Model '{modelname}' not implemented. "
            f"Choose from: mobilenet_v3, efficientnet_b4, efficientnet_b4_cbam, efficientnet_b4_spatial"
        )


if __name__ == '__main__':
    # Quick smoke test
    print("=" * 60)
    print("Testing Vanilla EfficientNet-B4 (baseline)...")
    model, image_size, *_ = model_selection('efficientnet_b4', num_out_classes=2)
    x = torch.randn(1, 3, image_size, image_size)
    out = model(x)
    print(f"  Input: {x.shape} -> Output: {out.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print()
    print("Testing Enhanced EfficientNet-B4 (CBAM)...")
    model_c, image_size_c, *_ = model_selection('efficientnet_b4_cbam', num_out_classes=2)
    x_c = torch.randn(1, 3, image_size_c, image_size_c)
    out_c = model_c(x_c)
    print(f"  Input: {x_c.shape} -> Output: {out_c.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model_c.parameters()):,}")
    print("=" * 60)
    print("All tests passed!")
