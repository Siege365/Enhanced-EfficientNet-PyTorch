"""
Model Selection Module - EfficientNet-B4 Variants

Provides a unified interface for selecting between:
  - 'efficientnet_b4':      Vanilla EfficientNet-B4 (baseline)
  - 'efficientnet_b4_cbam': Enhanced EfficientNet-B4 with CBAM attention

Author: Multi-Model Comparative Study Project
"""
import torch
import torch.nn as nn
from network.efficientnet import VanillaEfficientNetB4, EnhancedEfficientNetB4


class TransferModel(nn.Module):
    """
    Transfer learning wrapper for EfficientNet-B4 binary classification.

    Supports:
        - 'efficientnet_b4': Vanilla EfficientNet-B4 (baseline)
        - 'efficientnet_b4_cbam': EfficientNet-B4 + CBAM attention (enhanced)
    """

    def __init__(self, modelchoice, num_out_classes=2, dropout=0.0):
        super(TransferModel, self).__init__()
        self.modelchoice = modelchoice

        if modelchoice == 'efficientnet_b4':
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

        else:
            raise Exception(
                f"Invalid model choice '{modelchoice}'. "
                f"Choose from: efficientnet_b4, efficientnet_b4_cbam"
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
        modelname: 'efficientnet_b4' or 'efficientnet_b4_cbam'
        num_out_classes: Number of output classes (2 for real/fake)
        dropout: Dropout probability (None or 0 for no dropout)

    Returns:
        tuple: (model, image_size, pretrained_flag, input_list, augmentation)
    """
    if modelname == 'efficientnet_b4':
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

    else:
        raise NotImplementedError(
            f"Model '{modelname}' not implemented. "
            f"Choose from: efficientnet_b4, efficientnet_b4_cbam"
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
