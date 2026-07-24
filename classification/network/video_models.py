"""
Video Model Module - Phase 3: TSM + MHSA Video Deepfake Detection

Implements:
  1. TemporalShiftModule (TSM): Zero-parameter channel shifting across time T.
  2. MHSAHead: Multi-Head Self-Attention temporal aggregation head.
  3. VideoEfficientNetB4: Vanilla EfficientNet-B4 + TSM + MHSA wrapper initialized
     from Phase 1/2 best image checkpoint.

Author: Multi-Model Comparative Study Project
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
from network.efficientnet import VanillaEfficientNetB4


class TemporalShiftModule(nn.Module):
    """
    Temporal Shift Module (TSM) for 2D CNNs.
    
    Shifts a fraction (1/fold_div) of channels forward and backward along the
    temporal dimension T at zero parameter and zero computational cost.
    Enables temporal feature exchange inside standard 2D spatial convolutions.
    """
    def __init__(self, num_frames=8, fold_div=8):
        super(TemporalShiftModule, self).__init__()
        self.num_frames = num_frames
        self.fold_div = fold_div

    def forward(self, x):
        B_T, C, H, W = x.shape
        T = self.num_frames
        if B_T % T != 0 or T <= 1:
            return x

        B = B_T // T
        out = x.view(B, T, C, H, W).clone()
        fold = C // self.fold_div

        if fold > 0:
            # Shift left (forward in time)
            out[:, 1:, :fold] = out[:, :-1, :fold]
            out[:, 0, :fold] = 0.0

            # Shift right (backward in time)
            out[:, :-1, fold:2*fold] = out[:, 1:, fold:2*fold]
            out[:, -1, fold:2*fold] = 0.0

        return out.view(B_T, C, H, W)


class TSMBlockWrapper(nn.Module):
    """
    Wraps an MBConvBlock from EfficientNet-PyTorch to inject TSM prior to convolution.
    Compatible with Luke Melas's drop_connect_rate forwarding.
    """
    def __init__(self, block, num_frames=8, fold_div=8):
        super(TSMBlockWrapper, self).__init__()
        self.block = block
        self.tsm = TemporalShiftModule(num_frames=num_frames, fold_div=fold_div)

    def forward(self, x, drop_connect_rate=None):
        x = self.tsm(x)
        if drop_connect_rate is not None:
            return self.block(x, drop_connect_rate=drop_connect_rate)
        return self.block(x)


class MHSAHead(nn.Module):
    """
    Multi-Head Self-Attention (MHSA) Temporal Aggregation Head.
    
    Takes pooled feature vectors across time (B, T, D), applies multi-head self-attention
    to discover global temporal dependencies and inconsistencies across frames, pools over time,
    and classifies into real vs fake. Initialized from scratch.
    """
    def __init__(self, in_features=1792, num_heads=8, num_classes=2, dropout=0.5):
        super(MHSAHead, self).__init__()
        self.in_features = in_features
        self.num_heads = num_heads
        
        # Layer normalization before attention for stability
        self.norm1 = nn.LayerNorm(in_features)
        self.attn = nn.MultiheadAttention(embed_dim=in_features, num_heads=num_heads, dropout=0.2, batch_first=True)
        self.norm2 = nn.LayerNorm(in_features)
        
        # FFN block
        self.ffn = nn.Sequential(
            nn.Linear(in_features, in_features * 2),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features * 2, in_features),
            nn.Dropout(p=dropout)
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x shape: (B, T, D)
        residual = x
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = residual + attn_out
        
        residual = x
        x = residual + self.ffn(self.norm2(x))
        
        # Temporal mean pooling across sequence T
        pooled = x.mean(dim=1)  # (B, D)
        out = self.classifier(pooled)
        return out


class VideoEfficientNetB4(nn.Module):
    """
    Phase 3 Video Model: Vanilla EfficientNet-B4 + TSM + MHSA.
    
    Architecture:
      1. Backbone: Vanilla EfficientNet-B4 initialized with weights from Phase 1/2 best_model.pth.
      2. TSM: Injected into MBConv blocks (0 params).
      3. MHSA: Replaces 2D classifier to aggregate sequence temporal features (trained from scratch).
    """
    def __init__(self, num_classes=2, num_frames=8, pretrained_image_checkpoint=None, dropout=0.5):
        super(VideoEfficientNetB4, self).__init__()
        self.num_frames = num_frames
        
        # 1. Initialize Vanilla EfficientNet-B4 backbone
        vanilla = VanillaEfficientNetB4(num_classes=num_classes, dropout=0.0, pretrained=True)
        self.backbone = vanilla.backbone
        
        # Load Phase 1/2 pretrained checkpoint if provided
        if pretrained_image_checkpoint and os.path.exists(pretrained_image_checkpoint):
            print(f"  [VideoModel] Loading Phase 1/2 checkpoint: {pretrained_image_checkpoint}")
            state_dict = torch.load(pretrained_image_checkpoint, map_location='cpu')
            # Handle state dict unwrapping if stored under 'model_state_dict' or 'state_dict'
            if isinstance(state_dict, dict):
                if 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']
                elif 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
            
            # Strip 'model.backbone.' or 'backbone.' prefix if present from TransferModel wrapper
            clean_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('model.backbone.'):
                    clean_state_dict[k[len('model.backbone.'):]] = v
                elif k.startswith('backbone.'):
                    clean_state_dict[k[len('backbone.'):]] = v
                else:
                    clean_state_dict[k] = v
            
            missing, unexpected = self.backbone.load_state_dict(clean_state_dict, strict=False)
            print(f"  [VideoModel] Checkpoint loaded. Missing: {len(missing)} | Unexpected: {len(unexpected)}")
        elif pretrained_image_checkpoint:
            print(f"  [Warning] Checkpoint not found at {pretrained_image_checkpoint}. Using ImageNet weights.")
            
        # 2. Inject TSM into backbone MBConv blocks
        print(f"  [VideoModel] Injecting Temporal Shift Module (TSM, T={num_frames}) into MBConv blocks...")
        for i in range(len(self.backbone._blocks)):
            self.backbone._blocks[i] = TSMBlockWrapper(self.backbone._blocks[i], num_frames=num_frames, fold_div=8)
            
        # Remove original 2D FC classifier
        num_ftrs = self.backbone._fc.in_features
        self.backbone._fc = nn.Identity()
        
        # 3. Attach MHSA Head trained from scratch
        print(f"  [VideoModel] Attaching Multi-Head Self-Attention (MHSA) head (D={num_ftrs})...")
        self.mhsa_head = MHSAHead(in_features=num_ftrs, num_heads=8, num_classes=num_classes, dropout=dropout)

    def forward(self, x):
        """
        Input x shape: (B, T, C, H, W) or (B*T, C, H, W)
        Output shape: (B, num_classes)
        """
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
        else:
            B_T, C, H, W = x.shape
            T = self.num_frames
            B = B_T // T
            
        # Extract features through TSM-enhanced 2D backbone
        # extract_features returns spatial map (B*T, 1792, H', W')
        features = self.backbone.extract_features(x)
        features = self.backbone._avg_pooling(features)
        features = features.flatten(start_dim=1)  # (B*T, 1792)
        
        # Reshape to sequence (B, T, 1792)
        sequence_features = features.view(B, T, -1)
        
        # Pass through MHSA temporal aggregation head
        out = self.mhsa_head(sequence_features)
        return out


def video_model_selection(modelname="efficientnet_b4_tsm_mhsa", num_out_classes=2, num_frames=8, pretrained_checkpoint=None, dropout=0.5):
    """
    Factory function for Phase 3 video models.
    """
    if modelname == "efficientnet_b4_tsm_mhsa":
        model = VideoEfficientNetB4(
            num_classes=num_out_classes,
            num_frames=num_frames,
            pretrained_image_checkpoint=pretrained_checkpoint,
            dropout=dropout
        )
        return model, 380, num_frames
    else:
        raise NotImplementedError(f"Video model '{modelname}' not implemented.")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Phase 3 Video Model: EfficientNet-B4 + TSM + MHSA...")
    B, T, C, H, W = 2, 8, 3, 380, 380
    model, img_size, frames = video_model_selection(num_frames=T)
    x = torch.randn(B, T, C, H, W)
    out = model(x)
    print(f"  Input sequence: {x.shape} -> Output logits: {out.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,} | Trainable: {trainable_params:,}")
    print("=" * 60)
    print("Test passed successfully!")
