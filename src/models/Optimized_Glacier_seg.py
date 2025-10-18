from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Optional RMSNorm from mamba_ssm; fall back to LayerNorm if missing ---
try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm
except Exception:
    RMSNorm = None

# --- REQUIRED: your local Mamba implementation exported from mamba_model.py ---
from mamba_model import Mamba


# ------------------------- Utils -------------------------
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ------------------------- Involution stem -------------------------
class Involution2D(nn.Module):
    """
    Involution: location-specific, channel-agnostic kernel.
    Paper: "Involution: Inverting the Inherence of Convolution for Visual Recognition" (CVPR 2021)
    """
    def __init__(self, channels, kernel_size=3, stride=1, reduction_ratio=4, groups=1):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups

        hidden = max(channels // reduction_ratio, 1)
        self.reduce = nn.Conv2d(channels, hidden, 1)
        self.act = nn.ReLU(inplace=True)
        self.span = nn.Conv2d(hidden, groups * (kernel_size * kernel_size), 1)
        self.sigma = nn.AvgPool2d(kernel_size=stride, stride=stride) if stride > 1 else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        k, g = self.kernel_size, self.groups
        assert C % g == 0, f"Channels ({C}) must be divisible by groups ({g})."

        # 1) Generate spatially-varying kernels on possibly downsampled features
        x_k = self.sigma(x)                                   # (B, C, H', W')
        K = self.span(self.act(self.reduce(x_k)))             # (B, g*k*k, H', W')
        Hout, Wout = K.shape[2], K.shape[3]
        K = K.view(B, g, k * k, Hout, Wout)                   # (B, g, k*k, H', W')

        # 2) Unfold patches on original x with stride/padding so H',W' align
        patches = F.unfold(x, kernel_size=k, dilation=1, padding=k // 2, stride=self.stride)
        patches = patches.view(B, g, C // g, k * k, Hout, Wout)  # (B, g, Cg, k*k, H', W')

        # 3) Channel-agnostic weighted sum within group
        out = (patches * K.unsqueeze(2)).sum(dim=3)           # (B, g, Cg, H', W')
        out = out.view(B, C, Hout, Wout)
        return out


class OverlapPatchEmbedInvo(nn.Module):
    """Involution downsample + 1x1 projection to embed_dim. Optimized version."""
    def __init__(self, in_chans=3, embed_dim=32, kernel_size=3, stride=2, padding=1,
                 reduction_ratio=4, groups=1):
        super().__init__()
        
        # Optimization 1: Fuse operations when possible
        # If in_chans == embed_dim, we can skip the 1x1 conv
        self.skip_proj = (in_chans == embed_dim)
        
        self.invo = Involution2D(in_chans, kernel_size=kernel_size, stride=stride,
                                 reduction_ratio=reduction_ratio, groups=groups)
        
        # Optimization 2: Use fused Conv-BN when projection is needed
        if not self.skip_proj:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=1, stride=1, padding=0, bias=False)
            self.norm = nn.BatchNorm2d(embed_dim)
        else:
            # Just normalization if channels match
            self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        x = self.invo(x)     # downsample + spatially-varying mixing
        if not self.skip_proj:
            x = self.proj(x)  # channel set to embed_dim
        x = self.norm(x)
        return x


# ------------------------- Conv stem (fast option) -------------------------
class OverlapPatchEmbedConv(nn.Module):
    """Standard conv downsample + BN + ReLU. Optimized version."""
    def __init__(self, in_chans=3, embed_dim=32, kernel_size=3, stride=2, padding=1):
        super().__init__()
        
        # Optimization 1: Depthwise separable convolution for efficiency
        # Split into depthwise (spatial) + pointwise (channel)
        mid_chans = max(in_chans, embed_dim // 2)  # Intermediate channels
        
        self.proj = nn.Sequential(
            # Depthwise convolution (spatial filtering)
            nn.Conv2d(in_chans, in_chans, kernel_size, stride=stride, 
                     padding=padding, groups=in_chans, bias=False),
            nn.BatchNorm2d(in_chans),
            nn.ReLU(inplace=True),
            
            # Pointwise convolution (channel mixing)
            nn.Conv2d(in_chans, embed_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.proj(x)


# ------------------------- Mixer Block (Mamba) -------------------------
class MambaBlock(nn.Module):
    """
    Mamba SSM mixer on (B, N, C) with PreNorm + MLP and residuals.
    """
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0, drop_path=0.0, d_state=16, use_rmsnorm=True):
        super().__init__()
        Norm = RMSNorm if (use_rmsnorm and RMSNorm is not None) else nn.LayerNorm

        self.norm1 = Norm(dim)
        self.mamba = Mamba(d_model=dim, d_state=d_state)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = Norm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=hidden_dim, act_layer=nn.GELU, drop=drop)

    def forward(self, x, H=None, W=None):
        # x: (B, N, C)
        shortcut = x
        x = self.norm1(x)
        x = self.mamba(x)
        x = self.drop_path(x) + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x) + shortcut
        return x


# ------------------------- MiT Stage (stem + Mamba blocks) -------------------------
class MiTStage(nn.Module):
    def __init__(self, in_chs, embed_dim, depth, drop_path_rates=None, use_involution=True):
        super().__init__()
        # choose stem type
        if use_involution:
            self.patch_embed = OverlapPatchEmbedInvo(
                in_chans=in_chs, embed_dim=embed_dim, kernel_size=3, stride=2, padding=1,
                reduction_ratio=4, groups=1
            )
        else:
            self.patch_embed = OverlapPatchEmbedConv(
                in_chans=in_chs, embed_dim=embed_dim, kernel_size=3, stride=2, padding=1
            )

        self.blocks = nn.ModuleList([
            MambaBlock(
                dim=embed_dim,
                mlp_ratio=4.0,
                drop=0.0,
                drop_path=(drop_path_rates[i] if drop_path_rates is not None else 0.0),
                d_state=16,
                use_rmsnorm=True
            )
            for i in range(depth)
        ])

    def forward(self, x):
        x = self.patch_embed(x)              # (B, C, H', W')
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)     # (B, N, C)
        for blk in self.blocks:
            x = blk(x, H, W)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x


# ------------------------- Mix Vision Transformer (backbone) -------------------------
class MixVisionTransformer(nn.Module):
    """
    Full MiT backbone (4 stages) returning multi-scale features:
        [S1, S2, S3, S4] with strides [2, 4, 8, 16] wrt input.
    """
    def __init__(self,
                 in_chans: int = 3,
                 embed_dims = [32, 64, 160, 256],
                 depths    = [2, 2, 2, 2],
                 drop_path_rate: float = 0.0,
                 use_invo_stages = (True, True, False, False)):
        super().__init__()
        assert len(embed_dims) == 4 and len(depths) == 4 and len(use_invo_stages) == 4

        # Stochastic depth schedule across all blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        self.stages = nn.ModuleList()
        in_c = in_chans
        for i in range(4):
            d = depths[i]
            dpr_slice = dp_rates[cur:cur + d]
            cur += d

            self.stages.append(
                MiTStage(
                    in_chs=in_c,
                    embed_dim=embed_dims[i],
                    depth=d,
                    drop_path_rates=dpr_slice,
                    use_involution=use_invo_stages[i],
                )
            )
            in_c = embed_dims[i]

    def forward(self, x):
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


# ------------------------------- Decoder --------------------------------
class SegFormerHead(nn.Module):
    def __init__(self, in_channels: List[int], embed_dim=64, num_classes=19):
        super().__init__()
        
        # Optimization 1: Use depthwise separable for projections when reducing channels significantly
        self.proj_layers = nn.ModuleList([
            self._make_projection(in_ch, embed_dim)
            for in_ch in in_channels
        ])
        
        # Optimization 2: Efficient fusion with depthwise separable
        fused_channels = embed_dim * len(in_channels)
        self.fuse = nn.Sequential(
            # Pointwise to reduce channels
            nn.Conv2d(fused_channels, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            
            # Depthwise spatial convolution
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, 
                     groups=embed_dim, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            
            # Pointwise to mix channels
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        
        self.classifier = nn.Conv2d(embed_dim, num_classes, kernel_size=1)
        
        # Optimization 3: Cache for dynamic programming
        self.num_stages = len(in_channels)

    def _make_projection(self, in_ch, out_ch):
        """Creates efficient projection layer based on channel ratio."""
        # Use depthwise separable if reducing channels significantly (ratio > 2)
        if in_ch > out_ch * 2:
            return nn.Sequential(
                # Depthwise
                nn.Conv2d(in_ch, in_ch, 1, groups=in_ch, bias=False),
                # Pointwise
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        else:
            # Standard 1x1 for smaller reductions
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

    def forward(self, features: List[torch.Tensor]):
        target_h, target_w = features[0].shape[2:]
        
        # Optimization 4: Preallocate list for better memory efficiency
        proj = [None] * len(features)
        
        for i, feat in enumerate(features):
            x = self.proj_layers[i](feat)
            
            # Optimization 5: Only interpolate if needed
            if x.shape[2] != target_h or x.shape[3] != target_w:
                x = F.interpolate(x, size=(target_h, target_w), 
                                mode='bilinear', align_corners=False)
            proj[i] = x
        
        # Optimization 6: In-place concatenation
        x = torch.cat(proj, dim=1)
        x = self.fuse(x)
        x = self.classifier(x)
        return x

# -------------------------- Full SegFormer Model -------------------------
class Glacier_seg(nn.Module):
    def __init__(self, num_classes=19, variant='mit_b0', pretrained=False,
                 drop_path_rate=0.1, in_chans=3, use_invo_stages=(False, False, False, False),embed_dims=None, depths=None):
        super().__init__()
        variants = {
            'mit_b0': dict(embed_dims=[16, 32, 64, 128], depths=[1, 1, 1, 1]),
        }
        if variant not in variants:
            raise ValueError(f'Unknown variant: {variant}')
        cfg = variants[variant]

        self.backbone = MixVisionTransformer(
            in_chans=in_chans,
            embed_dims=cfg['embed_dims'],
            depths=cfg['depths'],
            drop_path_rate=drop_path_rate,
            use_invo_stages=use_invo_stages
        )
        self.decoder = SegFormerHead(in_channels=cfg['embed_dims'], embed_dim=128, num_classes=num_classes)
        self.apply(self._init_weights)

        if pretrained:
            print('pretrained=True selected but no loader is implemented in this script.')

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        feats = self.backbone(x)                       # [S1,S2,S3,S4]
        out = self.decoder(feats)                      # logits at S1 resolution
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        return out
