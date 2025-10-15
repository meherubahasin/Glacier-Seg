import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import List

# ------------------------- DropPath -------------------------
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

# ------------------------- MLP -------------------------
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
# ------------------------- Involution2D -------------------------
class Involution2D(nn.Module):
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

        x_k = self.sigma(x)
        K = self.span(self.act(self.reduce(x_k)))
        Hout, Wout = K.shape[2], K.shape[3]
        K = K.view(B, g, k * k, Hout, Wout)

        patches = F.unfold(x, kernel_size=k, padding=k // 2, stride=self.stride)
        patches = patches.view(B, g, C // g, k * k, Hout, Wout)

        out = (patches * K.unsqueeze(2)).sum(dim=3)
        out = out.view(B, C, Hout, Wout)
        return out

# ------------------------- OverlapPatchEmbed -------------------------
class OverlapPatchEmbedInvo(nn.Module):
    def __init__(self, in_chans=3, embed_dim=32, kernel_size=3, stride=2, reduction_ratio=4, groups=1):
        super().__init__()
        self.invo = Involution2D(in_chans, kernel_size=kernel_size, stride=stride,
                                 reduction_ratio=reduction_ratio, groups=groups)
        self.proj = nn.Conv2d(in_chans, embed_dim, 1)
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        x = self.invo(x)
        x = self.proj(x)
        x = self.norm(x)
        return x

class OverlapPatchEmbedConv(nn.Module):
    def __init__(self, in_chans=3, embed_dim=32, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.proj(x)

# ------------------------- MambaBlock -------------------------
class MambaBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0, drop_path=0.0, d_state=16, use_rmsnorm=False):
        super().__init__()
        Norm = nn.RMSNorm 
        self.norm1 = Norm(dim)
        self.mamba = nn.Identity()  # placeholder for actual Mamba SSM
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = Norm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=hidden_dim, act_layer=nn.GELU, drop=drop)

    def forward(self, x, H=None, W=None):
        shortcut = x
        x = self.norm1(x)
        x = self.mamba(x)  
        x = self.drop_path(x) + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x) + shortcut
        return x

# ------------------------- MiTStage -------------------------
class MiTStage(nn.Module):
    def __init__(self, in_chs, embed_dim, depth, drop_path_rates=None, use_involution=True):
        super().__init__()
        if use_involution:
            self.patch_embed = OverlapPatchEmbedInvo(in_chs, embed_dim)
        else:
            self.patch_embed = OverlapPatchEmbedConv(in_chs, embed_dim)

        self.blocks = nn.ModuleList([
            MambaBlock(embed_dim, drop_path=(drop_path_rates[i] if drop_path_rates is not None else 0.0))
            for i in range(depth)
        ])

    def forward(self, x):
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        for blk in self.blocks:
            x = blk(x, H, W)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x

# ------------------------- MixVisionTransformer -------------------------
class MixVisionTransformer(nn.Module):
    def __init__(self, in_chans=3, embed_dims=[32,64,160,256], depths=[2,2,2,2], drop_path_rate=0.0,
                 use_invo_stages=[True,True,False,False]):
        super().__init__()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.stages = nn.ModuleList()
        in_c = in_chans
        for i in range(4):
            d = depths[i]
            dpr_slice = dp_rates[cur:cur + d]
            cur += d
            self.stages.append(
                MiTStage(in_chs=in_c, embed_dim=embed_dims[i], depth=d, drop_path_rates=dpr_slice,
                         use_involution=use_invo_stages[i])
            )
            in_c = embed_dims[i]

    def forward(self, x):
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features

# ------------------------- SegFormerHead -------------------------
class SegFormerHead(nn.Module):
    def __init__(self, in_channels: List[int], embed_dim=128, num_classes=19):
        super().__init__()
        self.proj_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, embed_dim, 1, bias=False),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(inplace=True)
            ) for in_ch in in_channels
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim*len(in_channels), embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Conv2d(embed_dim, num_classes, 1)

    def forward(self, features: List[torch.Tensor]):
        target_h, target_w = features[0].shape[2:]
        proj = []
        for i, feat in enumerate(features):
            x = self.proj_layers[i](feat)
            if x.shape[2:] != (target_h, target_w):
                x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
            proj.append(x)
        x = torch.cat(proj, dim=1)
        x = self.fuse(x)
        x = self.classifier(x)
        return x

# ------------------------- Full Model Wrapper -------------------------
class GlacierSeg(nn.Module):
    def __init__(self, in_chans=3, num_classes=1):
        super().__init__()
        self.backbone = MixVisionTransformer(in_chans=in_chans)
        self.decoder = SegFormerHead([32, 64, 160, 256], num_classes=num_classes)

    def forward(self, x):
        features = self.backbone(x)
        out = self.decoder(features)
        return out

if __name__ == "__main__":
    model = GlacierSeg(in_chans=3, num_classes=1)
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print("Output shape:", y.shape)
