# ----------------------------------------------------------------------------#

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import random

# ----------------------------------------------------------------------------#

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w
    

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_se=True):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.se = SEBlock(out_channels) if use_se else nn.Identity()
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        return F.relu(out)

# ----------------------------------------------------------------------------#

class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: (B, N, D)
        h = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + h
        h = self.mlp(self.norm2(x))
        x = x + h
        return x
    
# ----------------------------------------------------------------------------#

class HybridOCR(nn.Module):
    def __init__(self, num_classes, embed_dim=512, num_heads=8, num_layers=4):
        super(HybridOCR, self).__init__()

        # CNN stem.
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64)
        )

        # Intermediate conv stages.
        self.layer1 = ResidualBlock(64, 128, stride=2)
        self.layer2 = ResidualBlock(128, 256, stride=2)

        # Projection to embedding dim for the transformer.
        self.proj = nn.Conv2d(256, embed_dim, kernel_size=1)

        # Positional encoding (learnable).
        self.pos_embedding = nn.Parameter(torch.randn(1, 196, embed_dim)) # 14x14 patches from 200 input approx.

        # Transformer encoder.
        self.transformer = nn.Sequential(
            *[TransformerEncoderBlock(embed_dim, num_heads=num_heads) for _ in range(num_layers)]
        )

        # Final residual block (post-transformer).
        self.final_fc = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # CNN features.
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)

        # Project to embeddings.
        x = self.proj(x) # (B, D, H, W)
        B, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2) # (B, N, D), where N=H*W

        # Adding positional embeddings.
        if x.size(1) != self.pos_embedding.size(1):
            # Interpolating if the feature map changed.
            pos = F.interpolate(self.pos_embedding.transpose(1, 2).reshape(1, -1, int(self.pos_embedding.size(1)**0.5), -1),
                                size=(H, W), mode='bilinear')
            pos = pos.flatten(2).transpose(1, 2)
        else:
            pos = self.pos_embedding
        
        x = x + pos

        # Transformer encoder.
        x = self.transformer(x)

        # Global average pooling (over sequence length).
        x = x.mean(dim=1)

        # Classification head.
        out = self.final_fc(x)
        return out
