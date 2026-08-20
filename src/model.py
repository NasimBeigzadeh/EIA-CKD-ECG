import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
"""Channel attention module used in CBAM."""

```
def __init__(self, in_channels, ratio=16):
    super().__init__()

    self.avg_pool = nn.AdaptiveAvgPool2d(1)
    self.max_pool = nn.AdaptiveMaxPool2d(1)

    self.fc = nn.Sequential(
        nn.Conv2d(
            in_channels,
            in_channels // ratio,
            kernel_size=1,
            bias=False,
        ),
        nn.ReLU(),
        nn.Conv2d(
            in_channels // ratio,
            in_channels,
            kernel_size=1,
            bias=False,
        ),
    )

    self.sigmoid = nn.Sigmoid()

def forward(self, x):
    avg_out = self.fc(self.avg_pool(x))
    max_out = self.fc(self.max_pool(x))

    return self.sigmoid(avg_out + max_out)
```

class SpatialAttention(nn.Module):
"""Spatial attention module used in CBAM."""

```
def __init__(self, kernel_size=7):
    super().__init__()

    self.conv = nn.Conv2d(
        2,
        1,
        kernel_size=kernel_size,
        padding=kernel_size // 2,
        bias=False,
    )

    self.sigmoid = nn.Sigmoid()

def forward(self, x):
    avg_out = torch.mean(x, dim=1, keepdim=True)
    max_out, _ = torch.max(x, dim=1, keepdim=True)

    attention = torch.cat([avg_out, max_out], dim=1)

    return self.sigmoid(self.conv(attention))
```

class CBAM(nn.Module):
"""Convolutional Block Attention Module."""

```
def __init__(self, in_channels):
    super().__init__()

    self.channel_attention = ChannelAttention(in_channels)
    self.spatial_attention = SpatialAttention()

def forward(self, x):
    x = x * self.channel_attention(x)
    x = x * self.spatial_attention(x)

    return x
```

class DepthwiseSeparableConv(nn.Module):
"""Depthwise separable convolution."""

```
def __init__(
    self,
    in_channels,
    out_channels,
    kernel_size=3,
    padding=1,
):
    super().__init__()

    self.depthwise = nn.Conv2d(
        in_channels,
        in_channels,
        kernel_size=kernel_size,
        padding=padding,
        groups=in_channels,
        bias=False,
    )

    self.pointwise = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        bias=False,
    )

def forward(self, x):
    return self.pointwise(self.depthwise(x))
```

class ResidualBlock(nn.Module):
"""Residual block with depthwise separable convolutions and CBAM."""

```
def __init__(self, in_channels, out_channels):
    super().__init__()

    self.conv1 = DepthwiseSeparableConv(
        in_channels,
        out_channels,
    )

    self.bn1 = nn.BatchNorm2d(out_channels)

    self.conv2 = DepthwiseSeparableConv(
        out_channels,
        out_channels,
    )

    self.bn2 = nn.BatchNorm2d(out_channels)

    self.cbam = CBAM(out_channels)

    self.shortcut = nn.Sequential()

    if in_channels != out_channels:
        self.shortcut = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

def forward(self, x):
    residual = self.shortcut(x)

    out = F.relu(
        self.bn1(
            self.conv1(x)
        )
    )

    out = self.cbam(out)

    out = self.bn2(
        self.conv2(out)
    )

    out = self.cbam(out)

    out = out + residual

    return F.relu(out)
```

class EfficientCNN(nn.Module):
"""
Lightweight CNN backbone used by EIA-CKD.

```
Input:
    RGB ECG image of shape [B, 3, 300, 300]

Output:
    Classification logits of shape [B, num_classes]
"""

def __init__(self, num_classes=5):
    super().__init__()

    self.conv0 = DepthwiseSeparableConv(3, 32)
    self.bn0 = nn.BatchNorm2d(32)
    self.pool0 = nn.MaxPool2d(2, 2)

    self.res0 = ResidualBlock(32, 32)
    self.pool00 = nn.MaxPool2d(2, 2)

    self.res1 = ResidualBlock(32, 64)
    self.pool1 = nn.MaxPool2d(2, 2)

    self.res2 = ResidualBlock(64, 64)
    self.pool2 = nn.MaxPool2d(2, 2)

    self.res3 = ResidualBlock(64, 128)
    self.pool3 = nn.MaxPool2d(2, 2)

    self.res4 = ResidualBlock(128, 128)
    self.pool4 = nn.AdaptiveAvgPool2d((1, 1))

    self.classifier = nn.Sequential(
        nn.Linear(128, 32),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(32, num_classes),
    )

def extract_features(self, x):
    x = F.relu(
        self.bn0(
            self.conv0(x)
        )
    )

    x = self.pool0(x)
    x = self.res0(x)

    x = self.pool00(x)
    x = self.res1(x)

    x = self.pool1(x)
    x = self.res2(x)

    x = self.pool2(x)
    x = self.res3(x)

    x = self.pool3(x)
    x = self.res4(x)

    x = self.pool4(x)
    x = torch.flatten(x, 1)

    return x

def forward(self, x):
    features = self.extract_features(x)

    return self.classifier(features)
```
