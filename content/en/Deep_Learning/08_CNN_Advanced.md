# 08. Advanced CNN - Famous Architectures

[Previous: CNN Basics](./07_CNN_Basics.md) | [Next: Transfer Learning](./09_Transfer_Learning.md)

---

## Learning Objectives

- Understand VGG, ResNet, and EfficientNet architectures
- Learn Skip Connection and Residual Learning
- Understand training problems of deep networks and solutions
- Implement with PyTorch

---

## 1. VGG (2014)

### Core Ideas

- Use only small filters (3×3)
- Improve performance by increasing depth
- Simple and consistent structure

### Architecture (VGG16)

```
Input 224×224×3
  ↓
Conv 3×3, 64 ×2 → MaxPool → 112×112×64
  ↓
Conv 3×3, 128 ×2 → MaxPool → 56×56×128
  ↓
Conv 3×3, 256 ×3 → MaxPool → 28×28×256
  ↓
Conv 3×3, 512 ×3 → MaxPool → 14×14×512
  ↓
Conv 3×3, 512 ×3 → MaxPool → 7×7×512
  ↓
FC 4096 → FC 4096 → FC 1000
```

### PyTorch Implementation

```python
def make_vgg_block(in_ch, out_ch, num_convs):
    layers = []
    for i in range(num_convs):
        layers.append(nn.Conv2d(
            in_ch if i == 0 else out_ch,
            out_ch, 3, padding=1
        ))
        layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(2, 2))
    return nn.Sequential(*layers)

class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            make_vgg_block(3, 64, 2),
            make_vgg_block(64, 128, 2),
            make_vgg_block(128, 256, 3),
            make_vgg_block(256, 512, 3),
            make_vgg_block(512, 512, 3),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
```

---

## 2. ResNet (2015)

### Problem: Vanishing Gradients

- Gradients vanish as network gets deeper
- Simply stacking layers degrades performance

### Solution: Residual Connection

```
        ┌─────────────────┐
        │                 │
x ──────┼───► Conv ──► Conv ──►(+)──► ReLU ──► Output
        │                 ↑
        └────────(identity)┘

Output = F(x) + x   (Residual Learning)
```

### Key Insight

- Learning identity function becomes easier
- Gradients flow directly through skip connections
- Can train networks with 1000+ layers

### PyTorch Implementation

```python
class BasicBlock(nn.Module):
    """ResNet basic block"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # Skip connection!
        out = F.relu(out)
        return out
```

### Bottleneck Block (ResNet-50+)

```python
class Bottleneck(nn.Module):
    """1×1 → 3×3 → 1×1 structure"""
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)
        return out
```

---

## 3. ResNet Variants

### Pre-activation ResNet

```
Original: x → Conv → BN → ReLU → Conv → BN → (+) → ReLU
Pre-act: x → BN → ReLU → Conv → BN → ReLU → Conv → (+)
```

### ResNeXt

```python
# Using grouped convolution
self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                       groups=32, padding=1)
```

### SE-ResNet (Squeeze-and-Excitation)

```python
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y  # Channel recalibration
```

---

## 4. EfficientNet (2019)

### Core Ideas

- Balanced scaling of depth, width, and resolution
- Compound Scaling

```
depth: α^φ
width: β^φ
resolution: γ^φ

α × β² × γ² ≈ 2 (computation constraint)
```

### MBConv Block

```python
class MBConv(nn.Module):
    """Mobile Inverted Bottleneck"""
    def __init__(self, in_ch, out_ch, expand_ratio, stride, se_ratio=0.25):
        super().__init__()
        hidden = in_ch * expand_ratio

        self.expand = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU()
        ) if expand_ratio != 1 else nn.Identity()

        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU()
        )

        self.se = SEBlock(hidden, int(in_ch * se_ratio))

        self.project = nn.Sequential(
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        )

        self.use_skip = stride == 1 and in_ch == out_ch

    def forward(self, x):
        out = self.expand(x)
        out = self.depthwise(out)
        out = self.se(out)
        out = self.project(out)
        if self.use_skip:
            out = out + x
        return out
```

---

## 5. Architecture Comparison

| Model | Parameters | Top-1 Acc | Features |
|-------|-----------|-----------|----------|
| VGG16 | 138M | 71.5% | Simple, memory-intensive |
| ResNet-50 | 26M | 76.0% | Skip Connection |
| ResNet-152 | 60M | 78.3% | Deeper version |
| EfficientNet-B0 | 5.3M | 77.1% | Efficient |
| EfficientNet-B7 | 66M | 84.3% | Best performance |

---

## 6. torchvision Pretrained Models

```python
import torchvision.models as models

# Load pretrained models
resnet50 = models.resnet50(weights='IMAGENET1K_V2')
efficientnet = models.efficientnet_b0(weights='IMAGENET1K_V1')
vgg16 = models.vgg16(weights='IMAGENET1K_V1')

# Feature extraction
resnet50.eval()
for param in resnet50.parameters():
    param.requires_grad = False

# Replace last layer (transfer learning)
resnet50.fc = nn.Linear(2048, 10)  # 10 classes
```

---

## 7. Model Selection Guide

### Recommendations by Use Case

| Situation | Recommended Model |
|-----------|------------------|
| Fast inference needed | MobileNet, EfficientNet-B0 |
| High accuracy needed | EfficientNet-B4~B7 |
| Educational/understanding | VGG, ResNet-18 |
| Memory constraints | MobileNet, ShuffleNet |

### Practical Tips

```python
# Check model size
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Calculate FLOPs (thop package)
from thop import profile
flops, params = profile(model, inputs=(torch.randn(1, 3, 224, 224),))
```

---

## Summary

### Core Concepts

1. **VGG**: Repeating small filters, deep networks
2. **ResNet**: Solve vanishing gradients with Skip Connections
3. **EfficientNet**: Efficient scaling

### Evolution

```
LeNet (1998)
  ↓
AlexNet (2012) - GPU usage
  ↓
VGG (2014) - Deeper
  ↓
GoogLeNet (2014) - Inception module
  ↓
ResNet (2015) - Skip Connection
  ↓
EfficientNet (2019) - Compound Scaling
  ↓
Vision Transformer (2020) - Attention
```

---

## Next Steps

In [09_Transfer_Learning.md](./09_Transfer_Learning.md), we'll learn transfer learning using pretrained models.
