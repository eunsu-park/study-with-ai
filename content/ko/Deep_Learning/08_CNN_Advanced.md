# 06. CNN 심화 - 유명 아키텍처

## 학습 목표

- VGG, ResNet, EfficientNet 아키텍처 이해
- Skip Connection과 Residual Learning
- 깊은 네트워크의 학습 문제와 해결책
- PyTorch로 구현

---

## 1. VGG (2014)

### 핵심 아이디어

- 작은 필터(3×3)만 사용
- 깊이를 늘려 성능 향상
- 단순하고 일관된 구조

### 구조 (VGG16)

```
입력 224×224×3
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

### PyTorch 구현

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

### 문제: 기울기 소실

- 네트워크가 깊어지면 기울기가 소실됨
- 단순히 층을 쌓으면 성능이 떨어짐

### 해결: Residual Connection

```
        ┌─────────────────┐
        │                 │
x ──────┼───► Conv ──► Conv ──►(+)──► ReLU ──► 출력
        │                 ↑
        └────────(identity)┘

출력 = F(x) + x   (잔차 학습)
```

### 핵심 인사이트

- 항등 함수 학습이 쉬워짐
- 기울기가 skip connection을 통해 직접 전파
- 1000층 이상도 학습 가능

### PyTorch 구현

```python
class BasicBlock(nn.Module):
    """ResNet 기본 블록"""
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
    """1×1 → 3×3 → 1×1 구조"""
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

## 3. ResNet 변형들

### Pre-activation ResNet

```
기존: x → Conv → BN → ReLU → Conv → BN → (+) → ReLU
Pre-act: x → BN → ReLU → Conv → BN → ReLU → Conv → (+)
```

### ResNeXt

```python
# 그룹 합성곱 사용
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
        return x * y  # 채널 재조정
```

---

## 4. EfficientNet (2019)

### 핵심 아이디어

- 깊이, 너비, 해상도의 균형 있는 스케일링
- Compound Scaling

```
depth: α^φ
width: β^φ
resolution: γ^φ

α × β² × γ² ≈ 2 (계산량 제약)
```

### MBConv 블록

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

## 5. 아키텍처 비교

| 모델 | 파라미터 | Top-1 Acc | 특징 |
|------|----------|-----------|------|
| VGG16 | 138M | 71.5% | 단순, 메모리 많이 사용 |
| ResNet-50 | 26M | 76.0% | Skip Connection |
| ResNet-152 | 60M | 78.3% | 더 깊은 버전 |
| EfficientNet-B0 | 5.3M | 77.1% | 효율적 |
| EfficientNet-B7 | 66M | 84.3% | 최고 성능 |

---

## 6. torchvision 사전 학습 모델

```python
import torchvision.models as models

# 사전 학습된 모델 로드
resnet50 = models.resnet50(weights='IMAGENET1K_V2')
efficientnet = models.efficientnet_b0(weights='IMAGENET1K_V1')
vgg16 = models.vgg16(weights='IMAGENET1K_V1')

# 특성 추출용
resnet50.eval()
for param in resnet50.parameters():
    param.requires_grad = False

# 마지막 층만 교체 (전이 학습)
resnet50.fc = nn.Linear(2048, 10)  # 10 클래스
```

---

## 7. 모델 선택 가이드

### 용도별 추천

| 상황 | 추천 모델 |
|------|----------|
| 빠른 추론 필요 | MobileNet, EfficientNet-B0 |
| 높은 정확도 필요 | EfficientNet-B4~B7 |
| 교육/이해 목적 | VGG, ResNet-18 |
| 메모리 제한 | MobileNet, ShuffleNet |

### 실전 팁

```python
# 모델 크기 확인
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# FLOPs 계산 (thop 패키지)
from thop import profile
flops, params = profile(model, inputs=(torch.randn(1, 3, 224, 224),))
```

---

## 정리

### 핵심 개념

1. **VGG**: 작은 필터 반복, 깊은 네트워크
2. **ResNet**: Skip Connection으로 기울기 소실 해결
3. **EfficientNet**: 효율적인 스케일링

### 발전 흐름

```
LeNet (1998)
  ↓
AlexNet (2012) - GPU 사용
  ↓
VGG (2014) - 더 깊게
  ↓
GoogLeNet (2014) - Inception 모듈
  ↓
ResNet (2015) - Skip Connection
  ↓
EfficientNet (2019) - Compound Scaling
  ↓
Vision Transformer (2020) - Attention
```

---

## 다음 단계

[09_Transfer_Learning.md](./09_Transfer_Learning.md)에서 사전 학습된 모델을 활용한 전이 학습을 배웁니다.
