# 07. 전이학습 (Transfer Learning)

## 학습 목표

- 전이학습의 개념과 이점
- 사전 학습 모델 활용
- 미세 조정(Fine-tuning) 전략
- 실전 이미지 분류 프로젝트

---

## 1. 전이학습이란?

### 개념

```
ImageNet으로 학습된 모델
        ↓
    저수준 특징 (에지, 텍스처) → 재사용
        ↓
    고수준 특징 → 새 데이터에 맞게 조정
        ↓
    새로운 분류 작업
```

### 이점

- 적은 데이터로도 높은 성능
- 빠른 학습
- 더 나은 일반화

---

## 2. 전이학습 전략

### 전략 1: 특성 추출 (Feature Extraction)

```python
# 사전 학습 모델의 가중치 고정
for param in model.parameters():
    param.requires_grad = False

# 마지막 층만 교체
model.fc = nn.Linear(2048, num_classes)
```

- 사전 학습된 특징 그대로 사용
- 마지막 분류층만 학습
- 데이터가 적을 때 적합

### 전략 2: 미세 조정 (Fine-tuning)

```python
# 전체 또는 일부 층 학습
for param in model.parameters():
    param.requires_grad = True

# 낮은 학습률 사용
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
```

- 사전 학습 가중치를 시작점으로
- 전체 네트워크 미세 조정
- 데이터가 충분할 때 적합

### 전략 3: 점진적 해동 (Gradual Unfreezing)

```python
# 1단계: 마지막 층만
for param in model.parameters():
    param.requires_grad = False
model.fc.requires_grad_(True)
train_for_epochs(5)

# 2단계: 마지막 블록도
model.layer4.requires_grad_(True)
train_for_epochs(5)

# 3단계: 전체
model.requires_grad_(True)
train_for_epochs(10)
```

---

## 3. PyTorch 구현

### 기본 전이학습

```python
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, datasets

# 1. 사전 학습 모델 로드
model = models.resnet50(weights='IMAGENET1K_V2')

# 2. 특성 추출기로 사용 (가중치 고정)
for param in model.parameters():
    param.requires_grad = False

# 3. 마지막 층 교체
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, num_classes)
)
```

### 데이터 전처리

```python
# ImageNet 정규화 사용
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])
```

---

## 4. 학습 전략

### 차등 학습률 (Discriminative Learning Rates)

```python
# 층별 다른 학습률
optimizer = torch.optim.Adam([
    {'params': model.layer1.parameters(), 'lr': 1e-5},
    {'params': model.layer2.parameters(), 'lr': 5e-5},
    {'params': model.layer3.parameters(), 'lr': 1e-4},
    {'params': model.layer4.parameters(), 'lr': 5e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3},
])
```

### 학습률 스케줄링

```python
# Warmup + Cosine Decay
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-3,
    epochs=epochs,
    steps_per_epoch=len(train_loader),
    pct_start=0.1  # 10% warmup
)
```

---

## 5. 다양한 사전 학습 모델

### torchvision 모델

```python
# 분류용
resnet50 = models.resnet50(weights='IMAGENET1K_V2')
efficientnet = models.efficientnet_b0(weights='IMAGENET1K_V1')
vit = models.vit_b_16(weights='IMAGENET1K_V1')

# 객체 검출용
fasterrcnn = models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')

# 세그멘테이션용
deeplabv3 = models.segmentation.deeplabv3_resnet50(weights='DEFAULT')
```

### timm 라이브러리

```python
import timm

# 사용 가능한 모델 확인
print(timm.list_models('*efficientnet*'))

# 모델 로드
model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=10)
```

---

## 6. 실전 프로젝트: 꽃 분류

### 데이터 준비

```python
# Flowers102 데이터셋
from torchvision.datasets import Flowers102

train_data = Flowers102(
    root='data',
    split='train',
    transform=train_transform,
    download=True
)

test_data = Flowers102(
    root='data',
    split='test',
    transform=val_transform
)
```

### 모델 및 학습

```python
class FlowerClassifier(nn.Module):
    def __init__(self, num_classes=102):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')

        # 마지막 층 교체
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

# 학습
model = FlowerClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
```

---

## 7. 주의사항

### 데이터 크기별 전략

| 데이터 크기 | 전략 | 설명 |
|-------------|------|------|
| 매우 적음 (<1000) | 특성 추출 | 마지막 층만 학습 |
| 적음 (1000-10000) | 점진적 해동 | 후반 층부터 해동 |
| 보통 (10000+) | 전체 미세 조정 | 낮은 학습률로 전체 학습 |

### 도메인 유사성

```
ImageNet과 유사 (동물, 사물):
    → 얕은 층도 그대로 사용 가능

ImageNet과 다름 (의료, 위성):
    → 깊은 층까지 미세 조정 필요
```

### 일반적인 실수

1. ImageNet 정규화 누락
2. 너무 높은 학습률
3. 훈련/평가 모드 전환 잊음
4. 가중치 고정 후 optimizer에 포함

---

## 8. 성능 향상 팁

### 데이터 증강

```python
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    normalize
])
```

### Label Smoothing

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

### Mixup / CutMix

```python
def mixup(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0))
    mixed_x = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return mixed_x, y_a, y_b, lam
```

---

## 정리

### 핵심 개념

1. **특성 추출**: 사전 학습 특징 재사용
2. **미세 조정**: 낮은 학습률로 전체 조정
3. **점진적 해동**: 후반 층부터 순차적 학습

### 체크리스트

- [ ] ImageNet 정규화 사용
- [ ] 적절한 학습률 선택 (1e-4 ~ 1e-5)
- [ ] model.train() / model.eval() 전환
- [ ] 데이터 증강 적용
- [ ] 조기 종료 설정

---

## 다음 단계

[08_RNN_기초.md](./08_RNN_기초.md)에서 순환 신경망을 학습합니다.
