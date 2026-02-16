# 05. CNN 기초 (Convolutional Neural Networks)

## 학습 목표

- 합성곱 연산의 원리 이해
- 풀링, 패딩, 스트라이드 개념
- PyTorch로 CNN 구현
- MNIST/CIFAR-10 분류

---

## 1. 합성곱 (Convolution) 연산

### 개념

이미지의 지역적 패턴(에지, 텍스처)을 감지합니다.

```
입력 이미지     필터(커널)      출력
[1 2 3 4]      [1 0]          [?]
[5 6 7 8]  *   [0 1]   =
[9 0 1 2]
```

### 수식

```
출력[i,j] = Σ Σ 입력[i+m, j+n] × 필터[m, n]
```

### 차원 계산

```
출력 크기 = (입력 - 커널 + 2×패딩) / 스트라이드 + 1

예: 입력 32×32, 커널 3×3, 패딩 1, 스트라이드 1
    = (32 - 3 + 2) / 1 + 1 = 32
```

---

## 2. 주요 개념

### 패딩 (Padding)

```
입력 테두리에 0을 추가하여 출력 크기 유지

padding='same': 출력 = 입력 크기
padding='valid': 패딩 없음 (출력 < 입력)
```

### 스트라이드 (Stride)

```
필터 이동 간격

stride=1: 한 칸씩 이동 (기본)
stride=2: 두 칸씩 이동 → 출력 크기 절반
```

### 풀링 (Pooling)

```
공간 크기 축소, 불변성 증가

Max Pooling: 영역 내 최대값
Avg Pooling: 영역 내 평균값
```

---

## 3. CNN 구조

### 기본 구조

```
입력 → [Conv → ReLU → Pool] × N → Flatten → FC → 출력
```

### LeNet-5 (1998)

```
입력 (32×32×1)
  ↓
Conv1 (5×5, 6채널) → 28×28×6
  ↓
MaxPool (2×2) → 14×14×6
  ↓
Conv2 (5×5, 16채널) → 10×10×16
  ↓
MaxPool (2×2) → 5×5×16
  ↓
Flatten → 400
  ↓
FC → 120 → 84 → 10
```

---

## 4. PyTorch Conv2d

### 기본 사용법

```python
import torch.nn as nn

# Conv2d(입력채널, 출력채널, 커널크기, stride, padding)
conv = nn.Conv2d(
    in_channels=3,      # RGB 이미지
    out_channels=64,    # 64개 필터
    kernel_size=3,      # 3×3 커널
    stride=1,
    padding=1           # same padding
)

# 입력: (batch, channels, height, width)
x = torch.randn(1, 3, 32, 32)
out = conv(x)  # (1, 64, 32, 32)
```

### MaxPool2d

```python
pool = nn.MaxPool2d(kernel_size=2, stride=2)
# 32×32 → 16×16
```

---

## 5. MNIST CNN 구현

### 모델 정의

```python
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv 블록 1
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Conv 블록 2
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        # FC 블록
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # x: (batch, 1, 28, 28)
        x = F.relu(self.conv1(x))  # (batch, 32, 28, 28)
        x = self.pool1(x)          # (batch, 32, 14, 14)

        x = F.relu(self.conv2(x))  # (batch, 64, 14, 14)
        x = self.pool2(x)          # (batch, 64, 7, 7)

        x = x.view(-1, 64 * 7 * 7) # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

### 학습 코드

```python
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 데이터 로드
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = datasets.MNIST('data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# 모델, 손실, 옵티마이저
model = MNISTNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 학습
for epoch in range(5):
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## 6. 특징 맵 시각화

```python
def visualize_feature_maps(model, image):
    """첫 번째 Conv 층의 특징 맵 시각화"""
    model.eval()
    with torch.no_grad():
        # 첫 번째 Conv 출력
        x = model.conv1(image)
        x = F.relu(x)

    # 그리드로 표시
    fig, axes = plt.subplots(4, 8, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        if i < x.shape[1]:
            ax.imshow(x[0, i].cpu().numpy(), cmap='viridis')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('feature_maps.png')
```

---

## 7. NumPy로 합성곱 이해 (참고)

```python
def conv2d_numpy(image, kernel):
    """NumPy로 2D 합성곱 구현 (교육용)"""
    h, w = image.shape
    kh, kw = kernel.shape
    oh, ow = h - kh + 1, w - kw + 1

    output = np.zeros((oh, ow))

    for i in range(oh):
        for j in range(ow):
            # 영역 추출
            region = image[i:i+kh, j:j+kw]
            # 요소별 곱셈 후 합산
            output[i, j] = np.sum(region * kernel)

    return output

# Sobel 에지 검출 예시
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

edges = conv2d_numpy(image, sobel_x)
```

> **참고**: 실제 CNN에서는 PyTorch의 최적화된 구현을 사용합니다.

---

## 8. 배치 정규화와 Dropout

### CNN에서 사용

```python
class CNNWithBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Conv용 BN
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(0.25)  # 2D Dropout

        self.fc1 = nn.Linear(32 * 14 * 14, 128)
        self.bn_fc = nn.BatchNorm1d(128)  # FC용 BN
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = x.view(-1, 32 * 14 * 14)
        x = self.fc1(x)
        x = self.bn_fc(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
```

---

## 9. CIFAR-10 분류

### 데이터

- 32×32 RGB 이미지
- 10개 클래스: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

### 모델

```python
class CIFAR10Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32→16

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16→8
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 128 * 8 * 8)
        x = self.classifier(x)
        return x
```

---

## 10. 정리

### 핵심 개념

1. **합성곱**: 지역 패턴 추출, 파라미터 공유
2. **풀링**: 공간 축소, 불변성 증가
3. **채널**: 다양한 특징 학습
4. **계층적 학습**: 저수준 → 고수준 특징

### CNN vs MLP

| 항목 | MLP | CNN |
|------|-----|-----|
| 연결 | 완전 연결 | 지역 연결 |
| 파라미터 | 많음 | 적음 (공유) |
| 공간 정보 | 무시 | 보존 |
| 이미지 | 비효율적 | 효율적 |

### 다음 단계

[08_CNN_Advanced.md](./08_CNN_Advanced.md)에서 ResNet, VGG 등 유명 아키텍처를 학습합니다.
