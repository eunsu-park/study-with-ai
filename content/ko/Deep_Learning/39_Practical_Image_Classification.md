[이전: 객체 탐지](./38_Object_Detection.md) | [다음: 실전 텍스트 분류 프로젝트](./40_Practical_Text_Classification.md)

---

# 39. 실전 이미지 분류 프로젝트

## 학습 목표

- CIFAR-10 분류 프로젝트 완성
- 데이터 증강 전략
- 학습 파이프라인 구축
- 성능 개선 기법 적용

---

## 1. 프로젝트 개요

### CIFAR-10 데이터셋

```
- 60,000장 (32×32 RGB)
- 10 클래스: airplane, automobile, bird, cat, deer,
            dog, frog, horse, ship, truck
- 훈련: 50,000 / 테스트: 10,000
```

### 목표 정확도

| 모델 | 목표 정확도 |
|------|------------|
| 간단한 CNN | 70-75% |
| ResNet-like | 85-90% |
| 전이학습 | 90%+ |

---

## 2. 데이터 준비

### 로드 및 전처리

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 정규화 값 (CIFAR-10)
mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)

# 훈련 변환 (증강 포함)
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# 테스트 변환 (증강 없음)
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# 데이터셋
train_data = datasets.CIFAR10('data', train=True, download=True,
                               transform=train_transform)
test_data = datasets.CIFAR10('data', train=False,
                              transform=test_transform)

# 로더
train_loader = DataLoader(train_data, batch_size=128, shuffle=True,
                          num_workers=4, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=256)
```

---

## 3. 모델 정의

### 기본 CNN

```python
class CIFAR10CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 32 → 16
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Block 2: 16 → 8
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Block 3: 8 → 4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
```

### ResNet Block

```python
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)
```

---

## 4. 학습 파이프라인

### 전체 코드

```python
def train_cifar10():
    # 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 100
    lr = 0.1

    # 모델
    model = CIFAR10CNN().to(device)

    # 손실 함수
    criterion = nn.CrossEntropyLoss()

    # 옵티마이저
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=5e-4
    )

    # 스케줄러
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    # 학습
    best_acc = 0
    for epoch in range(epochs):
        # 훈련
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

        train_acc = 100. * correct / total

        # 테스트
        model.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)

        test_acc = 100. * correct / total

        scheduler.step()

        print(f"Epoch {epoch+1}: Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%")

        # 최고 모델 저장
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_model.pth')

    print(f"\n최고 테스트 정확도: {best_acc:.2f}%")
```

---

## 5. 성능 개선 기법

### Mixup

```python
def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# 학습 루프에서
mixed_x, y_a, y_b, lam = mixup_data(data, target)
output = model(mixed_x)
loss = mixup_criterion(criterion, output, y_a, y_b, lam)
```

### CutMix

```python
def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    # 랜덤 박스
    W, H = x.size(3), x.size(2)
    cut_w = int(W * np.sqrt(1 - lam))
    cut_h = int(H * np.sqrt(1 - lam))
    cx, cy = np.random.randint(W), np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)

    return x, y, y[index], lam
```

### Label Smoothing

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

---

## 6. 결과 분석

### 혼동 행렬

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(model, test_loader, classes):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data.to(device))
            preds = output.argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(target.tolist())

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
```

### 클래스별 정확도

```python
def per_class_accuracy(model, test_loader, classes):
    model.eval()
    class_correct = [0] * 10
    class_total = [0] * 10

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data.to(device))
            preds = output.argmax(dim=1).cpu()
            for i in range(len(target)):
                label = target[i].item()
                class_total[label] += 1
                if preds[i] == label:
                    class_correct[label] += 1

    for i, cls in enumerate(classes):
        acc = 100 * class_correct[i] / class_total[i]
        print(f"{cls}: {acc:.2f}%")
```

---

## 7. 전이학습 적용

```python
import torchvision.models as models

# 사전 학습 모델
model = models.resnet18(weights='IMAGENET1K_V1')

# 마지막 층 수정
model.fc = nn.Linear(model.fc.in_features, 10)

# 첫 번째 Conv 수정 (CIFAR: 32×32)
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()  # 풀링 제거

model = model.to(device)
```

---

## 정리

### 체크리스트

- [ ] 데이터 증강 적용
- [ ] BatchNorm + Dropout 사용
- [ ] 적절한 스케줄러 (Cosine Annealing)
- [ ] Weight Decay 사용
- [ ] Mixup/CutMix 고려
- [ ] 모델 저장 및 분석

### 예상 결과

| 기법 | 테스트 정확도 |
|------|--------------|
| 기본 CNN | 75-80% |
| + 데이터 증강 | 80-85% |
| + Mixup | 85-88% |
| ResNet + 전이학습 | 90%+ |

---

## 다음 단계

[40_Practical_Text_Classification.md](./40_Practical_Text_Classification.md)에서 텍스트 분류 프로젝트를 진행합니다.
