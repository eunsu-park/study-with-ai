# 04. 학습 기법

[이전: 역전파](./03_Backpropagation.md) | [다음: Linear & Logistic Regression](./05_Impl_Linear_Logistic.md)

---

## 학습 목표

- 경사 하강법 변형 이해 (SGD, Momentum, Adam)
- 학습률 스케줄링
- 정규화 기법 (Dropout, Weight Decay, Batch Norm)
- 과적합 방지와 조기 종료

---

## 1. 경사 하강법 (Gradient Descent)

### 기본 원리

```
W(t+1) = W(t) - η × ∇L
```
- η: 학습률 (learning rate)
- ∇L: 손실 함수의 기울기

### 변형들

| 방법 | 수식 | 특징 |
|------|------|------|
| SGD | W -= lr × g | 단순, 느림 |
| Momentum | v = βv + g; W -= lr × v | 관성 추가 |
| AdaGrad | 적응적 학습률 | 희소 데이터에 유리 |
| RMSprop | 지수 이동 평균 | AdaGrad 개선 |
| Adam | Momentum + RMSprop | 가장 보편적 |

---

## 2. Momentum

관성을 추가하여 진동을 줄입니다.

```
v(t) = β × v(t-1) + ∇L
W(t+1) = W(t) - η × v(t)
```

### NumPy 구현

```python
def sgd_momentum(W, grad, v, lr=0.01, beta=0.9):
    v = beta * v + grad          # 속도 업데이트
    W = W - lr * v               # 가중치 업데이트
    return W, v
```

### PyTorch

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

---

## 3. Adam Optimizer

Momentum과 RMSprop의 장점을 결합합니다.

```
m(t) = β₁ × m(t-1) + (1-β₁) × g      # 1차 모멘트
v(t) = β₂ × v(t-1) + (1-β₂) × g²     # 2차 모멘트
m̂ = m / (1 - β₁ᵗ)                    # 편향 보정
v̂ = v / (1 - β₂ᵗ)
W = W - η × m̂ / (√v̂ + ε)
```

### NumPy 구현

```python
def adam(W, grad, m, v, t, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad ** 2)

    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    W = W - lr * m_hat / (np.sqrt(v_hat) + eps)
    return W, m, v
```

### PyTorch

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

---

## 4. 학습률 스케줄링

학습 중 학습률을 조절합니다.

### 주요 방법

| 방법 | 특징 |
|------|------|
| Step Decay | N 에폭마다 γ 배로 감소 |
| Exponential | lr = lr₀ × γᵉᵖᵒᶜʰ |
| Cosine Annealing | 코사인 함수로 감소 |
| ReduceLROnPlateau | 검증 손실 정체 시 감소 |
| Warmup | 초기에 점진적 증가 |

### PyTorch 예시

```python
# Step Decay
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Cosine Annealing
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# ReduceLROnPlateau
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=10, factor=0.5
)

# 학습 루프에서
for epoch in range(epochs):
    train(...)
    scheduler.step()  # 에폭 끝에 호출
```

---

## 5. Dropout

학습 중 랜덤하게 뉴런을 비활성화합니다.

### 원리

```
훈련: y = x × mask / (1 - p)   # mask는 Bernoulli(1-p)
추론: y = x                     # 마스크 없음
```

### NumPy 구현

```python
def dropout(x, p=0.5, training=True):
    if not training:
        return x
    mask = (np.random.rand(*x.shape) > p).astype(float)
    return x * mask / (1 - p)
```

### PyTorch

```python
class MLPWithDropout(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_p=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # 훈련 시만 활성화
        x = self.fc2(x)
        return x

# 추론 시
model.eval()  # dropout 비활성화
```

---

## 6. Batch Normalization

각 층의 입력을 정규화합니다.

### 수식

```
μ = mean(x)
σ² = var(x)
x̂ = (x - μ) / √(σ² + ε)
y = γ × x̂ + β   # 학습 가능한 파라미터
```

### NumPy 구현

```python
def batch_norm(x, gamma, beta, eps=1e-5, training=True,
               running_mean=None, running_var=None, momentum=0.1):
    if training:
        mean = np.mean(x, axis=0)
        var = np.var(x, axis=0)

        # 이동 평균 업데이트
        if running_mean is not None:
            running_mean = momentum * mean + (1 - momentum) * running_mean
            running_var = momentum * var + (1 - momentum) * running_var
    else:
        mean = running_mean
        var = running_var

    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta
```

### PyTorch

```python
class CNNWithBatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64, 10)
        self.bn_fc = nn.BatchNorm1d(10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = x.flatten(1)
        x = self.bn_fc(self.fc1(x))
        return x
```

---

## 7. Weight Decay (L2 정규화)

가중치 크기에 패널티를 부여합니다.

### 수식

```
L_total = L_data + λ × ||W||²
∇L_total = ∇L_data + 2λW
```

### PyTorch

```python
# 방법 1: optimizer에서 설정
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# 방법 2: 손실에 직접 추가
l2_lambda = 1e-4
l2_reg = sum(p.pow(2).sum() for p in model.parameters())
loss = criterion(output, target) + l2_lambda * l2_reg
```

---

## 8. 조기 종료 (Early Stopping)

검증 손실이 개선되지 않으면 학습을 중단합니다.

### PyTorch 구현

```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# 사용
early_stopping = EarlyStopping(patience=10)
for epoch in range(epochs):
    train_loss = train(model, train_loader)
    val_loss = validate(model, val_loader)

    early_stopping(val_loss)
    if early_stopping.early_stop:
        print(f"Early stopping at epoch {epoch}")
        break
```

---

## 9. 데이터 증강 (Data Augmentation)

훈련 데이터를 변형하여 다양성을 증가시킵니다.

### 이미지 데이터

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
```

---

## 10. NumPy vs PyTorch 비교

### Optimizer 구현

```python
# NumPy (수동 구현)
m = np.zeros_like(W)
v = np.zeros_like(W)
for t in range(1, epochs + 1):
    grad = compute_gradient(W, X, y)
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad**2
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    W -= lr * m_hat / (np.sqrt(v_hat) + eps)

# PyTorch (자동)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(epochs):
    loss = criterion(model(X), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## 정리

### 핵심 개념

1. **Optimizer**: Adam이 기본 선택, SGD+Momentum도 여전히 유효
2. **학습률**: 적절한 스케줄링으로 수렴 개선
3. **정규화**: Dropout, BatchNorm, Weight Decay 조합
4. **조기 종료**: 과적합 방지의 기본

### 권장 시작 설정

```python
# 기본 설정
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
```

---

## 다음 단계

[07_CNN_Basics.md](./07_CNN_Basics.md)에서 합성곱 신경망을 학습합니다.
