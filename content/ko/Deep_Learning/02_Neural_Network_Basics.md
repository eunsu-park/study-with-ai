# 02. 신경망 기초

[이전: 텐서와 오토그래드](./01_Tensors_and_Autograd.md) | [다음: 역전파](./03_Backpropagation.md)

---

## 학습 목표

- 퍼셉트론과 다층 퍼셉트론(MLP) 이해
- 활성화 함수의 역할과 종류
- PyTorch의 `nn.Module`로 신경망 구축

---

## 1. 퍼셉트론 (Perceptron)

가장 기본적인 신경망 단위입니다.

```
입력(x₁) ──w₁──┐
               │
입력(x₂) ──w₂──┼──▶ Σ(wᵢxᵢ + b) ──▶ 활성화 ──▶ 출력(y)
               │
입력(x₃) ──w₃──┘
```

### 수식

```
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b = Σwᵢxᵢ + b
y = activation(z)
```

### NumPy 구현

```python
import numpy as np

def perceptron(x, w, b, activation):
    z = np.dot(x, w) + b
    return activation(z)

# 예시: 단순 선형 출력
x = np.array([1.0, 2.0, 3.0])
w = np.array([0.5, -0.3, 0.8])
b = 0.1

z = np.dot(x, w) + b  # 1*0.5 + 2*(-0.3) + 3*0.8 + 0.1 = 2.4
```

---

## 2. 활성화 함수 (Activation Functions)

비선형성을 추가하여 복잡한 패턴을 학습합니다.

### 주요 활성화 함수

| 함수 | 수식 | 특징 |
|------|------|------|
| Sigmoid | σ(x) = 1/(1+e⁻ˣ) | 출력 0~1, 기울기 소실 문제 |
| Tanh | tanh(x) = (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | 출력 -1~1 |
| ReLU | max(0, x) | 가장 많이 사용, 간단하고 효과적 |
| Leaky ReLU | max(αx, x) | 음수 영역에서 작은 기울기 |
| GELU | x·Φ(x) | Transformer에서 사용 |

### NumPy 구현

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)
```

### PyTorch

```python
import torch.nn.functional as F

y = F.sigmoid(x)
y = F.relu(x)
y = F.tanh(x)
```

---

## 3. 다층 퍼셉트론 (MLP)

여러 층을 쌓아 복잡한 함수를 근사합니다.

```
입력층 ──▶ 은닉층1 ──▶ 은닉층2 ──▶ 출력층
(n개)     (h1개)      (h2개)      (m개)
```

### 순전파 (Forward Pass)

```python
# 2층 MLP 순전파
z1 = x @ W1 + b1       # 첫 번째 선형 변환
a1 = relu(z1)          # 활성화
z2 = a1 @ W2 + b2      # 두 번째 선형 변환
y = softmax(z2)        # 출력 (분류의 경우)
```

---

## 4. PyTorch nn.Module

PyTorch에서 신경망을 정의하는 표준 방법입니다.

### 기본 구조

```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
```

### nn.Sequential 사용

```python
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
```

---

## 5. 가중치 초기화

적절한 초기화는 학습 성능에 큰 영향을 미칩니다.

| 방법 | 특징 | 사용 |
|------|------|------|
| Xavier/Glorot | Sigmoid, Tanh에 적합 | `nn.init.xavier_uniform_` |
| He/Kaiming | ReLU에 적합 | `nn.init.kaiming_uniform_` |
| 영 초기화 | 사용 금지 (대칭성 문제) | - |

```python
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        nn.init.zeros_(m.bias)

model.apply(init_weights)
```

---

## 6. 실습: XOR 문제 해결

단층 퍼셉트론으로 해결 불가능한 XOR 문제를 MLP로 해결합니다.

### 데이터

```
입력      출력
(0, 0) → 0
(0, 1) → 1
(1, 0) → 1
(1, 1) → 0
```

### MLP 구조

```
입력(2) ──▶ 은닉(4) ──▶ 출력(1)
```

### PyTorch 구현

```python
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
    nn.Sigmoid()
)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

for epoch in range(1000):
    pred = model(X)
    loss = criterion(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## 7. NumPy vs PyTorch 비교

### MLP 순전파

```python
# NumPy (수동)
def forward_numpy(x, W1, b1, W2, b2):
    z1 = x @ W1 + b1
    a1 = np.maximum(0, z1)  # ReLU
    z2 = a1 @ W2 + b2
    return z2

# PyTorch (자동)
class MLP(nn.Module):
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

### 핵심 차이

| 항목 | NumPy | PyTorch |
|------|-------|---------|
| 순전파 | 직접 구현 | `forward()` 메서드 |
| 역전파 | 직접 미분 계산 | `loss.backward()` 자동 |
| 파라미터 관리 | 배열로 직접 관리 | `model.parameters()` |

---

## 정리

### 핵심 개념

1. **퍼셉트론**: 선형 변환 + 활성화 함수
2. **활성화 함수**: 비선형성 추가 (ReLU 권장)
3. **MLP**: 여러 층을 쌓아 복잡한 함수 학습
4. **nn.Module**: PyTorch의 신경망 기본 클래스

### NumPy로 구현하면서 배우는 것

- 행렬 연산의 의미
- 활성화 함수의 수학적 정의
- 순전파의 데이터 흐름

---

## 다음 단계

[03_Backpropagation.md](./03_Backpropagation.md)에서 역전파 알고리즘을 NumPy로 직접 구현합니다.
