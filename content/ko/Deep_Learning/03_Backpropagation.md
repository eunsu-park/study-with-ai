# 03. 역전파 이해

[이전: 신경망 기초](./02_Neural_Network_Basics.md) | [다음: 훈련 기법](./04_Training_Techniques.md)

---

## 학습 목표

- 역전파(Backpropagation) 알고리즘의 원리 이해
- 체인 룰(Chain Rule)을 이용한 기울기 계산
- NumPy로 역전파 직접 구현

---

## 1. 역전파란?

역전파는 신경망의 가중치를 학습하기 위한 알고리즘입니다.

```
순전파 (Forward):  입력 ──▶ 은닉층 ──▶ 출력 ──▶ 손실
역전파 (Backward): 입력 ◀── 은닉층 ◀── 출력 ◀── 손실
```

### 핵심 아이디어

1. **순전파**: 입력에서 출력까지 값 계산
2. **손실 계산**: 예측과 정답의 차이
3. **역전파**: 손실에서 입력 방향으로 기울기 전파
4. **가중치 업데이트**: 기울기를 이용해 가중치 조정

---

## 2. 체인 룰 (Chain Rule)

합성 함수의 미분 법칙입니다.

### 수식

```
y = f(g(x))

dy/dx = (dy/dg) × (dg/dx)
```

### 예시

```
z = x²
y = sin(z)
L = y²

dL/dx = (dL/dy) × (dy/dz) × (dz/dx)
      = 2y × cos(z) × 2x
```

---

## 3. 단일 뉴런의 역전파

### 순전파

```python
z = w*x + b      # 선형 변환
a = sigmoid(z)    # 활성화
L = (a - y)²     # 손실 (MSE)
```

### 역전파 (기울기 계산)

```python
dL/da = 2(a - y)                    # 손실의 활성화에 대한 기울기
da/dz = sigmoid(z) * (1 - sigmoid(z))  # 시그모이드 미분
dz/dw = x                           # 선형 변환의 가중치에 대한 기울기
dz/db = 1                           # 선형 변환의 편향에 대한 기울기

# 체인 룰 적용
dL/dw = (dL/da) × (da/dz) × (dz/dw)
dL/db = (dL/da) × (da/dz) × (dz/db)
```

---

## 4. 손실 함수

### MSE (Mean Squared Error)

```python
L = (1/n) × Σ(y_pred - y_true)²
dL/dy_pred = (2/n) × (y_pred - y_true)
```

### Cross-Entropy (분류)

```python
L = -Σ y_true × log(y_pred)
dL/dy_pred = -y_true / y_pred  # softmax와 결합 시 간단해짐
```

### Softmax + Cross-Entropy 결합

```python
# 놀라운 결과: 매우 간단해짐
dL/dz = y_pred - y_true  # softmax 입력에 대한 기울기
```

---

## 5. MLP 역전파

2층 MLP의 역전파 과정입니다.

### 구조

```
입력(x) → [W1, b1] → ReLU → [W2, b2] → 출력(y)
```

### 순전파

```python
z1 = x @ W1 + b1
a1 = relu(z1)
z2 = a1 @ W2 + b2
y_pred = z2  # 또는 softmax(z2)
```

### 역전파

```python
# 출력층
dL/dz2 = y_pred - y_true  # (softmax + CE의 경우)
dL/dW2 = a1.T @ dL/dz2
dL/db2 = sum(dL/dz2, axis=0)

# 은닉층
dL/da1 = dL/dz2 @ W2.T
dL/dz1 = dL/da1 * relu_derivative(z1)
dL/dW1 = x.T @ dL/dz1
dL/db1 = sum(dL/dz1, axis=0)
```

---

## 6. NumPy 구현 핵심

```python
class MLP:
    def backward(self, x, y_true, y_pred, cache):
        """역전파: 기울기 계산"""
        a1, z1 = cache

        # 출력층 기울기
        dz2 = y_pred - y_true
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0)

        # 은닉층 기울기 (체인 룰)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (z1 > 0)  # ReLU 미분
        dW1 = x.T @ dz1
        db1 = np.sum(dz1, axis=0)

        return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
```

---

## 7. PyTorch의 자동 미분

PyTorch에서는 이 모든 과정이 자동입니다.

```python
# 순전파
y_pred = model(x)
loss = criterion(y_pred, y_true)

# 역전파 (자동!)
loss.backward()

# 기울기 접근
print(model.fc1.weight.grad)
```

### 계산 그래프

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
z = y * 3
z.backward()

# x.grad = dz/dx = dz/dy × dy/dx = 3 × 2x = 12
```

---

## 8. 기울기 소실/폭발 문제

### 기울기 소실 (Vanishing Gradient)

- 원인: 시그모이드/tanh의 미분이 0에 가까움
- 해결: ReLU, 잔차 연결(Residual Connection)

### 기울기 폭발 (Exploding Gradient)

- 원인: 깊은 네트워크에서 기울기 누적
- 해결: Gradient Clipping, Batch Normalization

```python
# PyTorch에서 Gradient Clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## 9. 수치 기울기 검증

역전파 구현이 올바른지 확인하는 방법입니다.

```python
def numerical_gradient(f, x, h=1e-5):
    """수치 미분으로 기울기 계산"""
    grad = np.zeros_like(x)
    for i in range(x.size):
        x_plus = x.copy()
        x_plus.flat[i] += h
        x_minus = x.copy()
        x_minus.flat[i] -= h
        grad.flat[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad

# 검증
analytical_grad = backward(...)  # 해석적 기울기
numerical_grad = numerical_gradient(loss_fn, weights)
diff = np.linalg.norm(analytical_grad - numerical_grad)
assert diff < 1e-5, "Gradient check failed!"
```

---

## 정리

### 역전파의 핵심

1. **체인 룰**: 합성 함수 미분의 핵심
2. **국소적 계산**: 각 층에서 독립적으로 기울기 계산
3. **기울기 전파**: 출력에서 입력 방향으로 전파

### NumPy로 배우는 것

- 행렬의 전치와 곱셈의 의미
- 활성화 함수 미분의 역할
- 배치 처리에서의 기울기 합산

### PyTorch로 넘어가면

- `loss.backward()` 한 줄로 모든 기울기 계산
- 계산 그래프 자동 구성
- GPU 가속

---

## 다음 단계

[04_Training_Techniques.md](./04_Training_Techniques.md)에서 기울기를 이용한 가중치 업데이트 방법을 학습합니다.
