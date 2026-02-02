# 01. 텐서와 오토그래드

## 학습 목표

- 텐서(Tensor)의 개념과 NumPy 배열과의 차이점 이해
- PyTorch의 자동 미분(Autograd) 시스템 이해
- GPU 연산의 기초

---

## 1. 텐서란?

텐서는 다차원 배열을 일반화한 개념입니다.

| 차원 | 이름 | 예시 |
|------|------|------|
| 0D | 스칼라 | 단일 숫자 (5) |
| 1D | 벡터 | [1, 2, 3] |
| 2D | 행렬 | [[1,2], [3,4]] |
| 3D | 3D 텐서 | 이미지 (H, W, C) |
| 4D | 4D 텐서 | 배치 이미지 (N, C, H, W) |

---

## 2. NumPy vs PyTorch 텐서 비교

### 생성

```python
import numpy as np
import torch

# NumPy
np_arr = np.array([1, 2, 3])
np_zeros = np.zeros((3, 4))
np_rand = np.random.randn(3, 4)

# PyTorch
pt_tensor = torch.tensor([1, 2, 3])
pt_zeros = torch.zeros(3, 4)
pt_rand = torch.randn(3, 4)
```

### 변환

```python
# NumPy → PyTorch
tensor = torch.from_numpy(np_arr)

# PyTorch → NumPy
array = tensor.numpy()  # CPU 텐서만 가능
```

### 주요 차이점

| 기능 | NumPy | PyTorch |
|------|-------|---------|
| GPU 지원 | ❌ | ✅ (`tensor.to('cuda')`) |
| 자동 미분 | ❌ | ✅ (`requires_grad=True`) |
| 기본 타입 | float64 | float32 |
| 메모리 공유 | - | `from_numpy`는 공유 |

---

## 3. 자동 미분 (Autograd)

PyTorch의 핵심 기능으로, 역전파를 자동으로 계산합니다.

### 기본 사용법

```python
# requires_grad=True로 미분 추적 활성화
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x + 1  # y = x² + 3x + 1

# 역전파 (dy/dx 계산)
y.backward()

# 기울기 확인
print(x.grad)  # tensor([7.])  # dy/dx = 2x + 3 = 2*2 + 3 = 7
```

### 계산 그래프

```
    x ─────┐
           │
    x² ────┼──▶ + ──▶ y
           │
    3x ────┘
```

- **순전파**: 입력 → 출력 방향으로 계산
- **역전파**: 출력 → 입력 방향으로 기울기 계산

### 기울기 누적과 초기화

```python
# 기울기는 누적됨
x.grad.zero_()  # 학습 루프에서 항상 초기화 필요
```

---

## 4. 연산과 브로드캐스팅

```python
a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)

# 기본 연산
c = a + b           # 요소별 덧셈
c = a * b           # 요소별 곱셈 (아다마르 곱)
c = a @ b           # 행렬 곱셈
c = torch.matmul(a, b)  # 행렬 곱셈

# 브로드캐스팅
a = torch.tensor([[1], [2], [3]])  # (3, 1)
b = torch.tensor([10, 20, 30])     # (3,)
c = a + b  # (3, 3) 자동 확장
```

---

## 5. GPU 연산

```python
# GPU 사용 가능 확인
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# 텐서를 GPU로 이동
x = torch.randn(1000, 1000)
x_gpu = x.to(device)
# 또는
x_gpu = x.cuda()

# 연산 (같은 디바이스에서 수행)
y_gpu = x_gpu @ x_gpu

# 결과를 CPU로 가져오기
y_cpu = y_gpu.cpu()
```

---

## 6. 실습: NumPy vs PyTorch 자동 미분 비교

### 문제: f(x) = x³ + 2x² - 5x + 3의 x=2에서 미분값 구하기

수학적 해:
- f'(x) = 3x² + 4x - 5
- f'(2) = 3(4) + 4(2) - 5 = 12 + 8 - 5 = 15

### NumPy (수동 미분)

```python
import numpy as np

def f(x):
    return x**3 + 2*x**2 - 5*x + 3

def df(x):
    """수동으로 미분 계산"""
    return 3*x**2 + 4*x - 5

x = 2.0
print(f"f({x}) = {f(x)}")
print(f"f'({x}) = {df(x)}")  # 15.0
```

### PyTorch (자동 미분)

```python
import torch

x = torch.tensor([2.0], requires_grad=True)
y = x**3 + 2*x**2 - 5*x + 3

y.backward()
print(f"f({x.item()}) = {y.item()}")
print(f"f'({x.item()}) = {x.grad.item()}")  # 15.0
```

---

## 7. 주의사항

### in-place 연산

```python
# in-place 연산은 autograd와 충돌할 수 있음
x = torch.tensor([1.0], requires_grad=True)
# x += 1  # 오류 발생 가능
x = x + 1  # 새 텐서 생성 (안전)
```

### 기울기 추적 비활성화

```python
# 추론 시 메모리 절약
with torch.no_grad():
    y = model(x)  # 기울기 계산 안 함

# 또는
x.requires_grad = False
```

### detach()

```python
# 계산 그래프에서 분리
y = x.detach()  # y는 기울기 추적 안 함
```

---

## 정리

### NumPy에서 이해해야 할 것
- 텐서는 다차원 배열
- 행렬 연산 (곱셈, 전치, 브로드캐스팅)

### PyTorch에서 추가되는 것
- `requires_grad`: 자동 미분 활성화
- `backward()`: 역전파 수행
- `grad`: 계산된 기울기
- GPU 가속

---

## 다음 단계

[02_Neural_Network_Basics.md](./02_Neural_Network_Basics.md)에서 이 텐서와 자동 미분을 사용해 신경망을 구축합니다.
