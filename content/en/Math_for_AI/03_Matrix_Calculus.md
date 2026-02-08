# 03. Matrix Calculus

## Learning Objectives

- Understand and compute scalar-by-vector derivatives and gradients
- Learn the definition of the Jacobian matrix and how to apply the chain rule
- Understand the meaning of the Hessian matrix and its role in optimization
- Derive and utilize key matrix derivative identities
- Directly derive gradients of machine learning loss functions
- Understand PyTorch's automatic differentiation capabilities and use them for verification

---

## 1. Scalar-by-Vector Derivatives

### 1.1 Definition of Gradient

For a scalar function $f: \mathbb{R}^n \to \mathbb{R}$, the gradient is a vector containing all partial derivatives:

$$\nabla f(\mathbf{x}) = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}$$

The gradient points in the direction of steepest increase of the function.

### 1.2 Basic Examples

**Example 1**: $f(\mathbf{x}) = \mathbf{a}^T \mathbf{x}$

$$\frac{\partial}{\partial \mathbf{x}}(\mathbf{a}^T \mathbf{x}) = \mathbf{a}$$

**Example 2**: $f(\mathbf{x}) = \mathbf{x}^T \mathbf{x}$

$$\frac{\partial}{\partial \mathbf{x}}(\mathbf{x}^T \mathbf{x}) = 2\mathbf{x}$$

**Example 3**: $f(\mathbf{x}) = \mathbf{x}^T A \mathbf{x}$ (quadratic form)

$$\frac{\partial}{\partial \mathbf{x}}(\mathbf{x}^T A \mathbf{x}) = (A + A^T)\mathbf{x}$$

If $A$ is symmetric, this becomes $2A\mathbf{x}$.

### 1.3 Python Implementation: Derivative of Quadratic Forms

```python
import numpy as np
import torch
from sympy import symbols, Matrix, diff, simplify

# SymPy로 심볼릭 계산
print("=== SymPy 심볼릭 미분 ===")
x1, x2 = symbols('x1 x2')
x = Matrix([x1, x2])
A = Matrix([[2, 1], [1, 3]])

# f(x) = x^T A x
f = (x.T * A * x)[0]
print(f"f(x) = {f}")

# 그래디언트 계산
grad_f = Matrix([diff(f, x1), diff(f, x2)])
print(f"∇f = {simplify(grad_f)}")
print(f"(A + A^T)x = {simplify((A + A.T) * x)}")

# PyTorch로 수치 계산 및 검증
print("\n=== PyTorch 자동 미분 ===")
x_val = torch.tensor([1.0, 2.0], requires_grad=True)
A_torch = torch.tensor([[2.0, 1.0], [1.0, 3.0]])

# 순전파
f_val = x_val @ A_torch @ x_val
print(f"f(x) = {f_val.item():.4f}")

# 역전파
f_val.backward()
print(f"∇f (autograd) = {x_val.grad}")

# 공식으로 계산
grad_formula = (A_torch + A_torch.T) @ x_val.detach()
print(f"∇f (공식)      = {grad_formula}")
print(f"차이: {torch.norm(x_val.grad - grad_formula).item():.2e}")
```

### 1.4 Numerator Layout vs Denominator Layout

There are two conventions in matrix calculus:

- **Numerator layout**: The $(i,j)$ element of $\frac{\partial \mathbf{y}}{\partial \mathbf{x}}$ is $\frac{\partial y_i}{\partial x_j}$
- **Denominator layout**: The $(i,j)$ element of $\frac{\partial \mathbf{y}}{\partial \mathbf{x}}$ is $\frac{\partial y_j}{\partial x_i}$

This document uses the numerator layout.

## 2. Vector-by-Vector Derivatives: Jacobian

### 2.1 Definition of Jacobian Matrix

For a vector function $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$, the Jacobian matrix is:

$$J = \frac{\partial \mathbf{f}}{\partial \mathbf{x}} = \begin{bmatrix} \frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n} \end{bmatrix}$$

The size is $m \times n$.

### 2.2 Chain Rule with Jacobian

When $\mathbf{z} = \mathbf{g}(\mathbf{f}(\mathbf{x}))$:

$$\frac{\partial \mathbf{z}}{\partial \mathbf{x}} = \frac{\partial \mathbf{z}}{\partial \mathbf{y}} \frac{\partial \mathbf{y}}{\partial \mathbf{x}}$$

where $\mathbf{y} = \mathbf{f}(\mathbf{x})$, and the right-hand side is the product of Jacobian matrices.

### 2.3 Jacobian Computation Example

```python
import torch

# 함수 정의: f: R^2 -> R^3
def vector_function(x):
    """
    f([x1, x2]) = [x1^2 + x2,
                   x1 * x2,
                   sin(x1) + cos(x2)]
    """
    return torch.stack([
        x[0]**2 + x[1],
        x[0] * x[1],
        torch.sin(x[0]) + torch.cos(x[1])
    ])

x = torch.tensor([1.0, 2.0], requires_grad=True)

# PyTorch의 야코비안 계산
from torch.autograd.functional import jacobian

J = jacobian(vector_function, x)
print("야코비안 행렬 (3x2):")
print(J)

# 수동 계산으로 검증
x1, x2 = x[0].item(), x[1].item()
J_manual = torch.tensor([
    [2*x1, 1],
    [x2, x1],
    [np.cos(x1), -np.sin(x2)]
])
print("\n수동 계산:")
print(J_manual)
print(f"\n차이: {torch.norm(J - J_manual).item():.2e}")
```

### 2.4 Chain Rule Practice

```python
# 합성 함수의 야코비안: h(x) = g(f(x))
def f(x):
    """f: R^2 -> R^2"""
    return torch.stack([x[0]**2, x[0] + x[1]])

def g(y):
    """g: R^2 -> R^2"""
    return torch.stack([y[0] * y[1], y[0] - y[1]])

def h(x):
    """h = g ∘ f"""
    return g(f(x))

x = torch.tensor([1.0, 2.0])

# 방법 1: 직접 계산
J_h = jacobian(h, x)
print("J_h (직접):")
print(J_h)

# 방법 2: 체인 룰
J_f = jacobian(f, x)
y = f(x)
J_g = jacobian(g, y)
J_chain = J_g @ J_f
print("\nJ_g @ J_f (체인 룰):")
print(J_chain)

print(f"\n차이: {torch.norm(J_h - J_chain).item():.2e}")
```

## 3. Hessian Matrix

### 3.1 Definition of Hessian

The Hessian matrix of a scalar function $f: \mathbb{R}^n \to \mathbb{R}$ consists of second-order partial derivatives:

$$H = \nabla^2 f = \begin{bmatrix} \frac{\partial^2 f}{\partial x_1^2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial^2 f}{\partial x_n \partial x_1} & \cdots & \frac{\partial^2 f}{\partial x_n^2} \end{bmatrix}$$

By Schwarz's theorem, if $f$ is $C^2$ class, then $H$ is symmetric.

### 3.2 Properties of Hessian and Optimization

- **Positive Definite**: All eigenvalues > 0 → local minimum
- **Negative Definite**: All eigenvalues < 0 → local maximum
- **Indefinite**: Mixed positive/negative eigenvalues → saddle point

### 3.3 Role in Newton's Method

Newton's method update rule:

$$\mathbf{x}_{k+1} = \mathbf{x}_k - H^{-1}(\mathbf{x}_k) \nabla f(\mathbf{x}_k)$$

It uses the inverse of the Hessian to leverage second-order information.

### 3.4 Hessian Computation Example

```python
import torch
import numpy as np

# 함수 정의: f(x, y) = x^2 + xy + 2y^2
def f(x):
    return x[0]**2 + x[0]*x[1] + 2*x[1]**2

x = torch.tensor([1.0, 2.0], requires_grad=True)

# 그래디언트 계산
y = f(x)
grad = torch.autograd.grad(y, x, create_graph=True)[0]
print("∇f =", grad)

# 헤시안 계산 (각 그래디언트 성분을 다시 미분)
hessian = torch.zeros(2, 2)
for i in range(2):
    hessian[i] = torch.autograd.grad(grad[i], x, retain_graph=True)[0]

print("\n헤시안 행렬:")
print(hessian)

# 수동 계산: H = [[2, 1], [1, 4]]
H_manual = torch.tensor([[2.0, 1.0], [1.0, 4.0]])
print("\n수동 계산:")
print(H_manual)

# 고유값으로 정부호 판정
eigenvalues = torch.linalg.eigvalsh(hessian)
print(f"\n고유값: {eigenvalues}")
print("양정치 (극솟값):", torch.all(eigenvalues > 0).item())
```

### 3.5 Hessian and Convexity

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 볼록 함수: f(x, y) = x^2 + 2y^2
x = np.linspace(-3, 3, 50)
y = np.linspace(-3, 3, 50)
X, Y = np.meshgrid(x, y)
Z_convex = X**2 + 2*Y**2

# 안장점 함수: f(x, y) = x^2 - y^2
Z_saddle = X**2 - Y**2

fig = plt.figure(figsize=(14, 6))

# 볼록 함수
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z_convex, cmap='viridis', alpha=0.8)
ax1.set_title('볼록 함수: $f(x,y) = x^2 + 2y^2$\n헤시안 양정치', fontsize=12)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('f(x,y)')

# 안장점 함수
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X, Y, Z_saddle, cmap='plasma', alpha=0.8)
ax2.set_title('안장점: $f(x,y) = x^2 - y^2$\n헤시안 부정부호', fontsize=12)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('f(x,y)')

plt.tight_layout()
plt.savefig('hessian_surfaces.png', dpi=150)
plt.close()

print("헤시안과 함수 형태 시각화 저장: hessian_surfaces.png")
```

## 4. Matrix Derivative Identities

### 4.1 Key Identities

| Function | Derivative |
|----------|-----------|
| $\mathbf{a}^T \mathbf{x}$ | $\mathbf{a}$ |
| $\mathbf{x}^T \mathbf{x}$ | $2\mathbf{x}$ |
| $\mathbf{x}^T A \mathbf{x}$ | $(A + A^T)\mathbf{x}$ |
| $\mathbf{a}^T X \mathbf{b}$ | $\mathbf{a}\mathbf{b}^T$ |
| $\text{tr}(AB)$ | $B^T$ (w.r.t. $A$) |
| $\log \|A\|$ | $A^{-T}$ (transpose of inverse) |
| $\mathbf{x}^T A^{-1} \mathbf{x}$ | $-A^{-1}\mathbf{x}\mathbf{x}^T A^{-1}$ (w.r.t. $A$) |

### 4.2 Derivation: $\mathbf{x}^T A \mathbf{x}$

Using index notation:

$$f(\mathbf{x}) = \mathbf{x}^T A \mathbf{x} = \sum_{i,j} x_i A_{ij} x_j$$

Taking derivative with respect to $x_k$:

$$\frac{\partial f}{\partial x_k} = \sum_j A_{kj} x_j + \sum_i x_i A_{ik} = (A\mathbf{x})_k + (A^T\mathbf{x})_k$$

Therefore:

$$\nabla f = (A + A^T)\mathbf{x}$$

### 4.3 Derivation: $\text{tr}(AB)$

Using the trace property $\text{tr}(AB) = \sum_{ij} A_{ij} B_{ji}$:

$$\frac{\partial}{\partial A_{kl}} \text{tr}(AB) = \frac{\partial}{\partial A_{kl}} \sum_{ij} A_{ij} B_{ji} = B_{lk}$$

Therefore:

$$\frac{\partial \text{tr}(AB)}{\partial A} = B^T$$

### 4.4 Identity Verification Code

```python
import torch

# 항등식 1: ∂(x^T a)/∂x = a
x = torch.randn(5, requires_grad=True)
a = torch.randn(5)
f = x @ a
f.backward()
print("항등식 1: ∂(x^T a)/∂x = a")
print(f"autograd: {x.grad}")
print(f"공식:     {a}")
print(f"차이: {torch.norm(x.grad - a).item():.2e}\n")

# 항등식 2: ∂(x^T A x)/∂x = (A + A^T)x
x = torch.randn(5, requires_grad=True)
A = torch.randn(5, 5)
f = x @ A @ x
f.backward()
print("항등식 2: ∂(x^T A x)/∂x = (A + A^T)x")
print(f"autograd: {x.grad}")
expected = (A + A.T) @ x.detach()
print(f"공식:     {expected}")
print(f"차이: {torch.norm(x.grad - expected).item():.2e}\n")

# 항등식 3: ∂tr(AB)/∂A = B^T
A = torch.randn(4, 4, requires_grad=True)
B = torch.randn(4, 4)
f = torch.trace(A @ B)
f.backward()
print("항등식 3: ∂tr(AB)/∂A = B^T")
print(f"autograd:\n{A.grad}")
print(f"공식:\n{B.T}")
print(f"차이: {torch.norm(A.grad - B.T).item():.2e}")
```

## 5. Applications of Matrix Calculus in ML

### 5.1 MSE Loss Gradient Derivation

In regression problems, the loss function is:

$$L(\mathbf{w}) = \frac{1}{2n} \|\mathbf{y} - X\mathbf{w}\|^2$$

Gradient:

$$\nabla_\mathbf{w} L = -\frac{1}{n} X^T (\mathbf{y} - X\mathbf{w})$$

Derivation:

$$\nabla_\mathbf{w} L = \nabla_\mathbf{w} \frac{1}{2n}(\mathbf{y} - X\mathbf{w})^T(\mathbf{y} - X\mathbf{w})$$

Let $\mathbf{r} = \mathbf{y} - X\mathbf{w}$:

$$\nabla_\mathbf{w} L = \frac{1}{n} \nabla_\mathbf{w} \mathbf{r}^T \mathbf{r} = \frac{1}{n} \cdot 2 \mathbf{r}^T \nabla_\mathbf{w} \mathbf{r} = -\frac{1}{n} X^T \mathbf{r}$$

### 5.2 MSE Gradient Implementation and Verification

```python
import torch
import torch.nn as nn

# 데이터 생성
n, d = 100, 10
X = torch.randn(n, d)
y = torch.randn(n)
w = torch.randn(d, requires_grad=True)

# 방법 1: PyTorch autograd
pred = X @ w
loss = 0.5 * torch.mean((y - pred)**2)
loss.backward()
grad_autograd = w.grad.clone()

# 방법 2: 수동 유도 공식
residual = y - X @ w.detach()
grad_formula = -X.T @ residual / n

print("MSE 그래디언트 비교:")
print(f"autograd: {grad_autograd[:5]}")
print(f"공식:     {grad_formula[:5]}")
print(f"차이: {torch.norm(grad_autograd - grad_formula).item():.2e}")
```

### 5.3 Softmax Cross-Entropy Gradient

Softmax function:

$$\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

Cross-entropy loss:

$$L = -\sum_i y_i \log \sigma(\mathbf{z})_i$$

Gradient (for one-hot label $\mathbf{y}$):

$$\frac{\partial L}{\partial \mathbf{z}} = \sigma(\mathbf{z}) - \mathbf{y}$$

This concise form is derived from computing the Jacobian of the softmax.

### 5.4 Softmax Gradient Verification

```python
import torch
import torch.nn.functional as F

# 로짓과 타겟
logits = torch.randn(5, requires_grad=True)
target_class = 2  # 클래스 2가 정답

# 방법 1: PyTorch autograd
loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([target_class]))
loss.backward()
grad_autograd = logits.grad.clone()

# 방법 2: 수동 계산
probs = F.softmax(logits.detach(), dim=0)
y_onehot = torch.zeros(5)
y_onehot[target_class] = 1.0
grad_formula = probs - y_onehot

print("소프트맥스 교차 엔트로피 그래디언트:")
print(f"autograd: {grad_autograd}")
print(f"공식:     {grad_formula}")
print(f"차이: {torch.norm(grad_autograd - grad_formula).item():.2e}")
```

### 5.5 Backpropagation: Chain of Jacobians

Backpropagation in neural networks is the repeated application of the chain rule:

$$\frac{\partial L}{\partial \mathbf{w}_1} = \frac{\partial L}{\partial \mathbf{z}_L} \frac{\partial \mathbf{z}_L}{\partial \mathbf{z}_{L-1}} \cdots \frac{\partial \mathbf{z}_2}{\partial \mathbf{z}_1} \frac{\partial \mathbf{z}_1}{\partial \mathbf{w}_1}$$

Each term is a Jacobian, computed from right to left (reverse mode).

### 5.6 Linear Layer Gradient Derivation

Linear layer: $\mathbf{z} = W\mathbf{x} + \mathbf{b}$

Gradients with respect to loss $L$:

$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial \mathbf{z}} \mathbf{x}^T$$

$$\frac{\partial L}{\partial \mathbf{b}} = \frac{\partial L}{\partial \mathbf{z}}$$

$$\frac{\partial L}{\partial \mathbf{x}} = W^T \frac{\partial L}{\partial \mathbf{z}}$$

```python
# 선형 레이어 그래디언트 수동 구현
class LinearLayer:
    def __init__(self, in_dim, out_dim):
        self.W = torch.randn(out_dim, in_dim, requires_grad=False)
        self.b = torch.randn(out_dim, requires_grad=False)
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        return self.W @ x + self.b

    def backward(self, dL_dz):
        """dL_dz: 손실에 대한 출력의 그래디언트"""
        self.dW = torch.outer(dL_dz, self.x)  # (out_dim, in_dim)
        self.db = dL_dz  # (out_dim,)
        dL_dx = self.W.T @ dL_dz  # (in_dim,)
        return dL_dx

# 테스트
layer = LinearLayer(5, 3)
x = torch.randn(5)
z = layer.forward(x)
dL_dz = torch.randn(3)  # 가짜 그래디언트
dL_dx = layer.backward(dL_dz)

print("선형 레이어 역전파:")
print(f"dW 형태: {layer.dW.shape}")
print(f"db 형태: {layer.db.shape}")
print(f"dx 형태: {dL_dx.shape}")

# PyTorch로 검증
W_torch = layer.W.clone().requires_grad_(True)
b_torch = layer.b.clone().requires_grad_(True)
x_torch = x.clone().requires_grad_(True)

z_torch = W_torch @ x_torch + b_torch
z_torch.backward(dL_dz)

print(f"\ndW 차이: {torch.norm(layer.dW - W_torch.grad).item():.2e}")
print(f"db 차이: {torch.norm(layer.db - b_torch.grad).item():.2e}")
print(f"dx 차이: {torch.norm(dL_dx - x_torch.grad).item():.2e}")
```

## 6. Automatic Differentiation

### 6.1 Forward Mode vs Reverse Mode

**Forward Mode**:
- Propagates derivatives from inputs to outputs
- Efficient when there are $n$ inputs and 1 output
- Useful for directional derivatives

**Reverse Mode**:
- Propagates derivatives from outputs to inputs (backpropagation)
- Efficient when there is 1 output and $n$ inputs
- Used in deep learning (loss function is scalar)

### 6.2 Computational Graph

A computational graph represents operations as nodes and data flow as edges.

```python
# 계산 그래프 예시: f(x, y) = (x + y) * (x - y)
import torch

x = torch.tensor(3.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)

# 중간 변수 저장
a = x + y  # a = 5
b = x - y  # b = 1
f = a * b  # f = 5

print("계산 그래프:")
print(f"x={x.item()}, y={y.item()}")
print(f"a = x + y = {a.item()}")
print(f"b = x - y = {b.item()}")
print(f"f = a * b = {f.item()}")

# 역전파
f.backward()
print(f"\n∂f/∂x = {x.grad.item()}")
print(f"∂f/∂y = {y.grad.item()}")

# 수동 계산 검증
# f = (x+y)(x-y) = x^2 - y^2
# ∂f/∂x = 2x = 6
# ∂f/∂y = -2y = -4
print(f"\n수동 계산: ∂f/∂x = 2x = {2*x.item()}")
print(f"수동 계산: ∂f/∂y = -2y = {-2*y.item()}")
```

### 6.3 PyTorch Autograd Internals

```python
# 계산 그래프 시각화 (간단한 예)
import torch

x = torch.tensor([1.0, 2.0], requires_grad=True)
w = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
b = torch.tensor([0.5, 1.0], requires_grad=True)

# 순전파
z = w @ x + b  # 선형 변환
a = torch.relu(z)  # 활성화
loss = a.sum()  # 손실

print("계산 그래프 추적:")
print(f"grad_fn of z: {z.grad_fn}")
print(f"grad_fn of a: {a.grad_fn}")
print(f"grad_fn of loss: {loss.grad_fn}")

# 역전파
loss.backward()

print("\n그래디언트:")
print(f"∂L/∂x: {x.grad}")
print(f"∂L/∂w:\n{w.grad}")
print(f"∂L/∂b: {b.grad}")
```

### 6.4 Higher-Order Derivatives

```python
# 2차 미분 (헤시안 대각선)
x = torch.tensor(2.0, requires_grad=True)
y = x**4

# 1차 미분
dy_dx = torch.autograd.grad(y, x, create_graph=True)[0]
print(f"f(x) = x^4, f'(x) = 4x^3")
print(f"f'(2) = {dy_dx.item()} (예상: {4*2**3})")

# 2차 미분
d2y_dx2 = torch.autograd.grad(dy_dx, x)[0]
print(f"f''(x) = 12x^2")
print(f"f''(2) = {d2y_dx2.item()} (예상: {12*2**2})")
```

### 6.5 Limitations of Automatic Differentiation and Manual Implementation

While automatic differentiation is convenient, manual implementation is sometimes necessary:

- Memory efficiency (gradient checkpointing)
- Custom backpropagation logic
- Improved numerical stability

```python
# 커스텀 autograd 함수
class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[x < 0] = 0
        return grad_input

# 사용
x = torch.randn(5, requires_grad=True)
y = MyReLU.apply(x)
loss = y.sum()
loss.backward()

print("커스텀 ReLU 그래디언트:")
print(f"x: {x.detach()}")
print(f"y: {y.detach()}")
print(f"∂L/∂x: {x.grad}")
```

## Practice Problems

### Problem 1: Derive Matrix Derivative Identity
For $\mathbf{x} \in \mathbb{R}^n$ and $A \in \mathbb{R}^{n \times n}$, prove:

$$\frac{\partial}{\partial \mathbf{x}} (\mathbf{x}^T A \mathbf{x}) = (A + A^T)\mathbf{x}$$

Derive step-by-step using index notation and write code to verify using PyTorch.

### Problem 2: Logistic Regression Gradient
The loss function for logistic regression is:

$$L(\mathbf{w}) = -\frac{1}{n} \sum_{i=1}^n \left[ y_i \log \sigma(\mathbf{w}^T \mathbf{x}_i) + (1-y_i) \log(1-\sigma(\mathbf{w}^T \mathbf{x}_i)) \right]$$

where $\sigma(z) = 1/(1+e^{-z})$. Derive the gradient $\nabla_\mathbf{w} L$. Show that the result is:

$$\nabla_\mathbf{w} L = \frac{1}{n} X^T (\boldsymbol{\sigma} - \mathbf{y})$$

where $\boldsymbol{\sigma} = [\sigma(\mathbf{w}^T \mathbf{x}_1), \ldots, \sigma(\mathbf{w}^T \mathbf{x}_n)]^T$.

### Problem 3: Batch Normalization Gradient
Batch normalization is defined as:

$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

where $\mu = \frac{1}{n}\sum_i x_i$ and $\sigma^2 = \frac{1}{n}\sum_i (x_i - \mu)^2$.

Derive the gradient of $x_i$ with respect to loss $L$. Verify by comparing with PyTorch's `BatchNorm1d`.

### Problem 4: Softmax Jacobian
Compute the Jacobian of the softmax function $\sigma(\mathbf{z})_i = e^{z_i} / \sum_j e^{z_j}$:

$$\frac{\partial \sigma_i}{\partial z_j} = ?$$

Hint: Consider the cases $i=j$ and $i \neq j$ separately. Show that the result is:

$$\frac{\partial \sigma_i}{\partial z_j} = \sigma_i(\delta_{ij} - \sigma_j)$$

### Problem 5: L2 Regularization Gradient
For ridge regression loss function:

$$L(\mathbf{w}) = \frac{1}{2n}\|\mathbf{y} - X\mathbf{w}\|^2 + \frac{\lambda}{2}\|\mathbf{w}\|^2$$

Derive the gradient and obtain the normal equation. Compare gradient descent and the analytical solution using PyTorch.

## References

### Online Resources
- [Matrix Calculus for Deep Learning](https://explained.ai/matrix-calculus/) - Detailed matrix calculus tutorial
- [The Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) - Matrix derivative formula reference
- [PyTorch Autograd Documentation](https://pytorch.org/docs/stable/autograd.html)

### Textbooks
- Magnus & Neudecker, *Matrix Differential Calculus with Applications in Statistics and Econometrics*
- Goodfellow et al., *Deep Learning*, Chapter 6 (Numerical Computation)
- Boyd & Vandenberghe, *Convex Optimization*, Appendix A

### Papers
- Griewank & Walther, *Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation* (2008)
- Baydin et al., *Automatic Differentiation in Machine Learning: a Survey* (JMLR 2018)
