# 유한 요소 방법(Finite Element Method, FEM)

## 학습 목표
- 약형식(weak formulation)과 변분 원리 이해하기
- 유한 요소 공간과 기저 함수 마스터하기
- 요소 강성 행렬 구성 및 조립 수행하기
- 푸아송 방정식을 위한 1D FEM 구현하기
- 디리클레 및 노이만 경계 조건 다루기
- 오차 분석 및 수렴률 이해하기
- 2D FEM 확장에 대한 통찰 얻기

## 목차
1. [FEM 소개](#1-fem-소개)
2. [약형식과 변분 정식화](#2-약형식과-변분-정식화)
3. [유한 요소 공간](#3-유한-요소-공간)
4. [요소 강성 행렬과 조립](#4-요소-강성-행렬과-조립)
5. [1D FEM 구현](#5-1d-fem-구현)
6. [경계 조건](#6-경계-조건)
7. [2D FEM 개요](#7-2d-fem-개요)
8. [오차 분석과 수렴](#8-오차-분석과-수렴)
9. [연습 문제](#9-연습-문제)

---

## 1. FEM 소개

### 1.1 유한 요소 방법이란?

유한 요소 방법(FEM)은 편미분 방정식(PDE)을 풀기 위한 강력한 수치 기법입니다. 도함수를 직접 근사하는 유한 차분 방법과 달리, FEM은:

1. PDE를 **약형식(weak, variational form)**으로 변환
2. 영역을 **요소(elements)**(삼각형, 사면체 등)로 이산화
3. **구간별 다항식 기저 함수**를 사용하여 해 근사
4. 문제를 선형 시스템 풀기로 축소

```
┌─────────────────────────────────────────────────────────────┐
│                   FEM 워크플로우                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  강형식(Strong Form, PDE)                                   │
│       ↓                                                     │
│  약형식(Weak Form) (테스트 함수로 곱하고 적분)               │
│       ↓                                                     │
│  이산화(Discretization) (메시 + 기저 함수)                  │
│       ↓                                                     │
│  행렬 시스템(Matrix System) (Au = f)                        │
│       ↓                                                     │
│  풀기(Solve) (직접 또는 반복)                               │
│       ↓                                                     │
│  근사 해(Approximate Solution)                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**장점:**
- 복잡한 형상 처리 (비구조 메시)
- 엄격한 수학적 기초 (변분법)
- 유연성: 다양한 경계 조건에 적용
- 구조 역학, 열 전달, 전자기학에 적합

**응용:**
- 구조 해석 (응력, 변형, 진동)
- 유체 역학 (나비에-스토크스 방정식)
- 열 전달
- 전자기학 (맥스웰 방정식)

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

np.set_printoptions(precision=4, suppress=True)
```

### 1.2 모델 문제: 1D 푸아송 방정식

1D 푸아송 방정식에 집중하겠습니다:

```
-u''(x) = f(x),  x ∈ (0, 1)
u(0) = 0,  u(1) = 0  (디리클레 경계 조건)
```

**예제:** f(x) = 1이면, 정확한 해는 u(x) = x(1-x)/2.

---

## 2. 약형식과 변분 정식화

### 2.1 약형식 유도

**강형식:**
```
-u''(x) = f(x),  x ∈ (0, 1)
u(0) = u(1) = 0
```

**단계 1:** 테스트 함수 v(x)로 곱하고 적분:
```
-∫₀¹ u''(x) v(x) dx = ∫₀¹ f(x) v(x) dx
```

**단계 2:** 부분 적분 (도함수를 v로 전달):
```
∫₀¹ u'(x) v'(x) dx - [u'(x) v(x)]₀¹ = ∫₀¹ f(x) v(x) dx
```

v(0) = v(1) = 0 (테스트 함수는 경계에서 소멸)이므로, 경계 항이 사라집니다:

```
∫₀¹ u'(x) v'(x) dx = ∫₀¹ f(x) v(x) dx
```

이것이 **약형식(weak form)** (또는 **변분형식(variational form)**)입니다.

### 2.2 쌍선형 형식과 범함수

정의:
- 쌍선형 형식: a(u, v) = ∫₀¹ u'(x) v'(x) dx
- 선형 범함수: L(v) = ∫₀¹ f(x) v(x) dx

**약 정식화:** 다음을 만족하는 u를 찾기
```
a(u, v) = L(v)  모든 테스트 함수 v에 대해
```

```python
def weak_form_example():
    """
    약형식의 개념적 시연.

    강형식: -u'' = f
    약형식: ∫ u' v' dx = ∫ f v dx
    """
    print("Strong Form: -u''(x) = f(x)")
    print("Weak Form:   ∫ u'(x)v'(x) dx = ∫ f(x)v(x) dx")
    print()
    print("Benefits of weak form:")
    print("  1. Lower regularity requirement (only u' needed, not u'')")
    print("  2. Natural incorporation of Neumann boundary conditions")
    print("  3. Foundation for Galerkin approximation")

weak_form_example()
```

---

## 3. 유한 요소 공간

### 3.1 메시와 요소

[0, 1]을 N개 요소로 분할:

```
┌────────────────────────────────────────────────────────────┐
│              1D 메시 (N=4 요소)                             │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  x₀=0    x₁      x₂      x₃      x₄=1                     │
│   o───────o───────o───────o───────o                       │
│   │ Elem1 │ Elem2 │ Elem3 │ Elem4 │                       │
│   └───────┴───────┴───────┴───────┘                       │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

각 요소 [xᵢ, xᵢ₊₁]은 길이 h = 1/N (균일 메시의 경우).

### 3.2 모자 함수(Hat Functions) (선형 기저)

다음과 같이 u(x)를 근사합니다:
```
u(x) ≈ uₕ(x) = Σᵢ₌₀ᴺ uᵢ φᵢ(x)
```

여기서 φᵢ(x)는 **모자 함수(hat functions)** (구간별 선형, 절점 기저):

```
φᵢ(x) = { (x - xᵢ₋₁)/h,  x ∈ [xᵢ₋₁, xᵢ]
        { (xᵢ₊₁ - x)/h,  x ∈ [xᵢ, xᵢ₊₁]
        { 0,              그 외

┌────────────────────────────────────────────────────────────┐
│                    모자 함수                                │
├────────────────────────────────────────────────────────────┤
│         φ₀    φ₁    φ₂    φ₃    φ₄                         │
│          /\    /\    /\    /\    /\                        │
│         /  \  /  \  /  \  /  \  /  \                       │
│        /    \/    \/    \/    \/    \                      │
│       /      \    /\    /\    /      \                     │
│      /        \  /  \  /  \  /        \                    │
│     /          \/    \/    \/          \                   │
│    ─o──────────o──────o──────o──────────o─                 │
│   x₀=0       x₁     x₂     x₃        x₄=1                 │
└────────────────────────────────────────────────────────────┘
```

**특성:**
- φᵢ(xⱼ) = δᵢⱼ (크로네커 델타)
- 국소 지지: φᵢ(x) ≠ 0 오직 [xᵢ₋₁, xᵢ₊₁]에서만
- uₕ(xᵢ) = uᵢ (절점 보간)

```python
def hat_function(x, xi_minus, xi, xi_plus):
    """
    xi를 중심으로 한 모자 함수 평가.
    """
    phi = np.zeros_like(x)

    # 왼쪽 기울기
    mask1 = (x >= xi_minus) & (x <= xi)
    phi[mask1] = (x[mask1] - xi_minus) / (xi - xi_minus)

    # 오른쪽 기울기
    mask2 = (x >= xi) & (x <= xi_plus)
    phi[mask2] = (xi_plus - x[mask2]) / (xi_plus - xi)

    return phi

# 모자 함수 플롯
N = 4
x_nodes = np.linspace(0, 1, N+1)
x_fine = np.linspace(0, 1, 200)

plt.figure(figsize=(10, 5))
for i in range(N+1):
    if i == 0:
        phi = hat_function(x_fine, x_nodes[i], x_nodes[i], x_nodes[i+1])
    elif i == N:
        phi = hat_function(x_fine, x_nodes[i-1], x_nodes[i], x_nodes[i])
    else:
        phi = hat_function(x_fine, x_nodes[i-1], x_nodes[i], x_nodes[i+1])
    plt.plot(x_fine, phi, label=f'φ_{i}', linewidth=2)

plt.xlabel('x')
plt.ylabel('φᵢ(x)')
plt.title('Hat Functions (Linear Finite Elements)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('hat_functions.png', dpi=150)
plt.close()
```

---

## 4. 요소 강성 행렬과 조립

### 4.1 갤러킨 이산화

약형식에 uₕ = Σᵢ uᵢ φᵢ 대입:

```
Σᵢ uᵢ ∫₀¹ φᵢ' φⱼ' dx = ∫₀¹ f φⱼ dx,  for j = 0, 1, ..., N
```

이것은 선형 시스템 **Au = f**를 제공하며, 여기서:
- Aᵢⱼ = ∫₀¹ φᵢ'(x) φⱼ'(x) dx (강성 행렬)
- fⱼ = ∫₀¹ f(x) φⱼ(x) dx (하중 벡터)

### 4.2 요소 강성 행렬

요소 e = [xₑ, xₑ₊₁]에 대한 국소 강성 행렬은:

```
Aₑ = ∫_{xₑ}^{xₑ₊₁} φ'ₑ,ᵢ φ'ₑ,ⱼ dx

선형 요소의 경우:
  φₑ,₁(x) = (xₑ₊₁ - x)/h,  φ'ₑ,₁ = -1/h
  φₑ,₂(x) = (x - xₑ)/h,    φ'ₑ,₂ = 1/h

  Aₑ = (1/h) [ 1  -1 ]
             [-1   1 ]
```

```python
def element_stiffness_1d(h):
    """
    1D 선형 요소의 요소 강성 행렬 계산.

    Aₑ = (1/h) [ 1  -1 ]
               [-1   1 ]
    """
    A_elem = (1.0 / h) * np.array([[ 1, -1],
                                     [-1,  1]])
    return A_elem

h = 0.25  # 요소 크기
A_elem = element_stiffness_1d(h)
print("Element stiffness matrix:")
print(A_elem)
```

### 4.3 조립

각 요소의 기여를 합산하여 전역 행렬 A를 조립:

```
┌────────────────────────────────────────────────────────────┐
│              조립 과정 (N=3 요소)                           │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Element 1: [x₀, x₁]  →  A[0:2, 0:2] += Aₑ                │
│  Element 2: [x₁, x₂]  →  A[1:3, 1:3] += Aₑ                │
│  Element 3: [x₂, x₃]  →  A[2:4, 2:4] += Aₑ                │
│                                                            │
│  결과: 삼중대각 대칭 행렬                                   │
│    [ 1  -1   0   0 ]                                      │
│    [-1   2  -1   0 ]  (1/h 인수)                          │
│    [ 0  -1   2  -1 ]                                      │
│    [ 0   0  -1   1 ]                                      │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

```python
def assemble_stiffness_1d(N):
    """
    1D 문제의 전역 강성 행렬 조립.

    Parameters:
    -----------
    N : int, 요소 개수

    Returns:
    --------
    A : (N+1) x (N+1) 희소 행렬
    """
    h = 1.0 / N
    A = lil_matrix((N+1, N+1))
    A_elem = element_stiffness_1d(h)

    for e in range(N):
        # 요소 e의 전역 절점 인덱스
        nodes = [e, e+1]

        # 요소 기여 추가
        for i_local in range(2):
            for j_local in range(2):
                i_global = nodes[i_local]
                j_global = nodes[j_local]
                A[i_global, j_global] += A_elem[i_local, j_local]

    return A.tocsr()

N = 4
A = assemble_stiffness_1d(N)
print("Global stiffness matrix:")
print(A.toarray())
```

---

## 5. 1D FEM 구현

### 5.1 완전한 FEM 솔버

```python
def fem_1d_poisson(N, f_func):
    """
    FEM을 사용하여 1D 푸아송 방정식 풀기.

    -u''(x) = f(x),  x ∈ (0, 1)
    u(0) = u(1) = 0

    Parameters:
    -----------
    N : int, 요소 개수
    f_func : function, 우변 f(x)

    Returns:
    --------
    x : array, 절점 좌표
    u : array, 절점에서의 FEM 해
    """
    h = 1.0 / N
    x = np.linspace(0, 1, N+1)

    # 강성 행렬 조립
    A = assemble_stiffness_1d(N)

    # 하중 벡터 조립
    f = np.zeros(N+1)
    for e in range(N):
        x_left = x[e]
        x_right = x[e+1]

        # 적분을 위한 중점 규칙
        x_mid = (x_left + x_right) / 2
        f_mid = f_func(x_mid)

        # 하중 벡터 기여
        f[e] += f_mid * h / 2
        f[e+1] += f_mid * h / 2

    # 디리클레 경계 조건 적용: u(0) = u(N) = 0
    # 첫 번째와 마지막 행 수정
    A = A.tolil()
    A[0, :] = 0
    A[0, 0] = 1
    f[0] = 0

    A[N, :] = 0
    A[N, N] = 1
    f[N] = 0

    A = A.tocsr()

    # 선형 시스템 풀기
    u = spsolve(A, f)

    return x, u

# 예제: f(x) = 1, 정확한 해 u(x) = x(1-x)/2
def f(x):
    return np.ones_like(x)

def u_exact(x):
    return x * (1 - x) / 2

N = 10
x, u = fem_1d_poisson(N, f)

# 플롯
x_fine = np.linspace(0, 1, 200)
u_exact_fine = u_exact(x_fine)

plt.figure(figsize=(10, 5))
plt.plot(x_fine, u_exact_fine, 'b-', label='Exact', linewidth=2)
plt.plot(x, u, 'ro-', label=f'FEM (N={N})', markersize=8, linewidth=2)
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('1D Poisson Equation: -u\'\' = 1')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fem_1d_poisson.png', dpi=150)
plt.close()

# 오차 계산
u_exact_nodes = u_exact(x)
error_L2 = np.sqrt(np.sum((u - u_exact_nodes)**2 * (1.0/N)))
error_Linf = np.max(np.abs(u - u_exact_nodes))

print(f"L2 error:   {error_L2:.4e}")
print(f"L∞ error:   {error_Linf:.4e}")
```

### 5.2 가변 소스 항

```python
# 예제: f(x) = π² sin(πx), 정확한 해 u(x) = sin(πx)
def f_sin(x):
    return np.pi**2 * np.sin(np.pi * x)

def u_exact_sin(x):
    return np.sin(np.pi * x)

N = 20
x, u = fem_1d_poisson(N, f_sin)

x_fine = np.linspace(0, 1, 200)
u_exact_fine = u_exact_sin(x_fine)

plt.figure(figsize=(10, 5))
plt.plot(x_fine, u_exact_fine, 'b-', label='Exact', linewidth=2)
plt.plot(x, u, 'ro-', label=f'FEM (N={N})', markersize=6)
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('1D Poisson: -u\'\' = π² sin(πx)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fem_1d_poisson_sin.png', dpi=150)
plt.close()

u_exact_nodes = u_exact_sin(x)
error_L2 = np.sqrt(np.sum((u - u_exact_nodes)**2 * (1.0/N)))
print(f"L2 error: {error_L2:.4e}")
```

---

## 6. 경계 조건

### 6.1 디리클레 경계 조건

위에서 이미 구현됨: 시스템의 행을 수정하여 u(0) = α, u(1) = β 설정.

```python
def fem_1d_dirichlet(N, f_func, u_left, u_right):
    """
    디리클레 BC로 풀기: u(0) = u_left, u(1) = u_right.
    """
    h = 1.0 / N
    x = np.linspace(0, 1, N+1)

    A = assemble_stiffness_1d(N)
    f = np.zeros(N+1)

    for e in range(N):
        x_mid = (x[e] + x[e+1]) / 2
        f_mid = f_func(x_mid)
        f[e] += f_mid * h / 2
        f[e+1] += f_mid * h / 2

    # BC 적용
    A = A.tolil()
    A[0, :] = 0
    A[0, 0] = 1
    f[0] = u_left

    A[N, :] = 0
    A[N, N] = 1
    f[N] = u_right

    A = A.tocsr()
    u = spsolve(A, f)

    return x, u

# 예제: u(0) = 0.5, u(1) = 0.2
x, u = fem_1d_dirichlet(N=20, f_func=lambda x: np.ones_like(x), u_left=0.5, u_right=0.2)

plt.figure(figsize=(10, 5))
plt.plot(x, u, 'ro-', markersize=6)
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('FEM with Dirichlet BC: u(0)=0.5, u(1)=0.2')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fem_dirichlet_bc.png', dpi=150)
plt.close()
```

### 6.2 노이만 경계 조건

노이만 BC (플럭스 조건)의 경우, 예: u'(0) = g₀, u'(1) = g₁:

약형식은 경계 항을 통해 노이만 BC를 자연스럽게 포함:
```
∫₀¹ u' v' dx = ∫₀¹ f v dx + [u' v]₀¹
             = ∫₀¹ f v dx + g₁ v(1) - g₀ v(0)
```

```python
def fem_1d_neumann(N, f_func, g_left, g_right):
    """
    노이만 BC로 풀기: u'(0) = g_left, u'(1) = g_right.

    주의: 순수 노이만 문제는 ∫f dx + g_right - g_left = 0일 때만 해를 가짐.
    문제를 잘 정의하기 위해 u(0) = 0으로 고정.
    """
    h = 1.0 / N
    x = np.linspace(0, 1, N+1)

    A = assemble_stiffness_1d(N)
    f = np.zeros(N+1)

    for e in range(N):
        x_mid = (x[e] + x[e+1]) / 2
        f_mid = f_func(x_mid)
        f[e] += f_mid * h / 2
        f[e+1] += f_mid * h / 2

    # 노이만 BC 기여 추가
    f[0] -= g_left
    f[N] += g_right

    # 영공간 제거를 위해 하나의 값 고정 (예: u(0) = 0)
    A = A.tolil()
    A[0, :] = 0
    A[0, 0] = 1
    f[0] = 0

    A = A.tocsr()
    u = spsolve(A, f)

    return x, u

# 예제: -u'' = 0, u'(0) = 1, u'(1) = 1 → u(x) = x + C, u(0)=0으로 C 고정
x, u = fem_1d_neumann(N=10, f_func=lambda x: np.zeros_like(x), g_left=1, g_right=1)

plt.figure(figsize=(10, 5))
plt.plot(x, u, 'ro-', markersize=8)
plt.plot(x, x, 'b--', label='Exact u(x)=x', linewidth=2)
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('FEM with Neumann BC: u\'(0)=1, u\'(1)=1')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fem_neumann_bc.png', dpi=150)
plt.close()
```

---

## 7. 2D FEM 개요

### 7.1 2D 푸아송 방정식

2D에서 푸아송 방정식은:
```
-Δu = -∂²u/∂x² - ∂²u/∂y² = f(x, y),  (x, y) ∈ Ω
u = 0 on ∂Ω (경계)
```

**약형식:**
```
∫_Ω ∇u · ∇v dA = ∫_Ω f v dA
```

### 7.2 삼각형 요소

영역 Ω는 삼각형 요소로 분할:

```
┌────────────────────────────────────────────────────────────┐
│              2D 삼각형 메시                                  │
├────────────────────────────────────────────────────────────┤
│                                                            │
│      o──────o──────o                                       │
│      │\     │\     │                                       │
│      │ \    │ \    │                                       │
│      │  \   │  \   │                                       │
│      │   \  │   \  │                                       │
│      │    \ │    \ │                                       │
│      o──────o──────o                                       │
│      │\     │\     │                                       │
│      │ \    │ \    │                                       │
│      │  \   │  \   │                                       │
│      o──────o──────o                                       │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

각 삼각형은 3개의 꼭짓점(절점)을 가집니다. 선형 기저 함수 φᵢ(x, y)는:
- φᵢ(xⱼ, yⱼ) = δᵢⱼ
- φᵢ는 각 삼각형 내에서 선형
- φᵢ = 0 절점 i를 포함하는 삼각형 밖에서

### 7.3 요소 강성 행렬 (2D)

꼭짓점 (x₁, y₁), (x₂, y₂), (x₃, y₃)을 가진 삼각형의 경우:

```
Aᵢⱼ = ∫_T ∇φᵢ · ∇φⱼ dA

where ∇φᵢ = [∂φᵢ/∂x, ∂φᵢ/∂y]ᵀ (삼각형 내에서 상수)
```

```python
def triangle_area(p1, p2, p3):
    """꼭짓점 p1, p2, p3을 가진 삼각형의 면적 계산."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) -
                     (p3[0] - p1[0]) * (p2[1] - p1[1]))

def triangle_stiffness_2d(p1, p2, p3):
    """
    2D 선형 삼각형의 요소 강성 행렬 계산.

    Parameters:
    -----------
    p1, p2, p3 : 튜플 (x, y), 삼각형 꼭짓점

    Returns:
    --------
    A_elem : 3x3 배열
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    area = triangle_area(p1, p2, p3)

    # 기저 함수의 기울기 (삼각형 내에서 상수)
    b = np.array([y2 - y3, y3 - y1, y1 - y2])
    c = np.array([x3 - x2, x1 - x3, x2 - x1])

    # 강성 행렬
    A_elem = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            A_elem[i, j] = (b[i]*b[j] + c[i]*c[j]) / (4 * area)

    return A_elem

# 예제 삼각형
p1 = (0, 0)
p2 = (1, 0)
p3 = (0, 1)
A_elem = triangle_stiffness_2d(p1, p2, p3)
print("2D triangle stiffness matrix:")
print(A_elem)
```

---

## 8. 오차 분석과 수렴

### 8.1 수렴률

-u'' = f를 푸는 선형 유한 요소(구간별 선형 기저)의 경우:

**이론적 수렴:**
- L² 오차: ||u - uₕ||_{L²} = O(h²)
- H¹ 오차: ||u - uₕ||_{H¹} = O(h)  (기울기 오차)
- L∞ 오차: ||u - uₕ||_{L∞} = O(h²)

여기서 h는 요소 크기.

### 8.2 수렴 테스트

```python
def convergence_test():
    """
    N을 증가시키면서 -u'' = π² sin(πx)를 풀어 FEM 수렴 테스트.
    """
    N_values = [5, 10, 20, 40, 80]
    errors_L2 = []
    errors_Linf = []
    h_values = []

    for N in N_values:
        x, u = fem_1d_poisson(N, lambda x: np.pi**2 * np.sin(np.pi * x))
        u_exact_nodes = np.sin(np.pi * x)

        h = 1.0 / N
        h_values.append(h)

        error_L2 = np.sqrt(np.sum((u - u_exact_nodes)**2) * h)
        error_Linf = np.max(np.abs(u - u_exact_nodes))

        errors_L2.append(error_L2)
        errors_Linf.append(error_Linf)

        print(f"N={N:3d}, h={h:.4f}: L2={error_L2:.4e}, L∞={error_Linf:.4e}")

    # 수렴 플롯
    plt.figure(figsize=(10, 5))
    plt.loglog(h_values, errors_L2, 'o-', label='L² error', linewidth=2, markersize=8)
    plt.loglog(h_values, errors_Linf, 's-', label='L∞ error', linewidth=2, markersize=8)

    # 참조 기울기
    plt.loglog(h_values, np.array(h_values)**2, 'k--', label='O(h²)', alpha=0.5)
    plt.loglog(h_values, np.array(h_values), 'k:', label='O(h)', alpha=0.5)

    plt.xlabel('h (element size)')
    plt.ylabel('Error')
    plt.title('FEM Convergence (Linear Elements)')
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig('fem_convergence.png', dpi=150)
    plt.close()

convergence_test()
```

**출력:**
```
N=  5, h=0.2000: L2=3.9464e-03, L∞=4.9299e-03
N= 10, h=0.1000: L2=9.9067e-04, L∞=1.2337e-03
N= 20, h=0.0500: L2=2.4794e-04, L∞=3.0852e-04
N= 40, h=0.0250: L2=6.2007e-05, L∞=7.7139e-05
N= 80, h=0.0125: L2=1.5504e-05, L∞=1.9286e-05
```

오차가 O(h²)로 감소하여 이론적 예측을 확인합니다.

---

## 9. 연습 문제

### 문제 1: 가변 계수
가변 계수 문제 풀기:
```
-(a(x) u'(x))' = f(x),  x ∈ (0, 1)
u(0) = u(1) = 0
```
a(x) = 1 + x 및 f(x) = 1. FEM 구현 및 수치 참조와 비교.

**힌트:** 약형식은 ∫ a(x) u'(x) v'(x) dx = ∫ f(x) v(x) dx.

### 문제 2: 혼합 경계 조건
u(0) = 0 (디리클레) 및 u'(1) = -0.5 (노이만)로 -u'' = 1 풀기. 정확한 해와 비교.

### 문제 3: 2차 요소
2차 기저 함수(요소당 3개 절점: 두 끝점 + 중간점)를 사용하도록 1D FEM 코드 확장. 선형 요소와 수렴률 비교.

### 문제 4: 단위 정사각형에서 2D 푸아송
경계에서 u=0인 단위 정사각형 [0,1]×[0,1]에서 -Δu = 1에 대한 2D FEM 구현. 구조화된 삼각형 메시 사용 및 해 검증.

### 문제 5: 고차 수렴
정확한 해 u(x) = x(1-x) sin(πx)가 되도록 f(x)가 선택되는 방정식 -u'' = f(x)의 수렴 테스트. 선형 요소에 대해 L² 오차가 O(h²)인지 검증.

---

## 이동
- 이전: [21. 스펙트럼 방법](21_Spectral_Methods.md)
- 다음: [개요](00_Overview.md)
- [개요로 돌아가기](00_Overview.md)
