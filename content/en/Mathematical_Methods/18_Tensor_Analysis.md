# 18. Tensor Analysis

## Learning Objectives
- Understand the definition of tensors from the perspective of coordinate transformation laws and classify scalars, vectors, and matrices as special cases of tensors
- Use Einstein summation convention and index notation to concisely express and manipulate tensor equations
- Distinguish transformation laws for contravariant and covariant tensors and perform index raising/lowering through the metric tensor
- Understand the concepts of Christoffel symbols and covariant derivatives, and correctly calculate tensor differentiation in curvilinear coordinate systems
- Understand the definition and geometric meaning of the Riemann curvature tensor and calculate curvature in simple spaces
- Describe physical applications of tensor analysis (stress tensor, electromagnetic field tensor, Einstein equations) and perform calculations in Python

> **Why are tensors necessary?** Natural laws must be independent of the choice of coordinate system. Scalars (rank-0) and vectors (rank-1) alone cannot describe physical quantities such as stress, moment of inertia, and electromagnetic fields. Tensors are geometric objects that follow well-defined transformation laws under arbitrary coordinate transformations, providing a natural language to express physical laws in a coordinate-independent form.

---

## 1. Basic Concepts of Tensors

### 1.1 Motivation: Why Scalars and Vectors Are Not Enough

In physics, many quantities are sufficiently described by scalars (temperature, energy) or vectors (force, velocity). However, the following physical quantities require **two or more directional** information:

- **Stress tensor** $\sigma_{ij}$: direction of surface ($j$) and direction of force acting on that surface ($i$)
- **Moment of inertia tensor** $I_{ij}$: relationship between angular velocity direction and angular momentum direction
- **Electromagnetic field tensor** $F_{\mu\nu}$: antisymmetric tensor integrating electric and magnetic fields

These are all **rank-2 tensors**, having $n^2$ components in $n$-dimensional space.

### 1.2 Coordinate Transformations and Tensor Definition

Under coordinate transformation $x^i \to x'^i(x^1, x^2, \ldots, x^n)$, a **rank-$k$ tensor** is an object whose components follow specific transformation laws:

- **Rank-0 (scalar)**: $\phi' = \phi$ (invariant)
- **Rank-1 (vector)**: $A'^i = \frac{\partial x'^i}{\partial x^j}A^j$ (for contravariant vectors)
- **Rank-2 tensor**: $T'^{ij} = \frac{\partial x'^i}{\partial x^k}\frac{\partial x'^j}{\partial x^l}T^{kl}$

In general, **tensors can be defined as multilinear maps**. A rank-$(p,q)$ tensor is a multilinear function that takes $p$ covectors and $q$ contravariant vectors and produces a real number.

### 1.3 Tensor Rank/Order

| Rank | Number of components ($n$-dim) | Physical examples |
|------|-------------------------------|-------------------|
| 0 | $1$ | Temperature, mass, energy |
| 1 | $n$ | Force, velocity, electric field |
| 2 | $n^2$ | Stress, moment of inertia, metric tensor |
| 3 | $n^3$ | Piezoelectric tensor |
| 4 | $n^4$ | Riemann curvature tensor, elasticity tensor |

### 1.4 Python: Coordinate Transformation Example

```python
import numpy as np

# === 2차원 회전 변환에서 텐서 변환 법칙 검증 ===

# 회전 각도
theta = np.pi / 6  # 30도

# 변환 행렬: x'^i = R^i_j x^j
R = np.array([
    [np.cos(theta), np.sin(theta)],
    [-np.sin(theta), np.cos(theta)]
])
print(f"회전 행렬 R (θ = {np.degrees(theta):.0f}°):")
print(R)

# --- Rank-1 텐서 (벡터) 변환 ---
A = np.array([3.0, 4.0])  # 원래 좌표계의 벡터
A_prime = R @ A            # A'^i = R^i_j A^j
print(f"\n원래 벡터: A = {A}")
print(f"변환된 벡터: A' = {A_prime}")
print(f"|A| = {np.linalg.norm(A):.4f}, |A'| = {np.linalg.norm(A_prime):.4f}")
# 크기 보존 확인

# --- Rank-2 텐서 변환 ---
# 응력 텐서 예시
T = np.array([
    [10.0, 3.0],
    [3.0,  5.0]
])  # 대칭 텐서

# T'^{ij} = R^i_k R^j_l T^{kl} = R T R^T
T_prime = R @ T @ R.T
print(f"\n원래 텐서:\n{T}")
print(f"변환된 텐서:\n{T_prime}")

# 텐서의 불변량(trace, determinant) 확인
print(f"\ntr(T) = {np.trace(T):.4f}, tr(T') = {np.trace(T_prime):.4f}")
print(f"det(T) = {np.linalg.det(T):.4f}, det(T') = {np.linalg.det(T_prime):.4f}")
# 대각합과 행렬식은 좌표 변환에 불변
```

---

## 2. Index Notation and Einstein Summation Convention

### 2.1 Einstein Summation Convention

Einstein's summation convention is a notation that makes tensor expressions concise:

> **When an upper index and a lower index are repeated with the same letter in the same term, summation over that index is implied.**

$$
A^i B_i \equiv \sum_{i=1}^{n} A^i B_i
$$

**Free index**: an index that appears on both sides of an equation (represents each component of the equation)

**Dummy index**: a repeated index that is summed over by the summation convention (can be renamed)

$$
A^i B_i = A^j B_j \quad (\text{dummy index } i \text{ can be replaced with } j)
$$

### 2.2 Kronecker Delta $\delta_{ij}$

The **Kronecker delta** corresponds to the components of the identity matrix:

$$
\delta_{ij} = \begin{cases} 1 & (i = j) \\ 0 & (i \ne j) \end{cases}
$$

Key properties:
- $\delta_{ij} A^j = A^i$ (index substitution role)
- $\delta_{ii} = n$ (trace in $n$ dimensions)
- $\delta_{ij}\delta_{jk} = \delta_{ik}$

### 2.3 Levi-Civita Symbol $\varepsilon_{ijk}$

The **Levi-Civita symbol** is a completely antisymmetric symbol:

$$
\varepsilon_{ijk} = \begin{cases} +1 & (ijk) \text{ is an even permutation of }(123) \\ -1 & (ijk) \text{ is an odd permutation of }(123) \\ 0 & \text{if any index is repeated} \end{cases}
$$

**Representation of cross product and determinant:**

$$
(\mathbf{A} \times \mathbf{B})_i = \varepsilon_{ijk} A_j B_k
$$

$$
\det(M) = \varepsilon_{ijk} M_{1i} M_{2j} M_{3k}
$$

### 2.4 $\varepsilon$-$\delta$ Identity

A very useful identity in tensor calculations:

$$
\varepsilon_{ijk}\varepsilon_{imn} = \delta_{jm}\delta_{kn} - \delta_{jn}\delta_{km}
$$

From this identity, various identities such as vector triple products can be derived:

$$
\mathbf{A} \times (\mathbf{B} \times \mathbf{C}) = \mathbf{B}(\mathbf{A} \cdot \mathbf{C}) - \mathbf{C}(\mathbf{A} \cdot \mathbf{B})
$$

### 2.5 Python: Tensor Operations Using numpy.einsum

```python
import numpy as np

# === numpy.einsum: 아인슈타인 합산 규약의 Python 구현 ===

A = np.array([1.0, 2.0, 3.0])
B = np.array([4.0, 5.0, 6.0])

# 내적: A^i B_i
dot = np.einsum('i,i->', A, B)
print(f"내적 A·B = {dot}")  # 32.0

# 외적 (텐서곱): C^{ij} = A^i B^j
outer = np.einsum('i,j->ij', A, B)
print(f"\n외적 A⊗B:\n{outer}")

# 행렬-벡터 곱: (Mv)^i = M^{ij} v_j
M = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
v = np.array([1.0, 0.0, -1.0])
Mv = np.einsum('ij,j->i', M, v)
print(f"\nMv = {Mv}")

# 행렬 곱: (AB)^{ik} = A^{ij} B^{jk}
N = np.random.rand(3, 3)
MN = np.einsum('ij,jk->ik', M, N)
print(f"\n행렬 곱 MN (einsum):\n{MN}")
print(f"행렬 곱 MN (numpy):  \n{M @ N}")

# 대각합(trace): T^i_i
trace = np.einsum('ii->', M)
print(f"\ntr(M) = {trace}")

# 이중 축약(double contraction): A_{ij} B_{ij}
A2 = np.random.rand(3, 3)
B2 = np.random.rand(3, 3)
double_contract = np.einsum('ij,ij->', A2, B2)
print(f"A:B (이중 축약) = {double_contract:.4f}")

# --- 레비-치비타 기호와 외적 ---
# 3차원 레비-치비타 기호 생성
def levi_civita_3d():
    """3차원 레비-치비타 기호 ε_{ijk}를 생성"""
    eps = np.zeros((3, 3, 3))
    eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1
    eps[0, 2, 1] = eps[2, 1, 0] = eps[1, 0, 2] = -1
    return eps

eps = levi_civita_3d()

# 외적: (A × B)_i = ε_{ijk} A_j B_k
cross_einsum = np.einsum('ijk,j,k->i', eps, A, B)
cross_numpy = np.cross(A, B)
print(f"\nA × B (einsum): {cross_einsum}")
print(f"A × B (numpy):  {cross_numpy}")

# ε-δ 항등식 검증: ε_{ijk} ε_{imn} = δ_{jm}δ_{kn} - δ_{jn}δ_{km}
lhs = np.einsum('ijk,imn->jkmn', eps, eps)
delta = np.eye(3)
rhs = np.einsum('jm,kn->jkmn', delta, delta) - np.einsum('jn,km->jkmn', delta, delta)
print(f"\nε-δ 항등식 검증: {np.allclose(lhs, rhs)}")
```

---

## 3. Contravariant and Covariant Tensors

### 3.1 Contravariant Vector

Under coordinate transformation $x^i \to x'^i$, the components of a **contravariant vector** transform like coordinate differentials:

$$
A'^i = \frac{\partial x'^i}{\partial x^j}A^j
$$

Denoted by upper indices, the displacement vector $dx^i$ is a typical example.

### 3.2 Covariant Vector

The components of a **covariant vector (1-form)** transform like gradients:

$$
A'_i = \frac{\partial x^j}{\partial x'^i}A_j
$$

Denoted by lower indices, the partial derivative of a scalar function $\partial_i \phi = \frac{\partial \phi}{\partial x^i}$ is a typical example.

### 3.3 Mixed Tensors and Index Raising/Lowering

**Mixed tensor**: a tensor with both upper and lower indices. For example, a rank-$(1,1)$ tensor $T^i{}_j$:

$$
T'^i{}_j = \frac{\partial x'^i}{\partial x^k}\frac{\partial x^l}{\partial x'^j}T^k{}_l
$$

**Index raising/lowering** is performed through the metric tensor:

$$
A^i = g^{ij}A_j \quad (\text{raising}), \qquad A_i = g_{ij}A^j \quad (\text{lowering})
$$

### 3.4 Python: Contravariant/Covariant Transformation Example

```python
import numpy as np
import sympy as sp

# === 극좌표 ↔ 직교좌표 변환에서 반변/공변 벡터 ===

r_val, theta_val = 2.0, np.pi / 4  # (r, θ) = (2, 45°)

# 직교 좌표 → 극좌표 변환의 야코비안
# x = r cosθ, y = r sinθ
# ∂x^i/∂x'^j 계산 (x' = 극좌표, x = 직교좌표)

# ∂(x,y)/∂(r,θ) : 직교를 극좌표로 미분
J = np.array([
    [np.cos(theta_val), -r_val * np.sin(theta_val)],  # ∂x/∂r, ∂x/∂θ
    [np.sin(theta_val),  r_val * np.cos(theta_val)]   # ∂y/∂r, ∂y/∂θ
])

# 역야코비안: ∂(r,θ)/∂(x,y)
J_inv = np.linalg.inv(J)

print("야코비안 ∂(x,y)/∂(r,θ):")
print(J)
print(f"\n역야코비안 ∂(r,θ)/∂(x,y):")
print(J_inv)

# 직교좌표에서의 벡터 (반변 성분)
A_cart = np.array([1.0, 1.0])  # (Ax, Ay)

# 반변 변환: A'^i = (∂x'^i/∂x^j) A^j
# 극좌표의 반변 성분 = J_inv @ A_cart
A_polar_contra = J_inv @ A_cart
print(f"\n직교좌표 벡터 A = {A_cart}")
print(f"극좌표 반변 성분 (A^r, A^θ) = {A_polar_contra}")

# 공변 변환: A'_i = (∂x^j/∂x'^i) A_j
# 극좌표의 공변 성분 = J^T @ A_cart
A_polar_cov = J.T @ A_cart
print(f"극좌표 공변 성분 (A_r, A_θ) = {A_polar_cov}")

# 극좌표의 계량 텐서로 검증: g_{ij} = diag(1, r²)
g = np.diag([1.0, r_val**2])
A_cov_from_contra = g @ A_polar_contra
print(f"\ng_{ij} A^j = {A_cov_from_contra}")
print(f"직접 계산한 공변 성분과 일치: {np.allclose(A_polar_cov, A_cov_from_contra)}")
```

---

## 4. Metric Tensor

### 4.1 Line Element and Definition of Metric Tensor

The distance (line element) between two adjacent points in a coordinate system is defined by the metric tensor $g_{ij}$:

$$
ds^2 = g_{ij} \, dx^i \, dx^j
$$

The metric tensor is a **symmetric rank-2 covariant tensor** that defines the inner product: $g_{ij} = g_{ji}$.

### 4.2 Metric Tensors in Various Coordinate Systems

**Cartesian coordinates** $(x, y, z)$:

$$
ds^2 = dx^2 + dy^2 + dz^2 \quad \Rightarrow \quad g_{ij} = \delta_{ij} = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix}
$$

**Cylindrical coordinates** $(\rho, \phi, z)$:

$$
ds^2 = d\rho^2 + \rho^2 d\phi^2 + dz^2 \quad \Rightarrow \quad g_{ij} = \begin{pmatrix} 1 & 0 & 0 \\ 0 & \rho^2 & 0 \\ 0 & 0 & 1 \end{pmatrix}
$$

**Spherical coordinates** $(r, \theta, \phi)$:

$$
ds^2 = dr^2 + r^2 d\theta^2 + r^2 \sin^2\theta \, d\phi^2 \quad \Rightarrow \quad g_{ij} = \begin{pmatrix} 1 & 0 & 0 \\ 0 & r^2 & 0 \\ 0 & 0 & r^2\sin^2\theta \end{pmatrix}
$$

### 4.3 Metric Tensor on Surfaces

**2D sphere** ($r = R$ fixed):

$$
ds^2 = R^2 d\theta^2 + R^2 \sin^2\theta \, d\phi^2 \quad \Rightarrow \quad g_{ij} = R^2\begin{pmatrix} 1 & 0 \\ 0 & \sin^2\theta \end{pmatrix}
$$

### 4.4 Inverse Metric Tensor and Volume Element

The **inverse metric tensor** $g^{ij}$ satisfies $g^{ik}g_{kj} = \delta^i{}_j$.

**Volume element**: The volume element of a coordinate system is determined by the determinant of the metric tensor:

$$
dV = \sqrt{|g|} \, dx^1 \, dx^2 \cdots dx^n, \quad g = \det(g_{ij})
$$

### 4.5 Python: Metric Tensor Calculation

```python
import sympy as sp

# === 다양한 좌표계에서 계량 텐서 계산 ===

def compute_metric(coords, transform):
    """
    좌표 변환으로부터 계량 텐서를 계산한다.

    Parameters:
        coords: 곡선좌표 변수 리스트
        transform: 직교좌표 표현 리스트 [x(...), y(...), z(...)]
    Returns:
        계량 텐서 행렬 (SymPy Matrix)
    """
    n = len(coords)
    r = sp.Matrix(transform)
    g = sp.zeros(n, n)
    for i in range(n):
        for j in range(i, n):
            g[i, j] = sp.trigsimp(r.diff(coords[i]).dot(r.diff(coords[j])))
            g[j, i] = g[i, j]
    return g

# 구면 좌표
r, theta, phi = sp.symbols('r theta phi', positive=True)
g_sph = compute_metric(
    [r, theta, phi],
    [r * sp.sin(theta) * sp.cos(phi),
     r * sp.sin(theta) * sp.sin(phi),
     r * sp.cos(theta)]
)
print("구면 좌표 계량 텐서:")
sp.pprint(g_sph)
print(f"det(g) = {sp.trigsimp(g_sph.det())}")
print(f"sqrt(|g|) = {sp.sqrt(sp.trigsimp(g_sph.det()))}")
# r^4 sin^2(theta) → sqrt = r^2 sin(theta)

# 2차원 구면 (r = R 고정)
R = sp.Symbol('R', positive=True)
g_sphere = compute_metric(
    [theta, phi],
    [R * sp.sin(theta) * sp.cos(phi),
     R * sp.sin(theta) * sp.sin(phi),
     R * sp.cos(theta)]
)
print("\n2차원 구면 계량 텐서:")
sp.pprint(g_sphere)

# 역계량 텐서
g_sph_inv = g_sph.inv()
print("\n구면 좌표 역계량 텐서 g^{ij}:")
sp.pprint(sp.simplify(g_sph_inv))

# 검증: g^{ik} g_{kj} = δ^i_j
identity_check = sp.simplify(g_sph_inv * g_sph)
print(f"\ng^{{ik}} g_{{kj}} = I 검증: {identity_check == sp.eye(3)}")
```

---

## 5. Tensor Algebra

### 5.1 Tensor Addition and Scalar Multiplication

Only tensors of the same type can be added:

$$
(A + B)^{ij} = A^{ij} + B^{ij}, \quad (\alpha A)^{ij} = \alpha A^{ij}
$$

### 5.2 Tensor Product (Outer Product)

The tensor product of a rank-$(p,q)$ tensor and a rank-$(r,s)$ tensor produces a rank-$(p+r, q+s)$ tensor:

$$
(A \otimes B)^{ij}{}_{kl} = A^i{}_k \, B^j{}_l
$$

### 5.3 Contraction

Setting one upper index and one lower index to the same letter and summing reduces the rank by 2:

$$
T^i{}_{ij} = \sum_i T^i{}_{ij} \quad (\text{rank-}(2,1) \to \text{rank-}(1,0))
$$

Typical example: trace $T^i{}_i = \text{tr}(T)$

### 5.4 Symmetrization and Antisymmetrization

**Symmetric tensor**: $T_{ij} = T_{ji}$

$$
T_{(ij)} = \frac{1}{2}(T_{ij} + T_{ji}) \quad (\text{symmetrization})
$$

**Antisymmetric tensor**: $T_{ij} = -T_{ji}$

$$
T_{[ij]} = \frac{1}{2}(T_{ij} - T_{ji}) \quad (\text{antisymmetrization})
$$

Any rank-2 tensor can be decomposed into symmetric and antisymmetric parts:

$$
T_{ij} = T_{(ij)} + T_{[ij]}
$$

### 5.5 Python: Tensor Algebra Operations

```python
import numpy as np

# === 텐서 대수 연산 ===

# rank-2 텐서 (3×3)
T = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
], dtype=float)

# 대칭 부분과 반대칭 부분 분해
T_sym = 0.5 * (T + T.T)       # T_{(ij)}
T_antisym = 0.5 * (T - T.T)   # T_{[ij]}
print("원래 텐서 T:")
print(T)
print("\n대칭 부분 T_{(ij)}:")
print(T_sym)
print("\n반대칭 부분 T_{[ij]}:")
print(T_antisym)
print(f"\n복원 검증: T = T_sym + T_antisym? {np.allclose(T, T_sym + T_antisym)}")

# 텐서곱 (outer product)
A = np.array([1, 2, 3], dtype=float)
B = np.array([4, 5, 6], dtype=float)
AB_outer = np.einsum('i,j->ij', A, B)  # A^i B^j
print(f"\n텐서곱 A⊗B:\n{AB_outer}")

# 축약 (contraction)
# rank-2 텐서의 trace: T^i_i
trace_T = np.einsum('ii->', T)
print(f"\n축약 (trace): T^i_i = {trace_T}")

# rank-4 텐서에서 축약
R = np.random.rand(3, 3, 3, 3)
# R^i_{jkl} → 첫째와 셋째 인덱스 축약 → R^i_{jil} = S_{jl}
S = np.einsum('ijil->jl', R)
print(f"\nrank-4 텐서 축약 결과 (rank-2): shape = {S.shape}")
```

---

## 6. Covariant Derivative and Christoffel Symbols

### 6.1 Problems with Ordinary Differentiation

In curvilinear coordinate systems, the ordinary partial derivative of a tensor $\partial_i A^j$ is **not a tensor**. This is because the basis vectors themselves vary from point to point. A proper differential operator is needed, which is the **covariant derivative**.

### 6.2 Definition of Christoffel Symbols

The **Christoffel symbol of the second kind** $\Gamma^k{}_{ij}$ is calculated from the metric tensor:

$$
\Gamma^k{}_{ij} = \frac{1}{2}g^{kl}\left(\frac{\partial g_{jl}}{\partial x^i} + \frac{\partial g_{il}}{\partial x^j} - \frac{\partial g_{ij}}{\partial x^l}\right)
$$

Christoffel symbols are **not tensors** (they follow inhomogeneous transformation laws). They serve to correct for "non-inertial" effects of the coordinate system.

**Symmetry**: $\Gamma^k{}_{ij} = \Gamma^k{}_{ji}$ (when the metric tensor is torsion-free)

### 6.3 Covariant Derivative

**Covariant derivative of a contravariant vector**:

$$
\nabla_i A^j = \partial_i A^j + \Gamma^j{}_{ik} A^k
$$

**Covariant derivative of a covariant vector**:

$$
\nabla_i A_j = \partial_i A_j - \Gamma^k{}_{ij} A_k
$$

**Covariant derivative of a general tensor** (add $+\Gamma$ for each upper index, $-\Gamma$ for each lower index):

$$
\nabla_i T^j{}_k = \partial_i T^j{}_k + \Gamma^j{}_{il} T^l{}_k - \Gamma^l{}_{ik} T^j{}_l
$$

**Covariant derivative of the metric tensor**: $\nabla_i g_{jk} = 0$ (metric compatibility condition)

### 6.4 Parallel Transport and Geodesic Equation

The condition for **parallel transport** of a vector $A^i$ along a curve $x^i(\tau)$:

$$
\frac{DA^i}{D\tau} = \frac{dA^i}{d\tau} + \Gamma^i{}_{jk}\frac{dx^j}{d\tau}A^k = 0
$$

A **geodesic** is a curve that parallel transports its own tangent vector:

$$
\frac{d^2x^k}{d\tau^2} + \Gamma^k{}_{ij}\frac{dx^i}{d\tau}\frac{dx^j}{d\tau} = 0
$$

In flat space, geodesics are straight lines; on a sphere, they are great circles.

### 6.5 Python: Christoffel Symbols and Geodesic Calculation

```python
import sympy as sp

# === 크리스토펠 기호 계산 함수 ===

def christoffel_symbols(g, coords):
    """
    계량 텐서로부터 크리스토펠 기호 Γ^k_{ij}를 계산한다.

    Parameters:
        g: 계량 텐서 (SymPy Matrix)
        coords: 좌표 변수 리스트
    Returns:
        Γ[k][i][j] 형태의 3차원 리스트
    """
    n = len(coords)
    g_inv = g.inv()

    Gamma = [[[sp.Integer(0) for _ in range(n)]
              for _ in range(n)]
             for _ in range(n)]

    for k in range(n):
        for i in range(n):
            for j in range(n):
                s = sp.Integer(0)
                for l in range(n):
                    s += sp.Rational(1, 2) * g_inv[k, l] * (
                        sp.diff(g[j, l], coords[i]) +
                        sp.diff(g[i, l], coords[j]) -
                        sp.diff(g[i, j], coords[l])
                    )
                Gamma[k][i][j] = sp.simplify(s)
    return Gamma

# --- 구면 좌표에서 크리스토펠 기호 ---
r, theta, phi = sp.symbols('r theta phi', positive=True)
g_sph = sp.diag(1, r**2, r**2 * sp.sin(theta)**2)
coords_sph = [r, theta, phi]

Gamma_sph = christoffel_symbols(g_sph, coords_sph)

print("구면 좌표의 0이 아닌 크리스토펠 기호:")
names = ['r', 'θ', 'φ']
for k in range(3):
    for i in range(3):
        for j in range(i, 3):
            if Gamma_sph[k][i][j] != 0:
                print(f"  Γ^{names[k]}_{{{names[i]}{names[j]}}} = {Gamma_sph[k][i][j]}")

# --- 2차원 구면 위의 측지선 (수치 계산) ---
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def geodesic_sphere(tau, y, R_val=1.0):
    """
    단위 구면 위의 측지선 방정식.
    y = [θ, φ, dθ/dτ, dφ/dτ]
    """
    th, ph, dth, dph = y

    # 2차원 구면의 크리스토펠 기호
    # Γ^θ_{φφ} = -sinθ cosθ
    # Γ^φ_{θφ} = Γ^φ_{φθ} = cosθ/sinθ

    d2th = np.sin(th) * np.cos(th) * dph**2
    d2ph = -2.0 * np.cos(th) / np.sin(th) * dth * dph if np.sin(th) > 1e-10 else 0.0

    return [dth, dph, d2th, d2ph]

# 초기 조건: 적도에서 북동 방향으로 출발
th0, ph0 = np.pi / 2, 0.0
dth0, dph0 = -0.3, 1.0  # 북쪽으로 약간 + 동쪽으로

y0 = [th0, ph0, dth0, dph0]
sol = solve_ivp(geodesic_sphere, [0, 8], y0, max_step=0.01, dense_output=True)

# 구면 위에 측지선 표시
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# 구면 그리기
u = np.linspace(0, np.pi, 40)
v = np.linspace(0, 2*np.pi, 40)
U, V = np.meshgrid(u, v)
X = np.sin(U) * np.cos(V)
Y = np.sin(U) * np.sin(V)
Z = np.cos(U)
ax.plot_surface(X, Y, Z, alpha=0.15, color='lightblue')

# 측지선 (대원)
th_geo = sol.y[0]
ph_geo = sol.y[1]
xg = np.sin(th_geo) * np.cos(ph_geo)
yg = np.sin(th_geo) * np.sin(ph_geo)
zg = np.cos(th_geo)
ax.plot(xg, yg, zg, 'r-', linewidth=2.5, label='측지선 (대원)')
ax.plot([xg[0]], [yg[0]], [zg[0]], 'go', markersize=8, label='출발점')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('구면 위의 측지선')
ax.legend()
plt.tight_layout()
plt.savefig('geodesic_sphere.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 7. Curvature Tensor

### 7.1 Riemann Curvature Tensor

The **Riemann curvature tensor** $R^l{}_{kij}$ is defined by the non-commutativity of covariant derivatives:

$$
(\nabla_i \nabla_j - \nabla_j \nabla_i)A^l = R^l{}_{kij}A^k
$$

Explicit expression using Christoffel symbols:

$$
R^l{}_{kij} = \partial_i \Gamma^l{}_{jk} - \partial_j \Gamma^l{}_{ik} + \Gamma^l{}_{im}\Gamma^m{}_{jk} - \Gamma^l{}_{jm}\Gamma^m{}_{ik}
$$

### 7.2 Geometric Meaning

The Riemann curvature tensor measures **the extent to which a vector changes when parallel transported along a closed path**.

- Flat space: $R^l{}_{kij} = 0$ (parallel transport is path-independent)
- Curved space: $R^l{}_{kij} \ne 0$ (path-dependent)

### 7.3 Ricci Tensor, Scalar Curvature, and Einstein Tensor

**Ricci tensor**: contraction of the Riemann tensor

$$
R_{ij} = R^k{}_{ikj}
$$

**Scalar curvature (Ricci scalar)**: contraction of the Ricci tensor

$$
R = g^{ij}R_{ij}
$$

**Gaussian curvature on 2D surfaces**:

$$
K = \frac{R}{2} \quad (\text{in 2D})
$$

- Sphere ($r = a$): $K = 1/a^2 > 0$
- Plane: $K = 0$
- Hyperboloid: $K < 0$

**Einstein tensor**:

$$
G_{ij} = R_{ij} - \frac{1}{2}g_{ij}R
$$

$G_{ij}$ satisfies $\nabla_i G^{ij} = 0$ (by Bianchi identities), which is directly related to energy-momentum conservation.

### 7.4 Python: Curvature Calculation

```python
import sympy as sp

# === 리만 곡률 텐서 계산 함수 ===

def riemann_tensor(Gamma, coords):
    """
    크리스토펠 기호로부터 리만 곡률 텐서 R^l_{kij}를 계산한다.
    """
    n = len(coords)
    R = [[[[sp.Integer(0) for _ in range(n)]
           for _ in range(n)]
          for _ in range(n)]
         for _ in range(n)]

    for l in range(n):
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    # ∂_i Γ^l_{jk} - ∂_j Γ^l_{ik}
                    term = sp.diff(Gamma[l][j][k], coords[i]) - \
                           sp.diff(Gamma[l][i][k], coords[j])
                    # + Γ^l_{im} Γ^m_{jk} - Γ^l_{jm} Γ^m_{ik}
                    for m in range(n):
                        term += Gamma[l][i][m] * Gamma[m][j][k] - \
                                Gamma[l][j][m] * Gamma[m][i][k]
                    R[l][k][i][j] = sp.simplify(term)
    return R

def ricci_tensor(R_riem, n):
    """리치 텐서 R_{ij} = R^k_{ikj}"""
    Ric = sp.zeros(n, n)
    for i in range(n):
        for j in range(n):
            s = sp.Integer(0)
            for k in range(n):
                s += R_riem[k][i][k][j]
            Ric[i, j] = sp.simplify(s)
    return Ric

def scalar_curvature(Ric, g_inv, n):
    """스칼라 곡률 R = g^{ij} R_{ij}"""
    R_scalar = sp.Integer(0)
    for i in range(n):
        for j in range(n):
            R_scalar += g_inv[i, j] * Ric[i, j]
    return sp.simplify(R_scalar)

# --- 2차원 구면 (r = a)의 가우스 곡률 계산 ---
theta, phi = sp.symbols('theta phi', positive=True)
a = sp.Symbol('a', positive=True)

g_sphere = sp.diag(a**2, a**2 * sp.sin(theta)**2)
coords_sphere = [theta, phi]

# 크리스토펠 기호
Gamma_sp = christoffel_symbols(g_sphere, coords_sphere)
print("2차원 구면의 0이 아닌 크리스토펠 기호:")
snames = ['θ', 'φ']
for k in range(2):
    for i in range(2):
        for j in range(i, 2):
            if Gamma_sp[k][i][j] != 0:
                print(f"  Γ^{snames[k]}_{{{snames[i]}{snames[j]}}} = {Gamma_sp[k][i][j]}")

# 리만 텐서
R_riem_sp = riemann_tensor(Gamma_sp, coords_sphere)

# 리치 텐서
Ric_sp = ricci_tensor(R_riem_sp, 2)
print(f"\n리치 텐서 R_{{ij}}:")
sp.pprint(Ric_sp)

# 스칼라 곡률
g_sp_inv = g_sphere.inv()
R_sc = scalar_curvature(Ric_sp, g_sp_inv, 2)
print(f"\n스칼라 곡률 R = {R_sc}")

# 가우스 곡률 (2차원에서 K = R/2)
K = sp.simplify(R_sc / 2)
print(f"가우스 곡률 K = R/2 = {K}")
# 출력: K = 1/a^2 (양의 일정한 곡률 → 구면)
```

---

## 8. Physical Applications

### 8.1 Continuum Mechanics: Stress Tensor

The **Cauchy stress tensor** $\sigma_{ij}$ describes the stress acting on any surface within a continuum. When the normal direction of the surface is $\hat{n}$, the force per unit area (traction) acting on that surface is:

$$
t_i = \sigma_{ij} n_j
$$

$\sigma_{ij}$ is a symmetric tensor ($\sigma_{ij} = \sigma_{ji}$, by conservation of angular momentum), with diagonal components representing normal stress and off-diagonal components representing shear stress.

```python
import numpy as np
import matplotlib.pyplot as plt

# === 2D 응력 텐서의 모어 원 (Mohr's Circle) ===
# 응력 텐서 σ = [[σ_xx, τ_xy], [τ_xy, σ_yy]]
sigma_xx, sigma_yy, tau_xy = 50.0, 20.0, 15.0
sigma = np.array([[sigma_xx, tau_xy],
                   [tau_xy, sigma_yy]])

# 주응력 (고유값)
eigenvalues, eigenvectors = np.linalg.eigh(sigma)
sigma_1 = eigenvalues[1]  # 최대 주응력
sigma_2 = eigenvalues[0]  # 최소 주응력
print(f"주응력: σ₁ = {sigma_1:.2f} MPa, σ₂ = {sigma_2:.2f} MPa")
print(f"주응력 방향:\n{eigenvectors}")

# 모어 원 그리기
center = (sigma_1 + sigma_2) / 2
radius = (sigma_1 - sigma_2) / 2

fig, ax = plt.subplots(figsize=(8, 6))
circle = plt.Circle((center, 0), radius, fill=False, color='blue', linewidth=2)
ax.add_patch(circle)

# 원래 응력 상태 표시
ax.plot(sigma_xx, tau_xy, 'ro', markersize=8, label=f'(σ_xx, τ_xy) = ({sigma_xx}, {tau_xy})')
ax.plot(sigma_yy, -tau_xy, 'go', markersize=8, label=f'(σ_yy, -τ_xy) = ({sigma_yy}, {-tau_xy})')
ax.plot([sigma_1, sigma_2], [0, 0], 'k^', markersize=10, label=f'주응력 σ₁={sigma_1:.1f}, σ₂={sigma_2:.1f}')

ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
ax.set_xlabel('법선 응력 σ (MPa)')
ax.set_ylabel('전단 응력 τ (MPa)')
ax.set_title('모어 원 (Mohr\'s Circle)')
ax.set_aspect('equal')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('mohr_circle.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 8.2 Electromagnetism: Electromagnetic Field Tensor $F_{\mu\nu}$

In special relativistic electromagnetism, the electric field $\mathbf{E}$ and magnetic field $\mathbf{B}$ are unified into a single **antisymmetric rank-2 tensor** $F_{\mu\nu}$:

$$
F_{\mu\nu} = \begin{pmatrix} 0 & -E_x/c & -E_y/c & -E_z/c \\ E_x/c & 0 & -B_z & B_y \\ E_y/c & B_z & 0 & -B_x \\ E_z/c & -B_y & B_x & 0 \end{pmatrix}
$$

Tensor form of Maxwell's equations:

$$
\partial_\mu F^{\mu\nu} = \mu_0 J^\nu \quad (\text{inhomogeneous equations: Gauss's law + Ampère-Maxwell law})
$$

$$
\partial_{[\lambda} F_{\mu\nu]} = 0 \quad (\text{homogeneous equations: Faraday's law + magnetic Gauss's law})
$$

This representation makes covariance under Lorentz transformations manifest.

```python
import numpy as np

# === 전자기장 텐서 구성 및 로렌츠 변환 ===

c = 1.0  # 자연 단위 (c = 1)

def em_field_tensor(E, B):
    """전기장 E와 자기장 B로부터 전자기장 텐서 F_μν를 구성한다."""
    Ex, Ey, Ez = E
    Bx, By, Bz = B
    F = np.array([
        [0,      -Ex/c,  -Ey/c,  -Ez/c],
        [Ex/c,    0,     -Bz,     By  ],
        [Ey/c,    Bz,     0,     -Bx  ],
        [Ez/c,   -By,     Bx,     0   ]
    ])
    return F

# x 방향 전기장만 있는 경우
E = np.array([1.0, 0.0, 0.0])
B = np.array([0.0, 0.0, 0.0])
F = em_field_tensor(E, B)
print("전자기장 텐서 F_μν (순수 전기장):")
print(F)

# 로렌츠 변환 (x 방향, 속도 v = 0.6c)
v = 0.6
gamma = 1.0 / np.sqrt(1 - v**2)
beta = v

# 로렌츠 변환 행렬 Λ^μ'_ν
Lambda = np.array([
    [gamma,      -gamma*beta, 0, 0],
    [-gamma*beta, gamma,      0, 0],
    [0,           0,          1, 0],
    [0,           0,          0, 1]
])

# 텐서 변환: F'^{μν} = Λ^μ_α Λ^ν_β F^{αβ}
# 먼저 F^{μν} = η^{μα} η^{νβ} F_{αβ} (민코프스키 계량으로 인덱스 올림)
eta = np.diag([-1, 1, 1, 1])  # 민코프스키 계량 (-,+,+,+)
F_up = eta @ F @ eta           # F^{μν}

F_prime_up = Lambda @ F_up @ Lambda.T
F_prime = eta @ F_prime_up @ eta  # F'_{μν}로 내림

print(f"\n로렌츠 변환 후 (v = {v}c):")
print(f"F'_μν:")
print(np.round(F_prime, 4))

# 변환된 전기장과 자기장 추출
E_prime = np.array([F_prime[0, 1], F_prime[0, 2], F_prime[0, 3]]) * (-c)
B_prime = np.array([F_prime[2, 3], F_prime[3, 1], F_prime[1, 2]])
print(f"\n변환된 전기장: E' = {np.round(E_prime, 4)}")
print(f"변환된 자기장: B' = {np.round(B_prime, 4)}")
print("순수 전기장이 로렌츠 변환에 의해 자기장 성분을 획득함!")

# 로렌츠 불변량 검증
inv1 = -0.5 * np.einsum('ij,ij->', F, F)   # F_{μν}F^{μν}/2
inv1_prime = -0.5 * np.einsum('ij,ij->', F_prime, F_prime)
print(f"\n로렌츠 불변량 F_μν F^μν: 원래 = {inv1:.4f}, 변환 후 = {inv1_prime:.4f}")
```

### 8.3 General Relativity: Einstein Field Equations

The core of general relativity, the **Einstein field equations**, connects the geometry of spacetime (curvature) with matter-energy distribution:

$$
G_{\mu\nu} + \Lambda g_{\mu\nu} = \frac{8\pi G}{c^4}T_{\mu\nu}
$$

Where:
- $G_{\mu\nu} = R_{\mu\nu} - \frac{1}{2}g_{\mu\nu}R$: Einstein tensor (geometry)
- $\Lambda$: cosmological constant
- $T_{\mu\nu}$: energy-momentum tensor (matter)

**Schwarzschild metric**: spherically symmetric vacuum solution

$$
ds^2 = -\left(1 - \frac{r_s}{r}\right)c^2 dt^2 + \left(1 - \frac{r_s}{r}\right)^{-1}dr^2 + r^2 d\theta^2 + r^2\sin^2\theta \, d\phi^2
$$

where $r_s = 2GM/c^2$ is the Schwarzschild radius.

```python
import sympy as sp

# === 슈바르츠실트 계량의 크리스토펠 기호 계산 ===
t, r, theta, phi = sp.symbols('t r theta phi')
r_s, c_sym = sp.symbols('r_s c', positive=True)

# 슈바르츠실트 계량 텐서 (대각)
f = 1 - r_s / r  # f(r) = 1 - r_s/r

g_schw = sp.diag(
    -f * c_sym**2,   # g_{tt}
    1 / f,           # g_{rr}
    r**2,            # g_{θθ}
    r**2 * sp.sin(theta)**2  # g_{φφ}
)
coords_schw = [t, r, theta, phi]

print("슈바르츠실트 계량 텐서:")
sp.pprint(g_schw)

# 크리스토펠 기호 계산 (시간 소요 가능)
Gamma_schw = christoffel_symbols(g_schw, coords_schw)

print("\n슈바르츠실트 계량의 0이 아닌 크리스토펠 기호:")
coord_names = ['t', 'r', 'θ', 'φ']
for k in range(4):
    for i in range(4):
        for j in range(i, 4):
            val = Gamma_schw[k][i][j]
            if val != 0:
                print(f"  Γ^{coord_names[k]}_{{{coord_names[i]}{coord_names[j]}}} = {val}")
```

---

## Practice Problems

### Problem 1: Tensor Transformation

Given the 2D coordinate transformation $x' = x\cosh\alpha + y\sinh\alpha$, $y' = x\sinh\alpha + y\cosh\alpha$ (Lorentz boost):

(a) Find the transformation matrix $\frac{\partial x'^i}{\partial x^j}$.

(b) Transform the vector $A^i = (3, 4)$ to the new coordinate system ($\alpha = 0.5$).

(c) Transform the tensor $T^{ij} = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$ to the new coordinate system.

### Problem 2: Einstein Summation Convention

Write the following tensor expressions using Einstein summation convention (without summation symbols) and expand them in 3D:

(a) Dot product of vectors $\sum_i A^i B_i$

(b) Matrix multiplication $\sum_k M^i{}_k N^k{}_j$

(c) Reduce $\varepsilon_{ijk}\varepsilon_{imn}$ using the $\delta$-identity.

### Problem 3: Metric Tensor and Index Raising/Lowering

In 2D polar coordinates $(r, \theta)$ with $ds^2 = dr^2 + r^2 d\theta^2$:

(a) Find the metric tensor $g_{ij}$ and inverse metric tensor $g^{ij}$.

(b) Find the covariant components $A_i = g_{ij}A^j$ of the contravariant vector $A^i = (A^r, A^\theta) = (2, 1/r)$.

(c) Calculate $A^i A_i$ and verify that the vector magnitude squared is a coordinate-independent invariant.

### Problem 4: Christoffel Symbols

For the 2D polar coordinate metric tensor $g = \text{diag}(1, r^2)$:

(a) Calculate all Christoffel symbols $\Gamma^k{}_{ij}$.

(b) Calculate the covariant divergence $\nabla_i A^i$ of the vector field $A^r = \cos\theta$, $A^\theta = -\sin\theta / r$.

(c) Verify that this result matches the divergence in Cartesian coordinates.

### Problem 5: Geodesics

The metric of a 2D pseudosphere ($K = -1$) is given by $ds^2 = du^2 + e^{-2u}dv^2$.

(a) Find the Christoffel symbols.

(b) Calculate the Gaussian curvature $K$ and verify that $K = -1$.

(c) Write the geodesic equations and determine whether curves with $u = \text{const}$ are geodesics.

### Problem 6: Curvature Tensor

The metric of a torus surface is given in parameters $(\theta, \phi)$ as:

$$
ds^2 = a^2 d\theta^2 + (R + a\cos\theta)^2 d\phi^2
$$

where $R$ is the major radius and $a$ is the minor radius.

(a) Find the Gaussian curvature $K(\theta)$.

(b) Classify regions where $K > 0$, $K = 0$, and $K < 0$ by $\theta$ values.

(c) Verify that the Gauss-Bonnet theorem $\int K \, dA = 2\pi\chi$ gives $\chi = 0$ (Euler characteristic of the torus).

### Problem 7: Electromagnetic Field Tensor

Given electric field $\mathbf{E} = E_0 \hat{x}$ and magnetic field $\mathbf{B} = B_0 \hat{z}$:

(a) Construct the electromagnetic field tensor $F_{\mu\nu}$.

(b) Calculate the Lorentz invariants $F_{\mu\nu}F^{\mu\nu}$ and $\frac{1}{2}\varepsilon^{\mu\nu\rho\sigma}F_{\mu\nu}F_{\rho\sigma}$.

(c) Find $\mathbf{E}'$ and $\mathbf{B}'$ after a Lorentz boost in the $x$-direction with $v = 0.8c$.

### Problem 8: Principal Axes of Stress Tensor

A 3D stress tensor is given as:

$$
\sigma_{ij} = \begin{pmatrix} 100 & 30 & 0 \\ 30 & 50 & 20 \\ 0 & 20 & 80 \end{pmatrix} \text{ (MPa)}
$$

(a) Find the principal stresses and principal directions (eigenvalues/eigenvectors).

(b) Find the maximum shear stress.

(c) Calculate the von Mises stress: $\sigma_v = \sqrt{\frac{1}{2}[(\sigma_1-\sigma_2)^2 + (\sigma_2-\sigma_3)^2 + (\sigma_3-\sigma_1)^2]}$.

---

## Advanced Topics

### Differential Forms

The modern language of tensor analysis, **differential forms**, systematically handles antisymmetric covariant tensors. Using exterior algebra and the exterior derivative $d$:

- 0-form = scalar function
- 1-form = $\omega = A_i dx^i$
- 2-form = $F = \frac{1}{2}F_{ij}dx^i \wedge dx^j$

Maxwell's equations become extremely concise in differential forms: $dF = 0$, $d{*F} = J$

### Lie Derivative

The **Lie derivative** $\mathcal{L}_X T$ defines the change of a tensor $T$ along a vector field $X$ in a coordinate-independent way. This is a key tool for handling symmetries and conservation laws (Killing vector fields).

### Fiber Bundles and Gauge Theory

In Yang-Mills theory, the gauge connection is a generalization of Christoffel symbols, and curvature corresponds to the field strength tensor. This forms the mathematical foundation of the Standard Model.

### Computational Tools

| Tool | Language | Description |
|------|----------|-------------|
| `sympy.diffgeom` | Python | Differential geometry calculations |
| `einsteinpy` | Python | General relativity tensor calculations |
| `xAct` / `xTensor` | Mathematica | Professional tensor algebra system |
| `SageManifolds` | SageMath | Differential manifold calculations |
| `cadabra2` | Python/C++ | Field theory tensor calculations |

### Recommended References

1. **Boas, M. L.** (2005). *Mathematical Methods in the Physical Sciences*, 3rd ed., Chapter 10. Wiley.
2. **Carroll, S. M.** (2019). *Spacetime and Geometry*, 2nd ed. Cambridge University Press.
   - The best textbook for tensor analysis in general relativity
3. **Schutz, B.** (2009). *A First Course in General Relativity*, 2nd ed. Cambridge University Press.
4. **Arfken, G. B., Weber, H. J., & Harris, F. E.** (2012). *Mathematical Methods for Physicists*, 7th ed., Chapter 3. Academic Press.
5. **Nakahara, M.** (2003). *Geometry, Topology and Physics*, 2nd ed. CRC Press.
   - Mathematical foundations of differential forms, fiber bundles, and gauge theory

---

**Previous**: [17. Calculus of Variations](17_Calculus_of_Variations.md)
**Next**: Course complete! Return to [00. Overview](00_Overview.md)
