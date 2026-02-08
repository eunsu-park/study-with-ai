# 04. Partial Differentiation

> **Boas Chapter 4** — Partial differentiation is a core tool across all physical sciences, including thermodynamics, electromagnetism, and fluid dynamics, because almost every law of physics is described by multivariable functions.

---

## Learning Objectives

Upon completing this lesson, you will be able to:

- Understand the definition and notation of **partial derivatives** and compute partial derivatives of multivariable functions
- Apply the **multivariable chain rule** to find derivatives of composite functions
- Perform **implicit differentiation** and apply it to physical relationships
- Classify **extrema and saddle points** using the second derivative test
- Solve constrained optimization problems using the **Lagrange multiplier method**
- Understand the conditions for **exact differentials** and derive Maxwell relations in thermodynamics
- Expand **multivariate Taylor series** and apply them to physics approximations

---

## 1. Basics of Partial Differentiation

### 1.1 Multivariable Functions and Partial Derivatives

The **partial derivative** of a **multivariable function** $f(x, y, z, \ldots)$ with respect to a specific variable is obtained by treating all other variables as constants and differentiating only with respect to that variable:

$$
\frac{\partial f}{\partial x} = \lim_{\Delta x \to 0} \frac{f(x + \Delta x, y, z, \ldots) - f(x, y, z, \ldots)}{\Delta x}
$$

**Notation**: $\frac{\partial f}{\partial x}$, $f_x$, $\partial_x f$

**Higher-order partial derivatives**:

$$
\frac{\partial^2 f}{\partial x^2}, \quad \frac{\partial^2 f}{\partial x \partial y}, \quad \frac{\partial^2 f}{\partial y \partial x}
$$

> **Schwarz's theorem**: If the second partial derivatives of $f$ are continuous, the order of differentiation can be exchanged:
> $$\frac{\partial^2 f}{\partial x \partial y} = \frac{\partial^2 f}{\partial y \partial x}$$

```python
import numpy as np
import sympy as sp

# SymPy를 이용한 편미분
x, y, z = sp.symbols('x y z')
f = x**2 * y + sp.sin(x * y) + z * sp.exp(x)

print(f"f = {f}")
print(f"∂f/∂x = {sp.diff(f, x)}")
print(f"∂f/∂y = {sp.diff(f, y)}")
print(f"∂f/∂z = {sp.diff(f, z)}")

# 2차 편미분
print(f"\n∂²f/∂x² = {sp.diff(f, x, 2)}")
print(f"∂²f/∂x∂y = {sp.diff(f, x, y)}")
print(f"∂²f/∂y∂x = {sp.diff(f, y, x)}")
print(f"혼합 편미분 교환: {sp.simplify(sp.diff(f, x, y) - sp.diff(f, y, x)) == 0}")
```

### 1.2 Total Differential

The **total differential** of a function $f(x, y, z)$:

$$
df = \frac{\partial f}{\partial x} dx + \frac{\partial f}{\partial y} dy + \frac{\partial f}{\partial z} dz
$$

This is a first-order approximation of the change in $f$ due to independent changes in each variable.

**Physical meaning**: The infinitesimal change in pressure $P(V, T)$ is:
$$
dP = \left(\frac{\partial P}{\partial V}\right)_T dV + \left(\frac{\partial P}{\partial T}\right)_V dT
$$

> The subscript indicates the **variable being held constant**. This notation is very important in thermodynamics.

```python
# 이상기체 PV = nRT에서의 전미분
P, V, T, n, R = sp.symbols('P V T n R', positive=True)

# P = nRT/V
P_expr = n * R * T / V

dP_dV = sp.diff(P_expr, V)
dP_dT = sp.diff(P_expr, T)

print(f"P = {P_expr}")
print(f"(∂P/∂V)_T = {dP_dV}")
print(f"(∂P/∂T)_V = {dP_dT}")
print(f"dP = ({dP_dV})dV + ({dP_dT})dT")
```

### 1.3 Gradient

The **gradient** of a scalar function $f(x, y, z)$ is a vector whose components are the partial derivatives:

$$
\nabla f = \left(\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}, \frac{\partial f}{\partial z}\right)
$$

**Meaning of the gradient**:
- **Direction** of $\nabla f$: direction of steepest increase of $f$
- $|\nabla f|$: rate of increase in that direction (maximum directional derivative)
- $\nabla f$ is **perpendicular** to the level surface $f = \text{const}$

```python
import matplotlib.pyplot as plt

# 2D 기울기 시각화
f_2d = x**2 + y**2  # f(x,y) = x² + y²
grad_f = [sp.diff(f_2d, x), sp.diff(f_2d, y)]
print(f"f = {f_2d}")
print(f"∇f = ({grad_f[0]}, {grad_f[1]})")

# 수치 시각화
X, Y = np.meshgrid(np.linspace(-2, 2, 20), np.linspace(-2, 2, 20))
Z = X**2 + Y**2
U = 2 * X  # ∂f/∂x
V_field = 2 * Y  # ∂f/∂y

fig, ax = plt.subplots(figsize=(8, 7))
contour = ax.contour(X, Y, Z, levels=10, cmap='viridis')
ax.clabel(contour, inline=True, fontsize=8)
ax.quiver(X, Y, U, V_field, color='red', alpha=0.6, scale=50)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('등고선과 기울기 벡터 (∇f ⊥ 등고선)')
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('gradient_field.png', dpi=100, bbox_inches='tight')
plt.close()
```

---

## 2. Chain Rule

### 2.1 Multivariable Chain Rule

If $f(x, y)$ where $x = x(t)$, $y = y(t)$:

$$
\frac{df}{dt} = \frac{\partial f}{\partial x}\frac{dx}{dt} + \frac{\partial f}{\partial y}\frac{dy}{dt}
$$

More generally, if $f(x, y)$ where $x = x(s, t)$, $y = y(s, t)$:

$$
\frac{\partial f}{\partial s} = \frac{\partial f}{\partial x}\frac{\partial x}{\partial s} + \frac{\partial f}{\partial y}\frac{\partial y}{\partial s}
$$

$$
\frac{\partial f}{\partial t} = \frac{\partial f}{\partial x}\frac{\partial x}{\partial t} + \frac{\partial f}{\partial y}\frac{\partial y}{\partial t}
$$

**Matrix notation** (Jacobian):

$$
\begin{pmatrix} \frac{\partial f}{\partial s} \\ \frac{\partial f}{\partial t} \end{pmatrix}
= \begin{pmatrix} \frac{\partial x}{\partial s} & \frac{\partial y}{\partial s} \\ \frac{\partial x}{\partial t} & \frac{\partial y}{\partial t} \end{pmatrix}
\begin{pmatrix} \frac{\partial f}{\partial x} \\ \frac{\partial f}{\partial y} \end{pmatrix}
$$

```python
# 연쇄법칙 예제: 극좌표 변환
r, theta = sp.symbols('r theta', positive=True)
x_expr = r * sp.cos(theta)
y_expr = r * sp.sin(theta)

# f(x,y) = x² + y² 를 극좌표로 표현
f_xy = x**2 + y**2
f_polar = f_xy.subs([(x, x_expr), (y, y_expr)])
f_polar_simplified = sp.simplify(f_polar)
print(f"f(x,y) = {f_xy}")
print(f"f(r,θ) = {f_polar_simplified}")  # r²

# 연쇄법칙으로 ∂f/∂r 계산
df_dr_chain = sp.diff(f_xy, x).subs(x, x_expr).subs(y, y_expr) * sp.diff(x_expr, r) + \
              sp.diff(f_xy, y).subs(x, x_expr).subs(y, y_expr) * sp.diff(y_expr, r)
df_dr_direct = sp.diff(f_polar_simplified, r)

print(f"\n연쇄법칙: ∂f/∂r = {sp.simplify(df_dr_chain)}")
print(f"직접 미분: ∂f/∂r = {df_dr_direct}")

# 야코비안
J = sp.Matrix([[sp.diff(x_expr, r), sp.diff(x_expr, theta)],
               [sp.diff(y_expr, r), sp.diff(y_expr, theta)]])
print(f"\n야코비안 J =\n{J}")
print(f"det(J) = {sp.simplify(J.det())}")  # r (극좌표 야코비안)
```

### 2.2 Implicit Differentiation

When $y$ is implicitly defined as a function of $x$ by $F(x, y) = 0$:

$$
\frac{dy}{dx} = -\frac{\partial F / \partial x}{\partial F / \partial y} = -\frac{F_x}{F_y}
$$

**Derivation**: Differentiating both sides of $F(x, y(x)) = 0$ with respect to $x$:

$$
\frac{\partial F}{\partial x} + \frac{\partial F}{\partial y}\frac{dy}{dx} = 0
$$

**Extension to three variables**: If $z = z(x, y)$ from $F(x, y, z) = 0$:

$$
\frac{\partial z}{\partial x} = -\frac{F_x}{F_z}, \qquad \frac{\partial z}{\partial y} = -\frac{F_y}{F_z}
$$

```python
# 음함수 미분 예제: 타원 x²/4 + y²/9 = 1
F = x**2/4 + y**2/9 - 1

dy_dx = -sp.diff(F, x) / sp.diff(F, y)
print(f"F(x,y) = {F} = 0")
print(f"dy/dx = {dy_dx}")
print(f"      = {sp.simplify(dy_dx)}")

# 검증: y = 3√(1 - x²/4) 를 직접 미분
y_explicit = 3 * sp.sqrt(1 - x**2/4)
dy_dx_explicit = sp.diff(y_explicit, x)
print(f"\n명시적 미분: dy/dx = {dy_dx_explicit}")
print(f"간소화: {sp.simplify(dy_dx_explicit)}")

# 순환 관계 (cyclic relation)
# F(x,y,z) = 0일 때:
# (∂x/∂y)_z (∂y/∂z)_x (∂z/∂x)_y = -1
print("\n=== 순환 관계 (이상기체) ===")
# PV = nRT → F(P,V,T) = PV - nRT = 0
P_sym, V_sym, T_sym = sp.symbols('P V T', positive=True)
F_gas = P_sym * V_sym - n * R * T_sym

dP_dV_T = -sp.diff(F_gas, V_sym) / sp.diff(F_gas, P_sym)  # (∂P/∂V)_T
dV_dT_P = -sp.diff(F_gas, T_sym) / sp.diff(F_gas, V_sym)  # (∂V/∂T)_P
dT_dP_V = -sp.diff(F_gas, P_sym) / sp.diff(F_gas, T_sym)  # (∂T/∂P)_V

product = sp.simplify(dP_dV_T * dV_dT_P * dT_dP_V)
print(f"(∂P/∂V)_T = {dP_dV_T}")
print(f"(∂V/∂T)_P = {dV_dT_P}")
print(f"(∂T/∂P)_V = {dT_dP_V}")
print(f"곱 = {product}")  # -1
```

---

## 3. Extrema and Saddle Points

### 3.1 Necessary Condition for Extrema

A **critical point** of a multivariable function $f(x, y)$: $\nabla f = 0$

$$
\frac{\partial f}{\partial x} = 0, \quad \frac{\partial f}{\partial y} = 0
$$

### 3.2 Second Derivative Test

At a critical point $(x_0, y_0)$, the **Hessian matrix**:

$$
H = \begin{pmatrix} f_{xx} & f_{xy} \\ f_{yx} & f_{yy} \end{pmatrix}
$$

**Discriminant**:

$$
D = f_{xx} f_{yy} - (f_{xy})^2 = \det(H)
$$

| Condition | Conclusion |
|------|------|
| $D > 0$, $f_{xx} > 0$ | **Local minimum** |
| $D > 0$, $f_{xx} < 0$ | **Local maximum** |
| $D < 0$ | **Saddle point** |
| $D = 0$ | Test inconclusive (higher order needed) |

> Connection to positive definiteness: $D > 0$ and $f_{xx} > 0$ is equivalent to the Hessian $H$ being positive definite. This means the quadratic form $\frac{1}{2}\delta\mathbf{x}^T H \delta\mathbf{x} > 0$, hence a minimum.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 안장점을 포함하는 함수
f_saddle = x**2 - y**2  # 쌍곡 포물면

# 임계점 찾기
fx = sp.diff(f_saddle, x)
fy = sp.diff(f_saddle, y)
critical = sp.solve([fx, fy], [x, y])
print(f"f(x,y) = {f_saddle}")
print(f"임계점: {critical}")

# 헤시안
fxx = sp.diff(f_saddle, x, 2)
fxy = sp.diff(f_saddle, x, y)
fyy = sp.diff(f_saddle, y, 2)
D = fxx * fyy - fxy**2
print(f"\nH = [[{fxx}, {fxy}], [{fxy}, {fyy}]]")
print(f"D = {D}")
print(f"D < 0 → 안장점")

# 다른 예: 극소, 극대, 안장점을 모두 가진 함수
g = x**3 - 3*x*y**2  # 원숭이 안장 (monkey saddle)
h = (x**2 + y**2)**2 - 2*(x**2 - y**2)  # 두 극소, 두 안장점

# 3D 시각화
X_grid = np.linspace(-2, 2, 100)
Y_grid = np.linspace(-2, 2, 100)
X_mesh, Y_mesh = np.meshgrid(X_grid, Y_grid)

fig = plt.figure(figsize=(15, 5))

# 극소: x² + y²
ax1 = fig.add_subplot(131, projection='3d')
Z1 = X_mesh**2 + Y_mesh**2
ax1.plot_surface(X_mesh, Y_mesh, Z1, cmap='viridis', alpha=0.7)
ax1.set_title('극소: f = x² + y²')
ax1.set_xlabel('x')
ax1.set_ylabel('y')

# 안장점: x² - y²
ax2 = fig.add_subplot(132, projection='3d')
Z2 = X_mesh**2 - Y_mesh**2
ax2.plot_surface(X_mesh, Y_mesh, Z2, cmap='RdBu', alpha=0.7)
ax2.set_title('안장점: f = x² - y²')
ax2.set_xlabel('x')
ax2.set_ylabel('y')

# 극대: -(x² + y²)
ax3 = fig.add_subplot(133, projection='3d')
Z3 = -(X_mesh**2 + Y_mesh**2)
ax3.plot_surface(X_mesh, Y_mesh, Z3, cmap='plasma', alpha=0.7)
ax3.set_title('극대: f = -(x² + y²)')
ax3.set_xlabel('x')
ax3.set_ylabel('y')

plt.tight_layout()
plt.savefig('critical_points.png', dpi=100, bbox_inches='tight')
plt.close()
```

### 3.3 Practical Example: Distance Minimization

Problem of minimizing the distance from point $(x, y, z)$ to plane $2x + y - z = 5$.

Method 1: Minimize the squared distance $d^2 = (x-a)^2 + (y-b)^2 + (z-c)^2$ (covered in the next section on Lagrange multipliers)

Method 2: Use the distance formula directly $d = |2x + y - z - 5| / \sqrt{4 + 1 + 1}$

---

## 4. Lagrange Multipliers

### 4.1 Optimization with Equality Constraints

**Problem**: Find the extrema of $f(x, y)$ subject to the constraint $g(x, y) = 0$.

**Lagrange function**:

$$
\mathcal{L}(x, y, \lambda) = f(x, y) - \lambda \, g(x, y)
$$

**Necessary conditions**:

$$
\frac{\partial \mathcal{L}}{\partial x} = 0, \quad \frac{\partial \mathcal{L}}{\partial y} = 0, \quad \frac{\partial \mathcal{L}}{\partial \lambda} = 0
$$

The third condition is the constraint $g = 0$ itself.

**Geometric meaning**: At an extremum, $\nabla f$ and $\nabla g$ are **parallel**, i.e., $\nabla f = \lambda \nabla g$.

```python
# 예제: 원 x² + y² = 1 위에서 f(x,y) = x + y의 최댓값
lam = sp.Symbol('lambda')

f_obj = x + y
g_constraint = x**2 + y**2 - 1

L = f_obj - lam * g_constraint

# 연립방정식
eqs = [sp.diff(L, x), sp.diff(L, y), sp.diff(L, lam)]
print("라그랑주 조건:")
for eq in eqs:
    print(f"  {eq} = 0")

solutions = sp.solve(eqs, [x, y, lam])
print(f"\n해: {solutions}")

for sol in solutions:
    val = f_obj.subs([(x, sol[0]), (y, sol[1])])
    print(f"  ({sol[0]}, {sol[1]}): f = {val}, λ = {sol[2]}")
```

### 4.2 Multiple Constraints

With $m$ constraints $g_1 = 0, g_2 = 0, \ldots, g_m = 0$:

$$
\mathcal{L} = f - \lambda_1 g_1 - \lambda_2 g_2 - \cdots - \lambda_m g_m
$$

**Physics Application: Entropy Maximization**

In statistical mechanics, the **Boltzmann distribution** is the solution to the following optimization problem:

- Maximize: Entropy $S = -k_B \sum p_i \ln p_i$
- Constraint 1: $\sum p_i = 1$ (normalization of probabilities)
- Constraint 2: $\sum p_i E_i = \langle E \rangle$ (fixed average energy)

```python
# 라그랑주 승수법으로 볼츠만 분포 유도 (이산 경우)
# 3개 에너지 준위 E₁=0, E₂=1, E₃=2
p1, p2, p3 = sp.symbols('p1 p2 p3', positive=True)
lam1, lam2 = sp.symbols('lambda1 lambda2')

E_levels = [0, 1, 2]
probs = [p1, p2, p3]

# 엔트로피 (부호 반전하여 최소화로 변환)
S = sum(p * sp.ln(p) for p in probs)  # 최소화할 것 (-S)

# 구속 조건
g1 = p1 + p2 + p3 - 1
g2 = 0*p1 + 1*p2 + 2*p3 - sp.Symbol('E_avg')

L = S + lam1 * g1 + lam2 * g2

eqs = [sp.diff(L, p) for p in probs] + [g1]
print("라그랑주 조건 (∂L/∂pᵢ = 0):")
for i, eq in enumerate(eqs[:3]):
    print(f"  ln(p{i+1}) + 1 + λ₁ + {E_levels[i]}λ₂ = 0")

print("\n→ pᵢ ∝ exp(-λ₂ Eᵢ) : 볼츠만 분포!")
```

### 4.3 Numerical Optimization with SciPy

```python
from scipy.optimize import minimize

# 타원 x²/4 + y²/9 = 1 위에서 원점으로부터 최대/최소 거리
def objective(xy):
    return -(xy[0]**2 + xy[1]**2)  # 최대화 → 부호 반전

def constraint(xy):
    return xy[0]**2/4 + xy[1]**2/9 - 1  # = 0

from scipy.optimize import minimize
result = minimize(objective, x0=[1, 1],
                  constraints={'type': 'eq', 'fun': constraint},
                  method='SLSQP')

print(f"타원 위 원점에서 가장 먼 점: ({result.x[0]:.4f}, {result.x[1]:.4f})")
print(f"최대 거리: {np.sqrt(-result.fun):.4f}")

# 가장 가까운 점
result_min = minimize(lambda xy: xy[0]**2 + xy[1]**2, x0=[1, 0.1],
                       constraints={'type': 'eq', 'fun': constraint},
                       method='SLSQP')
print(f"타원 위 원점에서 가장 가까운 점: ({result_min.x[0]:.4f}, {result_min.x[1]:.4f})")
print(f"최소 거리: {np.sqrt(result_min.fun):.4f}")
```

---

## 5. Exact Differentials and Thermodynamics

### 5.1 Condition for Exact Differentials

For a differential form $M(x,y)\,dx + N(x,y)\,dy$ to be an **exact differential**, i.e., for there to exist a function $f$ such that $df = M\,dx + N\,dy$:

$$
\frac{\partial M}{\partial y} = \frac{\partial N}{\partial x}
$$

This is a direct consequence of **Schwarz's theorem**: If $M = \partial f / \partial x$, $N = \partial f / \partial y$, then $\partial M / \partial y = \partial^2 f / \partial y \partial x = \partial^2 f / \partial x \partial y = \partial N / \partial x$.

```python
# 완전미분 판별 예제
# (2xy + 3)dx + (x² + 4y)dy
M = 2*x*y + 3
N = x**2 + 4*y

dM_dy = sp.diff(M, y)
dN_dx = sp.diff(N, x)
print(f"M = {M}, N = {N}")
print(f"∂M/∂y = {dM_dy}, ∂N/∂x = {dN_dx}")
print(f"완전미분: {dM_dy == dN_dx}")

# 포텐셜 함수 f 구하기
# f = ∫M dx = x²y + 3x + h(y)
# ∂f/∂y = x² + h'(y) = N = x² + 4y
# → h'(y) = 4y → h(y) = 2y²
f_potential = x**2*y + 3*x + 2*y**2
print(f"\n포텐셜 함수: f = {f_potential}")
print(f"검증: ∂f/∂x = {sp.diff(f_potential, x)} (= M)")
print(f"검증: ∂f/∂y = {sp.diff(f_potential, y)} (= N)")
```

### 5.2 Thermodynamics and Maxwell Relations

Combining the first and second laws of thermodynamics:

$$
dU = TdS - PdV
$$

Since this is an exact differential ($U = U(S, V)$):

$$
T = \left(\frac{\partial U}{\partial S}\right)_V, \quad -P = \left(\frac{\partial U}{\partial V}\right)_S
$$

From the condition for exact differentials $\partial^2 U / \partial V \partial S = \partial^2 U / \partial S \partial V$:

$$
\left(\frac{\partial T}{\partial V}\right)_S = -\left(\frac{\partial P}{\partial S}\right)_V
$$

This is one of the **Maxwell relations**.

**Four thermodynamic potentials and Maxwell relations**:

| Potential | Differential | Maxwell Relation |
|--------|------|------------|
| $U(S,V)$ | $dU = TdS - PdV$ | $\left(\frac{\partial T}{\partial V}\right)_S = -\left(\frac{\partial P}{\partial S}\right)_V$ |
| $H(S,P)$ | $dH = TdS + VdP$ | $\left(\frac{\partial T}{\partial P}\right)_S = \left(\frac{\partial V}{\partial S}\right)_P$ |
| $F(T,V)$ | $dF = -SdT - PdV$ | $\left(\frac{\partial S}{\partial V}\right)_T = \left(\frac{\partial P}{\partial T}\right)_V$ |
| $G(T,P)$ | $dG = -SdT + VdP$ | $-\left(\frac{\partial S}{\partial P}\right)_T = \left(\frac{\partial V}{\partial T}\right)_P$ |

> **Mnemonic (Born square)**: By placing internal energy at the vertices and natural variables at the edges, you can systematically derive Maxwell relations.

```python
# 맥스웰 관계식 검증 (이상기체)
# PV = nRT → P = nRT/V
S, V_td = sp.symbols('S V', positive=True)
k, N_a = sp.symbols('k N_A', positive=True)

# 이상기체의 내부 에너지: U = (3/2)nRT = (3/2)nR * T(S,V)
# 간단한 모델: U(S,V) = A * exp(2S/3nR) * V^(-2/3)
# 여기서는 맥스웰 관계식의 구조만 확인

# 헬름홀츠 자유에너지 F(T,V)에서
# dF = -SdT - PdV
# 이상기체: F = nRT(1 - ln(T/T₀)) - nRT ln(V/V₀) + const

T_td = sp.Symbol('T', positive=True)
F_ideal = n*R*T_td*(1 - sp.ln(T_td)) - n*R*T_td*sp.ln(V_td)

S_from_F = -sp.diff(F_ideal, T_td)
P_from_F = -sp.diff(F_ideal, V_td)

print(f"F = {F_ideal}")
print(f"S = -∂F/∂T = {sp.simplify(S_from_F)}")
print(f"P = -∂F/∂V = {sp.simplify(P_from_F)}")

# 맥스웰: (∂S/∂V)_T = (∂P/∂T)_V
dS_dV = sp.diff(S_from_F, V_td)
dP_dT = sp.diff(P_from_F, T_td)
print(f"\n(∂S/∂V)_T = {dS_dV}")
print(f"(∂P/∂T)_V = {dP_dT}")
print(f"맥스웰 관계 성립: {sp.simplify(dS_dV - dP_dT) == 0}")
```

---

## 6. Multivariate Taylor Series

### 6.1 Two-Variable Taylor Expansion

Expansion of $f(x, y)$ near the point $(a, b)$:

$$
f(x, y) = f(a, b) + f_x(a,b)(x-a) + f_y(a,b)(y-b) \\
+ \frac{1}{2!}\left[f_{xx}(x-a)^2 + 2f_{xy}(x-a)(y-b) + f_{yy}(y-b)^2\right] + \cdots
$$

**Vector notation**: $\mathbf{x}_0 = (a, b)$, $\delta\mathbf{x} = (x-a, y-b)$

$$
f(\mathbf{x}_0 + \delta\mathbf{x}) = f(\mathbf{x}_0) + \nabla f \cdot \delta\mathbf{x} + \frac{1}{2} \delta\mathbf{x}^T H \, \delta\mathbf{x} + \cdots
$$

where $H$ is the Hessian matrix.

### 6.2 Physics Approximations

**Small-amplitude approximation**: Potential energy of a pendulum $U(\theta) = mgl(1 - \cos\theta)$ near $\theta = 0$:

$$
U \approx mgl \cdot \frac{\theta^2}{2} + O(\theta^4) \quad (\text{harmonic oscillator})
$$

**Multivariable function approximation**: $f(x, y) = e^{x}\sin(y)$ near the origin:

$$
f \approx y + xy + \frac{1}{2}(y - \frac{y^3}{6} + \cdots) \approx y + xy - \frac{y^3}{6} + \cdots
$$

```python
# 다변수 테일러 급수
f_taylor = sp.exp(x) * sp.sin(y)

# 원점에서 3차까지 전개
taylor_2d = sp.series(sp.series(f_taylor, x, 0, 4), y, 0, 4)
print(f"f = exp(x)sin(y)")
print(f"테일러 (3차): {taylor_2d}")

# SymPy의 다변수 테일러
from sympy import O
# 수동 전개
f0 = f_taylor.subs([(x, 0), (y, 0)])
fx0 = sp.diff(f_taylor, x).subs([(x, 0), (y, 0)])
fy0 = sp.diff(f_taylor, y).subs([(x, 0), (y, 0)])
fxx0 = sp.diff(f_taylor, x, 2).subs([(x, 0), (y, 0)])
fxy0 = sp.diff(f_taylor, x, y).subs([(x, 0), (y, 0)])
fyy0 = sp.diff(f_taylor, y, 2).subs([(x, 0), (y, 0)])

T2 = f0 + fx0*x + fy0*y + sp.Rational(1,2)*(fxx0*x**2 + 2*fxy0*x*y + fyy0*y**2)
print(f"\n2차 테일러: {T2}")

# 근사 정확도 비교
import matplotlib.pyplot as plt

x_vals = np.linspace(-1, 1, 50)
y_vals = np.linspace(-np.pi/2, np.pi/2, 50)
X_mesh, Y_mesh = np.meshgrid(x_vals, y_vals)

Z_exact = np.exp(X_mesh) * np.sin(Y_mesh)
Z_approx = Y_mesh + X_mesh*Y_mesh + 0.5*(X_mesh**2*Y_mesh)  # 주요 항

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, Z, title in [(axes[0], Z_exact, '정확한 값'),
                       (axes[1], Z_approx, '2차 근사'),
                       (axes[2], Z_exact - Z_approx, '오차')]:
    c = ax.contourf(X_mesh, Y_mesh, Z, levels=20, cmap='RdBu_r')
    plt.colorbar(c, ax=ax)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)

plt.tight_layout()
plt.savefig('taylor_2d.png', dpi=100, bbox_inches='tight')
plt.close()
```

---

## 7. Change of Variables and Jacobian

### 7.1 Jacobian Matrix and Determinant

The **Jacobian matrix** of a coordinate transformation $(x, y) \to (u, v)$:

$$
J = \frac{\partial(x, y)}{\partial(u, v)} = \begin{pmatrix} \frac{\partial x}{\partial u} & \frac{\partial x}{\partial v} \\ \frac{\partial y}{\partial u} & \frac{\partial y}{\partial v} \end{pmatrix}
$$

**Jacobian** (determinant): $\left|\frac{\partial(x, y)}{\partial(u, v)}\right| = \det(J)$

**Change of variables in multiple integrals**:

$$
\iint f(x, y) \, dx \, dy = \iint f(x(u,v), y(u,v)) \left|\frac{\partial(x, y)}{\partial(u, v)}\right| du \, dv
$$

### 7.2 Jacobians of Common Coordinate Transformations

| Transformation | Jacobian |
|------|---------|
| Polar $(r, \theta)$ | $r$ |
| Cylindrical $(r, \theta, z)$ | $r$ |
| Spherical $(r, \theta, \phi)$ | $r^2 \sin\theta$ |

```python
# 야코비안 계산 예제
r, theta, phi = sp.symbols('r theta phi', positive=True)

# 구면좌표 → 직교좌표
x_sph = r * sp.sin(theta) * sp.cos(phi)
y_sph = r * sp.sin(theta) * sp.sin(phi)
z_sph = r * sp.cos(theta)

J_spherical = sp.Matrix([
    [sp.diff(x_sph, r), sp.diff(x_sph, theta), sp.diff(x_sph, phi)],
    [sp.diff(y_sph, r), sp.diff(y_sph, theta), sp.diff(y_sph, phi)],
    [sp.diff(z_sph, r), sp.diff(z_sph, theta), sp.diff(z_sph, phi)]
])

jacobian_det = sp.simplify(J_spherical.det())
print("구면좌표 야코비안 행렬:")
sp.pprint(J_spherical)
print(f"\n|J| = {jacobian_det}")  # r² sin(θ)

# 체적 요소
print(f"\ndV = |J| dr dθ dφ = {jacobian_det} dr dθ dφ")

# 응용: 구의 체적
# V = ∫₀^a ∫₀^π ∫₀^{2π} r² sin(θ) dφ dθ dr
a = sp.Symbol('a', positive=True)
volume = sp.integrate(
    sp.integrate(
        sp.integrate(r**2 * sp.sin(theta), (phi, 0, 2*sp.pi)),
        (theta, 0, sp.pi)),
    (r, 0, a))
print(f"\n구의 체적 = {volume} = {sp.simplify(volume)}")  # (4/3)πa³
```

---

## Exercises

### Basic Problems

**1.** Find all first and second partial derivatives of $f(x, y) = x^3 y^2 + \sin(xy)$ and verify that $\frac{\partial^2 f}{\partial x \partial y} = \frac{\partial^2 f}{\partial y \partial x}$.

**2.** Express the Laplacian $\nabla^2 f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2}$ in polar coordinates $(r, \theta)$ in terms of $r$ and $\theta$.

**3.** Find all critical points of $f(x, y) = x^3 + y^3 - 3xy$ and classify each type (maximum/minimum/saddle point).

### Applied Problems

**4.** Using the Lagrange multiplier method, find the maximum and minimum values of $f(x, y, z) = x + 2y + 3z$ on the sphere $x^2 + y^2 + z^2 = 1$.

**5.** For the van der Waals equation $(P + a/V^2)(V - b) = RT$, find $\left(\frac{\partial P}{\partial T}\right)_V$ and $\left(\frac{\partial V}{\partial T}\right)_P$.

**6.** Calculate the three-dimensional Gaussian integral $\iiint e^{-(ax^2 + by^2 + cz^2)} \, dx \, dy \, dz$ (integration domain: all space) using spherical coordinate transformation and the Jacobian. (Hint: separate after principal axis transformation)

---

## References

- Boas, *Mathematical Methods in the Physical Sciences*, Chapter 4
- Arfken & Weber, *Mathematical Methods for Physicists*, Chapter 1
- Callen, *Thermodynamics and an Introduction to Thermostatistics* (Maxwell relations)
- Previous lesson: [03. Linear Algebra](./03_Linear_Algebra.md) (matrices, quadratic forms)
- Next lesson: [05. Vector Analysis](./05_Vector_Analysis.md)

---

## Next Lesson

[05. Vector Analysis](./05_Vector_Analysis.md) covers gradient, divergence, curl operators and integral theorems in depth.
