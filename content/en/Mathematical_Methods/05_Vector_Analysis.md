# 05. Vector Analysis

## Learning Objectives
- Understand the physical meaning of gradient, divergence, and curl operators and be able to calculate them
- Perform line integrals and surface integrals, and explain the criteria for conservative fields
- State and apply Green's theorem, Stokes' theorem, and the divergence theorem
- Convert Maxwell's equations between integral and differential forms
- Visualize vector fields using Python (SymPy, Matplotlib) and numerically compute line/surface integrals

---

## 1. Vector Differential Operators

Vector differential operators are essential tools for describing the spatial variation of scalar fields and vector fields. In a 3D Cartesian coordinate system, the nabla operator is defined as:

$$
\nabla = \hat{x}\frac{\partial}{\partial x} + \hat{y}\frac{\partial}{\partial y} + \hat{z}\frac{\partial}{\partial z}
$$

### 1.1 Gradient (∇f)

The **gradient** of a scalar field $f(x, y, z)$ is a vector field that indicates the direction and rate at which $f$ increases most rapidly.

$$
\nabla f = \frac{\partial f}{\partial x}\hat{x} + \frac{\partial f}{\partial y}\hat{y} + \frac{\partial f}{\partial z}\hat{z}
$$

**Physical Meaning:**
- Direction: the direction of fastest increase of $f$
- Magnitude: the rate of change in that direction (maximum directional derivative)
- Always perpendicular to level surfaces

**Example:** Temperature distribution $T(x, y) = x^2 + y^2$ (2D heat source)

```python
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy.vector import CoordSys3D

# === SymPy를 이용한 해석적 gradient 계산 ===
N = CoordSys3D('N')
x, y, z = sp.symbols('x y z')

# 스칼라장 정의
f = x**2 + y**2

# gradient 계산
grad_f = sp.diff(f, x)*N.i + sp.diff(f, y)*N.j
print(f"f = {f}")
print(f"∇f = {grad_f}")  # 2*x*N.i + 2*y*N.j

# === Matplotlib를 이용한 시각화 ===
X, Y = np.meshgrid(np.linspace(-3, 3, 30), np.linspace(-3, 3, 30))
T = X**2 + Y**2  # 온도 분포

# gradient 성분
dTdx = 2 * X
dTdy = 2 * Y

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 등고선 + gradient 벡터
ax = axes[0]
contour = ax.contourf(X, Y, T, levels=20, cmap='hot')
ax.quiver(X[::3, ::3], Y[::3, ::3], dTdx[::3, ::3], dTdy[::3, ::3],
          color='cyan', alpha=0.8)
plt.colorbar(contour, ax=ax, label='T(x,y)')
ax.set_title('온도 분포와 gradient 벡터')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')

# gradient 크기
ax = axes[1]
grad_mag = np.sqrt(dTdx**2 + dTdy**2)
im = ax.pcolormesh(X, Y, grad_mag, cmap='viridis', shading='auto')
plt.colorbar(im, ax=ax, label='|∇T|')
ax.set_title('gradient 크기 (변화율)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('gradient_visualization.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 1.2 Divergence (∇·F)

The **divergence** of a vector field $\mathbf{F} = F_x\hat{x} + F_y\hat{y} + F_z\hat{z}$ is a scalar quantity representing how much the field "spreads out" at each point.

$$
\nabla \cdot \mathbf{F} = \frac{\partial F_x}{\partial x} + \frac{\partial F_y}{\partial y} + \frac{\partial F_z}{\partial z}
$$

**Physical Meaning:**
- $\nabla \cdot \mathbf{F} > 0$: the point is a **source** — field spreads outward
- $\nabla \cdot \mathbf{F} < 0$: the point is a **sink** — field converges inward
- $\nabla \cdot \mathbf{F} = 0$: **solenoidal** — no creation or annihilation

```python
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

x, y = sp.symbols('x y')

# 원천이 있는 벡터장: F = (x, y) — 원점에서 퍼져나감
Fx_expr = x
Fy_expr = y
div_F = sp.diff(Fx_expr, x) + sp.diff(Fy_expr, y)
print(f"F = ({Fx_expr})x̂ + ({Fy_expr})ŷ")
print(f"∇·F = {div_F}")  # 2 (항상 양수 → 모든 점이 source)

# 비압축 벡터장: G = (-y, x) — 회전만 하는 장
Gx_expr = -y
Gy_expr = x
div_G = sp.diff(Gx_expr, x) + sp.diff(Gy_expr, y)
print(f"\nG = ({Gx_expr})x̂ + ({Gy_expr})ŷ")
print(f"∇·G = {div_G}")  # 0 (비압축)

# 시각화
X, Y = np.meshgrid(np.linspace(-2, 2, 15), np.linspace(-2, 2, 15))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# F = (x, y): divergence > 0 (source field)
ax = axes[0]
ax.quiver(X, Y, X, Y, color='red', alpha=0.7)
ax.set_title(f'F = (x, y),  ∇·F = {div_F} (원천장)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

# G = (-y, x): divergence = 0 (solenoidal)
ax = axes[1]
ax.quiver(X, Y, -Y, X, color='blue', alpha=0.7)
ax.set_title(f'G = (-y, x),  ∇·G = {div_G} (비압축장)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('divergence_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 1.3 Curl (∇×F)

The **curl** of a vector field $\mathbf{F}$ is a vector field representing the "rotational tendency" and axis of rotation at each point.

$$
\nabla \times \mathbf{F} = \begin{vmatrix} \hat{x} & \hat{y} & \hat{z} \\ \frac{\partial}{\partial x} & \frac{\partial}{\partial y} & \frac{\partial}{\partial z} \\ F_x & F_y & F_z \end{vmatrix}
$$

Expanded form:

$$
\nabla \times \mathbf{F} = \left(\frac{\partial F_z}{\partial y} - \frac{\partial F_y}{\partial z}\right)\hat{x} + \left(\frac{\partial F_x}{\partial z} - \frac{\partial F_z}{\partial x}\right)\hat{y} + \left(\frac{\partial F_y}{\partial x} - \frac{\partial F_x}{\partial y}\right)\hat{z}
$$

**Physical Meaning:**
- Direction: rotation axis according to the right-hand rule
- Magnitude: strength of rotation (circulation per unit area)
- $\nabla \times \mathbf{F} = \mathbf{0}$ indicates an **irrotational field** — necessary and sufficient condition for a conservative field (in simply connected domains)

```python
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

x, y, z = sp.symbols('x y z')

# 3D 벡터장: F = (-y, x, 0) — z축 둘레 회전
Fx, Fy, Fz = -y, x, sp.Integer(0)

# curl 계산
curl_x = sp.diff(Fz, y) - sp.diff(Fy, z)
curl_y = sp.diff(Fx, z) - sp.diff(Fz, x)
curl_z = sp.diff(Fy, x) - sp.diff(Fx, y)

print(f"F = ({Fx})x̂ + ({Fy})ŷ + ({Fz})ẑ")
print(f"∇×F = ({curl_x})x̂ + ({curl_y})ŷ + ({curl_z})ẑ")
# 결과: (0)x̂ + (0)ŷ + (2)ẑ → z 방향으로 균일한 회전

# 2D streamplot으로 시각화
X, Y = np.meshgrid(np.linspace(-3, 3, 30), np.linspace(-3, 3, 30))
U = -Y  # Fx = -y
V = X   # Fy = x
speed = np.sqrt(U**2 + V**2)

fig, ax = plt.subplots(figsize=(8, 8))
strm = ax.streamplot(X, Y, U, V, color=speed, cmap='coolwarm',
                      density=1.5, linewidth=1.5, arrowsize=1.5)
plt.colorbar(strm.lines, ax=ax, label='|F|')
ax.set_title('F = (-y, x): ∇×F = 2ẑ (균일한 회전장)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('curl_streamplot.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 1.4 Laplacian (∇²)

The **scalar Laplacian** is defined as the divergence of the gradient:

$$
\nabla^2 f = \nabla \cdot (\nabla f) = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2} + \frac{\partial^2 f}{\partial z^2}
$$

The **vector Laplacian** applies the scalar Laplacian to each component:

$$
\nabla^2 \mathbf{F} = (\nabla^2 F_x)\hat{x} + (\nabla^2 F_y)\hat{y} + (\nabla^2 F_z)\hat{z}
$$

**Physical Meaning:**
- Represents the difference between a point's value and the average of its neighborhood
- $\nabla^2 f > 0$: neighborhood average is greater than current value (local minimum tendency)
- $\nabla^2 f = 0$: **harmonic function** — solution to Laplace's equation

```python
import sympy as sp

x, y, z = sp.symbols('x y z')

# 조화함수인지 확인: f = 1/r (r = sqrt(x^2 + y^2 + z^2))
r = sp.sqrt(x**2 + y**2 + z**2)
f = 1 / r

laplacian_f = sp.diff(f, x, 2) + sp.diff(f, y, 2) + sp.diff(f, z, 2)
laplacian_f_simplified = sp.simplify(laplacian_f)
print(f"f = 1/r")
print(f"∇²f = {laplacian_f_simplified}")  # 0 (r ≠ 0에서 조화함수)

# 비조화함수 예시: g = x^2 + y^2
g = x**2 + y**2
laplacian_g = sp.diff(g, x, 2) + sp.diff(g, y, 2)
print(f"\ng = {g}")
print(f"∇²g = {laplacian_g}")  # 4 (비조화)
```

### 1.5 Vector Identities

Important identities frequently used in vector analysis:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Key Vector Identities                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. ∇×(∇f) = 0          curl of gradient is always 0          │
│     → conservative fields are always irrotational               │
│                                                                 │
│  2. ∇·(∇×F) = 0         divergence of curl is always 0        │
│     → magnetic field is always divergence-free (∇·B = 0)       │
│                                                                 │
│  3. ∇×(∇×F) = ∇(∇·F) - ∇²F                                   │
│     → curl of curl decomposition (used in EM wave equations)   │
│                                                                 │
│  4. ∇·(fF) = f(∇·F) + F·(∇f)         divergence of product   │
│  5. ∇×(fF) = f(∇×F) + (∇f)×F         curl of product         │
│  6. ∇(F·G) = (F·∇)G + (G·∇)F + F×(∇×G) + G×(∇×F)           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

```python
import sympy as sp
from sympy.vector import CoordSys3D, curl, divergence, gradient

N = CoordSys3D('N')
x, y, z = N.x, N.y, N.z

# 항등식 1 검증: curl(grad(f)) = 0
f = x**2 * y + y**2 * z + z**2 * x
grad_f = gradient(f, N)
curl_grad_f = curl(grad_f, N)
print(f"f = {f}")
print(f"∇f = {grad_f}")
print(f"∇×(∇f) = {curl_grad_f}")  # 0

# 항등식 2 검증: div(curl(F)) = 0
F = (x*y*z)*N.i + (x**2 - z)*N.j + (y*z**2)*N.k
curl_F = curl(F, N)
div_curl_F = divergence(curl_F, N)
print(f"\nF = {F}")
print(f"∇×F = {curl_F}")
print(f"∇·(∇×F) = {sp.simplify(div_curl_F)}")  # 0
```

---

## 2. Line Integrals

Line integrals integrate scalar or vector fields along curves, computing physical quantities such as work, circulation, and path length.

### 2.1 Line Integral of a Scalar Field

Integrating a scalar field $f$ along curve $C: \mathbf{r}(t) = (x(t), y(t), z(t))$, $a \leq t \leq b$:

$$
\int_C f \, ds = \int_a^b f(\mathbf{r}(t)) \left|\frac{d\mathbf{r}}{dt}\right| dt
$$

where $ds = |\mathbf{r}'(t)| \, dt$ is the arc length element.

**Applications:** mass of a wire with variable density, arc length

```python
import numpy as np
import sympy as sp

t = sp.Symbol('t')

# 예제: 나선 경로 r(t) = (cos t, sin t, t), 0 <= t <= 2pi 위에서
# f = x^2 + y^2 + z^2 의 선적분
x_t = sp.cos(t)
y_t = sp.sin(t)
z_t = t

f = x_t**2 + y_t**2 + z_t**2  # cos²t + sin²t + t² = 1 + t²

# dr/dt 계산
dx = sp.diff(x_t, t)
dy = sp.diff(y_t, t)
dz = sp.diff(z_t, t)
ds_dt = sp.sqrt(dx**2 + dy**2 + dz**2)
ds_dt_simplified = sp.simplify(ds_dt)
print(f"|dr/dt| = {ds_dt_simplified}")  # sqrt(2)

# 선적분 계산
integrand = f * ds_dt_simplified
result = sp.integrate(integrand, (t, 0, 2*sp.pi))
print(f"∫_C f ds = {sp.simplify(result)}")
print(f"수치값 = {float(result):.4f}")
```

### 2.2 Line Integral of a Vector Field (Work)

Integrating vector field $\mathbf{F}$ along curve $C$ gives **work**:

$$
W = \int_C \mathbf{F} \cdot d\mathbf{r} = \int_a^b \mathbf{F}(\mathbf{r}(t)) \cdot \mathbf{r}'(t) \, dt
$$

Component form:

$$
W = \int_C F_x \, dx + F_y \, dy + F_z \, dz
$$

```python
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

t = sp.Symbol('t')

# 벡터장 F = (y, -x) 에서 원형 경로를 따른 일(work) 계산
# 경로: r(t) = (cos t, sin t), 0 <= t <= 2pi

# 경로 매개변수화
x_t = sp.cos(t)
y_t = sp.sin(t)

# 벡터장 성분 (경로 위)
Fx = y_t    # F_x = y = sin t
Fy = -x_t   # F_y = -x = -cos t

# dr/dt
dx_dt = sp.diff(x_t, t)  # -sin t
dy_dt = sp.diff(y_t, t)  #  cos t

# F · dr/dt
integrand = Fx * dx_dt + Fy * dy_dt
integrand_simplified = sp.simplify(integrand)
print(f"F·dr/dt = {integrand_simplified}")  # -1

# 일(work) 계산
W = sp.integrate(integrand, (t, 0, 2*sp.pi))
print(f"W = ∮ F·dr = {W}")  # -2*pi (음수: 장이 경로와 반대 방향)

# 시각화: 벡터장과 경로
theta = np.linspace(0, 2*np.pi, 100)
X, Y = np.meshgrid(np.linspace(-1.5, 1.5, 12), np.linspace(-1.5, 1.5, 12))

fig, ax = plt.subplots(figsize=(8, 8))
ax.quiver(X, Y, Y, -X, color='steelblue', alpha=0.6, label='F = (y, -x)')
ax.plot(np.cos(theta), np.sin(theta), 'r-', linewidth=2, label='경로 C')
ax.annotate('', xy=(0.7, 0.7), xytext=(0.71, 0.69),
            arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax.set_title(f'∮ F·dr = {W} (시계 방향 순환)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('line_integral_work.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 2.3 Conservative Fields and Potential Functions

If vector field $\mathbf{F}$ is a **conservative field**, the line integral value is path-independent and depends only on endpoints.

$$
\mathbf{F} = \nabla \phi \quad \Longleftrightarrow \quad \int_A^B \mathbf{F} \cdot d\mathbf{r} = \phi(B) - \phi(A)
$$

**Equivalent conditions for conservative fields (in simply connected domains):**

```
┌─────────────────────────────────────────────────────────────────┐
│  The following conditions are equivalent (in simply connected): │
├─────────────────────────────────────────────────────────────────┤
│  (1) There exists a potential function φ such that F = ∇φ      │
│  (2) ∮_C F·dr = 0  (for any closed path)                       │
│  (3) ∫_A^B F·dr is path-independent                            │
│  (4) ∇×F = 0                                                   │
│  (5) F_x dx + F_y dy + F_z dz is an exact differential        │
└─────────────────────────────────────────────────────────────────┘
```

```python
import sympy as sp
from sympy.vector import CoordSys3D, curl

N = CoordSys3D('N')
x, y, z = N.x, N.y, N.z

# === 보존장 판별 예제 ===
# F1 = (2xy + z)x̂ + (x² + 2yz)ŷ + (x + y²)ẑ
F1 = (2*x*y + z)*N.i + (x**2 + 2*y*z)*N.j + (x + y**2)*N.k
curl_F1 = curl(F1, N)
print(f"F1 = {F1}")
print(f"∇×F1 = {curl_F1}")  # 0 → 보존장!

# 퍼텐셜 함수 구하기: ∂φ/∂x = 2xy + z
phi_x = sp.Symbol('phi')
phi = sp.integrate(2*x*y + z, x)  # x²y + xz + g(y,z)
print(f"\n∫ (2xy+z)dx = {phi} + g(y,z)")

# g(y,z) 결정: ∂φ/∂y = x² + ∂g/∂y = x² + 2yz → ∂g/∂y = 2yz
g = sp.integrate(2*y*z, y)  # y²z + h(z)
print(f"∫ 2yz dy = {g} + h(z)")

# h(z) 결정: ∂φ/∂z = x + y² + h'(z) = x + y² → h'(z) = 0 → h = C
phi_total = x**2 * y + x*z + y**2 * z
print(f"\nφ(x,y,z) = {phi_total}")

# 검증: ∇φ = F1?
from sympy.vector import gradient
grad_phi = gradient(phi_total, N)
print(f"∇φ = {grad_phi}")
print(f"F1 = ∇φ? {sp.simplify(grad_phi - F1) == N.zero}")

# === 비보존장 예제 ===
# F2 = (y)x̂ + (x + z)ŷ + (y + 1)ẑ — curl ≠ 0 확인
F2 = y*N.i + (x + z)*N.j + (y + 1)*N.k
curl_F2 = curl(F2, N)
print(f"\nF2 = {F2}")
print(f"∇×F2 = {curl_F2}")  # 비보존장 여부 확인
```

---

## 3. Surface Integrals

### 3.1 Surface Element and Normal Vector

When surface $S$ is parameterized by $(u, v)$: $\mathbf{r}(u, v) = (x(u,v),\, y(u,v),\, z(u,v))$

**Surface element:**

$$
d\mathbf{S} = \left(\frac{\partial \mathbf{r}}{\partial u} \times \frac{\partial \mathbf{r}}{\partial v}\right) du \, dv = \hat{n} \, dA
$$

where $\hat{n}$ is the unit normal vector and $dA = |d\mathbf{S}|$ is the surface area element.

**For surfaces given as $z = g(x, y)$:**

$$
d\mathbf{S} = \left(-\frac{\partial g}{\partial x}\hat{x} - \frac{\partial g}{\partial y}\hat{y} + \hat{z}\right) dx \, dy
$$

### 3.2 Surface Integral of a Scalar Field

$$
\iint_S f \, dA = \iint_D f(\mathbf{r}(u,v)) \left|\frac{\partial \mathbf{r}}{\partial u} \times \frac{\partial \mathbf{r}}{\partial v}\right| du \, dv
$$

**Applications:** surface area (when $f = 1$), total of physical quantity over surface

```python
import numpy as np
import sympy as sp
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# 구면 r = 1 의 면적 계산 (구면좌표)
theta, phi = sp.symbols('theta phi')

# 구면 매개변수화: r(θ, φ) = (sinθ cosφ, sinθ sinφ, cosθ)
r_theta = sp.Matrix([sp.cos(phi)*sp.cos(theta),  # ∂r/∂θ
                      sp.sin(phi)*sp.cos(theta),
                      -sp.sin(theta)])
r_phi = sp.Matrix([-sp.sin(phi)*sp.sin(theta),    # ∂r/∂φ
                    sp.cos(phi)*sp.sin(theta),
                    0])

# 외적: ∂r/∂θ × ∂r/∂φ
cross = r_theta.cross(r_phi)
dA = sp.simplify(cross.norm())
print(f"|∂r/∂θ × ∂r/∂φ| = {dA}")  # sin(theta) (θ ∈ [0, π]에서 양수)

# 면적 적분
area = sp.integrate(sp.sin(theta), (phi, 0, 2*sp.pi), (theta, 0, sp.pi))
print(f"구의 면적 = {area}")  # 4*pi

# 3D 시각화
u = np.linspace(0, np.pi, 40)
v = np.linspace(0, 2*np.pi, 40)
U, V = np.meshgrid(u, v)

X = np.sin(U) * np.cos(V)
Y = np.sin(U) * np.sin(V)
Z = np.cos(U)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, alpha=0.6, cmap='viridis')

# 법선 벡터 표시 (일부 점에서)
step = 8
for i in range(0, len(u), step):
    for j in range(0, len(v), step):
        px, py, pz = X[j, i], Y[j, i], Z[j, i]
        ax.quiver(px, py, pz, px*0.3, py*0.3, pz*0.3,
                  color='red', arrow_length_ratio=0.3)

ax.set_title('단위 구면과 법선 벡터')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.tight_layout()
plt.savefig('sphere_normal_vectors.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 3.3 Surface Integral of a Vector Field (Flux)

The **flux** of vector field $\mathbf{F}$ through surface $S$:

$$
\Phi = \iint_S \mathbf{F} \cdot d\mathbf{S} = \iint_S \mathbf{F} \cdot \hat{n} \, dA
$$

**Physical Meaning:**
- Electric flux: amount of electric field penetrating the surface (Gauss's law)
- Mass flux: mass flow rate through the surface

```python
import sympy as sp

x, y, z = sp.symbols('x y z')
u, v = sp.symbols('u v')

# 예제: F = (x, y, z)가 구면 r = R을 통과하는 플럭스
R = sp.Symbol('R', positive=True)
theta, phi = sp.symbols('theta phi')

# 구면 위에서 r̂ = (sinθ cosφ, sinθ sinφ, cosθ)
# 구면 위에서 F = R*(sinθ cosφ, sinθ sinφ, cosθ) = R*r̂
# F·n̂ = F·r̂ = R (구면에서 법선은 r̂ 방향)

# dA = R² sinθ dθ dφ
integrand = R * R**2 * sp.sin(theta)
flux = sp.integrate(integrand, (phi, 0, 2*sp.pi), (theta, 0, sp.pi))
print(f"F = (x, y, z) 가 구면 r={R}을 관통하는 플럭스:")
print(f"Φ = ∬ F·dS = {flux}")  # 4*pi*R^3

# 발산 정리로 검증: ∬ F·dS = ∭ (∇·F) dV
div_F = 3  # ∇·(x,y,z) = 1 + 1 + 1 = 3
volume = sp.Rational(4, 3) * sp.pi * R**3
flux_divergence = div_F * volume
print(f"\n발산 정리 검증: ∭ (∇·F)dV = 3 × (4/3)πR³ = {flux_divergence}")
print(f"일치 여부: {sp.simplify(flux - flux_divergence) == 0}")  # True
```

---

## 4. Integral Theorems

The three fundamental integral theorems of vector analysis connect differential operations (gradient, curl, divergence) with integrals (line, surface, volume).

```
┌─────────────────────────────────────────────────────────────────┐
│     Three Fundamental Theorems — by Dimension                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Dim    Theorem       Operator     Integrals                   │
│  ───────────────────────────────────────────────────            │
│  2D     Green         ∂/∂x, ∂/∂y   Area ↔ Line                 │
│  3D-S   Stokes        ∇×            Surface ↔ Line             │
│  3D-V   Gauss         ∇·            Volume ↔ Surface           │
│                                                                 │
│  Common pattern: ∫∫(differential) = ∮(boundary integral)       │
│                  "interior differential = boundary integral"   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.1 Green's Theorem

In 2D, for a simple closed curve $C$ enclosing region $D$:

$$
\oint_C (P \, dx + Q \, dy) = \iint_D \left(\frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y}\right) dA
$$

**Physical Meaning:** The **circulation** of a vector field equals the sum of the $z$-component curl over the interior.

```python
import numpy as np
import sympy as sp

x, y, t = sp.symbols('x y t')

# 예제: P = -y², Q = x² 에 대해 그린 정리 검증
# 영역: 단위 원 x² + y² ≤ 1
P = -y**2
Q = x**2

# 좌변: 선적분 (단위 원 경로)
x_t = sp.cos(t)
y_t = sp.sin(t)
dx_dt = sp.diff(x_t, t)
dy_dt = sp.diff(y_t, t)

P_on_C = P.subs([(x, x_t), (y, y_t)])
Q_on_C = Q.subs([(x, x_t), (y, y_t)])

line_integral = sp.integrate(P_on_C * dx_dt + Q_on_C * dy_dt, (t, 0, 2*sp.pi))
print(f"선적분 ∮(P dx + Q dy) = {line_integral}")

# 우변: 면적분 (극좌표)
r, theta = sp.symbols('r theta')
dQ_dx = sp.diff(Q, x)  # 2x
dP_dy = sp.diff(P, y)  # -2y
integrand = dQ_dx - dP_dy  # 2x + 2y

# 극좌표 변환
integrand_polar = integrand.subs([(x, r*sp.cos(theta)), (y, r*sp.sin(theta))])
area_integral = sp.integrate(integrand_polar * r, (r, 0, 1), (theta, 0, 2*sp.pi))
print(f"면적분 ∬(∂Q/∂x - ∂P/∂y)dA = {area_integral}")

print(f"\n그린 정리 성립: {sp.simplify(line_integral - area_integral) == 0}")
```

**Special form of Green's theorem — area formula:**

$$
A = \frac{1}{2} \oint_C (x \, dy - y \, dx)
$$

This formula is used in surveying and computer graphics to calculate polygon areas.

### 4.2 Stokes' Theorem

In 3D, for surface $S$ with boundary curve $C = \partial S$:

$$
\oint_C \mathbf{F} \cdot d\mathbf{r} = \iint_S (\nabla \times \mathbf{F}) \cdot d\mathbf{S}
$$

**Interpretation:** Circulation around boundary = flux of curl through surface

```python
import numpy as np
import sympy as sp

x, y, z, t = sp.symbols('x y z t')

# 예제: F = (y, -x, z²) 에 대해 스토크스 정리 검증
# 곡면 S: z = 1 - x² - y² (z ≥ 0인 포물면)
# 경계 C: z = 0에서 x² + y² = 1 (단위 원)

# --- curl(F) 계산 ---
Fx, Fy, Fz = y, -x, z**2

curl_x = sp.diff(Fz, y) - sp.diff(Fy, z)  # 0 - 0 = 0
curl_y = sp.diff(Fx, z) - sp.diff(Fz, x)  # 0 - 0 = 0
curl_z = sp.diff(Fy, x) - sp.diff(Fx, y)  # -1 - 1 = -2
print(f"∇×F = ({curl_x}, {curl_y}, {curl_z})")

# --- 좌변: 선적분 ∮_C F·dr ---
# C: r(t) = (cos t, sin t, 0), 0 ≤ t ≤ 2π (반시계)
x_t, y_t, z_t = sp.cos(t), sp.sin(t), sp.Integer(0)
dx_dt = sp.diff(x_t, t)
dy_dt = sp.diff(y_t, t)
dz_dt = sp.diff(z_t, t)

Fx_C = Fy_expr = y_t   # Fx = y = sin t
Fy_C = -x_t             # Fy = -x = -cos t
Fz_C = z_t**2           # Fz = z² = 0

line_int = sp.integrate(
    Fx_C * dx_dt + Fy_C * dy_dt + Fz_C * dz_dt,
    (t, 0, 2*sp.pi)
)
print(f"\n좌변 (선적분): ∮ F·dr = {line_int}")

# --- 우변: 면적분 ∬_S (∇×F)·dS ---
# 곡면 z = 1 - x² - y², dS = (-∂z/∂x, -∂z/∂y, 1) dx dy = (2x, 2y, 1) dx dy
# (∇×F)·dS = (0, 0, -2)·(2x, 2y, 1) dx dy = -2 dx dy

r_sym, theta_sym = sp.symbols('r_s theta_s')
surface_int = sp.integrate(
    -2 * r_sym,  # -2 × r (야코비안)
    (r_sym, 0, 1),
    (theta_sym, 0, 2*sp.pi)
)
print(f"우변 (면적분): ∬ (∇×F)·dS = {surface_int}")
print(f"스토크스 정리 성립: {line_int == surface_int}")
```

### 4.3 Divergence Theorem (Gauss's Theorem)

For closed surface $S$ enclosing volume $V$:

$$
\oiint_S \mathbf{F} \cdot d\mathbf{S} = \iiint_V (\nabla \cdot \mathbf{F}) \, dV
$$

**Interpretation:** Total flux through closed surface = sum of divergence throughout interior

```python
import numpy as np
import sympy as sp

x, y, z = sp.symbols('x y z')

# 예제: F = (x³, y³, z³), 닫힌 곡면 = 단위 구 x²+y²+z² = 1

# ∇·F = 3x² + 3y² + 3z² = 3r²
div_F = sp.diff(x**3, x) + sp.diff(y**3, y) + sp.diff(z**3, z)
print(f"∇·F = {div_F}")  # 3x² + 3y² + 3z²

# 체적 적분 (구면좌표)
r, theta, phi = sp.symbols('r theta phi')
div_F_spherical = 3 * r**2  # 3(x² + y² + z²) = 3r²
jacobian = r**2 * sp.sin(theta)

volume_int = sp.integrate(
    div_F_spherical * jacobian,
    (r, 0, 1),
    (theta, 0, sp.pi),
    (phi, 0, 2*sp.pi)
)
print(f"∭ (∇·F) dV = {volume_int}")  # 12π/5

# 직접 면적분으로 검증
# 구면 위에서 r̂ = (x, y, z) (단위구이므로 |r| = 1)
# F·r̂ = x⁴ + y⁴ + z⁴ (구면 위에서 x² + y² + z² = 1)
# 구면좌표: x = sinθ cosφ, y = sinθ sinφ, z = cosθ

F_dot_n = (sp.sin(theta)*sp.cos(phi))**4 + \
          (sp.sin(theta)*sp.sin(phi))**4 + \
          sp.cos(theta)**4

surface_int = sp.integrate(
    F_dot_n * sp.sin(theta),  # dA = sinθ dθ dφ
    (theta, 0, sp.pi),
    (phi, 0, 2*sp.pi)
)
surface_int_simplified = sp.simplify(surface_int)
print(f"∬ F·dS = {surface_int_simplified}")
print(f"일치: {sp.simplify(volume_int - surface_int_simplified) == 0}")
```

### 4.4 Relationship Between the Three Theorems

All three integral theorems are special cases of the **generalized Stokes' theorem**:

$$
\int_{\partial \Omega} \omega = \int_{\Omega} d\omega
$$

```
┌─────────────────────────────────────────────────────────────────┐
│          Unified View of Three Theorems                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Fundamental Theorem of Calculus (1D):                         │
│    ∫_a^b f'(x) dx = f(b) - f(a)                                │
│    "integral of derivative = difference of boundary values"    │
│                                                                 │
│  Green's Theorem (2D):                                         │
│    ∬_D (∂Q/∂x - ∂P/∂y) dA = ∮_{∂D} (P dx + Q dy)             │
│    "area integral of curl = line integral on boundary"         │
│                                                                 │
│  Stokes' Theorem (3D, surface↔boundary):                       │
│    ∬_S (∇×F)·dS = ∮_{∂S} F·dr                                 │
│    "surface integral of curl = line integral on boundary"      │
│                                                                 │
│  Gauss's Theorem (3D, volume↔boundary):                        │
│    ∭_V (∇·F) dV = ∬_{∂V} F·dS                                 │
│    "volume integral of divergence = surface integral"          │
│                                                                 │
│  Pattern: "interior differential = boundary values"            │
│           n-dimensional integral ↔ (n-1)-dimensional boundary  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Physical Applications

### 5.1 Electric Field and Gauss's Law

**Gauss's Law (integral form):**

$$
\oiint_S \mathbf{E} \cdot d\mathbf{S} = \frac{Q_{\text{enc}}}{\epsilon_0}
$$

**Gauss's Law (differential form):** Applying the divergence theorem:

$$
\nabla \cdot \mathbf{E} = \frac{\rho}{\epsilon_0}
$$

The divergence of the electric field is nonzero where charge density $\rho$ exists.

```python
import numpy as np
import matplotlib.pyplot as plt

# 점전하의 전기장 시각화 (2D 단면)
# E = q/(4πε₀) × r̂/r² (쿨롱 법칙)

q = 1.0  # 전하량 (임의 단위)
eps0 = 1.0  # ε₀ (단위계 편의상)

X, Y = np.meshgrid(np.linspace(-3, 3, 20), np.linspace(-3, 3, 20))
R = np.sqrt(X**2 + Y**2)
R = np.where(R < 0.3, 0.3, R)  # 특이점 방지

# 전기장 성분
k = q / (4 * np.pi * eps0)
Ex = k * X / R**3
Ey = k * Y / R**3
E_mag = np.sqrt(Ex**2 + Ey**2)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 양전하 (+q)
ax = axes[0]
ax.streamplot(X, Y, Ex, Ey, color=np.log(E_mag + 1), cmap='Reds',
              density=2, linewidth=1.2)
ax.plot(0, 0, 'ro', markersize=15, label='+q')
circle1 = plt.Circle((0, 0), 1.0, fill=False, color='gray', linestyle='--', label='가우스 면 r=1')
circle2 = plt.Circle((0, 0), 2.0, fill=False, color='gray', linestyle=':', label='가우스 면 r=2')
ax.add_patch(circle1)
ax.add_patch(circle2)
ax.set_title('양전하의 전기장 (발산 > 0)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')
ax.legend(loc='upper left', fontsize=9)
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)

# 쌍극자 (dipole): +q at (1,0), -q at (-1,0)
ax = axes[1]
d = 1.0
R1 = np.sqrt((X - d)**2 + Y**2)
R2 = np.sqrt((X + d)**2 + Y**2)
R1 = np.where(R1 < 0.3, 0.3, R1)
R2 = np.where(R2 < 0.3, 0.3, R2)

Ex_dip = k * (X - d) / R1**3 - k * (X + d) / R2**3
Ey_dip = k * Y / R1**3 - k * Y / R2**3
E_dip_mag = np.sqrt(Ex_dip**2 + Ey_dip**2)

ax.streamplot(X, Y, Ex_dip, Ey_dip, color=np.log(E_dip_mag + 1),
              cmap='coolwarm', density=2, linewidth=1.2)
ax.plot(d, 0, 'ro', markersize=12, label='+q')
ax.plot(-d, 0, 'bo', markersize=12, label='-q')
ax.set_title('전기 쌍극자 (dipole)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')
ax.legend(fontsize=9)
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)

plt.tight_layout()
plt.savefig('electric_field_gauss.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 5.2 Magnetic Field and Ampère's Law

**Ampère's Law (integral form):**

$$
\oint_C \mathbf{B} \cdot d\mathbf{r} = \mu_0 I_{\text{enc}}
$$

**Ampère's Law (differential form):** Applying Stokes' theorem:

$$
\nabla \times \mathbf{B} = \mu_0 \mathbf{J}
$$

where $\mathbf{J}$ is the current density.

**Physical Meaning:**
- Curl of magnetic field is proportional to current density
- Magnetic field lines form closed loops around currents (right-hand rule)
- $\nabla \cdot \mathbf{B} = 0$ — magnetic monopoles do not exist

```python
import numpy as np
import matplotlib.pyplot as plt

# 무한 직선 전류에 의한 자기장
# B = μ₀I/(2πr) × φ̂ (원통좌표)

mu0 = 1.0  # μ₀ (임의 단위)
I = 1.0    # 전류 (z 방향)

X, Y = np.meshgrid(np.linspace(-3, 3, 20), np.linspace(-3, 3, 20))
R = np.sqrt(X**2 + Y**2)
R = np.where(R < 0.3, 0.3, R)

# B = μ₀I/(2πr) × φ̂, 여기서 φ̂ = (-y/r, x/r, 0)
B_coeff = mu0 * I / (2 * np.pi * R)
Bx = B_coeff * (-Y / R)
By = B_coeff * (X / R)

fig, ax = plt.subplots(figsize=(8, 8))
B_mag = np.sqrt(Bx**2 + By**2)
strm = ax.streamplot(X, Y, Bx, By, color=np.log(B_mag + 0.01),
                      cmap='plasma', density=2, linewidth=1.5)
plt.colorbar(strm.lines, ax=ax, label='log|B|')

# 전류 위치 (원점, z 방향)
ax.plot(0, 0, 'g^', markersize=15, label='I (z 방향, 지면에서 나옴)')

# 앙페르 루프 표시
for r in [1.0, 2.0]:
    circle = plt.Circle((0, 0), r, fill=False, color='lime',
                         linestyle='--', linewidth=2)
    ax.add_patch(circle)

ax.set_title('직선 전류 주위의 자기장 (앙페르 법칙)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')
ax.legend(loc='upper left')
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
plt.tight_layout()
plt.savefig('magnetic_field_ampere.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 5.3 Continuity Equation in Fluid Dynamics

The **continuity equation** expressing mass conservation:

$$
\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{v}) = 0
$$

where $\rho$ is fluid density and $\mathbf{v}$ is the velocity field.

**Derivation:**
1. Mass outflow rate through closed surface = $\oiint_S \rho \mathbf{v} \cdot d\mathbf{S}$
2. Rate of mass change in volume = $-\frac{\partial}{\partial t}\iiint_V \rho \, dV$
3. Apply divergence theorem: $\iiint_V \left[\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{v})\right] dV = 0$
4. For arbitrary volume, integrand must be zero

**Incompressible fluid** ($\rho$ = constant):

$$
\nabla \cdot \mathbf{v} = 0
$$

```python
import numpy as np
import matplotlib.pyplot as plt

# 2D 비압축 유체 흐름 예시
# 흐름 함수(stream function) ψ를 이용: vx = ∂ψ/∂y, vy = -∂ψ/∂x
# → 자동으로 ∇·v = 0 만족

X, Y = np.meshgrid(np.linspace(-3, 3, 25), np.linspace(-3, 3, 25))

# 예제 1: 균일 흐름 + 원통 주위 흐름 (퍼텐셜 흐름)
# ψ = U*y*(1 - a²/r²), U = 자유류 속도, a = 원통 반지름
U_inf = 1.0
a = 1.0
R_sq = X**2 + Y**2
R_sq = np.where(R_sq < a**2, a**2, R_sq)  # 원통 내부 마스킹

Vx = U_inf * (1 - a**2 * (X**2 - Y**2) / R_sq**2)
Vy = -U_inf * 2 * a**2 * X * Y / R_sq**2

# 원통 내부 속도 = 0
mask = (X**2 + Y**2) < a**2
Vx[mask] = 0
Vy[mask] = 0

fig, ax = plt.subplots(figsize=(10, 8))
speed = np.sqrt(Vx**2 + Vy**2)
strm = ax.streamplot(X, Y, Vx, Vy, color=speed, cmap='RdYlBu_r',
                      density=2, linewidth=1.2)
plt.colorbar(strm.lines, ax=ax, label='|v| (속력)')

# 원통 표시
circle = plt.Circle((0, 0), a, color='gray', alpha=0.5)
ax.add_patch(circle)

# 발산 계산 (수치적)
dVx_dx = np.gradient(Vx, X[0], axis=1)
dVy_dy = np.gradient(Vy, Y[:, 0], axis=0)
div_v = dVx_dx + dVy_dy
max_div = np.max(np.abs(div_v[~mask]))
ax.set_title(f'원통 주위 비압축 유체 흐름  (max|∇·v| ≈ {max_div:.2e})')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('fluid_flow_cylinder.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 5.4 Maxwell's Equations: Integral and Differential Forms

Maxwell's equations completely describe electromagnetic phenomena. Through vector analysis integral theorems (Gauss, Stokes), we can convert between integral and differential forms.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    Maxwell's Equations                                   │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Law               Differential Form        Integral Form      Theorem  │
│  ─────────────────────────────────────────────────────────────────────── │
│                                                                          │
│  Gauss (electric)  ∇·E = ρ/ε₀              ∮ E·dS = Q/ε₀       Gauss   │
│                                                                          │
│  Gauss (magnetic)  ∇·B = 0                 ∮ B·dS = 0           Gauss   │
│                                                                          │
│  Faraday           ∇×E = -∂B/∂t            ∮ E·dr = -dΦ_B/dt   Stokes  │
│                                                                          │
│  Ampère-Maxwell    ∇×B = μ₀J + μ₀ε₀∂E/∂t                      Stokes  │
│                                     ∮ B·dr = μ₀I + μ₀ε₀ dΦ_E/dt        │
│                                                                          │
│  ─────────────────────────────────────────────────────────────────────── │
│                                                                          │
│  Physical Meaning:                                                       │
│  • Gauss (electric): Charges are sources of electric field              │
│  • Gauss (magnetic): Magnetic monopoles do not exist                    │
│  • Faraday: Changing magnetic field induces electric field              │
│  • Ampère-Maxwell: Current and changing E-field induce B-field          │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

**Converting from integral to differential form (Gauss's law example):**

$$
\oiint_S \mathbf{E} \cdot d\mathbf{S} = \frac{Q_{\text{enc}}}{\epsilon_0}
= \frac{1}{\epsilon_0}\iiint_V \rho \, dV
$$

Apply divergence theorem to left side:

$$
\iiint_V (\nabla \cdot \mathbf{E}) \, dV = \frac{1}{\epsilon_0}\iiint_V \rho \, dV
$$

Valid for arbitrary volume $V$:

$$
\nabla \cdot \mathbf{E} = \frac{\rho}{\epsilon_0}
$$

```python
import numpy as np
import matplotlib.pyplot as plt

# 맥스웰 방정식의 시각적 정리: 전기장과 자기장의 관계
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# === (1) 가우스 법칙 (전기): ∇·E = ρ/ε₀ ===
ax = axes[0, 0]
X, Y = np.meshgrid(np.linspace(-2, 2, 15), np.linspace(-2, 2, 15))
R = np.sqrt(X**2 + Y**2)
R = np.where(R < 0.3, 0.3, R)
Ex = X / R**3
Ey = Y / R**3
ax.quiver(X, Y, Ex, Ey, color='red', alpha=0.6)
circle = plt.Circle((0, 0), 0.2, color='red', alpha=0.8)
ax.add_patch(circle)
ax.set_title('(1) 가우스 법칙: ∇·E = ρ/ε₀\n전하 → 발산하는 E')
ax.set_aspect('equal')
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.grid(True, alpha=0.2)

# === (2) 가우스 법칙 (자기): ∇·B = 0 ===
ax = axes[0, 1]
# 자기 쌍극자 (단극자 없음)
d = 0.5
R1 = np.sqrt((X - 0)**2 + (Y - d)**2)
R2 = np.sqrt((X - 0)**2 + (Y + d)**2)
R1 = np.where(R1 < 0.3, 0.3, R1)
R2 = np.where(R2 < 0.3, 0.3, R2)
Bx = X / R1**3 - X / R2**3
By = (Y - d) / R1**3 - (Y + d) / R2**3
speed = np.sqrt(Bx**2 + By**2)
ax.streamplot(X, Y, Bx, By, color=np.log(speed + 0.1), cmap='Blues',
              density=2, linewidth=1)
ax.set_title('(2) 가우스 법칙: ∇·B = 0\n자기장선은 닫힌 루프')
ax.set_aspect('equal')
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.grid(True, alpha=0.2)

# === (3) 패러데이 법칙: ∇×E = -∂B/∂t ===
ax = axes[1, 0]
# 변하는 자기장 → 유도 전기장 (원형)
R_circ = np.sqrt(X**2 + Y**2)
R_circ = np.where(R_circ < 0.2, 0.2, R_circ)
Ex_ind = -Y / R_circ**2
Ey_ind = X / R_circ**2
ax.streamplot(X, Y, Ex_ind, Ey_ind, color='orange', density=1.5, linewidth=1.5)
ax.annotate('dB/dt\n(z 방향)', xy=(0, 0), fontsize=12, ha='center',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
ax.set_title('(3) 패러데이: ∇×E = -∂B/∂t\n변하는 B → 유도 E')
ax.set_aspect('equal')
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.grid(True, alpha=0.2)

# === (4) 앙페르-맥스웰: ∇×B = μ₀J + μ₀ε₀ ∂E/∂t ===
ax = axes[1, 1]
R_wire = np.sqrt(X**2 + Y**2)
R_wire = np.where(R_wire < 0.2, 0.2, R_wire)
Bx_wire = -Y / R_wire**2
By_wire = X / R_wire**2
ax.streamplot(X, Y, Bx_wire, By_wire, color='purple', density=1.5, linewidth=1.5)
ax.annotate('I or ∂E/∂t\n(z 방향)', xy=(0, 0), fontsize=12, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
ax.set_title('(4) 앙페르-맥스웰: ∇×B = μ₀J + μ₀ε₀∂E/∂t\n전류/변하는 E → B')
ax.set_aspect('equal')
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.grid(True, alpha=0.2)

plt.suptitle('맥스웰 방정식의 4가지 법칙', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('maxwell_equations.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## Practice Problems

### Problem 1: Gradient and Directional Derivative

For scalar field $f(x, y, z) = x^2 y + y^2 z + z^2 x$:
1. Find $\nabla f$.
2. Calculate the directional derivative at point $(1, 1, 1)$ in direction $\hat{u} = \frac{1}{\sqrt{3}}(1, 1, 1)$.
3. At point $(1, 1, 1)$, find the direction of fastest increase of $f$ and its rate.

### Problem 2: Divergence and Curl

Calculate the divergence and curl of the following vector fields, and determine if they are conservative:

(a) $\mathbf{F} = (yz, xz, xy)$

(b) $\mathbf{G} = (x^2 - y, y^2 + x, z)$

If conservative, find the potential function.

### Problem 3: Line Integrals

For vector field $\mathbf{F} = (2xy + z^2)\hat{x} + x^2\hat{y} + 2xz\hat{z}$:
1. Show that $\mathbf{F}$ is conservative and find potential function $\phi$.
2. Calculate the line integral from $(0, 0, 0)$ to $(1, 2, 3)$ using (a) the potential function, (b) direct integration along straight path $\mathbf{r}(t) = (t, 2t, 3t)$, and verify they match.

### Problem 4: Divergence Theorem Verification

For $\mathbf{F} = (x^2, y^2, z^2)$ in the unit cube $[0,1]^3$, verify the divergence theorem:
1. Calculate $\nabla \cdot \mathbf{F}$ and compute the volume integral.
2. Calculate surface integrals on all 6 faces and sum them.
3. Verify that the two results match.

### Problem 5: Stokes' Theorem and Physical Application

For uniform current density $\mathbf{J} = J_0 \hat{z}$ in a cylindrical wire of radius $a$:
1. Use Ampère's law (integral form) to find magnetic field $\mathbf{B}$ for $r < a$ and $r > a$.
2. For $r < a$, directly verify that $\nabla \times \mathbf{B} = \mu_0 \mathbf{J}$ (use cylindrical coordinate curl formula).

### Problem 6: Maxwell Equation Transformation

From Faraday's law integral form:

$$
\oint_C \mathbf{E} \cdot d\mathbf{r} = -\frac{d}{dt}\iint_S \mathbf{B} \cdot d\mathbf{S}
$$

Use Stokes' theorem to derive the differential form $\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}$. Describe the derivation step by step.

---

## References

### Textbooks
1. **Boas, M. L.** (2005). *Mathematical Methods in the Physical Sciences*, 3rd ed., Chapter 6. Wiley.
2. **Griffiths, D. J.** (2017). *Introduction to Electrodynamics*, 4th ed. Cambridge University Press.
   - Excellent reference for electromagnetic applications of vector analysis
3. **Arfken, G. B., Weber, H. J., & Harris, F. E.** (2012). *Mathematical Methods for Physicists*, 7th ed., Chapters 1-3. Academic Press.
4. **Schey, H. M.** (2005). *Div, Grad, Curl, and All That*, 4th ed. W.W. Norton.
   - Intuitive introduction to vector analysis

### Online Resources
1. **3Blue1Brown** — [Divergence and Curl](https://www.youtube.com/watch?v=rB83DpBJQsE): Visual understanding of divergence and curl
2. **MIT OCW 18.02** — Multivariable Calculus: Vector analysis lectures
3. **Paul's Online Math Notes** — Calculus III: Rich practice problems

### Python Tools
- `sympy.vector`: Symbolic vector calculus (`gradient`, `divergence`, `curl`)
- `matplotlib.pyplot.quiver`: 2D vector field arrow visualization
- `matplotlib.pyplot.streamplot`: Streamline visualization
- `mpl_toolkits.mplot3d`: 3D surface and vector visualization

---

## Next Lesson

- **Previous**: [02. Complex Numbers](02_Complex_Numbers.md) — Complex algebra, polar/exponential form, De Moivre's theorem
- **Next**: [04. Curvilinear Coordinates and Multiple Integrals](04_Curvilinear_Coordinates.md) — Cylindrical/spherical coordinates, Jacobian, coordinate transformations
