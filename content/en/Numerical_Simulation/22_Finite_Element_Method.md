# Finite Element Method (FEM)

## Learning Objectives
- Understand the weak formulation and variational principles
- Master finite element spaces and basis functions
- Construct element stiffness matrices and perform assembly
- Implement 1D FEM for the Poisson equation
- Handle Dirichlet and Neumann boundary conditions
- Understand error analysis and convergence rates
- Gain insight into 2D FEM extensions

## Table of Contents
1. [Introduction to FEM](#1-introduction-to-fem)
2. [Weak Form and Variational Formulation](#2-weak-form-and-variational-formulation)
3. [Finite Element Spaces](#3-finite-element-spaces)
4. [Element Stiffness Matrix and Assembly](#4-element-stiffness-matrix-and-assembly)
5. [1D FEM Implementation](#5-1d-fem-implementation)
6. [Boundary Conditions](#6-boundary-conditions)
7. [2D FEM Overview](#7-2d-fem-overview)
8. [Error Analysis and Convergence](#8-error-analysis-and-convergence)
9. [Practice Problems](#9-practice-problems)

---

## 1. Introduction to FEM

### 1.1 What is the Finite Element Method?

The Finite Element Method (FEM) is a powerful numerical technique for solving partial differential equations (PDEs). Unlike finite difference methods that approximate derivatives directly, FEM:

1. Converts the PDE into a **weak (variational) form**
2. Discretizes the domain into **elements** (triangles, tetrahedra, etc.)
3. Approximates the solution using **piecewise polynomial basis functions**
4. Reduces the problem to solving a linear system

```
┌─────────────────────────────────────────────────────────────┐
│                   FEM Workflow                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Strong Form (PDE)                                          │
│       ↓                                                     │
│  Weak Form (multiply by test function, integrate)          │
│       ↓                                                     │
│  Discretization (mesh + basis functions)                   │
│       ↓                                                     │
│  Matrix System (Au = f)                                     │
│       ↓                                                     │
│  Solve (direct or iterative)                               │
│       ↓                                                     │
│  Approximate Solution                                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Advantages:**
- Handles complex geometries (unstructured meshes)
- Rigorous mathematical foundation (variational calculus)
- Flexible: works for various boundary conditions
- Well-suited for structural mechanics, heat transfer, electromagnetics

**Applications:**
- Structural analysis (stress, strain, vibrations)
- Fluid dynamics (Navier-Stokes)
- Heat transfer
- Electromagnetics (Maxwell's equations)

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

np.set_printoptions(precision=4, suppress=True)
```

### 1.2 Model Problem: 1D Poisson Equation

We'll focus on the 1D Poisson equation:

```
-u''(x) = f(x),  x ∈ (0, 1)
u(0) = 0,  u(1) = 0  (Dirichlet boundary conditions)
```

**Example:** If f(x) = 1, the exact solution is u(x) = x(1-x)/2.

---

## 2. Weak Form and Variational Formulation

### 2.1 Deriving the Weak Form

**Strong form:**
```
-u''(x) = f(x),  x ∈ (0, 1)
u(0) = u(1) = 0
```

**Step 1:** Multiply by a test function v(x) and integrate:
```
-∫₀¹ u''(x) v(x) dx = ∫₀¹ f(x) v(x) dx
```

**Step 2:** Integrate by parts (transfer derivative to v):
```
∫₀¹ u'(x) v'(x) dx - [u'(x) v(x)]₀¹ = ∫₀¹ f(x) v(x) dx
```

Since v(0) = v(1) = 0 (test functions vanish at boundaries), the boundary term disappears:

```
∫₀¹ u'(x) v'(x) dx = ∫₀¹ f(x) v(x) dx
```

This is the **weak form** (also called **variational form**).

### 2.2 Bilinear Form and Functional

Define:
- Bilinear form: a(u, v) = ∫₀¹ u'(x) v'(x) dx
- Linear functional: L(v) = ∫₀¹ f(x) v(x) dx

**Weak formulation:** Find u such that
```
a(u, v) = L(v)  for all test functions v
```

```python
def weak_form_example():
    """
    Conceptual demonstration of weak form.

    Strong: -u'' = f
    Weak: ∫ u' v' dx = ∫ f v dx
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

## 3. Finite Element Spaces

### 3.1 Mesh and Elements

Divide [0, 1] into N elements:

```
┌────────────────────────────────────────────────────────────┐
│              1D Mesh (N=4 elements)                        │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  x₀=0    x₁      x₂      x₃      x₄=1                     │
│   o───────o───────o───────o───────o                       │
│   │ Elem1 │ Elem2 │ Elem3 │ Elem4 │                       │
│   └───────┴───────┴───────┴───────┘                       │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

Each element [xᵢ, xᵢ₊₁] has length h = 1/N (for uniform mesh).

### 3.2 Hat Functions (Linear Basis)

We approximate u(x) as:
```
u(x) ≈ uₕ(x) = Σᵢ₌₀ᴺ uᵢ φᵢ(x)
```

where φᵢ(x) are **hat functions** (piecewise linear, nodal basis):

```
φᵢ(x) = { (x - xᵢ₋₁)/h,  x ∈ [xᵢ₋₁, xᵢ]
        { (xᵢ₊₁ - x)/h,  x ∈ [xᵢ, xᵢ₊₁]
        { 0,              otherwise

┌────────────────────────────────────────────────────────────┐
│                    Hat Functions                           │
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

**Properties:**
- φᵢ(xⱼ) = δᵢⱼ (Kronecker delta)
- Local support: φᵢ(x) ≠ 0 only on [xᵢ₋₁, xᵢ₊₁]
- uₕ(xᵢ) = uᵢ (nodal interpolation)

```python
def hat_function(x, xi_minus, xi, xi_plus):
    """
    Evaluate hat function centered at xi.
    """
    phi = np.zeros_like(x)

    # Left slope
    mask1 = (x >= xi_minus) & (x <= xi)
    phi[mask1] = (x[mask1] - xi_minus) / (xi - xi_minus)

    # Right slope
    mask2 = (x >= xi) & (x <= xi_plus)
    phi[mask2] = (xi_plus - x[mask2]) / (xi_plus - xi)

    return phi

# Plot hat functions
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

## 4. Element Stiffness Matrix and Assembly

### 4.1 Galerkin Discretization

Substitute uₕ = Σᵢ uᵢ φᵢ into the weak form:

```
Σᵢ uᵢ ∫₀¹ φᵢ' φⱼ' dx = ∫₀¹ f φⱼ dx,  for j = 0, 1, ..., N
```

This gives the linear system **Au = f**, where:
- Aᵢⱼ = ∫₀¹ φᵢ'(x) φⱼ'(x) dx (stiffness matrix)
- fⱼ = ∫₀¹ f(x) φⱼ(x) dx (load vector)

### 4.2 Element Stiffness Matrix

For element e = [xₑ, xₑ₊₁], the local stiffness matrix is:

```
Aₑ = ∫_{xₑ}^{xₑ₊₁} φ'ₑ,ᵢ φ'ₑ,ⱼ dx

For linear elements:
  φₑ,₁(x) = (xₑ₊₁ - x)/h,  φ'ₑ,₁ = -1/h
  φₑ,₂(x) = (x - xₑ)/h,    φ'ₑ,₂ = 1/h

  Aₑ = (1/h) [ 1  -1 ]
             [-1   1 ]
```

```python
def element_stiffness_1d(h):
    """
    Compute element stiffness matrix for 1D linear element.

    Aₑ = (1/h) [ 1  -1 ]
               [-1   1 ]
    """
    A_elem = (1.0 / h) * np.array([[ 1, -1],
                                     [-1,  1]])
    return A_elem

h = 0.25  # Element size
A_elem = element_stiffness_1d(h)
print("Element stiffness matrix:")
print(A_elem)
```

### 4.3 Assembly

Assemble global matrix A by summing contributions from each element:

```
┌────────────────────────────────────────────────────────────┐
│              Assembly Process (N=3 elements)               │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Element 1: [x₀, x₁]  →  A[0:2, 0:2] += Aₑ                │
│  Element 2: [x₁, x₂]  →  A[1:3, 1:3] += Aₑ                │
│  Element 3: [x₂, x₃]  →  A[2:4, 2:4] += Aₑ                │
│                                                            │
│  Result: Tridiagonal symmetric matrix                     │
│    [ 1  -1   0   0 ]                                      │
│    [-1   2  -1   0 ]  (1/h factor)                        │
│    [ 0  -1   2  -1 ]                                      │
│    [ 0   0  -1   1 ]                                      │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

```python
def assemble_stiffness_1d(N):
    """
    Assemble global stiffness matrix for 1D problem.

    Parameters:
    -----------
    N : int, number of elements

    Returns:
    --------
    A : (N+1) x (N+1) sparse matrix
    """
    h = 1.0 / N
    A = lil_matrix((N+1, N+1))
    A_elem = element_stiffness_1d(h)

    for e in range(N):
        # Global node indices for element e
        nodes = [e, e+1]

        # Add element contribution
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

## 5. 1D FEM Implementation

### 5.1 Complete FEM Solver

```python
def fem_1d_poisson(N, f_func):
    """
    Solve 1D Poisson equation using FEM.

    -u''(x) = f(x),  x ∈ (0, 1)
    u(0) = u(1) = 0

    Parameters:
    -----------
    N : int, number of elements
    f_func : function, right-hand side f(x)

    Returns:
    --------
    x : array, node coordinates
    u : array, FEM solution at nodes
    """
    h = 1.0 / N
    x = np.linspace(0, 1, N+1)

    # Assemble stiffness matrix
    A = assemble_stiffness_1d(N)

    # Assemble load vector
    f = np.zeros(N+1)
    for e in range(N):
        x_left = x[e]
        x_right = x[e+1]

        # Midpoint rule for integration
        x_mid = (x_left + x_right) / 2
        f_mid = f_func(x_mid)

        # Contribution to load vector
        f[e] += f_mid * h / 2
        f[e+1] += f_mid * h / 2

    # Apply Dirichlet boundary conditions: u(0) = u(N) = 0
    # Modify first and last row
    A = A.tolil()
    A[0, :] = 0
    A[0, 0] = 1
    f[0] = 0

    A[N, :] = 0
    A[N, N] = 1
    f[N] = 0

    A = A.tocsr()

    # Solve linear system
    u = spsolve(A, f)

    return x, u

# Example: f(x) = 1, exact solution u(x) = x(1-x)/2
def f(x):
    return np.ones_like(x)

def u_exact(x):
    return x * (1 - x) / 2

N = 10
x, u = fem_1d_poisson(N, f)

# Plot
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

# Compute error
u_exact_nodes = u_exact(x)
error_L2 = np.sqrt(np.sum((u - u_exact_nodes)**2 * (1.0/N)))
error_Linf = np.max(np.abs(u - u_exact_nodes))

print(f"L2 error:   {error_L2:.4e}")
print(f"L∞ error:   {error_Linf:.4e}")
```

### 5.2 Variable Source Term

```python
# Example: f(x) = π² sin(πx), exact solution u(x) = sin(πx)
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

## 6. Boundary Conditions

### 6.1 Dirichlet Boundary Conditions

Already implemented above: set u(0) = α, u(1) = β by modifying rows in the system.

```python
def fem_1d_dirichlet(N, f_func, u_left, u_right):
    """
    Solve with Dirichlet BC: u(0) = u_left, u(1) = u_right.
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

    # Apply BC
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

# Example: u(0) = 0.5, u(1) = 0.2
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

### 6.2 Neumann Boundary Conditions

For Neumann BC (flux condition), e.g., u'(0) = g₀, u'(1) = g₁:

The weak form naturally incorporates Neumann BC through the boundary term:
```
∫₀¹ u' v' dx = ∫₀¹ f v dx + [u' v]₀¹
             = ∫₀¹ f v dx + g₁ v(1) - g₀ v(0)
```

```python
def fem_1d_neumann(N, f_func, g_left, g_right):
    """
    Solve with Neumann BC: u'(0) = g_left, u'(1) = g_right.

    Note: Pure Neumann problem has a solution only if ∫f dx + g_right - g_left = 0.
    We fix u(0) = 0 to make the problem well-posed.
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

    # Add Neumann BC contributions
    f[0] -= g_left
    f[N] += g_right

    # Fix one value to remove null space (e.g., u(0) = 0)
    A = A.tolil()
    A[0, :] = 0
    A[0, 0] = 1
    f[0] = 0

    A = A.tocsr()
    u = spsolve(A, f)

    return x, u

# Example: -u'' = 0, u'(0) = 1, u'(1) = 1 → u(x) = x + C, fix C by u(0)=0
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

## 7. 2D FEM Overview

### 7.1 2D Poisson Equation

In 2D, the Poisson equation is:
```
-Δu = -∂²u/∂x² - ∂²u/∂y² = f(x, y),  (x, y) ∈ Ω
u = 0 on ∂Ω (boundary)
```

**Weak form:**
```
∫_Ω ∇u · ∇v dA = ∫_Ω f v dA
```

### 7.2 Triangular Elements

The domain Ω is divided into triangular elements:

```
┌────────────────────────────────────────────────────────────┐
│              2D Triangular Mesh                            │
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

Each triangle has 3 vertices (nodes). Linear basis functions φᵢ(x, y) satisfy:
- φᵢ(xⱼ, yⱼ) = δᵢⱼ
- φᵢ is linear within each triangle
- φᵢ = 0 outside triangles containing node i

### 7.3 Element Stiffness Matrix (2D)

For a triangle with vertices (x₁, y₁), (x₂, y₂), (x₃, y₃):

```
Aᵢⱼ = ∫_T ∇φᵢ · ∇φⱼ dA

where ∇φᵢ = [∂φᵢ/∂x, ∂φᵢ/∂y]ᵀ (constant within triangle)
```

```python
def triangle_area(p1, p2, p3):
    """Compute area of triangle with vertices p1, p2, p3."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) -
                     (p3[0] - p1[0]) * (p2[1] - p1[1]))

def triangle_stiffness_2d(p1, p2, p3):
    """
    Compute element stiffness matrix for 2D linear triangle.

    Parameters:
    -----------
    p1, p2, p3 : tuples (x, y), triangle vertices

    Returns:
    --------
    A_elem : 3x3 array
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    area = triangle_area(p1, p2, p3)

    # Gradient of basis functions (constant in triangle)
    b = np.array([y2 - y3, y3 - y1, y1 - y2])
    c = np.array([x3 - x2, x1 - x3, x2 - x1])

    # Stiffness matrix
    A_elem = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            A_elem[i, j] = (b[i]*b[j] + c[i]*c[j]) / (4 * area)

    return A_elem

# Example triangle
p1 = (0, 0)
p2 = (1, 0)
p3 = (0, 1)
A_elem = triangle_stiffness_2d(p1, p2, p3)
print("2D triangle stiffness matrix:")
print(A_elem)
```

---

## 8. Error Analysis and Convergence

### 8.1 Convergence Rate

For linear finite elements (piecewise linear basis) solving -u'' = f:

**Theoretical convergence:**
- L² error: ||u - uₕ||_{L²} = O(h²)
- H¹ error: ||u - uₕ||_{H¹} = O(h)  (gradient error)
- L∞ error: ||u - uₕ||_{L∞} = O(h²)

where h is the element size.

### 8.2 Convergence Test

```python
def convergence_test():
    """
    Test FEM convergence by solving -u'' = π² sin(πx) with increasing N.
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

    # Plot convergence
    plt.figure(figsize=(10, 5))
    plt.loglog(h_values, errors_L2, 'o-', label='L² error', linewidth=2, markersize=8)
    plt.loglog(h_values, errors_Linf, 's-', label='L∞ error', linewidth=2, markersize=8)

    # Reference slopes
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

**Output:**
```
N=  5, h=0.2000: L2=3.9464e-03, L∞=4.9299e-03
N= 10, h=0.1000: L2=9.9067e-04, L∞=1.2337e-03
N= 20, h=0.0500: L2=2.4794e-04, L∞=3.0852e-04
N= 40, h=0.0250: L2=6.2007e-05, L∞=7.7139e-05
N= 80, h=0.0125: L2=1.5504e-05, L∞=1.9286e-05
```

The errors decrease as O(h²), confirming theoretical predictions.

---

## 9. Practice Problems

### Problem 1: Variable Coefficient
Solve the variable coefficient problem:
```
-(a(x) u'(x))' = f(x),  x ∈ (0, 1)
u(0) = u(1) = 0
```
with a(x) = 1 + x and f(x) = 1. Implement FEM and compare with a numerical reference.

**Hint:** Weak form is ∫ a(x) u'(x) v'(x) dx = ∫ f(x) v(x) dx.

### Problem 2: Mixed Boundary Conditions
Solve -u'' = 1 with u(0) = 0 (Dirichlet) and u'(1) = -0.5 (Neumann). Compare with exact solution.

### Problem 3: Quadratic Elements
Extend the 1D FEM code to use quadratic basis functions (3 nodes per element: two endpoints + midpoint). Compare convergence rate with linear elements.

### Problem 4: 2D Poisson on Unit Square
Implement 2D FEM for -Δu = 1 on the unit square [0,1]×[0,1] with u=0 on boundary. Use a structured triangular mesh and verify the solution.

### Problem 5: Higher-Order Convergence
Test convergence for the equation -u'' = f(x) where f(x) is chosen such that the exact solution u(x) = x(1-x) sin(πx). Verify that L² error is O(h²) for linear elements.

---

## Navigation
- Previous: [21. Spectral Methods](21_Spectral_Methods.md)
- Next: [Overview](00_Overview.md)
- [Back to Overview](00_Overview.md)
