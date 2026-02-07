# ODE Basics

## Overview

Ordinary Differential Equations (ODE) are equations that contain derivatives with respect to a single independent variable. They are widely used to describe temporal changes in physical systems.

---

## 1. Basic Concepts of ODE

### 1.1 Definition and Classification

```
General ODE:
F(t, y, y', y'', ..., y⁽ⁿ⁾) = 0

First-order ODE:
dy/dt = f(t, y)

nth-order ODE:
d^n y/dt^n = f(t, y, y', ..., y^(n-1))
```

### 1.2 Classification

```python
"""
1. Order: The order of the highest derivative
   - 1st order: y' = f(t, y)
   - 2nd order: y'' = f(t, y, y')

2. Linearity:
   - Linear: y'' + p(t)y' + q(t)y = g(t)
   - Nonlinear: y' = y²

3. Autonomy:
   - Autonomous: y' = f(y) (t not explicitly present)
   - Non-autonomous: y' = f(t, y)

4. Problem type:
   - Initial Value Problem (IVP): y(t₀) = y₀ given
   - Boundary Value Problem (BVP): y(a) = α, y(b) = β given
"""
```

### 1.3 Example ODEs

```python
import numpy as np
import matplotlib.pyplot as plt

# 1. Population growth model (exponential growth)
# dy/dt = ky, y(0) = y₀
# Solution: y(t) = y₀ * e^(kt)

def population_growth():
    k = 0.1  # Growth rate
    y0 = 100  # Initial population

    t = np.linspace(0, 50, 100)
    y = y0 * np.exp(k * t)

    plt.figure(figsize=(10, 4))
    plt.plot(t, y)
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title(f'Population Growth Model (k={k})')
    plt.grid(True)
    plt.show()

population_growth()

# 2. Radioactive decay
# dN/dt = -λN, N(0) = N₀
# Solution: N(t) = N₀ * e^(-λt)

def radioactive_decay():
    lambda_ = 0.05
    N0 = 1000
    half_life = np.log(2) / lambda_

    t = np.linspace(0, 100, 100)
    N = N0 * np.exp(-lambda_ * t)

    plt.figure(figsize=(10, 4))
    plt.plot(t, N)
    plt.axhline(y=N0/2, color='r', linestyle='--', label=f'Half-life: {half_life:.1f}')
    plt.xlabel('Time')
    plt.ylabel('Number of atoms')
    plt.title('Radioactive Decay')
    plt.legend()
    plt.grid(True)
    plt.show()

radioactive_decay()
```

---

## 2. Analytical Solutions

### 2.1 Separation of Variables

```python
"""
Form: dy/dt = g(t)h(y)

1/h(y) dy = g(t) dt
∫ 1/h(y) dy = ∫ g(t) dt

Example: dy/dt = ty
1/y dy = t dt
ln|y| = t²/2 + C
y = Ae^(t²/2)
"""

def separable_ode_example():
    # dy/dt = ty, y(0) = 1
    t = np.linspace(-2, 2, 100)
    y = np.exp(t**2 / 2)  # Analytical solution

    plt.figure(figsize=(8, 5))
    plt.plot(t, y)
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title("Solution of dy/dt = ty")
    plt.grid(True)
    plt.show()

separable_ode_example()
```

### 2.2 First-Order Linear ODE

```python
"""
dy/dt + p(t)y = q(t)

Integrating factor: μ(t) = e^(∫p(t)dt)

Solution: y = (1/μ) * ∫ μ*q dt + C/μ

Example: dy/dt + 2y = e^(-t)
μ = e^(2t)
y = e^(-2t) * ∫ e^(2t) * e^(-t) dt
y = e^(-2t) * (e^t + C)
y = e^(-t) + Ce^(-2t)
"""

def linear_ode_example():
    t = np.linspace(0, 5, 100)
    C = 1  # Determined from initial condition
    y = np.exp(-t) + C * np.exp(-2*t)

    plt.figure(figsize=(8, 5))
    plt.plot(t, y)
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title("Solution of dy/dt + 2y = e^(-t)")
    plt.grid(True)
    plt.show()

linear_ode_example()
```

### 2.3 Second-Order Linear ODE with Constant Coefficients

```python
"""
ay'' + by' + cy = 0

Characteristic equation: ar² + br + c = 0

Case 1: Two distinct real roots r₁, r₂
  y = C₁e^(r₁t) + C₂e^(r₂t)

Case 2: Repeated root r
  y = (C₁ + C₂t)e^(rt)

Case 3: Complex roots α ± βi
  y = e^(αt)(C₁cos(βt) + C₂sin(βt))
"""

def second_order_examples():
    t = np.linspace(0, 10, 200)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Case 1: y'' - 3y' + 2y = 0 (r=1, 2)
    y1 = np.exp(t) - np.exp(2*t)
    axes[0].plot(t, y1)
    axes[0].set_title("Distinct Real Roots")
    axes[0].set_xlabel('t')

    # Case 2: y'' - 2y' + y = 0 (r=1 repeated)
    y2 = (1 + t) * np.exp(t)
    axes[1].plot(t, y2)
    axes[1].set_title("Repeated Root")
    axes[1].set_xlabel('t')

    # Case 3: y'' + y = 0 (r=±i)
    y3 = np.cos(t) + np.sin(t)
    axes[2].plot(t, y3)
    axes[2].set_title("Complex Roots (Oscillation)")
    axes[2].set_xlabel('t')

    plt.tight_layout()
    plt.show()

second_order_examples()
```

---

## 3. Initial Value Problem (IVP)

### 3.1 Existence and Uniqueness

```python
"""
Lipschitz Condition:

|f(t, y₁) - f(t, y₂)| ≤ L|y₁ - y₂|

If this condition is satisfied, the solution exists and is unique.

Exception case:
dy/dt = 3y^(2/3), y(0) = 0
Solutions: y = 0 (trivial) or y = t³ (not unique)
"""

def non_unique_solution():
    t = np.linspace(-2, 2, 100)

    # Both solutions satisfy initial condition y(0) = 0
    y1 = np.zeros_like(t)  # y = 0
    y2 = t**3  # y = t³

    plt.figure(figsize=(8, 5))
    plt.plot(t, y1, label='y = 0')
    plt.plot(t, y2, label='y = t³')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title("dy/dt = 3y^(2/3) - Non-unique Solutions")
    plt.legend()
    plt.grid(True)
    plt.show()

non_unique_solution()
```

### 3.2 Converting Higher-Order ODE to First-Order System

```python
"""
Convert nth-order ODE to system of n first-order ODEs

Example: y'' + 4y' + 3y = 0

Transformation:
y₁ = y
y₂ = y' = y₁'

System:
y₁' = y₂
y₂' = -3y₁ - 4y₂

Matrix form:
[y₁']   [0   1 ] [y₁]
[y₂'] = [-3 -4] [y₂]
"""

def convert_to_system():
    # 2nd order ODE: y'' + 4y' + 3y = 0
    # Initial conditions: y(0) = 1, y'(0) = 0

    # Analytical solution for first-order system
    # Characteristic equation: r² + 4r + 3 = 0 → r = -1, -3
    # Solution: y = Ae^(-t) + Be^(-3t)
    # Apply initial conditions: A + B = 1, -A - 3B = 0
    # A = 3/2, B = -1/2

    t = np.linspace(0, 5, 100)
    y = 1.5 * np.exp(-t) - 0.5 * np.exp(-3*t)
    y_prime = -1.5 * np.exp(-t) + 1.5 * np.exp(-3*t)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(t, y, label='y(t)')
    axes[0].plot(t, y_prime, label="y'(t)")
    axes[0].set_xlabel('t')
    axes[0].set_title('Time Domain')
    axes[0].legend()
    axes[0].grid(True)

    # Phase plane
    axes[1].plot(y, y_prime)
    axes[1].set_xlabel('y')
    axes[1].set_ylabel("y'")
    axes[1].set_title('Phase Plane')
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

convert_to_system()
```

---

## 4. Phase Plane Analysis

### 4.1 Equilibrium Points and Stability

```python
"""
For dy/dt = f(y), equilibrium points: f(y*) = 0

Stability:
- f'(y*) < 0: Stable (converging)
- f'(y*) > 0: Unstable (diverging)
- f'(y*) = 0: Further analysis required
"""

def equilibrium_analysis():
    # dy/dt = y(1 - y) (logistic growth)
    # Equilibrium points: y* = 0 or y* = 1

    y = np.linspace(-0.5, 1.5, 100)
    f = y * (1 - y)

    plt.figure(figsize=(10, 5))

    # dy/dt vs y
    plt.subplot(1, 2, 1)
    plt.plot(y, f)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.scatter([0, 1], [0, 0], color='red', s=100, zorder=5)
    plt.xlabel('y')
    plt.ylabel('dy/dt')
    plt.title('dy/dt = y(1-y)')
    plt.grid(True)

    # f'(y) = 1 - 2y
    # f'(0) = 1 > 0: Unstable
    # f'(1) = -1 < 0: Stable

    # Time evolution
    plt.subplot(1, 2, 2)
    from scipy.integrate import solve_ivp

    t_span = (0, 10)
    t_eval = np.linspace(0, 10, 100)

    for y0 in [-0.1, 0.1, 0.5, 0.9, 1.1, 1.5]:
        sol = solve_ivp(lambda t, y: y*(1-y), t_span, [y0], t_eval=t_eval)
        plt.plot(sol.t, sol.y[0], label=f'y₀={y0}')

    plt.axhline(y=1, color='r', linestyle='--', label='Stable equilibrium')
    plt.axhline(y=0, color='b', linestyle='--', label='Unstable equilibrium')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.legend(loc='right')
    plt.title('Solutions from Various Initial Conditions')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

equilibrium_analysis()
```

### 4.2 2D Phase Plane

```python
def phase_plane_2d():
    """Phase plane for 2D system"""
    # Simple harmonic oscillator: x'' + x = 0
    # System: x' = v, v' = -x

    def harmonic_oscillator(t, state):
        x, v = state
        return [v, -x]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Vector field
    x_range = np.linspace(-2, 2, 15)
    v_range = np.linspace(-2, 2, 15)
    X, V = np.meshgrid(x_range, v_range)
    dX = V
    dV = -X

    axes[0].quiver(X, V, dX, dV, alpha=0.5)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('v')
    axes[0].set_title('Vector Field')
    axes[0].set_aspect('equal')

    # Trajectories
    from scipy.integrate import solve_ivp

    t_span = (0, 10)
    t_eval = np.linspace(0, 10, 200)

    for x0, v0 in [(1, 0), (0, 1), (1.5, 0.5)]:
        sol = solve_ivp(harmonic_oscillator, t_span, [x0, v0], t_eval=t_eval)
        axes[1].plot(sol.y[0], sol.y[1], label=f'({x0}, {v0})')

    axes[1].set_xlabel('x')
    axes[1].set_ylabel('v')
    axes[1].set_title('Phase Trajectories')
    axes[1].legend()
    axes[1].set_aspect('equal')
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

phase_plane_2d()
```

---

## 5. Physical Examples

### 5.1 Free Fall

```python
def free_fall():
    """Free fall under gravity (with air resistance)"""
    from scipy.integrate import solve_ivp

    # Parameters
    g = 9.8  # Gravitational acceleration
    k = 0.1  # Air resistance coefficient
    m = 1.0  # Mass

    # Equation of motion: m*dv/dt = mg - kv²
    # dv/dt = g - (k/m)v²

    def fall_with_drag(t, state):
        v = state[0]
        return [g - (k/m) * v * abs(v)]

    # Terminal velocity: v_terminal = sqrt(mg/k)
    v_terminal = np.sqrt(m * g / k)

    t_span = (0, 20)
    t_eval = np.linspace(0, 20, 200)

    sol = solve_ivp(fall_with_drag, t_span, [0], t_eval=t_eval)

    plt.figure(figsize=(10, 5))
    plt.plot(sol.t, sol.y[0])
    plt.axhline(y=v_terminal, color='r', linestyle='--',
                label=f'Terminal velocity: {v_terminal:.2f} m/s')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Free Fall with Air Resistance')
    plt.legend()
    plt.grid(True)
    plt.show()

free_fall()
```

### 5.2 RC Circuit

```python
def rc_circuit():
    """Transient response of RC circuit"""
    from scipy.integrate import solve_ivp

    # Parameters
    R = 1000  # Resistance (Ω)
    C = 1e-6  # Capacitance (F)
    V_source = 5  # Source voltage (V)
    tau = R * C  # Time constant

    # Charging: V_C' = (V_source - V_C) / (RC)
    def charging(t, V_C):
        return [(V_source - V_C[0]) / (R * C)]

    t_span = (0, 5 * tau)
    t_eval = np.linspace(0, 5*tau, 200)

    sol = solve_ivp(charging, t_span, [0], t_eval=t_eval)

    # Analytical solution
    t_analytical = np.linspace(0, 5*tau, 200)
    V_analytical = V_source * (1 - np.exp(-t_analytical / tau))

    plt.figure(figsize=(10, 5))
    plt.plot(sol.t * 1000, sol.y[0], 'b-', label='Numerical solution')
    plt.plot(t_analytical * 1000, V_analytical, 'r--', label='Analytical solution')
    plt.axhline(y=V_source * 0.632, color='g', linestyle=':',
                label=f'τ = {tau*1000:.3f} ms')
    plt.xlabel('Time (ms)')
    plt.ylabel('Capacitor Voltage (V)')
    plt.title('RC Circuit Charging')
    plt.legend()
    plt.grid(True)
    plt.show()

rc_circuit()
```

---

## Practice Problems

### Problem 1
Newton's law of cooling: dT/dt = -k(T - T_ambient)
Plot temperature vs time for initial temperature 90°C, ambient temperature 20°C, k=0.1.

```python
def exercise_1():
    from scipy.integrate import solve_ivp

    T_ambient = 20
    k = 0.1
    T0 = 90

    def cooling(t, T):
        return [-k * (T[0] - T_ambient)]

    sol = solve_ivp(cooling, (0, 50), [T0], t_eval=np.linspace(0, 50, 100))

    plt.figure(figsize=(10, 5))
    plt.plot(sol.t, sol.y[0])
    plt.axhline(y=T_ambient, color='r', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Temperature (°C)')
    plt.title("Newton's Law of Cooling")
    plt.grid(True)
    plt.show()

exercise_1()
```

### Problem 2
Damped oscillation: x'' + 2γx' + ω₀²x = 0
Plot phase plane and time response for ω₀ = 2, γ = 0.5.

```python
def exercise_2():
    from scipy.integrate import solve_ivp

    omega0 = 2
    gamma = 0.5

    def damped_oscillator(t, state):
        x, v = state
        return [v, -2*gamma*v - omega0**2*x]

    sol = solve_ivp(damped_oscillator, (0, 10), [1, 0],
                    t_eval=np.linspace(0, 10, 200))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(sol.t, sol.y[0])
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('x')
    axes[0].set_title('Time Response')
    axes[0].grid(True)

    axes[1].plot(sol.y[0], sol.y[1])
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('v')
    axes[1].set_title('Phase Plane')
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

exercise_2()
```

---

## Summary

| Concept | Content |
|------|------|
| ODE classification | Order, linearity, autonomy |
| Analytical solutions | Separation of variables, integrating factor, characteristic equation |
| Higher→First conversion | nth-order ODE → n first-order systems |
| Phase plane | Equilibrium points, stability, trajectory analysis |
| Existence/Uniqueness | Lipschitz condition |
