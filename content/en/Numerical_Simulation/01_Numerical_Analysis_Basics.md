# Numerical Analysis Basics

## Overview

Numerical analysis studies methods for approximately solving mathematical problems using computers. We will learn about floating-point representation, error analysis, numerical differentiation and integration, which form the foundation of simulation.

---

## 1. Floating-Point Representation

### 1.1 IEEE 754 Standard

```python
import numpy as np
import struct

# Check floating-point bit representation
def float_to_bits(f):
    """Convert float64 to bit string"""
    packed = struct.pack('>d', f)
    bits = ''.join(f'{b:08b}' for b in packed)
    return f"Sign: {bits[0]} | Exponent: {bits[1:12]} | Mantissa: {bits[12:]}"

print(float_to_bits(1.0))
print(float_to_bits(-1.0))
print(float_to_bits(0.1))

# Machine epsilon
print(f"\nfloat64 machine epsilon: {np.finfo(np.float64).eps}")
print(f"float32 machine epsilon: {np.finfo(np.float32).eps}")
```

### 1.2 Numerical Limits

```python
# Overflow and underflow
print("float64 range:")
print(f"  Minimum: {np.finfo(np.float64).min}")
print(f"  Maximum: {np.finfo(np.float64).max}")
print(f"  Smallest positive: {np.finfo(np.float64).tiny}")

# Precision loss example
a = 1e16
b = 1.0
print(f"\n1e16 + 1 - 1e16 = {(a + b) - a}")  # 0.0 (precision loss)
print(f"1 + 1e16 - 1e16 = {b + (a - a)}")    # 1.0 (correct result)
```

### 1.3 Rounding Error

```python
# 0.1 cannot be represented exactly
x = 0.1
print(f"0.1 actual value: {x:.20f}")
print(f"0.1 + 0.2 = 0.3? {0.1 + 0.2 == 0.3}")  # False

# Use tolerance for comparison
print(f"np.isclose: {np.isclose(0.1 + 0.2, 0.3)}")
```

---

## 2. Error Analysis

### 2.1 Error Types

```python
def analyze_error(true_value, approx_value):
    """Calculate absolute and relative error"""
    abs_error = abs(true_value - approx_value)
    rel_error = abs_error / abs(true_value) if true_value != 0 else float('inf')
    return abs_error, rel_error

# Example: π approximation
import math
approximations = [
    ("22/7", 22/7),
    ("355/113", 355/113),
    ("3.14159", 3.14159),
]

print("π approximation error analysis:")
for name, approx in approximations:
    abs_e, rel_e = analyze_error(math.pi, approx)
    print(f"  {name:10}: Absolute error={abs_e:.2e}, Relative error={rel_e:.2e}")
```

### 2.2 Numerical Stability

```python
# Unstable computation example: small difference from large number
def unstable_subtract(x):
    """Numerically unstable subtraction"""
    return (1 + x) - 1

def stable_subtract(x):
    """Numerically stable form"""
    return x

x_values = [1e-15, 1e-16, 1e-17]
print("Small number subtraction comparison:")
for x in x_values:
    print(f"  x={x}: Unstable={unstable_subtract(x):.2e}, Stable={stable_subtract(x):.2e}")
```

### 2.3 Condition Number

```python
# Matrix condition number
def analyze_condition_number():
    # Well-conditioned matrix
    A_good = np.array([[1, 0], [0, 1]])

    # Poorly-conditioned matrix
    A_bad = np.array([[1, 1], [1, 1.0001]])

    print("Condition number analysis:")
    print(f"  Identity matrix: {np.linalg.cond(A_good):.2f}")
    print(f"  Nearly singular matrix: {np.linalg.cond(A_bad):.2f}")

analyze_condition_number()
```

---

## 3. Numerical Differentiation

### 3.1 Finite Difference Method

```python
def numerical_derivatives(f, x, h=1e-5):
    """Various finite difference formulas"""
    # Forward difference
    forward = (f(x + h) - f(x)) / h

    # Backward difference
    backward = (f(x) - f(x - h)) / h

    # Central difference - more accurate
    central = (f(x + h) - f(x - h)) / (2 * h)

    return forward, backward, central

# Test: f(x) = sin(x), f'(x) = cos(x)
x = np.pi / 4
true_deriv = np.cos(x)

forward, backward, central = numerical_derivatives(np.sin, x)

print(f"Derivative of sin(x) at x = π/4:")
print(f"  True value: {true_deriv:.10f}")
print(f"  Forward difference: {forward:.10f}, Error: {abs(forward - true_deriv):.2e}")
print(f"  Backward difference: {backward:.10f}, Error: {abs(backward - true_deriv):.2e}")
print(f"  Central difference: {central:.10f}, Error: {abs(central - true_deriv):.2e}")
```

### 3.2 Higher-Order Derivatives

```python
def second_derivative(f, x, h=1e-5):
    """Second derivative (central difference)"""
    return (f(x + h) - 2*f(x) + f(x - h)) / h**2

# f(x) = sin(x), f''(x) = -sin(x)
x = np.pi / 4
true_second = -np.sin(x)
approx_second = second_derivative(np.sin, x)

print(f"\nSecond derivative:")
print(f"  True value: {true_second:.10f}")
print(f"  Approximation: {approx_second:.10f}")
print(f"  Error: {abs(approx_second - true_second):.2e}")
```

### 3.3 Effect of Step Size

```python
def analyze_step_size():
    """Error analysis based on step size"""
    f = np.sin
    x = 1.0
    true_deriv = np.cos(x)

    h_values = np.logspace(-1, -15, 15)
    errors = []

    for h in h_values:
        central = (f(x + h) - f(x - h)) / (2 * h)
        errors.append(abs(central - true_deriv))

    return h_values, errors

h_values, errors = analyze_step_size()

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.loglog(h_values, errors, 'bo-')
plt.xlabel('Step size h')
plt.ylabel('Error')
plt.title('Central Difference Error vs Step Size')
plt.grid(True)
plt.axvline(x=1e-8, color='r', linestyle='--', label='Near optimal')
plt.legend()
plt.show()
# Too small h: increased rounding error
# Too large h: increased truncation error
```

---

## 4. Numerical Integration

### 4.1 Trapezoidal Rule

```python
def trapezoidal(f, a, b, n):
    """Integration using trapezoidal rule"""
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)

    # Trapezoidal rule
    integral = h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])
    return integral

# Test: ∫₀^π sin(x) dx = 2
result = trapezoidal(np.sin, 0, np.pi, 100)
print(f"∫₀^π sin(x) dx:")
print(f"  True value: 2.0")
print(f"  Trapezoidal (n=100): {result:.10f}")
print(f"  Error: {abs(result - 2.0):.2e}")
```

### 4.2 Simpson's Rule

```python
def simpson(f, a, b, n):
    """Simpson's 1/3 rule (n must be even)"""
    if n % 2 != 0:
        n += 1

    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)

    # Simpson's rule: (h/3) * [y₀ + 4y₁ + 2y₂ + 4y₃ + ... + yₙ]
    integral = h/3 * (y[0] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-1:2]) + y[-1])
    return integral

result_trap = trapezoidal(np.sin, 0, np.pi, 10)
result_simp = simpson(np.sin, 0, np.pi, 10)

print(f"\nComparison at n=10:")
print(f"  Trapezoidal: {result_trap:.10f}, Error: {abs(result_trap - 2.0):.2e}")
print(f"  Simpson: {result_simp:.10f}, Error: {abs(result_simp - 2.0):.2e}")
```

### 4.3 SciPy Integration

```python
from scipy import integrate

# 1D integration
result, error = integrate.quad(np.sin, 0, np.pi)
print(f"\nscipy.integrate.quad:")
print(f"  Result: {result:.15f}")
print(f"  Estimated error: {error:.2e}")

# 2D integration
def f_2d(y, x):
    return x * y

result_2d, error_2d = integrate.dblquad(f_2d, 0, 1, 0, 1)
print(f"\n∫∫ xy dxdy (0~1):")
print(f"  Result: {result_2d:.10f}")  # 0.25
```

### 4.4 Convergence Analysis

```python
def convergence_analysis():
    """Integration convergence analysis"""
    true_value = 2.0  # ∫₀^π sin(x) dx
    n_values = [4, 8, 16, 32, 64, 128, 256]

    trap_errors = []
    simp_errors = []

    for n in n_values:
        trap_errors.append(abs(trapezoidal(np.sin, 0, np.pi, n) - true_value))
        simp_errors.append(abs(simpson(np.sin, 0, np.pi, n) - true_value))

    # Estimate convergence order
    print("Convergence analysis:")
    print(f"{'n':>6} {'Trapezoidal':>12} {'Simpson':>12}")
    for i, n in enumerate(n_values):
        print(f"{n:>6} {trap_errors[i]:>12.2e} {simp_errors[i]:>12.2e}")

    # Trapezoidal: O(h²), Simpson: O(h⁴)
    return n_values, trap_errors, simp_errors

convergence_analysis()
```

---

## 5. Practice Problems

### Problem 1: Numerical Differentiation
Calculate the derivative of f(x) = e^(-x²) at x=0.5 with various step sizes and analyze the error.

```python
def exercise_1():
    f = lambda x: np.exp(-x**2)
    f_prime = lambda x: -2*x * np.exp(-x**2)  # Analytical derivative

    x = 0.5
    true_value = f_prime(x)

    # Solution
    h_values = np.logspace(-1, -12, 12)
    for h in h_values:
        approx = (f(x + h) - f(x - h)) / (2 * h)
        print(f"h={h:.0e}: Error={abs(approx - true_value):.2e}")

exercise_1()
```

### Problem 2: Numerical Integration
Calculate ∫₀^1 e^(-x²) dx using trapezoidal and Simpson's rules.

```python
def exercise_2():
    f = lambda x: np.exp(-x**2)

    # scipy reference value
    true_val, _ = integrate.quad(f, 0, 1)

    # Solution
    for n in [10, 50, 100]:
        trap = trapezoidal(f, 0, 1, n)
        simp = simpson(f, 0, 1, n)
        print(f"n={n}: Trapezoidal={trap:.8f}, Simpson={simp:.8f}")

    print(f"True value: {true_val:.8f}")

exercise_2()
```

---

## Summary

| Concept | Key Content |
|------|----------|
| Floating-point | IEEE 754, machine epsilon, precision limits |
| Error types | Truncation error, rounding error, condition number |
| Numerical differentiation | Forward/backward/central difference, step size selection |
| Numerical integration | Trapezoidal (O(h²)), Simpson (O(h⁴)) |
