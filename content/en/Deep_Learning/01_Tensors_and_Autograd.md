# 01. Tensors and Autograd

## Learning Objectives

- Understand the concept of tensors and their differences from NumPy arrays
- Understand PyTorch's automatic differentiation (Autograd) system
- Learn the basics of GPU operations

---

## 1. What is a Tensor?

A tensor is a generalized concept of multi-dimensional arrays.

| Dimension | Name | Example |
|-----------|------|---------|
| 0D | Scalar | Single number (5) |
| 1D | Vector | [1, 2, 3] |
| 2D | Matrix | [[1,2], [3,4]] |
| 3D | 3D Tensor | Image (H, W, C) |
| 4D | 4D Tensor | Batch of images (N, C, H, W) |

---

## 2. NumPy vs PyTorch Tensor Comparison

### Creation

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

### Conversion

```python
# NumPy → PyTorch
tensor = torch.from_numpy(np_arr)

# PyTorch → NumPy
array = tensor.numpy()  # Only works for CPU tensors
```

### Key Differences

| Feature | NumPy | PyTorch |
|---------|-------|---------|
| GPU Support | ❌ | ✅ (`tensor.to('cuda')`) |
| Automatic Differentiation | ❌ | ✅ (`requires_grad=True`) |
| Default Type | float64 | float32 |
| Memory Sharing | - | `from_numpy` shares memory |

---

## 3. Automatic Differentiation (Autograd)

A core feature of PyTorch that automatically computes backpropagation.

### Basic Usage

```python
# Enable gradient tracking with requires_grad=True
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x + 1  # y = x² + 3x + 1

# Backpropagation (compute dy/dx)
y.backward()

# Check gradient
print(x.grad)  # tensor([7.])  # dy/dx = 2x + 3 = 2*2 + 3 = 7
```

### Computational Graph

```
    x ─────┐
           │
    x² ────┼──▶ + ──▶ y
           │
    3x ────┘
```

- **Forward pass**: Computation from input → output
- **Backward pass**: Gradient computation from output → input

### Gradient Accumulation and Initialization

```python
# Gradients accumulate
x.grad.zero_()  # Always initialize in training loop
```

---

## 4. Operations and Broadcasting

```python
a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)

# Basic operations
c = a + b           # Element-wise addition
c = a * b           # Element-wise multiplication (Hadamard product)
c = a @ b           # Matrix multiplication
c = torch.matmul(a, b)  # Matrix multiplication

# Broadcasting
a = torch.tensor([[1], [2], [3]])  # (3, 1)
b = torch.tensor([10, 20, 30])     # (3,)
c = a + b  # (3, 3) automatic expansion
```

---

## 5. GPU Operations

```python
# Check GPU availability
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Move tensor to GPU
x = torch.randn(1000, 1000)
x_gpu = x.to(device)
# Or
x_gpu = x.cuda()

# Operations (performed on the same device)
y_gpu = x_gpu @ x_gpu

# Bring result back to CPU
y_cpu = y_gpu.cpu()
```

---

## 6. Exercise: NumPy vs PyTorch Automatic Differentiation Comparison

### Problem: Find the derivative of f(x) = x³ + 2x² - 5x + 3 at x=2

Mathematical solution:
- f'(x) = 3x² + 4x - 5
- f'(2) = 3(4) + 4(2) - 5 = 12 + 8 - 5 = 15

### NumPy (Manual Differentiation)

```python
import numpy as np

def f(x):
    return x**3 + 2*x**2 - 5*x + 3

def df(x):
    """Manually compute derivative"""
    return 3*x**2 + 4*x - 5

x = 2.0
print(f"f({x}) = {f(x)}")
print(f"f'({x}) = {df(x)}")  # 15.0
```

### PyTorch (Automatic Differentiation)

```python
import torch

x = torch.tensor([2.0], requires_grad=True)
y = x**3 + 2*x**2 - 5*x + 3

y.backward()
print(f"f({x.item()}) = {y.item()}")
print(f"f'({x.item()}) = {x.grad.item()}")  # 15.0
```

---

## 7. Important Notes

### In-place Operations

```python
# In-place operations can conflict with autograd
x = torch.tensor([1.0], requires_grad=True)
# x += 1  # May cause error
x = x + 1  # Create new tensor (safe)
```

### Disabling Gradient Tracking

```python
# Save memory during inference
with torch.no_grad():
    y = model(x)  # No gradient computation

# Or
x.requires_grad = False
```

### detach()

```python
# Detach from computational graph
y = x.detach()  # y does not track gradients
```

---

## Summary

### What to Understand from NumPy
- Tensors are multi-dimensional arrays
- Matrix operations (multiplication, transpose, broadcasting)

### What PyTorch Adds
- `requires_grad`: Enable automatic differentiation
- `backward()`: Perform backpropagation
- `grad`: Computed gradients
- GPU acceleration

---

## Next Steps

In [02_Neural_Network_Basics.md](./02_Neural_Network_Basics.md), we'll use these tensors and automatic differentiation to build neural networks.
