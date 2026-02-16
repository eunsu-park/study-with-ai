# 03. Understanding Backpropagation

[Previous: Neural Network Basics](./02_Neural_Network_Basics.md) | [Next: Training Techniques](./04_Training_Techniques.md)

---

## Learning Objectives

- Understand the principles of the backpropagation algorithm
- Learn gradient calculation using the chain rule
- Implement backpropagation directly with NumPy

---

## 1. What is Backpropagation?

Backpropagation is an algorithm for training neural network weights.

```
Forward Pass:  Input ──▶ Hidden Layer ──▶ Output ──▶ Loss
Backward Pass: Input ◀── Hidden Layer ◀── Output ◀── Loss
```

### Core Ideas

1. **Forward Pass**: Compute values from input to output
2. **Loss Calculation**: Difference between prediction and ground truth
3. **Backward Pass**: Propagate gradients from loss towards input
4. **Weight Update**: Adjust weights using gradients

---

## 2. Chain Rule

The differentiation rule for composite functions.

### Formula

```
y = f(g(x))

dy/dx = (dy/dg) × (dg/dx)
```

### Example

```
z = x²
y = sin(z)
L = y²

dL/dx = (dL/dy) × (dy/dz) × (dz/dx)
      = 2y × cos(z) × 2x
```

---

## 3. Backpropagation for a Single Neuron

### Forward Pass

```python
z = w*x + b      # Linear transformation
a = sigmoid(z)    # Activation
L = (a - y)²     # Loss (MSE)
```

### Backward Pass (Gradient Calculation)

```python
dL/da = 2(a - y)                    # Gradient of loss w.r.t. activation
da/dz = sigmoid(z) * (1 - sigmoid(z))  # Sigmoid derivative
dz/dw = x                           # Gradient of linear transform w.r.t. weight
dz/db = 1                           # Gradient of linear transform w.r.t. bias

# Apply chain rule
dL/dw = (dL/da) × (da/dz) × (dz/dw)
dL/db = (dL/da) × (da/dz) × (dz/db)
```

---

## 4. Loss Functions

### MSE (Mean Squared Error)

```python
L = (1/n) × Σ(y_pred - y_true)²
dL/dy_pred = (2/n) × (y_pred - y_true)
```

### Cross-Entropy (Classification)

```python
L = -Σ y_true × log(y_pred)
dL/dy_pred = -y_true / y_pred  # Simplified when combined with softmax
```

### Softmax + Cross-Entropy Combined

```python
# Amazing result: becomes very simple
dL/dz = y_pred - y_true  # Gradient w.r.t. softmax input
```

---

## 5. MLP Backpropagation

Backpropagation process for a 2-layer MLP.

### Architecture

```
Input(x) → [W1, b1] → ReLU → [W2, b2] → Output(y)
```

### Forward Pass

```python
z1 = x @ W1 + b1
a1 = relu(z1)
z2 = a1 @ W2 + b2
y_pred = z2  # Or softmax(z2)
```

### Backward Pass

```python
# Output layer
dL/dz2 = y_pred - y_true  # (for softmax + CE)
dL/dW2 = a1.T @ dL/dz2
dL/db2 = sum(dL/dz2, axis=0)

# Hidden layer
dL/da1 = dL/dz2 @ W2.T
dL/dz1 = dL/da1 * relu_derivative(z1)
dL/dW1 = x.T @ dL/dz1
dL/db1 = sum(dL/dz1, axis=0)
```

---

## 6. NumPy Implementation Core

```python
class MLP:
    def backward(self, x, y_true, y_pred, cache):
        """Backpropagation: compute gradients"""
        a1, z1 = cache

        # Output layer gradients
        dz2 = y_pred - y_true
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0)

        # Hidden layer gradients (chain rule)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (z1 > 0)  # ReLU derivative
        dW1 = x.T @ dz1
        db1 = np.sum(dz1, axis=0)

        return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
```

---

## 7. PyTorch's Automatic Differentiation

In PyTorch, all of this is automatic.

```python
# Forward pass
y_pred = model(x)
loss = criterion(y_pred, y_true)

# Backward pass (automatic!)
loss.backward()

# Access gradients
print(model.fc1.weight.grad)
```

### Computational Graph

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
z = y * 3
z.backward()

# x.grad = dz/dx = dz/dy × dy/dx = 3 × 2x = 12
```

---

## 8. Vanishing/Exploding Gradient Problems

### Vanishing Gradient

- Cause: Derivatives of sigmoid/tanh close to 0
- Solution: ReLU, Residual Connections

### Exploding Gradient

- Cause: Gradient accumulation in deep networks
- Solution: Gradient Clipping, Batch Normalization

```python
# Gradient Clipping in PyTorch
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## 9. Numerical Gradient Verification

A method to verify if backpropagation implementation is correct.

```python
def numerical_gradient(f, x, h=1e-5):
    """Compute gradient using numerical differentiation"""
    grad = np.zeros_like(x)
    for i in range(x.size):
        x_plus = x.copy()
        x_plus.flat[i] += h
        x_minus = x.copy()
        x_minus.flat[i] -= h
        grad.flat[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad

# Verification
analytical_grad = backward(...)  # Analytical gradient
numerical_grad = numerical_gradient(loss_fn, weights)
diff = np.linalg.norm(analytical_grad - numerical_grad)
assert diff < 1e-5, "Gradient check failed!"
```

---

## Summary

### Core of Backpropagation

1. **Chain Rule**: Core of composite function differentiation
2. **Local Computation**: Gradients computed independently at each layer
3. **Gradient Propagation**: Propagate from output towards input

### What You Learn from NumPy

- Meaning of matrix transpose and multiplication
- Role of activation function derivatives
- Gradient summation in batch processing

### Moving to PyTorch

- All gradients computed in one line with `loss.backward()`
- Automatic computational graph construction
- GPU acceleration

---

## Next Steps

In [04_Training_Techniques.md](./04_Training_Techniques.md), we'll learn methods for weight updates using gradients.
