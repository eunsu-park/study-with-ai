# 04. Training Techniques

## Learning Objectives

- Understand gradient descent variants (SGD, Momentum, Adam)
- Learn learning rate scheduling
- Learn regularization techniques (Dropout, Weight Decay, Batch Norm)
- Learn overfitting prevention and early stopping

---

## 1. Gradient Descent

### Basic Principle

```
W(t+1) = W(t) - η × ∇L
```
- η: learning rate
- ∇L: gradient of loss function

### Variants

| Method | Formula | Characteristics |
|--------|---------|-----------------|
| SGD | W -= lr × g | Simple, slow |
| Momentum | v = βv + g; W -= lr × v | Adds inertia |
| AdaGrad | Adaptive learning rate | Good for sparse data |
| RMSprop | Exponential moving average | Improved AdaGrad |
| Adam | Momentum + RMSprop | Most commonly used |

---

## 2. Momentum

Adds inertia to reduce oscillations.

```
v(t) = β × v(t-1) + ∇L
W(t+1) = W(t) - η × v(t)
```

### NumPy Implementation

```python
def sgd_momentum(W, grad, v, lr=0.01, beta=0.9):
    v = beta * v + grad          # Update velocity
    W = W - lr * v               # Update weights
    return W, v
```

### PyTorch

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

---

## 3. Adam Optimizer

Combines advantages of Momentum and RMSprop.

```
m(t) = β₁ × m(t-1) + (1-β₁) × g      # 1st moment
v(t) = β₂ × v(t-1) + (1-β₂) × g²     # 2nd moment
m̂ = m / (1 - β₁ᵗ)                    # Bias correction
v̂ = v / (1 - β₂ᵗ)
W = W - η × m̂ / (√v̂ + ε)
```

### NumPy Implementation

```python
def adam(W, grad, m, v, t, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad ** 2)

    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    W = W - lr * m_hat / (np.sqrt(v_hat) + eps)
    return W, m, v
```

### PyTorch

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

---

## 4. Learning Rate Scheduling

Adjust learning rate during training.

### Main Methods

| Method | Characteristics |
|--------|----------------|
| Step Decay | Reduce by γ every N epochs |
| Exponential | lr = lr₀ × γᵉᵖᵒᶜʰ |
| Cosine Annealing | Reduce following cosine function |
| ReduceLROnPlateau | Reduce when validation loss plateaus |
| Warmup | Gradual increase at beginning |

### PyTorch Examples

```python
# Step Decay
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Cosine Annealing
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# ReduceLROnPlateau
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=10, factor=0.5
)

# In training loop
for epoch in range(epochs):
    train(...)
    scheduler.step()  # Call at end of epoch
```

---

## 5. Dropout

Randomly deactivates neurons during training.

### Principle

```
Training: y = x × mask / (1 - p)   # mask is Bernoulli(1-p)
Inference: y = x                   # No mask
```

### NumPy Implementation

```python
def dropout(x, p=0.5, training=True):
    if not training:
        return x
    mask = (np.random.rand(*x.shape) > p).astype(float)
    return x * mask / (1 - p)
```

### PyTorch

```python
class MLPWithDropout(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_p=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Active only during training
        x = self.fc2(x)
        return x

# During inference
model.eval()  # Disable dropout
```

---

## 6. Batch Normalization

Normalizes inputs at each layer.

### Formula

```
μ = mean(x)
σ² = var(x)
x̂ = (x - μ) / √(σ² + ε)
y = γ × x̂ + β   # Learnable parameters
```

### NumPy Implementation

```python
def batch_norm(x, gamma, beta, eps=1e-5, training=True,
               running_mean=None, running_var=None, momentum=0.1):
    if training:
        mean = np.mean(x, axis=0)
        var = np.var(x, axis=0)

        # Update running averages
        if running_mean is not None:
            running_mean = momentum * mean + (1 - momentum) * running_mean
            running_var = momentum * var + (1 - momentum) * running_var
    else:
        mean = running_mean
        var = running_var

    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta
```

### PyTorch

```python
class CNNWithBatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64, 10)
        self.bn_fc = nn.BatchNorm1d(10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = x.flatten(1)
        x = self.bn_fc(self.fc1(x))
        return x
```

---

## 7. Weight Decay (L2 Regularization)

Penalizes weight magnitudes.

### Formula

```
L_total = L_data + λ × ||W||²
∇L_total = ∇L_data + 2λW
```

### PyTorch

```python
# Method 1: Set in optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Method 2: Add directly to loss
l2_lambda = 1e-4
l2_reg = sum(p.pow(2).sum() for p in model.parameters())
loss = criterion(output, target) + l2_lambda * l2_reg
```

---

## 8. Early Stopping

Stop training when validation loss stops improving.

### PyTorch Implementation

```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# Usage
early_stopping = EarlyStopping(patience=10)
for epoch in range(epochs):
    train_loss = train(model, train_loader)
    val_loss = validate(model, val_loader)

    early_stopping(val_loss)
    if early_stopping.early_stop:
        print(f"Early stopping at epoch {epoch}")
        break
```

---

## 9. Data Augmentation

Transform training data to increase diversity.

### Image Data

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
```

---

## 10. NumPy vs PyTorch Comparison

### Optimizer Implementation

```python
# NumPy (manual implementation)
m = np.zeros_like(W)
v = np.zeros_like(W)
for t in range(1, epochs + 1):
    grad = compute_gradient(W, X, y)
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad**2
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    W -= lr * m_hat / (np.sqrt(v_hat) + eps)

# PyTorch (automatic)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(epochs):
    loss = criterion(model(X), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## Summary

### Core Concepts

1. **Optimizer**: Adam is the default choice, SGD+Momentum still valid
2. **Learning Rate**: Improve convergence with proper scheduling
3. **Regularization**: Combine Dropout, BatchNorm, Weight Decay
4. **Early Stopping**: Basic overfitting prevention

### Recommended Starting Settings

```python
# Basic configuration
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
```

---

## Next Steps

In [05_CNN_Basics.md](./05_CNN_Basics.md), we'll learn convolutional neural networks.
