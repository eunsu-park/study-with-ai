# 08. RNN Basics (Recurrent Neural Networks)

## Learning Objectives

- Understand the concept and structure of recurrent neural networks
- Process sequence data
- Use PyTorch nn.RNN
- Understand vanishing gradient problem

---

## 1. What is RNN?

### Characteristics of Sequential Data

```
Time series: [1, 2, 3, 4, 5, ...]  - Previous values affect next values
Text: "I go to school"              - Previous words affect next words
```

### MLP Limitations

- Fixed input size
- Ignores order information
- Cannot handle variable-length sequences

### RNN Solution

```
h(t) = tanh(W_xh × x(t) + W_hh × h(t-1) + b)

h(t): Current hidden state
x(t): Current input
h(t-1): Previous hidden state
```

---

## 2. RNN Structure

### Time Unrolling

```
    x1      x2      x3      x4
    ↓       ↓       ↓       ↓
  ┌───┐   ┌───┐   ┌───┐   ┌───┐
  │ h │──►│ h │──►│ h │──►│ h │──► Output
  └───┘   └───┘   └───┘   └───┘
    h0      h1      h2      h3
```

### Parameter Sharing

- Same W_xh, W_hh used at all time steps
- Can process variable-length sequences

---

## 3. PyTorch RNN

### Basic Usage

```python
import torch
import torch.nn as nn

# Create RNN
rnn = nn.RNN(
    input_size=10,    # Input dimension
    hidden_size=20,   # Hidden state dimension
    num_layers=2,     # Number of RNN layers
    batch_first=True  # Input: (batch, seq, feature)
)

# Input shape: (batch_size, seq_len, input_size)
x = torch.randn(32, 15, 10)  # Batch 32, Sequence 15, Features 10

# Forward pass
# output: Hidden states at all times (batch, seq, hidden)
# h_n: Last hidden state (layers, batch, hidden)
output, h_n = rnn(x)

print(f"output: {output.shape}")  # (32, 15, 20)
print(f"h_n: {h_n.shape}")        # (2, 32, 20)
```

### Bidirectional RNN

```python
rnn_bi = nn.RNN(
    input_size=10,
    hidden_size=20,
    num_layers=1,
    batch_first=True,
    bidirectional=True  # Bidirectional
)

output, h_n = rnn_bi(x)
print(f"output: {output.shape}")  # (32, 15, 40) - Forward+Backward
print(f"h_n: {h_n.shape}")        # (2, 32, 20) - Last state per direction
```

---

## 4. RNN Classifier Implementation

### Sequence Classification Model

```python
class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=1):
        super().__init__()
        self.rnn = nn.RNN(
            input_size, hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, seq, features)
        output, h_n = self.rnn(x)

        # Use last time step's hidden state
        # h_n[-1]: Last layer's hidden state
        out = self.fc(h_n[-1])
        return out
```

### Many-to-Many Structure

```python
class RNNSeq2Seq(nn.Module):
    """Sequence → Sequence"""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, _ = self.rnn(x)
        # Apply FC to all time steps
        out = self.fc(output)  # (batch, seq, output_size)
        return out
```

---

## 5. Vanishing Gradient Problem

### Problem

```
In long sequences:
h100 ← W_hh × W_hh × ... × W_hh × h1
                    ↑
            100 multiplications → Exploding or vanishing gradients
```

### Cause

- |W_hh| > 1: Exploding gradients
- |W_hh| < 1: Vanishing gradients

### Solutions

1. **Use LSTM/GRU** (next lesson)
2. **Gradient Clipping**

```python
# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## 6. Time Series Prediction Example

### Sine Wave Prediction

```python
import numpy as np

# Generate data
def generate_sin_data(seq_len=50, n_samples=1000):
    X = []
    y = []
    for _ in range(n_samples):
        start = np.random.uniform(0, 2*np.pi)
        seq = np.sin(np.linspace(start, start + 4*np.pi, seq_len + 1))
        X.append(seq[:-1].reshape(-1, 1))
        y.append(seq[-1])
    return np.array(X), np.array(y)

X, y = generate_sin_data()
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Model
class SinPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(1, 32, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        _, h_n = self.rnn(x)
        return self.fc(h_n[-1]).squeeze()
```

---

## 7. Text Classification Example

### Character-level RNN

```python
class CharRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, seq) - character indices
        embedded = self.embedding(x)  # (batch, seq, embed)
        output, h_n = self.rnn(embedded)
        out = self.fc(h_n[-1])
        return out

# Example
vocab_size = 27  # a-z + space
model = CharRNN(vocab_size, embed_size=32, hidden_size=64, num_classes=5)
```

---

## 8. Important Notes

### Input Shape

```python
# batch_first=True  → (batch, seq, feature)
# batch_first=False → (seq, batch, feature)  # Default
```

### Variable-length Sequences

```python
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Padded sequences and actual lengths
padded_seqs = ...  # (batch, max_len, features)
lengths = ...      # Actual length of each sequence

# Pack (ignore padding)
packed = pack_padded_sequence(padded_seqs, lengths,
                               batch_first=True, enforce_sorted=False)
output, h_n = rnn(packed)

# Unpack
output_padded, _ = pad_packed_sequence(output, batch_first=True)
```

---

## 9. RNN Variant Comparison

| Model | Advantages | Disadvantages |
|-------|-----------|---------------|
| Simple RNN | Simple, fast | Difficult to learn long sequences |
| LSTM | Learn long-term dependencies | Complex, slow |
| GRU | Similar to LSTM, simpler | - |

---

## Summary

### Core Concepts

1. **Recurrent Structure**: Previous state affects next computation
2. **Parameter Sharing**: Time-independent weights
3. **Gradient Problem**: Learning difficulty in long sequences

### Key Code

```python
rnn = nn.RNN(input_size, hidden_size, batch_first=True)
output, h_n = rnn(x)  # output: all, h_n: last
```

---

## Next Steps

In [09_LSTM_GRU.md](./09_LSTM_GRU.md), we'll learn LSTM and GRU.
