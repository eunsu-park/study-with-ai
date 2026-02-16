# 14. LSTM and GRU

[Previous: RNN Basics](./13_RNN_Basics.md) | [Next: LSTM & GRU Implementation](./15_Impl_LSTM_GRU.md)

---

## Learning Objectives

- Understand LSTM and GRU structures
- Learn gate mechanisms
- Learn long-term dependencies
- Implement with PyTorch

---

## 1. LSTM (Long Short-Term Memory)

### Problem: RNN's Vanishing Gradient

```
h100 ← W × W × ... × W × h1
            ↑
    Gradient converges to 0
```

### Solution: Cell State

```
LSTM = Cell State (long-term memory) + Hidden State (short-term memory)
```

### LSTM Structure

```
       ┌──────────────────────────────────────┐
       │            Cell State (C)              │
       │     ×─────(+)─────────────────────►    │
       │     ↑      ↑                           │
       │    forget  input                       │
       │    gate    gate                        │
       │     ↑      ↑                           │
h(t-1)─┴──►[σ]   [σ][tanh]    [σ]──►×──────►h(t)
           f(t)   i(t) g(t)   o(t)     ↑
                              output gate
```

### Gate Formulas

```python
# Forget Gate: How much to forget from previous memory
f(t) = σ(W_f × [h(t-1), x(t)] + b_f)

# Input Gate: How much new information to store
i(t) = σ(W_i × [h(t-1), x(t)] + b_i)

# Cell Candidate: New candidate information
g(t) = tanh(W_g × [h(t-1), x(t)] + b_g)

# Cell State Update
C(t) = f(t) × C(t-1) + i(t) × g(t)

# Output Gate: How much of cell state to output
o(t) = σ(W_o × [h(t-1), x(t)] + b_o)

# Hidden State
h(t) = o(t) × tanh(C(t))
```

---

## 2. GRU (Gated Recurrent Unit)

### Simplified Version of LSTM

```
GRU = Reset Gate + Update Gate
(Merges cell state and hidden state)
```

### GRU Structure

```
       Update Gate (z)
       ┌────────────────────────────┐
       │                            │
h(t-1)─┴──►[σ]───z(t)──────×──(+)──►h(t)
              │           ↑    ↑
              │      ┌────┘    │
              │      │   ×─────┘
              │      │   ↑
              ├──►[σ]   [tanh]
              │   r(t)    │
              │    │      │
              └────×──────┘
                Reset Gate (r)
```

### Gate Formulas

```python
# Update Gate: Ratio of previous state vs new state
z(t) = σ(W_z × [h(t-1), x(t)] + b_z)

# Reset Gate: How much to forget previous state
r(t) = σ(W_r × [h(t-1), x(t)] + b_r)

# Candidate Hidden
h̃(t) = tanh(W × [r(t) × h(t-1), x(t)] + b)

# Hidden State Update
h(t) = (1 - z(t)) × h(t-1) + z(t) × h̃(t)
```

---

## 3. PyTorch LSTM/GRU

### LSTM

```python
lstm = nn.LSTM(
    input_size=10,
    hidden_size=20,
    num_layers=2,
    batch_first=True,
    dropout=0.1,
    bidirectional=False
)

# Forward pass
# output: Hidden states at all times
# (h_n, c_n): Last (hidden, cell) states
output, (h_n, c_n) = lstm(x)
```

### GRU

```python
gru = nn.GRU(
    input_size=10,
    hidden_size=20,
    num_layers=2,
    batch_first=True
)

# Forward pass (no cell state)
output, h_n = gru(x)
```

---

## 4. LSTM Classifier

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        # Bidirectional so hidden_dim * 2
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x: (batch, seq) - token indices
        embedded = self.embedding(x)

        # LSTM
        output, (h_n, c_n) = self.lstm(embedded)

        # Combine bidirectional last hidden states
        # h_n: (num_layers*2, batch, hidden)
        forward_last = h_n[-2]  # Forward last layer
        backward_last = h_n[-1]  # Backward last layer
        combined = torch.cat([forward_last, backward_last], dim=1)

        return self.fc(combined)
```

---

## 5. Sequence Generation (Language Model)

```python
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        logits = self.fc(output)
        return logits, hidden

    def generate(self, start_token, max_len, temperature=1.0):
        self.eval()
        tokens = [start_token]
        hidden = None

        with torch.no_grad():
            for _ in range(max_len):
                x = torch.tensor([[tokens[-1]]])
                logits, hidden = self(x, hidden)

                # Temperature sampling
                probs = F.softmax(logits[0, -1] / temperature, dim=0)
                next_token = torch.multinomial(probs, 1).item()
                tokens.append(next_token)

        return tokens
```

---

## 6. LSTM vs GRU Comparison

| Item | LSTM | GRU |
|------|------|-----|
| Number of Gates | 3 (f, i, o) | 2 (r, z) |
| States | Cell + Hidden | Hidden only |
| Parameters | More | Fewer |
| Training Speed | Slower | Faster |
| Performance | Complex patterns | Similar or slightly lower |

### Selection Guide

- **LSTM**: Long sequences, complex dependencies
- **GRU**: Fast training, limited resources

---

## 7. Practical Tips

### Initialization

```python
# Initialize hidden state
def init_hidden(batch_size, hidden_size, num_layers, bidirectional):
    num_directions = 2 if bidirectional else 1
    h = torch.zeros(num_layers * num_directions, batch_size, hidden_size)
    c = torch.zeros(num_layers * num_directions, batch_size, hidden_size)
    return (h.to(device), c.to(device))
```

### Dropout Pattern

```python
class LSTMWithDropout(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        output, (h_n, _) = self.lstm(x)
        # Apply dropout to last hidden state
        dropped = self.dropout(h_n[-1])
        return self.fc(dropped)
```

---

## Summary

### Core Concepts

1. **LSTM**: Maintain long-term memory with cell state, 3 gates
2. **GRU**: Simplified LSTM, 2 gates
3. **Gates**: Control information flow (sigmoid × value)

### Key Code

```python
# LSTM
lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
output, (h_n, c_n) = lstm(x)

# GRU
gru = nn.GRU(input_size, hidden_size, batch_first=True)
output, h_n = gru(x)
```

---

## Next Steps

In [16_Attention_Transformer.md](./16_Attention_Transformer.md), we'll learn Seq2Seq and Attention.
