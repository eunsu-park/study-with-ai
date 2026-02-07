# 10. Attention and Transformer

## Learning Objectives

- Understand the principles of Attention mechanism
- Learn Self-Attention
- Understand Transformer architecture
- Implement with PyTorch

---

## 1. Need for Attention

### Seq2Seq Limitations

```
Encoder: "I go to school" → Fixed-size vector
                              ↓
Decoder: Fixed vector → "나는 학교에 간다"

Problem: Information loss when long sentences compressed
```

### Attention Solution

```
When decoder generates each output word,
it can "attend" to all encoder words

Generating "I" → High attention on "나는"
Generating "school" → High attention on "학교"
```

---

## 2. Attention Mechanism

### Formula

```python
# Query, Key, Value
Q = Current decoder state
K = All encoder states
V = All encoder states (usually same as K)

# Attention Score
score = Q @ K.T  # (query_len, key_len)

# Attention Weight (softmax)
weight = softmax(score / sqrt(d_k))  # Scaling

# Context
context = weight @ V  # Weighted sum
```

### Scaled Dot-Product Attention

```python
def attention(Q, K, V, mask=None):
    d_k = K.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    weights = F.softmax(scores, dim=-1)
    return weights @ V, weights
```

---

## 3. Self-Attention

### Concept

```
Each word attends to all other words in the same sequence

"The cat sat on the mat because it was tired"
"it" has high attention on "cat" → Pronoun resolution
```

### Formula

```python
# Generate Q, K, V from input X
Q = X @ W_Q
K = X @ W_K
V = X @ W_V

# Self-Attention
output = attention(Q, K, V)
```

---

## 4. Multi-Head Attention

### Idea

```
Multiple attention heads learn different relationships

Head 1: Grammatical relationships
Head 2: Semantic relationships
Head 3: Positional relationships
...
```

### Formula

```python
def multi_head_attention(Q, K, V, num_heads):
    d_model = Q.size(-1)
    d_k = d_model // num_heads

    # Split heads
    Q = Q.view(batch, seq, num_heads, d_k).transpose(1, 2)
    K = K.view(batch, seq, num_heads, d_k).transpose(1, 2)
    V = V.view(batch, seq, num_heads, d_k).transpose(1, 2)

    # Attention for each head
    attn_output, _ = attention(Q, K, V)

    # Combine heads
    output = attn_output.transpose(1, 2).contiguous().view(batch, seq, d_model)
    return output
```

---

## 5. Transformer Architecture

### Structure

```
Input → Embedding → Positional Encoding
                      ↓
┌─────────────────────────────────────┐
│  Multi-Head Self-Attention          │
│           ↓                         │
│  Add & LayerNorm                    │
│           ↓                         │
│  Feed Forward Network               │
│           ↓                         │
│  Add & LayerNorm                    │
└─────────────────────────────────────┘
            × N layers
                ↓
             Output
```

### Key Components

1. **Multi-Head Attention**
2. **Position-wise Feed Forward**
3. **Residual Connection**
4. **Layer Normalization**
5. **Positional Encoding**

---

## 6. Positional Encoding

### Necessity

```
Attention has no order information
→ Explicitly add position information
```

### Sinusoidal Encoding

```python
def positional_encoding(seq_len, d_model):
    PE = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000) / d_model))

    PE[:, 0::2] = torch.sin(position * div_term)
    PE[:, 1::2] = torch.cos(position * div_term)
    return PE
```

---

## 7. PyTorch Transformer

### Basic Usage

```python
import torch.nn as nn

# Transformer encoder
encoder_layer = nn.TransformerEncoderLayer(
    d_model=512,
    nhead=8,
    dim_feedforward=2048,
    dropout=0.1
)
encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

# Forward pass
x = torch.randn(10, 32, 512)  # (seq, batch, d_model)
output = encoder(x)
```

### Classification Model

```python
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (batch, seq)
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (seq, batch, d_model)
        x = self.transformer(x)
        x = x.mean(dim=0)  # Mean pooling
        return self.fc(x)
```

---

## 8. Vision Transformer (ViT)

### Idea

```
Split image into patches → Process as sequence

Image (224×224) → 16×16 patches (196 patches) → Transformer
```

### Structure

```python
class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, num_classes, d_model, nhead, num_layers):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2

        self.patch_embed = nn.Linear(patch_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # Extract and embed patches
        patches = extract_patches(x)
        x = self.patch_embed(patches)

        # Add CLS token
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Position embedding
        x = x + self.pos_embed

        # Transformer
        x = self.transformer(x.transpose(0, 1))

        # Classification (use CLS token)
        return self.fc(x[0])
```

---

## 9. Attention vs RNN Comparison

| Item | RNN/LSTM | Transformer |
|------|----------|-------------|
| Parallelization | Difficult | Easy |
| Long-range Dependencies | Difficult | Easy |
| Training Speed | Slow | Fast |
| Memory | O(n) | O(n²) |
| Position Information | Implicit | Explicit |

---

## 10. Practical Applications

### NLP

- BERT: Bidirectional encoder
- GPT: Decoder-based generation
- T5: Encoder-decoder

### Vision

- ViT: Image classification
- DETR: Object detection
- Swin Transformer: Hierarchical structure

---

## Summary

### Core Concepts

1. **Attention**: Calculate relevance with Query-Key-Value
2. **Self-Attention**: Reference all positions within sequence
3. **Multi-Head**: Learn various relationships simultaneously
4. **Positional Encoding**: Add order information

### Key Code

```python
# Scaled Dot-Product Attention
scores = Q @ K.T / sqrt(d_k)
weights = softmax(scores)
output = weights @ V

# PyTorch Transformer
encoder = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(d_model=512, nhead=8),
    num_layers=6
)
```

---

## Next Steps

In [11_Training_Optimization.md](./11_Training_Optimization.md), we'll learn advanced training techniques.
