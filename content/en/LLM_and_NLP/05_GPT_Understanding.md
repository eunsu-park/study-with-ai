# 05. GPT Understanding

## Learning Objectives

- Understanding GPT architecture
- Autoregressive language modeling
- Text generation techniques
- Evolution of GPT series

---

## 1. GPT Overview

### Generative Pre-trained Transformer

```
GPT = Stack of Transformer decoders

Features:
- Unidirectional (left→right)
- Autoregressive generation
- Trained via next token prediction
```

### BERT vs GPT

| Item | BERT | GPT |
|------|------|-----|
| Architecture | Encoder | Decoder |
| Direction | Bidirectional | Unidirectional |
| Training | MLM | Next token prediction |
| Use Cases | Understanding (classification, QA) | Generation (dialogue, writing) |

---

## 2. Autoregressive Language Modeling

### Training Objective

```
P(x) = P(x₁) × P(x₂|x₁) × P(x₃|x₁,x₂) × ...

Sentence: "I love NLP"
P("I") × P("love"|"I") × P("NLP"|"I love") × P("<eos>"|"I love NLP")

Loss: -log P(next token | previous tokens)
```

### Causal Language Modeling

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def causal_lm_loss(logits, targets):
    """
    logits: (batch, seq, vocab_size)
    targets: (batch, seq) - next token

    Input: [BOS, I, love, NLP]
    Target: [I, love, NLP, EOS]
    """
    batch_size, seq_len, vocab_size = logits.shape

    # (batch*seq, vocab) vs (batch*seq,)
    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        targets.view(-1),
        ignore_index=-100  # Ignore padding
    )
    return loss
```

---

## 3. GPT Architecture

### Structure

```
Input tokens
    ↓
Token Embedding + Position Embedding
    ↓
┌─────────────────────────────────┐
│  Masked Multi-Head Attention    │
│           ↓                     │
│  Add & LayerNorm                │
│           ↓                     │
│  Feed Forward                   │
│           ↓                     │
│  Add & LayerNorm                │
└─────────────────────────────────┘
            × N layers
    ↓
LayerNorm
    ↓
Linear (vocab_size)
    ↓
Softmax → Next token probability
```

### Implementation

```python
class GPTBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        # Pre-LayerNorm (GPT-2 style)
        ln_x = self.ln1(x)
        attn_out, _ = self.attn(ln_x, ln_x, ln_x, attn_mask=attn_mask)
        x = x + self.dropout(attn_out)

        ln_x = self.ln2(x)
        x = x + self.ffn(ln_x)

        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_heads=12,
                 num_layers=12, d_ff=3072, max_len=1024, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            GPTBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.token_emb.weight

        # Register causal mask
        mask = torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
        self.register_buffer('causal_mask', mask)

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        assert seq_len <= self.max_len

        # Embeddings
        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.drop(x)

        # Causal mask
        mask = self.causal_mask[:seq_len, :seq_len]

        # Transformer blocks
        for block in self.blocks:
            x = block(x, attn_mask=mask)

        x = self.ln_f(x)
        logits = self.head(x)  # (batch, seq, vocab)

        return logits
```

---

## 4. Text Generation

### Greedy Decoding

```python
def generate_greedy(model, input_ids, max_new_tokens):
    """Always select highest probability token"""
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
    return input_ids
```

### Temperature Sampling

```python
def generate_with_temperature(model, input_ids, max_new_tokens, temperature=1.0):
    """Control distribution with temperature"""
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
    return input_ids

# temperature < 1: more deterministic (prefer high probability tokens)
# temperature > 1: more random (increase diversity)
```

### Top-k Sampling

```python
def generate_top_k(model, input_ids, max_new_tokens, k=50, temperature=1.0):
    """Sample only from top k tokens"""
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)[:, -1, :] / temperature

            # Top-k filtering
            top_k_logits, top_k_indices = logits.topk(k, dim=-1)
            probs = F.softmax(top_k_logits, dim=-1)

            # Sampling
            idx = torch.multinomial(probs, num_samples=1)
            next_token = top_k_indices.gather(-1, idx)

            input_ids = torch.cat([input_ids, next_token], dim=1)
    return input_ids
```

### Top-p (Nucleus) Sampling

```python
def generate_top_p(model, input_ids, max_new_tokens, p=0.9, temperature=1.0):
    """Sample from tokens with cumulative probability up to p"""
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)

            # Sort probabilities in descending order
            sorted_probs, sorted_indices = probs.sort(descending=True)
            cumsum = sorted_probs.cumsum(dim=-1)

            # Mask tokens after p
            mask = cumsum - sorted_probs > p
            sorted_probs[mask] = 0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

            # Sampling
            idx = torch.multinomial(sorted_probs, num_samples=1)
            next_token = sorted_indices.gather(-1, idx)

            input_ids = torch.cat([input_ids, next_token], dim=1)
    return input_ids
```

---

## 5. GPT Series

### GPT-1 (2018)

```
- 12 layers, 768 dim, 117M parameters
- Trained on BooksCorpus
- Introduced fine-tuning paradigm
```

### GPT-2 (2019)

```
- Up to 48 layers, 1.5B parameters
- Trained on WebText (40GB)
- Discovered zero-shot capabilities
- "Too dangerous to release"

Size variants:
- Small: 117M (same as GPT-1)
- Medium: 345M
- Large: 762M
- XL: 1.5B
```

### GPT-3 (2020)

```
- 96 layers, 175B parameters
- Few-shot / In-context Learning
- Available only via API

Key findings:
- Perform various tasks with prompts alone
- Scaling laws: model size ↑ = performance ↑
```

### GPT-4 (2023)

```
- Multimodal (text + images)
- Longer context (8K, 32K, 128K)
- Improved reasoning capabilities
- Aligned with RLHF
```

---

## 6. HuggingFace GPT-2

### Basic Usage

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Text generation
input_text = "The quick brown fox"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Generate
output = model.generate(
    input_ids,
    max_length=50,
    num_return_sequences=1,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
```

### Generation Parameters

```python
output = model.generate(
    input_ids,
    max_length=100,           # Maximum length
    min_length=10,            # Minimum length
    do_sample=True,           # Use sampling
    temperature=0.8,          # Temperature
    top_k=50,                 # Top-k
    top_p=0.95,               # Top-p
    num_return_sequences=3,   # Number of sequences
    no_repeat_ngram_size=2,   # Prevent n-gram repetition
    repetition_penalty=1.2,   # Repetition penalty
    pad_token_id=tokenizer.eos_token_id
)
```

### Conditional Generation

```python
# Prompt-based generation
prompt = """
Q: What is the capital of France?
A:"""

input_ids = tokenizer.encode(prompt, return_tensors='pt')
output = model.generate(
    input_ids,
    max_new_tokens=20,
    do_sample=False  # Greedy
)
print(tokenizer.decode(output[0]))
```

---

## 7. In-Context Learning

### Zero-shot

```
Perform task with prompt alone:

"Translate English to French:
Hello, how are you? →"
```

### Few-shot

```
Include examples in prompt:

"Translate English to French:
Hello → Bonjour
Thank you → Merci
Good morning → Bonjour
How are you? →"
```

### Chain-of-Thought (CoT)

```
Guide step-by-step reasoning:

"Q: Roger has 5 tennis balls. He buys 2 more cans of 3 balls each.
How many balls does he have now?
A: Let's think step by step.
Roger started with 5 balls.
2 cans of 3 balls each = 6 balls.
5 + 6 = 11 balls.
The answer is 11."
```

---

## 8. KV Cache

### Efficient Generation

```python
class GPTWithKVCache(nn.Module):
    def forward(self, input_ids, past_key_values=None):
        """
        past_key_values: K, V cache from previous tokens
        Compute only for new token and update cache
        """
        if past_key_values is None:
            # Compute entire sequence
            ...
        else:
            # Compute only last token
            ...

        return logits, new_past_key_values

# During generation
past = None
for _ in range(max_new_tokens):
    logits, past = model(new_token, past_key_values=past)
    # O(1) complexity instead of O(n)
```

### HuggingFace KV Cache

```python
output = model.generate(
    input_ids,
    max_new_tokens=50,
    use_cache=True  # Enable KV Cache (default)
)
```

---

## Summary

### Generation Strategy Comparison

| Method | Advantages | Disadvantages | Use Cases |
|--------|-----------|---------------|-----------|
| Greedy | Fast, consistent | Repetitive, boring | Translation, QA |
| Temperature | Control diversity | Requires tuning | General generation |
| Top-k | Stable | Fixed k | General generation |
| Top-p | Adaptive | Slightly slower | Creative, dialogue |

### Key Code

```python
# HuggingFace GPT-2
from transformers import GPT2Tokenizer, GPT2LMHeadModel
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Generation
output = model.generate(input_ids, max_length=50, do_sample=True,
                        temperature=0.7, top_p=0.9)
```

---

## Next Steps

Learn about the HuggingFace Transformers library in [06_HuggingFace_Basics.md](./06_HuggingFace_Basics.md).
