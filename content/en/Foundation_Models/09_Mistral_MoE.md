# Mistral & Mixture of Experts

## Learning Objectives
- Understand the architectural features of Mistral 7B
- Grasp Mixture of Experts (MoE) concepts and operation principles
- Learn the Mixtral 8x7B structure
- Master the pros/cons and practical applications of Sparse MoE

---

## 1. Mistral 7B Overview

### 1.1 Mistral's Innovation

**Mistral 7B** is a model released by Mistral AI in 2023, achieving 13B-level performance with only 7B parameters.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Mistral 7B Features                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Performance Comparison (as of 2023.10):                        │
│  ┌───────────────────────────────────────────────────────┐      │
│  │  Model          │ Params │ MMLU  │ HellaSwag │ GSM8K │      │
│  │  ───────────────│────────│───────│───────────│───────│      │
│  │  LLaMA 2 7B     │ 7B     │ 45.3  │ 77.2      │ 14.6  │      │
│  │  LLaMA 2 13B    │ 13B    │ 54.8  │ 80.7      │ 28.7  │      │
│  │  Mistral 7B     │ 7B     │ 60.1  │ 81.3      │ 52.2  │ ←!   │
│  └───────────────────────────────────────────────────────┘      │
│                                                                 │
│  Key Technologies:                                               │
│  • Sliding Window Attention (SWA)                               │
│  • Grouped Query Attention (GQA)                                │
│  • Over-training with more data                                 │
│  • Flash Attention 2 optimization                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Mistral Architecture Specifications

```python
MISTRAL_CONFIGS = {
    "mistral-7b": {
        "dim": 4096,
        "n_layers": 32,
        "n_heads": 32,
        "n_kv_heads": 8,           # GQA
        "head_dim": 128,
        "hidden_dim": 14336,
        "vocab_size": 32000,
        "context_length": 32768,   # Technical limit
        "sliding_window": 4096,    # Sliding Window Attention
        "rope_theta": 10000.0,
    },
}

# Comparison with LLaMA 2 7B
LLAMA2_7B = {
    "dim": 4096,
    "n_layers": 32,
    "n_heads": 32,
    "n_kv_heads": 32,              # MHA (no GQA)
    "hidden_dim": 11008,
    "context_length": 4096,
    "sliding_window": None,        # Full attention
}
```

---

## 2. Sliding Window Attention (SWA)

### 2.1 Concept

**Sliding Window Attention** restricts each token to attend only to tokens within a fixed window.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Sliding Window Attention                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Full Attention (Traditional):                                  │
│  ────────────────────────                                       │
│  Every token attends to all previous tokens                     │
│  Complexity: O(n²)                                              │
│                                                                 │
│  Position:  1  2  3  4  5  6  7  8  9  10                       │
│  Token 10:  ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓                       │
│                                                                 │
│  Sliding Window (W=4):                                          │
│  ────────────────────────                                       │
│  Only attend to tokens within window size W                     │
│  Complexity: O(n × W)                                           │
│                                                                 │
│  Position:  1  2  3  4  5  6  7  8  9  10                       │
│  Token 10:  ✗  ✗  ✗  ✗  ✗  ✗  ✓  ✓  ✓  ✓                       │
│                         ↑     └───────┬───────┘                 │
│                    Window start       Window (W=4)              │
│                                                                 │
│  Layer Stacking Effect:                                         │
│  ────────────────────────                                       │
│  L layers → Effective receptive field = L × W                   │
│  32 layers × 4096 window = 131,072 token range!                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Implementation

```python
import torch
import torch.nn.functional as F
import math

def sliding_window_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    window_size: int = 4096,
    causal: bool = True,
):
    """
    Sliding Window Attention Implementation

    Args:
        query: (batch, n_heads, seq_len, head_dim)
        key: (batch, n_heads, seq_len, head_dim)
        value: (batch, n_heads, seq_len, head_dim)
        window_size: Window size
        causal: Whether to apply causal masking
    """
    batch, n_heads, seq_len, head_dim = query.shape
    scale = 1.0 / math.sqrt(head_dim)

    # Attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale

    # Create sliding window mask
    # Each position i attends from max(0, i-W+1) to i
    row_idx = torch.arange(seq_len).unsqueeze(1)  # (seq, 1)
    col_idx = torch.arange(seq_len).unsqueeze(0)  # (1, seq)

    # Causal: col <= row
    # Window: col >= row - window_size + 1
    if causal:
        mask = (col_idx <= row_idx) & (col_idx >= row_idx - window_size + 1)
    else:
        mask = torch.abs(row_idx - col_idx) < window_size

    # Apply mask
    mask = mask.to(scores.device)
    scores = scores.masked_fill(~mask, float('-inf'))

    # Softmax & output
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, value)

    return output

# Memory comparison
def compare_attention_memory(seq_len, window_size=4096):
    """Full vs Sliding Window memory comparison"""
    full_attention_mem = seq_len * seq_len  # O(n²)
    sliding_window_mem = seq_len * window_size  # O(n × W)

    print(f"Sequence length: {seq_len:,}")
    print(f"Full Attention: {full_attention_mem:,} elements")
    print(f"Sliding Window: {sliding_window_mem:,} elements")
    print(f"Memory savings: {(1 - sliding_window_mem/full_attention_mem)*100:.1f}%")

compare_attention_memory(32768, 4096)
# Sequence length: 32,768
# Full Attention: 1,073,741,824 elements
# Sliding Window: 134,217,728 elements
# Memory savings: 87.5%
```

### 2.3 Rolling Buffer KV Cache

```python
"""
Rolling Buffer: Process long sequences with fixed-size KV cache

Normal KV Cache:
- Store KV for all tokens
- Memory: O(seq_len)

Rolling Buffer:
- Store only window_size tokens
- Overwrite old KV
- Memory: O(window_size) = constant!

Example (window=4):
Step 1: [K1, K2, K3, K4]
Step 2: [K5, K2, K3, K4]  ← K5 stored at K1 position
Step 3: [K5, K6, K3, K4]  ← K6 stored at K2 position
...

Advantages:
- Can process infinite sequences (fixed memory)
- Constant inference speed

Disadvantages:
- Loses old information
- Compensated by layer stacking
"""

class RollingKVCache:
    def __init__(self, window_size: int, n_layers: int, n_kv_heads: int, head_dim: int):
        self.window_size = window_size
        self.cache_k = torch.zeros(n_layers, 1, window_size, n_kv_heads, head_dim)
        self.cache_v = torch.zeros(n_layers, 1, window_size, n_kv_heads, head_dim)
        self.pos = 0

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        """Add new KV to cache (circular buffer)"""
        seq_len = k.shape[1]
        for i in range(seq_len):
            idx = (self.pos + i) % self.window_size
            self.cache_k[layer_idx, :, idx] = k[:, i]
            self.cache_v[layer_idx, :, idx] = v[:, i]
        self.pos = (self.pos + seq_len) % self.window_size

    def get(self, layer_idx: int):
        return self.cache_k[layer_idx], self.cache_v[layer_idx]
```

---

## 3. Mixture of Experts (MoE) Basics

### 3.1 MoE Concept

**Mixture of Experts** is an architecture that improves efficiency by activating only some "expert" networks among many.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Mixture of Experts Concept                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Dense Model:                                                   │
│  ─────────────────                                              │
│  Input ──► [FFN (entire)] ──► Output                            │
│  • All parameters activated every time                          │
│  • Computation = proportional to parameter count                │
│                                                                 │
│  Sparse MoE Model:                                              │
│  ─────────────────                                              │
│                        ┌──► Expert 1 ──┐                        │
│                        │               │                        │
│  Input ──► Router ─────┼──► Expert 2 ──┼──► Combine ──► Output  │
│              ↓         │               │                        │
│         (Top-K select) └──► Expert 3 ──┘                        │
│                        └──► Expert N (inactive)                 │
│                                                                 │
│  • Router selects only K experts                                │
│  • Many parameters, little computation                          │
│  • Example: 8 experts, 2 activated → 1/4 computation            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Router (Gating Network)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TopKRouter(nn.Module):
    """
    Top-K Router: Select K experts for each input

    Formula:
    G(x) = softmax(TopK(x · W_g))

    Where TopK keeps only top K values, rest set to -inf
    """
    def __init__(self, dim: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.gate = nn.Linear(dim, num_experts, bias=False)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, dim)

        Returns:
            router_probs: (batch, seq_len, top_k) - Selected expert weights
            expert_indices: (batch, seq_len, top_k) - Selected expert indices
        """
        # Compute router logits
        logits = self.gate(x)  # (batch, seq_len, num_experts)

        # Top-K selection
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)

        # Softmax (among selected experts)
        router_probs = F.softmax(top_k_logits, dim=-1)

        return router_probs, top_k_indices

# Example
router = TopKRouter(dim=4096, num_experts=8, top_k=2)
x = torch.randn(2, 10, 4096)  # batch=2, seq=10
probs, indices = router(x)
print(f"Router probs shape: {probs.shape}")    # (2, 10, 2)
print(f"Expert indices shape: {indices.shape}")  # (2, 10, 2)
print(f"Selected experts for token 0: {indices[0, 0]}")  # e.g., tensor([3, 7])
```

### 3.3 Expert Layer

```python
class MoELayer(nn.Module):
    """
    Mixture of Experts Layer

    Each token is routed to Top-K experts for processing
    """
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # Router
        self.router = TopKRouter(dim, num_experts, top_k)

        # Experts (each is an independent FFN)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim, bias=False),
                nn.SiLU(),
                nn.Linear(hidden_dim, dim, bias=False)
            )
            for _ in range(num_experts)
        ])

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, dim)

        Returns:
            output: (batch, seq_len, dim)
        """
        batch, seq_len, dim = x.shape

        # Routing
        router_probs, expert_indices = self.router(x)
        # router_probs: (batch, seq_len, top_k)
        # expert_indices: (batch, seq_len, top_k)

        # Initialize output
        output = torch.zeros_like(x)

        # Process by each expert (simple implementation, actually more optimized)
        for k in range(self.top_k):
            expert_idx = expert_indices[:, :, k]  # (batch, seq_len)
            expert_prob = router_probs[:, :, k:k+1]  # (batch, seq_len, 1)

            for e in range(self.num_experts):
                # Find positions where this expert is selected
                mask = (expert_idx == e)
                if mask.any():
                    # Extract relevant tokens
                    selected = x[mask]  # (num_selected, dim)
                    # Apply expert
                    expert_output = self.experts[e](selected)
                    # Add weighted result to output
                    output[mask] += expert_prob[mask].squeeze(-1).unsqueeze(-1) * expert_output

        return output

# Usage example
moe = MoELayer(dim=4096, hidden_dim=14336, num_experts=8, top_k=2)
x = torch.randn(2, 10, 4096)
output = moe(x)
print(f"Output shape: {output.shape}")  # (2, 10, 4096)
```

---

## 4. Mixtral 8x7B

### 4.1 Architecture

**Mixtral 8x7B** is an MoE model with 8 experts, activating only 2 experts per layer.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Mixtral 8x7B Architecture                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                   Transformer Block                      │    │
│  │  ┌─────────────────────────────────────────────────┐    │    │
│  │  │              Attention (GQA)                     │    │    │
│  │  │  • 32 query heads, 8 KV heads                   │    │    │
│  │  │  • Sliding Window (4096)                        │    │    │
│  │  └─────────────────────────────────────────────────┘    │    │
│  │                        │                                │    │
│  │                        ▼                                │    │
│  │  ┌─────────────────────────────────────────────────┐    │    │
│  │  │         Sparse MoE FFN Layer                    │    │    │
│  │  │  ┌─────────────────────────────────────────┐    │    │    │
│  │  │  │              Router                      │    │    │    │
│  │  │  │         (Select Top-2)                   │    │    │    │
│  │  │  └────┬────┬────┬────┬────┬────┬────┬────┬─┘    │    │    │
│  │  │       │    │    │    │    │    │    │    │      │    │    │
│  │  │       ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼      │    │    │
│  │  │     [E1] [E2] [E3] [E4] [E5] [E6] [E7] [E8]     │    │    │
│  │  │      ✓         ✓                               │    │    │
│  │  │   Selected  Selected  Inactive  Inactive ...   │    │    │
│  │  └─────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  Total Parameters: ~46.7B (8 experts × 7B FFN params)           │
│  Active Parameters: ~12.9B (2/8 experts)                        │
│  Inference Speed: Similar to 12.9B dense model                  │
│  Performance: 70B dense model level                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Mixtral Specifications

```python
MIXTRAL_CONFIG = {
    "dim": 4096,
    "n_layers": 32,
    "n_heads": 32,
    "n_kv_heads": 8,
    "head_dim": 128,
    "hidden_dim": 14336,
    "vocab_size": 32000,

    # MoE settings
    "num_experts": 8,
    "num_experts_per_tok": 2,  # Top-K

    # Attention
    "sliding_window": 4096,
    "context_length": 32768,

    # Parameter calculation
    # Attention: 4 × dim² × n_layers = 4 × 4096² × 32 ≈ 2.1B
    # MoE FFN: 8 × 3 × dim × hidden × n_layers = 8 × 3 × 4096 × 14336 × 32 ≈ 44.6B
    # Total: ~46.7B
    # Active: ~12.9B (attention + 2/8 FFN)
}
```

### 4.3 Load Balancing Loss

One key challenge of MoE is the **expert imbalance** problem.

```python
def load_balancing_loss(router_probs, expert_indices, num_experts):
    """
    Load Balancing Loss: Encourage balanced expert usage

    Problem: Some experts overused (winner-take-all)
    Solution: Auxiliary loss to encourage balanced routing

    Formula:
    L_balance = α × Σ_e (f_e × P_e)

    f_e = Fraction of tokens assigned to expert e
    P_e = Average routing probability assigned to expert e
    α = Scaling coefficient (e.g., 0.01)
    """
    batch, seq_len, top_k = router_probs.shape
    num_tokens = batch * seq_len

    # f_e: Fraction selected for each expert
    expert_counts = torch.zeros(num_experts, device=router_probs.device)
    for e in range(num_experts):
        expert_counts[e] = (expert_indices == e).float().sum() / (num_tokens * top_k)

    # P_e: Average probability assigned to each expert
    expert_probs = torch.zeros(num_experts, device=router_probs.device)
    # (Simplified calculation - actually computed from gate logits)

    # Balance loss
    loss = (expert_counts * expert_probs).sum() * num_experts

    return loss

# During training
"""
total_loss = language_model_loss + alpha * load_balancing_loss
"""
```

---

## 5. Pros and Cons of MoE

### 5.1 Advantages

```
┌─────────────────────────────────────────────────────────────────┐
│                    Advantages of MoE                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Parameter Efficiency                                        │
│     • Many parameters, little computation                       │
│     • Mixtral 8x7B: 46.7B params, 12.9B active                  │
│     • Dense 70B performance, 13B speed                          │
│                                                                 │
│  2. Specialization                                              │
│     • Each expert learns different patterns/domains             │
│     • Example: Expert 1=math, Expert 2=code, Expert 3=language  │
│     • Can encode deeper specialized knowledge                   │
│                                                                 │
│  3. Scaling                                                     │
│     • Easy to expand model by adding experts                    │
│     • Increase capacity with minimal computation increase       │
│     • Google Switch Transformer: 1.6T params!                   │
│                                                                 │
│  4. Training Efficiency                                         │
│     • Can train larger models with same computation             │
│     • Advantageous from Scaling Law perspective                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Disadvantages

```
┌─────────────────────────────────────────────────────────────────┐
│                    Disadvantages of MoE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Memory Requirements                                         │
│     • Must load all experts into memory                         │
│     • Mixtral 8x7B: 46.7B params ≈ 93GB (FP16)                  │
│     • Requires large GPU memory for inference                   │
│                                                                 │
│  2. Training Instability                                        │
│     • Difficult to train router                                 │
│     • Expert imbalance (only some used)                         │
│     • Auxiliary loss tuning needed                              │
│                                                                 │
│  3. Distributed Training Complexity                             │
│     • Expert parallelism required                               │
│     • Communication overhead                                    │
│     • Load balancing difficult                                  │
│                                                                 │
│  4. Fine-tuning Challenges                                      │
│     • Need to adapt while maintaining expert specialization     │
│     • Fine-tune only some experts?                              │
│     • Research ongoing                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Mistral/Mixtral Practice

### 6.1 Using Mistral 7B

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load Mistral 7B
model_name = "mistralai/Mistral-7B-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Text generation
prompt = "[INST] Explain the concept of machine learning in simple terms. [/INST]"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.7,
    do_sample=True,
    top_p=0.9,
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### 6.2 Using Mixtral 8x7B

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Mixtral 8x7B (requires lots of memory!)
model_name = "mistralai/Mixtral-8x7B-v0.1"

# 4-bit quantization for memory savings
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# Usage
prompt = "[INST] Write a Python function to calculate fibonacci numbers. [/INST]"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=300)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 6.3 Efficient Serving with vLLM

```python
from vllm import LLM, SamplingParams

# vLLM efficiently serves MoE models
llm = LLM(
    model="mistralai/Mixtral-8x7B-v0.1",
    tensor_parallel_size=2,  # 2 GPUs
    dtype="float16",
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=200,
)

prompts = [
    "[INST] What is machine learning? [/INST]",
    "[INST] Explain quantum computing. [/INST]",
]

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Response: {output.outputs[0].text}")
    print("-" * 50)
```

---

## 7. MoE Variants

### 7.1 Major MoE Models

| Model | Organization | Experts | Top-K | Total Params | Active Params |
|-------|--------------|---------|-------|--------------|---------------|
| Switch Transformer | Google | 2048 | 1 | 1.6T | <1B |
| GLaM | Google | 64 | 2 | 1.2T | ~100B |
| Mixtral 8x7B | Mistral | 8 | 2 | 46.7B | 12.9B |
| Mixtral 8x22B | Mistral | 8 | 2 | 141B | 39B |
| DeepSeek MoE | DeepSeek | 64 | 6 | 145B | 22B |

### 7.2 Fine-grained MoE

```python
"""
Fine-grained MoE: More small experts

Traditional (Coarse-grained):
- 8 large experts, Top-2 selection
- Each expert covers broad range

Fine-grained (DeepSeek style):
- 64 small experts, Top-6 selection
- More granular specialization possible
- Increased routing flexibility

Advantages:
- More fine-grained specialization
- Better load balancing
- Scalability

Disadvantages:
- Routing overhead
- Training complexity
"""
```

---

## Summary

### Mistral Core
- **Sliding Window Attention**: O(W) memory for long sequences
- **GQA**: KV cache efficiency
- **Over-training**: Small model, lots of data

### MoE Core
- **Sparse Activation**: Many parameters, little computation
- **Router**: Top-K expert selection
- **Load Balancing**: Maintain expert balance

### Practical Selection Guide
| Situation | Recommended Model |
|-----------|-------------------|
| Single GPU (16GB) | Mistral 7B (4-bit) |
| 2× GPU (48GB) | Mixtral 8x7B (4-bit) |
| Server-grade (8× A100) | Mixtral 8x22B |
| Speed priority | Mistral 7B |
| Performance priority | Mixtral 8x7B+ |

### Next Steps
- [10_Long_Context_Models.md](10_Long_Context_Models.md): Long context processing
- [22_Inference_Optimization.md](22_Inference_Optimization.md): Efficient inference

---

## References

### Core Papers
- Jiang et al. (2023). "Mistral 7B"
- Jiang et al. (2024). "Mixtral of Experts"
- Fedus et al. (2022). "Switch Transformers: Scaling to Trillion Parameter Models"
- Du et al. (2022). "GLaM: Efficient Scaling of Language Models"

### Code & Resources
- [Mistral GitHub](https://github.com/mistralai/mistral-src)
- [HuggingFace Mistral](https://huggingface.co/mistralai)
- [vLLM MoE Support](https://docs.vllm.ai/)
