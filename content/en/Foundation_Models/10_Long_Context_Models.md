# 10. Long Context Models

## Overview

Standard Transformer Self-Attention has limitations in processing long sequences due to O(n²) complexity. This lesson covers various techniques for extending context length.

---

## 1. Importance of Context Length

### 1.1 Why Do We Need Long Context?

```
┌──────────────────────────────────────────────────────────────────┐
│                   Long Context Use Cases                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Document Analysis                                               │
│  - Entire papers (10K-50K tokens)                               │
│  - Legal documents (100K+ tokens)                               │
│  - Full book summarization                                       │
│                                                                  │
│  Code Understanding                                              │
│  - Entire codebase analysis                                     │
│  - Long function/class refactoring                              │
│  - Multi-file debugging                                         │
│                                                                  │
│  Agent Systems                                                   │
│  - Maintain long conversation history                           │
│  - Complex multi-step tasks                                     │
│  - Accumulated tool usage records                               │
│                                                                  │
│  RAG Improvement                                                 │
│  - Include more relevant documents                              │
│  - Provide full documents instead of fragments                  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 1.2 Context Length Comparison by Model

| Model | Context Length | Release Date |
|-------|----------------|--------------|
| GPT-3 | 2,048 | 2020 |
| GPT-3.5 | 4,096 / 16,384 | 2022-2023 |
| GPT-4 | 8,192 / 32,768 / 128K | 2023-2024 |
| Claude 2 | 100,000 | 2023 |
| Claude 3 | 200,000 | 2024 |
| Gemini 1.5 | 1,000,000 / 2,000,000 | 2024 |
| LLaMA 2 | 4,096 | 2023 |
| LLaMA 3 | 8,192 / 128K | 2024 |

---

## 2. Efficient Attention Mechanisms

### 2.1 Sparse Attention

```
┌─────────────────────────────────────────────────────────────┐
│                    Sparse Attention Patterns                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Local Attention        Global Attention                   │
│  ■ ■ ■ □ □ □ □         ■ □ □ □ □ □ □                      │
│  ■ ■ ■ ■ □ □ □         ■ ■ □ □ □ □ □                      │
│  □ ■ ■ ■ ■ □ □         ■ □ ■ □ □ □ □                      │
│  □ □ ■ ■ ■ ■ □         ■ □ □ ■ □ □ □                      │
│  □ □ □ ■ ■ ■ ■         ■ □ □ □ ■ □ □                      │
│  □ □ □ □ ■ ■ ■         ■ □ □ □ □ ■ □                      │
│  □ □ □ □ □ ■ ■         ■ □ □ □ □ □ ■                      │
│                                                             │
│  Longformer: Local + Global token combination               │
│  BigBird: Local + Global + Random                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Longformer Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LongformerAttention(nn.Module):
    """
    Longformer: Sliding Window + Global Attention

    Complexity: O(n × w) where w = window size
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        window_size: int = 256,
        global_tokens: int = 2  # [CLS], [SEP]
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.window_size = window_size
        self.global_tokens = global_tokens

        # Q, K, V projections
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        # Separate projection for global attention
        self.global_query = nn.Linear(hidden_size, hidden_size)
        self.global_key = nn.Linear(hidden_size, hidden_size)
        self.global_value = nn.Linear(hidden_size, hidden_size)

        self.output = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            attention_mask: (batch, seq_len)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Compute Q, K, V
        Q = self.query(hidden_states)
        K = self.key(hidden_states)
        V = self.value(hidden_states)

        # Reshape: (batch, seq_len, num_heads, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # 1. Sliding Window Attention (local)
        local_output = self._sliding_window_attention(Q, K, V)

        # 2. Global Attention (first global_tokens)
        global_output = self._global_attention(
            hidden_states, Q, K, V
        )

        # Combine (use global result for global token positions)
        output = local_output.clone()
        output[:, :self.global_tokens] = global_output[:, :self.global_tokens]

        # Output projection
        output = output.view(batch_size, seq_len, self.hidden_size)
        output = self.output(output)

        return output

    def _sliding_window_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor
    ) -> torch.Tensor:
        """
        Sliding Window Attention

        Each token only attends to tokens within window_size range
        """
        batch_size, seq_len, num_heads, head_dim = Q.shape
        w = self.window_size // 2

        # Add padding
        Q_padded = F.pad(Q, (0, 0, 0, 0, w, w), value=0)
        K_padded = F.pad(K, (0, 0, 0, 0, w, w), value=0)
        V_padded = F.pad(V, (0, 0, 0, 0, w, w), value=0)

        # Extract windows (unfold)
        # Actual implementation is more complex, this is simplified for understanding
        output = torch.zeros_like(Q)

        for i in range(seq_len):
            # Window for i-th token: [i, i + window_size]
            start = i
            end = i + self.window_size

            q_i = Q[:, i:i+1]  # (batch, 1, heads, dim)
            k_window = K_padded[:, start:end]  # (batch, window, heads, dim)
            v_window = V_padded[:, start:end]

            # Attention
            scores = torch.einsum('bihd,bjhd->bijh', q_i, k_window)
            scores = scores / math.sqrt(head_dim)
            weights = F.softmax(scores, dim=2)
            out_i = torch.einsum('bijh,bjhd->bihd', weights, v_window)

            output[:, i] = out_i[:, 0]

        return output

    def _global_attention(
        self,
        hidden_states: torch.Tensor,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor
    ) -> torch.Tensor:
        """Global Attention: global tokens attend to entire sequence"""
        batch_size, seq_len, _ = hidden_states.shape

        # Extract only global tokens
        global_hidden = hidden_states[:, :self.global_tokens]

        # Global Q, K, V
        global_Q = self.global_query(global_hidden)
        global_K = self.global_key(hidden_states)
        global_V = self.global_value(hidden_states)

        # Attention over entire sequence
        global_Q = global_Q.view(batch_size, self.global_tokens,
                                  self.num_heads, self.head_dim)
        global_K = global_K.view(batch_size, seq_len,
                                  self.num_heads, self.head_dim)
        global_V = global_V.view(batch_size, seq_len,
                                  self.num_heads, self.head_dim)

        # (batch, global, heads, seq) attention
        scores = torch.einsum('bghd,bshd->bghs', global_Q, global_K)
        scores = scores / math.sqrt(self.head_dim)
        weights = F.softmax(scores, dim=-1)

        # Output: (batch, global, heads, dim)
        output = torch.einsum('bghs,bshd->bghd', weights, global_V)

        return output
```

### 2.3 Flash Attention

```python
# Flash Attention is implemented as a CUDA kernel
# Here we only explain the concept

"""
Flash Attention Key Ideas:

1. Tiling:
   - Divide Q, K, V into blocks that fit in SRAM
   - Minimize HBM ↔ SRAM data transfer

2. Recomputation:
   - Don't store attention weights in forward pass
   - Recompute when needed in backward pass
   - Memory savings (O(n) vs O(n²))

3. Results:
   - Memory: O(n) vs O(n²)
   - Speed: 2-4x faster
   - Accuracy: Numerically identical
"""

# Using with PyTorch 2.0+
def use_flash_attention():
    import torch.nn.functional as F

    # Scaled Dot-Product Attention (automatically uses Flash Attention)
    Q = torch.randn(2, 8, 1024, 64, device='cuda')
    K = torch.randn(2, 8, 1024, 64, device='cuda')
    V = torch.randn(2, 8, 1024, 64, device='cuda')

    # PyTorch 2.0+ SDPA
    with torch.backends.cuda.sdp_kernel(
        enable_flash=True,
        enable_math=False,
        enable_mem_efficient=False
    ):
        output = F.scaled_dot_product_attention(Q, K, V)

    return output


# Using xFormers
def use_xformers():
    from xformers.ops import memory_efficient_attention

    Q = torch.randn(2, 1024, 8, 64, device='cuda')
    K = torch.randn(2, 1024, 8, 64, device='cuda')
    V = torch.randn(2, 1024, 8, 64, device='cuda')

    output = memory_efficient_attention(Q, K, V)
    return output
```

---

## 3. Position Encoding Extension

### 3.1 Problem: Extrapolation Beyond Training Length

```
Training: 4096 tokens
Inference: 8192+ tokens

Problem:
- Absolute position encoding: Positions after 4096 not learned
- RoPE: Requires interpolation/extrapolation
```

### 3.2 Position Interpolation (PI)

```python
def linear_position_interpolation(
    position_ids: torch.Tensor,
    original_max_length: int,
    extended_max_length: int
) -> torch.Tensor:
    """
    Linear Position Interpolation

    Idea: Scale new positions to original range

    Compress position_ids to [0, original_max_length)
    """
    scale = original_max_length / extended_max_length
    return position_ids.float() * scale


class RoPEWithInterpolation(nn.Module):
    """RoPE with Position Interpolation applied"""

    def __init__(
        self,
        dim: int,
        original_max_length: int = 4096,
        extended_max_length: int = 16384,
        base: float = 10000.0
    ):
        super().__init__()
        self.dim = dim
        self.original_max_length = original_max_length
        self.extended_max_length = extended_max_length
        self.base = base

        # Frequency calculation
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # Scale factor
        self.scale = original_max_length / extended_max_length

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, heads, dim)
            position_ids: (batch, seq_len)
        """
        # Position interpolation
        scaled_positions = position_ids.float() * self.scale

        # Frequency calculation
        freqs = torch.einsum('bi,d->bid', scaled_positions, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)

        cos = emb.cos().unsqueeze(2)  # (batch, seq, 1, dim)
        sin = emb.sin().unsqueeze(2)

        # Apply RoPE
        x_rope = self._apply_rope(x, cos, sin)

        return x_rope

    def _apply_rope(self, x, cos, sin):
        """Apply RoPE"""
        x1 = x[..., : self.dim // 2]
        x2 = x[..., self.dim // 2 :]

        rotated = torch.cat([-x2, x1], dim=-1)
        return x * cos + rotated * sin
```

### 3.3 YaRN (Yet another RoPE extension method)

```python
class YaRNRoPE(nn.Module):
    """
    YaRN: NTK-aware Interpolation

    Problem with Position Interpolation:
    - High frequency information loss (higher dimensions)

    YaRN Solution:
    - Low frequency: Interpolation
    - High frequency: Extrapolation
    - Adjust frequencies with NTK scaling
    """

    def __init__(
        self,
        dim: int,
        original_max_length: int = 4096,
        extended_max_length: int = 32768,
        base: float = 10000.0,
        beta_fast: float = 32,
        beta_slow: float = 1,
    ):
        super().__init__()
        self.dim = dim
        self.original_max_length = original_max_length
        self.extended_max_length = extended_max_length

        scale = extended_max_length / original_max_length

        # Calculate interpolation ratio per dimension
        # Low frequency (lower dimensions): More interpolation
        # High frequency (higher dimensions): Less interpolation (closer to extrapolation)
        dims = torch.arange(0, dim, 2)
        low = max(0, math.floor(dim * math.log(scale) / (2 * math.log(original_max_length))))
        high = min(dim // 2 - 1, math.ceil(dim * math.log(scale) / (2 * math.log(original_max_length))))

        # Ramp function to determine interpolation/extrapolation ratio
        ramp = torch.zeros(dim // 2)
        ramp[:low] = 0.0  # Extrapolation
        ramp[high:] = 1.0  # Interpolation

        if high > low:
            ramp[low:high] = (dims[low:high] - low) / (high - low)

        # NTK-aware base adjustment
        inv_freq = 1.0 / (base ** (dims.float() / dim))

        # Mix of interpolation and extrapolation
        inv_freq_inter = inv_freq / scale
        self.register_buffer(
            'inv_freq',
            (1 - ramp) * inv_freq + ramp * inv_freq_inter
        )

        # Attention scaling
        self.mscale = 0.1 * math.log(scale) + 1.0

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        # Frequency calculation (using already adjusted inv_freq)
        freqs = torch.einsum('bi,d->bid', position_ids.float(), self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)

        cos = emb.cos().unsqueeze(2) * self.mscale
        sin = emb.sin().unsqueeze(2) * self.mscale

        return self._apply_rope(x, cos, sin)
```

---

## 4. ALiBi (Attention with Linear Biases)

### 4.1 Concept

```
ALiBi: Position encoding without training

Idea:
- Don't use position encoding
- Instead, add distance-based penalty to attention scores
- Tokens further away get lower attention scores

Attention score modification:
score(q_i, k_j) = q_i · k_j - m × |i - j|

m: Slope per head (fixed, not learned)
m_h = 2^(-8/H) for head h (H = total number of heads)
```

### 4.2 Implementation

```python
class ALiBiAttention(nn.Module):
    """ALiBi: Attention with Linear Biases"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        max_seq_len: int = 8192
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)

        # ALiBi slopes: Decrease exponentially
        # 2^(-8/n), 2^(-8*2/n), ..., 2^(-8)
        slopes = self._get_alibi_slopes(num_heads)
        self.register_buffer('slopes', slopes)

        # Precompute distance matrix
        positions = torch.arange(max_seq_len)
        distance_matrix = positions.unsqueeze(0) - positions.unsqueeze(1)
        distance_matrix = distance_matrix.abs()
        self.register_buffer('distance_matrix', distance_matrix)

    def _get_alibi_slopes(self, num_heads: int) -> torch.Tensor:
        """Calculate ALiBi slope per head"""

        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(num_heads).is_integer():
            slopes = get_slopes_power_of_2(num_heads)
        else:
            # Interpolate to nearest power of 2
            closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
            slopes = get_slopes_power_of_2(closest_power_of_2)

            extra_slopes = get_slopes_power_of_2(2 * closest_power_of_2)
            extra_slopes = extra_slopes[0::2][:num_heads - closest_power_of_2]
            slopes = slopes + extra_slopes

        return torch.tensor(slopes).view(1, num_heads, 1, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # Compute Q, K, V
        Q = self.query(hidden_states)
        K = self.key(hidden_states)
        V = self.value(hidden_states)

        # Reshape
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose: (batch, heads, seq, dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # ALiBi bias: -m × |i - j|
        alibi_bias = -self.slopes * self.distance_matrix[:seq_len, :seq_len]
        scores = scores + alibi_bias

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=scores.device) * float('-inf'),
            diagonal=1
        )
        scores = scores + causal_mask

        # Attention weights
        weights = F.softmax(scores, dim=-1)

        # Output
        output = torch.matmul(weights, V)
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.hidden_size)
        output = self.output(output)

        return output
```

---

## 5. Ring Attention

### 5.1 Concept

```
Ring Attention: Distributed Long Context

Idea:
- Distribute sequence across multiple GPUs
- Each GPU processes its chunk + circulating KV
- Overlap communication and computation

┌────────────────────────────────────────────────┐
│                Ring Attention                   │
├────────────────────────────────────────────────┤
│                                                │
│  GPU 0: Q[0:n/4]     GPU 1: Q[n/4:n/2]        │
│          ↓ KV circulates   ↓ KV circulates    │
│  Step 1: K[0:n/4]    Step 1: K[n/4:n/2]       │
│  Step 2: K[n/4:n/2]  Step 2: K[n/2:3n/4]      │
│  Step 3: K[n/2:3n/4] Step 3: K[3n/4:n]        │
│  Step 4: K[3n/4:n]   Step 4: K[0:n/4]         │
│                                                │
│  KV circulates like a ring, combining with    │
│  each GPU's Q                                  │
│                                                │
└────────────────────────────────────────────────┘
```

### 5.2 Implementation Overview

```python
import torch.distributed as dist

def ring_attention_forward(
    Q: torch.Tensor,  # Local Q chunk
    K: torch.Tensor,  # Local K chunk
    V: torch.Tensor,  # Local V chunk
    world_size: int,
    rank: int
):
    """
    Ring Attention Forward Pass (conceptual implementation)

    Actual implementation requires CUDA kernels and complex synchronization
    """
    local_seq_len = Q.shape[1]

    # Accumulated attention output
    output = torch.zeros_like(Q)
    max_scores = torch.full((Q.shape[0], Q.shape[2], local_seq_len), float('-inf'))
    sum_exp = torch.zeros_like(max_scores)

    # Current KV
    current_K = K.clone()
    current_V = V.clone()

    for step in range(world_size):
        # Compute attention for this chunk's KV
        scores = torch.matmul(Q, current_K.transpose(-2, -1))
        scores = scores / math.sqrt(Q.shape[-1])

        # Online softmax (numerically stable)
        new_max = torch.max(scores.max(dim=-1).values, max_scores)
        exp_scores = torch.exp(scores - new_max.unsqueeze(-1))

        # Scale previous results
        scale = torch.exp(max_scores - new_max)
        output = output * scale.unsqueeze(-1) + torch.matmul(exp_scores, current_V)

        sum_exp = sum_exp * scale + exp_scores.sum(dim=-1)
        max_scores = new_max

        # Send KV to next GPU (ring)
        if step < world_size - 1:
            # Async send/recv
            send_rank = (rank + 1) % world_size
            recv_rank = (rank - 1) % world_size

            # Receive KV from next GPU
            current_K = ring_pass(current_K, send_rank, recv_rank)
            current_V = ring_pass(current_V, send_rank, recv_rank)

    # Final normalization
    output = output / sum_exp.unsqueeze(-1)

    return output


def ring_pass(tensor, send_rank, recv_rank):
    """Pass tensor in ring topology"""
    recv_tensor = torch.empty_like(tensor)

    send_op = dist.isend(tensor, send_rank)
    recv_op = dist.irecv(recv_tensor, recv_rank)

    send_op.wait()
    recv_op.wait()

    return recv_tensor
```

---

## 6. Practical Guide

### 6.1 Choosing Context Extension Method

```
┌──────────────────────────────────────────────────────────────┐
│              When to Use Which Method?                       │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  4K → 8K:                                                    │
│  - Position Interpolation (simple, good performance)         │
│  - Some fine-tuning recommended                              │
│                                                              │
│  4K → 32K:                                                   │
│  - YaRN (better than PI)                                    │
│  - Or ALiBi (if training from scratch)                      │
│                                                              │
│  32K → 100K+:                                                │
│  - Flash Attention essential                                 │
│  - Ring Attention (multi-GPU)                               │
│  - Consider Sparse Attention                                 │
│                                                              │
│  1M+:                                                        │
│  - Special architectures needed                              │
│  - Mamba/State Space Models                                  │
│  - Or extremely sparse attention                             │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 6.2 Practical Tips

```python
# 1. Gradient Checkpointing is essential
model.gradient_checkpointing_enable()

# 2. Use Mixed Precision
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    outputs = model(**inputs)

# 3. KV Cache optimization (during inference)
# - Sliding Window Cache
# - Paged Attention (vLLM)

# 4. Chunk-based processing (long documents)
def process_long_document(model, document, chunk_size=4096, overlap=512):
    """Process long document in chunks"""
    tokens = tokenizer.encode(document)
    results = []

    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = tokens[i:i + chunk_size]
        output = model.generate(chunk)
        results.append(output)

    return merge_results(results)
```

---

## References

### Papers
- Beltagy et al. (2020). "Longformer: The Long-Document Transformer"
- Dao et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention"
- Press et al. (2021). "Train Short, Test Long: Attention with Linear Biases"
- Peng et al. (2023). "YaRN: Efficient Context Window Extension of Large Language Models"

### Related Lessons
- [08_LLaMA_Family.md](08_LLaMA_Family.md) - RoPE Basics
- [09_Mistral_MoE.md](09_Mistral_MoE.md) - Sliding Window Attention
