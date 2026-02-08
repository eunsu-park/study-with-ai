# 17. Mathematics of Attention and Transformers

## Learning Objectives

- Understand the mathematical properties of the softmax function and its interpretation as a differentiable argmax
- Explain the mathematical principles of scaled dot-product attention and the necessity of scaling
- Understand the subspace projection interpretation of multi-head attention and its parameter efficiency
- Understand the mathematical foundations of positional encoding (sinusoidal, RoPE, ALiBi)
- Analyze the computational complexity of attention and understand efficient implementation methods
- Explain mathematical differences in major Transformer applications (BERT, GPT, cross-attention)

---

## 1. Mathematics of Softmax Function

### 1.1 Definition and Basic Properties

**Softmax function**:

$$\text{softmax}(\mathbf{z})_i = \frac{\exp(z_i)}{\sum_{j=1}^{n} \exp(z_j)}$$

**Properties**:
- **Probability distribution**: $\sum_i \text{softmax}(\mathbf{z})_i = 1$, all elements $\geq 0$
- **Order preservation**: $z_i > z_j \Rightarrow \text{softmax}(\mathbf{z})_i > \text{softmax}(\mathbf{z})_j$
- **Translation invariance**: $\text{softmax}(\mathbf{z} + c\mathbf{1}) = \text{softmax}(\mathbf{z})$

```python
import numpy as np
import matplotlib.pyplot as plt

def softmax(z):
    """
    소프트맥스 함수 (수치 안정성 포함)

    Parameters:
    -----------
    z : ndarray
        입력 벡터

    Returns:
    --------
    probs : ndarray
        확률 분포
    """
    # 수치 안정성: 최댓값 빼기
    z_shifted = z - np.max(z)
    exp_z = np.exp(z_shifted)
    probs = exp_z / np.sum(exp_z)
    return probs

# 예제
z = np.array([1.0, 2.0, 3.0, 4.0, 1.0])
probs = softmax(z)

print("입력 z:", z)
print("소프트맥스(z):", probs)
print("합:", np.sum(probs))
```

### 1.2 Interpretation as Smooth Argmax

**Hard max**: $\text{argmax}_i z_i$ (not differentiable)

**Soft max**: $\sum_i i \cdot \text{softmax}(\mathbf{z})_i$ (differentiable expectation)

It assigns high probability to large $z_i$ values but also considers other candidates.

```python
def visualize_softmax_vs_argmax():
    """소프트맥스와 argmax 비교"""
    z = np.linspace(-3, 3, 100)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1D 케이스: 두 개 원소
    for i, z1 in enumerate([-2, 0, 2]):
        z_vec = np.array([z1, 0])
        soft_probs = softmax(z_vec)

        axes[0].bar([i*3, i*3+1], soft_probs, width=0.8,
                    label=f'z=[{z1}, 0]')

    axes[0].set_title('Softmax probabilities')
    axes[0].set_ylabel('Probability')
    axes[0].legend()
    axes[0].grid(True, axis='y')

    # 온도 효과
    temperatures = [0.1, 1.0, 10.0]
    z_vec = np.array([1.0, 2.0, 3.0, 4.0])

    for tau in temperatures:
        soft_probs = softmax(z_vec / tau)
        axes[1].plot(soft_probs, 'o-', label=f'τ={tau}')

    axes[1].set_title('Temperature effect')
    axes[1].set_xlabel('Index')
    axes[1].set_ylabel('Probability')
    axes[1].legend()
    axes[1].grid(True)

    # 분포 변화
    max_val = 3.0
    z_other = np.linspace(-2, 2, 100)

    for temp in [0.5, 1.0, 2.0]:
        probs = []
        for z_o in z_other:
            z_vec = np.array([max_val, z_o])
            p = softmax(z_vec / temp)
            probs.append(p[0])  # 최댓값의 확률

        axes[2].plot(z_other, probs, label=f'τ={temp}')

    axes[2].set_title('Probability of max value')
    axes[2].set_xlabel('Other value')
    axes[2].set_ylabel('P(max)')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig('softmax_properties.png', dpi=150, bbox_inches='tight')
    print("소프트맥스 특성 시각화 저장 완료")

visualize_softmax_vs_argmax()
```

### 1.3 Temperature Parameter $\tau$

**Temperature-scaled softmax**:

$$\text{softmax}(\mathbf{z}/\tau)_i = \frac{\exp(z_i/\tau)}{\sum_j \exp(z_j/\tau)}$$

- **$\tau \to 0$**: One-hot distribution (hard max)
- **$\tau \to \infty$**: Uniform distribution
- **$\tau = 1$**: Standard softmax

**Applications**: Knowledge distillation, Gumbel-Softmax

### 1.4 Jacobian Computation

$$\frac{\partial \text{softmax}(\mathbf{z})_i}{\partial z_j} = \begin{cases}
\text{softmax}(\mathbf{z})_i (1 - \text{softmax}(\mathbf{z})_i) & \text{if } i = j \\
-\text{softmax}(\mathbf{z})_i \cdot \text{softmax}(\mathbf{z})_j & \text{if } i \neq j
\end{cases}$$

Matrix form: $J = \text{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^T$, where $\mathbf{p} = \text{softmax}(\mathbf{z})$

```python
def softmax_jacobian(z):
    """소프트맥스의 야코비안"""
    p = softmax(z)
    n = len(p)
    J = np.diag(p) - np.outer(p, p)
    return J

z = np.array([1.0, 2.0, 3.0])
J = softmax_jacobian(z)
print("소프트맥스 야코비안:")
print(J)
print("\n행의 합 (0이어야 함):", np.sum(J, axis=1))
```

## 2. Scaled Dot-Product Attention

### 2.1 Definition of Attention Mechanism

**Query (Q)**, **Key (K)**, **Value (V)** matrices:
- $Q \in \mathbb{R}^{n \times d_k}$
- $K \in \mathbb{R}^{m \times d_k}$
- $V \in \mathbb{R}^{m \times d_v}$

**Scaled dot-product attention**:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

**Step-by-step interpretation**:
1. **Similarity computation**: $S = QK^T \in \mathbb{R}^{n \times m}$
2. **Scaling**: $S' = S / \sqrt{d_k}$
3. **Softmax**: $A = \text{softmax}(S')$ (row-wise)
4. **Weighted sum**: $\text{Output} = A V$

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    스케일드 닷-프로덕트 어텐션

    Parameters:
    -----------
    Q : ndarray, shape (n, d_k)
        쿼리 행렬
    K : ndarray, shape (m, d_k)
        키 행렬
    V : ndarray, shape (m, d_v)
        값 행렬
    mask : ndarray, shape (n, m), optional
        어텐션 마스크 (0: 차단, 1: 허용)

    Returns:
    --------
    output : ndarray, shape (n, d_v)
        어텐션 출력
    attention_weights : ndarray, shape (n, m)
        어텐션 가중치
    """
    d_k = Q.shape[-1]

    # 유사도 점수
    scores = Q @ K.T / np.sqrt(d_k)

    # 마스킹 (옵션)
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)

    # 소프트맥스
    attention_weights = np.apply_along_axis(softmax, axis=1, arr=scores)

    # 가중 합
    output = attention_weights @ V

    return output, attention_weights

# 예제
np.random.seed(42)
n, m, d_k, d_v = 4, 5, 8, 8

Q = np.random.randn(n, d_k)
K = np.random.randn(m, d_k)
V = np.random.randn(m, d_v)

output, attn_weights = scaled_dot_product_attention(Q, K, V)

print("쿼리 shape:", Q.shape)
print("키 shape:", K.shape)
print("값 shape:", V.shape)
print("\n어텐션 출력 shape:", output.shape)
print("어텐션 가중치 shape:", attn_weights.shape)
print("\n어텐션 가중치 (각 행의 합=1):")
print(attn_weights)
print("행의 합:", np.sum(attn_weights, axis=1))
```

### 2.2 Need for Scaling: Variance Analysis

If each dimension of query and key is independent with $\mathbb{E}[q_i] = \mathbb{E}[k_i] = 0$, $\text{Var}(q_i) = \text{Var}(k_i) = 1$:

$$\text{Var}(q \cdot k) = \text{Var}\left(\sum_{i=1}^{d_k} q_i k_i\right) = d_k$$

**Problem**: When $d_k$ is large, the variance of $q \cdot k$ increases, potentially causing saturation in softmax with extreme values.

**Solution**: Dividing by $\sqrt{d_k}$ gives $\text{Var}(q \cdot k / \sqrt{d_k}) = 1$

```python
def demonstrate_scaling_effect():
    """스케일링 효과 시각화"""
    np.random.seed(42)
    d_k_values = [8, 32, 128, 512]
    n_samples = 1000

    fig, axes = plt.subplots(2, len(d_k_values), figsize=(16, 8))

    for idx, d_k in enumerate(d_k_values):
        # 랜덤 쿼리와 키 생성
        Q = np.random.randn(n_samples, d_k)
        K = np.random.randn(1, d_k)

        # 닷 프로덕트
        scores_unscaled = Q @ K.T
        scores_scaled = scores_unscaled / np.sqrt(d_k)

        # 히스토그램
        axes[0, idx].hist(scores_unscaled.flatten(), bins=50, alpha=0.7)
        axes[0, idx].set_title(f'd_k={d_k}, Unscaled')
        axes[0, idx].set_xlabel('Dot product value')
        axes[0, idx].axvline(0, color='r', linestyle='--')

        axes[1, idx].hist(scores_scaled.flatten(), bins=50, alpha=0.7)
        axes[1, idx].set_title(f'd_k={d_k}, Scaled')
        axes[1, idx].set_xlabel('Scaled value')
        axes[1, idx].axvline(0, color='r', linestyle='--')

        # 분산 출력
        var_unscaled = np.var(scores_unscaled)
        var_scaled = np.var(scores_scaled)
        print(f"d_k={d_k}: Var(unscaled)={var_unscaled:.2f}, Var(scaled)={var_scaled:.2f}")

    plt.tight_layout()
    plt.savefig('scaling_effect.png', dpi=150, bbox_inches='tight')
    print("\n스케일링 효과 시각화 저장 완료")

demonstrate_scaling_effect()
```

### 2.3 Self-Attention

**Self-attention**: $Q, K, V$ all derived from the same input

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

where $X \in \mathbb{R}^{n \times d_{\text{model}}}$ is the input sequence.

**Effect**: Each token interacts with all other tokens

## 3. Multi-Head Attention

### 3.1 Motivation and Definition

**Idea**: Perform attention in parallel across multiple representation subspaces

**Single head**:

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

where:
- $W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$
- $W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$
- $W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$

Typically $d_k = d_v = d_{\text{model}} / h$ (where $h$ is the number of heads)

**Multi-head attention**:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

where $W^O \in \mathbb{R}^{hd_v \times d_{\text{model}}}$

```python
def multi_head_attention(Q, K, V, num_heads, W_Q, W_K, W_V, W_O, mask=None):
    """
    멀티헤드 어텐션

    Parameters:
    -----------
    Q, K, V : ndarray, shape (n, d_model)
        쿼리, 키, 값
    num_heads : int
        헤드 수
    W_Q, W_K, W_V : list of ndarray
        각 헤드의 투영 행렬
    W_O : ndarray, shape (h*d_v, d_model)
        출력 투영 행렬
    mask : ndarray, optional
        어텐션 마스크

    Returns:
    --------
    output : ndarray, shape (n, d_model)
        멀티헤드 어텐션 출력
    all_attention_weights : list
        각 헤드의 어텐션 가중치
    """
    head_outputs = []
    all_attention_weights = []

    for i in range(num_heads):
        # 각 헤드에 대한 투영
        Q_i = Q @ W_Q[i]
        K_i = K @ W_K[i]
        V_i = V @ W_V[i]

        # 스케일드 닷-프로덕트 어텐션
        head_out, attn_weights = scaled_dot_product_attention(Q_i, K_i, V_i, mask)

        head_outputs.append(head_out)
        all_attention_weights.append(attn_weights)

    # 헤드 연결
    concatenated = np.concatenate(head_outputs, axis=-1)

    # 출력 투영
    output = concatenated @ W_O

    return output, all_attention_weights

# 예제
d_model = 64
num_heads = 8
d_k = d_v = d_model // num_heads  # 8

n = 10
X = np.random.randn(n, d_model)

# 가중치 초기화
W_Q = [np.random.randn(d_model, d_k) * 0.1 for _ in range(num_heads)]
W_K = [np.random.randn(d_model, d_k) * 0.1 for _ in range(num_heads)]
W_V = [np.random.randn(d_model, d_v) * 0.1 for _ in range(num_heads)]
W_O = np.random.randn(num_heads * d_v, d_model) * 0.1

mha_output, attn_weights_list = multi_head_attention(
    X, X, X, num_heads, W_Q, W_K, W_V, W_O
)

print(f"입력 shape: {X.shape}")
print(f"멀티헤드 어텐션 출력 shape: {mha_output.shape}")
print(f"헤드 수: {num_heads}")
print(f"각 헤드의 어텐션 가중치 shape: {attn_weights_list[0].shape}")
```

### 3.2 Meaning of Subspace Projection

Each head operates in a different subspace:
- **Syntactic head**: Captures grammatical structure
- **Semantic head**: Captures semantic relationships
- **Positional head**: Learns relative positions

### 3.3 Parameter Count Analysis

**Single-head attention**:
- $W^Q, W^K, W^V$: $3 \times d_{\text{model}}^2$
- Total: $3d_{\text{model}}^2$

**Multi-head attention** ($h$ heads, $d_k = d_v = d_{\text{model}}/h$):
- Each head's $W_i^Q, W_i^K, W_i^V$: $3h \times d_{\text{model}} \times \frac{d_{\text{model}}}{h} = 3d_{\text{model}}^2$
- $W^O$: $d_{\text{model}}^2$
- Total: $4d_{\text{model}}^2$

**Difference**: Multi-head learns more diverse representations with similar parameter count as single-head

## 4. Positional Encoding

### 4.1 Problem: Absence of Positional Information

Attention mechanism is **permutation-invariant**:
- Changing input order (excluding mask) changes output in the same way
- Positional information is needed for sequences

### 4.2 Sinusoidal Positional Encoding

**Original Transformer paper** (Vaswani et al., 2017):

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

**Frequency**: Different frequency $\omega_i = 1 / 10000^{2i/d_{\text{model}}}$ for each dimension $i$

```python
def sinusoidal_positional_encoding(max_len, d_model):
    """
    사인/코사인 위치 인코딩

    Parameters:
    -----------
    max_len : int
        최대 시퀀스 길이
    d_model : int
        모델 차원

    Returns:
    --------
    PE : ndarray, shape (max_len, d_model)
        위치 인코딩 행렬
    """
    PE = np.zeros((max_len, d_model))

    position = np.arange(max_len)[:, np.newaxis]  # (max_len, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    PE[:, 0::2] = np.sin(position * div_term)
    PE[:, 1::2] = np.cos(position * div_term)

    return PE

# 시각화
max_len = 100
d_model = 128

PE = sinusoidal_positional_encoding(max_len, d_model)

plt.figure(figsize=(12, 6))
plt.imshow(PE.T, aspect='auto', cmap='viridis')
plt.colorbar(label='Value')
plt.xlabel('Position')
plt.ylabel('Dimension')
plt.title('Sinusoidal Positional Encoding')
plt.tight_layout()
plt.savefig('positional_encoding.png', dpi=150, bbox_inches='tight')
print("위치 인코딩 시각화 저장 완료")
```

### 4.3 Mathematical Properties of Positional Encoding

**Linear transformation of relative positions**:

$$PE_{pos+k} = \mathbf{T}_k \cdot PE_{pos}$$

where $\mathbf{T}_k$ is a rotation matrix (for each frequency).

**Proof**: Trigonometric addition formulas

$$\sin(\omega(pos + k)) = \sin(\omega \cdot pos)\cos(\omega k) + \cos(\omega \cdot pos)\sin(\omega k)$$

This can be expressed as a 2D rotation matrix.

### 4.4 RoPE (Rotary Position Embedding)

**Idea**: Encode positional information into queries and keys via rotation

Rotate the 2D subspace $(q_{2i}, q_{2i+1})$ by angle $m\theta_i$ (where $m$ is position):

$$\begin{pmatrix} q_{2i}' \\ q_{2i+1}' \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}$$

**Advantages**:
- Inner product of query and key depends only on relative position
- Better extrapolation to longer sequences

```python
def rotary_position_embedding(q, k, positions, d_model):
    """
    RoPE (간단한 버전)

    Parameters:
    -----------
    q, k : ndarray, shape (n, d_model)
        쿼리와 키
    positions : ndarray, shape (n,)
        각 토큰의 위치
    d_model : int
        모델 차원

    Returns:
    --------
    q_rot, k_rot : ndarray
        회전된 쿼리와 키
    """
    assert d_model % 2 == 0

    # 주파수
    inv_freq = 1.0 / (10000 ** (np.arange(0, d_model, 2) / d_model))

    # 각도
    angles = positions[:, np.newaxis] * inv_freq[np.newaxis, :]  # (n, d_model/2)

    # 사인/코사인
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)

    # 회전 적용
    def apply_rotation(x, cos, sin):
        x_rot = np.zeros_like(x)
        x_rot[:, 0::2] = x[:, 0::2] * cos - x[:, 1::2] * sin
        x_rot[:, 1::2] = x[:, 0::2] * sin + x[:, 1::2] * cos
        return x_rot

    q_rot = apply_rotation(q, cos_angles, sin_angles)
    k_rot = apply_rotation(k, cos_angles, sin_angles)

    return q_rot, k_rot

# 예제
n = 10
d_model = 64
positions = np.arange(n)

q = np.random.randn(n, d_model)
k = np.random.randn(n, d_model)

q_rot, k_rot = rotary_position_embedding(q, k, positions, d_model)

print("원본 쿼리 shape:", q.shape)
print("회전된 쿼리 shape:", q_rot.shape)

# 상대 위치 의존성 확인
attn_original = q @ k.T
attn_rope = q_rot @ k_rot.T

print("\n원본 어텐션 점수 (pos=0 기준):", attn_original[0, :5])
print("RoPE 어텐션 점수 (pos=0 기준):", attn_rope[0, :5])
```

### 4.5 ALiBi (Attention with Linear Biases)

**Idea**: Add distance-proportional biases to attention scores

$$\text{softmax}(q_i K^T / \sqrt{d_k} + m \cdot [0, -1, -2, \ldots, -(i-1)])$$

where $m$ is a head-specific slope.

**Advantages**:
- No additional parameters
- Excellent extrapolation (sequences longer than training length)

## 5. Computational Complexity Analysis

### 5.1 Complexity of Self-Attention

Sequence length $n$, model dimension $d$:

**Time complexity**:
- $QK^T$: $O(n^2 d)$
- Softmax: $O(n^2)$
- $(AV)$: $O(n^2 d)$
- **Total**: $O(n^2 d)$

**Space complexity**: $O(n^2)$ (attention weight matrix)

**Bottleneck**: Long sequences (large $n$)

### 5.2 Flash Attention

**Idea**: Improve memory efficiency through tiling and recomputation

**Key points**:
1. Compute softmax in an online manner (no need to store entire $QK^T$)
2. Load tiles into GPU SRAM (fast memory) for computation
3. Recompute attention weights during backpropagation instead of storing them

**Effects**: Memory $O(n^2) \to O(n)$, speed improvement (reduced I/O)

### 5.3 Linear Attention

**Idea**: Replace $\text{softmax}(QK^T)$ with **kernel approximation**

$$\text{softmax}(q_i^T k_j) \approx \phi(q_i)^T \phi(k_j)$$

**Linear attention**:

$$\text{Attention}(Q, K, V) = \phi(Q) (\phi(K)^T V)$$

**Complexity**: First compute $\phi(K)^T V \in \mathbb{R}^{d_\phi \times d_v}$ ($O(nd_\phi d_v)$), then multiply with $\phi(Q)$ ($O(nd_\phi d_v)$) → **$O(nd)$**

**Example**: $\phi(x) = \text{elu}(x) + 1$ (Performer)

```python
def linear_attention(Q, K, V, phi=lambda x: np.maximum(0, x) + 1):
    """
    선형 어텐션 (간단한 근사)

    Parameters:
    -----------
    Q, K : ndarray, shape (n, d_k)
        쿼리와 키
    V : ndarray, shape (n, d_v)
        값
    phi : function
        특징 맵 함수

    Returns:
    --------
    output : ndarray, shape (n, d_v)
        어텐션 출력
    """
    # 특징 맵 적용
    Q_phi = phi(Q)
    K_phi = phi(K)

    # K^T V 먼저 계산 (n x d_v)
    KV = K_phi.T @ V  # (d_k, d_v)

    # Q와 곱셈
    output = Q_phi @ KV  # (n, d_v)

    # 정규화
    normalizer = Q_phi @ np.sum(K_phi, axis=0, keepdims=True).T
    output = output / (normalizer + 1e-6)

    return output

# 비교
n, d_k, d_v = 1000, 64, 64
Q = np.random.randn(n, d_k)
K = np.random.randn(n, d_k)
V = np.random.randn(n, d_v)

import time

# 표준 어텐션
start = time.time()
output_standard, _ = scaled_dot_product_attention(Q, K, V)
time_standard = time.time() - start

# 선형 어텐션
start = time.time()
output_linear = linear_attention(Q, K, V)
time_linear = time.time() - start

print(f"표준 어텐션 시간: {time_standard:.4f}s")
print(f"선형 어텐션 시간: {time_linear:.4f}s")
print(f"속도 향상: {time_standard / time_linear:.2f}x")
```

### 5.4 KV-Cache: Autoregressive Decoding

**Problem**: During autoregressive generation, recomputing entire sequence at each step

**Solution**: Cache keys and values from previous steps

$$K_{\text{new}} = [K_{\text{cached}}; k_{t+1}]$$
$$V_{\text{new}} = [V_{\text{cached}}; v_{t+1}]$$

**Complexity**: $O(td)$ per step (where $t$ is current length)

**Memory**: $O(t \cdot d \cdot \text{layers})$

## 6. ML Applications: BERT, GPT, Cross-Attention

### 6.1 BERT: Bidirectional Masking

**Masked Language Model** (MLM): Mask some tokens and predict them

**Attention mask**: Allow attention between all positions (bidirectional)

$$\text{Mask}_{ij} = 1 \quad \forall i, j$$

### 6.2 GPT: Causal Masking

**Autoregressive language model**: Predict next token looking only at previous tokens

**Attention mask**: Lower triangular matrix (causal mask)

$$\text{Mask}_{ij} = \begin{cases}
1 & \text{if } j \leq i \\
0 & \text{if } j > i
\end{cases}$$

```python
def create_causal_mask(seq_len):
    """인과적 마스크 생성"""
    mask = np.tril(np.ones((seq_len, seq_len)))
    return mask

seq_len = 5
causal_mask = create_causal_mask(seq_len)
print("인과적 마스크:")
print(causal_mask)

# 어텐션에 적용
Q = np.random.randn(seq_len, 8)
K = np.random.randn(seq_len, 8)
V = np.random.randn(seq_len, 8)

output_causal, attn_causal = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
print("\n인과적 어텐션 가중치:")
print(attn_causal)
```

### 6.3 Cross-Attention

**Encoder-decoder architecture**: Decoder attends to encoder output

- **Query**: Decoder state
- **Key/Value**: Encoder output

$$\text{CrossAttention}(Q_{\text{dec}}, K_{\text{enc}}, V_{\text{enc}})$$

**Applications**: Translation, summarization, image captioning

### 6.4 Mathematics of Mixture of Experts (MoE) Routing

**Idea**: Activate only a subset of multiple expert networks

**Gating network**: Compute probabilities with softmax

$$G(\mathbf{x}) = \text{softmax}(\mathbf{x}^T W_g)$$

**Top-K routing**: Select only top $k$ experts

$$\text{Output} = \sum_{i \in \text{TopK}(G(\mathbf{x}))} G(\mathbf{x})_i \cdot E_i(\mathbf{x})$$

**Mathematical challenge**: Differentiability of discrete selection (using Straight-Through Estimator)

## Practice Problems

### Problem 1: Proof of Softmax Jacobian
Prove that the Jacobian of the softmax function is $J = \text{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^T$. Use this to show that $\sum_j J_{ij} = 0$ (constraint from sum of probabilities).

### Problem 2: Attention Mechanism Implementation
Implement a complete multi-head attention layer using only NumPy. Include:
1. Linear projections ($W^Q, W^K, W^V, W^O$)
2. Scaled dot-product attention
3. Causal mask support
4. Gradient check (compare with numerical differentiation)

### Problem 3: Positional Encoding Analysis
For sinusoidal positional encoding:
1. Visualize how frequencies differ across dimensions
2. Plot inner product $PE_{pos_1}^T PE_{pos_2}$ as a function of position difference
3. Compare with RoPE: quantify how well relative position information is preserved

### Problem 4: Computational Complexity Experiments
Vary sequence length as [100, 500, 1000, 2000, 5000]:
1. Measure execution time of standard attention vs linear attention
2. Estimate memory usage (attention matrix size)
3. Measure output difference using L2 norm between the two methods
4. Plot complexity graphs (compare with theoretical $O(n^2)$ vs $O(n)$ curves)

### Problem 5: Simple Transformer Block
Implement a Transformer encoder block including:
1. Multi-head self-attention
2. Layer Normalization
3. Feed-forward network (two linear layers + ReLU)
4. Residual connections

Apply to a small sequence classification task and train it.

## References

### Papers
- Vaswani, A., et al. (2017). "Attention is All You Need." *NeurIPS*.
- Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." *NAACL*.
- Brown, T., et al. (2020). "Language Models are Few-Shot Learners." *NeurIPS*. [GPT-3]
- Su, J., et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position Embedding." *arXiv:2104.09864*.
- Press, O., et al. (2022). "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation." *ICLR*. [ALiBi]
- Dao, T., et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." *NeurIPS*.

### Online Resources
- [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/)
- [Attention? Attention! (Lilian Weng)](https://lilianweng.github.io/posts/2018-06-24-attention/)
- [Flash Attention Explained](https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad)
- [Transformers from Scratch (Peter Bloem)](https://peterbloem.nl/blog/transformers)

### Libraries
- `torch.nn.MultiheadAttention`: PyTorch implementation
- `transformers` (Hugging Face): Pre-trained models
- `einops`: Simplified tensor operations
