# 15. Graph Theory and Spectral Methods

## Learning Objectives

- Understand and implement mathematical representations of graphs (adjacency matrix, degree matrix, Laplacian)
- Explain eigenvalue decomposition and spectral properties of graph Laplacians
- Understand and implement the mathematical principles of spectral clustering algorithms
- Understand the mathematical foundations of random walks and the PageRank algorithm
- Understand the concepts of graph signal processing and graph Fourier transform
- Understand the mathematical foundations of GNNs (Graph Neural Networks) and message passing mechanisms

---

## 1. Mathematical Representation of Graphs

### 1.1 Graph Basics

A graph $G = (V, E)$ consists of a vertex set $V$ and an edge set $E \subseteq V \times V$.

**Graph Types:**
- **Undirected graph**: $(i,j) \in E \Rightarrow (j,i) \in E$
- **Directed graph**: edges have direction
- **Weighted graph**: each edge is assigned a weight $w_{ij}$

### 1.2 Adjacency Matrix

For a graph with $n$ vertices, the adjacency matrix $A \in \mathbb{R}^{n \times n}$:

$$A_{ij} = \begin{cases}
w_{ij} & \text{if } (i,j) \in E \\
0 & \text{otherwise}
\end{cases}$$

**Properties:**
- Undirected graph: $A = A^T$ (symmetric)
- Binary graph: $A_{ij} \in \{0, 1\}$
- $(i,j)$ element of $A^k$: number of paths of length $k$ from $i$ to $j$

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# 간단한 그래프 생성
def create_sample_graph():
    """
    5개 정점으로 구성된 무방향 그래프
    """
    n = 5
    A = np.array([
        [0, 1, 1, 0, 0],
        [1, 0, 1, 1, 0],
        [1, 1, 0, 1, 1],
        [0, 1, 1, 0, 1],
        [0, 0, 1, 1, 0]
    ])
    return A

A = create_sample_graph()
print("인접 행렬 A:")
print(A)
print(f"\n대칭성 확인: {np.allclose(A, A.T)}")

# 경로 수 계산
A2 = np.linalg.matrix_power(A, 2)
print(f"\n정점 0에서 정점 4로 가는 길이 2인 경로의 수: {A2[0, 4]}")
```

### 1.3 Degree Matrix

The degree of vertex $i$, $d_i = \sum_{j} A_{ij}$, is the number of connected edges.

The degree matrix $D \in \mathbb{R}^{n \times n}$ is a diagonal matrix:

$$D = \text{diag}(d_1, d_2, \ldots, d_n)$$

```python
def compute_degree_matrix(A):
    """차수 행렬 계산"""
    degrees = np.sum(A, axis=1)
    D = np.diag(degrees)
    return D, degrees

D, degrees = compute_degree_matrix(A)
print("차수 벡터:", degrees)
print("\n차수 행렬 D:")
print(D)
```

## 2. Graph Laplacian

### 2.1 Definition of Laplacian

**Unnormalized Laplacian**:

$$L = D - A$$

**Properties:**
- Symmetric: $L = L^T$
- Positive semidefinite: $\mathbf{x}^T L \mathbf{x} \geq 0$
- $L \mathbf{1} = \mathbf{0}$ (vector of all ones is an eigenvector with eigenvalue 0)

### 2.2 Quadratic Form of Laplacian

$$\mathbf{x}^T L \mathbf{x} = \mathbf{x}^T(D - A)\mathbf{x} = \sum_{i} d_i x_i^2 - \sum_{i,j} A_{ij} x_i x_j$$

For undirected graphs:

$$\mathbf{x}^T L \mathbf{x} = \frac{1}{2} \sum_{i,j} A_{ij}(x_i - x_j)^2$$

This is a smoothness measure that **quantifies differences between adjacent vertices**.

```python
def compute_laplacian(A):
    """그래프 라플라시안 계산"""
    D, _ = compute_degree_matrix(A)
    L = D - A
    return L

L = compute_laplacian(A)
print("라플라시안 행렬 L:")
print(L)

# 양반정치 확인 (모든 고유값 >= 0)
eigenvalues = np.linalg.eigvalsh(L)
print(f"\n라플라시안 고유값: {eigenvalues}")
print(f"최소 고유값: {eigenvalues[0]:.10f}")
```

### 2.3 Normalized Laplacian

**Symmetric normalized Laplacian**:

$$L_{\text{sym}} = D^{-1/2} L D^{-1/2} = I - D^{-1/2} A D^{-1/2}$$

**Random walk normalized Laplacian**:

$$L_{\text{rw}} = D^{-1} L = I - D^{-1} A$$

Eigenvalues of the normalized Laplacian lie in the range $[0, 2]$.

```python
def compute_normalized_laplacian(A):
    """정규화 라플라시안 계산"""
    D, degrees = compute_degree_matrix(A)

    # D^{-1/2} 계산
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))

    # L_sym = D^{-1/2} L D^{-1/2}
    L = compute_laplacian(A)
    L_sym = D_inv_sqrt @ L @ D_inv_sqrt

    # L_rw = D^{-1} L
    D_inv = np.diag(1.0 / degrees)
    L_rw = D_inv @ L

    return L_sym, L_rw

L_sym, L_rw = compute_normalized_laplacian(A)
print("정규화 라플라시안 L_sym:")
print(L_sym)

eig_sym = np.linalg.eigvalsh(L_sym)
print(f"\nL_sym 고유값: {eig_sym}")
```

### 2.4 Connected Components and Eigenvalues

**Theorem**: If a graph has $k$ connected components, the multiplicity of eigenvalue 0 of the Laplacian is $k$.

```python
def create_disconnected_graph():
    """두 개의 연결 성분을 가진 그래프"""
    # 성분 1: 정점 0, 1, 2
    # 성분 2: 정점 3, 4
    A = np.array([
        [0, 1, 1, 0, 0],
        [1, 0, 1, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0]
    ])
    return A

A_disconnected = create_disconnected_graph()
L_disconnected = compute_laplacian(A_disconnected)
eigenvalues_disc = np.linalg.eigvalsh(L_disconnected)

print("비연결 그래프의 라플라시안 고유값:")
print(eigenvalues_disc)
print(f"고유값 0의 개수 (연결 성분 수): {np.sum(np.abs(eigenvalues_disc) < 1e-10)}")
```

## 3. Spectral Clustering

### 3.1 Graph Cut Problem

When partitioning a graph into two parts $S$ and $\bar{S}$, the **cut cost** is:

$$\text{cut}(S, \bar{S}) = \sum_{i \in S, j \in \bar{S}} A_{ij}$$

**Normalized Cut**:

$$\text{Ncut}(S, \bar{S}) = \frac{\text{cut}(S, \bar{S})}{\text{vol}(S)} + \frac{\text{cut}(S, \bar{S})}{\text{vol}(\bar{S})}$$

where $\text{vol}(S) = \sum_{i \in S} d_i$ is the volume of subset $S$.

### 3.2 Fiedler Vector and Spectral Methods

The Ncut problem is NP-hard, but can be approximated using a relaxation with the **second smallest eigenvector of the Laplacian** (Fiedler vector).

**Rayleigh quotient**:

$$\min_{\mathbf{y}} \frac{\mathbf{y}^T L \mathbf{y}}{\mathbf{y}^T D \mathbf{y}} \quad \text{s.t. } \mathbf{y}^T D \mathbf{1} = 0$$

The solution is the second smallest eigenvector of the generalized eigenvalue problem $L \mathbf{y} = \lambda D \mathbf{y}$.

```python
def spectral_clustering(A, n_clusters=2):
    """
    스펙트럼 군집화 알고리즘

    Parameters:
    -----------
    A : ndarray
        인접 행렬
    n_clusters : int
        군집 수

    Returns:
    --------
    labels : ndarray
        각 정점의 군집 레이블
    """
    # 정규화 라플라시안 계산
    L_sym, _ = compute_normalized_laplacian(A)

    # 고유값 분해 (최소 n_clusters개의 고유벡터)
    eigenvalues, eigenvectors = eigh(L_sym)

    # 최소 n_clusters개의 고유벡터 선택
    U = eigenvectors[:, :n_clusters]

    # 행 정규화
    U_normalized = U / np.linalg.norm(U, axis=1, keepdims=True)

    # k-means 군집화
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(U_normalized)

    return labels, U

# 스펙트럼 군집화 적용
labels, U = spectral_clustering(A, n_clusters=2)
print("군집 레이블:", labels)
print("\n첫 2개 고유벡터:")
print(U)
```

### 3.3 Intuition Behind Spectral Clustering

- **Sign of Fiedler vector**: a good indicator for partitioning the graph into two parts
- **Magnitude of eigenvalues**: indicates separation between clusters
- **eigengap**: if the gap between $\lambda_k$ and $\lambda_{k+1}$ is large, $k$ clusters is appropriate

```python
def visualize_spectral_clustering():
    """스펙트럼 군집화 시각화"""
    # 더 큰 그래프 생성 (두 개의 밀집된 커뮤니티)
    np.random.seed(42)
    n1, n2 = 15, 15
    n = n1 + n2

    # 블록 행렬 구조
    A_block = np.zeros((n, n))

    # 커뮤니티 1 내부 연결 (밀집)
    for i in range(n1):
        for j in range(i+1, n1):
            if np.random.rand() < 0.6:
                A_block[i, j] = A_block[j, i] = 1

    # 커뮤니티 2 내부 연결 (밀집)
    for i in range(n1, n):
        for j in range(i+1, n):
            if np.random.rand() < 0.6:
                A_block[i, j] = A_block[j, i] = 1

    # 커뮤니티 간 연결 (희소)
    for i in range(n1):
        for j in range(n1, n):
            if np.random.rand() < 0.05:
                A_block[i, j] = A_block[j, i] = 1

    # 스펙트럼 군집화
    labels, U = spectral_clustering(A_block, n_clusters=2)

    # 라플라시안 고유값
    L_sym, _ = compute_normalized_laplacian(A_block)
    eigenvalues = np.linalg.eigvalsh(L_sym)

    # 시각화
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 인접 행렬
    axes[0].imshow(A_block, cmap='binary')
    axes[0].set_title('Adjacency Matrix')
    axes[0].set_xlabel('Node')
    axes[0].set_ylabel('Node')

    # 고유값 스펙트럼
    axes[1].plot(eigenvalues, 'o-')
    axes[1].axvline(x=2, color='r', linestyle='--', label='Eigengap')
    axes[1].set_xlabel('Index')
    axes[1].set_ylabel('Eigenvalue')
    axes[1].set_title('Laplacian Spectrum')
    axes[1].legend()
    axes[1].grid(True)

    # Fiedler 벡터
    axes[2].scatter(range(n), U[:, 1], c=labels, cmap='viridis', s=50)
    axes[2].axhline(y=0, color='r', linestyle='--')
    axes[2].set_xlabel('Node')
    axes[2].set_ylabel('Fiedler Vector Value')
    axes[2].set_title('Fiedler Vector (2nd eigenvector)')
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig('spectral_clustering.png', dpi=150, bbox_inches='tight')
    print("스펙트럼 군집화 시각화 저장 완료")

visualize_spectral_clustering()
```

## 4. Random Walk on Graphs

### 4.1 Transition Probability Matrix

A random walk moves from the current vertex to an adjacent vertex uniformly.

**Transition probability matrix**:

$$P = D^{-1} A$$

$P_{ij}$ is the probability of moving from vertex $i$ to vertex $j$.

```python
def compute_transition_matrix(A):
    """전이 확률 행렬 계산"""
    D, degrees = compute_degree_matrix(A)
    D_inv = np.diag(1.0 / degrees)
    P = D_inv @ A
    return P

P = compute_transition_matrix(A)
print("전이 확률 행렬 P:")
print(P)
print(f"\n각 행의 합 (확률의 합): {np.sum(P, axis=1)}")
```

### 4.2 Stationary Distribution

The stationary distribution $\pi$ satisfies:

$$\pi^T P = \pi^T$$

For a connected undirected graph, the stationary distribution is:

$$\pi_i = \frac{d_i}{\sum_j d_j}$$

```python
def compute_stationary_distribution(A):
    """정상 분포 계산"""
    _, degrees = compute_degree_matrix(A)
    pi = degrees / np.sum(degrees)
    return pi

pi = compute_stationary_distribution(A)
print("정상 분포 π:")
print(pi)

# 검증: π^T P = π^T
P = compute_transition_matrix(A)
pi_next = pi @ P
print(f"\n정상성 확인: {np.allclose(pi, pi_next)}")
```

### 4.3 PageRank Algorithm

PageRank adds teleportation to the random walk:

$$\mathbf{r} = (1 - d) \mathbf{e} + d P^T \mathbf{r}$$

where $d \in [0, 1]$ is the damping factor (typically 0.85), and $\mathbf{e}$ is the uniform distribution.

```python
def pagerank(A, d=0.85, max_iter=100, tol=1e-6):
    """
    PageRank 알고리즘

    Parameters:
    -----------
    A : ndarray
        인접 행렬
    d : float
        Damping factor
    max_iter : int
        최대 반복 횟수
    tol : float
        수렴 임계값

    Returns:
    --------
    r : ndarray
        PageRank 점수
    """
    n = A.shape[0]
    P = compute_transition_matrix(A)

    # 초기화: 균등 분포
    r = np.ones(n) / n

    for iteration in range(max_iter):
        r_new = (1 - d) / n + d * (P.T @ r)

        # 수렴 확인
        if np.linalg.norm(r_new - r, 1) < tol:
            print(f"수렴 완료: {iteration + 1}번 반복")
            break

        r = r_new

    return r

# 방향 그래프 생성 (웹페이지 링크 구조)
A_directed = np.array([
    [0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0],
    [1, 0, 0, 1, 1],
    [0, 0, 0, 0, 1],
    [0, 0, 1, 0, 0]
])

pagerank_scores = pagerank(A_directed)
print("\nPageRank 점수:")
for i, score in enumerate(pagerank_scores):
    print(f"페이지 {i}: {score:.4f}")
```

## 5. Graph Signal Processing

### 5.1 Graph Signals

A graph signal $\mathbf{f} \in \mathbb{R}^n$ is a value assigned to each vertex.

Examples: activity level of each user in a social network, measurement from each sensor in a sensor network

### 5.2 Graph Fourier Transform (GFT)

The eigenvectors of the Laplacian $\mathbf{u}_\ell$ are used as **frequency bases** of the graph.

$$L \mathbf{u}_\ell = \lambda_\ell \mathbf{u}_\ell$$

**Graph Fourier Transform**:

$$\hat{f}(\ell) = \langle \mathbf{f}, \mathbf{u}_\ell \rangle = \mathbf{u}_\ell^T \mathbf{f}$$

**Inverse transform**:

$$\mathbf{f} = \sum_{\ell=0}^{n-1} \hat{f}(\ell) \mathbf{u}_\ell$$

```python
def graph_fourier_transform(A, signal):
    """
    그래프 푸리에 변환

    Parameters:
    -----------
    A : ndarray
        인접 행렬
    signal : ndarray
        그래프 신호

    Returns:
    --------
    f_hat : ndarray
        주파수 영역 신호
    eigenvalues : ndarray
        라플라시안 고유값
    eigenvectors : ndarray
        라플라시안 고유벡터
    """
    L_sym, _ = compute_normalized_laplacian(A)
    eigenvalues, eigenvectors = eigh(L_sym)

    # 그래프 푸리에 변환
    f_hat = eigenvectors.T @ signal

    return f_hat, eigenvalues, eigenvectors

# 예제: 저주파 신호 생성
n = A.shape[0]
signal_smooth = np.array([1.0, 1.1, 0.9, 0.8, 1.0])

f_hat, eigenvalues, eigenvectors = graph_fourier_transform(A, signal_smooth)

print("원 신호:", signal_smooth)
print("주파수 영역 신호:", f_hat)
print("\n라플라시안 고유값 (주파수):", eigenvalues)
```

### 5.3 Graph Filtering

Filtering signals in the frequency domain:

$$\mathbf{f}_{\text{filtered}} = \sum_{\ell=0}^{n-1} h(\lambda_\ell) \hat{f}(\ell) \mathbf{u}_\ell$$

where $h(\lambda)$ is the filter function.

**Low-pass filter** (smoothing): keep only small $\lambda$ components
**High-pass filter** (edge detection): keep only large $\lambda$ components

```python
def graph_filter(A, signal, filter_func):
    """그래프 필터링"""
    f_hat, eigenvalues, eigenvectors = graph_fourier_transform(A, signal)

    # 주파수 영역에서 필터 적용
    f_hat_filtered = f_hat * filter_func(eigenvalues)

    # 역변환
    signal_filtered = eigenvectors @ f_hat_filtered

    return signal_filtered

# 저역 통과 필터
def lowpass_filter(eigenvalues, cutoff=0.5):
    return (eigenvalues < cutoff).astype(float)

# 고역 통과 필터
def highpass_filter(eigenvalues, cutoff=0.5):
    return (eigenvalues >= cutoff).astype(float)

# 노이즈가 있는 신호 생성
np.random.seed(42)
signal_noisy = signal_smooth + 0.3 * np.random.randn(n)

signal_lowpass = graph_filter(A, signal_noisy, lambda eig: lowpass_filter(eig, cutoff=1.0))
signal_highpass = graph_filter(A, signal_noisy, lambda eig: highpass_filter(eig, cutoff=1.0))

print("원 신호:", signal_smooth)
print("노이즈 신호:", signal_noisy)
print("저역 통과 필터 결과:", signal_lowpass)
print("고역 통과 필터 결과:", signal_highpass)
```

## 6. Mathematical Foundations of GNNs

### 6.1 Message Passing Framework

The core of GNNs is **message passing**:

$$\mathbf{h}_v^{(\ell+1)} = \sigma\left( \mathbf{W}^{(\ell)} \sum_{u \in \mathcal{N}(v)} \frac{\mathbf{h}_u^{(\ell)}}{c_{vu}} \right)$$

where:
- $\mathbf{h}_v^{(\ell)}$: feature of vertex $v$ at layer $\ell$
- $\mathcal{N}(v)$: neighbors of vertex $v$
- $c_{vu}$: normalization constant
- $\sigma$: activation function

### 6.2 Spectral Perspective: Graph Convolution

**Spectral graph convolution**:

$$\mathbf{g}_\theta \star \mathbf{f} = U \left( \text{diag}(\theta) U^T \mathbf{f} \right)$$

where $U$ is the eigenvector matrix of the Laplacian, and $\theta$ is a learnable filter parameter.

**Problem**: $O(n^2)$ computational complexity, eigenvalue decomposition required

### 6.3 ChebNet: Chebyshev Polynomial Approximation

Approximation using Chebyshev polynomials:

$$\mathbf{g}_\theta \star \mathbf{f} \approx \sum_{k=0}^{K-1} \theta_k T_k(\tilde{L}) \mathbf{f}$$

where:
- $\tilde{L} = \frac{2}{\lambda_{\max}} L - I$ is the rescaled Laplacian
- $T_k$ is the $k$-th Chebyshev polynomial: $T_0(x) = 1, T_1(x) = x, T_{k}(x) = 2xT_{k-1}(x) - T_{k-2}(x)$

### 6.4 GCN: First-order Approximation

Graph Convolutional Network (Kipf & Welling, 2017) is a simplification with $K=1$:

$$\mathbf{H}^{(\ell+1)} = \sigma\left( \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} \mathbf{H}^{(\ell)} \mathbf{W}^{(\ell)} \right)$$

where $\tilde{A} = A + I$ (adding self-loops), and $\tilde{D}$ is the degree matrix of $\tilde{A}$.

```python
def gcn_layer(A, H, W, activation=lambda x: np.maximum(0, x)):
    """
    GCN 레이어 구현

    Parameters:
    -----------
    A : ndarray, shape (n, n)
        인접 행렬
    H : ndarray, shape (n, d_in)
        입력 특징 행렬
    W : ndarray, shape (d_in, d_out)
        가중치 행렬
    activation : function
        활성화 함수 (기본: ReLU)

    Returns:
    --------
    H_out : ndarray, shape (n, d_out)
        출력 특징 행렬
    """
    n = A.shape[0]

    # 자기 루프 추가
    A_tilde = A + np.eye(n)

    # 차수 행렬
    D_tilde = np.diag(np.sum(A_tilde, axis=1))

    # 정규화: D^{-1/2} A D^{-1/2}
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D_tilde)))
    A_hat = D_inv_sqrt @ A_tilde @ D_inv_sqrt

    # 메시지 패싱 + 선형 변환 + 활성화
    H_out = activation(A_hat @ H @ W)

    return H_out

# 예제: 간단한 2층 GCN
np.random.seed(42)
n = 5
d_in = 3
d_hidden = 4
d_out = 2

# 초기 특징 행렬
X = np.random.randn(n, d_in)

# 가중치 행렬
W1 = np.random.randn(d_in, d_hidden) * 0.1
W2 = np.random.randn(d_hidden, d_out) * 0.1

# 순전파
H1 = gcn_layer(A, X, W1)
print("레이어 1 출력 shape:", H1.shape)

H2 = gcn_layer(A, H1, W2)
print("레이어 2 출력 shape:", H2.shape)
print("\n최종 임베딩:")
print(H2)
```

### 6.5 Graph Attention Networks (GAT)

GAT learns **attention weights** for neighbor vertices:

$$\alpha_{vu} = \frac{\exp(\text{LeakyReLU}(\mathbf{a}^T [\mathbf{W}\mathbf{h}_v \| \mathbf{W}\mathbf{h}_u]))}{\sum_{u' \in \mathcal{N}(v)} \exp(\text{LeakyReLU}(\mathbf{a}^T [\mathbf{W}\mathbf{h}_v \| \mathbf{W}\mathbf{h}_{u'}]))}$$

$$\mathbf{h}_v^{(\ell+1)} = \sigma\left( \sum_{u \in \mathcal{N}(v)} \alpha_{vu} \mathbf{W}^{(\ell)} \mathbf{h}_u^{(\ell)} \right)$$

```python
def graph_attention_layer(A, H, W, a, alpha=0.2):
    """
    간단한 그래프 어텐션 레이어

    Parameters:
    -----------
    A : ndarray, shape (n, n)
        인접 행렬
    H : ndarray, shape (n, d_in)
        입력 특징
    W : ndarray, shape (d_in, d_out)
        특징 변환 가중치
    a : ndarray, shape (2 * d_out,)
        어텐션 파라미터
    alpha : float
        LeakyReLU 기울기

    Returns:
    --------
    H_out : ndarray, shape (n, d_out)
        출력 특징
    """
    n = A.shape[0]

    # 특징 변환
    H_transformed = H @ W
    d_out = H_transformed.shape[1]

    # 어텐션 계산
    attention_scores = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if A[i, j] > 0 or i == j:  # 연결되어 있거나 자기 자신
                # 연결된 특징 [h_i || h_j]
                concat = np.concatenate([H_transformed[i], H_transformed[j]])

                # 어텐션 점수
                score = a @ concat
                attention_scores[i, j] = np.maximum(alpha * score, score)  # LeakyReLU
            else:
                attention_scores[i, j] = -np.inf

    # 소프트맥스
    attention_weights = np.exp(attention_scores - np.max(attention_scores, axis=1, keepdims=True))
    attention_weights = attention_weights / np.sum(attention_weights, axis=1, keepdims=True)

    # 어텐션 집계
    H_out = attention_weights @ H_transformed

    return H_out, attention_weights

# 예제
W_att = np.random.randn(d_in, d_hidden) * 0.1
a_att = np.random.randn(2 * d_hidden) * 0.1

H_gat, att_weights = graph_attention_layer(A, X, W_att, a_att)
print("GAT 출력 shape:", H_gat.shape)
print("\n어텐션 가중치:")
print(att_weights)
```

## Practice Problems

### Problem 1: Proof of Quadratic Form of Laplacian
For the Laplacian $L = D - A$ of an undirected graph, prove:

$$\mathbf{x}^T L \mathbf{x} = \frac{1}{2} \sum_{i,j} A_{ij}(x_i - x_j)^2$$

Use this result to show that the Laplacian is positive semidefinite.

### Problem 2: Spectral Clustering Implementation
Without using `sklearn`, implement a complete spectral clustering algorithm using only NumPy. You must also implement the k-means step. Include:

1. Normalized Laplacian computation
2. Eigenvalue decomposition
3. k-means clustering (Lloyd's algorithm)
4. Cluster quality evaluation using silhouette score

### Problem 3: Eigenvalue Interpretation of PageRank
Transform the PageRank equation $\mathbf{r} = (1 - d) \mathbf{e} + d P^T \mathbf{r}$ into an eigenvector problem. Explain that the dominant eigenvector of matrix $M = (1-d)\mathbf{e}\mathbf{1}^T + dP^T$ is the PageRank vector. Write code to compute this using the power iteration method.

### Problem 4: Graph Fourier Transform Application
Generate a ring graph with 20 vertices and perform the following:

1. Compute and visualize eigenvalues and eigenvectors of the Laplacian
2. Compute the graph Fourier transform of low-frequency signals (e.g., $f_i = \cos(2\pi i / 20)$) and high-frequency signals (e.g., $f_i = (-1)^i$)
3. Apply a Gaussian low-pass filter $h(\lambda) = \exp(-\lambda^2 / (2\sigma^2))$ and visualize the results

### Problem 5: GCN vs GAT Comparison
Design an experiment to compare the behavior of GCN and GAT layers on a small graph (10-20 vertices):

1. Compute output embeddings for both methods
2. Visualize attention weights to analyze the neighbor importance learned by GAT
3. Theoretically analyze computational complexity and measure actual execution time

## References

### Papers
- Chung, F. R. K. (1997). *Spectral Graph Theory*. American Mathematical Society.
- Von Luxburg, U. (2007). "A tutorial on spectral clustering." *Statistics and Computing*, 17(4), 395-416.
- Kipf, T. N., & Welling, M. (2017). "Semi-supervised classification with graph convolutional networks." *ICLR*.
- Veličković, P., et al. (2018). "Graph Attention Networks." *ICLR*.
- Bronstein, M. M., et al. (2021). "Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges." *arXiv:2104.13478*.

### Online Resources
- [Spectral Graph Theory (Spielman, Yale)](http://www.cs.yale.edu/homes/spielman/561/)
- [Graph Representation Learning Book (Hamilton)](https://www.cs.mcgill.ca/~wlh/grl_book/)
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
- [NetworkX Tutorial](https://networkx.org/documentation/stable/tutorial.html)

### Libraries
- `networkx`: graph creation and analysis
- `scipy.sparse`: sparse matrix operations
- `torch_geometric`: GNN implementation
- `spektral`: Keras-based GNN
