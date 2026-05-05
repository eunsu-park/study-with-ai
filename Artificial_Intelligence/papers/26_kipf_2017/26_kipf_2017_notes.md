---
title: "Semi-Supervised Classification with Graph Convolutional Networks"
authors: [Thomas N. Kipf, Max Welling]
year: 2017
journal: "International Conference on Learning Representations (ICLR)"
doi: "arXiv:1609.02907"
topic: Artificial_Intelligence
tags: [graph-neural-network, GCN, spectral-graph-theory, semi-supervised-learning, node-classification, geometric-deep-learning]
status: completed
date_started: 2026-04-28
date_completed: 2026-04-28
---

# 26. Semi-Supervised Classification with Graph Convolutional Networks / 그래프 합성곱 신경망을 이용한 준지도 분류

---

## 1. Core Contribution / 핵심 기여

**한국어**
이 논문은 **그래프 구조 데이터에 직접 작동하는 합성곱 신경망의 효율적 변형**, 즉 GCN(Graph Convolutional Network)을 제시합니다. 핵심은 두 가지입니다. 첫째, 저자들은 **spectral graph convolution의 1차 근사(first-order approximation)**로부터 매우 단순한 layer-wise propagation rule을 유도합니다:

$$H^{(l+1)} = \sigma\!\left(\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)}\right)$$

여기서 $\tilde{A} = A + I_N$은 self-loop를 추가한 인접 행렬, $\tilde{D}$는 그 차수 행렬입니다. 이 식은 Chebyshev 다항식 기반 $K$-localized 필터(Defferrard et al., 2016)에서 $K=1$로 단순화한 뒤, **renormalization trick**으로 수치 안정성을 확보한 결과입니다. 둘째, 저자들은 이 모델 $f(X, A)$를 **준지도 노드 분류**에 적용합니다. 기존 연구가 손실 함수에 graph Laplacian regularization 항(식 1)을 명시적으로 더하던 것과 달리, GCN은 인접 행렬 $A$에 직접 조건화되어 그래프 구조와 노드 특징을 동시에 학습합니다. 결과적으로 Citeseer 70.3%, Cora 81.5%, Pubmed 79.0%, NELL 66.0%로 기존 SOTA를 상당한 차이로 능가하며, 학습 시간은 4–48초로 비교 가능한 가장 빠른 방법(Planetoid)보다 한 자릿수 빠릅니다.

**English**
This paper introduces **Graph Convolutional Networks (GCNs)** — an efficient variant of CNNs that operates directly on graph-structured data. The contribution is twofold. First, the authors derive a remarkably simple layer-wise propagation rule from a **first-order approximation of spectral graph convolutions**:

$$H^{(l+1)} = \sigma\!\left(\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)}\right)$$

where $\tilde{A} = A + I_N$ is the adjacency matrix with self-loops added and $\tilde{D}$ is its degree matrix. This is obtained by truncating Chebyshev-polynomial filters (Defferrard et al., 2016) at $K=1$ and applying a **renormalization trick** for numerical stability. Second, the authors apply $f(X, A)$ to **semi-supervised node classification**. Unlike prior work that adds an explicit graph-Laplacian regularizer (Eq. 1) to the loss, GCN conditions the model directly on $A$, learning from both labeled and unlabeled nodes through propagation. The model achieves 70.3% on Citeseer, 81.5% on Cora, 79.0% on Pubmed, and 66.0% on NELL — beating SOTA by a significant margin while training in just 4–48 seconds, an order of magnitude faster than the closest competitor (Planetoid).

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction (§1) / 서론

**한국어**
저자들은 그래프 노드 분류 문제를 다룹니다 — 인용 네트워크에서 일부 문서만 라벨이 있을 때, 모든 문서를 분류해야 합니다. 이는 **graph-based semi-supervised learning** 문제로, 기존에는 다음과 같은 손실 함수로 정식화되었습니다:

$$\mathcal{L} = \mathcal{L}_0 + \lambda \mathcal{L}_{\text{reg}}, \quad \mathcal{L}_{\text{reg}} = \sum_{i,j} A_{ij} \|f(X_i) - f(X_j)\|^2 = f(X)^\top \Delta f(X) \tag{1}$$

여기서 $\mathcal{L}_0$는 라벨된 노드의 지도 학습 손실, $\mathcal{L}_{\text{reg}}$는 graph Laplacian regularization 항입니다. 이 항은 "**연결된 노드는 같은 레이블을 가진다**"는 가정에 의존하는데, 이는 너무 제한적입니다 — 엣지가 단순한 유사성을 넘어 더 풍부한 정보를 담을 수 있기 때문입니다.

저자의 핵심 아이디어: **regularizer를 제거하고, 대신 신경망 $f(X, A)$가 인접 행렬 $A$를 직접 입력으로 받도록 한다.** 이렇게 하면:
- 모든 노드(라벨 유무 무관)에 대해 그래프 구조 정보가 전파됨
- 라벨된 노드의 손실 $\mathcal{L}_0$로부터의 gradient가 그래프를 통해 모든 노드에 영향을 줌
- 가정이 완화되어 표현력이 증가함

**English**
The authors tackle **node classification** in graphs — labeling all documents in a citation network when only a small subset has labels. The standard formulation adds a Laplacian-regularization term:

$$\mathcal{L} = \mathcal{L}_0 + \lambda \mathcal{L}_{\text{reg}}, \quad \mathcal{L}_{\text{reg}} = f(X)^\top \Delta f(X)$$

This relies on the strong assumption that connected nodes share labels — too restrictive, because edges can encode more than mere similarity.

**Key idea**: drop the regularizer and let the neural network $f(X, A)$ take the adjacency matrix as direct input. Information then propagates through the graph regardless of which nodes are labeled, and gradient signals from labeled nodes flow to unlabeled ones via the propagation operator.

The two stated contributions:
1. A simple, well-behaved layer-wise propagation rule, motivated as a first-order approximation of spectral graph convolutions.
2. A demonstration that this model gives fast and scalable semi-supervised classification of nodes.

### Part II: Fast Approximate Convolutions on Graphs (§2) / 그래프 위의 빠른 근사 합성곱

**한국어**
이 섹션은 논문의 수학적 핵심입니다. 최종 목표는 다음의 multi-layer GCN propagation rule을 유도하는 것:

$$H^{(l+1)} = \sigma\!\left(\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)}\right) \tag{2}$$

여기서 $\tilde{A} = A + I_N$, $\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$, $H^{(0)} = X$ (입력 특징 행렬).

#### 2.1 Spectral Graph Convolutions / 스펙트럼 그래프 합성곱

**한국어**
신호 $x \in \mathbb{R}^N$ (각 노드에 스칼라 하나)와 Fourier domain의 필터 $g_\theta = \mathrm{diag}(\theta)$ ($\theta \in \mathbb{R}^N$)에 대해 spectral convolution은:

$$g_\theta \star x = U g_\theta(\Lambda) U^\top x \tag{3}$$

여기서 $L = I_N - D^{-1/2} A D^{-1/2} = U \Lambda U^\top$는 정규화된 graph Laplacian의 고유분해입니다. 직관: $U^\top x$는 $x$의 graph Fourier 변환, $g_\theta(\Lambda)$는 주파수별 곱셈, $U$는 역변환.

**문제**: $U$를 곱하는 것은 $O(N^2)$이고, $L$의 고유분해 자체가 큰 그래프에는 $O(N^3)$로 prohibitive.

**해결**: Hammond et al. (2011)은 $g_\theta(\Lambda)$를 Chebyshev 다항식의 $K$차 절단으로 근사할 수 있음을 보였습니다:

$$g_{\theta'}(\Lambda) \approx \sum_{k=0}^{K} \theta'_k T_k(\tilde{\Lambda}), \qquad \tilde{\Lambda} = \frac{2}{\lambda_{\max}}\Lambda - I_N \tag{4}$$

Chebyshev 다항식: $T_0(x)=1$, $T_1(x)=x$, $T_k(x) = 2x T_{k-1}(x) - T_{k-2}(x)$.

이를 신호에 적용하면:

$$g_{\theta'} \star x \approx \sum_{k=0}^{K} \theta'_k T_k(\tilde{L}) x, \qquad \tilde{L} = \frac{2}{\lambda_{\max}}L - I_N \tag{5}$$

**핵심 통찰**: $L^k$는 $k$-hop 이웃에만 의존하므로, 이 표현은 $K$-localized이며 비용이 $O(|\mathcal{E}|)$ (엣지 수에 선형)입니다.

**English**
For a signal $x \in \mathbb{R}^N$ (one scalar per node) and a Fourier-domain filter $g_\theta = \mathrm{diag}(\theta)$:

$$g_\theta \star x = U g_\theta(\Lambda) U^\top x$$

with eigendecomposition $L = I_N - D^{-1/2}AD^{-1/2} = U\Lambda U^\top$. Intuition: $U^\top x$ is $x$'s graph-Fourier transform, $g_\theta(\Lambda)$ multiplies frequency components, $U$ inverts.

The bottleneck is $O(N^2)$ multiplication by $U$ and $O(N^3)$ eigendecomposition. Hammond et al. (2011) approximate $g_\theta(\Lambda)$ as a $K$-th order Chebyshev expansion (Eq. 4), yielding the $K$-localized convolution of Eq. (5) with $O(|\mathcal{E}|)$ cost. Crucially, $L^k$ couples only $k$-hop neighborhoods, so the expansion is naturally local.

#### 2.2 Layer-wise Linear Model / 레이어별 선형 모델

**한국어**
**1차 근사 ($K=1$)**: 한 레이어가 직접 이웃만 보면, 더 깊은 모델이 필요해 보입니다. 그러나 저자들은 이것이 **장점**이라고 주장합니다 — 비선형 활성화 함수와 함께 여러 레이어를 쌓으면 풍부한 필터를 얻을 수 있고, 매개변수가 줄어 overfitting이 감소합니다. 또한 ResNet 스타일로 deeper 모델을 만들 수 있습니다 (paper #20 참조).

추가로 $\lambda_{\max} \approx 2$로 근사하면 (학습 중 신경망 매개변수가 적응할 것임):

$$g_{\theta'} \star x \approx \theta'_0 x + \theta'_1 (L - I_N) x = \theta'_0 x - \theta'_1 D^{-1/2} A D^{-1/2} x \tag{6}$$

**매개변수 공유**로 더 단순화: $\theta = \theta'_0 = -\theta'_1$:

$$g_\theta \star x \approx \theta\!\left(I_N + D^{-1/2} A D^{-1/2}\right) x \tag{7}$$

**문제**: $I_N + D^{-1/2}AD^{-1/2}$의 고유값 범위는 $[0, 2]$. 이를 deep network에서 반복 적용하면 수치 불안정성 (gradient explosion/vanishing).

**Renormalization trick**: $I_N + D^{-1/2}AD^{-1/2}$ 대신 $\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$ (단, $\tilde{A} = A + I_N$, $\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$).

이 변형은 self-loop를 명시적으로 인접 행렬에 추가하고, 그 결과 차수 행렬로 다시 정규화합니다. 고유값 범위가 $[0, 2]$에서 $[0, 1]$ 정도로 압축되어 안정적입니다 (정확히는 $\hat{A} = \tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$의 고유값은 $[-1, 1]$ 범위에 있으나, self-loop가 음의 고유값을 얼마나 포함하는지 줄여 학습이 안정화됩니다).

**일반화** (입력 채널 $C$, 출력 필터 수 $F$):

$$Z = \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} X \Theta \tag{8}$$

여기서 $X \in \mathbb{R}^{N \times C}$, $\Theta \in \mathbb{R}^{C \times F}$, $Z \in \mathbb{R}^{N \times F}$. 비용은 $O(|\mathcal{E}|FC)$ — 엣지 수에 선형. 이는 $\tilde{A}X$를 sparse–dense 행렬 곱으로 효율적으로 계산하기 때문입니다.

**English**
Setting $K=1$ might seem too local, but stacking many such layers with non-linearities recovers rich filters while using fewer parameters. Approximating $\lambda_{\max}\approx 2$ gives Eq. (6); sharing parameters $\theta = \theta'_0 = -\theta'_1$ yields Eq. (7), which has only one parameter per filter. Eigenvalues of $I_N + D^{-1/2}AD^{-1/2}$ lie in $[0, 2]$ — repeatedly applying it causes numerical issues. The **renormalization trick** replaces it with $\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$ where $\tilde{A} = A + I_N$, stabilizing the spectrum. Generalizing to $C$ input channels and $F$ filters gives Eq. (8) with $O(|\mathcal{E}|FC)$ complexity via sparse-dense multiplication.

### Part III: Semi-Supervised Node Classification (§3) / 준지도 노드 분류

**한국어**
2-layer GCN의 정확한 forward pass:

$$Z = f(X, A) = \mathrm{softmax}\!\left(\hat{A}\,\mathrm{ReLU}(\hat{A} X W^{(0)})\,W^{(1)}\right) \tag{9}$$

여기서 $\hat{A} = \tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$는 **전처리 단계에서 한 번만 계산**합니다. 가중치 행렬:
- $W^{(0)} \in \mathbb{R}^{C \times H}$ — 입력 → 은닉
- $W^{(1)} \in \mathbb{R}^{H \times F}$ — 은닉 → 출력 ($F$ = 클래스 수)

**손실 함수** (라벨된 노드만 사용):

$$\mathcal{L} = -\sum_{l \in \mathcal{Y}_L} \sum_{f=1}^{F} Y_{lf} \ln Z_{lf} \tag{10}$$

$\mathcal{Y}_L$은 라벨된 노드의 인덱스 집합. **중요**: 라벨 없는 노드의 정보도 propagation $\hat{A}$를 통해 학습에 영향을 줍니다.

**학습**: full-batch gradient descent (전체 데이터셋이 메모리에 들어가는 한 가능). $A$의 sparse 표현 시 메모리 $O(|\mathcal{E}|)$. Stochasticity는 dropout으로 도입.

**Figure 1 (도식)**:
```
Input layer (C 채널)        Output layer (F 클래스)
   X1 — X2                       Z1 — Z2
   |    |        ───────►        |    |
   X3 — X4         hidden        Z3 — Z4
                  layers            ↓
                                   라벨 Yi
```
Right panel: 학습된 hidden representation의 t-SNE 시각화. Cora 데이터셋(5% 라벨)에서 클래스가 명확히 분리됨.

**English**
The two-layer model is given by Eq. (9), with $\hat{A}$ precomputed once. Weight $W^{(0)} \in \mathbb{R}^{C\times H}$ maps inputs to hidden, $W^{(1)} \in \mathbb{R}^{H\times F}$ to outputs. Cross-entropy is computed only over labeled nodes (Eq. 10), but unlabeled nodes still influence learning through $\hat{A}$. Training uses full-batch gradient descent (memory $O(|\mathcal{E}|)$ with sparse $A$); dropout provides stochasticity. Figure 1 right shows t-SNE of hidden activations on Cora — classes form clear clusters from only 5% labels.

### Part IV: Related Work (§4) / 관련 연구

**한국어**
저자들은 두 갈래 — 그래프 기반 준지도 학습과 그래프 위의 신경망 — 모두에서 영감을 받습니다.

**그래프 기반 준지도 학습**:
- **Laplacian regularization**: label propagation (Zhu et al., 2003), manifold regularization (Belkin et al., 2006), deep semi-supervised embedding (Weston et al., 2012)
- **Skip-gram 기반 그래프 임베딩**: DeepWalk (Perozzi et al., 2014), LINE (Tang et al., 2015), node2vec (Grover & Leskovec, 2016) — random walk + skip-gram
- **Planetoid** (Yang et al., 2016): 다단계 파이프라인을 단순화

**그래프 위의 신경망**:
- **Graph Neural Networks** (Gori et al., 2005; Scarselli et al., 2009): RNN의 한 형태, fixed-point까지 propagation
- **Gated GNN** (Li et al., 2016): RNN 학습 기법 도입
- **Convolutional networks on graphs** (Duvenaud et al., 2015): 노드 차수별 별도 가중치 → 확장성 부족
- **Diffusion-CNN** (Atwood & Towsley, 2016): $O(N^2)$ 복잡도
- **Spectral CNN** (Bruna et al., 2014; Defferrard et al., 2016): GCN의 직접적 선조

**English**
Two threads inspire the work. **Graph-based semi-supervised learning** includes Laplacian regularization (label propagation, manifold regularization), skip-gram-based graph embeddings (DeepWalk, LINE, node2vec), and Planetoid. **Neural networks on graphs** include early GNNs (Scarselli, Gori), Gated GNNs, Duvenaud's degree-specific CNNs, diffusion-CNNs, and spectral CNNs (Bruna 2014; Defferrard 2016 — direct predecessor of GCN).

### Part V: Experiments (§5) / 실험

**한국어**

**데이터셋** (Table 1):
| Dataset | Type | Nodes | Edges | Classes | Features | Label rate |
|---|---|---|---|---|---|---|
| Citeseer | Citation network | 3,327 | 4,732 | 6 | 3,703 | 0.036 |
| Cora | Citation network | 2,708 | 5,429 | 7 | 1,433 | 0.052 |
| Pubmed | Citation network | 19,717 | 44,338 | 3 | 500 | 0.003 |
| NELL | Knowledge graph | 65,755 | 266,144 | 210 | 5,414 | 0.001 |

**실험 설정**:
- 2-layer GCN (Cora, Citeseer, Pubmed); 3-layer (NELL)
- 200 epochs, Adam optimizer, learning rate 0.01, early stopping (window 10)
- Glorot 초기화, row-normalized 입력
- Citation: dropout 0.5, L2 $5 \times 10^{-4}$, hidden 16
- NELL: dropout 0.1, L2 $10^{-5}$, hidden 64

**English**
Datasets: 3 citation networks (Citeseer, Cora, Pubmed) and 1 knowledge graph (NELL). Label rates range from 0.1% (NELL) to 5.2% (Cora). 2-layer GCN trained for ≤200 epochs with Adam (lr=0.01), early stopping; dropout and L2 regularization tuned per dataset.

### Part VI: Results (§6) / 결과

**한국어**

**Table 2 — 분류 정확도 (%)**:
| Method | Citeseer | Cora | Pubmed | NELL |
|---|---|---|---|---|
| ManiReg | 60.1 | 59.5 | 70.7 | 21.8 |
| SemiEmb | 59.6 | 59.0 | 71.1 | 26.7 |
| LP | 45.3 | 68.0 | 63.0 | 26.5 |
| DeepWalk | 43.2 | 67.2 | 65.3 | 58.1 |
| ICA | 69.1 | 75.1 | 73.9 | 23.1 |
| Planetoid* | 64.7 (26s) | 75.7 (13s) | 77.2 (25s) | 61.9 (185s) |
| **GCN (this paper)** | **70.3 (7s)** | **81.5 (4s)** | **79.0 (38s)** | **66.0 (48s)** |

GCN이 모든 데이터셋에서 SOTA를 능가하며, 소요 시간도 더 짧습니다. 특히 Cora에서 81.5%로 Planetoid의 75.7% 대비 +5.8%p 향상.

**Table 3 — Propagation model 비교 (Cora)**:
| Description | Propagation model | Citeseer | Cora | Pubmed |
|---|---|---|---|---|
| Chebyshev ($K=3$) | $\sum_{k=0}^K T_k(\tilde{L}) X \Theta_k$ | 69.8 | 79.5 | 74.4 |
| Chebyshev ($K=2$) | (위와 동일, $K=2$) | 69.6 | 81.2 | 73.8 |
| 1st-order (Eq. 6) | $X\Theta_0 + D^{-1/2}AD^{-1/2}X\Theta_1$ | 68.3 | 80.0 | 77.5 |
| Single param (Eq. 7) | $(I_N + D^{-1/2}AD^{-1/2})X\Theta$ | 69.3 | 79.2 | 77.4 |
| **Renormalization (Eq. 8)** | $\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}X\Theta$ | **70.3** | **81.5** | **79.0** |
| 1st-order term only | $D^{-1/2}AD^{-1/2}X\Theta$ | 68.7 | 80.5 | 77.8 |
| MLP (no graph) | $X\Theta$ | 46.5 | 55.1 | 71.4 |

**핵심 발견**:
- Renormalization trick이 모든 데이터셋에서 최고 성능
- MLP만 사용 시(그래프 무시) 정확도 급락 (Cora 55.1%) → 그래프 정보가 필수
- Higher-order Chebyshev ($K=2,3$)도 좋지만 매개변수 증가로 인한 이득은 미미

**Training time per epoch (Figure 2)**: random graph로 1k~10M 엣지에서 측정. GPU에서 wall-clock time이 엣지 수에 선형. 10M 엣지에서 GPU OOM (16GB).

**English**
GCN beats every baseline on all four datasets while running in 4–48 seconds — order of magnitude faster than Planetoid. The renormalization trick (Eq. 8) outperforms all alternative propagation models. Critical observation: removing the graph (MLP only) collapses accuracy to 55.1% on Cora — the graph carries most of the signal at low label rates. Higher-order Chebyshev ($K=2,3$) helps but marginally, suggesting the renormalization variant captures the right inductive bias. Training time scales linearly with edges.

### Part VII: Discussion & Limitations (§7) / 논의 및 한계

**한국어**
**장점**:
- Laplacian regularization 가정(엣지 = 유사성)을 완화
- Skip-gram 다단계 파이프라인보다 단순하고 효율적
- 모든 레이어에서 이웃 정보 propagation → ICA처럼 라벨만 모으는 것보다 강력

**한계**:
1. **메모리**: full-batch GD이므로 메모리는 데이터셋 크기에 선형. 매우 큰 그래프에서는 mini-batch SGD 필요하나 $K$-hop neighborhood가 메모리에 들어가야 함.
2. **Directed edges & edge features**: 현재 framework는 무방향 그래프만 자연스럽게 지원. NELL 처리는 방향 엣지를 양분 그래프로 변환하는 우회 사용.
3. **Locality 가정**: $K$-hop neighborhood만 의존, self-connection의 가중치가 이웃 엣지와 같다고 가정. trade-off 매개변수 $\lambda$ 도입 가능 (식 11): $\tilde{A} = A + \lambda I_N$.

**English**
Strengths: relaxes the edge-as-similarity assumption; simpler than skip-gram pipelines; aggregates feature info from neighbors in every layer. Limitations: memory grows linearly with dataset (full-batch); no native support for directed/edge-feature graphs (NELL workaround required); locality and equal self-connection weight assumed (could be relaxed with $\tilde{A} = A + \lambda I_N$, Eq. 11).

### Part VIII: Appendix A — Weisfeiler-Lehman Connection / 와이스파일러-레만 연결

**한국어**
이 부록은 GCN의 가장 깊은 통찰을 제공합니다. **1-dim Weisfeiler-Lehman algorithm (WL-1)** (1968)은 그래프 동형성 검사 알고리즘:

```
Algorithm: WL-1
Input: 초기 노드 색상 (h_1^0, ..., h_N^0)
Output: 최종 노드 색상
Repeat:
  for each node v_i:
    h_i^{t+1} = hash(sum_{j ∈ N_i} h_j^t)
  t = t + 1
Until 색상이 안정화
```

저자들은 이 hash 함수를 **신경망 레이어**로 대체하면:

$$h_i^{(l+1)} = \sigma\!\left(\sum_{j \in \mathcal{N}_i} \frac{1}{c_{ij}} h_j^{(l)} W^{(l)}\right) \tag{12}$$

$c_{ij} = \sqrt{d_i d_j}$로 선택하면 GCN propagation rule (식 2)이 정확히 복원됩니다. 따라서 **GCN은 1-dim WL 알고리즘의 미분 가능, 매개변수화된 일반화**입니다.

#### A.1 Random weights로도 의미 있는 임베딩 / Meaningful Embeddings with Random Weights

**한국어**
WL과의 유사성으로부터 **untrained GCN도 강력한 feature extractor**임을 보일 수 있습니다. Zachary's karate club network (34 노드, 4 클래스, modularity-based clustering으로 라벨링)에서:
- 3-layer GCN, 가중치 무작위 초기화 (Glorot)
- 입력 $X = I_N$ (노드 정체성만)
- 출력은 2D 임베딩

결과: random GCN이 클래스를 시각적으로 분리 (Figure 3b). DeepWalk와 비교 가능한 품질이지만 **학습 없이** 달성.

#### A.2 Semi-Supervised 노드 임베딩 / Semi-Supervised Node Embeddings

**한국어**
Karate club에서 **클래스당 단 한 노드만 라벨**(총 4개)로 학습:
- Adam, lr=0.01, 300 iterations, cross-entropy
- Figure 4: iteration 25, 50, 75, 100, 200, 300에서 임베딩 진화
- 결과: 4개 라벨만으로 4개 커뮤니티가 선형 분리됨

이는 그래프 구조 자체가 풍부한 prior임을 보여줍니다.

**English**
Appendix A reveals GCN's deepest insight: GCN is a **differentiable, parameterized generalization of the 1-dim Weisfeiler-Lehman algorithm**. Replace WL's hash with a neural-net layer (Eq. 12), and choose the normalization $c_{ij} = \sqrt{d_i d_j}$ — out pops Eq. (2). Even with random weights, GCN extracts meaningful representations on Zachary's karate club (Figure 3b). With one labeled node per class (4 total), training for 300 iterations linearly separates the four communities (Figure 4) — showing graph structure is itself a powerful prior.

### Part IX: Appendix B — Model Depth / 모델 깊이

**한국어**
1~10 layer 모델을 5-fold cross-validation으로 비교 (Cora, Citeseer, Pubmed, 모든 라벨 사용).

**Figure 5 결과**:
- 최적 깊이: 2~3 layer
- 7 layer 이상에서 학습이 어려워짐 — $K$-hop neighborhood가 너무 커지고 매개변수 수 증가
- **Residual connection** 도입으로 deeper 모델 학습 가능 (식 14): $H^{(l+1)} = \sigma(\hat{A} H^{(l)} W^{(l)}) + H^{(l)}$
- Residual variant는 deeper 모델에서 안정적이지만, 얕은 모델 대비 큰 이득은 없음

**English**
Best depth: 2–3 layers. Beyond 7 layers, training without residuals becomes hard — receptive field saturates and parameters grow. Adding residual connections (Eq. 14, ResNet-style) stabilizes deeper training but yields no significant accuracy gain over shallow models.

---

## 3. Key Takeaways / 핵심 시사점

1. **단 한 줄의 propagation rule이 모든 것을 바꾼다 / A one-line propagation rule changed the field**
   $H^{(l+1)} = \sigma(\hat{A} H^{(l)} W^{(l)})$. 이 단순한 규칙이 GraphSAGE, GAT, GIN 등 거의 모든 후속 GNN의 출발점이 되었습니다. / This deceptively simple rule became the foundation for almost every subsequent GNN.

2. **Spectral과 spatial이 만나는 지점 / Bridge between spectral and spatial views**
   1차 근사는 spectral 관점($K=1$ Chebyshev)에서 유도되지만, 실제로는 직접 이웃의 normalized aggregation이라는 spatial 해석을 갖습니다. 두 관점의 통일이 GCN의 우아함입니다. / The first-order approximation is derived spectrally but interpreted spatially — a unification that gives GCN its elegance.

3. **Renormalization trick은 단순하지만 결정적 / The renormalization trick is simple but decisive**
   $I_N + D^{-1/2}AD^{-1/2}$ → $\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$. 단지 self-loop 추가지만, 고유값 안정화와 정확도 향상(+0.5~1.5%p) 모두 달성. / Adding a self-loop and renormalizing stabilizes spectra and improves accuracy by 0.5–1.5%p — a remarkable return on a tiny change.

4. **그래프 자체가 강력한 prior / The graph itself is a strong prior**
   Cora에서 그래프 무시 시(MLP) 55.1%, GCN 81.5%. 클래스당 1개 라벨로도 communities를 분리. 라벨이 희소할수록 그래프 정보가 더 중요해집니다. / Without the graph, MLP gets 55.1% on Cora; GCN gets 81.5%. With only 1 labeled node per class, GCN separates communities — graph structure carries enormous signal.

5. **Laplacian regularization의 종말 / End of explicit Laplacian regularization**
   "연결된 노드 = 같은 레이블" 가정은 손실 함수에서 사라지고, 인접 행렬이 직접 모델 입력이 됩니다. 이는 graph-based 준지도 학습의 패러다임 전환. / Connected-nodes-share-labels assumption is dropped from the loss; the adjacency becomes a direct model input — a paradigm shift for graph-based SSL.

6. **WL 알고리즘과의 연결 / Connection to Weisfeiler-Lehman**
   GCN은 1-dim WL의 미분 가능, 매개변수화된 일반화. 이 통찰은 후속 연구(GIN by Xu et al., 2019)에서 GCN의 표현력 한계를 분석하고 더 강력한 변형을 제안하는 토대가 됩니다. / GCN ≈ a differentiable, parameterized 1-dim WL — this insight later powered representation-power analyses (e.g., GIN).

7. **얕은 모델이 최선 / Shallow is best**
   2~3 layer가 최적. 7+ layer에서 학습이 어렵고, residual도 큰 이득 없음. 이는 graph oversmoothing 문제를 암시 — 후속 연구의 큰 주제가 됩니다. / 2–3 layers is optimal. Deeper models are hard to train and show no gain — foreshadowing the over-smoothing problem that later work would tackle.

8. **확장성과 효율성 / Scalability and efficiency**
   $O(|\mathcal{E}|FC)$ — 엣지 수에 선형. Sparse-dense 행렬 곱으로 실제로 빠름. 4–48초 학습 시간은 동시대 다른 방법보다 한 자릿수 빠릅니다. / Linear in edges via sparse-dense multiplication. Training in seconds — an order of magnitude faster than peers.

---

## 4. Mathematical Summary / 수학적 요약

### Core derivation chain / 핵심 유도 체인

**Step 1**: Spectral graph convolution (식 3)
$$g_\theta \star x = U g_\theta(\Lambda) U^\top x$$
- $L = I_N - D^{-1/2}AD^{-1/2} = U\Lambda U^\top$ (정규화 Laplacian의 고유분해)
- 비용: $O(N^2)$ per multiplication, $O(N^3)$ for eigendecomposition

**Step 2**: Chebyshev polynomial approximation (식 4–5)
$$g_{\theta'} \star x \approx \sum_{k=0}^{K} \theta'_k T_k(\tilde{L}) x, \quad \tilde{L} = \frac{2}{\lambda_{\max}}L - I_N$$
- $T_0(x)=1$, $T_1(x)=x$, $T_k(x) = 2x T_{k-1}(x) - T_{k-2}(x)$
- 비용: $O(K|\mathcal{E}|)$
- 속성: $K$-localized ($K$-hop 이웃에만 의존)

**Step 3**: First-order approximation, $K=1, \lambda_{\max} \approx 2$ (식 6)
$$g_{\theta'} \star x \approx \theta'_0 x - \theta'_1 D^{-1/2}AD^{-1/2} x$$

**Step 4**: Parameter sharing, $\theta = \theta'_0 = -\theta'_1$ (식 7)
$$g_\theta \star x \approx \theta(I_N + D^{-1/2}AD^{-1/2}) x$$

**Step 5**: Renormalization trick (식 8)
$$Z = \tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}} X \Theta$$
- $\tilde{A} = A + I_N$, $\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$
- 비용: $O(|\mathcal{E}|FC)$ ($X \in \mathbb{R}^{N\times C}$, $\Theta \in \mathbb{R}^{C\times F}$)

**Step 6**: Multi-layer GCN (식 2, layer-wise)
$$H^{(l+1)} = \sigma\!\left(\hat{A} H^{(l)} W^{(l)}\right), \quad \hat{A} = \tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}$$

**Step 7**: Two-layer end-to-end (식 9)
$$Z = f(X, A) = \mathrm{softmax}\!\left(\hat{A}\,\mathrm{ReLU}(\hat{A} X W^{(0)})\,W^{(1)}\right)$$

**Step 8**: Loss (식 10)
$$\mathcal{L} = -\sum_{l \in \mathcal{Y}_L}\sum_{f=1}^{F} Y_{lf} \ln Z_{lf}$$

### Worked example: 4-node toy graph / 4-노드 토이 그래프 워크스루

**한국어**
다음 그래프를 고려:

```
1 — 2
|   |
3 — 4
```

인접 행렬과 self-loop 추가:
$$A = \begin{pmatrix} 0 & 1 & 1 & 0 \\ 1 & 0 & 0 & 1 \\ 1 & 0 & 0 & 1 \\ 0 & 1 & 1 & 0 \end{pmatrix}, \quad \tilde{A} = A + I_4 = \begin{pmatrix} 1 & 1 & 1 & 0 \\ 1 & 1 & 0 & 1 \\ 1 & 0 & 1 & 1 \\ 0 & 1 & 1 & 1 \end{pmatrix}$$

차수 행렬:
$$\tilde{D} = \mathrm{diag}(3, 3, 3, 3) \quad (\text{모든 노드의 차수 = 자기 + 이웃 2})$$

따라서:
$$\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2} = \frac{1}{3}\tilde{A} = \frac{1}{3}\begin{pmatrix} 1 & 1 & 1 & 0 \\ 1 & 1 & 0 & 1 \\ 1 & 0 & 1 & 1 \\ 0 & 1 & 1 & 1 \end{pmatrix}$$

**해석**: 노드 1의 새 표현은 자신($\frac{1}{3}$) + 이웃 2($\frac{1}{3}$) + 이웃 3($\frac{1}{3}$)의 가중합. 이 정규화 덕에 출력의 크기가 입력 크기와 비슷하게 유지됩니다.

**English**
For the 4-cycle graph above, $\tilde{A} = A + I_4$, $\tilde{D} = 3I_4$. Then $\hat{A} = \tilde{A}/3$ — each node's new representation is the equally-weighted average of itself and its two neighbors. Normalization preserves magnitude, ensuring stability across layers.

### Connection to Weisfeiler-Lehman (Eq. 12)

$$h_i^{(l+1)} = \sigma\!\left(\sum_{j \in \mathcal{N}_i} \frac{1}{c_{ij}} h_j^{(l)} W^{(l)}\right), \quad c_{ij} = \sqrt{d_i d_j}$$

이것은 정확히 GCN의 노드별 형태이며, hash 대신 학습 가능한 weight matrix $W^{(l)}$를 사용한 미분 가능 WL입니다.

### Parameter count / 매개변수 수

2-layer GCN on Cora ($C=1433$, $H=16$, $F=7$):
- $W^{(0)}$: $1433 \times 16 = 22{,}928$
- $W^{(1)}$: $16 \times 7 = 112$
- **Total**: $\sim 23{,}040$ parameters
- 비교: Planetoid는 nodes-as-parameters로 훨씬 더 많은 매개변수 사용

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1968 ─ Weisfeiler & Lehman: WL-1 algorithm (graph isomorphism)
        │
1997 ─ Hochreiter & Schmidhuber: LSTM (paper #9, sequence modeling)
        │
2003 ─ Zhu et al.: Label propagation (Laplacian regularization)
        │
2009 ─ Scarselli et al.: Graph Neural Networks (RNN-style)
        │
2011 ─ Hammond et al.: Wavelets on graphs (Chebyshev approximation)
        │
2014 ─ Bruna et al.: Spectral CNN on graphs
        │
2014 ─ Perozzi et al.: DeepWalk (random walk + skip-gram)
        │
2015 ─ Duvenaud et al.: Convolutional networks for molecular fingerprints
        │
2016 ─ Defferrard et al.: ChebNet (fast localized spectral filters)
        │       ★ 직접적 선조 / Direct predecessor
        │
2016 ─ Yang et al.: Planetoid (semi-supervised graph embeddings)
        │
2017 ─ ★★★ Kipf & Welling: GCN (this paper) ★★★
        │       ─ Chebyshev → 1차 근사 → renormalization
        │
2017 ─ Hamilton et al.: GraphSAGE (inductive variant)
        │
2018 ─ Veličković et al.: Graph Attention Networks (GAT)
        │
2019 ─ Xu et al.: GIN (Graph Isomorphism Network) — WL connection 활용
        │
2020 ─ Klicpera et al.: GNN over-smoothing 분석
        │
2021 ─ Jumper et al.: AlphaFold 2 (paper #37, attention + graph reasoning)
        │
2022+─ GraphCast (DeepMind, 날씨), PinSage (Pinterest)
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **#6 Rumelhart et al. (1986) — Backpropagation** | GCN 학습은 backpropagation의 표준 적용 / GCN training is standard backprop | High — propagation rule을 통한 chain rule 적용 / Foundation for training |
| **#9 Hochreiter & Schmidhuber (1997) — LSTM** | LSTM이 시간축에 propagation, GCN이 그래프 축에 propagation / LSTM propagates over time, GCN over graph | Medium — sequence model의 graph 일반화 관점 / Conceptual generalization |
| **#13 Krizhevsky et al. (2012) — AlexNet** | CNN의 weight sharing 아이디어를 그래프로 확장 / Generalizes CNN weight sharing to graphs | High — "convolution"의 의미를 일반화 / Generalizes the very notion of convolution |
| **#17 Bahdanau et al. (2014) — Attention** | GAT (2018)에서 GCN 위에 attention을 얹어 일반화 / Direct precursor to GAT | High — 후속 연구의 핵심 결합 / Combined in subsequent GAT work |
| **#18 Kingma & Ba (2014) — Adam** | GCN 실험에서 Adam optimizer 사용 (lr=0.01) / Used as the optimizer in all experiments | Medium — 실험 도구 / Practical tool |
| **#19 Ioffe & Szegedy (2015) — Batch Norm** | Renormalization trick은 정규화의 graph 버전 / Renormalization trick is a graph analogue of normalization | Medium — 유사한 안정화 철학 / Similar stabilization philosophy |
| **#20 He et al. (2015) — ResNet** | Appendix B에서 deeper GCN을 위해 residual connection 도입 / Residual connections used for deeper GCN in Appendix B | High — 직접 인용, 깊은 모델 학습 가능 / Directly cited for deeper models |
| **#25 Vaswani et al. (2017) — Transformer** | GAT를 거쳐 Transformer와 GNN의 결합이 활발 / Transformer + GNN fusion is a major research thread | High — 현대 architecture의 양대 산맥 / Two pillars of modern architectures |
| **#37 Jumper et al. (2021) — AlphaFold 2** | 단백질을 그래프로 보고 attention + graph reasoning 결합 / Treats proteins as graphs, combines attention and graph reasoning | High — GNN의 가장 영향력 있는 응용 / Most impactful GNN application |

---

## 7. References / 참고문헌

### Primary paper / 본 논문
- Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. *ICLR 2017*. arXiv:1609.02907.
- Code: https://github.com/tkipf/gcn
- Blog: http://tkipf.github.io/graph-convolutional-networks/

### Direct predecessors / 직접 선조
- Bruna, J., Zaremba, W., Szlam, A., & LeCun, Y. (2014). Spectral networks and locally connected networks on graphs. *ICLR 2014*.
- Defferrard, M., Bresson, X., & Vandergheynst, P. (2016). Convolutional neural networks on graphs with fast localized spectral filtering. *NIPS 2016*.
- Hammond, D. K., Vandergheynst, P., & Gribonval, R. (2011). Wavelets on graphs via spectral graph theory. *Applied and Computational Harmonic Analysis*, 30(2), 129–150.

### Graph-based semi-supervised learning baselines / 그래프 준지도 학습 기준선
- Belkin, M., Niyogi, P., & Sindhwani, V. (2006). Manifold regularization. *JMLR*, 7, 2399–2434.
- Perozzi, B., Al-Rfou, R., & Skiena, S. (2014). DeepWalk: Online learning of social representations. *KDD 2014*.
- Weston, J., Ratle, F., Mobahi, H., & Collobert, R. (2012). Deep learning via semi-supervised embedding. *Neural Networks: Tricks of the Trade*, 639–655.
- Yang, Z., Cohen, W., & Salakhutdinov, R. (2016). Revisiting semi-supervised learning with graph embeddings (Planetoid). *ICML 2016*.
- Zhu, X., Ghahramani, Z., & Lafferty, J. (2003). Semi-supervised learning using Gaussian fields and harmonic functions. *ICML 2003*.

### Neural networks on graphs / 그래프 위 신경망
- Atwood, J., & Towsley, D. (2016). Diffusion-convolutional neural networks. *NIPS 2016*.
- Duvenaud, D. K., et al. (2015). Convolutional networks on graphs for learning molecular fingerprints. *NIPS 2015*.
- Gori, M., Monfardini, G., & Scarselli, F. (2005). A new model for learning in graph domains. *IJCNN 2005*.
- Li, Y., Tarlow, D., Brockschmidt, M., & Zemel, R. (2016). Gated graph sequence neural networks. *ICLR 2016*.
- Scarselli, F., Gori, M., Tsoi, A. C., Hagenbuchner, M., & Monfardini, G. (2009). The graph neural network model. *IEEE TNN*, 20(1), 61–80.

### Foundations / 기초
- Adam optimizer: Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR 2015*.
- Dropout: Srivastava, N., et al. (2014). Dropout. *JMLR*, 15(1), 1929–1958.
- ResNet: He, K., et al. (2016). Deep residual learning for image recognition. *CVPR 2016*.
- t-SNE: van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. *JMLR*, 9, 2579–2605.
- Weisfeiler-Lehman: Weisfeiler, B., & Lehman, A. A. (1968). A reduction of a graph to a canonical form. *Nauchno-Technicheskaya Informatsia*, 2(9), 12–16.

### Datasets / 데이터셋
- Citation networks (Cora, Citeseer, Pubmed): Sen, P., et al. (2008). Collective classification in network data. *AI Magazine*, 29(3), 93.
- NELL: Carlson, A., et al. (2010). Toward an architecture for never-ending language learning. *AAAI 2010*.
- Karate club: Zachary, W. W. (1977). An information flow model for conflict and fission in small groups. *Journal of Anthropological Research*, 452–473.
