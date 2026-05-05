# Pre-reading Briefing: A Fast Learning Algorithm for Deep Belief Nets (Hinton, Osindero & Teh, 2006)
# 사전 읽기 브리핑: 심층 신뢰 네트워크의 빠른 학습 알고리즘 (Hinton, Osindero & Teh, 2006)

---

## 핵심 기여 / Core Contribution

이 논문은 **딥러닝 혁명의 기폭제**로 널리 인정받는 논문입니다. 2006년 이전까지 deep neural network(여러 은닉층을 가진 네트워크)는 학습이 매우 어렵다고 여겨졌습니다 — backpropagation이 깊은 네트워크에서 vanishing gradient 문제로 실패했기 때문입니다. Hinton은 이 문제를 **계층별 비지도 사전학습(layer-wise unsupervised pre-training)**으로 해결합니다. 핵심 아이디어는: ① **Restricted Boltzmann Machine (RBM)**을 하나씩 쌓아가며 각 층을 비지도로 사전학습하고 (greedy layer-wise training), ② 이렇게 초기화된 가중치를 시작점으로 전체 네트워크를 supervised fine-tuning합니다. 이 방법으로 MNIST에서 1.25%의 오류율을 달성 — 당시 SVM(1.4%)과 backprop 네트워크(1.5%)보다 우수한 성능입니다. 또한 이 모델이 **생성 모델(generative model)**이라는 점이 중요합니다 — 학습된 네트워크에서 새로운 이미지를 생성할 수 있으며, 네트워크 내부 표현을 시각적으로 해석할 수 있습니다.

This paper is widely recognized as the **catalyst of the deep learning revolution**. Before 2006, deep neural networks (with multiple hidden layers) were considered extremely difficult to train — backpropagation failed in deep networks due to the vanishing gradient problem. Hinton solves this with **layer-wise unsupervised pre-training**: ① Stack **Restricted Boltzmann Machines (RBMs)** one at a time, pre-training each layer unsupervised (greedy layer-wise training), then ② use these initialized weights as a starting point for supervised fine-tuning of the entire network. This achieves 1.25% error on MNIST — better than SVM (1.4%) and backprop networks (1.5%) at the time. Crucially, the model is a **generative model** — it can generate new images from the learned network and allows visual interpretation of internal representations.

---

## 역사적 맥락 / Historical Context

| 연도 / Year | 업적 / Milestone | 관련성 / Relevance |
|---|---|---|
| 1986 | Rumelhart, Hinton & Williams — Backpropagation (#6) | Deep network 학습의 기초이지만, 깊은 네트워크에서 vanishing gradient 문제 / Foundation for training, but vanishing gradient in deep networks |
| 1982 | Hopfield — Hopfield Network (#5) | 에너지 기반 연상 기억 — RBM의 이론적 배경 / Energy-based associative memory — theoretical basis for RBM |
| 1985 | Hinton & Sejnowski — Boltzmann Machine | RBM의 직접적 선조. 완전 연결 BM은 학습이 너무 느림 / Direct ancestor of RBM. Fully-connected BM too slow to train |
| 1995 | Hinton et al. — Wake-Sleep Algorithm | 생성 모델의 계층별 학습 시도. "모드 평균" 문제 존재 / Attempt at layer-wise learning of generative models. "Mode-averaging" problem |
| 1998 | LeCun et al. — LeNet-5 (#10) | CNN으로 이미지 인식 성공, 하지만 구조 지식에 의존 / CNN success in image recognition, but relied on structural knowledge |
| 2002 | Hinton — Contrastive Divergence (CD) | RBM의 빠른 근사 학습법 — 이 논문의 핵심 도구 / Fast approximate learning for RBMs — key tool of this paper |
| **2006** | **Hinton, Osindero & Teh — Deep Belief Nets** | **이 논문: 계층별 사전학습으로 deep network 학습 가능을 증명** / **This paper: proves deep networks can be trained via layer-wise pre-training** |
| 2006 | Hinton & Salakhutdinov — Deep Autoencoders | 같은 아이디어를 autoencoder에 적용, *Science*지 게재 / Same idea applied to autoencoders, published in *Science* |

**1990년대–2000년대 초반의 AI 겨울(2차)**: 1~2층 neural network의 한계가 드러나고, SVM과 Random Forest 등 "얕은" 모델이 실용적으로 우세했습니다. "Deep neural network는 학습 불가능"이 학계의 통설이었습니다. Hinton은 이 통설을 깨뜨린 것입니다.

**The 2nd AI Winter (1990s–early 2000s)**: Limitations of 1–2 layer neural networks were apparent, and "shallow" models like SVM and Random Forest dominated in practice. "Deep neural networks cannot be trained" was the academic consensus. Hinton shattered this consensus.

---

## 필요한 배경 지식 / Prerequisites

### 1. Energy-Based Models / 에너지 기반 모델

물리학의 Boltzmann 분포에서 영감을 받은 모델입니다. 시스템의 상태 $\mathbf{s}$에 **에너지** $E(\mathbf{s})$를 할당하고, 확률을 다음과 같이 정의합니다:

Models inspired by the Boltzmann distribution in physics. An **energy** $E(\mathbf{s})$ is assigned to each state $\mathbf{s}$, and probabilities are defined as:

$$P(\mathbf{s}) = \frac{e^{-E(\mathbf{s})}}{Z}, \quad Z = \sum_{\mathbf{s}} e^{-E(\mathbf{s})}$$

- **낮은 에너지** → 높은 확률 (선호되는 상태) / Low energy → high probability (preferred state)
- **높은 에너지** → 낮은 확률 (드문 상태) / High energy → low probability (rare state)
- $Z$: partition function (정규화 상수) — 모든 상태의 에너지를 합산. 계산이 매우 어려움 / normalization constant — sums over all states, extremely hard to compute

### 2. Sigmoid / Logistic Function

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

뉴런의 활성화 확률을 결정합니다. 입력의 가중합이 크면 → 활성화 확률 높음.

Determines the activation probability of a neuron. Large weighted sum of inputs → high activation probability.

### 3. Maximum Likelihood Learning / 최대 우도 학습

모델의 파라미터 $\theta$를 조정하여 훈련 데이터의 확률 $P(\text{data}|\theta)$를 최대화합니다. 로그를 취하면:

Adjust model parameters $\theta$ to maximize the probability of training data $P(\text{data}|\theta)$. Taking the log:

$$\theta^* = \arg\max_\theta \sum_{\text{data}} \log P(\mathbf{v}|\theta)$$

### 4. Gibbs Sampling

고차원 확률 분포에서 직접 샘플링이 어려울 때, **조건부 분포에서 번갈아 샘플링**하여 근사합니다:

When direct sampling from a high-dimensional distribution is difficult, approximate by **alternately sampling from conditional distributions**:

$$\mathbf{h} \sim P(\mathbf{h}|\mathbf{v}) \quad \to \quad \mathbf{v}' \sim P(\mathbf{v}|\mathbf{h}) \quad \to \quad \mathbf{h}' \sim P(\mathbf{h}|\mathbf{v}') \quad \to \quad \cdots$$

충분히 반복하면 정상 분포(equilibrium distribution)로 수렴합니다.

With enough iterations, converges to the equilibrium distribution.

### 5. KL Divergence (Kullback-Leibler Divergence)

두 확률 분포 사이의 "거리" (정확히는 비대칭적 차이):

The "distance" between two probability distributions (asymmetric):

$$KL(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)} \geq 0$$

$KL = 0$이면 두 분포가 동일합니다. 학습의 목표는 모델 분포 $Q$를 데이터 분포 $P$에 가깝게 만드는 것입니다.

$KL = 0$ when two distributions are identical. The goal of learning is to make model distribution $Q$ close to data distribution $P$.

### 6. 선행 논문 / Prior Papers

- **#5 Hopfield (1982)**: 에너지 기반 네트워크, 연상 기억 / Energy-based networks, associative memory
- **#6 Rumelhart, Hinton & Williams (1986)**: Backpropagation — 이 논문에서 fine-tuning에 사용 / Used for fine-tuning in this paper

---

## 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Description |
|---|---|
| **Belief Network** | 변수 간 조건부 의존 관계를 방향 비순환 그래프(DAG)로 표현한 확률 모델. 각 노드의 확률은 부모 노드들에 의해 결정됩니다. / A probabilistic model representing conditional dependencies as a directed acyclic graph (DAG). Each node's probability is determined by its parents. |
| **Restricted Boltzmann Machine (RBM)** | 가시 유닛(visible)과 은닉 유닛(hidden)의 **두 층**으로만 구성된 에너지 기반 모델. "Restricted"는 같은 층 내 유닛 간 연결이 **없다**는 뜻 — 이 덕분에 효율적 학습이 가능합니다. / An energy-based model with only **two layers**: visible and hidden units. "Restricted" means **no connections** within the same layer — enabling efficient learning. |
| **Contrastive Divergence (CD)** | RBM 학습을 위한 빠른 근사 알고리즘. 전체 Gibbs 체인을 수렴까지 돌리는 대신, 단 1~몇 스텝만 돌려서 gradient를 근사합니다. / A fast approximate algorithm for RBM learning. Instead of running the full Gibbs chain to convergence, approximates the gradient with just 1–few steps. |
| **Greedy Layer-wise Pre-training** | Deep network를 한 번에 학습하지 않고, RBM을 **한 층씩 아래에서 위로** 쌓아가며 학습하는 전략. 각 RBM이 아래 층의 출력을 입력으로 받아 새로운 표현을 학습합니다. / A strategy of training a deep network **one layer at a time from bottom to top**, rather than all at once. Each RBM takes the output of the layer below as input and learns a new representation. |
| **Complementary Prior** | Likelihood의 "explaining away" 효과를 정확히 상쇄하여, posterior가 **factorial** (완전 분해 가능)이 되게 하는 prior. 이것이 tied weights의 무한 모델에서 자연스럽게 나타남을 보입니다. / A prior that exactly cancels the "explaining away" effect of the likelihood, making the posterior **factorial** (fully factorizable). Shown to arise naturally in the infinite tied-weights model. |
| **Explaining Away** | Belief network에서 공통 효과를 관찰했을 때, 원인들이 사후적으로 반상관(anti-correlated)되는 현상. 예: "집이 흔들림"을 관찰 → "지진"과 "트럭 충돌" 중 하나가 확인되면 다른 하나의 확률이 줄어듦. 이것이 deep belief net의 추론을 어렵게 만드는 핵심 원인입니다. / In belief networks, observing a common effect makes causes anti-correlated. E.g., "house shakes" observed → confirming "earthquake" reduces probability of "truck hit." This is the key reason inference in deep belief nets is hard. |
| **Up-Down Algorithm** | Greedy pre-training 후 전체 네트워크를 fine-tune하는 알고리즘. Wake-sleep algorithm의 contrastive 변형으로, "mode averaging" 문제를 회피합니다. / Algorithm for fine-tuning the entire network after greedy pre-training. A contrastive variant of the wake-sleep algorithm that avoids the "mode averaging" problem. |
| **Associative Memory** | 상위 두 층이 형성하는 비방향(undirected) 그래프 모델. Hopfield network (#5)와 유사하게 연상 기억으로 작동합니다. / The undirected graphical model formed by the top two layers. Works as associative memory similar to Hopfield networks (#5). |

---

## 수식 미리보기 / Equations Preview

### 1. Logistic Belief Network의 조건부 확률 / Conditional Probability

$$p(s_i = 1) = \frac{1}{1 + \exp(-b_i - \sum_j s_j w_{ij})} \tag{1}$$

- $s_i$: 유닛 $i$의 이진 상태 (0 또는 1) / Binary state of unit $i$ (0 or 1)
- $b_i$: 유닛 $i$의 bias
- $w_{ij}$: 유닛 $j$에서 $i$로의 가중치 / Weight from unit $j$ to $i$
- **직관 / Intuition**: 부모 유닛들의 가중합이 크면 활성화 확률이 높음. 이것이 각 뉴런의 기본 작동 원리입니다. / Larger weighted sum from parent units → higher activation probability. This is the basic operating principle of each neuron.

### 2. RBM의 에너지 함수 / RBM Energy Function

$$E(\mathbf{v}, \mathbf{h}) = -\sum_i a_i v_i - \sum_j b_j h_j - \sum_{i,j} v_i w_{ij} h_j$$

- $\mathbf{v}$: visible units, $\mathbf{h}$: hidden units
- $a_i, b_j$: biases, $w_{ij}$: 가중치 / weights
- **직관 / Intuition**: 에너지가 낮을수록 그 상태가 더 가능성 높음. 가중치 $w_{ij}$가 크고 $v_i$와 $h_j$가 모두 활성화되면 에너지가 크게 감소 → 그 패턴이 학습됨. / Lower energy means more probable state. Large $w_{ij}$ with both $v_i$ and $h_j$ active greatly decreases energy → that pattern is learned.

### 3. Contrastive Divergence 학습 규칙 / CD Learning Rule

$$\Delta w_{ij} \propto \langle v_i h_j \rangle_{\text{data}} - \langle v_i h_j \rangle_{\text{recon}}$$

- $\langle v_i h_j \rangle_{\text{data}}$: 데이터를 clamp했을 때 $v_i$와 $h_j$의 상관관계 ("positive phase") / Correlation when data is clamped
- $\langle v_i h_j \rangle_{\text{recon}}$: 1 스텝 Gibbs 후의 reconstruction에서의 상관관계 ("negative phase") / Correlation after 1-step Gibbs reconstruction
- **직관 / Intuition**: 데이터에서의 상관관계를 높이고, 모델이 "꿈꾸는" 상관관계를 줄임. 데이터의 패턴을 더 잘 포착하도록 가중치를 조정합니다. / Increase data correlations, decrease "dreamed" correlations. Adjusts weights to better capture data patterns.

### 4. Variational Bound / 변분 하한

$$\log p(\mathbf{v}) \geq \sum_{\mathbf{h}} Q(\mathbf{h}|\mathbf{v}) \left[\log p(\mathbf{h}) + \log p(\mathbf{v}|\mathbf{h})\right] - \sum_{\mathbf{h}} Q(\mathbf{h}|\mathbf{v}) \log Q(\mathbf{h}|\mathbf{v}) \tag{8}$$

- 좌변: 데이터의 로그 확률 (최대화하고 싶은 것) / Left: log probability of data (what we want to maximize)
- 우변: 계산 가능한 하한 / Right: computable lower bound
- **핵심 / Key**: greedy 알고리즘이 새 층을 추가할 때마다 이 하한이 **반드시 개선**됨을 증명 → 깊을수록 무조건 좋거나 최소한 나빠지지 않음. / The greedy algorithm proves this bound **always improves** when adding a new layer → deeper is always better or at least never worse.

---

## 논문의 구조 / Paper Structure

| 섹션 / Section | 내용 / Content |
|---|---|
| §1 | 소개: 왜 deep network가 어려운지, 이 논문의 접근 / Introduction: why deep networks are hard, this paper's approach |
| §2 | Complementary priors: explaining away 문제의 해결 / Solving the explaining away problem |
| §3 | RBM과 무한 belief net의 등가성, CD 학습 / RBM equivalence to infinite belief nets, CD learning |
| §4 | Greedy layer-wise 알고리즘 + variational bound 증명 / Greedy algorithm + variational bound proof |
| §5 | Up-down fine-tuning 알고리즘 / Up-down fine-tuning algorithm |
| §6 | MNIST 실험 결과 (1.25% 오류) / MNIST experiments (1.25% error) |
| §7 | 네트워크의 "마음 들여다보기" — 이미지 생성 / "Looking into the mind" — image generation |
| §8 | 결론 / Conclusion |
| App. A | Complementary priors의 수학적 증명 / Mathematical proof of complementary priors |
| App. B | Up-down 알고리즘의 MATLAB 의사 코드 / MATLAB pseudocode for up-down algorithm |

---

## 읽기 팁 / Reading Tips

1. **Figure 2 (explaining away)**를 먼저 이해하세요. "지진"과 "트럭 충돌"의 예시가 전체 논문의 동기를 설명합니다. explaining away가 왜 deep network의 inference를 어렵게 만드는지 파악하면, complementary prior의 필요성이 자연스럽게 따라옵니다.
   Understand Figure 2 (explaining away) first. The "earthquake" and "truck hit" example explains the entire paper's motivation.

2. **Figure 3과 4**가 RBM과 CD를 시각적으로 보여줍니다. Figure 3의 tied weights 구조와 Figure 4의 Gibbs sampling 과정을 연결하여 이해하세요.
   Figures 3 and 4 visually show RBM and CD. Connect the tied weight structure in Figure 3 with the Gibbs sampling process in Figure 4.

3. **§4의 greedy algorithm**이 논문의 실질적 핵심입니다. "각 층을 추가할 때마다 variational bound가 개선된다"는 증명이 전체 방법론을 정당화합니다. 수식 (7)과 (8)에 집중하세요.
   The greedy algorithm in §4 is the practical core. Focus on equations (7) and (8) — the proof that adding each layer improves the variational bound justifies the entire methodology.

4. **Figure 8, 9** (§7)는 가장 인상적인 결과입니다. 네트워크가 학습한 것을 "상상"하여 이미지를 생성하는 것을 보여줍니다.
   Figures 8, 9 (§7) are the most impressive results — showing the network "imagining" by generating images from what it learned.

---

## Q&A

### Q1: Gibbs Sampling 상세 설명 / Detailed Explanation of Gibbs Sampling

#### 문제 상황 / The Problem

고차원 확률 분포 $P(x_1, x_2, \ldots, x_n)$에서 샘플을 뽑고 싶은데, **직접 샘플링이 불가능**한 경우가 많습니다. 예를 들어 RBM에서 $P(\mathbf{v}, \mathbf{h})$는 partition function $Z = \sum_{\mathbf{v},\mathbf{h}} e^{-E(\mathbf{v},\mathbf{h})}$ 때문에 직접 계산이 불가능합니다 — 이진 유닛이 1000개면 $2^{1000}$가지 상태를 합산해야 합니다.

When we want to sample from a high-dimensional distribution $P(x_1, x_2, \ldots, x_n)$, **direct sampling is often impossible**. For example, in RBMs, $P(\mathbf{v}, \mathbf{h})$ is intractable due to the partition function $Z = \sum_{\mathbf{v},\mathbf{h}} e^{-E(\mathbf{v},\mathbf{h})}$ — with 1000 binary units, we'd need to sum over $2^{1000}$ states.

#### 핵심 아이디어 / Core Idea

전체 분포에서 직접 뽑지 못하더라도, **각 변수의 조건부 분포**에서는 쉽게 뽑을 수 있는 경우가 많습니다. Gibbs sampling은 한 번에 하나의 변수씩 (또는 한 그룹씩) 조건부 분포에서 번갈아 샘플링합니다.

Even if direct sampling is impossible, sampling from each variable's **conditional distribution** is often easy. Gibbs sampling alternately samples from conditional distributions, one variable (or group) at a time.

2변수 예시: $P(x, y)$에서 샘플을 뽑고 싶다면:

2-variable example: to sample from $P(x, y)$:

```
초기값/Initial: x₀ = 임의값/arbitrary

반복/Iterate:
  y₁ ~ P(y | x₀)     ← x를 고정하고 y를 뽑음 / fix x, sample y
  x₁ ~ P(x | y₁)     ← y를 고정하고 x를 뽑음 / fix y, sample x
  y₂ ~ P(y | x₁)
  x₂ ~ P(x | y₂)
  ...

충분히 반복하면 (x_t, y_t)가 P(x, y)에서의 샘플처럼 됨
After enough iterations, (x_t, y_t) becomes a sample from P(x, y)
```

#### RBM에서의 Gibbs Sampling / Gibbs Sampling in RBM

RBM의 특별한 구조 덕분에 Gibbs sampling이 매우 효율적입니다. 같은 층 내 유닛 간 연결이 없으므로("Restricted"), **한 층의 모든 유닛을 동시에(병렬로)** 샘플링할 수 있습니다:

Thanks to RBM's special structure, Gibbs sampling is very efficient. Since there are no within-layer connections ("Restricted"), **all units in a layer can be sampled simultaneously (in parallel)**:

$$P(h_j = 1 | \mathbf{v}) = \sigma\left(b_j + \sum_i v_i w_{ij}\right)$$
$$P(v_i = 1 | \mathbf{h}) = \sigma\left(a_i + \sum_j h_j w_{ij}\right)$$

```
v⁰ (데이터/data)               ← 시작: 실제 데이터를 visible에 clamp / Start: clamp real data
  ↓ P(h|v)  [모든 h를 동시에 / all h simultaneously]
h⁰ (hidden 샘플/sample)
  ↓ P(v|h)  [모든 v를 동시에 / all v simultaneously]
v¹ (reconstruction)            ← 1스텝 후의 "꿈" / "dream" after 1 step
  ↓ P(h|v)
h¹
  ...
v∞ (평형 분포에서의 샘플)       ← 충분히 반복하면 모델의 분포에서의 샘플
                                  After enough iterations, sample from model distribution
```

**Contrastive Divergence (CD-1)**의 핵심 트릭: $v^\infty$까지 기다리지 않고 **딱 1 스텝만**(v⁰ → h⁰ → v¹) 돌려서 gradient를 근사합니다. 놀랍게도 이것만으로도 충분히 잘 작동합니다!

**Key trick of CD-1**: Instead of waiting until $v^\infty$, approximate the gradient with **just 1 step** (v⁰ → h⁰ → v¹). Surprisingly, this works well enough!

$$\Delta w_{ij} \propto \underbrace{\langle v_i^0 h_j^0 \rangle}_{\text{데이터의 상관관계 / data correlation}} - \underbrace{\langle v_i^1 h_j^1 \rangle}_{\text{1스텝 reconstruction의 상관관계 / reconstruction correlation}}$$

---

### Q2: KL Divergence 상세 설명 및 VAE와의 연결 / KL Divergence and Connection to VAE

#### 직관적 이해 / Intuitive Understanding

KL divergence는 **"한 확률 분포가 다른 분포와 얼마나 다른가"**를 측정합니다. "정보 손실량"으로 이해할 수 있습니다 — 진짜 분포 $P$ 대신 모델 분포 $Q$를 사용하면 얼마나 비효율적인가?

KL divergence measures **"how different one probability distribution is from another."** It can be understood as "information loss" — how inefficient is it to use model distribution $Q$ instead of true distribution $P$?

$$KL(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)} = E_{x \sim P}\left[\log \frac{P(x)}{Q(x)}\right]$$

#### 핵심 성질 / Key Properties

1. **항상 $\geq 0$** / Always $\geq 0$: $KL(P \| Q) \geq 0$ (Gibbs' inequality)
2. **$= 0$ iff $P = Q$**: 두 분포가 완전히 같을 때만 0 / Zero only when distributions are identical
3. **비대칭 / Asymmetric**: $KL(P \| Q) \neq KL(Q \| P)$ — 엄밀한 "거리"는 아님 / Not a true "distance"

#### 숫자 예시 / Numerical Example

동전 던지기. 진짜 분포 $P$: 앞=0.7, 뒤=0.3

Coin flip. True distribution $P$: heads=0.7, tails=0.3

**모델 A** ($Q_A$): 앞=0.6, 뒤=0.4 (꽤 비슷/quite similar)

$$KL(P \| Q_A) = 0.7 \ln\frac{0.7}{0.6} + 0.3 \ln\frac{0.3}{0.4} = 0.021$$

**모델 B** ($Q_B$): 앞=0.5, 뒤=0.5 (공정한 동전/fair coin)

$$KL(P \| Q_B) = 0.7 \ln\frac{0.7}{0.5} + 0.3 \ln\frac{0.3}{0.5} = 0.082$$

**모델 C** ($Q_C$): 앞=0.9, 뒤=0.1 (매우 편향/highly biased)

$$KL(P \| Q_C) = 0.7 \ln\frac{0.7}{0.9} + 0.3 \ln\frac{0.3}{0.1} = 0.154$$

→ 모델 A가 가장 가깝고(KL=0.021), 모델 C가 가장 멀다(KL=0.154). / Model A is closest, Model C is farthest.

#### 이 논문에서의 역할 / Role in This Paper

Contrastive Divergence는 본질적으로 **두 KL divergence의 차이**를 최소화합니다 (Eq. 6):

Contrastive Divergence essentially minimizes the **difference of two KL divergences** (Eq. 6):

$$KL(P^0 \| P_\theta^\infty) - KL(P_\theta^n \| P_\theta^\infty) \tag{6}$$

- $P^0$: 데이터 분포 / data distribution
- $P_\theta^\infty$: 모델의 평형 분포 / model's equilibrium distribution
- $P_\theta^n$: $n$ 스텝 Gibbs 후의 분포 / distribution after $n$ Gibbs steps

#### VAE와의 연결 — 네, 같은 KL divergence입니다! / Connection to VAE — Yes, the Same KL Divergence!

**VAE (Variational Autoencoder, #15 Kingma & Welling, 2013)**에서 사용되는 KL divergence와 정확히 같은 개념입니다. VAE의 수학적 기반은 이 논문(Hinton 2006)의 variational bound와 직접적으로 연결됩니다.

The KL divergence used in **VAE (#15 Kingma & Welling, 2013)** is exactly the same concept. VAE's mathematical foundation connects directly to this paper's variational bound.

**이 논문(2006)의 Variational Bound / This Paper's Variational Bound**:

$$\log p(\mathbf{v}) \geq E_{Q}[\log p(\mathbf{v}|\mathbf{h})] - KL(Q(\mathbf{h}|\mathbf{v}) \| p(\mathbf{h}))$$

**VAE(2013)의 ELBO / VAE's ELBO**:

$$\log p(\mathbf{x}) \geq \underbrace{E_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})]}_{\text{Reconstruction}} - \underbrace{KL(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))}_{\text{KL regularization}}$$

**비교 / Comparison**:

| | Hinton (2006) | VAE (2013) |
|---|---|---|
| 관측 변수 / Observed | $\mathbf{v}$ (visible) | $\mathbf{x}$ (data) |
| 잠재 변수 / Latent | $\mathbf{h}$ (hidden, 이진/binary) | $\mathbf{z}$ (latent code, 연속/continuous) |
| 근사 posterior | $Q(\mathbf{h}|\mathbf{v})$ (factorial) | $q_\phi(\mathbf{z}|\mathbf{x})$ (encoder 신경망/neural net) |
| 생성 모델 / Generative | $p(\mathbf{v}|\mathbf{h})$ (belief net) | $p_\theta(\mathbf{x}|\mathbf{z})$ (decoder 신경망/neural net) |
| 잠재 prior | 학습된 RBM / Learned RBM | $p(\mathbf{z}) = \mathcal{N}(0, I)$ |
| KL 역할 / KL role | 이론적 분석 + greedy bound / Theoretical analysis | **손실 함수에 직접 포함 / Directly in loss function** |
| 학습 / Training | CD + greedy layer-wise | Reparameterization trick + SGD |

**지적 계보 / Intellectual Lineage**:

```
Hinton (2006): variational bound로 deep generative model 정당화
               Justified deep generative models via variational bound
       │
       ├── "variational inference를 deep learning에 적용할 수 있다"
       │    "Variational inference can be applied to deep learning"
       ▼
Kingma & Welling (2013, VAE): 
   같은 variational bound를 신경망 encoder/decoder로 end-to-end 학습
   Same variational bound, learned end-to-end with neural net encoder/decoder
   KL(q(z|x) || p(z))를 손실 함수에 직접 포함
   KL directly included in the loss function
```

---

### Q3: Restricted Boltzmann Machine (RBM) 상세 설명 / Detailed Explanation of RBM

#### Boltzmann Machine에서 RBM으로 / From Boltzmann Machine to RBM

원래의 Boltzmann Machine (1985, Hinton & Sejnowski)은 **모든 유닛이 서로 연결**된 에너지 기반 모델입니다. 문제: 같은 층 내 유닛끼리도 연결되어 있어서 학습이 **극도로 느립니다**.

The original Boltzmann Machine (1985) has **all units connected to each other**. Problem: within-layer connections make learning **extremely slow**.

**"Restricted" = 같은 층 내 연결을 제거 / Remove within-layer connections**:

```
  h₁    h₂    h₃    h₄       hidden layer
  │╲  ╱│╲  ╱│╲  ╱│
  │ ╲╱ │ ╲╱ │ ╲╱ │          w_ij (가중치/weights)
  │ ╱╲ │ ╱╲ │ ╱╲ │
  │╱  ╲│╱  ╲│╱  ╲│
  v₁    v₂    v₃    v₄       visible layer

  ※ h₁-h₂ 연결 없음!  v₁-v₂ 연결 없음!
  ※ No h₁-h₂ connection!  No v₁-v₂ connection!
```

이 제약 덕분에 핵심적인 성질이 생깁니다:

This restriction creates key properties:

- **$\mathbf{v}$가 주어지면 모든 $h_j$가 조건부 독립** → 모든 hidden을 **동시에** 샘플링 가능 / Given $\mathbf{v}$, all $h_j$ are conditionally independent → sample all hidden **simultaneously**
- **$\mathbf{h}$가 주어지면 모든 $v_i$가 조건부 독립** → 모든 visible을 **동시에** 샘플링 가능 / Given $\mathbf{h}$, all $v_i$ are conditionally independent → sample all visible **simultaneously**

#### RBM의 수학 / Mathematics of RBM

**에너지 함수 / Energy Function**:

$$E(\mathbf{v}, \mathbf{h}) = -\sum_i a_i v_i - \sum_j b_j h_j - \sum_{i,j} v_i w_{ij} h_j$$

| 항 / Term | 의미 / Meaning |
|---|---|
| $-a_i v_i$ | visible 유닛 $i$의 bias. $a_i > 0$이면 $v_i = 1$이 선호됨 / Bias for visible unit $i$. If $a_i > 0$, $v_i = 1$ is preferred |
| $-b_j h_j$ | hidden 유닛 $j$의 bias. $b_j > 0$이면 $h_j = 1$이 선호됨 / Bias for hidden unit $j$. If $b_j > 0$, $h_j = 1$ is preferred |
| $-v_i w_{ij} h_j$ | $v_i$와 $h_j$ 사이의 상호작용. $w_{ij} > 0$이고 둘 다 1이면 에너지 감소 → 그 패턴이 선호됨 / Interaction. If $w_{ij} > 0$ and both are 1, energy decreases → that pattern is preferred |

**결합 확률 / Joint Probability**: 에너지가 낮을수록 확률이 높음 / Lower energy = higher probability

$$P(\mathbf{v}, \mathbf{h}) = \frac{1}{Z} e^{-E(\mathbf{v}, \mathbf{h})}, \quad Z = \sum_{\mathbf{v}, \mathbf{h}} e^{-E(\mathbf{v}, \mathbf{h})}$$

**조건부 확률 / Conditional Probabilities** (RBM의 핵심 연산 / Core operations of RBM):

visible → hidden (데이터를 보고 특징 추출 / extract features from data):

$$P(h_j = 1 | \mathbf{v}) = \sigma\left(b_j + \sum_i v_i w_{ij}\right)$$

hidden → visible (특징에서 데이터 복원 / reconstruct data from features):

$$P(v_i = 1 | \mathbf{h}) = \sigma\left(a_i + \sum_j h_j w_{ij}\right)$$

#### 숫자 예시 / Numerical Example

3개의 visible, 2개의 hidden으로 된 작은 RBM:

A small RBM with 3 visible, 2 hidden units:

$$W = \begin{pmatrix} 2 & -1 \\ 2 & -1 \\ -2 & 1 \end{pmatrix}, \quad \mathbf{a} = (0, 0, 0), \quad \mathbf{b} = (-1, -1)$$

**Forward pass**: $\mathbf{v} = (1, 1, 0)$ → $P(\mathbf{h}|\mathbf{v})$

$h_1$: $\sigma(b_1 + v_1 w_{11} + v_2 w_{21} + v_3 w_{31}) = \sigma(-1 + 2 + 2 + 0) = \sigma(3) \approx 0.95$

$h_2$: $\sigma(b_2 + v_1 w_{12} + v_2 w_{22} + v_3 w_{32}) = \sigma(-1 - 1 - 1 + 0) = \sigma(-3) \approx 0.05$

**해석 / Interpretation**: 입력 $(1, 1, 0)$을 보면 $h_1$이 강하게 활성화되고 $h_2$는 거의 비활성화. $h_1$은 "처음 두 pixel이 켜진 패턴"을 감지하는 특징 검출기. / $h_1$ strongly activates for input $(1,1,0)$. It's a feature detector for "first two pixels on."

**Backward pass**: $\mathbf{h} = (1, 0)$ → $P(\mathbf{v}|\mathbf{h})$

$v_1$: $\sigma(0 + 1 \cdot 2 + 0 \cdot (-1)) = \sigma(2) \approx 0.88$

$v_2$: $\sigma(0 + 1 \cdot 2 + 0 \cdot (-1)) = \sigma(2) \approx 0.88$

$v_3$: $\sigma(0 + 1 \cdot (-2) + 0 \cdot 1) = \sigma(-2) \approx 0.12$

**해석 / Interpretation**: $h_1$이 활성화되면 $(1, 1, 0)$과 유사한 패턴을 재구성합니다 — 원래 입력이 복원! / When $h_1$ activates, it reconstructs a pattern similar to $(1, 1, 0)$ — original input recovered!

#### RBM의 학습: Contrastive Divergence (CD-1) / Learning: CD-1

RBM 학습의 목표는 훈련 데이터의 확률 $P(\mathbf{v})$를 최대화하는 것입니다.

The goal of RBM learning is to maximize the probability of training data $P(\mathbf{v})$.

```
Step 1 (Positive phase):
  v⁰ = 훈련 데이터 / training data
  h⁰ ~ P(h|v⁰)        ← hidden 활성화 확률 계산 + 샘플링 / compute activation probabilities + sample
  positive = v⁰ᵀ h⁰    ← 데이터의 상관관계 / data correlations

Step 2 (Negative phase - 1 step reconstruction):  
  v¹ ~ P(v|h⁰)         ← hidden에서 visible 재구성 / reconstruct visible from hidden
  h¹ ~ P(h|v¹)         ← 재구성에서 다시 hidden / hidden from reconstruction
  negative = v¹ᵀ h¹     ← 재구성의 상관관계 / reconstruction correlations

Step 3 (가중치 업데이트 / Weight update):
  ΔW = η (positive - negative)
  Δa = η (v⁰ - v¹)
  Δb = η (h⁰ - h¹)
```

**직관 / Intuition**: "데이터의 패턴을 더 선호하고, 모델이 꿈꾸는 패턴을 덜 선호하도록" 가중치를 조정합니다. / Adjust weights to "prefer data patterns more, model's dreamed patterns less."

#### RBM을 쌓아서 Deep Belief Net 만들기 / Stacking RBMs to Build a Deep Belief Net

이것이 논문의 핵심 알고리즘입니다: / This is the paper's core algorithm:

```
=== Greedy Layer-wise Pre-training ===

Step 1: 첫 번째 RBM 학습 / Train first RBM
  데이터/Data (784 pixels) ↔ Hidden 1 (500 units)
  → CD로 W₁ 학습 / Learn W₁ via CD

Step 2: W₁ 고정, 두 번째 RBM 학습 / Freeze W₁, train second RBM
  H₁ 활성화를 "새 데이터"로 사용 / Use H₁ activations as "new data"
  Hidden 1 (500) ↔ Hidden 2 (500)
  → CD로 W₂ 학습 / Learn W₂ via CD

Step 3: W₂ 고정, 세 번째 RBM 학습 / Freeze W₂, train third RBM
  Hidden 2 (500) ↔ Hidden 3 (2000)
  → CD로 W₃ 학습 / Learn W₃ via CD

최종 구조 / Final structure:

  2000 units (top) ←→ labels (10 units)  ← 연상 기억/associative memory (undirected)
       ↕ W₃
  500 units                               ← directed (위→아래/top→down)
       ↕ W₂
  500 units  
       ↕ W₁
  784 pixels (bottom)
```

**핵심 보장 / Key guarantee**: 각 층을 추가할 때마다 데이터의 로그 확률에 대한 variational bound가 **반드시 개선되거나 최소한 유지**됩니다 (Eq. 8). / Adding each layer **always improves or at least maintains** the variational bound on data log probability (Eq. 8).

#### 왜 RBM이 혁신적이었나 / Why RBM Was Revolutionary

| 이전 / Before | RBM / After |
|---|---|
| Backprop으로 deep network 직접 학습 → vanishing gradient 실패 / Direct deep training via backprop → vanishing gradient failure | RBM 층별 사전학습 → 좋은 초기값 확보 후 fine-tuning / Layer-wise pre-training → good initialization then fine-tuning |
| Random 초기화 → 나쁜 local minimum에 갇힘 / Random init → trapped in bad local minima | 사전학습된 초기화 → 좋은 local minimum 근처에서 시작 / Pre-trained init → start near good local minima |
| Supervised만 → 라벨 필요 / Supervised only → labels needed | Unsupervised 사전학습 → 라벨 없는 대량 데이터 활용 / Unsupervised pre-training → leverage unlabeled data |
| 분류만 가능 / Classification only | **생성 모델** → 새 이미지 생성, 내부 표현 해석 가능 / **Generative model** → generate images, interpret representations |

비유 / Analogy: 에베레스트 등반에서 **베이스캠프에서 바로 정상 도전**(backprop random init)하면 실패하지만, **캠프 1 → 캠프 2 → 캠프 3을 순차적으로 설치**(greedy layer-wise)하면 정상에 도달합니다. / Climbing Everest: **rushing from base camp to summit** (backprop random init) fails, but **setting up Camp 1 → Camp 2 → Camp 3 sequentially** (greedy layer-wise) reaches the summit.
