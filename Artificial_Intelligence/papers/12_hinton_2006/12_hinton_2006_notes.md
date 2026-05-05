---
title: "A Fast Learning Algorithm for Deep Belief Nets"
authors: Geoffrey E. Hinton, Simon Osindero, Yee-Whye Teh
year: 2006
journal: "Neural Computation, Vol. 18(7), pp. 1527–1554"
doi: "10.1162/neco.2006.18.7.1527"
topic: Artificial Intelligence / Deep Learning
tags: [deep belief net, restricted boltzmann machine, RBM, contrastive divergence, greedy layer-wise pre-training, generative model, unsupervised learning, complementary prior, explaining away, variational bound, wake-sleep, MNIST]
status: completed
date_started: 2026-04-12
date_completed: 2026-04-13
---

# A Fast Learning Algorithm for Deep Belief Nets
**Geoffrey E. Hinton, Simon Osindero, Yee-Whye Teh (2006)**

---

## 핵심 기여 / Core Contribution

이 논문은 **딥러닝 혁명의 기폭제**로, 2006년 이전까지 "deep neural network는 학습 불가능"이라는 학계의 통설을 깨뜨렸습니다. 핵심 기여는 세 가지입니다. 첫째, deep belief network에서 추론(inference)을 어렵게 만드는 **"explaining away" 현상**을 **complementary prior**로 해결할 수 있음을 보였으며, tied weights를 가진 무한 directed 모델이 Restricted Boltzmann Machine (RBM)과 등가임을 증명하여 효율적 학습의 이론적 기반을 마련했습니다 (§2–3). 둘째, RBM을 한 층씩 아래에서 위로 쌓아가는 **greedy layer-wise pre-training** 알고리즘을 제안하고, 각 층을 추가할 때마다 데이터의 로그 확률에 대한 variational bound가 **반드시 개선됨**을 수학적으로 증명했습니다 (§4, Eq. 7–8). 이것은 "깊을수록 무조건 좋거나 최소한 나빠지지 않는다"는 최초의 이론적 보장입니다. 셋째, 이 사전학습된 가중치를 초기값으로 사용하여 supervised fine-tuning (up-down algorithm)을 적용한 결과, MNIST에서 **1.25% 오류율**을 달성했습니다 — 당시 SVM(1.4%)과 backprop 네트워크(1.5%)보다 우수하며, 특별한 전처리나 구조 지식 없이 달성한 최고 성능입니다. 또한 이 모델이 **생성 모델(generative model)**이라는 점이 획기적입니다 — 학습된 네트워크에서 새로운 이미지를 생성하고 (Figure 8, 9), 네트워크 내부 표현을 시각적으로 해석할 수 있습니다.

This paper is the **catalyst of the deep learning revolution**, shattering the academic consensus that "deep neural networks cannot be trained." It makes three key contributions. First, it shows that the **"explaining away" phenomenon** that makes inference difficult in deep belief networks can be solved with **complementary priors**, and proves that infinite directed models with tied weights are equivalent to Restricted Boltzmann Machines (RBMs), establishing a theoretical foundation for efficient learning (§2–3). Second, it proposes a **greedy layer-wise pre-training** algorithm that stacks RBMs one layer at a time from bottom to top, and mathematically proves that the variational bound on data log-probability **always improves** with each added layer (§4, Eq. 7–8) — the first theoretical guarantee that "deeper is always better or at least never worse." Third, using these pre-trained weights as initialization for supervised fine-tuning (up-down algorithm), it achieves **1.25% error** on MNIST — better than SVM (1.4%) and backprop networks (1.5%), the best result achieved without special preprocessing or structural knowledge. Crucially, the model is a **generative model** — it can generate new images from the learned network (Figures 8, 9) and allows visual interpretation of internal representations.

---

## 읽기 노트 / Reading Notes

### §1: Introduction — 왜 Deep Network가 어려운가 / Why Deep Networks Are Hard

#### 심층 네트워크의 학습 문제 / The Training Problem of Deep Networks

논문은 "densely-connected directed belief net에서 많은 은닉층을 가질 때 학습이 어렵다"는 문제로 시작합니다. 구체적으로 두 가지 근본적 어려움을 지적합니다:

The paper begins with the problem that "learning is difficult in densely-connected, directed belief nets with many hidden layers." It identifies two fundamental difficulties:

1. **Inference의 어려움**: 데이터가 주어졌을 때 은닉 변수들의 조건부 분포(posterior) $P(\mathbf{h}|\mathbf{v})$를 계산하기가 intractable합니다. 이것은 "explaining away" 현상 때문입니다 (§2에서 상세 설명). / **Inference difficulty**: Computing the posterior $P(\mathbf{h}|\mathbf{v})$ of hidden variables given data is intractable, due to the "explaining away" phenomenon.

2. **Variational 방법의 한계**: Variational approximation은 true posterior를 더 다루기 쉬운 분포로 근사하지만, 가장 깊은 은닉층에서 prior가 독립을 가정하므로 근사가 나빠집니다. 또한 모든 파라미터를 동시에 학습해야 하므로 파라미터 수에 따라 학습 시간이 급증합니다. / **Limitations of variational methods**: Approximations worsen at the deepest layers where the prior assumes independence, and all parameters must be learned together.

#### 논문의 해결책 개요 / Overview of the Solution

Hinton은 **하이브리드 모델**(Figure 1)을 제안합니다:

Hinton proposes a **hybrid model** (Figure 1):

- **상위 두 층**: 비방향(undirected) 연결 — **연상 기억(associative memory)**를 형성. Hopfield network (#5)과 유사 / **Top two layers**: undirected connections — form **associative memory**, similar to Hopfield networks (#5)
- **나머지 하위 층**: 방향(directed) 연결 — 연상 기억의 표현을 관측 가능한 변수(픽셀)로 변환하는 **생성 경로** / **Lower layers**: directed connections — **generative pathway** converting associative memory representations to observable variables (pixels)

이 구조의 핵심 장점 7가지를 나열합니다:

The paper lists 7 key advantages of this structure:

1. 수백만 파라미터도 빠르게 학습하는 greedy 알고리즘
2. 비지도 학습이지만 라벨 포함 가능
3. Fine-tuning으로 뛰어난 판별 성능
4. 분산 표현의 해석 용이성
5. 빠르고 정확한 inference
6. 국소적(local) 학습 규칙 — pre/post-synaptic 뉴런의 상태만 필요
7. 단순한 통신 — 확률적 이진 값만 전달

---

### §2: Complementary Priors — Explaining Away 문제의 해결 / Solving the Explaining Away Problem

#### Explaining Away 현상 / The Explaining Away Phenomenon

논문의 Figure 2가 이 현상을 명쾌하게 보여줍니다. "집이 흔들린다(house jumps)"를 관측했을 때의 두 가능한 원인:

Figure 2 clearly illustrates this phenomenon. When "house jumps" is observed, two possible causes:

- **지진(earthquake)**: bias = -10 (매우 드문 사건), 집 흔들림에 +20 기여 / Very rare event, +20 contribution to house jumping
- **트럭 충돌(truck hits house)**: bias = -10 (매우 드문 사건), 집 흔들림에 +20 기여 / Very rare event, +20 contribution

"집이 흔들림"이 관측되지 않은 상태에서, 두 원인은 **독립**입니다 (각각 $P \approx e^{-10}$). 하지만 "집이 흔들림"을 관측하면:

Without observing "house jumps," the two causes are **independent** (each $P \approx e^{-10}$). But upon observing "house jumps":

- 지진이 "켜졌다"고 확인되면 → 트럭 충돌이 "꺼질" 확률이 높아짐 (이미 설명됨)
- 트럭이 "켜졌다"고 확인되면 → 지진이 "꺼질" 확률이 높아짐

→ 관측을 통해 원래 독립이던 원인들이 **사후적으로 반상관(anti-correlated)**됩니다. 이것이 "explaining away"입니다.

→ Through observation, originally independent causes become **a posteriori anti-correlated**. This is "explaining away."

**왜 이것이 문제인가**: posterior $P(\mathbf{h}|\mathbf{v})$가 factorial이 아니게 됩니다 (은닉 변수들이 조건부 의존). 따라서 은닉 변수를 하나씩 독립적으로 추론할 수 없어서, inference가 exponentially 어려워집니다.

**Why this is a problem**: The posterior $P(\mathbf{h}|\mathbf{v})$ becomes non-factorial (hidden variables become conditionally dependent). Individual inference becomes exponentially hard.

#### Complementary Prior의 아이디어 / The Complementary Prior Idea

Hinton의 해결책: likelihood가 explaining away를 유발하면, **그 효과를 정확히 상쇄하는 prior**를 사용하면 됩니다!

Hinton's solution: if the likelihood causes explaining away, use a **prior that exactly cancels this effect**!

$$\underbrace{P(\mathbf{h}|\mathbf{v})}_{\text{posterior}} = \underbrace{P(\mathbf{v}|\mathbf{h})}_{\text{likelihood}} \cdot \underbrace{P(\mathbf{h})}_{\text{complementary prior}} / P(\mathbf{v})$$

만약 likelihood가 은닉 변수들 사이에 양의 상관관계를 도입하면, complementary prior는 정확히 반대의 음의 상관관계를 가져서, posterior가 **factorial** ($P(\mathbf{h}|\mathbf{v}) = \prod_j P(h_j|\mathbf{v})$)이 됩니다.

If the likelihood introduces positive correlations among hidden variables, the complementary prior has exactly the opposite negative correlations, making the posterior **factorial**: $P(\mathbf{h}|\mathbf{v}) = \prod_j P(h_j|\mathbf{v})$.

Appendix A에서 이 조건을 만족하는 likelihood의 일반적 형태가 다음임을 증명합니다 (Eq. 11):

Appendix A proves the general form of likelihood satisfying this condition (Eq. 11):

$$P(\mathbf{x}|\mathbf{y}) = \exp\left(\sum_j \Phi_j(\mathbf{x}, y_j) + \beta(\mathbf{x}) - \log \Omega(\mathbf{y})\right)$$

이 형태에서 complementary prior는 (Eq. 12):

The corresponding complementary prior is (Eq. 12):

$$P(\mathbf{y}) = \frac{1}{C} \exp\left(\log \Omega(\mathbf{y}) + \sum_j \alpha_j(y_j)\right)$$

#### 2.1: Tied Weights의 무한 Directed Model / Infinite Directed Model with Tied Weights

핵심적인 구성: Figure 3에서 보이는 것처럼, **모든 층에서 같은 가중치 행렬 $W$와 그 전치 $W^T$를 교대로 사용**하는 무한한 깊이의 directed 모델을 상상합니다.

The key construction: as shown in Figure 3, imagine an infinitely deep directed model that **alternates between the same weight matrix $W$ and its transpose $W^T$** at every layer.

```
  V₂  v₂ᵢ
  ↑  W^T  ↓ W
  H₁  h¹ⱼ
  ↑  W^T  ↓ W
  V₁  v¹ᵢ
  ↑  W^T  ↓ W
  H₀  h⁰ⱼ
  ↑  W^T  ↓ W
  V₀  v⁰ᵢ  ← 데이터/data
```

이 모델에서 데이터를 생성하려면: 매우 깊은 층에서 랜덤 상태로 시작하여, top-down "ancestral pass"로 각 층의 변수를 부모 층의 Bernoulli 분포에서 샘플링합니다. 이것은 alternating Gibbs sampling과 정확히 같은 과정이므로, 충분한 깊이 이후에는 **정상 분포(stationary distribution)**에서 샘플을 얻게 됩니다.

To generate data: start from a random state at a very deep layer, then sample each layer's variables from the Bernoulli distribution given by the parent layer (top-down ancestral pass). This is exactly the same as alternating Gibbs sampling, so after sufficient depth, we get samples from the **stationary distribution**.

**핵심 통찰**: 이 무한 directed 모델은 각 층에서 complementary prior를 자동으로 가집니다! tied weights가 이를 보장합니다. 따라서 posterior가 factorial이고, inference가 단순히 $W^T$를 사용한 한 번의 bottom-up pass로 가능합니다.

**Key insight**: This infinite directed model automatically has complementary priors at each layer! The tied weights guarantee this. Therefore, the posterior is factorial, and inference is simply one bottom-up pass using $W^T$.

---

### §3: RBM과 Contrastive Divergence / RBM and Contrastive Divergence

#### RBM과 무한 Directed Model의 등가성 / Equivalence of RBM and Infinite Directed Model

논문은 §2.1의 무한 tied-weights directed model이 사실 **RBM과 등가**임을 보입니다. 이것은 즉각적으로 명백하지는 않지만, RBM은 같은 층 내 연결이 없고 비방향 대칭 연결을 가진 모델이며, 이것은 alternating Gibbs sampling으로 생성하는 과정이 정확히 무한 directed model의 ancestral sampling과 동일하기 때문입니다.

The paper shows that the infinite tied-weights directed model of §2.1 is in fact **equivalent to an RBM**. An RBM has no within-layer connections and symmetric undirected connections, and its alternating Gibbs sampling process is exactly the same as the ancestral sampling of the infinite directed model.

이 등가성의 실용적 의미: **RBM을 학습하면 무한 directed belief net의 모든 층의 가중치를 동시에 학습하는 것과 같습니다.**

Practical implication: **Training an RBM is equivalent to simultaneously training all layers of an infinite directed belief net.**

#### Maximum Likelihood 학습 규칙 / Maximum Likelihood Learning Rule

단일 데이터 벡터 $\mathbf{v}^0$에 대한 logistic belief net의 학습 규칙 (Eq. 2):

The learning rule for a single data vector $\mathbf{v}^0$ in a logistic belief net (Eq. 2):

$$\frac{\partial \log p(\mathbf{v}^0)}{\partial w_{ij}} = \langle h_j^0(v_i^0 - \hat{v}_i^0) \rangle \tag{2}$$

여기서 $\hat{v}_i^0$은 hidden states로부터 확률적으로 재구성된 visible의 활성화 확률입니다. 무한 tied-weights 모델에서 모든 층의 가중치가 같으므로, 전체 derivative는 모든 층 쌍의 derivative를 합산한 것입니다 (Eq. 4):

Where $\hat{v}_i^0$ is the activation probability reconstructed stochastically from hidden states. In the infinite tied-weights model with same weights at all layers, the total derivative sums over all layer pairs (Eq. 4):

$$\frac{\partial \log p(\mathbf{v}^0)}{\partial w_{ij}} = \langle h_j^0(v_i^0 - \hat{v}_i^0) \rangle + \langle v_i^1(h_j^0 - \hat{h}_j^1) \rangle + \langle h_j^1(v_i^1 - \hat{v}_i^1) \rangle + \cdots \tag{4}$$

놀랍게도, "수직으로 정렬된" 항들이 상쇄되어 Boltzmann machine 학습 규칙만 남습니다 (Eq. 5):

Remarkably, the "vertically aligned" terms cancel, leaving only the Boltzmann machine learning rule (Eq. 5):

$$\frac{\partial \log p(\mathbf{v}^0)}{\partial w_{ij}} = \langle v_i^\infty h_j^\infty \rangle - \langle v_i^0 h_j^0 \rangle$$

#### Contrastive Divergence (CD) / Contrastive Divergence

정확한 학습은 Gibbs 체인을 수렴($\infty$)까지 돌려야 하지만, CD는 $n$ 스텝만 돌립니다 (논문에서는 $n=1$). 이것은 두 KL divergence의 차이를 최소화하는 것과 동등합니다 (Eq. 6):

Exact learning requires running the Gibbs chain to convergence ($\infty$), but CD runs only $n$ steps (the paper uses $n=1$). This is equivalent to minimizing the difference of two KL divergences (Eq. 6):

$$KL(P^0 \| P_\theta^\infty) - KL(P_\theta^n \| P_\theta^\infty) \tag{6}$$

sampling noise를 무시하면 이 차이는 항상 양수(non-negative)입니다 — Gibbs sampling이 $P_\theta^n$을 $P^0$보다 $P_\theta^\infty$에 더 가깝게 만들기 때문입니다. 따라서 **CD는 항상 올바른 방향으로 학습합니다**.

Ignoring sampling noise, this difference is always non-negative — because Gibbs sampling produces $P_\theta^n$ closer to $P_\theta^\infty$ than $P^0$. Thus **CD always learns in the right direction**.

---

### §4: Greedy Layer-wise Algorithm — 논문의 핵심 / The Paper's Core

#### 알고리즘 개요 / Algorithm Overview

복잡한 모델을 한 번에 학습하는 대신, 순차적으로 학습된 **단순한 모델들의 조합**으로 구성합니다. 이 아이디어는 boosting(Freund, 1995)이나 PCA의 반복적 차원 축소와 비슷하지만, 핵심적 차이는:

Instead of learning a complex model all at once, compose it from a sequence of **simpler models learned sequentially**. The idea resembles boosting or iterative PCA, but the key difference is:

- **Boosting**: 각 모델이 이전의 실수에 가중치를 재부여한 데이터에서 학습 / Each model learns from re-weighted data based on previous mistakes
- **이 알고리즘**: 각 모델이 이전 모델의 **출력을 비선형 변환한 "새 데이터"**에서 학습 / Each model learns from **"new data" that is a nonlinear transformation** of the previous model's output

Figure 5의 하이브리드 구조:

Hybrid structure from Figure 5:

- 상위 두 층: undirected 연결 (RBM, 연상 기억) — 학습 가능한 prior / Top two layers: undirected (RBM, associative memory) — learnable prior
- 하위 층: directed 연결 (위→아래의 생성 경로) / Lower layers: directed (top-down generative pathway)

#### Greedy Algorithm의 단계 / Steps of the Greedy Algorithm

1. 모든 가중치 행렬이 tied되어 있다고 가정하고 $\mathbf{W}_0$를 학습합니다 (= RBM 학습). 이것은 어렵지만, contrastive divergence로 빠르게 근사 가능합니다. / Learn $\mathbf{W}_0$ assuming all weight matrices are tied (= RBM training). Hard, but fast approximation via contrastive divergence.

2. $\mathbf{W}_0$를 고정하고 $\mathbf{W}_0^T$를 사용하여 데이터를 첫 번째 hidden 층의 "data"로 변환합니다. / Freeze $\mathbf{W}_0$ and use $\mathbf{W}_0^T$ to transform data into "data" for the first hidden layer.

3. 이 higher-level "data"에서 $\mathbf{W}_1$을 RBM으로 학습합니다. / Learn $\mathbf{W}_1$ as an RBM on this higher-level "data."

4. 반복합니다. / Repeat.

#### Variational Bound의 증명 / Proof of the Variational Bound

이것이 논문에서 가장 중요한 수학적 결과입니다. 에너지(energy) $E(\mathbf{v}, \mathbf{h})$에 대해, 데이터 $\mathbf{v}$의 로그 확률은 다음으로 정의됩니다 (Eq. 7):

This is the paper's most important mathematical result. For energy $E(\mathbf{v}, \mathbf{h})$, the log probability of data $\mathbf{v}$ is given by (Eq. 7):

$$E(\mathbf{v}^0, \mathbf{h}^0) = -\left[\log p(\mathbf{h}^0) + \log p(\mathbf{v}^0 | \mathbf{h}^0)\right] \tag{7}$$

Variational bound (Eq. 8):

$$\log p(\mathbf{v}^0) \geq \sum_{\mathbf{h}^0} Q(\mathbf{h}^0 | \mathbf{v}^0) \left[\log p(\mathbf{h}^0) + \log p(\mathbf{v}^0 | \mathbf{h}^0)\right] - \sum_{\mathbf{h}^0} Q(\mathbf{h}^0 | \mathbf{v}^0) \log Q(\mathbf{h}^0 | \mathbf{v}^0) \tag{8}$$

**핵심 논증**: greedy 알고리즘이 higher-level 가중치 $\mathbf{W}_1$을 변경할 때, lower-level 가중치 $\mathbf{W}_0$는 고정되어 있으므로:

**Key argument**: When the greedy algorithm changes higher-level weights $\mathbf{W}_1$, lower-level weights $\mathbf{W}_0$ are fixed, so:

- $Q(\mathbf{h}^0 | \mathbf{v}^0)$와 $p(\mathbf{v}^0 | \mathbf{h}^0)$는 변하지 않음 (이들은 $\mathbf{W}_0$에만 의존) / $Q(\mathbf{h}^0 | \mathbf{v}^0)$ and $p(\mathbf{v}^0 | \mathbf{h}^0)$ don't change (they depend only on $\mathbf{W}_0$)
- $p(\mathbf{h}^0)$만 변함 — 이것은 higher-level RBM이 정의하는 prior / Only $p(\mathbf{h}^0)$ changes — this is the prior defined by the higher-level RBM

따라서 bound의 변화는 $\sum Q(\mathbf{h}^0 | \mathbf{v}^0) \log p(\mathbf{h}^0)$의 변화에만 의존합니다. RBM을 학습하면 $p(\mathbf{h}^0)$이 $Q(\mathbf{h}^0 | \mathbf{v}^0)$에 더 가까워지므로 (= higher-level "data"의 확률을 높이므로), **bound가 반드시 개선됩니다**.

The change in bound depends only on changes in $\sum Q(\mathbf{h}^0 | \mathbf{v}^0) \log p(\mathbf{h}^0)$. Training the RBM makes $p(\mathbf{h}^0)$ closer to $Q(\mathbf{h}^0 | \mathbf{v}^0)$ (= increases probability of higher-level "data"), so the **bound always improves**.

**이 결과의 의미**: 처음에는 모든 가중치가 tied ($\mathbf{W}_0 = \mathbf{W}_1$)이어서 bound가 tight합니다. 가중치를 untie하면 bound가 loose해질 수 있지만, higher-level RBM을 학습하면 bound가 **반드시 올라갑니다**. 따라서 각 층을 추가할 때마다 generative model이 **개선되거나 최소한 유지**됩니다.

**Significance**: Initially all weights are tied ($\mathbf{W}_0 = \mathbf{W}_1$), so the bound is tight. Untying weights may loosen the bound, but training the higher-level RBM **always raises it**. Therefore, adding each layer **improves or at least maintains** the generative model.

---

### §5: Up-Down Algorithm — Fine-tuning / Fine-tuning

Greedy pre-training 이후, 전체 네트워크를 fine-tune하는 **up-down algorithm**을 소개합니다. 이것은 wake-sleep algorithm (Hinton et al., 1995)의 **contrastive 변형**으로, 원래 wake-sleep의 "mode averaging" 문제를 회피합니다.

After greedy pre-training, introduces the **up-down algorithm** for fine-tuning the entire network. This is a **contrastive variant** of the wake-sleep algorithm that avoids the original's "mode averaging" problem.

#### 알고리즘 구조 / Algorithm Structure

**Wake phase (bottom-up, "positive phase")**:
1. 데이터를 visible에 clamp / Clamp data to visible
2. Recognition weights (bottom-up)로 각 hidden 층을 순차적으로 활성화 / Activate each hidden layer sequentially using recognition weights
3. 상위 두 층의 연상 기억에서 몇 스텝의 Gibbs sampling 수행 / Perform a few Gibbs sampling steps in the top-level associative memory
4. "Positive phase" 통계량(상관관계) 기록 / Record "positive phase" statistics (correlations)

**Sleep phase (top-down, "negative phase")**:
1. 상위 연상 기억에서 Gibbs sampling으로 상태를 생성 / Generate states via Gibbs sampling from the top associative memory
2. Generative weights (top-down)로 각 층을 순차적으로 생성 / Generate each layer sequentially using generative weights
3. "Negative phase" 통계량 기록 / Record "negative phase" statistics

**가중치 업데이트 / Weight updates**:
- **Generative weights** (top-down): wake phase의 통계량으로 업데이트 / Update using wake phase statistics
- **Recognition weights** (bottom-up): sleep phase의 통계량으로 업데이트 / Update using sleep phase statistics  
- **Top-level associative memory**: 두 phase의 차이로 CD 업데이트 / CD update from difference of both phases

**Wake-sleep과의 핵심 차이**: 원래 wake-sleep에서 sleep phase는 generative model의 "꿈"을 사용하여 recognition weights를 학습하는데, generative model이 여러 mode를 가지면 mode들의 평균을 학습하는 문제가 있습니다. Up-down algorithm은 contrastive 형식을 사용하여 이 문제를 완화합니다.

**Key difference from wake-sleep**: The original wake-sleep's sleep phase uses the generative model's "dreams" to train recognition weights, but if the generative model has multiple modes, it learns the average of modes. The up-down algorithm uses a contrastive form to mitigate this.

---

### §6: MNIST 실험 / MNIST Experiments

#### 네트워크 구조 (Figure 1) / Network Architecture

```
2000 top-level units ←→ 10 label units (softmax)
        ↕ W₃
   500 hidden units
        ↕ W₂
   500 hidden units
        ↕ W₁
   784 pixels (28×28)
```

#### 학습 과정 / Training Process

1. **Greedy pre-training**: 각 RBM 층을 30 epoch 동안 학습. 하위 RBM의 visible은 실수값 픽셀(0~1), 상위 RBM은 하위의 activation probability를 입력으로 사용. 최상위 RBM에서 label은 10개 softmax 유닛으로 표현 / Each RBM layer trained for 30 epochs. Lower RBM visible units are real-valued pixels (0–1), upper RBMs use lower activation probabilities as input. At the top, labels are 10 softmax units

2. **Fine-tuning (up-down)**: 처음 100 epoch에서 CD 스텝 수를 3 → 6 → 마지막 100에서 10으로 증가 / First 100 epochs with 3 CD steps → 6 → last 100 with 10

3. **하이퍼파라미터**: learning rate, momentum, weight decay를 각 RBM 층과 fine-tuning에 대해 별도 설정. 10,000개의 validation set에서 결정 / Separate learning rate, momentum, weight decay for each layer and fine-tuning. Determined on 10,000 validation set

#### 결과 / Results

| 방법 / Method | MNIST 오류율 / Error |
|---|---|
| **이 논문의 generative model** | **1.25%** |
| SVM (degree 9 polynomial) | 1.4% |
| Backprop 784→500→300→10 | 1.51% |
| Backprop 784→800→10 | 1.53% |
| Nearest Neighbor (L3 norm) | 2.8% |
| **LeNet-5 (CNN, 특수 전처리)** | **0.95%** |

1.25%는 **permutation-invariant** (픽셀 순서를 무시하는) 조건에서의 최고 성능입니다. LeNet-5 (0.95%)는 CNN의 구조적 지식(공간 불변성)을 사용하므로 직접 비교 대상이 아닙니다.

1.25% is the best result under **permutation-invariant** conditions (ignoring pixel order). LeNet-5 (0.95%) uses CNN's structural knowledge (spatial invariance) and is not directly comparable.

#### Fine-tuning의 중요성 / Importance of Fine-tuning

- Greedy pre-training만으로는 2.49% 오류 / Greedy pre-training alone gives 2.49% error
- Up-down fine-tuning 후 **1.25%** → 사전학습이 좋은 초기값을 제공하고, fine-tuning이 이를 정밀 조정 / After up-down fine-tuning **1.25%** → pre-training provides good initialization, fine-tuning refines it

Random 초기화에서 시작하는 backprop은 1.5%에 그치지만, **사전학습된 초기화**에서 시작하면 1.25%까지 도달합니다. 이것은 사전학습이 loss landscape에서 **좋은 basin**을 찾아준다는 강력한 증거입니다.

Backprop from random initialization reaches only 1.5%, but starting from **pre-trained initialization** reaches 1.25%. Strong evidence that pre-training finds a **good basin** in the loss landscape.

---

### §7: Looking into the Mind — 네트워크의 내부 / Network's Internal Representations

#### 이미지 생성 / Image Generation

학습된 네트워크에서 이미지를 생성하는 과정:

Process of generating images from the trained network:

1. 상위 연상 기억에서 label 유닛을 특정 숫자로 clamp (예: "3") / Clamp label units to a specific digit (e.g., "3") at the top associative memory
2. 1000 iterations의 alternating Gibbs sampling 수행 / Perform 1000 iterations of alternating Gibbs sampling
3. 수렴된 상태를 top-down generative pass로 하위 층에 전달 / Pass the converged state down through the generative pathway
4. 최하위 층에서 이미지가 생성됨 / Image is generated at the bottom layer

**Figure 8**: 각 행이 한 숫자 클래스에 대한 10개의 생성된 샘플. 다양한 스타일의 숫자가 생성되며, 실제 사람의 손글씨와 구분하기 어려울 정도입니다. / Each row shows 10 generated samples for one digit class. Various handwriting styles are generated, almost indistinguishable from real handwriting.

**Figure 9**: 랜덤 이진 이미지를 초기 상태로 사용하고, Gibbs sampling 과정에서 점진적으로 숫자 형태로 진화하는 과정을 보여줍니다. 이것은 네트워크가 학습한 "숫자의 개념"이 연상 기억에 저장되어 있음을 시각적으로 보여줍니다. / Starting from a random binary image, shows the gradual evolution into digit shapes through Gibbs sampling. Visually demonstrates that the network's "concept of digits" is stored in the associative memory.

Hinton은 이것을 "네트워크의 마음을 들여다보기"라고 표현하며, "high-level 내부 표현이 진실한(veridical) 외부 세계를 구성하는 것 — 이 가상의 세계가 바로 이 그림이 보여주는 것"이라고 말합니다.

Hinton describes this as "looking into the mind of the network," saying "a high-level internal representation would constitute a veridical perception of a hypothetical, external world — that hypothetical world is what the figure shows."

---

### §8: Conclusion — 결론

Hinton은 핵심 발견들을 정리합니다:

1. Deep, densely-connected belief network를 **한 층씩** 학습할 수 있습니다. 단, 하위 층을 학습할 때 상위 층이 이미 존재하지만 아직 학습되지 않았다는 점이 통상적 layer-wise 학습과의 차이입니다. / Deep belief networks can be learned **one layer at a time**. The difference from conventional layer-wise learning is that upper layers exist but are not yet learned when training lower layers.

2. Factorial approximation이 유효하려면 true posterior가 가능한 한 factorial에 가까워야 하는데, 이것을 위해 complementary prior를 가지는 tied weights를 사용합니다. 이 undirected 모델은 **contrastive divergence**로 효율적으로 학습됩니다. / For factorial approximation to work, the true posterior must be close to factorial, achieved by tied weights with complementary priors. These undirected models are efficiently learned via CD.

3. 각 층을 학습 후 가중치를 untie하면 prior가 더 이상 factorial이 아니게 되고 recognition weights도 정확하지 않지만, variational bound를 사용하여 **generative model이 개선됨**을 보장합니다. / After learning each layer and untying weights, the prior is no longer factorial and recognition weights are inexact, but the variational bound guarantees the **generative model improves**.

4. 현재의 한계: 이미지에 대해서만 작동(비이진 값은 확률로 처리), 연상 기억의 top-down 사용에 제한, segmentation의 체계적 처리 부재, 순차적 주의 부재. 그러나 이 모델은 generative model의 여러 장점(라벨 없는 학습, interpretable representation, overfitting 저항)을 보여줍니다. / Current limitations: works only for images, limited top-down use of associative memory, no systematic segmentation handling, no sequential attention. However, the model demonstrates several advantages of generative models.

---

## 핵심 시사점 / Key Takeaways

1. **"Deep network는 학습 가능하다"의 최초 실증**: 1990년대부터 2000년대 초까지의 통설을 깨뜨린 가장 중요한 결과입니다. Vanishing gradient 문제를 **직접 해결**한 것이 아니라, **좋은 초기값을 제공**하여 우회한 것이 핵심 전략입니다. 이후 ReLU, batch normalization 등이 vanishing gradient 자체를 해결하면서 사전학습의 필요성은 줄었지만, "deep이 가능하다"는 이 논문의 증명이 모든 후속 연구의 문을 열었습니다.
   **First demonstration that "deep networks can be trained"**: Broke the consensus from the 1990s–2000s. The key strategy was not directly solving vanishing gradients but **providing good initialization** to circumvent it. Later advances (ReLU, batch norm) solved vanishing gradients directly, reducing the need for pre-training, but this paper's proof that "deep is possible" opened the door for all subsequent research.

2. **Unsupervised pre-training → Supervised fine-tuning 패러다임의 탄생**: 이 "먼저 비지도로 표현을 학습하고, 그 다음 지도 학습으로 정밀 조정"하는 패턴은 현대 AI의 핵심 패러다임이 되었습니다. GPT (대량 텍스트로 사전학습 → task별 fine-tuning), BERT, 그리고 Foundation Model의 개념 모두가 이 논문에서 시작된 패러다임입니다.
   **Birth of the unsupervised pre-training → supervised fine-tuning paradigm**: The pattern of "first learn representations unsupervised, then fine-tune supervised" became a core paradigm of modern AI. GPT (pre-train on massive text → fine-tune per task), BERT, and the Foundation Model concept all trace back to this paradigm.

3. **Greedy layer-wise training의 수학적 보장**: 각 층을 추가할 때마다 variational bound가 개선된다는 증명(Eq. 7–8)은 단순한 경험적 관찰이 아닌 이론적 보장입니다. "깊을수록 무조건 좋거나 나빠지지 않는다"는 이 결과는 deep architecture에 대한 확신을 심어주었습니다.
   **Mathematical guarantee of greedy layer-wise training**: The proof that the variational bound improves with each added layer (Eq. 7–8) is a theoretical guarantee, not just empirical observation. "Deeper is always better or never worse" instilled confidence in deep architectures.

4. **생성 모델의 힘**: 이 논문은 판별 모델(classification only)과 생성 모델(data generation + classification)의 차이를 극적으로 보여줍니다. 생성 모델은 라벨 없이 학습 가능하고, 더 많은 파라미터를 overfitting 없이 학습 가능하며, 내부 표현을 "꿈"으로 시각화할 수 있습니다. 이 아이디어는 VAE, GAN, Diffusion Model로 이어집니다.
   **Power of generative models**: Dramatically demonstrates the difference between discriminative (classification only) and generative (data generation + classification) models. Generative models can learn without labels, learn more parameters without overfitting, and visualize internal representations as "dreams." This idea leads to VAE, GAN, and Diffusion Models.

5. **Explaining away 문제와 complementary prior의 우아한 해결**: Belief network의 근본적 어려움(explaining away)을 complementary prior라는 우아한 수학적 장치로 해결한 것은, 문제를 정면 돌파하는 대신 문제의 구조를 이용하여 해결하는 전형적인 Hinton의 접근법입니다.
   **Elegant solution to explaining away via complementary priors**: Solving the fundamental difficulty of belief networks with the elegant mathematical device of complementary priors exemplifies Hinton's typical approach — exploiting the problem's structure rather than brute-forcing it.

6. **RBM = 무한 Directed Model의 등가성**: 이 등가성 증명은 순수한 이론적 결과를 넘어, "RBM 하나를 학습하면 무한히 깊은 directed model의 모든 가중치를 동시에 학습하는 것"이라는 실용적 통찰을 줍니다. 유한한 계산으로 무한한 깊이의 효과를 얻는 것입니다.
   **RBM = Infinite Directed Model equivalence**: This proof provides a practical insight beyond pure theory: "training one RBM simultaneously trains all weights of an infinitely deep directed model." Achieving infinite depth effects with finite computation.

7. **에베레스트 비유 — 좋은 초기화의 중요성**: Random 초기화(backprop 1.5%)에서는 도달하지 못하는 loss landscape의 영역에, 사전학습된 초기화(1.25%)로는 도달할 수 있습니다. 이것은 "최적화 알고리즘의 변경"이 아닌 "시작점의 변경"이라는 심오한 교훈입니다.
   **Everest analogy — importance of good initialization**: Regions of the loss landscape unreachable from random initialization (backprop 1.5%) become reachable from pre-trained initialization (1.25%). The profound lesson: changing the **starting point**, not the optimization algorithm.

---

## 수학적 요약 / Mathematical Summary

### Deep Belief Net 학습 알고리즘 / Deep Belief Net Learning Algorithm

```
=== Phase 1: Greedy Layer-wise Pre-training ===

For layer l = 0, 1, 2, ...:
  1. If l == 0: input_data = training data (pixels)
     Else:     input_data = P(h^(l-1) = 1 | input to layer l-1)
  
  2. Train RBM with visible = input_data, hidden = h^l
     Using Contrastive Divergence (CD-1):
       For each mini-batch:
         v⁰ = mini-batch data
         h⁰ ~ P(h|v⁰) = σ(b + v⁰W)
         v¹ ~ P(v|h⁰) = σ(a + h⁰Wᵀ)
         h¹ ~ P(h|v¹) = σ(b + v¹W)
         
         ΔW = η(v⁰ᵀh⁰ - v¹ᵀh¹)
         Δa = η(v⁰ - v¹)
         Δb = η(h⁰ - h¹)
  
  3. Freeze W_l, move to layer l+1

=== Phase 2: Fine-tuning (Up-Down Algorithm) ===

For each mini-batch:
  // Wake phase (bottom-up)
  For layer l = 0 to L-1:
    h^l = sample from P(h^l | h^(l-1))  using recognition weights
  Run Gibbs sampling at top-level RBM
  Record positive phase statistics
  
  // Sleep phase (top-down)  
  Start from top-level Gibbs state
  For layer l = L-1 to 0:
    generate h^l from P(h^l | h^(l+1)) using generative weights
  Record negative phase statistics
  
  // Update
  Generative weights: wake statistics
  Recognition weights: sleep statistics  
  Top-level RBM: contrastive divergence
```

### 핵심 정리 / Key Theorems

| 결과 / Result | 수식 / Equation | 의미 / Significance |
|---|---|---|
| RBM Energy | $E(\mathbf{v},\mathbf{h}) = -\mathbf{a}^T\mathbf{v} - \mathbf{b}^T\mathbf{h} - \mathbf{v}^T W\mathbf{h}$ | RBM의 에너지 함수 / Energy function of RBM |
| CD Learning Rule | $\Delta w_{ij} \propto \langle v_i h_j\rangle_{\text{data}} - \langle v_i h_j\rangle_{\text{recon}}$ | 효율적 RBM 학습 / Efficient RBM training |
| CD as KL difference | $KL(P^0\|P^\infty) - KL(P^n\|P^\infty)$ (Eq. 6) | CD는 항상 올바른 방향 / CD always learns correctly |
| Variational bound | $\log p(\mathbf{v}) \geq E_Q[\log p(\mathbf{v},\mathbf{h})] + H(Q)$ (Eq. 8) | 새 층 추가 시 반드시 개선 / Always improves when adding layers |

---

## 역사 속의 논문 / Paper in the Arc of History

```
1982  Hopfield — Hopfield Network (#5)
       │  에너지 기반 연상 기억
       ▼
1985  Hinton & Sejnowski — Boltzmann Machine
       │  확률적 에너지 기반 학습, 하지만 너무 느림
       ▼
1986  Rumelhart, Hinton & Williams — Backpropagation (#6)
       │  신경망 학습의 기초, 하지만 deep에서 vanishing gradient
       ▼
1995  Hinton et al. — Wake-Sleep Algorithm
       │  생성 모델의 계층별 학습 시도, mode averaging 문제
       │
1998  LeCun et al. — LeNet-5 (#10)
       │  CNN으로 이미지 인식 성공, 구조 지식에 의존
       ▼
2002  Hinton — Contrastive Divergence
       │  RBM의 빠른 근사 학습법 개발
       ▼
★ 2006  HINTON, OSINDERO & TEH — DEEP BELIEF NETS ★
       │  Greedy layer-wise pre-training
       │  RBM = 무한 directed model 등가성
       │  Variational bound 개선 증명
       │  MNIST 1.25% (최고 성능)
       ▼
2006  Hinton & Salakhutdinov — Deep Autoencoders (Science)
       │  같은 사전학습을 autoencoder에 적용, 차원 축소
       ▼
2012  Krizhevsky et al. — AlexNet (#13)
       │  사전학습 없이 ReLU + Dropout + GPU로 deep CNN 학습 성공
       │  → RBM 사전학습의 필요성이 줄어듦
       ▼
2013  Kingma & Welling — VAE (#15)
       │  같은 variational framework를 연속 잠재 변수 + 신경망으로
       ▼
2014  Goodfellow et al. — GAN (#16)
       │  생성 모델의 새로운 패러다임
       ▼
2017+ 현재: GPT, BERT, Foundation Models
       │  "사전학습 → fine-tuning" 패러다임의 극대화
```

---

## 다른 논문과의 연결 / Connections to Other Papers

| 논문 / Paper | 관계 / Relationship |
|---|---|
| #5 Hopfield (1982) | 에너지 기반 연상 기억의 선구. DBN의 상위 두 층이 Hopfield-like 연상 기억을 형성 / Pioneer of energy-based associative memory. Top two layers of DBN form Hopfield-like associative memory |
| #6 Rumelhart et al. (1986) — Backprop | Fine-tuning 단계의 기초. 하지만 random 초기화에서는 deep에서 실패 → 이 논문이 사전학습으로 해결 / Basis for fine-tuning. But fails in deep networks from random init → this paper solves with pre-training |
| #8 Cortes & Vapnik (1995) — SVM | MNIST에서 1.4% 오류 — 이 논문의 1.25%에 의해 넘어섬. 하지만 SVM은 이론적 보장(margin)이 있고, DBN은 생성 능력이 있음 / 1.4% on MNIST — surpassed by this paper's 1.25%. SVM has theoretical guarantees (margin), DBN has generative capability |
| #10 LeCun et al. (1998) — LeNet-5 | CNN의 구조적 지식(공간 불변성)으로 0.95% 달성. DBN은 구조 지식 없이 1.25% — 범용적 접근의 힘을 보여줌 / CNN achieves 0.95% with structural knowledge. DBN reaches 1.25% without — showing the power of a general approach |
| #11 Breiman (2001) — Random Forest | RF는 "얕은" 앙상블의 정점. 이 논문은 "깊은" 네트워크의 가능성을 열어, ML의 두 패러다임(shallow vs. deep)을 분기시킴 / RF is the pinnacle of "shallow" ensembles. This paper opens "deep" networks, bifurcating ML into shallow vs. deep paradigms |
| #13 Krizhevsky et al. (2012) — AlexNet | ReLU + Dropout + GPU로 사전학습 없이 deep CNN 직접 학습 성공. RBM 사전학습의 역사적 역할이 완료됨을 보여줌 / Successfully trains deep CNN without pre-training via ReLU + Dropout + GPU. Shows the historical role of RBM pre-training is fulfilled |
| #15 Kingma & Welling (2013) — VAE | 이 논문의 variational bound를 연속 잠재 변수 + 신경망 encoder/decoder로 발전. 같은 ELBO 프레임워크 / Develops this paper's variational bound with continuous latent variables + neural net encoder/decoder. Same ELBO framework |

---

## 참고문헌 / References

- Hinton, G. E., Osindero, S., & Teh, Y.-W., "A fast learning algorithm for deep belief nets", *Neural Computation*, Vol. 18(7), pp. 1527–1554, 2006.
- Hinton, G. E., "Training products of experts by minimizing contrastive divergence", *Neural Computation*, Vol. 14(8), pp. 1711–1800, 2002.
- Hinton, G. E., Dayan, P., Frey, B. J., & Neal, R., "The wake-sleep algorithm for self-organizing neural networks", *Science*, 268:1158–1161, 1995.
- Hinton, G. E. & Salakhutdinov, R., "Reducing the dimensionality of data with neural networks", *Science*, 313:504–507, 2006.
- Neal, R., "Connectionist learning of belief networks", *Artificial Intelligence*, 56:71–113, 1992.
- Neal, R. M. & Hinton, G. E., "A new view of the EM algorithm that justifies incremental, sparse and other variants", in *Learning in Graphical Models*, pp. 355–368, 1998.
