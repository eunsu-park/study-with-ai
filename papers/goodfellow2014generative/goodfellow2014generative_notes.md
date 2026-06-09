---
title: "Generative Adversarial Nets"
authors: Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio
year: 2014
journal: "Advances in Neural Information Processing Systems (NIPS)"
doi: "10.48550/arXiv.1406.2661"
topic: Artificial Intelligence
tags: [generative model, adversarial training, minimax game, deep learning, neural network]
status: completed
date_started: 2026-04-16
date_completed: 2026-04-16
---

# 16. Generative Adversarial Nets / 생성적 적대 신경망

---

## 1. Core Contribution / 핵심 기여

이 논문은 생성 모델을 학습하기 위한 완전히 새로운 프레임워크인 **Generative Adversarial Networks (GANs)**를 제안합니다. 핵심 아이디어는 두 개의 신경망 — 데이터를 생성하는 Generator(G)와 진짜/가짜를 구별하는 Discriminator(D) — 을 **적대적으로 동시에 학습**시키는 것입니다. 이 구조는 게임 이론의 minimax 2인 게임으로 공식화됩니다. 저자들은 비모수적(non-parametric) 설정에서 (1) 이 게임의 유일한 전역 최적해가 $p_g = p_{\text{data}}$임을 증명하고 (Theorem 1), (2) 제안된 Algorithm 1이 이 최적해에 수렴함을 증명합니다 (Proposition 2). 기존 생성 모델(RBM, DBN, DBM)이 Markov chain Monte Carlo (MCMC)나 근사 추론(approximate inference)을 필요로 한 반면, GAN은 **backpropagation만으로 학습**할 수 있고 **forward propagation만으로 샘플 생성**이 가능하다는 혁신적 장점을 가집니다. MNIST, TFD, CIFAR-10 실험에서 기존 모델과 비교하여 경쟁력 있는 생성 품질을 보여주었습니다.

This paper proposes **Generative Adversarial Networks (GANs)**, an entirely new framework for training generative models. The core idea is to **simultaneously train two neural networks in competition**: a Generator (G) that produces data, and a Discriminator (D) that distinguishes real from fake. This structure is formalized as a minimax two-player game from game theory. The authors prove in the non-parametric setting that (1) the unique global optimum of this game is $p_g = p_{\text{data}}$ (Theorem 1), and (2) the proposed Algorithm 1 converges to this optimum (Proposition 2). While previous generative models (RBMs, DBNs, DBMs) required Markov chain Monte Carlo (MCMC) or approximate inference, GANs can be **trained with backpropagation alone** and **generate samples with a single forward pass**. Experiments on MNIST, TFD, and CIFAR-10 demonstrate competitive generation quality compared to existing models.

---

## 2. Reading Notes / 읽기 노트

### Section 1: Introduction / 도입부 (p.1)

논문은 deep learning의 핵심 약속 — 인공지능 응용에서 만나는 데이터(자연 이미지, 음성, 자연어)에 대한 풍부한 계층적 확률 모델을 발견하는 것 — 으로 시작합니다. 2014년 시점에서 discriminative 모델은 backpropagation, dropout, piecewise linear unit 덕분에 큰 성공을 거뒀지만, generative 모델은 **다루기 어려운 확률 계산(intractable probabilistic computations)**과 piecewise linear unit 활용의 어려움 때문에 뒤처져 있었습니다.

The paper opens with deep learning's core promise — discovering rich, hierarchical probabilistic models for AI data (images, speech, language). By 2014, discriminative models had achieved major successes via backpropagation, dropout, and piecewise linear units, but generative models lagged behind due to **intractable probabilistic computations** in maximum likelihood estimation and difficulty leveraging piecewise linear units in the generative context.

저자들은 이 어려움을 **우회(sidestep)**하는 새로운 접근법을 제안합니다: 생성 모델을 적수(adversary)인 판별 모델과 대결시키는 것입니다. **위조지폐범 vs 경찰** 비유가 핵심입니다 — 위조지폐범(G)은 탐지되지 않을 가짜 화폐를 만들려 하고, 경찰(D)은 위조품을 찾으려 합니다. 경쟁이 계속되면 양쪽 모두 실력이 향상되어 결국 위조품과 진품이 구별 불가능해집니다.

The authors propose a new approach that **sidesteps** these difficulties: pitting a generative model against an adversary, a discriminative model. The **counterfeiter vs. police** analogy is central — the counterfeiter (G) tries to produce undetectable fake currency, while the police (D) try to detect counterfeits. Competition drives both to improve until fakes are indistinguishable from genuine articles.

### Section 2: Related Work / 관련 연구 (p.2)

저자들은 기존 생성 모델을 체계적으로 분류하고 GAN과 비교합니다:

The authors systematically categorize existing generative models and compare them with GANs:

1. **비방향 그래프 모델 / Undirected graphical models** (RBM, DBM): Partition function의 계산이 불가능(intractable)하여 MCMC로 근사. Mixing 문제가 학습을 어렵게 함.
   - RBMs/DBMs: Partition function is intractable, requiring MCMC approximation. Mixing problems hamper learning.

2. **Deep Belief Networks (DBN)**: 비방향 층 + 방향 층의 하이브리드. 빠른 layer-wise pretraining이 가능하지만, 양쪽의 계산 어려움을 모두 물려받음.
   - DBNs: Hybrid of undirected and directed layers. Fast layer-wise pretraining exists, but inherits computational difficulties from both model types.

3. **Score matching / Noise-contrastive estimation (NCE)**: Log-likelihood를 직접 다루지 않는 대안적 기준이지만, 학습된 확률밀도가 정규화 상수를 제외하고 해석적으로 명시되어야 함. NCE는 GAN과 유사하게 판별적 학습 기준을 사용하지만, 고정 노이즈 분포를 사용하므로 학습이 느려짐.
   - Score matching / NCE: Alternative criteria avoiding log-likelihood, but require analytically specified density up to a normalization constant. NCE uses a discriminative criterion like GANs, but with a fixed noise distribution, causing learning to slow.

4. **Generative Stochastic Networks (GSN)**: Markov chain의 한 단계를 수행하는 기계의 파라미터를 학습. GAN과 달리 샘플링에 Markov chain이 필요하여 feedback loop 문제가 있음.
   - GSNs: Learn parameters of a machine performing one step of a generative Markov chain. Unlike GANs, require Markov chain for sampling, causing feedback loop issues.

5. **VAE (Auto-encoding Variational Bayes)**: 같은 시기(2014)에 발표된 경쟁적 접근법. Reparameterization trick으로 backprop 가능하지만, 생성 이미지가 blurry한 경향.
   - VAE: Contemporary competing approach. Trainable via reparameterization trick, but tends to produce blurry generated images.

### Section 3: Adversarial Nets / 적대적 신경망 (pp.2-3)

**이 섹션이 논문의 핵심입니다.** GAN의 수학적 구조와 학습 알고리즘을 정의합니다.

**This section is the heart of the paper.** It defines the mathematical structure and training algorithm of GANs.

**구성 요소 / Components**:
- 노이즈 prior $p_z(z)$: Generator에 입력되는 잠재 벡터의 분포 (예: 균등 분포, 가우시안)
- Generator $G(z; \theta_g)$: 잠재 공간 $z$에서 데이터 공간 $x$로의 미분 가능한 매핑 (MLP)
- Discriminator $D(x; \theta_d)$: 입력 $x$가 진짜 데이터일 확률을 출력하는 스칼라 함수 (MLP)
- $p_z(z)$: Prior distribution on latent noise vectors (e.g., uniform, Gaussian)
- $G(z; \theta_g)$: Differentiable mapping from latent space $z$ to data space $x$ (MLP)
- $D(x; \theta_d)$: Scalar function outputting probability that $x$ is real data (MLP)

**목적함수 (Eq. 1)** — GAN의 핵심 수식:

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]$$

- D 관점: $V$를 최대화. 진짜 데이터에 높은 확률($D(x) \rightarrow 1$), 가짜 데이터에 낮은 확률($D(G(z)) \rightarrow 0$)을 부여하려 함.
- G 관점: $V$를 최소화. D가 가짜를 진짜로 착각하도록($D(G(z)) \rightarrow 1$) 만들려 함.
- D's view: Maximize $V$. Assign high probability to real ($D(x) \rightarrow 1$), low to fake ($D(G(z)) \rightarrow 0$).
- G's view: Minimize $V$. Fool D into classifying fake as real ($D(G(z)) \rightarrow 1$).

**실전 학습 / Practical Training (Algorithm 1)**:
- D를 완전히 최적화하는 것은 계산적으로 비현실적이며 overfitting 위험이 있음
- 대신 **k 스텝의 D 최적화 + 1 스텝의 G 최적화**를 교대로 반복 (논문에서는 $k=1$ 사용)
- Optimizing D to completion is computationally prohibitive and risks overfitting
- Instead, alternate **k steps of D optimization + 1 step of G optimization** (paper uses $k=1$)

**Gradient saturation 문제와 해결**:
- 학습 초기에 G가 매우 좋지 않을 때, $D$가 높은 확신으로 가짜를 거부하여 $\log(1 - D(G(z)))$가 포화됨 (gradient가 거의 0)
- 해결: G의 목적함수를 $\min_G \log(1 - D(G(z)))$ 대신 $\max_G \log D(G(z))$로 변경
- 동일한 고정점을 가지지만 학습 초기에 훨씬 강한 gradient를 제공
- Early in training, when G is poor, D rejects fakes with high confidence, saturating $\log(1 - D(G(z)))$ (near-zero gradient)
- Fix: Change G's objective from $\min_G \log(1 - D(G(z)))$ to $\max_G \log D(G(z))$
- Same fixed point but much stronger gradients early in learning

**Figure 1: 학습 과정 시각화 (a)→(d)**:
- (a) 학습 초기: $p_g$(초록)가 $p_{\text{data}}$(검정)에 가깝지만 불일치, $D$(파랑)는 부분적 분류기
- (b) D 최적화: $D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}$에 수렴
- (c) G 업데이트: D의 gradient가 G를 $p_{\text{data}}$ 방향으로 안내
- (d) 수렴: $p_g = p_{\text{data}}$이고 $D(x) = 1/2$ everywhere
- (a) Early training: $p_g$ (green) close to $p_{\text{data}}$ (black) but mismatched; $D$ (blue) is partial classifier
- (b) D optimization: Converges to $D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}$
- (c) G update: D's gradient guides G toward $p_{\text{data}}$
- (d) Convergence: $p_g = p_{\text{data}}$ and $D(x) = 1/2$ everywhere

$z$에서 $x$로의 화살표 매핑에서, G는 고밀도 영역에서 수축(contract)하고 저밀도 영역에서 팽창(expand)하여 $p_g$의 모양을 조절합니다. 노이즈 $z$는 다양성의 원천으로, 각 샘플이 서로 다른 데이터 포인트를 생성하게 합니다.

In the arrow mapping from $z$ to $x$, G contracts in high-density regions and expands in low-density regions to shape $p_g$. Noise $z$ serves as the source of diversity, with each sample generating a different data point.

### Section 4: Theoretical Results / 이론적 결과 (pp.3-5)

**이 섹션은 GAN의 수학적 정당성을 제공하는 핵심 증명들을 포함합니다.**

**This section contains the core proofs that provide mathematical justification for GANs.**

#### 4.1 Global Optimality of $p_g = p_{\text{data}}$

**Proposition 1**: G가 고정되었을 때, 최적 Discriminator는:

$$D^*_G(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)} \quad \text{(Eq. 2)}$$

증명: $V(G,D)$를 적분 형태로 전개하면:

$$V(G,D) = \int_x p_{\text{data}}(x) \log D(x) + p_g(x) \log(1 - D(x)) \, dx \quad \text{(Eq. 3)}$$

$y \rightarrow a\log(y) + b\log(1-y)$ 형태의 함수가 $[0,1]$에서 $y = \frac{a}{a+b}$에서 최대값을 가짐을 이용합니다. $a = p_{\text{data}}(x)$, $b = p_g(x)$로 놓으면 Eq. 2가 도출됩니다.

Proof: Expanding $V(G,D)$ in integral form (Eq. 3), and using the fact that $y \rightarrow a\log(y) + b\log(1-y)$ achieves its maximum on $[0,1]$ at $y = \frac{a}{a+b}$, with $a = p_{\text{data}}(x)$ and $b = p_g(x)$, gives Eq. 2.

**Theorem 1**: 가상 학습 기준 $C(G) = \max_D V(G,D)$의 전역 최솟값은 $p_g = p_{\text{data}}$일 때에만 달성되며, 그 값은 $-\log 4$입니다.

**Theorem 1**: The global minimum of the virtual training criterion $C(G) = \max_D V(G,D)$ is achieved if and only if $p_g = p_{\text{data}}$, at value $-\log 4$.

증명의 핵심 단계:

1. $p_g = p_{\text{data}}$이면 $D^*(x) = 1/2$이므로 $C(G) = \log(1/2) + \log(1/2) = -\log 4$
2. $C(G)$에서 $-\log 4$를 빼면:

$$C(G) = -\log(4) + KL\left(p_{\text{data}} \left\| \frac{p_{\text{data}} + p_g}{2}\right.\right) + KL\left(p_g \left\| \frac{p_{\text{data}} + p_g}{2}\right.\right) \quad \text{(Eq. 5)}$$

3. Jensen-Shannon Divergence로 정리:

$$C(G) = -\log(4) + 2 \cdot JSD(p_{\text{data}} \| p_g) \quad \text{(Eq. 6)}$$

4. JSD는 항상 $\geq 0$이고 두 분포가 같을 때만 0이므로, $C^* = -\log 4$가 전역 최솟값이며 $p_g = p_{\text{data}}$가 유일한 해.

Key proof steps: (1) If $p_g = p_{\text{data}}$, then $D^* = 1/2$ and $C(G) = -\log 4$. (2) Subtracting $-\log 4$ from $C(G)$ yields two KL divergence terms (Eq. 5). (3) These combine to give $2 \cdot JSD$ (Eq. 6). (4) Since JSD $\geq 0$ with equality only when distributions match, $C^* = -\log 4$ is the unique global minimum.

#### 4.2 Convergence of Algorithm 1

**Proposition 2**: G와 D가 충분한 capacity를 가지고, 각 스텝에서 D가 최적해에 도달하며, $p_g$가 기준을 개선하도록 업데이트되면, $p_g$는 $p_{\text{data}}$에 수렴합니다.

**Proposition 2**: If G and D have enough capacity, the discriminator reaches its optimum at each step, and $p_g$ is updated to improve the criterion, then $p_g$ converges to $p_{\text{data}}$.

증명 핵심: $U(p_g, D) = V(G,D)$를 $p_g$의 함수로 봤을 때, 이는 $p_g$에 대해 볼록(convex)합니다. 볼록 함수의 supremum(최적 D에서의 값)도 볼록이며 유일한 전역 최적점을 가지므로(Theorem 1에 의해), 충분히 작은 업데이트로 $p_g \rightarrow p_{\text{data}}$에 수렴합니다.

Proof key idea: $U(p_g, D) = V(G,D)$ viewed as a function of $p_g$ is convex in $p_g$. The supremum of convex functions is convex with a unique global optimum (by Theorem 1), so with sufficiently small updates, $p_g \rightarrow p_{\text{data}}$.

**중요한 단서**: 실제로는 $p_g$ 자체가 아닌 파라미터 $\theta_g$를 최적화하며, MLP에서는 multiple critical points가 존재합니다. 이론적 보장은 비모수적 설정에서만 성립하지만, MLP의 실질적 성능이 이를 정당화합니다.

**Important caveat**: In practice, we optimize $\theta_g$ rather than $p_g$ itself, and MLPs introduce multiple critical points. Theoretical guarantees hold only in the non-parametric setting, but practical MLP performance justifies the approach.

### Section 5: Experiments / 실험 (pp.5-6)

**데이터셋**: MNIST, Toronto Face Database (TFD), CIFAR-10

**모델 구성**:
- Generator: Rectifier linear activations + sigmoid activations
- Discriminator: Maxout activations + dropout
- 노이즈는 Generator의 최하위 층에만 입력

**평가 방법**: Parzen window (Gaussian kernel density estimation)
- 생성된 샘플에 가우시안 Parzen window를 피팅하여 $p_g$의 log-likelihood를 추정
- $\sigma$ 파라미터는 validation set으로 교차 검증
- 이 방법은 분산이 높고 고차원에서 잘 작동하지 않지만, 당시 사용 가능한 최선의 평가 방법

**Evaluation method**: Parzen window (Gaussian kernel density estimation)
- Fit Gaussian Parzen window to generated samples to estimate log-likelihood under $p_g$
- $\sigma$ cross-validated on validation set
- High variance and poor in high dimensions, but best available evaluation method at the time

**결과 (Table 1)**:

| Model | MNIST | TFD |
|---|---|---|
| DBN | $138 \pm 2$ | $1909 \pm 66$ |
| Stacked CAE | $121 \pm 1.6$ | $\mathbf{2110 \pm 50}$ |
| Deep GSN | $214 \pm 1.1$ | $1890 \pm 29$ |
| **Adversarial nets** | $\mathbf{225 \pm 2}$ | $2057 \pm 26$ |

GAN이 MNIST에서 최고 성능, TFD에서 두 번째 성능을 달성했습니다. 저자들은 이 결과가 기존 모델과 최소한 경쟁력이 있다고 주장합니다.

GANs achieved best performance on MNIST and second-best on TFD. The authors claim these results are at least competitive with existing models.

**Figure 2**: MNIST, TFD, CIFAR-10에서의 생성 샘플. 오른쪽 열은 가장 가까운 학습 샘플을 보여주어 모델이 학습 데이터를 암기하지 않았음을 입증합니다. 샘플들은 cherry-pick되지 않은 fair random draws이며, conditional mean이 아닌 실제 모델 분포에서의 샘플입니다.

**Figure 2**: Generated samples from MNIST, TFD, CIFAR-10. Rightmost column shows nearest training examples to demonstrate the model hasn't memorized training data. Samples are fair random draws (not cherry-picked) and actual model distribution samples (not conditional means).

**Figure 3**: 잠재 공간 $z$에서의 선형 보간(linear interpolation). 숫자 1→5→5→7 사이를 부드럽게 전환하는 모습을 보여주어, 잠재 공간이 의미 있는 연속적 표현을 학습했음을 시사합니다.

**Figure 3**: Linear interpolation in latent space $z$. Smooth transitions between digits 1→5→5→7 suggest the latent space has learned meaningful continuous representations.

### Section 6: Advantages and Disadvantages / 장단점 (p.7)

**장점 / Advantages**:
1. Markov chain 불필요 — backprop만으로 학습
2. 학습 중 추론(inference) 불필요
3. 다양한 함수를 모델에 통합 가능 (미분 가능한 모든 함수)
4. Generator가 데이터 샘플로 직접 업데이트되지 않음 — gradient만 Discriminator를 통해 흐름 → 입력의 구성 요소가 Generator 파라미터에 직접 복사되지 않음
5. 매우 날카롭고(sharp) 퇴화된(degenerate) 분포도 표현 가능 (Markov chain 기반 방법은 blurry해야 mixing 가능)

1. No Markov chains — training uses only backprop
2. No inference needed during learning
3. Wide variety of differentiable functions can be incorporated
4. Generator is not updated directly with data samples — only gradients flow through Discriminator
5. Can represent very sharp, even degenerate distributions (Markov chain methods need blurry distributions for mixing)

**단점 / Disadvantages**:
1. $p_g(x)$의 명시적 표현이 없음 — 확률을 직접 계산할 수 없음
2. D와 G의 동기화가 필요 — **"Helvetica scenario" (mode collapse)**: G가 너무 많은 $z$ 값을 같은 $x$ 값에 매핑하여 다양성을 잃는 현상. G를 D 업데이트 없이 너무 많이 학습시키면 발생.

1. No explicit representation of $p_g(x)$ — cannot compute probabilities directly
2. D and G must be synchronized — **"Helvetica scenario" (mode collapse)**: G maps too many $z$ values to the same $x$, losing diversity. Occurs if G is trained too much without updating D.

**Table 2**: 생성 모델 접근법 비교표. Training, Inference, Sampling, $p(x)$ 평가, 모델 설계 측면에서 directed graphical models, undirected models, generative autoencoders, adversarial models을 체계적으로 비교. Adversarial models의 핵심 장점: 샘플링이 쉽고, 모든 미분 가능 함수를 사용 가능. 핵심 단점: D와 G 동기화 필요, $p(x)$ 직접 계산 불가.

**Table 2**: Systematic comparison across Training, Inference, Sampling, evaluating $p(x)$, and Model design. Adversarial models' key advantages: easy sampling, any differentiable function permitted. Key disadvantages: D-G synchronization needed, $p(x)$ not directly computable.

### Section 7: Conclusions and Future Work / 결론 (pp.7-8)

저자들은 5가지 미래 확장 방향을 제시합니다:

The authors propose five future extensions:

1. **Conditional GAN**: $c$를 G와 D 모두에 입력하여 $p(x|c)$ 모델링 → 후에 cGAN (Mirza & Osindero, 2014), Pix2Pix (Isola et al., 2017) 등으로 실현
2. **Learned approximate inference**: 학습된 보조 네트워크로 $x$에서 $z$ 예측 → 후에 BiGAN, ALI 등으로 실현
3. **모든 조건부 모델링**: 파라미터를 공유하는 조건부 모델 군을 학습하여 $p(x_S | x_{\bar{S}})$ 근사
4. **Semi-supervised learning**: D의 feature를 제한된 레이블 데이터와 함께 분류에 활용
5. **효율성 개선**: G와 D의 조율 방법 개선, 더 나은 $z$ 분포 탐색

1. **Conditional GAN**: Add condition $c$ to both G and D for $p(x|c)$ → later realized as cGAN, Pix2Pix, etc.
2. **Learned approximate inference**: Auxiliary network to predict $z$ from $x$ → later realized as BiGAN, ALI
3. **All conditionals modeling**: Train family of conditional models sharing parameters to approximate $p(x_S | x_{\bar{S}})$
4. **Semi-supervised learning**: Use D's features for classification with limited labeled data
5. **Efficiency improvements**: Better methods for coordinating G and D, better sampling distributions for $z$

---

## 3. Key Takeaways / 핵심 시사점

1. **적대적 학습은 생성 모델의 패러다임을 바꿨다** — 기존에는 확률 분포를 명시적으로 정의하고 최대우도(MLE)로 학습했지만, GAN은 두 네트워크의 경쟁이라는 간접적 방식으로 암묵적 분포를 학습합니다. 이는 "어떻게 좋은 생성 모델을 만들 것인가"라는 질문에 대한 근본적으로 새로운 답입니다.
   - **Adversarial training changed the paradigm of generative modeling** — instead of explicitly defining probability distributions and training via MLE, GANs learn implicit distributions through competition between two networks. This is a fundamentally new answer to "how to build good generative models."

2. **Minimax 게임 공식화는 명확한 이론적 기반을 제공한다** — $\min_G \max_D V(D,G)$ 구조가 게임 이론의 Nash equilibrium과 연결되어, 최적해의 존재와 유일성(Theorem 1), 수렴성(Proposition 2)을 엄밀히 증명할 수 있었습니다.
   - **Minimax game formulation provides a clear theoretical foundation** — the $\min_G \max_D V(D,G)$ structure connects to Nash equilibrium in game theory, enabling rigorous proofs of existence, uniqueness (Theorem 1), and convergence (Proposition 2).

3. **JSD 최소화로의 환원이 핵심 이론적 통찰이다** — 최적 D를 대입하면 G의 비용함수가 $-\log 4 + 2 \cdot JSD(p_{\text{data}} \| p_g)$로 정리됩니다. GAN 학습은 본질적으로 생성 분포와 실제 분포 사이의 Jensen-Shannon Divergence를 최소화하는 것입니다.
   - **Reduction to JSD minimization is the key theoretical insight** — substituting optimal D into G's cost gives $-\log 4 + 2 \cdot JSD(p_{\text{data}} \| p_g)$. GAN training is essentially minimizing Jensen-Shannon Divergence between generated and real distributions.

4. **이론과 실전 사이의 간극이 명시적으로 인정된다** — 비모수적 설정에서의 수렴 보장이 유한 capacity MLP에서는 성립하지 않음을 저자들이 직접 인정합니다. 이 솔직함이 후속 연구(WGAN 등의 학습 안정화)에 길을 열었습니다.
   - **The gap between theory and practice is explicitly acknowledged** — the authors directly admit that convergence guarantees in the non-parametric setting don't hold for finite-capacity MLPs. This honesty paved the way for follow-up work (WGAN and other training stabilization methods).

5. **실전 트릭이 이론만큼 중요하다** — $\log(1 - D(G(z)))$ → $\log D(G(z))$ 변경은 수학적으로는 같은 고정점이지만, gradient saturation을 해결하여 실제 학습을 가능하게 합니다. 이는 deep learning 연구에서 이론적 우아함과 실전적 실용성이 모두 필요함을 보여줍니다.
   - **Practical tricks are as important as theory** — switching from $\log(1 - D(G(z)))$ to $\log D(G(z))$ has the same fixed point mathematically but solves gradient saturation, making actual training possible. This shows that deep learning research requires both theoretical elegance and practical pragmatism.

6. **Mode collapse는 GAN의 근본적 한계로 남는다** — "Helvetica scenario"로 언급된 이 문제는 2014년 원래 논문에서 이미 인식되었으며, 이후 수년간 GAN 연구의 핵심 과제로 남았습니다. D와 G의 균형 잡힌 학습이 결정적으로 중요합니다.
   - **Mode collapse remains a fundamental limitation of GANs** — the "Helvetica scenario" was already recognized in the original 2014 paper and remained a central challenge for years. Balanced training of D and G is critically important.

7. **샘플링의 단순성은 혁신적 장점이다** — 기존 모델(RBM, DBM)이 Markov chain mixing을 필요로 한 반면, GAN은 노이즈 $z$를 샘플링하고 단일 forward pass만으로 샘플을 생성합니다. 이 효율성이 이후 실시간 응용을 가능하게 했습니다.
   - **Simplicity of sampling is a revolutionary advantage** — while previous models (RBM, DBM) required Markov chain mixing, GANs generate samples with a single forward pass from noise $z$. This efficiency enabled real-time applications.

8. **논문의 확장 가능성이 뛰어났다** — Section 7에서 제안한 conditional GAN, semi-supervised learning 등 5가지 확장이 모두 후속 연구에서 성공적으로 실현되었습니다. 이는 프레임워크 자체의 유연성과 일반성을 입증합니다.
   - **The framework's extensibility proved exceptional** — all five extensions proposed in Section 7 were successfully realized in follow-up work. This demonstrates the flexibility and generality of the framework itself.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 GAN 목적함수 / GAN Objective Function

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))] \quad \text{(Eq. 1)}$$

| 항 / Term | 의미 / Meaning |
|---|---|
| $\mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)]$ | D가 진짜 데이터를 진짜로 판단하는 능력. D는 이를 최대화 / D's ability to classify real data as real. D maximizes this |
| $\mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$ | D가 가짜 데이터를 가짜로 판단하는 능력. D는 최대화, G는 최소화 / D's ability to reject fake data. D maximizes, G minimizes |
| $\min_G \max_D$ | D가 먼저 최적화(inner max), 그 결과를 G가 최소화(outer min) / D optimizes first (inner max), then G minimizes the result (outer min) |

### 4.2 최적 Discriminator / Optimal Discriminator

$$D^*_G(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)} \quad \text{(Eq. 2)}$$

| 변수 / Variable | 의미 / Meaning |
|---|---|
| $D^*_G(x)$ | G가 고정되었을 때 D의 최적 해 / Optimal D for a fixed G |
| $p_{\text{data}}(x)$ | 점 $x$에서의 실제 데이터 밀도 / Real data density at point $x$ |
| $p_g(x)$ | 점 $x$에서의 생성 데이터 밀도 / Generated data density at point $x$ |

해석 / Interpretation:
- $p_{\text{data}}(x) \gg p_g(x)$이면 $D^*(x) \approx 1$ (진짜가 지배적인 영역)
- $p_g(x) \gg p_{\text{data}}(x)$이면 $D^*(x) \approx 0$ (가짜가 지배적인 영역)
- $p_{\text{data}}(x) = p_g(x)$이면 $D^*(x) = 1/2$ (구별 불가)

### 4.3 적분 형태 전개 / Integral Form Expansion

$$V(G,D) = \int_x p_{\text{data}}(x) \log D(x) + p_g(x) \log(1 - D(x)) \, dx \quad \text{(Eq. 3)}$$

이 변환의 핵심: $z$ 공간의 기댓값을 $x$ 공간으로 변환합니다. $G(z)$의 분포가 $p_g$이므로:

$$\mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))] = \mathbb{E}_{x \sim p_g}[\log(1 - D(x))]$$

Key transformation: Convert expectation from $z$-space to $x$-space. Since $G(z)$'s distribution is $p_g$:

### 4.4 G의 비용함수와 JSD / G's Cost Function and JSD

최적 D를 대입한 G의 비용함수:

$$C(G) = \max_D V(G,D) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D^*_G(x)] + \mathbb{E}_{x \sim p_g}[\log(1 - D^*_G(x))] \quad \text{(Eq. 4)}$$

KL Divergence로 분해:

$$C(G) = -\log(4) + KL\left(p_{\text{data}} \left\| \frac{p_{\text{data}} + p_g}{2}\right.\right) + KL\left(p_g \left\| \frac{p_{\text{data}} + p_g}{2}\right.\right) \quad \text{(Eq. 5)}$$

Jensen-Shannon Divergence로 정리:

$$C(G) = -\log(4) + 2 \cdot JSD(p_{\text{data}} \| p_g) \quad \text{(Eq. 6)}$$

| 양 / Quantity | 의미 / Meaning |
|---|---|
| $JSD(p \| q)$ | $\frac{1}{2}KL(p \| m) + \frac{1}{2}KL(q \| m)$, where $m = \frac{p+q}{2}$. 대칭적 분포 거리 / Symmetric distribution distance |
| $-\log(4) \approx -1.386$ | 전역 최솟값. $p_g = p_{\text{data}}$일 때 달성 / Global minimum, achieved when $p_g = p_{\text{data}}$ |
| $2 \cdot JSD \geq 0$ | 항상 비음수, 두 분포가 같을 때만 0 / Always non-negative, zero only when distributions match |

### 4.5 Eq. 5 유도의 상세 과정 / Detailed Derivation of Eq. 5

Eq. 4에 $D^*_G(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}$를 대입하면:

$$C(G) = \mathbb{E}_{x \sim p_{\text{data}}}\left[\log \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}\right] + \mathbb{E}_{x \sim p_g}\left[\log \frac{p_g(x)}{p_{\text{data}}(x) + p_g(x)}\right]$$

$p_g = p_{\text{data}}$일 때의 값 $-\log 4$를 빼면, 각 항에 $\log 2$를 더하고 빼는 것과 같습니다:

$$= \mathbb{E}_{x \sim p_{\text{data}}}\left[\log \frac{p_{\text{data}}(x)}{\frac{p_{\text{data}}(x) + p_g(x)}{2}} - \log 2\right] + \mathbb{E}_{x \sim p_g}\left[\log \frac{p_g(x)}{\frac{p_{\text{data}}(x) + p_g(x)}{2}} - \log 2\right]$$

$$= -\log 4 + KL\left(p_{\text{data}} \| \frac{p_{\text{data}} + p_g}{2}\right) + KL\left(p_g \| \frac{p_{\text{data}} + p_g}{2}\right)$$

### 4.6 Algorithm 1: 학습 알고리즘 / Training Algorithm

```
for number of training iterations:
    for k steps:  # Discriminator 학습
        z^(1),...,z^(m) ~ p_z(z)          # 노이즈 미니배치 샘플링
        x^(1),...,x^(m) ~ p_data(x)       # 실제 데이터 미니배치 샘플링
        θ_d ← θ_d + η · ∇_{θ_d} (1/m) Σ [log D(x^(i)) + log(1 - D(G(z^(i))))]
    end for
    # Generator 학습 (1 step)
    z^(1),...,z^(m) ~ p_z(z)              # 노이즈 미니배치 샘플링
    θ_g ← θ_g - η · ∇_{θ_g} (1/m) Σ log(1 - D(G(z^(i))))
end for
```

실전에서는 G의 목적함수를 $\max_G \frac{1}{m}\sum \log D(G(z^{(i)}))$로 변경하여 gradient saturation을 해결합니다.

In practice, G's objective is changed to $\max_G \frac{1}{m}\sum \log D(G(z^{(i)}))$ to resolve gradient saturation.

### 4.7 수치 예제: 1D 가우시안에서의 GAN / Worked Example: GAN on 1D Gaussian

$p_{\text{data}} = \mathcal{N}(\mu=3, \sigma^2=1)$이고 $p_z = \mathcal{U}(0,1)$인 간단한 경우를 생각합니다.

Consider a simple case where $p_{\text{data}} = \mathcal{N}(\mu=3, \sigma^2=1)$ and $p_z = \mathcal{U}(0,1)$.

**초기 상태**: G가 임의의 선형 함수 $G(z) = 2z + 1$이면 (출력 범위 $[1, 3]$), $p_g$는 $[1,3]$의 균등 분포.

**Initial state**: If G is an arbitrary linear function $G(z) = 2z + 1$ (output range $[1,3]$), $p_g$ is uniform on $[1,3]$.

| $x$ | $p_{\text{data}}(x)$ | $p_g(x)$ | $D^*(x) = \frac{p_{\text{data}}}{p_{\text{data}} + p_g}$ |
|---|---|---|---|
| 0.5 | 0.009 | 0 | 1.0 |
| 2.0 | 0.242 | 0.5 | 0.326 |
| 3.0 | 0.399 | 0.5 | 0.443 |
| 4.5 | 0.070 | 0 | 1.0 |

D는 $p_g$가 0인 영역($x < 1$ 또는 $x > 3$)에서 $D^* = 1$, $p_g$가 $p_{\text{data}}$보다 큰 영역에서 $D^* < 0.5$. G는 이 gradient를 따라 출력을 $p_{\text{data}}$의 고밀도 영역(3 근처)으로 이동시키고, 분포의 모양도 가우시안에 가까워지도록 조정합니다.

D outputs $D^* = 1$ where $p_g = 0$, and $D^* < 0.5$ where $p_g > p_{\text{data}}$. G follows these gradients to shift outputs toward $p_{\text{data}}$'s high-density region (near 3) and reshape the distribution toward Gaussian.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1943  McCulloch & Pitts — 인공 뉴런 모델 (Paper #1)
  │
1986  Rumelhart et al. — Backpropagation (Paper #6)
  │    └─ GAN 학습의 기반: G와 D 모두 backprop으로 학습
  │
1995  Cortes & Vapnik — Support Vector Machines (Paper #8)
  │
1997  Hochreiter & Schmidhuber — LSTM (Paper #9)
  │
2006  Hinton et al. — Deep Belief Nets / RBM pretraining (Paper #12)
  │    └─ 생성 모델의 이전 패러다임: MCMC 기반 학습
  │
2012  Krizhevsky et al. — AlexNet (Paper #13)
  │    └─ Discriminative deep learning의 성공 → GAN의 D가 이 성공을 활용
  │
2013  Kingma & Welling — Variational Autoencoder (Paper #15)
  │    └─ GAN의 동시대 경쟁자: reparameterization trick 기반 생성 모델
  │
2014  ★ Goodfellow et al. — Generative Adversarial Nets ← 이 논문
  │    └─ 적대적 학습 프레임워크 제안. Markov chain / 근사 추론 불필요
  │
2014  Mirza & Osindero — Conditional GAN (cGAN)
  │    └─ 조건부 생성: Section 7 확장 #1 실현
  │
2015  Radford et al. — Deep Convolutional GAN (DCGAN)
  │    └─ CNN 아키텍처 + GAN: 안정적 학습을 위한 가이드라인 제시
  │
2017  Arjovsky et al. — Wasserstein GAN (WGAN)
  │    └─ JSD 대신 Wasserstein distance 사용 → 학습 안정성 대폭 개선
  │
2017  Isola et al. — Pix2Pix
  │    └─ paired image-to-image translation: cGAN의 실용적 확장
  │
2017  Zhu et al. — CycleGAN
  │    └─ unpaired image translation: cycle consistency loss
  │
2018  Karras et al. — Progressive GAN → StyleGAN (2019, 2020, 2021)
  │    └─ 고해상도 포토리얼리스틱 이미지 생성
  │
2020  Ho et al. — Denoising Diffusion Probabilistic Models (DDPM)
  │    └─ Diffusion model이 이미지 생성에서 GAN을 대체하기 시작
  │
2022  Rombach et al. — Stable Diffusion / Latent Diffusion Models
       └─ 텍스트-이미지 생성의 새로운 표준
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **#6 Rumelhart 1986** — Backpropagation | GAN의 G와 D 모두 backpropagation으로 학습됨. GAN의 핵심 장점 중 하나가 "backprop만 필요"라는 것 / Both G and D are trained via backpropagation. One of GAN's key advantages is "only backprop needed" | 직접적 기반 / Direct foundation |
| **#12 Hinton 2006** — Deep Belief Nets | DBN/RBM은 GAN 이전의 주요 생성 모델. MCMC 기반 학습의 한계(mixing, partition function)가 GAN 개발의 동기 / DBN/RBM were the primary generative models before GANs. Limitations of MCMC-based training motivated GAN development | 극복 대상 / What GANs overcome |
| **#13 Krizhevsky 2012** — AlexNet | Deep discriminative model의 성공이 GAN의 전제조건. Discriminator가 이 성공을 직접 활용 / Success of deep discriminative models was GAN's prerequisite. Discriminator directly leverages this success | 전제조건 / Prerequisite |
| **#15 Kingma 2013** — VAE | 같은 시기의 경쟁적 생성 모델 접근법. VAE는 명시적 확률 모델 + reparameterization trick, GAN은 암묵적 모델 + adversarial training / Contemporary competing generative model approach. VAE uses explicit probabilistic model + reparameterization, GAN uses implicit model + adversarial training | 동시대 대안 / Contemporary alternative |
| **Arjovsky 2017** — WGAN | GAN의 JSD를 Wasserstein distance (Earth Mover's Distance)로 교체. Mode collapse와 학습 불안정성 문제를 크게 개선 / Replaces GAN's JSD with Wasserstein distance. Significantly improves mode collapse and training instability | 핵심 후속 개선 / Key follow-up improvement |
| **Isola 2017** — Pix2Pix | Section 7 확장 #1 (conditional GAN)의 실용적 실현. Paired image-to-image translation에 GAN 적용 / Practical realization of Section 7 extension #1 (conditional GAN). Applies GAN to paired image-to-image translation | 응용 확장 / Application extension |
| **Ho 2020** — DDPM | Diffusion model이 이미지 생성에서 GAN을 대체. GAN의 mode collapse, 학습 불안정성 문제를 회피하면서 동등하거나 우수한 품질 달성 / Diffusion models replace GANs for image generation. Avoid mode collapse and training instability while achieving equal or better quality | 패러다임 전환 / Paradigm shift |

---

## 7. References / 참고문헌

- Goodfellow, I. J., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., and Bengio, Y., "Generative Adversarial Nets," *Advances in Neural Information Processing Systems (NIPS)*, 2014. [arXiv:1406.2661](https://arxiv.org/abs/1406.2661)
- Hinton, G. E., Osindero, S., and Teh, Y., "A fast learning algorithm for deep belief nets," *Neural Computation*, 18, 1527–1554, 2006.
- Krizhevsky, A., Sutskever, I., and Hinton, G., "ImageNet classification with deep convolutional neural networks," *NIPS*, 2012.
- Kingma, D. P. and Welling, M., "Auto-encoding variational bayes," *ICLR*, 2014. [arXiv:1312.6114](https://arxiv.org/abs/1312.6114)
- Mirza, M. and Osindero, S., "Conditional Generative Adversarial Nets," *arXiv preprint*, 2014. [arXiv:1411.1784](https://arxiv.org/abs/1411.1784)
- Radford, A., Metz, L., and Chintala, S., "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks," *ICLR*, 2016. [arXiv:1511.06434](https://arxiv.org/abs/1511.06434)
- Arjovsky, M., Chintala, S., and Bottou, L., "Wasserstein Generative Adversarial Networks," *ICML*, 2017.
- Isola, P., Zhu, J.-Y., Zhou, T., and Efros, A. A., "Image-to-Image Translation with Conditional Adversarial Networks," *CVPR*, 2017.
- Zhu, J.-Y., Park, T., Isola, P., and Efros, A. A., "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks," *ICCV*, 2017.
- Ho, J., Jain, A., and Abbeel, P., "Denoising Diffusion Probabilistic Models," *NeurIPS*, 2020. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
- Rumelhart, D. E., Hinton, G. E., and Williams, R. J., "Learning representations by back-propagating errors," *Nature*, 323, 533–536, 1986.
- Gutmann, M. and Hyvarinen, A., "Noise-contrastive estimation: A new estimation principle for unnormalized statistical models," *AISTATS*, 2010.
- Smolensky, P., "Information processing in dynamical systems: Foundations of harmony theory," in *Parallel Distributed Processing*, vol. 1, ch. 6, pp. 194–281, MIT Press, 1986.
