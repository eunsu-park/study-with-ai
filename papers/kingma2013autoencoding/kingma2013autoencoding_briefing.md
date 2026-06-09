---
title: "Pre-Reading Briefing: Auto-Encoding Variational Bayes (VAE)"
paper_id: "15_kingma_2013"
topic: Artificial Intelligence
date: 2026-04-15
type: briefing
---

# Auto-Encoding Variational Bayes: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Kingma, D.P. & Welling, M. "Auto-Encoding Variational Bayes." arXiv:1312.6114, 2013.
**Author(s)**: Diederik P. Kingma, Max Welling
**Year**: 2013

---

## 1. 핵심 기여 / Core Contribution

이 논문은 연속적인 latent variable을 가진 directed probabilistic model에서 효율적인 추론(inference)과 학습(learning)을 수행하는 새로운 방법을 제시합니다. 핵심 기여는 두 가지입니다: (1) **Stochastic Gradient Variational Bayes (SGVB) estimator** — variational lower bound의 reparameterization을 통해 standard stochastic gradient methods로 최적화할 수 있는 미분 가능한 추정량, (2) **Auto-Encoding Variational Bayes (AEVB) algorithm** — SGVB를 사용하여 recognition model(encoder)을 학습함으로써 데이터포인트당 비싼 반복적 추론(MCMC 등) 없이 효율적인 근사 사후 추론을 가능하게 하는 알고리즘. Recognition model로 neural network을 사용하면 **Variational Auto-Encoder (VAE)**가 됩니다.

This paper introduces a novel method for efficient inference and learning in directed probabilistic models with continuous latent variables. The two key contributions are: (1) the **Stochastic Gradient Variational Bayes (SGVB) estimator** — a differentiable estimator of the variational lower bound obtained via reparameterization, optimizable with standard stochastic gradient methods, and (2) the **Auto-Encoding Variational Bayes (AEVB) algorithm** — which uses the SGVB estimator to jointly train a recognition model (encoder) alongside the generative model, eliminating the need for expensive per-datapoint iterative inference (e.g., MCMC). When a neural network serves as the recognition model, the result is the **Variational Auto-Encoder (VAE)**.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

2013년은 deep learning이 폭발적으로 성장하던 시기입니다. AlexNet(2012)이 ImageNet에서 압도적 성능을 보여준 직후이며, 대부분의 연구가 discriminative model(분류, 인식)에 집중되어 있었습니다. 한편 **generative model** 분야에서는 Boltzmann machine이나 mean-field variational inference 같은 기존 방법이 주류였지만, 이들은 복잡한 모델에서 확장성(scalability) 문제를 겪고 있었습니다.

In 2013, deep learning was experiencing explosive growth. AlexNet (2012) had just dominated ImageNet, and most research focused on discriminative models. In the generative modeling space, existing approaches like Boltzmann machines and mean-field variational inference suffered from scalability issues with complex models.

기존의 variational inference는 conjugate model이나 mean-field approximation에 의존했고, 복잡한 likelihood function(예: neural network decoder)을 가진 모델에서는 posterior가 intractable하여 사용이 어려웠습니다. Wake-sleep algorithm(Hinton et al., 1995)이 비슷한 문제를 다뤘지만, 두 개의 서로 다른 목적함수를 최적화해야 하는 한계가 있었습니다.

Prior variational inference relied on conjugate models or mean-field approximations and struggled with complex likelihood functions (e.g., neural network decoders) where the posterior is intractable. The wake-sleep algorithm (Hinton et al., 1995) addressed similar problems but required optimizing two separate objective functions.

### 타임라인 / Timeline

```
1995 ── Wake-Sleep Algorithm (Hinton et al.)
         ↓ 인식 모델로 posterior 근사하는 아이디어 /
         ↓ Idea of approximating posterior with recognition model
2006 ── Deep Belief Networks (Hinton)
         ↓ 깊은 생성 모델의 부활 / Revival of deep generative models
2012 ── Variational Bayesian Inference with Stochastic Search (BJP12)
         ↓ 확률적 탐색 기반 variational inference / Stochastic search VB
2013 ── Stochastic Variational Inference (Hoffman et al.)
         ↓ 대규모 데이터 variational inference / Scalable VI
2013 ── ★ Auto-Encoding Variational Bayes (Kingma & Welling) ★
         ↓ Reparameterization trick + amortized inference
2014 ── Stochastic Backpropagation (Rezende et al.)
         ↓ 독립적으로 유사한 기법 개발 / Independent parallel work
2014 ── Generative Adversarial Networks (Goodfellow et al.)
         ↓ 또 다른 deep generative model paradigm / Another paradigm
2015 ── Ladder VAE, Importance Weighted AE
         ↓ VAE 확장 및 개선 / VAE extensions
```

---

## 3. 필요한 배경 지식 / Prerequisites

### 확률론 기초 / Probability Basics
- **Bayes' rule / 베이즈 정리**: $p(\mathbf{z}|\mathbf{x}) = p(\mathbf{x}|\mathbf{z})p(\mathbf{z})/p(\mathbf{x})$
- **Marginal likelihood / 주변 우도**: $p(\mathbf{x}) = \int p(\mathbf{x}|\mathbf{z})p(\mathbf{z})\,d\mathbf{z}$
- **Conditional probability / 조건부 확률** 및 **joint probability / 결합 확률**

### KL Divergence / KL 발산
- $D_{KL}(q \| p) = \mathbb{E}_q[\log q - \log p] \geq 0$
- 두 확률분포 사이의 "거리"를 측정 (비대칭) / Measures "distance" between distributions (asymmetric)
- KL divergence가 0이면 두 분포가 동일 / Zero iff distributions are identical

### Variational Inference 기초 / Variational Inference Basics
- Intractable posterior를 tractable distribution으로 근사하는 기법 / Approximating intractable posteriors
- ELBO (Evidence Lower Bound) 개념 / ELBO concept
- Reading list의 논문 #6 (Backpropagation)과 #12 (Deep Belief Networks)가 선행 지식 / Papers #6 and #12 as prerequisites

### Neural Networks / 신경망
- Multi-layer perceptron (MLP) 구조 / MLP architecture
- Backpropagation을 통한 gradient 계산 / Gradient computation via backpropagation
- Activation functions (sigmoid, tanh)

### Gaussian Distribution / 가우시안 분포
- 다변량 가우시안: $\mathcal{N}(\mathbf{x}; \boldsymbol{\mu}, \boldsymbol{\Sigma})$ / Multivariate Gaussian
- Diagonal covariance의 의미 / Meaning of diagonal covariance
- Log-likelihood 계산 / Log-likelihood computation

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **Latent variable / 잠재 변수** ($\mathbf{z}$) | 직접 관측되지 않는 숨겨진 변수. 데이터 생성 과정의 내부 표현. / Unobserved hidden variables representing the internal structure of data generation. |
| **Generative model / 생성 모델** ($p_\theta(\mathbf{x}|\mathbf{z})$) | 잠재 변수로부터 데이터를 생성하는 모델. "Decoder"라고도 부름. / Model that generates data from latent variables. Also called the "decoder." |
| **Recognition model / 인식 모델** ($q_\phi(\mathbf{z}|\mathbf{x})$) | True posterior $p_\theta(\mathbf{z}|\mathbf{x})$를 근사하는 모델. "Encoder"라고도 부름. / Model approximating the true posterior. Also called the "encoder." |
| **Variational lower bound (ELBO)** | $\log p(\mathbf{x})$의 하한. 이 값을 최대화하면 marginal likelihood가 간접적으로 최대화됨. / Lower bound on $\log p(\mathbf{x})$; maximizing it indirectly maximizes marginal likelihood. |
| **Reparameterization trick / 재매개변수화 트릭** | $\mathbf{z} \sim q_\phi(\mathbf{z}|\mathbf{x})$ 대신 $\mathbf{z} = g_\phi(\boldsymbol{\epsilon}, \mathbf{x})$, $\boldsymbol{\epsilon} \sim p(\boldsymbol{\epsilon})$으로 표현하여 gradient를 $\phi$에 대해 역전파 가능하게 만드는 기법. / Expressing random sampling as a deterministic function of noise, enabling backpropagation through $\phi$. |
| **Intractable / 다루기 어려운** | 적분이나 posterior를 해석적으로 계산할 수 없는 상태. / When integrals or posteriors cannot be computed analytically. |
| **Amortized inference / 상각 추론** | 데이터포인트마다 개별 최적화 대신, 하나의 encoder network가 모든 입력에 대해 posterior를 추론. / A single encoder network infers posteriors for all inputs, instead of optimizing per datapoint. |
| **SGVB estimator** | Stochastic Gradient Variational Bayes. Reparameterization trick을 사용한 variational lower bound의 미분 가능한 추정량. / Differentiable estimator of the variational lower bound using the reparameterization trick. |
| **AEVB algorithm** | Auto-Encoding VB. SGVB + recognition model을 결합한 학습 알고리즘. / Learning algorithm combining SGVB with a recognition model. |
| **Reconstruction error / 재구성 오류** | Decoder가 latent code로부터 원본 데이터를 얼마나 잘 복원하는지 측정. $\mathbb{E}_{q_\phi}[\log p_\theta(\mathbf{x}|\mathbf{z})]$. / Measures how well the decoder reconstructs original data from latent codes. |
| **KL regularization** | Approximate posterior를 prior에 가깝게 유지하는 정규화 항. $D_{KL}(q_\phi(\mathbf{z}|\mathbf{x}) \| p_\theta(\mathbf{z}))$. / Regularization term keeping the approximate posterior close to the prior. |

---

## 5. 수식 미리보기 / Equations Preview

### 수식 1: Variational Lower Bound (ELBO)

$$\log p_\theta(\mathbf{x}^{(i)}) \geq \mathcal{L}(\theta, \phi; \mathbf{x}^{(i)}) = -D_{KL}(q_\phi(\mathbf{z}|\mathbf{x}^{(i)}) \| p_\theta(\mathbf{z})) + \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x}^{(i)})}[\log p_\theta(\mathbf{x}^{(i)}|\mathbf{z})]$$

- **왼쪽**: 데이터포인트의 log marginal likelihood (우리가 최대화하고 싶은 값)
- **첫째 항**: KL divergence — approximate posterior가 prior와 얼마나 다른지 (정규화 역할)
- **둘째 항**: Expected reconstruction error — latent code에서 원본 데이터를 얼마나 잘 복원하는지
- Left: log marginal likelihood of a datapoint (what we want to maximize)
- First term: KL divergence — regularizer keeping approximate posterior close to prior
- Second term: Expected reconstruction error — how well data is reconstructed from latent codes

### 수식 2: Reparameterization Trick

$$\tilde{\mathbf{z}} = g_\phi(\boldsymbol{\epsilon}, \mathbf{x}) \quad \text{with} \quad \boldsymbol{\epsilon} \sim p(\boldsymbol{\epsilon})$$

Gaussian 경우 / For the Gaussian case:

$$\mathbf{z} = \boldsymbol{\mu}^{(i)} + \boldsymbol{\sigma}^{(i)} \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

- $q_\phi(\mathbf{z}|\mathbf{x})$에서 직접 sampling하면 gradient가 $\phi$로 흐를 수 없음 / Direct sampling from $q_\phi$ blocks gradients to $\phi$
- 대신 deterministic function + 외부 noise로 변환 → backpropagation 가능 / Transform to deterministic function + external noise → enables backpropagation

### 수식 3: SGVB Estimator (Gaussian prior & posterior)

$$\mathcal{L}(\theta, \phi; \mathbf{x}^{(i)}) \simeq \frac{1}{2}\sum_{j=1}^{J}\left(1 + \log((\sigma_j^{(i)})^2) - (\mu_j^{(i)})^2 - (\sigma_j^{(i)})^2\right) + \frac{1}{L}\sum_{l=1}^{L}\log p_\theta(\mathbf{x}^{(i)}|\mathbf{z}^{(i,l)})$$

- **첫째 항**: $-D_{KL}$ — Gaussian prior와 posterior일 때 해석적으로 계산 가능 (Appendix B)
- **둘째 항**: Reconstruction error — sampling으로 추정 ($L=1$도 충분)
- First term: $-D_{KL}$ — computed analytically for Gaussian prior/posterior (Appendix B)
- Second term: Reconstruction error — estimated by sampling ($L=1$ suffices)

### 수식 4: KL Divergence (Gaussian Case, Appendix B)

$$-D_{KL}(q_\phi(\mathbf{z}|\mathbf{x}^{(i)}) \| p_\theta(\mathbf{z})) = \frac{1}{2}\sum_{j=1}^{J}\left(1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2\right)$$

- $J$: latent space 차원 수 / dimensionality of latent space
- $\mu_j, \sigma_j$: encoder가 출력하는 평균과 표준편차의 $j$번째 원소 / $j$-th element of mean and std from encoder

---

## 6. 읽기 가이드 / Reading Guide

### 읽기 순서 / Suggested Reading Order

1. **Abstract & Introduction (§1)** — 문제 설정과 동기 파악. "intractable posterior"가 왜 문제인지 이해.
   - Understand the problem setup and motivation. Why is the intractable posterior a problem?

2. **§2.1 Problem Scenario** — 세 가지 핵심 도전과제 (intractability, large dataset, 세 가지 관련 문제) 파악.
   - Identify the three core challenges and three related problems.

3. **§2.2 The Variational Bound** — ELBO 유도 과정. 수식 (1)-(3)을 천천히 따라가기.
   - ELBO derivation. Follow equations (1)-(3) carefully.

4. **§2.4 The Reparameterization Trick** — 이 논문의 핵심 아이디어. Gaussian 예시를 먼저 이해한 후 일반화.
   - The core idea of the paper. Understand the Gaussian example first, then generalize.

5. **§2.3 The SGVB Estimator & AEVB** — Reparameterization trick을 ELBO에 적용. Algorithm 1 확인.
   - Apply the reparameterization trick to the ELBO. Study Algorithm 1.

6. **§3 Example: Variational Auto-Encoder** — 구체적인 VAE 구현. 수식 (9)-(10)이 실제 코드로 이어지는 부분.
   - Concrete VAE implementation. Equations (9)-(10) translate directly to code.

7. **§5 Experiments** — MNIST와 Frey Face 결과. Figure 2, 3 해석.
   - MNIST and Frey Face results. Interpret Figures 2 and 3.

8. **Appendix B** — KL divergence의 해석적 해. 구현에 직접 사용됨.
   - Analytical KL divergence solution. Directly used in implementation.

9. **Appendix C** — MLP encoder/decoder 구조 상세. 구현 참고용.
   - MLP encoder/decoder details. Reference for implementation.

### 주의할 점 / Points to Watch

- **§2.2의 수식 (1)**: $\log p_\theta(\mathbf{x}^{(i)}) = D_{KL} + \mathcal{L}$ — 이 분해가 전체 논문의 기초. KL divergence가 항상 ≥ 0이므로 $\mathcal{L}$은 진정한 lower bound.
  - Equation (1) is the foundation of the entire paper. Since KL ≥ 0, $\mathcal{L}$ is a genuine lower bound.

- **Figure 1**: Graphical model을 주의 깊게 보기. 실선(생성 모델)과 점선(인식 모델)의 관계.
  - Study the graphical model carefully. Solid lines = generative model, dashed lines = recognition model.

- **Encoder는 $\mu$와 $\sigma$를 출력**: 단일 점이 아닌 분포를 출력한다는 것이 핵심.
  - The encoder outputs a distribution (mean and variance), not a single point — this is key.

---

## 7. 현대적 의의 / Modern Significance

### 딥러닝 생성 모델의 초석 / Foundation of Deep Generative Models

VAE는 GAN(2014)과 함께 현대 deep generative model의 양대 축을 형성했습니다. 이후 수많은 확장이 이루어졌습니다:

VAE, alongside GANs (2014), formed the two pillars of modern deep generative modeling. Numerous extensions followed:

- **Conditional VAE (CVAE)** — 조건부 생성 / Conditional generation
- **β-VAE** — disentangled representation 학습 / Disentangled representation learning
- **VQ-VAE** — discrete latent space, 이후 DALL-E의 기초가 됨 / Discrete latent space, basis for DALL-E
- **Hierarchical VAE** — 여러 층의 latent variable / Multiple layers of latent variables

### Reparameterization Trick의 영향 / Impact of the Reparameterization Trick

이 논문에서 소개한 reparameterization trick은 VAE를 넘어 딥러닝 전반에 광범위하게 사용됩니다:

The reparameterization trick introduced here is used far beyond VAE:

- **Gumbel-Softmax** — discrete variable에 대한 reparameterization
- **Normalizing Flows** — 더 풍부한 posterior approximation
- **Diffusion Models** — 현대 이미지 생성의 핵심 (Stable Diffusion, DALL-E 등)

### 현대 생성 모델과의 연결 / Connection to Modern Generative Models

VAE의 latent space 개념과 encoder-decoder 구조는 현대 생성 AI의 기본 패러다임이 되었습니다. Diffusion model도 넓은 의미에서 VAE의 아이디어를 확장한 것으로 볼 수 있습니다 (hierarchical latent variable model의 극단적 형태).

The latent space concept and encoder-decoder architecture from VAE became fundamental paradigms in modern generative AI. Diffusion models can be viewed as an extreme extension of VAE ideas (as an extreme form of hierarchical latent variable models).

---

## Q&A

### Q1: Maximum Likelihood가 VAE에서 어떻게 활용되는가? / How is Maximum Likelihood used in VAE?

**목표 / Goal**: 데이터셋 $\mathbf{X} = \{\mathbf{x}^{(1)}, \dots, \mathbf{x}^{(N)}\}$가 주어졌을 때, 이 데이터를 가장 높은 확률로 생성할 수 있는 모델 파라미터 $\theta$를 찾는 것. 즉 $\log p_\theta(\mathbf{X}) = \sum_{i=1}^{N} \log p_\theta(\mathbf{x}^{(i)})$를 최대화.

Given dataset $\mathbf{X}$, find model parameters $\theta$ that maximize the probability of generating this data: maximize $\log p_\theta(\mathbf{X}) = \sum_{i=1}^{N} \log p_\theta(\mathbf{x}^{(i)})$.

**문제: Intractability / Problem: Intractability**

각 데이터포인트의 likelihood를 계산하려면 모든 가능한 latent variable $\mathbf{z}$에 대해 적분해야 한다:

To compute the likelihood of each datapoint, we must integrate over all possible latent variables $\mathbf{z}$:

$$p_\theta(\mathbf{x}) = \int p_\theta(\mathbf{x}|\mathbf{z})\, p_\theta(\mathbf{z})\, d\mathbf{z}$$

$p_\theta(\mathbf{x}|\mathbf{z})$가 neural network (decoder)이면 이 적분은 closed-form solution이 없다 (intractable).

When $p_\theta(\mathbf{x}|\mathbf{z})$ is a neural network (decoder), this integral has no closed-form solution (intractable).

**해결: ELBO를 대신 최대화 / Solution: Maximize ELBO Instead**

논문 수식 (1)에서 log-likelihood를 분해하면:

Decomposing the log-likelihood from equation (1) of the paper:

$$\log p_\theta(\mathbf{x}^{(i)}) = D_{KL}(q_\phi(\mathbf{z}|\mathbf{x}^{(i)}) \| p_\theta(\mathbf{z}|\mathbf{x}^{(i)})) + \mathcal{L}(\theta, \phi; \mathbf{x}^{(i)})$$

$D_{KL} \geq 0$이므로 항상 $\log p_\theta(\mathbf{x}^{(i)}) \geq \mathcal{L}$. 따라서 $\mathcal{L}$ (ELBO)을 최대화하면 $\log p_\theta(\mathbf{x})$도 간접적으로 올라간다.

Since $D_{KL} \geq 0$, we always have $\log p_\theta(\mathbf{x}^{(i)}) \geq \mathcal{L}$. Maximizing $\mathcal{L}$ (ELBO) indirectly pushes up $\log p_\theta(\mathbf{x})$.

**ELBO의 두 항 / Two Terms of the ELBO**

$$\mathcal{L} = \underbrace{-D_{KL}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))}_{\text{정규화 / Regularization}} + \underbrace{\mathbb{E}_{q_\phi}[\log p_\theta(\mathbf{x}|\mathbf{z})]}_{\text{재구성 / Reconstruction}}$$

| 항 / Term | 역할 / Role | 비유 / Analogy |
|---|---|---|
| KL term | Encoder가 너무 자유롭게 코드를 만들지 않도록 제한 / Keeps encoder outputs close to prior | "아이디어는 상식적인 범위 내에서" / "Ideas should stay within reasonable bounds" |
| Reconstruction | Decoder가 latent code에서 원본 데이터를 잘 복원 / Decoder reconstructs original data well | "아이디어로부터 원래 그림을 잘 그려내라" / "Draw the original picture well from the idea" |

**전체 흐름 요약 / Summary Flow**

```
Maximum Likelihood          직접 계산 불가 / Directly intractable
  log p(x) 최대화     ──→   intractable integral
       │
       │ lower bound 활용 / Use lower bound
       ▼
ELBO 최대화            ──→   = -KL + Reconstruction
  L(θ,φ;x)                  두 항 모두 계산 가능 / Both terms computable!
       │
       │ reparameterization trick
       ▼
SGD로 학습 가능         ──→   일반적인 neural network 학습과 동일
                             Same as standard neural network training
```

결국 VAE의 loss function은 **maximum likelihood를 근사적으로 수행하는 것**이다. 논문 §5의 실험에서 "variational lower bound"와 "estimated marginal likelihood"를 비교하는 것도 이 때문 — ELBO가 실제 log-likelihood에 얼마나 가까운지 확인하는 것.

The VAE loss function is an **approximate maximum likelihood procedure**. This is why §5 compares the "variational lower bound" with the "estimated marginal likelihood" — checking how close the ELBO is to the actual log-likelihood.

### Q2: VAE를 이해하기 위한 확률론 기초 / Probability Theory Basics for Understanding VAE

#### 2-1. 확률분포 / Probability Distribution

확률분포는 "어떤 값이 나올 가능성이 얼마나 되는가"를 수학적으로 표현한 것이다.

A probability distribution mathematically expresses "how likely is each possible value."

**이산(Discrete)**: 주사위처럼 값이 띄엄띄엄. 확률을 직접 더할 수 있음: $\sum_x P(X=x) = 1$

Discrete: like dice rolls, with countable outcomes. Probabilities sum to 1.

**연속(Continuous)**: 키, 온도처럼 값이 연속적. 특정 값의 확률은 0이므로 **확률밀도함수(PDF)** $p(x)$를 사용. 구간의 확률: $P(a \leq X \leq b) = \int_a^b p(x)\,dx$, 전체 적분 = 1.

Continuous: like height or temperature. Since probability of any exact value is 0, we use a **probability density function (PDF)** $p(x)$. Probability over an interval: $P(a \leq X \leq b) = \int_a^b p(x)\,dx$, total integral = 1.

> **VAE에서**: latent variable $\mathbf{z}$와 데이터 $\mathbf{x}$ 모두 **연속** 확률변수이다.
>
> In VAE: both latent variable $\mathbf{z}$ and data $\mathbf{x}$ are **continuous** random variables.

#### 2-2. 가우시안(정규) 분포 / Gaussian (Normal) Distribution

VAE에서 가장 많이 쓰이는 분포이다.

The most frequently used distribution in VAE.

**1차원 / Univariate:**

$$p(x) = \mathcal{N}(x; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

- $\mu$: 평균 (분포의 중심) / mean (center of distribution)
- $\sigma^2$: 분산 (퍼진 정도) / variance (spread), $\sigma$: 표준편차 / standard deviation

**다변량 / Multivariate:**

$$p(\mathbf{x}) = \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}, \boldsymbol{\Sigma})$$

- $\boldsymbol{\mu}$: 평균 벡터 / mean vector (예: $[\mu_1, \mu_2, \dots, \mu_J]$)
- $\boldsymbol{\Sigma}$: 공분산 행렬 / covariance matrix

**대각 공분산 (Diagonal covariance)**: VAE에서는 $\boldsymbol{\Sigma} = \text{diag}(\sigma_1^2, \sigma_2^2, \dots, \sigma_J^2)$, 즉 각 차원이 독립이라고 가정한다. 이러면 결합분포가 각 차원의 곱으로 분해되어 계산이 간단해진다:

In VAE, we assume $\boldsymbol{\Sigma} = \text{diag}(\sigma_1^2, \sigma_2^2, \dots, \sigma_J^2)$, meaning each dimension is independent. This simplifies the joint distribution to a product over dimensions:

$$p(\mathbf{x}) = \prod_{j=1}^{J} \mathcal{N}(x_j; \mu_j, \sigma_j^2)$$

> **VAE에서**: encoder가 출력하는 $\boldsymbol{\mu}$와 $\boldsymbol{\sigma}$는 바로 이 가우시안의 파라미터이다.
>
> In VAE: the $\boldsymbol{\mu}$ and $\boldsymbol{\sigma}$ output by the encoder are exactly the parameters of this Gaussian.

#### 2-3. 결합확률, 조건부확률, 주변확률 / Joint, Conditional, Marginal

**결합확률 / Joint Probability**: 두 변수가 동시에 특정 값을 가질 확률.

The probability that two variables simultaneously take specific values.

$$p(\mathbf{x}, \mathbf{z}) = p(\mathbf{x} \text{ and } \mathbf{z})$$

> VAE에서: $p_\theta(\mathbf{x}, \mathbf{z}) = p_\theta(\mathbf{x}|\mathbf{z})\,p_\theta(\mathbf{z})$ — 생성 모델의 전체 확률 / Full probability of the generative model.

**조건부확률 / Conditional Probability**: 한 변수의 값이 주어졌을 때 다른 변수의 확률.

The probability of one variable given the value of another.

$$p(\mathbf{x}|\mathbf{z}) = \frac{p(\mathbf{x}, \mathbf{z})}{p(\mathbf{z})}$$

> VAE에서 / In VAE:
> - $p_\theta(\mathbf{x}|\mathbf{z})$ = **decoder** — latent code가 주어졌을 때 데이터 생성 확률 / data generation probability given latent code
> - $p_\theta(\mathbf{z}|\mathbf{x})$ = **true posterior** — 데이터가 주어졌을 때 latent code의 확률 (계산 불가!) / latent code probability given data (intractable!)
> - $q_\phi(\mathbf{z}|\mathbf{x})$ = **encoder** — true posterior를 근사 / approximation of true posterior

**주변확률 / Marginal Probability**: 한 변수를 "적분해서 없앤" 확률. 비유: 어떤 그림($\mathbf{x}$)이 만들어질 전체 확률을 알려면, 가능한 모든 아이디어($\mathbf{z}$)를 고려해서 합산해야 한다.

The probability obtained by "integrating out" one variable. Analogy: to know the total probability of a painting ($\mathbf{x}$), we must consider all possible ideas ($\mathbf{z}$) and sum them up.

$$p(\mathbf{x}) = \int p(\mathbf{x}, \mathbf{z})\,d\mathbf{z} = \int p(\mathbf{x}|\mathbf{z})\,p(\mathbf{z})\,d\mathbf{z}$$

> VAE에서: 이 $p_\theta(\mathbf{x})$가 바로 **marginal likelihood**이며, maximum likelihood에서 최대화하고 싶은 값이다. 하지만 적분이 intractable!
>
> In VAE: this $p_\theta(\mathbf{x})$ is the **marginal likelihood** — what we want to maximize, but the integral is intractable!

#### 2-4. 베이즈 정리 / Bayes' Rule

$$p(\mathbf{z}|\mathbf{x}) = \frac{p(\mathbf{x}|\mathbf{z})\,p(\mathbf{z})}{p(\mathbf{x})}$$

| 항 / Term | 이름 / Name | VAE에서의 의미 / Meaning in VAE |
|---|---|---|
| $p(\mathbf{z}|\mathbf{x})$ | Posterior (사후확률) | 데이터를 보고 난 후 latent code의 확률 / Probability of latent code after observing data |
| $p(\mathbf{x}|\mathbf{z})$ | Likelihood (우도) | Decoder — latent code로부터 데이터 생성 확률 / Data generation probability from latent code |
| $p(\mathbf{z})$ | Prior (사전확률) | Latent code의 기본 분포 $\mathcal{N}(\mathbf{0}, \mathbf{I})$ / Default distribution of latent codes |
| $p(\mathbf{x})$ | Evidence (증거) | Marginal likelihood — 적분 필요! / Requires integration! |

VAE의 근본적 문제: Bayes' rule로 posterior $p(\mathbf{z}|\mathbf{x})$를 계산하고 싶지만, 분모 $p(\mathbf{x})$가 intractable이라 직접 계산이 불가능하다. 그래서 $q_\phi(\mathbf{z}|\mathbf{x})$ (encoder)로 **근사**한다.

The fundamental problem of VAE: we want to compute the posterior $p(\mathbf{z}|\mathbf{x})$ via Bayes' rule, but the denominator $p(\mathbf{x})$ is intractable. So we **approximate** it with $q_\phi(\mathbf{z}|\mathbf{x})$ (encoder).

#### 2-5. 기댓값 / Expectation

확률분포에 대한 "가중 평균"이다.

A "weighted average" over a probability distribution.

$$\mathbb{E}_{p(x)}[f(x)] = \int f(x)\,p(x)\,dx$$

"$p(x)$로부터 $x$를 반복 추출하여 $f(x)$를 계산하면, 평균적으로 이 값에 수렴한다."

"If we repeatedly sample $x$ from $p(x)$ and compute $f(x)$, the average converges to this value."

**Monte Carlo 추정 / Monte Carlo Estimation**: 실제로 적분을 계산하는 대신 sampling으로 근사한다:

Instead of computing the integral analytically, approximate by sampling:

$$\mathbb{E}_{p(x)}[f(x)] \approx \frac{1}{L}\sum_{l=1}^{L} f(x^{(l)}), \quad x^{(l)} \sim p(x)$$

> VAE에서: ELBO의 reconstruction term $\mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})]$은 이 Monte Carlo 추정으로 계산한다. 논문에서 $L=1$이면 충분하다고 한다.
>
> In VAE: the reconstruction term of the ELBO is computed via Monte Carlo estimation. The paper states $L=1$ suffices.

#### 2-6. KL Divergence / KL 발산

두 확률분포가 얼마나 다른지 측정하는 척도이다.

A measure of how different two probability distributions are.

$$D_{KL}(q \| p) = \mathbb{E}_q\left[\log \frac{q(x)}{p(x)}\right] = \int q(x) \log \frac{q(x)}{p(x)}\,dx$$

**핵심 성질 / Key Properties**:
1. 항상 0 이상 / Always non-negative: $D_{KL}(q \| p) \geq 0$
2. 0이면 동일 / Zero iff identical: $D_{KL}(q \| p) = 0 \iff q = p$
3. 비대칭 / Asymmetric: $D_{KL}(q \| p) \neq D_{KL}(p \| q)$ — 진정한 "거리"는 아님 / Not a true "distance"

직관적으로, $q$가 $p$를 근사하는 분포라면 KL divergence는 "$q$를 사용했을 때 $p$ 대비 잃는 정보량"이다. 작을수록 $q$가 $p$를 잘 근사한다.

Intuitively, if $q$ approximates $p$, KL divergence is "the information lost by using $q$ instead of $p$." Smaller values mean better approximation.

**가우시안의 경우 / Gaussian Case**: $q = \mathcal{N}(\mu, \sigma^2)$, $p = \mathcal{N}(0, 1)$일 때 (VAE에서 정확히 이 상황):

When $q = \mathcal{N}(\mu, \sigma^2)$ and $p = \mathcal{N}(0, 1)$ (exactly the VAE scenario):

$$D_{KL}(q \| p) = -\frac{1}{2}\left(1 + \log \sigma^2 - \mu^2 - \sigma^2\right)$$

다변량으로 확장 (각 차원 독립) / Multivariate extension (independent dimensions):

$$D_{KL} = -\frac{1}{2}\sum_{j=1}^{J}\left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

> VAE에서: 이것이 논문 Appendix B의 결과이며, ELBO의 KL term을 해석적으로 계산하는 데 사용된다. $\mu = 0$, $\sigma = 1$을 대입하면 KL = 0이 되어, encoder 출력이 prior와 동일할 때 정규화 손실이 없음을 확인할 수 있다.
>
> In VAE: this is the result from Appendix B, used to analytically compute the KL term of the ELBO. Substituting $\mu = 0$, $\sigma = 1$ gives KL = 0, confirming zero regularization loss when the encoder output matches the prior.

#### 2-7. 전체 개념 연결 / How It All Comes Together

```
Prior p(z) = N(0,I)
     │
     │ "z를 하나 뽑아서" / "sample a z"
     ▼
  ┌──────┐      Likelihood (Decoder)
  │  z   │ ───→ p_θ(x|z) ───→ x 생성 / generate x
  └──────┘
     ▲
     │ "x가 주어졌을 때 z는?" / "given x, what is z?"
     │
  Posterior p(z|x) ← Bayes' rule로 계산하고 싶지만 intractable!
     │                  Want to compute via Bayes' rule, but intractable!
     │ 그래서 근사 / So approximate:
     ▼
  q_φ(z|x) = N(μ_φ(x), σ_φ(x)²)  ← Encoder (neural network)
```

학습 목표: ELBO 최대화 / Training objective: maximize ELBO

$$\mathcal{L} = \underbrace{-D_{KL}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))}_{\text{해석적 계산 (가우시안) / Analytical (Gaussian)}} + \underbrace{\mathbb{E}_{q_\phi}[\log p_\theta(\mathbf{x}|\mathbf{z})]}_{\text{Monte Carlo 추정 (sampling) / Monte Carlo estimation}}$$

| 확률론 개념 / Probability Concept | VAE에서의 역할 / Role in VAE |
|---|---|
| 가우시안 분포 / Gaussian distribution | Prior, encoder 출력, decoder 출력의 형태 / Form of prior, encoder output, decoder output |
| 조건부확률 / Conditional probability | Encoder $q_\phi(\mathbf{z}|\mathbf{x})$, Decoder $p_\theta(\mathbf{x}|\mathbf{z})$ |
| 주변확률 (적분) / Marginal (integration) | Marginal likelihood $p(\mathbf{x})$ — intractable의 원인 / Source of intractability |
| 베이즈 정리 / Bayes' rule | True posterior 계산 — intractable의 원인 / Computing true posterior — intractable |
| 기댓값 + Monte Carlo / Expectation + MC | Reconstruction term 계산 / Computing reconstruction term |
| KL divergence | Encoder를 prior에 가깝게 정규화 + ELBO 유도 / Regularize encoder toward prior + derive ELBO |
