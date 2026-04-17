---
title: "Adam: A Method for Stochastic Optimization"
authors: Diederik P. Kingma, Jimmy Lei Ba
year: 2014
journal: "ICLR 2015 (arXiv:1412.6980)"
doi: "arXiv:1412.6980"
topic: Artificial Intelligence / Optimization
tags: [optimizer, adam, adaptive-learning-rate, momentum, stochastic-optimization, bias-correction, adamax]
status: completed
date_started: 2026-04-17
date_completed: 2026-04-17
---

# 18. Adam: A Method for Stochastic Optimization / Adam: 확률적 최적화를 위한 방법

---

## 1. Core Contribution / 핵심 기여

Adam(Adaptive Moment Estimation)은 gradient의 1차 moment(평균)와 2차 raw moment(비중심 분산)의 exponential moving average를 결합하여 각 파라미터에 대해 개별적인 adaptive learning rate를 계산하는 1차(first-order) stochastic optimization 알고리즘이다. 이 방법은 두 가지 기존 접근법 — sparse gradient에 강한 AdaGrad의 adaptive learning rate와, non-stationary objective에 적합한 RMSProp의 running average — 의 장점을 하나로 통합한다. Adam의 핵심 혁신은 **초기화 편향 보정(initialization bias correction)**으로, moment estimate를 0으로 초기화할 때 발생하는 편향을 수학적으로 유도하고 보정하는 기법이다. 또한 online convex optimization framework 하에서 $O(\sqrt{T})$ regret bound를 증명하여 이론적 수렴 보장을 제공하며, $L^p$ norm의 일반화로서 $p \to \infty$인 AdaMax variant도 제안한다. 기본 hyperparameter($\alpha=0.001$, $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$)가 대부분의 문제에서 잘 작동하여, 발표 이후 딥러닝의 사실상 기본 optimizer로 자리잡았다.

Adam (Adaptive Moment Estimation) is a first-order stochastic optimization algorithm that computes individual adaptive learning rates for each parameter by combining exponential moving averages of the first moment (mean) and second raw moment (uncentered variance) of the gradients. It unifies the advantages of two existing approaches — AdaGrad's adaptive learning rates that work well with sparse gradients, and RMSProp's running averages suited for non-stationary objectives. Adam's key innovation is **initialization bias correction**, which mathematically derives and corrects the bias arising from zero-initialization of the moment estimates. The paper also proves an $O(\sqrt{T})$ regret bound under the online convex optimization framework, providing theoretical convergence guarantees, and proposes AdaMax, a variant based on the $L^\infty$ norm ($p \to \infty$ generalization). The default hyperparameters ($\alpha=0.001$, $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$) work well across most problems, establishing Adam as the de facto default optimizer in deep learning since its publication.

---

## 2. Reading Notes / 읽기 노트

### Section 1: Introduction / 도입

논문은 stochastic gradient-based optimization이 과학과 공학의 많은 분야에서 핵심적이라는 점에서 출발한다. 많은 objective function은 확률적(stochastic)이다 — mini-batch 샘플링, dropout 같은 noise source가 있기 때문이다. 이런 환경에서 higher-order 방법은 비실용적이므로, first-order 방법에 집중한다.

The paper starts from the premise that stochastic gradient-based optimization is of core practical importance across many fields. Many objective functions are stochastic — due to mini-batch sampling, dropout noise, etc. Higher-order methods are impractical in this setting, so the paper restricts discussion to first-order methods.

Adam이라는 이름은 **ada**ptive **m**oment estimation에서 유래한다. 핵심 설계 목표:
- AdaGrad의 장점: sparse gradient를 잘 처리
- RMSProp의 장점: non-stationary objective에서 잘 작동
- 추가적 장점: gradient의 rescaling에 invariant하고, step size가 대략 $\alpha$로 bounded되며, 자연스러운 step size annealing을 수행

The name Adam derives from **ada**ptive **m**oment estimation. Core design goals:
- AdaGrad's advantage: handles sparse gradients well
- RMSProp's advantage: works well on non-stationary objectives
- Additional advantages: invariant to gradient rescaling, step sizes approximately bounded by $\alpha$, natural step size annealing

---

### Section 2: Algorithm / 알고리즘

이 섹션은 Adam의 전체 알고리즘(Algorithm 1)을 제시하고 그 핵심 성질을 분석한다.

This section presents the complete Adam algorithm (Algorithm 1) and analyzes its key properties.

#### Algorithm 1: Adam의 전체 절차 / Complete Procedure

$f(\theta)$를 파라미터 $\theta$에 대해 미분 가능한 noisy objective function이라 하자. $g_t = \nabla_\theta f_t(\theta_{t-1})$은 timestep $t$에서의 gradient이다.

Let $f(\theta)$ be a noisy objective function differentiable w.r.t. parameters $\theta$. Let $g_t = \nabla_\theta f_t(\theta_{t-1})$ be the gradient at timestep $t$.

1. **1차 moment update**: $m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t$
   - Gradient의 exponential moving average — 방향(mean) 추정
   - EMA of gradients — estimates the direction (mean)
2. **2차 moment update**: $v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2$
   - Gradient 제곱의 exponential moving average — 크기(uncentered variance) 추정
   - EMA of squared gradients — estimates the magnitude (uncentered variance)
3. **Bias correction**: $\hat{m}_t = m_t / (1 - \beta_1^t)$, $\hat{v}_t = v_t / (1 - \beta_2^t)$
   - 0 초기화로 인한 편향 보정
   - Corrects bias from zero initialization
4. **Parameter update**: $\theta_t = \theta_{t-1} - \alpha \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$
   - 평균 gradient 방향으로, 크기로 정규화하여 이동
   - Move in mean gradient direction, normalized by magnitude

기본 hyperparameter: $\alpha = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$. 모든 연산은 element-wise이다.

Default hyperparameters: $\alpha = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$. All operations are element-wise.

#### 효율적 구현 / Efficient Implementation

알고리즘의 마지막 세 줄을 다음으로 대체하면 연산 순서를 개선할 수 있다:

The last three lines of the loop can be replaced for computational efficiency:

$$\alpha_t = \alpha \cdot \sqrt{1 - \beta_2^t} / (1 - \beta_1^t)$$
$$\theta_t = \theta_{t-1} - \alpha_t \cdot m_t / (\sqrt{v_t} + \hat{\epsilon})$$

이렇게 하면 bias-corrected estimate를 명시적으로 계산하지 않고도 동일한 결과를 얻는다.

This avoids explicitly computing the bias-corrected estimates while producing identical results.

#### 2.1 Adam의 Update Rule 성질 / Properties of Adam's Update Rule

**Effective step size**: $\epsilon = 0$을 가정하면, effective step은 $\Delta_t = \alpha \cdot \hat{m}_t / \sqrt{\hat{v}_t}$이다.

**Effective step size**: Assuming $\epsilon = 0$, the effective step is $\Delta_t = \alpha \cdot \hat{m}_t / \sqrt{\hat{v}_t}$.

두 가지 upper bound가 존재한다:
- $(1-\beta_1) > \sqrt{1-\beta_2}$인 경우: $|\Delta_t| \leq \alpha \cdot (1-\beta_1) / \sqrt{1-\beta_2}$
- 그 외 (더 일반적): $|\Delta_t| \leq \alpha$

Two upper bounds exist:
- When $(1-\beta_1) > \sqrt{1-\beta_2}$: $|\Delta_t| \leq \alpha \cdot (1-\beta_1) / \sqrt{1-\beta_2}$
- Otherwise (more common): $|\Delta_t| \leq \alpha$

이 성질의 의미: $\alpha$가 파라미터 공간에서 **trust region**의 크기를 설정한다. 현재 gradient 추정이 충분한 정보를 제공하지 않는 범위를 넘어서지 않도록 한다. 따라서 $\alpha$의 적절한 크기를 사전에 알기 쉽다.

Significance: $\alpha$ establishes the size of a **trust region** in parameter space — the update won't go beyond where the current gradient estimate provides sufficient information. This makes it easy to know the right scale of $\alpha$ in advance.

**Signal-to-Noise Ratio (SNR)**: $\hat{m}_t / \sqrt{\hat{v}_t}$ 비율을 SNR이라 부른다. 최적점에 가까워지면 gradient의 SNR이 감소하여 effective step size가 자연스럽게 줄어든다 — **자동 annealing** 효과.

**Signal-to-Noise Ratio (SNR)**: The ratio $\hat{m}_t / \sqrt{\hat{v}_t}$ is called the SNR. As we approach an optimum, the gradient's SNR decreases, naturally shrinking the effective step size — an **automatic annealing** effect.

**Scale invariance**: gradient를 상수 $c$로 rescaling하면, $\hat{m}_t$는 $c$배, $\hat{v}_t$는 $c^2$배가 되어, $(c \cdot \hat{m}_t) / (\sqrt{c^2 \cdot \hat{v}_t}) = \hat{m}_t / \sqrt{\hat{v}_t}$로 상쇄된다. Adam의 effective step size는 gradient의 scale에 invariant하다.

**Scale invariance**: If gradients are rescaled by a constant $c$, then $\hat{m}_t$ scales by $c$ and $\hat{v}_t$ scales by $c^2$, so $(c \cdot \hat{m}_t) / (\sqrt{c^2 \cdot \hat{v}_t}) = \hat{m}_t / \sqrt{\hat{v}_t}$. Adam's effective step size is invariant to the scale of the gradients.

---

### Section 3: Initialization Bias Correction / 초기화 편향 보정

이 섹션은 Adam의 핵심 혁신인 bias correction을 수학적으로 유도한다.

This section mathematically derives the bias correction, one of Adam's key innovations.

**문제 설정**: $v_0 = 0$으로 초기화한 2차 moment의 EMA를 생각하자:

**Problem setup**: Consider the EMA of the second moment initialized at $v_0 = 0$:

$$v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2$$

이를 재귀적으로 전개하면:

Expanding recursively:

$$v_t = (1 - \beta_2) \sum_{i=1}^{t} \beta_2^{t-i} \cdot g_i^2 \tag{1}$$

**기댓값 계산**: $\mathbb{E}[g_t^2]$가 stationary라면 (시간에 따라 변하지 않으면):

**Expected value**: If $\mathbb{E}[g_t^2]$ is stationary:

$$\mathbb{E}[v_t] = \mathbb{E}[g_t^2] \cdot (1 - \beta_2) \sum_{i=1}^{t} \beta_2^{t-i} = \mathbb{E}[g_t^2] \cdot (1 - \beta_2^t) \tag{2-4}$$

따라서 $\mathbb{E}[v_t] \neq \mathbb{E}[g_t^2]$이다. 차이는 $(1 - \beta_2^t)$ 인수에 의한 것이다. 이를 보정하려면:

So $\mathbb{E}[v_t] \neq \mathbb{E}[g_t^2]$. The discrepancy is the factor $(1 - \beta_2^t)$. To correct this:

$$\hat{v}_t = v_t / (1 - \beta_2^t) \quad \Rightarrow \quad \mathbb{E}[\hat{v}_t] = \mathbb{E}[g_t^2]$$

**수치 예시 / Numerical example**: $\beta_2 = 0.999$일 때:
- $t=1$: $1 - 0.999^1 = 0.001$ → 보정 계수 $1/0.001 = 1000$배
- $t=10$: $1 - 0.999^{10} \approx 0.01$ → 보정 계수 약 100배
- $t=1000$: $1 - 0.999^{1000} \approx 0.632$ → 보정 계수 약 1.58배
- $t \to \infty$: $1 - 0.999^t \to 1$ → 보정 불필요

Bias correction이 없으면, 특히 $\beta_2$가 1에 가까울 때 초기 step에서 moment 추정이 실제보다 훨씬 작아져서, 초기 parameter update가 매우 크고 불안정해질 수 있다.

Without bias correction, especially when $\beta_2$ is close to 1, the moment estimates are much smaller than their true values in early steps, leading to excessively large and unstable parameter updates.

**Non-stationary 경우**: $\mathbb{E}[g_t^2]$가 시간에 따라 변하면 추가 오차항 $\zeta$가 발생하지만, $\beta_1$을 적절히 설정하면 이를 작게 유지할 수 있다.

**Non-stationary case**: If $\mathbb{E}[g_t^2]$ varies over time, an additional error term $\zeta$ arises, but it can be kept small by choosing $\beta_1$ appropriately.

---

### Section 4: Convergence Analysis / 수렴 분석

Online convex optimization framework (Zinkevich, 2003)를 사용하여 Adam의 수렴을 분석한다. 비용 함수의 sequence $f_1(\theta), f_2(\theta), ..., f_T(\theta)$가 주어지면, regret는:

Convergence is analyzed under the online convex optimization framework (Zinkevich, 2003). Given a sequence of cost functions $f_1(\theta), f_2(\theta), ..., f_T(\theta)$, regret is:

$$R(T) = \sum_{t=1}^{T} [f_t(\theta_t) - f_t(\theta^*)] \tag{5}$$

여기서 $\theta^* = \arg\min_{\theta \in \mathcal{X}} \sum_{t=1}^{T} f_t(\theta)$는 최적의 고정점이다.

Where $\theta^* = \arg\min_{\theta \in \mathcal{X}} \sum_{t=1}^{T} f_t(\theta)$ is the best fixed point.

**Theorem 4.1** (핵심 결과): $f_t$가 bounded gradient를 가지고, $\gamma = \beta_1^2 / \sqrt{\beta_2} < 1$이며, $\alpha_t = \alpha/\sqrt{t}$로 설정하면:

**Theorem 4.1** (Main result): If $f_t$ has bounded gradients, $\gamma = \beta_1^2 / \sqrt{\beta_2} < 1$, and $\alpha_t = \alpha/\sqrt{t}$:

$$R(T) \leq \frac{D^2}{2\alpha(1-\beta_1)} \sum_{i=1}^{d} \sqrt{T\hat{v}_{T,i}} + \frac{\alpha(1+\beta_1)G_\infty}{(1-\beta_1)\sqrt{1-\beta_2}(1-\gamma)^2} \sum_{i=1}^{d} \|g_{1:T,i}\|_2 + \frac{D_\infty^2 G_\infty \sqrt{1-\beta_2}}{2\alpha(1-\beta_1)(1-\lambda)^2}$$

**Corollary 4.2**: 평균 regret의 bound:

**Corollary 4.2**: Bound on average regret:

$$R(T)/T = O(1/\sqrt{T})$$

이는 $\lim_{T\to\infty} R(T)/T = 0$을 의미하여, Adam이 수렴함을 보장한다. 이 bound는 online convex optimization의 best known result와 comparable하다. Data가 sparse하면 summation term이 upper bound보다 훨씬 작아져서, Adam(과 AdaGrad)은 $O(\log d \sqrt{T})$를 달성할 수 있다 — non-adaptive 방법의 $O(\sqrt{dT})$보다 개선된 결과이다.

This implies $\lim_{T\to\infty} R(T)/T = 0$, guaranteeing Adam converges. This bound is comparable to the best known results in online convex optimization. When data is sparse, the summation terms are much smaller than their upper bounds, so Adam (and AdaGrad) can achieve $O(\log d \sqrt{T})$ — an improvement over $O(\sqrt{dT})$ for non-adaptive methods.

**주의**: 이 분석은 convex setting에서만 유효하다. Deep learning의 non-convex objective에 대한 수렴 보장은 제공하지 않지만, 실험적으로 Adam은 non-convex 문제에서도 잘 작동한다.

**Caveat**: This analysis is only valid in the convex setting. It does not provide convergence guarantees for the non-convex objectives of deep learning, though empirically Adam works well on non-convex problems.

---

### Section 5: Related Work / 관련 연구

이 섹션은 Adam과 기존 optimizer들의 수학적 관계를 명확히 한다.

This section clarifies the mathematical relationships between Adam and existing optimizers.

**RMSProp과의 관계**: RMSProp with momentum은 rescaled gradient에 momentum을 적용하지만, Adam은 gradient 자체의 1차/2차 moment를 별도로 추적한다. 또한 RMSProp은 bias correction이 없어서, $\beta_2$가 1에 가까울 때 (sparse gradient에서 필요) 초기에 매우 큰 step이 발생하고 종종 diverge한다 — Section 6.4에서 실험적으로 확인.

**Relationship to RMSProp**: RMSProp with momentum applies momentum on the rescaled gradient, while Adam tracks the first and second moments of the gradient separately. RMSProp also lacks bias correction, leading to very large initial steps and often divergence when $\beta_2$ is close to 1 (needed for sparse gradients) — confirmed experimentally in Section 6.4.

**AdaGrad와의 관계**: $\beta_2 \to 1$로 보내면, $\lim_{\beta_2 \to 1} \hat{v}_t = t^{-1} \cdot \sum_{i=1}^{t} g_i^2$이 되어, AdaGrad는 Adam의 특수 경우($\beta_1 = 0$, infinitesimal $(1-\beta_2)$, annealed $\alpha_t = \alpha \cdot t^{-1/2}$)에 해당한다. 단, 이 대응은 bias correction이 있을 때만 성립한다 — bias correction 없이 $\beta_2 \to 1$로 보내면 무한대 bias와 무한대 parameter update가 발생한다.

**Relationship to AdaGrad**: Taking $\beta_2 \to 1$, $\lim_{\beta_2 \to 1} \hat{v}_t = t^{-1} \cdot \sum_{i=1}^{t} g_i^2$, so AdaGrad corresponds to a special case of Adam ($\beta_1 = 0$, infinitesimal $(1-\beta_2)$, annealed $\alpha_t = \alpha \cdot t^{-1/2}$). This correspondence only holds with bias correction — without it, taking $\beta_2 \to 1$ leads to infinite bias and infinite parameter updates.

**Natural Gradient Descent (NGD)와의 관계**: Adam은 preconditioner로서 Fisher information matrix의 대각 근사($\hat{v}_t$)를 사용한다. NGD와 유사하지만, Adam(과 AdaGrad)의 preconditioner는 Fisher information matrix의 역행렬의 대각에 대한 제곱근으로, vanilla NGD보다 더 보수적이다.

**Relationship to Natural Gradient Descent (NGD)**: Adam employs a preconditioner that approximates the diagonal of the Fisher information matrix ($\hat{v}_t$). Similar to NGD, but Adam's (and AdaGrad's) preconditioner is the square root of the inverse of the diagonal Fisher information matrix approximation — more conservative than vanilla NGD.

---

### Section 6: Experiments / 실험

네 가지 실험으로 Adam의 성능을 검증한다.

Four experiments validate Adam's performance.

#### 6.1 Logistic Regression / 로지스틱 회귀

**설정**: MNIST (784차원)와 IMDB BoW (10,000차원, sparse), mini-batch 128, $\alpha_t = \alpha/\sqrt{t}$ decay 적용.

**Setup**: MNIST (784-dim) and IMDB BoW (10,000-dim, sparse), mini-batch 128, $\alpha_t = \alpha/\sqrt{t}$ decay.

**결과**:
- MNIST: Adam ≈ SGD+Nesterov momentum, 둘 다 AdaGrad보다 빠름
- IMDB (sparse): AdaGrad가 SGD+Nesterov를 크게 앞섬, Adam은 AdaGrad와 동등한 속도로 수렴
- Adam은 dense와 sparse 양쪽 모두에서 경쟁력 있음

**Results**:
- MNIST: Adam ≈ SGD+Nesterov momentum, both faster than AdaGrad
- IMDB (sparse): AdaGrad outperforms SGD+Nesterov by large margin, Adam converges as fast as AdaGrad
- Adam is competitive in both dense and sparse settings

#### 6.2 Multi-layer Neural Networks / 다층 신경망

**설정**: MNIST, 2개 hidden layer (각 1000 units), ReLU activation, mini-batch 128.

**Setup**: MNIST, 2 hidden layers (1000 units each), ReLU activation, mini-batch 128.

**결과**:
- Deterministic cross-entropy + $L_2$ weight decay: Adam이 SFO보다 iteration과 wall-clock 모두에서 빠름. SFO는 iteration당 5-10배 느리고, 메모리가 mini-batch 수에 linear
- Dropout stochastic regularization: Adam이 AdaGrad, RMSProp, SGD+Nesterov, AdaDelta 모두를 앞섬
- SFO는 stochastic regularization(dropout)을 사용하면 수렴 실패 — deterministic subfunction을 가정하기 때문

**Results**:
- Deterministic cross-entropy + $L_2$ weight decay: Adam faster than SFO in both iterations and wall-clock. SFO is 5-10x slower per iteration with memory linear in number of minibatches
- Dropout stochastic regularization: Adam outperforms AdaGrad, RMSProp, SGD+Nesterov, AdaDelta
- SFO fails with stochastic regularization (dropout) — assumes deterministic subfunctions

#### 6.3 Convolutional Neural Networks / 합성곱 신경망

**설정**: CIFAR-10, 3-stage CNN (5×5 conv + 3×3 max pool, stride 2) + FC layer (1000 ReLU units), dropout, mini-batch 128.

**Setup**: CIFAR-10, 3-stage CNN (5×5 conv + 3×3 max pool, stride 2) + FC layer (1000 ReLU units), dropout, mini-batch 128.

**결과**:
- 초기 3 epoch: Adam과 AdaGrad 모두 빠른 진전, 하지만 장기적으로 Adam과 SGD가 AdaGrad보다 훨씬 빠르게 수렴
- CNN에서는 2차 moment estimate $\hat{v}_t$가 몇 epoch 후 0에 가까워져서 $\epsilon$이 지배적 → 2차 moment가 poor approximation
- 이 경우 1차 moment(momentum)를 통한 mini-batch variance 감소가 더 중요
- Adam은 SGD+momentum 대비 marginal improvement이지만, 레이어별 learning rate를 자동으로 적응시킴

**Results**:
- First 3 epochs: Adam and AdaGrad both make rapid progress, but long-term Adam and SGD converge considerably faster than AdaGrad
- In CNNs, $\hat{v}_t$ approaches zero after a few epochs, dominated by $\epsilon$ → second moment is a poor approximation
- First moment (momentum) for reducing minibatch variance is more important here
- Adam shows marginal improvement over SGD+momentum, but auto-adapts learning rate scale for different layers

#### 6.4 Bias-correction Term 실험 / Bias Correction Experiment

**설정**: VAE (Kingma & Welling, 2013), 500 hidden units, softplus, 50-dim latent. $\beta_1 \in [0, 0.9]$, $\beta_2 \in [0.99, 0.999, 0.9999]$, $\log_{10}(\alpha) \in [-5, ..., -1]$.

**Setup**: VAE (Kingma & Welling, 2013), 500 hidden units, softplus, 50-dim latent. $\beta_1 \in [0, 0.9]$, $\beta_2 \in [0.99, 0.999, 0.9999]$, $\log_{10}(\alpha) \in [-5, ..., -1]$.

**결과**:
- Bias correction 없으면, $\beta_2$가 1에 가까울 때 학습 초기에 불안정성 발생 (특히 첫 몇 epoch)
- Sparse gradient에서는 $\beta_2$가 1에 가까워야 하므로, bias correction의 중요성이 더 커짐
- 학습 후반부에도 hidden unit이 특정 패턴에 특화되면서 gradient가 sparser해져서, bias correction의 효과가 더 커짐
- Adam은 **모든 hyperparameter 설정에서 bias correction 없는 버전(≈ RMSProp with momentum) 대비 동등 이상**

**Results**:
- Without bias correction, instabilities occur in early training when $\beta_2$ is close to 1 (especially first few epochs)
- Since $\beta_2$ needs to be close to 1 for sparse gradients, bias correction becomes even more important
- Even later in training, as hidden units specialize and gradients become sparser, bias correction's effect grows
- Adam **matches or outperforms the version without bias correction (≈ RMSProp with momentum) across all hyperparameter settings**

---

### Section 7: Extensions / 확장

#### 7.1 AdaMax / 아다맥스

Adam의 2차 moment update를 $L^p$ norm으로 일반화한다. Timestep $t$에서 step size는 $v_t^{1/p}$에 반비례한다:

Adam's second moment update is generalized to the $L^p$ norm. The step size at timestep $t$ is inversely proportional to $v_t^{1/p}$:

$$v_t = \beta_2^p \cdot v_{t-1} + (1 - \beta_2^p) \cdot |g_t|^p \tag{6}$$

$p \to \infty$로 보내면:

Taking $p \to \infty$:

$$u_t = \lim_{p \to \infty} (v_t)^{1/p} = \max(\beta_2 \cdot u_{t-1}, |g_t|) \tag{12}$$

이 유도의 핵심 단계: $\lim_{p\to\infty}(1-\beta_2^p)^{1/p}$이 사라지고, $\lim_{p\to\infty}(\sum a_i^p)^{1/p} = \max(a_i)$를 이용하여 max 연산이 나타난다.

Key step in the derivation: $\lim_{p\to\infty}(1-\beta_2^p)^{1/p}$ vanishes, and using $\lim_{p\to\infty}(\sum a_i^p)^{1/p} = \max(a_i)$, the max operation emerges.

AdaMax의 update rule (Algorithm 2):

$$m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t$$
$$u_t = \max(\beta_2 \cdot u_{t-1}, |g_t|)$$
$$\theta_t = \theta_{t-1} - (\alpha / (1 - \beta_1^t)) \cdot m_t / u_t$$

AdaMax의 장점:
- 2차 moment에 대한 bias correction이 불필요 (max 연산은 0-initialization bias가 없음)
- Step size에 더 간단한 bound: $|\Delta_t| \leq \alpha$
- 기본 hyperparameter: $\alpha = 0.002$, $\beta_1 = 0.9$, $\beta_2 = 0.999$

Advantages of AdaMax:
- No bias correction needed for the second moment (max operation has no zero-initialization bias)
- Simpler bound on step size: $|\Delta_t| \leq \alpha$
- Default hyperparameters: $\alpha = 0.002$, $\beta_1 = 0.9$, $\beta_2 = 0.999$

#### 7.2 Temporal Averaging / 시간 평균

Noisy stochastic approximation에서는 마지막 iterate보다 평균이 더 나은 일반화를 보인다. Polyak-Ruppert averaging을 Adam에 적용할 수 있다:

In noisy stochastic approximation, averaging often generalizes better than the last iterate. Polyak-Ruppert averaging can be applied to Adam:

$$\bar{\theta}_t = \beta_2 \cdot \bar{\theta}_{t-1} + (1 - \beta_2) \cdot \theta_t, \quad \hat{\theta}_t = \bar{\theta}_t / (1 - \beta_2^t)$$

---

### Section 8: Conclusion / 결론

Adam은 gradient-based optimization을 위한 간단하고 계산 효율적인 알고리즘이다. AdaGrad의 sparse gradient 처리 능력과 RMSProp의 non-stationary objective 처리 능력을 결합하며, 이론적 수렴 보장을 제공한다. Adam은 robust하고, 다양한 non-convex optimization 문제에 적합하다.

Adam is a simple and computationally efficient algorithm for gradient-based optimization. It combines AdaGrad's ability to handle sparse gradients with RMSProp's ability to handle non-stationary objectives, while providing theoretical convergence guarantees. Adam is robust and well-suited to a wide range of non-convex optimization problems.

---

## 3. Key Takeaways / 핵심 시사점

1. **Adam은 두 계열 optimizer의 통합이다** — Momentum 계열(1차 moment)과 Adaptive learning rate 계열(2차 moment, RMSProp)을 하나의 알고리즘으로 결합했다. 이는 단순한 혼합이 아니라, 각 moment를 별도로 추적하고 bias correction으로 일관성을 보장하는 체계적 설계이다.
**Adam unifies two lineages of optimizers** — It combines the Momentum lineage (1st moment) and the Adaptive learning rate lineage (2nd moment, RMSProp) into a single algorithm. This is not a naive mixture but a systematic design where each moment is tracked separately, with bias correction ensuring consistency.

2. **Bias correction은 Adam의 핵심 혁신이다** — 0 초기화에서 발생하는 편향을 $(1-\beta^t)$로 나누어 보정하는 기법은 수학적으로 간단하지만, RMSProp에 없어서 $\beta_2$가 1에 가까울 때 불안정성을 야기했다. $\beta_2 = 0.999$에서 초기 timestep의 보정 계수가 1000배에 달한다는 점이 이 기법의 중요성을 보여준다.
**Bias correction is Adam's key innovation** — Dividing by $(1-\beta^t)$ to correct zero-initialization bias is mathematically simple but crucial. Its absence in RMSProp caused instability when $\beta_2$ is close to 1. The correction factor reaching 1000x at the initial timestep with $\beta_2 = 0.999$ demonstrates its importance.

3. **Effective step size가 $\alpha$로 bounded되어 trust region을 형성한다** — $|\hat{m}_t/\sqrt{\hat{v}_t}| \leq 1$ (Cauchy-Schwarz)이므로, step size가 대략 $\alpha$를 넘지 않는다. 이는 hyperparameter 튜닝을 극적으로 쉽게 만드는 성질이다 — $\alpha$를 파라미터 공간에서 "이동할 수 있는 최대 거리"로 직관적으로 해석할 수 있다.
**The effective step size is bounded by $\alpha$, forming a trust region** — Since $|\hat{m}_t/\sqrt{\hat{v}_t}| \leq 1$ (Cauchy-Schwarz), the step size approximately never exceeds $\alpha$. This dramatically simplifies hyperparameter tuning — $\alpha$ can be intuitively interpreted as the "maximum distance to travel" in parameter space.

4. **SNR이 자동 annealing을 제공한다** — 최적점에 가까워지면 gradient의 기댓값(signal)은 0에 가까워지지만 분산(noise)은 유지되어, SNR = $\hat{m}_t/\sqrt{\hat{v}_t}$이 자연스럽게 감소한다. 별도의 learning rate schedule 없이도 학습 후반에 step이 줄어드는 효과를 얻는다.
**The SNR provides automatic annealing** — Near an optimum, the expected gradient (signal) approaches zero while variance (noise) persists, so SNR = $\hat{m}_t/\sqrt{\hat{v}_t}$ naturally decreases. This achieves diminishing steps in later training without an explicit learning rate schedule.

5. **AdaGrad와 RMSProp은 Adam의 특수 경우로 복원된다** — $\beta_1 = 0$이고 $\beta_2 \to 1$이면 AdaGrad에 대응하고, bias correction을 제거하면 RMSProp with momentum에 대응한다. 이 수학적 관계가 Adam의 설계를 정당화하며, bias correction 없이는 이 대응이 성립하지 않는다는 점이 중요하다.
**AdaGrad and RMSProp are recovered as special cases of Adam** — With $\beta_1 = 0$ and $\beta_2 \to 1$, Adam corresponds to AdaGrad; removing bias correction yields RMSProp with momentum. This mathematical relationship justifies Adam's design, and importantly, the correspondence breaks down without bias correction.

6. **AdaMax는 $L^\infty$ norm의 우아한 결과물이다** — $L^p$ norm을 $p \to \infty$로 보내면 max 연산이 나타나서, $u_t = \max(\beta_2 \cdot u_{t-1}, |g_t|)$라는 재귀적으로 매우 간단한 형태가 된다. 2차 moment의 bias correction이 불필요하고, step size의 bound가 $|\Delta_t| \leq \alpha$로 더 깔끔하다.
**AdaMax is an elegant result of the $L^\infty$ norm** — Taking the $L^p$ norm to $p \to \infty$ yields a max operation, producing the remarkably simple recursive form $u_t = \max(\beta_2 \cdot u_{t-1}, |g_t|)$. No bias correction is needed for the second moment, and the step size bound $|\Delta_t| \leq \alpha$ is cleaner.

7. **CNN에서는 1차 moment가 2차 moment보다 중요하다** — CIFAR-10 CNN 실험에서 2차 moment estimate가 몇 epoch 후 0에 가까워져서 $\epsilon$이 지배적이 되었다. 이 경우 Adam의 이점은 주로 momentum을 통한 mini-batch variance 감소에서 온다. 이는 모든 architecture에서 2차 moment가 동등하게 유용하지 않음을 시사한다.
**In CNNs, the 1st moment is more important than the 2nd moment** — In the CIFAR-10 CNN experiment, the second moment estimate approached zero after a few epochs, becoming dominated by $\epsilon$. Adam's advantage here comes mainly from variance reduction via momentum. This suggests the second moment is not equally useful across all architectures.

8. **기본 hyperparameter가 대부분의 문제에서 작동한다** — $\alpha=0.001$, $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$이라는 기본값이 logistic regression, MLP, CNN, VAE 등 다양한 실험에서 좋은 성능을 보였다. 이 "set and forget" 성질이 Adam이 기본 optimizer로 채택된 가장 실용적인 이유이다.
**Default hyperparameters work across most problems** — The defaults $\alpha=0.001$, $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$ performed well across diverse experiments (logistic regression, MLP, CNN, VAE). This "set and forget" property is the most practical reason Adam became the default optimizer.

---

## 4. Mathematical Summary / 수학적 요약

### Adam Algorithm (Algorithm 1)

**입력 / Input**:
- $\alpha$: stepsize (default 0.001)
- $\beta_1, \beta_2 \in [0, 1)$: exponential decay rates (default 0.9, 0.999)
- $f(\theta)$: stochastic objective function
- $\theta_0$: initial parameter vector

**초기화 / Initialization**:
- $m_0 \leftarrow 0$, $v_0 \leftarrow 0$, $t \leftarrow 0$

**반복 / Loop** (while not converged):

$$t \leftarrow t + 1$$

$$g_t = \nabla_\theta f_t(\theta_{t-1})$$

$$m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t \quad \text{(biased 1st moment estimate)}$$

$$v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2 \quad \text{(biased 2nd raw moment estimate)}$$

$$\hat{m}_t = m_t / (1 - \beta_1^t) \quad \text{(bias-corrected 1st moment)}$$

$$\hat{v}_t = v_t / (1 - \beta_2^t) \quad \text{(bias-corrected 2nd moment)}$$

$$\theta_t = \theta_{t-1} - \alpha \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon) \quad \text{(parameter update)}$$

### Bias Correction 유도 / Derivation

$$v_t = (1 - \beta_2) \sum_{i=1}^{t} \beta_2^{t-i} \cdot g_i^2$$

$$\mathbb{E}[v_t] = \mathbb{E}[g_t^2] \cdot (1 - \beta_2^t) + \zeta$$

$$\therefore \hat{v}_t = v_t / (1 - \beta_2^t) \quad \Rightarrow \quad \mathbb{E}[\hat{v}_t] \approx \mathbb{E}[g_t^2]$$

### Effective Step Size Bounds

$$|\Delta_t| = \left|\alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t}}\right| \lessapprox \alpha \quad \text{(since } |\mathbb{E}[g]| / \sqrt{\mathbb{E}[g^2]} \leq 1 \text{)}$$

### AdaMax Algorithm (Algorithm 2)

$$m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t$$

$$u_t = \max(\beta_2 \cdot u_{t-1}, |g_t|) \quad \text{(exponentially weighted } L^\infty \text{ norm)}$$

$$\theta_t = \theta_{t-1} - \frac{\alpha}{1 - \beta_1^t} \cdot \frac{m_t}{u_t}$$

### Convergence Guarantee (Convex Setting)

$$R(T) = O(\sqrt{T}) \quad \Rightarrow \quad R(T)/T = O(1/\sqrt{T}) \to 0$$

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1847 ──── Cauchy — Gradient Descent
           └─ 최적화의 기원 / Origin of optimization

1951 ──── Robbins & Monro — Stochastic Approximation
           └─ SGD의 이론적 기초 / Theoretical foundation of SGD

1964 ──── Polyak — SGD + Momentum
           └─ 관성을 이용한 수렴 가속화 / Accelerated convergence via inertia

1983 ──── Nesterov — Nesterov Accelerated Gradient (NAG)
           └─ Look-ahead gradient, overshooting 감소 / Look-ahead, reduces overshooting

1986 ──── Rumelhart, Hinton, Williams — Backpropagation [Paper #6]
           └─ 신경망에서 gradient 계산의 기초 / Foundation of gradient computation in NNs

2011 ──── Duchi et al. — AdaGrad
           └─ Per-parameter adaptive learning rate, sparse gradient에 강함
           └─ 문제: learning rate 단조 감소 → 학습 정체

2012 ──── Tieleman & Hinton — RMSProp (Coursera lecture, unpublished)
           └─ EMA of squared gradients (AdaGrad의 누적합 문제 해결)
           └─ 정식 논문 없음, bias correction 없음

2012 ──── Zeiler — AdaDelta
           └─ AdaGrad 개선, learning rate 불필요

2013 ──── Sutskever et al. — Importance of Initialization and Momentum
           └─ Nesterov momentum이 딥러닝에서 효과적임을 입증

2014 ──── Kingma & Ba — Adam ← THIS PAPER
           └─ Momentum + RMSProp + bias correction
           └─ O(√T) regret bound, AdaMax variant

2018 ──── Reddi et al. — AMSGrad
           └─ Adam의 non-convergence 사례 발견, v_t의 max 유지로 해결

2019 ──── Loshchilov & Hutter — AdamW (Decoupled Weight Decay)
           └─ Adam에서 L2 reg ≠ weight decay 문제 해결
           └─ 현재 가장 널리 사용되는 Adam 변형

2019 ──── Liu et al. — RAdam (Rectified Adam)
           └─ Learning rate warm-up의 필요성 분석 및 자동화

2023 ──── Chen et al. — Lion (Evolved Sign Momentum)
           └─ Sign-based, Adam보다 메모리 효율적
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **Rumelhart et al. (1986)** — Backpropagation [Paper #6] | Adam이 최적화하는 gradient를 계산하는 기반 알고리즘 / The foundational algorithm that computes the gradients Adam optimizes | Adam은 backpropagation으로 계산된 gradient에 adaptive learning rate를 적용하는 것 / Adam applies adaptive learning rates to gradients computed by backpropagation |
| **Duchi et al. (2011)** — AdaGrad | Adam의 직접적 선행 연구. Sparse gradient를 위한 adaptive learning rate 도입. Adam은 $\beta_1=0$, $\beta_2 \to 1$일 때 AdaGrad의 특수 경우 / Direct predecessor. Introduced adaptive learning rates for sparse gradients. Adam reduces to AdaGrad when $\beta_1=0$, $\beta_2 \to 1$ | Adam이 해결한 "learning rate 단조 감소" 문제의 원인 / Source of the "monotonically decreasing learning rate" problem Adam solves |
| **Tieleman & Hinton (2012)** — RMSProp | Adam의 직접적 선행 연구. EMA of squared gradients로 AdaGrad 문제 해결. Adam은 bias correction을 제거하면 RMSProp+momentum에 대응 / Direct predecessor. Fixed AdaGrad via EMA. Adam without bias correction corresponds to RMSProp+momentum | Adam의 2차 moment update의 직접적 출처 / Direct source of Adam's 2nd moment update |
| **Kingma & Welling (2013)** — VAE | 같은 제1저자(Kingma). VAE의 학습에 Adam이 사용됨. Section 6.4의 bias correction 실험에서 VAE를 실험 모델로 사용 / Same first author. Adam used to train VAEs. VAE used as the experimental model for bias correction experiments in Section 6.4 | Adam의 실용적 검증에 사용된 모델 / Model used for practical validation of Adam |
| **Sutskever et al. (2013)** — Importance of Initialization and Momentum | Nesterov momentum이 딥러닝에서 효과적임을 입증. 학습 후반에 momentum coefficient를 줄이면 수렴이 개선됨을 발견 — Adam의 $\beta_{1,t}$ decay와 연결 / Showed Nesterov momentum effective for deep learning. Found reducing momentum coefficient late in training improves convergence — connects to Adam's $\beta_{1,t}$ decay | Momentum 계열에서 Adam으로의 전이를 촉진 / Facilitated transition from Momentum lineage to Adam |
| **Reddi et al. (2018)** — AMSGrad | Adam이 특정 convex 문제에서 수렴하지 않는 반례를 제시. $\hat{v}_t$의 max를 유지하여 해결 / Presented counterexamples where Adam fails to converge on certain convex problems. Fixed by maintaining max of $\hat{v}_t$ | Adam의 이론적 한계를 발견하고 수정한 후속 연구 / Follow-up that identified and corrected Adam's theoretical limitations |
| **Loshchilov & Hutter (2019)** — AdamW | Adam에서 $L_2$ regularization과 weight decay가 동등하지 않음을 밝히고, decoupled weight decay를 제안. 현재 가장 널리 사용되는 Adam 변형 / Showed $L_2$ regularization ≠ weight decay in Adam, proposed decoupled weight decay. Currently the most widely used Adam variant | Adam의 가장 중요한 실용적 개선 / The most important practical improvement to Adam |

---

## 7. References / 참고문헌

- Duchi, J., Hazan, E., and Singer, Y. "Adaptive subgradient methods for online learning and stochastic optimization." *JMLR*, 12:2121–2159, 2011.
- Tieleman, T. and Hinton, G. "Lecture 6.5 - RMSProp." *COURSERA: Neural Networks for Machine Learning*, 2012.
- Zeiler, M. D. "Adadelta: An adaptive learning rate method." *arXiv:1212.5701*, 2012.
- Sutskever, I., Martens, J., Dahl, G., and Hinton, G. "On the importance of initialization and momentum in deep learning." *ICML-13*, pp. 1139–1147, 2013.
- Kingma, D. P. and Welling, M. "Auto-Encoding Variational Bayes." *ICLR*, 2013.
- Sohl-Dickstein, J., Poole, B., and Ganguli, S. "Fast large-scale optimization by unifying stochastic gradient and quasi-newton methods." *ICML-14*, pp. 604–612, 2014.
- Zinkevich, M. "Online convex programming and generalized infinitesimal gradient ascent." 2003.
- Graves, A. "Generating sequences with recurrent neural networks." *arXiv:1308.0850*, 2013.
- Amari, S. "Natural gradient works efficiently in learning." *Neural Computation*, 10(2):251–276, 1998.
- Pascanu, R. and Bengio, Y. "Revisiting natural gradient for deep networks." *arXiv:1301.3584*, 2013.
- Polyak, B. T. and Juditsky, A. B. "Acceleration of stochastic approximation by averaging." *SIAM J. Control and Optimization*, 30(4):838–855, 1992.
- Reddi, S. J., Kale, S., and Kumar, S. "On the convergence of Adam and beyond." *ICLR*, 2018.
- Loshchilov, I. and Hutter, F. "Decoupled weight decay regularization." *ICLR*, 2019.
