---
title: "Pre-Reading Briefing: Adam: A Method for Stochastic Optimization"
paper_id: "18_kingma_2014"
topic: Artificial_Intelligence
date: 2026-04-17
type: briefing
---

# Adam: A Method for Stochastic Optimization: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Kingma, D. P. & Ba, J. L. "Adam: A Method for Stochastic Optimization", ICLR 2015. arXiv:1412.6980
**Author(s)**: Diederik P. Kingma (University of Amsterdam, OpenAI), Jimmy Lei Ba (University of Toronto)
**Year**: 2014 (published at ICLR 2015)

---

## 1. 핵심 기여 / Core Contribution

Adam(Adaptive Moment Estimation)은 1차 gradient만을 사용하는 stochastic optimization 알고리즘입니다. 이 방법은 gradient의 1차 moment(평균)와 2차 moment(분산)의 exponential moving average를 결합하여 각 parameter에 대해 개별적인 adaptive learning rate를 계산합니다. 특히 초기 timestep에서 moment 추정의 bias를 교정하는 기법을 도입하여, 학습 초기의 불안정성을 해결했습니다. Adam은 구현이 간단하고, 메모리 효율적이며, hyperparameter 튜닝이 거의 불필요하여, 논문 발표 이후 딥러닝의 사실상 기본 optimizer가 되었습니다.

Adam (Adaptive Moment Estimation) is a first-order gradient-based stochastic optimization algorithm. It combines exponential moving averages of the first moment (mean) and second moment (uncentered variance) of the gradients to compute individual adaptive learning rates for each parameter. A key innovation is its initialization bias correction technique, which counters the zero-bias in the early timesteps when moment estimates are initialized at zero. Adam is simple to implement, memory-efficient, and requires little hyperparameter tuning — making it the de facto default optimizer in deep learning since its publication.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

2012년 AlexNet의 ImageNet 우승 이후 딥러닝이 급속히 성장하던 시기입니다. 네트워크는 점점 깊어지고 파라미터 수가 폭발적으로 증가하면서, 효율적인 optimization이 핵심 과제가 되었습니다. 기존의 SGD with momentum은 잘 작동했지만 learning rate 스케줄링에 대한 세심한 튜닝이 필요했습니다. 한편 AdaGrad(2011)는 sparse gradient에 강했지만 learning rate가 단조 감소하는 문제가 있었고, RMSProp(2012)은 이를 해결했지만 이론적 보장이 부족했습니다. Adam은 이 두 가지 접근법의 장점을 통합하면서 이론적 수렴 보장까지 제공한 논문입니다.

After AlexNet's ImageNet victory in 2012, deep learning was growing rapidly. Networks were getting deeper and parameter counts were exploding, making efficient optimization a critical challenge. Standard SGD with momentum worked well but required careful learning rate scheduling. AdaGrad (2011) handled sparse gradients effectively but suffered from monotonically decreasing learning rates. RMSProp (2012) addressed this but lacked theoretical convergence guarantees. Adam unified the strengths of both approaches while also providing theoretical convergence bounds.

### Optimizer의 계보: SGD에서 Adam까지 / Optimizer Genealogy: From SGD to Adam

Optimizer의 발전사에는 **두 가지 독립적인 계열**이 있으며, Adam은 이 두 계열이 합류한 지점입니다. SGD의 핵심 문제 2가지가 각 계열의 동기가 되었습니다:
- **문제 A**: gradient가 noisy해서 진동이 심함 → **Momentum 계열**이 해결
- **문제 B**: 모든 파라미터에 동일한 learning rate를 쓰는 건 비효율적 → **Adaptive Learning Rate 계열**이 해결

The history of optimizers has **two independent lineages**, and Adam is the point where they converge. Two core problems with SGD motivated each lineage:
- **Problem A**: Noisy gradients cause oscillation → **Momentum lineage** addresses this
- **Problem B**: A single learning rate for all parameters is inefficient → **Adaptive learning rate lineage** addresses this

---

#### 기원: Gradient Descent와 SGD / Origin: Gradient Descent and SGD

**Vanilla Gradient Descent**는 가장 기본적인 형태입니다:

$$\theta_{t+1} = \theta_t - \alpha \cdot \nabla f(\theta_t)$$

전체 데이터셋에 대해 gradient를 계산하고, 그 반대 방향으로 파라미터를 업데이트합니다. 문제는 데이터가 크면 한 번 업데이트하는 데 너무 오래 걸린다는 것입니다.

Vanilla Gradient Descent computes the gradient over the entire dataset and updates parameters in the opposite direction. The problem is that a single update becomes prohibitively expensive for large datasets.

**Stochastic Gradient Descent (SGD)**는 전체 데이터 대신 mini-batch로 gradient를 추정합니다:

$$\theta_{t+1} = \theta_t - \alpha \cdot g_t \quad \text{where } g_t = \nabla f_{\text{batch}}(\theta_t)$$

훨씬 빠르지만, gradient 추정에 noise가 생겨 수렴 경로가 지그재그로 흔들립니다.

SGD estimates the gradient from a mini-batch instead of the full dataset. Much faster, but the gradient estimate is noisy, causing the convergence path to zigzag.

---

#### 계열 1: Momentum 계열 (문제 A 해결) / Lineage 1: Momentum (Solving Problem A)

**SGD + Momentum (Polyak, 1964)** — 이전 gradient의 **관성(velocity)**을 유지합니다:

$$v_t = \beta \cdot v_{t-1} + g_t$$
$$\theta_{t+1} = \theta_t - \alpha \cdot v_t$$

직관적으로, 공이 언덕을 굴러 내려갈 때 관성이 붙는 것과 같습니다. 같은 방향의 gradient가 반복되면 가속되고, 방향이 바뀌면 진동이 상쇄됩니다.

Intuitively, like a ball rolling down a hill gaining inertia. Repeated gradients in the same direction accelerate the update; opposing directions cancel out oscillations.

```
SGD만 쓸 때 (지그재그):          Momentum 추가 (부드러운 경로):
SGD only (zigzag):             With Momentum (smooth):
   ╱╲╱╲╱╲                        ──────────→
      ╱╲╱╲→ 목표(goal)                    → 목표(goal)
```

**Nesterov Accelerated Gradient (NAG, Nesterov 1983)** — Momentum의 개선판입니다. "먼저 momentum 방향으로 이동한 후, 그 위치에서 gradient를 계산"합니다:

$$v_t = \beta \cdot v_{t-1} + \nabla f(\theta_t - \beta \cdot v_{t-1})$$
$$\theta_{t+1} = \theta_t - \alpha \cdot v_t$$

일반 momentum은 "현재 위치에서 gradient를 보고 관성을 더하는" 반면, Nesterov는 **"관성으로 갈 곳을 먼저 보고 그곳의 gradient로 보정"**합니다. 이 "look-ahead"가 overshooting을 줄여줍니다. Sutskever et al. (2013)은 이 Nesterov momentum이 딥러닝에서도 효과적임을 보여주었습니다.

Standard momentum computes the gradient at the current position and then adds inertia. Nesterov first looks ahead to where momentum would take us, then computes the gradient there. This "look-ahead" reduces overshooting. Sutskever et al. (2013) demonstrated that Nesterov momentum is effective for deep learning as well.

---

#### 계열 2: Adaptive Learning Rate 계열 (문제 B 해결) / Lineage 2: Adaptive Learning Rate (Solving Problem B)

**AdaGrad (Duchi et al., 2011)** — 핵심 아이디어: **자주 업데이트된 파라미터는 learning rate를 줄이고, 드물게 업데이트된 파라미터는 크게 유지**합니다.

Core idea: **reduce learning rate for frequently updated parameters, keep it large for rarely updated ones**.

$$G_t = \sum_{i=1}^{t} g_i^2 \quad \text{(gradient 제곱의 누적합 / cumulative sum of squared gradients)}$$
$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{G_t} + \epsilon} \cdot g_t$$

Sparse feature (예: NLP에서 희귀 단어)에 매우 효과적입니다. 자주 등장하는 feature의 gradient는 $G_t$가 빠르게 커져서 learning rate가 줄고, 드문 feature는 큰 learning rate를 유지합니다.

Highly effective for sparse features (e.g., rare words in NLP). Frequently occurring features accumulate large $G_t$, shrinking their learning rate, while rare features maintain a large learning rate.

**치명적 문제 / Critical problem**: $G_t$가 **단조 증가**합니다. 학습이 진행될수록 learning rate가 계속 줄어들어, 결국 너무 작아져서 학습이 멈춥니다. 이것이 "aggressive learning rate decay" 문제입니다.

$G_t$ **monotonically increases**. As training progresses, learning rates shrink continuously, eventually becoming too small for any meaningful update. This is the "aggressive learning rate decay" problem.

**AdaDelta (Zeiler, 2012)** — AdaGrad의 단조 감소 문제를 해결하기 위해, gradient 제곱의 전체 누적합 대신 exponential moving average를 사용합니다. 또한 learning rate $\alpha$ 자체도 필요 없도록 설계했습니다. 하지만 실전에서는 RMSProp에 밀렸습니다.

Uses an EMA of squared gradients instead of the cumulative sum. Also eliminates the need for a manual learning rate $\alpha$. However, it was eclipsed by RMSProp in practice.

**RMSProp (Tieleman & Hinton, 2012)** — Geoff Hinton이 Coursera 강의에서 소개한 방법으로, **정식 논문이 없습니다** (강의 슬라이드에서 제안). AdaGrad의 문제를 가장 간결하게 해결합니다:

Introduced by Geoff Hinton in a Coursera lecture — **no formal paper exists**. The most concise fix to AdaGrad's problem:

$$v_t = \beta \cdot v_{t-1} + (1-\beta) \cdot g_t^2$$
$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{v_t} + \epsilon} \cdot g_t$$

AdaGrad와의 핵심 차이: gradient 제곱의 **누적합** 대신 **exponential moving average**를 사용합니다. 이렇게 하면 오래된 gradient의 영향이 자연스럽게 사라져서, learning rate가 0으로 수렴하는 문제가 없습니다.

Key difference from AdaGrad: uses an **exponential moving average** instead of a **cumulative sum** of squared gradients. Old gradients naturally fade away, preventing the learning rate from converging to zero.

```
AdaGrad:  G_t = g₁² + g₂² + ... + g_t²    (계속 커짐 → lr → 0 / keeps growing → lr → 0)
RMSProp:  v_t = 0.9·v_{t-1} + 0.1·g_t²     (최근 값 위주 → 안정적 / recent values dominate → stable)
```

**RMSProp의 한계 / RMSProp's limitations**: 이론적 수렴 보장이 없고, bias correction이 없어서 $\beta$가 1에 가까울 때 초기 학습이 불안정할 수 있습니다. 또한 정식 논문이 아닌 강의 슬라이드에서 제안되어 체계적인 분석이 부족했습니다.

No theoretical convergence guarantee. No bias correction, leading to potential instability in early timesteps when $\beta$ is close to 1. Originally proposed in lecture slides rather than a formal paper, lacking systematic analysis.

---

#### 두 계열의 합류: Adam (2014) / Convergence of Two Lineages: Adam (2014)

Adam은 위 두 계열을 **하나로 통합**합니다:

Adam **unifies both lineages**:

| 구성 요소 / Component | 출처 / Origin | Adam에서의 역할 / Role in Adam |
|---|---|---|
| $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$ | **Momentum** 계열 | 1차 moment — gradient의 방향(평균) 추정 / 1st moment — estimates gradient direction (mean) |
| $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$ | **RMSProp** 계열 | 2차 moment — gradient의 크기(분산) 추정 / 2nd moment — estimates gradient magnitude (variance) |
| $\hat{m}_t, \hat{v}_t$ (bias correction) | **Adam의 독자적 기여** | 초기화 편향 보정 / Initialization bias correction (Adam's original contribution) |

```
          Momentum 계열                    Adaptive 계열
          Momentum Lineage                 Adaptive Lineage
          (gradient 방향/direction)         (gradient 크기/magnitude)
               │                               │
    SGD+Momentum (1964)                  AdaGrad (2011)
         │                                │         │
    Nesterov (1983)                 AdaDelta (2012)  RMSProp (2012)
         │                                          │
         └──────────────┬───────────────────────────┘
                        │
                   Adam (2014)
                   = 1st moment (momentum)
                   + 2nd moment (RMSProp)
                   + bias correction (new)
                        │
                   AdaMax (2014, same paper)
                   = L∞ norm variant
```

#### Adam이 승리한 이유 / Why Adam Won

1. **Momentum의 장점**: noisy gradient를 평활화하여 안정적인 방향 추정 / Smooths noisy gradients for stable direction estimation
2. **RMSProp의 장점**: 파라미터별 adaptive learning rate / Per-parameter adaptive learning rate
3. **Bias correction**: 초기 학습의 안정성 보장 (RMSProp에 없던 것) / Ensures stability in early training (absent in RMSProp)
4. **Trust region 성질**: step size가 대략 $\alpha$로 bounded → hyperparameter 튜닝이 쉬움 / Step size approximately bounded by $\alpha$ → easy hyperparameter tuning
5. **구현 간단, 메모리 효율적**: 파라미터당 2개의 추가 변수($m_t, v_t$)만 필요 / Simple implementation, only 2 extra variables per parameter ($m_t, v_t$)

### 타임라인 / Timeline

```
1847 ──── Cauchy — Gradient Descent
           └─ 최적화의 기원 / Origin of optimization

1951 ──── Robbins & Monro — Stochastic Approximation
           └─ SGD의 이론적 기초 / Theoretical foundation of SGD

1964 ──── Polyak — SGD + Momentum
           └─ 관성을 이용한 수렴 가속화 / Accelerated convergence via inertia

1983 ──── Nesterov — Nesterov Accelerated Gradient (NAG)
           └─ Look-ahead gradient로 overshooting 감소
           └─ Reduces overshooting via look-ahead gradient

2011 ──── AdaGrad (Duchi et al.)
           └─ Adaptive per-parameter learning rates; good for sparse gradients
           └─ Problem: learning rate monotonically decreases → training stalls

2012 ──── RMSProp (Tieleman & Hinton, Coursera lecture)
           └─ EMA of squared gradients (not cumulative like AdaGrad)
           └─ Unpublished — introduced in a lecture, no formal paper

2012 ──── AdaDelta (Zeiler)
           └─ Another approach to fix AdaGrad's diminishing learning rate

2013 ──── Sutskever et al. — Importance of initialization and momentum
           └─ Nesterov momentum shown effective for deep networks

2014 ──── Adam (Kingma & Ba) ← THIS PAPER
           └─ Combines momentum (1st moment) + RMSProp (2nd moment)
           └─ Adds bias correction for initialization
           └─ Provides O(√T) regret bound

2014 ──── AdaMax (Kingma & Ba, same paper)
           └─ L∞ norm variant of Adam — simpler bound on step size
```

---

## 3. 필요한 배경 지식 / Prerequisites

### 수학적 배경 / Mathematical Background

| 개념 / Concept | 설명 / Description |
|---|---|
| **Gradient Descent** | 목적함수의 gradient 반대 방향으로 파라미터를 업데이트하는 최적화 기법 / Optimization by updating parameters in the negative gradient direction |
| **Stochastic Gradient Descent (SGD)** | 전체 데이터 대신 mini-batch로 gradient를 추정하여 업데이트 / Using mini-batch gradient estimates instead of full-batch |
| **Momentum** | 이전 gradient 방향의 관성을 유지하여 수렴을 가속화 / Maintaining velocity from previous gradient directions to accelerate convergence |
| **Exponential Moving Average (EMA)** | $v_t = \beta \cdot v_{t-1} + (1-\beta) \cdot x_t$ — 최근 값에 더 큰 가중치를 두는 이동 평균 / Weighted average giving more importance to recent values |
| **Moment estimation** | 1차 moment = 평균 $\mathbb{E}[g]$, 2차 moment = $\mathbb{E}[g^2]$ (uncentered variance) / 1st moment = mean, 2nd raw moment = expected squared value |
| **Convex optimization** | 볼록 함수의 최적화 — 이론적 수렴 분석의 기초 / Optimization of convex functions — basis for convergence analysis |
| **Online learning / Regret** | Regret $R(T) = \sum_{t=1}^{T}[f_t(\theta_t) - f_t(\theta^*)]$ — 온라인 학습에서 최적해 대비 누적 손실 / Cumulative loss compared to the best fixed point |

### 선행 논문 / Prior Papers in This Series

- **Paper #6 (Rumelhart et al., 1986)**: Backpropagation — gradient 계산의 기초 / Foundation of gradient computation in neural networks

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **Adam** | **Ada**ptive **M**oment Estimation의 약자. 1차, 2차 moment의 적응적 추정을 결합한 optimizer / Portmanteau of Adaptive Moment Estimation |
| **First moment estimate ($m_t$)** | Gradient의 exponential moving average — gradient의 평균 방향을 추정 / EMA of gradients — estimates the mean direction |
| **Second moment estimate ($v_t$)** | Gradient 제곱의 exponential moving average — gradient의 크기(분산)를 추정 / EMA of squared gradients — estimates uncentered variance |
| **Bias correction** | 초기화 편향 보정: $\hat{m}_t = m_t/(1-\beta_1^t)$, $\hat{v}_t = v_t/(1-\beta_2^t)$ / Corrects the zero-initialization bias in early timesteps |
| **Stepsize ($\alpha$)** | 학습률 — Adam에서는 effective step이 대략 $\alpha$로 bounded됨 / Learning rate — in Adam, effective steps are approximately bounded by $\alpha$ |
| **$\beta_1, \beta_2$** | Moment estimate의 decay rate. 기본값: $\beta_1=0.9$, $\beta_2=0.999$ / Exponential decay rates for moment estimates |
| **$\epsilon$** | 수치 안정성을 위한 작은 상수 (기본값: $10^{-8}$) / Small constant for numerical stability |
| **Signal-to-Noise Ratio (SNR)** | $\hat{m}_t / \sqrt{\hat{v}_t}$ — 최적점에 가까워지면 SNR이 감소하여 자연스러운 annealing 효과 / Ratio that naturally decreases near optima, providing automatic annealing |
| **Trust region** | Effective step size가 $\alpha$로 bounded되어, 파라미터 공간에서 신뢰 영역을 형성 / The bounded step size creates a trust region in parameter space |
| **AdaMax** | Adam의 $L^p$ norm 일반화에서 $p \to \infty$인 특수 경우 — $L^\infty$ norm 사용 / Special case of $L^p$ norm generalization of Adam where $p \to \infty$ |
| **Regret bound** | $R(T)/T = O(1/\sqrt{T})$ — Adam의 이론적 수렴 보장 (convex setting) / Theoretical convergence guarantee in the online convex optimization framework |

---

## 5. 수식 미리보기 / Equations Preview

### (1) Adam 핵심 업데이트 규칙 / Core Update Rule

$$m_t = \beta_1 \cdot m_{t-1} + (1-\beta_1) \cdot g_t$$
$$v_t = \beta_2 \cdot v_{t-1} + (1-\beta_2) \cdot g_t^2$$

- $m_t$: gradient의 1차 moment (평균) 추정 / First moment estimate (mean of gradients)
- $v_t$: gradient의 2차 moment (uncentered variance) 추정 / Second moment estimate (uncentered variance)
- $\beta_1, \beta_2$: decay rate (기본값 0.9, 0.999) / Decay rates for moment estimates
- $g_t$: timestep $t$에서의 gradient / Gradient at timestep $t$

### (2) Bias 보정 / Bias Correction

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

초기에 $m_0 = 0, v_0 = 0$으로 초기화하므로, 초기 timestep에서 moment 추정이 0 방향으로 편향됩니다. $(1-\beta^t)$로 나누어 이를 보정합니다.

Since $m_0 = 0$ and $v_0 = 0$, the moment estimates are biased toward zero in early timesteps. Dividing by $(1 - \beta^t)$ corrects this initialization bias.

### (3) 파라미터 업데이트 / Parameter Update

$$\theta_t = \theta_{t-1} - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

- $\alpha$: stepsize (learning rate)
- $\epsilon$: 수치 안정성 상수 ($10^{-8}$) / Numerical stability constant
- 효과: gradient의 평균 방향으로 이동하되, gradient의 크기(분산)로 정규화 / Move in the mean gradient direction, normalized by gradient magnitude

### (4) Effective step size의 bound

$$|\Delta_t| \lessapprox \alpha$$

Effective step size $\Delta_t = \alpha \cdot \hat{m}_t / \sqrt{\hat{v}_t}$는 대략 $\alpha$로 bounded됩니다. 이는 $|\mathbb{E}[g]|/\sqrt{\mathbb{E}[g^2]} \leq 1$이기 때문입니다. 이 성질이 Adam의 "trust region" 해석을 가능하게 합니다.

The effective step is approximately bounded by $\alpha$ because $|\mathbb{E}[g]|/\sqrt{\mathbb{E}[g^2]} \leq 1$. This property enables the "trust region" interpretation of Adam.

### (5) AdaMax 업데이트 ($L^\infty$ variant)

$$u_t = \max(\beta_2 \cdot u_{t-1}, |g_t|)$$
$$\theta_t = \theta_{t-1} - \frac{\alpha}{1-\beta_1^t} \cdot \frac{m_t}{u_t}$$

$L^p$ norm을 $p \to \infty$로 보내면 max 연산이 되어, bias correction이 필요 없는 깔끔한 형태가 됩니다.

Taking the $L^p$ norm to $p \to \infty$ yields a max operation, producing a clean formulation that doesn't require bias correction for the second moment.

---

## 6. 읽기 가이드 / Reading Guide

### 핵심 구간 / Must-Read Sections

| 구간 / Section | 중요도 / Priority | 이유 / Why |
|---|---|---|
| **Algorithm 1** (p.2) | ★★★ | Adam의 전체 알고리즘 — 이것만 이해하면 핵심을 파악한 것 / The complete algorithm — understanding this is understanding Adam |
| **Section 2** (pp.2-3) | ★★★ | 알고리즘 설명과 update rule의 성질 (trust region, SNR, scale invariance) / Algorithm description and key properties |
| **Section 3** (pp.3-4) | ★★★ | Bias correction의 유도 — Adam의 핵심 혁신 중 하나 / Derivation of bias correction — one of Adam's key innovations |
| **Section 5** (pp.4-5) | ★★☆ | RMSProp, AdaGrad와의 관계 — Adam의 positioning 이해 / Relationship to RMSProp and AdaGrad |
| **Section 7.1** (pp.8-9) | ★★☆ | AdaMax — $L^\infty$ norm variant, 우아한 수학적 유도 / Elegant mathematical derivation of the infinity norm variant |
| **Section 4** (p.4) | ★☆☆ | Convergence analysis — 이론적 보장, 수학적으로 무거움 / Theoretical guarantee, mathematically heavy |
| **Section 6** (pp.5-8) | ★☆☆ | 실험 결과 — 결과 그래프 위주로 빠르게 / Experiments — skim the result figures |

### 읽기 순서 제안 / Suggested Reading Order

1. **Algorithm 1** (p.2)을 먼저 정독 — 전체 알고리즘을 한눈에 파악
2. **Section 2**를 읽으며 각 단계의 직관 이해 (trust region, SNR)
3. **Section 3**의 bias correction 유도를 따라가기
4. **Section 5**에서 RMSProp/AdaGrad와의 연결 확인
5. **Section 7.1**의 AdaMax 유도 (수학적으로 아름다운 부분)
6. **Section 6** 실험은 Figure 1-4를 중심으로 빠르게 스캔
7. **Section 4**는 수렴 정리 결과만 확인 (증명은 Appendix)

### 주의할 점 / Things to Watch For

- **Bias correction이 왜 필요한지**: $m_0 = 0$에서 시작하면 초기 gradient 추정이 실제보다 작아짐. 특히 $\beta_2$가 1에 가까울수록 문제가 심각 → Section 3에서 수학적으로 유도
- **RMSProp과의 미묘한 차이**: RMSProp with momentum은 rescaled gradient에 momentum을 적용하지만, Adam은 gradient 자체의 1차/2차 moment를 별도로 추적
- **SNR의 자동 annealing 효과**: 최적점에 가까워지면 gradient의 SNR이 감소 → step size가 자연스럽게 줄어듦

---

## 7. 현대적 의의 / Modern Significance

### 현재의 위상 / Current Status

Adam은 2014년 발표 이후 딥러닝에서 가장 널리 사용되는 optimizer입니다. PyTorch, TensorFlow 등 모든 주요 프레임워크에 기본 구현되어 있으며, 새로운 모델을 학습할 때 첫 번째로 시도하는 optimizer로 자리잡았습니다.

Adam has been the most widely used optimizer in deep learning since 2014. It is implemented as a default in all major frameworks (PyTorch, TensorFlow) and is typically the first optimizer tried when training new models.

### 후속 발전 / Subsequent Developments

| 발전 / Development | 설명 / Description |
|---|---|
| **AMSGrad (2018)** | Adam의 수렴 실패 사례를 지적하고, $v_t$의 maximum을 유지하여 해결 / Fixed convergence failure cases by maintaining max of $v_t$ |
| **AdamW (Loshchilov & Hutter, 2019)** | Weight decay를 Adam에 올바르게 적용하는 방법 — 현재 가장 널리 사용되는 변형 / Proper weight decay for Adam — currently the most widely used variant |
| **LAMB/LARS** | 대규모 batch training을 위한 layer-wise adaptive rate scaling / Layer-wise adaptive rate scaling for large-batch training |
| **RAdam (2019)** | Learning rate warm-up의 필요성을 분석하고 자동화 / Analyzed and automated learning rate warm-up |
| **Lion (2023)** | Sign-based optimizer — Adam보다 메모리 효율적 / Sign-based optimizer, more memory efficient than Adam |

### 한계와 논쟁 / Limitations and Debates

- **Generalization gap**: Adam으로 학습한 모델이 잘 튜닝된 SGD+momentum보다 일반화 성능이 떨어질 수 있다는 보고가 있음 (특히 computer vision)
- **Weight decay 문제**: L2 regularization과 weight decay가 Adam에서 동등하지 않음 → AdamW가 이를 해결
- **Non-convergence 사례**: 특정 조건에서 Adam이 수렴하지 않을 수 있음 → AMSGrad가 이를 분석

Despite these limitations, Adam (especially AdamW) remains the workhorse optimizer of modern deep learning, from training LLMs to diffusion models.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
