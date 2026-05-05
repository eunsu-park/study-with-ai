---
title: "Denoising Diffusion Probabilistic Models"
authors: [Jonathan Ho, Ajay Jain, Pieter Abbeel]
year: 2020
journal: "Advances in Neural Information Processing Systems (NeurIPS) 33"
doi: "arXiv:2006.11239"
topic: Artificial_Intelligence
tags: [diffusion-model, generative-model, score-matching, denoising-score-matching, U-Net, variational-inference, langevin-dynamics, image-synthesis]
status: completed
date_started: 2026-04-28
date_completed: 2026-04-28
---

# 35. Denoising Diffusion Probabilistic Models / 잡음 제거 확산 확률 모델

---

## 1. Core Contribution / 핵심 기여

**한국어**
이 논문은 **확산 확률 모델(Diffusion Probabilistic Model)**을 GAN과 경쟁 가능한 고품질 이미지 생성기로 끌어올린 결정적 작품입니다. Sohl-Dickstein et al. (2015)에서 제시된 확산 모델 framework를 단순화하고 재매개변수화하여, 마침내 실용적 결과를 얻었습니다. 핵심 기여는 다음 세 가지로 요약됩니다.

(1) **노이즈 예측($\epsilon$-prediction) 재매개변수화**: 가우시안 reverse process의 평균 $\mu_\theta(x_t, t)$를 직접 예측하는 대신, **forward step에서 더해진 노이즈 $\epsilon$**을 신경망이 예측하도록 변환. 이는 단순한 산수 변환이지만, 학습 손실을 극도로 단순화시킵니다.

(2) **단순화된 손실 $L_\text{simple}$**: 정식 변분 하한(ELBO)을 단순한 가중 MSE
$$L_\text{simple} = \mathbb{E}_{t, x_0, \epsilon}\!\left[\big\|\epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\ t)\big\|^2\right]$$
으로 환원. 식의 가중치를 떨어뜨리는 것이 오히려 샘플 품질을 향상시킨다는 비직관적 발견.

(3) **denoising score matching과의 등가성 입증**: $\epsilon$-prediction은 다중 노이즈 스케일에서의 denoising score matching과 정확히 같은 학습 목표를 갖고, 샘플링은 annealed Langevin dynamics와 같다. 즉 DDPM = (variational inference 관점) = (score-based generative model 관점).

결과: CIFAR10에서 unconditional FID **3.17** (당시 SOTA), Inception Score **9.46**. LSUN bedroom 256² FID **4.90**, LSUN church 256² FID **7.89**, CelebA-HQ 256²에서 ProgressiveGAN과 비슷한 품질. 모델 크기는 CIFAR10용 35.7M, 256² 모델 114M 매개변수.

**English**
This paper transforms **diffusion probabilistic models** from a five-year-old theoretical curiosity into a state-of-the-art image generator. Building on Sohl-Dickstein et al. (2015), the authors simplify and reparameterize the framework until practical, GAN-competitive results emerge. Three contributions distinguish the work.

(1) **Noise-prediction ($\epsilon$-prediction) reparameterization**: instead of learning the reverse-process Gaussian mean $\mu_\theta(x_t, t)$ directly, the network predicts the **noise $\epsilon$** that was added in the forward step. A simple algebraic substitution — but it dramatically simplifies the loss.

(2) **Simplified loss $L_\text{simple}$**: the formal variational bound (ELBO) collapses to a weighted MSE objective $\mathbb{E}\!\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$. Counter-intuitively, *dropping* the variance-dependent weight yields *better* sample quality.

(3) **Equivalence to denoising score matching**: the $\epsilon$-prediction objective is identical to denoising score matching across multiple noise scales, and the sampling procedure is annealed Langevin dynamics. DDPM unifies the variational-inference and score-based viewpoints.

Results: unconditional CIFAR10 FID **3.17** (SOTA at publication), Inception Score **9.46**; LSUN bedroom 256² FID **4.90**, LSUN church 256² FID **7.89**; CelebA-HQ 256² rivals ProgressiveGAN. The CIFAR10 model has 35.7M parameters; 256² models 114M.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction (§1) / 서론

**한국어**
저자들은 GAN, autoregressive model, flow, VAE가 모두 고품질 샘플을 생성하지만, 확산 모델만은 5년간 잠자고 있었음을 지적합니다 (Sohl-Dickstein 2015 이후 발전이 거의 없었음). 본 논문은 확산 모델로 처음으로 high-quality 샘플을 생성합니다. 핵심 발견은 **특정 매개변수화가 다중 노이즈 스케일에서의 denoising score matching과 등가**라는 점입니다.

또한 저자들은 정직하게 한계도 언급합니다: 다른 likelihood-based 모델 대비 **로그가능도가 경쟁력 없음**. 그러나 lossless codelength의 대부분이 imperceptible image detail을 기술하는 데 사용된다는 정량 분석을 §4.3에서 제시합니다. 샘플링 절차는 일종의 **progressive lossy decompression** — autoregressive decoding의 일반화로 볼 수 있다고 주장.

**English**
GANs, autoregressive models, flows, and VAEs all produce high-quality samples, but diffusion models had stagnated for five years since Sohl-Dickstein et al. (2015). This paper achieves the first high-quality diffusion samples. The key insight: **a specific parameterization makes diffusion models equivalent to denoising score matching across multiple noise scales** during training and to annealed Langevin dynamics during sampling.

The authors are candid about a limitation: **log-likelihood is not competitive** with other likelihood-based models. However, §4.3 shows that most of the lossless codelength is consumed describing imperceptible image details, and reframes diffusion sampling as **progressive lossy decompression** — a generalization of autoregressive decoding.

### Part II: Background (§2) / 배경

#### 2.1 Forward (diffusion) process / 순방향(확산) 과정

**한국어**
확산 모델은 latent 변수 $x_1, \ldots, x_T$를 갖는 latent variable model:
$$p_\theta(x_0) = \int p_\theta(x_{0:T})\, dx_{1:T}$$
모든 latent는 데이터 $x_0$와 같은 차원입니다.

**Forward (diffusion) process** $q$는 학습되지 않고 **고정된 가우시안 Markov chain**입니다 (식 2):
$$q(x_{1:T}|x_0) = \prod_{t=1}^{T} q(x_t|x_{t-1}), \qquad q(x_t|x_{t-1}) = \mathcal{N}(x_t;\ \sqrt{1-\beta_t}\, x_{t-1},\ \beta_t I) \tag{2}$$

여기서 $\beta_1, \ldots, \beta_T$는 분산 스케줄. 본 논문은 $T=1000$, $\beta_1=10^{-4}$에서 $\beta_T=0.02$까지 선형 증가.

**핵심 성질 (식 4)**: $\alpha_t := 1 - \beta_t$, $\bar\alpha_t := \prod_{s=1}^t \alpha_s$로 정의하면
$$q(x_t|x_0) = \mathcal{N}(x_t;\ \sqrt{\bar\alpha_t}\, x_0,\ (1-\bar\alpha_t) I) \tag{4}$$

이것이 학습을 가능하게 하는 **마법의 식**입니다. 임의의 $t$에서 한 번에 $x_t$를 샘플링할 수 있어 Markov chain 시뮬레이션 불필요. 재매개변수화하면:
$$x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

**English**
Diffusion models are latent-variable models with $x_1, \ldots, x_T$ matching the data dimension. The **forward process** $q$ is fixed (no learnable parameters) — a Gaussian Markov chain (Eq. 2):
$$q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}\, x_{t-1},\ \beta_t I)$$

Here $T=1000$ with linear $\beta$ schedule from $10^{-4}$ to $0.02$. The crucial closed-form (Eq. 4) lets us sample $x_t$ at any $t$ in one shot:
$$q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar\alpha_t}\, x_0,\ (1-\bar\alpha_t)I)$$
i.e. $x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$. This is what makes training a single-step random sample.

#### 2.2 Reverse process / 역방향 과정

**한국어**
**Reverse process** $p_\theta$는 학습되며 가우시안 conditional로 매개변수화 (식 1):
$$p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1}|x_t), \qquad p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1};\ \mu_\theta(x_t, t),\ \Sigma_\theta(x_t, t)) \tag{1}$$

$p(x_T) = \mathcal{N}(0, I)$로 시작합니다. **왜 가우시안인가?** $\beta_t$가 충분히 작으면 reverse도 가우시안이 됩니다 (Feller, 1949). 즉 forward와 reverse가 **같은 함수형**을 갖는다는 보장.

#### 2.3 Variational bound / 변분 하한

**한국어**
음의 로그가능도의 상한은 (식 3):
$$\mathbb{E}[-\log p_\theta(x_0)] \leq \mathbb{E}_q\!\left[-\log \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}\right] = L \tag{3}$$

이것은 분산 감소를 위해 KL divergence로 다시 쓸 수 있습니다 (식 5):
$$L = \mathbb{E}_q\!\bigg[\underbrace{D_{\text{KL}}(q(x_T|x_0)\,\|\,p(x_T))}_{L_T} + \sum_{t>1} \underbrace{D_{\text{KL}}(q(x_{t-1}|x_t,x_0)\,\|\,p_\theta(x_{t-1}|x_t))}_{L_{t-1}} \underbrace{- \log p_\theta(x_0|x_1)}_{L_0}\bigg] \tag{5}$$

**Forward process posterior** (식 6, 7) — $x_0$를 조건으로 알 때 reverse가 닫힌 형태로 가우시안:
$$q(x_{t-1}|x_t, x_0) = \mathcal{N}(x_{t-1};\ \tilde\mu_t(x_t, x_0),\ \tilde\beta_t I) \tag{6}$$

$$\tilde\mu_t(x_t, x_0) = \frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t} x_0 + \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t} x_t, \qquad \tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t \tag{7}$$

따라서 식 (5)의 모든 KL은 두 가우시안 간의 KL — **Rao-Blackwellized 닫힌 형태**로 계산 가능 (Monte Carlo 불필요).

**English**
The negative log-likelihood is upper-bounded by Eq. (3), which after rewriting (Eq. 5) decomposes as $L = L_T + \sum_{t>1} L_{t-1} + L_0$, where each $L_{t-1}$ is a KL between Gaussians (Eq. 6, 7). The **forward posterior** $q(x_{t-1}|x_t, x_0)$ has closed-form mean $\tilde\mu_t$ and variance $\tilde\beta_t$. All KL terms have closed forms (Rao-Blackwellized) — no Monte-Carlo variance needed.

### Part III: Diffusion models and denoising autoencoders (§3) / 확산 모델과 잡음 제거 오토인코더

#### 3.1 Forward process and $L_T$

**한국어**
저자들은 $\beta_t$를 학습 가능 매개변수로 두지 않고 **고정 상수**로 사용합니다. 따라서 $q$에 학습 가능 매개변수가 없으므로 $L_T$는 학습 중 상수이며 **무시 가능**.

#### 3.2 Reverse process and $L_{1:T-1}$ — the parameterization choice / 역과정 매개변수화

**한국어**
이것이 본 논문의 **핵심 기여**입니다. 두 가지 선택을 합니다.

**선택 1: Variance $\Sigma_\theta$**. $\Sigma_\theta(x_t, t) = \sigma_t^2 I$로 학습하지 않고 **시간 의존 상수**. 두 가지 후보가 비슷한 결과:
- $\sigma_t^2 = \beta_t$ — $x_0 \sim \mathcal{N}(0, I)$ 가정하 최적
- $\sigma_t^2 = \tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$ — $x_0$가 결정적일 때 최적

이 두 값은 reverse process entropy의 상/하한에 해당.

**선택 2: Mean $\mu_\theta$ — $\epsilon$-prediction reparameterization**. 

$L_{t-1}$은 다음과 같이 (식 8):
$$L_{t-1} = \mathbb{E}_q\!\left[\frac{1}{2\sigma_t^2}\big\|\tilde\mu_t(x_t, x_0) - \mu_\theta(x_t, t)\big\|^2\right] + C \tag{8}$$

$\mu_\theta$를 $\tilde\mu_t$의 예측기로 두는 것이 가장 직관적이지만, 저자들은 식 (4)의 재매개변수화 $x_t(x_0, \epsilon) = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon$를 식 (7)에 대입하여 $\tilde\mu_t$를 $\epsilon$의 함수로 다시 씁니다:
$$\tilde\mu_t = \frac{1}{\sqrt{\alpha_t}}\!\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon\right)$$

따라서 자연스럽게 (식 11):
$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\!\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t, t)\right) \tag{11}$$

여기서 $\epsilon_\theta$는 **노이즈를 예측하는 신경망**. 이를 식 (8)에 대입하면 (식 12):
$$L_{t-1} = \mathbb{E}_{x_0, \epsilon}\!\left[\frac{\beta_t^2}{2\sigma_t^2 \alpha_t (1-\bar\alpha_t)}\big\|\epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon,\ t)\big\|^2\right] \tag{12}$$

이것은 정확히 **다중 노이즈 스케일에서의 denoising score matching 손실**! Score $\nabla_x \log q(x|x_0)$가 $-\epsilon/\sqrt{1-\bar\alpha_t}$이기 때문입니다.

샘플링 (식 11에 대응):
$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\!\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t, t)\right) + \sigma_t z, \quad z \sim \mathcal{N}(0, I)$$

이는 정확히 **annealed Langevin dynamics 한 스텝**.

**English**
This is the paper's central contribution. **Choice 1**: $\Sigma_\theta(x_t, t) = \sigma_t^2 I$ is fixed, not learned. Two options give similar results: $\sigma_t^2 = \beta_t$ (optimal for $x_0\sim\mathcal{N}(0,I)$) and $\sigma_t^2 = \tilde\beta_t$ (optimal for deterministic $x_0$). **Choice 2**: instead of predicting the mean $\tilde\mu_t$, parameterize the network to predict the **noise $\epsilon$** that was added in the forward process. After substituting Eq. (4) into Eq. (7), we get Eq. (11) — and Eq. (8) becomes Eq. (12), which is exactly **denoising score matching** at multiple noise scales (since $\nabla_x \log q(x|x_0) = -\epsilon/\sqrt{1-\bar\alpha_t}$). The sampling step matches one step of **annealed Langevin dynamics**.

#### 3.3 Data scaling, reverse process decoder, and $L_0$

**한국어**
이미지 픽셀 $\{0, 1, \ldots, 255\}$를 $[-1, 1]$로 스케일. 마지막 단계 $L_0 = -\log p_\theta(x_0|x_1)$는 가우시안에서 유도된 **이산 디코더** (식 13) — 각 픽셀의 로그가능도를 가우시안 PDF의 적분으로 평가. Bits/dim을 잘 정의된 lossless codelength로 만듭니다.

#### 3.4 Simplified training objective / 단순화된 학습 목표

**한국어**
정식 변분 하한은 식 (12)의 가중치를 갖지만, 저자들은 **가중치를 떨어뜨린** 단순한 변형을 제안 (식 14):

$$\boxed{L_\text{simple}(\theta) := \mathbb{E}_{t, x_0, \epsilon}\!\left[\big\|\epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\ t)\big\|^2\right]} \tag{14}$$

여기서 $t \sim \text{Uniform}\{1, \ldots, T\}$. **놀라운 발견**: 이 단순화된 손실이 정식 ELBO보다 **샘플 품질이 더 높다**. 이유는 가중치 $\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar\alpha_t)}$가 작은 $t$ (즉 노이즈가 작은 단계)에 큰 가중치를 줘서 매우 미세한 denoising을 학습하지만, $L_\text{simple}$은 큰 $t$의 어려운 denoising에 더 집중하기 때문.

**Algorithm 1 (Training)** (논문에서 그대로):
```
1: repeat
2:   x_0 ~ q(x_0)
3:   t ~ Uniform({1, ..., T})
4:   ε ~ N(0, I)
5:   Take gradient descent step on
        ∇_θ ‖ε - ε_θ(√ᾱ_t x_0 + √(1-ᾱ_t) ε, t)‖²
6: until converged
```

**Algorithm 2 (Sampling)** (논문에서 그대로):
```
1: x_T ~ N(0, I)
2: for t = T, ..., 1 do
3:   z ~ N(0, I) if t > 1, else z = 0
4:   x_{t-1} = (1/√α_t)(x_t - β_t/√(1-ᾱ_t) · ε_θ(x_t, t)) + σ_t z
5: end for
6: return x_0
```

학습이 **6줄**, 샘플링이 **5줄**. 이 11줄이 DDPM의 본질입니다.

**English**
The variational bound has the weight in Eq. (12), but the authors propose a **weight-dropped** variant (Eq. 14):
$$L_\text{simple} = \mathbb{E}_{t, x_0, \epsilon}\!\left[\|\epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)\|^2\right]$$
with $t\sim\text{Uniform}\{1,\ldots,T\}$. Counter-intuitively, dropping the weight **improves** sample quality — because the formal weight up-weights very-small-$t$ (tiny-noise) terms whose denoising is trivial; $L_\text{simple}$ focuses capacity on the harder large-$t$ denoising. The training algorithm is 6 lines, sampling is 5 lines.

### Part IV: Experiments (§4) / 실험

#### 4.0 Setup / 실험 설정

**한국어**
- $T = 1000$ steps
- 선형 $\beta$ 스케줄: $\beta_1 = 10^{-4} \to \beta_T = 0.02$. 결과적으로 $L_T \approx 10^{-5}$ bits/dim — $x_T$가 $\mathcal{N}(0, I)$에 매우 가까움.
- **U-Net** backbone (PixelCNN++ 영향): Wide ResNet 기반, 4개 feature map resolution (32×32 모델), 6개 (256×256 모델). 각 resolution마다 2개 residual block, 16×16 resolution에 self-attention.
- **시간 임베딩**: Transformer sinusoidal positional embedding을 각 residual block에 추가.
- Group normalization (weight normalization 대신).
- CIFAR10: 35.7M 매개변수. 256×256 (LSUN, CelebA-HQ): 114M 매개변수.
- 학습: Adam, lr $2\times10^{-4}$ (32×32) / $2\times10^{-5}$ (256×256), batch 128 (32×32) / 64 (256×256), EMA decay 0.9999.
- TPU v3-8: CIFAR10 21 steps/sec, 800k steps에 10.6 시간. 256² 모델: 2.2 steps/sec, 0.5–2.4M steps.

**English**
$T=1000$, linear $\beta$ schedule $10^{-4}\!\to\!0.02$, giving $L_T \approx 10^{-5}$ bits/dim. **U-Net** backbone (PixelCNN++ inspired): Wide-ResNet style with group normalization, 4 resolutions (32×32 model) or 6 (256×256), two residual blocks per resolution, self-attention at the 16×16 feature map. Time $t$ is injected via Transformer sinusoidal positional embedding added in each residual block. Parameters: 35.7M (CIFAR10), 114M (256²). Adam optimizer ($2\!\times\!10^{-4}$ or $2\!\times\!10^{-5}$), batch 128 / 64, EMA decay 0.9999. CIFAR10 trains in ~10 hours on TPU v3-8.

#### 4.1 Sample quality / 샘플 품질

**한국어**

**Table 1 — CIFAR10 Inception Score, FID, NLL**:

| Model | IS | FID | NLL Test (Train) |
|---|---|---|---|
| Conditional EBM | 8.30 | 37.9 | – |
| BigGAN | 9.22 | 14.73 | – |
| StyleGAN2+ADA (cond) | **10.06** | **2.67** | – |
| Diffusion (Sohl-Dickstein 2015 original) | – | – | ≤ 5.40 |
| NCSN | 8.87 | 25.32 | – |
| NCSNv2 | – | 31.75 | – |
| StyleGAN2+ADA | 9.74 | 3.26 | – |
| **Ours ($L$, fixed isotropic $\Sigma$)** | 7.67 | 13.51 | ≤ 3.70 (3.69) |
| **Ours ($L_\text{simple}$)** | **9.46** | **3.17** | ≤ 3.75 (3.72) |

**핵심 관찰**:
- $L_\text{simple}$로 학습한 모델: FID **3.17** — **모든 unconditional 모델 중 SOTA**, conditional StyleGAN2+ADA에만 뒤짐.
- 정식 ELBO ($L$) 학습 모델은 **NLL은 더 좋지만 (3.70 vs 3.75) 샘플 품질은 훨씬 나쁨 (FID 13.51 vs 3.17)**. 이것이 본 논문의 핵심 trade-off 발견.
- training/test FID: training set 기준 3.17, test set 기준 5.24 (둘 다 양호) — 과적합 없음.

**LSUN 결과**:
- LSUN bedroom 256² FID **4.90** (Figure 4)
- LSUN church 256² FID **7.89** (Figure 3)
- ProgressiveGAN과 비슷하거나 우수한 수준.

**English**

**Table 1**: $L_\text{simple}$ training gives **FID 3.17, IS 9.46** on unconditional CIFAR10 — SOTA among unconditional models, beaten only by conditional StyleGAN2+ADA. Training on the formal ELBO $L$ gives **better NLL but worse FID** (FID 13.51 vs 3.17) — the paper's key trade-off insight: the loss reweighting in $L_\text{simple}$ trades likelihood for perceptual quality. Test FID 5.24 (vs train FID 3.17) shows no overfitting. LSUN bedroom 256² FID **4.90**, LSUN church 256² FID **7.89** — comparable to or better than ProgressiveGAN.

#### 4.2 Reverse process parameterization and training objective ablation / 매개변수화/손실 ablation

**한국어**

**Table 2 — Ablation (CIFAR10, unconditional)**:

| Objective | IS | FID |
|---|---|---|
| **$\mu$-prediction (baseline)** | | |
| $L$, learned diagonal $\Sigma$ | 7.28 ± 0.10 | 23.69 |
| $L$, fixed isotropic $\Sigma$ | 8.06 ± 0.09 | 13.22 |
| $\|\tilde\mu - \mu_\theta\|^2$ | – (unstable) | – |
| **$\epsilon$-prediction (ours)** | | |
| $L$, learned diagonal $\Sigma$ | – (unstable) | – |
| $L$, fixed isotropic $\Sigma$ | 7.67 ± 0.13 | 13.51 |
| $\|\epsilon - \epsilon_\theta\|^2$ ($L_\text{simple}$) | **9.46 ± 0.11** | **3.17** |

**핵심 발견**:
1. **$\mu$-prediction은 정식 ELBO로만 잘 작동**: $\mu$를 예측하면 단순 MSE는 불안정.
2. **Learned $\Sigma$는 항상 불안정**: diagonal $\Sigma$를 학습하려 시도하면 scale-out-of-range로 학습 실패.
3. **$\epsilon$-prediction + $L_\text{simple}$이 압도적 우승**: $\epsilon$-prediction은 정식 $L$로도 $\mu$-prediction과 비슷하지만, $L_\text{simple}$로는 **훨씬 더 좋음** (FID 13.51 → 3.17).

#### 4.3 Progressive coding / 점진적 코딩

**한국어**
샘플 품질이 우수한 데도 NLL이 다른 likelihood 모델보다 떨어지는 이유를 분석. CIFAR10 모델의 lossless codelength 1.78 bits/dim이 실제 distortion인 0.95 RMSE (0–255 스케일)에 해당하며, 코드 길이의 **절반 이상이 imperceptible distortion**을 기술하는 데 사용된다고 보임.

**Progressive lossy compression** (Algorithm 3, 4)를 도입: rate-distortion 곡선을 그려보면 (Figure 5), 코드를 받기 시작하면 distortion이 급격히 감소하다가 끝부분에서 imperceptible bits가 대부분. 즉 **diffusion = excellent lossy compressor**.

**Progressive generation** (Figure 6): reverse process 중간에 $\hat x_0$를 예측 (식 15: $\hat x_0 = (x_t - \sqrt{1-\bar\alpha_t}\epsilon_\theta(x_t))/\sqrt{\bar\alpha_t}$)하면, 큰 $t$에서는 coarse 구조만, 작은 $t$에서 fine detail이 등장 — autoregressive decoding과 유사한 progressive 행동.

**Connection to autoregressive decoding** (식 16): variational bound는
$$L = D_\text{KL}(q(x_T)\|p(x_T)) + \mathbb{E}_q\!\left[\sum_{t\geq 1} D_\text{KL}(q(x_{t-1}|x_t)\|p_\theta(x_{t-1}|x_t))\right] + H(x_0)$$

특수한 경우 — diffusion length $T = D$ (데이터 차원)이고 $q(x_t|x_0)$가 처음 $t$ 좌표를 마스크 — 로 두면 **autoregressive 모델 학습과 같은 손실**이 됨. 즉 diffusion은 일반화된 bit ordering을 갖는 autoregressive model이며, Gaussian noise가 마스킹보다 더 자연스러운 inductive bias라는 가설.

**English**
Despite strong sample quality, NLL is below other likelihood models. The CIFAR10 model has 1.78 bits/dim codelength, which corresponds to RMSE 0.95 on the [0,255] scale — more than half the codelength describes imperceptible distortion. **Progressive lossy compression** (Algorithms 3 & 4): the rate-distortion curve (Fig. 5) plummets at low rates and flattens at the imperceptible end. **Progressive generation** (Fig. 6): predicting $\hat x_0$ from intermediate $x_t$ (Eq. 15) shows coarse structure first, fine details last — analogous to autoregressive decoding. Eq. (16) makes the connection precise: with $T=D$ (data dimension) and $q$ being a coordinate-masking process, the diffusion loss reduces to autoregressive training. Diffusion is "autoregressive with generalized bit ordering and Gaussian-noise inductive bias."

#### 4.4 Interpolation / 보간

**한국어**
두 이미지 $x_0, x_0'$를 latent space에서 보간 (Figure 8). Forward로 $x_t, x_t'$를 만들고, 선형 보간 $\bar x_t = (1-\lambda)x_t + \lambda x_t'$ 후 reverse로 디코딩. $t=500$에서 부드러운 attribute interpolation (포즈, 피부톤, 머리스타일, 표정, 배경) — 단 안경 같은 detail은 보간 안 됨. $t=1000$에서는 완전 새 샘플.

**English**
Linearly interpolate $x_t = (1-\lambda)x_t + \lambda x_t'$ at intermediate $t$ then decode via reverse process (Fig. 8). At $t=500$ smooth attribute interpolation (pose, skin tone, hair, expression, background) — but not fine details like eyewear. At $t=1000$ fully novel samples.

### Part V: Related Work (§5) / 관련 연구

**한국어**
- **Flows, VAEs**: diffusion은 $q$가 매개변수 없고 top-level latent $x_T$가 데이터와 mutual information이 0이라는 점에서 차이.
- **Score matching, NCSN (Song & Ermon 2019)**: 본 논문의 $\epsilon$-prediction이 다중 노이즈 score matching과 등가. NCSN과의 차이 (Appendix C):
  1. U-Net + self-attention (NCSN: RefineNet + dilated conv)
  2. Forward에서 $\sqrt{1-\beta_t}$로 신호를 줄여 분산 안정 (NCSN은 누락)
  3. Forward가 신호를 거의 완전히 파괴 ($L_T \approx 0$) → $x_T$ prior와 aggregate posterior가 일치
  4. Sampler 계수가 forward $\beta_t$로부터 엄밀히 유도 — variational inference로 sampler를 직접 학습 (NCSN은 hand-tuned)
- **Energy-based models**, **diffusion as variational inference for Langevin chains** (Sohl-Dickstein 2015), **infusion training**, **variational walkback**, **GSN**.

**English**
Distinguished from flows/VAEs (no parameters in $q$, near-zero mutual info between $x_T$ and data), and from NCSN (Song & Ermon 2019) by U-Net+attention architecture, $\sqrt{1-\beta_t}$ signal scaling, near-zero $L_T$, and rigorously derived sampler coefficients.

### Part VI: Conclusion (§6) / 결론

**한국어**
저자들은 확산 모델이 (1) 고품질 샘플을 낼 수 있고 (2) 변분 추론, denoising score matching, annealed Langevin dynamics, autoregressive decoding, progressive lossy compression이 모두 같은 수학적 객체임을 보였다고 정리. 이미지 외 도메인과 다른 생성 모델의 component로서의 활용을 미래 과제로 제시.

**English**
The authors argue diffusion models offer high sample quality plus deep connections to variational inference, denoising score matching, Langevin dynamics, autoregressive decoding, and progressive lossy compression — all the same mathematical object viewed differently. Future work: other modalities and use as components in larger systems.

### Part VII: Appendices / 부록 요약

**한국어**
- **Appendix A**: ELBO 식 (5), (16), (22)–(26) 자세한 유도. $\log q(x_0)$를 KL 항들로 분해하는 정확한 대수.
- **Appendix B**: 실험 세부. U-Net 구조, 35.7M / 114M 매개변수, group normalization, sinusoidal time embedding 위치, dropout (CIFAR10 0.1), random horizontal flip, EMA 0.9999.
- **Appendix C**: NCSN과의 비교 (위 §5 참조).
- **Appendix D**: 추가 샘플과 nearest-neighbor 분석 — 생성 샘플이 학습 데이터의 단순 복제가 아님 입증.

**English**
Appendix A: detailed ELBO derivations. Appendix B: experimental details — U-Net architecture, 35.7M / 114M parameters, group normalization, sinusoidal time embeddings injected per residual block, CIFAR10 dropout 0.1, random horizontal flips, EMA 0.9999. Appendix C: detailed NCSN comparison. Appendix D: additional samples and nearest-neighbor analysis showing samples are not memorized training images.

---

## 3. Key Takeaways / 핵심 시사점

1. **노이즈 예측 재매개변수화는 단순한 산수 변환이지만 학습 목표를 극적으로 단순화한다 / Noise-prediction is a trivial reparameterization that radically simplifies the loss**
   $\mu_\theta$ 예측 → $\epsilon_\theta$ 예측은 식 (4) 대입에 불과하지만, ELBO를 $\|\epsilon - \epsilon_\theta(x_t, t)\|^2$로 만든다. 이 한 번의 변환이 모든 후속 다음 5년 연구의 출발점이다. / Substituting Eq. (4) into Eq. (7) is "just algebra" — yet it turns the ELBO into $\|\epsilon - \epsilon_\theta(x_t, t)\|^2$, the launching pad for the next five years of generative modeling.

2. **단순화된 손실 $L_\text{simple}$이 정식 ELBO보다 더 좋은 샘플을 만든다 / The simplified loss $L_\text{simple}$ beats the formal ELBO on sample quality**
   FID 3.17 ($L_\text{simple}$) vs FID 13.51 ($L$, $\epsilon$-pred). 가중치 제거가 큰 $t$ 단계 (어려운 denoising)에 capacity를 더 할당하기 때문. likelihood와 perceptual quality의 명시적 trade-off 발견. / 4× lower FID with $L_\text{simple}$ — by dropping weights, capacity shifts to harder large-$t$ denoising. An explicit likelihood-vs-perception trade-off.

3. **확산 모델은 score-based 모델과 정확히 같다 / Diffusion = score-based generative models**
   $\epsilon$-prediction은 다중 노이즈 스케일에서의 denoising score matching과 등가. Algorithm 2의 reverse step은 annealed Langevin dynamics 한 스텝. 두 분리된 연구 갈래가 같은 수학적 객체임을 입증. / The $\epsilon$-prediction objective is exactly multi-scale denoising score matching, and the reverse step is one step of annealed Langevin dynamics — unifying two previously separate threads.

4. **Closed-form forward sampling이 학습을 한 줄로 만든다 / Closed-form forward sampling makes training a one-liner**
   $q(x_t|x_0) = \mathcal{N}(\sqrt{\bar\alpha_t}x_0, (1-\bar\alpha_t)I)$ — Markov chain을 시뮬레이션할 필요 없이 임의의 $t$에서 한 번에 $x_t$를 만든다. 이것이 학습 효율의 비밀. / The closed-form Eq. (4) lets training pick any $t$ in one shot — no Markov-chain simulation. This is the efficiency secret.

5. **$\beta$ 스케줄은 매우 작아야 한다 / The $\beta$ schedule must be tiny**
   $\beta_1 = 10^{-4}$에서 $\beta_T = 0.02$, $T = 1000$. $L_T \approx 10^{-5}$ bits/dim — $x_T$가 $\mathcal{N}(0,I)$와 거의 일치. 작은 $\beta$는 reverse가 가우시안임을 보장 (Feller, 1949)하고 forward가 가역적임을 의미. / $T=1000$ with $\beta_1\!=\!10^{-4} \to \beta_T\!=\!0.02$. The smallness of $\beta$ guarantees Gaussian reverse (Feller 1949) and ensures $x_T$ matches the prior to $10^{-5}$ bits/dim.

6. **U-Net + sinusoidal time embedding이 표준 백본이 된다 / U-Net + sinusoidal $t$-embedding becomes standard**
   PixelCNN++ inspired Wide-ResNet U-Net, 16×16 self-attention, group normalization, residual block마다 시간 임베딩 injection. 이 조합은 거의 변하지 않은 채로 Stable Diffusion까지 이어진다. / This combination — Wide-ResNet U-Net, attention at 16×16, group norm, time embedding per residual block — survives nearly unchanged through Stable Diffusion.

7. **Diffusion은 progressive lossy decompression이다 / Diffusion is progressive lossy decompression**
   Codelength 1.78 bits/dim에서 RMSE 0.95 — 코드의 절반 이상이 imperceptible detail. Coarse-to-fine 생성 (Figure 6)은 일반화된 bit-ordering을 가진 autoregressive 모델로 해석 가능. 가우시안 노이즈가 마스킹보다 자연스러운 inductive bias. / 1.78 bits/dim codelength corresponds to RMSE 0.95 — over half describes imperceptible detail. Coarse-to-fine generation (Fig. 6) is autoregressive with generalized bit ordering, and Gaussian noise is a more natural inductive bias than masking.

8. **GAN 시대의 종식을 알린 첫 번째 신호 / The first signal of the GAN era's end**
   2014–2019 GAN 황금기 이후, DDPM이 unconditional CIFAR10에서 SOTA를 차지하며 likelihood 기반 모델의 재기를 알림. 2021 "Diffusion beats GANs" → 2022 Stable Diffusion → 현재 DALL·E 3, Imagen, Sora의 직접 선조. / After 2014–2019 GAN dominance, DDPM took unconditional CIFAR10 SOTA and signaled the resurgence of likelihood-based modeling. Direct ancestor of DALL·E 2, Stable Diffusion, Imagen, and Sora.

---

## 4. Mathematical Summary / 수학적 요약

### Core derivation chain / 핵심 유도 체인

**Step 1**: Forward process (Eq. 2)
$$q(x_t | x_{t-1}) = \mathcal{N}(x_t;\ \sqrt{1-\beta_t}\, x_{t-1},\ \beta_t I)$$

**Step 2**: Closed-form $q(x_t|x_0)$ (Eq. 4)
$$q(x_t|x_0) = \mathcal{N}(x_t;\ \sqrt{\bar\alpha_t}\, x_0,\ (1-\bar\alpha_t) I)$$
$$\quad\alpha_t = 1-\beta_t,\quad \bar\alpha_t = \prod_{s=1}^t \alpha_s$$

**Step 3**: Forward posterior (Eq. 6, 7)
$$q(x_{t-1}|x_t, x_0) = \mathcal{N}(x_{t-1};\ \tilde\mu_t(x_t, x_0),\ \tilde\beta_t I)$$
$$\tilde\mu_t = \frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t} x_0 + \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t} x_t$$
$$\tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t} \beta_t$$

**Step 4**: Variational bound (Eq. 5)
$$L = \mathbb{E}_q\!\left[L_T + \sum_{t>1} L_{t-1} + L_0\right]$$
$$L_T = D_\text{KL}(q(x_T|x_0)\,\|\,p(x_T))$$
$$L_{t-1} = D_\text{KL}(q(x_{t-1}|x_t, x_0)\,\|\,p_\theta(x_{t-1}|x_t))$$
$$L_0 = -\log p_\theta(x_0|x_1)$$

**Step 5**: $\epsilon$-prediction reparameterization. From Eq. (4), $x_0 = (x_t - \sqrt{1-\bar\alpha_t}\epsilon)/\sqrt{\bar\alpha_t}$. Substitute into $\tilde\mu_t$:
$$\tilde\mu_t(x_t, x_0(x_t,\epsilon)) = \frac{1}{\sqrt{\alpha_t}}\!\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon\right)$$

Parameterize $\mu_\theta$ to match (Eq. 11):
$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\!\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t, t)\right)$$

**Step 6**: Loss in noise form (Eq. 12)
$$L_{t-1} = \mathbb{E}_{x_0, \epsilon}\!\left[\frac{\beta_t^2}{2\sigma_t^2 \alpha_t (1-\bar\alpha_t)}\big\|\epsilon - \epsilon_\theta(x_t, t)\big\|^2\right]$$
where $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$.

**Step 7**: Simplified loss (Eq. 14)
$$\boxed{L_\text{simple} = \mathbb{E}_{t, x_0, \epsilon}\!\left[\big\|\epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\ t)\big\|^2\right]}$$

**Step 8**: Sampling (Algorithm 2)
$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\!\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t, t)\right) + \sigma_t\, z$$
$$z \sim \mathcal{N}(0, I) \text{ if } t > 1, \quad z = 0 \text{ if } t = 1$$
with $\sigma_t^2 = \beta_t$ (or $\tilde\beta_t$).

### Worked numerical example: one denoising step / 한 단계 잡음 제거 워크스루

**한국어**
설정: $T = 1000$, 선형 $\beta$, $\beta_t = 10^{-4} + (0.02 - 10^{-4})\frac{t-1}{T-1}$. 

$t = 500$에서:
- $\beta_{500} \approx 10^{-4} + 0.0199 \cdot \frac{499}{999} \approx 0.0100$
- $\alpha_{500} = 1 - \beta_{500} \approx 0.9900$
- $\bar\alpha_{500} = \prod_{s=1}^{500} \alpha_s$를 계산 → 약 $\bar\alpha_{500} \approx 0.0817$ (수치적으로)
- 따라서 $\sqrt{\bar\alpha_{500}} \approx 0.286$, $\sqrt{1-\bar\alpha_{500}} \approx 0.958$

**Forward step**: $x_0 = 1.0$ (1D 단순화)이고 $\epsilon = 0.5$이면
$$x_{500} = 0.286 \cdot 1.0 + 0.958 \cdot 0.5 = 0.286 + 0.479 = 0.765$$
원래 신호의 약 29%만 남고 노이즈가 우세.

**Reverse step**: 신경망이 $\epsilon_\theta(x_{500}, 500) = 0.49$를 예측 (실제 $0.5$에 가까움)이라 가정. 그러면
$$\mu_\theta = \frac{1}{\sqrt{0.99}}\!\left(0.765 - \frac{0.01}{0.958} \cdot 0.49\right) = \frac{1}{0.995}(0.765 - 0.00511) = \frac{0.760}{0.995} = 0.7637$$

$\sigma_t^2 = \beta_t = 0.01$, 즉 $\sigma_t = 0.1$. $z \sim \mathcal{N}(0, 1)$이 $z = -0.3$이라 하면
$$x_{499} = 0.7637 + 0.1 \cdot (-0.3) = 0.7337$$

**해석**: 신경망이 노이즈를 거의 정확히 예측했다면 $x_{499}$의 $x_0$ 비율 추정치 $(x_{499} - \sqrt{1-\bar\alpha_{499}}\epsilon_\theta)/\sqrt{\bar\alpha_{499}}$가 1.0에 더 가까워짐. 1000번 반복하면 노이즈에서 데이터로의 부드러운 경로가 만들어집니다.

**English**
Setup: $T=1000$, linear $\beta_t$. At $t=500$: $\beta_{500}\!\approx\!0.01$, $\alpha_{500}\!\approx\!0.99$, $\bar\alpha_{500}\!\approx\!0.0817$, so $\sqrt{\bar\alpha_{500}}\!\approx\!0.286$ and $\sqrt{1-\bar\alpha_{500}}\!\approx\!0.958$. **Forward**: with $x_0=1.0$, $\epsilon=0.5$, $x_{500} = 0.286(1.0) + 0.958(0.5) = 0.765$ — only ~29% of the original signal remains. **Reverse**: if the network predicts $\epsilon_\theta=0.49$, then $\mu_\theta = (1/\sqrt{0.99})(0.765 - (0.01/0.958)(0.49)) \approx 0.7637$. With $\sigma_t = 0.1$ and $z = -0.3$, $x_{499} = 0.7337$. Repeat 1000 times to walk from $\mathcal{N}(0,1)$ noise to a data sample.

### Connection to score / score와의 연결

Tweedie's formula links score to noise-prediction:
$$\nabla_x \log q(x_t|x_0) = -\frac{x_t - \sqrt{\bar\alpha_t}x_0}{1-\bar\alpha_t} = -\frac{\epsilon}{\sqrt{1-\bar\alpha_t}}$$

So
$$s_\theta(x_t, t) = -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1-\bar\alpha_t}}$$

학습 손실 $\|\epsilon - \epsilon_\theta\|^2$은 정확히 denoising score matching $\|s - s_\theta\|^2 \cdot (1-\bar\alpha_t)$ — DDPM과 score-based 모델은 같은 객체.

### Parameter count / 매개변수 수

- CIFAR10 32×32 모델: **35.7M** parameters
- LSUN/CelebA-HQ 256×256 모델: **114M** parameters
- 큰 LSUN bedroom 모델: **256M** parameters

### Training/sampling cost

- CIFAR10: TPU v3-8 기준 21 steps/sec @ batch 128. 800k steps → 10.6 시간. 256개 샘플링: 17초
- 256² LSUN/CelebA-HQ: 2.2 steps/sec @ batch 64. 128 샘플링: 300초 (5분) — **GAN 대비 매우 느림**

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1907 ─ Langevin: stochastic differential equations
        │
1949 ─ Feller: small-step Markov chain reverse is Gaussian
        │
2005 ─ Hyvärinen: Score Matching
        │
2011 ─ Vincent: Denoising Score Matching ↔ DAE equivalence
        │
2013 ─ Kingma & Welling: VAE (paper #15)
        │       ─ ELBO + reparameterization trick
        │
2014 ─ Goodfellow et al.: GAN (paper #16)
        │
2015 ─ ★ Sohl-Dickstein et al.: Diffusion probabilistic models (founder)
        │       ─ non-equilibrium thermodynamics inspired
        │       ─ established forward/reverse Markov chain framework
        │       ─ but sample quality lagged GANs for 5 years
        │
2018 ─ Karras et al.: ProgressiveGAN (256² benchmark)
        │
2019 ─ Song & Ermon: NCSN (Noise Conditional Score Network)
        │       ─ score-based generative models with annealed Langevin
        │
2019 ─ Song & Ermon: NCSNv2
        │
2020 ─ ★★★ Ho, Jain, Abbeel: DDPM (this paper) ★★★
        │       ─ ε-prediction reparameterization
        │       ─ L_simple weighted MSE loss
        │       ─ unification: VI + score matching + Langevin
        │       ─ FID 3.17 on CIFAR10 (SOTA unconditional)
        │
2021 ─ Nichol & Dhariwal: Improved DDPM
        │       ─ learned variance, cosine β schedule
        │
2021 ─ Song, Sohl-Dickstein et al.: Score SDE
        │       ─ continuous-time unification (SDE/ODE)
        │
2021 ─ Dhariwal & Nichol: Diffusion beats GANs on Image Synthesis
        │       ─ classifier guidance, ImageNet 256 SOTA
        │
2022 ─ Ho & Salimans: Classifier-Free Guidance
        │
2022 ─ Rombach et al.: Latent Diffusion (Stable Diffusion)
        │       ─ run diffusion in VAE latent space → text-to-image at scale
        │
2022 ─ Saharia et al.: Imagen (Google text-to-image)
        │
2022 ─ Ramesh et al.: DALL·E 2
        │
2023 ─ Peebles & Xie: DiT (Diffusion Transformers)
        │
2024 ─ OpenAI Sora, Google Veo: video diffusion at scale
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **#15 Kingma & Welling (2014) — VAE** | DDPM은 VAE를 1000-layer hierarchical로 확장, 다만 encoder $q$가 매개변수 없음. ELBO 분해는 같은 변분 추론 framework. / DDPM is essentially a 1000-layer hierarchical VAE with a parameter-free encoder $q$. ELBO decomposition is the same VI framework. | **High** — 직접적 framework 계승 / Direct framework inheritance |
| **#16 Goodfellow et al. (2014) — GAN** | DDPM은 GAN의 unconditional CIFAR10 SOTA를 빼앗음 (FID 3.17 vs StyleGAN 3.26). 학습 안정성과 mode coverage에서 우월. / DDPM dethroned GANs on unconditional CIFAR10 (FID 3.17 vs StyleGAN 3.26) with better stability and mode coverage. | **High** — 직접 경쟁자, 패러다임 전환 / Direct competitor; paradigm shift |
| **#19 Ioffe & Szegedy (2015) — Batch Norm** (Wu & He 2018 GroupNorm) | DDPM은 group normalization 사용 — batch norm 대체. / Uses group normalization in U-Net (replacing weight/batch norm). | **Medium** — 학습 안정화 / Training stabilization |
| **#20 He et al. (2015) — ResNet** | U-Net 백본이 Wide-ResNet 기반. residual connection 핵심. / U-Net backbone is Wide-ResNet based; residual blocks essential. | **High** — 아키텍처 구성 요소 / Architectural building block |
| **#22 Ronneberger et al. (2015) — U-Net** | Reverse 함수 $\epsilon_\theta$의 백본. encoder-decoder + skip connection. / U-Net is the backbone for $\epsilon_\theta$ — encoder-decoder with skips. | **High** — 직접 백본 / Direct backbone |
| **#25 Vaswani et al. (2017) — Transformer** | Sinusoidal positional embedding으로 시간 $t$ 인코딩. self-attention layer를 16×16에서 사용. 후속 DiT (2023)에서는 U-Net 자체를 Transformer로 대체. / Sinusoidal embedding for $t$; self-attention at 16×16. Successor DiT (2023) replaces U-Net entirely with Transformer. | **High** — 구성 요소 + 후속 변형 / Building block + successor variant |
| **Sohl-Dickstein et al. (2015) — Original Diffusion** | DDPM의 직접적 framework 출처. 본 논문이 5년 잠자던 framework를 부활시킴. / The original framework — DDPM revives it after a 5-year dormancy. | **High** — 직계 선조 / Direct ancestor |
| **Song & Ermon (2019) — NCSN** | $\epsilon$-prediction이 NCSN의 denoising score matching과 등가. DDPM의 sampler가 annealed Langevin과 등가. / DDPM's $\epsilon$-prediction = NCSN's denoising score matching; DDPM's sampler = annealed Langevin. | **High** — 등가성 입증 / Equivalence established |
| **Vincent (2011) — Denoising Score Matching** | $\epsilon$-prediction 손실의 이론적 기초. / Theoretical foundation for the $\epsilon$-prediction loss. | **High** — 핵심 이론 / Core theory |
| **#18 Kingma & Ba (2014) — Adam** | 모든 실험에서 Adam 사용 (lr $2\times10^{-4}$). / Adam used in all experiments. | **Low** — 도구 / Tool |

---

## 7. References / 참고문헌

### Primary paper / 본 논문
- Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. *Advances in Neural Information Processing Systems (NeurIPS)*, 33. arXiv:2006.11239.
- Code: https://github.com/hojonathanho/diffusion

### Direct predecessors / 직접 선조
- Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., & Ganguli, S. (2015). Deep unsupervised learning using nonequilibrium thermodynamics. *ICML 2015*. arXiv:1503.03585.
- Song, Y., & Ermon, S. (2019). Generative modeling by estimating gradients of the data distribution. *NeurIPS 2019* (NCSN). arXiv:1907.05600.
- Song, Y., & Ermon, S. (2020). Improved techniques for training score-based generative models. *NeurIPS 2020* (NCSNv2). arXiv:2006.09011.
- Vincent, P. (2011). A connection between score matching and denoising autoencoders. *Neural Computation*, 23(7), 1661–1674.
- Hyvärinen, A. (2005). Estimation of non-normalized statistical models by score matching. *JMLR*, 6, 695–709.

### Architecture / 아키텍처
- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. *MICCAI 2015*.
- Salimans, T., Karpathy, A., Chen, X., & Kingma, D. P. (2017). PixelCNN++. *ICLR 2017*.
- Wu, Y., & He, K. (2018). Group normalization. *ECCV 2018*.
- Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS 2017*.
- Zagoruyko, S., & Komodakis, N. (2016). Wide residual networks. *BMVC 2016*.

### Variational inference & generative models / 변분 추론 및 생성 모델
- Kingma, D. P., & Welling, M. (2014). Auto-encoding variational Bayes. *ICLR 2014* (VAE).
- Goodfellow, I., et al. (2014). Generative adversarial nets. *NeurIPS 2014*.
- Karras, T., Aila, T., Laine, S., & Lehtinen, J. (2018). Progressive growing of GANs. *ICLR 2018*.
- Karras, T., et al. (2020). Training generative adversarial networks with limited data. *NeurIPS 2020* (StyleGAN2+ADA).

### Successors / 후속 연구
- Nichol, A., & Dhariwal, P. (2021). Improved denoising diffusion probabilistic models. *ICML 2021*.
- Song, Y., et al. (2021). Score-based generative modeling through stochastic differential equations. *ICLR 2021* (Score SDE).
- Dhariwal, P., & Nichol, A. (2021). Diffusion models beat GANs on image synthesis. *NeurIPS 2021*.
- Ho, J., & Salimans, T. (2022). Classifier-free diffusion guidance. *NeurIPS 2021 Workshop*. arXiv:2207.12598.
- Rombach, R., et al. (2022). High-resolution image synthesis with latent diffusion models. *CVPR 2022* (Stable Diffusion).
- Saharia, C., et al. (2022). Photorealistic text-to-image diffusion models with deep language understanding. *NeurIPS 2022* (Imagen).
- Peebles, W., & Xie, S. (2023). Scalable diffusion models with transformers. *ICCV 2023* (DiT).

### Foundations / 기초
- Adam: Kingma, D. P., & Ba, J. (2015). *ICLR 2015*.
- Feller, W. (1949). On the theory of stochastic processes. *Annals of Mathematical Statistics*.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning. *CVPR 2016*.
- Welling, M., & Teh, Y. W. (2011). Bayesian learning via stochastic gradient Langevin dynamics. *ICML 2011*.

### Datasets / 데이터셋
- CIFAR10: Krizhevsky, A., & Hinton, G. (2009).
- CelebA-HQ: Karras et al. (2018).
- LSUN: Yu et al. (2015).
