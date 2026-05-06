---
title: "Diffusion Posterior Sampling for General Noisy Inverse Problems"
authors: Hyungjin Chung, Jeongsol Kim, Michael T. McCann, Marc L. Klasky, Jong Chul Ye
year: 2023
venue: "ICLR 2023"
arxiv: "2209.14687"
topic: Low-SNR Imaging / Diffusion Models for Inverse Problems
tags: [diffusion-model, posterior-sampling, inverse-problems, score-based, tweedie, laplace-approximation, dps, nonlinear-inverse-problems, phase-retrieval, deblurring]
status: completed
date_started: 2026-05-06
date_completed: 2026-05-06
---

# 28. Diffusion Posterior Sampling for General Noisy Inverse Problems / 일반 잡음 역문제를 위한 확산 사후 표집

---

## 1. Core Contribution / 핵심 기여

### 한국어
이 논문은 **사전 학습된 확산 모델(diffusion model)을 일반(비선형) 잡음 역문제의 사후 표집기로 사용하는 단일 알고리즘 DPS(Diffusion Posterior Sampling)** 를 제안한다. 핵심 아이디어는 다음과 같다.

(i) 역문제 $\boldsymbol y = \mathcal A(\boldsymbol x_0) + \boldsymbol n$에서 사후 score $\nabla_{\boldsymbol x_t} \log p_t(\boldsymbol x_t \mid \boldsymbol y) = \nabla_{\boldsymbol x_t} \log p_t(\boldsymbol x_t) + \nabla_{\boldsymbol x_t} \log p(\boldsymbol y \mid \boldsymbol x_t)$로 분해되지만, 시간-의존 likelihood $p(\boldsymbol y \mid \boldsymbol x_t) = \int p(\boldsymbol y \mid \boldsymbol x_0) p(\boldsymbol x_0 \mid \boldsymbol x_t) \, d\boldsymbol x_0$는 해석적으로 다루기 어렵다.

(ii) **핵심 근사(Theorem 1)**: $p(\boldsymbol y \mid \boldsymbol x_t) \approx p(\boldsymbol y \mid \hat{\boldsymbol x}_0(\boldsymbol x_t))$로 근사. 여기서 $\hat{\boldsymbol x}_0(\boldsymbol x_t) = \mathbb{E}[\boldsymbol x_0 \mid \boldsymbol x_t]$는 **Tweedie 공식**으로 score 모델에서 닫힌 형태로 얻는 사후 평균이다. 이 근사의 Jensen-gap 오차 한계가 증명되었으며, $\boldsymbol x_t$가 $\boldsymbol x_0$의 노이즈 버전이므로 신뢰 영역 좁을 때 정확.

(iii) **알고리즘**: 표준 ancestral DDPM 샘플러에 한 단계 추가 — $\boldsymbol x_{t-1} \leftarrow \boldsymbol x'_{t-1} - \zeta_t \nabla_{\boldsymbol x_t} \|\boldsymbol y - \mathcal A(\hat{\boldsymbol x}_0(\boldsymbol x_t))\|^2$ (Gaussian) 또는 Poisson 음수로그가능도. autograd가 score U-Net을 통과하도록 backprop 한 번 추가.

(iv) **결과**: SR×4, Gaussian/motion deblurring, **nonlinear deblurring**, **Fourier phase retrieval** 등 7개 과제에서 FFHQ/ImageNet에 대해 DDRM, MCG, plug-and-play(PnPADMM, ADMM-TV, Score-SDE) 등 모든 기존 방법을 perceptual(LPIPS, FID) 지표에서 능가. 특히 nonlinear 문제에서 SVD 기반 방법(DDRM)은 적용 자체가 불가능했다.

### English
This paper introduces **Diffusion Posterior Sampling (DPS)**, a single algorithm that turns a pre-trained diffusion model into a posterior sampler for **general (possibly nonlinear) noisy inverse problems** $\boldsymbol y = \mathcal A(\boldsymbol x_0) + \boldsymbol n$.

(i) The posterior score decomposes as $\nabla_{\boldsymbol x_t} \log p_t(\boldsymbol x_t \mid \boldsymbol y) = \underbrace{\nabla_{\boldsymbol x_t} \log p_t(\boldsymbol x_t)}_{\text{prior score (known)}} + \underbrace{\nabla_{\boldsymbol x_t} \log p(\boldsymbol y \mid \boldsymbol x_t)}_{\text{likelihood (intractable)}}$, and the time-dependent likelihood is the key obstacle since $\boldsymbol y$ depends on $\boldsymbol x_0$ rather than $\boldsymbol x_t$.

(ii) **Key approximation (Theorem 1)**: $p(\boldsymbol y \mid \boldsymbol x_t) \simeq p(\boldsymbol y \mid \hat{\boldsymbol x}_0(\boldsymbol x_t))$, where $\hat{\boldsymbol x}_0$ is the **Tweedie posterior mean** computable in closed form from the learned score. The Jensen-gap error is bounded, vanishing as the noise level $\sigma_t \to 0$.

(iii) **Algorithm**: One backprop step is added to the ancestral DDPM sampler: at each step, take a gradient through the score-network $\hat{\boldsymbol x}_0$-prediction with respect to $\boldsymbol x_t$ to push the iterate toward measurement consistency.

(iv) **Results**: On FFHQ and ImageNet, DPS surpasses DDRM, MCG, plug-and-play methods on Gaussian/motion deblur, super-resolution, inpainting, **and nonlinear tasks** (Fourier phase retrieval, nonlinear deblur) where SVD-based methods fail entirely. Best LPIPS and FID across the board, with measurement-consistent reconstructions.

---

## 2. Reading Notes / 읽기 노트

### Part I: §1 Introduction & §2 Background / 도입과 배경

#### 한국어
- **확산 모델 사전(prior)**: Score-based generative model은 데이터 분포 $p(\boldsymbol x)$의 score $\nabla_{\boldsymbol x} \log p(\boldsymbol x)$를 학습. SDE 형식(Song et al. 2021b): $d\boldsymbol x = \boldsymbol f(\boldsymbol x, t) dt + g(t) d\boldsymbol w$. VP-SDE(DDPM)에서 $\boldsymbol f = -\frac{\beta(t)}{2}\boldsymbol x$, $g = \sqrt{\beta(t)}$.
- **역방향 SDE**: $d\boldsymbol x = [\boldsymbol f(\boldsymbol x, t) - g(t)^2 \nabla_{\boldsymbol x} \log p_t(\boldsymbol x)] dt + g(t) d\bar{\boldsymbol w}$. score $\boldsymbol s_\theta(\boldsymbol x_t, t) \approx \nabla \log p_t(\boldsymbol x_t)$를 학습한 후 역방향 SDE를 적분하면 사전 표집 가능.
- **역문제 사후 표집의 어려움**: 측정 $\boldsymbol y = \mathcal A(\boldsymbol x_0) + \boldsymbol n$이 주어지면 사후 $p(\boldsymbol x_0 \mid \boldsymbol y)$에서 표집해야 함. 베이즈 규칙으로 $\nabla_{\boldsymbol x_t} \log p_t(\boldsymbol x_t \mid \boldsymbol y) = \nabla_{\boldsymbol x_t} \log p_t(\boldsymbol x_t) + \nabla_{\boldsymbol x_t} \log p_t(\boldsymbol y \mid \boldsymbol x_t)$. 둘째 항이 핵심 난제.
- **선행 접근의 한계**:
  - **Projection**: Song et al. 2021b, ILVR: $\boldsymbol x_t$를 측정 부공간으로 직접 사영. 잡음이 클 때 잡음을 그대로 부공간 안으로 밀어넣어 증폭.
  - **Spectral-domain (DDRM, $\Pi$GDM)**: SVD로 forward operator 분해 후 spectral 도메인에서 noise injection. SVD 비용이 크고, **분리가능한 가우시안 디블러** 등 좁은 클래스에만 적용 가능. 비선형 문제 불가능.
  - **MCG (Chung et al. 2022b)**: gradient + projection. projection이 manifold를 벗어나 누적 오차 → 발산 위험.

#### English
- **Diffusion priors** learn the score $\nabla_{\boldsymbol x} \log p(\boldsymbol x)$ of the data distribution; the reverse SDE
  $d\boldsymbol x = [\boldsymbol f - g^2 \nabla \log p_t(\boldsymbol x)] dt + g d\bar{\boldsymbol w}$
  generates samples from the prior once the score $\boldsymbol s_\theta$ is trained.
- **Posterior-sampling difficulty**: Bayes rule yields $\nabla_{\boldsymbol x_t} \log p_t(\boldsymbol x_t \mid \boldsymbol y) = \nabla_{\boldsymbol x_t} \log p_t(\boldsymbol x_t) + \nabla_{\boldsymbol x_t} \log p_t(\boldsymbol y \mid \boldsymbol x_t)$. The first term is the learned prior; the second (time-dependent likelihood) is intractable because $\boldsymbol y$ depends on $\boldsymbol x_0$, not directly on $\boldsymbol x_t$.
- **Limitations of prior work**: projection methods amplify noise in noisy settings; spectral-domain (DDRM, $\Pi$GDM) needs SVD and only applies to narrow linear classes; MCG combines gradient + projection but the projection can throw the iterate off-manifold.

---

### Part II: §3.1 Posterior Sampling by Diffusion / 확산을 통한 사후 표집

#### 한국어 — Theorem 1 (Laplace approximation of the time-dependent likelihood)

**핵심 식**:
$$
p(\boldsymbol y \mid \boldsymbol x_t) = \int p(\boldsymbol y \mid \boldsymbol x_0) p(\boldsymbol x_0 \mid \boldsymbol x_t) d\boldsymbol x_0 \quad (8)
$$
일반적으로 분석 불가. **DPS의 핵심 근사**:
$$
p(\boldsymbol y \mid \boldsymbol x_t) \simeq p(\boldsymbol y \mid \hat{\boldsymbol x}_0(\boldsymbol x_t)), \quad \hat{\boldsymbol x}_0(\boldsymbol x_t) := \mathbb E[\boldsymbol x_0 \mid \boldsymbol x_t] \quad (9)
$$
즉 적분에서 대표값 하나를 뽑아 사용. **Tweedie 공식** (Efron 2011, Robbins 1956)에 의해 VP-SDE의 가우시안 전이 $p(\boldsymbol x_t \mid \boldsymbol x_0) = \mathcal N(\sqrt{\bar\alpha_t}\boldsymbol x_0, (1-\bar\alpha_t) \boldsymbol I)$에 대해
$$
\hat{\boldsymbol x}_0(\boldsymbol x_t) = \frac{1}{\sqrt{\bar\alpha_t}}\left(\boldsymbol x_t + (1-\bar\alpha_t)\boldsymbol s_\theta(\boldsymbol x_t, t)\right) \quad (10)
$$
이 닫힌 형태로 얻어짐. 따라서 측정 모델(가우시안 잡음)에 대해
$$
\nabla_{\boldsymbol x_t} \log p(\boldsymbol y \mid \boldsymbol x_t) \simeq -\frac{1}{\sigma^2} \nabla_{\boldsymbol x_t} \|\boldsymbol y - \mathcal A(\hat{\boldsymbol x}_0(\boldsymbol x_t))\|_2^2 \quad (11)
$$
계산은 PyTorch autograd로 score U-Net을 backprop 한 번 통과시켜 구현. **Theorem 1**은 이 근사의 Jensen-gap 오차가 $\boldsymbol x_0 \mapsto \log p(\boldsymbol y \mid \boldsymbol x_0)$의 곡률 × $p(\boldsymbol x_0 \mid \boldsymbol x_t)$의 분산으로 한계지어짐을 증명.

#### English — Theorem 1
The intractable likelihood (8) is replaced by point estimate (9) at the Tweedie posterior mean (10). For Gaussian noise the gradient becomes (11), implemented with one backprop through the score network. Theorem 1 bounds the Jensen-gap by the curvature of $\boldsymbol x_0 \mapsto \log p(\boldsymbol y \mid \boldsymbol x_0)$ times the variance of $p(\boldsymbol x_0 \mid \boldsymbol x_t)$ — small at low noise levels, justifying the approximation.

---

### Part III: §3.2 Algorithm / 알고리즘

#### 한국어
**Gaussian DPS (Algorithm 1)**:

```
Input: T, y, {sigma_t}_{t=1}^T, {tilde sigma_t}, {zeta_t}
x_T ~ N(0, I)
for t = T-1 to 0:
    s_hat = s_theta(x_t, t)                                    # score network
    x_hat0 = (1/sqrt(alpha_bar_t)) * (x_t + (1-alpha_bar_t) * s_hat)  # Tweedie
    z ~ N(0, I)
    x'_{t-1} = (sqrt(alpha_t)*(1-alpha_bar_{t-1}))/(1-alpha_bar_t) * x_t
             + (sqrt(alpha_bar_{t-1})*beta_t)/(1-alpha_bar_t) * x_hat0
             + tilde_sigma_t * z                              # standard DDPM step
    x_{t-1} = x'_{t-1} - zeta_t * grad_{x_t} ||y - A(x_hat0)||_2^2
return x_0
```

- **Step size $\zeta_t$**: 논문은 $\zeta_t = \zeta'/\|\boldsymbol y - \mathcal A(\hat{\boldsymbol x}_0)\|$ (norm-normalised, Eq. 17) 사용. 잔차 norm으로 정규화하면 측정값 스케일 무관하게 안정.
- **Poisson 잡음 (Algorithm 2)**: gaussian likelihood를 $\sum_i (\Lambda \mathcal A(\hat{\boldsymbol x}_0))_i - y_i \log(\Lambda \mathcal A(\hat{\boldsymbol x}_0))_i$로 교체. 같은 backprop 골격.
- **MCG와의 차이**: MCG는 gradient 직후 projection $\boldsymbol x_t \leftarrow P(\boldsymbol x_t)$로 강제 일관성 부과 → manifold 이탈 누적. DPS는 **projection 단계가 없음**. 만약 manifold가 정확하다면 gradient 자체가 manifold tangent에 접한다(보조정리 1).

#### English
- The DPS sampler adds **one line** to ancestral DDPM: a gradient-descent step on the squared measurement residual evaluated at the Tweedie mean.
- The step size $\zeta_t$ is normalised by the residual norm: $\zeta_t = \zeta'/\|\boldsymbol y - \mathcal A(\hat{\boldsymbol x}_0)\|$ (Eq. 17), giving robustness across measurement scales.
- For Poisson noise (Algorithm 2), the Gaussian residual is replaced by the Poisson NLL, otherwise the algorithm is identical.
- Crucially DPS has **no projection step** unlike MCG: removing it eliminates manifold-leaving accumulation errors.

---

### Part IV: §3.3 Geometric Interpretation / 기하학적 해석

#### 한국어
- **Manifold hypothesis**: 데이터 매니폴드 $\mathcal M_0 \subset \mathbb R^d$가 차원 $d_0 \ll d$. Noise 확산 시각 $t$에서 데이터는 $\mathcal M_t$에 (대략) 머물며 noise level 따라 부풀어남.
- **Lemma 1 (Tangent gradient)**: gradient $\nabla_{\boldsymbol x_t} \log p(\boldsymbol y \mid \hat{\boldsymbol x}_0(\boldsymbol x_t))$는 chain rule 상 $\frac{\partial \hat{\boldsymbol x}_0}{\partial \boldsymbol x_t}$를 인수로 가져 $\mathcal M_t$의 접공간에 거의 평행. Projection 없이도 매니폴드 위에서 움직임.
- **MCG의 발산 메커니즘 (Fig. 3)**: projection은 noise 부공간을 그대로 측정 평면에 박아 매니폴드 이탈 → 후속 score step에서 큰 잔차 → 결국 발산.

#### English
- Under the manifold hypothesis the noisy data at time $t$ lies on $\mathcal M_t$. Lemma 1 shows the DPS update direction is (approximately) tangent to $\mathcal M_t$ because the chain-rule Jacobian $\partial \hat{\boldsymbol x}_0/\partial \boldsymbol x_t$ projects gradients onto data-tangent directions.
- MCG's projection forces hard consistency at the cost of leaving $\mathcal M_t$ — Fig. 3 visualises the resulting drift and divergence.

---

### Part V: §4 Experiments / 실험

#### 한국어
- **Datasets**: FFHQ 256×256 (1k validation), ImageNet 256×256 (1k validation). Pretrained score models from Choi et al. 2021 / Dhariwal & Nichol 2021.
- **Tasks (linear)**: SR (×4 bicubic), Gaussian deblur ($\sigma=3.0$ kernel), Motion deblur (intensity 0.5), random box inpainting (128×128 mask), random pixel inpainting (90% missing).
- **Tasks (nonlinear)**: Fourier **phase retrieval** ($\boldsymbol y = |\mathcal F \mathcal P \boldsymbol x|$, $\mathcal P$ 4× zero-pad), nonlinear deblurring (Tran et al. 2021의 강한 비선형 PSF).
- **측정 잡음**: $\boldsymbol n \sim \mathcal N(0, 0.05^2 \boldsymbol I)$ (linear), Poisson $\lambda = 1.0$ (별도 실험).
- **Baselines**: DDRM, MCG, PnP-ADMM (BM3D prior), TV-ADMM, Score-SDE.

**Quantitative (FFHQ, Table 1, sampled rows)**:

| Method        | SR×4 LPIPS↓ | Gauss Deblur LPIPS↓ | Box Inpaint LPIPS↓ | Phase Retrieval LPIPS↓ |
|---------------|-------------|---------------------|--------------------|------------------------|
| DPS (ours)    | **0.214**   | **0.214**           | **0.111**          | **0.399**              |
| DDRM          | 0.239       | 0.298               | —                  | n/a (nonlinear)        |
| MCG           | 0.520       | 0.340               | 0.309              | 0.692                  |
| PnP-ADMM      | 0.405       | 0.692               | 0.151              | n/a                    |
| Score-SDE     | 0.534       | 0.436               | 0.235              | n/a                    |

(숫자는 논문 Table 1의 부분 발췌. 모든 베이스라인 대비 DPS가 LPIPS, FID 모두 우위.)

- **Phase retrieval**: 4번 다른 시드로 표집해 가장 일관된 결과 선택 (논문 Fig. 8). DPS는 SOTA prDeep과 비교 가능 또는 우월.
- **NFEs**: DPS는 1000 NFEs (DDPM 단계 수와 동일). DDRM은 20-100. 속도-품질 trade-off 존재.

#### English
- **Datasets**: FFHQ-256 and ImageNet-256, 1k validation images each.
- **Tasks**: SR×4, Gaussian/motion deblurring, box/random inpainting (linear); Fourier phase retrieval and nonlinear deblurring (nonlinear). Noise $\sigma=0.05$ (Gauss) or Poisson $\lambda=1.0$.
- **Baselines**: DDRM, MCG, PnP-ADMM (BM3D), TV-ADMM, Score-SDE.
- **Headline numbers** (FFHQ, LPIPS↓): DPS = 0.214 / 0.214 / 0.111 / 0.399 (SR / GaussDeblur / BoxInpaint / PhaseRetrieval) — uniformly best across all rows of Table 1, both LPIPS and FID. Spectral methods are NA on nonlinear tasks.
- DPS uses 1000 NFEs (full DDPM trajectory) versus DDRM's 20-100 — quality is bought with compute.

---

### Part VI: §4.4 Ablations & §5 Discussion / 절제 실험과 토론

#### 한국어
- **Step size**: norm-normalised $\zeta_t$ (Eq. 17)이 fixed $\zeta_t$보다 안정. 너무 크면 prior의 분포에서 벗어나 비현실적, 너무 작으면 측정 일관성 부족.
- **Projection vs. no-projection**: MCG의 projection을 제거하면 거의 DPS와 동일 (소거 실험). Projection이 손해라는 직접 증거.
- **Limits**: 1000 NFEs (느림). 비등방 noise는 likelihood 모델 변경 필요. Severe nonlinearity에서 Tweedie 근사 불충분할 수 있음.
- **Open**: 더 빠른 sampler(DPM-Solver+) 결합, blind inverse problem (forward operator도 모름).

#### English
- Norm-normalised step size is essential; fixed step diverges or under-fits.
- Removing MCG's projection effectively reproduces DPS — projection itself is detrimental in noisy settings.
- Limits: 1000 NFE cost, dependence on Gaussian/Poisson noise models, potential breakdown at extreme nonlinearity.
- Future: faster samplers (DPM-Solver), blind problems, integration with conditional generative priors.

---

## 3. Key Takeaways / 핵심 시사점

### 한국어
1. **Tweedie 공식이 핵심 도구다** — 학습된 score $\boldsymbol s_\theta(\boldsymbol x_t, t)$로부터 $\hat{\boldsymbol x}_0 = \mathbb E[\boldsymbol x_0 \mid \boldsymbol x_t]$를 닫힌 형태로 즉시 얻을 수 있어 시간-의존 likelihood를 시간-독립 likelihood로 환원할 수 있다.
2. **하나의 알고리즘이 선형/비선형/잡음 통합 처리** — forward operator $\mathcal A$의 미분만 가능하면 linear, nonlinear, MRI, CT, phase retrieval 모두 동일 코드로 처리.
3. **Projection은 잡음 환경의 적이다** — 깨끗한 측정에서는 강제 일관성 도움. 잡음이 들어가는 순간 noise 부공간을 매니폴드에 박아 발산. DPS의 "projection 제거"가 핵심 설계.
4. **Gradient의 자동적 매니폴드 적합성** — Tweedie 사후 평균을 통한 backprop은 chain rule 상 데이터 매니폴드 접공간 방향으로 자연스럽게 정렬 (Lemma 1). 추가 manifold 정규화 불필요.
5. **사전 학습된 비조건부 모델 + 추론 시 가이드** — score 네트워크는 측정 모델을 모른 채 학습. 추론 시점에 측정 정보 주입 — generative prior와 inverse-problem solver의 분리가 가능.
6. **Step-size 정규화의 중요성** — 잔차 norm으로 정규화한 $\zeta_t$는 측정 스케일/SNR 변화에 불변. 실용적 hyperparameter 견고성의 핵심.
7. **Speed-quality trade-off는 미해결 과제** — 1000 NFEs는 실시간/대량 처리에 부담. 후속 연구(DPM-Solver-DPS, $\Pi$GDM)가 이를 공략.

### English
1. **Tweedie's formula is the workhorse** — turns time-dependent likelihood into time-independent by revealing $\hat{\boldsymbol x}_0$ in closed form from the learned score.
2. **One algorithm, all forward models** — differentiability of $\mathcal A$ is the only requirement; linear, nonlinear, Gaussian/Poisson all handled.
3. **Projection harms in noisy settings** — DPS's central design choice is dropping the projection step that earlier methods used.
4. **Manifold-tangency is automatic** — backprop through $\hat{\boldsymbol x}_0$ aligns updates with $\mathcal M_t$ tangent directions (Lemma 1); no extra projection needed.
5. **Decoupling prior from forward model** — score net is unconditional and reusable; measurement information enters only at inference, opening a clean modular design.
6. **Norm-normalised step size** is critical for hyperparameter robustness across SNR and operator scales.
7. **Speed remains an open problem** — 1000 NFEs is heavy; subsequent works (DPM-Solver+DPS, $\Pi$GDM) push toward fewer steps.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Forward and reverse SDEs / 전방·역방향 SDE
**VP-SDE** (DDPM continuous limit):
$$
d\boldsymbol x = -\tfrac{1}{2}\beta(t)\boldsymbol x \, dt + \sqrt{\beta(t)}\, d\boldsymbol w
$$
**Reverse SDE**:
$$
d\boldsymbol x = \left[-\tfrac{1}{2}\beta(t)\boldsymbol x - \beta(t)\nabla_{\boldsymbol x}\log p_t(\boldsymbol x)\right] dt + \sqrt{\beta(t)} \, d\bar{\boldsymbol w}
$$

### 4.2 Conditional reverse SDE / 조건부 역방향 SDE
$$
d\boldsymbol x = \left[-\tfrac{1}{2}\beta(t)\boldsymbol x - \beta(t)\big(\nabla\log p_t(\boldsymbol x) + \nabla\log p_t(\boldsymbol y \mid \boldsymbol x)\big)\right] dt + \sqrt{\beta(t)}\, d\bar{\boldsymbol w}
$$

### 4.3 Tweedie's formula / 트위디 공식
For VP-SDE with $\boldsymbol x_t = \sqrt{\bar\alpha_t}\boldsymbol x_0 + \sqrt{1-\bar\alpha_t}\boldsymbol \epsilon$:
$$
\hat{\boldsymbol x}_0(\boldsymbol x_t) := \mathbb E[\boldsymbol x_0 \mid \boldsymbol x_t] = \frac{1}{\sqrt{\bar\alpha_t}}\Big(\boldsymbol x_t + (1-\bar\alpha_t)\boldsymbol s_\theta(\boldsymbol x_t, t)\Big)
$$

### 4.4 DPS likelihood gradient / DPS 가능도 기울기
**Gaussian** $\boldsymbol y = \mathcal A(\boldsymbol x_0) + \mathcal N(0, \sigma^2 \boldsymbol I)$:
$$
\nabla_{\boldsymbol x_t} \log p(\boldsymbol y \mid \boldsymbol x_t) \simeq -\frac{1}{\sigma^2}\nabla_{\boldsymbol x_t}\big\|\boldsymbol y - \mathcal A(\hat{\boldsymbol x}_0(\boldsymbol x_t))\big\|_2^2
$$
**Poisson** $\boldsymbol y \sim \text{Poisson}(\Lambda \mathcal A(\boldsymbol x_0))$:
$$
\nabla_{\boldsymbol x_t} \log p(\boldsymbol y \mid \boldsymbol x_t) \simeq \nabla_{\boldsymbol x_t}\sum_i\Big(y_i \log[\Lambda\mathcal A(\hat{\boldsymbol x}_0)]_i - [\Lambda\mathcal A(\hat{\boldsymbol x}_0)]_i\Big)
$$

### 4.5 DPS update / DPS 갱신식
DDPM 평균값 $\boldsymbol\mu_t(\boldsymbol x_t, \hat{\boldsymbol x}_0)$를 갖는 표준 Markov chain:
$$
\boldsymbol x'_{t-1} \sim \mathcal N\big(\boldsymbol\mu_t(\boldsymbol x_t, \hat{\boldsymbol x}_0), \tilde\sigma_t^2 \boldsymbol I\big)
$$
$$
\boldsymbol x_{t-1} = \boldsymbol x'_{t-1} - \zeta_t \nabla_{\boldsymbol x_t}\big\|\boldsymbol y - \mathcal A(\hat{\boldsymbol x}_0(\boldsymbol x_t))\big\|_2^2
$$

### 4.6 Worked example / 워크드 예제: 1D 다운샘플링

목표: $x_0 \in \mathbb R^4$에서 $\boldsymbol y = \boldsymbol A x_0 + \boldsymbol n$, $\boldsymbol A = \begin{pmatrix} 0.5 & 0.5 & 0 & 0 \\ 0 & 0 & 0.5 & 0.5 \end{pmatrix}$ (2×4 평균 풀링), $\sigma = 0.1$.

진실 $\boldsymbol x_0 = (1, 1, -1, -1)^\top$, $\boldsymbol y = (1.0, -1.0)^\top + \boldsymbol n \approx (0.97, -1.05)^\top$.

확산 시각 $t$에서 $\boldsymbol x_t = \sqrt{\bar\alpha_t}\boldsymbol x_0 + \sqrt{1-\bar\alpha_t}\boldsymbol\epsilon$, $\bar\alpha_t = 0.5$. 가정 $\boldsymbol x_t = (0.71, 0.50, -0.85, -0.55)^\top$, score $\boldsymbol s_\theta(\boldsymbol x_t, t) \approx -\boldsymbol x_t$ (가우시안 prior 가정 → 정확).

Tweedie 추정:
$\hat{\boldsymbol x}_0 = \frac{1}{\sqrt{0.5}}\big(\boldsymbol x_t + 0.5 \cdot (-\boldsymbol x_t)\big) = \frac{0.5}{\sqrt{0.5}}\boldsymbol x_t = \sqrt{0.5} \cdot \boldsymbol x_t \approx (0.50, 0.35, -0.60, -0.39)$.

잔차: $\boldsymbol y - \boldsymbol A\hat{\boldsymbol x}_0 \approx (0.97, -1.05) - (0.43, -0.49) = (0.54, -0.56)$. $\|\cdot\|_2^2 \approx 0.605$.

기울기 (autograd로): $\nabla_{\boldsymbol x_t} \|\boldsymbol y - \boldsymbol A\hat{\boldsymbol x}_0\|^2 = 2\cdot \frac{\partial \hat{\boldsymbol x}_0}{\partial \boldsymbol x_t}^\top \boldsymbol A^\top (\boldsymbol A\hat{\boldsymbol x}_0 - \boldsymbol y)$. $\partial\hat{\boldsymbol x}_0/\partial\boldsymbol x_t = \sqrt{0.5}\boldsymbol I$ (가우시안 prior에서). 결과는 $\sqrt{0.5}$ 배율을 곱한 일반 least-squares gradient.

이는 standard PnP gradient와 동일한 방향. DPS의 차이: score가 비-가우시안 prior일 때 Jacobian $\partial\hat{\boldsymbol x}_0/\partial\boldsymbol x_t$가 데이터 매니폴드의 접 사영을 자동으로 적용 → manifold-aware gradient.

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1956 -- Robbins introduces empirical Bayes; Tweedie's formula
        for posterior mean of Gaussian-mixture observation.
   |
1980s -- Plug-and-play and ADMM are still decades away;
        regularised inverse problems use TV/Tikhonov priors.
   |
2011 -- Efron revives Tweedie's formula for modern empirical Bayes.
   |
2013 -- Plug-and-Play prior (Venkatakrishnan et al.):
        replace prox with off-the-shelf denoiser.
   |
2015 -- Sohl-Dickstein et al. introduce diffusion as a generative model.
2019 -- Song & Ermon: NCSN — score matching for image generation.
   |
2020 -- DDPM (Ho et al.) — quality breakthrough on CIFAR/CelebA.
2021 -- Song et al. unify diffusion as SDE; conditional generation
        via projection (ILVR, Score-SDE inverse).
   |
2022 -- Spectral methods (DDRM, $\Pi$GDM) handle linear inverse
        problems via SVD.
   |
2022 -- MCG (Chung et al.) — gradient + projection; works but
        prone to manifold leakage in noisy settings.
   |
*** 2023 — DPS (this paper, ICLR) ***
        Drop projection. Use Tweedie + autograd. Handles
        nonlinear (phase retrieval, nonlinear deblur) with one
        unified algorithm.
   |
2023 -- Latent-DPS, $\Pi$GDM, DiffPIR (CVPR-NTIRE), Pseudoinverse-Guided.
2024 -- DPS-style guidance becomes standard for science (MRI,
        CT, dark-matter, low-SNR microscopy).
```

```
1956 -- Robbins / Tweedie's formula
2011 -- Efron's modern empirical Bayes revival
2020 -- DDPM (Ho et al.)
2022 -- DDRM (linear-only, SVD-based)
2023 -- DPS: Tweedie + autograd, all-purpose noisy inverse solver
2023 -- DiffPIR: PnP framework with diffusion (parallel work)
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| 관련 논문 / Related paper | 관계 / Relation |
|----------------------|------------------------------------------------------|
| **Ho et al. 2020 (DDPM)** — 기반 prior. DPS는 DDPM/score-SDE를 미수정 상태로 사용. / Foundational prior; DPS uses pre-trained DDPM/score-SDE without modification. |
| **Song et al. 2021b (Score-SDE)** — SDE 형식·역방향 SDE 도출의 기반. ILVR/projection 기반 inverse는 DPS의 비교군. / Provides SDE/reverse-SDE machinery; DPS's baselines. |
| **Kawar et al. 2022 (DDRM)** — SVD 기반 spectral 접근. linear 한정. DPS는 nonlinear까지 확장. / Spectral SVD method limited to linear; DPS extends to nonlinear. |
| **Chung et al. 2022b (MCG)** — gradient + projection. DPS는 projection 제거가 핵심 설계 변경. / Direct precursor; DPS removes projection for stability. |
| **Robbins 1956 / Efron 2011 (Tweedie)** — 사후 평균 폐쇄형 공식. DPS의 수학적 핵심. / Closed-form posterior-mean identity at the heart of DPS. |
| **Zhu et al. 2023 (DiffPIR, paper #30)** — 동일 동기, PnP-ADMM/HQS 골격. DPS는 ancestral DDPM 단계에서 직접 backprop. / Concurrent; DiffPIR uses PnP/HQS structure where DPS uses native diffusion sampler. |
| **Daras et al. 2023 (Ambient, paper #29)** — score 학습 단계의 corruption 처리. DPS는 추론 단계의 corruption 역전. / Complementary: Ambient handles training-time corruption, DPS handles inference-time corruption. |
| **Donoho-Johnstone 1994 (paper #01)** — wavelet shrinkage as the analytic counterpart of denoising-as-MAP-prior. score가 학습된 nonlinear 일반화. / Wavelet shrinkage is the linear/analytic ancestor of learned score-based denoising. |

---

## 7. References / 참고문헌

- Chung, H., Kim, J., McCann, M. T., Klasky, M. L., & Ye, J. C. (2023). "Diffusion Posterior Sampling for General Noisy Inverse Problems." *ICLR 2023*. arXiv:2209.14687.
- Ho, J., Jain, A., & Abbeel, P. (2020). "Denoising Diffusion Probabilistic Models." *NeurIPS 2020*.
- Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021). "Score-Based Generative Modeling through Stochastic Differential Equations." *ICLR 2021*.
- Kawar, B., Elad, M., Ermon, S., & Song, J. (2022). "Denoising Diffusion Restoration Models." *NeurIPS 2022*.
- Chung, H., Sim, B., Ryu, D., & Ye, J. C. (2022). "Improving Diffusion Models for Inverse Problems Using Manifold Constraints (MCG)." *NeurIPS 2022*.
- Efron, B. (2011). "Tweedie's Formula and Selection Bias." *JASA*, 106(496), 1602–1614.
- Robbins, H. (1956). "An Empirical Bayes Approach to Statistics." *Berkeley Symp.*
- Venkatakrishnan, S. V., Bouman, C. A., & Wohlberg, B. (2013). "Plug-and-Play Priors for Model Based Reconstruction." *GlobalSIP*.
- Code: https://github.com/DPS2022/diffusion-posterior-sampling

---

## Appendix A. Implementation Notes / 부록 A. 구현 노트

### 한국어
- **U-Net 통과 backprop**: PyTorch에서 `x_t.requires_grad_(True)`로 설정 후 score 호출 → Tweedie → measurement loss → `loss.backward()`. backprop 한 번에 score U-Net forward + backward (메모리 ~2× forward only).
- **VRAM 비용**: 256×256 이미지에 기본 score U-Net (~100M params) 사용 시 단일 step에 ~3-4 GB. 1000 step × batch 1 = ~수 분/이미지 (A100).
- **Numerical stability**: $\sqrt{\bar\alpha_t}$가 매우 작은 t (가까운 t=T)에서는 Tweedie가 부풀어남 ($1/\sqrt{\bar\alpha_t}$ 발산). 적절한 step warm-up 필요.
- **Gradient checkpoint**: 메모리 부족 시 PyTorch checkpoint를 사용해 score U-Net 중간 activation을 재계산 → 시간 ~1.3×, 메모리 절반.

### English
- Backprop through the score U-Net is implemented by setting `x_t.requires_grad_(True)`, calling the score net, computing the Tweedie estimate, evaluating the measurement loss, and calling `.backward()`. One backward pass costs roughly 2× the forward pass.
- For 256×256 images and a 100M-parameter U-Net, a single DPS step needs ~3-4 GB; 1000 steps × batch 1 takes a few minutes per image on an A100.
- Numerical care: $1/\sqrt{\bar\alpha_t}$ explodes at high $t$, so the Tweedie estimate is unreliable in the first few steps — common practice is to skip or warm-up the gradient term there.
- Gradient checkpointing trades ~30% extra time for ~50% less memory if needed.
