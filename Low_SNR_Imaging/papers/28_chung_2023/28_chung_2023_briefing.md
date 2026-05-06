---
title: "Pre-Reading Briefing: Diffusion Posterior Sampling for General Noisy Inverse Problems (DPS)"
paper_id: "28_chung_2023"
topic: Low_SNR_Imaging
date: 2026-05-06
type: briefing
---

# Diffusion Posterior Sampling for General Noisy Inverse Problems (DPS): Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: H. Chung, J. Kim, M. T. McCann, M. L. Klasky, J. C. Ye, *ICLR* 2023, arXiv:2209.14687
**Author(s)**: Hyungjin Chung, Jeongsol Kim, Michael T. McCann, Marc L. Klasky, Jong Chul Ye
**Year**: 2023

---

## 1. 핵심 기여 / Core Contribution

### 한국어
이 논문은 **사전 학습된 확산 모델(diffusion model)을 일반(비선형) 잡음 역문제의 사후 표집기로 사용하는 단일 알고리즘 DPS (Diffusion Posterior Sampling)** 를 제안한다. 핵심 통찰은 시간-의존 likelihood $p(\boldsymbol y \mid \boldsymbol x_t) = \int p(\boldsymbol y \mid \boldsymbol x_0) p(\boldsymbol x_0 \mid \boldsymbol x_t) d\boldsymbol x_0$ 가 분석 불가능한 적분이지만, **Tweedie 사후 평균** $\hat{\boldsymbol x}_0(\boldsymbol x_t) = \mathbb E[\boldsymbol x_0 \mid \boldsymbol x_t]$ 한 점으로 근사 (Theorem 1, Jensen-gap bounded)하면 autograd로 계산 가능한 닫힌 형태를 얻는다는 것이다. 표준 ancestral DDPM sampler에 backprop 한 번을 추가하여 measurement-consistency gradient $\nabla_{\boldsymbol x_t} \|\boldsymbol y - \mathcal A(\hat{\boldsymbol x}_0)\|^2$을 더하는 것이 알고리즘 전부. **DDRM의 SVD 기반 spectral 한계를 극복** — Fourier phase retrieval, 비선형 deblur, Poisson noise 등 일반 forward operator + non-Gaussian noise에 직접 적용 가능. FFHQ/ImageNet에서 LPIPS, FID 모두 SOTA.

### English
This paper introduces **Diffusion Posterior Sampling (DPS)**, a single algorithm that turns a pre-trained diffusion model into a posterior sampler for **general (possibly nonlinear) noisy inverse problems** $\boldsymbol y = \mathcal A(\boldsymbol x_0) + \boldsymbol n$. The key insight: the time-dependent likelihood $p(\boldsymbol y \mid \boldsymbol x_t)$ is an intractable integral, but replacing $\boldsymbol x_0$ by the **Tweedie posterior mean** $\hat{\boldsymbol x}_0(\boldsymbol x_t) = \mathbb E[\boldsymbol x_0 \mid \boldsymbol x_t]$ (closed form from the learned score) gives a tractable approximation whose Jensen-gap is bounded (Theorem 1). The algorithm is just one extra backprop step added to ancestral DDPM: a gradient on the measurement residual $\|\boldsymbol y - \mathcal A(\hat{\boldsymbol x}_0)\|^2$ taken through the score U-Net. This **lifts DDRM's SVD-based linearity restriction**: Fourier phase retrieval, nonlinear deblur, Poisson noise are all handled with the same code. SOTA LPIPS and FID on FFHQ/ImageNet 256.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

**한국어**: 2022년 시점에서 사전학습 diffusion을 inverse problem에 사용하는 방법은 셋이었다: (i) **Projection** (Song 2021, ILVR) — 측정 부분공간으로 직접 사영. 잡음을 그대로 부공간 안으로 밀어 넣어 noisy 환경에서 발산. (ii) **Spectral-domain** (DDRM, $\Pi$GDM) — SVD 분해. 분리가능 가우시안 디블러 등 좁은 *선형* 클래스만. (iii) **MCG** (Chung 2022b) — gradient + projection. projection이 매니폴드 이탈 누적. 모두 한계가 있었다. 한편 phase retrieval, nonlinear deblur, MRI motion correction 등 *비선형* 문제는 spectral 방법으로 다룰 수조차 없었다. DPS는 (a) Tweedie 공식의 활용, (b) projection 제거, (c) 매니폴드-접 자동성 (Lemma 1)을 통해 이 모든 한계를 한번에 해결한다.

**English**: As of 2022, three approaches existed for using pre-trained diffusion in inverse problems: (i) **projection** (Song 2021, ILVR) — direct subspace projection, which amplifies noise; (ii) **spectral-domain** (DDRM, $\Pi$GDM) — SVD-based, restricted to narrow linear classes; (iii) **MCG** (Chung 2022b) — gradient + projection, with manifold-leaving error accumulation. Nonlinear problems (phase retrieval, nonlinear deblur, MRI motion) were essentially out of reach. DPS resolves all of these via (a) Tweedie's formula, (b) removal of the projection step, and (c) automatic manifold-tangency (Lemma 1).

### 타임라인 / Timeline

```
1956 — Robbins / Tweedie's formula (empirical Bayes)
2011 — Efron's modern Tweedie revival
2013 — Plug-and-Play (Venkatakrishnan)
2020 — DDPM (Ho et al.)
2021 — Score-based SDEs; projection-based ILVR
2021 — Kadkhodaie-Simoncelli (paper #25, denoiser-prior linear)
2022 — DDRM (paper #26, SVD spectral, linear-only)
2022 — MCG (Chung+, gradient + projection, manifold-leaky)
2023 ★★ DPS (THIS PAPER) — Tweedie + autograd, all-purpose
2023 — DiffPIR (paper #30, concurrent PnP-HQS)
```

---

## 3. 필요한 배경 지식 / Prerequisites

**한국어**:
- **Score-based SDE / VP-SDE**: $d\boldsymbol x = -\frac{\beta(t)}{2}\boldsymbol x dt + \sqrt{\beta(t)} d\boldsymbol w$, 역방향 SDE.
- **DDPM forward / reverse**: $q(\boldsymbol x_t \mid \boldsymbol x_0) = \mathcal N(\sqrt{\bar\alpha_t}\boldsymbol x_0, (1-\bar\alpha_t) \boldsymbol I)$, ancestral sampler.
- **Tweedie 공식**: $\hat{\boldsymbol x}_0 = \mathbb E[\boldsymbol x_0 \mid \boldsymbol x_t]$의 닫힌 형태.
- **Manifold hypothesis**: 데이터가 저차원 매니폴드 $\mathcal M_0 \subset \mathbb R^d$ 근처에 집중.
- **Bayes 분해**: $\nabla_{\boldsymbol x_t} \log p_t(\boldsymbol x_t \mid \boldsymbol y) = \nabla \log p_t(\boldsymbol x_t) + \nabla \log p_t(\boldsymbol y \mid \boldsymbol x_t)$.
- **PyTorch autograd**: U-Net을 통과하는 backprop.
- **Forward operator**: linear (SR, deblur, inpaint) + nonlinear (phase retrieval $|\mathcal F \mathcal P \boldsymbol x|$, nonlinear deblur).
- **Jensen's inequality / gap**: 적분 vs. point estimate의 차이 분석.

**English**:
- **Score-based SDE / VP-SDE**: $d\boldsymbol x = -\frac{\beta(t)}{2}\boldsymbol x dt + \sqrt{\beta(t)} d\boldsymbol w$ and the reverse SDE.
- **DDPM forward / reverse**: $q(\boldsymbol x_t \mid \boldsymbol x_0) = \mathcal N(\sqrt{\bar\alpha_t}\boldsymbol x_0, (1-\bar\alpha_t)\boldsymbol I)$, ancestral sampler.
- **Tweedie's formula**: closed form for $\hat{\boldsymbol x}_0 = \mathbb E[\boldsymbol x_0 \mid \boldsymbol x_t]$.
- **Manifold hypothesis**: data concentrates on a low-dimensional manifold $\mathcal M_0$.
- **Bayes split**: $\nabla_{\boldsymbol x_t} \log p_t(\boldsymbol x_t \mid \boldsymbol y) = \nabla \log p_t(\boldsymbol x_t) + \nabla \log p_t(\boldsymbol y \mid \boldsymbol x_t)$.
- **PyTorch autograd**: backprop through a U-Net.
- **Forward operators**: linear (SR, deblur, inpaint) and nonlinear (Fourier phase retrieval $|\mathcal F \mathcal P \boldsymbol x|$, nonlinear deblur).
- **Jensen's inequality / gap**: difference between integral and point-estimate.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| Posterior sampling | $p(\boldsymbol x_0 \mid \boldsymbol y)$에서의 표집. inverse problem의 베이지안 해 / Sampling from the posterior under measurement; Bayesian solution to inverse problems. |
| Time-dependent likelihood | $p(\boldsymbol y \mid \boldsymbol x_t) = \int p(\boldsymbol y \mid \boldsymbol x_0)p(\boldsymbol x_0\mid \boldsymbol x_t)d\boldsymbol x_0$. 분석 불가 / The intractable integral linking measurement to the diffusion variable at time $t$. |
| Tweedie posterior mean $\hat{\boldsymbol x}_0$ | $\mathbb E[\boldsymbol x_0 \mid \boldsymbol x_t]$, score로부터 closed form / Closed-form posterior mean from the learned score; Eq. 10. |
| Laplace / point-estimate approximation | $p(\boldsymbol y \mid \boldsymbol x_t) \approx p(\boldsymbol y \mid \hat{\boldsymbol x}_0)$. Theorem 1로 Jensen-gap 한계 / Approximate the integral by evaluating at the posterior mean; Theorem 1 bounds the error. |
| Forward operator $\mathcal A$ | 측정 모델 $\boldsymbol y = \mathcal A(\boldsymbol x_0) + \boldsymbol n$. 선형 또는 비선형 / Measurement map; can be nonlinear. |
| Phase retrieval | $\boldsymbol y = |\mathcal F \mathcal P \boldsymbol x|$. magnitude only, phase 잃음. 본질적으로 비선형 / Magnitude-only Fourier measurement; classical nonlinear inverse problem. |
| MCG | Manifold-Constrained Gradient (Chung 2022b). gradient + projection / Predecessor combining gradient with hard projection; suffers from manifold-leaving accumulation. |
| Projection step | 측정 부분공간으로의 hard 투영. DPS는 *제거* / Hard projection onto the measurement subspace; DPS deliberately removes it. |
| Lemma 1 (tangent gradient) | $\nabla_{\boldsymbol x_t} \log p(\boldsymbol y \mid \hat{\boldsymbol x}_0)$가 매니폴드 접 방향에 정렬 / The DPS update is automatically tangent to the diffusion manifold $\mathcal M_t$. |
| Step size $\zeta_t$ | norm-normalised: $\zeta_t = \zeta'/\|\boldsymbol y - \mathcal A(\hat{\boldsymbol x}_0)\|$ / Normalised by residual norm for SNR-invariant robustness. |
| Poisson DPS | Gaussian residual을 Poisson NLL로 교체 (Algorithm 2) / Same algorithm with Poisson negative log-likelihood replacing the Gaussian residual. |
| NFE = 1000 | DPS의 비용. DDRM 20-100, DiffPIR ≤100 / 1000 score-network forward passes per image; DPS's main cost. |

---

## 5. 수식 미리보기 / Equations Preview

**핵심 1: Bayes 분해 / Conditional score split**

$$
\nabla_{\boldsymbol x_t}\log p_t(\boldsymbol x_t \mid \boldsymbol y) = \nabla_{\boldsymbol x_t}\log p_t(\boldsymbol x_t) + \nabla_{\boldsymbol x_t}\log p_t(\boldsymbol y \mid \boldsymbol x_t)
$$

**한국어**: 사후 score = prior score (학습된 $\boldsymbol s_\theta$) + likelihood score (시간-의존, intractable). 둘째 항이 핵심 난제.

**English**: Posterior score = (known) prior score + (intractable) time-dependent likelihood score. The second term is the obstacle DPS solves.

**핵심 2: Tweedie 공식 / Tweedie's formula (Eq. 10)**

$$
\hat{\boldsymbol x}_0(\boldsymbol x_t) = \frac{1}{\sqrt{\bar\alpha_t}}\Big(\boldsymbol x_t + (1-\bar\alpha_t)\,\boldsymbol s_\theta(\boldsymbol x_t, t)\Big)
$$

**한국어**: 학습된 score $\boldsymbol s_\theta$만 알면 사후 평균이 닫힌 형태. 이것이 시간-의존 likelihood를 시간-독립 likelihood로 환원하는 다리.

**English**: Closed-form posterior mean from the learned score — the bridge that reduces a time-dependent integral to a time-independent function evaluation.

**핵심 3: DPS likelihood gradient (Gaussian, Eq. 11)**

$$
\nabla_{\boldsymbol x_t}\log p(\boldsymbol y \mid \boldsymbol x_t) \simeq -\frac{1}{\sigma^2}\nabla_{\boldsymbol x_t}\bigl\|\boldsymbol y - \mathcal A(\hat{\boldsymbol x}_0(\boldsymbol x_t))\bigr\|_2^2
$$

**한국어**: PyTorch autograd로 score U-Net을 통과하는 backprop 한 번이면 된다. forward operator $\mathcal A$의 미분 가능성만 요구.

**English**: One backprop through the score U-Net suffices. Only requires $\mathcal A$ to be differentiable — applies to phase retrieval, nonlinear deblur, etc.

**핵심 4: DPS update / DPS 갱신식**

$$
\boldsymbol x_{t-1} = \boldsymbol x'_{t-1} - \zeta_t\,\nabla_{\boldsymbol x_t}\bigl\|\boldsymbol y - \mathcal A(\hat{\boldsymbol x}_0(\boldsymbol x_t))\bigr\|_2^2
$$

**한국어**: 표준 ancestral DDPM step $\boldsymbol x'_{t-1}$에 measurement-consistency 보정을 더함. $\zeta_t = \zeta'/\|\boldsymbol y - \mathcal A(\hat{\boldsymbol x}_0)\|$로 norm-normalised.

**English**: Standard ancestral DDPM step plus a measurement-consistency correction with a residual-norm-normalised step size. *No projection step.*

---

## 6. 읽기 가이드 / Reading Guide

**한국어**:
- **§2 (Background)**: SDE 형식과 reverse SDE 도출. 익숙하지 않으면 Song 2021b를 먼저 훑어볼 것.
- **§3.1 (Theorem 1)**: 핵심 근사. Jensen-gap의 곡률 × 분산 한계. Appendix의 증명 디테일은 최초 독해에서는 skip 가능.
- **§3.2 (Algorithm 1)**: 코드 5줄. PyTorch에서 `x_t.requires_grad_(True)` → score 호출 → Tweedie → measurement loss → `.backward()`. 반드시 직접 의사코드를 적어 볼 것.
- **§3.3 (Lemma 1)**: 매니폴드-접 자동성. chain rule로 $\partial \hat{\boldsymbol x}_0/\partial \boldsymbol x_t$가 매니폴드 접공간 사영 역할. MCG의 발산 메커니즘과 대비.
- **§4 (Experiments)**: phase retrieval (Fig. 8)에 주목 — DDRM이 적용 *불가능*한 case에서 DPS의 진가.
- **Common stumbling blocks**: (1) 왜 Tweedie 근사가 좋은가 (low noise level에서 분산 → 0), (2) Lemma 1의 chain rule이 실제로 어떻게 매니폴드 접에 정렬되는지, (3) projection 제거가 noisy 환경에서 *왜* 결정적인지 (MCG와의 대조).

**English**:
- **§2 Background**: SDE machinery and reverse SDE; skim Song 2021b if unfamiliar.
- **§3.1 (Theorem 1)**: the key approximation. The Jensen-gap is bounded by curvature × variance. Skip detailed proof on first read.
- **§3.2 (Algorithm 1)**: the code is 5 lines — write it out yourself: `x_t.requires_grad_(True)` → score → Tweedie → measurement loss → `.backward()`.
- **§3.3 (Lemma 1)**: automatic manifold tangency via the chain-rule Jacobian $\partial\hat{\boldsymbol x}_0/\partial\boldsymbol x_t$. Contrast with MCG's projection-induced drift.
- **§4 Experiments**: pay attention to phase retrieval (Fig. 8) — a regime where DDRM is *inapplicable*.
- **Stumbling blocks**: (1) why the Tweedie approximation is good (variance vanishes at low noise), (2) how the chain rule actually aligns with manifold tangents, (3) why dropping projection is decisive in noisy settings (contrast with MCG).

---

## 7. 현대적 의의 / Modern Significance

**한국어**: DPS는 *score-based inverse problem solving의 universal API*가 되었다. forward operator $\mathcal A$의 미분 가능성만 요구하므로 의료 영상(MRI, CT, 형광 deconvolution), 천체관측(coronagraphy, phase retrieval, Fourier ptychography), 마이크로스코피(low-photon, motion-blur) 등 *비선형 + non-Gaussian noise* 가 일반적인 도메인에 직접 응용된다. 이후 후속 연구들 — DPM-Solver-DPS, $\Pi$GDM, ReSample, latent-DPS — 가 (a) NFE 절감 (1000 → 50-100), (b) latent diffusion 결합, (c) blind problem(forward operator도 모름) 으로 확장한다. 본 reading list에서는 DDRM (paper #26)의 spectral 한계를 극복하는 직계 후속이며, DiffPIR (paper #30)과 PnP-HQS vs ancestral DDPM의 두 직교 접근을 형성한다. low-SNR imaging의 핵심 score-prior + measurement-likelihood 워크플로우.

**English**: DPS has become the **universal API for score-based inverse problem solving**. Requiring only differentiability of $\mathcal A$, it applies directly to medical imaging (MRI, CT, fluorescence deconvolution), astronomy (coronagraphy, phase retrieval, Fourier ptychography), and microscopy (low-photon, motion-blur) — domains where nonlinear forward operators and non-Gaussian noise are typical. Successors — DPM-Solver-DPS, $\Pi$GDM, ReSample, latent-DPS — extend the framework with (a) reduced NFE (1000 → 50-100), (b) latent diffusion, (c) blind problems (unknown forward operators). Within this reading list, DPS is the direct successor that lifts DDRM's (#26) spectral linearity restriction and forms — together with DiffPIR (#30) — two orthogonal integration patterns (PnP-HQS vs ancestral DDPM). It is the canonical score-prior + measurement-likelihood workflow for low-SNR imaging.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
