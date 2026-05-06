---
title: "Pre-Reading Briefing: Noise2Noise: Learning Image Restoration without Clean Data"
paper_id: "16_lehtinen_2018"
topic: Low_SNR_Imaging
date: 2026-05-06
type: briefing
---

# Noise2Noise: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Lehtinen, J., Munkberg, J., Hasselgren, J., Laine, S., Karras, T., Aittala, M., & Aila, T., "Noise2Noise: Learning Image Restoration without Clean Data", *Proc. 35th ICML*, PMLR 80, pp. 2965–2974 (2018). arXiv:1803.04189.
**Author(s)**: Jaakko Lehtinen, Jacob Munkberg, Jon Hasselgren, Samuli Laine, Tero Karras, Miika Aittala, Timo Aila
**Year**: 2018

---

## 1. 핵심 기여 / Core Contribution

### 한국어
이 논문은 영상 복원(denoising, MRI 재구성, Monte-Carlo rendering 등)을 위한 회귀 신경망 학습에 **깨끗한 ground-truth 영상이 전혀 필요 없다**는 충격적인 사실을 *기초 통계학*만으로 보였다. 핵심 통찰은 한 줄: MSE 손실로 학습된 회귀기는 *조건부 기댓값* $\mathbb E[y\mid x]$로 수렴하므로, 타겟 $y$를 *zero-mean noise* 로 오염시킨 $\hat y$로 바꿔도 $\mathbb E[\hat y\mid x] = \mathbb E[y\mid x]$이라 *최적 파라미터가 변하지 않는다*. 유한 데이터에서는 추가 분산이 생기지만 학습 데이터가 충분하면 평균되어 사라진다 ($\sigma_n^2 / N_{\rm train}$).

저자들은 이 단일 통찰로 (i) Gaussian, (ii) Poisson, (iii) Bernoulli, (iv) impulse(annealed $L_0$), (v) Monte-Carlo path-tracing artefact, (vi) under-sampled MRI 재구성을 *깨끗한 영상 한 장도 보지 않고* 학습 — 결과 PSNR이 깨끗 타겟 학습 *오라클* 수준과 0.01 dB 이내에서 일치(BSD300/$\sigma=25$: 31.07 vs 31.06 dB). 이 논문은 이후 모든 self-supervised denoising(Noise2Void, Noise2Self, Cryo-CARE, Self2Self, Neighbor2Neighbor)의 출발점.

### English
This paper establishes — using only basic statistics — that a regression network for image restoration can be trained **without any clean reference images**. The single insight: a network trained with MSE loss converges to the conditional expectation of its target. Replacing clean targets with zero-mean-corrupted targets does not change $\mathbb E[\hat y\mid x] = \mathbb E[y\mid x]$, so the optimum is unchanged. Finite-sample variance scales as $\sigma_n^2 / N_{\rm train}$ and vanishes for large datasets. The authors verify this across Gaussian, Poisson, Bernoulli, impulse (annealed $L_0$), Monte-Carlo path-tracing artefacts, and undersampled MRI — matching clean-target performance to within statistical noise (e.g., BSD300/$\sigma=25$: clean 31.07 dB vs N2N 31.06 dB). The result lifts a long-standing requirement and is the foundation of every subsequent self-supervised denoiser (Noise2Void, Noise2Self, Cryo-CARE, Self2Self, Neighbor2Neighbor).

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting
**한국어**: 2017년 시점 deep denoising은 "supervised regression" 패러다임이 정착해 있었다 — DnCNN(Zhang 2017), RED30(Mao 2016), CARE(Weigert 2017) 등이 모두 *깨끗한 reference 영상 + 잡음 영상* 페어를 요구. 이는 사진(long-exposure 가능)에는 자연스럽지만 (i) 살아있는 생체 시료(life-cell, cryo-EM), (ii) 임상 MRI(full k-space는 임상에서 비현실), (iii) Monte-Carlo 렌더링(131k spp/픽셀 렌더는 GPU 시간 ~40분/이미지) 같은 도메인에서는 *물리적 또는 경제적으로* 불가능. Lehtinen et al.은 NVIDIA/Aalto 출신 그래픽스 연구자로, 출발점이 사진이 아니라 *MC rendering*이었다. 그들은 "rendering noise는 zero-mean by construction"임을 알았고, "MSE loss는 conditional mean에 수렴" 이라는 *정확한 통계학*을 결합해 supervised denoising의 기초 가정을 한 줄로 무너뜨렸다.

**English**: By 2017, deep denoising had settled into the "supervised regression" paradigm — DnCNN (Zhang 2017), RED30 (Mao 2016), CARE (Weigert 2017) all required *clean-reference + noisy* training pairs. This worked for photography (long exposures are easy) but failed for (i) live-cell biology / cryo-EM, (ii) clinical MRI (fully sampled k-space is impractical), and (iii) Monte-Carlo rendering (~40 min/image at 131k spp). Lehtinen et al., a graphics group from NVIDIA/Aalto, came at it from the rendering side: they knew rendering noise is zero-mean by construction and combined that with the elementary statistical fact that MSE loss converges to a conditional mean — destroying the clean-target assumption with a one-line argument.

### 타임라인 / Timeline
```
1964 ─── Huber — robust M-estimators (mean / median / mode)
2005 ─── Buades+ — NL-means (paper #4)
2007 ─── Dabov+ — BM3D (paper #7)
2016 ─── Mao+ — RED30 (supervised CNN denoiser)
2017 ─── Zhang+ — DnCNN; Weigert+ — CARE (both require clean targets)
2018 ★★ Lehtinen+ — Noise2Noise (THIS PAPER)
2019 ─── Buchholz+ — Cryo-CARE (paper #15, applied N2N to cryo-EM)
2019 ─── Krull+ — Noise2Void (paper #17, single-image extension)
2019 ─── Batson-Royer — Noise2Self (paper #18, J-invariance theory)
2020 ─── Quan+ — Self2Self; 2021 — Huang+ — Neighbor2Neighbor
```

---

## 3. 필요한 배경 지식 / Prerequisites

- **수학 / Math**:
  - Conditional expectation $\mathbb E[Y\mid X]$ 의 정의와 *iterated expectation*: $\mathbb E[Y] = \mathbb E_X[\mathbb E_{Y\mid X}[Y]]$.
  - MSE loss와 conditional mean의 관계: $\arg\min_z \mathbb E[(z-Y)^2] = \mathbb E[Y]$.
  - $L_1 \to$ median, $L_0 \to$ mode (M-estimator family).
- **딥러닝 / Deep learning**:
  - U-Net (Ronneberger+ 2015), CNN regression, residual networks (RED30).
  - Stochastic gradient descent, mini-batch training, gradient averaging.
  - L_1 / L_2 / annealed L_0 손실 함수의 학습 동역학.
- **확률 / Probability**:
  - 독립 확률변수의 합, 분산 누적 ($\mathrm{Var}/N$).
  - Zero-mean 조건의 의미.
- **응용 / Applications** (개념만):
  - Monte-Carlo path tracing의 spp(samples-per-pixel) 개념.
  - MRI의 k-space sampling 과 IFFT 기반 영상 재구성.
- **선행 논문 / Prior reading**:
  - 없어도 되지만 paper #14(GAT + BM3D), #7(BM3D)와 비교 시 도움.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| Conditional expectation | $\mathbb E[Y\mid X]$ — $X$가 주어졌을 때 $Y$의 평균. MSE 회귀의 최적해. / Mean of $Y$ given $X$; optimum of MSE regression. |
| M-estimator | $\arg\min_z \mathbb E[L(z,Y)]$ 형태의 점 추정기. / Point estimator from minimising expected loss. |
| RED30 | Mao+ 2016, 30층 hierarchical residual encoder-decoder. 본 논문 §3.1 baseline. / 30-layer residual encoder-decoder; baseline in §3.1. |
| Blind denoising | $\sigma$를 학습 시 random range에서 sampling → 추론 시 $\sigma$ 추정 불필요. / Random-$\sigma$ training; inference without knowing noise level. |
| Capture budget | $N$ latents × $M$ shots/latent = 고정 측정 비용. N2N의 sample efficiency 분석. / Total acquisition budget; basis for N2N's sample-efficiency analysis. |
| Brown noise | Gaussian noise + spatial smoothing → spatially correlated. / Spatially correlated Gaussian noise. |
| Annealed $L_0$ | $(|\,\cdot\,| + \epsilon)^\gamma$, $\gamma : 2 \to 0$ — mode-seeking. / Mode-seeking loss with $\gamma$ annealed. |
| HDR loss | Reinhard tone mapping을 입력에만 적용; 손실은 relative-MSE in linear space. / Tone-map only inputs; loss in linear-luminance space. |
| Spp (samples per pixel) | Monte-Carlo rendering 의 픽셀당 광선 샘플 수. / Number of rays sampled per pixel in MC path tracing. |
| Bernoulli mask | 픽셀이 $p$ 확률로 *측정됨*; 마스크 픽셀에서만 손실. / Pixel observed with probability $p$; loss only on observed pixels. |
| K-space | MRI의 Fourier 도메인 표현; 부분 샘플링이 reconstruction artefact 발생. / Fourier-domain representation of MRI data. |
| ImageNet 50k | 학습에 사용된 ImageNet 검증 영상 50k 장. / 50k ImageNet validation images used as training data. |

---

## 5. 수식 미리보기 / Equations Preview

### 5.1 Point estimation as ancestor (Eq. 2) / 점 추정의 조상
$$
z^* = \arg\min_z \mathbb E_y[L(z,y)]
$$
$L = L_2 \Rightarrow z^* = \mathbb E[y]$; $L = L_1 \Rightarrow z^* = \mathrm{median}(y)$; $L = L_0 \Rightarrow z^* = \mathrm{mode}(y)$. M-estimator 의 통계량 매핑. / The loss-statistic mapping at the heart of M-estimators.

### 5.2 Standard supervised regression (Eq. 4–5) / 표준 지도학습
$$
\theta^* = \arg\min_\theta \mathbb E_{(x,y)}[L(f_\theta(x), y)] = \arg\min_\theta \mathbb E_x\bigl[\mathbb E_{y\mid x}[L(f_\theta(x), y)]\bigr]
$$
조건부 분해 → 내부 기댓값만 $x$ 의존 → 타겟 분포를 *조건부 기댓값이 같은 다른 분포*로 바꿔도 최적 $\theta$ 불변. / Conditional factorisation: only the conditional expectation of $y$ given $x$ matters for the optimum.

### 5.3 The Noise2Noise principle / 노이즈투노이즈 원리
$$
\hat x = s + n, \quad \hat y = s + n', \quad n \perp n', \quad \mathbb E[n] = \mathbb E[n'] = 0
$$
$$
\boxed{\;\mathbb E[\hat y \mid \hat x] = \mathbb E[s\mid \hat x] + \mathbb E[n'\mid \hat x] = \mathbb E[s\mid \hat x]\;}
$$
zero-mean noisy 타겟이 깨끗 타겟과 동일 conditional mean → 최적 $\theta$ 불변. / Zero-mean noisy targets share the conditional mean of clean targets, so the optimum is preserved.

### 5.4 Algebraic gradient identity / 그래디언트 동등성
$$
\frac{\partial}{\partial \theta} \mathbb E_{y\mid x}[(f_\theta(x) - y)^2]
= 2\bigl(f_\theta(x) - \mathbb E[y\mid x]\bigr)\frac{\partial f_\theta(x)}{\partial \theta}
$$
그래디언트는 *오직* $\mathbb E[y\mid x]$에만 의존 → 노이즈 타겟 사용해도 학습 동역학이 깨끗 타겟과 동일. / The gradient depends only on the conditional mean, so noisy targets give identical training dynamics in expectation.

### 5.5 Excess variance bound / 추가 분산 한계 (appendix-level)
$$
\mathrm{Var}_{\theta^*}^{\mathrm{N2N}} - \mathrm{Var}_{\theta^*}^{\mathrm{clean}} \approx \frac{\sigma_n^2}{N_{\mathrm{train}}}
$$
타겟 잡음 분산을 학습 영상 수로 나눈 것 → 충분히 큰 데이터셋에서 *공짜로* clean-target과 같음. / Excess variance scales as target-noise-variance / training-set-size; vanishes for large $N_{\rm train}$.

---

## 6. 읽기 가이드 / Reading Guide

### 한국어
**우선 읽을 부분 / Focus first**:
1. **§2 Theoretical Background (Eqs. 2–6)** — 2 페이지짜리 핵심. 모든 후속 self-supervised denoising 의 기반. M-estimator → MSE → conditional mean → noisy target invariance 의 *논리 체인*을 정확히 따라가기.
2. **§3.1 Gaussian Table 1** — 31.07 vs 31.06 dB. *0.01 dB 차이*가 통계적 noise 수준임을 확인 (5-seed CI ±0.02 dB).
3. **Fig. 1c capture-budget 분석** — N2N이 깨끗 타겟보다 *더* sample-efficient 한 이유. $M(M-1)$ noisy/noisy 페어 vs 1 clean per latent.
4. **§3.2 손실 함수 매칭** — Zero-mean → $L_2$, median-preserving (text overlay) → $L_1$, mode-preserving (impulse) → annealed $L_0$. *손실은 잡음 통계량에 맞춰야* 한다는 핵심.
5. **§3.3 Monte-Carlo** — N2N이 *해석 불가능한* (heavy-tailed, signal-dependent) 노이즈에서도 동작함을 보이는 가장 강력한 시연.

**자주 헷갈리는 지점 / Common stumbling blocks**:
- N2N은 "노이즈를 노이즈로 매핑"이 아니다 — 학습은 조건부 기댓값으로 수렴하므로 *입력 노이즈를 평균화* 하는 함수를 배우는 것.
- Bernoulli 결과 (32.02 dB > 31.85 dB clean target) 가 의외 — *target stochasticity*가 dropout-like regularisation 역할.
- Spatial correlation (brown noise) 은 *수렴 속도*만 늦추고 *최종 PSNR*은 거의 같음. 잡음 평균화의 어려움이지 본질적 한계가 아님.
- HDR Monte-Carlo loss (Eq. given in §3.3) 는 분모의 그래디언트를 0으로 처리하는 미묘한 trick — relative-MSE 가 *조건부 기댓값*을 바꾸지 않게 하는 디테일.

### English
**Focus first**:
1. **§2 Theoretical Background (Eqs. 2–6)** — Two pages that ground every subsequent self-supervised denoiser. Trace the logic chain *precisely*: M-estimator → MSE → conditional mean → noisy-target invariance.
2. **§3.1 Gaussian Table 1** — 31.07 vs 31.06 dB. Verify that the 0.01 dB gap is statistical noise (5-seed CI ±0.02 dB).
3. **Fig. 1c capture-budget analysis** — Why N2N is *more* sample-efficient than clean-target training: $M(M-1)$ noisy-noisy pairs from $M$ realisations of $N$ latents.
4. **§3.2 loss-noise matching** — Zero-mean → $L_2$, median-preserving (text overlay) → $L_1$, mode-preserving (impulse) → annealed $L_0$. The principle that *loss must match the relevant statistic of the noise*.
5. **§3.3 Monte-Carlo** — The strongest demonstration: N2N works even on noise that is heavy-tailed, signal-dependent, and analytically intractable.

**Common stumbling blocks**:
- N2N is *not* "mapping noise to noise" — training converges to the conditional mean, so the network learns a function that *averages out* input noise.
- The Bernoulli result (32.02 dB > clean-target 31.85 dB) is counterintuitive — *target stochasticity* acts as dropout-like regularisation.
- Spatial correlation (brown noise) only slows *convergence*, not the final PSNR. It's a gradient-averaging difficulty, not a fundamental limit.
- The HDR Monte-Carlo loss (§3.3) zeros the gradient of its denominator — a subtle trick to keep relative-MSE from shifting the conditional mean.

---

## 7. 현대적 의의 / Modern Significance

### 한국어
Noise2Noise는 *self-supervised denoising 시대의 출발점*이다. 이후의 모든 변종(Noise2Void, Noise2Self, Cryo-CARE, Self2Self, Neighbor2Neighbor, Probabilistic N2V, DivNoising)은 N2N의 가정 — *paired independent noisy realisations* — 을 *어떻게 완화하는가*의 변주다. 더 넓게 보면 N2N의 핵심 통찰("conditional-expectation invariance")은 score-based diffusion(score function이 forward noising schedule에 의존하지 않는 conditional expectation)과도 같은 통계적 원리에 뿌리를 둔다. 또한 본 논문이 시연한 *MC rendering / MRI reconstruction* 응용은 deep generative prior 와 deep image reconstruction 의 자연스러운 확장으로 이어졌다 (DDIM-MC, score-based MRI). 실용적으로는 *cryo-EM, fluorescence microscopy, 임상 MRI, 라이브 이미징* 등 *깨끗한 GT가 물리적으로 불가능한* 모든 분야에서 deep denoising을 가능하게 한 *결정적 다리*다.

### English
Noise2Noise is the **starting point of the self-supervised denoising era**. Every subsequent variant (Noise2Void, Noise2Self, Cryo-CARE, Self2Self, Neighbor2Neighbor, Probabilistic N2V, DivNoising) is a relaxation of N2N's assumption — *paired independent noisy realisations*. More broadly, N2N's central insight ("conditional-expectation invariance") shares roots with score-based diffusion (the score function is a conditional expectation independent of the forward noising schedule) and modern denoising-prior frameworks. The MC-rendering and MRI-reconstruction applications demonstrated here led directly to score-based MRI and DDIM-style rendering. Practically, N2N is the *decisive bridge* that made deep denoising possible in fields where clean ground truth is physically impossible: cryo-EM, fluorescence microscopy, clinical MRI, and live imaging.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
