---
title: "Pre-Reading Briefing: Optimal Inversion of the Generalized Anscombe Transformation for Poisson-Gaussian Noise"
paper_id: "14_makitalo_2013"
topic: Low_SNR_Imaging
date: 2026-05-06
type: briefing
---

# Optimal Inversion of the Generalized Anscombe Transformation: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Mäkitalo, M., & Foi, A., "Optimal inversion of the generalized Anscombe transformation for Poisson-Gaussian noise", *IEEE Trans. Image Process.*, 22(1), 91–103 (2013).
**Author(s)**: Markku Mäkitalo, Alessandro Foi
**Year**: 2013

---

## 1. 핵심 기여 / Core Contribution

### 한국어
이 논문은 **VST(variance-stabilising transformation) + Gaussian denoiser + inverse VST** 파이프라인의 마지막 약점인 *역변환 단계의 편향*을 해결한다. 핵심 통찰: 일반화 Anscombe 변환(GAT)은 분산은 안정화하지만 *평균*에 시스템적 편향을 도입한다. 단순한 algebraic inverse $(D/2)^2 - 3/8 - \sigma^2$는 *매우 어두운 영역에서 음수*가 되거나 PSNR 4–5 dB 손실을 일으킨다. 본 논문은 GAT의 정확한 비편향 역변환 $\mathcal I_\sigma : E\{f_\sigma(z)|y,\sigma\} \mapsto y$를 정의하고 (i) $(y,\sigma)$ 격자에서 수치 적분 → LUT로 구현, (ii) maximum-likelihood inverse와 일치함을 증명, (iii) 5-항 closed-form 근사 ($L^\infty = 0.0468$). 이 정확한 inverse를 BM3D와 결합하면 Cameraman peak 1, σ=0.1에서 algebraic inverse 15.7 dB 대비 **20.2 dB**(+4.5 dB) — PURE-LET와 동등하면서 plug-and-play로 *임의의 가우시안 디노이저*를 활용 가능. 이는 잡음별 알고리즘 설계가 아닌 "VST + 좋은 AWGN denoiser + exact inverse"라는 새 패러다임을 정립.

### English
This paper fixes the long-standing weak point of VST-based denoising pipelines: **the bias introduced at the inverse-VST stage**. The forward generalised Anscombe transformation (GAT) stabilises variance but introduces a systematic mean bias that the algebraic inverse $(D/2)^2 - 3/8 - \sigma^2$ does not undo — at low photon counts it even produces *negative* outputs and loses 4–5 dB in PSNR. The authors define the **exact unbiased inverse** $\mathcal I_\sigma: E\{f_\sigma(z)|y,\sigma\} \mapsto y$, compute it via numerical quadrature on a $(y,\sigma)$ grid, store it as a lookup table, and show it (i) coincides with the maximum-likelihood inverse, (ii) admits a 5-term closed-form approximation with $L^\infty = 0.0468$. Pairing this exact inverse with BM3D matches PURE-LET (paper #13) at all noise levels while leveraging *any* off-the-shelf AWGN denoiser plug-and-play. This refutes paper #13's implicit claim that bypassing VSTs is necessary, and shifts the field's focus toward designing better AWGN denoisers.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting
**한국어**: 2010–2012년 저광량 영상 처리는 두 갈래의 정점에 있었다. (i) Luisier-Blu-Unser의 PURE-LET(paper #13)은 VST를 *우회*해 직접 포아송-가우시안 잡음을 다루며 "VST는 저광자에서 실패"라는 입장; (ii) Mäkitalo-Foi 자신은 2011년에 *순수 포아송*에 대한 exact unbiased inverse를 제시. 본 논문은 이 후자를 *혼합* Poisson-Gaussian으로 확장하면서 (i)의 진단을 부분적으로 *반박*한다 — 분산이 아니라 *평균* 편향이 문제였고, 그것은 inverse-side에서 고칠 수 있다는 것이다. BM3D(2007)가 이미 표준 AWGN denoiser로 자리잡은 시점에서, "Anscombe + BM3D + exact inverse"가 paper #13의 PURE-LET과 동등 또는 우월함을 입증함으로써 이후 deep AWGN denoiser(DnCNN, FFDNet, Restormer)가 등장할 때 곧바로 동일 파이프라인에 끼워넣을 수 있는 길을 열었다.

**English**: By 2010–2012, low-light denoising had crystallised around two camps: (i) PURE-LET (paper #13) bypassed VSTs entirely, arguing they fail at low photon counts; (ii) Mäkitalo-Foi themselves had, in 2011, given an exact unbiased inverse for *pure-Poisson* Anscombe. This paper extends that 2011 result to the *mixed* Poisson-Gaussian regime and partially overturns PURE-LET's diagnosis — the real problem was *mean bias*, not variance, and it could be fixed on the inverse side. With BM3D (2007) already standard for AWGN, the new pipeline "Anscombe + BM3D + exact inverse" matched or beat PURE-LET, paving the way for the subsequent deep AWGN denoisers (DnCNN, FFDNet, Restormer) to slot into the same pipeline.

### 타임라인 / Timeline
```
1948 ─── Anscombe — VST for pure Poisson (paper #11)
1995 ─── Murtagh-Starck-Bijaoui — generalised Anscombe (GAT) for Poisson-Gaussian
2007 ─── Dabov-Foi-Katkovnik-Egiazarian — BM3D (paper #7; standard AWGN denoiser)
2008 ─── Foi+ — practical Poisson-Gaussian noise fitting (auto $\alpha,\sigma$)
2011 ─── Mäkitalo-Foi — exact unbiased inverse for pure-Poisson Anscombe
2011 ─── Luisier-Blu-Unser — PURE-LET (paper #13)
2013 ★★ Mäkitalo-Foi — exact unbiased inverse of GAT (THIS PAPER)
2017+ ── DnCNN / FFDNet / Restormer — deep AWGN denoisers slotting into the same pipeline
```

---

## 3. 필요한 배경 지식 / Prerequisites

- **수학 / Math**:
  - Conditional expectation $E[X|Y]$, MMSE estimator의 정의.
  - $f_\sigma$ nonlinear → $f_\sigma^{-1}(E[f_\sigma(z)]) \ne E[z]$ 의 비교환성.
  - Numerical quadrature (수치 적분), lookup table interpolation.
  - Maximum likelihood estimation 의 기본.
- **신호처리 / Signal processing**:
  - Variance-stabilising transformation: 비등분산 잡음을 등분산화.
  - AWGN denoiser (BM3D, BLS-GSM)의 입력/출력 가정.
  - Affine reduction: $\check z = \alpha p + \check n \to z = p + n$ 변수 치환.
- **통계 / Statistics**:
  - Poisson-Gaussian compound: $z = p + n$, $p \sim \mathcal P(y)$, $n \sim N(0,\sigma^2)$.
  - 분산-안정화 vs 평균-편향: 분산만 정규화해도 평균에는 *시스템적 시프트*가 남음.
- **선행 논문 / Prior reading**:
  - Paper #7 (BM3D) — 본 논문의 핵심 가우시안 디노이저.
  - Paper #11 (Anscombe) — VST 의 forward 단계.
  - Paper #13 (PURE-LET) — 본 논문이 *반박*하는 직접 경쟁자.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| GAT | Generalised Anscombe Transform — Poisson-Gaussian 용 forward VST. / Forward VST for Poisson-Gaussian noise. |
| Algebraic inverse | $(D/2)^2 - 3/8 - \sigma^2$ — forward GAT의 단순 대수적 역. 평균 편향 보정 못함. / The naive algebraic inverse of GAT; does not correct mean bias. |
| Exact unbiased inverse $\mathcal I_\sigma$ | $E\{f_\sigma(z)|y,\sigma\} \mapsto y$ — 조건부 기댓값 기반 정확한 역변환. / Conditional-expectation-based exact inverse map. |
| LUT | Lookup Table — $\mathcal I_\sigma$를 $(y,\sigma)$ 격자에서 사전 계산해 저장. / Pre-computed table of $\mathcal I_\sigma$ on a $(y,\sigma)$ grid. |
| ML inverse | Maximum-likelihood inverse — 잔차 분포 가정 하에 가장 그럴듯한 $y$. $\mathcal I_\sigma$와 일치. / Maximum-likelihood inverse; coincides with $\mathcal I_\sigma$ under unimodal-residual assumption. |
| Asymptotic inverse | $\mathcal I_0(D) - \sigma^2$ — $D$ 또는 $\sigma$가 큰 한계에서 $\mathcal I_\sigma$ 의 근사. / Approximation valid when $D$ or $\sigma$ is large. |
| Affine reduction | $z = (\check z - \mu)/\alpha$ 변수 치환으로 단순 모델로 환원. / Variable change reducing the scaled-shifted model to $z = p + n$. |
| BM3D | Block-Matching 3D — 표준 AWGN denoiser (paper #7). / Standard AWGN denoiser used as the Gaussian step. |
| Foi+ 2008 estimator | sample-mean / sample-variance regression으로 $\alpha, \sigma$ 자동 추정. / Automatic noise-parameter estimation from a single noisy raw image. |
| Heteroscedasticity | 픽셀별 분산이 $y$에 의존하는 비등분산 성질. / Pixel-wise variance dependent on the underlying $y$. |
| Pipeline plug-and-play | VST + 임의의 AWGN denoiser + exact inverse 조합으로 잡음 모델 자동 처리. / Combines any AWGN denoiser with VST + exact inverse to handle Poisson-Gaussian. |

---

## 5. 수식 미리보기 / Equations Preview

### 5.1 Generalised Anscombe Transformation (Eq. 7) / 일반화 안스콤
$$
f_\sigma(z) = \begin{cases} 2\sqrt{z + \tfrac{3}{8} + \sigma^2}, & z > -\tfrac{3}{8} - \sigma^2 \\ 0, & \text{otherwise} \end{cases}
$$
$\sigma=0$이면 표준 Anscombe로 환원. 분산은 거의 1로 안정화되지만 평균은 *편향됨*. / Reduces to the standard Anscombe when $\sigma=0$; stabilises variance to ~1 but introduces a systematic mean bias.

### 5.2 Exact unbiased inverse (Eq. 9, 10) / 정확한 비편향 역변환
$$
\mathcal I_\sigma : E\{f_\sigma(z) | y, \sigma\} \longmapsto y
$$
$$
E\{f_\sigma(z) | y, \sigma\} = \int 2\sqrt{z + \tfrac{3}{8} + \sigma^2} \sum_{k=0}^\infty \tfrac{y^k e^{-y}}{k!} \cdot \tfrac{1}{\sqrt{2\pi\sigma^2}} e^{-(z-k)^2/(2\sigma^2)} dz
$$
적분을 96 × 1199 격자에서 수치적으로 평가 후 LUT 저장, 보간으로 평가. / Compute the integral on a 96 × 1199 $(y,\sigma)$ grid via numerical quadrature, store as a lookup table.

### 5.3 ML interpretation (Eq. 14) / ML 해석
$$
\mathcal I_{\rm ML}(D) = \begin{cases} \mathcal I_\sigma(D), & D \ge E\{f_\sigma(z) | 0, \sigma\} \\ 0, & \text{otherwise} \end{cases}
$$
잔차 $\xi = D - E\{f_\sigma(z)|y,\sigma\}$가 0-mode unimodal일 때 ML inverse와 정확히 일치. / Coincides with the ML inverse under a 0-mode unimodal residual assumption.

### 5.4 Closed-form approximation (Eq. 21) / 닫힌형 근사
$$
\widetilde{\mathcal I_\sigma}(D) = \tfrac{1}{4} D^2 + \tfrac{1}{4}\sqrt{\tfrac{3}{2}}\,D^{-1} - \tfrac{11}{8}\,D^{-2} + \tfrac{5}{8}\sqrt{\tfrac{3}{2}}\,D^{-3} - \tfrac{1}{8} - \sigma^2
$$
$L^2$ error 0.0069, $L^\infty$ error 0.0468. LUT 없이도 $\mathcal I_\sigma$ 근사 가능. / 5-term polynomial closed form; $L^\infty$ error 0.0468 — accurate enough for practice without the LUT.

### 5.5 Algebraic inverse for comparison / 비교용 대수 역변환
$$
\mathcal I_{\rm alg}(D) = \tfrac{1}{4} D^2 - \tfrac{3}{8} - \sigma^2, \quad
\mathcal I_{\rm asy}(D) = \tfrac{1}{4} D^2 - \tfrac{1}{8} - \sigma^2
$$
$\mathcal I_{\rm alg}$는 단순 algebraic 역; $\mathcal I_{\rm asy}$는 점근적으로 비편향. 둘 다 저광량에서 실패. / Naive algebraic and asymptotic-unbiased forms; both fail at low photon counts.

---

## 6. 읽기 가이드 / Reading Guide

### 한국어
**우선 읽을 부분 / Focus first**:
1. **§III-A Eq. 9–10** — 핵심 정의. *왜* algebraic inverse가 틀렸는지 (비선형 $f_\sigma$와 expectation 의 비교환성) 직관적 이해.
2. **Fig. 1(c)** — 알지브래익 vs 정확한 inverse 의 시각적 차이. $\sigma=1$일 때 algebraic이 $\hat y \to -1.375$ (음수)로 발산함을 확인.
3. **§III-B ML interpretation** — Appendix B의 unimodal-residual 가정 하에 $\mathcal I_\sigma = \mathcal I_{\rm ML}$. 통계적으로 "최적"의 정확한 의미.
4. **§IV Table I peak 1 column** — 4.5 dB 격차 ($\mathcal I_{\rm asy}$ 15.55 vs $\mathcal I_\sigma$ 20.23) 가 "쓸모없음 vs 쓸만함"의 경계임을 시각적으로 확인.

**자주 헷갈리는 지점 / Common stumbling blocks**:
- $E\{f_\sigma(z)|y,\sigma\} \ne f_\sigma(E\{z|y,\sigma\}) = f_\sigma(y)$ — 비선형 함수와 기댓값은 교환되지 않는다 (Jensen-style).
- Algebraic inverse 가 *왜* 음수를 만드는가: $D = 0$일 때 $(0/2)^2 - 3/8 - \sigma^2 = -3/8 - \sigma^2 < 0$.
- Affine reduction (§II Eq. 4–5): $\alpha, \mu$는 calibration 파라미터; 모든 분석은 $z = p + n$의 단순 모델에서 진행.
- 본 논문은 *forward* GAT는 그대로 두고 *inverse*만 고친다. paper #11(Anscombe), #13(PURE-LET)와의 비교에서 forward 단계 차이가 아님에 주의.

### English
**Focus first**:
1. **§III-A Eqs. 9–10** — Core definition. Build intuition for *why* the algebraic inverse is wrong (nonlinear $f_\sigma$ does not commute with expectation).
2. **Fig. 1(c)** — Visual comparison of algebraic vs exact inverses. Note that at $\sigma=1$, the algebraic inverse goes to $-1.375$ as $D \to 0$ (negative — physically meaningless).
3. **§III-B ML interpretation** — Under the unimodal-residual assumption (Appendix B), $\mathcal I_\sigma = \mathcal I_{\rm ML}$ — the precise meaning of "optimal".
4. **§IV Table I peak-1 column** — The 4.5 dB gap ($\mathcal I_{\rm asy}$ 15.55 vs $\mathcal I_\sigma$ 20.23) is the boundary between "unusable" and "useful" output.

**Common stumbling blocks**:
- $E\{f_\sigma(z)|y,\sigma\} \ne f_\sigma(E\{z|y,\sigma\}) = f_\sigma(y)$ — nonlinear functions don't commute with expectation.
- Why the algebraic inverse produces negatives: at $D=0$, $(0/2)^2 - 3/8 - \sigma^2 = -3/8 - \sigma^2 < 0$.
- The affine reduction (§II) absorbs $\alpha, \mu$ into a unit-scale model $z = p + n$ — all subsequent analysis lives in this reduced model.
- The paper *only* modifies the inverse step; the forward GAT is unchanged. When comparing with #11 and #13, do not conflate forward and inverse roles.

---

## 7. 현대적 의의 / Modern Significance

### 한국어
이 논문이 정립한 "VST + AWGN denoiser + exact inverse" 파이프라인은 deep learning 시대에도 표준으로 살아남았다. DnCNN(2017), FFDNet(2018), Restormer(2022) 등 deep AWGN denoiser가 발전함에 따라 그 어떤 새 모델도 *그대로* GAT + Restormer + $\mathcal I_\sigma$ 형태로 저광량 영상에 적용 가능 — 잡음별 재학습 불필요. 또한 self-supervised denoising(Noise2Noise, Noise2Self) 시대에도 GAT는 *변환 후 등분산 가우시안* 가정을 단순화시키는 전처리로 자주 쓰인다. 본 논문의 더 깊은 교훈은 "잡음 모델 구체성을 forward stage에 흡수하고 inverse stage에서 정확하게 보정하라" — 이는 score-based diffusion(score function이 noise schedule과 분리)이나 plug-and-play prior(prior는 모델, likelihood는 분리 처리)의 디자인 철학과 같은 줄기에 있다.

### English
The "VST + AWGN denoiser + exact inverse" pipeline this paper established remains the standard even in the deep-learning era. As DnCNN (2017), FFDNet (2018), and Restormer (2022) advanced AWGN denoising, each could *immediately* be deployed for low-light imaging via GAT + ModernAWGN + $\mathcal I_\sigma$ — no per-noise retraining. Even in the self-supervised denoising era (Noise2Noise, Noise2Self), GAT is often used as a preprocessing step to simplify the noise model. The deeper lesson — "absorb noise-model specifics into a forward stage and correct exactly on the inverse" — runs in the same vein as score-based diffusion (where the score is decoupled from the noise schedule) and plug-and-play priors (where the prior is the model and the likelihood is handled separately).

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
