---
title: "Pre-Reading Briefing: Adaptive Wavelet Thresholding for Image Denoising and Compression"
paper_id: "03_chang_2000"
topic: Low_SNR_Imaging
date: 2026-05-06
type: briefing
---

# Adaptive Wavelet Thresholding for Image Denoising and Compression: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Chang, S. G., Yu, B., & Vetterli, M., "Adaptive Wavelet Thresholding for Image Denoising and Compression", *IEEE Transactions on Image Processing*, 9(9), 1532–1546 (2000). [DOI: 10.1109/83.862633]
**Author(s)**: S. Grace Chang, Bin Yu, Martin Vetterli
**Year**: 2000

---

## 1. 핵심 기여 / Core Contribution

### 한국어
이 논문은 두 가지 기여를 한다. (A) **BayesShrink 임계기**: 자연 영상의 wavelet 계수가 **일반화 가우시안 분포(GGD)** $\propto \exp\{-(\alpha|x|)^\beta\}$ 를 따른다는 경험적 사실을 prior 로 두고, 베이즈 위험을 최소화하는 soft-threshold 를 유도. 풀어보면 **닫힌 형태** 의 경험적 임계값 $T_B = \sigma^2/\sigma_X$ 를 얻는다 — 모든 GGD 모양 $\beta \in [0.5, 4]$ 에 대해 *최적 위험의 5% 이내*. **subband-adaptive** (각 scale × orientation 별 $\sigma_X$ 추정) 이라 SureShrink 보다 보통 ~8% 좋고 ~10× 빠르다. (B) **MDL 기반 동시 denoising + compression**: 임계화의 zero-zone 과 양자화의 dead-zone 이 일치한다는 통찰에서 출발해 Rissanen MDL 원리로 $(\hat T, \hat\Delta, \hat m)$ 을 동시에 결정 — 압축 자체가 denoising 을 한다.

### English
Two contributions: (A) **BayesShrink threshold**: Models wavelet coefficients with a **Generalized Gaussian Distribution (GGD)** prior $\propto \exp\{-(\alpha|x|)^\beta\}$ and minimises the Bayes risk over soft thresholds, obtaining the near-optimal **closed-form threshold** $T_B = \sigma^2/\sigma_X$. This formula is within **5% of the optimal Bayes risk** across the entire GGD shape range $\beta \in [0.5, 4]$, requiring no estimate of $\beta$. Subband-adaptive (per scale × orientation) and parameter-free, BayesShrink typically beats SureShrink by ~8% MSE while running ~10× faster. (B) **MDL-based simultaneous denoising + compression**: Coupling the thresholding zero-zone with the quantizer dead-zone, Rissanen-MDL jointly chooses $(\hat T, \hat\Delta, \hat m)$ — lossy compression *is* denoising in this view.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting
1990년대 후반, JPEG2000 의 wavelet-based 코딩이 표준화 직전이었고, Mallat-Zhong (1992), Antonini+ (1992) 등이 자연 영상의 wavelet 계수가 GGD 를 따른다는 것을 정립했다. Donoho-Johnstone (1994, 1995) 이 thresholding 이론을 닦았지만, *영상* 에 적용했을 때 universal threshold ($M = 512^2$ 에서 $\sigma\sqrt{2\log M} \approx 5.13\sigma$) 는 너무 커서 과도하게 매끈하게 만들었고, SureShrink 는 비용이 컸다. 이 논문은 *Bayesian* 관점으로 두 한계를 동시에 극복하며, *압축과 denoising 이 같은 절차* 라는 통찰까지 전달한다.
By the late 1990s JPEG2000's wavelet coder was nearly standardised, and Mallat-Zhong (1992), Antonini+ (1992) had established that natural-image wavelet subbands follow GGD. Donoho-Johnstone's thresholding theory worked beautifully for 1-D, but on $512\times 512$ images the universal threshold $\sigma\sqrt{2\log M} \approx 5.13\sigma$ over-smoothed, while SureShrink was expensive. This paper used the *Bayesian* viewpoint to overcome both limitations and tied compression and denoising into a single MDL framework.

### 타임라인 / Timeline
```
1989 ─── Mallat — 2-D wavelet decomposition for natural images
1989 ─── Rissanen — MDL principle
1992 ─── Mallat-Zhong, Antonini+ — wavelet coefficients ~ Generalized Gaussian
1994 ─── Donoho-Johnstone — VisuShrink (paper #1)
1995 ─── Donoho-Johnstone — SureShrink (paper #2)
1996 ─── Simoncelli-Adelson — Bayesian wavelet coring (Gaussian-scale-mixture)
1999 ─── Hyrkkö-Selesnick — analytical Laplacian-prior threshold
2000 ★★ CHANG-YU-VETTERLI — BayesShrink + MDL (THIS PAPER)
2002 ─── Sendur-Selesnick — Bivariate shrinkage (parent-child)
2003 ─── Portilla+ — BLS-GSM
2007 ─── Dabov+ — BM3D (paper #7)
```

---

## 3. 필요한 배경 지식 / Prerequisites

- **Paper #1 + #2 / Papers #1 and #2**: VisuShrink, SureShrink, soft thresholding, MAD/0.6745.
- **2-D dyadic DWT / 2-D dyadic DWT**: $\{LL_J, HH_k, HL_k, LH_k\}_{k=1}^J$ subband 분해.
- **GGD 분포 / Generalized Gaussian Distribution**: $\beta=2$ Gaussian, $\beta=1$ Laplacian, $\beta<1$ heavy-tailed.
- **Bayesian risk / Bayes risk**: $r(T) = E_{X,Y}(\eta_T(Y) - X)^2$, prior + likelihood.
- **분산 분해 / Variance decomposition**: $Y = X + V$ 독립 → $\sigma_Y^2 = \sigma_X^2 + \sigma^2$.
- **MDL 원리 / Minimum Description Length (Rissanen)**: $L(\mathbf{Y}, \hat{\mathbf{X}}) = L(\mathbf{Y}|\hat{\mathbf{X}}) + L(\hat{\mathbf{X}})$.
- **Quantization 기초 / Quantization basics**: dead-zone scalar quantizer, bin width $\Delta$, zero-zone width.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| BayesShrink | GGD prior + Bayes-optimal soft-threshold = $\sigma^2/\sigma_X$ / Bayes-optimal soft-threshold under GGD prior. |
| Generalized Gaussian (GGD) | $\propto \exp\{-(\alpha|x|)^\beta\}$, shape $\beta$ / Heavy-tailed distribution; $\beta\in[0.5,1]$ for natural images. |
| Subband-adaptive | scale × orientation 마다 다른 threshold / Different threshold per (scale, orientation) subband. |
| $\sigma_X$ (signal std) | subband 별 신호 표준편차 / Per-subband signal standard deviation. |
| $\hat\sigma_X = \sqrt{(\hat\sigma_Y^2 - \hat\sigma^2)_+}$ | empirical 추정 (clipped) / Empirical estimator (clipped to non-negative). |
| OracleShrink | 진짜 $\sigma_X$ 사용한 BayesShrink baseline / BayesShrink with true $\sigma_X$ — upper bound benchmark. |
| Zero-zone | 양자화기에서 0으로 매핑되는 구간 / Quantizer interval mapped to zero — coincides with thresholding. |
| Dead-zone quantizer | 중심 zero-zone + 외부 균등 bin / Central zero-zone + uniform outer bins. |
| MDL principle | 코드 길이 최소화 / Code-length minimisation criterion. |
| Coding length $L(\hat X)$ | quantized 신호 표현 비트 / Bits to represent quantised signal. |
| HH_1 subband | 가장 미세한 대각 detail — 잡음 추정 / Finest diagonal detail — used for $\hat\sigma$ via MAD. |

---

## 5. 수식 미리보기 / Equations Preview

**GGD prior (Eq. 5)**:
$$
GG_{\sigma_X, \beta}(x) = C(\sigma_X, \beta)\,\exp\bigl\{-(\alpha(\sigma_X, \beta)|x|)^\beta\bigr\}, \quad \alpha = \sigma_X^{-1}\sqrt{\Gamma(3/\beta)/\Gamma(1/\beta)}
$$

**Bayes risk for soft thresholding (Eq. 6)**:
$$
r(T) = E_X E_{Y|X}(\eta_T(Y) - X)^2, \quad Y|X \sim N(X, \sigma^2), \quad X \sim GG_{\sigma_X, \beta}
$$

**BayesShrink closed-form threshold (Eq. 12)** — 핵심:
$$
T_B(\sigma_X) = \frac{\sigma^2}{\sigma_X}
$$
직관: high SNR ($\sigma/\sigma_X \ll 1$) 에서 작은 임계 (신호 보존), low SNR ($\sigma/\sigma_X \gg 1$) 에서 큰 임계 (잡음 제거). 모든 $\beta$ 에 대해 진짜 $T^*$ 의 5% 이내.
Intuition: small threshold at high SNR (preserve signal), large threshold at low SNR (kill noise). Within 5% of the true Bayes-optimal $T^*$ across all GGD shapes.

**Parameter estimation (Eqs. 16–20)**:
$$
\hat\sigma = \mathrm{Median}(|Y_{ij}|)/0.6745 \;\;(Y \in HH_1), \qquad \hat\sigma_Y^2 = n^{-2}\sum Y_{ij}^2, \qquad \hat\sigma_X = \sqrt{(\hat\sigma_Y^2 - \hat\sigma^2)_+}
$$
음수면 subband 통째 0 ($\hat\sigma_X = 0 \Rightarrow$ 모든 계수 zeroed).

**MDL criterion (Eq. 21)**:
$$
L(\mathbf{Y}, \hat{\mathbf{X}}) = L(\mathbf{Y}|\hat{\mathbf{X}}) + L(\hat{\mathbf{X}})
$$
Rissanen MDL: $(\hat T, \hat\Delta, \hat m)$ 이 두 항의 합을 최소화.

---

## 6. 읽기 가이드 / Reading Guide

### 한국어
- **§I (Intro)**: Fig. 1 의 *thresholding ≈ zero-zone quantization* 관찰을 꼭 보고 시작.
- **§II.A (GGD + Bayes risk)**: 핵심. Eq. 5–7 은 정의·세팅, Eq. 9–12 가 진짜 결론. **Fig. 4–6** (다양한 $\beta$ 에 대해 $T^*$ vs $T_B = 1/\sigma_X$ 곡선) 이 *왜 이 단순 식이 잘 작동하는지* 시각으로 보여줌 — 반드시 확인.
- **§II.B (Parameter estimation)**: paper #1 의 MAD/0.6745 + 단순 분산 차분. $\hat\sigma_X = 0$ 처리 (entire subband zeroed) 도 잊지 말 것.
- **§III (MDL)**: 첫 읽기엔 *통찰* (zero-zone = thresholding) 만. 코딩 길이의 정확한 식은 second pass.
- **§IV (Experiments)**: Table I-IV 에서 BayesShrink vs SureShrink vs OracleShrink 비교. ~5–10% MSE 차이.
- **흔한 걸림돌**: (i) $T_B = \sigma^2/\sigma_X$ 의 *unit* — 둘 다 같은 단위면 $T_B$ 도 같은 단위. (ii) $\hat\sigma_X = 0$ 시 *전체 subband 0* 처리는 표시되지 않은 곳에서 묵시적 (Eq. 19 의 $\infty$ 동치). (iii) BayesShrink 는 *scale × orientation* (3J 개) subband-adaptive — SureShrink (J 개) 보다 finer.

### English
- **§I Introduction**: start with Fig. 1's observation that *thresholding ≈ zero-zone quantization*.
- **§II.A (GGD + Bayes risk)**: the core. Equations 5–7 set up; Eqs. 9–12 are the punchline. **Figs. 4–6** (the $T^*$ vs $T_B = 1/\sigma_X$ curves for various $\beta$) visualise *why* the simple formula works well — must read.
- **§II.B Parameter estimation**: MAD/0.6745 (from paper #1) plus a trivial variance subtraction. Don't forget the $\hat\sigma_X = 0$ case (entire subband zeroed).
- **§III MDL**: on first reading, just absorb the *insight* (zero-zone = thresholding). The exact code-length formulas can wait.
- **§IV Experiments**: Tables I-IV compare BayesShrink vs SureShrink vs OracleShrink — ~5–10% MSE differences.
- **Common stumbling blocks**: (i) units of $T_B = \sigma^2/\sigma_X$ — same as the coefficient units when $\sigma$ and $\sigma_X$ share units. (ii) When $\hat\sigma_X = 0$, the entire subband is zeroed — implicit in Eq. 19's $\infty$ case. (iii) BayesShrink is subband-adaptive over *scale × orientation* (3J subbands), finer than SureShrink's $J$ levels.

---

## 7. 현대적 의의 / Modern Significance

### 한국어
BayesShrink 는 *Bayesian wavelet shrinkage* 의 표준 baseline 이다. 후속의 BLS-GSM (Portilla+ 2003) 은 GGD 대신 Gaussian-scale-mixture prior 를 사용해 더 정교한 결과를 얻고, Bivariate shrinkage (Sendur-Selesnick 2002) 는 parent-child 의존성을 추가한다. BM3D (paper #7) 의 Wiener 단계는 본질적으로 Bayesian shrinkage — BayesShrink 의 후예. Curvelet/Contourlet/Shearlet (paper #5, #6, #8) 도 BayesShrink-style $\sigma^2/\sigma_X$ 임계값을 *방향성 subband* 에 그대로 적용 가능. MDL 부분은 JPEG2000 의 zero-tree wavelet 코딩과 사상적 연결이 있고, 현대 *learned image compression* (Ballé+ 2018) 의 rate-distortion + denoising joint training 의 직계 선조이다. Deep-learning denoiser 도 학습된 *subband-adaptive* shrinkage 를 내부에 포함한다.

### English
BayesShrink remains the standard baseline for *Bayesian wavelet shrinkage*. Successors include BLS-GSM (Portilla+ 2003, Gaussian-scale-mixture priors) and bivariate shrinkage (Sendur-Selesnick 2002, parent-child dependence). BM3D's (#7) Wiener stage is essentially Bayesian shrinkage, descended from BayesShrink. Curvelet/Contourlet/Shearlet (#5, #6, #8) all apply BayesShrink-style $\sigma^2/\sigma_X$ thresholds to directional subbands. The MDL part connects ideologically to JPEG2000's zero-tree wavelet coding and presages modern *learned image compression* (Ballé+ 2018), where rate-distortion training jointly compresses and denoises. Deep denoisers internalise learned subband-adaptive shrinkage as well.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
