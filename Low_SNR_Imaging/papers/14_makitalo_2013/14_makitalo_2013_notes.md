---
title: "Optimal Inversion of the Generalized Anscombe Transformation for Poisson-Gaussian Noise"
authors: Markku Mäkitalo, Alessandro Foi
year: 2013
journal: "IEEE Trans. Image Processing 22(1), pp. 91-103"
doi: "10.1109/TIP.2012.2202675"
topic: Low-SNR Imaging / Variance-Stabilising Transforms
tags: [generalized-anscombe-transform, gat, exact-unbiased-inverse, maximum-likelihood-inverse, poisson-gaussian, vst, bm3d, denoising-pipeline]
status: completed
date_started: 2026-05-06
date_completed: 2026-05-06
---

# 14. Optimal Inversion of the Generalized Anscombe Transformation for Poisson-Gaussian Noise / 포아송-가우시안 잡음을 위한 일반화 안스콤 변환의 최적 역변환

---

## 1. Core Contribution / 핵심 기여

### 한국어
Mäkitalo, Foi(2013)는 **VST + Gaussian denoise + inverse VST 파이프라인의 결정적 약점 — inverse VST 단계의 편향 — 을 해결**한다. 핵심 통찰: VST는 변환된 자료의 분산을 안정화하지만 *평균*에는 시스템적 편향을 도입한다. 이 편향은 단순한 algebraic inverse $\mathcal I^{-1}_{\rm alg}(D) = (D/2)^2 - 3/8 - \sigma^2$으로는 보정되지 않는다. 본 논문 핵심 기여 네 가지:
(i) **Exact Unbiased Inverse $\mathcal I_\sigma$ (Eq. 9, 10)**: 일반화 Anscombe 변환 $f_\sigma$의 정확한 역변환을 정의:
$$
\mathcal I_\sigma : E\{f_\sigma(z) | y, \sigma\} \mapsto E\{z | y, \sigma\} = y
$$
$E\{f_\sigma(z) | y, \sigma\}$는 forward Anscombe의 *조건부 기댓값*; 그 역사상이 정확한 unbiased inverse. Closed form 평가 불가 → $(y, \sigma)$ 격자에서 수치 적분 (Eq. 10) 후 LUT 사용.
(ii) **ML Interpretation (Theorem in §III-B, Eq. 14)**: $\mathcal I_\sigma$는 maximum likelihood inverse와 일치 (특정 합리적 가정 하에). 즉 “어떤 통계적 의미에서” 최적.
(iii) **Asymptotic & Closed-Form Approximations (Eq. 15-22)**: $\sigma$ 큰 경우 $\mathcal I_\sigma \approx \mathcal I_0 - \sigma^2$ (단순 보정). $D$ 큰 경우 5차 polynomial closed form (Eq. 21).
(iv) **Practical pipeline = GAT + BM3D + $\mathcal I_\sigma$**: BM3D를 가우시안 denoiser로 사용. Cameraman peak 1, σ = 0.1 (입력 PSNR 3.20 dB)에서:
- BM3D + algebraic inverse: 15.72 dB (실패)
- BM3D + asymptotic inverse $\mathcal I_{\rm asy}$: 15.55 dB (실패)
- **BM3D + $\mathcal I_\sigma$: 20.23 dB**
- UWT/BDCT PURE-LET (paper #13): 20.35 dB
- **저광자 영역에서 PURE-LET와 동등 또는 우월하며**, 더 정교한 가우시안 denoiser (BM3D)를 plug-and-play로 활용 가능.

### English
Mäkitalo and Foi (2013) **fix the critical weak point of the VST-based denoising pipeline: the bias introduced by the inverse VST stage**. The forward variance-stabilising transformation perfectly stabilises variance, but introduces a systematic bias in the *mean* that the algebraic inverse $\mathcal I^{-1}_{\rm alg}(D) = (D/2)^2 - 3/8 - \sigma^2$ does not undo correctly. Four key contributions:
(i) **Exact Unbiased Inverse $\mathcal I_\sigma$ (Eq. 9, 10)**: Define the exact inverse of the generalised Anscombe transform $f_\sigma$ as the inverse of the conditional expectation map:
$$
\mathcal I_\sigma : E\{f_\sigma(z) | y, \sigma\} \mapsto E\{z | y, \sigma\} = y.
$$
Computed by tabulating $E\{f_\sigma(z) | y, \sigma\}$ at a grid of $(y, \sigma)$ values via numerical quadrature (Eq. 10), then interpolating.
(ii) **Maximum-Likelihood Interpretation (Eq. 14)**: Under reasonable assumptions about the denoiser's residual error, $\mathcal I_\sigma$ coincides with the ML inverse.
(iii) **Asymptotic & Closed-Form Approximations (Eq. 15-22)**: For large $\sigma$, $\mathcal I_\sigma \approx \mathcal I_0 - \sigma^2$. For large $D$, a 5-term closed form (Eq. 21) is accurate to $L^\infty$ error 0.0468.
(iv) **Practical pipeline GAT + BM3D + $\mathcal I_\sigma$**: At Cameraman peak 1, σ = 0.1 (input PSNR 3.20 dB):
- BM3D + algebraic inverse: 15.72 dB (failure)
- BM3D + asymptotic inverse $\mathcal I_{\rm asy}$: 15.55 dB (failure)
- **BM3D + $\mathcal I_\sigma$: 20.23 dB**
- UWT/BDCT PURE-LET: 20.35 dB
- **At very low photon counts, BM3D + $\mathcal I_\sigma$ matches or beats PURE-LET**, while leveraging plug-and-play state-of-the-art Gaussian denoisers.

---

## 2. Reading Notes / 읽기 노트

### Part I: §I-II Setting and the GAT / 설정과 일반화 안스콤

#### 한국어
- **잡음 모델 (Eq. 1)**: $\check z_i = \alpha p_i + \check n_i$, $p_i \sim \mathcal P(y_i)$, $\check n_i \sim N(\mu, \check\sigma^2)$. 광자수 $\alpha$배 후 read-noise 더해짐.
- **Affine reduction (Eq. 4-5)**: $z = (\check z - \mu)/\alpha$, $\sigma = \check\sigma/\alpha$ 변수 치환으로 $z = p + n$, $p \sim \mathcal P(y)$, $n \sim N(0, \sigma^2)$로 단순화. 이후 모든 분석은 단순 모델에서.
- **Generalised Anscombe Transform (GAT, Eq. 7)**:
  $$
  f_\sigma(z) = \begin{cases} 2\sqrt{z + \tfrac{3}{8} + \sigma^2}, & z > -\tfrac{3}{8} - \sigma^2 \\ 0, & \text{otherwise} \end{cases}
  $$
  - 순 포아송 ($\sigma = 0$)에서 표준 Anscombe $2\sqrt{z + 3/8}$로 환원.
  - 분산 안정화 효과 (Fig. 1b): $\sigma$ 큰 영역에서 정상 (variance ≈ 1), $\sigma$ 작은 영역에서 *언더슈트* ($y \to 0$ 근방).

#### English
- Mixed-noise model (Eq. 1): clipped/scaled Poisson + read-noise CCD model.
- Affine reduction (Eq. 4-5) absorbs scale $\alpha$ and Gaussian mean $\mu$ into a unit-scale model $z = p + n$, $p \sim \mathcal P(y)$, $n \sim N(0, \sigma^2)$.
- GAT (Eq. 7) is the standard variance-stabilising transformation; it reduces to the classical Anscombe transform in the pure-Poisson limit $\sigma = 0$.
- Variance is stabilised approximately to 1, but undershoots significantly at low $y$ when $\sigma$ is small.

---

### Part II: §III-A Exact Unbiased Inverse / 정확한 비편향 역변환

#### 한국어
- **개념**: 잡음 제거 알고리즘이 GAT의 *기댓값* $E\{f_\sigma(z) | y, \sigma\}$를 정확히 추정한다고 가정 (이상적 denoiser). 그러면 우리가 알아야 할 것은:
  $$
  \mathcal I_\sigma: E\{f_\sigma(z) | y, \sigma\} \longmapsto y
  $$
  - 이는 *선형 algebraic inverse가 아닌* 함수 — 비선형 $f_\sigma$와 비선형 기댓값 연산자가 합성된 결과.
- **계산** (Eq. 10):
  $$
  E\{f_\sigma(z) | y, \sigma\} = \int_{-\infty}^{+\infty} f_\sigma(z) \cdot p(z | y, \sigma)\, dz = \int 2\sqrt{z + \tfrac{3}{8} + \sigma^2} \sum_{k=0}^\infty \tfrac{y^k e^{-y}}{k!} \cdot \tfrac{1}{\sqrt{2\pi\sigma^2}} e^{-(z-k)^2/(2\sigma^2)} dz
  $$
  - 격자 $96 \times 1199$ ($\sigma \in [0.01, 50]$, $y \in \{0, ..., 200\}$)에서 수치 적분 → LUT.
  - 실행시간 영향 미미 (5.4 s BM3D vs 0.2 s exact inverse on Lena 512×512).
- **Algebraic inverse vs exact inverse 차이 (Fig. 1c)**:
  - Algebraic $f_\sigma^{-1}(D) = (D/2)^2 - 3/8 - \sigma^2$ — $\sigma = 1$에서 $D = 0$으로 가면 $\hat y \to -1.375$ (음수, 비현실적).
  - Exact $\mathcal I_\sigma$ — $\sigma = 1$에서 $D = 0$으로 가면 $\hat y \to $ 적정한 양의 값 (LUT으로부터).

#### English
- The exact inverse $\mathcal I_\sigma$ maps the *expectation* $E\{f_\sigma(z) | y, \sigma\}$ back to the underlying $y$. It is *not* the algebraic inverse, because the forward transform and the expectation operator do not commute.
- Eq. (10) gives the integral defining $E\{f_\sigma(z) | y, \sigma\}$ — a Poisson-weighted Gaussian convolution of $f_\sigma$. Numerically evaluated on a 96 × 1199 grid ($\sigma \in [0.01, 50]$, $y \in [0, 200]$) and tabulated.
- Fig. 1(c) shows the algebraic inverse can produce *negative* outputs at low $D$, while the exact inverse remains physically reasonable.

---

### Part III: §III-B-D Optimality and Asymptotic Behaviour / 최적성과 점근 거동

#### 한국어
- **ML Interpretation (Theorem in Appendix B)**: 가정 — denoiser의 잔차 $\xi = D - E\{f_\sigma(z) | y, \sigma\}$가 0 모드의 단봉(unimodal) 분포를 따른다고 하면, ML inverse는:
  $$
  \mathcal I_{\rm ML}(D) = \begin{cases} \mathcal I_\sigma(D), & D \ge E\{f_\sigma(z) | 0, \sigma\} \\ 0, & D < E\{f_\sigma(z) | 0, \sigma\} \end{cases}
  $$
  - 즉 본 논문의 $\mathcal I_\sigma$가 ML inverse와 일치 (단, 매우 작은 $D$에서 0으로 클리핑).
- **Asymptotic behaviour (Eq. 15-18)**:
  - $\sigma \to \infty$: $\mathcal I_\sigma \approx \mathcal I_0 - \sigma^2$ (Eq. 15) — 단순 보정.
  - $D \to \infty$ (fixed $\sigma$): $y = \mathcal I_\sigma(D) = \mathcal I_0(D) - \sigma^2 + O(D^{-4})$ (Eq. 16).
  - $\sigma \to \infty$ (fixed $y$): $y = \mathcal I_0(D) - \sigma^2 + O(\sigma^{-2})$ (Eq. 17).
  - $\sigma \to 0$ (fixed $y$): $y = \mathcal I_0(D) - \sigma^2 + O(\sigma^2)$ (Eq. 18).

- **Closed-form approximation (Eq. 21)**: $\mathcal I_0$의 closed form (Mäkitalo-Foi 2011)을 사용:
  $$
  \widetilde{\mathcal I_\sigma}(D) = \tfrac{1}{4} D^2 + \tfrac{1}{4}\sqrt{\tfrac{3}{2}} D^{-1} - \tfrac{11}{8} D^{-2} + \tfrac{5}{8}\sqrt{\tfrac{3}{2}} D^{-3} - \tfrac{1}{8} - \sigma^2
  $$
  - $L^2$ error 0.0069, $L^\infty$ error 0.0468 (Eq. 23-24) — 매우 정확.

#### English
- ML interpretation (proof in Appendix B): under the assumption that the denoiser residual is unimodal-around-zero, $\mathcal I_\sigma$ coincides with the ML inverse. Hence "optimal" in a precise statistical sense.
- Asymptotic behaviour: $\mathcal I_\sigma \to \mathcal I_0 - \sigma^2$ as either $D \to \infty$, $\sigma \to \infty$, or $\sigma \to 0$. The error in this asymptotic form is:
  - $O(D^{-4})$ for fixed $\sigma$, large $D$.
  - $O(\sigma^{-2})$ for fixed $y$, large $\sigma$.
  - $O(\sigma^2)$ for fixed $y$, small $\sigma$.
- Closed-form approximation (Eq. 21) extends the Mäkitalo-Foi 2011 closed form for pure Poisson by subtracting $\sigma^2$. Error metrics: $L^2 = 0.0069$, $L^\infty = 0.0468$ — within image-processing tolerance.

---

### Part IV: §IV Experiments / 실험

#### 한국어
- **Setup**: Cameraman 256×256, Fluorescent Cells 512×512, Lena 512×512.
- 실험 그리드: peak $\in \{1, 2, 5, 10, 20, 30, 60, 120\}$, $\sigma = $ peak/10 ($\sigma \in \{0.1, 0.2, 0.5, 1, 2, 3, 6, 12\}$).
- 비교 알고리즘: GAT + Gaussian denoiser (BM3D 또는 BLS-GSM) + 세 inverse 중 하나 ($\mathcal I_\sigma$, $\mathcal I_{\rm asy}$, $\mathcal I_{\rm alg}$). UWT/BDCT PURE-LET (paper #13)도 비교.
- **결과 (Table I, PSNR; Cameraman 256×256)**:
  - peak 1, σ = 0.1 (input 3.20 dB):
    - GAT + BM3D + $\mathcal I_\sigma$: **20.23**
    - GAT + BM3D + $\mathcal I_{\rm asy}$: 15.55
    - GAT + BM3D + $\mathcal I_{\rm alg}$: 15.72
    - GAT + BLS-GSM + $\mathcal I_\sigma$: 18.46
    - UWT/BDCT PURE-LET: 20.35
  - peak 10, σ = 1 (input 12.45 dB):
    - GAT + BM3D + $\mathcal I_\sigma$: **25.52**
    - GAT + BM3D + $\mathcal I_{\rm asy}$: 25.52
    - GAT + BM3D + $\mathcal I_{\rm alg}$: 24.80
    - PURE-LET: 24.68
  - peak 30, σ = 3 (input 15.91 dB):
    - GAT + BM3D + $\mathcal I_\sigma$: **27.30**
    - PURE-LET: 26.51
- **Cameraman peak 1 (Fig. 9)**: $\mathcal I_{\rm asy}$와 $\mathcal I_{\rm alg}$는 영상이 *완전히 어둡게 죽음* (PSNR 15-16 dB), $\mathcal I_\sigma$는 정상 복원 (20.23 dB). 시각적으로 4-5 dB 격차는 “보이거나 보이지 않거나”의 차이.
- **Robustness to estimated parameters (Table III)**: 잡음 추정 $(\alpha_{\rm est}, \sigma_{\rm est})$을 사용해도 결과가 거의 동일 (Cameraman peak 1: 20.34 dB) — 실 application에서도 견고.
- **시간 (Lena 512×512)**:
  - GAT: 0.005 s
  - BM3D: 5.4 s
  - Exact inverse $\mathcal I_\sigma$ (LUT): 0.2 s
  - 합계: $\sim 5.6$ s, vs UWT/BDCT PURE-LET 42 s.

#### English
- Test images: Cameraman 256×256, Fluorescent Cells 512×512, Lena 512×512. Peak intensities 1-120, with $\sigma = \mathrm{peak}/10$.
- Compared GAT + {BM3D, BLS-GSM} + {$\mathcal I_\sigma$, $\mathcal I_{\rm asy}$, $\mathcal I_{\rm alg}$} vs UWT/BDCT PURE-LET.
- At peak 1, σ = 0.1 (Cameraman): $\mathcal I_{\rm asy}$ and $\mathcal I_{\rm alg}$ give 15-16 dB (failure), but $\mathcal I_\sigma$ gives 20.23 dB — comparable to PURE-LET 20.35 dB.
- The improvement persists across all tested noise levels.
- Robust under parameter mis-estimation.
- Computational overhead of $\mathcal I_\sigma$ is negligible ($\sim 0.2$ s for Lena 512×512).

---

### Part V: §V Conclusion / 결론

#### 한국어
- 본 논문은 GAT 기반 잡음 제거 파이프라인의 *결정적 약점*인 inverse 단계를 해결.
- 핵심 통찰: 분산 안정화는 좋지만, *평균 편향*은 별도로 보정해야. Algebraic inverse는 분산만 안정화 가정 하에 설계됨 — 평균 편향은 무시.
- $\mathcal I_\sigma$의 LUT 구현은 모든 $(y, \sigma)$ 영역에서 작동, 컴퓨팅 오버헤드 미미.
- 시사점: 잡음별 전용 알고리즘 (PURE-LET, NLM 등) 설계 대신, **AWGN denoiser + proper VST + exact inverse** 조합이 동등 또는 우월한 성능 — 알고리즘 설계의 지향점을 바꾼다.

#### English
- This paper closes the long-standing weakness of VST-based pipelines: bias from the inverse VST.
- Key insight: variance is stabilised by the forward transform but the mean is biased; the algebraic inverse only undoes variance-stabilisation, not the mean shift.
- The exact unbiased inverse, computed once via numerical quadrature and stored in a LUT, has negligible runtime overhead.
- Implication for the field: instead of designing noise-specific denoisers (PURE-LET, Poisson NLM), one can use **AWGN denoiser + proper VST + exact inverse** to match or beat them — the locus of innovation shifts to AWGN denoising.

---

## 3. Key Takeaways / 핵심 시사점

1. **Algebraic inverse는 *시간순서가 잘못된* 역변환 / The algebraic inverse is "out of order"** — 우리가 가진 것은 *denoised* $D \approx E\{f_\sigma(z) | y, \sigma\}$이지 $f_\sigma(z)$ 자체가 아니다. 비선형 $f_\sigma$와 기댓값 연산자가 *교환 불가*이므로, $f_\sigma^{-1}(E\{f_\sigma(z)\}) \ne E\{z\}$. 이 commutativity 실패가 paper의 출발점.
   The algebraic inverse uses $f_\sigma^{-1}$, but what we recover is the *expectation* of $f_\sigma(z)$ — not $f_\sigma$ itself. Because $f_\sigma$ is nonlinear, $f_\sigma^{-1}(E\{f_\sigma(z)\}) \ne E\{z\}$. This failure of commutativity is the central conceptual point.

2. **Exact unbiased inverse는 LUT으로 구현 가능 / Exact inverse via lookup table** — Eq. (10)의 적분을 $(y, \sigma)$ 격자에서 한 번 계산해 저장. 추론 시 interpolation. 이는 paper 일자에 96 × 1199 격자 ($\sim 100k$ 항목) — 메모리·시간 부담 없음.
   The exact inverse is computed once via numerical quadrature on a ($y, \sigma$) grid and stored in a lookup table. Approximately 100k entries — trivial in memory and runtime.

3. **$\mathcal I_\sigma$는 ML inverse와 일치 / The exact unbiased inverse coincides with the ML inverse** — Appendix B Theorem (Eq. 14). 따라서 “정확하지만 임의의” 선택이 아니라, 통계적 의미에서 *최적*. 어떤 다른 inverse보다 (적절한 가정 하에) 더 정확한 $y$ 추정.
   Under reasonable assumptions about the denoiser residual distribution, $\mathcal I_\sigma$ is the maximum-likelihood inverse — making it not just empirically good but provably optimal.

4. **저광자 영역에서 $\mathcal I_\sigma$는 critical 차이를 만든다 / The exact inverse makes the difference at very low photon counts** — Cameraman peak 1: $\mathcal I_{\rm asy}$ 15.55 dB vs $\mathcal I_\sigma$ 20.23 dB. 4.7 dB 격차 = 시각적 “쓸모없음” vs “쓸만함.” 매우 어두운 형광현미경 영상에서 결정적.
   At peak intensity 1, the exact inverse gives 20.23 dB while the asymptotic gives only 15.55 dB — a 4.7 dB gap that visually distinguishes "unusable" from "useful". Critical for very-low-light fluorescence microscopy.

5. **VST + BM3D + $\mathcal I_\sigma$는 PURE-LET와 동등 / VST + BM3D + $\mathcal I_\sigma$ matches PURE-LET** — Cameraman peak 1: 20.23 vs 20.35 dB; peak 10: 25.52 vs 24.68 dB. 즉 paper #13 (PURE-LET)의 핵심 결과 — “VST를 우회하는 것이 우월” — 가 부분적으로 *반박됨*. 단순한 VST + 좋은 가우시안 denoiser가 동등.
   The pipeline VST + BM3D + $\mathcal I_\sigma$ matches or beats PURE-LET (paper #13) at all tested noise levels, refuting the implicit claim of paper #13 that bypassing the VST is necessary for top performance.

6. **알고리즘 설계의 지향점 변화 / Implication: design AWGN denoisers, not noise-specific ones** — 포아송용 알고리즘 (PURE-LET, NLM)을 따로 설계할 필요 없이, 잘 만든 가우시안 denoiser (BM3D, FFDNet, Restormer)을 plug-and-play로 사용 가능. 잡음 모델은 VST + exact inverse가 처리.
   For a developer, this paper's lesson is: instead of building noise-specific denoisers, build excellent AWGN denoisers and combine them with VST + exact inverse. This shifts the entire field's design focus.

7. **Closed-form approximation (Eq. 21)도 충분히 정확 / The closed-form is also accurate** — $L^2$ error 0.0069, $L^\infty$ 0.0468 (very small). LUT이 부담스러운 시나리오 (예: 매우 큰 $\sigma$ 범위)에서도 5-항 다항식만으로 $\mathcal I_\sigma$ 근사 가능.
   The closed-form approximation (Eq. 21) achieves $L^\infty$ error of 0.0468 — sufficient for most image-processing applications even without the LUT.

8. **잡음 파라미터 추정에 견고함 / Robust to estimated noise parameters** — Table III: $\alpha_{\rm est}, \sigma_{\rm est}$을 raw image에서 자동 추정해 사용해도 PSNR/SSIM이 거의 변화 없음. Real-world에서 noise parameter를 정확히 알 수 없으므로 매우 중요.
   Table III shows that using estimated $\alpha, \sigma$ from the noisy image (rather than ground-truth values) results in negligible PSNR loss. This is essential for real-world deployment.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Noise model / 잡음 모델
$$
\check z_i = \alpha p_i + \check n_i, \quad p_i \sim \mathcal P(y_i), \quad \check n_i \sim N(\mu, \check\sigma^2)
$$
After affine reduction (Eq. 4-5): $z_i = p_i + n_i$, $p_i \sim \mathcal P(y_i)$, $n_i \sim N(0, \sigma^2)$.

### 4.2 Generalised Anscombe Transformation (Eq. 7) / 일반화 안스콤
$$
\boxed{\;f_\sigma(z) = \begin{cases} 2\sqrt{z + \tfrac{3}{8} + \sigma^2}, & z > -\tfrac{3}{8} - \sigma^2 \\ 0, & \text{otherwise} \end{cases}\;}
$$
Reduces to standard Anscombe $f_0(z) = 2\sqrt{z + 3/8}$ when $\sigma = 0$.

### 4.3 Conditional expectation (Eq. 10) / 조건부 기댓값
$$
E\{f_\sigma(z) | y, \sigma\} = \int_{-\infty}^{+\infty} 2\sqrt{z + \tfrac{3}{8} + \sigma^2}\,p(z|y,\sigma)\,dz
$$
$$
= \int_{-\infty}^{+\infty} 2\sqrt{z + \tfrac{3}{8} + \sigma^2} \sum_{k=0}^\infty \frac{y^k e^{-y}}{k!} \cdot \frac{e^{-(z-k)^2/(2\sigma^2)}}{\sqrt{2\pi\sigma^2}} dz
$$
### 4.4 Exact unbiased inverse (Eq. 9) / 정확한 비편향 역변환
$$
\boxed{\;\mathcal I_\sigma : E\{f_\sigma(z) | y, \sigma\} \longmapsto y\;}
$$
Computed by tabulating $E\{f_\sigma(z) | y, \sigma\}$ on a 96 × 1199 grid and inverting numerically.

### 4.5 ML interpretation (Eq. 14) / ML 해석
Under the assumption $\xi = D - E\{f_\sigma(z) | y, \sigma\} \sim U_0$ (unimodal, mode at 0):
$$
\mathcal I_{\rm ML}(D) = \begin{cases} \mathcal I_\sigma(D), & D \ge E\{f_\sigma(z) | 0, \sigma\} \\ 0, & D < E\{f_\sigma(z) | 0, \sigma\} \end{cases}
$$
### 4.6 Asymptotic forms (Eq. 15-18) / 점근 형태
$$
\mathcal I_\sigma(D) \approx \mathcal I_0(D) - \sigma^2 \quad (D \to \infty, \sigma \to \infty, \sigma \to 0)
$$
Errors:
- $O(D^{-4})$ as $D \to \infty$ for fixed $\sigma$ (Eq. 16).
- $O(\sigma^{-2})$ as $\sigma \to \infty$ for fixed $y$ (Eq. 17).
- $O(\sigma^2)$ as $\sigma \to 0$ for fixed $y$ (Eq. 18).

### 4.7 Closed-form approximation (Eq. 21) / 닫힌형 근사
$$
\widetilde{\mathcal I_\sigma}(D) = \tfrac{1}{4}D^2 + \tfrac{1}{4}\sqrt{\tfrac{3}{2}}\,D^{-1} - \tfrac{11}{8}\,D^{-2} + \tfrac{5}{8}\sqrt{\tfrac{3}{2}}\,D^{-3} - \tfrac{1}{8} - \sigma^2
$$
Error metrics (Eq. 23-24):
- $\|\widetilde{\mathcal I_\sigma}(E\{f_\sigma(z)|y,\sigma\}) - y\|_2 / \mathrm{std}(z|y,\sigma) = 0.0069$
- $\|\widetilde{\mathcal I_\sigma}(\cdot) - y\|_\infty = 0.0468$

### 4.8 Algebraic inverse (for comparison) / 대수적 역변환 (비교용)
$$
\mathcal I_{\rm alg}(D) = \tfrac{1}{4} D^2 - \tfrac{3}{8} - \sigma^2
$$
$$
\mathcal I_{\rm asy}(D) = \tfrac{1}{4} D^2 - \tfrac{1}{8} - \sigma^2 \quad \text{(asymptotically unbiased)}
$$
### 4.9 Worked numerical example / 수치 예시
For a Cameraman pixel with true $y = 1$, $\sigma = 0.1$:
- Forward GAT: $f_\sigma(z) \approx 2\sqrt{1 + 0.375 + 0.01} = 2\sqrt{1.385} \approx 2.354$.
- After (perfect) Gaussian denoising of patch around this pixel: $D \approx E\{f_\sigma(z) | 1, 0.1\}$. Numerically (paper Eq. 10): $D \approx 2.298$.
- Algebraic inverse: $\mathcal I_{\rm alg}(D) = (2.298/2)^2 - 3/8 - 0.01 = 1.320 - 0.385 = 0.935$. True $y = 1$, bias $\approx -0.065$.
- Exact unbiased inverse via LUT: $\mathcal I_\sigma(D) \approx 1.000$. True $y = 1$, bias $\approx 0$.

For peak 1, σ = 0.1, Cameraman 256×256:
- Algebraic + BM3D: 15.72 dB.
- Asymptotic + BM3D: 15.55 dB.
- Exact + BM3D: **20.23 dB** — a 4.7 dB improvement from getting the inverse right.

### 4.10 Pipeline / 전체 파이프라인
```
noisy image y_check (Poisson-Gaussian)
        |
        v
[1] Estimate (alpha, sigma) via Foi+ 2008  
        |
        v
[2] Affine reduction: z = (y_check - mu)/alpha
        |
        v
[3] Forward GAT: D_in = f_sigma(z)
        |
        v
[4] Gaussian denoiser (BM3D, BLS-GSM, etc.) -> D_out
        |
        v
[5] Exact unbiased inverse: y_hat = I_sigma(D_out) via LUT
        |
        v
[6] Restore scale: y_check_hat = alpha * y_hat + mu
```

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1948 ─── Anscombe — Poisson VST (paper #11)
                    ↳ algebraic inverse implicitly assumed
1995 ─── Murtagh-Starck-Bijaoui — generalised Anscombe transform (GAT)
                    ↳ extends to Poisson-Gaussian
2007 ─── Dabov-Foi-Katkovnik-Egiazarian — BM3D
                    ↳ best AWGN denoiser of its time
2008 ─── Foi-Trimeche-Katkovnik-Egiazarian — Poisson-Gaussian fitting
                    ↳ enables automatic (alpha, sigma) estimation
2010 ─── Deledalle-Tupin-Denis — Poisson NL Means (paper #12)
                    ↳ bypasses VST via NLM + PURE
2011 ─── Luisier-Blu-Unser — PURE-LET (paper #13)
                    ↳ bypasses VST via wavelet PURE
2011 ─── Mäkitalo-Foi — exact unbiased inverse for pure POISSON Anscombe
                    ↳ direct ancestor of this paper
2013 ★★ MAKITALO-FOI: Optimal Inverse of Generalised Anscombe (this paper)
                    ↳ exact unbiased inverse for Poisson-GAUSSIAN GAT
                    ↳ BM3D + GAT becomes state of the art
2017 ─── DnCNN, FFDNet, Restormer — deep AWGN denoisers
                    ↳ paired with VST for Poisson-Gaussian denoising
2018+ ── Self-supervised denoisers (Noise2Noise, Noise2Self)
                    ↳ Anscombe still relevant for low-photon settings.
```

이 논문은 **VST 파이프라인의 부활**이자, **잡음별 알고리즘 설계의 종언**을 시사한다. 이후 deep AWGN denoiser들이 발전하면서, “VST + DnCNN + exact inverse”가 표준 파이프라인이 됨.

This paper is **the resurrection of the VST pipeline** and signals the *end of noise-specific algorithm design*. As deep AWGN denoisers matured, "VST + DnCNN + exact inverse" became the standard pipeline for low-light imaging.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **Anscombe (1948)** *Biometrika* (#11 in topic) | Forward VST | Foundation; this paper closes the inverse-side gap. |
| **Murtagh-Starck-Bijaoui (1995)** *A&A Suppl.* | Generalised Anscombe transform | Forward GAT used here. |
| **Mäkitalo-Foi (2011)** *IEEE TIP* | Exact unbiased inverse for pure Poisson | Direct precursor; this paper extends to Poisson-Gaussian. |
| **Dabov+ (2007)** *IEEE TIP* (#7) | BM3D | Used as AWGN denoiser in pipeline; main paired algorithm. |
| **Portilla-Strela-Wainwright-Simoncelli (2003)** *IEEE TIP* | BLS-GSM | Alternative AWGN denoiser (paired in Tables I-II). |
| **Foi+ (2008)** *IEEE TIP* | Poisson-Gaussian noise fitting | Used for automatic $\alpha, \sigma$ estimation. |
| **Luisier-Blu-Unser (2011)** *IEEE TIP* (#13 in topic) | PURE-LET | Direct competitor; this paper matches/beats it via VST + BM3D. |
| **Deledalle-Tupin-Denis (2010)** *ICIP* (#12) | Poisson NL Means | Alternative direct-Poisson approach. |
| **Donoho-Johnstone (1995)** *JASA* (#2) | SureShrink | Combined with Anscombe gives a baseline (Haar-Fisz). |
| **Fryzlewicz-Nason (2004)** *J. Comput. Graph. Stat.* | Haar-Fisz multiscale Anscombe | Multiscale variant; generalised by this paper. |
| **Stein (1981)** *Annals of Statistics* | SURE | Unbiased risk concept; complementary to unbiased *inverse*. |

---

## 7. References / 참고문헌

- Mäkitalo, M., & Foi, A., "Optimal inversion of the generalized Anscombe transformation for Poisson-Gaussian noise", *IEEE Trans. Image Process.*, 22(1), 91–103 (2013). [DOI: 10.1109/TIP.2012.2202675]
- Mäkitalo, M., & Foi, A., "Optimal inversion of the Anscombe transformation in low-count Poisson image denoising", *IEEE Trans. Image Process.*, 20(1), 99–109 (2011).
- Mäkitalo, M., & Foi, A., "A closed-form approximation of the exact unbiased inverse of the Anscombe variance-stabilizing transformation", *IEEE Trans. Image Process.*, 20(9), 2697–2698 (2011).
- Anscombe, F. J., "The transformation of Poisson, binomial and negative-binomial data", *Biometrika*, 35(3-4), 246–254 (1948).
- Murtagh, F., Starck, J.-L., & Bijaoui, A., "Image restoration with noise suppression using a multiresolution support", *Astron. Astrophys. Suppl.*, 112, 179–189 (1995).
- Dabov, K., Foi, A., Katkovnik, V., & Egiazarian, K., "Image denoising by sparse 3D transform-domain collaborative filtering", *IEEE Trans. Image Process.*, 16(8), 2080–2095 (2007).
- Portilla, J., Strela, V., Wainwright, M. J., & Simoncelli, E. P., "Image denoising using scale mixtures of Gaussians in the wavelet domain", *IEEE Trans. Image Process.*, 12(11), 1338–1351 (2003).
- Foi, A., Trimeche, M., Katkovnik, V., & Egiazarian, K., "Practical Poissonian-Gaussian noise modeling and fitting for single-image raw data", *IEEE Trans. Image Process.*, 17(10), 1737–1754 (2008).
- Luisier, F., Blu, T., & Unser, M., "Image denoising in mixed Poisson-Gaussian noise", *IEEE Trans. Image Process.*, 20(3), 696–708 (2011).
- Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P., "Image quality assessment: From error visibility to structural similarity", *IEEE Trans. Image Process.*, 13(4), 600–612 (2004).
