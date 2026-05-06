---
title: "Adaptive Wavelet Thresholding for Image Denoising and Compression"
authors: S. Grace Chang, Bin Yu, Martin Vetterli
year: 2000
journal: "IEEE Transactions on Image Processing 9(9), pp. 1532–1546"
doi: "10.1109/83.862633"
topic: Low-SNR Imaging / Wavelet Denoising
tags: [wavelet, bayesshrink, generalized-gaussian, ggd, bayesian-threshold, subband-adaptive, image-denoising, image-compression, mdl, simultaneous-coding-denoising]
status: completed
date_started: 2026-05-06
date_completed: 2026-05-06
---

# 3. Adaptive Wavelet Thresholding for Image Denoising and Compression / 영상 노이즈 제거와 압축을 위한 적응적 웨이블릿 임계화

---

## 1. Core Contribution / 핵심 기여

### 한국어
이 논문은 두 가지 기여를 한다.

(A) **BayesShrink 임계기**: 자연 영상의 웨이블릿 계수가 **일반화 가우시안 분포(GGD)** $\propto \exp\{-(\alpha|x|)^\beta\}$를 따른다는 경험적 사실을 *prior*로 두고, 베이즈 위험 $E_{X,Y}(\hat X - X)^2$을 최소화하는 soft-threshold를 유도. 풀어 보면 거의 **닫힌 형태**의 경험적 임계값
$$
\boxed{\;T_B(\sigma_X) \;=\; \frac{\sigma^2}{\sigma_X}\;}
$$
을 얻는다. 여기서 $\sigma^2$은 잡음 분산, $\sigma_X^2$은 *해당 subband*의 신호 분산. 이는 모든 GGD 파라미터 $\beta \in [0.5, 4]$에 대해 최적 위험의 **5% 이내**. **subband-adaptive**: 각 $k$-스케일 $HH_k, HL_k, LH_k$에 따로 $\sigma_X$ 추정.

(B) **MDL 기반 동시 denoising + compression**: 임계화의 zero-zone과 양자화의 dead-zone이 일치한다는 통찰에서 출발해, Rissanen MDL 원리로 quantization step $\Delta$, bin 개수 $m$, zero-zone 폭을 동시 결정. 즉 *압축 자체*가 denoising 효과를 가짐.

VisuShrink는 너무 큰 universal threshold($M = 512^2 = 2.6 \times 10^5$에서 $\sqrt{2\log M} \approx 5.1$)로 *과도하게 매끄럽게* 만들고, SureShrink는 SURE 최소화 비용이 큼. BayesShrink는 둘 사이를 메운다 — *parameter-free 데이터 의존* 임계값 + *closed form* + 대부분의 경우 SureShrink보다 우수 (8% MSE 개선이 일반적).

### English
Two contributions:

(A) **BayesShrink threshold**: Models wavelet coefficients with the **Generalized Gaussian Distribution (GGD)** prior $\propto \exp\{-(\alpha|x|)^\beta\}$, minimises the Bayes risk $E_{X,Y}(\hat X - X)^2$ over soft thresholds, and obtains the near-optimal **closed-form threshold**
$$
T_B(\sigma_X) = \sigma^2/\sigma_X
$$
where $\sigma^2$ is the noise variance and $\sigma_X^2$ the signal variance in *each subband*. It is within **5% of the optimal Bayes risk** over all GGD shapes $\beta \in [0.5, 4]$. Subband-adaptive: $\sigma_X$ estimated per $(k, \mathrm{orientation})$ subband.

(B) **MDL-based simultaneous denoising + compression**: Coupling the thresholding zero-zone with the quantizer dead-zone yields a coder whose Rissanen-MDL-optimal parameters $(T, \Delta, m)$ jointly perform denoising and compression. Lossy compression *is* denoising in this view.

The threshold $T_B = \sigma^2/\sigma_X$ is appealing: when SNR is high ($\sigma/\sigma_X \ll 1$) the threshold is small (preserve most signal); when SNR is low ($\sigma/\sigma_X \gg 1$) the threshold is large (kill the noise). VisuShrink's $\sigma\sqrt{2\log M}$ ignores $\sigma_X$ and over-smooths; SureShrink uses minimisation per level but is more expensive. BayesShrink bridges them: data-driven, closed-form, parameter-free, typically 8% better MSE than SureShrink.

---

## 2. Reading Notes / 읽기 노트

### Part I: §I Introduction / 서론

#### 한국어
- 영상 $g_{ij} = f_{ij} + \varepsilon_{ij}$, $\varepsilon \overset{iid}{\sim} N(0, \sigma^2)$. MSE $N^{-2}\sum(\hat f_{ij} - f_{ij})^2$ 최소화 목표.
- 핵심 통찰 (Fig. 1): 임계화 함수 $\eta_T(x)$와 zero-zone을 가진 양자화기는 *근사적으로 동일*. → 압축이 denoising을 한다.
- 비교 대상: VisuShrink (paper #1), SureShrink (paper #2).
- 본 논문 메서드: BayesShrink (분석적·subband-adaptive), 그리고 MDL-based 동시 압축+denoising.

#### English
Image model $g = f + \varepsilon$; minimise MSE. Insight from Fig. 1: thresholding ≈ quantization with zero-zone. Compares against VisuShrink and SureShrink.

---

### Part II: §II Wavelet Thresholding and Threshold Selection / 웨이블릿 임계화와 임계값 선택

#### 한국어 — §II.A GGD prior

자연 영상의 각 wavelet subband의 계수는 *일반화 가우시안 분포*로 잘 근사:
$$
GG_{\sigma_X, \beta}(x) = C(\sigma_X, \beta)\exp\bigl\{-(\alpha(\sigma_X, \beta)|x|)^\beta\bigr\} \quad (5)
$$
$$
\alpha(\sigma_X, \beta) = \sigma_X^{-1}\sqrt{\Gamma(3/\beta)/\Gamma(1/\beta)}, \quad C(\sigma_X, \beta) = \frac{\beta\alpha(\sigma_X, \beta)}{2\Gamma(1/\beta)}
$$
- $\beta = 2$: Gaussian.
- $\beta = 1$: Laplacian.
- 자연 영상의 wavelet 히스토그램은 $\beta \in [0.5, 1]$ 범위 (Fig. 3): peaked at zero with heavy tails — sparse-friendly.

**경험적 사실**: Fig. 3에서 4개 영상 (goldhill, lena, barbara, baboon)의 모든 subband를 GGD로 fit, R²>0.95.

#### English — §II.A GGD prior
Wavelet subband coefficients follow GGD with shape $\beta \in [0.5, 1]$ — peaked, heavy-tailed. Provides the prior for Bayesian threshold derivation.

#### 한국어 — §II.A Bayesian threshold derivation

**Bayes risk** for soft thresholding (Eq. 6):
$$
r(T) = E_X E_{Y|X}(\hat X - X)^2 = E_X E_{Y|X}(\eta_T(Y) - X)^2
$$
where $Y|X \sim N(X, \sigma^2)$ and $X \sim GG_{\sigma_X, \beta}$. Find $T^*(\sigma_X, \beta) = \arg\min r(T)$ (Eq. 7).

**Special case $\beta = 2$ (Gaussian prior)**:
$$
r(T) = \sigma^2 w\left(\frac{\sigma_X^2}{\sigma^2}, \frac{T}{\sigma}\right) \quad (9)
$$
$w$는 Gaussian phi/Phi 함수의 명시적 식 (Eq. 10). 수치 최소화로 $T^*$ 도출 → Fig. 4 (a) 곡선이 다음 단순 식과 거의 일치:
$$
T_B(\sigma_X) = 1/\sigma_X \quad (\sigma = 1, \quad \beta = 2 \text{ approximation, Eq. 11})
$$
스케일링하면:
$$
\boxed{\;T_B(\sigma_X) = \sigma^2/\sigma_X \quad (\text{Eq. 12, general $\sigma$})\;}
$$
**Special case $\beta = 1$ (Laplacian)**:
Hyrkkö-Selesnick (1999, ref [17])이 *해석적 식*을 도출: $T_h^* = \sqrt 2 \sigma^2/\sigma_X$ (hard) 또는 $T_B = 1/\sigma_X$와 거의 일치 (soft, Fig. 5).

**일반 $\beta$** (Fig. 6): $\beta = 0.6$부터 4까지 $T^*(\sigma_X, \beta)$을 수치 계산하면 $T_B(\sigma_X) = 1/\sigma_X$이 모든 곡선의 *가운데*를 잘 가로지름. 위험 차이 5% 이내.

**임계값 $T_B$의 직관**:
- $\sigma/\sigma_X \ll 1$ (high SNR): $T_B/\sigma$ 작음 → 신호 보존.
- $\sigma/\sigma_X \gg 1$ (low SNR): $T_B/\sigma$ 큼 → 잡음 제거.

#### English — §II.A
The optimal Bayes risk threshold for soft-thresholding under a GGD prior has no closed form in general but is well-approximated by $T_B(\sigma_X) = \sigma^2/\sigma_X$ within 5% across $\beta \in [0.5, 4]$. Intuition: high SNR → small threshold; low SNR → large threshold.

---

### Part III: §II.B Parameter Estimation / 파라미터 추정

#### 한국어
GGD에서 $\beta$는 $T_B$에 *명시적으로 들어가지 않으므로* 추정 불필요 (오직 $\sigma_X$와 $\sigma$만 필요).

**$\sigma$** (잡음): paper #1과 동일하게 MAD/0.6745:
$$
\hat\sigma = \frac{\mathrm{Median}(|Y_{ij}|)}{0.6745}, \quad Y_{ij} \in HH_1 \quad (16)
$$
**$\sigma_X^2$** (subband 신호): $Y = X + V$, $\mathrm{Cov}(X, V) = 0$ → 분산 분해
$$
\sigma_Y^2 = \sigma_X^2 + \sigma^2 \quad (17)
$$
$$
\hat\sigma_Y^2 = \frac{1}{n^2}\sum_{i, j} Y_{ij}^2 \quad (18)
$$
$$
\hat\sigma_X = \sqrt{\max(\hat\sigma_Y^2 - \hat\sigma^2, 0)} \quad (20)
$$
**Threshold** (Eq. 19):
$$
\hat T_B(\hat\sigma_X) = \begin{cases} \hat\sigma^2/\hat\sigma_X & \hat\sigma_X > 0 \\ \infty (\text{모든 계수 } 0\text{ 처리}) & \hat\sigma_X = 0 \end{cases}
$$
$\hat\sigma_X = 0$ (즉 $\hat\sigma_Y \le \hat\sigma$)인 경우는 신호가 잡음에 완전히 가려진 subband — *모든 계수를 0으로*. 큰 $\sigma$ (예 $\sigma > 20$ for grayscale)에서 발생.

#### English — §II.B Parameter estimation
Need only $\sigma$ (via MAD/0.6745 of $HH_1$) and $\sigma_X$ per subband (via subband empirical variance minus $\sigma^2$, clipped to zero). $\beta$ does *not* appear in $T_B$ explicitly.

---

### Part IV: §III MDL Principle for Compression-based Denoising / 압축 기반 노이즈 제거

#### 한국어
**핵심 통찰**: 임계화 + 양자화 = 단일 절차. Fig. 1처럼 zero-zone 양자화는 임계값 $T$ 외에도 zero-zone 밖의 *bin width* $\Delta$와 *bin 수* $m$을 결정해야 함. **MDL 원리 (Rissanen)**:
$$
L(\mathbf Y, \hat{\mathbf X}) = L(\mathbf Y | \hat{\mathbf X}) + L(\hat{\mathbf X}) \quad (21)
$$
- $L(\mathbf Y|\hat{\mathbf X})$: residual coding length (Gaussian 가정 하에 ½log(2πe·distortion))
- $L(\hat{\mathbf X})$: quantized signal coding length (Shannon-Fano, GGD 기반)

이 두 항의 합을 최소화하는 $(\hat T, \hat\Delta, \hat m)$을 선택. 결과적으로 (i) zero-zone에서는 BayesShrink threshold $\hat T_B$ 사용 (이미 결정), (ii) 외부 양자화 $\Delta$는 신호 분산에 비례하게 선택.

**효과**: 영상 압축 자체로 denoising 70-95% 달성. 단, 압축 자체의 양자화 잡음이 추가되므로 *낮은 $\sigma$*에서는 BayesShrink 단독이 더 좋음. *높은 $\sigma$*에선 압축+denoising이 시너지를 냄.

#### English — §III MDL
Couple thresholding with quantization. MDL principle minimises $L(\mathbf Y|\hat{\mathbf X}) + L(\hat{\mathbf X})$. Zero-zone width set by BayesShrink $\hat T_B$; outer quantization step $\Delta$ chosen to match GGD entropy. At high noise, this simultaneous coding-denoising is competitive with pure denoising while also compressing.

---

### Part V: §IV Experimental Results / 실험 결과

#### 한국어
**Test images**: Lena, Barbara, goldhill, baboon (512×512). Wavelet: Daub-8.
**Noise levels**: $\sigma = 5, 10, 22.5$ (8-bit grayscale).
**Comparison**: SureShrink (paper #2), OracleShrink (BayesShrink with true $\sigma_X$), OracleThresh (best hard-threshold using true image), BayesShrink, BayesShrink + MDL compression.

**관찰**:
1. **BayesShrink ≈ OracleShrink within 5%** — closed-form approximation 정확.
2. **BayesShrink > SureShrink most of the time** (8% MSE 개선이 일반).
3. **Worst case**: BayesShrink가 SureShrink 대비 1% 이내 나쁨.
4. **MDL compression**: $\sigma = 22.5$ (high noise)에서 BayesShrink와 비슷한 MSE + compression.

특히 $\sigma = 10$에서 BayesShrink는 SureShrink 대비 평균 5-10% 개선, OracleShrink의 95% 이내.

#### English — §IV Experiments
Across 4 images, 3 noise levels, BayesShrink lands within 5% of OracleShrink (the parameter-free version with true $\sigma_X$) and beats SureShrink ~8% of the time. The MDL-compressed version achieves comparable MSE with simultaneous compression, especially at high noise.

---

## 3. Key Takeaways / 핵심 시사점

1. **GGD prior로 자연 영상 통계를 단순 모델링 / GGD captures natural-image statistics** — Wavelet subband 계수가 GGD $\beta \in [0.5, 1]$를 따른다는 광범위한 경험적 사실 (Mallat, Simoncelli 등). 영점에서 peaked + heavy tail = sparse coding의 토대.
   GGD with $\beta \in [0.5, 1]$ (peaked at zero, heavy-tailed) is the universal empirical model for wavelet subbands of natural images.

2. **$T_B = \sigma^2/\sigma_X$는 단순하지만 거의 최적 / $\sigma^2/\sigma_X$ is simple yet near-optimal** — 베이즈 위험의 진짜 최적 $T^*(\sigma_X, \beta)$는 $\beta$에 의존하지만, *모든* $\beta \in [0.5, 4]$에 대해 $T_B = \sigma^2/\sigma_X$는 5% 이내 이상적. 따라서 $\beta$ 추정 불필요 → parameter-free.
   The closed-form $\sigma^2/\sigma_X$ approximates the true Bayes-optimal threshold to within 5% over the entire GGD shape range — bypassing the need to estimate $\beta$.

3. **$\sigma_X^2 = \sigma_Y^2 - \sigma^2$ (clipped)는 두 항 분해 / Variance decomposition** — 신호와 잡음 독립 → $\sigma_Y^2 = \sigma_X^2 + \sigma^2$. 단순 차분으로 $\sigma_X$ 추정. 음수가 되면 $\sigma_X = 0$ → subband 통째로 0 (즉 잡음에 완전 가려졌음).
   Independence of signal and noise gives the trivial decomposition, allowing $\sigma_X^2$ to be estimated by $(\hat\sigma_Y^2 - \hat\sigma^2)_+$. When the subband is too noisy ($\hat\sigma_Y \le \hat\sigma$), the entire subband is zeroed.

4. **Subband-adaptive (orientation + scale) / Subband-adaptive thresholding** — SureShrink는 *scale-only*, BayesShrink는 *scale × orientation*: 각 $(k, \{HH, HL, LH\})$ subband마다 별도 $\hat\sigma_X$. 자연 영상의 *방향성*이 다르므로(예 horizontal vs vertical edges) 이 추가 미세화가 효과 있음.
   SureShrink adapts only over scale; BayesShrink adapts over scale *and* orientation, which matters because natural images have direction-dependent statistics.

5. **Closed-form은 SURE보다 빠르고 단순 / Closed-form beats SURE on speed & simplicity** — SureShrink는 각 level마다 $O(d \log d)$ sort + minimisation. BayesShrink는 $O(n^2)$ 변동 한 번 + 나누기 한 번. 1024×1024 영상에서 BayesShrink는 SureShrink보다 ~10× 빠름 (그러면서 비슷하거나 조금 더 좋은 결과).
   BayesShrink runs in linear time (per subband: variance + division), versus SureShrink's $O(d\log d)$ sort — about 10× faster while matching or beating the MSE.

6. **임계화는 사실 압축이다 / Thresholding is a special case of compression** — Fig. 1: zero-zone 양자화는 $\Delta \to 0$ 극한에서 정확히 thresholding. 이 통찰이 paper의 두 번째 부분 (§III)을 가능케 함 — *압축 자체가 denoising* 한다.
   Soft thresholding is the limit of dead-zone quantization with infinite bin density outside the dead-zone — so any reasonable lossy coder is implicitly denoising.

7. **MDL이 quantizer 디자인을 자동화 / MDL automates quantizer design** — Rissanen MDL은 *코드 길이* 관점에서 $(\hat T, \hat\Delta, \hat m)$을 선택 — 단일 객관 함수가 두 작업 ($\hat T$: denoising, $(\hat\Delta, \hat m)$: compression)을 동시 결정. 압축률·왜곡 trade-off는 자동.
   MDL provides a single objective function (combined coding length) that simultaneously chooses denoising threshold and quantizer parameters — automating the rate–distortion tradeoff.

8. **VisuShrink는 너무 매끈, BayesShrink는 적당히 매끈 / VisuShrink over-smooths, BayesShrink balances** — VisuShrink $T = \sigma\sqrt{2\log M}$에서 $M = 512^2 = 2.6\times 10^5$이면 $T \approx 5.13\sigma$ → 거의 모든 *signal* 계수도 cut. 이게 VisuShrink가 영상에서 "blurry" 보이는 이유. $T_B \sim \sigma^2/\sigma_X$는 일반적으로 훨씬 작음 (low-frequency subband에서 $\sigma_X \gg \sigma$ → $T_B \ll \sigma$).
   VisuShrink's $\sigma\sqrt{2\log M}$ is wasteful for images ($M = 512^2$); BayesShrink's data-driven threshold is *much* smaller in low-frequency subbands where signal dominates, preserving structure better.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Image model and target / 영상 모델과 목표
$$
g_{ij} = f_{ij} + \varepsilon_{ij}, \quad \varepsilon_{ij} \overset{iid}{\sim} N(0, \sigma^2)
$$
2-D dyadic orthogonal wavelet transform: $Y = \mathcal W g, X = \mathcal W f, V = \mathcal W \varepsilon$, all in matrix form.

### 4.2 GGD prior / 일반화 가우시안 사전분포
$$
GG_{\sigma_X, \beta}(x) = C(\sigma_X, \beta)\exp\bigl\{-(\alpha|x|)^\beta\bigr\}
$$
$$
\alpha(\sigma_X, \beta) = \sigma_X^{-1}\sqrt{\Gamma(3/\beta)/\Gamma(1/\beta)}
$$
### 4.3 Bayes risk and BayesShrink threshold / 베이즈 위험과 BayesShrink 임계값
$$
r(T) = E_X E_{Y|X}(\eta_T(Y) - X)^2, \quad T^* = \arg\min r(T)
$$
$$
\boxed{\;T_B(\sigma_X) = \sigma^2/\sigma_X \quad (\text{closed-form approximation})\;}
$$
Risk $\le 1.05 \cdot r(T^*)$ for all $\beta \in [0.5, 4]$.

### 4.4 Parameter estimation / 파라미터 추정
$$
\hat\sigma = \mathrm{Median}(|Y_{ij}|)/0.6745, \quad Y \in HH_1
$$
$$
\hat\sigma_Y^2 = \frac{1}{n^2}\sum_{i,j} Y_{ij}^2 \quad \text{(per subband)}
$$
$$
\hat\sigma_X = \sqrt{\max(\hat\sigma_Y^2 - \hat\sigma^2, 0)}
$$
$$
\hat T_B = \begin{cases} \hat\sigma^2/\hat\sigma_X & \hat\sigma_X > 0 \\ \max_{i,j}|Y_{ij}| & \hat\sigma_X = 0 \text{ (zero subband)}\end{cases}
$$
### 4.5 BayesShrink algorithm / BayesShrink 알고리즘
**Input**: noisy image $g$, wavelet (Daub-8 typical), $J$ levels.
**Step 1**: $Y = \mathcal W g$ (2-D DWT, $J$ levels → $LL_J + \{HH_k, HL_k, LH_k\}_{k=1}^J$ subbands).
**Step 2**: $\hat\sigma$ from $HH_1$ MAD/0.6745.
**Step 3**: For each detail subband $B \in \{HH_k, HL_k, LH_k\}$, $k = 1, \ldots, J$:
  a. $\hat\sigma_Y^2 = n_B^{-1} \sum_{Y \in B} Y^2$, $n_B$ = subband size.
  b. $\hat\sigma_X = \sqrt{\max(\hat\sigma_Y^2 - \hat\sigma^2, 0)}$.
  c. Threshold: $T_B = \hat\sigma^2/\hat\sigma_X$ (or set entire subband to 0 if $\hat\sigma_X = 0$).
  d. Apply soft thresholding to all coefficients in $B$.
**Step 4**: Keep $LL_J$ untouched.
**Step 5**: $\hat f = \mathcal W^{-1} \hat X$.

### 4.6 Worked example / 수치 예시
$\sigma = 10$ (8-bit grayscale). For Lena $LH_1$ subband (high-frequency horizontal detail): $\hat\sigma_Y \approx 12$ → $\hat\sigma_X = \sqrt{144 - 100} = 6.6$ → $T_B = 100/6.6 \approx 15.1$. For $LH_3$ (low-frequency, mostly signal): $\hat\sigma_Y \approx 50$ → $\hat\sigma_X \approx 49$ → $T_B \approx 2.0$.
Compare with VisuShrink universal $T_U = 10\sqrt{2\log 262144} = 51.3$ — would zero out *all* $LH_3$ coefficients! BayesShrink's level-adaptive 2.0 vs 51.3 for low-frequency band is the key difference.

### 4.7 MDL compression-denoising criterion / MDL 압축-노이즈 제거 기준
$$
L(\mathbf Y, \hat{\mathbf X}) = L(\mathbf Y|\hat{\mathbf X}) + L(\hat{\mathbf X}) \quad (21)
$$
Coding-length terms are evaluated assuming Gaussian residual + GGD source.

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1989 ─── Mallat — 2-D wavelet decomposition for natural images
          ↳ recognises subband statistics resemble Laplacian
1989 ─── Rissanen — MDL principle (Stochastic Complexity)
1992 ─── Mallat-Zhong — "wavelet coefficients of natural images
          tend to follow a generalised Gaussian"
1993 ─── Antonini-Barlaud-Mathieu-Daubechies — JPEG2000 GGD modelling
1994 ─── Donoho-Johnstone — VisuShrink/RiskShrink (paper #1)
1995 ─── Donoho-Johnstone — SureShrink (paper #2)
1995 ─── Saito — "Simultaneous noise suppression and signal compression
          using a library of orthonormal bases and the MDL criterion"
1996 ─── Simoncelli-Adelson — Bayesian estimation of wavelet coefficients
          ↳ uses Gaussian-scale-mixture priors
1999 ─── Hyrkkö-Selesnick — analytical Laplacian-prior threshold
2000 ★★ CHANG-YU-VETTERLI — BayesShrink + MDL (THIS PAPER)
                              ↳ closed-form sigma^2/sigma_X
                              ↳ subband-adaptive
                              ↳ simultaneous compression-denoising
2002 ─── Sendur-Selesnick — Bivariate shrinkage (parent-child)
2003 ─── Portilla-Strela-Wainwright-Simoncelli — BLS-GSM
                              ↳ Gaussian scale mixture, more sophisticated prior
2007 ─── Dabov+ — BM3D (paper #7) — uses similar shrinkage in 3D groups
2010+ ── DnCNN, Restormer — deep replacements; learn shrinkage implicitly
```

이 논문은 *"baby Bayes"* — GGD라는 단순한 prior에서 시작해 닫힌 형태로 도달. 후속의 BLS-GSM·EBM 등 더 정교한 prior 모델들의 출발점이다. 또한 MDL 부분은 wavelet-based JPEG2000 코덱의 zero-tree 코딩과 사상적으로 연결.

This paper is "baby Bayes": a simple GGD prior leads to a clean closed-form threshold. Sets the template for BLS-GSM and later sophisticated priors. The MDL part dovetails with JPEG2000-era zero-tree wavelet coding.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **Donoho-Johnstone (1994)** *Biometrika* (paper #1) | VisuShrink baseline | BayesShrink directly addresses VisuShrink's over-smoothing on images ($M = 512^2$ makes universal threshold too large). |
| **Donoho-Johnstone (1995)** *JASA* (paper #2) | SureShrink predecessor | BayesShrink replaces level-dependent SURE minimisation with closed-form $\sigma^2/\sigma_X$; 8% better, 10× faster typically. |
| **Mallat (1989)** *IEEE PAMI* | 2-D dyadic DWT | Provides the subband decomposition into $\{LL_J, HH_k, HL_k, LH_k\}$. |
| **Simoncelli-Adelson (1996)** | Bayesian wavelet shrinkage | Direct Bayesian ancestor; uses Gaussian-scale-mixture prior. BayesShrink is the GGD analog. |
| **Saito (1995)** *Wavelets in Geophysics* | Simultaneous denoising-compression via MDL | The §III MDL part of this paper is a refinement of Saito's earlier work for hard-thresholding to soft-threshold + GGD context. |
| **Hyrkkö-Selesnick (1999)** | Analytical Laplacian-prior threshold | Provides the $\beta = 1$ exact result; this paper notes $T_B = 1/\sigma_X$ approximates it within 0.8% of optimal risk. |
| **Portilla-Strela-Wainwright-Simoncelli (2003)** *IEEE TIP* | BLS-GSM | Successor: replaces GGD with Gaussian scale mixture for stronger natural-image modelling. |
| **Dabov+ (2007)** *IEEE TIP* (paper #7) | BM3D | Uses similar Wiener (≈ Bayesian) shrinkage but in 3-D groups; BayesShrink heritage in collaborative filtering step. |
| **Sendur-Selesnick (2002)** *SPL* | Bivariate shrinkage | Adds parent-child correlation to GGD prior → bivariate joint distribution. |

---

## 7. References / 참고문헌

- Antonini, M., Barlaud, M., Mathieu, P., & Daubechies, I., "Image coding using wavelet transform", *IEEE Trans. Image Processing*, 1, 205–220 (1992).
- Chang, S. G., Yu, B., & Vetterli, M., "Adaptive wavelet thresholding for image denoising and compression", *IEEE Trans. Image Processing*, 9(9), 1532–1546 (2000). [DOI: 10.1109/83.862633]
- Donoho, D. L., & Johnstone, I. M., "Ideal spatial adaptation by wavelet shrinkage", *Biometrika*, 81(3), 425–455 (1994).
- Donoho, D. L., & Johnstone, I. M., "Adapting to unknown smoothness via wavelet shrinkage", *J. American Statistical Association*, 90(432), 1200–1224 (1995).
- Mallat, S., "A theory for multiresolution signal decomposition: the wavelet representation", *IEEE PAMI*, 11, 674–693 (1989).
- Hyrkkö, A., & Selesnick, I. W., "Threshold of soft-shrinkage for the Laplacian prior", *Tampere International Center for Signal Processing Report* (1999).
- Rissanen, J., *Stochastic Complexity in Statistical Inquiry*, World Scientific (1989).
- Saito, N., "Simultaneous noise suppression and signal compression using a library of orthonormal bases and the minimum description length criterion", in *Wavelets in Geophysics*, eds. Foufoula-Georgiou & Kumar (1994).
- Simoncelli, E. P., & Adelson, E. H., "Noise removal via Bayesian wavelet coring", *Proc. IEEE ICIP*, 379–382 (1996).
