---
title: "Pre-Reading Briefing: Image Denoising in Mixed Poisson-Gaussian Noise (PURE-LET)"
paper_id: "13_luisier_2011"
topic: Low_SNR_Imaging
date: 2026-05-06
type: briefing
---

# Image Denoising in Mixed Poisson-Gaussian Noise: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Luisier, F., Blu, T., & Unser, M., "Image denoising in mixed Poisson-Gaussian noise", *IEEE Trans. Image Process.*, 20(3), 696–708 (2011).
**Author(s)**: Florian Luisier, Thierry Blu, Michael Unser
**Year**: 2011

---

## 1. 핵심 기여 / Core Contribution

### 한국어
이 논문은 실제 광검출기(CCD/CMOS, 형광현미경)에서 발생하는 **포아송 + 가우시안 혼합 잡음** $y = z + b$ (샷 잡음 $z \sim \mathcal P(x)$, 읽기 잡음 $b \sim N(0,\sigma^2)$)에 대한 *unbiased risk estimator*인 **PURE**(Poisson-Gaussian Unbiased Risk Estimate)를 도출하고, **LET**(Linear Expansion of Thresholds)로 폐형(closed-form) 최적화를 가능하게 한다. PURE는 Stein 보조정리(가우시안)와 Hudson 항등식(포아송)을 결합한 것으로 SURE의 자연스러운 일반화이며, $\sigma=0$이면 순수 PURE로, $x \to \infty$이면 SURE로 환원된다. LET 매개변수화는 추정기를 $K$개의 thresholding expert의 *선형 결합*으로 쓰기 때문에 PURE가 계수에 대해 *2차식*이 되어 $K\times K$ 선형방정식만 풀면 최적해가 closed form으로 얻어진다 — 그리드 탐색·반복 최적화 불필요.

### English
This paper derives **PURE** (Poisson-Gaussian Unbiased Risk Estimate), a clean-target-free MSE estimator for the mixed Poisson + Gaussian sensor noise model $y = z + b$ that matches real CCD/CMOS and fluorescence-microscopy data. PURE combines Stein's lemma (for the Gaussian part) with Hudson's identity (for the Poisson part), generalising SURE to the mixed regime. The companion **LET** (Linear Expansion of Thresholds) parameterisation writes the denoiser as a linear combination of $K$ thresholding experts, making PURE *quadratic* in the coefficients — so the optimum is obtained by solving a small $K\times K$ linear system in closed form, without grid search or iteration. PURE-LET outperforms Anscombe + BLS-GSM and Platelet at all tested noise levels, and is 30–1000× faster than competing Poisson denoisers.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting
**한국어**: 2010년 전후의 저광량 영상 잡음 제거는 두 갈래로 나뉘어 있었다. 한쪽은 Anscombe 변환(VST)으로 포아송을 가우시안화한 뒤 BLS-GSM·BM3D 같은 가우시안 디노이저를 적용하는 *간접* 접근; 다른 쪽은 Willett-Nowak의 platelet, Kolaczyk의 corrected threshold처럼 포아송을 *직접* 모델링하는 접근. 그러나 (i) Anscombe 기반 방법은 매우 어두운 영역(peak < 5)에서 분산 안정화가 깨져 실패했고, (ii) 직접 포아송 알고리즘은 *읽기 잡음*까지 함께 처리하기 어려웠다. Luisier·Blu·Unser는 SURE의 unbiased risk 패러다임을 *혼합 잡음*으로 일반화함으로써 이 두 약점을 동시에 우회한다.

**English**: Around 2010, low-photon denoising had split into two camps: (i) *indirect* — apply an Anscombe variance-stabilising transform (VST) and then a Gaussian denoiser like BLS-GSM or BM3D; (ii) *direct* — model Poisson statistics natively (Willett-Nowak's Platelet, Kolaczyk's corrected thresholds, Lefkimmiatis's Poisson-Haar HMT). Both had blind spots: Anscombe broke down at sub-photon counts (peak < 5), and direct Poisson methods had no clean way to fold in *read noise*. Luisier, Blu, and Unser closed both gaps by generalising SURE's unbiased-risk paradigm to the mixed Poisson-Gaussian regime.

### 타임라인 / Timeline
```
1948 ─── Anscombe — Poisson VST (paper #11)
1978 ─── Hudson — natural identity for Poisson (analogue of Stein lemma)
1981 ─── Stein — SURE / Stein's lemma for Gaussian
1995 ─── Donoho-Johnstone — SureShrink (paper #2; SURE on wavelets)
2003 ─── Portilla+ — BLS-GSM (Bayesian Gaussian baseline)
2007 ─── Willett-Nowak — Platelet (penalised-likelihood Poisson)
2010 ─── Luisier-Vonesch-Blu-Unser — fast Haar PURE-LET (precursor)
2011 ★★ Luisier-Blu-Unser — PURE-LET for mixed Poisson-Gaussian (THIS PAPER)
2013 ─── Mäkitalo-Foi — exact unbiased GAT inverse (paper #14, partly counters this work)
```

---

## 3. 필요한 배경 지식 / Prerequisites

- **수학 / Math**:
  - Stein's lemma: $E[b^T f(y)] = \sigma^2 E[\mathrm{div}\, f(y)]$ for $b \sim N(0,\sigma^2 I)$ — 가우시안 잡음의 *unbiased risk* 기저.
  - Hudson's identity: $E[x f(z)] = E[z f(z-1)]$ for $z \sim \mathcal P(x)$ — Poisson용 Stein 유사 항등식.
  - 1차 Taylor 전개: $f(y - e_n) \approx f(y) - \partial f/\partial y_n$ (smooth $f$용).
- **신호처리 / Signal processing**:
  - Wavelet / undecimated wavelet transform (UWT), block-DCT (BDCT) 의 perfect reconstruction.
  - Soft / hard thresholding, $C^\infty$ smooth nonlinearity.
- **통계 / Statistics**:
  - Mean squared error (MSE), unbiased estimation, signal-dependent variance.
  - Photon counting model: $\mathrm{Var}(z) = E[z] = x$ for Poisson.
- **선행 논문 / Prior reading**:
  - Paper #2 (SureShrink) — SURE의 wavelet 적용.
  - Paper #11 (Anscombe) — VST의 원조 및 비교 대상.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| PURE | Poisson-Gaussian Unbiased Risk Estimate — MSE의 비편향 추정량. 클린 GT 없이 계산 가능. / Unbiased estimator of MSE for Poisson-Gaussian noise; computable without clean ground truth. |
| LET | Linear Expansion of Thresholds — 추정기를 $K$ thresholding expert의 선형 결합으로 작성. / Writes the denoiser as a linear combination of $K$ thresholding experts. |
| Stein's lemma | 가우시안 noise 의 risk를 divergence로 표현. / Expresses Gaussian-noise risk via divergence. |
| Hudson's identity | Poisson noise의 Stein 유사물. $E[xf(z)] = E[zf(z-1)]$. / Poisson analogue of Stein, $E[xf(z)] = E[zf(z-1)]$. |
| GDC | Group-Delay Compensation — undecimated wavelet 의 lowpass / bandpass 사이 위상 보정 allpass 필터. / Allpass filter correcting phase difference between lowpass and bandpass channels in UWT. |
| UWT | Undecimated Wavelet Transform — translation-invariant wavelet 변환. / Translation-invariant wavelet transform. |
| BDCT | Block Discrete Cosine Transform — JPEG-style block transform. / JPEG-style block transform. |
| Subband-dependent threshold | $t_j(\bar w) = \sqrt{\beta_j |\bar w| + \sigma^2}$ — 신호 강도에 따라 자동 조정. / Threshold $t_j(\bar w) = \sqrt{\beta_j |\bar w| + \sigma^2}$ that adapts to local signal magnitude. |
| Multi-base LET | UWT + BDCT 처럼 다중 변환을 동시에 사용. / Combines multiple transforms (e.g., UWT + BDCT) in one LET system. |
| Reliability check | 매우 어두운 subband에서 Taylor 근사 무효 → 그 subband 제외. / Excludes subbands where Taylor approximation breaks down. |
| Foi noise estimation | sample-mean / sample-variance 회귀로 $\alpha,\sigma$ 자동 추정. / Automatic $\alpha,\sigma$ estimation via local mean-variance regression. |
| PSNR | Peak Signal-to-Noise Ratio — 영상 화질 비교 지표 (dB). / Standard image-quality metric in dB. |

---

## 5. 수식 미리보기 / Equations Preview

### 5.1 Mixed noise model / 혼합 잡음 모델
$$
\mathbf y = \mathbf z + \mathbf b, \quad z_n \sim \mathcal P(x_n), \quad b_n \sim N(0, \sigma^2)
$$
포아송 샷 잡음 + 가우시안 읽기 잡음의 합. 두 잡음원은 독립. / Poisson shot noise plus independent Gaussian read noise.

### 5.2 PURE for Poisson-Gaussian (Theorem 1, Eq. 2) / 핵심 정리
$$
\hat\varepsilon = \tfrac{1}{N}\bigl(\|\mathbf f(\mathbf y)\|^2 - 2 \mathbf y^T \mathbf f^-(\mathbf y) + 2\sigma^2\,\mathrm{div}\{\mathbf f^-(\mathbf y)\}\bigr) + \tfrac{1}{N}(\|\mathbf y\|^2 - \mathbf 1^T \mathbf y) - \sigma^2
$$
$\mathbf f^-(\mathbf y)_n = f_n(\mathbf y - \mathbf e_n)$이고, $\mathrm{div}$는 추정량의 발산. $E[\hat\varepsilon] = \mathrm{MSE}$ 만족. / Unbiased estimator of MSE; the only $\mathbf x$-dependent quantity has been replaced by observable $\mathbf y$ via Stein + Hudson.

### 5.3 LET parameterisation and linear system (Eq. 14, 15) / LET 매개변수화
$$
\mathbf f(\mathbf y) = \sum_{k=1}^K a_k \mathbf f_k(\mathbf y), \quad \mathbf M \mathbf a = \mathbf c
$$
$\mathbf f_k$ are $K$ thresholding experts; PURE is *quadratic* in $\{a_k\}$ → closed-form $K \times K$ linear system. / LET writes the denoiser as a linear combination of $K$ thresholding experts, making PURE quadratic and the optimum a closed-form linear-system solution.

### 5.4 Subband-dependent threshold (Eq. 19) / 부대역 의존 임계
$$
t_j(\bar w) = \sqrt{\beta_j |\bar w| + \sigma^2}
$$
$|\bar w|$는 lowpass scaling coefficient ≈ local signal estimate; $\beta_j$는 scale-dependent factor. 포아송 잡음의 $\sigma \propto \sqrt x$ 스케일링과 $\sigma^2$ 읽기 잡음을 결합. / Combines a Poisson-noise std proxy $\sqrt{\beta_j|\bar w|}$ with the Gaussian read-noise term.

### 5.5 Smooth thresholding (Eq. 20) / 부드러운 임계화
$$
\theta_j(w, \bar w) = a_{j,1}\,w + a_{j,2}\,w \exp\!\bigl(-(w/(3 t_j(\bar w)))^8\bigr)
$$
첫 항은 linear (no thresholding), 둘째 항은 hard-like 차감. $C^\infty$ 매끄러움이 LET 최적화의 closed-form gradient에 필수. / First term is linear, second is a smooth hard-like nonlinearity. $C^\infty$ smoothness is required for the LET closed-form gradient.

---

## 6. 읽기 가이드 / Reading Guide

### 한국어
**우선 읽을 부분 / Focus first**:
1. **§II Theorem 1 (PURE 유도)** — Stein lemma + Hudson identity 결합이 어떻게 unbiased estimator를 만드는지 단계별로 따라가기. Hudson identity 증명 스케치(notes §4.13)를 옆에 두면 도움.
2. **§II-C Taylor approximation** — 정확한 PURE는 $N$번 알고리즘 재실행이 필요하므로 1차 Taylor 근사가 *실용성의 핵심 트릭*. Eq. (6)이 왜 정확한지(매끄러운 $f_n$).
3. **§III-E LET 매개변수화** — Eq. (14)에서 PURE가 $a_k$에 대해 *2차식*임을 확인. Eq. (15) 선형방정식이 어떻게 최적해를 closed form으로 주는지.

**자주 헷갈리는 지점 / Common stumbling blocks**:
- $\mathbf f^-(\mathbf y)_n = f_n(\mathbf y - \mathbf e_n)$ 표기. 좌표별로 1을 빼고 평가 — 단일 좌표가 다른 점에서.
- "Subband-dependent threshold"는 *고정된 임계*가 아니라 픽셀별 함수. $\bar w$가 위치마다 다르므로 임계도 위치마다 다르다.
- LET는 "$K$ 후보 중 하나 고르기"가 아니라 *모두를 가중 결합*. 가중치는 PURE가 결정.
- Multi-base LET (Eq. 23)에서 UWT와 BDCT 결합은 단순 평균이 아니라 PURE-기반 *최적 가중*.

### English
**Focus first**:
1. **§II Theorem 1 (PURE derivation)** — Trace step-by-step how Stein's lemma + Hudson's identity build an unbiased estimator. Keep the Hudson identity proof sketch (notes §4.13) beside you.
2. **§II-C Taylor approximation** — Exact PURE requires re-running the denoiser $N$ times; the 1st-order Taylor expansion is the *key practical trick* that makes PURE computable in one pass.
3. **§III-E LET parameterisation** — Verify in Eq. (14) that PURE is *quadratic* in $\{a_k\}$. Understand why Eq. (15)'s linear system gives the closed-form optimum.

**Common stumbling blocks**:
- The notation $\mathbf f^-(\mathbf y)_n = f_n(\mathbf y - \mathbf e_n)$ subtracts 1 from coordinate $n$ only.
- "Subband-dependent threshold" is a *function*, not a fixed scalar — it varies with position because $\bar w$ does.
- LET is not "pick one of $K$" but *weighted combination of all $K$* with weights determined by PURE.
- Multi-base LET (Eq. 23) combines transforms via PURE-optimal weights, not simple averaging.

---

## 7. 현대적 의의 / Modern Significance

### 한국어
PURE-LET의 핵심 아이디어 — *unbiased risk estimator 를 클린 타겟 없이 학습 신호로 사용* — 는 2018년 이후 self-supervised deep denoising (Noise2Noise, Noise2Self, SURE-trained CNN)의 직접적 선조다. 특히 Stein-trained CNN 계열(Metzler 2018, Zhussip 2018)은 SURE를 deep network 학습 손실로 사용하는데, PURE는 그 Poisson-Gaussian 확장에 해당. 또한 본 논문이 도입한 *closed-form linear LET*은 후일 deep ensembling(다수 네트워크의 가중 결합)과 맥락이 닿아 있으며, Foi의 GAT 기반 방법(paper #14)에서 *반박*되긴 하지만 unbiased risk 패러다임 자체는 여전히 살아 있다. 실 형광현미경 응용에서 paper #14(GAT + BM3D + exact inverse)와 함께 가장 많이 인용되는 두 표준 알고리즘 중 하나.

### English
PURE-LET's core idea — using an *unbiased risk estimator as a training signal without clean targets* — is a direct ancestor of post-2018 self-supervised deep denoising (Noise2Noise, Noise2Self, SURE-trained CNNs by Metzler 2018 / Zhussip 2018). Stein-trained CNNs are essentially "deep SURE"; PURE is the Poisson-Gaussian extension. The closed-form linear LET also foreshadows modern deep ensembling (weighted combination of multiple networks). Although Mäkitalo-Foi's exact unbiased GAT inverse (paper #14) partially counters PURE-LET's "bypass-VST" claim, the unbiased-risk paradigm itself remains foundational, and PURE-LET is still one of the two most-cited reference algorithms (alongside GAT + BM3D + exact inverse) for low-light fluorescence-microscopy denoising.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
