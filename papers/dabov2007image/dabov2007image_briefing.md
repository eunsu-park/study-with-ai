---
title: "Pre-Reading Briefing: Image Denoising by Sparse 3-D Transform-Domain Collaborative Filtering (BM3D)"
paper_id: "07_dabov_2007"
topic: Low_SNR_Imaging
date: 2026-05-06
type: briefing
---

# BM3D (Dabov+ 2007): Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Dabov, K., Foi, A., Katkovnik, V., & Egiazarian, K., "Image denoising by sparse 3-D transform-domain collaborative filtering", *IEEE Trans. Image Process.* 16(8), 2080–2095 (2007). [DOI: 10.1109/TIP.2007.901238]
**Author(s)**: Kostadin Dabov, Alessandro Foi, Vladimir Katkovnik, Karen Egiazarian
**Year**: 2007

---

## 1. 핵심 기여 / Core Contribution

### 한국어
BM3D는 NLM(논문 #4)의 *block matching* 아이디어와 wavelet shrinkage(논문 #1–3)의 *transform-domain shrinkage*를 결합한 **2-단계 알고리즘**이다. (1) 유사 패치를 stacking하여 3-D group을 만든 뒤, (2) 분리형 3-D 변환 ($\mathcal T_{2D} \otimes \mathcal T_{1D}$)에서 hard-threshold를 적용하고, (3) inverse transform 후 inverse-variance 가중평균(aggregation)으로 basic estimate를 얻는다. Step 2는 이 basic estimate를 *pilot signal*로 삼아 4-D Wiener shrinkage를 다시 한 번 수행한다. 결과는 2007 시점 SOTA — Lena σ=25에서 **32.08 dB**, 모든 baseline(BLS-GSM, K-SVD, NLM, SA-DCT)을 0.5–1.5 dB 능가하며 이후 약 10년간 *non-learned* denoising의 표준이 되었다.

### English
BM3D fuses **block matching** (the patch-similarity idea of NLM, paper #4) with **transform-domain shrinkage** (the wavelet-thresholding tradition of papers #1–3) into a **two-step algorithm**. (1) Stack mutually similar 2-D patches into a 3-D group; (2) apply a separable 3-D transform with hard-thresholding (Step 1) followed by empirical Wiener filtering (Step 2) using the Step-1 result as pilot; (3) aggregate block-wise estimates with inverse-variance weights. Achieving **32.08 dB on Lena at σ=25**, BM3D set the 2007 SOTA and remained the dominant non-learned denoiser until deep CNNs (DnCNN, 2017) edged past it by ~0.3 dB.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

#### 한국어
2000년대 중반은 image denoising에서 두 갈래의 접근이 경쟁하던 시기였다. 한쪽은 *transform-domain* 진영(VisuShrink/SureShrink/BayesShrink, BLS-GSM)으로 wavelet 또는 GSM 사전을 활용한 단일 영상 변환 후 thresholding/Bayesian shrinkage를 사용했다. 다른 쪽은 *spatial-domain* 진영(NLM, exemplar-based methods)으로 영상 *내부의 self-similarity*를 활용해 픽셀을 직접 평균했다. BM3D는 이 둘을 **물리적으로 통합**한다 — 비국소 패치 그룹을 변환 영역에서 처리.

#### English
By the mid-2000s, image denoising had two competing camps: the *transform-domain* school (VisuShrink, BayesShrink, BLS-GSM) using wavelet/GSM priors with single-image thresholding, and the *spatial-domain* school (NLM, exemplar-based) exploiting in-image self-similarity through pixel averaging. BM3D unified the two — non-local patch groups processed in a transform domain — yielding a synergy neither approach achieved alone.

### 타임라인 / Timeline

```
1994 ─── Donoho-Johnstone — VisuShrink (paper #1, transform thresholding)
2000 ─── Chang-Yu-Vetterli — BayesShrink (paper #3)
2003 ─── Portilla+ — BLS-GSM (Gaussian-scale-mixture wavelet prior)
2005 ─── Buades-Coll-Morel — Non-Local Means (paper #4, spatial NL)
2006 ─── Aharon-Elad-Bruckstein — K-SVD (learned dictionary)
2007 ★★ DABOV-FOI-KATKOVNIK-EGIAZARIAN — BM3D (THIS PAPER)
2017 ─── Zhang+ — DnCNN (deep CNN finally surpasses BM3D, narrowly)
2022 ─── Zamir+ — Restormer (transformer denoising)
```

---

## 3. 필요한 배경 지식 / Prerequisites

### 한국어
- **Wavelet hard/soft thresholding**: 논문 #1–3의 변환영역 shrinkage 원리
- **Non-Local Means (NLM)**: 논문 #4의 patch similarity와 weighted averaging
- **Wiener filtering**: 신호 분산 추정을 활용한 MMSE shrinkage
- **분리형 (separable) 변환**: 2-D DCT/DWT, 1-D Haar의 텐서곱 구조
- **Inverse-variance weighting (BLUE)**: 독립 추정값들의 최적 결합

### English
- **Wavelet hard/soft thresholding** (papers #1–3)
- **Non-Local Means** (paper #4): patch similarity, weighted averaging
- **Wiener filtering**: MMSE shrinkage using a pilot signal
- **Separable transforms**: tensor products $\mathcal T_{2D} \otimes \mathcal T_{1D}$
- **Inverse-variance weighting (BLUE)** for combining independent estimates

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| Block matching / 블록 매칭 | $L^2$ 거리 기준 search window 내의 유사 패치 stacking. NLM에서 유래. / Stacking patches similar (in $L^2$) within a search window — inherited from NLM. |
| 3-D group / 3-D 그룹 | $|S|$개 유사 패치를 쌓은 $N_1 \times N_1 \times |S|$ tensor. / A stack of $|S|$ similar patches forming a 3-D tensor. |
| Collaborative filtering / 협력 필터링 | 그룹 내 *각각의* 패치에 대한 추정값을 동시에 생성. / Produces an estimate for *each* patch in the group, not a single average. |
| Hard-threshold prefilter / 하드 임계화 사전필터 | Step 1의 거리 측정에서 noise 영향을 줄이기 위한 변환 영역 임계화. / Threshold applied during distance computation to suppress noise. |
| Empirical Wiener / 경험적 위너 | Pilot signal을 사용한 element-wise shrinkage $W = |Y|^2 / (|Y|^2 + \sigma^2)$. / Element-wise shrinkage using a pilot estimate. |
| Aggregation / 집계 | 같은 픽셀에 대한 다수 추정값을 inverse-variance 가중평균. / Combine multiple per-pixel estimates with inverse-variance weights. |
| Sparsity in 3-D / 3-D 희소성 | 유사 패치 stacking 후 분리형 3-D 변환에서 매우 적은 계수만 유의. / After stacking, the separable 3-D transform makes the group highly sparse. |
| Two-step refinement / 2단계 정제 | HT basic estimate → Wiener filtering using basic as pilot. / Step 1 hard-threshold gives a basic estimate; Step 2 Wiener uses it as pilot. |
| Inter-fragment correlation / 패치 간 상관 | 그룹 내 *서로 다른 패치 사이*의 유사성. 1-D 변환이 활용. / Similarity *between* stacked patches; exploited by the 1-D transform along stack direction. |
| Kaiser window / 카이저 창 | Aggregation 시 경계 효과 완화를 위한 weighting. / Boundary-effect mitigation in aggregation. |
| Predictive search BM / 예측 검색 BM | 이전 검색 결과 부근에서만 작은 영역 검색 — 속도 향상. / Search only near previous match for speed. |

---

## 5. 수식 미리보기 / Equations Preview

**관측 모델 / Observation model**:
$$
z(x) = y(x) + \eta(x), \quad \eta \sim \mathcal N(0, \sigma^2 I)
$$

**Step 1 — 협력 하드 임계화 / Collaborative hard-thresholding (Eq. 6)**:
$$
\hat{\mathbf Y}^{ht}_S = \mathcal T^{ht-1}_{3D}\bigl(\Upsilon\bigl(\mathcal T^{ht}_{3D}\,\mathbf Z_S\bigr)\bigr), \quad \Upsilon(c) = c\cdot \mathbf 1\{|c|>\lambda_{3D}\sigma\}
$$

**Aggregation 가중치 / Aggregation weight (Eq. 10)**:
$$
w^{ht}_{x_R} = \frac{1}{\sigma^2 N^{x_R}_{\text{har}}}, \quad N^{x_R}_{\text{har}} = \#\{\text{retained 3-D coefficients}\}
$$

**Step 2 — 경험적 Wiener shrinkage / Empirical Wiener (Eq. 8–9)**:
$$
\mathbf W_S = \frac{|\mathcal T^{wie}_{3D} \hat{\mathbf Y}^{basic}_S|^2}{|\mathcal T^{wie}_{3D} \hat{\mathbf Y}^{basic}_S|^2 + \sigma^2}, \quad \hat{\mathbf Y}^{wie}_S = \mathcal T^{wie-1}_{3D}\bigl(\mathbf W_S \cdot \mathcal T^{wie}_{3D}\,\mathbf Z_S\bigr)
$$

**Wiener weight (Eq. 11)**:
$$
w^{wie}_{x_R} = \frac{1}{\sigma^2 \|\mathbf W_S\|^2_2}
$$

---

## 6. 읽기 가이드 / Reading Guide

### 한국어
- **§II (개념 프레임워크)**: grouping, collaborative filtering, intra/inter-fragment 상관의 정의를 정확히 파악. 이후 모든 설명의 기초.
- **§III (알고리즘)**: Eq. 4–11을 손으로 따라가며 *blocks → groups → transform → shrink → invert → aggregate*의 흐름을 머리에 그려 둘 것. 특히 Step 1 distance metric의 hard-threshold prefilter (Eq. 4) 의도를 이해.
- **§IV (빠른 구현)**: predictive search, sliding-block transform 사전 계산, Kaiser window 등은 PSNR보다 *속도*를 위한 트릭. 첫 읽기에서는 가볍게 통과 가능.
- **§V (CBM3D)**: opponent color space에서 luminance만으로 BM 수행하는 이유 — chrominance가 잡음에 약함.
- **§VI Table II (변환 민감도)**: 1-D 변환(Haar)이 2-D 변환 선택보다 훨씬 결정적 — *inter-fragment* 상관이 본질이라는 핵심 통찰.
- **흔한 오해**: BM3D는 "그룹 평균"이 아니다. 그룹 내 *각 패치*마다 별도 추정값이 나오고, 그것들이 aggregation에서 결합된다.

### English
- **§II (conceptual framework)**: nail down grouping, collaborative filtering, and intra/inter-fragment correlation — these notions drive every later section.
- **§III (algorithm)**: trace Eqs. 4–11 by hand. Visualise *blocks → groups → transform → shrink → invert → aggregate*. Note why the distance metric uses a hard-threshold prefilter (Eq. 4).
- **§IV (fast realisation)**: predictive search, pre-computed sliding 2-D transforms, Kaiser windows are speed tricks. Skim on first reading.
- **§V (CBM3D)**: BM is performed in luminance only — chrominance is too noisy.
- **§VI Table II**: the 1-D transform along the stack direction matters far more than the 2-D transform — the deep insight that *inter-fragment* correlation is the real source of sparsity.
- **Common pitfall**: BM3D is *not* "group averaging". Each patch in a group receives its own estimate; aggregation then combines those.

---

## 7. 현대적 의의 / Modern Significance

### 한국어
BM3D는 약 10년간 모든 *학습 없는* denoising의 baseline이었으며, 2017년 DnCNN이 +0.35 dB 정도로 좁게 추월했지만 *training data가 부족한 도메인*(의료, 천문, 형광현미경)에서는 여전히 표준이다. 후속 V-BM4D(논문 #9)와 BM4D(논문 #10)는 BM3D의 4-D/volumetric 직접 확장이며, Restormer/ViT의 self-attention은 사실상 *학습된* BM3D similarity weights로 해석할 수 있다. 또한 PnP-ADMM, RED 등 plug-and-play priors에서 BM3D는 가장 자주 사용되는 denoiser regulariser이다 — *training-free*, *closed-form*, *strong prior*라는 세 가지 장점 때문.

### English
BM3D held SOTA among non-learned denoisers for about a decade; deep CNNs (DnCNN 2017) surpassed it by only ~0.35 dB, and BM3D remains the practical default whenever training data is scarce (medical, astronomical, fluorescence microscopy). V-BM4D (paper #9) and BM4D (paper #10) are direct 4-D/volumetric extensions. Self-attention in Restormer/ViT can be interpreted as a *learned* generalisation of BM3D's similarity weights. In Plug-and-Play ADMM and RED frameworks, BM3D is the most commonly invoked regulariser denoiser thanks to its three virtues: *training-free*, *closed-form*, *strong empirical prior*.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
