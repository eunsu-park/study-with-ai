---
title: "Pre-Reading Briefing: Multi-Scale Gaussian Normalization for Solar Image Processing"
paper_id: "38_morgan_2014"
topic: Low_SNR_Imaging
date: 2026-05-06
type: briefing
---

# Multi-Scale Gaussian Normalization for Solar Image Processing: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Morgan, H. & Druckmüller, M. (2014). *Multi-Scale Gaussian Normalization for Solar Image Processing.* Solar Physics, **289**(8), 2945–2955. DOI: 10.1007/s11207-014-0523-9
**Author(s)**: Huw Morgan, Miloslav Druckmüller
**Year**: 2014

---

## 1. 핵심 기여 / Core Contribution

이 논문은 SDO/AIA 같은 EUV 영상의 **막대한 동적 범위(dynamic range)** — 활동 영역의 매우 밝은 작은 영역이 영상 전체 대비 범위를 지배해 어두운 미세구조를 보이지 않게 만드는 — 를 동시에 해결하면서 **계산적으로 매우 효율적** 인 단순한 영상 강조 기법, **Multi-Scale Gaussian Normalization (MGN)** 을 제시한다. 핵심 아이디어는 (i) 여러 폭 $w \in \{1.25, 2.5, 5, 10, 20, 40\}$ pixel 의 Gaussian kernel 로 **국지 평균과 표준편차** 를 계산해 **국지 정규화** 하고 (식 1–2), (ii) arctan 변환으로 출력 값 범위를 통제하며 (식 3), (iii) 전역 γ-변환 영상과 가중 평균으로 결합 (식 4–5) 하는 세 단계로, $w \gtrsim 3$ 픽셀에서 국지 표준편차가 안정화되는 점을 이용해 잡음 영역을 자동으로 평탄화하면서도 큰 스케일 맥락을 유지한다. 4096×4096 AIA 전체 영상을 ∼40 초에 처리할 만큼 빠르고, NAFE 같은 비교 기법보다 **1 차 자릿수 이상 빠르다**.

The paper presents **Multi-Scale Gaussian Normalization (MGN)** — a simple yet powerful image-enhancement technique designed to handle the **enormous dynamic range** of EUV solar images (where a few very bright active-region pixels dominate the contrast and hide faint structure elsewhere) while remaining **computationally cheap** enough for routine SDO/AIA processing. The recipe has three steps: (i) **local Gaussian normalisation** — at each spatial scale $w \in \{1.25, 2.5, 5, 10, 20, 40\}$ pixels, the image is normalised by its Gaussian-weighted local mean and standard deviation (Eqs. 1–2); (ii) **arctan transformation** (Eq. 3) to control the output range and prevent saturation; and (iii) **weighted recombination** with a global γ-transformed image (Eqs. 4–5) to retain large-scale context. The procedure intrinsically flattens noisy regions because the local standard deviation stabilises for $w \gtrsim 3$ pixels, processes a 4096×4096 AIA image in ∼ 40 s on a laptop, and is **an order of magnitude faster** than NAFE while delivering comparable visual quality.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

EUV 코로나 영상은 동적 범위가 극단적이다. 활동 영역의 가장 밝은 25 % 픽셀이 전체 영상 대비를 점유하고, 어두운 영역(quiet Sun, off-limb)의 미세구조는 사실상 보이지 않는다. 단순 sqrt/log/γ 변환은 빠르지만 한계가 있고, 시간 차분(time-differencing)은 정적 구조를 죽인다. 이전 세대 코로나 영상 강조 기법은 두 갈래로 발전했다: (a) **wavelet 기반 다중스케일 처리** (Stenborg & Cobelli 2003; Stenborg, Vourlidas & Howard 2008) — 매우 좋은 결과지만 계산 비용이 크다. (b) **국지 적응 normalization**, 특히 Druckmüller (2013) 의 NAFE (Noise Adaptive Fuzzy Equalization) — 매우 깨끗한 결과지만 역시 계산이 무겁다. SDO/AIA 가 11 초 cadence 로 4096² 영상을 쏟아내는 시대에 **계산 효율** 이 결정적 변수가 됐다. MGN 은 wavelet 패키지의 다중스케일 사고와 NAFE 의 국지 적응 사고를 하나로 묶되, **separable Gaussian filtering** 만으로 구현해 드라마틱하게 빠르다.

EUV coronal images have an extreme dynamic range — the brightest 25% of active-region pixels dominate the full contrast budget, and faint quiet-Sun / off-limb structure becomes invisible. Simple sqrt / log / γ transforms are fast but limited; time-differencing kills static structure. Previous coronal image-enhancement techniques split into two lineages: (i) wavelet-based multiscale processing (Stenborg & Cobelli 2003; Stenborg, Vourlidas & Howard 2008) — very high quality but computationally heavy; (ii) locally-adaptive normalisation, especially Druckmüller's NAFE (2013) — very clean output but again expensive. With SDO/AIA delivering 4096² frames at 11 s cadence, computational efficiency became decisive. MGN unites the multiscale and locally-adaptive ideas using only **separable Gaussian filtering**, achieving wavelet-quality output an order of magnitude faster.

### 타임라인 / Timeline

```
1996 ─ EIT (SOHO) — first EUV coronal imaging at high cadence
2003 ─ Stenborg & Cobelli — wavelet packet equalisation for EIT
2006 ─ Morgan, Habbal & Woo — NRGF (Normalising Radial Graded Filter, white-light)
2008 ─ Stenborg, Vourlidas & Howard — wavelet enhancement for EUVI/STEREO
2010 ─ SDO/AIA launched: 4096² @ 11 s, six EUV channels
2011 ─ Druckmüllerová, Morgan, Habbal — Fourier normalising-radial filter
2012 ─ SDO archive grows; image-processing efficiency now critical
2013 ─ Druckmüller — NAFE (Noise Adaptive Fuzzy Equalization)
2014 ─ ★ Morgan & Druckmüller — MGN (THIS PAPER, Solar Phys. 289, 2945)
2014+ MGN adopted by Helioviewer, JHelioviewer, AIA pipelines, IRIS / Hi-C / SWAP / EUI
2020+ Compared / paired with deep-learning EUV super-resolution / denoising
```

---

## 3. 필요한 배경 지식 / Prerequisites

- **2-D Gaussian filtering**: separable convolution $B \otimes k_w(x,y) = (B \otimes k_w^{1D})$ along rows then columns. / Gaussian 필터링과 separable 합성.
- **Local statistics**: weighted local mean and weighted local standard deviation. / 국지 평균과 표준편차.
- **γ correction**: $C_g = ((B-a_0)/(a_1-a_0))^{1/\gamma}$, with $\gamma$ typically 2.5–4. / γ 보정.
- **Histogram equalisation (analogy)**: MGN is similar in spirit to adaptive histogram equalisation (CLAHE). / 적응 히스토그램 평활화와의 유비.
- **EUV imaging characteristics**: Poisson photon noise, broadband line emission, exposure-time normalisation. / EUV 영상의 통계적 특성.
- **Scale-space theory** (Lindeberg 1994): how features at different physical scales are revealed at different Gaussian widths. / 스케일 스페이스 이론.
- **NRGF / NAFE** (Morgan-Habbal-Woo 2006; Druckmüller 2013): predecessors that motivated MGN. / 선행 기법.
- **Paper #35 (NRGF)**: scale-space-aware single-image enhancement for white-light corona. / 백색광 enhancement.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **MGN** | Multi-Scale Gaussian Normalization — 본 논문 알고리즘의 약어. / The paper's algorithm. |
| **Local mean (Gaussian-weighted)** | $B \otimes k_w$, 너비 $w$ 의 Gaussian kernel 로 weighted 평균. / Convolution with Gaussian kernel. |
| **Local standard deviation $\sigma_w$** | $\sqrt{\big[(B - B\otimes k_w)^2\big] \otimes k_w}$, weighted local std. / 가중 국지 표준편차. |
| **Locally normalised image $C$** | $C = (B - B\otimes k_w)/\sigma_w$, 평균 0 표준편차 1 로 정규화. / Mean-0, std-1 normalisation. |
| **arctan compression $C'$** | $C' = \arctan(kC)$, 출력 범위 통제. / Output range control. |
| **γ-corrected global image $C_g$** | $C_g = ((B-a_0)/(a_1-a_0))^{1/\gamma}$, 전역 톤 매핑. / Global tone mapping. |
| **Final blend $I$** | $I = h\,C_g + ((1-h)/n)\sum_i g_i C_i'$, 전역 + 국지 가중 평균. / Final weighted blend. |
| **Scale weights $g_i$** | 작은 $w$ 에서는 작게, $w \gtrsim 3$ 에선 1 에 수렴. 잡음 평탄화의 핵심. / Scale weights damp noise at fine scales. |
| **Global weight $h$** | 전역 톤 (γ-변환) 의 비중, 본 논문 기본값 0.7. / Global-tone weight. |
| **NAFE** | Noise Adaptive Fuzzy Equalization (Druckmüller 2013), MGN 의 직접적 선행. / Direct predecessor. |
| **NRGF** | Normalising Radial Graded Filter (Morgan-Habbal-Woo 2006), 백색광 코로나용. / White-light predecessor. |
| **Wavelet packet equalisation** | Stenborg-Cobelli (2003) 의 고비용 다중스케일 enhancement. / Wavelet baseline. |

---

## 5. 수식 미리보기 / Equations Preview

**(1) 국지 정규화 / Local Gaussian normalisation**

$$
C = \frac{B - B \otimes k_w}{\sigma_w}, \qquad \sigma_w = \sqrt{\big[(B - B\otimes k_w)^2\big] \otimes k_w}.
$$

너비 $w$ 의 Gaussian kernel $k_w$ 로 평균을 빼고 국지 표준편차로 나눈다. 결과 $C$ 는 평균 0, 표준편차 1. / Subtract Gaussian-weighted mean, divide by Gaussian-weighted std.

**(2) arctan 변환 / arctan compression**

$$
C' = \arctan(k\, C),\qquad k \approx 0.7.
$$

$C$ 의 분포가 $\sim \mathcal N(0, 1)$ 에 가까우므로, arctan 은 0 부근을 증폭하고 양 극단을 압축한다. 출력 saturation 방지. / Amplify near 0, compress tails — prevents saturation.

**(3) 전역 γ-보정 / Global γ correction**

$$
C_g = \left(\frac{B - a_0}{a_1 - a_0}\right)^{1/\gamma},\qquad \gamma \in [2.5, 4],\;\text{paper uses}\;\gamma = 3.2.
$$

$a_0, a_1$ 는 입력 최소/최대. 전역 톤 매핑. / Global tone mapping.

**(4) 최종 결합 / Final blend**

$$
I = h\,C_g + \frac{1-h}{n}\sum_{i=1}^{n} g_i\, C_i',\qquad h = 0.7.
$$

전역 γ 영상에 가중치 $h$, 그리고 $n$ 개의 다중스케일 국지 정규화 영상 $C_i'$ 의 가중 평균에 가중치 $1-h$. / Weighted blend.

**(5) 스케일 가중치 / Scale weights**

작은 Gaussian kernel ($w<3$) 에서 평균 국지 표준편차 $\langle\sigma_w\rangle$ 는 전역 표준편차의 약 60 % 까지 떨어진다 (Fig. 4 of paper). 이 영역은 noise-dominated 이므로 $g_i$ 를 작게 (∼0.6) 두어 잡음 증폭을 막는다. $w \gtrsim 3$ 에선 $\langle\sigma_w\rangle \to 1$ 이므로 $g_i \to 1$. / Damp small-scale, noise-dominated kernels.

(Math delimiters: `$...$` inline, `$$...$$` block.)

---

## 6. 읽기 가이드 / Reading Guide

1. **Section 1 — Introduction (pp. 2945–2946).** EUV 영상 처리의 근본적 어려움 (동적 범위, 시간 차분의 한계, wavelet 의 비용) 을 설명. / Read for the motivation: dynamic range and computational cost.
2. **Section 2 — Observations (pp. 2946–2948).** 전형적 AIA 171 Å 영상의 픽셀 값 분포 분석. quiet Sun / active-region base / active region / off-limb 네 영역의 히스토그램 비교 (Fig. 2) 가 결정적. / The four-histogram comparison defines the problem.
3. **Section 3 — Method (pp. 2948–2950).** 식 (1)–(5), Fig. 3, Fig. 4. **이 섹션이 곧 알고리즘 명세** 이며, pseudocode (10 단계) 가 정확히 적혀 있어 그대로 구현 가능. / This section is the implementation spec.
4. **Section 4 — Results (pp. 2951–2954).** AIA 171 Å, Hi-C, SWAP, LASCO C2 네 종류 영상에 적용한 결과. 백색광 코로나에까지 적용 가능함을 보여 NRGF 와 비교한다. / The four worked examples; note the comparison to NRGF for LASCO.
5. **Section 5 — Summary (p. 2955).** 처리 시간 및 NAFE 와의 비교. **MacBook Pro Core i7, 8 GB RAM 으로 4096² AIA 영상 ∼40 초** — 핵심 수치. / The 40-second AIA timing is the key practical claim.

---

## 7. 현대적 의의 / Modern Significance

**EUV 영상 강조의 사실상 표준.** SDO/AIA, STEREO/EUVI, Solar Orbiter EUI, PROBA-2/SWAP, IRIS, Hi-C 영상의 시각화에서 MGN 은 Helioviewer, JHelioviewer, SunPy/IDL Solarsoft 의 기본 enhancement 옵션이다. 단순한 separable Gaussian 만 사용하므로 GPU/SIMD 구현이 쉽고, 오프라인 + 온라인 모두 활용된다. **Coronagraph (백색광) 에까지 확장** 가능해 LASCO C2/C3, SECCHI COR1/COR2, Metis 시각화에도 사용된다. 이후 deep-learning EUV super-resolution / denoising 모델 (e.g., Park et al. 2020, Lim et al. 2021) 의 visualisation pipeline 의 베이스라인으로도 자리 잡았다.

**De-facto standard for EUV image visualisation.** MGN is the default enhancement option in Helioviewer, JHelioviewer, SunPy and IDL Solarsoft for SDO/AIA, STEREO/EUVI, Solar Orbiter EUI, PROBA-2/SWAP, IRIS and Hi-C imagery. Because it uses only separable Gaussian filtering, it is trivially GPU-/SIMD-parallelisable, suitable for both offline post-processing and on-line displays. It also extends to **white-light coronagraphs** — LASCO C2/C3, SECCHI COR1/COR2, Metis — and serves as the visualisation baseline for modern deep-learning EUV super-resolution / denoising work (Park et al. 2020; Lim et al. 2021).

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
