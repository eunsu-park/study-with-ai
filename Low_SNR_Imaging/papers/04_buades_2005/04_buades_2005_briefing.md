---
title: "Pre-Reading Briefing: A Non-Local Algorithm for Image Denoising"
paper_id: "04_buades_2005"
topic: Low_SNR_Imaging
date: 2026-05-06
type: briefing
---

# A Non-Local Algorithm for Image Denoising: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Buades, A., Coll, B., & Morel, J.-M., "A Non-Local Algorithm for Image Denoising", *Proc. IEEE CVPR 2005*, Vol. 2, pp. 60–65 (2005). [DOI: 10.1109/CVPR.2005.38]
**Author(s)**: Antoni Buades, Bartomeu Coll, Jean-Michel Morel
**Year**: 2005

---

## 1. 핵심 기여 / Core Contribution

### 한국어
이 논문은 영상 노이즈 제거 분야에 두 가지 결정적 기여를 한다. (A) **Method noise** — 노이즈 제거 알고리즘 $D_h$ 의 품질을 정량 평가하는 새로운 척도. *거의 노이즈 없는* 영상 $u$ 에 알고리즘을 적용했을 때의 차이 $u - D_h u$ 를 *method noise* 로 정의 (Definition 1). 좋은 알고리즘은 method noise 가 백색잡음처럼 보여야 하며, 구조(에지·텍스처)가 method noise 에 남으면 알고리즘이 *그 구조를 부수고 있다* 는 의미. Theorem 1–3 으로 Gaussian, anisotropic, TV, neighborhood 필터의 *수학적 한계* 를 명시. (B) **Non-Local Means (NL-means)** — 영상의 **자기 유사성(self-similarity)** 을 활용. 픽셀 $i$ 의 추정값은 영상 전체에서 $i$ 의 주변 patch $\mathcal N_i$ 와 비슷한 patch $\mathcal N_j$ 를 가진 모든 픽셀 $j$ 의 가중 평균. patch 비교가 픽셀 비교보다 잡음에 강건하면서 에지·텍스처를 보존한다. (C) **Consistency theorem (Thm 4)**: stationarity 하에 NL-means 가 *조건부 기댓값* $E[Y_i|X_i]$ 에 수렴 → MSE 최적.

### English
Two decisive contributions plus a consistency result. (A) **Method noise**: a novel quality measure — apply $D_h$ to a *nearly noiseless* image $u$ and inspect the difference $u - D_h u$. Good algorithms produce method noise resembling white noise; structures (edges, texture) appearing in the method noise reveal what the algorithm is destroying. Theorems 1–3 derive method-noise expressions for Gaussian / anisotropic / TV / neighborhood filters, exposing their fundamental limitations. (B) **Non-Local Means (NL-means)**: exploits image **self-similarity** by replacing each pixel with a weighted average over the entire image, with weights based on Gaussian-weighted $L^2$ patch similarity. Patch matching is noise-robust and preserves textures/edges. (C) **Consistency (Theorem 4)**: under stationarity, NL-means converges to the conditional expectation $E[Y_i|X_i]$ — the MSE-optimal Bayes estimator.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting
2005년까지 영상 denoising 의 모든 주류 기법은 *국소(local)*: Gaussian (Gabor 1960), anisotropic diffusion (Perona-Malik 1990), TV (Rudin-Osher-Fatemi 1992), bilateral (Tomasi-Manduchi 1998), wavelet shrinkage (paper #1–3). 이들 모두 작은 근방 안에서 평균하므로 *국소 미분 연산자 한계* 에 갇혀 있었다. 한편 Efros-Leung (1999) 의 *texture synthesis by non-parametric sampling* 이 *영상 내 patch 가 영상 어디에선가 비슷한 모양으로 반복된다* 는 *self-similarity* 를 명시적으로 활용했지만, denoising 에는 적용되지 않았다. 이 논문은 두 흐름을 결합해 *비국소 평균* 을 denoising 에 가져왔고, BM3D, K-SVD non-local, 그리고 transformer 의 self-attention 까지 이어지는 *non-local 시대* 를 열었다.
By 2005 every mainstream denoiser was *local*: Gaussian (Gabor 1960), anisotropic diffusion (Perona-Malik 1990), TV (Rudin-Osher-Fatemi 1992), bilateral (Tomasi-Manduchi 1998), wavelet shrinkage (#1–3) — all averaged within a small neighborhood. Meanwhile Efros-Leung's (1999) *texture synthesis by non-parametric sampling* exploited the fact that *patches recur throughout natural images*, but no one had applied this to denoising. This paper bridged the two, opening the *non-local era* leading to BM3D, K-SVD non-local sparse coding, and ultimately transformer self-attention.

### 타임라인 / Timeline
```
1960 ─── Gabor — Gaussian smoothing
1985 ─── Yaroslavsky — neighborhood filter (intensity-similarity)
1990 ─── Perona-Malik — anisotropic diffusion
1992 ─── Rudin-Osher-Fatemi — Total Variation
1994 ─── Donoho-Johnstone — wavelet shrinkage (paper #1)
1998 ─── Tomasi-Manduchi — Bilateral filter
1999 ─── Efros-Leung — texture synthesis by non-parametric patch sampling
2005 ★★ BUADES-COLL-MOREL — Non-Local Means (THIS PAPER)
2007 ─── Dabov+ — BM3D (paper #7) = NLM block-matching + transform shrinkage
2017 ─── Vaswani+ — Transformer self-attention = learned NL-means weights
2022 ─── Restormer / SwinIR — transformer denoisers
```

---

## 3. 필요한 배경 지식 / Prerequisites

- **영상 노이즈 모델 / Image noise model**: $v(i) = u(i) + n(i)$, $n \sim N(0, \sigma^2)$ iid.
- **Patch 표현 / Patch representation**: 픽셀 주변 $P\times P$ window 를 vector 로.
- **가중 평균과 kernel / Kernel-weighted averaging**: $\sum w_j v_j$, $w_j \ge 0$, $\sum w_j = 1$.
- **국소 PDE 기반 필터 / Local PDE-based filters**: Gaussian, anisotropic diffusion, TV minimisation 의 개념적 이해.
- **조건부 기댓값 / Conditional expectation**: $E[Y|X]$ 의 MSE 최적성.
- **Stationary mixing field / Stationary mixing field**: 일관성 정리의 가정.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| Method noise | $u - D_h u$ on a clean image — 알고리즘 진단 도구 / Algorithm-diagnostic residual on a clean image. |
| Self-similarity | 자연 영상의 patch 가 영상 곳곳에 반복 / Natural-image patches recur throughout the image. |
| Non-Local Means (NL-means) | patch 유사도 기반 가중 평균 / Weighted average with patch-similarity weights. |
| Patch ($\mathcal{N}_i$) | 픽셀 $i$ 중심의 정사각 window (보통 $7\times 7$) / Square window centred at $i$ (typically $7\times 7$). |
| Search window | 매칭 patch 를 찾는 영역 (보통 $21\times 21$) / Region searched for matching patches. |
| Filter strength $h$ | 가중치 kernel 폭, $\approx 10\sigma$ / Kernel width, ≈ $10\sigma$. |
| Gaussian-weighted $L^2$ distance | $\sum_k g_a(k)(v(i+k) - v(j+k))^2$ / Patch distance weighted by a Gaussian over offsets. |
| Bilateral filter | intensity-similarity neighborhood filter (Tomasi-Manduchi) / Single-pixel intensity-similarity averaging. |
| Anisotropic diffusion | Perona-Malik gradient-conditioned smoothing / Gradient-aware local smoothing PDE. |
| Total variation (TV) | Rudin-Osher-Fatemi convex regularisation / Convex edge-preserving regulariser. |
| Conditional expectation | $E[Y_i \| X_i]$ — MSE-optimal Bayes estimator. |
| Block matching | NLM 과 BM3D 가 공유하는 patch 검색 단계 / Patch-search step shared by NLM and BM3D. |

---

## 5. 수식 미리보기 / Equations Preview

**Method noise (Definition 1)**:
$$
\text{Method noise}(D_h, u) = u - D_h u
$$
좋은 알고리즘에서는 white-noise 처럼 보여야 함. 구조가 보이면 그 구조를 알고리즘이 흐리고 있다는 신호.
A good algorithm produces white-noise-like residuals; visible structure indicates blurring.

**Theorems 1–3** — local-filter 한계의 정리:
$$
u - G_h * u = -h^2\Delta u + o(h^2) \quad \text{(Gaussian)}
$$
$$
u - AF_h\,u = -\tfrac{1}{2}h^2 |Du|\,\mathrm{curv}(u) + o(h^2) \quad \text{(anisotropic, Perona-Malik)}
$$
$$
u - TVF_\lambda\,u = -\frac{\mathrm{curv}(TVF_\lambda u)}{2\lambda} \quad \text{(TV, Rudin-Osher-Fatemi)}
$$
모두 *국소 미분 연산자* 를 method noise 에 가짐 → edge·texture 파괴.
All have local-differential method noise → destroy edges/textures.

**NL-means (the algorithm)** — 핵심:
$$
NL[v](i) = \sum_{j} w(i,j)\,v(j), \qquad w(i,j) = \frac{1}{Z(i)}\exp\left(-\frac{\|v(\mathcal N_i) - v(\mathcal N_j)\|^2_{2,a}}{h^2}\right)
$$
$$
Z(i) = \sum_j \exp\left(-\|v(\mathcal N_i) - v(\mathcal N_j)\|^2_{2,a}/h^2\right), \quad \|v(\mathcal N_i) - v(\mathcal N_j)\|^2_{2,a} = \sum_k g_a(k)(v(i+k) - v(j+k))^2
$$

**잡음 강건성 / Noise robustness**:
$$
E\|v(\mathcal N_i) - v(\mathcal N_j)\|^2_{2,a} = \|u(\mathcal N_i) - u(\mathcal N_j)\|^2_{2,a} + 2\sigma^2
$$
잡음 항은 *모든 distance 에 균일하게 더해져* 순서가 보존됨.
The noise term is a *constant* added to every distance, preserving the ordering of similarities.

**Consistency (Theorem 4)** — MSE 최적성:
$$
NL_n(j) \xrightarrow{\text{a.s.}} E[Y_j | X_j = v(\mathcal N_j \setminus \{j\})]
$$

---

## 6. 읽기 가이드 / Reading Guide

### 한국어
- **§1 (Intro)**: 모든 기존 방법이 *국소* 평균이라는 사실, 그리고 *비국소* 가 새로움이라는 점만.
- **§2 (Method noise)**: 핵심 분석 도구. **Theorems 1–3** 의 *결과 식* 만 외우기 (공식 유도는 first pass 에서 skim). Fig. 4 의 method noise 비교는 시각으로 확인.
- **§3 (NL-means)**: 페이퍼의 심장. *식 한 줄* — $w(i,j) = e^{-\|\cdot\|^2/h^2}/Z$ 만 정확히 이해하면 끝. Patch distance 의 가우시안 가중 부분은 *중심 픽셀 강조* 의 기능.
- **§4 (Consistency)**: 첫 읽기엔 "stationarity 하에 conditional expectation 으로 수렴" 결과만. 증명은 second pass.
- **§5 (Experiments)**: Table 1 (MSE 비교), Fig. 4 (method noise), Fig. 5 (시각). Lena $\sigma = 20$: NL-means 68 vs TV 110 등.
- **흔한 걸림돌**: (i) self-similarity 는 *완전한 반복* 이 아니라 *대략적 유사성*. (ii) $w(i,i) = 1$ (자기 자신과 거리 0) 이 너무 큰 가중치를 주는 문제 — 실제 구현에서 $w(i,i)$ 를 *주변의 max* 로 대체하는 트릭. (iii) 이론은 *영상 전체 검색* 을 요구하지만 실용적으론 $21\times 21$ search window 로 잘림 — 이게 NL-means 의 $O(N^2)$ 비용을 $O(N\cdot S^2)$ 로 줄임. (iv) $h \approx 10\sigma$ 가 경험 최적 — 너무 작으면 매칭 부족, 너무 크면 over-averaging.

### English
- **§1 Introduction**: take only that all prior methods average *locally* and the new idea is *non-local*.
- **§2 Method noise**: the core diagnostic. Memorise the *result formulas* of Theorems 1–3 (skim the derivations on first pass). Fig. 4 visualises the method-noise comparison.
- **§3 NL-means**: the heart of the paper. The single equation $w(i,j) = e^{-\|\cdot\|^2/h^2}/Z$ is enough; the Gaussian patch-weighting just emphasises central pixels.
- **§4 Consistency**: take "under stationarity, NL-means converges to the conditional expectation" — proof can wait.
- **§5 Experiments**: Table 1 (MSE comparison), Fig. 4 (method noise), Fig. 5 (visual). Lena $\sigma=20$: NL-means MSE 68 vs TV 110, ~2.1 dB PSNR margin.
- **Common stumbling blocks**: (i) self-similarity is *approximate*, not exact repetition. (ii) $w(i,i) = 1$ (self-distance zero) gives too much weight to $i$ itself — practical implementations replace $w(i,i)$ with the max of other weights. (iii) The theory requires search over the entire image, but in practice a $21\times 21$ window cuts cost from $O(N^2)$ to $O(N\cdot S^2)$. (iv) $h \approx 10\sigma$ is empirically optimal — too small misses matches, too large over-averages.

---

## 7. 현대적 의의 / Modern Significance

### 한국어
NL-means 는 *비국소(non-local) 시대* 의 결정적 분기점이다. **BM3D** (paper #7) 는 NL-means 의 patch grouping 과 transform-domain shrinkage (paper #1–3) 를 결합한 직계 후계자. **K-SVD non-local sparse coding** (Mairal+ 2009), **WNNM** (Gu+ 2014) 도 모두 NL-means 의 patch 매칭 단계를 활용. 더 흥미롭게, **transformer 의 self-attention** 은 *학습된 NL-means weights* — query-key 유사도가 patch distance 를, softmax 가 exponential kernel 을 대체. 결국 **Restormer**, **SwinIR**, **NAFNet** 같은 최첨단 image-restoration 모델은 모두 NL-means 의 *비국소 평균* 사상을 이어받는다. Method noise 라는 진단 도구도 여전히 유효 — deep denoiser 의 residual 을 보면 무엇을 학습했는지 진단 가능. 천체관측·플라즈마 imaging 같은 low-SNR 영역에서 NL-means + 변형 (NLM-Poisson) 은 여전히 baseline 으로 사용된다.

### English
NL-means is the decisive pivot to the *non-local era*. **BM3D** (#7) is the direct successor — combining NLM's block matching with transform-domain shrinkage (#1–3). **K-SVD non-local sparse coding** (Mairal+ 2009) and **WNNM** (Gu+ 2014) all adopt NLM's patch-matching step. More remarkably, **transformer self-attention** is *learned NL-means weights* — query-key similarity replaces patch distance, softmax replaces the exponential kernel. State-of-the-art restoration models like **Restormer**, **SwinIR**, and **NAFNet** all inherit NL-means' *non-local averaging* paradigm. Method noise as a diagnostic remains useful — inspecting a deep denoiser's residual reveals what it has learned. In low-SNR astrophysical and plasma imaging, NLM and its variants (NLM-Poisson) remain standard baselines.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
