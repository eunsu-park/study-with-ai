---
title: "Pre-Reading Briefing: Gray and Color Image Contrast Enhancement by the Curvelet Transform"
paper_id: "33_starck_2003"
topic: Low_SNR_Imaging
date: 2026-05-06
type: briefing
---

# Gray and Color Image Contrast Enhancement by the Curvelet Transform: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: J.-L. Starck, F. Murtagh, E. J. Candès, D. L. Donoho, "Gray and Color Image Contrast Enhancement by the Curvelet Transform," *IEEE Transactions on Image Processing*, Vol. 12, No. 6, pp. 706–717, June 2003. DOI: 10.1109/TIP.2003.813140
**Author(s)**: Jean-Luc Starck, Fionn Murtagh, Emmanuel J. Candès, David L. Donoho
**Year**: 2003

---

## 1. 핵심 기여 / Core Contribution

이 논문은 곡선형 에지(curvilinear edge)를 다중스케일에서 표현하는 데 최적화된 *curvelet transform* 을 활용하여 영상의 대비를 강화하는 새로운 방법을 제시한다. 핵심은 (i) 잡음을 증폭하지 않고, (ii) 가장 약한 에지를 가장 크게, 가장 강한 에지는 보존하도록 휘어지는 비선형 매핑 함수 $y_c(x,\sigma)$ 를 curvelet 계수에 적용한 뒤, (iii) 역변환으로 영상을 재구성하는 것이다. Wavelet 기반 강조와 Multiscale Retinex (MSR), Histogram Equalization 과 비교했을 때, 잡음이 있는 곡선/에지 영상에서 curvelet 강조가 *edge detection 회수율* 과 *segmentation 충실도* 두 측면에서 더 좋은 결과를 보였다.

This paper presents a contrast-enhancement method based on the *curvelet transform*, which is matched to multiscale curvilinear edges. The key elements are: (i) not amplifying noise, (ii) applying a nonlinear coefficient-mapping function $y_c(x,\sigma)$ that boosts faint edges most while preserving strong edges, and (iii) reconstructing the image by inverse curvelet transform. Compared with wavelet enhancement, Multiscale Retinex (MSR) and histogram equalization, curvelet enhancement gives better edge-detection recovery and better segmentation fidelity on noisy edge-rich images.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

1990년대 후반에는 wavelet 기반 영상 처리 (Mallat 1989, Velde 1999) 가 대비 강화의 사실상 표준이었으나, isotropic wavelet basis 는 곡선/시트(sheet) 형태의 비등방 구조를 효율적으로 표현하지 못한다는 한계가 알려져 있었다. 이에 Candès & Donoho (1999, 2000) 가 *ridgelet* 과 *curvelet* 을 제안했는데, 이 논문은 수학적 이론이 막 정립된 curvelet 변환을 실용적인 image enhancement 작업에 처음으로 본격 적용한 사례이다.

By the late 1990s wavelet-based image processing (Mallat 1989, Velde 1999) was the *de facto* standard for contrast enhancement, but isotropic wavelet bases were known to be inefficient for curvilinear or sheet-like anisotropic structures. Candès & Donoho (1999, 2000) introduced *ridgelets* and *curvelets* to address this. This paper is one of the first practical applications of the (then-new) curvelet transform to image enhancement.

### 타임라인 / Timeline

```
1980 ─ Land's Retinex theory
1989 ─ Mallat dyadic wavelet
1996 ─ Jobson Multiscale Retinex (MSR)
1999 ─ Candès "Ridgelets"
1999 ─ Velde wavelet-based enhancement
2000 ─ Candès & Donoho first curvelet construction
2002 ─ Starck-Candès-Donoho curvelet denoising (TIP)
2003 ─ THIS PAPER: curvelet contrast enhancement
2006 ─ Second-generation (Fast Discrete) Curvelets
```

---

## 3. 필요한 배경 지식 / Prerequisites

- **Wavelet transform & "à trous" algorithm / 웨이블릿 변환 및 à trous 알고리듬**: 다중해상도 분해와 redundant wavelet 알고리듬에 대한 이해.
- **Radon transform / 라돈 변환**: ridgelet 정의의 기반이 되는 적분 변환.
- **Ridgelet & curvelet construction / Ridgelet 및 curvelet 구성**: 1-D wavelet on Radon slices, dyadic block partitioning, $\text{width}\approx\text{length}^2$ 스케일링 법칙.
- **Histogram equalization & MSR / 히스토그램 균등화 및 MSR**: 비교 대상이 되는 고전적 대비 강화 기법.
- **Noise standard deviation estimation / 잡음 표준편차 추정**: MAD-based estimator on finest wavelet scale.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| Curvelet transform | 비등방 곡선 구조를 표현하는 다중스케일 다중방향 변환 / Multiscale multi-directional transform tuned to anisotropic curvilinear structures |
| Ridgelet | Radon slice 위의 1-D wavelet — 직선/리지 구조를 표현 / 1-D wavelet on Radon slices, representing line/ridge features |
| à trous algorithm | 다운샘플링 없이 redundant wavelet pyramid 를 만드는 알고리듬 / Stationary wavelet algorithm without down-sampling, used for redundancy |
| Block partitioning | 영상을 $b\times b$ 중첩 블록으로 분할 후 각 블록에 ridgelet 적용 / Decomposition of an image into overlapping $b\times b$ blocks before applying ridgelet to each |
| $\text{width}\approx\text{length}^2$ | Curvelet 의 비등방 스케일링 법칙 / Anisotropic scaling law of curvelets |
| Enhancement function $y_c(x,\sigma)$ | 잡음을 증폭하지 않고 약한 에지를 부각하는 비선형 매핑 / Nonlinear coefficient mapping that boosts weak edges without amplifying noise |
| $c$, $m$, $p$, $s$ parameters | $c$=잡음 임계, $m$=강조 상한, $p$=비선형성, $s$=동적범위 압축 / $c$ noise threshold, $m$ saturation, $p$ nonlinearity, $s$ dynamic-range compression |
| MSR (Multiscale Retinex) | 다중스케일 SSR 의 가중합으로 색 항상성을 모사 / Weighted sum of single-scale retinex outputs modeling color constancy |
| MVI (subjective evaluation) | 주관적 영상 품질 평가 / Subjective image quality evaluation |
| Markov-Potts segmentation | 5-컴포넌트 Gaussian + Potts 공간 모형으로 분할 평가 / Five-component Gaussian + Potts model for segmentation evaluation |

---

## 5. 수식 미리보기 / Equations Preview

다중스케일 분해 (multiscale decomposition):

$$
I(x,y) = c_J(x,y) + \sum_{j=1}^{J} w_j(x,y)
$$

Wavelet 강조 함수 (Velde 1999):

$$
y(x) = \begin{cases} (m/c)^p, & |x|<c\\ (m/|x|)^p, & c\le|x|<m\\ 1, & |x|\ge m\end{cases}
$$

본 논문의 잡음 인지형 curvelet 강조 함수 (noise-aware curvelet enhancement, Eq. 10):

$$
y_c(x,\sigma) = \begin{cases}
1, & x<c\sigma\\
\dfrac{x-c\sigma}{c\sigma}\left(\dfrac{m}{c\sigma}\right)^p + \dfrac{2c\sigma-x}{c\sigma}, & c\sigma\le x<2c\sigma\\
(m/x)^p, & 2c\sigma\le x<m\\
(m/x)^s, & x\ge m
\end{cases}
$$

Ridgelet 정의 (Eq. 7):

$$
\psi_{a,b,\theta}(x_1,x_2) = a^{-1/2}\,\psi\!\left(\frac{x_1\cos\theta + x_2\sin\theta - b}{a}\right)
$$

Radon 변환:

$$
Rf(\theta,t) = \iint f(x_1,x_2)\,\delta(x_1\cos\theta + x_2\sin\theta - t)\,dx_1\,dx_2
$$

(Math delimiters: `$...$` inline, `$$...$$` block — never `\(...\)` or `\[...\]`)

---

## 6. 읽기 가이드 / Reading Guide

- **Sec. I (Introduction)** — Retinex/SSR/MSR 와 wavelet 강조 (Velde) 를 차례대로 검토. Eq. (1)–(5) 까지가 prior art.
- **Sec. II (Curvelet transform)** — 핵심 수학 영역. Ridgelet 정의 (Eq. 6–9) → block partitioning → curvelet pyramid. Fig. 3 의 flowgraph 가 알고리듬 전체 구조를 설명한다.
- **Sec. III (Contrast enhancement)** — Eq. 10 의 4-구간 비선형 함수가 본 논문의 핵심. 파라미터 $c, m, p, s$ 의 역할을 정확히 이해할 것. 컬러 영상 확장 (LUV 공간, gradient norm $e=\sqrt{c_L^2+c_u^2+c_v^2}$) 도 본 절에 포함.
- **Sec. IV (Evaluation)** — Edge detection (Fig. 9–10), segmentation (Fig. 11–15) 으로 정량 평가. SNR=2 일 때 wavelet 54.77% vs curvelet (new) 73.91% 같은 수치를 노트할 것.
- **Sec. V (Conclusion)** — 영상에 잡음이 있을 때 curvelet 이 명확한 우위; 잡음 없는 영상에서는 wavelet 과 비슷.

- **Sec. I (Introduction)** — Sequentially reviews Retinex/SSR/MSR and wavelet enhancement (Velde). Eq. (1)–(5) cover prior art.
- **Sec. II (Curvelet transform)** — Core math. Ridgelet (Eq. 6–9) → block partitioning → curvelet pyramid; the Fig. 3 flowgraph shows the whole architecture.
- **Sec. III (Contrast enhancement)** — The 4-piece nonlinear function (Eq. 10) is the heart of the paper. Be sure you understand the roles of $c, m, p, s$. The color extension via LUV space and gradient norm $e=\sqrt{c_L^2+c_u^2+c_v^2}$ is included here.
- **Sec. IV (Evaluation)** — Quantitative comparison via edge detection (Fig. 9–10) and segmentation (Fig. 11–15). Note specific numbers like SNR=2 wavelet 54.77% vs new curvelet 73.91%.
- **Sec. V (Conclusion)** — Curvelet wins clearly on noisy images; comparable to wavelet when noise-free.

---

## 7. 현대적 의의 / Modern Significance

Curvelet 은 이후 fast discrete curvelet transform (FDCT, Candès et al. 2006) 으로 발전하여 의료영상, 지진 자료, 그리고 무엇보다 **태양 코로나의 약한 곡선 구조 (CME, streamer)** 검출에 활용되었다. 본 논문이 정의한 *noise-aware enhancement curve* 는 LASCO/Mauna Loa K-coronameter 같은 저-SNR 코로나그래프 영상에서 CME 가시화를 향상시키는 표준 도구가 되었으며, NRGF (Normalized Radial Graded Filter), MGN (Multi-scale Gaussian Normalization, Morgan & Druckmüller 2014) 등의 코로나 enhancement 알고리듬의 직접적 선조이다. 또한 sparse representation-based denoising / inpainting / deconvolution 에서 ridgelet/curvelet dictionary 가 isotropic wavelet 보다 우수함을 실증한 이정표로도 평가된다.

Curvelets later evolved into the fast discrete curvelet transform (FDCT, Candès et al. 2006) and were adopted in medical imaging, seismic data analysis and — most relevantly here — in detecting **faint curvilinear features in the solar corona (CMEs, streamers)**. The *noise-aware enhancement curve* defined in this paper became a standard tool for visualizing CMEs in low-SNR coronagraph imagery (LASCO, Mauna Loa K-coronameter) and is a direct ancestor of NRGF and MGN (Multi-scale Gaussian Normalization, Morgan & Druckmüller 2014). The paper is also a milestone in showing that ridgelet/curvelet dictionaries outperform isotropic wavelets for sparse-representation-based denoising / inpainting / deconvolution.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
