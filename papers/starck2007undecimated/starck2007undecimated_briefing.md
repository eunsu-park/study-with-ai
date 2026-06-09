---
title: "Pre-Reading Briefing: The Undecimated Wavelet Decomposition and its Reconstruction"
paper_id: "36_starck_2007"
topic: Low_SNR_Imaging
date: 2026-05-06
type: briefing
---

# The Undecimated Wavelet Decomposition and its Reconstruction: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Starck, J.-L., Fadili, J., & Murtagh, F. (2007). The Undecimated Wavelet Decomposition and its Reconstruction. *IEEE Transactions on Image Processing*, **16**(2), 297–309. DOI: 10.1109/TIP.2006.887733
**Author(s)**: Jean-Luc Starck, Jalal Fadili, Fionn Murtagh
**Year**: 2007

---

## 1. 핵심 기여 / Core Contribution

이 논문은 **undecimated wavelet transform (UWT)** 의 두 표준 형태 — 일반(orthogonal-style) UWT 와 등방(isotropic) UWT (IUWT) — 의 관계를 정형화하고, 비분할(redundant) 변환의 자유도를 활용해 **양(positive)의 합성 필터 뱅크** 를 설계해 wavelet 임계처리 후 흔히 발생하는 **ringing artifact** 를 크게 줄이는 새로운 필터 뱅크를 제시한다. 또한 임계처리(thresholding) 후 정확한 역변환이 불가능한 비등기(nondecimated) 표현의 한계를 우회하기 위한 **반복 (Landweber-style) 재구성 알고리즘** 을 정리한다.

This paper formalizes the relation between two standard undecimated wavelet transforms — the general UWT and the isotropic UWT (IUWT) — and exploits the redundancy of the decomposition to design new filter banks whose **synthesis filters are positive**, dramatically reducing the **ringing artifacts** that plague wavelet-based denoising. It also presents an **iterative reconstruction (Landweber/POCS) scheme** that gives a consistent reconstruction from a thresholded subset of redundant wavelet coefficients, which a single direct synthesis cannot achieve.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

1990 년대를 거치며 Daubechies 의 직교/이중직교 wavelet 과 Mallat 의 multiresolution 이론이 wavelet 변환을 정식화했고, JPEG2000 (DWT) 으로 상업화에 성공했다. 그러나 **비분할(downsampling) 단계의 손실로 인한 translation-variance** 가 디노이징·디컨볼루션·검출 같은 문제에서 강한 아티팩트를 만들어, 천문/생체 영상 커뮤니티는 일찌감치 redundant·shift-invariant 표현 (Holschneider 의 à-trous, Mallat 의 stationary WT) 으로 옮겨가 있었다. Starck et al. (2002) 은 등방 UWT 가 분할 wavelet 보다 디노이징에서 2.5 dB 이상 우수함을 보였다.

By the late 1990s the orthogonal/biorthogonal wavelet framework of Daubechies and Mallat had been packaged as JPEG2000. But the **loss of translation-invariance from the decimation step** was a known weakness for denoising, deconvolution, and detection. Astronomy and biomedical-imaging communities had already adopted redundant, shift-invariant representations (Holschneider's à-trous, Mallat's stationary WT). Starck et al. (2002) had quantified that thresholding on an undecimated transform gains > 2.5 dB over the decimated one in denoising.

### 타임라인 / Timeline

```
1989 ─ Mallat: orthogonal multiresolution analysis
1989 ─ Holschneider et al.: à-trous (with-holes) algorithm
1992 ─ Daubechies: Ten Lectures on Wavelets
1992 ─ Shensa: relation between à-trous and Mallat algorithms
1998 ─ Mallat: A Wavelet Tour of Signal Processing
2002 ─ Starck & Murtagh: IUWT in astronomical image analysis
2002 ─ Durand & Froment: TV regularisation of wavelet coefficients
2005 ─ Tropp et al.: alternating projection for designing tight frames
2006 ─ Steidl, Weickert et al.: equivalence of soft thresholding & TV
2007 ─ ★ Starck, Fadili, Murtagh — UWT/IUWT relation + positive
        synthesis filter bank + iterative reconstruction (THIS PAPER)
2009 ─ Starck, Murtagh & Fadili: "Sparse Image and Signal Processing"
```

---

## 3. 필요한 배경 지식 / Prerequisites

- **2-D separable wavelets (Mallat)**: scaling function $\phi$, wavelet $\psi$, analysis pair $(h,g)$, synthesis pair $(\tilde h, \tilde g)$. / 2-D 분리형 wavelet 의 스케일링/디테일 함수와 분석/합성 필터쌍.
- **z-transform & perfect-reconstruction filter banks**: $H(z^{-1})\tilde H(z) + G(z^{-1})\tilde G(z) = 1$. / z-변환과 완전재구성 조건.
- **à-trous algorithm**: insert $2^j-1$ zeros between filter taps; compute analysis without subsampling. / 필터 탭 사이에 0 을 삽입하는 비분할 알고리즘.
- **Frames vs. tight frames**: redundant representations $\alpha = \mathcal W S$ where $\mathcal R \mathcal W = I$ on the range of $\mathcal W$. / 프레임 이론.
- **Soft/hard thresholding & Donoho-Johnstone shrinkage**: $\Delta_T(\alpha)=\mathrm{sgn}(\alpha)\max(|\alpha|-T,0)$. / 임계처리.
- **POCS (Projection Onto Convex Sets) / Landweber iteration**: alternating projection for constrained inverse problems. / 볼록집합 사영법.
- **Paper #1 (Daubechies/Mallat)**: foundational orthogonal wavelet construction. / 직교 wavelet 의 기초.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **UWT (Undecimated Wavelet Transform)** | 분할(downsample)을 생략한 wavelet 변환. 각 스케일이 원영상과 같은 크기 → translation-invariant 하지만 redundant. / Wavelet transform without subsampling; every band has the original image size, redundant but shift-invariant. |
| **IUWT (Isotropic UWT)** | 1-D B-spline scaling 으로부터 만든 등방 2-D scaling 함수 사용. 한 스케일당 한 개의 detail band (방향성 없음). 천문 이미지에 적합. / 2-D isotropic scaling from B-spline; one detail band per scale, no orientation; suited to astronomical/biology images. |
| **à-trous algorithm** | 필터 탭 사이에 영을 끼워 ($h^{(j)}$) 다단 분해를 빠르게 수행. / "With holes" — insert zeros between taps to compute multiscale without down-sampling. |
| **Astro filter bank** | $h^{1D} = [1,4,6,4,1]/16$ (3차 B-spline 평활 필터), $g^{1D} = \delta - h^{1D}$. 천문에서 표준. / Standard cubic-spline analysis filter used in astronomy. |
| **Nonorthogonal positive synthesis filter bank** | 합성쪽 필터 $\tilde g$ 를 $\delta + h$ 로 잡아 양으로 만든 필터 뱅크 → ringing 강건. / Choose synthesis $\tilde g = \delta + h$ so that $\tilde g$ is positive everywhere → ringing-robust. |
| **Tight frame / non-tight frame** | $\sum_j |\hat h_j|^2 = c$ 이면 tight frame; 본 논문 필터 뱅크는 frame 이지만 일반적으로 non-tight. / If filter Fourier magnitudes sum to a constant, frame is tight; this paper's banks are generally non-tight frames. |
| **POCS reconstruction** | 다중분해 support 와 양 cone 등 볼록집합 위로 교대 사영. / Alternating projections onto multi-resolution support and positivity cones. |
| **Landweber iteration** | $\tilde S^{n+1} = \tilde S^n + \mathcal R [\alpha_T - \mathcal W \tilde S^n]$. / Gradient-descent-like iteration that converges in the range of $\mathcal W$. |
| **Multiresolution support $M$** | 임계처리에 의해 살아남는 계수의 마스크. / Mask of coefficients retained after thresholding. |
| **Ringing artifact** | 임계처리 후 신호의 불연속점 근방에 나타나는 진동. / Oscillations near edges/discontinuities in thresholded reconstructions. |
| **MCA (Morphological Component Analysis)** | 두 사전(예: DCT + UWT) 으로 신호를 두 형태소로 분해. / Split a signal into morphological components living in different dictionaries. |

---

## 5. 수식 미리보기 / Equations Preview

**(1) à-trous decomposition recursion / à-trous 분해 점화식**

$$
c_{j+1}[l] = (\bar h^{(j)} * c_j)[l] = \sum_k h[k]\, c_j[l + 2^j k], \qquad w_{j+1}[l] = (\bar g^{(j)} * c_j)[l]
$$

각 스케일에서 다운샘플 없이 `holes` 를 끼운 필터로 점진적으로 평활/디테일 분리. / At each scale the inserted-zero filter $h^{(j)}$ produces a coarser smooth and a detail without subsampling.

**(2) Exact-reconstruction condition / 완전재구성 조건**

$$
H(z^{-1})\tilde H(z) + G(z^{-1})\tilde G(z) = 1
$$

비분할 자유도가 커서 한 분석 쌍 $(h,g)$ 에 다수의 합성 쌍 $(\tilde h, \tilde g)$ 이 가능. / Many synthesis pairs satisfy this, giving design freedom.

**(3) Astro filter & nonorthogonal positive bank / 양의 비직교 합성 뱅크**

$$
h = h^{1D}*h^{1D},\;\; h^{1D}=\frac{[1,4,6,4,1]}{16},\;\; g = \delta - h*h, \;\; \tilde h = h, \;\; \tilde g = \delta
$$

이 선택에서 Section III-A 의 파생 결과 $\tilde g = \delta + h$ 가 모두 양수가 되어 ringing 이 사라진다. / With this choice the equivalent synthesis function is positive everywhere → no ringing.

**(4) Iterative reconstruction (Landweber) / 반복 재구성**

$$
\tilde S^{n+1} = P_{+}\!\left(\tilde S^n + \mathcal R\left[\alpha_T - \mathcal W \tilde S^n\right]\right)
$$

$\alpha_T$ 는 임계처리된 계수, $\mathcal W$ 는 분석, $\mathcal R$ 는 합성, $P_+$ 는 양 cone projection. / Thresholded coefficients are projected back consistently with the analysis operator and a positivity prior.

**(5) Three-direction decomposition reconstructs IUWT scale / 세 방향 합 = IUWT 디테일**

$$
w_j^{1} + w_j^{2} + w_j^{3} = c_{j-1} - c_{j}
$$

vertical / horizontal / diagonal 세 방향 detail 을 합치면 IUWT 의 등방 detail band 와 정확히 일치. / Summing the three directional detail bands of the standard UWT recovers the IUWT detail band exactly.

(Math delimiters: `$...$` inline, `$$...$$` block.)

---

## 6. 읽기 가이드 / Reading Guide

1. **Section II — UWT vs IUWT 의 정의 와 á-trous 점화식.** 식 (1), (4), (5) 에 집중. 두 변환이 동일한 분석 필터 뱅크에서 단지 그룹화 방식만 다르다는 점을 확인. / Read for definitions; confirm UWT and IUWT share the same filter bank but group bands differently.
2. **Section III — 핵심 기여(필터 뱅크 설계).** 비직교 + 양의 합성 필터 뱅크 식 (16)–(21). $\tilde g = \delta + h$ 가 어떻게 ringing 을 죽이는지 직관을 가져갈 것. / Core contribution; understand why $\tilde g = \delta + h$ kills ringing.
3. **Section IV — 반복 재구성.** POCS = Landweber 동치 결과(식 24–28) 와 양 제약. 후속 sparse-image-restoration 문헌의 표준 도구. / The Landweber/POCS equivalence (Eqs. 24–28) becomes the standard sparse-restoration tool.
4. **Section V — 실험.** Lena·truncated-Gaussian·square-with-noise·MCA bumps+sine 네 실험. Threshold-vs-MSE 곡선과 ringing 의 시각적 비교를 한 차례씩 따라가면 수식의 효과가 체감된다. / Walk through 4 experiments; the threshold-vs-MSE curves and the row-cut comparison make the effect tangible.
5. **마지막에 식 (29)–(30) 의 MCA 알고리즘** 을 확인하면 이후 #44 (Starck-Elad-Donoho 2005, MCA) 와 자연스럽게 연결된다. / Eqs. 29–30 (MCA) bridge to subsequent MCA papers.

---

## 7. 현대적 의의 / Modern Significance

**천문 영상 처리의 산업 표준.** ISAP, MR/1, Sparse2D 등 천문 wavelet 패키지 모두 IUWT (식 11–12) 를 채택했고, NRGF·MGN 같은 코로나 영상 enhancement 와 결합해 SDO/AIA, STEREO/EUVI, Solar Orbiter EUI 의 이미지 처리 기본 도구가 되었다. 또한 양의 합성 필터 + iterative thresholding 아이디어는 이후 **deep learning 의 multiscale/U-Net 구조 (LISTA, learned wavelet)** 및 천문 ML denoising (e.g., DeepCME, TransUNet on AIA) 의 동기를 제공했다.

**De-facto industry standard for astronomical image processing.** Toolboxes such as ISAP, MR/1 and Sparse2D rely on the IUWT formulation (Eqs. 11–12); coupled with NRGF and MGN, it underpins routine processing for SDO/AIA, STEREO/EUVI and Solar Orbiter EUI. The positive-synthesis + iterative-thresholding philosophy also inspired modern multi-scale neural architectures (LISTA, learned wavelets, U-Net denoisers used in astronomical pipelines).

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
