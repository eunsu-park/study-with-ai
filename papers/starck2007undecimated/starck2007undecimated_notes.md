---
title: "The Undecimated Wavelet Decomposition and its Reconstruction"
authors: Jean-Luc Starck, Jalal Fadili, Fionn Murtagh
year: 2007
journal: "IEEE Transactions on Image Processing, Vol. 16, No. 2, pp. 297–309"
doi: "10.1109/TIP.2006.887733"
topic: Low_SNR_Imaging
tags: [wavelet, undecimated, isotropic, à-trous, denoising, ringing-artifact, frame, POCS, Landweber, MCA, astronomy]
status: completed
date_started: 2026-05-06
date_completed: 2026-05-06
---

# 36. The Undecimated Wavelet Decomposition and its Reconstruction / 비분할 wavelet 분해와 재구성

---

## 1. Core Contribution / 핵심 기여

The paper consolidates two undecimated wavelet transforms — the **standard three-orientation undecimated wavelet transform (UWT)** and the **isotropic undecimated wavelet transform (IUWT)** — into a single design framework, makes their relation explicit (the sum of the three directional bands in the UWT equals the IUWT detail at each scale, Eq. 12), and uses the redundancy thus exposed to design **non-orthogonal filter banks whose synthesis filter $\tilde g$ is positive**. This single design choice — keeping the analysis pair $(h, g=\delta-h)$ but using $\tilde g = \delta + h$ on synthesis — kills the ringing oscillations that contaminate wavelet-thresholding restorations near edges and singularities. Because the resulting frame is **non-tight**, exact reconstruction from a thresholded subset of coefficients is not direct, so the paper unifies a Landweber/POCS iterative scheme (Eqs. 24–28) that consistently inverts the analysis on the multiresolution support. Experiments on Lena, a truncated Gaussian, a noisy square, and an MCA toy (bumps + sine) show the new bank, with iterative reconstruction and a positivity constraint, yields the lowest MSE and the cleanest edges of all variants tested.

이 논문은 비분할 wavelet 변환의 두 표준 형태 — 세 방향(가로/세로/대각) UWT 와 등방형 IUWT — 를 하나의 설계 틀로 묶고, 이 둘이 사실상 같은 분석 필터 뱅크를 다른 방식으로 그룹화한 것임을 식 (12) 로 명시한 뒤, 거기서 드러나는 redundancy 를 이용해 **양(positive)의 합성 필터** 를 갖는 비직교 필터 뱅크를 새로 설계한다. 분석쌍 $(h,\ g=\delta-h)$ 은 그대로 두고 합성쌍을 $\tilde h = h,\ \tilde g = \delta + h$ 로 잡는 단순한 선택만으로 wavelet 임계처리 후 흔히 나타나는 edge 부근의 ringing 진동이 거의 사라진다. 단 이 frame 은 일반적으로 tight 하지 않아 임계처리된 subset 에서 직접적인 완전재구성은 불가능하므로, 본 논문은 multiresolution support 위에서의 Landweber/POCS iterative scheme (식 24–28) 으로 통일적인 역변환을 제공한다. Lena, truncated Gaussian, noisy square, MCA bumps+sine 네 실험에서 (반복 + 양 제약) 결합된 새 필터 뱅크가 가장 낮은 MSE 와 가장 깨끗한 edge 를 보여준다.

---

## 2. Reading Notes / 읽기 노트

### Part I: Notation and the à-trous Skeleton / 표기와 à-trous 골격 (pp. 297–298)

The decimated DWT used in JPEG2000 sacrifices translation-invariance for orthogonality. Astronomers, biologists and signal-processing practitioners had instead adopted the **continuous WT** (full redundancy, no reconstruction) or the **à-trous (with-holes) algorithm** (Holschneider 1989; Shensa 1992). The à-trous recursion underlies both UWT and IUWT — at scale $j$ the analysis filter is $h^{(j)}$, defined by inserting $2^j-1$ zeros between successive taps of $h$:

$$
c_{j+1}[l] = (\bar h^{(j)} * c_j)[l] = \sum_k h[k]\, c_j[l + 2^j k], \qquad w_{j+1}[l] = (\bar g^{(j)} * c_j)[l]. \tag{1}
$$

Here $c_j$ is the **smooth (low-pass) array at scale $j$** and $w_j$ is the corresponding **detail (wavelet) array**. The detail at the next-coarser scale is just the difference $w_{j+1}[l] = c_j[l] - c_{j+1}[l]$ (Eq. 9), provided $g = \delta - h$. Reconstruction reads

$$
c_j[l] = \tfrac12\big[(\check h^{(j)} * c_{j+1})[l] + (\tilde g^{(j)} * w_{j+1})[l]\big]. \tag{2}
$$

The exact reconstruction condition,

$$
H(z^{-1})\tilde H(z) + G(z^{-1})\tilde G(z) = 1, \tag{3}
$$

now admits **infinitely many synthesis pairs $(\tilde h, \tilde g)$** because there is no decimation — the redundancy is what gives design freedom.

표기를 정리하자. $h[n]$ 은 1-D 분석 필터, $\bar h[n]=h[-n]$ 은 시간역전, $h^{(j)}$ 는 탭 사이에 $2^j-1$ 개의 0 을 끼운 변형이다. 식 (1) 의 $c_j$ 는 스케일 $j$ 의 평활 배열, $w_j$ 는 디테일 배열이며, $g=\delta-h$ 를 택하면 식 (9) 에 의해 $w_{j+1}=c_j-c_{j+1}$ 로 단순화된다. 핵심은 식 (3) 의 완전재구성 조건이 비분할 변환에서 무한히 많은 합성쌍을 허용한다는 점 — 이 자유도 위에서 본 논문의 설계가 이루어진다.

### Part II: 2-D UWT, IUWT and their Equivalence / 2-D UWT 와 IUWT 의 동등성 (pp. 298–299)

The 2-D **standard UWT** uses the same separable design as Mallat's DWT but with the à-trous (no-decimation) recursion:

$$
\begin{aligned}
c_{j+1}[k,l] &= (\bar h^{(j)} \bar h^{(j)} * c_j)[k,l],\\
w^{1}_{j+1}[k,l] &= (\bar g^{(j)} \bar h^{(j)} * c_j)[k,l]\quad \text{(vertical)}, \\
w^{2}_{j+1}[k,l] &= (\bar h^{(j)} \bar g^{(j)} * c_j)[k,l]\quad \text{(horizontal)}, \\
w^{3}_{j+1}[k,l] &= (\bar g^{(j)} \bar g^{(j)} * c_j)[k,l]\quad \text{(diagonal)}.
\end{aligned} \tag{4}
$$

Three detail bands per scale, redundancy $3(J-1)+1$.

The **IUWT** (Section II-B) keeps a single, isotropic detail per scale by choosing a **2-D B-spline scaling** function $\phi(x,y)=\phi_1(x)\phi_1(y)$ with the cubic B-spline $\phi_1$, and defining the wavelet as the difference between two resolutions:

$$
\tfrac{1}{4}\psi(x/2, y/2) = \phi(x,y) - \tfrac{1}{4}\phi(x/2, y/2). \tag{5}
$$

The associated 1-D filter is the cubic-spline kernel $h^{1D} = [1,4,6,4,1]/16$, which is separable in 2-D ($h[k,l] = h^{1D}[k]\,h^{1D}[l]$), and $g = \delta - h*h$. There is **only one detail band per scale** and reconstruction is the trivial sum:

$$
c_0[k,l] = c_J[k,l] + \sum_{j=1}^J w_j[k,l]. \tag{10}
$$

The two transforms are not different filter banks — they are **two ways to group** the bands of the same à-trous filter bank. Indeed (Eq. 12),

$$
w^{1}_j + w^{2}_j + w^{3}_j = c_{j-1} - c_j. \tag{12}
$$

i.e., **summing the three directional UWT bands at scale $j$ recovers the IUWT detail band at scale $j$**, so summing all bands recovers the original image exactly.

2-D UWT 는 Mallat 의 직교 DWT 와 같은 분리형 구조를 쓰지만 à-trous 점화식으로 다운샘플 없이 분해해 스케일당 3 개 detail (수직/수평/대각) 을 만든다. IUWT 는 등방 cubic B-spline 스케일링 ($\phi_1(x)$, 식 5) 을 쓰고 wavelet 을 두 해상도의 차이로 정의해 스케일당 1 개의 등방 detail band 만 남긴다. 사용하는 1-D 필터는 $h^{1D}=[1,4,6,4,1]/16$ 로 같다. 식 (12) 가 보여주듯 UWT 의 세 방향 detail 의 합이 IUWT 의 그 스케일 detail 과 일치하므로, 두 변환은 본질적으로 **같은 필터 뱅크의 다른 그룹화** 다.

### Part III: Designing the New Filter Banks / 새 필터 뱅크 설계 (pp. 299–301)

This is the contribution. With analysis $(h,\ g=\delta-h)$ fixed (the cubic-spline Astro bank), there is freedom in $(\tilde h, \tilde g)$. A particularly useful choice (Eq. 16):

$$
H(z) = \tfrac{1+z^{-1}}{2},\quad G(z) = \tfrac{z^{-1}-1}{2},\quad \check H(z) = \tfrac{z^{2}+4z+6+4z^{-1}+z^{-2}}{16},
$$

with $\tilde h = h$. Then exact reconstruction (3) forces

$$
\tilde G(z) = \frac{1 - \tilde H(z)\,H(z^{-1})}{G(z^{-1})},
$$

which simplifies, for the spline bank, to (Eq. 18) $\tilde g = [1,6,16,-6,-1]/16$ — still oscillatory. **Surprising twist** (Section III-A): if instead one keeps $\tilde h = h$ and **chooses $\tilde \phi = \phi$ on synthesis**, the reconstruction filter $\tilde g$ becomes

$$
\tilde g = \delta + h.
$$

For $h = [1,4,6,4,1]/16$ this gives $\tilde g = [1,4,22,4,1]/16$ — **all positive**! Property 1 says (i) the analysis bank $(h,\delta-h)$ implements a (non-tight) frame, and (ii) FIR perfect reconstruction is possible with this positive synthesis bank. Fig. 3 visualises the back-projections and confirms positivity for vertical/horizontal/diagonal coefficients — there is no negative lobe near a coefficient, which is the geometric reason ringing disappears.

분석쌍 $(h, g=\delta-h)$ 은 고정한 채 합성쌍에 자유가 있다. 식 (16)–(18) 의 직접적인 해는 여전히 진동하는 $\tilde g$ 를 주지만, $\tilde\phi = \phi$ 로 두면 자연스럽게 $\tilde g = \delta + h$ 가 나오고 — 예컨대 $h=[1,4,6,4,1]/16$ 이면 $\tilde g=[1,4,22,4,1]/16$ 으로 — 모든 탭이 양수다 (Property 1). Fig. 3 의 백프로젝션 시각화가 이 사실을 한눈에 보여준다 — 한 wavelet 계수의 inverse image 에 음의 lobe 가 없으므로 임계처리 후 ringing 도 발생하지 않는다.

The Haar special case (Section III-B, Eq. 18) shows the same construction yields $\tilde g = z^3[1,6,16,-6,-1]/16$ for direct Haar; non-orthogonal synthesis filters with positivity remain available. Section III-C derives yet another branch ($\tilde \phi = \phi$ with $\hat\psi(2\nu) = (\hat\phi^2(\nu)-\hat\phi^2(2\nu))/\hat\phi(\nu)$, $\tilde g = \delta$) where reconstruction uses **only the smooth filter $\tilde h$** — sometimes useful for edge-detection workflows.

Haar 의 경우 (식 18) 도 동일 설계 흐름으로 양의 비정형 합성 필터를 유도할 수 있고, Section III-C 는 합성 단계에서 디테일 필터 $\tilde g=\delta$ 인 분기 (식 22) 도 제공해 edge detection 응용에 유용하다.

### Part IV: Iterative Reconstruction (POCS / Landweber) / 반복 재구성 (pp. 301–303)

A key practical issue: on a **redundant** transform, applying $\mathcal R \mathcal W$ to a thresholded coefficient set $\alpha_T$ does **not** generally satisfy $\alpha = \mathcal W \mathcal R \alpha$ — i.e., $\alpha_T \notin \text{range}(\mathcal W)$. The cure is alternating projection (POCS) onto three convex sets:

- $W = \{\alpha : \alpha = \mathcal W S,\; S \in \ell^2(\mathbb Z^2)\}$ — range of analysis;
- $\mathcal M = \{\alpha : M_{j,k}\alpha_{j,k} = M_{j,k}\alpha_{T,j,k}\}$ — multiresolution support match;
- $\mathcal C = \{\alpha : \mathcal R\alpha \ge 0\}$ — positivity of the reconstructed image.

The Landweber-style iteration with positivity is

$$
\tilde S^{n+1} = P_{+}\!\left(\tilde S^n + \mathcal R\!\left[\alpha_T - \mathcal W \tilde S^n\right]\right). \tag{25}
$$

Equivalence to POCS is shown by composing the projectors $\mathcal P_W = \mathcal W \mathcal R$, $\mathcal P_{\mathcal M}\tilde\alpha_{j,k} = \alpha_{T,j,k}$ on support and $\tilde\alpha_{j,k}$ off support, $\mathcal P_{\mathcal C}\tilde\alpha = \mathcal W P_{+}\mathcal R\tilde\alpha$, then verifying that one round of alternating projection is exactly Eq. (24)/(25). Convergence depends on the chosen analysis/reconstruction frame — for the Moore–Penrose pseudo-inverse it is guaranteed; the authors note empirical convergence for their banks even when $\mathcal R$ is not the pseudo-inverse.

비분할 변환은 redundant 이라 임계처리된 계수 $\alpha_T$ 가 $\mathcal W$ 의 range 안에 있지 않을 수 있다. 따라서 Eq. (25) 의 Landweber 반복으로 (i) range $W$, (ii) multiresolution support $\mathcal M$, (iii) 양 cone $\mathcal C$ 세 볼록집합으로 교대 사영해야 일관된 재구성이 얻어진다. 이는 POCS 와 정확히 동치이며, Moore–Penrose pseudo-inverse 의 경우 수렴이 보장된다.

### Part V: Experiments / 실험 (pp. 303–306)

**(A) Lena nonlinear approximation, threshold 0–30, MSE.** Five variants compared (Fig. 6): bi-orthogonal DWT (worst, MSE up to 0.65), UWT 7/9 with direct synthesis, UWT 7/9 with iterative synthesis, **non-orthogonal positive bank with direct synthesis** (worse than DWT! — confirms direct synthesis is wrong for non-tight frames), **non-orthogonal positive bank with iterative reconstruction (best, MSE ≈ 0.45)**. The take-away: iterating *is essential* whenever the frame is non-tight.

**Lena 비선형 근사 실험.** 임계 0–30 구간 MSE 곡선 (Fig. 6) 에서 (i) bi-orthogonal DWT 가 가장 나쁘고 (MSE ≈ 0.65), (ii) 양의 비정형 필터 뱅크에 직접 합성을 쓰면 그보다도 더 나쁘며, (iii) 양의 비정형 + 반복 재구성이 가장 좋다 (MSE ≈ 0.45). 비정형 frame 에서는 반복이 필수임을 정량적으로 보여준다.

**(B) Truncated Gaussian (Fig. 7–8).** The image is a piecewise-smooth Gaussian (σ=25, max=1) with a sharp transition. Now the **best result is from the non-orthogonal positive bank (with iteration)** — better than the 7/9 UWT — because the spline scaling function is intrinsically Gaussian-like and matches the test image. MSE ≈ 0.18 vs. 0.37 for bi-orthogonal DWT.

**Truncated Gaussian.** 이 데이터는 sharp transition 을 가진 Gaussian. 여기서 양의 비정형 + 반복 결과가 7/9 UWT 보다도 우수 (MSE 0.18 vs. 0.37) — 스케일링 함수의 형태가 데이터와 잘 맞기 때문.

**(C) Ringing artifact (Fig. 9).** A row of the reconstructed truncated-Gaussian (threshold = 2.5) is plotted for six variants. The plain 7/9 UWT shows pronounced oscillations near the edge; iterative reconstruction reduces but does not eliminate them; the **non-orthogonal positive bank** produces a row that is almost monotonic with no ringing. Adding positivity damps remaining oscillations elsewhere but can amplify oscillations close to the edge — a subtle reminder.

**Ringing 시각화 (Fig. 9).** threshold=2.5 로 임계처리 후 한 행을 보면, 7/9 UWT 직접 합성은 edge 부근에서 강하게 진동하고, 반복 재구성은 줄여주나 잔존, 양의 비정형 필터 뱅크는 거의 단조롭게 매끈한 행을 만든다. 양 제약은 edge 멀리에선 진동을 죽이지만 edge 가까이에선 살짝 증폭할 수 있다.

**(D) Edge detection (Fig. 10).** A noisy square (Gaussian noise σ=3) is denoised and a Canny edge detector is applied. The **non-orthogonal positive bank gives noticeably fewer spurious edges** than the 7/9 UWT — direct evidence that ringing was producing false edges in the latter.

**엣지 검출.** σ=3 의 Gaussian noise 가 더해진 square 이미지에서 양의 비정형 필터 뱅크로 denoising 한 결과에 Canny 를 적용하면, 7/9 UWT 보다 가짜 엣지가 훨씬 적게 검출된다.

**(E) MCA decomposition (Eqs. 29–30, Fig. 11).** A 1-D signal containing three bumps + a sine + Gaussian noise is decomposed by **MCA**: bumps go into the wavelet dictionary, the sine into the DCT. The new positive UWT with positivity gives the cleanest bump recovery while introducing no oscillations into the DCT residual. This experiment seeds the entire **morphological component analysis** literature that follows (Starck-Elad-Donoho 2005).

**MCA 분해.** bumps + sine + noise 신호를 wavelet (bumps) 과 DCT (sine) 두 사전으로 분해하는 식 (29) 의 MCA. 양의 비정형 + positivity 가 가장 깨끗한 bump 복원과 잔여 DCT 분리를 보인다 — 이후 MCA 문헌의 출발점.

### Part VI: Conclusion / 결론 (pp. 306–308)

The paper advances three theses: (1) the redundancy of the UWT can be used to design new filter banks (positive, non-orthogonal) whose synthesis function is physically meaningful — Gaussian-like — and whose back-projection is positive, so wavelet thresholding produces no ringing; (2) the Haar undecimated bank, normally dismissed for its irregularity, can be regularised similarly and is uniquely useful for Poisson-noise denoising; (3) the **iterative reconstruction** scheme (Eqs. 24–28) is required for non-tight frames and is naturally the right vehicle to add side constraints (TV, $\ell^1$, positivity). These three pieces became the operational standard of the astronomical image-processing community.

논문은 세 가지 결론을 제시한다. (1) UWT 의 redundancy 로 양의 비직교 합성 필터를 설계할 수 있고, 이는 wavelet thresholding 의 ringing 을 자연스럽게 없앤다. (2) Haar 비분할 필터 뱅크도 같은 설계 흐름으로 regularised 될 수 있어 Poisson 잡음 처리에 유용하다. (3) Non-tight frame 의 일관된 역변환에는 iterative reconstruction 이 필수이며, TV·$\ell^1$·positivity 같은 보조 제약을 자연스럽게 결합할 수 있다. 이 세 결론은 천문 영상 커뮤니티의 표준 도구로 정착했다.

---

## 3. Key Takeaways / 핵심 시사점

1. **UWT and IUWT are the same filter bank, regrouped.** The only difference is whether you keep three directional detail bands per scale or sum them into one isotropic band; Eq. (12) makes the equivalence explicit. / **UWT 와 IUWT 는 같은 필터 뱅크의 다른 그룹화** 일 뿐이다. 식 (12) 가 등동성을 명시한다.

2. **Redundancy = design freedom.** Without decimation the perfect-reconstruction condition (Eq. 3) is satisfied by infinitely many synthesis pairs, opening room for designs that orthogonal/biorthogonal wavelets cannot reach. / **Redundancy 는 설계 자유도** 다. 비분할이라 식 (3) 의 완전재구성 조건이 무한히 많은 합성쌍을 허용한다.

3. **Positive synthesis filters kill ringing.** Choosing $\tilde h = h$ and $\tilde \phi = \phi$ forces $\tilde g = \delta + h$, which is positive for the cubic-spline analysis filter. Back-projection of any single coefficient is then positive, so thresholding cannot create negative side-lobes — i.e., no ringing. / **양의 합성 필터가 ringing 을 없앤다.** $\tilde h = h$, $\tilde\phi=\phi$ 의 선택이 자연스럽게 $\tilde g = \delta + h$ 를 주고, cubic-spline $h$ 에 대해 모든 탭이 양수이므로 어떤 단일 계수의 백프로젝션도 양수다 → ringing 없음.

4. **Direct synthesis is wrong for non-tight frames; iteration is mandatory.** Lena experiment (Fig. 6) shows the new positive bank with *direct* synthesis is worse than the bi-orthogonal DWT; with *iterative* reconstruction it becomes the best. / **Non-tight frame 에서 직접 합성은 잘못된 답** 이다. Fig. 6 에서 직접 합성은 bi-orthogonal DWT 보다도 나쁘지만, 반복 재구성을 쓰면 가장 좋아진다.

5. **The Landweber iteration = POCS on three convex sets.** Range of $\mathcal W$, multiresolution support, and image positivity. The iteration $\tilde S^{n+1} = P_+(\tilde S^n + \mathcal R[\alpha_T - \mathcal W \tilde S^n])$ is precisely the alternating projection composition. / **Landweber 반복 = POCS** ($W,\mathcal M,\mathcal C$ 세 볼록집합 위 교대 사영). 식 (25) 가 그 정확한 합성이다.

6. **The IUWT structure connects to all subsequent solar-coronal enhancement.** The cubic-spline $h^{1D}=[1,4,6,4,1]/16$ + difference-of-resolutions wavelet is the same transform under MR/1, ISAP, and the toolboxes used by NRGF (#35) and MGN (#38). / **IUWT 구조는 이후 태양 코로나 enhancement 의 공통 기반.** $[1,4,6,4,1]/16$ 큐빅 스플라인 + 해상도 차분 wavelet 이 NRGF, MGN, MR/1, ISAP 까지 동일하다.

7. **MCA seeds itself here.** Eqs. (29)–(30) define the alternate projection / thresholding loop that becomes Starck-Elad-Donoho 2005 (MCA) and underpins all later sparsity-promoting source separation. / **MCA 는 여기서 시작.** 식 (29)–(30) 의 교대 사영-임계처리 루프가 곧 MCA 의 표준 알고리즘이 된다.

8. **Ringing → false edges; positivity → cleaner detection.** The Canny experiment (Fig. 10) is a small but vital demonstration that artifact suppression in restoration directly improves downstream tasks (edge/feature detection). / **Ringing 은 가짜 엣지를 만든다** — Fig. 10 의 Canny 실험은 복원 단계의 아티팩트 억제가 곧바로 하위 작업의 정확도를 올림을 보여준다.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 1-D à-trous decomposition / 1-D à-trous 분해

$$
\boxed{\;c_{j+1}[l] = \sum_k h[k]\, c_j[l + 2^j k], \qquad w_{j+1}[l] = \sum_k g[k]\, c_j[l + 2^j k]\;}
$$

- $c_j$: smooth at scale $j$ (low-pass cascade) / 스케일 $j$ 의 평활.
- $w_j$: detail at scale $j$ (band-pass output) / 스케일 $j$ 의 디테일.
- $h$: 1-D analysis low-pass filter (typically cubic-spline $[1,4,6,4,1]/16$).
- $g = \delta - h$: high-pass complement, giving the simple identity $w_{j+1} = c_j - c_{j+1}$.
- Inserted-zero pattern: $h^{(j)}$ has $2^j-1$ zeros between successive non-zero taps.

### 4.2 2-D UWT and IUWT relation / 2-D UWT 와 IUWT 의 관계

UWT: three details per scale,

$$
w_{j+1}^{1} = \bar g\bar h * c_j,\quad w_{j+1}^{2} = \bar h\bar g * c_j,\quad w_{j+1}^{3} = \bar g\bar g * c_j.
$$

IUWT: one detail per scale,

$$
w_{j+1} = c_j - c_{j+1}.
$$

Bridge identity:

$$
\boxed{\;w_j^{1} + w_j^{2} + w_j^{3} = c_{j-1} - c_j = w_j^{\text{IUWT}}\;}
$$

so the original image is recovered by either grouping:

$$
c_0[k,l] = c_J[k,l] + \sum_{j=1}^{J}\sum_{d=1}^{3} w_j^{d}[k,l] \qquad\text{(UWT)} \tag{11}
$$

$$
c_0[k,l] = c_J[k,l] + \sum_{j=1}^{J} w_j[k,l] \qquad\text{(IUWT, Eq. 10)}.
$$

### 4.3 Perfect-reconstruction & frame design / 완전재구성과 프레임 설계

$$
\boxed{\;H(z^{-1})\tilde H(z) + G(z^{-1})\tilde G(z) = 1\;} \tag{3}
$$

For analysis $g = \delta - h$, frame tightness requires $|\hat h(\nu)|^2 + |\hat g(\nu)|^2 = c$. With $\hat g = 1 - \hat h$ this forces $\hat h \equiv 1$ (Eq. 7–8), so the bank is **inevitably non-tight** — yet still a frame.

Two synthesis branches make the bank usable:

- **Direct positive bank** (Section III-A): $\tilde h = h$, $\tilde g = \delta + h$. For $h=[1,4,6,4,1]/16$, $\tilde g = [1,4,22,4,1]/16$ — every tap positive.
- **Smooth-only synthesis** (Section III-C): $\tilde g = \delta$, reconstruction uses only $\tilde h$.

### 4.4 Iterative reconstruction (Eqs. 24–28) / 반복 재구성

$$
\tilde S^{n+1} = \tilde S^{n} + \mathcal R\big[\alpha_T - \mathcal W\tilde S^{n}\big] \qquad\text{(Landweber)} \tag{24}
$$

$$
\boxed{\;\tilde S^{n+1} = P_{+}\!\left(\tilde S^{n} + \mathcal R\big[\alpha_T - \mathcal W\tilde S^{n}\big]\right)\;} \qquad\text{(with positivity, Eq. 25)}
$$

Equivalent POCS form (Eq. 28):

$$
\tilde\alpha^{n+1} = \mathcal P_{\mathcal C}\circ\mathcal P_{\mathcal M}\circ\mathcal P_W\,\tilde\alpha^{n} = \mathcal W P_+\mathcal R\big[M\alpha_T + (I-M)\mathcal W\mathcal R\tilde\alpha^{n}\big].
$$

- $\mathcal W$: analysis (UWT/IUWT) / 분석 연산자.
- $\mathcal R$: synthesis (direct addition for IUWT) / 합성 연산자.
- $M$: multiresolution support mask (1 if coefficient was retained, 0 otherwise).
- $P_+$: pixel-wise positivity projection.

### 4.5 Soft thresholding / 소프트 임계처리

$$
\Delta_T(\alpha) = \mathrm{sgn}(\alpha)\,\max(|\alpha| - T,\, 0).
$$

For Gaussian noise with std $\sigma$, the universal threshold is $T = K\sigma_j$ with $K\in\{3,4,5\}$ depending on aggressiveness, where $\sigma_j$ is the noise standard deviation at scale $j$ (different at each scale for the IUWT because the cubic-spline filter changes its propagation).

### 4.6 MCA loop / MCA 루프 (Eqs. 29–30)

$$
\min_{X_b, X_s}\;\|\mathbf{W}X_b\|_1 + \|\mathbf{C}X_s\|_1 \quad\text{s.t.}\quad \|Y - X_b - X_s\|_2 < \sigma.
$$

Block-coordinate / threshold-decreasing iteration:

$$
\tilde X_b^{(k+1)} = \mathbf F_{\mathbf W,\lambda_k}(Y - \tilde X_s^{(k)}),\qquad \tilde X_s^{(k+1)} = \mathbf F_{\mathbf C,\lambda_k}(Y - \tilde X_b^{(k+1)}). \tag{30}
$$

$\mathbf F_{\mathbf T,\lambda}$ = decompose with $\mathbf T$, hard-threshold, reconstruct.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
 1989  Mallat — orthogonal multiresolution analysis (MRA) of L²(ℝ)
   │
 1989  Holschneider, Kronland-Martinet, Morlet, Tchamitchian — à-trous algorithm
   │
 1992  Daubechies — "Ten Lectures on Wavelets" (orthogonal compactly-supported)
   │
 1992  Shensa — relation between à-trous and Mallat algorithms
   │
 1995  Burt & Adelson revisited — Laplacian pyramid as redundant wavelet
   │
 1998  Mallat — "A Wavelet Tour of Signal Processing" (textbook canonisation)
   │
 2002  Starck & Murtagh — Astronomical Image and Data Analysis (IUWT in astronomy)
   │
 2002  Durand & Froment — TV regularisation of wavelet coefficients
   │
 2005  Tropp, Dhillon, Heath, Strohmer — alternating projection for tight-frame design
   │
 2006  Steidl, Weickert et al. — equivalence of soft thresholding and TV regularisation
   │
 2007 ★ Starck, Fadili, Murtagh — UWT/IUWT unification + positive synthesis
        + iterative reconstruction (THIS PAPER)
   │
 2009  Starck, Murtagh & Fadili — "Sparse Image and Signal Processing" (book)
   │
 2010s Inpainting / curvelets / shearlets / learned wavelets (LISTA, U-Net)
        all build on the iterative-reconstruction skeleton established here.
   │
 2014  Morgan & Druckmüller (#38) MGN — multi-scale Gaussian normalization
        re-uses the same cubic-spline scale-space philosophy.
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Mallat (1989) — MRA | Orthogonal multiresolution analysis is the foundation; UWT *removes* the decimation step that DWT inherits. / 직교 MRA 가 모태이며, UWT 는 DWT 의 decimation 만 제거한 것. | High: precondition for redundancy. / 매우 높음. |
| Daubechies (1992) — Ten Lectures | Orthogonal/biorthogonal compactly-supported wavelets; gives the perfect-reconstruction condition. / 직교/이중직교 wavelet 의 완전재구성 조건. | High. |
| Holschneider et al. (1989) — à-trous | Provides the exact insertion-zero recursion that Section II builds on. / Section II 의 점화식이 그대로 사용. | Direct algorithmic ancestor. |
| Shensa (1992) — DWT/à-trous link | Bridges Mallat's pyramid and à-trous; Starck et al. note a comparable bridge UWT↔IUWT here. / 비슷한 동등성 결과. | Methodological. |
| Starck et al. (2002) — IUWT for astronomy | The IUWT defined here was already in routine astronomical use; this paper formalises the bank. / 본 논문이 IUWT 를 정형화. | Direct continuation. |
| Donoho-Johnstone (1994) — wavelet shrinkage | Soft-thresholding rule used in the experiments; ringing is a known weakness this paper attacks. / 임계처리의 ringing 문제를 직접 공격. | High. |
| Starck-Elad-Donoho (2005) — MCA | Eqs. (29)–(30) of this paper *are* the MCA algorithm; the 2005 paper expands it. / 본 논문의 식 29-30 이 MCA 의 시드. | Direct seed. |
| Morgan & Druckmüller (2014, #38) — MGN | Same cubic-spline scale-space philosophy, but with arctan + γ recombination instead of UWT. / 같은 큐빅스플라인 스케일 스페이스. | Topical. |
| Morgan, Habbal & Woo (2006) — NRGF (#35) | Coronal-image enhancement built on radial Gaussian normalisation; complements UWT for white-light corona. / 비교 enhancement 기법. | Topical. |
| Burt & Adelson (1983) — Laplacian pyramid | Conceptual ancestor; Eq. (10) is essentially the Laplacian-pyramid identity for IUWT. / IUWT 의 합 성질이 라플라시안 피라미드와 동일. | Conceptual. |

---

## 7. References / 참고문헌

- Starck, J.-L., Fadili, J., & Murtagh, F. (2007). *The Undecimated Wavelet Decomposition and its Reconstruction.* IEEE Trans. Image Process., **16**(2), 297–309. DOI: 10.1109/TIP.2006.887733
- Mallat, S. G. (1989). *A theory for multiresolution signal decomposition: the wavelet representation.* IEEE Trans. PAMI, **11**(7), 674–693.
- Daubechies, I. (1992). *Ten Lectures on Wavelets.* SIAM.
- Holschneider, M., Kronland-Martinet, R., Morlet, J., & Tchamitchian, P. (1989). *A real-time algorithm for signal analysis with the help of the wavelet transform.* In *Wavelets: Time-Frequency Methods and Phase-Space.* Springer, pp. 286–297.
- Shensa, M. J. (1992). *Discrete wavelet transforms: wedding the à trous and Mallat algorithms.* IEEE Trans. Signal Process., **40**(10), 2464–2482.
- Starck, J.-L. & Murtagh, F. (2002). *Astronomical Image and Data Analysis.* Springer.
- Donoho, D. L. & Johnstone, I. M. (1994). *Ideal spatial adaptation by wavelet shrinkage.* Biometrika, **81**(3), 425–455.
- Starck, J.-L., Elad, M., & Donoho, D. L. (2005). *Image decomposition via the combination of sparse representations and a variational approach.* IEEE Trans. Image Process., **14**(10), 1570–1582.
- Morgan, H. & Druckmüller, M. (2014). *Multi-Scale Gaussian Normalization for Solar Image Processing.* Solar Phys., **289**, 2945–2955. DOI: 10.1007/s11207-014-0523-9.
- Burt, P. J. & Adelson, E. H. (1983). *The Laplacian pyramid as a compact image code.* IEEE Trans. Commun., **31**, 532–540.
- Steidl, G., Weickert, J., Brox, T., Mrázek, P., & Welk, M. (2003). *On the equivalence of soft wavelet shrinkage, total variation diffusion, total variation regularization, and SIDEs.* Tech. Rep. 26, Univ. Bremen.
