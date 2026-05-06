---
title: "The Contourlet Transform: An Efficient Directional Multiresolution Image Representation"
authors: Minh N. Do, Martin Vetterli
year: 2005
journal: "IEEE Transactions on Image Processing 14(12), pp. 2091–2106"
doi: "10.1109/TIP.2005.859376"
topic: Low-SNR Imaging / Directional Multiscale Transforms
tags: [contourlet, directional-transform, laplacian-pyramid, directional-filter-bank, dfb, parabolic-scaling, geometric-image-representation, do-vetterli, multiresolution-multidirection]
status: completed
date_started: 2026-05-06
date_completed: 2026-05-06
---

# 5. The Contourlet Transform: An Efficient Directional Multiresolution Image Representation / 컨투어릿 변환: 효율적 방향성 다중해상 영상 표현

---

## 1. Core Contribution / 핵심 기여

### 한국어
이 논문은 영상의 *기하학적 구조*(매끄러운 윤곽선)를 효율적으로 포착하는 *이산 영역에서 직접 정의된* 새로운 변환 **Contourlet Transform**을 제시한다. 핵심 기여 4가지:

(A) **5가지 wish list**: 이상적인 영상 표현이 갖춰야 할 조건 — (1) multiresolution, (2) localisation, (3) critical sampling, (4) **directionality**, (5) **anisotropy** (긴 가로 vs 짧은 세로 base함수). 분리형 2-D 웨이블릿은 (1)–(3)만 제공; (4), (5) 누락. Curvelet (Candès-Donoho)은 모두 만족하나 *연속 영역*에서 정의돼 직사각 격자에 어색.

(B) **Double iterated filter bank** (Fig. 7): **Laplacian Pyramid (LP)** + **Directional Filter Bank (DFB)** 의 직렬 연결. LP는 점 불연속을 multi-scale로 분해 (Burt-Adelson 1983), DFB는 점들을 **선형 구조로 묶음** (Bamberger-Smith 1992). 결과: $2^{l_j}$ 방향, $j$-스케일에서 contourlet 계수 $c^{(l_j)}_{j,k}[\mathbf n]$, 0 ≤ k < $2^{l_j}$. **Theorem 1**: tight frame, redundancy < 4/3, $O(N)$ 알고리즘, 임의 directional decomposition tree.

(C) **Parabolic scaling**: contourlet 함수의 support가 $\text{width} \approx C 2^j$, $\text{length} \approx C 2^{j + l_j - 2}$이라 $\text{width} \propto \text{length}^2$ 만족 (Eq. 28). 이는 $C^2$ 곡선을 sparse하게 표현하기 위한 *결정적* 조건 (Candès-Donoho's parabolic scaling 원리).

(D) **Optimal NLA rate (Theorem 4)**: parabolic scaling + directional vanishing moments (DVM) 조건 하에서 contourlet은 piecewise $C^2$ smooth + $C^2$ contours에 대해 *최적* nonlinear approximation rate $\|f - \hat f_M\|^2_2 \lesssim (\log M)^3 M^{-2}$. 동일 클래스에서 분리형 웨이블릿은 $O(M^{-1})$, Fourier는 $O(M^{-1/2})$에 그침.

**실험 결과**: Lena $\sigma$ noise → wavelet PSNR 29.41 dB → contourlet 30.47 dB (**+1.06 dB**); Barbara NLA at $M = 4096$: wavelet 24.34 dB → contourlet 25.70 dB (+1.36 dB).

### English
The paper introduces the **Contourlet Transform** as a directional multiresolution image representation defined *directly in the discrete domain*. Four key contributions:

(A) Identifies a five-criterion **wish list**: multiresolution, localisation, critical sampling, **directionality**, **anisotropy** (long-narrow basis elements). Separable 2-D wavelets miss directionality and anisotropy; continuous-domain curvelets satisfy all but lack natural discrete-grid implementation.

(B) **Double iterated filter bank** (Fig. 7): cascade of Laplacian Pyramid (LP) for multiresolution + Directional Filter Bank (DFB, Bamberger-Smith) for directional decomposition. Theorem 1 establishes tight frame property, redundancy < 4/3, $O(N)$ complexity.

(C) **Parabolic scaling**: $\text{width} \propto \text{length}^2$ — essential for sparse representation of $C^2$ contours.

(D) **Optimal NLA rate** (Theorem 4): $(\log M)^3 M^{-2}$ for piecewise-$C^2$ images with $C^2$ contours, beating separable wavelets' $O(M^{-1})$.

Empirical: Lena denoising — wavelet PSNR 29.41 dB → contourlet 30.47 dB (+1.06 dB).

---

## 2. Reading Notes / 읽기 노트

### Part I: §I-II Wish list and related work / 위시 리스트와 선행 연구

#### 한국어
- Fig. 1: 매끈한 윤곽선을 표현할 때 wavelet은 *isotropic 정사각 dot*들로 점점이 추적해야 함. 새 표현은 contour를 따라 *기다란 sketch*로 효율적 추적.
- Wavelet vs new scheme: $M$개 계수로 NLA. Wavelets need $O(M)$ "dots" along contour; new scheme uses $O(\sqrt M)$ "sketches".
- 인간 시각계 (V1 simple cells, Hubel-Wiesel 1962, Olshausen-Field 1996)는 *국소·방향성·다중스케일* 구조 — 영상 표현이 따라야 할 자연 모델.
- **Curvelets** (Candès-Donoho 2004): 연속영역에서 정의, polar 좌표 → 이산영역에서 *블록 ridgelet* 등 복잡한 후처리 필요. 직사각 격자에 부자연.
- **Bandelets, wedgelets, brushlets, complex wavelets** 모두 부분적 만족. Contourlet은 *iterated filter bank* 구조라 $O(N)$ + 다중 트리 구조 자연스러움.

#### English
The wish list of five criteria isolates what natural-image-friendly representations need; only continuous-domain curvelets satisfy all five before this paper. Contourlet provides the discrete-domain equivalent.

---

### Part II: §III Discrete construction / 이산 구성

#### 한국어 — §III.A Concept

**Insight**: 자연 영상의 wavelet 계수는 *contour를 따라* 클러스터링됨. 이 클러스터를 *명시적으로* 묶어내면 더 sparse한 표현을 얻을 수 있음 → "wavelet-like for edge detection + local directional for contour gathering" → **double filter bank**.

#### 한국어 — §III.B Laplacian Pyramid (Burt-Adelson 1983)

LP의 한 레벨 (Fig. 2):
- Analysis: input $x \to a = \mathbf M \cdot Hx$ (downsampled lowpass) → bandpass $b = x - G(\uparrow_{\mathbf M} a)$
- 반복하여 multi-scale 분해 $\{a_J, b_J, b_{J-1}, \ldots, b_1\}$
- LP의 redundancy: $1 + \sum 1/4^j < 4/3$ — wavelets의 critical sampling보다 약간 추가, 그러나 *각 레벨 1개 bandpass*만 생성하므로 frequency scrambling 없음 (wavelet에서 highpass downsample이 low band로 fold-back 하는 문제 회피).
- 본 논문은 Do-Vetterli (2003)의 *frame-based* 재구성 사용 (Fig. 2(b)).

#### English — §III.B LP
LP gives clean bandpass at each level without frequency aliasing (unlike critically-sampled wavelet). Redundancy < 4/3.

#### 한국어 — §III.C Directional Filter Bank (DFB)

Bamberger-Smith의 $l$-level binary tree → $2^l$ wedge-shaped frequency partition (Fig. 3). 본 논문은 *quincunx fan filter banks* + *shearing* 으로 단순화 (Fig. 4-5).
- 0 ≤ $k < 2^{l-1}$: mostly horizontal, 2^{l-1} ≤ $k < 2^l$: mostly vertical
- 샘플링 행렬 $\mathbf S^{(l)}_k$는 diagonal: $\text{diag}(2^{l-1}, 2)$ 또는 $\text{diag}(2, 2^{l-1})$ (Eq. 3)
- DFB의 equivalent filters는 *Radon-like*: 가는 사선들 — Fig. 6 "Radonlets"
- DFB는 *low frequency를 잘 처리 못함* → LP가 먼저 lowpass 분리 후 bandpass에만 DFB 적용 (Fig. 7).

#### English — §III.C DFB
$2^l$ wedge-shaped frequency partition via iterated quincunx fan filter banks + shearing operators. Equivalent filters look like fine slanted lines ("Radonlets"). DFB poorly handles low frequencies, hence the LP→DFB cascade.

#### 한국어 — §III.D Discrete contourlet transform

전체 알고리즘 (Fig. 7): $a_0 = $ input → LP → $\{b_j, a_J\}_j$ → 각 $b_j$에 $l_j$-level DFB → contourlet 계수 $\{c^{(l_j)}_{j, k}\}$.

**Theorem 1 (key properties)**:
1. PR LP + PR DFB → contourlet은 frame
2. Orthogonal LP + orthogonal DFB → tight frame, frame bounds = 1
3. Redundancy < 4/3
4. Basis function support: width ≈ $C 2^j$, length ≈ $C 2^{j + l_j - 2}$
5. $O(N)$ complexity

#### English — §III.D
The composite LP→DFB transform is the discrete contourlet transform. Theorem 1: tight frame, redundancy < 4/3, O(N) complexity, anisotropic basis support.

---

### Part III: §IV-V DMRA and approximation rate / DMRA와 근사률

#### 한국어 — §IV Directional MRA framework

연속영역 *contourlet 함수* $\lambda^{(l)}_{j,k}(\mathbf t) = \sum d^{(l)}_k[\mathbf m] \mu_{j,m}(\mathbf t)$ (Eq. 18)이 $L_2(\mathbb R^2)$ detail subspace $W^{(l)}_{j,k}$의 tight frame 형성 (Proposition 3). Theorem 3: 이산 contourlet 계수는 정확히 inner product $\langle f, \lambda^{(L+l_j)}_{j,k,n}\rangle$ (Eq. 26).

#### 한국어 — §V Parabolic scaling and DVM

**§V.A**: $C^2$ 곡선 $u(v) \approx (\kappa/2)v^2$을 polynomial로 *국소 근사*하면 $\text{width} \sim (\kappa/8) \text{length}^2$ → parabolic scaling (Eq. 28). Contourlet support $\text{width} = C 2^j$, $\text{length} = C 2^{j + l_j - 2}$이 $\text{length}^2 \propto \text{width}$ 되려면:
$$
l_j = l_{j_0} + \lfloor (j_0 - j)/2 \rfloor \quad (29)
$$
즉 *2 스케일마다 directional 분해 1단계 추가* (Fig. 8). 이는 $O(\log M)$ 수의 directions across scales를 의미.

**§V.B**: **Directional Vanishing Moments (DVM)**: 모든 1-D slice가 $p$-차 vanishing moments. Lemma 1: $p$차 DVM이면 $C^2$ 함수와 contourlet의 inner product $\le \|\partial^p_y \lambda\|_\infty \cdot d^3_{j,k}$. Theorem 4 (NLA): parabolic scaling + DVM $p \ge 2$이면
$$
\|f - \hat f_M\|^2_2 \lesssim (\log M)^3 M^{-2} \quad (\text{piecewise } C^2 \text{ + } C^2 \text{ contours})
$$
#### English — §V Approximation rate
Parabolic scaling $(\text{width} \propto \text{length}^2)$ and directional vanishing moments together let contourlets attain the optimal $M^{-2}$ NLA rate (up to log factors) for piecewise $C^2$ images — beating separable wavelets' $M^{-1}$.

---

### Part IV: §VI Numerical Experiments / 수치 실험

#### 한국어
- 6 LP levels with "9-7" biorthogonal filters; DFB with "23-45" biorthogonal quincunx filters (Phoong+).
- Number of DFB levels doubles every other scale (parabolic scaling), max 5 at finest.

**Fig. 13 (Peppers, Barbara)**: LP에서 2 pyramidal levels × {4, 8} directions 분해. Contour를 따라 정렬된 spike pattern.

**Fig. 14 (basis comparison)**: Wavelet 5개 vs contourlet 4개 — contourlet은 *기다란 oriented* 모양, wavelet은 *짧은 isotropic*.

**Fig. 15 (NLA progression)**: Peppers를 $M = 4, 16, 64, 256$ 계수로 재구성:
- Wavelet: 점들이 contour 따라 흩뿌려짐
- Contourlet: contour를 따라 *sketch lines* — Fig. 1의 새 scheme 그대로 시각 확인

**Fig. 16 (Barbara, M = 4096)**: Wavelet PSNR 24.34 dB → contourlet 25.70 dB. (직물 texture 특히 잘 표현)

**Fig. 17 (Lena denoising, hard threshold)**: Original 24.42 dB noisy → Wavelet 29.41 dB → Contourlet **30.47 dB** (+1.06 dB).

#### English — §VI
Reproduces a +1.06 dB PSNR improvement on Lena denoising over wavelet baseline using simple hard thresholding on contourlet coefficients. Visual: contourlet sketches contours far more efficiently than wavelet's isotropic dots.

---

## 3. Key Takeaways / 핵심 시사점

1. **분리형 wavelet의 *방향성* 한계는 본질적 / Separable wavelets are fundamentally non-directional** — 분리형 2-D wavelet은 $\psi(x)\phi(y), \phi(x)\psi(y), \psi(x)\psi(y)$의 *3개* 방향만 — horizontal, vertical, diagonal. 자연 영상은 *연속적 방향*이 필요. 결과: contour 표현이 $O(\sqrt M)$ "dots"으로 비효율적.
   Separable wavelets offer only 3 directions; natural images need a continuum of directions, leading to inefficient $\sqrt{M}$-many "dots" along contours.

2. **Curvelet의 polar geometry는 *이산 격자*에 부자연 / Curvelets' polar geometry is unnatural on rectangular grids** — Curvelet은 연속영역에서 정의, 이산화 시 회전·블록 ridgelet 등 복잡한 후처리. Contourlet은 처음부터 *직사각 격자에서 직접* 정의 → wavelet-like 알고리즘 + iterated filter bank 자연스러움.
   Contourlets sidestep curvelets' polar-coordinate awkwardness on Cartesian image grids by being defined directly via discrete filter banks.

3. **LP + DFB의 *분리* 설계가 핵심 / Separating multiscale and multidirection is the key insight** — Multiscale (LP)과 multidirection (DFB)을 *별도 stage*로 분해하면 (i) 각 스케일에서 *다른 수의 방향* 채택 가능, (ii) DFB의 lowpass 약점을 LP가 보완, (iii) tree 구조 일반화 가능 (contourlet packets, Fig. 8).
   Decoupling multiscale (LP) and multidirection (DFB) lets each scale have its own number of directions, addresses DFB's poor low-frequency behaviour, and yields flexible packet generalisations.

4. **Parabolic scaling $\text{width} \propto \text{length}^2$이 $C^2$ curve를 정확히 따라간다 / Parabolic scaling fits $C^2$ contours** — $C^2$ 곡선의 Taylor 전개는 $u(v) \sim (\kappa/2)v^2$, 즉 *너비 ∝ 길이²*. Basis 함수 support도 같은 비율이어야 효율적 표현. Contourlet은 $l_j = l_{j_0} + \lfloor(j_0 - j)/2\rfloor$ 식으로 이 비율 자동 만족.
   Parabolic scaling is the unique aspect-ratio law that locally fits $C^2$ curves; contourlets naturally satisfy it via the rule "double directions every other scale".

5. **Directional Vanishing Moments (DVM)가 wavelet의 vanishing moments 일반화 / DVM generalises wavelet vanishing moments** — Wavelet의 $p$차 vanishing moments는 polynomial 신호에 직교 → polynomial 영역에서 계수가 0. DVM은 *방향별로* 같은 조건 → $C^2$ 영역에서 contourlet 계수가 $O(d^3)$로 작아짐. 결국 $M^{-2}$ NLA rate.
   DVM extends wavelet's polynomial-orthogonality direction-wise, yielding the optimal $M^{-2}$ NLA rate.

6. **Tight frame + $< 4/3$ redundancy / Tight frame, low redundancy** — Wavelet의 critical sampling은 잃지만, redundancy < 4/3 (LP의 4/3 + DFB는 critical). Tight frame이라 Parseval 같은 식 보존 → energy 분석·shrinkage 모두 깔끔.
   Sacrifices critical sampling for a tight frame with redundancy < 4/3, preserving Parseval-like energy decomposition for shrinkage analysis.

7. **$O(N)$ 복잡도 / $O(N)$ complexity** — Wavelet과 동일. Iterated filter bank 구조라 LP는 $O(N)$, DFB도 $O(N)$. 폴리페이즈 / FIR 구현 가능. 따라서 *practical* — wavelet의 *대체*로 즉시 사용 가능.
   Like the DWT, contourlet runs in $O(N)$ — practical drop-in replacement for separable wavelet in many pipelines.

8. **Denoising에서 1 dB 이상 개선 / >1 dB denoising improvement** — Lena 30.47 dB vs wavelet 29.41 dB. 단순 hard thresholding에서도. 더 정교한 inter-coefficient 모델 (Po-Do 2006)은 더 큰 개선. BM3D 같은 최신 기법은 nonlocal + transform-domain 결합 (paper #7) — contourlet의 directional structure는 그 transform-domain step에 직접 활용 가능.
   Even with simple hard thresholding, contourlet beats separable wavelet by >1 dB on Lena denoising; sophisticated inter-coefficient priors push this further.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Wish list / 위시 리스트
1. Multiresolution
2. Localisation (space + frequency)
3. Critical sampling (basis or low-redundancy frame)
4. **Directionality**: many directions
5. **Anisotropy**: elongated supports

### 4.2 Laplacian Pyramid (one level)
$$
a[\mathbf n] = (Hx \downarrow_{\mathbf M})[\mathbf n], \quad b[\mathbf n] = x[\mathbf n] - (Ga\uparrow_{\mathbf M})[\mathbf n]
$$
Iterate: $a_J, b_J, b_{J-1}, \ldots, b_1$. Redundancy ratio: $1 + \sum_{j=1}^J 4^{-j} < 4/3$.

### 4.3 Directional Filter Bank
$l$-level binary tree → $2^l$ wedge-shaped subbands. Sampling matrices (Eq. 3):
$$
\mathbf S^{(l)}_k = \begin{cases} \text{diag}(2^{l-1}, 2) & 0 \le k < 2^{l-1} \\ \text{diag}(2, 2^{l-1}) & 2^{l-1} \le k < 2^l \end{cases}
$$
### 4.4 Discrete contourlet transform
**Algorithm**:
1. $a_0[\mathbf n] = x[\mathbf n]$
2. For $j = 1, \ldots, J$:
   a. LP: $a_{j-1} \to (a_j, b_j)$
   b. DFB: apply $l_j$-level DFB to $b_j$ → contourlet coefficients $c^{(l_j)}_{j,k}[\mathbf n]$, $k = 0, \ldots, 2^{l_j} - 1$
3. Output: $\{c^{(l_j)}_{j,k}[\mathbf n]\}_{j,k,\mathbf n} \cup \{a_J[\mathbf n]\}$

### 4.5 Parabolic scaling (Eq. 29)
$$
l_j = l_{j_0} + \lfloor (j_0 - j)/2 \rfloor, \quad j \le j_0
$$
Yields support sizes:
$$
\text{width} \approx C 2^j, \quad \text{length} \approx C 2^{j + l_j - 2} \approx C 2^{j_0 + l_{j_0}/2 - 2 + j/2}
$$
Ratio width/length² ≈ const.

### 4.6 NLA rate (Theorem 4)
For $f$ piecewise $C^2$ with $C^2$ contours, with parabolic-scaled contourlet + $p$-DVM, $p \ge 2$:
$$
\|f - \hat f_M\|^2_2 \lesssim (\log M)^3 M^{-2}
$$
Compared to:
- Separable wavelets: $O(M^{-1})$
- Fourier: $O(M^{-1/2})$

### 4.7 Worked example / 수치 예시
Lena $512 \times 512$, 6 LP levels, DFB levels $l_j \in \{3, 3, 4, 4, 5, 5\}$ (parabolic scaling). Total directional subbands ≈ $2^3 \cdot 2 + 2^4 \cdot 2 + 2^5 \cdot 2 = 16 + 32 + 64 = 112$ directional bands, much richer than wavelet's $3 \cdot 6 = 18$ detail subbands.

For PSNR-26 dB hard thresholding (paper Fig. 17): wavelet 29.41 dB vs contourlet 30.47 dB → MSE ratio $10^{0.106} \approx 1.28$, so contourlet has 22% lower MSE.

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1962 ─── Hubel-Wiesel — V1 simple cells: local, oriented, multiscale
1983 ─── Burt-Adelson — Laplacian Pyramid
1987 ─── Daugman — 2-D Gabor wavelets
1989 ─── Mallat — multiresolution wavelet analysis
1992 ─── Bamberger-Smith — Directional Filter Bank (DFB)
1996 ─── Olshausen-Field — natural-image sparse coding favors localised+oriented
1999 ─── Donoho — wedgelets (edge-adaptive)
2000 ─── Candès-Donoho — continuous-domain ridgelets, curvelets
2003 ─── Do-Vetterli — Framing pyramids (LP → tight frame)
2004 ─── Candès-Donoho — second-generation curvelets
2005 ★★ DO-VETTERLI — Contourlet Transform (THIS PAPER)
                        ↳ discrete-domain directional multiresolution
                        ↳ LP + DFB
2005 ─── Candès-Demanet-Donoho-Ying — Fast Discrete Curvelet (paper #6)
                        ↳ digital curvelet implementation
2006 ─── Po-Do — Directional multiscale modeling using HMM on contourlets
2008 ─── Easley-Labate-Lim — Discrete Shearlet (paper #8)
                        ↳ shearing instead of rotation
2010+ ── Curvelet/contourlet/shearlet largely superseded by:
                        ↳ BM3D (paper #7) — nonlocal + transform-domain
                        ↳ DnCNN/Restormer — learned filters
```

**위치**: Contourlet은 *연속영역 curvelet의 이산 격자 친화적 구현*. Curvelet (paper #6)과 Shearlet (paper #8)은 사촌 — 같은 wish list, 다른 구현. 이후 BM3D가 nonlocal patch grouping + transform-domain shrinkage의 결합으로 시각·MSE 모두 압도, deep learning이 학습된 directional filters로 대체.

This paper is contemporaneous with second-gen curvelet and slightly precedes shearlet — three approaches to the same wish list, differing in whether they use rotations, shears, or filter banks.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **Burt-Adelson (1983)** | Laplacian Pyramid | The LP stage of the contourlet cascade is a direct adoption of the original LP. |
| **Bamberger-Smith (1992)** *IEEE TSP* | Directional Filter Bank | The DFB is a simplified version of Bamberger-Smith's tree-structured DFB. |
| **Mallat (1989)** *IEEE PAMI* | Wavelet MRA | Contourlet's continuous-domain DMRA framework is the directional generalisation. |
| **Candès-Donoho (2004)** *Comm. Pure Appl. Math.* | Continuous curvelets | Contourlet is the discrete-domain analog with similar parabolic scaling and approximation rates. |
| **Donoho-Johnstone (1994)** *Biometrika* (paper #1) | Wavelet thresholding | Contourlet thresholding uses the same shrinkage logic on contourlet coefficients. |
| **Chang-Yu-Vetterli (2000)** *IEEE TIP* (paper #3) | BayesShrink | BayesShrink generalises naturally to contourlet subbands; same $\sigma^2/\sigma_X$ formula per directional subband. |
| **Candès+ (2006)** *Multiscale Model. Simul.* (paper #6) | Fast Discrete Curvelet | Sister approach: rotation-based partitioning vs contourlet's filter-bank approach. |
| **Easley+ (2008)** *ACHA* (paper #8) | Discrete Shearlet | Sister: uses shearing instead of rotation/filter banks; affine-group-friendly. |
| **Dabov+ (2007)** *IEEE TIP* (paper #7) | BM3D | Combines nonlocal grouping with transform shrinkage; contourlet structure could replace DCT in BM3D for directional images. |
| **Po-Do (2006)** *IEEE TIP* | Contourlet HMM | Successor work modelling inter-scale contourlet dependencies. |

---

## 7. References / 참고문헌

- Bamberger, R. H., & Smith, M. J. T., "A filter bank for the directional decomposition of images: theory and design", *IEEE TSP*, 40(4), 882–893 (1992).
- Burt, P. J., & Adelson, E. H., "The Laplacian pyramid as a compact image code", *IEEE Trans. Communications*, COM-31(4), 532–540 (1983).
- Candès, E. J., & Donoho, D. L., "New tight frames of curvelets and optimal representations of objects with piecewise $C^2$ singularities", *Comm. Pure Appl. Math.*, 57, 219–266 (2004).
- Do, M. N., & Vetterli, M., "Framing pyramids", *IEEE TSP*, 51(9), 2329–2342 (2003).
- Do, M. N., & Vetterli, M., "The contourlet transform: an efficient directional multiresolution image representation", *IEEE TIP*, 14(12), 2091–2106 (2005). [DOI: 10.1109/TIP.2005.859376]
- Donoho, D. L., "Wedgelets: nearly-minimax estimation of edges", *Annals of Statistics*, 27, 859–897 (1999).
- Mallat, S., *A Wavelet Tour of Signal Processing*, 2nd ed., Academic Press (1999).
- Olshausen, B. A., & Field, D. J., "Emergence of simple-cell receptive field properties by learning a sparse code for natural images", *Nature*, 381, 607–609 (1996).
