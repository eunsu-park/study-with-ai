---
title: "Sparse Directional Image Representations using the Discrete Shearlet Transform"
authors: Glenn Easley, Demetrio Labate, Wang-Q Lim
year: 2008
journal: "Applied and Computational Harmonic Analysis 25(1), pp. 25–46"
doi: "10.1016/j.acha.2007.09.003"
topic: Low-SNR Imaging / Directional Multiscale Transforms
tags: [shearlet, composite-wavelets, shear, parabolic-scaling, parseval-frame, directional-transform, easley-labate-lim, discrete-shearlet-transform, multiresolution-analysis]
status: completed
date_started: 2026-05-06
date_completed: 2026-05-06
---

# 8. Sparse Directional Image Representations using the Discrete Shearlet Transform / 이산 쉬어릿 변환에 의한 희소 방향성 영상 표현

---

## 1. Core Contribution / 핵심 기여

### 한국어
이 논문은 **shearlet transform**의 첫 번째 *이산 구현*을 제시한다. Shearlet은 **composite wavelet** 이론의 특수 사례 — *anisotropic dilation* $A_0$와 **shear** $B_0$의 두 행렬 군으로 생성:
$$
A_0 = \begin{pmatrix} 4 & 0 \\ 0 & 2 \end{pmatrix}, \qquad B_0 = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}
$$
**Shearlet 시스템** (Eq. 2.6): $\psi^{(0)}_{j, \ell, k}(x) = 2^{3j/2} \psi^{(0)}(B_0^\ell A_0^j x - k)$, $j \ge 0$, $-2^j \le \ell \le 2^j - 1$, $k \in \mathbb Z^2$. $\mathcal D_0$ (수평 cone)에 대한 Parseval frame 형성.

핵심 기여 4가지:

(A) **Composite wavelet 프레임워크의 수학적 단순성**: Curvelet은 *rotation* $R_\theta$를 사용해 단일 mother로부터 곡선 분포 생성 — 격자에 부자연. Shearlet은 *shear* $B_0^\ell$ — 격자 보존 변환. 따라서 *유한한 연산자 군*이 단일 함수에 작용해 시스템 생성 (wavelet의 직접 일반화).

(B) **Optimal sparsity** (Theorem of [19]): piecewise $C^2$ image with $C^2$ curve singularities에 대해 $\|f - f^S_N\|^2 \le CN^{-2}(\log N)^3$. 이는 curvelet과 같은 *최적* NLA rate; wavelet의 $O(N^{-1})$을 결정적으로 능가.

(C) **이산 변환 알고리즘** (Section 3): Laplacian Pyramid (multi-scale) + pseudo-polar grid에서의 directional band-pass filtering (multi-direction). $O(N^2 \log N)$ 복잡도.

(D) **Curvelet 대비 직사각 격자 친화** + **방향 수가 매 fine scale마다 doubling**: 커블릿은 *every other* scale에서 doubling. Shearlet은 *every* finer scale에서 doubling — 동일 $j$에서 더 fine한 angular 해상도.

### English
First **discrete implementation** of the shearlet transform — a special case of *composite wavelets* with anisotropic dilation $A_0$ and shear $B_0$. The system $\{\psi^{(0)}_{j,\ell,k}(x) = 2^{3j/2}\psi^{(0)}(B_0^\ell A_0^j x - k)\}$ forms a Parseval frame.

Four contributions: (A) shears (vs. rotations) make shearlets natural on Cartesian grids; (B) optimal sparsity $N^{-2}(\log N)^3$ (curvelet-equivalent); (C) discrete algorithm via LP + pseudo-polar directional filtering, $O(N^2 \log N)$; (D) directional resolution doubles at every finer scale (vs. every other for curvelets).

---

## 2. Reading Notes / 읽기 노트

### Part I: §1 Motivation / 동기

#### 한국어
- Wavelet의 한계: $C^2$ image with $C^2$ edges에서 $\varepsilon_M \le CM^{-1}$ (cf. $C^2$ smooth면 $M^{-2}$). 즉 edge가 wavelet representation을 *비효율적*으로 만듦.
- Curvelet (Candès-Donoho 2004): rotation 기반, $\varepsilon_M \le C(\log M)^3 M^{-2}$ — 최적. 하지만 polar geometry로 이산화 어려움.
- Contourlet (Do-Vetterli 2005, paper #5): filter bank 기반. Curvelet의 *이산화 시도*. 그러나 본질적으로 *이산 시간 구조*만, MRA 미존재.
- Shearlet 답: composite wavelet 이론에서 shear $B_0$와 dilation $A_0$ 군으로 시스템 정의 → 이산화 자연스러움 + MRA 연결 가능.

#### English
Wavelets give $M^{-1}$, curvelets $(\log M)^3 M^{-2}$ but with awkward polar geometry, contourlets discretise but lack MRA. Shearlets bridge: composite-wavelet construction with shears (Cartesian-friendly) + connection to MRA.

---

### Part II: §2 Shearlet Definition / 정의

#### 한국어 — Composite wavelets

$L^2(\mathbb R^2)$의 *affine systems with composite dilations* (Eq. 2.2):
$$
\mathcal A_{AB}(\psi) = \{|\det A|^{j/2} \psi(B^\ell A^j x - k): j, \ell \in \mathbb Z, k \in \mathbb Z^2\}
$$
$A^j$은 *scale*, $B^\ell$은 *area-preserving geometric transformation* (rotation, shear).

**Shearlet 특수 사례**:
$$
A_0 = \begin{pmatrix} 4 & 0 \\ 0 & 2 \end{pmatrix} \quad \text{(parabolic scaling)}, \qquad B_0 = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix} \quad \text{(unit shear)}
$$
$A_0$: $\xi_1$ 방향 4× scale, $\xi_2$ 방향 2× — **parabolic scaling** (width $\sim 2^{2j}$, length $\sim 2^j$ → width ∝ length²).
$B_0$: 단위 shear — $(\xi_1, \xi_2) \to (\xi_1, \xi_1 + \xi_2)$.

#### 한국어 — Mother shearlet
$$
\hat\psi^{(0)}(\xi) = \hat\psi_1(\xi_1) \hat\psi_2(\xi_2/\xi_1) \quad (2.3)
$$
$\hat\psi_1$: $[-1/2, -1/16] \cup [1/16, 1/2]$에서 $C^\infty$ (radial 부분).
$\hat\psi_2$: $[-1, 1]$에서 $C^\infty$ (angular 부분).

**Admissibility** (Eq. 2.4-2.5):
$$
\sum_j |\hat\psi_1(2^{-2j}\omega)|^2 = 1, \quad \sum_\ell |\hat\psi_2(2^j \omega - \ell)|^2 = 1
$$
**Frequency support** (Fig. 1b): trapezoid 폭 $2^j$, 길이 $2^{2j}$, 기울기 $\ell 2^{-j}$. Spatial support는 $2^{-j} \times 2^{-2j}$ (Fourier-spatial duality).

#### 한국어 — Two cones $\mathcal D_0, \mathcal D_1$
- $\mathcal D_0$: $|\xi_1| \ge 1/8, |\xi_2/\xi_1| \le 1$ (수평 콘) — $\psi^{(0)}$ Parseval frame
- $\mathcal D_1$: $|\xi_2| \ge 1/8, |\xi_1/\xi_2| \le 1$ (수직 콘) — $\psi^{(1)}$ Parseval frame
$$
A_1 = \begin{pmatrix} 2 & 0 \\ 0 & 4 \end{pmatrix}, \quad B_1 = \begin{pmatrix} 1 & 0 \\ 1 & 1 \end{pmatrix}
$$
**전체 합** (Theorem 2.1): $\{\varphi_k\} \cup \{\psi^{(d)}_{j,\ell,k}: d=0,1, ...\}$는 $L^2(\mathbb R^2)$의 Parseval frame. Coarse-scale $\varphi$ + 두 cone의 fine-scale shearlets.

#### 한국어 — Shearlet의 5가지 성질

1. **Well localized**: frequency-domain compact support, spatial fast decay.
2. **Parabolic scaling**: $\hat\psi_{j,\ell,k}$ supported on $2^j \times 2^{2j}$ box → spatial support $2^{-j} \times 2^{-2j}$.
3. **Highly directional sensitivity**: $\psi_{j,\ell,k}$ oriented along $\ell 2^{-j}$ slope. Number of orientations doubles at every finer $j$.
4. **Spatially localised**: translations on lattice $\mathbb Z^2$.
5. **Optimally sparse** (Theorem from [19]):
$$
\|f - f^S_N\|^2_2 \le CN^{-2}(\log N)^3 \quad \text{for } f \in C^2 \text{ minus piecewise } C^2 \text{ curves}
$$
#### English — Shearlet system
Defined by composite wavelets with anisotropic dilation $A_0$ and unit shear $B_0$. Parabolic scaling (length² ∝ width), Cartesian-domain natural, directional sensitivity doubles at every scale, optimally sparse for $C^2$ edges.

---

### Part III: §3 Discrete Shearlet Transform / 이산 변환

#### 한국어 — §3.1 Frequency-domain implementation

전체 알고리즘 (Fig. 2):
1. **Laplacian Pyramid**: $f_d^{j-1} \to (f_a^j, f_d^j)$. 본 논문은 LP를 multi-scale 분해에만 사용 (anisotropic subsampling 없음).
2. **2-D DFT** of bandpass $f_d^j$.
3. **Pseudo-polar grid**: $(\xi_1, \xi_2) \to (u, v)$, $u = \xi_1, v = \xi_2/\xi_1$ ($\mathcal D_0$) 또는 $u = \xi_2, v = \xi_1/\xi_2$ ($\mathcal D_1$).
4. **1-D band-pass filtering**: $g_j(u, v) \overline{W(2^j v - \ell)}$ — pseudo-polar grid의 *각도 변수* $v$에 대한 1-D band-pass.
5. **Inverse 2-D FFT**: shearlet 계수 $\langle f, \psi^{(0)}_{j,\ell,k}\rangle$ (Eq. 3.12).

**Pseudo-polar DFT (PDFT, Definition 3.1)**: $\hat f_1[k_1, k_2], \hat f_2[k_1, k_2]$ — Cartesian DFT의 lines along 원점 통과 *변광 기울기*. $O(N^2 \log N)$ via fast slant stack algorithm 또는 직접 FFT 추출.

#### 한국어 — §3.2 Correlation with theory
- 본 이산 구현은 *anisotropic subsampling*을 *적용하지 않음* → 매우 redundant.
- Three-level decomposition: $2^j N^2 + 2^{j-1}(N/4)^2 + (N/16)^2$ coefficients ($2^j$ directional subbands at coarsest, more at finer).
- 그래도 $N^{-1+p}$ ($p > 0$ 임의)의 NLA decay rate 실험적 확인 → 이론적 $N^{-2}$와 일치 (anisotropic subsampling 추가 시).

#### 한국어 — §3.3 Time-domain implementation
대안: window를 *시간 영역*에 직접 정의 → wavelet 필터 bank로 구현. 더 sparse한 representation 가능.

#### English — Discrete shearlet
LP for multi-scale + pseudo-polar grid + 1-D band-pass filter for directional. Frequency-domain implementation $O(N^2 \log N)$, highly redundant (no anisotropic subsampling).

---

### Part IV: §4 Numerical Experiments / 수치 실험

#### 한국어 (간략)
- Lena, Goldhill 등 표준 이미지에 hard-thresholding denoising 적용
- Wavelet (db4) vs Shearlet 비교 — Shearlet이 +0.5-1.5 dB PSNR 우수
- NLA: 같은 $M$ coefficient로 reconstruction 시 shearlet이 wavelet보다 시각·MSE 모두 우수
- Curvelet과 비슷한 성능, 그러나 *Cartesian-natural*이라 구현 단순.

#### English
On standard test images, shearlet hard-thresholding outperforms wavelet by 0.5-1.5 dB and matches curvelet within ~0.2 dB while being far simpler to implement on grids.

---

## 3. Key Takeaways / 핵심 시사점

1. **Shears는 회전의 격자 친화적 대체 / Shears replace rotations on Cartesian grids** — $B_0 = \begin{pmatrix}1&1\\0&1\end{pmatrix}$은 단위 격자를 단위 격자로 매핑 → 이산화 자연스러움. $R_\theta$는 격자를 격자로 안 매핑 → 보간 필요. Shears는 *area-preserving group action* — 이론적으로도 우아.
   Shears ($B_0^\ell$) preserve the integer lattice, making shearlet construction discrete-grid-natural; rotations require interpolation.

2. **Composite wavelet 프레임워크는 wavelet의 직접 일반화 / Composite wavelets are a direct generalisation of wavelets** — 일반 wavelet $\psi(A^j x - k)$에 두 번째 군 $B^\ell$ 추가만. 따라서 wavelet 이론 (MRA, vanishing moments, smoothness)이 모두 자연스럽게 일반화. Curvelet은 polar 좌표라 이런 연결이 어려움.
   Composite wavelets simply add a second matrix group $B^\ell$ to wavelets' $A^j$ — preserving the entire wavelet framework (MRA, vanishing moments).

3. **방향 수가 *매 fine scale마다* doubling / Directional resolution doubles at every scale** — Curvelet은 *every other* scale에서 (parabolic scaling 만족 위해 $2^j$ directions). Shearlet은 *every* scale에서 doubling (parabolic scaling은 $A_0$ anisotropy로 자동 만족). 결과: 같은 spatial scale에서 더 많은 방향.
   Shearlet doubles directions at every scale (vs every other for curvelets), giving finer angular resolution at the same spatial scale.

4. **Optimal NLA rate $(\log N)^3 N^{-2}$ / Curvelet-equivalent NLA** — $C^2$ image with $C^2$ curve singularities에 대해. Wavelet의 $N^{-1}$을 능가, curvelet과 동등. Shearlet의 *parabolic scaling*과 *vanishing moments*가 함께 이 rate 보장.
   Shearlets achieve the same optimal NLA rate as curvelets — the parabolic scaling + vanishing moments combination is what counts.

5. **이산 알고리즘은 LP + pseudo-polar / Discrete algorithm: LP + pseudo-polar** — LP가 multi-scale, pseudo-polar grid가 directional. 둘 다 standard tools — fast slant stack, FFT extraction, etc. $O(N^2 \log N)$. Curvelet의 wrapping FDCT와 *유사한 비용*.
   Discrete shearlet uses Laplacian Pyramid + pseudo-polar grid (fast slant stack). $O(N^2\log N)$, comparable to curvelet wrapping FDCT.

6. **Anisotropic subsampling이 redundancy의 핵심 / Anisotropic subsampling is the key to low redundancy** — 본 논문 구현은 *anisotropic 다운샘플링 없음* → 고도로 redundant. 이후 논문들 (Lim 2010, Häuser-Kutyniok 2013)에서 anisotropic subsampling 추가 → critically sampled / low-redundancy shearlet 가능.
   This paper's implementation lacks anisotropic subsampling; subsequent work (e.g. Lim 2010) introduces it for low-redundancy shearlets.

7. **Curvelet, Contourlet, Shearlet은 같은 wish list의 세 답 / Curvelet, Contourlet, Shearlet are three answers to the same wish list** — Multiscale + directional + parabolic scaling + tight frame + Cartesian discretisation. Curvelet: rotation + Fourier wedge. Contourlet: LP + DFB filter bank. Shearlet: shear + LP + pseudo-polar. *대체로 비슷한 PSNR* — 선택은 응용 + 사용자 친숙도에 따라.
   The three directional transforms achieve similar PSNR; the choice is mostly about ease of use and theoretical preference.

8. **Deep learning이 모두 능가하지만 / Deep learning supersedes all three but** — 2015+ 시점에서 DnCNN/Restormer가 모든 directional transforms를 능가. 하지만 (i) training data 필요, (ii) sparse representation theory와의 연결 약함, (iii) 작은 image / domain shift에서 실패. Shearlet/Curvelet/Contourlet은 *수학적 보증 + non-learned*로 의료·과학 영역에서 여전히 유효.
   Deep learning beats all three transforms, but they remain valuable in scientific imaging (medical, astrophysical) where training data is scarce and mathematical guarantees matter.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Composite wavelets / 복합 웨이블릿
$$
\mathcal A_{AB}(\psi) = \{|\det A|^{j/2}\psi(B^\ell A^j x - k): j, \ell \in \mathbb Z, k \in \mathbb Z^2\}
$$
Parseval frame condition: $\sum_{j,\ell,k} |\langle f, \psi_{j,\ell,k}\rangle|^2 = \|f\|^2$.

### 4.2 Shearlet matrices / 쉬어릿 행렬
$$
A_0 = \begin{pmatrix}4&0\\0&2\end{pmatrix}, \quad B_0 = \begin{pmatrix}1&1\\0&1\end{pmatrix} \quad (\text{horizontal cone}\,\mathcal D_0)
$$
$$
A_1 = \begin{pmatrix}2&0\\0&4\end{pmatrix}, \quad B_1 = \begin{pmatrix}1&0\\1&1\end{pmatrix} \quad (\text{vertical cone}\,\mathcal D_1)
$$
### 4.3 Mother shearlet / 모-쉬어릿
$$
\hat\psi^{(0)}(\xi_1, \xi_2) = \hat\psi_1(\xi_1)\hat\psi_2(\xi_2/\xi_1)
$$
with $\hat\psi_1 \in C^\infty$ on $[-1/2, -1/16] \cup [1/16, 1/2]$, $\hat\psi_2 \in C^\infty$ on $[-1, 1]$.

### 4.4 Shearlet system (Theorem 2.1)
$$
\{\varphi_k\} \cup \{\psi^{(d)}_{j,\ell,k}: j \ge 0, -2^j+1 \le \ell \le 2^j-2, k\} \cup \{\tilde\psi^{(d)}_{j,\ell,k}\}_{d=0,1}
$$
forms a Parseval frame for $L^2(\mathbb R^2)$.

### 4.5 Optimal NLA rate
$$
\|f - f^S_N\|^2 \le CN^{-2}(\log N)^3 \quad (C^2\text{-piecewise smooth, }C^2\text{ singularity curves})
$$
### 4.6 Discrete shearlet algorithm
**Input**: image $f \in \ell^2(\mathbb Z_N^2)$.
**Step 1**: Laplacian Pyramid → bandpass $\{f_d^j\}$ and final lowpass.
**Step 2**: For each $j$, compute $\hat f_d^j$ on pseudo-polar grid:
  \[(u, v) = (\xi_1, \xi_2/\xi_1) \text{ on }\mathcal D_0, \quad (\xi_2, \xi_1/\xi_2) \text{ on }\mathcal D_1\]
**Step 3**: For each direction $\ell$, apply 1-D band-pass filter $W(2^j v - \ell)$ along $v$.
**Step 4**: Inverse FFT (or inverse PDFT) → shearlet coefficients $\langle f, \psi_{j,\ell,k}\rangle$.

Complexity: $O(N^2 \log N)$ via fast slant stack or direct extraction.

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1989 ─── Mallat — wavelet MRA
2000-04 ─ Candès-Donoho — first/second-gen curvelets
2005 ─── Do-Vetterli — Contourlet (paper #5)
2005 ─── Guo-Kutyniok-Labate — Composite wavelets framework introduced
2006 ─── Candès-Demanet-Donoho-Ying — Fast Discrete Curvelet (paper #6)
2007 ─── Labate-Lim-Kutyniok-Weiss — first shearlet paper (continuous)
2008 ★★ EASLEY-LABATE-LIM — Discrete Shearlet (THIS PAPER)
                              ↳ Cartesian-natural via shears
                              ↳ LP + pseudo-polar discrete algorithm
2010 ─── Lim — anisotropic-subsampled critically-sampled shearlet
2013 ─── Häuser-Kutyniok — α-shearlet (parameter family)
2017+ ── ShearLab software (shearlab.org), `pyshearlab` Python port
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **Do-Vetterli (2005)** *IEEE TIP* (paper #5) | Contourlet | Contourlet is filter-bank based; shearlet is composite-wavelet based. Both achieve similar tiling. |
| **Candès+ (2006)** *Multiscale Model. Simul.* (paper #6) | FDCT | Same NLA rate; shearlet is Cartesian-natural via shears, curvelet uses rotations. |
| **Guo-Kutyniok-Labate (2005)** | Composite wavelets framework | Provides the mathematical foundation that shearlets specialise. |
| **Donoho-Johnstone (1994)** *Biometrika* (paper #1) | Wavelet thresholding | Shearlet thresholding follows the same framework; coefficients are simply replaced. |
| **Burt-Adelson (1983)** | Laplacian Pyramid | The multi-scale stage of the discrete shearlet. |
| **Dabov+ (2007)** *IEEE TIP* (paper #7) | BM3D | Different paradigm; could combine via shearlet-3D in BM3D's collaborative filtering. |
| **Lim (2010), Kutyniok+ (2013)** | Critically-sampled shearlet | Successor work introducing anisotropic subsampling for low redundancy. |
| **ShearLab (2014+)** | Reference software | Matlab + Python implementations available at shearlab.org. |

---

## 7. References / 참고문헌

- Burt, P. J., & Adelson, E. H., "The Laplacian pyramid as a compact image code", *IEEE Trans. Communications*, COM-31(4), 532–540 (1983).
- Candès, E. J., Demanet, L., Donoho, D. L., & Ying, L., "Fast discrete curvelet transforms", *Multiscale Modeling & Simulation*, 5(3), 861–899 (2006).
- Do, M. N., & Vetterli, M., "The contourlet transform: an efficient directional multiresolution image representation", *IEEE TIP*, 14(12), 2091–2106 (2005).
- Easley, G., Labate, D., & Lim, W.-Q., "Sparse directional image representations using the discrete shearlet transform", *Applied and Computational Harmonic Analysis*, 25(1), 25–46 (2008). [DOI: 10.1016/j.acha.2007.09.003]
- Guo, K., Kutyniok, G., & Labate, D., "Sparse multidimensional representations using anisotropic dilation and shear operators", in *Wavelets and Splines* (2005).
- Labate, D., Lim, W.-Q., Kutyniok, G., & Weiss, G., "Sparse multidimensional representation using shearlets", *SPIE Wavelets XI*, 5914 (2007).
- Lim, W.-Q., "The discrete shearlet transform: a new directional transform and compactly supported shearlet frames", *IEEE TIP*, 19(5), 1166–1180 (2010).
- Mallat, S., *A Wavelet Tour of Signal Processing*, 2nd ed., Academic Press (1999).
