---
title: "Fast Discrete Curvelet Transforms"
authors: Emmanuel J. Candès, Laurent Demanet, David L. Donoho, Lexing Ying
year: 2006
journal: "Multiscale Modeling & Simulation 5(3), pp. 861–899"
doi: "10.1137/05064182X"
topic: Low-SNR Imaging / Directional Multiscale Transforms
tags: [curvelet, fdct, fast-discrete-curvelet-transform, wrapping, usfft, parabolic-scaling, candes-donoho, second-generation-curvelet, cartesian-coronization, optimal-nla]
status: completed
date_started: 2026-05-06
date_completed: 2026-05-06
---

# 6. Fast Discrete Curvelet Transforms / 고속 이산 컬블릿 변환

---

## 1. Core Contribution / 핵심 기여

### 한국어
이 논문은 *연속영역에서 정의된 second-generation curvelet*의 **두 가지 디지털 구현(FDCT)**을 제시한다:

(A) **FDCT via USFFT** (unequally spaced fast Fourier transform): 연속이론에 더 충실한 비균등 격자 보간 기반. 회전된 polar wedge $U_j(R_{\theta_\ell}\omega)$를 직접 계산.

(B) **FDCT via wrapping**: Cartesian 격자에 자연스러운 *shear*-based 구현. 핵심 단계:
1. 영상에 2-D FFT 적용
2. *Cartesian wedge window* $\tilde U_{j, \ell}(\omega) = \tilde W_j(\omega) V_j(S_{\theta_\ell}\omega)$와 multiply
3. *Wrapping*: shear된 wedge를 unit periodic cell로 wrapping
4. 역 2-D FFT → curvelet 계수 $c^D(j, \ell, k)$

핵심 성능: $O(n^2 \log n)$ flops — FFT의 6-10배. *Numerical isometry* (wrapping FDCT는 정확히 tight frame).

(C) **Cartesian coronization**: polar 좌표 (Hubel-Wiesel 같은 회전)을 *concentric squares*로 대체. 회전 $R_\theta$ → shear $S_\theta = \begin{pmatrix} 1 & 0 \\ -\tan\theta & 1 \end{pmatrix}$. 이는 직사각 격자에 자연스러운 변환을 가능하게 함.

(D) **Three key motivations** (§1.2):
1. **Curve-punctuated smoothness**의 sparse 표현: $\|f - f_m\|^2 \le C(\log m)^3 m^{-2}$ — wavelets의 $O(m^{-1})$을 결정적으로 능가.
2. **Wave propagator**의 sparse 표현: hyperbolic PDE의 solution operator가 curvelet domain에서 *near-diagonal* (Smith 1998).
3. **Ill-posed inverse problems** (CT, deconvolution): 잡음·결손 데이터에서 *식별 가능한 부분*과 *식별 불가능한 부분*을 깔끔히 분리.

(E) **CurveLab 소프트웨어** (curvelet.org): 본 논문의 모든 알고리즘 (USFFT, wrapping, 3D 변환) Matlab + C++ 공개 구현.

### English
The paper provides two digital implementations of the *second-generation curvelet transform* — both running in $O(n^2 \log n)$ and 6–10× the cost of an FFT:

(A) **FDCT via USFFT**: closer to continuous theory; uses unequally-spaced FFTs to approximate rotated polar wedges.

(B) **FDCT via wrapping**: simpler, faster, *numerically tight frame*. Multiplies the image FFT by Cartesian wedge windows (using shears instead of rotations), wraps the support into a unit cell, then inverse-FFTs to obtain curvelet coefficients.

(C) Replaces polar coordinates with **Cartesian coronization** based on concentric squares and shears, making the transform discrete-grid-friendly.

(D) Curvelets achieve **optimal nonlinear approximation rate** $(\log m)^3 m^{-2}$ for piecewise smooth functions with $C^2$-singularity curves, beating wavelets' $O(m^{-1})$.

(E) Reference implementation: **CurveLab** (Matlab + C++) at curvelet.org.

---

## 2. Reading Notes / 읽기 노트

### Part I: §1 Why curvelets / 왜 컬블릿인가

#### 한국어
- **Wavelet의 한계**: 점 특이점 (delta function) 표현엔 최적이지만, *곡선·면 따른 특이점*은 비효율적. $C^2$ 곡선 따라 불연속인 함수에서 wavelet은 *isolated dots*로 곡선을 추적 → $O(m^{-1})$ NLA rate.
- **Three motivations**:
  1. **Edge representation**: $\|f - f_m\|^2 \le C(\log m)^3 m^{-2}$ — 같은 $M$개 계수로 wavelet보다 *훨씬* 작은 오차. 잡음 제거 시 wavelet보다 MSE *order of magnitude* 우수.
  2. **Wave propagators**: 선형 hyperbolic PDE $\partial_t u + \sum A_k \partial_{x_k} u + Bu = 0$의 solution operator $E_t$가 curvelet domain에서 *sparse + well-organized* — 주대각 근방에 빠르게 감쇠. Curvelet은 hyperbolic operator의 *near-eigenfunction*.
  3. **Ill-posed inverse problems**: tomographic reconstruction에서 데이터 누락 부분의 *식별 가능 영역*을 curvelet expansion이 깔끔히 분리.

#### English
Curvelets address three problems where wavelets are sub-optimal: edge representation ($(\log m)^3 m^{-2}$ vs $m^{-1}$), wave-propagator sparsification (curvelets are near-eigenfunctions of hyperbolic PDEs), and ill-posed inverse problems (microlocal phase-space localisation).

---

### Part II: §2 Continuous-time Curvelet Transform / 연속영역 변환

#### 한국어 — Construction

**Radial window** $W(r)$ and **angular window** $V(t)$: smooth, supported on $r \in (1/2, 2)$ and $t \in [-1, 1]$, satisfy admissibility:
$$
\sum_{j=-\infty}^\infty W^2(2^j r) = 1 \quad (2.1), \qquad \sum_{\ell=-\infty}^\infty V^2(t - \ell) = 1 \quad (2.2)
$$
**Frequency window for scale $j$**:
$$
U_j(r, \theta) = 2^{-3j/4} W(2^{-j} r) V\left(\frac{2^{\lfloor j/2\rfloor}\theta}{2\pi}\right) \quad (2.3)
$$
Polar "wedge" — angular width $2\pi \cdot 2^{-\lfloor j/2\rfloor}$ (decreases as scale increases), radial width $\sim 2^j$.

**Mother curvelet** $\varphi_j$ defined via FT $\hat\varphi_j(\omega) = U_j(\omega)$. **Curvelets at all scales/angles/positions**:
$$
\varphi_{j,\ell,k}(x) = \varphi_j\bigl(R_{\theta_\ell}(x - x^{(j,\ell)}_k)\bigr)
$$
$$
\theta_\ell = 2\pi \cdot 2^{-\lfloor j/2\rfloor}\ell, \quad x^{(j,\ell)}_k = R_{\theta_\ell}^{-1}(k_1 \cdot 2^{-j}, k_2 \cdot 2^{-j/2})
$$
Sampling grid is *anisotropic*: spacing $2^{-j}$ along ridge, $2^{-j/2}$ across.

#### 한국어 — Properties (§2 list)

1. **Tight frame** (Eq. 2.6-2.7):
$$
f = \sum_{j, \ell, k} \langle f, \varphi_{j,\ell,k}\rangle \varphi_{j,\ell,k}, \quad \sum |\langle f, \varphi_{j,\ell,k}\rangle|^2 = \|f\|^2
$$
2. **Parabolic scaling** (Eq. 2.8): $\text{length} \approx 2^{-j/2}$, $\text{width} \approx 2^{-j}$ → $\text{width} \approx \text{length}^2$.
3. **Oscillatory behavior**: ridge 방향으로 진동, 직각 방향 lowpass.
4. **Vanishing moments** (Eq. 2.9): $\int \varphi_j(x_1, x_2)\,x_1^n\,dx_1 = 0$ for $0 \le n < q$. 무한한 vanishing moments (compact support away from origin in frequency).

#### English — Continuous Curvelet
Curvelets are localised in space, scale, and angle, with parabolic-scaling $\text{length}^2 \propto \text{width}$ and infinitely many vanishing moments. They form a tight frame for $L^2(\mathbb R^2)$.

---

### Part III: §3 Digital Curvelet Transforms / 디지털 변환

#### 한국어 — §3.1 Cartesian Coronization

연속이론의 *polar* 좌표는 직사각 디지털 격자에 부자연스러움. 해결:
- **Concentric squares** 대신 concentric circles → $\Phi_j(\omega) = \phi(2^{-j}\omega_1)\phi(2^{-j}\omega_2)$, $\tilde W_j = \sqrt{\Phi_{j+1}^2 - \Phi_j^2}$.
- **Shear** $S_\theta = \begin{pmatrix} 1 & 0 \\ -\tan\theta & 1 \end{pmatrix}$ 대신 회전 $R_\theta$.
- Cartesian wedge: $\tilde U_{j, \ell}(\omega) = \tilde W_j(\omega) V(2^{\lfloor j/2\rfloor} S_{\theta_\ell}\omega \cdot e_1 / S_{\theta_\ell}\omega \cdot e_2)$.
- 결과 **concentric tiling** (Fig. 2): polar 와 매우 유사하지만 격자 친화.

#### 한국어 — §4 USFFT-based FDCT (algorithm)
1. $\hat f = \mathrm{FFT}_{2D}(f)$.
2. 각 $(j, \ell)$에 대해:
   a. Multiply $\hat f$ by $\tilde U_{j, \ell}$.
   b. Wrap (or interpolate via USFFT) the wedge support to $[0, L_{1,j}] \times [0, L_{2,j}]$.
   c. Inverse FFT → curvelet coefficients $c^D(j, \ell, k)$.
3. Coarse scale: 별도 lowpass.
4. 출력: $c^D(j, \ell, k_1, k_2)$.

#### 한국어 — §6 Wrapping-based FDCT (the practical algorithm)

핵심 통찰: shear된 wedge $\tilde U_{j, \ell}$의 spatial support는 *기울어진 직사각형*. 이를 *axis-aligned* 직사각형으로 *wrapping* (periodicization).

**알고리즘**:
1. $\hat f = \mathrm{FFT}_{2D}(f) / n^2$.
2. For each $j$, $\ell$:
   a. $\tilde f_{j,\ell}[\omega_1, \omega_2] = \hat f[\omega_1, \omega_2] \tilde U_{j,\ell}[\omega_1, \omega_2]$.
   b. *Wrap* $\tilde f_{j,\ell}$ to the rectangle $[0, L_{1,j}) \times [0, L_{2,j})$ (periodic shift).
   c. $c^D(j, \ell, k) = \mathrm{IFFT}_{2D}(\text{wrapped } \tilde f_{j,\ell})$.

**Properties**: numerical isometry (sum of $|c^D|^2$ = sum of $|f|^2$), 6-10× FFT cost, 정확히 $O(n^2 \log n)$. Inverse (adjoint) algorithm: 위 단계 역순 + summation.

**Parameter**: $L_{1,j} \approx 2 \cdot 2^j$, $L_{2,j} \approx 2 \cdot 2^{j/2}$ → 영상 $n \times n$에 대해 약 $\sum_j 2 \cdot 2^j \cdot 2 \cdot 2^{j/2} \cdot \text{angles}_j \approx n^2$ coefficients (low redundancy).

#### English — Wrapping FDCT
1. 2-D FFT of input image.
2. For each (scale, angle), multiply by the Cartesian wedge window, wrap the result into an axis-aligned rectangle, inverse-FFT.
3. $O(n^2 \log n)$ total, ~6-10× FFT, numerically tight (Parseval to within machine precision).

---

### Part IV: §7-9 Refinements and Conclusions / 정제·결론

#### 한국어
- **§7 Refinements**: 
  - Real-valued curvelet output via complex conjugation pairing
  - Special "fine scale" curvelets (cosine-based wavelet at finest level for energy preservation)
  - 3D FDCT (analogous construction in cubes)
- **§8 Numerical experiments**: 
  - Wrapping FDCT 정밀도: residual $\sim 10^{-13}$ with double precision (truly numerical isometry)
  - 영상 $512 \times 512$ wrapping FDCT: ~0.5 sec (Matlab) — FFT의 ~7×
  - Inverse: 거의 같은 시간
- **§9 Connections**: 
  - First-generation curvelets (Candès-Donoho 2000)에 비해 *훨씬 단순*하고 *완전 투명한 수학적 구조*
  - Contourlet (Do-Vetterli 2005, paper #5)과 사촌: 같은 wish list, 다른 구현 (filter bank vs FFT-domain wedge)
  - Ridgelet은 기본 buildig block로는 더 이상 쓰이지 않음

#### English
The wrapping FDCT achieves numerical isometry to machine precision ($\sim 10^{-13}$), runs at ~7× the cost of an FFT, and is invertible. It supersedes first-generation curvelets which required ridgelet preprocessing on phase-space blocks.

---

## 3. Key Takeaways / 핵심 시사점

1. **두 번째 세대 컬블릿은 첫 번째보다 압도적으로 단순 / Second-gen curvelets are radically simpler** — 1차 세대는 ridgelet on phase-space blocks 등 복잡한 전처리. 2차 세대는 FFT 한 번 + wedge 곱 + wrap + inverse FFT만. 수학적으로도 closed-form polar wedges로 깔끔.
   The second-generation construction collapses to four operations (FFT → multiply by wedge → wrap → inverse FFT), replacing the cumbersome ridgelet-based first-generation approach.

2. **Wrapping이 USFFT보다 실용적 / Wrapping is the practical choice** — 두 FDCT 모두 $O(n^2 \log n)$지만 wrapping은 (i) 수학적으로 정확히 tight frame, (ii) 구현 간단, (iii) ~7× FFT cost. USFFT는 연속이론에 더 가깝지만 비균등 보간이 추가 cost. 모든 실용 응용은 wrapping 사용.
   Wrapping FDCT is preferred: tight frame, simple, 7× FFT cost. USFFT is theoretically closer to the continuous transform but slower.

3. **Cartesian coronization은 polar의 격자 친화 버전 / Cartesian coronization adapts polar to grids** — Concentric circles → concentric squares, rotation → shear. 격자에 자연스러우면서 거의 같은 frequency tiling. Contourlet의 LP+DFB와는 다른 접근 (Fourier-domain windowing vs spatial filter banks).
   Cartesian coronization (squares + shears) gives a grid-natural alternative to polar wedges, while preserving the parabolic-scaling tiling.

4. **Optimal NLA rate $(\log m)^3 m^{-2}$ / Optimal nonlinear approximation rate** — Piecewise smooth + $C^2$ singularity curves에서. Wavelet $O(m^{-1})$을 *order of magnitude* 능가. 통계적 함의: noisy data로부터 같은 클래스의 함수 *MSE optimal* 추정 가능.
   Curvelets achieve the information-theoretically optimal NLA rate for objects with $C^2$ singularities, an order of magnitude better than wavelets in MSE terms.

5. **Wave propagator의 near-diagonal 표현 / Near-diagonal representation of wave propagators** — Hyperbolic PDE $E_t$의 curvelet matrix $E_t(n, n') = \langle\varphi_n, E_t\varphi_{n'}\rangle$이 super-polynomial 빠르게 감쇠 (Smith 1998). Curvelet은 near-eigenfunction. 이 성질이 numerical PDE solver의 새로운 길.
   Curvelets are near-eigenfunctions of hyperbolic wave operators; this enables sparse representations of wave propagators and new fast PDE solvers.

6. **Ill-posed inverse problem의 dual decomposition / Dual decomposition of ill-posed inverse problems** — $f = \sum_{n \in \text{Good}} \langle f, \varphi_n\rangle \varphi_n + \sum_{n \in \text{Bad}}$. Curvelet의 microlocal localisation이 어떤 부분이 데이터에서 *복원 가능*한지 명시적으로 분리. CT 같은 missing-data 문제에 유용.
   Curvelet's microlocal localisation cleanly separates the recoverable from the unrecoverable parts of an ill-posed inverse problem.

7. **Vanishing moments는 무한대 / Curvelets have infinitely many vanishing moments** — Compact support away from frequency origin → 모든 polynomial $x_1^n$에 직교. Wavelet의 finite $M$ vanishing moments보다 강함 → polynomial 영역에서 *완벽히* sparse.
   Curvelets are compactly supported away from the frequency origin, giving them infinitely many vanishing moments — perfect orthogonality to polynomial regions.

8. **CurveLab은 사실상 표준 / CurveLab is the de facto standard** — Matlab + C++. 모든 후속 curvelet-based 연구의 baseline. Python으로는 `curvelops` (PyLops 일부)가 wrap. 실용적으로는 contourlet/shearlet 라이브러리도 있지만 CurveLab만큼 성숙하지 않음.
   CurveLab is the standard reference implementation; Python users access it via `curvelops`. Other directional transforms (contourlet, shearlet) lack equally mature open-source software.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Continuous curvelet
Radial $W$ and angular $V$ windows satisfying admissibility (Eq. 2.1, 2.2). Frequency window:
$$
U_j(r, \theta) = 2^{-3j/4} W(2^{-j} r) V\left(\frac{2^{\lfloor j/2\rfloor}\theta}{2\pi}\right)
$$
Mother curvelet $\hat\varphi_j(\omega) = U_j(\omega)$. Curvelet at scale $j$, angle $\theta_\ell$, location $x^{(j,\ell)}_k$:
$$
\varphi_{j,\ell,k}(x) = \varphi_j(R_{\theta_\ell}(x - x^{(j,\ell)}_k))
$$
Coefficient: $c(j,\ell,k) = \langle f, \varphi_{j,\ell,k}\rangle$.

### 4.2 Tight frame and Parseval
$$
f = \sum_{j,\ell,k} \langle f, \varphi_{j,\ell,k}\rangle \varphi_{j,\ell,k}, \quad \sum |c(j,\ell,k)|^2 = \|f\|^2_{L^2}
$$
### 4.3 Parabolic scaling
$$
\text{length} \approx 2^{-j/2}, \quad \text{width} \approx 2^{-j} \quad \Rightarrow \quad \text{width} \approx \text{length}^2
$$
### 4.4 Wrapping FDCT algorithm
**Forward**:
```
F[ω1, ω2] = FFT2D(f)
for j in scales:
    for ℓ in angles_j:
        # Multiply by Cartesian wedge window (Eq. 3.3)
        T = F * Ũ_{j,ℓ}
        # Wrap T onto axis-aligned rectangle of size L1_j × L2_j
        T_wrapped = wrap_into_rectangle(T, L1_j, L2_j)
        c[j, ℓ] = IFFT2D(T_wrapped)
return c, plus coarse scale
```
**Inverse** (adjoint):
```
F = 0
for j, ℓ:
    T_wrapped = FFT2D(c[j, ℓ])
    T = unwrap(T_wrapped)
    F += T * conjugate(Ũ_{j,ℓ})
f = IFFT2D(F)
```

### 4.5 NLA rate (continuous theory)
$$
\|f - f_m\|^2_{L^2} \le C \cdot (\log m)^3 \cdot m^{-2}
$$
for $f$ piecewise smooth + $C^2$-curve singularities. This rate is *optimal*.

### 4.6 Worked numerical example / 수치 예시
$n \times n = 512 \times 512$ image:
- 5 levels of curvelet (j = 1,...,5)
- Angles per level: 16, 32, 32, 64, 64 (paper default), total ~200 directional subbands
- Wrapping FDCT cost: $O(n^2 \log n) \approx 512^2 \cdot \log_2 512 \approx 2.4 \times 10^6$ flops × 7 = $1.7 \times 10^7$ flops
- ~0.5 sec on 2006 CPU; today < 0.1 sec
- Memory: ~2-3× input image (low redundancy)

### 4.7 Tight-frame numerical verification
Wrapping FDCT gives Parseval residual:
$$
\left|\sum |c(j,\ell,k)|^2 - \|f\|^2\right| / \|f\|^2 \approx 10^{-13}
$$
in double precision — truly numerical isometry.

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1909 ─── Haar wavelet
1989 ─── Mallat — wavelet MRA
1994 ─── Donoho-Johnstone — VisuShrink (paper #1)
1996 ─── Candès — Ridgelet transform (1-D singularities along lines)
2000 ─── Candès-Donoho — First-generation curvelets
                          ↳ block ridgelets on phase-space blocks (cumbersome)
2002 ─── Donoho-Huo — Beamlets (related construction)
2004 ─── Candès-Donoho — Second-generation curvelets (continuous)
                          ↳ direct frequency-domain construction
2005 ─── Do-Vetterli — Contourlet (paper #5)
                          ↳ filter-bank approach to similar problem
2006 ★★ CANDÈS-DEMANET-DONOHO-YING — Fast Discrete Curvelet (THIS PAPER)
                          ↳ FDCT via USFFT + FDCT via wrapping
                          ↳ CurveLab software released
2008 ─── Easley-Labate-Lim — Discrete Shearlet (paper #8)
                          ↳ shears as group action (alternative to rotation)
2010+ ── Curvelet/shearlet largely superseded for image denoising by:
                          ↳ BM3D (paper #7) — nonlocal + transform shrinkage
                          ↳ Deep learning (DnCNN, transformer denoisers)
                  But curvelets remain used in:
                          ↳ seismic data (anisotropic structure)
                          ↳ medical CT reconstruction (microlocal analysis)
                          ↳ wave-equation solvers (near-eigenfunction)
```

이 논문은 curvelet 이론을 *실용적으로 사용 가능한 도구*로 만들었다 — 수학적 우아함과 알고리즘적 접근성의 교차점.

This paper turned curvelets from a theoretical construct into a practical tool — bridging mathematical elegance and algorithmic accessibility.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **Candès-Donoho (2000)** | First-gen curvelets via block ridgelets | This paper supersedes that construction with a far simpler frequency-domain approach. |
| **Candès-Donoho (2004)** *Comm. Pure Appl. Math.* | Second-gen continuous curvelets | Provides the continuous-domain definition that this paper digitises. |
| **Do-Vetterli (2005)** *IEEE TIP* (paper #5) | Contourlet | Sister approach: filter banks vs. Fourier-wedge windows. Both achieve $(\log m)^3 m^{-2}$ NLA. |
| **Easley-Labate-Lim (2008)** *ACHA* (paper #8) | Discrete shearlet | Third sister: uses shears as a *group action* (with affine-invariance benefits). |
| **Donoho-Johnstone (1994)** *Biometrika* (paper #1) | Wavelet thresholding | Curvelet thresholding follows the same framework, applied to curvelet coefficients. |
| **Smith (1998)** | Wave equations and FIO | Provides the microlocal analysis underlying curvelets' near-eigenfunction property for hyperbolic operators. |
| **Mallat (1999)** *Wavelet Tour* | Standard reference | Provides the wavelet baseline against which curvelets compare. |
| **Dabov+ (2007)** *IEEE TIP* (paper #7) | BM3D | Different paradigm (nonlocal grouping); could in principle use curvelets in the 3-D transform step. |
| **Starck-Murtagh-Fadili** *Sparse Image and Signal Processing* | Textbook | Comprehensive treatment of curvelets, contourlets, and other directional transforms. |

---

## 7. References / 참고문헌

- Candès, E. J., "Ridgelets: theory and applications", PhD thesis, Stanford (1998).
- Candès, E. J., & Donoho, D. L., "Curvelets — a surprisingly effective nonadaptive representation for objects with edges", in *Curve and Surface Fitting* (1999).
- Candès, E. J., & Donoho, D. L., "New tight frames of curvelets and optimal representations of objects with piecewise $C^2$ singularities", *Comm. Pure Appl. Math.*, 57, 219–266 (2004).
- Candès, E. J., Demanet, L., Donoho, D. L., & Ying, L., "Fast discrete curvelet transforms", *Multiscale Modeling & Simulation*, 5(3), 861–899 (2006). [DOI: 10.1137/05064182X]
- Do, M. N., & Vetterli, M., "The contourlet transform: an efficient directional multiresolution image representation", *IEEE TIP*, 14(12), 2091–2106 (2005).
- Donoho, D. L., & Huo, X., "Beamlet pyramids: a new form of multiresolution analysis", *Proc. SPIE 4119*, 434–444 (2000).
- Mallat, S., *A Wavelet Tour of Signal Processing*, 2nd ed., Academic Press (1999).
- Smith, H., "A Hardy space for Fourier integral operators", *J. Geom. Anal.*, 8(4), 629–653 (1998).
- Starck, J.-L., Murtagh, F., & Fadili, J. M., *Sparse Image and Signal Processing*, Cambridge University Press (2010).
- CurveLab: https://www.curvelet.org
