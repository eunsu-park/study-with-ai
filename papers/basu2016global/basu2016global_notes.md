---
title: "Global Seismology of the Sun"
authors: [Sarbani Basu]
year: 2016
journal: "Living Reviews in Solar Physics"
doi: "10.1007/s41116-016-0003-4"
topic: Living_Reviews_in_Solar_Physics
tags: [helioseismology, solar-interior, p-modes, inversions, rotation, tachocline, standard-solar-model, asteroseismology]
status: completed
date_started: 2026-04-23
date_completed: 2026-04-23
---

# 49. Global Seismology of the Sun / 태양의 전역 지진학

---

## 1. Core Contribution / 핵심 기여

Sarbani Basu (2016) provides a comprehensive review of **global helioseismology** — the study of the Sun's resonant normal modes (acoustic p-modes, surface f-modes, and buoyancy g-modes) and what their frequencies, splittings, and line-widths reveal about solar interior structure and dynamics. The paper traces the full logical chain from (i) the hydrostatic stellar-structure equations and adiabatic oscillation equations, through (ii) the quantisation conditions that produce discrete ω_{n,ℓ} modes and Duvall's asymptotic law, to (iii) the inverse problem in which millions of observed frequencies from BiSON, GONG, SoHO/MDI, and SDO/HMI are turned into precise profiles of sound speed c²(r), density ρ(r), the convection-zone base r_b = 0.713 ± 0.001 R_☉, the surface helium abundance Y_s = 0.2485 ± 0.0035, and the two-dimensional differential rotation Ω(r,θ). Results include the discovery of the **tachocline** (a thin shear layer at ~0.71 R_☉), confirmation that the radiative interior rotates as a solid body (~430 nHz), and structural agreement with Standard Solar Models at the <0.5% level in c² between 0.2 and 0.9 R_☉.

Basu (2016)는 **전역 태양지진학**에 대한 종합 리뷰로서, 태양의 공명 진동 모드(음향 p-mode, 표면 f-mode, 부력 g-mode)의 주파수·분열·선폭이 태양 내부 구조와 동역학에 대해 무엇을 알려주는지를 다룬다. 논문은 (i) 정역학 평형과 단열 진동 방정식으로부터 시작해 (ii) 이산 고유 진동수 ω_{n,ℓ}를 낳는 양자화 조건과 Duvall 점근 법칙을 거쳐 (iii) BiSON, GONG, SoHO/MDI, SDO/HMI에서 관측된 수백만 주파수를 역전(inversion)하여 음속 c²(r), 밀도 ρ(r), 대류대 바닥 r_b = 0.713 ± 0.001 R_☉, 표면 He 함량 Y_s = 0.2485 ± 0.0035, 2차원 차등회전 Ω(r,θ)를 구하는 전체 논리 체계를 제시한다. 주요 결과로 **타코클린**(≈0.71 R_☉의 얇은 전단층) 발견, 복사 내부의 고체체 회전(≈430 nHz), 그리고 0.2 ≤ r/R_☉ ≤ 0.9 구간에서 표준 태양 모델과의 c² 일치도 <0.5% 등을 포함한다.

---

## 2. Reading Notes / 읽기 노트

### Part I: Stellar Structure & Oscillation Equations / 별 구조와 진동 방정식

The review starts (Sect. 2) with the classical stellar-structure equations in mass variable m: continuity dr/dm = 1/(4πr²ρ), hydrostatic equilibrium dP/dm = −Gm/(4πr⁴), energy ds d(L)/dm = ε − ε_ν − C_P dT/dt + (δ/ρ) dP/dt, and temperature gradient dT/dm = −GmT/(4πr⁴P) ∇, closed by equation of state, opacity κ, nuclear rates ε, and diffusion (11 eqs for microphysics). A **Standard Solar Model (SSM)** is a calibrated evolutionary sequence that matches present L_☉, R_☉, Z/X by tuning (α, Y_0) — and (sometimes) Z_0 — with "standard" microphysics (OPAL or OP opacity, OPAL EoS, Grevesse+Sauval or Asplund abundances).

리뷰는 §2에서 질량변수 m에 대한 고전적 별 구조 방정식으로 시작한다 — 연속방정식, 정역학평형, 에너지 보존, 온도기울기 — 그리고 상태방정식, 불투명도 κ, 핵반응률 ε, 확산 항이 마이크로 피직스 입력으로 들어간다. **표준 태양 모델(SSM)**은 현재의 L_☉, R_☉, Z/X를 맞추기 위해 (α, Y_0)을 보정한 진화 계산이며 표준 마이크로 피직스(OPAL/OP 불투명도, OPAL EoS, 태양 원소 함량)를 사용한다.

Section 3 derives the **linear adiabatic oscillation equations** by perturbing the fluid-dynamical equations (continuity, momentum, Poisson, energy) to first order:

$$\rho_1 + \nabla\cdot(\rho_0 \boldsymbol\xi) = 0, \qquad \rho \ddot{\boldsymbol\xi} = -\nabla P_1 + \rho_0 \nabla\Phi_1 + \rho_1\nabla\Phi_0, \qquad \nabla^2\Phi_1 = 4\pi G \rho_1,$$

together with the adiabatic closure P_1 + ξ·∇P = c²(ρ_1 + ξ·∇ρ). Decomposing into spherical harmonics ξ_r = ξ_r(r) Y_ℓ^m exp(−iωt) yields a 4th-order ODE system (eqs. 35, 37, 39) with eigenfunctions labelled by (n, ℓ, m) and frequencies ω_{n,ℓ}. In the **Cowling approximation** (neglecting Φ_1) and neglecting H_p⁻¹ terms far from the surface, a compact 2nd-order form emerges:

$$\boxed{\ \frac{d^2\xi_r}{dr^2} = \frac{\omega^2}{c^2}\left(1 - \frac{N^2}{\omega^2}\right)\left(\frac{S_\ell^2}{\omega^2} - 1\right)\xi_r\ }$$

where the **Brunt–Väisälä (buoyancy) frequency** is

$$N^2 = g\left(\frac{1}{\Gamma_1 P}\frac{dP}{dr} - \frac{1}{\rho}\frac{d\rho}{dr}\right),$$

and the **Lamb frequency** is S_ℓ² = ℓ(ℓ+1) c²/r². Oscillatory solutions exist only when both factors have the same sign — giving **p-modes** (ω² > max(N², S_ℓ²), pressure-restored, outer envelope) and **g-modes** (ω² < min(N², S_ℓ²), buoyancy-restored, radiative core). f-modes are surface-gravity modes with dispersion ω² ≃ gk.

§3는 유체역학 방정식(연속성·운동량·Poisson·에너지)을 1차 섭동하여 **선형 단열 진동 방정식**을 유도한다. 구면조화함수 분리로 (n, ℓ, m) 표기 고유값 문제를 얻고, Cowling 근사와 표면 근방이 아닌 경우 H_p⁻¹ 항 무시하면 2차 미분방정식으로 축약된다. Brunt–Väisälä 진동수 N과 Lamb 진동수 S_ℓ이 전파 가능 영역을 정의한다: p-mode는 외층에 갇히고, g-mode는 복사 내부에 갇힌다.

### Part II: Properties of Modes – p, g, f / 모드의 성질

**P-modes** (Sect. 3.3.1): trapped between a lower turning point r_t satisfying c²(r_t)/r_t² = ω²/[ℓ(ℓ+1)] and the surface. For high-n they satisfy the quantisation condition

$$\int_{r_t}^{R}\left(\omega^2 - S_\ell^2\right)^{1/2}\frac{dr}{c} = (n + \alpha_p)\pi,$$

which, after a change of variable w = ω/L (L = ℓ+½), becomes **Duvall's Law**:

$$F(w) = \int_{r_t}^{R}\left(1 - \frac{L^2 c^2}{\omega^2 r^2}\right)^{1/2}\frac{dr}{c} = \frac{(n+\alpha_p)\pi}{\omega}.$$

Plotting (n+α_p)π/ω against w collapses all observed modes onto a single curve (Fig. 2 of the paper). Tassoul's higher-order asymptotic expansion gives ν_{n,ℓ} ≃ (n + ℓ/2 + α_p) Δν, i.e., p-modes are approximately equally spaced in frequency with **large separation** Δν = (2∫₀^R dr/c)⁻¹; the **small separation** δν_{n,ℓ} ≡ ν_{n,ℓ} − ν_{n−1,ℓ+2} ≃ −(4ℓ+6)Δν/(4π² ν_{n,ℓ}) ∫₀^R (dc/dr)(dr/r) is a sensitive diagnostic of the **core sound-speed gradient** (and hence evolutionary state / age).

**P-mode**는 하부 전환점 r_t와 표면 사이에 갇히며, 고차 고유수 n에 대해 정상파 조건으로 **Duvall 법칙**을 만족한다. 관측된 모든 모드가 w = ω/L을 변수로 하는 하나의 곡선으로 축소된다. Tassoul 점근 전개는 주파수가 대분리 Δν 만큼 거의 등간격임을 보이며, 소분리 δν_{n,ℓ}는 중심부 음속 구배(=별의 나이)의 지표이다.

**G-modes** (Sect. 3.3.2): trapped between N-turning points, equally spaced in *period* P = 2π/ω. In the Sun, g-modes are evanescent in the convection zone and have not been unambiguously detected despite claims by García et al. (2007, 2010).

**F-modes** (n=0): surface-gravity modes, ω² ≃ gk, used for precise radius determination (Antia 1998 r_☉ estimate).

**Propagation diagram** (Fig. 1 in the paper): plotting N² (blue) and S_ℓ² (red, for various ℓ) against r/R_☉ shows: N² is negative in the convection zone (> 0.71 R_☉) and small in the core, peaking at ~500 µHz around 0.1 R_☉; S_ℓ² rises rapidly with ℓ and diverges at r → 0. A 1000 µHz p-mode of ℓ = 5 is trapped roughly between 0.1 R_☉ and R_☉; a 200 µHz g-mode is confined to the radiative interior.

전파 다이어그램은 N²와 S_ℓ²의 r-의존성을 보여준다. 대류대에서 N² < 0이므로 g-mode는 소멸하고, 핵 근처에서 S_ℓ²가 급증하므로 p-mode는 충분히 깊이 들어가지 못한다.

### Part III: History of Solar Models & The Surface Term / 태양 모델의 역사와 표면항

Sect. 4 reviews solar-model construction: Schwarzschild's 1946 model (pre-pp), the 1970s "standard solar model" focused on neutrino flux (Bahcall–Sears), the 1980s inclusion of diffusion (Cox et al. 1989, Christensen-Dalsgaard et al. 1993), and modern Model S (Christensen-Dalsgaard et al. 1996). Sect. 5 addresses the **"surface term"**: when one compares mode frequencies of a good SSM with observations, the residuals δν(Sun − Model) increase systematically with frequency above ~2000 µHz (Fig. 4 of paper; up to ~15 µHz at 4 mHz). This is caused by inadequate modelling of the near-surface super-adiabatic layer where convection breaks 1D MLT and non-adiabatic effects matter. The residuals can be factored as δν/ν = (structure-dependent) + F_surf(ω)/E_i, where E_i is mode inertia (Eq. 82), and removing Q_{nℓ} = E_{nℓ}(ν)/E_{ℓ=0}(ν) scaling collapses curves at different ℓ onto a common function of ω only.

§4는 태양 모델 역사(Schwarzschild 1946 → Bahcall–Sears 1972 → Model S 1996)를, §5는 **표면항 문제**를 다룬다: 관측과 SSM 주파수 차이가 ~2 mHz 이상에서 주파수에 체계적으로 증가하며, 이는 표층의 super-adiabatic 영역을 1D MLT로 모사할 때 생기는 한계 때문이다. 모드 관성 E_i로 스케일하면 ℓ-의존성이 사라져 F_surf(ω) 하나의 함수로 표현된다.

### Part IV: Inversions / 역전

**Asymptotic inversions** (Sect. 6.1): From Duvall's law, Gough (1984) derived the Abel-type integral

$$w^3 \frac{dF}{dw} = \int_w^{a_s}\left(1 - \frac{a^2}{w^2}\right)^{-1/2}\frac{d\ln r}{d\ln a}da, \quad a \equiv c(r)/r,$$

which can be inverted analytically to give r(a) and hence c(r) (Eq. 86). This gave the first sound-speed profile of the Sun.

**Full inversions** (Sect. 6.2): Start from the perturbed momentum equation ρ \ddot{ξ} = −∇P_1 + ρ_0 g_1 + ρ_1 g (Eq. 93). Using Chandrasekhar's variational principle for the Hermitian eigenvalue problem, linearise around a reference model to get the fundamental **inversion equation**:

$$\boxed{\ \frac{\delta\omega_i}{\omega_i} = \int_0^R K^i_{c^2,\rho}(r)\,\frac{\delta c^2}{c^2}(r)\,dr + \int_0^R K^i_{\rho,c^2}(r)\,\frac{\delta\rho}{\rho}(r)\,dr + \frac{F_{\rm surf}(\omega_i)}{E_i}\ }$$

where K^i_{c²,ρ}, K^i_{ρ,c²} are **sensitivity kernels** computed from the reference model and depend on the eigenfunctions ξ_{r,nℓ}, ξ_{t,nℓ} (Fig. 10, 11 of the paper). Kernels for different variable pairs (u≡P/ρ, Y) are obtained via the chain rule with the equation of state: δΓ_1/Γ_1 = Γ_{1,P} δP/P + Γ_{1,ρ} δρ/ρ + Γ_{1,Y} δY.

Two main inversion techniques are described in detail:

- **Regularised Least Squares (RLS)** (§6.3.1): expand δc²/c², δρ/ρ as cubic B-splines in knots spaced in acoustic depth τ. Solve matrix A**x** = **d** with Tikhonov regularisation χ²_reg = χ² + α² ∫(d²q/dr²)² dr. Smoothing parameter α chosen from L-curve balancing χ² vs ||L||². Solves via SVD.

- **Optimally Localised Averages (OLA / SOLA)** (§6.3.2): find coefficients c_i(r_0) such that the **averaging kernel** K(r_0,r) = Σ c_i K^i_{c²,ρ}(r) is well-localised around r_0 while cross-term kernel C(r_0,r) = Σ c_i K^i_{ρ,c²}(r) and error σ(r_0)² = Σ c_i² σ_i² remain small. Backus–Gilbert 1968 formalism; Pijpers & Thompson 1992 SOLA uses a Gaussian target T(r_0,r) = A exp(−[(r−r_0)/Δ(r_0)]²). The reliable inversions emerge when RLS and OLA agree (Sekii 1997).

**역전**: 관측된 주파수 차이 δω_i/ω_i를 δc²/c², δρ/ρ 프로파일로 역산한다. 두 주요 기법은 RLS(cubic B-spline 기반, χ² + 스무딩 페널티 최소화)와 OLA/SOLA(Backus–Gilbert, 평균화 커널을 r_0 주변에 국소화하여 Σ c_i δω_i/ω_i = 평균 δc²/c²)이다. 두 기법이 일치하는 영역만 신뢰 가능하다.

### Part V: Structure Results / 구조 결과

**Sect. 7.1 Sound speed & the neutrino problem**: Modern inversions (Fig. 16) show |δc²/c²| < 0.5% for 0.2 < r/R_☉ < 0.9 and |δρ/ρ| < 2%. The extremely small structural disagreement with SSMs was a major clue that the **solar neutrino problem** — the factor-of-3 deficit in Homestake Cl detector (Davis 1964–1994), confirmed by Kamiokande/Gallex/SAGE — lay in particle physics, not solar modelling. Bahcall et al. 1998 showed a non-standard solar model matching the Cl detector would have δc²/c² ≈ 10%, far exceeding helioseismic constraints. The **Sudbury Neutrino Observatory (SNO, 2002)** closed the problem by directly measuring neutrino flavour oscillation.

§7.1: 표준 태양 모델과의 음속 차이는 중심부 부근 <0.5% 수준이며, 이는 **태양 중성미자 문제**가 태양 모델의 결함이 아니라 중성미자 진동(particle physics)임을 의미했고, SNO(2002)가 이를 확정했다.

**Sect. 7.2 Convection-zone base & tachocline**: The temperature gradient jumps from adiabatic (interior) to radiative at r_b, creating a sharp feature in dc²/dr. Christensen-Dalsgaard et al. (1991) and Basu & Antia (1997) used the dimensionless W(r) ≡ (r²/Gm)(dc²/dr) and found

$$\boxed{\ r_b = 0.713 \pm 0.001\, R_\odot\ }$$

independent of metallicity. Overshoot below r_b is constrained to <0.05 H_p (Monteiro et al. 1994; Basu 1997) from the absence of an extended acoustic glitch signature (Fig. 18 of the paper). Helium abundance in the convection zone is derived from the Γ_1 dip in the He II ionisation zone:

$$Y_s = 0.2485 \pm 0.0035 \text{ (using OPAL EoS, Basu & Antia 2004)},$$

and the initial helium from SSM calibration:

$$Y_0 = 0.273 \pm 0.006 \text{ (Serenelli & Basu 2010)}.$$

§7.2: 대류대 바닥은 r_b = 0.713 ± 0.001 R_☉에 위치하며, **타코클린** 바로 아래의 음향 글리치 신호로부터 overshoot는 <0.05 H_p로 제한된다. 표면 He 함량 Y_s = 0.2485 ± 0.0035, 초기 Y_0 = 0.273 ± 0.006.

**Sect. 7.3–7.5 Testing physics & the abundance issue**: Inversions test EoS (OPAL best for interior, MHD better at surface), opacity (seismology requires 6–26% increase over Asplund 2005/2009 abundances), and diffusion (Fig. 20 clearly shows diffusion is required). The **solar abundance problem**: spectroscopic revisions (Asplund et al. 2005, 2009 → Z/X ≈ 0.018) lowered CNO abundances by ~30%, producing solar models with wrong r_b, Y_s, and δc²/c² (Table 3 of paper). Resolution requires either opacity increase, extra mixing below CZ, or revised abundance scale (Caffau+ 2011 Z/X = 0.0209 helps).

§7.3–7.5: 역전 결과는 EoS, 불투명도, 확산을 검증한다. **태양 원소 함량 문제**: 최신 spectroscopy (Asplund 2005/2009)가 Z/X ≈ 0.018로 낮추면 SSM이 r_b, Y_s, δc²/c² 모두에서 관측과 어긋난다. Caffau et al. 2011 (Z/X = 0.0209)이 부분적으로 완화한다.

### Part VI: Rotation & Asphericity / 회전과 비구형성

**Sect. 8 Departures from spherical symmetry**: Rotation is the dominant asphericity. The full frequency is

$$\frac{\omega_{n\ell m}}{2\pi} = \nu_{n\ell 0} + \sum_{j=1}^{j_\max} a_j(n,\ell)\,\mathcal{P}_j^{n\ell}(m),$$

with odd-j splittings encoding Ω(r,θ) (advection, Coriolis) and even-j encoding magnetic fields / structural asphericity / second-order centrifugal effects. For uniform rotation with angular velocity Ω_s, a simple first-order result is:

$$\boxed{\ \delta\omega_m = m\,\Omega_s\quad\text{i.e.}\quad \Omega_s \approx \frac{m\,\omega_0}{2\pi}\text{ splitting per m}\ }$$

for the idealised case; for differential rotation Ω(r,θ), full 2-D inversion of the splitting kernels is needed.

**Solar rotation profile** (Fig. 26 of the paper; Howe et al. 2005):

- **Convection zone (r > 0.71 R_☉)**: differential, Ω/2π ≈ 460 nHz at equator, ≈ 330 nHz at pole (25 vs 35 day periods).
- **Radiative interior (r < 0.71 R_☉)**: solid-body at ≈ 430 nHz.
- **Tachocline** at 0.71 R_☉: thin shear layer, thickness <0.05 R_☉.
- **Near-surface shear layer**: Ω decreases with radius in outer 5% (~5% change).

Integral rotation quantities:

$$H = (190.0 \pm 1.5)\times 10^{39}\,\text{kg m}^2\,\text{s}^{-1}, \quad T = (253.4 \pm 7.2)\times 10^{33}\,\text{kg m}^2\,\text{s}^{-2}, \quad J_2 = (2.18 \pm 0.06)\times 10^{-7}.$$

**Sect. 8.2 Magnetic fields & asphericity**: Even-a coefficients constrain sub-surface toroidal fields; Antia et al. (2000b) find an upper limit of 0.3 MG at the CZ base, Baldner et al. (2009) find a near-surface toroidal field of ~380 G at 0.999 R_☉ and ~1.4 kG at 0.996 R_☉. Acoustic asphericity is ~10⁻⁴, peaking at ~0.92 R_☉, consistent with equator being cooler than mid-latitudes.

§8: 회전이 주된 비구형성이다. 분열계수 a_j에서 홀수 차수는 Ω(r,θ), 짝수 차수는 자기장·구조 비대칭을 담는다. 태양 회전 프로파일(Howe+ 2005, Fig. 26): 대류대는 차등회전(적도 460 nHz, 극 330 nHz), 복사 내부는 고체체 ≈430 nHz, 이들 사이 **타코클린**은 ≈0.71 R_☉의 얇은 전단층이다.

### Part VII: Solar Cycle & Asteroseismology / 태양 주기와 항성지진학

**Sect. 9 Solar-cycle-related changes**: Oscillation frequencies increase with magnetic activity — at solar maximum, p-mode frequencies are ~0.4 µHz higher at 3 mHz than at minimum (Fig. 27 of the paper). The shift scales as Q_{nℓ} δν ∝ ν², resembling a surface term, implying near-surface magnetic perturbations. Mode line-widths (damping) and powers also change; the "peculiar" solar cycle 24 (weaker than 23) saw smaller frequency shifts and reduced g-mode–proxy near-surface B field.

**Sect. 11 Asteroseismology**: For unresolved stars, only low-ℓ modes are observable. Two basic asteroseismic observables are Δν (Eq. 170) and ν_max (Eq. 171):

$$\frac{\Delta\nu}{\Delta\nu_\odot} \simeq \sqrt{\frac{M/M_\odot}{(R/R_\odot)^3}}, \quad \frac{\nu_{\max}}{\nu_{\max,\odot}} \simeq \frac{M/M_\odot}{(R/R_\odot)^2 \sqrt{T_{\rm eff}/T_{\rm eff,\odot}}}.$$

These scaling relations yield stellar mass and radius to ~5% for Sun-like stars (Kepler/CoRoT era). Red giants exhibit **mixed modes** (simultaneously p-like envelope and g-like core behaviour) that probe core rotation; Beck et al. (2012) found red-giant cores rotate ~5–10× faster than envelopes, challenging angular-momentum-transport theories.

§9: 태양 주기 의존 변화는 태양 최대시 주파수 증가(~0.4 µHz @ 3 mHz)로 나타나며, 표면항과 유사한 주파수 스케일링에서 표층 자기 섭동이 원인임을 보인다.
§11: 항성지진학은 저차 ℓ 모드만 관측 가능. Δν와 ν_max 스케일링은 질량·반지름을 ~5% 정밀도로 준다. 적색 거성의 mixed mode는 내부 회전을 탐색한다.

### Part VIII: Mode Excitation, Line-Widths, and Observational Networks / 모드 여기, 선폭, 관측 네트워크

**Mode excitation** (Sect. 10): solar p-modes are **stochastically excited** by turbulent convection in the sub-photospheric layers. Goldreich & Keeley (1977), Balmforth (1992), and more recent 3-D simulations (Stein & Nordlund 2001; Samadi et al. 2003, 2007) compute the energy injection rate from Reynolds stresses and entropy fluctuations acting as driving sources. The resulting mode amplitude follows a Lorentzian power spectrum:

$$P(\omega) = \frac{A_0}{(\omega - \omega_0)^2 + \gamma^2/4},$$

where γ is the mode line-width inversely proportional to mode lifetime (1/γ ~ a few days for ~3 mHz modes). Line-widths grow dramatically above ν_max ≈ 3.1 mHz as modes approach the acoustic cutoff and damping overtakes excitation. Amplitudes peak near ν_max with Gaussian envelope.

**모드 여기**: 태양 p-mode는 광구 아래 난류 대류에 의해 **확률적으로 여기**된다. 전력 스펙트럼은 각 모드 주변에 Lorentzian 선형을 가지며, 선폭 γ는 모드 수명의 역수이다. 진폭은 ν_max ≈ 3.1 mHz에서 극대.

**Observational networks** (Sect. 1): For uninterrupted observations required to resolve narrow mode peaks one needs either a ground-based network or a space mission:

- **BiSON** (Birmingham Solar-Oscillations Network, 1976+): 6 stations worldwide, Sun-as-a-star low-ℓ (ℓ ≤ 3) Doppler measurements.
- **GONG** (Global Oscillation Network Group, 1995+): 6 stations, resolved-disc Doppler imagery, ℓ ≤ 200.
- **SoHO/MDI** (1995–2011): space-based, ℓ ≤ 250.
- **SDO/HMI** (2010+): 4096×4096 full-disc Doppler at 45-s cadence, ℓ up to ~1000.

Network data are combined into frequency tables (e.g., BiSON-13 of Basu et al. 2009) that form the basis for inversions.

**관측 네트워크**: 연속 관측을 위해 지상 네트워크(BiSON 저차 모드, GONG 분해 디스크)와 우주 임무(SoHO/MDI, SDO/HMI)가 운영되고, 이들이 결합되어 BiSON-13 등 주파수 테이블로 제공된다.

### Part IX: The Solar Abundance Problem in Detail / 태양 원소 함량 문제 상세

Prior to 2005 solar spectroscopic abundances (Grevesse & Sauval 1998) gave Z/X = 0.0245. These produced SSMs in excellent agreement with helioseismic r_b and Y_s. The revision by Asplund et al. (2005a, 2009) — based on 3-D hydrodynamic photospheric models including non-LTE — reduced CNO abundances by ~30%, giving Z/X = 0.0165 (2005) and Z/X = 0.018 (2009). SSMs built with these low-Z compositions disagree strongly with helioseismology (Table 3 of Basu 2016):

| Model | Z/X | R_CZ (R_☉) | Y_s | Y_0 |
|---|---|---|---|---|
| Helioseismic | – | 0.713 ± 0.001 | 0.2485 ± 0.0035 | 0.273 ± 0.006 |
| GS98 (old) | 0.023 | 0.7139 | 0.2456 | 0.2755 |
| AGS05 (low-Z) | 0.0165 | 0.7259 | 0.2286 | 0.2586 |
| AGSS09 | 0.018 | 0.7205 | 0.2352 | 0.2650 |
| C+11 (Caffau+) | 0.0209 | 0.7150 | 0.2415 | 0.2711 |

Proposed resolutions include opacity increases (6–26% required), extra mixing below CZ, or revised abundance scale (Caffau+ 2011 partially restores agreement). This problem remains a central open issue in solar physics.

§7.5 (확장): Asplund 2005/2009의 저-Z 원소 함량은 관측된 r_b, Y_s, 음속 프로파일과 불일치하며, 해결은 불투명도 증가(≥6%), 추가 혼합, 또는 Caffau 등 개정된 함량에서 부분적으로만 가능하다. 이는 태양 물리학의 미해결 과제이다.

### Part X: Limitations and Future / 한계와 미래

Basu (§12) identifies remaining challenges:
- **Near-surface layers (r > 0.96 R_☉)**: poor reliability of high-ℓ (ℓ > 250) modes limits near-surface probing. HMI high-cadence data promises improvement.
- **Solar core (r < 0.05 R_☉)**: limited by absence of detected g-modes; only low-degree p-modes (which have some core sensitivity via small separations) constrain core structure.
- **Long-term coverage**: continuous helioseismic data exist for only ~30 years (low-ℓ) and ~20 years (intermediate-ℓ), insufficient to characterise cycle-to-cycle variability.
- **Asteroseismology**: needs better surface-term corrections and treatment of stars hotter/cooler than the Sun.

§12: 향후 과제는 표층(>0.96 R_☉) 고차 ℓ 모드 신뢰성, 코어(<0.05 R_☉) 탐색을 위한 g-mode 검출, 장기 관측 확장, 항성지진학 표면항 보정.

---

## 3. Key Takeaways / 핵심 시사점

1. **The Sun oscillates in ~10⁷ normal modes, but observable ones are mostly high-n, low-to-intermediate-ℓ p-modes near 3 mHz** — Solar oscillation amplitudes are ~10 cm/s (velocity) with periods of ~5 minutes; the observable p-modes satisfy ω² > N² and ω² > S_ℓ² and occupy a well-defined ridge pattern in the (ℓ, ν) diagnostic diagram. 태양은 수백만의 공명 모드로 진동하지만, 관측 가능한 것은 대부분 고차 n, 저~중간 ℓ의 ~3 mHz p-mode이며 l-ν 다이어그램에서 명확한 능선 구조를 이룬다.

2. **Duvall's Law collapses all p-modes onto a single curve and enables asymptotic c(r) inversion via Abel transform** — The quantity (n+α_p)π/ω plotted against w = ω/L is a universal function determined by the sound-speed profile; this was the first way the solar interior was probed. Duvall 법칙은 모든 p-mode를 하나의 곡선으로 모으며, Abel 변환으로 c(r) 역전이 가능한 최초의 방법이었다.

3. **Full non-asymptotic inversions use linearised kernels K^i_{c²,ρ}, K^i_{ρ,c²} from a reference SSM** — RLS and OLA/SOLA are complementary techniques; reliable results require both to agree. Solved via cubic B-splines + Tikhonov regularisation (RLS) or Backus–Gilbert localised averages (OLA). 완전 역전은 참조 모델의 선형 커널을 이용하며, RLS와 OLA 두 기법의 일치를 확인해야 신뢰할 수 있다.

4. **The Sun's interior agrees with SSMs to <0.5% in c² and <2% in ρ over 0.2–0.9 R_☉** — This remarkable agreement made the solar neutrino problem's resolution particle-physical (SNO 2002 confirmed neutrino oscillation); a "solar-model" solution would have required δc²/c² ~ 10%. 태양 내부는 SSM과 <0.5% c² 수준에서 일치하며, 이 강한 제약이 중성미자 문제의 해결을 입자물리학(SNO 2002)에서 찾도록 이끌었다.

5. **The convection-zone base lies at 0.713 ± 0.001 R_☉ with overshoot <0.05 H_p** — This is independent of the heavy-element abundance adopted and is a fundamental constraint on any solar model or dynamo theory. 대류대 바닥은 r_b = 0.713 ± 0.001 R_☉로 원소 함량과 독립이며, 모든 태양 모델·다이나모 이론의 근본 제약이다.

6. **Solar rotation is differential in the CZ (~460 nHz equator, ~330 nHz poles) and nearly solid-body below (~430 nHz), connected by the tachocline** — The tachocline at 0.71 R_☉ is a thin (<0.05 R_☉) shear layer fundamental to dynamo theory and angular-momentum transport; its existence was a helioseismic discovery (Kosovichev, Thompson+ mid-1990s). 태양 회전은 대류대에서 차등회전(적도 460 nHz, 극 330 nHz), 그 아래 고체체(≈430 nHz), 둘을 잇는 타코클린은 0.71 R_☉의 얇은 전단층으로 다이나모 이론의 핵심이다.

7. **The surface He abundance Y_s = 0.2485 ± 0.0035 constrains primordial Big Bang nucleosynthesis & stellar evolution** — Derived from the Γ_1 dip in the He II ionisation zone; diffusion and settling give Y_0 − Y_s ≈ 0.025. 표면 헬륨 함량 Y_s = 0.2485 ± 0.0035는 He II 이온화 영역의 Γ_1 함몰로부터 도출되며, Big Bang 핵합성과 별 진화 이론을 제약한다.

8. **Global helioseismology's method generalises to asteroseismology, yielding M, R to ~5% via (Δν, ν_max) scaling** — Kepler, CoRoT, and upcoming TESS/PLATO apply these techniques to thousands of stars. Red-giant mixed modes have revealed that stellar cores rotate much faster than envelopes, challenging angular-momentum-transport theories. 전역 태양지진학 방법론은 항성지진학으로 일반화되어 Kepler/CoRoT이 (Δν, ν_max) 스케일로 M·R을 ~5% 정밀도로 산출하고, TESS/PLATO가 그 뒤를 잇는다.

---

## 4. Mathematical Summary / 수학적 요약

**(A) Stellar structure (hydrostatic equilibrium) / 정역학평형**

$$\frac{dP}{dr} = -\rho g, \qquad g(r) = \frac{G m(r)}{r^2}, \qquad m(r) = \int_0^r 4\pi s^2 \rho(s)\,ds.$$

Radial pressure gradient balances gravity. 압력 구배가 중력을 상쇄.

**(B) Adiabatic sound speed / 단열 음속**

$$c^2 = \frac{\Gamma_1 P}{\rho}, \qquad \Gamma_1 = \left(\frac{\partial \ln P}{\partial \ln \rho}\right)_{\rm ad}.$$

Γ_1 ≈ 5/3 for fully ionised gas, dips in ionisation zones (He II at ~0.98 R_☉). 완전 이온화 가스에서 Γ_1 ≈ 5/3, 이온화 영역에서 함몰.

**(C) Wave equation inside the Sun (2nd-order, high-n, high-ℓ, Cowling) / 태양 내부 파동방정식**

$$\frac{d^2 \xi_r}{dr^2} = K(r)\,\xi_r, \qquad K(r) = \frac{\omega^2}{c^2}\left(1 - \frac{N^2}{\omega^2}\right)\left(\frac{S_\ell^2}{\omega^2} - 1\right).$$

With

$$N^2 = g\left(\frac{1}{\Gamma_1 P}\frac{dP}{dr} - \frac{1}{\rho}\frac{d\rho}{dr}\right), \qquad S_\ell^2 = \frac{\ell(\ell+1)c^2}{r^2}.$$

Propagation (K > 0) only when sign[ω²−N²] = sign[ω²−S_ℓ²]. 전파(K > 0)는 두 부호가 같을 때만 성립.

**(D) Quantisation / Duvall's Law / 양자화**

$$\int_{r_t}^{R}\sqrt{K(r)}\,dr = (n + \alpha_p)\pi \;\Rightarrow\; F(w) \equiv \int_{r_t}^R\!\!\left(1 - \frac{L^2 c^2}{\omega^2 r^2}\right)^{1/2}\frac{dr}{c} = \frac{(n + \alpha_p)\pi}{\omega}.$$

**(E) Tassoul's asymptotic relation & ridge dispersion / Tassoul 점근관계**

$$\nu_{n,\ell} \simeq \left(n + \frac{\ell}{2} + \alpha_p\right)\Delta\nu, \qquad \Delta\nu = \left(2\int_0^R \frac{dr}{c}\right)^{-1}.$$

For the Sun Δν ≈ 135 µHz. p-mode ridges in the l-ν diagram follow

$$\omega_{n,\ell} \approx (n + \tfrac{1}{2}) \cdot \frac{\pi}{\int_0^R dr/c} \cdot \left[1 + \mathcal{O}\!\left(\frac{\ell(\ell+1)}{\omega^2}\right)\right].$$

Small separation:

$$\delta\nu_{n,\ell} = \nu_{n,\ell} - \nu_{n-1,\ell+2} \simeq -(4\ell+6)\frac{\Delta\nu}{4\pi^2 \nu_{n,\ell}}\int_0^R \frac{dc}{dr}\frac{dr}{r}.$$

**(F) Acoustic cut-off & upper turning point / 음향 차단 주파수**

$$\omega_c^2 = \frac{c^2}{4H_\rho^2}\left(1 - 2\frac{dH_\rho}{dr}\right), \qquad H_\rho = -\frac{dr}{d\ln\rho}.$$

For isothermal atmosphere ω_c ≈ c ρ / (2P). At ω > ω_c modes escape as running waves.

**(G) Rotational splitting & kernel / 회전 분열**

For slow differential rotation Ω(r,θ), first-order splitting is

$$\delta\omega_{n\ell m} = 2\pi\, m \int_0^R\!\!\int_0^\pi K_{n\ell m}(r,\theta)\,\Omega(r,\theta)\, r\sin\theta\, d\theta\, dr,$$

with rotation kernel K_{n\ell m} determined by (ξ_r, ξ_t) and Y_ℓ^m. For **uniform rotation**, Ω constant ⇒ δω_m/2π = m Ω_s / (2π), i.e.

$$\boxed{\ \Omega_s \approx \frac{m\,\omega_0}{2\pi}\ }$$

interpreted as the splitting per unit m.

**(H) Structure-inversion integral equation / 구조 역전 방정식**

$$\frac{\delta\omega_i}{\omega_i} = \int_0^R K^i_{c^2,\rho}(r)\,\frac{\delta c^2}{c^2}(r)\,dr + \int_0^R K^i_{\rho,c^2}(r)\,\frac{\delta\rho}{\rho}(r)\,dr + \frac{F_{\rm surf}(\omega_i)}{E_i},$$

with mode inertia

$$E_{n,\ell} = \frac{\int_0^R \rho[\xi_r^2 + \ell(\ell+1)\xi_t^2]r^2\, dr}{M\left[|\xi_r(R)|^2 + \ell(\ell+1)|\xi_t(R)|^2\right]}.$$

**(I) RLS & SOLA inversion cost functionals / 역전 비용 함수**

RLS (least squares + Tikhonov):

$$\chi^2_{\rm reg} = \sum_{i=1}^N \left(\frac{\delta\omega_i/\omega_i - \Delta\omega_i}{\sigma_i}\right)^2 + \alpha^2 \int_0^R\left[\left(\frac{d^2}{dr^2}\frac{\delta\rho}{\rho}\right)^2 + \left(\frac{d^2}{dr^2}\frac{\delta c^2}{c^2}\right)^2\right]dr.$$

SOLA (subtractive OLA):

$$\chi^2_{\rm SOLA}(r_0) = \int\!\!\left(\sum_i c_i(r_0)K^i_{c^2,\rho} - \mathcal{T}(r_0,r)\right)^2\!dr + \beta \int\!\!\left(\sum_i c_i K^i_{\rho,c^2}\right)^2\!dr + \mu\sum_{i,j} c_i c_j E_{ij}.$$

**(J) Numerical values / 수치값**

| Quantity | Value |
|---|---|
| p-mode frequency peak | ν_max ≈ 3.1 mHz |
| Large separation Δν_☉ | ≈ 135.1 µHz |
| Small separation δν | ≈ 9 µHz (ℓ=0,2) |
| Convection-zone base | r_b = 0.713 ± 0.001 R_☉ |
| Surface helium | Y_s = 0.2485 ± 0.0035 |
| Initial helium | Y_0 = 0.273 ± 0.006 |
| Equatorial rotation (surface) | ~460 nHz (period ~25 d) |
| Polar rotation (surface) | ~330 nHz (period ~35 d) |
| Radiative interior rotation | ~430 nHz (solid body) |
| Solar angular momentum H | (190.0 ± 1.5)×10³⁹ kg·m²/s |
| Rotational kinetic energy T | (253.4 ± 7.2)×10³³ kg·m²/s² |
| Gravitational quadrupole J₂ | (2.18 ± 0.06)×10⁻⁷ |
| Solar-cycle ν shift @ 3 mHz | ~0.4 µHz (max minus min) |

**Worked numerical example — p-mode trapping**:
For a 3 mHz (ω = 2π×3000 ≈ 1.88×10⁴ rad/s) ℓ=20 p-mode, the lower turning point satisfies c(r_t)/r_t = ω/L = 1.88×10⁴/20.5 = 920 m/s/Mm. Using a typical c(r) ≈ 500 km/s at r ≈ 0.5 R_☉ ≈ 3.5×10⁸ m, we get c/r ≈ 1430 m/s/Mm > 920 ⇒ r_t deeper than 0.5 R_☉ (~0.4 R_☉). Higher ℓ ⇒ shallower trapping; hence modes of different ℓ sample different depths.

**Worked example — Duvall-based inversion check**:
For the solar c(r), ∫₀^R dr/c ≈ 3700 s (half round-trip acoustic time), giving Δν = 1/(2×3700) ≈ 135 µHz — matching observations to within a percent.

**Worked example — rotational splitting**:
At the equator ω_s/2π = 460 nHz (period 25.2 d). For an ℓ=20, m=20 mode centred at 3 mHz, δν_m ≈ m × 460 nHz = 9.2 µHz — directly observable in high-ℓ spectra and the basis for 2-D rotation inversions.

**Worked example — propagation diagram numerical values**:
At r = 0.5 R_☉, Model S gives N ≈ 400 µHz (buoyancy frequency), c ≈ 250 km/s. For ℓ = 5, S_ℓ = √30 × c/r ≈ √30 × 250/(0.5 × 6.96×10⁵) ≈ 3.9×10⁻³ rad/s ⇒ S_ℓ/2π ≈ 620 µHz. A 3 mHz p-mode (ω/2π = 3000 µHz ≫ S_ℓ, N) propagates; a 200 µHz g-mode (< S_ℓ, < N) is trapped.

**Worked example — tachocline detection via splitting**:
In the convection zone (r > 0.71 R_☉) the splittings depend strongly on θ (differential rotation). Below 0.71 R_☉ they are almost independent of θ (solid-body). The step transition width in the inverted Ω(r,θ=eq) vs r profile gives the tachocline thickness; analyses by Basu & Antia (2001) and Charbonneau et al. (1999) give thickness w = (0.039 ± 0.013) R_☉, centred at r_c = (0.692 ± 0.005) R_☉.

---

## 4b. Solar Cycle & Near-Surface Structural Changes / 태양 주기와 표층 구조 변화 (expanded)

**Solar-cycle frequency shifts** (Sect. 9 in the paper): Oscillation frequencies scale with solar activity indices like the 10.7 cm radio flux or Mg II core-to-wing ratio. Woodard & Noyes (1985) first noted this correlation; modern measurements (Chaplin et al. 2007) find

$$\Delta\nu(\text{cycle max} - \text{cycle min}) \approx 0.4\ \mu\text{Hz at 3 mHz},$$

following a Q_{nℓ} scaling Q_{nℓ} δν ∝ ν² that resembles a surface term — indicating that the frequency-sensitive magnetic perturbations lie within ~0.01 R_☉ of the surface. Even-order a-coefficients (a_2, a_4) also show solar-cycle variation; Baldner et al. (2009) infer a near-surface (0.999 R_☉) toroidal field of 380 ± 30 G and a deeper (0.996 R_☉) field of 1.4 ± 0.2 kG.

§9 요약: 태양 주기에 따라 주파수가 활동 지수와 상관관계를 보이며 최대–최소 차이는 3 mHz에서 약 0.4 µHz이다. 스케일링이 표면항 형태여서 자기 섭동은 표면 근방에 집중됨을 시사한다.

**g-modes — the "holy grail"**: Solar g-modes probe the core but decay exponentially through the convection zone, giving predicted surface amplitudes <1 mm/s. Appourchaux et al. (2010) reviewed the long history of searches; García et al. (2007) claim a detection of periodicity consistent with dipole-mode asymptotic spacing in GOLF data, but individual mode identification remains disputed. Detection of g-modes would revolutionise solar-core diagnostics.

**Cycle 24 peculiarity**: Cycle 24 (2008–2019) was ~30% weaker than cycle 23. Basu et al. (2012a) showed that near-surface magnetic layer during the cycle 23/24 minimum could have been anticipated, using helioseismic pre-minimum indicators — a "seismic prediction" success.

§9 확장: 태양 g-mode는 코어 탐색의 성배이나 CZ 통과로 진폭이 <1 mm/s로 억제된다. García+ 2007의 dipole-mode 주기성 검출 주장은 논쟁 중. Cycle 24 약화는 seismic precursor로 예측 가능했다.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
 1962  Leighton et al.: 5-min oscillation discovery
        │
 1970  Ulrich: trapped p-mode hypothesis
        │
 1975  Deubner k-ν diagnostic diagram
        │
 1979  Claverie et al.: first resolved modes
        │
 1983  Duvall & Harvey: modern frequency tables
        │
 1988  Christensen-Dalsgaard+: differential linearised inversions
        │
 1991  Christensen-Dalsgaard+: r_b determination
        │
 1995  GONG + SoHO/MDI launched
        │                     ────────────── Paper #5 (Gizon & Birch 2005): local helioseismology
 1996  Christensen-Dalsgaard+ Model S; 2-D Ω(r,θ) by Thompson+, Schou+
        │
 2002  SNO confirms ν-oscillation → solves solar neutrino problem
        │
 2005  Asplund+ low-Z abundances → "solar abundance problem" emerges
        │
 2009  Kepler launched → asteroseismology revolution
        │
 ★★★★★ 2016  Basu Living Review (THIS PAPER): comprehensive synthesis ★★★★★
        │
 2018  TESS launch; 2025+ PLATO planned
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **Paper #5 – Gizon & Birch (2005), "Local Helioseismology" (*Living Reviews*)** | Complementary partner review: Basu covers global modes (averaged spherical structure & rotation); Gizon & Birch cover local probes (time-distance, ring diagrams, holography) for 3-D subsurface flows and magnetic activity. Together they span the full helioseismology domain. | 쌍으로 읽어야 할 상보적 리뷰: 전역은 구조·평균회전, 로컬은 3D 아표면 흐름. |
| **Paper (Christensen-Dalsgaard 2002), "Helioseismology"** (*Rev. Mod. Phys.*) | Predecessor review focused on inversions and interior physics. Basu 2016 extends with solar-cycle variations, abundance problem, asteroseismology links. | 선행 리뷰, Basu 2016이 태양 주기·원소 함량·항성지진학 연장. |
| **Paper (Thompson et al. 1996, Schou et al. 1998)** | First reliable 2-D Ω(r,θ) inversion showing tachocline. Basu 2016 cites and updates with Howe et al. (2005) profiles. | 2D 회전 역전의 첫 결과; 타코클린 발견. |
| **Paper (Bahcall et al. 1998 / Bahcall 2003)** | Solar neutrino problem review. Basu (§7.1) uses c² inversions to rule out solar-model solutions, pointing to particle-physics. | 중성미자 문제 논리적 연결. |
| **Paper (Christensen-Dalsgaard et al. 1996, Model S)** | The canonical reference SSM used by Basu throughout for comparison and kernel construction. | 기준 태양 모델. |
| **Paper (Aerts, Christensen-Dalsgaard, Kurtz 2010), "Asteroseismology"** (textbook) | Basu §11 summarises; the textbook provides full treatment. | §11의 교과서적 확장. |
| **Paper (Duvall 1982), "A dispersion law for solar oscillations"** | Original discovery of Duvall's Law; Basu §3.3.1, §6.1 builds everything from it. | Duvall 법칙의 원전. |
| **Paper (Tassoul 1980), "Asymptotic approximations..."** | Source of the p-mode and g-mode asymptotic dispersion relations used throughout §3.3. | 점근관계의 수학적 기반. |

---

## 7. References / 참고문헌

- Basu, S., "Global seismology of the Sun", *Living Reviews in Solar Physics* **13**, 2 (2016). DOI: [10.1007/s41116-016-0003-4](https://doi.org/10.1007/s41116-016-0003-4)
- Gizon, L., and Birch, A. C., "Local Helioseismology", *Living Reviews in Solar Physics* **2**, 6 (2005).
- Christensen-Dalsgaard, J., "Helioseismology", *Rev. Mod. Phys.* **74**, 1073 (2002).
- Christensen-Dalsgaard, J. et al., "The current state of solar modeling (Model S)", *Science* **272**, 1286 (1996).
- Duvall, T. L. Jr., "A dispersion law for solar oscillations", *Nature* **300**, 242 (1982).
- Duvall, T. L. Jr., and Harvey, J. W., "Observations of solar oscillations of low and intermediate degree", *Nature* **302**, 24 (1983).
- Tassoul, M., "Asymptotic approximations for stellar nonradial pulsations", *ApJS* **43**, 469 (1980).
- Thompson, M. J. et al., "Differential rotation and dynamics of the solar interior", *Science* **272**, 1300 (1996).
- Schou, J. et al., "Helioseismic studies of differential rotation in the solar envelope by the SOI using MDI", *ApJ* **505**, 390 (1998).
- Howe, R., "Solar interior rotation and its variation", *Living Rev. Solar Phys.* **6**, 1 (2009).
- Christensen-Dalsgaard, J., Gough, D. O., Thompson, M. J., "The depth of the solar convection zone", *ApJ* **378**, 413 (1991).
- Basu, S., and Antia, H. M., "Helioseismology and solar abundances", *Phys. Rep.* **457**, 217 (2008).
- Bahcall, J. N., Basu, S., Pinsonneault, M., "How uncertain are solar neutrino predictions?", *Phys. Lett. B* **433**, 1 (1998).
- Serenelli, A. M., Basu, S., "Determining the initial helium abundance of the Sun", *ApJ* **719**, 865 (2010).
- Backus, G. E., Gilbert, J. F., "The resolving power of gross earth data", *Geophys. J.* **16**, 169 (1968).
- Pijpers, F. P., Thompson, M. J., "Faster formulations of the optimally localized averages method", *A&A* **262**, L33 (1992).
- Aerts, C., Christensen-Dalsgaard, J., Kurtz, D. W., *Asteroseismology* (Springer, 2010).
