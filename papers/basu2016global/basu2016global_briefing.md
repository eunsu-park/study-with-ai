---
title: "Pre-Reading Briefing: Global Seismology of the Sun"
paper_id: "49"
topic: Living_Reviews_in_Solar_Physics
date: 2026-04-23
type: briefing
---

# Global Seismology of the Sun: Pre-Reading Briefing / 태양의 전역 지진학 사전 읽기 브리핑

**Paper**: Basu, S., "Global Seismology of the Sun", *Living Reviews in Solar Physics* **13**:2 (2016). DOI: 10.1007/s41116-016-0003-4
**Author(s)**: Sarbani Basu (Yale University)
**Year**: 2016

---

## 1. 핵심 기여 / Core Contribution

이 논문은 전역 태양지진학(global helioseismology)의 이론적 토대와 관측적 성과를 집대성한 종합 리뷰이다. 태양 내부 구조와 동역학을 관측 가능한 광역 진동 모드(고차 p-mode)의 주파수와 그 분열(splitting)로부터 어떻게 역으로 추정(inversion)하는지를 체계적으로 설명하며, 표준 태양 모델(Standard Solar Model, SSM)과의 비교, 수렴대 바닥(convection-zone base, tachocline)과 헬륨 함량(Y_s), 차등회전(differential rotation), 태양 주기에 따른 미세 변화 등을 모두 다룬다. 태양 지진학이 태양 중성미자 문제(solar neutrino problem)의 해답을 제공하고, 태양 내부 구조를 ~0.05–0.96 R_☉ 구간에서 수 퍼센트 이내 정밀도로 결정할 수 있게 했음을 보여준다.

This paper is a comprehensive review of the theoretical foundations and observational achievements of global helioseismology. It systematically explains how the interior structure and dynamics of the Sun can be inverted from the frequencies and splittings of observable global oscillation modes (primarily high-order p-modes), and covers the comparison with Standard Solar Models (SSMs), the convection-zone base and tachocline, the helium abundance (Y_s), differential rotation, and small solar-cycle-related changes. It demonstrates how helioseismology provided the resolution to the solar neutrino problem and enabled determination of solar interior structure to within a few percent between roughly 0.05 R_☉ and 0.96 R_☉.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

1960년대 Leighton 등이 태양 표면의 5분 진동(5-minute oscillation)을 관측한 이래, 1970년대 Ulrich, Leibacher & Stein 등은 이 진동이 단순한 표면 현상이 아니라 내부에 갇힌 음향파(trapped acoustic waves, p-modes)임을 제안했다. 1975년 Deubner의 k-ν 다이어그램에서 능선(ridge) 구조가 확인되었고, 1979년 Claverie 등과 1983년 Duvall & Harvey의 관측으로 개별 모드가 분해되면서 현대 태양지진학이 시작되었다. 1990년대에는 BiSON, GONG 네트워크와 SoHO/MDI 우주관측이 연속적·전구적 관측 데이터를 제공하여 내부 음속·밀도·회전 프로파일을 정밀하게 결정할 수 있게 되었다.

Since Leighton et al.'s discovery of the solar 5-minute oscillation in the early 1960s, theoretical work in the 1970s (Ulrich; Leibacher & Stein) proposed that these oscillations are trapped internal acoustic waves (p-modes) rather than surface phenomena. Deubner (1975) confirmed the predicted k-ν ridges, and Claverie et al. (1979) and Duvall & Harvey (1983) resolved individual modes, marking the birth of modern helioseismology. In the 1990s, the BiSON and GONG ground-based networks and space-based SoHO/MDI provided continuous, full-disc observations that enabled precise determination of interior sound-speed, density, and rotation profiles.

### 타임라인 / Timeline

```
1962 ────── Leighton et al.: 5-minute oscillation 발견
1970 ────── Ulrich: trapped p-mode 이론 제안
1975 ────── Deubner: k-ν ridges 확인
1979 ────── Claverie et al.: 개별 global modes 분해
1983 ────── Duvall & Harvey: modern frequency tables
1988 ────── Christensen-Dalsgaard et al.: linearized inversion kernels
1990 ────── Libbrecht et al. (BBSO) frequencies published
1995 ────── GONG network + SoHO/MDI launched
1996 ────── Kosovichev/Christensen-Dalsgaard et al.: first 2D rotation profile
2002 ────── SNO confirms neutrino oscillation → solves neutrino problem
2005 ────── Asplund et al. low-Z abundances → "solar abundance problem"
2010 ────── Kepler launch enables asteroseismology of Sun-like stars
2016 ────── Basu review (this paper)
```

---

## 3. 필요한 배경 지식 / Prerequisites

- **별의 구조 방정식 / Stellar structure equations**: 연속방정식, 정역학평형, 에너지 보존, 온도기울기 (radiative vs. convective). Continuity, hydrostatic equilibrium, energy conservation, temperature gradient with radiative/convective transport.
- **유체역학 및 선형 섭동론 / Fluid dynamics & linear perturbation theory**: Eulerian vs. Lagrangian perturbation, 연속·운동량·Poisson 방정식의 섭동. Perturbed continuity, momentum, Poisson equations.
- **구면조화함수 / Spherical harmonics**: Y_ℓ^m (θ, φ) decomposition, degree ℓ, azimuthal order m.
- **음향파 이론 / Acoustic-wave theory**: sound speed c² = Γ₁P/ρ, wave equation with turning points.
- **Sturm–Liouville 고유값 문제 / Eigenvalue problems**: 고유진동수 ω_{n,ℓ}와 고유함수 ξ_{n,ℓ}.
- **역문제 / Inverse problems**: Regularised Least Squares (RLS), Optimally Localised Averages (OLA/SOLA) — Backus–Gilbert formalism.
- **표준 태양 모델 / Standard Solar Model (SSM) concept**: Model S (Christensen-Dalsgaard 1996).

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **p-mode** | Pressure-driven acoustic oscillation; dominant observed solar modes (periods ~3–8 min, ω² > S_ℓ² and > N²). 압력(음향) 진동 모드. |
| **g-mode** | Buoyancy-driven internal gravity mode; trapped in radiative interior, ω² < N², not yet cleanly detected in the Sun. 부력 복원의 내부 중력 모드. |
| **f-mode** | Fundamental/surface-gravity mode (n=0), ω² ≃ gk, used for radius determination. 표면중력 모드. |
| **Brunt–Väisälä frequency N** | N² = g(d ln P/dr / Γ₁ − d ln ρ/dr); buoyancy frequency; N²<0 ⇒ convection. 부력(파)진동수. |
| **Lamb frequency S_ℓ** | S_ℓ² = ℓ(ℓ+1)c²/r²; horizontal acoustic cut-off. 수평 음향 차단 주파수. |
| **Acoustic cut-off ω_c** | ω_c² = c²/(4H_ρ²)(1 − 2 dH_ρ/dr); upper turning point for p-modes. 음향 차단 주파수. |
| **Large separation Δν** | Δν = (2 ∫₀^R dr/c)^{−1}; mean spacing of consecutive n at fixed ℓ. 대분리. |
| **Splitting coefficients a_j(n,ℓ)** | Expansion of ν_{n,ℓ,m} − ν_{n,ℓ,0} in polynomials of m; odd a_j from rotation, even from asphericity/B-field. m-방향 분열계수. |
| **Inversion kernels K^i_{c²,ρ}, K^i_{ρ,c²}** | Linear response of δω_i/ω_i to structure perturbations δc²/c², δρ/ρ. 역전 커널. |
| **Tachocline** | Thin shear layer at base of convection zone (≈0.71 R_☉) between differentially-rotating CZ and solid-body radiative core. 타코클린(수렴대 바닥 shear layer). |
| **Standard Solar Model (SSM)** | Calibrated (α, Y_0) evolution model matching present L_☉, R_☉, Z/X, with standard microphysics. 표준 태양 모델. |
| **Surface term** | Systematic frequency offset from inadequate modelling of near-surface layers; filtered out in inversions via smooth F(ω). 표면항(표층 보정). |

---

## 5. 수식 미리보기 / Equations Preview

**(1) Hydrostatic equilibrium / 정역학평형**

$$\frac{dP}{dr} = -\rho g, \qquad g = \frac{G m(r)}{r^2}.$$

**(2) Adiabatic sound speed / 단열 음속**

$$c^2 = \frac{\Gamma_1 P}{\rho}.$$

태양 지진학의 중심 관측량: p-mode 주파수는 주로 c(r) 프로파일에 민감하다. The central observable: p-mode frequencies depend primarily on the c(r) profile.

**(3) Wave equation inside the Sun (Cowling, high-n, high-ℓ) / 태양 내부 파동방정식**

$$\frac{d^2 \xi_r}{dr^2} = \frac{\omega^2}{c^2}\left(1 - \frac{N^2}{\omega^2}\right)\left(\frac{S_\ell^2}{\omega^2} - 1\right)\xi_r.$$

진동하는 해는 ω² > S_ℓ² 와 ω² > N² (p-mode), 또는 ω² < S_ℓ² 와 ω² < N² (g-mode) 영역에서 존재. Oscillatory solutions exist where ω² > S_ℓ², N² (p-modes) or ω² < S_ℓ², N² (g-modes).

**(4) Duvall's Law / Duvall 법칙**

$$F(w) = \int_{r_t}^{R}\left(1 - \frac{L^2 c^2}{\omega^2 r^2}\right)^{1/2}\frac{dr}{c} = \frac{(n+\alpha_p)\pi}{\omega}, \quad w \equiv \frac{\omega}{L}, \; L = \ell+\tfrac{1}{2}.$$

모든 관측된 p-모드가 (n+α_p)π/ω 대 w의 단일 곡선으로 축소된다. All observed p-modes collapse onto a single curve in (n+α_p)π/ω vs. w.

**(5) Asymptotic dispersion (Tassoul) / 점근 분산관계**

$$\nu_{n,\ell} \simeq \left(n + \frac{\ell}{2} + \alpha_p\right)\Delta\nu, \qquad \Delta\nu = \left(2\int_0^R \frac{dr}{c}\right)^{-1}.$$

저차 고n 모드의 주파수; Δν는 평균 밀도(음속 적분)의 역수. For low-ℓ high-n modes; Δν is the inverse acoustic travel time.

**(6) Rotational splitting / 회전 분열**

$$\frac{\omega_{n,\ell,m} - \omega_{n,\ell,0}}{2\pi} = m \int\!\!\int K_{n\ell}^m(r,\theta)\,\Omega(r,\theta)\, dr\, d\theta,$$

즉 고유 대칭성을 깨뜨리는 회전은 (2ℓ+1)개의 m-다중항을 만든다. 등가 단순형 (uniform rotation): Ω_s ≃ m ω_0/(2π). Uniform-rotation simple form.

**(7) Structure-inversion linearised relation / 구조역전 선형관계**

$$\frac{\delta\omega_i}{\omega_i} = \int K^i_{c^2,\rho}(r)\,\frac{\delta c^2}{c^2}\,dr + \int K^i_{\rho,c^2}(r)\,\frac{\delta\rho}{\rho}\,dr + \frac{F_{\rm surf}(\omega_i)}{E_i}.$$

관측된 Sun-minus-Model 주파수차를 δc²/c², δρ/ρ 프로파일로 역전한다. Observed Sun-minus-Model frequency differences are inverted to δc²/c², δρ/ρ profiles.

---

## 6. 읽기 가이드 / Reading Guide

**1회독(Overview, 3 h)**: §1 서론, §3 oscillation equations, §6 inversions, §7 structure results, §8 rotation. / First pass (overview): Intro, oscillation equations, inversions, structure, rotation.

**2회독(Derivations, 6 h)**: §3.1–§3.3의 방정식 유도, Duvall 법칙과 점근 공식 (§3.3.1), RLS와 OLA 알고리즘(§6.3), 자세히. Second pass: derive equations §3.1–3.3, work through Duvall's law and asymptotic expressions (§3.3.1), RLS/OLA algorithms (§6.3).

**핵심 그림 / Key figures**: Fig.1 propagation diagram (N vs S_ℓ), Fig.2 Duvall's law, Fig.10–11 kernels, Fig.13 averaging kernels, Fig.16 δc²/c², δρ/ρ, Fig.17 r_b tachocline position, Fig.26 2-D Ω(r,θ) profile, Fig.27 solar-cycle frequency shifts.

**반드시 수행할 연습 / Must-do exercises**: (1) Propagation diagram 그리기, (2) Duvall 법칙 Abel-inversion으로 c(r) 복원, (3) Lorentzian 선형을 이용한 모드 power spectrum 모의, (4) 간단한 synthetic rotation splitting 생성.

---

## 7. 현대적 의의 / Modern Significance

전역 태양지진학은 태양 내부를 "실험실"로 변모시킨 대표적 성공 사례이다. 태양 중성미자 문제를 입자물리학 문제(neutrino oscillation, Sudbury Neutrino Observatory 2002)로 귀결시켰으며, 표준 태양 모델의 검증 도구로서 대류대 깊이(r_b = 0.713 ± 0.001 R_☉), 표면 He 함량(Y_s = 0.2485 ± 0.0035), 초기 He 함량(Y_0 = 0.273 ± 0.006)을 제공한다. 태양 주기에 따른 주파수·회전 변화 측정으로 활동 주기 물리도 진전시켰다. 이 리뷰의 방법론은 Kepler·CoRoT 그리고 곧 올 TESS·PLATO 미션이 수만 별에 적용하는 asteroseismology의 토대이며, 별의 질량·반지름·나이를 수 퍼센트 정밀도로 결정하게 해준다. 또한 §5의 로컬 지진학(Gizon & Birch 2005, Paper #5)과 상보적이다: 전역은 전체 반지름에 걸친 평균 구조·회전을 주고, 로컬은 표면 근처 3-D 유동·자기 활동을 본다.

Global helioseismology is a paradigmatic success story, turning the Sun's interior into a "laboratory." It reduced the solar neutrino problem to a particle-physics issue (neutrino oscillation; SNO 2002), and provides fundamental benchmarks for Standard Solar Models: convection-zone base r_b = 0.713 ± 0.001 R_☉, surface helium Y_s = 0.2485 ± 0.0035, initial helium Y_0 = 0.273 ± 0.006. Solar-cycle changes in frequencies and rotation have advanced our understanding of activity-cycle physics. The methodology underpins modern asteroseismology via Kepler/CoRoT and the forthcoming TESS/PLATO missions, enabling determination of stellar masses, radii, and ages to a few percent for tens of thousands of stars. It is complementary to local helioseismology (Gizon & Birch 2005, Paper #5): global helioseismology yields spherically-averaged structure and rotation throughout the radius, while local helioseismology probes three-dimensional subsurface flows and magnetic activity.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
