---
title: "Solar Structure and Evolution"
authors: ["Jørgen Christensen-Dalsgaard"]
year: 2021
journal: "Living Reviews in Solar Physics"
doi: "10.1007/s41116-020-00028-3"
topic: Living_Reviews_in_Solar_Physics
tags: [solar-structure, stellar-evolution, helioseismology, neutrinos, abundance-problem, standard-solar-model]
status: completed
date_started: 2026-04-23
date_completed: 2026-04-23
---

# 73. Solar Structure and Evolution / 태양의 구조와 진화

---

## 1. Core Contribution / 핵심 기여

This review provides a **comprehensive and self-contained treatment of the interior of the Sun**, covering the full chain of theory (stellar-structure equations + microphysics), evolution (pre-main-sequence through late stages), standard solar models (using Model S as reference), and all major observational tests (helioseismology, solar neutrinos, light-element abundances). Christensen-Dalsgaard — one of the architects of Model S (Christensen-Dalsgaard et al. 1996) and the sound-speed inversion program — emphasises two narratives: (i) the **triumph** of classical SSM theory, which converged to the helioseismic sound-speed profile to within ~0.2% in most of the interior and which reconciled the neutrino problem through neutrino oscillations (a physics revision, not a model revision); and (ii) the **unresolved solar abundance problem**, whereby 3D-hydrodynamics-based photospheric abundances (AGS05 → AGSS09, Z/X ≈ 0.0181) lower the opacity near the convection-zone base enough to create ~1% sound-speed discrepancies, mismatch the convection-zone depth (d_cz / R = 0.287 ± 0.001 observed vs ~0.284 for low-Z models), and fail to reproduce the helioseismic Y_s = 0.2485 ± 0.0034.

본 리뷰는 **태양 내부를 포괄적이고 자립적으로 다루는 종합 문헌**이다. 이론 체인(항성 구조 방정식 + 미시물리), 진화 단계(pre-main-sequence부터 후기까지), Model S를 기준으로 한 표준 태양 모델(SSM), 그리고 모든 주요 관측 검증(헬리오사이스몰로지, 태양 중성미자, 경원소 abundance)을 다룬다. Christensen-Dalsgaard는 Model S (1996)와 sound-speed inversion 프로그램의 설계자 중 한 명으로서 두 흐름을 강조한다: (i) **고전 SSM 이론의 승리**—내부 대부분 영역에서 헬리오사이스몰로지 음속과 ~0.2% 이내로 수렴했으며, 중성미자 문제는 중성미자 진동이라는 "입자 물리 수정"으로 해결됨; (ii) **미해결 태양 조성 문제**—3D 유체역학 기반 광구 abundance (AGSS09, Z/X ≈ 0.0181)를 사용하면 대류층 바닥 근처 opacity가 낮아져 ~1% 수준의 음속 불일치가 생기고, 대류층 깊이(관측: d_cz/R = 0.287 ± 0.001; 저메탈 모델: ~0.284)와 헬리오사이스몰로지 Y_s = 0.2485 ± 0.0034 값을 재현하지 못함.

---

## 2. Reading Notes / 읽기 노트

### Part I: Basics of Stellar Modelling (§2.1) / 항성 모델링 기초

The SSM is built on **four structure equations** under spherical symmetry (rotation, magnetic fields, hydrodynamical instabilities neglected):

SSM은 구대칭(회전, 자기장, 유체역학적 불안정성 무시) 하에 **네 개의 구조 방정식**으로 구성된다:

**Hydrostatic equilibrium (Eq. 1) / 정역학 평형**:
$$\frac{dp}{dr} = -\frac{G m \rho}{r^2}$$

**Mass conservation (Eq. 2) / 질량 보존**:
$$\frac{dm}{dr} = 4\pi r^2 \rho$$

**Energy equation (Eq. 3) / 에너지 방정식**:
$$\frac{dL}{dr} = 4\pi r^2 \left[\rho \epsilon - \rho\frac{d}{dt}\left(\frac{e}{\rho}\right) + \frac{p}{\rho}\frac{d\rho}{dt}\right]$$

**Temperature gradient (Eqs. 4-5) / 온도 구배**:
$$\frac{dT}{dr} = \nabla \frac{T}{p}\frac{dp}{dr}, \qquad \nabla_{\rm rad} = \frac{3}{16\pi a \tilde{c} G}\frac{\kappa p}{T^4}\frac{L(r)}{m(r)}$$

**Composition evolution (Eq. 6) / 조성 변화**:
$$\frac{\partial X_i}{\partial t} = \mathcal{R}_i + \frac{1}{r^2 \rho}\frac{\partial}{\partial r}\left[r^2 \rho \left(D_i \frac{\partial X_i}{\partial r} + V_i X_i\right)\right]$$
where 𝓡_i is the nuclear reaction rate, D_i the diffusion coefficient and V_i the settling velocity.

**Schwarzschild criterion (Eq. 8) / 대류 불안정 조건**:
$$\nabla_{\rm rad} > \nabla_{\rm ad} \Rightarrow \text{convectively unstable}$$
When the radiative gradient needed to carry L exceeds the adiabatic gradient, convection sets in; in the Sun this happens at r ≈ 0.713 R_sun outward.

복사로 L을 운반하는 데 필요한 구배가 단열 구배를 초과하면 대류가 발생. 태양에서는 r ≈ 0.713 R_sun 이상에서 이 조건이 만족됨.

### Part II: Basic Properties of the Sun (§2.2) / 태양의 기본 물리량

The Sun's globally determined parameters are used as calibration targets:

SSM은 다음의 관측된 태양 기본 물리량을 재현하도록 교정된다:

| Quantity / 물리량 | Value / 값 | Source / 출처 |
|---|---|---|
| GM_sun | 1.32712438 × 10²⁶ cm³/s² | Planetary motion |
| M_sun | 1.98848 × 10³³ g (CODATA 2014) | via G |
| R_sun (photospheric) | 6.95508 ± 0.00026 × 10¹⁰ cm | Brown & Christensen-Dalsgaard (1998) |
| L_sun | 3.828 × 10³³ erg/s (Kopp et al. 2016) | Solar irradiance |
| Age t_sun | 4.570 ± 0.006 Gyr | Meteoritic dating |
| Surface Z/X | GN93: 0.0245 / AGSS09: 0.0181 | Spectroscopy |
| Y_s (envelope) | 0.2485 ± 0.0034 | Helioseismology (Basu & Antia 2004) |
| Surface Ω/(2π) | 415.5 − 65.3 cos²θ − 66.7 cos⁴θ nHz (Eq. 11) | Doppler |
| d_cz / R | 0.287 ± 0.001 | Helioseismology |

The photospheric helium abundance **cannot** be obtained from spectroscopy (He lines form in uncertain chromospheric conditions) → helioseismic Y_s is the only reliable determination.

광구 He abundance는 분광학으로 측정 불가 (He 선은 불확실한 chromospheric 조건에서 형성됨). 따라서 Y_s = 0.2485는 오직 helioseismology만이 제공함.

### Part III: Microphysics (§2.3) / 미시물리

**Equation of State (§2.3.1) / 상태방정식**:
Simplest approximation = fully ionized ideal gas (Eq. 12):
$$p \simeq \frac{k_B \rho T}{\mu m_u}, \quad \nabla_{\rm ad} \simeq 2/5, \quad \Gamma_1 \simeq 5/3$$
with mean molecular weight $\mu = 4/(3 + 5X - Z)$ (Eq. 13). For the solar core X ≈ 0.35 → μ ≈ 0.84.
But realistic EOS (MHD, OPAL1996, OPAL2005, CEFF, SAHA-S) must include partial ionization (important near surface), Coulomb interactions (Γ_e = e²/(d_e k_B T)), partial electron degeneracy (ζ_e = λ_e³ n_e ≈ 2e^ψ), and relativistic effects (x_e = k_B T/(m_e c²) ≈ 0.0026 at centre). The relativistic correction to Γ_1 (Eq. 20) is detectable helioseismically:
$$\frac{\delta \Gamma_1}{\Gamma_1} \simeq -\frac{2 + 2X}{3 + 5X}x_e$$
(Elliott & Kosovichev 1998).

**Opacity (§2.3.2) / 불투명도**: Rosseland mean κ from OPAL (Rogers & Iglesias 1992, 1996) or OP (Badnell et al. 2005). Iron-group elements contribute dominantly at log T ~ 6-7 (CZ base). A ~20-30% opacity change near the CZ base propagates directly into the sound-speed difference (Sect. 4.2).

**Energy Generation (§2.3.3) / 에너지 생성**:
Net reaction: 4 ¹H → ⁴He + 2e⁺ + 2ν_e, releasing **26.73 MeV** per ⁴He (neutrino losses ~0.52 MeV, so usable ≈ 26.21 MeV).
- **PP-I chain** (Eq. 24): ¹H(¹H, e⁺ν_e)²D(¹H, γ)³He(³He, 2¹H)⁴He — 77% of solar luminosity.
- **PP-II**: ³He(⁴He, γ)⁷Be(e⁻,ν_e)⁷Li(¹H, ⁴He)⁴He — 23%.
- **PP-III**: ⁷Be(¹H, γ)⁸B(e⁺ν_e)⁸Be(⁴He)⁴He — 0.02%, but crucial for ⁸B neutrino flux (high energy).
- **CNO cycle** (Eq. 26): ¹²C(¹H,γ)¹³N(e⁺ν_e)¹³C(¹H,γ)¹⁴N(¹H,γ)¹⁵O(e⁺ν_e)¹⁵N(¹H,⁴He)¹²C. Bottleneck reaction ¹⁴N(¹H,γ)¹⁵O → T²⁰ dependence. Only **1.3% of luminosity** in present Sun, but ~11% at the centre where T is highest.

반응률의 온도 의존성 (Eq. 23): PP chain n ≈ 4 (약한 의존), CNO cycle n ≈ 20 (강한 의존; Z = 7인 ¹⁴N의 쿨롱 장벽이 크기 때문).

**Diffusion and Settling (§2.3.4) / 확산과 침강**:
Helium diffusion reduces Y_s by ~0.023 over the solar lifetime (Bahcall & Pinsonneault 1992). Inclusion of diffusion in SSMs improved helioseismic agreement dramatically (Christensen-Dalsgaard et al. 1993). Diffusion timescale in the Sun is ~10¹¹ yr below the convective envelope, so the effect is modest but essential for matching Z_s / X_s after a lifetime.

### Part IV: Convection (§2.5) / 대류 처리

Standard approach = **mixing-length theory (MLT)** (Böhm-Vitense 1958):
$$F_{\rm con} \sim \rho c_p T \frac{\ell^2 g^{1/2}}{H_p^{3/2}} (\nabla - \nabla_{\rm ad})^{3/2} \quad (\text{Eq. 32})$$
with ℓ = α_ML H_p (Eq. 34). α_ML is calibrated to reproduce solar R_sun, giving α_ML ~ 1.8-2.0 depending on formulation. Nearly the entire CZ is adiabatic (∇ ≈ ∇_ad) except the thin super-adiabatic layer just below the photosphere. 3D hydrodynamical simulations (Trampedach et al. 2014a,b; Magic et al. 2013) now provide realistic near-surface models, but they are too expensive for full evolutionary calculations.

### Part V: Model S and Sensitivity Analysis (§4) / Model S와 민감도 분석

**Model S** specifications (Christensen-Dalsgaard et al. 1996):
- Age: 4.6 Gyr, R = 6.9599 × 10¹⁰ cm, L = 3.846 × 10³³ erg/s
- EOS: OPAL (Rogers et al. 1996)
- Opacity: OPAL92 interior, Kurucz91 low-T
- Composition: GN93 (Z/X = 0.0245)
- Nuclear: Bahcall & Pinsonneault (1995)
- Convection: Böhm-Vitense (1958)

Properties of present-Sun Model S (Table 2):
- **X_0 = 0.70911, Z_0 = 0.019631** (initial)
- **T_c = 15.667 × 10⁶ K** (central)
- **ρ_c = 153.86 g/cm³** (central)
- **X_c = 0.33765** (central hydrogen)
- **Y_s = 0.24464** (surface helium)
- **d_cz / R = 0.28844** (CZ depth)

Sensitivity results (Table 3, in units of 10⁻³):
| Change | δX_0 | δT_c/T_c | δY_s | δ(d_cz/R) |
|---|---|---|---|---|
| Age 4.6 → 4.57 Gyr | -0.234 | -0.499 | 0.314 | -0.399 |
| GS98 composition | -2.152 | 1.857 | 2.220 | -4.999 |
| OP05 opacity | 1.988 | -1.265 | -0.991 | -1.511 |
| No electron screening | 1.174 | 5.457 | 3.464 | -2.450 |
| No diffusion | 8.868 | -13.477 | 19.793 | -15.202 |

→ No-diffusion case shows the dramatic effect of settling: Y_s raised by 0.02 because helium does not settle out.

### Part VI: Helioseismic Tests (§5.1) / 헬리오사이스몰로지 검증

Inversion of ~10⁶ observed p-mode frequencies yields differences $\delta_r c^2 / c^2$ and $\delta_r \rho / \rho$ between Sun and model (via Eq. 61):
$$\frac{\delta \omega_{nl}}{\omega_{nl}} = \int_0^{R_s}\left[K_{c^2,\rho}^{nl}(r)\frac{\delta_r c^2}{c^2}(r) + K_{\rho,c^2}^{nl}(r)\frac{\delta_r \rho}{\rho}(r)\right]dr + Q_{nl}^{-1}\mathcal{F}(\omega_{nl})$$

**Key results**:
- Model S agrees with the Sun to better than ~0.2% in c² over 0.2 < r/R < 0.7 (Fig. 39).
- A notable positive peak δ_r c² ≈ +0.004 appears just below CZ base (r/R ≈ 0.7) — reflects excessive H gradient from He settling.
- **d_cz = (0.287 ± 0.003) R_sun** (Christensen-Dalsgaard et al. 1991).
- **Tachocline** center at r_c = (0.693 ± 0.002) R_sun, width w = (0.039 ± 0.013) R_sun (Charbonneau et al. 1999).
- **Y_s = 0.2485 ± 0.0034** from Γ_1 glitch in HeII ionization zone (Basu & Antia 2004).
- Interior rotates nearly uniformly at ~435 nHz below CZ base (Fig. 44); no rapidly rotating core.

### Part VII: Solar Neutrino Results (§5.2) / 태양 중성미자 결과

Historical problem: Homestake (³⁷Cl) measured **2.56 ± 0.23 SNU** vs SSM prediction ~8 SNU (factor 3 deficit). Further experiments confirmed:

| Experiment | Measured | SSM (B16-GS98) | Ratio |
|---|---|---|---|
| Homestake (Cl) | 2.56 SNU | 7.84 SNU | 0.33 |
| SuperK (water, ⁸B) | 2.35 × 10⁶ cm⁻²s⁻¹ | 5.46 × 10⁶ | 0.43 |
| SAGE+GALLEX+GNO (Ga) | 66.1 SNU | 125.6 SNU | 0.53 |
| SNO (D₂O, CC) | 1.76 × 10⁶ cm⁻²s⁻¹ | — | — |
| SNO (D₂O, all ν) | 5.25 × 10⁶ cm⁻²s⁻¹ | 5.46 | 0.96 |
| Borexino (pp) | 6.6 × 10¹⁰ cm⁻²s⁻¹ | — | — |

**Resolution (SNO 2001-2002)**: Total neutrino flux (all flavours via neutral-current reaction on D₂O) agreed with SSM → the deficit was due to **neutrino oscillations** (ν_e → ν_μ, ν_τ), not a model problem. The survival probability at 10 MeV is P(ν_e → ν_e) = 0.317 ± 0.018 (Aharmim et al. 2013, SNO combined).

결과: 태양 중성미자 문제는 태양 모델이 아니라 **중성미자 물리학의 수정**(non-zero mass, MSW effect)으로 해결됨. 2002년 Nobel Prize (Davis, Koshiba), 2015년 Nobel Prize (McDonald).

Borexino also recently (2020b) detected **CNO neutrinos** at a rate consistent with both GS98 and AGSS09 models.

### Part VIII: The Solar Abundance Problem (§6) / 태양 조성 문제

**Revised abundances** (3D hydrodynamical atmosphere + NLTE):

| Element | AG89 | GN93 | GS98 | AGS05 | AGSS09 |
|---|---|---|---|---|---|
| log ε(C) | 8.56 | 8.55 | 8.52 | 8.39 | 8.43 |
| log ε(N) | 8.05 | 7.97 | 7.92 | 7.78 | 7.83 |
| log ε(O) | 8.93 | 8.87 | 8.83 | 8.66 | 8.69 |
| **Z/X** | **0.0275** | **0.0245** | **0.0231** | **0.0165** | **0.0181** |

The ~30% decrease in C, N, O abundances lowers the interior opacity, worsening agreement with helioseismology: the CZ becomes ~0.003 R_sun shallower than observed, Y_s drops below 0.248, and the sound-speed difference grows to ~1% (Fig. 40a, dot-dashed curve).

C, N, O abundance가 30% 감소 → 내부 opacity 감소 → helioseismic 부합성 악화. CZ 깊이 ~0.003 R_sun 얕아지고, Y_s 감소, 음속 차이 ~1% 증가.

**Proposed remedies**:
1. **Opacity increase**: 15-25% at CZ base would restore agreement, but OP vs OPAL comparisons suggest uncertainties ≲ 5-10%.
2. **Turbulent mixing** below the CZ base reduces the steep H gradient (Christensen-Dalsgaard et al. 2018, Fig. 41).
3. **Early accretion** of metal-poor material.
4. **Mass loss** stripping heavy elements preferentially.

Currently no single solution works; the problem remains open in 2021.

### Part IX: Light Element Abundances (§5.3) / 경원소 abundance

**Lithium**: Solar surface Li is depleted by a factor ~150 relative to meteoritic value (Asplund et al. 2009). This requires mixing at T > 2.5 × 10⁶ K, deeper than the present CZ base (T_bc ≈ 2.2 × 10⁶ K in Model S) — evidence for **extra mixing** below the CZ.
**Beryllium**: Not depleted (requires T > 3.5 × 10⁶ K for destruction) → mixing does not reach that temperature.
**³He/⁴He ratio**: Solar wind value (4.64 ± 0.09) × 10⁻⁴ matches initial value → no significant ³He dredge-up, constraining extra mixing.

리튬의 극단적 감소 (factor ~150)는 Schatzman (1969) 이래 알려진 고전적 문제. Be 미감소와 함께 mixing의 깊이가 2.5-3.5 × 10⁶ K 사이임을 제약.

### Part X: Solar Evolution Summary (§3) / 태양 진화 개요

**Pre-main-sequence (PMS)**: Gravitational contraction; early deuterium burning; Hayashi track (fully convective), then Henyey track (radiative core develops). Duration ~30 Myr before ZAMS.
**Main sequence**: Present Sun at age 4.57 Gyr, ~half-way through MS lifetime of ~10 Gyr. Central X has decreased from 0.71 to 0.34.
**Late stages**: After core-H exhaustion at ~10 Gyr, the Sun becomes subgiant, then ascends red-giant branch, helium flash at tip (~12 Gyr), horizontal branch He burning, asymptotic giant branch, then planetary nebula and white-dwarf remnant (~0.54 M_sun CO core).

주계열 이전(PMS): 중력수축과 D burning; Hayashi trajectory (fully convective) → Henyey trajectory. ZAMS 도달까지 ~30 Myr. 주계열: 4.57 Gyr (수명 ~10 Gyr의 절반). 중심 X는 0.71 → 0.34로 감소. 후기: ~10 Gyr에서 subgiant → red giant branch → helium flash → horizontal branch → AGB → planetary nebula → white dwarf.

### Part XI: Numerical Example — ⁸B Neutrino Flux Sensitivity / 수치 예시: ⁸B 중성미자 민감도

The ⁸B neutrino flux scales as Φ_⁸B ∝ T_c¹⁸ (Bahcall 1989). With Model S central T_c = 1.5667 × 10⁷ K giving Φ_⁸B = 5.46 × 10⁶ cm⁻²s⁻¹, a 1% decrease in T_c (to 1.551 × 10⁷ K) would reduce Φ_⁸B by a factor (0.99)¹⁸ ≈ 0.835, i.e., ~17% reduction.

Homestake predicted ~8 SNU, observed 2.56 SNU — factor 3.1 deficit. To reduce the SSM prediction by factor 3 would require T_c to drop by:
$$(T_c^{\rm new} / T_c^{\rm old})^{18} = 1/3 \Rightarrow T_c^{\rm new} / T_c^{\rm old} = 3^{-1/18} \approx 0.940$$
i.e., a 6% reduction in T_c — incompatible with helioseismic sound-speed constraints (which fix T/μ everywhere). Hence the neutrino deficit could not be a model problem.

⁸B 중성미자는 T_c¹⁸에 비례. Factor 3 감소를 위해 T_c 6% 감소가 필요했는데, 이는 helioseismology가 금지하는 값. 따라서 중성미자 문제는 모델 문제가 될 수 없었음.

### Part XII: Towards Distant Stars (§7) / 다른 별들로의 확장

The Sun serves as ground truth for asteroseismology with Kepler/TESS. **Solar twins** (same mass, age, composition) reveal subtle differences: e.g., the Sun appears slightly depleted in refractories relative to volatiles (Meléndez et al. 2009) — perhaps related to planet formation. Scaling relations:
$$\nu_{\rm max} \propto \frac{g}{\sqrt{T_{\rm eff}}}, \qquad \Delta\nu \propto \sqrt{\bar{\rho}}$$
connect asteroseismic observables to stellar parameters. The Sun's ν_max ≈ 3090 μHz and Δν ≈ 135 μHz anchor these calibrations.

---

## 3. Key Takeaways / 핵심 시사점

1. **The SSM is a remarkable triumph / SSM은 놀라운 성공이다** — 단 네 개의 구조 방정식과 미시물리만으로 계산된 1D 모델이 태양 내부 음속을 헬리오사이스몰로지와 0.2% 이내로 재현한다. 이것은 기초 물리학(중력, 이상기체 + 보정, 핵반응, 복사 수송)이 항성 내부 조건(T > 10⁷ K, ρ > 100 g/cm³)에서 정확히 작동함을 입증한다. / A 1D model using only four structure equations plus microphysics reproduces the helioseismic sound speed to 0.2% — a testament to the validity of basic physics (gravity, ideal gas with corrections, nuclear reactions, radiative transfer) under extreme stellar conditions.

2. **The neutrino problem was a physics problem, not a model problem / 중성미자 문제는 모델이 아닌 물리학 문제였다** — Homestake-SNO 논쟁에서 태양 모델을 수정하려는 수십 년의 시도(WIMPs, rapidly rotating core, core mixing 등) 모두 실패했고, 결국 **중성미자 진동**이라는 particle physics 수정으로 해결되었다. 이는 SSM 예측이 옳았고 particle physics가 틀렸음을 보여준 역사적 사례다. / Decades of attempts to modify solar models (WIMPs, rapid rotation, core mixing) all failed; the answer was **neutrino oscillations** — a revision of particle physics, not astrophysics. SSM predictions were correct; the Standard Model of particle physics needed extension.

3. **Helium settling is observationally verified / 헬륨 침강은 관측으로 입증됨** — 초기 Y_0 ≈ 0.271 대비 현재 Y_s = 0.2485로 ~0.023 감소. 확산/침강을 SSM에 포함했을 때 음속 부합성이 극적으로 개선(Christensen-Dalsgaard et al. 1993). / Observed envelope Y_s = 0.2485 vs initial Y_0 ≈ 0.271 = a 0.023 decrease due to settling, confirmed by dramatic improvement in helioseismic agreement when diffusion was added in 1993.

4. **Convection zone depth is a stringent constraint / 대류층 깊이는 엄격한 제약** — d_cz / R = 0.287 ± 0.001은 opacity, composition, diffusion physics에 매우 민감. AGSS09 조성은 이 값을 재현하지 못한다. / The CZ depth 0.287 ± 0.001 is acutely sensitive to opacity, composition and diffusion physics; AGSS09 composition fails to reproduce it.

5. **The solar abundance problem is the central unresolved puzzle / 태양 조성 문제는 현재의 핵심 미해결 문제** — 3D hydrodynamics 기반 AGSS09 (Z/X = 0.0181)는 다른 태양형 별들과 일관되지만 helioseismology를 위반하고, GS98 (Z/X = 0.0231)은 helioseismology에 부합하지만 항성 대기 이론과 불일치. 2021년 현재 해결 미정. / The 3D-hydro-based AGSS09 (Z/X=0.0181) is consistent with solar-type stars but violates helioseismology; GS98 (Z/X=0.0231) fits helioseismology but conflicts with atmosphere theory. Unresolved as of 2021.

6. **CNO neutrinos opened a new probe / CNO 중성미자는 새 탐침을 열었다** — Borexino (Agostini et al. 2020b)의 CNO neutrino 검출은 태양 core 금속량에 직접 민감한 제약을 제공. 앞으로 high-Z vs low-Z 논쟁을 해결할 잠재력이 있다. / Borexino's 2020 CNO neutrino detection provides a direct probe of core metallicity, with potential to distinguish high-Z from low-Z compositions.

7. **The Sun rotates nearly rigidly in the radiative interior / 태양 복사층은 거의 강체 회전** — 대류층의 latitudinal 차등 회전은 tachocline (~0.693 R_sun)에서 급격히 종료되고, 그 아래는 ~435 nHz의 거의 일정한 rotation rate. 초기 빠른 회전은 모두 소실됨 — angular momentum transport 메커니즘은 여전히 미해결. / The radiative interior rotates uniformly at ~435 nHz below the tachocline; initial rapid rotation was lost by an unidentified angular-momentum transport mechanism.

8. **Model S remains the reference / Model S는 여전히 기준** — 1996년 개발된 Model S는 EOS, opacity, 조성 갱신이 이후 많았음에도 불구하고, 여전히 heritage로 인해 helioseismic inversion의 reference model로 사용됨. 이 논문은 그 canonical documentation. / Despite many EOS, opacity and composition updates since 1996, Model S remains the canonical reference for helioseismic inversions; this review is its documentation.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Stellar Structure Equations / 항성 구조 방정식

The four structure equations (with diffusion/settling) under spherical symmetry:
$$\boxed{
\begin{aligned}
\frac{dp}{dr} &= -\frac{G m \rho}{r^2} \quad &&\text{(hydrostatic)} \\
\frac{dm}{dr} &= 4\pi r^2 \rho \quad &&\text{(mass)} \\
\frac{dL}{dr} &= 4\pi r^2 \left[\rho \epsilon - \rho\frac{d(e/\rho)}{dt} + \frac{p}{\rho}\frac{d\rho}{dt}\right] \quad &&\text{(energy)} \\
\frac{dT}{dr} &= \nabla \frac{T}{p}\frac{dp}{dr} \quad &&\text{(gradient)} \\
\frac{\partial X_i}{\partial t} &= \mathcal{R}_i + \frac{1}{r^2 \rho}\frac{\partial}{\partial r}\left[r^2 \rho (D_i \partial_r X_i + V_i X_i)\right] \quad &&\text{(composition)}
\end{aligned}
}$$

### 4.2 Temperature Gradients / 온도 구배

Radiative diffusion approximation:
$$\nabla_{\rm rad} = \frac{3}{16\pi a \tilde{c} G}\frac{\kappa p}{T^4}\frac{L(r)}{m(r)}$$

Adiabatic (fully ionized ideal gas):
$$\nabla_{\rm ad} = \left(\frac{d\ln T}{d\ln p}\right)_{\rm ad} \simeq \frac{2}{5}, \qquad \Gamma_1 \simeq 5/3$$

Schwarzschild instability:
$$\nabla_{\rm rad} > \nabla_{\rm ad} \Rightarrow \text{convection}$$

### 4.3 Equation of State / 상태 방정식

Ideal gas approximation:
$$p \simeq \frac{k_B \rho T}{\mu m_u}, \qquad \mu = \frac{4}{3 + 5X - Z}$$

Coulomb parameter (importance of interactions):
$$\Gamma_e = \frac{e^2}{d_e k_B T}, \quad d_e = \left(\frac{3}{4\pi n_e}\right)^{1/3}$$

Electron degeneracy parameter:
$$\zeta_e = \lambda_e^3 n_e \simeq 2 e^{\psi}, \quad \lambda_e = \frac{h}{(2\pi m_e k_B T)^{1/2}}$$

Relativistic correction to Γ_1:
$$\frac{\delta\Gamma_1}{\Gamma_1} \simeq -\frac{2+2X}{3+5X}x_e, \qquad x_e = \frac{k_B T}{m_e c^2}$$

At solar centre x_e ≈ 0.0026, giving δΓ_1/Γ_1 ≈ -0.001 — helioseismically detectable.

### 4.4 Nuclear Energy Generation / 핵 에너지 생성

Overall fusion: 4¹H → ⁴He + 2e⁺ + 2ν_e, ΔE = 26.73 MeV (− ν losses).

Reaction rate temperature dependence (Eq. 23):
$$r_{12} \propto T^n, \quad n = \frac{\eta - 2}{3}, \quad \eta = 42.487(Z_1 Z_2 A)^{1/3} T_6^{-1/3}$$
where $A = A_1 A_2/(A_1 + A_2)$ is reduced mass.

For p-p at T_6 ≈ 15: η ≈ 13.6, n ≈ 3.9 → ε_pp ∝ T⁴ approximately.
For ¹⁴N(¹H,γ) at T_6 ≈ 15: Z_1Z_2A large → n ≈ 20 → ε_CNO ∝ T²⁰.

Average neutrino losses per ⁴He:
- PP-I: 0.263 MeV (so effective 26.21 MeV)
- PP-II: 1.06 MeV
- PP-III: 7.46 MeV

Branching fractions at present solar centre: PP-I 23%, PP-II 77%, PP-III 0.2% (by reactions); luminosity contributions 77%, 23%, 0.02%.

### 4.5 Mixing-Length Convection / 혼합거리 대류

Convective flux:
$$F_{\rm con} \sim \rho c_p T \frac{\ell^2 g^{1/2}}{H_p^{3/2}}(\nabla - \nabla_{\rm ad})^{3/2}$$

Mixing length:
$$\ell = \alpha_{\rm ML} H_p, \qquad H_p = -\left(\frac{d\ln p}{dr}\right)^{-1}$$

Turbulent pressure:
$$p_t \sim \rho v^2 \sim \frac{\rho \ell^2 g}{H_p}(\nabla - \nabla_{\rm ad})$$

### 4.6 Helioseismic Inversion / 헬리오사이스몰로지 역변환

Frequency differences:
$$\frac{\delta \omega_{nl}}{\omega_{nl}} = \int_0^{R_s}\left[K_{c^2,\rho}^{nl}(r)\frac{\delta_r c^2}{c^2} + K_{\rho,c^2}^{nl}(r)\frac{\delta_r \rho}{\rho}\right]dr + Q_{nl}^{-1}\mathcal{F}(\omega_{nl})$$

Mass conservation constraint:
$$\int_0^{R_s}\frac{\delta_r \rho}{\rho}\rho r^2 dr = 0$$

Asymptotic frequency relation (Tassoul 1980):
$$\nu_{nl} \approx \Delta\nu\left(n + \frac{l}{2} + \epsilon\right) - d_{nl}$$

Large separation:
$$\Delta\nu = \left(2\int_0^R \frac{dr}{c}\right)^{-1}$$

Small separation (probes core):
$$\delta\nu_{nl} = \nu_{nl} - \nu_{n-1,l+2} \simeq -\frac{4l+6}{4\pi^2\nu_{nl}}\Delta\nu \int_0^R \frac{dc}{dr}\frac{dr}{r}$$

### 4.7 Worked Example: Central Temperature from Homology / 중심 온도 계산

For an ideal gas M_sun star, homology scaling gives
$$T_c \propto \frac{\mu M}{R}$$
With μ ≈ 0.84 (X = 0.35, Z = 0.02), M = 2 × 10³³ g, R = 7 × 10¹⁰ cm:
$$T_c \sim \frac{G M \mu m_u}{k_B R} \sim \frac{(6.67\times10^{-8})(2\times10^{33})(0.84)(1.66\times10^{-24})}{(1.38\times10^{-16})(7\times10^{10})}$$
$$\sim 1.9 \times 10^7 \text{ K}$$
Actual Model S value: T_c = 1.567 × 10⁷ K (the homology estimate overestimates because it uses mass-weighted μ; accurate structure needs the full equations).

단순 homology 추정치 (~1.9 × 10⁷ K)는 실제 Model S 값 (1.567 × 10⁷ K)보다 높음 — 실제 구조는 복사 수송과 변화하는 μ(r)를 풀어야 함.

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1870  Lane: First polytropic stellar model (mechanical equilibrium)
1906  Schwarzschild: Convective instability criterion
1920  Eddington: Hydrogen fusion as stellar energy source
1938  Bethe & Critchfield: PP chain | Bethe: CNO cycle
1954  Salpeter: Electron screening of nuclear reactions
1957  Schwarzschild et al.: First calibrated 1 M_sun SSM
1962  Leighton et al.: Solar 5-minute oscillations discovered
1968  Davis et al.: Homestake neutrino deficit (3 SNU vs 20 SNU)
              ⇨ "Solar neutrino problem" born
1976  Hill et al.: Global solar oscillations claim (controversial)
1979  Claverie et al.: Confirmed global 5-min modes
1980  Grec et al.: South Pole observations establish helioseismology
1984  Duvall: Asymptotic frequency relation
1985  Christensen-Dalsgaard et al.: First sound-speed inversion
1988  MHD equation of state (Mihalas, Hummer, Däppen)
1993  Christensen-Dalsgaard et al.: Diffusion/settling in SSM
       ⇨ sound speed agreement dramatically improved
1996  Model S (Christensen-Dalsgaard et al.) published — REFERENCE
1998  Grevesse & Sauval: GS98 composition (Z/X = 0.0231)
2001  SNO first results (CC + ES) hint at oscillations
2002  SNO neutral-current confirms oscillations — PROBLEM SOLVED
2005  Asplund et al. (AGS05): 3D hydro abundances, Z/X = 0.0165
       ⇨ "Solar abundance problem" born
2009  Asplund et al. (AGSS09): refined, Z/X = 0.0181
2011  Caffau et al. (C11): Z/X = 0.0209 (intermediate)
2017  Vinyoles et al.: B16-GS98 & B16-AGSS09 standard SSMs
2020  Borexino detects CNO neutrinos — directly probes core Z
2021  Christensen-Dalsgaard: THIS COMPREHENSIVE REVIEW
```

This review sits at a pivotal moment: classical SSM theory is validated to unprecedented precision, but the abundance problem demands new physics (extra mixing, opacity corrections, or revised spectroscopy).

이 리뷰는 중요한 시점에 위치: 고전 SSM 이론은 전례 없는 정밀도로 검증되었지만, 조성 문제는 새로운 물리학(추가 혼합, opacity 수정, 또는 분광학 재검토)을 요구함.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **#5 Bethe (1939) "Energy Production in Stars"** | Provides the PP chain and CNO cycle reactions that power this review's §2.3.3 | PP-I/II/III branching, CNO bottleneck at ¹⁴N, T²⁰ dependence — all from Bethe's 1939 foundation / 이 리뷰 §2.3.3의 PP chain과 CNO cycle이 Bethe 논문에서 유래 |
| **#48 Basu (2016) "Global Seismology"** | Provides the helioseismic inversion techniques used throughout §5.1 | Sound-speed inversions, optimally localized averages, kernel methods are core Basu methodology / §5.1의 inversion 방법론의 근거 |
| **#49 Stix (2002) "The Sun: An Introduction"** | Textbook complement covering atmospheres and dynamics; Christensen-Dalsgaard focuses on interior | Stix covers what CD21 does not (chromosphere, corona, magnetic activity) / 태양 대기와 활동 영역은 Stix, 내부는 본 리뷰가 담당 |
| **Bahcall et al. (2005) SSM B05** | Alternative SSM with focus on neutrino predictions | Compared in §5.2; essentially same core physics but different EOS/opacity combinations / §5.2에서 비교됨; 본질적으로 동일 core physics |
| **Asplund et al. (2009) AGSS09** | Source of the revised solar composition | Directly creates the "solar abundance problem" discussed in §6 / §6의 조성 문제를 직접 촉발 |
| **Vinyoles et al. (2017) B16-GS98/B16-AGSS09** | Modern SSM update with both compositions | Provides the neutrino flux predictions quoted throughout §5.2 / §5.2의 중성미자 flux 예측치 제공 |
| **SNO (Aharmim et al. 2013)** | Neutrino oscillation confirmation | Resolves the neutrino problem as particle physics, not astrophysics / 중성미자 문제를 astrophysics가 아닌 particle physics로 해결 |
| **Borexino (Agostini et al. 2020b)** | CNO neutrino detection | Opens new probe of solar core Z; future resolution of abundance problem / 태양 core 금속량의 새 탐침, 조성 문제 해결의 미래 열쇠 |

---

## 7. References / 참고문헌

- Christensen-Dalsgaard, J., "Solar structure and evolution", Living Reviews in Solar Physics, 18:2 (2021). DOI: 10.1007/s41116-020-00028-3
- Christensen-Dalsgaard, J., et al., "The current state of solar modeling", Science, 272, 1286-1292 (1996). [Model S paper]
- Bahcall, J. N., Pinsonneault, M. H., "Solar models with helium and heavy-element diffusion", Rev. Mod. Phys., 67, 781 (1995).
- Grevesse, N., Sauval, A. J., "Standard solar composition", Space Sci. Rev., 85, 161 (1998). [GS98]
- Asplund, M., Grevesse, N., Sauval, A. J., Scott, P., "The chemical composition of the Sun", ARA&A, 47, 481 (2009). [AGSS09]
- Aharmim, B., et al. (SNO Collaboration), "Combined analysis of all three phases of solar neutrino data from SNO", Phys. Rev. C, 88, 025501 (2013).
- Basu, S., Antia, H. M., "Helioseismology and solar abundances", Phys. Rep., 457, 217 (2008).
- Bahcall, J. N., "Neutrino Astrophysics", Cambridge University Press (1989).
- Vinyoles, N., et al., "A new generation of standard solar models", ApJ, 835, 202 (2017).
- Böhm-Vitense, E., "Über die Wasserstoffkonvektionszone...", Z. Astrophys., 46, 108 (1958).
- Rogers, F. J., Iglesias, C. A., "Radiative atomic Rosseland mean opacity tables", ApJS, 79, 507 (1992). [OPAL opacities]
- Bethe, H. A., "Energy production in stars", Phys. Rev., 55, 434 (1939).
- Schwarzschild, K., "Über das Gleichgewicht der Sonnenatmosphäre", Göttinger Nachr., 41 (1906).
- Trampedach, R., et al., "A grid of 3D stellar atmosphere models", MNRAS, 442, 805 (2014).
- Agostini, M., et al. (Borexino Collaboration), "Experimental evidence of neutrinos produced in the CNO fusion cycle in the Sun", Nature, 587, 577 (2020).
