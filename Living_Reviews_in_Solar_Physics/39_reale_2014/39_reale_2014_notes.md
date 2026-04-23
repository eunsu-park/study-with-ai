---
title: "Coronal Loops: Observations and Modeling of Confined Plasma"
authors: [Fabio Reale]
year: 2014
journal: "Living Reviews in Solar Physics"
doi: "10.12942/lrsp-2014-4"
topic: Living_Reviews_in_Solar_Physics
tags: [coronal-loops, corona, mhd, hydrodynamics, nanoflares, scaling-laws, coronal-heating, RTV, EBTEL]
status: completed
date_started: 2026-04-23
date_completed: 2026-04-23
---

# 39. Coronal Loops: Observations and Modeling of Confined Plasma / 코로나 루프: 관측과 갇힌 플라스마 모델링

---

## 1. Core Contribution / 핵심 기여

**English**: Reale (2014) offers an exhaustive Living Reviews synthesis of coronal loop physics — the arch-shaped magnetic flux tubes confining million-kelvin plasma that form the building blocks of the X-ray bright solar corona. The review is organized in two interlocking halves: an *observational framework* (Section 3) covering loop populations (bright points, active region, giant arches, flaring), morphology and fine structuring (down to ~100 km), thermal diagnostics (filter ratios, DEM, EM-loci), temporal variability (steady vs. impulsive), and flows/waves; and a *theoretical framework* (Section 4) built on the 1-D compressible hydrodynamic equations (continuity, momentum, energy) with Spitzer thermal conduction, optically-thin radiation Λ(T), and external heating H. This theoretical apparatus yields the celebrated RTV scaling laws T_max = 1.4×10³ (pL)^(1/3) and H = 3 p^(7/6) L^(-5/6), and leads naturally into multi-stranded, impulsively-heated (nanoflare) scenarios. The central unresolved question the review frames is whether coronal heating is dominated by DC processes (magnetic reconnection — nanoflares) or AC processes (wave dissipation — Alfvén, slow magnetosonic).

**한국어**: Reale (2014)는 백만 도 플라스마를 가두는 아치형 자기 플럭스 튜브인 코로나 루프에 대한 Living Reviews 종합 리뷰로, X선으로 밝게 빛나는 태양 코로나의 기본 구성 요소를 다룬다. 리뷰는 두 축으로 구성된다: *관측 틀* (Section 3) — 루프 집단(bright points, 활동영역, 거대 아치, flaring), 형태학과 미세구조(~100 km까지), 열적 진단(filter ratio, DEM, EM-loci), 시간 변동성(steady vs. 임펄시브), 흐름과 파동; *이론 틀* (Section 4) — Spitzer 열전도, 광학적으로 얇은 복사 Λ(T), 외부 가열 H를 포함한 1차원 압축성 유체 방정식(연속, 운동량, 에너지). 이 이론 체계는 유명한 RTV 스케일링 법칙 T_max = 1.4×10³ (pL)^(1/3) 와 H = 3 p^(7/6) L^(-5/6)을 유도하며, 자연스럽게 다중 가닥·임펄시브 가열(nanoflare) 시나리오로 이어진다. 리뷰가 제시하는 중심 미해결 문제는 코로나 가열이 DC 과정(자기 재결합 — 나노플레어)과 AC 과정(파동 소산 — Alfvén, slow magnetosonic) 중 어느 것이 지배적인가이다.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction and Historical Keynotes / 서론과 역사적 요지 (pp. 5–9)

**English**: The corona is revealed outside eclipses only in X-ray and EUV bands because the optically-thin >1 MK plasma emits most of its radiation there. Because the tenuous plasma is optically thin, intensity scales as n², so locally enhanced density appears as bright structures — the loops. Historical milestones: Grotrian (1939) and Edlén (1943) identified "coronium" as highly-ionized iron; Giacconi et al. (1965) achieved arcmin X-ray imaging; Vaiana et al. (1968) took the first X-ray flare image at few arcsec resolution; Skylab/S-054 (1973) took 32,000 photographs at 2″ resolution, enabling Rosner et al. (1978) to establish that loop lifetimes exceed their cooling times (hence quasi-equilibrium → scaling laws). SoHO (1995–), TRACE (1998), Hinode (2006), SDO (2010), and Hi-C (2013) progressively refined spatial (0.2″), spectral, and temporal (12 s) resolution.

**한국어**: 코로나는 일식 밖에서는 X선과 EUV 밴드에서만 보이는데, 광학적으로 얇은 >1 MK 플라스마가 대부분의 복사를 이 대역에서 방출하기 때문이다. 플라스마가 광학적으로 얇으므로 복사 강도가 n²에 비례하고, 국소적으로 밀도가 증가한 구조가 밝게 보이며 이것이 루프이다. 역사적 이정표: Grotrian (1939)과 Edlén (1943)이 "coronium"을 고이온화 철로 동정; Giacconi et al. (1965)의 arcmin X선 이미징; Vaiana et al. (1968)의 최초 X선 플레어 이미지(수 arcsec); Skylab/S-054 (1973)의 2″ 해상도 32,000장 촬영으로 Rosner et al. (1978)이 루프 수명이 냉각 시간보다 길다는 것(준평형 → 스케일링)을 확립. SoHO (1995–), TRACE (1998), Hinode (2006), SDO (2010), Hi-C (2013)로 공간(0.2″)·분광·시간(12초) 해상도가 점진적으로 향상되었다.

### Part II: Observational Framework — General Properties and Classification / 일반 특성과 분류 (Section 3.1, pp. 9–11)

**English**: Table 1 gives the canonical coronal loop parameter ranges across four morphological classes (Vaiana 1973):

| Type / 유형 | Length L [10⁹ cm] | T [MK] | n [10⁹ cm⁻³] | p [dyne cm⁻²] |
|---|---|---|---|---|
| Bright points | 0.1–1 | 2 | 5 | 3 |
| Active region | 1–10 | 3 | 1–10 | 1–10 |
| Giant arches | 10–100 | 1–2 | 0.1–1 | 0.1 |
| Flaring loops | 1–10 | >10 | >50 | >100 |

The magnetic field strength inferred from p ~ B²/8π (β ≈ 1) is B ~ 0.1–10 G. A key morphological fact is that loop cross-section is approximately **constant along the length** above the transition region, at variance with the strong divergence expected from potential (dipole) field extrapolation. Chae et al. (1998c) capture the transition-region variation: $A(T)/A_h = [1+(\Lambda^2-1)(T/T_h)^\nu]^{1/2}/\Lambda$, with $\Lambda=30$, $\nu=3.6$, $T_h=10^6$ K. Classification by temperature (Table 2): **cool** (0.1–1 MK, UV lines), **warm** (1–1.5 MK, EUV imagers like TRACE), **hot** (≥2 MK, X-ray and Fe XVI/XVII).

**한국어**: Table 1은 Vaiana (1973)의 네 가지 형태학적 분류에 따른 표준 파라미터 범위를 제공한다. 플라스마 베타 β ≈ 1 가정에서 p ~ B²/8π로부터 자기장 B ~ 0.1–10 G이 추정된다. 핵심 형태학적 사실은 루프 단면적이 전이영역 위에서 길이 방향으로 **거의 일정**하다는 것이며, 이는 potential(dipole) field extrapolation에서 예상되는 강한 발산과 대비된다. Chae et al. (1998c)의 전이영역 단면 변화식은 $A(T)/A_h = [1+(\Lambda^2-1)(T/T_h)^\nu]^{1/2}/\Lambda$, $\Lambda=30$, $\nu=3.6$, $T_h=10^6$ K. 온도별 분류(Table 2): **cool** (0.1–1 MK, UV 라인), **warm** (1–1.5 MK, TRACE 등 EUV 이미저), **hot** (≥2 MK, X선과 Fe XVI/XVII).

### Part III: Morphology, Fine Structuring, and Thermal Diagnostics / 형태학·미세구조·열적 진단 (Section 3.2–3.3, pp. 12–26)

**English**: Loops are approximately semicircular (Fig. 3), but STEREO triangulation reveals deviations up to 30% from circularity and non-planarity (Aschwanden et al., 2008, 2009). Fine structuring appears on scales of 100–1000 km: Hi-C (Cirtain et al., 2013) resolved 300–1000 km strands with evidence of magnetic braiding. Thermal diagnostics rely on filter ratios $R_{ij} = I_i/I_j = G_i(T)/G_j(T)$ which invert to give an isothermal T-estimate, but coronal loops are generally **multi-thermal** along the line of sight, requiring differential emission measure (DEM) inversion — an ill-posed problem yielding non-unique solutions. The CHIANTI atomic database (Dere et al., 1997, 2009) is the standard spectral tool. **EM-loci plots** (Fig. 7) plot EM_j = I_j/G_j(T) vs. T for multiple filters j; a tight crossing implies isothermal plasma, a spread implies multi-thermality. Hot-loop studies (Porter & Klimchuk, 1995) showed T ~ 2–6 MK with pressure 0.1–20 dyne cm⁻²; density in hot cores up to 10^10.5 cm⁻³.

**한국어**: 루프는 대체로 반원형(Fig. 3)이지만 STEREO 삼각측량 결과 원형에서 최대 30% 편차와 non-planarity가 드러났다(Aschwanden et al., 2008, 2009). 미세구조는 100–1000 km 규모로 나타나며, Hi-C (Cirtain et al., 2013)는 300–1000 km strand와 자기 braiding 증거를 분해했다. 열적 진단은 filter ratio $R_{ij} = I_i/I_j = G_i(T)/G_j(T)$ 에 의존하는데, 이는 등온 온도 추정을 주지만 루프는 일반적으로 **multi-thermal**하여 differential emission measure (DEM) 역변환이 필요하다 — 이는 비유일 해를 갖는 ill-posed 문제이다. CHIANTI 원자 데이터베이스가 표준 분광 도구이다. **EM-loci plot** (Fig. 7)은 여러 필터 j에 대해 EM_j = I_j/G_j(T) 대 T를 그린 것으로, 타이트한 교차는 등온 플라스마, 분산은 multi-thermal을 의미한다. Hot loop 연구(Porter & Klimchuk, 1995)는 T ~ 2–6 MK, 압력 0.1–20 dyne cm⁻²; hot core 밀도는 10^10.5 cm⁻³까지.

### Part IV: Temporal Analysis, Flows, and Waves / 시간 분석·흐름·파동 (Section 3.4–3.5, pp. 26–33)

**English**: Most loops show remarkable steadiness over timescales much longer than cooling times (τ_c ~ 1000 s), suggesting sustained heating (Rosner et al., 1978). Yet on smaller timescales nanoflare-like fluctuations of 10²⁵ erg (for hot SXT loops) and 10²³ erg (for warm TRACE loops) are observed (Sakamoto et al., 2008), with occurrence rates 0.4 and 30 nanoflares s⁻¹ respectively. Flows: siphon flows from pressure differences between footpoints; loop filling/draining from transient heating/cooling. SUMER and EIS reveal redshifts (~5–15 km/s) in transition region lines and blueshifts in hotter lines (Fig. 10, Dadashi et al., 2011), consistent with chromospheric evaporation. Waves: transverse kink oscillations (Nakariakov et al., 1999; Aschwanden et al., 1999a) — periods 2–10 min, damping times comparable to periods; slow magnetosonic (De Moortel et al., 2000); Alfvén waves (Tomczyk et al., 2007). Coronal seismology inverts these to infer B-field.

**한국어**: 대부분의 루프는 냉각 시간(τ_c ~ 1000 s)보다 훨씬 긴 시간 척도에서 현저히 안정적이어서 지속적 가열을 시사한다(Rosner et al., 1978). 그러나 더 작은 시간 척도에서는 10²⁵ erg(hot SXT loops)와 10²³ erg(warm TRACE loops) 규모의 nanoflare 변동이 관측되며(Sakamoto et al., 2008), 발생률은 각각 0.4, 30 nanoflares s⁻¹이다. 흐름: 발자국 간 압력차에 의한 siphon flow; 과도 가열/냉각에 의한 루프 채움/비움. SUMER과 EIS는 전이영역 라인에서 ~5–15 km/s 적색편이, 더 뜨거운 라인에서 청색편이(Fig. 10, Dadashi et al., 2011)를 보이며, 이는 채층 증발과 일치한다. 파동: 횡방향 kink 진동(Nakariakov et al., 1999; Aschwanden et al., 1999a) — 주기 2–10분, 감쇠 시간 주기에 필적; slow magnetosonic(De Moortel et al., 2000); Alfvén 파동(Tomczyk et al., 2007). 코로나 지진학은 이를 역변환하여 B 필드를 추정한다.

### Part V: Loop Physics and Modeling — Basics / 루프 물리와 모델링 — 기본 (Section 4.1, pp. 34–38)

**English**: For β ≪ 1, T ~ few MK, n ~ 10⁸–10¹⁰ cm⁻³, loop plasma is a compressible fluid flowing only along B. With constant cross-section assumed and curvature neglected, the 1-D hydrodynamic equations are:

Mass:   $\partial n/\partial t + \partial(nv)/\partial s = 0$ — written as $dn/dt = -n \partial v/\partial s$ (Eq. 4)

Momentum: $n m_H dv/dt = -\partial p/\partial s + n m_H g + \partial(\mu \partial v/\partial s)/\partial s$ (Eq. 5)

Energy: $d\epsilon/dt + (p+\epsilon)\partial v/\partial s = H - n² \beta_i P(T) + \mu(\partial v/\partial s)² + F_c$ (Eq. 6)

with $p = (1+\beta_i)n k_B T$, $\epsilon = (3/2)p + n \beta_i \chi$, and conductive flux $F_c = \partial(\kappa T^{5/2} \partial T/\partial s)/\partial s$ (Eq. 8). Here s is loop coordinate, v plasma velocity, μ viscosity, P(T) radiative losses per unit EM, β_i ionization fraction, κ ≈ 9×10⁻⁷ cgs (Spitzer), χ hydrogen ionization potential, H(s,t) heat input. For impulsive events, conductive cooling time is $\tau_c = 10.5 \kappa_B L^2 n_c / (\kappa T_0^{5/2}) \approx 1500\, n_9 L_9^2/T_6^{5/2}$ seconds (Eq. 12); radiative cooling time $\tau_r = 3 k_B T_M/(n_M P(T_M)) \approx 3000\, T_{M,6}^{3/2}/n_{M,9}$ s (Eq. 13), using P(T)=bT^α with b=1.5×10⁻¹⁹, α=-1/2 near 10 MK.

**한국어**: β ≪ 1, T ~ 수 MK, n ~ 10⁸–10¹⁰ cm⁻³ 조건에서 루프 플라스마는 B를 따라서만 흐르는 압축성 유체이다. 단면이 일정하고 곡률을 무시하면 1차원 유체역학 방정식은 다음과 같다 (질량, 운동량, 에너지 식 4–6). 여기서 p = (1+β_i)nk_BT, ε = (3/2)p + nβ_iχ, 전도 플럭스 F_c = ∂(κT^(5/2) ∂T/∂s)/∂s. s는 루프 좌표, v는 속도, μ는 점성, P(T)는 EM 단위당 복사 손실, β_i는 이온화 분율, κ ≈ 9×10⁻⁷ cgs (Spitzer), χ는 수소 이온화 퍼텐셜, H(s,t)는 가열 입력. 임펄시브 이벤트에서 전도 냉각 시간은 $\tau_c \approx 1500\, n_9 L_9^2/T_6^{5/2}$ 초, 복사 냉각 시간은 $\tau_r \approx 3000\, T_{M,6}^{3/2}/n_{M,9}$ 초.

### Part VI: Monolithic (Static) Loops — RTV Scaling Laws / 정적 루프와 RTV 스케일링 (Section 4.1.1, pp. 37–38)

**English**: For loops in hydrostatic equilibrium with symmetry about the apex, constant cross-section, L ≪ pressure scale height, uniform heating H(s)=const, and low conductive flux at the transition-region base, the energy balance between heating, conductive losses, and radiation gives the **Rosner-Tucker-Vaiana scaling laws** (Rosner et al., 1978):

$$T_{0,6} = 1.4\,(p L_9)^{1/3} \quad (\text{Eq. 9})$$
$$H_{-3} = 3\,p^{7/6}\,L_9^{-5/6} \quad (\text{Eq. 10})$$

where T_{0,6} is max (apex) temperature in MK, p in dyne cm⁻², L_9 is length in 10⁹ cm, and H_{-3} is volumetric heating rate in 10⁻³ erg cm⁻³ s⁻¹. These are in agreement with Skylab data within a factor 2. A worked example: for an active region loop with L = 5×10⁹ cm and p = 2 dyne cm⁻², T_max = 1.4×(2×5)^(1/3) MK = 1.4 × 10^(1/3) MK ≈ 3.0 MK; H = 3 × 2^(7/6) × 5^(-5/6) × 10⁻³ erg cm⁻³ s⁻¹ ≈ 1.7×10⁻³ erg cm⁻³ s⁻¹. Extensions: Serio et al. (1981) for loops above pressure scale height; Martens (2010) for non-uniform heating.

**한국어**: 정점에 대한 대칭, 일정 단면, L ≪ 압력 스케일 높이, 균일 가열 H(s)=const, 전이영역 밑면에서 낮은 전도 플럭스 조건 하에서, 가열·전도 손실·복사 사이의 에너지 균형으로부터 **Rosner-Tucker-Vaiana 스케일링 법칙** (Rosner et al., 1978)이 유도된다 (식 9, 식 10). T_{0,6}는 정점 최대 온도(MK), p는 dyne cm⁻², L_9는 길이(10⁹ cm), H_{-3}는 체적 가열률(10⁻³ erg cm⁻³ s⁻¹). Skylab 데이터와 factor 2 내에서 일치. 예제: L = 5×10⁹ cm, p = 2 dyne cm⁻²의 활동영역 루프 → T_max ≈ 3.0 MK, H ≈ 1.7×10⁻³ erg cm⁻³ s⁻¹. 확장: Serio et al. (1981)의 긴 루프, Martens (2010)의 비균일 가열.

### Part VII: Structured (Dynamic) Loops — Four-Phase Cooling / 4단계 냉각 (Section 4.1.2, pp. 38–44)

**English**: A loop strand ignited by an impulsive heat pulse evolves through four phases (Reale, 2007; Fig. 15/16):

- **Phase I — Heating**: From pulse start to temperature peak. Heat conducted rapidly downwards; T rises across the whole loop.
- **Phase II — Evaporation**: T settles to T_0; chromospheric plasma evaporates upward, filling the loop at isothermal sound-crossing timescale $\tau_{sd} = L/\sqrt{2k_B T_0/m} \approx 80\, L_9/\sqrt{T_{0,6}}$ s (Eq. 11).
- **Phase III — Conductive cooling**: Pulse ends; conduction dominates cooling with $\tau_c$ (Eq. 12). Density still rising; loop becomes **overdense**.
- **Phase IV — Radiative cooling**: When $\tau_c = \tau_r$, density peaks at $T_M$; radiation dominates thereafter with $\tau_r$ (Eq. 13). The decay in density–temperature space follows a path below (or approaching) the QSS equilibrium curve.

Global cooling timescale (Serio et al., 1991; Reale, 2007): $\tau_s = 4.8\times 10^{-4} L/\sqrt{T_0} \approx 500\, L_9/\sqrt{T_{0,6}}$ s (Eq. 14). Equilibrium density at apex: $n_0 = 1.3\times 10^6\, T_0^2/L$ (Eq. 16); time to reach flare steady state $t_{eq} \approx 2.3\,\tau_s$ (Eq. 15). Temperature at density maximum: $T_{M,6} = 0.9\,(n_{M,9} L_9)^{1/2}$ (Eq. 19). Duration of Phase III: $\Delta t_{0-M} \approx \tau_c \ln\psi$ with $\psi = T_0/T_M$ (Eq. 20).

**한국어**: 임펄시브 가열 펄스로 점화된 루프 가닥은 4단계를 거친다(Reale, 2007; Fig. 15/16): (I) **가열** — 펄스 시작부터 온도 정점까지, 열이 빠르게 하부로 전도되며 T 상승; (II) **증발** — T가 T_0로 정착하고 채층 플라스마가 상승하여 루프를 채움, 등온 음속 횡단 시간 τ_sd ≈ 80 L_9/√T_{0,6} 초; (III) **전도 냉각** — 펄스 종료 후 전도가 τ_c로 지배, 밀도는 여전히 증가하며 루프가 **overdense**가 됨; (IV) **복사 냉각** — τ_c = τ_r이 되는 시점에 밀도 정점 T_M, 이후 복사가 τ_r로 지배. 전역 냉각 시간 척도 τ_s ≈ 500 L_9/√T_{0,6} 초. 평형 밀도 n_0 = 1.3×10⁶ T_0²/L; flare 정상 상태 도달 시간 t_eq ≈ 2.3 τ_s. 밀도 최대시 온도 T_{M,6} = 0.9 (n_{M,9} L_9)^(1/2). Phase III 지속: Δt_{0-M} ≈ τ_c ln ψ, ψ = T_0/T_M.

### Part VIII: Fine Structuring, Flows, and Heating / 미세구조·흐름·가열 (Section 4.2–4.4, pp. 44–56)

**English**: Multi-strand models convolve many independent hydrodynamic strand solutions. Nanoflare-heated strand ensembles (Warren et al., 2002, 2003) explain flat filter-ratio distributions along TRACE warm loops and the observed "overdensity" (density higher than equilibrium RTV prediction — Winebarger et al., 2003a). Filament widths of 15 km → flat DEM; 150 km → peaked DEM (Cargill & Klimchuk, 2004). The nanoflare intensity distribution follows a power law $dN/dE \propto E^{-\alpha}$; Hudson (1991) showed that α > 2 is required for nanoflares alone to heat the corona. Observationally α ≈ 1.5–2.5 in various estimates. Klimchuk's (2006) six-step picture of coronal heating: (1) energy source, (2) conversion to heat, (3) plasma response, (4) emission spectrum, (5) observables. **DC (nanoflare) heating** (Parker, 1988): twisting of field lines by photospheric motions produces tangential discontinuities; reconnection ignites when misalignment exceeds ~15°. **AC (wave) heating** (Ionson, 1978; Hollweg, 1984): resonant absorption of Alfvén waves at ω_A ≈ 2π v_A/L, phase mixing, turbulent cascade. Both mechanisms dissipate on small scales; observational discrimination remains challenging.

**한국어**: Multi-strand 모델은 많은 독립된 유체역학 strand 해를 합성한다. Nanoflare로 가열된 strand 앙상블(Warren et al., 2002, 2003)은 TRACE warm loop에서 관측되는 균일 filter ratio 분포와 "overdensity"(평형 RTV 예측보다 높은 밀도 — Winebarger et al., 2003a)를 설명한다. Filament 폭 15 km → flat DEM, 150 km → peaked DEM (Cargill & Klimchuk, 2004). Nanoflare 강도 분포는 멱법칙 $dN/dE \propto E^{-\alpha}$; Hudson (1991)은 nanoflare 단독으로 코로나 가열에 α > 2가 필요함을 보였다. 관측적으로 α ≈ 1.5–2.5. Klimchuk (2006)의 여섯 단계 코로나 가열: (1) 에너지원, (2) 열로 변환, (3) 플라스마 반응, (4) 방출 스펙트럼, (5) 관측량. **DC(nanoflare) 가열** (Parker, 1988): 광구 운동이 field line을 꼬아 접선 불연속을 생성, 어긋남이 ~15°를 넘으면 재결합 점화. **AC(wave) 가열** (Ionson, 1978; Hollweg, 1984): ω_A ≈ 2π v_A/L 에서 Alfvén 파동의 공명 흡수, phase mixing, 난류 캐스케이드. 두 기구 모두 작은 규모에서 소산되며, 관측적 구별은 여전히 도전 과제이다.

### Part IX: EBTEL and Large-Scale Modeling / EBTEL과 대규모 모델링 (Section 4.1 & 4.5, pp. 36–37, 55–56)

**English**: The Enthalpy-Based Thin-Layer (EBTEL) 0-D model (Klimchuk et al., 2008; Cargill et al., 2012a,b) evolves loop-averaged T, n, p by treating enthalpy flux between corona and transition region. Key idea: during conductive cooling, energy lost by conduction to the TR emerges as enthalpy flux driving evaporation; during radiative cooling, enthalpy flux reverses (draining). EBTEL solves:
$$\frac{1}{2}L\frac{dp}{dt} = Q - (F_c + R_c), \quad L\frac{dn}{dt} = (F_c - R_{tr})/(5k_B T)$$
(schematic), enabling simulation of thousands of strand evolutions for large-scale active-region studies. Large-scale 3-D MHD "ab initio" models (Gudiksen & Nordlund, 2005; Hansteen et al., 2007) span photosphere → corona with non-grey, non-LTE radiative transfer and field-aligned conduction; they reproduce warm loop populations at ~10^6 K with magnetic dissipation rates (3–4)×10⁶ erg cm⁻² s⁻¹.

**한국어**: EBTEL (Enthalpy-Based Thin-Layer) 0-D 모델 (Klimchuk et al., 2008; Cargill et al., 2012a,b)은 코로나와 전이영역 간 엔탈피 플럭스를 고려해 루프 평균 T, n, p를 진화시킨다. 핵심: 전도 냉각 중 전이영역으로 손실되는 에너지가 엔탈피 플럭스로 나타나 증발을 유발; 복사 냉각 중 엔탈피 플럭스는 반대로 뒤집힘(비움). EBTEL은 위 방정식을 풀어 수천 개 strand 진화를 시뮬레이션할 수 있다. 대규모 3D MHD "ab initio" 모델 (Gudiksen & Nordlund, 2005; Hansteen et al., 2007)은 광구 → 코로나 전역에 걸쳐 non-grey, non-LTE radiative transfer와 field-aligned 전도를 포함; ~10⁶ K warm loop 인구를 (3–4)×10⁶ erg cm⁻² s⁻¹ 자기 소산률로 재현한다.

### Part X: Stellar Coronal Loops and Conclusions / 항성 코로나 루프와 결론 (Section 5–6, pp. 57–59)

**English**: Most solar-type stars have hotter, denser, more active coronae than the Sun (Güdel, 2004). Their loops are spatially unresolved but probed via flare decay analyses (Reale, 2002a, 2007) yielding loop length estimates. Young stars show giant loops exceeding the stellar radius (Favata et al., 2005). The solar corona as "Rosetta stone" approach (Orlando et al., 2000; Peres et al., 2000) applies solar-derived populations (quiet, AR, AR-cores, flares) to interpret stellar X-ray luminosities. Key open questions: nature of elementary loop components (monolithic vs 100 km strands); heating mechanism (DC vs AC); role of turbulent cascade; spicule contribution to coronal mass and heating. Future prospects: Solar Orbiter, Parker Solar Probe, IRIS cross-instrument synergy, MUSE/EUVST next-gen spectrometers.

**한국어**: 대부분의 태양형 항성은 태양보다 더 뜨겁고 밀도 높으며 활동적인 코로나를 가진다(Güdel, 2004). 그들의 루프는 공간 분해되지 않지만 플레어 decay 분석으로 탐구된다(Reale, 2002a, 2007). 젊은 별은 항성 반경을 초과하는 거대 루프를 보인다(Favata et al., 2005). 태양 코로나를 "Rosetta stone"으로 사용하는 접근(Orlando et al., 2000; Peres et al., 2000)은 태양에서 도출된 인구(quiet, AR, AR-cores, flares)로 항성 X선 광도를 해석한다. 주요 미해결 질문: 기본 루프 요소의 성질(monolithic vs 100 km strand); 가열 기구(DC vs AC); 난류 캐스케이드의 역할; 스피큘의 코로나 질량과 가열 기여. 미래 전망: Solar Orbiter, Parker Solar Probe, IRIS 다기기 협업, MUSE/EUVST 차세대 분광기.

### Part XI: Hydrostatic Loop Profiles (Serio Model) / 정역학 루프 단면 (Serio 모델) (Fig. 14, p. 39)

**English**: Beyond RTV scaling, the full hydrostatic solution of the 1-D equations gives detailed profiles T(s), n(s), p(s) along the loop. Figure 14 of the review (Serio et al., 1981 model) shows two representative cases: a high-pressure active-region (AR) loop with p ≈ 1 dyne cm⁻², n_apex ≈ 10⁹ cm⁻³, T_apex ≈ 3×10⁶ K; and a low-pressure "Empty" loop with p ≈ 0.01 dyne cm⁻², n_apex ≈ 10⁷ cm⁻³, T_apex ≈ 10⁶ K. The temperature rises steeply through the transition region (T ~ 10⁴ → 10⁶ K over ~1–100 km), then approaches its apex value gradually over the rest of the loop. Density falls exponentially with scale height $H_p = k_B T/(m g)$ which for 1 MK is ~50,000 km. For L > H_p loops the equilibrium breaks down (Serio et al., 1981 correction). The narrowness of the transition region is dictated by the balance between heat conduction and radiative losses; a too-coarse numerical grid drifts the TR, biasing chromospheric evaporation predictions (Bradshaw & Cargill, 2013).

**한국어**: RTV 스케일링을 넘어, 1차원 방정식의 전체 정역학 해는 루프를 따라 T(s), n(s), p(s)의 상세 단면을 제공한다. 리뷰의 Figure 14 (Serio et al., 1981 모델)는 두 대표 경우를 보여준다: 고압 활동영역(AR) 루프 p ≈ 1 dyne cm⁻², n_apex ≈ 10⁹ cm⁻³, T_apex ≈ 3×10⁶ K; 저압 "Empty" 루프 p ≈ 0.01 dyne cm⁻², n_apex ≈ 10⁷ cm⁻³, T_apex ≈ 10⁶ K. 온도는 전이영역에서 가파르게 상승(T ~ 10⁴ → 10⁶ K, ~1–100 km)하고, 그 후 정점값으로 완만히 접근한다. 밀도는 스케일 높이 $H_p = k_B T/(m g)$ 로 지수적으로 감소하며, 1 MK에서 ~50,000 km이다. L > H_p 인 루프에서는 평형이 깨진다(Serio et al., 1981 보정). 전이영역의 좁음은 열전도와 복사 손실 균형이 결정하며, 조악한 수치 격자는 TR을 표류시켜 채층 증발 예측을 편향시킨다(Bradshaw & Cargill, 2013).

### Part XII: Siphon Flows and Shocks / Siphon 흐름과 충격파 (Section 4.3, pp. 46–48)

**English**: When the two footpoints have different pressures (p₁ > p₂), a steady flow develops from high-pressure to low-pressure foot. The model of Orlando et al. (1995b) extends RTV to include subsonic flows; Orlando et al. (1995a) handles supersonic flows which inevitably generate a stationary shock at the transition from supersonic to subsonic downstream. Figure 18 shows a characteristic profile: subsonic upflow → sonic point near apex → supersonic downflow → shock near footpoint → subsonic again. Such shocks can shift the volumetric heating rate inferred by diagnostics and produce non-equilibrium ionization signatures in UV lines (blueshifted line profiles; Spadaro et al., 1990). Siphon flows thus complicate the classical hydrostatic picture and are one reason why observed warm loops deviate from RTV scaling. Critical Mach-number flows also alter density profiles and filter-ratio diagnostics.

**한국어**: 두 발자국의 압력이 다를 때(p₁ > p₂), 고압 → 저압 발자국으로 정상 흐름이 발생한다. Orlando et al. (1995b) 모델은 RTV를 확장해 subsonic 흐름을 포함시켰고, Orlando et al. (1995a)는 supersonic 흐름을 다루며 supersonic → subsonic 전이에서 필연적으로 정지 충격파가 생성됨을 보였다. Figure 18은 특징적 단면을 보여준다: subsonic 상승류 → 정점 근처 sonic point → supersonic 하강류 → 발자국 근처 충격파 → 다시 subsonic. 이러한 충격파는 진단이 추정하는 체적 가열률을 이동시키고, UV 라인에서 비평형 이온화 신호(청색편이 라인 프로파일; Spadaro et al., 1990)를 생성한다. 따라서 siphon 흐름은 고전적 정역학 그림을 복잡하게 만들며, warm loop가 RTV 스케일링에서 벗어나는 한 이유이다.

### Part XIII.5: Heating Signatures in Emission Measure Distributions / EM 분포의 가열 신호 (Section 4.4.1, pp. 51–53)

**English**: The shape of the DEM distribution encodes information about the temporal distribution of heating events. A broad (multi-thermal) distribution arises when many strands are randomly heated for short times and spend most of the time cooling across many temperatures ("crossing" intermediate T). A peaked (near-isothermal) distribution indicates sustained heating keeping plasma near a single T. Cargill & Klimchuk (2004) semi-analytical model showed that nanoflare-heated loops with strand diameters ~15 km → flat DEM (many cooling strands); ~150 km → peaked DEM. The cool-side slope of the DEM ($dEM/dT \propto T^a$ for T < T_peak) depends on low-frequency nanoflares (Warren et al., 2010b; Bradshaw & Klimchuk, 2011): broad emission measure distributions are a signature of low-frequency heating, peaked distributions of high-frequency. High frequency heating explains warm loop lifetime, high density, narrow DEM; but *not* the higher-T loops in X-rays (Warren et al., 2010a), suggesting multiple heating regimes coexist.

**한국어**: DEM 분포 형태는 가열 이벤트의 시간적 분포 정보를 담는다. 넓은 (multi-thermal) 분포는 많은 strand가 짧게 무작위로 가열되고 대부분의 시간을 많은 온도를 "지나며" 냉각할 때 발생한다. 뾰족한 (near-isothermal) 분포는 단일 T 근처 플라스마를 유지하는 지속적 가열을 나타낸다. Cargill & Klimchuk (2004) 반해석 모델은 strand 직경 ~15 km → flat DEM (많은 냉각 strand); ~150 km → peaked DEM임을 보였다. DEM의 cool-side 기울기 ($dEM/dT \propto T^a$, T < T_peak)는 저빈도 나노플레어에 의존한다 (Warren et al., 2010b; Bradshaw & Klimchuk, 2011): 넓은 EM 분포는 저빈도 가열, 뾰족한 분포는 고빈도 가열의 신호. 고빈도 가열은 warm loop 수명, 고밀도, 좁은 DEM을 설명하지만, X선의 더 뜨거운 루프는 *설명하지 못해* (Warren et al., 2010a), 여러 가열 영역이 공존함을 시사한다.

### Part XIV: Chromospheric Upflows and Spicules / 채층 상승류와 스피큘 (Section 3.5.1, p. 32)

**English**: De Pontieu et al. (2007a, 2009, 2011) proposed that type-II spicules — finger-like ejections with speeds 50–150 km/s lasting a few minutes — supply mass and energy to the corona via rapid heating to million-K temperatures. Evidence: spatial/temporal correlation between spicules and faint upflows in EUV Fe XIV, manifested as a blueshifted wing asymmetry. This would shift the coronal heating source from the corona itself to the chromosphere, posing a new challenge to theory. However, Klimchuk (2012) estimated that only a small fraction of coronal plasma can be supplied by chromospheric upflows: the implied mass flux is too high relative to observed flows. Alternative interpretations (Tian et al., 2011; Kamio et al., 2011) re-interpret the same signatures as waves or repetitive upflows. The debate highlights the intimate coupling between corona and chromosphere that all modern loop models must address.

**한국어**: De Pontieu et al. (2007a, 2009, 2011)은 type-II 스피큘 — 50–150 km/s 속도로 수 분 지속되는 손가락 모양 분출 — 이 빠른 가열을 통해 백만 K 온도까지 코로나에 질량과 에너지를 공급한다고 제안했다. 증거: 스피큘과 EUV Fe XIV에서 감지된 희미한 상승류의 공간·시간 상관, 청색편이 날개 비대칭으로 나타남. 이는 코로나 가열원을 코로나 자체에서 채층으로 이동시켜 이론에 새로운 도전을 제기한다. 그러나 Klimchuk (2012)은 채층 상승류로 공급될 수 있는 코로나 플라스마가 소량에 불과함을 추정했다: 암시되는 질량 플럭스가 관측된 흐름에 비해 너무 크기 때문이다. 대안적 해석(Tian et al., 2011; Kamio et al., 2011)은 같은 신호를 파동이나 반복 상승류로 재해석한다. 이 논쟁은 모든 현대 루프 모델이 다루어야 할 코로나-채층의 긴밀한 결합을 강조한다.

---

## 3. Key Takeaways / 핵심 시사점

1. **Coronal loops are the atomic units of the X-ray corona / 코로나 루프는 X선 코로나의 기본 단위** — The bright X-ray corona is essentially a collection of magnetic flux tubes confining plasma with T = 10⁵–10⁷ K and n = 10⁸–10¹² cm⁻³. Because optically thin emission ∝ n², dense loops dominate brightness. / X선 코로나는 본질적으로 10⁵–10⁷ K, 10⁸–10¹² cm⁻³ 플라스마를 가두는 자기 플럭스 튜브의 집합이다. 광학적으로 얇은 방출이 n²에 비례하므로 밀도 높은 루프가 밝기를 지배한다.

2. **RTV scaling laws are the null hypothesis / RTV 스케일링 법칙이 null hypothesis** — T_max = 1.4×10³(pL)^(1/3) and H = 3p^(7/6)L^(-5/6) capture the steady-state of a uniformly-heated loop. Deviations (e.g. overdense TRACE warm loops) signal dynamic, non-equilibrium conditions requiring time-dependent modeling. / T_max = 1.4×10³(pL)^(1/3) 와 H = 3p^(7/6)L^(-5/6)는 균일 가열 루프의 정상 상태를 포착한다. 벗어남(예: overdense TRACE warm loop)은 비평형 동적 조건의 신호이다.

3. **Four-phase cooling diagram is universal / 4단계 냉각 다이어그램의 보편성** — Impulsive heating → evaporation (τ_sd) → conductive cooling (τ_c) → radiative cooling (τ_r). The loop traces a characteristic path in density-temperature space below the QSS equilibrium curve, enabling "seismology" of the heating function from observed cooling slopes. / 임펄시브 가열 → 증발 → 전도 냉각 → 복사 냉각. 루프는 QSS 평형 곡선 아래에서 특징적 밀도-온도 경로를 그리며, 냉각 기울기로 가열 함수를 추정할 수 있다.

4. **Fine structuring is real but sub-resolution / 미세구조는 실재하나 해상도 이하** — Hi-C resolves 300–1000 km strands; theoretical and indirect evidence (coronal rain, DEM breadth, fuzzy hotter lines) support further substructure down to ~100 km. This motivates multi-strand modeling as the current paradigm. / Hi-C는 300–1000 km strand를 분해; 이론적·간접적 증거(coronal rain, DEM 너비, fuzzy 라인)가 ~100 km까지의 하위 구조를 지지한다. 이는 multi-strand 모델링을 현재 패러다임으로 자리잡게 한다.

5. **DC vs AC heating dichotomy remains open / DC vs AC 가열 이분법 미해결** — Nanoflares (DC) vs Alfvén wave dissipation (AC) are the two principal candidates. Power-law index α of dN/dE ~ E^(-α) must exceed 2 for nanoflares alone to heat the corona; observations typically yield α = 1.5–2.5. Turbulent cascades may provide common ground. / 나노플레어(DC)와 Alfvén 파동 소산(AC)이 주요 후보. dN/dE ~ E^(-α)의 α가 2를 초과해야 nanoflare 단독 가열 가능; 관측은 α = 1.5–2.5. 난류 캐스케이드가 공통 기반일 수 있다.

6. **Flows are ubiquitous and complex / 흐름은 편재하며 복잡** — Redshifts (5–15 km/s) in TR lines, blueshifts in hotter coronal lines, siphon flows, chromospheric upflows via spicules — all indicate constant mass cycling. Flows drag ions into the corona (explaining abundances) and may carry heating energy. / 전이영역 라인의 적색편이(5–15 km/s), 더 뜨거운 코로나 라인의 청색편이, siphon flow, 스피큘 채층 상승류 — 지속적 질량 순환. 흐름은 이온을 코로나로 끌어올리고(함량 설명), 가열 에너지를 운반할 수 있다.

7. **EBTEL enables large-scale modeling / EBTEL로 대규모 모델링 가능** — The 0-D enthalpy-flux framework compresses full 1-D hydrodynamics into tractable ODEs, enabling statistical ensembles of thousands of strands per active region. Essential for stellar applications where spatial resolution is absent. / 0-D 엔탈피 플럭스 프레임워크는 전체 1D 유체역학을 다루기 쉬운 ODE로 압축하여, 활동영역당 수천 가닥 통계 앙상블을 가능케 한다. 공간 해상도가 없는 항성 응용에서 필수적.

8. **Coronal seismology inverts loop oscillations to B / 루프 진동으로 B 추정하는 코로나 지진학** — Kink period P = 2L/c_k with c_k ~ v_A allows inferring coronal magnetic field strength (~10 G) from TRACE/AIA oscillation observations. A unique direct probe of the otherwise invisible coronal B-field. / Kink 주기 P = 2L/c_k, c_k ~ v_A 로부터 TRACE/AIA 진동 관측에서 코로나 자기장 세기(~10 G)를 추정. 보이지 않는 코로나 B 필드의 유일한 직접 탐침.

9. **Hot-warm-cool dichotomy may reflect physical distinctions / hot-warm-cool 이분법은 물리적 차이를 반영할 수 있음** — Observations hint that warm loops may not simply be hot loops at different instrument sensitivity; they often appear overdense and multi-thermal, suggesting transient/non-equilibrium heating states distinct from the quasi-steady hot loops seen in X-rays. This motivates classification based on *physical regime* rather than bandpass. / 관측은 warm loop가 단순히 다른 기기 감도의 hot loop가 아닐 수 있음을 시사한다; 종종 overdense하고 multi-thermal하여, X선에서 보이는 준정상 hot loop와는 구별되는 과도/비평형 가열 상태를 암시한다. 이는 bandpass가 아닌 *물리적 영역* 기반 분류를 지지한다.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 One-Dimensional Hydrodynamic Equations / 1차원 유체역학 방정식

The fundamental system (Eqs. 4–8 of the review):

$$\frac{dn}{dt} = -n\frac{\partial v}{\partial s}$$

$$n m_H \frac{dv}{dt} = -\frac{\partial p}{\partial s} + n m_H g + \frac{\partial}{\partial s}\left(\mu \frac{\partial v}{\partial s}\right)$$

$$\frac{d\epsilon}{dt} + (p+\epsilon)\frac{\partial v}{\partial s} = H - n^2 \beta_i P(T) + \mu\left(\frac{\partial v}{\partial s}\right)^2 + F_c$$

with equation of state $p = (1+\beta_i) n k_B T$, internal energy $\epsilon = \frac{3}{2}p + n\beta_i\chi$, and Spitzer thermal conductive flux:

$$F_c = \frac{\partial}{\partial s}\left(\kappa T^{5/2}\frac{\partial T}{\partial s}\right), \quad \kappa = 9\times 10^{-7}\ \text{erg cm}^{-1}\,\text{s}^{-1}\,\text{K}^{-7/2}$$

**Variables / 변수**:
- n: hydrogen number density [cm⁻³] / 수소 수밀도
- v: plasma bulk velocity [cm/s] / 플라스마 벌크 속도
- s: coordinate along loop [cm] / 루프 방향 좌표
- T: temperature [K] / 온도
- p: gas pressure [dyne cm⁻²] / 가스 압력
- ε: internal energy density [erg cm⁻³] / 내부 에너지 밀도
- H(s,t): heating rate per unit volume [erg cm⁻³ s⁻¹] / 단위 체적당 가열률
- Λ(T) or P(T): radiative loss function [erg cm³ s⁻¹] / 복사 손실 함수
- κ: Spitzer thermal conductivity coefficient / Spitzer 열전도 계수

### 4.2 RTV Scaling Laws / RTV 스케일링

Under hydrostatic equilibrium, apex symmetry, constant A, L ≪ scale height, uniform H:

$$\boxed{T_{0,6} = 1.4\,(p\,L_9)^{1/3}}$$

$$\boxed{H_{-3} = 3\,p^{7/6}\,L_9^{-5/6}}$$

Units: $T_{0,6} = T_0/10^6$ K, $L_9 = L/10^9$ cm, $p$ in dyne cm⁻², $H_{-3} = H/10^{-3}$ erg cm⁻³ s⁻¹.

**Worked example / 예제**: For an active region loop L = 2×10⁹ cm, p = 4 dyne cm⁻²:
- $T_{0,6} = 1.4 \times (4\times 2)^{1/3} = 1.4 \times 2 = 2.8$ → **T_max ≈ 2.8 MK**
- $H_{-3} = 3 \times 4^{7/6} \times 2^{-5/6} \approx 3 \times 5.04 \times 0.561 \approx 8.5$ → **H ≈ 8.5×10⁻³ erg cm⁻³ s⁻¹**

### 4.3 Cooling Timescales / 냉각 시간

Isothermal sound-crossing (evaporation timescale):
$$\tau_{sd} = \frac{L}{\sqrt{2k_B T_0/m}} \approx 80\, \frac{L_9}{\sqrt{T_{0,6}}}\ \text{s}$$

Conductive cooling:
$$\tau_c = \frac{3 n_c k_B T_0 L^2}{\frac{2}{7}\kappa T_0^{7/2}} = \frac{10.5\, n_c k_B L^2}{\kappa T_0^{5/2}} \approx 1500\,\frac{n_9 L_9^2}{T_{0,6}^{5/2}}\ \text{s}$$

Radiative cooling (with $P(T)=b T^\alpha$, $b=1.5\times 10^{-19}$, $\alpha=-1/2$):
$$\tau_r = \frac{3 k_B T_M}{n_M P(T_M)} \approx 3000\,\frac{T_{M,6}^{3/2}}{n_{M,9}}\ \text{s}$$

Global thermodynamic timescale (Serio et al. 1991):
$$\tau_s = 4.8\times 10^{-4}\,\frac{L}{\sqrt{T_0}} \approx 500\,\frac{L_9}{\sqrt{T_{0,6}}}\ \text{s}$$

### 4.4 Radiative Loss Function / 복사 손실 함수

Approximation used in the review: $P(T) = b T^\alpha$ with $b = 1.5\times 10^{-19}$ erg cm³ K^{1/2} s⁻¹ and $\alpha = -1/2$ (near 10 MK regime). More generally $\Lambda(T)$ is a piecewise-tabulated function from CHIANTI (Dere et al., 1997, 2009): peaks near T ~ 10⁵.³ K, flat plateau 10⁶–10⁷ K, decreasing toward 10⁸ K. Rosner-Tucker-Vaiana (1978) used a piecewise power-law approximation.

### 4.5 Emission Measure Diagnostics / 방출 측도 진단

For optically thin plasma observed in filter j:
$$I_j = \int \text{EM}(T)\, G_j(T)\, dT, \quad \text{EM} = \int_V n^2\, dV$$

Filter ratio for isothermal case:
$$R_{ij} = \frac{I_i}{I_j} = \frac{G_i(T)}{G_j(T)}$$

Differential emission measure $dEM/dT$ is obtained from multi-line or multi-filter inversion — an ill-posed inverse problem.

### 4.6 Nanoflare Statistics / 나노플레어 통계

Impulsive event intensity distribution (Hudson, 1991):
$$\frac{dN}{dE} = N_0 E^{-\alpha}$$

For nanoflares to heat the corona, the total energy integral must converge at low E, requiring $\alpha > 2$. Observed estimates yield α = 1.5–2.5 across different events. Individual nanoflare energies ~10²³ erg (warm TRACE loops) to 10²⁵ erg (hot SXT loops) (Sakamoto et al., 2008).

### 4.7 Kink Mode Coronal Seismology / kink 모드 코로나 지진학

For a cylindrical magnetic flux tube (internal Alfvén speed $v_{A,i}$, external $v_{A,e}$, density contrast $\rho_i/\rho_e$):

$$c_k = \sqrt{\frac{\rho_i v_{A,i}^2 + \rho_e v_{A,e}^2}{\rho_i + \rho_e}} = v_{A,i}\sqrt{\frac{2}{1 + \rho_e/\rho_i}}\quad \text{(in low-}\beta, \text{coronal regime)}$$

Standing kink mode fundamental period:
$$P_{\text{kink}} = \frac{2L}{c_k}$$

Inverting an observed period (e.g. P = 300 s, L = 1.5×10¹⁰ cm) gives $c_k = 2L/P = 10³$ km/s → if ρ_e/ρ_i = 0.1, then v_{A,i} ≈ 750 km/s, and for n = 10⁹ cm⁻³ (ρ = 1.67×10⁻¹⁵ g cm⁻³), **B ≈ v_{A,i}√(4πρ) ≈ 30 G**.

### 4.8 EBTEL 0-D Equations / EBTEL 0-D 방정식

Loop-averaged quantities $\bar{T}$, $\bar{n}$, $\bar{p}$ evolve under:

$$\frac{L}{2}\frac{d\bar{p}}{dt} = \frac{2}{3}\left(Q L - R_c - F_{TR}\right)$$

$$\frac{L}{2}\frac{d\bar{n}}{dt} = \frac{c_2 F_{TR}}{5 k_B \bar{T}} - R_{TR\text{-to-corona enthalpy}}$$

where $R_c = \bar{n}^2 \Lambda(\bar{T}) L$ is coronal radiation, $F_{TR}$ the transition-region conductive flux, $c_2 \approx 0.9$ is a correction factor. EBTEL captures evaporation/draining through the enthalpy flux term without resolving the transition region — a major computational saving.

### 4.9 Pressure Scale Height and Gravity / 압력 스케일 높이와 중력

In the corona, the hydrostatic pressure scale height is:
$$H_p = \frac{k_B T}{m_H g_\odot} \approx 5 \times 10^9 \left(\frac{T}{10^6\,\text{K}}\right)\ \text{cm}$$

with solar surface gravity $g_\odot = 2.74\times 10^4$ cm s⁻². For T = 1 MK, $H_p \approx 50{,}000$ km; for T = 3 MK, $H_p \approx 150{,}000$ km. Loops with $L > H_p$ violate the constant-pressure assumption of the original RTV derivation, and need the extended Serio et al. (1981) treatment.

### 4.10 Plasma Beta in Loops / 루프에서의 플라스마 베타

$$\beta = \frac{p_{\text{gas}}}{p_{\text{mag}}} = \frac{n k_B T}{B^2/8\pi}$$

For n = 10⁹ cm⁻³, T = 10⁶ K, B = 10 G: $\beta = (10^9 \times 1.38\times 10^{-16} \times 10^6)/(10^2/(8\pi)) \approx 1.4\times 10^{-7}/3.98 \approx 0.035$. Thus β ≪ 1 everywhere in typical coronal loops, justifying the assumption that magnetic field controls plasma motion and heat conduction is field-aligned.

### 4.11 Alfvén Speed and Wave Heating / Alfvén 속도와 파동 가열

Alfvén speed in a fully-ionized hydrogen plasma:
$$v_A = \frac{B}{\sqrt{4\pi\rho}} = \frac{B}{\sqrt{4\pi n m_H}} \approx 2.2\times 10^{11}\, \frac{B}{\sqrt{n}}\ \text{cm/s}$$

For B = 10 G, n = 10⁹ cm⁻³: $v_A \approx 2.2\times 10^{11} \times 10/\sqrt{10^9} \approx 7\times 10^7$ cm/s = 700 km/s. Resonant absorption condition: waves with frequency $\omega_A \approx 2\pi v_A/L$ are efficiently absorbed. For L = 10¹⁰ cm, this gives $P = L/v_A \approx 150$ s, matching observed kink periods.

### 4.12 Energy Budget Estimate / 에너지 수지 추정

Total coronal heating requirement (Withbroe & Noyes, 1977): $F_{\text{corona}} \sim 10^7$ erg cm⁻² s⁻¹ for active regions. For a loop with volume $V = A \cdot L$, base area A, the volumetric heating rate times L must match this flux: $H \cdot L \sim F/A$. Taking H = 10⁻³ erg cm⁻³ s⁻¹ and L = 10¹⁰ cm gives H·L = 10⁷ erg cm⁻² s⁻¹, consistent with required flux — validating RTV-scale energy budget.

### 4.13 Derivation Sketch of RTV / RTV 유도 개요

**English**: Starting from the hydrostatic energy equation with v = 0, uniform pressure p (since L ≪ H_p), constant H, and zero conductive flux at base:
$$\frac{d}{ds}\left(\kappa T^{5/2} \frac{dT}{ds}\right) + H - n^2 \Lambda(T) = 0$$

With $n = p/(2 k_B T)$ and $\Lambda(T) = \chi_0 T^\gamma$ (piecewise power law), one integrates once with boundary conditions $T(0) = 0$, $T(s_{\text{apex}}) = T_0$, $dT/ds|_{\text{apex}} = 0$. The integration constants produce relations between $T_0$, L, and the heating rate per unit EM. Dimensional analysis of these constraints yields the scaling $T_0 \propto (pL)^{1/3}$, with the numerical prefactor 1.4×10³ calibrated against numerical solutions using the standard Λ(T). The volumetric heating rate scaling H ~ p^{7/6} L^{-5/6} follows from substituting T_0 back and imposing energy balance.

**한국어**: 정역학 에너지 방정식 (v = 0), 균일 압력 p (L ≪ H_p이므로), 일정 H, 밑면 전도 플럭스 0 조건에서 시작: 위 방정식. $n = p/(2 k_B T)$와 $\Lambda(T) = \chi_0 T^\gamma$ (구간별 멱법칙)을 대입하고, 경계조건 $T(0) = 0$, $T(s_{\text{apex}}) = T_0$, $dT/ds|_{\text{apex}} = 0$으로 한 번 적분한다. 차원 분석에서 $T_0 \propto (pL)^{1/3}$ 스케일링이 나오며, 수치 계수 1.4×10³은 표준 Λ(T)를 사용한 수치해에 맞춰 보정된다. 체적 가열률 스케일링 H ~ p^{7/6} L^{-5/6}는 T_0를 다시 대입하고 에너지 균형을 부과하여 얻는다.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1939 ── Grotrian/Edlén: coronium identified as hot Fe
            │  (first evidence corona is MK-hot plasma)
            │
1965 ── Giacconi: rocket X-ray imaging of Sun
            │  (first X-ray view of corona)
            │
1973 ── Vaiana et al.: Skylab morphology classification
            │  (BP, AR, LSS — the loop taxonomy)
            │
1978 ── ★ Rosner, Tucker & Vaiana: RTV scaling laws ★
            │  (theoretical anchor of loop physics)
            │
1988 ── Parker: nanoflare coronal heating
            │  (impulsive DC heating paradigm)
            │
1999 ── Nakariakov & TRACE: coronal seismology of kink modes
            │  (MHD wave diagnostics of loops)
            │
2002 ── Warren et al.: nanoflare-heated multi-strand loops
            │  (explains overdense warm loops)
            │
2006 ── Klimchuk: "Solving the coronal heating problem"
            │  (six-step framework)
            │
2008 ── Klimchuk, Patsourakos & Cargill: EBTEL 0-D model
            │  (computationally efficient multi-strand modeling)
            │
2010 ── Reale: Living Reviews (first edition, lrsp-2010-5)
            │
2013 ── Cirtain/Hi-C: 150 km strands with magnetic braiding
            │
★ 2014 ── THIS REVIEW: Reale 2014 (lrsp-2014-4) ★
            │  (updated synthesis, 557 refs)
            │
2018 ── Parker Solar Probe launched (in-situ near-Sun)
            │
2020 ── Solar Orbiter launched (EUI, SPICE)
            │
2026 ── (Current date) — MUSE/EUVST missions approaching
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **Rosner, Tucker & Vaiana (1978)** | Foundational RTV scaling laws that anchor Sec. 4.1.1 / 이 논문 Sec. 4.1.1의 근간 | The static equilibrium baseline against which all dynamic loop models are compared / 모든 동적 루프 모델 비교 기준 |
| **Parker (1988) "Nanoflares"** | DC heating paradigm in Sec. 4.4.1 / Sec. 4.4.1의 DC 가열 패러다임 | Introduces the impulsive magnetic-reconnection heating concept that dominates modern loop-physics debate / 현대 루프 물리 논쟁의 핵심 개념 도입 |
| **Klimchuk (2006) "Coronal heating revisited"** | Six-step heating framework adopted in Sec. 4.4 / Sec. 4.4에서 채택한 여섯 단계 틀 | Organizes the heating problem into tractable sub-questions / 가열 문제를 다루기 쉬운 하위 질문으로 정리 |
| **Aschwanden (2004) "Physics of the Solar Corona"** | Textbook counterpart cited throughout / 전반에 걸쳐 인용되는 교과서적 카운터파트 | Broader textbook treatment of corona; Reale's review is more focused on loops / 더 넓은 코로나 교과서; Reale은 루프에 초점 |
| **Klimchuk, Patsourakos & Cargill (2008) EBTEL** | 0-D modeling framework discussed in Sec. 4.1 / Sec. 4.1 논의된 0-D 모델링 | Computationally efficient alternative to full 1-D loop codes / 전체 1D 루프 코드의 효율적 대안 |
| **Nakariakov et al. (1999) TRACE kink oscillations** | Seismology of Sec. 3.5.2 / Sec. 3.5.2의 지진학 | Opens the era of coronal seismology for B-field measurement / B 필드 측정 코로나 지진학 시대 개척 |
| **Cirtain et al. (2013) Hi-C braiding** | Fine-structure evidence in Sec. 3.2.2 / Sec. 3.2.2의 미세구조 증거 | Pushes resolved loop cross-sections below 200 km, constraining multi-strand theories / 해상 단면을 200 km 이하로 밀어 multi-strand 이론 제약 |
| **De Moortel & Nakariakov (2012) waves review** | AC heating and wave observations / AC 가열과 파동 관측 | Complementary LRSP review on waves, cited for Sec. 3.5.2 and 4.4.2 / 파동에 대한 상보적 LRSP 리뷰 |
| **Hudson (1991) nanoflare power law** | Heating energy distribution / 가열 에너지 분포 | Established the α>2 criterion for nanoflare-dominated heating / nanoflare 지배 가열을 위한 α>2 기준 확립 |
| **Warren, Winebarger & Hamilton (2002)** | Multi-stranded loop modeling / multi-stranded 루프 모델링 | Demonstrated that nanoflare-heated strand ensembles explain overdense warm loops / nanoflare 가열 strand 앙상블이 overdense warm loop 설명 |
| **Antiochos et al. (1999)** | Prominence/condensation formation / 프라미넌스 형성 | Shows that footpoint-concentrated heating leads to catastrophic cooling / 발자국 집중 가열이 catastrophic cooling 유발 |
| **De Pontieu et al. (2011)** | Spicule-corona connection / 스피큘-코로나 연결 | Proposes chromosphere as loop-plasma source, challenging classical picture / 채층을 루프 플라스마 공급원으로 제시 |

---

## 7. References / 참고문헌

- Reale, F. (2014). "Coronal Loops: Observations and Modeling of Confined Plasma", *Living Rev. Solar Phys.*, **11**, 4. DOI: 10.12942/lrsp-2014-4
- Rosner, R., Tucker, W. H., & Vaiana, G. S. (1978). "Dynamics of the quiescent solar corona", *Astrophys. J.*, **220**, 643–665.
- Vaiana, G. S., Krieger, A. S., & Timothy, A. F. (1973). "Identification and analysis of structures in the corona from X-ray photography", *Solar Phys.*, **32**, 81–116.
- Parker, E. N. (1988). "Nanoflares and the solar X-ray corona", *Astrophys. J.*, **330**, 474–479.
- Klimchuk, J. A. (2006). "On solving the coronal heating problem", *Solar Phys.*, **234**, 41–77.
- Klimchuk, J. A., Patsourakos, S., & Cargill, P. J. (2008). "Highly Efficient Modeling of Dynamic Coronal Loops", *Astrophys. J.*, **682**, 1351–1362.
- Cargill, P. J., Bradshaw, S. J., & Klimchuk, J. A. (2012). "Enthalpy-Based Thermal Evolution of Loops. II. Improvements to the Model", *Astrophys. J.*, **752**, 161.
- Nakariakov, V. M., Ofman, L., DeLuca, E. E., Roberts, B., & Davila, J. M. (1999). "TRACE observation of damped coronal loop oscillations", *Science*, **285**, 862–864.
- Aschwanden, M. J., Fletcher, L., Schrijver, C. J., & Alexander, D. (1999). "Coronal Loop Oscillations Observed with TRACE", *Astrophys. J.*, **520**, 880–894.
- Reale, F. (2007). "Diagnostics of stellar flares from X-ray observations: from the decay to the rise phase", *Astron. Astrophys.*, **471**, 271–279.
- Serio, S., Reale, F., Jakimiec, J., Sylwester, B., & Sylwester, J. (1991). "Thermodynamics of cooling flaring coronal loops", *Astron. Astrophys.*, **241**, 197–202.
- Cirtain, J. W., Golub, L., Winebarger, A. R., et al. (2013). "Energy release in the solar corona from spatially resolved magnetic braids", *Nature*, **493**, 501–503.
- Warren, H. P., Winebarger, A. R., & Hamilton, P. S. (2002). "Hydrodynamic modeling of active region loops", *Astrophys. J. Lett.*, **579**, L41–L44.
- Cargill, P. J., & Klimchuk, J. A. (2004). "Nanoflare Heating of the Corona Revisited", *Astrophys. J.*, **605**, 911–920.
- Hudson, H. S. (1991). "Solar flares, microflares, nanoflares, and coronal heating", *Solar Phys.*, **133**, 357–369.
- Gudiksen, B. V., & Nordlund, Å. (2005). "An Ab Initio Approach to the Solar Coronal Heating Problem", *Astrophys. J.*, **618**, 1020–1030.
- Dere, K. P., Landi, E., Mason, H. E., et al. (1997). "CHIANTI – an atomic database for emission lines", *Astron. Astrophys. Suppl.*, **125**, 149–173.
- De Moortel, I., & Nakariakov, V. M. (2012). "Magnetohydrodynamic waves and coronal seismology: an overview", *Phil. Trans. R. Soc. A*, **370**, 3193–3216.
- Dadashi, N., Teriaca, L., & Solanki, S. K. (2011). "The quiet Sun average Doppler shift of coronal lines up to 2 MK", *Astron. Astrophys.*, **534**, A90.
- Sakamoto, Y., Tsuneta, S., & Vekstein, G. (2008). "A nanoflare heating model...", *Astrophys. J.*, **689**, 1421–1432.
- Aschwanden, M. J. (2004). *Physics of the Solar Corona: An Introduction*, Springer-Praxis.
- Spitzer, L. (1962). *Physics of Fully Ionized Gases*, Wiley. [thermal conductivity]
- Bradshaw, S. J., & Cargill, P. J. (2010). "The Cooling of Coronal Plasmas. III. Enthalpy Transfer as a Mechanism for Energy Loss", *Astrophys. J.*, **717**, 163–174.
- Tomczyk, S., McIntosh, S. W., Keil, S. L., et al. (2007). "Alfvén Waves in the Solar Corona", *Science*, **317**, 1192–1196.
- Favata, F., Flaccomio, E., Reale, F., et al. (2005). "Bright X-Ray Flares in Orion Young Stars", *Astrophys. J. Suppl.*, **160**, 469–502.
- Güdel, M. (2004). "X-ray astronomy of stellar coronae", *Astron. Astrophys. Rev.*, **12**, 71–237.
- Winebarger, A. R., Warren, H. P., & Mariska, J. T. (2003). "Transition Region and Coronal Explorer and Soft X-Ray Telescope Active Region Loop Observations: Comparisons with Static Solutions of the Hydrodynamic Equations", *Astrophys. J.*, **587**, 439–449.
- Orlando, S., Peres, G., & Reale, F. (1995). "Solar loops with stationary flows", *Astrophys. J.*, **455**, 718–725.
- Peres, G., Orlando, S., Reale, F., Rosner, R., & Hudson, H. (2000). "The Sun as an X-Ray Star. II.", *Astrophys. J.*, **528**, 537–551.
- De Pontieu, B., McIntosh, S. W., Carlsson, M., et al. (2011). "The Origins of Hot Plasma in the Solar Corona", *Science*, **331**, 55.
- Antiochos, S. K., MacNeice, P. J., Spicer, D. S., & Klimchuk, J. A. (1999). "The Dynamic Formation of Prominence Condensations", *Astrophys. J.*, **512**, 985–991.
- Chae, J., Yun, H. S., & Poland, A. I. (1998c). "Temperature Dependence of Ultraviolet Line Average Doppler Shifts in the Quiet Sun", *Astrophys. J. Suppl.*, **114**, 151.
- Serio, S., Peres, G., Vaiana, G. S., Golub, L., & Rosner, R. (1981). "Closed coronal structures. II.", *Astrophys. J.*, **243**, 288–305.
- Bray, R. J., Cram, L. E., Durrant, C., & Loughhead, R. E. (1991). *Plasma Loops in the Solar Corona*, Cambridge University Press.
- Withbroe, G. L., & Noyes, R. W. (1977). "Mass and energy flow in the solar chromosphere and corona", *Annu. Rev. Astron. Astrophys.*, **15**, 363–387.
- Ionson, J. A. (1978). "Resonant absorption of Alfvénic surface waves and the heating of solar coronal loops", *Astrophys. J.*, **226**, 650–673.
- Hollweg, J. V. (1984). "Resonances of coronal loops", *Astrophys. J.*, **277**, 392–403.
