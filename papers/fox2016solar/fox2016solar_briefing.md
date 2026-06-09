---
title: "Pre-Reading Briefing: The Solar Probe Plus Mission — Humanity's First Visit to Our Star"
paper_id: "40_fox_2016"
topic: Solar_Observation
date: 2026-04-25
type: briefing
---

# The Solar Probe Plus Mission: Humanity's First Visit to Our Star — Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Fox, N.J., Velli, M.C., Bale, S.D., Decker, R., Driesman, A., Howard, R.A., Kasper, J.C., Kinnison, J., Kusterer, M., Lario, D., Lockwood, M.K., McComas, D.J., Raouafi, N.E., Szabo, A., "The Solar Probe Plus Mission: Humanity's First Visit to Our Star," *Space Science Reviews* **204**, 7-48 (2016). DOI: 10.1007/s11214-015-0211-6
**Author(s)**: N.J. Fox et al. (14 authors, JHUAPL/JPL/UCB/NRL/UMich/SAO/SwRI/UTSA/GSFC)
**Year**: 2016 (mission renamed to **Parker Solar Probe** in May 2017)

---

## 1. 핵심 기여 / Core Contribution

이 논문은 NASA의 Solar Probe Plus(SPP, 후에 Parker Solar Probe로 개명) 미션의 과학 목표, 미션 설계, 우주선 구성을 종합적으로 제시한 **미션 개요 논문(mission overview)** 이다. SPP는 인류 역사상 최초로 **태양 코로나 내부(<10 R_S)** 까지 진입하여 *in-situ* 관측을 수행하는 우주선이다. 7년 동안 7번의 금성 중력 보조(Venus Gravity Assist, VGA)를 통해 근일점을 35.7 R_S에서 9.86 R_S(0.0459 AU, 약 690만 km)까지 단계적으로 낮추며, 24번의 태양 근접 통과(perihelion pass)를 수행한다. 세 가지 과학 목표는 (1) 코로나 가열과 태양풍 가속의 에너지 흐름 추적, (2) 태양풍 발원 영역의 플라즈마와 자기장 구조 결정, (3) 고에너지 입자(SEP)의 가속·수송 메커니즘 탐구이다.

This paper is the **mission overview** for NASA's Solar Probe Plus (SPP, later renamed Parker Solar Probe) — the first spacecraft to fly into the low solar corona (<10 R_S) and perform *in-situ* measurements inside the Alfvén critical surface. Over 7 years, seven Venus Gravity Assists (VGAs) walk down the perihelion from 35.7 R_S to 9.86 R_S (0.0459 AU), enabling 24 perihelion passes. Three overarching science objectives are: (1) trace energy flow that heats the corona and accelerates the solar wind; (2) determine structure and dynamics of the plasma and magnetic fields at solar wind sources; (3) explore mechanisms that accelerate and transport energetic particles. The paper also describes the four instrument suites (FIELDS, ISIS, SWEAP, WISPR), thermal protection system (TPS), and mission operations concept.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

태양 근접 미션의 개념은 1958년 NRC Simpson 위원회 보고서에서 처음 제안되었다. 50년 이상 "Solar Probe"는 헬리오피직스(heliophysics)의 최우선 미션 후보로 거론되어 왔으며, 2003·2013 NRC Decadal Survey에서 거듭 재확인되었다. 초기 설계(예: SP2005)는 **목성 중력 보조**를 사용하여 단 1-2회 ~4 R_S까지 극궤도로 접근하는 방식이었으나, 막대한 비용과 데이터 수집의 한계 때문에 2008년 STDT 보고서(McComas et al. 2008)에서 새로운 컨셉인 SPP(다중 금성 중력 보조 + 24궤도 + 9.86 R_S 황도면 근일점)로 전환되었다.

The concept of a near-Sun mission dates back to the 1958 NRC Simpson Committee Report. For over five decades, Solar Probe has been a top-priority heliophysics mission, reaffirmed in the 2003 and 2013 NRC Decadal Surveys. Earlier designs (e.g., SP2005) used a **Jupiter Gravity Assist** for one or two polar passes at ~4 R_S, but cost and limited data return led the 2008 STDT report (McComas et al. 2008) to redesign the mission as SPP — using multiple Venus gravity assists for 24 ecliptic-plane orbits with a 9.86 R_S perihelion. This trades a slightly larger closest approach distance for far more orbits and ~13× more time inside 30 R_S.

### 타임라인 / Timeline

```
1958 ─── Simpson Committee (NRC) recommends near-Sun mission
1962-2007 ─── Eight major science/engineering studies
2003 ─── NRC Decadal Survey: Solar Probe top priority
2005 ─── SP2005 design: Jupiter GA, 2 polar passes at 4 R_S
2008 ─── STDT (McComas) redesigns: SPP with 7 Venus GAs, 24 orbits
2010 ─── Four instrument suites selected (FIELDS, ISIS, SWEAP, WISPR)
2013 ─── NRC Decadal Survey reconfirms priority
2014 ─── Mission confirmed by NASA (Phase B)
2015 ─── This paper (received Dec 2014, accepted Oct 2015)
2017 ─── Renamed Parker Solar Probe (May 2017, honoring E.N. Parker)
2018 ─── Launched 12 Aug 2018 (Delta IV Heavy + STAR-48B)
2018 Nov ─── First perihelion at 35.7 R_S
2024 Dec ─── First minimum perihelion at 9.86 R_S (originally planned)
```

---

## 3. 필요한 배경 지식 / Prerequisites

- **Parker (1958) 태양풍 모델 / Parker's solar wind**: Supersonic outflow expanded from a hot corona; defines critical radius and Parker spiral.
- **Alfvén critical surface / 알펜 임계면**: Surface where solar wind speed equals Alfvén speed (V_A); located ~10-20 R_S. Below this surface, magnetic field "rigidly" connects plasma to the Sun (sub-Alfvénic).
- **Coronal heating problem / 코로나 가열 문제**: Why is the corona ~10^6 K when the photosphere is ~6000 K? Two main candidates: (a) wave/turbulence dissipation (Alfvén waves), (b) reconnection (nanoflares).
- **Solar Energetic Particles (SEPs)**: Two classes — impulsive (flare-related, ^3He-rich) and gradual (CME-shock-driven, GeV protons).
- **Gravity assist mechanics / 중력 보조 역학**: Patched-conic flyby that exchanges momentum with a planet; for SPP, Venus flybys *reduce* angular momentum to lower perihelion.
- **Orbital mechanics / 궤도역학**: Vis-viva equation, Hohmann-like elliptical orbits, energy and angular momentum conservation.
- **Thermal radiation / 열복사**: Stefan-Boltzmann law (P = σT⁴), inverse-square scaling of solar flux (475 Suns at 9.86 R_S).
- **Plasma physics**: Magnetic reconnection, MHD turbulence, gyroresonance, wave-particle interactions.
- **Heliospheric observations / 기존 관측**: Helios (0.3 AU min), Ulysses (polar orbit), STEREO, ACE, Wind, SOHO, MESSENGER.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **R_S (solar radius)** | 1 R_S ≈ 696,000 km; SPP closest approach is 9.86 R_S ≈ 6.86 × 10⁶ km / 태양 반지름 |
| **Alfvén critical surface** | V_solar_wind = V_A; below this surface plasma is magnetically connected to the Sun / 알펜 임계면 |
| **Venus Gravity Assist (VGA)** | Flyby that bends trajectory and reduces perihelion incrementally / 금성 중력 보조 |
| **TPS (Thermal Protection System)** | Carbon-carbon/carbon-foam sandwich heat shield, 2.3 m diameter / 열보호 시스템 |
| **475 Suns** | Solar irradiance at 9.86 R_S = 475 × 1366 W/m² ≈ 649 kW/m² / 태양 상수의 475배 |
| **Quasi-corotation interval** | Near 35 R_S, SPP angular speed ≈ Sun's rotation; samples one flux tube radially / 준동시회전 구간 |
| **FIELDS** | Electromagnetic fields investigation (fluxgate + search-coil mag, 5 antennas) |
| **ISIS** | Integrated Science Investigation of the Sun (EPI-Hi + EPI-Lo energetic particles) |
| **SWEAP** | Solar Wind Electrons Alphas and Protons (2 ESAs + Faraday Cup SPC) |
| **WISPR** | Wide-Field Imager for Solar Probe (white-light coronagraph) |
| **eVDF / pVDF** | Electron / proton Velocity Distribution Functions / 전자·양성자 속도 분포함수 |
| **CME / SEP** | Coronal Mass Ejection / Solar Energetic Particle |
| **C3** | Hyperbolic launch energy ~154 km²/s² (Delta IV Heavy class required) |

---

## 5. 수식 미리보기 / Equations Preview

### Solar irradiance scaling / 태양 복사 스케일링

$$
S(r) = S_\oplus \left(\frac{1\,\text{AU}}{r}\right)^2
$$

At r = 9.86 R_S = 0.0459 AU: S = 1366 × (1/0.0459)² ≈ 1366 × 475 ≈ 6.49 × 10⁵ W/m². / 9.86 R_S에서 태양 복사는 지구 대비 475배.

### Vis-viva (orbital speed) / 궤도 속도

$$
v^2 = G M_\odot \left(\frac{2}{r} - \frac{1}{a}\right)
$$

For perihelion 9.86 R_S, aphelion ~0.73 AU (a ≈ 0.39 AU): v_peri ≈ 195 km/s — fastest human-made object. / 인류가 만든 가장 빠른 물체.

### Stefan-Boltzmann balance for TPS / TPS 열평형

$$
\alpha \, S(r) = \varepsilon \, \sigma \, T^4
$$

With α/ε ≈ 0.6 (alumina coating) and S = 6.5 × 10⁵ W/m²: T_eq ≈ 1700 K (front face). / TPS 전면 ~1700 K, 후면 ~30°C 유지.

### Alfvén speed / 알펜 속도

$$
V_A = \frac{B}{\sqrt{\mu_0 \rho}}
$$

Critical surface: V_solar_wind(r_A) = V_A(r_A). Below r_A, sub-Alfvénic. / 임계 반경 미만은 sub-Alfvénic.

### Parker spiral / 파커 나선

$$
B_\phi / B_r = -\Omega_\odot \, r / v_{sw}
$$

At SPP perihelion, very small angle — nearly radial field. / 근일점에서 거의 완전 방사형 자기장.

---

## 6. 읽기 가이드 / Reading Guide

1. **Section 1-2 (Introduction & Science Overview)**: Read carefully. Three science objectives are the spine of the entire paper. Each subsection (2.1, 2.2, 2.3) develops one objective with figures from prior missions (Helios, Ulysses, Wind, ACE).
2. **Section 3 (Observations & Requirements)**: Skim Table 1 — it's a matrix of objectives × measurements × instrument types. Useful as reference, not narrative.
3. **Section 4 (Mission Design)**: This is the engineering heart. Pay attention to Figs. 13-14 (trajectory, walkdown), Table 3 (orbit time), and the TPS description (Sec. 4.2-4.3).
4. **Section 5 (Science Operations)**: Skim — describes data flow between SOC/MOC. Useful for context on encounter vs. cruise phase.
5. **Figures to study**: Fig. 1 (Ulysses bimodal wind), Fig. 12 (sub-Alfvénic region map), Fig. 13 (orbit geometry), Fig. 14 (perihelion walkdown), Figs. 15-16 (spacecraft layout).

읽기 순서 추천: Abstract → Sec. 1 → Box 1 → Sec. 2 (objectives 1-3) → Sec. 4.1-4.3 (engineering) → Tables 1-3.

---

## 7. 현대적 의의 / Modern Significance

이 논문은 미션 발사(2018년 8월) 직전의 "blueprint" 문서이다. PSP는 2021년 이미 Alfvén 임계면을 통과했고, "switchback"(자기장 S자형 굴곡), 먼지없는 영역(dust-free zone), 그리고 9.86 R_S에서의 직접 in-situ 관측을 달성했다. 이 논문에서 제기한 세 가지 과학 질문은 PSP의 모든 후속 발견(Bale et al. 2019, Kasper et al. 2019, Howard et al. 2019, Bale et al. 2023)의 기본 틀이 되었다. 동시에 ESA의 Solar Orbiter(2020 발사)와의 시너지를 명시적으로 언급한 점도 주목할 만하다.

This paper is the *blueprint* document right before launch. PSP launched 12 Aug 2018, crossed the Alfvén critical surface in 2021 (Kasper et al. 2021), discovered magnetic "switchbacks" (Bale et al. 2019), confirmed a dust-free zone, and reached 9.86 R_S as planned. The three science objectives framed every PSP discovery paper. The paper also foreshadows the synergy with ESA's Solar Orbiter (launched 2020) which provides remote-sensing context. Reading this 2016 mission overview alongside post-launch results is a textbook case of how mission-design choices (24 orbits vs. 2 polar passes, 9.86 R_S vs. 4 R_S) translate into different scientific harvests.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
