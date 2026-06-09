---
title: "Pre-Reading Briefing: Solar Wind Electrons Alphas and Protons (SWEAP) Investigation"
paper_id: "54_kasper_2016"
topic: Solar_Observation
date: 2026-04-25
type: briefing
---

# Solar Wind Electrons Alphas and Protons (SWEAP) Investigation: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Kasper, J.C., Abiad, R., Austin, G., et al. (2016). "Solar Wind Electrons Alphas and Protons (SWEAP) Investigation: Design of the Solar Wind and Coronal Plasma Instrument Suite for Solar Probe Plus". *Space Science Reviews*, 204, 131–186. DOI: 10.1007/s11214-015-0206-3
**Author(s)**: Justin C. Kasper, Robert Abiad, Gerry Austin, Marianne Balat-Pichelin, Stuart D. Bale, John W. Belcher, et al. (large collaboration)
**Year**: 2016 (published online 29 October 2015)

---

## 1. 핵심 기여 / Core Contribution

이 논문은 NASA Solar Probe Plus (SPP, 후일 Parker Solar Probe로 개명) 미션에 탑재될 4-센서 SWEAP 측정기 묶음의 설계, 과학 목표, 예상 성능을 종합 정리한 PDR (Preliminary Design Review) 시점의 보고서이다. SWEAP은 (1) 태양을 직접 바라보는 Solar Probe Cup (SPC, Faraday Cup), (2) 램 방향 이온/전자용 정전 분석기 SPAN-A, (3) 반(反)램 방향 전자용 SPAN-B, (4) 전체를 제어하는 SWEAP Electronics Module (SWEM)으로 구성된다. SWEAP은 9.86 R☉ 근일점에서 코로나/태양풍 전이 영역의 양성자, 알파, 전자의 3차원 속도 분포 함수(VDF)를 직접 측정해 태양풍 가속·코로나 가열·에너지 입자 가속 메커니즘을 풀어내는 것을 목표로 한다.

This paper is the preliminary design review (PDR) snapshot of the SWEAP instrument suite — a four-sensor plasma package on NASA's Solar Probe Plus (later renamed Parker Solar Probe). SWEAP comprises (1) the Solar Probe Cup (SPC), a heat-shield-edge Faraday Cup that looks directly at the Sun, (2) SPAN-A, a combined ion/electron electrostatic analyzer on the ram side with a time-of-flight ion mass section, (3) SPAN-B, an electron ESA on the anti-ram side, and (4) the SWEAP Electronics Module (SWEM) that controls the suite. SWEAP is designed to deliver 3-D ion (proton, alpha, heavy) and electron velocity distribution functions inside 0.25 AU — eventually as close as 9.86 R☉ — to disentangle the mechanisms of solar wind acceleration, coronal heating, and energetic-particle acceleration in the inner heliosphere.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

태양풍의 존재는 1958년 Parker의 이론과 1960년대 Mariner 2의 관측으로 확립되었지만, 60년 동안 모든 in-situ 측정은 0.3 AU 바깥(주로 Helios 1, 2)에서만 이루어졌다. 코로나 가열의 정체와 태양풍 가속의 시작점은 미지 영역이었다. SPP는 처음으로 Alfvén critical surface 안쪽으로 진입해 태양풍이 형성되는 현장을 직접 sampling 하는 미션이며, SWEAP은 그 현장 plasma 분광계 역할을 한다. 본 논문은 2013년 10월 PDR을 통과한 시점의 설계 동결 문서로서, 후속 PSP 운영 논문들의 baseline reference로 작동한다.

The existence of the solar wind was established by Parker's 1958 theory and 1960s Mariner 2 observations, yet for 60 years every in-situ plasma measurement came from beyond 0.3 AU (mainly Helios 1, 2). The mystery of coronal heating and the launch of the solar wind remained inaccessible to direct sampling. SPP is the first mission to dive inside the Alfvén critical surface and measure the nascent solar wind in situ, with SWEAP serving as its plasma spectrometer. This paper is the design-frozen reference following SWEAP's October 2013 PDR and underpins the entire PSP plasma data archive that began in 2018.

### 타임라인 / Timeline

```
1958 ─ Parker, Solar wind theory
1962 ─ Mariner 2: first in-situ solar wind detection (Faraday cup heritage)
1974/76 ─ Helios 1, 2: closest in-situ measurements (0.29 AU)
1983 ─ Carlson et al., Top-hat ESA design
1995 ─ Wind launch (SWE Faraday Cup, key SWEAP heritage)
2007 ─ NRC Heliophysics Decadal Survey: prioritize Solar Probe
2013 Oct ─ SWEAP PDR (this paper's snapshot)
2014 Jan ─ SPP Mission PDR confirmed
2015 Oct ─ This paper published (Kasper et al. 2016)
2018 Aug ─ Parker Solar Probe launch
2018–24 ─ 24 progressively closer encounters; final perihelion 9.86 R☉
```

---

## 3. 필요한 배경 지식 / Prerequisites

- **Plasma physics fundamentals**: Maxwellian VDF, moments (n, V, T), Debye shielding, gyroradius, plasma beta. / 플라즈마 물리 기본: 맥스웰 분포, VDF 모멘트, 디바이 차폐, gyroradius, 플라즈마 베타.
- **Solar wind phenomenology**: fast vs slow streams, alpha/proton ratio, e-strahl, halo, core electrons, heliospheric current sheet. / 태양풍 현상학: 고속/저속 흐름, alpha-to-proton 비, e-strahl, halo, core 전자, 헬리오스피어 전류층.
- **Faraday Cup principle**: AC-modulated retarding-potential plate, synchronous detection, reduced distribution function (RDF). / Faraday Cup 원리: AC 변조 retarding-potential 판, 동기 검파, 축소 분포 함수.
- **Top-hat electrostatic analyzer**: Carlson 1983 hemispherical ESA, energy-per-charge selection, MCP detection. / Top-hat 정전 분석기: Carlson 1983 반구형 ESA, energy/charge 선택, MCP 검출.
- **Time-of-flight mass spectrometry**: pre-acceleration, carbon foil start/stop, m/q determination. / 비행시간 질량 분광: 가속, 카본 포일 start/stop, m/q 결정.
- **Heat-shield FOV geometry**: SPP TPS blocks anti-sun half-sky for instruments behind shield. / 열차폐 FOV 기하: SPP TPS는 차폐 뒤 측정기의 반(反)태양 반구 시야를 가린다.
- **Aberration**: at perihelion v_orbital ≈ 200 km/s tilts the apparent solar wind flow toward the ram side. / Aberration: 근일점 궤도속도 ~200 km/s가 보이는 태양풍 흐름 방향을 램 방향으로 기울인다.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **SPC (Solar Probe Cup)** | 열차폐 모서리에 장착되어 태양을 직접 보는 Faraday Cup. ±28° 반각 FOV, 50 eV–8 keV/q 이온, 50 eV–2 keV 전자, 4-quadrant collector. / Heat-shield-edge sun-staring Faraday Cup; ±28° half-angle FOV, 50 eV–8 keV/q ions, 50 eV–2 keV electrons, four-quadrant collectors. |
| **SPAN-A / SPAN-B** | 램/반-램 면의 top-hat ESA. SPAN-A는 ion ESA(+TOF) 와 e-ESA를 동봉; SPAN-B는 e-ESA 단독으로 anti-ram 면. 240°×120° FOV. / Ram-side / anti-ram-side top-hat ESAs; SPAN-A pairs an ion ESA (with TOF) with an e-ESA, SPAN-B is a stand-alone e-ESA. 240°×120° FOV each. |
| **SWEM** | SWEAP Electronics Module: 전원·CCSDS 텔레메트리·온보드 모멘트·압축·SPAN flash buffer (10×8 GB) 관리. / SWEAP Electronics Module managing power, CCSDS telemetry, on-board moments, compression and the 10×8 GB flash buffer for SPAN. |
| **RDF (Reduced Distribution Function)** | FC가 측정하는 1-D 함수: V_∥(=E/q || LOS) 축으로 적분한 분포. SPC의 1차 출력. / The 1-D distribution along the line-of-sight parallel velocity that a Faraday cup measures by sweeping its modulator window. |
| **HV Modulator Grid** | SPC 변조 전극: -2 kV~+8 kV DC + 50–800 V AC @ 1280 Hz 정사각/사인 변조; 두 전압 사이의 입자만 collector에 도달. / High-voltage modulator grid: -2 to +8 kVDC plus 50–800 V AC at 1280 Hz; only particles with E/q between the two extrema reach the collectors. |
| **Synchronous Detection** | AC 변조 주파수에서만 신호를 봄으로써 SEP 잡음·photoelectron·thermionic 전류를 차단. / Lock-in style detection at the modulator frequency that rejects DC noise sources (SEP penetration, photoelectrons, thermionic emission). |
| **Top-hat ESA** | Carlson(1983) 의 균일 360° 평면 응답을 갖는 반구형 정전 분석기. ΔR/R = 0.03, ~7% E/q 분해능. / Hemispherical analyzer with uniform 360° planar response (Carlson 1983); inter-hemisphere gap ΔR/R = 0.03 yields ~7% E/q resolution. |
| **MCP (Microchannel Plate)** | 단일 입자를 ~10⁶ 배 증폭하는 detector. SPAN: e-sensor는 chevron, ion-sensor는 Z-stack. / Particle multiplier with ~10⁶ gain; e-sensor uses chevron pair, ion-sensor uses Z-stack for larger pulses. |
| **Mechanical / Electrostatic Attenuator** | SPAN 동적 범위 확장: SMA 기반 1회 visor + spoiler 전압으로 outer hemisphere에 ~25% V_inner 인가, 각각 ×10 감쇠. / Two stages of dynamic range: a one-shot SMA mechanical visor (×10) plus a "spoiler" voltage on the outer hemisphere (×10). |
| **Strahl** | 자기력선 정렬된 전자 빔. e-VDF 가운데 ~2–3× core thermal speed에 위치하며 코로나 열속(heat flux)을 운반. / Field-aligned electron beam carrying coronal heat flux at ~2–3 thermal speeds. |
| **Aberration** | SPP 근일점 궤도속도(~200 km/s)가 빛의 시차처럼 흐름 방향을 ram 쪽으로 기울이는 효과. / Apparent rotation of the solar wind flow toward the ram side caused by the spacecraft's ~200 km/s orbital velocity at perihelion. |

---

## 5. 수식 미리보기 / Equations Preview

**(1) Reduced Distribution Function (RDF)** — Faraday Cup이 측정하는 1차 양:
$$ F(v_\| ) = \int\!\!\int f(\mathbf{v})\, dv_x\, dv_y \quad\text{with}\quad v_\| = \sqrt{2qE/m} $$
SPC가 modulator 창 [E_low, E_high] 사이에 있는 이온의 적분 전류를 측정하면 dF/dE에 비례한 신호가 얻어진다. / SPC measures ∫dF/dE over a finite E/q window; sweeping the window reconstructs F(v_∥).

**(2) Faraday Cup AC-Coupled Current** — 변조 전압 V(t) = V_DC + V_AC sin(ωt) 일 때 plate 전류는 ω에 동기된 AC 성분만 동기 검파로 추출된다.
$$ I_\text{AC}(t) \propto \frac{\partial F}{\partial E} \cdot V_\text{AC} \sin(\omega t) $$
이 방식이 thermal noise (SEP, photoemission)를 자연스럽게 제거. / Lock-in detection at ω rejects DC noise.

**(3) VDF Moments** — VDF f(v) 로부터 plasma parameter 계산:
$$ n = \int f\, d^3v,\quad \mathbf{V} = \frac{1}{n}\int \mathbf{v} f\, d^3v,\quad P_{ij} = m \int (v_i - V_i)(v_j - V_j) f\, d^3v $$
SPC: 4-collector 비율로 V_y, V_z 추정; SPAN: deflection×anode 합으로 3-D 적분. / SPC infers transverse flow from current ratios across 4 collectors; SPAN integrates over deflection × anode bins.

**(4) ESA Energy-Charge Relation** — top-hat 분석기:
$$ \frac{E}{q} = k_\text{ESA}\, V_\text{inner} $$
ΔR/R = 0.03 → ΔE/E ≈ 7%. / Inner-hemisphere voltage selects E/q with ~7% resolution.

**(5) Aberrated Flow at Closest Approach** — SPP 좌표계에서 보이는 흐름:
$$ \mathbf{V}_\text{obs} = \mathbf{V}_\text{SW} - \mathbf{V}_\text{SC} $$
9.86 R☉에서 V_SC ≈ 200 km/s가 흐름 방향을 ~25° 기울여 SPC + SPAN-A 결합 FOV가 필수. / At 9.86 R☉, the spacecraft's ~200 km/s orbital velocity tilts the apparent flow ~25°, requiring SPC + SPAN-A combined coverage.

---

## 6. 읽기 가이드 / Reading Guide

- **Section 1.1 (과학 목표 / science objectives)** — 세 개 overarching goals (Sources / Heating / Energetic Particles) 와 그 아래 sub-goal 별 측정 요구를 정독하라. Table 1 (slow vs fast vs transient), Table 2 (5 heating mechanisms), Table 3 (Level-1 requirements) 가 측정기 설계를 결정한다. / Read these sections carefully; Tables 1–3 set the design trade space.
- **Section 1.2 (FOV simulation)** — Fig. 8/9 의 Monte Carlo로 SPC + SPAN-A 결합 검출 비율(99–100%) 의 정당성을 보여줌. / Justifies the dual-instrument architecture.
- **Section 3 (SPC)** — Faraday Cup 원리 → AC modulator → 4 collector → thermal/electrical 설계. Fig. 14 (modulator window 개념), Fig. 17 (cross-section), Fig. 18 (energy window response) 가 핵심. / Crucial figures for understanding Faraday cup physics.
- **Section 4 (SPAN)** — top-hat ESA, 양/음 deflectors (±60°), TOF for ion mass, MCP, mechanical+electrostatic attenuator, 모드 (Coarse/Targeted/Alternating Sweep). Fig. 24 (광선 추적), Fig. 25 (deflection 응답), Fig. 26 (proton VDF 시뮬레이션) 이 핵심. / Key figures for ESA optics.
- **Section 5 (Operations & Data Products)** — SPC: 0.874 s @ 128 E×4I; SPAN: 0.437 s targeted; data levels L0–L4; Tables 5–9. / Data product hierarchy.
- **Tables 1, 3, 4, 8, 9** — 한 페이지로 압축된 측정기 사양의 보고. 반드시 책갈피. / Bookmark these summary tables.

---

## 7. 현대적 의의 / Modern Significance

- 이 PDR snapshot은 2018년 8월 Parker Solar Probe 발사 이후 모든 SWEAP 데이터(L2/L3) 의 기준 reference로 살아있다. PSP가 Alfvén surface를 처음 통과한 2021년 4월 발견(Kasper et al. 2021, PRL) 부터 switchback (Bale et al. 2019, Kasper et al. 2019), 알파-양성자 differential streaming, 코로나 가열 measurement 까지 — 모두 이 논문의 calibration·FOV·cadence 결정 위에서 가능했다.
- The PDR document remains the canonical reference for every SWEAP L2/L3 product after PSP's 2018 launch. Every major PSP discovery — sub-Alfvénic plasma (Kasper et al. 2021), magnetic switchbacks (Bale et al. 2019), differential streaming and ion heating in the corona — relies on the calibrations, FOV choices, and cadences established here.
- SPC의 Faraday-cup AC 동기 검파 + 4-quadrant flow-angle algorithm은 강한 SEP 환경에서도 안정적 흐름 측정을 가능케 했고, Wind/SWE 이래 가장 성공적인 FC 디자인이 되었다. / SPC's AC synchronous detection + four-quadrant flow-angle solution proved to be the most SEP-robust solar wind plasma measurement since Wind/SWE.
- SPAN top-hat ESA 와 ion-TOF는 MAVEN/STATIC heritage 위에서 완성되었으며, 미래 미션(Solar Orbiter SWA, IMAP, ESCAPADE) 의 직계 조상이다. / SPAN's top-hat + TOF heritage from MAVEN/STATIC continues into Solar Orbiter SWA, IMAP, and ESCAPADE.

---

## Q&A

(읽기 세션 중 추가됨 / Populated during reading session)
