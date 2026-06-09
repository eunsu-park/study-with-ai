---
title: "Pre-Reading Briefing: Integrated Science Investigation of the Sun (ISIS): Design of the Energetic Particle Investigation"
paper_id: "55_mccomas_2016"
topic: Solar_Observation
date: 2026-04-25
type: briefing
---

# Integrated Science Investigation of the Sun (ISIS): Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: McComas, D.J., Alexander, N., Angold, N., Bale, S., Beebe, C., Birdwell, B., et al., "Integrated Science Investigation of the Sun (ISIS): Design of the Energetic Particle Investigation", *Space Science Reviews* **204**, 187–256 (2016). DOI: 10.1007/s11214-014-0059-1
**Author(s)**: D.J. McComas (PI, SwRI) and 50+ co-authors from SwRI, JHU/APL, Caltech/JPL, GSFC, U. Arizona, U. Michigan, U. Delaware, UNH, UC Berkeley
**Year**: 2014 (online) / 2016 (volume)

---

## 1. 핵심 기여 / Core Contribution

ISIS (Integrated Science Investigation of the Sun) 는 NASA Solar Probe Plus(SPP, 후에 Parker Solar Probe로 개명) 임무에 탑재된 에너지 입자(energetic particle) 관측기 슈트로, 태양 표면으로부터 9 태양반경(R_S) 이내까지 진입하여 태양 코로나(corona)와 내부 헬리오스피어(inner heliosphere)에서 SEP(Solar Energetic Particle)의 기원(origin), 가속(acceleration), 수송(transport)을 in situ로 측정한다. ISIS 는 **EPI-Lo**(저에너지: 양성자 ~0.02–15 MeV/nuc, 전자 25–1000 keV)와 **EPI-Hi**(고에너지: 양성자 ~1–200 MeV/nuc, 전자 0.5–6 MeV)의 두 상보적 기기로 구성되며, 합쳐서 약 0.02 MeV/nuc 부터 200 MeV/nuc 까지 7 십수배(decades)의 에너지 범위와 거의 반구 시야각(half-sphere FOV)을 동시에 제공한다. 본 논문은 2014년 1월 SPP PDR(Preliminary Design Review) 시점의 ISIS 과학 목표, 측정 요구사항, 두 기기의 광학·전자·기계 설계, 보정 계획, 그리고 데이터 운영 흐름을 종합적으로 기술한다.

ISIS (Integrated Science Investigation of the Sun) is the energetic-particle instrument suite on NASA's Solar Probe Plus (SPP, later renamed Parker Solar Probe) mission, which dives to within 9 solar radii (R_S) of the Sun's surface to make in-situ measurements of the origin, acceleration, and transport of solar energetic particles (SEPs) in the corona and inner heliosphere. ISIS comprises two complementary instruments: **EPI-Lo** (ions ~0.02–15 MeV/nuc, electrons 25–1000 keV) — a time-of-flight (TOF) mass spectrometer with 80 apertures sampling nearly a full hemisphere — and **EPI-Hi** (ions ~1–200 MeV/nuc, electrons 0.5–6 MeV) — three SSD-based dE/dx vs. E telescopes (HET, LET1, LET2). Together they span more than seven decades in energy and resolve all major heavy-ion species and 3He/4He. This paper documents the ISIS science goals, performance requirements, mechanical/electrical/optical designs of both instruments, calibration plans, and the Science Operations Center (SOC) data architecture as of the SPP PDR in January 2014.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

1980년대 중반 이전까지 태양 플레어(solar flare)는 SEP의 거의 유일한 가속원으로 여겨졌으나, CME(Coronal Mass Ejection) 충격파(shock)가 발견된 이후 SEP는 두 가지 주요 가속 메커니즘 — (1) 충격파 가속(shock acceleration, 주로 큰 gradual SEP 사건), (2) 자기 재결합(magnetic reconnection)에 의한 플레어/제트 가속(주로 3He-rich impulsive SEP 사건) — 으로 이해되어 왔다(Reames 1999, 2013). 그러나 1 AU에서 관측되는 SEP는 가속·수송 효과가 혼합되어 있어 기원 분리가 어렵다. Helios(0.3–1 AU), STEREO/IMPACT(LET, HET), ACE/ULEIS, MESSENGER/EPS 등이 그간 부분적 관측을 제공했으나, 0.25 AU 이내의 코로나 가속 영역 내부 직접 측정은 SPP가 최초이다. ISIS는 STEREO LET/HET, JEDI(Juno), RBSPICE, MMS/EIS, MESSENGER/EPS, PEPSSI(New Horizons), Cassini/MIMI 의 누적 유산(heritage) 위에 설계되었다.

Before the mid-1980s, solar flares were thought to be the dominant SEP source. The discovery and recognition of CME-driven shocks then established two distinct accelerators: (1) diffusive shock acceleration (DSA) at CME shocks producing "gradual" SEP events, and (2) magnetic-reconnection-related processes in flares producing the smaller "impulsive" 3He-rich events (Reames 1999, 2013). Existing 1 AU data suffer from severe transport mixing — particles from multiple sources overlap by the time they reach Earth. SPP — and ISIS specifically — addresses this by performing the first direct, repeated in-situ sampling within ~10 R_S, where CME shocks are still fastest and self-generated Alfvén turbulence is most intense. ISIS designs heavily on the heritage of STEREO/IMPACT (LET, HET), MESSENGER/EPS, JEDI/Juno, RBSPICE/Van Allen Probes, MMS/EIS, PEPSSI/New Horizons, and Cassini/MIMI "hockey-puck" TOF spectrometers.

### 타임라인 / Timeline

```
1976  ┐ NOAA/GOES proton archive begins (used for SEP rate forecasts)
1977  │ Helios-1, IMP-8 SEP observations (Wibberenz & Cane 2006)
1985  │ Ellison & Ramaty DSA spectral form  j ∝ E^-γ exp(-E/E_0)
1995  │ Ulysses SWICS suprathermal pickup-ion spectra
1997  │ ACE / ULEIS launch — heavy-ion SEP composition reference
2006  │ STEREO A/B IMPACT (LET + HET) launched — direct heritage for EPI-Hi
2008  │ MESSENGER/EPS, New Horizons/PEPSSI heritage for EPI-Lo TOF
2012  │ Van Allen Probes / RBSPICE heritage for EPI-Lo electronics
2014  ▶ SPP PDR (January)  —  THIS PAPER captures ISIS design at PDR
2015  │ MMS / EIS launch — final heritage for EPI-Lo TOF
2018  │ Parker Solar Probe launches (12 Aug 2018)
2019+ │ First perihelia inside 35.7 R_S, then progressively to 9.86 R_S by 2024
```

---

## 3. 필요한 배경 지식 / Prerequisites

| Topic / 주제 | Why needed / 필요한 이유 |
|---|---|
| Time-of-Flight mass spectrometry / 비행시간 질량 분석 | EPI-Lo의 핵심 측정 원리 — TOF + E_SSD 로 m/q와 species 결정 / Core EPI-Lo principle: TOF + SSD energy gives mass and species |
| dE/dx vs. E technique / dE/dx-E 기법 | EPI-Hi의 입자 식별 원리 — Bethe-Bloch 손실에 따른 ΔE_thin·E_thick 트랙 / Particle ID via Bethe-Bloch energy-loss tracks in stacked SSDs |
| Diffusive Shock Acceleration / 확산 충격파 가속 | CME shock에서 SEP 가속의 1차 Fermi 메커니즘 / First-order Fermi mechanism for SEPs at CME shocks |
| Q/M 종속성 / charge-to-mass ratio scaling | E_0 ∝ (Q/M)^α 로 spectral break 결정 (Cohen et al. 2005, Li et al. 2009) |
| Microchannel Plate / 마이크로채널 플레이트 | EPI-Lo TOF의 전자 검출 — start/stop 신호 증폭 / Secondary-electron amplification for TOF start/stop |
| Solid-State Detector physics / 반도체 검출기 물리 | EPI-Hi 모든 측정의 기반; thickness uniformity, dead-layer, leakage current |
| Parker spiral magnetic field / 파커 나선 자기장 | ISIS FOV 설계 기준 (10° 이내) — 0.05/0.25/0.7 AU에서 spiral 각도 / FOV alignment basis |
| Suprathermal ion populations / 초열적 이온 | f(v) ∝ v^-5 quiet-time tail (Fisk & Gloeckler 2012) — SEP seed population |
| Bohr-Landau straggling / 보어-란다우 스트래글링 | LET 박막 검출기 통과 시 dE 분산 — 3He/4He 분리 한계 결정 |

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **SEP (Solar Energetic Particle)** | 태양에서 가속되어 헬리오스피어로 방출되는 keV–GeV 영역의 양성자·전자·중이온 / Particles (ions, electrons) accelerated at the Sun, keV–GeV |
| **EPI-Lo / EPI-Hi** | ISIS의 저에너지 / 고에너지 기기 — TOF 분광계 / dE-E 망원경 / ISIS low- and high-energy instruments |
| **TOF (Time of Flight)** | 두 박막 사이를 통과하는 시간을 측정하여 v를, SSD 에너지와 결합해 m을 결정 / Velocity from foil-to-foil traversal, mass from m = 2E/v² |
| **SSD (Solid-State Detector)** | Si 기반 입자 검출기 — pulse-height로 deposited energy 측정 / Silicon detector measuring deposited energy |
| **MCP (Microchannel Plate)** | 박막에서 방출된 secondary electron을 ~10⁶배 증폭 / Amplifier for secondary electrons from foils |
| **HET / LET1 / LET2** | EPI-Hi의 세 망원경 — High-Energy / 양단 LET / 단방향 LET / EPI-Hi's three telescopes |
| **dE/dx-E technique** | 얇은 검출기 ΔE × 두꺼운 검출기 E' 로 Z와 E 결정 (Bethe-Bloch) / Stopping-power vs. residual-energy ID |
| **Gradual / Impulsive SEP** | CME shock 기반 (수일 지속, 수소 풍부) / 플레어 reconnection 기반 (수시간, 3He·heavy-ion 풍부) |
| **DSA (Diffusive Shock Acceleration)** | 충격파 양측에서 입자가 자기-등방화 산란을 통해 1차 Fermi로 가속 / 1st-order Fermi at compressional shock |
| **Q/M (charge-to-mass ratio)** | spectral break 위치 E_0 ∝ (Q/M)^α (α ≈ 1.75 for Li et al. 2009) |
| **Suprathermal tail** | 1.5–20 × v_SW 의 power-law f(v) ∝ v^-5 — SEP seed pool / Quiet-time v^-5 ion tail |
| **PHASIC** | EPI-Hi의 dual-gain Pulse Height Analysis ASIC (16 channels, 11-bit Wilkinson ADC) |
| **TPS (Thermal Protection System)** | SPP의 carbon-carbon heat shield — ISIS 는 그 그림자(umbra)에 위치 |

---

## 5. 수식 미리보기 / Equations Preview

**1) TOF mass equation (EPI-Lo) / TOF 질량 방정식**:
$$m = 2E_{SSD}\,\left(\frac{t_{\text{TOF}}}{L}\right)^2$$
foil-to-SSD 거리 L (≈ 8.73 cm), TOF 시간 t (≈ 1–100 ns), SSD에 적층된 에너지 E_SSD 로부터 이온 질량 m을 결정. species는 (TOF, E)-평면의 곡선 트랙으로 분리됨 (Fig. 21).

**2) Bethe-Bloch energy loss (EPI-Hi dE/dx-E) / Bethe-Bloch 에너지 손실**:
$$-\frac{dE}{dx} \propto \frac{Z^2}{\beta^2}\,\ln\!\left(\frac{2 m_e c^2 \beta^2 \gamma^2}{I}\right)$$
얇은 ΔE 검출기에서의 에너지 손실은 ~Z²/β² 로 스케일되어 같은 E에서 Z가 큰 핵일수록 ΔE가 큼 → Fig. 32, 33의 element 분리 트랙 형성.

**3) Ellison-Ramaty DSA spectrum / Ellison-Ramaty 충격파 스펙트럼**:
$$j(E) = j_0\,E^{-\gamma}\exp\!\left(-\frac{E}{E_0}\right),\qquad E_0 \propto (Q/M)^{\alpha}$$
Mewaldt et al. 2005에서 γ ≈ 1.3, Li et al. 2009에서 α ≈ 1.75 (quasi-parallel strong shock). e-folding energy E_0가 종(species)에 따라 Q/M으로 분리되어 spectral break 만듬 (Fig. 9).

**4) Geometric factor / 기하학적 인자**:
$$R = G \cdot j$$
계수율 R [s⁻¹] = G [cm² sr] × differential intensity j [(cm² sr s MeV/nuc)⁻¹] × ΔE. EPI-Lo G ≥ 0.05 cm² sr (per pixel), EPI-Hi single telescope G ≈ 0.5 cm² sr.

**5) Suprathermal v⁻⁵ tail / 초열적 v⁻⁵ 꼬리**:
$$f(v) \propto v^{-5}\quad\Rightarrow\quad \frac{dN}{dE} \propto E^{-1.5}$$
quiet 태양풍 조건에서 보편적으로 관측되는 spectral index −1.5 (Fisk & Gloeckler 2012).

---

## 6. 읽기 가이드 / Reading Guide

| Section / 섹션 | 페이지 | 읽는 법 / How to read |
|---|---|---|
| §1 Introduction & Science | 1–14 | SEP의 두 종류, 왜 가까이 가야 하는지, 6개의 관측 가능 SEP science questions에 집중 / Focus on the two SEP sources, why we need close-in measurements, and the 6 driving science questions |
| §2 Suite Overview | 15–22 | ISIS 위치(TPS 그림자), FOV 맵(Fig. 18, 19), 자원 표(Table 3) / ISIS placement, FOV maps, resource table |
| §3 EPI-Lo | 22–34 | TOF 원리(Fig. 20), species separation(Fig. 21), 광학적 필터링(Pd/Al foil) / TOF principle, species track, optical filtering |
| §4 EPI-Hi | 34–50 | HET/LET 구조(Fig. 29, 30), Monte Carlo response(Fig. 32, 33), PHASIC ASIC(Fig. 37), dynamic threshold / Telescope geometry, MC responses, PHASIC, dynamic thresholds |
| §5 Operations & Data | 50–60 | 두 모드(Normal R≤0.25 AU, Low-rate R>0.25 AU), L0–L2 data products(Table 7) / Two modes and data product hierarchy |
| §6 Summary & §7 Acronyms | 60–63 | 빠른 요약과 약어 참조 / Concise summary + acronym lookup |

---

## 7. 현대적 의의 / Modern Significance

ISIS는 Parker Solar Probe(PSP, 2018-08-12 발사)의 핵심 입자 측정 슈트로서, 24개 perihelion(2024년 9.86 R_S까지)을 통해 SEP의 기원/가속/수송에 관한 첫 in-situ 데이터셋을 제공한다. PSP 시대의 주요 발견 — SEP seed population의 0.1 AU 변동, CME shock의 0.25 AU 이내 형성·소멸, 3He-rich micro-event의 빈도, suprathermal v⁻⁵ tail의 보편성 — 은 모두 ISIS 측정을 토대로 한다. 또한 ISIS의 설계는 후속 임무인 ESA Solar Orbiter EPD(2020), 향후 Heliophysics System Observatory의 입자 분광계(HelioSwarm, MUSE 등) 설계의 직접 참조점이 된다. EPI-Lo의 80-aperture 반구 sampling과 EPI-Hi의 5중 45° cone FOV는 향후 small-sat 분광계 설계에서도 빈번히 인용되는 표준 패턴이다.

ISIS is the cornerstone particle suite on Parker Solar Probe (PSP, launched 12 Aug 2018), providing the first in-situ dataset on SEP origins/acceleration/transport across 24 perihelia (down to 9.86 R_S in 2024). Major PSP-era discoveries — variability of the SEP seed population at 0.1 AU, CME-shock formation and dissipation inside 0.25 AU, abundance of 3He-rich micro-events, universality of the suprathermal v⁻⁵ tail — all rest on ISIS measurements. The instrument architecture (80-aperture hemispheric TOF for EPI-Lo, five overlapping 45° cones for EPI-Hi) has become a reference design for subsequent missions including ESA Solar Orbiter EPD (2020) and concept studies for HelioSwarm, MUSE, and small-sat particle spectrometers in the next decade.

---

## Q&A

(읽기 세션 중 추가됨 / Populated during reading session)
