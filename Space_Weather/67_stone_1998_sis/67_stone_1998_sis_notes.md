---
title: "The Solar Isotope Spectrometer for the Advanced Composition Explorer"
authors: ["E. C. Stone", "C. M. S. Cohen", "W. R. Cook", "A. C. Cummings", "B. Gauld", "B. Kecman", "R. A. Leske", "R. A. Mewaldt", "M. R. Thayer", "B. L. Dougherty", "R. L. Grumm", "B. D. Milliken", "R. G. Radocinski", "M. E. Wiedenbeck", "E. R. Christian", "S. Shuman", "T. T. von Rosenvinge"]
year: 1998
journal: "Space Science Reviews"
doi: "10.1023/A:1005027929871"
topic: Space_Weather
tags: [SEP, ACR, GCR, isotope, spectrometer, ACE, silicon_detector, dE_dx_E_technique, mass_resolution, instrument_paper]
status: completed
date_started: 2026-04-25
date_completed: 2026-04-25
---

# 67. The Solar Isotope Spectrometer for the Advanced Composition Explorer / ACE 위성 탑재 태양 동위원소 분광기

---

## 1. Core Contribution / 핵심 기여

본 논문은 ACE (Advanced Composition Explorer) 위성의 핵심 기기 중 하나인 SIS (Solar Isotope Spectrometer)의 설계, 제작, 보정 및 첫 운용 결과를 종합 보고하는 instrument paper이다. SIS는 He(Z=2)에서 Zn(Z=30)에 이르는 고에너지 입자(~10–100 MeV/nucleon)의 동위원소 조성을 ~0.15–0.35 amu r.m.s. 질량 분해능으로 측정하기 위해 정밀하게 설계된 두 개의 동일한 silicon solid-state 검출기 망원경이다. 각 망원경은 17개의 실리콘 검출기 — 75 µm 두께의 양면 위치감응 매트릭스 검출기(M1, M2) 한 쌍과 100 µm ~ 3.75 mm 두께로 점진적으로 두꺼워지는 stack 검출기(T1–T8) — 로 구성되어, dE/dx 다중 측정과 trajectory 재구성을 결합한 정밀 핵종 식별을 수행한다. 38.4 cm² sr의 기하학적 인자는 기존 SEP isotope 분광기들 대비 압도적으로 커서, 큰 gradual SEP 이벤트에서 ¹³C, ¹⁵N, ¹⁸O, ²²Ne, ²⁵Mg, ²⁶Mg, ²⁹Si, ³⁰Si, ³⁴S, ⁵⁴Fe, ⁵⁸Ni 등 희귀 동위원소를 통계적으로 의미 있는 수로 측정할 수 있다.

This paper is a comprehensive instrument description of the Solar Isotope Spectrometer (SIS), one of nine instruments aboard NASA's Advanced Composition Explorer (ACE), launched 25 August 1997 to the L1 Lagrange point. SIS measures the isotopic composition of energetic nuclei from helium (Z = 2) to zinc (Z = 30) over ~10 to ~100 MeV nucleon⁻¹ with mass resolution σ ≈ 0.15 amu (oxygen) to σ ≈ 0.35 amu (iron). The instrument consists of two identical telescopes, each comprising 17 silicon detectors: a pair of 75-µm thick, two-dimensional position-sensitive matrix detectors (M1, M2) separated by 6 cm forming a hodoscope, and an energy-loss/range stack of 15 single-electrode detectors (T1–T8) of graduated thickness from 100 µm to 3.75 mm. Particle identification combines multiple dE/dx measurements with residual-energy and trajectory data using a refined version of the standard dE/dx–E technique. The geometry factor of 38.4 cm² sr — substantially larger than any earlier SEP isotope spectrometer (e.g., MAST on SAMPEX) — together with custom 16-channel VLSI pulse-height analyzers and on-board prioritized event buffers, allow SIS to harvest rare isotopes (¹³C, ¹⁵N, ¹⁸O, ²²Ne, ²⁵Mg, ²⁶Mg, ²⁹Si, ³⁰Si, ³⁴S, ⁵⁴Fe, ⁵⁸Ni) with statistically meaningful counts during large gradual SEP events, anomalous cosmic-ray fluxes during solar minimum, and low-energy galactic cosmic-ray observations.

---

## 2. Reading Notes / 읽기 노트

### Part I: Science Motivation (Section 2) / 과학적 동기

#### 2.1 SEP Isotopic Composition / 태양고에너지입자(SEP) 동위원소 조성

논문은 태양 물질 표본(sample)의 직접 동위원소 측정이 천문학적으로 매우 어려움에 주목하며, SEP가 태양 코로나의 직접 표본을 제공한다는 점을 강조한다. SEP은 두 부류로 분류된다 — gradual events (수일 ~ 수주 지속, CME 충격파 가속, 코로나 온도 ~2×10⁶ K에 해당하는 Fe⁺¹⁵ 전하 상태)와 impulsive events (플레어 가속, ³He-rich, Fe⁺²⁰ → ~10⁷ K). Reames (1995) 분류에 따르면 ³He/⁴He < 0.1인 이벤트들의 Fe/O 비율이 ~3배 코로나 값보다 낮은 반면, ³He-rich 이벤트는 코로나 대비 Fe/O가 ~10배 증가한다. SIS의 첫 번째 목표는 large gradual events에서 코로나 isotope 조성을 직접 측정하는 것이며, Figure 6은 ¹³C/¹²C, ¹⁵N/¹⁴N, ¹⁸O/¹⁶O, ²²Ne/²⁰Ne, ²⁶Mg/²⁴Mg 등 기존 측정과 큰 불확실성을 시각화한다. Figure 7은 1992년 10월 30일 이벤트 규모로 SIS가 측정할 수 있는 입자 수를 추정 — ⁵⁶Fe ~10⁵, ⁶⁴Zn ~10 events.

The Sun contains the bulk of solar-system matter, yet direct knowledge of solar isotopic composition is sparse. Most "solar system" isotopic abundances (Anders & Grevesse 1989) are actually measured in meteorites or terrestrial samples, with only a handful of solar isotope spectroscopic measurements available. SEPs provide direct samples of solar coronal material accelerated to high energies. SEPs fall into two classes: **gradual events** (long rise times, days–weeks duration, charge states characteristic of T ≈ 2×10⁶ K such as Fe⁺¹⁵, accelerated by CME-driven shocks) and **impulsive events** (smaller, shorter-lived, with enhanced ³He, Fe, and electrons; charge states such as Fe⁺²⁰ implying T ≈ 10⁷ K, indicating flare-heated plasma). Figure 5 shows that gradual events match coronal abundances within ~25%, whereas impulsive events show Q/M-dependent enhancements of 5–10× for heavy elements. SIS's primary SEP science goal is to harvest direct isotope ratios in large gradual events. Figure 7 estimates that for the 30 October 1992 event scale, SIS would have measured ~10⁶ ¹⁶O nuclei, ~10⁵ ⁵⁶Fe nuclei, and ~10 ⁶⁴Zn nuclei — orders of magnitude more than previous instruments.

#### 2.2 Anomalous Cosmic-Ray Composition / 비정상 우주선 조성

ACR은 LISM(local interstellar medium)에서 들어온 중성 입자가 태양풍/UV에 의해 이온화되어 termination shock에서 가속된 단일 이온화 입자 (Fisk-Kozlovsky-Ramaty 1974, Pesses et al. 1981)이다. 일곱 개 원소(H, He, C, N, O, Ne, Ar)에서 5 ~ 25 MeV/nucl 영역의 anomalous flux 증가가 관측되었다. SAMPEX/HILT 측정(Klecker et al. 1995)은 ACR이 단일이온화임을 확인했고, 이로부터 가속 후 통과한 물질량은 < 1 µg/cm²로 제한된다. SIS의 둘째 목표는 ACR의 ²²Ne/²⁰Ne (Neon-A vs Neon-B 식별), ¹⁸O/¹⁶O, ¹⁵N/¹⁴N 등을 측정해 LISM의 동위원소 조성을 결정하는 것이다. 솔라 미니멈 1년간 예상 카운트: ~50,000 ACR oxygen, ~8000 N, ~2500 Ne. Figure 9은 ²²Ne/²⁰Ne 비를 GCRS, Neon-A (0.12), Neon-B (0.07) 시나리오와 비교한다.

ACRs are a singly-ionized component originating from interstellar neutral atoms that drift into the heliosphere, become ionized by solar UV or charge exchange with the solar wind, are convected to the termination shock, and accelerated to >10 MeV/nucl. Seven elements (H, He, C, N, O, Ne, Ar) show anomalous spectra during solar minimum. SAMPEX/HILT (Klecker et al. 1995) confirmed Q ≈ +1 for the bulk of ACR N, O, Ne below 20 MeV/nucl, although a fraction become multiply charged at higher energies (Mewaldt et al. 1996). SIS's ACR science includes (a) measurement of isotopic ratios of ACR nuclei to constrain the LISM composition, (b) discrimination between Neon-A (0.12) and Neon-B (0.07) for ²²Ne/²⁰Ne, and (c) search for ACR contributions to S, Si, Fe at the 10⁻⁴ × O level. Estimated yields per solar-minimum year: ~50,000 ACR O, ~8000 ACR N, ~2500 ACR Ne.

#### 2.3 Galactic Cosmic-Ray Isotopes / 은하 우주선 동위원소

CRIS (Cosmic Ray Isotope Spectrometer)이 ~100 ~ 500 MeV/nucl의 GCR isotope를 담당하며 SIS는 그 아래 영역을 보완한다. ²²Ne/²⁰Ne는 GCR ~0.6, Neon-A/B ~0.10–0.12로 GCR 소스 물질이 Wolf-Rayet star 같은 ²²Ne 풍부 소스를 포함함을 시사한다. ²⁹Si/²⁸Si, ³⁰Si/²⁸Si 측정은 GCR source의 Si 동위원소 비를 결정한다. ~50 MeV/nucl 아래에서는 ACR이 우세해지며 ²²Ne/²⁰Ne 비가 급감한다(Fig. 9 right panel).

CRIS covers GCR isotopes from ~100 to ~500 MeV/nucl with G > 200 cm² sr; SIS extends measurements to lower energies (~10–~100 MeV/nucl). The ²²Ne/²⁰Ne GCR ratio of ~0.6 (corrected to ~0.27–0.5 source ratio) is anomalously high relative to Neon-A or solar wind, requiring contributions from Wolf-Rayet stars or core helium burning (Cassé & Paul 1982; Prantzos et al. 1987). Below ~50 MeV/nucl the ACR component dominates and the measured ²²Ne/²⁰Ne ratio drops sharply.

### Part II: Instrument Approach (Section 3.1, 3.2) / 계측 접근법

#### Particle Identification Principle / 입자 식별 원리

핵심 식별 원리는 표준 dE/dx–E 기법의 정제된 형태이다. 전하 Z, 질량 M, 속도 v인 입자의 운동에너지는 E ∝ Mv², 에너지 손실율은 dE/dx ∝ Z²/v². 두 양을 곱하면

$$ (dE/dx) \cdot E \propto Z^2 M $$

이 속도와 무관한 양이 된다. dE/dx vs E 평면에서 각 (Z, M) 핵종은 독립적인 hyperbola를 그린다. ΔE/Δx ≈ ΔE/(0.10 g/cm² Si)로 근사하고, 잔여 에너지 E'는 다음 stack 검출기에서 측정된다. 인접 원소 간 간격은 대략 ∝ Z이고, 동일 원소 내 동위원소 간 간격은 원소 간 간격의 ~1/8이므로, 동위원소가 8개 미만이라면 모호함 없이 식별 가능하다 — SIS 대상 모든 원소가 이 조건을 만족한다.

The particle ID exploits the fact that for a non-relativistic ion, kinetic energy E ∝ Mv² and energy-loss rate dE/dx ∝ Z²/v² (Bethe-Bloch). Their product (dE/dx) E ∝ Z²M is velocity-independent. In the dE/dx–E plane each (Z, M) species traces a hyperbola, with element spacing scaling like Z and isotope spacing within an element ~1/8 of element spacing. As long as an element has fewer than 8 isotopes (true for all SIS target elements), particle ID is unambiguous. Figure 10 (GSI calibration) demonstrates well-resolved B, C, N, O bands with the inset showing ¹²C, ¹³C, ¹⁴C tracks.

#### Improved Particle ID Equation / 개선된 식별 방정식

해상도 개선은 ΔE 검출기 두께 L이 입자 range의 상당 비율일 때 가능하다. 입사각 θ로 통과한 입자의 range 변화를 두께로 등치:

$$ R_{Z,M}(E/M) - R_{Z,M}(E'/M) = L \sec\theta \quad (\text{Eq. 2}) $$

여기서 R_{Z,M}(E/M)은 Hubert et al. (1990) 같은 표에서 얻는다. 측정량은 ΔE = E – E', E', L (~0.1% 정확도로 보정됨), θ. 먼저 mass-to-charge ratio M/Z를 가정하고 Z에 대해 풀어 정수 charge를 얻은 후, 그 Z로 M을 푼다. 이 implicit equation은 Appendix A에서 power-law approximation R_{Z,M}(E/M) ≃ k M/Z² (E/M)^a를 통해 explicit form으로 변환된다.

Resolution is significantly improved when L is a substantial fraction of the particle range. For a particle at incidence angle θ, the change in range across the ΔE detector equals its slant thickness:

$$ R_{Z,M}(E/M) - R_{Z,M}(E'/M) = L\sec\theta \tag{2} $$

Solving (2) is iterative: assume mass-to-charge ratio M/Z, solve for Z (which lands on a clean integer histogram), then with Z fixed, solve for M. Using a power-law approximation R_{Z,M}(E/M) ≃ k M/Z² (E/M)^a (with a ≈ 1.7 from Hubert et al. 1990) one obtains an explicit form (Equation A2) used iteratively. Dead-layer corrections substitute ΔE → ΔE + δE, L → L + l.

#### Design Requirements / 설계 요구사항

- 동위원소 분해능 ≤ 0.25 amu (인접 동위원소가 ~100배 abundance 차이 → Stone 1973 기준).
- 입자 trajectory 각도분해능 ≈ 0.2°, 검출기 두께 정확도 ~0.1%.
- 큰 SEP에서 양성자 flux > 10⁵ cm⁻²s⁻¹sr⁻¹에서도 Z ≥ 10 이온을 효과적으로 트리거할 수 있어야 함.
- 5–25 MeV/nucl ACR을 위해 thin entry detector + 큰 geometry factor + low threshold.
- 100 MeV/nucl 이상 SEP를 위해 두꺼운 stop detector 필요 (T7 = 3.75 mm).
- Mass resolution σ_M에 기여하는 항 — energy-loss straggling, multiple scattering, electron pickup/loss, trajectory uncertainty, detector thickness uniformity, electronics noise — 모두 partial derivatives로 관리.

Mass resolution ≤ 0.25 amu requires (a) thickness uniformity ≤ 0.1%, (b) angular resolution ≤ ~0.2°, (c) full charge collection at biases ≥ 20 V above depletion, (d) electronics noise < 100 keV r.m.s., and (e) dynamic range ≥ 1000 for elements He–Zn. The instrument must remain usable when proton fluxes exceed 10⁵ cm⁻² s⁻¹ sr⁻¹ during large SEPs without missing rare heavy nuclei (Z ≥ 10). Two telescope designs allow ~10–~100 MeV/nucl coverage with thin top detectors for low-energy ACRs and thick stops (T7 = 3.75 mm of Si) for highest-energy SEPs.

### Part III: Detector Hardware (Sections 3.3–3.5) / 검출기 하드웨어

#### Telescope Layout / 망원경 구조

각 망원경은 17개 Si 검출기로 구성:

| Position | Name | Thickness | Active Area | Geom. Factor |
|---|---|---|---|---|
| Matrix 1 | M1 | 0.075 mm | 34 cm² | – |
| Matrix 2 | M2 | 0.075 mm | 34 cm² | 38.4 cm² sr |
| Stack 1 | T1 | 0.1 mm | 65 cm² | 38.4 cm² sr |
| Stack 2 | T2 | 0.1 mm | 65 cm² | 37.2 cm² sr |
| Stack 3 | T3 | 0.25 mm | 65 cm² | 36.6 cm² sr |
| Stack 4 | T4 | 0.5 mm | 65 cm² | 34.4 cm² sr |
| Stack 5 | T5 | 0.75 mm | 65 cm² | 33.0 cm² sr |
| Stack 6 | T6 | 2.65 mm (3 wafers) | 65 cm² | 27.2 cm² sr |
| Stack 7 | T7 | 3.75 mm (6 wafers) | 65 cm² | 19.4 cm² sr |
| Stack 8 | T8 | 1.0 mm | 65 cm² | 19.4 cm² sr (anti) |

총 ~8.25 mm of Si stopping power. M1과 M2는 6 cm 간격으로 trajectory 재구성. Opening angle 95° full (Figure 11). 입사창 Kapton 3장 두께는 ~31 µm Si-equivalent. T6는 3개 wafers (1.0, 0.9, 0.75 mm), T7은 6개 wafers (1.0, 0.75, 0.5, 0.5, 0.5, 0.5 mm)을 합친 composite device.

Each telescope contains 17 high-purity Si detectors (Table II): two 75-µm position-sensitive matrix detectors (M1, M2), and a stack of T1–T8 with thicknesses 0.1, 0.1, 0.25, 0.5, 0.75, 2.65 (composite), 3.75 (composite), 1.0 mm. Stack detectors have 65 cm² active area (46 mm radius circles flattened on one side over a 42 mm chord). All detectors are ion-implanted n-type ⟨111⟩ float-zone Si manufactured by Micron Semiconductor Ltd. The crystal axes are aligned to enable channelling recognition. T6 and T7 are composite devices summing outputs of three and six wafers respectively. Total stopping thickness: ~8.25 mm of Si.

#### Matrix Detector Trajectory System / 매트릭스 검출기 trajectory 시스템

M1, M2는 octagonal (~75 µm 두께, 34 cm² active area) 검출기로 양면이 64개 strip으로 분할된다. Strip 폭 0.96 mm, 간격 0.040 mm, 양면이 직교(orthogonal)이므로 X, Y 좌표 동시 측정. 총 strip 수는 망원경당 4 surfaces × 64 = 256, SIS 전체로는 512. 각 strip은 독립 12-bit Wilkinson PHA로 펄스 높이 분석. 위치 분해능 ~0.29 mm, 6 cm 분리에서 각도 분해능 ~0.25° r.m.s. (모든 각도 평균). G-10 mount, 실리콘 레진(KJR-9022E)으로 본딩, Kapton flex-strip으로 고밀도 connector를 통해 PHA 보드에 연결.

The matrix detectors (M1, M2) are octagonal silicon ion-implanted detectors with 75 µm thickness and 34 cm² active area. Each face is divided into 64 metalized strips (0.96 mm wide, 0.040 mm gap), with the X-side and Y-side strips orthogonal. Each strip is read out by an independent 12-bit Wilkinson PHA (256 PHAs per telescope, 512 total). With M1 and M2 separated by 6 cm and position resolution ~0.29 mm, the angular resolution is ~0.25° r.m.s. averaged over all incidence angles. The matrix detectors are bonded into G-10 mounts with silicone resin (Shin-etsu KJR-9022E), with redundant aluminum wirebonds to gold-plated copper pads on the mount. Flex-circuit Kapton "flex-strips" carry the 128 strip signals to high-density connectors mating with the matrix PHA boards.

#### Stack Detectors / 스택 검출기

T1–T8은 single-electrode Si detectors, biased ≥ 20 V above depletion to ensure full charge collection. Noise level 100–400 keV r.m.s.. Dead layers are uniform within ±0.05 µm, total 0.1–0.6 µm. T1과 T2 모듈은 함께 묶이고, T3+T4, T5+T6, T7+T8 모듈도 짝지어 각 telescope당 6개 모듈 (4 stack + 2 matrix). Detector thickness mapped using dual-laser interferometer (Milliken et al. 1995) for ≤ 0.1% uniformity over 65 cm² area.

The stack T1–T8 detectors are single-electrode wafers fabricated by Micron Semiconductor Ltd. and mounted in custom G-10 modules. Each is biased at least 20 V above depletion. Noise: 100–400 keV r.m.s. Detector thickness must be known to ≤ 0.1% over the active area to achieve ≤ 0.1 amu mass error for Fe-group nuclei. Thickness mapping was done by a dual-laser interferometer (Milliken et al. 1995), supplemented by 870 MeV ³⁶Ar accelerator scans. Maps revealed dead layers ≤ 0.6 µm, uniform to ±0.05 µm.

### Part IV: Electronics, Logic, and Onboard Processing (Sections 3.6–3.8) / 전자부, 동시계수논리, 처리

#### Custom VLSI Pulse-Height Analyzers / 맞춤형 VLSI PHA

매트릭스 검출기 신호는 SIS-전용 CMOS VLSI 칩 (UTMC, 1.2 µm radiation-hard process) 으로 처리. 각 칩에 16 PHA 채널 (총 32 chips/SIS). Stack 검출기는 16개 별도 PHA hybrid (Teledyne, custom bipolar ASIC) 사용. 핵심 사양:

| Parameter | Matrix VLSI | Stack hybrid |
|---|---|---|
| PHAs/chip | 16 | 1 |
| Power/channel | 13 mW | ~40 mW |
| Full scale | 31 pC (700 MeV Si) | 12 pC ~ 12000 MeV |
| Dynamic range | 1400:1 | 2000:1 |
| ADC | 12-bit Wilkinson | 12-bit Wilkinson |
| Dead time | ~6 + 0.125 N µs | ~256 µs max |
| Gain stability | < ±20 ppm/°C | 20 ppm/°C |
| Radiation tolerance | ~100 kRads | – |

Charge-sensitive amplifier (CSA) → buffer → switched hold capacitor (C1, C2, C3) → comparator + Wilkinson 12-bit grey counter. Pedestal voltage applied via C4 ensures non-zero rundown for small inputs. Leakage compensation uses 8-bit IDAC on each channel; auto-balanced every 512 s to within ~2 nA. PHA gain calibration uses heavy-ion data: same particle deposits same energy in front+back strip pair, so strip-pair ratios give all 128 strip gains relative to a reference.

Matrix detector strip signals are processed by 16-channel custom CMOS VLSI ASICs (32 per SIS) fabricated in UTMC's radiation-hard 1.2 µm process. Each ASIC contains 16 charge-sensitive amplifiers (CSA), buffer/sample-and-hold/Wilkinson 12-bit ADC chains. Power consumption is 13 mW/channel, with dynamic range 1400:1 (~100 keV r.m.s. noise excluding quantization), gain stability < ±20 ppm/°C, radiation tolerance ~100 kRads. Each strip has its own current-output digital-to-analog converter (IDAC) for leakage current cancellation, balanced every 512 s. Stack detector signals use a separate Teledyne-fabricated bipolar PHA hybrid (16 per SIS, one per stack detector) with custom application-specific IC at Harris Semiconductor. Hybrid noise: 100 keV r.m.s.; gain stability 20 ppm/°C; offset variation < 0.5 channels over –20 °C to +40 °C.

#### Coincidence Logic and Trigger Modes / 동시계수논리와 트리거 모드

Table IV에 제시된 6개의 별도 동시계수 등식 (Z=1, Z=2, Z≥3 각 등급별 multiple modes):

```
Z ≥ 3:  M1M·M2M·Hor                                  (low matrix threshold)
Z ≥ 3:  M1H·M2H·Hor                  (Hy-en False, He-en False, high matrix)
Z ≥ 3:  M1L·M2L·Hor·ADC3            (Hy-en False, He-en False, high matrix)
Z = 2:  M1M·M2M·Hor̄                  (He-en True)
Z = 2:  M1L·M2L·Mor·Hor̄·ADC2        (He-en True)
Z = 1:  M1L·M2L·Mor̄·ADC2            (Hy-en True)
```

용어:
- M1M, M2M = 4 MeV discriminator (medium threshold)
- M1H, M2H = 16 MeV discriminator (high threshold; Z≥3 distinguishing)
- M1L, M2L = software ≥ 0.5 MeV
- Mor = 'OR' of all medium discriminators (T1–T7)
- Hor = 'OR' of all high discriminators (M1–T7)
- ADC2 = 2 consecutive stack ADC coincidence
- ADC3 = 3 consecutive stack ADC coincidence
- Hy-Enable, He-Enable = timer-controlled flags throttling H/He throughput

큰 SEP에서 양성자 분석을 throttle하기 위해 "Hy timer" (0–130 s 가변, 보통 ~10 s 설정) 사용. Helium도 별도 "He timer". 이렇게 H, He, Z≥3 입자가 telemetry rate 1992 bps에 적합하게 분리된다.

Table IV gives six coincidence equations distinguishing Z = 1, 2, ≥3 with two threshold levels. The "medium" (M1M/M2M) discriminator at ~4 MeV blocks penetrating protons; the "high" (M1H/M2H) at ~16 MeV admits only Z ≥ 3. "Mor" is the OR of all medium discriminators on T1–T7; "Hor" is the OR of all high discriminators on M1–T7. ADC2 (two consecutive stack ADCs in coincidence) is a backup trigger for nuclei stopping in T1 that don't quite trigger M1, M2; ADC3 (three consecutive stack ADCs) is for light nuclei C, N, O stopping deep in the stack. Two programmable timers (Hy-timer, He-timer) reset rapidly after H or He events to throttle their analysis rate, keeping H + He at ≤ 1% of analyzed events under high-rate conditions.

#### Priority Buffer System / 우선순위 버퍼 시스템

큰 SEP에서 Z ≥ 6 ions이 ≥ 100 s⁻¹ trigger 가능하지만 1992 bps telemetry는 10–15 events/s만 전송 가능 → 95개 prioritized buffers (Table VII). Buffers는 다음 axes로 분할:
- Charge: H, He, 3 ≤ Z ≤ 9, Z ≥ 10 (4 categories)
- Range: 0 (stops in M2), 1, 2 (stops T1, T2), 3, 4 (T3, T4), 5, 6, 7 (T5–T7), 8 (penetrates)
- Zenith angle: ≤ 15°, 15°–25°, ≥ 25°
- Quality flags: mh (multi-hodoscope tracks), hz (hazard within 20 µs of previous PHA)

Class system (4 levels, 0–3): Class 0 = highest desirability (high-Z penetrating). Buffer는 우선순위 순으로 readout되지만, 매 256-s major frame 시작에서 첫 N 이벤트는 buffer를 cycle하여 모든 buffer가 sampled 되도록 한다.

In intense SEP events, the on-board priority system selects the most interesting events for the limited 1992 bps downlink. 95 prioritized buffers (Table VII) sort events by (a) charge category (H, He, 3 ≤ Z ≤ 9, Z ≥ 10), (b) range (last detector triggered), (c) zenith angle (≤ 15°, 15°–25°, ≥ 25°), and (d) quality flags. The "class" system (0–3) prevents on-board processor saturation by discarding low-class events when more than 20 of that class are awaiting telemetry. Buffers are read out in priority order, except the first N events of each 256-s major frame are read by cycling through all buffers.

#### Event Compression / 이벤트 압축

12-bit pulse heights from 8 stack PHAs + 256 matrix PHAs per event would require ~1.5 minor frames (1.5 s) at 1992 bps to send a single event. Variable-length compression sends only triggered strip pulse heights with strip identification (20 bits per non-zero strip). Typical event in T4 with 6 strips = 20 bytes. Average ~10 events/s, > 10× improvement.

Event compression is essential: each event has 8 stack PHAs (12-bit) + 256 matrix PHAs (12-bit). Sending all of these uncompressed would require ~1.5 minor frames per event. Instead, only triggered strips and their identifications are sent; matrix info uses 20 bits per non-zero strip. By default, max 10/9/6/6 strips on M1-top/M1-bot/M2-top/M2-bot are telemetered. Typical compressed event: 20 bytes (vs. 109 bytes maximum for stim event triggering all).

### Part V: Calibration and Performance (Section 4) / 보정과 성능

#### Detector Thickness Calibration / 검출기 두께 보정

세 가지 보정 기법:
1. **Particle calibration**: 870 MeV ³⁶Ar at MSU/NSCL, raster-scanned across detector. Thinner detectors: 9 spots, beam stops in well-characterized E' detector. Thicker: full raster.
2. **Sliver method**: Detector slivers cut from flat-edge detectors measured to 0.12 µm with mechanical micrometer; depth-vs-E' calibration.
3. **Interferometer mapping**: Dual-laser system (Milliken et al. 1995) maps 30 stack detectors. Compared to particle data in Figure 17 (750 µm detector, contours 750–754 µm).

이 결과 detector thickness는 0.1% 정확도, dead layers 0.1–0.6 µm uniform within ±0.05 µm로 결정됐다. ³⁶Ar에 의한 radiation hardness test: 2×10⁸ particles/cm², leakage current rose 4×–7× to ~7 µA, no measurable charge collection degradation observed.

Detector thickness calibration was the most demanding step. Three methods were combined: (1) accelerator beams of 870 MeV ³⁶Ar at MSU/NSCL (raster-scanned, beam stops in residual-E detector), (2) "slivers" of detector material left over after cutting flats, measured to 0.12 µm with mechanical micrometers and used for absolute depth/E' calibration, and (3) dual-laser interferometric mapping (Milliken et al. 1995) for full 2-D thickness profiles. Figure 17 shows excellent agreement between particle and interferometer measurements of a 750 µm detector. Radiation-hardness check: 2×10⁸ ³⁶Ar/cm² accumulated; leakage rose 4–7× but charge collection unchanged.

#### Instrument Calibrations / 기기 보정

세 차례의 정밀 calibration 수행:
1. MSU/NSCL Feb 1996: 100 MeV/nucl ²⁰Ne, ⁴⁰Ar, ⁶⁰Ni — engineering model.
2. MSU 1997 Apr: spare detector + flight electronics calibration unit.
3. **GSI Darmstadt Jun 1996**: flight instrument; 300 MeV/nucl ¹⁸O and 300, 500, 700 MeV/nucl ⁵⁶Fe.

GSI 보정에서 ⁵⁶Fe at 500 MeV/nucl + polyethylene fragmentation target → produces full Z-range fragment spectrum. Figure 18 shows ΔE (T1–T6) vs E' (T7) for Z ≈ 20 fragments — Cl, Ar, K, Ca, Sc tracks clearly resolved with isotope substructure. 검증 항목: 매트릭스 trajectory uniformity, threshold 정확도, coincidence logic, ADC linearity, 카운팅 비율, priority/readout 시스템, 입사각/에너지에 따른 응답 변화, range-energy 정확도, leakage current 균형.

Three full-instrument calibrations: (1) February 1996 at MSU/NSCL with 100 MeV/nucl ²⁰Ne, ⁴⁰Ar, ⁶⁰Ni on engineering model, (2) April 1997 at MSU on a calibration unit of spare detectors and flight electronics, (3) June 1996 at GSI Darmstadt with the flight instrument using 300 MeV/nucl ¹⁸O and 300, 500, 700 MeV/nucl ⁵⁶Fe. A polyethylene fragmentation target produced a full Z-spectrum (Figure 18 shows Cl, Ar, K, Ca, Sc tracks resolved). The instrument was mounted on a 4-DOF stage to test angular response. All nine functional aspects (matrix uniformity, thresholds, coincidence logic, ADC response, counting rates, priority readout, angle/energy variation, range-energy accuracy, leakage balancing) passed.

#### Energy Range and Geometry Factor / 에너지 범위와 기하학적 인자

Figure 19에서 SIS의 응답 영역이 Z 1–30, 1 ~ 1000 MeV/nucl 평면에 표시된다. Element-only band은 stopping range 너머까지 (특정 종류만 식별), isotope-resolving band은 더 좁다. Table I 발췌:
- O: 10–90 MeV/nucl (isotope)
- Si: 13–125 MeV/nucl
- Fe: 17–170 MeV/nucl

Figure 20은 He, O, Fe에 대한 kinetic energy 의존 geometry factor: ~38 cm² sr 평탄 영역, 그 후 빠르게 0으로 감소 (입자가 망원경 측면으로 빠지면 제거됨).

Figure 19 displays SIS response in the Z–E plane: 1 ≤ Z ≤ 30 with three bands (isotope-resolving, element-only via penetration, integral fluxes). Energy intervals for isotope analysis: O 10–90 MeV/nucl; Si 13–125; Fe 17–170. Below the lower limit, particles stop in M2 alone (range 0); above the upper limit, they penetrate T7 and only element ID is possible. Figure 20 plots the geometry factor versus energy: He, O, Fe each peak at ~38 cm² sr (combined two telescopes), with the high-energy cutoff scaling as ∼(stack range)/(beam energy).

#### Real-Time Solar Wind Contribution / 실시간 태양풍 데이터 기여

NASA HQ가 1996년 6월에 SEP가 우주정거장 우주인 방사선 위협이 될 수 있는지 실시간 모니터링을 요청했다. SIS는 본래 ≥ 10 MeV proton에 직접 응답하지 않지만, T4 singles (≈ 10–30 MeV 양성자) 와 T6·T7 coincidence (≈ 30–80 MeV 양성자) rate를 32-s마다 RTSW 데이터 스트림에 추가하여 NOAA에 실시간 제공했다.

In response to a NASA HQ inquiry (June 1996) about astronaut radiation protection on the Space Station, SIS was reprogrammed to provide T4 singles (~10–30 MeV protons, ~55 cm² sr nominal) and T6·T7 coincidence (~30–80 MeV protons, ~35 cm² sr) rates every 32 s for the NOAA Real-Time Solar Wind (RTSW) system (Zwickl et al. 1998).

#### In-Flight Performance / 초기 운용 결과

ACE는 1997년 8월 25일 발사. 8월 27일 SIS power-on, 정상 작동 확인. 첫 몇 달간 quiet-time data → ACR C, N, O, Ne, Ar 저에너지 상승 확인 + 최초의 ACR ²²Ne, ¹⁸O 에너지 스펙트럼 발표.

1997년 11월 6일 large gradual SEP event: 6 ≤ Z ≤ 30, ~100 MeV/nucl까지 통계 정밀 측정. 일부 원소(C, O, Ne, Mg, Si, S, Ar, Ca, Fe, Ni)에서 다중 동위원소 분리 확인. 예비 mass resolution: σ_O ≈ 0.17 amu, σ_Fe ≈ 0.40 amu r.m.s. (range 4–7). Matrix readout이 chance-coincidence (low-energy proton + heavy ion) trajectory 식별에 효과적임을 입증. 임펄시브 ³He-rich 이벤트 다수 확인.

ACE was launched 25 August 1997; SIS turned on 27 August and found operating as designed. During the first months (quiet time), SIS measured 2 ≤ Z ≤ 28 and reported the first ACR ²²Ne and ¹⁸O energy spectra. The 6 November 1997 large gradual SEP event provided the first high-rate test: 6 ≤ Z ≤ 30 spectra to ~100 MeV/nucl with multiple resolved isotopes for C, O, Ne, Mg, Si, S, Ar, Ca, Fe, Ni. Preliminary mass resolution: σ ≈ 0.17 amu at O and σ ≈ 0.40 amu at Fe (ranges 4–7), both quiet-time and in the SEP event. The matrix readout successfully discriminated chance-coincidence events (heavy ion + low-energy proton), and several impulsive ³He-rich events were also detected.

### Part VI: Appendix Highlights / 부록 요점

#### Appendix A — Iterative Mass Calculation / 반복식 질량 계산

기본 식: $\mathcal{R}_{Z,M}((\Delta E + E')/M) - \mathcal{R}_{Z,M}(E'/M) = L$. R ∝ M/Z² 사용 후 양변에 M을 명시적으로 분리:

$$ M \simeq M_0 \left[ \frac{\mathcal{R}_{Z,M_0}((\Delta E + E')/M_0) - \mathcal{R}_{Z,M_0}(E'/M_0)}{L} \right]^{1/(a-1)} $$

$a \approx 1.7$로 두면 빠르게 수렴. Dead layer $l$ 보정: $\Delta E \to \Delta E + \delta E$, $L \to L + l$.

The implicit equation for M is solved via M_0 → M_n iteration using the power-law approximation R(E/M) ≃ k M/Z²(E/M)^a. The factor (a–1)⁻¹ in the exponent (with a ≈ 1.7) gives convergence in ~3–5 iterations for typical SIS events. Dead-layer δE correction is bookkept by ΔE → ΔE + δE, L → L + l.

#### Appendix B — Matrix PHA Circuit Details / 매트릭스 PHA 회로 상세

Figure 21 block diagram: detector strip → DC-coupled CSA (15 pF feedback) → unity gain buffer → switched hold capacitors C1, C2, C3 (3 µs alternation between C2 and C3 for baseline tracking). Comparator input from sample-and-held C1 vs. baseline; current source ramps C1 down at known rate; counter latched at zero crossing → 12-bit grey-coded output. Pedestal C4 ensures non-zero rundown for small inputs. Discriminator branch via shaper (~500 ns peaking) with two discriminators per channel for Z = 1 / Z ≥ 2 distinction.

The CSA integrates strip charge with ~100 ns time constant. Two C2/C3 hold capacitors alternate every 3 µs to track DC drift. SW1 captures the signal in C1; a current source ramps C1 down to baseline (Wilkinson method); the time interval is digitized by an 8 MHz counter latched into a 12-bit register. C4 pedestal ensures a non-zero rundown even for null input. CSA reset switch periodically discharges ~512 s; an external IDAC compensates leakage to ≤ 2 nA.

---

## 3. Key Takeaways / 핵심 시사점

1. **dE/dx · E 기법의 정밀화 / Refined dE/dx · E technique** — 식 (1) (dE/dx)·E ∝ Z²M로 시작하여 식 (2) (range-difference equation)으로 일반화함으로써, ΔE 검출기 두께 L이 입자 range의 상당 비율인 경우 mass resolution이 크게 개선된다. SIS는 stack의 모든 (T_i, T_{i+1}) 쌍을 ΔE-E 분석에 활용하므로 redundant mass estimates가 가능하다. The improvement of using L comparable to the stopping range, plus iterating with R(E/M) ≃ k M/Z² (E/M)^a power-law, gives explicit (Equation A2) and rapidly convergent mass formulas.

2. **두 개의 동일 망원경 + 38.4 cm² sr 기하학적 인자 / Twin-telescope, 38.4 cm² sr design** — 단일 망원경 대신 두 개를 쌍둥이로 운영하여 통계, 신뢰성, in-flight calibration 모두 두 배가 된다. 결과적으로 큰 gradual SEP에서 ⁶⁴Zn, ⁵⁸Ni 같은 1% 미만 동위원소도 측정 가능. The geometry factor exceeds previous instruments (e.g., MAST) by factors of several, enabling detection of rare isotopes such as ⁶⁴Zn (~10 events) and ⁵⁸Ni (~few hundred) during a single 30 October 1992-class event.

3. **2D 위치감응 매트릭스 검출기 hodoscope / 2D position-sensitive matrix hodoscope** — M1, M2의 양면 64×64 직교 strip 구조로 trajectory를 ~0.25° r.m.s.로 결정. 6 cm 분리에서 ~0.29 mm 위치 분해능. Strip별 독립 PHA로 chance-coincidence 거부. The X–Y orthogonal strip readout of M1 and M2, each 64-strip on each face (256 strips per telescope, 512 total), provides angular resolution ~0.25° r.m.s., critical for accurate L sec θ in Equation (2). Independent strip PHAs reject chance coincidences (heavy ion + low-energy proton).

4. **맞춤형 16채널 CMOS VLSI PHA 칩 / Custom 16-channel CMOS VLSI PHA chip** — 32 chips × 16 channels = 512 PHAs in compact, low-power form. UTMC radiation-hard 1.2 µm process, 13 mW/channel, 1400:1 dynamic range, 100 kRads tolerance. 우주 입자기기에서 first-generation per-strip ASIC PHA의 표본. The custom Matrix VLSI chip (16 PHAs/chip, 13 mW/channel, 1400:1 dynamic range, ~100 kRads tolerance) was the first ASIC of its kind for space particle instrumentation, allowing 512 independent strip readouts in a thermal/power budget that would have been impossible with discrete electronics.

5. **온보드 우선순위 버퍼/클래스 시스템 / On-board priority buffer + class system** — 95개 prioritized buffer로 telemetry-limited SEP 환경에서 가장 흥미로운 high-Z 이벤트를 보존. Table VII는 charge × range × angle × quality 의 4차원 정렬 체계. A 4-D event sorting (charge × range × zenith × quality) into 95 buffers, plus a class system (0–3) preventing processor saturation, ensures that with only 1992 bps telemetry, ~10 events/s of highest-Z, highest-range events are downlinked even when ≥ 100 Z ≥ 6 events/s trigger SIS.

6. **0.1% 검출기 두께 정확도 요건 / 0.1% detector thickness uniformity requirement** — Mass resolution ≤ 0.1 amu @ Fe를 위해 두께 정확도 0.1% 필요. 인터페로미터 + 가속기 두 방법으로 30 stack detectors 모두 매핑. Achieving ≤ 0.1 amu mass error at Fe required mapping all 30 stack detectors with ≤ 0.1% thickness uniformity, accomplished by combining accelerator beam (³⁶Ar at 870 MeV) raster scans with dual-laser interferometric maps (Milliken et al. 1995). Sliver-method micrometry provided absolute calibration to 0.12 µm.

7. **운영 모드의 다중 트리거 / Multi-mode trigger architecture** — ADC2, ADC3 보조 트리거가 M1·M2 트리거를 놓친 경우(특히 deep-stopping H, He)에도 의미있는 카운트 보존. Hy-, He-timer로 H, He 분석률 ~1%로 throttle. Backup ADC2 (two consecutive stack ADCs in coincidence) and ADC3 (three consecutive) triggers catch H and He that don't quite trigger M1·M2 due to deep stopping (~4 MeV thresholds), while programmable Hy- and He-timers (0–130 s) throttle their analyzed rate to ~1% during high-flux events, preserving heavy-ion bandwidth.

8. **첫 운용 결과의 검증 / Validation by November 1997 SEP event** — 발사 후 ~3개월 만에 발생한 1997년 11월 6일 large gradual SEP에서 6 ≤ Z ≤ 30, σ_O ≈ 0.17 amu, σ_Fe ≈ 0.40 amu r.m.s. 달성. 이는 GSI calibration 예측과 일치하며 SIS 설계 철학을 in-flight 검증한 사례. The 6 November 1997 event provided the first in-flight high-rate validation: ranges 4–7 yielded σ ≈ 0.17 amu at O and σ ≈ 0.40 amu at Fe, matching pre-flight GSI calibration. Multiple isotopes resolved for C, O, Ne, Mg, Si, S, Ar, Ca, Fe, Ni; matrix readout successfully rejected chance-coincidence events.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Particle Identification / 입자 식별

**Eq. 1 — Zeroth-order invariant**

$$ \left(\frac{dE}{dx}\right) \cdot E \propto Z^2 M $$

비상대론적 입자 (E ∝ Mv², dE/dx ∝ Z²/v²)에서 곱 (dE/dx)·E는 속도 v와 무관 → dE/dx vs E 평면에서 (Z, M) 핵종은 hyperbola를 그린다. (Non-relativistic ions: E ∝ Mv², dE/dx ∝ Z²/v². Their product is velocity-independent, giving hyperbolic tracks separated by Z²M in the dE/dx–E plane.)

**Eq. 2 — Refined range-difference identity**

$$ R_{Z,M}(E/M) - R_{Z,M}(E'/M) = L\,\sec\theta $$

| Symbol | 의미 / Meaning |
|---|---|
| $R_{Z,M}(E/M)$ | charge Z, mass M인 입자의 kinetic energy E에서의 range (Hubert et al. 1990) |
| $E$ | total kinetic energy entering ΔE detector |
| $E'$ | residual energy after ΔE detector |
| $\Delta E$ | $E - E'$ — energy deposited in ΔE detector (measured) |
| $L$ | thickness of ΔE detector (measured to ~0.1%) |
| $\theta$ | incidence angle (measured by hodoscope, ~0.25° r.m.s.) |

이 implicit equation은 Z, M에 대해 풀린다. 보통 M/Z를 가정하고 Z를 정수로 결정한 뒤 M을 풉니다. (This implicit equation in Z and M is solved by first assuming a mass-to-charge ratio M/Z to obtain a clean Z histogram, then iterating for M with Z fixed.)

### 4.2 Iterative Mass Formula / 반복식 질량공식 (Appendix A)

Power-law approximation $R_{Z,M}(E/M) \simeq k\, M/Z^2 \, (E/M)^a$ (실제 데이터에서 $a \approx 1.7$):

**Eq. A1**

$$ \mathcal{R}_{Z,M_0}\!\left(\frac{\Delta E + E'}{M}\right) - \mathcal{R}_{Z,M_0}\!\left(\frac{E'}{M}\right) = (M_0/M)\, L $$

**Eq. A2**

$$ M \simeq M_0\left[\frac{\mathcal{R}_{Z,M_0}((\Delta E + E')/M_0) - \mathcal{R}_{Z,M_0}(E'/M_0)}{L}\right]^{1/(a-1)} $$

수렴: 보통 3–5회 반복으로 ≤ 0.01 amu 변화. (Convergence: typically 3–5 iterations to within ≤ 0.01 amu.)

Dead-layer correction: $\Delta E \to \Delta E + \delta E$, $L \to L + l$. ($\delta E$는 dead layer $l$에서의 예상 추가 에너지 손실, mass guess로부터 계산됨.)

### 4.3 Mass Resolution Decomposition / 질량 분해능 분해

식 (2)의 partial derivatives로 모든 잡음원의 기여를 분리:

$$ \sigma_M^2 = \left(\frac{\partial M}{\partial \Delta E}\right)^2 \sigma_{\Delta E}^2 + \left(\frac{\partial M}{\partial E'}\right)^2 \sigma_{E'}^2 + \left(\frac{\partial M}{\partial L}\right)^2 \sigma_L^2 + \left(\frac{\partial M}{\partial \theta}\right)^2 \sigma_\theta^2 + \sigma_{\text{straggling}}^2 + \sigma_{\text{scattering}}^2 + \sigma_{\text{pickup}}^2 $$

| 잡음원 / Source | 표현 / Expression | 영향 / Impact |
|---|---|---|
| Electronics noise | $\sigma_E$ ≤ 100 keV r.m.s. | Linear in σ_M for thin detectors |
| Detector thickness | $\sigma_L/L$ ≤ 0.1% | Dominant for Fe-group; hence interferometer mapping |
| Trajectory angle | $\sigma_\theta$ ≈ 0.25° r.m.s. | Significant at large θ, sec θ amplification |
| Energy-loss straggling | physics-determined (Vavilov) | Irreducible |
| Multiple scattering | physics-determined (Highland) | Irreducible |
| Electron pickup/loss | physics-determined | Negligible above ~10 MeV/nucl for SIS targets |

The total mass resolution σ_M² is the sum of contributions from instrumental (electronics noise σ_E ≤ 100 keV r.m.s., thickness uniformity σ_L/L ≤ 0.1%, trajectory angle σ_θ ≤ 0.25° r.m.s.) and physical (energy-loss straggling, multiple scattering, electron pickup/loss) sources. SIS achieved σ_O ≈ 0.17 amu and σ_Fe ≈ 0.40 amu r.m.s. in flight, well below the 0.25 amu adjacent-isotope-resolving requirement (Stone 1973).

### 4.4 Geometry and Range / 기하 및 range

| 매개변수 / Parameter | 값 / Value |
|---|---|
| Telescope opening angle | 95° full |
| M1–M2 separation | 6 cm |
| Position resolution | 0.29 mm |
| Angular resolution | 0.25° r.m.s. (averaged) |
| Total stopping thickness | ~8.25 mm Si |
| Active areas (matrix / stack) | 34 cm² / 65 cm² |
| Number of strips | 4 surfaces × 64 strips = 256 per telescope (512 total) |
| Geometry factor (max) | 38.4 cm² sr (sum of two telescopes) |
| Mass | 21.9 kg |
| Power | 17.8 W (nominal) |
| Bit rate | 1992 bps |

**Energy intervals for isotope analysis (Table I):**
- O: 10–90 MeV/nucl
- Si: 13–125 MeV/nucl
- Fe: 17–170 MeV/nucl

**Charge intervals:** primary 4 ≤ Z ≤ 28, extended 1 ≤ Z ≤ 30 (with ADC2/ADC3 triggers).

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1968 ─ Stone et al., OGO-6 (early Si detector telescope)
   │
1973 ─ Stone, "Cosmic Ray Isotopes" (ICRC) — dE/dx-E formalism standard
   │
1978 ─ 23 Sep large flare — first SEP isotope measurement (Caltech / Mewaldt)
   │
1984 ─ Mewaldt et al., ISEE-3 SEP isotopes (Astrophys. J. 280)
   │
1985 ─ Breneman & Stone, photospheric vs coronal abundances (Astrophys. J. 299)
   │
1989 ─ Anders & Grevesse "Solar System Abundances" (reference table)
   │
1992 ─ SAMPEX / MAST launch (Cook et al. 1993, IEEE)
   │
1995 ─ Reames "Solar Energetic Particles — A Paradigm Shift" (review)
   │
1997 ─ ACE launch ── ★ THIS PAPER (Stone et al. 1998)
   │
1997 ─ Stone et al., CRIS instrument paper (Space Sci. Rev. 86)
   │   ── CRIS covers ~100–500 MeV/nucl GCR isotopes; SIS extends to ~10–100 MeV/nucl
   │
2003 ─ Halloween events (Oct/Nov 2003) — SIS measures ¹³C/¹²C, ²²Ne/²⁰Ne in extreme SEP
   │
2008 ─ Mewaldt et al., extensive SEP isotope catalog from cycle 23 (ACE/SIS)
   │
2017 ─ Sep 2017 SEP storm — SIS still operating, 20 yr in space
   │
2024 ─ Solar cycle 25 SEP harvest with SIS, IMAP/HIT successor
```

Stone et al. 1998은 SEP isotope spectrometer 설계의 standard reference로 자리잡았으며, STEREO/HET, Solar Orbiter/SIS, IMAP/HIT 등 이후 임무가 모두 이 설계 철학을 직접 차용했다. (Stone et al. 1998 became the canonical reference for SEP isotope spectrometer design; STEREO/HET, Solar Orbiter/SIS, IMAP/HIT all directly inherited the twin-telescope, position-sensitive Si hodoscope, multi-ΔE-E, custom VLSI PHA architecture.)

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **Stone (1973), "Cosmic Ray Isotopes"**, ICRC 13 | Defines the (dE/dx)·E ∝ Z²M particle ID and resolution requirements ≤0.25 amu | Foundational physics underpinning Equations 1–2; Stone reuses his own formalism 25 years later |
| **Mewaldt et al. (1984a), Astrophys. J. 280**, ISEE-3 SEP isotopes | First high-resolution SEP isotope measurements | SIS's direct predecessor; SIS aims to extend results with ~100× more events per equivalent flux |
| **Anders & Grevesse (1989)**, Geochim. Cosmochim. Acta 53 | Standard solar-system abundance table | SEP gradual events compared to this table to test coronal vs photospheric origin |
| **Fisk, Kozlovsky, Ramaty (1974)**, Astrophys. J. Lett. 190 | ACR origin: pickup-ion acceleration at termination shock | Justifies SIS's ACR isotope objectives (Section 2.2) |
| **Reames (1995)**, Rev. Geophys. Suppl. 33 | Gradual vs impulsive SEP paradigm | Frames SIS science requirements; ³He-rich detection capability |
| **Cook et al. (1993a)**, IEEE Trans. Geosci. Remote Sensing 31 | MAST instrument on SAMPEX | Direct predecessor instrument; SIS is the next-generation, larger-G version |
| **Hubert, Bimbot, Gauvin (1990)**, At. Data Nucl. Data Tables 46 | Range-energy tables for 2.5–500 MeV/nucl heavy ions | Fundamental input to Equation 2 — every mass calculation uses these tables |
| **Wiedenbeck et al. (1996)**, SPIE Conf. Proc. 2806 | 2D position-sensitive Si detector design for ACE/SIS | Companion paper detailing matrix detector hardware |
| **Milliken et al. (1995)**, ICRC 24 | Dual-laser interferometric thickness mapping | Calibration technique enabling 0.1% thickness uniformity |
| **Stone et al. (1998b)**, Space Sci. Rev. 86 (CRIS paper) | CRIS instrument (~100–500 MeV/nucl GCR isotopes) | Sister instrument; together SIS+CRIS span ~10–500 MeV/nucl isotope measurements |
| **Klecker et al. (1995)**, Astrophys. J. 442 | SAMPEX/HILT charge state measurements of ACR | Confirmed Q ≈ +1 ACRs, validating the LISM-origin model SIS will refine |
| **Zwickl et al. (1998)**, Space Sci. Rev. 86 | NOAA Real-Time Solar Wind System using ACE | SIS T4-singles and T6·T7 rates contribute proton flux monitoring for space weather |

---

## 7. References / 참고문헌

- Stone, E. C., Cohen, C. M. S., Cook, W. R., Cummings, A. C., Gauld, B., Kecman, B., Leske, R. A., Mewaldt, R. A., Thayer, M. R., Dougherty, B. L., Grumm, R. L., Milliken, B. D., Radocinski, R. G., Wiedenbeck, M. E., Christian, E. R., Shuman, S., and von Rosenvinge, T. T. (1998). "The Solar Isotope Spectrometer for the Advanced Composition Explorer." *Space Science Reviews* **86**, 357–408. DOI: 10.1023/A:1005027929871
- Anders, E. and Grevesse, N. (1989). "Abundances of the Elements: Meteoritic and Solar." *Geochim. Cosmochim. Acta* **53**, 197–214.
- Cook, W. R., et al. (1993a). "MAST: A Mass Spectrometer Telescope for Studies of the Isotopic Composition of Solar, Anomalous, and Galactic Cosmic Ray Nuclei." *IEEE Trans. Geosci. Remote Sensing* **31**, 557–564.
- Fisk, L. A., Kozlovsky, B., and Ramaty, R. (1974). "An Interpretation of the Observed Oxygen and Nitrogen Enhancements in Low-Energy Cosmic Rays." *Astrophys. J. Lett.* **190**, L35–L38.
- Hubert, F., Bimbot, R., and Gauvin, H. (1990). "Range and Stopping-Power Tables for 2.5–500 MeV/Nucleon Heavy Ions In Solids." *At. Data Nucl. Data Tables* **46**, 1–213.
- Klecker, B., et al. (1995). "Charge State of Anomalous Cosmic-Ray Nitrogen, Oxygen, and Neon: SAMPEX Observations." *Astrophys. J. Lett.* **442**, L69–L72.
- Mewaldt, R. A., Spalding, J. D., and Stone, E. C. (1984a). "A High-Resolution Study of the Isotopes of Solar Flare Nuclei." *Astrophys. J.* **280**, 892–901.
- Milliken, B. D., Leske, R. A., and Wiedenbeck, M. E. (1995). "Silicon Detector Studies with an Interferometric Thickness Mapper." *Proc. 24th Int. Cosmic Ray Conf., Rome* **4**, 1283–1286.
- Reames, D. V. (1995). "Solar Energetic Particles — A Paradigm Shift." *Rev. Geophys. Suppl.* **33**, 585–589.
- Stone, E. C. (1973). "Cosmic Ray Isotopes." *Proc. 13th Int. Cosmic Ray Conf., Denver* **5**, 3615–3626.
- Stone, E. C., et al. (1998a). "The Advanced Composition Explorer." *Space Sci. Rev.* **86**, 1.
- Stone, E. C., et al. (1998b). "The Cosmic Ray Isotope Spectrometer for the Advanced Composition Explorer." *Space Sci. Rev.* **86**, 285.
- Wiedenbeck, M. E., et al. (1996). "Two-Dimensional Position-Sensitive Silicon Detectors for the ACE Solar Isotope Spectrometer." *SPIE Conf. Proc.* **2806**, 176–187.
- Zwickl, R. D., et al. (1998). "The NOAA Real-Time-Solar-Wind (RTSW) System Using ACE Data." *Space Sci. Rev.* **86**, 635.
