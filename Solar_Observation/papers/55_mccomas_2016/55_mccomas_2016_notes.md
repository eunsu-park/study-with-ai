---
title: "Integrated Science Investigation of the Sun (ISIS): Design of the Energetic Particle Investigation"
authors: D.J. McComas et al. (51 authors)
year: 2016
journal: "Space Science Reviews 204, 187–256"
doi: "10.1007/s11214-014-0059-1"
topic: Solar_Observation
tags: [Parker_Solar_Probe, SPP, ISIS, EPI-Lo, EPI-Hi, SEP, energetic_particles, TOF, dE/dx-E, DSA, instrumentation]
status: completed
date_started: 2026-04-25
date_completed: 2026-04-25
---

# 55. Integrated Science Investigation of the Sun (ISIS): Design of the Energetic Particle Investigation / 태양 통합 과학 탐사(ISIS): 에너지 입자 탐사 장비 설계

---

## 1. Core Contribution / 핵심 기여

The ISIS investigation, led by D.J. McComas (SwRI), is the energetic-particle science suite on NASA's Solar Probe Plus (SPP) mission — now Parker Solar Probe — which from 2018 onward repeatedly traverses the inner heliosphere down to **9 R_S** of the Sun's surface. This paper, written at the time of the SPP Preliminary Design Review (January 2014), comprehensively documents (1) the three driving science questions that ISIS is built to answer — *origins, acceleration, and transport* of solar energetic particles (SEPs) — and (2) the detailed mechanical, electrical, optical, software, and operational design of the two-instrument complement: **EPI-Lo** (a novel 80-aperture time-of-flight ion mass spectrometer + electron SSDs covering ions ~0.02–15 MeV/nuc and electrons 25–1000 keV) and **EPI-Hi** (three SSD-based dE/dx-E telescopes — HET, LET1, LET2 — covering ions ~1–200 MeV/nuc and electrons 0.5–6 MeV). Together the suite spans more than seven decades of energy with hemispheric angular coverage, mass-resolves ³He from ⁴He, and identifies major heavy-ion species H–Fe (and select isotopes ²⁰Ne/²²Ne).

ISIS는 D.J. McComas (SwRI) 가 PI로 이끄는 NASA Solar Probe Plus (SPP) — 현 Parker Solar Probe — 임무의 에너지 입자 과학 슈트(suite)이다. SPP는 2018년 발사 후 태양 표면으로부터 9 태양반경(R_S) 까지 반복 진입한다. 본 논문은 2014년 1월 SPP PDR(Preliminary Design Review) 시점에 작성되어, ISIS가 답할 세 가지 핵심 과학 질문 — SEP의 (1) 기원(origin), (2) 가속(acceleration), (3) 수송(transport) — 과 두 기기의 기계·전자·광학·소프트웨어·운영 설계를 상세히 기술한다. 두 기기는 **EPI-Lo** (80개 구멍 비행시간 이온 질량 분광계 + 전자 SSD; 이온 0.02–15 MeV/nuc, 전자 25–1000 keV) 와 **EPI-Hi** (3개 SSD 기반 dE/dx-E 망원경: HET, LET1, LET2; 이온 1–200 MeV/nuc, 전자 0.5–6 MeV) 로 구성된다. 합치면 7십수배(decades) 이상의 에너지 범위와 거의 반구 시야각을 커버하며, ³He/⁴He 분리, 그리고 H–Fe 의 모든 주요 중이온 종(species) 을 식별한다 (²⁰Ne/²²Ne 동위원소 일부 포함).

---

## 2. Reading Notes / 읽기 노트

### Part I: Scientific Background and Three Driving Questions / 과학적 배경과 세 가지 핵심 질문

#### 1.1 SPP and ISIS Mission Goals / SPP·ISIS 임무 목표 (pp. 188–190)

SPP는 태양 코로나(corona)와 태양풍의 출발 영역인 < 60 R_S 영역을 처음으로 in situ 측정한다. ISIS의 세 과학 질문(p. 189):

1. **SEP의 기원/seed 모집단(seed population)은 무엇인가?**
2. **이 SEP들은 어떻게 가속되는가?** (reconnection, shock, turbulence)
3. **어떤 메커니즘이 입자 모집단을 헬리오스피어로 수송하는가?**

기기 슈트는 SPP의 다른 페이로드 — SWEAP (태양풍 플라즈마, Kasper et al. 2016), FIELDS (전자기장, Bale et al. 2016), WISPR (광시야 영상기, Vourlidas et al. 2016) — 와 좌표화 운영된다.

The three driving science questions of ISIS are: (1) What is the origin/seed population of SEPs? (2) How are they accelerated? (3) What transport mechanisms move them outward into the heliosphere? ISIS is one of four SPP investigations (SWEAP, FIELDS, WISPR, ISIS) and is coordinated with all of them.

#### 1.2 Two Sources of SEPs / 두 가지 SEP 발생원 (pp. 190–193)

- **Gradual SEPs**: CME-driven shocks가 광범위한 IMF (interplanetary magnetic field) line을 채우고 수일 지속. 양성자 기반, intensity ~10⁵–10⁷ (cm² sr s MeV)⁻¹ at ~0.01–10 MeV/nuc, fluence 위에 거대 (Fig. 2A).
- **Impulsive SEPs**: Solar flare reconnection 또는 jet 분출 결과. ³He/⁴He, electron, heavy ion(Ne–Fe) 강하게 enhanced. 수시간 지속 (Fig. 2B). Q/M 종속 분류 농축 패턴이 핵심 진단 (Mason 2007 Fig. 8).

Helios-1 (0.3 AU) vs IMP-8 (1 AU)의 1980년 5월 사건은 5 회 분리된 분출이 1 AU에서 단일 사건처럼 합쳐짐을 보여 줌 (Fig. 3) — 이것이 SPP가 가까이 가야 하는 핵심 이유.

The two types differ in duration, abundance pattern, and underlying mechanism. The Helios-1 / IMP-8 comparison (Fig. 3) shows five distinct flare injections at 0.3 AU smearing into a single event by 1 AU — the canonical demonstration that close-in measurements are essential for source separation.

#### 1.3 Six Specific Acceleration Questions / 여섯 가지 가속 메커니즘 질문 (p. 13)

1. 같은 속도의 CME가 왜 peak proton intensity가 4 십수배(4 orders of magnitude)로 다른가? (Kahler 2000)
2. 같은 운동 에너지를 가진 CME가 왜 가속된 입자에 매우 다른 에너지를 부여하는가?
3. spectral break의 Q/M 의존성은 어떻게 결정되는가? E_0 ∝ (Q/M)^α, α ~ 1.75 (Li et al. 2009)
4. 자가 여기 Alfvén wave가 streaming limit에 어떻게 영향을 주는가? (Reames & Ng 1998)
5. 선행 CME의 turbulent wake 또는 suprathermal seed 효과는?
6. 왜 일부 CME만 GLE (Ground Level Event) 를 만드는가? (Mewaldt et al. 2012a)

#### 1.4 Suprathermal Ions: The Universal v⁻⁵ Tail / 초열적 이온 (pp. 197–200)

Fisk & Gloeckler (2012) 는 거의 모든 태양풍 조건에서 1.5–6 × v_SW 의 양성자 phase space density가 f(v) ∝ v⁻⁵ (i.e. dN/dE ∝ E⁻¹·⁵) 의 power law를 가짐을 보였다 (Fig. 6). 이것이 SEP의 universal seed 후보. 6–20 × v_SW 영역은 더 가변적이고 사건 종류에 따라 다양하다.

#### 1.5 ISIS Science Requirements / 측정 요구사항 (Table 2, p. 14)

| Functional parameter | Measurement requirement |
|---|---|
| Energy range — electrons | < 0.05 to > 3 MeV |
| Energy range — p / ions | < 0.05 to > 50 MeV/nuc |
| Energy resolution | > 6 bins/decade |
| Cadence — electrons | < 1 s for selected rates |
| Cadence — p+/ions | 5 s for selected rates |
| FOV | > π/2 sr in both sunward & anti-sunward, including 10° around Parker spiral at perihelion |
| Angular sectoring — e⁻ | < 45° |
| Angular sectoring — ions | < 30° |
| Composition | At least H, ³He, C, O, Ne, Mg, Si, Fe |
| Max intensity (< 1 MeV) | > 10⁶ particles cm⁻² sr⁻¹ s⁻¹ |
| Max intensity (> 1 MeV) | > 5 × 10⁵ particles cm⁻² sr⁻¹ s⁻¹ |

---

### Part II: ISIS Suite Overview / ISIS 슈트 개요

#### 2.1 Mechanical Configuration / 기계 구성 (pp. 215–217)

- **Location**: SPP 본체 후방 ram side, TPS umbra 안쪽 (Fig. 17).
- **ISIS Bracket**: EPI-Lo 와 EPI-Hi 를 함께 hold하는 단일 통합 bracket. ULTEM 1.27 cm spacer로 thermal isolation, copper strap으로 ground.
- **Mass / Power Resources** (Table 3):
  - EPI-Hi: 3.628 kg CBE → 4.320 kg total; 5.810 W instrument power
  - EPI-Lo: 3.435 kg CBE → 4.091 kg total; 4.170 W instrument power
  - ISIS bracket: 0.973 kg total
  - **ISIS total**: 9.384 kg, 11.770 W
- **Telemetry**: ISIS uncompressed 15.180 Gbit/orbit; compressed (75%) 11.435 Gbit/orbit; packetized (105%) 12.007 Gbit/orbit.

The ISIS suite occupies the SPP afterdeck umbra behind the carbon-carbon TPS, with all five 45° HET/LET cones plus 80 EPI-Lo apertures pointed within 10° of the nominal Parker spiral (perihelion). Total ISIS allocation: 9.4 kg, 11.8 W.

#### 2.2 Fields of View / 시야각 (pp. 215–216)

- **EPI-Hi**: 5 overlapping 45°-half-angle cones (HET aft, HET fwd, LET1 aft, LET1 fwd, LET2). HET-fwd 와 LET1-fwd 는 TPS와 충돌하지 않도록 의도적으로 향함 (Fig. 18).
- **EPI-Lo**: 8개 wedge × 10 aperture = 80개 구멍 reaching > π/2 sr; nearly hemispheric (Fig. 19).
- Parker spiral magnetic field 평균 방향(0.05, 0.25, 0.7 AU) 모두 두 기기의 cone 내부에 있음.

---

### Part III: EPI-Lo — The Time-of-Flight Mass Spectrometer / EPI-Lo — 비행시간 질량 분광계

#### 3.1 EPI-Lo Overview / 개요 (pp. 222–225)

- **Heritage**: MESSENGER/EPS, PEPSSI/New Horizons, JEDI/Juno, RBSPICE/Van Allen Probes, MMS/EIS — APL "hockey-puck" TOF lineage.
- **Geometry**: 8 sensor wedges + electronics box (Fig. 15, 25). 각 wedge에 10 entrance aperture, MCP assembly, SSD assembly, collimator set.
- **Apertures**: 80 total, sampling > π/2 sr without articulation.
- **Geometric factor per pixel**: G > 0.05 cm² sr (required), > 0.07 cm² sr (goal).
- **Energy range** (Table 4): electrons 25–1000 keV; ions 50 keV/nuc–15 MeV total E (goal 20 keV/nuc–85 MeV total E for Fe).
- **Energy resolution**: 11% (goal); 45% (required).

#### 3.2 TOF Mass Equation / TOF 질량 방정식

이온이 Start foil(carbon-polyimide-Al, 100 nm)을 통과 → secondary electron 생성 → MCP에서 ~10⁶배 증폭 → start pulse. 다시 Stop foil(65 nm)을 통과 → stop pulse. 마지막으로 SSD에 적층되어 E_SSD 측정.

TOF 시간 t 와 비행 거리 L (~8.73 cm) 로부터 속도 v = L/t. 이로부터 입자 질량:

$$\boxed{\,m = \frac{2\,E_{SSD}\,t^2}{L^2}\,}$$

이상적이면 (TOF, E_SSD) 평면에서 각 종(species)이 다른 곡선 트랙(쌍곡선)을 가짐 — Fig. 21에서 H, ³He, ⁴He, C, O, Ne, Mg, Si, Fe 가 분리됨. ³He–⁴He 분리는 특히 22.5°와 90° 위치에서 가장 우수: ³He/⁴He 비율을 1:100까지 분해 (≥ 1.5 MeV/nuc).

The mass relation is m = 2·E·(t/L)² (in natural units) — or, with E in keV and t in ns and L in cm, m[amu] = 2 E·t²/(L² · 0.01044). Each species follows a hyperbolic track in (TOF, E)-space; Fig. 21 shows H, ³He, ⁴He, C, O, Ne, Mg, Si, Fe well separated, with ³He/⁴He resolvable to 1:100 above ~1.5 MeV/nuc in optimal apertures.

#### 3.3 Electron Detection / 전자 검출

- 같은 entrance aperture 사용. SSD shielded by ~2 μm Al flashing.
- 25–1000 keV. 일부 mode에서는 Start pulse + SSD pulse 동시조건으로 entrance 식별, 더 우수한 background rejection.
- 8 pixels/sensor (sector). G > 0.05 cm² sr.

#### 3.4 Light/UV Filtering / 광·UV 필터링 (pp. 224–227)

근접 태양 환경의 가장 큰 도전: **Lyα (121.6 nm) 산란광**. Most apertures: Lyα ~10⁹ photons cm⁻² s⁻¹ sr⁻¹ at > 20° elongation. **TPS 가장자리 4개 aperture**: peak ~10¹² photons cm⁻² s⁻¹ sr⁻¹.

해법:
- Start foil composition: 24 μg cm⁻² Al + 18 μg cm⁻² Pd (large elongations); 7.3 μg cm⁻² Al + 18 μg cm⁻² Pd + 7.3 μg cm⁻² Al on stop (small elongations near TPS).
- Pd 는 X-ray/EUV 차단; Al 은 가시광 차단.
- Predicted SSD noise from photons: ~0.04 keV μs⁻¹ (large elongation) to 0.7 keV μs⁻¹ (TPS-near) — well below 7 keV μs⁻¹ electronic floor.

#### 3.5 Solar Wind Electron Mitigation / 태양풍 전자 완화

Perihelion에서 solar wind electron flux ~2×10¹² cm⁻² s⁻¹. Aperture pinhole로 침투 가능 → start rate 한도 ~10⁶ s⁻¹ 초과 위험. 대책: collimator에 **double foil**(0.5 cm 분리, 1000 Å polyimide + 50 Å C flashing). Pinhole alignment 확률 → 최대 4×10⁵ s⁻¹ 가능, 한도 만족.

#### 3.6 Electronics / 전자 (pp. 232–235, Fig. 24)

- **Quadrant** × 4 covers 2 wedges each. 20 start anode + 2 stop anode (delay line) per quadrant.
- **HVPS**: max 3300 V to MCP/electron optics, 250 V SSD bias.
- **Event Board**: pulse-detect/discriminator/ADC chains, FPGA-based event logic, soft-core processor (VHDL).
- **Pulse-width energy method** (Paschalidis 2008): 1 MeV 이상에서 dynamic range 확장 (~1.5 → 15–20 MeV total). Three approaches considered for full Fe coverage to 85 MeV.
- Singles rates < 10⁶ s⁻¹ design; coincidence ~70,000 s⁻¹ valid event handling.

#### 3.7 Calibration / 보정 (pp. 240–242)

- APL particle accelerator: H, He, O, noble gas ions 20–170 keV, 10²–10⁶ cm⁻² s⁻¹.
- ²⁴¹Am alpha source for in-lab response.
- GSFC accelerator: protons (8 keV–1 MeV), heavy ions (He, O, Ar) up to ~20 MeV.
- LBNL & INL accelerators for high-energy response.
- In-flight: relative MCP gain via Start/Stop/SSD rate ratios; on-board pulsers; cross-cal with EPI-Hi.

---

### Part IV: EPI-Hi — The dE/dx vs. E Telescopes / EPI-Hi — dE/dx-E 망원경

#### 4.1 Overview / 개요 (pp. 242–245)

- **Heritage**: STEREO/IMPACT LET (Mewaldt et al. 2008b) and HET (von Rosenvinge et al. 2008).
- **Three telescopes**:
  - **HET** (High-Energy Telescope) — double-ended, electrons 0.5–6 MeV, p/He 1–100 MeV/nuc, Z ≥ 6 to 100 MeV/nuc.
  - **LET1** — double-ended, p/He 1–20 MeV/nuc, electrons 0.5–2 MeV.
  - **LET2** — single-ended (one side blocked by spacecraft), same range as LET1.
- **Geometric factor**: ~0.5 cm² sr per cone (5 cones).
- **Detector materials**: ion-implanted Si SSDs, thicknesses 12 μm (L0), 25 μm (L1), 500 μm (H1, H2), 1000 μm (L2–L6, H3–H5; doubled for effective 2000 μm).

#### 4.2 dE/dx-E Particle Identification Principle / 입자 식별 원리

Bethe-Bloch 에너지 손실(non-relativistic limit):

$$-\frac{dE}{dx} \approx \frac{4\pi n e^4 Z^2}{m_e v^2}\,\ln\!\left(\frac{2 m_e v^2}{I}\right) \propto \frac{Z^2}{\beta^2}$$

여기서 Z 는 입자의 원자번호, β = v/c, n 은 매질의 전자수 밀도, I 는 평균 이온화 에너지. 얇은 ΔE 검출기(L0/L1 또는 H1)에서의 손실 ΔE × 두꺼운 잔여 검출기 E' 의 곱은 같은 종(species)에서 일정한 곡선:

$$\Delta E \cdot E' \approx \frac{m\,Z^{2}}{C}\,(\text{hyperbolic, Z-separated})$$

각 element가 (ΔE, E') 평면에 별개의 트랙을 가짐 (Fig. 32, 33). Fig. 32 Monte Carlo simulation (HET, large SEP event composition with H suppressed by 2000): H, He, C, N, O, Ne, Mg, Si, Fe 트랙이 stopping branch와 penetrating branch로 명확히 분리.

#### 4.3 Detector Geometry / 검출기 기하 (Fig. 30, p. 36)

- **L0**: 12 μm thin Si membrane, 1 cm² active center inside ~9 cm² thin window.
- **L1**: 25 μm; central bull's-eye + 4 quadrant sectors.
- **L2/L3/L4/L5/L6**: 1000 μm; central + annular guard, segmented 4 quadrants.
- **H1/H2**: 500 μm; central + 5-segment bull's-eye + 4 surrounding quadrants (25 sectors total, Fig. 35a).
- **H3/H4/H5**: paired 1000 μm = effective 2000 μm; same segmentation pattern.

The thin L0 detector gives the very low energy threshold (ions ≥ ~1 MeV penetrate L0 and trigger an L0·L1 coincidence). L0 is mounted as a thin Si membrane on a thicker support frame so it can accept the full L1·L2 acceptance cone. LET particles are identified either via L0·L1 coincidence (full FOV) or via L1·L2 coincidence (when L0 misses the trajectory).

#### 4.4 ³He/⁴He Isotope Separation / ³He/⁴He 동위원소 분리 (Fig. 34, p. 252)

Lab test using ²⁴⁴Cm α-source through L0 (12 μm) + L1 (25 μm). Resulting ΔE-E' track shows ⁴He resolved from ³He down to ratios "as small as a few percent" — adequate for typical impulsive SEP events where ³He/⁴He can reach 1 to 1000s.

³He/⁴He 분해는 angular sector 선택 + Bohr/Landau straggling 보정으로 향상. 가장 직각에 가까운 trajectory에서 가장 정확.

#### 4.5 Angular Sectoring (Fig. 35) / 각도 섹터링

Each telescope subdivides 45°-half-angle cone into 25 sectors via two position-sensitive detectors (one rotated 45° relative to other). Sector width ~25° max, but ≥ 80% of geometric acceptance falls within 15° of the mean direction — sufficient for pitch-angle distribution measurement.

#### 4.6 Geometric Factor & Dynamic Range (Fig. 36) / 기하 인자 및 동적 범위

LET stopping particles: G ~0.4 cm² sr at peak. HET stopping G ~0.6 cm² sr peaking at 5–10 MeV/nuc.

**Dynamic Threshold System** (Mewaldt et al. 2008b, von Rosenvinge et al. 2008): when count rate in a segment threatens dead-time saturation (~10⁵ s⁻¹), thresholds raise on selected segments to retain heavy-ion sensitivity while shedding p/electron events. Lario & Decker (2011) estimate: ~5% chance during prime mission of an event with > 1.5 × 10⁸ cm⁻² sr⁻¹ s⁻¹ proton intensity (>1 MeV).

**Pixel mode**: segments of 1 mm² (1000 μm) or 0.36 mm² (500 μm) — 5% of normal area. Threshold raised so that only protons depositing ≥ 12 MeV (1000 μm) or ≥ 8 MeV (500 μm) trigger. Gives crude proton intensity in extreme events.

#### 4.7 PHASIC ASIC / PHASIC ASIC (pp. 257–259, Fig. 37, Table 6)

- 16 dual-gain PHA channels per chip.
- Charge-sensitive preamp + dual post-amp (high gain × 1.56 buffer + low gain).
- Wilkinson 11-bit ADCs (12-bit overflow). Bipolar shaping, 1.9 μs peak.
- Programmable feedback capacitance 5–75 pF in 5 pF steps.
- High/Low gain ratio 68 or 40, configurable.
- Radiation tolerance: > 100 krad, no latchup below 80 MeV/(mg/cm²).
- Dead time per trigger: 4–67 μs (pulse-height dependent).
- Cross-talk between PHAs < 0.2%.

#### 4.8 Digital Control / 디지털 제어 (pp. 259–263)

MISC (Minimal Instruction Set Computer, Forth-running) on each of 3 telescope boards + DPU board, in RTAX250 FPGA. SRAM per telescope, MRAM on DPU for non-volatile code & lookup tables. On-board species-energy classification using ΔE-E' template tables.

#### 4.9 Gamma Rays and Neutrons / 감마선과 중성자 (p. 260)

HET central stack H3A–H3B (1.2 cm³ silicon volume) detects 0.5–6 MeV γ-rays via Compton scatter or pair production, plus 2–20 MeV neutrons via ²⁸Si(n,p)²⁸Al or ²⁸Si(n,α)²⁵Mg reactions (Mewaldt et al. 1977). Anti-coincidence rejection of charged particles using surrounding segments and H2A/H2B.

#### 4.10 Calibration / 보정 (pp. 262–264)

- LBNL 88-inch cyclotron: heavy-ion cocktail beam (Cr, Cu, Kr, etc.) — Fig. 40 shows tracks identified up to Z=36.
- NSCL Michigan State: ⁵⁸Ni at 160 MeV/nuc; fragmentation produces H–Ni response calibration.
- ¹⁰⁶Ru β-source for electron response; Geant4 simulation cross-check.
- In-flight: dual-gain PHASIC self-calibration via test pulser; cross-cal with EPI-Lo.

---

### Part V: Science Operations and Data Products / 과학 운영과 데이터 산출물

#### 5.1 Operating Modes / 운영 모드 (p. 265)

| Mode | Heliocentric Distance | Description |
|---|---|---|
| **Normal Science Mode** | R ≤ 0.25 AU | Full nominal power; high data collection + EPI-Lo burst data |
| **Low-Rate Science Mode** | R > 0.25 AU | Full power when not downlinking; reduced data rate |
| **Calibration Mode** | Outside 0.25 AU when SEP intensity is sufficient | Sample PHA data to validate species/energy/direction tables |
| **Software Upload** / **Safe** | As needed | Quiet state, awaiting commands |

각 perihelion encounter ~10–11 일 지속. SPP는 24개 highly elliptical orbit (period 168 → 88일). SOC가 N, N+1, N+2 perihelion에 대한 명령 사전계획 운영.

#### 5.2 ISIS Science Operations Center (SOC) / 과학 운영 센터 (pp. 266–270)

- Hosted at University of New Hampshire (UNH), heritage from IBEX SOC.
- L0 (raw), L1 (calibrated counts → flux), L2 (combined products + ancillary), L3+ (model-derived).
- Public quick-look: 60 days post-downlink (looser 6 mo for first 3 orbits).
- Public science data: 6 months post-downlink.
- Final deep archive: 12 mo after end of mission (NASA SPDF).
- Data formats: CDF + SPASE metadata.

#### 5.3 Data Products (Table 7) / 데이터 산출물

| Level | EPI-Lo example | EPI-Hi example |
|---|---|---|
| L0 | electron rates, ion rates, PHA events, TOF-only events | events (Z & E), Z vs. E matrices, direction matrices |
| L1 | particle intensities (cm² sr s MeV/nuc)⁻¹ | particle intensities for LET & HET |
| L1 expanded | (rare) | individual event records |
| L2 | combined SEP/CIR event lists, anisotropy fits | combined; element abundance ratios |

---

## 3. Key Takeaways / 핵심 시사점

1. **Three-pillar SEP science (origins, acceleration, transport)** — ISIS의 세 질문은 SPP의 모든 입자 측정 설계를 직접 결정한다 (FOV, 에너지 범위, 종 분리, 시간 분해능). 어떤 한 측면도 개별적으로 답할 수 없으며 0.25 AU 이내 직접 측정만이 transport 효과를 unmix할 수 있다. ISIS's three pillars — origins, acceleration, transport — drive every design decision. None can be answered alone, and only direct sub-0.25 AU sampling can disentangle transport effects from source/acceleration effects.

2. **Two complementary techniques: TOF (EPI-Lo) + dE/dx-E (EPI-Hi)** — TOF는 낮은 에너지(~20 keV/nuc)에서 우수한 종 분리(특히 ³He/⁴He), dE/dx-E는 ~1 MeV/nuc 이상에서 우수한 elemental separation. 두 기기의 ~1 MeV/nuc 부근 overlap이 cross-calibration의 기반. The two techniques are complementary: TOF excels at low energies (~20 keV/nuc) with isotope resolution, dE/dx-E covers higher energies with strong elemental separation. Their overlap near 1 MeV/nuc is essential for cross-calibration.

3. **Hemispheric sampling without articulation** — EPI-Lo의 80개 fixed aperture와 EPI-Hi의 5개 45° cone은 mechanical scanning 없이 즉각적인 pitch-angle distribution을 제공한다. 이는 ms-to-min 입자 분포 변화를 잡아내는 데 필수적이며, 전통적 회전 기기로는 불가능. The 80 fixed EPI-Lo apertures + 5 EPI-Hi cones provide instantaneous pitch-angle distributions without mechanical articulation — essential for ms-to-min variability that scanning instruments would alias.

4. **Solar Probe is unique: light/UV filtering is the central engineering challenge** — SPP 근접 환경에서 Lyα ~10¹² photons cm⁻² s⁻¹ sr⁻¹은 EPI-Lo MCP에 직접 위협. Pd/Al multi-layer Start foil, double-foil collimator, 그리고 detector geometry constraints가 모두 해결책의 일부. The unique close-Sun environment makes light/UV mitigation (Lyα ~10¹² ph/cm²/s/sr near TPS) the central engineering challenge, addressed via Pd/Al multi-layer foils, double-foil collimators, and aperture geometry.

5. **Q/M-dependent spectral break is the key SEP-acceleration diagnostic** — E_0 ∝ (Q/M)^α 의 측정(Fig. 9, Mewaldt et al. 2005, Li et al. 2009)은 충격파 acceleration 모델을 직접 검증하며, 모든 종을 ~50 MeV/nuc까지 동시 측정해야 의미 있는 fit이 가능. ISIS는 이를 위해 H–Fe 전체 종을 ≥ 6 bins/decade로 cover. Measuring the (Q/M)^α scaling of spectral breaks directly tests shock-acceleration models. Meaningful fits require simultaneous H–Fe coverage at ≥ 6 bins/decade — a primary ISIS requirement.

6. **Heritage-driven design with rad-hard upgrades** — EPI-Hi은 STEREO LET/HET 직계, EPI-Lo는 PEPSSI/RBSPICE/JEDI/EIS 계보. SPP의 100 krad 환경에 맞춰 PHASIC ASIC re-spin (ON-Semi C5N → Aerospace processing), foil thickness 증가, double-foil pinhole mitigation. Both instruments are direct descendants of flight-proven hardware (STEREO LET/HET; MESSENGER/PEPSSI/JEDI/RBSPICE/EIS), with rad-hard upgrades (>100 krad) and Sun-environment hardening.

7. **Dynamic threshold and pixel mode handle 9-decade intensity dynamic range** — peak SEP rate (Lario & Decker 2011) 가능: > 1.5 × 10⁸ cm⁻² sr⁻¹ s⁻¹ for > 1 MeV protons. ISIS는 (a) 동적 threshold raise (heavy-ion 보존), (b) 1 mm² pixel detector segments(0.36 mm²)로 ~5% area에서 saturation 회피하여 peak intensity recovery 가능. Dynamic thresholds + ~5%-area pixel segments handle SEP intensities spanning 9 decades — including extreme events that would otherwise saturate the analog chain.

8. **Coordinated SOC + HSO integration** — ISIS SOC at UNH은 EPI-Lo와 EPI-Hi 데이터를 SWEAP, FIELDS, WISPR 와 결합하고, Solar Orbiter EPD, ground-based observers, NASA archive와의 cross-calibration도 수행. 단일 기기 데이터로는 풀 수 없는 물리 (예: shock crossing 시 plasma + B + particle simultaneous) 를 가능케 함. The ISIS SOC at UNH integrates EPI-Lo + EPI-Hi data with SWEAP, FIELDS, WISPR, plus Solar Orbiter EPD and ground-based observers — enabling coordinated science (e.g., simultaneous plasma + B + particle at a shock crossing) impossible with single-instrument data.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 EPI-Lo TOF Mass Spectrometry / EPI-Lo TOF 질량 분광법

(1) Velocity from TOF — 이온이 거리 L = 8.73 cm를 t [ns] 동안 비행:
$$v = \frac{L}{t}$$

(2) Mass from TOF + SSD energy:
$$m = \frac{2 E_{SSD}}{v^2} = \frac{2 E_{SSD}\,t^2}{L^2}$$

수치적으로(L in cm, t in ns, E in keV): m[amu] ≈ 0.01931 × E[keV] × (t[ns]/L[cm])².

(3) Species track equation in (TOF, E) plane:
$$E_{SSD} = \frac{m}{2}\left(\frac{L}{t}\right)^2 \implies \log E = -2\log t + \log\left(\frac{mL^2}{2}\right)$$

각 종(species)이 (logE, logt) 평면에서 기울기 −2의 직선 트랙을 가짐 (Fig. 21).

### 4.2 EPI-Hi dE/dx vs. E Identification / EPI-Hi dE/dx-E 식별

(4) Bethe-Bloch (non-relativistic, β ≪ 1):
$$-\frac{dE}{dx} = \frac{4\pi N_A r_e^2 m_e c^2 \rho Z_t}{A_t}\,\frac{Z^2}{\beta^2}\,\ln\!\left(\frac{2 m_e c^2 \beta^2}{I}\right)$$

- ρ, Z_t/A_t, I: 매질(Si)의 밀도, atomic number/mass ratio, mean ionization energy (~173 eV for Si)
- Z, β: 입사 입자의 원자번호와 v/c
- N_A r_e²: 4πN_A r_e² m_e c² ≈ 0.307 MeV cm² g⁻¹

(5) Range-Energy approximation for non-relativistic ions:
$$R \approx \frac{1}{C}\,\frac{m^{1-\alpha}\,E^{\alpha}}{Z^2}, \quad \alpha \approx 1.7$$

이로부터 stopping branch에서 ΔE × E' 곡선 트랙 도출.

(6) Track in (ΔE, E') plane (얇은 ΔE 검출기 두께 d, 나머지 E' 잔여):
$$\Delta E \approx \frac{dE}{dx}\bigg|_{E'+\Delta E} \cdot d \propto \frac{Z^2}{\beta^2(E')}\cdot d$$

→ 동일한 species(고정 m, Z)는 단일 곡선; 다른 species는 분리된 곡선 (Fig. 32, 33).

### 4.3 DSA Spectrum near the Sun / 태양 근처 DSA 스펙트럼

(7) Ellison-Ramaty(1985) form (Fig. 9):
$$j(E) = j_0 \, E^{-\gamma}\,\exp\!\left(-\frac{E}{E_0}\right)$$

- γ ≈ 1.3 (Mewaldt et al. 2005, large gradual SEPs)
- E_0 = e-folding energy where exponential cutoff dominates

(8) Q/M scaling (Li et al. 2009, quasi-parallel strong shock):
$$E_0(\text{species}) = E_0(p)\,\left(\frac{Q/M}{Q_p/M_p}\right)^\alpha,\quad \alpha = 1.75 \pm 0.17$$

이는 충격파 가속에서 입자가 더 큰 maximum energy에 도달하려면 자기-여기 turbulence에 더 잘 결합해야 함을 시사 (Q/M↑ → 더 빠른 gyroradius scaling).

### 4.4 Suprathermal Power Law / 초열적 분포의 거듭제곱 법칙

(9) Quiet-time universal tail (Fisk & Gloeckler 2012):
$$f(v) \propto v^{-5} \;\;\Longleftrightarrow\;\; \frac{dN}{dE} \propto E^{-1.5}$$

(differential intensity j ∝ E^{-1.5}; spectral index γ = 1.5)

(10) Connection to compressional second-order Fermi (Fisk & Gloeckler 2012): 압축 영역에서 stochastic acceleration이 보편적 v⁻⁵ asymptotic을 만든다.

### 4.5 Resource Equations / 자원식

(11) Telemetry budget (Table 3):
$$D_{\text{packetized}} = 1.05 \times \left[0.75 \times (D_{\text{Lo,raw}} + D_{\text{Hi,raw}}) + D_{\text{Lo,burst}} + D_{\text{Hi,burst}}\right]$$
ISIS total: 12.007 Gbit/orbit packetized (compression factor ~0.75).

(12) Geometric factor relation:
$$R_{\text{count}} = G \cdot \int j(E,\Omega)\,dE\,d\Omega \approx G\,\bar{j}\,\Delta E$$
→ EPI-Lo G ~ 0.05 cm² sr × 80 pixels; EPI-Hi G ~ 0.5 cm² sr × 5 cones.

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1977 ─ Mewaldt et al.: Solid-state neutral H detection in cosmic rays → SSD heritage for HET
1980 ─ Ahlen: Bohr-Landau straggling theory → sets ³He/⁴He resolution limit
1985 ─ Ellison & Ramaty: DSA spectral form  j ∝ E^-γ exp(-E/E_0)
1989 ─ Mason et al.: First impulsive 3He-rich event spectra (ISEE-3)
1995 ─ Helios-1, IMP-8 SEP comparison highlights need for close-in observations
1999 ─ Reames: Two-class SEP paradigm cemented (gradual vs. impulsive)
2005 ─ Mewaldt et al.: 28 SEP spectral breaks vs. Q/M (E_0 ∝ (Q/M)^α)
2007 ─ McComas et al.: Initial SPP/Solar Probe science definition
2008 ─ STEREO LET (Mewaldt) + HET (von Rosenvinge) launched — direct EPI-Hi heritage
2008 ─ PEPSSI/New Horizons launched — TOF heritage for EPI-Lo
2009 ─ Li et al.: α = 1.75 ± 0.17 for Q/M scaling at quasi-parallel strong shocks
2012 ─ Fisk & Gloeckler: Universal v^-5 quiet-time suprathermal tail
2014 ▶ ISIS PDR (THIS PAPER) — Solar Probe Plus instrument design frozen
2018 ─ Parker Solar Probe launched (12 Aug); first perihelion 35.7 R_S
2019+─ ISIS first SEP measurements inside 0.25 AU
2024 ─ PSP perihelion 9.86 R_S (ISIS active throughout)
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **Fox et al. 2016** (Paper #40, Solar Probe Plus mission overview) | Parent mission paper — defines SPP science goals that ISIS implements at instrument level | Direct: ISIS is one of four SPP investigations |
| **Bale et al. 2016** (Paper #52, FIELDS) | Coordinated B-field & E-field measurements for shock crossings, wave-particle resonance, pitch-angle anisotropy interpretation | High: combined ISIS + FIELDS solves DSA / cyclotron-resonance ambiguity |
| **Vourlidas et al. 2016** (Paper #53, WISPR) | Imaging context for CME shocks, jets, magnetic reconnection sites | High: WISPR identifies CME → ISIS measures particles → SWEAP gives ambient |
| **Kasper et al. 2016** (Paper #54, SWEAP) | Solar wind plasma context — n_p, T_p, v_SW for shock parameters and seed-population identification | Direct: SWEAP suprathermal tails feed into ISIS SEP source attribution |
| **Mewaldt et al. 2008b** (STEREO/LET) | Direct heritage for EPI-Hi LET detector design, dynamic threshold concept, dE/dx-E response simulations | Foundational: EPI-Hi is the next-generation LET |
| **von Rosenvinge et al. 2008** (STEREO/HET) | Direct heritage for EPI-Hi HET detector stack and segmentation | Foundational: EPI-Hi HET inherits design |
| **Mauk et al. 2014** (MMS/EIS) | Most recent EPI-Lo heritage (TOF + position-sensitive MCP design) | Direct: nearly identical TOF concept |
| **Mitchell et al. 2013** (RBSPICE) | Heritage for EPI-Lo electronics and FPGA event logic | Direct: same MISC processor, similar ASIC chain |
| **Reames 2013** (Two SEP sources review) | Theoretical framework for ISIS's three driving questions | Foundational: defines impulsive/gradual paradigm |
| **Fisk & Gloeckler 2012** (v^-5 tail) | Universal quiet-time suprathermal tail — predicted seed population for SEPs | High: ISIS will test universality at < 0.25 AU |
| **Mewaldt et al. 2005** (Q/M scaling) | Empirical (Q/M)^α scaling of spectral breaks that ISIS will refine close to acceleration sites | High: motivates H–Fe full coverage |
| **Li et al. 2009** (α = 1.75) | Theoretical interpretation of Q/M scaling for quasi-parallel strong shock | High: predicts ISIS observable |

---

## 7. References / 참고문헌

### Primary

- McComas, D.J., Alexander, N., Angold, N., Bale, S., Beebe, C., Birdwell, B., et al., **"Integrated Science Investigation of the Sun (ISIS): Design of the Energetic Particle Investigation"**, *Space Science Reviews* **204**, 187–256 (2016). DOI: [10.1007/s11214-014-0059-1](https://doi.org/10.1007/s11214-014-0059-1)

### Companion Mission Papers (same SSR volume)

- Fox, N.J., et al., "The Solar Probe Plus Mission: Humanity's First Visit to Our Star", *Space Sci. Rev.* **204**, 7–48 (2016).
- Bale, S.D., et al., "The FIELDS Instrument Suite for Solar Probe Plus", *Space Sci. Rev.* **204**, 49–82 (2016).
- Kasper, J.C., et al., "Solar Wind Electrons Alphas and Protons (SWEAP) Investigation", *Space Sci. Rev.* **204**, 131–186 (2016).
- Vourlidas, A., et al., "The Wide-Field Imager for Solar Probe Plus (WISPR)", *Space Sci. Rev.* **204**, 83–130 (2016).

### Heritage Instruments

- Mewaldt, R.A., et al., "The Low-Energy Telescope (LET) and SEP Central Electronics for the STEREO Mission", *Space Sci. Rev.* **136**, 285–362 (2008b). DOI: 10.1007/s11214-007-9288-x
- von Rosenvinge, T.T., et al., "The High Energy Telescope for STEREO", *Space Sci. Rev.* **136**, 391–435 (2008).
- McNutt, R.L., et al., "PEPSSI Science Investigation on New Horizons", *Space Sci. Rev.* **140**, 315–385 (2008).
- Mauk, B.H., et al., "The Energetic Particle Detector (EPD) Investigation and the EIS for MMS", *Space Sci. Rev.* (2014). DOI: 10.1007/s11214-014-0055-5
- Mitchell, D.G., et al., "Radiation Belt Storm Probes Ion Composition Experiment (RBSPICE)", *Space Sci. Rev.* **179**, 263–308 (2013). DOI: 10.1007/s11214-013-9965-x
- Andrews, G.B., et al., "MESSENGER EPS", *Space Sci. Rev.* **131**, 523–556 (2007).

### Theory and Background

- Reames, D.V., "Particle Acceleration at the Sun and in the Heliosphere", *Space Sci. Rev.* **90**, 413–491 (1999).
- Reames, D.V., "The Two Sources of Solar Energetic Particles", *Space Sci. Rev.* **175**, 53–92 (2013). DOI: 10.1007/s11214-013-9958-9
- Mason, G.M., "³He-Rich Solar Energetic Particle Events", *Space Sci. Rev.* **130**, 231–242 (2007).
- Ellison, D.C. & Ramaty, R., "Shock Acceleration of Electrons and Ions in Solar Flares", *Astrophys. J.* **298**, 400–408 (1985). DOI: 10.1086/163623
- Mewaldt, R.A., et al., "Solar Energetic Particle Spectral Breaks", AIP Conf. Proc. **781**, 227–232 (2005).
- Li, G., et al., "Shock Geometry and Spectral Breaks in Large SEP Events", *Astrophys. J.* **702**, 998–1004 (2009). DOI: 10.1088/0004-637X/702/2/998
- Fisk, L.A. & Gloeckler, G., "Particle Acceleration in the Heliosphere: Implications for Astrophysics", *Space Sci. Rev.* **173**, 433–458 (2012). DOI: 10.1007/s11214-012-9899-8
- Lario, D. & Decker, R.B., "Estimation of Solar Energetic Proton Mission-Integrated Fluences", *Space Weather* **9**, S11003 (2011). DOI: 10.1029/2011SW000708
- Ahlen, S.P., "Theoretical and Experimental Aspects of the Energy Loss of Relativistic Heavily Ionizing Particles", *Rev. Mod. Phys.* **52**, 121 (1980).
- Paschalidis, N.P., "The Power Spectrum Analyzer", *IEEE Nuclear Science Symposium Conference Record* (2008).
- Mewaldt, R.A., et al., "Neutral Hydrogen Atoms in Cosmic Ray Telescopes", *Space Sci. Instrum.* **3**, 231–242 (1977).
