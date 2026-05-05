---
title: "High Energy Neutral Atom (HENA) Imager for the IMAGE Mission"
authors: ["D. G. Mitchell", "S. E. Jaskulek", "C. E. Schlemm", "E. P. Keath", "R. E. Thompson", "B. E. Tossman", "J. D. Boldt", "J. R. Hayes", "G. B. Andrews", "N. Paschalidis", "D. C. Hamilton", "R. A. Lundgren", "E. O. Tums", "P. Wilson IV", "H. D. Voss", "D. Prentice", "K. C. Hsieh", "C. C. Curtis", "F. R. Powell"]
year: 2000
journal: "Space Science Reviews"
doi: "10.1023/A:1005207308094"
topic: Space_Weather
tags: [ENA, ring_current, IMAGE_mission, HENA, magnetospheric_imaging, charge_exchange, instrument_paper, geocorona, MCP, SSD, TOF, substorm_injection]
status: completed
date_started: 2026-04-25
date_completed: 2026-04-25
---

# 72. High Energy Neutral Atom (HENA) Imager for the IMAGE Mission / IMAGE 임무용 고에너지 중성원자 영상장치

---

## 1. Core Contribution / 핵심 기여

Mitchell et al. (2000) describe the design, calibration, and operational concept of the **High Energy Neutral Atom (HENA) imager** — the highest-energy member of IMAGE's three-instrument Neutral Atom Imaging (NAI) suite, covering ~10 keV/nuc up to ~500 keV/nuc. HENA is a slit camera with a 90°×120° field of view that simultaneously runs two complementary back-plane detectors: a position-sensitive **microchannel plate (MCP)** array and a **pixelated solid-state detector (SSD)** array. Combining a charged-particle-sweeping collimator (±4 kV serrated plates), an ultra-thin entrance foil (Si–polyimide–C, 14.5 μg/cm² total), front-foil/back-foil secondary-electron timing, and triple-coincidence logic, HENA delivers angular resolution of ~4°×6° at high energy (>80 keV/nuc), velocity resolution of ~50 km/s (1 ns TOF), energy resolution ΔE/E ≤ 0.25, mass discrimination among H/He/O above 30 keV/nuc, and a geometric factor of ~1.6 cm²·sr (oxygen). With a 2-min spin-cadence accumulation matching IMAGE's spacecraft spin period, HENA produces the first minute-resolution all-sky movies of energetic ENA emission from the storm-time ring current and substorm ion-injection regions.

본 논문은 IMAGE 임무의 세 가지 중성원자 영상장치(NAI) 중 최고에너지 대역(약 10 keV/nuc 부터 약 500 keV/nuc)을 담당하는 **HENA(High Energy Neutral Atom) 영상장치**의 설계·보정·운영 개념을 기술한다. HENA는 90°×120° 시야의 슬릿 카메라로, 위치민감 마이크로채널판(MCP) 배열과 픽셀화된 반도체 검출기(SSD) 배열을 후방 평면에서 동시에 운용한다. ±4 kV 톱니형 편향판으로 구성된 하전입자 청소 collimator, Si–polyimide–C 3중 박막(총 14.5 μg/cm²)의 입구 foil, 전·후방 foil의 이차전자 타이밍, 그리고 삼중 일치(coincidence) 논리를 결합하여 80 keV/nuc 이상에서 약 4°×6°의 각해상도, 1 ns TOF에 해당하는 약 50 km/s의 속도해상도, ΔE/E ≤ 0.25의 에너지해상도, 30 keV/nuc 이상에서 H/He/O 종 식별, 산소 기준 약 1.6 cm²·sr의 기하 인자를 달성한다. IMAGE의 2분 spin 주기와 맞물려, HENA는 지자기폭풍의 링 전류와 서브스톰 주입 영역에서 방출되는 고에너지 ENA를 분 단위 시간해상도로 영상화하는 첫 사례를 제공한다.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction and Science Requirements / 서론과 과학 요구사항 (pp. 67–73)

**English** — The paper opens by explaining ENA imaging's physical basis: a charge-exchange collision between a trapped energetic ion and a cold exospheric neutral produces an energetic neutral atom that follows a straight, magnetic-field-independent trajectory and can therefore be detected at large distance to form a global image of the emitting region. Because ENA flux at a remote observer is

$$J_{ENA}(E,\hat{\Omega}) = \int_{LOS} \sigma_{cx}(E)\, n_H(\vec{r})\, j_{ion}(E,\hat{\Omega},\vec{r})\, ds,$$

each pixel encodes a line-of-sight integral that — with knowledge of the geocoronal density n_H and the cross-section σ_cx — can be inverted to recover the underlying 3-D ion distribution. IMAGE's three NAI sensors split the energy range: LENA (0.01–0.5 keV), MENA (1–30 keV), and HENA (~10 to 500 keV/nuc). HENA's two top-level science requirements are (1) imaging the inner magnetosphere (including ring current) on a 300-s timescale and (2) resolving major species contributing to neutral-atom flux. Figure 2 shows a Roelof-style simulated ENA image during an active ring-current/ion injection, labeling the ring current at low and pre-injection altitudes, ion-injection boundaries (morning, evening, night), and a near-Earth plasma-sheet injection. Figure 3 reproduces the ISEE/MEPI Sept-29-1978 storm dataset and a HENA-resolution simulation: peak pixels exceed 10⁴ counts/pixel and the lowest discernible level is ~50 counts/pixel, demonstrating that HENA can follow magnetospheric dynamics at the 2-min spin cadence. A summary table maps each of seven IMAGE objectives (global morphology, plasma source ID, plasma energization, substorms/convection, trapped radiation, auroral source, heliospheric shocks) to a HENA data product.

**한국어** — 본 절은 ENA 영상화의 물리적 기반을 설명하며 시작한다. 자기권에 갇혀 있던 고에너지 ion과 차가운 외기권 중성원자의 전하교환 충돌은 자기장과 무관한 직선 경로를 따르는 고에너지 중성원자를 생성하므로, 멀리 떨어진 관측자가 이를 검출하여 방출 영역의 전역 영상을 구성할 수 있다. 원격 관측자에서의 ENA 플럭스는

$$J_{ENA}(E,\hat{\Omega}) = \int_{시선} \sigma_{cx}(E)\, n_H(\vec{r})\, j_{ion}(E,\hat{\Omega},\vec{r})\, ds$$

로, 각 픽셀은 시선(line-of-sight) 적분 정보를 담고 있으며 — geocorona 밀도 n_H와 단면적 σ_cx를 알고 있다면 — 이로부터 3차원 ion 분포를 역산할 수 있다. IMAGE의 세 NAI 장비는 에너지 영역을 LENA(0.01–0.5 keV), MENA(1–30 keV), HENA(약 10–500 keV/nuc)로 분할한다. HENA의 두 가지 최상위 과학 요구는 (1) 링 전류를 포함한 내부 자기권의 300초 시간 척도 영상화, (2) 중성원자 플럭스에 기여하는 주요 종(species)의 분해이다. Figure 2는 활성 링 전류/ion 주입 시 모의된 ENA 영상을 Roelof 스타일로 보여주며 저고도 및 사전 주입 링 전류, 주입 경계(아침·저녁·야간), 근지구 플라즈마 시트 주입 영역 등을 표시한다. Figure 3은 ISEE/MEPI 1978년 9월 29일 폭풍 데이터와 HENA 해상도의 모의 영상을 보여주는데, 정점 픽셀은 10⁴ counts/pixel을 넘고 식별 가능한 최저 수준은 약 50 counts/pixel이어서 HENA가 2분 spin 주기로 자기권 동역학을 추적할 수 있음을 보여준다. 표는 IMAGE의 7가지 목표(전역 형태, 플라즈마 소스 식별, 에너지화, 서브스톰/대류, 갇힌 방사선, 오로라 소스, 헬리오스피어 충격)를 HENA 데이터 산출물에 대응시킨다.

### Part II: HENA Sensor Architecture / HENA 센서 구조 (Section 2.1, pp. 73–84)

**English** — Table 2.1.1 catalogs HENA's measurement specifications: energy range 20–500 keV/nuc, ΔE/E ≤ 0.25, velocity resolution ~50 km/s (1 ns TOF), composition (H, He, O, heavies), FOV 120°×90°, angular resolution ~4°×6° at high energy and degrading to <8°×<12° below ~40 keV/nuc due to foil scattering, time resolution 2 s for PHA events and 2 min (1 spin) for images, geometric factor G·ε ≈ 1.6 (O) and ≈ 0.3 (H), Ly-α rejection <3×10⁻⁷, electron and ion rejection ~10⁻⁵, dynamic range 10⁷, and pixel array dimensions 4 masses × 8 energies × 40 angles × 120 angles. The sensor head (Figure 4) consists of: (i) an electrically-biased serrated magnesium **collimator** with adjacent plates at ±4 kV (commandable up to ±6 kV) to sweep charged particles up to 500 keV/e into the plate walls (laboratory accelerator tests at Figure 5 show >4-orders-of-magnitude rejection); (ii) a thin **front foil** (Si–polyimide–C, 6.5+7.0+1.0 μg/cm², supplied by Luxel) that produces secondary electrons accelerated by a local E-field perpendicular to the foil, then steered by wires/shaped electrodes onto a **1-D imaging start MCP**; (iii) the original ENA continuing through the foil with some scattering, hitting either a second polyimide+C foil in front of the **2-D imaging stop MCP**, or a **pixelated SSD** (10×24 pixels, ~5×10 cm² total area, MICRON detectors); (iv) a **coincidence/SSD-stop MCP** triggered by back-scattered secondary electrons within <40 ns travel-time window. The collimator FOV is 90° azimuth × 120° elevation, with elevation θ defined in planes containing the collimator fins and azimuth φ in the spin plane. The deflector plates are serrated to inhibit forward-scattering of incident charged particles into the detector.

**한국어** — Table 2.1.1은 HENA의 측정 사양을 정리한다: 에너지 범위 20–500 keV/nuc, ΔE/E ≤ 0.25, 속도 해상도 약 50 km/s (1 ns TOF), 조성(H, He, O, heavies), FOV 120°×90°, 고에너지 각해상도 약 4°×6°, 약 40 keV/nuc 이하에서는 박막 산란으로 <8°×<12°까지 저하, PHA 이벤트 시간해상도 2초·영상 시간해상도 2분(1 spin), 기하 인자 G·ε ≈ 1.6 (O) 및 ≈ 0.3 (H), Ly-α 거부율 <3×10⁻⁷, 전자·이온 거부율 약 10⁻⁵, 동적 범위 10⁷, 픽셀 배열 4질량 × 8에너지 × 40각도 × 120각도. 센서 헤드(Figure 4)는 다음으로 구성된다: (i) ±4 kV(명령 시 ±6 kV까지)가 인접 판에 인가되는 톱니형 마그네슘 **collimator**가 500 keV/e 이하의 하전입자를 판 벽으로 휩쓸어내며, Figure 5의 가속기 시험은 4자릿수 이상의 거부율을 보여준다; (ii) **전방 foil**(Si–polyimide–C, 6.5+7.0+1.0 μg/cm², Luxel 제공)이 이차전자를 생성하고, foil에 수직인 국소 E-field가 가속한 뒤 wire와 성형 전극이 **1-D 영상 시작 MCP**로 유도; (iii) 원래의 ENA는 약간의 산란을 거쳐 foil을 통과한 뒤 **2-D 영상 정지 MCP** 앞 두 번째 박막 또는 **픽셀화된 SSD**(10×24 픽셀, 약 5×10 cm², MICRON 제공)에 도달; (iv) **일치/SSD-stop MCP**가 후방 산란 이차전자를 <40 ns 이내에 수집한다. Collimator의 FOV는 azimuth 90° × elevation 120°이며, elevation θ는 collimator 핀이 포함된 평면, azimuth φ는 spin 평면에 정의된다. 톱니형 deflector 판은 입사 하전입자의 전방 산란을 억제한다.

### Part III: Measurement Technique and Mass Determination / 측정 원리와 질량 결정 (Sections 2.1.1–2.1.3, pp. 76–78)

**English** — In the **MCP back-plane mode**, an incoming neutral penetrates the front foil, ejects 1–10 secondary electrons (Meckbach et al. 1975 spectrum), the start-anode position records the entrance-slit location and the MCP timing pulse defines t₀; the same neutral then exits the back foil ~10 cm downstream, ejecting more secondaries that are accelerated into the 2-D MCP, registering position (x,y) and timing t₁. TOF = t₁−t₀ together with the path length d give velocity v = d/TOF, and combined with PHA on the front and back MCPs the species can be inferred from the secondary-electron yield, which scales roughly as Z (oxygen produces several times more secondaries than hydrogen, see Figure 6 — at 31 keV/nuc the Front-PH vs Back-PH plot separates O from H clearly). In the **SSD back-plane mode**, the neutral strikes one of 240 SSD pixels (each ~4 mm square, the array is 10×24 = 240 pixels over 5×10 cm²), depositing energy E (1/2 m v²) measured by pulse-height analysis on the AMTEK A225 hybrid preamps; secondary electrons from the SSD surface travel back 6–10 ns to the C/S MCP for the stop timing. The combined SSD energy + TOF give mass m = 2E·TOF²/d² and therefore species ID. SSD energy resolution is ~7 keV FWHM, providing ΔE/E = 0.23 at 30 keV; the practical 30 keV minimum proton energy is set jointly by foil energy loss and the 7 keV electronic threshold. Both back-planes share the same start-imaging system so the two are simultaneously operating, providing redundancy and complementary mass determinations.

**한국어** — **MCP 후방 평면 모드**에서, 입사 중성원자는 전방 박막을 투과하면서 1–10개의 이차전자를 방출하고(Meckbach et al. 1975 스펙트럼), 시작 anode 위치가 슬릿 입구 위치를, MCP 타이밍 펄스가 t₀를 기록한다. 같은 중성원자는 약 10 cm 떨어진 후방 박막을 통과하면서 추가 이차전자를 방출하고, 이는 2-D MCP에 가속되어 위치 (x,y)와 t₁을 기록한다. TOF = t₁−t₀와 경로 길이 d로부터 v = d/TOF, 그리고 전·후 MCP의 PHA를 결합하면 이차전자 수율(대략 Z에 비례; 산소는 수소보다 수 배 더 많은 이차전자 생성, Figure 6: 31 keV/nuc에서 Front-PH vs Back-PH 산점도가 O와 H를 명확히 분리)로부터 종(species)을 추론한다. **SSD 후방 평면 모드**에서는 중성원자가 240개의 SSD 픽셀 중 하나(각 ~4 mm², 10×24 배열로 총 5×10 cm²)에 부딪혀 펄스 높이 분석(AMTEK A225 하이브리드 전치증폭기)으로 E (1/2 m v²)를 측정한다. SSD 표면에서의 이차전자는 6–10 ns의 비행 후 C/S MCP에 도달하여 정지 타이밍을 제공한다. SSD 에너지와 TOF의 결합으로 질량 m = 2E·TOF²/d², 따라서 종 식별이 가능하다. SSD 에너지해상도는 약 7 keV FWHM로 30 keV에서 ΔE/E = 0.23을 제공하며, 실용 최저 양성자 에너지 30 keV는 foil 에너지손실과 7 keV 전자 임계의 결합으로 결정된다. 두 후방 평면은 동일한 시작 영상 시스템을 공유하여 동시에 작동하며, 중복성과 상보적 질량 결정을 제공한다.

### Part IV: Foils, UV Sensitivity, and Background Rejection / 박막, UV 감도와 배경 거부 (Sections 2.1.4 & 2.1.6, pp. 78–82)

**English** — The two foils — a Si–polyimide–C front foil (6.5 + 7.0 + 1.0 μg/cm²) and a polyimide–C back foil (7.0 + 5.0 μg/cm²) — serve dual purposes: secondary-electron production for timing, and Ly-α attenuation. Measured Ly-α transmittance of the front foil is 1.5×10⁻³ (Hsieh et al. 1980; Powell 1993; Powell et al. 1990); since the photoelectron yield from a C exit surface under Ly-α is ~1 %, with 0.32–1.0×10¹⁰ Ly-α photons/sec incident (geometric factor 4 cm²·sr, 10–30 kR Earth-glow brightness; Rairden et al. 1986), the design budget is ≤10⁵ Ly-α-induced secondary electrons/sec at the start MCP and ≤10³/sec at the stop MCP. The back foil reduces Ly-α to the 2-D MCP by an additional factor ~100. The HENA event-validation logic requires (i) a start MCP pulse, (ii) a stop pulse (back-plane MCP or SSD), and (iii) a coincidence pulse from C/S MCP — all within a TOF window <100 ns and coincidence window ~40 ns. The accidental false-coincidence rate from uncorrelated background is

$$R_{false} \approx 4\times 10^{-15} \cdot R_{start} \cdot R_{stop} \cdot R_{coinc}$$

so even with R_start = R_coinc ≈ 1.5×10⁵ /s and R_stop ≈ 1.5×10³ /s the EUV-induced false rate is ~1 event/s in low-foreground regions. By imposing a minimum TOF >6 ns (corresponding to a ~400 keV proton), penetrators (cosmic rays, >2 MeV magnetospheric electrons) producing only a single secondary at a time are further rejected. With the SSD's intrinsic insensitivity to EUV photons, the SSD-side false rate is even lower. A motorized **shutter** (Figure 10) with 90°×30° clear FOV is closed for 1 s when the Sun enters the FOV during each 2-min spin (over half the mission), providing additional sunlight mitigation; the shutter also carries radioactive calibration sources.

**한국어** — 두 박막 — Si–polyimide–C 전방 박막(6.5+7.0+1.0 μg/cm²)과 polyimide–C 후방 박막(7.0+5.0 μg/cm²) — 은 이중 역할을 한다: 타이밍을 위한 이차전자 생성과 Ly-α 감쇠. 전방 박막의 측정된 Ly-α 투과율은 1.5×10⁻³ (Hsieh et al. 1980; Powell 1993; Powell et al. 1990)이고, C 출사면의 Ly-α 광전자 수율은 약 1 %이며, 입사 Ly-α 광자 플럭스 0.32–1.0×10¹⁰ /s (기하 인자 4 cm²·sr, 지구 광휘 10–30 kR, Rairden et al. 1986) 조건에서 설계 예산은 시작 MCP에서 ≤10⁵ Ly-α 유도 이차전자/s, 정지 MCP에서 ≤10³ /s이다. 후방 박막은 2-D MCP로의 Ly-α를 추가로 약 100배 감쇠한다. HENA의 이벤트 유효성 논리는 (i) 시작 MCP 펄스, (ii) 정지 펄스(후방 MCP 또는 SSD), (iii) C/S MCP의 일치 펄스를 모두 TOF <100 ns 그리고 일치 창 약 40 ns 내에서 요구한다. 무상관 배경에 의한 우연 일치 비율은

$$R_{false} \approx 4\times 10^{-15} \cdot R_{start} \cdot R_{stop} \cdot R_{coinc}$$

이며, R_start = R_coinc ≈ 1.5×10⁵ /s, R_stop ≈ 1.5×10³ /s 조건에서도 EUV 유도 false 비율은 저 전경 영역에서 약 1/s에 불과하다. 최소 TOF >6 ns (~400 keV 양성자에 해당)를 부과하여 penetrator (우주선, >2 MeV 자기권 전자)로서 단일 이차전자만 생성하는 입자도 추가로 거부한다. SSD는 EUV 광자에 본질적으로 둔감하므로 SSD측 false 비율은 더욱 낮다. 모터 구동 **셔터**(Figure 10, 90°×30° 개방 FOV)는 매 2분 spin 중 태양이 FOV에 들어올 때 1초 동안 닫혀 추가 태양광 완화를 제공한다; 셔터에는 방사성 보정 선원도 장착되어 있다.

### Part V: Angular Resolution and Foil Scattering / 각해상도와 박막 산란 (Section 2.1.5, pp. 79–82)

**English** — Each detected ENA is binned into a sky map according to its species, velocity, and trajectory; high-velocity images use 3° bins, lower-velocity images use 6° bins, both finer than the intrinsic FWHM (oversampling). Calibration FWHMs (θ_FWHM, φ_FWHM) for hydrogen are: 100 keV → (5.25–6°, 2.5–4°); 70 keV → (6–9°, 3–5°); 50 keV → (9–12°, 6–8° at the helium-equivalent velocity); 40 keV → (10–13°, 7–9°). For the MCP back plane, φ-resolution is dominated by front-foil scattering at all energies; θ-resolution is foil-scattering-dominated below ~60 keV and electron-optics-limited above. Figure 9 models the secondary-electron-induced position spread, predicting ~2–4 mm FWHM for single-electron events. For the SSD back plane, the limiting factors at high energy are SSD pixel size (~4 mm) and electron-optics smear; the best-case angular resolution is ~3°×4° FWHM for oxygen, degraded to ~3°×6° for hydrogen. These match or modestly exceed the mission requirement of <0.7 in ΔE/E and the 8°×6° angular target. The empirical scaling roughly follows θ_FWHM ∝ E^(−1/2) below 60 keV.

**한국어** — 검출된 ENA는 종, 속도, 궤적에 따라 천구 지도에 binning된다; 고속 영상은 3° bin, 저속 영상은 6° bin을 사용하여 본질적 FWHM보다 미세하게 oversampling한다. 수소에 대한 보정 FWHM (θ_FWHM, φ_FWHM)는 다음과 같다: 100 keV → (5.25–6°, 2.5–4°); 70 keV → (6–9°, 3–5°); 50 keV → (9–12°, 6–8°, 헬륨 동등 속도 기준); 40 keV → (10–13°, 7–9°). MCP 후방 평면에서 φ 해상도는 모든 에너지에서 전방 foil 산란이 지배하며, θ 해상도는 60 keV 이하에서는 foil 산란이, 이상에서는 전자 광학이 지배한다. Figure 9는 이차전자 유도 위치 분산을 모델링하며 단일 전자 이벤트에 대해 약 2–4 mm FWHM을 예측한다. SSD 후방 평면의 고에너지 한계 요인은 SSD 픽셀 크기(~4 mm)와 전자 광학 smear이며, 최선의 각해상도는 산소에서 약 3°×4° FWHM, 수소에서 약 3°×6°로 저하된다. 이는 임무 요구사항(ΔE/E < 0.7, 각도 8°×6°)을 충족하거나 약간 상회한다. 경험적 스케일링은 60 keV 이하에서 θ_FWHM ∝ E^(−1/2) 형태에 가깝다.

### Part VI: Calibration at GSFC Van de Graaff / GSFC Van de Graaff에서의 보정 (Section 3, pp. 89–97)

**English** — HENA was calibrated at the GSFC Van de Graaff accelerator facility, mounted on an articulation mechanism inside a 36-inch vacuum chamber. Beams of O, N, He, and H were accelerated to 10–200 keV/nuc and the angular and energy responses were measured for both back planes. **MCP back plane**: at 50 keV/nuc oxygen (i.e., 800 keV total energy) the FWHMs are θ ≈ 5° and φ ≈ 3.3° (Figures 14–16); at 100 keV hydrogen the FWHMs are similar (Figures 17–18). **SSD back plane**: at 100 keV hydrogen, energy histogram (Figure 20) shows a clean ~7 keV FWHM peak around 100 keV with a small low-energy tail (a few quirky pixels) and a tiny high-energy tail (3 pixels). The corrected TOF distribution (Figure 21) peaks at ~13 ns with a ~1 ns FWHM main peak and a secondary peak at ~21 ns from "bouncing" secondary electrons at the outer-edge SSD pixels (offsets adding ~8 ns). When edge pixels are excluded (Figure 22), the residual high-TOF tail drops by >300×. SSD calibration also identified four pixel-quality categories (Figure 23): 'sometimes low energy', 'erroneous pixel ID', 'spuriously high energy', and 'excessively noisy'. All such pixel-level idiosyncrasies are tracked in flight software, and in-flight calibration uses the shutter-mounted radioactive source plus an electronic calibrator. The instrument generally meets or modestly exceeds requirements for φ; θ meets or falls slightly short.

**한국어** — HENA는 GSFC Van de Graaff 가속기 시설의 36인치 진공 챔버 내 articulation 기구에 장착되어 보정되었다. O, N, He, H 빔이 10–200 keV/nuc로 가속되어 두 후방 평면 모두의 각도·에너지 응답이 측정되었다. **MCP 후방 평면**: 50 keV/nuc 산소(총 에너지 800 keV)에서 FWHM은 θ ≈ 5°, φ ≈ 3.3° (Figures 14–16); 100 keV 수소도 유사 (Figures 17–18). **SSD 후방 평면**: 100 keV 수소에서 에너지 히스토그램(Figure 20)은 약 100 keV 주변에 ~7 keV FWHM의 깨끗한 피크를 보이며 저에너지 꼬리(소수의 quirky 픽셀)와 미세한 고에너지 꼬리(3 픽셀)를 동반. 보정된 TOF 분포(Figure 21)는 약 13 ns에서 정점을 이루고 주 피크 FWHM은 약 1 ns이며, 외각 SSD 픽셀에서 이차전자가 "튀어" 약 8 ns 추가됨에 따른 21 ns 부근의 보조 피크가 있다. 경계 픽셀을 제외하면(Figure 22) 잔류 고-TOF 꼬리는 300배 이상 감소한다. SSD 보정은 또한 4가지 픽셀 품질 카테고리를 식별했다(Figure 23): '간헐 저에너지', '잘못된 픽셀 ID', '비정상 고에너지', '과도한 잡음'. 이러한 픽셀 수준의 비정상성은 비행 소프트웨어에서 추적되며, 비행 중 보정은 셔터 장착 방사성 선원과 전자 보정기를 함께 사용한다. 기기는 일반적으로 φ 요구사항을 충족하거나 약간 상회하고, θ는 충족하거나 약간 미달한다.

### Part VII: Data Products, Operations, and Hardware Specifications / 데이터 산출물·운영·하드웨어 사양 (Section 4 + Appendix A, pp. 96–112)

**English** — Section 4 outlines the **Level-0/1/2 data hierarchy**: Level-0 is time-ordered raw packets archived daily at the Goddard SMOC and NSSDC; Level-1 is "Browse Products" (CDF format, ISTP-compliant) — quick-look images posted within hours, finalized within 3 days; Level-2 and higher are forward-modeled or image-inverted ion-distribution products, ring-current flux, Dst-style diagnostics, generated by the HENA team and other investigators. Image inversion software (Roelof & Skinner 1999) is delivered alongside the data. **Telemetry**: 38.4 kbps to the CIDP, 135 000 kbits per 13.5-hr orbit, with 26 parallel image planes (75 Kbytes SRAM, double-buffered to 150 Kbytes) plus 4 PHA event categories. Image data are log-compressed to 8 bits/pixel and Rice-encoded. **Hardware**: total mass 19.05 kg (sensor 12.87 kg, MEU 5.325 kg, blanket 0.28 kg, cables 0.859 kg); orbit-average power 14.6 W (peak <18.7 W) with normal-ops 13.9 W, calibration 13.7 W, ambient-ops 10.3 W, decontamination 15 W; operational temperature 0 to +20 °C preferred; total ionizing dose target 50 krad (electronics rated 100 krad, factor-of-2 RDM). **Operational modes** (Figure 12 state diagram): Normal, Calibration, High-power (acoustic-cover release), Sleep, POR, plus reserved SSD-off/MCP-on and SSD-on/MCP-off. **DPU**: Harris RTX2010 microprocessor running FORTH; 70 % of flight code inherited from Cassini/MIMI INCA; capable of analyzing ~6000 ENA events/sec.

**한국어** — Section 4은 **Level-0/1/2 데이터 계층**을 설명한다: Level-0는 시간순 원시 패킷, GSFC SMOC와 NSSDC에 매일 보관; Level-1은 "Browse Products" (CDF 포맷, ISTP 준수) — 수 시간 내 게시되는 quick-look 영상이 3일 내 최종화; Level-2와 그 이상은 forward-modeling 또는 image-inversion으로 도출한 ion 분포, 링 전류 플럭스, Dst 형 진단치이며 HENA 팀과 다른 연구자가 생성한다. 영상 역산 소프트웨어(Roelof & Skinner 1999)는 데이터와 함께 제공된다. **Telemetry**: CIDP로 38.4 kbps, 13.5시간 궤도당 135 000 kbits, 26개의 병렬 영상 평면(75 KB SRAM, double-buffer로 150 KB)과 4개의 PHA 이벤트 카테고리. 영상은 픽셀당 8비트로 log 압축 후 Rice 인코딩된다. **하드웨어**: 총 질량 19.05 kg (센서 12.87 kg, MEU 5.325 kg, 블랭킷 0.28 kg, 케이블 0.859 kg); 궤도 평균 전력 14.6 W (최대 <18.7 W); 정상 운용 13.9 W, 보정 13.7 W, ambient 10.3 W, 오염 제거 15 W; 운용 온도 0 ~ +20 °C 선호; 총 이온화 누적선량 목표 50 krad (전자장비는 100 krad 등급, RDM 2배). **운용 모드**(Figure 12 상태도): Normal, Calibration, High-power(음향 커버 해제), Sleep, POR 및 SSD-off/MCP-on, SSD-on/MCP-off 보존 모드. **DPU**: Harris RTX2010 마이크로프로세서가 FORTH를 실행; 비행 코드의 70 %는 Cassini/MIMI INCA에서 상속; 약 6000 ENA 이벤트/s 분석 가능.

---

## 3. Key Takeaways / 핵심 시사점

1. **Triple-coincidence + foil + collimator architecture solves the multi-decade challenge of imaging a faint neutral signal in a sea of charged particles and EUV photons.** — HENA's combination of ±4 kV serrated deflection plates (>10⁵ ion rejection), 1.5×10⁻³ Ly-α-transmittance front foil, and start/stop/coincidence MCP triple coincidence yields a false-event rate of ~4×10⁻¹⁵ × R_start·R_stop·R_coinc, robust enough to image ENA fluxes inside the trapped-radiation belts. / 삼중 일치 + foil + collimator 구조는 강한 하전입자·EUV 배경 속에서 희미한 중성 신호를 영상화하는 수십 년의 과제를 해결한다. ±4 kV 톱니 편향판(이온 거부 >10⁵), Ly-α 투과율 1.5×10⁻³의 전방 박막, 시작/정지/일치 MCP 삼중 일치를 결합하여 false 이벤트 비율 약 4×10⁻¹⁵ × R_start·R_stop·R_coinc을 달성, trapped radiation 벨트 내에서도 ENA를 영상화할 수 있다.
2. **Dual MCP and SSD back-planes provide redundancy and complementary mass determination.** — MCP-side mass from front/back PH ratio (Figure 6) covers the case of SSD failure; SSD-side mass from E (PHA) + TOF gives the primary species ID with ΔE/E ≤ 0.25 from 30 keV/nuc upward. / 이중 후방 평면(MCP+SSD)은 중복성과 상보적 질량 결정을 제공한다. MCP 측 질량은 front/back PH 비(Figure 6)로 SSD 고장 대비; SSD 측 질량은 E (PHA) + TOF로 30 keV/nuc 이상에서 ΔE/E ≤ 0.25의 주요 종 식별을 담당한다.
3. **Below ~60 keV/nuc, foil scattering — not optics — limits angular resolution.** — Calibration shows θ_FWHM rising from ~5° at 100 keV to ~10–13° at 40 keV (hydrogen), following an approximate E^(−1/2) law set by multiple Coulomb scattering in the C-polyimide-C entrance foil. This sets the floor on inversion fidelity for low-energy ring-current ions. / 약 60 keV/nuc 이하에서는 광학이 아닌 박막 산란이 각해상도를 제한한다. 보정 결과 θ_FWHM은 100 keV에서 약 5°에서 40 keV에서 약 10–13°(수소)로 상승하며, 이는 C-polyimide-C 전방 박막에서의 다중 Coulomb 산란이 설정하는 약 E^(−1/2) 법칙을 따른다. 이는 저에너지 링 전류 ion에 대한 역산 충실도의 하한을 결정한다.
4. **2-min spin cadence + ~1 cm²·sr geometric factor → minute-resolution storm dynamics.** — A simulated September-29-1978 storm (Figure 3) yields >10⁴ counts/pixel in the brightest emission and ≥50 counts/pixel at the lowest discernible level, demonstrating S/N adequate to follow ring-current development without integration over the storm. / 2분 spin 주기 + 약 1 cm²·sr 기하 인자 → 분 단위 폭풍 동역학 추적. 모의된 1978년 9월 29일 폭풍(Figure 3)은 최대 발광 영역에서 >10⁴ counts/pixel, 최저 식별 수준에서 ≥50 counts/pixel을 산출하여, 폭풍 동안 적분 없이 링 전류 발달을 추적할 수 있는 S/N을 보여준다.
5. **HENA inherits from Cassini/MIMI INCA, accelerating mission readiness.** — 70 % of flight code, the Harris RTX2010 DPU, the FORTH interpretive language, the wax-actuated acoustic door, and many calibration techniques were directly transferred from Mitchell et al. (1993, 1998) INCA. This heritage turned a complex magnetospheric-imaging instrument into a low-risk delivery for the IMAGE Medex programme. / HENA는 Cassini/MIMI INCA로부터 계승하여 임무 준비 속도를 높인다. 비행 코드의 70 %, Harris RTX2010 DPU, FORTH 인터프리터 언어, 왁스 구동 음향 도어, 다수의 보정 기법이 Mitchell et al. (1993, 1998)의 INCA에서 직접 이전되었다. 이러한 유산은 복잡한 자기권 영상 장비를 IMAGE Medex 프로그램의 저위험 납품으로 전환시켰다.
6. **The 90°×120° FOV combined with spacecraft spin yields a ~3π-sr all-sky map every spin.** — The slit camera's instantaneous FOV is filled in azimuth as the spacecraft rotates with a 2-min period, allowing global coverage every spin without the cost or risk of a scan platform. / 90°×120° FOV와 우주선 spin의 결합으로 spin마다 약 3π sr의 전체 천구 지도를 산출한다. 슬릿 카메라의 순간 FOV는 2분 주기 spin에 따라 azimuth로 채워지며, scan 플랫폼의 비용·위험 없이 spin마다 전역 커버리지를 제공한다.
7. **HENA establishes a quantitative framework — forward modeling and image inversion — that downstream missions adopt.** — The Roelof & Skinner (1999) inversion code accompanying HENA Level-2 products formalized the techniques that subsequently powered TWINS stereo inversion, Cassini/INCA Saturn ENA, and JEDI/JUNO Jupiter ENA analyses. / HENA는 후속 임무들이 채택할 정량적 프레임워크 — forward modeling과 image inversion — 를 확립한다. HENA Level-2 산출물에 동반된 Roelof & Skinner (1999) 역산 코드는 이후 TWINS 스테레오 역산, Cassini/INCA 토성 ENA, JEDI/JUNO 목성 ENA 분석에 활용된 기법을 공식화했다.
8. **Operational design — shutter, decontamination heaters, redundant detectors, EEPROM-based flight code — minimizes single-point failures over the 2-year mission.** — Spring-loaded fail-safe shutter, latching wax actuator for the acoustic door, EDAC-protected SRAM, "Replacement Heater" survival mode, and the Sleep/POR/SSD-off/MCP-off branches in the state diagram (Figure 12) reflect mature instrument-engineering practice. / 운용 설계 — 셔터, 오염제거 히터, 중복 검출기, EEPROM 기반 비행 코드 — 가 2년 임무 동안 단일 실패점을 최소화한다. 스프링 fail-safe 셔터, 음향 도어용 latch 왁스 액추에이터, EDAC 보호 SRAM, "Replacement Heater" 생존 모드, 상태도(Figure 12)의 Sleep/POR/SSD-off/MCP-off 분기는 성숙한 기기 공학 실무를 반영한다.

---

## 4. Mathematical Summary / 수학적 요약

### (i) ENA forward problem / ENA 정문제

The expected differential ENA flux at the spacecraft along a unit direction $\hat{\Omega}$ at energy $E$:

$$\boxed{\,J_{ENA}(E,\hat{\Omega}) = \int_0^\infty \sigma_{cx}(E)\, n_H(\vec{r}(s))\, j_{ion}(E,\hat{\Omega},\vec{r}(s))\, ds\,}$$

- $\sigma_{cx}(E)$ — charge-exchange cross section (e.g., for H+ + H → H + H+, σ ≈ 1.4×10⁻¹⁵ cm² at 10 keV, falling as ~E⁻¹·² above 50 keV) / 전하교환 단면적
- $n_H(\vec{r})$ — cold geocoronal neutral density (Chamberlain model below) / 차가운 외기권 중성수소 밀도
- $j_{ion}$ — trapped-ion differential flux (#/cm²/s/sr/keV) along the LOS / 갇힌 ion 차분 플럭스
- The integration $ds$ is along the line of sight from the spacecraft outward. / 시선을 따른 적분.

### (ii) Chamberlain geocorona (simplified) / Chamberlain 외기권 (단순)

$$n_H(r) \approx n_0\left(\frac{r_0}{r}\right)^3,\qquad n_0 \sim 10^4\,\text{cm}^{-3}\text{ at } r_0 = 3R_E$$

The r⁻³ falloff dominates above ~3 R_E; closer in, the full Chamberlain function with Boltzmann factor is needed. / 약 3 R_E 이상에서는 r⁻³ 감쇠 지배.

### (iii) Time-of-flight / mass relation / 비행 시간 / 질량 관계

$$\boxed{\,m = \dfrac{2 E_{SSD}}{(d/\text{TOF})^2} = \dfrac{2 E_{SSD}\,\text{TOF}^2}{d^2}\,}$$

For HENA d ≈ 10 cm. A 50 keV/nuc proton (E_SSD = 50 keV) has v = (2E/m)^(1/2) = 3.1×10⁸ cm/s and TOF = 32 ns. A 50 keV/nuc oxygen (m=16 amu, E_SSD = 800 keV) has the same velocity and TOF, but its E_SSD is 16× larger — combining both directly yields species. / HENA에서 d ≈ 10 cm. 50 keV/nuc 양성자는 v = 3.1×10⁸ cm/s, TOF = 32 ns. 동일 속도의 50 keV/nuc 산소는 TOF 동일하나 E_SSD가 16배. 두 측정의 결합으로 종 결정.

### (iv) Velocity resolution from TOF / TOF로부터의 속도 해상도

$$\frac{\Delta v}{v} = \frac{\Delta\text{TOF}}{\text{TOF}}$$

With ΔTOF ≈ 1 ns and TOF ≈ 32 ns at 50 keV/nuc proton, Δv/v ≈ 0.03, i.e., Δv ≈ 50 km/s as quoted in Table 2.1.1 (recall 1 keV proton ≡ ~440 km/s; 50 keV proton ≡ ~3100 km/s; 50 km/s out of 3100 km/s ≈ 1.6 %, consistent with mid-range performance). / ΔTOF ≈ 1 ns, TOF ≈ 32 ns에서 Δv/v ≈ 3 %, 즉 Δv ≈ 50 km/s.

### (v) Energy resolution / 에너지 해상도

$$\frac{\Delta E}{E} = \sqrt{\left(\frac{\Delta E_{SSD}}{E}\right)^2 + \left(\frac{2\,\Delta\text{TOF}}{\text{TOF}}\right)^2}$$

with Δ E_SSD ≈ 7 keV (Fano + electronic noise) and ΔTOF ≈ 1 ns. At 30 keV/nuc proton: ΔE/E ≈ 0.25 (meeting the requirement <0.7). / 30 keV/nuc 양성자에서 ΔE/E ≈ 0.25.

### (vi) False-coincidence rate / 우연 일치 비율

$$\boxed{\,R_{false} = T_{val}\cdot T_{coinc}\cdot R_{start}\,R_{stop}\,R_{coinc} \approx 4\times 10^{-15}\, R_{start}\,R_{stop}\,R_{coinc}\,}$$

With T_val = 100 ns and T_coinc = 40 ns. Plugging in EUV-induced singles rates 1.5×10⁵, 1.5×10³, 1.5×10⁵ Hz yields R_false ≈ 1 event/s — far below typical foreground rates of 10³ /s in the inner magnetosphere. / 위 단일 비율 대입 시 R_false ≈ 1 /s — 내부 자기권의 전형적 전경 10³ /s보다 훨씬 낮다.

### (vii) Counts per pixel per spin / spin당 픽셀당 카운트

$$N_{pix} = J_{ENA}(E,\hat{\Omega})\cdot G_{pix}\cdot \varepsilon\cdot \Delta E \cdot \tau$$

with G_pix·ε ≈ 0.0027 cm²·sr per 3°×3° pixel, ΔE the energy bin width, and τ ≈ 120 s the spin period. For peak storm-time J_ENA ≈ 10³ /(cm²·s·sr·keV), ΔE ≈ 30 keV, and 120 s integration, N_pix ≈ 10⁴ counts/pixel — matching Figure 3's simulation. / 폭풍 정점 J_ENA ≈ 10³ 등 대입 시 N_pix ≈ 10⁴ counts/pixel — Figure 3의 시뮬레이션과 일치.

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1971 ─── Roelof: theoretical proposal of ENA imaging concept
1978 ─── ISEE-1/MEPI energetic-ion measurements → input for HENA simulations
1985 ─── Roelof, Mitchell, Williams: first ISEE-1 ENA detection (50 keV)
1987 ─── Roelof: Storm-time ring-current ENA image (proof of concept)
1989 ─── McEntire & Mitchell, Keath et al.: ENA instrumentation papers
1993 ─── Mitchell et al.: INCA paper for Cassini/MIMI (HENA's direct ancestor)
1996 ─── Lui et al.: Geotail/EPIC first composition-resolved ENA
1997 ─── Henderson et al.: Polar/CEPPAD ENA images
1998 ─── Hesse & Birn: substorm ENA modeling
       ─── Hsieh & Curtis: ENA imaging review (>10 keV)
       ─── Mitchell et al.: Cassini INCA flight description
       ─── McComas et al.: low-energy ENA review
1999 ─── Roelof & Skinner: ENA + EUV ion-distribution inversion
═══►  2000 ─── ★ HENA paper (this work) — IMAGE launched 25-Mar-2000 ★
2001 ─── First HENA storm images published (C:son Brandt et al., Mitchell et al.)
2003 ─── HENA papers: composition-resolved ring-current evolution
2008 ─── TWINS stereo ENA — direct successor of HENA
2009 ─── IBEX heliosphere ENA imaging (different energy regime, similar lineage)
2017 ─── Juno/JEDI begins Jupiter ENA imaging
2020+ ─── ENA imaging mature for Earth (TWINS), Saturn (Cassini/INCA), Mercury,
          Jupiter; HENA stands as the foundational Earth ring-current imager.
```

The paper is the **mid-point** of magnetospheric ENA imaging history: it consolidates ~30 years of conceptual and instrumentation work (Roelof through INCA) into the first dedicated mission-scale instrument, and seeds ~25 years of multi-planet successor missions. / 본 논문은 자기권 ENA 영상화 역사의 중간점에 해당한다 — Roelof로부터 INCA에 이르는 약 30년의 개념·기기 연구를 첫 임무급 전용 장비로 통합하고, 이후 약 25년의 다행성 후속 임무들의 씨앗이 되었다.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Roelof, Mitchell & Williams (1985) "Energetic Neutrals (E≥50 keV) from the Ring Current: IMP 7/8 and ISEE-1" | First detection of ring-current ENA in HENA's energy range | Established the source flux levels HENA was designed to image / HENA의 설계 플럭스 기준을 확립 |
| Roelof (1987) "Energetic Neutral Atom Image of a Storm-Time Ring Current" | First ENA image (proof of concept) | Defined the science problem HENA solves at high time resolution / HENA가 고시간해상도로 풀고자 한 과학 문제 정의 |
| Mitchell et al. (1993, 1998) "INCA: The Ion Neutral Camera for Cassini/MIMI" | Direct hardware and software heritage | 70 % of HENA flight code, DPU, foils, mechanical concepts inherited from INCA / 비행 코드 70 %, DPU, foil, 기계 개념이 INCA에서 계승 |
| Henderson et al. (1997) "First ENA Images from Polar/CEPPAD" | Concurrent Earth ENA observations | Demonstrated ENA imaging is possible in Earth orbit; HENA dramatically improved time/angular resolution / 지구 궤도 ENA 영상 가능성을 입증; HENA가 시간·각도 해상도를 대폭 개선 |
| Funsten, McComas & Gruntman (1998) "Neutral Atom Imaging: UV Rejection Techniques" | Foil and triple-coincidence design rationale | Provides theoretical foundation for HENA's Ly-α suppression / HENA Ly-α 억제의 이론적 토대 제공 |
| Hesse & Birn (1998) "Neutral Atom Imaging of the Plasma Sheet" | Predicted near-Earth tail ENA signatures | Validated HENA's substorm-injection science case / HENA의 서브스톰 주입 과학 사례를 검증 |
| Roelof & Skinner (1999) "Extraction of Ion Distributions from Magnetospheric ENA and EUV Images" | Image-inversion algorithm shipped with HENA Level-2 | Enables quantitative ion-distribution science from HENA data / HENA 데이터에서 정량적 ion 분포 과학을 가능케 함 |
| Burch (2000) "IMAGE Mission Overview" | Companion paper in the same SSR volume | Provides spacecraft, orbit, and mission-level context that HENA is embedded in / HENA가 속한 우주선·궤도·임무 맥락 제공 |
| Pollock et al. (2000) "MENA Instrument" | Sister NAI instrument 1–30 keV | HENA's complementary medium-energy partner; together they cover 1–500 keV / HENA의 중간 에너지 짝; 함께 1–500 keV 커버 |
| Moore et al. (2000) "LENA Instrument" | Sister NAI 0.01–0.5 keV | Completes the IMAGE NAI energy ladder with HENA at the top / IMAGE NAI 에너지 사다리에서 HENA가 최고 단을 차지 |

---

## 7. References / 참고문헌

- Mitchell, D. G. et al., "High Energy Neutral Atom (HENA) Imager for the IMAGE Mission", *Space Science Reviews* **91**, 67–112, 2000. DOI: 10.1023/A:1005207308094
- Roelof, E. C., Mitchell, D. G. & Williams, D. J., "Energetic neutral atoms (E ≥ 50 keV) from the ring current: IMP 7/8 and ISEE-1", *J. Geophys. Res.* **90**, 10991, 1985.
- Roelof, E. C., "Energetic neutral atom image of a storm-time ring current", *Geophys. Res. Lett.* **14**, 652, 1987.
- Mitchell, D. G. et al., "INCA, The Ion Neutral Camera for Energetic Neutral Atom Imaging of the Saturnian Magnetosphere", *Optical Engineering* **32**, 3096, 1993.
- Mitchell, D. G. et al., "The Ion Neutral Camera for the Cassini Mission to Saturn and Titan", AGU Geophysical Monograph **103**, 281, 1998.
- Henderson, M. G. et al., "First Energetic Neutral Atom Images from Polar", *Geophys. Res. Lett.* **24**, 1167, 1997.
- Funsten, H. O., McComas, D. J. & Gruntman, M. A., "Neutral Atom Imaging: UV Rejection Techniques", AGU Geophysical Monograph **103**, 251, 1998.
- Hesse, M. & Birn, J., "Neutral Atom Imaging of the Plasma Sheet: Fluxes and Instrument Requirements", AGU Geophysical Monograph **103**, 297, 1998.
- Hsieh, K. C. & Curtis, C. C., "Imaging Space Plasma with Energetic Neutral Atoms above 10 keV", AGU Geophysical Monograph **103**, 235, 1998.
- McComas, D. J., Funsten, H. O. & Scime, E. E., "Advances in Low Energy Neutral Atom Imaging", AGU Geophysical Monograph **103**, 275, 1998.
- Roelof, E. C. & Skinner, A. J., "Extraction of Ion Distributions from Magnetospheric ENA and EUV Images", *Planetary Space Sci.* (in review, 1999).
- Williams, D. J., Roelof, E. C. & Mitchell, D. G., "Global Magnetospheric Imaging", *Rev. Geophys.* **30**, 183, 1992.
- Lui, A. T. Y. et al., "First Composition Measurements of Energetic Neutral Atoms", *Geophys. Res. Lett.* **23**, 2641, 1996.
- C:son Brandt, P. et al., "Energetic Neutral Atom Imaging at Low Altitudes from the Swedish Microsatellite Astrid", *J. Geophys. Res.* **104**, 2367, 1999.
- Rairden, R. L., Frank, L. A. & Craven, J. D., "Geocoronal Imaging with Dynamics Explorer", *J. Geophys. Res.* **91**, 13613, 1986.
- Meckbach, W., Braunstein, G. & Arista, N., "Secondary-Electron Emission in the Backward and Forward Directions from Thin Carbon Foils Traversed by 25–250 keV Proton Beams", *J. Phys. B* **8**, L344, 1975.
- Hsieh, K. C., Keppler, E. & Schmidtke, G., "Extreme Ultraviolet Induced Forward Photoemission from Thin Carbon Foils", *J. Appl. Phys.* **51**, 2242, 1980.
