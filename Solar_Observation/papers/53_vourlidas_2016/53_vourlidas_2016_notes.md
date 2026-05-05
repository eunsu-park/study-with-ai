---
title: "The Wide-Field Imager for Solar Probe Plus (WISPR)"
authors: ["A. Vourlidas", "R.A. Howard", "S.P. Plunkett", "C.M. Korendyke", "A.F.R. Thernisien", "D. Wang", "N. Rich", "M.T. Carter", "D.H. Chua", "D.G. Socker", "M.G. Linton", "J.S. Morrill", "S. Lynch", "A. Thurn", "P. Van Duyne", "R. Hagood", "G. Clifford", "P.J. Grey", "M. Velli", "P.C. Liewer", "J.R. Hall", "E.M. DeJong", "Z. Mikic", "P. Rochus", "E. Mazy", "V. Bothmer", "J. Rodmann"]
year: 2016
journal: "Space Science Reviews 204, 83–130"
doi: "10.1007/s11214-014-0114-y"
topic: Solar_Observation
tags: [WISPR, Parker_Solar_Probe, heliospheric_imager, Thomson_scattering, coronagraph, instrumentation, F-corona, K-corona, APS_CMOS, baffle_design, stray_light, SECCHI_heritage]
status: completed
date_started: 2026-04-25
date_completed: 2026-04-25
---

# 53. The Wide-Field Imager for Solar Probe Plus (WISPR) / 솔라 프로브 플러스용 광시야 영상기 (WISPR)

---

## 1. Core Contribution / 핵심 기여

This paper is the definitive instrument description of WISPR — the only optical imager on NASA's Solar Probe Plus (SPP, later renamed Parker Solar Probe). WISPR is a two-telescope white-light heliospheric imager that delivers 95° radial × 58° transverse field of view by tiling an Inner telescope (13.5°–53° elongation) and an Outer telescope (50°–108°). Both feed independent radiation-hardened 2K × 2K APS CMOS detectors developed for SoloHI. Mounted on the ram-side of the spacecraft and looking past the heat shield's edge, WISPR rides to within 9.86 R⊙ of Sun center at perihelion, where its line-of-sight Thomson-scattering geometry passes through and behind the spacecraft, making WISPR an effective in-situ imager. The paper traces the science requirements through nine science questions (seven SPP Level-1 objectives plus two unique WISPR goals on dust and Vulcanoids) into a Science Requirements Traceability Matrix (SRTM), an end-to-end optical, mechanical, electrical and operational design, and a complete data-product / data-archive plan.

이 논문은 NASA Solar Probe Plus(SPP, 후일 Parker Solar Probe)의 유일한 광학 영상기 WISPR의 결정적 instrument paper이다. WISPR은 두 개의 망원경(Inner 13.5°–53°, Outer 50°–108°)을 결합하여 95° radial × 58° transverse FOV를 형성하는 백색광 헬리오스피어 영상기로, 각 망원경마다 독립적인 SoloHI 유산의 방사선 내성 2K × 2K APS CMOS 검출기를 갖는다. 우주선 ram-side에 장착되어 열차폐 가장자리를 넘어 코로나를 바라보는 WISPR은 근일점 9.86 R⊙까지 접근하며, 이 위치에서는 Thomson 산란 LOS가 우주선 자체를 관통하기에 사실상 'in-situ' 영상기로 기능한다. 본 논문은 9가지 과학 질문(SPP Level-1 7개 + WISPR 고유 2개)을 SRTM(과학요구사항 추적 행렬)로 전개하고, 광학·기계·전자·운용 전 영역의 end-to-end 설계와 자료처리·아카이브 계획까지 종합 기술한다.

---

## 2. Reading Notes / 읽기 노트

### Part I: Science Background and Mission Context (§1) / 과학 배경과 미션 맥락

The paper opens by acknowledging the central deficit of solar-wind science: although Parker (1958) and Snyder et al. (1963) settled the existence of the wind, the closest in-situ probe (Helios) reached only 0.3 AU, where the wind has already evolved past its source region. Remote-sensing imagery (LASCO, SECCHI/COR2, HI1/HI2) provides the missing global view but cannot resolve the fine scale of the heating/acceleration zone. SPP, designed for an unprecedented 9.86 R⊙ perihelion (35 R⊙ within three months of launch, 24 perihelia over a 7-year prime mission, three Venus gravity assists), addresses this deficit by physically entering the corona. WISPR's role is to provide the visual context that links the local in-situ measurements (FIELDS, SWEAP, ISIS-EPI) to the large-scale corona.

이 논문은 태양풍 과학의 중심 결핍을 지적하며 시작한다 — Parker(1958)와 Snyder et al.(1963) 이후 태양풍 존재는 확립됐지만, 가장 가까웠던 in-situ 탐사선 Helios도 0.3 AU에 머물러 태양풍이 이미 진화를 마친 후의 영역만 측정할 수 있었다. 원격 영상(LASCO, SECCHI/COR2, HI1/HI2)은 전체 모습을 보여주지만 가열·가속 영역의 미세구조는 분해하지 못한다. SPP는 9.86 R⊙(7년 prime mission 동안 24회 근일점, 3회 Venus 중력 보조)이라는 전례 없는 근일점으로 코로나에 직접 진입함으로써 이 간극을 메운다. WISPR의 역할은 in-situ 관측(FIELDS, SWEAP, ISIS-EPI)을 대규모 코로나에 연결하는 시각적 맥락을 제공하는 것이다.

**Field-of-view comparison (Table 1)**: WISPR at 0.044 AU perihelion sees 2.2°–20° elongation, equivalent to looking from 1 AU at ~1 R⊙ resolution of 17 arcsec; at 0.1 AU it covers 4.0°–41°; at 0.25 AU it covers 9.5°–83°. Compared to SECCHI/COR2 (2.5°–15° at 1 AU, 30 arcsec), WISPR delivers ~2× better resolution at perihelion and a much wider FOV. The Alfvén surface, theoretically at 10–30 R⊙, lies inside WISPR's FOV at all perihelia. / 표 1 비교 — WISPR은 근일점에서 SECCHI/COR2보다 약 2배 좋은 해상도, 훨씬 넓은 FOV를 제공하며, 이론적 Alfvén 표면(10–30 R⊙)이 모든 근일점에서 시야에 들어온다.

**Thomson-scattering geometry (Figure 2)**: As the spacecraft sits inside or near the Thomson surface, the locus of maximum scattering passes through the spacecraft itself. At elongation ε = 90° this means the maximum-scattering plasma is *at the spacecraft*, so WISPR becomes an in-situ imager at that pixel. The 5 %, 50 %, 95 % cumulative-brightness contours along each LOS show that at perihelion only emission within ~10 R⊙ of the LOS contributes 50 % of the signal, whereas at the start of the encounter (≈40 R⊙) the same fraction extends much farther. Thus WISPR's "depth of field" along each LOS shrinks dramatically as it approaches perihelion — a unique property never before exploited. / WISPR 위치가 Thomson surface 근처이므로 ε = 90°에서 산란 최대 위치가 우주선 자체에 일치하고, 그 픽셀에서 WISPR은 in-situ 영상기로 작동한다. 누적 5/50/95 % 등고선은 근일점에서 LOS 깊이가 10 R⊙ 정도로 좁아지는 독특한 성질을 보인다.

**Nine science questions (§1.3, Table 2 SRTM)**: WISPR maps onto the three SPP Level-1 objectives plus two WISPR-unique goals.
- **L-1 #1 (sources of fast/slow wind)**: SQ1 streamer structure, SQ2 streamer evolution into wind, SQ3 source steady vs intermittent. WISPR images streamer current sheets, blobs (5-h periodicity, Viall et al. 2010), HCS thickness, fast wind plume/interplume contrast within coronal holes.
- **L-1 #2 (energy flow → heating, acceleration)**: SQ4 atmosphere-corona energy transfer, SQ5 corona-heliosphere coupling. WISPR maps fast/slow boundary thickness, density power spectra (the "wave turbulence" program).
- **L-1 #3 (energetic particle acceleration)**: SQ6 shocks/reconnection/waves/turbulence, SQ7 transport across field lines. WISPR images CME-driven shocks at <10 R⊙ at 5–10 min cadence (13–26 frames per 2000 km/s CME).
- **WISPR-unique #1 (dust)**: SQ8 dust environment (F-corona radial gradient, dust-free zone search), SQ9 dust-plasma interactions.
- 9개 과학 질문은 SPP의 3개 Level-1 목표(빠른/느린 바람의 근원, 에너지 흐름, 입자 가속)와 WISPR 고유 2개(먼지 환경, 먼지-플라즈마 상호작용)에 대응한다.

**Unique science: F-corona and Vulcanoids (§1.4)**: At 1 AU the F-corona dominates over K above 4 R⊙, but its forward-scattering origin means most of it comes from dust halfway between Sun and observer, masking near-Sun dust structure. As WISPR moves inward, the dust-contributing region first concentrates (brightness rises) then rolls over near the predicted dust-free zone (<4 R⊙, Russell 1929 — never confirmed). WISPR will also extend the search for Vulcanoids (sunward of Mercury, 0.08–0.2 AU), which prior LASCO/MESSENGER/SECCHI searches set only upper limits on. / 1 AU에서 F-corona는 4 R⊙ 이상에서 K-corona를 압도하지만 전방 산란 특성상 우주선과 태양 중간의 먼지가 대부분이므로 근일점 부근 먼지 구조를 가린다. WISPR이 안쪽으로 이동하면 먼지 기여 영역이 집중되다가 dust-free zone(<4 R⊙) 근방에서 굴러 떨어질 것으로 예측. Vulcanoid(머큐리 안쪽 0.08–0.2 AU 미세 천체) 탐색도 가능.

### Part II: Instrument Overview (§2) / 기기 개관

**System architecture (§2.1, Fig. 9)**: WISPR consists of (1) the WIM (WISPR Instrument Module) housing two telescope assemblies, three baffle systems (forward F1–F3, interior I1–I7, peripheral aperture-hood), the Camera Interface Electronics (CIE), and a one-shot door, plus (2) the IDPU (Instrument Data Processing Unit) inside the spacecraft bulkhead with the DPU (Data Processing Unit) and LVPS (Low-Voltage Power Supply). Total mass: WIM 9.8 kg + IDPU 1.1 kg ≈ 10.9 kg. Average power: 7 W. WIM envelope: 58 × 30 × 46 cm. Telemetry allocation: 23 Gbits per 0.25-AU orbit (10-day science encounter), average 26.6 kbps. / WISPR은 망원경 + baffle + CIE를 포함한 WIM(9.8 kg)과 IDPU(1.1 kg)로 구성되며 평균 전력 7 W, 궤도당 23 Gbits(평균 26.6 kbps).

**Two-telescope rationale (§2.2, Fig. 10)**: A single 180° UFOV wide lens would be intercepted by the FIELDS antennas mounted on the sunward side of SPP, creating intolerable stray light from antenna tips that reach ~1800 °C. Splitting into Inner (13.5°–53°) + Outer (50°–108°) telescopes lets peripheral baffles capture the antenna scatter without sacrificing science FOV. The 13.5° inner cutoff (= 2.3 R⊙ at 9.86 R⊙ perihelion) is set by (a) the heat-shield umbra (8° + 2° offpoint margin) and (b) practical mass/height limits. / 단일 광시야 렌즈는 sunward 위치의 FIELDS 안테나에 의해 간섭받기에, Inner + Outer 분할로 peripheral baffle이 안테나 산란을 차단하도록 설계. 안쪽 13.5° 컷오프는 열차폐 umbra(8° + 2° 여유) + 실용 한계로 결정.

**Instrument characteristics (Table 3)**: Inner f = 28 mm, aperture 42 mm² (~7.3 mm entrance pupil), bandpass 490–740 nm; Outer f = 19.8 mm, aperture 51 mm² (~8.1 mm pupil), bandpass 475–725 nm. Plate scale 1.2–1.7 arcmin/pixel. Predicted RMS spot at 20° from boresight: Inner 19.5 µm (2.34 arcmin), Outer 19.9 µm (3.38 arcmin). Pointing accuracy <0.5°, F1-baffle leading-edge placement <13 mm. Stray-light requirement: <2 × 10⁻⁹ B/B☉ at 9.86 R⊙ (inner) and <2 × 10⁻¹² B/B☉ at 0.25 AU (outer). Photometric calibration <20 % absolute (predicted ~3 % via standard stars), plate scale <4 %. Calibrated SNR: 20 at inner FOV at perihelion to 5 at outer edge at 0.25 AU. / 표 3 핵심 — Inner f = 28 mm (F#3.83), Outer f = 19.8 mm (F#4.04), 픽셀당 1.2–1.7 arcmin, 산란광 요구 1.4 × 10⁻¹¹ B/B☉ 수준, SNR 20→5.

### Part III: Environmental Challenges (§2.4) / 환경적 도전

**Radiation (§2.4.1)**: Total ionizing dose 24 krad behind 100 mils Al for 7 years. EEE parts margin 2× (60 krad behind 100 mils). LET threshold <25 MeV-cm²/mg (SEU) or 100 MeV-cm²/mg (latch-up). APS detectors avoid CCD-style charge-transfer-efficiency (CTE) degradation because each pixel is read independently, but they still suffer dark-current and DC-non-uniformity increases that need on-board scrubbing. / 7년 누적 TID 24 krad(100 mils Al 후), EEE 마진 2배, APS 검출기는 픽셀별 readout으로 CTE 열화 회피.

**High-speed dust impacts (§2.4.2, Figs. 11–12)**: SPP perihelion dust velocity ~170 km/s. JHU-APL/UTEP model (Mehoke et al. 2012) predicts ~100 hits from 10-µm grains and ~1000 hits from 0.1-µm grains during 7 years on the heat shield. The team tested three glass candidates at the MPIK Dust Accelerator (Heidelberg, October 2012) with iron particles (0.5–3 µm, 0.5–8 km/s, three angles 0°, 45°, 70°). Sapphire was most impact-resistant (2-µm craters, 1.5 % damage area, 0.5 % per 10⁵ impacts) but immature for space; BK7+DLC produced unexpected ring-craters from coating delamination (5.9 %, 1.4 % per 10⁵); standard BK7 gave 5-µm craters (2.2 %, 1.5 % per 10⁵). Baseline: BK7 — flight-proven, modest damage, predicted 0.6 % damaged area on the outer telescope objective by end of life. The damaged-glass BSDF was measured and folded into the stray-light model (Fig. 13: BOL vs EOL). / SPP 근일점 먼지 속도 ~170 km/s, 7년간 10-µm 알갱이 약 100회 충돌. MPIK Dust Accelerator 시험으로 세 유리(BK7, BK7+DLC, sapphire) 비교; baseline은 BK7로 예상 손상 면적 0.6 %.

### Part IV: Optical Design (§3.1) / 광학 설계

**Lens layouts (Fig. 14, Table 4)**: Inner telescope is a 5-element refractive lens with 40° × 40° FOV, F#3.83, entrance pupil 7.31 mm, RMS spot 19 µm. Outer telescope is a 6-element lens with 58° × 58° FOV, F#4.04, entrance pupil 8.08 mm, RMS spot 20 µm. Resolution at center field is optimized for 33.5° (Inner) and 79° (Outer). The bandpass is set with long/short cutoff filters deposited on internal lens surfaces (SECCHI/HI heritage). / Inner는 5요소 굴절 렌즈(40°×40°), Outer는 6요소(58°×58°). 둘 다 RMS spot ≈ 20 µm.

**Three-tier baffle system (Figs. 15–17)**:
1. **Forward baffle assembly (F1–F3)**: A1 entrance aperture is the inner-telescope aperture; F1 is the leading edge that defines the 9.12° forward UFOV (avoiding the 8.07° heat-shield umbra by 1.05° tolerance). F1 leading-edge placement is critical: <13 mm tolerance. Three linear occulters in the forward baffle attenuate diffraction from the heat-shield edge (Fig. 16 shows ~10²⁰ rejection by the F1+F2+F3 sequence over 30°).
2. **Interior baffle assembly (I1–I7)**: CFRP (carbon-fiber reinforced polymer) panels coated with Aeroglaze Z307 (heritage from STEREO/HI), oriented to prevent any single reflection from spacecraft surfaces outside the aft UFOV from reaching A1. I1–I7 also vignette the inner-telescope FOV from 60 % at 13.5° to 30 % at 14° to manage the sharp brightness gradient near the limb.
3. **Peripheral baffle / aperture hood assembly**: An Al plate with cutouts around both telescope apertures, built specifically to mask the FIELDS antenna scatter (1800 °C antenna tips radiating in the visible).

The result is a worst-case predicted stray-light level of 1.4 × 10⁻¹¹ B/B☉ at the detector inner FOV at minimum perihelion (9.86 R⊙) — 55× below the 7.9 × 10⁻¹⁰ B/B☉ requirement. EOL prediction (with 0.6 % dust damage on objective, F-corona, and dust-impact scattering) remains within requirement (Fig. 13).

**Three-tier baffle 시스템**: (1) Forward F1–F3 — A1 입구 + 9.12° forward UFOV, F1 위치 공차 <13 mm, 3개 선형 occulter가 열차폐 회절을 ~10²⁰ 억제(Fig. 16). (2) Interior I1–I7 — Aeroglaze Z307 코팅 CFRP, 어떤 단일 반사도 A1에 도달하지 않게 정향. 내부 inner FOV vignetting 13.5°→60 %, 14°→30 %. (3) Peripheral / aperture hood — Al 판 + 컷아웃, FIELDS 안테나 산란 차단. 최종 worst-case 산란광 1.4 × 10⁻¹¹ B/B☉(요구 7.9 × 10⁻¹⁰ 대비 55배 여유).

### Part V: Mechanical, Electrical, and Detector Design (§3.2–3.4) / 기계·전자·검출기 설계

**Mechanical (§3.2)**: WIM primary structure is a CFRP composite honeycomb panel; bipod legs (Ti-Al composite tubes) maintain >80 Hz first mode and address CTE mismatch with the Al spacecraft panel. Door is multilayer CFRP on an Invar mold for CTE matching, opened once via a shape-memory Ejection Release Mechanism (ERM) — the only WISPR moving part. Mounted on +X, +Y panel rotated −20° about Z for optimal target coverage. / WIM은 CFRP 하니콤 + Ti-Al bipod, 1차 모드 >80 Hz, 단발성 SMA door가 유일 가동부.

**Camera electronics (§3.3.1, Fig. 21, Table 5)**: APS detector format 2048 × 1920 (10 µm pixels, 5T-PPD pixel design), Jazz 0.18 µm process, QE >34.3 % avg over 470–740 nm, radiation tolerant to 100 krad TID, read noise 7–13 e⁻/pix EOL, dark current 1.57–1.9 e⁻/s/pix EOL, full well 20,000–21,300 e⁻/pix, 14-bit ADC, 2 Mpix/s readout, operating <−55 °C (passive radiator cold finger). The detector splits into top/bottom 960 × 2048 halves readable independently for redundancy. SECCHI/HI CCDs operate at <−70 °C for reference. CIE (Camera Interface Card + Detector Interface Boards + Detector Readout Boards) provides command/telemetry over LVDS UART (3.3 V, 19.2k baud), 14-bit A/D, and a 2 Mpix/s SPI link with ≤256-byte headers. / APS 사양: 2048 × 1920, 10 µm 픽셀, 14-bit, 100 krad TID 견딤, EOL 읽기잡음 7–13 e⁻, 동작 온도 <−55 °C, 패시브 라디에이터 냉각.

**IDPU (§3.4)**: 21.2 × 11.6 × 5 cm Mg-alloy box, 1070 g, 7.3 W. DPU FPGA (Actel RTAX2000) hosts a SCIP processor (a programmable FORTH processor for housekeeping) and an Image Processor (FPGA-based) that performs bias subtraction, pixel binning (1×, 2×, 4×), saturation/starvation detection, two-frame cosmic-ray scrub, frame summation up to N images, masking, bit truncation, lossy/lossless compression, and packetization. Memory: 156 Mb SRAM image buffer + 3 Gb SDRAM secondary buffer + 64 Gb flash tertiary (sufficient for 2 orbits). LVPS converts 28 V spacecraft bus to 5 V digital + ±6.6 V analog. / IDPU는 Mg 합금 21.2 × 11.6 × 5 cm, 1070 g, 7.3 W. RTAX2000 FPGA가 SCIP 프로세서 + Image Processor를 호스팅하여 바이어스·binning·CR scrub·summation·압축·패킷화 처리. 메모리 156 Mb SRAM + 3 Gb SDRAM + 64 Gb flash.

### Part V-bis: Detailed Optical Performance Reasoning / 광학 성능 상세

**Why two telescopes physically (Fig. 17, top vs bottom)?** The paper presents a striking before-after stray-light prediction. The original single-lens (180° UFOV) WISPR concept produced unacceptable stray-light levels because the FIELDS antennas — radiating in the visible at 1800 °C tip temperature — sat directly within the UFOV. The bottom panels of Fig. 17 show that splitting into Inner (40°×40° FOV) + Outer (58°×58° FOV) telescopes lets peripheral baffles capture the antenna scatter, dropping the predicted stray light by orders of magnitude across both detectors. This is qualitatively visible as the deep-blue (low) levels in the lower panels, whereas the single-lens upper panel shows large red regions from antenna intrusion. / Fig. 17 비교 — 단일 광시야 렌즈에서는 1800 °C FIELDS 안테나의 가시광 복사가 UFOV 내에 직접 들어와 산란광 수준이 허용 불가였지만, Inner + Outer 분할 + peripheral baffle로 antenna 산란을 차단하니 검출기 양쪽에서 수십 배 개선됐다.

**FRED Monte-Carlo stray-light modeling**: The team used a CAD model of the spacecraft + WISPR + FIELDS antennas in FRED Optical Engineering Software with Monte-Carlo ray tracing. This was the first time such an approach was applied to a coronagraph-class instrument with structures intruding into the UFOV — the lessons learned will guide future "occulter-like imagers in crowded spacecraft environments". / FRED 광학 엔지니어링 소프트웨어 + Monte-Carlo 광선추적으로 우주선 + WISPR + FIELDS 안테나 CAD 모델 분석. UFOV에 구조물이 침범하는 코로나그래프급 기기에 처음 적용된 방법론이다.

**Vignetting strategy (§3.1)**: Inside the inner telescope's 13.5° edge, the I7 baffle progressively vignettes the FOV from 60 % at 13.5° to 30 % at 14°. This is a deliberate choice — sharp brightness gradients near the limb would otherwise saturate pixels and produce ghosts. The wide-field lens itself adds a natural cos⁴θ vignetting (θ = angle from boresight), which combines with the baffle vignetting. The result is a smooth roll-off matching the steep coronal brightness gradient. / 내측 가장자리에서 I7 baffle이 점진적 vignetting(13.5°→60 %, 14°→30 %)을 도입; 광시야 렌즈 자체의 cos⁴θ와 결합하여 가파른 코로나 휘도 경사에 부드럽게 적응한다.

**End-of-life vs Beginning-of-life (Fig. 13)**: BOL stray light at the inner telescope (e=13°) shows uniform ~10⁻¹¹ B/B☉ patches; at the outer telescope (e=50° to e=108°) the BOL pattern is dominated by photon noise. EOL with 0.6 % dust damage on the objective shifts the inner telescope to ~10⁻⁹ B/B☉ in some patches (the brighter scene at small elongation amplifies any optical defects), while outer telescope EOL remains nearly unchanged because the K-corona is much fainter at large elongation. This asymmetric degradation tolerance is by design. / BOL/EOL 비교 — Inner는 EOL에서 0.6 % 먼지 손상이 일부 패치를 ~10⁻⁹ 수준으로 끌어올리지만(작은 ε에서 K-corona가 밝기에 결함이 증폭), Outer는 EOL에서도 거의 불변. 비대칭 열화 내성은 의도된 설계.

### Part VI: Operations and Data Products (§4) / 운용과 자료 산출물

**Operational regions (Tables 6–7)**: The 10-day science encounter (spacecraft <0.25 AU from Sun) is split into Perihelion (1.5 days, <0.07 AU), Inner (1.5 days, 0.07–0.11 AU), Mid (3 days, 0.11–0.18 AU), Outer (4 days, 0.18–0.25 AU). At perihelion: full-frame cadence 2.5 min for 13.5°–38.5° (576 frames/day, 4.71 Gbits), inner FOV subframe at 5.5 sec cadence (3927 frames/day, 2.80 Gbits), totalling 7.93 Gbits/day, 11.90 Gbits/orbit, 91.8 kbps avg. At outer: cadence relaxes to 20–40 min, 2.44 Gbits/orbit (7.1 kbps). / 10일 science encounter는 4단계로 분할 — Perihelion(1.5일, <0.07 AU)에서 inner FOV subframe 5.5초 cadence·91.8 kbps; Outer는 7.1 kbps로 완화.

**On-board processing**: cosmic-ray scrubbing (compare 2 frames pixel-by-pixel), pixel binning (2×2 baseline, 4×4 at large heliocentric distance to maintain SNR ≥ 5), N-frame summation (for SNR), bias subtraction, lossy compression at high cadence, lossless at low cadence. Image headers carry 64 APIDs for downlink-priority management. The DPU SpaceWire output is limited to 350 kbps. / 온보드 처리 — CR scrub(2-frame 비교), 2×2 binning(4×4 at 큰 거리), N-프레임 합산, 압축 등.

**Data products (§4.4, Table 9)**: L1 (uncalibrated FITS, quick-look + final per orbit), L2 (calibrated FITS, user-generated via Solarsoft tools), L3 (browse images PNG/JPG, browse movies MPG, J-maps for time-elongation, synoptic / Carrington maps), L4 (CME masses, annual). Open data policy. NRL maintains primary archive + daily-updated copy at TBD partner. Calibration accuracy: ~3 % via in-flight standard-star calibration (no maneuvers required given the wide FOV — stars transit the FOV automatically). / 자료 단계 — L1 quick-look/final FITS, L2 calibrated, L3 browse 이미지/J-map/Carrington, L4 CME 질량. 별을 이용한 in-flight 측광 보정 ~3 %.

**Operations infrastructure**: Science Operations Center (SOC) at NRL uses GSEOS suite + Heliospheric Imager Planning Tool (HIPT) to translate science plans into IDPU schedule files. Telemetry pipeline: MOC → SOC (real-time socket or SFTP L0 files) → MySQL housekeeping DB + Image Processing Pipeline (IPP) producing L1 FITS. WISPR Data Analysis Tools (DAT) released via Solarsoft IDL library. / SOC@NRL — GSEOS + HIPT으로 일정 생성, IPP로 L1 FITS 생산, Solarsoft DAT 배포.

**Calibration strategy (§4.1.3)**: Three-phase per-encounter sequence: (1) detect any in-encounter degradation of detectors/lenses; (2) anneal the APS detector to recover dark-current/non-uniformity; (3) build a pre-perihelion baseline. Photometric calibration uses the in-flight stellar transits — because of WISPR's wide FOV, no spacecraft maneuver is needed to bring stars into view, unlike SECCHI which uses dedicated star observations. Final accuracy ~3 % (well below the 20 % requirement). LED calibration lamps on the detector handle vignetting and flat-field. / 보정 전략 — 매 encounter 3단계 시퀀스(검출기·렌즈 열화 감지 → APS annealing → pre-perihelion baseline). 측광 보정은 별 천이 자동 활용(광시야 덕분), 정확도 ~3 %. LED 램프로 vignetting·flat-field 보정.

**Documentation and archive philosophy (§4.4)**: The WISPR team commits to (1) instrument description, (2) calibration & validation methodology, (3) cross-instrument validation, (4) FITS header definition, (5) metadata products — all open via the WISPR website. The FITS header will mirror the SPASE catalog (community standard), facilitating cross-mission queries. Final L2 calibrated dataset will be re-generated at end-of-mission and delivered to a NASA archive. / 문서화·아카이브 철학 — 5개 문서 카테고리, FITS 헤더는 SPASE 표준에 맞춤, EOM에 최종 L2 데이터셋 NASA 아카이브 이관.

### Part VII: Summary and Outlook (§5) / 요약과 전망

The paper closes with three statements of import: (1) WISPR will provide the first tomographic 3D maps of inner-coronal density structure with greatly reduced F-corona interference; (2) it will verify or rule out the existence of a dust-free zone (Russell 1929) and constrain Vulcanoid populations within Mercury's orbit; (3) it will image structures at spatial scales close to the dissipation range and capture the formation moments of CME-driven shocks. The PDR was complete at the time of writing; CDR was scheduled for December 2014. The team explicitly notes "no major concerns for the instrument at this point" — environmental challenges (stray light, dust, radiation) had been addressed, the flight detectors were ready, and the project was on schedule and within budget. The paper's last sentence captures the spirit: "We are all excited for the time when we will gaze at the first images from inside the Sun's atmosphere." / 결론 — (1) F-corona 간섭이 크게 줄어든 첫 inner-corona 3D 토모그래피, (2) dust-free zone 검증과 Vulcanoid 제한, (3) dissipation 스케일 미세구조와 CME 충격파 형성 순간 영상화. PDR 완료, CDR 2014.12 예정, "instrument에 주요 우려 없음". 마지막 문장은 "태양 대기 안에서의 첫 영상을 기다리는 흥분"으로 마무리.

---

## 3. Key Takeaways / 핵심 시사점

1. **WISPR is the first imager that operates inside the corona it images** — by riding to 9.86 R⊙ perihelion, its line-of-sight Thomson-scattering geometry passes through and beyond the spacecraft, making each ε ≈ 90° pixel an in-situ measurement and shrinking LOS depth-of-field to ~10 R⊙ at perihelion. This is fundamentally different from any prior heliospheric imager (LASCO, SECCHI/HI, SoloHI from 0.28 AU). / WISPR은 자신이 영상화하는 코로나 안에서 동작하는 최초의 영상기로, 9.86 R⊙ 근일점에서 LOS Thomson 기하가 우주선을 통과하므로 ε≈90° 픽셀이 사실상 in-situ가 되고 LOS 깊이는 10 R⊙로 좁아진다 — LASCO·SECCHI/HI·SoloHI와 근본적으로 다른 관측 모드.

2. **Two-telescope split is not a science choice but an engineering necessity** — the FIELDS antennas force the team to abandon the natural 180° single-lens design. The split (Inner 13.5°–53° + Outer 50°–108°) lets peripheral baffles intercept antenna scatter without losing science FOV, and is a textbook example of stray-light driving instrument architecture. / 두 망원경 분할은 과학적 선택이 아닌 공학적 필연 — FIELDS 안테나의 sunward 위치가 단일 180° 광시야 렌즈를 불가능하게 하여, peripheral baffle로 안테나 산란을 차단할 수 있도록 Inner + Outer로 분할했다. 산란광이 기기 아키텍처를 결정한 교과서적 사례.

3. **APS CMOS replaces CCD for radiation hardness** — unlike CCDs (LASCO, SECCHI), APS reads out per-pixel, eliminating CTE degradation under radiation. WISPR inherits the SoloHI APS (Korendyke et al. 2013): 2048×1920, 10-µm 5T-PPD, Jazz 0.18 µm, QE >34 %, radiation-tolerant to 100 krad, and operates only at <−55 °C (vs SECCHI/CCD at <−70 °C) thanks to the per-pixel architecture. / APS CMOS는 방사선 내성을 위해 CCD를 대체 — 픽셀별 readout으로 CTE 열화 회피, 100 krad TID 견딤, 동작 온도 −55 °C로 완화. SoloHI 유산.

4. **Stray-light budget achieved by a triple-baffle system spanning ~10²⁰ rejection** — the combination of heat-shield leading edge + F1–F3 forward baffles + I1–I7 interior baffles + peripheral aperture hood reduces direct solar diffraction by ~10²⁰ over 30° (Fig. 16) and yields a worst-case 1.4 × 10⁻¹¹ B/B☉ at perihelion — 55× below requirement. EOL with 0.6 % dust damage on the objective remains within budget. / 산란광 예산은 3중 baffle로 ~10²⁰ 억제 달성 — 열차폐 + F1–F3 forward + I1–I7 interior + peripheral aperture hood의 조합이 worst-case 1.4 × 10⁻¹¹ B/B☉(요구 대비 55배 여유)를 보장하고, 0.6 % 먼지 손상 EOL에서도 만족.

5. **Dust impact testing at the MPIK Dust Accelerator drove the final glass choice** — testing BK7, BK7+DLC, and sapphire at 0.5–8 km/s with iron particles showed sapphire had smallest craters but lacked space heritage; BK7+DLC produced unexpected coating-detachment ring craters; standard BK7 was selected as the proven flight option with predicted 0.6 % damaged area. The measured BSDF of damaged BK7 was folded into the stray-light model. / 최종 유리 선택은 MPIK Dust Accelerator 충돌 시험이 결정 — sapphire는 손상 최소이나 우주 비행 이력 없음, BK7+DLC는 코팅 박리 ring crater, BK7이 proven flight option으로 선택 (예상 EOL 손상 0.6 %).

6. **The Science Requirements Traceability Matrix (SRTM) is the design backbone** — Table 2 traces nine science questions through measurement objectives → observation requirements (radial coverage 14°–90°, transverse 25°–55°, latitude ±40°, spatial resolution ≤6.4 arcmin, photometric SNR ≥20, cadence ≤4 sec at 14°) → instrument design parameters. The 6.4-arcmin resolution, ≤16.5-min synoptic cadence, and 14°–90° radial coverage are Level-1 fixed requirements; everything else is derived. / SRTM(과학요구사항 추적 행렬)은 설계의 척추 — 9개 과학 질문이 측정 목표→관측 요구→기기 설계로 추적되며, 6.4 arcmin 해상도, ≤16.5 min synoptic cadence, 14°–90° radial coverage가 Level-1 고정 요구.

7. **Operational cadence flexes by ~10× across the orbit to balance SNR vs telemetry** — at perihelion (×0.07 AU) inner FOV subframes run at 5.5 sec cadence; at outer (0.25 AU) full-frame cadence drops to 40 min. Total daily volume ranges 7.93 Gbits (Perihelion) → 0.63 Gbits (Outer). The 23 Gbits/orbit allocation is the hard constraint that drives binning (2×2 baseline, 4×4 at large distance) and frame summation. / 궤도에 따라 cadence가 ~10× 유연 — 근일점에서 5.5 sec subframe, outer에서 40 min full-frame. 23 Gbits/궤도 할당이 binning과 합산을 결정.

8. **The WISPR–SECCHI–SoloHI lineage demonstrates instrument evolution** — WISPR re-uses the SECCHI/HI baffle architecture (Socker et al. 2000), the SoloHI APS detector (Korendyke et al. 2013) and CIE electronics, the LASCO calibration philosophy (~3 % via stars), and the SECCHI Image Processing Pipeline. The miniaturization is dramatic: WISPR 9.8 kg vs SECCHI 72 × 42 × 24 cm (~2.5× smaller volume). / WISPR–SECCHI–SoloHI 계보는 기기 진화의 명확한 사례 — SECCHI/HI baffle 구조, SoloHI APS·CIE, LASCO 측광 철학, SECCHI IPP를 모두 재사용·축소(2.5× 작은 부피)한다.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Thomson scattering brightness (Howard & Tappin 2009 form)

For an electron at scattering angle χ from the Sun-observer line, with $u$ the limb-darkening coefficient:

$$
B(\chi) \;\propto\; N_e(r)\,\sigma_T\,\Bigl[(1-u)\,B_T(\chi)\;+\;u\,B_R(\chi)\Bigr]
$$

with $\sigma_T = 6.652 \times 10^{-29}$ m² the Thomson cross-section. The geometric factors $B_T, B_R$ peak at χ = 90° (Thomson surface) where $\sin^2\chi = 1$. Integrating along an LOS from the spacecraft at heliocentric distance $d$ at elongation $\varepsilon$:

$$
B_{\text{LOS}}(\varepsilon) \;=\; \int_0^{\infty} B(\chi(s),N_e(r(s)))\, ds
$$

where $r(s) = \sqrt{d^2 + s^2 - 2 d s \cos(\pi - \varepsilon)}$ for an LOS leaving the spacecraft at elongation ε. The Thomson-surface point along this LOS is $s_{TS} = d\cos\varepsilon$, giving $r_{TS} = d\sin\varepsilon$. / 우주선이 헬리오중심 거리 $d$, 이격각 $\varepsilon$에서 보는 LOS의 Thomson surface 거리는 $r_{TS} = d\sin\varepsilon$.

### 4.2 Elongation–distance conversion (key WISPR numbers)

| Spacecraft distance d / 우주선 거리 | ε = 13.5° → r_TS | ε = 53° → r_TS | ε = 90° → r_TS | ε = 108° → r_TS |
|---|---|---|---|---|
| 9.86 R⊙ (perihelion) | 2.30 R⊙ | 7.87 R⊙ | 9.86 R⊙ | LOS goes behind S/C |
| 0.07 AU (15 R⊙) | 3.50 R⊙ | 12.0 R⊙ | 15.0 R⊙ | — |
| 0.25 AU (53.7 R⊙) | 12.5 R⊙ | 42.9 R⊙ | 53.7 R⊙ | — |

These three rows reproduce the WISPR-relevant entries in the paper's Table 1 to within rounding. / 표 1의 WISPR 행을 반올림 수준으로 재현.

### 4.3 FOV tiling and overlap

Inner: ε ∈ [13.5°, 53°], cross-track 40°. Outer: ε ∈ [50°, 108°], cross-track 58°. Combined radial: 13.5°–108° (95° span). Overlap ε ∈ [50°, 53°] (3° crossfade region). Cross-track combined: 58° (set by Outer). / 중첩 ε ∈ [50°, 53°], 결합 radial 95°, transverse 58°.

### 4.4 Plate scale and resolution

For pixel pitch $p = 10\,\mu\text{m}$ and focal length $f$:

$$
\text{plate scale} \;=\; \frac{p}{f}\quad[\text{rad/pix}]
$$

| Telescope | f (mm) | rad/pix | arcsec/pix | arcmin/pix |
|---|---|---|---|---|
| Inner | 28.0 | 3.57 × 10⁻⁴ | 73.6 | 1.23 |
| Outer | 19.8 | 5.05 × 10⁻⁴ | 104.1 | 1.74 |

Spatial resolution at the Sun: at d = 9.86 R⊙, 1.23 arcmin × d = 350 km on the Thomson-surface plane (≈ 0.0005 R⊙); at d = 1 AU AU_eq it would correspond to 17 arcsec. / Inner 28 mm·10 µm → 1.23 arcmin/pix; 9.86 R⊙ 거리에서 350 km 해상도.

### 4.5 SNR and stray-light budget

Photon noise alone (Fig. 7, 1-s exposure at 9.5 R⊙): photon-noise floor ~10⁻¹¹ B/B☉ at ε ≈ 90°. F-corona (equatorial, dotted) ≈ 10⁻⁹ at ε = 5° → 10⁻¹² at ε = 100°. K-corona (equatorial, red) ≈ 10⁻⁸ at ε = 5° → 10⁻¹³ at ε = 100°. Stray-light requirement and prediction:

$$
B_{\text{stray}}^{\text{req}} \;=\; 7.9 \times 10^{-10}\,B_\odot \quad(\text{worst case})
$$

$$
B_{\text{stray}}^{\text{pred}} \;=\; 7.5\times10^{-13}\,B_\odot \;\;\text{(diffraction only)}
$$

$$
B_{\text{stray}}^{\text{pred,total}} \;=\; 1.4\times10^{-11}\,B_\odot \;\;\text{(F-corona + dust + antennas)}
$$

The total prediction is 55× below the requirement. / Worst-case 산란광 예측 1.4 × 10⁻¹¹ B/B☉ — 요구 대비 55배 여유.

### 4.6 Telemetry budget

$$
T_{\text{daily}} \;=\; \sum_{i \in \text{regions}} \left( N_i^{\text{full}} \cdot S_i^{\text{full}} + N_i^{\text{sub}} \cdot S_i^{\text{sub}} \right) + T_{\text{HK}} + T_{\text{CCSDS}}
$$

with $N_i$ frame counts, $S_i$ frame sizes (post-compression). Per orbit: 23 Gbits = sum over Perihelion (11.9) + Inner (5.4) + Mid (3.25) + Outer (2.44) Gbits. Avg downlink 26.6 kbps. / 궤도당 23 Gbits = Perihelion 11.9 + Inner 5.4 + Mid 3.25 + Outer 2.44 Gbits.

### 4.7 Detector exposure-time / SNR scaling

For a uniform-brightness scene with photon flux $\Phi$ photons/pix/s, exposure $t_{\text{exp}}$, dark current $D$ e⁻/s/pix, read noise $\sigma_R$ e⁻/pix RMS, $N$ summed exposures:

$$
\text{SNR} \;=\; \frac{N\,\Phi\,t_{\text{exp}}}{\sqrt{N\,\Phi\,t_{\text{exp}} + N\,D\,t_{\text{exp}} + N\,\sigma_R^2}}
$$

With WISPR EOL $\sigma_R = 13$ e⁻, $D = 1.9$ e⁻/s/pix at < −55 °C, full well 21,300 e⁻, and bandpass-integrated K-corona surface brightness, the team predicts SNR = 20 at inner FOV at perihelion (1-s exposures) and SNR = 5 at outer edge at 0.25 AU after 4×4 binning + summation. / WISPR EOL 잡음 매개변수: σ_R = 13 e⁻, D = 1.9 e⁻/s/pix, full well 21,300 e⁻ → SNR 20→5.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1929  Russell: predicts dust-free zone < 4 R⊙ (never observationally confirmed)
1958  Parker: solar wind theoretical paper
1963  Snyder, Neugebauer & Rao: Mariner 2 in-situ confirmation of solar wind
1974-76 Helios 1/2: closest approach 0.29 AU; F-corona, dust photometry (Leinert et al. 1981)
1985  Jackson & Leinert: Helios 90° photometer detects CME crossing → Thomson surface concept seed
1995  SOHO/LASCO C1/C2/C3 (Brueckner et al.); SOHO/UVCS (Kohl et al.)
1997  Sheeley et al.: streamer blob velocity profiles (LASCO)
1998  Leinert et al.: 1997 reference for diffuse night sky brightness (F-corona model basis)
2000  Socker et al.: STEREO/HI baffle design (heritage for WISPR)
2003  Vourlidas & Howard (in prep, formal 2006): proper Thomson-scattering treatment
2006  Vourlidas & Howard: Thomson surface concept for wide-FOV imagers
2006  Thernisien & Howard: ray-like F-corona structures
2006  STEREO launch (SECCHI: COR1/COR2/HI1/HI2)
2008  Kaiser et al.: STEREO mission introduction
2008  Howard et al.: SECCHI instrument paper (direct WISPR ancestor)
2009  Howard & Tappin: white-light heliospheric imaging formalism
2010  Viall et al.: 5-h periodic density structures in HI1
2013  Korendyke et al.: SoloHI APS detector development (direct WISPR detector heritage)
2013  Howard et al.: SoloHI instrument paper (mechanical/optical heritage)
2014  Bale et al.: FIELDS (constrains WISPR mounting)
2014  Fox et al.: Solar Probe Plus mission paper
2014  WISPR Preliminary Design Review (PDR), this paper accepted Oct 2014
*** 2016 (online 2015): THIS PAPER — Vourlidas et al. WISPR instrument paper ***
2018  Parker Solar Probe launch (renamed from SPP)
2019  Howard et al. (Nature): first PSP/WISPR coronal observations, switchbacks
2020  Solar Orbiter / SoloHI launch
2021  PSP crosses Alfvén surface (de Forest, Cranmer, Velli, Liewer)
2024+ Continuing PSP perihelia at 9.86 R⊙
```

This paper sits at the convergence of SECCHI/HI heritage (baffles, IPP), SoloHI heritage (APS detector, electronics, mechanical), Helios heritage (F-corona, Thomson geometry), and the new SPP mission framing. It is the "design moment" — written between PDR (Jan 2014) and CDR (Dec 2014) — that locks the science requirements and design choices subsequently realized in flight from 2018 onward.

이 논문은 SECCHI/HI(baffle, IPP), SoloHI(APS·전자·기계), Helios(F-corona, Thomson), 그리고 새 SPP 미션의 합류점에 위치한다. PDR(2014.1)과 CDR(2014.12) 사이에 작성된 '설계 순간'의 문서로서, 2018년 이후 비행으로 실현될 과학 요구와 설계 결정을 고정한다.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Howard et al. (2008), SECCHI Instrument Paper, Space Sci. Rev. 136 | Direct architectural ancestor: COR1/COR2/HI1/HI2 baffle and lens heritage | 매우 높음 — WISPR baffle·IPP·calibration 철학의 직접적 모체 |
| Korendyke et al. (2013), SPIE | SoloHI APS detector development → WISPR detector | 매우 높음 — WISPR APS는 SoloHI APS와 동일 family |
| Howard et al. (2013), SoloHI SPIE paper | Sister instrument; mechanical/optical heritage explicitly cited | 매우 높음 — "SoloHI provides many of the design elements adapted into the WISPR design" |
| Vourlidas & Howard (2006), ApJ 642, 1216 | Proper treatment of CME brightness; Thomson surface concept | 높음 — WISPR 광학 기하의 이론적 기초 |
| Howard & Tappin (2009), Space Sci. Rev. 147 | Heliospheric imager formalism (LOS integration) | 높음 — Thomson 산란 적분의 표준 형식 |
| Socker et al. (2000), SPIE 4139 | STEREO/HI baffle paper | 높음 — WISPR 3중 baffle 설계의 직접 인용 |
| Bale et al. (2016), FIELDS, Space Sci. Rev. | The reason for the two-telescope split | 매우 높음 — FIELDS 안테나 위치가 WISPR 광학 분할을 강제 |
| Fox et al. (2016), SPP mission paper, Space Sci. Rev. | Mission framing, orbit, 9.86 R⊙ perihelion | 높음 — 모든 WISPR 과학 요구의 미션 컨텍스트 |
| Kasper et al. (2016), SWEAP, Space Sci. Rev. | In-situ counterpart for WISPR images | 높음 — WISPR + SWEAP 동시 관측이 핵심 과학 결과 |
| McComas et al. (2016), ISIS, Space Sci. Rev. | SEP measurement linked to WISPR shock images | 중간 — L-1 #3 과학 목표(입자 가속) |
| Mehoke et al. (2012), AERO Conf. | SPP dust impact model | 중간 — WISPR 유리 선택의 기준 |
| Viall et al. (2009, 2010) | 5-h periodic density structures observed by SECCHI/HI1 | 중간 — WISPR "wave turbulence" 프로그램의 직접 자극 |
| Thernisien et al. (2006), Sol. Phys. 233 | LASCO calibration via stars | 중간 — WISPR 측광 보정 ~3 % 절차의 모델 |
| Sheeley et al. (1997), ApJ 484 | Streamer blob velocities | 중간 — WISPR L-1 #1 과학 목표의 LASCO 기준선 |
| Leinert et al. (1981, 1998) | Helios F-corona reference / 1 AU diffuse-sky brightness | 중간 — WISPR F-corona 과학 목표의 비교 기준 |

---

## 7. References / 참고문헌

- A. Vourlidas, R.A. Howard, S.P. Plunkett, et al., "The Wide-Field Imager for Solar Probe Plus (WISPR)", Space Sci. Rev. 204, 83–130 (2016). DOI: 10.1007/s11214-014-0114-y. (This paper)
- E.N. Parker, "Cosmic ray modulation by solar wind", Phys. Rev. 110(6), 1445–1449 (1958).
- C.W. Snyder, M. Neugebauer, U.R. Rao, "The solar wind velocity and its correlation with cosmic-ray variations and with solar and geomagnetic activity", J. Geophys. Res. 68, 6361 (1963).
- N. Fox et al., "The solar probe plus mission", Space Sci. Rev. (2015, this issue).
- R.A. Howard et al., "Sun Earth Connection Coronal and Heliospheric Investigation (SECCHI)", Space Sci. Rev. 136, 67–115 (2008).
- A. Howard et al., "The solar and heliospheric imager (SoloHI) instrument for the solar orbiter mission", Proc. SPIE (2013).
- C.M. Korendyke et al., "Development of the SoloHI active pixel sensor", Proc. SPIE (2013).
- S. Bale et al., "The FIELDS investigation", Space Sci. Rev. (2015, this issue).
- J. Kasper et al., "The SWEAP investigation", Space Sci. Rev. (2015, this issue).
- D. McComas et al., "The ISIS investigation", Space Sci. Rev. (2015, this issue).
- A. Vourlidas, R.A. Howard, "The proper treatment of coronal mass ejection brightness: a new methodology and implications for observations", ApJ 642, 1216 (2006).
- T. Howard, S.J. Tappin, "Interplanetary coronal mass ejections observed in the heliosphere: 1. Review of theory", Space Sci. Rev. 147, 31–54 (2009).
- D.G. Socker et al., "The NASA Solar Terrestrial Relations Observatory (STEREO) mission heliospheric imager", Proc. SPIE 4139, 284–293 (2000).
- B.V. Jackson, C. Leinert, "HELIOS images of coronal mass ejections", JGR 90, 10 (1985).
- N.M. Viall, H.E. Spence, J.T. Kasper, "Are periodic solar wind number density structures formed in the solar corona?", Geophys. Res. Lett. 36, 23102 (2009).
- N.M. Viall et al., "Examining periodic solar-wind density structures observed in the SECCHI heliospheric imagers", Sol. Phys. 267, 175 (2010).
- N.R. Sheeley Jr. et al., "Measurements of flow speeds in the corona between 2 and 30 R⊙", ApJ 484, 472 (1997).
- A.F. Thernisien, R.A. Howard, "Electron density modeling of a streamer using LASCO data of 2004 January and February", ApJ 642, 523 (2006).
- A.F. Thernisien et al., "Photometric calibration of the Lasco-C3 coronagraph using stars", Sol. Phys. 233, 155 (2006).
- C. Leinert, B. Moster, "Evidence for dust accumulation just outside the orbit of Venus", Astron. Astrophys. 472, 335 (2007).
- C. Leinert et al., "The 1997 reference of diffuse night sky brightness", Astron. Astrophys. Suppl. Ser. 127, 1 (1998).
- I. Mann et al., "Dust cloud near the Sun", Space Sci. Rev. 110, 269 (2004).
- D.S. Mehoke et al., "A review of the solar probe plus dust protection approach", Aerospace Conf. IEEE (2012). DOI: 10.1109/AERO.2012.6187076.
- H.N. Russell, "On meteoric matter near the Sun", ApJ 69, 49 (1929).
- A.J. Steffl et al., "A search for vulcanoids with the STEREO heliospheric imager", Icarus 223, 48 (2013).
- C.E. DeForest et al., "Inbound waves in the solar corona: a direct indicator of Alfvén surface location", ApJ 787, 124 (2014).
- M.J. Aschwanden, et al., (general coronal physics references — Cargill 2009 review used as primer in the paper).
