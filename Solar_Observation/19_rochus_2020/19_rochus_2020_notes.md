---
title: "The Solar Orbiter EUI Instrument: The Extreme Ultraviolet Imager"
authors: P. Rochus, F. Auchère, D. Berghmans, L. Harra, W. Schmutz, U. Schühle, et al. (EUI Consortium)
year: 2020
journal: "Astronomy & Astrophysics, Vol. 642, A8"
doi: "10.1051/0004-6361/201936663"
topic: Solar_Observation
tags: [solar_orbiter, eui, euv_imaging, cmos_aps, multilayer_optics, coronal_heating, lyman_alpha, space_instrument, campfires]
status: completed
date_started: 2026-04-17
date_completed: 2026-04-17
---

# 19. The Solar Orbiter EUI Instrument: The Extreme Ultraviolet Imager / Solar Orbiter EUI 기기: 극자외선 영상기

---

## 1. Core Contribution / 핵심 기여

**English**
EUI (Extreme Ultraviolet Imager) is the three-telescope EUV imaging suite aboard ESA-NASA's Solar Orbiter (launched February 2020). It consists of (a) the **Full Sun Imager (FSI)**, a single-mirror off-axis Herschelian telescope imaging the whole Sun at **17.4 nm and 30.4 nm** across a 3.8° × 3.8° field of view, (b) the **High-Resolution Imager in EUV (HRI_EUV)**, an off-axis Cassegrain at 17.4 nm with 1000″ × 1000″ FOV, and (c) the **High-Resolution Imager in Lyman-α (HRI_Lya)**, an off-axis Gregorian at 121.6 nm. The paper describes the full instrument design — optics, mirror multilayer coatings, thin-film filters, back-thinned dual-gain 3072 × 3072 CMOS-APS detectors, filter wheels, thermal/mechanical architecture, electronics (LEON3 processor with a WICOM JPEG2000-like compression ASIC), RTEMS flight software, cleanliness control, and a six-program observation concept — and reports pre-flight performance verification including end-to-end radiometric calibration at PTB/BESSY.

EUI's design is shaped by a chain of hard constraints: Solar Orbiter's deep-space vantage forces tight mass/power/telemetry budgets; the 0.28 AU perihelion delivers 13× the solar flux at Earth, so entrance apertures must protrude through the heat shield while staying small; limited downlink (only ~0.1 bpp feasible for the Synoptic program) demands aggressive onboard processing and event-driven observation. The paper shows how each choice — single-mirror FSI (4× pupil-area reduction vs. two-mirror), hexagonal entrance pupil (mesh-grid matching), APS detectors (random-access, radiation-hard), and dual-gain readout (5 e⁻ read noise + 120 ke⁻ full well) — flows from these constraints toward the science goal of resolving solar-atmospheric fine structure at ≤200 km on the Sun with sub-second cadence, while providing the first sustained out-of-ecliptic EUV imaging of the poles.

**한국어**
EUI(Extreme Ultraviolet Imager)는 2020년 2월 발사된 ESA-NASA Solar Orbiter 탑재의 세 망원경 EUV 영상 패키지이다. (a) **Full Sun Imager(FSI)** — 단일 반사경 오프축 헤르셸리언 망원경으로 3.8° × 3.8° 시야에서 **17.4 nm와 30.4 nm** 듀얼 밴드 전태양 촬영, (b) **HRI_EUV** — 17.4 nm 오프축 카세그레인, 1000″ × 1000″ 시야, (c) **HRI_Lya** — 121.6 nm 오프축 그레고리안. 논문은 광학, 다층막 반사 코팅, 박막 필터, 3072 × 3072 dual-gain CMOS-APS 검출기, 필터휠, 열·기계 구조, 전자부(LEON3 프로세서 + WICOM JPEG2000-류 압축 ASIC), RTEMS 비행 소프트웨어, 오염 관리, 6종 관측 프로그램, 그리고 PTB/BESSY 복사 교정까지 **기기 전체를 종합적으로 기술**한다.

EUI의 설계는 일련의 강한 제약 사슬에서 나온다 — Solar Orbiter의 심우주 임무 특성이 질량·전력·텔레메트리 예산을 압박하고, 0.28 AU 근일점에서는 지구 궤도 대비 **13배 태양 플럭스**가 걸려 입사구는 열차폐를 뚫고 나오되 최대한 작아야 하며, 다운링크가 협소해(Synoptic 프로그램은 ~0.1 bpp 수준) 공격적 온보드 처리와 이벤트 기반 관측이 필수다. 논문은 각 설계 선택 — FSI의 단일 반사경(2-거울 대비 동공 면적 1/4), 육각 입사동(메시 격자 정합), APS 검출기(랜덤 액세스·방사선 내성), 듀얼 게인 판독(5 e⁻ 판독 잡음 + 120 ke⁻ 포화) — 이 어떻게 **태양 표면 ≤200 km 미세 구조를 sub-second cadence로 분해**하고 최초의 황도면 이탈 극지역 EUV 촬영까지 달성하는지를 보여준다.

---

## 2. Reading Notes / 읽기 노트

### Part I: Scientific Objectives (§2) / 과학 목표

**English**
EUI answers three overlapping Solar Orbiter science questions. (i) **Corona-heliosphere connection**: The "transition corona" at 1–3 R☉ — the poorly observed gap between coronagraphs and EUV imagers — holds the physical interface where magnetic control of the plasma yields to outflow dominance. FSI's 3.8° × 3.8° FOV corresponds to **(14.3 R☉)² at 1 AU and still (4.0 R☉)² at perihelion**, providing a uniquely large coronal EUV field that overlaps Metis coronagraph observations. (ii) **Solar atmosphere at high resolution**: The HRIs achieve 1″ = 2-pixel resolution, a **pixel footprint of (100 km)² on the Sun at perihelion**. Temporal sampling at the 150 km/s sound speed in a 1 MK plasma implies cadence of seconds — both HRI cameras operate at ≥1 Hz in full frame, faster in subfields. (iii) **High-latitude perspective**: Venus gravity assists progressively raise Solar Orbiter's inclination above 30°, giving **unprecedented top-down views of the solar poles**, where polar-plume dynamics, bright points, and convective-cell boundaries (chromosphere via HRI_Lya) constrain the global dynamo.

**한국어**
EUI는 Solar Orbiter의 세 가지 과학 질문에 대응한다. (i) **코로나-헬리오스피어 연결**: 기존 코로나그래프와 EUV 영상기 사이의 공백인 1–3 R☉ "전이 코로나" — 플라즈마에 대한 자기 지배가 흐름 지배로 넘어가는 물리적 경계 — 를 처음으로 깊이 있게 관측. FSI 시야는 **1 AU에서 (14.3 R☉)², 근일점에서도 (4.0 R☉)²** 로, Metis 코로나그래프와 겹치는 유례없는 크기다. (ii) **고해상 태양 대기**: HRI의 1″ = 2픽셀 분해능은 근일점에서 **태양 표면 (100 km)² 픽셀 풋프린트**를 의미. 1 MK 플라즈마의 음속 150 km/s를 제대로 샘플링하려면 초 단위 케이던스가 필요하며, HRI 두 카메라 모두 전체 프레임 1 Hz 이상(서브필드에서는 더 빠름)을 지원. (iii) **고위도 관점**: 금성 중력 보조로 경사각이 30°를 넘어서며 태양 극지역 하향 관측 — 극 플룸 역학, 밝은 점, HRI_Lya의 대류 셀 경계 등이 전역 다이나모를 제약한다.

### Part II: Optical Design (§3) / 광학 설계

**Nominal parameter summary (from Tables 1 and 2) / 공식 파라미터 요약:**

| Parameter | FSI | HRI_EUV | HRI_Lya |
|---|---|---|---|
| Topology | Off-axis single-mirror Herschelian | Off-axis Cassegrain (two-mirror) | Off-axis Gregorian (two-mirror) |
| Wavelength(s) | 17.4 + 30.4 nm (dual-band) | 17.4 nm | 121.6 nm (Lyman-α) |
| Entrance pupil | Hexagonal, 2.75 mm edge | 47.4 mm diameter | 30 mm diameter |
| Focal length | 462.5 mm | 4187 mm | 5804 mm |
| Plate scale | 4.46″/10-μm pixel | 50 arcsec/mm (0.5″/pixel) | 31.5 arcsec/mm |
| FOV (on 2048² or 3072² array) | 3.8° × 3.8° (3072²) | 1000″ × 1000″ | 1000″ × 1000″ (effective 992″ × 992″) |
| Primary mirror | 66×66 mm² (53×53 useful), K=−0.732, RoC 925 mm | 80 mm off-axis Cass primary, RoC 1518.067 mm, CC=−1 | 80 mm off-axis Greg primary, RoC 1143 mm, CC=−1 |
| Secondary mirror | — | 11.44 mm off-axis, RoC 256.774 mm, CC=−2.04 | 7 mm off-axis, RoC 91 mm, CC=−0.65 |
| Detector | 3072×3072 APS, 10 μm | 2048×2048 subarray | 2048×2048 subarray (via fiber taper, 14.1 μm virtual pixel) |
| Filter wheel operations (qualified) | ~1,000,000 | 400 | 400 |

**한국어** Table 1과 Table 2에서 추출한 공식 설계값 — FSI는 단일 반사경 Herschelian, HRI_EUV는 2-거울 Cassegrain, HRI_Lya는 2-거울 Gregorian. 세 망원경 모두 오프축 설계로, 입사동과 검출기를 광축에서 벗어나게 배치해 뒷쪽 장애물 없이 빛이 검출기에 도달.

**FSI — single-mirror Herschelian**

**English**
Classical two-mirror designs failed to maintain image quality over 3.8° × 3.8° within the mass/volume budget. Auchère et al. (2005)'s solution was a single off-axis concave ellipsoidal mirror (conic constant K = −0.732, 66 × 66 mm substrate with 53 × 53 mm useful area, RoC 925 mm). Using one mirror allowed a **4× reduction in entrance-pupil area**, critical for keeping the solar heat input manageable. The hexagonal entrance pupil (2.75 mm edge length) is located **in front of** the telescope at 737.35 mm from the mirror — this both geometrically halves the peak irradiance at the entrance filter and permits its hex geometry to match the hexagonal support mesh (0.4 mm pitch) of the aluminium entrance filter, **nulling mesh-grid diffraction modulation** when pupil footprint is an integer multiple of grid period (Auchère et al. 2011). Spectral selection is by a dual-band Al/Mo/SiC multilayer + redundant Al/Zr/Al (for 17.4 nm) or Al/Mg/Al (for 30.4 nm) filter-wheel filters. Effective focal length 462.5 mm → plate scale **4.46″/pixel**. A deployable **occulting disc (8.87 mm diameter) on the door** suppresses direct solar disc intensity by 6 × 10⁻³ at 30.4 nm and 4 × 10⁻³ at 17.4 nm for enabled extended-corona observations (up to 10 R☉) — this became FSI's role after Metis descoped its 30.4 nm channel.

**한국어**
전통적 2-거울 설계는 질량·부피 내에서 3.8° 시야 전체의 상 질을 유지할 수 없었다. Auchère et al. (2005)의 해법은 **단일 오프축 오목 타원 반사경**(K=−0.732, 66×66 mm 기판에 53×53 mm 유효 면적, RoC 925 mm). 단일 반사경으로 **입사동 면적을 1/4로 감소** — 태양 열부하 관리의 핵심이다. 2.75 mm 변의 육각 입사동을 반사경 앞 737.35 mm에 배치 → (1) 필터 중심의 조도를 기하적으로 절반으로 줄이고, (2) 알루미늄 입사 필터의 육각 지지 메시(0.4 mm 피치)와 **육각 형상이 정합**되어 입사동 풋프린트가 격자 주기의 정수배일 때 **메시 회절 변조가 상쇄**됨(Auchère et al. 2011). 분광 선택은 Al/Mo/SiC 듀얼 밴드 반사막 + 필터휠의 Al/Zr/Al(17.4 nm용) 또는 Al/Mg/Al(30.4 nm용) 필터 조합. 초점거리 462.5 mm → 스케일 **4.46″/픽셀**. 도어에 설치된 **occulting disc(8.87 mm)** 가 태양 원반을 30.4 nm에서 6 × 10⁻³, 17.4 nm에서 4 × 10⁻³ 수준으로 차단해 **10 R☉까지 확장 코로나 관측**을 가능케 한다 — Metis가 30.4 nm 채널을 포기한 이후 FSI가 이 역할을 맡았다.

**HRI_EUV — off-axis Cassegrain at 17.4 nm**

**English**
Two-mirror Cassegrain (not Ritchey-Chrétien; chosen for **mirror-positioning tolerance** despite losing some stray-light rejection since no intermediate focus exists). Primary RC 1518.067 mm, CC = −1; secondary RC 256.774 mm, CC = −2.04. Pupil 47.4 mm, focal length 4187 mm → plate scale **50 arcsec/mm = 0.5″/pixel (10 μm pixels)**. FOV 1000″ × 1000″, corresponding to (17′)² on 2048 × 2048 pixel subfield → at 1 AU this is (1 R☉)² and at 0.28 AU (0.28 R☉)². Periodic Al/Mo/SiC multilayer (30 periods, 8.86 nm thickness) tuned to 17.4 nm with a 3 nm SiC capping for thermal-cycling stability. Entrance filter is Al foil on a 20 lpi nickel mesh grid; four-slot filter wheel at exit pupil (open + 150 nm Al + occulter + dark) provides redundancy. FSI and HRI_EUV share the same FPA optical path.

**한국어**
2-거울 카세그레인 구성(릿치-크레티앵이 아닌 이유: 중간 초점이 없어 스트레이 라이트 억제는 약하지만 **거울 위치 공차가 관대**해 제작/정렬이 쉽다). 주경 RC 1518.067 mm(CC=−1), 부경 RC 256.774 mm(CC=−2.04). 입사동 47.4 mm, 초점거리 4187 mm → 스케일 **50″/mm = 0.5″/픽셀(10 μm 픽셀)**. 시야 1000″ × 1000″(2048×2048 서브필드에서 17′×17′) → 1 AU에서 (1 R☉)², 0.28 AU에서 (0.28 R☉)². 주기적 Al/Mo/SiC 다층막(30주기, 8.86 nm 두께)을 17.4 nm에 조정, 열 사이클 안정성을 위해 3 nm SiC 캡핑층. 입사 필터는 20 lpi 니켈 메시 위 Al 박막, 출사동 측에는 4-슬롯 필터휠(개방 + 150 nm Al + occulter + 암실)이 이중화. FSI와 HRI_EUV는 공통 FPA 구조.

**HRI_Lya — off-axis Gregorian at 121.6 nm**

**English**
Gregorian chosen because the **intermediate focus enables a field stop** that, combined with internal baffles, drastically reduces out-of-FOV stray light — essential for imaging Lyman-α against a bright solar continuum ~12 orders brighter. Mirrors use Al/MgF₂ coating (>86% reflectivity at 121.6 nm). A broadband interference entrance filter (Pelham Type 122 NB-40D) blocks visible/IR/EUV/X-rays; a narrow-band focal-plane filter (Acton Type 122 XN-2D) isolates Lyα to **>90% spectral purity**. The detector is an **intensified APS (I-APS)**: a KBr-photocathode microchannel plate (MCP) converts incoming UV to electrons accelerated onto a P46 phosphor screen, fiber-optically demagnified (factor 1.41) onto the CMOS sensor, yielding a 2048 × 2048 subarea with **1″/2pixel**. Intensifier gain varies ~10×/100 V between 400–750 V operating range.

**한국어**
그레고리안 선택 이유: **중간 초점이 시야 스톱을 허용** — 내부 배플과 결합해 시야 밖 스트레이 라이트를 크게 억제한다. Lyman-α는 ~12 차수 더 밝은 태양 연속 스펙트럼을 배경으로 촬영해야 하므로 이 기능이 필수. Al/MgF₂ 코팅(121.6 nm에서 반사율 86% 이상). 광대역 간섭 입사 필터(Pelham Type 122 NB-40D)가 가시광/IR/EUV/X-ray 차단, 초점면 협대역 필터(Acton Type 122 XN-2D)가 Lyα만 통과시켜 **90% 이상 스펙트럼 순도**. 검출기는 **강화 APS(I-APS)** — KBr 광음극 MCP가 UV를 전자로 변환, 가속 후 P46 인광 스크린에 충돌, 광섬유 광학계로 1.41배 축소되어 CMOS 센서에 전달 → 2048 × 2048 유효 영역, **1″/2픽셀**. 증폭기 게인은 400–750 V에서 100 V당 ~10× 변화.

### Part III: Mechanical, Thermal, Filter Wheels (§4) / 기계, 열, 필터휠

**English**
EUI uses **passive thermal design with detector cooling**. OBS (Optical Bench System) is a CFRP sandwich panel thermally isolated from the spacecraft via three titanium A-shape mounts, operating in −20 to +50 °C. At 0.28 AU, the instrument front (doors, baffles) receives **17.5 kW/m²** (13 solar constants) — decoupled from OBS and thermally coupled to spacecraft radiators via heat pipes. **Detectors must be cooled below −40 °C (target −60 °C, ±5 °C stability)** via dedicated thermal straps to spacecraft radiators (through Cold Element and Medium Element interfaces). The two HRIs share a common CE/ME interface; FSI has its own. Post-launch decontamination heaters (84.3 Ω, 9.3 W each @ 28 V) can bake detectors to limit contamination buildup. Filter wheels use custom dual-winding Phyton GmbH stepper motors (200 steps + micro-stepping for ±0.17° positioning accuracy) qualified for **~1 million operations** (FSI) / 400 operations (HRI_EUV).

**한국어**
EUI는 **능동 제어 없는 수동 열설계 + 검출기 냉각**. OBS(광학 벤치 시스템)는 CFRP 샌드위치 패널로 구성되어 3개 Ti A-형 마운트로 우주선과 열절연되어 −20 ~ +50 °C에서 작동. 0.28 AU에서 기기 전면(도어·배플)에 **17.5 kW/m²(13 태양 상수)** 가 걸리며, OBS로부터 열적으로 분리되고 히트 파이프로 우주선 방열판에 연결된다. **검출기는 −40 °C 이하로 냉각(목표 −60 °C, ±5 °C 안정도)** — Cold Element와 Medium Element 인터페이스를 통한 전용 열 스트랩으로 우주선 방열판에 연결. 두 HRI는 공통 CE/ME, FSI는 독립. 발사 후 오염 방지 가열기(각 84.3 Ω, 28 V에서 9.3 W)가 검출기를 베이킹해 오염 축적을 제한. 필터휠은 Phyton GmbH 커스텀 이중 권선 스테퍼 모터(200 스텝 + 마이크로 스테핑, ±0.17° 정밀도) — FSI **약 100만회**, HRI_EUV 400회 작동 인증.

### Part IV: Electronics & Detectors (§5) / 전자부와 검출기

**English**
All EUI electronics live in the CEB (Common Electronics Box, 0.007 m³ Al housing with Aeroglaze Z306 thermal coating, four PCBs: Processor, Compression, Auxiliary, Power), except the Front End Electronics (FEE) co-located with each camera.

The EUI detector is a **back-thinned dual-gain CMOS-APS**, 3072 × 3072 pixels at 10 μm pitch (HRI channels window down to 2048 × 2048). Two "stitching blocks" of 3072 × 1536 pixels form each device. Per-pixel **dual-gain readout**: high-gain channel for low read noise (<5 e⁻ rms required, shot noise floor), low-gain for high saturation (full well 120 ke⁻, shared with high gain and ADC-clipped). Theoretical high/low gain ratio 22.3, device-dependent and configurable. Dark current <10 e⁻/pixel/s at −40 °C; MTF at Nyquist >50% at 17.4 nm; QE >80%; crosstalk at working wavelength <5%. HRI_Lya uses the same APS with an intensifier mounted in front (front-sided illumination, demagnification 1.41 via fiber taper).

**한국어**
모든 EUI 전자부는 CEB(Common Electronics Box, 0.007 m³ Al 하우징, Aeroglaze Z306 열 코팅, 4개 PCB: Processor, Compression, Auxiliary, Power)에 집중되어 있으며, 각 카메라와 같이 위치하는 Front End Electronics(FEE)는 예외.

EUI 검출기는 **back-thinned dual-gain CMOS-APS**, 3072 × 3072 (10 μm 픽셀; HRI는 2048 × 2048로 윈도우). 각 소자는 3072 × 1536의 두 "스티칭 블록" 조합. 픽셀당 **듀얼 게인 판독** — 고게인은 저잡음(<5 e⁻ rms, 샷잡음 한계) 용, 저게인은 고포화(포화 120 ke⁻, 고게인과 공유하되 ADC 클립) 용. 이론적 고/저게인 비 22.3, 소자·설정 의존. −40 °C에서 암전류 <10 e⁻/픽셀/초, 17.4 nm에서 Nyquist MTF >50%, QE >80%, 작동 파장 크로스톡 <5%. HRI_Lya는 동일 APS에 강화기(전면 조사, 광섬유 테이퍼로 1.41배 축소).

**Processor & Compression**

**English**
- Processor Board: **Cobham-Gaisler UT699 LEON3 SPARC** CPU with 4 SpaceWire interfaces; gigabytes of SDRAM for One-Hour Queue (OHQ) and Spacecraft Output Buffer (SOB); MRAM for boot/app code; Actel FPGA for memory/buffer interfaces.
- Compression Board: FPGA + **WICOM ASIC** (JPEG2000-like MRCPB, developed for PLEIADES/EADS-Astrium). WICOM performs on-the-fly image pre-processing (per-pixel and global gain/offset correction, bad-pixel replacement, cosmic-ray median filtering, rebinning 2×2 or 4×4, thumbnails, histograms, recoding) followed by wavelet compression. Design goal: compress the next image within **half a second** during the previous exposure. Typical ratios: **<0.1 bpp** with binning+recoding (well below the sensor's 14-bit native depth → >100× compression).
- Auxiliary Board: drives stepper motors for filter wheels and doors; monitors electrical currents; performs switching (heaters, launch locks, camera power).
- Power Board: four isolated DC/DC PSUs (one CEB, three cameras).

**한국어**
- 프로세서 보드: **Cobham-Gaisler UT699 LEON3 SPARC** CPU, SpaceWire 4개 인터페이스, OHQ(One-Hour Queue)·SOB(Spacecraft Output Buffer)용 기가바이트급 SDRAM, 부트/응용 코드용 MRAM, 메모리 인터페이스용 Actel FPGA.
- 압축 보드: FPGA + **WICOM ASIC** (JPEG2000-류 MRCPB, PLEIADES/EADS-Astrium용으로 개발). WICOM이 온더플라이 전처리(픽셀 단위·전역 게인/오프셋, 불량 픽셀 치환, 우주선 중앙값 필터, 2×2/4×4 비닝, 썸네일, 히스토그램, 재코딩) + 웨이블릿 압축 수행. 설계 목표: 이전 노출 중에 **0.5초 이내**에 다음 이미지 처리 완료. 전형적 압축비 **<0.1 bpp**(비닝+재코딩 포함, 14-bit 원본 대비 100배 이상 압축).
- 보조 보드: 필터휠·도어 스테퍼 모터 구동, 전류 모니터링, 가열기·발사잠금·카메라 전원 스위칭.
- 전원 보드: 격리형 DC/DC PSU 4개(CEB용 1개, 카메라 3개).

### Part V: Flight Software, Cleanliness, Performance (§6–§8) / 비행 SW, 오염 관리, 성능

**English**
- **RTOS**: Gaisler-modified **RTEMS 4.10.2** on LEON3, compiled with GNU tools on Linux; ECSS + MISRA compliant. Software split into *Basic* (boot, health monitoring; in ROM) and *Oper* (full functionality; re-writable, patchable).
- **Autonomy**: Only ~150 telecommands allowed per day. All observation programs stored as **Science Tables** — once in "Sci" mode, a single telecommand triggers an entire program with exposure sequences, loops, waits.
- **Cleanliness**: Contamination limits (Table 3) set by <5% throughput loss: entrance baffle tolerates 370 ng/cm² molecular + 300 ppm particulate. Instrument integrated in ISO-5 cleanroom, 80 °C bake-out before delivery, continuous purge gas during ground operations.
- **Pre-flight performance**:
  - Read noise <5 e⁻ rms (high gain), dark <10 e⁻/pix/s at −40 °C — both confirmed
  - QE >80% at 17.4 nm confirmed
  - HRI_EUV spectral response measured at PTB from 10–100 nm (baseline spectral purity >90% between Fe IX at 17.11 nm and Fe X at 17.45/17.72 nm for quiet Sun/active region)
  - HRI_Lya purity improved by narrow-band focal filter; intensifier gain calibrated 300–900 V at MPS with monochromatic Lyα source
  - End-to-end radiometric calibration at PTB/MLS in Berlin; full vacuum tank beamline calibration

**한국어**
- **RTOS**: Gaisler 수정 **RTEMS 4.10.2** on LEON3, Linux에서 GNU 툴로 컴파일; ECSS + MISRA 준수. 소프트웨어는 *Basic*(부트·건강 모니터링; ROM)과 *Oper*(전 기능; 재쓰기·패치 가능)로 이중화.
- **자율성**: 하루 ~150 개 텔레커맨드만 허용. 모든 관측 프로그램은 **Science Table**로 저장 — "Sci" 모드에 진입하면 단일 텔레커맨드로 노출 시퀀스·루프·대기까지 전체 프로그램 실행.
- **오염 관리**: 오염 한계(Table 3)는 <5% 처리량 손실 기준 — 입사 배플은 370 ng/cm² 분자성 + 300 ppm 입자성 허용. 기기는 ISO-5 클린룸에서 조립, 납품 전 80 °C 베이크아웃, 지상 운용 시 연속 퍼지가스 공급.
- **비행 전 성능**:
  - 판독 잡음 <5 e⁻ rms(고게인), −40 °C에서 암전류 <10 e⁻/픽셀/초 — 모두 확인됨
  - 17.4 nm에서 QE >80% 확인
  - HRI_EUV 스펙트럼 응답 PTB에서 10–100 nm 측정(조용한 태양/활동 영역에서 Fe IX 17.11 nm와 Fe X 17.45/17.72 nm 사이 기본 스펙트럼 순도 >90%)
  - HRI_Lya 순도는 협대역 초점 필터로 개선; 증폭기 게인은 MPS에서 단색 Lyα 광원으로 300–900 V 교정
  - 베를린 PTB/MLS의 전자 저장링 빔라인에서 엔드-투-엔드 복사 교정, 전용 진공 탱크 수행

### Part V-bis: Cleanliness and Contamination (§7) / 청결도와 오염 관리

**English**
As a UV instrument, EUI is exceptionally sensitive to **organic molecular contamination** (strongly absorbing in the EUV) and **particulate contamination** (scatters stray light, reducing contrast). The paper defines a quantitative contamination budget based on a 50% maximum acceptable throughput loss, uniformly distributed across six optical surfaces (7% loss per surface). This yields the tolerance figures in Table 3:

| Sensitive Area | Molecular (ng/cm²) | Particulate (ppm) |
|---|---|---|
| Entrance baffle | 370 | 300 |
| Internal surfaces | 370 | 100 |
| External surfaces | 500 | 300 |

Contamination-control features implemented:
1. **Three independent reusable entrance door mechanisms** (one per channel) — unlike one-shot doors, they can be closed during non-operations to limit UV/particulate ingress, and they *reflect* the perihelion heat flux when closed (acting as a second-line heat-rejection surface after the spacecraft heat shield). A labyrinth opening around the lid circumference allows venting without letting air flow quickly enough to damage filters.
2. **CFRP structural panels with venting-only channels** — each telescope has its own compartmentalized purge inlet/outlet, with flow restrictions to prevent pressure-change-induced filter rupture.
3. **Class ISO-5 cleanroom assembly** + pre-delivery bake-out at 80 °C, monitored by a temperature-controlled quartz crystal microbalance (TQCM) held at −20 °C; bake-out ends when TQCM mass deposition rate <1% for 3 hours.
4. **Detector "cold cup"** slightly colder than the FPA itself, designed to **trap outgassing contaminants before they reach the sensor**. Dedicated annealing heaters (84.3 Ω, 9.3 W at 28 V per camera) enable periodic bake-out of any accumulated condensate.

**한국어**
UV 기기로서 EUI는 **유기 분자 오염**(EUV에서 강한 흡수)과 **입자성 오염**(스트레이 라이트 산란으로 대비 저하)에 극도로 민감. 논문은 최대 수용 가능 처리량 손실 50%를 6개 광학 표면에 균등 분배(표면당 7%)하는 정량 오염 예산을 정의 — Table 3의 허용치:

| 민감 영역 | 분자 (ng/cm²) | 입자 (ppm) |
|---|---|---|
| 입사 배플 | 370 | 300 |
| 내부 표면 | 370 | 100 |
| 외부 표면 | 500 | 300 |

구현된 오염 관리 기능:
1. **세 개의 독립된 재사용 가능 입사 도어** (채널당 하나) — 일회성 도어와 달리 비운용 시 닫아 UV·입자 유입을 차단하고, 닫힌 상태에서 **근일점 열유속을 반사**(우주선 열차폐에 이은 2차 열거절 면으로 작동). 도어 주변 labyrinth 구조가 필터 손상 없이 환기 가능.
2. **CFRP 구조 패널 + 환기 전용 채널** — 각 망원경이 독립 구획의 퍼지 입/출구를 갖고, 유량 제한으로 압력 급변 필터 파열을 방지.
3. **ISO-5 클린룸 조립** + 납품 전 80 °C 베이크아웃, −20 °C로 제어되는 TQCM(수정결정 미세저울)로 모니터링; TQCM 침적률이 3시간 동안 <1%면 베이크아웃 종료.
4. **검출기 "Cold Cup"** — FPA 자체보다 약간 더 차갑게 유지되어 **아웃가싱 오염 물질을 센서 이전에 포집**. 카메라당 전용 annealing 가열기(84.3 Ω, 28 V에서 9.3 W)가 주기적 베이크아웃으로 누적 응축물 제거.

### Part VI: Concepts of Operations (§9) / 운용 개념

**English**
EUI operations reflect Solar Orbiter's deep-space nature: telemetry is bandwidth-limited and interrupted during "sup-aligns" (solar conjunctions). Science data are collected in three **Remote Sensing Windows (RSWs)** per orbit (perihelion + max latitudes N/S); outside RSWs, only synoptic FSI is active.

Six observation programs (configurable parameters are initial guesses, refined in-flight):

| Program | Telescope | Cadence | Compression | Purpose |
|---|---|---|---|---|
| **Synoptic** | FSI 17.4 + 30.4 nm, bin 2×2 | 15 min | 0.32 bpp high | Continuous background context |
| **Global Eruptive Event** | FSI 17.4 or 30.4 nm, no binning | 10 s | 1.20 bpp low | Dimmings, EIT waves |
| **Faint High Corona** | FSI, 60 × 1-min stacked → 1-hour | 1 hr | 2.40 bpp very low | Extended corona with occulter |
| **Coronal Hole** | HRI_EUV + HRI_Lya | 30 s | ~very low | Deep coronal-hole regions |
| **Discovery** | HRI_EUV full + HRI_Lya partial | 1 s (or sub-s subfield) | 0.80 bpp medium | Highest-cadence, highest-res dynamics |
| **Find Event** | FSI 17.4 or 30.4 nm, bin 4×4 | 1 min | 0.12 bpp very high | Thumbnail-driven flare/eruption trigger |
| **Beacon** | FSI 17.4 + 30.4, bin 4×4 | 15 min | 0.12 bpp very high | Low-latency context for ground planning |

An on-board **event-detection algorithm (SOFAST, Bonte et al. 2013)** runs on FSI thumbnails every minute: brightness-difference thresholding over successive thumbnails triggers preservation and re-prioritization of buffered HRI data (One-Hour Queue). Inter-Instrument Communication (IIC) messages propagate events between Solar Orbiter instruments. EUI data centre is at ROB (Brussels); four data levels L0 (raw) → L1 (engineering) → L2 (calibrated, default) → L3 (visualization JPEG2000) are distributed via SOAR and JHelioviewer.

**한국어**
EUI 운용은 Solar Orbiter의 심우주 특성을 반영 — 다운링크가 제한적이며 태양 합(conjunction) 동안 단절된다. 과학 데이터는 궤도당 3개 **원격 감지 창(RSW)** (근일점 + 최대 위도 N/S)에서 집중 수집되고, RSW 밖에서는 FSI 시놉틱만 작동한다.

6종 관측 프로그램(매개변수는 초기값, 임무 중 최적화):

| 프로그램 | 망원경 | 케이던스 | 압축 | 목적 |
|---|---|---|---|---|
| **Synoptic** | FSI 17.4 + 30.4, 2×2 bin | 15분 | 0.32 bpp 고품질 | 연속 배경 맥락 |
| **Global Eruptive Event** | FSI 단일 밴드, bin 없음 | 10초 | 1.20 bpp 저압축 | Dimming·EIT파 |
| **Faint High Corona** | FSI, 60×1분 누적 → 1시간 | 1시간 | 2.40 bpp 극저압축 | Occulter로 확장 코로나 |
| **Coronal Hole** | HRI_EUV + HRI_Lya | 30초 | 매우 낮음 | 어두운 코로나 홀 영역 |
| **Discovery** | HRI_EUV 전체 + HRI_Lya 부분 | 1초(서브필드는 sub-s) | 0.80 bpp 중간 | 최고 케이던스·고해상 역학 |
| **Find Event** | FSI 17.4 또는 30.4, 4×4 bin | 1분 | 0.12 bpp 매우 높음 | 썸네일 기반 플레어/분출 트리거 |
| **Beacon** | FSI 17.4 + 30.4, 4×4 bin | 15분 | 0.12 bpp 매우 높음 | 지상 계획용 저지연 맥락 |

온보드 **이벤트 탐지 알고리즘(SOFAST, Bonte et al. 2013)** 이 FSI 썸네일에서 매분 작동 — 연속 썸네일의 밝기 차분 역치 초과 시 One-Hour Queue의 HRI 버퍼 데이터를 보존·우선순위 재조정. Inter-Instrument Communication(IIC) 메시지로 Solar Orbiter 기기들 간 이벤트가 전파된다. EUI 데이터센터는 ROB(브뤼셀) 소재; 4단계 데이터 — L0(raw) → L1(engineering) → L2(calibrated, 기본) → L3(visualization JPEG2000) — 를 SOAR와 JHelioviewer를 통해 배포.

---

## 3. Key Takeaways / 핵심 시사점

1. **Constraint-driven design is the instrument's hallmark.** — **제약 주도 설계가 핵심.** Every EUI choice (single-mirror FSI, front-pupil hex geometry, APS detectors, WICOM compression, event-driven operations) traces back to Solar Orbiter-specific constraints: 13× solar flux, mass/power budgets, telemetry bandwidth, and radiation environment. This is an exemplary case study in how science-to-engineering translation works for deep-space instruments. / 모든 EUI 설계 선택은 Solar Orbiter 고유의 제약(13× 태양 플럭스, 질량·전력 예산, 다운링크, 방사선 환경)으로 거슬러 올라간다. 심우주 기기에서 과학→공학 변환이 어떻게 작동하는지 보여주는 모범 사례.

2. **APS/CMOS detectors replace CCDs for reason, not fashion.** — **APS/CMOS 검출기 채용은 유행이 아닌 필연.** Their random-access windowing supports subfield sub-second cadence; per-pixel amplification enables high-gain low-noise + low-gain high-full-well in the same readout (5 e⁻ rms + 120 ke⁻ saturation); no charge transfer makes them radiation-hard (essential in Solar Orbiter's proton environment). This decision sets a precedent for future near-Sun missions. / 랜덤 액세스 윈도잉이 서브필드 sub-second 케이던스를 지원하고, 픽셀당 증폭이 동일 판독에서 저잡음·고포화(5 e⁻ rms + 120 ke⁻)를 가능케 하며, charge transfer가 없어 방사선 내성이 우수하다. 근태양 미션의 선례.

3. **Telemetry is the real bottleneck — data strategy dominates.** — **다운링크가 진짜 병목 — 데이터 전략이 지배.** Solar Orbiter's few-kbps to Mbps downlink means EUI's raw data-production capacity far exceeds what can be sent home. The solution: WICOM ASIC wavelet compression achieving <0.1 bpp, event-driven priority rewrites via SOFAST on FSI thumbnails, the One-Hour Queue buffer, and a formal hierarchy of six observation programs trading off cadence/FOV/compression. / Solar Orbiter 다운링크는 수 kbps–Mbps 수준이어서 EUI 원시 데이터 생산 능력이 압도적으로 초과. 해법: WICOM ASIC 웨이블릿 압축 <0.1 bpp, FSI 썸네일 기반 SOFAST 이벤트 트리거, One-Hour Queue 버퍼, 케이던스/시야/압축을 교환하는 6 프로그램 계층.

4. **The hexagonal pupil is a beautiful optical trick.** — **육각 입사동은 아름다운 광학 트릭.** Matching the hexagonal entrance pupil geometry to the hexagonal mesh grid (0.4 mm pitch) of the Al entrance filter, and placing the pupil in front of the filter at 737 mm, geometrically halves filter irradiance *and* nulls mesh-diffraction modulation when the pupil footprint on the filter equals an integer number of grid periods. A single design choice solves two problems at once (Auchère et al. 2005, 2011). / Al 입사 필터의 육각 메시(0.4 mm 피치)에 **육각 입사동**을 형상 정합시키고 필터 앞 737 mm에 배치 → (1) 필터 조도 기하적 절반 감소 + (2) 입사동 풋프린트가 격자 주기의 정수배일 때 메시 회절 변조 상쇄. 단일 설계로 두 문제 동시 해결.

5. **Division of labor within the payload is essential.** — **탑재체 내부의 역할 분담이 필수.** EUI's restriction to 17.4, 30.4, and 121.6 nm is not a capability gap — it reflects Solar Orbiter's design where SPICE (EUV spectrometer, multi-T diagnostics), Metis (coronagraph, extended corona), PHI (high-res magnetograph), STIX (X-ray), and the in-situ suite complement EUI. Understanding any single instrument requires understanding its position in the payload. / EUI가 17.4/30.4/121.6 nm만 다루는 것은 능력 부족이 아닌 역할 분담 — SPICE(EUV 분광기·다온도), Metis(코로나그래프·확장 코로나), PHI(자기장), STIX(X-ray), 인시투 기기군이 EUI를 보완. 단일 기기 이해에는 탑재체 내 위치 이해가 필수.

6. **Dual-gain readout doubles the dynamic range.** — **듀얼 게인 판독이 동적 범위를 두 배로.** By wiring two amplifiers per pixel with a gain ratio ~22.3, EUI captures faint off-limb features (high gain, read-noise limited at 5 e⁻) and bright flare kernels (low gain, 120 ke⁻ saturation) in the same exposure. Software selects per pixel based on saturation. This sidesteps the classical trade-off between read noise and full well that has long constrained EUV imaging. / 픽셀당 증폭기 2개(게인 비 ~22.3)로 배선하여 희미한 off-limb(고게인, 5 e⁻ 판독잡음 한계)와 밝은 플레어 핵(저게인, 120 ke⁻ 포화)을 같은 노출에서 동시 포착, 포화 여부에 따라 픽셀별 소프트웨어 선택. EUV 영상을 오래 제약해온 판독잡음↔풀웰 교환을 회피.

7. **The "transition corona" gap is uniquely addressable by FSI.** — **"전이 코로나" 공백은 FSI만의 기회.** Between EUV imagers (≤1.3 R☉) and coronagraphs (≥2 R☉) lies the magnetically-to-flow-dominated transition region — critical for mapping coronal origin to in-situ solar-wind measurements. FSI's 14.3 R☉ FOV at 1 AU plus its occulting-disc-enabled 10 R☉ imaging fills this gap, complementing Solar Orbiter's Metis coronagraph. / EUV 영상기(≤1.3 R☉)와 코로나그래프(≥2 R☉) 사이 "전이 코로나"는 자기 지배에서 흐름 지배로 넘어가는 영역 — 코로나 기원과 인시투 태양풍 측정 간 연결에 핵심. FSI의 1 AU에서 14.3 R☉ 시야와 occulting disc를 활용한 10 R☉ 촬영이 이 공백을 메운다.

8. **Event detection + prioritization is how modern space observatories trade raw volume for scientific value.** — **이벤트 탐지 + 우선순위가 현대 우주 관측소의 raw 용량을 과학 가치로 교환하는 방법.** SOFAST on FSI thumbnails, IIC messages across instruments, the One-Hour Queue buffer, and post-hoc ground prioritization together form a pipeline where transient events (flares, CMEs, campfires) rescue high-cadence HRI data from overwrites. This architecture will generalize to future missions like MUSE, Solar-C/EUVST. / FSI 썸네일 기반 SOFAST + 기기 간 IIC 메시지 + One-Hour Queue 버퍼 + 지상 후처리 우선순위가 결합되어, 순간적 이벤트(플레어·CME·campfire)가 덮어쓰기에서 고케이던스 HRI 데이터를 구출하는 파이프라인. 차세대 MUSE·Solar-C/EUVST로 일반화될 아키텍처.

---

## 4. Mathematical Summary / 수학적 요약

### (1) Plate scale and pixel footprint / 플레이트 스케일과 픽셀 풋프린트

$$
\text{Plate scale} = \frac{206{,}265''}{f[\text{mm}]} \cdot p[\text{mm}] \qquad \text{Pixel footprint} = d_\odot \cdot \theta_{\text{pix}}
$$

**English** For HRI_EUV: $f = 4187$ mm, $p = 0.01$ mm → plate scale = 0.493″/pixel. At $d_\odot = 0.28$ AU = 4.19 × 10⁷ km, pixel footprint = $0.493''/(206{,}265) \times 4.19 \times 10^7$ km = **100 km**. For FSI: $f = 462.5$ mm → 4.46″/pixel.

**한국어** HRI_EUV: 초점거리 4187 mm, 픽셀 10 μm → 0.493″/픽셀. 0.28 AU에서 **픽셀 풋프린트 100 km**. FSI: 초점거리 462.5 mm → 4.46″/픽셀.

### (2) Dual-band multilayer Bragg condition / 듀얼 밴드 다층막 Bragg 조건

$$
m\lambda = 2d \cos\theta \cdot \sqrt{1 - \frac{2\bar\delta}{\cos^2\theta}}
$$

**English** For FSI's superposition coating: a first Al/Mo/SiC periodic stack (period chosen) for 17.4 nm first-order + an Al buffer + a second Al/Mo/SiC stack (4 periods of 16.5 nm) for 30.4 nm first-order. HRI_EUV coating: 30 periods of 8.86 nm, tuned to 17.4 nm with 3 nm SiC capping. Small $\bar\delta$ term accounts for index-of-refraction deviation from unity at EUV wavelengths.

**한국어** FSI의 중첩 코팅: 17.4 nm 1차용 Al/Mo/SiC 주기층 + Al 버퍼 + 30.4 nm 1차용 Al/Mo/SiC 주기층(4주기 × 16.5 nm). HRI_EUV 코팅: 8.86 nm × 30주기, 17.4 nm에 조정, SiC 3 nm 캡핑. $\bar\delta$ 항은 EUV에서 굴절률이 1에서 벗어나는 효과 보정.

### (3) Radiometric signal chain / 복사계측 신호 사슬

$$
N_{\text{DN}} = B_\lambda(T) \cdot \Omega_{\text{pix}} \cdot A_{\text{pupil}} \cdot R_{\text{ml}}^{n_{\text{mirror}}}(\lambda) \cdot \tau_{\text{entrance}}(\lambda) \cdot \tau_{\text{focal}}(\lambda) \cdot \eta_{\text{QE}}(\lambda) \cdot \frac{\Delta t}{g}
$$

**English** $B_\lambda$ = solar spectral radiance (depends on feature temperature via line emissivity), $\Omega_{\text{pix}} = \theta_{\text{pix}}^2$, $A_{\text{pupil}}$ = effective entrance area, $R_{\text{ml}}^{n_{\text{mirror}}}$ = multilayer reflectivity raised to number of bounces (1 for FSI, 2 for HRI), $\tau_{\text{entrance, focal}}$ = thin-film filter transmissions, $\eta_{\text{QE}}$ = detector QE, $\Delta t$ = integration time, $g$ = electrons-per-DN. Setting required S/N ≥ 10 with minimum 2 photons per pixel detectable fixes the design constraints.

**한국어** $B_\lambda$ = 태양 스펙트럼 라디언스(구조의 온도에 선 방출율 통해 의존), $\Omega_{\text{pix}}$ = 픽셀 고체각, $A_{\text{pupil}}$ = 유효 입사 면적, $R_{\text{ml}}^{n_{\text{mirror}}}$ = 다층막 반사율(FSI는 1회, HRI는 2회 반사), $\tau$ = 박막 필터 투과율, $\eta_{\text{QE}}$ = 검출기 양자효율, $\Delta t$ = 적분 시간, $g$ = DN당 전자수. 픽셀당 2광자 최소 검출 + S/N ≥ 10 요구가 설계 제약을 결정.

### (4) Solar flux at perihelion / 근일점 태양 플럭스

$$
F(r) = F_\oplus \left(\frac{1\,\text{AU}}{r}\right)^2 = 1361 \times \frac{1}{0.28^2} \approx 17{,}400 \text{ W/m}^2
$$

**English** "13 solar constants" at perihelion dictates the heat-shield feedthrough, heat-rejection entrance filter, and decoupling of the OBS front from the optical bench — the 17.5 kW/m² figure quoted in the paper.

**한국어** 근일점에서 "13 태양 상수"는 열차폐 피드스루, 열거절 입사 필터, OBS 전면과 광학 벤치의 열적 분리 설계를 지배 — 논문에 명시된 17.5 kW/m² 수치.

### (5) Dual-gain readout equation / 듀얼 게인 판독식

$$
S_{\text{HG}} = \min(g_{\text{HG}} \cdot N_e, S_{\text{ADC,max}}), \qquad S_{\text{LG}} = g_{\text{LG}} \cdot N_e
$$

$$
\frac{g_{\text{HG}}}{g_{\text{LG}}} \approx 22.3, \qquad \text{FullWell}_{\text{shared}} = 120\,\text{ke}^-
$$

**English** Per pixel, both outputs are read. Software uses HG when unsaturated (lowest noise floor 5 e⁻ rms), falls back to LG when HG is ADC-clipped. This effectively doubles the bit depth without increasing readout time — a key enabler for EUV imaging of both quiet Sun and flare kernels.

**한국어** 픽셀당 두 출력이 동시 판독. 소프트웨어는 HG가 포화되지 않을 때 HG(최저 5 e⁻ rms 잡음) 사용, HG가 ADC 클립되면 LG로 전환. 판독 시간 증가 없이 비트 심도를 사실상 두 배로 — 조용한 태양과 플레어 핵을 동시에 EUV 영상화하는 핵심 이점.

### (6) Worked example: HRI_EUV photon budget at 0.28 AU / 수치 예제: 0.28 AU에서 HRI_EUV 광자 수지

**English** Quiet-Sun 17.4 nm specific intensity $I \approx 10^{13}$ photons cm⁻² s⁻¹ sr⁻¹ (typical EIT/AIA-derived value). HRI_EUV pixel angular size $\theta_{\text{pix}} = 0.493''$ → $\Omega_{\text{pix}} = (0.493/206{,}265)^2 \approx 5.7 \times 10^{-12}$ sr. Entrance pupil area $A = \pi(2.37)^2 / 4 \approx 17.6$ cm². Bounce-wise efficiency: two multilayer reflections at ~40% each (0.16), entrance Al filter ~40%, focal filter ~40%, QE 80%. Net throughput $\approx 0.16 \times 0.40 \times 0.40 \times 0.80 \approx 0.021$.

$$
\text{Rate} \approx 10^{13} \times 5.7 \times 10^{-12} \times 17.6 \times 0.021 \approx 21 \text{ e}^-/\text{pixel/s}
$$

At 1 s integration, signal = 21 e⁻ ≫ read noise 5 e⁻ → **S/N ≈ √21 ≈ 4.6** (shot-limited). For active regions the signal is ~10× higher, easily exceeding the S/N = 10 requirement. At 0.28 AU, the solid-angle-per-pixel geometry on the Sun drops as (1 AU / r)² which, multiplied with the natural brightness, gives the **same intensity per pixel** (intensity is distance-independent for resolved sources) but **12.8× higher flux per pixel** — so signal rates scale up dramatically, enabling sub-second cadence for dynamic subfield observing.

**한국어** 조용한 태양 17.4 nm 비강도 $I \approx 10^{13}$ photons cm⁻² s⁻¹ sr⁻¹ (EIT/AIA 기준 전형값). HRI_EUV 픽셀 각크기 0.493″ → $\Omega_{\text{pix}} \approx 5.7 \times 10^{-12}$ sr. 입사동 면적 $A = \pi(2.37)^2/4 \approx 17.6$ cm². 투과 효율 연쇄: 다층막 반사 2회 (~40%씩 → 0.16) × 입사 Al 필터 0.40 × 초점 필터 0.40 × QE 0.80 ≈ **총 0.021**.

$$
\text{광자율} \approx 10^{13} \times 5.7\times 10^{-12} \times 17.6 \times 0.021 \approx 21\text{ e}^-/\text{픽셀/초}
$$

1초 적분 시 신호 21 e⁻ ≫ 판독잡음 5 e⁻ → **S/N ≈ √21 ≈ 4.6** (샷 한계). 활동 영역은 ~10× 더 높아 S/N = 10 요건 쉽게 충족. 0.28 AU에서 공간 분해된 구조의 단위 면적 밝기는 거리 무관이지만, **픽셀당 플럭스는 12.8× 증가** — 신호율이 크게 증가하여 동적 서브필드의 sub-second 케이던스가 가능해진다.

### (7) Onboard compression pipeline / 온보드 압축 파이프라인

$$
\text{Raw 14 bpp} \xrightarrow{\text{gain/offset}} \text{calibrated} \xrightarrow{\text{bad-pix + cosmic}} \text{cleaned} \xrightarrow{\text{bin } n\times n} \text{reduced} \xrightarrow{\text{recode 8 bit}} \text{recoded} \xrightarrow{\text{WICOM wavelet}} < 0.1\text{ bpp}
$$

**English** The stages must complete within the 0.5 s time to receive the next image (running on a hardware pipeline, not CPU). The pixel-processing maps (per-pixel gain, offset, bad-pixel replacement, cosmic-ray threshold) are pre-loaded into compression SRAM from ground-verified calibration tables.

**한국어** 각 단계는 다음 이미지 수신까지의 0.5초 이내에 완료되어야 함 — CPU가 아닌 하드웨어 파이프라인에서 실행. 픽셀 처리 맵(픽셀별 게인/오프셋, 불량 픽셀 치환, 우주선 역치)은 지상 검증된 교정 테이블에서 압축 SRAM에 사전 로드.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1973 ──── Skylab ATM [first extended EUV imaging]
            │
1989 ──── CORONAS-F (Russia) — predecessor single-mirror EUV designs
            │
1995 ──── SOHO launch ──── EIT (this series #9) ──── LASCO (this series #10)
            │                │                        │
1998 ──── TRACE (this series #11) ─ 1″ single-band EUV on Sun-synch LEO
            │
2005 ──── Auchère et al. 2005 ─ conceptual FSI single-mirror design with hex pupil
            │
2006 ──── Hinode (this series #14–15) ─ SOT + EIS + XRT
            │
2006 ──── STEREO EUVI (this series #17) ─ twin ecliptic spacecraft, 4-band EUV
            │
2010 ──── SDO launch ──── AIA (this series #12) + HMI (#13) + EVE
            │           7 EUV channels, 0.6″, 12 s cadence, GEO orbit
2012 ──── PROBA2/SWAP (Seaton et al.) ─ SWAP single-band 17.4 nm;
            │                           first EUV APS detector flown
2017 ──── EUI delivered to Airbus UK for spacecraft integration
2018 ──── DKIST first light (this series #6)
            │
2020 Feb ──── Solar Orbiter launch (Atlas V 411, Cape Canaveral)
            │
2020 ──── ★★ Rochus et al. (THIS PAPER) — EUI instrument description in A&A 642, A8
            │                             ─ alongside Solar Orbiter special issue
2020 ──── Anderson et al. (this series #18, SPICE spectrograph A&A 642, A14)
2020 ──── Müller et al. (to be added #36, mission overview A&A 642, A1)
2020 Jun ──── First Light; HRI_EUV reveals ubiquitous "campfires"
            │
2021 ──── Berghmans et al. ─ "campfires" paper published (Nature Astron.)
            │                 re-ignites nanoplare coronal heating debate
2022–── Perihelion passages ≤0.3 AU; full EUI science operations
2025–── High-latitude phase begins (inclination >30°);
            │  first high-resolution EUV views of solar poles
Future ──── MUSE (2027, multi-slit UV/EUV spectrograph), Solar-C/EUVST (2028+),
            inheriting EUI design heritage (APS, onboard event detection)
```

**English**
EUI sits at the confluence of three decades of space EUV imaging evolution: from SOHO/EIT's first routine EUV to AIA's multi-channel thermal diagnostics, now to Solar Orbiter's close-perihelion high-resolution imaging. The paper predates "campfires" (discovered 4 months after publication acceptance) but describes all the enabling technology. EUI is the design template for the next generation of near-Sun UV/EUV instruments.

**한국어**
EUI는 30년 우주 EUV 영상 진화의 세 흐름이 합류하는 지점 — SOHO/EIT의 최초 정규 EUV → AIA의 다채널 열진단 → Solar Orbiter의 근태양 고해상 영상. 본 논문은 "campfires" 발견(논문 게재 수락 4개월 후)보다 앞서지만 이를 가능케 한 모든 기술을 기술. 차세대 근태양 UV/EUV 기기의 설계 원형.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **#8 Domingo et al. 1995 (SOHO overview)** | Foundational mission architecture: SOHO at L1 carried the first routine EUV payload. EUI inherits the mission-level concept (remote + in-situ payload) but reshapes it for close-perihelion/out-of-ecliptic. / SOHO는 EUI의 미션 수준 개념(원격+인시투) 모델. EUI는 이를 근태양·황도면 이탈로 재구성. | **High** — direct ancestor mission template. / 직접 조상 미션 템플릿. |
| **#9 Delaboudinière et al. 1995 (EIT)** | First routine 4-band EUV imager on SOHO. EUI's 17.4 and 30.4 nm bands are direct EIT descendants; multilayer coating lineage (Al-based) traces to EIT. FSI is functionally "EIT at 0.28 AU". / SOHO의 최초 4-밴드 EUV 영상기. EUI의 17.4·30.4 nm는 EIT의 직계 후손. | **Very high** — technical lineage. / 기술 계보. |
| **#11 Handy et al. 1999 (TRACE)** | TRACE demonstrated 1″ EUV imaging with single-passband optimization. EUI's HRI_EUV concept (single-band, high resolution over small FOV) is TRACE's philosophy pushed to 100 km footprints. / TRACE의 단일 밴드·고해상도 철학을 100 km 풋프린트까지 극한으로 밀고 간 것이 HRI_EUV. | **High** — philosophy inheritance. / 철학 계승. |
| **#12 Lemen et al. 2012 (SDO/AIA)** | Contrast: AIA covers 7 EUV channels for thermal diagnostics; EUI covers only 2 EUV + Lyman-α because SPICE provides spectroscopic multi-T info. EUI and AIA represent orthogonal points in the imaging vs. spectroscopy trade space. / AIA(7채널, 열진단)와 EUI(2 EUV + Lyα, SPICE 분업)는 영상·분광 교환 공간에서 직교점. | **Very high** — contrast / role clarification. / 대조 및 역할 명료화. |
| **#17 Wülser et al. 2004 (STEREO/EUVI)** | STEREO demonstrated twin-spacecraft EUV imaging for 3D reconstruction. EUI's off-ecliptic vantage complements this — now stereo + polar views become possible. / 쌍둥이 우주선 EUV → 3D 재구성. EUI의 황도면 이탈 시야가 stereo + 극 관측을 완성. | **High** — geometric complement. / 기하학적 보완. |
| **#18 Anderson et al. 2020 (Solar Orbiter SPICE)** | Same spacecraft payload. SPICE = multi-T EUV spectroscopy; EUI = broadband EUV imaging. They trade tasks: EUI provides context + small-scale imaging, SPICE provides thermal diagnostics. Joint observation programs (HRI + SPICE slit-scan) are core mission deliverables. / 같은 우주선 탑재, 역할 분담 — EUI는 맥락+미세영상, SPICE는 열진단. | **Critical** — same payload partner. / 같은 탑재체 파트너. |
| **#35 Pesnell et al. 2012 (SDO overview)** | SDO overview gives contrast reference for Solar Orbiter's constraints. SDO's GEO orbit allows continuous 1.5 Gbps downlink; Solar Orbiter's deep-space vantage forces the reverse trade. Compare to understand why EUI looks so different from AIA+EVE despite covering similar wavelengths. / SDO의 GEO·1.5 Gbps 다운링크 vs. Solar Orbiter의 심우주 제약. EUI가 AIA와 왜 그토록 다른지 이해하려면 비교 필수. | **Very high** — design trade contrast. / 설계 교환 대조. |
| **#36 Müller et al. 2020 (Solar Orbiter overview, to be added)** | Mission-level anchor — defines the four science questions EUI addresses and the orbit profile that shapes EUI. Essential parent context. / 미션 수준 앵커 — EUI가 답하는 4개 과학 질문과 궤도 프로파일 정의. 필수 상위 맥락. | **Critical** — mission parent paper. / 미션 부모 논문. |

---

## 7. References / 참고문헌

### Primary paper / 주 논문
- Rochus, P., Auchère, F., Berghmans, D., et al. "The Solar Orbiter EUI Instrument: The Extreme Ultraviolet Imager." *A&A* 642, A8 (2020). DOI: [10.1051/0004-6361/201936663](https://doi.org/10.1051/0004-6361/201936663)

### Key cited references from the paper / 논문 내 주요 참고
- **FSI concept**: Auchère, F., Song, X., Rouesnel, F., et al. "Innovative designs for the imaging of the solar corona in the extreme ultraviolet." *Proc. SPIE* 5901, 298 (2005).
- **Hex pupil / mesh grid**: Auchère, F., Rizzi, J., Philippon, A., & Rochus, P. "Automated search for stellar occultations with the EIT, TRACE, and SECCHI EUV telescopes." *J. Opt. Soc. Am. A* 28, 40 (2011).
- **SOHO mission overview**: Domingo, V., Fleck, B., & Poland, A. I. *Sol. Phys.* 162, 1 (1995). [Read paper #8]
- **EIT instrument**: Delaboudinière, J.-P., Artzner, G. E., Brunaud, J., et al. *Sol. Phys.* 162, 291 (1995). [Read paper #9]
- **TRACE instrument**: Handy, B. N., Acton, L. W., Kankelborg, C. C., et al. *Sol. Phys.* 187, 229 (1999). [Read paper #11]
- **SDO/AIA**: Lemen, J. R., Title, A. M., Akin, D. J., et al. *Sol. Phys.* 275, 17 (2012). [Read paper #12]
- **SWAP EUV APS heritage**: Seaton, D. B., Berghmans, D., Nicula, B., et al. *Sol. Phys.* 286, 43 (2013a).
- **SWAP first EUV APS results**: Seaton, D. B., De Groof, A., Shearer, P., Berghmans, D., & Nicula, B. *ApJ* 777, 72 (2013b).
- **Al multilayer coatings**: Meltchakov, E., Hecquet, C., Roulliay, M., et al. *Appl. Phys. A: Mater. Sci. Process.* 98, 111 (2010); subsequent SPIE papers 2011, 2013.
- **WICOM compression lineage**: Nicula, B., Berghmans, D., & Hochedez, J.-F. *Sol. Phys.* 228, 253 (2005).
- **Onboard event detection (SOFAST)**: Bonte, K., Berghmans, D., De Groof, A., Steed, K., & Poedts, S. *Sol. Phys.* 286, 185 (2013).
- **Solar Orbiter mission overview**: Müller, D., St. Cyr, O. C., Zouganelis, I., et al. *A&A* 642, A1 (2020). [To be added as paper #36]
- **SPICE instrument (sister paper)**: SPICE Consortium (Anderson, M., et al.) *A&A* 642, A14 (2020). [Read paper #18]
- **Metis coronagraph**: Antonucci, E., Romoli, M., Andretta, V., et al. *A&A* 642, A10 (2020).
- **"Campfires" discovery (post-this-paper)**: Berghmans, D., Auchère, F., Long, D. M., et al. "Extreme-UV quiet Sun brightenings observed by the Solar Orbiter/EUI." *A&A* 656, L4 (2021).

### Supporting background / 보조 배경
- Parker, E. N. "Nanoflares and the solar X-ray corona." *ApJ* 330, 474 (1988). [Nanoflare hypothesis]
- Robbrecht, E., Verwichte, E., Berghmans, D., et al. "Slow magnetoacoustic waves in coronal loops." *A&A* 370, 591 (2001). [Loop wave detection with EIT]
- Cirtain, J. W., Golub, L., Winebarger, A. R., et al. "Energy release in the solar corona from spatially resolved magnetic braids." *Nature* 493, 501 (2013). [Hi-C braiding observations]
