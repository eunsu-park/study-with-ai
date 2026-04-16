---
title: "The EUV Imaging Spectrometer for Hinode"
authors: J.L. Culhane, L.K. Harra, A.M. James et al.
year: 2007
journal: "Solar Physics, Vol. 243, pp. 19–61"
doi: "10.1007/s01007-007-0293-1"
topic: Solar_Observation
tags: [EUV, spectroscopy, Hinode, EIS, coronal diagnostics, space instrumentation]
status: completed
date_started: 2026-04-16
date_completed: 2026-04-16
---

# 15. The EUV Imaging Spectrometer for Hinode / Hinode용 EUV 영상 분광기

---

## 1. Core Contribution / 핵심 기여

이 논문은 Hinode 위성에 탑재된 EUV Imaging Spectrometer (EIS)의 설계, 교정, 성능을 포괄적으로 기술합니다. EIS는 170–210 Å (단파장 대역, SW)과 250–290 Å (장파장 대역, LW) 두 개의 EUV 파장 대역에서 태양 코로나와 상부 전이영역의 방출선을 관측하는 분광기입니다. 이전 세대 장비인 SOHO/CDS 대비 유효 면적이 약 10배 증가하고, 스펙트럼 분해능이 10배 향상되었으며, 공간 분해능(2″)도 2–3배 개선되었습니다. 이러한 성능 향상은 multilayer coating을 적용한 normal incidence 광학계와 back-illuminated thinned CCD 검출기 채택으로 달성되었습니다. EIS는 플라즈마 속도(±5 km s⁻¹), 선폭(비열적 속도 ±25 km s⁻¹), 온도, 밀도, 원소 조성을 측정하여 코로나 가열, 플레어 역학, 태양풍 기원 등 핵심 과학 문제에 답하도록 설계되었습니다.

This paper provides a comprehensive description of the design, calibration, and performance of the EUV Imaging Spectrometer (EIS) aboard the Hinode satellite. EIS is a spectrometer that observes emission lines from the solar corona and upper transition region in two EUV wavelength bands: 170–210 Å (short wavelength, SW) and 250–290 Å (long wavelength, LW). Compared to its predecessor SOHO/CDS, EIS achieves approximately a factor of ten increase in effective area, a factor of ten improvement in spectral resolution, and a factor of two to three improvement in spatial resolution (2″). These performance gains were achieved through the adoption of a normal incidence optical system with multilayer coatings and back-illuminated thinned CCD detectors. EIS was designed to measure plasma velocities (±5 km s⁻¹), line widths (nonthermal velocities ±25 km s⁻¹), temperatures, densities, and elemental compositions to address key science questions including coronal heating, flare dynamics, and solar wind origins.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction and Scientific Aims (Sections 1–2) / 서론 및 과학적 목표

Hinode 미션의 세 가지 핵심 과학 목표는: (1) 활동영역과 quiet Sun에서의 코로나 가열 메커니즘 규명, (2) 플레어·CME 등 과도현상(transient phenomena)의 메커니즘 확립, (3) 광구에서 코로나로의 에너지 전달 과정 탐구입니다.

The three main scientific goals of the Hinode mission are: (1) determine the mechanisms responsible for heating the corona in active regions and the quiet Sun, (2) establish the mechanisms that cause transient phenomena (flares, CMEs), and (3) investigate processes for energy transfer from photosphere to corona.

EIS는 이를 위해 방출선 강도, Doppler 속도, 선폭, 온도, 밀도를 측정합니다. EIS가 다루는 구체적 과학 주제들:

EIS measures line intensities, Doppler velocities, line widths, temperatures, and densities to address these goals. Specific science topics include:

- **코로나/광구 속도장 비교 / Coronal/photospheric velocity field comparison**: SOT와의 공동 관측으로 활동영역 자기장과 코로나 플라즈마의 연결성 연구
- **코로나 루프 가열 / Coronal AR heating**: 루프 내 재결합(reconnection) 증거 탐색, 온도·밀도·속도 변화의 시공간 분석
- **코로나 지진학 / Coronal seismology**: 5분 진동부터 대규모 충격파까지 파동의 검출 및 모드 식별
- **CME 전조 및 발생 / CME onsets and signatures**: 자기 twisted flux rope 제거 과정에서 속도 측정으로 twist 정도 평가
- **플레어 플라즈마 생성 / Flare produced plasma**: Fe XXIV 등 고온 이온의 영상 분광으로 플라즈마 생성 과정 규명
- **플레어 재결합 유입/유출 / Flare reconnection inflow and outflow**: 기존 영상 관측의 한계를 넘어 분광적 속도 증거 확보
- **Quiet Sun 과도현상 / Quiet Sun transient events**: 네트워크 경계, 코로나 홀 경계에서의 재결합과 bi-directional jet 관측

### Part II: Instrument Overview (Section 3) / 장비 개요

이전 EUV 분광기들(SOHO CDS, 50–500 Å)은 grazing incidence 광학계를 사용했으며, microchannel plate 검출기(QE ≤ 20%)를 채택했습니다. EIS의 핵심 설계 혁신:

Previous EUV spectrometers (SOHO CDS, 50–500 Å) used grazing incidence optics and microchannel plate detectors (QE ≤ 20%). Key design innovations of EIS:

1. **Normal incidence optics**: Mo/Si multilayer coating으로 특정 EUV 파장대에서 높은 반사율 달성 (SW: 32%, LW: 23%)
2. **Back-illuminated thinned CCDs**: QE가 microchannel plate의 2–3배 (SW: 44±4%, LW: 39±4%)
3. **D-shaped 이중 대역 설계 / Dual-band D-shaped design**: 거울과 격자를 D자 형태로 분할하여 두 파장 대역을 동시 최적화

이로 인해 CDS 대비 유효 면적 ~10배 증가, 스펙트럼 분해능 ~10배 향상, 공간 분해능 2–3배 개선을 달성했습니다.

This yields ~10× increase in effective area, ~10× improvement in spectral resolution, and 2–3× improvement in spatial resolution compared to CDS.

**EIS 핵심 성능 파라미터 (Table 1) / Key EIS Performance Parameters:**

| 파라미터 / Parameter | 값 / Value |
|---|---|
| 파장 대역 / Wavelength bands | 170–210 Å, 250–290 Å |
| 최대 유효 면적 / Peak effective areas | 0.30 cm² (SW), 0.11 cm² (LW) |
| 주경 / Primary mirror | 15 cm 직경, Mo/Si multilayer coating |
| 격자 / Grating | Toroidal/laminar, 4200 lines/mm, Mo/Si multilayer |
| CCD | Back-thinned e2v CCD 42-20, 2048 × 1024 pixels |
| 판 축척 / Plate scale | 13.53 μm/arcsec (CCD), 9.40 μm/arcsec (slit) |
| 공간 분해능 / Spatial resolution | 2″ (1 pixel) |
| 시야 / Field of view | 6′ × 8.5′, offset center ±15′ E-W |
| 래스터 / Raster | 1″ in 0.7 s (최소 스텝 0.123″) |
| 슬릿/슬롯 / Slit/slot widths | 1, 2, 40, 266″ |
| 스펙트럼 분해능 / Spectral resolution | 47 mÅ (FWHM) at 185 Å; 1 pixel = 22 mÅ ≈ 25 km s⁻¹ pixel⁻¹ |
| 온도 범위 / Temperature coverage | log T = 4.7, 5.6, 5.8, 5.9, 6.0–7.3 K |
| CCD 프레임 읽기 시간 / CCD frame read time | 0.8 s |
| 동시 관측 선 수 / Simultaneous lines | 최대 25개 |

### Part III: Optical Design and Components (Section 4) / 광학 설계 및 부품

#### 4.1 입구 필터 / Entrance Filter (pp. 27–28)

1500 Å 두께의 고순도 알루미늄(VDA) 박막 필터가 170–290 Å EUV 광자는 통과시키고, 가시광·적외선·자외선은 차단합니다. 가시광 차단율은 ≤ 8.3 × 10⁻⁷입니다. 필터는 4개의 사분면으로 나뉘어 40 μm 니켈 메쉬(390 μm 중심 간격) 위에 지지되며, 메쉬의 개구율은 80%입니다. 전송률은 Brookhaven NSLS의 싱크로트론 방사를 이용하여 측정되었으며, 평균 ~40–50%입니다.

1500 Å thick high-purity vapor deposited aluminum (VDA) thin film filters pass EUV photons at 170–290 Å while blocking visible, infrared, and UV light. Visible light rejection is ≤ 8.3 × 10⁻⁷. Filters are divided into four quadrants supported on a 40 μm nickel mesh with 390 μm center spacing, achieving 80% open area. Transmittance was measured using synchrotron radiation at Brookhaven NSLS, averaging ~40–50%.

Clamshell (CLM) 어셈블리가 발사 시 필터를 보호하며, 궤도에서 HOP (High Output Paraffin) 열 구동기로 개방됩니다.

The Clamshell (CLM) assembly protects filters during launch and opens in orbit via High Output Paraffin (HOP) thermal actuators.

#### 4.2 주경 및 스캐닝 메커니즘 / Primary Mirror and Scanning (pp. 28–30)

주경은 초정밀 연마된 off-axis paraboloid (초점 거리 1939 mm, 직경 160 mm, 유효 직경 150 mm)입니다. 거울 면정밀도는 λ/47 rms (6328 Å 기준), 미세 거칠기는 < 4 Å rms입니다.

The primary mirror is a superpolished off-axis paraboloid (focal length 1939 mm, 160 mm diameter, 150 mm usable diameter). Surface figure accuracy is < λ/47 rms at 6328 Å, microroughness is < 4 Å rms.

스캐닝은 두 가지 방식으로 수행됩니다:

Scanning is performed in two ways:

- **PZT (piezoelectric transducer) 미세 스캔 / Fine scan**: N-S 축 회전으로 E-W 방향 영상 이동. 최대 300″ 틸트 → 슬릿 위치에서 600″ 영상 이동. 래스터 관측의 주된 방식.
- **리니어 스테퍼 모터 / Coarse linear motion**: 볼스크류 구동, 0.30″ 스텝 크기. 총 범위 ±825″. 관측 시야 재지향(repointing)용.

PZT 성능: 폐쇄 루프 제어로 재현성 < 2″ (30분 기준). 미러 기구 전체의 총 예상 스팟 직경은 19.12 μm (Table 2, RSS 합산).

PZT performance: reproducibility < 2″ over 30 minutes with closed-loop control. Total predicted spot diameter of the mirror mechanism is 19.12 μm (Table 2, root sum square).

#### 4.3 셔터 및 슬릿 교환 메커니즘 / Shutter and Slit Exchange (pp. 30–32)

패들 휠 메커니즘이 4개의 슬릿/슬롯을 90° 회전으로 교환합니다. 실리콘 기판 식각으로 제작된 슬릿의 실측 폭 (Table 3):

A paddle wheel mechanism exchanges four slits/slots by 90° rotations. Slits fabricated by etching silicon substrates, measured widths (Table 3):

| 공칭 / Nominal [″] | 실측 / Measured [″] |
|---|---|
| 1 | 1.01 |
| 2 | 2.02 |
| 40 | 40.9 |
| 266 | 266.6 |

셔터는 100 ms 노출에서 < 5% 측광 오차를 달성하는 브러시리스 DC 모터로 구동됩니다.

The shutter is driven by a brushless DC motor achieving < 5% photometric error for 100 ms exposures.

#### 4.4 오목 격자 / Concave Grating (pp. 32–34)

Toroidal concave grating으로, 분산 방향 곡률 반경 1182.98 mm, 수직 방향 1178.28 mm입니다. 핵심 사양:

Toroidal concave grating with radii of 1182.98 mm in the dispersion direction and 1178.28 mm in the perpendicular direction. Key specifications:

- 격자 선 밀도: 4200 lines/mm (홀로그래픽 기록 한계)
- 홈 깊이: 60 Å, land-to-groove 비율 0.85:1
- 라미나 프로파일 → 1차 회절 효율 최대화, 고차(m > 5) 효율 < 1%
- 1차 회절 효율: SW 8.0% (196 Å), LW 7.9% (271 Å)
- 100 mm 용융 실리카 기판, 유효 면적 90 mm 직경

Mo/Si multilayer를 D 형태로 분할하여 SW와 LW 각각에 최적화된 코팅을 적용했습니다. 격자 초점 메커니즘은 스테퍼 모터 + 볼스크류 조합으로 2.8 μm 스텝 구동합니다.

Mo/Si multilayers are divided into D-shaped sectors, each optimized for SW and LW bands. The grating focus mechanism uses a stepper motor + ball screw combination with 2.8 μm steps.

#### 4.5 듀얼 CCD 카메라 / Dual CCD Camera (pp. 34–37)

e2v CCD 42-20, back-illuminated and thinned. 두 개의 CCD가 각각 SW와 LW 대역을 담당합니다.

Two e2v CCD 42-20 devices (back-illuminated, thinned), each covering SW and LW bands respectively.

| 파라미터 / Parameter | 값 / Value |
|---|---|
| 배열 크기 / Array size | 2048 × 1024 pixels |
| 픽셀 크기 / Pixel size | 13.5 μm × 13.5 μm |
| 공간 축척 / Spatial scale | 1″/pixel |
| 스펙트럼 축척 / Spectral scale | 0.0223 Å/pixel |
| 읽기 속도 / Readout rate | 2 μs/pixel |
| 풀 웰 용량 / Full well capacity | ~90k electrons |
| 앰프 이득 / Amplifier gain | 6.6 ± 0.03 electrons/DN |
| 양자 효율 / QE (SW) | 44 ± 4% |
| 양자 효율 / QE (LW) | 39 ± 4% (250 Å), 37 ± 4% (290 Å) |
| 동작 온도 / Operating temperature | −40 to −45 °C |
| 암전류 / Dark current | ~0.005 electrons pixel⁻¹ s⁻¹ (at −50 °C) |

170 Å에서 광자당 ~20 광전자, 290 Å에서 ~12 광전자를 생성합니다. 최소 검출 신호는 1 광자(12 광전자)에 해당합니다.

At 170 Å, each photon generates ~20 photoelectrons; at 290 Å, ~12 photoelectrons. The minimum detectable signal corresponds to 1 photon (12 photoelectrons).

INVAR 판에 마운트하여 CTE 매칭 (CCD 실리콘: 2.6 × 10⁻⁶ °C⁻¹, INVAR: 1.3 × 10⁻⁶ °C⁻¹). 최대 25개의 소프트웨어 윈도우를 선택하여 특정 스펙트럼 영역만 전송할 수 있습니다.

Mounted on INVAR plates for CTE matching. Up to 25 software windows can be selected to transmit only specific spectral regions.

### Part IV: Mechanical and Thermal Design (Section 5) / 기계적·열적 설계

#### 5.1 기계 설계 / Mechanical Design (pp. 37–40)

구조체는 알루미늄 허니컴 코어 + CFRP (M55/RS3 시아네이트 에스테르 수지) 면재 샌드위치 패널입니다. 핵심 요구사항:

The structure uses aluminum honeycomb core + CFRP (M55/RS3 cyanate ester resin) face sheet sandwich panels. Key requirements:

- 총 질량 < 23 kg (구조체 자체는 ~23 kg으로 전체 장비 질량의 40%)
- 1차 공진 주파수 > 60 Hz (실측 59 Hz — 요구값보다 1 Hz 낮지만 허용)
- CTE < 0.4 ppm/°C (실측)
- 진동 시험: 12.5 g (측면), 19 g (종방향), Half Sine Shock 16.7 gₐpk × 10 ms
- 3년 분자 오염 fluence < 2.7 × 10⁻⁶ g cm⁻² (엄격한 진공 베이크아웃 요구)

장비 전체 크기: 3.54 m 길이, 0.55 m 폭, 0.25 m 높이.

Total instrument dimensions: 3.54 m length, 0.55 m width, 0.25 m height.

#### 5.2 열 설계 / Thermal Design (pp. 40–42)

680 km 태양 동기 극궤도에서의 열적 안정성 확보가 핵심입니다.

Maintaining thermal stability in a 680 km Sun-synchronous polar orbit is critical.

- 광학 벤치 온도 기울기 < 10 °C (운영 모드)
- CCD 냉각: 0.216 m² 전용 라디에이터 → CCD 온도 −49 to −58 °C (cold case), −45 to −49 °C (hot case)
- 내부 발열 ~14 W (MHC + ROE 유닛에서 거의 균등 분배)
- 12개 운영 히터 (최대 15 W 전력 예산)로 온도 기울기 유지
- Particle shield가 CCD보다 차갑게 유지되어 오염 방지 및 방사선 차폐

### Part V: Electronic Design (Section 6) / 전자 설계

#### 6.1 MHC (Mechanism and Heater Control) / 메커니즘 및 히터 제어 (pp. 42–44)

MHC는 Intel 8085 방사선 경화 마이크로프로세서 기반으로, RS-422 인터페이스를 통해 ICU와 통신합니다. 125개 파라미터를 관리하며 Safe/Idle/Command Active 세 가지 운영 모드가 있습니다.

The MHC is based on a radiation-hardened Intel 8085 microprocessor, communicating with the ICU via RS-422 interface. It manages 125 parameters with three operating modes: Safe, Idle, and Command Active.

#### 6.2 ICU (Instrument Control Unit) / 장비 제어 유닛 (pp. 45–47)

TEMIC 21020 DSP (20 MHz) 기반. 5개의 PCB: spacecraft interface, processor board, camera/mechanism controller, analog monitor, power supply unit. 우주선 질량 메모리의 15% (총 7 Gbit 중 ~1 Gbit)를 EIS가 사용합니다. 노르웨이 Svalbard 지상국 활용으로 일 15회 이상 접촉이 가능하여, 무손실 압축 시 ~100 kb/s 데이터 전송률을 달성합니다.

Based on TEMIC 21020 DSP (20 MHz). Five PCBs: spacecraft interface, processor board, camera/mechanism controller, analog monitor, power supply unit. EIS uses 15% of spacecraft mass memory (~1 Gbit out of 7 Gbit total). Norwegian Svalbard ground station enables ≥15 daily contacts, achieving ~100 kb/s data rate with lossless compression.

소프트웨어 모드: Boot → Standby → Manual → Auto → Bake-out, Emergency (Figure 21).

Software modes: Boot → Standby → Manual → Auto → Bake-out, Emergency (Figure 21).

### Part VI: Instrument Calibration and Performance (Section 7) / 장비 교정 및 성능

#### 7.1 파장 교정 / Wavelength Calibration (pp. 47–50)

Penning 방전 램프(He, Ne, Mg)를 사용한 EUV 방사로 교정. He II 256.32 Å 선의 FWHM = 0.056 Å (2.5 pixels), 분해능 λ/Δλ = 4570.

Calibrated using EUV radiation from a Penning discharge lamp (He, Ne, Mg). The He II 256.32 Å line yields FWHM = 0.056 Å (2.5 pixels), resolving power λ/Δλ = 4570.

분산 관계 (2차 다항식 피팅) / Dispersion relation (second-order polynomial fit):

$$\lambda(p) = \lambda_0 + Ap + Bp^2$$

장파장 대역 (LW): $\lambda_0 = 199.9389$ Å, $A = 0.022332$ Å/pixel, $B = -1.329 \times 10^{-8}$ Å/pixel² (표준편차 0.00415 Å, 32개 표준선)

단파장 대역 (SW): $\lambda_0 = 166.131$ Å, $A = 0.022317$ Å/pixel, $B = -1.268 \times 10^{-8}$ Å/pixel² (표준편차 0.00386 Å, 65개 표준선)

LW band: $\lambda_0 = 199.9389$ Å, $A = 0.022332$ Å/pixel, $B = -1.329 \times 10^{-8}$ Å/pixel² (std. dev. 0.00415 Å, 32 standard lines)

SW band: $\lambda_0 = 166.131$ Å, $A = 0.022317$ Å/pixel, $B = -1.268 \times 10^{-8}$ Å/pixel² (std. dev. 0.00386 Å, 65 standard lines)

$B$ 항의 기여는 매우 작아(2048 pixel에서 ~0.05 Å) 분산은 사실상 선형에 가깝습니다.

The $B$ term contribution is very small (~0.05 Å across 2048 pixels), so the dispersion is essentially linear.

#### 7.2 유효 면적 결정 / Effective Aperture Determination (pp. 50–55)

Brookhaven 싱크로트론 광원(SLS)에서 거울·격자 반사율, 필터 투과율을 개별 측정한 후, 교정용 hollow cathode 램프로 end-to-end 응답도(responsivity) $D_p$를 실측했습니다.

Individual measurements of mirror/grating reflectivities and filter transmittances at Brookhaven SLS, followed by end-to-end responsivity $D_p$ measurement using a calibrated hollow cathode lamp.

**유효 면적 (Figure 30) / Effective Area:**
- SW 대역 피크: ~0.30 cm² (약 185 Å 부근)
- LW 대역 피크: ~0.11 cm² (약 270 Å 부근)

실측 응답도와 예측 응답도 간에 ~1.6배 차이가 있어 정규화 계수(Norm = 1.60)를 적용했습니다. 이 차이의 상당 부분은 CCD QE에 기인하며, 실제 비행 CCD의 QE는 공학 모델 대비 ~60%로 추정됩니다.

A factor of ~1.6 discrepancy between measured and predicted responsivities required a normalization factor (Norm = 1.60). Much of this difference is attributed to CCD QE, with flight CCD QE estimated at ~60%.

**전체 절대 교정 불확도: 22% (상대 표준 불확도)** — 거울 면적(2%), 슬릿 면적(4%), QE 산포(4%) 등의 합산.

**Overall absolute calibration uncertainty: 22% (relative standard uncertainty)** — sum of mirror area (2%), slit area (4%), QE spread (4%), etc.

#### 7.3 장비 성능 — 예상 카운트율 / Instrument Performance — Expected Count Rates (pp. 55–59)

CHIANTI V4 데이터베이스의 합성 스펙트럼과 DEM 곡선을 사용하여, 세 가지 태양 조건에서의 예상 카운트율을 산출했습니다 (Tables 11–13):

Using synthetic spectra from CHIANTI V4 database and DEM curves, expected count rates were calculated for three solar conditions (Tables 11–13):

**Quiet Sun (Table 11) — 가장 밝은 선들 / Brightest lines:**
- Fe XII 195.12 Å: 36.63 DN s⁻¹ pixel⁻¹ (log T = 6.10)
- Fe XII 193.52 Å: 20.60 DN s⁻¹ pixel⁻¹

**Active Region (Table 12) — 가장 밝은 선들 / Brightest lines:**
- Fe XII 195.12 Å: 690.66 DN s⁻¹ pixel⁻¹ (log T = 6.10)
- Fe XII 193.52 Å: 388.63 DN s⁻¹ pixel⁻¹
- Fe XV 284.16 Å: 226.17 DN s⁻¹ pixel⁻¹ (log T = 6.30, LW 대역 최고)

**Flare (Table 13) — 가장 밝은 선들 / Brightest lines:**
- Fe XXIV 192.03 Å: 119458.12 DN s⁻¹ pixel⁻¹ (log T = 7.20)
- Ca XVII 192.82 Å: 148003.11 DN s⁻¹ pixel⁻¹ (log T = 6.70)
- Fe XV 284.16 Å: 35546.88 DN s⁻¹ pixel⁻¹ (log T = 6.30)

이 수치들로부터, 활동영역에서 12개 방출선의 영상을 ~1–2분 내에 획득할 수 있으며, 플레어 활동 루프 50 Mm 구간은 ~1분 내 스캔 가능합니다.

From these numbers, useful images of an active region in 12 emission lines can be obtained in ~1–2 minutes, and a 50 Mm section of a flaring active region loop can be scanned in ~1 minute.

복사율에서 DN으로의 변환 (Equation 2):

Conversion from radiance to data numbers (Equation 2):

$$I_\lambda = (D_L / D_p)(1/A_s)\left((180.0 \times 60.0^2)/\pi\right)^2 \text{ photons cm}^{-2}\text{ s}^{-1}\text{ sr}^{-1}$$

검출기 픽셀당 초당 등록 광자수의 기본 표현 (Equation 5):

Basic expression for photons registered per detector pixel per second (Equation 5):

$$N_\lambda = \phi_\lambda A \omega_d T_{\text{ff}}(\lambda) T_{\text{spider}} R_m(\lambda) E_g(\lambda) V_d(\lambda) T_{\text{sf}}(\lambda) E_{\text{det}}(\lambda)$$

여기서 각 항은 전면 필터 투과율, 거미줄 구조 차폐율, 거울 반사율, 격자 효율, 비네팅 계수, 슬릿 필터 투과율, 검출기 QE를 나타냅니다.

Where each term represents front filter transmission, spider assembly blocking fraction, mirror reflectivity, grating efficiency, vignetting factor, slit filter transmission, and detector QE respectively.

### Part VII: Conclusions (Section 8) / 결론

EIS는 1 MK 이하부터 20 MK까지의 온도 범위에서 2″ 공간 분해능, ±5 km s⁻¹ 속도 측정 정밀도를 달성합니다. 선 프로파일 분석으로 비열적 효과와 난류를 탐지할 수 있으며, DEM 재구성과 원소 조성(FIP effect) 분석이 가능합니다. multilayer coating + back-illuminated CCD + toroidal grating의 조합으로 170–290 Å 대역에서 이전 장비 대비 10배의 유효 면적과 높은 데이터율을 달성하여, 과도현상의 고시간 분해능 연구가 가능해졌습니다.

EIS achieves 2″ spatial resolution and ±5 km s⁻¹ velocity measurement precision across the temperature range from below 1 MK to 20 MK. Line profile analysis enables detection of nonthermal effects and turbulence, along with DEM reconstruction and elemental composition (FIP effect) analysis. The combination of multilayer coatings + back-illuminated CCDs + toroidal grating delivers 10× the effective area of previous instruments in the 170–290 Å range with higher data rates, enabling high-cadence studies of transient phenomena.

---

## 3. Key Takeaways / 핵심 시사점

1. **Normal incidence가 EUV 분광의 패러다임을 바꿨다** — Grazing incidence에서 normal incidence + multilayer coating으로의 전환은 유효 면적을 10배 증가시키고, 광학계를 크게 단순화했습니다. 이는 Mo/Si multilayer 기술의 성숙 덕분에 가능했습니다.
   Normal incidence changed the paradigm of EUV spectroscopy — the transition from grazing to normal incidence + multilayer coatings increased effective area by 10× and greatly simplified the optical system, enabled by the maturation of Mo/Si multilayer technology.

2. **이중 D-shaped 코팅 설계가 단일 광학계로 두 파장 대역을 동시 커버한다** — 거울과 격자 각각을 D자 형태로 분할하여 SW/LW에 별도 최적화된 multilayer를 적용함으로써, 단 2개의 광학 소자로 170–290 Å 전체를 커버합니다.
   The dual D-shaped coating design covers two wavelength bands simultaneously with a single optical system — by dividing mirror and grating into D-shaped sectors with separately optimized multilayers for SW/LW, the entire 170–290 Å range is covered with just two optical elements.

3. **Back-illuminated CCD가 microchannel plate를 대체했다** — QE 39–44%의 back-thinned CCD는 microchannel plate (≤20%)에 비해 2–3배 높은 양자 효율을 제공하며, hygroscopic KBr 코팅이 불필요하여 장기 안정성도 우수합니다.
   Back-illuminated CCDs replaced microchannel plates — back-thinned CCDs with 39–44% QE provide 2–3× higher quantum efficiency than microchannel plates (≤20%), and eliminate the need for hygroscopic KBr coatings, improving long-term stability.

4. **22 mÅ/pixel의 스펙트럼 분해능이 ~3 km/s Doppler 측정을 가능하게 한다** — 이는 코로나 플라즈마의 느린 흐름(upflows, downflows)과 비열적 속도를 정량적으로 측정할 수 있는 수준입니다.
   22 mÅ/pixel spectral resolution enables ~3 km/s Doppler measurements — sufficient for quantitatively measuring slow coronal plasma flows (upflows, downflows) and nonthermal velocities.

5. **오염 관리가 EUV 장비의 생존을 결정한다** — 3년 분자 fluence < 2.7 × 10⁻⁶ g cm⁻²라는 극도로 엄격한 요구사항은 6주간의 진공 베이크아웃, 지속적인 N₂ 퍼지, CFRP 아웃가싱 제어 등 방대한 오염 관리 프로그램을 필요로 했습니다.
   Contamination control determines the survival of EUV instruments — the extremely strict requirement of < 2.7 × 10⁻⁶ g cm⁻² molecular fluence over 3 years necessitated a 6-week vacuum bake-out, continuous N₂ purge, and extensive CFRP outgassing control.

6. **슬릿과 슬롯의 이중 기능이 분광과 영상을 겸비한다** — 좁은 슬릿(1–2″)은 고분해능 분광에, 넓은 슬롯(40–266″)은 단색 영상(monochromatic imaging)에 사용되어, 하나의 장비로 두 가지 관측 모드를 제공합니다.
   The dual slit/slot capability combines spectroscopy and imaging — narrow slits (1–2″) for high-resolution spectroscopy and wide slots (40–266″) for monochromatic imaging, providing two observation modes from a single instrument.

7. **CHIANTI 데이터베이스와의 연동이 장비 성능 예측의 핵심이다** — CHIANTI V4의 합성 스펙트럼과 DEM 모델을 사용하여 quiet Sun, active region, flare 조건별 예상 카운트율을 산출한 것은, 관측 프로그램 설계와 노출 시간 최적화에 필수적입니다.
   Integration with the CHIANTI database is key to performance prediction — using CHIANTI V4 synthetic spectra and DEM models to calculate expected count rates for quiet Sun, active region, and flare conditions is essential for observation program design and exposure time optimization.

8. **SOT·XRT와의 동시 관측이 EIS의 과학적 가치를 극대화한다** — 광구 자기장(SOT, 0.2″)–코로나 구조(XRT)–코로나 플라즈마 진단(EIS)의 동시 관측은 태양 대기의 자기적 연결성을 종합적으로 연구할 수 있게 합니다.
   Coordinated observations with SOT and XRT maximize EIS's scientific value — simultaneous observation of photospheric magnetic fields (SOT, 0.2″), coronal structure (XRT), and coronal plasma diagnostics (EIS) enables comprehensive study of magnetic connectivity in the solar atmosphere.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 방출선 강도 / Emission Line Intensity

광학적으로 얇은 플라즈마에서의 방출선 강도:

Emission line intensity from optically thin plasma:

$$I_\lambda = \int G(T) \cdot \text{DEM}(T) \, dT \quad \text{photons cm}^{-2}\text{ s}^{-1}\text{ sr}^{-1}$$

여기서 $G(T)$는 emissivity function (온도 비의존 파라미터 포함), $\text{DEM}(T)$는 differential emission measure입니다. CHIANTI V4 데이터와 $10^{16}$ cm⁻³ 일정 압력을 가정하여 계산됩니다.

Where $G(T)$ is the emissivity function (including temperature-independent parameters) and $\text{DEM}(T)$ is the differential emission measure. Calculated using CHIANTI V4 data assuming constant pressure of $10^{16}$ cm⁻³.

### 4.2 파장 분산 관계 / Wavelength Dispersion Relation

$$\lambda(p) = \lambda_0 + Ap + Bp^2$$

| 파라미터 / Parameter | LW 대역 / LW Band | SW 대역 / SW Band |
|---|---|---|
| $\lambda_0$ [Å] | 199.9389 | 166.131 |
| $A$ [Å/pixel] | 0.022332 | 0.022317 |
| $B$ [Å/pixel²] | $-1.329 \times 10^{-8}$ | $-1.268 \times 10^{-8}$ |
| 피팅 표준편차 / Fit std. dev. [Å] | 0.00415 | 0.00386 |

스펙트럼 분해능 / Spectral resolution: FWHM = 0.056 Å (2.5 pixels) at He II 256 Å → $\lambda / \Delta\lambda = 4570$

### 4.3 검출기 등록 광자수 / Detected Photon Count

검출기 픽셀당 초당 등록 광자수:

Photons registered per detector pixel per second:

$$N_\lambda = \phi_\lambda A \omega_d T_{\text{ff}}(\lambda) T_{\text{spider}} R_m(\lambda) E_g(\lambda) V_d(\lambda) T_{\text{sf}}(\lambda) E_{\text{det}}(\lambda)$$

각 항의 의미 / Meaning of each term:
- $\phi_\lambda$: 태양 복사 강도 / Solar radiance [photons cm⁻² s⁻¹ sr⁻¹]
- $A$: 거울 면적 / Mirror area [cm²]
- $\omega_d$: 검출기 픽셀 입체각 / Detector pixel solid angle [sr]
- $T_{\text{ff}}$: 전면 필터 투과율 / Front filter transmittance
- $T_{\text{spider}}$: 필터 메쉬 개구율 / Filter mesh open area fraction (~0.80)
- $R_m$: 거울 반사율 / Mirror reflectivity (SW: ~32%, LW: ~23%)
- $E_g$: 격자 효율 / Grating efficiency (1st order: ~8%)
- $V_d$: 비네팅 계수 / Vignetting factor (LW > 272 Å에서 영향)
- $T_{\text{sf}}$: 슬릿 필터 투과율 / Slit filter transmittance
- $E_{\text{det}}$: 검출기 양자 효율 / Detector quantum efficiency

유효 면적으로 표현하면 / Expressed using effective area:

$$N_\lambda = I_\lambda A_{\text{eff}}(\lambda) \omega_d$$

### 4.4 복사율-DN 변환 / Radiance to DN Conversion

$$I_\lambda = \frac{D_L}{D_p} \cdot \frac{1}{A_s} \cdot \left(\frac{180.0 \times 60.0^2}{\pi}\right)^2 \quad \text{photons cm}^{-2}\text{ s}^{-1}\text{ sr}^{-1}$$

- $D_L$: 스펙트럼 선의 디지털 신호 / Digital signal of spectrum line [DN sec⁻¹]
- $D_p$: 응답도 / Responsivity [DN/photon]
- $A_s$: 슬릿이 조사하는 원천 면적 / Source area illuminated by slit [arcsec²]

### 4.5 Doppler 속도 및 비열적 속도 / Doppler Velocity and Nonthermal Velocity

$$v_{\text{Doppler}} = c \cdot \frac{\Delta\lambda}{\lambda_0}$$

1 pixel = 22 mÅ ≈ 25 km s⁻¹ → 속도 측정 정밀도 ±5 km s⁻¹

비열적 속도 추출 / Nonthermal velocity extraction:

$$\Delta\lambda_{\text{obs}}^2 = \Delta\lambda_{\text{inst}}^2 + 4\ln 2 \cdot \frac{\lambda_0^2}{c^2}\left(\frac{2k_BT}{m_i} + \xi^2\right)$$

여기서 $\xi$는 비열적 속도 (±25 km s⁻¹ 정밀도).

Where $\xi$ is the nonthermal velocity (±25 km s⁻¹ precision).

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1985  Skylab ATM 데이터 분석 완료 — EUV 분광의 첫 번째 황금기
      Skylab ATM data analysis complete — first golden age of EUV spectroscopy
      |
1995  SOHO 발사 — CDS & SUMER: grazing incidence EUV 분광의 새 시대
      SOHO launch — CDS & SUMER: new era of grazing incidence EUV spectroscopy
      |
1998  TRACE 발사 — EUV 영상의 공간 분해능 혁명 (~1″), 그러나 분광 없음
      TRACE launch — spatial resolution revolution in EUV imaging (~1″), but no spectroscopy
      |
2002  RHESSI 발사 — 하드 X선/감마선 영상 분광
      RHESSI launch — hard X-ray/gamma-ray imaging spectroscopy
      |
2006  Hinode(Solar-B) 발사 — SOT + XRT + EIS 동시 관측 시작
      Hinode launch — coordinated SOT + XRT + EIS observations begin
      |
2007  ★ 본 논문: EIS 장비 기술 및 성능 상세 보고 ★
      ★ This paper: detailed EIS instrument description and performance ★
      |
2010  SDO 발사 — AIA (전천 EUV 영상) + EVE (전일면 EUV 분광)
      SDO launch — AIA (full-disk EUV imaging) + EVE (full-disk EUV spectroscopy)
      |
2013  IRIS 발사 — FUV/NUV 고분해능 분광 (전이영역/채층에 특화)
      IRIS launch — FUV/NUV high-resolution spectroscopy (specialized for TR/chromosphere)
      |
2020  Solar Orbiter 발사 — SPICE: EIS의 후속 EUV 분광기
      Solar Orbiter launch — SPICE: EIS's successor EUV spectrometer
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Harrison et al. (1995) — SOHO/CDS | EIS의 직접적 전신. Grazing incidence + microchannel plate 설계. EIS는 CDS 대비 유효 면적 10배, 분해능 10배 향상 / Direct predecessor of EIS. Grazing incidence + microchannel plate design. EIS achieves 10× effective area and resolution improvement over CDS | 높음 / High — 설계 동기와 성능 비교의 기준 / Design motivation and performance benchmark |
| Tsuneta et al. (2008) — Hinode/SOT (#14) | 같은 Hinode 미션의 광학 망원경. SOT의 광구 자기장 관측과 EIS의 코로나 진단이 상보적 / Optical telescope on the same Hinode mission. SOT's photospheric magnetic field observations complement EIS's coronal diagnostics | 높음 / High — 동시 관측 파트너 / Coordinated observation partner |
| Korendyke et al. (2006) — EIS optics | EIS 광학계의 상세 설계와 메커니즘 기술 / Detailed design and mechanism description of EIS optics | 높음 / High — 본 논문의 광학 섹션의 기반 참고문헌 / Foundation reference for the optics section |
| Lang et al. (2006) — EIS calibration | EIS의 실험실 교정에 대한 상세 보고 / Detailed report on laboratory calibration of EIS | 높음 / High — 본 논문 Section 7의 기반 / Foundation for Section 7 |
| Seely et al. (2004) — Mo/Si multilayers | EIS에 사용된 Mo/Si multilayer coating의 최적화 설계 / Optimization design of Mo/Si multilayer coatings used in EIS | 중간 / Medium — 핵심 기술의 원천 / Source of core technology |
| Dere et al. (1997) — CHIANTI database | EIS 성능 예측에 사용된 원자 데이터베이스 / Atomic database used for EIS performance prediction | 중간 / Medium — 과학적 성능 평가의 기반 / Foundation for scientific performance assessment |

---

## 7. References / 참고문헌

- J.L. Culhane et al., "The EUV Imaging Spectrometer for Hinode," *Solar Phys.*, 243, 19–61, 2007. [DOI: 10.1007/s01007-007-0293-1]
- R.A. Harrison et al., "The Coronal Diagnostic Spectrometer for the Solar and Heliospheric Observatory," *Solar Phys.*, 162, 233, 1995.
- S. Tsuneta et al., "The Solar Optical Telescope for the Hinode Mission: An Overview," *Solar Phys.*, 249, 167–196, 2008. [DOI: 10.1007/s11207-008-9174-z]
- C.M. Korendyke et al., "Optics and mechanisms for the EIS on Solar-B," *Appl. Opt.*, 45, 8674, 2006.
- J. Lang et al., "Laboratory calibration of the EUV Imaging Spectrometer for the Solar-B satellite," *Appl. Opt.*, 45, 8689, 2006.
- J.F. Seely et al., "Multilayer-coated laminar grating and toroidal mirror," *Appl. Opt.*, 43, 1463, 2004.
- K.P. Dere et al., "CHIANTI — An Atomic Database for Emission Lines," *Astron. Astrophys. Suppl.*, 125, 149, 1997.
- P.R. Young et al., "CHIANTI — An Atomic Database for Emission Lines. VI," *Astron. Astrophys.*, 2003.
- H. Warren, "A Solar Minimum Irradiance Spectrum for Wavelengths below 1200 Å," *Astrophys. J. Suppl.*, 157, 147, 2005.
- H. Warren, G.A. Doschek, "Properties of flare loops," *Astrophys. J.*, 618, L157, 2005.
