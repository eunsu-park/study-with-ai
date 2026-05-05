---
title: "Extreme Ultraviolet Variability Experiment (EVE) on the Solar Dynamics Observatory (SDO): Overview of Science Objectives, Instrument Design, Data Products, and Model Developments"
authors: "Woods, T.N., Eparvier, F.G., Hock, R., Jones, A.R., Woodraska, D., Judge, D., Didkovsky, L., Lean, J., Mariska, J., Warren, H., McMullin, D., Chamberlin, P., Berthiaume, G., Bailey, S., Fuller-Rowell, T., Sojka, J., Tobiska, W.K., Viereck, R."
year: 2012
journal: "Solar Physics 275, 115-143"
doi: "10.1007/s11207-009-9487-6"
topic: Solar_Observation
tags: [EVE, SDO, EUV, irradiance, MEGS, ESP, space-weather, FISM, NRLEUV]
status: completed
date_started: 2026-04-25
date_completed: 2026-04-25
---

# 51. Extreme Ultraviolet Variability Experiment (EVE) on SDO / SDO 탑재 극자외선 변동성 실험

---

## 1. Core Contribution / 핵심 기여

EVE는 NASA의 Solar Dynamics Observatory (SDO)에 탑재된 세 가지 주요 관측기(EVE, HMI, AIA) 중 하나로, 0.1–105 nm 파장 범위에서 태양 극자외선(EUV) 분광 복사 조도(spectral irradiance)를 0.1 nm 분광 분해능, 10초 시간 분해능, 20% 절대 정확도로 동시 측정하기 위해 설계된 분광기 모음이다. EVE는 다섯 개의 검출기 채널 — MEGS-A (5–37 nm grazing-incidence), MEGS-B (35–105 nm normal-incidence dual-pass), MEGS-SAM (0.1–7 nm pinhole + photon counting), MEGS-P (121.6 nm Lyman-α), 그리고 ESP (0.1–39 nm 5채널 광대역 transmission grating photometer) — 을 결합하여 기존의 SOHO/SEM, TIMED/SEE 같은 일별/광대역 측정에서 도약한 시-분광 능력을 제공한다. 본 논문은 (i) 네 가지 과학 목표(EUV 명세화, 변동 이해, 예보, 지구 환경 응답), (ii) 각 기기의 광학·검출기·필터 설계 (Table 1), (iii) Level 0C → 0CS → 1 → 2 → 3 데이터 제품 위계 (Table 2), (iv) 39개 EUV 라인/밴드 목록 (Table 3), 그리고 (v) NRLEUV (DEM 기반 물리 모델), FISM (경험적 1 nm/1분 모델), SIP/SOLAR2000/SFLR (운용 시스템), CTIPe·NRLMSIS·JB2006/2008 (대기 모델)과의 모델 연계를 종합적으로 정리한다.

EVE is one of three primary instruments on NASA's Solar Dynamics Observatory (SDO; alongside HMI and AIA), designed to measure solar EUV spectral irradiance from 0.1 to 105 nm with simultaneous 0.1 nm spectral resolution, 10 s temporal cadence, and 20 % absolute accuracy. EVE combines five detector channels — MEGS-A (5–37 nm grazing-incidence spectrograph), MEGS-B (35–105 nm normal-incidence dual-pass cross-dispersing spectrograph), MEGS-SAM (0.1–7 nm pinhole camera with photon counting), MEGS-P (121.6 nm Lyman-α photometer), and ESP (0.1–39 nm five-channel broadband transmission-grating photometer) — to provide a temporal-spectral capability that leapfrogs prior daily/broadband measurements (SOHO/SEM, TIMED/SEE). The paper systematically presents (i) four science objectives (specify EUV irradiance, understand variability, forecast variations, understand geospace response), (ii) optical/detector/filter design of each instrument (Table 1), (iii) the Level 0C → 0CS → 1 → 2 → 3 data-product hierarchy (Table 2), (iv) a 39-line/band EUV target list (Table 3), and (v) the modeling ecosystem — NRLEUV (DEM-based physics model), FISM (empirical 1 nm/1 min model), SIP/SOLAR2000/SFLR (operational systems), and CTIPe/NRLMSIS/JB2006/2008 (atmospheric models) — that EVE both feeds and validates.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction (Section 1) / 서론

EVE는 SDO LWS 미션의 세 가지 관측기 중 하나로, AIA(완전 디스크 EUV 이미저)와 HMI(자기장 매핑) 와 함께 작동한다. SDO의 과학적 주제는 "예보 운용을 가능케 하는 과학 연구"(research to operations) 이며, EVE는 이 주제에서 EUV 복사 조도 입력을 책임진다. EVE의 측정은 SOHO/SEM(1996년~)과 TIMED/SEE(2003년~) 의 EUV 기록을 연속하면서 분해능을 크게 개선한다. EVE 개발 과정에서 다음 기술들이 상당히 진보했다:

- **Radiation-hard back-illuminated CCD** — MIT-LL 제작, 1024 × 2048, –70°C에서 ~1 e⁻ 수준 노이즈 (Westhoff et al. 2007).
- **Off-Rowland-circle grazing-incidence design** — Crotser et al. 2007. CCD를 거의 normal incidence에 배치하여 grazing-incidence 분광기에서 매우 높은 감도 달성.
- **MEGS-SAM pinhole + photon counting** — 분광기 없이 0.1–7 nm XUV 스펙트럼 + 이미지를 제공.

EVE plays the EUV-input role within SDO's three-instrument suite (EVE, HMI, AIA) under the LWS theme of "research to operations". Its measurements continue the SOHO/SEM (1996–) and TIMED/SEE (2003–) EUV records while greatly improving resolution. EVE's development advanced several technologies: radiation-hard back-illuminated CCDs (~1 e⁻ noise at –70 °C), off-Rowland-circle grazing-incidence design, and MEGS-SAM's pinhole+photon-counting approach.

### Part II: EVE Science Plan (Section 2) / EVE 과학 계획

EVE는 네 가지 과학 목표를 추구한다 (p. 118):

1. **Objective 1 — Specify EUV irradiance**: 초/분/시간/년 단위 변동을 정량적으로 명세 / Specify variability across timescales.
2. **Objective 2 — Understand why variations occur**: 자기장-복사 연결, DEM-based 물리 / Magnetic-field–irradiance connection.
3. **Objective 3 — Forecast EUV**: nowcast (실시간) ~ 5 solar rotations (예보) / nowcast to multi-rotation forecast.
4. **Objective 4 — Geospace response**: 이온층/열권/위성 항력 / Ionosphere/thermosphere/drag.

**Figure 1**은 NRLMSIS 모델로 추정한 열권 온도(400 → 1100 K)와 중성 밀도(500 km에서 10× 변동)의 11년 주기 변화를 보여준다. **Figure 2**는 1965–2005 EUV 광자 에너지 (λ < 120 nm) vs 태양풍 운동 에너지 비교로, EUV가 지구 상부 대기의 주요 에너지 입력임을 강조한다.

EVE pursues four science objectives. Figure 1 shows NRLMSIS-modeled thermospheric temperature (400 → 1100 K) and 500 km neutral density (factor-of-10 variation) over the 11-year cycle. Figure 2 shows that EUV photon energy (λ < 120 nm) dominates over solar-wind kinetic input as the primary driver of Earth's upper atmosphere.

#### Objective 1 detail / 목표 1 상세

EUV 스펙트럼은 채층/전이층/코로나에서 $10^4$–$10^7$ K 범위로 방출되는 수천 개의 라인과 몇몇 연속체로 구성된다. 라인별로 다른 온도/밀도 영역에서 발생하므로 변동성은 파장 의존성이 복잡하다. TIMED/SEE는 ~20 % 절대 정확도, 1 nm 분해능을 제공했다. EVE는 일부 파장에서 ~10 % 절대 정확도와 0.1 nm 분해능을 달성한다.

The EUV spectrum has thousands of emission lines and a few continua emitted at $10^4$–$10^7$ K. EVE achieves ~10 % absolute accuracy at some wavelengths and 0.1 nm resolution, improving on TIMED/SEE's 20 % and 1 nm.

#### Objective 2 detail / 목표 2 상세

EUV 복사는 비균질한 태양 대기에서 emission measure가 자기장에 의해 조절되는 영역(active regions, coronal holes, quiet Sun, active network)에서 나온다. 4 성분 NRLEUV 모델은 이 비균질성을 emission-measure distribution과 영상 데이터의 조합으로 모델링한다. EVE + AIA + HMI 의 결합은 자기 플럭스 → 코로나 휘도 → 디스크 통합 복사 의 인과관계를 추적할 수 있게 한다.

EUV emission originates in inhomogeneous solar atmosphere where magnetic fields modulate emission measures (active regions, coronal holes, quiet Sun, active network). The 4-component NRLEUV model uses Skylab-derived DEMs combined with imaging. EVE+AIA+HMI together trace magnetic flux → coronal brightness → disk-integrated irradiance.

#### Objective 3 detail / 목표 3 상세

세 가지 시간 척도의 예보 기법:
- **장기(년)**: F10.7 ↔ EUV 통계 회귀 / Statistical regression.
- **중기(자전, ~27일)**: HMI far-side helioseismology + AIA limb-brightening (Figure 5). / Far-side helioseismology + east-limb forecasting.
- **단기(시간~일)**: HMI/AIA precursor 자기 구성 + EVE 적분 + flare/EUV 연관성. / Precursor configurations.

Three forecast timescales: long (yearly, F10.7-based regression), medium (rotation, HMI helioseismology + east-limb forecasting in Figure 5), and short (hours, precursor magnetic configurations).

#### Objective 4 detail / 목표 4 상세

EUV 변동은 직간접적으로 우주 기상 현상을 일으킨다. EVE는 CTIPe (NOAA), TDIM/IFM/GAIM (USU), NRLMSIS (NRL) 모델에 입력되며 측정과 시뮬레이션의 검증 루프를 닫는다.

EUV variability drives space-weather phenomena directly and indirectly. EVE feeds CTIPe (NOAA), TDIM/IFM/GAIM (USU), NRLMSIS (NRL) models, closing the measurement–simulation validation loop.

### Part III: EVE Instrumentation (Section 3) / EVE 기기 설계

#### Table 1 요약 / Summary of Table 1

| Instrument | λ Range (nm) | Δλ (nm) | Cadence (s) | Detector | Description |
|---|---|---|---|---|---|
| MEGS-A1 (slit 1) | 5–18 | 0.1 | 10 | 1024×2048 BI CCD | Grazing 80°, off-Rowland |
| MEGS-A2 (slit 2) | 17–37 | 0.1 | 10 | 1024×2048 BI CCD | Grazing 80°, off-Rowland |
| MEGS-B | 35–105 | 0.1 | 10 | 1024×2048 BI CCD | Normal-incidence, dual-pass |
| MEGS-SAM | 0.1–7 | 1 | 10 | corner of MEGS-A CCD | Pinhole, photon counting |
| MEGS-P | 121.6 | 10 | 0.25 | AXUV-100 (10×10 mm²) | Lyman-α photometer |
| ESP Ch.1 | 36.6 | 4.7 | 0.25 | AXUV-SP2 (6×16 mm²) | Transmission grating |
| ESP Ch.2 | 25.7 | 4.5 | 0.25 | AXUV-SP2 | Transmission grating |
| ESP Ch.3 | dark | N/A | 0.25 | AXUV-SP2 | Dark reference |
| ESP Ch.8 | 18.2 | 3.6 | 0.25 | AXUV-SP2 | Transmission grating |
| ESP Ch.9 | 30.4 | 3.8 | 0.25 | AXUV-SP2 | Transmission grating |
| ESP Ch.4–7 (QD) | 0.1–7 | 6 | 0.25 | Quad Diode | Zeroth order, X-ray |

#### MEGS-A (Section 3.1) / MEGS-A 기기

MEGS-A는 80° grazing-incidence, off-Rowland-circle 분광기로 5–37 nm를 0.1 nm 분해능 이하로 측정한다. 두 입구 슬릿(slit 1, slit 2)이 각각 20 μm × 2 mm 크기로 위/아래로 배치되어 있다. Off-Rowland circle 설계는 CCD를 normal-incidence 가까이 배치하여 (입사면에 수직) MEGS-A의 감도를 크게 향상시킨다 (Crotser et al. 2007).

**필터 휠 mechanism**:
- Slit 1 primary: Zr (280 nm) / C (20 nm) → 5–18 nm pass.
- Slit 2 primary: Al (200 nm) / Ge (20 nm) / C (20 nm) → 17–37 nm pass.
- Secondary order check: Zr (230 nm) / Si (120 nm) / C (20 nm) → 13–18 nm.
- Secondary order check: Al (180 nm) / Mg (300 nm) → 25–37 nm.
- Blanked-off position: dark measurement.

**격자**: Jobin Yvon (JY) 제작 spherical holographic, R = 600 mm (radius of curvature), Pt 코팅, 767 grooves mm⁻¹, laminar groove profile (홀수 차수 억제). 입사각 α = 80°, 회절각 β = 73°–79°.

**검출기**: MIT-LL back-illuminated split-frame transfer CCD, 1024 × 2048 픽셀, 15 μm 픽셀, 두 절반 (각 512 × 2048) 8개 출력으로 동시 읽기. Gain ~2 e⁻/DN, noise ~2 e⁻ (–70°C 이하). CCD 운용 온도 –95°C ± 5°C, 25 mm Al 차폐로 임무 동안 < 10 kRad 누적.

MEGS-A is an 80° grazing-incidence, off-Rowland spectrograph covering 5–37 nm at sub-0.1 nm resolution. Two 20 μm × 2 mm entrance slits feed slit 1 (5–18 nm) and slit 2 (17–37 nm). The off-Rowland design places the CCD near normal incidence to greatly improve sensitivity. The grating is JY spherical holographic with R = 600 mm, Pt coating, 767 grooves mm⁻¹, laminar grooves to suppress even orders. The MIT-LL back-illuminated split-frame transfer CCD (1024×2048, 15 μm pixels) operates at –95±5 °C, with ~2 e⁻ noise and 25 mm Al shielding limiting dose to <10 kRad.

#### MEGS-B (Section 3.2) / MEGS-B 기기

MEGS-B는 normal-incidence, double-pass cross-dispersing Rowland circle 분광기로 35–105 nm를 0.1 nm 미만 분해능으로 측정한다. 단일 입구 슬릿(35 μm × 3.5 mm)을 사용하며, 35–105 nm 영역에는 신뢰할 수 있는 광역 차단 필터가 없으므로 두 격자(900 + 2140 grooves mm⁻¹)를 차례로 사용해 차수 분리를 달성한다.

**격자 1**: $d_1 = 1111$ nm, $\alpha_1 = 1.8°$, $\beta_1 = 4°$–$7°$.
**격자 2**: $d_2 = 467$ nm, $\alpha_2 = 14°$, $\beta_2 = 19°$–$28°$.

검출기는 MEGS-A와 동일 (MIT-LL CCD), 0.1 nm 광학 분해능에 약 3 픽셀 매핑.

MEGS-B is a normal-incidence, double-pass cross-dispersing Rowland-circle spectrograph for 35–105 nm with sub-0.1 nm resolution. A single 35 μm × 3.5 mm slit feeds two gratings ($d_1 = 1111$ nm with $\alpha_1=1.8°$, $\beta_1=4°$–$7°$; $d_2 = 467$ nm with $\alpha_2=14°$, $\beta_2=19°$–$28°$) for order separation, since no broad-band 35–105 nm filter exists. Detector is identical to MEGS-A's CCD; ~3 pixels per 0.1 nm element.

#### MEGS-SAM (Section 3.3) / MEGS-SAM 기기

SAM은 MEGS-A 본체 안에 위치한 핀홀 카메라로 별도 개구를 사용해 태양 이미지를 MEGS-A CCD의 한 모서리 영역에 투영한다. **세 가지 모드**:

1. **Aspect-monitor mode**: UV 필터 (visible/UV light) 사용, 태양 이미지 중심 결정 → 약 1 arcminute 정확도의 정렬 정보.
2. **XUV photon-counting mode**: Ti (300 nm) / Al (150 nm) / C (40 nm) 필터로 0.1–7 nm 만 통과. 핀홀과 필터를 최적화하여 10초 CCD 적분 동안 단일 광자 사건이 분리되도록 함. 광자 사건의 픽셀 카운트 합 ∝ 광자 에너지 → ≈1 nm 분광 분해능. 픽셀 위치 → 태양 이미지 → XUV 이미지. 수 분 동안 누적하면 XUV 이미지 생성.
3. **Dark mode**: Filter wheel blanked.

핀홀 직경 26 μm, MEGS-A CCD의 모서리 사용. SAM은 정상 운용에서는 XUV 모드.

SAM is a pinhole camera within MEGS-A that uses a separate aperture to image the Sun onto a corner of the MEGS-A CCD. Three modes: (1) aspect-monitor (UV filter, centroid Sun ~1 arcmin); (2) XUV photon-counting (Ti/Al/C filter, 0.1–7 nm) where single-photon events during 10 s integration give energy from charge magnitude (~1 nm spectral resolution) and position from event location (XUV image after multi-minute summation); (3) dark.

#### MEGS-P (Section 3.4) / MEGS-P 기기

MEGS-P는 IRD silicon photodiode를 MEGS-B 첫 번째 격자의 –1차에 배치한 광도계이다. Acton 122 XN interference 필터(중심 121.6 nm, 10 nm 광역)로 Lyman-α를 분리한다. 추가로 동일 photodiode가 마스크된 채 옆에 있어 background (dark) 보정용. 측정의 99% 이상은 Lyman-α이다.

MEGS-P is an IRD Si photodiode at the –1 order of MEGS-B's first grating, with an Acton 122 XN interference filter (121.6 nm center, 10 nm band). A masked twin diode provides dark/background. >99 % of signal is Lyman-α.

#### ESP (Section 3.5) / ESP 기기

ESP는 IRD silicon photodiode를 사용하는 비집속 transmission grating 광도계로 SOHO/SEM (Judge et al. 1998)의 후속이다. **구성**:
- Al (150 nm) foil filter (Luxel) — out-of-band 차단.
- Free-standing transmission grating (X-Opt Inc.) — 2500 lines mm⁻¹, 단일 박판(no substrate) 형태.
- ±1차 위치에 silicon photodiode (AXUV-SP2, 6 × 16 mm²) — 18.2, 25.7, 30.4, 36.6 nm 중심, ~4 nm bandpass.
- 0차 위치에 quadrant Si photodiode (AXUV-PS5) + 추가 Ti (300 nm) / C (40 nm) 필터 → 0.1–7 nm 격리. Quadrants 합 = 광역, 차감 → 디스크 상 플레어 위치 (Didkovsky et al. 2010).
- 25.7 nm 채널은 dark reference로도 활용.
- 측정 cadence 0.25초 — EVE 전체에서 가장 빠름.

ESP is a non-focusing transmission-grating spectrograph with IRD Si photodiodes, descended from SOHO/SEM. It uses a 150 nm Al foil filter, a free-standing 2500 lines mm⁻¹ X-Opt grating, photodiodes (AXUV-SP2, 6×16 mm²) at ±1 order centered at 18.2, 25.7, 30.4, 36.6 nm with ~4 nm bandpass, and a quadrant Si diode (AXUV-PS5) at 0 order behind a 300 nm Ti / 40 nm C filter for 0.1–7 nm. Quadrant differencing localizes flares on the disk. Cadence is 0.25 s (fastest in EVE).

#### EVE Optical Package (Section 3.6) / EVE 광학 패키지

EOP에 모든 기기를 마운트, EEB(Electronics Box)는 SDO 데크에 별도 마운트. 자원: 질량 54 kg, 평균 전력 44 W, 하우스키핑 텔레메트리 2 kbps, 과학 텔레메트리 7 Mbps, S-band 인터페이스 RS-1553.

EOP holds all instruments; EEB on SDO deck. Resources: 54 kg mass, 44 W average power, 2 kbps housekeeping, 7 Mbps science telemetry, RS-1553/S-band interfaces.

### Part IV: Data Products (Section 4) / 데이터 제품

#### Level 0C/0CS — Space-Weather Products / 우주 기상 제품

ASCII 형식, ~15 분 지연 (0CS는 ~1 분). 6–105 nm, 0.1 nm 간격, 1 분 평균. 광대역 측정(MEGS-P, ESP, ESP QD)도 포함. 15분 지연은 White Sands DDS 처리 시간(~3분)에 기인.

ASCII files, ~15 min latency (0CS ~1 min). 6–105 nm at 0.1 nm intervals, 1-min averaging. Includes broadband measurements. The 15 min delay is mostly White Sands DDS processing.

#### Level 1 — Per-instrument fully calibrated / 기기별 완전 보정

각 기기별 분리 (MEGS-A1, MEGS-A2, MEGS-B, MEGS-P, MEGS-SAM, ESP). 0.02 nm 간격, 10초 cadence (MEGS), 0.25 s (ESP/MEGS-P). 14:00 UT부터 전일 처리 시작 (이전 24시간 ancillary 데이터 완성 위해), 48 노드 GNU-Linux 클러스터에서 병렬 처리. 1시간 단위 파일.

Each instrument separately calibrated. 0.02 nm intervals, 10 s cadence (MEGS), 0.25 s (ESP/MEGS-P). Processing starts at 14:00 UT each day for the prior UTC day. Hourly files; 48-node GNU-Linux cluster.

#### Level 2 — Merged spectra / 병합 스펙트럼

MEGS-A1 (6–16 nm) + MEGS-A2 (16–37 nm) + MEGS-B (37–105 nm) → 0.02 nm 균일 격자에 splice. 1시간 단위 파일, 10 초 cadence. **Table 3**의 39개 라인/밴드를 추출.

MEGS-A1 (6–16 nm) + A2 (16–37 nm) + B (37–105 nm) spliced onto 0.02 nm uniform grid. Hourly files, 10 s cadence. Extracts 39 lines/bands from Table 3.

#### Level 3 — Daily averages / 일평균

1년 단위 파일, 0.02 nm/0.1 nm/1 nm 빈. 일평균은 중앙값(median)으로 계산하여 플레어 제외 → quiet-Sun reference. SAM 일평균 0.1–6 nm 별도 제공.

Yearly files at 0.02/0.1/1 nm bins. Daily values are medians (excluding flares) → quiet-Sun reference. SAM daily averages over 0.1–6 nm in 0.1 nm bins.

#### Table 3 — Lines and bands / 라인과 밴드

39개 항목, 핵심 examples:

| Line/Band | λ (nm) | log T | Description |
|---|---|---|---|
| Fe XVIII | 9.39 | 6.8 | Hot flare line |
| Fe XX | 13.29 | 7.0 | Very hot flare |
| Fe XXIII | 13.29 | — | (paired w/ XX) |
| Fe IX | 17.11 | 5.8 | AIA 171 channel |
| Fe XII | 19.51 | 6.1 | AIA 193 |
| Fe XV | 28.42 | 6.3 | AIA 211 / late phase |
| He II | 30.4 | 4.7 | Strong chromosphere |
| H I | 121.6 | 4.3 | Lyman-α (MEGS-P) |
| AIA 94 | 9.35–9.44 | 5.4 | Fe XVIII proxy |
| AIA 304 | 29.74–31.01 | 4.8 | He II proxy |
| GOES B | 25–34 nm | — | EUVS-B |
| GOES/XRS-B | 0.1–0.8 nm | — | XRS proxy |

Table 3 captures Fe lines spanning $\log T = 5.4$ – $7.0$ (Fe IX through Fe XXIII), He II, H I Lyman-α, AIA-equivalent bands, GOES proxies, and SOHO/SEM equivalents.

### Part V: Solar Irradiance Models (Section 5) / 태양 복사 모델

#### NRLEUV (Section 5.1) / NRLEUV 모델

물리 모델 (Warren, Mariska, Lean 2001; Warren 2005). 4 컴포넌트(coronal hole, quiet Sun, active network, active region) DEM에서 시작:

$$I_{\rm line}(\lambda) = \frac{1}{4\pi} \int G(T, n_e, \lambda) \, \xi(T) \, dT$$

$\xi(T) = n_e^2 \, dV/dT$는 Skylab spectroheliogram 으로부터 도출. CHIANTI/atomic database 와 결합하여 광학적으로 얇은 라인을 계산. 광학적으로 두꺼운 라인은 관측값 사용. 전체 디스크는 image 기반 fractional coverage + limb-brightening curve로 적분. SDO 시대에는 AIA 광범위 온도 커버리지(13.1 nm Fe XX/XXIII, 9.4 nm Fe XVIII)로 DEM을 매 픽셀 매 시간 cadence로 계산하여 NRLEUV를 비약적으로 향상시킬 예정.

NRLEUV is a physics-based 4-component model (coronal hole, quiet Sun, active network, active region). DEM $\xi(T) = n_e^2 \, dV/dT$ from Skylab data + CHIANTI gives optically thin lines; observed values used for thick lines. Full-disk irradiance integrates with image-based fractional coverage and limb-brightening curves. SDO/AIA enables per-pixel, per-cadence DEMs.

#### FISM (Section 5.2) / FISM 모델

경험 모델 (Chamberlin, Woods, Eparvier 2007, 2008). 0.1–190 nm, 1 nm 분해능, 1 분 cadence:

$$E_{\rm FISM}(\lambda, t) = E_{\rm QS}(\lambda) + C_{\rm SR}(\lambda) P_{\rm SR}(t) + C_{\rm SC}(\lambda) P_{\rm SC}(t) + C_{\rm flare}(\lambda) P_{\rm flare}(t)$$

- Daily component: TIMED/SEE (0.1–119 nm) + UARS/SOLSTICE (119–190 nm).
- Solar-rotation: Mg II core-to-wing index.
- Solar-cycle: F10.7 또는 SEE composite.
- Flare: TIMED/SEE 30 flares (11 임펄시브 + 19 점진).

EVE는 FISM의 베이스 데이터셋을 100% duty cycle (TIMED/SEE는 3%) 로 확장하고 0.1 nm 분해능으로 향상시킬 예정.

FISM is empirical, 0.1–190 nm at 1 nm/1 min, with quiet-Sun + solar-rotation (Mg II proxy) + solar-cycle (F10.7) + flare components based on 30 SEE flares (11 impulsive + 19 gradual). EVE extends FISM with 100 % duty cycle and 0.1 nm resolution.

#### SIP / SOLAR2000 / SFLR (Section 5.3) / SIP 시스템

SET가 운용하는 하이브리드 시스템. SIP (Solar Irradiance Platform) → SOLAR2000 (long-term reference) + SFLR/SOLARFLARE (real-time flare via GOES/XRS → Mewe model effective temperature → 0.1 nm spectrum, 0.05–30 nm, 2 min cadence) + SOLAR2000 (>30 nm).

SET-operated hybrid: SOLAR2000 (long-term) + SFLR (GOES/XRS → Mewe effective temperature → 0.1 nm, 0.05–30 nm, 2 min). EVE provides correction reference spectra.

### Part VI: Earth's Atmospheric Models (Section 6) / 지구 대기 모델

EVE의 네 번째 목표(지구 환경 응답)를 위한 모델 협력자.

- **CTIPe** (NOAA/CIRES, Fuller-Rowell): 결합 열권/이온층/플라즈마권 전기역학 3D 물리 모델. 운영 응답 시 EUV 입력을 매 cadence로 받음.
- **TDIM/IFM/GAIM** (Utah State, Sojka/Schunk): F-region 이온층 모델 + Kalman 필터 데이터 동화 (GAIM).
- **NRLMSIS** (Picone et al. 2002): 경험 상부 대기 모델, F10.7 + 81-day 평균 + Ap. EUV 직접 사용으로 진화 중 (Emmert, Picone, Meier 2008).
- **JB2006/JB2008** (Bowman et al. 2008a,b): 향상된 Jacchia 기반 밀도 모델, F10.7 + S10.7 + M10.7 + Y10.7 솔라 인덱스. JB2008은 Lyman-α + X-ray 추가, Dst 기반 지자기 효과.

CTIPe (NOAA): 3D thermosphere/ionosphere/plasmasphere physics-based, EUV input every cadence. TDIM/IFM/GAIM (USU): F-region ionosphere with Kalman-filter assimilation. NRLMSIS: empirical density model evolving toward direct EUV input. JB2006/2008 (Bowman): Jacchia-based with F10.7/S10.7/M10.7/Y10.7 indices; JB2008 adds Lyman-α and X-ray.

### Part VII: Summary (Section 7) / 요약

EVE는 0.1–105 nm 범위 EUV 스펙트럼을 비약적인 분광·시간 분해능과 정확도로 측정하여 (i) 우주 기상 운영, (ii) 태양·대기 물리 연구, (iii) 플레어와 그 지구 영향 연구를 지원한다. 사전 비행 보정은 2007년 GSFC에서 완료, 측정 요구사항을 모두 충족했다. 임무 launch는 2010년 2월 (또는 그 이후), 5년 nominal mission.

EVE measures 0.1–105 nm EUV spectral irradiance with unprecedented resolution and accuracy, supporting space-weather operations, solar/atmospheric research, and flare studies. Pre-flight calibration completed at GSFC in 2007; nominal 5-year mission from February 2010.

---

## 3. Key Takeaways / 핵심 시사점

1. **Five complementary detector channels span 0.1–105 nm with 0.1 nm spectral resolution / 다섯 검출기 채널이 0.1–105 nm를 0.1 nm 분해능으로 커버한다** — MEGS-A1 (5–18 nm), MEGS-A2 (17–37 nm), MEGS-B (35–105 nm), MEGS-SAM (0.1–7 nm photon counting), MEGS-P (121.6 nm), ESP (0.1–39 nm broadband). 이전 미션(SOHO/SEM 4채널, TIMED/SEE 1 nm 일별)에 비해 분광·시간 능력 모두 큰 도약. EVE eliminates the historical trade-off between spectral resolution and temporal cadence by combining grazing-incidence, normal-incidence, pinhole, and photometer technologies.

2. **10 s cadence resolves flare evolution across temperature / 10초 cadence로 플레어의 온도별 진화 추적 가능** — Fe XX/XXIII (impulsive, $T \sim 10^7$ K), Fe XVI/XV (gradual, $T \sim 10^{6.4}$ K), He II (chromospheric, $T \sim 10^{4.7}$ K) 라인이 모두 10초 단위로 분리 측정된다. 이는 플레어 EUV late phase (Woods et al. 2011) 같은 발견을 가능케 했다. The 10-second cadence resolves the differential evolution of EUV lines spanning $10^4$–$10^7$ K, enabling discoveries such as the EUV late phase of flares.

3. **Off-Rowland-circle grazing incidence is a sensitivity breakthrough / Off-Rowland-circle grazing incidence는 감도 혁신** — 전통적 Rowland circle 디자인은 CCD를 grazing 입사각에 배치하여 효율을 손상시킨다. EVE의 off-Rowland 설계는 CCD를 normal-incidence 가까이로 이동시켜 photon detection efficiency를 극대화 (Crotser et al. 2007). MEGS-A's off-Rowland design places the CCD near normal incidence while preserving grazing-incidence dispersion benefits, dramatically improving sensitivity.

4. **MEGS-SAM combines pinhole imaging + X-ray photon counting / MEGS-SAM은 핀홀 이미징과 X선 광자 계수 결합** — 분광기 없이 0.1–7 nm 스펙트럼과 XUV 이미지를 동시 제공. 단일 광자 사건의 전하량 ∝ 에너지 → 파장. CCD 픽셀 위치 → 태양 디스크 위치. SAM is a clever hybrid: a 26 μm pinhole projects an XUV solar image onto the MEGS-A CCD corner, where single-photon charge magnitudes give wavelength (~1 nm) and pixel positions give source location.

5. **15-minute latency Level 0C product enables operational space weather / 15분 지연 Level 0C 제품으로 운영 우주 기상 가능** — ASCII 형식, 6–105 nm at 0.1 nm, 1분 평균. NOAA SWPC, USAF, ESA의 위성 항력·GPS·HF 통신 모델에 직접 입력. The 0CS variant (~1 min latency) feeds ionospheric forecast models (GAIM, IFM, CTIPe). EVE's <15-min-latency Level 0C product (1 min cadence, 0.1 nm spectra) is the first operational space-weather EUV product, enabling near-real-time atmospheric model forcing.

6. **Two-stage modeling ecosystem: NRLEUV (physics) + FISM (empirical) / 두 층 모델 생태계: NRLEUV(물리) + FISM(경험)** — NRLEUV는 4-component DEM과 SDO/AIA 영상을 결합해 자기장 → 라인 강도 인과를 추적. FISM은 EVE 측정을 통계적으로 표현해 우주 기상 작업에 즉시 사용 가능한 1 nm/1 min 스펙트럼 제공. 두 모델은 EVE 데이터로 보정되며 상호 보완. NRLEUV provides physical understanding (DEM-based line synthesis); FISM provides operational outputs (1 nm/1 min); EVE data calibrates both.

7. **EVE establishes the SDO triad with HMI and AIA / EVE는 HMI·AIA와 함께 SDO 트라이어드를 형성** — HMI는 광구 자기장 (PHOTOSPHERE), AIA는 코로나 영상 (CORONA), EVE는 디스크 통합 EUV 스펙트럼 (IRRADIANCE). 세 기기 동시 관측은 자기 플럭스 → 코로나 가열 → EUV 출력 → 지구 대기 응답 의 인과 사슬을 처음으로 정량화 가능하게 한다. Together with HMI (magnetograms) and AIA (coronal images), EVE closes the chain magnetic field → coronal heating → EUV irradiance → atmospheric response, enabling end-to-end Sun–Earth modeling.

8. **20% accuracy improves to 10% at key wavelengths / 20% 정확도가 핵심 파장에서 10%로 개선** — Pre-flight calibration (NIST SURF beam line, 2007 GSFC) + annual sounding-rocket underflights (NASA 36-series). Sounding rocket = 동일 EVE 프로토타입 기기를 동일 보정 표준에 노출 → 우주 EVE의 sensitivity drift 보정. 이는 SOHO/SEM과 TIMED/SEE 시대의 표준 ~20% 정확도를 능가한다. EVE achieves ~20 % absolute accuracy across 0.1–105 nm and ~10 % at key wavelengths, supported by NIST SURF pre-flight calibration plus annual sounding-rocket underflights — exceeding SOHO/SEM and TIMED/SEE accuracies.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 분광기 광학 / Spectrograph optics

**Grating equation**:
$$d \, (\sin\alpha + \sin\beta) = m \, \lambda$$

For MEGS-A: $d = 1/767\,\mathrm{mm} = 1.304\,\mu\mathrm{m}$, $\alpha = 80°$, $m = 1$. At $\lambda = 30.4$ nm:

$$\sin\beta = \frac{\lambda}{d} - \sin\alpha = \frac{0.0304}{1.304} - \sin 80° = 0.0233 - 0.9848 = -0.9615$$

$\beta \approx -73.9°$ — physically diffracted on the same side as incidence (sign convention).

**Resolution from groove count**:
$$R = m \, N_{\rm illuminated}$$

For 0.1 nm at 30 nm: $R = 300$, requiring $N_{\rm illuminated} \approx 300$ grooves. With 767 grooves/mm and a typical illuminated width, MEGS-A meets this with margin.

**Rowland circle radius**: For MEGS-A, $R_{\rm grating} = 600$ mm; Rowland circle radius is $R_g/2 = 300$ mm. MEGS-B: $R_1 = 200$ mm, $R_2 = 200$ mm.

### 4.2 Spectral irradiance retrieval / 분광 복사 조도 검출

Per-pixel calibrated EUV irradiance at 1 AU:

$$E_\lambda(\lambda, t) = \frac{[C_{\rm pix}(t) - C_{\rm dark}(t)] \cdot G_{\rm CCD}}{A_{\rm eff}(\lambda) \cdot \Delta t \cdot \Delta\lambda} \cdot \frac{h c}{\lambda} \cdot \left(\frac{r(t)}{1\,\mathrm{AU}}\right)^2$$

- $C_{\rm pix}$: pixel digital number (DN). / 픽셀 카운트.
- $G_{\rm CCD}$: CCD gain ≈ 2 e⁻/DN (MEGS). / CCD 이득.
- $A_{\rm eff}$: 유효 면적 [cm²] = 슬릿 면적 × 격자 효율 × 필터 투과율 × CCD QE.
- $\Delta t = 10$ s.
- $\Delta\lambda = 0.02$ nm (MEGS pixel scale; 0.1 nm requires summing ~3–5 pixels).
- $hc/\lambda$: photon energy, e.g., at 30 nm, 41.3 eV = $6.62 \times 10^{-18}$ J.
- $(r/1\,\mathrm{AU})^2$: 1 AU 정규화. / 1 AU normalization.

### 4.3 Photon counting (MEGS-SAM) / 광자 계수

Si detector charge-energy relation: $E_{\rm photon} = w_{\rm Si} \cdot N_{e^-}$ with $w_{\rm Si} = 3.66$ eV/e⁻ at room temperature. For $\lambda = 1.0$ nm photon: $E = 1240$ eV, $N_{e^-} \approx 339$ — well above CCD read noise.

$$\lambda \;[\mathrm{nm}] = \frac{1240}{E_{\rm photon}\;[\mathrm{eV}]} = \frac{1240}{w_{\rm Si} \cdot N_{e^-}}$$

Histogram of pixel-event charges over 10 s integration → SAM XUV spectrum.

### 4.4 NRLEUV / DEM-based line synthesis

$$I_{\rm line}(\lambda_0) = \frac{1}{4\pi} \int_{T} G(T, n_e, \lambda_0) \; \xi(T) \; d T, \quad \xi(T) = n_e^2 \frac{dV}{dT} \;[\mathrm{cm}^{-5}\,\mathrm{K}^{-1}]$$

with contribution function $G(T) = (n_H/n_e)\,A_X\,(N_{\rm ion}/N_X)(T)\,(N_j/N_{\rm ion})(T,n_e)\,(A_{ji}\,\Delta E_{ji}/n_e)$ from CHIANTI.

Disk-integrated irradiance:

$$E_\lambda = \sum_{\rm component} f_c \cdot I_{\rm line, c}(\lambda) \cdot L(\theta) $$

where $f_c$ is fractional coverage of component $c$ (coronal hole, quiet Sun, active network, active region) and $L(\theta)$ is limb-brightening function.

### 4.5 FISM / Empirical model

$$E_{\rm FISM}(\lambda, t) = E_{\rm QS}(\lambda) + C_{\rm SR}(\lambda) \cdot P_{\rm SR}(t) + C_{\rm SC}(\lambda) \cdot P_{\rm SC}(t) + C_{\rm flare}(\lambda) \cdot P_{\rm flare}(t)$$

- $P_{\rm SC}(t) = F_{10.7}(t) - F_{10.7}^{\rm min}$: 태양 주기 대용. / Solar-cycle proxy.
- $P_{\rm SR}(t) = \mathrm{Mg\,II}(t) - \mathrm{Mg\,II}^{\rm avg}$: 자전 대용. / Rotation proxy.
- $P_{\rm flare}(t) = (\mathrm{XRS\text{-}B}(t) - \mathrm{XRS\text{-}B}_{\rm bg})^\alpha$: 플레어 대용. / Flare proxy.
- 계수 $C_*(\lambda)$는 wavelength-binned 회귀로 결정. / Coefficients fit per wavelength bin.

### 4.6 Quad-diode flare localization (ESP) / Quad-diode 위치 결정

Four quadrants $Q_1, Q_2, Q_3, Q_4$. Total signal $S = \sum Q_i$. Pointing offsets:

$$\Delta x = \frac{(Q_1 + Q_4) - (Q_2 + Q_3)}{S}, \quad \Delta y = \frac{(Q_1 + Q_2) - (Q_3 + Q_4)}{S}$$

ESP의 0차 quad는 ±1차 채널의 추가 보정 정보(시야 흔들림, 플레어 위치) 제공.

### 4.7 Worked example: MEGS-B count rate at 30.4 nm / 작업 예제

He II 30.4 nm quiet-Sun irradiance ≈ 4 mW m⁻² nm⁻¹ = $4 \times 10^{-3}$ W m⁻² nm⁻¹. Photon energy $E_\gamma = hc/\lambda = (1240/30.4)$ eV = 40.8 eV = $6.53 \times 10^{-18}$ J. Photon flux per nm:

$$\Phi_\gamma = \frac{4 \times 10^{-3}}{6.53 \times 10^{-18}} = 6.1 \times 10^{14}\,\mathrm{photons\,m^{-2}\,s^{-1}\,nm^{-1}}$$

For 10 cm² effective collecting area × 0.02 nm pixel × 10 s integration:

$$N_\gamma = 6.1 \times 10^{14} \times 10^{-3} \times 0.02 \times 10 = 1.2 \times 10^{11}$$

photons collected per pixel per integration. Even with 1% net efficiency this gives ~$10^9$ DN, ample SNR.

### 4.8 Effective area chain / 유효 면적 사슬

$$A_{\rm eff}(\lambda) = A_{\rm aperture} \cdot T_{\rm filter}(\lambda) \cdot \eta_{\rm grating}(\lambda) \cdot \mathrm{QE}_{\rm CCD}(\lambda)$$

Each factor calibrated separately at NIST SURF; total uncertainty in quadrature gives ~10–20 %.

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1979──1991──1996──2002──2003──2010──2014──2016──2024
  │     │     │     │     │     │     │     │     │
AE-E  UARS  SOHO  TIMED Hinode SDO  IRIS  GOES- PUNCH/
SEUM  SOLST SEM/  /SEE  EIS    /EVE        R     MUSE
            EIT                /AIA       /EXIS  era
(daily) (FUV) (broad) (1nm)  (0.1nm)
                                10s
```

- **1979 — AE-E SEUM**: 초기 daily EUV 측정 (~10 nm 분해능). / Early daily EUV.
- **1991 — UARS SOLSTICE**: FUV (119–420 nm) 측정 시작. / FUV measurements.
- **1996 — SOHO SEM**: 4채널 광대역 EUV (Judge et al. 1998), 15 s cadence. / 4-channel broadband.
- **2002 — TIMED SEE**: 0.1–193 nm, 1 nm, daily averages, 3 % duty cycle. / 1 nm daily.
- **2003 — Hinode/EIS**: EUV 분광 영상 (170–290 Å). / EUV imaging spectrograph.
- **2010 — SDO/EVE**: 0.1–105 nm, 0.1 nm, 10 s, 100 % duty cycle. / **THIS PAPER**.
- **2014 — IRIS**: 채층/전이층 분광 (1331–1407 Å, 2783–2834 Å). / Chromospheric spectroscopy.
- **2016 — GOES-R/EXIS**: 운영 EUV 후속 (Eparvier et al. 2009 design). / Operational EUV continuation.
- **2024 — PUNCH, MUSE era**: 차세대 코로나/EUV 분광 영상. / Next-gen imaging spectrographs.

EVE marks the transition from low-cadence/low-resolution irradiance monitors to a Sun-as-a-star high-resolution spectrograph that simultaneously serves operational space weather and solar physics research.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Pesnell et al. 2012 (SDO mission) | EVE는 SDO의 세 기기 중 하나 / EVE is one of SDO's three instruments | High — SDO 미션 컨텍스트 |
| Lemen et al. 2012 (AIA, paper #12) | AIA EUV 영상은 EVE 분광과 동시 측정 / AIA imaging complements EVE spectra | Very High — DEM 계산에 직접 입력 |
| Scherrer et al. 2012 (HMI, paper #13) | HMI 자기장 → EVE 복사 인과 / HMI magnetograms drive EVE irradiance | High — Sun-Earth chain |
| Hock et al. 2010 (MEGS calibration) | MEGS의 사전 비행 보정 상세 / Detailed MEGS pre-flight calibration | Critical — 보정 세부 |
| Didkovsky et al. 2010 (ESP calibration) | ESP의 사전 비행 보정 상세 / Detailed ESP pre-flight calibration | Critical — 보정 세부 |
| Chamberlin, Woods, Eparvier 2007/2008 (FISM) | FISM 모델은 EVE의 1 nm/1 min 운용 산출 / FISM is EVE's operational empirical model | Very High — 운용 모델 |
| Warren, Mariska, Lean 2001 (NRLEUV) | NRLEUV는 EVE의 물리 기반 모델 / NRLEUV is EVE's physics-based model | Very High — 물리 모델 |
| Judge et al. 1998 (SOHO/SEM) | EVE는 SEM의 직접 후속 / EVE directly extends SEM | High — 측정 연속성 |
| Woods et al. 2005 (TIMED/SEE) | EVE는 SEE의 분광·시간 분해능 향상 후속 / EVE supersedes SEE in spectral/temporal resolution | High — 측정 연속성 |
| Crotser et al. 2007 (off-Rowland) | MEGS-A의 핵심 광학 혁신 / Core optical innovation of MEGS-A | Critical — 기기 설계 |
| Westhoff et al. 2007 (CCD) | EVE CCD 검출기 사양 / EVE CCD detector | Critical — 검출기 |
| Bowman et al. 2008a,b (JB2006/2008) | JB 모델은 EVE 데이터를 사용 / JB models consume EVE data | High — 운용 응용 |
| Picone et al. 2002 (NRLMSIS) | NRLMSIS는 EVE 측정으로 향상 / NRLMSIS evolves with EVE input | High — 대기 모델 |
| Woods et al. 2011 (EUV late phase) | EVE 발견 사례 / Hallmark EVE discovery (post-publication) | Very High — 과학 결과 |

---

## 7. References / 참고문헌

- Woods, T.N., Eparvier, F.G., Hock, R., et al., "Extreme Ultraviolet Variability Experiment (EVE) on the Solar Dynamics Observatory (SDO): Overview of Science Objectives, Instrument Design, Data Products, and Model Developments", *Solar Physics* **275**, 115–143, 2012. DOI: 10.1007/s11207-009-9487-6.
- Pesnell, W.D., Thompson, B.J., Chamberlin, P.C., "The Solar Dynamics Observatory (SDO)", *Solar Physics* **275**, 3–15, 2012.
- Lemen, J.R., Title, A.M., Akin, D.J., et al., "The Atmospheric Imaging Assembly (AIA) on the Solar Dynamics Observatory (SDO)", *Solar Physics* **275**, 17–40, 2012.
- Scherrer, P.H., Schou, J., Bush, R.I., et al., "The Helioseismic and Magnetic Imager (HMI) Investigation for the Solar Dynamics Observatory (SDO)", *Solar Physics* **275**, 207–227, 2012.
- Hock, R.A., Chamberlin, P.C., Woods, T.N., Crotser, D., Eparvier, F.G., Furst, M., Triplett, M.A., Woodraska, D.L., "Extreme Ultraviolet Variability Experiment (EVE) Multiple EUV Grating Spectrographs (MEGS): Radiometric Calibrations and Results", *Solar Physics* (in press, 2010). DOI: 10.1007/s11207-010-9520-9.
- Didkovsky, L., Judge, D., Wieman, S., Woods, T., Jones, A., "EUV SpectroPhotometer (ESP) in Extreme Ultraviolet Variability Experiment (EVE): Algorithms and Calibrations", *Solar Physics* (in press, 2010). DOI: 10.1007/s11207-009-9485-8.
- Crotser, D.A., Woods, T.N., Eparvier, F.G., Ucker, G., Kohnert, R.A., Berthiaume, G., Weitz, D., "MEGS-A and EVE Science", *SPIE Proc.* **6689**, 66890M, 2007.
- Westhoff, R.C., Rose, M.K., Gregory, J.A., et al., "Backside-Illuminated CCDs for SDO EVE", *SPIE Proc.* **6686**, 668604, 2007.
- Judge, D.L., McMullin, D.R., Ogawa, H.S., et al., "First Solar EUV Irradiances Obtained from SOHO by the CELIAS/SEM", *Solar Phys.* **177**, 161–173, 1998.
- Woods, T.N., Eparvier, F.G., Bailey, S.M., Chamberlin, P.C., Lean, J., Rottman, G.J., Solomon, S.C., Tobiska, W.K., Woodraska, D.L., "Solar EUV Experiment (SEE): Mission overview and first results", *J. Geophys. Res.* **110**, A01312, 2005.
- Warren, H.P., Mariska, J.T., Lean, J., "A new model of solar EUV irradiance variability. 1. Model formulation", *J. Geophys. Res.* **106**, 15745, 2001.
- Chamberlin, P.C., Woods, T.N., Eparvier, F.G., "Flare Irradiance Spectral Model (FISM): Daily component algorithms and results", *Space Weather* **5**, S07005, 2007.
- Chamberlin, P.C., Woods, T.N., Eparvier, F.G., "Flare Irradiance Spectral Model (FISM): Flare component algorithms and results", *Space Weather* **6**, S05001, 2008.
- Bowman, B.R., Tobiska, W.K., Marcos, F.A., Valladares, C., "The JB2006 empirical thermospheric density model", *J. Atmos. Sol. Terr. Phys.* **70**, 774, 2008a.
- Picone, J.M., Hedin, A.E., Drob, D.P., Aikin, A.C., "NRLMSISE-00 empirical model of the atmosphere", *J. Geophys. Res.* **107**(A12), 1468, 2002.
- Woods, T.N., Hock, R., Eparvier, F., et al., "New solar extreme-ultraviolet irradiance observations during flares", *Astrophys. J.* **739**, 59, 2011 (post-publication; demonstrates EUV late phase).
