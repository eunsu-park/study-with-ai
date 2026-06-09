---
title: "The Solar Orbiter SPICE Instrument: An Extreme UV Imaging Spectrometer"
authors: SPICE Consortium (M. Anderson, T. Appourchaux, F. Auchère et al.)
year: 2020
journal: "Astronomy & Astrophysics, Vol. 642, A14"
doi: "10.1051/0004-6361/201935574"
topic: Solar_Observation
tags: [solar_orbiter, spice, euv_spectroscopy, imaging_spectrometer, coronal_diagnostics, fip_effect, instrument_paper]
status: completed
date_started: 2026-04-17
date_completed: 2026-04-17
---

# 18. The Solar Orbiter SPICE Instrument / Solar Orbiter SPICE 기기

---

## 1. Core Contribution / 핵심 기여

**한국어**
SPICE(Spectral Imaging of the Coronal Environment)는 ESA/NASA **Solar Orbiter** 미션에 탑재된 극자외선(EUV) 영상 분광기이다. 본 논문은 SPICE의 **과학 목표 → 광학·기계·열·전자부 설계 → 발사 전 성능·보정 → 운영 개념 → 데이터 처리** 를 총망라한 공식 기기 논문(instrument paper)이다. 핵심 기여는 세 가지로 요약된다. 첫째, **0.28 AU 근일점(~13× 태양 상수, 17 kW/m² 열부하)** 이라는 전례 없는 환경에서 작동하는 광학 설계로서, **off-axis parabola 단일 거울 + Toroidal Variable Line Space(TVLS) 그레이팅** 의 2-미러 광학을 채택하고, **boron carbide(B₄C, 10 nm) 이색성(dichroic) 코팅** 으로 UV/VIS/IR 스펙트럼의 ~70%를 투과·방열시켜 기기 내부 열부하를 31.7 W로 제한한다. 둘째, **채층(~10 000 K, H I Lyβ)부터 플레어 코로나(~10 MK, Fe XX)까지** 의 온도대를 한 번의 관측으로 동시 커버하는 스펙트럼 라인 세트(Table 1, 19개 라인)를 제공하며, 이는 기존 SUMER(파장 stepping 필요)·EIS(뜨거운 코로나 편중)·IRIS(저온 편중)의 공백을 메운다. 셋째, Solar Orbiter의 **out-of-ecliptic 궤도 경사(>30°)** 를 활용하여 **태양 극지의 EUV 분광 최초 관측** 과 **FIP-bias 지도 제작** 으로 태양풍의 소스 영역을 in-situ 입자 관측(SWA, MAG, EPD)과 직접 연결하는 "tracer" 역할을 수행한다. 기기의 발사 전 성능은 설계 요구(공간분해능 4″×2 pixels, 분광분해능 0.04 nm FWHM, 도플러 정확도 ~5 km/s)를 모두 충족함이 지상 보정에서 확인되었다.

**English**
SPICE (Spectral Imaging of the Coronal Environment) is an extreme ultraviolet imaging spectrometer onboard the ESA/NASA **Solar Orbiter** mission. This instrument paper comprehensively describes SPICE's **science objectives, optical/mechanical/thermal/electronics design, pre-launch performance and calibration, operations concept, and data processing**. Three key contributions stand out. First, the optical design operates under the extreme environment at **0.28 AU perihelion (~13× solar constant, 17 kW/m² heat load)** using a minimalist two-reflection chain: an **off-axis parabola telescope** feeding a **Toroidal Variable Line Space (TVLS) grating**, with a **boron carbide (B₄C, 10 nm) dichroic coating** that dumps ~70% of the solar UV/VIS/IR flux to space and limits the internal heat load to 31.7 W. Second, SPICE's carefully selected line list (Table 1, 19 lines) simultaneously samples the temperature range from the **chromosphere (~10 000 K, H I Lyβ) to flaring corona (~10 MK, Fe XX)** in a single observation — closing the gap left by SUMER (wavelength-stepping), EIS (hot-corona biased), and IRIS (cool-biased). Third, by exploiting Solar Orbiter's **>30° out-of-ecliptic inclination**, SPICE will deliver the **first-ever EUV spectroscopy of the solar polar regions** and produce **FIP-bias maps** that trace solar-wind source regions in conjunction with in-situ instruments (SWA, MAG, EPD). Ground calibration demonstrated that SPICE meets all its science-driving requirements: ~4″×2 pixel spatial resolution, 0.04 nm FWHM spectral resolution, and ~5 km/s Doppler velocity accuracy.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction & Scientific Objectives (§1–2) / 서론과 과학 목표

**한국어 — §1 Introduction (p. 1)**
Solar Orbiter는 2020년 2월 발사된 ESA-NASA 공동 미션으로, Helios 미션(1974)을 크게 개선하여 **근일점 0.28 AU + 궤도 경사 30°** 라는 전례 없는 관측 조건을 제공한다. 10개 기기 중 SPICE는 원격관측(remote-sensing)군에 속하며, 총 **45.3 Gbits/6개월 텔레메트리 버짓** 내에서 동작한다. 임무의 중심 과학 질문: **"태양은 헬리오스피어를 어떻게 창조하고 제어하는가?"**

SPICE가 답하려는 구체적 문제:
- 태양풍의 소스 영역과 가속 메커니즘 / Sources and acceleration of solar wind
- CME의 원인과 태양풍과의 상호작용 / Causes/evolution of CMEs
- 태양 에너지 입자(SEP)의 기원과 가속 / Origin/acceleration of SEPs
- 태양 내부 → 광구 → 헬리오스피어로 이어지는 물질·자기장 흐름 / Plasma & magnetic field flow from surface to heliosphere

**§2 Scientific objectives and opportunities (p. 2–4)**
§2.1에서 SPICE의 관측량(Observables)을 설명: **선택된 EUV 라인의 강도 + 라인 프로파일** 로부터 **온도, 밀도, 유출속도(Doppler), 원소 조성, 난류 상태** 를 도출한다. Table 1이 핵심:

| Ion | λ (Å) | log T [K] | Intensity QS | Intensity AR |
|---|---|---|---|---|
| H I | 1025.72 | 4.0 | 372.2 | 883.5 |
| C III | 977.03 | 4.5 | 312.1 | 563.7 |
| O VI | 1031.93 | 5.5 | 139.0 | 8268.2 |
| Ne VIII | 770.42 | 5.8 | 7.8 | 63.9 |
| Mg IX | 706.02 | 5.9 | 0.9 | 3.0 |
| Si XII | 520.67† | 6.3 | 2.5 | 31.2 |
| Fe XVIII | 974.84 | 6.9 | — | 6.9 |
| Fe XX | 721.55 | 7.0 | — | 1428.2 (M7.6 플레어) |

(★ 표시는 완전 라인 프로파일이 반환되는 강선, † 표시는 2차 회절에서 관측)

**중요 비교**:
- **EIS**: >1 MK만 강력 / mostly >1 MK
- **IRIS**: 0.3–8 MK / 0.3–8 MK
- **SUMER**: 넓은 범위이나 파장 stepping 필요 / wide range but required wavelength stepping
- **SPICE**: 10 000 K – 10 MK **동시** 관측 / **simultaneous** 10 000 K – 10 MK coverage

§2.2에서 SPICE와 다른 Solar Orbiter 기기들의 협력 전략:
- **SPICE + SWA**: 원격 FIP-bias 지도 ↔ in-situ 태양풍 조성 ⇒ 소스 영역 연결
- **SPICE + PHI**: 자기장 ↔ 상층 대기 응답 ⇒ 가열 메커니즘
- **SPICE + EUI**: 채층/TR 동역학과 코로나 이미지 결합

### Part II: Instrument Overview & Optical Design (§3–4) / 기기 개요와 광학 설계

**§3 Instrument overview (p. 4)**
SPICE의 광학 사슬은 **단 2개의 반사면**으로 구성된다:
1. **단일 거울 오프-축 포물선 망원경(off-axis paraboloid, 거의 수직 입사)**: 태양 이미지를 슬릿에 형성
2. **TVLS 회절격자**: 슬릿을 분산·확대하여 두 검출기 어레이에 재결상

왜 2 반사인가? 각 반사면의 EUV 반사율이 ~30%에 불과하므로, 반사 횟수를 최소화해야 신호가 남는다.

**4가지 기계 메커니즘 / Four mechanisms**:
- SDM (Slit Door Mechanism): 오염 방지용 입구 도어
- Scan Focus Mechanism (SFM): 거울의 틸트(스캔)와 초점 조정
- SCM (Slit Change Mechanism): 2″, 4″, 6″, 30″ 4개 슬릿 교체
- Detector Assembly (DA) door: 검출기용 진공 도어

**§4 Optical design (p. 5–8)**

*4.1 Imaging resolution (p. 5–6)*
목표 성능: **LSF FWHM ~4 pixels** (Table 3)
- 설계 수차 기여: 2 pixels
- 광학 부품 공차: 2.5 pixels
- 위치 공차: 1.5 pixels
- 검출기 PSF FWHM: 2 pixels
- 우주선 jitter(10 s): 1 pixel
- **RSS 합: 4.2 pixels** (요구 충족)

회절격자 방정식:
$$
\sin(\theta_m) = m \cdot \frac{\lambda}{d} + \sin(\theta_i)
$$

SPICE 값:
- $m = 1$ (주), 2 (LW의 단파장)
- $d = 1/2400$ mm
- $\theta_i = -1.7584°$, $\theta_m = +8.5498°$ (74.7 nm), $+12.24°$ (101.12 nm)
- 분산 (dispersion): 0.0095 nm/pixel @ 74 nm, 0.0083 nm/pixel @ 101 nm

*4.2 Telescope mirror (p. 6–7)*
- 직경 95 × 95 mm, 초점거리 622 mm, f/14.30
- 기판: UV-grade fused silica (Suprasil 300)
- 반사 코팅: **B₄C 10 nm, dichroic** (EUV 30% 반사, 그 외 투과)
- 표면조도: <0.2 nm rms (AFM 측정 0.17 nm)
- 도형 오차: <λ/20 rms @ 633 nm (측정: 0.028 waves)
- **입자 편향기(particle deflector)**: -2.5 kV로 하전 입자를 걸러내어 거울 보호

*4.3 Spectrometer slits (p. 7)*
- 4개 슬릿을 single carriage에 일렬 배치
- 각 슬릿은 0.5 mm 실리콘에 vee-groove 에칭, 금 코팅
- 3개 좁은 슬릿(2″, 4″, 6″) + 1개 넓은 30″ 슬릿(slit이 아닌 "slot")
- **Dumbbell aperture**: 좁은 슬릿 양끝에 0.5′×0.5′ 구멍 → pointing 정보

*4.4 Diffraction grating (p. 7)*
- **TVLS (Toroidal Variable Line Space)**: 격자선 간격이 표면에 따라 ~1% chirp
- 2400 lines/mm, 홀로그래픽 제작
- 동일한 B₄C 코팅이지만 두께 20 nm (반사율 증가, 열부하 작아 가능)
- 회절효율: ~9% (싱크로트론 측정)

### Part III: Mechanical, Thermal & Electronics Design (§5–7) / 기계·열·전자부 설계

**§5 Mechanical and thermal design (p. 8–10)**

*5.1 Mechanical (p. 8)*
- SOU (SPICE Optics Unit): CFRP+Al honeycomb, 총 질량 ~13 kg, 1100×350×280 mm
- 3점 quasi-kinematic mount: 1개 고정, 2개 flexure blade — spacecraft와의 CTE 차이(~1 mm) 흡수
- Resonance frequency: ≥140 Hz 요구 → 실측 224 Hz

*5.2 Thermal (p. 9 + Fig. 9)*
**열 에너지 흐름 (Fig. 9, EOL perihelion)**:
- 입사: 31.7 W (52×52 mm 개구를 통해)
- 주거울로 반사 → 우주로 방열: 66% = 20.9 W
- SOU 내부 흡수: 10.7 W
  - Heat dump radiator로: 2.5 W (우주로 2.5 W)
  - SOU 구조 흡수: 12.1 W
  - Heaters: 1.1 W
- CE (cold element) 인터페이스: 7.2 W
- Spacecraft 온도: 50°C, SPICE 구조 평균: 55°C

주거울 자체는 ~70°C로 주변보다 뜨겁다 (초기 cold phase에서 오염 방지). 검출기는 heater로 **-20°C**에 유지.

**§6 Mechanisms and detector assembly (p. 9–12)**

*6.5 Detector Assembly (p. 12)*
- 2개의 동일한 APS 카메라 (SW, LW)
- 각각: 1024×1024 pixels HAS2 CMOS APS + KBr 광전면 + MCP intensifier
- 작동: EUV 광자 → KBr 광전면에서 광전자 방출 → MCP에서 증폭 → 형광막 발광 → 광섬유 → APS에서 디지털화
- 판독 소음: Correlated Double Sampling (CDS)으로 kTC 잡음 제거, 14-bit 디지털화

**§7 Electronics (p. 12–14)**

*7.1 SEB (SPICE Electronics Box)*: 6개 보드(HVPS, DPM, 2× MIM, LVPS, Backplane)

*7.3 DPM*: 8051 마이크로컨트롤러 + FPGA. 모든 메커니즘 제어, FEE 제어, 이미지 처리/압축

*7.5 FSW*: C + 어셈블리. 4개 모드: **Startup/Standby/Engineering/Operate** (Fig. 15)

*7.9 Compression*: **Spectral Hybrid Compression (SHC)** — 스펙트럼축으로 FFT → Fourier 계수들을 wavelet으로 lossy 압축. 최대 **20:1 압축비** (논문 첫 등장, DeForest 2015 특허 기반).

### Part IV: Testing, Calibration, Performance (§8–9) / 시험과 보정

**§9 Characterisation and calibration (p. 19–22)**

*9.1 Detector QE (Table 4)*:
| λ (nm) | QE_SW (%) | QE_LW (%) |
|---|---|---|
| 49.0 | 22.8 | 22.2 |
| 73.7 | 8.7 | 8.8 |
| 83.4 | 17.2 | 17.5 |
| 104.8 | 24.9 | 21.9 |

최저 QE는 73.7 nm 근처 (~9%)

*9.2 Telescope optical tests*: 파면오차 WFE = **0.2 waves peak-to-valley @ 633 nm** (베스트 초점, FOV 중심). 초점 기구는 ±500 μm 범위.

*9.3 VUV tests*: 베를린 PTB hollow-cathode 아르곤 소스로 **3개 빔 위치에서 스펙트럼 이미지 획득** (Fig. 22). LSF FWHM 요구 4 픽셀을 FOV 대부분에서 충족, worst-case (FOV 가장자리) ~4.5 픽셀.

*9.4 Wavelength calibration (Table 5)*:
- SW 측정 범위: 69.7008–78.9280 nm, 분산 **0.009562 nm/pixel** (설계 0.009515, 일치)
- LW 측정 범위: 96.8783–104.919 nm, 분산 **0.008307 nm/pixel** (설계 0.0083)

온도에 따른 스펙트럼 이동:
- Cold 케이스(18.9°C): 장파장 쪽 ~10 픽셀 이동
- Warm 케이스(51.8°C): 단파장 쪽 ~15 픽셀 이동
→ 궤도 중 온도 변화에 따라 재보정 필요

*9.5 Photometric sensitivity*:

$$
N = \frac{L}{h\nu} A_\mathrm{ape} \Omega_S R(\lambda) t_\mathrm{exp}
$$

$R(\lambda)$는 기기 응답도(instrument responsivity):

$$
R(\lambda) = R_\mathrm{mir}(\lambda) \cdot \eta_\mathrm{gra}(\lambda) \cdot \mathrm{QDE}_\mathrm{det}(\lambda) \cdot g_\mathrm{det}
$$

- $R_\mathrm{mir}$: 거울 반사율
- $\eta_\mathrm{gra}$: 격자 절대 효율
- $\mathrm{QDE}_\mathrm{det}$: 검출기 양자검출효율
- $g_\mathrm{det}$: 검출기 이득(DN/광전자)

**유효 면적(effective area)** Fig. 24: 74 nm 근처 최소(~0.5 mm²), 100 nm 근처 최대(~10 mm²), 1차/2차 차수 모두 포함.

*9.6 SHC 검증*: Hinode/EIS 고품질 스펙트럼에 포아송 노이즈 추가 → SHC 압축 후 복원 → 원본과 비교. 결과:
1. 라인 강도: 광자계수 노이즈의 <25% 이내 재현
2. 도플러 시프트: ≤0.1 픽셀 rms
3. 라인 폭: ≤0.2 픽셀 rms
4. 사이드로브 강도비: ≤5%

### Part V: Operations & Data Processing (§10–11) / 운영과 데이터 처리

**§10 Operations concept (p. 22–23)**
- **SOOP (Solar Orbiter Observing Program)** 단위로 계획 (궤도당 1회, SWT 승인)
- SOOP은 **BOP (Basic Observing Program)** 들로 구성
- BOP은 **study** 들을 실행 — study는 (λ, y, x, t) 데이터큐브 하나 획득
- Study 종류: full spectrum, raster (spatial scan), sit-and-stare (X=0), scanned time series
- On-board LUTs: 64개 study 저장 (그중 16개는 엔지니어링용)
- 계획 도구: Django/Python + Bootstrap/JQuery GUI

**§11 Data processing (p. 23–24)**
**3개 레벨**:
- **L1**: 보정되지 않은 데이터 (engineering units), 시간을 UTC로, 좌표를 Sun-centre로
- **L2**: 보정된 데이터 (물리 단위), flat-field·dark·geometric 보정 포함
- **L3**: 고수준 산출물 — 라인 강도, 속도, 폭, FIP bias 등 (자동 또는 수동 처리)

FITS 파일 구조: 4D 데이터큐브 **(X, Y, dispersion, time)**

SolarSoft (IDL) 기반 시각화·분석 도구 제공. JHelioviewer와 연동.

### Part VI: Summary (§12) / 요약

핵심: SPICE는 70.4–79.0 nm + 97.3–104.9 nm에서 **온도·밀도·유동·조성 진단**을 수행하며, Solar Orbiter의 원격↔in-situ 연결의 **유일한 도구**이다. 자기적으로 닫힌 영역에서 채층부터 가장 뜨거운 코로나까지의 난류 상태를 특성화하여 가열·동역학 메커니즘을 규명한다.

---

## 3. Key Takeaways / 핵심 시사점

1. **"온도 커버리지 × 동시성"이 SPICE의 유일 차별점이다 / Temperature coverage × simultaneity is SPICE's unique selling point** — 19개 EUV 라인을 통해 $\log T$ = 4.0 (H I)부터 $\log T$ = 7.0 (Fe XX)까지 연속적으로 샘플링하며, **한 번의 관측으로 동시에** 기록한다. SUMER는 stepping이 필요했고, EIS는 >1 MK, IRIS는 <8 MK에 국한되었다. 이로써 TR-corona 경계의 동적 현상(스파이큘, 나노플레어, 파동)을 **온도 단면으로** 추적할 수 있다.

2. **Dichroic B₄C 거울 + 2-반사 구조가 0.28 AU 운용을 가능하게 한다 / Dichroic B₄C mirror + two-reflection chain enables 0.28 AU operation** — 10 nm 두께의 B₄C 코팅은 EUV만 30% 반사하고 나머지 태양 플럭스(UV/VIS/IR ~70%)를 투과시켜 방열 거울(HRM)로 보내 우주로 돌려보낸다. 결과적으로 31.7 W의 입사 중 SOU 내부에는 10.7 W만 남는다. 또한 광학 사슬이 단 2반사(거울+격자)로 최소화되어 낮은 EUV 반사율(~30%)에서도 충분한 신호 수준을 유지한다.

3. **FIP-bias 지도는 태양풍의 "화학 추적자" 역할을 한다 / FIP-bias maps serve as the "chemical tracer" of solar wind** — 저-FIP 원소(Si, Mg, Fe)는 코로나·태양풍에서 ~4배 강화된다. SPICE는 저-FIP(Si, Mg, Fe)와 고-FIP(H, C, O, Ne) 라인을 모두 관측하여 **FIP bias 지도** 를 만들 수 있고, 이를 SWA가 측정한 in-situ 조성과 대조해 "이 태양풍 덩어리가 태양의 어디서 왔는가"를 추적할 수 있다. 이것이 Solar Orbiter의 "remote-sensing ↔ in-situ 연결" 전략의 핵심 메커니즘이다.

4. **TVLS 그레이팅은 단일 소자로 focusing + dispersion + aberration 보정을 수행한다 / TVLS grating performs focusing + dispersion + aberration correction in a single element** — 기존 near-normal-incidence 분광기는 색수차·비점수차 보정을 위해 별도의 거울이 필요했으나, toroidal 표면 + 격자선 ~1% chirp로 두 곡률 반경과 선간격 변화를 결합해 보정을 격자 내부로 흡수한다. EUV 대역 반사율이 낮아 반사면 최소화가 필수적이므로, TVLS는 SPICE의 photon budget을 유지하는 **결정적 설계 선택**이다.

5. **Spectral Hybrid Compression (SHC)이 telemetry 제약을 극복한다 / Spectral Hybrid Compression breaks the telemetry bottleneck** — Solar Orbiter의 궤도에 따라 데이터 하향 대역폭이 크게 변하므로, SPICE는 평균 17.5 kbit/s만 허용된다. SHC는 스펙트럼 방향으로 FFT → 주요 Fourier 계수만 wavelet lossy 압축하여 **최대 20:1** 압축을 달성하면서, 라인 진단에 필요한 핵심 파라미터(강도·중심·폭)는 무손실 계수에 보존한다. 검증 결과 도플러 ≤0.1 픽셀 rms, 선폭 ≤0.2 픽셀 rms로 과학 요구 이내.

6. **Dumbbell aperture가 포인팅/보정의 핵심 기준점 역할을 한다 / Dumbbell apertures serve as critical pointing/alignment references** — 좁은 슬릿(2″, 4″, 6″) 양끝의 0.5′×0.5′ 정사각 구멍은 태양의 작은 영역을 2차원 영상으로 기록하여, (1) 슬릿과 검출기의 롤 각도 측정, (2) 스캔 도중 포인팅 표류 추적, (3) 검출기 flat-field 위치 고정에 사용된다. 이는 SPICE처럼 **단일 차원 슬릿** 을 rastering으로 2D 만드는 분광기에서 시스템적 기하 오차를 억제하는 영리한 설계.

7. **Out-of-ecliptic 관측이 태양풍 기원 문제를 재정의한다 / Out-of-ecliptic viewing redefines the solar-wind-origin problem** — 모든 이전 EUV 분광기는 황도면(±7°)에서만 태양을 보았다. Solar Orbiter는 2025년 이후 30° 이상으로 경사되며, SPICE는 **인류 최초로 태양 극지의 EUV 분광** 을 수행한다. 극지 coronal hole이 고속 태양풍의 주 소스로 추정되어 왔지만, 지구 궤도에서는 edge-on으로밖에 볼 수 없었다. 이제 face-on 분광 관측이 가능해져 "빠른 바람이 어디서, 어떻게 시작되는가"라는 60년 된 문제에 직접적 증거를 제공한다.

8. **정량적으로 검증된 과학 요구-성능 매칭 / Quantitatively verified science-requirements match** — 지상 보정에서 설계 요구 대부분이 충족됨을 **숫자로** 입증: LSF FWHM = 4.2 pixels (요구 ≤4), 파장 보정 분산 0.009562 vs 설계 0.009515 nm/pixel (0.5% 오차), 유효 면적 ~10 mm² (100 nm 근처), QE 8.7–24.9% (파장별), 도플러 정확도 ~5 km/s. 이 수치들은 instrument paper가 단순 설계 기술서가 아니라 **science-driving performance budget의 정량적 종결문** 임을 보여준다.

---

## 4. Mathematical Summary / 수학적 요약

### (1) 회절격자 방정식 / Grating equation

$$
\sin(\theta_m) = m \cdot \frac{\lambda}{d} + \sin(\theta_i)
$$

- **의미 / Meaning**: 격자 간격 $d$가 파장 $\lambda$를 차수 $m$에 따라 다른 각도 $\theta_m$로 회절. SPICE의 1차 모드가 주. LW 검출기의 단파장 부분(48–53 nm)은 2차로 관측되며 SW와 섞이지 않음.
- **SPICE values**: $d = 1/2400$ mm, $\theta_i = -1.7584°$, $\theta_m$ = +8.5498° (SW), +12.2398° (LW).
- **분산(dispersion)**:
$$
\frac{d\lambda}{dx} = \frac{d\cos\theta_m}{m \cdot f_\mathrm{gra}} \cdot \mathrm{pixel\_size}
$$
  - 실측: 0.009562 nm/pixel (SW), 0.008307 nm/pixel (LW)

### (2) 도플러 속도 / Doppler velocity

$$
v_\mathrm{LOS} = c \cdot \frac{\lambda_\mathrm{obs} - \lambda_0}{\lambda_0}
$$

- **의미 / Meaning**: 라인 중심의 이동으로 시선방향 플라즈마 속도를 측정.
- **SPICE 정확도**: ~5 km/s (긴 파장에서 라인 centroid ~0.0028 nm 측정 정확도로부터 환산).
- **1 픽셀 해당 속도** (@ λ = 100 nm): $v = c \cdot (0.0083 \text{ nm} / 100 \text{ nm}) = 24.9$ km/s. 즉 5 km/s 정확도를 위해서는 서브픽셀 centroid 피팅 필요.

### (3) 선폭 분해 / Linewidth decomposition

$$
\sigma_\mathrm{obs}^2 = \underbrace{\sigma_\mathrm{inst}^2}_{\approx(\mathrm{LSF\_FWHM}/2.355)^2} + \frac{\lambda_0^2}{c^2}\left(\frac{2 k_B T_i}{M_i} + \xi^2\right)
$$

- **의미 / Meaning**: 관측 선폭을 기기·열·비열 성분으로 분리. SPICE의 LSF FWHM ≈ 4 pixels이므로 $\sigma_\mathrm{inst} \approx 1.7$ pixel.
- **활용**: $\xi$ (비열 속도)는 파동·난류 진단의 핵심 관측량.

### (4) Emission measure와 라인 강도 / Emission measure and line intensity

$$
I_\mathrm{line} = \frac{\mathrm{Ab}(X)}{4\pi} \int G(T, n_e) \, n_e n_H \, dh = \frac{\mathrm{Ab}(X)}{4\pi} \cdot \bar{G}(T) \cdot \mathrm{EM}
$$

$$
\mathrm{EM} = \int n_e n_H \, dh
$$

- **의미 / Meaning**: 라인 강도는 원소 존재량 × contribution function × emission measure에 비례. 여러 라인의 강도로부터 EM(T)과 Ab(X)를 분리해 **FIP bias 지도** 제작.
- **FIP bias**:
$$
\mathrm{FIP\_bias} = \frac{\mathrm{Ab}(X)_\mathrm{corona}}{\mathrm{Ab}(X)_\mathrm{photosphere}}
$$
  - 저-FIP 원소(Si, Mg, Fe): ~4 (corona), 1 (photosphere)
  - 고-FIP 원소(H, C, O, Ne): ~1 (변화 없음)

### (5) 태양 플럭스의 거리의존성 / Solar flux vs. heliocentric distance

$$
F(r) = F_\oplus \left(\frac{1 \text{ AU}}{r}\right)^2
$$

- **@ r = 0.28 AU**: $F = 12.76 \, F_\oplus \approx 17{,}400$ W/m²
- **@ r = 0.95 AU**: $F = 1.11 \, F_\oplus$ (거의 1 AU 수준)

### (6) SPICE photometric signal equation / 측광 신호식

$$
N = \frac{L}{h\nu} \cdot A_\mathrm{ape} \cdot \Omega_S \cdot R(\lambda) \cdot t_\mathrm{exp}
$$

- $N$: 픽셀당 검출된 DN (총합)
- $L$: 분광 라디언스(W m⁻² sr⁻¹) — 관측 대상의 EUV 밝기
- $h\nu$: 광자 에너지
- $A_\mathrm{ape}$ = 43.5 × 43.5 mm² = 1892.25 mm² (입구 개구)
- $\Omega_S$: 슬릿 단일 픽셀의 입체각
- $t_\mathrm{exp}$: 노출시간

기기 응답도 $R(\lambda)$:

$$
R(\lambda) = R_\mathrm{mir}(\lambda) \cdot \eta_\mathrm{gra}(\lambda) \cdot \mathrm{QDE}_\mathrm{det}(\lambda) \cdot g_\mathrm{det}
$$

**수치 예시 (O VI 103.2 nm, Active Region)**:
- Table 1: QS 기준 139 ph/pixel/s, AR 기준 8268 ph/pixel/s
- $R_\mathrm{mir}$ ≈ 0.3, $\eta_\mathrm{gra}$ ≈ 0.09, QDE ≈ 0.2
- 유효면적 $\approx A_\mathrm{ape} \cdot R_\mathrm{mir} \cdot \eta_\mathrm{gra} \cdot$ QDE ≈ 10 mm² (100 nm 근방, Fig. 24 일치)

### (7) TVLS 격자의 chirp 방정식 / TVLS grating chirp

일반 holographic grating line number density:

$$
d^{-1}(x) = d_0^{-1} \left[1 + \gamma \frac{x}{L}\right], \quad \gamma \approx 0.01
$$

- 여기서 $\gamma$는 "chirp" 파라미터, $L$은 격자 길이, $x$는 표면 위치.
- SPICE의 실제 chirp 제어 정밀도: ~5% (toroidal 곡률과 조합).

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1957 ───── Sputnik → 우주 시대 개막
1960s ──── 초기 로켓 EUV 분광 관측 (Parker 1958 태양풍 예측)
1974 ───── Helios 1 발사 (최초 0.3 AU 근접, EUV 분광기 없음)
1995 ───── SOHO 발사 (SUMER/CDS: 최초 지속 EUV 분광)
2006 ───── Hinode 발사 (EIS: 2″ 해상도, 뜨거운 코로나 편중)
2013 ───── Fludra et al.: SPICE 광학 설계 최초 공개
2014 ───── IRIS 발사 (0.4″ 해상도, 채층/TR 편중)
2018 ───── Parker Solar Probe 발사 (in-situ 근일점 0.05 AU)
2019.03 ── SPICE 비행 모델 논문 접수 (이 논문)
2020.02 ── Solar Orbiter 발사 (SPICE 탑재)
2020.08 ── 본 논문 A&A Solar Orbiter Special Issue 게재
2022 ───── Solar Orbiter 첫 근일점 (~0.29 AU)
2025 ───── Solar Orbiter 궤도 경사 >17° (out-of-ecliptic 시작)
2026 ───── (현재) SPICE 극지 EUV 분광 관측 진행 중
```

**중요 시사 / Historical significance**: SPICE는 **50년 계보(Skylab → SMM/UVSP → SOHO/SUMER → Hinode/EIS → IRIS)** 의 정점에서, 처음으로 (i) 넓은 온도 범위를 한 관측으로, (ii) out-of-ecliptic에서, (iii) 0.28 AU 근일점에서 수행하는 분광기이다. SPICE 없이는 Solar Orbiter 미션의 "remote + in-situ 통합" 목표가 완성되지 않는다.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **#12. Parker 1958** (Solar wind prediction) | SPICE가 태양풍 소스를 찾는 대상 현상 자체를 예측한 기초 논문 / Predicted the solar wind phenomenon SPICE now traces to source | ★★★★★ 과학 목표의 원천 |
| **#15. Culhane et al. 2007 (Hinode/EIS)** | 직계 기술 선조. EIS는 EUV 분광기의 현대적 설계 템플릿. SPICE는 EIS의 한계(>1 MK 편중, ecliptic-only)를 극복 / Direct technical predecessor; SPICE overcomes EIS's >1 MK bias and ecliptic-only limitation | ★★★★★ 필수 배경 |
| **#16. De Pontieu et al. 2014 (IRIS)** | 상보적 관계. IRIS는 0.4″로 채층/TR을 고해상, SPICE는 더 낮은 해상도이나 **온도 범위 확장** + **out-of-ecliptic** | ★★★★★ 필수 배경 |
| **#10. Wilhelm et al. 1995 (SOHO/SUMER)** | 파장 범위는 유사, 그러나 SUMER는 wavelength stepping 필요. SPICE는 동시 관측으로 동적 현상 포착 가능 | ★★★★☆ 직접 비교 대상 |
| **#13. Sheeley et al. 1997 (slow wind)** | SPICE FIP-bias 지도가 slow wind의 helmet streamer 기원을 직접 검증할 수 있음 / SPICE's FIP-bias maps can directly test the helmet-streamer origin of slow wind | ★★★★☆ 과학적 응용 |
| **#17. Solanki et al. 2020 (Solar Orbiter PHI)** | 자매 기기. PHI는 광구 자기장, SPICE는 상층 대기 응답 → 가열 메커니즘 연구 | ★★★★☆ 미션 내 협력 |
| **Müller et al. 2020 (Solar Orbiter overview)** | 미션 큰 그림 논문. SPICE의 맥락 제공 | ★★★★☆ 미션 맥락 |
| **DeForest 2015 (SHC patent)** | SPICE가 채택한 Spectral Hybrid Compression 알고리즘의 원천 / Source of the Spectral Hybrid Compression algorithm SPICE adopts | ★★★☆☆ 구현 세부 |
| **Del Zanna & Mason 2018 (EUV review)** | FIP 효과와 EUV 진단의 현대적 리뷰. SPICE 데이터 해석의 이론 프레임 / Modern review of FIP effect and EUV diagnostics — the theory frame for interpreting SPICE data | ★★★★☆ 이론 배경 |

---

## 7. References / 참고문헌

**본 논문 인용 / Paper citation**:
- SPICE Consortium (Anderson, M., Appourchaux, T., Auchère, F. et al.), "The Solar Orbiter SPICE instrument — An extreme UV imaging spectrometer", *Astronomy & Astrophysics*, Vol. 642, A14 (2020). DOI: [10.1051/0004-6361/201935574](https://doi.org/10.1051/0004-6361/201935574)

**직접 인용된 핵심 참고문헌 / Key references cited**:
- Fludra, A., Griffin, D., Caldwell, M., et al., "SPICE EUV spectrometer for the Solar Orbiter mission", *Proc. SPIE 8862*, 88620F-1 (2013). — SPICE 광학 설계 원논문
- Müller, D., Marsden, R. G., St. Cyr, O. C., & Gilbert, H. R., "Solar Orbiter: Exploring the Sun–Heliosphere connection", *Sol. Phys.*, 285, 25 (2013).
- Müller, D., St. Cyr, O. C., Zouganelis, I., et al., "The Solar Orbiter mission: Science overview", *A&A*, 642, A1 (2020).
- Thomas, R. J., "A new solution for a toroidal-grating spectrograph...", *Proc. SPIE 4853*, 411 (2003). — TVLS grating
- Curdt, W., Brekke, P., Feldman, U., et al., "The SUMER spectral atlas of solar-disk features", *A&A*, 375, 591 (2001). — SPICE 라인 선택의 근거
- Curdt, W., Landi, E., & Feldman, U., "The SUMER spectral atlas of solar coronal features", *A&A*, 427, 1045 (2004). — SPICE 라인 선택의 근거
- Culhane, J. L., Harra, L. K., James, A. M., et al., "The EUV Imaging Spectrometer for Hinode", *Sol. Phys.*, 243, 19 (2007). — 직계 선조 EIS
- De Pontieu, B., Title, A. M., Lemen, J. R., et al., "The Interface Region Imaging Spectrograph (IRIS)", *Sol. Phys.*, 289, 2733 (2014). — IRIS
- Wilhelm, K., Curdt, W., Marsch, E., et al., "SUMER — Solar Ultraviolet Measurements of Emitted Radiation", *Sol. Phys.*, 162, 189 (1995). — SOHO/SUMER
- von Steiger, R., Schwadron, N. A., Fisk, L. A., et al., "Composition of quasi-stationary solar wind flows from Ulysses/SWICS", *J. Geophys. Res.*, 105, 27217 (2000). — FIP bias in situ
- DeForest, C. E., "Systems and Methods for Hybrid Compression of Spectral Image Data", US Patent No. 9,031,336 (2015). — SHC
- Del Zanna, G., & Mason, H. E., "Solar UV and X-ray spectral diagnostics", *Living Rev. Solar Phys.*, 15, 5 (2018).
- Schühle, U., Uhlig, H., Curdt, W., et al., "Boron carbide mirror coatings...", *2nd Solar Orbiter Workshop* (2007). — B₄C dichroic coating

**미션 자매 논문 (Solar Orbiter Special Issue) / Companion papers**:
- Auchère, F., Andretta, V., Antonucci, E., et al., "The envisaged cadence of Solar Orbiter's remote-sensing instruments", *A&A*, 642, A6 (2020).
- Rochus, P., Auchère, F., Berghmans, D., et al., "The Solar Orbiter EUI instrument", *A&A*, 642, A8 (2020).
- Owen, C. J., Bruno, R., Livi, S., et al., "The Solar Orbiter Solar Wind Analyser", *A&A*, 642, A16 (2020).
- Solanki, S. K., del Toro Iniesta, J. C., Woch, J., et al., "The Polarimetric and Helioseismic Imager on Solar Orbiter", *A&A*, 642, A11 (2020).
- Zouganelis, I., De Groof, A., Walsh, A. P., et al., "The Solar Orbiter Science Activity Plan", *A&A*, 642, A3 (2020).
