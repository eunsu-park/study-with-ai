---
title: "EVE on SDO — Science Objectives, Instrument Design, Data Products"
authors: "T. N. Woods, F. G. Eparvier, R. Hock, et al."
year: 2012
journal: "Solar Physics, 275, 115–143"
doi: "10.1007/s11207-009-9487-6"
topic: "Solar_Physics"
paper_number: 36
tags: [EUV, irradiance, SDO, EVE, MEGS, ESP, flares, space-weather, thermosphere]
date_read: "2026-04-27"
---

# EVE on SDO: Overview / SDO 탑재 EVE 개요

## 1. Core Contribution / 핵심 기여

**English**:
This paper is the foundational reference for the **Extreme Ultraviolet Variability Experiment (EVE)** aboard NASA's Solar Dynamics Observatory (SDO), launched February 2010. EVE was designed to measure the solar EUV spectral irradiance from **0.1 to 105 nm** with **0.1 nm spectral resolution**, **10-second temporal cadence** (0.25 s for photometers), and **~20% absolute accuracy** — a combination of capabilities unprecedented in space-based solar observations. The paper articulates four science objectives: (i) specify EUV irradiance and its variability; (ii) understand why it varies; (iii) forecast EUV variability; and (iv) understand the geospace response. It describes the instrument suite — two **MEGS** (Multiple EUV Grating Spectrographs: MEGS-A 5–37 nm, MEGS-B 35–105 nm), the **ESP** (EUV SpectroPhotometer, 0.1–39 nm broadbands), and **MEGS-P** (Lyman-alpha 121.6 nm), plus **MEGS-SAM** (Solar Aspect Monitor for 0.1–7 nm photon counting). It also outlines the data product hierarchy (Level 0–3) and the empirical FISM and physics-based NRLEUV models that EVE data will feed.

**한국어**:
본 논문은 2010년 2월 발사된 NASA Solar Dynamics Observatory(SDO) 탑재 **EUV 변동 실험(EVE)**의 기초 참고문헌이다. EVE는 태양 EUV 분광 방사 조도를 **0.1–105 nm** 영역에서 **0.1 nm 분광 분해능**, **10초 시간 주기**(광도계는 0.25초), **약 20% 절대 정확도**로 측정하기 위해 설계되었으며, 이는 당시 우주 기반 태양 관측에서 전례 없는 사양 조합이다. 논문은 네 가지 과학 목표를 제시한다: (i) EUV 방사 조도와 변동성 명시; (ii) 변동 원인 이해; (iii) EUV 변동 예보; (iv) 지구권(geospace) 반응 이해. 기기 구성으로는 두 대의 **MEGS**(다중 EUV 격자 분광기: MEGS-A 5–37 nm, MEGS-B 35–105 nm), **ESP**(EUV 분광 광도계, 0.1–39 nm 광대역), **MEGS-P**(Lyman-α 121.6 nm), 그리고 **MEGS-SAM**(0.1–7 nm 광자 계수용 태양 정렬 모니터)을 포함한다. 또한 데이터 제품 계층(Level 0–3)과 EVE 데이터를 활용할 경험적 FISM 및 물리 기반 NRLEUV 모델을 기술한다.

---

## 2. Reading Notes / 읽기 노트

### 2.1 Introduction (pp. 115–117)

**English**:
SDO is the first LWS mission with a 5-year nominal lifetime starting February 2010. Its mission goal: understand solar variability and its societal/technological impacts. Three onboard instruments — **HMI** (Helioseismic and Magnetic Imager, photospheric magnetograms and Doppler), **AIA** (Atmospheric Imaging Assembly, full-disk EUV/UV imaging), **EVE** (this paper). HMI provides the magnetic source, AIA the spatial morphology, and EVE the spectrally-resolved irradiance — together they form a complete solar-driver dataset for space weather. EVE extends the SOHO and TIMED EUV irradiance records to higher cadence and resolution. Three EVE technology firsts are noted: (1) radiation-hard, back-illuminated 1024×2048 MIT Lincoln Lab CCDs with <1 e- read noise at -70°C; (2) a near-normal grazing-incidence grating with the CCD on the grating itself (Crotser et al. 2007), achieving much higher sensitivity; (3) MEGS-SAM pinhole X-ray imager and Solar Aspect Monitor for sub-arcsec pointing.

**한국어**:
SDO는 2010년 2월 시작된 5년 수명의 첫 LWS 임무이다. 임무 목표는 태양 변동성과 사회·기술적 영향을 이해하는 것. 탑재 기기 3종은 **HMI**(광구 자력도/도플러), **AIA**(전면 디스크 EUV/UV 영상), **EVE**(본 논문). HMI는 자기장 원천, AIA는 공간 형태, EVE는 분광 방사 조도를 제공하여 우주기상 구동 데이터셋을 완성한다. EVE는 SOHO·TIMED 기록을 더 높은 주기·분해능으로 확장한다. EVE의 기술적 신규성 3가지: (1) MIT 링컨 연구소의 방사선 내성 후면 조사 1024×2048 CCD, -70°C에서 1e- 미만 읽기 잡음; (2) 격자 위에 CCD를 직접 배치한 근수직 빗각 격자(Crotser et al. 2007); (3) 부각초 정렬 정밀도의 MEGS-SAM 핀홀 X선 이미저.

### 2.2 EVE Science Plan (pp. 117–122)

**English**:
EUV photons (10–121 nm) and XUV (0.1–10 nm) from the chromosphere, transition region, and corona deposit energy in Earth's upper atmosphere within ~8 minutes. The EUV/XUV spectrum varies by factors of 2 to several orders of magnitude depending on wavelength and timescale (seconds to decades). Figure 1 shows that thermospheric neutral density at 500 km changes by factor >10 from solar minimum to maximum. Figure 2 compares EUV photon energy (~5 mW m^-2) to solar wind kinetic energy input (~0.5 mW m^-2) over decades — EUV dominates by an order of magnitude.

**한국어**:
채층, 천이층, 코로나에서 방출되는 EUV(10–121 nm)와 XUV(0.1–10 nm) 광자는 약 8분 내 지구 상층 대기에 에너지를 전달한다. EUV/XUV 스펙트럼은 파장과 시간 척도(초~수십 년)에 따라 2배에서 수 차수까지 변동한다. 그림 1은 500 km 열권 중성 밀도가 태양 극소~극대 사이 10배 이상 변함을 보인다. 그림 2는 EUV 광자 에너지(약 5 mW m^-2)가 태양풍 운동 에너지(약 0.5 mW m^-2)를 한 차수 압도함을 보인다.

#### Objective 1 — Specify EUV Irradiance / EUV 방사 조도 명시

**English**: Build a database of the EUV spectrum and its variations during flares, active-region evolution, and the solar cycle. TIMED/SEE achieved ~20% accuracy at 1-nm resolution; EVE improves to 10% at some wavelengths and 0.1 nm resolution with 10-second cadence — far beyond SOHO/SEM and GOES/XRS broadband measurements.

**한국어**: 플레어, 활동 영역 진화, 태양 주기에 걸친 EUV 스펙트럼 변동 데이터베이스 구축. TIMED/SEE는 1 nm 분해능에서 20% 정확도를 달성했고, EVE는 일부 파장에서 10%까지, 0.1 nm 분해능 + 10초 주기로 SOHO/SEM·GOES/XRS 광대역 측정을 압도한다.

#### Objective 2 — Understand Why EUV Varies / EUV 변동 원인 이해

**English**: Connect EUV variability to magnetic-flux emergence, transport, and the solar dynamo. Approach: combine HMI magnetograms, AIA imagery, and EVE spectra. Figure 5 shows the "east-limb forecasting" technique — east-limb EIT 28.4 nm flux predicts disk-integrated flux 3–10 days ahead because active regions rotate from east to west across the disk. Figure 6 shows SET's IDAR algorithm forecasting F10.7 from active-region areas in EIT images.

**한국어**: EUV 변동을 자기 플럭스 출현·수송·태양 다이나모와 연결. 접근법: HMI 자력도 + AIA 영상 + EVE 스펙트럼. 그림 5는 "동쪽 가장자리 예보" 기법 — 동쪽 가장자리 EIT 28.4 nm 플럭스가 활동 영역 동→서 자전으로 인해 디스크 통합 플럭스를 3–10일 미리 예측. 그림 6은 SET의 IDAR 알고리즘으로 EIT 영상의 활동 영역 면적으로 F10.7을 예보.

#### Objective 3 — Forecast EUV Variability / EUV 변동 예보

**English**: USAF needs 3-hr to 72-hr forecasts; NOAA provides 5-day to 45-day F10.7 forecasts. EVE will improve these via (a) east-limb flux as 3–10 day predictor, (b) helioseismic far-side imaging (SOHO/MDI) for emergent active regions, (c) backscattered Lyman-alpha (SOHO/SWAN) for far-side activity, (d) statistical methods using EVE database. Short-term flare forecasts will leverage helicity, dimming, and shadow signatures from HMI/AIA.

**한국어**: 미 공군은 3–72시간 예보 필요; NOAA는 5–45일 F10.7 예보 제공. EVE는 (a) 동쪽 가장자리 플럭스 3–10일 예측, (b) MDI 헬리오시즘으로 후면 활동 영역, (c) SWAN의 Lyman-α 후방 산란, (d) EVE 데이터베이스 통계 방법으로 개선. 단기 플레어 예보는 HMI/AIA의 헬리시티·디밍·그림자 신호 활용.

#### Objective 4 — Geospace Response / 지구권 반응

**English**: EVE will drive the NOAA CTIPe (Coupled Thermosphere Ionosphere Plasmasphere electrodynamics), Utah State TDIM, and GAIM models with high-cadence flare spectra — a first for ionospheric flare-response simulation.

**한국어**: EVE는 NOAA CTIPe, 유타 주립대 TDIM, GAIM 모델에 고주기 플레어 스펙트럼을 입력하여 전리권 플레어 반응 시뮬레이션을 처음으로 가능하게 한다.

### 2.3 Measurement Requirements (p. 124, Table 1)

**English**: Spectral coverage 0.1–5 nm at 1 nm resolution + 5–105 nm at 0.1 nm + 121.6 nm Lyman-alpha. Flare-detection cadence 0.25 s (ESP photometer). Spectral cadence 10 s. Absolute accuracy 20% (10% goal). Operational data latency <15 minutes for the Level 0C near-real-time space-weather product.

**한국어**: 분광 영역 0.1–5 nm @ 1 nm + 5–105 nm @ 0.1 nm + 121.6 nm Lyman-α. 플레어 검출 주기 0.25초(ESP). 분광 주기 10초. 절대 정확도 20%(목표 10%). 운영 데이터 지연 15분 미만(Level 0C 근실시간 우주기상 제품).

### 2.4 Instrument Description / 기기 기술

**English**:
- **MEGS-A** (5–37 nm, 0.1 nm res): Grazing-incidence spectrograph with the CCD mounted on the grating itself — a Crotser et al. (2007) design that boosts sensitivity by 10× over conventional grazing optics. Two slits (SAM channel and main spectrum) share the CCD.
- **MEGS-B** (35–105 nm, 0.1 nm res): Normal-incidence dual-pass spectrograph (cross-dispersed); two reflections required because single-grating coatings cannot cover 35–105 nm efficiently.
- **MEGS-SAM** (0.1–7 nm): Pinhole imager on the MEGS-A CCD; counts individual X-ray photons and measures their energy via CCD pulse height.
- **MEGS-P** (121.6 ± 5 nm): Photodiode behind a Lyman-alpha filter; tracks H I emission.
- **ESP** (0.1–7 nm + 17.1, 25.7, 30.4, 36.6 nm bands): Quadrant photodiodes behind a transmission grating; quadrant geometry gives the flare location on the disk via centroiding.

**한국어**:
- **MEGS-A** (5–37 nm, 0.1 nm 분해능): 격자 자체에 CCD를 장착한 빗각 입사 분광기 — Crotser et al. (2007) 설계로 전통적 빗각 광학보다 감도 10배 향상. 두 슬릿(SAM 채널·주 스펙트럼)이 CCD 공유.
- **MEGS-B** (35–105 nm, 0.1 nm 분해능): 수직 입사 이중 패스 분광기(교차 분산); 단일 격자 코팅이 35–105 nm를 효율적으로 커버할 수 없으므로 두 번 반사.
- **MEGS-SAM** (0.1–7 nm): MEGS-A CCD 위 핀홀 이미저; 개별 X선 광자를 계수하고 CCD 펄스 높이로 에너지 측정.
- **MEGS-P** (121.6 ± 5 nm): Lyman-α 필터 뒤 광 다이오드; H I 방출 추적.
- **ESP** (0.1–7 nm + 17.1, 25.7, 30.4, 36.6 nm 대역): 투과 격자 뒤 사분면 광 다이오드; 사분면 기하구조로 디스크 상 플레어 위치 중심화.

### 2.5 Data Products / 데이터 제품

| Level | Cadence | Content | Latency |
|-------|---------|---------|---------|
| 0B | 10 s | Raw counts | None |
| 0C | 1 min | Real-time space-weather (broadbands + 0.1 nm spectra) / 실시간 우주기상 | <15 min |
| 2 | 0.25 s (photometers) / 10 s (spectra) | Calibrated irradiance / 보정된 방사 조도 | Hours |
| 3 | Daily / Hourly | Averages / 평균 | Days |

### 2.6 Models / 모델

**English**:
- **FISM (Flare Irradiance Spectral Model)**, Chamberlin, Woods & Eparvier 2007/2008: empirical, uses GOES X-ray flux + Lyman-alpha as proxies to predict the EUV spectrum at 1-nm × 60-s resolution; EVE data refine FISM.
- **NRLEUV**, Warren, Mariska & Lean 2001: physics-based, uses differential emission measure (DEM) distributions from full-disk imagery.
- **SIP**, Tobiska 2004/2008: hybrid system combining real-time data and reference spectra for operations.

**한국어**:
- **FISM (플레어 방사 조도 분광 모델)**, Chamberlin et al. 2007/2008: 경험 모델, GOES X선 플럭스 + Lyman-α를 프록시로 1 nm × 60초 EUV 스펙트럼 예측; EVE 데이터로 정밀화.
- **NRLEUV**, Warren et al. 2001: 물리 기반, 전면 디스크 영상의 차등 방출 측도(DEM) 분포 사용.
- **SIP**, Tobiska 2004/2008: 운영용 실시간 데이터·참조 스펙트럼 결합 하이브리드 시스템.

### 2.7 Calibration Strategy / 보정 전략

**English**:
EVE achieves its 20% absolute accuracy through a layered calibration approach. Pre-flight calibration was performed at the NIST SURF-III synchrotron, providing absolute spectral irradiance traceable to NIST standards. In-flight calibration uses (a) periodic underflight rocket experiments — duplicate calibration rockets fly NIST-calibrated copies of EVE channels and observe the Sun for ~5 minutes, providing absolute irradiance ground truth; (b) cross-calibration between MEGS-A, MEGS-B, ESP, and MEGS-P at overlap wavelengths (35–37 nm); (c) onboard calibration filters that can be inserted to monitor degradation. The combination addresses optical-element degradation (carbon contamination on grating surfaces), CCD sensitivity drift, and filter pinhole growth — all common failure modes of long-duration UV space instruments.

**한국어**:
EVE는 20% 절대 정확도를 계층적 보정 접근으로 달성한다. 사전 비행 보정은 NIST SURF-III 싱크로트론에서 수행하여 NIST 표준 추적 가능한 절대 분광 방사 조도를 제공한다. 비행 중 보정은 (a) 주기적 언더플라이트 로켓 실험 — NIST 보정 EVE 채널 복제본을 탑재한 보정 로켓이 약 5분간 태양을 관측하여 절대 방사 조도 기준값 제공; (b) 중첩 파장(35–37 nm)에서 MEGS-A·MEGS-B·ESP·MEGS-P 교차 보정; (c) 열화 모니터링을 위한 삽입 가능 보정 필터. 이 조합은 광학 소자 열화(격자 표면의 탄소 오염), CCD 감도 표류, 필터 핀홀 성장 — 장기 UV 우주 기기의 일반적 고장 모드 — 를 모두 다룬다.

### 2.8 Operational Real-Time Data / 운영 실시간 데이터

**English**:
The Level 0C real-time space-weather product is delivered to NOAA SWPC with <15-minute latency. It contains: (1) ESP broadband irradiances at 0.25 s cadence, (2) MEGS-A spectra at 1-min cadence (binned from 10-s native), (3) MEGS-B spectra at 1-min cadence, (4) Lyman-alpha photometer at 0.25 s. SWPC ingests these into operational space-weather scales (R-scale radio blackouts), GPS-error nowcasts, and satellite-drag advisories. This near-real-time pipeline is a deliberate departure from prior research instruments and demonstrates LWS's "research-to-operations" mandate. Failure-mode handling: if MEGS goes offline, ESP+MEGS-P alone provide a degraded but still operational data product.

**한국어**:
Level 0C 실시간 우주기상 제품은 15분 미만 지연으로 NOAA SWPC에 전달된다. 내용: (1) ESP 광대역 방사 조도 0.25초 주기, (2) MEGS-A 스펙트럼 1분 주기(원래 10초에서 비닝), (3) MEGS-B 스펙트럼 1분 주기, (4) Lyman-α 광도계 0.25초. SWPC는 이를 운영 우주기상 척도(R-scale 무선 통신 두절), GPS 오차 현황 예보, 위성 항력 권고에 통합한다. 이 근실시간 파이프라인은 기존 연구용 기기와의 의도적 차별화이며 LWS의 "연구→운영" 명령을 구현한다. 고장 모드 처리: MEGS가 오프라인되면 ESP+MEGS-P 단독으로 저하되었지만 여전히 작동하는 데이터 제품 제공.

### 2.9 Wavelength Heritage and Key Lines / 파장 유산과 주요 선

**English**:
Key EUV emission lines that EVE resolves with 0.1 nm resolution and their formation temperatures:
- He II 30.4 nm (chromosphere/transition region, log T = 4.7) — proxy for chromospheric heating
- He I 58.4 nm (chromosphere, log T = 4.5) — cooler chromospheric counterpart
- Fe IX 17.1 nm (corona, log T = 5.8) — quiet corona
- Fe XII 19.5 nm (corona, log T = 6.1) — active region corona
- Fe XV 28.4 nm (active region, log T = 6.3) — flare warm component
- Fe XVI 33.5 nm (hot active region, log T = 6.4) — late-phase EUV peak
- Fe XX 13.3 nm (flare, log T = 7.0) — flare hot component
- Fe XXIII 13.3 nm (flare, log T = 7.2) — impulsive phase
- O V 63.0 nm (transition region, log T = 5.4)
- H I Lyman-alpha 121.6 nm (chromosphere, log T = 4.0) — strongest UV line; thermospheric NO photolysis driver

**한국어**:
EVE가 0.1 nm 분해능으로 분리하는 주요 EUV 방출선과 형성 온도:
- He II 30.4 nm (채층/천이층, log T = 4.7) — 채층 가열 프록시
- He I 58.4 nm (채층, log T = 4.5) — 더 차가운 채층 대응선
- Fe IX 17.1 nm (코로나, log T = 5.8) — 정상 코로나
- Fe XII 19.5 nm (코로나, log T = 6.1) — 활동 영역 코로나
- Fe XV 28.4 nm (활동 영역, log T = 6.3) — 플레어 따뜻한 성분
- Fe XVI 33.5 nm (뜨거운 활동 영역, log T = 6.4) — 후기 단계 EUV 피크
- Fe XX 13.3 nm (플레어, log T = 7.0) — 플레어 뜨거운 성분
- Fe XXIII 13.3 nm (플레어, log T = 7.2) — 충격 단계
- O V 63.0 nm (천이층, log T = 5.4)
- H I Lyman-α 121.6 nm (채층, log T = 4.0) — 가장 강한 UV 선; 열권 NO 광분해 구동

---

## 3. Key Takeaways / 핵심 시사점

1. **0.1 nm × 10 s × 20% accuracy / 0.1 nm × 10초 × 20% 정확도**
   - **English**: This combination is the headline. 0.1 nm spectral resolution separates major EUV emission lines (e.g., He II 30.4 nm, Fe IX 17.1 nm, Fe XV 28.4 nm). 10-s cadence captures flare impulsive phases. 20% absolute accuracy meets atmospheric model needs.
   - **한국어**: 이 조합이 핵심이다. 0.1 nm 분해능은 주요 EUV 방출선(He II 30.4 nm, Fe IX 17.1 nm, Fe XV 28.4 nm 등)을 분리한다. 10초 주기는 플레어 충격 단계를 포착. 20% 절대 정확도는 대기 모델 요구사항 충족.

2. **EUV dominates space-weather energy / EUV는 우주기상 에너지를 지배**
   - **English**: EUV photon energy (~5 mW m^-2) into the upper atmosphere is roughly 10× the solar-wind kinetic energy input. Without EUV measurements, ionosphere/thermosphere modeling is fundamentally limited.
   - **한국어**: EUV 광자 에너지(약 5 mW m^-2)는 태양풍 운동 에너지의 약 10배. EUV 측정 없이는 전리권·열권 모델링이 본질적으로 제한된다.

3. **MEGS-A grazing-CCD design is novel / MEGS-A의 격자-CCD 설계는 혁신적**
   - **English**: Mounting the CCD directly on the grating (Crotser et al. 2007) eliminates a reflection and dramatically increases sensitivity at 5–37 nm where conventional optics suffer.
   - **한국어**: CCD를 격자 위에 직접 장착(Crotser et al. 2007)하여 반사 한 번을 제거하고 5–37 nm 영역에서 감도를 극적으로 향상.

4. **MEGS-SAM enables soft X-ray imaging at no extra cost / MEGS-SAM은 추가 비용 없이 연-X선 영상 가능**
   - **English**: A pinhole on a corner of the MEGS-A CCD turns the spectrograph into a soft X-ray (0.1–7 nm) imager with photon-counting energy resolution — a clever multi-use of one detector.
   - **한국어**: MEGS-A CCD 모서리의 핀홀이 분광기를 광자 계수 에너지 분해능을 가진 연-X선(0.1–7 nm) 이미저로 변환 — 한 검출기의 영리한 다용도 활용.

5. **East-limb flux as 3–10 day forecast / 동쪽 가장자리 플럭스가 3–10일 예보**
   - **English**: Active regions rotate east-to-west across the disk in 14 days. Measuring east-limb 28.4 nm flux today gives a preview of disk-integrated flux 3–10 days ahead.
   - **한국어**: 활동 영역은 14일에 걸쳐 동→서로 자전. 오늘의 동쪽 가장자리 28.4 nm 플럭스는 3–10일 후 디스크 통합 플럭스의 예고편.

6. **EVE drives ionospheric flare models for the first time / EVE가 처음으로 전리권 플레어 모델 구동**
   - **English**: Prior ionospheric models used daily-averaged or proxy-based EUV inputs. EVE's 10-s spectra enable end-to-end simulation of flare-driven sudden ionospheric disturbances (SIDs).
   - **한국어**: 기존 전리권 모델은 일평균 또는 프록시 기반 EUV 입력 사용. EVE의 10초 스펙트럼은 플레어 구동 갑작스런 전리권 교란(SID)의 종단 시뮬레이션을 가능케 한다.

7. **Hybrid Level 0C product blurs research/operations boundary / 하이브리드 Level 0C 제품이 연구·운영 경계를 흐림**
   - **English**: <15 min latency means EVE feeds operational space-weather forecasting almost in real time — a deliberate "transition research to operations" goal of LWS.
   - **한국어**: 15분 미만 지연은 EVE가 거의 실시간으로 운영 우주기상 예보를 공급함을 의미 — LWS의 의도적 "연구→운영 전환" 목표.

8. **Late-phase EUV is a new flare regime / 후기 단계 EUV는 새로운 플레어 영역**
   - **English**: EVE's continuous spectral coverage revealed that some flares show a second peak in warm coronal lines hours after the GOES soft X-ray peak (Woods et al. 2011) — energy release in higher loops not detected by GOES alone.
   - **한국어**: EVE의 연속 분광 관측은 일부 플레어가 GOES 연-X선 피크 후 수시간 뒤 따뜻한 코로나 선에서 두 번째 피크를 보임을 밝혔다(Woods et al. 2011) — GOES 단독으로는 감지 못한 더 높은 루프의 에너지 방출.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Spectral Irradiance Definition / 분광 방사 조도 정의

$$
E_\lambda(\lambda, t) = \frac{dP}{dA \, d\lambda}\bigg|_{1\,\mathrm{AU}} \quad [\mathrm{W\,m^{-2}\,nm^{-1}}]
$$

**English**: $E_\lambda$ is the power per unit area per unit wavelength at 1 AU. Integrating over a passband $[\lambda_1, \lambda_2]$ gives the band irradiance $E_{\mathrm{band}} = \int_{\lambda_1}^{\lambda_2} E_\lambda \, d\lambda$.

**한국어**: $E_\lambda$는 1 AU에서 단위 면적·단위 파장당 전력. 통과 대역 $[\lambda_1, \lambda_2]$에 대해 적분하면 대역 방사 조도 $E_{\mathrm{band}}$.

### 4.2 Photon-Energy Conversion / 광자 에너지 변환

$$
n_\gamma(\lambda) = \frac{E_\lambda \cdot \lambda}{h c} \quad [\mathrm{photons\,m^{-2}\,s^{-1}\,nm^{-1}}]
$$

**English**: Photon flux density. $h = 6.626 \times 10^{-34}$ J·s; $c = 3 \times 10^8$ m/s. For $\lambda = 30.4$ nm, photon energy $hc/\lambda = 6.5 \times 10^{-18}$ J = 40.8 eV.

**한국어**: 광자 플럭스 밀도. $\lambda = 30.4$ nm에서 광자 에너지 $hc/\lambda = 6.5 \times 10^{-18}$ J = 40.8 eV.

### 4.3 Instrument Response and Count Rate / 기기 응답 및 계수율

$$
C(\lambda, t) = E_\lambda(\lambda, t) \cdot A_{\mathrm{eff}}(\lambda) \cdot \Delta\lambda \cdot \Delta t \cdot \frac{\lambda}{hc}
$$

**English**: Counts per pixel per integration. $A_{\mathrm{eff}}(\lambda)$ is the effective area (cm^2), the product of geometric area, grating efficiency, filter transmission, and CCD quantum efficiency. $\Delta\lambda$ is per-pixel spectral width (~0.02 nm, oversampled 5× to give 0.1 nm resolution after binning).

**한국어**: 픽셀·적분당 계수. $A_{\mathrm{eff}}(\lambda)$는 유효 면적(cm^2): 기하학적 면적 × 격자 효율 × 필터 투과율 × CCD 양자 효율. $\Delta\lambda$는 픽셀당 분광 폭(약 0.02 nm, 5배 오버샘플하여 비닝 후 0.1 nm 분해능).

### 4.4 Flare Detection SNR / 플레어 검출 SNR

$$
\mathrm{SNR} = \frac{C_{\mathrm{flare}} - C_{\mathrm{quiet}}}{\sqrt{C_{\mathrm{flare}} + C_{\mathrm{quiet}} + N_{\mathrm{read}}^2}}
$$

**English**: For ESP at 0.25 s cadence, quiet-Sun count rate is ~10^4 counts; an X-class flare doubles or more, giving SNR > 50 — easy detection.

**한국어**: 0.25초 주기 ESP에서 정상 태양 계수율 약 10^4; X급 플레어는 2배 이상으로 SNR > 50 — 쉬운 검출.

### 4.5 East-Limb Forecast Correlation / 동쪽 가장자리 예보 상관

$$
\hat{F}_{\mathrm{disk}}(t + \tau) = a + b \cdot F_{\mathrm{east-limb}}(t)
$$

**English**: Linear regression with lag $\tau \approx 3$–10 days. Lean, Picone & Emmert (2009) report correlation coefficients > 0.7 for 28.4 nm Fe XV emission.

**한국어**: 시간 지연 $\tau \approx$ 3–10일의 선형 회귀. Lean et al. (2009)는 28.4 nm Fe XV 방출에 대해 상관계수 > 0.7 보고.

### 4.6 Worked Example — He II 30.4 nm during a Flare / He II 30.4 nm 플레어 사례

**English**: Quiet-Sun He II 30.4 nm irradiance at 1 AU is about $E_{\lambda} \approx 0.10$ mW m^-2 nm^-1 over a 0.1 nm linewidth, giving a band irradiance of 0.01 mW m^-2. During an X1 flare, the line typically increases by ~30%. With EVE's MEGS-A effective area of ~0.05 cm^2 at 30.4 nm and 10-s integration:

$C \approx (0.01 \times 10^{-3} \, \mathrm{W\,m^{-2}}) \times (5 \times 10^{-6} \, \mathrm{m^2}) \times 10 \, \mathrm{s} \times \lambda/(hc)$
$\approx 7.7 \times 10^4$ photons → easily detectable above read noise (<1 e-).

**한국어**: 1 AU의 정상 태양 He II 30.4 nm 방사 조도는 0.1 nm 선폭에 대해 약 $E_\lambda \approx 0.10$ mW m^-2 nm^-1 → 대역 방사 조도 0.01 mW m^-2. X1 플레어 시 약 30% 증가. MEGS-A 유효 면적 30.4 nm에서 약 0.05 cm^2, 10초 적분으로 약 $7.7 \times 10^4$ 광자 → 읽기 잡음(<1 e-) 위로 쉽게 검출.

### 4.7 Thermospheric Heating from EUV / EUV에 의한 열권 가열

$$
Q_{\mathrm{EUV}}(z) = \sum_i n_i(z) \int \sigma_i(\lambda) \, F_\lambda(\lambda, z) \, d\lambda
$$

**English**: Volumetric heating rate $Q$ at altitude $z$ depends on species number density $n_i$, photoionization cross-section $\sigma_i(\lambda)$, and EUV photon flux $F_\lambda$ (already attenuated by overlying atmosphere). Roughly half the absorbed energy heats the gas; the rest goes into ionization and chemistry.

**한국어**: 고도 $z$의 부피 가열률 $Q$는 화학종 수밀도 $n_i$, 광이온화 단면적 $\sigma_i(\lambda)$, EUV 광자 플럭스 $F_\lambda$(상층 대기에 의해 이미 감쇠)에 의존. 흡수 에너지의 대략 절반은 가스를 가열하고 나머지는 이온화·화학에 사용.

### 4.8 EVE Native Pixel-to-Wavelength Mapping / EVE 픽셀-파장 변환

$$
\lambda_p = \lambda_0 + (p - p_0) \cdot \delta\lambda_{\mathrm{pix}}
$$

**English**: Linear dispersion approximation for MEGS, where $\delta\lambda_{\mathrm{pix}} \approx 0.02$ nm/pixel. Final spectra are binned by 5 pixels to obtain the nominal 0.1 nm resolution while preserving line-centroid precision.

**한국어**: MEGS의 선형 분산 근사, $\delta\lambda_{\mathrm{pix}} \approx 0.02$ nm/픽셀. 최종 스펙트럼은 5픽셀 비닝하여 공칭 0.1 nm 분해능을 얻으면서 선 중심 정밀도 보존.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1962    OSO-1: first space EUV irradiance measurements
         /  최초 우주 EUV 측정
   |
1980s   SME, AE-E missions: limited spectral coverage
         /  제한된 분광 영역
   |
1991    UARS/SOLSTICE: 0.2 nm UV irradiance
         /  0.2 nm UV 방사 조도
   |
1995    SOHO/SEM: broadband EUV at 26-34 nm, daily cadence
         /  광대역 EUV, 일주기
   |
2002    TIMED/SEE: 1 nm EUV spectra, daily cadence
         /  1 nm EUV 스펙트럼, 일주기
   |
2005    F10.7 + Mg II as proxies dominate operational forecasting
         /  F10.7 + Mg II 프록시가 운영 예보 지배
   |
2007    Crotser et al. — grazing-CCD grating innovation
         /  격자-CCD 일체형 혁신
   |
2007    Chamberlin et al. — FISM empirical flare model
         /  FISM 경험 플레어 모델
   |
2010 ★  SDO launched February 11; EVE first light February 25
         /  SDO 발사, EVE 초광 (THIS PAPER)
   |
2011    EVE late-phase EUV emission discovered (Woods et al.)
         /  EVE 후기 단계 EUV 방출 발견
   |
2014    MEGS-A CEB anomaly — MEGS-A retired May 2014
         /  MEGS-A CEB 이상 — 2014년 5월 운용 종료
   |
2018    GOES-R EXIS adopts ESP-style channels
         /  GOES-R EXIS가 ESP 스타일 채널 채택
   |
2025+   Future LWS missions inherit EVE design philosophy
         /  미래 LWS 임무가 EVE 설계 철학 계승
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 |
|--------------|-------------------|
| Pesnell, Thompson & Chamberlin 2012 (SDO mission) | Parent SDO mission overview / SDO 임무 개요 |
| Lemen et al. 2012 (AIA) | Sister instrument; AIA's narrow EUV bands cross-calibrate with EVE / 자매 기기; AIA 좁은 EUV 대역이 EVE와 교차 보정 |
| Schou et al. 2012 (HMI) | Provides the magnetic source for EVE EUV variations / EVE EUV 변동의 자기 원천 제공 |
| Woods et al. 2005 (TIMED/SEE) | Predecessor EUV irradiance instrument / 선행 EUV 방사 조도 기기 |
| Chamberlin, Woods & Eparvier 2007/2008 (FISM) | Empirical flare model EVE refines / EVE가 정밀화하는 경험 플레어 모델 |
| Warren, Mariska & Lean 2001 (NRLEUV) | Physics-based DEM model EVE validates / EVE가 검증하는 DEM 기반 물리 모델 |
| Hock et al. 2010 (EVE instrument) | Detailed instrument description companion paper / 상세 기기 기술 동반 논문 |
| Didkovsky et al. 2010 (ESP calibration) | ESP pre-flight calibration companion / ESP 사전 비행 보정 동반 |
| Woods et al. 2011 (EUV late phase) | First major science result enabled by EVE / EVE가 가능케 한 최초의 주요 과학 결과 |
| Tobiska 2004/2008 (SIP) | Operational hybrid system using EVE data / EVE 데이터 사용 운영 하이브리드 시스템 |

---

## 7. References / 참고문헌

- Woods, T. N. et al., "Extreme Ultraviolet Variability Experiment (EVE) on the Solar Dynamics Observatory (SDO): Overview of Science Objectives, Instrument Design, Data Products, and Model Developments", Solar Physics, 275, 115–143, 2012. [DOI:10.1007/s11207-009-9487-6]
- Pesnell, W. D., Thompson, B. J., & Chamberlin, P. C., "The Solar Dynamics Observatory (SDO)", Solar Phys., 275, 3, 2012.
- Hock, R. et al., "Extreme Ultraviolet Variability Experiment (EVE) Multiple EUV Grating Spectrographs (MEGS): Radiometric Calibrations and Results", Solar Phys., 2010.
- Didkovsky, L. V. et al., "EUV SpectroPhotometer (ESP) in Extreme Ultraviolet Variability Experiment (EVE)", Solar Phys., 2010.
- Crotser, D. A. et al., "EUV spectrograph optical design developments for SDO-EVE", SPIE Proc. 6689, 2007.
- Chamberlin, P. C., Woods, T. N., & Eparvier, F. G., "Flare Irradiance Spectral Model (FISM)", Space Weather, 2007 (preliminary), 2008 (full).
- Warren, H. P., Mariska, J. T., & Lean, J., "A new model of solar EUV irradiance variability — NRLEUV", J. Geophys. Res., 106, 15745, 2001.
- Woods, T. N. & Rottman, G. J., "Solar Ultraviolet Variability over Time Periods of Aeronomic Interest", AGU Geophys. Mon. 130, 2002.
- Woods, T. N. et al., "TIMED/SEE results", J. Geophys. Res., 110, A01312, 2005.
- Lean, J., Picone, J. M., & Emmert, J. T., "Quantitative forecasting of near-term solar activity and upper atmospheric density", J. Geophys. Res., 114, A07301, 2009.
- Woods, T. N. et al., "New Solar EUV Irradiance Observations during Flares — Late phase", Astrophys. J., 739, 59, 2011.
- Knipp, D. J. et al., "Direct and Indirect Thermospheric Heating Sources for Solar Cycles 21–23", Solar Phys., 224, 495, 2005.
- Tobiska, W. K. et al., "The SOLAR2000 empirical solar irradiance model and forecast tool", JASTP, 62, 1233, 2000.
- Picone, J. M., Hedin, A. E., Drob, D. P., & Aikin, A. C., "NRLMSISE-00 empirical model of the atmosphere", J. Geophys. Res., 107, 1468, 2002.
