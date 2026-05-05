---
title: "The Solar Dynamics Observatory (SDO)"
authors: "W. Dean Pesnell, B.J. Thompson, P.C. Chamberlin"
year: 2012
journal: "Solar Physics, Vol. 275, pp. 3–15"
doi: "10.1007/s11207-011-9841-3"
topic: Solar Observation
tags: [SDO, LWS, AIA, HMI, EVE, GEO-synchronous orbit, Ka-band, JSOC, White Sands, space weather, NASA, GSFC]
status: completed
date_started: 2026-04-16
date_completed: 2026-04-16
---

# 35. The Solar Dynamics Observatory (SDO) / 태양 동역학 관측위성 (SDO)

---

## 1. Core Contribution / 핵심 기여

SDO(Solar Dynamics Observatory)는 NASA의 **Living With a Star (LWS)** 프로그램의 첫 번째 미션으로, 2010년 2월 11일 케네디 우주센터에서 Atlas V 401 로켓으로 발사되었다. SDO는 태양 변동성(solar variability)의 원인을 이해하고 우주 기상(space weather)을 예측하는 것을 목표로 한다. 세 개의 과학 기기 — **AIA**(Atmospheric Imaging Assembly, LMSAL), **HMI**(Helioseismic and Magnetic Imager, Stanford), **EVE**(EUV Variability Experiment, LASP/CU Boulder) — 를 탑재하고 있으며, 경사 지구정지궤도(inclined geosynchronous orbit, 28.5° 경사각, 102°W 경도)에서 운용된다. 뉴멕시코 White Sands의 전용 Ka-band(26 GHz) 지상국을 통해 **150 Mbit/s**의 연속 다운링크를 수행하며, 하루 약 **1.5 TB**의 과학 데이터를 전송한다. 이는 발사 당시 NASA 과학 미션 중 최대 데이터 생산량이었다. SDO는 소모품이 없어 추진제 수명만 100년 이상으로 추정되며, 95% 이상의 관측 듀티 사이클을 달성한다. SDO는 SOHO의 후계 미션으로서, SOHO의 12개 기기를 3개의 훨씬 강력한 기기로 대체하였다.

SDO (Solar Dynamics Observatory) is the **first mission** of NASA's **Living With a Star (LWS)** program, launched on 11 February 2010 from Kennedy Space Center aboard an Atlas V 401 rocket. SDO's goal is to understand the causes of solar variability and to predict space weather. It carries three science instruments — **AIA** (Atmospheric Imaging Assembly, LMSAL), **HMI** (Helioseismic and Magnetic Imager, Stanford), and **EVE** (EUV Variability Experiment, LASP/CU Boulder) — and operates in an inclined geosynchronous orbit (28.5° inclination, 102°W longitude). Through a dedicated Ka-band (26 GHz) ground station at White Sands, New Mexico, SDO performs continuous downlink at **150 Mbit/s**, transmitting approximately **1.5 TB** of science data per day. This was the largest data output of any NASA science mission at the time of launch. SDO has no consumables, with an estimated propellant lifetime exceeding 100 years, and achieves an observing duty cycle greater than 95%. As the successor to SOHO, SDO replaced SOHO's 12 instruments with 3 far more capable ones.

---

## 2. Reading Notes / 읽기 노트

### §1 Preface / 서문 (pp. 3–4)

LWS 프로그램의 목표는 인간 활동에 영향을 미치는 태양 변동성의 원인을 이해하는 것이다. 우주 기상의 실질적 위험은 다양하다: 위성 손상(satellite damage), 전력망 장애(power grid failures), GPS 교란(GPS disruption), 항공 방사선 노출(aviation radiation exposure). LWS 프로그램에는 SDO 외에도 **RBSP**(Van Allen Probes), **BARREL**(balloon-borne), **Solar Probe Plus**(현재 Parker Solar Probe), 그리고 ESA와의 공동 미션인 **Solar Orbiter**가 포함된다. 국제 LWS(ILWS) 프로그램은 이러한 노력을 전 세계적으로 조율한다.

The LWS program aims to understand the causes of solar variability that affect human activities. The practical risks of space weather are diverse: satellite damage, power grid failures, GPS disruption, and aviation radiation exposure. Besides SDO, the LWS program includes **RBSP** (Van Allen Probes), **BARREL** (balloon-borne), **Solar Probe Plus** (now Parker Solar Probe), and the joint ESA mission **Solar Orbiter**. The International LWS (ILWS) program coordinates these efforts globally.

### §2 Introduction / 서론 (pp. 4–5)

SDO는 태양 활동 — 플레어(flares), 코로나 물질 방출(CMEs), 태양주기(solar cycle) — 을 **예측**하는 것을 목표로 설계되었다. 자기장 토폴로지(magnetic field topology)의 모니터링이 핵심이다. 태양주기 24(Solar Cycle 24)는 2008년 12월에 시작되었으며, 평균 이하의 피크가 2013년에 예측되었다. 두 기기(AIA, EVE)가 EUV에 초점을 맞추고 있어 태양 복사 변동에 대한 포괄적 모니터링을 제공한다.

SDO was designed to **predict** solar activity — flares, coronal mass ejections (CMEs), and the solar cycle. Monitoring magnetic field topology is central. Solar Cycle 24 began in December 2008, with a below-average peak predicted for 2013. Two instruments (AIA, EVE) focus on EUV, providing comprehensive monitoring of solar irradiance variations.

SDO는 매일 약 **150,000장의 고해상도 풀디스크 태양 영상**과 **9,000개의 EUV 스펙트럼**을 전송한다. 5년 주미션(prime mission) 기간 동안 총 **3–4 PB**의 원시 데이터가 예상되었다.

SDO transmits approximately **150,000 high-resolution full-disk solar images** and **9,000 EUV spectra** daily. Over the 5-year prime mission, a total of **3–4 PB** of raw data was projected.

### §3 History / 역사 (pp. 5–6)

SDO는 이전 세대 미션들 — **Yohkoh**(1991), **SOHO**(1995), **TRACE**(1998) — 의 과학적·기술적 유산을 계승·강화한 플래그십(flagship) 미션이다. SOHO/MDI가 시선방향(LOS) 자기장만 측정했던 것과 달리, SDO/HMI는 풀디스크 **벡터 자기장(vector magnetograph)** 관측 능력을 갖추고 있다.

SDO is a flagship mission that inherits and enhances the scientific and technical heritage of previous-generation missions — **Yohkoh** (1991), **SOHO** (1995), and **TRACE** (1998). Unlike SOHO/MDI, which measured only line-of-sight (LOS) magnetic fields, SDO/HMI has full-disk **vector magnetograph** capability.

주요 개발 이정표:

Key development milestones:

| 시기 / Date | 이벤트 / Event |
|---|---|
| 2000년 11월 | Science Definition Team 구성 (Hathaway 위원장) |
| 2002년 8월 | HMI/EVE Science Investigation Teams 선정 |
| 2003년 11월 | AIA Science Investigation Team 선정 |
| 2008년 9월 | 제작 및 시험 완료 |
| 2010년 2월 11일 | Atlas V 401로 발사 (KSC) |
| 2010년 5월 1일 | 과학 운용 시작 |

| 2000 November | Science Definition Team formed (Hathaway, chair) |
| 2002 August | HMI/EVE Science Investigation Teams selected |
| 2003 November | AIA Science Investigation Team selected |
| 2008 September | Build and test completed |
| 2010 February 11 | Launch on Atlas V 401 (KSC) |
| 2010 May 1 | Science operations began |

Atlas V 발사체의 지연으로 인해 원래 일정보다 늦게 발사되었다. SDO의 **네 가지 고유 특징**이 강조된다: (i) 지속적인 고속 데이터 전송(sustained high data rate), (ii) 자동 파이프라인을 갖춘 연속 다운링크(continuous downlink with automated pipeline), (iii) 극도로 정밀한 지향(extremely accurate pointing), (iv) 높은 듀티 사이클의 장기 미션 수명(long mission life with high duty cycle).

Launch was delayed from the original schedule due to Atlas V delays. Four **unique features** of SDO are highlighted: (i) sustained high data rate, (ii) continuous downlink with automated pipeline, (iii) extremely accurate pointing, (iv) long mission life with high duty cycle.

소모품이 없다는 점이 특히 중요하다 — 추진제 수명만으로도 **100년 이상** 운용 가능하며, 이는 미션 수명이 기기 성능 저하에 의해서만 제한됨을 의미한다.

The absence of consumables is particularly notable — propellant lifetime alone allows operation for **over 100 years**, meaning mission lifetime is limited only by instrument degradation.

### §4 Spacecraft Summary / 우주선 개요 (p. 7)

SDO 우주선의 핵심 사양:

SDO spacecraft key specifications:

| 항목 / Parameter | 값 / Value |
|---|---|
| 안정화 방식 / Stabilization | 3축 안정화(three-axis stabilized), 완전 이중화(fully redundant) |
| 총 질량 / Total mass | ~3,000 kg |
| 기기 질량 / Instrument mass | ~300 kg |
| 우주선 질량 / S/C mass | ~1,300 kg |
| 연료 질량 / Fuel mass | ~1,400 kg |
| 크기 / Dimensions | 4.7 m (태양 방향) × 2.2 m (측면) |
| 태양 전지판 / Solar arrays | 6.6 m², 1,500 W |
| 다운링크 / Downlink | Ka-band 26 GHz, 150 Mbit/s (130 science + 20 overhead) |
| HGA | 2개, 궤도당 1회전하여 White Sands 추적 |
| MOC | GSFC (Goddard Space Flight Center) |
| 지상국 / Ground station | White Sands, NM |

우주선의 형상은 "**Homeplate**" 형태로, 태양 전지판이 고이득 안테나(HGA)의 시야를 차단하지 않도록 설계되었다. 두 개의 HGA는 궤도당 한 번씩 회전하여 White Sands 지상국을 지속적으로 추적한다. 이 설계는 150 Mbit/s의 연속 다운링크를 가능케 하는 핵심 요소이다.

The spacecraft has a "**Homeplate**" shape, designed so that the solar panels do not block the High-Gain Antenna (HGA) field of view. Two HGAs rotate once per orbit to continuously track the White Sands ground station. This design is the key enabler of the 150 Mbit/s continuous downlink.

### §5 Science Goals / 과학 목표 (p. 8)

SDO의 과학 목표는 세 가지 수준의 표로 정리된다:

SDO's science goals are organized into three levels of tables:

**Table 1: 7개 Level-1 과학 질문 / 7 Level-1 Science Questions**

1. 태양주기의 기원과 메커니즘은 무엇인가? / What is the mechanism of the solar cycle?
2. 활동영역(active region)의 자기 플럭스는 어떻게 생성·분산되는가? / How is active region magnetic flux generated and dispersed?
3. 자기 재결합(magnetic reconnection)은 어떻게 발생하는가? / How does magnetic reconnection occur?
4. EUV 복사(irradiance)의 변동 원인은 무엇인가? / What causes EUV irradiance variations?
5. CME와 플레어를 발생시키는 자기 구조는 무엇인가? / What magnetic configurations produce CMEs and flares?
6. 태양풍(solar wind)의 예측이 가능한가? / Can the solar wind be predicted?
7. 우주 기상 예보가 가능한가? / Is space weather forecasting possible?

모든 7개 질문은 **자기장(magnetic field)**과 그 변동성을 중심으로 구성되어 있다. 이는 HMI의 벡터 자기장 관측이 SDO 과학의 핵심임을 반영한다.

All 7 questions center on the **magnetic field** and its variability. This reflects that HMI's vector magnetograph observations are central to SDO science.

**Table 2**: 5개 기기 과학 목표(instrument science objectives). **Table 3**: 5개 측정 목표(measurement objectives). 이들은 Level-1 질문을 구체적인 관측 요구사항으로 분해한 것이다.

**Table 2**: 5 instrument science objectives. **Table 3**: 5 measurement objectives. These decompose the Level-1 questions into specific observational requirements.

### §6 Science Investigation Teams / 과학 조사 팀 (p. 9)

세 기기의 핵심 관측 능력:

Core observational capabilities of the three instruments:

**Table 4: SDO 기기 사양 / SDO Instrument Specifications**

| 기기 / Instrument | 관측 내용 / Measurement | 핵심 사양 / Key Specs |
|---|---|---|
| **AIA** | 고속 풀디스크 EUV/UV 영상 | 10채널, 4096², 0.6"/pixel, 12s cadence |
| **HMI** | Dopplergram, LOS/벡터 자기장 | 4096², 1"/pixel, Fe I 617.3 nm, 45/135s cadence |
| **EVE** | EUV 스펙트럼 복사 | 6.5–105 nm, 0.1 nm 분해능, 10s cadence |

#### §6.1 AIA / 대기 영상 장치 (p. 9)

AIA는 **4대의 독립 망원경**으로 구성되며, 40'(arcmin) 시야로 태양 전면을 영상화한다. 공간 분해능 **1.2"**(Nyquist 기준, 0.6"/pixel), **10개 파장 채널**(EUV 7, UV 2, 가시광 1), 12초 케이던스. PI: Alan Title(LMSAL), 제작: Lockheed Martin Solar & Astrophysics Laboratory.

AIA consists of **4 independent telescopes**, imaging the full Sun with a 40' (arcmin) FOV. Spatial resolution is **1.2"** (Nyquist, 0.6"/pixel), with **10 wavelength channels** (7 EUV, 2 UV, 1 visible) at 12-second cadence. PI: Alan Title (LMSAL), built by Lockheed Martin Solar & Astrophysics Laboratory.

#### §6.2 EVE / 극자외선 변동 실험 (p. 9)

EVE는 세 개의 하위 기기로 구성된다:

EVE consists of three sub-instruments:

- **MEGS** (Multiple EUV Grating Spectrograph): 회절격자 분광기, 파장범위 6.5–105 nm, 분광 분해능 0.1 nm
- **ESP** (EUV SpectroPhotometer): 방사계(radiometer), 광대역(broadband) 측정
- **SAM** (Solar Aspect Monitor): 핀홀 카메라, 0.1–7 nm

- **MEGS** (Multiple EUV Grating Spectrograph): grating spectrometers, wavelength range 6.5–105 nm, spectral resolution 0.1 nm
- **ESP** (EUV SpectroPhotometer): radiometers, broadband measurements
- **SAM** (Solar Aspect Monitor): pinhole camera, 0.1–7 nm

케이던스는 **10초**이며, EVE의 절대 복사 보정은 AIA 영상 데이터의 방사 측정적 기준(radiometric anchor)을 제공한다. PI: Tom Woods, 제작: LASP (Laboratory for Atmospheric and Space Physics), CU Boulder.

Cadence is **10 seconds**, and EVE's absolute irradiance calibration provides the radiometric anchor for AIA imaging data. PI: Tom Woods, built by LASP (Laboratory for Atmospheric and Space Physics), CU Boulder.

#### §6.3 HMI / 태양진동 및 자기장 관측기 (p. 9)

HMI는 **Michelson 간섭계 + 조절 가능 Lyot 필터**를 사용하여 Fe I 617.3 nm 흡수선을 관측한다. 핵심 데이터 산출물:

HMI uses a **Michelson interferometer + tunable Lyot filter** to observe the Fe I 617.3 nm absorption line. Key data products:

- **풀디스크 Dopplergram**: 45초 케이던스 — 태양진동(helioseismology) 연구
- **LOS 자기장(magnetogram)**: 45초 케이던스 — 시선방향 자기장
- **벡터 자기장(vector magnetogram)**: 135초 케이던스 — 3차원 자기장 벡터

- **Full-disk Dopplergrams**: 45 s cadence — for helioseismology
- **LOS magnetograms**: 45 s cadence — line-of-sight magnetic field
- **Vector magnetograms**: 135 s cadence — 3D magnetic field vector

4096×4096 CCD, 1"/pixel. PI: Phil Scherrer, 제작: Stanford University. SOHO/MDI의 직접적 후계 기기이다. MDI는 LOS 자기장만 측정할 수 있었으나, HMI는 풀디스크 벡터 자기장 관측이 가능하여 광구(photosphere) 자기장의 3차원 구조를 직접 측정할 수 있다. 이는 SDO 과학의 7개 Level-1 질문 모두에 핵심적인 데이터를 제공한다.

4096×4096 CCD, 1"/pixel. PI: Phil Scherrer, built at Stanford University. Direct successor to SOHO/MDI. While MDI could only measure LOS magnetic fields, HMI enables full-disk vector magnetograph observations, directly measuring the 3D structure of photospheric magnetic fields. This provides critical data for all 7 Level-1 science questions.

### §7–8 Operations and Ground System / 운용 및 지상 시스템 (pp. 9–12)

SDO의 궤도 선택이 미션 설계의 핵심이다:

SDO's orbit selection is central to the mission design:

**지구정지 경사궤도(GEO-synchronous inclined orbit)의 장점:**

1. **연속 접촉(>95% duty)**: 단일 지상국으로 거의 24시간 연속 통신 가능. LEO 위성(TRACE, Hinode)이 궤도당 수분의 지상국 접촉 시간만 가진 것과 대비.
2. **고속 데이터 전송**: 150 Mbit/s Ka-band 다운링크. SOHO(L1 궤도)의 0.04 Mbit/s 대비 **3,750배** 향상.
3. **안정적 열환경**: GEO 궤도는 LEO보다 열적으로 안정적이어서 기기 보정에 유리.

**Advantages of GEO-synchronous inclined orbit:**

1. **Continuous contact (>95% duty)**: Nearly 24-hour continuous communication possible with a single ground station. Contrasts with LEO satellites (TRACE, Hinode) having only minutes of ground station contact per orbit.
2. **High-speed data transfer**: 150 Mbit/s Ka-band downlink. A **3,750×** improvement over SOHO (L1 orbit) at 0.04 Mbit/s.
3. **Stable thermal environment**: GEO orbit is thermally more stable than LEO, favorable for instrument calibration.

**단점:**

- **식 시즌(eclipse seasons)**: 춘·추분 전후 약 **3주간**, 하루 최대 **72분**의 식(eclipse)이 발생. 이 기간 동안 배터리 전력으로 운용하며, 기기 온도 변화에 주의가 필요.

**Disadvantages:**

- **Eclipse seasons**: Approximately **3 weeks** around equinoxes, with up to **72 minutes** of eclipse per day. During these periods, the spacecraft operates on battery power, requiring attention to instrument temperature variations.

지상 시스템 구성:

Ground system architecture:

- **White Sands 지상국**: 2대의 전용 **18 m Ka-band 안테나**. SDO 전용이므로 DSN(Deep Space Network)과 공유하지 않음.
- **MOC (Mission Operations Center)**: GSFC에 위치. 우주선 명령 및 건강 모니터링(health monitoring) 담당.
- **JSOC (Joint Science Operations Center)**: Stanford 대학교에 위치. 과학 데이터 처리, 아카이빙, 배포를 총괄. DRMS(Data Record Management System) 사용.

- **White Sands ground station**: Two dedicated **18 m Ka-band antennas**. Dedicated to SDO, not shared with the DSN (Deep Space Network).
- **MOC (Mission Operations Center)**: Located at GSFC. Responsible for spacecraft commanding and health monitoring.
- **JSOC (Joint Science Operations Center)**: Located at Stanford University. Manages science data processing, archiving, and distribution. Uses DRMS (Data Record Management System).

데이터 흐름: 우주선 → White Sands (Ka-band) → GSFC (MOC) → Stanford (JSOC) → 전 세계 공개.

Data flow: Spacecraft → White Sands (Ka-band) → GSFC (MOC) → Stanford (JSOC) → worldwide public access.

### §9 Conclusion / 결론 (pp. 12–13)

AIA의 첫 영상은 **2010년 3월 27일**에 도어 개방 후 취득되었다. 모든 SDO 데이터는 **무료로 공개**되며, **95% 이상**의 관측 듀티 사이클을 달성하고 있다. SDO는 2010년 이후 태양 물리학 연구의 표준 데이터 소스로 자리 잡았으며, 2025년 현재까지도 운용 중이다.

AIA's first images were obtained on **27 March 2010** after door opening. All SDO data is **freely available**, and an observing duty cycle exceeding **95%** is achieved. SDO has established itself as the standard data source for solar physics research since 2010 and remains operational as of 2025.

---

## 3. Key Takeaways / 핵심 시사점

1. **GEO-sync 궤도는 데이터율을 위한 선택 / GEO-sync orbit chosen for data rate** — SDO의 경사 지구정지궤도는 150 Mbit/s의 연속 다운링크를 가능케 한다. 이는 SOHO(L1 궤도, 0.04 Mbit/s) 대비 3,750배 향상으로, 고해상도 풀디스크 영상의 연속 전송이라는 SDO의 핵심 요구사항을 충족한다. 궤도 선택이 전체 미션 설계를 결정한 사례이다.
   SDO's inclined geosynchronous orbit enables 150 Mbit/s continuous downlink — a 3,750× improvement over SOHO (L1 orbit, 0.04 Mbit/s). This satisfies SDO's core requirement of continuous transmission of high-resolution full-disk images. A case where orbit selection determined the entire mission design.

2. **3개 기기로 SOHO의 12개를 대체 / 3 instruments replace SOHO's 12** — SDO는 AIA, HMI, EVE 세 기기만으로 SOHO의 12개 기기보다 훨씬 뛰어난 과학 성능을 제공한다. 이는 2000년대의 검출기 기술(back-thinned CCD, 4096²), 광학 기술(다층막 코팅), 데이터 처리 기술의 발전을 반영한다.
   SDO delivers far superior science performance with just three instruments (AIA, HMI, EVE) compared to SOHO's 12. This reflects advances in 2000s detector technology (back-thinned CCDs, 4096²), optical technology (multilayer coatings), and data processing capabilities.

3. **소모품 없음 → 100년 이상 추진제 수명 / No consumables → 100+ year propellant lifetime** — SDO는 소모품 없이 설계되어 추진제만으로 100년 이상 운용 가능하다. 미션 수명은 기기 성능 저하(주로 CCD 방사선 손상, 다층막 코팅 열화)에 의해서만 제한된다. 이는 장기 태양 모니터링에 이상적인 설계이다.
   SDO was designed without consumables, enabling over 100 years of operation on propellant alone. Mission lifetime is limited only by instrument degradation (primarily CCD radiation damage, multilayer coating degradation). This is an ideal design for long-term solar monitoring.

4. **하루 1.5 TB — 발사 당시 NASA 최대 / 1.5 TB/day — NASA's largest at launch** — SDO는 발사 시점(2010년)에서 NASA 과학 미션 중 최대의 데이터 생산량을 기록했다. 5년 주미션 기간 동안 3–4 PB의 원시 데이터가 예상되었으며, 이 데이터 관리를 위해 Stanford JSOC의 전용 인프라가 구축되었다.
   SDO recorded the largest data output of any NASA science mission at launch (2010). Over the 5-year prime mission, 3–4 PB of raw data was projected, requiring dedicated infrastructure at Stanford's JSOC for data management.

5. **7개 Level-1 과학 질문은 자기장 중심 / 7 Level-1 science questions center on magnetic field** — SDO의 모든 과학 질문은 자기장의 생성, 진화, 소멸 과정을 중심으로 구성되어 있다. 이는 HMI의 벡터 자기장 관측이 SDO 과학의 근간임을 의미하며, AIA와 EVE는 자기장 변동의 코로나/복사 응답을 측정하는 역할을 한다.
   All of SDO's science questions are organized around the generation, evolution, and dissipation of magnetic fields. This means HMI's vector magnetograph observations are the foundation of SDO science, with AIA and EVE measuring the coronal/irradiance response to magnetic field variations.

6. **HMI 벡터 자기장은 MDI 대비 핵심 업그레이드 / HMI vector magnetograph is key upgrade over MDI** — SOHO/MDI는 시선방향(LOS) 자기장만 측정할 수 있었으나, HMI는 풀디스크 벡터 자기장을 135초 케이던스로 관측한다. 이를 통해 광구 자기장의 3차원 구조를 직접 측정할 수 있으며, 비강제 자기장 외삽(NLFFF extrapolation)의 경계 조건으로 사용된다.
   SOHO/MDI could only measure line-of-sight (LOS) magnetic fields, but HMI observes full-disk vector magnetic fields at 135-second cadence. This enables direct measurement of the 3D structure of photospheric magnetic fields, used as boundary conditions for non-linear force-free field (NLFFF) extrapolation.

7. **EVE는 AIA 영상과 절대 복사 보정 사이의 연결 / EVE bridges AIA imaging and absolute radiometric calibration** — EVE의 고정밀 EUV 스펙트럼 복사 측정은 AIA 영상 데이터의 방사 측정적 기준(radiometric anchor)을 제공한다. 이는 AIA의 다층막 코팅 열화를 추적하고, 장기적 복사 변동 연구에 필수적이다.
   EVE's high-precision EUV spectral irradiance measurements provide the radiometric anchor for AIA imaging data. This is essential for tracking AIA multilayer coating degradation and for long-term irradiance variation studies.

8. **전용 Ka-band 지상국 — DSN 비사용 / Dedicated Ka-band ground station — no DSN sharing** — White Sands의 전용 18 m 안테나 2대는 SDO만을 위해 운용된다. DSN을 사용하지 않으므로 다른 행성 탐사 미션과의 시간 경쟁이 없으며, 이는 95% 이상의 듀티 사이클 달성에 필수적이다.
   Two dedicated 18 m antennas at White Sands operate exclusively for SDO. Without using the DSN, there is no time competition with planetary exploration missions, which is essential for achieving the >95% duty cycle.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 GEO-synchronous Orbit Altitude / 지구정지궤도 고도

지구정지궤도 반경은 케플러 제3법칙으로부터:

The geosynchronous orbital radius follows from Kepler's third law:

$$r = \left(\frac{G M T^2}{4\pi^2}\right)^{1/3}$$

여기서 $G = 6.674 \times 10^{-11}\;\text{N m}^2\text{ kg}^{-2}$, $M = 5.972 \times 10^{24}\;\text{kg}$ (지구 질량), $T = 86{,}400\;\text{s}$ (항성 주기):

Where $G = 6.674 \times 10^{-11}\;\text{N m}^2\text{ kg}^{-2}$, $M = 5.972 \times 10^{24}\;\text{kg}$ (Earth mass), $T = 86{,}400\;\text{s}$ (sidereal period):

$$r \approx 42{,}164\;\text{km} \quad \Rightarrow \quad h = r - R_{\oplus} \approx 42{,}164 - 6{,}378 \approx 35{,}786\;\text{km}$$

SDO는 이 고도에서 경사각 28.5°(KSC 발사 위도에 의해 결정)로 궤도를 돈다. 경사각이 0°가 아니므로 정지(geostationary)가 아닌 **정지동기(geosynchronous)** 궤도이다 — 지상에서 볼 때 8자(figure-8) 궤적을 그린다.

SDO orbits at this altitude with 28.5° inclination (determined by KSC launch latitude). Since the inclination is not 0°, it is a **geosynchronous** (not geostationary) orbit — it traces a figure-8 ground track as seen from Earth.

### 4.2 Data Rate and Volume / 데이터 전송률 및 데이터량

SDO 전체 다운링크:

SDO total downlink:

$$R_{\text{total}} = 150\;\text{Mbit/s} = 130\;\text{Mbit/s (science)} + 20\;\text{Mbit/s (overhead)}$$

일일 데이터량(95% 듀티 사이클 적용):

Daily data volume (with 95% duty cycle):

$$V_{\text{day}} = 150\;\text{Mbit/s} \times 86{,}400\;\text{s} \times 0.95 = 1.231 \times 10^{13}\;\text{bit} \approx 1.5\;\text{TB/day}$$

5년 주미션 총 데이터량:

Total data volume over 5-year prime mission:

$$V_{5\text{yr}} = 1.5\;\text{TB/day} \times 365.25\;\text{days/yr} \times 5\;\text{yr} \approx 2.7\;\text{PB}$$

논문에서는 원시 데이터(미압축) 기준 **3–4 PB**로 제시하며, 이는 Rice 압축 전의 데이터량에 해당한다.

The paper states **3–4 PB** based on raw (uncompressed) data, which corresponds to the data volume before Rice compression.

### 4.3 SOHO vs SDO Data Rate Comparison / SOHO vs SDO 데이터율 비교

$$\frac{R_{\text{SDO}}}{R_{\text{SOHO}}} = \frac{150\;\text{Mbit/s}}{0.04\;\text{Mbit/s}} = 3{,}750$$

이 3,750배 향상은 두 가지 요인의 조합이다: (1) Ka-band(26 GHz) vs S-band(2.3 GHz), (2) GEO-sync(35,786 km) vs L1(1.5 × 10⁶ km) 거리 차이. 전파 신호의 전력은 거리의 제곱에 반비례하므로:

This 3,750× improvement combines two factors: (1) Ka-band (26 GHz) vs S-band (2.3 GHz), (2) GEO-sync (35,786 km) vs L1 (1.5 × 10⁶ km) distance difference. Since radio signal power is inversely proportional to the square of distance:

$$\frac{P_{\text{GEO}}}{P_{\text{L1}}} \propto \left(\frac{d_{\text{L1}}}{d_{\text{GEO}}}\right)^2 = \left(\frac{1.5 \times 10^6}{35{,}786}\right)^2 \approx 1{,}760$$

거리 차이만으로도 약 1,760배의 링크 예산(link budget) 이점이 있으며, Ka-band의 높은 대역폭이 나머지 향상을 제공한다.

Distance difference alone provides approximately 1,760× link budget advantage, with Ka-band's higher bandwidth providing the remaining improvement.

### 4.4 Eclipse Duration / 식 지속 시간

GEO-sync 궤도에서 춘·추분 전후 식 시즌:

Eclipse seasons around equinoxes in GEO-sync orbit:

- 식 시즌 기간 / Season duration: 약 **3주** (~21 days)
- 최대 식 지속 시간 / Maximum eclipse: **72분/일** (day of equinox)
- 식 기간 비율 / Eclipse fraction: $72/1440 = 5\%$

이 5%의 식 시간이 SDO의 관측 듀티 사이클을 95%로 제한하는 주요 요인이다.

This 5% eclipse time is the primary factor limiting SDO's observing duty cycle to 95%.

### 4.5 Worked Example: SDO Daily Image Count / 수치 예제: SDO 일일 영상 수

AIA의 경우:

For AIA:

$$N_{\text{AIA}} = \frac{86{,}400\;\text{s}}{12\;\text{s}} \times 8\;\text{channels} \times 0.95 = 54{,}720\;\text{images/day}$$

HMI의 경우 (Dopplergram + LOS + vector):

For HMI (Dopplergrams + LOS + vector):

$$N_{\text{HMI,Dopp+LOS}} = \frac{86{,}400}{45} \times 0.95 \approx 1{,}824\;\text{images/day (each type)}$$

$$N_{\text{HMI,vector}} = \frac{86{,}400}{135} \times 0.95 \approx 608\;\text{images/day}$$

EVE의 경우:

For EVE:

$$N_{\text{EVE}} = \frac{86{,}400}{10} \times 0.95 \approx 8{,}208\;\text{spectra/day}$$

총합: AIA ~55,000 + HMI ~4,000 + EVE ~8,000 ≈ 약 67,000 데이터 산출물/일. 논문에서는 ~150,000 영상 + 9,000 스펙트럼을 언급하는데, 이는 HMI의 다중 필터그램(filtergram)과 AIA의 추가 보정 영상을 포함한 수치이다.

Total: AIA ~55,000 + HMI ~4,000 + EVE ~8,000 ≈ ~67,000 data products/day. The paper mentions ~150,000 images + 9,000 spectra, which includes HMI's multiple filtergrams and AIA's additional calibration images.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1991       1995       1998       2006       2010       2018       2025
  |          |          |          |          |          |          |
Yohkoh     SOHO      TRACE    STEREO      SDO       PSP       SDO
(SXT)    (12 inst)  (EUV/UV)  (twin)   (AIA/HMI/  (in situ)  still
                                         EVE)               operating
  |          |          |          |          |          |          |
  ·----------·----------·----------·----------·----------·----------·
  
  First X-ray    L1 continuous   High-res    Multi-view   Full-disk    Inner
  full-disk      solar obs       EUV from    3D corona    + high-res   heliosphere
  imaging        + helioseism    space                    + high-rate  exploration
  
Key Milestones:
  
1991  Yohkoh/SXT — first continuous full-disk soft X-ray imaging
1995  SOHO — L1 orbit, 12 instruments, MDI helioseismology, 0.04 Mbit/s
1998  TRACE — sub-arcsecond EUV, 8.5' FOV, LEO
2000  SDO SDT formed (Hathaway, chair)
2002  HMI/EVE SITs selected
2003  AIA SIT selected
2006  STEREO — twin spacecraft, 3D coronal imaging
2010  SDO launched (Feb 11), first light (Mar 27), science ops (May 1) ← THIS PAPER
2012  Paper published in Solar Physics special issue (Vol. 275)
2018  Parker Solar Probe launched — in situ inner heliosphere
2020  Solar Orbiter launched — close-up EUV + in situ
2025  SDO still operational, 15+ years of continuous data
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| 논문 / Paper | 연결 / Connection | 관련성 / Relevance |
|---|---|---|
| **#8 Domingo et al. (1995) — SOHO** | SDO의 직접적 선행 미션. SOHO의 L1 궤도 연속 관측 개념을 GEO-sync에서 3,750배 높은 데이터율로 계승. SOHO의 12개 기기를 SDO의 3개로 대체. / Direct predecessor mission. SDO inherits SOHO's L1 continuous observation concept with 3,750× higher data rate from GEO-sync. SOHO's 12 instruments replaced by SDO's 3. | 높음 / High |
| **#12 Lemen et al. (2012) — AIA** | SDO의 세 기기 중 하나인 AIA의 상세 기기 논문. 본 논문(Pesnell 2012)은 미션 개요이고, Lemen 2012는 AIA 기기에 대한 심층 기술 논문. 동일 Solar Physics 특집호(Vol. 275)에 수록. / Detailed instrument paper for AIA, one of SDO's three instruments. This paper (Pesnell 2012) is the mission overview, while Lemen 2012 provides in-depth technical details of AIA. Published in the same Solar Physics special issue (Vol. 275). | 높음 / High |
| **#13 Scherrer et al. (2012) — HMI** | SDO의 동반 기기. HMI 벡터 자기장은 SDO 7개 Level-1 과학 질문의 핵심 데이터. MDI(SOHO)의 후계. / SDO companion instrument. HMI vector magnetograph data is central to all 7 SDO Level-1 science questions. Successor to MDI (SOHO). | 높음 / High |
| **#9 Delaboudinière et al. (1995) — EIT** | AIA의 선행 기기. EIT의 4채널, 1024², 12분 케이던스 설계를 AIA가 10채널, 4096², 12초로 대폭 향상. 정보율 400–22,000배 향상. / AIA's predecessor instrument. AIA massively improved EIT's 4-channel, 1024², 12-min cadence design to 10-channel, 4096², 12-second. Information rate improvement of 400–22,000×. | 높음 / High |
| **#11 Handy et al. (1999) — TRACE** | AIA의 가장 직접적 기술적 선행자. Cassegrain 설계, AEC, GT/ISS를 AIA가 계승. TRACE의 8.5' FOV → AIA 41'. / AIA's most direct technical predecessor. AIA inherited Cassegrain design, AEC, GT/ISS from TRACE. TRACE's 8.5' FOV → AIA 41'. | 높음 / High |
| **#5 Harvey et al. (1996) — GONG** | 지상 태양진동 관측 네트워크. HMI의 우주 기반 태양진동 관측은 GONG의 지상 관측과 상호 보완적. 대기 시상(seeing) 제한 없는 HMI 데이터가 GONG의 분해능을 크게 초과. / Ground-based helioseismology network. HMI's space-based helioseismology observations complement GONG's ground observations. HMI data without atmospheric seeing limitations far exceeds GONG's resolution. | 중간 / Medium |
| **#4 Goode et al. (2012) — BBSO/NST** | 지상 고분해능 태양 관측. SDO/AIA의 0.6" 픽셀은 지상 적응광학 관측(~0.1")에 비해 분해능이 낮지만, 풀디스크 연속 관측에서 압도적 우위. 상호 보완적 관계. / Ground-based high-resolution solar observation. SDO/AIA's 0.6" pixel is lower resolution than ground AO observations (~0.1"), but overwhelmingly superior in full-disk continuous observation. Complementary relationship. | 중간 / Medium |

---

## 7. References / 참고문헌

- Pesnell, W.D., Thompson, B.J., Chamberlin, P.C., "The Solar Dynamics Observatory (SDO)," Solar Physics, 275, 3–15, 2012. [DOI: 10.1007/s11207-011-9841-3](https://doi.org/10.1007/s11207-011-9841-3)
- Lemen, J.R., et al., "The Atmospheric Imaging Assembly (AIA) on the Solar Dynamics Observatory (SDO)," Solar Physics, 275, 17–40, 2012. [DOI: 10.1007/s11207-011-9776-8](https://doi.org/10.1007/s11207-011-9776-8)
- Scherrer, P.H., et al., "The Helioseismic and Magnetic Imager (HMI) Investigation for the Solar Dynamics Observatory (SDO)," Solar Physics, 275, 207–227, 2012. [DOI: 10.1007/s11207-011-9834-2](https://doi.org/10.1007/s11207-011-9834-2)
- Woods, T.N., et al., "Extreme Ultraviolet Variability Experiment (EVE) on the Solar Dynamics Observatory (SDO): Overview of Science Objectives, Instrument Design, Data Products, and Model Developments," Solar Physics, 275, 115–143, 2012. [DOI: 10.1007/s11207-009-9487-6](https://doi.org/10.1007/s11207-009-9487-6)
- Domingo, V., Fleck, B., Poland, A.I., "The SOHO Mission: An Overview," Solar Physics, 162, 1–37, 1995. [DOI: 10.1007/BF00733425](https://doi.org/10.1007/BF00733425)
- Delaboudinière, J.-P., et al., "EIT: Extreme-Ultraviolet Imaging Telescope for the SOHO Mission," Solar Physics, 162, 291–312, 1995. [DOI: 10.1007/BF00733432](https://doi.org/10.1007/BF00733432)
- Handy, B.N., et al., "The Transition Region and Coronal Explorer," Solar Physics, 187, 229–260, 1999. [DOI: 10.1023/A:1005166902804](https://doi.org/10.1023/A:1005166902804)
- Scherrer, P.H., et al., "The Solar Oscillations Investigation — Michelson Doppler Imager," Solar Physics, 162, 129–188, 1995. [DOI: 10.1007/BF00733429](https://doi.org/10.1007/BF00733429)
- Harvey, J.W., et al., "The Global Oscillation Network Group (GONG) Project," Science, 272, 1284–1286, 1996. [DOI: 10.1126/science.272.5266.1284](https://doi.org/10.1126/science.272.5266.1284)
