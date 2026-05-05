---
title: "Pre-reading Briefing — The Solar Dynamics Observatory (SDO)"
paper: "Pesnell, Thompson & Chamberlin 2012, Sol. Phys. 275, 3"
date: 2026-04-27
topic: Solar_Physics
tags: [SDO, LWS, HMI, AIA, EVE, mission, space-weather]
---

# Pre-reading Briefing / 사전 브리핑

## Why this paper matters / 이 논문의 중요성

**EN.** The *Solar Dynamics Observatory* (SDO) is the cornerstone mission of NASA's *Living With a Star* (LWS) program and has produced the largest, highest-cadence, multi-wavelength dataset in solar physics history. This paper is the "mission paper" that describes the spacecraft, its three instrument suites (HMI, AIA, EVE), the inclined geosynchronous orbit, the 130 Mbps Ka-band downlink, and the science questions that drove the design. Almost every SDO-era paper cites it.

**KR.** *Solar Dynamics Observatory* (SDO)는 NASA *Living With a Star* (LWS) 프로그램의 핵심 미션이며, 태양 물리학 역사상 가장 크고, 가장 높은 시간 해상도로, 다파장 관측 데이터를 생산해 왔습니다. 본 논문은 우주선, 세 가지 관측기군(HMI·AIA·EVE), 경사 지구정지궤도, 130 Mbps Ka-band 다운링크, 그리고 설계를 주도한 과학 질문을 기술하는 "미션 논문"입니다. SDO 시대의 거의 모든 논문이 본 논문을 인용합니다.

## Prerequisites / 선수 지식

| Item / 항목 | English | 한국어 |
|---|---|---|
| Solar atmosphere layers | Photosphere, chromosphere, transition region, corona; characteristic temperatures 6 000 K → 3×10⁶ K. | 광구, 채층, 전이층, 코로나; 특성 온도 6 000 K → 3×10⁶ K. |
| EUV emission | Highly ionized lines (Fe IX–XXIV) emitted in the optically thin corona; dominant cooling 1–122 nm. | 광학적으로 얇은 코로나에서 방출되는 고이온화 선(Fe IX–XXIV); 1–122 nm 주요 냉각. |
| Helioseismology | Doppler-shift oscillations (p-modes) used to infer subsurface flows and structure. | 도플러 진동(p-mode)을 이용해 표면 아래 유동·구조를 추정. |
| Zeeman effect | Spectral-line splitting in magnetic field; basis for HMI vector magnetograms. | 자기장에 의한 분광선 분리; HMI 벡터 자기그림의 기초. |
| Geosynchronous orbit | 35 786 km altitude, 24-h period; "inclined GEO" tilts the orbit so the satellite traces a figure-8. | 고도 35 786 km, 주기 24h; "경사 GEO"는 궤도를 기울여 8자 궤적을 그림. |
| Ka-band telemetry | ~26 GHz downlink supporting 100+ Mbps rates. | ~26 GHz 다운링크, 100 Mbps급 지원. |

## Key vocabulary / 핵심 용어

- **LWS (Living With a Star)**: NASA program connecting heliophysics to societal impact / 태양과 사회 영향을 잇는 NASA 프로그램.
- **AIA (Atmospheric Imaging Assembly)**: 4-telescope EUV/UV imager, 10 channels, 1.2″ resolution / 4망원경 EUV/UV 영상기, 10채널, 1.2″ 해상도.
- **HMI (Helioseismic & Magnetic Imager)**: Full-disk Dopplergrams + LOS + vector magnetograms / 전면 도플러그램·LOS·벡터 자기그림.
- **EVE (EUV Variability Experiment)**: Solar EUV irradiance 0.1–105 nm + Ly-α / 태양 EUV 복사조도 0.1–105 nm + Ly-α.
- **JSOC**: Joint Science Operations Center at Stanford, archives HMI/AIA / Stanford 공동 과학 운영 센터, HMI·AIA 보관.
- **Cadence**: Time between successive observations; AIA 12 s, HMI 45 s LOS / 연속 관측 간 시간; AIA 12초, HMI 45초.
- **Inclined geosynchronous orbit**: 28.5° inclination, 35 800 km altitude / 28.5° 경사, 35 800 km 고도.

## Pre-reading Q&A / 사전 Q&A

**Q1. Why a geosynchronous orbit instead of polar Sun-synchronous LEO?**
- EN. SDO produces ~1.5 TB/day. No space-qualified recorder could store that volume; continuous Ka-band streaming to a single dedicated ground station was needed. Inclined GEO gives 24-h contact with one station (White Sands) at the cost of two annual eclipse seasons and higher radiation dose.
- KR. SDO는 일 ~1.5 TB를 생산합니다. 그만한 용량을 저장할 우주용 기록기가 없으므로, 단일 전용 지상국으로의 연속 Ka-band 스트리밍이 필요했습니다. 경사 GEO는 한 지상국(White Sands)과의 24시간 접속을 제공하지만, 연 2회 식 시즌과 더 높은 방사선량을 감수합니다.

**Q2. Why three instruments?**
- EN. Each addresses a different layer/process: HMI = magnetic field (origin), AIA = atmospheric response (transport/release), EVE = EUV irradiance (Earth impact). Together they trace the lifecycle of the magnetic field.
- KR. 각 기기가 다른 층/과정을 담당합니다: HMI = 자기장(기원), AIA = 대기 반응(수송·방출), EVE = EUV 복사조도(지구 영향). 합쳐서 자기장 일생을 추적합니다.

**Q3. What is "1.5 TB/day" composed of?**
- EN. ~150 000 high-resolution full-Sun images and 9 000 EUV spectra per day. AIA dominates the bandwidth (4 telescopes × ~12 s cadence × 4096² × 16 bit).
- KR. 일 약 15만 장 전면 영상 + 9 000개 EUV 스펙트럼. AIA가 대역폭을 주도(4 망원경 × ~12s × 4096² × 16 bit).

**Q4. Why 130 Mbps?**
- EN. Sized to drain the onboard buffers in real time given AIA/HMI/EVE raw rates. The 150 Mbps Ka-band link reserves 20 Mbps for encoding overhead.
- KR. AIA·HMI·EVE 원시율을 실시간 비울 수 있도록 산정. 150 Mbps Ka-band 중 20 Mbps는 부호화 오버헤드.

**Q5. What science questions drove the requirements?**
- EN. The 7 SDT report questions (Hathaway et al. 2001): solar cycle drivers, AR flux concentration, reconnection topology, EUV irradiance origin, CME/flare initiation, solar-wind connection, predictability.
- KR. SDT 보고서의 7개 질문(Hathaway 외 2001): 태양주기 구동, AR 자속 집중, 재결합 위상, EUV 조도 기원, CME·플레어 점화, 태양풍 연결, 예측 가능성.

## How to read the paper / 읽기 전략

**EN.** Skim Sections 1–3 (mission context). Spend time on Section 4 (Spacecraft Summary) and Section 6 (Instrument SITs) — these are the most-cited tables. Section 8 (orbit) is short but conceptually rich: the trade study between LEO and GEO is a model of mission design. Section 7 (data capture) explains the 95%/72-day completeness rule that frames all future data products.

**KR.** 1–3장(미션 배경)은 빠르게 훑고, 4장(우주선 요약)과 6장(기기 SIT)에 시간을 쓰세요 — 가장 많이 인용되는 표입니다. 8장(궤도)은 짧지만 개념적으로 풍부 — LEO와 GEO 간 교환 분석은 미션 설계의 모범입니다. 7장(데이터 수집)은 향후 모든 데이터 산출물의 95%/72일 완전성 규칙을 설명합니다.
