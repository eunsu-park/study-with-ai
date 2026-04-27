---
title: "The Solar Dynamics Observatory (SDO)"
authors: "W. Dean Pesnell, B. J. Thompson, P. C. Chamberlin"
year: 2012
journal: "Solar Physics 275, 3–15"
doi: "10.1007/s11207-011-9841-3"
date: 2026-04-27
topic: Solar_Physics
paper_number: 33
tags: [SDO, LWS, HMI, AIA, EVE, mission, space-weather, helioseismology, EUV]
status: completed
---

# The Solar Dynamics Observatory (SDO)
# 태양역학관측위성(SDO)

## 1. Core Contribution / 핵심 기여

**EN.** This paper is the canonical mission overview for the *Solar Dynamics Observatory*, the first space-weather mission of NASA's *Living With a Star* program. SDO was launched on 11 February 2010 from Kennedy Space Center on an Atlas V (AV-021) and reached its operational inclined geosynchronous orbit (28.5° inclination, ~35 800 km altitude, longitude over the New Mexico ground station) by 16 March 2010, beginning science return on 1 May 2010. The paper specifies the spacecraft (3-axis stabilized, 3 000 kg launch mass, 1 500 W solar array, 4.7 m × 2.2 m bus, 6.6 m² array, 6.1 m antenna span), the three Science Investigation Teams — *Helioseismic and Magnetic Imager* (HMI; full-disk Dopplergrams, LOS magnetograms at 45 s, vector magnetograms at 12 min), *Atmospheric Imaging Assembly* (AIA; four telescopes, ten EUV/UV/visible channels, 1.2″ two-pixel resolution, ≥40′ field), and *EUV Variability Experiment* (EVE; spectral irradiance 0.1–105 nm + Ly-α, 0.1 nm resolution, 10 s cadence) — and the operational architecture: 130 Mbps science data plus 20 Mbps encoding overhead at Ka-band (~26 GHz), continuous downlink to a dedicated dual-antenna ground station at White Sands, ~1.5 TB/day of products, and 3–4 PB raw over the 5-year prime mission. The paper formalizes the seven Level-1 science questions (originally posed by Hathaway et al. 2001), the four instrument science objectives, and the five measurement objectives, which together bound every SDO observing program.

**KR.** 본 논문은 NASA *Living With a Star* 프로그램의 첫 우주기상 미션인 *Solar Dynamics Observatory*에 대한 정식 미션 개요입니다. SDO는 2010년 2월 11일 케네디 우주센터에서 Atlas V (AV-021)로 발사되어 2010년 3월 16일에 운용 경사 지구정지궤도(경사각 28.5°, 고도 약 35 800 km, 뉴멕시코 지상국 경도)에 진입했고, 2010년 5월 1일 과학 관측을 시작했습니다. 본 논문은 우주선(3축 안정화, 발사 질량 3 000 kg, 1 500 W 태양전지, 4.7 m × 2.2 m 본체, 6.6 m² 어레이, 안테나 폭 6.1 m), 세 개의 과학 연구 팀 — *Helioseismic and Magnetic Imager*(HMI; 전면 도플러그램, 45초 LOS 자기그림, 12분 벡터 자기그림), *Atmospheric Imaging Assembly*(AIA; 4 망원경, 10개 EUV/UV/가시 채널, 1.2″ 2픽셀 해상도, ≥40′ 시야), *EUV Variability Experiment*(EVE; 0.1–105 nm + Ly-α 분광 복사조도, 0.1 nm 분해능, 10초 시간 해상도) — 및 운용 구조: Ka-band(약 26 GHz)에서 130 Mbps 과학 데이터 + 20 Mbps 부호화 오버헤드, White Sands 전용 이중 안테나 지상국으로의 연속 다운링크, 약 1.5 TB/일, 5년 주 미션 동안 원시 3–4 PB를 명세합니다. 논문은 또한 SDO 모든 관측 계획의 경계를 정하는 7개 1단계 과학 질문(Hathaway 외 2001 최초 제기), 4개 기기 과학 목표, 5개 측정 목표를 공식화합니다.

## 2. Reading Notes / 읽기 노트

### 2.1 Section 1 — Preface: LWS and SDO (p. 4)
- **EN.** *Living With a Star* (LWS) is a strategic, system-of-systems heliophysics program: it bundles strategic missions (SDO, RBSP, Solar Probe Plus, Solar Orbiter), missions of opportunity (BARREL balloon array), targeted research, a space-environment testbed, and partnerships (e.g., ESA on Solar Orbiter). The unifying goal is "predictive capability" — moving heliophysics from descriptive science to operational forecasting. SDO is the first LWS mission; RBSP is the second.
- **KR.** *Living With a Star*(LWS)는 전략적 시스템-오브-시스템 태양·지구·우주 환경 프로그램입니다: 전략 미션(SDO, RBSP, Solar Probe Plus, Solar Orbiter), 기회의 임무(BARREL 풍선 어레이), 표적 연구, 우주환경 시험장, 파트너십(예: ESA Solar Orbiter)을 묶습니다. 통합 목표는 "예측 능력" — 태양물리학을 기술 과학에서 운용 예보로 전환하는 것입니다. SDO는 LWS 첫 미션, RBSP는 두 번째입니다.

### 2.2 Section 2 — Introduction (p. 4–5)
- **EN.** SDO targets two timescales. *Short term*: precursors of flares/CMEs by tracking magnetic-field topology before and during eruption. *Long term*: the dynamo — how field is amplified, transported, and destroyed inside the Sun, including emergence of active regions across an 11-year cycle. Solar Cycle 24 (started Dec 2008, predicted below-average peak in 2013) gave SDO an unusual quiet-Sun laboratory after the unusually deep Cycle 23/24 minimum. Two SDO-era discoveries are highlighted in the introduction: (i) the global field needs time to relax even after modest flares/filaments (Schrijver & Title 2011), and (ii) low-class flares often radiate energy underestimated by GOES X-ray detectors, with a long EUV "afterglow" that affects atmospheric drag (Woods et al. 2011b).
- **KR.** SDO는 두 시간 척도를 겨냥합니다. *단기*: 플레어·CME 전후 자기장 위상 추적으로 전조 탐지. *장기*: 다이나모 — 태양 내부에서 자기장이 증폭·수송·소멸되는 과정과 11년 주기에 걸친 활동영역 출현. 2008년 12월 시작된 태양주기 24(2013년 평균 이하 극대 예측)는 깊었던 23/24 극소기 이후 SDO에게 이례적 조용한 태양 실험실을 제공했습니다. 본 절의 SDO 시대 발견 두 건: (i) 전구적 자기장은 작은 플레어·필라멘트 후에도 이완 시간이 필요(Schrijver & Title 2011), (ii) 저등급 플레어가 GOES X-ray로는 과소평가된 에너지를 EUV "잔광"으로 오래 방출해 대기 항력에 영향(Woods 외 2011b).

- **EN.** The introduction also frames the data scale: ~150 000 high-resolution full-Sun images + 9 000 EUV spectra per day → ~1.5 TB/day science data → 3–4 PB raw over 5 years.
- **KR.** 데이터 규모도 도입부에서 명시: 일 ~15만 장 전면 영상 + 9 000개 EUV 스펙트럼 → ~1.5 TB/일 → 5년 동안 원시 3–4 PB.

### 2.3 Section 3 — History (p. 5–6)
- **EN.** SDO traces back to MDI on SOHO (helioseismology + LOS magnetograms) and to Yohkoh/SOHO/Skylab/TRACE (EUV/X-ray imaging), augmented by the SEE/TIMED EUV-irradiance demonstration (Woods et al. 2005) which proved 1–122 nm coverage was scientifically essential. The Science Definition Team (chaired by Hathaway, study scientist Thompson) was formed Nov 2000; SDT report issued Jul 2001 (Hathaway et al. 2001), AO 02-OSS-01 released 18 Jan 2002. HMI and EVE SITs selected 19 Aug 2002, AIA SIT 7 Nov 2003. Build/test completed Sep 2008; Atlas V delays moved SDO to Florida storage until Jun 2009; launch 11 Feb 2010. Four uniquely-NASA features: (i) sustained high data-production rate with rad-hard electronics and <1 ns timing, (ii) continuous automated science downlink, (iii) extreme pointing and stability for image-to-image registration, (iv) 5-year high-duty-cycle lifetime. Three of four are validated; lifetime is yet-to-prove (and as of 2026 has been exceeded).
- **KR.** SDO 계보는 SOHO의 MDI(헬리오사이즈몰로지·LOS 자기그림)와 Yohkoh·SOHO·Skylab·TRACE(EUV·X-ray 영상)로 거슬러 올라가며, SEE/TIMED의 EUV 복사조도 시연(Woods 외 2005)이 1–122 nm 관측의 과학적 필수성을 입증했습니다. 과학 정의팀(SDT; 의장 Hathaway, 연구 과학자 Thompson)은 2000년 11월 결성, SDT 보고서는 2001년 7월 발행(Hathaway 외 2001), AO 02-OSS-01은 2002년 1월 18일 공고. HMI·EVE SIT는 2002년 8월 19일, AIA SIT는 2003년 11월 7일 선정. 제작·시험은 2008년 9월 완료, Atlas V 지연으로 2009년 6월까지 플로리다에서 보관, 2010년 2월 11일 발사. NASA 미션의 네 가지 독특한 특징: (i) 방사선 내성 전자장치와 <1 ns 타이밍을 갖춘 지속적 고데이터 생산, (ii) 연속 자동화 과학 다운링크, (iii) 영상-영상 등록을 위한 극도로 정확한 지향·안정성, (iv) 5년 고가동 수명. 네 항목 중 셋은 검증, 수명은 미검증(2026년 현재 초과 달성).

### 2.4 Section 4 — Spacecraft Summary (p. 7)

| Parameter / 항목 | Value / 값 |
|---|---|
| Stabilization | 3-axis, fully redundant |
| Mission duration | 5 yr prime + ≥5 yr fuel margin |
| Launch mass | 3 000 kg (instruments 300, bus 1 300, fuel 1 400) |
| Sun-axis length | 4.7 m |
| Bus side | 2.2 m |
| Solar-array span | 6.1 m |
| HGA span | 6.0 m |
| Solar array | 6.6 m², 1 500 W EOL @ 16 % efficiency, "homeplate" shape |
| HGA | Two; rotate once/orbit to track ground station |
| Downlink | Ka-band ~26 GHz, 150 Mbps total (130 data + 20 encoding) |
| MOC | Goddard SFC, MD |
| Ground station | White Sands Complex, NM (dual antenna) |

- **EN.** Two pieces of buried engineering: (1) the "homeplate" solar-array shape exists so the rectangular array does not occult the high-gain antennas, which sweep ±90° once per orbit; (2) thrusters and main engine are on the *opposite* (anti-Sun) face of the bus to prevent plume contamination of optics. The main engine was disabled after final orbit insertion — only thrusters remain.
- **KR.** 묻혀 있는 엔지니어링 두 가지: (1) "홈플레이트" 모양 태양전지는 사각 어레이가 고이득 안테나(궤도당 ±90°를 휘젓는)의 시야를 막지 않기 위함, (2) 추력기와 주엔진은 광학계 오염 방지를 위해 본체의 반대(반-태양) 면에 위치. 최종 궤도 진입 후 주엔진은 비활성화 — 추력기만 운용.

### 2.5 Section 5 — Science Goals (p. 8) — Tables 1, 2, 3
- **EN.** Three nested tables. Table 1: seven Level-1 science questions (cycle drivers, AR flux processing, small-scale reconnection, EUV irradiance origin, CME/flare configurations, near-Earth solar wind from photospheric input, forecast feasibility). Table 2: four instrument science objectives (energy budget into Earth, near-surface shear-layer dynamics, deep dynamo, chromosphere/corona dynamics). Table 3: five measurement objectives (multi-cycle coverage, EUV spectral irradiance at fast cadence, full-disk Doppler, full-disk vector B, multi-T images at fast cadence). Three of seven questions are explicitly forecasting questions — predictive capability is hard-coded into requirements.
- **KR.** 세 개 중첩 표. 표 1: 7개 1단계 과학 질문(주기 구동, AR 자속 처리, 소규모 재결합, EUV 조도 기원, CME·플레어 구성, 광구 입력으로부터의 근지구 태양풍, 예보 가능성). 표 2: 4개 기기 과학 목표(지구 에너지 예산, 표면 근방 전단층 역학, 깊은 다이나모, 채층·코로나 역학). 표 3: 5개 측정 목표(다중주기 커버리지, EUV 분광 조도 고시간 해상도, 전면 도플러, 전면 벡터 자기장, 고속 다온도 영상). 7개 질문 중 3개가 명시적 예보 질문 — 예측 능력이 요구사항에 박혀 있습니다.

### 2.6 Section 6 — Science Investigation Teams (p. 9–10)

#### 2.6.1 AIA — Atmospheric Imaging Assembly
- **EN.** PI Alan Title (LMSAL). Four telescopes, each a multilayer Cassegrain. Ten bandpasses: seven EUV (94, 131, 171, 193, 211, 304, 335 Å), two UV (1600, 1700 Å), one visible (4500 Å). Field ≥40′, two-pixel resolution 1.2″, 4096² CCDs. Temperature diagnostic range 6 000 K to 3 × 10⁶ K. Built by LMSAL (Palo Alto). Detailed in Lemen et al. 2011.
- **KR.** PI Alan Title (LMSAL). 4 망원경, 각각 다층 코팅 카세그레인. 10 채널: EUV 7개(94, 131, 171, 193, 211, 304, 335 Å), UV 2개(1600, 1700 Å), 가시 1개(4500 Å). 시야 ≥40′, 2픽셀 분해능 1.2″, 4096² CCD. 온도 진단 범위 6 000 K – 3 × 10⁶ K. LMSAL(Palo Alto) 제작. Lemen 외 2011 상세.

#### 2.6.2 EVE — EUV Variability Experiment
- **EN.** PI Tom Woods (LASP, U. Colorado). Three components: MEGS (grating spectrometers, 6.5–105 nm at 0.1 nm resolution), ESP (ESP radiometer, 0.1–7 nm and several EUV bands), SAM (pinhole camera with MEGS-A CCD, 0.1–7 nm X-ray photons). Lyman-α at 121.6 nm by silicon photodiode in MEGS-B. Cadence 10 s. Successor in spirit to SOHO/SEM.
- **KR.** PI Tom Woods (LASP, U. Colorado). 세 구성요소: MEGS(격자 분광기, 6.5–105 nm, 0.1 nm 분해능), ESP(라디오미터, 0.1–7 nm 및 몇몇 EUV 대역), SAM(MEGS-A CCD를 사용하는 핀홀 카메라, 0.1–7 nm X-ray 광자). MEGS-B 내 실리콘 광다이오드가 121.6 nm Lyman-α 측정. 시간 해상도 10초. SOHO/SEM의 후속 정신.

#### 2.6.3 HMI — Helioseismic & Magnetic Imager
- **EN.** PI Phil Scherrer (Stanford), built by LMSAL. Measures Doppler and Stokes parameters across the Fe I 617.3 nm photospheric line. Dopplergrams every 45 s, 1″ two-pixel, ~25 m s⁻¹ noise, 95 % data recovery, 99 % completeness. LOS magnetograms every 45 s, 1″ pixel, 17 G noise, ±3 kG dynamic range. Vector magnetograms every 12 min, polarization accuracy ≥0.3 %. First rapid-cadence full-disk vector magnetic field. Detailed in Schou et al. 2011.
- **KR.** PI Phil Scherrer (Stanford), LMSAL 제작. Fe I 617.3 nm 광구선의 도플러 및 Stokes 매개변수 측정. 도플러그램 45초마다, 1″ 2픽셀, 잡음 약 25 m s⁻¹, 데이터 회수율 95 %, 완전성 99 %. LOS 자기그림 45초마다, 1″ 픽셀, 잡음 17 G, 동적 범위 ±3 kG. 벡터 자기그림 12분마다, 편광 정확도 ≥0.3 %. 최초의 고속 전면 벡터 자기장. Schou 외 2011 상세.

### 2.7 Section 7 — Science Data Capture (p. 10–11)
- **EN.** Driven by helioseismology, which needs uninterrupted intervals. Requirement: at least 22 individual 72-day periods over 5 years with ≥95 % capture (HMI), 90 % (AIA, EVE). If >5 % (HMI) or >10 % (AIA/EVE) is lost in a 72-day block, PIs may jointly extend it. Planned interruptions: thruster maneuvers, HGA handovers, calibration rolls, equinox eclipses (44 h/yr), Mercury (2016, 2019) and Venus (2012) transits (counted as good data, used for calibration; 100 h/yr), SEU/SEEs (34 h/yr), unplanned (rain attenuation, sun-station-RFI, equipment) ~112.5 h/yr. Five 72-day periods completed without interruption by the time of writing; ground system forwarded 99.97 % of IM_PDUs; HMI recovered 97 % of Dopplergrams/magnetograms.
- **KR.** 헬리오사이즈몰로지가 끊김 없는 구간을 요구하므로 추진. 요구사항: 5년 동안 22개 이상의 72일 구간에서 ≥95 %(HMI), 90 %(AIA·EVE) 회수. 72일 동안 손실이 5 %(HMI) 또는 10 %(AIA·EVE)를 초과하면 PI들이 합의해 구간을 연장. 계획된 중단: 추력기 기동, HGA 핸드오버, 보정 롤, 분점 식(연 44h), Mercury(2016, 2019)·Venus(2012) 통과(좋은 보정 데이터로 계산, 연 100h), SEU·SEE(연 34h), 계획되지 않은 중단(강우 감쇠, 태양-지상국 정렬 RFI, 장비) 약 연 112.5h. 작성 시점까지 5개 72일 구간이 무중단 완료, 지상계는 IM_PDU의 99.97 % 전송, HMI는 도플러그램·자기그림 97 % 회수.

### 2.8 Section 8 — SDO Orbit and Mission Phases (p. 11–12)
- **EN.** Inclined geosynchronous orbit, inclination 28.5°, altitude 35 800 km, longitude over White Sands. The figure-8 ground track has width (i/4) sin i = 3.3°. *Why not Sun-sync LEO?* Because no recorder existed for ~1.5 TB/day, and multi-station LEO downlink is too complex. *Trade-offs paid:* higher launch cost (vs. LEO), two annual ~2–3-week eclipse seasons, three lunar transits per year, higher radiation belt edge dose (additional shielding added). *Insertion:* Atlas V/Centaur placed SDO in GTO (apogee 35 350, perigee 2 500 km). Nine apogee burns raised perigee to 35 350; three thruster burns then equalized at 35 800 km. Excessive fuel slosh delayed final orbit to 16 March 2010. Two maneuver types: ΔV (longitude/inclination station-keeping) and ΔH (momentum dump from reaction wheels). End-of-life disposal: above the geo belt, depleted.
- **KR.** 경사 지구정지궤도, 경사각 28.5°, 고도 35 800 km, 경도는 White Sands 위. 8자 지상궤도 폭 (i/4) sin i = 3.3°. *Sun-sync LEO가 아닌 이유*: ~1.5 TB/일을 저장할 기록기가 없고, 다중 지상국 LEO 다운링크는 너무 복잡. *지불한 대가*: LEO 대비 높은 발사 비용, 연 2회 ~2–3주 식 시즌, 연 3회 달 통과, 방사선대 가장자리 통과로 인한 높은 선량(추가 차폐 적용). *진입*: Atlas V/Centaur가 SDO를 GTO(원점 35 350, 근점 2 500 km)에 배치 → 9회 원점 분사로 근점 35 350 km로 상승 → 3회 추력기 분사로 35 800 km 균등화. 과도한 연료 슬로싱으로 최종 궤도 도달이 2010년 3월 16일까지 지연. 두 종류 기동: ΔV(경도·경사 유지)와 ΔH(반작용 휠 모멘텀 덤프). 임무 종료 처분: 정지궤도 벨트 위, 에너지 고갈.

### 2.9 Section 9 — SDO Data (p. 12 onward)
- **EN.** Ka-band stream → New Mexico ground station → telemetry files → SOCs (JSOC at Stanford for HMI/AIA, LASP for EVE). Each instrument gets a fixed slice of the 130 Mbps. Data latency near-realtime ~15 min. FITS files at full resolution from JSOC; reduced products via partner sites. Full mission archive will be PB-class; the team published reduction methods so outsiders can "observe the database."
- **KR.** Ka-band 스트림 → 뉴멕시코 지상국 → 텔레메트리 파일 → SOC(HMI·AIA는 Stanford JSOC, EVE는 LASP). 각 기기가 130 Mbps 중 고정된 비율 사용. 근실시간 데이터 지연 약 15분. 전체 해상도 FITS는 JSOC에서, 축소 산출물은 파트너 사이트. 전체 미션 아카이브는 PB급; 팀은 외부 사용자가 "데이터베이스를 관측"할 수 있도록 처리 방법을 공개.

## 3. Key Takeaways / 핵심 시사점

1. **Mission-as-Sensor / 미션이 곧 센서.**
   - EN. SDO is intentionally designed so that the science is unbroken time series, not campaigns. The 22 × 72-day requirement in Section 7 is the contractual articulation of this philosophy: discontinuous helioseismology is bad helioseismology.
   - KR. SDO는 캠페인이 아닌 끊김 없는 시계열이 과학 산출물이 되도록 의도적으로 설계되었습니다. 7장의 22 × 72일 요구사항은 이 철학의 계약적 표현입니다 — 불연속 헬리오사이즈몰로지는 나쁜 헬리오사이즈몰로지입니다.

2. **Inclined GEO is a Data-Rate Decision / 경사 GEO는 데이터율 결정.**
   - EN. The orbit choice is essentially driven by 130 Mbps. Without continuous Ka-band visibility to one station, ~1.5 TB/day cannot leave the satellite. Polar Sun-sync LEO would give 100 % solar visibility but require an impossible onboard recorder. The mission paid for orbit complexity to avoid storage complexity.
   - KR. 궤도 선택은 본질적으로 130 Mbps에 의해 결정되었습니다. 단일 지상국에 대한 연속 Ka-band 가시성 없이 약 1.5 TB/일이 위성을 떠날 수 없습니다. 극궤도 Sun-sync LEO는 태양 가시성 100 %를 주지만 불가능한 탑재 기록기를 필요로 합니다. 미션은 저장 복잡성을 피하기 위해 궤도 복잡성을 지불했습니다.

3. **Three-Pronged Magnetic-Field Lifecycle / 3-갈래 자기장 일생.**
   - EN. HMI (where the field is born), AIA (how the field redistributes energy in the atmosphere), EVE (what the released energy looks like at Earth) sample three sequential stages of the same physics. Removing any one removes a step in the chain from interior dynamo to ionospheric impact.
   - KR. HMI(자기장이 탄생하는 곳), AIA(자기장이 대기에서 에너지를 재분배하는 방식), EVE(방출 에너지가 지구에 도달하는 모습)는 같은 물리의 순차적 세 단계를 표본화합니다. 어느 하나를 제거하면 내부 다이나모에서 전리권 영향까지의 사슬에서 한 단계가 사라집니다.

4. **Predictive Capability is a Requirement, Not a Hope / 예측 능력은 희망이 아닌 요구사항.**
   - EN. Three of the seven Level-1 questions (Table 1) ask explicitly about prediction. SDO must enable precursor identification and 1-day-ahead nowcasts; this is why cadence is seconds-to-minutes rather than hours.
   - KR. 7개 1단계 질문(표 1) 중 3개가 명시적으로 예측을 묻습니다. SDO는 전조 식별과 1일 전 즉보를 가능케 해야 합니다 — 그래서 시간 해상도가 시간 단위가 아닌 초·분 단위입니다.

5. **EUV Bandpass Choice ≈ DEM Coverage / EUV 대역 선택 ≈ DEM 커버리지.**
   - EN. AIA's seven EUV bands were chosen to cover ions from Fe IX (~0.7 MK) to Fe XXIV (~20 MK). Each bandpass is essentially a Gaussian in log T centered on a different ion's contribution function. Together they support differential emission measure (DEM) inversions — the workhorse of coronal thermodynamics in the SDO era.
   - KR. AIA의 EUV 7대역은 Fe IX(~0.7 MK)에서 Fe XXIV(~20 MK)까지의 이온을 커버하도록 선택되었습니다. 각 대역은 본질적으로 log T 상에서 서로 다른 이온의 기여 함수에 중심을 둔 가우시안입니다. 합쳐서 미분 방출 측정(DEM) 역산을 지원하며, 이는 SDO 시대 코로나 열역학의 주축입니다.

6. **The 130 Mbps Number is a Design Theorem / 130 Mbps는 설계 정리.**
   - EN. Adding the raw rates of AIA (4 telescopes × ~12 s × 4096² × 16 bit ≈ 67 Mbps), HMI (a few full-disk channels at 45 s ≈ 55 Mbps), and EVE (~few Mbps) yields ≈130 Mbps. The number is not arbitrary; it is the satisfaction of a system of inequalities.
   - KR. AIA(4 망원경 × ~12s × 4096² × 16 bit ≈ 67 Mbps), HMI(45초 단위 몇 개 전면 채널 ≈ 55 Mbps), EVE(~수 Mbps)의 원시율을 합하면 ≈130 Mbps. 임의의 값이 아니라 부등식 체계의 충족입니다.

7. **Calibration Embedded in Mission Plan / 미션 계획에 내장된 보정.**
   - EN. Mercury and Venus transits are *counted as good data* because the disk-crossing dark spots are absolute geometric calibrators. This is rare in NASA missions — usually transits are losses; here they are gains.
   - KR. Mercury·Venus 통과는 *좋은 데이터로 계산*됩니다 — 디스크 횡단의 어두운 점이 절대 기하 보정자이기 때문입니다. NASA 미션에서 이는 드뭅니다 — 보통 통과는 손실인데, 여기서는 이득입니다.

8. **Open Data as Architecture / 개방 데이터로서의 아키텍처.**
   - EN. The paper repeatedly emphasizes that any user can "observe the database." With ~15-min latency and PB-class archives, SDO is operationally a public observatory rather than a PI-team observatory. This data-sharing posture became the model for subsequent missions (Parker Solar Probe, Solar Orbiter).
   - KR. 논문은 누구든지 "데이터베이스를 관측"할 수 있음을 반복 강조합니다. 약 15분 지연과 PB급 아카이브로, SDO는 운용상 PI 팀 관측소가 아니라 공공 관측소입니다. 이 데이터 공개 자세는 후속 미션(Parker Solar Probe, Solar Orbiter)의 모델이 되었습니다.

9. **Inclined GEO Trades Sun Time for Bandwidth / 경사 GEO는 태양 시간을 대역폭과 교환.**
   - EN. A polar Sun-synchronous LEO sees the Sun ~100 % of the time but cannot continuously stream its data. SDO's inclined GEO sees the Sun about 95 % of the time (eclipse seasons cost ~44 h/yr) but maintains a 24/7 ground link. The mission paper makes the explicit calculation that this 5 % loss is far cheaper than building a 1.5 TB/day on-board recorder.
   - KR. 극궤도 Sun-sync LEO는 태양을 ~100 % 보지만 연속 데이터 스트림을 보낼 수 없습니다. SDO의 경사 GEO는 태양을 약 95 % 보지만(식 시즌 연 ~44h 손실) 24/7 지상 링크를 유지합니다. 미션 논문은 이 5 % 손실이 1.5 TB/일 탑재 기록기 제작보다 훨씬 저렴하다는 명시적 계산을 합니다.

10. **Cycle 24 Quiet Sun as Calibration Asset / 주기 24 조용한 태양은 보정 자산.**
    - EN. SDO launched into the deepest solar minimum since 1913. Counter-intuitively, this *helped* mission calibration: with few flares and CMEs, the 2010–2011 quiet baseline gave clean reference spectra and instrument-cross-calibration windows that would have been impossible during high activity.
    - KR. SDO는 1913년 이후 가장 깊은 태양 극소기에 발사되었습니다. 직관에 반해, 이는 미션 보정을 *도왔습니다* — 플레어·CME가 적어 2010–2011 조용한 기준선이 높은 활동 시기에는 불가능했을 깨끗한 참조 스펙트럼과 기기 교차 보정 창을 제공했습니다.

### 2.9.1 SDO Data Products & Latency / SDO 데이터 산출물·지연

| Product / 산출물 | Source / 출처 | Latency / 지연 | Note / 비고 |
|---|---|---|---|
| HMI Dopplergram | JSOC | ~15 min near-realtime | 45 s cadence, ~25 m/s noise |
| HMI LOS magnetogram | JSOC | ~15 min | 45 s, 17 G noise, ±3 kG |
| HMI Vector magnetogram | JSOC | hours-day | 12 min cadence, ≥0.3 % polarization |
| AIA Level 1 FITS | JSOC | ~15 min | 12 s/channel, 4096² |
| EVE MEGS spectra | LASP | ~15 min | 0.1 nm res., 6.5–105 nm |
| EVE ESP irradiance | LASP | ~15 min | bands 0.1–39 nm |

- **EN.** Data products at JSOC follow a Level-0/1/1.5/2 hierarchy. Level-0 is reconstructed telemetry; Level-1 adds basic flat-fielding and bad-pixel correction; Level-1.5 corrects geometric and radiometric effects; Level-2 includes derived products (DEM, magnetic-field inversion). Most users start at Level-1.5.
- **KR.** JSOC 데이터 산출물은 Level-0/1/1.5/2 계층을 따릅니다. Level-0은 재구성 텔레메트리, Level-1은 기본 평탄장·불량 픽셀 보정, Level-1.5는 기하·복사 보정, Level-2는 유도 산출물(DEM, 자기장 역산)을 포함합니다. 대부분 사용자는 Level-1.5에서 시작합니다.

### 2.10 Worked example — End-to-end information chain / 종단 간 정보 사슬 예시

- **EN.** Trace one piece of "useful science" through the SDO architecture. (1) An active region rotates onto the visible disk; HMI's Fe I 617.3 nm Zeeman observations produce a vector magnetogram every 12 minutes, sampling the *source* of the magnetic energy. (2) Three minutes later, AIA 171 Å (Fe IX, ~0.7 MK) shows coronal loops re-arranging above that region; AIA 94 Å (Fe XVIII, ~6 MK) shows hot post-flare loops; the joint cadence is 12 s. (3) EVE captures the Lyman-α and 30.4 nm spike from chromospheric and transition-region response within tens of seconds, integrated over the full disk. (4) The data is buffered, encoded, and Ka-band downlinked at 130 Mbps to White Sands within seconds. (5) The JSOC at Stanford (HMI/AIA) and LASP (EVE) ingest the streams and produce FITS Level-1.5 products within ~15 min. (6) Space-weather forecasters at NOAA SWPC retrieve the products and update flare/CME warnings. The architecture is a continuous pipeline from photon to forecast; latency, not raw quality, is the system-level metric.
- **KR.** "유용한 과학" 한 조각을 SDO 아키텍처를 통해 따라가 봅시다. (1) 활동영역이 보이는 면으로 자전; HMI의 Fe I 617.3 nm 제이만 관측이 12분마다 벡터 자기그림을 생산하여 자기 에너지의 *원천*을 표본화. (2) 3분 후 AIA 171 Å(Fe IX, ~0.7 MK)이 그 위 코로나 루프 재배치를, AIA 94 Å(Fe XVIII, ~6 MK)이 뜨거운 후속 플레어 루프를 보여 줌; 합동 시간 해상도 12초. (3) EVE가 채층·전이층 반응의 Lyman-α 및 30.4 nm 스파이크를 수십 초 안에 전면 적분으로 포착. (4) 데이터는 버퍼링·부호화 후 Ka-band로 130 Mbps로 White Sands에 수 초 내 다운링크. (5) Stanford JSOC(HMI·AIA)와 LASP(EVE)가 스트림을 받아 ~15분 내 FITS Level-1.5 산출. (6) NOAA SWPC 우주기상 예보관이 산출물을 받아 플레어·CME 경보 갱신. 광자에서 예보까지의 연속 파이프라인; 시스템 수준 지표는 원시 품질이 아니라 지연 시간.

## 4. Mathematical Summary / 수학적 요약

### 4.1 Inclined GEO geometry / 경사 정지궤도 기하학

The geosynchronous orbital radius r_GEO satisfies Kepler's third law for a 24-hour sidereal period:

$$r_{\mathrm{GEO}} = \left(\frac{G M_\oplus T^2}{4\pi^2}\right)^{1/3}$$

with $G M_\oplus = 3.986 \times 10^{14}~\mathrm{m^3 s^{-2}}$ and $T = 86\,164~\mathrm{s}$ → $r_{\mathrm{GEO}} \approx 42\,164~\mathrm{km}$ (altitude ≈ 35 786 km).

The ground-track figure-8 has half-width in latitude $\approx i$ (here 28.5°) and half-width in longitude

$$\Delta\lambda \approx \frac{i}{4}\sin i \approx \frac{28.5^\circ}{4}\sin 28.5^\circ \approx 3.4^\circ,$$

matching the paper's "(i/4) sin i = 3.3°" statement.

- **EN.** *Term-by-term:* $i$ is inclination in radians, $i/4$ is the leading expansion coefficient for an inclined-circular orbit, $\sin i$ couples to the projection of the inclined plane onto the equator. The longitudinal excursion is what produces the "elongated figure-eight" the paper describes.
- **KR.** *항별 해설:* $i$는 라디안 단위 경사각, $i/4$는 경사 원궤도의 주도 항, $\sin i$는 경사면을 적도에 투영. 이 경도 진동이 논문이 묘사하는 "긴 8자"를 만듭니다.

### 4.2 Data rate / 데이터율

The total downlink rate is split:

$$R_{\mathrm{Ka}} = R_{\mathrm{data}} + R_{\mathrm{enc}} = 130 + 20 = 150~\mathrm{Mbps}.$$

The instrument raw rate budget approximately satisfies

$$R_{\mathrm{AIA}} + R_{\mathrm{HMI}} + R_{\mathrm{EVE}} \le R_{\mathrm{data}}.$$

A back-of-envelope:
- AIA: $4 \text{ telescopes} \times \dfrac{4096^2 \text{ pix} \times 16 \text{ bit}}{12 \text{ s}} \approx 67~\mathrm{Mbps}$ (sharing among 10 channels).
- HMI: $\dfrac{\sim\!10 \times 4096^2 \times 16}{45} \approx 60~\mathrm{Mbps}$ before compression.
- EVE: a few Mbps for spectra at 10 s.

After lossless compression these fit in 130 Mbps.

### 4.3 Daily science volume / 일일 과학 데이터 용량

$$V_{\mathrm{day}} = R_{\mathrm{data}} \times 86\,400~\mathrm{s} = 130 \times 10^{6} \times 86\,400 \approx 1.12 \times 10^{13}~\mathrm{bit} \approx 1.4~\mathrm{TB}.$$

The paper rounds to ~1.5 TB/day. Over 5 years this is

$$V_{5\mathrm{yr}} = 1.5~\mathrm{TB} \times 365 \times 5 \approx 2.7~\mathrm{PB}.$$

The paper quotes 3–4 PB raw, accounting for telemetry overhead and pre-compression frames.

### 4.4 Eclipse season geometry / 식 시즌 기하학

For an inclined GEO satellite, eclipses occur near the equinoxes when the orbit plane contains the Earth–Sun line. The eclipse-season half-duration is

$$\Delta t_{\mathrm{ecl}} \approx \frac{R_\oplus}{r_{\mathrm{GEO}}\,|\sin i|}\,\frac{T_\oplus}{2\pi},$$

giving daily eclipses of up to ~72 minutes for two ~3-week windows per year — consistent with "44 hours/year" in the paper.

### 4.5 Radiation dose at inclined GEO / 경사 GEO에서의 방사선 선량

The orbit grazes the outer edge of Earth's outer electron radiation belt twice per orbit. The fluence per orbit scales approximately as

$$F \sim \int_{\mathrm{orbit}} \Phi_e(L,\,B) \,\mathrm{d}t,$$

where $\Phi_e$ is the omnidirectional electron flux at McIlwain $L$-shell and magnetic latitude $B$. With $L \approx 6$–7 at SDO's apogee, expected daily fluence at >1 MeV electrons is several orders of magnitude higher than at LEO Sun-sync orbits. The paper notes "additional shielding was added" — concretely, several mm of aluminum equivalents around radiation-sensitive electronics and detectors.

- **EN.** *Term-by-term:* the integrand is the line integral of flux along the orbit; over a 5-year mission this becomes a total ionizing dose (TID) budget that drives part-derating and shielding mass.
- **KR.** *항별 해설:* 피적분량은 궤도를 따른 플럭스 선적분; 5년 임무에 걸쳐 총 이온화선량(TID) 예산이 되며, 부품 디레이팅과 차폐 질량을 결정.

### 4.7 AIA temperature response (DEM context) / AIA 온도 반응

For a coronal channel $c$, the observed intensity is

$$I_c = \int K_c(T)\,\mathrm{DEM}(T)\,\mathrm{d}T,$$

where $K_c(T)$ is the temperature response kernel (combining wavelength response and ion contribution functions) and $\mathrm{DEM}(T) = n_e^2\,\mathrm{d}h/\mathrm{d}T$ is the differential emission measure. The choice of seven EUV bands provides seven independent constraints on DEM(T) — the foundation for SDO-era thermal coronal diagnostics.

- **EN.** *Term-by-term:* $K_c(T)$ peaks where each channel's dominant ion has its peak abundance in ionization equilibrium; integrating against DEM yields the per-pixel intensity. Inverting requires regularization because the kernel is ill-conditioned.
- **KR.** *항별 해설:* $K_c(T)$는 각 채널의 주도 이온이 이온화 평형에서 풍부도가 최대인 곳에서 최댓값을 가지며, DEM과의 적분이 픽셀 강도를 줍니다. 커널이 잘못-조건화되어 역산은 정칙화가 필요합니다.

### 4.8 Eclipse loss budget arithmetic / 식 손실 예산 산술

The two annual eclipse seasons span ~3 weeks each. A simple estimate of total annual eclipse hours:

$$T_{\mathrm{ecl}} \approx 2 \times 21~\mathrm{days} \times \overline{t}_{\mathrm{ecl,daily}},$$

where $\overline{t}_{\mathrm{ecl,daily}}$ averages from 0 to ~72 minutes across the season. Approximating triangularly: $\overline{t}_{\mathrm{ecl,daily}} \approx 36~\mathrm{min} \approx 0.6~\mathrm{h}$, giving $T_{\mathrm{ecl}} \approx 2 \times 21 \times 0.6 \approx 25~\mathrm{h}$. The paper's "44 hours/year" implies a wider effective window or longer mean eclipse duration; the discrepancy reflects penumbra and recovery time included in the operational budget.

Combined with planned losses (calibration rolls, transits-as-data, SEUs) and unplanned losses (rain attenuation), the total observation outage budget is:

$$T_{\mathrm{outage,total}} = 44~\mathrm{h} + 100~\mathrm{h} + 34~\mathrm{h} + 112.5~\mathrm{h} \approx 290.5~\mathrm{h/yr},$$

or about 3.3 % of the year — comfortably within the 5 % HMI completeness target.

- **EN.** *Term-by-term:* eclipse hours are forced by orbital geometry; transit hours are *gain* (calibration); SEU/SEE hours are stochastic (radiation environment); unplanned hours are weather/RFI/equipment. Their sum sets the floor for the data-capture margin.
- **KR.** *항별 해설:* 식 시간은 궤도 기하로 강제, 통과 시간은 *이득*(보정), SEU·SEE 시간은 확률적(방사선 환경), 계획되지 않은 시간은 날씨·RFI·장비. 합계가 데이터 수집 여유의 하한을 결정.

## 5. Paper in the Arc of History / 역사 속의 논문

```
1973  Skylab/ATM ─────── First sustained EUV/X-ray solar imaging from space
1981  SMM ─────────────── Coronagraph + flare spectrometers
1991  Yohkoh ──────────── Soft/hard X-ray imaging, full solar cycle
1995  SOHO ────────────── MDI helioseismology + EIT EUV imaging
1998  TRACE ───────────── 1″ EUV imaging (proof-of-concept for AIA)
2002  TIMED/SEE ───────── EUV irradiance demonstration (proof-of-concept for EVE)
2006  Hinode ──────────── SOT vector magnetograms (proof-of-concept for HMI vector)
2006  STEREO ──────────── Stereoscopic EUV
2010  ★ SDO ★ ────────── HMI + AIA + EVE; 1.5 TB/day; this paper
2012  RBSP/Van Allen Probes — Second LWS mission
2018  Parker Solar Probe — In-situ corona
2020  Solar Orbiter ───── Off-ecliptic + remote sensing
2025+ SDO continues operating well past prime mission
```

- **EN.** SDO is positioned at the inflection point where each predecessor's proof-of-concept (helioseismology, EUV imaging at 1″, EUV irradiance, vector B) was simultaneously matured into an operational, space-weather-ready facility.
- **KR.** SDO는 각 선행 미션의 개념 증명(헬리오사이즈몰로지, 1″ EUV 영상, EUV 복사조도, 벡터 B)이 동시에 성숙되어 운용·우주기상 대비 시설로 통합되는 변곡점에 위치합니다.

## 5.1 Comparison with Concurrent and Predecessor Missions / 동시대·이전 미션과의 비교

| Mission | Orbit | Data Rate | Imaging Cadence | Magnetic Field | Lifespan |
|---|---|---|---|---|---|
| Skylab/ATM (1973) | LEO crewed | film return | minutes | none | 9 mo |
| SMM (1980–1989) | LEO 28.5° | ~1 Mbps | minutes | LOS only | 9 yr |
| Yohkoh (1991–2001) | LEO | ~50 kbps | seconds (SXT) | none | 10 yr |
| SOHO (1995–) | L1 halo | 200 kbps | EIT 12 min | MDI LOS | 30+ yr |
| TRACE (1998–2010) | LEO Sun-sync | ~5 Mbps | seconds | none | 12 yr |
| Hinode (2006–) | LEO Sun-sync | ~4 Mbps | minutes | SOT vector | 18+ yr |
| **SDO (2010–)** | **iGEO 28.5°** | **130 Mbps** | **AIA 12 s** | **HMI vector 12 min** | **15+ yr** |
| IRIS (2013–) | LEO Sun-sync | ~0.7 Mbps | seconds (slit) | none | 11+ yr |

- **EN.** SDO's data rate is more than an order of magnitude beyond any predecessor and remains unmatched by any subsequent solar-imaging mission. Its operational longevity (16+ years as of 2026) has converted it into a multi-cycle observatory beyond the original design intent.
- **KR.** SDO의 데이터율은 어떤 선행 미션보다 한 자릿수 이상 크며, 이후 어떤 태양 영상 미션도 따라잡지 못했습니다. 운용 장수(2026년 현재 16년 이상)는 본래 설계 의도를 넘어 다중 주기 관측소로 전환시켰습니다.

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper | Relation / 관계 |
|---|---|
| Hathaway et al. 2001 (SDT report) | Source of the seven Level-1 science questions (Table 1) / 7개 1단계 과학 질문의 출처. |
| Lemen et al. 2011 (AIA) | Companion paper detailing AIA optics, channels, calibration / AIA 광학·채널·보정 동반 논문. |
| Schou et al. 2011 (HMI) | Companion paper detailing HMI design and pipeline / HMI 설계·파이프라인 동반 논문. |
| Woods et al. 2011a (EVE) | Companion paper detailing EVE / EVE 동반 논문. |
| Woods et al. 2005 (TIMED/SEE) | Demonstrated need for EUV irradiance measurement / EUV 복사조도 관측 필요성 입증. |
| Schrijver & Title 2011 | First SDO-era flare-relaxation result cited in introduction / 도입부에 인용된 첫 SDO 시대 플레어 이완 결과. |
| Solanki et al. 2004 | Solar-activity context: highest in 10 000 yr / 태양활동 맥락: 1만년 중 최고. |
| Tann, Pages & Silva 2005 | SDO Ground System reference / SDO 지상계 참조. |
| Pence et al. (FITS) | FITS file format used at JSOC / JSOC에서 사용하는 FITS 포맷. |

## 7. References / 참고문헌

- Pesnell, W. D., Thompson, B. J., & Chamberlin, P. C. "The Solar Dynamics Observatory (SDO)", *Solar Physics* **275**, 3–15 (2012). [DOI:10.1007/s11207-011-9841-3]
- Hathaway, D. H., et al. "Science Definition Team report for SDO" (2001).
- Lemen, J. R., et al. "The Atmospheric Imaging Assembly (AIA) on SDO", *Solar Physics* **275**, 17–40 (2011). [DOI:10.1007/s11207-011-9776-8]
- Schou, J., et al. "Design and Ground Calibration of HMI on SDO", *Solar Physics* **275**, 229–259 (2011). [DOI:10.1007/s11207-011-9842-2]
- Woods, T. N., et al. "EUV Variability Experiment (EVE) on SDO", *Solar Physics* **275**, 115–143 (2011a). [DOI:10.1007/s11207-009-9487-6]
- Woods, T. N., et al. (2005). TIMED/SEE EUV irradiance measurements.
- Schrijver, C. J., & Title, A. M. "Long-range magnetic couplings between solar flares and coronal mass ejections observed by SDO and STEREO", *J. Geophys. Res.* **116**, A04108 (2011).
- Solanki, S. K., et al. "Unusual activity of the Sun during recent decades compared to the previous 11,000 years", *Nature* **431**, 1084 (2004).
- Tann, H., Pages, R., & Silva, R. "The SDO Ground System" (2005).
- Pence, W. D., et al. "Definition of the Flexible Image Transport System (FITS)", *A&A* **524**, A42 (2010).

## 8. Glossary / 용어집

| Acronym / 약어 | Meaning / 의미 |
|---|---|
| AIA | Atmospheric Imaging Assembly — EUV/UV/visible imager / EUV·UV·가시 영상기 |
| AO | Announcement of Opportunity — NASA solicitation / NASA 공모 |
| BARREL | Balloon Array for RBR Electron Losses / 풍선 어레이 RBR 전자 손실 |
| CME | Coronal Mass Ejection / 코로나 질량 방출 |
| DEM | Differential Emission Measure / 미분 방출 측정 |
| EOL | End of Life / 수명 종료 |
| ESP | EUV SpectroPhotometer (part of EVE) / EUV 분광 광도계 |
| EUV | Extreme Ultraviolet (1–122 nm) / 극자외선 |
| EVE | EUV Variability Experiment / EUV 변동성 실험 |
| FITS | Flexible Image Transport System / 유연 영상 전송 시스템 |
| FOT | Flight Operations Team / 비행 운영팀 |
| GEO | Geosynchronous Earth Orbit / 지구정지궤도 |
| GTO | Geosynchronous Transfer Orbit / 지구정지전이궤도 |
| HGA | High-Gain Antenna / 고이득 안테나 |
| HMI | Helioseismic & Magnetic Imager / 헬리오사이즈믹·자기 영상기 |
| IM_PDU | Instrument Multiplexing Protocol Data Unit / 기기 다중화 프로토콜 데이터 단위 |
| ILWS | International Living With a Star / 국제 LWS |
| JSOC | Joint Science Operations Center (Stanford) / 공동 과학 운영 센터 |
| LASP | Lab. for Atmospheric & Space Physics (CU Boulder) / 대기·우주물리연구소 |
| LMSAL | Lockheed Martin Solar & Astrophysics Lab / 록히드 마틴 태양·천체물리 연구소 |
| LOS | Line Of Sight / 시선 |
| LWS | Living With a Star / 별과 함께 살기 |
| MDI | Michelson Doppler Imager (on SOHO) / 마이켈슨 도플러 영상기 |
| MEGS | Multiple EUV Grating Spectrograph (in EVE) / 다중 EUV 격자 분광기 |
| MOC | Mission Operations Center / 미션 운영 센터 |
| RBSP | Radiation Belt Storm Probes / 방사선대 폭풍 탐사선 |
| SAM | Solar Aspect Monitor (in EVE) / 태양 시점 모니터 |
| SDOG(S) | SDO Ground (Station) / SDO 지상국 |
| SDT | Science Definition Team / 과학 정의팀 |
| SEE | Solar EUV Experiment (TIMED) / 태양 EUV 실험 |
| SEU/SEE | Single Event Upset/Effect / 단일 사건 이상·효과 |
| SIT | Science Investigation Team / 과학 연구팀 |
| SOC | Science Operations Center / 과학 운영 센터 |
| TID | Total Ionizing Dose / 총 이온화 선량 |
| ΔV / ΔH | Velocity / Angular-momentum maneuver / 속도·각운동량 기동 |

## 9. Closing remarks / 마무리

- **EN.** Reading this paper today (2026), one is struck by how durable the design choices have proven. The inclined GEO is still operating; HMI, AIA, and EVE continue to produce science; the JSOC pipeline is the de-facto standard for solar imaging archives. The "predict-the-Sun" framing has matured into operational space-weather services at NOAA SWPC and ESA. SDO is the model heliophysics observatory.
- **KR.** 2026년 현재 본 논문을 읽으면, 설계 선택이 얼마나 오래 견뎌 왔는지가 인상적입니다. 경사 GEO는 여전히 운용 중이며, HMI·AIA·EVE는 계속 과학을 산출하고, JSOC 파이프라인은 사실상 태양 영상 아카이브의 표준입니다. "태양을 예측하라"는 프레이밍은 NOAA SWPC와 ESA의 운용 우주기상 서비스로 성숙했습니다. SDO는 표본적 태양·우주물리 관측소입니다.

