---
title: "Fox et al. 2016 — The Solar Probe Plus Mission: Humanity's First Visit to Our Star"
date: 2026-04-27
topic: Solar Physics
tags: [parker_solar_probe, mission_design, corona, solar_wind, in_situ]
paper_number: 37
status: in_progress
---

# Pre-Reading Briefing / 사전 학습 브리핑

## 1. Why This Paper Matters / 이 논문이 중요한 이유

**English.** Fox et al. (2016) is the official mission overview paper for **Solar Probe Plus (SPP)**, later renamed **Parker Solar Probe (PSP)** — the first spacecraft designed to fly into the Sun's corona. Published in *Space Science Reviews* in 2016, it consolidates over five decades of mission planning (since the 1958 Simpson Committee Report) and presents the science objectives, mission design, spacecraft, and instrument suite for what would become humanity's closest approach to a star (perihelion 9.86 R_sun = ~6.16 million km from Sun's center). Every subsequent PSP paper builds on this reference document.

**한국어.** Fox et al. (2016)은 **Solar Probe Plus (SPP)** — 후일 **Parker Solar Probe (PSP)** 로 명명됨 — 의 공식 미션 개요 논문이다. 2016년 *Space Science Reviews*에 게재된 이 논문은 1958년 Simpson Committee 보고서부터 50여 년의 미션 계획을 집대성하며, 인류 역사상 가장 가까이 별에 접근하는 우주선(근일점 9.86 R_sun ≈ 태양 중심에서 약 616만 km)의 과학 목표, 미션 설계, 우주선, 탑재체를 제시한다. 이후의 모든 PSP 논문은 이 문서를 참조한다.

## 2. Historical Context / 역사적 맥락

**English.** Theoretical foundations:
- **Grotrian (1939) / Saha (1942)** — discovery that the corona is ~10^6 K, far hotter than the photosphere (5800 K). The "coronal heating problem" is born.
- **Parker (1958)** — predicts the supersonic solar wind. Confirmed by Mariner 2 (1962).
- **NRC 1958 (Simpson Committee)** — first explicit recommendation for a "solar probe" to send a spacecraft inside Mercury's orbit.
- **Helios 1/2 (1974/76)** — closest approach until PSP at 0.29 AU = 62 R_sun.
- **Ulysses (1990–2009)** — out-of-ecliptic mission revealing the bimodal solar wind.
- **Solar Probe 2005 (SP2005)** — earlier polar Jupiter-gravity-assist concept (~4 R_sun perihelion, but only 1–2 passes). Replaced in 2008 by SPP design (24 ecliptic passes via Venus gravity assists).

**한국어.** 이론적 기반:
- **Grotrian (1939) / Saha (1942)** — 코로나 온도가 광구(5800 K)보다 훨씬 높은 ~10^6 K임을 발견. "코로나 가열 문제" 탄생.
- **Parker (1958)** — 초음속 태양풍 예측. Mariner 2 (1962)로 확인.
- **NRC 1958 (Simpson 위원회)** — 수성 궤도 안쪽으로 우주선을 보내는 "태양 탐사선" 권고의 시초.
- **Helios 1/2 (1974/76)** — PSP 이전 최단 거리(0.29 AU = 62 R_sun) 관측.
- **Ulysses (1990–2009)** — 황도면 외 미션, 이중 모드(bimodal) 태양풍 발견.
- **Solar Probe 2005 (SP2005)** — 이전의 목성 중력 도움 극궤도 안(~4 R_sun 근일점, 1–2회 통과). 2008년 Venus 중력 도움을 사용한 24회 통과의 SPP 설계로 교체.

## 3. Three Top-Level Science Objectives / 세 가지 최상위 과학 목표

**English.**
1. **Trace the flow of energy that heats the solar corona and accelerates the solar wind** — How is energy from the lower atmosphere transferred to the corona and wind? What shapes non-equilibrium VDFs?
2. **Determine the structure and dynamics of the plasma and magnetic fields at the sources of the solar wind** — How does the magnetic field connect photosphere → heliosphere? Are sources steady or intermittent?
3. **Explore mechanisms that accelerate and transport energetic particles** — Roles of shocks, reconnection, waves, turbulence?

**한국어.**
1. **코로나를 가열하고 태양풍을 가속하는 에너지 흐름 추적** — 하부 대기에서 코로나/풍으로 에너지가 어떻게 전달되는가? 비평형 VDF는 어떻게 형성되는가?
2. **태양풍 원천에서 플라스마와 자기장의 구조 및 동역학 결정** — 광구 → 헬리오스피어 자기장 연결 방식? 원천이 정상적인가 간헐적인가?
3. **에너지 입자의 가속 및 수송 메커니즘 탐색** — 충격파, 재결합, 파동, 난류의 역할?

## 4. Key Vocabulary / 핵심 용어

| Term | Korean | Definition |
|------|--------|------------|
| Perihelion | 근일점 | Closest distance to Sun in orbit |
| R_sun (R_S) | 태양 반경 | 695,700 km |
| Venus Gravity Assist (VGA) | 금성 중력 도움 | Trajectory shaping using Venus's gravity |
| Alfvén critical surface | 알펜 임계면 | Where solar wind speed = local Alfvén speed |
| TPS | 열 보호 시스템 | Thermal Protection System (carbon-carbon heat shield, 1377 °C) |
| FIELDS | 자기장/전기장 탑재체 | Magnetometers + electric antennas + search coil |
| SWEAP | 태양풍 플라스마 탑재체 | Solar Wind Electrons Alphas and Protons (SPC + SPAN) |
| ISʘIS | 에너지 입자 탑재체 | Integrated Science Investigation of the Sun (EPI-Hi + EPI-Lo) |
| WISPR | 백색광 영상기 | Wide-field Imager for Solar Probe |
| 475 Suns | 태양 강도 475배 | Irradiance at 9.86 R_sun = 475 × 1366 W/m² |
| eVDF / pVDF | 전자/양성자 속도분포함수 | Electron/proton velocity distribution function |
| Strahl | 자기장 평행 전자 빔 | Field-aligned superthermal electron population |

## 5. Mission Architecture in Brief / 미션 구조 요약

**English.** Launch July 31 – Aug 18, 2018 on Delta IV Heavy + STAR-48B upper stage (C3 = 154 km²/s², the highest ever required). Seven Venus flybys (VF1–VF7) over 7 years, walking the perihelion down from 35.66 R_sun (orbit 1) → 9.86 R_sun (orbits 22–24). 24 total orbits. Spacecraft: 685 kg, ~3 m tall, 2.3 m diameter at TPS. Power from a small actively-cooled solar array (~6500 W heat dissipation at perihelion); flat-faced TPS keeps the bus in shadow. Communication via Ka-band 0.6 m HGA on anti-ram side.

**한국어.** 2018년 7월 31일 – 8월 18일 Delta IV Heavy + STAR-48B 상단 로켓으로 발사 (C3 = 154 km²/s², 역대 최고). 7년에 걸쳐 7회의 금성 근접 통과 (VF1–VF7), 근일점을 35.66 R_sun (1회) → 9.86 R_sun (22–24회)로 단계적으로 낮춤. 총 24궤도. 우주선: 685 kg, 높이 ~3 m, TPS 직경 2.3 m. 전력은 소형 능동 냉각식 태양전지판 (근일점에서 ~6500 W 열 방출), 평면 TPS가 본체를 그림자 속에 둠. 통신은 anti-ram 면의 Ka-band 0.6 m HGA로 수행.

## 6. Pre-Reading Q&A / 사전 학습 Q&A

### Q1. Why 9.86 R_sun and not closer? / 왜 9.86 R_sun이며 더 가깝지 않은가?
**English.** Below ~10 R_sun the spacecraft would cross the predicted Alfvén critical surface (~12–13 R_sun in Verdini models), entering the genuine solar corona where the wind is sub-Alfvénic and waves can travel sunward. 9.86 R_sun balances scientific reach (sampling sub-Alfvénic plasma) against thermal/engineering limits (TPS rated to 475 Suns; cooling system caps at 6500 W).

**한국어.** ~10 R_sun 아래에서는 예측된 Alfvén 임계면(Verdini 모델 기준 ~12–13 R_sun)을 가로지르므로, 풍이 준-Alfvén 이하이고 파동이 태양 쪽으로 전파할 수 있는 진정한 코로나에 진입한다. 9.86 R_sun은 과학적 도달(준-Alfvén 이하 플라스마 표본)과 열/공학적 한계(TPS 475 Suns 기준; 냉각 6500 W 상한) 사이의 균형이다.

### Q2. Why Venus gravity assists, not Jupiter? / 왜 목성이 아닌 금성 중력 도움인가?
**English.** SP2005 used a single Jupiter flyby for a polar 4 R_sun pass — extreme orbit inclination but only 1–2 perihelia and a 14-year cruise. SPP's seven Venus flybys produce 24 ecliptic perihelia at slightly higher closest distance (10 R_sun vs 4 R_sun), trading depth for coverage. The McComas et al. (2008) trade study showed 24 passes × hours-near-Sun > 2 passes × longer-near-Sun for the science return.

**한국어.** SP2005는 단일 목성 통과로 극궤도 4 R_sun 통과 — 극단적 궤도 경사이지만 근일점 1–2회와 14년 운항. SPP의 7회 금성 통과는 24회 황도면 근일점을 다소 먼 거리(4 → 10 R_sun)에서 제공. McComas et al. (2008) 트레이드 연구는 24회 × 태양 근접 시간 > 2회 × 더 긴 시간이 과학 산출에 유리함을 보였다.

### Q3. How does the TPS survive 475 Suns? / TPS는 어떻게 475 Suns를 견디는가?
**English.** The TPS is a **carbon-carbon composite / carbon foam sandwich**, ~11.4 cm thick, with a sun-facing alumina (Al₂O₃) coating that reflects most visible light. Front face reaches ~1377 °C in vacuum thermal equilibrium, while the rear face stays near 30 °C. Spacecraft bus stays in the 5.82° umbra cone. Solar arrays peek out from behind the shield and are water-cooled (titanium platen, pumped water → CSPRs radiator).

**한국어.** TPS는 **carbon-carbon 복합재 / 탄소 폼 샌드위치**, ~11.4 cm 두께, 태양 면 알루미나(Al₂O₃) 코팅으로 가시광 대부분을 반사한다. 전면은 진공 열평형에서 ~1377 °C, 후면은 ~30 °C. 본체는 5.82° umbra 원뿔 안에 위치. 태양전지판은 차폐 뒤에서 일부만 노출되고 수냉식(티타늄 플레이튼, 펌프 물 → CSPR 라디에이터).

### Q4. What does "475 Suns" mean quantitatively? / "475 Suns"의 정량적 의미는?
**English.** Solar irradiance scales as 1/r². At 1 AU it's 1366 W/m² ("solar constant"). At 9.86 R_sun = 0.0459 AU, irradiance = 1366 × (1/0.0459)² ≈ 6.49 × 10⁵ W/m² ≈ 475 × solar constant. That's enough to melt aluminum, vaporize most polymers, and require an active cooling loop to keep solar cells below 150 °C.

**한국어.** 태양 복사조도는 1/r²로 변한다. 1 AU에서 1366 W/m² ("태양 상수"). 9.86 R_sun = 0.0459 AU에서 1366 × (1/0.0459)² ≈ 6.49 × 10⁵ W/m² ≈ 475 × 태양 상수. 이는 알루미늄을 녹이고 대부분 폴리머를 기화시키며, 태양전지를 150 °C 이하로 유지하려면 능동 냉각 회로가 필수적임을 의미한다.

### Q5. What are the four instrument suites and what do they measure? / 4가지 탑재체와 측정 대상은?
**English.**
- **FIELDS** (UC Berkeley) — 3 fluxgate magnetometers + search coil + 5 electric antennas; DC to ~MHz E&B fields, density, plasma waves, radio bursts.
- **SWEAP** (Smithsonian Astrophysical Observatory) — Solar Probe Cup (SPC, Faraday cup on ram side, peeks past TPS) + SPAN-A/B (electrostatic analyzers); ion/electron VDFs, n, V, T.
- **ISʘIS** (SwRI) — EPI-Hi (high energy 1–100 MeV) + EPI-Lo (~10 keV–MeV); energetic particle composition and spectra.
- **WISPR** (NRL) — wide-field white-light imager; coronal structure ahead of/behind spacecraft, CMEs, streamers.

**한국어.**
- **FIELDS** (UC Berkeley) — fluxgate 자력계 3 + 서치 코일 + 전기 안테나 5; DC ~ MHz E&B 장, 밀도, 플라스마 파동, 전파 폭발.
- **SWEAP** (SAO) — Solar Probe Cup (SPC, ram 면 Faraday 컵, TPS 뒤에서 노출) + SPAN-A/B (정전 분석기); 이온/전자 VDF, n, V, T.
- **ISʘIS** (SwRI) — EPI-Hi (고에너지 1–100 MeV) + EPI-Lo (~10 keV–MeV); 에너지 입자 조성 및 스펙트럼.
- **WISPR** (NRL) — 광시야 백색광 영상기; 우주선 전후방 코로나 구조, CME, streamer.

## 7. Reading Strategy / 읽기 전략

**English.** Skim §1–2 (Introduction & Science Overview) carefully — that's the conceptual core. Tables 1 (objectives → measurements) and 2 (instruments → institutions) are essential references. §3 (Science Implementation), §4 (Mission Design), and §5 (Science Operations) are engineering-heavy; focus on Tables 3 (orbit time at <30, 20, 15, 10 R_sun) and Figs. 13–14 (trajectory). §6 (Summary) re-states the case in non-technical language.

**한국어.** §1–2 (서론 & 과학 개요)는 개념적 핵심이므로 정독. Table 1 (목표 → 측정)과 Table 2 (탑재체 → 기관)는 필수 참조. §3 (과학 구현), §4 (미션 설계), §5 (과학 운용)은 공학 중심; Table 3 (<30, 20, 15, 10 R_sun 궤도 체류 시간)과 Fig. 13–14 (궤도) 집중. §6 (요약)은 비기술적 언어로 사례를 재진술.

## References for Briefing / 브리핑 참고 문헌
- Fox, N.J., Velli, M.C., Bale, S.D., et al., "The Solar Probe Plus Mission: Humanity's First Visit to Our Star", *Space Sci. Rev.* 204, 7–48, 2016. doi:10.1007/s11214-015-0211-6
- McComas et al., 2008 — SPP Science and Technology Definition Team Report.
- Verdini & Velli, 2007; Verdini et al., 2009 — Alfvén turbulence models.
