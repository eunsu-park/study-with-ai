---
title: "Fox et al. 2016 — The Solar Probe Plus Mission: Humanity's First Visit to Our Star"
date: 2026-04-27
date_completed: 2026-04-27
topic: Solar Physics
tags: [parker_solar_probe, mission_design, corona, solar_wind, in_situ, FIELDS, SWEAP, ISIS, WISPR]
paper_number: 37
status: completed
---

# Reading Notes / 읽기 노트
**Fox, N.J., Velli, M.C., Bale, S.D., et al. (2016)**
*Space Science Reviews* 204, 7–48. doi:10.1007/s11214-015-0211-6

---

## 1. Core Contribution / 핵심 기여

**English.** This paper presents the official mission overview for **Solar Probe Plus (SPP)** — later renamed **Parker Solar Probe (PSP)** — the first spacecraft designed to fly into the Sun's corona. SPP launched in August 2018 (within the planning window described here) on a Delta IV Heavy rocket with a STAR-48B upper stage. Using **seven Venus gravity assists** over a 7-year nominal mission, it walks its perihelion down from 35.66 R_sun on orbit 1 to **9.86 R_sun on the final three orbits** (closer than any human-made object has ever been to a star). At closest approach, the spacecraft endures **~475 times the solar irradiance experienced at Earth (≈649 kW/m²)**, surviving thanks to a carbon-carbon composite **Thermal Protection System (TPS)** that operates at ~1377 °C while keeping the bus near room temperature. The four instrument suites (**FIELDS, SWEAP, ISʘIS, WISPR**) provide in-situ measurements of fields, plasmas, energetic particles, and white-light imaging across **24 orbits** that spend a cumulative ~937 hours inside 20 R_sun and ~14.85 hours inside 10 R_sun. SPP's three top-level science objectives — (1) trace the energy that heats the corona and accelerates the solar wind, (2) determine plasma/field structure at solar wind sources, and (3) explore energetic particle acceleration mechanisms — directly target questions outstanding since the 1958 Simpson Committee report.

**한국어.** 이 논문은 **Solar Probe Plus (SPP)** — 후일 **Parker Solar Probe (PSP)** 로 명명됨 — 의 공식 미션 개요를 제시한다. SPP는 태양 코로나로 비행하도록 설계된 최초의 우주선으로, 본 논문에 기술된 발사 창 내인 2018년 8월 Delta IV Heavy 로켓과 STAR-48B 상단으로 발사되었다. 7년 명목 미션 동안 **7회의 금성 중력 도움**을 통해 근일점을 1궤도의 35.66 R_sun에서 **마지막 3궤도의 9.86 R_sun**까지 단계적으로 낮춘다 (인공 물체가 별에 이렇게 가까이 간 적 없음). 최근접 시 우주선은 **지구 대비 약 475배의 태양 복사조도(≈649 kW/m²)**를 견디며, ~1377 °C에서 작동하는 carbon-carbon 복합재 **열 보호 시스템(TPS)**이 본체를 상온 근처로 유지하기에 생존한다. 4가지 탑재체(**FIELDS, SWEAP, ISʘIS, WISPR**)는 **24궤도**에 걸쳐 장(field), 플라스마, 에너지 입자, 백색광 영상의 현장 측정을 제공하며, 누적 ~937시간은 20 R_sun 이내, ~14.85시간은 10 R_sun 이내에 머문다. SPP의 세 최상위 과학 목표 — (1) 코로나 가열 및 태양풍 가속 에너지 추적, (2) 태양풍 원천 플라스마/장 구조 결정, (3) 에너지 입자 가속 메커니즘 탐구 — 는 1958년 Simpson Committee 보고서 이래 미해결인 질문을 정조준한다.

---

## 2. Reading Notes / 읽기 노트 (Section-by-Section)

### §1. Introduction (pp. 8–10)

**English.** The paper opens by framing SPP's purpose: sample the corona to reveal heating and solar wind acceleration. Box 1 traces the lineage from the 1958 NRC Space Studies Board recommendation through the 2003 and 2013 NRC Decadal Surveys; eight major science studies between 1962 and 2008. The earlier "Solar Probe 2005" (SP2005) baselined a polar Jupiter-gravity-assist trajectory with one or two ~4 R_sun perihelia; the SPP redesign — settled in McComas et al. (2008) — uses seven Venus gravity assists for **24 ecliptic perihelia** ending at <10 R_sun. This trades closest-approach distance (4 → ~10 R_sun) for orbital count (2 → 24) and total near-Sun time (~160 → 2100+ hours inside 30 R_sun).

**한국어.** 논문 서두는 SPP의 목적을 명확히 한다: 코로나를 표본 추출하여 가열과 태양풍 가속을 밝힌다. Box 1은 1958년 NRC Space Studies Board 권고로부터 2003년 및 2013년 NRC Decadal Surveys까지 계보를 추적; 1962–2008년 사이 8개 주요 과학 연구. 이전 "Solar Probe 2005" (SP2005)는 ~4 R_sun 근일점 1–2회의 극궤도 목성 중력 도움 궤적을 기준; SPP 재설계 — McComas et al. (2008)에서 확정 — 는 **24회 황도면 근일점** (<10 R_sun 종료)을 위해 7회 금성 중력 도움을 사용한다. 이는 최근접 거리(4 → ~10 R_sun)를 궤도 수(2 → 24) 및 30 R_sun 이내 총 체류 시간(~160 → 2100+시간)과 교환한다.

### §2. Science Overview — Three Objectives (pp. 10–24)

**English.** Section 2 elaborates three top-level objectives, each split into sub-questions (Fig. unnumbered, in-text list, p. 10):

**Objective 1 — Trace the flow of energy:**
- 1a: How is energy transferred and dissipated in corona/wind?
- 1b: What shapes non-equilibrium VDFs?
- 1c: How do coronal processes affect solar wind in heliosphere?

**Objective 2 — Plasma/field at solar wind sources:**
- 2a: Source-region magnetic field connection?
- 2b: Steady or intermittent sources?
- 2c: How do coronal structures evolve into solar wind?

**Objective 3 — Energetic particle acceleration/transport:**
- 3a: Roles of shocks, reconnection, waves, turbulence?
- 3b: Source populations and physical conditions?
- 3c: Transport in corona/heliosphere?

Figure 1 (Ulysses + SOHO/LASCO/Mauna Loa composites) anchors the discussion — bimodal solar wind at minimum becomes mixed at maximum, so SPP across a ~7 year mission spans the activity transition. Figures 2–6 motivate the turbulent-cascade-driven heating models (Verdini, Cranmer) and present spectral break evolution and proton temperature anisotropy constraints (Bale 2009; Matteini 2007). Box-by-box Table 1 (pp. 26–27) maps each objective sub-question to a strategy and the four instrument suites' contribution.

**한국어.** §2는 세 최상위 목표를 각각 하위 질문으로 세분한다 (p. 10 본문 목록):

**목표 1 — 에너지 흐름 추적:**
- 1a: 코로나/풍에 에너지가 어떻게 전달되고 소산되는가?
- 1b: 비평형 VDF는 어떻게 형성되는가?
- 1c: 코로나 과정이 헬리오스피어 태양풍에 어떤 영향?

**목표 2 — 태양풍 원천의 플라스마/장:**
- 2a: 원천 영역 자기장 연결?
- 2b: 정상 vs 간헐?
- 2c: 코로나 구조가 태양풍으로 진화하는 방식?

**목표 3 — 에너지 입자 가속/수송:**
- 3a: 충격파, 재결합, 파동, 난류의 역할?
- 3b: 원천 모집단 및 물리 조건?
- 3c: 코로나/헬리오스피어 수송?

Figure 1 (Ulysses + SOHO/LASCO/Mauna Loa 합성)이 논의의 닻 — 최소기 이중 모드 태양풍이 최대기에 혼합되므로, SPP는 ~7년 미션을 통해 활동도 전이를 포괄한다. Figures 2–6은 난류 캐스케이드 가열 모델(Verdini, Cranmer)을 동기화하고 스펙트럼 브레이크 진화 및 양성자 온도 비등방성 제약(Bale 2009; Matteini 2007)을 제시한다. Table 1 (pp. 26–27)은 각 목표 하위 질문을 전략 및 4개 탑재체의 기여로 매핑한다.

### §3. Science Implementation — Instruments (pp. 27–29)

**English.** Table 2 lists four instrument suites and their lead institutions:

| Suite | Lead | Function |
|-------|------|----------|
| FIELDS | UC Berkeley | 3 fluxgate magnetometers + search coil + 5 electric antennas; DC to MHz E/B + plasma waves + density + radio |
| ISʘIS | Southwest Research Institute (SwRI) | EPI-Hi + EPI-Lo, 10s of keV to 100 MeV particles |
| SWEAP | Smithsonian Astrophysical Observatory (SAO) | SPC (Faraday cup, peeks past TPS) + SPAN-A/B/Ion (electrostatic analyzers); ion/electron VDFs, n, V, T |
| WISPR | Naval Research Laboratory (NRL) | Wide-field white-light imager; CMEs, streamers, dust |

Marco Velli serves as Observatory Scientist. The instruments are designed for redundancy: FIELDS, SWEAP, ISʘIS each contribute to multiple objectives, so loss of any one suite still allows minimum mission success.

**한국어.** Table 2는 4개 탑재체와 주관 기관을 나열한다 (위 표). Marco Velli가 Observatory Scientist 역할. 탑재체는 중복성으로 설계 — FIELDS, SWEAP, ISʘIS는 다중 목표에 기여하므로, 어느 한 탑재체 상실 시에도 최소 미션 성공 달성 가능.

### §4. Mission Design (pp. 29–35)

**English.** This section is the engineering core.

- **§4.0 Trajectory** (Fig. 13): Launch July 31 – Aug 18, 2018. C3 = 154 km²/s² (highest ever required). Six weeks post-launch: first Venus flyby (VF1, 9/28/2018) → elliptical orbit, perihelion P1 = 35.66 R_sun (12/19/2018... actually first perihelion 11/01/2018 per Fig. 13). Subsequent VFs (VF2 12/22/2019; VF3 7/06/2020; VF4 2/16/2021; VF5 10/15/2021; VF6 8/16/2023; VF7 11/02/2024) walk the perihelion down. Final 3 orbits (22, 23, 24) at **9.86 R_sun**. Backup launch May 2019 adds one extra VGA + ~1 year mission.

- **§4.1 Near-Sun Environment** (p. 31): Irradiance at 9.86 R_sun = 475 × solar constant ≈ 649 kW/m². Sun is NOT a point source at this distance — it subtends ~5.82° umbra cone. Penumbra exposure of secondary solar array section reduces effective irradiance to "only" ~25 Suns at end-of-life, with 70° wing incidence angle. Dust impact concerns: high-velocity particles in dust cloud near Sun.

- **§4.2 Spacecraft** (Figs. 15, 16; p. 32): 685 kg launch mass, ~3 m tall, 2.3 m TPS diameter. **Anti-ram side**: SWEAP SPC (peeks past TPS), Solar Array Cooling System, High Gain Antenna, Solar Array Wings (2). **Ram side**: FIELDS Antenna (4), ISʘIS Suite (EPI-Lo, EPI-Hi), FIELDS Magnetometers (3), SWEAP SPAN A+, WISPR. Cooling: water pumped through titanium platen → 4 CSPRs (Cooling System Primary Radiators), dissipates ~6500 W at perihelion, holds solar cells <150 °C. Communications: 0.6 m Ka-band HGA on anti-ram side. Avionics: triple SBC (single board computer) processor — prime + hot spare + backup spare; redundant X/Ka radios.

- **§4.3 Technology Development** (pp. 35–37): TPS = carbon-carbon composite + carbon foam core + alumina coating. Tested in launch + thermal environments. Solar arrays validated under 25 Suns at end-of-life conditions. SACS (Solar Array Cooling System) tested at full scale.

- **§4.4 Mission Operations**: Two phases per orbit — solar encounter (<0.25 AU = 53.7 R_sun) at 100% instrument duty, then cruise/downlink phase. SSRs (solid-state recorders) sized for 2 orbits to balance downlink. Off-Sun pointing 45° beyond 0.82 AU during cruise to maintain bus thermal range.

**한국어.** 이 절은 공학적 핵심이다.

- **§4.0 궤적** (Fig. 13): 발사 2018년 7월 31일 – 8월 18일. C3 = 154 km²/s² (역대 최고). 발사 후 6주: 첫 금성 통과 (VF1, 2018-09-28) → 타원 궤도, 첫 근일점 P1 = 35.66 R_sun. 이후 VF2 (2019-12-22), VF3 (2020-07-06), VF4 (2021-02-16), VF5 (2021-10-15), VF6 (2023-08-16), VF7 (2024-11-02)로 근일점 점진적 감소. 마지막 3궤도 (22, 23, 24) **9.86 R_sun**. 백업 발사 2019년 5월은 추가 VGA + ~1년 미션.

- **§4.1 근태양 환경** (p. 31): 9.86 R_sun 복사조도 = 475 × 태양 상수 ≈ 649 kW/m². 이 거리에서 태양은 점광원 아님 — ~5.82° umbra 원뿔 형성. 보조 태양전지판 섹션의 penumbra 노출로 유효 복사조도가 EOL 시 "단지" ~25 Suns로 감소, 70° wing 입사각. 먼지 충돌 우려: 태양 근처 먼지 구름의 고속 입자.

- **§4.2 우주선** (Figs. 15, 16; p. 32): 발사 질량 685 kg, 높이 ~3 m, TPS 직경 2.3 m. **Anti-ram 면**: SWEAP SPC (TPS 뒤에서 노출), 태양전지판 냉각계, 고이득 안테나, 태양전지판 날개 (2). **Ram 면**: FIELDS 안테나 (4), ISʘIS Suite (EPI-Lo, EPI-Hi), FIELDS 자력계 (3), SWEAP SPAN A+, WISPR. 냉각: 티타늄 플레이튼을 통한 펌프 물 → 4개 CSPR (Cooling System Primary Radiators), 근일점에서 ~6500 W 방출, 태양전지 <150 °C 유지. 통신: anti-ram 면 0.6 m Ka-band HGA. 항전: 삼중 SBC 프로세서 — prime + hot spare + backup spare; 중복 X/Ka 라디오.

- **§4.3 기술 개발** (pp. 35–37): TPS = carbon-carbon 복합재 + 탄소 폼 코어 + 알루미나 코팅. 발사 + 열환경 시험. 태양전지판 EOL 25 Suns 조건 검증. SACS 풀스케일 시험.

- **§4.4 미션 운용**: 궤도당 2단계 — 태양 조우(<0.25 AU = 53.7 R_sun) 시 탑재체 100% 듀티, 이후 크루즈/다운링크 단계. SSR은 2궤도 분량으로 다운링크 균형. 0.82 AU 이외에서 45° off-Sun 포인팅으로 본체 열 범위 유지.

### §5. SPP Science Operations (pp. 38–41)

**English.** Decoupled operations: each Science Operations Center (SOC) handles its own instrument independently of bus and other instruments. The Mission Operations Center (MOC) at JHU/APL handles bus, command, performance assessment. Communications via NASA Deep Space Network (DSN). Science planning by Orbit Planning Team (OPT) → Orbit Operations Template (OOT) → Instrument Teams (Fig. 20). Data products: Levels 0–4 per instrument (Table 4). SPP Science Data Portal serves community access.

**한국어.** 분리된 운용: 각 Science Operations Center (SOC)이 본체 및 다른 탑재체와 독립적으로 자기 탑재체 처리. JHU/APL의 Mission Operations Center (MOC)이 본체, 명령, 성능 평가 담당. 통신은 NASA DSN. 과학 계획은 Orbit Planning Team (OPT) → Orbit Operations Template (OOT) → 탑재체 팀 (Fig. 20). 데이터 산출물: 탑재체별 Level 0–4 (Table 4). SPP Science Data Portal은 커뮤니티 접근 제공.

### §5.1 Orbit Time Budget (Table 3, p. 31)

**English.** Table 3 enumerates time spent inside radial thresholds for each of the 24 solar passes:

| Threshold | Total time across 24 orbits |
|-----------|-----------------------------|
| <30 R_sun | **2130.85 hours** (~88.8 days) |
| <20 R_sun | **937.58 hours** (~39.1 days) |
| <15 R_sun | **440.03 hours** (~18.3 days) |
| <10 R_sun | **14.85 hours** |

The first two perihelia (P1, P2) at 35.66 R_sun produce zero time inside 30 R_sun (just outside). The middle orbits (10–16) at 13.28 R_sun give ~107 hr/orbit inside 30 R_sun. The final three orbits (22, 23, 24) at 9.86 R_sun give ~4.95 hr/orbit inside 10 R_sun — the precious sub-Alfvénic sampling. Color coding in the original table shows the perihelion stages.

**한국어.** Table 3은 24회 태양 통과 각각에 대해 반경 임계값 이내 체류 시간을 나열한다 (위 표 참조). 처음 두 근일점 (P1, P2)은 35.66 R_sun으로 30 R_sun 이내 시간이 0 (바로 바깥). 중간 궤도 (10–16)는 13.28 R_sun으로 30 R_sun 이내 ~107시간/궤도. 마지막 3궤도 (22, 23, 24)는 9.86 R_sun으로 10 R_sun 이내 ~4.95시간/궤도 — 귀중한 준-Alfvén 이하 표본. 원문 표의 색 부호는 근일점 단계를 나타낸다.

### §6. Summary and Conclusions (pp. 41–43)

**English.** Reaffirms that SPP addresses coronal heating questions outstanding since Grotrian (1939) and Saha (1942). The seven Venus encounters lower the perihelion to <10 R_sun over 24 orbits. Beyond fundamental physics, SPP enables better space weather prediction — solar activity affects magnetospheres, aurora, satellite communications, power grids, pipelines, airline radiation exposure, astronaut safety, and possibly the heliospheric termination shock and interstellar medium. The conclusion frames the mission as "humanity's first visit to a star."

**한국어.** SPP가 Grotrian (1939)과 Saha (1942) 이래 미해결인 코로나 가열 질문을 해결함을 재확인. 7회 금성 조우로 24궤도에 걸쳐 근일점을 <10 R_sun까지 낮춤. 기초 물리를 넘어, SPP는 더 나은 우주 날씨 예측을 가능케 함 — 태양 활동은 자기권, 오로라, 위성 통신, 전력망, 송유관, 항공 방사선 노출, 우주비행사 안전, 그리고 헬리오스피어 종단 충격파 및 성간 매질에 영향. 결론은 미션을 "별을 향한 인류 최초의 방문"으로 자리매김한다.

---

## 3. Key Takeaways / 핵심 시사점

**1. 9.86 R_sun is the engineering-imposed sweet spot / 9.86 R_sun은 공학적으로 결정된 최적점.**
**EN.** This is just inside the predicted Alfvén critical surface (~12–13 R_sun), where the solar wind transitions from sub-Alfvénic (waves can travel both ways) to super-Alfvénic. PSP first crossed it on 28 April 2021 (orbit 8, post-publication confirmation).
**KR.** 예측된 Alfvén 임계면(~12–13 R_sun) 바로 안쪽으로, 태양풍이 준-Alfvén 이하(파동 양방향 전파 가능)에서 초-Alfvén으로 전이하는 지점. PSP는 2021년 4월 28일 (궤도 8) 처음 통과 (논문 발표 후 확인).

**2. Venus gravity assists trade depth for breadth / 금성 중력 도움은 깊이를 폭과 교환.**
**EN.** Each VGA shrinks orbital energy by ~0.5–4 R_sun in perihelion — six VGA-spaced phases (35.66 → 27.85 → 20.35 → 15.98 → 11.42 → 9.86 R_sun). 24 orbits give 2130.85 hours inside 30 R_sun (Table 3) vs SP2005's ~160 hours.
**KR.** 각 VGA가 근일점에서 궤도 에너지를 ~0.5–4 R_sun씩 감소 — 6개 VGA-간격 단계(35.66 → 27.85 → 20.35 → 15.98 → 11.42 → 9.86 R_sun). 24궤도가 30 R_sun 이내 2130.85시간 제공 (Table 3) vs SP2005의 ~160시간.

**3. The TPS is a passive thermal device, not active cooling / TPS는 능동 냉각이 아닌 수동 열소자.**
**EN.** Carbon-carbon front face emits in the IR enough to balance 475 Suns at ~1377 °C. The carbon foam core has very low conductivity, so heat doesn't reach the bus. Active cooling (water loop) is only for the small solar array sections that must remain illuminated for power.
**KR.** Carbon-carbon 전면이 IR 방출을 통해 475 Suns를 ~1377 °C에서 평형. 탄소 폼 코어는 전도도가 매우 낮아 본체에 열이 도달하지 않음. 능동 냉각(물 회로)은 전력을 위해 조사되어야 하는 소형 태양전지판 섹션에만 사용.

**4. SWEAP SPC must look at the Sun directly / SWEAP SPC는 태양을 직접 응시해야 함.**
**EN.** A Faraday cup measuring the supersonic ion bulk flow can't hide behind the TPS — it pokes out into the solar environment. This is the most thermally extreme instrument, requiring titanium-zirconium-molybdenum (TZM) collector plates and tungsten grids.
**KR.** 초음속 이온 다발 흐름을 측정하는 Faraday 컵은 TPS 뒤에 숨을 수 없음 — 태양 환경으로 돌출. 가장 열적으로 극한 탑재체로, 티타늄-지르코늄-몰리브덴 (TZM) 수집판과 텅스텐 격자를 요구.

**5. WISPR provides the only true remote sensing / WISPR이 유일한 진정한 원격 탐사.**
**EN.** Other suites are in-situ. WISPR's wide-field white-light imager photographs coronal structure 13.5° to 108° from Sun-spacecraft line, allowing context for in-situ measurements ("here is the streamer that this plasma came from") and mapping CME progression at unprecedented spatial resolution near the Sun.
**KR.** 다른 탑재체는 현장 측정. WISPR의 광시야 백색광 영상기는 태양-우주선 선 13.5° ~ 108° 코로나 구조를 촬영, 현장 측정 맥락 ("이 플라스마가 나온 streamer가 여기"), 태양 근처 전례 없는 공간 해상도로 CME 진행 추적.

**6. Decoupled SOC operations enable institutional autonomy / 분리된 SOC 운용은 기관 자율성을 가능케 함.**
**EN.** Each instrument PI institution operates its own SOC independently — UC Berkeley for FIELDS, SAO for SWEAP, SwRI for ISʘIS, NRL for WISPR. The MOC at JHU/APL coordinates only the bus. This pattern, used in Cassini and earlier missions, scales well for instrument teams with deep heritage.
**KR.** 각 탑재체 PI 기관이 자체 SOC을 독립 운용 — FIELDS는 UC Berkeley, SWEAP는 SAO, ISʘIS는 SwRI, WISPR는 NRL. JHU/APL MOC은 본체만 조정. Cassini 등 이전 미션에서 사용된 이 패턴은 깊은 유산을 가진 탑재체 팀에 잘 확장.

**7. Single fault tolerance, not single point of failure / 단일 결함 내성, 단일 실패 지점 없음.**
**EN.** Triple-redundant SBC (prime + hot spare + cold spare), redundant radios, redundant IMU. Even with one processor failed, two remain. The 5.82° umbra cone with 2° pointing margin tolerates wheel-failure scenarios.
**KR.** 삼중 중복 SBC (prime + hot spare + cold spare), 중복 라디오, 중복 IMU. 한 프로세서 고장에도 둘이 남음. 2° 포인팅 여유를 가진 5.82° umbra 원뿔이 휠 고장 시나리오를 견딤.

**8. SPP samples 60+% of a solar cycle / SPP는 태양 주기의 60% 이상을 표본.**
**EN.** 7-year mission from 2018 covers from solar minimum (cycle 24/25 transition) toward next maximum (~2024–2025). This is critical because the corona's structure changes radically across the cycle (Fig. 1: Ulysses minimum vs. maximum).
**KR.** 2018년부터 7년 미션이 태양 최소기(주기 24/25 전이)에서 다음 최대기(~2024–2025)까지 포괄. 주기에 걸쳐 코로나 구조가 급변하므로 (Fig. 1: Ulysses 최소기 vs 최대기) 핵심적.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Inverse-square law for solar irradiance / 태양 복사조도 역제곱 법칙

$$F(r) = F_\oplus \left( \frac{r_\oplus}{r} \right)^2$$

**EN.** F_⊕ = 1366 W/m² is the solar constant at 1 AU (r_⊕ = 1 AU = 215 R_sun). At r = 9.86 R_sun = 0.0459 AU:

$$F(9.86\,R_\odot) = 1366 \times (1/0.0459)^2 \approx 1366 \times 474.6 \approx 6.49 \times 10^5\,\text{W/m}^2$$

This is the famous "**475 Suns**" figure. Total solar luminosity L_⊙ = 3.828 × 10²⁶ W is recovered from F × 4πr² at any r.

**KR.** F_⊕ = 1366 W/m²는 1 AU에서의 태양 상수 (r_⊕ = 1 AU = 215 R_sun). r = 9.86 R_sun = 0.0459 AU에서:

$$F(9.86\,R_\odot) \approx 6.49 \times 10^5\,\text{W/m}^2$$

이것이 유명한 "**475 Suns**" 수치. 총 태양 광도 L_⊙ = 3.828 × 10²⁶ W는 어떤 r에서든 F × 4πr²로 복원.

### 4.2 TPS radiative equilibrium / TPS 복사 평형

For a flat carbon-carbon plate exposed to flux F with absorptivity α and emissivity ε (assuming negligible conduction to rear face):

$$\alpha F = \varepsilon \sigma T^4$$

$$T = \left( \frac{\alpha F}{\varepsilon \sigma} \right)^{1/4}$$

**EN.** With α/ε ≈ 1 (high-temperature carbon and alumina), σ = 5.670 × 10⁻⁸ W m⁻² K⁻⁴, F = 6.49 × 10⁵ W/m²:

$$T = (6.49 \times 10^5 / 5.670 \times 10^{-8})^{1/4} \approx (1.145 \times 10^{13})^{1/4} \approx 1838\,\text{K} \approx 1565\,°\text{C}$$

The paper quotes 1377 °C = 1650 K — slightly lower because TPS includes some lateral conduction, the alumina coating reflects some incident flux (lower α), and the geometry is not perfectly flat.

**KR.** α/ε ≈ 1 (고온 탄소 + 알루미나 가정), σ = 5.670 × 10⁻⁸ W m⁻² K⁻⁴, F = 6.49 × 10⁵ W/m²:

$$T \approx 1838\,\text{K} \approx 1565\,°\text{C}$$

논문 인용 값 1377 °C = 1650 K는 다소 낮음 — TPS의 일부 측면 전도, 알루미나 코팅의 입사 일부 반사 (낮은 α), 완벽히 평면 아닌 형상 때문.

### 4.3 Vis-viva equation for highly elliptical orbit / 고이심률 궤도 vis-viva 방정식

$$v^2 = G M_\odot \left( \frac{2}{r} - \frac{1}{a} \right)$$

**EN.** With GM_⊙ = 1.327 × 10²⁰ m³/s², for the final SPP orbit with perihelion r_p = 9.86 R_sun and aphelion r_a ≈ 0.73 AU (period 88 days):
- a = (r_p + r_a)/2 ≈ (0.0459 + 0.73)/2 = 0.388 AU
- At perihelion, v_p ≈ √(GM_⊙(2/r_p − 1/a)) ≈ 192 km/s
- The paper quotes **195 km/s** — fastest human-built object ever.

**KR.** GM_⊙ = 1.327 × 10²⁰ m³/s², SPP 최종 궤도 r_p = 9.86 R_sun, r_a ≈ 0.73 AU (주기 88일):
- a ≈ 0.388 AU
- 근일점 v_p ≈ 192 km/s
- 논문 인용 **195 km/s** — 인간이 만든 가장 빠른 물체.

### 4.4 Alfvén speed and critical surface / Alfvén 속도와 임계면

$$v_A = \frac{B}{\sqrt{\mu_0 \rho}} = \frac{B}{\sqrt{\mu_0 m_p n_p}}$$

**EN.** The Alfvén critical surface r_A is where solar wind speed v_sw equals v_A. Below r_A, MHD waves can propagate sunward; above, the wind sweeps everything outward. SPP samples both sides of r_A — predicted ~12–13 R_sun by Verdini et al. (2009). Typical values at r_A: B ≈ 100 nT × (R_sun/r)², n_p ≈ 10⁵–10⁶ cm⁻³ at corona base.

**KR.** Alfvén 임계면 r_A는 태양풍 속도 v_sw = v_A인 곳. r_A 아래에서는 MHD 파동이 태양 쪽으로 전파 가능; 위에서는 풍이 모두 외향 운반. SPP는 r_A 양측을 표본 — Verdini et al. (2009) 예측 ~12–13 R_sun. r_A 전형 값: B ≈ 100 nT × (R_sun/r)², n_p ≈ 10⁵–10⁶ cm⁻³ (코로나 기저).

### 4.5 Worked example — flux ratio at perihelion / 근일점 flux 비율 계산 예제

**English.** Compute the brightness factor at SPP's closest approach relative to Earth:

Step 1: Convert 9.86 R_sun to AU.
- 1 R_sun = 6.957 × 10⁸ m, 1 AU = 1.496 × 10¹¹ m
- 9.86 R_sun = 9.86 × 6.957 × 10⁸ = 6.860 × 10⁹ m
- = 6.860 × 10⁹ / 1.496 × 10¹¹ = 0.04586 AU

Step 2: Apply inverse-square ratio.
- F(perihelion) / F(Earth) = (1 / 0.04586)² = (21.81)² = **475.5**

Step 3: Multiply by solar constant.
- F(perihelion) = 475.5 × 1361 W/m² = **6.471 × 10⁵ W/m²**

This is enough thermal flux to vaporize aluminum, melt most steels at the surface, and require a ~1377 °C TPS surface temperature for radiative balance.

**한국어.** SPP 최근접점에서 지구 대비 밝기 비율 계산:

1단계: 9.86 R_sun → AU 변환.
- 9.86 R_sun = 0.04586 AU

2단계: 역제곱 비율 적용.
- F(근일점) / F(지구) = (21.81)² = **475.5**

3단계: 태양 상수와 곱.
- F(근일점) = 475.5 × 1361 W/m² = **6.471 × 10⁵ W/m²**

이는 알루미늄을 기화시키고 대부분의 강철 표면을 녹이며, 복사 평형을 위해 ~1377 °C TPS 표면 온도를 요구하는 충분한 열속이다.

### 4.6 Mission energy budget / 미션 에너지 수지

C3 (characteristic energy) for launch:

$$C_3 = v_\infty^2 = v_\text{launch}^2 - \frac{2GM_\oplus}{R_\oplus}$$

**EN.** SPP requires C3 = 154 km²/s² (highest ever for a deep-space mission). Compare: Pluto's New Horizons C3 = 158 km²/s², Voyager 1 C3 ≈ 109 km²/s². The Delta IV Heavy + STAR-48B upper stage was the only US launch system that could deliver this on the launch window.

**KR.** SPP는 C3 = 154 km²/s² 요구 (역대 심우주 미션 최고). 비교: New Horizons C3 = 158 km²/s², Voyager 1 C3 ≈ 109 km²/s². Delta IV Heavy + STAR-48B 상단이 발사 창에 이를 전달할 수 있는 유일한 미국 발사체.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1939          1958            1962-2008          2008             2014           2018          2024
 |             |                 |                 |                |               |             |
Grotrian     Simpson         eight major       SPP STDT       SPP project     PSP launch    PSP final
Saha         Committee:      science studies   Report          confirmed      (this paper's  perihelion
discover     "build a        culminating in    (McComas):     by NASA         schedule)      9.86 R_sun
T_corona     solar probe!"   redesign from    Venus-VGA                                     (post-paper
~10^6 K                     polar Jupiter to   24-orbit                                      milestone)
                            equatorial Venus  baseline
                            VGA                                  
                                                                 ↓
                                                          Fox+ 2016 (THIS PAPER)
                                                          consolidates objectives,
                                                          mission, instruments
```

**English.** The paper is the cornerstone reference for the entire PSP literature. Every subsequent PSP science paper (Bale 2019, Kasper 2019, Howard 2019, Whittlesey 2020, etc.) cites Fox+ 2016 for mission context.

**한국어.** 본 논문은 전체 PSP 문헌의 초석 참조이다. 모든 후속 PSP 과학 논문 (Bale 2019, Kasper 2019, Howard 2019, Whittlesey 2020 등)이 미션 맥락을 위해 Fox+ 2016을 인용한다.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper | Topic | Connection |
|-------|-------|-----------|
| **Parker (1958)** | Solar wind theory | The mission tests Parker's predictions in situ; spacecraft renamed "Parker Solar Probe" in 2017 to honor him |
| **McComas et al. (2008)** | SPP STDT Report | Original mission design study cited throughout |
| **Bale et al. (2016)** | FIELDS instrument paper | Companion paper detailing electric/magnetic fields suite |
| **Kasper et al. (2016)** | SWEAP instrument paper | Companion paper detailing plasma suite |
| **McComas et al. (2014)** | ISʘIS instrument paper | Companion paper detailing energetic particle suite |
| **Vourlidas et al. (2015)** | WISPR instrument paper | Companion paper detailing white-light imager |
| **Verdini & Velli (2007); Verdini et al. (2009)** | Alfvén turbulence | Models SPP will test (Figs. 2–3) |
| **Bale et al. (2009); Matteini et al. (2007)** | Proton T anisotropy | Constraints SPP will refine (Fig. 5) |
| **Bruno & Carbone (2013)** | Solar wind turbulence | Spectral break evolution (Fig. 6) |
| **Howard et al. (2019)** | First WISPR results | Post-launch validation of mission's imaging promise |

---

## 6.5 Post-Publication Reality Check (2018–2025) / 발표 후 실제 결과 점검 (2018–2025)

**English.** Although Fox+ 2016 is a pre-launch document, much can now be checked against post-launch reality:

- **Launch**: Successfully on Aug 12, 2018 (within the July 31 – Aug 18 window stated). Renamed Parker Solar Probe in May 2017 (after Eugene Parker, who attended launch).
- **First perihelion**: P1 = 35.7 R_sun on 6 Nov 2018 — matched mission design within hours.
- **Alfvén surface crossing**: First documented sub-Alfvénic encounter on 28 Apr 2021 (orbit 8) at ~18 R_sun — somewhat farther than the 12–13 R_sun predicted by Verdini, but well within the corona.
- **Switchbacks**: Bale et al. (2019, *Nature*) reported magnetic field reversals ("switchbacks") that were unanticipated at the time of Fox+ 2016 — a transformative discovery.
- **Final perihelion**: Achieved 9.86 R_sun on 24 Dec 2024 — matching mission design exactly.
- **Mission extension**: Beyond the 7-year nominal, NASA extended PSP operations into the late 2020s.

**한국어.** Fox+ 2016은 발사 전 문서이지만, 많은 부분을 발사 후 실제와 대조 가능:

- **발사**: 2018년 8월 12일 성공적으로 발사 (논문에 기술된 7월 31일 – 8월 18일 창 내). 2017년 5월 Parker Solar Probe로 개명 (Eugene Parker 박사 명명, 그는 발사를 직접 참관).
- **첫 근일점**: P1 = 35.7 R_sun (2018-11-06) — 미션 설계와 시간 단위 일치.
- **Alfvén 면 통과**: 2021-04-28 (궤도 8) ~18 R_sun에서 첫 준-Alfvén 이하 조우 — Verdini 예측 12–13 R_sun보다 다소 멀지만 코로나 내부.
- **Switchback**: Bale et al. (2019, *Nature*)이 Fox+ 2016 시점 예상되지 않았던 자기장 역전 ("switchback")을 보고 — 변혁적 발견.
- **최종 근일점**: 2024-12-24 9.86 R_sun 달성 — 미션 설계와 정확히 일치.
- **미션 연장**: 7년 명목 이후 NASA가 PSP 운용을 2020년대 후반까지 연장.

## 7. References / 참고문헌

- Fox, N.J., Velli, M.C., Bale, S.D., Decker, R., Driesman, A., Howard, R.A., Kasper, J.C., Kinnison, J., Kusterer, M., Lario, D., Lockwood, M.K., McComas, D.J., Raouafi, N.E., Szabo, A., "The Solar Probe Plus Mission: Humanity's First Visit to Our Star", *Space Science Reviews* **204**, 7–48, 2016. doi:10.1007/s11214-015-0211-6
- Grotrian, W., "Zur Frage der Deutung der Linien im Spektrum der Sonnenkorona", *Naturwissenschaften* 27, 214 (1939).
- Saha, M.N., "The Solar Corona", *Nature* 149, 524–525 (1942).
- Parker, E.N., "Dynamics of the interplanetary gas and magnetic fields", *ApJ* 128, 664 (1958).
- McComas, D.J., et al., "Understanding coronal heating and solar wind acceleration: case for in situ near-Sun measurements", *Rev. Geophys.* 45, 1004 (2007).
- McComas, D.J., et al., "Weakest solar wind of the space age and the current 'mini' solar maximum", *ApJ* 779, 2 (2013).
- Verdini, A., Velli, M., "Alfvén waves and turbulence in the solar atmosphere and solar wind", *ApJ* 662, 669–676 (2007). doi:10.1086/510710
- Verdini, A., Velli, M., Buchlin, E., "Turbulence in the sub-Alfvénic solar wind driven by reflection of low-frequency Alfvén waves", *ApJ* 700, L39 (2009).
- Bale, S.D., Goetz, K., Harvey, P.R., et al., "The FIELDS instrument suite for Solar Probe Plus", *Space Sci. Rev.* (2015) [in same Topical Issue].
- Kasper, J.C., Abiad, R., Austin, G., et al., "Solar wind electrons alphas and protons (SWEAP) investigation", *Space Sci. Rev.* (2015) [in same Topical Issue]. doi:10.1007/s11214-015-0206-3
- McComas, D.J., et al., "Integrated Science Investigation of the Sun (ISIS): design of the energetic particle investigation", *Space Sci. Rev.* (2014). doi:10.1007/s11214-014-0059-1
- Vourlidas, A., Howard, R.A., Plunkett, S.P., et al., "The Wide-Field Imager for Solar Probe Plus (WISPR)", *Space Sci. Rev.* (2015). doi:10.1007/s11214-014-0114-y
- Bale, S.D., Kasper, J.C., Howes, G.G., et al., "Magnetic fluctuation power near proton temperature anisotropy instability thresholds in the solar wind", *Phys. Rev. Lett.* 103, 211101 (2009).
- Matteini, L., Landi, S., Hellinger, P., et al., "Evolution of the solar wind proton temperature anisotropy from 0.3 to 2.5 AU", *Geophys. Res. Lett.* 34, L20105 (2007).
- Bruno, R., Carbone, V., "The solar wind as a turbulence laboratory", *Living Rev. Solar Phys.* 10, 2 (2013).
