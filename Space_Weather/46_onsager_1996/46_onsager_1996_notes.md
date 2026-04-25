---
title: "Operational Uses of the GOES Energetic Particle Detectors"
authors: ["T. G. Onsager", "R. Grubb", "J. Kunches", "L. Matheson", "D. Speich", "R. Zwickl", "H. Sauer"]
year: 1996
journal: "Proc. SPIE 2812 (GOES-8 and Beyond), 281–290"
doi: "10.1117/12.254075"
topic: Space_Weather
tags: [GOES, EPS, HEPAD, energetic_particles, SEP, deep_dielectric_charging, NOAA_SEC, geosynchronous, operational, alerts_warnings]
status: completed
date_started: 2026-04-25
date_completed: 2026-04-25
---

# 46. Operational Uses of the GOES Energetic Particle Detectors / GOES 에너지 입자 검출기의 운영적 활용

---

## 1. Core Contribution / 핵심 기여

This paper is a definitive 1996 technical-operations description of the GOES-8/9 Energetic Particle Sensor (EPS) and High-Energy Proton and Alpha Detector (HEPAD), authored by the NOAA Space Environment Center (SEC) team that actually ran the operations. It documents three things that together define modern U.S. civilian space-weather operations: (i) the *instrument suite* — a passively shielded silicon-telescope plus three dome modules covering protons 0.7–900 MeV, alphas 4–500 MeV, and electrons >0.6 MeV; supplemented by a HEPAD with Cherenkov PMT covering protons 330–700+ MeV and alphas 2560–3400+ MeV; (ii) the *data-processing pipeline* — count-rate to flux conversion, 10-day low-pass GCR-background subtraction, secondary-response correction using a power-law spectral assumption, and spectrum-dependent effective geometric factors for wide-band electron channels; and (iii) the *operational uses* — real-time alerts (>10 pfu of >10 MeV protons, >1 pfu of >100 MeV protons, >1000 pfu of >2 MeV electrons) and warnings, real-time data distribution to USAF and the Internet, and archival for post-event spacecraft anomaly forensics and long-term cosmic-ray studies.

본 논문은 NOAA Space Environment Center가 실제로 운영해 온 GOES-8/9의 EPS 및 HEPAD 기기군을 1996년 시점에서 권위 있게 정리한 기술-운영 문헌입니다. 본문은 다음 세 가지를 동시에 정의합니다: (1) **기기 구성** — 수동 차폐된 실리콘 망원경(0.7–15 MeV 양성자, 4–61 MeV 알파)과 세 개의 돔 모듈(15–900 MeV 양성자, 60–500 MeV 알파, >0.6/2/4 MeV 전자), 그리고 Cherenkov PMT를 이용한 HEPAD(>330 MeV 양성자, >2560 MeV 알파). (2) **데이터 처리 파이프라인** — 카운트율→플럭스 변환, 10일 저역 통과(low-pass) 필터를 통한 우주선(GCR) 배경 제거, 인접 채널 간 멱함수 가정 기반의 부수 응답(secondary response) 보정, 광대역 전자 채널의 스펙트럼 종속 유효 기하인자 계산. (3) **운영적 사용** — 실시간 alerts(>10 MeV 양성자 10 pfu, >100 MeV 양성자 1 pfu, >2 MeV 전자 1000 pfu), warnings(임박 사건 사전 통보), USAF·인터넷 배포, 그리고 위성 이상 사후 분석 및 장기 우주선 연구를 위한 아카이빙입니다. 이 운영 체계는 오늘날 SWPC의 S-스케일(SEP scale)과 GOES-R 시리즈의 검증된 직계 조상입니다.

---

## 2. Reading Notes / 읽기 노트

### Part I: Instrument Description (§1) / 기기 설명

**EPS Telescope (Figure 1, p.282)** / **EPS 망원경**
- Two silicon surface barrier detectors: 50 μm (front) + 500 μm (back).
- 두 개의 실리콘 표면 장벽 검출기(앞 50 μm, 뒤 500 μm).
- Field of view (FOV): ~35° half-angle.
- Sweeping magnets exclude electrons below ~100 keV; aluminum foil acts as light shield. Passively shielded — *no active anti-coincidence*.
- 약 100 keV 이하 전자를 자석으로 배제하고, 알루미늄 박막으로 광 차폐. 능동(anti-coincidence) 차폐는 없음.
- Channels: P1 (0.7–4.2 MeV), P2 (4.2–8.7 MeV), P3 (8.7–14.5 MeV) protons; A1 (4–10), A2 (10–21), A3 (21–61 MeV) alphas.

**EPS Dome (Figure 2, p.284)** / **EPS 돔**
- Three independent modules (D3, D4, D5), each two 1500-μm Si detectors.
- 세 개의 독립 돔 모듈(D3, D4, D5), 각각 1500 μm 두께 Si 검출기 두 장.
- Different moderators set energy thresholds: D3 has 0.125 g/cm² aluminum (low-E threshold); D4 has 1.57 g/cm² aluminum; D5 has 8.0 g/cm² copper for the highest-energy protons.
- 모더레이터 두께가 임계 에너지를 결정. D3는 가장 얇은 Al, D4는 두꺼운 Al, D5는 두꺼운 Cu(고에너지용).
- FOV: ~30° half-angle in equatorial plane × ~60° in elevation.
- Channels: P4 (15–40), P5 (38–82), P6 (84–200), P7 (110–900 MeV) protons; A4–A6 alphas; E1 (>0.6), E2 (>2), E3 (>4 MeV) electrons.

**HEPAD (Figure 3, p.285)** / **HEPAD**
- Telescope geometry with two Si surface-barrier detectors + Cherenkov radiation photomultiplier tube.
- 망원경 형상의 두 개의 Si 검출기 + Cherenkov 광전증배관.
- FOV ~34° half-angle; mounted anti-Earthward to reduce Earth-albedo contamination.
- Channels: P8 (330–420), P9 (420–510), P10 (510–700), P11 (>700 MeV) protons; A7 (2560–3400), A8 (>3400 MeV) alphas.

**Particle/energy discrimination** / **입자·에너지 판별**: coincidence/anti-coincidence logic on the two-detector pulse pairs, but because the shielding is passive, *secondary* (out-of-aperture / out-of-energy) particles can contaminate counts. Table 1 (p.283) lists secondary energy ranges and geometric factors that the correction algorithm must subtract. / 두 검출기 펄스의 동시계수/반동시계수 논리로 입자와 에너지를 판별하나, 수동 차폐만 있어 1차 영역 외 입자가 차폐를 뚫고 들어와 카운트를 오염시킬 수 있음. Table 1에 부수 에너지 범위와 기하인자가 명시되어 있고, 이를 보정 알고리즘에서 제거.

**Pointing change since GOES-8** / **GOES-8 이후 시야 변경**: GOES-8부터는 회전(spinning)이 멈추고 3축 안정화로 전환되어, EPS는 서쪽, HEPAD는 지구 반대 방향으로 *고정* 시야를 가짐. 이는 SEP 초기의 강한 비등방성을 측정할 때 운영적 제약을 만듦.
GOES-8 onward is body-fixed (no longer spinning); EPS points westward and HEPAD anti-Earthward, which constrains how SEP-onset anisotropies are sampled.

**Table 1 numerical highlights** / **Table 1 핵심 수치**:
- Channel response factors range from 0.194 (P1) to 839 cm² sr MeV (P7) — the largest is the broad-band P7 capturing >100 MeV "tail."
- HEPAD P11 and A8 quote a single geometric factor G = 0.73 cm² sr because they are integral channels.
- Derived integral proton flux levels output to operators: >1, >5, >10, >30, >50, >60, >100 MeV.

### Part II: Data Processing (§2) / 데이터 처리

**Telemetry & cadence** / **텔레메트리·주기**: data delivered to NOAA SEC Boulder in real time at ~20 s resolution; archived as raw telemetry and 5-minute averages. / 실시간 약 20초 분해능으로 SEC Boulder로 수신, 원시 텔레메트리와 5분 평균으로 저장.

**Electron processing (spectrum-dependent G_eff)** / **전자 처리(스펙트럼 종속 유효 G)**:
- Wide energy passbands force G to depend on local spectral shape.
- Procedure: estimate spectral slope from neighboring electron-channel pairs, then look up G_eff from laboratory calibration.
- E2 (>2 MeV) is special — response is ~energy-independent in this dome, so a *fixed* G = 0.05 cm² sr is used.
- Typical values: E1 G ≈ 0.075 ± 0.025 cm² sr; E3 G ≈ 0.017 ± 0.006 cm² sr.
- A linear proton-flux contamination correction (using 8.7–200 MeV protons) is also applied to electron channels.
- 전자 채널은 광대역이므로 G가 스펙트럼 모양에 의존. 인접 채널 쌍에서 기울기를 추정해 실험실 보정으로부터 G_eff를 산출. E2(>2 MeV)는 응답이 에너지에 거의 독립이라 G=0.05 cm² sr로 고정. E1, E3는 평균값과 불확도 제시. 양성자 오염은 8.7–200 MeV 양성자 플럭스에 선형 비례하는 보정으로 제거.

**Proton processing** / **양성자 처리**: two corrections.
1. **GCR background subtraction (10-day low-pass)**: rolling 10-day minimum count rate is subtracted from the current rate. To avoid latching onto an artificially elevated minimum during multi-day SEP elevations, the new 10-day minimum is *only* adopted if it is within a specified range of the previous; otherwise the previous minimum is reused. / 직전 10일 동안의 최소 카운트율을 GCR 배경으로 간주해 차감. SEP가 며칠 지속되는 동안 인위적으로 높은 최소값으로 락(lock)되지 않도록, 새 최소값이 직전과 일정 범위 안에 있어야만 갱신; 그렇지 않으면 직전 값을 유지.
2. **Secondary-response correction (top-down, power-law)**: starting from highest-energy P7, assume a power-law spectrum between adjacent channels to estimate the secondary contribution to lower-energy channels. The estimated secondary counts are subtracted, then the algorithm walks down the energy ladder using *corrected* upper-channel counts. Each 5-minute sample is processed *independently* (no temporal coupling). / 가장 높은 P7부터 시작, 인접 채널 간 멱함수 스펙트럼을 가정해 부수 카운트 기여를 추정·차감. 보정된 상위 채널 카운트를 사용해 하위 채널의 부수 응답을 다시 추정. 각 5분 샘플 독립 처리.

**Distribution** / **배포**: real-time delivery to NOAA SEC, USAF 50th Weather Squadron (Colorado Springs), and partial Web availability. Long-term archive at NGDC. / 실시간으로 NOAA SEC, USAF 50기상대대(Colorado Springs)로 전송; 일부는 웹 공개; 장기 보관은 NGDC.

### Part III: Particle Environment at GEO (§3) / 정지궤도 입자 환경

Three categories: (1) trapped, (2) solar-origin, (3) cosmic rays. 입자 환경은 (1) 자기권 포획, (2) 태양 기원, (3) 우주선의 세 부류. Time scales range from minutes (SEP onset, substorm injection) to decades (solar cycle modulation of GCR). / 시간 척도는 분(SEP, 부폭풍)에서 수십 년(태양 주기 GCR 변조)에 이름.

**Figure 4 (p.286) — October 1995 GOES-8** / **Figure 4 — 1995년 10월**:
- Panel 1–3: integral electron flux >0.6, >2, >4 MeV.
- Panel 4–7: differential proton flux 0.7–4, 4–9, 9–15, 15–40 MeV.
- *Diurnal variation* is dramatic in trapped electrons and the lowest proton channel: noon → high flux, midnight → low (because the magnetospheric magnetic field is asymmetric — compressed sunward, stretched anti-sunward, so the GOES drift trajectory dips into different L-shells).
- A *small SEP event* begins ~20 October and lasts a few days, visible in the 4–9, 9–15, 15–40 MeV panels.

**§3.1 Trapped particles** / **자기권 포획 입자**:
- Outer-belt electrons confined by Earth's field for hours–days; injected via ionospheric/magnetotail processes; lost by atmospheric precipitation, charge exchange, magnetopause convection.
- 외부 복사대 전자는 수 시간–수일간 포획. 전리층/자기꼬리에서 주입되고, 상층대기 침전, 전하교환, 자기권계면 통과로 손실.
- Two operational hazards: surface charging by >10 keV electrons (Stevens et al. 1986), deep dielectric charging by >1 MeV electrons (Stassinopoulous & Brucker 1996). Atmospheric chemistry impact via NOx production by precipitating relativistic electrons (Callis et al. 1991).
- 운영적 위협: >10 keV 전자의 표면 충전, >1 MeV 전자의 심부 유전체 충전(ESD); 침전 상대론적 전자의 NOx 생성으로 대기 화학에 영향.
- *Solar-wind-speed correlation*: MeV electron flux at GEO is strongly governed by solar wind speed (Paulikas & Blake 1979). Peak MeV-e flux usually within ~2 days after onset of high-speed stream (Baker et al. 1990, linear prediction filter).
- MeV 전자는 태양풍 속도에 강하게 의존. 고속 흐름(high-speed stream) 도래 후 약 2일 내 정점.
- **Figure 5 (p.287)** — Jan–Mar 1996: GOES-8 >2 MeV electrons (top) and WIND SWE solar wind speed (bottom) show clear correlation. The late-March 1996 prolonged elevated-electron interval coincides with multiple reported spacecraft anomalies (Baker et al. 1996, NASA ISTP Newsletter — including the famous Anik E1 failure). / 1996년 3월 말 장기간 고전자 플럭스 시기에 다수 위성 이상(Anik E1 등) 보고와 일치.

**§3.2 Solar energetic protons** / **태양 에너지 양성자**:
- Likelihood peaks near solar maximum but events occur at any solar-cycle phase (Shea 1988).
- 태양 극대기 부근에서 발생 빈도 최고이나, 임의의 주기 위상에서 발생 가능.
- Acceleration: combination of CME/flare-region acceleration plus subsequent re-acceleration by traveling interplanetary shocks (Kahler 1993).
- 가속 메커니즘: CME/flare 부근 가속 + 행성간 충격파에 의한 재가속.
- Propagation: predominantly along the interplanetary magnetic field (IMF), with **ExB perpendicular drift** (Roelof 1979). Parker spiral connects active region longitude to Earth — Cane et al. (1988) describe how shock geometry vs. connection longitude shapes the temporal flux profile.
- 전파: 주로 IMF를 따라 이동하면서 ExB 수직 드리프트. 능동 영역 경도와 충격파 위치 관계가 도착 플럭스 프로필을 좌우.
- Onset: along-field particles arrive at Earth in tens of minutes — initial flux is *highly anisotropic and field-aligned*; isotropization occurs from scattering far beyond Earth.
- 자기력선을 따라 가장 빠른 입자는 수십 분 내 도착. 초기 플럭스는 자기력선 정렬 방향으로 강하게 비등방.
- **Operational concern**: body-fixed GOES-8+ may *miss the onset* if pointed away from the beamed flux, *underestimate* omnidirectional flux when off-beam, or *overestimate* when on-beam.
- 비회전 GOES-8 이후, 시야가 빔 방향 밖이면 시작을 놓치거나 등방 플럭스를 과소/과대 추정할 수 있음.
- Effects: hazard to astronauts (especially ISS high-inclination orbit during EVA), Single Event Upsets (SEUs), and degradation of solar panels.
- 영향: 우주인 EVA 위험(고경사 궤도 ISS), SEU, 태양전지판 출력 저하.

### Part IV: NOAA SEC Operations (§4) / NOAA SEC 운영

**Table 2 (p.288) operational structure** / **운영 구조**:
| Service / 서비스 | Trigger / 발효 조건 |
|---|---|
| **Alert** (observed) | Proton: >10 pfu @ >10 MeV **or** >1 pfu @ >100 MeV; Electron: >10³ pfu @ >2 MeV |
| **Warning** (imminent) | Proton: >10 pfu @ >10 MeV **or** >1 pfu @ >100 MeV (predicted) |
| Real-time Internet | Integral proton: >1, >5, >10, >30, >50, >60, >100 MeV; Integral electron: >0.6, >2, >4 MeV |
| Real-time USAF | All Table 1 derived quantities |
| Archive | All Table 1 quantities at NGDC, Boulder |

**Threshold rationale** / **임계값 근거**:
- 10 pfu @ >10 MeV: corresponds to onset of significant Polar Cap Absorption (PCA), the well-known HF radio blackout in the polar D-region.
- 10 pfu @ >10 MeV: 의미 있는 Polar Cap Absorption(PCA)의 시작점 — 극지 D-영역 HF 통신 장애.
- 1 pfu @ >100 MeV: heavy-ionizing component thought relevant for SEU and human-dose concerns.
- 1 pfu @ >100 MeV: SEU/인체 선량과 관련된 침투력 강한 성분.
- Electron 1000 pfu @ >2 MeV: chosen via *user community polling* — a moderately low value capturing the majority of customer needs across diverse spacecraft shielding/devices, despite lack of a single physics-derived number for deep dielectric charging.
- 전자 1000 pfu: 사용자 커뮤니티 폴링으로 결정. 차폐와 부품이 위성마다 달라 단일 물리 기준이 없으므로 다수의 운영자 요구를 절충한 값.

**Operational SEP-as-precursor doctrine** / **SEP 선행 지표 운영 원칙**: enhanced energetic-proton flux is used as a *precursor* of likely enhanced geomagnetic activity within ~2 days; SEC combines this with active-region locations (e.g., from X-ray) to issue predictions and to support Johnson Space Center during Shuttle missions and the upcoming ISS. / 양성자 플럭스 상승은 약 2일 내의 지자기 활동 강화를 예고하는 선행 지표로 사용되어, JSC의 Space Shuttle/ISS 운용 지원에 활용.

**Public access** / **대중 접근**: SEC Web site (http://www.sec.noaa.gov in 1996) — current-day and previous two days plots of >0.6 and >2 MeV electron flux; tabular integral fluxes downloadable. / SEC 웹사이트(1996년 당시 sec.noaa.gov)에 당일+직전 2일 플롯과 표 형태 데이터 제공.

**Post-event forensics** / **이상 사후 분석**: SEC and USAF Services Center are routinely contacted when satellite operators see anomalous behavior; archived GOES data underpin the assessment of whether the natural environment caused the anomaly. / 위성 이상 시 SEC와 USAF가 자연환경 기여 여부를 GOES 아카이브로 판정.

**Long-term science applications** / **장기 과학 응용**: HEPAD >200 MeV cosmic-ray monitoring; strong correlation with ground neutron monitors (Sauer 1993); concept of *effective heliospheric potential* enables long-term dose estimates for airline crews (O'Brien et al. 1992) and astronauts. / HEPAD가 >200 MeV 우주선을 지속 감시, 지상 중성자 모니터와 강한 상관. *유효 태양권 전위* 개념으로 항공 승무원/우주인의 장기 누적 선량 추정.

### Part V: Summary (§5) / 요약

GOES energetic-particle instruments serve a dual role: real-time operational input for civilian/military space-weather services, and long-term archive for science. Coverage spans near-Earth, solar, and heliospheric processes. Relativistic electrons and high-energy protons are continuously monitored; alerts issued at hazardous levels; some predictability of geomagnetic activity follows from SEP onset.

GOES 에너지 입자 기기는 실시간 민·군 우주기후 서비스와 장기 과학 아카이브의 이중 역할을 함. 근지구·태양·태양권 전반의 과정을 포괄. 상대론적 전자와 고에너지 양성자를 지속 감시하고, 위험 레벨에서 alerts를 발행하며, SEP 시작으로부터 지자기 활동의 일부 예측이 가능.

---

## 3. Key Takeaways / 핵심 시사점

1. **One instrument suite, three particle populations** — EPS+HEPAD jointly resolve trapped, solar, and cosmic-ray components with energy coverage from 0.7 MeV protons to >3.4 GeV alphas, eliminating the need for separate trapped-vs-SEP-vs-GCR monitors.
   하나의 기기 세트로 세 모집단 — EPS+HEPAD가 0.7 MeV 양성자부터 >3.4 GeV 알파까지 포괄해 포획 입자, 태양 입자, 우주선을 단일 시스템에서 동시에 측정.

2. **Passive shielding forces a nontrivial correction pipeline** — the absence of active anti-coincidence means *secondary response* is unavoidable; the iterative top-down (P7→P1) power-law correction is the canonical method that downstream products inherit.
   수동 차폐로 인한 정교한 보정 파이프라인 — 능동 anti-coincidence가 없어 부수 응답이 필연이고, P7부터 아래로 멱함수 보정을 반복하는 알고리즘이 표준이 됨.

3. **The 10-day low-pass GCR background subtraction is operationally critical** — the safeguard that prevents latching onto an SEP-elevated minimum is what keeps the small-event detection (e.g., the October 20, 1995 event in Figure 4) sensitive.
   10일 저역 통과 GCR 배경 제거가 운영의 핵심 — SEP 도중 인위적으로 높은 최소값으로 락되는 것을 막는 안전장치 덕분에 작은 사건도 감지.

4. **The 10/1/1000 thresholds are codified here** — >10 MeV protons at 10 pfu (PCA onset), >100 MeV protons at 1 pfu (SEU/dose), >2 MeV electrons at 10³ pfu (deep dielectric charging) are the trio that still defines today's SWPC alerts and S/G-scale anchors.
   10/1/1000 임계값의 제도화 — >10 MeV 양성자 10 pfu(PCA), >100 MeV 양성자 1 pfu(SEU/선량), >2 MeV 전자 1000 pfu(심부 유전체 충전)이라는 세 값이 오늘날 SWPC alerts와 S/G 스케일의 기준점.

5. **Solar-wind speed governs MeV electrons at GEO** — the Figure 5 correlation reaffirms Paulikas & Blake (1979) and connects directly to spacecraft anomaly statistics (Anik E1, March 1996), giving operators a 2-day lead time from a high-speed-stream onset.
   GEO MeV 전자는 태양풍 속도가 지배 — Figure 5는 Paulikas & Blake(1979)를 재확인하고 1996년 3월 Anik E1 등 다수 위성 이상과 직접 연결되어, 고속 흐름 도래 후 약 2일의 사전 통보 시간을 운영자에게 제공.

6. **Body-fixed GOES creates a SEP-onset anisotropy bias** — since GOES-8 no longer spins, beam-aligned vs. off-beam pointing systematically over- or underestimates omnidirectional flux at the most operationally important moment (the first 30–60 minutes), motivating later multi-look-direction sensor designs (e.g., GOES-R SGPS).
   3축 안정 GOES의 SEP 시작 비등방성 편향 — 회전이 없어 빔 방향과 시야 정합 여부에 따라 등방 플럭스를 과대/과소 추정하며, 이는 가장 중요한 초기 30–60분에 발생. GOES-R SGPS의 다방향 시야 설계 동기.

7. **The data are evidence in spacecraft anomaly forensics** — SEC/USAF use GOES archives to attribute (or exonerate) the natural environment when operators report anomalies, embedding the EPS/HEPAD into satellite insurance, design, and reliability ecosystems.
   위성 이상 사후 포렌식의 증거 — SEC/USAF가 GOES 아카이브로 자연환경 기여 여부를 판정하며, 이는 위성 보험·설계·신뢰성 생태계에 EPS/HEPAD를 깊이 편입시킴.

8. **HEPAD bridges to galactic cosmic-ray science** — continuous >200 MeV proton monitoring correlates with ground neutron monitors and feeds the effective-heliospheric-potential framework for airline-crew and astronaut dose, demonstrating space-weather operational data are also primary scientific data.
   HEPAD는 GCR 과학과의 연결 다리 — >200 MeV 양성자 지속 감시는 지상 중성자 모니터와 상관되며, 유효 태양권 전위 모델로 항공 승무원·우주인 선량을 산출. 운영 데이터가 1차 과학 데이터의 역할도 동시에 수행.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Differential flux from count rate / 카운트율로부터의 미분 플럭스

For a channel with channel response factor $R = G \cdot \Delta E$ (cm² sr MeV) and count rate $C$ (cps):
$$
J(E) \;=\; \frac{C}{R} \;=\; \frac{C}{G \, \Delta E} \quad \left[\text{particles}/(\text{cm}^2\,\text{s sr MeV})\right].
$$
For an integral channel (e.g., HEPAD P11) with single geometric factor $G$ (cm² sr):
$$
J_\text{int}(>E) \;=\; \frac{C}{G} \quad \left[\text{particles}/(\text{cm}^2\,\text{s sr})\right].
$$

미분 채널은 응답 인자 $R = G\Delta E$로 카운트율을 나눠 미분 플럭스를, 적분 채널은 단일 기하 인자 $G$로 적분 플럭스를 얻습니다.

### 4.2 Spectrum-dependent effective geometric factor (electrons) / 전자 채널의 스펙트럼 종속 유효 G

For wide-band electron channels, with a power-law local spectrum $J(E) \propto E^{-\gamma}$:
$$
\hat\gamma \;=\; -\frac{\ln(J_a / J_b)}{\ln(E_a / E_b)}, \qquad G_\text{eff} \;=\; G_\text{eff}(\hat\gamma)
$$
where $\hat\gamma$ is the slope estimated from neighboring channel pair $(a,b)$ and $G_\text{eff}(\hat\gamma)$ is read from the calibration table (Panametrics FH-CAL-0053195 and laboratory data).

광대역 전자 채널에서 인접 채널 쌍으로 멱함수 기울기 $\hat\gamma$를 추정하고, 실험실 보정으로부터 $G_\text{eff}(\hat\gamma)$를 룩업.

E2 (>2 MeV dome) is treated as a **special case** with energy-independent response: $G_{E2} = 0.05$ cm² sr (fixed). / E2는 응답이 에너지 독립이므로 G=0.05 cm² sr 고정.

### 4.3 GCR-background low-pass filter / 우주선 배경 저역 통과 필터

Define the rolling 10-day minimum (per channel):
$$
B(t) \;=\; \min_{\tau \in [t-10\text{d},\,t]} C(\tau),
$$
with the safeguard:
$$
B(t) \;=\;
\begin{cases}
B_\text{candidate}(t) & \text{if } |B_\text{candidate}(t) - B(t-1)| \le \Delta_\text{tol} \\[4pt]
B(t-1) & \text{otherwise}.
\end{cases}
$$
Background-corrected count rate:
$$
C_\text{corr}(t) \;=\; C(t) - B(t).
$$

10일 최소값을 배경으로 차감하되, SEP 등으로 인위적으로 상승한 최소값에 락되지 않도록 직전 값과의 차이가 허용 범위 이내일 때만 갱신.

### 4.4 Secondary-response correction (top-down power law) / 부수 응답 보정

For each channel $i$ (in order P7, P6, …, P1), with secondary energy bands $\{[E_k^{lo}, E_k^{hi}]\}$ and secondary geometric factors $\{G_k^{(i)}\}$:
$$
C_i^\text{primary} \;=\; C_i^\text{total} \;-\; \sum_k G_k^{(i)} \int_{E_k^{lo}}^{E_k^{hi}} J^\text{secondary}(E)\,dE.
$$
The secondary flux $J^\text{secondary}(E)$ inside band $k$ is estimated from the **already-corrected** flux of the higher-energy neighbor channel $j$ ($E_j > E_k$) under a local power-law spectrum:
$$
J^\text{secondary}(E) \;=\; J_j \,\left(\frac{E}{E_j}\right)^{-\gamma_{ij}}, \qquad \gamma_{ij} \;=\; -\frac{\ln(J_j / J_{j'})}{\ln(E_j / E_{j'})}.
$$

각 5분 샘플마다 P7부터 시작해 인접 상위 채널의 보정 플럭스로부터 멱함수 외삽으로 부수 카운트를 추정·차감.

### 4.5 Integral flux derivation / 적분 플럭스 산출

From a set of differential channels $\{(E_i, J_i)\}$ with assumed piecewise power-law:
$$
J(>E_0) \;=\; \int_{E_0}^{\infty} J(E)\,dE \;\approx\; \sum_i \int_{E_i^{lo}}^{E_i^{hi}} J_i \left(\frac{E}{E_i}\right)^{-\gamma_i} dE.
$$

미분 채널들로부터 구간별 멱함수 가정으로 적분 플럭스(>1, >5, >10, >30, >50, >60, >100 MeV) 산출.

### 4.6 Operational alert/warning logic / 운영 알림·경보 논리

Boolean alert:
$$
\text{Alert}_\text{proton}(t) \;=\; \mathbb{1}\!\left[\,J_{>10\,\text{MeV}}(t) > 10\ \text{pfu}\,\right] \;\lor\; \mathbb{1}\!\left[\,J_{>100\,\text{MeV}}(t) > 1\ \text{pfu}\,\right],
$$
$$
\text{Alert}_\text{electron}(t) \;=\; \mathbb{1}\!\left[\,J_{>2\,\text{MeV}}(t) > 10^{3}\ \text{pfu}\,\right].
$$
The S-scale (introduced post-1996) extends this with logarithmic bins: $J_{>10\,\text{MeV}} \ge 10^k$ pfu maps to $S_k$ for $k=1,\dots,5$.

Boolean 형태의 alert 논리. 1996년 이후 도입된 S-스케일은 $J_{>10\text{ MeV}} \ge 10^k$ pfu를 $S_k$($k=1\!\!\sim\!\!5$)로 정의해 본 임계값을 로그 빈으로 확장.

### 4.7 Worked numerical example / 수치 예제

Suppose the EPS dome P6 channel (84–200 MeV, $R=129$ cm² sr MeV) reports raw count rate $C = 64.5$ cps, while the rolling 10-day minimum is $B = 1.5$ cps (cosmic-ray background). Then:
$$
C_\text{corr} = 64.5 - 1.5 = 63.0 \ \text{cps}, \qquad J_{P6} = \frac{63.0}{129} \approx 0.488 \ \text{p/(cm}^2 \text{s sr MeV)}.
$$
If the integral >100 MeV flux derived from P6 + P7 piecewise power-law gives $J_{>100} = 1.4$ pfu, the >100 MeV alert (>1 pfu) **fires**, even though the >10 MeV channel may not have crossed 10 pfu yet — illustrating why two thresholds (10 MeV @ 10 pfu and 100 MeV @ 1 pfu) are needed: hard-spectrum events trigger the 100 MeV alert first.

P6 채널 카운트 64.5 cps에서 GCR 배경 1.5 cps를 빼면 63 cps; 응답 인자 129로 나누면 0.488 p/(cm² s sr MeV). 두 채널의 멱함수 외삽으로 >100 MeV 적분이 1.4 pfu가 나오면 >100 MeV alert가 먼저 발효. 이는 단단한(hard) 스펙트럼 사건에 대해 두 임계값이 필요한 이유를 보여줌.

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1962 ──── 1972 ──── 1979 ──── 1986 ──── 1988 ──── 1990 ──── 1993 ──── 1994 ──── 1996 ──── 2003 ──── 2016
 │         │         │         │         │         │         │         │         │         │         │
Van     SOLRAD    Paulikas    Stevens    Shea      Baker     Sauer    GOES-8   ONSAGER   NOAA      GOES-R
Allen   first     & Blake     surface    SEP       LPF       HEPAD    launch   ET AL.   S-G-R    SGPS/EHIS
belts   GOES-     SW-driven   charging   profiles  forecast  >685     (3-axis) SPIE      scales    multi-look
        SEM       MeV elec.   AFGL TR    review    of MeV-e  MeV       fixed   2812      formal    multi-energy
                  Geophys.    1986       JPL 88-28 GRL 1990  ICRC '93  EPS              alert      design
                  Monograph                                                              standard
```

본 논문의 위치 / Where this paper sits: 1979 Paulikas–Blake correlation과 1990 Baker LPF 예측 모델, 1988 Shea SEP 통계가 이미 확립된 직후, GOES-8의 새 3축 안정 EPS/HEPAD 데이터를 운영적으로 처음 종합 정리한 NOAA SEC의 "운영 매뉴얼". 이후 2003년 NOAA Space Weather Scales(S/G/R)의 임계값과 2016년 GOES-R SGPS 다방향 설계의 직접적 계기.

This 1996 paper sits at the moment when the physics inputs (Paulikas–Blake 1979, Shea 1988, Baker 1990, Sauer 1993) are mature, and codifies them into the operational doctrine that NOAA's 2003 Space Weather Scales (S/G/R) and the 2016+ GOES-R SGPS multi-look-direction design directly inherit.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Stevens et al. (1986) AFGL TR-85-0043 | Surface charging by >10 keV electrons / >10 keV 전자 표면 충전 | Justifies low-energy electron monitoring rationale / 저에너지 전자 감시 근거 |
| Stassinopoulous & Brucker (1996) AIAA 96-0739 | Radiation-induced satellite anomalies / 방사선 유발 위성 이상 | Direct context for >2 MeV electron alert 1000 pfu threshold / >2 MeV 전자 1000 pfu 알림 직접 근거 |
| Callis et al. (1991) JGR 96, 2939 | Relativistic electron precipitation → stratospheric NOx / 상대론적 전자 침전과 성층권 NOx | Atmospheric-chemistry pathway for trapped-electron data / 포획 전자 데이터의 대기화학 경로 |
| Paulikas & Blake (1979) Geophys. Monogr. 21, 180 | Solar-wind-driven MeV electrons at GEO / GEO MeV 전자의 태양풍 의존성 | The physics correlation Onsager visualizes in Figure 5 / Figure 5에서 시각화하는 물리 상관 |
| Ogilvie et al. (1995) SSR 71, 55 | WIND SWE solar wind plasma instrument / WIND SWE 태양풍 플라즈마 기기 | Provides the SW-speed time series in Figure 5 / Figure 5의 태양풍 속도 데이터 출처 |
| Baker et al. (1996) NASA ISTP Newsletter 6 | Anik E1 spacecraft failure assessment / Anik E1 위성 운영 실패 평가 | Operational consequence motivating the alert system / 알림 시스템을 정당화하는 운영 결과 |
| Baker et al. (1990) JGR 95, 15133 | Linear prediction filter for relativistic e- at 6.6 R_E / 6.6 R_E 상대론적 전자 LPF 예측 | Statistical forecasting framework using GOES data / GOES 데이터 기반 통계 예측 골격 |
| Shea (1988) JPL Pub. 88-28 | SEP intensity/time profiles at 1 AU / 1 AU SEP 강도·시간 프로필 | Climatology that calibrates Onsager's threshold rationale / 임계값 근거를 보정하는 기후학 |
| Kahler (1993) JGR 98, 5607 | CME and long-rise SEP / CME과 장시간 상승 SEP | Acceleration-physics framework cited in §3.2 / §3.2가 인용하는 가속 물리 |
| Roelof (1979) Geophys. Monogr. 21, 220 | SEP propagation: corona to magnetotail / SEP 전파(코로나→자기꼬리) | Theoretical basis for IMF-aligned propagation + ExB drift / IMF 정렬 전파+ExB 드리프트의 이론 근거 |
| Cane, Reames, von Rosenvinge (1988) JGR 93, 9555 | IP shock role in SEP longitude distribution / 행성간 충격파의 SEP 경도 분포 역할 | Source-longitude vs. arrival profile mapping / 발원지 경도와 도착 프로필의 대응 |
| Sauer (1993) ICRC Vol. 3, 250 | HEPAD >685 MeV proton observations / HEPAD >685 MeV 양성자 관측 | Predecessor cosmic-ray validation paper by co-author / 동일 저자의 GCR 검증 선행 논문 |
| O'Brien et al. (1992) Radiat. Prot. Dosim. 45, 145 | Aircraft crew radiation exposure / 항공 승무원 방사선 노출 | HEPAD's effective-heliospheric-potential application / HEPAD 유효 태양권 전위 응용 |
| NOAA Space Weather Scales (2003) | S/G/R operational scales / S/G/R 운영 스케일 | Direct inheritor of 10/1/1000 thresholds documented here / 본 논문의 10/1/1000 임계값을 직접 계승 |
| GOES-R SGPS/EHIS (2016+) | Modern energetic-particle suite on GOES-16/17/18 / 최신 GOES-R EPS 후속 | Multi-look-direction design responding to anisotropy concern in §3.2 / §3.2의 비등방성 우려에 대응한 다방향 설계 |

---

## 7. References / 참고문헌

**Primary paper / 본 논문**:
- Onsager, T. G., R. Grubb, J. Kunches, L. Matheson, D. Speich, R. Zwickl, and H. Sauer, "Operational uses of the GOES energetic particle detectors," *Proc. SPIE* **2812** (GOES-8 and Beyond), 281–290, 1996. DOI: 10.1117/12.254075

**Cited in the paper / 논문 내 참고문헌**:
1. N. J. Stevens et al., "Environmental interactions technology status," AFGL-TR-85-0043, 1986.
2. E. G. Stassinopoulous and G. J. Brucker, "Radiation induced anomalies in satellites," AIAA 34th Aerospace Sci. Meeting 96-0739, January 1996.
3. L. B. Callis et al., "Precipitating relativistic electrons: Their long-term effect on stratospheric odd nitrogen levels," *J. Geophys. Res.* **96**, 2939, 1991.
4. G. A. Paulikas and J. B. Blake, "Effects of the solar wind on magnetospheric dynamics: Energetic electrons at the synchronous orbit," in *Quantitative Modeling of Magnetospheric Processes*, Geophys. Monograph **21**, 180, 1979.
5. K. W. Ogilvie et al., "SWE, A comprehensive plasma instrument for the WIND spacecraft," *Space Sci. Rev.* **71**, 55–77, 1995.
6. D. N. Baker et al., "An assessment of space environmental conditions during the recent Anik E1 spacecraft operational failure," *NASA ISTP Newsletter* **6**, June 1996.
7. D. N. Baker, R. L. McPherron, T. E. Cayton, and R. W. Klebesadel, "Linear prediction filter analysis of relativistic electron properties at 6.6 R_E," *J. Geophys. Res.* **95**, 15133, 1990.
8. M. A. Shea, "Intensity/time profiles of solar particle events at one astronomical unit," *Interplanetary Particle Environment*, JPL Publ. 88-28, 75, April 1988.
9. S. W. Kahler, "Coronal mass ejections and long rise time of solar energetic particle events," *J. Geophys. Res.* **98**, 5607, 1993.
10. E. C. Roelof, "Solar energetic particles: From the corona to the magnetotail," in *Quantitative Modeling of Magnetospheric Processes*, Geophys. Monograph **21**, 220, 1979.
11. H. V. Cane, D. V. Reames, and T. T. von Rosenvinge, "The role of interplanetary shocks in the longitude distribution of solar energetic particles," *J. Geophys. Res.* **93**, 9555, 1988.
12. H. H. Sauer, "GOES observations of energetic protons to E>685 MeV: Description and Data Comparison," 23rd Int. Cosmic Ray Conf., Vol. 3, 250–253, 1993.
13. K. O'Brien, W. Friedberg, F. E. Duke, L. Snyder, E. B. Darden Jr., and H. H. Sauer, "The exposure of aircraft crews to radiations of extraterrestrial origin," *Radiat. Prot. Dosim.* **45**, 145, 1992.

**Additional contextual references / 추가 맥락 참고**:
- NOAA Space Weather Prediction Center, "NOAA Space Weather Scales" (S/G/R), 2003. https://www.swpc.noaa.gov/noaa-scales-explanation
- Kress, B. T. et al., "An overview of the Space Environment In-Situ Suite (SEISS) on GOES-R," *Space Weather* **18**, 2020 — successor instrument suite.
- Rodriguez, J. V. et al., "Inter-calibration of GOES 8–15 solar proton measurements," *Space Weather* **12**, 92, 2014 — modern cross-calibration of channels documented here.
- Mertens, C. J. et al., "NAIRAS aircraft radiation model development, dose climatology, and initial validation," *Space Weather* **11**, 603, 2013 — modern descendant of the O'Brien (1992) aircraft-crew dose framework.
- Onsager, T. G. et al., "Operations and applications of the GOES energetic particle detectors," in *Solar Drivers of Interplanetary and Terrestrial Disturbances*, ASP Conf. Ser., 1996 — companion proceedings.

---

## Appendix A: GOES Channel Cheat Sheet / GOES 채널 요약

| Ch | Detector | Primary range / 1차 범위 | $R = G\Delta E$ (cm² sr MeV) | Notes / 비고 |
|---|---|---|---|---|
| P1 | Telescope | 0.7–4.2 MeV | 0.194 | low-E protons / 저에너지 양성자 |
| P2 | Telescope | 4.2–8.7 MeV | 0.252 | |
| P3 | Telescope | 8.7–14.5 MeV | 0.325 | |
| P4 | Dome 3 | 15–40 MeV | 5.21 | thin Al moderator / 얇은 Al |
| P5 | Dome 4 | 38–82 MeV | 14.5 | thick Al / 두꺼운 Al |
| P6 | Dome 5 | 84–200 MeV | 129 | thick Cu / 두꺼운 Cu |
| P7 | Dome 5 | 110–900 MeV | 839 | broad band / 광대역 |
| P8 | HEPAD | 330–420 MeV | 65.7 | Cherenkov |
| P9 | HEPAD | 420–510 MeV | 65.7 | Cherenkov |
| P10 | HEPAD | 510–700 MeV | 139 | Cherenkov |
| P11 | HEPAD | >700 MeV | $G=0.73$ cm² sr | integral / 적분 |
| E1 | Dome 3 | >0.6 MeV | spectrum-dependent | $G\!\sim\!0.075$ |
| E2 | Dome 3 | >2 MeV | $G=0.05$ fixed / 고정 | flat response / 평탄 응답 |
| E3 | Dome 4 | >4 MeV | spectrum-dependent | $G\!\sim\!0.017$ |

This summary is the operationally critical lookup table — engineers and forecasters use it daily to convert raw count rates into the integral fluxes that feed the alert thresholds.
이 요약표는 운영적으로 핵심적인 룩업 테이블 — 엔지니어와 예보관이 매일 사용해 원시 카운트율을 alert 임계값에 들어가는 적분 플럭스로 변환합니다.
