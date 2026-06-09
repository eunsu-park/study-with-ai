---
title: "Pre-Reading Briefing: The Solar Orbiter mission — Science overview"
paper_id: "36_muller_2020"
topic: Solar_Observation
date: 2026-04-17
type: briefing
---

# The Solar Orbiter mission — Science overview: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Müller, D., St. Cyr, O. C., Zouganelis, I., et al., "The Solar Orbiter mission — Science overview", *Astronomy & Astrophysics*, Vol. 642, A1 (2020). DOI: 10.1051/0004-6361/202038467
**Author(s)**: Daniel Müller (ESA Project Scientist), Chris St. Cyr (NASA Deputy Project Scientist), Yannis Zouganelis, and the full Solar Orbiter SWT + project team
**Year**: 2020

---

## 1. 핵심 기여 / Core Contribution

**한국어**
본 논문은 ESA-NASA 공동 임무 **Solar Orbiter(2020년 2월 10일 발사)** 의 **최상위 미션·과학 개요 문서**이며, A&A 642권 Solar Orbiter 특집호의 기준(앵커) 논문이다. (1) **네 가지 최상위 과학 질문**(태양풍과 코로나 자기장 기원, 태양 과도 현상과 헬리오스피어 변동성, 태양 분출과 고에너지 입자 방출, 태양 다이나모와 헬리오스피어 연결)을 정의하고, (2) 각 질문을 답하기 위한 **10기의 탑재체(원격 감지 6 + 인시투 4)** — EUI, SPICE, STIX, PHI, Metis, SoloHI + SWA, EPD, MAG, RPW — 의 역할을 명시하며, (3) **0.28 AU 근일점**과 금성 중력보조로 최대 **24°~33° 황도 경사각**에 도달하는 궤도 설계를 설명하고, (4) **Remote Sensing Windows(RSW)** 와 **Solar Orbiter Observing Plans(SOOPs)** 를 통한 원격-인시투 동시 운용 전략, (5) **Parker Solar Probe와의 합동 관측** 설계, (6) Cruise → Nominal Mission Phase(NMP) → Extended Mission Phase(EMP)로 이어지는 임무 단계별 과학 목표를 제시한다. 이 논문은 모든 개별 기기 논문(EUI Rochus 2020, SPICE Anderson 2020, PHI Solanki 2020 등)의 "상위 앵커"로 작동하며, "원격 관측은 인시투 관측의 태양 표면 기원을 맥락화해야 하고, 인시투 관측은 원격 관측이 본 플라즈마를 실제로 감지해야 한다"는 **"연결 과학(Connection Science)"** 패러다임을 공식화한다.

**English**
This paper is the **top-level mission and science overview** of the ESA-NASA Solar Orbiter mission (launched 10 February 2020) and serves as the reference anchor for the A&A 642 Solar Orbiter special issue. It (1) defines **four top-level science questions** (origin of solar wind and coronal magnetic field; transients driving heliospheric variability; solar eruptions producing energetic particle radiation; solar dynamo driving Sun–heliosphere connections), (2) assigns the payload — **10 instruments (6 remote-sensing + 4 in-situ)** — EUI, SPICE, STIX, PHI, Metis, SoloHI plus SWA, EPD, MAG, RPW — to those questions, (3) describes the **0.28 AU perihelion** orbit with Venus gravity assists reaching 24°–33° heliographic inclination, (4) explains the **Remote Sensing Windows (RSWs)** and **Solar Orbiter Observing Plans (SOOPs)** that coordinate remote and in-situ observations, (5) lays out **joint operations with Parker Solar Probe**, and (6) describes mission phases: Cruise → Nominal Mission Phase (NMP) → Extended Mission Phase (EMP). It functions as the authoritative parent document for all individual instrument papers (EUI Rochus 2020, SPICE Anderson 2020, PHI Solanki 2020, etc.) and formalizes the **"Connection Science"** paradigm — remote sensing must contextualize the surface origin of plasma measured in situ, and in-situ must sample plasma actually imaged remotely.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

**한국어**
Solar Orbiter는 20년 이상의 설계 기간을 가진 미션이다. ESA의 Cosmic Vision 계획에서 2000년대 초반에 개념이 제기되었고, 2011년 ESA M-class 미션으로 최종 선정, 2020년 발사에 이르렀다. 이 미션은 세 개의 과학적 공백을 동시에 메우려는 시도다 — (1) SOHO(1995, L1 궤도)·STEREO(2006, 황도면 쌍둥이)·SDO(2010, GEO)·Hinode(2006)·PSP(2018, 근일점 ~9.86 R☉)의 한계인 **고위도 관측 부재**, (2) 태양 원격 관측과 인시투 측정을 **같은 플라즈마 패킷에 대해** 수행할 수 없었던 거리적 제약, (3) 0.5 AU보다 가까운 거리에서의 고해상도 원격 관측 부재. 2000년대 말까지 "태양권 미션의 다음 세대"가 무엇인지에 대한 학계 합의는 분명했다: **"가까이, 황도 밖으로, 원격+인시투 동시 운용"** 이라는 세 조건. Solar Orbiter가 이를 구현했다.

**English**
Solar Orbiter was conceived over 20 years: proposed in ESA's Cosmic Vision in the early 2000s, selected as ESA's M-class mission in 2011, and launched in February 2020. It addresses three simultaneous gaps left by prior missions: (1) the **absence of high-latitude solar observations** (SOHO at L1, STEREO in ecliptic, SDO in GEO, Hinode LEO, PSP with polar-ecliptic orbit but no remote imaging); (2) the distance constraint that prevented remote sensing and in-situ measurements of **the same plasma parcel**; (3) the lack of high-resolution remote sensing from closer than ~0.5 AU. By the late 2000s, the consensus for the "next heliospheric mission" was clear: **close, out-of-ecliptic, simultaneous remote + in-situ**. Solar Orbiter delivers all three.

### 타임라인 / Timeline

```
1990s ──── Ulysses (polar orbit, in-situ only, no remote sensing)
1995 ──── SOHO launch (L1, remote+in-situ but 1 AU, ecliptic)
2006 ──── STEREO launch (twin ecliptic spacecraft, remote sensing)
2010 ──── SDO launch (GEO, continuous high-res remote)
2011 ──── ESA selects Solar Orbiter as M-class mission
2017–19 ── Solar Orbiter spacecraft + instrument integration
2018 Aug ── Parker Solar Probe launch (near-Sun in-situ, no remote imaging)
2020 Feb 10 ── Solar Orbiter launch (Atlas V 411, Cape Canaveral)
2020 Jun ── First Light phase — all 10 instruments activated
2020 ────── ★ THIS PAPER (Müller et al., A&A 642, A1)
2020 ────── A&A 642 Solar Orbiter special issue: ~20 instrument/science papers
2020–21 ── Cruise Phase (calibration, in-situ only outside RSWs)
2021 Nov 27 ─ Nominal Mission Phase (NMP) begins (Venus GAM #2)
2022 Mar 26 ─ First perihelion at 0.32 AU
2025–── Extended Mission Phase (EMP) — inclination >30°, polar views
Future ──── Solar-C/EUVST (JAXA), MUSE (NASA) — build on Solar Orbiter heritage
```

---

## 3. 필요한 배경 지식 / Prerequisites

**한국어**
- **논문 #8 Domingo et al. 1995 (SOHO 미션 개요)**: 원격 + 인시투 결합 개념의 원조; 12개 기기 탑재, L1 궤도, 휼리오피직스의 20년 기준.
- **논문 #35 Pesnell et al. 2012 (SDO 미션 개요)**: 대조 케이스 — 지구동기 궤도의 연속 고대역 다운링크 vs. Solar Orbiter의 심우주 제약.
- **중력 보조(Gravity Assist) 궤도**: Solar Orbiter는 **금성 7회 + 지구 1회** 의 중력 보조로 궤도를 점진적으로 변형 — 근일점을 조이면서 황도 경사각을 올린다. 에너지 보존·각운동량 이전 원리 숙지 필요.
- **태양풍의 고속·저속 성분**: 고속 태양풍(≳700 km/s, 극지역 coronal hole 기원)과 저속 태양풍(~400 km/s, 적도 활동 영역/streamer belt 기원)의 원천 구분이 미션의 핵심 질문 1.
- **태양 분출의 종류**: CME(Coronal Mass Ejection), SEP(Solar Energetic Particles), flare, filament eruption의 차이.
- **태양 자기 사이클**: 11년 태양 흑점 사이클과 22년 Hale 자기 사이클. 극지역 자기장 반전이 미션 질문 4의 핵심.
- **Parker Solar Probe 미션 (Fox et al. 2016)**: NASA 파트너 미션. Solar Orbiter와의 합동 관측 기획이 본 논문의 주요 부분.
- **RSW / SOOP 개념**: Remote Sensing Window — 관측 자원을 집중하는 궤도상의 기간; SOOP — 여러 기기를 공통 과학 목표로 엮는 관측 계획.

**English**
- **Paper #8 Domingo et al. 1995 (SOHO overview)**: The ancestor of combined remote + in-situ payloads; 12 instruments at L1; the 20-year baseline of heliophysics.
- **Paper #35 Pesnell et al. 2012 (SDO overview)**: The contrast case — GEO orbit with continuous 150 Mbps downlink vs. Solar Orbiter's deep-space constraints.
- **Gravity-assist orbits**: Solar Orbiter uses **seven Venus flybys + one Earth flyby** to progressively tighten perihelion and raise heliographic inclination. Requires understanding of energy-conservation and angular-momentum transfer.
- **Solar wind components**: fast (≳700 km/s, polar coronal holes) vs. slow (~400 km/s, equatorial active regions/streamer belt) — origin distinction is Mission Question 1.
- **Types of solar transients**: Coronal Mass Ejections (CMEs), Solar Energetic Particles (SEPs), flares, filament eruptions.
- **Solar magnetic cycle**: 11-year sunspot + 22-year Hale cycles; polar field reversal underpins Mission Question 4.
- **Parker Solar Probe (Fox et al. 2016)**: NASA partner mission; joint observation planning is central to this paper.
- **RSW / SOOP concepts**: Remote Sensing Window (resource-concentrated orbital periods) and Solar Orbiter Observing Plan (multi-instrument coordinated observations under a shared science goal).

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **Perihelion / 근일점** | Solar Orbiter's closest approach to the Sun: 0.28 AU (≈60 R☉) at end of mission; 0.32 AU during NMP. / Solar Orbiter의 태양 최접근: 임무 말 0.28 AU(≈60 R☉), NMP 동안 0.32 AU. |
| **Heliographic inclination / 태양 경사각** | Angle between orbit plane and solar equator. Reaches 24° in NMP, up to 33° in EMP. / 궤도면과 태양 적도의 각도; NMP에서 24°, EMP에서 최대 33°. |
| **Venus GAM** | Venus Gravity Assist Manoeuvre — 7 flybys used to shape the orbit. / 7회 수행된 금성 중력 보조. |
| **Remote Sensing instruments** | EUI (EUV imager), SPICE (EUV spectrograph), STIX (X-ray), PHI (magnetograph/He), Metis (coronagraph), SoloHI (heliospheric imager). / 6개 원격 기기. |
| **In-situ instruments** | SWA (solar wind analyser), EPD (energetic particle), MAG (magnetometer), RPW (radio + plasma waves). / 4개 인시투 기기. |
| **Connection Science / 연결 과학** | Remote + in-situ measurements of the same plasma parcel to connect coronal structures to their heliospheric counterparts. / 같은 플라즈마에 대한 원격+인시투 동시 관측으로 코로나-헬리오스피어 구조를 연결. |
| **RSW** | Remote Sensing Window — three 10-day periods per orbit (perihelion + max N/S latitudes) when remote instruments run full cadence. / 궤도당 3회의 10일 원격 관측 집중 기간(근일점 + 최대 남·북 위도). |
| **SOOP** | Solar Orbiter Observing Plan — multi-instrument coordinated observation program under a shared science goal. / 공통 과학 목표를 위한 다기기 협력 관측 계획. |
| **NMP / EMP** | Nominal / Extended Mission Phase (2021–2026 / 2026–2030). / 정규 임무(2021–2026) / 연장 임무(2026–2030). |
| **Quadrature / 사분체** | Orbital configuration where Solar Orbiter and Earth are 90° apart as seen from the Sun, enabling stereoscopic views. / 태양에서 봤을 때 Solar Orbiter와 지구가 90° 떨어진 배치 — 입체 관측 가능. |
| **Co-rotation / 공회전** | Near perihelion, spacecraft angular velocity approximates solar rotation — it "flies with" a rotating feature for days. / 근일점에서 우주선 각속도가 태양 자전과 비슷해져 특정 구조를 며칠 동안 추적. |
| **SEP event** | Solar Energetic Particle event — sudden flux enhancement from flares/CME shocks, observed by EPD/RPW. / 플레어·CME 충격파에서의 순간적 플럭스 상승, EPD/RPW로 관측. |

---

## 5. 수식 미리보기 / Equations Preview

### (1) 중력 보조 속도 변화 / Gravity assist ΔV

$$
\Delta \vec v = 2 \vec v_{\text{planet}} \cdot (1 - \cos\alpha)
$$

**한국어** $\alpha$는 행성 기준 접근/이탈 속도 벡터가 회전한 각도. Solar Orbiter의 금성 플라이바이는 순방향/역방향 회전을 조합해 **근일점을 줄이면서 황도 경사각을 올린다** — 두 궤도 파라미터를 하나의 플라이바이로 동시 조정.

**English** $\alpha$ is the angle through which the spacecraft's velocity vector rotates in the planet's frame. Solar Orbiter's Venus flybys combine prograde/retrograde rotations to **simultaneously tighten perihelion and raise inclination** — two orbital parameters adjusted per flyby.

---

### (2) 태양풍 속도 — Parker 태양풍 방정식 / Parker solar wind

$$
u \frac{du}{dr} = -\frac{1}{\rho}\frac{dp}{dr} - \frac{GM_\odot}{r^2}
$$

**한국어** Parker(1958)의 등온 구면대칭 태양풍 풀이가 만드는 **임계 속도 조건**. Solar Orbiter의 인시투 관측은 이 이론이 예측하는 태양풍 가속 영역을 0.28 AU에서 처음으로 직접 샘플링한다.

**English** Parker's (1958) isothermal spherically symmetric solar wind equation defining the **critical-velocity condition**. Solar Orbiter's in-situ sampling at 0.28 AU probes the theoretical acceleration region for the first time at this distance.

---

### (3) Parker 나선 자기장 / Parker spiral magnetic field

$$
\vec B(r, \phi) = B_r(r_0) \left(\frac{r_0}{r}\right)^2 \hat r - B_r(r_0) \left(\frac{r_0}{r}\right) \frac{\Omega_\odot}{u} \sin\theta \, \hat\phi
$$

**한국어** 회전하는 태양에서 방출된 자기력선이 태양풍과 함께 나선 구조를 형성. $\Omega_\odot$ 태양 자전, $u$ 태양풍 속도, $r_0$ 기준 거리. **코로나 표면의 어느 위치가 우주선의 인시투 측정과 연결되는지**를 결정하는 핵심 식 — "Connection Science"의 기하학적 기반.

**English** Magnetic field lines ejected from the rotating Sun form a spiral embedded in the radially streaming wind. $\Omega_\odot$ = solar rotation, $u$ = wind speed, $r_0$ = reference. Determines **which coronal surface location maps to an in-situ measurement** — the geometric backbone of Connection Science.

---

### (4) 임무 통신 지연 / Communication one-way light time

$$
\tau_{\text{owlt}} = \frac{d_{\oplus \text{-} \text{SC}}}{c}
$$

**한국어** Solar Orbiter가 지구에서 $d$만큼 떨어져 있을 때, 명령/텔레메트리의 편도 지연 $\tau = d/c$. 1 AU에서 ~500 s, 0.28 AU 근일점 근처에서 Sun-Solar Orbiter-Earth 삼각형에 따라 최대 ~1500 s. **실시간 제어 불가능**을 의미 — SOOP 기반 자율 관측이 필수.

**English** At distance $d$ from Earth, the one-way command/telemetry delay is $d/c$ — ~500 s at 1 AU, up to ~1500 s near perihelion depending on the Sun-SC-Earth geometry. **Precludes real-time control** → mandates SOOP-based autonomous observations.

---

### (5) RSW 과학 데이터 예산 / RSW science data budget

$$
D_{\text{RSW}} \;=\; \underbrace{R_{\text{downlink}}}_{\sim 5\text{--}100\,\text{kbps}} \times \underbrace{t_{\text{RSW}}}_{3 \times 10\,\text{days}} \times \eta_{\text{contact}}
$$

**한국어** 궤도당 데이터 예산의 부등식. 전형적 SO 다운링크율은 근일점 근처에서도 수십 kbps~수 Mbps에 그치므로 원격 데이터는 고도로 압축·우선순위화되어야 한다. 기기당 일일 ~20 Gb 생산 vs. 다운링크 ~수백 Mb/일의 격차가 온보드 이벤트 탐지/우선순위 결정 논리를 정당화한다.

**English** Per-orbit data budget. Solar Orbiter's downlink rate is only tens of kbps to a few Mbps even near perihelion, forcing aggressive compression and prioritization. The ~20 Gb/day per-instrument generation vs. few-hundred Mb/day downlink gap motivates onboard event detection and priority algorithms.

---

## 6. 읽기 가이드 / Reading Guide

**한국어**
이 논문은 **미션 개요**이므로 개별 기기 논문보다 구조가 유연하다. 다음 순서를 권장한다:

1. **§1 Introduction + §2 Scientific objectives**: 네 가지 과학 질문을 확실히 외울 것. 이후 모든 기기/궤도/SOOP 논의의 출발점.
2. **§3 Mission design and operations**: 궤도 그림을 **반드시 시각적으로 익힐 것**. Venus GAM 시퀀스, 근일점 변화, 경사각 진화. 여기서 "왜 10년이 걸리는가?"가 답해진다.
3. **§4 Spacecraft and payload**: 10개 기기의 표와 관측 능력을 훑는다. 이미 읽은 #18 SPICE, #19 EUI는 확인차, 나머지는 개요 수준으로 흡수.
4. **§5 Observation strategy (SOOPs, RSWs)**: 어떻게 다기기 관측이 조율되는가. **이것이 논문의 차별점** — 기기 논문에는 없는 내용.
5. **§6 Connection with Parker Solar Probe**: PSP와의 합동 관측 설계. 쌍방이 어떻게 보완하는지.
6. **§7 Mission phases**: Cruise → NMP → EMP 단계별 과학 목표. 장기 계획의 관점.
7. **§8 Expected science output + Conclusions**: 기대 성과. 이후 논문들(특히 Zouganelis 2020 연결 과학 종합 논문)과 연결 지점.

**핵심 질문을 스스로에게 던지며 읽기**: "왜 이 미션이 필요했나? SDO/SOHO로는 왜 안 됐나?" → (1) 근태양 고해상도, (2) 고위도 관측, (3) 같은 플라즈마의 원격+인시투, 세 답이 분명해지면 논문의 모든 선택을 이해할 수 있다.

**English**
As a **mission overview**, the structure is more flexible than instrument papers. Recommended reading order:

1. **§1–§2 Introduction + Scientific objectives**: Memorize the four top-level science questions — they anchor every subsequent discussion.
2. **§3 Mission design and operations**: **Visually memorize the orbit evolution diagram.** Venus GAM sequence, perihelion tightening, inclination evolution. Answers "why does this take 10 years?"
3. **§4 Spacecraft and payload**: Skim the 10-instrument table. Instruments you've read (#18 SPICE, #19 EUI) in detail; others at overview level.
4. **§5 Observation strategy (SOOPs, RSWs)**: How multi-instrument coordination works. **This is the paper's unique content** — not in instrument papers.
5. **§6 Parker Solar Probe collaboration**: Complementary joint operations.
6. **§7 Mission phases**: Cruise → NMP → EMP goals; long-term planning.
7. **§8 Expected science + Conclusions**: Deliverables and downstream papers (especially Zouganelis 2020 connection-science synthesis).

**Read with three questions in mind**: "Why this mission? Why not SDO/SOHO?" → (1) close-perihelion high-resolution, (2) high-latitude views, (3) remote + in-situ of the same plasma. Once these answers are clear, every design choice in the paper follows.

---

## 7. 현대적 의의 / Modern Significance

**한국어**
Solar Orbiter는 2020년대 heliophysics의 **방향타 미션**이다. 본 논문이 정의한 "Connection Science" 패러다임은 이후의 모든 제안된 차세대 heliophysics 미션(Solar-C/EUVST, MUSE, Vigil)의 과학 정당화 논리에 흡수되었다. 2026년 현재 EMP 진입 직전인 미션은 이미 (1) EUI의 **campfire 발견** — 나노플레어 가설 재점화, (2) PHI의 극지 자기장 고해상 매핑 — 다이나모 이론 제약, (3) Parker Solar Probe와의 **Switchback 공동 관측** — 태양풍 난류 기원 규명, 등 수많은 발견을 생성했다. 본 논문은 이 발견들의 "도대체 왜 이 미션이 이렇게 설계되었는가"에 대한 공식적 답변이며, 향후 10년의 발견이 모두 이 프레임워크 내에서 해석될 것이다.

**English**
Solar Orbiter is **the compass-setting mission of 2020s heliophysics**. The "Connection Science" paradigm defined in this paper now permeates the science justification of every proposed next-generation mission (Solar-C/EUVST, MUSE, Vigil). As of 2026, on the brink of EMP, the mission has already produced (1) EUI's **campfire discovery** reigniting the nanoflare debate, (2) PHI's high-resolution polar magnetic field maps constraining dynamo theory, (3) **switchback joint observations** with Parker Solar Probe pinning down solar-wind turbulence origins. This paper is the official answer to "why is this mission designed this way?" — and the framework through which all upcoming decade's discoveries will be interpreted.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
