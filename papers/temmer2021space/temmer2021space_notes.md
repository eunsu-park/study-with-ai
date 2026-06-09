---
title: "Space Weather: The Solar Perspective — An Update to Schwenn (2006)"
authors: ["Manuela Temmer"]
year: 2021
journal: "Living Reviews in Solar Physics"
doi: "10.1007/s41116-021-00030-3"
topic: Living_Reviews_in_Solar_Physics
tags: [space_weather, CME, flares, SEPs, solar_wind, forecasting, STEREO, SDO, PSP, Solar_Orbiter]
status: completed
date_started: 2026-04-23
date_completed: 2026-04-23
---

# 75. Space Weather: The Solar Perspective (2021 Update) / 우주 기상: 태양 관점 (2021 업데이트)

> **Note / 참고**: This paper is an **explicit update** to Paper #9 (Schwenn 2006, "Space Weather: The Solar Perspective", Living Reviews in Solar Physics). Where Paper #9 framed the field with pre-STEREO, single-viewpoint tools, Paper #75 integrates 15 years of multi-viewpoint (STEREO), high-resolution (SDO), and close-Sun (PSP, Solar Orbiter) discoveries.
> **참고**: 본 논문은 Paper #9 (Schwenn 2006)의 **명시적 업데이트**이다. Paper #9는 STEREO 이전의 단일 시점 도구로 분야를 틀 잡았고, Paper #75는 15년간의 다중 시점(STEREO), 고해상도(SDO), 태양 근접(PSP, Solar Orbiter) 관측 성과를 통합한다.

---

## 1. Core Contribution / 핵심 기여

**한국어:**
본 논문은 Schwenn(2006)의 Living Review를 업데이트한 종합 리뷰로서, 우주 기상을 태양 관점에서 구조화한다. 저자는 (i) 플레어, (ii) CME, (iii) SEP, (iv) SIR을 네 개의 주요 현상으로 정의하고, 이들이 모두 **자기 재결합**이라는 공통 물리 기반을 공유함을 강조한다. 2006년 이후 15년간 가장 중요한 변화는 관측 플랫폼의 다변화다: STEREO 쌍성 위성(2006–)은 3D CME 재구성과 ICME의 원격-in situ 통합 추적을 가능하게 했으며, SDO(2010–)는 초 단위 EUV/자기장 분해능을 제공했고, Parker Solar Probe(2018–)와 Solar Orbiter(2020–)는 0.05–0.3 AU까지 태양에 접근해 기원 영역을 직접 탐침한다. 이 결과 저자는 다음을 체계화한다: CME의 초기 진화(0.5 $R_\odot$ 상공에서의 가속 피크), stealth CME의 자기 재구성 기원, SEP 사건의 wide-spread longitudinal 분포(단일 충격파가 다중 시점에서 관측됨), 태양풍 preconditioning에 의한 CME 전파 변화, 그리고 예보 모델링(DBM, EUHFORIA, ENLIL, Flarecast, COMESEP, ML). 2017년 9월 사건(NOAA 12673)은 플레어-CME-SEP 사슬 전체를 추적한 현대 우주 기상 연구의 정수다.

**English:**
This paper is a comprehensive review updating Schwenn (2006) that structures Space Weather from the solar perspective. The author defines four key phenomena — (i) flares, (ii) CMEs, (iii) SEPs, and (iv) SIRs — and emphasizes that all share **magnetic reconnection** as their common physical foundation. The most significant changes in the 15 years since 2006 come from observational platform diversification: STEREO twin spacecraft (2006–) enabled 3D CME reconstructions and seamless remote-to-in-situ ICME tracking; SDO (2010–) provides sub-minute EUV/magnetic resolution; Parker Solar Probe (2018–) and Solar Orbiter (2020–) probe origin regions directly at 0.05–0.3 AU. The author systematically covers: CME early evolution (acceleration peak at ~0.5 $R_\odot$), stealth CMEs originating from magnetic reconfiguration in the upper corona, wide-spread longitudinal distribution of SEPs (single shock seen from multiple viewpoints), solar wind preconditioning altering CME propagation, and forecasting models (DBM, EUHFORIA, ENLIL, Flarecast, COMESEP, ML). The September 2017 events (NOAA 12673) serve as the pinnacle of modern Space Weather research, tracing the entire flare-CME-SEP chain.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction and Space Weather Context / 도입부와 우주 기상 맥락 (Sec. 1–2)

**한국어 (p. 2–8):**
서론에서 저자는 태양을 "활성 별(active star)"로 정의하며, 우주기상의 네 가지 현상을 열거한다: CME, 플레어, SEP, 태양풍 상호작용 영역. 역사적 시작은 1859년 Carrington 사건(전신망 교란, 저위도 오로라, SEP 이벤트는 1959, 1960, 1972 이벤트의 약 2배 크기로 추정됨; Cliver & Dietrich 2013). 태양 관측의 기원: 1940년대 스펙트럼 관측, 1960년대 메트릭 전파 관측과 지자기 폭풍 급발단(SSC) 연구, 1971년 Tousey의 최초 CME, 1974년 MacQueen의 Skylab CME 카탈로그. Fig. 2는 현재 운영 중인 주요 임무를 화성(MAVEN, 320 $R_\odot$)에서 수성(BepiColombo, 83 $R_\odot$)까지 배치하며, 이들 모두가 우주 기상 관측 네트워크를 형성함을 보여준다.

우주 기상 핵심 파라미터: 도착하는 교란을 예측하기 위해 필요한 것은 (i) 행성간 자기장의 남북 성분 $B_z$, (ii) 태양풍 속도 $v$, (iii) 밀도 $n$. 특히 **대류 전기장 $VB_s$** ($B_s = B_z < 0$)이 Dst 폭풍 지수와 높은 상관을 보인다(Baker 1981; Wu & Lepping 2002; Gopalswamy 2008).

**English (p. 2–8):**
In the introduction, the author frames the Sun as an "active star" and enumerates four Space Weather phenomena: CMEs, flares, SEPs, and solar wind stream interaction regions. The historical origin is the September 1, 1859 Carrington event (telegraph disruption, low-latitude aurorae, associated SEP event estimated ~twice the size of 1959, 1960, 1972 events; Cliver & Dietrich 2013). Solar observational heritage: 1940s spectral observations, 1960s metric radio and storm sudden commencement (SSC) studies, Tousey's first CME in 1971, MacQueen's Skylab CME catalog in 1974. Fig. 2 maps current operational missions from Mars (MAVEN, 320 $R_\odot$) to Mercury (BepiColombo, 83 $R_\odot$), all forming the Space Weather observational network.

Key Space Weather parameters: To forecast an incoming disturbance requires (i) north-south IMF component $B_z$, (ii) solar wind speed $v$, (iii) density $n$. The **convective electric field $VB_s$** ($B_s = B_z < 0$) shows high correlation with the Dst storm index (Baker 1981; Wu & Lepping 2002; Gopalswamy 2008).

### Part II: Magnetic Reconnection — Common Ground / 자기 재결합: 공통 기반 (Sec. 3)

**한국어 (p. 8–10):**
저자는 플레어, CME, SEP이 모두 **자기 재결합과 자유 자기 에너지 방출**이라는 공통 물리에 기반함을 강조한다. Fig. 4는 에너지적 플레어-CME-SEP 이벤트의 세 가지 시간 단계를 보여준다:
1. **초기 단계 (왼쪽 패널)**: 확률론적 가속이 자기장 선을 따라 고에너지 입자를 가속. 태양 방향으로는 SXR/EUV 플레어 방출, 반대 방향(interplanetary)으로는 SEP 생성.
2. **중간 단계 (중앙)**: CSHKP 모델에 따라 CME 본체(flux rope)와 후-분출 아케이드(post-eruptive arcade, PEA)가 형성됨.
3. **후기 단계 (오른쪽)**: CME가 Parker 나선 IMF를 압축하며 전파, bow shock이 diffusive shock acceleration(DSA)으로 SEP을 행성간 공간에서 가속.

현대의 핵심 한계: **코로나 자기장을 직접 측정할 수 있는 장비가 현재 없으며**, 따라서 PFSS, NLFFF 같은 외삽 모델에 의존한다.

**English (p. 8–10):**
The author emphasizes that flares, CMEs, and SEPs all rest on the common physical basis of **magnetic reconnection and free magnetic energy release**. Fig. 4 depicts three temporal stages of an energetic flare-CME-SEP event:
1. **Early stage (left panel)**: Stochastic acceleration accelerates high-energy particles along magnetic field lines. Sunward — SXR/EUV flare emission; anti-sunward (interplanetary) — SEP production.
2. **Middle stage (center)**: Per the CSHKP model, the CME body (flux rope) and post-eruptive arcade (PEA) form.
3. **Late stage (right)**: The CME compresses Parker-spiral IMF, with the bow shock accelerating SEPs in interplanetary space via Diffusive Shock Acceleration (DSA).

Key modern limitation: **There is currently no instrument capable of direct coronal magnetic field measurements**, so we rely on extrapolations (PFSS, NLFFF models).

### Part III: Solar Flares / 태양 플레어 (Sec. 4)

**한국어 (p. 10–14):**
**4.1 Active region eruptive capability (활성영역 분출 능력):** 활성영역의 자기 복잡도가 플레어/CME 생성 확률을 지배한다. Yashiro(2006)은 X-class 플레어의 CME 연관률을 >80%로 보고했다. 그러나 "confined flares" — CME 동반 없이 플레어만 발생 — 는 가려진(bipolar overlying) 자기장 구성 때문에 발생하며 SEP를 만들지 않는다(Gopalswamy 2009). 이들은 단파 통신/네비게이션 교란(solar flare effect, geomagnetic crochet)을 유발할 수 있지만, false Space Weather alert의 원인이기도 하다. Decay index(자기 플럭스의 고도 감쇠율)가 failed eruptions에서 full eruptions보다 낮다(Török & Kliem 2005).

**4.2 Eruptive flares: 일반 특성:** 플레어는 $10^{19}$–$10^{32}$ erg의 에너지를 몇 시간 규모로 방출. Fig. 5는 플레어-CME-SEP의 시간 관계를 보여준다: pre-flare thermal(SXR/EUV) → impulsive phase(HXR footpoints, Bremsstrahlung) → main phase → decay. **Neupert 효과**: HXR flux rise = d(SXR flux)/dt. 11월 18일 2003 이벤트(M3.9, SXR 3.6×10⁻⁵ W/m²)는 solar cycle 23의 가장 강한 지자기 폭풍 중 하나(Dst = -472 nT)를 유발.

플럭스 로프 오리엔테이션 추론: (i) sigmoidal 구조(S 또는 reverse-S), (ii) post-eruptive arcade skewness, (iii) H$\alpha$ filament barb 방향, (iv) hemispheric helicity rule. Fig. 7은 2012년 6월 14일 이벤트의 right-handed flux rope를 여러 proxy로 식별.

**English (p. 10–14):**
**4.1 Active region eruptive capability:** The magnetic complexity of active regions governs flare/CME production. Yashiro (2006) reported >80% CME association for X-class flares. However, "confined flares" — flares without CMEs — arise from bipolar overlying magnetic field configurations and do not produce SEPs (Gopalswamy 2009). They can cause shortwave communication/navigation disturbances (solar flare effect, geomagnetic crochet) but also drive false Space Weather alerts. The decay index (magnetic flux height gradient) is lower for failed eruptions than full eruptions (Török & Kliem 2005).

**4.2 Eruptive flares — general characteristics:** Flares release $10^{19}$–$10^{32}$ erg over hours. Fig. 5 shows the flare-CME-SEP temporal relation: pre-flare thermal (SXR/EUV) → impulsive phase (HXR footpoints, Bremsstrahlung) → main phase → decay. **Neupert effect**: HXR flux rise equals d(SXR flux)/dt. The November 18, 2003 event (M3.9, SXR 3.6×10⁻⁵ W/m²) caused one of cycle 23's strongest geomagnetic storms (Dst = -472 nT).

Flux rope orientation proxies: (i) sigmoidal structures (S or reverse-S), (ii) post-eruptive arcade skewness, (iii) H$\alpha$ filament barb orientation, (iv) hemispheric helicity rule. Fig. 7 identifies a right-handed flux rope in the June 14, 2012 event using multiple proxies.

### Part IV: Coronal Mass Ejections / 코로나질량방출 (Sec. 5)

**한국어 (p. 14–27):**
**5.1 CME 일반 특성:** CME는 광학적으로 얇은 구조로, Thomson 산란을 통해 백색광에서 관측됨. 주요 하위 구조: **(1) shock (yellow arrow)**, **(2) CME 본체 (green)**, **(3) 공동 cavity**, **(4) 확장하는 magnetic flux rope (red)**, **(5) 프로미넨스 물질 또는 CME flank 중첩으로 인한 밝기 증가 (orange)**. 통계적 특성 (Vourlidas 2010; Lamy 2019):
- 반경 속도: 평균 300–500 km/s, 최대 3000 km/s
- 가속도: 0.1–10 km/s²
- 각 너비: 30–65°
- 질량: $10^{14}$–$10^{16}$ g
- CME/태양풍 밀도비: 15 $R_\odot$에서 ~11, 30 $R_\odot$에서 ~6 (Temmer 2021)

Cycle 24는 cycle 23 대비 다르다: CME 빈도 ↑, 너비 ↑ (Lamy 2019; Dagnew 2020). 이유: 헬리오스피어 압력 감소 (McComas 2013; Gopalswamy 2014).

**5.2 CME 초기 진화:** SDO/AIA EUV + LASCO/STEREO-SECCHI의 결합으로 저 코로나에서부터 추적 가능. Fig. 10은 HXR 플레어 방출과 CME 가속의 싱크로나이즈드 거동을 보여준다: **CME 가속 피크는 0.5 $R_\odot$ 상공 (20–30분 지속)**, 이후 propagation phase로 전이. 이는 플레어-CME 피드백 관계(Temmer 2008, 2010; Vršnak 2008)로 설명됨. Dimming 영역은 flux rope footpoints에 위치하며 CME 질량 추정에 사용됨(Temmer 2017b; Dissauer 2018).

**5.2.1 Shock formation:** 충격파는 1.2–1.8 $R_\odot$에서 형성됨 (local Alfvén 속도 최소). Type II radio burst가 충격 형성 증거. 3.8 $R_\odot$에서 local Alfvén 속도 최대 (Mann 1999; Vršnak 2002). 충격 형성 높이는 SEP 스펙트럼 경도와 연관 — 낮은 높이 → 강한 충격 → 하드 SEP 스펙트럼 (Gopalswamy 2017a).

**5.2.2 Stealth CMEs:** 태양 표면 서명 없이 발생하는 CME (dimming 없음, 플레어 없음, 필라멘트 분출 없음). 상부 코로나의 저에너지 자기장 재구성에서 기원 추정. "Problem storms"와 missed Space Weather events의 원인 (D'Huys 2014; Nitta & Mulligan 2017).

**5.3 다중 시점 관측의 장점:** STEREO (A: 2006–, B: signal lost 2014-10) 쌍성 위성이 제공하는 ~45°/년의 분리각은 3D CME 재구성을 가능하게 함. 주요 도구: GCS (Graduated Cylindrical Shell; Thernisien 2006, 2009) forward model — croissant 형태의 flux rope 기하학. Fig. 15: 2012년 3월 7-11일 이벤트의 GCS 피팅. **Fit 관계식**: $V_{rad} = 0.88 V_{exp}$ (Dal Lago 2003; Schwenn 2005), 빠른 CME의 경우 $V_{rad} \approx V_{exp}$ (Michalek 2009).

**Sec. 6 — ICMEs:** Heliospheric Imagers (STEREO-HI, WISPR on PSP, SoloHI on SO)는 Sun-Earth 공간 전체를 커버. **Drag-based Model (DBM; Vršnak 2013)**: 태양풍 항력이 CME를 감속/가속시켜 도착 속도를 조정. Fig. 18은 well-defined ICME의 in-situ 서명을 보여준다: shock → sheath (compressed, turbulent) → magnetic obstacle (smooth rotating B, low β, low T, linearly decreasing v). 1 AU에서 magnetic ejecta 통과 시간 ~1일; interacting CMEs/flank hits는 ~3일까지 지속.

**English (p. 14–27):**
**5.1 CME general characteristics:** CMEs are optically thin structures observed in white light via Thomson scattering. Main substructures: **(1) shock (yellow arrow)**, **(2) CME body (green)**, **(3) cavity**, **(4) expanding magnetic flux rope (red)**, **(5) brightness increase from prominence material or overlapping flanks (orange)**. Statistical properties (Vourlidas 2010; Lamy 2019):
- Radial speed: mean 300–500 km/s, max 3000 km/s
- Acceleration: 0.1–10 km/s²
- Angular width: 30–65°
- Mass: $10^{14}$–$10^{16}$ g
- CME/solar wind density ratio: ~11 at 15 $R_\odot$, ~6 at 30 $R_\odot$ (Temmer 2021)

Cycle 24 differs from cycle 23: ↑ CME frequency, ↑ width (Lamy 2019; Dagnew 2020). Reason: reduced heliospheric pressure (McComas 2013; Gopalswamy 2014).

**5.2 CME early evolution:** Combined SDO/AIA EUV + LASCO/STEREO-SECCHI enables tracking from the low corona. Fig. 10 shows synchronized HXR flare emission and CME acceleration: **CME acceleration peaks at ~0.5 $R_\odot$ above the surface (lasting 20–30 min)**, then transitions to propagation phase. This is explained by the flare-CME feedback (Temmer 2008, 2010; Vršnak 2008). Dimming regions, located at flux rope footpoints, are used for CME mass estimation (Temmer 2017b; Dissauer 2018).

**5.2.1 Shock formation:** Shocks form at 1.2–1.8 $R_\odot$ (local Alfvén speed minimum). Type II radio bursts are evidence of shock formation. Local Alfvén speed maximum at ~3.8 $R_\odot$ (Mann 1999; Vršnak 2002). Shock formation height correlates with SEP spectral hardness — lower heights → stronger shock → harder SEP spectra (Gopalswamy 2017a).

**5.2.2 Stealth CMEs:** CMEs without surface signatures (no dimming, no flare, no filament eruption). Likely originate from low-energy magnetic reconfiguration in the upper corona. Cause of "problem storms" and missed Space Weather events (D'Huys 2014; Nitta & Mulligan 2017).

**5.3 Multi-viewpoint observations:** STEREO (A: 2006–, B: signal lost 2014-10) provides ~45°/year separation angle, enabling 3D CME reconstruction. Key tool: GCS (Graduated Cylindrical Shell; Thernisien 2006, 2009) forward model — croissant-shaped flux rope geometry. Fig. 15: GCS fitting of the March 7-11, 2012 event. **Fit relation**: $V_{rad} = 0.88 V_{exp}$ (Dal Lago 2003; Schwenn 2005); for fast CMEs, $V_{rad} \approx V_{exp}$ (Michalek 2009).

**Sec. 6 — ICMEs:** Heliospheric Imagers (STEREO-HI, WISPR on PSP, SoloHI on SO) cover full Sun-Earth space. **Drag-based Model (DBM; Vršnak 2013)**: solar wind drag decelerates/accelerates the CME to adjust its arrival speed. Fig. 18 shows in-situ signatures of a well-defined ICME: shock → sheath (compressed, turbulent) → magnetic obstacle (smooth rotating B, low β, low T, linearly decreasing v). Passage time at 1 AU ~1 day; interacting CMEs/flank hits extend to ~3 days.

### Part V: Solar Energetic Particles / 태양에너지입자 (Sec. 7)

**한국어 (p. 31–37):**
**7.1 SEP 일반 특성:** keV–GeV 에너지 범위, 지상 중성자 모니터에 도달하는 GeV 양성자는 **GLE (Ground Level Enhancement)**라 칭함. 두 개체군(Fig. 19): **Gradual events** (CME 충격파에 의한 DSA, 광범위 longitudinal 분포, 장시간 지속) vs **Impulsive events** (플레어 재결합 가속, 좁은 연결성, $^3$He-rich).

SEP 조건: (1) 재결합 가속, (2) 열린 자기장 선으로의 탈출 경로. Confined flares는 SEP를 만들지 않음 (Trottet 2015). 관찰자와의 자기 연결성이 핵심 (Reames 2009). Fig. 20: 충격파 노즈(apex)는 낮은 높이에서 연결되고, flank는 더 높은 곳에서 연결됨 — SEP 온셋 시간에 지연 발생.

**7.2 다중 시점 SEP 관측:** Fig. 22: 2014년 2월 25일 이벤트 — STEREO-A (152°), STEREO-B (160°), SOHO의 세 궤도에서 동일 SEP 이벤트 관측. Wide-spread angular 분포 메커니즘: (i) CME 측면 확장, (ii) 다중 CME 상호작용(CME-CME conglomerates), (iii) 복잡한 자기장의 field line draping. Cycle 24의 2개 GLE: May 17, 2012 및 September 10, 2017 (Gopalswamy 2013b). Fig. 23: GLE (May 17, 2012)는 더 낮은 shock 형성 높이(1.38 $R_\odot$)와 더 길게 구동되는 충격파 → 더 높은 에너지 입자 생성.

**English (p. 31–37):**
**7.1 SEP general characteristics:** keV–GeV energy range; GeV protons reaching ground-level neutron monitors are called **GLEs (Ground Level Enhancements)**. Two populations (Fig. 19): **Gradual events** (DSA by CME shock, wide longitudinal distribution, long duration) vs **Impulsive events** (flare reconnection acceleration, narrow connectivity, $^3$He-rich).

SEP conditions: (1) reconnection acceleration, (2) open field line escape path. Confined flares produce no SEPs (Trottet 2015). Magnetic connectivity to the observer is key (Reames 2009). Fig. 20: shock nose (apex) connects at low heights, flanks connect higher — delays in SEP onset times.

**7.2 Multi-viewpoint SEPs:** Fig. 22: the February 25, 2014 event — the same SEP event observed from three orbits: STEREO-A (152°), STEREO-B (160°), and SOHO. Wide-spread angular distribution mechanisms: (i) CME lateral expansion, (ii) multi-CME interaction (CME-CME conglomerates), (iii) field line draping in complex magnetic fields. Cycle 24 produced only 2 GLEs: May 17, 2012 and September 10, 2017 (Gopalswamy 2013b). Fig. 23: the GLE (May 17, 2012) had lower shock formation height (1.38 $R_\odot$) and longer-driven shock → higher-energy particles.

### Part VI: Energy Budget / 에너지 예산 (Sec. 8)

**한국어 (p. 37–38):**
Fig. 24의 에너지 분배 (Aschwanden 2017 통계):
- Free magnetic energy $E_{mag}$의 **~87%가 방출됨**
- 방출 에너지 중 **~10%가 CME 운동 에너지 ($E_{CME}$)**
- **~80%가 입자 가속**
- SEP는 Free mag E의 **~10%**
- SEP는 $E_{CME}$의 **~3% 소산**

CME 속도가 SEP 특성과 가장 강한 상관을 보이며, 이는 CME 구동 shock acceleration 가설과 일치 (Mewaldt 2006; Papaioannou 2016).

**English (p. 37–38):**
Energy partition from Fig. 24 (Aschwanden 2017 statistics):
- **~87% of free magnetic energy $E_{mag}$ is released**
- Of released energy: **~10% becomes CME kinetic ($E_{CME}$)**
- **~80% goes to particle acceleration**
- SEPs account for **~10% of free mag E**
- SEPs dissipate **~3% of $E_{CME}$**

CME speed shows strongest correlation with SEP characteristics, consistent with the CME-driven shock acceleration hypothesis (Mewaldt 2006; Papaioannou 2016).

### Part VII: Solar Wind Structure / 태양풍 구조 (Sec. 9)

**한국어 (p. 38–48):**
**9.1 일반 특성:** 태양풍은 코로나 온도에 의해 가속된 지속적 플라즈마 흐름. Fig. 25는 Ulysses 세 궤도에서 관측된 태양 사이클에 따른 latitude 분포. 빠른 태양풍(>450 km/s)은 코로나 홀에서, 느린 태양풍(<400 km/s)은 streamer belt에서 기원. Table 1은 1976-2000 OMNI 데이터 평균:

| Parameter | Fast wind | Slow wind | CME shock-sheath | CME ejecta |
|---|---|---|---|---|
| $v_p$ (km/s) | >450–500 | <400–450 | 450±110 | 410±110 |
| $n_p$ (cm$^{-3}$) | 6.6±5.1 | 10.8±7.1 | 14.3±10.6 | 10.1±8.0 |
| $B$ (nT) | 6.4±3.5 | 5.9±2.9 | 8.5±4.5 | 12.0±5.2 |
| $T_p$ ($\times 10^4$ K) | 13.1±11.8 | 4.4±4.4 | 12.9±17.6 | 4.5±6.6 |
| Dst (nT) | -28.7±25.9 | -10.7±18.2 | -21.5±33.0 | -52.1±45.8 |

**9.3 태양풍 구조가 CME/SEP 진화에 미치는 영향:** CME는 주변 태양풍에 적응하며 형태, 속도, 방향을 바꿈. Fast wind에서 전파하는 narrow massive CME는 가장 짧은 transit time (Vršnak 2010). 낮은 latitude 코로나 홀도 low-speed 태양풍 변동의 원천일 수 있음 (Bale 2019, PSP 결과).

**9.4 Preconditioning:** 단일 CME는 태양풍 흐름을 2–5일간 교란. CME-CME 또는 CME-CIR 상호작용은 강한 preconditioning을 생성. July 23, 2012 이벤트: 가장 빠른 CME 중 하나로 1 AU 거리를 21시간 이하에 주파 (Liu 2014). 선행 CME (July 19)가 drag 파라미터를 10배 낮춤 (Temmer & Nitta 2015). 만약 지구 방향이었다면, Dst = -600에서 -1100 nT로 예상됨 (Ngwira 2014; Baker 2013).

**English (p. 38–48):**
**9.1 General characteristics:** Solar wind is a continuous plasma flow accelerated by coronal temperature. Fig. 25 shows latitude distribution over the solar cycle from three Ulysses orbits. Fast wind (>450 km/s) originates in coronal holes; slow wind (<400 km/s) from the streamer belt. Table 1 averages OMNI data 1976-2000 (see Korean table above).

**9.3 Effect of solar wind structure on CME/SEP evolution:** CMEs adapt to the ambient solar wind, altering their shape, speed, and direction. Narrow massive CMEs in fast wind have the shortest transit times (Vršnak 2010). Low-latitude coronal holes may also source slow-wind fluctuations (Bale 2019, PSP result).

**9.4 Preconditioning:** A single CME can disturb solar wind flow for 2–5 days. CME-CME or CME-CIR interactions produce strong preconditioning. The July 23, 2012 event — one of the fastest CMEs ever — traveled 1 AU in <21 hours (Liu 2014). A preceding CME (July 19) lowered the drag parameter by one order of magnitude (Temmer & Nitta 2015). If Earth-directed, expected Dst = -600 to -1100 nT (Ngwira 2014; Baker 2013).

### Part VIII: September 2017 Events — The Synthesis Case / 2017년 9월 사건 — 종합 사례 (Sec. 10)

**한국어 (p. 48–53):**
활성영역 **NOAA 12673**가 cycle 24 최강 폭풍 생산. 자기장 αβγ 구조. 2017년 9월 4–10일:
- **5 X-class 플레어** + 39 M-class 플레어
- **X9.3 플레어 (Sept 6)**: cycle 24 최대, SXR 시작 11:53 UT
- **X8.2 플레어 (Sept 10)**: 세 번째로 큼
- **첫 halo CME (Sept 6)**: LASCO/C2 첫 관측 12:24 UT, 투영 속도 1570 km/s, 소스 위치 S08W33
- **두 번째 halo CME (Sept 10)**: LASCO 투영 속도 1490 km/s, X8.2 플레어와 함께
- **첫 SEP 이벤트**: Sept 6 12:15 UT ~ Sept 7 23:25 UT
- **두 번째 SEP 이벤트 (GLE)**: Sept 10 16:25 UT ~ Sept 11 11:40 UT (서쪽 limb 근처, 유리한 자기 연결성)
- **지자기 폭풍**: Dst min = -142 nT on September 7

Fig. 35: Sept 6 이벤트의 플레어 리본, dimming 영역, PEA를 보여줌. Reconnected flux를 PEA로부터 유도하여 magnetized CME 모델(EUHFORIA, ENLIL with flux rope)에 입력 (Scolini 2020). Fig. 36: 다중 shock과 magnetic ejecta가 WIND L1에서 관측됨. Sept 6-9 이벤트들의 "shock-in-a-cloud" (ICME2 shock이 ICME1 magnetic cloud로 전파) → geoeffectiveness 2배 강화 (Shen 2018). Fig. 37: ACE/EPAM, GOES/EPEAD, GOES/HEPAD proton intensities, Dst 지수, 중성자 모니터(Forbush decrease) 전시.

2017년 9월 사건은 Mars에서도 측정됨 — MAVEN 궤도선 및 Curiosity rover 지표면 관측 (Hassler 2018).

**English (p. 48–53):**
Active region **NOAA 12673** produced cycle 24's strongest storms. Magnetic αβγ configuration. September 4–10, 2017:
- **5 X-class flares** + 39 M-class flares
- **X9.3 flare (Sept 6)**: cycle 24 maximum, SXR onset 11:53 UT
- **X8.2 flare (Sept 10)**: third largest
- **First halo CME (Sept 6)**: LASCO/C2 first seen 12:24 UT, projected speed 1570 km/s, source S08W33
- **Second halo CME (Sept 10)**: LASCO projected 1490 km/s, with X8.2 flare
- **First SEP event**: Sept 6 12:15 UT – Sept 7 23:25 UT
- **Second SEP event (GLE)**: Sept 10 16:25 UT – Sept 11 11:40 UT (near west limb, favorable magnetic connectivity)
- **Geomagnetic storm**: Dst min = -142 nT on September 7

Fig. 35: Sept 6 event shows flare ribbons, dimming regions, PEA. Reconnected flux from PEA is used as input to magnetized CME models (EUHFORIA, ENLIL with flux rope; Scolini 2020). Fig. 36: multiple shocks and magnetic ejecta at WIND L1. The Sept 6-9 events' "shock-in-a-cloud" (ICME2 shock propagating into ICME1 magnetic cloud) → 2× geoeffectiveness (Shen 2018). Fig. 37: ACE/EPAM, GOES/EPEAD, GOES/HEPAD proton intensities, Dst, neutron monitor Forbush decrease.

September 2017 events were also measured at Mars — MAVEN orbiter and Curiosity rover surface observations (Hassler 2018).

### Part IX: Forecasting / 예보 모델링 (Sec. 11)

**한국어 (p. 53–55):**
**플레어 예보**: 활성영역 광자권 자기장으로부터 통계적 관계. Flarecast (EU), NASA/CCMC scoreboard. 주요 플레어는 예측 가능하나 single event 예측 불확실성 큼.

**CME 도착 시간**:
1. **경험적**: transit time-speed 관계 (Gopalswamy 2001)
2. **영상 기반**: STEREO/HI 또는 radio IPS를 사용해 50 $R_\odot$ 너머까지 추적 (Colaninno 2013; Rollett 2016)
3. **해석적**: **DBM (Drag-Based Model; Vršnak 2013)** — 가장 널리 사용됨
4. **수치 MHD**: EUHFORIA (Pomoell & Poedts 2018), ENLIL (Odstrčil & Pizzo 1999), CORHEL (Riley 2012), SUSANOO (Shiota & Kataoka 2016)
5. **자기화 CME 플럭스 로프**: 관측된 reconnected flux를 모델 입력으로 (Scolini 2019; Singh 2019)

**SEP 예보**: PROTONS, PPS, ESPERTA, FORSPEF, SOLPENCO, HESPERIA, SEPForecast (COMESEP).

**앙상블 및 ML**: Lee 2013; Mays 2015; Dumbović 2018; Amerstorfer 2018; Camporeale 2019.

**English (p. 53–55):**
**Flare forecasting**: Statistical relations from photospheric magnetic field. Flarecast (EU), NASA/CCMC scoreboard. Major flares predictable but single-event uncertainty large.

**CME arrival time**:
1. **Empirical**: transit time-speed relation (Gopalswamy 2001)
2. **Image-based**: Using STEREO/HI or radio IPS to track beyond 50 $R_\odot$ (Colaninno 2013; Rollett 2016)
3. **Analytical**: **DBM (Drag-Based Model; Vršnak 2013)** — most widely used
4. **Numerical MHD**: EUHFORIA (Pomoell & Poedts 2018), ENLIL (Odstrčil & Pizzo 1999), CORHEL (Riley 2012), SUSANOO (Shiota & Kataoka 2016)
5. **Magnetized CME flux rope**: observed reconnected flux as model input (Scolini 2019; Singh 2019)

**SEP forecasting**: PROTONS, PPS, ESPERTA, FORSPEF, SOLPENCO, HESPERIA, SEPForecast (COMESEP).

**Ensemble and ML**: Lee 2013; Mays 2015; Dumbović 2018; Amerstorfer 2018; Camporeale 2019.

---

## 3. Key Takeaways / 핵심 시사점

1. **Multi-viewpoint observations revolutionized CME kinematics / 다중 시점 관측은 CME 운동학을 혁신했다** — STEREO 쌍성 위성이 2006년 발사 후 3D CME 재구성을 가능하게 함. GCS(Graduated Cylindrical Shell) 모델이 표준 도구가 됨. 투영 효과 제거로 실제 속도 측정이 가능해졌고, $V_{rad} = 0.88 V_{exp}$(Dal Lago 2003) 같은 핵심 스케일링 관계 확립. Single-viewpoint LASCO 관측은 halo CME에서 방향/속도를 결정하기 어려움이 명확해졌다. / STEREO twin spacecraft launched 2006 enabled 3D CME reconstruction. GCS became the standard tool. Removing projection effects enabled true-speed measurements and established key scaling relations like $V_{rad} = 0.88 V_{exp}$ (Dal Lago 2003). Single-viewpoint LASCO has clear difficulty determining direction/speed for halo CMEs.

2. **CME early evolution is the key to forecasting / CME 초기 진화가 예보의 핵심이다** — CME 가속 피크는 **0.5 $R_\odot$** 상공에서 발생하며 HXR 플레어 방출과 동기화됨. 이는 CSHKP 모델의 플레어-CME 피드백과 일치. Shock 형성은 **1.2–1.8 $R_\odot$** (local Alfvén 속도 최소)에서 발생하고, Type II radio burst가 그 증거. 낮은 높이에서 형성된 강한 shock는 더 하드한 SEP 스펙트럼을 만든다. / CME acceleration peaks at **0.5 $R_\odot$** above the surface, synchronized with HXR flare emission, consistent with the CSHKP flare-CME feedback. Shocks form at **1.2–1.8 $R_\odot$** (local Alfvén minimum), evidenced by Type II radio bursts. Stronger shocks formed at lower heights produce harder SEP spectra.

3. **Stealth CMEs and the "problem storm" mystery / Stealth CME와 "문제 폭풍"의 미스터리** — 태양 표면 서명 없이 발생하는 CME가 존재하며, 이는 상부 코로나의 저에너지 자기장 재구성에서 기원. 이는 Space Weather 예보의 근본적 한계를 드러낸다 — 원인이 관측되지 않으면 조기 경보가 불가능. ESA Lagrange L5 임무(2027 예정)가 해결책을 제시할 수 있다. / CMEs without surface signatures exist, originating from low-energy magnetic reconfiguration in the upper corona. This exposes a fundamental Space Weather forecasting limit — if the cause is unobservable, early warning is impossible. ESA Lagrange L5 mission (planned 2027) may address this.

4. **Preconditioning of interplanetary space / 행성간 공간의 preconditioning** — 단일 CME가 행성간 공간의 배경 태양풍을 2–5일간 교란하며, 후속 CME의 drag 파라미터를 최대 10배 낮춘다. July 23, 2012 "super-CME"는 선행 CME에 의한 preconditioning 덕분에 21시간 이하에 1 AU를 주파. 만약 지구 방향이었다면 Dst = -600에서 -1100 nT의 파멸적 폭풍이 됐을 것. 이는 **CME-CME 상호작용이 single event 예보만큼 중요함**을 보여준다. / A single CME disturbs the background solar wind for 2–5 days and can reduce the drag parameter for subsequent CMEs by up to 10×. The July 23, 2012 "super-CME" traversed 1 AU in <21 hours thanks to preceding CME preconditioning. Earth-directed, it would have produced catastrophic Dst = -600 to -1100 nT. This shows **CME-CME interaction is as important as single-event forecasting**.

5. **Energy budget consolidation / 에너지 예산의 확립** — Aschwanden 2017의 통계 연구로 에너지 분배가 정량화됨: 자유 자기 에너지의 ~87%가 방출, 중 ~10%가 CME 운동 에너지, ~80%가 입자 가속, ~10%가 SEP. SEP는 CME 운동 에너지의 ~3%를 소산. CME 속도가 SEP 특성과 가장 강한 상관을 보이며, 이는 **CME-driven shock acceleration 가설의 통계적 확증**이다. / Aschwanden 2017 statistics quantified energy partition: ~87% of free magnetic energy released; of that, ~10% to CME kinetic, ~80% to particle acceleration, ~10% to SEPs. SEPs dissipate ~3% of CME kinetic energy. CME speed most strongly correlates with SEP characteristics — **statistical confirmation of CME-driven shock acceleration hypothesis**.

6. **DBM as the workhorse forecasting model / DBM은 예보의 표준 모델** — 항력 기반 모델(Drag-Based Model; Vršnak 2013)은 복잡한 MHD 시뮬레이션 없이도 CME 도착 시간을 합리적으로 예측. 지수형 해 $v(r) = v_{SW} + (v_0 - v_{SW}) \exp(-\gamma(r - r_0))$가 ESA/SSA 실시간 운영에 채택. EUHFORIA, ENLIL 같은 MHD 모델은 더 정확하지만 연산 비용이 크다. DBM의 계승자: magnetized CME flux rope를 입력으로 하는 EUHFORIA-with-flux-rope (Scolini 2019). / The Drag-Based Model (DBM; Vršnak 2013) provides reasonable CME arrival time predictions without costly MHD. The exponential solution $v(r) = v_{SW} + (v_0 - v_{SW}) \exp(-\gamma(r - r_0))$ is operationalized at ESA/SSA. MHD models (EUHFORIA, ENLIL) are more accurate but expensive. DBM's successor: EUHFORIA-with-flux-rope using magnetized CME as input (Scolini 2019).

7. **Machine learning in Space Weather / 우주 기상에서의 머신러닝** — Camporeale 2019 리뷰가 체계화한 ML 기법들이 플레어 예보(Flarecast), SEP 예보(COMESEP, FORSPEF), CME 도착 시간에 채택됨. 그러나 훈련 데이터의 희소성(GLE는 cycle당 2–3건)과 "black box" 해석가능성 문제가 여전히 존재. 앙상블 모델(ENLIL Ensemble, Dumbović 2018)이 불확실성 정량화를 제공하는 현실적 대안. / Camporeale 2019 review systematized ML in flare forecasting (Flarecast), SEP forecasting (COMESEP, FORSPEF), and CME arrival time. Challenges: training data scarcity (GLEs: 2–3 per cycle) and "black box" interpretability. Ensemble models (ENLIL Ensemble, Dumbović 2018) provide practical uncertainty quantification.

8. **September 2017: synthesis of modern Space Weather science / 2017년 9월: 현대 우주 기상 과학의 종합** — NOAA 12673에서 X9.3 (Sept 6), X8.2 (Sept 10) 플레어. 두 halo CME, 두 SEP 이벤트, GLE (Sept 10, cycle 24 마지막). Dst = -142 nT. "Shock-in-a-cloud" 상호작용으로 geoeffectiveness 2배 강화. Mars까지 추적됨(MAVEN, Curiosity). 이 사건은 **Sun-Earth-Mars 간 우주 기상 시스템 이해의 정점**을 보여준다 — 단일 활성영역에서 시작한 사건 사슬이 태양계를 가로지르며 추적되는 시대가 도래했다. / NOAA 12673 produced X9.3 (Sept 6) and X8.2 (Sept 10). Two halo CMEs, two SEP events, a GLE (Sept 10, cycle 24's last). Dst = -142 nT. "Shock-in-a-cloud" interaction doubled geoeffectiveness. Tracked to Mars (MAVEN, Curiosity). This event represents **the apex of Sun-Earth-Mars Space Weather understanding** — an era where event chains from a single active region are traced across the solar system.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Drag-Based Model (DBM) — CME Propagation / CME 전파
Equation of motion for CME interacting with solar wind via aerodynamic drag:
$$\frac{dv}{dt} = -\gamma (v - v_{SW}) \, |v - v_{SW}|$$

Analytical solution for radial distance r:
$$v(r) = v_{SW} + \frac{v_0 - v_{SW}}{1 \pm \gamma (v_0 - v_{SW})(r - r_0)}$$

or in exponential form (for simplified case):
$$v(r) \approx v_{SW} + (v_0 - v_{SW}) \, \exp[-\gamma (r - r_0)]$$

**Terms / 항목:**
- $v(r)$: CME speed at heliocentric distance $r$ / 태양 중심 거리 $r$에서의 CME 속도
- $v_{SW}$: background solar wind speed (asymptote) / 배경 태양풍 속도(점근선)
- $v_0$: initial CME speed at $r_0$ / 초기 CME 속도
- $\gamma$: drag parameter, units km$^{-1}$; typical values $10^{-8}$–$10^{-7}$ km$^{-1}$ / 항력 파라미터
- $\pm$: + if $v_0 > v_{SW}$ (decelerating), − if $v_0 < v_{SW}$ (accelerating) / 감속/가속 부호

**Numerical example / 수치 예**: $v_0 = 1500$ km/s, $v_{SW} = 400$ km/s, $\gamma = 2 \times 10^{-8}$ km$^{-1}$, $r_0 = 20 R_\odot \approx 1.4 \times 10^7$ km, $r = 1$ AU $= 1.496 \times 10^8$ km.
$v(1\text{AU}) = 400 + 1100/[1 + 2 \times 10^{-8} \times 1100 \times 1.35 \times 10^8] \approx 400 + 1100/3.98 \approx 676$ km/s.
Transit time $\approx (1.5 \times 10^8 \text{ km})/(\text{average } v \approx 900 \text{ km/s}) \approx 46$ hr.

### 4.2 Alfvén Mach Number and Shock Formation / 알프벤 수와 충격파 형성
$$M_A = \frac{v_{CME} - v_{SW}}{v_A}, \quad v_A = \frac{B}{\sqrt{\mu_0 \rho}} = \frac{B}{\sqrt{\mu_0 n_p m_p}}$$

- $M_A > 1$: supercritical shock, DSA active / 초임계 shock, DSA 활성화
- Shock formation at local Alfvén minimum (1.2–1.8 $R_\odot$) / local Alfvén 최소에서 형성
- Local maximum at 3.8 $R_\odot$ / local 최대

**Example**: at 2 $R_\odot$, $B \approx 1$ G = $10^{-4}$ T, $n_p \approx 10^7$ cm$^{-3}$, $\rho = n_p m_p \approx 1.67 \times 10^{-14}$ kg/m$^3$.
$v_A = 10^{-4} / \sqrt{4\pi \times 10^{-7} \times 1.67 \times 10^{-14}} \approx 690$ km/s.
For $v_{CME} = 1500$ km/s, $v_{SW} = 200$ km/s: $M_A = 1300/690 \approx 1.9$ → strong shock.

### 4.3 Dst Index Prediction (Burton-type) / Dst 지수 예측
The pressure-corrected Dst:
$$Dst^* = Dst - b\sqrt{P_{dyn}} + c$$

Time evolution (Burton 1975, O'Brien & McPherron 2000):
$$\frac{dDst^*}{dt} = Q(VB_s) - \frac{Dst^*}{\tau}$$

Empirical injection function:
$$Q(VB_s) = \begin{cases} -4.4 \times (VB_s - 0.5) & \text{nT/hr if } VB_s > 0.5 \text{ mV/m} \\ 0 & \text{otherwise} \end{cases}$$

Decay time: $\tau \approx 7.7$ hr (O'Brien & McPherron 2000) / 감쇠 시간

**Terms**:
- $Dst^*$: pressure-corrected storm time disturbance / 압력 보정 폭풍 교란
- $V$: solar wind speed (km/s)
- $B_s = -B_z$ if $B_z < 0$, else 0 (nT) / 남향 자기장 성분
- $VB_s$: convective electric field (mV/m) / 대류 전기장
- $P_{dyn} = n m_p v^2$: dynamic pressure / 동적 압력

**Carrington event estimate / 1859 캐링턴 사건 추정**:
- Extreme $VB_s \approx 20$ mV/m (sustained for 2–4 hours)
- Ring current injection $\sim -100$ nT/hr
- Estimated Dst: **-850 to -1760 nT** (Siscoe 2006; Lakhina 2013; Tsurutani 2003) vs 2003 Halloween storm Dst = -422 nT and 1989 March Quebec storm Dst = -589 nT.

**2003 Halloween storms / 2003 할로윈 폭풍**: Dst = -422 nT on Oct 30. Flare X17 (Oct 28), X28+ (Nov 4, saturated detector).

### 4.4 CME Radial vs Lateral Expansion / CME 반경 대 측면 팽창
Dal Lago (2003), Schwenn (2005):
$$V_{rad} = 0.88 \, V_{exp}$$

For cone angle $w$ (half-width):
$$V_{rad} = \frac{1}{2}(1 + \cot w) V_{exp}$$

For very fast CMEs (Michalek 2009):
$$V_{rad} \approx V_{exp}$$

### 4.5 SEP Onset Time vs Distance / SEP 온셋 시간-거리 관계
For a relativistic proton ($E \sim 1$ GeV):
$$t_{SEP} = \frac{L_{path}}{v_p} = \frac{L_{path}}{c\sqrt{1 - 1/\gamma_L^2}}$$

Parker spiral path length:
$$L_{path} = \int_{r_0}^{r} \sqrt{1 + \left(\frac{\Omega r}{v_{SW}}\right)^2} dr$$

Typical values: 1 AU Parker spiral at $v_{SW} = 400$ km/s → $L_{path} \approx 1.17$ AU, SEP travel time $\approx 8$–10 min for GeV protons.

### 4.6 ICME Geoeffectiveness Matrix / ICME 지자기 영향 매트릭스
Dst correlation with $VB_s$ (Kane 2005; Gopalswamy 2008):
$$Dst_{min} \approx -0.02 \, (V \cdot B_s)_{max} \cdot \Delta t$$

where $V$ in km/s, $B_s$ in nT, $\Delta t$ in hr.

| $V$ (km/s) \ $B_s$ (nT) | 5 | 10 | 20 | 40 |
|---|---|---|---|---|
| 400 | -20 | -40 | -80 | -160 |
| 600 | -30 | -60 | -120 | -240 |
| 800 | -40 | -80 | -160 | -320 |
| 1500 | -75 | -150 | -300 | -600 |

Stronger storms (super-intense, Dst < -500 nT) require $V \gtrsim 1500$ km/s and $B_s \gtrsim 50$ nT sustained.

### 4.7 Energy Partition (Aschwanden 2017) / 에너지 분배
$$E_{mag} \xrightarrow{\text{reconnection}} 0.87 \, E_{mag} = E_{released}$$
$$E_{released} = 0.10 \, E_{CME} + 0.80 \, E_{particles} + 0.10 \, E_{thermal}$$
$$E_{SEP} \approx 0.03 \, E_{CME}$$

### 4.8 X-class Flare Scaling / X-class 플레어 스케일링
GOES 1–8 Å SXR peak flux:
- C-class: $10^{-6}$ W/m²
- M-class: $10^{-5}$ W/m²
- X-class: $10^{-4}$ W/m²
- X10: $10^{-3}$ W/m²

**Occurrence frequency (per solar cycle, cycle 23)**:
- X-class: ~175 per cycle 23 (~15/year at max)
- M-class: ~1200 per cycle 23
- X10+: ~15 per cycle 23

### 4.9 Extreme Event Return Period / 극단 사건 재발 주기
Power-law distribution (Crosby 1993; Cliver & Dietrich 2013):
$$N(>E) \propto E^{-\alpha}, \quad \alpha \approx 1.4-1.8$$

Estimated return periods:
- Carrington-class event: **~500 years** (Love 2012; Schrijver 2012)
- March 1989-class: ~40 years
- Sept 2017-class: ~11 years (each cycle maximum)

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1859 ─ Carrington event (SEP estimate 2× Aug 1972)
│
1971 ─ Tousey: first CME observation (OSO-7)
1974 ─ MacQueen: Skylab CME catalog
│
1989 ─ March 1989 Quebec storm (Dst = -589 nT)
│
1995 ─ SOHO launch (LASCO, EIT)
2003 ─ Halloween storms (Dst = -422 nT, X28+ flare)
│
2006 ─ Schwenn "Space Weather: The Solar Perspective" [Paper #9]  ◄── BASE REVIEW
2006 ─ STEREO launch (3D CME era begins)
2008 ─ Thernisien GCS model
│
2010 ─ SDO launch (AIA, HMI, EVE)
2012 ─ July 23 super-CME (L5 miss, Dst estimate = -1100 nT)
2013 ─ Vršnak DBM model
│
2014 ─ STEREO-B signal lost
2017 ─ Aschwanden energy partition statistics
2017 ─ September 2017 events (NOAA 12673, X9.3, GLE)
│
2018 ─ Parker Solar Probe launch (0.05 AU)
2018 ─ Pomoell & Poedts EUHFORIA
2020 ─ Solar Orbiter launch (0.3 AU, out-of-ecliptic)
│
2021 ─ Temmer "Space Weather: The Solar Perspective (update)" [Paper #75, this] ◄── UPDATE
│
2027 ─ ESA Lagrange L5 mission (planned)
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **Paper #9: Schwenn (2006) Space Weather: The Solar Perspective** | Direct predecessor — this paper is its 15-year update / 직접적 선행 논문 — 15년 업데이트 | Highest — read Paper #9 first to see baseline; #75 reveals STEREO/SDO/PSP-era advances |
| **Chen (2011) Coronal mass ejections: models and their observational basis** | CME model review; Temmer cites extensively on halo CMEs / CME 모델 리뷰 | High — conceptual framework for CME physics referenced throughout Sec. 5 |
| **Pulkkinen (2007) Space weather: terrestrial perspective** | Complementary geomagnetic-side review / 지자기 측면의 보완 리뷰 | High — Temmer defers to Pulkkinen for geomagnetic impact details |
| **Kilpua et al. (2017) Coronal mass ejections and their sheath regions in interplanetary space** | ICME sheath physics / ICME sheath 물리 | High — Sec. 6 builds on Kilpua's sheath classification |
| **van Driel-Gesztelyi & Green (2015) Active region evolution** | AR source region physics / AR 소스 영역 | Medium — referenced in Sec. 3, 4 for flare source evolution |
| **Toriumi & Wang (2019) Flare-productive active regions** | AR flare productivity / AR 플레어 생산성 | Medium — referenced in Sec. 4 for AR classification |
| **Desai & Giacalone (2016) Large gradual SEP events** | SEP acceleration review / SEP 가속 리뷰 | Medium — Sec. 7 defers for detailed SEP physics |
| **Gombosi et al. (2018) Coronal and solar wind MHD modeling** | Numerical solar wind models / 수치 태양풍 모델 | Medium — Sec. 9, 11 for background MHD modeling |
| **Richardson (2018) Solar wind stream interaction regions throughout the heliosphere** | SIR/CIR review / SIR/CIR 리뷰 | Medium — Sec. 9 background on solar wind structures |
| **Camporeale (2019) Challenge of machine learning in Space Weather** | ML for Space Weather / ML 기법 | Medium — Sec. 11 survey of ML methods |
| **Hathaway (2010) The solar cycle** | Solar cycle background / 태양 사이클 배경 | Low — context for cycle 23/24 comparison |
| **Gopalswamy (2016) History of CME research** | CME historical perspective / CME 역사 | Low — Sec. 2 historical context |

---

## 7. References / 참고문헌

**Primary paper / 본 논문:**
- Temmer, M. (2021). Space weather: the solar perspective — An update to Schwenn (2006). *Living Reviews in Solar Physics*, 18:4. DOI: 10.1007/s41116-021-00030-3

**Directly updated / 직접 업데이트한 논문:**
- Schwenn, R. (2006). Space Weather: The Solar Perspective. *Living Reviews in Solar Physics*, 3:2.

**Key cited works / 주요 인용 논문:**
- Aschwanden, M. J. et al. (2017). Global Energetics of Solar Flares. V. Energy closure. *ApJ*, 836:17. DOI: 10.3847/1538-4357/836/1/17
- Bale, S. D. et al. (2019). Highly structured slow solar wind emerging from an equatorial coronal hole. *Nature*, 576:237.
- Burton, R. K. et al. (1975). An empirical relationship between interplanetary conditions and Dst. *JGR*, 80:4204.
- Cliver, E. W. & Dietrich, W. F. (2013). The 1859 space weather event revisited. *J Space Weather Space Clim*, 3:A31.
- Dal Lago, A. et al. (2003). A comparison between average values for ICME and CME radial and expansion speeds. *Solar Phys*, 215:135.
- Emslie, A. G. et al. (2012). Global energetics of thirty-eight large solar eruptive events. *ApJ*, 759:71.
- Fox, N. J. et al. (2016). The Solar Probe Plus Mission. *Space Sci Rev*, 204:7.
- Gopalswamy, N. (2016). History and development of coronal mass ejections as a key player in solar terrestrial relationship. *Geosci Lett*, 3:8.
- Kilpua, E. K. J. et al. (2017). Coronal mass ejections and their sheath regions in interplanetary space. *Living Rev Sol Phys*, 14:5.
- Müller, D. et al. (2020). The Solar Orbiter mission. *A&A*, 642:A1.
- Pomoell, J. & Poedts, S. (2018). EUHFORIA: European heliospheric forecasting information asset. *J Space Weather Space Clim*, 8:A35.
- Scolini, C. et al. (2020). CME-CME interactions as sources of CME geo-effectiveness. *ApJSS*, 247:21.
- Temmer, M. & Nitta, N. V. (2015). Interplanetary Propagation Behavior of the Fast Coronal Mass Ejection on 23 July 2012. *Solar Phys*, 290:919.
- Thernisien, A. F. R. et al. (2006). Modeling of flux rope coronal mass ejections. *ApJ*, 652:763.
- Vršnak, B. et al. (2013). Propagation of interplanetary coronal mass ejections: The drag-based model. *Solar Phys*, 285:295.
- Yashiro, S. et al. (2006). Visibility of coronal mass ejections as a function of flare location and intensity. *JGR*, 111:A12S05.
