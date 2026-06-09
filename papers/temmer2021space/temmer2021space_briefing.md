---
title: "Pre-Reading Briefing: Space Weather — The Solar Perspective"
paper_id: "75"
topic: Living_Reviews_in_Solar_Physics
date: 2026-04-23
type: briefing
---

# Space Weather: The Solar Perspective — Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Temmer, M. (2021). "Space weather: the solar perspective — An update to Schwenn (2006)". *Living Reviews in Solar Physics*, 18:4. DOI: 10.1007/s41116-021-00030-3
**Author(s)**: Manuela Temmer (University of Graz, Austria)
**Year**: 2021

---

## 1. 핵심 기여 / Core Contribution

**한국어:**
이 논문은 Schwenn(2006)의 Living Review를 15년 만에 업데이트한 것으로, 태양 관점에서 우주 기상(Space Weather)을 종합적으로 검토한다. 태양을 활성 별(active star)로 보고, 태양 활동이 만들어내는 코로나질량방출(CME), 플레어(flare), 태양에너지입자(SEP), 태양풍 흐름 상호작용 영역(SIR) 등이 지구를 포함한 행성간 공간에 미치는 영향을 다룬다. 핵심 업데이트는 (i) STEREO 임무의 다중 시점 관측 혁명, (ii) SDO의 고해상도 EUV·자기장 관측, (iii) Parker Solar Probe(2018)와 Solar Orbiter(2020)의 태양 근접 관측, (iv) 예보 모델링(DBM, EUHFORIA, ENLIL)과 머신러닝 기법의 발전이다. 저자는 2017년 9월 사건(NOAA 12673 활성영역에서 발생한 X9.3 및 X8.2 플레어)을 대표 사례로 들어, 플레어-CME-SEP 사슬이 태양 관측 서명으로부터 지자기 영향(Dst 지수)까지 어떻게 추적될 수 있는지를 보여준다.

**English:**
This paper is a 15-year update to Schwenn's (2006) Living Review, providing a comprehensive survey of Space Weather from the solar perspective. Viewing the Sun as an active star, it covers how solar activity phenomena — CMEs, flares, SEPs, and solar wind stream interaction regions (SIRs) — affect interplanetary space and planetary atmospheres. Key updates include (i) the multi-viewpoint observational revolution enabled by STEREO, (ii) SDO's high-resolution EUV and magnetic field observations, (iii) close-Sun observations from Parker Solar Probe (launched 2018) and Solar Orbiter (launched 2020), and (iv) advances in forecasting (DBM, EUHFORIA, ENLIL) and machine-learning methods. The September 2017 storm sequence (X9.3 and X8.2 flares from NOAA AR 12673) is presented as the canonical case showing how the flare-CME-SEP chain propagates from solar surface signatures to geomagnetic impact (Dst index) at Earth.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

**한국어:**
우주 기상의 개념적 출발점은 1859년 Carrington 사건으로, 전신망 고장과 저위도 오로라를 유발한 극단적 지자기 폭풍이다. 현대적 우주 기상 연구는 (1) 1971년 Tousey의 최초 CME 관측과 1974년 MacQueen의 Skylab 관측, (2) 1995년 SOHO 발사(LASCO, EIT), (3) 2006년 STEREO 쌍성 위성의 3D 관측 혁명, (4) 2010년 SDO 발사(AIA, HMI), (5) 2018년 Parker Solar Probe, 2020년 Solar Orbiter 발사로 발전해왔다. Schwenn(2006) 이후 15년간 우주기상 연구는 통계 기반에서 물리 기반으로, 단일 시점에서 다중 시점으로, 경험적 관계에서 MHD 모델링으로 패러다임 전환을 경험했다.

**English:**
The conceptual starting point of Space Weather is the 1859 Carrington event — an extreme geomagnetic storm that disrupted telegraph networks and produced aurorae at low latitudes. Modern Space Weather research advanced through (1) the first CME observations by Tousey (1971) and MacQueen's Skylab (1974), (2) the 1995 launch of SOHO (LASCO, EIT), (3) the 2006 STEREO twin-spacecraft 3D revolution, (4) the 2010 SDO launch (AIA, HMI), and (5) Parker Solar Probe (2018) and Solar Orbiter (2020). In the 15 years since Schwenn (2006), the field has transitioned from statistics-driven to physics-driven, from single-viewpoint to multi-viewpoint, and from empirical relations to MHD modeling as dominant paradigms.

### 타임라인 / Timeline

```
1859 ── Carrington event (Dst ≈ -850 to -1760 nT estimate)
1971 ── First CME observation (Tousey, OSO-7)
1974 ── Skylab white-light coronagraph CME catalog (MacQueen)
1989 ── March 1989 storm: Quebec blackout (Dst = -589 nT)
1995 ── SOHO launch (LASCO/EIT)
2003 ── Halloween storms (Oct-Nov 2003, X28+ flare, Dst = -422 nT)
2006 ── Schwenn Living Review "Space Weather: The Solar Perspective"  ◄── [Paper #9]
2006 ── STEREO launch (twin spacecraft, 3D CME)
2010 ── SDO launch (AIA, HMI, EVE)
2012 ── July 2012 super-CME (L5 miss, would have been Dst ≈ -600 to -1100 nT)
2017 ── September 2017 events: NOAA 12673, X9.3+X8.2 flares, Dst = -142 nT
2018 ── Parker Solar Probe launch (perihelion 0.05 AU)
2020 ── Solar Orbiter launch (out-of-ecliptic, 0.3 AU)
2021 ── Temmer Living Review (this paper) ◄── [UPDATE to Paper #9]
2027 ── ESA Lagrange L5 mission (planned)
```

---

## 3. 필요한 배경 지식 / Prerequisites

**한국어:**
1. **MHD 기초**: Alfvén 속도 $v_A = B/\sqrt{\mu_0 \rho}$, 자기 재결합, 동결된(frozen-in) 자기장 조건
2. **태양 자기장**: 활성영역 분류(McIntosh, Hale's classification), 자기 헬리시티, 자기 플럭스 로프(flux rope)
3. **코로나질량방출(CME) 모델**: CSHKP 표준 모델, 그라드-샤프라노프 구조, 자기 파열(magnetic breakout), 토크 불안정성(torus instability)
4. **입자 가속**: 확산 충격 가속(Diffusive Shock Acceleration, DSA), 확률론적 가속, 자기 재결합 가속
5. **행성간 물리**: Parker 나선, 태양풍 분류(고속/저속), SIR/CIR 구조, IMF의 $B_z$ 성분
6. **지자기 지수**: Dst(ring current), Kp(global), AE(auroral), $VB_s$ 전기장 결합

**English:**
1. **MHD fundamentals**: Alfvén speed $v_A = B/\sqrt{\mu_0 \rho}$, magnetic reconnection, frozen-in flux condition
2. **Solar magnetism**: Active region classification (McIntosh, Hale), magnetic helicity, magnetic flux rope topology
3. **CME models**: CSHKP standard model, Grad-Shafranov flux rope fitting, magnetic breakout, torus instability
4. **Particle acceleration**: Diffusive Shock Acceleration (DSA), stochastic acceleration, reconnection acceleration
5. **Interplanetary physics**: Parker spiral, solar wind classification (fast/slow), SIR/CIR structure, IMF $B_z$ component
6. **Geomagnetic indices**: Dst (ring current), Kp (global), AE (auroral), $VB_s$ electric field coupling

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **CME (Coronal Mass Ejection)** | 태양으로부터 방출되는 자화 플라즈마 덩어리. 질량 $10^{14}$–$10^{16}$ g, 속도 300–3000 km/s / Magnetized plasma cloud ejected from the Sun. Mass $10^{14}$–$10^{16}$ g, speed 300–3000 km/s |
| **ICME** | CME의 행성간 대응체(interplanetary CME), in-situ로 관측 / Interplanetary counterpart of CME, measured in-situ |
| **SEP (Solar Energetic Particles)** | 태양 이벤트로 가속된 고에너지 입자. keV–GeV 범위 / High-energy particles accelerated in solar events. keV–GeV range |
| **GLE (Ground Level Enhancement)** | GeV급 양성자가 지상 중성자 모니터에 도달하는 극단 SEP 이벤트 / Extreme SEP event where GeV protons reach ground-level neutron monitors |
| **Stealth CME** | 태양 표면 서명 없이 발생하는 "숨은 CME", problem storm의 원인 / "Hidden" CME without surface signatures, cause of problem storms |
| **DBM (Drag-Based Model)** | CME 속도 예측 모델; $v(r) = v_{SW} + (v_0 - v_{SW}) \exp(-\gamma(r - r_0))$ / CME propagation model; see equation |
| **SIR/CIR** | Stream Interaction Region / Co-rotating Interaction Region, 고속-저속 태양풍 경계 / Fast-slow solar wind interaction boundaries |
| **Dst index** | 지구 적도 링 전류 교란; 지자기 폭풍 강도 지표(nT) / Equatorial ring current disturbance index; storm intensity measure (nT) |
| **Halo CME** | 관측자를 향해(또는 반대로) 전파하며 태양 디스크를 완전히 둘러싸는 것처럼 보이는 CME / CME that appears to surround the solar disk, propagating towards/away from observer |
| **Flux rope** | 꼬인 자기장 구조; CME의 핵심 / Twisted magnetic field structure; CME core |
| **CSHKP model** | Carmichael-Sturrock-Hirayama-Kopp-Pneuman 표준 플레어 모델 / Standard eruptive flare model |
| **Preconditioning** | 선행 CME가 후속 CME의 전파 조건을 변경하는 현상 / Prior CME altering conditions for subsequent CME propagation |

---

## 5. 수식 미리보기 / Equations Preview

**1. Drag-Based Model (DBM) / 항력 기반 모델:**
$$\frac{dv}{dr} = -\gamma (v - v_{SW}) |v - v_{SW}|$$
해석적 해(solution):
$$v(r) = v_{SW} + \frac{v_0 - v_{SW}}{1 \pm \gamma (v_0 - v_{SW})(r - r_0)}$$
한국어: 태양풍 속도 $v_{SW}$로 점근하는 CME 감속/가속을 모사. $\gamma$는 항력 파라미터($10^{-8}$–$10^{-7}$ km$^{-1}$).
English: Models CME deceleration/acceleration asymptoting to solar wind speed $v_{SW}$. $\gamma$ is drag parameter ($10^{-8}$–$10^{-7}$ km$^{-1}$).

**2. Alfvén Mach number (shock) / 충격파 알프벤 수:**
$$M_A = \frac{v_{CME} - v_{SW}}{v_A}, \quad v_A = \frac{B}{\sqrt{\mu_0 \rho}}$$
한국어: $M_A > 1$일 때 충격파 형성. SEP 가속에 핵심.
English: Shock forms when $M_A > 1$. Key to SEP acceleration.

**3. Dst prediction (Burton-type) / Dst 예측식:**
$$\frac{dDst^*}{dt} = Q(VB_s) - \frac{Dst^*}{\tau}$$
여기서 $Q(VB_s) \approx a \cdot VB_s$ (대류 전기장 결합), $\tau$는 링 전류 감쇠 시간 ($\sim 8$ hr).
where $Q(VB_s) \approx a \cdot VB_s$ (convection electric field coupling), $\tau$ is ring current decay time ($\sim 8$ hr).

**4. CME radial vs. lateral expansion / CME 반경 대 측면 팽창:**
$$V_{rad} = 0.88 \cdot V_{exp}$$ (Dal Lago 2003, Schwenn 2005)
또는 / or $V_{rad} \approx V_{exp}$ (Michalek 2009, very fast CMEs)

**5. Energy partition / 에너지 분배 (Emslie et al. 2012, Aschwanden 2017):**
- Free magnetic energy $E_{mag}$ → ~10% CME kinetic ($E_{CME}$) + ~80% particle acceleration + ~10% thermal
- SEPs dissipate ~3% of $E_{CME}$

---

## 6. 읽기 가이드 / Reading Guide

**한국어:**
- **Section 1–3**: 도입, 우주 기상 정의, 자기 재결합 공통 기반 — 필수 배경 (빠르게 읽기)
- **Section 4 (Flares)**: CSHKP 모델, 플레어-CME 피드백, Neupert 효과 — 중점 학습
- **Section 5 (CMEs)**: 다중 시점, stealth CME, 조기 진화 — **핵심 섹션**, 천천히 읽기
- **Section 6 (ICMEs)**: HI 관측, in-situ 서명, flux rope 재구성
- **Section 7 (SEPs)**: gradual vs impulsive, 자기 연결성, GLE — 그림 19–22 주목
- **Section 8 (Energy budget)**: Aschwanden 2017 통계 결과 핵심
- **Section 9 (Solar wind)**: SIR/CIR, preconditioning, 태양풍 분류 — 배경 지식
- **Section 10 (Sept 2017)**: 종합 사례 연구 — **반드시 자세히 읽기**, 그림 32–37
- **Section 11 (Forecasting)**: DBM, EUHFORIA, ENLIL, ML — 실용적 응용

**English:**
- **Sec. 1–3**: Introduction, definition, reconnection — essential background (skim)
- **Sec. 4 (Flares)**: CSHKP, flare-CME feedback, Neupert effect — focus
- **Sec. 5 (CMEs)**: Multi-viewpoint, stealth, early evolution — **core**, read carefully
- **Sec. 6 (ICMEs)**: HI observations, in-situ, flux rope reconstruction
- **Sec. 7 (SEPs)**: Gradual vs impulsive, connectivity, GLE — see Figs. 19–22
- **Sec. 8 (Energy budget)**: Aschwanden 2017 statistics are key
- **Sec. 9 (Solar wind)**: SIR/CIR, preconditioning, classification — background
- **Sec. 10 (Sept 2017)**: Synthesis case study — **must read in detail**, Figs. 32–37
- **Sec. 11 (Forecasting)**: DBM, EUHFORIA, ENLIL, ML — practical applications

---

## 7. 현대적 의의 / Modern Significance

**한국어:**
이 리뷰는 2020년대 우주 기상 연구의 표준 참고 문헌이다. 과학적으로는 Parker Solar Probe와 Solar Orbiter 시대의 시작을 기록하며, 이들이 태양 코로나 하부 물리에 대한 새로운 창을 열 것임을 예고한다. 사회적으로는 ARTEMIS 달 임무와 Mars 탐사 시대에 인간 우주 비행사 방사선 안전을 위한 과학적 기반을 제공한다. 기술적으로는 정지궤도 위성, 항공 통신, 지상 전력망(GIC, Geomagnetically Induced Currents)에 대한 위협 평가에 활용된다. 산업적으로는 DBM과 같은 간단한 해석적 모델이 ESA/SSA의 실시간 예보에 채택되어, 연구와 운용 사이의 다리(R2O, Research-to-Operations)를 구축한다. 머신러닝 기법은 플레어 예보(Flarecast), SEP 예보(COMESEP), CME 도착 시간 예보에서 빠르게 확산 중이다.

**English:**
This review serves as the standard reference for 2020s Space Weather research. Scientifically, it marks the dawn of the Parker Solar Probe and Solar Orbiter era, anticipating new windows into low-coronal physics. Societally, it provides the scientific basis for radiation safety of astronauts in the ARTEMIS lunar missions and Mars exploration era. Technologically, it underpins threat assessments for geostationary satellites, aviation communication, and ground power grids (GIC — Geomagnetically Induced Currents). Industrially, simple analytical models like DBM are operationally adopted in ESA/SSA real-time forecasting, bridging Research-to-Operations (R2O). Machine-learning methods are rapidly proliferating in flare forecasting (Flarecast), SEP forecasting (COMESEP), and CME arrival time prediction.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
