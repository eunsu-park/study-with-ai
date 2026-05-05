---
title: "Pre-Reading Briefing: Geomagnetic Storms and Their Impacts on the U.S. Power Grid"
paper_id: "25_kappenman_2010"
topic: Space_Weather
date: 2026-04-19
type: briefing
---

# Geomagnetic Storms and Their Impacts on the U.S. Power Grid: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Kappenman, J. (2010). *Geomagnetic Storms and Their Impacts on the U.S. Power Grid* (Meta-R-319). Metatech Corporation, prepared for Oak Ridge National Laboratory under subcontract 6400009137.
**Author(s)**: John G. Kappenman (Metatech Corporation)
**Year**: January 2010

---

## 1. 핵심 기여 / Core Contribution

### 한국어
이 보고서(Meta-R-319)는 **극한 지자기 폭풍(extreme geomagnetic storms)이 미국 본토 전력망(U.S. Power Grid)에 미치는 위협을 최초로 정량화한 본격적 엔지니어링 평가**다. Kappenman은 Metatech가 10년 이상 개발해 온 검증된 결합 모델(coupled model)—(1) 지자기 폭풍 환경 모델, (2) 3-D 지층 지전도 구조 모델, (3) 연속 전압·GIC 흐름을 추적하는 미 대륙 EHV(345–765 kV) 송전망 회로 모델, (4) 변압기 반-주기 포화(half-cycle saturation) 모델—을 통합해 **1989년 3월 13–14일 Great Geomagnetic Storm**을 재현하고, 이를 **Carrington급(1859년) 시나리오**로 확장했다. 그 결과: (a) 약 **4,800 V/km** 수준의 지전장(geoelectric field)을 가정할 때, 미국 내 **300개 이상의 대형 EHV 변압기**가 GIC 유도 내부 과열(internal heating)로 **영구 손상**될 수 있으며, (b) 이는 **1.3억 명 이상**에게 영향을 주는 장기 정전(수개월~수년)을 초래할 수 있고, (c) 이러한 변압기는 주문 제작·리드타임 12–24개월로 **신속 교체가 불가능**하다는 것을 보여주었다. 이 보고서는 이후 FERC, NERC의 GMD(geomagnetic disturbance) 표준 제정(특히 NERC TPL-007) 및 National Academies "Severe Space Weather Events: Understanding Societal and Economic Impacts"(2008) 이후의 정책 논의를 결정적으로 밀어붙인 문건이다.

### English
This report (Meta-R-319) is **the first full-scale engineering assessment quantifying the threat that extreme geomagnetic storms pose to the continental U.S. power grid**. Kappenman integrates Metatech's decade-long coupled modeling framework — (1) a geomagnetic storm environment model, (2) a 3-D layered-Earth ground conductivity model that computes induced geoelectric fields, (3) a continental-scale EHV (345–765 kV) transmission circuit model tracking voltages and GIC flows on every node, and (4) a transformer half-cycle saturation model — to reproduce the **13–14 March 1989 Great Geomagnetic Storm** (the event that collapsed Hydro-Québec) and then extend the analysis to a **Carrington-class (1859) scenario**. Key findings: (a) at assumed peak geoelectric fields near **~4,800 V/km**, over **300 large EHV transformers** across the United States could suffer permanent damage from GIC-induced internal overheating; (b) this could produce a long-duration blackout affecting **>130 million people** for months to years; (c) such transformers are custom-built with 12–24 month lead times and cannot be rapidly replaced. The report decisively drove post-2008 U.S. policy debate (following the National Academies' *Severe Space Weather Events* report) and underpinned FERC/NERC GMD standards (notably NERC TPL-007).

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

**한국어**: 1989년 3월 Québec 블랙아웃(9시간 전력 상실, 600만 명 피해)은 지자기 폭풍이 "이론적 위협"이 아니라 **실측된 시스템 붕괴 원인**임을 증명했다. 하지만 2000년대 중반까지도 미국 규제당국·전력회사는 이를 **지역적·복구 가능한 사건**으로 보는 경향이 강했다. 2008년 National Academies 보고서("Severe Space Weather Events")가 **Carrington 이벤트 재발 시 $1–2조 규모 피해**를 경고하면서 연방 차원의 대응이 시작되었고, Kappenman 2010 보고서는 이 정책 창(policy window)에 정확히 엔지니어링 근거를 제공했다. 또한 2009년은 EMP Commission(전자기 펄스 위원회) 보고서와 맞물려 "natural GMD = natural E3"이라는 관점(지자기 폭풍의 저주파 성분이 고고도 핵폭발의 E3 성분과 유사한 메커니즘으로 전력망을 손상시킴)이 무게를 얻은 시기였다.

**English**: The March 1989 Québec blackout (9-hour loss of power, 6 million customers) proved that geomagnetic storms are **an empirically demonstrated cause of system collapse**, not just a theoretical concern. Yet into the mid-2000s, U.S. utilities and regulators still treated GMD as a **regional, recoverable event**. The 2008 National Academies report ("*Severe Space Weather Events: Understanding Societal and Economic Impacts*") broke the framing by warning of **\$1–2 trillion in damages** from a Carrington-class recurrence, opening a federal policy window. Kappenman's 2010 report arrived precisely into that window, providing the engineering basis regulators needed. The report also landed alongside the EMP Commission's work, reinforcing the "natural GMD behaves like natural E3" framing — that the low-frequency component of a severe storm damages the grid through the same mechanism (geoelectric field → GIC → transformer saturation) as the E3 late-time pulse from a high-altitude nuclear detonation.

### 타임라인 / Timeline

```
1859 ──── Carrington Event (first recorded extreme storm, telegraph fires)
            │
1921 ──── May 1921 storm (3× larger dB/dt than 1989 at some stations)
            │
1940 ──── March 1940 storm (major HV transmission anomalies, pre-GIC era)
            │
1972 ──── August 1972 storm (Bell telephone cable outages)
            │
1989 Mar ── Québec blackout (9h outage, 6M customers, transformer damage)  ★ benchmark event
            │
1991 ──── Kappenman et al. first detailed Québec forensic analysis
            │
2003 Oct ── Halloween Storms (Malmö, Sweden blackout, Eskom SA transformer damage)
            │
2008 ──── NAS "Severe Space Weather Events" report
            │
2010 Jan ── ★ Kappenman Meta-R-319 (THIS PAPER) — delivered to ORNL
            │
2011 ──── Kappenman JASTP companion papers
            │
2013 ──── FERC Order 779 → NERC begins GMD standard development
            │
2014–2016  NERC TPL-007-1/2 (GMD Vulnerability Assessment Standard) adopted
            │
2019+ ──── GMD benchmark event set at 8 V/km (much lower than Kappenman's 20 V/km scenarios) — ongoing controversy
```

---

## 3. 필요한 배경 지식 / Prerequisites

**필수 (Required)**
- **Paper #17 & #24** (선행 논문): GIC (Geomagnetically Induced Currents) 유도 메커니즘, Québec 붕괴 메커니즘. / Prior coverage of how GIC is induced from dB/dt and how it collapsed Hydro-Québec in 1989.
- **Faraday's law of induction** (기본 전자기학): ∇×**E** = −∂**B**/∂t. 시간 변화하는 자기장이 지표에 수평 전기장을 유도하고, 이 전기장이 장거리 송전선을 따라 준-DC 전류를 흘린다. / Time-varying magnetic field induces a horizontal geoelectric field at Earth's surface, which drives quasi-DC currents along long transmission lines.
- **Power system basics**: 3상 AC 송전(three-phase transmission), Y-grounded vs. delta winding, 345/500/765 kV EHV class, per-phase circuit model. / Three-phase AC transmission, neutral-grounded wye transformers, EHV voltage classes.
- **Transformer saturation**: B-H 히스테리시스, DC bias가 magnetization curve를 비대칭으로 밀어 반-주기 포화(half-cycle saturation)를 유발한다는 개념. / DC offset pushes the B-H curve asymmetric, driving the core into saturation on half of each AC cycle.

**권장 (Helpful)**
- **Earth conductivity structure**: 심도별 전도도(σ(z))가 서로 다른 층으로 이루어진 1-D/3-D 지층 모델; 고전도 층이 있으면 유도 전기장이 감소, 저전도(암반) 지역은 증폭. / Layered-Earth σ(z) model; resistive shields amplify the induced field, conductive basements suppress it.
- **Reactive power Q**: 변압기 포화 시 무효전력 소비가 급증해 전압 붕괴(voltage collapse) 위험. / Saturation causes MVAR demand spikes → voltage collapse risk.
- **GIC flow physics**: 변압기 중성점(neutral)이 접지된 Y권선에서만 GIC 경로가 형성된다. Delta-winding은 GIC를 차단. / GIC flows only through grounded-wye neutrals; delta windings block GIC.
- **Storm indices**: *Kp*, *Dst*, *AE*, **dB/dt (nT/min)** — dB/dt가 유도 E장의 핵심 동인. / *dB/dt* is the immediate driver of induced E.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **GIC (Geomagnetically Induced Current)** | 지자기 폭풍 중 지표 유도 전기장에 의해 송전선·파이프라인 등 긴 도체를 따라 흐르는 준-직류(quasi-DC, 0.001–0.1 Hz). 변압기 중성점을 통해 지접지(ground)로 귀환. / Quasi-DC current (0.001–0.1 Hz) driven through long conductors by the induced geoelectric field; returns via transformer neutrals to ground. |
| **dB/dt** | 지표 자기장의 시간 변화율(nT/min). Faraday 법칙에 따라 유도 전기장 크기의 직접 동인. 1989년 Québec 기록 ≈ **480 nT/min**; Carrington 재현 시 **>5,000 nT/min** 가정. / Rate of change of the surface magnetic field; the direct driver of induced E. March 1989 peak ~480 nT/min; Carrington-level scenarios assume >5,000 nT/min. |
| **Geoelectric Field (E, V/km)** | dB/dt + 지층 전도도 구조로부터 계산되는 지표 수평 전기장. 송전선 양단 전압은 약 **E · L** (L = 선로 길이). / Surface horizontal E-field computed from dB/dt and σ(z); line-end voltage ≈ E × L. |
| **Half-cycle Saturation (반-주기 포화)** | GIC가 변압기 권선에 DC 바이어스를 가해 B-H 곡선이 한쪽으로 밀려, AC 사이클의 반주기마다 코어가 포화되는 현상. 무효전력 소비 급증, 고조파 주입, 철심(+구조강) 과열. / DC bias offsets the B-H curve; core saturates on one half-cycle each AC period — massive MVAR draw, harmonic injection, and core/tie-plate overheating. |
| **EHV Transformer** | Extra-High Voltage 변압기 (≥345 kV). 미국 변압기 중 GIC에 가장 취약한 클래스. 주문 제작, 리드타임 12–24개월, 단가 \$10M+. / ≥345 kV class. Most GIC-vulnerable population; custom-built, 12–24 month lead times, >\$10M each. |
| **Tie-plate / Core-form GSU** | EHV 발전기 승압 변압기(GSU)의 금속 구조 부품. GIC 포화 시 누설 자속이 이곳에 집중되어 수분~수십분 내 수백 °C 국부 과열 → 절연 탄화. / Structural steel parts in generator step-up transformers; leakage flux from saturated cores concentrates here, causing hot spots >400°C and insulation failure. |
| **MVA / MVAR** | 피상전력/무효전력 단위. 포화 변압기의 MVAR 흡수가 수백 MVAR에 달함. / Apparent / reactive power in megavolt-amperes (reactive); a saturated transformer can absorb hundreds of MVAR. |
| **Voltage Collapse** | 무효전력 수급 붕괴로 시스템 전압이 회복 불가능하게 하락하는 현상. 1989 Québec 붕괴의 즉각적 메커니즘. / Loss of reactive-power balance drives voltages into unrecoverable decline; the immediate mechanism of the 1989 Québec collapse. |
| **3-D Earth Conductivity Model** | 경도·위도·심도별 σ의 3차원 지층 모델. 암반(shield) 지역은 E 증폭, 퇴적분지·해양은 감쇠. / 3-D model of subsurface conductivity; resistive shields amplify E, sedimentary basins / oceans suppress it. |
| **Carrington Event** | 1859년 9월 1–2일 관측된 사상 최대 지자기 폭풍. 본 보고서의 "극한 시나리오" 벤치마크. / Benchmark extreme-storm scenario (Sept 1859), the largest geomagnetic storm in the instrumental/historical record. |
| **Reactive Power Sink** | 지자기 폭풍 중 수천 대의 변압기가 동시에 MVAR을 소비하여 생기는 시스템 전체 무효전력 적자. / System-wide MVAR deficit when thousands of transformers saturate simultaneously. |
| **Auroral Electrojet** | 고위도 이온권의 동서 방향 전류. Substorm 중 급격히 강화되며 지표 dB/dt의 주요 원천. / High-latitude ionospheric current; its substorm intensification is the dominant driver of surface dB/dt. |

---

## 5. 수식 미리보기 / Equations Preview

### (1) Faraday 유도 (지전장의 출처) / Faraday induction — origin of the geoelectric field

$$
\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}
$$

**한국어**: 시간 변화하는 지자기장이 지표에 수평 전기장을 유도한다. 주파수 영역에서 1-D 층상 지층 가정 시 표면 전기장은 플레인-웨이브 임피던스 $Z(\omega)$를 통해 자기장과 연결된다.

**English**: A time-varying magnetic field induces a horizontal electric field at the surface. For a 1-D layered Earth in the frequency domain, the surface E and B are linked by the plane-wave surface impedance $Z(\omega)$.

$$
E_x(\omega) = Z(\omega)\,\frac{B_y(\omega)}{\mu_0}, \qquad Z(\omega) = \sqrt{\frac{i\omega\mu_0}{\sigma_{\text{eff}}(\omega)}}
$$

- $Z(\omega)$: 주파수 의존 표면 임피던스 (Ω). 고전도 지층이면 $|Z|$ 작고 $E$ 약화, 저전도(암반)면 $|Z|$ 크고 $E$ 강화.
- $\sigma_{\text{eff}}$: 유효 전도도(층상 구조의 가중치).

### (2) 송전선 유도 전압 / Induced line voltage

$$
V_{\text{line}} = \int_L \mathbf{E}\cdot d\boldsymbol{\ell} \;\approx\; E\, L \cos\theta
$$

- $L$: 선로 길이 (km). 765 kV 장거리선은 수백 km → 수백 V 유도.
- $\theta$: $\mathbf{E}$ 방향과 선로 사이 각도.
- **한국어**: 같은 지전장 하에서도 **긴 선로**가 더 큰 유도 전압을 받는다. 미국 EHV망이 유독 취약한 이유.
- **English**: For a given E-field, **longer lines pick up more voltage**. This is precisely why the continental U.S. EHV grid — with many 765 kV lines hundreds of km long — is so exposed.

### (3) GIC 회로 방정식 (단순화 형태) / GIC circuit equation (simplified)

$$
I_{\text{GIC}} = \frac{V_{\text{line}}}{R_{\text{line}} + R_{\text{wind}} + R_{\text{gnd},1} + R_{\text{gnd},2}}
$$

- 모든 저항은 **DC 저항** (GIC가 준-DC이므로). 리액턴스 무시.
- 실제 모델은 그리드 전체에 대해 Kirchhoff 노드 해석(수천 노드) — 본 보고서 Section 1.3의 핵심.
- **한국어**: AC 계통 보호용 임피던스(리액터)는 GIC를 차단하지 못한다. GIC에겐 그리드가 사실상 "DC 저항 네트워크"로 보인다.
- **English**: AC protective impedances (reactors) do not block GIC — to a DC-like current, the grid appears as a pure resistive network. Solved at continental scale by full nodal analysis.

### (4) 변압기 반-주기 포화 / Transformer half-cycle saturation

$$
B(t) = B_{\text{AC}}\sin(\omega t) + B_{\text{DC}}\;,\qquad B_{\text{DC}} \propto \frac{N\, I_{\text{GIC}}}{\mathcal{R}_{\text{core}}}
$$

- AC 플럭스에 DC 오프셋이 더해지면 매 AC 사이클의 한쪽 반주기에서 $|B|$가 포화점을 초과.
- 포화 시 자화전류(magnetizing current) 파형이 수 ms 동안 첨예한 스파이크가 되어 고조파(2, 3, 4, 5차 등) 주입 + MVAR 수요 폭증.

### (5) 변압기 국부 과열 / Transformer hot-spot heating

$$
T_{\text{hot}}(t) = T_{\text{ambient}} + \int_0^t \frac{P_{\text{leak}}(\tau)}{C_{\text{th}}}\,d\tau
$$

- $P_{\text{leak}}$: 포화로 인해 코어 밖으로 흘러나온 **누설 자속**이 tie-plate·탱크 벽 등 구조강에서 유도하는 와전류 손실 (비선형, $I_{\text{GIC}}$의 거의 2제곱).
- $C_{\text{th}}$: 열 용량. 기름 순환 냉각이 따라가지 못하면 **수분 내** >200°C 국부 온도 상승 가능.
- **한국어**: 이것이 "폭풍이 끝나도 변압기는 죽어 있다"의 물리적 근거.
- **English**: This is the physical basis for the report's most consequential claim — that a storm can leave hundreds of transformers permanently damaged even after the geoelectric field has subsided.

---

## 6. 읽기 가이드 / Reading Guide

**한국어**: 이 보고서는 분량이 많고(본문 약 130쪽 + 부록), 엔지니어링 보고서 특유의 반복 서술이 있다. 다음 순서로 읽으면 효율적이다.

1. **Foreword + Section 1 (모델 개요)** — 4개 결합 모델(환경·지층·전력망·변압기)의 구조를 먼저 숙지. 그림 1-1, 1-6, 1-11 등의 블록 다이어그램이 전체 논리의 뼈대.
2. **Section 2 (1989년 3월 사건 재현)** — 이 보고서의 **검증 섹션**. 모델이 실제 붕괴를 재현하는가? Section 2.1 (Québec 붕괴) + 2.2 (미국 측 substorm 분석: 7:40, 10:50, 21:20, 24:30 UT 4개 구간). Section 2.3의 변압기 내부 과열 사례 — 핵심 피해 메커니즘 증거.
3. **Section 3 (극한 시나리오)** — 본론. 1921년·Carrington급 dB/dt 가정 → GIC 분포 지도 → 300+ 변압기 손상 예측. **Figure 3-x 시리즈(GIC/전압 분포 지도)**가 가장 많이 인용되는 부분.
4. **Section 4 (위험 변압기 평가)** — 어떤 변압기가, 왜, 얼마나 위험한가. 경험적 손상 데이터 + 내부 과열 임계치. 정책적 함의가 가장 강한 섹션.
5. **Appendix A1**: 시스템 고장 임계치·릴레이 오작동 — GIC가 고조파 주입으로 보호 릴레이를 어떻게 '잘못' 작동시키는지.

**English**: This is a substantial report (~130-page main body + appendices) with engineering-style repetition. Recommended reading order:

1. **Foreword + Section 1 (model overview)** — internalize the four coupled models (storm environment → ground conductivity → EHV circuit → transformer). Block diagrams in Fig 1-1, 1-6, 1-11 scaffold the whole argument.
2. **Section 2 (reproducing March 1989)** — the **validation section**. Does the model reproduce the actual collapse? 2.1 (Québec collapse) + 2.2 (four U.S.-side substorm intervals: 7:40, 10:50, 21:20, 24:30 UT). Section 2.3's transformer-overheating case studies are the central damage-mechanism evidence.
3. **Section 3 (extreme scenarios)** — the headline. 1921-level / Carrington-class dB/dt → GIC distribution maps → prediction of 300+ transformer losses. The **GIC/voltage contour maps in Fig 3-x** are the most-cited content.
4. **Section 4 (at-risk transformer assessment)** — which transformers, why, and how much margin. Empirical damage data + internal heating thresholds. The most policy-consequential section.
5. **Appendix A1** — system-failure thresholds and relay misoperation; how GIC-induced harmonics trip protective relays incorrectly.

**읽는 동안 계속 자문 / Questions to keep asking while reading**:
- 이 수치(dB/dt, V/km, # transformers)의 **가정**은 무엇이며, 가정이 흔들리면 결론이 얼마나 흔들리는가?
- 1989년 벤치마크와 Carrington 시나리오의 **선형 외삽**이 물리적으로 정당한가?
- 3-D 지층 전도도 모델의 **불확실성**이 전체 오차의 어디쯤에 있는가?
- 이후 NERC가 채택한 **8 V/km 벤치마크**와 Kappenman의 ~20 V/km 가정은 왜 다른가?

---

## 7. 현대적 의의 / Modern Significance

### 한국어
- **규제 임팩트**: 이 보고서(+ 2008 NAS 보고서)가 FERC Order 779 (2013) → NERC **TPL-007 GMD Vulnerability Assessment Standard** (2014~) 제정의 기술적 근거였다. 미국 모든 BPS(Bulk Power System) 사업자는 현재 GMD 벤치마크 시나리오에 대한 취약성 평가가 **의무**다.
- **논쟁의 중심**: Kappenman의 Carrington급 예측(~4,800 V/km 지역)은 일부 연구자(Love et al., Pulkkinen et al.)로부터 "과도하게 외삽되었다"는 비판을 받았다. NERC 벤치마크는 결국 **8 V/km** (100년 재현주기)로 타협되었는데, 이 간극이 "GMD overhype vs underhype" 논쟁의 뿌리다.
- **기후·우주기상 복합 리스크**: 최근 NOAA/NASA의 우주기상 예보(DSCOVR, SWFO-L1), 그리고 2024년 5월 Gannon Storm(G5) 경험은 Kappenman의 결합 모델링 접근이 옳았음을 재확인했다 — 단지 규모 가정이 쟁점일 뿐.
- **변압기 예비재 정책**: STEP(Spare Transformer Equipment Program) 등 북미 전력회사 간 예비 변압기 공유 프로그램의 경제적 정당성이 이 보고서의 피해 추정에서 나온다.
- **연구 후속**: Pulkkinen et al. (2017)의 ExPRE 모델, Love et al.의 지자기 극값 통계, Oughton et al. (2017)의 경제 피해 재추정 등이 모두 Kappenman 프레임워크를 비판적으로 계승했다.

### English
- **Regulatory impact**: This report, together with the 2008 NAS report, was the technical basis for **FERC Order 779 (2013)** and **NERC TPL-007 GMD Vulnerability Assessment Standard (2014→)**. All U.S. Bulk Power System operators are now required to perform GMD vulnerability assessments against a benchmark event.
- **At the center of the controversy**: Kappenman's Carrington-class predictions (regional peaks ~4,800 V/km) were criticized by other researchers (Love et al., Pulkkinen et al., Oughton et al.) as aggressively extrapolated. NERC's adopted benchmark ultimately settled at **8 V/km** (100-year return level) — a large gap. The disagreement over *how* extreme extreme-storms actually get is directly traceable to this report.
- **Integrated space-weather risk**: Recent events and infrastructure (DSCOVR, SWFO-L1, the May 2024 Gannon G5 storm) vindicated Kappenman's *coupled-modeling* approach — only the magnitude assumptions remain contested.
- **Spare-transformer policy**: Industry programs like **STEP** (Spare Transformer Equipment Program) derive their economic justification from damage estimates rooted in this report.
- **Successor science**: Pulkkinen et al. (2017) ExPRE modeling, Love et al.'s extreme-value statistics of geomagnetic activity, and Oughton et al. (2017) economic re-estimates all inherit Kappenman's framework — even when they argue with its numbers.

### 이 논문을 reading list의 #24 → #25 흐름 속에서 / In context of papers #24 → #25

Paper #24가 "GIC가 실제로 전력망을 붕괴시킨 사건(Québec 1989)"을 기록한 forensic paper였다면, **#25는 그 물리를 모델화하여 *미래*를 예측하는 engineering assessment**다. 즉 Québec = 과거형 증거, Kappenman 2010 = 미래 시나리오. 이 둘을 짝으로 읽어야 GMD 리스크의 전체 논증 구조가 보인다.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
