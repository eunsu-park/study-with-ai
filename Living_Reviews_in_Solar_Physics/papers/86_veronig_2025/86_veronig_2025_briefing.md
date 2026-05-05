---
title: "Pre-Reading Briefing: Coronal Dimmings and What They Tell Us About Solar and Stellar Coronal Mass Ejections"
paper_id: "86"
topic: Living_Reviews_in_Solar_Physics
date: 2026-04-23
type: briefing
---

# Coronal Dimmings and CMEs: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Veronig, A. M., Dissauer, K., Kliem, B., Downs, C., Hudson, H. S., Jin, M., Osten, R., Podladchikova, T., Prasad, A., Qiu, J., Thompson, B., Tian, H., Vourlidas, A., "Coronal Dimmings and What They Tell Us About Solar and Stellar Coronal Mass Ejections", Living Reviews in Solar Physics, 22:2, 2025.
**Author(s)**: Astrid M. Veronig, Karin Dissauer, et al. (13 authors)
**Year**: 2025
**DOI**: 10.1007/s41116-025-00041-4

---

## 1. 핵심 기여 / Core Contribution

이 리뷰 논문은 코로나 디밍(coronal dimming) 현상에 대한 포괄적 총람이다. 1990년대 후반 SOHO/EIT와 Yohkoh/SXT로 처음 체계적으로 관측된 이후 25년간 축적된 관측·이론·시뮬레이션 결과를 종합한다. 코로나 디밍은 EUV 및 soft X-ray 영역에서 나타나는 국소적, 급격한 코로나 밝기 감소 현상으로, CME의 발생·질량 손실·자기 연결성(magnetic connectivity)의 진단 도구 역할을 한다. 저자들은 기존의 "core/secondary" 분류를 넘어 **자속계(magnetic flux system)에 기반한 새로운 물리 중심 분류체계**를 제안하며, 최근의 sun-as-a-star 및 항성 관측을 통해 코로나 디밍이 **항성 CME 탐지의 프록시(proxy)**로 확장될 수 있음을 보인다.

This review paper is a comprehensive synthesis of coronal dimming research: localized, sudden decreases of coronal EUV and soft X-ray emission that impulsively develop during the lift-off and early expansion of a CME. Drawing on 25+ years of SOHO/EIT, Yohkoh/SXT, SDO/AIA, and SDO/EVE observations alongside MHD simulations, the authors interpret dimmings as "footprints" of the erupting flux rope and indicators of coronal mass loss. Crucially, they propose a **new physics-driven categorization scheme based on the magnetic flux systems involved in the eruption** (flux-rope, strapping-flux, exterior, open-flux, and complex dimmings), and they extend the solar paradigm to **stellar CMEs**, where dimmings detected in EUV (SDO/EVE Sun-as-a-Star) and X-ray (XMM-Newton on Proxima Cen, AB Dor, AU Mic) light curves provide the first viable observational proxy for CMEs on late-type stars.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

1990년대 후반은 SOHO 미션(1995 발사)과 Yohkoh 미션(1991 발사)의 황금기였다. 전(全)원반(full-disk) EUV 이미징(SOHO/EIT, 1995)과 고감도 soft X-ray 이미징(Yohkoh/SXT, 1991)으로 처음 고카덴스로 코로나를 연속 관측할 수 있게 되면서, Hudson et al. (1996), Thompson et al. (1998), Zarro et al. (1999) 등이 CME와 동시에 발생하는 코로나 밝기 급감 현상을 발견하였다. Sterling and Hudson (1997)은 이 dimming이 **flux rope의 "footprint"**임을 제안하였다. 2010년 SDO 발사 후 AIA(6개 EUV 채널, 12초 카덴스)와 EVE(전원반 적분 EUV 분광)로 인해 dimming 연구는 대폭 확장되었으며, 2020년대 들어 XMM-Newton·Chandra·EUVE·HST/COS 관측을 통해 항성 CME 탐지에 적용되기 시작했다.

The late 1990s marked the golden era of SOHO (1995) and Yohkoh (1991). Full-disk EUV imaging (SOHO/EIT) and high-sensitivity soft X-ray imaging (Yohkoh/SXT) enabled continuous, high-cadence monitoring of the corona. Hudson et al. (1996), Thompson et al. (1998), and Zarro et al. (1999) discovered localized coronal brightness decreases accompanying CMEs. Sterling and Hudson (1997) first interpreted dimmings as **flux-rope footprints**. The 2010 launch of SDO — with AIA (6 EUV channels, 12 s cadence) and EVE (full-Sun EUV spectroscopy) — dramatically expanded dimming science. Since ~2020, XMM-Newton, Chandra, EUVE, and HST/COS have extended the paradigm to stellar CMEs on late-type stars.

### 타임라인 / Timeline

```
1956 ─ Waldmeier: "koronale Löcher" (coronal holes)
1973 ─ Skylab X-ray observations: "transient coronal holes"
1982 ─ Sheeley et al.: streamer blowouts in coronagraphs
1991 ─ Yohkoh/SXT launch → SXR dimming observations
1995 ─ SOHO launch (EIT, LASCO, CDS)
1996 ─ Hudson et al.: first systematic dimming observations
1997 ─ Sterling & Hudson: flux-rope "footprints" interpretation
1998 ─ Thompson et al.: SOL1997-05-12 twin dimming (classic event)
2010 ─ SDO launch (AIA, EVE, HMI) → modern dimming era
2014 ─ Mason et al.: Sun-as-a-Star EVE dimmings
2016 ─ Harra et al.: stellar CME dimming proxy proposed
2018 ─ Alvarado-Gómez et al.: MHD simulation of confined stellar CME
2021 ─ Veronig et al.: first stellar dimmings (AB Dor, AU Mic, Proxima Cen)
2022 ─ Loyd et al.: HST/COS dimmings on ε Eri (Fe XII, Fe XXI)
2025 ─ Veronig et al.: this comprehensive review
```

---

## 3. 필요한 배경 지식 / Prerequisites

**한국어**:
- **MHD 기초**: frozen-in flux, Alfvén 속도, 자기 재결합, Parker wind
- **태양 코로나 물리**: 광학적으로 얇은 플라즈마(optically thin), emission measure $EM = \int n_e^2 dL$, DEM(Differential Emission Measure) 분석
- **EUV 분광**: AIA 채널별 형성 온도(171 Å: 0.9 MK, 193 Å: 1.6 MK, 211 Å: 2 MK, 335 Å: 2.5 MK, 94/131 Å: hot), Fe XII 195.12 Å 등의 Fe 이온 진단선
- **CME와 flare**: two-ribbon flare 모델, flux rope 구조, strapping field, CSHKP/standard model
- **관측 기법**: base-difference / base-ratio / running-difference imaging, persistence maps
- **논문 #22 (Chen 2011)**: flux rope의 MHD 모델 — 이 리뷰 전반에 걸쳐 핵심
- **논문 #29 (Webb 2012)**: CME의 관측적 특성 — 코로나그래프 이미징, ICME 측정

**English**:
- **MHD fundamentals**: frozen-in flux, Alfvén speed, magnetic reconnection, Parker wind
- **Solar corona physics**: optically thin emission, emission measure $EM = \int n_e^2 dL$, DEM analysis
- **EUV spectroscopy**: AIA channel formation temperatures (171 Å: 0.9 MK, 193 Å: 1.6 MK, 211 Å: 2 MK, 335 Å: 2.5 MK, 94/131 Å: hot), Fe-ion diagnostic lines (Fe XII 195.12 Å, Fe XIII 202.04 Å, etc.)
- **CMEs and flares**: two-ribbon flare model, flux rope geometry, strapping field, CSHKP/standard model
- **Observational techniques**: base-difference / base-ratio / running-difference imaging, persistence maps
- **Paper #22 (Chen 2011)**: MHD models of flux ropes — foundational background
- **Paper #29 (Webb 2012)**: CME observational properties — coronagraphic imaging, ICME measurements

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **Coronal dimming** | EUV/SXR 코로나 복사의 국소적 급감; 주로 density depletion에 기인. / Localized impulsive decrease in EUV/SXR coronal emission, primarily from density depletion. |
| **Core (twin) dimming** | Flux rope 양쪽 footprint에 대응되는 깊고 좁은 dimming 쌍. / Deep, compact twin dimmings marking the flux rope footpoints. |
| **Secondary dimming** | 확장·재결합으로 더 넓게 퍼진 얕은 dimming. / Broader, shallow dimming from expansion/reconnection of overlying flux. |
| **Pre-eruption dimming** | 플레어 시작 전 수십 분~수 시간 동안 관측되는 약한 dimming. / Weak dimming detected before flare onset, indicating slow quasi-steady rise of MFR. |
| **Base-ratio / base-difference image** | 사전 이미지 대비 비율/차 영상 — dimming 검출의 표준 기법. / Pre-event image ratio/difference; standard dimming detection technique. |
| **Persistence map** | 시간 적분된 최소 강도 영상 — dimming의 전체 공간 범위 파악. / Time-integrated minimum-intensity map showing total dimming extent. |
| **DEM / Differential Emission Measure** | $DEM(T) = n_e^2 (dh/d\log T)$ — 온도별 방출 밀도 분포. / Temperature-resolved emission distribution. |
| **Flux rope (MFR)** | 꼬인(twisted) 자속관 — CME의 핵심 구조. Chen 2011 참조. / Twisted magnetic flux tube forming CME core. |
| **Strapping flux** | MFR를 억제하는 위 덮개 자기장. / Overlying confining magnetic field. |
| **Sun-as-a-star** | 태양을 공간 적분해 마치 별처럼 관측하는 방식. / Disk-integrated observations treating the Sun as unresolved star. |
| **Stealth CME** | 원반 관측에 거의 흔적을 남기지 않는 CME. / CME without clear low-coronal signatures. |
| **Magnetic cloud (MC)** | 1 AU 행성간 공간에서 측정된 MFR 구조. / Twisted magnetic structure measured in-situ at 1 AU. |
| **Absorption dimming** | 상승하는 필라먼트 물질에 의한 광학적 흡수로 나타나는 dimming. / Dimming due to optical absorption by erupted filament material. |

---

## 5. 수식 미리보기 / Equations Preview

### (1) 질량 손실 추정 / Mass loss from dimming
$$M = \delta N \cdot S \cdot L \cdot m_p$$
어디서 $\delta N$은 전자 밀도 변화, $S$는 dimming 면적, $L$은 dimming 깊이, $m_p$는 양성자 질량. 면적 $S$는 전원반 영상에서, $L = \sqrt{S}$로 근사하는 경우가 많음.
Here $\delta N$ is electron density change, $S$ is dimming area, $L$ is depth, $m_p$ proton mass. Often $L \approx \sqrt{S}$.

### (2) Dimming 강도 감소 / Intensity decrease
$$\frac{I_\lambda}{I_{\lambda,0}} \approx \left(\frac{L}{L_0}\right)^{1-2\alpha} \frac{R_\lambda(T)}{R_\lambda(T_0)}$$
팽창에 의한 emission measure와 온도 변화로 dimming 깊이를 예측. $\alpha=3$은 self-similar 팽창, $\alpha=1$은 1차원 선형 팽창. 단열 팽창 $\eta=2/3$, 등온 $\eta=0$.
Predicts dimming depth from expansion-induced EM and temperature changes. $\alpha=3$: self-similar, $\alpha=1$: linear LOS expansion. $\eta=2/3$: adiabatic, $\eta=0$: isothermal.

### (3) Dimming 면적 성장률 / Area growth rate
$$\frac{dA}{dt}, \quad \frac{d\Phi}{dt} = \int_A B_z \, dA$$
Impulsive dimming 단계에서 면적과 자속의 시간 변화율; CME 가속과 소프트 X-ray 플레어 피크와 강하게 상관.
$dA/dt$ and magnetic flux rate $d\Phi/dt$ correlate tightly with CME acceleration and SXR flare peak.

### (4) 항성 CME 질량 추정 (Loyd et al. 2022) / Stellar CME mass
$$\delta_{max} = \frac{F_{CME}}{F_{pre}}, \qquad m = \frac{\mu \delta_{max} F_{pre}}{n G(T, n)}$$
관측된 분광선 dimming 깊이 $\delta_{max}$에서 CME 질량을 추정. $F_{pre}$는 사전 flux, $G(T,n)$은 방출률 함수.
Fractional dimming depth yields CME mass estimate via the emissivity $G(T,n)$ and pre-flare flux.

### (5) Dissauer 회복 지수 / Recovery timescale
$$I(t) = I_{min} + (I_0 - I_{min})(1 - e^{-(t-t_2)/\tau})$$
지수함수적 회복; 평균 $\tau \sim 4.8 \pm 0.3$ hr (Reinard & Biesecker 2008 통계).
Exponential recovery with mean $\tau \sim 4.8 \pm 0.3$ hr.

---

## 6. 읽기 가이드 / Reading Guide

**한국어**:
이 리뷰는 124페이지로 매우 방대하다. 다음 순서로 읽는 것이 효과적이다:
1. **Sect. 1-2 (Intro, History)**: 빠르게 훑어 전체 맥락 이해 (15분)
2. **Sect. 3 (Observations)**: 핵심 물리 — main phase, recovery, pre-eruption을 중점적으로 (60분)
3. **Sect. 4 (Relations)**: Dimming-CME mass, kinematics, flare correlation — 통계 결과와 그림 중심 (45분)
4. **Sect. 5 (New categorization)**: 저자들의 **새로운 기여** — 자속계 기반 분류. 표(Table 1 or Fig. 46)를 참조 (45분)
5. **Sect. 6 (Simulations)**: MHD 예제 — 건너뛰어도 좋음 (20분)
6. **Sect. 7 (From Sun to stars)**: 항성 CME 응용 — 이 과제의 핵심. Figs. 56, 57, 58 주의깊게 (60분)
7. **Sect. 8 (Conclusions)**: 요약 및 미래 mission (10분)

**English**:
This 124-page review is massive. Recommended reading order:
1. **Sect. 1-2 (Intro, History)**: Skim for context (15 min)
2. **Sect. 3 (Observations)**: Core physics — main phase, recovery, pre-eruption (60 min)
3. **Sect. 4 (Relations)**: Dimming-CME mass/kinematics/flare correlations — focus on statistics and figures (45 min)
4. **Sect. 5 (New categorization)**: Authors' **key novel contribution** — flux-system-based categorization (45 min)
5. **Sect. 6 (Simulations)**: MHD examples — can skim (20 min)
6. **Sect. 7 (From Sun to stars)**: Stellar CME application — crucial for this study. Read Figs. 56, 57, 58 carefully (60 min)
7. **Sect. 8 (Conclusions)**: Summary and future missions (10 min)

---

## 7. 현대적 의의 / Modern Significance

**한국어**:
이 리뷰는 세 가지 측면에서 중대한 의의가 있다. 첫째, 우주 환경 예보(space weather) 관점에서 **코로나 디밍은 지구를 향하는 halo CME의 조기 경보 신호**가 될 수 있다. Coronagraph가 CME를 잡기 수십 분 전에 원반 관측으로 CME 출발을 확인할 수 있기 때문이다. 둘째, **자속계 기반 새로운 분류체계**는 관측과 MHD 모델을 직접 연결하며, 자기 재결합의 다양한 양상을 식별할 수 있는 물리적 틀을 제공한다. 셋째, **항성 CME 검출** — 지난 수십 년간 항성 플레어는 무수히 관측되었으나 CME 검출은 극히 드물었다. 코로나 디밍을 프록시로 삼는 방법론은 AU Mic, Proxima Cen, EK Dra 같은 가까운 활성 별에서 CME 빈도를 추정하고, **외계행성 거주가능성 평가**(CME에 의한 대기 박탈, 오존층 파괴, 자기권 압박)에 중요한 입력을 제공한다. 미래 미션(ESCAPE, Arcus, MUSE, SunCET, Vigil)이 이 방법론을 확장할 것이다.

**English**:
This review has significance in three major aspects. First, from a **space weather perspective**, coronal dimmings serve as early-warning signatures of Earth-directed halo CMEs — they appear on the disk tens of minutes before a CME enters the coronagraph field of view. Second, the **new flux-system-based categorization** directly connects observations to MHD models and provides a physical framework for identifying the various reconnection pathways involved. Third, **stellar CME detection** — while stellar flares are observed ubiquitously, CME detections remain extremely rare. The dimming-as-proxy methodology enables CME occurrence estimates for nearby active stars (AU Mic, Proxima Cen, EK Dra) and provides critical input to **exoplanet habitability assessment** (atmospheric erosion, ozone destruction, magnetospheric compression by CMEs). Upcoming missions (ESCAPE, Arcus, MUSE, SunCET, Vigil) will extend this methodology.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
