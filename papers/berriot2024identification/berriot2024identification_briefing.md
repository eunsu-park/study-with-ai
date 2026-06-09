---
title: "Pre-Reading Briefing: Identification of a single plasma parcel during a radial alignment of the Parker Solar Probe and Solar Orbiter"
paper_id: "06_berriot_2024"
topic: Heliosphere_Solar_Wind
date: 2026-05-06
type: briefing
---

# Identification of a single plasma parcel during a radial alignment of the Parker Solar Probe and Solar Orbiter: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Berriot, E., Démoulin, P., Alexandrova, O., Zaslavsky, A., & Maksimovic, M. (2024). *Identification of a single plasma parcel during a radial alignment of the Parker Solar Probe and Solar Orbiter*. A&A, 686, A114. [DOI: 10.1051/0004-6361/202449285]
**Author(s)**: Etienne Berriot, Pascal Démoulin, Olga Alexandrova, Arnaud Zaslavsky, Milan Maksimovic (LESIA, Observatoire de Paris)
**Year**: 2024

---

## 1. 핵심 기여 / Core Contribution

### 한국어
2021-04-29 PSP(~0.075 au)와 Solar Orbiter(~0.9 au)가 거의 같은 태양 방사 방향(라디얼 alignment)에 놓인 드문 기회를 활용해, **두 우주선이 실제로 같은 슬로 태양풍 plasma parcel을 측정했는지를 정량적으로 식별하는 새로운 방법론**을 제시한다. 핵심 아이디어는 다음과 같다:

1. PSP가 어느 시각 $t_{\rm in}$에 횡단한 plasma parcel의 위치를 일정 가속도(constant acceleration) 모델로 시간 적분해 외부로 propagate한다.
2. propagate된 parcel과 Solar Orbiter 궤적 사이의 최소 거리 $d_{\rm min}(t_{\rm in})$을 계산한다.
3. $d_{\rm min}$이 가장 작은 시각이 같은 parcel이 두 우주선을 통과하는 line-up 시간이다.
4. 가속도 $a$를 자유 파라미터로 두고, Solar Orbiter에서 관측된 양성자 속도와 모델 예측 속도의 차이 $|\Delta V|$를 최소화하는 $a$를 선택한다.

이 방법으로 ~1.5 h 지속의 밀도 구조 하나가 PSP에서 Solar Orbiter까지 ~137 h(약 5.7일)에 걸쳐 ~0.825 au를 가로질러 **여전히 인식 가능한 형태로 보존됨**을 보였다. 또한 슬로 태양풍 parcel이 **~200 km/s → ~300 km/s로 유의미하게 가속**됨을 정량화했다 ($a \approx 0.2\,\rm m/s^{2}$ 수준).

### English
Using the rare radial alignment of the Parker Solar Probe (PSP, ~0.075 au) and Solar Orbiter (SolO, ~0.9 au) on 29 April 2021, this paper introduces **a new quantitative method to identify whether the two spacecraft actually sampled the same slow-wind plasma parcel**. The core idea:

1. Forward-propagate the position of a plasma parcel crossed by PSP at some time $t_{\rm in}$, assuming constant acceleration.
2. Compute the minimum distance $d_{\rm min}(t_{\rm in})$ between the propagated parcel trajectory and Solar Orbiter's orbit.
3. The $t_{\rm in}$ for which $d_{\rm min}$ is smallest is the candidate line-up time.
4. Treat the acceleration $a$ as a free parameter, scan over a range, and pick the value that minimises the difference $|\Delta V|$ between modelled and SolO-observed proton speeds.

A density structure of ~1.5 h crossing duration is shown to remain recognisable after a ~137 h transit across ~0.825 au of inner heliosphere, and the slow-wind parcel is found to **accelerate significantly from ~200 to ~300 km/s** ($a \approx 0.2\,\rm m/s^{2}$, i.e. $\Delta V/\tau \approx 100\,{\rm km/s}/137\,{\rm h}$).

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

#### 한국어
태양풍 가속과 in-situ 변형은 코로나 가열 문제와 함께 **태양물리학의 미해결 핵심 문제**이다. 두 우주선이 같은 방사 방향에 놓이는 "plasma line-up" 구성은 1976년 발사된 Helios 1·2 시대부터 이용되어 왔다:

- **Schwenn & Marsch (1983)**: Helios 1(0.51 au)과 Helios 2(0.72 au) 라인업에서 *동일 plasma의 라디얼 진화*를 처음 통계적으로 다룸. 일정 속도 315 km/s로 가정.
- **Telloni et al. (2021)**: PSP(0.1 au)–SolO(1 au) 라인업에서 **자기장 강도 cross-correlation**으로 line-up 시간 추정. 가속 거의 0으로 가정.
- **Alberti et al. (2022)**: PSP(0.17 au)–BepiColombo(0.6 au) line-up. *상호정보(mutual information)*와 sliding window correlation 사용.

이들 모두 **(a) 일정 속도, (b) 라디얼 전파, (c) cross-correlation 기반 매칭**이라는 강한 가정을 깔고 있어 가속도가 실제로는 무시할 수 없을 때 line-up 시간 추정이 부정확해진다. 본 논문은 *그 가정을 풀어 가속도까지 추정*한다는 점에서 새롭다.

#### English
Solar-wind acceleration and its in-situ evolution are, together with coronal heating, a core unresolved problem in solar physics. "Plasma line-up" configurations — two spacecraft on the same radial line — have been exploited since Helios 1·2 (launched 1974/76):

- **Schwenn & Marsch (1983)** — first statistical study using Helios 1 (0.51 au) and Helios 2 (0.72 au) line-ups; assumed a constant 315 km/s.
- **Telloni et al. (2021)** — PSP (0.1 au) / SolO (1 au) line-up; line-up times estimated from magnetic-field magnitude cross-correlation, near-zero acceleration assumed.
- **Alberti et al. (2022)** — PSP (0.17 au) / BepiColombo (0.6 au); mutual information and sliding-window correlation.

All assume **(a) constant speed, (b) radial propagation, (c) cross-correlation matching**. When acceleration is non-negligible — as it appears to be in the inner heliosphere — these assumptions distort the inferred line-up time. The Berriot+ method **lifts assumption (a)** and self-consistently fits the acceleration.

### 타임라인 / Timeline

```
1958 ──── Parker: solar wind hydrodynamic prediction
1962 ──── Mariner 2: first in-situ solar wind detection
1974/76 ─ Helios 1 & 2 launched (0.3 au perihelion)
1981a/b ─ Schwenn (Helios "plasma line-up" concept)
1983 ──── Schwenn & Marsch (constant-speed line-up evolution)
1995 ──── SOHO/Ulysses era (slow vs. fast wind dichotomy)
2018 ──── Parker Solar Probe launch (Aug 12)
2020 ──── Solar Orbiter launch (Feb 10)
2021-04-29 ─ ★ PSP / SolO radial alignment (this paper)
2021 ──── Telloni et al. (PSP-SolO line-up, B-field correlation)
2022 ──── Alberti et al. (PSP-BepiColombo, mutual information)
2024 ──── ★ Berriot et al. (this paper) — constant-a method
```

---

## 3. 필요한 배경 지식 / Prerequisites

### 한국어

1. **태양풍 가속 이론 / Solar-wind acceleration**: Parker (1958)의 등온 hydrodynamic 모델, slow vs. fast wind dichotomy(슬로 ~300-500 km/s, 패스트 ~500-800 km/s), 가속이 ~10-30 $R_\odot$ 안쪽에서 끝난다는 통념 vs. 최근 inner-heliosphere 가속 증거.
2. **궤도 기하 / Orbital geometry**: PSP 근일점 0.046 au, SolO 근일점 0.28 au; 이 둘의 황도면 ecliptic 기준 경도($\phi$)·위도($\theta$)가 거의 같아지는 시점이 라디얼 alignment.
3. **RTN 좌표계**: $\hat{R}$ = 태양 방사 방향, $\hat{T}$ = 태양 자전축 × $\hat{R}$, $\hat{N}$ = $\hat{R} \times \hat{T}$. 모든 우주선이 사용하는 표준.
4. **In-situ 측정**:
   - PSP/SWEAP/SPAN-i: proton 분포에서 $N_p, V_p$ 도출
   - SolO/SWA/PAS: 양성자+알파 합산 분포 (이 논문에서는 $\sim$100% 양성자 가정)
   - PSP/FIELDS, SolO/MAG: 자기장
5. **상관 분석**: Pearson $r$, cross-correlation, 시차(lag) 매칭, mutual information 개념.
6. **수치 propagation**: 운동 방정식 $\ddot R = a$를 시간 적분해 $R(t)$ 얻기.

### English

1. **Solar-wind acceleration**: Parker's (1958) isothermal hydrodynamic model; slow (~300-500 km/s) vs. fast (~500-800 km/s) wind; classical view of acceleration ending within ~10-30 $R_\odot$ vs. recent evidence of further inner-heliosphere acceleration.
2. **Orbital geometry**: PSP perihelion 0.046 au, SolO perihelion 0.28 au; radial alignment occurs when their heliographic longitude ($\phi$) and latitude ($\theta$) momentarily coincide.
3. **RTN frame**: $\hat{R}$ radial outward, $\hat{T}$ along solar rotation × $\hat{R}$, $\hat{N} = \hat{R} \times \hat{T}$.
4. **In-situ measurements**: PSP/SWEAP/SPAN-i proton moments $(N_p, V_p)$; SolO/SWA/PAS (proton + alpha total, treated here as ~100% proton); PSP/FIELDS and SolO/MAG magnetic fields.
5. **Correlation analysis**: Pearson $r$, cross-correlation, lag matching, mutual information.
6. **Numerical propagation**: integrating $\ddot R = a$ to obtain $R(t)$.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **Plasma parcel** | 태양풍의 한 작은 유체 요소. 자기장과 함께 태양에서 외향 전파됨. / A small fluid element of solar wind, advected outward with the magnetic field. |
| **Plasma line-up** | 두 우주선이 같은 방사선 위에서 같은 plasma parcel을 차례로 측정하는 구성. / Two spacecraft on the same radial line, intended to sample the same parcel sequentially. |
| **Radial alignment** | 두 우주선의 $(\phi, \theta)$가 동시에 같아지는 기하 (이 논문에서 2021-04-29 $t_0=00{:}45$ UT). / Geometry where both spacecraft simultaneously have the same heliographic $(\phi, \theta)$. |
| **Constant-velocity propagation** | parcel이 일정 속도 $V_{\rm in}$로 전파됨을 가정. line-up 시간 추정의 1차 근사. / Assumes a parcel propagates at constant $V_{\rm in}$. First-order line-up estimate. |
| **Constant-acceleration model** | $a$를 자유 파라미터로 두고 $R_{\rm out}(t) = R_{\rm in} + V_{\rm in}\tau + \tfrac{1}{2}a\tau^2$. 본 논문 핵심 모델. / Lets $a$ be a free parameter; the paper's core model. |
| **$t_{\rm in}$** | parcel이 PSP를 통과한 시각 (스캔 변수). / Time at which the parcel crosses PSP (scan variable). |
| **$t_{\rm out}$** | 같은 parcel이 propagate되어 SolO 궤도에 가장 가까워지는 시각. / Time at which the propagated parcel reaches its closest approach to SolO. |
| **$\tau = t_{\rm out} - t_{\rm in}$** | parcel transit time. ~137 h 내외. / Transit time of the parcel; ~137 h here. |
| **$d_{\rm min}$** | propagate된 parcel과 SolO 사이의 최소 거리. line-up 강도의 정량 지표. / Minimum distance between the propagated parcel and SolO; quantifies the line-up. |
| **$d_{\rm MIN}$** | $t_{\rm in}$ 스캔 중 가장 작은 $d_{\rm min}$. 최적 line-up. / The smallest $d_{\rm min}$ over the $t_{\rm in}$ scan; the optimal line-up. |
| **$\Delta V$** | SolO 관측 $V_{\rm out}$와 모델 예측 $V_{\rm out}$의 차. $a$ 추정에 사용. / Difference between observed and modelled outflow speed; used to fit $a$. |
| **Pearson correlation $r$** | 두 시계열의 선형 동조도. line-up 신뢰도 검증용. / Linear correlation of two time series; used to validate the line-up. |
| **Density structure** | 양성자 밀도 $N_p$의 시간적 패턴. parcel을 식별하는 "지문". / A temporal pattern in proton density; the "fingerprint" used to identify a parcel. |

---

## 5. 수식 미리보기 / Equations Preview

### Eq. (3) — 가속을 포함한 plasma parcel 위치 / Parcel position with acceleration

$$
\boldsymbol{R}(t, t_{\rm in}) \;=\; \boldsymbol{R}_{\rm in}(t_{\rm in}) \;+\; \int_{t_{\rm in}}^{t} \boldsymbol{V}(t', t_{\rm in})\,\mathrm{d}t'
$$
- $\boldsymbol{R}_{\rm in}(t_{\rm in}) \equiv \boldsymbol{R}_{\rm PSP}(t_{\rm in})$: PSP가 parcel을 만난 위치
- $\boldsymbol{V}$: parcel 속도(임의 프로파일 가능)
- 본질적으로 운동 방정식의 적분 형태

### Eq. (4) — parcel과 SolO 사이 거리 / Distance from parcel to SolO

$$
d(t, t_{\rm in}) \;=\; \bigl| \boldsymbol{R}_{\rm SolO}(t) \;-\; \boldsymbol{R}(t, t_{\rm in}) \bigr|
$$
$d$가 $t$에 대해 최소가 되는 점을 $d_{\rm min}(t_{\rm in})$이라 하고, 이를 다시 $t_{\rm in}$에 대해 최소화한 $d_{\rm MIN}$이 최적 line-up.

### Eq. (6)-(7) — 일정 가속 모델 / Constant-acceleration model

$$
\boldsymbol{R}(t) \;=\; \boldsymbol{R}_{\rm in} \;+\; (t - t_{\rm in})\boldsymbol{V}_{\rm in} \;+\; \tfrac{1}{2}(t - t_{\rm in})^2\,\boldsymbol{a}
$$
$$
\boldsymbol{V}(t) \;=\; \boldsymbol{V}_{\rm in} \;+\; (t - t_{\rm in})\,\boldsymbol{a}
$$
- 가장 단순한 가속 모델: 시간에 무관한 상수 $\boldsymbol{a}$
- $\tau = t_{\rm out} - t_{\rm in}$로 두면 $\boldsymbol{R}_{\rm out} = \boldsymbol{R}_{\rm in} + \tau \boldsymbol{V}_{\rm in} + \tfrac{1}{2}\tau^2 \boldsymbol{a}$

### Eq. (10) — 가속도 추정식 / Acceleration estimator

$$
\boldsymbol{a} \;=\; \frac{\| \boldsymbol{V}_{\rm in} + \boldsymbol{V}_{\rm out} \|}{2\,\| \boldsymbol{R}_{\rm out} - \boldsymbol{R}_{\rm in} \|}\,\bigl(\boldsymbol{V}_{\rm out} - \boldsymbol{V}_{\rm in}\bigr)
$$
- 가속도를 **관측된 두 속도와 거리만으로** 닫힌 형태로 표현
- $d(t, t_{\rm in})$ 최소화 단계와 독립이라는 점이 중요

### Eq. (12) — 속도 잔차 / Velocity residual

$$
\Delta V \;=\; \langle V_{\rm p, SolO} \rangle \;-\; V_{\rm out}
$$
- $\langle V_{\rm p, SolO} \rangle$: SolO에서 관측된 양성자 속도(1 h 평균)
- $V_{\rm out}$: 모델 예측 $V_{\rm out} = V_{\rm in} + a\,\tau$의 라디얼 성분
- $|\Delta V|$를 최소화하는 $a$를 선택 → 본 논문의 핵심 절차

---

## 6. 읽기 가이드 / Reading Guide

### 한국어

논문은 12쪽 분량으로 다음 5개 섹션이 핵심이다.

| 섹션 | 페이지 | 무엇을 / 어떻게 읽나 |
|---|---|---|
| **§1 Introduction** | 1-2 | 선행 line-up 연구(Schwenn, Telloni, Alberti) 비교를 빠르게 읽고, 이 논문의 차별점("가속도 자체를 추정")을 머리에 담기. |
| **§2 Data and line-up configuration** | 2 | Fig. 1 — PSP/SolO 궤도와 alignment 시간 $t_0$. Eq. (1)-(2) 시간 정의는 단순. SWEAP/SPAN-i, SWA/PAS, FIELDS, MAG의 어떤 데이터가 사용되는지 표/구절을 메모. |
| **§3 Ballistic propagation model** | 2-5 | **가장 중요한 섹션.** Fig. 2의 모식도로 propagation 절차 이해 → Eq. (3)-(5)로 일반 정의 → §3.1 일정 속도 → §3.2 일정 가속도 → Eq. (10)이 핵심. Fig. 3, 5는 각각 일정 속도/가속도에서 $d_{\rm min}$과 $\tau$ 결과. 두 그림의 $d_{\rm MIN}$ 위치가 일치하는지 확인. |
| **§4 Identification of the same density structure** | 5-9 | line-up 시점에서 $N_p$의 1.5 h 구조를 어떻게 식별·검증하는지. cross-correlation 신뢰도, lag, 자기장 일치 여부. Fig. 6-8 메모. |
| **§5 Discussion / Summary** | 9-11 | 가속도 결과(~0.2 m/s²), 137 h 횡단 후에도 구조 보존, slow-wind 가속의 함의. |

**읽기 팁 / Tips**
- $t_{\rm in}$과 $t_{\rm out}$의 차이를 끝까지 헷갈리지 말 것 — $t_{\rm in}$은 PSP 시각, $t_{\rm out}$은 SolO 시각.
- Fig. 5의 panel d가 가장 정보 밀도가 높음 ($\tau$, $t_{\rm out}$ vs. $t_{\rm in}$ — 가속도가 $\tau$를 ~50 h 단축).
- Eq. (10)의 분모 $\|\boldsymbol{R}_{\rm out} - \boldsymbol{R}_{\rm in}\|$이 ~0.825 au임을 늘 기억.

### English

The paper is 12 pages; five sections matter.

| Section | Pages | What / How to read |
|---|---|---|
| **§1 Introduction** | 1-2 | Skim the Schwenn / Telloni / Alberti comparison; note the differentiator: this paper *fits* the acceleration. |
| **§2 Data and line-up configuration** | 2 | Fig. 1 — orbits and alignment time $t_0$. Note which instruments and cadences are used (SWEAP/SPAN-i, SWA/PAS, FIELDS, MAG). |
| **§3 Ballistic propagation model** | 2-5 | **Most important.** Fig. 2 schematic → Eq. (3)-(5) general framework → §3.1 constant velocity → §3.2 constant acceleration → Eq. (10) is the punchline. Compare $d_{\rm MIN}$ locations across Figs. 3 and 5. |
| **§4 Identification of the same density structure** | 5-9 | How the 1.5 h $N_p$ structure is identified and validated (cross-correlation confidence, lag, B-field consistency). |
| **§5 Discussion / Summary** | 9-11 | Acceleration result (~0.2 m/s²), structure preserved over 137 h, implications for slow-wind acceleration. |

**Tips**
- Keep $t_{\rm in}$ (PSP-time) vs. $t_{\rm out}$ (SolO-time) straight throughout.
- Fig. 5d packs the most information ($\tau$, $t_{\rm out}$ vs. $t_{\rm in}$ — acceleration shortens $\tau$ by ~50 h).
- Remember $\|\boldsymbol{R}_{\rm out} - \boldsymbol{R}_{\rm in}\| \approx 0.825$ au throughout.

---

## 7. 현대적 의의 / Modern Significance

### 한국어

1. **slow solar wind 가속의 직접 증거**: Sanchez-Diaz et al. (2016), Maksimovic et al. (2020), Dakeyo et al. (2022) 등이 통계적으로 시사한 inner-heliosphere 가속을 **하나의 plasma parcel 차원에서** 정량화. ~200 → ~300 km/s는 30-50 % 가속률.
2. **plasma line-up 표준 방법론**: 향후 PSP/SolO/BepiColombo 라디얼 alignment 분석의 기준이 될 가능성. 일정 속도 가정의 한계가 드러남($\tau$가 ~50 h 빗나감).
3. **MHD 난류 진화 연구의 토대**: 같은 parcel을 두 거리에서 보면 inertial-range 스펙트럼·정규화 cross-helicity 등의 라디얼 진화를 직접 추적 가능 → Berriot+ 후속 논문의 기반.
4. **Solar Orbiter 미션의 First Results 기여**: A&A "Solar Orbiter First Results (Nominal Mission Phase)" 특별호 수록.

### English

1. **Direct evidence of slow-wind acceleration**: Quantifies, at the single-parcel level, the inner-heliosphere acceleration that statistical studies (Sanchez-Diaz+ 2016, Maksimovic+ 2020, Dakeyo+ 2022) had only suggested. The 30-50 % acceleration (~200 → ~300 km/s) is striking.
2. **Standard methodology for plasma line-ups**: Likely to become the reference for future PSP / SolO / BepiColombo radial-alignment analyses. The constant-velocity approximation is shown to mis-time $\tau$ by ~50 h.
3. **Foundation for MHD-turbulence radial evolution**: With the same parcel pinned at two heliocentric distances, downstream studies can track inertial-range spectra and cross-helicity radially.
4. **Part of A&A "Solar Orbiter First Results (Nominal Mission Phase)" special issue.**

---

## Q&A

### Q1. 타임 딜레이를 어떻게 해결했지? / How was the time delay resolved?

**핵심 답 / Bottom line**: τ = 137.6 h. 이는 **2단계 방법**으로 결정됨:
(1) ballistic propagation으로 거친 추정 (132-138 h 범위 좁힘)
(2) 라디얼 팽창 보정된 cross-correlation으로 0.1 h 분해능 정밀 결정.

#### Step 1 — Ballistic propagation (§3)

각 후보 $t_{\rm in}$(PSP에서 parcel을 만난 시각)마다 일정 가속도 $a$로 parcel 위치 $\boldsymbol{R}(t, t_{\rm in})$을 시간 적분하고 (Eq. 3, 6), parcel과 SolO 궤도 사이 거리 $d(t, t_{\rm in}) = \|\boldsymbol{R}_{\rm SolO}(t) - \boldsymbol{R}(t, t_{\rm in})\|$ (Eq. 4)을 $t$에 대해 최소화 → $d_{\rm min}(t_{\rm in})$. 이를 다시 $t_{\rm in}$에 대해 최소화 → 최적 line-up 후보.

| 가정 | 결과 $\tau$ | 비고 |
|---|---|---|
| 일정 속도 (§3.1, Fig. 3) | 145-185 h (40 h spread) | $V_{\rm in,PSP}$ 변동이 그대로 $\tau$에 전이 |
| 일정 가속도 (§3.2, Fig. 5d) | 132-138 h (좁아짐) | $V_{\rm in}$ 변동을 $a\tau$ 항이 부분 흡수 |
| 1차 추정 | $t_{\rm in} \approx 2.25$ h, $t_{\rm out} \approx 135$ h | $d_{\rm MIN} \approx 7 \times 10^6$ km |

⚠️ 중요: $d_{\rm MIN}$이 plasma dynamics가 아니라 **spacecraft 위도 차 $\Delta\theta \approx 3°$**(즉 $l_{\Delta\theta} \approx 7 \times 10^6$ km)로 정해진다는 사실이 propagation만으로는 $\tau$가 정확하지 않다는 단서.

#### Step 2 — Cross-correlation 정밀 보정 (§4)

논문 인용 (p. 6): *"The propagation model provides a first estimate of the time intervals corresponding to the plasma line-up. This estimate has to be made more precise, however, and it must be confirmed."*

절차:
1. PSP에서 $t \in [-0.5, 1.5]$ h 구간의 $N_p$ 밀도 enhancement + 동시에 anticorrelated B 감소를 "지문"으로 선택 (구조 지속 ~2 h)
2. SolO 데이터에서 $\tau \in [125, 145]$ h를 **0.1 h 간격**으로 스캔
3. 세 가지 상관계수 동시 계산 (Eqs. 14, 16, 17), 모두 $R$-팽창 보정 후

| 식 | 계수 | 특성 |
|---|---|---|
| Eq. (14) | Pearson $\rho_{X,Y}$ | 진폭 무관, false peak 많음 |
| Eq. (16) | 정규화 공분산 $\sigma_{X,Y}/\max$ | 큰 구조 우호 |
| Eq. (17) | 역 $\chi^2$: $1/\chi_{X,Y}$ | **봉우리 가장 좁고 또렷 — 논문 채택** |

결과 (Table 1, Fig. 7): 세 계수 모두 $\tau = 137.6$ h에서 absolute maximum.
- $N_p$: $\rho = 0.90$
- $B$: $\rho = 0.81$

#### 라디얼 팽창 보정 (cross-correlation의 핵심 전제)

$$
\delta X_c(t) = \delta X(t)\,(R_X/R_0)^{\varepsilon}, \qquad
\delta Y_c(t+\tau) = \delta Y(t+\tau)\,(R_Y/R_0)^{\varepsilon}
$$
- $N_p$: $\varepsilon = 2$ (구형 팽창 → $N_p \propto R^{-2}$)
- $B$: $\varepsilon = 1.6$ (Parker spiral, $R^{-2}$보다 천천히 감쇠; Mussmann+ 1977, Schwenn & Marsch 1990 통계)
- $R_0 = 1$ au

이 보정 없이는 PSP(0.075 au)의 $N_p \sim 4000$ cm⁻³와 SolO(0.9 au)의 $\sim 30$ cm⁻³가 직접 비교 불가.

#### 왜 cross-correlation이 잘 통하나? (Fig. 9 통찰)

**구조 전체에 같은 가속 프로파일이 적용되면, 구조의 두 부분 사이 *시간* 간격은 라디얼 진화 후에도 보존된다** — 단지 $\tau$만큼 시프트될 뿐. 즉 PSP에서의 $N_p(t)$와 SolO에서의 $N_p(t-\tau)$는 (R-보정 후) 거의 같은 함수가 됨. Fig. 8a에서 ①②③④로 표시된 4개의 5-20 min substructure가 양쪽에서 다 보임이 결정적 증거.

#### $\tau$ vs $a$ 결정의 분리

| 자유 파라미터 | 결정 기준 | 데이터 |
|---|---|---|
| 가속도 $a$ | $\|\Delta V\| = \|\langle V_{p,SolO}\rangle - V_{\rm out}\|$ 최소화 (Eq. 12) | SolO에서의 양성자 속도 |
| 타임 딜레이 $\tau$ | $R$-보정된 $1/\chi_{X,Y}$ 최대화 (Eq. 17) | $N_p$, B의 시간 패턴 |

→ **두 파라미터가 서로 다른 관측량으로 분리 결정**되므로 degeneracy 없음.

#### Nonradial propagation 효과 (Appendix A)

순수 라디얼 가정의 한계 점검:
- $V_T$ (azimuthal) 추가 → $t_{\rm in}$을 약간 시프트, $\tau$는 거의 불변
- $V_N$ (latitudinal) 추가 → $d_{\rm MIN}$을 $7 \times 10^6$ → $\sim 2 \times 10^6$ km로 줄여 (PSP-SolO를 같은 위도에 가깝게)
- 결론: $\tau$는 1 h 이내로만 흔들림 → 137.6 h ± O(1 h) 추정에 영향 없음.

#### 한 줄 요약 / One-liner

> Propagation 모델은 $\tau$의 *대략적 위치*만 잡고, 진짜 정밀 결정은 $R$-팽창 보정된 cross-correlation이 한다. $a$와 $\tau$는 서로 다른 관측량(속도 vs. 밀도 패턴)으로 분리 결정되므로 degeneracy가 없다.

> The propagation model only brackets where $\tau$ is; the precise value comes from R-corrected cross-correlation. Because $a$ is fixed by velocity ($|\Delta V|$) and $\tau$ by the density-pattern correlation, the two are inferred from independent observables and do not degenerate.

