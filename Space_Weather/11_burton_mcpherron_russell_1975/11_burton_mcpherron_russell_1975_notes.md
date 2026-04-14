---
title: "An Empirical Relationship between Interplanetary Conditions and Dst"
authors: Rande K. Burton, Robert L. McPherron, Christopher T. Russell
year: 1975
journal: "Journal of Geophysical Research, Vol. 80, No. 31, pp. 4204–4214"
topic: Space Weather / Geomagnetic Storm Forecasting
tags: [Dst, ring current, Burton equation, solar wind, IMF Bz, dawn-dusk electric field, dynamic pressure, magnetopause compression, half-wave rectifier, energy balance, storm prediction, decay time, injection function, pressure correction, empirical model]
status: completed
date_started: 2026-04-11
date_completed: 2026-04-11
---

# An Empirical Relationship between Interplanetary Conditions and Dst (1975)
# 행성간 조건과 Dst 사이의 경험적 관계 (1975)

---

## 핵심 기여 / Core Contribution

Burton, McPherron & Russell (1975)은 태양풍 관측 데이터로부터 지자기 폭풍 강도 지수인 Dst를 예측하는 **최초의 정량적 경험식**을 도출한 논문이다. 이 논문의 핵심 기여는 세 가지이다: (1) Dst 변화를 **에너지 수지 방정식**(1차 ODE)으로 정식화했다 — ring current에 대한 에너지 주입(태양풍 dawn-dusk 전기장에 비례)과 에너지 손실(지수 감쇠, $\tau \approx 7.7$시간)의 균형으로 Dst의 시간 변화를 기술한다. (2) 태양풍 동압($P_{dyn}$)이 Dst에 미치는 **magnetopause 압축 효과를 분리**하여, ring current의 순수 기여만을 나타내는 "pressure-corrected Dst" ($Dst^*$)를 정의했다. (3) 에너지 주입 함수 $Q$가 **half-wave rectifier** 형태임을 실증했다 — 즉, dawn-dusk 전기장 $E_y$가 임계값(~0.5 mV/m)을 초과할 때만 ring current 주입이 발생하며, 이는 Dungey (1961)의 southward IMF reconnection 이론과 정확히 일치한다. 저자들은 1967년과 1968년의 7개 폭풍 구간에 대해 이 처방을 적용하여 관측 Dst와 예측 Dst가 놀라울 정도로 일치함을 보였다. 이 "Burton equation"은 이후 50년간 모든 경험적 지자기 폭풍 예보 모델의 **표준 틀**이 되었으며, O'Brien & McPherron (2000), Temerin & Li (2002, 2006) 등에 의해 개선되었지만 기본 구조는 변하지 않았다.

Burton, McPherron & Russell (1975) derived the **first quantitative empirical formula** for predicting the Dst geomagnetic storm intensity index from solar wind observations. Three key contributions: (1) Formulated Dst variation as an **energy balance equation** (first-order ODE) — balancing energy injection into the ring current (proportional to the solar wind dawn-dusk electric field) against energy loss (exponential decay, $\tau \approx 7.7$ hours). (2) **Separated the magnetopause compression effect** of solar wind dynamic pressure ($P_{dyn}$) on Dst, defining the "pressure-corrected Dst" ($Dst^*$) representing only the ring current contribution. (3) Demonstrated empirically that the injection function $Q$ has a **half-wave rectifier** form — ring current injection occurs only when the dawn-dusk electric field $E_y$ exceeds a threshold (~0.5 mV/m), consistent with Dungey's (1961) southward IMF reconnection theory. Applied to 7 storm intervals in 1967–1968, the prescription showed remarkable agreement between observed and predicted Dst. This "Burton equation" became the **standard framework** for all empirical geomagnetic storm forecasting models for the next 50 years, refined by O'Brien & McPherron (2000), Temerin & Li (2002, 2006) and others, but with the fundamental structure unchanged.

---

## 읽기 노트 / Reading Notes

### 1. Introduction — Dst의 의미와 한계 / Introduction — Meaning and Limitations of Dst

논문은 자기권이 태양풍의 다양한 교란에 반응하여 여러 형태의 교란(sudden commencement, main phase, recovery 등)을 생성한다는 관측 사실로 시작한다. 이 교란 형태들 사이의 **정량적** 관계가 당시까지 충분히 확립되지 않았다는 것이 핵심 동기이다.

The paper begins with the observation that the magnetosphere responds to various solar wind disturbances, producing different disturbance types (sudden commencement, main phase, recovery, etc.). The key motivation is that **quantitative** relationships between these disturbance types were not sufficiently established at the time.

**Dst의 정의**: 중위도 지자기 관측소 네트워크에서 측정한 $H$ 성분의 quiet day 기준 편차를 위도 보정 후 평균한 값이다.

**Dst definition**: Average of the $H$-component deviations from quiet-day baseline at mid-latitude stations, corrected for latitude.

$$Dst = \frac{1}{N} \sum_i \frac{H_{\text{observed},i} - H_{\text{quiet},i}}{\cos \lambda_i}$$

여기서 $\lambda_i$는 각 관측소의 지자기 위도이다. Burton et al.은 Table 1에 나열된 **10개 관측소**를 사용했다 (M'Bour, San Juan, Fredericksburg, Dallas, Boulder, Tucson, Tashkent, Kakioka, Hermanus, Tamanrasset 등). 이 관측소들의 지자기 위도는 20°N–45°N 범위에 분포한다.

Where $\lambda_i$ is each station's geomagnetic latitude. Burton et al. used **10 stations** listed in Table 1 (M'Bour, San Juan, Fredericksburg, Dallas, Boulder, Tucson, Tashkent, Kakioka, Hermanus, Tamanrasset, etc.), distributed at geomagnetic latitudes 20°N–45°N.

저자들은 Dst가 여러 전류 시스템의 기여를 포함한다고 설명한다:

Authors explain that Dst contains contributions from multiple current systems:

- **Ring current** ($H_R$): 가장 중요한 기여. 에너지 이온의 gradient-curvature drift에 의해 서향으로 흐르는 환상 전류. 지표면에서 $H$ 성분을 음의 방향으로 교란
- **Magnetopause currents** ($H_M$): Chapman-Ferraro 전류. 태양풍 동압에 의한 자기권 압축 → 지표면에서 $H$ 성분을 양의 방향으로 교란 (sudden commencement)
- **Tail currents** ($H_T$): cross-tail 전류. 지표면에서 소량의 음의 기여

- **Ring current** ($H_R$): dominant contributor. Westward toroidal current from gradient-curvature drift of energetic ions. Negative $H$ disturbance at surface
- **Magnetopause currents** ($H_M$): Chapman-Ferraro currents. Magnetospheric compression by dynamic pressure → positive $H$ at surface (sudden commencement)
- **Tail currents** ($H_T$): cross-tail current. Small negative contribution at surface

따라서 관측된 Dst는:

Therefore, observed Dst is:

$$Dst = H_R + H_M + H_T + H_{\text{other}}$$

**핵심 통찰**: Ring current의 순수 기여($H_R$)만을 분리하려면, magnetopause 압축 효과($H_M$)를 빼야 한다. 이것이 "pressure-corrected Dst*"의 동기이다.

**Key insight**: To isolate the pure ring current contribution ($H_R$), the magnetopause compression effect ($H_M$) must be subtracted. This motivates the "pressure-corrected Dst*."

---

### 2. 데이터 / Data

저자들은 Explorer 33, Explorer 35 위성의 MIT 플라즈마 실험 데이터와 GSFC 자력계 데이터를 사용했다:

Authors used MIT plasma experiment and GSFC magnetometer data from Explorer 33 and Explorer 35:

- **기간**: 1967년과 1968년 (2년)
- **태양풍 데이터**: Explorer 33 (1966–1968, 지구 근처 궤도), Explorer 35 (1967 이후, 달 궤도)
- **Dst 데이터**: World Data Center에서 제공한 2.5분 해상도 디지타이즈 데이터
- **폭풍 선별**: Dst < -40 nT인 약 35개 폭풍 중, 태양풍 데이터 커버리지가 좋은 것만 사용
- **자기장 해상도**: 5분 (Explorer 33의 자력계 데이터), 82.1초 (Explorer 35)
- **플라즈마 해상도**: hourly (속도, 밀도)

- **Period**: 1967 and 1968 (2 years)
- **Solar wind data**: Explorer 33 (1966–1968, near-Earth orbit), Explorer 35 (post-1967, lunar orbit)
- **Dst data**: 2.5-min resolution digitized data from World Data Center
- **Storm selection**: ~35 storms with Dst < -40 nT; only those with good solar wind coverage used
- **Magnetic field resolution**: 5 min (Explorer 33 magnetometer), 82.1 sec (Explorer 35)
- **Plasma resolution**: hourly (velocity, density)

**제한사항**: hourly 평균 태양풍 속도와 밀도 → 빠른 변동에 대한 감도 저하. 또한 자기장은 5분이지만 플라즈마는 1시간 → 동압 산출에 시간 불일치.

**Limitations**: hourly-averaged solar wind velocity and density → reduced sensitivity to rapid variations. Also, 5-min magnetic field but hourly plasma → temporal mismatch in dynamic pressure calculation.

---

### 3. Dst의 분해 / Decomposition of Dst

저자들은 Dst를 물리적 원천별로 분해하는 핵심적인 단계를 수행한다:

Authors perform the critical step of decomposing Dst by physical source:

$$Dst = Dst^* + b\sqrt{P_{dyn}} - c$$

따라서:

Therefore:

$$Dst^* = Dst - b\sqrt{P_{dyn}} + c$$

여기서:

Where:

- $Dst^*$: pressure-corrected Dst (ring current의 순수 기여)
- $b\sqrt{P_{dyn}}$: magnetopause Chapman-Ferraro 전류의 기여 (동압에 비례)
- $c$: quiet-time 보정 상수 (Dst가 완전 조용할 때도 0이 아닐 수 있으므로)

- $Dst^*$: pressure-corrected Dst (pure ring current contribution)
- $b\sqrt{P_{dyn}}$: magnetopause Chapman-Ferraro current contribution (proportional to $\sqrt{P_{dyn}}$)
- $c$: quiet-time correction constant (Dst may not be exactly 0 during perfectly quiet times)

**왜 $\sqrt{P_{dyn}}$에 비례하는가?** Chapman-Ferraro 이론에 따르면, magnetopause의 stand-off 거리 $r_{mp}$는 동압의 $-1/6$승에 비례한다:

**Why proportional to $\sqrt{P_{dyn}}$?** According to Chapman-Ferraro theory, magnetopause stand-off distance $r_{mp}$ scales as the $-1/6$ power of dynamic pressure:

$$r_{mp} \propto P_{dyn}^{-1/6}$$

지표면에서의 자기장 교란은 $r_{mp}^{-3}$에 비례하므로, $\Delta H \propto P_{dyn}^{1/2}$가 된다. 이 결과는 Siscoe and Crooker [1974]에 의해 이론적으로 확인되었고, Hirshberg et al.에 의해 관측적으로도 확인되었다.

The surface magnetic field perturbation scales as $r_{mp}^{-3}$, so $\Delta H \propto P_{dyn}^{1/2}$. This was confirmed theoretically by Siscoe and Crooker [1974] and observationally by Hirshberg et al.

Burton et al.이 결정한 상수값:

Constants determined by Burton et al.:

| 매개변수 / Parameter | 값 / Value | 단위 / Units |
|---|---|---|
| $b$ | 0.20 | $\gamma$ (eV cm$^{-3}$)$^{-1/2}$ |
| $c$ | 20 | $\gamma$ (nT) |
| $a$ (= $1/\tau$) | $3.6 \times 10^{-5}$ | s$^{-1}$ |
| $d$ | $-1.5 \times 10^{-3}$ | $\gamma$ (mV m$^{-1}$)$^{-1}$ s$^{-1}$ |

참고: 여기서 $\gamma = $ nT (nanotesla). 1 $\gamma$ = 1 nT.

Note: here $\gamma =$ nT (nanotesla). 1 $\gamma$ = 1 nT.

---

### 4. Burton Equation의 도출 / Derivation of the Burton Equation

#### 4.1 에너지 수지 접근 / Energy Balance Approach

Dessler, Parker, Sckopke (DPS) 관계에 따르면, Dst는 ring current 입자의 총 운동 에너지에 비례한다. 따라서 Dst*의 시간 변화는 ring current에 대한 에너지 주입률과 손실률의 차이이다:

According to the Dessler-Parker-Sckopke (DPS) relation, Dst is proportional to the total kinetic energy of ring current particles. Therefore, the time derivative of Dst* equals the difference between energy injection and loss rates:

$$\frac{dDst^*}{dt} = Q - \frac{Dst^*}{\tau}$$

여기서 $Q = F(E_y)$는 태양풍 전기장에 의존하는 에너지 주입 함수이고, $Dst^*/\tau$는 ring current의 자연 감쇠를 나타낸다.

Where $Q = F(E_y)$ is the energy injection function depending on solar wind electric field, and $Dst^*/\tau$ represents natural decay of the ring current.

#### 4.2 감쇠 시간 상수 결정 / Determining the Decay Time Constant

저자들은 **ring current 주입이 없는 구간**(quiet day나 recovery phase 초반)에서 감쇠율을 측정했다. $Q = 0$이면:

Authors measured the decay rate during **intervals with no ring current injection** (quiet days or early recovery phase). If $Q = 0$:

$$\frac{dDst^*}{dt} = -\frac{Dst^*}{\tau}$$

$$Dst^*(t) = Dst^*_0 \cdot e^{-t/\tau}$$

Figure 1은 동압을 보정한 후 23개 구간에 대해 ring current 주입이 없는 기간의 감쇠율을 도시한 것이다. 이 데이터에서 **$\tau \approx 7.7$ 시간** ($1/a = 1/(3.6 \times 10^{-5} \text{ s}^{-1}) = 27{,}778 \text{ s} \approx 7.7$ hr)을 결정했다.

Figure 1 shows the decay rate for 23 intervals with no ring current injection, after pressure correction. From these data, **$\tau \approx 7.7$ hours** was determined ($1/a = 1/(3.6 \times 10^{-5} \text{ s}^{-1}) = 27{,}778 \text{ s} \approx 7.7$ hr).

**중요한 세부 사항**: 감쇠는 Dst* = 0이 아니라 $Dst^* = -c = -20$ nT까지만 진행된다. 즉, quiet-time ring current가 존재하며, 완전한 "zero"는 아니다. 이는 $c = 20$ $\gamma$의 물리적 의미이다.

**Important detail**: Decay proceeds not to Dst* = 0 but to $Dst^* = -c = -20$ nT. A quiet-time ring current exists, and perfect "zero" is never reached. This is the physical meaning of $c = 20$ $\gamma$.

#### 4.3 에너지 주입 함수 결정 / Determining the Injection Function

Figure 2는 23개 구간에서 **동압이 거의 일정한 시기**의 ring current 주입률을 dawn-dusk 전기장 $E_y$의 함수로 도시한 것이다. 주입률은 관측된 $dDst^*/dt$에서 감쇠항 $Dst^*/\tau$를 빼서 계산했다.

Figure 2 plots the ring current injection rate for 23 intervals of **approximately constant dynamic pressure** as a function of the dawn-dusk electric field $E_y$. Injection rates were computed by subtracting the decay term $Dst^*/\tau$ from the observed $dDst^*/dt$.

**결과** (Figure 2 오른쪽 패널):

**Result** (Figure 2, right panel):

- $E_y < 0.5$ mV/m (북향 IMF 또는 약한 남향): **주입 없음** ($Q = 0$)
- $E_y > 0.5$ mV/m (충분히 강한 남향): **주입률이 $E_y$에 선형 비례**

- $E_y < 0.5$ mV/m (northward or weak southward IMF): **no injection** ($Q = 0$)
- $E_y > 0.5$ mV/m (sufficiently strong southward): **injection rate linearly proportional to $E_y$**

$$F(E_y) = \begin{cases} 0 & E_y < 0.5 \text{ mV/m} \\ d(E_y - 0.5) & E_y > 0.5 \text{ mV/m} \end{cases}$$

$d = -1.5 \times 10^{-3}$ $\gamma$ (mV/m)$^{-1}$ s$^{-1}$

Figure 3은 동일한 데이터를 log-log 스케일로 도시한 것이다. 기울기 1의 직선(선형 관계)과 기울기 2의 직선($E_y^2$에 비례)을 모두 비교했고, **선형 관계가 더 나은 적합**임을 보였다.

Figure 3 shows the same data on log-log scale. Both slope-1 (linear) and slope-2 ($E_y^2$) lines were compared, and **the linear relation provided a better fit**.

**물리적 해석**: 이 half-wave rectifier 형태는 Dungey의 reconnection 이론과 일치한다. 남향 IMF일 때만 주간측 reconnection이 일어나 태양풍 에너지가 자기권에 유입되고, 이 에너지가 궁극적으로 ring current에 축적된다.

**Physical interpretation**: This half-wave rectifier form is consistent with Dungey's reconnection theory. Only southward IMF drives dayside reconnection, allowing solar wind energy to enter the magnetosphere and ultimately accumulate in the ring current.

#### 4.4 전기장 정류(Rectification)에 대한 참고 / Note on Electric Field Rectification

저자들은 주입이 $E_y = VB_s$의 dawn-to-dusk 성분에만 의존한다는 점을 논의한다. $E_y$의 $Y$ 성분(solar magnetospheric Y)에 대한 의존성은 **"다소 놀라운(somewhat surprising)"** 결과라고 언급한다. $Y$ 성분은 단순히 radial solar wind velocity × north-south magnetic field이기 때문이다. 이는 이후 coupling function 연구의 기초가 되었다.

Authors discuss that injection depends only on the dawn-to-dusk component $E_y = VB_s$, noting the dependence on the $Y$ component (solar magnetospheric Y) is **"somewhat surprising."** The $Y$ component is simply radial solar wind velocity × north-south magnetic field. This became foundational for later coupling function studies.

#### 4.5 고주파 필터링 / High-Frequency Filtering

저자들은 빠르게 진동하는 전기장을 천천히 변하는 필드로 정류(rectify)하는 것이 물리적으로 타당함을 지적한다. 이를 모델에 반영하기 위해 **corner frequency 2 cph (cycles per hour), 6 dB/octave attenuation의 low-pass filter**를 적용했다. 필터의 corner frequency는 5분~2시간의 시간 스케일에 해당하며, 이는 자기권이 태양풍 변동에 반응하는 시간 스케일(전달 시간 + 자기권 내부 지연)에 해당한다.

Authors note that fast-oscillating electric fields should be rectified as slowly varying fields for physical consistency. They applied a **low-pass filter with corner frequency 2 cph (cycles per hour) and 6 dB/octave attenuation**. The corner frequency corresponds to 5 min–2 hour timescales, matching the magnetosphere's response time to solar wind variations (transit time + internal magnetospheric delay).

수학적으로, 실제 적용된 전기장과 동압은:

Mathematically, the actually applied electric field and dynamic pressure are:

$$E'(t) = h(t) * E(t - t_{sw} - t_m)$$
$$P'(t) = P(t - t_{sw})$$

여기서 $h(t)$는 자기권의 impulse response function, $t_{sw}$는 태양풍 전달 시간, $t_m$은 자기권 내부 지연이다. 실제로 저자들은 $t_m = 0$으로 설정하고, bow shock에서 magnetopause까지의 전달 시간만 프록시를 사용하여 보정했다.

Where $h(t)$ is the magnetosphere's impulse response, $t_{sw}$ is solar wind transit time, and $t_m$ is internal magnetospheric delay. In practice, authors set $t_m = 0$ and corrected only for bow shock to magnetopause transit time using a proxy.

---

### 5. 최종 처방과 검증 / Final Prescription and Validation

#### 5.1 완전한 Burton Equation / Complete Burton Equation

최종 처방 (Equation 4):

Final prescription (Equation 4):

$$\frac{d}{dt}Dst_0 = F(E) - aDst_0$$

여기서:

Where:

$$Dst_0 = Dst - b(P)^{1/2} + c$$

$$F(E) = 0, \quad E_y < 0.50 \text{ mV/m}$$
$$F(E) = d(E_y - 0.5), \quad E_y > +0.50 \text{ mV/m}$$

상수:

Constants:

$$a = 3.6 \times 10^{-5} \text{ s}^{-1}, \quad b = 0.20 \text{ } \gamma \text{ (eV cm}^{-3}\text{)}^{-1/2}$$
$$c = 20 \text{ } \gamma, \quad d = -1.5 \times 10^{-3} \text{ } \gamma \text{ (mV m}^{-1}\text{)}^{-1} \text{ s}^{-1}$$

$$E = VB_z \times 10^{-3} \text{ mV/m}, \quad P = nV^2 \times 10^{-2} \text{ eV/cm}^3$$

#### 5.2 검증 — 7개 폭풍 구간 / Validation — 7 Storm Intervals

저자들은 Table 2에 나열된 **7개 폭풍 구간** (1967년 4개, 1968년 3개)에 대해 Burton equation을 적용했다:

Authors applied the Burton equation to **7 storm intervals** listed in Table 2 (4 from 1967, 3 from 1968):

| 폭풍 구간 / Storm Interval | Quiet Day |
|---|---|
| Feb 7–8, 1967 | Jan 31, 1967 |
| Feb 15–17, 1967 | Jan 31, 1967 |
| Feb 23–24, 1967 | Jan 31, 1967 |
| Feb 27–28, 1968 | Jan 9, 1968 |
| Mar 3–5, 1968 | Apr 20, 1968 |
| May 1–2, 1968 | Apr 20, 1968 |
| Feb 15–16, 1968 | Jan 9, 1968 |

**Figures 4–10**: 각 폭풍에 대해 상단부터 (1) $\sqrt{P_{dyn}}$, (2) $E_y$, (3) 예측 Dst (점선) vs 관측 Dst (실선)을 도시했다. 모든 그래프에서 Dst는 양의 값이 아래를 향하도록(음의 값이 위를 향하도록) 도시되어 있어, 폭풍이 아래로 깊이 들어가는 것으로 시각화된다.

**Figures 4–10**: For each storm, plotted from top: (1) $\sqrt{P_{dyn}}$, (2) $E_y$, (3) predicted Dst (dashed) vs observed Dst (solid). In all plots, Dst is plotted with positive values downward, so storms visually dip deeper.

#### 5.3 개별 폭풍 분석 / Individual Storm Analysis

**February 23–24, 1967** (Figure 4) — "이상적 폭풍"

이 폭풍은 모든 특징을 가장 깨끗하게 보여준다:
- 0000 UT: $Dst \approx -10$ $\gamma$, $E_y$ 소폭 진동
- 0800 UT: 동압 증가 → Dst 양의 방향 이동 (sudden commencement 유사)
- 1200 UT: sudden commencement — 동압 급증으로 Dst 양의 첨두
- 1300–1900 UT: **$E_y$ 지속적 양** (남향 IMF) → ring current 주입 → **main phase** → Dst 하강, 최저 약 -60 $\gamma$
- 1900 UT 이후: $E_y$ 음으로 전환 (북향 IMF) → 주입 중단 → **recovery phase** 시작
- 그러나 recovery 중에도 소량의 양의 $E_y$에 의한 간헐적 주입이 회복을 지연시킴

This storm shows all features most cleanly: SC at ~1200 UT, sustained $E_y > 0$ driving the main phase (1300–1900 UT), then recovery when $E_y$ reverses.

**예측과 관측의 일치도**: 매우 양호. 다만 2월 24일의 느린 회복 구간에서 관측 Dst가 예측보다 느리게 회복 → **감쇠 시간이 폭풍 후반에 길어지는 가능성** 시사.

**Prediction-observation agreement**: Very good. However, during the slow recovery on Feb 24, observed Dst recovered more slowly than predicted → suggesting **decay time may be longer during late recovery**.

**February 7–8, 1967** (Figure 5) — 동압 변동이 두드러진 폭풍

- 0000 UT Feb 7: $E_y \approx 0$, 동압 감소 → $Dst$ 소폭 감소 (동압 효과)
- 0800 UT: 동압 증가 시작 → $Dst$ 양의 방향
- 1500–1700 UT: 동압 증가와 동시에 $E_y$ 양으로 전환 → 주입과 동압 효과가 동시 작용
- 1800–2300 UT: **main phase** — $E_y$ 지속적 양, $Dst$ 하강
- 0000 UT Feb 8: $E_y \approx 0$ → recovery 시작
- 0700 UT Feb 8: 다시 $E_y$ 양 → **2차 주입** → Dst 재하강

A storm where dynamic pressure variations are prominent and a second injection episode occurs.

**February 15–17, 1967** (Figure 6) — 복잡한 다단계 폭풍

극도로 변동성이 큰 전기장을 가진 복잡한 폭풍. 주입과 동압 효과가 교차하며, 예측이 정성적으로는 잘 맞지만, 전기장의 높은 변동성 때문에 세부 구간에서 차이 발생.

A complex multi-phase storm with highly variable electric field. The prediction qualitatively matches well but differs in details due to high electric field variability.

**February 15–16, 1968** (Figure 7)

- 0000 UT Feb 15: $Dst > 0$ — 동압 효과. 전기장은 7시간 동안 음(negative, 즉 북향 IMF)
- Dst 변동은 처음 7시간 동안 **거의 전적으로 동압에 의해** 구동됨
- 그 후 $E_y$ 양으로 전환 → main phase → Dst $\approx -60$ $\gamma$

Illustrates that Dst variations can be **driven almost entirely by dynamic pressure** during periods with northward IMF.

**February 27–29, 1968** (Figure 8) — 동압과 주입의 균형

$Dst \approx -30$ $\gamma$ 수준에서 주입과 감쇠가 **준정상 균형**을 이루는 구간이 있다. 이는 Burton equation의 정상 상태 해에 해당한다:

Has intervals where injection and decay reach **quasi-steady balance** at $Dst \approx -30$ $\gamma$. This corresponds to the steady-state solution of the Burton equation:

$$Dst^*_{\text{steady}} = Q \cdot \tau$$

**March 3–5, 1968** (Figure 9)

3일간 전기장이 거의 항상 양(남향 IMF) → 주입과 감쇠의 균형 유지 → $Dst \approx -30$ ~ $-50$ $\gamma$ 수준에서 지속. 날카로운 동압 하강(0730 UT Mar 5)이 $Dst$ 하강을 촉발한 흥미로운 구간.

Three-day period with nearly always positive $E_y$, maintaining balance. A sharp dynamic pressure drop on Mar 5 triggered a Dst decrease.

**May 1–2, 1968** (Figure 10)

점진적 main phase — $E_y$가 서서히 증가하며 1200 UT에 최저 약 -50 $\gamma$ 도달. 동압은 거의 일정. 가장 "깨끗한" ring current 주입/감쇠 과정을 보여주는 사례.

Gradual main phase with slowly increasing $E_y$. Nearly constant dynamic pressure — cleanest demonstration of pure ring current injection/decay.

---

### 6. Discussion — 모델의 한계와 Dst의 의미 / Discussion — Model Limitations and Meaning of Dst

저자들은 Discussion에서 몇 가지 중요한 한계를 솔직히 논의한다:

Authors candidly discuss several important limitations:

#### 6.1 Dst 자체의 한계 / Limitations of Dst Itself

- Dst는 자기권계면 안팎의 전류의 **근사치**이다. 이상적인 Dst는 더 많은 적도 관측소와 더 나은 quiet day 정의를 필요로 함
- **Quiet day 선택 문제**: 진정한 "조용한 날"은 매우 드물다. 저자들은 QQ day(quiet day of the month) 중 가장 조용한 날을 선택했지만, QQ day도 교란이 있을 수 있다. Quiet day의 $H$ 성분 차이가 20 $\gamma$에 이를 수 있어, 10 $\gamma$ 수준의 불일치를 초래
- February 15–16, 1968 (Figure 7) 사례에서 이 문제가 명확히 드러남: 0000 UT에서 quiet day 차이에 의한 10 $\gamma$ 이상의 초기 offset이 이후 6–8시간의 적합에 영향

- Dst is an **approximation** of currents inside and outside the magnetopause. Ideal Dst requires more equatorial stations and better quiet-day definition
- **Quiet day selection problem**: truly quiet days are rare. Authors selected quietest QQ day, but QQ days can have disturbances. Quiet-day $H$ differences can reach 20 $\gamma$, causing ~10 $\gamma$ discrepancies
- This problem is clearly illustrated in the Feb 15–16, 1968 case (Figure 7)

#### 6.2 감쇠 시간의 비일정성 / Non-Constant Decay Time

이 논문에서 가장 중요한 미해결 문제이다. 저자들은 단일 감쇠 시간 $\tau \approx 7.7$시간을 사용했지만, 여러 폭풍(특히 Feb 23–24, 1967)에서 회복 단계 후반의 감쇠가 예측보다 느리다. 이는 다음을 시사한다:

The most important unresolved issue in this paper. Authors used a single $\tau \approx 7.7$ hours, but in several storms (especially Feb 23–24, 1967), late recovery is slower than predicted. This suggests:

- 감쇠 시간이 **ring current 에너지(즉 Dst 크기)에 의존**할 수 있다. 강한 폭풍일수록 감쇠가 빠르고, 약해지면 감쇠가 느려짐
- 물리적으로: charge exchange, Coulomb scattering, wave-particle interaction 등의 손실 메커니즘이 에너지에 따라 다른 시간 스케일을 가지기 때문
- **이후 O'Brien & McPherron (2000)**이 $\tau$를 $E_y$의 함수로 만들어 이 문제를 부분적으로 해결

- Decay time may **depend on ring current energy** (i.e., Dst magnitude). Stronger storms decay faster, weaker storms slower
- Physically: loss mechanisms (charge exchange, Coulomb scattering, wave-particle interactions) have different timescales depending on particle energy
- **Later O'Brien & McPherron (2000)** partially solved this by making $\tau$ a function of $E_y$

#### 6.3 Sudden Commencement의 처리 / Treatment of Sudden Commencement

동압의 급격한 변화(step function)는 $Dst$에서 SC(sudden commencement)를 생성한다. Burton equation에서는 이것이 $b\sqrt{P_{dyn}}$ 항으로 자연스럽게 처리된다. 그러나 SC의 진폭이 $b\sqrt{P_{dyn}}$의 변화와 정확히 일치하지 않는 경우가 있으며, 이는 동압 데이터의 시간 해상도 문제일 수 있다.

Rapid dynamic pressure changes (step functions) produce SC in Dst. The Burton equation handles this naturally through the $b\sqrt{P_{dyn}}$ term. However, SC amplitudes do not always match $b\sqrt{P_{dyn}}$ changes exactly, possibly due to temporal resolution of dynamic pressure data.

#### 6.4 극한 폭풍에 대한 함의 / Implications for Extreme Storms

저자들은 마지막 섹션에서 극한 폭풍의 가능성을 추정한다:

Authors estimate extreme storm potential in the final section:

- $Dst \approx -300$ $\gamma$ 폭풍: 시간당 주입률 100–130 $\gamma$/hour, 50 $\gamma$/hour의 감쇠와 균형을 이루며 5–8시간 지속 필요
- 이는 $E_y \approx 20$–25 mV/m, $B_z \approx -20$–$-50$ $\gamma$ 수준의 전기장 필요
- $V_{sw} = 500$–1000 km/s의 태양풍 속도에서 $B_z \approx -24$ $\gamma$가 1시간 유지 시 Dst $\approx -150$ 가능
- 1957년과 1958년의 큰 폭풍 중 이 수준의 필드가 30분간 관측된 적이 있음
- **결론**: Burton equation은 가장 큰 폭풍과도 "not inconsistent"하다

- $Dst \approx -300$ $\gamma$ storm: requires injection rates of 100–130 $\gamma$/hour balanced against 50 $\gamma$/hour decay for 5–8 hours
- Requires $E_y \approx 20$–25 mV/m, $B_z \approx -20$–$-50$ $\gamma$ electric fields
- With $V_{sw} = 500$–1000 km/s, $B_z \approx -24$ $\gamma$ maintained for 1 hour could produce Dst $\approx -150$
- Fields at this level were observed for 30 min during 1957 and 1958 great storms
- **Conclusion**: Burton equation is "not inconsistent" with even the largest storms

---

### 7. 주입 함수의 가능한 대안적 해석 / Alternative Interpretations of the Injection Function

저자들은 $Q$가 단순히 $E_y$에 선형인 것이 유일한 가능성은 아니라고 논의한다:

Authors discuss that $Q$ being simply linear in $E_y$ is not the only possibility:

1. **다른 태양풍 매개변수의 영향**: ring current 주입은 전기장뿐 아니라 다른 매개변수(예: 태양풍 밀도, Mach number, IMF $B_y$ 성분)에도 의존할 수 있다. 그러나 "태양풍-자기권 상호작용이 매우 복잡하므로" 전기장만의 함수로 근사하는 것이 합리적이라고 판단
2. **주파수 의존성**: 자기권의 응답이 주파수 의존적일 수 있으며, $H(\omega)$의 형태가 단순 low-pass가 아닐 수 있다
3. **비선형성**: 실제로 일부 데이터 점은 $E_y^2$에 비례하는 것처럼 보이기도 하지만, 전체적으로 선형 적합이 더 좋다 (Figure 3)

1. **Other solar wind parameters**: injection may depend on density, Mach number, IMF $B_y$, etc. But approximating as a function of electric field alone is reasonable given the complexity
2. **Frequency dependence**: magnetospheric response may be frequency-dependent
3. **Nonlinearity**: some data points suggest $E_y^2$ dependence, but linear fit is overall better (Figure 3)

---

## 핵심 시사점 / Key Takeaways

1. **Burton equation은 지자기 폭풍 예보의 "F = ma"이다.** 단 하나의 1차 ODE로 태양풍 → 지자기 폭풍의 전 과정을 기술한다는 것은 놀라운 단순성이다. 이 단순성의 원천은 ring current이 지자기 폭풍의 지배적 원인이라는 물리적 사실과, DPS 관계에 의해 Dst가 ring current 에너지의 직접적 대리변수라는 점이다.
   **The Burton equation is the "F = ma" of geomagnetic storm forecasting.** Describing the entire solar wind → geomagnetic storm process with a single first-order ODE is remarkable simplicity. This simplicity stems from the physical fact that the ring current is the dominant storm driver, and the DPS relation making Dst a direct proxy for ring current energy.

2. **Half-wave rectifier는 Dungey reconnection의 경험적 증거이다.** 남향 IMF일 때만 ring current 주입이 일어난다는 실증적 결과는, Dungey (1961)의 이론적 예측을 15년 만에 정량적으로 확인한 것이다. 이는 단순한 상관관계가 아니라, 물리적 메커니즘(reconnection → 대류 → 입자 주입 → ring current 강화)의 전 과정을 하나의 경험식으로 포착한 것이다.
   **The half-wave rectifier is empirical evidence for Dungey reconnection.** The empirical result that injection occurs only during southward IMF quantitatively confirms Dungey's (1961) theoretical prediction 15 years later. This is not mere correlation but captures the entire physical mechanism chain in one empirical formula.

3. **동압 보정 $Dst^*$의 분리는 물리적 통찰의 승리이다.** 관측된 Dst에서 동압 효과를 분리함으로써, sudden commencement과 ring current 효과가 섞인 복잡한 시계열에서 ring current의 순수 기여만을 추출할 수 있었다. 이 분리 없이는 injection function의 깨끗한 결정이 불가능했을 것이다.
   **Separating pressure-corrected $Dst^*$ is a triumph of physical insight.** By isolating the dynamic pressure effect, the pure ring current contribution was extracted from complex time series mixing SC and ring current. Without this separation, clean determination of the injection function would have been impossible.

4. **단일 감쇠 시간의 한계가 이미 논문 안에 있다.** $\tau \approx 7.7$시간은 "평균"이지만, 강한 폭풍의 초기 회복은 더 빠르고 후기 회복은 더 느리다. 이 관측은 이후 O'Brien & McPherron (2000)의 가변 감쇠 시간 모델과 Liemohn et al. (2001)의 에너지 의존 감쇠 모델로 발전했다.
   **Limitations of a single decay time are already visible in the paper.** $\tau \approx 7.7$ hours is an "average," but early recovery of strong storms is faster and late recovery is slower. This observation later evolved into O'Brien & McPherron (2000)'s variable decay model and Liemohn et al. (2001)'s energy-dependent decay model.

5. **임계 전기장 $E_c = 0.5$ mV/m은 "viscous interaction"의 흔적일 수 있다.** 약한 남향 IMF에서도 주입이 일어나지 않는 이유는, 자기권이 일정 수준 이상의 에너지 투입이 있어야 ring current까지 입자를 수송할 수 있기 때문이다. 이는 magnetotail의 "reservoir" 효과와 관련될 수 있다.
   **The threshold $E_c = 0.5$ mV/m may be a trace of "viscous interaction."** Ring current injection requires energy input above a certain level to transport particles all the way to the ring current region. This may relate to the magnetotail "reservoir" effect.

6. **이 논문은 "space weather forecasting"이라는 분야를 사실상 창시했다.** Burton equation 이전에는 태양풍 데이터로부터 지상 자기 교란을 정량적으로 예측하는 방법이 없었다. 이 방정식은 L1 위성(ACE, DSCOVR)의 실시간 태양풍 데이터로 Dst를 1시간 전에 예측하는 운용 예보 시스템의 원형이 되었다.
   **This paper effectively founded "space weather forecasting" as a discipline.** Before the Burton equation, there was no quantitative method to predict ground magnetic disturbances from solar wind data. This equation became the prototype for operational forecasting systems that predict Dst one hour ahead using real-time L1 data (ACE, DSCOVR).

7. **Explorer 33/35의 hourly 플라즈마 데이터가 한계였다.** 이후 IMP-8, ISEE, Wind, ACE 등의 고시간해상도 태양풍 관측이 Burton equation의 매개변수를 정교화하는 데 기여했다. 특히 1분 해상도의 태양풍 데이터는 injection function의 비선형 특성을 더 명확히 드러냈다.
   **Hourly plasma data from Explorer 33/35 was a limitation.** Later high-cadence solar wind observations (IMP-8, ISEE, Wind, ACE) refined the Burton equation parameters. Especially 1-min resolution data revealed nonlinear features of the injection function more clearly.

---

## 수학적 요약 / Mathematical Summary

### Complete Burton Equation

$$\boxed{\frac{dDst^*}{dt} = Q(E_y) - \frac{Dst^*}{\tau}}$$

### Pressure Correction

$$Dst^* = Dst - b\sqrt{P_{dyn}} + c$$

$$P_{dyn} = \frac{1}{2}m_p n V_{sw}^2 \quad \text{[nPa]}$$

$$b = 15.8 \text{ nT/nPa}^{1/2}, \quad c = 20 \text{ nT}$$

(논문 원본 단위: $b = 0.20$ $\gamma$ (eV cm$^{-3}$)$^{-1/2}$, $P$ in eV/cm$^3$)

### Injection Function (Half-Wave Rectifier)

$$Q(E_y) = \begin{cases} 0 & E_y \leq E_c \\ d(E_y - E_c) & E_y > E_c \end{cases}$$

$$E_y = -V_{sw} B_z \quad \text{[mV/m]}$$

$$d = -1.5 \times 10^{-3} \text{ nT/s per mV/m}$$
$$= -5.4 \text{ nT/hr per mV/m}$$

$$E_c = 0.5 \text{ mV/m}$$

### Decay

$$\tau = \frac{1}{a} = \frac{1}{3.6 \times 10^{-5} \text{ s}^{-1}} = 27{,}778 \text{ s} \approx 7.7 \text{ hr}$$

### Steady-State Solution

에너지 주입이 일정한 경우:

For constant energy injection:

$$Dst^*_{\text{steady}} = Q \cdot \tau$$

예: $E_y = 5$ mV/m일 때 $Q = d(5 - 0.5) = -1.5 \times 10^{-3} \times 4.5 = -6.75 \times 10^{-3}$ nT/s

Example: for $E_y = 5$ mV/m, $Q = d(5 - 0.5) = -6.75 \times 10^{-3}$ nT/s

$$Dst^*_{\text{steady}} = -6.75 \times 10^{-3} \times 27{,}778 \approx -188 \text{ nT}$$

### Analytical Solution (Recovery Phase)

$Q = 0$일 때:

When $Q = 0$:

$$Dst^*(t) = Dst^*_0 \cdot e^{-t/\tau}$$

### General Analytical Solution

$Q = $ const일 때:

When $Q = $ const:

$$Dst^*(t) = Q\tau + (Dst^*_0 - Q\tau)e^{-t/\tau}$$

---

## 역사 속의 논문 / Paper in the Arc of History

```
1940  Chapman & Bartels — Dst/Kp 지수 체계화
  │
1958  Parker — 태양풍 예측
  │
1961  Dungey — Southward IMF → reconnection
  │       "에너지가 어떻게 ring current에 도달?"
  │
1966  Dessler-Parker-Sckopke — Dst ∝ ring current 에너지
  │       수학적 기초 마련
  │
1969  Davis & Sugiura — Hourly Dst index 공식 발표
  │
1973  McPherron, Russell & Aubry — Substorm 3단계 모델
  │       서브스톰 → 입자 주입 → ring current
  │
  ╞══ ★ 1975  Burton, McPherron & Russell ★
  │       Dst = f(Ey, Pdyn) — 최초의 폭풍 예보 방정식
  │       Half-wave rectifier = Dungey 이론의 경험적 확인
  │       단일 감쇠 시간 τ ≈ 7.7 hr
  │
1982  Cowley — 통합 대류 모델 (물리적 기반 보강)
  │
1989  Allen et al. — March 1989 Hydro-Quebec 정전
  │       Burton eq.의 사회적 중요성 부각
  │
1994  Gonzalez et al. — 폭풍 분류 + Burton 기반 기준
  │
2000  O'Brien & McPherron — 가변 τ(Ey) 모델
  │       Burton eq.의 첫 번째 주요 개선
  │
2002  Temerin & Li — 고차 항 추가 (Bz, Ey, Pdyn의 비선형 결합)
  │
2006  Temerin & Li — 최종 개선 모델
  │
2019  Camporeale et al. — ML이 Burton eq.을 대체/보완
```

---

## 다른 논문과의 연결 / Connections to Other Papers

| 논문 / Paper | 관계 / Relationship |
|---|---|
| **#3 Chapman & Bartels (1940)** | Dst 지수를 체계화한 원조. Burton이 여기서 정의된 Dst를 예측 대상으로 사용 |
| **#4 Parker (1958)** | 태양풍의 존재를 예측. Burton equation의 입력인 $V_{sw}$, $n$, $B_z$는 모두 Parker의 태양풍 이론에서 비롯 |
| **#6 Dungey (1961)** | Southward IMF reconnection → half-wave rectifier의 물리적 기초. Burton eq.이 이 이론의 정량적 확인 |
| **#10 McPherron, Russell & Aubry (1973)** | 서브스톰의 3단계 모델. 반복적 서브스톰에 의한 입자 주입이 ring current을 강화시키는 과정의 이해 → Burton의 injection function의 물리적 기반 |
| **→ #12 Cowley (1982)** | Burton의 경험식에 물리적 기반을 제공하는 통합 대류 모델. 왜 $E_y$가 ring current 주입을 구동하는지의 자기권 역학적 설명 |
| **→ #15 Gonzalez et al. (1994)** | Burton equation의 Dst 임계값을 사용하여 폭풍 강도 분류 체계를 확립. Moderate (-50~-100), Intense (-100~-250), Super-storm (<-250) |
| **→ #17 Allen et al. (1989)** | March 1989 폭풍에 Burton equation 적용 시, Dst < -600 nT 수준의 극한 폭풍에서 모델의 한계 노출 |
| **→ #33 Camporeale et al. (2019)** | Burton equation을 LSTM, Random Forest 등의 ML 모델로 대체/보완하는 현대적 접근. 그러나 ML 모델의 해석 가능성에서 Burton eq.이 여전히 우위 |

---

## 참고문헌 / References

- Burton, R.K., McPherron, R.L., and Russell, C.T., "An Empirical Relationship between Interplanetary Conditions and Dst," *J. Geophys. Res.*, Vol. 80, pp. 4204–4214, 1975. [DOI: 10.1029/JA080i031p04204]
- Dungey, J.W., "Interplanetary Magnetic Field and the Auroral Zones," *Physical Review Letters*, Vol. 6, pp. 47–48, 1961.
- Dessler, A.J., and Parker, E.N., "Hydromagnetic Theory of Geomagnetic Storms," *J. Geophys. Res.*, Vol. 64, pp. 2239–2252, 1959.
- Sckopke, N., "A General Relation between the Energy of Trapped Particles and the Disturbance Field near the Earth," *J. Geophys. Res.*, Vol. 71, pp. 3125–3130, 1966.
- Siscoe, G.L., and Crooker, N.U., "On the Partial Ring Current Contribution to Dst," *J. Geophys. Res.*, Vol. 79, pp. 1110–1117, 1974.
- O'Brien, T.P., and McPherron, R.L., "An Empirical Phase Space Analysis of Ring Current Dynamics: Solar Wind Control of Injection and Decay," *J. Geophys. Res.*, Vol. 105, pp. 7707–7719, 2000.
- McPherron, R.L., Russell, C.T., and Aubry, M.P., "Satellite Studies of Magnetospheric Substorms on August 15, 1968: 9. Phenomenological Model for Substorms," *J. Geophys. Res.*, Vol. 78, pp. 3131–3149, 1973.
- Gonzalez, W.D., et al., "What Is a Geomagnetic Storm?," *J. Geophys. Res.*, Vol. 99, pp. 5771–5792, 1994.
