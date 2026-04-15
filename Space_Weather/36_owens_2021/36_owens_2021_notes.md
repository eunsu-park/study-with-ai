---
title: "Extreme Space-Weather Events and the Solar Cycle"
authors: Mathew J. Owens, Mike Lockwood, Luke A. Barnard, Chris J. Scott, Carl Haines, Allan Macneil
year: 2021
journal: "Solar Physics"
doi: "10.1007/s11207-021-01831-3"
topic: Space_Weather
tags: [extreme-events, solar-cycle, aa-index, geomagnetic-storms, probabilistic-modeling, Monte-Carlo, Hale-cycle]
status: completed
date_started: 2026-04-15
date_completed: 2026-04-15
---

# 36. Extreme Space-Weather Events and the Solar Cycle / 극한 우주기상 사건과 태양 주기

---

## 1. Core Contribution / 핵심 기여

이 논문은 150년간의 보정된 지자기 지수 $aa_H$ 기록을 사용하여 극한 우주기상 사건의 발생이 태양 주기에 의해 체계적으로 조절됨을 통계적으로 입증합니다. 저자들은 네 가지 확률론적 모델(Random, Phase, Phase+Amp, EarlyLate)을 구축하고 Monte Carlo 시뮬레이션과 비교하여 세 가지 핵심 질문에 답합니다: (1) 극한 사건이 태양 주기의 활동기/정적기에 의해 정렬되는가? (2) 큰 태양 주기(높은 흑점 수)가 더 많은 극한 사건을 생산하는가? (3) 홀수/짝수 번호 주기에서 극한 사건의 시기가 다른가? 세 질문 모두에 대해 통계적으로 유의미한 긍정적 답을 제시하며, 특히 세 번째 발견은 22년 Hale 자기 주기와 CME 자기장 극성의 관계로 물리적으로 해석합니다. 이 결과들을 결합하여 Solar Cycle 25의 극한 사건 확률을 세 가지 시나리오(소형/중형/대형 주기)로 예측합니다.

This paper uses the 150-year corrected geomagnetic index $aa_H$ record to statistically demonstrate that extreme space-weather event occurrence is systematically modulated by the solar cycle. The authors construct four probabilistic models (Random, Phase, Phase+Amp, EarlyLate) and compare them with Monte Carlo simulations to answer three key questions: (1) Are extreme events ordered by the active/quiet phases of the solar cycle? (2) Do larger solar cycles (higher sunspot numbers) produce more extreme events? (3) Do extreme events behave differently in odd- and even-numbered cycles? They provide statistically significant affirmative answers to all three, with the third finding interpreted physically through the 22-year Hale magnetic cycle and CME magnetic-field polarity. Combining these results, they forecast extreme-event probability for Solar Cycle 25 under three scenarios (small/moderate/large cycle).

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction (Section 1, pp.1–2) / 서론

논문은 우주기상(space weather)의 정의에서 출발합니다 — 지구 근처 우주 환경에서 플라즈마, 자기장, 고에너지 입자의 변동성을 총칭하며, 특히 지자기 폭풍(geomagnetic storms)에 초점을 맞춥니다. 지자기 폭풍은 대규모 태양풍 구조(주로 CME)가 지구 근처에 도달하여 발생합니다.

The paper begins by defining space weather — the variability of plasma, magnetic fields, and energetic particles in the near-Earth space environment — focusing specifically on geomagnetic storms triggered by arrival of large-scale solar-wind structures (primarily CMEs).

저자들은 세 가지 시간 스케일의 예측을 구분합니다:
1. **단기 예보 (short-term forecast)**: 개별 사건의 시기 예측 (1–4일 리드 타임)
2. **기후학 (climatology)**: 장기적 사건 확률 — 본 논문의 초점
3. **중기 확률론적 예보 (medium-term probabilistic)**: 태양 주기 내에서의 확률 변동

The authors distinguish three prediction timescales: (1) short-term event forecasting (1–4 day lead), (2) climatology — long-term event probability (this paper's focus), (3) medium-term probabilistic forecasting — probability variation within the solar cycle.

핵심 문제는 "data paucity curse"입니다 — 1-in-100-years 사건을 통계적으로 정의하려면 수백 년의 데이터가 필요하지만, 균질한 지자기 기록은 약 150년밖에 없습니다. 극한값 통계(extreme-value statistics)가 대안이지만, 이는 분포의 꼬리 부분에서 설정된 경향이 가장 극한적인 사건에도 적용된다고 가정합니다.

The key challenge is the "data paucity curse" — defining a 1-in-100-year event statistically requires several hundred years of data, but homogeneous geomagnetic records span only ~150 years. Extreme-value statistics offer an alternative but assume that trends established at lower intensities extend to the most extreme events.

### Part II: Background (Section 2, pp.3–4) / 배경

태양 주기에 따른 우주기상 사건의 변동은 오래전부터 알려져 있었습니다 — Dalton (1834)의 극광 기록과 Sabine (1852)의 지자기 교란 주기성 발견까지 거슬러 올라갑니다. 중등도 폭풍의 경우 빈도 분석만으로 경험적 경향을 확립할 수 있지만, 극한 사건으로 갈수록 사건 수가 급감하여 통계적 패턴 확립이 어려워집니다.

The solar-cycle variation of space-weather events has been known since Dalton (1834) and Sabine (1852). For moderate storms, simple frequency analysis establishes empirical trends, but as events become more extreme, the number of events drops sharply, making statistical pattern establishment increasingly difficult.

이전 연구들의 결론이 엇갈리는 상황을 정리합니다:
- **Kilpua et al. (2015)**: 약한 폭풍은 하강기에, 강한 폭풍은 극대기 근처에 집중. 주기 크기와 폭풍 발생의 상관은 높은 임계값에서 약해짐 → "조용한 태양도 superstorm을 발사할 수 있다"
- **Vennerstrom et al. (2016)**: "폭풍은 태양 주기의 모든 단계에서 발생"하지만 정량화되지 않음
- **Chapman et al. (2020)**: 폭풍 발생이 bimodal (활동기에 높은 확률, 정적기에 매우 낮은 확률). 큰 폭풍이 높은 흑점 수 시기에 발생하는 경향

Previous studies reached conflicting conclusions: Kilpua et al. (2015) found correlation between cycle size and storm occurrence declining at high thresholds; Vennerstrom et al. (2016) noted storms occur in all cycle phases; Chapman et al. (2020) found bimodal storm occurrence and a tendency for larger storms during higher sunspot numbers.

핵심 쟁점: 이러한 불일치가 실제 물리적 차이를 반영하는 것인지, 아니면 단순히 적은 표본 크기에서 오는 통계적 잡음인지?

The key issue: do these disagreements reflect genuine physical differences, or simply statistical noise from small sample sizes?

### Part III: Data and Storm Selection (Section 3, pp.4–5) / 데이터 및 폭풍 선택

**태양 주기 정의**: 월별 흑점 수(monthly sunspot number)를 사용하여 주기 시작(태양 극소기)과 종료를 정의합니다. 시작점은 흑점의 평균 위도가 불연속적으로 변하는 시점으로 식별합니다 (Owens et al., 2011). Solar Cycle 25는 2020년 말에 시작된 것으로 가정합니다.

**Solar cycle definition**: Monthly sunspot numbers define cycle start (solar minimum) and end. Start is identified by the discontinuous change in average sunspot latitude (Owens et al., 2011). Solar Cycle 25 is assumed to have begun at the end of 2020.

**태양 주기 위상(phase)**: 0 (극소기 시작) → 1 (다음 극소기 종료)로 선형 정규화. 흑점 수 변동은 phase ~0.35에서 피크를 보이며 비대칭적 감소(extended decay)를 나타냅니다.

**Solar cycle phase**: Linearly normalized from 0 (minimum start) to 1 (next minimum end). Sunspot number peaks at phase ~0.35 with asymmetric (extended) decay.

**$aa_H$ 지수**: Mayaud (1975)의 $aa$ 지수를 Lockwood et al. (2018a,b)이 보정한 버전. 개별 관측소 데이터에서 재구축하여 관측소 이동과 지자기극의 장기 이동을 보정했습니다. 일별(daily), 27일, 연간 평균이 Figure 1에 표시됩니다.

**$aa_H$ index**: Corrected version of Mayaud (1975)'s $aa$ index by Lockwood et al. (2018a,b). Rebuilt from individual station data, correcting for station relocations and long-term geomagnetic pole motion.

**폭풍 정의**: 저자들은 의도적으로 단순한 정의를 채택합니다 — 일평균 $aa_H$에 임계값을 적용. 이는 복잡한 정의(시작/종료 시간 설정)와 비교했을 때, 극한 사건에 대해 거의 동일한 결과를 냅니다. 예: 99.99th percentile의 상위 6개 폭풍은 Lockwood et al. (2019)의 상위 6개 사건과 정확히 동일합니다.

**Storm definition**: The authors intentionally adopt a simple definition — thresholds applied to calendar-day means of $aa_H$. For the most extreme events, this produces nearly identical results to more complex definitions. E.g., the top 6 storms by the 99.99th percentile are exactly the same as Lockwood et al. (2019)'s top 6 events.

일평균 사용의 장점: (1) 태양 기원의 지속적 태양풍 구동 활동에 초점 (vs. 내부 자기권 프로세스에 의존하는 단시간 substorm), (2) UT 변동의 반랜덤 효과를 억제.

Advantages of daily means: (1) focuses on sustained solar-wind-driven activity (vs. brief substorms from internal magnetospheric processes), (2) suppresses semi-random UT variation effects.

**네 가지 임계값** (1868–2018 전체 기간의 percentile):

| Percentile | $aa_H$ (nT) | 폭풍 수 / N storms | 재현 주기 / Recurrence |
|---|---|---|---|
| 90th | 37 | 5,479 | ~2주 / ~2 weeks |
| 99th | 77 | 548 | ~3개월 / ~3 months |
| 99.9th | 165 | 55 | ~3년 / ~3 years |
| 99.99th | 290 | 6 | ~25년 / ~25 years |

### Part IV: Modelling Storm Occurrence (Section 4, pp.7–8) / 폭풍 발생 모델링

관측된 경향의 통계적 유의성을 검증하기 위해 네 가지 확률 모델을 구축합니다. 각 모델은 이전 모델에 하나의 물리적 요소를 추가하는 계층적 구조입니다:

Four probability models are constructed to test statistical significance, each adding one physical element to the previous in a hierarchical structure:

**① Random model (귀무가설 / null hypothesis)**
- 폭풍이 완전히 랜덤하게 발생한다고 가정
- 상대 확률이 모든 시점에서 동일 (Figure 3 파란선)
- Assumes storms occur completely randomly with equal probability at all times

**② Phase model**
- 활동기(phase 0.18–0.79)의 폭풍 확률이 정적기보다 9배 높다고 가정
- 활동기/정적기 비율 9:1은 135 nT 임계값에서 관측과 일치하도록 설정 (100개 사건 확보)
- Assumes storm probability is 9× higher during active phase (0.18–0.79) than quiet phase

**③ Phase+Amp model**
- Phase model에 주기 진폭 의존성 추가
- 활동기 확률을 주기 진폭에 비례하여 스케일링: $1.5 \times A / \langle SSN \rangle$
- $A$ = 해당 주기의 평균 흑점 수, $\langle SSN \rangle$ = 전체 기간 평균
- Adds cycle-amplitude dependence: active-phase probability scaled by $1.5 \times A / \langle SSN \rangle$

**④ EarlyLate model**
- Phase+Amp에 홀수/짝수 주기의 초기/후기 활동기 비대칭 추가
- 짝수 주기: 초기 활동기 확률 +60%, 후기 -60%
- 홀수 주기: 초기 -60%, 후기 +60%
- 60% 값은 99.9th percentile (55개 사건)의 관측에서 설정
- Adds odd/even early/late asymmetry: even cycles +60% early / -60% late; odd cycles reversed

**Monte Carlo 방법**: 각 모델에서 누적 확률 함수(CDF)를 구성하고, 난수를 생성하여 폭풍 시간을 할당합니다. 예: 99.9th percentile의 55개 사건이면 55번 반복. 이 과정을 5000회 반복하여 중앙값과 1σ/2σ 범위를 구합니다.

**Monte Carlo method**: Construct cumulative probability functions (CDFs) for each model, then generate storm times via random number sampling. For the 99.9th percentile with 55 events, sample 55 times. Repeat 5000 iterations to obtain median and 1σ/2σ ranges.

### Part V: Are Extreme Events Ordered by the Solar Cycle? (Section 5, pp.8–10) / 극한 사건이 태양 주기에 의해 정렬되는가?

**Figure 4 분석** — 핵심 결과 그림:
- **상단 (정적기)**: 관측된 폭풍 발생 확률(검은 점)이 Random model과 Phase model 사이에 위치. 정적기에도 폭풍이 발생하지만, Random model이 과대추정.
- **중간 (활동기)**: 관측 값이 Phase model과 잘 일치. Random model은 과소추정.
- **하단 (활동기-정적기 차이)**: Random model은 **2σ 수준에서 기각**. 즉 "폭풍 발생이 태양 주기와 무관하다"는 귀무가설은 모든 임계값에서 기각됩니다.

**Figure 4 analysis** — key result figure:
- **Top (quiet phase)**: Observed storm probability (black dots) lies between Random and Phase models. Storms occur in quiet phase but Random model overestimates.
- **Middle (active phase)**: Observations agree well with Phase model. Random model underestimates.
- **Bottom (active-quiet difference)**: Random model is **rejected at the 2σ level**. The null hypothesis of random storm occurrence is rejected for all storm thresholds.

99th percentile 이하에서는 관측이 Phase model에서 Random model 방향으로 벗어나는 경향이 있습니다. 이는 낮은 임계값에서 SIR(Stream Interaction Region) 기원 폭풍이 포함되기 때문으로 해석됩니다 — SIR은 태양 주기 하강기/극소기에도 빈번하여 태양 주기 의존성을 "희석"시킵니다. 반면 99th percentile 이상의 극한 사건은 거의 순수하게 CME 기원이며, Phase model과 잘 일치합니다.

Below the 99th percentile, observations deviate from the Phase model toward the Random model. This is interpreted as inclusion of SIR-driven storms at lower thresholds — SIRs are frequent during the declining phase/minimum, "diluting" solar-cycle dependence. Above the 99th percentile, extreme events are almost purely CME-driven and agree well with the Phase model.

### Part VI: Do Bigger Cycles Produce More Extreme Events? (Section 6, pp.10–11) / 큰 주기가 더 많은 극한 사건을 생산하는가?

**Figure 5** — 주기당 평균 폭풍 발생 확률 vs. 주기 진폭(평균 흑점 수)의 산점도:
- 90th percentile: $r = 0.80$, $p = 0.0005$ (강한 상관)
- 99th percentile: $r = 0.93$, $p = 0.0000$ (매우 강한 상관)
- 99.9th percentile: $r = 0.53$, $p = 0.0491$ (유의하나 약함)
- 99.99th percentile: $r = 0.63$, $p = 0.0162$ (유의미)

모든 임계값에서 귀무가설(상관 없음)이 95% 신뢰수준(2σ)에서 기각됩니다. 그러나 99.99th percentile에서는 6개 사건만 있고, 9개 주기가 폭풍이 0개이므로 단일 사건이 상관을 크게 변경할 수 있습니다.

For all thresholds, the null hypothesis of zero correlation is rejected at the 95% (2σ) confidence level. However, at the 99.99th percentile with only 6 events and 9 cycles with zero storms, a single event could significantly change the correlation.

**Figure 6** — 상관계수 $r$을 $aa_H$ 임계값의 함수로 표시:
- 피크 상관은 99th percentile 근처 ($r \approx 0.9$)
- 더 극한적인 임계값에서 $r$이 감소
- **핵심**: 이 감소가 실제 관계 약화인지, 표본 크기 감소의 효과인지?

Phase+Amp model이 이 $r$ 감소 패턴을 **정확히 재현**합니다 — 즉, 주기 진폭과 폭풍 발생 사이의 근본적 관계에 변화가 없어도, 표본 크기 감소만으로 관측된 $r$ 감소를 설명할 수 있습니다. 이것은 Kilpua et al. (2015)의 "상관이 약해진다"는 결론이 통계적 환상(statistical artifact)임을 시사합니다.

The Phase+Amp model **exactly reproduces** this $r$-decline pattern — meaning the observed decline can be explained purely by reduced sample size, without any change in the underlying relationship. This suggests Kilpua et al. (2015)'s conclusion of "weakening correlation" is a statistical artifact.

**Phase+Amp model의 스케일링**: 최적 피팅에서 활동기 확률을 $1.5 \times A / \langle SSN \rangle$ 으로 스케일링. 이는 가장 크고 가장 작은 주기 사이에 폭풍 발생 확률의 약 3배 차이를 만듭니다.

Phase+Amp scaling: Best fit scales active-phase probability by $1.5 \times A / \langle SSN \rangle$, producing ~3× difference in storm probability between the largest and smallest cycles.

### Part VII: Do Extreme Events Behave Differently in Odd and Even Cycles? (Section 7, pp.11–12) / 홀수/짝수 주기에서 극한 사건이 다르게 행동하는가?

**Figure 7** — 가장 흥미로운 결과. 활동기를 초기(early)와 후기(late)로 나누고, 짝수/홀수 주기를 분리:

- **낮은 임계값 (90th–99th)**: 초기/후기 차이 없음
- **99.9th percentile 이상**: 명확한 차이 출현
  - **짝수 주기**: 극한 사건이 초기 활동기에 집중
  - **홀수 주기**: 극한 사건이 후기 활동기에 집중

**Figure 7** — the most interesting result. Splitting the active phase into early and late halves, separated by odd/even cycles:
- Low thresholds (90th–99th): no early/late difference
- Above 99.9th percentile: clear difference emerges — even cycles cluster early, odd cycles cluster late

99.9th percentile (55 사건): 짝수 주기의 3개 사건은 모두 초기 활동기, 홀수 주기의 3개 사건은 모두 후기 활동기. 각 사건이 초기/후기에 동등한 확률로 발생한다면:

$$p = 0.5^6 = 0.016$$

즉 이것이 우연일 확률은 1.6%로, 통계적으로 유의미합니다.

The probability of this pattern occurring by chance is $0.5^6 = 0.016$ (1.6%), statistically significant.

EarlyLate model에서 60% 보정을 적용하면 관측과 일치하며, Phase+Amp model (초기/후기 차이 없음)은 귀무가설로서 기각됩니다.

### Part VIII: Solar Cycle 25 Predictions (Section 8, pp.12–13) / Solar Cycle 25 예측

세 가지 규칙을 결합하여 SC25의 폭풍 확률을 예측합니다:
1. **태양 주기 위상** (활동기/정적기)
2. **주기 진폭** (흑점 수 크기)
3. **홀수/짝수 비대칭** (SC25는 홀수 → 후기 활동기에 극한 사건 집중)

Three rules combined to forecast SC25 storm probability: (1) solar-cycle phase, (2) cycle amplitude, (3) odd/even asymmetry (SC25 is odd → extreme events concentrate in late active phase).

**Figure 8** — 세 가지 시나리오:
- **소형 주기 (SC12급)**: 빨간 실선. 극한 사건 확률 낮음
- **중형 주기 (SC23급)**: 검은 파선. 중간 확률
- **대형 주기 (SC19급)**: 파란선. 극한 사건 확률 높음

99th percentile ($aa_H > 77$ nT) 폭풍의 경우, 태양 주기 위상과 진폭 규칙만 필요합니다. 주기 진폭에 따라 확률이 약 3배 차이.

99.99th percentile ($aa_H > 290$ nT) 폭풍의 경우, 홀수/짝수 규칙도 추가로 필요합니다. SC25가 홀수이므로 모든 시나리오에서 극한 사건이 후기 활동기(2026년 초 이후 예상)에 피크를 보입니다.

For 99.99th percentile storms, the odd/even rule is also needed. Since SC25 is odd, all scenarios show extreme events peaking in the late active phase (expected after early 2026).

정량적 예측: 대형 주기(SC19급)의 경우, 향후 11년간 99.99th percentile 폭풍이 최소 1번 발생할 통합 확률은 약 54%. 소형 주기(SC12급)에서는 약 24%로 감소.

Quantitative forecast: For a large cycle (SC19-class), integrated probability of at least one 99.99th percentile storm over the next 11 years is ~54%. For a small cycle (SC12-class), drops to ~24%.

### Part IX: Summary and Discussion (Sections 9–10, pp.13–17) / 요약 및 논의

**요약 (Section 9)**:
1. 귀무가설(랜덤 발생) 기각: 활동기의 폭풍 확률이 정적기보다 약 9배 높음. 99.99th percentile (25년 1번 수준)까지 유효.
2. 주기 진폭과 폭풍 발생의 상관: 1-in-100-years 사건 확률이 주기 크기에 따라 약 3배 차이.
3. 홀수/짝수 비대칭: 짝수 주기에서는 초기 활동기, 홀수 주기에서는 후기 활동기에 극한 사건 집중. 99.99th percentile에서 유의미.

**물리적 해석 (Section 10)** — Figure 9이 핵심:

태양 주기 위상과 진폭 의존성은 직관적입니다 — 활동기에 흑점 수와 활동 영역이 증가하면 CME 발생률이 높아지고, 더 큰 주기에서는 이 효과가 증폭됩니다.

The solar-cycle phase and amplitude dependence is intuitive — more sunspots/active regions during active phase means more CMEs, amplified in larger cycles.

홀수/짝수 비대칭은 더 복잡하고 흥미롭습니다. 저자들은 여러 가능성을 검토합니다:

1. **Hale의 법칙과 CME flux-rope 극성**: CME의 flux-rope 극성은 흑점의 극성 법칙(Hale's law)을 따르며, 홀수와 짝수 주기에서 반전됩니다 (Bothmer and Rust, 1997). 그러나 다른 극성의 CME의 geoeffectiveness 차이는 크지 않다는 연구도 있어 (Fenrich and Luhmann, 1998), 이것만으로 충분한 설명이 되지 않습니다.

2. **극성 주기 (Polarity cycles)**: 태양 극대기에서 극대기로 이어지는 기간 동안 태양의 대규모 쌍극자 자기장이 특정 방향을 유지합니다. $qA > 0$ 주기 (양의 극성이 북쪽)에서는 대규모 코로나 자기장이 적도 부근에서 남향으로 정렬되어 geoeffective합니다. 저자들은 이 overlying field가 CME를 직접 강화하기보다, 이미 강한 CME 사건을 "극한" 범주로 밀어올리는 역할을 한다고 제안합니다.

3. **Sheath 영역의 강화**: overlying field가 CME 앞의 sheath 영역에서 추가적인 flux-rope나 자기장 구조로 작용하여 geoeffectiveness를 높일 수 있습니다.

The odd/even asymmetry is more complex. The authors propose that polarity cycles ($qA > 0$ vs. $qA < 0$) control the large-scale coronal magnetic field orientation, which can enhance geoeffectiveness of already-severe events into the extreme category. Enhanced heliospheric magnetic-field strengths have been reported in $qA > 0$ cycles.

**흥미로운 관찰**: 1859년 Carrington event와 2012년 7월 STEREO-A CME — 가장 유명한 두 극한 사건이 모두 짝수 주기(SC10, SC24)에서 발생했으며, 둘 다 초기 활동기(phase 0.30, 0.29)에 있었습니다. 이는 본 논문의 EarlyLate model과 정확히 일치합니다 (비록 $aa_H$ 데이터에는 포함되지 않지만).

Notable observation: The 1859 Carrington event and July 2012 STEREO-A CME — two of the most famous extreme events — both occurred in even-numbered cycles (SC10, SC24) during the early active phase (phase 0.30, 0.29), exactly consistent with the EarlyLate model.

---

## 3. Key Takeaways / 핵심 시사점

1. **극한 사건도 태양 주기를 따른다** — 활동기의 폭풍 발생 확률이 정적기보다 약 9배 높으며, 이는 99.99th percentile (25년 1번 수준)까지 유효합니다. "조용한 태양에서도 극한 사건이 발생한다"는 이전 결론은 통계적 표본 부족의 환상이었습니다.
   Extreme events follow the solar cycle — active-phase probability is ~9× higher than quiet phase, valid up to the 99.99th percentile. Previous conclusions about extreme events during quiet Sun were statistical artifacts from small samples.

2. **큰 주기 = 더 많은 극한 사건** — 주기 진폭과 폭풍 발생의 상관계수가 99th percentile에서 $r = 0.93$에 달하며, 가장 큰 주기와 가장 작은 주기 사이에 1-in-100-years 사건 확률이 약 3배 차이납니다.
   Bigger cycles produce more extreme events — correlation at the 99th percentile reaches $r = 0.93$, with ~3× probability difference between largest and smallest cycles.

3. **홀수/짝수 주기의 극한 사건 시기가 다르다** — 짝수 주기는 초기 활동기, 홀수 주기는 후기 활동기에 집중. 이 패턴은 99.9th percentile 이상에서만 나타나며, 우연 확률 1.6%로 통계적으로 유의합니다.
   Extreme events cluster at different times in odd vs. even cycles — even cycles favor the early active phase, odd cycles the late active phase. Chance probability is 1.6%.

4. **"Data paucity curse"는 극복 가능하다** — 적은 표본에서도 확률론적 모델과 Monte Carlo 비교를 통해 유의미한 결론을 도출할 수 있습니다. 핵심은 단일 임계값이 아닌 전체 범위를 스캔하는 것입니다.
   The "data paucity curse" can be overcome — probabilistic models with Monte Carlo comparison yield meaningful conclusions even from small samples. The key is scanning the full threshold range.

5. **SIR vs. CME 기원의 분리가 자연스럽게 나타난다** — 낮은 임계값에서의 Phase model 편차는 SIR 기여를, 높은 임계값에서의 순수한 Phase model 일치는 CME 지배를 반영합니다. 단일 분석 프레임워크에서 두 폭풍 기원의 다른 태양 주기 의존성을 드러냅니다.
   SIR vs. CME origins separate naturally — Phase model deviations at low thresholds reflect SIR contributions, while pure Phase model agreement at high thresholds reflects CME dominance.

6. **Hale 주기(22년)가 극한 사건에 중요하다** — 11년 Schwabe 주기뿐 아니라 22년 자기 주기가 대규모 코로나 자기장 구조를 통해 극한 사건의 geoeffectiveness에 영향을 미칩니다. 이는 overlying coronal field와 sheath region의 자기장 강화로 해석됩니다.
   The 22-year Hale cycle matters for extreme events — beyond the 11-year Schwabe cycle, the magnetic polarity cycle influences extreme-event geoeffectiveness through overlying coronal field structures and sheath region enhancement.

7. **Solar Cycle 25에 대한 실용적 예측** — SC25는 홀수 주기이므로 극한 사건이 후기 활동기(2026년 초 이후)에 집중될 것으로 예상됩니다. 주기 크기에 따라 99.99th percentile 사건의 11년 통합 확률이 24%(소형)~54%(대형)입니다.
   Practical SC25 forecast — as an odd-numbered cycle, extreme events should concentrate in the late active phase (after early 2026). Integrated 11-year probability of a 99.99th percentile event ranges from 24% (small cycle) to 54% (large cycle).

8. **폭풍 정의의 단순성이 강점이 될 수 있다** — 일평균 $aa_H$ 임계값이라는 단순한 정의가 극한 사건에서는 복잡한 정의와 동일한 결과를 내며, Monte Carlo 시뮬레이션에서의 대량 생성에도 적합합니다.
   Simplicity in storm definition can be a strength — daily mean $aa_H$ thresholds produce identical results to complex definitions for extreme events and are well-suited for mass generation in Monte Carlo simulations.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 폭풍 정의 / Storm Definition

일평균 $aa_H$에 임계값 적용:

$$\text{Storm day if: } \frac{1}{8}\sum_{i=1}^{8} aa_{H,i} > \text{threshold}$$

여기서 $aa_{H,i}$는 하루 8개의 3시간 $aa_H$ 값입니다.

Where $aa_{H,i}$ are the eight 3-hourly $aa_H$ values per day.

임계값은 1868–2018 전체 분포의 percentile로 설정:
- 90th: 37 nT, 99th: 77 nT, 99.9th: 165 nT, 99.99th: 290 nT

### 4.2 태양 주기 위상 / Solar Cycle Phase

$$\phi = \frac{t - t_{\min,\text{start}}}{t_{\min,\text{end}} - t_{\min,\text{start}}}$$

여기서 $t_{\min}$은 연속된 두 태양 극소기의 시점. $\phi \in [0, 1]$.

Where $t_{\min}$ are times of consecutive solar minima. $\phi \in [0, 1]$.

- 활동기 / Active phase: $0.18 \leq \phi \leq 0.79$
- 정적기 / Quiet phase: $\phi < 0.18$ 또는 $\phi > 0.79$

### 4.3 확률 모델 / Probability Models

**Random model**:

$$P_{\text{Random}}(t) = \frac{1}{T}$$

여기서 $T$는 전체 기간의 길이 (일수). 모든 시점에서 동일한 확률.

Where $T$ is total period length (days). Equal probability at all times.

**Phase model**:

$$P_{\text{Phase}}(t) = \begin{cases} R \cdot P_0 & \text{if } 0.18 \leq \phi(t) \leq 0.79 \text{ (active)} \\ P_0 & \text{if otherwise (quiet)} \end{cases}$$

여기서 $R = 9$ (활동기/정적기 확률 비율), $P_0$는 전체 확률이 1이 되도록 정규화.

Where $R = 9$ (active/quiet probability ratio), $P_0$ normalized so total probability equals 1.

**Phase+Amp model**:

$$P_{\text{Phase+Amp}}(t) = \begin{cases} R \cdot \left(1.5 \cdot \frac{A_c}{\langle SSN \rangle}\right) \cdot P_0 & \text{if active} \\ P_0 & \text{if quiet} \end{cases}$$

여기서:
- $A_c$ = 주기 $c$의 평균 흑점 수 / Mean sunspot number for cycle $c$
- $\langle SSN \rangle$ = 전체 기간(1868–2018) 평균 흑점 수 / Mean sunspot number over entire record
- 계수 1.5는 관측 피팅에서 결정 / Factor 1.5 determined from observation fitting

**EarlyLate model**:

짝수 주기 / Even cycles:
$$P_{\text{early}} = P_{\text{Phase+Amp}} \times 1.6, \quad P_{\text{late}} = P_{\text{Phase+Amp}} \times 0.4$$

홀수 주기 / Odd cycles:
$$P_{\text{early}} = P_{\text{Phase+Amp}} \times 0.4, \quad P_{\text{late}} = P_{\text{Phase+Amp}} \times 1.6$$

60% 보정 (×1.6 / ×0.4)은 99.9th percentile (55개 사건) 관측에서 설정.

60% correction (×1.6 / ×0.4) set from 99.9th percentile (55 events) observations.

### 4.4 Monte Carlo 시뮬레이션 / Monte Carlo Simulation

1. 모델의 상대 확률 $P(t)$에서 누적 분포 함수(CDF) $C(t)$ 구성:

$$C(t) = \sum_{t'=0}^{t} P(t')$$

2. 균일 난수 $u \sim U(0,1)$ 생성
3. $C(t_{\text{storm}})$이 $u$에 가장 가까운 시점 $t_{\text{storm}}$을 폭풍 시점으로 할당
4. 동일한 날에 두 폭풍이 배정되지 않도록 확인
5. N개 폭풍에 대해 반복 (N은 관측된 폭풍 수)
6. 전체 과정을 5000회 반복하여 통계 분포 구성

Steps: (1) Build CDF from model probability, (2) Draw uniform random number, (3) Assign storm time at closest CDF value, (4) Check no duplicate days, (5) Repeat for N storms, (6) Repeat 5000 times for statistics.

### 4.5 상관 분석 / Correlation Analysis

주기별 평균 폭풍 발생 확률 $\bar{P}_c$와 주기 진폭 $A_c$ (평균 흑점 수)의 선형 상관:

$$r = \frac{\sum_c (A_c - \bar{A})(\bar{P}_c - \bar{\bar{P}})}{\sqrt{\sum_c (A_c - \bar{A})^2 \sum_c (\bar{P}_c - \bar{\bar{P}})^2}}$$

| Percentile | $r$ (관측) | $p$ | N (주기) |
|---|---|---|---|
| 90th | 0.80 | 0.0005 | 14 |
| 99th | 0.93 | 0.0000 | 14 |
| 99.9th | 0.53 | 0.0491 | 14 |
| 99.99th | 0.63 | 0.0162 | 14 |

### 4.6 홀수/짝수 비대칭 유의성 / Odd/Even Asymmetry Significance

99.9th percentile에서 짝수 주기 3개 사건 모두 초기, 홀수 주기 3개 사건 모두 후기에 발생할 우연 확률:

$$p = 0.5^3 \times 0.5^3 = 0.5^6 = 0.016$$

각 사건이 독립적으로 초기/후기에 50:50 확률로 발생한다는 귀무가설 하에서.

Under the null hypothesis that each event independently has 50:50 probability of occurring in early vs. late active phase.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1834  Dalton: auroral record periodicity
       극광 기록의 주기성
  |
1852  Sabine: periodical laws of magnetic disturbance
       자기 교란의 주기적 법칙
  |
1859  Carrington event (even cycle SC10, early active phase)
       Carrington 사건 (짝수 주기, 초기 활동기)
  |
1868  aa index begins (Mayaud)
       aa 지수 기록 시작
  |
1940  Chapman & Bartels: Geomagnetism (Dst, Kp 체계화)
       지자기학 (지수 체계화)
  |
1949  Bartels: Kp index introduced
       Kp 지수 도입
  |
1994  Embrechts & Schmidli: extreme-value statistics
       극한값 통계학
  |
2002  Richardson, Cane & Cliver: geomagnetic activity over 3 cycles
       3주기 지자기 활동
  |
2013  Baker et al.: July 2012 near-miss extreme CME
       2012년 7월 극한 CME 근접 통과
  |
2013  Cliver & Dietrich: Carrington event revisited
       Carrington 사건 재방문
  |
2015  Kilpua et al.: storm occurrence vs. cycle characteristics
       폭풍 발생 vs. 주기 특성
  |
2016  Vennerstrom et al.: extreme storms 1868-2010
       극한 폭풍 1868-2010
  |
2018  Lockwood et al.: aa_H index developed
       aa_H 지수 개발
  |
2020  Chapman et al.: bimodal storm occurrence & cycle modulation
       이중봉 폭풍 발생 & 주기 조절
  |
2021  >>>>  Owens et al.: THIS PAPER
       극한 사건의 태양 주기 의존성 통계적 입증
  |
2025~ Solar Cycle 25 maximum — testing predictions
       SC25 극대기 — 예측 검증
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Chapman & Bartels (1940) — SW #3 | $aa$, Dst, Kp 지수 체계의 기초 / Foundation of geomagnetic index systems | 본 논문이 사용하는 $aa_H$의 원류. 150년 기록의 출발점 / Origin of $aa_H$ used in this paper. Starting point of the 150-year record |
| Burton, McPherron & Russell (1975) — SW #11 | Dst와 태양풍 매개변수의 경험적 관계 / Empirical relation between Dst and solar wind | 폭풍 강도 예보의 시초. 본 논문은 이를 확률론적 프레임으로 확장 / Pioneer of storm intensity forecasting. This paper extends to probabilistic framework |
| Gonzalez et al. (1994) — SW #15 | 지자기 폭풍의 정의와 분류 / Definition and classification of geomagnetic storms | 본 논문의 "폭풍"이 무엇인지에 대한 기초 개념 제공. Dst 임계값 기반 분류 / Provides foundational concept of what a "storm" is |
| Kilpua et al. (2015) | 폭풍 발생과 주기 특성의 상관 감소 보고 / Reports declining correlation between storm occurrence and cycle characteristics | 본 논문이 직접 반박 — 상관 감소는 표본 크기 효과이지 실제 관계 약화가 아님 / Directly refuted — correlation decline is a sample-size effect, not a weakening relationship |
| Lockwood et al. (2018a,b) | $aa_H$ 지수 개발 / Development of $aa_H$ index | 본 논문의 핵심 데이터 소스. 보정된 균질 기록이 없었다면 이 분석은 불가능 / Core data source. Analysis impossible without corrected homogeneous record |
| Chapman, Horne & Watkins (2020) | 이중봉 폭풍 발생과 태양 주기 조절 보고 / Reports bimodal storm occurrence and solar cycle modulation | 본 논문의 직접적 선행 연구. 여기서 보고된 bimodal 패턴을 확장하고 물리적으로 해석 / Direct predecessor. Extends and physically interprets the bimodal pattern |
| Baker et al. (2013) — SW #29 | 2012년 7월 극한 CME 분석 / Analysis of July 2012 extreme CME | 짝수 주기(SC24) 초기 활동기(phase 0.29)에 발생 — EarlyLate model과 일치하는 독립적 사례 / Even cycle, early active phase — independent case consistent with EarlyLate model |
| Bothmer & Rust (1997) | CME flux-rope 극성과 Hale 법칙의 관계 / CME flux-rope polarity and Hale's law | Section 10의 물리적 해석의 기초. CME 극성이 홀수/짝수 주기에서 반전됨 / Foundation for Section 10's physical interpretation |

---

## 7. References / 참고문헌

- Owens, M.J., Lockwood, M., Barnard, L.A., Scott, C.J., Haines, C., Macneil, A., "Extreme Space-Weather Events and the Solar Cycle", *Solar Physics*, 296, 82, 2021. [DOI: 10.1007/s11207-021-01831-3]
- Lockwood, M., Chambodut, A., Barnard, L.A., Owens, M.J., Clarke, E., Mendel, V., "A homogeneous aa index: 1. Secular variation", *J. Space Weather Space Clim.*, 8, A53, 2018a. [DOI]
- Lockwood, M., Finch, I.D., Chambodut, A., Barnard, L.A., Owens, M.J., Clarke, E., "A homogeneous aa index: 2. Hemispheric asymmetries and the equinoctial variation", *J. Space Weather Space Clim.*, 8, A58, 2018b. [DOI]
- Chapman, S.C., Horne, R.B., Watkins, N.W., "Using the index over the last 14 solar cycles to characterize extreme geomagnetic activity", *Geophys. Res. Lett.*, 47, e2019GL086524, 2020. [DOI]
- Chapman, S.C., McIntosh, S.W., Leamon, R.J., Watkins, N.W., "Quantifying the solar cycle modulation of extreme space weather", *Geophys. Res. Lett.*, 47, e2020GL087795, 2020. [DOI]
- Kilpua, E.K.J., Olspert, N., Grigorievskiy, A., et al., "Statistical study of strong and extreme geomagnetic disturbances and solar cycle characteristics", *Astrophys. J.*, 806, 272, 2015. [DOI]
- Vennerstrom, S., Lefevre, L., Dumbovic, M., et al., "Extreme geomagnetic storms - 1868-2010", *Solar Phys.*, 291, 1447, 2016. [DOI]
- Mayaud, P.-N., "Analysis of storm sudden commencements for the years 1868–1967", *J. Geophys. Res.*, 80, 111, 1975. [DOI]
- Mayaud, P.-N., "Derivation, Meaning, and Use of Geomagnetic Indices", *Geophys. Monogr. Ser.*, 22, Am. Geophys. Union, Washington, 1980.
- Baker, D.N., et al., "A major solar eruptive event in July 2012: defining extreme space weather scenarios", *Space Weather*, 11, 585, 2013. [DOI]
- Bothmer, V., Rust, D.M., "The field configuration of magnetic clouds and the solar cycle", In: Crooker, Joselyn, Feynmann (eds.) *Geophys. Mono. Ser.*, 99, Am. Geophys. Union, 1997.
- Embrechts, P., Schmidli, H., "Modelling of extremal events in insurance and finance", *Z. Oper.-Res.*, 39, 1, 1994. [DOI]
- Cliver, E.W., Dietrich, W.F., "The 1859 space weather event revisited: limits of extreme activity", *J. Space Weather Space Clim.*, 3, A31, 2013. [DOI]
- Haines, C., Owens, M.J., Barnard, L., Lockwood, M., Ruffenach, A., "The variation of geomagnetic storm duration with intensity", *Solar Phys.*, 294, 154, 2019. [DOI]
- Richardson, I.G., Cane, H.V., Cliver, E.W., "Sources of geomagnetic activity during nearly three solar cycles (1972–2000)", *J. Geophys. Res.*, 107, 1187, 2002. [DOI]
