---
title: "Pre-Reading Briefing: Extreme Space-Weather Events and the Solar Cycle"
paper_id: "36_owens_2021"
topic: Space_Weather
date: 2026-04-15
type: briefing
---

# Extreme Space-Weather Events and the Solar Cycle: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Owens, M.J., Lockwood, M., Barnard, L.A., Scott, C.J., Haines, C., Macneil, A., "Extreme Space-Weather Events and the Solar Cycle", *Solar Physics*, 296, 82, 2021. DOI: 10.1007/s11207-021-01831-3
**Author(s)**: Mathew J. Owens, Mike Lockwood, Luke A. Barnard, Chris J. Scott, Carl Haines, Allan Macneil
**Year**: 2021

---

## 1. 핵심 기여 / Core Contribution

이 논문은 150년간의 $aa_H$ 지자기 지수 기록을 사용하여 극한 우주기상 사건(extreme space-weather events)의 발생이 태양 주기(solar cycle)에 의해 어떻게 조절되는지를 체계적으로 분석합니다. 저자들은 네 가지 확률론적 모델(Random, Phase, Phase+Amp, EarlyLate)을 구축하여 세 가지 핵심 질문에 답합니다: (1) 극한 사건이 태양 주기에 의해 정렬되는가? (2) 큰 태양 주기가 더 많은 극한 사건을 생산하는가? (3) 홀수/짝수 주기에서 극한 사건의 시기가 다른가? 세 질문 모두에 대해 '예'라는 답을 통계적으로 입증하며, 이를 Solar Cycle 25의 극한 사건 확률 예측에 적용합니다.

This paper uses the 150-year $aa_H$ geomagnetic index record to systematically analyze how extreme space-weather event occurrence is modulated by the solar cycle. The authors construct four probabilistic models (Random, Phase, Phase+Amp, EarlyLate) to answer three key questions: (1) Are extreme events ordered by the solar cycle? (2) Do bigger cycles produce more extreme events? (3) Do extreme events behave differently in odd and even cycles? They statistically confirm "yes" to all three, and apply the findings to forecast extreme-event probability for Solar Cycle 25.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

우주기상 분야에서 중등도 폭풍(moderate storms)이 태양 주기를 따른다는 것은 오래전부터 알려져 있었습니다. 태양 극대기에 더 빈번하고, 극소기에는 줄어듭니다. 그러나 가장 극한적인 사건(예: 1859년 Carrington event급)에 대해서는 이 관계가 성립하는지 오랜 논쟁이 있었습니다. 극한 사건은 정의상 드물기 때문에 ("data paucity curse") 통계적 검증이 어려웠고, 일부 연구(Kilpua et al., 2015)는 "조용한 태양도 super storm을 발사할 수 있다"고 결론지었습니다.

In space weather, it has long been known that moderate geomagnetic storms follow the solar cycle — more frequent at solar maximum, fewer at minimum. However, whether the most extreme events (e.g., Carrington-class) follow the same pattern has been debated for years. Extreme events are rare by definition (the "data paucity curse"), making statistical verification difficult. Some studies (Kilpua et al., 2015) concluded that "the quieter Sun can also launch superstorms."

2018년에 Lockwood et al.이 $aa_H$ 지수를 개발하여 station 위치 변화와 지구 자기장의 장기 변화를 보정한 균질한 150년 기록을 제공했습니다. 이 새로운 데이터셋이 본 논문의 핵심 도구입니다. 또한 2020년에 Chapman et al.이 폭풍 발생이 bimodal(이중봉)임을 보고한 직후의 연구로, 이를 더 체계적으로 확장합니다.

In 2018, Lockwood et al. developed the $aa_H$ index, providing a homogeneous 150-year record corrected for station location changes and secular changes in Earth's magnetic field. This new dataset is the key tool for this paper. Additionally, Chapman et al. (2020) had just reported bimodal storm occurrence, and this paper systematically extends those findings.

### 타임라인 / Timeline

```
1868         aa index begins (Mayaud)
  |            aa 지수 기록 시작
1852         Sabine: periodical laws of magnetic disturbance
  |            자기 교란의 주기적 법칙
1859         Carrington event (most famous extreme event)
  |            Carrington 사건 (가장 유명한 극한 사건)
2012         July CME near-miss (Baker et al., 2013)
  |            7월 CME 근접 통과
2015         Kilpua et al.: storm occurrence vs. cycle amplitude
  |            폭풍 발생 vs. 주기 진폭
2018         Lockwood et al.: aa_H index developed
  |            aa_H 지수 개발
2020         Chapman et al.: bimodal storm occurrence
  |            이중봉 폭풍 발생 보고
2021  >>>>   Owens et al.: THIS PAPER
  |            본 논문
2025~        Solar Cycle 25 maximum expected
             Solar Cycle 25 극대기 예상
```

---

## 3. 필요한 배경 지식 / Prerequisites

### 지자기 지수 / Geomagnetic Indices
- **$aa$ 지수**: 1868년부터 두 개의 대척점(antipodal) 관측소에서 측정된 3시간 range 지수. 가장 긴 균질한 지자기 활동 기록을 제공합니다.
- **$aa$ index**: A 3-hourly range index measured from two antipodal stations since 1868. Provides the longest homogeneous record of geomagnetic activity.
- **$aa_H$ 지수**: Lockwood et al. (2018)이 개발한 보정 버전. Station 이동과 지구 자기극 이동을 보정하여 더 정확한 장기 기록을 제공합니다.
- **$aa_H$ index**: Corrected version developed by Lockwood et al. (2018). Accounts for station relocations and secular drift of Earth's magnetic poles.

### 태양 주기 / Solar Cycle
- 약 11년 주기의 흑점 수 변동. Sunspot minimum에서 시작하여 maximum까지 증가한 후 감소.
- ~11-year periodicity of sunspot number variation. Starts at sunspot minimum, increases to maximum, then declines.
- **Solar cycle phase**: 0 (minimum/시작) → 1 (다음 minimum/종료)로 선형 정규화.
- **Active phase**: phase 0.18–0.79 (태양 극대기 중심). **Quiet phase**: 나머지 기간.
- **짝수/홀수 주기 (Even/Odd cycles)**: 태양 자기 극성이 22년 Hale 주기를 따르므로, 짝수 주기와 홀수 주기에서 자기장 구조가 다릅니다.

### 확률론적 모델링 / Probabilistic Modeling
- **Monte Carlo 시뮬레이션**: 난수 생성을 반복하여 관측과 모델을 비교하는 통계적 방법.
- **Monte Carlo simulation**: Statistical method comparing observations to models through repeated random sampling.
- **Null hypothesis testing / 귀무가설 검정**: Random model을 귀무가설로 사용하여 관측된 패턴이 우연인지 검정.

### CME와 SIR / CMEs and SIRs
- **CME (Coronal Mass Ejection)**: 대규모 태양 자기장 방출. 가장 극한적인 지자기 폭풍의 원인. 태양 주기를 따름.
- **SIR (Stream Interaction Region)**: 빠른/느린 태양풍의 상호작용 영역. 약한 폭풍의 원인. 태양 주기 하강 단계에서 빈번.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| $aa_H$ index | Lockwood et al. (2018)이 개발한 보정된 지자기 활동 지수. 150년 균질 기록 제공 / Corrected geomagnetic activity index by Lockwood et al. (2018). Provides 150-year homogeneous record |
| Storm-occurrence probability | 특정 $aa_H$ 임계값을 초과하는 일수의 연간 비율 / Fraction of days per year meeting a particular $aa_H$ threshold |
| Solar-cycle phase | 태양 주기 내 위치 (0 = minimum 시작, 1 = 종료) / Position within the solar cycle (0 = start at minimum, 1 = end) |
| Active/Quiet phase | 활동기: phase 0.18–0.79, 정적기: 나머지 / Active: phase 0.18–0.79, Quiet: the rest |
| Percentile thresholds | 90th (37 nT), 99th (77 nT), 99.9th (165 nT), 99.99th (290 nT) — 각각 2주, 3개월, 3년, 25년에 1번 수준 / Correspond to ~1-in-2-weeks, 3-months, 3-years, 25-years events |
| Phase model | 활동기 폭풍 확률이 정적기보다 9배 높다고 가정하는 모델 / Model assuming storm probability is 9× higher in active vs. quiet phase |
| Phase+Amp model | Phase model에 주기 진폭(흑점 수) 의존성을 추가한 모델 / Phase model plus cycle-amplitude (sunspot number) dependence |
| EarlyLate model | Phase+Amp에 홀수/짝수 주기의 초기/후기 활동기 차이를 추가한 모델 / Phase+Amp plus odd/even cycle early/late active-phase asymmetry |
| Hale cycle | 22년 태양 자기 주기. 자기 극성이 11년마다 반전 / 22-year solar magnetic cycle. Polarity reverses every ~11 years |
| Geoeffectiveness | 태양풍 구조가 지자기 폭풍을 유발하는 효율 / Efficiency of a solar wind structure in driving geomagnetic storms |
| Overlying coronal field | CME 위의 대규모 코로나 자기장. CME의 지자기 효과를 증폭시킬 수 있음 / Large-scale coronal magnetic field above CMEs. Can enhance geoeffectiveness |
| Sheath region | CME 전면의 압축된 태양풍 영역. 독자적으로 지자기 효과를 가질 수 있음 / Compressed solar wind region ahead of CME. Can be geoeffective independently |

---

## 5. 수식 미리보기 / Equations Preview

### 5.1 폭풍 정의 / Storm Definition

폭풍은 일평균 $aa_H$ 값에 임계값을 적용하여 정의됩니다:

$$\text{Storm day: } \overline{aa_H} > \text{threshold}$$

네 가지 임계값이 사용됩니다:
- 90th percentile: $aa_H > 37$ nT (N = 5479 storms)
- 99th percentile: $aa_H > 77$ nT (N = 548)
- 99.9th percentile: $aa_H > 165$ nT (N = 55)
- 99.99th percentile: $aa_H > 290$ nT (N = 6)

Storms are defined by applying thresholds to calendar-day means of $aa_H$. Four thresholds are used corresponding to the 90th, 99th, 99.9th, and 99.99th percentiles.

### 5.2 Phase+Amp 모델의 활동기 확률 스케일링 / Phase+Amp Active-Phase Scaling

$$P_{\text{active}} = P_{\text{base}} \times 1.5 \times \frac{A}{\langle SSN \rangle}$$

여기서:
- $A$ = 해당 주기의 평균 흑점 수 / Mean sunspot number for that cycle
- $\langle SSN \rangle$ = 전체 기간(1868–2018) 평균 흑점 수 / Mean sunspot number over the entire record
- 계수 1.5는 관측 데이터 피팅에서 도출 / Factor 1.5 derived from fitting to observations

### 5.3 EarlyLate 모델의 비대칭 보정 / EarlyLate Asymmetry Correction

짝수 주기(even cycles): 초기 활동기 확률을 60% 증가, 후기 활동기 확률을 60% 감소
홀수 주기(odd cycles): 초기 활동기 확률 감소, 후기 활동기 확률 증가

Even cycles: early active-phase probability increased by 60%, late active-phase decreased by 60%.
Odd cycles: early active-phase probability decreased, late active-phase increased.

### 5.4 상관계수와 귀무가설 검정 / Correlation and Null Hypothesis

주기 진폭과 폭풍 발생 확률 사이의 선형 상관계수 $r$과 귀무가설 기각 확률 $p$:
- 99th percentile: $r = 0.93$, $p = 0.0000$ (매우 강한 상관)
- 99.99th percentile: $r = 0.63$, $N = 14$ (여전히 유의미, $p < 0.05$)

Linear correlation coefficient $r$ and null-hypothesis rejection probability $p$ between cycle amplitude and storm-occurrence probability.

### 5.5 홀수/짝수 주기 차이의 우연 확률 / Chance Probability of Odd/Even Difference

$$p = 0.5^6 = 0.016$$

99.9th percentile의 55개 사건 중, 짝수 주기의 3개 사건은 모두 초기 활동기에, 홀수 주기의 3개 사건은 모두 후기 활동기에 발생. 이것이 우연일 확률은 1.6%.

For the 55 events at the 99.9th percentile, even-cycle events cluster in the early active phase while odd-cycle events cluster in the late active phase. The probability of this occurring by chance is 1.6%.

---

## 6. 읽기 가이드 / Reading Guide

### 추천 읽기 순서 / Recommended Reading Order

1. **Abstract + Section 1 (Introduction)** — 논문의 세 가지 핵심 질문을 파악하세요.
   Identify the three key questions the paper addresses.

2. **Section 2 (Background)** — 이전 연구들이 왜 "data paucity curse"에 부딪혔는지 이해하세요.
   Understand why previous studies hit the "data paucity curse."

3. **Section 3 (Data and Storm Selection)** — $aa_H$ 데이터와 폭풍 정의 방법을 확인하세요. Figure 1이 핵심입니다.
   Check the $aa_H$ data and storm definition. Figure 1 is key.

4. **Section 4 (Modelling Storm Occurrence)** — 네 가지 모델의 구조를 이해하세요. Figure 3이 핵심입니다.
   Understand the four model structures. Figure 3 is essential.

5. **Section 5 (Are Extreme Events Ordered by the Solar Cycle?)** — 첫 번째 질문에 대한 답. Figure 4에 집중하세요.
   Answer to Q1. Focus on Figure 4.

6. **Section 6 (Do Bigger Cycles Produce More Extreme Events?)** — 두 번째 질문. Figures 5-6에 집중.
   Answer to Q2. Focus on Figures 5-6.

7. **Section 7 (Do Extreme Events Behave Differently in Odd and Even Cycles?)** — 가장 흥미로운 결과. Figure 7에 집중.
   Most interesting result. Focus on Figure 7.

8. **Section 8 (Solar Cycle 25)** — 실용적 응용. Figure 8의 세 가지 시나리오를 비교하세요.
   Practical application. Compare three scenarios in Figure 8.

9. **Section 10 (Discussion)** — Figure 9의 Hale cycle 해석이 핵심. 왜 홀수/짝수 주기가 다른지 물리적으로 설명합니다.
   Figure 9's Hale cycle interpretation is key. Physical explanation for odd/even differences.

### 특히 주의할 부분 / Pay Special Attention To

- **Figure 2**: Superposed epoch analysis — 짝수/홀수 주기에서 폭풍 발생 시기가 정말 다른지 직접 눈으로 확인하세요.
  Visually confirm whether storm occurrence timing truly differs between even/odd cycles.
- **Figure 4 하단 패널**: Active-Quiet 차이 — Random model은 2σ 수준에서 기각됨을 확인하세요.
  Bottom panel shows Active-Quiet difference — confirm Random model is rejected at 2σ.
- **Figure 9**: Hale 주기와 CME 극성의 관계 다이어그램 — 이 논문에서 가장 물리적인 통찰입니다.
  Hale cycle and CME polarity relationship diagram — the most physical insight in this paper.

---

## 7. 현대적 의의 / Modern Significance

### Solar Cycle 25에 대한 직접적 예측 / Direct Prediction for Solar Cycle 25

이 논문은 Solar Cycle 25의 극한 사건 확률을 구체적으로 예측합니다. SC25는 홀수 주기이므로 극한 사건이 활동기 후반에 집중될 것으로 예상됩니다 (2026년 초 이후). 현재(2026년) SC25 극대기에 접근하고 있어 이 예측의 검증 시기에 있습니다.

This paper specifically forecasts extreme-event probability for Solar Cycle 25. Since SC25 is odd-numbered, extreme events are expected to concentrate in the late active phase (after early 2026). We are currently (2026) approaching SC25 maximum, making this a timely test of the predictions.

### 인프라 보호 계획 / Infrastructure Protection Planning

극한 우주기상 사건의 확률을 태양 주기 크기와 시기로 예측할 수 있다는 것은 전력망, 위성, GPS 등의 장기 계획에 중요한 도구를 제공합니다. 큰 주기 vs. 작은 주기에서 1-in-100-years 사건의 확률이 약 3배 차이남.

Being able to forecast extreme event probability from solar cycle magnitude and timing provides an important tool for long-term planning of power grids, satellites, GPS, etc. The probability of a 1-in-100-year event differs by a factor of ~3 between large and small cycles.

### 방법론적 기여 / Methodological Contribution

Monte Carlo 기반의 확률론적 모델 비교 접근법은 적은 표본 크기에서도 통계적 유의성을 검증하는 강력한 방법을 제시합니다. "data paucity curse"를 극복하는 전략으로서 다른 우주기상 연구에도 적용 가능합니다.

The Monte Carlo-based probabilistic model comparison approach demonstrates a powerful method for testing statistical significance even with small sample sizes. As a strategy for overcoming the "data paucity curse," it is applicable to other space weather studies.

---

## Q&A

### Q1: 왜 std 기반이 아니라 percentile 기반 임계값인가? / Why percentile-based thresholds instead of std-based?

$aa_H$ 분포가 정규분포가 아닌 극도의 right-skewed 분포이기 때문입니다. 이런 분포에서 mean ± std는 극한 사건의 빈도를 제대로 반영하지 못합니다. Percentile 기반은 직관적 재현 주기에 대응되며 (90th → 2주, 99th → 3개월, 99.9th → 3년, 99.99th → 25년), 다양한 강도 수준에서 태양 주기 의존성의 변화를 추적할 수 있습니다. 극한값 통계학에서도 std 기반 접근은 부적절한 것으로 알려져 있습니다 (Embrechts and Schmidli, 1994).

The $aa_H$ distribution is extremely right-skewed, not Gaussian. Mean ± std poorly captures extreme-event frequency in such distributions. Percentile-based thresholds map to intuitive recurrence times and allow tracking how solar-cycle dependence changes across intensity levels. Std-based approaches are known to be inappropriate in extreme-value statistics.

### Q2: $aa$ 지수의 정의와 다른 지자기 지수와의 관계 / Definition of $aa$ index and relation to other geomagnetic indices

$aa$는 1868년 Mayaud가 도입한 3시간 range 지수입니다. 대척점(antipodal) 두 관측소(영국 Hartland, 호주 Canberra)에서 수평 자기장의 최대-최소 변동폭을 측정합니다. 국소 K 지수(준로그 0-9)를 선형 nT로 변환 후 평균. $aa_H$는 Lockwood et al. (2018)이 관측소 이전과 지자기극 이동을 보정한 버전입니다. Dst(환전류 강도), Kp(13개소 전구 지수), AE(극광 전류)와 보완적 관계이며, $aa$의 최대 강점은 150년 균질 기록의 길이입니다.

The $aa$ index is a 3-hourly range index from two antipodal stations (UK, Australia), measuring max-min horizontal field variation. $aa_H$ adds corrections for station relocations and geomagnetic pole drift. Its key advantage over Dst, Kp, AE is the 150-year homogeneous record length.

### Q3: 임계값 선택에 따른 결과 민감도 / Sensitivity of results to threshold choice

저자들은 이를 인지하고 의도적으로 넓은 범위를 스캔합니다. 실제로 임계값에 따라 결과가 질적으로 달라지는 것이 핵심 발견입니다: 낮은 임계값(90-99th)에서는 CME+SIR 혼합으로 단순 Phase model이 충분하고, 높은 임계값(99.9-99.99th)에서는 CME 지배로 EarlyLate model이 필요합니다. Figure 6에서 상관계수 $r$이 99th 근처에서 피크 후 감소하지만, 이는 표본 크기 감소 때문이지 실제 관계 약화가 아님을 Phase+Amp model이 재현합니다. 단일 임계값 대신 전체 범위를 스캔하는 것이 방법론적 강점입니다.

The authors intentionally scan a wide range. Results qualitatively change with threshold — a key finding itself: low thresholds (CME+SIR mix) need only the Phase model, high thresholds (CME-dominated) require the EarlyLate model. The decline in correlation at extreme thresholds is reproduced by the Phase+Amp model as a sample-size effect, not a weakening relationship.

### Q4: 지자기 지수 종합 비교 / Comprehensive Comparison of Geomagnetic Indices

#### K 지수 (K index) — 기본 단위 / The Basic Unit

단일 관측소에서 3시간 동안 수평 자기장의 최대-최소 변동폭(range)을 측정하는 준로그(quasi-logarithmic) 정수 스케일 (0–9). 자기장 교란 범위가 수 nT ~ 수백 nT로 넓기 때문에 로그 스케일을 사용합니다. 관측소 위도에 따라 K=9에 해당하는 nT 값이 다릅니다.

A quasi-logarithmic integer scale (0–9) measuring the max-min range of horizontal magnetic field at a single station over 3 hours. Log scale is used because disturbance ranges span orders of magnitude. The nT value for K=9 varies by station latitude.

```
K =  0   1    2    3    4    5    6     7     8     9
     |    |    |    |    |    |    |     |     |     |
     0   5   10   20   40   70  120   200   330   500  nT  (중위도 기준 / mid-latitude)
```

#### Kp (planetary K) — 전구적 활동 등급 / Global Activity Rating

- **도입 / Introduced**: Julius Bartels, 1949
- **관측소 / Stations**: 13개 중위도 관측소 (전구 분포 / globally distributed)
- **스케일 / Scale**: 0, 0+, 1-, 1, 1+, ... 9 (28단계, 준로그 / 28 steps, quasi-logarithmic)
- **시간 해상도 / Resolution**: 3시간
- **계산 / Computation**: 각 관측소 K → 위도 보정 Ks → 13개 평균
- **NOAA G-scale**: Kp5 = G1 (Minor), Kp6 = G2, Kp7 = G3, Kp8 = G4, Kp9 = G5 (Extreme)
- **용도 / Use**: 가장 널리 사용되는 실시간 우주기상 지표 / Most widely used real-time space weather indicator

#### ap (planetary amplitude) — Kp의 선형 변환 / Linear Transform of Kp

Kp를 선형 nT 스케일로 변환하여 산술 연산(평균, 합계)을 가능하게 합니다.

Converts Kp to linear nT scale, enabling arithmetic operations (averaging, summation).

```
Kp:  0   1   2   3   4   5   6   7   8   9
ap:  0   3   7  15  27  48  80 140 240 400  (nT)
```

#### Ap (daily planetary amplitude) — 일간 요약 / Daily Summary

하루 8개 ap 값의 산술 평균: $Ap = \frac{1}{8}\sum_{i=1}^{8} ap_i$. 장기 통계에 편리합니다.

Arithmetic mean of eight 3-hourly ap values per day. Convenient for long-term statistics.

#### aa (antipodal amplitude) — 최장 기록 / Longest Record

- **도입 / Introduced**: Pierre-Noël Mayaud, 1971 (1868년까지 소급 / backdated to 1868)
- **관측소 / Stations**: 단 2개, 대척점 배치 (영국 + 호주 / UK + Australia, antipodal)
- **스케일 / Scale**: 선형 nT
- **계산 / Computation**: $aa = (a_{\text{UK}} + a_{\text{Australia}}) / 2$
- **핵심 장점 / Key advantage**: 1868년부터 150+ 년 균질 기록 — 극한 사건 통계에 필수적 / 150+ year homogeneous record, essential for extreme-event statistics
- **2개소만 사용하는 이유 / Why only 2 stations**: 적은 수의 관측소가 장기 균질성 유지에 유리. 대척점 배치로 지역 편향을 상쇄 / Fewer stations easier to maintain homogeneously. Antipodal placement cancels local bias

#### $aa_H$ (homogeneous aa) — 보정 버전 / Corrected Version

Lockwood et al. (2018)이 aa에 두 가지 보정을 추가: (1) 관측소 이전 시 감도 변화 보정 (영국: Greenwich → Abinger → Hartland), (2) 지구 자기극 장기 이동에 따른 관측 편향 제거. **본 논문에서 사용하는 지수입니다.**

Lockwood et al. (2018) added two corrections to aa: (1) sensitivity changes during station relocations, (2) secular drift of Earth's magnetic poles. **This is the index used in this paper.**

#### 기타 주요 지수 / Other Key Indices

| 지수 / Index | 측정 대상 / Measures | 관측소 / Stations | 시간 해상도 / Resolution | 기록 시작 / Since |
|---|---|---|---|---|
| **Dst** | 환전류 강도 / Ring current intensity | 저위도 4개소 / 4 low-lat | 1시간 / 1 hour | 1957 |
| **SYM-H** | Dst 고해상도 버전 / High-res Dst | 저위도 6개소 / 6 low-lat | 1분 / 1 min | 1981 |
| **AE** (AU/AL) | 극광대 전류 / Auroral electrojet | 고위도 12개소 / 12 high-lat | 1분 / 1 min | 1957 |
| **PC** | 극관 대류 / Polar cap convection | 북극+남극 각 1개소 | 1분 / 1 min | 1975 |

#### 계보도 / Family Tree

```
                     K 계열 (range 기반)              독립 지수
                          │                      (다른 측정 원리)
          ┌───────────────┼───────────┐
          │               │           │
     국소 K 지수       Kp (전구적)   aa (대척점)     Dst    AE
     (1개 관측소)      (13개소)     (2개소)       (환전류) (극광대)
                          │           │
                       ap (선형)   aa_H (보정)
                          │
                       Ap (일평균)

    준로그 스케일   ←→   선형 nT 스케일
```

본 논문에서 $aa_H$를 선택한 이유: **150년 균질 기록 + 선형 스케일 + 보정된 정확성** — 극한 사건 통계에 최적의 지수입니다.

The reason $aa_H$ was chosen for this paper: **150-year homogeneous record + linear scale + corrected accuracy** — the optimal index for extreme-event statistics.
