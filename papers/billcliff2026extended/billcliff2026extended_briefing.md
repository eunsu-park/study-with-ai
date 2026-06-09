---
title: "Pre-Reading Briefing: Extended Lead-Time Geomagnetic Storm Forecasting With Solar Wind Ensembles and Machine Learning"
paper_id: "37_billcliff_2026"
topic: Space_Weather
date: 2026-04-15
type: briefing
---

# Extended Lead-Time Geomagnetic Storm Forecasting With Solar Wind Ensembles and Machine Learning: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Billcliff, M., Smith, A. W., Owens, M., Woo, W. L., Barnard, L., Edward-Inatimi, N., & Rae, I. J. (2026). *Space Weather*, 24, e2025SW004823.
**Author(s)**: M. Billcliff, A. W. Smith, M. Owens, W. L. Woo, L. Barnard, N. Edward-Inatimi, I. J. Rae
**Year**: 2026

---

## 1. 핵심 기여 / Core Contribution

이 논문은 태양풍 앙상블 모델링과 머신러닝을 결합하여 지자기 폭풍 예보의 리드 타임을 기존 30–90분에서 최대 24시간 이상으로 확장한 연구입니다. 핵심 아이디어는 (1) MAS 3D MHD 모델의 Carrington map 출력에서 위도 섭동을 통해 100개의 태양풍 속도 프로파일 앙상블을 추출하고, (2) 각각을 1D HUXt 모델로 지구까지 전파한 뒤, (3) 각 앙상블 멤버에 대해 개별 로지스틱 회귀 분류기로 폭풍 확률을 예측하고, (4) OMNI 데이터와의 오차(MAE)로 가중 평균하여 최종 확률적 예보를 산출하는 것입니다. Hp30 ≥ 5 기준으로 6시간 리드 타임에서 ROC AUC 0.751, BSS_clim 0.595를 달성했습니다.

This paper extends geomagnetic storm forecast lead time from the current 30–90 minutes to up to 24+ hours by combining solar wind ensemble modeling with machine learning. The key idea is: (1) extract 100 solar wind velocity profile ensembles from MAS 3D MHD Carrington maps via latitudinal perturbations, (2) propagate each through the 1D HUXt model to Earth, (3) train individual logistic regression classifiers per ensemble member for storm probability, and (4) aggregate via MAE-weighted mean using OMNI data comparison. At Hp30 ≥ 5 threshold with 6-hr lead time, the model achieves ROC AUC of 0.751 and BSS_clim of 0.595.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

현재 운용 중인 지자기 폭풍 예보 시스템은 주로 L1 지점(태양-지구 사이 약 150만 km)의 위성 데이터에 의존합니다. 태양풍이 L1에서 지구까지 도달하는 데 약 30–90분이 걸리므로, 이것이 현재 예보의 최대 리드 타임입니다. 1989년 Hydro-Quebec 정전 사태 이후 더 긴 리드 타임의 필요성이 절실해졌으며, 특히 전력망 운용자들은 변압기 보호 조치를 위해 최소 수 시간의 사전 경고가 필요합니다.

Current operational geomagnetic storm forecasting relies primarily on L1 satellite data (~1.5 million km from Earth). Since solar wind takes ~30–90 min from L1 to Earth, this sets the maximum current forecast lead time. Since the 1989 Hydro-Quebec blackout, longer lead times have been urgently needed — power grid operators require at least several hours of advance warning for transformer protection.

태양 근처(21.5 R☉)에서 태양풍 조건을 모델링하면 리드 타임을 1–3일까지 확장할 수 있지만, 이 접근법은 공간적 불확실성이라는 근본적 문제를 안고 있습니다. 이 논문은 앙상블 기법으로 이 불확실성을 체계적으로 다루면서도 계산 효율을 유지하는 방법을 제시합니다.

Modeling solar wind conditions near the Sun (21.5 R☉) can extend lead times to 1–3 days, but this approach faces fundamental spatial uncertainty. This paper addresses this uncertainty systematically through ensemble methods while maintaining computational efficiency.

### 타임라인 / Timeline

```
1975  Burton et al. — 최초의 Dst 경험적 예보 공식 / First empirical Dst forecasting equation
2000  Arge & Pizzo — WSA 모델 (운용 태양풍 예보의 기초) / WSA model (foundation of operational SW forecasting)
2003  Odstrcil — ENLIL 3D MHD 모델 / ENLIL 3D MHD model
2017  Owens & Riley — Carrington map 섭동 앙상블 기법 제안 / Proposed Carrington map perturbation ensemble
2018  Tan et al. — LSTM으로 Kp 예보 / Kp forecasting with LSTM
2020  M. Owens et al. — HUXt 모델 공개 (1D reduced-physics) / HUXt model released
2020  Chakraborty & Morley — 확률적 Kp 예보 / Probabilistic Kp prediction
2022  Barnard & Owens — HUXt 앙상블 보정 / HUXt ensemble calibration
2022  Bernoux et al. — EUV 기반 2-7일 지자기 활동 예보 / EUV-based 2-7 day geomagnetic forecasting
2024  Edward-Inatimi et al. — 앙상블 보정 기법 확장 / Extended ensemble calibration techniques
2026  ★ Billcliff et al. — 본 논문: 앙상블 + ML 결합으로 24hr+ 확률적 폭풍 예보 / This paper
```

---

## 3. 필요한 배경 지식 / Prerequisites

### 지자기 지수 / Geomagnetic Indices
- **Kp 지수**: 3시간 간격, 0–9 범위 (9에서 상한). 전통적 지자기 활동 지표.
  **Kp index**: 3-hourly, range 0–9 (capped at 9). Traditional geomagnetic activity measure.
- **Hp30 지수**: 30분 간격, 상한 없음 (open-ended). Kp와 동일한 분포이지만 더 높은 시간 해상도와 극한 폭풍 구별 가능.
  **Hp30 index**: 30-min cadence, open-ended (no cap). Same distribution as Kp but with higher time resolution and ability to distinguish extreme storms.
- **NOAA G-scale**: G1(minor, Kp=5) ~ G5(extreme, Kp=9). 이 논문에서 폭풍 정의: Hp30_MAX ≥ 4.66 (G1 이상).
  **NOAA G-scale**: G1(minor, Kp=5) to G5(extreme, Kp=9). Storm definition in this paper: Hp30_MAX ≥ 4.66 (G1 or above).

### 태양풍 모델링 / Solar Wind Modeling
- **Carrington map**: 한 Carrington 회전(~27.3일) 동안의 태양 표면 관측을 하나의 2D 지도로 합성한 것.
  **Carrington map**: Solar surface observations over one Carrington rotation (~27.3 days) stitched into a single 2D map.
- **MAS (Magnetohydrodynamic Algorithm outside a Sphere)**: 태양 코로나의 3D MHD 시뮬레이션. 1 R☉ ~ 21.5 R☉까지의 태양풍 속도 맵을 출력.
  **MAS**: 3D MHD simulation of the solar corona. Outputs solar wind velocity maps from 1 R☉ to 21.5 R☉.
- **HUXt (Heliospheric Upwind eXtrapolation with time dependency)**: 1D reduced-physics 태양풍 전파 모델. 21.5 R☉에서 지구(1 AU)까지 태양풍을 전파. 빠른 계산 속도가 장점 (100개 앙상블 9분, Apple M2).
  **HUXt**: 1D reduced-physics solar wind propagation model. Propagates solar wind from 21.5 R☉ to Earth (1 AU). Key advantage: fast computation (100 ensembles in 9 min on Apple M2).

### 머신러닝 / Machine Learning
- **로지스틱 회귀 (Logistic Regression)**: 이진 분류용 통계 모델. 이 논문에서 각 앙상블 멤버에 대해 개별 분류기를 훈련.
  **Logistic Regression**: Statistical model for binary classification. This paper trains individual classifiers per ensemble member.
- **앙상블 가중 평균 (Weighted Mean)**: MAE 기반으로 각 분류기의 출력에 가중치를 부여하여 최종 확률을 산출.
  **Weighted Mean**: Assigns weights to each classifier output based on MAE to produce final probability.
- **ROC AUC**: 분류기의 판별 능력 측정. 0.5 = 무작위, 1.0 = 완벽.
  **ROC AUC**: Measures discriminative ability. 0.5 = random, 1.0 = perfect.
- **Brier Skill Score (BSS)**: 확률적 예보의 정확도와 신뢰도. BSS_clim > 0은 기후학 예보보다 우수, ≥ 0.2는 의미 있는 스킬.
  **BSS**: Probabilistic forecast accuracy and reliability. BSS_clim > 0 beats climatology, ≥ 0.2 is meaningful skill.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **Hp30** | 30분 해상도의 지자기 활동 지수. Kp와 동일한 스케일이지만 상한이 없음. / 30-min resolution geomagnetic activity index. Same scale as Kp but with no upper cap. |
| **Carrington Rotation** | 태양이 지구에서 보아 한 바퀴 자전하는 데 걸리는 기간 (~27.28일). / Period for one apparent solar rotation as seen from Earth (~27.28 days). |
| **MAS** | 태양 코로나의 3D MHD 수치 모델. Carrington map으로부터 21.5 R☉에서의 태양풍 속도를 계산. / 3D MHD numerical model of the solar corona. Computes solar wind velocity at 21.5 R☉ from Carrington maps. |
| **HUXt** | 1D reduced-physics 태양풍 전파 모델. 계산이 매우 빠르며 대규모 앙상블에 적합. / 1D reduced-physics solar wind propagation model. Extremely fast computation, suitable for large ensembles. |
| **Ensemble perturbation** | MAS Carrington map에서 지구 경로의 위도를 사인파로 섭동하여 다양한 태양풍 프로파일을 추출하는 기법. / Technique of perturbing Earth's heliolatitude path on MAS Carrington maps sinusoidally to extract diverse solar wind profiles. |
| **OMNI** | 지구 근처의 태양풍 속성, IMF, 지자기 지수의 처리된 데이터 모음. 1963년부터 제공. / Processed collection of near-Earth solar wind properties, IMF, and geomagnetic indices. Available from 1963. |
| **v − OMNI** | 앙상블 멤버의 예측 태양풍 속도와 관측된 OMNI 속도의 차이. 분류기의 입력 피처로 사용. / Difference between ensemble member's predicted solar wind velocity and observed OMNI velocity. Used as classifier input feature. |
| **MAE-weighted mean** | 각 앙상블 멤버의 OMNI 대비 MAE에 반비례하는 가중치로 최종 확률을 산출. 정확한 멤버에 더 큰 가중치 부여. / Final probability computed with weights inversely proportional to each ensemble member's MAE vs. OMNI. Gives more weight to accurate members. |
| **BSS_clim** | 기후학 예보(균형 데이터셋에서 확률 0.5) 대비 Brier Skill Score. 양수값은 기후학보다 우수한 예보. / Brier Skill Score relative to climatology forecast (probability 0.5 on balanced dataset). Positive values indicate skill above climatology. |
| **Lead time (Lt)** | 예보를 발행하는 시점과 예보 대상 기간(forecast window) 시작 사이의 시간 간격. / Time gap between when the forecast is issued and the start of the forecast window. |
| **Forecast window** | 24시간 고정 기간. 이 기간 내에 Hp30_MAX ≥ threshold이면 "폭풍", 아니면 "비폭풍"으로 분류. / Fixed 24-hr period. Classified as "Storm" if Hp30_MAX ≥ threshold within this window, otherwise "Non-Storm." |
| **27-day recurrence** | 태양 자전 주기(~27일)에 기반한 베이스라인 예보. 27일 전과 동일한 조건이 반복된다고 가정. / Baseline forecast based on solar rotation period (~27 days). Assumes conditions repeat from 27 days prior. |

---

## 5. 수식 미리보기 / Equations Preview

### 수식 1: 앙상블 섭동 경로 / Ensemble Perturbation Path

$$\theta(\phi) = \theta_E + \theta_{MAX} \sin(\phi + \phi_0)$$

- $\theta$: 섭동된 경로의 태양 위도 / heliolatitude of the perturbed path
- $\phi$: Carrington 경도 / Carrington longitude
- $\theta_E$: 섭동되지 않은 지구의 태양 위도 / unperturbed Earth heliolatitude
- $\theta_{MAX}$: 최대 섭동 진폭 ($\sigma = 7.5°$인 정규분포에서 추출) / maximum perturbation amplitude (drawn from normal distribution with $\sigma = 7.5°$)
- $\phi_0$: 위상 ($[0, 2\pi]$의 균일분포에서 추출) / phase (drawn from uniform distribution on $[0, 2\pi]$)

이 사인파 섭동으로 지구 경로를 다양하게 변화시켜 Carrington map에서 서로 다른 태양풍 속도 프로파일을 추출합니다. / This sinusoidal perturbation varies Earth's path to extract different solar wind velocity profiles from the Carrington map.

### 수식 2: 로지스틱 함수 / Logistic Function

$$P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_1 + \beta_2 X_2 + \cdots + \beta_n X_n)}}$$

- $P(Y=1|X)$: 주어진 입력에서 폭풍 발생 확률 / probability of storm occurrence given inputs
- $\beta_0$: 절편 / intercept
- $\beta_1, \ldots, \beta_n$: 예측 변수 계수 / predictor variable coefficients
- $X_1, \ldots, X_n$: 입력 피처 (v, Δv, v−OMNI, Hp30) / input features

입력 피처: (a) 예측 태양풍 속도 v, (b) v의 기울기 Δv, (c) v − OMNI 차이, (d) 입력 윈도우 내 Hp30 데이터. / Input features: (a) predicted solar wind velocity v, (b) gradient Δv, (c) v − OMNI difference, (d) Hp30 data in input window.

### 수식 3–5: MAE 가중 평균 / MAE-Weighted Mean

$$w_j = \frac{1}{MAE_j^2}$$

$$w_j^{norm} = \frac{w_j}{\sum_j w_j}$$

$$\hat{y} = \sum_j w_j^{norm} \cdot p_j$$

- $w_j$: 앙상블 멤버 $j$의 가중치 (MAE의 역제곱에 비례) / weight for ensemble member $j$ (inversely proportional to MAE squared)
- $MAE_j$: 멤버 $j$의 HUXt 출력과 OMNI 간의 평균 절대 오차 / Mean Absolute Error between member $j$'s HUXt output and OMNI
- $p_j$: 멤버 $j$의 분류기가 출력한 폭풍 확률 / storm probability from member $j$'s classifier
- $\hat{y}$: 최종 가중 확률적 예보 / final weighted probabilistic forecast

MAE가 작은(더 정확한) 앙상블 멤버에 더 큰 가중치를 부여하여 최종 예보의 품질을 높입니다. / Members with smaller MAE (more accurate) receive larger weights, improving final forecast quality.

### 수식 6–8: 평가 메트릭 / Evaluation Metrics

$$\text{TPR} = \frac{TP}{TP + FN}, \quad \text{FPR} = \frac{FP}{FP + TN}$$

$$BS = \frac{1}{N}\sum_{i=1}^{N}(p_i - o_i)^2$$

$$BSS_{clim} = 1 - \frac{BS}{BS_{clim}}$$

- $BS$: Brier Score — 확률 예보($p_i$)와 실제 관측($o_i \in \{0,1\}$)의 평균 제곱 오차 / mean squared error between forecast probability and observed outcome
- $BSS_{clim}$: 기후학 대비 스킬. 양수 = 기후학보다 우수, 0.2 이상 = 의미 있는 부가가치. / Skill relative to climatology. Positive = better than climatology, ≥ 0.2 = meaningful added value.

---

## 6. 읽기 가이드 / Reading Guide

### 읽기 순서 권장 / Recommended Reading Order

1. **Abstract + Plain Language Summary** (p.1): 전체 그림을 먼저 파악하세요. "24-hr window", "probabilistic forecast"이 핵심 키워드.
   Get the big picture first. "24-hr window" and "probabilistic forecast" are key phrases.

2. **Section 2 (Data)** — 특히 2.1 (p.2-3): Kp vs Hp30의 차이를 이해하는 것이 중요합니다. Figure 1의 2003 Halloween Storm 비교를 주의 깊게 보세요.
   Understanding Kp vs Hp30 differences is crucial. Study Figure 1's Halloween Storm comparison carefully.

3. **Figure 2 (p.5)**: 전체 모델 파이프라인의 핵심 개요도. Model A → B → C(i) → C(ii)의 흐름을 이해하면 나머지가 쉬워집니다.
   The key schematic of the entire model pipeline. Understanding the A → B → C(i) → C(ii) flow makes everything else easier.

4. **Section 3 (Models)** (p.4-8): 각 모델 컴포넌트를 Figure 2와 대조하며 읽으세요. 3.1.1 (앙상블 추출)과 3.3.2 (가중 평균)가 이 논문의 가장 독창적인 부분입니다.
   Read each model component cross-referencing Figure 2. Sections 3.1.1 (ensemble extraction) and 3.3.2 (weighted mean) are the most original contributions.

5. **Section 3.4 (Forecast Setup)** (p.8): Table 1과 함께 입력 윈도우 vs 예보 윈도우의 구조를 이해하세요. Lead time 개념이 여기서 정의됩니다.
   Understand input window vs forecast window structure with Table 1. Lead time concept is defined here.

6. **Section 5 (Results)** (p.9-13): Figure 4 (히트맵)와 Figure 6 (calibration plot)이 핵심. BSS_clim과 ROC AUC를 리드 타임별로 어떻게 변하는지 추적하세요.
   Figures 4 (heatmaps) and 6 (calibration plots) are key. Track how BSS_clim and ROC AUC change with lead time.

7. **Section 6 (Discussion)** (p.14-16): Figure 7의 4가지 사례 연구(TP, FP, FN, TN)가 모델의 강점과 한계를 직관적으로 보여줍니다. 특히 FN 사례에서 CME의 영향에 주목하세요.
   The 4 case studies in Figure 7 intuitively show model strengths and limitations. Note the CME impact in the FN case.

### 주의 깊게 볼 포인트 / Points to Watch

- **CME 배제의 영향**: 이 모델은 ambient solar wind만 다루며 CME를 포함하지 않습니다. 이것이 FN (미탐지)의 주된 원인입니다.
  **Impact of CME exclusion**: This model only handles ambient solar wind, not CMEs. This is the primary cause of false negatives.
- **24hr vs 36hr 리드 타임 성능 변화**: 24hr까지는 성능이 크게 유지되지만 36hr에서 약간 감소. 왜 그런지 생각해보세요.
  **Performance change at 24hr vs 36hr lead times**: Performance largely holds up to 24hr but slightly decreases at 36hr. Consider why.
- **데이터 불균형 처리**: 8739개 윈도우 중 폭풍은 27%뿐. 어떻게 균형을 맞추었는지 주목하세요.
  **Class imbalance handling**: Only 27% of 8739 windows are storms. Note how they balance this.

---

## 7. 현대적 의의 / Modern Significance

이 논문은 우주기상 예보 분야에서 여러 중요한 의미를 가집니다:

This paper has several important implications for space weather forecasting:

1. **물리 모델 + ML 하이브리드 접근법의 검증**: 순수 데이터 기반도, 순수 물리 기반도 아닌 두 접근법의 결합이 효과적임을 입증. 이는 현대 우주기상 예보의 주요 트렌드입니다.
   **Validation of physics + ML hybrid approach**: Demonstrates that combining physics-based and data-driven methods is effective — a major trend in modern space weather forecasting.

2. **운용 가능성**: HUXt가 이미 University of Reading에서 실시간 운용 중이며, MAS를 WSA로 대체하면 near-real-time 운용이 가능합니다. 코드가 공개되어 있습니다 (Zenodo).
   **Operational viability**: HUXt is already running operationally at University of Reading. Substituting WSA for MAS enables near-real-time operation. Code is publicly available (Zenodo).

3. **확률적 예보 패러다임**: 결정론적 "폭풍이 올 것이다/아니다" 대신 "24시간 내 폭풍 확률 72%"와 같은 확률적 예보를 제공하여 의사결정자에게 더 유용한 정보를 제공합니다.
   **Probabilistic forecasting paradigm**: Instead of deterministic "storm/no storm," provides probabilities like "72% chance of storm within 24 hours" — more useful for decision-makers.

4. **향후 연구 방향**: CME 앙상블 통합, 3D MHD 앙상블, B_z 예측 통합 등이 자연스러운 확장 방향으로, 이 프레임워크가 미래 연구의 기반이 될 수 있습니다.
   **Future research directions**: CME ensemble integration, 3D MHD ensembles, B_z prediction integration are natural extensions — this framework could serve as the foundation for future research.

5. **경제적 영향**: 영국의 경우 Carrington급 사건의 피해 추정치가 £15.9B이며, 현재 예보 능력으로도 £2.9B까지 줄일 수 있고, 이 연구와 같은 확장된 리드 타임은 £900M까지 줄일 가능성을 제시합니다.
   **Economic impact**: For the UK, a Carrington-class event damage estimate is £15.9B, reducible to £2.9B with current forecasting and potentially to £900M with extended lead times like this study provides.

---

## Q&A

(읽기 세션 중 추가됨 / Populated during reading session)
