---
title: "Extended Lead-Time Geomagnetic Storm Forecasting With Solar Wind Ensembles and Machine Learning"
authors: M. Billcliff, A. W. Smith, M. Owens, W. L. Woo, L. Barnard, N. Edward-Inatimi, I. J. Rae
year: 2026
journal: "Space Weather, 24, e2025SW004823"
doi: "10.1029/2025SW004823"
topic: Space Weather / Geomagnetic Storm Forecasting
tags: [Hp30, geomagnetic storm, solar wind ensemble, HUXt, MAS, logistic regression, probabilistic forecasting, machine learning, Carrington map, OMNI]
status: completed
date_started: 2026-04-15
date_completed: 2026-04-15
---

# 37. Extended Lead-Time Geomagnetic Storm Forecasting With Solar Wind Ensembles and Machine Learning / 태양풍 앙상블과 머신러닝을 이용한 확장 리드 타임 지자기 폭풍 예보

---

## 1. Core Contribution / 핵심 기여

이 논문은 태양풍 앙상블 수치 모델링과 머신러닝을 결합하여 지자기 폭풍 예보의 리드 타임을 기존 L1 기반의 30–90분에서 최대 24–36시간으로 확장하는 프레임워크를 제안합니다. MAS 3D MHD 모델의 Carrington map 출력에서 위도 섭동을 통해 100개의 태양풍 속도 프로파일 앙상블을 추출하고, 각각을 1D reduced-physics HUXt 모델로 지구까지 전파합니다. 이후 각 앙상블 멤버에 대해 개별 로지스틱 회귀 분류기를 훈련하여 Hp30 지자기 폭풍 확률을 예측하고, OMNI 관측과의 MAE(평균절대오차)에 반비례하는 가중 평균으로 최종 확률적 예보를 산출합니다. 이 프레임워크는 Hp30_MAX ≥ 5 기준으로 6시간 리드 타임에서 ROC AUC 0.751, BSS_clim 0.595를 달성하며, 24시간 리드 타임에서도 ROC AUC 0.645, BSS_clim 0.529로 의미 있는 예보 스킬을 유지합니다. 특히 확률적 예보의 calibration이 우수하여, 운용 환경에서의 의사결정 지원에 실질적으로 활용 가능한 수준입니다.

This paper proposes a framework that extends geomagnetic storm forecast lead times from the current L1-based 30–90 minutes to 24–36 hours by combining solar wind ensemble numerical modeling with machine learning. One hundred solar wind velocity profile ensembles are extracted from MAS 3D MHD Carrington map output via latitudinal perturbations, and each is propagated to Earth using the 1D reduced-physics HUXt model. Individual logistic regression classifiers are trained per ensemble member to predict Hp30 geomagnetic storm probability, and the final probabilistic forecast is produced via MAE-weighted averaging against OMNI observations. The framework achieves ROC AUC of 0.751 and BSS_clim of 0.595 at 6-hr lead time for Hp30_MAX ≥ 5, and maintains meaningful forecast skill at 24-hr lead time (ROC AUC 0.645, BSS_clim 0.529). The probabilistic forecasts are notably well-calibrated, making them practically usable for operational decision support.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction / 서론 (Section 1, pp.1–2)

지자기 폭풍은 자기권의 대규모 교란으로, 위성, 통신, 전력망에 심각한 영향을 미칩니다. 현재 예보 시스템은 L1 위성(태양-지구 사이 Lagrangian L1 점) 데이터에 의존하여 30–90분의 리드 타임만 제공합니다.

Geomagnetic storms are large disruptions of the magnetosphere that can impact satellites, communications, and power grids. Current forecasting systems rely on L1 satellite data, providing only 30–90 min lead time.

저자들은 두 가지 핵심 동기를 제시합니다: (1) 인프라 보호를 위한 긴 리드 타임의 필요성 — Oughton et al. (2019)에 따르면 영국에서 대규모 Carrington급 폭풍 시 완화 조치 없이 £15.9B, 현재 예보 능력으로 £2.9B, 최적 예보로 £900M의 손실이 예상됩니다. (2) 태양풍 전파 시간(1–3일)을 활용하면 리드 타임을 크게 확장할 수 있지만, 태양풍 매핑과 전파 과정의 공간적 불확실성이 문제입니다.

The authors present two key motivations: (1) the need for longer lead times for infrastructure protection — Oughton et al. (2019) estimates UK damages of £15.9B for a Carrington-class event without mitigation, £2.9B with current forecasting, and potentially £900M with optimal forecasting; (2) solar wind travel time (1–3 days) can extend lead times significantly, but spatial uncertainties in solar wind mapping and propagation remain problematic.

기존 단기 리드 타임 Kp 예보 모델(Tan et al., 2018; Chakraborty & Morley, 2020)은 L1 데이터를 입력으로 사용하여 미래 태양풍 조건에 대한 정보가 없습니다. 이 논문의 접근법은 "미래" 태양풍 프로파일의 앙상블을 제공하여 근본적으로 다릅니다.

Existing short-lead-time Kp forecasting models (Tan et al., 2018; Chakraborty & Morley, 2020) use L1 data as input and have no information about future solar wind conditions. This paper's approach is fundamentally different by providing an ensemble of "future" solar wind profiles.

### Part II: Data — 지자기 지수와 OMNI / Geomagnetic Indices and OMNI (Section 2, pp.2–3)

#### 2.1 Hp30 지수 선택의 근거 / Rationale for Hp30 Index Choice

저자들은 세 가지 지자기 지수를 비교합니다: Kp (Bartels, 1949), Hp60, Hp30 (Yamazaki et al., 2022). 핵심 비교:

The authors compare three geomagnetic indices: Kp, Hp60, and Hp30. Key comparison:

| 지수 / Index | 시간 해상도 / Cadence | 범위 / Range | 한계 / Limitation |
|---|---|---|---|
| Kp | 3시간 / 3-hourly | 0–9 (상한 있음 / capped) | 극한 폭풍 구별 불가 / Cannot distinguish extreme storms |
| Hp60 | 1시간 / hourly | 상한 없음 / open-ended | Hp30보다 낮은 해상도 / Lower resolution than Hp30 |
| Hp30 | 30분 / 30-min | 상한 없음 / open-ended | 1995년부터 데이터 / Data from 1995 only |

Hp30을 선택한 이유: (1) 더 높은 시간 해상도로 24시간 윈도우 내 더 많은 데이터 포인트 확보, (2) 상한이 없어 극한 폭풍 구별 가능 — 예를 들어 Figure 1의 2003 Halloween 폭풍에서 Kp는 최대값 9에서 정체하지만 Hp30은 11.66까지 상승하여 폭풍의 실제 강도를 반영합니다.

Hp30 was chosen because: (1) higher temporal resolution captures more data points within 24-hr windows, (2) the open-ended scale distinguishes extreme storms — e.g., during the 2003 Halloween storm (Figure 1), Kp saturates at 9 while Hp30 reaches 11.66, reflecting actual storm intensity.

**폭풍 정의 / Storm Definition**: Hp30_MAX ≥ 4.66 within a 24-hr forecast window. 이 값은 NOAA G-scale의 G1 (minor) 이상에 해당하며, Kp = 5에 대응합니다.

**Storm definition**: Hp30_MAX ≥ 4.66 within a 24-hr forecast window, corresponding to G1 (minor) or above on the NOAA G-scale, equivalent to Kp = 5.

#### 2.2 OMNI 데이터 / OMNI Data

OMNI 데이터셋은 지구 근처의 태양풍 속성, IMF, 지자기 지수의 처리된 데이터 모음으로 1963년부터 제공됩니다. 이 연구에서는 시간 해상도(hourly) 태양풍 속도 데이터를 사용하며, 두 가지 용도로 활용됩니다:

The OMNI dataset is a processed collection of near-Earth solar wind properties, IMF, and geomagnetic indices available from 1963. This study uses hourly solar wind velocity data for two purposes:

1. **가중 방법 / Weighting method**: 앙상블 멤버의 HUXt 출력과 OMNI의 비교를 통한 MAE 계산 → 가중치 산출.
   MAE calculation by comparing ensemble member HUXt output with OMNI → weight computation.
2. **파생 피처 / Derived feature**: v − OMNI (앙상블 예측 속도와 OMNI 관측 속도의 차이)를 분류기 입력으로 사용.
   v − OMNI (difference between ensemble predicted velocity and OMNI observed velocity) as classifier input.

중요한 제한: OMNI에는 ambient solar wind와 transient(CME)가 모두 포함되지만, HUXt 앙상블은 ambient만 모델링하므로 CME 존재 시 불일치가 발생합니다. 이는 강한 폭풍 예보 성능 저하의 원인입니다.

Important limitation: OMNI contains both ambient solar wind and transients (CMEs), but HUXt ensembles model only ambient wind, causing discrepancies when CMEs are present. This is a key source of forecast performance degradation for strong storms.

### Part III: Models — 모델 파이프라인 / Model Pipeline (Section 3, pp.4–8)

전체 모델은 세 단계로 구성됩니다 (Figure 2):

The full model consists of three stages (Figure 2):

```
Model A (MAS)          → Model B (HUXt)        → Model C (ML)
3D MHD 코로나 모델       1D 태양풍 전파            폭풍 분류
21.5 R☉ Carrington map   21.5 R☉ → 1 AU          확률적 예보
```

#### 3.1 Model A — MAS (pp.4–6)

MAS (Magnetohydrodynamic Algorithm outside a Sphere)는 Riley et al. (2001)의 3D MHD 모델로, 태양 코로나(1 R☉ ~ 21.5 R☉)를 시뮬레이션합니다. 광구 자기장 관측을 내부 경계 조건으로 사용하여 21.5 R☉에서의 Carrington map (태양풍 속도 2D 맵)을 출력합니다.

MAS is Riley et al.'s (2001) 3D MHD model simulating the solar corona (1 R☉ to 21.5 R☉). Using photospheric magnetic field observations as inner boundary conditions, it outputs a Carrington map (2D solar wind velocity map) at 21.5 R☉.

**앙상블 추출 (Section 3.1.1)** — 이 논문의 핵심 혁신 중 하나:

**Ensemble extraction (Section 3.1.1)** — one of the paper's key innovations:

태양이 자전하면서 지구는 Carrington map 위에서 근사적으로 직선 경로를 추적합니다. 태양풍 속도 구조의 위치 불확실성을 고려하기 위해, 지구 경로를 사인파로 섭동합니다:

As the Sun rotates, Earth traces an approximately straight line on the Carrington map. To account for positional uncertainty in solar wind speed structures, Earth's path is perturbed sinusoidally:

$$\theta(\phi) = \theta_E + \theta_{MAX} \sin(\phi + \phi_0)$$

- $\theta_{MAX}$: 정규분포 $N(0, \sigma^2)$에서 추출, $\sigma = 7.5°$. 공간적 불확실성을 충분히 커버하는 값.
  Drawn from $N(0, \sigma^2)$ with $\sigma = 7.5°$, sufficient to cover spatial uncertainties.
- $\phi_0$: 균일분포 $U(0, 2\pi)$에서 추출.
  Drawn from $U(0, 2\pi)$.

Figure 3은 Carrington rotation 1900 (1995-09-02 ~ 1995-09-30)의 MAS 출력과 50개 섭동 경로에서 추출된 속도 프로파일을 보여줍니다. 같은 Carrington map에서도 위도에 따라 태양풍 속도가 크게 달라짐을 확인할 수 있으며, 이것이 앙상블의 다양성을 보장합니다.

Figure 3 shows MAS output for Carrington rotation 1900 and velocity profiles extracted from 50 perturbation paths. Solar wind velocities vary significantly with latitude even on the same map, ensuring ensemble diversity.

최종적으로 100개의 섭동 경로(≥50 필요)에서 각각 하나의 태양풍 속도 프로파일을 추출하여 앙상블을 구성합니다. Apple M2 (8 cores, 16 GB RAM)에서 100개 앙상블 시뮬레이션에 약 9분 소요됩니다.

100 perturbation paths (≥50 needed) each yield one solar wind velocity profile, forming the ensemble. About 9 minutes for 100 ensemble simulations on Apple M2 (8 cores, 16 GB RAM).

#### 3.2 Model B — HUXt (pp.7)

HUXt (Heliospheric Upwind eXtrapolation with time dependency)는 1D reduced-physics 태양풍 전파 모델입니다 (M. Owens et al., 2020; Barnard & Owens, 2022). MAS가 출력한 21.5 R☉에서의 각 앙상블 속도 프로파일을 경계 조건으로 사용하여 태양풍을 지구(1 AU)까지 전파합니다.

HUXt is a 1D reduced-physics solar wind propagation model. It takes each ensemble velocity profile from MAS at 21.5 R☉ as boundary condition and propagates solar wind to Earth (1 AU).

**HUXt의 장점과 한계 / Advantages and Limitations**:
- **장점 / Advantages**: 계산 속도가 매우 빠름(대규모 앙상블에 적합), Python 오픈소스 패키지로 공개, CME 시뮬레이션 기능도 있음(본 연구에서는 미사용).
  Extremely fast computation (suitable for large ensembles), open-source Python package, has CME simulation capability (not used here).
- **한계 / Limitations**: 1D이므로 자기장 성분(B_z 등)을 출력하지 않음 — 이는 폭풍 강도에 중요한 파라미터이지만 현재 프레임워크에서는 태양풍 속도만 사용합니다. 또한 radial flow 가정이 있어 복잡한 태양권 구조를 완전히 포착하지 못합니다.
  1D so cannot output magnetic field components (B_z etc.) — important for storm intensity but the current framework uses only solar wind velocity. Also assumes radial flow, not fully capturing complex heliospheric structures.

#### 3.3 Model C — 머신러닝 폭풍 분류기 / ML Storm Classifiers (pp.7–8)

모델 C는 두 단계로 구성됩니다:

Model C consists of two stages:

**C(i) — 개별 로지스틱 회귀 분류기 앙상블 / Ensemble of Individual Logistic Regression Classifiers**

각 HUXt 앙상블 멤버에 대해 별도의 로지스틱 회귀 분류기를 훈련합니다. 즉, 100개의 앙상블 멤버 → 100개의 독립 분류기입니다.

A separate logistic regression classifier is trained for each HUXt ensemble member: 100 ensemble members → 100 independent classifiers.

로지스틱 함수:
$$P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_1 + \beta_2 X_2 + \cdots + \beta_n X_n)}}$$

각 분류기의 입력 피처 (시계열, 입력 윈도우 = $T_0 - 24$hr ~ $T_0$):

Input features for each classifier (time series, input window = $T_0 - 24$hr to $T_0$):

| 피처 / Feature | 설명 / Description | 입력 윈도우 / Input Window | 포스트 입력 윈도우 / Post Input Window |
|---|---|---|---|
| HUXt $v$ | 예측 태양풍 속도 / Predicted solar wind velocity | ✓ | ✓ ($T_0$ ~ $T_0 + 48$hr) |
| HUXt $\Delta v$ | 속도 기울기 (유한 차분) / Velocity gradient (finite differences) | ✓ | ✓ |
| $v - \text{OMNI}$ | 예측-관측 차이 / Predicted-observed difference | ✓ | ✗ |
| Hp30 데이터 / data | 관측된 Hp30 지수 / Observed Hp30 index | ✓ | ✗ |

핵심: HUXt 속도 $v$와 $\Delta v$는 입력 윈도우뿐만 아니라 $T_0$ 이후 48시간까지도 사용됩니다. 이것이 "미래를 볼 수 있는" 장점입니다 — HUXt가 이미 미래의 태양풍을 시뮬레이션했기 때문입니다. $Lt \leq 24$시간일 때, 태양풍 앙상블의 길이는 예보 윈도우를 완전히 커버하고 약간 더 넘어갑니다.

Key point: HUXt velocities $v$ and $\Delta v$ are used not only in the input window but also up to 48 hr after $T_0$. This is the "see the future" advantage — HUXt has already simulated future solar wind. When $Lt \leq 24$ hr, the ensemble length fully covers and slightly exceeds the forecast window.

로지스틱 회귀 선택 이유: (1) 단순하여 소규모 데이터셋(2345 storm + 2345 non-storm)에서도 잘 작동, (2) 확률적 출력이 명확한 결정 경계를 제공, (3) L-BFGS 솔버로 효율적 학습 가능.

Logistic regression was chosen for: (1) simplicity working well on small datasets (2345 storm + 2345 non-storm), (2) probabilistic output with clear decision boundary, (3) efficient training with L-BFGS solver.

**C(ii) — MAE 가중 평균 / MAE-Weighted Mean**

100개 분류기의 출력을 단일 확률적 예보로 집약합니다. OMNI 관측과의 MAE가 작은 멤버에 더 큰 가중치를 부여합니다:

Aggregates 100 classifier outputs into a single probabilistic forecast, weighting members with smaller MAE against OMNI more heavily:

$$w_j = \frac{1}{MAE_j^2}$$

$$w_j^{norm} = \frac{w_j}{\sum_j w_j}$$

$$\hat{y} = \sum_j w_j^{norm} \cdot p_j$$

여러 집약 전략(로지스틱 회귀, 약한 멤버 필터링 등)을 평가한 결과, 가중 평균이 가장 일관된 성능을 보였습니다. 가중치는 각 출력의 MAE에 비례하며, 이는 해당 HUXt 태양풍 속도 프로파일과 OMNI의 일치도를 반영합니다.

After evaluating several aggregation strategies (logistic regression, weak member filtering, etc.), the weighted average showed the most consistent performance. Weights are proportional to each output's MAE, reflecting how well the corresponding HUXt solar wind velocity profile matches OMNI.

#### 3.4 예보 설정 / Forecast Setup (p.8)

**핵심 개념 정의 / Key Concept Definitions**:

- **입력 윈도우 (Input window)**: $T_0 - 24$hr ~ $T_0$. 모든 관측 데이터가 여기서 사용 가능.
  $T_0 - 24$hr to $T_0$. All observed data available here.
- **예보 시점 ($T_0$)**: 예보가 만들어지는 시점.
  The time point when the forecast is made.
- **리드 타임 ($Lt$)**: $T_0$에서 예보 윈도우 시작까지의 간격.
  Gap from $T_0$ to the start of the forecast window.
- **예보 윈도우 (Forecast window)**: $T_0 + Lt$ ~ $T_0 + Lt + 24$hr. 이 기간 내 Hp30_MAX ≥ 4.66이면 "폭풍."
  $T_0 + Lt$ to $T_0 + Lt + 24$hr. "Storm" if Hp30_MAX ≥ 4.66 within this window.

**데이터 분할 / Data Split**:
- 훈련:테스트 = 80:20 (Carrington rotation 기반 체계적 분할, 랜덤 아님).
  Train:test = 80:20 (systematic split based on Carrington rotations, not random).
- 태양 주기의 다양한 위상이 훈련/테스트에 모두 포함되도록 설계.
  Designed so that different phases of the solar cycle are represented in both sets.
- 1 Carrington rotation 테스트 후 4 Carrington rotation 훈련 패턴 반복.
  Pattern: 1 Carrington rotation test, then 4 training, repeating.

**클래스 불균형 처리 / Class Imbalance Handling**:
- 전체 8739개 윈도우 중 폭풍은 2345개 (27%). 비폭풍을 랜덤 드롭아웃하여 2345개로 균형화.
  Of 8739 windows, only 2345 are storms (27%). Non-storms randomly dropped to 2345 for balance.
- 균형화 후 기후학 확률 = 0.5 → BSS_clim의 기준이 됨.
  After balancing, climatological probability = 0.5, serving as BSS_clim reference.

**교차 검증 / Cross Validation**:
- 5-fold CV (테스트 폴드 이동) × 5 random seeds = 25회 실행으로 성능 변동성 평가.
  5-fold CV (shifting test fold) × 5 random seeds = 25 runs to assess performance variability.

### Part IV: Metrics / 평가 지표 (Section 4, pp.9)

두 가지 핵심 지표를 사용합니다:

Two key metrics are used:

**1. ROC AUC (Receiver Operating Characteristic Area Under Curve)**

$$TPR = \frac{TP}{TP + FN}, \quad FPR = \frac{FP}{FP + TN}$$

TPR vs FPR 곡선 아래 면적. 0.5 = 무작위 추측, 1.0 = 완벽한 판별. 분류기의 판별 능력(discriminative skill)을 측정합니다.

Area under the TPR vs FPR curve. 0.5 = random guessing, 1.0 = perfect discrimination. Measures the classifier's discriminative skill.

**2. BSS_clim (Brier Skill Score relative to Climatology)**

$$BS = \frac{1}{N}\sum_{i=1}^{N}(p_i - o_i)^2$$

$$BSS_{clim} = 1 - \frac{BS}{BS_{clim}}$$

$BS$는 확률 예보의 평균 제곱 오차, $BSS_{clim}$은 기후학 대비 상대적 스킬. 양수 = 기후학보다 우수, 0.2 이상 = 의미 있는 부가가치 (Wilks, 2011). BSS_clim은 판별 스킬과 확률적 정확도/신뢰도를 모두 포착합니다.

$BS$ is the mean squared error of probability forecasts, $BSS_{clim}$ is skill relative to climatology. Positive = better than climatology, ≥ 0.2 = meaningful added value (Wilks, 2011). BSS_clim captures both discriminative skill and probabilistic accuracy/reliability.

### Part V: Results / 결과 (Section 5, pp.9–15)

#### 5.1 베이스라인 대비 성능 / Comparison to Baseline Models (pp.10–11)

두 베이스라인 모델:

Two baseline models:

- **Persistence**: 예보 직전 시점의 Hp30가 임계값을 초과했는지로 예보. 단기에 유리.
  Whether Hp30 exceeded threshold just before forecast time. Favors short lead times.
- **27-day recurrence**: 27일 전(태양 자전 주기)에 폭풍이었는지로 예보. Hp30의 27일 자기상관 r = 0.25.
  Whether it was a storm 27 days prior (solar rotation period). Hp30 autocorrelation at 27 days: r = 0.25.

**Figure 4 — 핵심 결과 히트맵 (Hp30 ≥ 5 기준) / Key Result Heatmaps (Hp30 ≥ 5 threshold)**:

| Model | Metric | Lt=1hr | Lt=3hr | Lt=6hr | Lt=12hr | Lt=24hr | Lt=36hr |
|---|---|---|---|---|---|---|---|
| Weighted Mean | BSS_clim | 0.631 | 0.618 | 0.595 | 0.568 | 0.529 | 0.510 |
| Persistence | BSS_clim | 0.172 | 0.163 | 0.146 | 0.122 | 0.087 | 0.058 |
| 27-day Recurrence | BSS_clim | 0.259 | 0.260 | 0.280 | 0.274 | 0.263 | 0.288 |
| Weighted Mean | ROC AUC | 0.801 | 0.785 | 0.751 | 0.714 | 0.645 | 0.589 |
| Persistence | ROC AUC | 0.586 | 0.581 | 0.573 | 0.561 | 0.544 | 0.529 |
| 27-day Recurrence | ROC AUC | 0.630 | 0.630 | 0.640 | 0.637 | 0.631 | 0.644 |

핵심 관찰: / Key observations:
- 가중 평균 모델은 모든 리드 타임과 메트릭에서 persistence를 능가합니다.
  The weighted mean model outperforms persistence at all lead times and metrics.
- BSS_clim 기준으로 가중 평균은 27-day recurrence를 모든 리드 타임에서 크게 능가합니다.
  By BSS_clim, the weighted mean substantially outperforms 27-day recurrence at all lead times.
- ROC AUC 기준으로 24hr 이후에는 27-day recurrence가 가중 평균을 근소하게 앞서는데, 이는 27-day recurrence가 결정론적이라 확률적 정확도(BSS)에서는 불리하지만 판별력(ROC AUC)은 긴 리드 타임에서 유지되기 때문입니다.
  By ROC AUC, 27-day recurrence slightly outperforms the weighted mean after 24hr because recurrence is deterministic — it suffers in probabilistic accuracy (BSS) but maintains discrimination at long lead times.
- BSS_clim이 일관되게 0.2 이상으로, 36hr 리드 타임에서도 의미 있는 스킬을 유지합니다.
  BSS_clim is consistently above 0.2, maintaining meaningful skill even at 36-hr lead time.

#### 5.2 폭풍 강도와 리드 타임의 영향 / Impact of Storm Strength and Lead Time (pp.11)

Figure 5는 다양한 폭풍 강도 임계값(Hp30_MAX ≥ 5, 6, 7, 8)에 대한 성능을 보여줍니다:

Figure 5 shows performance across storm intensity thresholds (Hp30_MAX ≥ 5, 6, 7, 8):

- BSS_clim은 모든 임계값에서 0.2를 크게 초과하여 안정적인 성능을 보입니다.
  BSS_clim consistently exceeds 0.2 across all thresholds, showing stable performance.
- ROC AUC는 임계값이 높아질수록(더 강한 폭풍) 오히려 증가하는 경향: Hp30 ≥ 5에서 0.801 → Hp30 ≥ 8에서 0.876 (Lt=1hr).
  ROC AUC tends to increase with higher thresholds (stronger storms): 0.801 at Hp30 ≥ 5 → 0.876 at Hp30 ≥ 8 (Lt=1hr).
- 그러나 높은 임계값에서는 표본 크기가 작아져 신뢰 구간이 넓어집니다 (Hp30 ≥ 8: 테스트셋에 24개 기간만 존재).
  However, higher thresholds have smaller sample sizes, widening confidence intervals (Hp30 ≥ 8: only 24 periods in test set).
- 3hr ~ 6hr 사이에서 가장 큰 성능 하락이 관찰됩니다. 이는 입력 윈도우 내 자기권 상태 정보(Hp30, v−OMNI)의 영향이 줄어들고, 앙상블 데이터에 대한 의존도가 높아지는 전환점입니다.
  The largest performance drop occurs between 3hr and 6hr lead times, marking the transition point where magnetospheric state information in the input window becomes less influential and reliance on ensemble data increases.

#### 5.3 예보 보정 / Forecast Calibration (pp.12–13)

Figure 6은 다양한 리드 타임과 Hp30 임계값에 대한 calibration plot을 보여줍니다. Calibration은 예보 확률과 관측 빈도의 일치도를 평가합니다.

Figure 6 shows calibration plots for various lead times and Hp30 thresholds. Calibration evaluates how well forecast probabilities match observed frequencies.

핵심 결과: / Key findings:
- Hp30 ≥ 5에서 모든 리드 타임에 걸쳐 잘 보정되어 있음 — 예보선이 y=x 완벽 보정선에 가깝게 따릅니다.
  Well-calibrated across all lead times at Hp30 ≥ 5 — forecast lines closely follow the y=x perfect calibration line.
- IQR (25-75 percentile) 범위가 일관되게 좁아 테스트셋에 걸쳐 안정적인 확률을 보입니다.
  IQR ranges are consistently narrow, showing stable probabilities across the test set.
- 36hr 리드 타임에서 낮은 확률(< 0.5) 영역의 보정이 평탄해지는 경향 — 모델이 비폭풍에 대해 보수적(약간 과예보)입니다.
  At 36-hr lead time, calibration flattens for low probabilities (< 0.5) — model is conservative (slightly over-predicts) for non-storms.
- Hp30 ≥ 7 이상에서 보정 품질이 저하됨 — CME 구동 대형 이벤트가 모델에 포함되지 않고 표본 크기가 작기 때문입니다.
  Calibration quality degrades for Hp30 ≥ 7 — due to CME-driven large events not included in the model and small sample sizes.

### Part VI: Discussion / 논의 (Section 6, pp.13–16)

#### 6.1 사례 연구 — 4가지 Contingency Table 사례 / Case Studies (pp.13–14)

Figure 7은 12hr 리드 타임에서 TP, FP, FN, TN 각각의 대표적 사례를 보여줍니다:

Figure 7 shows representative cases of TP, FP, FN, TN at 12-hr lead time:

**True Positive (TP)**: 2003-08-06. 태양풍 앙상블이 OMNI와 같은 증가 추세를 정확히 포착. 앙상블 멤버들이 corotating 구조에 의한 속도 증가를 일관되게 예측하여 0.562의 폭풍 확률을 산출. 27-day recurrence도 정확히 예보함 (같은 corotating 구조가 27일 전에도 존재).

**TP**: 2003-08-06. Solar wind ensemble accurately captures the same increasing trend as OMNI. Ensemble members consistently predict velocity increase from a corotating structure, yielding 0.562 storm probability. 27-day recurrence also correctly forecasts (same corotating structure existed 27 days prior).

**False Positive (FP)**: 2019-09-05. 앙상블이 OMNI와 비슷한 하강 추세를 보이지만, 100개 중 19개 멤버가 급격한 속도 상승을 잘못 예측. MAE가 가장 낮은(가장 정확한) 3개 멤버가 이 급상승을 포함하여 가중치가 높아져 오경보 발생.

**FP**: 2019-09-05. Ensemble shows similar downward trend as OMNI, but 19 of 100 members incorrectly predict a sharp velocity rise. The 3 members with lowest MAE (highest weight) happen to contain this sharp rise, contributing to the false alarm.

**False Negative (FN)**: 2000-04-06. 예보 윈도우 시작 근처에서 CME 도착이 발생. CME의 속도가 ~600 km/s로 특별히 빠르지 않지만, CME 시점에 B_z가 북향에서 남향으로 전환되어 강한 지자기 활동을 유발. 이 모델은 CME도 B_z도 포함하지 않으므로 이러한 사건을 원리적으로 예보할 수 없습니다.

**FN**: 2000-04-06. A CME arrival occurs near the start of the forecast window. The CME velocity is ~600 km/s (not particularly fast), but B_z flips from northward to southward at the CME time, driving strong geomagnetic activity. Since this model includes neither CMEs nor B_z, it fundamentally cannot forecast such events.

**True Negative (TN)**: 2009-07-24. 대부분의 앙상블 멤버가 ~300 km/s의 낮은 태양풍 속도를 예측. 일부 멤버만 속도 상승을 예측하지만, 빠르게 소산되는 구조이므로 27-day recurrence는 오경보를 내지만 가중 평균 모델은 정확히 비폭풍으로 분류.

**TN**: 2009-07-24. Most ensemble members predict low solar wind velocity ~300 km/s. Some predict a velocity rise, but it's a dissipating structure, so 27-day recurrence falsely alarms while the weighted mean model correctly classifies as non-storm.

#### 6.2–6.4 앙상블 보정과 한계 / Ensemble Calibration and Limitations (pp.15–16)

**앙상블 보정**: Barnard et al. (2023), Lang et al. (2021)의 데이터 동화 접근법으로 STEREO-A, STEREO-B, ACE의 in-situ 데이터를 사용하여 HUXt 경계 조건을 업데이트하면 RMSE가 31.4% 감소합니다. 이 연구에서는 미사용이나 향후 통합 예정.

**Ensemble calibration**: Data assimilation approaches by Barnard et al. (2023) and Lang et al. (2021) using STEREO-A, STEREO-B, ACE in-situ data to update HUXt boundary conditions reduce RMSE by 31.4%. Not used here but planned for future integration.

**현재 앙상블의 한계**:
1. **CME 미포함**: 가장 큰 한계. 강한 폭풍(Hp30 ≥ 7)의 많은 부분이 CME 구동인데, 현재 프레임워크는 ambient solar wind만 모델링합니다.
   **No CMEs**: The biggest limitation. Many strong storms (Hp30 ≥ 7) are CME-driven, but the framework models only ambient solar wind.
2. **1D 모델 한계**: HUXt는 속도만 출력하므로 IMF(특히 B_z), 밀도, 동압을 제공하지 못합니다.
   **1D model limitation**: HUXt outputs velocity only — cannot provide IMF (especially B_z), density, or dynamic pressure.
3. **MAS 데이터 가용성**: MAS 솔루션이 실시간으로 제공되지 않음. WSA로 대체하면 near-real-time 운용이 가능하지만, 두 모델 간 차이로 인해 재훈련이 필요합니다.
   **MAS data availability**: MAS solutions not available in real time. Substituting WSA enables near-real-time operation, but retraining needed due to inter-model differences.

#### 6.5 운용화 / Operational Forecasting (p.16)

HUXt 태양풍 앙상블은 이미 University of Reading에서 실시간 운용 중입니다 (https://research.reading.ac.uk/met-spate/huxt-forecast/). 운용화를 위한 주요 변경 사항:

HUXt solar wind ensembles are already running operationally at the University of Reading. Key modifications for operationalization:

1. MAS → WSA 대체 (WSA는 실시간 가용)
   Substitute MAS with WSA (WSA available in real time)
2. OMNI → ACE 또는 DSCOVR 실시간 데이터 대체
   Substitute OMNI with ACE or DSCOVR real-time data
3. Hp30 지수는 이미 실시간 제공 (GFZ Potsdam)
   Hp30 index already available in real time (GFZ Potsdam)
4. Model C(i) 재훈련 필요
   Model C(i) retraining needed

코드가 Zenodo에 공개되어 있습니다 (Billcliff, 2025).

Code is publicly available on Zenodo (Billcliff, 2025).

### Part VII: Conclusion / 결론 (Section 7, pp.16–17)

주요 결론:
1. 100개 앙상블 멤버가 ambient solar wind의 변동성을 충분히 포착합니다.
   100 ensemble members sufficiently capture ambient solar wind variability.
2. MAE 기반 가중 평균이 물리 기반과 데이터 기반 모델을 효과적으로 결합합니다.
   MAE-weighted averaging effectively combines physics-based and data-driven models.
3. 24hr 리드 타임까지 의미 있는 스킬을 유지하며, 24~36hr에서도 성능 하락이 크지 않습니다.
   Meaningful skill is maintained up to 24-hr lead time, with modest degradation at 24–36 hr.
4. 확률적 예보가 잘 보정되어 있어 운용 의사결정에 실질적으로 활용 가능합니다.
   Probabilistic forecasts are well-calibrated, practically usable for operational decision support.
5. 향후 연구: CME 앙상블 통합, WSA 경계 조건, B_z 예측 통합이 핵심 과제입니다.
   Future work: CME ensemble integration, WSA boundary conditions, and B_z prediction integration are key challenges.

---

## 3. Key Takeaways / 핵심 시사점

1. **앙상블 기법이 태양풍 예보의 공간적 불확실성을 체계적으로 다룹니다** — MAS Carrington map 위에서 지구 경로를 σ=7.5°의 사인파로 섭동하여 100개의 다양한 태양풍 프로파일을 생성함으로써, 단일 결정론적 예보가 갖는 위치 불확실성 문제를 극복합니다. 이는 기상학의 앙상블 예보 패러다임을 우주기상에 성공적으로 적용한 사례입니다.
   **Ensemble methods systematically address spatial uncertainty in solar wind forecasting** — By sinusoidally perturbing Earth's path on MAS Carrington maps with σ=7.5° to generate 100 diverse solar wind profiles, the approach overcomes the positional uncertainty inherent in single deterministic forecasts. This successfully applies the meteorological ensemble forecasting paradigm to space weather.

2. **물리 모델과 ML의 하이브리드 접근이 각각의 단독 사용보다 우수합니다** — 물리 기반 모델(MAS+HUXt)이 "미래 태양풍"의 다양한 시나리오를 제공하고, ML(로지스틱 회귀)이 이를 관측 데이터와 결합하여 최적 예보를 산출합니다. 순수 물리 모델의 체계적 편향과 순수 ML의 물리적 비일관성을 동시에 극복합니다.
   **The physics + ML hybrid approach outperforms each used alone** — Physics-based models (MAS+HUXt) provide diverse "future solar wind" scenarios, and ML (logistic regression) combines these with observational data for optimal forecasts. This overcomes both the systematic biases of pure physics models and the physical inconsistencies of pure ML.

3. **MAE 기반 가중 평균이 단순하면서도 효과적인 앙상블 집약 방법입니다** — 정확도가 높은 앙상블 멤버에 MAE의 역제곱에 비례하는 가중치를 부여하는 것만으로, 더 복잡한 집약 방법(로지스틱 메타 회귀 등)보다 일관된 성능을 달성합니다. 이는 "적절한 단순성"의 원칙을 잘 보여줍니다.
   **MAE-weighted averaging is a simple yet effective ensemble aggregation method** — Simply weighting members inversely proportional to MAE squared achieves more consistent performance than more complex aggregation methods (logistic meta-regression, etc.). This exemplifies the principle of "appropriate simplicity."

4. **Hp30 지수가 Kp보다 폭풍 예보에 더 적합합니다** — 30분 해상도와 상한 없는 스케일이 (1) 24시간 윈도우 내 더 풍부한 시계열 정보를 제공하고, (2) 2003 Halloween 폭풍(Hp30 = 11.66 vs Kp = 9)처럼 극한 폭풍을 구별할 수 있게 합니다.
   **Hp30 is more suitable than Kp for storm forecasting** — The 30-min resolution and open-ended scale (1) provide richer time-series information within 24-hr windows, and (2) distinguish extreme storms like the 2003 Halloween storm (Hp30 = 11.66 vs Kp = 9).

5. **CME 미포함이 가장 큰 성능 제한 요인입니다** — FN 사례 분석에서 CME 도착 시 B_z 남향 전환이 강한 지자기 활동을 유발하지만, HUXt는 속도만 모델링하고 CME를 포함하지 않아 이를 원리적으로 예보할 수 없습니다. 이는 Hp30 ≥ 7 이상의 강한 폭풍에서 특히 심각하며, 향후 CME 앙상블 통합이 필수적인 연구 방향입니다.
   **CME exclusion is the primary performance bottleneck** — FN case analysis shows CME arrivals with southward B_z flip driving strong geomagnetic activity, but HUXt models only velocity without CMEs and fundamentally cannot forecast such events. This is especially severe for strong storms (Hp30 ≥ 7), making CME ensemble integration an essential research direction.

6. **3–6시간 리드 타임이 핵심 전환점입니다** — 이 구간에서 성능의 가장 큰 하락이 발생하며, 이는 입력 윈도우 내의 자기권 상태 정보(관측된 Hp30, v−OMNI)의 영향력이 줄어들고 앙상블 데이터에 대한 의존도가 높아지는 전환을 반영합니다. PACF 분석(Figure A2)도 Hp30의 자기상관이 약 21시간에서 통계적 유의성을 잃음을 보여줍니다.
   **3–6 hr lead time is the critical transition point** — The largest performance drop occurs in this range, reflecting the transition from magnetospheric state information in the input window (observed Hp30, v−OMNI) to dependence on ensemble data. PACF analysis (Figure A2) shows Hp30 autocorrelation loses statistical significance at about 21 hours.

7. **확률적 예보의 보정(calibration)이 실용적 가치의 핵심입니다** — BSS_clim과 ROC AUC가 아무리 높아도, 예보 확률이 실제 발생 빈도와 일치하지 않으면 의사결정자에게 유용하지 않습니다. Figure 6의 calibration plot은 이 모델의 확률이 관측 빈도와 잘 일치함을 보여주어, "폭풍 확률 70%"라는 예보를 문자 그대로 신뢰할 수 있음을 의미합니다.
   **Forecast calibration is the key to practical value** — No matter how high BSS_clim and ROC AUC are, forecasts are useless for decision-makers if predicted probabilities don't match observed frequencies. Figure 6's calibration plots show this model's probabilities match observations well, meaning a "70% storm probability" forecast can be taken at face value.

8. **운용화까지의 경로가 구체적으로 제시됩니다** — MAS→WSA 대체, OMNI→DSCOVR/ACE 대체, 기존 University of Reading HUXt 실시간 시스템 활용 등 명확한 단계가 제시되어, 연구 결과의 운용 전환 가능성이 높습니다. 코드 공개(Zenodo)도 재현성과 확장성을 지원합니다.
   **A concrete path to operationalization is outlined** — Clear steps including MAS→WSA substitution, OMNI→DSCOVR/ACE substitution, and leveraging the existing University of Reading real-time HUXt system are outlined, making operational transition highly feasible. Code availability on Zenodo supports reproducibility and extensibility.

---

## 4. Mathematical Summary / 수학적 요약

### 전체 모델 파이프라인 / Full Model Pipeline

**Step 1: 앙상블 경로 생성 / Ensemble Path Generation**

100개의 섭동 경로를 Carrington map 위에서 생성:

$$\theta_k(\phi) = \theta_E + \theta_{MAX,k} \sin(\phi + \phi_{0,k}), \quad k = 1, \ldots, 100$$

여기서 / where:
- $\theta_{MAX,k} \sim N(0, (7.5°)^2)$
- $\phi_{0,k} \sim U(0, 2\pi)$

**Step 2: 태양풍 속도 추출 / Solar Wind Velocity Extraction**

각 경로 $k$에서 Carrington map의 태양풍 속도를 추출하여 1D 속도 프로파일 $v_k(\phi)$을 얻습니다.

Extract solar wind velocity from the Carrington map along each path $k$ to obtain 1D velocity profiles $v_k(\phi)$.

**Step 3: HUXt 전파 / HUXt Propagation**

각 $v_k$를 21.5 $R_\odot$에서 1 AU까지 1D reduced-physics 방정식으로 전파:

Propagate each $v_k$ from 21.5 $R_\odot$ to 1 AU via 1D reduced-physics equations:

$$v_k(r, t) \xrightarrow{\text{HUXt}} v_k^{Earth}(t)$$

출력: 지구에서의 시간별 태양풍 속도 예측 시계열 $v_k^{Earth}(t)$, $k = 1, \ldots, 100$.

Output: Hourly solar wind velocity prediction time series at Earth $v_k^{Earth}(t)$, $k = 1, \ldots, 100$.

**Step 4: 피처 구성 / Feature Construction**

각 앙상블 멤버 $k$에 대해 4가지 시계열 피처 생성:

For each ensemble member $k$, construct 4 time-series features:

$$X_k = \left[ v_k^{Earth}(t),\; \Delta v_k(t),\; (v_k^{Earth}(t) - v_{OMNI}(t)),\; Hp30(t) \right]$$

- $\Delta v_k(t) = v_k^{Earth}(t) - v_k^{Earth}(t-1)$ (유한 차분 기울기 / finite-difference gradient)
- 입력 윈도우: $t \in [T_0 - 24\text{hr}, T_0]$, 포스트: $v_k$와 $\Delta v_k$만 $t \in [T_0, T_0 + 48\text{hr}]$까지 확장
  Input window: $t \in [T_0 - 24\text{hr}, T_0]$; post: $v_k$ and $\Delta v_k$ extend to $t \in [T_0, T_0 + 48\text{hr}]$

**Step 5: 개별 분류기 훈련 / Individual Classifier Training**

각 앙상블 멤버 $k$에 대해 로지스틱 회귀 분류기 학습:

Train a logistic regression classifier for each ensemble member $k$:

$$p_k = P(Y=1 \mid X_k) = \frac{1}{1 + e^{-(\beta_{0,k} + \boldsymbol{\beta}_k^\top \mathbf{X}_k)}}$$

타깃 변수 / Target variable:
$$Y = \begin{cases} 1 & \text{if } Hp30_{MAX} \geq 4.66 \text{ in forecast window} \\ 0 & \text{otherwise} \end{cases}$$

**Step 6: MAE 가중 집약 / MAE-Weighted Aggregation**

각 멤버의 OMNI 대비 MAE 계산:

Compute each member's MAE against OMNI:

$$MAE_k = \frac{1}{T}\sum_{t=1}^{T}\left|v_k^{Earth}(t) - v_{OMNI}(t)\right|$$

가중치와 최종 예보:

Weights and final forecast:

$$w_k = \frac{1}{MAE_k^2}, \quad w_k^{norm} = \frac{w_k}{\sum_{j=1}^{100} w_j}, \quad \hat{y} = \sum_{k=1}^{100} w_k^{norm} \cdot p_k$$

**Step 7: 폭풍 판정 / Storm Decision**

$$\text{Forecast} = \begin{cases} \text{Storm} & \text{if } \hat{y} \geq \text{threshold} \\ \text{Non-Storm} & \text{otherwise} \end{cases}$$

(ROC AUC는 모든 threshold에 대해 평가, BSS_clim은 확률 $\hat{y}$를 직접 사용.)

(ROC AUC evaluated over all thresholds; BSS_clim uses probability $\hat{y}$ directly.)

### 평가 메트릭 / Evaluation Metrics

$$TPR = \frac{TP}{TP + FN}, \quad FPR = \frac{FP}{FP + TN}$$

$$\text{ROC AUC} = \int_0^1 TPR(FPR)\, d(FPR)$$

$$BS = \frac{1}{N}\sum_{i=1}^{N}(p_i - o_i)^2, \quad BSS_{clim} = 1 - \frac{BS}{BS_{clim}}$$

여기서 $BS_{clim} = 0.25$ (균형 데이터셋에서 기후학 확률 0.5 → $BS = (0.5)^2 = 0.25$).

Where $BS_{clim} = 0.25$ (climatology probability 0.5 on balanced dataset → $BS = (0.5)^2 = 0.25$).

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1949  Bartels                Kp 지수 정의 / Kp index defined
  │
1975  Burton et al.          최초의 Dst 경험적 예보 공식 / First empirical Dst forecasting equation
  │
2000  Arge & Pizzo           WSA 모델 — 운용 태양풍 예보의 기초 / WSA model — foundation of operational SW forecasting
  │
2001  Riley et al.           MAS 3D MHD 모델 / MAS 3D MHD model
  │
2003  Odstrcil               ENLIL — 운용 CME 전파 모델 / ENLIL — operational CME propagation model
  │
2017  Owens & Riley          Carrington map 섭동 앙상블 기법 제안 / Carrington map perturbation ensemble proposed
  │
2018  Tan et al.             LSTM으로 Kp 예보 (L1 데이터 기반) / LSTM-based Kp forecasting (L1 data)
  │
2019  Camporeale et al.      우주기상 ML 종합 리뷰 / Comprehensive ML in space weather review
  │
2020  M. Owens et al.        HUXt 모델 공개 (1D reduced-physics) / HUXt model release
  │   Chakraborty & Morley   확률적 Kp 예보 / Probabilistic Kp prediction
  │
2022  Yamazaki et al.        Hp30/Hp60 지수 도입 / Hp30/Hp60 indices introduced
  │   Barnard & Owens        HUXt 앙상블 보정 프레임워크 / HUXt ensemble calibration framework
  │   Bernoux et al.         EUV 기반 2-7일 예보 / EUV-based 2-7 day forecasting
  │
2024  Edward-Inatimi et al.  앙상블 보정 기법 확장 / Extended ensemble calibration techniques
  │
2026  ★ Billcliff et al.     본 논문: 앙상블 + ML 하이브리드 24hr 확률적 폭풍 예보
  │                          This paper: ensemble + ML hybrid 24hr probabilistic storm forecasting
  ▼
미래  CME 앙상블 통합, 3D MHD 앙상블, B_z 예측
      CME ensemble integration, 3D MHD ensembles, B_z prediction
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Riley et al. (2001) — MAS 3D MHD model | 본 논문 Model A의 기초. MAS가 Carrington map에서 21.5 R☉의 태양풍 속도 맵을 생성. / Foundation of Model A. MAS generates solar wind velocity maps at 21.5 R☉. | 직접 사용 / Directly used |
| M. Owens et al. (2020) — HUXt model | Model B의 핵심 도구. 1D reduced-physics로 21.5 R☉에서 1 AU까지 태양풍 전파. 빠른 계산이 대규모 앙상블을 가능하게 함. / Core tool for Model B. Enables large-scale ensembles through fast 1D propagation. | 직접 사용 / Directly used |
| Owens & Riley (2017) — Carrington map perturbation | 앙상블 추출의 사인파 섭동 기법의 원형. / Prototype for the sinusoidal perturbation technique for ensemble extraction. | 방법론적 기반 / Methodological basis |
| Arge & Pizzo (2000) — WSA model | MAS의 대안으로 실시간 운용 가능한 태양풍 모델. 운용화 시 MAS를 WSA로 대체 예정. / Alternative to MAS for real-time operation. WSA substitution planned for operationalization. | 운용화 경로 / Operationalization path |
| Camporeale et al. (2019) — ML in Space Weather review | 우주기상 ML의 모범 사례와 과제를 정립. 본 논문이 이 프레임워크 위에 구축. / Established best practices and challenges for ML in space weather. This paper builds on this framework. | 이론적 맥락 / Theoretical context |
| Yamazaki et al. (2022) — Hp30/Hp60 indices | 이 논문에서 사용하는 Hp30 지수를 도입. Kp 대비 30분 해상도와 상한 없는 스케일의 장점을 제공. / Introduced the Hp30 index used in this paper. Provides 30-min resolution and open-ended scale advantages over Kp. | 데이터 기반 / Data foundation |
| Burton et al. (1975) — Empirical Dst relationship (SW #11) | 태양풍 파라미터와 지자기 지수의 경험적 관계의 원형. 본 논문은 이를 확률적 앙상블 프레임워크로 확장. / Prototype of empirical relationship between solar wind parameters and geomagnetic indices. This paper extends it to a probabilistic ensemble framework. | 개념적 선행 연구 / Conceptual predecessor |
| Barnard et al. (2023) / Lang et al. (2021) — Data assimilation for HUXt | STEREO/ACE in-situ 데이터로 HUXt 경계 조건을 보정하여 RMSE 31.4% 감소. 본 논문의 자연스러운 확장 방향. / Calibrate HUXt boundary conditions with STEREO/ACE in-situ data, reducing RMSE by 31.4%. Natural extension of this paper. | 향후 통합 예정 / Planned future integration |
| Owens et al. (2021) — Extreme Space-Weather Events (SW #36) | 극한 우주기상 사건의 태양 주기 의존성을 통계적으로 입증. 본 논문의 확률적 예보가 이러한 극한 사건에 대한 대비에 기여할 수 있음. / Statistically demonstrated solar cycle dependence of extreme events. This paper's probabilistic forecasts can contribute to preparedness for such events. | 주제적 연결 / Thematic connection |

---

## 7. References / 참고문헌

- Arge, C. N., & Pizzo, V. (2000). Improvement in the prediction of solar wind conditions using near-real time solar magnetic field updates. *Journal of Geophysical Research*, 105(A5), 10465–10479.
- Barnard, L., & Owens, M. (2022). HUXt — An open-source, computationally efficient reduced-physics solar wind model, written in python. https://doi.org/10.5281/zenodo.4889326
- Barnard, L., Owens, M., Scott, C., Lang, M., & Lockwood, M. (2023). Sa-hux — A particle filter data assimilation scheme for Cme time-elongation profiles. *Space Weather*, 21(6), e2023SW003487.
- Bartels, J. (1949). The standardized index, ks, and the planetary index, kp. *IATME Bulletin*, 12B, 97–120.
- Billcliff, M. (2025). storm_forecasting. MB. Code for "Extended Lead-Time..." (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.17571893
- Burton, R., McPherron, R., & Russell, C. (1975). An empirical relationship between interplanetary conditions and Dst. *Journal of Geophysical Research*, 80(31), 4204–4214.
- Camporeale, E., Wing, S., & Johnson, J. (2019). Machine learning techniques for space weather. *Geophysical Research Letters*, 51(1), e2023GL106049.
- Chakraborty, S., & Morley, S. K. (2020). Probabilistic prediction of geomagnetic storms and the kp index. *Journal of Space Weather and Space Climate*, 10, 36.
- Edward-Inatimi, N., Barnard, L., Turner, H., Marsh, M., Gonzi, S., et al. (2024). Adapting ensemble-calibration techniques to probabilistic solar-wind forecasting. *Space Weather*, 22(12), e2024SW004164.
- Lang, M., Witherington, J., Turner, H., Owens, M. J., & Riley, P. (2021). Improving solar wind forecasting using data assimilation. *Space Weather*, 19(7), e2020SW002698.
- Odstrcil, D. (2003). Modeling 3-D solar wind structure. *Advances in Space Research*, 32(4), 497–506.
- Oughton, E. J., Hapgood, M., Richardson, G. S., Beggan, C. D., Thomson, A. W., Gibbs, M., et al. (2019). A risk assessment framework for the socioeconomic impacts of electricity transmission infrastructure failure due to space weather. *Risk Analysis*, 39(5), 1022–1043.
- Owens, M., Lang, M., Barnard, L., Riley, P., Ben-Nun, M., Scott, C. J., et al. (2020). A computationally efficient, time-dependent model of the solar wind for use as a surrogate to three-dimensional numerical magnetohydrodynamic simulations. *Solar Physics*, 295(3), 43.
- Owens, M. J., & Riley, P. (2017). Probabilistic solar wind forecasting using large ensembles of near-sun conditions with a simple one-dimensional "upwind" scheme. *Space Weather*, 15(11), 1461–1474.
- Riley, P., Linker, J. A., & Mikić, Z. (2001). An empirically-driven global MHD model of the solar Corona and inner heliosphere. *Journal of Geophysical Research*, 106(A8), 15889–15901.
- Tan, Y., Hu, Q., Wang, Z., & Zhong, Q. (2018). Geomagnetic index kp forecasting with lstm. *Space Weather*, 16(4), 406–416.
- Wilks, D. S. (2011). *Statistical methods in the atmospheric sciences* (Vol. 100). Academic Press.
- Yamazaki, Y., Matzka, J., Stolle, C., Kervalishvili, G., Rauberg, J., Bronkalla, O., et al. (2022). Geomagnetic activity index hpo. *Geophysical Research Letters*, 49(10), e2022GL098860.
