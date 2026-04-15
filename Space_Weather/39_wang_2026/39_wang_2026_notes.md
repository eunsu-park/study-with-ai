---
title: "Daily Predictions of F10.7 and F30 Solar Indices With Deep Learning"
authors: Zhenduo Wang, Yasser Abduallah, Jason T. L. Wang, Haimin Wang, Yan Xu, Vasyl Yurchyshyn, Vincent Oria, Khalid A. Alobaid, Xiaoli Bai
year: 2026
journal: "Journal of Geophysical Research: Space Physics"
doi: "10.1029/2025JA034868"
topic: Space_Weather
tags: [F10.7, F30, solar-index, deep-learning, SINet, TimesNet, FFT, CNN, time-series-forecasting, space-weather]
status: completed
date_started: 2026-04-15
date_completed: 2026-04-15
---

# Daily Predictions of F10.7 and F30 Solar Indices With Deep Learning

> Wang, Z., Abduallah, Y., Wang, J. T. L., Wang, H., Xu, Y., Yurchyshyn, V., Oria, V., Alobaid, K. A., & Bai, X. (2026).
> *Journal of Geophysical Research: Space Physics*. DOI: 10.1029/2025JA034868

---

## 1. 핵심 기여 / Core Contribution

본 논문은 태양 활동 지수인 F10.7 (10.7 cm 파장, 2800 MHz)과 F30 (30 cm 파장, 1 GHz)의 중기(1~60일) 일일 예보를 위한 deep learning 모델 **SINet (Solar Index Network)**을 제안한다. SINet은 TimesNet 아키텍처를 태양 지수 예측에 맞게 개선한 모델로, FFT를 통해 시계열 데이터의 주기적 패턴을 추출한 뒤 2D convolution으로 처리하는 독특한 접근법을 사용한다. 핵심 결과로, SINet의 고정 예측(SINet_f) 변형이 ARIMA, LSTM, CNN, LSTM+, TCN 등 5가지 기존 방법을 F10.7 예측에서 일관되게 능가했으며, 특히 태양 극대기(2014년)에 TCN 대비 RMSE 15.14%, MAPE 8.81% 향상을 달성했다. 또한 **F30에 대한 최초의 deep learning 기반 예측 방법**을 제시했다는 점에서 큰 의의가 있다.

This paper proposes **SINet (Solar Index Network)**, a deep learning model for medium-term (1–60 day) daily forecasting of solar activity indices F10.7 (10.7 cm wavelength, 2800 MHz) and F30 (30 cm wavelength, 1 GHz). SINet is an adaptation of the TimesNet architecture tailored for solar index prediction, employing a distinctive approach that extracts periodic patterns from time-series data via FFT and processes them with 2D convolutions. The key finding is that SINet's fixed prediction variant (SINet_f) consistently outperforms five comparison methods—ARIMA, LSTM, CNN, LSTM+, and TCN—for F10.7 prediction. Notably, during the solar maximum year of 2014, SINet_f improved over TCN by 15.14% in RMSE and 8.81% in MAPE. Furthermore, this work represents the **first deep learning method for F30 prediction**, marking a significant contribution to operational space weather forecasting.

---

## 2. 읽기 노트 / Reading Notes

### 2.1 서론 / Introduction (Section 1)

**태양 전파 플럭스 지수의 중요성 / Importance of Solar Radio Flux Indices**

F10.7과 F30은 지상 관측으로 측정하는 태양 활동 지수로, 단위는 sfu (solar flux unit)이다.

F10.7 and F30 are solar activity indices measured by ground-based observations, expressed in solar flux units (sfu).

$$1 \text{ sfu} = 10^{-22} \text{ W} \cdot \text{m}^{-2} \cdot \text{Hz}^{-1}$$

- **F10.7** (10.7 cm = 2800 MHz): 태양 UV 방사선이 상층 대기에 미치는 영향을 추적하는 주요 지표. 1947년부터 캐나다 국립연구위원회(NRC)에서 연속 관측. / Key proxy for tracking UV radiation impact on the upper atmosphere. Continuously observed since 1947 by Canada's NRC.
- **F30** (30 cm = 1 GHz): F10.7보다 더 민감한 지수로, 열권 밀도 모델링 개선에 기여 가능. 일본 Toyokawa/Nobeyama 관측소에서 측정. / More sensitive index than F10.7, potentially improving thermospheric density modeling. Measured at Japan's Toyokawa/Nobeyama observatories.

**기존 방법의 한계 / Limitations of Existing Methods**

기존 방법들은 예보 기간이 대부분 27일 이하로 제한되어 있었다. 예보 기간 대 간격 비(forecast horizon:cadence ratio)를 비교하면:

Most existing methods were limited to forecast horizons of 27 days or less. Comparing the forecast horizon:cadence ratio:

| 방법 / Method | 예보 기간 / Horizon | 비율 / Ratio |
|---|---|---|
| Informer (K. Zhang et al. 2024) | 27일 / 27 days | 27:1 |
| Kalman Filter (Petrova 2024) | 24일 / 24 days | 24:1 |
| **SINet (본 논문 / this paper)** | **60일 / 60 days** | **60:1** |

SINet은 기존 최대치의 2배 이상인 60:1 비율을 달성하여 중기 예보 영역을 크게 확장했다.

SINet achieves a 60:1 ratio, more than doubling the previous best and significantly extending the medium-term forecasting range.

### 2.2 데이터 / Data (Section 2)

**데이터 소스 및 분할 / Data Sources and Splitting**

- F10.7: NOAA 제공, 1957~2021년 / From NOAA, 1957–2021
- F30: Toyokawa/Nobeyama 관측소, 1957~2021년 / Toyokawa/Nobeyama observatories, 1957–2021
- 학습(Training): 1957~2008 (검증 10% 분리) / Training: 1957–2008 (10% held out for validation)
- 테스트(Test): 2009~2021 (학습과 **완전 분리**) / Test: 2009–2021 (**disjoint** from training)

**Min-max 정규화 / Min-Max Normalization**

모든 입력 데이터는 [0, 1] 범위로 정규화된다:

All input data are normalized to the [0, 1] range:

$$X_{\text{norm}} = \frac{X - X_{\min}}{X_{\max} - X_{\min}}$$

- $X$: 원래 값 / original value
- $X_{\min}$: 학습 데이터의 최솟값 / minimum in training data
- $X_{\max}$: 학습 데이터의 최댓값 / maximum in training data

**두 가지 예측 방식 / Two Prediction Approaches**

1. **고정 예측 (SINet_f / Fixed Prediction)**: 과거 30일(d-29 ~ d)을 입력으로, 미래 60일(d+1 ~ d+60)을 **한 번에** 출력. / Input 30 days (d-29 to d), output all 60 days (d+1 to d+60) **simultaneously**.
2. **순환 예측 (SINet_r / Rolling Prediction)**: 슬라이딩 윈도우 방식으로 1일씩 예측. 2~60일차는 이전 예측값을 입력으로 재사용. / Sliding window predicting 1 day at a time. Steps 2–60 reuse previous predictions as input.

### 2.3 모델 구조 / Model Architecture (Section 3)

**SINet = TimesNet의 태양 지수 예측용 개선 / SINet = TimesNet Enhanced for Solar Index Prediction**

SINet은 Wu et al. (2023)의 TimesNet을 기반으로 세 가지 핵심 수정을 적용한 모델이다.

SINet is based on TimesNet by Wu et al. (2023) with three key modifications.

**TimesBlock 구조 / TimesBlock Architecture**

SINet은 2개의 TimesBlock을 순차적으로 쌓아 구성된다. 각 TimesBlock의 처리 과정은 다음과 같다:

SINet stacks 2 TimesBlocks sequentially. Each TimesBlock processes data as follows:

**Step 1: FFT 기반 주기 추출 / FFT-Based Period Extraction**

1D 입력 시계열에 Fast Fourier Transform을 적용하여 가장 유의한 $k=3$개의 주기 성분을 추출한다.

FFT is applied to the 1D input time series to extract the $k=3$ most significant periodic components.

$$\mathbf{A} = \text{Avg}\left(\text{Amp}\left(\text{FFT}(\mathbf{X}^{1D})\right)\right)$$

$$\{f_1, f_2, \ldots, f_k\} = \arg\text{Topk}_{f_* \in \{1, \ldots, \lfloor T/2 \rfloor\}}(\mathbf{A})$$

$$p_i = \left\lceil \frac{T}{f_i} \right\rceil, \quad i \in \{1, 2, \ldots, k\}$$

- $\mathbf{A}$: 주파수 진폭의 평균 / averaged frequency amplitudes
- $f_i$: $i$번째로 큰 진폭에 해당하는 주파수 / frequency corresponding to $i$-th largest amplitude
- $T$: 시계열 길이 (= 30) / time series length (= 30)
- $p_i$: $i$번째 주기 / $i$-th period
- $k = 3$: 선택할 주기 수 / number of periods to select

**Step 2: 1D → 2D 변환 / 1D to 2D Reshaping**

추출된 각 주기 $p_i$에 따라 1D 시계열을 2D 텐서로 재구성한다:

The 1D time series is reshaped into a 2D tensor according to each detected period $p_i$:

$$\mathbf{X}^{2D}_i = \text{Reshape}_{p_i, f_i}(\text{Padding}(\mathbf{X}^{1D})), \quad i \in \{1, 2, \ldots, k\}$$

- $\text{Padding}$: 주기 정수배에 맞추기 위한 zero-padding / zero-padding to match integer multiples of the period
- $\text{Reshape}_{p_i, f_i}$: $(p_i \times f_i)$ 크기의 2D 텐서로 변환 / reshape into $(p_i \times f_i)$ 2D tensor

**Step 3: Dual-Inception Model (2D Convolution)**

각 2D 텐서는 dual-inception model로 처리된다. 이 모델은 두 개의 inception block이 Gelu 활성화 함수로 연결된 구조이다.

Each 2D tensor is processed by a dual-inception model consisting of two inception blocks connected by a Gelu activation function.

**첫 번째 Inception Block (차원 증가) / First Inception Block (Dimension Increase):**
- 입력 / Input: $30 \times C$ ($C = 32$)
- 6개의 InConvBlock, 각각 다른 커널 크기 / 6 InConvBlocks with different kernel sizes:
  - $1 \times 1$, $3 \times 3$, $5 \times 5$, $7 \times 7$, $9 \times 9$, $11 \times 11$
- 출력 / Output: $30 \times 64$

**Gelu 활성화 함수 / Gelu Activation:**

$$\text{Gelu}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]$$

- $\Phi(x)$: 표준 정규 분포의 누적 분포 함수 / CDF of the standard normal distribution

**두 번째 Inception Block (차원 감소) / Second Inception Block (Dimension Decrease):**
- 입력 / Input: $30 \times 64$
- 6개의 OutConvBlock, 동일한 커널 크기 / 6 OutConvBlocks with same kernel sizes
- 출력 / Output: $30 \times 32$

**Step 4: 적응적 집계 / Adaptive Aggregation**

FFT 진폭에 softmax를 적용하여 각 주기 성분의 가중치를 계산하고, 가중 합으로 최종 출력을 생성한다:

Softmax is applied to FFT amplitudes to compute weights for each periodic component, producing the final output via weighted sum:

$$\hat{\mathbf{A}}_{f_1}, \hat{\mathbf{A}}_{f_2}, \ldots, \hat{\mathbf{A}}_{f_k} = \text{Softmax}\left(\mathbf{A}_{f_1}, \mathbf{A}_{f_2}, \ldots, \mathbf{A}_{f_k}\right)$$

$$\mathbf{X}^{1D} = \sum_{i=1}^{k} \hat{\mathbf{A}}_{f_i} \times \text{Trunc}\left(\text{Reshape}_{1D}\left(\hat{\mathbf{X}}^{2D}_i\right)\right)$$

- $\hat{\mathbf{A}}_{f_i}$: $i$번째 주기 성분의 softmax 가중치 / softmax weight for the $i$-th periodic component
- $\text{Trunc}$: padding 제거 / removal of padding
- $\text{Reshape}_{1D}$: 2D를 1D로 복원 / flatten 2D back to 1D

**Step 5: 잔차 연결 / Residual Connection**

최종 출력에 원래 입력을 더하여 잔차 연결을 수행한다:

A residual connection adds the original input to the final output:

$$\mathbf{X}_{\text{out}} = \mathbf{X}_{\text{in}} + \mathbf{X}^{1D}_{\text{processed}}$$

**TimesNet과의 3가지 차이점 / Three Differences from TimesNet**

| 항목 / Feature | TimesNet | SINet |
|---|---|---|
| 주기 수 ($k$) / Num. periods | 5 | **3** |
| 손실 함수 / Loss function | SMAPE | **MSE** |
| 데이터 유형 / Data type | 다변량 / Multivariate | **단변량 / Univariate** |
| 모델 차원 / Model dimension | 32 | **64** |

$k=3$으로 줄임으로써 모델 복잡도를 낮추면서도 성능을 유지했다. MSE 손실 함수는 태양 지수 예측에 더 적합한 것으로 확인되었다.

Reducing $k$ to 3 lowers model complexity while maintaining performance. MSE loss was found more suitable for solar index prediction.

**학습 설정 / Training Configuration**

| 하이퍼파라미터 / Hyperparameter | 값 / Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Dropout | 0.3 |
| Batch size | 32 |
| Epochs | 10 |
| GPU | NVIDIA A100 |

하이퍼파라미터 튜닝은 scikit-learn의 grid search를 사용했다.

Hyperparameter tuning was performed via grid search using scikit-learn.

### 2.4 평가 지표 / Evaluation Metrics (Section 4)

세 가지 지표를 사용하며, MAPE를 주요 비교 기준으로 삼는다:

Three metrics are used, with MAPE as the primary comparison criterion:

**RMSE (Root Mean Squared Error):**

$$\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i - y_i)^2}$$

**MAE (Mean Absolute Error):**

$$\text{MAE} = \frac{1}{N}\sum_{i=1}^{N}|\hat{y}_i - y_i|$$

**MAPE (Mean Absolute Percentage Error):**

$$\text{MAPE} = \frac{1}{N}\sum_{i=1}^{N}\frac{|\hat{y}_i - y_i|}{y_i} \times 100\%$$

- $\hat{y}_i$: 예측값 / predicted value
- $y_i$: 관측값 / observed value
- $N$: 샘플 수 / number of samples

### 2.5 F10.7 예측 결과 / F10.7 Prediction Results (Section 4.1–4.2)

**5-fold 교차 검증 MAPE (%) / 5-Fold Cross-Validation MAPE (%)**

| 예보일 / Day | SINet_f | SINet_r | ARIMA | LSTM | CNN | LSTM+ | TCN |
|---|---|---|---|---|---|---|---|
| 1일 / 1st | **2.3** | **2.3** | 2.4 | 2.8 | 3.0 | 2.5 | 2.4 |
| 27일 / 27th | **8.0** | 8.8 | 9.1 | 8.9 | 8.5 | 8.7 | 8.6 |
| 45일 / 45th | **9.1** | 9.3 | 11.3 | 11.0 | 9.6 | 9.6 | 9.4 |
| 60일 / 60th | **10.1** | 10.9 | 11.6 | 11.1 | 10.7 | 10.4 | 10.4 |

SINet_f는 모든 예보 기간에서 최저 MAPE를 달성했다. 특히 27일차에서 차이가 두드러지며, 2위인 TCN(8.6%) 대비 0.6%p 향상되었다.

SINet_f achieves the lowest MAPE at all forecast horizons. The gap is especially notable at day 27, with a 0.6 percentage point improvement over the runner-up TCN (8.6%).

**SINet_f 예보 기간별 상세 결과 (Fold 1, 2009–2021) / SINet_f Detailed Results by Horizon (Fold 1, 2009–2021):**

| 예보일 / Day | RMSE (sfu) | MAE (sfu) | MAPE (%) |
|---|---|---|---|
| 1일 / 1st | 3.88 | 2.38 | 2.2 |
| 27일 / 27th | 13.74 | 8.76 | 8.1 |
| 45일 / 45th | 15.09 | 9.82 | 9.1 |
| 60일 / 60th | 16.32 | 10.92 | 10.2 |

### 2.6 F30 예측 결과 / F30 Prediction Results (Section 4.3)

**5-fold 교차 검증 MAPE (%) / 5-Fold Cross-Validation MAPE (%)**

| 예보일 / Day | SINet_f | SINet_r | ARIMA | LSTM | CNN | LSTM+ | TCN |
|---|---|---|---|---|---|---|---|
| 1일 / 1st | **2.0** | **2.0** | 2.1 | 2.6 | 3.4 | 2.4 | 2.4 |
| 27일 / 27th | **7.1** | 7.9 | 7.6 | 8.5 | 8.2 | 7.6 | 7.5 |
| 45일 / 45th | **8.7** | 8.9 | 10.5 | 10.6 | 9.0 | 9.9 | 9.2 |
| 60일 / 60th | **9.5** | 10.2 | 10.7 | 10.3 | 10.2 | 9.6 | 9.6 |

이것은 **F30에 대한 최초의 deep learning 예측 결과**이다. F10.7 대비 전반적으로 낮은 MAPE를 보이는데, 이는 F30 데이터의 변동성이 상대적으로 작기 때문이다.

These are the **first deep learning prediction results for F30**. The overall lower MAPE compared to F10.7 reflects the relatively lower variability of F30 data.

**SINet_f 예보 기간별 상세 결과 (F30, Fold 1) / SINet_f Detailed Results by Horizon (F30, Fold 1):**

| 예보일 / Day | RMSE (sfu) | MAE (sfu) | MAPE (%) |
|---|---|---|---|
| 1일 / 1st | 2.05 | 1.40 | 2.0 |
| 27일 / 27th | 7.65 | 5.17 | 7.1 |
| 45일 / 45th | 9.16 | 6.25 | 8.7 |
| 60일 / 60th | 9.99 | 6.81 | 9.5 |

### 2.7 태양 극대기 성능 / Solar Maximum Performance (Section 4.4)

**2014년 태양 극대기, 60일 예보 / 2014 Solar Maximum, 60-Day Forecast**

태양 극대기에는 플럭스 변동이 커서 예측이 어려운데, SINet_f는 이 기간에 특히 큰 개선을 보인다.

Prediction is more challenging during solar maximum due to larger flux variability, and SINet_f shows particularly significant improvement during this period.

**F10.7 (2014):**

| 모델 / Model | RMSE (sfu) | MAE (sfu) | MAPE (%) |
|---|---|---|---|
| SINet_f | **26.52** | **20.94** | **14.5** |
| TCN | 31.25 | 24.19 | 15.9 |
| 개선율 / Improvement | 15.14% | 13.44% | 8.81% |

**F30 (2014):**

| 모델 / Model | RMSE (sfu) | MAE (sfu) | MAPE (%) |
|---|---|---|---|
| SINet_f | **14.53** | **11.23** | **11.0** |
| TCN | 19.32 | 15.72 | 14.7 |
| 개선율 / Improvement | 24.79% | 28.56% | 25.17% |

F30에서 개선이 더 크다. TCN 대비 RMSE 24.79%, MAPE 25.17% 향상은 SINet_f의 FFT 기반 주기 추출이 태양 극대기의 복잡한 변동성을 더 잘 포착함을 시사한다.

The improvement is more pronounced for F30. The 24.79% RMSE and 25.17% MAPE improvement over TCN suggests that SINet_f's FFT-based period extraction better captures the complex variability during solar maximum.

### 2.8 사례 연구: NOAA AR 12673 / Case Study: NOAA AR 12673 (Section 4.5)

2017년 9월의 super-flaring 활동 영역 NOAA AR 12673은 X9.3 플레어를 포함하는 극단적 사건이다. 이 기간(Figure 10에서 9월 4~10일) 예측 오차가 급격히 증가했다.

NOAA AR 12673 in September 2017 was a super-flaring active region including an X9.3 flare—an extreme event. During this period (September 4–10 in Figure 10), prediction errors spike sharply.

모델이 이 사건을 잘 예측하지 못하는 이유는 학습 데이터에 이처럼 급격한 자기 플럭스 출현(magnetic flux emergence) 사례가 매우 적기 때문이다. 이는 극단적 우주 날씨 사건 예측의 근본적 한계를 보여준다.

The model struggles with this event because very few training samples exist for such rapid magnetic flux emergence. This highlights a fundamental limitation in predicting extreme space weather events.

### 2.9 운영 배포 / Operational Deployment (Section 4.6)

SINet_f는 NJIT의 운영 시스템 (`nature.njit.edu/solardb/`)에 배포되었다. 2025년 3~6월 NOAA/SWPC 시스템과 비교한 결과:

SINet_f has been deployed at NJIT's operational system (`nature.njit.edu/solardb/`). Comparison with NOAA/SWPC system during March–June 2025:

| 시스템 / System | MAPE (%) |
|---|---|
| SINet_f | **13.8** |
| NOAA/SWPC | 14.0 |

실제 운영 환경에서도 SINet_f가 NOAA/SWPC 공식 시스템과 경쟁력 있는(소폭 우수한) 성능을 보인다.

In real operational conditions, SINet_f shows competitive (slightly superior) performance against the official NOAA/SWPC system.

### 2.10 논의 / Discussion (Section 5)

**입력 길이 선택 / Input Length Selection**

30일 입력은 약 1 태양 자전 주기에 해당한다. 태양 변동성은 7개월 impulse function을 가지지만, 실험 결과 2~4개월 입력은 유사한 결과를 내면서 학습 시간이 크게 증가했다.

The 30-day input corresponds to approximately one solar rotation period. Solar variability has a 7-month impulse function, but experiments showed 2–4 month inputs produced similar results with much longer training times.

**FFT 성분 수 ($k$) / Number of FFT Components ($k$)**

- $k = 1, 2$: 성능 저하 / degraded performance
- $k = 3$: 최적 / optimal
- $k > 3$: 유사한 성능, 더 긴 학습 시간 / similar performance, longer training time

**TCN과의 비교 분석 / Comparative Analysis with TCN**

SINet_f가 2009~2021년 전체 평균에서는 최고 성능을 보이지만, 일부 저변동성 연도(예: 2016년)에서는 TCN이 더 좋은 성능을 보인다. TCN은 시간적 패턴 포착에 강점이 있어 저변동성 기간에 유리하고, SINet_f는 주기적 패턴 추출에 강점이 있어 극대기/극소기에 유리하다.

While SINet_f shows the best average performance over 2009–2021, TCN performs better in some low-variability years (e.g., 2016). TCN excels at capturing temporal patterns during low-variability periods, while SINet_f excels at extracting periodic patterns during solar maximum/minimum.

**자기상관 분석 / Autocorrelation Analysis**

모든 방법이 lag 1에서 거의 1.0에 가까운 높은 자기상관을 보인다. 이는 태양 복사가 하루 만에 크게 변하지 않는 물리적 특성을 반영한다. 더 큰 lag에서는 자기상관이 감소한다.

All methods show high autocorrelation near 1.0 at lag 1. This reflects the physical property that solar irradiance does not change significantly in one day. Autocorrelation decreases at larger lags.

---

## 3. 핵심 시사점 / Key Takeaways

1. **FFT 기반 주기 추출의 효과 / Effectiveness of FFT-Based Period Extraction**: SINet은 FFT로 시계열의 지배적 주기를 추출한 뒤 2D convolution으로 처리하는 독특한 접근법을 사용한다. 이 방법이 태양 활동의 주기적 특성(자전 주기 ~27일 등)을 포착하는 데 효과적임을 입증했다. / SINet uses a distinctive approach of extracting dominant periods via FFT then processing with 2D convolution. This proves effective at capturing the periodic characteristics of solar activity (rotation period ~27 days, etc.).

2. **고정 예측이 순환 예측보다 우수 / Fixed Prediction Outperforms Rolling Prediction**: SINet_f가 SINet_r보다 일관되게 우수한 성능을 보인다. 순환 예측은 오차가 누적되는 반면, 고정 예측은 60일을 한 번에 출력하여 오차 전파를 방지한다. / SINet_f consistently outperforms SINet_r. Rolling prediction accumulates errors, while fixed prediction outputs all 60 days at once, preventing error propagation.

3. **태양 극대기에서의 현저한 개선 / Significant Improvement During Solar Maximum**: 2014년 극대기에 F30 예측에서 TCN 대비 RMSE 24.79%, MAPE 25.17% 향상은 실용적으로 매우 중요하다. 극대기가 우주 날씨 예보에서 가장 중요한 시기이기 때문이다. / The 24.79% RMSE and 25.17% MAPE improvement over TCN for F30 during the 2014 solar maximum is practically significant, as solar maximum is the most critical period for space weather forecasting.

4. **F30 최초의 deep learning 예측 / First Deep Learning Prediction for F30**: F10.7보다 더 민감한 F30 지수의 deep learning 예측을 처음 시도하여, 열권 밀도 모델링 개선의 가능성을 열었다. / The first deep learning prediction of the more sensitive F30 index opens possibilities for improved thermospheric density modeling.

5. **60일 예보 범위 달성 / Achievement of 60-Day Forecast Range**: 기존 최대 27일에서 60일로 예보 범위를 2배 이상 확장했다. 이는 중기 우주 날씨 예보의 실용적 가치를 크게 높인다. / The forecast range was more than doubled from the previous maximum of 27 days to 60 days, significantly increasing the practical value of medium-term space weather forecasting.

6. **극단적 사건의 한계 인정 / Acknowledging Limitations with Extreme Events**: AR 12673 사례 연구에서 급격한 자기 플럭스 출현에 대한 예측 한계를 솔직히 인정했다. 학습 데이터에 이런 사건이 적다는 근본적 문제를 지적한다. / The AR 12673 case study honestly acknowledges prediction limitations for rapid magnetic flux emergence. It points out the fundamental problem of few such events in training data.

7. **운영 시스템 수준의 검증 / Operational System-Level Validation**: NJIT 운영 시스템에 배포 후 NOAA/SWPC와 비교하여 경쟁력을 확인(13.8% vs 14.0% MAPE)한 것은 학술 연구에서 실용 단계로의 전환을 보여준다. / Deployment at NJIT's operational system and validation against NOAA/SWPC (13.8% vs 14.0% MAPE) demonstrates the transition from academic research to practical application.

8. **모델 단순성과 효율성 / Model Simplicity and Efficiency**: 단 10 epoch, $k=3$ FFT 성분, 2개 TimesBlock으로 구성된 상대적으로 간단한 구조로 최고 성능을 달성했다. 복잡한 모델이 항상 더 좋은 것은 아님을 보여준다. / Achieving top performance with a relatively simple architecture of only 10 epochs, $k=3$ FFT components, and 2 TimesBlocks shows that more complex models are not always better.

---

## 4. 수학적 요약 / Mathematical Summary

### 4.1 데이터 정규화 / Data Normalization

$$X_{\text{norm}} = \frac{X - X_{\min}}{X_{\max} - X_{\min}}$$

- $X$: 원래 관측값 (sfu) / original observed value (sfu)
- $X_{\min}, X_{\max}$: 학습 세트의 최소/최대값 / min/max from training set

### 4.2 FFT 주기 추출 / FFT Period Extraction

$$\mathbf{A} = \text{Avg}\left(\text{Amp}\left(\text{FFT}(\mathbf{X}^{1D})\right)\right)$$

- $\mathbf{X}^{1D}$: 1D 입력 시계열 (길이 $T=30$) / 1D input time series (length $T=30$)
- $\text{FFT}$: 고속 푸리에 변환 / Fast Fourier Transform
- $\text{Amp}$: 진폭 스펙트럼 / amplitude spectrum
- $\text{Avg}$: 채널 간 평균 / average across channels

$$p_i = \left\lceil \frac{T}{f_i} \right\rceil$$

- $p_i$: $i$번째 주기 (일 단위) / $i$-th period (in days)
- $f_i$: $i$번째 주파수 / $i$-th frequency

### 4.3 2D 변환 / 2D Transformation

$$\mathbf{X}^{2D}_i = \text{Reshape}_{p_i, f_i}\left(\text{Padding}\left(\mathbf{X}^{1D}\right)\right)$$

- $\text{Padding}$: 시계열 길이를 $p_i$의 정수배로 맞추는 zero-padding / zero-padding to make time series length a multiple of $p_i$
- 결과 크기: $p_i \times f_i$ / resulting size: $p_i \times f_i$

### 4.4 적응적 집계 / Adaptive Aggregation

$$\hat{\mathbf{A}}_{f_1}, \ldots, \hat{\mathbf{A}}_{f_k} = \text{Softmax}\left(\mathbf{A}_{f_1}, \ldots, \mathbf{A}_{f_k}\right)$$

$$\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{k} e^{z_j}}$$

$$\mathbf{X}^{1D}_{\text{out}} = \sum_{i=1}^{k} \hat{\mathbf{A}}_{f_i} \times \text{Trunc}\left(\text{Reshape}_{1D}\left(\hat{\mathbf{X}}^{2D}_i\right)\right)$$

- $\hat{\mathbf{A}}_{f_i}$: $i$번째 주기의 softmax 정규화된 가중치 / softmax-normalized weight for $i$-th period
- $\text{Trunc}$: padding 부분 제거 / removal of padded portion
- $k = 3$: 주기 성분 수 / number of periodic components

### 4.5 Gelu 활성화 함수 / Gelu Activation Function

$$\text{Gelu}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]$$

- $\Phi(x)$: 표준 정규 분포의 누적 분포 함수 / cumulative distribution function of standard normal
- $\text{erf}$: 오차 함수 / error function

### 4.6 평가 지표 / Evaluation Metrics

$$\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i - y_i)^2}$$

$$\text{MAE} = \frac{1}{N}\sum_{i=1}^{N}|\hat{y}_i - y_i|$$

$$\text{MAPE} = \frac{1}{N}\sum_{i=1}^{N}\frac{|\hat{y}_i - y_i|}{y_i} \times 100\%$$

- $\hat{y}_i$: $i$번째 예측값 (sfu) / $i$-th predicted value (sfu)
- $y_i$: $i$번째 관측값 (sfu) / $i$-th observed value (sfu)
- $N$: 테스트 샘플 수 / number of test samples

### 4.7 MSE 손실 함수 / MSE Loss Function

$$\mathcal{L}_{\text{MSE}} = \frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i - y_i)^2$$

TimesNet은 SMAPE를 사용하지만, SINet은 MSE를 사용한다. MSE는 큰 오차에 더 큰 페널티를 부여하여 극단값 예측에 유리하다.

TimesNet uses SMAPE, but SINet uses MSE. MSE penalizes large errors more heavily, favoring extreme value prediction.

### 4.8 구체적 수치 예시 / Concrete Numerical Example

**F10.7 1일 예보 오차 계산 예시 / F10.7 1-Day Forecast Error Calculation Example**

2009~2021년 테스트 기간의 F10.7 평균이 약 108 sfu라고 가정하면:

Assuming the mean F10.7 during the 2009–2021 test period is approximately 108 sfu:

- SINet_f 1일 예보: MAE = 2.38 sfu, MAPE = 2.2%
- 검증: $\text{MAPE} \approx \frac{2.38}{108} \times 100\% \approx 2.2\%$ ✓

60일 예보에서:
At 60-day forecast:
- MAE = 10.92 sfu → 하루 평균 ~10.92 sfu의 오차 / average daily error of ~10.92 sfu
- 이는 평균 F10.7 대비 약 10.1%에 해당 / corresponds to approximately 10.1% of mean F10.7
- RMSE = 16.32 > MAE = 10.92 → 일부 큰 오차가 존재함을 시사 / RMSE > MAE suggests some large outlier errors exist

**2014년 극대기 개선률 계산 / 2014 Solar Maximum Improvement Calculation**

F10.7, RMSE 개선률:
$$\text{Improvement} = \frac{31.25 - 26.52}{31.25} \times 100\% = \frac{4.73}{31.25} \times 100\% = 15.14\%$$

F30, MAPE 개선률:
$$\text{Improvement} = \frac{14.7 - 11.0}{14.7} \times 100\% = \frac{3.7}{14.7} \times 100\% = 25.17\%$$

---

## 5. 역사 속의 논문 / Paper in the Arc of History

```
태양 지수 예측 역사 / History of Solar Index Prediction
==========================================================

1947    F10.7 관측 시작 (캐나다 NRC)
        F10.7 observations begin (Canada NRC)
        |
1957    F30 관측 시작 (일본 Toyokawa)
        F30 observations begin (Japan Toyokawa)
        |
~2000   전통적 통계 기법: 회귀, ARIMA 등
        Classical statistical methods: regression, ARIMA, etc.
        |
2018    LSTM 기반 예보 등장 (Liu et al.)
        LSTM-based forecasting emerges
        |
2022    Zhu et al. — LSTM for F10.7, 27일 예보
        Zhu et al. — LSTM for F10.7, 27-day forecast
        |
2023    Wu et al. — TimesNet 아키텍처 발표
        Wu et al. — TimesNet architecture published
        |
2024    Wang et al. — TCN for F10.7, 60일 예보
        Wang et al. — TCN for F10.7, 60-day forecast
        |
        Jerse & Marcucci — LSTM+ for F10.7
        |
        K. Zhang et al. — Informer for F10.7, 27일 예보
        K. Zhang et al. — Informer for F10.7, 27-day forecast
        |
2026 ★  Wang et al. — SINet (본 논문)
        ├─ F10.7 60일 예보: MAPE 10.1% (최고 성능)
        │  F10.7 60-day: MAPE 10.1% (best performance)
        ├─ F30 최초 deep learning 예측
        │  First deep learning prediction for F30
        └─ 운영 시스템 배포 (NJIT)
           Operational deployment (NJIT)
        |
미래    다변량 입력 (SDO 이미지 등) 활용 가능성
Future  Potential for multivariate inputs (SDO images, etc.)
```

---

## 6. 다른 논문과의 연결 / Connections to Other Papers

| 논문 / Paper | 연결 / Connection | 방향 / Direction |
|---|---|---|
| Wu et al. (2023) — TimesNet | SINet의 기반 아키텍처. $k$, 손실 함수, 모델 차원을 변경. / Base architecture for SINet. Modified $k$, loss function, model dimension. | 직접 기반 / Direct foundation |
| Wang et al. (2024) — TCN | SINet의 주요 비교 대상. 같은 저자 그룹의 이전 연구. / Primary comparison target. Previous work by same author group. | 개선 대상 / Improved upon |
| Zhu et al. (2022) — LSTM | LSTM 기반 F10.7 27일 예보. 비교 대상 방법 중 하나. / LSTM-based F10.7 27-day forecast. One of the comparison methods. | 비교 / Compared |
| Jerse & Marcucci (2024) — LSTM+ | LSTM 개선 모델. 비교 대상 방법 중 하나. / Enhanced LSTM model. One of the comparison methods. | 비교 / Compared |
| K. Zhang et al. (2024) — Informer | Transformer 기반 F10.7 27일 예보. 예보 범위에서 SINet이 2배 이상 우수. / Transformer-based F10.7 27-day forecast. SINet more than doubles the forecast range. | 비교 / Compared |
| Petrova (2024) — Kalman Filter | 통계적 방법의 24일 예보. 방법론적으로 대조적 접근. / Statistical method with 24-day forecast. Methodologically contrasting approach. | 간접 비교 / Indirectly compared |
| Dudok de Wit et al. (2014) | F30과 F10.7의 관계 연구. F30의 우월성 근거 제공. / Study of F30–F10.7 relationship. Provides rationale for F30's superiority. | 동기 부여 / Motivation |
| Abduallah et al. (2024) — Paper #38 | 같은 연구 그룹의 태양 플레어 예측 deep learning 연구. 방법론적 연관성. / Same research group's deep learning work on solar flare prediction. Methodological kinship. | 같은 그룹 / Same group |

---

## 7. 참고문헌 / References

- Wang, Z., Abduallah, Y., Wang, J. T. L., Wang, H., Xu, Y., Yurchyshyn, V., Oria, V., Alobaid, K. A., & Bai, X. (2026). "Daily Predictions of F10.7 and F30 Solar Indices With Deep Learning." *Journal of Geophysical Research: Space Physics*. DOI: 10.1029/2025JA034868
- Wu, H., Hu, T., Liu, Y., Zhou, H., Wang, J., & Long, M. (2023). "TimesNet: Temporal 2D-variation modeling for general time series analysis." *ICLR 2023*.
- Wang, Z., et al. (2024). "TCN-based solar index prediction." *(Previous work by same group on F10.7 60-day forecasting)*
- Zhu, G., et al. (2022). "LSTM-based F10.7 prediction." *(27-day forecast horizon)*
- Jerse, G., & Marcucci, A. (2024). "LSTM+ for F10.7 prediction."
- Zhang, K., et al. (2024). "Informer-based F10.7 prediction." *(27-day forecast with Transformer architecture)*
- Petrova, E. (2024). "Kalman filter for solar index prediction." *(24-day forecast horizon)*
- Dudok de Wit, T., et al. (2014). "Synoptic radio observations as proxies for upper atmosphere modeling."
- Abduallah, Y., et al. (2024). "Deep learning for solar flare prediction." *(Same NJIT research group)*
