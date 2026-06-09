---
title: "Pre-Reading Briefing: Daily Predictions of F10.7 and F30 Solar Indices With Deep Learning"
paper_id: "39_wang_2026"
topic: Space_Weather
date: 2026-04-15
type: briefing
---

# Daily Predictions of F10.7 and F30 Solar Indices With Deep Learning: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Wang, Z., Abduallah, Y., Wang, J. T. L., Wang, H., Xu, Y., Yurchyshyn, V., Oria, V., Alobaid, K. A., & Bai, X. (2026). Daily Predictions of F10.7 and F30 Solar Indices With Deep Learning. *Journal of Geophysical Research: Space Physics*, 131, e2025JA034868.
**Author(s)**: Zhenduo Wang, Yasser Abduallah, Jason T. L. Wang, Haimin Wang, Yan Xu, Vasyl Yurchyshyn, Vincent Oria, Khalid A. Alobaid, Xiaoli Bai
**Year**: 2026

---

## 1. 핵심 기여 / Core Contribution

이 논문은 **SINet(Solar Index Network)**이라는 새로운 딥러닝 모델을 제안하여, F10.7과 F30 태양 활동 지수를 **1~60일** 앞서 일별로 예측합니다. SINet은 TimesNet 아키텍처를 개량하여, FFT(Fast Fourier Transform)로 시계열의 주기적 성분을 주파수 영역에서 추출한 뒤, dual-inception 구조의 2D CNN으로 시간적 패턴을 학습합니다. F10.7 예측에서 5개 기존 방법(ARIMA, LSTM, CNN, LSTM+, TCN)을 능가했으며, **F30 예측에 딥러닝을 최초로 적용**한 연구입니다. 60일 예측 MAPE가 F10.7은 10.2%, F30은 9.5%로, 최고 성능의 비교 방법(TCN) 대비 각각 0.6%p, 0.1%p 낮은 우수한 결과를 달성했습니다.

This paper proposes **SINet (Solar Index Network)**, a novel deep learning model for daily prediction of F10.7 and F30 solar activity indices **1–60 days in advance**. SINet enhances the TimesNet architecture by using FFT to extract periodic components in the frequency domain, then learning temporal patterns via a dual-inception 2D CNN structure. It outperforms five existing methods (ARIMA, LSTM, CNN, LSTM+, TCN) on F10.7 prediction and is the **first deep learning approach applied to F30 prediction**. The 60-day forecast MAPE is 10.2% for F10.7 and 9.5% for F30, surpassing the best comparison method (TCN) by 0.6%p and 0.1%p respectively.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

F10.7(10.7 cm 파장 = 2800 MHz)은 1947년부터 측정된 가장 오래되고 널리 쓰이는 태양 활동 지수입니다. 태양 자외선(UV) 복사의 proxy로서 상층 대기 모델링과 우주기상 예보에 핵심적입니다. F30(30 cm 파장 = 1 GHz)은 더 민감한 지수로, 열권 밀도와 태양 자극 반응 연구에 유용하나 상대적으로 연구가 적었습니다.

F10.7 (10.7 cm wavelength = 2800 MHz) has been measured since 1947 and is the most widely used solar activity index. It serves as a proxy for solar ultraviolet (UV) radiation, critical for upper atmosphere modeling and space weather forecasting. F30 (30 cm wavelength = 1 GHz) is a more sensitive index useful for thermospheric density and solar stimulation studies, but has received comparatively less attention.

전통적 예측 방법으로는 선형 예측(Warren et al., 2017), adaptive Kalman filter(Petrova et al., 2021), McNish-Lincoln 방법 등이 있었습니다. 딥러닝 시대에는 LSTM(Zhu et al., 2022), TCN(Wang et al., 2024), Informer(K. Zhang et al., 2024) 등이 F10.7 예측에 적용되었지만, 대부분 27일 이내의 단기 예측에 머물렀고 F30에 대한 딥러닝 예측은 전무했습니다.

Traditional forecasting methods included linear forecasting (Warren et al., 2017), adaptive Kalman filter (Petrova et al., 2021), and the McNish-Lincoln method. In the deep learning era, LSTM (Zhu et al., 2022), TCN (Wang et al., 2024), and Informer (K. Zhang et al., 2024) were applied to F10.7, but mostly limited to short-term forecasts (≤27 days), and no deep learning method had been applied to F30.

### 타임라인 / Timeline

```
1947        F10.7 측정 시작 / F10.7 measurement begins
  |
1970        ARIMA (Box & Jenkins) — 시계열 통계 예측의 표준 / Standard time series statistical forecast
  |
2017        Warren et al. — F10.7 선형 예측 / Linear F10.7 forecast
  |
2021        Petrova et al. — Adaptive Kalman filter로 F10.7/F30 24개월 예측 / F10.7/F30 24-month forecast
  |
2022        Zhu et al. — LSTM으로 F10.7 예측 / LSTM-based F10.7 prediction
  |
2023        Wu et al. — TimesNet 제안 (일반 시계열) / TimesNet proposed (general time series)
  |
2024        K. Zhang et al. — Informer로 F10.7 1-27일 예측 / Informer for F10.7 1-27 day forecast
  |         Wang et al. — TCN으로 F10.7 단기 예측 / TCN for short-term F10.7
  |
2026  >>>   Wang et al. — SINet: F10.7/F30 1-60일 예측 (본 논문) / SINet: F10.7/F30 1-60 day forecast (this paper)
```

---

## 3. 필요한 배경 지식 / Prerequisites

### 태양 활동 지수 / Solar Activity Indices

- **F10.7 지수**: 태양의 10.7 cm (2800 MHz) 파장 전파 플럭스. 단위는 sfu (solar flux unit, 1 sfu = 10⁻²² W·m⁻²·Hz⁻¹). 태양 흑점 수와 강한 상관관계가 있으며, 상층 대기 UV 복사의 proxy로 사용됩니다.
- **F10.7 index**: Solar radio flux at 10.7 cm (2800 MHz). Unit: sfu (solar flux unit, 1 sfu = 10⁻²² W·m⁻²·Hz⁻¹). Strongly correlated with sunspot number; used as a proxy for upper atmospheric UV radiation.

- **F30 지수**: 태양의 30 cm (1 GHz) 파장 전파 플럭스. F10.7보다 민감하여 열권 밀도 변화에 대한 반응이 더 큽니다. 토요카와(Toyokawa)와 노베야마(Nobeyama) 시설에서 측정됩니다.
- **F30 index**: Solar radio flux at 30 cm (1 GHz). More sensitive than F10.7 to thermospheric density changes. Measured at Toyokawa and Nobeyama facilities.

### 시계열 예측 기초 / Time Series Forecasting Basics

- **고정 예측 (Fixed prediction)**: 과거 30일 데이터로 미래 60일을 한 번에 예측. 모든 예측값이 실제 관측값 기반.
- **Fixed prediction**: Predict 60 future days at once from 30 past days. All predictions are based on actual observations.

- **롤링 예측 (Rolling prediction)**: 슬라이딩 윈도우 방식으로 하루씩 전진하며 1일 예측을 반복. 이전 예측값을 다음 입력에 사용.
- **Rolling prediction**: Sliding window approach, advancing one day at a time, iteratively using previous predictions as inputs.

### TimesNet 아키텍처 / TimesNet Architecture

Wu et al. (2023)이 제안한 일반 시계열 분석 모델. **핵심 아이디어**: 1D 시계열을 FFT로 주파수 분해하여 주요 주기 성분을 추출한 뒤, 2D 텐서로 변환하여 2D convolution으로 처리. 이를 통해 시간적 패턴(단기+장기)을 동시에 포착합니다.

TimesNet, proposed by Wu et al. (2023), is a general time series analysis model. **Key idea**: Decompose a 1D time series via FFT to extract dominant periodic components, reshape into 2D tensors, and process with 2D convolutions. This captures both short- and long-term temporal patterns simultaneously.

### 전제 논문 / Prerequisite Papers from This Series

- **Paper #33** (Petrova et al., 2021): Adaptive Kalman filter로 F10.7/F30 중기 예측 — 본 논문이 비교 기준으로 참조
- **Paper #35** (관련 딥러닝 방법): 시계열 딥러닝 예측의 기초

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **SINet (Solar Index Network)** | 본 논문에서 제안한 딥러닝 모델. TimesNet을 태양 지수 예측에 맞게 개조 / The proposed deep learning model, adapted from TimesNet for solar index prediction |
| **F10.7** | 10.7 cm 태양 전파 플럭스 (sfu 단위). 태양 활동의 표준 proxy / 10.7 cm solar radio flux (in sfu). Standard solar activity proxy |
| **F30** | 30 cm 태양 전파 플럭스. F10.7보다 민감 / 30 cm solar radio flux. More sensitive than F10.7 |
| **sfu (solar flux unit)** | 태양 전파 플럭스 단위. 1 sfu = 10⁻²² W·m⁻²·Hz⁻¹ / Unit of solar radio flux |
| **TimesBlock** | SINet의 핵심 구성요소. FFT + Inception Block + Residual Connection / Core building block of SINet |
| **Dual-Inception Structure** | 2개의 inception block(InConv + OutConv)을 Gelu 활성화로 연결한 구조 / Two inception blocks connected by Gelu activation |
| **FFT (Fast Fourier Transform)** | 시간 영역 → 주파수 영역 변환. 주기적 성분 추출에 사용 / Time-to-frequency domain transform for extracting periodic components |
| **MAPE (Mean Absolute Percentage Error)** | 주요 성능 지표. 예측 오차를 실제값 대비 백분율로 표현 / Primary metric; expresses error as percentage of actual value |
| **Min-Max Normalization** | 데이터를 [0,1] 범위로 정규화: $X_{norm} = (X - X_{min})/(X_{max} - X_{min})$ / Scales data to [0,1] range |
| **5-Fold Cross Validation** | 시간 순서를 유지하며 5개 fold로 분할하여 모델 안정성 검증 / Temporal-order-preserving 5-fold split for robustness verification |
| **Autocorrelation** | 시계열의 자기상관. lag-k에서 예측값과 k일 전 예측값의 상관도 측정 / Self-correlation in time series at various lag values |
| **NOAA AR 12673** | 2017년 9월의 초활동 활성 영역. X9.3 플레어 발생. 모델 성능의 극한 테스트 사례 / Super-active region in Sep 2017; extreme test case for model performance |

---

## 5. 수식 미리보기 / Equations Preview

### 5.1 Min-Max 정규화 / Min-Max Normalization

$$X_{norm} = \frac{X - X_{min}}{X_{max} - X_{min}}$$

- $X$: 원본 일별 F10.7/F30 값 / Original daily F10.7/F30 value
- $X_{min}$, $X_{max}$: 해당 데이터셋의 최솟값, 최댓값 / Min and max of the dataset
- 결과: [0, 1] 범위로 정규화 / Result: normalized to [0, 1]

### 5.2 RMSE (Root Mean Square Error)

$$\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i - y_i)^2}$$

- $\hat{y}_i$: 예측값 / Predicted value
- $y_i$: 실제 관측값 / Actual observed value
- $N$: 테스트 샘플 수 / Number of test samples

### 5.3 MAE (Mean Absolute Error)

$$\text{MAE} = \frac{1}{N}\sum_{i=1}^{N}|\hat{y}_i - y_i|$$

### 5.4 MAPE (Mean Absolute Percentage Error) — 주요 지표 / Primary Metric

$$\text{MAPE} = \frac{1}{N}\sum_{i=1}^{N}\frac{|\hat{y}_i - y_i|}{y_i} \times 100\%$$

- 논문에서 가장 중시하는 성능 지표 / The paper's primary performance metric
- RMSE/MAE는 sfu 단위, MAPE는 백분율 / RMSE/MAE in sfu, MAPE in percentage

### 5.5 자기상관 / Autocorrelation at Lag k

$$\hat{\rho}_k = \frac{\sum_{t=k+1}^{N}(\hat{y}_t - \bar{\hat{y}})(\hat{y}_{t-k} - \bar{\hat{y}})}{\sum_{t=1}^{N}(\hat{y}_t - \bar{\hat{y}})^2}, \quad k = 1, 27, 45, 60$$

- $\hat{y}_t$: 시점 $t$에서의 예측값 / Predicted value at time $t$
- $\bar{\hat{y}}$: 예측 시계열의 평균 / Mean of the predicted series
- 1일 예측이 높은 자기상관을 보이는 이유를 분석하는 데 사용 / Used to analyze why 1-day forecasts show high autocorrelation

---

## 6. 읽기 가이드 / Reading Guide

### 읽기 순서 권장 / Recommended Reading Order

1. **Abstract + Plain Language Summary** (p.1): 전체 연구를 한 눈에 파악. "medium-term" = 1-60일, "forecast horizon:cadence ratio = 60:1"이라는 핵심 성과를 주목하세요.
   Read the full study at a glance. Note the key achievement: "forecast horizon:cadence ratio = 60:1."

2. **Section 1 – Introduction** (pp.1-2): 기존 방법들의 한계(단기 예측만 가능, F30 미지원)와 SINet의 차별점을 이해하세요.
   Understand limitations of existing methods and SINet's differentiation.

3. **Section 2 – Data** (pp.2-3): Figure 1의 F10.7/F30 시계열 데이터, 학습/검증/테스트 분할 방식, **고정 예측 vs 롤링 예측** (Figure 2)의 차이를 꼼꼼히 읽으세요. 이 두 예측 방식이 실험 전체를 관통합니다.
   Read Figure 1 data splits and Figure 2's fixed vs. rolling prediction carefully — these two approaches run through all experiments.

4. **Section 3 – Methodology** (pp.4-5): Figure 3의 SINet 아키텍처가 핵심입니다. (a) 전체 구조(2개 TimesBlock), (b) TimesBlock 내부(FFT → Reshape → Inception → Aggregate), (c) Dual-Inception 구조(6개 InConv + Gelu + 6개 OutConv). Table 1의 커널 크기 변화(1×1 ~ 11×11)에 주목하세요.
   Figure 3 is the heart of the paper. Focus on the architecture and Table 1's kernel size progression.

5. **Section 4 – Results** (pp.6-14): Table 3(F10.7)과 Table 4(F30)의 5-fold 결과가 핵심 비교표입니다. Figure 6, 9의 예측 시각화와 Figure 10의 NOAA AR 12673 사례 연구를 주목하세요.
   Tables 3-4 are the core comparison. Figures 6, 9 for visualization, Figure 10 for the AR 12673 case study.

6. **Section 5 – Discussion** (pp.11-12): 입력 길이(30일 = 1 태양 자전), FFT 주기 성분 수(k=3), TCN 대비 강점/약점, 자기상관 분석을 이해하세요.
   Understand input length choices, FFT component count, TCN comparison nuances, and autocorrelation analysis.

### 주의할 점 / Key Points to Watch

- **SINet_f vs SINet_r**: 고정(fixed) 예측과 롤링(rolling) 예측의 차이. SINet_f가 일관되게 더 우수합니다.
- **SINet_f vs SINet_r**: Fixed vs rolling prediction. SINet_f consistently outperforms SINet_r.

- **Solar maximum에서의 성능 저하**: 2014년 태양 극대기에서 모든 방법이 정확도가 떨어지지만, SINet이 TCN 대비 개선폭이 가장 큽니다.
- **Performance degradation at solar maximum**: All methods lose accuracy in 2014, but SINet shows the largest improvement over TCN.

- **AR 12673 사례**: 급격한 자기장 변화가 있는 활성 영역에서 모델의 한계가 드러납니다 — 이는 모든 태양 활동 예측 모델의 공통 과제입니다.
- **AR 12673 case**: The model's limitations emerge during rapid magnetic flux emergence — a common challenge for all solar activity forecasting models.

---

## 7. 현대적 의의 / Modern Significance

### 우주기상 실용적 관점 / Space Weather Operations Perspective

SINet_f는 NJIT의 실시간 F10.7 예측 시스템(nature.njit.edu/solardb/)에 이미 운영 배포되었습니다. NOAA/SWPC의 기존 45일 예측 시스템(MAPE 14.0%)과 비교하여 SINet_f가 13.8% MAPE로 소폭 우수한 성능을 보여, 기존 운영 시스템을 보완할 수 있음을 입증했습니다.

SINet_f has already been operationally deployed at NJIT's real-time F10.7 forecasting system. Compared to NOAA/SWPC's existing 45-day forecast system (MAPE 14.0%), SINet_f achieves 13.8% MAPE, demonstrating it can complement existing operational systems.

### 기술적 혁신 / Technical Innovation

1. **FFT 기반 주기성 포착**: 태양 자전 주기(~27일)를 비롯한 주기적 성분을 주파수 영역에서 명시적으로 추출하여 CNN에 전달하는 접근법은, 순수 시간 영역 방법(LSTM, TCN)보다 태양 물리의 주기적 특성에 더 적합합니다.
   FFT-based periodicity capture explicitly extracts periodic components (including ~27-day solar rotation) in the frequency domain, better suited for solar physics than pure time-domain methods.

2. **F30 예측의 선구**: F30은 열권 밀도 모델링에 F10.7보다 유용한 proxy로 주목받고 있으며, 이 논문이 딥러닝 F30 예측의 길을 열었습니다.
   F30 is increasingly recognized as a better proxy than F10.7 for thermospheric modeling; this paper pioneers deep learning F30 prediction.

3. **60:1 예측-케이던스 비율**: 일별 데이터로 60일을 예측하는 것은 기존 방법들(최대 27:1)을 크게 넘어서는 도전적 목표입니다.
   A 60:1 forecast-to-cadence ratio far exceeds existing methods (max 27:1).

### 한계와 향후 과제 / Limitations and Future Work

- 태양 극대기와 급격한 활성 영역 출현 시 성능 저하 — 데이터 증강, 이상 탐지 모델 병행 등이 제안됨
- Performance drops during solar maximum and rapid AR emergence — data augmentation and anomaly detection models suggested
- 입력 길이 30일(1 태양 자전)이 최적이지만, 7개월 impulse function을 완전히 포착하지 못하는 한계
- Input length of 30 days (1 solar rotation) is optimal but cannot fully capture the 7-month impulse function

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
