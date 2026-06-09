---
title: "Solar Wind Prediction Using Deep Learning — Reading Notes"
authors: Vishal Upendran, Mark C. M. Cheung, Shravan Hanasoge, Ganapathy Krishnamurthi
year: 2020
venue: "Space Weather (AGU)"
doi: 10.1029/2020SW002478
arxiv: 2006.05825
date_read: 2026-04-27
topic: Space_Weather
tags: [deep_learning, CNN, LSTM, solar_wind, EUV, AIA, SDO, WSA-ENLIL, coronal_holes, explainability, WindNet]
---

# Solar Wind Prediction Using Deep Learning
# 딥러닝을 이용한 태양풍 예측

## 1. Core Contribution / 핵심 기여

**한국어:** Upendran et al. (2020)은 **WindNet**이라는 end-to-end 딥러닝 모델을 제안하여, SDO/AIA 193 Å, 211 Å EUV 풀디스크 이미지로부터 라그랑주 1점(L1)의 일평균 태양풍(SW) 속도를 직접 예측한다. 모델은 ImageNet 사전학습 ConvNet (Inception-v3 계열)으로 EUV 이미지에서 시각 특징을 추출한 뒤, LSTM 셀로 H일치 입력 시퀀스를 처리하여 D일 후의 SW 속도를 회귀한다. 어떤 물리 법칙(MHD, PFSS, WSA 경험식)도 명시적으로 주입하지 않았음에도 NASA OMNIWEB 일평균 SW 속도와 **Pearson r = 0.55 ± 0.03**의 상관계수를 달성하며, 이는 운영급 WSA-ENLIL의 r ≈ 0.50과 비교할 만한 수준이다. 또한 Naive mean, N-day persistence, XGBoost, SVM 회귀의 모든 baseline을 능가한다.

**한국어 (확장):** 핵심 혁신은 두 가지 측면에서 의의를 가진다. 첫째, hand-engineered feature (CH 면적, 자속관 확장계수, PFSS 외삽 등) 없이 픽셀 → SW 속도 매핑이 가능함을 보였다는 점. 둘째, occlusion/Grad-CAM 활성 맵 분석을 통해 모델이 fast wind (≳500 km/s) 예측 시 약 3-4일 전 코로나 홀(CH)에 강한 활성을, slow wind 예측 시에는 활동 영역(AR)에 활성을 보여, Krieger+ (1973)과 Wang & Sheeley Jr. (1990) 등 50년에 걸친 헬리오피직스 경험칙을 데이터로부터 자동 재발견했음을 입증했다는 점이다.

**English:** Upendran et al. (2020) introduce **WindNet**, an end-to-end deep learning model that predicts daily-averaged solar wind (SW) speed at L1 directly from SDO/AIA 193 Å and 211 Å full-disk EUV images. The architecture couples an ImageNet-pretrained ConvNet (Inception-v3 family) feature extractor with an LSTM sequence regressor that ingests H days of inputs and outputs SW speed D days later. With **no explicit physics** (no MHD, no PFSS, no WSA empirical relation), it attains a Pearson correlation **r = 0.55 ± 0.03** against NASA OMNIWEB daily SW data — competitive with operational WSA-ENLIL (r ≈ 0.50) — and outperforms every baseline (naive mean, N-day persistence, XGBoost regression, SVM regression).

**English (extended):** The contribution is twofold. First, it demonstrates that a pixel → SW-speed regression is feasible without any hand-engineered features (CH area, flux-tube expansion factor, PFSS extrapolation). Second, occlusion / Grad-CAM activation analysis shows that the network attends to coronal holes (CHs) ~3-4 days before fast-wind events and to active regions (ARs) before slow-wind events, automatically recovering the 50-year heuristics of Krieger+ (1973) and Wang & Sheeley Jr. (1990) — i.e., the model "discovered" CH/AR-SW associations purely from data.

---

## 2. Reading Notes / 읽기 노트

### 2.1 Introduction (pp. 2-3) / 서론

**한국어:** 서론은 (a) 우주기상의 정의, (b) SW가 자기권과 상호작용하여 지자기폭풍, 오로라, 전력망 교란을 일으키는 메커니즘, (c) 기존 SW 예측의 세 갈래 — 경험적, 물리 기반, 하이브리드 — 를 다룬다. WSA-ENLIL과 MAS-ENLIL이 운영 표준이며, Jian et al. (2015)는 GONG-MAS-ENLIL의 hourly r = 0.57, GONG-WSA-ENLIL의 r = 0.50을 보고했다. CH-fast wind 연관은 Krieger+ (1973)에서 시작하여 Wang & Sheeley Jr. (1990)의 자속관 확장계수 역상관, Rotter+ (2012, 2015), Tsurutani et al., Temmer et al. (2018)이 EUV 분수 CH 면적으로 r ≈ 0.60-0.78까지 끌어올렸다. Yang et al. (2018)은 PFSS 출력을 입력으로 하는 NN으로 r = 0.74를 달성했으나 여전히 hand-engineered feature 의존적이다.

**English:** The introduction reviews (a) the definition of space weather, (b) how SW couples to the magnetosphere causing geomagnetic storms, aurorae, and grid disruptions, and (c) the three SW prediction paradigms — empirical, physics-based, and hybrid. WSA-ENLIL and MAS-ENLIL are operational standards; Jian et al. (2015) report hourly r = 0.57 for GONG-MAS-ENLIL and r = 0.50 for GONG-WSA-ENLIL. The CH-fast-wind link traces from Krieger+ (1973) to Wang & Sheeley Jr. (1990)'s inverse flux-tube expansion factor relation, with Rotter+ (2012, 2015) and Temmer+ (2018) achieving r ≈ 0.60-0.78 from EUV fractional CH area. Yang+ (2018) reach r = 0.74 with a NN ingesting PFSS outputs, yet still depend on hand-engineered features.

**한국어:** 본 논문은 ConvNet + LSTM 조합으로 hand-engineered feature 없이 EUV 이미지에서 직접 SW 속도를 예측하는 첫 시도이다. 학습은 supervised regression이며 prior physics는 사전학습 가중치 (ImageNet) 형태로만 들어간다.

**English:** This paper is the first attempt at hand-engineered-feature-free SW speed regression from EUV images using a ConvNet + LSTM. Learning is supervised regression; prior knowledge enters only via ImageNet-pretrained weights (transfer learning).

### 2.2 Data and Metrics (pp. 4-9) / 데이터와 지표

#### 2.2.1 EUV Dataset (p. 4-5) / EUV 데이터셋

**한국어:** 입력은 SDOML (Galvez+ 2019)의 AIA 193 Å, 211 Å 채널로, 512×512 픽셀(4.8″/px), 6분 cadence이다. 매일 00:00 UTC 영상(없으면 가장 가까운 시각)을 사용하여 일일 1장 풀디스크 영상을 만든다. EUV는 동적 범위가 8-bit를 크게 초과하므로 다음과 같이 log-scaling + threshold/saturation을 적용한다.

$$
x(193) = \begin{cases} \log(125.0) & x \le \log(125.0) \\ \log(5000.0) & x \ge \log(5000.0) \\ x & \text{else}\end{cases} \quad (1)
$$
$$
x(211) = \begin{cases} \log(25.0) & x \le \log(25.0) \\ \log(2500.0) & x \ge \log(2500.0) \\ x & \text{else}\end{cases} \quad (2)
$$

이후 픽셀값을 [0, 255]로 재스케일하여 ImageNet pretrained network 입력 분포와 맞춘다. Threshold/saturation 조합 sweep을 수행했고, (log 250, log 10000)은 r = 0.46, (log 100, log 1000)은 r = 0.35, (log 125, log 5000)이 best r = 0.48±0.03을 보였다(이 단계는 best 입력 정규화 탐색 단계의 수치).

**English:** Inputs come from SDOML (Galvez+ 2019): AIA 193 Å and 211 Å channels at 512×512 px (4.8″/px), 6-min cadence. Each day uses the 00:00 UTC frame (or the nearest available) to produce one daily full-disk image. Because EUV dynamic range vastly exceeds 8 bits, log-scaling plus thresholding/saturation are applied per Eqs. (1) and (2). After scaling, pixels are mapped to [0, 255] to match the ImageNet pretrained network's expected input. A coarse sweep of (low, high) thresholds shows (log 250, log 10000) → r = 0.46; (log 100, log 1000) → r = 0.35; the best is (log 125, log 5000) → r = 0.48 ± 0.03 (these are search-stage values).

#### 2.2.2 SW Dataset (p. 6) / 태양풍 데이터셋

**한국어:** 타겟은 OMNIWEB의 일평균 SW 속도. 일중 변동(σ)을 측정 불확실성으로 사용한다. AIA 데이터 결측(193 Å에서 31일, 211 Å에서 30일 누락)은 해당일을 학습/평가에서 제거한다. Fig. 2는 2011-01-01~10 10일치 SW 속도 시계열 (~340 km/s slow wind에서 ~600 km/s fast HSE로의 천이를 보여줌). Fig. 3은 전체 데이터셋의 SW 속도 분포 (peak ~400 km/s, 양쪽으로 250-700 km/s)와 σ 분포 (peak ~10-20 km/s).

**English:** The target is daily-averaged SW speed from OMNIWEB; the diurnal standard deviation σ serves as per-sample measurement uncertainty. Days with missing AIA frames (31 days for 193 Å, 30 days for 211 Å) are removed. Fig. 2 shows a 10-day SW speed time series (2011-01-01 → 10) with a transition from ~340 km/s slow wind to ~600 km/s fast HSE. Fig. 3 displays the full SW speed distribution (peak ~400 km/s, range 250-700 km/s) and σ distribution (peak ~10-20 km/s).

#### 2.2.3 Dataset Partitioning and Cross Validation (p. 6-7) / 분할과 교차검증

**한국어:** 데이터 기간: 2011-01-01 ~ 2018-12-09 (cycle 24). 시간적 누수를 막기 위해 **20일 단위 batch**로 자른 뒤 5 fold에 무작위 배정한다. 한 batch 내 단일 불연속이 있으면 직전 일자에서 20일 이전 구간을 같은 자리에 다시 샘플링한다. 다중 불연속(전체 데이터에 단 2건)은 폐기. 결과: 211 Å에서 157 batch, 193 Å에서 158 batch.

**English:** Period: 2011-01-01 to 2018-12-09 (cycle 24). To avoid temporal leakage, data are sliced into **20-day contiguous batches** and randomly assigned to 5 folds. If a single discontinuity exists in a batch, the 20 days preceding the discontinuity replace the affected portion; batches with multiple discontinuities (only 2 occurrences) are discarded. Result: 157 batches at 211 Å, 158 at 193 Å.

**한국어:** 5-fold CV: fold i를 test, 나머지 4 fold를 training으로 5번 학습 → 평균 ± 표준편차. 영상은 224×224로 OpenCV bilinear resize하고 RGB 3 channel로 복제(ImageNet 사전학습 입력 호환). SW 속도는 fold별 max/min으로 [0, 1] 정규화.

**English:** 5-fold CV: hold fold i as test, train on the remaining 4; cycle through all 5 → report mean ± std. Images are resized to 224×224 via OpenCV bilinear interpolation and replicated into 3 RGB channels (ImageNet input compatibility). SW speeds are scaled to [0, 1] using each fold's max/min.

#### 2.2.4 Control Hyperparameters (p. 7-8) / 제어 하이퍼파라미터

**한국어:** **History H** = 입력일 수, **Delay D** = 마지막 입력일과 예측일 사이 간격. 예) T 예측, H=4, D=3 → 입력 = T-6, T-5, T-4, T-3 (3일 lead). 모든 조합 H=1..4 × D=1..4 = 16개 모델 변형을 학습한다 (Fig. 5).

**English:** **History H** = number of input days; **Delay D** = gap between last input day and prediction day. Example: predicting T with H=4, D=3 uses inputs T-6, T-5, T-4, T-3 (3-day lead time). All 16 combinations H=1..4 × D=1..4 are trained (Fig. 5).

#### 2.2.5 Metrics for Comparison (p. 8-10) / 비교 지표

**한국어:** 회귀 지표 세 개:

$$
\chi^2 = \frac{1}{N}\sum_i (\hat y_i - y_i)^2 \quad (3), \qquad
\chi^2_{\text{red}} = \frac{1}{N}\sum_i \frac{(\hat y_i - y_i)^2}{\sigma_i^2} \quad (4),
$$
$$
r = \frac{\sum_i (y_i - \bar y)(\hat y_i - \bar{\hat y})}{\sigma_y \sigma_{\hat y}} \quad (5).
$$

RMSE = √χ². Fold 평균 r은 Fischer z-space에서 평균 후 역변환(편향 제거). 표준오차 $S(x) = \sigma(x)/\sqrt{N(x)}$.

**English:** Three regression metrics: χ² (Eq. 3), σ-weighted χ²_red (Eq. 4), Pearson r (Eq. 5). RMSE = √χ². Per-fold r averaged in Fischer z-space then inverse-transformed to remove bias. Reported uncertainty is standard error $S(x) = \sigma(x)/\sqrt{N(x)}$.

**한국어:** **HSE Threat Score** (Owens+ 2005; Jian+ 2015 알고리즘):
1. 1일 전보다 +50 km/s 빠른 시점을 모두 마킹
2. 연속 마크를 묶어 HSE 정의, 시작/종료 시각 결정
3. HSE 시작 2일 전~시작 사이 최저속도 = Vmin; 시작~종료+1일 사이 최고속도 = Vmax
4. SIR 그룹화, stream interface 시간 결정, 중복 제거
5. Vmin > 500, Vmax < 400, 또는 ΔV < 100 km/s인 SIR 폐기

$$
\mathrm{TS} = \frac{TP}{TP + FN + FP} \quad (6).
$$

H=1, D=1 persistence는 정의상 TS = 1.0이므로 비교에서 제외.

**English:** **HSE Threat Score** (Owens+ 2005; Jian+ 2015 algorithm): (1) mark all timepoints +50 km/s faster than the day prior; (2) group consecutive marks into HSEs, find start/end; (3) Vmin from 2 days before start, Vmax from start to end+1; (4) cluster SIRs, resolve interface times; (5) discard SIRs with Vmin > 500 or Vmax < 400 or ΔV < 100 km/s. TS = TP/(TP+FN+FP). The H=1, D=1 persistence trivially yields TS = 1, so it is excluded from comparison.

**한국어:** ICME 처리: 연구 기간 중 170 ICMEs (336일 영향)이 있으나 본 연구에서는 제거하지 않는다. ICME-driven SW은 본 연구 범위 밖이지만 L1 측정에는 포함되므로 train/test 데이터에 그대로 둔다.

**English:** ICME treatment: 170 ICMEs (affecting 336 days) occur in-period and are *not* excluded — predicting CME-driven SW is out of scope, but those events do affect L1 measurements and are kept in the dataset.

### 2.3 Modelling and Methods (pp. 10+) / 모델링과 방법

#### 2.3.1 Benchmark Models / 벤치마크 모델

**한국어:** 5개 baseline:
1. **Naive mean**: batch 평균값 출력 — ML 모델이 반드시 능가해야 할 하한.
2. **N-day persistence**: H+D-1일 전 SW 속도 그대로 출력. H, D의 합에만 의존(중복).
3. **27-day persistence**: 27일 전 SW 속도 출력 — Carrington 주기 강한 baseline.
4. **XGBoost regression**: gradient-boosted tree 회귀.
5. **SVM regression**: 커널 SVM.

후자 셋은 SW 시계열만으로 autoregression이며 EUV 이미지를 사용하지 않는다 (vs. WindNet).

**English:** Five baselines: (1) **naive mean** — outputs the batch mean (lower bound any ML model must beat); (2) **N-day persistence** — outputs SW from H+D-1 days prior, depending only on the sum H+D; (3) **27-day persistence** — outputs SW from 27 days prior, leveraging Carrington recurrence; (4) **XGBoost regression**; (5) **SVM regression**. The last three perform autoregression on SW alone — they do *not* use EUV images, unlike WindNet.

#### 2.3.2 WindNet Architecture / WindNet 구조

**한국어:** WindNet은 두 단계로 구성된다.
- **특징 추출기 (Feature extractor)**: ImageNet 사전학습 ConvNet (Szegedy+ 2015 Inception-v3 계열). 입력 224×224×3 (RGB로 복제된 EUV 채널) → 고차원 특징 벡터.
- **시계열 회귀기 (Temporal regressor)**: H일치 특징 벡터를 LSTM (Hochreiter & Schmidhuber 1997)에 입력하여 시간적 의존성을 인코딩 → fully-connected layer → SW 속도 [0, 1] 회귀.

학습 손실: MSE (χ²). 사전학습 가중치는 fine-tune되며, LSTM과 회귀 head는 from-scratch 학습된다.

**English:** WindNet has two stages:
- **Feature extractor**: ImageNet-pretrained ConvNet (Szegedy+ 2015 Inception-v3 family). Input 224×224×3 (RGB-replicated EUV channels) → high-dimensional feature vector.
- **Temporal regressor**: H feature vectors fed to an LSTM (Hochreiter & Schmidhuber 1997) encoding temporal dependencies → fully-connected → regression to SW speed in [0, 1].

Loss: MSE (χ²). Pretrained weights are fine-tuned; LSTM and regression head are trained from scratch.

#### 2.3.3 Activation / Visualization / 활성 시각화

**한국어:** 모델이 어디를 보는지 알기 위해 Grad-CAM류 (Selvaraju+ 2016) 활성 맵을 사용한다. test set 예측을 fast (>500 km/s)와 slow (<400 km/s)로 그룹화한 뒤 그룹별 평균 활성 맵을 계산. fast 그룹 평균 활성 vs. 211 Å AR 마스크, 193 Å CH 마스크를 비교.

CH 마스크: 193 Å 강도 임계 (어두운 영역) + 형태학 연산. AR 마스크: 211 Å 강도 임계 (밝은 영역). mean activation = (mask 영역 활성의 평균) / (전체 활성 평균)으로 정규화.

**English:** Grad-CAM-style (Selvaraju+ 2016) activation maps reveal where the model attends. Test predictions are split into fast (>500 km/s) and slow (<400 km/s) groups; mean activation maps per group are computed. Fast-group activation is compared with 211 Å AR masks and 193 Å CH masks.

CH mask: 193 Å intensity threshold (dark regions) + morphological cleanup. AR mask: 211 Å intensity threshold (bright regions). The "mean activation" metric = (average activation inside mask) / (overall activation average).

### 2.4 Results (anticipated, pp. 11+) / 결과 (예상)

**한국어:** 본 단계까지 읽은 내용으로부터 추론한 핵심 결과는:
- WindNet은 모든 (H, D) 조합에서 naive mean과 N-day persistence를 능가.
- 최적 r = 0.55 ± 0.03 (작은 H, D 영역에서). Lead time D ≈ 3-4일에서 여전히 유의미한 r 유지 → 운영 예보 (3-4일 forewarning)에 적합.
- HSE Threat Score는 H=1, D=1 persistence (TS=1) 다음으로 WindNet이 가장 높음.
- Activation map: fast wind 예측 시 D = 3-4일 전 193 Å CH에 집중 활성, slow wind 예측 시 211 Å AR에 활성 → 물리적으로 일관.

**English:** Inferences from the read sections:
- WindNet beats naive mean and N-day persistence at every (H, D).
- Best r = 0.55 ± 0.03 (small H, D regime). Useful skill persists at D ≈ 3-4 days, suitable for operational 3-4-day lead forecasting.
- HSE Threat Score: WindNet is highest after the trivial H=1, D=1 persistence (TS=1).
- Activation maps: fast-wind predictions concentrate on 193 Å CHs ~3-4 days prior; slow-wind predictions concentrate on 211 Å ARs — physically consistent.

### 2.5 Conclusions (anticipated) / 결론 (예상)

**한국어:** WindNet은 hand-engineered feature 없이 EUV 이미지에서 SW 속도를 예측할 수 있음을 보였고, 운영 표준 WSA-ENLIL과 견줄 만한 r을 달성했다. 활성 맵 분석은 모델이 50년 헬리오피직스 경험칙을 자동 재발견함을 보여 deep learning + 우주기상의 향후 연구 방향(IMF Bz, multi-channel, multi-cycle)을 제시한다.

**English:** WindNet shows SW-speed prediction is feasible without hand-engineered features, achieving r competitive with operational WSA-ENLIL. Activation analysis demonstrates the network rediscovers 50-year heliophysics heuristics, charting future directions (IMF Bz prediction, multi-channel inputs, multi-cycle generalization) for DL + space weather.

---

## 3. Key Takeaways / 핵심 시사점

1. **End-to-end learning eliminates feature engineering / 종단간 학습은 특징공학을 제거한다**
   - 한국어: PFSS, 자속관 확장계수, CH 면적 같은 hand-crafted feature 없이 픽셀 → SW 속도 매핑이 학습 가능함을 처음으로 증명.
   - English: First demonstration that pixel → SW-speed mapping is learnable without hand-crafted features (PFSS, flux-tube expansion factor, CH fractional area).

2. **Transfer learning bridges scarce solar data and rich vision priors / 전이학습은 희소한 태양 데이터와 풍부한 비전 prior를 잇는다**
   - 한국어: ImageNet 사전학습 ConvNet 가중치를 그대로 활용하여 태양 데이터(~수천 일)만으로 학습이 안정화됨. 224×224 RGB 호환 입력 설계가 실용적 트릭.
   - English: ImageNet-pretrained ConvNet weights stabilize training on only ~thousands of solar samples; the 224×224 RGB input adapter is a practical engineering trick.

3. **WindNet matches operational WSA-ENLIL with no physics / WindNet은 물리 없이 운영 WSA-ENLIL을 따라잡는다**
   - 한국어: r = 0.55 ± 0.03은 Jian+ (2015)의 GONG-WSA-ENLIL r ≈ 0.50과 비슷하거나 약간 우수. 데이터 기반이 첫 시도에서 물리 모델과 동급에 도달.
   - English: r = 0.55 ± 0.03 matches/slightly exceeds Jian+ (2015)'s GONG-WSA-ENLIL r ≈ 0.50 — data-driven parity with operational physics on first attempt.

4. **Activation maps validate "physics emergence" / 활성 맵은 "물리 발현"을 검증한다**
   - 한국어: fast-wind 예측 시 D ≈ 3-4일 전 193 Å CH에 평균 활성, slow-wind 예측 시 211 Å AR에 활성 → Krieger+ (1973), Wang & Sheeley (1990) 경험칙과 정성적 일치.
   - English: Fast-wind predictions concentrate activation on 193 Å CHs ~3-4 days prior; slow-wind on 211 Å ARs — qualitatively matching Krieger+ (1973) and Wang & Sheeley (1990) heuristics.

5. **Persistence is a strong, honest baseline / 지속성은 강력하고 정직한 baseline이다**
   - 한국어: 27일 Carrington 주기 때문에 27-day persistence가 의외로 강한 baseline. ML 논문 신뢰성을 위해 반드시 비교해야 한다는 교훈.
   - English: The 27-day Carrington recurrence makes 27-day persistence a surprisingly strong baseline. ML papers in space weather *must* benchmark against it.

6. **Daily averages trade temporal resolution for SNR / 일평균은 시간해상도를 SNR과 거래한다**
   - 한국어: 일평균은 high-frequency (Alfvén, microstreams) 잡음을 줄여 거시 CH/AR 구조 매핑에 집중하게 함. 운영 예보에는 hourly 모델로의 후속 확장이 필요.
   - English: Daily averaging suppresses high-frequency noise (Alfvén waves, microstreams), letting the model focus on macroscopic CH/AR structure. Operational forecasting requires future hourly-resolution extensions.

7. **20-day batched 5-fold CV prevents temporal leakage / 20일 배치 기반 5-fold CV는 시간적 누수를 막는다**
   - 한국어: 단순 random split은 인접일 누수로 r을 부풀린다. 20일 contiguous batch + fold 무작위 배정이 시계열 ML 표준 관행.
   - English: A naive random split would inflate r via adjacent-day leakage. 20-day contiguous batches assigned randomly to folds is the proper time-series ML practice.

8. **Reduced χ² and Threat Score complement Pearson r / 감축 χ²와 위협 점수는 Pearson r를 보완한다**
   - 한국어: r은 추세만 잡고 절대오차는 못 잡는다. χ²_red는 측정 σ로 가중하여 의미 있는 fit 평가, TS는 이벤트(HSE) 검출 능력을 평가. 세 지표 조합이 균형 잡힌 평가를 제공.
   - English: r captures trend but ignores scale; χ²_red weights by measurement σ for meaningful goodness-of-fit; TS evaluates event (HSE) detection. The three together yield balanced assessment.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 EUV preprocessing / EUV 전처리

$$
x(193) = \begin{cases} \log(125.0) & x \le \log(125.0) \\ \log(5000.0) & x \ge \log(5000.0) \\ x & \text{otherwise}\end{cases} \quad (1)
$$
$$
x(211) = \begin{cases} \log(25.0) & x \le \log(25.0) \\ \log(2500.0) & x \ge \log(2500.0) \\ x & \text{otherwise}\end{cases} \quad (2)
$$

**한국어:** $x$는 픽셀 강도의 자연로그. 하한 (low)으로 노이즈/배경 픽셀을 클립하여 dark CH 잡음을 안정화하고, 상한 (high)으로 AR 포화를 잡아 dynamic range를 8-bit 컴퓨터 비전 입력에 맞춘다. 그 후 [0, 255]로 재정규화.

**English:** $x$ is the log of pixel intensity. Clipping at the low bound stabilizes dark-CH noise; clipping at the high bound saturates AR brightness; both compress the EUV dynamic range to fit 8-bit computer-vision inputs. Pixels are then rescaled to [0, 255].

### 4.2 Mean square error / 평균제곱오차

$$
\chi^2 = \frac{1}{N} \sum_{i=1}^{N} (\hat y_i - y_i)^2 \quad (3)
$$

**한국어:** $\hat y_i$는 모델 예측, $y_i$는 OMNIWEB 관측. 학습 손실로 사용되며 RMSE = $\sqrt{\chi^2}$ (단위 km/s).

**English:** $\hat y_i$ is the model prediction; $y_i$ is the OMNIWEB observation. Used as training loss; RMSE = $\sqrt{\chi^2}$ (km/s).

### 4.3 Reduced chi-squared / 정규화된 χ²

$$
\chi^2_{\text{red}} = \frac{1}{N} \sum_{i=1}^{N} \frac{(\hat y_i - y_i)^2}{\sigma_i^2} \quad (4)
$$

**한국어:** $\sigma_i$는 일평균 SW 속도의 일중 표준편차(measurement uncertainty). 큰 σ 데이터는 큰 잔차를 허용하고 작은 σ 데이터는 엄격히 평가한다. $\chi^2_{\text{red}} \approx 1$이면 모델이 측정 잡음 수준까지 적합.

**English:** $\sigma_i$ is the diurnal std of the daily-averaged SW speed (measurement uncertainty). High-σ samples tolerate large residuals; low-σ samples are judged strictly. $\chi^2_{\text{red}} \approx 1$ implies the model fits to within measurement noise.

### 4.4 Pearson correlation / 피어슨 상관계수

$$
r = \frac{\sum_{i=1}^{N} (y_i - \bar y)(\hat y_i - \bar{\hat y})}{\sigma_y \sigma_{\hat y}} \quad (5)
$$

**한국어:** -1 ≤ r ≤ 1. 추세 일치도를 측정하며 절대 스케일에 무관. fold별 r은 Fischer z-변환 $z = \tanh^{-1}(r)$ 후 평균하여 편향을 제거하고, 다시 $r = \tanh(z)$로 역변환.

**English:** -1 ≤ r ≤ 1; measures trend agreement, scale-invariant. Per-fold r values are averaged in Fischer z-space ($z = \tanh^{-1}(r)$) to remove bias, then back-transformed via $r = \tanh(z)$.

### 4.5 Standard error / 표준오차

$$
S(x) = \frac{\sigma(x)}{\sqrt{N(x)}}
$$

**한국어:** 5-fold CV에서 보고하는 ±값은 표본 표준편차가 아니라 **표준오차** (평균 추정의 불확실성). N(x) = 5 (fold 수).

**English:** The ± reported across 5-fold CV is the **standard error** (uncertainty of the mean) — not the sample std. N(x) = 5 folds.

### 4.6 HSE Threat Score / HSE 위협 점수

$$
\mathrm{TS} = \frac{TP}{TP + FN + FP} \quad (6)
$$

**한국어:** TP=관측+예측 모두 HSE; FN=관측 HSE 누락; FP=잘못 예측한 HSE. 이벤트 검출 정확도를 직접 평가. H=1, D=1 persistence는 TS=1 (자명).

**English:** TP = HSE in both observation and prediction; FN = missed HSE; FP = spurious predicted HSE. Directly measures event-detection accuracy. The H=1, D=1 persistence trivially yields TS = 1.

### 4.7 Worked example / 수치 예제

**한국어:** N = 100일치 예측, 평균 관측 SW = 400 km/s, 평균 σ_i = 20 km/s, RMSE = 80 km/s 라 하자.
- $\chi^2 = (80)^2 = 6400 \,(\text{km/s})^2$
- $\chi^2_{\text{red}} = 6400 / 400 = 16$ → 측정 잡음의 약 16배 잔차 → 추가 개선 여지.
- 만약 r = 0.55라면 $z = \tanh^{-1}(0.55) \approx 0.618$. 5-fold 평균 후 $\tanh(\bar z)$로 환산.
- HSE 30개 중 TP=18, FN=12, FP=10이면 TS = 18/(18+12+10) = 0.45.

**English:** N = 100 predictions, mean observed SW = 400 km/s, mean σ_i = 20 km/s, RMSE = 80 km/s.
- $\chi^2 = (80)^2 = 6400 \,(\text{km/s})^2$.
- $\chi^2_{\text{red}} = 6400/400 = 16$ — residuals are ~16× measurement noise; significant headroom for improvement.
- For r = 0.55, $z = \tanh^{-1}(0.55) \approx 0.618$; average across folds and back-transform.
- For 30 HSEs with TP=18, FN=12, FP=10, TS = 18/(18+12+10) = 0.45.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1968 Parker hydrodynamic SW model
  |
1969 Altschuler & Newkirk — PFSS coronal model
  |
1973 Krieger et al. — first CH-fast wind link via Skylab X-ray
  |
1990 Wang & Sheeley Jr. — flux-tube expansion factor predictor
  |
1999 Linker et al. — MHD-based SW propagation
  |
2003 Arge & Pizzo — WSA empirical formula
  |
2008 Owens et al. — review of SW prediction methods
  |
2012-2015 Rotter, Tsurutani, Temmer — EUV CH fractional area regression (r ≈ 0.6-0.78)
  |
2015 Jian et al. — MAS-/WSA-ENLIL benchmark (hourly r ≈ 0.50-0.57)
  |
2015 Szegedy et al. — Inception-v3 ConvNet (transfer-learning workhorse)
  |
2016 Selvaraju et al. — Grad-CAM activation maps
  |
2018 Yang et al. — NN with PFSS+heuristic inputs (r ≈ 0.74)
  |
2019 Galvez et al. — SDOML public ML-ready SDO dataset
  |
2020 Upendran et al. — *WindNet*: first end-to-end CNN+LSTM, EUV → SW speed (r ≈ 0.55) ← THIS PAPER
  |
2021+ Reiss, Bailey, Brown — Bz prediction, multi-spacecraft ensembles, transformer SW models
```

**한국어:** 본 논문은 PFSS-WSA-ENLIL 라인의 50년 물리 모델 전통과 2010년대 컴퓨터 비전 혁명(ImageNet, Inception, Grad-CAM)을 우주기상 분야에 처음 접목한 변곡점이다.

**English:** This paper is the inflection point where the 50-year PFSS-WSA-ENLIL physics tradition meets the 2010s computer-vision revolution (ImageNet, Inception, Grad-CAM), bringing end-to-end deep learning to space weather forecasting.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 |
|---|---|
| Parker 1958 (#? AI / SW 곡선) | SW의 물리적 기원 — WindNet이 학습으로 근사하는 매핑의 근원 |
| Altschuler & Newkirk 1969 (PFSS) | WindNet이 명시적으로 사용하지 않는 coronal field model — 비교 baseline 제공 |
| Krieger et al. 1973 | CH-fast wind 관계의 원조 — WindNet의 활성 맵이 자동 재발견 |
| Wang & Sheeley Jr. 1990 | flux-tube expansion factor 경험식 — WSA의 W |
| Arge & Pizzo 2003 | WSA empirical relation — WSA-ENLIL의 핵심 |
| LeCun et al. 1998 / Krizhevsky+ 2012 / Szegedy+ 2015 | ConvNet 계보 — WindNet의 backbone |
| Hochreiter & Schmidhuber 1997 | LSTM — WindNet의 시계열 회귀기 |
| Selvaraju et al. 2016 (Grad-CAM) | 활성 맵 시각화 — WindNet의 explainability 기법 |
| Galvez et al. 2019 (SDOML) | WindNet의 입력 데이터셋 |
| Jian et al. 2015 | WSA-ENLIL hourly r ≈ 0.50 — WindNet 비교 기준 |
| Yang et al. 2018 | NN + PFSS hybrid r ≈ 0.74 — WindNet의 전임자 |
| Reiss et al. 2016, Bu et al. 2019 | HSE 정의 채택 |
| Owens et al. 2008 | SW 예측 review — 본 논문 introduction의 토대 |
| Hu et al., Bailey et al. (post-2020) | Bz 예측, transformer 기반 모델 — WindNet의 후속 연구 |

---

## 7. References / 참고문헌

**Primary / 본 논문**
- Upendran, V., Cheung, M. C. M., Hanasoge, S., Krishnamurthi, G. (2020). "Solar wind prediction using deep learning." *Space Weather*, 18, e2020SW002478. DOI: 10.1029/2020SW002478. arXiv: 2006.05825.

**Heliophysics / 헬리오피직스**
- Altschuler, M. D., & Newkirk, G. (1969). "Magnetic fields and the structure of the solar corona." *Solar Physics*, 9, 131-149.
- Arge, C. N., & Pizzo, V. J. (2000). "Improvement in the prediction of solar wind conditions using near-real time solar magnetic field updates." *JGR*, 105, 10465.
- Galvez, R. et al. (2019). "A machine-learning data set prepared from the NASA Solar Dynamics Observatory mission." *ApJS*, 242, 7.
- Jian, L. K. et al. (2015). "Comparison of observations at ACE and Ulysses with Enlil model results." *Solar Physics*, 290, 2245-2263.
- Krieger, A. S., Timothy, A. F., & Roelof, E. C. (1973). "A coronal hole and its identification as the source of a high velocity solar wind stream." *Solar Physics*, 29, 505-525.
- Lemen, J. R. et al. (2012). "The Atmospheric Imaging Assembly (AIA) on the Solar Dynamics Observatory." *Solar Physics*, 275, 17-40.
- Linker, J. A. et al. (1999). "Magnetohydrodynamic modeling of the solar corona." *JGR*, 104, 9809.
- Owens, M. J., Spence, H. E., McGregor, S., et al. (2008). "Metrics for solar wind prediction models." *Space Weather*, 6, S08001.
- Pesnell, W. D., Thompson, B. J., & Chamberlin, P. C. (2012). "The Solar Dynamics Observatory (SDO)." *Solar Physics*, 275, 3-15.
- Riley, P., Linker, J. A., & Mikic, Z. (2006). "Modeling the heliospheric current sheet." *JGR*, 111, A12.
- Rotter, T. et al. (2012, 2015). "Relation between coronal hole areas on the Sun and the solar wind parameters at 1 AU." *Solar Physics*.
- Schwenn, R. (2006). "Space weather: the solar perspective." *LRSP*, 3, 2.
- Temmer, M., Hinterreiter, J., & Reiss, M. A. (2018). "Coronal hole evolution from multi-viewpoint data." *JSWSC*, 8, A18.
- Wang, Y.-M., & Sheeley Jr., N. R. (1990). "Solar wind speed and coronal flux-tube expansion." *ApJ*, 355, 726.
- Yang, Y. et al. (2018). "Prediction of the solar wind speed by the artificial neural network." *Solar Physics*, 293, 142.

**Deep Learning / 딥러닝**
- Bradski, G. (2000). The OpenCV library. *Dr. Dobb's Journal*.
- Chen, T., & Guestrin, C. (2016). "XGBoost: a scalable tree boosting system." *KDD*.
- Ciresan, D. et al. (2011). "Flexible, high performance convolutional neural networks for image classification." *IJCAI*.
- Deng, J. et al. (2009). "ImageNet: a large-scale hierarchical image database." *CVPR*.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
- Hochreiter, S., & Schmidhuber, J. (1997). "Long short-term memory." *Neural Computation*, 9, 1735-1780.
- LeCun, Y., Bengio, Y., & Hinton, G. (2015). "Deep learning." *Nature*, 521, 436-444.
- Selvaraju, R. R. et al. (2016). "Grad-CAM: visual explanations from deep networks via gradient-based localization." *ICCV*.
- Sutskever, I., Vinyals, O., & Le, Q. V. (2014). "Sequence to sequence learning with neural networks." *NeurIPS*.
- Szegedy, C. et al. (2015). "Rethinking the Inception architecture for computer vision." *CVPR*.
- Walt, S. v. d., Colbert, S. C., & Varoquaux, G. (2011). "The NumPy array." *CSE*, 13, 22-30.

**Other / 기타**
- Bu, X. et al. (2019). "Forecasting high-speed solar wind streams from solar images." *Space Weather*, 17, 1040.
- Reiss, M. A. et al. (2016). "Verification of high-speed solar wind stream forecasts using operational solar wind models." *Space Weather*, 14, 495.

---

*Notes compiled 2026-04-27 for paper #35 in Space_Weather curriculum (~415 lines).*
