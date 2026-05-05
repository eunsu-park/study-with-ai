---
title: "Solar Wind Prediction Using Deep Learning — Pre-Reading Briefing"
authors: Vishal Upendran, Mark C. M. Cheung, Shravan Hanasoge, Ganapathy Krishnamurthi
year: 2020
venue: "Space Weather (AGU), arXiv:2006.05825"
doi: 10.1029/2020SW002478
date_briefed: 2026-04-27
topic: Space_Weather
tags: [deep_learning, CNN, solar_wind, EUV, AIA, SDO, WSA-ENLIL, coronal_holes, space_weather]
---

# 사전 브리핑 / Pre-Reading Briefing

## 1. 한 문단 요약 / One-Paragraph Summary

**한국어:** Upendran et al. (2020)은 SDO/AIA의 193 Å, 211 Å EUV 풀디스크 이미지를 입력으로 받아 L1 (라그랑주 1점) 지점에서 측정된 일평균 태양풍(SW) 속도를 직접 예측하는 딥러닝 모델 **WindNet**을 제안한다. WindNet은 ImageNet으로 사전 학습된 ConvNet (Inception-v3 계열) 특징 추출기와 LSTM 시계열 회귀기를 결합한 구조로, 물리 법칙(MHD, PFSS 등)을 명시적으로 주입하지 않고도 NASA OMNIWEB 일평균 SW 속도와 **r = 0.55 ± 0.03**의 상관계수를 달성한다. 특히 fast wind (≳500 km/s) 예측 시 모델이 약 3-4일 전의 코로나 홀(CH) 영역에 높은 활성을 보이고, slow wind 예측 시에는 활동 영역(AR)에 활성을 보여 기존 헬리오피직스 경험칙과 부합하는 패턴을 학습했음을 보였다. 이는 EUV 이미지로부터 SW를 직접 예측한 최초의 end-to-end 딥러닝 우주기상 예보 사례로, WSA-ENLIL과 견줄 만한 성능을 데이터 기반으로 달성했다.

**English:** Upendran et al. (2020) introduce **WindNet**, a deep learning model that directly predicts daily-averaged solar wind (SW) speed at L1 from SDO/AIA full-disk Extreme Ultraviolet (EUV) images at 193 Å and 211 Å. WindNet couples an ImageNet-pretrained ConvNet (Inception-v3 family) feature extractor with an LSTM sequence regressor; without injecting any explicit physics (no MHD, no PFSS), it achieves a Pearson correlation **r = 0.55 ± 0.03** with NASA OMNIWEB daily SW speed. Crucially, post-hoc activation analysis shows that for fast-wind predictions the network attends to coronal holes (CHs) about 3-4 days prior, and for slow-wind predictions it attends to active regions (ARs) — patterns that match longstanding heliophysics heuristics. This is the first end-to-end deep-learning forecast of SW speed from EUV imagery, achieving performance comparable to traditional WSA-ENLIL with a purely data-driven pipeline.

---

## 2. 왜 이 논문이 중요한가 / Why This Paper Matters

**한국어:**
- **End-to-end 학습**: PFSS 외삽, 자속관 확장계수, CH 면적 등 hand-engineered feature를 거치지 않고 **EUV 픽셀 → SW 속도** 회귀를 단일 신경망이 학습한다.
- **WSA-ENLIL과의 정량 비교**: Jian et al. (2015)이 보고한 WSA-ENLIL r ≈ 0.50, MAS-ENLIL r ≈ 0.57과 직접 비교 가능한 r = 0.55를 달성하여, 데이터 기반 모델이 첫 시도에서도 물리 모델에 견줄 수 있음을 보였다.
- **설명가능성(Explainability)**: occlusion/activation map을 통해 모델이 "왜" 그런 예측을 했는지 시각화하고, 이를 CH와 AR이라는 물리적 구조와 연결했다.
- **공개 데이터셋(SDOML)**: Galvez et al. (2019)의 ML-ready SDO 데이터셋을 활용하여 재현성을 확보했다.

**English:**
- **End-to-end learning**: a single network maps **EUV pixels → SW speed** without hand-engineered features such as PFSS extrapolations, flux-tube expansion factors, or CH fractional area.
- **Quantitative parity with WSA-ENLIL**: WindNet's r = 0.55 directly compares to Jian et al. (2015)'s WSA-ENLIL r ≈ 0.50 and MAS-ENLIL r ≈ 0.57, showing that data-driven models can rival physics models on first attempt.
- **Explainability**: occlusion / activation maps reveal *why* the network predicts what it predicts and link those reasons to physical CH and AR structures.
- **Open dataset (SDOML)**: builds on Galvez et al. (2019)'s ML-ready SDO data, ensuring reproducibility.

---

## 3. 사전 지식 / Prerequisites

| 분야 / Area | 필수 개념 / Required Concepts |
|---|---|
| Solar Physics | EUV emission lines (193 Å Fe XII ~1.6 MK; 211 Å Fe XIV ~2 MK), coronal holes vs. active regions, solar wind types (fast/slow), L1 in-situ measurements (OMNIWEB) |
| Space Weather Models | WSA-ENLIL pipeline (PFSS → WSA empirical f(f) → ENLIL MHD), persistence baseline, 27-day Carrington recurrence |
| Deep Learning | CNN basics (conv, pool, ReLU), Inception/ImageNet transfer learning, LSTM, regression loss (MSE), cross-validation, batch normalization |
| Statistics | Pearson r with Fisher z-transform, χ², χ²_red, Threat Score (TP/(TP+FN+FP)) |

**한국어 요약:** EUV 영상의 천체물리적 의미(고온 코로나 플라즈마 방출), 태양풍이 발생하는 코로나 구조, ConvNet과 LSTM의 기본 동작, 그리고 회귀 평가 지표(상관계수, 정규화된 χ², HSE Threat Score) 정도의 배경이 필요하다.

**English summary:** Need familiarity with EUV imagery physics (hot coronal plasma emission), coronal sources of the solar wind, basic ConvNet + LSTM mechanics, and regression metrics (correlation, reduced χ², HSE Threat Score).

---

## 4. 핵심 어휘 / Key Vocabulary

| Term | 한국어 | Definition |
|---|---|---|
| **WindNet** | 윈드넷 | 본 논문에서 제안한 ConvNet+LSTM 결합 SW 예측 모델 / proposed ConvNet+LSTM SW predictor |
| **AIA** | 대기영상조립체 | Atmospheric Imaging Assembly onboard SDO; 4-telescope full-disk EUV/UV/visible imager |
| **SDOML** | SDO ML 데이터셋 | Galvez+ 2019 ML-ready SDO dataset, 512×512 @ 4.8″/px, 6-min cadence |
| **OMNIWEB** | OMNIWEB DB | NASA in-situ SW speed/IMF dataset at L1 |
| **CH (Coronal Hole)** | 코로나 홀 | Open-field, low-density EUV-dark region — source of fast SW |
| **AR (Active Region)** | 활동 영역 | Magnetically complex EUV-bright region — associated with slow SW / CMEs |
| **History H, Delay D** | 이력/지연 | Number of input days (H) and lead time (D) hyperparameters |
| **HSE** | 고속풍 강화 이벤트 | High Speed Enhancement (Owens+ 2005; Jian+ 2015 algorithm) |
| **TS (Threat Score)** | 위협 점수 | TP / (TP+FN+FP) for HSE detection |
| **WSA-ENLIL** | WSA-ENLIL | Wang-Sheeley-Arge empirical + ENLIL MHD operational SW model |
| **PFSS** | 전위장 광원 표면 | Potential Field Source Surface (Altschuler & Newkirk 1969) coronal field model |
| **Persistence Model** | 지속성 모델 | Forecast = today's SW (or 27-day prior) — strong but trivial baseline |

---

## 5. 예상 질문(Q&A) / Anticipated Questions

### Q1. 왜 193 Å와 211 Å만 사용했는가? / Why only 193 Å and 211 Å?

**한국어:** 이 두 채널은 코로나 홀(어두움)과 활동 영역(밝음) 대비를 가장 명확히 보여주는 ~1.6-2 MK 코로나 라인이다. 211 Å (Fe XIV, ~2 MK)는 AR을 강조하고, 193 Å (Fe XII, ~1.6 MK)는 CH를 잘 드러낸다. 둘을 함께 사용하면 fast/slow 두 SW source의 정보를 모두 입력할 수 있어 효율적이다. 또한 사전학습된 ConvNet은 3채널 RGB 입력을 기대하므로 (193, 211, 채널 복제) 등으로 채널 수를 맞추기에도 적합하다.

**English:** These two channels best highlight the contrast between dark CHs and bright ARs at coronal temperatures of ~1.6-2 MK. 211 Å (Fe XIV, ~2 MK) emphasizes ARs while 193 Å (Fe XII, ~1.6 MK) reveals CHs. Together they encode both fast- and slow-wind sources. They also slot into the 3-channel RGB input expected by the ImageNet-pretrained ConvNet.

### Q2. 일평균(daily-averaged) SW 속도를 사용한 이유는? / Why daily averages instead of hourly?

**한국어:** 일중 SW 변동이 평균값에 비해 작고, 그 변동 자체가 측정 불확실성(σ)을 정의한다. 일평균은 SNR이 높고, AIA 풀디스크 이미지(자전 시간 ~27일)와의 시간 스케일과도 잘 맞는다. 또한 short-time fluctuation (Alfvén waves, microstreams)을 제거해 거시적 CH/AR 구조와의 매핑에 집중할 수 있다.

**English:** Diurnal SW fluctuations are small relative to the daily mean, and that fluctuation itself defines the per-day uncertainty σ. Daily means have higher SNR and align better with the rotational timescale of full-disk AIA snapshots, allowing the model to focus on macroscopic CH/AR structures and ignore Alfvén waves / microstreams.

### Q3. History H와 Delay D는 무엇인가? / What are History H and Delay D?

**한국어:** **H**는 입력으로 사용하는 EUV 영상의 일수, **D**는 가장 최근 입력일과 예측 대상일 사이의 간격이다. 예를 들어 예측일이 T이고 H=4, D=3이면 입력은 T-6, T-5, T-4, T-3 네 일치이다(즉 3일 lead time). 논문은 H = 1..4, D = 1..4의 16개 조합을 모두 학습하여 lead time vs. 정확도 trade-off를 정량화했다.

**English:** **H** is the number of EUV input days; **D** is the gap between the most recent input day and the prediction day. For T, H=4, D=3 the inputs are T-6, T-5, T-4, T-3 (i.e. a 3-day lead time). The paper trains all 16 combinations (H=1..4 × D=1..4) to map the lead-time vs. skill trade-off.

### Q4. 모델은 어떻게 평가되는가? / How is the model evaluated?

**한국어:** 세 가지 회귀 지표 — RMSE (sqrt(χ²)), 측정 σ로 가중된 χ²_red, Pearson r (Fisher z-space 평균) — 와 한 가지 이벤트 지표 HSE Threat Score를 사용한다. 5-fold cross-validation을 수행하며, 데이터를 20일 단위 batch로 자른 뒤 5 fold에 무작위 배정하여 시간적 데이터 누수를 막는다.

**English:** Three regression metrics — RMSE (= √χ²), σ-weighted χ²_red, Pearson r (averaged in Fisher z-space) — plus one event-based metric, HSE Threat Score. 5-fold cross-validation is used, with 20-day contiguous batches randomly assigned to folds to prevent temporal leakage.

### Q5. 왜 단순한 baseline (persistence, mean, XGBoost, SVM)도 함께 비교하는가? / Why also compare to naive baselines?

**한국어:** SW은 27일 Carrington 자전 주기로 강한 자기상관을 가져 N-day persistence 모델이 의외로 강력한 baseline이다. ML 논문에서 흔한 함정은 단순 회귀가 이를 이기지 못하는 경우인데, Upendran은 이 점을 정직하게 다루며 WindNet이 모든 baseline을 능가함을 보인다(특히 작은 H, D에서).

**English:** SW exhibits strong 27-day Carrington autocorrelation, making N-day persistence a deceptively strong baseline. A common pitfall in ML-for-space-weather papers is failing to beat persistence; Upendran explicitly benchmarks against persistence, naive mean, XGBoost, SVM regression and shows WindNet outperforms all (especially at small H, D).

### Q6. 모델은 어떻게 "물리"를 학습했는가? / How does the model "learn physics"?

**한국어:** Selvaraju+ (2016) Grad-CAM류 활성 맵을 SW 속도 별로 평균하여 시각화한다. fast wind (≳500 km/s) 예측 시 D ≈ 3-4 일 전의 193 Å CH 영역에 강한 활성이, slow wind 예측 시 AR 영역에 활성이 나타난다. 이는 Krieger+ (1973), Wang & Sheeley (1990) 등의 경험칙과 정성적으로 일치하며, 모델이 명시적 물리 prior 없이도 CH/AR-SW 연관을 "발견"했음을 시사한다.

**English:** They visualise Grad-CAM-style activation maps averaged conditional on predicted SW speed. For fast-wind (≳500 km/s) predictions, strong activation appears at 193 Å CHs ~3-4 days prior; for slow-wind predictions, activation concentrates at ARs. This matches Krieger+ (1973) and Wang & Sheeley (1990) heuristics, suggesting the network "discovered" the CH/AR-SW link without any explicit physics prior.

### Q7. 한계는 무엇인가? / What are the limitations?

**한국어:** (1) ICME(인터플라네터리 CME) 효과를 제외하지 않아 transient event가 학습에 잡음으로 들어간다 (170 ICMEs / 336일). (2) 일평균만 사용해 시간 해상도가 거칠다. (3) 단일 cycle 24의 데이터(2011-2018)만 사용하여 cycle-dependent bias 가능. (4) 자기장 정보(SDO/HMI)를 사용하지 않아 IMF Bz 예측은 미해결로 남는다. (5) r=0.55가 절대적으로 높지는 않으며, 운영 예보로 가려면 추가 개선이 필요하다.

**English:** (1) ICMEs are not excluded (170 ICMEs over 336 days), injecting transient noise. (2) Only daily averages used → coarse temporal resolution. (3) Trained on a single cycle (24, 2011-2018), risking cycle-dependent bias. (4) No magnetograms (SDO/HMI), so IMF Bz prediction remains open. (5) r = 0.55 is not high in absolute terms — operational deployment needs further improvement.

---

## 6. 읽기 전략 / Reading Strategy

**한국어:**
1. Abstract + Plain Language Summary (p.2)로 메인 결과를 잡는다.
2. §2 Data and Metrics — preprocessing (log scaling Eq. 1, 2), 5-fold cross-validation, history/delay 정의 (Fig. 5), 평가 지표 (Eqs. 3-6) 를 차례로 이해한다.
3. §3 Modelling — WindNet 아키텍처 (ConvNet + LSTM), 다섯 baseline (mean, persistence, XGBoost, SVM)을 파악한다.
4. §4 Results — 16개 (H, D) 조합에 대한 r, χ²_red, RMSE, TS 표/그림을 본 baseline 대비 우위와 lead-time 의존성을 확인한다.
5. §5 Activation analysis — fast/slow 케이스별 평균 활성 맵에서 CH/AR 매칭을 본다.

**English:**
1. Read Abstract + Plain Language Summary (p.2) for the headline result.
2. §2 Data and Metrics — work through preprocessing (log scaling Eqs. 1, 2), 5-fold CV, history/delay definitions (Fig. 5), and evaluation metrics (Eqs. 3-6).
3. §3 Modelling — understand the WindNet architecture (ConvNet + LSTM) and the five baselines (mean, persistence, XGBoost, SVM).
4. §4 Results — read the (H, D)-grid tables/plots for r, χ²_red, RMSE, TS to assess baseline dominance and lead-time dependence.
5. §5 Activation analysis — examine averaged activation maps conditional on fast vs. slow predictions and the CH/AR correspondence.

---

## 7. 핵심 수식 미리보기 / Key Equations Preview

### EUV 전처리 / EUV Preprocessing
$$
x(193) = \begin{cases} \log(125.0) & x \le \log(125.0) \\ \log(5000.0) & x \ge \log(5000.0) \\ x & \text{else}\end{cases}
$$
$$
x(211) = \begin{cases} \log(25.0) & x \le \log(25.0) \\ \log(2500.0) & x \ge \log(2500.0) \\ x & \text{else}\end{cases}
$$

### 평가 지표 / Evaluation Metrics
$$
\chi^2 = \tfrac{1}{N}\sum_i (\hat y_i - y_i)^2, \qquad
\chi^2_{\text{red}} = \tfrac{1}{N}\sum_i \frac{(\hat y_i - y_i)^2}{\sigma_i^2}, \qquad
r = \frac{\sum_i (y_i - \bar y)(\hat y_i - \bar{\hat y})}{\sigma_y \sigma_{\hat y}}.
$$

### HSE Threat Score
$$
\mathrm{TS} = \frac{TP}{TP + FN + FP}.
$$

---

## 8. 읽은 후 점검할 것 / Post-Reading Checklist

- [ ] WindNet의 정확한 layer 구성 (Inception-v3 backbone? freeze 전략?)을 정리
- [ ] (H, D) 그리드의 best 조합 (논문은 어떤 H, D에서 r=0.55 달성?)
- [ ] WSA-ENLIL과의 정량 비교 표 작성
- [ ] CH/AR 활성 맵 검증을 위한 segmentation 알고리즘 (binary mask, 강도 임계)
- [ ] Threat Score가 H=1, D=1 persistence 대비 얼마나 향상됐는지

- [ ] Locate WindNet's exact layer composition (Inception-v3 backbone? freezing strategy?)
- [ ] Identify the best (H, D) combination (where does r = 0.55 occur?)
- [ ] Tabulate quantitative comparison against WSA-ENLIL
- [ ] Note segmentation algorithm for CH/AR activation validation (binary masks, intensity thresholds)
- [ ] Threat Score uplift relative to H=1, D=1 persistence baseline

---

*Briefing prepared 2026-04-27 for paper #35 in Space_Weather curriculum.*
