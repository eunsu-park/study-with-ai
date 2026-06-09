---
title: "Reading Notes — Camporeale (2019): The Challenge of Machine Learning in Space Weather"
date: 2026-04-27
topic: Space Weather
tags: [machine_learning, space_weather, gray_box, uncertainty_quantification, imbalanced_data]
---

# Camporeale 2019 — The Challenge of Machine Learning in Space Weather: Nowcasting and Forecasting

## 1. Core Contribution / 핵심 기여

**한국어.** 이 논문은 우주기상 예보(nowcasting/forecasting)에 머신러닝(ML)을 적용한 약 25년간의 연구를 종합하고, 향후 10년간 커뮤니티가 풀어야 할 핵심 도전 과제들을 제시하는 "Grand Challenge Review" 논문이다. Camporeale은 (1) 지자기 지수, (2) 정지궤도 상대론적 전자, (3) 태양 플레어, (4) CME 도달 시간, (5) 태양풍 속도 등 다섯 가지 분야에서 ML이 어떻게 사용되어 왔는지를 비판적으로 정리하고, "gray-box" 접근(물리 기반과 데이터 기반의 결합)과 확률적(probabilistic) 예보 패러다임으로의 전환을 강력하게 주장한다. 또한 ML 워크플로(문제 정식화 → 데이터 전처리 → 알고리즘 선택 → overfitting 회피 → 평가)를 정리한 교육적 절(§3-4)과 21개 metric을 정리한 Table 3을 제공함으로써, 우주기상 ML 연구의 사실상의 표준 참고문헌이 되었다. 마지막으로 6개의 미해결 문제(information, gray-box, surrogate, uncertainty, "too often too quiet" imbalance, knowledge discovery)를 제기한다.

**English.** This Grand Challenge review surveys roughly 25 years of machine learning (ML) applications in space weather nowcasting and forecasting, and proposes a roadmap of open problems for the next decade. Camporeale critically reviews five application areas — (1) geomagnetic indices, (2) relativistic electrons at GEO, (3) solar flares, (4) CME arrival time, (5) solar wind speed — and forcefully advocates for two paradigm shifts: (a) the *gray-box* approach combining physics-based and data-driven models, and (b) probabilistic, uncertainty-aware forecasting. The paper also serves as a tutorial (§3-4) with a comprehensive metrics table (Table 3, 21 metrics) and codifies the ML workflow (problem formulation → preprocessing → algorithm selection → overfitting → evaluation). It closes with six grand challenges: the information, gray-box, surrogate, uncertainty, *"too often too quiet"* (imbalance), and knowledge-discovery problems. The review has become the de facto reference for the ML-in-space-weather community.

## 2. Reading Notes / 읽기 노트

### 2.1 §1-2: AI History and the ML Renaissance / AI 역사와 ML 르네상스 (pp. 1-5)

**한국어.** 저자는 AI의 역사를 "spring과 winter의 주기"로 묘사하며, 1956 Dartmouth workshop 이후 여러 차례의 환멸을 거쳤다고 정리한다. 그러나 *"this time is different"*라는 입장에서, 현재의 AI 봄은 (a) 빅데이터 시대, (b) GPU 가속, (c) 거대 IT 기업의 투자 — 세 가지 요소가 동시에 충족된 역사적으로 유례없는 시기라고 본다. NASnet 신경망이 ImageNet을 정복하기 위해 4일간 500개의 GPU를 사용한 사례를 제시하며, scikit-learn, theano, PyMC 등의 오픈소스 생태계가 비전문가의 진입 장벽을 낮췄다고 평한다. 우주기상 데이터(ACE, Wind, DSCOVR, SOHO, SDO, OMNI, VAP, GOES, POES, GPS, DMSP, ground magnetometers)는 모두 "successful ML application의 ingredient"를 갖추고 있다고 강조한다.

**English.** The author frames AI history as cycles of "springs and winters" since the 1956 Dartmouth workshop. He argues *"this time is different"* because three enabling factors now coincide for the first time: (a) the era of big data, (b) GPU computing (citing NASnet using 500 GPUs for 4 days on ImageNet/CIFAR-10), and (c) massive investment by IT giants (Google, Facebook). The open-source Python stack (theano, scikit-learn, PyMC, astroML, emcee) has democratized access. Crucially, space weather has all the ingredients needed for successful ML: decades of public in-situ and remote-sensing data from ACE, Wind, DSCOVR, SOHO, SDO, OMNI, Van Allen Probes, GOES, POES, GPS, DMSP, and ground magnetometers (Table 1).

### 2.2 §3: ML in Space Weather and Gray-Box / 우주기상에서의 ML과 회색상자 (pp. 5-12)

**한국어.** 저자는 ML이 우주기상에 새로운 것이 아니며 1990년대 초부터 신경망 예측 시도가 있었다고 지적한다. 그러나 *"왜 계속 시도하는가?"*라는 회의적 질문에 대해 두 가지 답을 제시한다: (1) 모든 것을 시도한 것이 아니다(특히 CNN은 거의 미접근), (2) 성공의 세 가지 enabler가 이제야 갖춰졌다.

**Gray-box 패러다임 (Table 2):**
| 항목 | White-box (physics) | Black-box (ML) |
|------|---------------------|----------------|
| Computational cost | 일반적으로 비쌈, real-time 어려움 | 훈련은 비쌀 수 있으나 실행은 매우 빠름 |
| Robustness | 미관측 데이터에 강건 | 훈련 범위 밖에서는 외삽 불가 |
| Assumptions | 물리 근사 기반 | 최소한의 가정 |
| Consistency with obs | 사후(a posteriori) 검증 | 사전(a priori) 강제 |
| Toward gray-box | Data-driven parameterization of inputs | Physics-based constraints |
| Uncertainty | Monte-Carlo ensemble 필요 | 내장 가능 |

저자는 우주기상이 "gray-box의 최적 후보"라고 주장하며, 세 가지 구현 방식을 제시한다: (1) Bayesian 역문제로 white-box 파라미터를 데이터에서 추정 ($p(\mathbf{m}|\mathbf{d}) \propto p(\mathbf{d}|\mathbf{m})p(\mathbf{m})$), (2) ML이 white-box의 한 모듈을 surrogate로 대체, (3) ensemble combination.

§3.1-3.4은 ML 작업 분류이다: supervised regression ($y = f(\mathbf{x}) + \varepsilon$), supervised classification (logistic regression with sigmoid + cross-entropy), unsupervised clustering (k-means, SOM), dimensionality reduction (PCA). Box 1은 NN을 수학적으로 정의: $y(\mathbf{x}) = \sum_{i=1}^q w_i \sigma\left(\sum_{j=1}^{N_i} a_{ij}x_j + b_i\right)$.

**English.** Camporeale notes that ML in space weather is not new — neural-network predictions go back to the early 1990s — but answers the skeptical *"hasn't everything been tried?"* with two responses: (i) not everything (CNNs barely touched until 2018), and (ii) the three enablers (data, GPUs, software) are only now in place.

The **gray-box paradigm** (Table 2) compares white-box (physics) and black-box (data) approaches across cost, robustness, assumptions, observation consistency, and uncertainty handling. Three implementations are sketched: (1) Bayesian inverse problems to estimate physics parameters via $p(\mathbf{m}|\mathbf{d}) \propto p(\mathbf{d}|\mathbf{m})p(\mathbf{m})$; (2) ML as surrogate for a costly module in the physics chain (Sun → bow shock → magnetosphere → radiation belt → ionosphere); (3) weighted ensembles of physics and ML predictions.

§3.1-3.4 establish the ML task taxonomy: supervised regression $y = f(\mathbf{x}) + \varepsilon$ minimizing MSE; supervised classification using logistic regression $\hat{y} = \sigma(z) = 1/(1+e^{-z})$ with cross-entropy $C(y,z) = (y-1)\log(1-\sigma(z)) - y\log(\sigma(z))$; unsupervised clustering (k-means, self-organizing maps); and dimensionality reduction (PCA). Box 1 formalizes a neural network as $y(\mathbf{x}) = \sum_{i=1}^q w_i \sigma\left(\sum_{j=1}^{N_i} a_{ij}x_j + b_i\right)$, citing Cybenko (1989) for the universal approximation theorem.

### 2.3 §4: ML Workflow / ML 워크플로 (pp. 12-18)

**한국어.** 저자는 ML 적용 워크플로를 5단계로 정리한다.

1. **Problem formulation**: 회귀/분류/군집화/차원축소 중 어느 것인가? 물리적 동기로 입력 변수가 정해지는가? 시간 인과성을 고려했는가?
2. **Data selection and preprocessing**: 결측치, 이상치, 데이터 augmentation으로 imbalance 완화. Information theory (mutual information, transfer entropy)로 인과적으로 의미있는 입력 변수만 선택 (Wing et al., 2016).
3. **Algorithm selection**: parametric (linear, polynomial regression, NN) vs non-parametric (k-means, GP, SVM, kernel methods). 정확도, 훈련 시간, 복잡도 trade-off.
4. **Overfitting and model selection**: 다항식 회귀(Figure 3) — order 9면 10개 점을 정확히 통과하지만 일반화 실패. 해법: train/validation/test split, cross-validation, BIC/AIC, MDL.
5. **Testing and metrics**: 시계열 데이터의 random split 위험성. *Skill score*: $\text{skill} = \text{(model)} - \text{(persistence/climatology baseline)}$. **Bias-variance decomposition**: 복잡한 모델 = low bias, high variance / 단순 모델 = high bias, low variance.

**Table 3 metrics 요약:**
- *Deterministic classification*: TPR, TNR, FPR, PPV, ACC, F1, **HSS**, **TSS** (TSS = TPR − FPR, "unbiased w.r.t. class-imbalance").
- *Probabilistic classification*: **Brier score** $BS = \frac{1}{N}\sum (f_i - o_i)^2$, ignorance score (logarithmic).
- *Deterministic regression*: MSE, RMSE, NRMSE, MAE, ARE, correlation $cc$, prediction efficiency $PE$, median symmetric accuracy.
- *Probabilistic regression*: Continuous Rank Probability Score (CRPS), ignorance.

**English.** §4 codifies a 5-step ML workflow.

1. **Problem formulation** — regression/classification/clustering/dim-reduction? Physically motivated inputs? Time causality?
2. **Data selection & preprocessing** — gap handling, outlier handling, data augmentation for imbalance. Use information theory (mutual information, transfer entropy) to select causal inputs (e.g., Wing et al., 2016 on solar wind drivers of radiation-belt electrons).
3. **Algorithm selection** — parametric (linear/polynomial regression, NN) vs non-parametric (k-means, GP, SVM, kernel methods); trade-off between accuracy, training time, scalability.
4. **Overfitting & model selection** — Figure 3 shows polynomial regression where order $l=9$ fits 10 points exactly but generalizes poorly. Cures: train/val/test split, k-fold cross-validation, BIC/AIC/MDL.
5. **Testing and metrics** — random split is dangerous on time series due to autocorrelation; use temporally disjoint test sets. *Skill score* = model performance minus a persistence or climatology baseline. **Bias-variance decomposition**: flexible (high-capacity) models have low bias and high variance; rigid models, the opposite.

The **Table 3 metric zoo** (the most-cited part of §4):
| Class | Metric | Formula |
|-------|--------|---------|
| Det. classification | TPR, FPR, ACC, F1 | standard |
| Det. classification | HSS₁, HSS₂, **TSS** | TSS = TPR − FPR |
| Prob. classification | **Brier score** | $\frac{1}{N}\sum (f_i - o_i)^2$ |
| Det. regression | MSE, RMSE, MAE, $cc$, $PE$ | standard |
| Prob. regression | **CRPS**, ignorance | $\frac{1}{N}\sum_i \int (\hat{F}_i(z) - H(z-y_i))^2 dz$ |

TSS is preferred for imbalanced flare data because it is unbiased w.r.t. class prior; F1 and accuracy are class-imbalance-sensitive.

### 2.4 §5: Application Reviews / 응용 분야 리뷰 (pp. 18-38)

**한국어.** §5.1 **지자기 지수**: $K_p$, $D_{st}$ 예측은 1990년대부터 NN의 시범 케이스였다. SWPC/NOAA는 2010-2018년 Wing et al. (2005) NN 모델로 1시간 ahead $K_p$를 운영하다가 2018년 Geospace MHD 모델로 교체. NARMAX 모델 (Boynton, Balikhin et al., 2011-2018)은 $D_{st}$, $K_p$를 연속적 nonlinear ARX 모델로 학습. **§5.2 GEO 상대론적 전자**: Reeves et al. SHELLS, Boynton NARMAX, Balikhin SNB³GEO 등이 운영. **§5.3 SEP**: Fernandes 2015. **§5.4 태양 플레어**: 핵심 단원. Bobra & Couvidat (2015)가 SDO/HMI 벡터 자기장 13개 SHARP 파라미터로 SVM ≥M1 24-hr 예측, **TSS ~ 0.8** 달성, 1:53의 강한 imbalance를 active region 단위 split으로 처리. Nishizuka et al. 2017 (DeFN) — 2015년 데이터에 TSS ~ 0.6, 0.8 (M, C). FLARECAST EU-H2020 (Florios et al. 2018) — NN, SVM, RF 비교, RF가 TSS ~ 0.6으로 약간 우위. Jonas et al. 2018 — fixed-time forecast → time-window forecast로 재정의, 자동 추출 feature와 hand-crafted feature 비교, TSS ~ 0.8. Huang et al. 2018 — 100×100 active region patch를 CNN에 직접 입력 (full black-box), TSS C-class ~ 0.5, X-class ~ 0.7.

**§5.5 CME**: J. Liu et al. 2018 — 182 geo-effective CMEs, SVM, RMSE ~ 7.3 hours. Bobra & Ilonidis 2016 — flare/CME 동반 여부 SVM TSS ~ 0.7. **§5.6 태양풍 속도**: Wintoft & Lundstedt 1997, 1999 — PFSS 자기장 + RBF NN, 3-day ahead RMSE ~ 90 km/s, $cc$ ~ 0.58. Yang et al. 2018 — PFSS 7개 attributes + 27일 전 속도, 4-day ahead $cc$ ~ 0.74, RMSE ~ 68 km/s (state of the art). 그러나 *persistence model*이 27-day 주기로 인해 매우 강한 baseline ($cc$ ~ 0.5, RMSE ~ 95 km/s).

**핵심 메시지 (Recapitulation)**: ML 방법이 통계 방법보다 우수함은 입증되었으나, simple persistence/empirical 모델을 이기기는 여전히 어렵다. SDO 8년 데이터(< 1 solar cycle)가 충분한지는 미해결.

**English.** §5 reviews five application areas with critical assessments.

**§5.1 Geomagnetic indices**: $K_p$, $D_{st}$ have been NN testbeds since the 1990s. SWPC/NOAA ran Wing et al. (2005) NN for 1-hr ahead $K_p$ from 2010-2018, replaced by the Geospace MHD model in 2018. The NARMAX (Nonlinear AutoRegressive Moving Average with eXogenous inputs) family (Boynton, Balikhin et al.) provides explicit nonlinear models for $D_{st}$, $K_p$.

**§5.2 GEO relativistic electrons**: NARMAX, SHELLS, SNB³GEO operational at SWPC.

**§5.3 SEP**: Fernandes 2015.

**§5.4 Solar flares (key subsection)**:
| Study | Method | Result |
|-------|--------|--------|
| Bobra & Couvidat 2015 | SVM, 13 SHARP params, AR-split | TSS ~ 0.8 (≥M1, 24h) |
| Nishizuka et al. 2017 (DeFN) | NN, SDO/HMI + chromosphere | TSS ~ 0.8 (M), 0.6 (C) |
| Florios et al. 2018 (FLARECAST) | NN/SVM/RF comparison | RF best, TSS ~ 0.6 |
| Jonas et al. 2018 | Time-window forecast, auto + hand-crafted features | TSS ~ 0.8 |
| Huang et al. 2018 | CNN on 100×100 AR patches, full black-box | TSS C ~ 0.5, X ~ 0.7 |

The flare imbalance ratio is 1:53 for 24-hr ≥M1 prediction; Bobra & Couvidat's active-region-disjoint split is the canonical methodology for avoiding data leakage.

**§5.5 CMEs**: J. Liu et al. 2018 — SVM on 182 geo-effective CMEs, 18 features from LASCO + OMNI, RMSE ~ 7.3 hr arrival time. Bobra & Ilonidis 2016 — SVM to distinguish flare-only vs flare+CME ARs, TSS ~ 0.7. Inceoglu et al. 2018 — 3-class SVM/NN, TSS ~ 0.9.

**§5.6 Solar wind speed**: Wintoft & Lundstedt 1997, 1999 — RBF NN on PFSS source-surface field, 3-day ahead RMSE ~ 90 km/s, $cc$ ~ 0.58. Yang et al. 2018 — PFSS attributes + lagged speed (one solar rotation = 27 days), 4-day ahead $cc$ ~ 0.74, RMSE ~ 68 km/s — state of the art. Yet a *27-day persistence baseline* already gives $cc$ ~ 0.5, RMSE ~ 95 km/s — a strong baseline.

**Recapitulation**: ML beats classical statistical methods, but persistence/empirical baselines remain hard to beat for solar wind. An open question: are 8 years of SDO data (< 1 full solar cycle) sufficient training data?

### 2.5 §6: New Trends / 새로운 동향 (pp. 38-41)

**한국어.** 저자는 ML 응용을 (a) 인간 작업의 자동화·가속과 (b) 지식 발견(knowledge discovery)으로 양분한다. AlphaGo가 인간이 모르던 수를 발견한 것이 후자의 예. 우주기상에서 흥미로운 동향:

1. **Physics-Informed Neural Networks (PINN)**: Raissi & Karniadakis 2017a,b — NN으로 PDE를 풀고 자유 파라미터를 데이터에서 추정 (Burgers eq. shocks).
2. **Auto-ML**: 유전 알고리즘으로 NN 구조 자체를 탐색.
3. **Adversarial training**: $\mathbf{x}' = \mathbf{x} + \varepsilon \cdot \text{sign}(\nabla_\mathbf{x} L(\mathbf{x},y))$ (Goodfellow 2015). GAN으로 TEC map 생성 (Z. Chen et al. 2019).

**English.** §6 distinguishes two ML purposes: (a) automating tasks already mastered by humans; (b) genuine *knowledge discovery* (e.g., AlphaGo's "move 37" — a strategy unknown to centuries of human play). Three trends predicted to enter space physics:

1. **Physics-Informed Neural Networks** (Raissi & Karniadakis 2017a,b) — solving PDEs by enforcing the equation as a soft constraint on a NN's output, exploiting the fact that NNs are analytically differentiable. Already shown to capture Burgers-equation shocks.
2. **Auto-ML** — using genetic algorithms to search architecture space, particularly promising for multi-domain problems (radiation belts, ring current, solar wind, ionosphere).
3. **Adversarial training and GANs** — adversarial examples $\mathbf{x}' = \mathbf{x} + \varepsilon \cdot \text{sign}(\nabla_\mathbf{x} L(\mathbf{x},y))$ (Goodfellow et al. 2015) increase robustness; GANs generate synthetic data, e.g. Z. Chen et al. 2019 generating TEC maps.

### 2.6 §7: Conclusions and Six Grand Challenges / 결론과 6대 과제 (pp. 41-42)

**한국어.** 1. **The information problem** — 예보에 필요한 *최소 물리적 정보*는 무엇인가? 자기력선상도(magnetograms)와 EUV로 플레어 예측이 충분한가?
2. **The gray-box problem** — 물리 모델과 데이터 모델을 어떻게 결합할 것인가? Bayesian data assimilation, parameter estimation, ensemble.
3. **The surrogate problem** — 어느 모듈을 ML surrogate로 대체할 수 있는가? Multi-fidelity 모델. 보존법칙 같은 물리 제약을 강제할 수 있는가?
4. **The uncertainty problem** — 단일 예측을 넘어 확률적 예측으로의 전환. Non-intrusive UQ (Monte Carlo ensemble).
5. **The "too often too quiet" problem** — 우주기상은 "조용한 시간이 많고 폭풍은 짧은" 극단적 imbalance. Synthetic data augmentation의 정당성?
6. **The knowledge discovery problem** — black-box 모델에서 어떻게 새로운 물리적 통찰을 추출할 것인가? "make it work vs make it understandable"의 딜레마.

**English.** §7.1 lists six grand challenges that have become the agenda of the field:

1. **The information problem** — What is the *minimum physical information* required to make a forecast? Is photospheric magnetic + EUV imagery enough for flares?
2. **The gray-box problem** — How to optimally combine physics-based and data-driven models? Approaches: Bayesian data assimilation, parameter estimation, ensembles.
3. **The surrogate problem** — Which modules in the space weather chain can be replaced by black-box surrogates? Multi-fidelity models. Can surrogates enforce physical constraints (conservation laws)?
4. **The uncertainty problem** — Most space weather services give single-point predictions. The community must adopt probabilistic forecasting via non-intrusive UQ (Monte Carlo ensembles).
5. **The "too often too quiet" problem** — Space weather is severely class-imbalanced. Is synthetic data augmentation legitimate without degrading information content?
6. **The knowledge discovery problem** — How to distill physical understanding from a black-box ML model? The "make it work vs make it understandable" dilemma — perhaps the deepest issue, since trust requires interpretability.

## 3. Key Takeaways / 핵심 시사점

**1. Gray-box is the future / 회색상자가 미래다.**
- **Korean.** 순수 black-box도 순수 white-box도 아닌 hybrid가 우주기상에 가장 적합하다. 물리 모델은 외삽에 강하지만 비싸고, ML은 빠르지만 훈련 범위 밖에서는 위험하다.
- **English.** Pure black- or white-box approaches are both suboptimal. Hybrids exploit physics's robustness for unseen regimes and ML's speed for routine forecasting.

**2. Probabilistic forecasting is mandatory / 확률적 예보로의 전환은 필수다.**
- **Korean.** Single-point forecast → calibrated probabilistic forecast. Reliability diagram, Brier score, CRPS로 평가.
- **English.** The community must shift from single-point to calibrated probabilistic forecasts, validated via reliability diagrams, Brier score, and CRPS.

**3. Class imbalance demands TSS, not accuracy / 클래스 불균형은 TSS를 요구한다.**
- **Korean.** 1:53 ratio에서 accuracy ~ 98%는 의미 없음. TSS = TPR − FPR이 prior-invariant.
- **English.** With 1:53 imbalance, ~98% accuracy is meaningless. TSS = TPR − FPR is prior-invariant; HSS, F1 are alternatives. Brier and CRPS handle probabilistic forecasts.

**4. Temporal data leakage must be prevented / 시간적 데이터 누설을 막아야 한다.**
- **Korean.** Random split은 자기상관이 있는 시계열에서 metric을 인위적으로 부풀린다. Active region 단위 또는 시간적으로 단절된 split이 필수.
- **English.** Random splits inflate metrics due to temporal autocorrelation. Bobra & Couvidat's active-region-disjoint split, or chronologically disjoint train/test, is the canonical fix.

**5. Persistence is a strong baseline / 지속성 모델은 강력한 baseline이다.**
- **Korean.** 태양풍 속도는 27-day 자전 주기로 인해 persistence가 $cc$ ~ 0.5을 자연스럽게 달성. *Skill score* (모델 − baseline)로 보고하는 것이 정직.
- **English.** For solar wind, a 27-day persistence baseline achieves $cc$ ~ 0.5 trivially. ML papers should report *skill score* (model − baseline), not absolute metric.

**6. Information theory chooses inputs / 정보 이론으로 입력을 선택하라.**
- **Korean.** Wing et al. (2016) — mutual information, transfer entropy로 인과적 lag와 driver를 찾음. "ingest rubbish"를 막는 정량적 도구.
- **English.** Mutual information and transfer entropy (Wing et al. 2016) identify causal lags and drivers, preventing "rubbish-in, rubbish-out".

**7. CNNs and full-image inputs are largely untapped / CNN과 이미지 입력은 미개척이다.**
- **Korean.** 2018년까지 거의 모든 플레어 ML은 hand-crafted feature 사용. Huang et al. 2018만 100×100 patch를 CNN에 직접 입력. 본격적 CNN 시대는 본 논문 이후에 시작.
- **English.** Up to 2018, almost all flare ML used hand-crafted features. Huang et al. 2018 was the lone exception feeding raw 100×100 AR patches to a CNN. The full CNN era began *after* this review.

**8. The "too often too quiet" problem is fundamental / 폭풍이 너무 드물다는 것은 본질적 문제이다.**
- **Korean.** 우주기상은 본질적으로 imbalance — 데이터가 늘어도 자연스럽게 해결되지 않을 수 있다. SMOTE 같은 oversampling이나 GAN 합성 데이터의 정당성은 미해결.
- **English.** Imbalance is structural to space weather, not a data-collection problem; it does not vanish with more data. The legitimacy of SMOTE oversampling or GAN-synthesized augmentation is unresolved.

## 4. Mathematical Summary / 수학적 요약

### 4.1 Bayesian inverse problem (Eq. 1)
$$p(\mathbf{m}|\mathbf{d}) \propto p(\mathbf{d}|\mathbf{m})\, p(\mathbf{m})$$
- $\mathbf{m}$ — model parameters (random vector). 모델 파라미터.
- $\mathbf{d}$ — observed data, related via forward model $F(\mathbf{m}) \approx \mathbf{d}$. 순방향 모델로 연결된 관측 데이터.
- $p(\mathbf{m}|\mathbf{d})$ — posterior PDF. 사후 확률 밀도.
- $p(\mathbf{d}|\mathbf{m}) \propto \exp(-\|F(\mathbf{m}) - \mathbf{d}\|^2 / 2\sigma^2)$ — likelihood (Gaussian noise). 우도.
- $p(\mathbf{m})$ — prior. 사전 분포.

### 4.2 Supervised regression (Eq. 2)
$$y = f(\mathbf{x}) + \varepsilon$$
- $\mathbf{x} \in \mathbb{R}^{N_i}$ — multidimensional input. 다차원 입력.
- $y \in \mathbb{R}$ — scalar output. 스칼라 출력.
- $f$ — unknown nonlinear map approximated from training set $\{\mathbf{x}_{obs}^i, y_{obs}^i\}$.
- $\varepsilon$ — stochastic noise (observation error + latent variables). 잠재 변수 효과.
- Cost: $MSE = \frac{1}{N_T}\sum (\hat{y}^i - y_{obs}^i)^2$ or $MAE = \frac{1}{N_T}\sum |\hat{y}^i - y_{obs}^i|$.

### 4.3 Neural network (Eq. 3, Box 1)
$$y(\mathbf{x}) = \sum_{i=1}^{q} w_i\, \sigma\!\left(\sum_{j=1}^{N_i} a_{ij} x_j + b_i\right)$$
- $\sigma(\cdot)$ — activation; sigmoid (limits 0,1) or ReLU ($\sigma(s)=\max(0,s)$). 활성화 함수.
- $a_{ij}, b_i$ — weights, biases of the hidden layer. $q$ — neurons. 뉴런 수.
- $w_i$ — output-layer weights. 출력층 가중치.
- *Cybenko (1989)*: any continuous function can be approximated for $q$ large enough. 임의의 연속 함수 근사 가능.
- Trained by backpropagation: gradient of the cost w.r.t. weights via chain rule + SGD.

### 4.4 Logistic regression / Sigmoid (Eq. 4)
$$\hat{y} = \sigma(z) = \frac{1}{1+e^{-z}}$$
- Squashes real-valued $z$ into $[0, 1]$ — interpretable as event probability. 사건 확률로 해석.

### 4.5 Cross-entropy cost (Eq. 5)
$$C(y, z) = (y - 1)\log(1 - \sigma(z)) - y\log(\sigma(z))$$
- $y \in \{0, 1\}$ — ground truth. 정답 레이블.
- $C \to \infty$ when $|y - \hat{y}| = 1$ (totally wrong); $C \to 0$ when $|y - \hat{y}| \to 0$.

### 4.6 Adversarial perturbation (Eq. 6, FGSM)
$$\mathbf{x}' = \mathbf{x} + \varepsilon\, \text{sign}\!\left(\nabla_{\mathbf{x}} L(\mathbf{x}, y)\right)$$
- $\varepsilon$ — small step. 작은 섭동 크기.
- Increases loss, exposes vulnerability; adversarial training adds these to training data. 강건성 향상.

### 4.7 Key metrics (Table 3)
- **TSS** (True Skill Statistic): $TSS = TPR - FPR = \frac{TP}{TP+FN} - \frac{FP}{FP+TN}$, range $[-1, 1]$, **unbiased w.r.t. class imbalance**.
- **HSS₁** (Heidke Skill Score): $HSS_1 = (TP+TN-N)/P = TPR(2 - 1/PPV)$, range $(-\infty, 1]$.
- **HSS₂**: $HSS_2 = \frac{2(TP\cdot TN - FN\cdot FP)}{P(FN+TN) + N(TP+FP)}$, range $[-1, 1]$, skill vs random.
- **Brier score**: $BS = \frac{1}{N}\sum_{i=1}^N (f_i - o_i)^2$, $f_i \in [0,1]$ probability, $o_i \in \{0,1\}$ outcome. Negatively oriented (BS=0 perfect).
- **CRPS**: $CRPS = \frac{1}{N}\sum_i \int_{-\infty}^{\infty}(\hat{F}_i(z) - H(z-y_i))^2\, dz$, generalization of MAE for probabilistic regression.

### 4.8 Worked Example: TSS for Imbalanced Flare Forecast / 워크드 예시
**한국어.** 24시간 ≥M1 플레어 예측, 1년 데이터, $P = 100$ flare days, $N = 5300$ quiet days (1:53). 모델 A: TP=80, FN=20, FP=300, TN=5000. TPR = 80/100 = 0.80, FPR = 300/5300 = 0.057. TSS = 0.80 − 0.057 = **0.74**. Accuracy = (80+5000)/5400 = 0.941. *Always-quiet* baseline: TPR=0, FPR=0, TSS = 0, Accuracy = 5300/5400 = **0.981**. 정확도가 모델 A보다 높지만 skill은 0이다.

**English.** 24-hr ≥M1 flare prediction with $P=100$ flare days, $N=5300$ quiet days (1:53 imbalance). Model A: TP=80, FN=20, FP=300, TN=5000 → TPR=0.80, FPR=0.057, **TSS=0.74**, Accuracy=0.941. *Always-quiet* baseline: TPR=0, FPR=0, **TSS=0**, Accuracy=0.981. The trivial baseline has *higher* accuracy than Model A but zero skill — illustrating why TSS dominates the flare literature.

## 5. Paper in the Arc of History / 역사 속의 논문

```
1956 Dartmouth — birth of AI
   |
1990s Lundstedt & Wintoft, Gleisner et al. — first NN for Dst, Kp
   |
2005 Wing et al. — operational Kp NN at SWPC (deployed 2010-2018)
   |
2012 Krizhevsky AlexNet — CNN renaissance (ImageNet)
   |
2015 Bobra & Couvidat — SVM flares with SHARP, TSS ~ 0.8
   |
2017 Camporeale, Carè, Borovsky — solar wind classification by ML
   |
2018 explosion — 28 ML-in-space-weather papers in one year (Figure 4)
   |
*** 2019 Camporeale Grand Challenge Review ***  ← THIS PAPER
   |
2020+ AIDA / PAGER EU projects — gray-box, UQ, adversarial training
   |
2022+ Transformers, foundation models, helio-foundation models
```

**한국어.** 본 논문은 1990년대 NN 시대와 2020년대 deep-learning/foundation-model 시대를 잇는 inflection point에 위치한다. 2018년 28편의 ML 논문 폭증 직후, Figure 4에 그 추세를 명시적으로 시각화하면서 도전 과제를 정의한 점에서 의의가 크다.

**English.** Camporeale 2019 sits at the inflection point between the 1990s NN era and the post-2020 deep-learning/foundation-model era. The 2018 explosion (28 ML-in-space-weather papers in a single year, Figure 4) is the paper's immediate context. By naming the six grand challenges, it set the agenda for the AIDA/PAGER EU programs and the modern UQ-aware, gray-box generation of papers.

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper | Year | Connection / 연결 |
|-------|------|-------------------|
| LeCun 1990 (#11 in reading list) | 1990 | Backpropagation foundation cited in Box 1 / 신경망 학습의 기반 |
| LeCun, Bengio, Hinton (#15) | 2015 | Deep learning Nature review, prerequisite / 딥러닝 리뷰, 필수 선수 |
| Bobra & Couvidat 2015 | 2015 | Canonical SVM flare paper, AR-disjoint split / 표준 플레어 SVM |
| Bobra & Ilonidis 2016 | 2016 | Flare/CME co-occurrence SVM, TSS ~ 0.7 |
| Wing, Johnson, Camporeale, Reeves 2016 | 2016 | Mutual information for radiation-belt drivers / 정보이론 적용 |
| Camporeale, Carè, Borovsky 2017 | 2017 | Solar wind classification ML, by same author / 동일 저자 |
| Florios et al. 2018 (FLARECAST) | 2018 | EU H2020 ML flare project / EU 프로젝트 |
| Huang et al. 2018 | 2018 | First full-CNN flare model on raw HMI patches / 본격 CNN |
| Camporeale, Chu, Agapitov, Bortnik 2019 | 2019 | UQ companion paper — calibrated Gaussian forecasts / UQ 후속 |
| Liemhon et al. 2018 | 2018 | Metric review for space weather / 평가 지표 리뷰 |

## 6.1 Additional Quantitative Highlights / 추가 정량적 하이라이트

**한국어.**
- **Figure 4 통계**: 1993-2018년 ML in space weather 논문 수가 1995년 ~ 5편/년에서 2018년 ~ 28편/년으로 약 5배 증가. 이 가속이 본 리뷰의 즉각적 동기.
- **Imbalance ratios**: Bobra & Couvidat 2015의 24-hr ≥M1 forecast positive:negative = 1:53. 12-hr는 더 극단적.
- **CME arrival error**: 32개 모델, 139개 forecast (2013-2017) 평균 MAE = 11.2-22.6 hr, σ ≈ 20 hr. *6년간 향상이 없다*는 점이 주목할 결과 (Riley et al. 2018).
- **Solar wind speed best**: Yang et al. 2018, 4-day ahead, $cc \sim 0.74$, $RMSE \sim 68$ km/s. Persistence baseline은 $cc \sim 0.5$, $RMSE \sim 95$ km/s.
- **NASnet GPU usage**: 500 GPUs × 4 days for ImageNet/CIFAR-10 architecture search — ML compute scale.

**English.**
- **Figure 4 stats**: ML-in-space-weather publications grew from ~5/yr in 1995 to ~28/yr in 2018, a 5x acceleration motivating this review.
- **Imbalance ratios**: Bobra & Couvidat (2015) 24-hr ≥M1 positive:negative = 1:53; 12-hr forecasts even more extreme.
- **CME arrival**: Riley et al. 2018 statistical analysis of 32 models / 139 forecasts gives MAE = 11.2-22.6 hr, σ ≈ 20 hr — *no substantial improvement in 6 years*.
- **Solar wind**: Yang et al. 2018 4-day ahead state-of-the-art is $cc \sim 0.74$, $RMSE \sim 68$ km/s vs. persistence baseline $cc \sim 0.5$, $RMSE \sim 95$ km/s.

## 6.2 Why This Review Was Necessary / 이 리뷰가 필요했던 이유

**한국어.** 2018년 한 해에만 28편의 ML 논문이 출간되면서, 우주기상 커뮤니티는 비교 불가능한 결과들의 홍수에 직면했다. 동일한 문제(예: 24-hr ≥M1 플레어 예측)에 대해서도 서로 다른 데이터셋, 서로 다른 split 방식, 서로 다른 metric으로 보고되어 "apple to apple" 비교가 불가능했다. Camporeale 2019의 표준 metric Table (Table 3), gray-box 패러다임의 명문화, 6대 도전 과제의 정의는 이 혼돈을 정리하는 메타-기여(meta-contribution)이다.

**English.** With 28 ML papers in a single year (2018), the community faced a flood of non-comparable results: same problem (e.g., 24-hr ≥M1 flares), different datasets, different splits, different metrics — apple-to-apple comparison impossible. Camporeale 2019's contributions are partly *meta-contributions*: the canonical metric Table 3, the formalization of the gray-box paradigm, and the naming of the six grand challenges, all of which gave the community a shared vocabulary.

## 6.25 Worked Example: Reliability Diagram / 신뢰도 다이어그램 워크드 예시

**한국어.** Calibrated probabilistic forecast의 검증 절차:
1. 모델이 예측한 확률을 10개 bin (0-0.1, 0.1-0.2, ..., 0.9-1.0)으로 나눈다.
2. 각 bin에 속한 sample들의 *실제 발생 빈도* (observed frequency)를 계산한다.
3. (predicted prob, observed freq)를 scatter plot으로 그림.
4. Perfect calibration = $y = x$ 대각선. 위쪽이면 under-confident, 아래쪽이면 over-confident.

예: bin 0.7 (모델이 0.65-0.75 확률을 예측한 모든 sample)에 100개 sample이 있을 때, 그 중 실제 사건 발생이 70번이면 calibrated. 50번이면 over-confident, 90번이면 under-confident.

**English.** Validation procedure for calibrated probabilistic forecasts:
1. Bin predicted probabilities into 10 buckets (0-0.1, 0.1-0.2, ..., 0.9-1.0).
2. Compute *observed event frequency* in each bucket.
3. Scatter (predicted prob, observed freq).
4. Perfect calibration = the $y=x$ diagonal. Above = under-confident; below = over-confident.

Example: in bucket 0.7 (model predicts 0.65-0.75) with 100 samples, if 70 events actually occurred → calibrated; 50 → over-confident; 90 → under-confident. Brier score and reliability diagrams are *complementary*: a model can have low Brier but be miscalibrated (and vice versa).

## 6.3 Limitations Acknowledged by the Author / 저자가 인정한 한계

**한국어.**
- 리뷰는 *불완전하고 다소 편향적*이라고 저자가 명시 (저자 본인의 연구 편향 가능성).
- Section 5는 다섯 분야에 한정 — ionosphere, plasmasphere, SEP은 §5.7에서 짧게만 다룸.
- 강화학습(RL)은 §6에서 AlphaGo만 언급, 우주기상 적용은 미래 과제.
- Reliability diagram, Platt scaling, isotonic regression은 metric으로만 짧게 언급.

**English.**
- The author explicitly admits the review is *"necessarily incomplete and somewhat biased"*.
- §5 is limited to five areas; ionosphere, plasmasphere, SEP get a brief §5.7.
- Reinforcement learning is mentioned only via AlphaGo in §6; RL for space weather is left as future work.
- Calibration tools (reliability diagrams, Platt scaling, isotonic regression) are cited but not analyzed in depth.

## 7. References / 참고문헌

- Camporeale, E. (2019). The Challenge of Machine Learning in Space Weather: Nowcasting and Forecasting. *Space Weather*, 17. doi:10.1029/2018SW002061. arXiv:1903.05192.
- Bobra, M. G., & Couvidat, S. (2015). Solar Flare Prediction Using SDO/HMI Vector Magnetic Field Data with a Machine-Learning Algorithm. *ApJ*, 798(2), 135.
- Bobra, M. G., & Ilonidis, S. (2016). Predicting Coronal Mass Ejections Using Machine Learning Methods. *ApJ*, 821(2), 127.
- Boynton, R., Balikhin, M., Wei, H.-L., & Lang, Z.-Q. (2018). Applications of NARMAX in Space Weather. *Machine Learning Techniques for Space Weather* (Camporeale, Wing, Johnson eds.), Elsevier, 203-236.
- Cybenko, G. (1989). Approximation by superpositions of a sigmoidal function. *Mathematics of Control, Signals, and Systems*, 2, 303-314.
- Florios, K. et al. (2018). Forecasting Solar Flares Using Magnetogram-based Predictors and Machine Learning. *Solar Physics*, 293.
- Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). Explaining and Harnessing Adversarial Examples. arXiv:1412.6572.
- Huang, X., et al. (2018). Deep Learning Based Solar Flare Forecasting Model. *ApJ*, 856, 7.
- Jonas, E., Bobra, M. G., Shankar, V., Hoeksema, J. T., & Recht, B. (2018). Flare Prediction Using Photospheric and Coronal Image Data. *Solar Physics*, 293.
- LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521, 436-444.
- Lundstedt, H., & Wintoft, P. (1994). Prediction of geomagnetic storms from solar wind data with the use of a neural network. *Annales Geophysicae*, 12.
- Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. *ICML*.
- Raissi, M., & Karniadakis, G. E. (2018). Hidden Physics Models. *J. Comp. Phys.*, 357.
- Wing, S., Johnson, J. R., Camporeale, E., & Reeves, G. D. (2016). Information theoretical approach to discovering solar wind drivers of the outer radiation belt. *JGR*, 121.
- Wintoft, P., & Lundstedt, H. (1999). Prediction of daily average solar wind velocity from solar magnetic field observations using neural networks. *Phys. Chem. Earth*, 24.
- Riley, P., et al. (2018). Forecasting the Arrival Time of Coronal Mass Ejections: Analysis of the CCMC CME Scoreboard. *Space Weather*, 16.
- Liu, J., Ye, Y., Shen, C., Wang, Y., & Erdélyi, R. (2018). A New Tool for CME Arrival Time Prediction Using Machine Learning Algorithms. *ApJ*, 855, 109.
- Yang, Y., Shen, F., Yang, Z., & Feng, X. (2018). Prediction of Solar Wind Speed at 1 AU Using an Artificial Neural Network. *Space Weather*, 16.
- Nishizuka, N., et al. (2017). Solar Flare Prediction Model with Three Machine-learning Algorithms. *ApJ*, 835, 156.
- Inceoglu, F., et al. (2018). Using Machine Learning to Investigate Solar Flare Energy Release. *ApJ*, 861, 128.
