---
title: "A Comparison of Flare Forecasting Methods. II — Benchmarks, Metrics, and Performance Results"
paper: "Leka, K. D.; Park, S.-H.; Kusano, K.; Andries, J.; Barnes, G.; et al. 2019, ApJS, 243, 36"
doi: "10.3847/1538-4365/ab2e12"
date: 2026-04-27
topic: Space_Weather
tags: [flare-forecasting, benchmark, skill-scores, TSS, BSS, ApSS, MSESS, operational, ensemble]
---

# A Comparison of Flare Forecasting Methods. II — Benchmarks, Metrics, and Performance Results
# 태양 플레어 예측 방법 비교 II — 벤치마크, 메트릭, 성능 결과

---

## 1. Core Contribution / 핵심 기여

**EN**: Leka et al. (2019, "Paper II") report the first community-wide, head-to-head, *operational* solar flare forecast comparison. Eighteen forecast variants from eleven institutions (NOAA/SWPC, MetOffice/MOSWOC, SIDC, Bureau of Meteorology, NICT, KMA/SELab, Trinity College Dublin, NJIT, NASA/MSFC, NWRA, ESA, University of Bradford) submitted full-disk daily probabilistic forecasts for the testing interval 2016-01-01 — 2017-12-31 (731 days) using two agreed-upon "exceedance" event definitions: C1.0+/0/24 (188 event-days) and M1.0+/0/24 (26 event-days). Each method was evaluated with a *suite* of metrics — Reliability/Attribute diagrams, ROC curves, Brier Skill Score (BSS), MSESS_clim (referenced to a 120-day prior climatology), Appleman Skill Scores (ApSS, ApSS_clim), True Skill Statistic / Hanssen-Kuiper / Peirce Skill Score (TSS), Equitable Threat Score (ETS), Heidke / Proportion Correct, Frequency Bias, ROC Skill Score / Gini coefficient — at a uniform probability threshold P_th = 0.5. The headline finding is that **many methods consistently exceed the no-skill reference but no single method is best across all metrics and event classes**. Rankings depend strongly on whether one is forecasting C1.0+ (relatively common, a "calibration" task) or M1.0+ (rare, a "discrimination" task), and on whether the metric rewards reliability (BSS, MSESS_clim), discrimination (Gini, TSS), or categorical accuracy (HSS, ApSS). The paper establishes the modern benchmark and an open dataset (Leka & Park 2019, Harvard Dataverse, doi:10.7910/DVN/HYP74O) that all subsequent ML/DL flare forecasters must measure themselves against.

**KO**: Leka et al. (2019, "Paper II")는 처음으로 커뮤니티 전반의 head-to-head 태양 플레어 *운영* 예측 비교를 보고했습니다. 11개 기관(NOAA/SWPC, MetOffice/MOSWOC, SIDC, 호주 기상청, NICT, KMA/SELab, Trinity College Dublin, NJIT, NASA/MSFC, NWRA, ESA, Bradford 대학)에서 18개 예측 변형이 2016-01-01 ~ 2017-12-31(731일) 테스트 구간에 대해 전면 일간 확률 예보를 제출했으며, 합의된 두 가지 "exceedance" 이벤트 정의(C1.0+/0/24: 188 event-days, M1.0+/0/24: 26 event-days)를 사용했습니다. 각 방법은 균일한 확률 임계값 P_th = 0.5에서 *세트 메트릭*(Reliability/Attribute 다이어그램, ROC 곡선, BSS, MSESS_clim(120일 사전 기후값 기준), ApSS, ApSS_clim, TSS/Hanssen-Kuiper/Peirce, ETS, Heidke/PC, Frequency Bias, ROC Skill Score/Gini)으로 평가되었습니다. 핵심 결과: **다수의 방법이 일관되게 no-skill 기준을 초과하지만 모든 메트릭과 이벤트 클래스에서 최고인 단일 방법은 없음**. 순위는 C1.0+(상대적으로 흔함, "보정" 과제) vs M1.0+(드묾, "판별" 과제) 예측 여부, 그리고 메트릭이 보정(BSS, MSESS_clim), 판별(Gini, TSS), 범주형 정확도(HSS, ApSS) 중 무엇을 보상하는지에 강하게 의존합니다. 본 논문은 현대적 벤치마크와 공개 데이터셋(Leka & Park 2019, Harvard Dataverse, doi:10.7910/DVN/HYP74O)을 확립하여 이후 모든 ML/DL 플레어 예보기가 측정 기준으로 삼아야 할 표준이 되었습니다.

---

## 2. Reading Notes / 읽기 노트

### 2.1 Introduction (pp. 2–3) / 서론

**EN**: Solar flares drive immediate space-weather impacts: sudden ionospheric disturbances that disrupt HF radio and radar (M/X-class events especially), and association with energetic particle events and CMEs. The motivation for systematic forecast evaluation comes from a 2009 SoHO/MDI-era workshop (which produced Paper I, Barnes et al. 2016). With SDO/HMI data after 2010, a new generation of forecasters emerged. In 2017 the Center for International Collaborative Research (CICR) at ISEE Nagoya University convened operational teams (government, private, academic) for a 3-day workshop. The criterion for participation was strict — methods must be running operationally, "providing a forecast on a routine, consistent basis using only data available prior to the issuance time." Forecaster-In-The-Loop (FITL, human-edited) was permitted. Daily forecasts were preferred but not required.

**KO**: 태양 플레어는 즉각적인 우주기상 영향을 일으킵니다: HF 무선과 레이더를 교란하는 급작스런 전리권 교란(M/X급 특히)과 고에너지 입자 이벤트, CME와의 연관성. 체계적인 예보 평가의 동기는 Paper I(Barnes et al. 2016)을 만든 2009년 SoHO/MDI 시대 워크숍에서 비롯되었습니다. 2010년 이후 SDO/HMI 데이터로 새로운 세대의 예보기가 등장했습니다. 2017년 ISEE 나고야대 CICR가 운영팀(정부, 민간, 학계)을 모아 3일 워크숍을 열었습니다. 참여 기준은 엄격: "발행 시간 이전에 사용 가능한 데이터만으로 정기적·일관적 기준으로 예보를 제공하는" 운영 중인 방법이어야 했습니다. FITL(인간이 편집한)이 허용되었고, 일간 예보가 선호되었으나 필수는 아니었습니다.

### 2.2 Comparison Methodology — Participating Methods (Section 2, Table 1, p. 5) / 비교 방법론 — 참여 방법

**EN**: Table 1 enumerates 18 method variants. Highlights:
- **A-EFFORT** (ESA/SSA, Georgoulis & Rust 2007) — Athens Effective Solar Flare Forecasting, helicity/connectivity-based.
- **AMOS** (KMA/Kyung Hee, Lee et al. 2012) — automatic McIntosh-occurrence-probability sunspot classification.
- **ASAP** (Bradford, Colak & Qahwaji 2008/2009) — automated solar activity prediction, ML on sunspot images.
- **ASSA** (KSWC/SELab, Hong et al. 2014, Lee et al. 2013) — Automatic Solar Synoptic Analyzer.
- **BOM (FlarecastII)** (Australian BoM, Steward et al. 2011/2017).
- **CLIM120** — 120-day prior climatology, included as a "no-skill" sanity check method.
- **DAFFS** & **DAFFS-G** (NWRA, Leka et al. 2018) — Discriminant Analysis Flare Forecasting, full HMI/GONG version and GONG+GOES-only stripped version.
- **MAG4 family** (NASA/MSFC, Falconer et al. 2011) — MAG4W (HMI), MAG4WF (HMI + flare history), MAG4VW (vector), MAG4VWF (vector + history).
- **MCSTAT** & **MCEVOL** (Trinity College Dublin, Gallagher et al. 2002, Bloomfield et al. 2012, McCloskey et al. 2018) — SolarMonitor.org FPS, statistical and evolutionary versions.
- **MOSWOC** (UK MetOffice, Murray et al. 2017) — human-edited forecasts.
- **NICT** (Japan, Kubo et al. 2017) — NICT-human (FITL).
- **NJIT** (Park et al. 2010) — magnetic helicity-based.
- **NOAA** (NOAA/SWPC, Crown 2012) — operational human forecast.
- **SIDC** (Royal Observatory of Belgium, Berghmans et al. 2005, Devos et al. 2014) — human-generated.

**KO**: 표 1은 18개 방법 변형을 나열합니다. 주요 항목:
- **A-EFFORT** (ESA/SSA): 헬리시티/연결성 기반.
- **AMOS** (KMA/경희대): 자동 McIntosh 발생확률 흑점 분류.
- **ASAP** (Bradford): 흑점 이미지 ML 자동 예측.
- **ASSA** (KSWC/SELab): 자동 태양 시놉틱 분석기.
- **BOM (FlarecastII)** (호주 기상청).
- **CLIM120**: 120일 사전 기후값, "no-skill" sanity check 방법으로 포함.
- **DAFFS, DAFFS-G** (NWRA): 판별 분석 기반 예보.
- **MAG4 계열** (NASA/MSFC): MAG4W/WF/VW/VWF 4종.
- **MCSTAT, MCEVOL** (Trinity Dublin): SolarMonitor.org FPS 통계/진화 버전.
- **MOSWOC** (영국 MetOffice): 인간 편집 예보.
- **NICT** (일본): NICT-human(FITL).
- **NJIT**: 자기 헬리시티 기반.
- **NOAA** (NOAA/SWPC): 운영 인간 예보.
- **SIDC** (벨기에 왕립천문대): 인간 생성.

### 2.3 Event Definitions and Testing Interval (Section 2.1, Table 2, p. 6) / 이벤트 정의와 테스트 구간

**EN**: The testing interval 2016-01-01 — 2017-12-31 was chosen to balance *training* and *test* needs for HMI-dependent methods (HMI near-real-time data is only available from late 2012 onward). Event thresholds follow NOAA/SWPC: "lower-limit + exceedance," based on GOES XRS 1–8 Å bands. C1.0+ corresponds to peak flux ≥ 1.0 × 10^{−6} W m^{−2}; M1.0+ to ≥ 1.0 × 10^{−5} W m^{−2}; X1.0+ to ≥ 1.0 × 10^{−4} W m^{−2}, all *with no upper limit* (exceedance). Background/pre-flare subtraction is *not* performed (consistent with operational practice). Validity period is 24 hr, latency is effectively 0 hr — only "one-day" forecasts are evaluated. Table 2 reports event counts:

| Class | # Quiet Days | # Event Days | Climatology rate |
|-------|-------------:|-------------:|-----------------:|
| C1.0+ | 543 | 188 | 0.257 |
| M1.0+ | 705 | 26 | 0.036 |
| X1.0+ | 728 | 3 | 0.004 |

X1.0+ events are too few for meaningful statistics, so only C1.0+ and M1.0+ are quantitatively evaluated. Most centers issue forecasts near 00 UT; SIDC issues at 12:30 UT and NICT at 06:00 UT, so custom event lists matched to issuance time were constructed (yielding 183/185 C1.0+ event-days for NICT/SIDC respectively, 27 M1.0+ for both).

**KO**: 테스트 구간 2016-01-01 ~ 2017-12-31은 HMI 의존 방법의 *훈련* 및 *테스트* 요구사항을 균형 맞추기 위해 선택되었습니다(HMI 근실시간 데이터는 2012년 말부터 가용). 이벤트 임계값은 NOAA/SWPC 정의를 따름: GOES XRS 1–8 Å 밴드 기반 "하한 + exceedance." C1.0+ ≥ 1.0 × 10^{−6} W m^{−2}, M1.0+ ≥ 1.0 × 10^{−5}, X1.0+ ≥ 1.0 × 10^{−4}, 모두 *상한 없음*(exceedance). 배경/플레어 전 차감은 수행하지 않음(운영 관행과 일치). 유효 기간 24시간, 지연 0시간 — "1일" 예보만 평가. 표 2 이벤트 수:

| 클래스 | 정온일 | 이벤트일 | 기후값 비율 |
|--------|-------:|---------:|------------:|
| C1.0+ | 543 | 188 | 0.257 |
| M1.0+ | 705 | 26 | 0.036 |
| X1.0+ | 728 | 3 | 0.004 |

X1.0+는 의미 있는 통계에 너무 적어 C1.0+와 M1.0+만 정량 평가. 대부분 센터는 00 UT 근처 발행; SIDC는 12:30 UT, NICT는 06:00 UT이므로 발행 시각에 맞춘 맞춤 이벤트 목록 구성(NICT/SIDC C1.0+ 각각 183/185 event-days, 양자 모두 M1.0+ 27 event-days).

### 2.4 Standard Metrics (Section 2.2, pp. 7–8) / 표준 메트릭

**EN**: The metric philosophy follows Jolliffe & Stephenson (2012). Two graphical tools and a battery of skill-scores are used:

1. **Reliability / Attribute Diagram**: Plots 20 bins of predicted probability vs observed event frequency. A perfect forecast lies on x = y. The diagram also shows the climatology rate (horizontal line) and the "no-skill" line (bisector between climatology and x = y). Bin populations are shown via small red squares. Reliability diagrams test *calibration*.
2. **ROC curve**: POD vs POFD as P_th varies. Perfect goes (0,0) → (0,1) → (1,1); the x = y line is no-skill. ROC tests *discrimination but not reliability*.

**Skill scores** computed at P_th = 0.5:
- **BSS** (Brier Skill Score): based on MSE of probabilities vs binary outcomes; reference = test-period climatology.
- **MSESS_clim**: same as BSS but reference = 120-day prior climatology (truly operational unskilled reference).
- **ROCSS / Gini coefficient** = 2·AUC − 1: ROC summary, normalized.
- **PSS / TSS / Hanssen-Kuiper** = POD − POFD: discrimination of dichotomous forecasts.
- **HSS** (Heidke Skill Score): reference = random forecast. Penalizes random guessing.
- **ApSS** (Appleman Skill Score): "across-the-board" climatology reference.
- **ApSS_clim**: ApSS with 120-day prior climatology reference.
- **ETS** (Equitable Threat Score): random reference; favored in meteorology for rare events.
- **PC** (Proportion Correct / Accuracy): can be misleadingly high for rare events.
- **FB** (Frequency Bias): systematic over-/under-forecasting, complement to TSS.

**KO**: 메트릭 철학은 Jolliffe & Stephenson (2012)을 따릅니다. 두 가지 그래픽 도구와 일련의 skill-score를 사용합니다:

1. **Reliability/Attribute Diagram**: 20개 bin의 예측 확률 vs 관측 이벤트 빈도. 완벽 예보는 x = y 위. 기후값 비율(수평선)과 "no-skill" 선(기후값과 x = y의 이등분선)도 표시. bin 모집단은 작은 빨간 사각형으로 표시. *보정* 검사.
2. **ROC 곡선**: P_th 변화에 따른 POD vs POFD. 완벽 = (0,0)→(0,1)→(1,1); x = y 선이 no-skill. *판별* 검사(보정은 아님).

**Skill score**(P_th = 0.5에서 계산):
- **BSS**: 확률 vs 이진 결과의 MSE 기반; 기준 = 테스트 기간 기후값.
- **MSESS_clim**: BSS와 동일하지만 기준 = 120일 사전 기후값(진정 운영적 unskilled 기준).
- **ROCSS / Gini** = 2·AUC − 1: ROC 요약, 정규화.
- **PSS / TSS / Hanssen-Kuiper** = POD − POFD: 이분법 예보의 판별력.
- **HSS**: 기준 = 무작위 예보.
- **ApSS**: "전체 기후값" 기준.
- **ApSS_clim**: 120일 사전 기후값 기준 ApSS.
- **ETS**: 무작위 기준; 희귀 이벤트에 기상학에서 선호.
- **PC**: 정확도; 희귀 이벤트에 오도 가능.
- **FB** (Frequency Bias): 체계적 과/과소예보, TSS 보완.

### 2.5 Why P_th = 0.5 and not TSS_max (p. 8–9) / 왜 P_th = 0.5이고 TSS_max가 아닌가

**EN**: A critical methodological choice. TSS_max (the maximum TSS over all P_th) is popular in the literature but problematic for two reasons:
1. It is determined *on the test set*, leaking test information back into the score — non-operational.
2. The optimal P_th depends on event rate, which varies with solar cycle; lessons from the testing interval do not transfer.
Therefore the authors **uniformly fix P_th = 0.5** for all dichotomous metrics. They acknowledge that this penalizes methods calibrated to a different operational threshold but argue it ensures a fair, reproducible, and operationally meaningful comparison. Methods retain the option to provide custom P_th from training; none did.

**KO**: 핵심적인 방법론적 선택. TSS_max(모든 P_th 위 최대 TSS)는 문헌에서 인기 있지만 두 가지 이유로 문제가 됩니다:
1. *테스트 세트에서* 결정되어 테스트 정보가 score에 누설됨 — 비-운영적.
2. 최적 P_th는 이벤트율에 의존하고 태양활동주기에 따라 변하므로, 테스트 구간의 교훈은 전이되지 않음.
따라서 저자들은 모든 이분법 메트릭에 대해 **P_th = 0.5로 일률 고정**. 이것이 다른 운영 임계값에 보정된 방법에 불리할 수 있음을 인정하지만 공정·재현·운영적으로 의미 있는 비교를 보장한다고 주장. 방법들은 훈련에서 도출된 맞춤 P_th를 제공할 옵션이 있었으나 제출자는 없었음.

### 2.6 Highlighted Metrics — No-skill Operational Reference (Section 2.3, p. 10) / 강조 메트릭 — no-skill 운영 기준

**EN**: For *operational* settings, the appropriate reference is the *best unskilled forecast available*, not a hypothetical perfect climatology. Following Sharpe & Murray (2017), the prior 120-day event rate is used as the no-skill reference (CLIM120). Figure 1 shows that CLIM120 varies from > 0.5 to < 0.5 for C1.0+ within the testing interval — the unskilled reference itself fluctuates with the solar cycle decline. Two metrics use this reference: **MSESS_clim** (analog of BSS but referenced to CLIM120) and **ApSS_clim** (Appleman with CLIM120 reference). CLIM120 is itself included as a "method" in all evaluations as a sanity check.

**KO**: *운영* 설정에서는 적절한 기준이 *가용한 최선의 unskilled 예보*이지 가상의 완벽한 기후값이 아닙니다. Sharpe & Murray (2017)을 따라 사전 120일 이벤트율을 no-skill 기준(CLIM120)으로 사용. 그림 1은 테스트 구간 내에서 CLIM120이 C1.0+에 대해 > 0.5에서 < 0.5로 변동함을 보여줌 — 기준선 자체가 태양활동주기 쇠퇴와 함께 변동. 두 메트릭이 이 기준을 사용: **MSESS_clim**(BSS의 아날로그이나 CLIM120 기준)과 **ApSS_clim**(CLIM120 기준 Appleman). CLIM120 자체가 모든 평가에 "방법"으로 포함되어 sanity check 역할.

### 2.7 Method Performances — Reliability (Section 3, Figure 2, p. 11) / 방법별 성능 — Reliability

**EN**: Figure 2 shows reliability diagrams for all methods, top panel for M1.0+/0/24, bottom for C1.0+/0/24. Key qualitative findings:
- Methods that are heavy probability forecasters (e.g., DAFFS, MCEVOL, MCSTAT, NJIT for C1.0+) generally lie close to or above the no-skill bisector across most bins — indicating reasonable calibration.
- Several human-edited services (NOAA, SIDC, MOSWOC) show clusters of forecasts at characteristic discrete probability levels (10%, 25%, 50%, 75%) — this is a known artifact of categorical human forecasting.
- For M1.0+ at high probability bins (≥ 0.5), most methods have very few sample points (small bin counts), so the observed frequency is noisy — this is a fundamental limitation of the rare-event regime.
- CLIM120 forms a near-vertical line in its own bin (since it issues nearly the same value every day) and serves as the reference baseline.

**KO**: 그림 2는 모든 방법의 reliability diagram을 보여줌 — 위 패널 M1.0+/0/24, 아래 C1.0+/0/24. 주요 정성 결과:
- 무거운 확률 예보기(C1.0+에 대한 DAFFS, MCEVOL, MCSTAT, NJIT)는 대부분의 bin에서 no-skill 이등분선 근처 또는 위 — 합리적 보정.
- 일부 인간 편집 서비스(NOAA, SIDC, MOSWOC)는 특징적인 이산 확률 수준(10%, 25%, 50%, 75%)에 예보가 군집 — 범주형 인간 예보의 알려진 아티팩트.
- M1.0+의 고확률 bin(≥ 0.5)에서 대부분의 방법은 표본이 매우 적어 관측 빈도가 잡음 — 희귀 이벤트 영역의 근본적 한계.
- CLIM120은 자체 bin에서 거의 수직선(거의 매일 동일 값을 발행)으로 기준 baseline 역할.

### 2.8 Method Performances — ROC (Figure 3, p. 12) / 방법별 성능 — ROC

**EN**: Figure 3 plots ROC curves. Most methods bow significantly above the diagonal x = y line, confirming non-trivial discrimination. M1.0+ ROC curves are generally smoother and more concave (better discrimination) than C1.0+ for many methods, because M1.0+ events are concentrated in a few highly active regions and are easier to flag than the more diffuse C1.0+ population. Methods such as ASSA, MCSTAT, MCEVOL, DAFFS show notably high AUC. CLIM120 is by construction near the diagonal. NJIT for C1.0+ shows a step-like ROC, suggesting limited probability resolution.

**KO**: 그림 3은 ROC 곡선을 표시. 대부분의 방법이 대각선 x = y보다 상당히 위로 휘어져 비자명한 판별력 확인. M1.0+ ROC 곡선이 일반적으로 더 매끄럽고 오목(더 나은 판별)하며 — M1.0+ 이벤트가 매우 활성화된 몇 개 영역에 집중되어 더 분산된 C1.0+ 집단보다 표시하기 쉬움. ASSA, MCSTAT, MCEVOL, DAFFS 등이 특히 높은 AUC. CLIM120은 구조상 대각선 근처. C1.0+에 대한 NJIT는 계단형 ROC — 제한된 확률 분해능 시사.

### 2.9 Skill-Score Tables (Section 3 cont., Figure 4 referenced) / Skill-score 테이블

**EN**: The full numerical metric tables (BSS, MSESS_clim, ApSS, ApSS_clim, TSS, HSS, ETS, PC, FB, Gini) by method × event class are presented in Figure 4 of the paper (visualized as ranking plots). Verbal summary from the paper:
- For **C1.0+/0/24**, top-tier methods by BSS/MSESS_clim include MCSTAT, MCEVOL, DAFFS, ASSA, AMOS — methods with sophisticated active-region characterization.
- For **M1.0+/0/24**, top-tier methods include DAFFS, MCSTAT, MOSWOC (human), MAG4 family — physics-motivated and human-edited services do well in the rare regime.
- **TSS** and **Gini** rankings differ from BSS rankings — discrimination-favored methods may be poorly calibrated and vice versa.
- **CLIM120** scores 0.0 on MSESS_clim and ApSS_clim by construction — confirming the metrics work.
- Methods that score below 0.0 on a skill score are *worse than no-skill* — a few configurations fall here for some metrics, illustrating that not all "operational" methods exceed the trivial reference for all event definitions.

**KO**: 모든 수치 메트릭 표(BSS, MSESS_clim, ApSS, ApSS_clim, TSS, HSS, ETS, PC, FB, Gini)는 방법 × 이벤트 클래스로 논문 그림 4에 순위 plot으로 제시. 정성 요약:
- **C1.0+/0/24**: BSS/MSESS_clim 상위 — MCSTAT, MCEVOL, DAFFS, ASSA, AMOS — 정교한 활성영역 특성화 방법.
- **M1.0+/0/24**: 상위 — DAFFS, MCSTAT, MOSWOC(인간), MAG4 계열 — 물리 기반과 인간 편집 서비스가 희귀 영역에서 우수.
- **TSS**와 **Gini** 순위는 BSS 순위와 다름 — 판별 우호적 방법이 보정 빈약할 수 있음, 그 반대도 성립.
- **CLIM120**은 구조상 MSESS_clim과 ApSS_clim에서 0.0 — 메트릭 작동 확인.
- skill score가 0.0 미만인 방법은 *no-skill보다 나쁨* — 일부 구성이 일부 메트릭에서 여기에 속함, 모든 "운영" 방법이 모든 이벤트 정의에 대해 자명한 기준을 초과하지는 않음을 보여줌.

### 2.10 Practical Implications / 실용적 함의

**EN**: The lack of a single dominant method has three practical consequences for operational space-weather centers:
1. **Ensembles are warranted**: Combining methods that are strong on different metrics or event classes is a clear path to higher overall skill (a topic for Paper III/IV and follow-on work).
2. **Match metric to use case**: Customers needing reliable probability forecasts should weight BSS/MSESS_clim heavy methods; customers needing yes/no alerts should weight TSS/HSS heavy methods.
3. **The "best" forecaster depends on the solar cycle phase**: As event rates change, calibration drifts; periodic re-evaluation is essential.

**KO**: 단일 우월 방법의 부재는 운영 우주기상 센터에 세 가지 실용적 결과를 가져옵니다:
1. **앙상블 필요**: 다른 메트릭 또는 이벤트 클래스에서 강한 방법들을 결합하는 것이 전반 skill 향상의 명확한 경로(Paper III/IV 및 후속 연구 주제).
2. **메트릭과 사용 사례 정합**: 신뢰할 수 있는 확률 예보가 필요한 고객은 BSS/MSESS_clim이 강한 방법에, yes/no 경보가 필요한 고객은 TSS/HSS가 강한 방법에 가중.
3. **"최고" 예보기는 태양활동주기 단계에 의존**: 이벤트율 변화에 따라 보정이 표류; 주기적 재평가 필수.

---

## 3. Key Takeaways / 핵심 시사점

1. **First common-test-set comparison / 첫 공통 테스트 세트 비교**
   - **EN**: Eighteen operational variants from eleven institutions evaluated on identical 731-day interval (2016–2017) with identical event definitions. This is the methodological foundation that subsequent ML papers (Bobra & Couvidat 2015 onward) cite as their benchmark target.
   - **KO**: 11개 기관의 18개 운영 변형을 동일한 731일 구간(2016–2017)과 동일한 이벤트 정의로 평가. 이후 ML 논문들(Bobra & Couvidat 2015 이후)이 벤치마크 표적으로 인용하는 방법론적 기반.

2. **Metric suite > single number / 단일 숫자보다 메트릭 세트**
   - **EN**: The paper deliberately reports BSS, MSESS_clim, ApSS, ApSS_clim, TSS, HSS, ETS, PC, FB, Gini — not a single "winner score." A method strong on calibration may be weak on discrimination.
   - **KO**: 단일 "우승 점수"가 아닌 BSS, MSESS_clim, ApSS, ApSS_clim, TSS, HSS, ETS, PC, FB, Gini를 의도적으로 보고. 보정에 강한 방법이 판별에 약할 수 있음.

3. **Operational reference must be 120-day prior climatology / 운영 기준은 120일 사전 기후값**
   - **EN**: A *true* operational baseline never uses test-period information. CLIM120 (event rate over the prior 120 days, computed daily) is the appropriate reference forecast — and varies dramatically with the solar cycle.
   - **KO**: *진정* 운영 기준선은 테스트 기간 정보를 절대 사용하지 않음. CLIM120(매일 계산되는 직전 120일 이벤트율)이 적절한 기준 예보 — 태양활동주기와 함께 극적으로 변동.

4. **No single winner / 단일 우승자 없음**
   - **EN**: Top-ranked methods change with event class (C1.0+ vs M1.0+) and metric. This *is* the result, not a failure of the comparison. It motivates ensemble approaches and metric-aware deployment.
   - **KO**: 상위 방법은 이벤트 클래스(C1.0+ vs M1.0+)와 메트릭에 따라 변화. 이것이 비교의 실패가 아니라 *결과*. 앙상블 접근과 메트릭 인식 배포의 동기.

5. **P_th = 0.5 is the operationally honest choice / P_th = 0.5가 운영적으로 정직한 선택**
   - **EN**: TSS_max optimizes a threshold using test-set knowledge. P_th = 0.5 (or any value chosen ahead of time without test information) is the only fair operational evaluation.
   - **KO**: TSS_max는 테스트 세트 지식으로 임계값을 최적화. 테스트 정보 없이 사전 선택된 P_th = 0.5(또는 어떤 값)만이 공정한 운영 평가.

6. **Class imbalance dominates rare-event evaluation / 클래스 불균형이 희귀 이벤트 평가 지배**
   - **EN**: With only 26 M1.0+ event-days out of 731, naive accuracy ≥ 96% for trivial "never flare" forecasts. PC alone is misleading; TSS, BSS, ApSS_clim are required.
   - **KO**: 731일 중 단 26 M1.0+ event-day로, 자명한 "절대 플레어 없음" 예보의 단순 정확도는 ≥ 96%. PC만으로는 오도; TSS, BSS, ApSS_clim 필수.

7. **Human (FITL) services remain competitive / 인간(FITL) 서비스는 여전히 경쟁력**
   - **EN**: NOAA, SIDC, MOSWOC, NICT-human achieve scores comparable to fully automated/ML methods, especially for M1.0+. Human pattern recognition + automated flagging is a viable hybrid.
   - **KO**: NOAA, SIDC, MOSWOC, NICT-human은 완전 자동/ML 방법과 비교 가능한 점수, 특히 M1.0+에서. 인간 패턴 인식 + 자동 플래깅이 실행 가능한 하이브리드.

8. **Open data enables reproducibility / 공개 데이터가 재현성 가능케 함**
   - **EN**: All probability forecasts are released at Harvard Dataverse (doi:10.7910/DVN/HYP74O). This is the default test set for new flare forecasters and enables ensemble construction by third parties.
   - **KO**: 모든 확률 예보가 Harvard Dataverse(doi:10.7910/DVN/HYP74O)에 공개. 새로운 플레어 예보기의 기본 테스트 세트이며 제3자에 의한 앙상블 구성 가능.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Contingency Table (2 × 2) / 분할표

```
                     Observed
                    Yes    No
Forecast  Yes |  TP    FP  |
          No  |  FN    TN  |
```
Total N = TP + FP + FN + TN. Event rate s = (TP + FN) / N.

### 4.2 Probability of Detection / Probability of False Detection

$$
\mathrm{POD} = \frac{TP}{TP + FN}, \qquad \mathrm{POFD} = \frac{FP}{FP + TN}
$$

**EN**: POD (also called recall or hit rate) = fraction of actual events correctly forecast. POFD (false alarm rate) = fraction of non-events erroneously forecast as events. ROC curves plot POD on y-axis vs POFD on x-axis as P_th sweeps from 0 → 1.

**KO**: POD(재현율, hit rate) = 정확히 예보된 실제 이벤트 비율. POFD(오경보율) = 이벤트로 잘못 예보된 비-이벤트 비율. ROC 곡선은 P_th를 0→1로 변경하며 y축 POD vs x축 POFD로 표시.

### 4.3 True Skill Statistic (Hanssen-Kuiper / Peirce)

$$
\mathrm{TSS} = \mathrm{POD} - \mathrm{POFD} = \frac{TP}{TP + FN} - \frac{FP}{FP + TN}
$$

**EN**: TSS ranges from −1 to +1. Perfect TSS = 1; pure no-skill (random or constant) = 0. TSS is invariant to event rate, making it suitable for rare-event evaluation. It corresponds to the *vertical distance* of an ROC point from the diagonal.

**KO**: TSS는 −1에서 +1 범위. 완벽 TSS = 1; 순수 no-skill(무작위 또는 상수) = 0. TSS는 이벤트율에 불변하여 희귀 이벤트 평가에 적합. ROC 점의 대각선으로부터의 *수직 거리*에 대응.

### 4.4 Heidke Skill Score (HSS)

$$
\mathrm{HSS} = \frac{2(TP \cdot TN - FP \cdot FN)}{(TP + FN)(FN + TN) + (TP + FP)(FP + TN)}
$$

**EN**: HSS measures improvement over a *random* reference forecast that preserves the event rate and forecast bias. Perfect = 1, random = 0.

**KO**: HSS는 이벤트율과 예보 bias를 보존하는 *무작위* 기준 예보에 대한 개선 측정. 완벽 = 1, 무작위 = 0.

### 4.5 Brier Score and BSS

$$
\mathrm{BS} = \frac{1}{N} \sum_{i=1}^{N} (p_i - o_i)^2, \qquad \mathrm{BSS} = 1 - \frac{\mathrm{BS}}{\mathrm{BS}_{\mathrm{ref}}}
$$

**EN**: p_i is the forecast probability for day i, o_i ∈ {0, 1} is the observed binary outcome. BS = 0 is perfect; BS = 1 is worst. The reference BS_ref is computed using the climatological event rate s as a constant forecast (BS_ref = s(1−s) for the testing-period climatology). BSS > 0 means improvement over climatology, BSS = 1 perfect, BSS < 0 worse than climatology.

**KO**: p_i는 i일의 예보 확률, o_i ∈ {0, 1}는 관측 이진 결과. BS = 0이 완벽; BS = 1이 최악. 기준 BS_ref는 기후값 이벤트율 s를 상수 예보로 사용하여 계산(테스트 기간 기후값 BS_ref = s(1−s)). BSS > 0이면 기후값 대비 개선, BSS = 1 완벽, BSS < 0 기후값보다 나쁨.

### 4.6 MSESS with prior 120-day climatology (MSESS_clim)

$$
\mathrm{MSESS}_{\mathrm{clim}} = 1 - \frac{\sum_{i=1}^{N}(p_i - o_i)^2}{\sum_{i=1}^{N}(c_i^{120} - o_i)^2}
$$

**EN**: c_i^{120} is the prior-120-day event rate computed for day i (varies daily). This is the operationally honest unskilled reference; CLIM120 score on this metric is 0 by construction.

**KO**: c_i^{120}은 i일에 대해 계산된 직전 120일 이벤트율(매일 변동). 이것이 운영적으로 정직한 unskilled 기준; CLIM120의 이 메트릭 점수는 구조상 0.

### 4.7 Appleman Skill Score (ApSS)

$$
\mathrm{ApSS} = 1 - \frac{N_{\mathrm{wrong}}}{N_{\mathrm{wrong, \, ref}}}
$$

**EN**: N_wrong = number of wrong dichotomous forecasts (FP + FN). The reference forecast is the "across-the-board" climatology forecast — predict event every day if s > 0.5, else predict no-event every day. ApSS = 1 perfect, 0 no-skill, < 0 worse than the trivial across-the-board forecast.

**KO**: N_wrong = 잘못된 이분 예보 수(FP + FN). 기준 예보는 "전반적" 기후값 예보 — s > 0.5면 매일 이벤트, 그렇지 않으면 매일 비-이벤트. ApSS = 1 완벽, 0 no-skill, < 0 자명한 전반적 예보보다 나쁨.

### 4.8 ROC Skill Score (ROCSS) / Gini coefficient

$$
\mathrm{ROCSS} = 2 \cdot \mathrm{AUC} - 1
$$

**EN**: AUC = area under the ROC curve. AUC = 0.5 for no-skill, 1.0 for perfect; ROCSS rescales to [−1, 1] with no-skill at 0. Equivalent to the Gini coefficient. Probabilistic interpretation: AUC = P(p_event > p_non-event) for a randomly chosen event/non-event pair.

**KO**: AUC = ROC 곡선 아래 면적. AUC = 0.5 no-skill, 1.0 완벽; ROCSS는 [−1, 1]로 재스케일, no-skill이 0. Gini 계수와 동등. 확률적 해석: AUC = 무작위 이벤트/비-이벤트 쌍에서 P(p_event > p_non-event).

### 4.9 Frequency Bias

$$
\mathrm{FB} = \frac{TP + FP}{TP + FN}
$$

**EN**: FB = 1 means the forecast issues "yes" the same number of times as events occur. FB > 1 over-forecasting, FB < 1 under-forecasting. Complements TSS: a method may have high TSS but bad FB.

**KO**: FB = 1이면 예보 "yes" 발행 횟수가 이벤트 발생 횟수와 같음. FB > 1 과예보, FB < 1 과소예보. TSS 보완: 높은 TSS에도 나쁜 FB일 수 있음.

### 4.10 Worked Example / 실제 예시 — M1.0+/0/24 hypothetical method

Suppose a method on the 731-day testing interval produces (P_th = 0.5):
- TP = 12, FN = 14, FP = 35, TN = 670
- Event rate s = (12 + 14) / 731 = 0.0356

Calculations:
- POD = 12 / 26 = 0.462
- POFD = 35 / 705 = 0.0496
- TSS = 0.462 − 0.0496 = **0.412**
- HSS numerator = 2 × (12 × 670 − 35 × 14) = 2 × (8040 − 490) = 15100
- HSS denom = (26)(684) + (47)(705) = 17784 + 33135 = 50919
- HSS = 15100 / 50919 = **0.297**
- PC = (12 + 670) / 731 = 0.933 — looks "good" but FB = 47/26 = 1.81 (heavy over-forecasting)
- A trivial "always no-event" forecast would score PC = 705/731 = 0.964 — *higher* than this method! Hence skill scores beat raw accuracy.

### 4.11 Probability Threshold P_th sweep / 확률 임계값 P_th 스위프

**EN**: For each P_th ∈ {0.0, 0.05, 0.10, …, 0.95, 1.0}, classify day i as "yes" if p_i ≥ P_th, "no" otherwise. Recompute (TP, FP, FN, TN) and hence (POD, POFD). The set of (POFD, POD) points traces the ROC curve. AUC is computed by the trapezoidal rule (np.trapezoid).

**KO**: 각 P_th ∈ {0.0, 0.05, 0.10, …, 0.95, 1.0}에 대해 p_i ≥ P_th면 i일을 "yes", 그렇지 않으면 "no"로 분류. (TP, FP, FN, TN) 따라서 (POD, POFD) 재계산. (POFD, POD) 점들의 집합이 ROC 곡선을 추적. AUC는 사다리꼴 법칙(np.trapezoid)으로 계산.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
Timeline of Solar Flare Forecasting Evaluation
─────────────────────────────────────────────────────────────────
1986  ─ Sawyer, Warwick, Dennett: McIntosh-classification climatology
1996  ─ Murphy: "What is a good forecast?" — verification primer
2002  ─ Gallagher et al.: SolarMonitor.org FPS launched
2007  ─ Schrijver: nonpotentiality predictors
2009  ─ NWRA flare-forecasting workshop (MDI era)
2011  ─ Falconer et al.: MAG4 free-energy proxy
2012  ─ Bloomfield et al.: TSS/HSS for flare prediction made standard
2015  ─ Bobra & Couvidat: SVM on SHARP features (modern ML era)
2016  ─ Barnes et al. (Paper I): MDI-era method comparison
2017  ─ ISEE Nagoya CICR workshop — operational comparison launched
2017  ─ Sharpe & Murray: 120-day prior climatology reference
2017  ─ Murray et al.: MOSWOC verification framework
2017  ─ Nishizuka et al.: deep learning flare forecasting
═════ 2019  ─ Leka, Park, Kusano, Andries+ "Paper II" (THIS PAPER)
        First operational community-wide head-to-head
        Public benchmark dataset on Harvard Dataverse
2019  ─ Leka et al. (Paper III): implementation-detail analysis
2019  ─ Park et al. (Paper IV): temporal pattern errors
2020+ ─ Era of "compare against Leka+2019" for new ML methods
─────────────────────────────────────────────────────────────────
```

**EN**: Paper II sits at the inflection point between the SDO/HMI-era method proliferation (2010–2018) and the deep-learning era (2018+). It establishes the canonical evaluation harness — common testing interval, common event definitions, common metric suite, common no-skill reference — that all subsequent flare forecasters use.

**KO**: Paper II는 SDO/HMI 시대 방법 확산기(2010–2018)와 딥러닝 시대(2018+) 사이의 변곡점에 위치. 이후 모든 플레어 예보기가 사용하는 표준 평가 하네스 — 공통 테스트 구간, 공통 이벤트 정의, 공통 메트릭 세트, 공통 no-skill 기준 — 를 확립.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper | Year | Connection / 연결 |
|-------|------|-------------------|
| Barnes et al. (Paper I) | 2016 | Methodology origin (MDI workshop) / 방법론 기원 (MDI 워크숍) |
| **Leka, Park, Kusano, Andries+ (Paper II)** | **2019** | **THIS PAPER — operational benchmarks / 본 논문 — 운영 벤치마크** |
| Leka et al. (Paper III) | 2019 | Why methods differ — implementation analysis / 방법 차이 이유 — 구현 분석 |
| Park et al. (Paper IV) | 2019 | Temporal error patterns / 시간 오차 패턴 |
| Jolliffe & Stephenson | 2012 | Verification metric textbook / 검증 메트릭 교과서 |
| Bloomfield et al. | 2012 | TSS/HSS adoption for flares / TSS/HSS 채택 |
| Sharpe & Murray | 2017 | 120-day climatology reference / 120일 기후값 기준 |
| Murray et al. | 2017 | MOSWOC verification / MOSWOC 검증 |
| Bobra & Couvidat | 2015 | First modern ML flare forecast / 첫 현대 ML 플레어 예측 |
| Nishizuka et al. | 2017 | Deep learning flare forecast / 딥러닝 플레어 예측 |
| Florios et al. | 2018 | ML benchmarking on SHARP / SHARP에서 ML 벤치마킹 |
| Murphy | 1996 | "Good forecast" foundational essay / "좋은 예보" 기초 에세이 |
| Falconer et al. | 2011 | MAG4 method origin / MAG4 방법 기원 |
| Gallagher et al. | 2002 | SolarMonitor.org FPS origin / SolarMonitor.org FPS 기원 |
| Georgoulis & Rust | 2007 | A-EFFORT origin / A-EFFORT 기원 |

---

## 7. References / 참고문헌

- Leka, K. D.; Park, S.-H.; Kusano, K.; Andries, J.; Barnes, G.; Bingham, S.; Bloomfield, D. S.; et al. "A Comparison of Flare Forecasting Methods. II. Benchmarks, Metrics, and Performance Results for Operational Solar Flare Forecasting Systems," *ApJS*, 243, 36, 2019. [DOI: 10.3847/1538-4365/ab2e12]
- Barnes, G.; Leka, K. D.; Schrijver, C. J.; et al. "A Comparison of Flare Forecasting Methods. I. Results from the All-Clear Workshop," *ApJ*, 829, 89, 2016.
- Leka, K. D.; Park, S.-H.; Kusano, K.; et al. "A Comparison of Flare Forecasting Methods. III. Systematic Behaviors of Operational Solar Flare Forecasting Systems," *ApJ*, 881, 101, 2019.
- Park, S.-H.; Leka, K. D.; Kusano, K.; et al. "A Comparison of Flare Forecasting Methods. IV. Evaluating Consecutive-day Forecasting Patterns," *ApJ*, 890, 124, 2020.
- Jolliffe, I. T.; Stephenson, D. B. *Forecast Verification: A Practitioner's Guide in Atmospheric Science*, 2nd ed., Wiley, 2012.
- Bloomfield, D. S.; Higgins, P. A.; McAteer, R. T. J.; Gallagher, P. T. "Toward Reliable Benchmarking of Solar Flare Forecasting Methods," *ApJL*, 747, L41, 2012.
- Sharpe, M. A.; Murray, S. A. "On the Verification of Operational Space Weather Forecasts: The Importance of Test Periods," *Space Weather*, 15, 1383, 2017.
- Murray, S. A.; Bingham, S.; Sharpe, M.; Jackson, D. R. "Flare Forecasting at the Met Office Space Weather Operations Centre," *Space Weather*, 15, 577, 2017.
- Murphy, A. H. "The Finley Affair: A Signal Event in the History of Forecast Verification," *Wea. Forecasting*, 11, 3, 1996.
- Bobra, M. G.; Couvidat, S. "Solar Flare Prediction Using SDO/HMI Vector Magnetic Field Data with a Machine-Learning Algorithm," *ApJ*, 798, 135, 2015.
- Nishizuka, N.; Sugiura, K.; Kubo, Y.; et al. "Solar Flare Prediction Model with Three Machine-Learning Algorithms Using Ultraviolet Brightening and Vector Magnetograms," *ApJ*, 835, 156, 2017.
- Florios, K.; Kontogiannis, I.; Park, S.-H.; et al. "Forecasting Solar Flares Using Magnetogram-based Predictors and Machine Learning," *Solar Phys.*, 293, 28, 2018.
- Falconer, D.; Barghouty, A. F.; Khazanov, I.; Moore, R. "A Tool for Empirical Forecasting of Major Flares, CMEs, and Solar Particle Events from a Proxy of Active-Region Free Magnetic Energy," *Space Weather*, 9, S04003, 2011.
- Open dataset: Leka, K. D.; Park, S.-H. "A Comparison of Flare Forecasting Methods II: Data and Supporting Code," Harvard Dataverse, 2019. [DOI: 10.7910/DVN/HYP74O]
