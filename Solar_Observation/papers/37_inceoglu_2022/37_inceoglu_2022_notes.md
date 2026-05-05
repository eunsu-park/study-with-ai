---
title: "Identification of Coronal Holes on AIA/SDO Images Using Unsupervised Machine Learning"
authors: ["Fadil Inceoglu", "Yuri Y. Shprits", "Stephan G. Heinemann", "Stefano Bianco"]
year: 2022
journal: "The Astrophysical Journal, 930:118 (11pp)"
doi: "10.3847/1538-4357/ac5f43"
topic: Solar_Observation
tags: [coronal_holes, AIA, SDO, k-means, unsupervised_ML, image_segmentation, space_weather, EUV]
status: completed
date_started: 2026-04-20
date_completed: 2026-04-20
---

# 37. Identification of Coronal Holes on AIA/SDO Images Using Unsupervised Machine Learning / AIA/SDO 영상에서 비지도 머신러닝을 이용한 코로나홀 식별

---

## 1. Core Contribution / 핵심 기여

### English

This paper presents a **pixel-wise k-means clustering pipeline** for automatically identifying coronal holes (CHs) in EUV images from the Atmospheric Imaging Assembly (AIA) on the Solar Dynamics Observatory (SDO). The authors process 237 dates spanning 2010 November through 2016 December (selected to cover solar cycle 24 and to overlap with the trustworthy CATCH binary maps) using the 171, 193, and 211 Å passbands in five different input configurations: single-channel 193 Å, single-channel 211 Å, two-channel composite 2CC (193+211), three-channel composite 3CC (171+193+211), and the two-channel overlap 2CO (intersection of 193 and 211 binary masks). A carefully designed preprocessing pipeline — limb-brightening correction, PSF deconvolution, log-normal transformation, and bimodal-Gaussian threshold determination at $\mu \pm 4\sigma$ — is followed by k-means with $k=3$ (chosen via a scree plot) and morphological cleanup (opening with min size 200 px, closing with disk radius 2 px). The headline result is that the 2CC and 193 Å configurations achieve median $\text{IoU} = 0.64 \pm 0.14$ and $0.62 \pm 0.14$ respectively, and median $\text{TSS} = 0.93 \pm 0.06$ and $0.91 \pm 0.06$ against CATCH ground truth — **better than HEK/SPoCA's** $\text{IoU} = 0.53 \pm 0.13$ and **competitive with the CHRONNOS CNN** (mean IoU = 0.63, TSS = 0.81; Jarolim et al. 2021), despite using a far simpler algorithm and only 3 channels (vs. CHRONNOS's 7 channels + magnetograms). The paper closes with a strong call for the community to build an **observer-consensus CH "ground truth" database**, since the lack of agreement on CH boundaries fundamentally limits supervised methods.

### 한국어

본 논문은 SDO 위성의 AIA(Atmospheric Imaging Assembly) 가 촬영한 EUV 영상에서 코로나홀(CH)을 자동 식별하기 위한 **픽셀 단위 k-means 군집화 파이프라인**을 제시한다. 저자들은 2010년 11월부터 2016년 12월까지의 237개 날짜(태양주기 24를 커버하면서 신뢰성 있는 CATCH 이진 마스크와 겹치도록 선택됨)를 대상으로, 171·193·211 Å 통과대역을 5가지 입력 구성으로 처리한다: 단일 채널 193 Å, 단일 채널 211 Å, 2채널 합성 2CC(193+211), 3채널 합성 3CC(171+193+211), 그리고 두 채널 중첩 2CO(193·211 이진 마스크의 교집합). 정교하게 설계된 전처리 파이프라인 — 가장자리 밝기 보정, PSF 디컨볼루션, 로그-정규 변환, $\mu \pm 4\sigma$ 임계값을 정하는 양봉 가우시안 피팅 — 다음에 scree-plot으로 결정된 $k=3$의 k-means가 수행되고, 형태학적 정리(opening: min size 200 px; closing: disk radius 2 px)가 마무리한다. 핵심 결과: 2CC와 193 Å 구성이 CATCH 기준선에 대해 각각 중앙값 $\text{IoU} = 0.64 \pm 0.14$, $0.62 \pm 0.14$ 그리고 $\text{TSS} = 0.93 \pm 0.06$, $0.91 \pm 0.06$을 달성하며, 이는 **HEK/SPoCA의 $\text{IoU} = 0.53 \pm 0.13$보다 우수**하고, **7개 채널 + 자기도를 사용하는 CHRONNOS CNN(평균 IoU=0.63, TSS=0.81; Jarolim et al. 2021)과 견줄만**하다 — 단 3개 채널과 훨씬 단순한 알고리즘으로. 논문은 CH 경계에 대한 합의 부재가 지도학습 방법을 근본적으로 제한한다고 지적하며, **관측자 합의(observer-consensus) CH "ground truth" 데이터베이스** 구축의 필요성을 강하게 호소하며 마무리한다.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction & Motivation / 서론 및 동기 (§1)

**English**

CHs are dark, low-density, low-temperature regions in the corona where magnetic field lines are predominantly **open**, allowing plasma to escape into the heliosphere as the **steady fast solar wind** (~700 km/s). They appear as dark patches in EUV/X-ray imagery (typically observed near 193–195 Å) because their lower density and temperature reduce continuum emission. The temporal evolution of CHs follows the solar cycle: during solar minimum, large polar CHs dominate; during the inclining phase, CHs appear at any latitude and are short-lived; during the maximum, CHs become small and confined to mid-latitudes; during the declining phase, long-lived mid-latitude CHs form near the equator (Hewins et al. 2020).

The paper reviews seven major CH detection methods:
1. **Manual identification** — He I 10830 Å near-infrared observations (Harvey & Recely 2002)
2. **Histogram-based intensity thresholding** — CHARM at 193, 195 Å (Krista & Gallagher 2009)
3. **Multithermal intensity segmentation** — CHIMERA using 171, 193, 211 Å (Garton et al. 2018)
4. **Intensity gradient with semi-automation** — CATCH (Heinemann et al. 2019)
5. **Spatial possibilistic clustering** — SPoCA, posted to HEK (Verbeeck et al. 2014; Hurlburt et al. 2012)
6. **U-Net CNN segmentation** — Illarionov & Tlatov (2018), trained on SPoCA-CH binary maps
7. **Progressively-grown CNN** — CHRONNOS using 7 AIA channels + HMI magnetograms (Jarolim et al. 2021)

The paper situates itself between these approaches: aiming for the simplicity and interpretability of the threshold-based methods while leveraging the data-driven flexibility that ML provides.

**한국어**

CH는 코로나의 어둡고 저밀도·저온의 영역으로, 자기력선이 주로 **열려 있어** 플라즈마가 태양권으로 빠져나가 **정상 고속 태양풍(~700 km/s)** 을 형성한다. EUV/X-선 영상(주로 193–195 Å)에서 어두운 패치로 보이는 이유는 낮은 밀도와 온도로 연속체 방출이 줄기 때문이다. CH의 시간 변동은 태양주기를 따른다: 극소기에는 큰 극관(polar) CH가 우세하고, 상승기에는 모든 위도에 단명한 CH가 나타나며, 극대기에는 작고 중위도에 국한되며, 하강기에는 적도 근처에 장수하는 중위도 CH가 형성된다(Hewins et al. 2020).

논문은 일곱 가지 주요 CH 탐지 방법을 검토한다:
1. **수동 식별** — He I 10830 Å 근적외선 관측 (Harvey & Recely 2002)
2. **히스토그램 기반 강도 임계** — 193, 195 Å에서의 CHARM (Krista & Gallagher 2009)
3. **다중 온도 강도 분할** — 171, 193, 211 Å를 사용하는 CHIMERA (Garton et al. 2018)
4. **강도 기울기 + 반자동** — CATCH (Heinemann et al. 2019)
5. **공간 가능론적 군집화** — SPoCA, HEK에 게시 (Verbeeck et al. 2014; Hurlburt et al. 2012)
6. **U-Net CNN 분할** — Illarionov & Tlatov (2018), SPoCA-CH 이진 마스크로 학습
7. **점진적 성장 CNN** — 7개 AIA 채널 + HMI 자기도를 사용하는 CHRONNOS (Jarolim et al. 2021)

본 논문은 이들 사이에 자리잡는다: 임계값 기반 방법의 단순성·해석가능성을 추구하면서, ML이 제공하는 데이터 기반 유연성을 활용한다.

### Part II: Data / 데이터 (§2)

**English**

- **Instrument**: AIA on SDO; full-disk 4096×4096 pixel images every 12 s, 0.6″/pixel (Lemen et al. 2012)
- **Wavelengths**: 171 Å (Fe IX, ~6×10⁵ K, upper transition region), 193 Å (Fe XII, XXIV, ~1.5×10⁶ K, quiet corona + flare plasma), 211 Å (Fe XIV, ~2×10⁶ K, active-region corona)
- **Exposure**: 2 s passband data
- **Date selection**: 237 days from 2010 November through 2016 December — specifically the **last two months of each year**. This window was chosen because (a) it overlaps with the CATCH binary maps (which only exist for that period), (b) CATCH is reliable in this period with minimal uncertainties, and (c) it covers solar cycle 24
- **Heliographic restriction**: CATCH maps are restricted to longitudinal range $[-400, 400]$ arcsec, where CHs can be identified more robustly (Jarolim et al. 2021); HEK polygons were converted to binary maps with the same restriction
- **Comparison data**: HEK CH polygons (SPoCA results) and CATCH binary maps for the same dates, both converted to a common pixel grid

**한국어**

- **기기**: SDO에 탑재된 AIA; 12초마다 풀-디스크 4096×4096 픽셀 영상, 0.6″/픽셀 (Lemen et al. 2012)
- **파장대**: 171 Å (Fe IX, ~6×10⁵ K, 상부 전이영역), 193 Å (Fe XII, XXIV, ~1.5×10⁶ K, 정온 코로나 + 플레어 플라즈마), 211 Å (Fe XIV, ~2×10⁶ K, 활동영역 코로나)
- **노출**: 2초 통과대역 데이터
- **날짜 선정**: 2010년 11월부터 2016년 12월까지 237일 — 특히 **각 해의 마지막 두 달**. 이 윈도우는 (a) CATCH 이진 마스크와 중첩되고(해당 기간에만 존재), (b) 해당 기간 CATCH는 불확실성이 작아 신뢰할 만하며, (c) 태양주기 24를 커버하기 때문에 선택됨
- **태양 좌표 제한**: CATCH 마스크는 경도 범위 $[-400, 400]$ arcsec로 제한 — 이 영역에서 CH가 더 견고하게 식별 가능 (Jarolim et al. 2021); HEK 다각형도 동일 제한으로 이진 마스크 변환
- **비교 데이터**: 동일 날짜의 HEK CH 다각형(SPoCA 결과)과 CATCH 이진 마스크, 모두 공통 픽셀 그리드로 변환

### Part III: Preprocessing Pipeline / 전처리 파이프라인 (§3.1)

**English**

The preprocessing pipeline is the unsung hero of the paper. It transforms raw AIA Level-1 data into a numerically well-conditioned input for k-means:

1. **Level-1 import** with `aiapy` (Barnes et al. 2020a, b) and `SunPy` (The SunPy Community 2020; Mumford et al. 2021)
2. **Instrument degradation correction** — accounts for sensor-aging and contamination
3. **Pointing/observer correction** — corrects for SDO's small pointing errors and orbital motion
4. **Registration & alignment** — co-aligns 171, 193, 211 channels to a common grid
5. **Normalization** to counts/pixel/second
6. **Limb-brightening correction** (annulus method; Verbeeck et al. 2014) — removes the apparent edge brightening caused by the longer line-of-sight through the optically-thin corona near the limb
7. **PSF deconvolution** — removes the instrument's point-spread-function
8. **Rescale** from 4096² to 1024² using spline interpolation (computational efficiency)
9. **Log-normal transformation** — applies $\log_{10}$ to compress the dynamic range and convert the highly skewed intensity distribution into something closer to bimodal Gaussian
10. **Bimodal Gaussian fitting** of the log-intensity histogram (Figure 2). When bimodal fitting fails on a date, a unimodal fit is used; the two-peak case is the norm, with the **higher peak representing CH (dark) pixels** (Heinemann et al. 2019). From the fit, the mean $\mu$ and standard deviation $\sigma$ of the higher-intensity peak are obtained.
11. **Threshold determination**:
    - Lower threshold: $T_{\text{low}} = \mu - 4\sigma$
    - Upper threshold: $T_{\text{up}} = \mu + 4\sigma$
12. **Threshold clipping ("stacking")** — pixels below $T_{\text{low}}$ are clipped to $T_{\text{low}}$, pixels above $T_{\text{up}}$ are clipped to $T_{\text{up}}$. This **increases the contrast** in the remaining range so that k-means cleanly separates CH/QS/AR.

A subtle but important detail: on 27 dates, the calculated $T_{\text{low}}$ for 211 Å is **negative** (the lower-Gaussian standard deviation extends below zero counts because of the underlying log-normal shape). Negative thresholds have no physical meaning, so the threshold is set to **zero** on those dates (Figure 3b).

**한국어**

전처리 파이프라인은 본 논문의 숨은 영웅이다. raw AIA Level-1 데이터를 k-means에 적합한 수치적으로 잘 조건화된 입력으로 변환한다:

1. **Level-1 임포트**: `aiapy` (Barnes et al. 2020a, b), `SunPy` (The SunPy Community 2020; Mumford et al. 2021) 사용
2. **기기 열화 보정** — 센서 노화와 오염에 대한 보정
3. **포인팅/관측자 보정** — SDO의 작은 포인팅 오차 및 궤도 운동 보정
4. **정렬 및 등록** — 171, 193, 211 채널을 공통 그리드로 정렬
5. **정규화**: counts/pixel/second 단위로
6. **가장자리 밝기 보정** (annulus 방법; Verbeeck et al. 2014) — 광학적으로 얇은 코로나에서 가장자리 근처의 LOS가 길어져 발생하는 명목 밝기 증가 제거
7. **PSF 디컨볼루션** — 기기의 PSF 영향 제거
8. **재스케일**: 스플라인 보간으로 4096² → 1024² (계산 효율성)
9. **로그-정규 변환**: $\log_{10}$ 적용으로 동적 범위 압축, 매우 비대칭한 강도 분포를 양봉 가우시안에 가까운 형태로 변환
10. **양봉 가우시안 피팅** of log-intensity 히스토그램 (Figure 2). 양봉 피팅이 실패하는 날짜는 단봉 피팅 사용. 양봉이 표준이며, **더 높은 봉이 CH(어두운) 픽셀** (Heinemann et al. 2019). 피팅에서 더 높은 강도 봉의 평균 $\mu$와 표준편차 $\sigma$ 추출
11. **임계값 결정**:
    - 하한: $T_{\text{low}} = \mu - 4\sigma$
    - 상한: $T_{\text{up}} = \mu + 4\sigma$
12. **임계 클리핑("쌓기")** — $T_{\text{low}}$ 미만 픽셀은 $T_{\text{low}}$로, $T_{\text{up}}$ 초과 픽셀은 $T_{\text{up}}$로 클리핑. 이를 통해 남는 범위의 **대비가 증가**하여 k-means가 CH/QS/AR을 깔끔하게 분리

미묘하지만 중요한 세부사항: 27일에 대해 211 Å의 계산된 $T_{\text{low}}$가 **음수**가 된다(저측 가우시안의 표준편차가 로그-정규 분포의 모양 때문에 0 카운트 아래로 확장). 음수 임계값은 물리적 의미가 없으므로 해당 날짜에는 **0**으로 설정 (Figure 3b).

### Part IV: k-means Clustering & Postprocessing / k-means 군집화 및 후처리 (§3.2)

**English**

After preprocessing, four input data sets are constructed:
- (i) 193 Å image alone (1D pixel vectors)
- (ii) 211 Å image alone (1D pixel vectors)
- (iii) 2CC = 193 + 211 (2D pixel vectors $(I_{193}, I_{211})$)
- (iv) 3CC = 171 + 193 + 211 (3D pixel vectors $(I_{171}, I_{193}, I_{211})$)

For each, **pixel-wise k-means** is run by minimizing the within-cluster sum of squared distances (SSD):

$$\text{SSD} = \sum_{i=1}^{k} \sum_{\mathbf{x} \in C_i} \|\mathbf{x} - \boldsymbol{\mu}_i\|^2$$

Lloyd-style iteration (MacQueen 1967; Likas et al. 2003): random init → assign each pixel to nearest centroid (Euclidean) → update centroid to cluster mean → repeat until SSD plateaus. Each pixel is one independent data point — **no spatial information enters the clustering itself** (spatial coherence is restored later via morphological operations).

**Choosing $k$ via scree plot** (Paparrizos & Gravano 2015): SSD is computed for $k = 1, 2, \ldots, 10$ on the 193 Å image of 2017 December 8. The curve drops steeply from $k=1$ to $k=3$, then flattens — an elbow at $k=3$. Hence the segmentation produces three clusters interpreted as:
- **Dark cluster** → CHs (lowest intensity)
- **Quiet Sun (QS)** → middle intensity
- **Bright cluster** → active regions (ARs, highest intensity)

The reason $k=2$ is rejected is that with only two clusters, ARs and QS would merge, leading to over-estimation of CH-darkness contrast (since the bright "AR pulls" would be lost), which would in turn over-estimate CH areas. With $k=3$, ARs are isolated as their own cluster.

**Binary CH map construction**: merge dark + bright clusters → label as non-CH (this leaves only the CH cluster as 1; everything else 0). Then apply morphological cleanup using `scikit-image` (van der Walt et al. 2014):
- **Opening** (erosion → dilation) with min object size **200 pixels** and connectivity **10 pixels** — removes small dotted-like spurious detections
- **Closing** (dilation → erosion) with disk-shaped footprint **radius 2 pixels** — fills small holes inside identified CHs

The closing footprint is intentionally small (radius 2) to avoid filling larger genuine bright points inside CHs (e.g., **coronal bright points**, Karachik et al. 2006; Hong et al. 2014; Wyper et al. 2018) which are real physical structures that should remain as holes in the binary mask.

**The 2CO (two-channel overlap) variant**: rather than running k-means on a 2D vector input, 2CO is computed as the **pixel-wise intersection** of the binary maps from 193 Å and 211 Å. A pixel is CH only if both single-channel methods classified it as CH — a conservative approach that suppresses false positives.

**한국어**

전처리 후 4가지 입력 데이터셋이 구성된다:
- (i) 193 Å 단일 (1D 픽셀 벡터)
- (ii) 211 Å 단일 (1D 픽셀 벡터)
- (iii) 2CC = 193 + 211 (2D 픽셀 벡터 $(I_{193}, I_{211})$)
- (iv) 3CC = 171 + 193 + 211 (3D 픽셀 벡터 $(I_{171}, I_{193}, I_{211})$)

각각에 대해 **픽셀 단위 k-means**를 수행하여 군집내 제곱거리합(SSD)을 최소화:

$$\text{SSD} = \sum_{i=1}^{k} \sum_{\mathbf{x} \in C_i} \|\mathbf{x} - \boldsymbol{\mu}_i\|^2$$

Lloyd 반복(MacQueen 1967; Likas et al. 2003): 무작위 초기화 → 각 픽셀을 가장 가까운 centroid에 할당(유클리드 거리) → centroid를 군집 평균으로 갱신 → SSD가 평탄해질 때까지 반복. 각 픽셀이 하나의 독립적 데이터 포인트 — **공간 정보는 클러스터링 자체에 들어가지 않음**(공간적 일관성은 후처리 형태학적 연산으로 복원).

**$k$ 선택은 scree plot으로** (Paparrizos & Gravano 2015): 2017년 12월 8일의 193 Å 영상에 대해 $k = 1, 2, \ldots, 10$의 SSD 계산. 곡선이 $k=1$에서 $k=3$까지 가파르게 떨어지고 그 이후 평탄 — $k=3$에서 팔꿈치(elbow). 따라서 분할은 세 군집을 생성하며 다음과 같이 해석:
- **어두운 군집** → CH (최저 강도)
- **정온 태양 (QS)** → 중간 강도
- **밝은 군집** → 활동영역 (AR, 최고 강도)

$k=2$가 기각된 이유는 두 군집만으로는 AR과 QS가 합쳐져서 CH-어두움 대비가 과대평가되고(밝은 "AR 끌어당김"이 사라지므로), 결과적으로 CH 영역이 과대평가되기 때문이다. $k=3$에서는 AR이 자체 군집으로 분리된다.

**이진 CH 마스크 구성**: 어두운 + 밝은 군집을 병합 → 비-CH로 라벨링(이로써 CH 군집만 1, 나머지 0). 그 다음 `scikit-image` (van der Walt et al. 2014)로 형태학적 정리:
- **Opening** (침식 → 팽창): min object size **200 픽셀**, connectivity **10 픽셀** — 작은 점-같은 가짜 탐지 제거
- **Closing** (팽창 → 침식): 디스크 footprint **반경 2 픽셀** — 식별된 CH 내부의 작은 구멍 채움

closing footprint는 의도적으로 작게 설정(반경 2) — CH 내부의 더 큰 진짜 밝은 점, 예를 들어 **coronal bright points**(Karachik et al. 2006; Hong et al. 2014; Wyper et al. 2018) 같은 실제 물리적 구조를 유지하기 위함.

**2CO(두 채널 중첩) 변형**: 2D 벡터 입력에 k-means를 돌리는 대신, 2CO는 193 Å와 211 Å의 이진 마스크의 **픽셀 단위 교집합**으로 계산. 두 단일 채널 방법 모두 CH로 분류한 픽셀만 CH로 인정 — 거짓양성을 억제하는 보수적 접근.

### Part V: Evaluation Metrics & Results / 평가 지표 및 결과 (§3.3, §3.4, §3.5)

**English**

**Pixel-wise evaluation metrics** are calculated against CATCH binary maps as the ground truth, using a confusion matrix at each pixel:

$$\text{IoU} = \frac{TP}{TP + FP + FN} \quad \text{(Jaccard index, Jaccard 1912)}$$

$$\text{TSS} = \frac{TP}{TP+FN} - \frac{FP}{FP+TN} \quad \text{(Hanssen \& Kuipers 1965)}$$

These are robust to class imbalance — only ~1–10% of solar-disk pixels are CH, so naive accuracy would be misleading.

**Headline results (Figure 5)**: median ± median absolute deviation across the 237-date sample.

| Method | IoU vs CATCH | TSS vs CATCH |
|---|---|---|
| **AIA 193**  | $0.62 \pm 0.14$ | $0.91 \pm 0.06$ |
| AIA 211 | $0.51 \pm 0.20$ | $0.83 \pm 0.13$ |
| **2CC** (193+211) | **$0.64 \pm 0.14$** | **$0.93 \pm 0.06$** |
| 3CC (171+193+211) | $0.50 \pm 0.21$ | $0.61 \pm 0.29$ |
| 2CO (193 ∩ 211) | $0.61 \pm 0.19$ | $0.73 \pm 0.21$ |
| HEK (SPoCA) | $0.53 \pm 0.13$ | $0.73 \pm 0.13$ |

Key observations:
- **2CC is the winner** — best IoU (0.64) and best TSS (0.93)
- **AIA 193 alone is nearly as good** — single-channel adequate for routine use
- **3CC drops dramatically** — adding 171 Å hurts. Probably 171 (Fe IX, ~10⁶ K, transition region) does not show CH-vs-QS contrast as cleanly as the hotter 193/211 channels because at 171 Å, both QS and CH show strong network/internetwork emission
- **2CO is conservative but loses TSS** (0.73 vs 0.93 for 2CC) — strict intersection drops some genuine CH pixels (high FN), even though it maintains high IoU
- **HEK/SPoCA is outperformed by every k-means variant except 3CC** — important sociological finding for the field

**CH areas (Figure 7, §3.4)**: CH coverage is computed in % of solar-disk area, with each pixel area corrected for projection by

$$A_i = \frac{A_{i,\text{proj}}}{\cos \alpha_i}$$

where $A_{i,\text{proj}}$ is the apparent pixel area and $\alpha_i$ is the heliographic angular distance from disk center. Pearson correlation coefficients with CATCH:

| Method | Pearson r vs CATCH |
|---|---|
| HEK | **0.88** |
| 2CC | 0.82 |
| AIA 193 | 0.81 |
| 2CO | 0.79 |
| 3CC | 0.75 |
| AIA 211 | 0.73 |

Interestingly, HEK has the highest area correlation with CATCH ($r=0.88$) despite lower pixel-wise IoU (0.53). This means HEK identifies similar **total areas** but different **specific pixels**. In contrast, 2CC and 193 Å have both high pixel agreement (IoU) and high area correlation.

**Yearly correlations (Figure 6)**: correlations vary year-to-year. After 2014 (declining/late phase of cycle 24), all methods become similar and evolve in parallel. Earlier (during max), discrepancies are larger — methods disagree most when the solar atmosphere is most active.

**Three case studies (Figure 8, §3.5)**: 2012 Nov 5 (inclining), 2014 Dec 7 (just after maximum), 2016 Dec 7 (declining):
- 2012 Nov 5: AIA 193, 3CC, 2CO match CATCH (only one CH at ~[0, 500] arcsec); AIA 211 and 2CC match HEK better
- 2014 Dec 7: 193, 2CC, 2CO show similar CH coverage to CATCH
- 2016 Dec 7: All methods except 3CC match HEK and CATCH; total CH coverage reaches a maximum, with one large CH extending from south pole to equator

**Temporal consistency check (Figure 9)**: 2CC is run for 9 consecutive days (2015 Nov 3–11). The detected CH evolution is consistent with **solar rotation** (~13°/day at equator), and a new equatorial CH appears between Nov 6 and Nov 11. This rules out frame-by-frame instability.

**Annual area trends (Figure 10)**: 2CC, CATCH, and HEK areas covary well during 2016 (declining phase), confirming consensus during stable conditions. In 2012 (active phase), HEK CH areas tend to be larger than 2CC and CATCH, reflecting genuine ambiguity at maximum.

**한국어**

**픽셀 단위 평가 지표**는 CATCH 이진 마스크를 ground truth로 하여 픽셀별 혼동 행렬로 계산:

$$\text{IoU} = \frac{TP}{TP + FP + FN} \quad \text{(Jaccard 지수, Jaccard 1912)}$$

$$\text{TSS} = \frac{TP}{TP+FN} - \frac{FP}{FP+TN} \quad \text{(Hanssen \& Kuipers 1965)}$$

이들은 클래스 불균형에 강건하다 — 태양 디스크의 ~1–10% 픽셀만 CH이므로 단순 정확도는 오도될 수 있다.

**핵심 결과 (Figure 5)**: 237 날짜 샘플의 중앙값 ± 중앙절대편차.

| 방법 | IoU vs CATCH | TSS vs CATCH |
|---|---|---|
| **AIA 193**  | $0.62 \pm 0.14$ | $0.91 \pm 0.06$ |
| AIA 211 | $0.51 \pm 0.20$ | $0.83 \pm 0.13$ |
| **2CC** (193+211) | **$0.64 \pm 0.14$** | **$0.93 \pm 0.06$** |
| 3CC (171+193+211) | $0.50 \pm 0.21$ | $0.61 \pm 0.29$ |
| 2CO (193 ∩ 211) | $0.61 \pm 0.19$ | $0.73 \pm 0.21$ |
| HEK (SPoCA) | $0.53 \pm 0.13$ | $0.73 \pm 0.13$ |

핵심 관찰:
- **2CC가 우승** — 최고 IoU(0.64) 및 최고 TSS(0.93)
- **AIA 193 단일도 거의 동등** — 운영용으로 단일 채널 충분
- **3CC는 극적 하락** — 171 Å 추가가 해롭다. 아마도 171 (Fe IX, ~10⁶ K, 전이영역)에서는 QS와 CH 모두 강한 network/internetwork 방출을 보여 CH 대비가 깔끔하지 않기 때문
- **2CO는 보수적이지만 TSS 손실** (0.73 vs 2CC의 0.93) — 엄격한 교집합이 일부 진짜 CH 픽셀을 떨어뜨림(높은 FN), IoU는 유지하지만
- **HEK/SPoCA가 3CC를 제외한 모든 k-means 변형에 의해 능가됨** — 분야에서 사회학적으로 중요한 발견

**CH 면적 (Figure 7, §3.4)**: CH 커버리지를 태양 디스크 면적의 %로 계산하며, 각 픽셀 면적은 다음과 같이 투영 보정:

$$A_i = \frac{A_{i,\text{proj}}}{\cos \alpha_i}$$

여기서 $A_{i,\text{proj}}$는 명목 픽셀 면적, $\alpha_i$는 디스크 중심으로부터의 태양 좌표 각거리. CATCH와의 Pearson 상관계수:

| 방법 | Pearson r vs CATCH |
|---|---|
| HEK | **0.88** |
| 2CC | 0.82 |
| AIA 193 | 0.81 |
| 2CO | 0.79 |
| 3CC | 0.75 |
| AIA 211 | 0.73 |

흥미롭게도 HEK는 픽셀 단위 IoU가 낮음에도(0.53), 면적 상관계수가 가장 높다($r=0.88$). 이는 HEK가 비슷한 **총 면적**을 식별하지만 **다른 특정 픽셀**들을 식별함을 의미한다. 반면 2CC와 193 Å는 픽셀 일치(IoU)와 면적 상관 모두 높다.

**연도별 상관관계 (Figure 6)**: 상관관계가 해마다 변동. 2014년 이후(주기 24의 하강/말기) 모든 방법이 비슷해지고 평행하게 진화. 그 이전(극대기)에는 불일치가 더 크다 — 태양 대기가 가장 활동적일 때 방법들이 가장 불일치한다.

**세 가지 사례 연구 (Figure 8, §3.5)**: 2012년 11월 5일(상승), 2014년 12월 7일(극대기 직후), 2016년 12월 7일(하강):
- 2012-11-05: AIA 193, 3CC, 2CO가 CATCH와 일치(~[0, 500] arcsec에 단 하나의 CH); AIA 211과 2CC는 HEK와 더 일치
- 2014-12-07: 193, 2CC, 2CO가 CATCH와 비슷한 CH 커버리지
- 2016-12-07: 3CC를 제외한 모든 방법이 HEK·CATCH와 일치; 총 CH 커버리지 최대치 도달, 남극에서 적도까지 뻗는 큰 CH 하나

**시간적 일관성 체크 (Figure 9)**: 2CC를 9일 연속(2015년 11월 3-11일)으로 실행. 탐지된 CH 진화가 **태양 자전**(적도에서 ~13°/일)과 일치하며, 11월 6-11일 사이에 새로운 적도 CH가 등장. 프레임별 불안정성 배제.

**연간 면적 추세 (Figure 10)**: 2CC, CATCH, HEK 면적이 2016년(하강기) 동안 잘 공변동 — 안정 조건에서 합의 확인. 2012년(활동기)에는 HEK CH 면적이 2CC와 CATCH보다 큰 경향 — 극대기의 본질적 모호성 반영.

### Part VI: Discussion & Call for Ground Truth / 토론 및 ground truth 호소 (§4)

**English**

The discussion frames the broader significance:
- CHs drive the steady fast solar wind, which produces **corotating interaction region (CIR)** -driven storms — the so-called **HILDCAA events** (High-Intensity Long-Duration Continuous AE Activity; Tsurutani & Gonzalez 1987)
- Therefore reliable CH detection is operationally important for space-weather forecasting
- The paper compares directly with **CHRONNOS** (Jarolim et al. 2021), a progressively-grown CNN that uses **all 7 AIA channels + HMI line-of-sight magnetograms**, and reports CHRONNOS achieves mean IoU = 0.63 and TSS = 0.81 against the same CATCH baseline. This paper's 2CC achieves mean IoU = 0.64 and TSS = 0.93 with **only 3 channels and pixel-wise k-means** — a striking demonstration of "simple is powerful"
- The authors then strongly argue: if observers (Linker et al. 2021; Reiss et al. 2021) can build a **community-consensus CH database**, supervised methods could move forward; until then, both supervised and unsupervised approaches are evaluated against inconsistent or single-observer truth

The conclusion statement encapsulates the paper's positioning:

> *"In conclusion, as an unsupervised ML method, using the k-means clustering provides better results with those from complex methods, such as CNNs. ... More importantly, our study shows that there is a need for a CH database where a consensus about the CH boundaries is reached by observers independently, and which can be used as the 'ground truth' when using a supervised method or just to evaluate the goodness of the models."*

**한국어**

토론은 더 넓은 의의를 제시한다:
- CH는 정상 고속 태양풍을 발생시키며, 이는 **공회전 상호작용 영역(CIR)** 기반 폭풍 — 이른바 **HILDCAA 이벤트**(High-Intensity Long-Duration Continuous AE Activity; Tsurutani & Gonzalez 1987)를 만든다
- 따라서 신뢰할 만한 CH 탐지는 우주기상 예보에 운영적으로 중요
- 본 논문은 **CHRONNOS** (Jarolim et al. 2021), **모든 7개 AIA 채널 + HMI LOS 자기도**를 사용하는 점진적 성장 CNN과 직접 비교 — CHRONNOS는 동일 CATCH 기준선에서 평균 IoU=0.63, TSS=0.81 달성. 본 논문의 2CC는 **단 3개 채널과 픽셀 단위 k-means만으로** 평균 IoU=0.64, TSS=0.93 달성 — "단순한 것이 강력하다"의 인상적 입증
- 저자들은 강하게 주장한다: 관측자들(Linker et al. 2021; Reiss et al. 2021)이 **커뮤니티 합의 CH 데이터베이스**를 구축할 수 있다면, 지도학습 방법이 전진할 수 있다. 그때까지는 지도·비지도 방법 모두 비일관적이거나 단일 관측자 진실에 대해 평가될 뿐

결론 문장이 본 논문의 자리매김을 함축한다:

> *"결론적으로, 비지도 ML 방법으로 k-means 군집화를 사용하면 CNN과 같은 복잡한 방법의 결과보다 더 나은 결과를 제공한다. ... 더 중요하게는, 본 연구는 관측자가 독립적으로 CH 경계에 대한 합의를 이루고 지도학습 방법 사용 시 또는 모델의 우수성을 평가할 때 'ground truth'로 사용될 수 있는 CH 데이터베이스가 필요함을 보여준다."*

---

## 3. Key Takeaways / 핵심 시사점

1. **Pixel-wise k-means with $k=3$ achieves CH segmentation competitive with CNNs.** 2CC reaches median IoU = 0.64 and TSS = 0.93 against CATCH ground truth — better than HEK/SPoCA (IoU = 0.53) and matching CHRONNOS (IoU = 0.63, TSS = 0.81) despite using only 3 channels vs CHRONNOS's 7 channels + magnetograms. / **$k=3$의 픽셀 단위 k-means가 CNN과 견줄 만한 CH 분할을 달성한다.** 2CC는 CATCH 기준에 대해 중앙값 IoU=0.64, TSS=0.93 — HEK/SPoCA(IoU=0.53)보다 우수, CHRONNOS(IoU=0.63, TSS=0.81)와 동등(7채널+자기도 vs 3채널만으로).

2. **The preprocessing pipeline matters more than the clustering algorithm.** The success of pixel-wise k-means hinges on (a) limb-brightening correction, (b) PSF deconvolution, (c) log-normal transformation, (d) bimodal-Gaussian thresholding at $\mu \pm 4\sigma$, and (e) systematic morphological cleanup. Without these, raw k-means would not perform competitively. / **전처리 파이프라인이 군집화 알고리즘보다 중요하다.** 픽셀 단위 k-means의 성공은 (a) 가장자리 밝기 보정, (b) PSF 디컨볼루션, (c) 로그-정규 변환, (d) $\mu \pm 4\sigma$의 양봉 가우시안 임계화, (e) 체계적 형태학적 정리에 달려 있다. 이들 없이는 raw k-means가 경쟁력 있게 작동하지 않을 것.

3. **More wavelengths is not always better — the 3CC penalty.** Adding 171 Å to the 193+211 stack drops IoU from 0.64 (2CC) to 0.50 (3CC) and TSS from 0.93 to 0.61. The reason: 171 Å (Fe IX, ~10⁶ K) shows network and internetwork structures in QS that mimic CH-like darkness, blurring the cluster boundaries. **Channel selection should be physics-informed.** / **파장 수가 많을수록 좋은 것은 아니다 — 3CC 페널티.** 193+211 스택에 171 Å 추가가 IoU를 0.64(2CC)에서 0.50(3CC)으로, TSS를 0.93에서 0.61로 떨어뜨림. 이유: 171 Å (Fe IX, ~10⁶ K)에서는 QS의 network·internetwork 구조가 CH-같은 어둠을 흉내내어 군집 경계를 흐림. **채널 선택은 물리에 기반해야 한다.**

4. **Pixel-level agreement (IoU) and total-area agreement (Pearson r) measure different things.** HEK has the highest area correlation with CATCH ($r=0.88$) but low pixel IoU (0.53), meaning it identifies similar **total** CH areas but different **specific** pixels. 2CC and 193 Å score well on both, suggesting genuinely better pixel-level segmentation. **Choose evaluation metrics that match the downstream use case** — operational forecasting may care about total area, while reconnection studies need pixel boundaries. / **픽셀 일치(IoU)와 총 면적 일치(Pearson r)는 서로 다른 것을 측정한다.** HEK는 CATCH와 면적 상관($r=0.88$) 최고이지만 픽셀 IoU(0.53) 저조 — 비슷한 **총** CH 면적이지만 다른 **특정** 픽셀 식별. 2CC와 193 Å는 두 지표 모두 양호 — 진정으로 더 나은 픽셀 수준 분할 시사. **하류 사용 사례에 맞는 평가 지표 선택** — 운영 예보는 총 면적 중시, 재결합 연구는 픽셀 경계 필요.

5. **The 2CO conservative criterion trades TSS for IoU.** Requiring 193 Å AND 211 Å agreement (intersection) gives competitive IoU (0.61) but drops TSS to 0.73 — the strict criterion creates many false negatives (missed CH pixels). Useful when you want to minimize false positives but not for total-coverage estimates. / **2CO 보수적 기준은 TSS를 IoU와 교환한다.** 193 Å AND 211 Å 동시 일치(교집합) 요구가 경쟁력 있는 IoU(0.61)를 주지만 TSS는 0.73으로 하락 — 엄격한 기준이 많은 거짓음성(놓친 CH 픽셀)을 만든다. 거짓양성 최소화에는 유용하지만 총 커버리지 추정에는 부적절.

6. **Solar-cycle dependence of method agreement.** All methods (k-means, HEK, CATCH) become similar and evolve in parallel after 2014 (declining phase) but disagree more during the maximum (2012-2014). **CH detection is hardest at solar maximum** when ARs are abundant and CHs are smaller and more transient. Algorithmic robustness should be evaluated phase-by-phase, not just averaged. / **방법 일치의 태양주기 의존성.** 모든 방법(k-means, HEK, CATCH)이 2014년 이후(하강기) 비슷해지고 평행하게 진화, 극대기(2012-2014)에는 더 불일치. **CH 탐지는 극대기에 가장 어렵다** — AR이 많고 CH가 작고 단명할 때. 알고리즘 견고성은 단순 평균이 아니라 위상별로 평가해야 한다.

7. **The labeling crisis is the field's actual bottleneck.** The paper's most important argument is sociological, not technical: without an observer-consensus CH ground truth, supervised methods cannot reach their full potential, and even excellent unsupervised results are evaluated against potentially flawed reference data. The same critique applies to filaments, AR boundaries, and CME identification across solar physics. / **라벨링 위기가 분야의 실제 병목이다.** 본 논문의 가장 중요한 주장은 기술적이 아닌 사회학적이다: 관측자 합의 CH ground truth 없이는 지도학습 방법이 잠재력을 발휘할 수 없으며, 우수한 비지도 결과조차 결함이 있을 수 있는 참조 데이터에 대해 평가된다. 같은 비판이 필라멘트, AR 경계, CME 식별 등 태양물리 전반에 적용된다.

8. **Reproducibility is high because the pipeline is interpretable.** Every step (limb correction, deconvolution, log-normal, $\mu \pm 4\sigma$, k-means $k=3$, opening-closing) is a well-known operation with documented parameters. Reproducing the paper requires only `aiapy`, `SunPy`, `scikit-image`, and `scikit-learn` — no proprietary trained weights, no GPU. **This is what scientific ML should look like.** / **파이프라인이 해석 가능하므로 재현성이 높다.** 모든 단계(가장자리 보정, 디컨볼루션, 로그-정규, $\mu \pm 4\sigma$, k-means $k=3$, opening-closing)가 잘 알려진 연산이며 파라미터가 문서화되어 있다. 논문 재현에는 `aiapy`, `SunPy`, `scikit-image`, `scikit-learn`만 필요 — 사적 학습 가중치 없음, GPU 불필요. **이것이 과학적 ML이 추구해야 할 모습이다.**

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 k-means objective (Lloyd's algorithm) / k-means 목적함수 (Lloyd 알고리즘)

$$\boxed{\;\text{SSD}(C_1, \ldots, C_k) = \sum_{i=1}^{k} \sum_{\mathbf{x} \in C_i} \|\mathbf{x} - \boldsymbol{\mu}_i\|_2^2\;}$$

- $C_i$ — the $i$-th cluster (a set of pixel-vectors) / $i$번째 군집
- $\mathbf{x} \in \mathbb{R}^d$ — pixel-intensity vector ($d=1, 2, 3$ for single/2CC/3CC inputs) / 픽셀 강도 벡터 ($d=1, 2, 3$)
- $\boldsymbol{\mu}_i = \frac{1}{|C_i|} \sum_{\mathbf{x} \in C_i} \mathbf{x}$ — centroid of cluster $i$ / 군집 $i$의 중심
- $\|\cdot\|_2$ — Euclidean norm / 유클리드 노름

**Lloyd iteration** (until SSD stops decreasing) / **Lloyd 반복** (SSD 감소 정지까지):
1. **Assign**: $C_i \leftarrow \{\mathbf{x} : i = \arg\min_j \|\mathbf{x} - \boldsymbol{\mu}_j\|_2\}$
2. **Update**: $\boldsymbol{\mu}_i \leftarrow \frac{1}{|C_i|}\sum_{\mathbf{x}\in C_i}\mathbf{x}$

For $k=3$ on a single-channel image, this reduces to finding three intensity values $\mu_{\text{CH}} < \mu_{\text{QS}} < \mu_{\text{AR}}$ that minimize the variance within each tier. / 단일 채널 영상에서 $k=3$은 각 단계 내 분산을 최소화하는 세 강도값 $\mu_{\text{CH}} < \mu_{\text{QS}} < \mu_{\text{AR}}$을 찾는 문제로 귀결.

### 4.2 Bimodal-Gaussian thresholding / 양봉 가우시안 임계화

The log-intensity histogram is fit by / 로그-강도 히스토그램을:

$$h(x) = w_1\,\mathcal{N}(x; \mu_1, \sigma_1^2) + w_2\,\mathcal{N}(x; \mu_2, \sigma_2^2)$$

where $w_1 + w_2 = 1$ are mixing weights, and $\mathcal{N}(x;\mu,\sigma^2) = \frac{1}{\sqrt{2\pi}\sigma}\exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$. The **higher-intensity peak** (parameters $\mu, \sigma$) is then used for threshold determination:

$$\boxed{\;T_{\text{low}} = \mu - 4\sigma, \quad T_{\text{up}} = \mu + 4\sigma\;}$$

After thresholding (clipping):
$$I'_{\text{px}} = \begin{cases} T_{\text{low}}, & I_{\text{px}} < T_{\text{low}} \\ I_{\text{px}}, & T_{\text{low}} \le I_{\text{px}} \le T_{\text{up}} \\ T_{\text{up}}, & I_{\text{px}} > T_{\text{up}} \end{cases}$$

**Interpretation / 해석**: clipping squeezes out very dark and very bright outliers (cosmic rays, instrument artifacts, extreme features) so that k-means operates on a well-conditioned dynamic range. The choice of $4\sigma$ is empirical (paper §3.1 also tested $3\sigma$ and $5\sigma$, finding $4\sigma$ optimal — see §4 of the paper).

### 4.3 Confusion-matrix metrics / 혼동 행렬 지표

For a binary CH mask $\hat{Y}$ (predicted) vs $Y$ (CATCH ground truth) — pixel by pixel:

| | $Y=1$ (CATCH=CH) | $Y=0$ (CATCH=non-CH) |
|---|---|---|
| $\hat{Y}=1$ (pred=CH) | TP | FP |
| $\hat{Y}=0$ (pred=non-CH) | FN | TN |

$$\text{IoU} = \frac{TP}{TP + FP + FN}$$

$$\text{TSS} = \frac{TP}{TP+FN} - \frac{FP}{FP+TN} = \text{Recall} - \text{FPR}$$

**Why both metrics?** / **왜 두 지표 모두?**
- **IoU** ranges from 0 (no overlap) to 1 (perfect agreement); penalizes FPs and FNs equally / 0(겹침 없음)~1(완벽 일치); FP·FN 동등 패널티
- **TSS** ranges from -1 (worst) to +1 (best); robust to large class imbalance because the FPR denominator $FP+TN$ is dominated by the abundant non-CH pixels / -1(최악)~+1(최고); 풍부한 비-CH 픽셀이 분모 $FP+TN$을 지배하므로 큰 클래스 불균형에 강건
- TSS = 0 corresponds to random guessing / TSS=0은 무작위 추측

### 4.4 Projection-corrected pixel area / 투영 보정 픽셀 면적

$$\boxed{\;A_i = \frac{A_{i,\text{proj}}}{\cos \alpha_i}\;}$$

- $A_{i,\text{proj}}$ — apparent (projected) pixel area at pixel $i$ / 명목 픽셀 면적
- $\alpha_i$ — heliographic angular distance from the disk center (so $\cos\alpha_i$ is the foreshortening factor) / 디스크 중심으로부터의 태양 좌표 각거리
- At disk center $\alpha=0$, no correction; at limb $\alpha \to 90°$, $\cos\alpha \to 0$ and $A_i \to \infty$ — projection correction is **most aggressive near the limb** where pixels represent the largest physical areas
- Total CH area on the disk is then $\sum_{\text{CH pixels}} A_i$, and CH coverage fraction is this divided by total solar disk area / 디스크 상 총 CH 면적은 $\sum_{\text{CH 픽셀}} A_i$, CH 커버리지 비율은 이를 총 태양 디스크 면적으로 나눈 값

### 4.5 Worked numerical example / 수치 사례

Consider a hypothetical 1024×1024 AIA 193 Å image after preprocessing. After log-normal transformation and bimodal Gaussian fit:
- Higher peak: $\mu = 3.0$ log-counts, $\sigma = 0.4$ → $T_{\text{low}} = 1.4$, $T_{\text{up}} = 4.6$
- After clipping, k-means with $k=3$ converges to centroids $\mu_{\text{CH}} = 1.8$, $\mu_{\text{QS}} = 3.1$, $\mu_{\text{AR}} = 4.2$
- Pixel assignment: ~5% CH, ~85% QS, ~10% AR
- After morphological opening (min size 200 px), ~50,000 CH pixels remain (out of ~52,400 raw)
- After projection correction and conversion: total CH coverage ≈ 4.7% of solar-disk area
- If CATCH binary map gives ~50,200 CH pixels, then TP ≈ 45,000, FP ≈ 5,000, FN ≈ 5,200, TN ≈ 944,800
- $\text{IoU} = 45000 / (45000 + 5000 + 5200) \approx 0.815$ (above the median 0.62)
- $\text{TSS} = 45000/50200 - 5000/949800 \approx 0.896 - 0.005 = 0.891$ (close to median 0.91)

이 가상 사례는 실제 보고된 분포 $\text{IoU} = 0.62 \pm 0.14$, $\text{TSS} = 0.91 \pm 0.06$ 범위 내에서 어떻게 픽셀 일치도와 TSS가 계산되는지 보여준다. / This hypothetical example illustrates how IoU and TSS are calculated within the reported ranges of $\text{IoU} = 0.62 \pm 0.14$, $\text{TSS} = 0.91 \pm 0.06$.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1912 ──── Jaccard, "The distribution of the flora in the alpine zone"
            │  → Jaccard index, later renamed IoU in computer vision
            │
1965 ──── Hanssen & Kuipers (Royal Netherlands Met. Inst.)
            │  → True Skill Statistic, foundational forecast verification metric
            │
1967 ──── MacQueen, "Some methods for classification and analysis"
            │  → k-means clustering algorithm (Berkeley Symposium)
            │
1973 ──── Skylab ATM observations
            │  → First clear identification of coronal holes in soft X-rays
            │
1995 ──── SOHO launched (Domingo, Fleck, Poland 1995)
            │  → EIT begins multiband EUV imaging at 171/195/284/304 Å
            │
2002 ──── Harvey & Recely, "Polar coronal holes during cycles 22 and 23"
            │  → Manual He I 10830 Å CH catalog establishes reference
            │
2009 ──── Krista & Gallagher, CHARM
            │  → Histogram-based intensity thresholding for automated CH detection
            │
2010 ──── SDO launched (Pesnell, Thompson & Chamberlin 2012)
2012 ──── Lemen et al., AIA reference paper / Hurlburt et al., HEK
            │  → 4096² full-disk EUV imaging every 12 s in 7 bands
            │
2014 ──── Verbeeck et al., SPoCA (spatial possibilistic clustering)
            │  → First production unsupervised CH/AR detection on AIA, posted to HEK
            │
2018 ──── Garton et al., CHIMERA
            │  → Multi-thermal segmentation using 171, 193, 211 Å
2018 ──── Illarionov & Tlatov, U-Net CNN
            │  → First deep-learning CH detection (trained on SPoCA labels)
            │
2019 ──── Heinemann et al., CATCH
            │  → Intensity-gradient semi-automated method, trusted CH binary maps
            │
2021 ──── Jarolim et al., CHRONNOS
            │  → Progressively-grown CNN; 7 AIA channels + HMI magnetograms;
            │    CHRONNOS achieves IoU = 0.63, TSS = 0.81 vs CATCH
2021 ──── Linker et al. / Reiss et al.
            │  → Begin community discussions of CH boundary uncertainties
            │
2022 ──── ★ THIS PAPER (Inceoglu et al.) ★
            │  → Pixel-wise k-means + 3 channels achieves IoU = 0.64, TSS = 0.93
            │  → Outperforms HEK/SPoCA, matches CHRONNOS with simpler method
            │  → Strong call for observer-consensus CH ground-truth database
            │
202X ──── (Future) Community CH ground truth?
            → Multi-observer consensus database;
              Solar Orbiter EUI / STEREO stereoscopic CH identification;
              Temporal-coherence-aware clustering;
              Integration with PFSS magnetic-field extrapolation
```

**Position in the field / 분야에서의 자리**: Inceoglu et al. (2022)는 "복잡한 딥러닝 vs 단순한 비지도 ML"이라는 잘못된 이분법을 깨는 중요한 비교 논문이며, 동시에 분야의 다음 단계 (community ground truth) 를 명시적으로 호소함으로써 의제 설정 역할을 한다. / Inceoglu et al. (2022) is an important comparison paper that breaks the false dichotomy of "complex deep learning vs simple unsupervised ML", and at the same time plays an agenda-setting role by explicitly calling for the field's next step (community ground truth).

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **#12 Lemen et al. 2012 — AIA on SDO** | Defines the very instrument (AIA) and passbands (171, 193, 211 Å) used in this paper / 본 논문이 사용하는 기기(AIA)와 통과대역(171, 193, 211 Å)을 정의 | Essential background — read first if AIA is new / 필수 배경 — AIA가 처음이면 먼저 읽기 |
| **#13 Scherrer et al. 2012 — HMI on SDO** | HMI provides the line-of-sight magnetograms that CHRONNOS (the comparison CNN) adds to the AIA channels; this paper deliberately does *not* use magnetograms | Sets up the comparison: does adding magnetograms (CHRONNOS) help over EUV-only k-means? Answer here: not by much / 비교 설정: 자기도 추가(CHRONNOS)가 EUV-only k-means 대비 도움이 되는가? 여기서의 답: 그다지 아님 |
| **#35 Pesnell, Thompson & Chamberlin 2012 — SDO** | Mission-level overview that frames AIA + HMI within the broader SDO observatory | Provides context for why SDO data is so well-suited (12-s cadence, full disk, multi-channel) for studies like this / SDO 데이터가 이런 연구에 왜 적합한지(12초 간격, 풀 디스크, 다채널) 맥락 제공 |
| **Heinemann et al. 2019 — CATCH** | Provides the **ground-truth** binary maps used to evaluate this paper's k-means results; CATCH itself is intensity-gradient + manual-curation | The paper's evaluation depends on CATCH being trustworthy — itself a single-observer reference, which is why the authors call for multi-observer consensus / 본 논문 평가는 CATCH의 신뢰성에 의존 — CATCH 자체도 단일 관측자 참조이기에 저자들이 다관측자 합의를 호소 |
| **Jarolim et al. 2021 — CHRONNOS** | The state-of-the-art CNN this paper compares to; CHRONNOS uses 7 AIA channels + HMI magnetograms in a progressively-grown CNN | Direct head-to-head: CHRONNOS IoU = 0.63 / TSS = 0.81 vs this paper's 2CC IoU = 0.64 / TSS = 0.93 — proof that simple+well-preprocessed > complex+raw / 직접 대결: CHRONNOS IoU=0.63/TSS=0.81 vs 본 논문 2CC IoU=0.64/TSS=0.93 — 단순+잘-전처리 > 복잡+raw |
| **Verbeeck et al. 2014 — SPoCA** | Spatial possibilistic clustering method whose results populate HEK; conceptually similar (clustering) but spatial-aware vs this paper's pixel-wise approach | Useful counterpoint: spatial-aware clustering does *not* necessarily beat pixel-wise + post-hoc morphology / 유용한 대조: 공간 인식 군집화가 픽셀 단위 + 사후 형태학을 반드시 이기지는 않음 |
| **Krista & Gallagher 2009 — CHARM** | Earliest fully-automated CH detection (histogram-based thresholding); this paper extends the spirit (intensity statistics) to multichannel + ML | Shows the algorithmic lineage from manual histogram thresholding → k-means thresholding / 수동 히스토그램 임계 → k-means 임계로의 알고리즘 계보 |
| **Garton et al. 2018 — CHIMERA** | Multi-thermal segmentation also using 171, 193, 211 Å; provides the closest single-paper baseline using identical wavelengths but without ML | Important comparison for the wavelength-choice argument / 파장 선택 논거의 중요 비교 |

---

## 7. References / 참고문헌

### Primary paper / 본 논문
- Inceoglu, F., Shprits, Y. Y., Heinemann, S. G., & Bianco, S., "Identification of Coronal Holes on AIA/SDO Images Using Unsupervised Machine Learning", *The Astrophysical Journal*, 930:118 (11pp), 2022. DOI: [10.3847/1538-4357/ac5f43](https://doi.org/10.3847/1538-4357/ac5f43)

### Key methodological references / 주요 방법론 참고문헌
- MacQueen, J., "Some methods for classification and analysis of multivariate observations", in *Proc. 5th Berkeley Symp. on Mathematical Statistics and Probability*, Vol. 1, pp. 281–297, Berkeley: Univ. of California Press, 1967.
- Likas, A., Vlassis, N., & Verbeek, J., "The global k-means clustering algorithm", *Pattern Recognition*, 36, 451, 2003.
- Paparrizos, J., & Gravano, L., "k-Shape: Efficient and Accurate Clustering of Time Series", in *Proc. 2015 ACM SIGMOD Int. Conf. on Management of Data*, p. 1855, 2015.
- Jaccard, P., "The Distribution of the Flora in the Alpine Zone", *New Phytologist*, 11, 37, 1912.
- Hanssen, A. W., & Kuipers, W. J. A., "On the relationship between the frequency of rain and various meteorological parameters", *Koninklijk Nederlands Meteorologisch Instituut Mededelingen Verhandelingen*, 81, 15, 1965.

### Coronal-hole detection lineage / CH 탐지 계보
- Harvey, K. L., & Recely, F., "Polar Coronal Holes During Cycles 22 and 23", *Solar Physics*, 211, 31, 2002.
- Krista, L. D., & Gallagher, P. T., "Automated Coronal Hole Detection Using Local Intensity Thresholding Techniques (CHARM)", *Solar Physics*, 256, 87, 2009.
- Verbeeck, C., Delouille, V., Mampaey, B., & De Visscher, R., "The SPoCA-suite", *A&A*, 561, A29, 2014.
- Garton, T. M., Gallagher, P. T., & Murray, S. A., "Automated coronal hole identification via multi-thermal intensity segmentation (CHIMERA)", *JSWSC*, 8, A02, 2018.
- Heinemann, S. G., Temmer, M., Heinemann, N., et al., "Statistical Analysis and Catalog of Non-polar Coronal Holes Covering the SDO-Era Using CATCH", *Solar Physics*, 294, 144, 2019.
- Heinemann, S. G., Temmer, M., Hofmeister, S. J., et al., "The Coronal Hole Bibliography. I. ...", *Solar Physics*, 296, 141, 2021.
- Illarionov, E. A., & Tlatov, A. G., "Segmentation of coronal holes in solar disc images with a convolutional neural network", *MNRAS*, 481, 5014, 2018.
- Jarolim, R., Veronig, A. M., Hofmeister, S., et al., "Multi-channel coronal hole detection with convolutional neural networks (CHRONNOS)", *A&A*, 652, A13, 2021.

### Instruments & data infrastructure / 기기 및 데이터 인프라
- Pesnell, W. D., Thompson, B. J., & Chamberlin, P. C., "The Solar Dynamics Observatory (SDO)", *Solar Physics*, 275, 3, 2012.
- Lemen, J. R., Title, A. M., Akin, D. J., et al., "The Atmospheric Imaging Assembly (AIA) on the Solar Dynamics Observatory (SDO)", *Solar Physics*, 275, 17, 2012.
- Scherrer, P. H., Schou, J., Bush, R. I., et al., "The Helioseismic and Magnetic Imager (HMI) Investigation for the Solar Dynamics Observatory (SDO)", *Solar Physics*, 275, 207, 2012.
- Hurlburt, N., Cheung, M., Schrijver, C., et al., "Heliophysics Event Knowledgebase for the Solar Dynamics Observatory (SDO) and Beyond", *Solar Physics*, 275, 67, 2012.
- The SunPy Community, Barnes, W. T., Bobra, M. G., et al., "The SunPy Project: Open Source Development and Status of the Version 1.0 Core Package", *ApJ*, 890, 68, 2020.
- Mumford, S. J., Freij, N., Christe, S., et al., "SunPy v3.0.3", Zenodo, doi:10.5281/zenodo.5751998, 2021.
- Barnes, W. T., Cheung, M., Bobra, M., et al., "aiapy: A Python Package for Analyzing Solar EUV Image Data from AIA v0.3.1", Zenodo, doi:10.5281/zenodo.4274931, 2020a.
- Barnes, W. T., Cheung, M. C. M., Bobra, M. G., et al., "aiapy", *JOSS*, 5, 2801, 2020b.
- van der Walt, S., Schönberger, J. L., Nunez-Iglesias, J., et al., "scikit-image: image processing in Python", *PeerJ*, 2, e453, 2014.
- Ronneberger, O., Fischer, P., & Brox, T., "U-Net: Convolutional Networks for Biomedical Image Segmentation", in *MICCAI 2015*, Springer, 2015.

### Coronal-hole physics & solar wind / CH 물리 및 태양풍
- Wilcox, J. M., "The Interplanetary Magnetic Field. Solar Origin and Terrestrial Effects", *Space Sci. Rev.*, 8, 258, 1968.
- Cranmer, S. R., "Coronal Holes", *Living Reviews in Solar Physics*, 6, 3, 2009.
- Schwenn, R., "Space Weather: The Solar Perspective", *Living Reviews in Solar Physics*, 3, 2, 2006.
- Tsurutani, B. T., & Gonzalez, W. D., "The Cause of High-Intensity Long-Duration Continuous AE Activity (HILDCAAs)", *Planet. Space Sci.*, 35, 405, 1987.
- Eastwood, J. P., Biffis, E., Hapgood, M. A., et al., "The Economic Impact of Space Weather", *Risk Analysis*, 37, 206, 2017.

### Coronal bright points (relevant to morphological closing parameter choice) / coronal bright points (형태학적 closing 파라미터 선택과 관련)
- Karachik, N., Pevtsov, A. A., & Sattarov, I., "Bright Points in the Solar Corona Identified in Soft X-Ray Images", *ApJ*, 642, 562, 2006.
- Hong, J., Jiang, Y., Yang, J., et al., "Coronal Bright Points Associated with Minifilament Eruptions", *ApJ*, 796, 73, 2014.
- Wyper, P. F., DeVore, C. R., Karpen, J. T., Antiochos, S. K., & Yeates, A. R., "A Model for Coronal Hole Bright Points and Jets due to Moving Magnetic Elements", *ApJ*, 864, 165, 2018.

### CH consensus / Solar wind context (called for in §4) / CH 합의 / 태양풍 맥락 (§4에서 호소됨)
- Linker, J. A., Heinemann, S. G., Temmer, M., et al., "Coronal Hole Detection and Open Magnetic Flux", *ApJ*, 918, 21, 2021.
- Reiss, M. A., Muglach, K., Möstl, C., et al., "The Observational Uncertainty of Coronal Hole Boundaries in Automated Detection Schemes", *ApJ*, 913, 28, 2021.
- Hewins, I. M., Gibson, S. E., Webb, D. F., et al., "The Evolution of Coronal Holes over Three Solar Cycles ...", *Solar Physics*, 295, 161, 2020.

### Foundational solar-cycle observation / 기초 태양주기 관측
- Schwabe, H., "Sonnen-Beobachtungen im Jahre 1843", *Astronomische Nachrichten*, 21, 233, 1844.
- Marsch, E., "Solar Wind and Inner Heliosphere", *Living Reviews in Solar Physics*, 3, 1, 2006.
- Schmidhuber, J., "Deep learning in neural networks: An overview", *Neural Networks*, 61, 85, 2014.
- LeCun, Y., Bengio, Y., & Hinton, G., "Deep learning", *Nature*, 521, 436, 2015.
