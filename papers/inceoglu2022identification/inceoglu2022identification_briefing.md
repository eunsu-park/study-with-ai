---
title: "Pre-Reading Briefing: Identification of Coronal Holes on AIA/SDO Images Using Unsupervised Machine Learning"
paper_id: "37_inceoglu_2022"
topic: Solar_Observation
date: 2026-04-20
type: briefing
---

# Identification of Coronal Holes on AIA/SDO Images Using Unsupervised Machine Learning: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Inceoglu, F., Shprits, Y. Y., Heinemann, S. G., & Bianco, S., "Identification of Coronal Holes on AIA/SDO Images Using Unsupervised Machine Learning", *The Astrophysical Journal*, 930:118 (11pp), 2022. DOI: [10.3847/1538-4357/ac5f43](https://doi.org/10.3847/1538-4357/ac5f43)
**Author(s)**: Fadil Inceoglu (GFZ Potsdam / NCEI/NOAA), Yuri Y. Shprits (GFZ Potsdam / UCLA), Stephan G. Heinemann (MPS Göttingen), Stefano Bianco (GFZ Potsdam)
**Year**: 2022 (Received Jan 28, 2022; Accepted Mar 19, 2022; Published May 10, 2022)

---

## 1. 핵심 기여 / Core Contribution

### 한국어
이 논문은 SDO 위성의 AIA(Atmospheric Imaging Assembly) 가 촬영한 EUV 통과대역 영상 (171 Å, 193 Å, 211 Å)에 대해 **픽셀 단위 k-means 비지도 군집화 (unsupervised clustering)** 를 적용하여 코로나홀(Coronal Hole, CH)을 자동 탐지한다. 세 개의 통과대역을 단일 채널, 2채널 합성(2CC), 3채널 합성(3CC), 그리고 두 통과대역의 중첩(2CO)으로 조합하여 네 가지 입력에 대한 결과를 비교한다. 체계적인 전처리(가장자리 밝기 보정, PSF 디컨볼루션, 로그-정규 변환, 양방향 가우시안 임계화)와 후처리(형태학적 열기/닫기 연산)을 함께 적용하면, 단순한 픽셀-단위 k-means가 **CNN 기반 복잡한 방법(SPoCA, U-Net 등)과 비견할 만한 성능**을 낸다는 것을 보인다. 더 중요하게는, 관측자 간 합의가 이루어진 **"ground truth" CH 데이터베이스의 부재**가 지도학습 방법의 한계를 만들고 있음을 지적한다.

### English
This paper applies **pixel-wise k-means unsupervised clustering** to EUV passband images (171 Å, 193 Å, 211 Å) from the Atmospheric Imaging Assembly (AIA) on the Solar Dynamics Observatory (SDO) to automatically identify coronal holes (CHs). Four input configurations are compared: single-channel (193 Å, 211 Å), two-channel composite (2CC), three-channel composite (3CC), and two-channel overlap (2CO). The authors show that with systematic preprocessing (limb-brightening correction, PSF deconvolution, log-normal transformation, bimodal-Gaussian thresholding) and postprocessing (morphological opening/closing), simple pixel-wise k-means produces results **competitive with complex methods such as SPoCA and CNN-based approaches**. Crucially, they argue that the absence of a community-agreed "ground-truth" CH database is a fundamental obstacle for supervised methods, and that an observer-consensus database is needed.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

#### 한국어
코로나홀은 1973년 Skylab의 X-선 관측에서 처음 명확히 식별되었으며, 이후 EUV·He I 10830 Å 적외선 관측에서도 어두운 영역으로 나타났다. CH는 **열린 자기력선** 구조로, 밀도와 온도가 주변보다 낮아 연속체 방출이 적어 어둡게 보인다. 또한 CH는 **고속 태양풍(fast solar wind, ~700 km/s)** 의 발원지이므로, 우주기상 예보에서 정확한 CH 경계 탐지가 핵심이다.

지난 20년간 CH 탐지 방법은 (1) **수동 식별** (He I 10830 Å; Harvey & Recely 2002), (2) **단순 임계값 기반** (CHARM; Krista & Gallagher 2009), (3) **다중 온도 분할** (CHIMERA; Garton et al. 2018), (4) **강도 기울기 + 반자동** (CATCH; Heinemann et al. 2019), (5) **공간 가능론적 군집화** (SPoCA; Verbeeck et al. 2014), (6) **U-Net CNN** (Illarionov & Tlatov 2018), (7) **점진적 성장 CNN** (Jarolim et al. 2021)로 진화했다. 그러나 각 방법은 임계값·구조요소·학습 데이터에 따라 결과가 다르며, **표준화된 CH 정의가 없다**.

#### English
Coronal holes were first clearly identified in 1973 by Skylab X-ray observations, then in EUV and He I 10830 Å near-infrared images, where they appear as dark regions. CHs consist of **open magnetic field lines** with lower density and temperature, producing weaker continuum emission. They are also the source of the **fast solar wind (~700 km/s)**, making accurate CH-boundary identification critical for space weather forecasting.

Over the past two decades, CH detection methods have evolved through (1) **manual identification** (He I 10830 Å; Harvey & Recely 2002), (2) **histogram-based intensity thresholding** (CHARM; Krista & Gallagher 2009), (3) **multi-thermal segmentation** (CHIMERA; Garton et al. 2018), (4) **intensity-gradient semi-automated methods** (CATCH; Heinemann et al. 2019), (5) **spatial possibilistic clustering** (SPoCA; Verbeeck et al. 2014), (6) **U-Net CNNs** (Illarionov & Tlatov 2018), and (7) **progressively-grown CNNs** (Jarolim et al. 2021). Each method depends on choices of thresholds, structuring elements, or training data, and **no standardised CH definition exists**.

### 타임라인 / Timeline

```
1973 ──── Skylab X-ray observations: CHs first clearly identified
            │
1990s ──── SOHO/EIT (1995): EUV imaging revolution begins
            │
2002 ──── Harvey & Recely: He I 10830 Å manual CH identification
            │
2009 ──── CHARM (Krista & Gallagher): histogram-based thresholding
            │
2010 ──── SDO launched; AIA produces full-disk EUV imagery every 12 s
            │
2012 ──── Hurlburt et al.: HEK (Heliophysics Event Knowledge base)
2014 ──── SPoCA (Verbeeck et al.): spatial possibilistic clustering
            │
2018 ──── CHIMERA (Garton et al.): multi-thermal segmentation
            │  Illarionov & Tlatov: U-Net CNN approach
2019 ──── CATCH (Heinemann et al.): intensity-gradient semi-automated method
            │
2021 ──── Jarolim et al.: progressively-grown CNN (multi-channel)
            │
2022 ──── ★ THIS PAPER ★ (Inceoglu et al.):
            pixel-wise k-means ≈ CNN performance;
            calls for observer-consensus ground truth
```

---

## 3. 필요한 배경 지식 / Prerequisites

### 한국어
1. **태양 대기 구조와 코로나홀**: 광구→채층→전이영역→코로나의 온도/밀도 구조, 닫힌/열린 자기력선의 차이, CH가 고속 태양풍의 발원지인 이유
2. **AIA/SDO 기기**: 12초마다 4096×4096 픽셀 풀-디스크 영상, 1픽셀 = 0.6″, 7개 EUV 채널 중 본 논문은 171 Å (Fe IX, ~10⁶ K, 정온 코로나), 193 Å (Fe XII/XXIV, ~1.5×10⁶ K, 코로나/플레어), 211 Å (Fe XIV, ~2×10⁶ K, 활동영역 코로나)
3. **태양 활동주기**: Schwabe 11년 주기, sunspot maximum/minimum, CH의 위치 변화 (극관 → 중위도 → 적도)
4. **k-means 군집화**: Lloyd 반복 알고리즘, 무작위 초기화 → centroid 갱신 → SSD 최소화. 입력은 픽셀 강도값(1·2·3차원 벡터)
5. **이미지 전처리 기법**:
   - **Limb-brightening correction** (annulus 방법): LOS가 길어져 발생하는 가장자리 밝기 증가 보정
   - **PSF deconvolution**: 기기 응답함수의 영향 제거
   - **로그-정규(log-normal) 변환**: 강도 분포의 비대칭성 완화
   - **Bimodal Gaussian fitting**: 어두운(CH+QS) 봉우리와 밝은(QS+AR) 봉우리 분리
6. **형태학적 연산 (morphological operations)**: scikit-image의 erosion/dilation, opening/closing, 작은 점 제거(min size 200 px), 작은 구멍 채우기(disk radius 2 px)
7. **평가 지표**: 혼동 행렬(TP/TN/FP/FN), IoU = Jaccard index, TSS = Hanssen-Kuipers discriminant
8. **Python 패키지**: SunPy, aiapy, scikit-image, scikit-learn

### English
1. **Solar atmosphere & coronal holes**: temperature/density profile from photosphere → chromosphere → transition region → corona; closed vs. open magnetic field lines; why CHs are the source of fast solar wind
2. **AIA/SDO instrument**: full-disk 4096×4096 px every 12 s, 0.6″/px, 7 EUV channels; this paper uses 171 Å (Fe IX, ~10⁶ K, quiet corona), 193 Å (Fe XII/XXIV, ~1.5×10⁶ K, corona/flare), 211 Å (Fe XIV, ~2×10⁶ K, active-region corona)
3. **Solar activity cycle**: 11-year Schwabe cycle, sunspot maximum/minimum, CH locations evolve (polar caps → midlatitudes → equator)
4. **k-means clustering**: Lloyd's iterative algorithm — random init → centroid update → SSD minimization. Inputs are pixel intensities (1-, 2-, or 3-D vectors)
5. **Image preprocessing techniques**:
   - **Limb-brightening correction** (annulus method): removes apparent edge brightening from longer LOS
   - **PSF deconvolution**: removes instrument response
   - **Log-normal transformation**: tames the asymmetric intensity distribution
   - **Bimodal Gaussian fitting**: separates dark (CH + QS) and bright (QS + AR) peaks
6. **Morphological operations**: scikit-image erosion/dilation, opening/closing; small-spot removal (min size 200 px), small-hole filling (disk radius 2 px)
7. **Evaluation metrics**: confusion matrix (TP/TN/FP/FN), IoU = Jaccard index, TSS = Hanssen-Kuipers discriminant
8. **Python packages**: SunPy, aiapy, scikit-image, scikit-learn

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **Coronal Hole (CH)** | 열린 자기력선 영역으로 밀도·온도가 낮아 EUV에서 어둡게 보임. 고속 태양풍의 발원지. / Open magnetic-field region with low density/temperature, appearing dark in EUV. Source of the fast solar wind. |
| **AIA (Atmospheric Imaging Assembly)** | SDO에 탑재된 EUV 영상 망원경. 7개 통과대역, 0.6″ 픽셀, 12초 주기. / EUV imaging telescope on SDO. 7 passbands, 0.6″/px, 12-s cadence. |
| **Passband** | 특정 Fe 이온 방출선 중심의 좁은 파장 대역 (e.g., Fe IX @ 171 Å). / Narrow wavelength band centered on specific Fe ion emission lines. |
| **k-means clustering** | k개의 군집 중심으로 데이터를 분할하는 비지도 알고리즘. SSD 최소화. / Unsupervised algorithm partitioning data into k clusters by minimizing SSD. |
| **SSD (Sum of Squared Distances)** | $\sum_{i=1}^{k}\sum_{x\in C_i}\|x-\mu_i\|^2$. 군집 내 산포도. / Within-cluster sum of squares, the k-means objective. |
| **Scree plot** | SSD vs k 그래프. "팔꿈치(elbow)"에서 최적 k 결정. / SSD vs k curve; the "elbow" indicates the optimal k. |
| **CATCH** | "Collection of Analysis Tools for Coronal Holes". 강도 기울기 기반 반자동 CH 탐지 도구. 본 논문의 비교 기준선. / Intensity-gradient-based semi-automated CH detection; baseline used here. |
| **HEK (Heliophysics Event Knowledge base)** | 태양 이벤트(AR, CH, flare 등)의 메타데이터 카탈로그. / Catalog of solar event metadata (ARs, CHs, flares, …). |
| **SPoCA** | "Spatial Possibilistic Clustering Algorithm". HEK에 결과가 게시되는 자동 CH 탐지 알고리즘. / Spatial possibilistic clustering algorithm whose results are published in HEK. |
| **Limb brightening** | 광시야 가장자리에서 LOS가 길어 코로나 방출이 더 많이 적분되어 밝아 보이는 현상. / Apparent intensity increase near solar limb due to a longer LOS through the optically thin corona. |
| **PSF (Point Spread Function)** | 기기가 점광원에 응답하는 형태. 디컨볼루션으로 보정. / Instrument's response to a point source; removed via deconvolution. |
| **Morphological opening/closing** | 침식 후 팽창(opening) → 작은 점 제거. 팽창 후 침식(closing) → 작은 구멍 채움. / Erosion-then-dilation removes small spots; dilation-then-erosion fills small holes. |
| **IoU (Intersection over Union)** | $\text{TP}/(\text{TP}+\text{FP}+\text{FN})$. 두 이진 마스크의 겹침 척도. / Overlap measure between two binary masks. |
| **TSS (True Skill Statistic)** | $\text{TPR}-\text{FPR}$. 클래스 불균형에서도 유효한 균형 정확도. / Balanced accuracy unaffected by class imbalance. |
| **2CC / 3CC / 2CO** | 2-channel composite / 3-channel composite / 2-channel overlap of binary masks (193+211 Å overlap만 CH로 인정). / Two/three-channel composite or two-channel overlap of binary masks (only pixels classified as CH in both 193 and 211 Å are kept). |

---

## 5. 수식 미리보기 / Equations Preview

### 5.1 k-means 목적함수 / k-means objective (SSD)

$$\text{SSD} = \sum_{i=1}^{k} \sum_{\mathbf{x} \in C_i} \|\mathbf{x} - \boldsymbol{\mu}_i\|^2$$

- $C_i$: $i$번째 군집, $\boldsymbol{\mu}_i$: 그 군집의 중심(centroid). $\mathbf{x}$는 픽셀 강도 벡터(1D, 2D, 또는 3D). / $C_i$ is the $i$-th cluster, $\boldsymbol{\mu}_i$ its centroid, and $\mathbf{x}$ a pixel-intensity vector (1D, 2D, or 3D depending on the input configuration).
- 각 픽셀을 가장 가까운 centroid에 할당하고, centroid를 군집 평균으로 갱신하는 것을 SSD가 더 이상 감소하지 않을 때까지 반복한다 (Lloyd 알고리즘). / Each pixel is assigned to the nearest centroid, then centroids are updated to cluster means until SSD no longer decreases (Lloyd's algorithm).

### 5.2 임계값 결정 / Threshold determination

각 통과대역의 후처리된 강도 분포에 양봉(bimodal) 가우시안을 피팅하여 평균 $\mu$와 표준편차 $\sigma$를 얻은 뒤: / After fitting a bimodal Gaussian to the postprocessed intensity distribution, extract the mean $\mu$ and standard deviation $\sigma$:

$$T_{\text{low}} = \mu - 4\sigma, \qquad T_{\text{up}} = \mu + 4\sigma$$

- $T_{\text{low}}$ 미만 픽셀은 모두 $T_{\text{low}}$로, $T_{\text{up}}$ 초과는 $T_{\text{up}}$로 클리핑(stack)하여 대비를 증가시킨다. / Pixels below $T_{\text{low}}$ are clipped to $T_{\text{low}}$ and pixels above $T_{\text{up}}$ to $T_{\text{up}}$, increasing the dynamic range used by k-means.

### 5.3 IoU (Intersection over Union, Jaccard 1912)

$$\text{IoU} = \frac{TP}{TP + FP + FN}$$

- $TP$: 두 이진 마스크 모두 CH인 픽셀, $FP$: 본 방법은 CH지만 CATCH는 비-CH, $FN$: 본 방법은 비-CH지만 CATCH는 CH. / $TP$: pixels labeled CH in both maps; $FP$: CH in our method but not in CATCH; $FN$: CH in CATCH but not in our method.
- 0(불일치) ~ 1(완벽 일치)의 값. / Ranges from 0 (no overlap) to 1 (perfect agreement).

### 5.4 TSS (True Skill Statistic, Hanssen & Kuipers 1965)

$$\text{TSS} = \frac{TP}{TP + FN} - \frac{FP}{FP + TN} = \text{TPR} - \text{FPR}$$

- 클래스 불균형(CH는 픽셀 수가 매우 적음)에 강건한 평가지표. / Robust to class imbalance (CH pixels are very sparse).
- $-1$ ~ $+1$ 범위. $0$: 무작위 추측 수준, $1$: 완벽. / Range $[-1, 1]$; $0$ = random guessing, $1$ = perfect skill.

### 5.5 Scree-plot 기반 최적 $k$ / Optimal $k$ via the scree plot

저자들은 $k=1, 2, \ldots, 10$에 대해 SSD를 계산해 "팔꿈치"가 $k=3$에서 나타남을 확인하였다. 이는 영상이 (i) 어두운 영역(CH), (ii) 정온 태양(QS), (iii) 밝은 영역(AR)의 세 군집으로 자연스럽게 분리됨을 의미한다. / The authors compute SSD for $k=1, 2, \ldots, 10$ and find an elbow at $k=3$, showing that the images naturally split into three clusters: (i) dark regions (CHs), (ii) quiet Sun (QS), and (iii) bright regions (active regions, ARs).

---

## 6. 읽기 가이드 / Reading Guide

### 한국어 권장 읽기 순서

1. **Section 1 (Introduction, pp. 1-2)** — 천천히 정독.
   - CH의 정의, 태양풍과의 관계, 우주기상에서의 중요성을 확실히 이해
   - 기존 탐지 방법들(CHARM, CHIMERA, CATCH, SPoCA, U-Net) 의 차이점과 한계 정리
2. **Section 2 (Data, p. 2)** — 빠르게 통독.
   - 어떤 데이터(AIA 171/193/211 Å, 237 dates, 2010-2016 11~12월), 왜 그 기간을 선택했는지 (CATCH 신뢰성 + cycle 24 커버)
3. **Section 3.1 (Preprocessing, pp. 2-4)** — 정독, 그림 2-3 함께 검토.
   - 전처리 파이프라인: degradation → registration → normalization → limb-brightening → deconvolution → 1024×1024 rescale → log-normal → bimodal Gaussian → $\mu \pm 4\sigma$ 클리핑
   - **Figure 2**: 전처리 전후의 PD(probability density) 비교
   - **Figure 3**: $\mu$, $\mu - 4\sigma$의 시간 변동 — 211 Å에서 음수 임계값이 27일에서 발생 (0으로 처리)
4. **Section 3.2 (k-means clustering, p. 4-5)** — 정독, **Figure 4 (scree-plot)** 와 함께.
   - 4가지 입력: 193, 211, 2CC(193+211), 3CC(171+193+211); k=3 선택 근거
   - 후처리: dark + bright cluster를 비-CH로 합침 → CH 이진 마스크
   - 형태학적 정리: opening (min size 200 px), closing (disk radius 2 px)
   - **2CO**: 193과 211의 이진 마스크 교집합만 CH로 인정 (보수적 접근)
5. **Section 3.3 (Evaluation, p. 5+)** — **Figure 5 (violin plots) 가 핵심**.
   - IoU·TSS의 분포 (vs CATCH ground truth, vs HEK)
   - **핵심 결과**: 193 Å와 2CC가 가장 높은 일치도 (IoU ~0.62-0.64, TSS ~0.91-0.93). 흥미롭게도 HEK(SPoCA)의 CATCH 일치도(IoU ~0.53)보다 본 방법이 더 좋다
6. **Section 4 (Discussion & Conclusions)** — 통독.
   - 단순 k-means가 CNN과 견줄 수 있는 이유 (잘 설계된 전처리)
   - 핵심 메시지: **observer-consensus CH database가 필요함**

### English Recommended Reading Order

1. **Section 1 (Introduction, pp. 1-2)** — read carefully.
   - Solidify your understanding of CH definition, link to solar wind, importance for space weather
   - Note differences/limitations of prior methods (CHARM, CHIMERA, CATCH, SPoCA, U-Net)
2. **Section 2 (Data, p. 2)** — skim quickly.
   - What data (AIA 171/193/211 Å, 237 dates from Nov-Dec each year, 2010-2016) and why this period (CATCH reliability + cycle-24 coverage)
3. **Section 3.1 (Preprocessing, pp. 2-4)** — read carefully alongside Figs. 2-3.
   - Pipeline: degradation → registration → normalization → limb-brightening → deconvolution → 1024² rescale → log-normal → bimodal Gaussian → $\mu \pm 4\sigma$ clipping
   - **Figure 2**: PD before vs after postprocessing
   - **Figure 3**: temporal variation of $\mu$, $\mu - 4\sigma$ — 211 Å yields negative thresholds on 27 days (set to zero)
4. **Section 3.2 (k-means clustering, pp. 4-5)** — read carefully with **Figure 4 (scree plot)**.
   - Four inputs: 193, 211, 2CC (193+211), 3CC (171+193+211); rationale for k=3
   - Post-processing: merge dark+bright clusters into non-CH → binary CH mask
   - Morphological cleanup: opening (min size 200 px), closing (disk radius 2 px)
   - **2CO**: keep only intersection of 193 and 211 CH masks (conservative)
5. **Section 3.3 (Evaluation, p. 5+)** — **Figure 5 (violin plots) is the key result**.
   - Distributions of IoU and TSS (vs CATCH ground truth, vs HEK)
   - **Headline result**: 193 Å and 2CC give the best agreement (IoU ~0.62-0.64, TSS ~0.91-0.93). Interestingly, this method outperforms HEK (SPoCA) vs CATCH (IoU ~0.53)
6. **Section 4 (Discussion & Conclusions)** — skim through.
   - Why simple k-means competes with CNNs (well-designed preprocessing)
   - Core takeaway: **need for an observer-consensus CH database**

### 읽으면서 메모할 질문 / Questions to keep in mind

- 한국어
  1. 왜 hot 채널인 211 Å보다 193 Å가 더 좋은 일치도를 보이는가?
  2. 3CC(171 추가)가 왜 오히려 성능이 떨어지는가? (171의 lower-density CH-QS 대비 부족?)
  3. CATCH를 ground truth로 쓰는 것이 정당한가? CATCH 자체도 단일 관측자의 결과인데?
  4. k=3이 최적이라는 것이 모든 날짜에 동일한가? (활동 극대기 vs 극소기)
  5. 형태학적 연산의 파라미터(min size 200 px, disk radius 2 px)는 어떻게 정해졌는가?

- English
  1. Why does 193 Å give better agreement than the hotter 211 Å?
  2. Why does adding 171 Å (3CC) actually hurt performance? (Insufficient CH-QS contrast at 171?)
  3. Is using CATCH as ground truth defensible, given CATCH itself is the work of a single observer?
  4. Is k=3 truly optimal on every date (solar maximum vs minimum)?
  5. How were the morphological parameters (min size 200 px, disk radius 2 px) chosen?

---

## 7. 현대적 의의 / Modern Significance

### 한국어
1. **"단순한 것이 강력하다"의 재확인**: 잘 설계된 전처리 + 고전적 k-means가 복잡한 CNN과 견줄 수 있다는 점은, 실시간 우주기상 운영 환경에서 **계산 효율성과 해석 가능성**이 중요한 경우 큰 의의가 있다 (CNN은 GPU 필요, k-means는 CPU로 충분).
2. **"라벨 위기(labeling crisis)"의 명확한 지적**: 지도학습이 만개한 2020년대에 저자들은 "CH가 무엇인가에 대한 합의가 없다"는 근본 문제를 짚어낸다. 이는 CH 분야뿐 아니라 태양물리 전반(필라멘트, 활동영역 경계, CME 식별 등)에 적용되는 메시지다.
3. **운영 우주기상 예보와의 연결**: NOAA·NCEI의 공저자가 포함되어 있어, 결과가 학술적 호기심에 머물지 않고 실제 예보 운영에 활용될 가능성이 높다.
4. **차세대 연구 방향**: (1) 다중 관측자 합의 데이터베이스 구축, (2) 자기력선 추적과의 통합 (PFSS extrapolation), (3) Solar Orbiter EUI·STEREO와의 결합으로 입체적 CH 식별, (4) 시계열 일관성을 강제하는 군집화 (이번 연구는 각 날짜를 독립적으로 처리).
5. **Solar_Observation 트랙에서의 위치**: Phase 7 (Calibration, Data Processing & Techniques)의 좋은 사례 연구. AIA 표준 처리 파이프라인 (`aiapy`) 사용법을 함께 학습할 수 있다.

### English
1. **"Simple is powerful" revisited**: Carefully designed preprocessing + classical k-means competing with complex CNNs has real implications for **operational space weather pipelines**, where computational efficiency and interpretability matter (k-means runs on CPU, CNNs require GPUs).
2. **A clear statement of the "labeling crisis"**: In an era when supervised learning dominates, the authors point at a foundational problem — there is **no community-agreed definition of a CH**. The same message generalises to filament detection, active-region boundaries, CME identification, and more.
3. **Direct link to operational space weather forecasting**: With NOAA/NCEI co-authorship, results are more likely to feed real forecasting pipelines than to remain purely academic.
4. **Future directions**: (1) build a multi-observer consensus CH database; (2) integrate with magnetic-field extrapolation (PFSS); (3) combine AIA with Solar Orbiter EUI / STEREO for stereoscopic CH identification; (4) enforce temporal consistency in clustering (this paper treats each date independently).
5. **Place in the Solar_Observation track**: a strong case study for Phase 7 (Calibration, Data Processing & Techniques). Reading this paper is also an opportunity to learn the standard AIA processing pipeline (`aiapy`).

---

## Q&A

### Q1. 클러스터링을 픽셀 단위로 진행한건가? / Was the clustering done pixel-by-pixel?

**한국어**

네. 본 논문의 핵심은 **pixel-wise k-means** 입니다. 각 픽셀이 하나의 독립적인 데이터 포인트로 취급되고, 픽셀의 강도값(들)을 좌표로 하는 공간에서 $k=3$ 군집으로 분할됩니다.

입력 벡터의 차원은 사용한 통과대역 수와 같습니다:

| 입력 구성 / Configuration | 픽셀당 벡터 / Per-pixel vector |
|---|---|
| 193 Å 단일 / single | 1D: $(I_{193})$ |
| 211 Å 단일 / single | 1D: $(I_{211})$ |
| 2CC (193+211) | 2D: $(I_{193}, I_{211})$ |
| 3CC (171+193+211) | 3D: $(I_{171}, I_{193}, I_{211})$ |

1024×1024 영상 한 장당 약 $10^6$개 픽셀이 클러스터링 입력이 됩니다.

**핵심 함의**:
1. **공간 정보는 클러스터링에 들어가지 않음** — 픽셀의 위치(좌표)나 이웃 픽셀과의 관계는 무시. CNN의 합성곱처럼 지역 패턴을 학습하지 않습니다.
2. **공간적 일관성은 후처리에서 부여** — 형태학적 opening (작은 점 제거, min size 200 px) + closing (작은 구멍 채움, disk radius 2 px).
3. **2CO (two-channel overlap)** — 193 Å 마스크와 211 Å 마스크에서 둘 다 CH로 분류된 픽셀만 최종 CH로 인정 (보수적 접근).
4. **단순함의 장점** — CPU에서 빠르게 실행되고, 군집 중심(centroid)으로 결과를 해석 가능 (CNN의 black-box 문제 없음).

**English**

Yes. The core method is **pixel-wise k-means**. Each pixel is an independent data point, and the clustering partitions the pixels into $k=3$ clusters in the space spanned by the pixel intensity (or intensities).

The dimensionality of the input vector equals the number of passbands used:

| Configuration | Per-pixel vector |
|---|---|
| 193 Å only | 1D: $(I_{193})$ |
| 211 Å only | 1D: $(I_{211})$ |
| 2CC (193+211) | 2D: $(I_{193}, I_{211})$ |
| 3CC (171+193+211) | 3D: $(I_{171}, I_{193}, I_{211})$ |

A 1024×1024 image yields roughly $10^6$ data points per frame.

**Key implications**:
1. **No spatial information enters the clustering** — the pixel's position and its relation to neighbours are ignored, unlike CNN convolutions which learn local patterns.
2. **Spatial coherence is imposed in postprocessing** — morphological opening (removes small dots, min size 200 px) + closing (fills small holes, disk radius 2 px).
3. **2CO (two-channel overlap)** — only pixels classified as CH in both the 193 Å and 211 Å masks are kept (a conservative criterion).
4. **Advantages of simplicity** — runs fast on a CPU and is interpretable through the cluster centroids, avoiding the CNN black-box problem.
