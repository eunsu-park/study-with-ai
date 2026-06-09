---
title: "Pre-Reading Briefing: Automatic 3D Reconstruction of Coronal Mass Ejections Based on Dual-viewpoint Observations and Machine Learning"
paper_id: "40_lin_2025"
topic: Space_Weather
date: 2026-04-20
type: briefing
---

# Automatic 3D Reconstruction of Coronal Mass Ejections Based on Dual-viewpoint Observations and Machine Learning: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Lin, R., Yang, Y., Shen, F., Pi, G., & Li, Y., "Automatic 3D Reconstruction of Coronal Mass Ejections Based on Dual-viewpoint Observations and Machine Learning", *The Astrophysical Journal Supplement Series*, 280:44 (17pp), 2025 September. DOI: [10.3847/1538-4365/adf433](https://doi.org/10.3847/1538-4365/adf433)
**Authors**: Rongpei Lin (林荣沛, NSSC/CAS), Yi Yang (杨易, NSSC/CAS), Fang Shen (沈芳, NSSC/CAS), Gilbert Pi (Charles Univ. Prague), Yucong Li (李雨淙, MUST Macau)
**Year**: 2025 (Received Apr 18, 2025; Accepted Jul 23, 2025; Published Sep 11, 2025)

---

## 1. 핵심 기여 / Core Contribution

### 한국어
이 논문은 SOHO/LASCO C2 와 STEREO-A/COR2 **이중 시점 코로나그래프 관측**으로부터 코로나 질량 방출(CME)의 **3D 구조를 자동 재구성**하는 통합 알고리즘을 제시한다. 알고리즘은 세 단계로 구성된다: (1) **영역 획득(region acquisition)** — LeNet-5 변형 CNN으로 CME 포함/비포함 영상을 분류, PCA로 colocalization map 생성, Otsu 방법으로 이진화하여 CME 영역 마스크를 얻음; (2) **모델 구성(model construction)** — 6개 기하 파라미터(경도 $\phi$, 위도 $\theta$, 기울기 $\gamma$, half angle $\alpha$, aspect ratio $\kappa$, 높이 $h$)로 GCS(Graduated Cylindrical Shell) 모델을 구축하여 두 코로나그래프 FOV에 투영; (3) **함수 최적화(function optimization)** — 영역 마스크와 모델 투영 사이의 morphological 불일치를 정량화하는 객관 함수를 정의하고 **differential evolution(DE)** 알고리즘으로 최적 파라미터 집합을 도출한다. 4개의 대표 CME 이벤트와 2007-2018년 **97개 CME**의 통계 분석으로 알고리즘 정확도를 입증하며, 2D 단일 시점 측정이 **속도를 평균 8% 과소추정**하고 **폭을 47% 과대추정**함을 정량화한다(halo CME에서 더 극심: 속도 -29%, 폭 +46%). 이는 dual-viewpoint observation에 기반한 **세계 최초의 자동 3D CME 재구성 프레임워크**이며, 향후 MHD 시뮬레이션의 초기 입력값과 ML 기반 우주기상 예보의 표준 데이터를 제공한다.

### English
This paper presents an integrated algorithm that **automatically reconstructs the 3D structure of coronal mass ejections (CMEs)** from **dual-viewpoint coronagraph observations** taken by SOHO/LASCO C2 and STEREO-A/COR2. The algorithm consists of three stages: (1) **Region acquisition** — a LeNet-5-based CNN classifies images as CME/no-CME, then PCA-derived colocalization maps refined by Otsu's method yield binary CME region masks; (2) **Model construction** — six geometric parameters (longitude $\phi$, latitude $\theta$, tilt $\gamma$, half angle $\alpha$, aspect ratio $\kappa$, height $h$) define a GCS (Graduated Cylindrical Shell) model whose 3D shell is projected onto the FOVs of both coronagraphs; (3) **Function optimization** — an objective function quantifying the morphological discrepancy between region masks and model projections is maximised by the **differential evolution (DE)** algorithm to derive the optimal parameter set. Four representative CME events plus a statistical study of **97 CMEs (2007-2018)** demonstrate the algorithm's accuracy, quantifying that 2D single-viewpoint measurements **underestimate velocity by 8% on average** and **overestimate width by 47%** (more severely for halo CMEs: -29% in velocity and +46% in width). This is the **first automatic 3D CME reconstruction framework based on dual-viewpoint observations**, providing standard inputs for both MHD simulations and ML-based space-weather forecasting.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

#### 한국어
CME는 1970년대 OSO-7와 Skylab 코로나그래프로 처음 인식된 이래, **우주기상의 가장 큰 구동자**로 자리잡았다. 단일 시점 LASCO 관측(1995–현재)에 의존하는 카탈로그(CDAW SOHO/LASCO; Yashiro 2004)는 시간이 많이 들고, halo CME의 경우 **투영 효과(projection effect)** 로 폭과 속도 측정이 크게 왜곡된다. STEREO 미션(2006 발사; Kaiser 2005)은 SOHO와 함께 **이중 시점 관측**을 가능하게 했고, 이를 활용한 3D 재구성 방법들 — polarization ratio (Moran & Davila 2004), triangulation (Liewer+2007, 2010; Mierla+2008, 2009, 2010), forward modeling — 이 발전했다. 그중 **GCS 모델** (Thernisien+2006, 2009; Thernisien 2011)은 flux rope 형태의 CME를 6개 파라미터로 표현하여 가장 널리 쓰이는 forward modeling 방법이 되었다.

그러나 GCS 피팅은 **수동 작업**이었다: 사용자가 6개 파라미터를 일일이 조정하여 두 시점의 영상에 wire frame을 맞춰야 했다. Bosman+2012는 1060개 CME, Shen+2013/2014는 86개 halo CME, Kay & Gopalswamy 2017은 45개 Earth-directed CME에 GCS를 수동 적용했다. Kay & Palmerio 2024는 24개 다른 카탈로그 비교에서 **수동 개입이 ~27% (폭), 29% (aspect ratio), 19% (속도) 의 체계적 차이**를 만든다고 보고했다.

한편 ML 기반 CME 탐지·추적 연구들이 등장: Wang+2019a (CNN 영역 검출), Alshehhi & Marpu 2021 (VGG-16 + PCA + K-Means), Lin+2024b (track-match), Yang+2025 (CNN+Transformer+Kalman). 그러나 이들은 모두 **2D 영역 검출**에 머물렀으며 **자동 3D 재구성**은 아직 개척되지 않은 영역이었다. 본 논문은 영역 검출 ML과 GCS 모델 + DE 최적화를 결합하여 그 공백을 메운다.

#### English
Since CMEs were first recognised by OSO-7 and Skylab coronagraphs in the 1970s, they have been established as the **dominant drivers of space weather**. Catalogs based on single-viewpoint LASCO observations (1995–present), such as the CDAW SOHO/LASCO catalog (Yashiro 2004), are time-consuming to compile, and for halo CMEs the **projection effect** strongly distorts the measured widths and velocities. The STEREO mission (launched 2006; Kaiser 2005) enabled **dual-viewpoint observations** with SOHO, and several 3D reconstruction techniques followed: polarization ratio (Moran & Davila 2004), triangulation (Liewer+2007, 2010; Mierla+2008, 2009, 2010), and forward modeling. Among these, the **GCS model** (Thernisien+2006, 2009; Thernisien 2011) — representing flux-rope-like CMEs with six parameters — has become the most widely used forward-modeling method.

However, GCS fitting has historically been a **manual process**: users adjusted the six parameters by hand to match wire-frame projections to both viewpoints. Bosman+2012 catalogued 1060 CMEs, Shen+2013/2014 fit 86 halo CMEs, and Kay & Gopalswamy 2017 reconstructed 45 Earth-directed CMEs — all manually. Kay & Palmerio 2024 reported that **human intervention introduces typical systematic differences of ~27% (width), 29% (aspect ratio), and 19% (velocity)** across catalogs.

In parallel, ML-based CME detection and tracking emerged: Wang+2019a (CNN region detection), Alshehhi & Marpu 2021 (VGG-16 + PCA + K-Means), Lin+2024b (track-match), Yang+2025 (CNN + Transformer + Kalman). However, all of these stopped at **2D region detection** — **automatic 3D reconstruction** remained unexplored. This paper closes that gap by combining region-detection ML with GCS modeling and DE optimisation.

### 타임라인 / Timeline

```
1970s ── OSO-7 / Skylab — first coronagraph CME identifications
            │
1979 ── Otsu — automatic image thresholding method
            │
1995 ── SOHO launched — LASCO C2/C3 coronagraphs (Brueckner+1995)
            │  CDAW SOHO/LASCO catalog (Yashiro 2004) — manual single-view
            │
1997 ── Storn & Price — Differential Evolution optimization algorithm
1998 ── LeCun et al. — LeNet-5 CNN architecture
            │
2004 ── Robbrecht & Berghmans — CACTus (wavelet-based auto CME detection)
2004 ── Moran & Davila — polarization-ratio 3D reconstruction
            │
2006 ── STEREO launched (Kaiser 2005) — twin spacecraft, dual viewpoints
2006 ── Thernisien+ — GCS (Graduated Cylindrical Shell) model introduced
            │
2007–2018 ── Productive STEREO-A/COR2 + LASCO C2 dual-viewpoint era
            │
2011 ── Thernisien — GCS mathematical formalism solidified
2012 ── Bosman+ — 1060-event manual GCS catalog from SECCHI/COR2
            │
2017 ── Kay & Gopalswamy — 45 Earth-directed CMEs with GCS + magnetic profiles
2018 ── Vourlidas+ — dual-viewpoint CME catalog (manual)
            │
2019 ── Wang+ — CNN-based CME region detection (2D only)
2021 ── Alshehhi & Marpu — VGG-16 + PCA + K-Means (2D only)
2024 ── Lin+ (this group) — track-match for 2D consecutive frames
2024 ── Kay & Palmerio — quantifies 19-29% inter-catalog scatter from manual fitting
2025 ── Yang+ — CNN + Transformer + Kalman tracking (2D only)
            │
2025 ── ★ THIS PAPER (Lin et al.) ★
            │  → First fully-automatic 3D CME reconstruction framework
            │  → CNN region detection + GCS model + DE optimization
            │  → 97 CMEs analysed; quantifies 2D bias (8% velocity, 47% width)
            │
202X ── (Future) Real-time pipeline integration with MHD simulators
                  Solar Orbiter / PSP joining as third viewpoint
```

---

## 3. 필요한 배경 지식 / Prerequisites

### 한국어
1. **CME 기본 물리**: 코로나 자속관(flux rope) 구조, 에너지 방출 메커니즘(자기 재결합), CME-flare 관계, halo CME의 정의(태양 디스크를 둘러싸 보이는 CME → 지구 방향 방출 시사)
2. **코로나그래프 영상**: occulting disk, Thomson scattering(자유전자에 의한 백색광 산란), running difference 영상(연속 프레임 빼기 → 변화 강조)
3. **SOHO/LASCO C2 와 STEREO-A/COR2 미션**: L1 vs heliocentric 궤도, FOV 차이, 관측 시점의 angular separation
4. **GCS 모델 (Thernisien 2011)**: croissant 모양 — 두 conical leg + 토러스 front (flux rope 모방). 6개 파라미터:
   - $\phi$ (longitude), $\theta$ (latitude) — 발생 방향
   - $\gamma$ (tilt) — flux rope 축 방향
   - $\alpha$ (half angle) — leg 사이 각도
   - $\kappa$ (aspect ratio) — leg 두께/높이 비, $\kappa = \sin\delta$
   - $h$ (height) — apex까지 거리
5. **CNN 기초**: LeNet-5 구조 (conv + pool + fc), cross-entropy loss, SGD, sigmoid/softmax 분류
6. **PCA (주성분 분석)**: 공분산 행렬 → 고유분해 → 가장 큰 고유값 방향으로 투영. 본 논문에서는 CNN 마지막 conv layer의 feature tensor $I \in \mathbb{R}^{h \times w \times d}$ 에 적용하여 colocalization map 생성
7. **Otsu 방법** (Otsu 1979): 이미지 픽셀을 두 클래스로 나누는 최적 임계값을 intra-class 분산 최대화로 자동 결정
8. **Differential Evolution** (Storn & Price 1997): 진화 알고리즘, 4단계 — 초기화(uniform) → mutation ($v = x_{r_1} + F(x_{r_2} - x_{r_3})$) → crossover (CR) → selection (greedy)
9. **Convex hull** (Sklansky 1982): 점 집합의 외부 경계를 구하는 알고리즘. 본 논문에서는 GCS 투영점들에서 이진 마스크 생성에 사용
10. **Heliographic coordinates**: Stonyhurst (지구중심 고정) vs Carrington (태양 자전 고정), 위도·경도 정의
11. **Halo CME 측정 한계**: CDAW에서 폭이 360°로 마킹되는 문제 (실제 각폭 측정 불가)

### English
1. **Basic CME physics**: coronal flux-rope structures, energy release (magnetic reconnection), CME-flare relationship, halo CME definition (CMEs that appear to surround the solar disk → indicate Earth-directed launches)
2. **Coronagraph imagery**: occulting disk, Thomson scattering (white-light scattering by free electrons), running-difference images (sequential frame subtraction to enhance changes)
3. **SOHO/LASCO C2 and STEREO-A/COR2 missions**: L1 vs heliocentric orbits, FOV differences, the angular separation between viewpoints at observation time
4. **GCS model (Thernisien 2011)**: croissant shape — two conical legs + toroidal front (mimicking a flux rope). Six parameters:
   - $\phi$ (longitude), $\theta$ (latitude) — eruption direction
   - $\gamma$ (tilt) — flux-rope axis orientation
   - $\alpha$ (half angle) — angle between legs
   - $\kappa$ (aspect ratio) — leg-thickness-to-height ratio, $\kappa = \sin\delta$
   - $h$ (height) — distance to apex
5. **CNN basics**: LeNet-5 architecture (conv + pool + fc), cross-entropy loss, SGD optimisation, sigmoid/softmax classification
6. **PCA (Principal Component Analysis)**: covariance matrix → eigendecomposition → projection onto largest-eigenvalue direction. In this paper, applied to the CNN's last conv-layer feature tensor $I \in \mathbb{R}^{h \times w \times d}$ to produce a colocalization map
7. **Otsu's method (Otsu 1979)**: automatic threshold selection that splits image pixels into two classes by maximising inter-class variance
8. **Differential Evolution (Storn & Price 1997)**: evolutionary algorithm, four steps — initialisation (uniform) → mutation ($v = x_{r_1} + F(x_{r_2} - x_{r_3})$) → crossover (CR) → selection (greedy)
9. **Convex hull (Sklansky 1982)**: algorithm for the outer boundary of a point set. Used here to convert projected GCS points into binary masks
10. **Heliographic coordinates**: Stonyhurst (Earth-fixed) vs Carrington (Sun-rotation-fixed), latitude/longitude definitions
11. **Halo CME measurement limits**: CDAW's marking of width as 360° (true angular extent is not directly measurable)

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **CME (Coronal Mass Ejection)** | 코로나로부터 태양권으로 방출되는 자화 플라즈마의 거대 구조. 우주기상 폭풍의 주요 구동자. / Massive ejection of magnetised plasma from the corona into the heliosphere; primary driver of geomagnetic storms. |
| **GCS (Graduated Cylindrical Shell)** | Thernisien+ 2006/2011의 croissant 모양 CME 모델. 두 개의 conical leg + 토러스 형태의 front. / Croissant-shaped CME model with two conical legs + toroidal front, by Thernisien+ 2006/2011. |
| **Flux rope** | 비틀린 자기력선이 형성한 관 구조. CME의 핵심 자기 구조로 여겨짐. / Tube-like structure of twisted magnetic field lines, considered the core magnetic structure of CMEs. |
| **Halo CME** | 태양 디스크를 둘러싸 보이는 CME → 지구 방향(또는 그 반대) 방출. CDAW에서는 폭=360°로 표기. / CME that appears to surround the solar disk → Earth-directed (or anti-Earth). CDAW marks width as 360°. |
| **Running difference** | 코로나그래프 시간연속 영상에서 이전 프레임을 빼서 변화(=전파하는 CME)를 강조. / Subtracts the previous frame from the current to highlight changes (= propagating CME) in time-series coronagraph images. |
| **LASCO C2** | SOHO에 탑재된 코로나그래프, FOV 2.2-6 R_⊙. 1995-현재. / Coronagraph on SOHO, FOV 2.2–6 R_⊙. 1995–present. |
| **STEREO-A/COR2** | STEREO-A 위성의 코로나그래프, FOV 2.5-15 R_⊙. SOHO와 다른 시점. / Coronagraph on STEREO-A, FOV 2.5–15 R_⊙. Observes from a different viewpoint than SOHO. |
| **Colocalization map** | CNN feature tensor를 PCA로 1차원 투영한 강도 맵. CME 위치 정보를 포함. / Intensity map obtained by PCA projection of CNN features; encodes CME location. |
| **Otsu's method** | intra-class variance를 최대화하는 임계값으로 영상을 자동 이진화. / Automatic image binarisation by choosing the threshold that maximises inter-class variance. |
| **Differential Evolution (DE)** | 진화 알고리즘 — 무작위 초기화 + 돌연변이 + 교차 + 선택의 반복. 6차원 비선형 최적화에 적합. / Evolutionary algorithm with random initialisation + mutation + crossover + selection; suited for 6D non-convex optimisation. |
| **Object function $F(p)$** | similarity term − β · area penalty. 모델-관측 일치도와 모델 크기 제약 동시 표현. / similarity term − β · area penalty; jointly enforces fit accuracy and prevents the model from blowing up. |
| **Similarity term** | 영역 마스크와 GCS 투영의 픽셀별 일치 비율 (Eq. 2). / Pixel-wise agreement fraction between the region mask and the GCS projection (Eq. 2). |
| **Area penalty** | GCS 투영이 너무 커지지 않도록 하는 정규화 항 (Eq. 3). / Regularisation that prevents the GCS projection from growing too large (Eq. 3). |
| **β (area-penalty coefficient)** | 0과 1 사이의 가중치. 본 논문에서는 sweep search로 β=0.12 결정. / Weight between 0 and 1; selected by sweep search to be β=0.12 in this paper. |
| **Convex hull** | 점 집합의 외부 경계 (모든 내각 < 180°). 투영점들에서 이진 마스크 생성. / Outer polygon of a point set (all interior angles < 180°); turns the projected points into a binary mask. |
| **Stonyhurst coordinates** | 태양면 위치를 지구 중심 기준 위도·경도로 표현하는 좌표계. / Heliographic coordinates centred on the Earth-Sun line for latitude/longitude. |
| **CDAW catalog** | NASA Goddard의 SOHO/LASCO CME 카탈로그. 수동 식별, 단일 시점. / NASA Goddard's SOHO/LASCO CME catalog; manually identified, single-viewpoint. |

---

## 5. 수식 미리보기 / Equations Preview

### 5.1 객관 함수 / Object function (Eq. 1, 2, 3, 4)

$$F(\mathbf{p}) = \frac{1}{n}\sum_{i=1}^{n} \left[\,\text{simi}(c^i, e^i(\mathbf{p})) - \text{area}(e^i(\mathbf{p}))\,\right]$$

$$\text{simi}(c^i, e^i(\mathbf{p})) = \frac{\sum_{j=1}^{N_i} c_j^i\, e_j^i(\mathbf{p})}{N_i}, \qquad \text{area}(e^i(\mathbf{p})) = \beta \cdot \frac{\sum_{j=1}^{N_i} \text{IF}(e_j^i(\mathbf{p}), 1)}{N_i}$$

$$\text{IF}(x, y) = \begin{cases} 1 & \text{if } x = y \\ 0 & \text{if } x \neq y \end{cases}$$

- $\mathbf{p} = (\phi, \theta, \gamma, \alpha, \kappa, h)$ — 6개 GCS 파라미터 / 6 GCS parameters
- $c_j^i$ — $i$번째 코로나그래프 영상의 $j$번째 픽셀 (region labeling, 0/1) / pixel $j$ of the region labeling of viewpoint $i$
- $e_j^i(\mathbf{p})$ — 파라미터 $\mathbf{p}$로 구성된 GCS 모델을 $i$번째 코로나그래프 FOV에 투영한 마스크의 $j$번째 픽셀 / pixel $j$ of the GCS projection onto viewpoint $i$
- $n=2$ — 코로나그래프 수 (LASCO C2 + STEREO-A/COR2) / number of viewpoints
- $N_i$ — $i$번째 영상의 픽셀 수 / pixel count
- $\beta = 0.12$ — area penalty 가중치 (sweep search로 결정) / area-penalty coefficient (chosen via sweep search)

**해석 / Interpretation**: similarity는 두 마스크의 픽셀 단위 일치를 +방향으로, area는 모델 마스크가 너무 커지는 것에 −방향으로 페널티. β=0이면 모델이 무한히 커지고, β=1이면 모델이 점으로 수축한다(Fig. 4 참고). β=0.12는 두 항이 적절히 균형을 이루는 값. / Similarity rewards pixel-wise mask agreement; area penalises model masks that grow too large. β=0 lets the model blow up; β=1 collapses it to a point (see Fig. 4). β=0.12 balances the two terms.

### 5.2 GCS 폭 공식 / GCS width formula

$$w = 2(\alpha + \delta), \qquad \delta = \arcsin \kappa$$

- $w$ — CME의 3D 각폭 / 3D angular width
- $\alpha$ — half angle (두 leg 사이의 각도의 절반) / half angle
- $\kappa$ — aspect ratio
- $\delta$ — leg의 중심축과 옆면 사이의 각도 / angle between a leg's central axis and its side

이 공식은 leg의 두께($\kappa$를 통해)를 포함한 **물리적 각폭**을 정의한다. half angle만으로는 leg가 얇은 막대로 모델링되어 폭을 과소평가한다. / This formula defines the **physical angular width** including the leg thickness (through $\kappa$). Using only the half angle would underestimate the width by treating the legs as infinitely thin.

### 5.3 Differential Evolution mutation / DE 돌연변이 (Eq. 7, 8, 9)

$$v_{i, G} = x_{r_1, G} + F \cdot (x_{r_2, G} - x_{r_3, G})$$

$$u_{j, i, G} = \begin{cases} v_{j, i, G} & \text{if } \text{rand}_j \le CR \text{ or } \text{rnbr}(i) = j \\ x_{j, i, G} & \text{otherwise} \end{cases}$$

- $G$ — 세대 인덱스 / generation index
- $i$ — 솔루션 인덱스 ($i = 1, \ldots, N_p$) / solution index
- $r_1, r_2, r_3$ — 무작위로 선택된 서로 다른 솔루션 인덱스 / three distinct random solution indices
- $F \in (0, 2]$ — scaling factor (mutation 강도) / scaling factor controlling mutation strength
- $CR \in [0, 1]$ — crossover rate / crossover probability
- $\text{rnbr}(i)$ — $[1, D]$ 범위의 무작위 정수 (적어도 한 차원은 mutated 보장) / random integer ensuring at least one mutated dimension

### 5.4 DE selection (greedy) / DE 선택 (탐욕)

$$x_{i, G+1} = \begin{cases} u_{i, G} & \text{if } F(u_{i, G}) \ge F(x_{i, G}) \\ x_{i, G} & \text{otherwise} \end{cases}$$

- 자식이 부모보다 객관 함수 값이 같거나 크면 교체, 아니면 부모 유지 / replace parent only if the child has equal or higher objective-function value
- 단순한 ($\mu, \lambda$) 진화 전략으로, **항상 최적해의 비악화** 보장 / a simple ($\mu, \lambda$) evolution strategy that **guarantees non-deterioration** of the best solution

### 5.5 PCA dimension reduction / PCA 차원 축소

CNN의 마지막 conv layer에서 feature tensor $I \in \mathbb{R}^{h \times w \times d}$를 얻은 후: / Given the feature tensor $I \in \mathbb{R}^{h \times w \times d}$ from the last conv layer:

1. **공분산 행렬 계산** / Compute covariance matrix: $\Sigma = \frac{1}{hw}\sum (I_{ij} - \bar{I})(I_{ij} - \bar{I})^\top \in \mathbb{R}^{d \times d}$
2. **고유분해** / Eigendecomposition: $\Sigma = V \Lambda V^\top$
3. **최대 고유값 방향으로 투영** / Project onto largest-eigenvalue direction: $M = I \cdot v_1$, $M \in \mathbb{R}^{h \times w}$
4. **결과**: colocalization map (값의 부호 = 픽셀 상관의 방향, 크기 = 상관의 강도)

이 맵은 [0, 255] 범위로 정규화 후 **Otsu 방법**으로 이진화하여 CME 영역 마스크 $c$를 얻는다. / The map is normalised to [0, 255] and binarised by **Otsu's method** to yield the CME region mask $c$.

---

## 6. 읽기 가이드 / Reading Guide

### 한국어 권장 읽기 순서

1. **§1 Introduction (pp. 1-2)** — 천천히 정독.
   - CME의 우주기상 영향 이해
   - 기존 카탈로그(CDAW, CACTus, SEEDS 등)와 자동화 시도들의 한계 파악
   - 본 논문이 채우는 공백("first automatic 3D CME reconstruction") 확인
2. **§2 Data (p. 3)** — 빠르게 통독.
   - SOHO/LASCO C2 + STEREO-A/COR2, 2013-2018 (12,230 images, 6142 with CME, 6088 without)
   - 전처리: Gaussian filter → running difference → 2048² → 512² rescale
3. **§3.1 Region Acquisition (pp. 3-5)** — 정독, **Figure 2 (workflow) 함께**.
   - LeNet-5 변형 CNN (분류) → PCA (colocalization map) → Otsu (이진화)
   - **Figure 2**: 영역 획득의 시각적 흐름
4. **§3.2 Model Construction (pp. 5-6)** — 정독, **Figure 3 (GCS schematic) 와 함께**.
   - 6개 파라미터의 기하학적 의미
   - 폭 공식 $w = 2(\alpha + \delta)$ 도출
   - convex hull로 이진 마스크 생성
5. **§3.3 Function Optimization (pp. 6-7)** — **수학적 핵심**, 천천히.
   - **§3.3.1 Object Function**: similarity term + area penalty term의 직관 (Fig. 4의 β=0 vs β=1 비교가 핵심)
   - **§3.3.2 Optimization Algorithm**: DE의 4단계 절차 (Eq. 6-10)
6. **§4 Results (pp. 8-15)** — 사례별로 빠르게.
   - **§4.1.1-4.1.4**: 4개 대표 CME 이벤트 (Figs. 5-8, Tables 1-2)
     - 2012-08-02, 2012-01-08, 2012-11-09, 2011-10-01 (halo)
   - **§4.2 Statistics on 97 CMEs**:
     - **§4.2.1** width-velocity 분포
     - **§4.2.2** halo vs non-halo 비교
     - **§4.2.3** 2D vs 3D 차이 정량화 ⭐ (핵심 결과)
7. **§5 Conclusion** — 통독.

### English Recommended Reading Order

1. **§1 Introduction (pp. 1-2)** — read carefully.
   - Understand CMEs' impact on space weather.
   - Map the limitations of prior catalogs (CDAW, CACTus, SEEDS, etc.) and automation efforts.
   - Identify the gap this paper closes ("first automatic 3D CME reconstruction").
2. **§2 Data (p. 3)** — skim quickly.
   - SOHO/LASCO C2 + STEREO-A/COR2, 2013-2018 (12 230 images, 6142 with CME, 6088 without).
   - Preprocessing: Gaussian filter → running difference → 2048² → 512² rescale.
3. **§3.1 Region Acquisition (pp. 3-5)** — read carefully **with Figure 2**.
   - LeNet-5-based CNN (classifier) → PCA (colocalization map) → Otsu (binarisation).
   - **Figure 2** visualises the region-acquisition flow.
4. **§3.2 Model Construction (pp. 5-6)** — read carefully **with Figure 3** (GCS schematic).
   - Geometric meaning of the six parameters.
   - Derivation of the width formula $w = 2(\alpha + \delta)$.
   - Convex hull → binary mask.
5. **§3.3 Function Optimization (pp. 6-7)** — **mathematical core**, take it slow.
   - **§3.3.1 Object Function**: intuition for the similarity + area-penalty terms (Figure 4's β=0 vs β=1 comparison is essential).
   - **§3.3.2 Optimization Algorithm**: the four DE steps (Eqs. 6-10).
6. **§4 Results (pp. 8-15)** — case by case, fast.
   - **§4.1.1-4.1.4**: four representative CME events (Figs. 5-8, Tables 1-2):
     2012-08-02, 2012-01-08, 2012-11-09, 2011-10-01 (halo).
   - **§4.2 Statistics on 97 CMEs**:
     - **§4.2.1** width-velocity distribution
     - **§4.2.2** halo vs non-halo comparison
     - **§4.2.3** quantification of 2D vs 3D differences ⭐ (headline result)
7. **§5 Conclusion** — skim through.

### 읽으면서 메모할 질문 / Questions to keep in mind

- 한국어
  1. β=0.12를 0.10 간격 sweep search로 결정했다 — 더 정밀한 탐색(예: Bayesian)으로 더 좋은 값을 찾을 수 있는가?
  2. 2012-01-08 사례에서 CDAW 속도(557)가 3D 속도(696)보다 작은데, 본 논문은 이를 "lateral expansion velocity > radial velocity"로 설명한다 (Gopalswamy 2009; Shen 2013; Majumdar 2020). 이 설명이 모든 halo CME에 일반화되는가?
  3. CNN을 SOHO와 STEREO 영상에 대해 **별도로 학습**시켰다 — 단일 통합 모델이 더 좋을지, 아니면 채널/스펙트럼 차이 때문에 분리가 합리적인지?
  4. 폭 +47%, 속도 -8%의 통계는 2007-2018 데이터에서 도출 — 2018 이후 STEREO-A의 위치 변화(태양 뒤로 이동, 통신 두절기) 이후 다시 가능한 dual-view 시기에도 같은 수치가 나올까?
  5. GCS 모델은 flux rope를 가정 — flux rope가 발달하지 않은 "stealth CME"나 jet-like CME에는 적용 가능한가?
  6. Halo CME의 폭 측정은 단일 시점에서 본질적으로 불가능 — 본 논문의 3D 폭 추정은 신뢰할 만한 ground truth가 없는데 어떻게 검증되는가?

- English
  1. β=0.12 was found by a 0.10-interval sweep — could a finer search (e.g. Bayesian) find a better value?
  2. For 2012-01-08, the CDAW velocity (557) is smaller than the 3D velocity (696); the paper attributes this to "lateral-expansion velocity > radial velocity" (Gopalswamy 2009; Shen 2013; Majumdar 2020). Does this explanation generalise to all halo CMEs?
  3. The CNN was trained **separately** on SOHO and STEREO images — would a single unified model be better, or does the inter-instrument difference justify separation?
  4. The +47% width / −8% velocity statistics come from 2007-2018 data — does the same hold after STEREO-A's recent re-emergence following the gap caused by its solar-conjunction passage?
  5. The GCS model assumes a flux rope — is the algorithm applicable to "stealth CMEs" or jet-like CMEs that lack a developed flux rope?
  6. Halo-CME width is fundamentally unobservable from a single viewpoint — how is the paper's 3D width estimate validated when there is no reliable ground truth?

---

## 7. 현대적 의의 / Modern Significance

### 한국어
1. **자동화의 의미**: GCS 피팅이 수동에서 자동으로 넘어가는 것은 단순 편의가 아니다 — Kay & Palmerio 2024가 보고한 19-29%의 카탈로그 간 수동 편차를 제거하여, **실시간 우주기상 운영 파이프라인**에 통합 가능한 일관성 있는 3D CME 카탈로그를 가능케 한다.
2. **MHD 시뮬레이션과의 결합**: ENLIL, EUHFORIA, SUSANOO 같은 행성간 MHD 모델은 정확한 CME 초기 조건(launch direction, velocity, width)을 필요로 한다. 본 논문의 자동 3D 파라미터는 이런 모델의 **표준 입력**이 될 수 있다.
3. **ML 기반 우주기상 예보의 입력 데이터**: 도착 시각 예측(Sudar+2016, Wang+2019b, Liu+2018, Li+2024) 모델들은 모두 단일 시점 카탈로그에 의존했다. 본 논문이 47% 과대된 폭과 8% 과소된 속도를 보정한 ground truth를 제공하면, 이런 ML 모델의 **체계적 편향이 정량적으로 줄어들 수 있다**.
4. **Halo CME 평가의 정량화**: halo CME에서 CDAW 폭 = 360°는 정보가 없는 수치. 본 논문은 -29% 속도, +46% 폭의 체계적 편향을 정량화하여, halo CME에 대한 **새로운 보정 인자(correction factor)** 를 제시한다.
5. **Solar Orbiter / Parker Solar Probe / Vigil 시대로의 확장**: STEREO-A의 위치 제약(2018년 이후 통신 두절기 동안 dual-view 불가)이 있었지만, 2020년대 새로운 미션들로 다중 시점 관측이 다시 가능. 본 논문의 알고리즘은 **3-view 이상으로 확장 가능**하다 (similarity term의 합을 늘리기만 하면 됨).
6. **Space_Weather 트랙에서의 위치**: Phase 4 (CME-storm relationships) 의 후속이자 Phase 5 (ML-based forecasting) 의 데이터 인프라 역할. 동시에 Solar_Observation의 Phase 7 (data processing) 과도 연결.

### English
1. **What automation actually buys**: moving GCS fitting from manual to automatic is more than convenience — it eliminates the 19-29% inter-catalog scatter from human intervention reported by Kay & Palmerio (2024), enabling a **consistent 3D CME catalog suitable for real-time space-weather operations**.
2. **Coupling to MHD simulations**: interplanetary MHD models such as ENLIL, EUHFORIA, and SUSANOO require accurate CME initial conditions (launch direction, velocity, width). This paper's automatic 3D parameters are a candidate **standard input** for such models.
3. **Better inputs for ML-based forecasting**: arrival-time predictors (Sudar+2016, Wang+2019b, Liu+2018, Li+2024) have all depended on single-viewpoint catalogs. With ground truth corrected for the +47% width / −8% velocity bias from this paper, the **systematic biases of those ML models can be quantitatively reduced**.
4. **Quantitative assessment of halo CMEs**: CDAW's 360° width for halo CMEs carries little physical information. This paper quantifies a systematic bias of −29% in velocity and +46% in width, providing a new **correction factor** for halo events.
5. **Extension into the Solar Orbiter / PSP / Vigil era**: STEREO-A's geometric limitations (no dual-view during its solar-conjunction outage after 2018) interrupted multi-viewpoint analyses, but the new missions of the 2020s restore them. The algorithm here is **directly extensible to three or more viewpoints** — one need only sum more similarity terms.
6. **Place in the Space_Weather track**: a successor to Phase 4 (CME-storm relationships) and a data-infrastructure piece for Phase 5 (ML-based forecasting). It also bridges into Solar_Observation Phase 7 (data processing).

---

## Q&A

### Q1. 단일 시점에서 코로나그래프 데이터를 보면 여러 해가 나올 수 밖에 없잖아? 이것을 어떻게 해결할까? / Single-view coronagraph data is degenerate — how is the ambiguity resolved?

**한국어**

이것이 dual-viewpoint 관측이 필요한 **근본적 이유**다. 단일 코로나그래프는 3D CME를 **plane-of-sky(POS)에 투영한 2D 영상**만 제공하므로, 다음의 정보 손실이 발생한다:

| 측정 가능 (POS) | 측정 불가 (LOS 방향) |
|---|---|
| Position angle (디스크 중심 기준 방향) | Heliocentric 경도/위도 (LOS와의 각도 $\Theta$) |
| 겉보기 폭 $w_{2D}$ | 진짜 3D 폭 $w_{3D}$ |
| 겉보기 속도 $v_{2D}$ (POS 성분) | 진짜 3D 속도 $v_{3D}$ |
| 시간 변화 (running difference) | 깊이 방향 구조 |

투영 관계식 (개략):

$$w_{2D} \approx f(w_{3D},\ \Theta), \qquad v_{2D} \approx v_{3D}\cdot \sin\Theta$$

→ **하나의 $(w_{2D}, v_{2D})$에 대해 무수히 많은 $(w_{3D}, v_{3D}, \Theta)$ 조합이 가능**하다. 특히 halo CME ($\Theta \approx 0$, 관측자 방향)는 $\sin\Theta \approx 0$이므로 속도가 거의 측정 불가하고, 폭은 CDAW에서 360°로 marking된다 (정보 없음).

**해결 방법들**:

1. **Dual-viewpoint forward modeling (본 논문의 접근)** — 두 시점 $i = 1, 2$에서 동일한 3D 모델을 투영하여 각 관측에 동시에 fit. 단일 시점: 제약 < 자유도 → **degenerate**. 두 시점: 제약 ≥ 6 자유도 → **유일해**. STEREO-A의 SOHO와의 angular separation이 클수록 (~90°에서 최적) 삼각측량 기하가 좋아진다.
2. **Triangulation** (Liewer+2007; Mierla+2008) — 두 시점에서 동일 feature 식별 → stereo geometry로 3D 위치 직접 계산. 모델 가정 없음, 단 feature 매칭이 어려움.
3. **Polarization ratio** (Moran & Davila 2004) — Thomson scattering의 편광 정도가 산란 평면과 LOS 사이 각도에 의존 → 단일 시점에서도 산란체와 POS 사이 거리를 부분 추정.
4. **다른 물리 채널과 결합** — EUV로 발생 위치 제약, in-situ 측정으로 backward 검증, Type II radio burst로 shock 속도 제약, IPS로 LOS 밀도 구조.

**본 논문의 한계**: halo CME 검증에는 ground truth가 없어 CDAW 비교만 가능 (진짜 정확도는 in-situ 도착으로만 검증). STEREO-A의 위치 의존(2014-2016 통신 두절기)으로 그 시기는 사용 불가.

**English**

This is the **fundamental reason** that dual-viewpoint observation is needed. A single coronagraph delivers only the **2D projection of the 3D CME onto the plane of sky (POS)**, with the following information loss:

| Measurable (POS) | Not measurable (LOS direction) |
|---|---|
| Position angle (direction relative to disk centre) | Heliocentric longitude/latitude (angle $\Theta$ to LOS) |
| Apparent width $w_{2D}$ | True 3D width $w_{3D}$ |
| Apparent velocity $v_{2D}$ (POS component) | True 3D velocity $v_{3D}$ |
| Temporal evolution (running difference) | Depth-direction structure |

Approximate projection relations:

$$w_{2D} \approx f(w_{3D},\ \Theta), \qquad v_{2D} \approx v_{3D}\cdot \sin\Theta$$

→ **For one $(w_{2D}, v_{2D})$ pair, infinitely many $(w_{3D}, v_{3D}, \Theta)$ triples are possible**. For halo CMEs ($\Theta \approx 0$, directed at the observer), $\sin\Theta \approx 0$ so velocity is essentially unmeasurable and CDAW marks the width as 360° (no real information).

**Resolution approaches**:

1. **Dual-viewpoint forward modeling (this paper)** — project the same 3D model onto two viewpoints $i=1,2$ and fit both simultaneously. Single view: constraints < degrees of freedom → **degenerate**. Two views: constraints ≥ 6 → **unique** (or near-unique). The larger STEREO-A's angular separation from SOHO (optimal ~90°), the better the triangulation geometry.
2. **Triangulation** (Liewer+2007; Mierla+2008) — identify the same feature in both views; use stereo geometry to compute 3D position directly. Model-agnostic, but feature matching is hard.
3. **Polarization ratio** (Moran & Davila 2004) — the degree of polarization in Thomson-scattered light depends on the angle between scattering plane and LOS, partially constraining the distance from POS even with a single viewpoint.
4. **Combination with other physical channels** — use EUV imagery to fix the launch location, in-situ measurements at 1 AU for backward validation, Type II radio bursts for shock speeds, IPS for LOS density structure.

**This paper's limitation**: for halo-CME validation, no ground truth exists — comparison with CDAW only (true accuracy can only be verified by in-situ arrival). The method is unavailable when STEREO-A is geometrically unfavourable (e.g., the 2014-2016 conjunction outage).

### Q2. 딥러닝 모델의 출력은, GCS 모델의 입력 파라미터인 거지? / Is the deep-learning model's output the input parameters of the GCS model?

**한국어**

**아니다.** 이것은 중요하게 구분해야 하는 부분이다. 본 논문의 데이터 흐름은 다음과 같다:

```
코로나그래프 영상 (두 시점)
        │
        ▼
┌─────────────────────────┐
│ CNN (LeNet-5 변형)       │  ← 학습됨 (SOHO / STEREO 각각 별도)
│ 출력: CME 포함/비포함     │  ※ 파라미터 출력 아님!
└─────────────────────────┘
        │ (마지막 conv layer의 feature tensor)
        ▼
┌─────────────────────────┐
│ PCA → Otsu              │  ← 학습 없음, 선형 대수 + 통계
│ 출력: CME 영역 마스크 c^i │  (0/1 binary mask, 아직 3D 아님)
└─────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────┐
│ Differential Evolution 최적화            │  ← 학습 없음, 진화 알고리즘
│ F(p) = simi − β·area 최대화               │
│ 출력: 6 GCS 파라미터 (φ, θ, γ, α, κ, h)   │  ★ 여기서 나옴
└──────────────────────────────────────────┘
        │
        ▼ 3D CME 구조 (GCS croissant)
```

**핵심 구분**:

| 역할 | 산출 | 방법 |
|---|---|---|
| **CNN + PCA + Otsu** | 2D CME 영역 마스크 (어느 픽셀이 CME인가?) | 학습된 영상 분할 |
| **DE 최적화** | 6 GCS 파라미터 (어떤 3D 모양이 그 마스크와 맞나?) | 비학습 evolutionary search |

- CNN은 "이 픽셀이 CME인가?"만 판단한다. 3D 구조나 파라미터를 예측하지 않는다.
- GCS 파라미터 6개는 DE가 iteratively 탐색한다. 각 후보 파라미터 $\mathbf{p}$로부터 GCS 모델을 생성 → 두 시점 FOV에 투영($e^i(\mathbf{p})$) → CNN이 추출한 마스크 $c^i$와 비교(similarity) → 더 좋은 방향으로 mutate.

**왜 end-to-end CNN으로 파라미터를 직접 예측하지 않나?**

1. **학습 데이터 부족**: GCS로 피팅된 "정답 파라미터" 데이터셋이 소규모 (Bosman+2012의 1060개 정도). CNN 회귀 학습에는 보통 훨씬 많은 라벨이 필요
2. **해석 가능성**: 중간 산출물(2D 마스크)이 시각적으로 검증 가능. 잘못되면 어디가 문제인지 파악 가능
3. **모듈성**: GCS 외 다른 parametric 모델로 교체 시 CNN 재학습 불필요
4. **수학적 정당성**: GCS 투영은 엄밀한 기하 변환(convex hull) → ML의 근사로 바꿀 이유 없음

**English**

**No.** This is an important distinction. The data flow of the paper is:

```
Coronagraph images (two viewpoints)
        │
        ▼
┌─────────────────────────┐
│ CNN (LeNet-5 variant)   │  ← trained (separately for SOHO / STEREO)
│ Output: CME yes/no      │  ※ not parameters!
└─────────────────────────┘
        │ (feature tensor from last conv layer)
        ▼
┌─────────────────────────┐
│ PCA → Otsu              │  ← no training, linear algebra + statistics
│ Output: CME region mask c^i │  (0/1 binary mask, not yet 3D)
└─────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│ Differential Evolution optimisation     │  ← no training, evolutionary search
│ Maximise F(p) = simi − β·area           │
│ Output: 6 GCS parameters (φ, θ, γ, α, κ, h) │  ★ here
└─────────────────────────────────────────┘
        │
        ▼ 3D CME structure (GCS croissant)
```

**Key distinction**:

| Role | Product | Method |
|---|---|---|
| **CNN + PCA + Otsu** | 2D CME region mask (which pixels are CME?) | Learned image segmentation |
| **DE optimisation** | 6 GCS parameters (which 3D shape matches those masks?) | Non-learned evolutionary search |

- The CNN only decides "is this pixel CME?" — it does not predict any 3D structure or parameters.
- The 6 GCS parameters are found by DE iteratively: for each candidate $\mathbf{p}$, generate a GCS model → project onto both FOVs $e^i(\mathbf{p})$ → compare with the CNN-derived mask $c^i$ (similarity) → mutate toward better fits.

**Why not predict the parameters end-to-end with a CNN?**

1. **Scarcity of training labels**: the set of CMEs with "ground-truth" GCS parameters is small (e.g., Bosman+2012 with ~1060 events). CNN regression typically needs far more.
2. **Interpretability**: the intermediate product (2D mask) is visually verifiable; if something goes wrong, the failure point is visible.
3. **Modularity**: the CNN does not need retraining if the GCS is replaced by another parametric model.
4. **Mathematical rigor**: the GCS projection is an exact geometric operation (convex hull) — there is no reason to replace it with an ML approximation.

### Q3. CNN으로 하여금 GCS 6 파라미터를 출력하게끔 하는 것은 어떤 어려움이 있을까? / What are the difficulties of training a CNN to directly output the 6 GCS parameters?

**한국어**

이론적으로 가능하지만, 다음과 같은 다층적 어려움이 겹친다:

**1. 데이터 문제**

- (a) **학습 라벨 부족**: 모든 GCS-피팅 카탈로그를 합쳐도 ~수천 개 (Bosman+2012의 1060개가 최대). 6차원 회귀를 위한 일반적 딥러닝 요구량은 $O(10^4) \sim O(10^5)$이며, **6차원 공간을 채우려면 $O(10^6)$** 필요한데 실제 데이터는 $O(10^3)$ 규모.
- (b) **라벨 노이즈**: 기존 GCS 카탈로그는 모두 수동 피팅 → Kay & Palmerio 2024는 19-29% 카탈로그 간 차이 보고. "ground truth"가 사람마다 다른 noisy label → 회귀 모델 분산 증가.

**2. 수학적/구조적 문제**

- (c) **주기적 파라미터**: $\phi$ (longitude)는 $[0°, 360°)$의 원형 변수 — $\phi=359°$와 $\phi=1°$는 실제 $2°$ 차이지만 MSE는 $358°$로 계산. $\sin\phi, \cos\phi$ 인코딩이나 von Mises loss 필요.
- (d) **파라미터 간 스케일 차이**: $\phi, \theta \in [-180°, 180°]$ vs $\kappa \in [0, 1]$ vs $h \in [1, 30] R_\odot$. 각 파라미터의 작은 변화가 결과에 주는 영향이 매우 다름.
- (e) **Loss 설계의 근본 문제**: parameter-space MSE는 비선형성으로 부적절. image-space loss(IoU)는 더 합리적이지만 **GCS 투영을 differentiable하게 구현**해야 함 (convex hull은 미분 불가).
- (f) **다중 모드 해**: 단일 관측에 여러 파라미터 조합이 동등하게 잘 fit. CNN 회귀는 point estimate → 여러 해의 **평균값(=비물리적 중간 값)** 을 출력하기 쉬움. DE는 explicit하게 여러 후보 탐색.
- (g) **우주선 기하 의존성**: STEREO-A와 SOHO의 angular separation은 매일 변함. CNN에 우주선 위치를 추가 입력으로 줘야 학습 복잡성 폭증. DE는 projection 함수가 우주선 위치를 자동 처리.

**3. 일반화 문제**

- (h) **OOD 일반화**: 학습 시 보지 못한 STEREO 위치, 새로운 코로나그래프(Solar Orbiter Metis, PSP WISPR)에서 성능 급락. Forward modeling은 새 시점 추가가 자명.
- (i) **물리 제약**: $\kappa \in (0, 1)$, $\alpha \in (0°, 90°)$, $h > 1 R_\odot$ 등의 경계를 CNN free regression은 자연스레 보장 못함.
- (j) **Halo CME 엣지 케이스**: 본질적으로 ambiguous → 학습 데이터 더 적음 → 소수 모드에서 과적합 가능.

**4. 운영/디버깅 문제**

- (k) **해석 불가능**: end-to-end 실패 시 어디가 문제인지 불명. 모듈형은 각 단계 별도 검증 가능.
- (l) **적응적 계산**: DE는 어려운 사례에 더 많은 generation 할애 가능. CNN inference는 fixed-time.

**시도된 적이 있나?** 있다. (1) **Synthetic training data** (MHD 시뮬레이션 합성 영상)로 라벨 부족 해결, 단 sim-to-real gap; (2) **Weak supervision** (영역 마스크 + 일부 파라미터 라벨); (3) **Differentiable rendering + CNN** — 가장 유망하지만 구현 복잡. 본 논문은 이런 어려움을 우회하여 **CNN은 잘하는 것(영역 분할)만 시키고, 6D 역문제는 evolutionary search로 해결**하는 실용적 hybrid를 선택했다. Scientific ML의 전형적 패턴.

**English**

It's theoretically possible, but several difficulties stack up:

**1. Data problems**

- (a) **Label scarcity**: even combining all GCS-fitted catalogs gives only ~thousands (max ~1060 from Bosman+2012). Typical deep regression needs $O(10^4) \sim O(10^5)$, and **adequately filling a 6D parameter space needs $O(10^6)$** — actual data is $O(10^3)$.
- (b) **Label noise**: existing GCS catalogs are all manually fit → Kay & Palmerio 2024 reports 19-29% inter-catalog scatter. "Ground truth" varies by person → noisy labels → larger regression variance.

**2. Mathematical/structural problems**

- (c) **Periodic parameters**: $\phi$ (longitude) is circular on $[0°, 360°)$ — $\phi=359°$ and $\phi=1°$ are physically only $2°$ apart but MSE computes $358°$. Needs $\sin\phi/\cos\phi$ encoding or von Mises losses.
- (d) **Parameter scale heterogeneity**: $\phi, \theta \in [-180°, 180°]$ vs $\kappa \in [0, 1]$ vs $h \in [1, 30] R_\odot$. Equal-magnitude perturbations have very different physical effects.
- (e) **Fundamental loss-design problem**: parameter-space MSE is inappropriate due to nonlinearity. Image-space loss (IoU) is more reasonable but requires the **GCS projection to be implemented differentiably** (convex hull is non-differentiable).
- (f) **Multi-modal solutions**: a single observation can be fit equally well by multiple parameter combinations. CNN regression is a point estimate → tends to output the **mean (a non-physical intermediate)**. DE explicitly searches multiple candidates.
- (g) **Spacecraft-geometry dependence**: STEREO-A and SOHO's angular separation changes daily. The CNN needs spacecraft positions as additional inputs → training complexity explodes. DE handles this naturally via the projection function.

**3. Generalization problems**

- (h) **OOD generalization**: performance drops on STEREO positions not seen at training time, or on new coronagraphs (Solar Orbiter Metis, PSP WISPR). Forward modeling adds new viewpoints trivially.
- (i) **Physical constraints**: bounds like $\kappa \in (0, 1)$, $\alpha \in (0°, 90°)$, $h > 1 R_\odot$ are not naturally enforced by CNN free regression.
- (j) **Halo-CME edge cases**: inherently ambiguous → training data is even sparser → susceptible to overfitting on the minority mode.

**4. Operations/debugging problems**

- (k) **No interpretability**: when end-to-end fails, the source of the failure is unknown. Modular pipelines let each stage be validated separately.
- (l) **Adaptive compute**: DE can spend more generations on hard events. CNN inference is fixed-time.

**Has anyone tried?** Yes. (1) **Synthetic training data** from MHD simulations to overcome label scarcity, but with a sim-to-real gap; (2) **Weak supervision** (mask labels + partial parameter labels); (3) **Differentiable rendering + CNN** — most promising but complex. This paper sidesteps all these by **letting the CNN do what it's good at (region segmentation) and solving the 6D inverse problem with evolutionary search** — a typical scientific-ML hybrid pattern.
