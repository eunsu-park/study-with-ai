---
title: "Automatic 3D Reconstruction of Coronal Mass Ejections Based on Dual-viewpoint Observations and Machine Learning"
authors: ["Rongpei Lin", "Yi Yang", "Fang Shen", "Gilbert Pi", "Yucong Li"]
year: 2025
journal: "The Astrophysical Journal Supplement Series, 280:44 (17pp)"
doi: "10.3847/1538-4365/adf433"
topic: Space_Weather
tags: [coronal_mass_ejections, GCS_model, dual_viewpoint, SOHO_LASCO, STEREO_COR2, CNN, PCA, Otsu, differential_evolution, 3D_reconstruction, projection_effect]
status: completed
date_started: 2026-04-20
date_completed: 2026-04-20
---

# 40. Automatic 3D Reconstruction of Coronal Mass Ejections Based on Dual-viewpoint Observations and Machine Learning / 이중 시점 관측과 머신러닝에 기반한 코로나 질량 방출의 자동 3D 재구성

---

## 1. Core Contribution / 핵심 기여

### English
This paper presents the **first fully-automatic algorithm for 3D reconstruction of coronal mass ejections (CMEs)** using dual-viewpoint coronagraph observations from SOHO/LASCO C2 and STEREO-A/COR2. The algorithm decomposes the inverse problem into three modular stages: (1) **region acquisition** — a LeNet-5-based CNN classifies each image as containing a CME or not, then PCA projects the last conv-layer feature tensor onto a single colocalization map, which is binarised by Otsu's method to yield the pixel-level CME region mask $c^i$ for each viewpoint; (2) **model construction** — six geometric parameters of the GCS (Graduated Cylindrical Shell) model (longitude $\phi$, latitude $\theta$, tilt $\gamma$, half angle $\alpha$, aspect ratio $\kappa$, height $h$) define the 3D croissant whose surface points are projected onto each coronagraph's FOV and converted to a binary mask $e^i(\mathbf{p})$ via convex hull; (3) **function optimization** — an objective function $F(\mathbf{p}) = \frac{1}{n}\sum_i [\text{simi}(c^i, e^i(\mathbf{p})) - \beta\cdot\text{area}(e^i(\mathbf{p}))]$ with $\beta = 0.12$ (chosen by sweep search over $[0.1, 0.4]$) is maximised by the **Differential Evolution** algorithm. Four representative CMEs (2012-08-02, 2012-01-08, 2012-11-09, 2011-10-01 halo) and statistical analysis of **97 CME events** spanning the rising and declining phases of solar cycle 24 (2007-2018) demonstrate the algorithm's accuracy. The headline statistics: **3D velocities are on average 8% higher** than 2D CDAW velocities (non-halo: 596 vs 555 km/s; halo: 1211 vs 1111 km/s), and **2D widths overestimate 3D widths by ~47%** on average for non-halo events (CDAW avg 130° vs reconstructed avg 69°), with halo widths shrinking from CDAW's nominal 360° to a real average of 118° (range 52°-153°). The width-velocity correlation coefficient drops from 0.67 (2D) to 0.52 (3D) but the linear-fit slope steepens from 2.24 to 8.06 km/s/deg, confirming that wider CMEs are genuinely faster after projection effects are removed. The paper's broader contribution is providing **a reproducible automatic pipeline that eliminates the 19-29% inter-catalog scatter introduced by manual GCS fitting** (Kay & Palmerio 2024), making 3D CME parameters available as standard inputs for MHD simulations (ENLIL, EUHFORIA) and ML-based arrival-time forecasting.

### 한국어
본 논문은 SOHO/LASCO C2와 STEREO-A/COR2의 이중 시점 코로나그래프 관측을 사용하여 코로나 질량 방출(CME)을 **완전히 자동으로 3D 재구성하는 최초의 알고리즘**을 제시한다. 알고리즘은 역문제를 세 개의 모듈형 단계로 분해한다: (1) **영역 획득(region acquisition)** — LeNet-5 변형 CNN이 각 영상을 CME 포함/비포함으로 분류하고, PCA가 마지막 conv layer의 feature tensor를 하나의 colocalization map으로 투영하며, Otsu 방법이 이를 이진화하여 각 시점의 픽셀-단위 CME 영역 마스크 $c^i$를 생성; (2) **모델 구성(model construction)** — GCS(Graduated Cylindrical Shell) 모델의 6개 기하 파라미터(경도 $\phi$, 위도 $\theta$, 기울기 $\gamma$, half angle $\alpha$, aspect ratio $\kappa$, 높이 $h$)가 3D croissant 형상을 정의하고, 그 표면점을 각 코로나그래프 FOV에 투영한 후 convex hull로 이진 마스크 $e^i(\mathbf{p})$를 생성; (3) **함수 최적화(function optimization)** — $\beta = 0.12$(sweep search over $[0.1, 0.4]$로 결정)인 객관 함수 $F(\mathbf{p}) = \frac{1}{n}\sum_i [\text{simi}(c^i, e^i(\mathbf{p})) - \beta\cdot\text{area}(e^i(\mathbf{p}))]$를 **Differential Evolution** 알고리즘으로 최대화. 4개의 대표 CME(2012-08-02, 2012-01-08, 2012-11-09, 2011-10-01 halo)와 태양주기 24의 상승기·하강기를 가로지르는 **97개 CME 이벤트**(2007-2018)의 통계 분석으로 알고리즘 정확도를 입증한다. 핵심 통계: **3D 속도가 2D CDAW 속도보다 평균 8% 더 높음**(non-halo: 596 vs 555 km/s; halo: 1211 vs 1111 km/s), **2D 폭이 3D 폭보다 평균 ~47% 과대추정**(non-halo CDAW 평균 130° vs 재구성 평균 69°), halo 폭은 CDAW의 명목 360°에서 실제 평균 118°(범위 52°-153°)로 축소. 폭-속도 상관계수는 0.67(2D)에서 0.52(3D)로 떨어지지만 linear fit 기울기가 2.24에서 8.06 km/s/deg로 가팔라져 투영 효과 제거 후에도 **넓은 CME는 진정으로 더 빠르다는 사실**을 확인. 논문의 더 큰 기여는 **수동 GCS 피팅이 만드는 19-29% 카탈로그 간 분산을 제거하는** 재현 가능한 자동 파이프라인을 제공하여(Kay & Palmerio 2024), MHD 시뮬레이션(ENLIL, EUHFORIA)과 ML 기반 도착 시각 예보의 표준 입력으로 사용할 수 있게 한다.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction & Motivation / 서론 및 동기 (§1)

**English**

CMEs are massive ejections of magnetised plasma from the corona that drive **major space-weather effects**: geomagnetic storms, substorms, ionospheric disturbances, solar energetic particles, and aurorae. They damage satellites, GPS, and power grids (Eastwood+2017; Webb & Howard 2012). Accurate prediction of CME arrival, propagation direction, speed, and width is therefore a foundational space-weather problem.

The CME observation history relevant to this paper:
- **1995 SOHO launch**: LASCO C2/C3 began continuous coronagraph imaging, single-viewpoint.
- **2006 STEREO launch**: twin spacecraft enabled true dual-viewpoint observations of CMEs.
- **CDAW SOHO/LASCO catalog** (Yashiro 2004): the most widely used catalog, but compiled manually, single-view only, and now ~30 years long.
- **Automated 2D CME detection**: CACTus (Robbrecht & Berghmans 2004, wavelet-based), SEEDS (Olmedo+2008), CORIMP (Byrne+2012), the dual-viewpoint catalog of Vourlidas+2017.

The paper highlights three lineages of CME analysis that have grown in parallel but rarely been combined:
1. **2D ML detection**: Wang+2019a (CNN region detection), Alshehhi & Marpu 2021 (VGG-16 + PCA + K-Means), Lin+2024b (track-match), Yang+2025 (CNN + Transformer + Kalman).
2. **3D reconstruction methods**: polarization ratio (Moran & Davila 2004), triangulation (Liewer+2007, 2010; Mierla+2008, 2009, 2010), forward modeling — culminating in the **GCS model** (Thernisien+2006, 2009; Thernisien 2011) which became the dominant tool.
3. **Manual GCS catalogs**: Bosman+2012 (1060 events), Shen+2013/2014 (86 halo CMEs), Kay & Gopalswamy 2017 (45 Earth-directed), Majumdar+2020, Gandhi+2024 (>360 events).

The paper identifies a **gap**: 3D reconstruction has remained largely manual. Kay & Palmerio (2024) compared 24 different catalogs and found typical inter-catalog differences of ~27% (width), 29% (aspect ratio), 19% (velocity) — purely from human variation in GCS fitting. The authors aim to close this gap with the first automated 3D pipeline.

The paper situates itself relative to prior work by the same group: their previous Lin+2024b track-match paper handles 2D CME tracking; this paper extends the same group's expertise into 3D.

**한국어**

CME는 코로나에서 방출되는 거대한 자화 플라즈마 덩어리로 **주요 우주기상 효과**를 일으킨다: 지자기 폭풍, substorm, 전리층 교란, 태양 에너지 입자, 극광. 위성, GPS, 전력망에 손상을 입힌다(Eastwood+2017; Webb & Howard 2012). 따라서 CME의 도착, 전파 방향, 속도, 폭을 정확히 예측하는 것은 우주기상의 기초적 문제다.

본 논문과 관련된 CME 관측 역사:
- **1995 SOHO 발사**: LASCO C2/C3가 연속적 코로나그래프 영상 시작, 단일 시점.
- **2006 STEREO 발사**: 쌍둥이 위성이 진정한 이중 시점 CME 관측 가능케 함.
- **CDAW SOHO/LASCO 카탈로그**(Yashiro 2004): 가장 널리 쓰이는 카탈로그, 그러나 수동 작성, 단일 시점, 현재 ~30년 길이.
- **자동 2D CME 탐지**: CACTus(Robbrecht & Berghmans 2004, 웨이블릿 기반), SEEDS(Olmedo+2008), CORIMP(Byrne+2012), Vourlidas+2017의 이중 시점 카탈로그.

논문은 평행적으로 발전했지만 거의 결합되지 않은 세 개의 CME 분석 계보를 강조한다:
1. **2D ML 탐지**: Wang+2019a (CNN 영역 검출), Alshehhi & Marpu 2021 (VGG-16 + PCA + K-Means), Lin+2024b (track-match), Yang+2025 (CNN + Transformer + Kalman).
2. **3D 재구성 방법**: polarization ratio (Moran & Davila 2004), triangulation (Liewer+2007, 2010; Mierla+2008, 2009, 2010), forward modeling — **GCS 모델**(Thernisien+2006, 2009; Thernisien 2011)이 지배적 도구로 자리잡음.
3. **수동 GCS 카탈로그**: Bosman+2012 (1060 events), Shen+2013/2014 (86 halo), Kay & Gopalswamy 2017 (45 Earth-directed), Majumdar+2020, Gandhi+2024 (>360 events).

논문은 **공백**을 지적한다: 3D 재구성은 대체로 수동에 머물러 있다. Kay & Palmerio (2024)는 24개의 다른 카탈로그를 비교하여 ~27%(폭), 29%(aspect ratio), 19%(속도)의 카탈로그 간 차이를 발견 — 순전히 GCS 피팅의 인간 변동에서 발생. 저자들은 이 공백을 첫 번째 자동화 3D 파이프라인으로 메우려 한다.

본 논문은 같은 그룹의 이전 작업과 관련하여 자리잡는다: 이전 Lin+2024b track-match 논문이 2D CME 추적을 다뤘고, 본 논문은 같은 그룹의 전문성을 3D로 확장한다.

### Part II: Data / 데이터 (§2)

**English**

- **Instruments**: SOHO/LASCO C2 (FOV 2.2-6 R_⊙) + STEREO-A/COR2 (FOV 2.5-15 R_⊙)
- **Time range**: 2013–2018 (covers solar cycle 24 maximum to declining phase, with full activity variation)
- **Data source**: Heliophysics Application Programming Interface (HAPI) — `gs671-suske.ndc.nasa.gov`
- **Preprocessing**:
  1. Gaussian filter for noise suppression
  2. **Running difference** images: $I_t - I_{t-1}$ to highlight the propagating CME
  3. Resize from 2048² to 512² pixels (compute efficiency)
  4. Manual quality check, removing images with abnormalities
- **Final dataset**: 12 230 images, **6142 with CME** and **6088 without CME** (very nearly balanced)
- **Label source for CNN training**: each image classified by visual inspection (binary: CME/no-CME)
- **97 CME events** for the statistical analysis: morphology must be clearly separable from background structures (avoiding noisy or overlapping events)

**한국어**

- **기기**: SOHO/LASCO C2 (FOV 2.2-6 R_⊙) + STEREO-A/COR2 (FOV 2.5-15 R_⊙)
- **기간**: 2013-2018 (태양주기 24 극대기에서 하강기, 전체 활동 변동 포함)
- **데이터 소스**: Heliophysics Application Programming Interface (HAPI) — `gs671-suske.ndc.nasa.gov`
- **전처리**:
  1. 노이즈 억제용 Gaussian 필터
  2. **Running difference** 영상: $I_t - I_{t-1}$로 전파하는 CME 강조
  3. 2048² → 512² 픽셀 재스케일 (계산 효율)
  4. 수동 품질 체크, 비정상 영상 제거
- **최종 데이터셋**: 12,230 영상, **CME 포함 6142개**, **비포함 6088개** (거의 정확히 균형)
- **CNN 학습용 라벨**: 각 영상을 시각적으로 분류 (binary: CME/no-CME)
- **통계 분석용 97 CME 이벤트**: 형태가 배경 구조와 명확히 분리 가능해야 함 (노이즈나 중첩 이벤트 회피)

### Part III: Region Acquisition / 영역 획득 (§3.1)

**English**

The region-acquisition stage is a three-step pipeline that turns raw coronagraph images into per-pixel CME masks.

**Step 1 — CNN Classification**:
- Architecture: a modification of LeNet-5 (LeCun+1998), composed of conv + pooling + fully-connected layers
- Two **independent models** trained: one for SOHO/LASCO C2 images, one for STEREO-A/COR2 images (because their FOVs and pixel statistics differ)
- Loss: cross-entropy (binary classification: CME present vs absent)
- Optimiser: Stochastic Gradient Descent (SGD)
- After training, the CNN can predict CME presence with "high accuracy" (specific number not stated)

**Step 2 — PCA Colocalization Map**:
- Extract feature tensor $I \in \mathbb{R}^{h \times w \times d}$ from the **last convolutional layer** of the trained CNN
- Compute covariance matrix of the $d$ feature channels at each pixel position
- Eigendecomposition: project onto the **largest-eigenvalue eigenvector** (the direction of maximum variance, presumed to encode CME location)
- Result: 2D **colocalization map** $M \in \mathbb{R}^{h \times w}$
- The sign of values indicates positive/negative correlation; magnitude indicates strength
- This is the technique of Wei+2019 from object detection in computer vision

**Step 3 — Otsu Binarisation**:
- Normalise colocalization map to $[0, 255]$
- Apply **Otsu's method** (Otsu 1979): automatically choose the threshold that **maximises inter-class variance**
- Pixels above threshold → 1 (CME), below → 0 (non-CME)
- Output: binary CME region mask $c^i$ for viewpoint $i$

A **resizing step** is needed because the colocalization map size (originally 53×53) is smaller than the original 512² image. The authors later upgraded to 125×125 for better resolution. This is acknowledged as a source of minor inconsistency in the conclusion.

**한국어**

영역 획득 단계는 raw 코로나그래프 영상을 픽셀-단위 CME 마스크로 변환하는 3단계 파이프라인이다.

**1단계 — CNN 분류**:
- 아키텍처: LeNet-5 변형 (LeCun+1998), conv + pooling + fully-connected layer로 구성
- **독립적 두 모델** 학습: SOHO/LASCO C2 영상용 하나, STEREO-A/COR2 영상용 하나 (FOV와 픽셀 통계가 다르므로)
- Loss: cross-entropy (이진 분류: CME 포함 vs 비포함)
- Optimiser: Stochastic Gradient Descent (SGD)
- 학습 후 CNN은 "높은 정확도"로 CME 존재 예측 가능 (구체 수치 미기재)

**2단계 — PCA Colocalization Map**:
- 학습된 CNN의 **마지막 conv layer**에서 feature tensor $I \in \mathbb{R}^{h \times w \times d}$ 추출
- 각 픽셀 위치에서 $d$개 feature 채널의 공분산 행렬 계산
- 고유분해: **가장 큰 고유값의 고유벡터**로 투영 (최대 분산 방향, CME 위치 정보 인코딩 추정)
- 결과: 2D **colocalization map** $M \in \mathbb{R}^{h \times w}$
- 값의 부호 = 픽셀 상관의 방향, 크기 = 상관의 강도
- 이는 컴퓨터 비전의 객체 검출에서 Wei+2019의 기법

**3단계 — Otsu 이진화**:
- Colocalization map을 $[0, 255]$로 정규화
- **Otsu 방법**(Otsu 1979) 적용: **inter-class 분산을 최대화**하는 임계값 자동 선택
- 임계값 이상 픽셀 → 1 (CME), 이하 → 0 (비-CME)
- 출력: 시점 $i$의 이진 CME 영역 마스크 $c^i$

Colocalization map 크기(원래 53×53)가 원본 512² 영상보다 작으므로 **재스케일 단계**가 필요. 저자들은 나중에 더 좋은 해상도를 위해 125×125로 업그레이드. 결론에서 이것이 작은 불일치의 원인으로 인정됨.

### Part IV: GCS Model Construction / GCS 모델 구성 (§3.2)

**English**

The GCS model (Thernisien+2006; Thernisien 2011) parametrically describes a CME's 3D shape with 6 parameters and a fixed **croissant-with-flux-rope** topology:

| Parameter | Symbol | Meaning |
|---|---|---|
| Longitude | $\phi$ | Heliographic longitude of the eruption (Stonyhurst) |
| Latitude | $\theta$ | Heliographic latitude of the eruption |
| Tilt | $\gamma$ | Angular deviation between the foot-point line and the solar equator (axis orientation of the flux rope) |
| Half angle | $\alpha$ | Angle between each leg's central axis and the model's principal axis |
| Aspect ratio | $\kappa$ | Leg-thickness-to-height ratio, $\kappa = \sin\delta$ where $\delta$ is the angle between leg's central axis and side |
| Height | $h$ | Distance from origin to the model's apex (in solar radii) |

The 3D shape consists of:
- **Two conical legs** with foot points anchored on the solar surface
- A **toroidal front** (croissant body) that radially expands as a torus whose radius scales with heliocentric distance — mimicking a flux rope expanding while propagating

The model is constructed in a Sun-centred orthogonal frame $(O, x, y, z)$. After construction, the surface points are obtained in heliocentric coordinates and **projected onto the heliographic planes from each spacecraft's perspective** (using the formalism of Thernisien 2011 or its Python re-implementation by Larsson 2020).

**Convex hull**: The projected points form a scattered set. To convert them into a binary mask, the authors apply the **convex hull algorithm** (Sklansky 1982) — the smallest convex polygon enclosing all projected points. Pixels inside the polygon are labeled 1; outside, 0. This yields the projected CME mask $e^i(\mathbf{p})$ for each viewpoint $i$.

**3D angular width formula**:
$$w = 2(\alpha + \delta), \qquad \delta = \arcsin\kappa$$

This includes the leg thickness via $\kappa$, giving the **physical angular extent** rather than just the half-angle between legs.

**한국어**

GCS 모델(Thernisien+2006; Thernisien 2011)은 6개 파라미터와 고정된 **flux rope 형태의 croissant** 토폴로지로 CME의 3D 형상을 파라메트릭하게 표현한다:

| 파라미터 | 기호 | 의미 |
|---|---|---|
| 경도 | $\phi$ | 발생의 태양 좌표 경도 (Stonyhurst) |
| 위도 | $\theta$ | 발생의 태양 좌표 위도 |
| 기울기 | $\gamma$ | foot-point 선과 태양 적도 사이 각편차 (flux rope 축 방향) |
| Half angle | $\alpha$ | 각 leg의 중심축과 모델 주축 사이 각도 |
| Aspect ratio | $\kappa$ | leg 두께/높이 비, $\kappa = \sin\delta$ ($\delta$는 leg 중심축과 옆면 사이 각도) |
| 높이 | $h$ | 원점에서 모델 apex까지 거리 (태양 반경 단위) |

3D 형상 구성 요소:
- 발 끝점이 태양 표면에 고정된 **두 개의 conical leg**
- heliocentric 거리에 비례해 반경이 커지는 토러스로 방사형 팽창하는 **토러스 형태의 front** (croissant 몸체) — 전파하며 팽창하는 flux rope를 모방

모델은 태양 중심 직교 좌표계 $(O, x, y, z)$에서 구성. 구성 후 표면점은 heliocentric 좌표로 얻어지고, **각 우주선 관점에서 heliographic 평면에 투영**됨 (Thernisien 2011의 형식 또는 Larsson 2020의 Python 재구현 사용).

**Convex hull**: 투영된 점들은 흩뿌려진 집합을 이룬다. 이를 이진 마스크로 변환하기 위해 저자들은 **convex hull 알고리즘**(Sklansky 1982) 적용 — 모든 투영점을 포함하는 가장 작은 볼록 다각형. 다각형 내부 픽셀은 1, 외부는 0으로 라벨. 이로써 각 시점 $i$의 투영 CME 마스크 $e^i(\mathbf{p})$ 생성.

**3D 각폭 공식**:
$$w = 2(\alpha + \delta), \qquad \delta = \arcsin\kappa$$

이는 $\kappa$를 통해 leg 두께를 포함하여, 단순히 leg 사이 half-angle이 아닌 **물리적 각폭**을 제공.

### Part V: Function Optimization / 함수 최적화 (§3.3)

**English**

The reconstruction is formulated as an optimisation: find the 6 GCS parameters $\mathbf{p}$ that maximise the agreement between the CNN-derived region mask $c^i$ and the GCS-projected mask $e^i(\mathbf{p})$, simultaneously for both viewpoints.

**Object function (Eq. 1, 2, 3)**:
$$F(\mathbf{p}) = \frac{1}{n}\sum_{i=1}^{n} \left[\text{simi}(c^i, e^i(\mathbf{p})) - \text{area}(e^i(\mathbf{p}))\right]$$
$$\text{simi}(c^i, e^i(\mathbf{p})) = \frac{\sum_{j=1}^{N_i} c_j^i e_j^i(\mathbf{p})}{N_i}, \quad \text{area}(e^i(\mathbf{p})) = \beta\cdot\frac{\sum_{j=1}^{N_i} \text{IF}(e_j^i(\mathbf{p}), 1)}{N_i}$$
$$\text{IF}(x, y) = \begin{cases} 1 & x = y \\ 0 & x \neq y \end{cases}$$

- $n = 2$ (two viewpoints)
- $N_i$ = number of pixels in viewpoint $i$
- Similarity: counts pixels where both masks agree (= 1)
- Area penalty: counts pixels where the GCS mask is 1, weighted by $\beta$

**Why the area penalty is essential**: with similarity alone, the optimisation has a degenerate solution — making the GCS mask cover the entire FOV maximises overlap with $c^i$ trivially, regardless of CME morphology. The authors demonstrate this with the 2016-04-04 CME event (Figure 4):
- $\beta = 0$: GCS wire frame inflates to fill nearly the entire FOV (especially in SOHO/LASCO C2)
- $\beta = 1$: opposite extreme — GCS shrinks to a tiny region (over-penalised)
- $\beta = 0.12$: balanced fit (chosen via sweep search over $[0.1, 0.4]$ at $0.10$ interval, then refined to $[0.1, 0.2]$ at smaller interval)

**Two design principles for object function** (paper §3.3.1):
1. **Similarity term outweighing**: $\text{simi}(c^i, e^i) > \text{area}(e^i(\mathbf{p}))$ — overall morphology alignment must dominate
2. **Magnitude matching**: $0 < \frac{\text{simi}}{\text{area}} < 1$ — both terms should be of similar order to balance morphological accuracy and model compactness

**Differential Evolution (DE) algorithm** (Storn & Price 1997):
The 6D non-convex optimisation has many local maxima, so a global optimiser is needed. DE is a population-based evolutionary algorithm with four operations:

1. **Initialisation** (Eq. 6):
$$x_{j, i, 0} = \text{rand}(0, 1) \cdot (x_j^U - x_j^L) + x_j^L$$
- $N_p$ solutions, each a 6D vector
- Uniform sampling within parameter bounds $[x_j^L, x_j^U]$

2. **Mutation** (Eq. 7):
$$v_{i, G} = x_{r_1, G} + F\cdot(x_{r_2, G} - x_{r_3, G})$$
- $r_1, r_2, r_3$ are three distinct random solutions
- $F \in (0, 2]$ is the scaling factor (mutation strength)
- Creates a "differential" trial vector

3. **Crossover** (Eq. 8, 9):
$$u_{j, i, G} = \begin{cases} v_{j, i, G} & \text{if } \text{rand}_j \le CR \text{ or } \text{rnbr}(i) = j \\ x_{j, i, G} & \text{otherwise} \end{cases}$$
- $CR \in [0, 1]$ is crossover probability
- $\text{rnbr}(i)$ ensures at least one dimension is from the mutant (preventing trivial copies)

4. **Selection** (Eq. 10, greedy):
$$x_{i, G+1} = \begin{cases} u_{i, G} & \text{if } F(u_{i, G}) \ge F(x_{i, G}) \\ x_{i, G} & \text{otherwise} \end{cases}$$
- Replace parent only if child has equal or higher objective value
- Guarantees non-deterioration of the best solution

The DE iterates until convergence or a maximum generation count is reached. Output: the best 6 GCS parameters.

**Multi-frame fitting**: The authors fit each pair of STEREO-A/COR2 and SOHO/LASCO C2 observations at the same time, producing arrays of CME parameters across multiple time steps. **Velocity** is derived by linear fit to the height-time plot; other parameters use the **median** value across the array as the representative value.

**한국어**

재구성은 최적화로 공식화된다: CNN으로 도출된 영역 마스크 $c^i$와 GCS 투영 마스크 $e^i(\mathbf{p})$ 사이의 일치를 두 시점 동시에 최대화하는 6개 GCS 파라미터 $\mathbf{p}$를 찾는다.

**객관 함수 (Eq. 1, 2, 3)**:
$$F(\mathbf{p}) = \frac{1}{n}\sum_{i=1}^{n} \left[\text{simi}(c^i, e^i(\mathbf{p})) - \text{area}(e^i(\mathbf{p}))\right]$$
$$\text{simi}(c^i, e^i(\mathbf{p})) = \frac{\sum_{j=1}^{N_i} c_j^i e_j^i(\mathbf{p})}{N_i}, \quad \text{area}(e^i(\mathbf{p})) = \beta\cdot\frac{\sum_{j=1}^{N_i} \text{IF}(e_j^i(\mathbf{p}), 1)}{N_i}$$
$$\text{IF}(x, y) = \begin{cases} 1 & x = y \\ 0 & x \neq y \end{cases}$$

- $n = 2$ (두 시점)
- $N_i$ = 시점 $i$의 픽셀 수
- Similarity: 두 마스크가 일치(= 1)하는 픽셀 카운트
- Area penalty: GCS 마스크가 1인 픽셀 카운트, $\beta$ 가중

**왜 area penalty가 필수인가**: similarity만으로는 최적화가 degenerate solution을 가진다 — GCS 마스크가 전체 FOV를 덮으면 $c^i$와의 overlap이 자명하게 최대화되며 CME 형태와 무관. 저자들은 2016-04-04 CME 이벤트로 이를 시연(Figure 4):
- $\beta = 0$: GCS wire frame이 거의 전체 FOV를 채울 정도로 부풀어 오름 (특히 SOHO/LASCO C2에서)
- $\beta = 1$: 반대 극단 — GCS가 작은 영역으로 수축 (과다 페널티)
- $\beta = 0.12$: 균형 잡힌 fit ($[0.1, 0.4]$에 대해 $0.10$ 간격 sweep search, 그 다음 $[0.1, 0.2]$에서 더 작은 간격으로 정제하여 결정)

**객관 함수의 두 설계 원칙** (논문 §3.3.1):
1. **Similarity term 우세**: $\text{simi}(c^i, e^i) > \text{area}(e^i(\mathbf{p}))$ — 전반적 형태 일치가 지배해야 함
2. **크기 매칭**: $0 < \frac{\text{simi}}{\text{area}} < 1$ — 두 항이 비슷한 크기로 형태 정확도와 모델 압축성 균형

**Differential Evolution (DE) 알고리즘** (Storn & Price 1997):
6D 비볼록 최적화는 많은 local maxima를 가지므로 global optimiser 필요. DE는 인구 기반 진화 알고리즘이며 4개 연산:

1. **초기화** (Eq. 6):
$$x_{j, i, 0} = \text{rand}(0, 1) \cdot (x_j^U - x_j^L) + x_j^L$$
- $N_p$개 솔루션, 각각 6D 벡터
- 파라미터 경계 $[x_j^L, x_j^U]$ 내 균일 샘플링

2. **돌연변이** (Eq. 7):
$$v_{i, G} = x_{r_1, G} + F\cdot(x_{r_2, G} - x_{r_3, G})$$
- $r_1, r_2, r_3$은 서로 다른 무작위 솔루션
- $F \in (0, 2]$는 scaling factor (돌연변이 강도)
- "differential" 시도 벡터 생성

3. **교차** (Eq. 8, 9):
$$u_{j, i, G} = \begin{cases} v_{j, i, G} & \text{if } \text{rand}_j \le CR \text{ or } \text{rnbr}(i) = j \\ x_{j, i, G} & \text{otherwise} \end{cases}$$
- $CR \in [0, 1]$은 교차 확률
- $\text{rnbr}(i)$가 적어도 한 차원이 mutant에서 오도록 보장 (자명한 복사 방지)

4. **선택** (Eq. 10, 탐욕):
$$x_{i, G+1} = \begin{cases} u_{i, G} & \text{if } F(u_{i, G}) \ge F(x_{i, G}) \\ x_{i, G} & \text{otherwise} \end{cases}$$
- 자식이 부모와 같거나 높은 목적함수 값이면 교체
- 최적해의 비악화 보장

DE는 수렴까지 또는 최대 세대 수 도달까지 반복. 출력: 최적의 6 GCS 파라미터.

**다중 프레임 피팅**: 저자들은 동시간의 STEREO-A/COR2와 SOHO/LASCO C2 관측 쌍 각각을 피팅하여 다중 시간 단계에 걸친 CME 파라미터 배열 생성. **속도**는 height-time plot에 대한 linear fit으로 도출; 나머지 파라미터는 배열의 **중앙값**을 대표값으로 사용.

### Part VI: Case Studies / 사례 연구 (§4.1)

**English**

The paper analyses four representative CMEs in detail (Tables 1-2, Figs. 5-8):

**Case 1: 2012-08-02 (non-halo)**
- 3D: lat=−23°, lon=102°, tilt=80°, $\kappa$=0.37, half angle=10°, width=74°, **velocity=796 km/s**
- 2D (CDAW): width=108°, CPA=259°, **velocity=563 km/s**
- 3D velocity is **41% higher** than CDAW velocity (largest of the four)
- 3D width is **31% smaller** than CDAW width
- Confirms that even non-halo CMEs are subject to projection-effect biases
- Wire frame nicely envelops the CME structure in both viewpoints (Fig. 5)

**Case 2: 2012-01-08 (large angular extent)**
- 3D: lat=−19°, lon=237°, tilt=54°, $\kappa$=0.39, half angle=23°, width=99°, **velocity=696 km/s**
- 2D: width=174° (76% larger than 3D), velocity=557 km/s (20% smaller than 3D)
- Streamers visible in northwest (LASCO) and northeast (STEREO) but do not affect the reconstruction
- Demonstrates the algorithm's robustness against background streamers

**Case 3: 2012-11-09 (intensive eruption with clear flux rope)**
- 3D: lat=−39°, lon=244°, tilt=55°, $\kappa$=0.18, half angle=19°, width=86°, **velocity=891 km/s**
- 2D: width=262° (205% larger), velocity=771 km/s (13% smaller)
- The GCS conical legs cleanly connect to the solar disk; the cross-section matches the CME front
- Most dramatic case of width overestimation in 2D

**Case 4: 2011-10-01 (HALO CME)**
- 3D: lat=8°, lon=231°, tilt=−62°, $\kappa$=0.43, half angle=42°, width=151°, **velocity=1210 km/s**
- 2D: width=360° (CDAW marks halo as 360°), velocity=1238 km/s (only 2% larger than 3D!)
- Halo CME with Earth-impact potential
- A counterintuitive observation: 2D velocity is **slightly larger** than 3D velocity here
- The authors attribute this to **lateral expansion velocity > radial propagation velocity** for face-on halo events (citing Gopalswamy 2009; Shen+2013; Majumdar+2020) — when looking face-on, what LASCO measures as expansion is partially the lateral component
- Bright spots in NW quadrant of LASCO are interpreted as transient eruptions distinct from the main CME (and disappear in the next frame); the dual-viewpoint algorithm successfully cross-validates and ignores them

**Key takeaway from cases**: the dual-viewpoint algorithm's main advantage isn't just "better halo CME measurement" — it's **systematic correction of projection effects across all CMEs**, with the magnitude of correction depending on the angle to the line of sight.

**한국어**

논문은 4개의 대표 CME를 상세히 분석한다 (Tables 1-2, Figs. 5-8):

**사례 1: 2012-08-02 (non-halo)**
- 3D: lat=−23°, lon=102°, tilt=80°, $\kappa$=0.37, half angle=10°, width=74°, **velocity=796 km/s**
- 2D (CDAW): width=108°, CPA=259°, **velocity=563 km/s**
- 3D 속도가 CDAW 속도보다 **41% 더 높음** (네 사례 중 최대)
- 3D 폭이 CDAW 폭보다 **31% 작음**
- non-halo CME조차 투영 효과 편향에서 자유롭지 않음을 확인
- wire frame이 두 시점 모두에서 CME 구조를 잘 감쌈 (Fig. 5)

**사례 2: 2012-01-08 (큰 각폭)**
- 3D: lat=−19°, lon=237°, tilt=54°, $\kappa$=0.39, half angle=23°, width=99°, **velocity=696 km/s**
- 2D: width=174° (3D보다 76% 큼), velocity=557 km/s (3D보다 20% 작음)
- 북서(LASCO)와 북동(STEREO)에 streamer 보이지만 재구성에 영향 없음
- 배경 streamer에 대한 알고리즘의 견고성 입증

**사례 3: 2012-11-09 (명확한 flux rope의 격렬한 분출)**
- 3D: lat=−39°, lon=244°, tilt=55°, $\kappa$=0.18, half angle=19°, width=86°, **velocity=891 km/s**
- 2D: width=262° (205% 큼), velocity=771 km/s (13% 작음)
- GCS conical legs가 깔끔하게 태양 디스크에 연결; cross-section이 CME front와 일치
- 2D 폭 과대추정의 가장 극적인 사례

**사례 4: 2011-10-01 (HALO CME)**
- 3D: lat=8°, lon=231°, tilt=−62°, $\kappa$=0.43, half angle=42°, width=151°, **velocity=1210 km/s**
- 2D: width=360° (CDAW가 halo를 360°로 마킹), velocity=1238 km/s (3D보다 단 2% 큼!)
- 지구 충격 가능성 있는 halo CME
- 직관에 반하는 관찰: 2D 속도가 3D 속도보다 **약간 더 큼**
- 저자들은 이를 face-on halo 이벤트에서 **lateral expansion velocity > radial propagation velocity**로 설명 (Gopalswamy 2009; Shen+2013; Majumdar+2020 인용) — face-on으로 볼 때 LASCO가 expansion으로 측정하는 것이 부분적으로 lateral 성분
- LASCO의 NW quadrant 밝은 점은 본 CME와 별개의 일시적 분출로 해석 (다음 프레임에서 사라짐); 이중 시점 알고리즘이 이들을 성공적으로 cross-validation하여 무시

**사례에서의 핵심 시사점**: 이중 시점 알고리즘의 주된 장점은 단순히 "더 나은 halo CME 측정"이 아니다 — **모든 CME에 걸친 투영 효과의 체계적 보정**이며, 보정의 크기는 LOS와의 각도에 의존.

### Part VII: Statistics on 97 CMEs / 97개 CME 통계 (§4.2)

**English**

97 CME events spanning the rising and declining phase of solar cycle 24. Of these:
- **70 non-halo CMEs**
- **27 halo CMEs**

These are analysed separately because halo-CME 2D parameters have large deviations.

**§4.2.1 Velocity distributions (Figs. 9, 10)**

| Population | Source | Average velocity | Median velocity | Range |
|---|---|---|---|---|
| Non-halo | CDAW | 555 km/s | 528 | 187–1118 |
| Non-halo | This paper (3D) | 596 km/s | 585 | 192–1435 |
| Halo | CDAW | 1111 km/s | 1060 | — |
| Halo | This paper (3D) | 1211 km/s | 1163 | — |

**Headline**: 3D velocities are on average **8% higher** than 2D velocities (across non-halo + halo combined). This is consistent with prior literature: Sudar+2016 found 20% underestimate; Majumdar+2020 found 3% higher avg 3D speed; Gandhi+2024 found 8% higher (665 vs 613 km/s) — same as this paper.

The number of high-velocity events also increases when reconstructing in 3D (long tail to 1435 km/s vs 1118 km/s in CDAW).

**§4.2.1 Width distributions (Figs. 11, 12)**

| Population | Source | Average width | Median width | Range |
|---|---|---|---|---|
| Non-halo | CDAW | 130° | 132° | 16°–209° |
| Non-halo | This paper (3D) | 69° | 65° | — |
| Halo | This paper (3D) | 118° | 121° | 52°–153° |

**Headline**: 2D widths overestimate 3D widths by **47% on average** (non-halo). For halo events: CDAW always reports 360°, but real 3D widths range from 52° to 153° with avg 118° — a dramatic correction. Many halo events have widths shortened to $[32°, 139°]$, and events with widths >180° drop to zero.

Compared to prior work:
- Yeh+2005: 77° → 58° after projection correction
- Shen+2013: avg halo width 103°
- Jang+2016: avg 3D width 83° vs 2D 223°
- Gandhi+2024: avg true width 77° vs CDAW 189°

This paper's 118° average for halos is in the middle of these literature values — confirming the corrections are real even if magnitudes vary slightly with method.

**§4.2.2 Velocity-Width relationship (Fig. 13)**

The relationship has been studied repeatedly (Burkepile+2004, Yashiro 2004, Shen+2013, Jang+2016, Majumdar+2020, Gandhi+2024) — wider CMEs tend to be faster.

| Source | Pearson cc | Linear-fit slope |
|---|---|---|
| CDAW (2D) | **0.67** | 2.24 km/s/deg |
| This paper (3D) | **0.52** | 8.06 km/s/deg |

**Counterintuitive finding**: 3D correlation (0.52) is **lower** than 2D (0.67), but the slope is **~4× steeper**. The lower correlation reflects that 3D parameters have fewer artifacts that artificially inflate the correlation (e.g., halo widths all = 360° in 2D create spurious correlation), while the steeper slope shows the **true physical relationship** between width and velocity is stronger than 2D suggests.

Compared to literature:
- Shen+2013 (after GCS reconstruction): cc = 0.48
- Jang+2016: 2D cc = 0.47, 3D cc = 0.54
- Gandhi+2024: 2D cc = 0.60, 3D cc = 0.44

All P-values < 0.01 → statistically significant positive correlation. This paper's results are consistent with the broader literature.

**한국어**

태양주기 24의 상승기·하강기를 가로지르는 97개 CME 이벤트. 이 중:
- **non-halo 70개**
- **halo 27개**

halo CME의 2D 파라미터가 큰 편차를 가지므로 별도로 분석.

**§4.2.1 속도 분포 (Figs. 9, 10)**

| 모집단 | 출처 | 평균 속도 | 중앙 속도 | 범위 |
|---|---|---|---|---|
| Non-halo | CDAW | 555 km/s | 528 | 187–1118 |
| Non-halo | 본 논문 (3D) | 596 km/s | 585 | 192–1435 |
| Halo | CDAW | 1111 km/s | 1060 | — |
| Halo | 본 논문 (3D) | 1211 km/s | 1163 | — |

**핵심**: 3D 속도가 2D 속도보다 평균 **8% 더 높음** (non-halo + halo 합산). 이는 선행 문헌과 일치: Sudar+2016은 20% 과소 발견; Majumdar+2020은 3% 더 높은 평균 3D 속도; Gandhi+2024는 8% 더 높음 (665 vs 613 km/s) — 본 논문과 동일.

3D 재구성 시 고속 이벤트 수도 증가 (CDAW의 1118 km/s vs 본 논문 1435 km/s까지의 긴 꼬리).

**§4.2.1 폭 분포 (Figs. 11, 12)**

| 모집단 | 출처 | 평균 폭 | 중앙 폭 | 범위 |
|---|---|---|---|---|
| Non-halo | CDAW | 130° | 132° | 16°–209° |
| Non-halo | 본 논문 (3D) | 69° | 65° | — |
| Halo | 본 논문 (3D) | 118° | 121° | 52°–153° |

**핵심**: 2D 폭이 3D 폭보다 평균 **47% 과대추정** (non-halo). halo 이벤트: CDAW는 항상 360° 보고하지만 실제 3D 폭은 52°~153° 범위, 평균 118° — 극적 보정. 많은 halo 이벤트 폭이 $[32°, 139°]$로 축소되며, 폭 >180° 이벤트는 0개로 떨어짐.

선행 작업과 비교:
- Yeh+2005: 투영 보정 후 77° → 58°
- Shen+2013: halo 평균 폭 103°
- Jang+2016: 3D 평균 폭 83° vs 2D 223°
- Gandhi+2024: 평균 진짜 폭 77° vs CDAW 189°

본 논문의 halo 평균 118°는 이들 문헌 값의 중간 — 보정이 실재함을 확인 (방법에 따라 크기가 약간 다르더라도).

**§4.2.2 속도-폭 관계 (Fig. 13)**

이 관계는 반복적으로 연구됨 (Burkepile+2004, Yashiro 2004, Shen+2013, Jang+2016, Majumdar+2020, Gandhi+2024) — 넓은 CME가 더 빠른 경향.

| 출처 | Pearson cc | Linear-fit 기울기 |
|---|---|---|
| CDAW (2D) | **0.67** | 2.24 km/s/deg |
| 본 논문 (3D) | **0.52** | 8.06 km/s/deg |

**직관에 반하는 발견**: 3D 상관관계(0.52)가 2D(0.67)보다 **낮음**, 그러나 기울기는 **~4배 더 가파름**. 낮은 상관은 3D 파라미터가 인위적으로 상관을 부풀리는 artifact가 적음을 반영(예: 2D에서 halo 폭이 모두 360°이면 가짜 상관 생성), 더 가파른 기울기는 폭과 속도의 **진짜 물리적 관계**가 2D가 시사하는 것보다 강함을 보임.

문헌과 비교:
- Shen+2013 (GCS 재구성 후): cc = 0.48
- Jang+2016: 2D cc = 0.47, 3D cc = 0.54
- Gandhi+2024: 2D cc = 0.60, 3D cc = 0.44

모든 P-value < 0.01 → 통계적으로 유의한 양의 상관. 본 논문의 결과는 광범위한 문헌과 일치.

### Part VIII: Discussion & Limitations / 토론 및 한계 (§5)

**English**

The paper articulates several limitations and future directions:

1. **GCS shape assumption**: GCS is an empirical model of a flux-rope CME with axisymmetric structure. CMEs with **clear three-part flux-rope morphology** reconstruct better; events without clear flux rope (e.g., stealth CMEs, jet-like CMEs) are harder.

2. **Resizing inconsistency**: PCA's colocalization map output (originally 53×53) is smaller than the original 512² coronagraph image. Resizing introduces minor pixel-level inconsistencies. Authors increased map size to 125×125 in later iterations.

3. **β value selection**: $\beta = 0.12$ chosen by a sweep search over $[0.1, 0.4]$ at 0.10 interval, then $[0.1, 0.2]$ at smaller interval. A more principled optimisation (e.g., Bayesian) could in principle find a better value.

4. **Axisymmetric idealisation**: Real CMEs are not perfectly axisymmetric, so perfect alignment with both viewpoints is sometimes impossible. The paper acknowledges this for some events.

5. **Solar-cycle and STEREO geometry constraints**: Analysis is limited to 2007-2018 because (a) CDAW has reliable comparison data, (b) STEREO-A's geometry was favourable (it suffered solar conjunction in 2014-2015 and was lost from contact in 2014-2015 partially). The pipeline can be reapplied to post-2018 data.

6. **Comparison with Kay & Palmerio 2024 LLAMACoRe dataset**: This dataset collected manual CME reconstructions from many catalogs and showed inherent subjective biases between catalogs. Verbeke+2023 also confirmed dual/multi-viewpoint reconstruction reduces parameter uncertainty. This paper's automation directly addresses the subjective-bias problem.

7. **Future applications**:
   - **Initialisation of MHD simulations** (ENLIL, EUHFORIA, etc.) — requires accurate launch parameters
   - **CME arrival time prediction** — both physics-based and ML-based forecasters need correct 3D parameters
   - **Real-time space weather forecasting** — automation enables operational use

**한국어**

논문은 몇 가지 한계와 향후 방향을 명시한다:

1. **GCS 형태 가정**: GCS는 axisymmetric 구조의 flux-rope CME 경험적 모델. **명확한 three-part flux-rope 형태**의 CME가 더 잘 재구성됨; flux rope가 명확하지 않은 이벤트(stealth CME, jet-like CME 등)는 더 어려움.

2. **재스케일 불일치**: PCA의 colocalization map 출력(원래 53×53)이 원본 512² 코로나그래프 영상보다 작음. 재스케일이 작은 픽셀-단위 불일치 도입. 저자들은 후속 반복에서 map 크기를 125×125로 증가.

3. **β 값 선택**: $\beta = 0.12$가 $[0.1, 0.4]$에 대해 0.10 간격 sweep search, 그 다음 $[0.1, 0.2]$에서 더 작은 간격으로 결정. 더 원칙적 최적화(예: Bayesian)로 더 좋은 값 가능.

4. **Axisymmetric 이상화**: 실제 CME는 완벽히 axisymmetric하지 않으므로, 두 시점과 완벽한 정렬이 때로 불가능. 일부 이벤트에 대해 논문이 이를 인정.

5. **태양주기 및 STEREO 기하 제약**: 분석은 2007-2018에 제한 — (a) CDAW가 신뢰할 만한 비교 데이터 가짐, (b) STEREO-A 기하가 호의적이었음(2014-2015에 solar conjunction과 부분적 통신 두절). 파이프라인은 2018 이후 데이터에 재적용 가능.

6. **Kay & Palmerio 2024 LLAMACoRe 데이터셋과 비교**: 이 데이터셋은 많은 카탈로그의 수동 CME 재구성을 모았고 카탈로그 간 본질적 주관적 편향을 보여줌. Verbeke+2023도 dual/multi-viewpoint 재구성이 파라미터 불확실성을 줄임을 확인. 본 논문의 자동화는 주관적 편향 문제를 직접 해결.

7. **향후 응용**:
   - **MHD 시뮬레이션 초기화** (ENLIL, EUHFORIA 등) — 정확한 발사 파라미터 필요
   - **CME 도착 시각 예측** — 물리 기반·ML 기반 예보기 모두 올바른 3D 파라미터 필요
   - **실시간 우주기상 예보** — 자동화로 운영 사용 가능

---

## 3. Key Takeaways / 핵심 시사점

1. **First fully-automatic 3D CME reconstruction pipeline.** The paper combines learned region detection (CNN+PCA+Otsu) with physics-based forward modeling (GCS) and global optimisation (DE), eliminating the manual parameter tweaking that has dominated GCS catalogs for nearly two decades. This addresses the 19-29% inter-catalog scatter from human variation reported by Kay & Palmerio (2024). / **최초의 완전 자동 3D CME 재구성 파이프라인.** 학습 기반 영역 검출(CNN+PCA+Otsu)과 물리 기반 forward modeling(GCS), global 최적화(DE)를 결합하여, 거의 20년간 GCS 카탈로그를 지배해온 수동 파라미터 조정을 제거. 이는 Kay & Palmerio (2024)가 보고한 인간 변동에서 발생하는 19-29% 카탈로그 간 분산을 해결.

2. **Modular hybrid > monolithic end-to-end CNN.** The ML and the physics are deliberately decoupled: CNN does region segmentation (easy, well-supervised), DE does 6D parameter inference (hard, ill-posed). End-to-end CNN regression to GCS parameters would face label scarcity (~10³ vs ~10⁵-10⁶ needed), periodic-parameter losses, multi-modal solutions, and OOD generalisation problems. The hybrid is a textbook example of scientific-ML pragmatism. / **모듈형 hybrid > 단일 end-to-end CNN.** ML과 물리를 의도적으로 분리: CNN은 영역 분할(쉬움, 지도학습 가능), DE는 6D 파라미터 추론(어려움, 부적절 문제). End-to-end CNN GCS 파라미터 회귀는 라벨 부족(~10³ vs 필요한 ~10⁵-10⁶), 주기적 파라미터 loss, 다중 모드 해, OOD 일반화 문제를 겪을 것. 이 hybrid는 scientific ML 실용주의의 교과서적 예시.

3. **The area penalty term is essential, not optional.** Without it, optimization is degenerate — the GCS mask inflates to fill the FOV trivially maximising similarity. With $\beta = 0.12$, the GCS shrinks back to physical CME size. This illustrates a general principle: when fitting parametric models to image data, **regularisation against trivial solutions** is structurally necessary. / **Area penalty 항이 필수다.** 이것 없이는 최적화가 degenerate — GCS 마스크가 FOV를 채워 similarity를 자명하게 최대화. $\beta = 0.12$로 GCS가 물리적 CME 크기로 다시 수축. 이는 일반적 원칙을 보여줌: 파라메트릭 모델을 영상 데이터에 fit할 때 **자명한 해에 대한 정규화**가 구조적으로 필요.

4. **Quantified projection-effect biases: −8% velocity, +47% width on average.** Single-viewpoint LASCO underestimates 3D velocity by ~8% (Sudar+2016 found 20%, Gandhi+2024 8% — this paper agrees with Gandhi). Width is overestimated by ~47% on average for non-halo events. These numbers are now usable as **standard correction factors** for any pre-2024 CME catalog used as input to forecasting models. / **투영 효과 편향 정량화: 평균 속도 −8%, 폭 +47%.** 단일 시점 LASCO는 3D 속도를 ~8% 과소 추정 (Sudar+2016은 20%, Gandhi+2024는 8% — 본 논문은 Gandhi와 일치). 폭은 non-halo 이벤트에서 평균 ~47% 과대 추정. 이 수치들은 이제 예보 모델 입력으로 사용되는 2024년 이전 CME 카탈로그에 대한 **표준 보정 인자**로 활용 가능.

5. **Halo CMEs are the most dramatic correction case.** CDAW reports halo widths as 360° (an information-free placeholder), but the algorithm produces real 3D widths in $[52°, 153°]$ with average 118°. For Earth-directed CMEs (the most space-weather-relevant ones), this is the difference between "no real width measurement" and "physical estimate" — directly impacting MHD simulation initialisation. / **Halo CME는 가장 극적인 보정 사례.** CDAW는 halo 폭을 360°로 보고(정보 없는 플레이스홀더), 알고리즘은 평균 118°의 $[52°, 153°]$ 범위 실제 3D 폭 생성. 지구 방향 CME(가장 우주기상 관련)의 경우 "실제 폭 측정 없음"과 "물리적 추정" 사이의 차이 — MHD 시뮬레이션 초기화에 직접 영향.

6. **Counterintuitive halo result: 2D speed > 3D speed for face-on halos.** For the 2011-10-01 halo event, CDAW velocity (1238 km/s) is slightly *larger* than 3D velocity (1210 km/s). The authors explain this as **lateral expansion velocity > radial propagation velocity** for face-on CMEs (Gopalswamy 2009; Shen+2013; Majumdar+2020). When LASCO sees a halo head-on, it measures the expansion of the cross-section (lateral component), not the actual radial speed. This subtle physics point is one of the paper's most important insights. / **직관에 반하는 halo 결과: face-on halo에서 2D 속도 > 3D 속도.** 2011-10-01 halo 이벤트의 CDAW 속도(1238 km/s)가 3D 속도(1210 km/s)보다 약간 *더 큼*. 저자들은 face-on CME의 **lateral expansion velocity > radial propagation velocity** 로 설명 (Gopalswamy 2009; Shen+2013; Majumdar+2020). LASCO가 halo를 head-on으로 보면 단면의 expansion(lateral 성분)을 측정하지 실제 radial 속도가 아님. 이 미묘한 물리 포인트는 논문의 가장 중요한 통찰 중 하나.

7. **Width-velocity relationship is genuinely steeper in 3D.** The Pearson cc drops from 0.67 (2D) to 0.52 (3D), but the linear-fit slope steepens from 2.24 to 8.06 km/s/deg. Lower cc reflects the removal of artifacts (halo widths all=360° create spurious correlation in 2D); steeper slope shows the true physics. **A wider CME is genuinely faster** — 4× more so than 2D suggests. / **폭-속도 관계는 3D에서 진정으로 더 가파름.** Pearson cc는 0.67(2D)에서 0.52(3D)로 떨어지지만, linear-fit 기울기는 2.24에서 8.06 km/s/deg로 가팔라짐. 낮은 cc는 artifact 제거 반영(2D에서 halo 폭이 모두 360°이면 가짜 상관 생성); 더 가파른 기울기는 진짜 물리. **넓은 CME가 진정으로 더 빠름** — 2D가 시사하는 것보다 4배 더.

8. **GCS is shape-restrictive — limited to flux-rope CMEs.** The paper acknowledges: events without clear three-part flux-rope morphology (stealth CMEs, jets, prominence eruptions without a clear flux rope) reconstruct poorly. Future work needs more flexible parametric models or non-parametric (e.g., implicit neural representation) approaches. / **GCS는 형태 제약 — flux-rope CME에 한정.** 논문 인정: 명확한 three-part flux-rope 형태가 없는 이벤트(stealth CME, jet, 명확한 flux rope 없는 prominence 분출)는 잘 재구성되지 않음. 향후 작업은 더 유연한 파라메트릭 모델 또는 비파라메트릭(예: implicit neural representation) 접근 필요.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Object function / 객관 함수 (Eq. 1, 2, 3, 4)

$$\boxed{\;F(\mathbf{p}) = \frac{1}{n}\sum_{i=1}^{n} \left[\,\text{simi}(c^i, e^i(\mathbf{p})) - \text{area}(e^i(\mathbf{p}))\,\right]\;}$$

$$\text{simi}(c^i, e^i(\mathbf{p})) = \frac{\sum_{j=1}^{N_i} c_j^i\, e_j^i(\mathbf{p})}{N_i}$$

$$\text{area}(e^i(\mathbf{p})) = \beta \cdot \frac{\sum_{j=1}^{N_i} \text{IF}(e_j^i(\mathbf{p}), 1)}{N_i}, \qquad \text{IF}(x, y) = \begin{cases} 1 & x = y \\ 0 & x \neq y \end{cases}$$

| Symbol | Meaning |
|---|---|
| $\mathbf{p} = (\phi, \theta, \gamma, \alpha, \kappa, h)$ | 6 GCS parameters / 6 GCS 파라미터 |
| $c_j^i \in \{0, 1\}$ | pixel $j$ of the CME region mask of viewpoint $i$ (CNN-derived) / 시점 $i$의 CME 영역 마스크 픽셀 (CNN 도출) |
| $e_j^i(\mathbf{p}) \in \{0, 1\}$ | pixel $j$ of the GCS-projected mask onto viewpoint $i$ / 시점 $i$로 투영된 GCS 마스크 픽셀 |
| $n = 2$ | number of viewpoints (LASCO C2 + STEREO-A/COR2) |
| $N_i$ | total pixels in viewpoint $i$ |
| $\beta = 0.12$ | area-penalty coefficient (sweep-search optimal) / area penalty 계수 (sweep search 최적) |

**Two design principles** (paper §3.3.1) / **두 설계 원칙**:
1. **Similarity outweighs**: $\text{simi}(c^i, e^i) > \text{area}(e^i(\mathbf{p}))$ — overall morphology dominates / 전반적 형태가 지배
2. **Magnitude matching**: $0 < \frac{\text{simi}}{\text{area}} < 1$ — both terms of similar order / 두 항이 비슷한 크기

### 4.2 GCS 3D angular width / GCS 3D 각폭

$$\boxed{\;w = 2(\alpha + \delta), \quad \delta = \arcsin\kappa\;}$$

- $w$ — physical 3D angular width / 물리적 3D 각폭
- $\alpha$ — half angle (between leg axes) / leg 축 사이 half-angle
- $\kappa$ — aspect ratio (leg thickness / height) / leg 두께/높이 비
- $\delta = \arcsin\kappa$ — angle from leg central axis to leg side / leg 중심축에서 옆면까지의 각도

**Why $\arcsin$?** The leg is a cone of radius $\kappa h$ at apex height $h$, so $\sin\delta = \kappa h / h = \kappa$. The factor of 2 accounts for both legs symmetric about the principal axis. / **왜 $\arcsin$?** leg는 apex 높이 $h$에서 반경 $\kappa h$인 원뿔이므로 $\sin\delta = \kappa h / h = \kappa$. 인자 2는 주축에 대해 대칭인 두 leg를 고려.

### 4.3 Differential Evolution operations / DE 연산 (Eq. 6-10)

**Initialisation / 초기화** (uniform in bounds):
$$x_{j, i, 0} = \text{rand}(0, 1) \cdot (x_j^U - x_j^L) + x_j^L$$

**Mutation / 돌연변이** (DE/rand/1):
$$\boxed{\;v_{i, G} = x_{r_1, G} + F\cdot(x_{r_2, G} - x_{r_3, G})\;}$$
- $r_1, r_2, r_3$ are three distinct random indices (also distinct from $i$) / 서로 다른 무작위 인덱스 ($i$와도 다름)
- $F \in (0, 2]$ — scaling factor controlling mutation magnitude / 돌연변이 크기 조절

**Crossover / 교차** (binomial):
$$u_{j, i, G} = \begin{cases} v_{j, i, G} & \text{if } \text{rand}_j \le CR \;\text{or}\; \text{rnbr}(i) = j \\ x_{j, i, G} & \text{otherwise} \end{cases}$$
- $CR \in [0, 1]$ — crossover probability / 교차 확률
- $\text{rnbr}(i)$ — random integer in $[1, D]$ ensuring at least one mutated dim / 적어도 한 차원이 mutate되도록 보장

**Selection / 선택** (greedy):
$$x_{i, G+1} = \begin{cases} u_{i, G} & \text{if } F(u_{i, G}) \ge F(x_{i, G}) \\ x_{i, G} & \text{otherwise} \end{cases}$$
- Replace parent only if child has equal or higher objective / 자식이 부모와 같거나 높을 때만 교체
- Guarantees population's best score is non-decreasing / 인구의 최고 점수가 비감소 보장

### 4.4 PCA dimensionality reduction / PCA 차원 축소

Given CNN's last conv-layer feature tensor $I \in \mathbb{R}^{h \times w \times d}$: / CNN의 마지막 conv layer feature tensor $I \in \mathbb{R}^{h \times w \times d}$가 주어지면:

1. **Reshape**: $I \to X \in \mathbb{R}^{(hw) \times d}$ (each pixel as a $d$-dim feature vector)
2. **Centre**: $X \leftarrow X - \bar{X}$
3. **Covariance**: $\Sigma = \frac{1}{hw} X^\top X \in \mathbb{R}^{d \times d}$
4. **Eigendecomposition**: $\Sigma = V \Lambda V^\top$, with eigenvalues $\lambda_1 \ge \lambda_2 \ge \ldots \ge \lambda_d$
5. **Project onto largest direction**: $M = X v_1 \in \mathbb{R}^{hw}$, reshape back to $\mathbb{R}^{h \times w}$
6. **Result**: colocalization map $M$ — large $|M|$ values indicate pixels with strong CME signal

### 4.5 Otsu's method / Otsu 방법 (Otsu 1979)

Given image histogram normalised to $[0, 255]$, search for threshold $t$ maximising **inter-class variance**: / 정규화된 영상 히스토그램에서 **inter-class 분산**을 최대화하는 임계값 $t$ 탐색:

$$\sigma_B^2(t) = w_0(t) w_1(t) \left[\mu_0(t) - \mu_1(t)\right]^2$$

- $w_0(t), w_1(t)$ — fractions of pixels below/above threshold / 임계값 미만/이상 픽셀 비율
- $\mu_0(t), \mu_1(t)$ — means of below/above classes / 미만/이상 클래스의 평균
- Optimal $t^* = \arg\max_t \sigma_B^2(t)$

**Equivalent**: minimising **intra-class variance** (the two are complementary by total-variance decomposition). / **동등**: **intra-class 분산** 최소화 (전체 분산 분해로 두 가지가 보완적).

### 4.6 Worked numerical example / 수치 사례

Consider a hypothetical CME with true GCS parameters $\mathbf{p}_{\text{true}} = (\phi=120°, \theta=−30°, \gamma=45°, \alpha=20°, \kappa=0.4, h=10\,R_\odot)$:

- **3D width**: $w = 2(\alpha + \arcsin\kappa) = 2(20° + 23.6°) = 87.2°$
- If observed face-on by LASCO (LOS aligned with CME direction): apparent width could be ~360° (halo)
- If observed by STEREO-A separated by 90°: side view, apparent width ≈ 87° (close to true)
- Without dual-view, you cannot distinguish a face-on halo (true width 87°) from an edge-on wide CME (true width 360°)
- Object function evaluation:
  - Suppose $c^1$ (LASCO mask) has 5000 CME pixels out of $N_1 = 512^2 = 262144$ → fraction 1.9%
  - Suppose $e^1(\mathbf{p}_{\text{true}})$ has 4500 pixels overlapping with $c^1$, plus 1000 disagreement → simi = 4500/262144 ≈ 0.0172
  - $e^1$ projected mask has 5500 total pixels → area = $0.12 \times 5500/262144 \approx 0.0025$
  - Per-viewpoint $F$ contribution: $0.0172 - 0.0025 = 0.0147$
  - Two-viewpoint average → $F \approx 0.014$
- A bad parameter set with bloated GCS might get simi $\approx 0.018$ but area $\approx 0.06$ → $F \approx -0.04$ (worse despite higher simi)

This illustrates why area penalty is critical. / 이는 area penalty가 왜 중요한지 보여준다.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1979 ──── Otsu — automatic image thresholding by inter-class variance
            │
1982 ──── Sklansky — convex hull algorithm
            │
1995 ──── SOHO launched (Brueckner+1995) — LASCO C2/C3 begin
            │  CDAW SOHO/LASCO catalog (Yashiro 2004) — manual single-view
            │
1997 ──── Storn & Price — Differential Evolution (DE) algorithm
1998 ──── LeCun+ — LeNet-5 CNN architecture
            │
2004 ──── Robbrecht & Berghmans — CACTus (wavelet-based auto CME detection, 2D)
2004 ──── Moran & Davila — polarization-ratio 3D reconstruction
            │
2006 ──── STEREO launched (Kaiser 2005) — twin spacecraft, dual viewpoints
2006 ──── Thernisien+ — GCS (Graduated Cylindrical Shell) model introduced
            │
2007–2018 ── Productive STEREO-A/COR2 + LASCO C2 dual-viewpoint era
            │
2008 ──── Mierla+ — 3D CME kinematics from dual-view triangulation
2009 ──── Thernisien+ — GCS catalog of 26 events (manual fit)
2011 ──── Thernisien — GCS mathematical formalism solidified
2012 ──── Bosman+ — 1060-event manual GCS catalog from SECCHI/COR2
            │
2013-14 ── Shen+ — 86 halo CMEs with GCS, found 47% width inflation
2017 ──── Kay & Gopalswamy — 45 Earth-directed CMEs with GCS + magnetic profiles
2018 ──── Vourlidas+ — dual-viewpoint CME catalog (manual)
            │
2019 ──── Wang+ — CNN-based CME region detection (2D only)
2020 ──── Majumdar+ — 3D evolution kinematics with GCS
2020 ──── Larsson — Python re-implementation of GCS model
2021 ──── Alshehhi & Marpu — VGG-16 + PCA + K-Means (2D only)
            │
2024 ──── Lin+ (this group) — track-match 2D CME tracking (CNN)
2024 ──── Gandhi+ — 360+ CMEs with GCS, found 8% velocity correction
2024 ──── Kay & Palmerio — 24-catalog comparison: 19-29% manual scatter, LLAMACoRe
2025 ──── Yang+ (this group) — CNN + Transformer + Kalman (2D)
            │
2025 ──── ★ THIS PAPER (Lin et al.) ★
            │  → First fully-automatic 3D CME reconstruction
            │  → CNN region detection + GCS + Differential Evolution
            │  → 97 CMEs analysed; quantifies 8% velocity, 47% width biases
            │  → Validates against existing manual catalogs
            │
202X ──── (Future) Real-time pipeline integration
                  Solar Orbiter Metis / PSP WISPR as 3rd, 4th viewpoints
                  Differentiable rendering + neural fields for non-flux-rope CMEs
                  Bayesian-optimal β and uncertainty quantification
```

**Position in the field / 분야에서의 자리**: Lin et al. (2025)는 **3D CME 재구성을 자동화한 첫 논문**이다. 두 가지 평행한 연구 흐름 — (1) ML 기반 2D CME 탐지(2019-2025)와 (2) 수동 GCS 3D 카탈로그(2006-2024) — 을 결합하여, scientific-ML hybrid 패턴의 모범 사례를 제공한다. Kay & Palmerio (2024)가 정량화한 카탈로그 간 19-29% 분산을 자동화로 해결할 수 있는 길을 열었다. / Lin et al. (2025) is the **first paper to automate 3D CME reconstruction**. By combining two parallel research streams — (1) ML-based 2D CME detection (2019-2025) and (2) manual GCS 3D catalogs (2006-2024) — it provides an exemplar of the scientific-ML hybrid pattern, opening a path to automatically resolve the 19-29% inter-catalog scatter quantified by Kay & Palmerio (2024).

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **Thernisien et al. 2006/2009/2011 — GCS model** | Defines the parametric 3D CME shape used by this paper / 본 논문이 사용하는 파라메트릭 3D CME 형상 정의 | Essential foundational reading — without GCS, no 3D parameters / 필수 기초 — GCS 없이는 3D 파라미터 없음 |
| **Storn & Price 1997 — Differential Evolution** | The optimisation algorithm used to maximise the object function / 객관 함수 최대화에 사용되는 최적화 알고리즘 | Methodologically central; explains why a population-based global optimiser was needed / 방법론적 핵심; 왜 인구 기반 global optimiser가 필요했는지 설명 |
| **LeCun et al. 1998 — LeNet-5** | The CNN architecture (modified) used for CME image classification / CME 영상 분류용 CNN 아키텍처 (변형) | Shows the modesty of ML choice — a 1998 architecture suffices when the task is simple binary classification / ML 선택의 절제를 보여줌 — 작업이 단순 이진 분류일 때 1998년 아키텍처로 충분 |
| **Otsu 1979 — Automatic thresholding** | Used to convert the PCA colocalization map into a binary CME mask / PCA colocalization map을 이진 CME 마스크로 변환에 사용 | A 46-year-old algorithm doing critical work — illustrates the value of classical methods in modern hybrid pipelines / 46년 된 알고리즘이 핵심 역할 — 현대 hybrid 파이프라인에서 고전 방법의 가치 입증 |
| **Yashiro 2004 — CDAW SOHO/LASCO catalog** | The 2D ground-truth catalog this paper compares against / 본 논문이 비교하는 2D ground truth 카탈로그 | Reference for the 8%/47% bias quantification — without CDAW, no comparison baseline / 8%/47% 편향 정량화의 기준 — CDAW 없이는 비교 baseline 없음 |
| **Kay & Palmerio 2024 — LLAMACoRe / 24-catalog comparison** | Quantifies the 19-29% inter-catalog scatter from manual GCS fitting / 수동 GCS 피팅의 19-29% 카탈로그 간 분산 정량화 | Provides the **motivation** for automating GCS — "why do this work" answer / GCS 자동화의 **동기** 제공 — "왜 이 작업을 하는가" 답변 |
| **Heinemann et al. 2019 / Inceoglu et al. 2022 — CH detection (#37)** | Sister paper in spirit: also automates a parameter-extraction task (CHs there, CMEs here) using ML + classical methods / 정신적 sister paper: ML + 고전 방법으로 파라미터 추출 작업을 자동화 (CH는 거기, CME는 여기) | Both papers exemplify the scientific-ML hybrid where ML does perception (segmentation) and physics/math does inference / 두 논문 모두 ML이 perception (분할), 물리/수학이 inference를 하는 scientific-ML hybrid의 예시 |
| **Wang et al. 2019a — CNN for CME region detection** | Predecessor paper that did 2D CME region detection only / 2D CME 영역 검출만 수행한 선행 논문 | Shows what was missing — Lin et al. (2025) extends to 3D / 무엇이 부족했는지 보여줌 — Lin et al. (2025)이 3D로 확장 |
| **Jarolim et al. 2021 — CHRONNOS CNN** | Conceptually parallel: end-to-end CNN for solar segmentation (CHs in CHRONNOS, CMEs here in 2D mask part) / 개념적 평행: 태양 분할용 end-to-end CNN (CHRONNOS는 CH, 여기는 2D 마스크 부분 CME) | Compare/contrast — Lin et al. is more modular, CHRONNOS is more end-to-end / 비교/대조 — Lin et al.이 더 모듈형, CHRONNOS가 더 end-to-end |

---

## 7. References / 참고문헌

### Primary paper / 본 논문
- Lin, R., Yang, Y., Shen, F., Pi, G., & Li, Y., "Automatic 3D Reconstruction of Coronal Mass Ejections Based on Dual-viewpoint Observations and Machine Learning", *The Astrophysical Journal Supplement Series*, 280:44 (17pp), 2025. DOI: [10.3847/1538-4365/adf433](https://doi.org/10.3847/1538-4365/adf433)

### GCS model lineage / GCS 모델 계보
- Thernisien, A. F. R., Howard, R. A., & Vourlidas, A., "Modeling of flux rope coronal mass ejections", *ApJ*, 652, 763, 2006.
- Thernisien, A., Vourlidas, A., & Howard, R. A., "Forward Modeling of Coronal Mass Ejections Using STEREO/SECCHI Data", *Solar Physics*, 256, 111, 2009.
- Thernisien, A., "Implementation of the Graduated Cylindrical Shell Model for the Three-dimensional Reconstruction of Coronal Mass Ejections", *ApJS*, 194, 33, 2011.
- Larsson, J., "Python implementation of the GCS model", PhD Thesis, 2020.
- Bosman, E., "GCS catalog of 1060 CMEs from SECCHI/COR2", PhD Thesis, Univ. Goettingen, 2017 (also Bosman et al. 2012).

### Methodological references / 방법론 참고문헌
- MacQueen, J., "Some methods for classification and analysis of multivariate observations", *Proc. 5th Berkeley Symp.*, 281, 1967.
- Otsu, N., "A threshold selection method from gray-level histograms", *IEEE Trans. Systems, Man, Cybernetics*, 9, 62, 1979.
- Sklansky, J., "Finding the convex hull of a simple polygon", *Pattern Recognition Letters*, 1, 79, 1982.
- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P., "Gradient-based learning applied to document recognition", *Proc. IEEE*, 86, 2278, 1998.
- Storn, R., & Price, K., "Differential Evolution — A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces", *J. Global Optimization*, 11, 341, 1997.
- Kenneth, V. P., Rainer, M. S., Jouni, A. L., et al., *Differential Evolution* (Berlin: Springer), 2005.
- Qiang, J., & Mitchell, C., "An Adaptive Unified Differential Evolution Algorithm for Global Optimization", LBNL-6853E, 2015.
- Wei, X.-S., Zhang, C.-L., Wu, J., Shen, C., & Zhou, Z.-H., "Unsupervised object discovery and co-localization by deep descriptor transformation", *Pattern Recognition*, 88, 113, 2019.

### CME 3D reconstruction history / CME 3D 재구성 역사
- Moran, T. G., & Davila, J. M., "Three-Dimensional Polarimetric Imaging of Coronal Mass Ejections", *Science*, 305, 66, 2004.
- Mierla, M., Davila, J., Thompson, W., et al., "A Quick Method for Estimating the Propagation Direction of Coronal Mass Ejections", *Solar Physics*, 252, 385, 2008.
- Mierla, M., Inhester, B., Antunes, A., et al., "On the 3-D reconstruction of CMEs using STEREO data", *Annales Geophysicae*, 28, 203, 2010.
- Liewer, P. C., De Jong, E. M., Hall, J. R., et al., "Stereoscopic Analysis of the 26 April 2008 CME", *Solar Physics*, 256, 57, 2010.
- Shen, C., Wang, Y., Pan, Z., et al., "Full halo coronal mass ejections: Arrival at the Earth", *JGRA*, 119, 5107, 2014.
- Shen, C., Wang, Y., Pan, Z., et al., "Statistical analysis of halo CME width and projection effects", *JGRA*, 118, 6858, 2013.
- Kay, C., & Gopalswamy, N., "The effects of uncertainty on coronal mass ejection arrival predictions", *JGRA*, 122, 810, 2017.
- Majumdar, S., Pant, V., Patel, R., & Banerjee, D., "Connecting 3D Evolution of Coronal Mass Ejections to Their Source Regions", *ApJ*, 899, 6, 2020.
- Gandhi, H., Patel, R., Pant, V., et al., "A statistical investigation of the dependence of CME parameters on geomagnetic storm intensity using GCS-reconstructed 3D parameters", *Space Weather*, 22, e2023SW003805, 2024.
- Kay, C., & Palmerio, E., "LLAMACoRe: a comparison of 24 CME catalogs and the importance of standardisation", *Space Weather*, 22, e2023SW003796, 2024.
- Verbeke, C., Mays, M. L., Temmer, M., et al., "Quantitative validation of WSA-ENLIL+Cone CME prediction model", *Advances in Space Research*, 72, 5243, 2023.

### CME 2D detection ML / CME 2D 탐지 ML
- Robbrecht, E., & Berghmans, D., "Automated recognition of coronal mass ejections (CMEs)", *A&A*, 425, 1097, 2004.
- Olmedo, O., Zhang, J., Wechsler, H., Poland, A., & Borne, K., "SEEDS — Solar Eruptive Event Detection System", *Solar Physics*, 248, 485, 2008.
- Byrne, J. P., Morgan, H., Habbal, S. R., & Gallagher, P. T., "CORIMP — Coronal Image Processing", *ApJ*, 752, 144, 2012.
- Wang, P., Zhang, Y., Feng, L., et al., "A New Automatic Tool for CME Detection and Tracking with Machine-learning Techniques", *ApJS*, 244, 9, 2019a.
- Wang, Y., Liu, J., & Jiang, Y., "A novel deep learning framework for accurate prediction of CME arrival time", *ApJ*, 881, 15, 2019b.
- Alshehhi, R., & Marpu, P. R., "Unsupervised Detection of Coronal Mass Ejections in SOHO/LASCO C2 and C3 Images", *Solar Physics*, 296, 104, 2021.
- Yang, Y., Shen, F., Yang, Z., & Feng, X., "Prediction of solar wind speed at 1 AU using an artificial neural network", *Space Weather*, 16, 1227, 2018.
- Lin, R., Luo, Z., He, J., et al., "A Track-match Algorithm for Automated Detection of Coronal Mass Ejections", *Space Weather*, 22, e2023SW003561, 2024a.
- Lin, R., Yang, Y., Shen, F., Pi, G., & Li, Y., "Predicting CME arrival time at Earth based on deep learning", *Space Weather*, 22, e2024SW003951, 2024b.

### Instrument & data / 기기 및 데이터
- Brueckner, G. E., Howard, R. A., Koomen, M. J., et al., "The Large Angle Spectroscopic Coronagraph (LASCO)", *Solar Physics*, 162, 357, 1995.
- Kaiser, M. L., "The STEREO Mission: An Overview", *Advances in Space Research*, 36, 1483, 2005.
- Howard, R. A., Moses, J. D., Vourlidas, A., et al., "Sun Earth Connection Coronal and Heliospheric Investigation (SECCHI)", *Space Science Reviews*, 136, 67, 2008.
- Yashiro, S., Gopalswamy, N., Michalek, G., et al., "A catalog of white light coronal mass ejections observed by the SOHO spacecraft", *JGRA*, 109, A07105, 2004.

### CME-storm physics / CME-폭풍 물리학
- Gary, D. E., Gergely, T. E., & Kundu, M. R., "Solar coronal magnetic fields", *Science*, 234, 1486, 1986.
- Webb, D. F., & Howard, T. A., "Coronal Mass Ejections: Observations", *Living Reviews in Solar Physics*, 9, 3, 2012.
- Tsurutani, B. T., & Gonzalez, W. D., "The cause of high-intensity long-duration continuous AE activity (HILDCAAs)", *Planet. Space Sci.*, 35, 405, 1987.
- Eastwood, J. P., Biffis, E., Hapgood, M. A., et al., "The Economic Impact of Space Weather", *Risk Analysis*, 37, 206, 2017.
- Burkepile, J. T., Hundhausen, A. J., Stanger, A. L., et al., "Role of projection effects on solar coronal mass ejection properties", *JGRA*, 109, A03103, 2004.
- Yeh, C.-T., Ding, M. D., & Chen, P. F., "Statistical study of coronal mass ejections", *Solar Physics*, 229, 313, 2005.
- Jang, S., Moon, Y.-J., Kim, R.-S., Lee, H., & Cho, K.-S., "Comparison between 2D and 3D parameters of CMEs", *ApJ*, 821, 95, 2016.
- Gopalswamy, N., Yashiro, S., Stenborg, G., et al., "An Empirical Approach to Predict the Geomagnetic Storms Associated with CMEs", *Earth, Moon, and Planets*, 104, 295, 2009.

### Arrival-time prediction / 도착 시각 예측
- Sudar, D., Vrsnak, B., & Dumbovic, M., "Predicting coronal mass ejections transit times to Earth with neural network", *MNRAS*, 456, 1542, 2016.
- Liu, J., Ye, Y., Shen, C., Wang, Y., & Erdelyi, R., "A new tool for CME arrival time prediction using machine learning", *ApJ*, 855, 109, 2018.
- Riley, P., Mays, M. L., Andries, J., et al., "Forecasting the Arrival Time of Coronal Mass Ejections", *Space Weather*, 16, 1245, 2018.
- Kay, C., Palmerio, E., Riley, P., et al., "Updating Measures of CME Arrival Time Errors", *Space Weather*, 22, e2024SW003951, 2024.
- Li, X., Zheng, Y., Wang, X., & Wang, L., "Predicting Solar Energetic Particles Using SDO/HMI Vector Magnetic Data Products and a Bidirectional LSTM Network", *ApJ*, 891, 10, 2020.

### Foundational / 기초
- Camporeale, E., "The Challenge of Machine Learning in Space Weather: Nowcasting and Forecasting", *Space Weather*, 17, 1166, 2019.
- Schmidhuber, J., "Deep learning in neural networks: An overview", *Neural Networks*, 61, 85, 2014.
