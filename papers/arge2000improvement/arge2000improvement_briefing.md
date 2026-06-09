---
title: "Pre-Reading Briefing: An Empirical Solar Wind Model as a Basis for Space Weather Prediction (WSA)"
paper_id: "18_arge_2000"
topic: Space_Weather
date: 2026-04-17
type: briefing
---

# An Empirical Solar Wind Model as a Basis for Space Weather Prediction (WSA): Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Arge, C. N. and V. J. Pizzo (2000), "An Empirical Solar Wind Model as a Basis for Space Weather Prediction," *J. Geophys. Res.*, 105(A5), 10465–10479, doi:10.1029/1999JA000262
**Author(s)**: C. Nick Arge, Victor J. Pizzo
**Year**: 2000

---

## 1. 핵심 기여 / Core Contribution

이 논문은 **Wang-Sheeley-Arge (WSA) 모델**을 도입하여, 태양 광구(photosphere)의 자기장 관측으로부터 태양풍 속도를 예측하는 개선된 경험적 방법을 제시합니다. 기존의 Wang-Sheeley (WS) 모델이 coronal hole boundary 근처에서 태양풍 속도를 과소평가하는 문제를 해결하기 위해, **flux tube expansion factor ($f_s$)**에 더해 **source surface 위의 footpoint과 가장 가까운 coronal hole boundary까지의 거리($\theta_b$)**를 새로운 매개변수로 도입했습니다. 이 두 변수의 조합으로 태양풍 속도를 더 정확하게 매핑할 수 있게 되었으며, 이 모델은 현재 NOAA/SWPC의 ENLIL 시스템에 태양풍 입력을 제공하는 **운용 우주기상 예보의 핵심 구성 요소**입니다.

This paper introduces the **Wang-Sheeley-Arge (WSA) model**, an improved empirical method for predicting solar wind speed from photospheric magnetic field observations. To address the shortcoming of the original Wang-Sheeley (WS) model—which underestimated solar wind speed near coronal hole boundaries—the authors add a new parameter: **the angular distance from the footpoint on the source surface to the nearest coronal hole boundary ($\theta_b$)**. Combined with the existing **flux tube expansion factor ($f_s$)**, this two-parameter mapping produces significantly more accurate solar wind speed predictions. The WSA model is now a **cornerstone of operational space weather forecasting**, providing ambient solar wind input to NOAA/SWPC's ENLIL system.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

1990년대 후반, 우주기상 예보는 크게 두 가지 도전에 직면해 있었습니다. 첫째, Parker (1958, Paper #4)가 예측하고 확인된 태양풍이 **왜 속도가 다양한지** — 고속풍(~700-800 km/s)과 저속풍(~300-400 km/s)이 왜 존재하는지를 이해하고 예측하는 것이 핵심 과제였습니다. 둘째, 이러한 태양풍 구조를 **실시간으로 예보**하여 지자기 폭풍(Burton et al. 1975, Paper #11의 Dst 예측)에 대비할 필요가 있었습니다.

In the late 1990s, space weather forecasting faced two major challenges. First, understanding and predicting **why solar wind speed varies** — why fast wind (~700-800 km/s) and slow wind (~300-400 km/s) exist — was a key problem since Parker (1958, Paper #4) predicted the solar wind's existence. Second, there was an operational need to **forecast this solar wind structure in real time** to prepare for geomagnetic storms (as quantified by Burton et al. 1975, Paper #11).

Wang & Sheeley (1990)는 **flux tube expansion factor**와 태양풍 속도 사이의 반비례 관계를 발견했습니다: 자기장 flux tube가 적게 팽창하는 곳(coronal hole 중심부)에서 빠른 태양풍이 나오고, 많이 팽창하는 곳(coronal hole 경계)에서 느린 태양풍이 나옵니다. 그러나 이 WS 모델은 coronal hole boundary 바로 근처에서 관측되는 빠른 태양풍을 재현하지 못하는 체계적 오류가 있었습니다. Arge & Pizzo (2000)는 이 문제를 해결했습니다.

Wang & Sheeley (1990) discovered an **inverse relationship between flux tube expansion factor and solar wind speed**: regions where magnetic flux tubes expand less (coronal hole centers) produce fast wind, while regions of greater expansion (coronal hole boundaries) produce slow wind. However, this WS model had a systematic error: it failed to reproduce the fast solar wind observed just near coronal hole boundaries. Arge & Pizzo (2000) solved this problem.

### 타임라인 / Timeline

```
1958  Parker           태양풍 존재 예측 / Solar wind existence predicted (Paper #4)
1962  Mariner 2        태양풍 최초 직접 확인 / First direct solar wind confirmation
1969  Altschuler &     PFSS 모델 개발 / PFSS model developed
      Newkirk
1973  Munro &          Coronal holes ↔ 고속 태양풍 관계 확립
      Withbroe         Coronal holes ↔ fast solar wind link established
1975  Burton et al.    Dst 경험적 공식 / Empirical Dst formula (Paper #11)
1990  Wang & Sheeley   Flux tube expansion ↔ 태양풍 속도 반비례 관계
                       Inverse relation: expansion factor ↔ solar wind speed
1995  Wang & Sheeley   WS 모델 개선 / WS model refinements
>>>>  2000  Arge & Pizzo     WSA 모델: θ_b 매개변수 추가 <<<< 이 논문
                       WSA model: added θ_b parameter  <<<< THIS PAPER
2003  Odstrcil         ENLIL 3D MHD 모델과 WSA 결합
                       ENLIL 3D MHD model coupled with WSA
2004  Arge et al.      WSA 모델 추가 개선 및 검증 / Further WSA improvements
현재  NOAA/SWPC        WSA-ENLIL 운용 예보 시스템
      present          WSA-ENLIL operational forecasting system
```

---

## 3. 필요한 배경 지식 / Prerequisites

### Potential Field Source Surface (PFSS) 모델 / PFSS Model

PFSS 모델은 태양 광구의 자기장(magnetogram)으로부터 코로나 자기장 구조를 근사적으로 계산합니다. 핵심 가정:
- 광구($r = R_\odot$)와 source surface($r = R_{ss} \approx 2.5 R_\odot$) 사이에서 전류가 없음($\nabla \times \mathbf{B} = 0$)
- Source surface에서 자기장이 순수 방사형(radial)

The PFSS model approximates the coronal magnetic field structure from photospheric magnetic field observations (magnetograms). Key assumptions:
- Current-free ($\nabla \times \mathbf{B} = 0$) between the photosphere ($r = R_\odot$) and source surface ($r = R_{ss} \approx 2.5 R_\odot$)
- Purely radial magnetic field at the source surface

이 모델로 **open field lines** (태양풍이 탈출하는 열린 자기력선)과 **closed field lines**을 구분할 수 있습니다.

This allows distinguishing **open field lines** (where solar wind escapes) from **closed field lines**.

### Flux Tube Expansion Factor ($f_s$)

Flux tube expansion factor는 광구에서 source surface까지 자기 flux tube가 얼마나 팽창하는지를 나타냅니다:

The flux tube expansion factor quantifies how much a magnetic flux tube expands from the photosphere to the source surface:

$$f_s = \frac{R_\odot^2}{R_{ss}^2} \cdot \frac{B(R_\odot)}{B(R_{ss})}$$

- $f_s \approx 1$: 최소 팽창 (coronal hole 중심) → 고속풍 / Minimal expansion (coronal hole center) → fast wind
- $f_s \gg 1$: 대폭 팽창 (coronal hole 경계, streamer 근처) → 저속풍 / Large expansion (near boundaries/streamers) → slow wind

### Coronal Holes

Coronal holes은 태양 코로나에서 자기장이 열린(open) 영역으로, X-선/EUV 영상에서 어둡게 보입니다. 고속 태양풍의 원천입니다.

Coronal holes are regions of open magnetic field in the solar corona, appearing dark in X-ray/EUV images. They are the source of fast solar wind.

### Parker의 태양풍 모델 (Paper #4)

Parker (1958)는 코로나가 정수압 평형(hydrostatic equilibrium)을 유지할 수 없어 초음속으로 팽창하는 태양풍이 불가피하다는 것을 보였습니다. WSA 모델은 이 태양풍의 **속도 구조**를 경험적으로 예측합니다.

Parker (1958) showed that the corona cannot maintain hydrostatic equilibrium and must expand as a supersonic solar wind. The WSA model empirically predicts the **speed structure** of this solar wind.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **PFSS (Potential Field Source Surface)** | 광구 자기장에서 코로나 자기장을 전류 없는 가정으로 외삽하는 모델 / Model extrapolating coronal field from photospheric data assuming current-free corona |
| **Source surface** | PFSS에서 자기장이 순수 방사형이 되는 구면 ($\approx 2.5 R_\odot$) / Spherical surface where $B$ becomes purely radial |
| **Flux tube expansion factor ($f_s$)** | 자기 flux tube의 광구-source surface 간 팽창비 / Ratio of flux tube cross-section expansion from photosphere to source surface |
| **Coronal hole** | 열린 자기장 영역, 고속 태양풍 원천 / Open magnetic field region, source of fast solar wind |
| **Coronal hole boundary distance ($\theta_b$)** | Source surface에서 footpoint과 가장 가까운 coronal hole 경계까지의 각거리 / Angular distance from footpoint to nearest coronal hole boundary on source surface |
| **Wang-Sheeley (WS) model** | $f_s$만으로 태양풍 속도를 예측하는 원래 경험적 모델 / Original empirical model predicting wind speed from $f_s$ alone |
| **WSA model** | $f_s$와 $\theta_b$를 결합한 개선 모델 (이 논문) / Improved model combining $f_s$ and $\theta_b$ (this paper) |
| **Magnetogram** | 태양 광구의 시선 방향 자기장 관측 / Observation of line-of-sight magnetic field on the photosphere |
| **Carrington rotation** | 태양의 1자전 (~27.27일), synoptic map의 기본 단위 / One solar rotation (~27.27 days), basic unit for synoptic maps |
| **Synoptic map** | 한 Carrington rotation 동안의 광구 자기장을 합성한 전구 지도 / Full-Sun photospheric magnetic field map composited over one Carrington rotation |
| **ENLIL** | WSA 출력을 입력으로 사용하는 3D MHD heliospheric 모델 / 3D MHD heliospheric model using WSA output as inner boundary |
| **Stream interaction region (SIR)** | 고속풍과 저속풍이 상호작용하는 영역 / Region where fast and slow wind interact |

---

## 5. 수식 미리보기 / Equations Preview

### 5.1 기존 Wang-Sheeley 관계 / Original Wang-Sheeley Relation

$$V_{sw} = V_0 + \frac{V_1}{(1 + f_s)^{\alpha}}$$

- $V_{sw}$: 예측된 태양풍 속도 / Predicted solar wind speed
- $V_0$: 최소 태양풍 속도 (~250-300 km/s) / Minimum solar wind speed
- $V_1$: 속도 범위 (~500 km/s) / Speed range
- $f_s$: flux tube expansion factor
- $\alpha$: 경험적 지수 / Empirical exponent

$f_s$가 작으면 (coronal hole 중심) → $V_{sw}$ 높음, $f_s$가 크면 → $V_{sw}$ 낮음.

Small $f_s$ (coronal hole center) → high $V_{sw}$; large $f_s$ → low $V_{sw}$.

### 5.2 WSA 모델의 개선된 관계 / Improved WSA Relation

$$V_{sw} = V_0 + \frac{V_1}{(1 + f_s)^{\alpha}} \left[ 1 - \beta \cdot \exp\left(-\left(\frac{\theta_b}{\gamma}\right)^{\delta}\right) \right]^3$$

새로 추가된 항이 핵심:
- $\theta_b$: source surface 위에서 footpoint과 가장 가까운 coronal hole boundary까지의 각거리
- $\beta, \gamma, \delta$: 경험적 매개변수

The newly added term is the key:
- $\theta_b$: angular distance from footpoint to nearest coronal hole boundary on source surface
- $\beta, \gamma, \delta$: empirical parameters

**물리적 의미**: coronal hole 경계 가까이($\theta_b$ 작음)에서도 $f_s$가 클 수 있지만, $\theta_b$가 작으면 속도 감소 보정이 줄어들어 빠른 풍속을 허용합니다. 경계에서 멀면($\theta_b$ 큼) 보정 효과가 없어져 기존 WS 관계를 따릅니다.

**Physical meaning**: Near coronal hole boundaries (small $\theta_b$), $f_s$ can be large, but the small $\theta_b$ reduces the speed penalty, allowing faster wind. Far from boundaries (large $\theta_b$), the correction vanishes and the original WS relation applies.

### 5.3 Flux Tube Expansion Factor 정의 / Definition

$$f_s = \frac{(R_\odot / R_{ss})^2}{|B_r(R_{ss})| / |B_r(R_\odot)|}$$

여기서 $B_r$는 PFSS 모델에서 계산된 방사 방향 자기장입니다.

Where $B_r$ is the radial magnetic field component computed from the PFSS model.

### 5.4 PFSS 포텐셜장 / PFSS Potential Field

$$\nabla^2 \Phi = 0 \quad \text{for} \quad R_\odot \leq r \leq R_{ss}$$

경계조건 / Boundary conditions:
- $B_r(R_\odot) = -\partial\Phi/\partial r|_{R_\odot}$ (magnetogram에서)
- $B_\theta(R_{ss}) = B_\phi(R_{ss}) = 0$ (순수 방사형)

---

## 6. 읽기 가이드 / Reading Guide

### 추천 읽기 순서 / Recommended Reading Order

1. **Abstract & Introduction (§1)**: 논문의 목표와 WS 모델의 한계를 파악 / Understand the goal and limitations of the WS model
2. **Model Description (§2)**: WSA의 핵심 — $\theta_b$ 매개변수 도입과 경험적 관계식에 집중 / Focus on the $\theta_b$ parameter and the empirical speed relation
3. **PFSS Setup (§2 일부)**: PFSS 계산 방법과 synoptic map 처리 방법 / How PFSS is computed and synoptic maps are processed
4. **Results (§3-4)**: 모델 검증 — 관측된 태양풍 속도와 비교 / Model validation against observed solar wind speed
5. **Discussion (§5)**: 모델의 물리적 해석과 한계 / Physical interpretation and limitations

### 주목할 포인트 / Key Points to Watch

- **Figure 분석**: 각 그림에서 WS 모델(기존)과 WSA 모델(개선)의 차이를 비교하세요 / Compare WS vs WSA predictions in each figure
- **Coronal hole boundary 근처 개선**: $\theta_b$가 어떻게 boundary 근처의 속도 예측을 개선하는지 주목 / Watch how $\theta_b$ improves boundary-region speed predictions
- **매개변수 결정**: 경험적 매개변수들이 어떤 데이터로 fitting되었는지 확인 / Note what data the empirical parameters are fitted to
- **Carrington rotation별 비교**: 다양한 solar activity 조건에서의 모델 성능 / Model performance across different solar activity conditions

### 건너뛰어도 되는 부분 / Sections to Skim

- PFSS의 상세 수치 구현 (spherical harmonics 전개) — 개념만 이해하면 충분 / Detailed PFSS numerical implementation — conceptual understanding suffices

---

## 7. 현대적 의의 / Modern Significance

### 운용 우주기상 예보 / Operational Space Weather Forecasting

WSA 모델은 현재 **NOAA Space Weather Prediction Center (SWPC)**에서 운용하는 **WSA-ENLIL** 시스템의 핵심 구성 요소입니다. WSA가 태양풍 배경(ambient solar wind)을 제공하고, ENLIL이 이를 3D MHD로 태양에서 지구까지 전파시킵니다. CME가 발사되면 ENLIL에 "cone model"로 삽입하여 지구 도달 시간을 예측합니다.

The WSA model is a **core component of the WSA-ENLIL system** operated by **NOAA's Space Weather Prediction Center (SWPC)**. WSA provides the ambient solar wind background, and ENLIL propagates it via 3D MHD from the Sun to Earth. When a CME is launched, it is inserted into ENLIL as a "cone model" to predict Earth arrival time.

### 후속 발전 / Subsequent Developments

- **WSA 개선**: Arge et al. (2003, 2004) — 추가 매개변수 조정, 더 많은 Carrington rotation 검증
- **ENLIL 결합**: Odstrcil (2003) — WSA 출력을 ENLIL 내부 경계로 사용
- **앙상블 예보**: 최근에는 WSA의 입력 불확실성을 고려한 앙상블 예보가 연구됨
- **머신러닝 대안**: WSA의 경험적 관계를 ML로 대체/보완하려는 시도들이 진행 중

### 이 논문이 중요한 이유 / Why This Paper Matters

1. **실무 영향력**: 전 세계 우주기상 예보 기관의 표준 도구 / Standard tool for global space weather forecasting agencies
2. **개념적 단순함**: 복잡한 MHD 시뮬레이션 없이 자기장 관측만으로 태양풍 예측 / Predicts solar wind from magnetic observations without full MHD
3. **확장성**: ENLIL, HAFv2 등 다양한 heliospheric 모델의 입력으로 사용 가능 / Input to various heliospheric models
4. **교육적 가치**: 코로나 자기장 → 태양풍 속도의 물리적 연결 고리를 명확히 보여줌 / Clearly demonstrates the corona-to-wind physical connection

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
