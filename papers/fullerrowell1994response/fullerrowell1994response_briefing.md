---
title: "Pre-Reading Briefing: Response of the Thermosphere and Ionosphere to Geomagnetic Storms"
paper_id: "16_fullerrowell_1994"
topic: Space_Weather
date: 2026-04-16
type: briefing
---

# Response of the Thermosphere and Ionosphere to Geomagnetic Storms: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Fuller-Rowell, T. J., M. V. Codrescu, R. J. Moffett, and S. Quegan, "Response of the Thermosphere and Ionosphere to Geomagnetic Storms", *J. Geophys. Res.*, 99(A3), 3893–3914, 1994. DOI: 10.1029/93JA02015
**Author(s)**: Timothy J. Fuller-Rowell, Mihail V. Codrescu, R. J. Moffett, S. Quegan
**Year**: 1994

---

## 1. 핵심 기여 / Core Contribution

이 논문은 지자기 폭풍 동안 열권(thermosphere)과 전리층(ionosphere)이 어떻게 반응하는지를 포괄적으로 시뮬레이션한 선구적 연구입니다. NCAR Thermosphere-Ionosphere General Circulation Model (TIGCM)을 사용하여 폭풍 시 고위도 에너지 주입이 열권의 중성 조성(O/N₂ 비율), 온도, 바람 패턴을 어떻게 변화시키고, 이러한 변화가 전리층 전자 밀도에 어떤 영향을 미치는지 체계적으로 분석했습니다.

This paper is a pioneering comprehensive simulation study of how the thermosphere and ionosphere respond to geomagnetic storms. Using the NCAR Thermosphere-Ionosphere General Circulation Model (TIGCM), it systematically analyzed how high-latitude energy input during storms alters neutral composition (O/N₂ ratio), temperature, and wind patterns in the thermosphere, and how these changes affect ionospheric electron density. The key insight is that storm-time thermospheric composition changes — particularly the decrease in the O/N₂ ratio at mid-latitudes — are the primary driver of the negative ionospheric storm effect (reduced F-region electron density).

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

1990년대 초, 지자기 폭풍의 전리층 효과는 관측적으로 잘 알려져 있었지만 물리적 메커니즘은 논쟁 중이었습니다. 폭풍 시 전리층 F-영역의 전자 밀도가 감소하는 "음(negative) 폭풍 효과"가 왜 발생하는지에 대해 여러 설명이 경쟁하고 있었습니다: 열권 바람에 의한 플라즈마 수송, 조성 변화(O/N₂ 비율 감소), 전기장 변화 등.

By the early 1990s, ionospheric effects of geomagnetic storms were well-documented observationally, but the physical mechanisms were debated. Multiple explanations competed to explain the "negative storm effect" (decreased F-region electron density during storms): plasma transport by thermospheric winds, composition changes (O/N₂ ratio decrease), and electric field variations.

Fuller-Rowell과 동료들은 TIGCM이라는 최초의 자기일관적(self-consistent) 열권-전리층 결합 모델을 사용하여 이 문제를 해결하려 했습니다. Richmond(공저자)의 AMIE 기법(논문 #13)을 통해 현실적인 고위도 전기장/강수 패턴을 모델에 입력할 수 있었고, Gonzalez et al.(논문 #15)이 정의한 폭풍 분류 체계 속에서 이 연구가 위치합니다.

### 타임라인 / Timeline

```
1970s   Jacchia empirical thermosphere models (관측 기반 열권 모델)
        ↓
1980    Dickinson et al. — First NCAR TGCM (열권 대순환 모델)
        ↓
1982    Rishbeth & Garriott — O/N₂ hypothesis proposed (O/N₂ 가설 제안)
        ↓
1987    Roble et al. — TIGCM (열권-전리층 결합 GCM)
        ↓
1988    Richmond & Kamide — AMIE technique (AMIE 기법, Paper #13)
        ↓
1992    Prölss — Comprehensive review of ionospheric storm effects
        ↓
★ 1994  Fuller-Rowell et al. — THIS PAPER (thermosphere-ionosphere storm response)
        ↓
1994    Gonzalez et al. — Storm classification (폭풍 분류, Paper #15)
        ↓
2000s   CTIPe, WACCM-X — Next-generation coupled models
```

---

## 3. 필요한 배경 지식 / Prerequisites

### 열권 물리학 / Thermospheric Physics

열권(~90–500 km)은 태양 EUV와 지자기 활동에 의해 가열됩니다. 주요 성분은 원자산소(O), 분자질소(N₂), 분자산소(O₂)입니다. 높이에 따른 이들의 혼합비가 "조성(composition)"을 결정하며, 특히 O/N₂ 비율이 전리층 F-영역 밀도의 핵심 제어 인자입니다.

The thermosphere (~90–500 km) is heated by solar EUV and geomagnetic activity. Its main constituents are atomic oxygen (O), molecular nitrogen (N₂), and molecular oxygen (O₂). Their mixing ratios as a function of altitude define the "composition," and the O/N₂ ratio is the key controlling factor for F-region ionospheric density.

### 열권-전리층 결합 / Thermosphere-Ionosphere Coupling

- **O가 많으면** → 전리 생성 증가 (O + hν → O⁺ + e⁻) → 전자 밀도 증가
- **N₂가 많으면** → 재결합 증가 (O⁺ + N₂ → NO⁺ + N) → 전자 밀도 감소
- 따라서 O/N₂ 비율 감소 = 음의 전리층 폭풍 효과

- **More O** → increased ionization (O + hν → O⁺ + e⁻) → higher electron density
- **More N₂** → increased recombination (O⁺ + N₂ → NO⁺ + N) → lower electron density
- Therefore, decreased O/N₂ ratio = negative ionospheric storm effect

### 선행 논문 연결 / Connection to Prerequisites

- **Paper #13 (Richmond 1988, AMIE)**: 이 논문에서 사용하는 고위도 전기장 패턴의 관측적 기반을 제공합니다.
- **Paper #15 (Gonzalez 1994)**: 폭풍의 정의와 분류 체계를 확립하여, 이 논문이 분석하는 "지자기 폭풍"의 정량적 기준을 제공합니다.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **Thermosphere / 열권** | 고도 ~90–500 km의 대기층. 태양 EUV에 의해 가열되며, 위성 궤도가 존재하는 영역 / Atmospheric layer ~90–500 km heated by solar EUV; where satellites orbit |
| **O/N₂ ratio / O/N₂ 비율** | 원자산소 대 분자질소 칼럼 밀도 비. 전리층 F-영역 밀도의 핵심 제어 인자 / Column density ratio of atomic oxygen to molecular nitrogen; key controller of F-region density |
| **TIGCM** | Thermosphere-Ionosphere General Circulation Model. NCAR의 자기일관적 열권-전리층 결합 수치 모델 / Self-consistent coupled thermosphere-ionosphere numerical model from NCAR |
| **Negative storm effect / 음의 폭풍 효과** | 지자기 폭풍 시 F-영역 전자 밀도가 감소하는 현상 / Decrease in F-region electron density during geomagnetic storms |
| **Positive storm effect / 양의 폭풍 효과** | 폭풍 초기에 일부 지역에서 전자 밀도가 증가하는 현상 / Increase in electron density in certain regions during early storm phase |
| **Joule heating / 줄 가열** | 이온-중성 충돌을 통한 전기에너지 → 열에너지 변환 / Conversion of electrical to thermal energy via ion-neutral collisions |
| **Auroral precipitation / 오로라 강수** | 자기권에서 열권으로 돌진하는 에너지 입자 / Energetic particles precipitating from magnetosphere into thermosphere |
| **Composition bulge / 조성 돌출** | 폭풍 시 고위도에서 형성되어 적도 쪽으로 이동하는 N₂-풍부 영역 / N₂-enriched region forming at high latitudes and migrating equatorward during storms |
| **Scale height / 스케일 높이** | 대기 밀도가 e⁻¹배로 감소하는 고도 차이. 온도에 비례 / Altitude difference over which atmospheric density decreases by factor e⁻¹; proportional to temperature |
| **Satellite drag / 위성 항력** | 열권 밀도 증가에 의한 위성 궤도 감속. 우주기상의 핵심 실용적 영향 / Orbital deceleration of satellites due to thermospheric density increase; key practical space weather impact |

---

## 5. 수식 미리보기 / Equations Preview

### 5.1 이온 생성과 소멸 / Ion Production and Loss

F-영역에서 주요 이온(O⁺)의 생성과 소멸:

$$q(O^+) = n(O) \cdot I \cdot \sigma_{O}$$

여기서 $n(O)$는 원자산소 밀도, $I$는 태양 EUV 플럭스, $\sigma_O$는 광이온화 단면적입니다.

손실 반응(loss reaction):

$$O^+ + N_2 \rightarrow NO^+ + N \quad (k_1)$$
$$O^+ + O_2 \rightarrow O_2^+ + O \quad (k_2)$$

손실률은 $L = k_1 \cdot n(N_2) + k_2 \cdot n(O_2)$이므로, **N₂ 밀도가 증가하면 O⁺ 손실이 증가**합니다.

### 5.2 정수압 평형과 스케일 높이 / Hydrostatic Equilibrium and Scale Height

$$H_i = \frac{k_B T}{m_i g}$$

여기서 $H_i$는 성분 $i$의 스케일 높이, $k_B$는 볼츠만 상수, $T$는 온도, $m_i$는 분자 질량, $g$는 중력가속도입니다.

온도가 상승하면 → 스케일 높이 증가 → 무거운 성분(N₂)이 더 높은 고도까지 올라감 → O/N₂ 비율 감소.

When temperature increases → scale height increases → heavier species (N₂) extend to higher altitudes → O/N₂ ratio decreases.

### 5.3 줄 가열률 / Joule Heating Rate

$$Q_J = \sigma_P |\mathbf{E} + \mathbf{V}_n \times \mathbf{B}|^2$$

여기서 $\sigma_P$는 Pedersen 전도도, $\mathbf{E}$는 전기장, $\mathbf{V}_n$은 중성풍, $\mathbf{B}$는 자기장입니다. 폭풍 시 고위도에서 $E$가 크게 증가하면 줄 가열이 급증합니다.

### 5.4 운동량 방정식 (중성풍) / Momentum Equation (Neutral Wind)

$$\frac{\partial \mathbf{V}_n}{\partial t} + (\mathbf{V}_n \cdot \nabla)\mathbf{V}_n + 2\boldsymbol{\Omega} \times \mathbf{V}_n = -\frac{\nabla p}{\rho} + \nu_{ni}(\mathbf{V}_i - \mathbf{V}_n) + \mathbf{g} + \text{viscosity}$$

이온 드래그 항 $\nu_{ni}(\mathbf{V}_i - \mathbf{V}_n)$이 이온-중성 결합의 핵심입니다. 폭풍 시 이온 대류가 강해지면 중성풍도 가속됩니다.

---

## 6. 읽기 가이드 / Reading Guide

### 집중해서 읽을 부분 / Focus Sections

1. **Introduction (§1)**: 열권-전리층 폭풍 반응에 대한 기존 연구 정리. 경쟁 가설들을 파악하세요.
   - Read for: competing hypotheses about storm effects
2. **Model description (§2)**: TIGCM의 구조와 입력 조건. 모델의 한계와 가정을 이해하세요.
   - Read for: model architecture, assumptions, and limitations
3. **Storm simulation results (§3–4)**: 핵심 결과 섹션. 조성 변화(O/N₂), 온도, 바람의 시공간 진화를 따라가세요.
   - **가장 중요**: O/N₂ 비율의 시간-위도 변화 그림
   - Read for: spatiotemporal evolution of composition, temperature, and winds
4. **Ionospheric response (§5)**: 열권 변화가 전리층에 미치는 영향. 양/음의 폭풍 효과 메커니즘 구분.
   - Read for: how thermospheric changes translate to ionospheric effects

### 빠르게 훑어볼 부분 / Skim Sections

- 수치 기법 상세 (numerical methods details)
- 민감도 테스트의 모든 케이스 (scan for key conclusions)

### 읽기 전략 / Reading Strategy

이 논문은 시뮬레이션 결과 중심이므로, **그림(figures)을 먼저 살펴보는 것**을 추천합니다. 특히 O/N₂ 비율, 온도, 바람 패턴의 시간 진화를 보여주는 컬러 맵을 주목하세요. 텍스트는 이 그림들을 설명하고 해석하는 구조입니다.

This paper is simulation-result-centric, so **start by scanning the figures**. Pay special attention to color maps showing temporal evolution of O/N₂, temperature, and wind patterns. The text is structured to explain and interpret these figures.

---

## 7. 현대적 의의 / Modern Significance

이 논문은 현대 우주기상 예보의 핵심 물리 메커니즘을 확립했습니다:

This paper established the core physical mechanisms for modern space weather forecasting:

1. **위성 항력 예보 / Satellite Drag Forecasting**: 열권 밀도 변화 모델링은 저궤도 위성(ISS, Starlink 등) 궤도 예측의 핵심. 2022년 SpaceX가 지자기 폭풍으로 Starlink 위성 40기를 잃은 사건이 이 물리학의 실제 중요성을 보여줌.
2. **GPS/GNSS 오차 / Navigation Errors**: 전리층 폭풍 효과는 GPS 정확도에 직접 영향.
3. **후속 모델 발전 / Model Evolution**: TIGCM → TIE-GCM → WACCM-X로 이어지는 모델 계보의 핵심 검증 연구.
4. **O/N₂ 관측 / Composition Monitoring**: GUVI/TIMED 위성의 O/N₂ 비율 관측은 이 논문의 이론적 예측을 직접 검증하는 도구.

---

## Q&A

### Q1: F층(F-region)이란? / What is the F-layer?

#### 전리층 구조 / Ionosphere Structure

전리층은 태양 자외선/X선에 의해 대기가 이온화된 영역으로, 고도에 따라 여러 층으로 나뉩니다.

The ionosphere is the region where the atmosphere is ionized by solar UV/X-rays, divided into layers by altitude.

| 층 / Layer | 고도 / Altitude | 주간 특성 / Daytime Characteristics |
|---|---|---|
| **D층** | ~60–90 km | 약한 이온화, 야간 소멸 / Weak ionization, disappears at night |
| **E층** | ~90–150 km | 중간 이온화 / Moderate ionization |
| **F1층** | ~150–200 km | 주간에만 존재 / Daytime only |
| **F2층** | ~200–500 km+ | **전리층 최대 전자밀도 / Maximum electron density** |

#### F2층이 중요한 이유 / Why F2 Matters

**F2층**은 전리층에서 전자 밀도가 가장 높은 곳(peak: ~300 km 부근, $N_mF2$)이며, 이 논문에서 다루는 "전리층 폭풍 효과"가 주로 나타나는 영역입니다.

The **F2 layer** has the highest electron density in the ionosphere (peak ~300 km, $N_mF2$) and is where the "ionospheric storm effects" discussed in this paper primarily manifest.

핵심 화학 반응 / Key chemistry:

- **생성 / Production**: $O + h\nu \rightarrow O^+ + e^-$ (원자산소의 광이온화 / photoionization of atomic oxygen)
- **소멸 / Loss**: $O^+ + N_2 \rightarrow NO^+ + N$ (분자질소와 반응 / reaction with molecular nitrogen)

F2층의 전자 밀도는 **O와 N₂의 상대적 비율**에 의해 결정됩니다:

F2-layer electron density is determined by the **relative ratio of O to N₂**:

- **O가 많으면 / More O** → 이온 생성 ↑ → 전자 밀도 ↑ / ion production ↑ → electron density ↑
- **N₂가 많으면 / More N₂** → 이온 소멸 ↑ → 전자 밀도 ↓ / ion loss ↑ → electron density ↓

#### F2층이 특별한 이유 / Why F2 is Anomalous

아래층(D, E, F1)은 **광화학 평형(photochemical equilibrium)** 상태라서 태양이 지면 빠르게 재결합하여 소멸합니다. 하지만 F2층은:

Lower layers (D, E, F1) are in **photochemical equilibrium** and quickly recombine after sunset. However, the F2 layer:

1. **고도가 높아** 대기 밀도가 낮음 → 재결합 속도가 느림 / High altitude → low atmospheric density → slow recombination
2. **야간에도 상당한 전자 밀도 유지** / Maintains significant electron density even at night
3. 화학뿐 아니라 **수송(transport)** — 바람, 확산, 전기장에 의한 플라즈마 이동 — 이 중요 / Transport (winds, diffusion, electric fields) plays a major role alongside chemistry

이것이 F2층을 "anomalous layer"라고 부르는 이유입니다. 단순한 Chapman 이론(태양 천정각만으로 예측)으로는 설명이 안 됩니다.

This is why F2 is called the "anomalous layer" — simple Chapman theory (predicting from solar zenith angle alone) cannot explain it.

#### 이 논문과의 연결 / Connection to This Paper

Fuller-Rowell et al. (1994)의 핵심 주장:

> 지자기 폭풍 → 고위도 가열 → 열권 온도 상승 → N₂ 스케일 높이 증가 → **O/N₂ 비율 감소** → F2층 전자 밀도 감소 (음의 폭풍 효과)

> Geomagnetic storm → high-latitude heating → thermospheric temperature rise → N₂ scale height increase → **O/N₂ ratio decrease** → F2-layer electron density decrease (negative storm effect)

---

### Q2: TEC와 F2층 전자밀도의 관계 / TEC vs. F2-layer Electron Density

#### TEC의 정의 / Definition of TEC

TEC(Total Electron Content)는 수신기에서 위성까지 경로 위의 **전체 칼럼 전자 수**입니다.

TEC is the **total number of electrons in a column** from receiver to satellite.

$$TEC = \int_{\text{receiver}}^{\text{satellite}} n_e \, ds \quad [\text{단위 / unit: el/m}^2, \; 1 \text{ TECU} = 10^{16} \text{ el/m}^2]$$

즉, D층부터 위성 고도까지 **모든 층의 전자를 적분**한 값입니다.

This integrates electrons from **all layers** from the D-layer to satellite altitude.

#### F2층의 압도적 기여 / F2's Dominant Contribution

| 층 / Layer | TEC 기여 비율 / Contribution to TEC |
|---|---|
| D, E층 | ~5–10% |
| F1층 | ~10–15% |
| **F2층** | **~60–80%** |
| Topside (F2 위 / above F2) | ~10–20% |

F2층의 전자 밀도가 압도적으로 높기 때문에 **TEC 변화의 대부분은 F2층 변화에 의해 결정**됩니다. 특히 폭풍 시 TEC가 감소하면, 그것은 거의 F2층의 $N_mF2$ 감소를 반영합니다.

Because F2-layer electron density is overwhelmingly dominant, **most TEC variation is driven by F2-layer changes**. When TEC decreases during storms, it largely reflects a decrease in $N_mF2$.

#### 주의할 경우 / Caveats

그러나 동일시하면 안 되는 상황도 있습니다:

However, there are situations where they should not be equated:

- **플라즈마권 기여 / Plasmasphere contribution**: GPS TEC는 고도 ~20,000 km까지 적분하므로 plasmasphere 전자도 포함 (조용한 시기에 10–20%) / GPS TEC integrates up to ~20,000 km, including plasmaspheric electrons (10–20% during quiet times)
- **Topside 변화 / Topside variations**: 폭풍 시 플라즈마가 위쪽으로 올라가면 $N_mF2$는 줄어도 topside 기여가 늘어 TEC는 덜 변할 수 있음 / During storms, upward plasma transport can reduce $N_mF2$ while increasing topside contribution, moderating TEC change
- **Slab thickness 변화**: TEC/$N_mF2$ 비율(= slab thickness)이 폭풍 시 변하므로, TEC와 $N_mF2$가 항상 같은 방향으로 움직이지는 않음 / The TEC/$N_mF2$ ratio (slab thickness) varies during storms, so TEC and $N_mF2$ do not always co-vary

#### 결론 / Conclusion

> 일상적으로 "TEC 변화 ≈ F2층 전자밀도 변화"라고 생각하는 것은 **1차 근사로 충분히 유효**합니다. 다만 정밀한 분석에서는 구분이 필요합니다.

> As a **first-order approximation**, "TEC change ≈ F2-layer electron density change" is valid. However, precise analysis requires distinguishing between them.
