---
title: "Pre-Reading Briefing: Solar Prominences: Theory and Models"
paper_id: "61"
topic: Living_Reviews_in_Solar_Physics
date: 2026-04-23
type: briefing
---

# Solar Prominences: Theory and Models — Fleshing out the magnetic skeleton: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Gibson, S. E., "Solar Prominences: Theory and Models — Fleshing out the magnetic skeleton", Living Reviews in Solar Physics, 15:7 (2018). DOI: 10.1007/s41116-018-0016-2
**Author(s)**: Sarah E. Gibson (National Center for Atmospheric Research, Boulder)
**Year**: 2018

---

## 1. 핵심 기여 / Core Contribution

**English**: This review synthesizes decades of theoretical and modeling work on solar prominences, framing them as cool (~7,500 K), dense plasma suspended within the hot (~1 MK) corona by magnetic forces. Gibson introduces a "magnetic skeleton" metaphor: the long-lived magnetic scaffold — comprising sheared arcades, flux ropes, and spheromaks — provides the bones (magnetic dips), while dynamic and thermodynamic processes add the flesh and blood (mass condensation, flows, eventual eruption). The review critically compares competing models (Kippenhahn–Schlüter dipped arcade vs. Kuperus–Raadu flux rope; normal vs. inverse polarity; sheared arcade vs. flux rope), connects them to observational signatures (chirality, sigmoids, cavities), and surveys the state of the art in 3D MHD simulations that self-consistently couple magnetism with energy transport.

**한국어**: 이 리뷰는 태양 홍염(prominence)에 대한 수십 년간의 이론 및 모델링 연구를 종합하여, 뜨거운 코로나(~1 MK) 속에 차갑고(~7,500 K) 조밀한 플라즈마가 자기력으로 어떻게 매달려 있는지를 설명한다. Gibson은 "자기 골격(magnetic skeleton)"이라는 은유를 도입하여, 장수명 자기 구조(sheared arcade, flux rope, spheromak)가 뼈대(자기 dip)를 제공하고, 동역학적·열역학적 과정이 살과 피(질량 응축, 유동, 궁극적 폭발)를 더한다고 본다. 리뷰는 경쟁 모델들(Kippenhahn–Schlüter vs. Kuperus–Raadu; normal vs. inverse polarity; sheared arcade vs. flux rope)을 비판적으로 비교하고, 관측 신호(chirality, sigmoid, cavity)와 연결하며, 자기장과 에너지 수송을 자체 일관적으로 결합하는 최신 3D MHD 시뮬레이션을 조망한다.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

**English**: By 2018 the prominence community had accumulated rich multi-wavelength observations from SDO/AIA, Hinode/SOT, MLSO/CoMP, and IRIS, together with increasingly sophisticated 3D numerical simulations. Earlier reviews (Mackay et al. 2010; Labrosse et al. 2010) established the observational and magnetic picture; the companion book *Solar Prominences* (Engvold & Vial 2015) surveyed subtopics. Gibson's task was to update the theory/models chapter and integrate the emergence of thermal-nonequilibrium (TNE) driven formation with 3D MHD treatments that abandon the static force-free assumption.

**한국어**: 2018년까지 홍염 연구 커뮤니티는 SDO/AIA, Hinode/SOT, MLSO/CoMP, IRIS 등 다파장 관측과 정교한 3D 수치 시뮬레이션을 축적했다. 선행 리뷰(Mackay et al. 2010; Labrosse et al. 2010)가 관측 및 자기 구조의 큰 그림을 확립했고, 동반 저서 *Solar Prominences*(Engvold & Vial 2015)가 세부 주제를 조망했다. Gibson은 이론/모델 장을 최신화하고, TNE 기반 홍염 형성과 정적 force-free 가정을 벗어난 3D MHD 처리법을 통합하는 역할을 맡았다.

### 타임라인 / Timeline

```
1957 ├─ Kippenhahn & Schlüter: dipped arcade model (KS)
1967 ├─ Rust: B ~ 5-10 G, plasma β << 1 established
1974 ├─ Kuperus & Raadu: inverse-polarity flux rope (KR)
1983 ├─ Malherbe & Priest: normal/inverse configurations
1989 ├─ Leroy: inverse configurations dominant (75-90%)
1991 ├─ Antiochos & Klimchuk: thermal nonequilibrium (TNE)
1994 ├─ Antiochos et al.: 3D sheared-arcade dipped fields
1998 ├─ Titov & Démoulin: analytic 3D flux-rope model (TDm)
1998 ├─ Gibson & Low: spheromak-type prominence model
2004 ├─ Fan & Gibson: flux-rope emergence into arcade
2011 ├─ Berger et al.: prominence bubbles, Rayleigh–Taylor
2012 ├─ Luna et al.: 3D TNE in sheared arcade
2016 ├─ Xia & Keppens: full 3D MHD prominence-in-flux-rope
2017 ├─ Fan: 3D MHD in spherical coordinates with eruption
2018 ◄── Gibson review
```

---

## 3. 필요한 배경 지식 / Prerequisites

**English**:
- **MHD basics**: Induction equation, force balance including Lorentz force $\mathbf{J}\times\mathbf{B}$, frozen-in condition.
- **Plasma β**: Ratio of thermal to magnetic pressure $\beta = p/(B^2/2\mu_0)$; low-β means magnetic forces dominate.
- **Force-free fields**: $\nabla\times\mathbf{B}=\alpha\mathbf{B}$; constant-α is linear, spatially varying α is nonlinear.
- **Magnetic topology**: Flux rope vs. arcade; X-lines, separatrices, QSLs, BPSS.
- **Paper #36 (Parenti 2014, companion observational review)**: Observed morphology, mass, temperature, velocity of prominences.
- **Solar atmosphere structure**: Chromosphere, transition region, corona; scale heights.

**한국어**:
- **MHD 기초**: 유도 방정식, Lorentz 힘 $\mathbf{J}\times\mathbf{B}$ 포함 힘 균형, frozen-in 조건.
- **플라즈마 β**: 열 압력/자기 압력 비 $\beta = p/(B^2/2\mu_0)$; low-β는 자기력 지배.
- **Force-free 장**: $\nabla\times\mathbf{B}=\alpha\mathbf{B}$; 상수 α(선형), 공간 변동 α(비선형).
- **자기 위상수학**: Flux rope vs. arcade; X-선, separatrix, QSL, BPSS.
- **논문 #36 (Parenti 2014, 관측 동반 리뷰)**: 관측된 홍염 형태, 질량, 온도, 속도.
- **태양 대기 구조**: 채층, 전이영역, 코로나; 척도 높이(scale height).

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| Prominence / 홍염 | Cool (~7,500 K), dense plasma in a magnetic structure, seen bright above the limb / 코로나 위로 보이는 ~7,500 K의 차갑고 조밀한 자기 구조 내 플라즈마 |
| Filament / 필라멘트 | Same object seen in absorption against the disk / 원반 배경에서 흡수로 관측되는 동일 구조 |
| PIL (Polarity Inversion Line) / 극성 반전선 | Locus where radial photospheric field changes sign; filaments always lie above a PIL / 광구 자기장 부호가 바뀌는 선 |
| Magnetic dip / 자기 dip | Local concavity of field lines that can cradle plasma against gravity / 중력에 저항해 플라즈마를 받칠 수 있는 자기장 오목부 |
| Kippenhahn–Schlüter model / KS 모델 | 2.5D dipped-arcade model (1957) where Lorentz force in a current sheet balances gravity / 전류 시트의 Lorentz 힘이 중력을 상쇄하는 2.5D dip arcade 모델 |
| Kuperus–Raadu model / KR 모델 | 2.5D current-filament model (1974) with image currents producing inverse-polarity field / 이미지 전류가 inverse 극성 장을 만드는 2.5D 전류 필라멘트 모델 |
| Sheared arcade / Shear된 arcade | Arcade whose field lines are displaced parallel to the PIL; sheared enough can form dips in 3D / PIL 방향으로 shear된 arcade; 3D에서 dip 형성 가능 |
| Flux rope / 자속 튜브 | Bundle of twisted magnetic field lines wrapping a central axis; topologically distinct from potential arcade / 중심축을 감는 뒤틀린 자속 다발; 포텐셜 arcade와 위상학적으로 구분 |
| Force-free field / Force-free 장 | $\nabla\times\mathbf{B}=\alpha\mathbf{B}$; current parallel to field, Lorentz force zero / 전류가 자기장에 평행, Lorentz 힘 0 |
| Chirality / Chirality (손대칭성) | Handedness of magnetic structure; dextral/sinistral, dominant hemispheres follow rules / 자기 구조의 손대칭성; dextral/sinistral |
| Sigmoid / 시그모이드 | S- or inverse-S-shaped soft X-ray active region prone to eruption / 폭발 성향을 가진 S 또는 역-S형 SXR 구조 |
| Coronal cavity / 코로나 cavity | Dark elliptical region around quiescent prominences indicating plasma depletion within surrounding flux rope / 주위 flux rope 내 플라즈마 결핍을 나타내는 홍염 주변 암흑 타원 영역 |
| Thermal Nonequilibrium (TNE) / 열 비평형 | Catastrophic cooling/condensation along magnetic loops driven by localized footpoint heating / 풋포인트 가열에 의한 loop 상 파국적 냉각·응축 |
| BPSS (Bald Patch Separatrix Surface) / BPSS | Separatrix surface emanating from a bald patch where field touches photosphere tangentially / 광구에 접선 방향으로 접하는 bald patch에서 뻗는 separatrix 표면 |
| TDm (Titov–Démoulin) / TDm 모델 | Analytic 3D current-carrying flux rope equilibrium (1999) / 3D 해석적 전류 운반 flux rope 평형 (1999) |

---

## 5. 수식 미리보기 / Equations Preview

**English**:

**KS prominence support** — the Lorentz force from a current sheet supports the cool mass:
$$\rho g = B_x(x=0) \cdot [B_z] / \mu_0$$
where $[B_z]$ is the jump in $B_z$ across the sheet and $B_x(x=0)$ is the horizontal field through the sheet.

**Plasma β** — magnetic dominance criterion in prominences:
$$\beta = \frac{2\mu_0 p}{B^2} \ll 1$$
with $B \sim 5\text{-}30$ G and $T \sim 7{,}500$ K in a prominence embedded in a corona of $T \sim 1$ MK.

**Force-free equation** — quasi-equilibrium field:
$$\nabla\times\mathbf{B} = \alpha(\mathbf{r})\,\mathbf{B}, \qquad \mathbf{B}\cdot\nabla\alpha = 0$$

**KS analytic sheet solution** — 2.5D self-similar:
$$B_x(z) = B_{x0},\qquad B_z(z) = B_{z,\infty}\tanh(z/H),\qquad H = \frac{B_{x0} B_{z,\infty}}{\mu_0\, p_0\, g}$$
(schematic — the sheet thickness $H$ is set by the ratio of magnetic tension to weight).

**Kuperus–Raadu filament height** — line current at height $h$ with image current at $-h$:
$$F_{\text{lift}} = \frac{\mu_0 I^2}{4\pi h} - \frac{1}{c}I B_{\text{bg}} - M g = 0$$

**한국어**: 위 수식은 (1) KS 모델에서 전류 시트를 가로지르는 $B_z$ 점프와 시트 내 $B_x$가 중력을 상쇄함, (2) plasma β가 극도로 작아 자기력이 열압 대비 압도적임, (3) force-free 조건 $\nabla\times\mathbf{B}=\alpha\mathbf{B}$가 준평형을 지배함, (4) KS 해석해의 sheet 두께 $H$가 자기 장력/무게 비로 설정됨, (5) KR 모델에서 선 전류와 이미지 전류, 배경장, 중력이 평형을 이룸을 보여준다.

---

## 6. 읽기 가이드 / Reading Guide

**English**:
- **Section 2 (pp. 3–10)**: Focus on the conceptual distinction between dipped arcade (KS) and flux rope (KR), and the subtle normal vs. inverse dichotomy. Study Fig. 2–4 carefully.
- **Section 3 (pp. 11–20)**: Observational constraints — chirality, sigmoids, cavities. Connect each observational signature to model predictions.
- **Section 4 (pp. 21–28)**: Beyond the static skeleton — dynamics, TNE thermodynamics, and full 3D MHD. Skim equations but understand the physical logic (why force-free assumption breaks down when mass is loaded).
- Pair with **Paper #36 (Parenti 2014)** for the observational counterpart.

**한국어**:
- **Section 2 (pp. 3–10)**: Dipped arcade(KS)와 flux rope(KR)의 개념적 차이, normal vs. inverse의 미묘한 구분에 집중. Fig. 2–4를 꼼꼼히 학습.
- **Section 3 (pp. 11–20)**: 관측적 제약 — chirality, sigmoid, cavity. 각 관측 신호를 모델 예측과 연결.
- **Section 4 (pp. 21–28)**: 정적 골격을 넘어 — 동역학, TNE 열역학, 전면 3D MHD. 수식은 훑되, 질량이 실리면 force-free 가정이 왜 무너지는지 물리적 논리를 이해.
- 관측 동반 논문인 **Paper #36 (Parenti 2014)**과 함께 읽을 것.

---

## 7. 현대적 의의 / Modern Significance

**English**: Prominences are both a fundamental magnetohydrostatic laboratory for the quiet Sun and the progenitors of coronal mass ejections (CMEs) that drive space weather. The flux-rope vs. sheared-arcade debate directly maps onto CME initiation scenarios (torus instability, tether-cutting reconnection, kink instability). Cavity polarimetry with CoMP and eventually DKIST will test flux-rope topology predictions. Data-driven 3D MHD with thermodynamics (Xia & Keppens 2016; Fan 2017) is now the frontier, and Gibson's review lays out the conceptual scaffolding on which these simulations rest.

**한국어**: 홍염은 조용한 태양의 자기정수역학적 실험실인 동시에, 우주 기상을 좌우하는 코로나 질량 방출(CME)의 기원이다. Flux rope vs. sheared arcade 논쟁은 CME 발동 시나리오(torus 불안정성, tether-cutting 재결합, kink 불안정성)와 직결된다. CoMP, 그리고 곧 DKIST로 이뤄질 cavity 편광 관측은 flux rope 위상 예측을 검증할 것이다. 열역학을 포함한 자료 주도 3D MHD(Xia & Keppens 2016; Fan 2017)가 현재 최전선이며, Gibson의 리뷰는 이 시뮬레이션들이 딛고 설 개념적 골격을 제시한다.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
