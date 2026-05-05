---
title: "Pre-Reading Briefing: The EUV Imaging Spectrometer for Hinode"
paper_id: "15_culhane_2007"
topic: Solar_Observation
date: 2026-04-16
type: briefing
---

# The EUV Imaging Spectrometer for Hinode: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: J.L. Culhane, L.K. Harra, A.M. James et al., "The EUV Imaging Spectrometer for Hinode," *Solar Physics*, Vol. 243, pp. 19–61, 2007.
**Author(s)**: J.L. Culhane, L.K. Harra, A.M. James et al.
**Year**: 2007

---

## 1. 핵심 기여 / Core Contribution

Hinode 위성에 탑재된 EUV Imaging Spectrometer (EIS)의 설계, 성능, 그리고 과학적 목표를 상세히 기술한 논문입니다. EIS는 극자외선(EUV) 영역(170–210 Å 및 250–290 Å)에서 고분해능 분광 관측을 수행하여, 태양 코로나와 전이영역의 플라즈마 진단(온도, 밀도, 속도, 원소 조성)을 가능하게 합니다. 이전 EUV 분광기들(SOHO/CDS, SUMER)에 비해 공간 분해능(2″)과 스펙트럼 분해능이 크게 향상되었으며, 특히 다중 온도 영역에서의 동시 관측 능력이 핵심적입니다.

This paper provides a detailed description of the design, performance, and scientific objectives of the EUV Imaging Spectrometer (EIS) aboard the Hinode satellite. EIS performs high-resolution spectroscopic observations in the EUV range (170–210 Å and 250–290 Å), enabling plasma diagnostics (temperature, density, velocity, elemental composition) of the solar corona and transition region. Compared to its predecessors (SOHO/CDS, SUMER), EIS offers significantly improved spatial resolution (2″) and spectral resolution, with the key capability of simultaneous multi-temperature observations.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

2000년대 초반, 태양 물리학 커뮤니티는 SOHO 위성(1995년 발사)의 CDS와 SUMER 분광기로부터 코로나 분광학의 혁명적 발전을 경험했습니다. 그러나 이들 장비의 공간 분해능(~4–6″)으로는 코로나 가열 메커니즘이나 플레어의 미세 구조를 충분히 분석하기 어려웠습니다. Hinode(Solar-B) 미션은 2006년 9월에 발사되어, SOT(광학 망원경, 논문 #14에서 학습), XRT(X선 망원경), 그리고 EIS 세 가지 주요 장비를 탑재했습니다.

In the early 2000s, the solar physics community had experienced revolutionary advances in coronal spectroscopy from SOHO's CDS and SUMER spectrometers (launched 1995). However, their spatial resolution (~4–6″) was insufficient for analyzing coronal heating mechanisms or fine structures in flares. The Hinode (Solar-B) mission launched in September 2006, carrying three main instruments: SOT (optical telescope, studied in Paper #14), XRT (X-ray telescope), and EIS.

### 타임라인 / Timeline

```
1995  SOHO 발사 — CDS & SUMER가 코로나 EUV 분광의 새 시대를 열다
      SOHO launch — CDS & SUMER open new era of coronal EUV spectroscopy
1998  TRACE 발사 — EUV 영상 관측의 공간 분해능 혁명 (~1″)
      TRACE launch — spatial resolution revolution in EUV imaging (~1″)
2002  RHESSI 발사 — 하드 X선/감마선 영상 분광
      RHESSI launch — hard X-ray/gamma-ray imaging spectroscopy
2006  Hinode(Solar-B) 발사 — SOT + XRT + EIS 동시 관측 시작
      Hinode launch — coordinated SOT + XRT + EIS observations begin
2007  ★ 본 논문: EIS 장비 기술 및 성능 상세 보고 ★
      ★ This paper: detailed EIS instrument description and performance ★
2010  SDO 발사 — AIA가 전천 EUV 영상을, EVE가 전일면 EUV 분광 제공
      SDO launch — AIA provides full-disk EUV imaging, EVE full-disk EUV spectroscopy
```

---

## 3. 필요한 배경 지식 / Prerequisites

### 3.1 EUV 분광학 기초 / EUV Spectroscopy Basics

- **극자외선(EUV)** 파장 대역(100–400 Å)은 코로나 및 전이영역 플라즈마의 방출선이 풍부한 영역입니다. 이 파장에서의 분광 관측으로 플라즈마의 물리적 성질을 진단할 수 있습니다.
- The **extreme ultraviolet (EUV)** wavelength range (100–400 Å) is rich in emission lines from coronal and transition region plasma. Spectroscopic observations at these wavelengths enable plasma diagnostics.

### 3.2 플라즈마 진단 기법 / Plasma Diagnostic Techniques

- **Emission measure (EM)**: 시선 방향을 따른 전자 밀도 제곱의 적분으로, 방출선 강도와 직접 관련됩니다.
- **Density-sensitive line ratios**: 같은 이온의 서로 다른 전이에서 나오는 두 방출선의 비율이 전자 밀도에 민감하게 변합니다 (예: Fe XII 186.88/195.12 Å).
- **Doppler shift & line width**: 방출선의 중심 파장 이동은 시선 속도를, 선폭 확장은 열적/비열적 운동을 나타냅니다.

### 3.3 이전 논문 (#14) 복습 / Review of Paper #14

- Hinode/SOT의 광학 설계(Gregorian 망원경)와 회절 한계 영상(0.2″) 개념을 복습해두면 좋습니다. EIS와 SOT는 동시 관측을 통해 광구 자기장과 코로나 플라즈마의 연결성을 연구합니다.
- Reviewing Hinode/SOT's optical design (Gregorian telescope) and diffraction-limited imaging (0.2″) from Paper #14 will be helpful. EIS and SOT perform coordinated observations to study the connectivity between photospheric magnetic fields and coronal plasma.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **EIS (EUV Imaging Spectrometer)** | Hinode의 EUV 분광기. 170–210 Å (단파장) 및 250–290 Å (장파장) 대역 관측 / Hinode's EUV spectrometer covering short-wave (170–210 Å) and long-wave (250–290 Å) bands |
| **Multilayer coating** | 특정 EUV 파장만 선택적으로 반사하는 다층 박막 코팅. Mo/Si 층 구조 사용 / Thin-film multilayer coatings that selectively reflect specific EUV wavelengths, using Mo/Si layer structure |
| **Off-axis paraboloid** | 비축 포물면 — 빛의 차단 없이 초점을 맞추는 광학 설계 / Off-axis parabolic mirror — focuses light without central obscuration |
| **Toroidal grating** | 두 축의 곡률이 다른 회절격자. 수차 보정과 분광을 동시에 수행 / Diffraction grating with different curvatures in two axes, performing aberration correction and dispersion simultaneously |
| **Slit** | 분광기의 입구 슬릿. 1″ 및 2″ 폭 선택 가능 / Spectrometer entrance slit; 1″ and 2″ width options available |
| **Slot** | 넓은 시야(40″, 266″)를 제공하는 개구부. 단색 영상(monochromatic imaging) 획득용 / Wide aperture (40″, 266″) for obtaining monochromatic images |
| **CCD (Back-illuminated)** | 후면 조사형 CCD — EUV 광자를 직접 검출하며 양자 효율이 높음 / Back-illuminated CCD directly detects EUV photons with high quantum efficiency |
| **Differential Emission Measure (DEM)** | 온도별 방출 기여도 분포. 다중 방출선 분석으로 코로나 온도 구조를 재구성 / Distribution of emission contribution vs. temperature; reconstructed from multiple emission lines |
| **FIP effect (First Ionization Potential)** | 제1 이온화 에너지가 낮은 원소(Fe, Si 등)가 코로나에서 광구 대비 풍부해지는 현상 / Enhancement of low-FIP elements (Fe, Si, etc.) in the corona relative to photospheric abundances |
| **Nonthermal velocity** | 열적 운동 이상의 선폭 확장을 유발하는 속도 성분 — 파동, 난류 등의 지표 / Velocity component causing line broadening beyond thermal motion — indicator of waves, turbulence, etc. |
| **Raster scan** | 슬릿을 태양면 위에서 단계적으로 이동시키며 2D 분광 영상을 구축 / Building 2D spectral images by stepping the slit across the solar surface |

---

## 5. 수식 미리보기 / Equations Preview

### 5.1 방출선 강도 / Emission Line Intensity

$$I_{ij} = \frac{1}{4\pi} \int A_X \, G(T, n_e) \, n_e^2 \, dl$$

- $I_{ij}$: 전이 $i \to j$의 관측 강도 (photons cm⁻² s⁻¹ sr⁻¹) / Observed intensity of transition $i \to j$
- $A_X$: 원소 X의 풍부도 / Abundance of element X
- $G(T, n_e)$: 기여 함수 (이온화 평형, 들뜸 계수 포함) / Contribution function (includes ionization equilibrium, excitation coefficients)
- $n_e$: 전자 밀도 / Electron density
- $dl$: 시선 방향 적분 경로 / Line-of-sight path element

### 5.2 밀도 민감 선비 / Density-Sensitive Line Ratio

$$R = \frac{I(\lambda_1)}{I(\lambda_2)} = f(n_e)$$

같은 이온의 두 방출선 비율이 전자 밀도의 함수입니다. 예를 들어 Fe XII 186.88 Å / 195.12 Å 비율은 $n_e \sim 10^8 – 10^{11}$ cm⁻³ 범위에서 밀도에 민감합니다.

The ratio of two emission lines from the same ion is a function of electron density. For example, the Fe XII 186.88 Å / 195.12 Å ratio is sensitive to density in the range $n_e \sim 10^8 – 10^{11}$ cm⁻³.

### 5.3 Doppler 속도 / Doppler Velocity

$$v = c \cdot \frac{\Delta\lambda}{\lambda_0}$$

- $\Delta\lambda$: 관측 파장과 정지 파장의 차이 / Difference between observed and rest wavelength
- $\lambda_0$: 정지 파장 / Rest wavelength
- $c$: 광속 / Speed of light

EIS의 스펙트럼 분해능은 약 22 mÅ (1″ 슬릿)으로, 이는 ~3 km/s의 Doppler 속도에 해당합니다.

EIS's spectral resolution is about 22 mÅ (1″ slit), corresponding to a Doppler velocity of ~3 km/s.

### 5.4 비열적 속도 / Nonthermal Velocity

$$\Delta\lambda_{obs}^2 = \Delta\lambda_{inst}^2 + 4\ln 2 \cdot \frac{\lambda_0^2}{c^2}\left(\frac{2k_BT}{m_i} + \xi^2\right)$$

- $\Delta\lambda_{inst}$: 장비 프로파일 폭 / Instrumental profile width
- $T$: 이온 온도 / Ion temperature
- $m_i$: 이온 질량 / Ion mass
- $\xi$: 비열적 속도 — 파동이나 난류의 존재를 나타내는 지표 / Nonthermal velocity — indicator of waves or turbulence

---

## 6. 읽기 가이드 / Reading Guide

### 필수 섹션 / Must-Read Sections

1. **Introduction (Section 1)** — EIS의 과학적 동기와 이전 EUV 분광기와의 차이점 개관
   Overview of EIS's scientific motivation and differences from previous EUV spectrometers

2. **Instrument Description (Sections 2–4)** — 광학 설계(off-axis paraboloid + toroidal grating), 검출기(back-illuminated CCD), 슬릿/슬롯 시스템의 기술적 상세
   Optical design, detector, and slit/slot system technical details

3. **Performance & Calibration (Section 5)** — 유효 면적, 분해능, 감도 등 핵심 성능 지표
   Effective area, resolution, sensitivity, and other key performance metrics

4. **Scientific Objectives (Section 6)** — EIS가 다루는 핵심 과학 질문들: 코로나 가열, 플레어, 태양풍 기원
   Key science questions: coronal heating, flares, solar wind origin

### 가볍게 읽을 섹션 / Skim-Read Sections

- **Thermal design, electronics** — 공학적 세부사항은 전체 그림 이해 후 필요시 참조
  Engineering details — refer back as needed after understanding the big picture

### 읽기 전략 / Reading Strategy

1. Introduction에서 "왜 EIS가 필요한가"를 먼저 파악하세요.
   First understand "why EIS is needed" from the Introduction.

2. Figure들을 먼저 훑어보세요 — 광학 경로도(optical layout), CCD 이미지 예시, 유효 면적 곡선이 핵심입니다.
   Scan the figures first — optical layout, CCD image examples, and effective area curves are key.

3. 성능 수치(공간 분해능, 스펙트럼 분해능, 유효 면적)를 SOHO/CDS와 비교하며 읽으세요.
   Read performance numbers (spatial resolution, spectral resolution, effective area) while comparing with SOHO/CDS.

---

## 7. 현대적 의의 / Modern Significance

EIS는 2006년 발사 이후 거의 20년째 운영 중이며, 코로나 분광학의 핵심 장비로 남아있습니다. EIS 데이터는 수천 편의 논문에 활용되었으며, 특히:

EIS has been operating for nearly 20 years since its 2006 launch and remains a cornerstone instrument for coronal spectroscopy. EIS data has been used in thousands of papers, particularly for:

- **코로나 가열 문제 / Coronal heating problem**: EIS의 비열적 속도 측정은 Alfvén 파동에 의한 코로나 가열 메커니즘의 핵심 관측 증거를 제공했습니다.
  EIS's nonthermal velocity measurements have provided key observational evidence for Alfvén wave-driven coronal heating.

- **태양풍 기원 / Solar wind origins**: 코로나 홀 경계와 활동영역 주변의 upflow를 EIS로 관측하여, 태양풍의 발원지를 추적하는 연구가 활발합니다.
  EIS observations of upflows at coronal hole boundaries and near active regions have enabled tracing of solar wind source regions.

- **차세대 장비 설계 / Next-generation instrument design**: Solar Orbiter의 SPICE와 같은 후속 EUV 분광기의 설계에 EIS의 경험이 직접 반영되었습니다.
  EIS experience directly informed the design of successor EUV spectrometers like Solar Orbiter's SPICE.

- **다파장 공동 관측 / Multi-wavelength coordinated observations**: SDO/AIA, IRIS 등과의 공동 관측을 통해 태양 대기의 다층 구조 연구에 핵심적 역할을 합니다.
  Coordinated observations with SDO/AIA, IRIS, etc. play a central role in studying the multi-layered solar atmosphere.

---

## Q&A

### Q1: EUV Imager의 Temperature Response Function vs. EUV Spectrograph의 Contribution Function

**질문**: EUV imager에는 temperature response function이라는 개념이 있는데, EUV spectrograph에도 비슷한 개념이 있는가?

**답변**:

두 장비의 접근 방식이 근본적으로 다릅니다.

Both instruments approach temperature diagnostics in fundamentally different ways.

#### EUV Imager: Temperature Response Function $R(T)$

EUV 영상기(예: SDO/AIA)는 **넓은 파장 대역 필터**를 사용합니다. 하나의 채널(예: 193 Å)에 여러 이온의 방출선이 **섞여서** 들어오기 때문에, "이 채널에서 관측된 신호가 각 온도에서 얼마나 기여받는가"를 나타내는 temperature response function $R(T)$가 필요합니다:

EUV imagers (e.g., SDO/AIA) use **broadband filters**. Multiple emission lines from different ions are **blended** in a single channel (e.g., 193 Å), so a temperature response function $R(T)$ is needed to describe how much each temperature contributes to the observed signal:

$$DN = \int R(T) \cdot DEM(T) \, dT$$

즉, 관측값(DN)에서 온도 정보를 **역산(inversion)**해야 합니다.

The temperature information must be **inverted** from the observed data numbers (DN).

#### EUV Spectrograph: Contribution Function $G(T, n_e)$

EIS 같은 분광기는 **개별 방출선을 분리**해서 관측합니다. 각 방출선은 특정 이온(예: Fe XII 195.12 Å)에서 나오고, 그 이온은 특정 온도 범위에서만 존재합니다:

Spectrographs like EIS **resolve individual emission lines**. Each line comes from a specific ion (e.g., Fe XII 195.12 Å), which exists only in a specific temperature range:

$$I = \frac{1}{4\pi} \int A_X \cdot G(T, n_e) \cdot n_e^2 \, dl$$

- $G(T, n_e)$는 **개별 방출선 하나**에 대한 함수 / Function for a **single emission line**
- 이온화 평형(ionization fraction) + 들뜸 계수(excitation rate) 포함 / Includes ionization fraction + excitation rate
- 온도에 대해 **좁은 피크**를 가짐 (특정 온도에서만 강함) / Has a **narrow peak** in temperature (strong only at a specific temperature)

#### 핵심 비교 / Key Comparison

| | EUV Imager | EUV Spectrograph |
|---|---|---|
| **관측 단위 / Observation unit** | 채널 (넓은 파장 대역) / Channel (broadband) | 개별 방출선 (좁은 파장) / Individual line (narrow) |
| **온도 함수 / Temperature function** | Temperature response $R(T)$ — 여러 선 혼합 / multiple lines blended | Contribution function $G(T)$ — 단일 선 / single line |
| **온도 선택성 / Temperature selectivity** | 낮음 (여러 온도 혼합) / Low (multi-thermal blend) | 높음 (특정 이온 = 특정 온도) / High (specific ion = specific temperature) |
| **온도 진단 / Temperature diagnostics** | DEM 역산 필요 (ill-posed) / DEM inversion needed | 직접 선택 가능 + DEM도 가능 / Direct selection + DEM possible |

#### 비유 / Analogy

- **Imager의 $R(T)$**: 여러 색이 섞인 조명 아래에서 "이 밝기에 빨간색이 얼마나 기여하나?"를 역추정 / Estimating how much red contributes to brightness under mixed-color lighting
- **Spectrograph의 $G(T)$**: 프리즘으로 색을 분리해서 빨간색만 직접 측정 / Using a prism to separate colors and directly measure only red
