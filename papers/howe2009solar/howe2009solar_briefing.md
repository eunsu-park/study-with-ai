---
title: "Pre-Reading Briefing: Solar Interior Rotation and its Variation"
paper_id: "15_howe_2009"
topic: Living_Reviews_in_Solar_Physics
date: 2026-04-15
type: briefing
---

# Solar Interior Rotation and its Variation: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Howe, R., "Solar Interior Rotation and its Variation", *Living Rev. Solar Phys.*, **6**, 1 (2009). DOI: 10.12942/lrsp-2009-1
**Author(s)**: Rachel Howe (National Solar Observatory, Tucson, AZ)
**Year**: 2009

---

## 1. 핵심 기여 / Core Contribution

이 리뷰 논문은 태양 내부 자전에 대한 관측적 이해의 발전 과정을 약 40년에 걸쳐 종합적으로 정리합니다. 1960년대 태양 핵 자전을 편평도로 추정하려던 시도부터, 일진학(helioseismology)의 발전을 거쳐, 태양 주기 23 동안 연속적 관측 데이터로 밝혀낸 상세한 내부 자전 프로파일까지를 다룹니다. 주요 주제로는 핵과 복사 내부의 강체 자전, tachocline 전단층, 대류층의 차등 자전, 표면 근처 전단층, 비틀림 진동(torsional oscillation), 그리고 tachocline의 시간 변동을 포함합니다.

This review comprehensively surveys the development of observational understanding of the Sun's interior rotation over approximately forty years. Starting from 1960s attempts to determine the solar core rotation from oblateness, proceeding through the development of helioseismology, it reaches the detailed modern picture of internal rotation deduced from continuous helioseismic observations during solar cycle 23. Key topics include the approximately rigid rotation of the core and radiative interior, the tachocline shear layer, differential rotation in the convection zone, the near-surface shear layer, the torsional oscillation pattern, and possible temporal variations at the base of the convection zone.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

태양의 표면 차등 자전(적도가 극보다 빠르게 회전)은 17세기 흑점 관측부터 알려져 있었습니다. 그러나 태양 **내부**의 자전 프로파일은 직접 관측이 불가능했습니다. 1960년대 Dicke의 태양 편평도 측정은 일반 상대성 이론 검증과 관련하여 논쟁을 불러일으켰고, 이는 태양 내부 자전에 대한 관심을 촉발했습니다. 1970년대 후반 일진학이 등장하면서 음향파를 이용한 내부 탐사가 가능해졌고, 1995년 GONG 네트워크와 1996년 SOHO/MDI의 가동으로 태양 주기 23 전체를 아우르는 연속 고품질 데이터가 확보되었습니다.

The Sun's surface differential rotation (equator rotating faster than poles) was known from sunspot observations since the 17th century. However, the interior rotation profile was inaccessible to direct observation. In the 1960s, Dicke's solar oblateness measurements sparked controversy related to tests of General Relativity, stimulating interest in the Sun's interior rotation. With the advent of helioseismology in the late 1970s, probing the interior using acoustic waves became possible. The operation of GONG (1995) and SOHO/MDI (1996) provided continuous high-quality data spanning essentially all of solar cycle 23.

### 타임라인 / Timeline

```
1960s   Dicke의 태양 편평도 측정 → 핵 자전 논쟁 / Dicke's oblateness → core rotation controversy
1975    Deubner, 5분 p-mode 진동 발견 / Deubner discovers 5-min p-mode oscillations
1979    Birmingham 그룹, 저차수 전구 모드 발견 / Birmingham group: global low-degree modes
1980    Howard & LaBonte, 비틀림 진동 발견 / Howard & LaBonte discover torsional oscillation
1984    Duvall & Harvey, 남극 관측으로 내부 자전 최초 추론 / South Pole: first interior rotation
1986    BBSO 100일 관측 (Libbrecht) / BBSO 100-day observations
1989    Brown et al., 대류층 내 자전 프로파일 / Brown et al., convection zone rotation profile
1989    Tachocline 발견 (Brown et al.) / Tachocline discovery
1995    GONG 네트워크 가동 / GONG network begins operation
1996    SOHO/MDI 가동 / SOHO/MDI begins operation
1998    Schou et al., MDI 144일 자전 역산 / Schou et al., MDI 144-day rotation inversion
2000    Howe et al., tachocline 1.3년 신호 보고 / Howe et al., 1.3-yr tachocline signal
2005    Howe et al., 기울어진 등자전 윤곽 발견 / Howe et al., slanted contours discovery
2008    Miesch et al., 태양과 유사한 3D 시뮬레이션 / Miesch et al., solar-like 3D simulation
2009    본 리뷰 출판, SDO 발사 대기 / This review published, awaiting SDO launch
```

---

## 3. 필요한 배경 지식 / Prerequisites

### 일진학 기초 / Helioseismology Basics (Paper #5: Gizon & Birch 2005)
- **p-mode (음압 모드)**: 태양 내부에서 공명하는 음향파. 복원력은 압력. 양자수 $(n, l, m)$으로 분류.
  **p-modes (pressure modes)**: Acoustic waves resonating inside the Sun. Restoring force is pressure. Classified by quantum numbers $(n, l, m)$.
- **구면 조화 함수**: $Y_l^m(\theta, \phi)$ — 모드의 각도 구조 기술. $l$은 차수(degree), $m$은 방위각 차수(azimuthal order).
  **Spherical harmonics**: $Y_l^m(\theta, \phi)$ — describe angular structure. $l$ = degree, $m$ = azimuthal order.
- **모드 침투 깊이**: 낮은 $l$의 모드일수록 태양 깊은 곳까지 도달. $l=0$은 중심부까지, $l \ge 200$은 표면 수 Mm만 탐사.
  **Mode penetration depth**: Lower $l$ modes penetrate deeper. $l=0$ reaches the center, $l \ge 200$ probes only a few Mm below the surface.

### 역산 문제 / Inversion Problem
- **Forward problem**: 주어진 자전 프로파일에서 주파수 분리(splitting) 예측.
  Given a rotation profile, predict frequency splittings.
- **Inverse problem**: 관측된 splitting에서 자전 프로파일 추론. 두 가지 주요 방법: RLS (정규화 최소제곱법)와 OLA (최적 국소 평균법).
  From observed splittings, infer the rotation profile. Two main methods: RLS (Regularized Least Squares) and OLA (Optimally Localized Averaging).

### 태양 구조 / Solar Structure
- **대류층 (Convection Zone)**: $r > 0.713 R_\odot$, 차등 자전이 지배적.
  $r > 0.713 R_\odot$, differential rotation dominates.
- **복사 내부 (Radiative Interior)**: $r < 0.713 R_\odot$, 대략 강체 자전.
  $r < 0.713 R_\odot$, approximately rigid rotation.
- **Tachocline**: 두 영역 사이의 얇은 전단층, $r \approx 0.69 R_\odot$.
  Thin shear layer between the two regions, $r \approx 0.69 R_\odot$.

### 주요 관측 시설 / Key Observing Facilities
- **GONG**: 전 세계 6개소 지상 네트워크, 중간 차수($l$) p-mode 연속 관측 (1995~).
  Six-station ground network, continuous medium-$l$ p-mode observations.
- **SOHO/MDI**: 우주 기반 Michelson Doppler Imager, 중/고 차수 모드 (1996~).
  Space-based imager, medium/high-$l$ modes.
- **BiSON**: 전구(Sun-as-a-star) 관측, 저차수 모드 전문.
  Integrated-sunlight observations, specialized in low-$l$ modes.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **Rotational splitting** / 자전 분리 | 태양 자전이 같은 $l$, 다른 $m$을 가진 모드의 주파수 축퇴를 제거하는 현상. $\delta\nu_{m,l} \equiv \nu_{-m,l} - \nu_{+m,l}$. 1차 근사에서 자전율에 비례. / The lifting of frequency degeneracy between modes of same $l$ but different $m$ due to solar rotation. Proportional to rotation rate to first order. |
| **Tachocline** / 타코클라인 | 대류층 하단, 차등 자전에서 강체 자전으로 전이하는 얇은 전단층 ($r \approx 0.69 R_\odot$, 두께 $\sim 0.02{-}0.05 R_\odot$). 태양 다이나모에 핵심적 역할. / Thin shear layer at base of convection zone transitioning from differential to rigid rotation. Key for the solar dynamo. |
| **Torsional oscillation** / 비틀림 진동 | 태양 주기에 맞춰 이동하는, 평균보다 빠르고 느린 대상(zonal) 흐름 패턴. 활동 영역의 적도 방향 이동과 연관. / Pattern of faster- and slower-than-average zonal flows migrating with the solar cycle, associated with equatorward drift of activity belts. |
| **Near-surface shear** / 표면 근처 전단 | $r > 0.95 R_\odot$에서 가장 빠른 자전층과 표면 사이의 자전 감소 영역. / Region of decreasing rotation between the fastest-rotating layer at $\sim 0.95 R_\odot$ and the surface. |
| **Differential rotation** / 차등 자전 | 자전율이 위도에 따라 달라지는 현상. 적도 $\sim 460$ nHz, 극 $\sim 320$ nHz. / Rotation rate varying with latitude. Equator $\sim 460$ nHz, poles $\sim 320$ nHz. |
| **Splitting coefficients** / 분리 계수 | 주파수의 $m$-의존성을 다항식으로 전개한 계수 $a_j$. 홀수 차수($a_1, a_3, \ldots$)는 자전 비대칭, 짝수 차수는 구조 비구대칭 정보. / Coefficients $a_j$ of polynomial expansion of $m$-dependence of frequencies. Odd-order encode rotation; even-order encode structural asphericity. |
| **Averaging kernel** / 평균 커널 | 역산에서 특정 위치의 추론된 자전율이 실제 자전 프로파일의 어떤 가중 평균인지를 보여주는 함수. / Function showing how the inferred rotation at a given location is a weighted average of the true rotation profile. |
| **RLS** (Regularized Least Squares) | 매끄러움 벌칙항을 포함한 최소제곱 역산 방법. 계산이 빠르지만 커널 국소화가 보장되지 않음. / Least-squares inversion with smoothness penalty. Computationally fast but kernel localization not guaranteed. |
| **OLA/SOLA** (Optimally Localized Averaging) | 평균 커널을 목표 함수(Gaussian 등)에 최대한 가깝게 만드는 역산 방법. 해석이 더 명확하지만 계산 비용이 높음. / Inversion that shapes averaging kernels to match a target function. Clearer interpretation but computationally expensive. |
| **$g$-modes** / 중력 모드 | 복원력이 부력(중력)인 모드. 핵 자전에 민감하지만 표면 진폭이 극히 작아 확정적 검출이 되지 않았음. / Modes with gravity (buoyancy) as restoring force. Sensitive to core rotation but surface amplitudes extremely small — no definitive detection. |
| **Oblateness** / 편평도 | 자전으로 인한 태양의 적도 방향 팽창. $J_2$ 사중극 모멘트와 관련. / Equatorial bulge of the Sun due to rotation. Related to $J_2$ quadrupole moment. |

---

## 5. 수식 미리보기 / Equations Preview

### 5.1 유체 요소의 변위 / Radial Displacement of a Fluid Element
$$\delta r(r,\theta,\phi,t) = \sum_{m=-l}^{l} a_{nlm} \xi_{nl}(r) Y_l^m(\theta,\phi) e^{i\omega_{nlm} t} \tag{1}$$

- $\xi_{nlm}$: 모드의 radial eigenfunction / radial eigenfunction of the mode
- $Y_l^m(\theta,\phi)$: 구면 조화 함수 / spherical harmonic
- $\omega_{nlm}$: 모드의 주파수 / mode frequency
- 태양 자전이 없으면 $\omega_{nlm}$은 $m$에 의존하지 않음 (축퇴). 자전이 이 축퇴를 제거.
  Without rotation, $\omega_{nlm}$ is independent of $m$ (degenerate). Rotation breaks this degeneracy.

### 5.2 역산 기본 방정식 / Inversion Basic Equation
$$d_i = \int_0^{R_\odot} \int_0^{\pi} K_i(r,\theta) \Omega(r,\theta) \, dr \, d\theta + \epsilon_i \tag{11}$$

- $d_i$: $i$번째 관측 데이터 (splitting) / $i$-th observed datum (splitting)
- $K_i(r,\theta)$: 공간 가중 함수 (커널) / spatial weighting function (kernel)
- $\Omega(r,\theta)$: 추론하고자 하는 자전 프로파일 / rotation profile to be inferred
- $\epsilon_i$: 측정 오차 / measurement error
- 핵심: 관측값은 커널로 가중된 자전 프로파일의 적분. 역산은 이 적분 방정식의 역을 구하는 것.
  Key: Each observation is an integral of the rotation profile weighted by a kernel. Inversion seeks to reverse this.

### 5.3 역산 해 / Inversion Solution
$$\bar{\Omega}(r_0,\theta_0) = \sum_{i=1}^{M} c_i(r_0,\theta_0) d_i \tag{14}$$

- $c_i$: 역산 계수 / inversion coefficients
- $(r_0, \theta_0)$: 추론 목표 위치 / target location
- 계수 $c_i$의 선택 방법이 RLS와 OLA 역산의 핵심 차이.
  The choice of coefficients $c_i$ is the essential difference between RLS and OLA inversions.

### 5.4 표면 차등 자전 경험식 / Surface Differential Rotation (Empirical)
$$\frac{\Omega_m}{2\pi} = 462 - 74\mu^2 - 53\mu^4 \text{ nHz} \tag{25}$$
$$\frac{\Omega_p}{2\pi} = 452 - 49\mu^2 - 84\mu^4 \text{ nHz} \tag{26}$$

- $\mu = \sin(\text{latitude})$
- 식 (25): 자기장 추적(magnetic features), 식 (26): 표면 플라즈마(Doppler)
  Eq. (25): magnetic feature tracking; Eq. (26): surface plasma (Doppler)
- Doppler 자전이 약간 느린 이유: 자기 요소가 더 빨리 도는 하층에 고정. Near-surface shear의 증거.
  Doppler rotation is slightly slower because tracers are anchored in the faster-rotating layer below — evidence for near-surface shear.

### 5.5 SOLA 최소화 함수 / SOLA Minimization Functional
$$\int_0^R \int_0^\pi [\mathcal{T}(r_0,\theta_0;r,\theta) - \mathcal{K}(r_0,\theta_0;r,\theta)]^2 \, r \, dr \, d\theta + \lambda \sum_{i=1}^{M} [\sigma_i c_i(r_0,\theta_0)]^2 \tag{22}$$

- $\mathcal{T}$: 목표 커널 (Gaussian 또는 Lorentzian) / target kernel
- $\mathcal{K}$: 실제 평균 커널 / actual averaging kernel
- $\lambda$: 해상도와 오차 증폭 사이의 tradeoff 매개변수 / tradeoff parameter between resolution and error amplification

---

## 6. 읽기 가이드 / Reading Guide

### 필수 섹션 / Essential Sections (첫 번째 읽기)
1. **Section 1 (Introduction)** — 4개의 주요 자전 구조(핵/복사 내부, tachocline, 대류층, 표면 전단)의 개요. Figure 1을 주의 깊게 보세요.
   Overview of four key rotation features. Study Figure 1 carefully.

2. **Section 2 (Acoustic Modes)** — 2.1-2.2절 집중. Rotational splitting의 물리적 의미와 splitting coefficients 이해.
   Focus on 2.1-2.2. Physical meaning of rotational splitting and coefficients.

3. **Section 3 (Inversion Basics)** — 3.1, 3.4, 3.5절 집중. RLS와 OLA 역산의 핵심 차이를 이해. Figure 11, 12 비교.
   Focus on 3.1, 3.4, 3.5. Key differences between RLS and OLA. Compare Figures 11, 12.

4. **Section 6 (The Tachocline)** — Table 2의 측정값들 비교. 태양 다이나모와의 연결.
   Compare measurements in Table 2. Connection to solar dynamo.

5. **Section 9 (Torsional Oscillation)** — Figures 24-28 집중. 태양 주기와의 관계.
   Focus on Figures 24-28. Relationship to solar cycle.

### 참고 섹션 / Reference Sections (두 번째 읽기)
- **Section 4** — 관측 역사 개요. Figure 13의 타임라인 참고.
- **Section 5** — 핵 자전 논쟁의 역사. 편평도 논쟁(5.1)과 저차수 splitting(5.5-5.7)의 어려움.
- **Section 7** — 대류층 자전의 관측 역사. Figure 18, 19가 핵심.
- **Section 8** — 표면 근처 전단. 식 (25)-(26)의 차이 이해.
- **Section 10** — Tachocline 시간 변동. 1.3년 신호의 불확실성.

### 핵심 Figure / Key Figures
| Figure | 내용 / Content |
|--------|--------------|
| **Fig. 1** | 태양 내부 자전 단면도 — 모든 주요 구조를 한눈에 / Cross-section of solar interior rotation — all key features at a glance |
| **Fig. 6** | $l$-$\nu$ 다이어그램의 모드별 탐사 영역 / Mode sensitivity regions in $l$-$\nu$ diagram |
| **Fig. 11, 12** | RLS vs. SOLA 평균 커널 비교 / RLS vs. SOLA averaging kernel comparison |
| **Fig. 18** | MDI 144일 자전 역산: 4가지 방법 비교 / MDI 144-day rotation inversion: 4 methods compared |
| **Fig. 19** | GONG 평균 자전 프로파일 (기울어진 윤곽선) / GONG mean rotation profile (slanted contours) |
| **Fig. 24** | MDI f-mode 비틀림 진동 패턴 / MDI f-mode torsional oscillation pattern |
| **Fig. 25-27** | 다양한 깊이/위도에서의 자전율 시간 변동 / Rotation rate time variations at various depths/latitudes |

---

## 7. 현대적 의의 / Modern Significance

### 태양 다이나모 이론에 대한 핵심 관측 제약 / Key Observational Constraints for Dynamo Theory
이 리뷰에서 정리한 내부 자전 관측 결과들은 태양 다이나모 모델의 가장 중요한 제약 조건입니다. 특히:
The internal rotation observations compiled in this review provide the most important constraints for solar dynamo models:

- **Tachocline의 존재와 얇기**: 다이나모가 대류층 하단의 전단층에서 작동함을 시사 → flux-transport dynamo 모델의 근거.
  The existence and thinness of the tachocline suggest the dynamo operates at the base of the convection zone.
- **대류층의 등자전 윤곽이 원통형이 아닌 ~25° 기울어진 형태**: Taylor-Proudman 정리의 위반 → Coriolis 힘과 열 구배의 상호작용.
  Constant-rotation contours tilted ~25° rather than cylindrical: violation of Taylor-Proudman theorem.
- **비틀림 진동의 깊이 침투**: 단순한 표면 현상이 아닌 대류층 전체에 걸친 현상 → 다이나모-자전 피드백의 증거.
  Torsional oscillation penetrating through the convection zone: evidence for dynamo-rotation feedback.

### SDO/HMI 시대와 그 이후 / The SDO/HMI Era and Beyond
2009년 리뷰 작성 시점에서 기대했던 SDO/HMI는 2010년에 성공적으로 발사되어, MDI보다 더 높은 해상도로 15년 이상의 관측을 제공하고 있습니다. 핵 자전, 고위도 표면 전단, tachocline 변동 등 이 리뷰에서 미해결로 남긴 문제들에 대한 연구가 계속되고 있습니다.
SDO/HMI, successfully launched in 2010, has provided 15+ years of higher-resolution observations. Research continues on the open questions left in this review: core rotation, high-latitude surface shear, and tachocline variations.

### 항성 물리학으로의 확장 / Extension to Stellar Physics
태양 내부 자전 연구의 방법론은 *Kepler* 및 *TESS* 위성의 별진학(asteroseismology) 데이터에 적용되어, 적색 거성 내부의 자전을 측정하는 데 활용되고 있습니다 (특히 $g$-mode를 통한 핵 자전 측정).
The methodologies of solar interior rotation studies have been applied to asteroseismic data from *Kepler* and *TESS*, measuring internal rotation of red giants (especially core rotation via $g$-modes).

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
