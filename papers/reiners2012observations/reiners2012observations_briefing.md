---
title: "Pre-Reading Briefing: Observations of Cool-Star Magnetic Fields"
paper_id: "28"
topic: Living_Reviews_in_Solar_Physics
date: 2026-04-23
type: briefing
---

# Observations of Cool-Star Magnetic Fields: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Reiners, A. (2012), "Observations of Cool-Star Magnetic Fields", *Living Reviews in Solar Physics*, **8**, 1. DOI: 10.12942/lrsp-2012-1
**Author(s)**: Ansgar Reiners (Georg-August-Universität, Institut für Astrophysik, Göttingen)
**Year**: 2012

---

## 1. 핵심 기여 / Core Contribution

**EN**: This Living Review surveys the full landscape of observational techniques used to detect and characterize magnetic fields on stars cooler than spectral type F — the "cool stars" that resemble the Sun or are less massive (G, K, M dwarfs, pre-main-sequence stars, brown dwarfs, and some giants). Reiners lays out the physical basis of the Zeeman effect, the Stokes I / Q / U / V polarimetric formalism, the ambiguity between field strength B and filling factor f (the observable is only their product Bf), Zeeman Doppler Imaging (ZDI), and Least Squares Deconvolution (LSD), then marshals several hundred magnetic field measurements into seven master tables. The review ties these measurements to the rotation–activity–magnetic-field relation, the Rossby-number description of dynamo efficiency, the saturation regime at Ro ~ 0.1, the transition to fully convective interiors at ~M3/M4, and the unified energy-flux scaling that connects planets, brown dwarfs, and stars.

**KR**: 본 Living Review는 F형보다 차가운 "cool star"(태양형 G형, K형, M형 왜성, 전주계열성, 갈색왜성, 일부 거성)의 자기장을 관측적으로 검출·특성화하는 모든 기법을 총망라한다. Reiners는 Zeeman 효과의 물리적 기초, Stokes I/Q/U/V 편광 형식, 자기장 세기 B와 채움 인자 f 사이의 근본적 축퇴(관측량은 오직 Bf뿐)를 설명하고, Zeeman Doppler Imaging(ZDI)과 Least Squares Deconvolution(LSD) 같은 진보된 재구성 기법을 정리한 뒤, 수백 개의 자기장 측정치를 7개의 마스터 표로 정리한다. 또한 회전–활동–자기장 관계, 다이나모 효율을 Rossby 수로 기술하는 방식, Ro ~ 0.1 부근의 포화 영역, M3/M4 부근의 완전대류 전이, 행성–갈색왜성–별을 하나로 잇는 에너지 플럭스 스케일링까지 연결짓는다.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

**EN**: By 2012 solar physics had produced an exquisitely detailed picture of the solar magnetic field (SOHO/MDI, Hinode, SDO), but the Sun is a single, slow-rotator data point. To test whether the solar dynamo paradigm is universal, observers needed magnetic field measurements on many stars spanning a wide range of mass, age, and rotation. After Robinson (1980) demonstrated Fourier-transform line-width comparisons, Saar (1988, 1996), Johns-Krull & Valenti (1996, 2000), Donati et al. (1997, 2008), and Reiners & Basri (2006, 2007, 2009) pushed the field forward with improved radiative transfer, infrared spectroscopy (Ti I 2.22 μm, FeH 1 μm), Stokes-V ZDI, and LSD. This review consolidates that quarter-century of effort.

**KR**: 2012년경 태양물리학은 SOHO/MDI, Hinode, SDO 관측을 통해 태양 자기장을 매우 자세히 그려냈지만, 태양은 단 하나의 느리게 자전하는 데이터 포인트에 불과하다. 태양 다이나모 모형이 보편적인지 검증하려면 다양한 질량·나이·자전율을 가진 여러 별에서 자기장을 측정해야 했다. Robinson(1980)의 Fourier 변환 선폭 비교 이후, Saar(1988, 1996), Johns-Krull & Valenti(1996, 2000), Donati 외(1997, 2008), Reiners & Basri(2006, 2007, 2009) 등이 복사 전달 개선, 적외선 분광(Ti I 2.22 μm, FeH 1 μm), Stokes-V ZDI, LSD 등으로 분야를 밀고 나갔다. 본 리뷰는 그 사반세기의 성과를 정리한다.

### 타임라인 / Timeline

```
1897 ───── Zeeman effect discovered (Zeeman)
1924 ───── Hanle effect (Hanle)
1947 ───── Babcock — first stellar magnetic fields (Ap stars)
1971 ───── Preston — Zeeman analyzer for Ap stars
1980 ───── Robinson — Fourier-transform stellar Zeeman (sun-like)
1985 ───── Saar & Linsky — first Zeeman in M dwarf (AD Leo, 2.22 μm)
1988 ───── Saar — improved Zeeman analysis framework
1989 ───── Semel — Zeeman Doppler Imaging (ZDI) introduced
1995 ───── Valenti et al. — IR Zeeman in K dwarfs (ε Eri)
1996 ───── Johns-Krull & Valenti — FeI 8468 Å M-dwarf fields
1997 ───── Donati et al. — Least Squares Deconvolution (LSD)
2000 ───── Johns-Krull & Valenti — multi-component M dwarf fields
2003 ───── Pizzolato et al. — X-ray/Rossby rotation-activity relation
2006 ───── Reiners & Basri — FeH 1 μm for M-dwarf magnetism
2008 ───── Donati et al., Morin et al. — Stokes-V M dwarf ZDI maps
2009 ───── Christensen et al. — unified planet/star dynamo scaling
2012 ───── Reiners — THIS REVIEW
```

---

## 3. 필요한 배경 지식 / Prerequisites

**EN**:
- **Atomic physics**: LS coupling, quantum numbers J, L, S, M, selection rules ΔM = −1, 0, +1
- **Stellar atmospheres**: absorption line formation, Doppler broadening, rotational broadening (v sin i), radiative transfer
- **Polarization**: Stokes parameters I, Q, U, V; linear vs. circular polarization; weak-field approximation
- **Stellar structure**: convective envelopes, radiative cores, the tachocline; the M3/M4 fully-convective boundary
- **Dynamo theory basics**: α- and Ω-effects, Rossby number Ro = P_rot / τ_conv, activity saturation
- **Stellar activity indicators**: Ca II H&K, Hα, coronal X-rays, chromospheric emission

**KR**:
- **원자물리**: LS 결합, 양자수 J, L, S, M, 선택규칙 ΔM = −1, 0, +1
- **항성 대기**: 흡수선 형성, Doppler 넓힘, 자전 넓힘(v sin i), 복사 전달
- **편광**: Stokes 매개변수 I, Q, U, V; 선편광 vs. 원편광; weak-field 근사
- **항성 구조**: 대류 외피, 복사 중심핵, tachocline; M3/M4 완전대류 경계
- **다이나모 이론 기초**: α-, Ω-효과, Rossby 수 Ro = P_rot / τ_conv, 활동도 포화
- **항성 활동 지표**: Ca II H&K, Hα, 코로나 X-선, 채층 방출

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **Zeeman effect** | Splitting of an atomic energy level into (2J+1) sublevels in a magnetic field / 자기장 속에서 원자 에너지 준위가 (2J+1)개 부준위로 갈라지는 현상 |
| **Landé g-factor** | Dimensionless sensitivity of a level or transition to magnetic splitting, g = 3/2 + [S(S+1) − L(L+1)] / [2J(J+1)] / 자기 분리에 대한 전이의 무차원 민감도 |
| **Stokes vectors (I, Q, U, V)** | Four-parameter description of polarization: total intensity, two linear polarizations, circular polarization / 편광 상태를 기술하는 4개 매개변수 (총 세기, 두 선편광, 원편광) |
| **Filling factor f** | Fraction of the stellar surface covered by magnetic regions; observable is Bf (not B alone) / 자기 영역이 덮는 항성 표면 비율; 관측값은 B가 아닌 Bf |
| **Weak-field approximation** | V(v) ∝ g B ∂I/∂v valid when Zeeman splitting ≪ Doppler width / Zeeman 분리가 Doppler 폭보다 훨씬 작을 때 유효한 근사 |
| **Zeeman Doppler Imaging (ZDI)** | Inversion of time-series polarization spectra to reconstruct stellar surface magnetic maps / 시계열 편광 스펙트럼을 반전하여 항성 표면 자기장 지도를 복원 |
| **Least Squares Deconvolution (LSD)** | Multi-line co-addition technique that boosts S/N by assuming all lines share a common broadening function / 모든 선이 공통 넓힘 함수를 공유한다고 가정하여 S/N을 향상시키는 다중선 누적 기법 |
| **Rossby number (Ro)** | Ratio of rotation period to convective overturn time, Ro = P_rot / τ_conv; governs dynamo efficiency / 자전 주기와 대류 뒤집힘 시간의 비율; 다이나모 효율을 지배 |
| **Tachocline** | Thin shear layer at the base of the convection zone where the solar large-scale dynamo is believed to operate / 대류대 바닥의 얇은 전단층, 태양 대규모 다이나모가 작동하는 장소로 여겨짐 |
| **Fully convective star** | Star of spectral type later than ~M3/M4 with no radiative core, hence no tachocline / M3/M4형보다 늦은 별로, 복사 중심핵이 없어 tachocline도 없음 |
| **Equipartition field** | Magnetic field strength at which magnetic pressure B²/8π balances gas pressure / 자기 압력 B²/8π이 기체 압력과 균형을 이루는 자기장 세기 |
| **Saturation regime** | Regime Ro ≲ 0.1 where activity/field no longer grows with faster rotation / Ro ≲ 0.1 영역에서 자전이 빨라져도 활동/자기장이 더 이상 증가하지 않는 영역 |

---

## 5. 수식 미리보기 / Equations Preview

**1. Landé g-factor / Landé g-인자** (Eq. 1):
$$g_i = \frac{3}{2} + \frac{S_i(S_i+1) - L_i(L_i+1)}{2 J_i (J_i+1)}$$
**EN**: Dimensionless sensitivity of an atomic level to Zeeman splitting, derived from LS coupling.
**KR**: LS 결합으로 유도되는 원자 준위의 Zeeman 분리 민감도.

**2. Zeeman wavelength shift / Zeeman 파장 이동** (Eq. 3):
$$\Delta\lambda = 46.67 \, g \, \lambda_0^2 \, B \quad \text{(mÅ, with } \lambda_0 \text{ in μm, } B \text{ in kG)}$$
**EN**: The λ² dependence makes IR observations far more sensitive than optical.
**KR**: λ² 의존성 때문에 적외선 관측이 가시광보다 훨씬 민감하다.

**3. Zeeman velocity shift / Zeeman 속도 이동** (Eq. 4):
$$\Delta v = 1.4 \, \lambda_0 \, g \, B \quad \text{(km/s, with } \lambda_0 \text{ in μm, } B \text{ in kG)}$$
**EN**: A 1 kG field produces ~1 km/s at visible wavelengths — smaller than typical line widths and spectrograph resolution.
**KR**: 1 kG 자기장은 가시광에서 ~1 km/s 이동만 만들어내며, 이는 일반적 선폭·분해능보다 작다.

**4. Weak-field Stokes V / Weak-field Stokes V 근사** (Eq. 5):
$$V(v) \propto g_i \, B \, \frac{\partial I(v)}{\partial v}$$
**EN**: Stokes V is proportional to the derivative of Stokes I when Zeeman splitting is small; this is the foundation of LSD/ZDI in cool stars.
**KR**: Zeeman 분리가 작을 때 Stokes V는 Stokes I의 미분에 비례; 이는 cool star의 LSD/ZDI의 기반.

**5. Rotation-activity / 회전-활동 관계**:
$$\frac{L_X}{L_{\rm bol}} \propto Ro^{-\beta}, \quad Ro = P_{\rm rot}/\tau_{\rm conv}, \quad (Ro \gtrsim 0.1)$$
$$\frac{L_X}{L_{\rm bol}} \approx \text{const}, \quad (Ro \lesssim 0.1, \text{ saturation})$$
**EN**: Rossby-number description of magnetic activity; saturates below Ro ~ 0.1.
**KR**: Rossby 수로 기술되는 자기 활동도; Ro ~ 0.1 이하에서 포화.

---

## 6. 읽기 가이드 / Reading Guide

**EN**:
- **Section 2 (Methodology)**: The most important pedagogical part — read carefully if new to Zeeman/Stokes. Focus on §2.1.1–2.1.5 for the Bf degeneracy and §2.1.7 for ZDI.
- **Section 3 (Measurements)**: Skim the tables; don't try to memorize individual stars. Note the systematic differences between optical and IR Stokes I results in sun-like stars (§3.1.1).
- **Section 4 (Rotation–activity)**: Core conceptual result — the Rossby-number picture. Pay attention to Figure 19.
- **Sections 5–7**: Brief. Equipartition sets an upper bound of a few kG; geometry reconstructions from Donati et al. and Morin et al. show different field topologies at the fully-convective boundary.
- **Key figures**: Figures 1, 2, 3–4 (Stokes simulations), 6 (ZDI toy model), 12–13 (59 Vir χ² maps), 14 (M dwarf Zeeman broadening), 19 (Bf vs Ro), 23 (ZDI topology map).

**KR**:
- **2절 (방법론)**: 가장 중요한 교육적 부분 — Zeeman/Stokes에 익숙하지 않다면 꼼꼼히 읽을 것. Bf 축퇴에 대해서는 §2.1.1–2.1.5를, ZDI는 §2.1.7을 집중적으로.
- **3절 (측정)**: 표는 훑어만 볼 것; 개별 별을 외우려 하지 말 것. 태양형 별의 가시광 vs 적외선 Stokes I 결과 차이에 주목(§3.1.1).
- **4절 (회전-활동)**: 핵심 개념적 결과 — Rossby 수 그림. 그림 19에 주목.
- **5–7절**: 간결. Equipartition이 수 kG 상한을 설정. Donati 외, Morin 외의 기하 재구성은 완전대류 경계에서 자기장 위상이 달라짐을 보여줌.
- **핵심 그림**: 그림 1, 2, 3–4 (Stokes 시뮬레이션), 6 (ZDI 장난감 모형), 12–13 (59 Vir χ² 지도), 14 (M 왜성 Zeeman 넓힘), 19 (Bf vs Ro), 23 (ZDI 위상 지도).

---

## 7. 현대적 의의 / Modern Significance

**EN**: This review is the go-to reference for anyone entering stellar magnetic-field observations. Its insights directly inform exoplanet habitability (stellar winds, XUV flux from M-dwarf hosts), gyrochronology (rotation–age relations rely on magnetic braking), protostellar and T Tauri magnetospheric accretion, brown-dwarf / exoplanet interior dynamos (Christensen scaling), and tests of stellar dynamo theory in regimes (fully convective, saturated) unavailable on the Sun. It also provides the essential measurement-methodology context for modern spectropolarimeters (ESPaDOnS, HARPSpol, CARMENES, SPIRou, PEPSI, CRIRES+).

**KR**: 본 리뷰는 항성 자기장 관측 분야에 입문하는 사람에게 표준 참고문헌이다. 외계행성 거주 가능성(항성풍, M형 별 숙주 주변 XUV), 자이로크로놀로지(자전–나이 관계는 자기 제동에 의존), 원시별·T Tauri 자기권 강착, 갈색왜성/외계행성 내부 다이나모(Christensen 스케일링), 태양에서는 관측할 수 없는 영역(완전대류·포화)에서의 항성 다이나모 이론 검증 등에 직접 활용된다. 또한 최신 분광편광계(ESPaDOnS, HARPSpol, CARMENES, SPIRou, PEPSI, CRIRES+)의 방법론적 배경을 제공한다.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
