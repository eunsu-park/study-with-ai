---
title: "Pre-Reading Briefing: Interaction Between Convection and Pulsation"
paper_id: "47"
topic: Living_Reviews_in_Solar_Physics
date: 2026-04-23
type: briefing
---

# Interaction Between Convection and Pulsation: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Houdek, G. & Dupret, M.-A., "Interaction Between Convection and Pulsation", *Living Reviews in Solar Physics*, **12**, 8 (2015). DOI: 10.1007/lrsp-2015-8
**Author(s)**: Günter Houdek (Stellar Astrophysics Centre, Aarhus University) & Marc-Antoine Dupret (Institut d'Astrophysique et de Géophysique, Université de Liège)
**Year**: 2015

---

## 1. 핵심 기여 / Core Contribution

**한국어**: 본 리뷰는 맥동(pulsation)과 난류 대류(turbulent convection)의 상호작용을 기술하는 1차원 시간 의존 대류(Time-Dependent Convection, TDC) 모델의 현재 이해를 집대성한다. 특히 두 가지 주류 형식론 — Gough(1977a,b)의 혼합 거리(mixing-length) 기반 국소/비국소 모델과 Unno(1967)를 일반화한 Grigahcène et al.(2005) 모델 — 을 체계적으로 비교한다. 저자들은 이 모델들이 고전적 맥동성(classical pulsators: Cepheid, RR Lyrae, δ Scuti, γ Dor, Mira, roAp)과 태양형 진동(solar-like oscillations)을 갖는 별들의 표면 진동수 이동(surface effects), 맥동 모드 안정성(mode stability), 감쇠율(damping rate) η, 진폭, 선폭 Γ = 2η을 어떻게 재현하는지 보여준다.

**English**: This review consolidates the present understanding of one-dimensional time-dependent convection (TDC) models that describe the interaction between pulsation and turbulent convection. It focuses on a systematic comparison of two prevailing formalisms — the mixing-length-based local/nonlocal model of Gough (1977a,b) and the Grigahcène et al. (2005) generalization of Unno (1967). The authors demonstrate how these models reproduce the surface frequency effects, mode stability, damping rates η, amplitudes, and linewidths Γ = 2η in classical pulsators (Cepheids, RR Lyrae, δ Scuti, γ Doradus, Miras, rapidly oscillating Ap stars) and in stars supporting stochastically excited solar-like oscillations.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

**한국어**: 고전적 Hertzsprung–Russell 불안정대(instability strip)의 저온 적색 경계(red edge)를 재현하는 것은 수십 년 동안 해결되지 않은 문제였다. 단열(adiabatic) 또는 동결 대류(frozen convection) 가정을 사용한 초기 모델은 적색 경계를 설명하지 못했고, 이는 대류와 맥동의 결합이 필수적임을 시사했다. 한편 1960년대 R. Leighton의 태양 5분 진동 발견 이후, 태양 진동 모드가 확률적으로 여기(stochastic excitation)되고 대류가 감쇠 기작을 제공한다는 인식이 커졌다. 본 리뷰는 CoRoT(2006)와 Kepler(2009) 시대의 풍부한 asteroseismic 데이터가 축적된 시점에 등장했다.

**English**: Reproducing the cool red edge of the classical Hertzsprung–Russell instability strip has been an outstanding problem for decades. Early models using adiabatic or frozen-convection assumptions failed to explain the red edge, implying that pulsation–convection coupling is essential. In parallel, after Leighton's 1960s discovery of solar 5-minute oscillations, it became clear that solar p modes are stochastically excited and that convection provides the damping. This review appeared when CoRoT (2006) and Kepler (2009) had generated vast asteroseismic datasets demanding quantitative TDC models.

### 타임라인 / Timeline

```
1925 ──────── Prandtl: mixing-length theory / Prandtl 혼합거리 이론
1958 ──────── Böhm-Vitense: stellar MLT / 별 MLT
1962 ──────── Baker & Kippenhahn: linear instability of Cepheids
1965-1977 ─── Gough: time-dependent MLT (local & nonlocal)
1967,1977 ─── Unno: time-dependent convection model
1979 ──────── Baker & Gough: red edge of RR Lyr using TDC
1992 ──────── Balmforth: solar p-mode damping with Gough TDC
1995 ──────── Rosenthal et al.: turbulent pressure in solar envelope
2005 ──────── Grigahcène et al.: nonradial generalization of Unno TDC
2006+ ─────── CoRoT (2006), Kepler (2009) → precision linewidths Γ
2012 ──────── Belkacem et al.: Kepler linewidth comparison
2014 ──────── Appourchaux et al.: Γ ∝ T_eff^13 in Kepler data
2015 ──────── Houdek & Dupret review (this paper)
```

---

## 3. 필요한 배경 지식 / Prerequisites

**한국어**:
- **유체역학**: Navier–Stokes, Reynolds 평균화, Boussinesq 근사, Reynolds stress tensor
- **항성 구조**: Schwarzschild 대류 판정, 혼합거리 이론(MLT), 정수압 평형
- **맥동이론**: 선형 단열/비단열 맥동, 작업 적분(work integral) W, p-모드/g-모드
- **헬리오/ 별지진학**: 태양 5-분 진동(ν~3 mHz), 파워 스펙트럼과 Lorentzian 프로파일, 선폭(linewidth)
- **수학**: 복소 고유진동수 ω = ω_r + iω_i, 감쇠율 η = ω_i, 섭동(perturbation)
- **관측**: 불안정대 적색 경계, κ-기작(opacity bump), Cepheid/RR Lyr/δ Sct 분류

**English**:
- **Fluid dynamics**: Navier–Stokes, Reynolds averaging, Boussinesq approximation, Reynolds stress tensor
- **Stellar structure**: Schwarzschild criterion, mixing-length theory (MLT), hydrostatic equilibrium
- **Pulsation theory**: Linear adiabatic/nonadiabatic pulsation, work integral W, p modes / g modes
- **Helio- and asteroseismology**: Solar 5-minute oscillations (ν~3 mHz), Lorentzian profile fits, linewidths
- **Mathematics**: Complex eigenfrequency ω = ω_r + iω_i, damping rate η = ω_i, perturbation theory
- **Observations**: red edge of instability strip, κ-mechanism (opacity bump), Cepheid/RR Lyr/δ Sct taxonomy

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| Mixing length ℓ | 혼합거리. 대류 요소가 이웃과 섞이기 전 이동하는 평균 거리. ℓ = α H_p (α~1-2). Average distance a convective parcel travels before mixing. |
| Superadiabatic gradient β | 초단열 온도 기울기 β = ∂T̄/∂q - ∂T̄/∂q|_ad. Drives buoyancy. |
| Reynolds stress ρuu | 난류 운동량 플럭스 텐서. isotropic 부분 = turbulent pressure p_t; anisotropic 부분 = σ_t. |
| Turbulent pressure p_t = ρ⟨w²⟩ | 난류 압력. 태양 표면에서 총 압력의 ~15%, δ Sct에서는 ~70%까지. |
| Convective heat flux F_c | 대류 열 플럭스 F_c ≃ ρ c_p ⟨w T'⟩. Heat transport by convective eddies. |
| Growth rate η / Damping rate | 허수 고유진동수 η = ω_i. η > 0 → 불안정(driving); η < 0 → 감쇠(damping). |
| Linewidth Γ = 2η | 파워 스펙트럼 Lorentzian의 FWHM. 선형 감쇠율의 2배 (각진동수 단위). |
| κ-mechanism | 불투명도(opacity) 상승 영역에서 작동하는 여기 기작. Drives Cepheids, RR Lyr, δ Sct. |
| Convective blocking | γ Dor g-mode 여기 기작. 대류 경계에서 radiative flux 변조. Also "convective shunting". |
| Time-dependent convection (TDC) | 맥동 주기와 대류 시간 규모가 비슷할 때, 대류 플럭스의 맥동 섭동을 포함하는 처리. |
| Frozen convection | δF_c = δp_t = 0으로 가정하는 근사. 짧은 주기 모드에서 실패. |
| Eddy survival probability P(r,t,t_0) | Gough 모델의 수학적 장치. 에디가 시각 t_0에 생성되어 t까지 살아남을 확률. |
| Work integral W | 한 주기 동안 모드에 전달되는 에너지. W>0 → 불안정, W<0 → 안정. |
| Surface effect | 관측 진동수와 단열 계산 사이의 고차 n 잔차. 태양에서 ~13 μHz. |
| Anisotropy parameter Φ = ⟨\|u\|²⟩/u_r² | 대류 속도장 이방성. Φ=3 → 등방(isotropic). |

---

## 5. 수식 미리보기 / Equations Preview

**한국어/English bilingual**

### (1) Mean momentum equation with turbulent pressure / 난류 압력 포함 평균 운동량 방정식

$$\bar{\rho}\frac{d\bm{U}}{dt} = \bar{\rho}\bar{\bm{g}} - \nabla(\bar{p} + p_t) - \nabla\cdot\bm{\sigma}_t$$

**한**: Reynolds 응력이 p_t (등방)와 σ_t (비등방)로 분리되어 정수압 방정식을 수정.
**En**: Reynolds stress is split into turbulent pressure p_t (isotropic) and σ_t (deviatoric), modifying hydrostatic support.

### (2) Mixing-length convective flux / 혼합거리 대류 플럭스

$$F_c \simeq \bar{\rho}\,c_p\,\overline{w T'} \propto \bar{\rho}\,c_p\,\beta^{3/2}\,\ell^2$$

**한**: 대류 열 플럭스는 초단열 기울기 β의 3/2 제곱에 비례. MLT의 핵심 닫힘(closure) 관계.
**En**: Convective heat flux scales as β^(3/2). The core closure relation of MLT.

### (3) Complex eigenfrequency & work integral / 복소 고유진동수와 작업 적분

$$\frac{\omega_i}{\omega_r} = \frac{W_g + W_t + F}{2\pi\omega_r^2\int_{m_b}^M|\delta r|^2 dm} = -\hat{\eta}_g - \hat{\eta}_t + \hat{F}$$

**한**: 허수 부분 ω_i(= η)는 가스압(W_g), 난류압(W_t) 작업 기여로 결정. W>0이면 불안정.
**En**: The imaginary part ω_i (= η) is determined by gas-pressure (W_g) and turbulent-pressure (W_t) work contributions. W>0 means instability.

### (4) p-mode line profile / p-모드 선 프로파일

$$P(\nu) = \frac{V_{\rm rms}^2}{\pi\,\Gamma/2}\,\frac{1}{1 + [(\nu - \nu_0)/(\Gamma/2)]^2}, \quad \Gamma = 2\eta/(2\pi)$$

**한**: 확률적으로 여기된 p-모드의 파워 스펙트럼은 선폭 Γ = 2η을 갖는 Lorentzian. Kepler 관측과 직접 비교 가능.
**En**: The power spectrum of a stochastically excited p mode is a Lorentzian with FWHM Γ = 2η. Directly comparable with Kepler.

### (5) Kjeldsen et al. (2008) surface-effect correction / Kjeldsen 표면 효과 보정

$$\delta\nu = a\,\left(\frac{\nu}{\nu_0}\right)^b$$

**한**: 관측-모델 진동수 차이를 경험적 거듭제곱법칙으로 보정. 태양에서 b~4.9.
**En**: Empirical power-law correction for observed-model frequency differences; b~4.9 in the Sun.

---

## 6. 읽기 가이드 / Reading Guide

**한국어**:
1. **§1-2 개요**: Reynolds 분리 접근과 평균 방정식을 파악. 첫 읽기에서는 (8)-(30)을 세부적으로 따라가기보다 전체 논리 흐름을 이해.
2. **§3 (TDC 모델)**: 가장 두꺼운 섹션. §3.2(Gough 국소)와 §3.4(Unno/Grigahcène 비국소 비방사)에 집중. §3.5-3.6은 두 모델의 차이 표이므로 속독.
3. **§4 (Reynolds stress 모델)**: Xiong 모델은 참고용. 시간 없으면 건너뛰기.
4. **§5 (진동수 효과)**: Figure 2, 3, 4는 필수. "surface effect" 개념과 turbulent pressure p_t의 역할을 이해.
5. **§6 (구동/감쇠)**: 핵심 응용부. 6.1 work integral 형식, 6.2 고전 맥동성, 6.3 태양형/적색거성 중 관심 있는 섹션 위주.
6. **§7 (다색 광도측정)**: 모드 식별 관련. 필요 시 참고.
7. **§8 결론 + 부록 A-E**: Gough와 Grigahcène 모델의 완전한 섭동 계수는 부록에서. 구현할 때 참조.

**English**:
1. **§1-2 Overview**: Grasp the Reynolds separation and mean equations. First pass: focus on logic, not on re-deriving (8)-(30).
2. **§3 (TDC models)**: The densest section. Concentrate on §3.2 (Gough local) and §3.4 (Unno/Grigahcène nonradial). §3.5-3.6 are comparison tables — skim.
3. **§4 (Reynolds stress models)**: Xiong model is peripheral; skip if time is short.
4. **§5 (frequency effects)**: Figures 2, 3, 4 are essential. Understand "surface effect" and the role of p_t.
5. **§6 (driving/damping)**: The main application section. §6.1 work-integral formalism, §6.2 classical pulsators, §6.3 solar-like / red-giant linewidths.
6. **§7 (photometry)**: For mode identification. Skim if not immediately relevant.
7. **§8 + Appendices A-E**: Complete perturbed-convection coefficients for Gough and Grigahcène models — reference when implementing.

---

## 7. 현대적 의의 / Modern Significance

**한국어**: CoRoT와 Kepler(그리고 후속 TESS, PLATO) 시대에 asteroseismology가 별 내부 물리의 정밀 측정 도구가 되면서, 시간 의존 대류 모델의 예측 정확도가 별의 반지름·나이·질량 결정 정확도의 주된 제한 요인이 되었다. 태양 진동수의 "surface effect"(~13 μHz)는 헬리오세이즘 역전(inversion)의 계통 오차를 지배하며, Kjeldsen et al.(2008)의 거듭제곱법칙 보정은 수천 개의 Kepler 별 모델링에 일상적으로 적용된다. Γ ∝ T_eff^13 같은 선폭의 유효 온도 의존성은 TDC 모델의 엄격한 시험대가 되고 있다. 3D 유체역학 시뮬레이션(Stein & Nordlund, Trampedach, Magic)이 1D TDC 모델을 교정하는 기준점을 제공하는 현대 흐름의 직접적 선구자이다.

**English**: In the CoRoT/Kepler (and subsequent TESS/PLATO) era, asteroseismology has become a precision probe of stellar interiors, and the predictive accuracy of time-dependent convection models is now a dominant systematic in the determination of stellar radii, ages, and masses. The solar "surface effect" (~13 μHz residual at ν ≳ 2.5 mHz) controls the systematic error budget of helioseismic inversions. The Kjeldsen et al. (2008) power-law correction is routinely applied to thousands of Kepler targets. Scaling relations like Γ ∝ T_eff^13 are stringent tests of TDC physics. The review directly anticipates the modern workflow where 3D hydrodynamical simulations (Stein & Nordlund, Trampedach, Magic) calibrate 1D TDC models.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
