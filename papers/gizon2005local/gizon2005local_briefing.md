# Pre-reading Briefing: Local Helioseismology
# 사전 읽기 브리핑: 국소 일진학

**Paper**: Gizon, L. & Birch, A. C. (2005)
**Journal**: *Living Reviews in Solar Physics*, **2**, 6
**DOI**: 10.12942/lrsp-2005-6
**Length**: 130+ pages — 매우 긴 기술적 리뷰

---

## 핵심 기여 / Core Contribution

이 리뷰는 **국소 일진학(local helioseismology)**의 이론적 기초와 관측 기법을 포괄적으로 정리한 최초의 체계적 리뷰입니다. Global helioseismology가 구형 조화 함수의 고유진동수 분석에 기반하여 태양 내부의 경도 평균된(축대칭) 구조만 탐사할 수 있는 데 반해, local helioseismology는 표면에서 관측된 **전체 파동장(full wavefield)**을 해석하여 태양 내부의 3차원적 비균질성 — 흐름, 음속 변동, 자기 구조 — 을 탐사합니다. 저자들은 다섯 가지 주요 기법 — Fourier-Hankel 분해, ring-diagram 분석, time-distance helioseismology, helioseismic holography, direct modeling — 을 수학적으로 엄밀하게 기술하고, 각 기법으로 얻은 과학적 결과(차등 회전, 자오면 순환, 초과립, 흑점 구조, far-side imaging)를 체계적으로 정리합니다.

This review is the first systematic survey comprehensively covering both the theoretical foundations and observational techniques of **local helioseismology**. While global helioseismology, based on eigenfrequency analysis of spherical harmonics, can only probe longitudinally-averaged (axisymmetric) solar structure, local helioseismology interprets the **full wavefield** observed at the surface to probe 3D inhomogeneities — flows, sound-speed variations, magnetic structures — inside the Sun. The authors rigorously describe five main techniques — Fourier-Hankel decomposition, ring-diagram analysis, time-distance helioseismology, helioseismic holography, and direct modeling — and systematically organize the scientific results obtained by each.

---

## 역사적 맥락 / Historical Context

```
1962  Leighton et al. — 태양 5분 진동 발견
         Discovery of solar 5-minute oscillations
  |
1970  Ulrich — 정상 음향파로 해석
         Interpreted as standing acoustic waves
  |
1975  Deubner — 분산 관계 관측적 확인
         Observational confirmation of dispersion relation
  |
1977  Goldreich & Keeley — 난류 대류에 의한 여기 메커니즘
         Excitation mechanism by turbulent convection
  |
1987  Braun et al. — 흑점의 음향파 흡수 발견 (Fourier-Hankel 분석)
         Discovery of acoustic wave absorption by sunspots
  |
1988  Hill — ring-diagram 분석 최초 도입
         Ring-diagram analysis first introduced
  |
1990  Lindsey & Braun — helioseismic holography 도입
         Helioseismic holography introduced
  |
1993  Duvall et al. — time-distance helioseismology 최초 발표
         Time-distance helioseismology first published
  |     Lindsey et al. — "local helioseismology" 용어 최초 사용
         Term "local helioseismology" first used in print
  |
1995  SOHO 발사 (MDI 장비) / SOHO launched (MDI instrument)
  |     GONG 네트워크 가동 / GONG network operational
  |
1997  Duvall et al. — 초과립 흐름 최초 관측 (time-distance)
         First observation of supergranular flows
  |
2000  Birch & Kosovichev — Born 근사 travel-time 커널
         Born approximation travel-time kernels
  |
2001  Braun & Lindsey — far-side imaging 최초 시연
         First demonstration of far-side imaging
  |
2002  Gizon & Birch — 분산 소스 모델의 포괄적 forward problem
         Comprehensive forward problem with distributed sources
  |
>>> 2005  Gizon & Birch — 이 리뷰 논문 <<<
           This review
```

---

## 필요한 배경 지식 / Prerequisites

### 파동 물리학 / Wave Physics
- **파동 방정식과 Green 함수** / Wave equation and Green's functions
- **분산 관계** ($\omega$-$k$ 다이어그램) / Dispersion relations ($\omega$-$k$ diagrams)
- **정상파 vs 전파파** / Standing vs propagating waves

### 수학 / Mathematics
- **Fourier 분석** (2D & 3D FFT) / Fourier analysis
- **구면 조화 함수** $Y_l^m(\theta, \phi)$ / Spherical harmonics
- **Bessel/Hankel 함수** / Bessel/Hankel functions
- **Born 근사** (산란 이론) / Born approximation (scattering theory)
- **역문제와 정칙화** (RLS, OLA) / Inverse problem and regularization

### 태양 물리 / Solar Physics
- **태양 진동의 기본** / Basics of solar oscillations: p-modes, f-modes, g-modes
- **초과립 대류** / Supergranulation
- **차등 회전과 자오면 순환** / Differential rotation and meridional circulation

### 선행 논문 / Prior Papers
- LRSP #2 Miesch (2005) — 대류층 역학 (tachocline, 차등 회전)
- LRSP #4 Sheeley (2005) — 자오면 흐름과 flux transport (local helioseismology가 확인)

---

## 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **p-mode** | 압력이 복원력인 음향 모드. 태양 내부를 관통하며, 고차 모드일수록 더 깊이 침투. / Acoustic mode with pressure as restoring force. Higher-order modes penetrate deeper. |
| **f-mode** | 표면 중력파. 분산 관계 $\omega^2 = gk$. 표층 근처만 탐사. / Surface gravity wave. Dispersion: $\omega^2 = gk$. Probes near-surface only. |
| **Time-distance helioseismology** / 시간-거리 일진학 | 두 표면 점 사이의 음향파 travel time을 측정하여 경로 상의 비균질성을 탐사. 교차 상관 함수가 핵심. / Measures acoustic wave travel times between two surface points to probe inhomogeneities along the path. Cross-covariance is key. |
| **Ring-diagram analysis** / 링 다이어그램 분석 | 태양 표면의 작은 패치($15° \times 15°$)의 국소 $\boldsymbol{k}$-$\omega$ 파워 스펙트럼에서 링 형태의 분산 곡선을 분석. 흐름에 의한 Doppler shift로 수평 흐름 추정. / Analyzes ring-shaped dispersion curves in local $\boldsymbol{k}$-$\omega$ power spectra from small patches. Horizontal flows estimated from Doppler shifts. |
| **Helioseismic holography** / 일진 홀로그래피 | Kirchhoff 적분을 이용하여 표면 파동장으로부터 태양 내부의 파동장을 재구성. ingression(수렴파)과 egression(발산파) 개념 사용. / Reconstructs wavefield in solar interior from surface observations using Kirchhoff integral. Uses ingression (converging) and egression (diverging) concepts. |
| **Cross-covariance** / 교차 공분산 | 두 표면 점의 필터링된 Doppler 속도 시계열 사이의 시간 상관. "solar seismogram"으로 볼 수 있음. / Temporal correlation between filtered Doppler velocity time series at two surface points. Can be seen as a "solar seismogram." |
| **Travel time** / 전파 시간 | 파동이 한 점에서 다른 점으로 전파하는 데 걸리는 시간. $\tau_+$와 $\tau_-$의 차이($\tau_{\text{diff}}$)가 흐름에, 평균($\tau_{\text{mean}}$)이 음속 변화에 민감. / Time for wave to travel from one point to another. Difference $\tau_{\text{diff}}$ is sensitive to flows, mean $\tau_{\text{mean}}$ to sound-speed changes. |
| **Sensitivity kernel** / 민감도 커널 | 태양 모델의 국소 변화가 관측량(travel time 등)에 미치는 영향을 기술하는 3D 함수. "banana-doughnut" 형태로 유명. / 3D function describing how local changes in the solar model affect observables (travel times, etc.). Famous "banana-doughnut" shape. |
| **Born approximation** / Born 근사 | 산란 이론에서 약한 섭동의 1차 근사. 파동장의 변화를 비섭동 Green 함수로 표현. / First-order approximation for weak perturbations in scattering theory. Wavefield changes expressed in terms of unperturbed Green's functions. |
| **Far-side imaging** / 원면 영상화 | Phase-sensitive holography의 특수 경우. 가시 디스크의 파동장으로 태양 뒷면의 활동 영역을 감지. / Special case of phase-sensitive holography. Detects active regions on the far side from wavefield on visible disk. |
| **RLS** / 정칙화 최소 자승법 | Regularized Least Squares — 데이터 적합과 해의 매끄러움 사이의 균형을 조절하는 역산법. / Balances data misfit and solution smoothness in inversions. |
| **OLA** | Optimally Localized Averages — 공간적으로 국소화된 평균 커널을 생산하면서 오차 증폭을 억제하는 역산법. / Produces spatially localized averaging kernels while controlling error amplification. |

---

## 수식 미리보기 / Equations Preview

### 1. 태양 진동의 고유모드 분해 / Normal Mode Decomposition

$$\delta r(r,\theta,\phi,t) = \sum_{n,l} \sum_{m=-l}^{l} a_{nlm}\,\xi_{nl}(r)\,Y_l^m(\theta,\phi)\,e^{i\omega_{nlm}t}$$

- $n$: 방사 차수 (깊이 관통도 결정) / Radial order (determines depth penetration)
- $l$: 구면 조화 차수 (수평 파수 $k \simeq l/R_\odot$) / Spherical harmonic degree
- $m$: 방위각 차수 / Azimuthal order

### 2. 파동 운동 방정식 / Wave Equation of Motion

$$\mathcal{L}\boldsymbol{\xi} = \boldsymbol{S}$$

$$\mathcal{L}\boldsymbol{\xi} = -\rho_0\frac{d_0^2\boldsymbol{\xi}}{dt^2} + \nabla[\gamma p_0\nabla\cdot\boldsymbol{\xi} + \boldsymbol{\xi}\cdot\nabla p_0] - (\nabla\cdot\boldsymbol{\xi})\nabla p_0 - \boldsymbol{\xi}\cdot\nabla(\nabla p_0)$$

### 3. 교차 공분산 (time-distance의 핵심) / Cross-covariance

$$C(\boldsymbol{x}_1, \boldsymbol{x}_2, t) = \frac{h_t}{T-|t|}\sum_{t'}\Psi(\boldsymbol{x}_1,t')\,\Psi(\boldsymbol{x}_2, t'+t)$$

이것이 **solar seismogram** — 양의 time lag은 $\boldsymbol{x}_1 \to \boldsymbol{x}_2$ 방향 파동, 음의 time lag은 반대 방향.
This is the **solar seismogram** — positive time lags for waves from $\boldsymbol{x}_1 \to \boldsymbol{x}_2$, negative for opposite.

### 4. Travel time 측정 / Travel Time Measurement (Gizon & Birch 2004)

$$\tau_\pm(\boldsymbol{x}_1, \boldsymbol{x}_2) = h_t \sum_t W_\pm(\boldsymbol{\Delta},t)\left[C(\boldsymbol{x}_1,\boldsymbol{x}_2,t) - C^0(\boldsymbol{\Delta},t)\right]$$

$C^0$: 기준 모델의 교차 공분산, $W_\pm$: 가중 함수. 노이즈에 강건한 정의.
$C^0$: reference model cross-covariance, $W_\pm$: weight functions. Robust to noise.

### 5. 선형 역문제 / Linear Forward Problem

$$\delta d_i = \sum_\alpha \int_\odot d^3\boldsymbol{r}\,K_\alpha^i(\boldsymbol{r})\,\delta q_\alpha(\boldsymbol{r})$$

$K_\alpha^i$: 민감도 커널, $\delta q_\alpha$: 태양 모델의 섭동(흐름, 음속 등).
$K_\alpha^i$: sensitivity kernel, $\delta q_\alpha$: perturbation to solar model (flows, sound speed, etc.).

### 6. Ring-diagram 피팅 모델 / Ring-diagram Fitting Model

$$P_{\text{fit}}(\psi, \omega) = \frac{A}{1 + (\omega - \omega_0 - kU_x\cos\psi - kU_y\sin\psi)^2/\gamma^2} + Bk^{-3}$$

$U_x, U_y$: 수평 흐름에 의한 Doppler shift, $\gamma$: 반치폭, $\omega_0$: 공진 주파수.

---

## 논문 구조 안내 / Paper Structure Guide

| 섹션 / Section | 내용 / Content | 페이지 / Pages | 난이도 |
|---|---|---|---|
| §1 Outline | 개요 | 1p | 쉬움 |
| §2 Observations | 데이터(TON, GONG, MDI)와 태양 진동 기본 특성 | 3p | 쉬움 |
| §3 Models | 파동 방정식, Green 함수, Born 근사, 자기장 효과 | 12p | **어려움** |
| §4 Methods | 5가지 기법: FH분해, ring-diagram, time-distance, holography, direct modeling | 36p | **어려움** |
| §5 Results | 과학적 결과: 회전, 자오면 흐름, 활동 영역, 초과립 | 49p | 보통 |
| §6 Acknowledgements | — | 1p | — |

**읽기 전략**: §2 → §3.1-3.4 (이론 기초) → §4.2 (ring-diagram, 가장 직관적) → §4.3 (time-distance) → §4.4 (holography) → §5.1 (대규모 흐름) → §5.3 (초과립)

§3.5-3.7과 §4.1, 4.5는 수학적으로 심화된 내용으로, 첫 읽기에서는 건너뛰어도 됩니다.
