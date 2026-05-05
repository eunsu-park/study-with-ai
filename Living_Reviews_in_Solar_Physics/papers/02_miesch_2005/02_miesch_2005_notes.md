---
title: "Large-Scale Dynamics of the Convection Zone and Tachocline"
authors: Mark S. Miesch
year: 2005
journal: "Living Rev. Solar Phys., 2, 1"
doi: "10.12942/lrsp-2005-1"
topic: Living Reviews in Solar Physics / Solar Interior Dynamics
tags: [convection zone, tachocline, differential rotation, meridional circulation, solar dynamo, helioseismology, anelastic, ASH, Reynolds stress, angular momentum]
status: completed
date_started: 2026-04-08
date_completed: 2026-04-08
---

# Large-Scale Dynamics of the Convection Zone and Tachocline
# 대류층과 타코클라인의 대규모 역학

## Core Contribution / 핵심 기여

This comprehensive ~150-page review establishes the theoretical and computational framework for understanding large-scale dynamics in the solar interior — specifically the convection zone ($r \geq 0.71\,R_\odot$) and the tachocline at its base. Helioseismology has revealed that the Sun's differential rotation features a ~30% angular velocity contrast between equator (~27 days) and poles (~35 days), with nearly radial angular velocity contours at mid-latitudes, and a remarkably thin tachocline ($\Delta_t \sim 0.04\,R_\odot$) transitioning to nearly uniform rotation in the radiative interior. Weak meridional circulation (~20 m/s poleward at the surface) and elusive giant cells complete the observational picture. Three-dimensional global simulations using the Anelastic Spherical Harmonic (ASH) code can reproduce the broad features of differential rotation through angular momentum transport by Reynolds stresses balanced against meridional circulation, with departures from Taylor-Proudman cylindrical rotation explained by thermal wind (latitudinal entropy gradients). However, five outstanding challenges remain: (1) polar vortex problem, (2) insufficient equator-to-pole angular velocity contrast, (3) nearly radial $\Omega$ contours at mid-latitudes, (4) rotational shear layers (near-surface and tachocline), and (5) torsional oscillations. The thinness of the tachocline is most likely maintained by fossil poloidal magnetic fields confined in the radiative interior (Gough & McIntyre 1998 model), rather than by purely hydrodynamic processes such as anisotropic turbulent diffusion (Spiegel & Zahn 1992).

이 ~150페이지 종합 리뷰는 태양 내부 — 대류층($r \geq 0.71\,R_\odot$)과 그 하부의 타코클라인 — 의 대규모 역학을 이해하기 위한 이론적·수치적 프레임워크를 확립합니다. 일진학(helioseismology)은 태양의 차등 회전이 적도(~27일)와 극(~35일) 사이 ~30%의 각속도 차이를 가지며, 중위도에서 거의 방사상 각속도 등고선, 복사 내부의 균일 회전으로 전이하는 매우 얇은 타코클라인($\Delta_t \sim 0.04\,R_\odot$)을 가짐을 밝혔습니다. 약한 자오 순환(~20 m/s 표면 극향)과 아직 관측되지 않은 거대 세포가 관측적 그림을 완성합니다. ASH 코드를 이용한 3D 전구 시뮬레이션은 Reynolds 응력에 의한 각운동량 수송과 열풍 균형(위도 방향 엔트로피 기울기)을 통해 차등 회전의 대략적 특징을 재현하지만, 다섯 가지 핵심 도전 과제가 남아 있습니다: (1) 극 와류 문제, (2) 불충분한 적도-극 각속도 대비, (3) 중위도의 거의 방사상 $\Omega$ 등고선, (4) 회전 전단층, (5) 비틀림 진동. 타코클라인의 얇음은 순수 유체역학적 과정(Spiegel & Zahn 1992의 비등방 난류 확산)보다는 복사 내부에 갇힌 화석 극방향 자기장(Gough & McIntyre 1998 모델)에 의해 유지될 가능성이 높습니다.

---

## Reading Notes / 읽기 노트

### Section 1: A Turbulent Sun / 난류적 태양

태양은 근본적으로 난류적인 천체입니다. 대류층은 표면 아래 약 200 Mm($r \geq 0.71\,R_\odot$)에 걸쳐 있으며, 다양한 규모의 대류가 존재합니다:

**다중 대류 규모 (Multiple convection scales)**:
- **Granulation** (~1–2 Mm): 광구에서 직접 관측. 수명 ~10분. 뜨거운 상승류(밝은 중심)와 차가운 하강류(어두운 경계)의 패턴
- **Mesogranulation** (~5 Mm): 존재 여부가 논쟁적. 독립적인 대류 규모인지, 아니면 granulation과 supergranulation 사이의 스펙트럼 연속체의 일부인지 불확실
- **Supergranulation** (~30 Mm): 수명 ~1–2일. Doppler 관측에서 명확히 보이며, 자기장 네트워크(chromospheric network)를 형성. 수평 속도 ~300–500 m/s이 수직 속도보다 훨씬 큼
- **Giant cells** (>100 Mm): 이론적으로 예측되지만 아직 명확한 관측 증거 없음 — Section 3.5에서 상세 논의

이 다양한 규모는 모두 대류 에너지를 수송하며, 특히 대규모 대류가 **차등 회전(differential rotation)**과 **자오 순환(meridional circulation)**을 구동합니다.

**두 가지 다이나모 (Two dynamos)**:
- **국소 다이나모 (Local/small-scale dynamo)**: 대류층 전체에 걸친 작은 규모의 난류적 자기장 발생. "magnetic carpet"이라 불리는 조용한 태양(quiet Sun)의 자기장을 설명. 태양 주기와 무관하게 항상 작동
- **전구 다이나모 (Global dynamo)**: 차등 회전과 나선형 대류에 의한 대규모 자기장 발생. 22년 자기 주기(11년 활동 주기)를 구동. 타코클라인이 핵심적 역할을 할 것으로 추정됨

Miesch는 이 리뷰의 범위를 **대규모 역학** — 차등 회전, 자오 순환, 그리고 이들을 유지하는 각운동량·에너지 수송 과정 — 으로 명확히 한정합니다. 소규모 대류의 세부 구조보다는 그것이 대규모 흐름에 미치는 집합적 효과에 초점을 맞춥니다.

---

### Section 2: Probing the Solar Interior / 태양 내부 탐사

태양 내부를 탐사하는 세 가지 상보적 방법을 리뷰합니다:

#### 2.1 Global Helioseismology / 전구 일진학

태양 내부 회전 프로파일의 가장 강력한 진단 도구입니다:

- **p-mode (pressure modes)**: 음파의 정상파. 태양의 주된 진동 모드로 ~5분 주기 진동으로 관측됨
- **구면 조화 함수 분해**: 표면 속도장을 $Y_\ell^m(\theta, \phi)$로 분해. $\ell$은 구면 조화 차수(수평 파수), $m$은 방위 차수, $n$은 방사 차수
- **회전 분열 (Rotational splitting)**: 비회전 태양에서 $2\ell+1$개의 축퇴된 $m$ 모드가 회전에 의해 분열. 분열 패턴에서 내부 각속도 $\Omega(r, \theta)$를 역산(inversion)
- **역산 기법**: Regularized Least Squares (RLS)와 Optimally Localized Averages (OLA) 두 방법이 상보적으로 사용됨. RLS는 정규화에 민감하고, OLA는 각 격자점에 대해 최적화된 averaging kernel을 구성

핵심 제한: p-mode는 에너지 밀도 분포상 $r \sim 0.4\,R_\odot$ 이하의 깊은 내부에서 민감도가 급격히 떨어집니다. 내부 코어의 회전은 중력 모드(g-mode)로 탐사해야 하지만, g-mode 검출은 아직 확립되지 않았습니다.

#### 2.2 Local Helioseismology / 국소 일진학

표면 아래 수십 Mm까지의 3D 흐름 구조를 매핑합니다:

- **Ring-diagram analysis**: 작은 영역(~15°×15°)의 국소 파워 스펙트럼에서 $k_x$-$k_y$ 단면이 원형이 아닌 타원형 → 흐름 속도와 방향 결정
- **Time-distance helioseismology**: 두 지점 간의 파동 전파 시간 차이에서 경로 평균 흐름을 추정. 더 깊은 층까지 도달 가능
- **Acoustic holography**: 표면 관측에서 내부의 음향 이미지를 재구성. 태양 뒷면의 활동 영역도 검출 가능

이 기법들이 밝혀낸 것:
- Supergranulation의 깊이 구조 (수 Mm)
- 활동 영역 아래의 흐름 구조
- Torsional oscillation의 깊이 의존성
- 다중 세포 자오 순환의 증거

#### 2.3 Surface Observations / 표면 관측

- **Dopplergram**: 시선 속도 측정 → 수평·수직 흐름 직접 관측
- **Feature tracking**: 흑점, 자기 밝은점, supergranule 등을 추적 → 표면 흐름의 패턴
- **irradiance variations**: 태양 총 방사 조도(TSI)의 미세 변동에서 대류의 열적 특성 추론

---

### Section 3: Observations / 관측

이 섹션은 시뮬레이션 검증의 기준이 되는 관측 사실들을 상세히 정리합니다.

#### 3.1 Differential Rotation / 차등 회전

일진학이 밝혀낸 태양 내부 회전의 핵심 특성:

1. **적도-극 대비**: 적도 각속도 $\Omega_{\rm eq}/2\pi \approx 460$ nHz (주기 ~25.2일, 항성 좌표), 극 $\Omega_{\rm pole}/2\pi \approx 330$ nHz (주기 ~35일). 약 30%의 대비. 표면 관측의 표준 피팅:

$$\Omega(\theta)/2\pi = A + B\cos^2\theta + C\cos^4\theta$$

여기서 $A \approx 460$ nHz, $B \approx -60$ nHz, $C \approx -75$ nHz (Snodgrass & Ulrich, 1990). $\theta$는 여위도(colatitude).

2. **거의 방사상 등고선 (Nearly radial contours)**: 중위도(~15°–60°)에서 각속도 등고선이 놀랍게도 거의 방사상(회전축에 수직이 아닌 태양 중심을 향함). 이것은 단순한 Taylor-Proudman 정리 예측(원통형 등고선)과 크게 다르며, Section 4.3.2의 thermal wind 해석이 필요한 핵심 관측 사실입니다.

3. **표면 전단층 (Near-Surface Shear Layer, NSSL)**: 표면 아래 ~35 Mm($r > 0.95\,R_\odot$)에서 각속도가 깊이에 따라 증가. 이 층에서의 방사상 기울기 $\partial\Omega/\partial r > 0$은 위도에 따라 변하며, ~35° 위도에서 부호가 바뀜

4. **대류층 하부**: $r \sim 0.71\,R_\odot$ 근처에서 차등 회전이 급격히 사라지고 타코클라인을 통해 균일 회전으로 전이

#### 3.2 Tachocline / 타코클라인

대류층과 복사 내부 사이의 얇은 전이층으로, 이름은 Spiegel & Zahn (1992)이 그리스어 $\tau\alpha\chi\acute{o}\varsigma$ (빠른)와 "cline" (기울기)에서 만들었습니다:

**관측적 특성화**:
- 회전 프로파일의 전이를 오차 함수(error function)로 피팅:

$$f(r) = \frac{1}{2}\left\{1 + \mathrm{erf}\left[\frac{2(r - r_t)}{\Delta_t}\right]\right\}$$

여기서 $r_t$는 타코클라인 중심 반경, $\Delta_t$는 두께

- **적도에서**: $r_t/R_\odot = 0.693 \pm 0.002$, $\Delta_t/R_\odot = 0.039 \pm 0.013$ (Charbonneau et al., 1999)
- **60° 위도에서**: $r_t/R_\odot = 0.710 \pm 0.002$, $\Delta_t/R_\odot = 0.042 \pm 0.013$
- **Prolate shape (장축 모양)**: 고위도의 $r_t$가 적도보다 약간 높음 — 타코클라인이 완벽한 구가 아님
- 대류층 하부 경계($r_b = 0.713\,R_\odot$)와 거의 겹치지만, 적도에서는 타코클라인의 상당 부분이 복사 영역 안에 위치

**물리적 중요성**:
- 여기서 차등 회전의 전단(shear)이 강한 toroidal 자기장을 생성 ($\Omega$-effect)
- 복사층의 안정한 성층이 자기 부력(magnetic buoyancy)에 의한 자기장 부상을 억제하여 충분히 증폭될 시간을 제공
- 대류 침투(convective penetration)와 내부파(internal waves)가 상호작용하는 동적 영역

#### 3.3 Torsional Oscillations / 비틀림 진동

차등 회전에 중첩된 시간 의존적 변동:

- **11년 주기**: 태양 활동 주기와 동기화된 더 빠른/더 느린 회전의 띠(bands)
- **진폭**: ~2–5 nHz (기본 회전의 ~1%)
- **전파 패턴**: 두 가지 분기(branch)
  - **저위도 분기**: 활동 영역과 함께 적도 쪽으로 이동. "butterfly diagram"과 평행
  - **고위도 분기**: 극 쪽으로 이동하며 새 주기의 시작과 연관
- **깊이**: 대류층 전체에 걸쳐 존재 (단순한 표면 현상이 아님)
- **해석**: Lorentz force feedback (자기장이 흐름에 미치는 역작용)으로 부분적 설명 가능. 열적 교란(active region에 의한)의 역할도 제안됨

#### 3.4 Meridional Circulation / 자오 순환

적도에서 극 방향(또는 그 반대)의 대규모 축대칭 흐름:

- **표면**: ~20 m/s 극향(poleward). Doppler 관측과 feature tracking에서 비교적 일관
- **대류층 내부**: 매우 불확실. 국소 일진학에서 단일 세포(표면 극향, 대류층 하부 적도향)부터 다중 세포(위도·깊이에 따른 여러 순환 루프)까지 다양한 결과
- **대류층 하부**: ~2–3 m/s 적도향이 flux-transport dynamo 모델에서 필요하지만, 직접 관측은 아직 신뢰성 부족
- **시간 변동**: 태양 주기에 따라 변동하며, 활동 극대기에 약해지는 경향

**중요한 점**: 많은 flux-transport dynamo 모델이 깊은 대류층에서의 단일 세포 적도향 자오 순환에 의존하지만, 시뮬레이션은 일관되게 다중 세포 구조를 예측합니다. 이 불일치가 Section 6.4에서 상세히 논의됩니다.

#### 3.5 Giant Cells / 거대 세포

이론적으로 대류층 전체 깊이에 걸치는 대규모 대류 세포가 존재해야 합니다:

- **예측**: 혼합 길이 이론(mixing-length theory)에서 대류 규모 ~ 압력 스케일 높이 $H_P$. 대류층 하부에서 $H_P \sim 50$–60 Mm → 거대 세포 예측
- **관측 상한**: 표면에서 ~5 m/s 이하. 이보다 큰 신호는 검출되지 않음
- **왜 보이지 않는가**: (1) 회전 효과가 대류 규모를 억제, (2) 거대 세포의 시간 규모가 자전 주기보다 길어 관측이 어려움, (3) 표면에서의 신호가 작은 규모의 대류에 의해 가려짐
- **Rossby waves**: 최근 연구에서 시간-위도 스펙트럼에서 Rossby wave 패턴이 검출되기 시작 — 거대 세포와 구별이 필요

#### 3.6 Base of the Convection Zone / 대류층 하부 경계

일진학에서 가장 정밀하게 결정된 태양 구조 매개변수 중 하나:

$$r_b/R_\odot = 0.713 \pm 0.003$$

이 값은 음속 프로파일의 불연속(실제로는 급격한 기울기 변화)에서 결정됩니다. 대류층에서는 단열 온도 기울기(adiabatic gradient), 복사층에서는 복사 온도 기울기(radiative gradient)가 지배하므로, 전이 지점에서 음속 기울기가 변합니다.

**오버슈트(Overshoot)**: 대류가 안정 성층 영역으로 침투하는 정도. 일진학적 제약: $d_{\rm ov} < 0.05\,H_P$ — 매우 좁음. 이것은 Section 8.1의 대류 침투 논의와 직결됩니다.

#### 3.8 Solar Magnetism / 태양 자기

대규모 역학과 밀접하게 연관된 자기 특성들:

- **22년 자기 주기**: 11년 활동 주기(흑점 수)의 두 배. 한 주기의 극성이 다음 주기에 반전
- **Hale's law**: 쌍극 활동 영역(bipolar active regions)의 선행/후행 극성이 반구에 따라 체계적. 매 주기마다 반전
- **Joy's law**: 쌍극 영역의 축이 적도에 대해 체계적으로 기울어짐 (선행점이 적도 쪽)
- **Helicity rules**: 북반구에서 좌선 나선(left-handed), 남반구에서 우선 나선(right-handed)이 우세
- **Butterfly diagram**: 활동 영역의 출현 위도가 주기 초(~40°)에서 주기 말(~5°)까지 적도 쪽으로 이동
- **극 자기장 반전**: 활동 극대기 무렵에 극 자기장 극성이 반전 — 후행점의 자속 수송이 주요 원인

**자기장 강도 추정**: 타코클라인에서 $B \sim 10^4$–$10^5$ G의 toroidal 자기장이 필요 (흑점과 활동 영역의 부상 자속관 모델에서 요구). 이 강한 자기장이 대류층 역학에 역작용할 가능성 → Section 6.5의 MHD 시뮬레이션에서 다룸.

---

### Section 4: Fundamental Concepts / 기본 개념

이 섹션은 리뷰의 이론적 뼈대를 구성합니다. 모든 시뮬레이션 결과의 해석에 필수적인 방정식과 정리들이 제시됩니다.

#### 4.1 Anelastic Approximation / 비탄성 근사

태양 대류 시뮬레이션의 기반이 되는 핵심 근사:

**문제**: 완전 압축성(fully compressible) 방정식에서 음파의 전파 속도($c_s \sim 200$ km/s)가 대류 속도($v \sim 100$ m/s)보다 ~2000배 빠름. 음파를 분해하려면 시간 간격 $\Delta t$가 극도로 작아야 하며 (CFL 조건), 대류의 시간 규모($\sim$ 일–달)를 추적하는 것이 비현실적.

**해법**: 열역학 변수를 기준 상태(reference state)와 섭동(perturbation)으로 분해:

$$\rho = \bar{\rho}(r) + \rho'(r, \theta, \phi, t), \quad P = \bar{P}(r) + P', \quad T = \bar{T}(r) + T', \quad S = \bar{S}(r) + S'$$

기준 상태 $\bar{\rho}$, $\bar{P}$, $\bar{T}$, $\bar{S}$는 구대칭이고 정역학 평형(hydrostatic equilibrium)을 만족:

$$\frac{d\bar{P}}{dr} = -\bar{\rho}g$$

**비탄성 조건**: 연속 방정식에서 $\partial\rho'/\partial t$ 항을 제거:

$$\nabla \cdot (\bar{\rho}\,\mathbf{v}) = 0$$

이것이 음파를 필터링하면서도 밀도의 공간적 변화(대류층에서 ~100배 차이)를 유지합니다. Boussinesq 근사($\rho = \text{const}$ 가정)와 달리 밀도 성층(stratification)을 포함하므로 태양 대류에 훨씬 적합합니다.

**유효 조건**: Mach number $M = v/c_s \ll 1$ — 태양 대류에서 $M \sim 10^{-4}$–$10^{-3}$이므로 매우 잘 성립.

#### 4.2 Energy Conservation / 에너지 보존

열평형(thermal equilibrium)과 에너지 수송의 상세한 분석:

**에너지 플럭스 분해**: 전체 에너지 플럭스는 6가지 성분으로 분해됩니다. 구면 평균하여 방사 방향 성분만 고려하면:

$$\mathcal{F}_r = \mathcal{F}_{\rm KE} + \mathcal{F}_{\rm EN} + \mathcal{F}_{\rm RD} + \mathcal{F}_{\rm PF} + \mathcal{F}_{\rm VD} + \mathcal{F}_{\rm BS} = L_\odot / (4\pi r^2) \tag{Eq.~2}$$

각 성분의 물리적 의미:
- **$\mathcal{F}_{\rm KE}$ (Kinetic Energy flux)**: 운동 에너지의 이류(advection). 대류가 운동 에너지를 운반
- **$\mathcal{F}_{\rm EN}$ (Enthalpy flux)**: 엔탈피 플럭스 $= \bar{\rho}C_P\overline{v_r T'}$. **대류의 주된 에너지 수송 메커니즘**. 상승류가 뜨겁고 하강류가 차가우면 양의 (외향) 플럭스
- **$\mathcal{F}_{\rm RD}$ (Radiative Diffusion flux)**: 복사 확산. 대류층에서는 작지만 하부와 표면 근처에서 중요
- **$\mathcal{F}_{\rm PF}$ (Pressure flux / acoustic flux)**: 압력 일(pressure work). $= \overline{v_r P'}$
- **$\mathcal{F}_{\rm VD}$ (Viscous Diffusion flux)**: 점성 확산에 의한 에너지 수송. 일반적으로 무시 가능
- **$\mathcal{F}_{\rm BS}$ (Buoyancy Source)**: 부력에 의한 잠재 에너지의 운동 에너지로의 전환

**열평형 조건**: 정상 상태에서 구각(spherical shell)을 통과하는 총 에너지 플럭스가 태양 광도와 같아야 합니다:

$$\oint \mathbf{\mathcal{F}} \cdot d\mathbf{A} = L_\odot \tag{Eq.~3}$$

이것이 시뮬레이션이 달성해야 하는 기본 제약 조건입니다.

**열적 시간 규모**: 대류층의 열적 시간 규모(Kelvin-Helmholtz time)는:

$$\tau_{\rm KH} \sim \frac{\int \bar{\rho} C_P \bar{T}\, dV}{L_\odot} \sim 10^5 \text{ yr}$$

이것은 시뮬레이션이 열평형에 도달하는 데 걸리는 시간입니다. 실제 시뮬레이션은 이 시간 규모까지 실행할 수 없으므로, 기준 상태를 태양 구조 모델에서 가져와 초기화합니다.

#### 4.3 Angular Momentum / 각운동량

**이 섹션이 리뷰 전체의 핵심 이론적 뼈대입니다.**

비관성 회전 좌표계에서 단위 질량당 비각운동량(specific angular momentum):

$$\mathcal{L} = \lambda^2 \Omega$$

여기서 $\lambda = r\sin\theta$는 회전축까지의 거리(cylindrical radius), $\Omega$는 총 각속도(기준 좌표계 회전 + 섭동).

**각운동량 수송 방정식** (축대칭 평균):

$$\frac{\partial}{\partial t}\left(\bar{\rho}\lambda^2\langle\Omega\rangle\right) = -\nabla \cdot \left(\bar{\rho}\lambda \langle v_M\rangle \lambda\langle\Omega\rangle + \bar{\rho}\lambda\langle v'_M \lambda v'_\phi\rangle - \frac{\lambda}{4\pi}\langle B_M\rangle\lambda\langle B_\phi\rangle - \frac{\lambda}{4\pi}\langle B'_M \lambda B'_\phi\rangle + \ldots \right)$$

이를 5가지 플럭스로 정리합니다:

1. **MC (Meridional Circulation flux)**: $\bar{\rho}\lambda\langle\mathbf{v}_M\rangle\lambda\langle\Omega\rangle$ — 자오 순환이 각운동량을 이류(advection)
2. **RS (Reynolds Stress flux)**: $\bar{\rho}\lambda\langle v'_r v'_\phi\rangle\hat{r} + \bar{\rho}\lambda\langle v'_\theta v'_\phi\rangle\hat{\theta}$ — 난류 상관(turbulent correlations)에 의한 수송
3. **MS (Maxwell Stress flux)**: $-\frac{\lambda}{4\pi}\langle B'_r B'_\phi\rangle\hat{r} - \frac{\lambda}{4\pi}\langle B'_\theta B'_\phi\rangle\hat{\theta}$ — 자기 응력에 의한 수송
4. **MT (Mean Toroidal field flux)**: $-\frac{\lambda}{4\pi}\langle B_M\rangle\lambda\langle B_\phi\rangle$ — 축대칭 자기장에 의한 수송
5. **VD (Viscous Diffusion)**: 점성 응력에 의한 수송 (실제 태양에서는 무시 가능하나 시뮬레이션에서는 유한)

##### 4.3.1 Reynolds Stress vs Meridional Circulation / Reynolds 응력 대 자오 순환

**핵심 균형**: 정상 상태에서 주된 각운동량 수지(budget)는 Reynolds 응력(RS)과 자오 순환(MC)의 균형입니다:

- **Reynolds stress**: 주로 적도 쪽으로 각운동량을 수송하여 적도의 빠른 회전을 유지하는 데 기여
- **Meridional circulation**: 표면의 극향 흐름이 각운동량을 극 쪽으로 운반 → 차등 회전을 약화시키는 방향으로 작용

이 두 효과의 미묘한 균형이 관측된 차등 회전 프로파일을 결정합니다. Coriolis force가 대류 운동에 체계적인 비등방성을 부여하여 양의(적도향) Reynolds 응력을 생성하는 것이 핵심 메커니즘입니다.

##### 4.3.2 Taylor-Proudman Theorem and Thermal Wind / Taylor-Proudman 정리와 열풍

**Taylor-Proudman 정리**: 빠르게 회전하고 비점성인 유체에서, 정상 상태이며 비선형항이 무시 가능하면:

$$(\mathbf{\Omega}_0 \cdot \nabla)\mathbf{v} = 0$$

즉, 흐름은 회전축 방향으로 변하지 않아야 합니다 → **원통형 등고선(cylindrical contours)**. 각속도의 경우:

$$\frac{\partial\Omega}{\partial z} = 0 \quad (\text{where } z \text{ is parallel to the rotation axis})$$

그런데 태양의 관측된 등고선은 원통형이 아니라 **거의 방사상**입니다!

**Thermal wind 해석**: Taylor-Proudman 균형에서의 이탈은 **부력 효과** (baroclinicity, 등압면과 등밀도면이 일치하지 않음)에 의해 가능합니다. 구면 좌표에서 축대칭 방위각 와도(azimuthal vorticity) 방정식의 정상 상태 균형:

$$\boxed{\Omega_0 \frac{\partial \langle v_\phi \rangle}{\partial z} \approx \frac{g}{2C_P \lambda r}\frac{\partial \langle S \rangle}{\partial \theta}}$$

이것이 **thermal wind equation**입니다. 좌변은 각속도의 $z$-방향(회전축 방향) 기울기, 우변은 위도 방향 엔트로피 기울기입니다.

**물리적 해석**: 극이 적도보다 뜨거우면 ($\partial\langle S\rangle/\partial\theta < 0$ in the Northern Hemisphere, $\theta$ measured from pole) 방사상 방향의 $\Omega$ 등고선이 가능합니다. 수치적으로 극-적도 온도 차이 $\Delta T \sim 10$ K이면 충분 — 매우 작은 양이지만 이것이 관측된 회전 프로파일의 핵심입니다.

이 위도 방향 엔트로피 기울기의 기원은 아직 완전히 이해되지 않았습니다. 대류의 위도 의존적 열수송, 자오 순환에 의한 열 이류, 타코클라인에서의 열 교환 등이 후보로 제시됩니다.

#### 4.4 Meridional Circulation Maintenance / 자오 순환 유지

자오 순환은 자체적으로 구동되는 것이 아니라 다른 힘의 잔여 균형(residual balance)에 의해 유지됩니다:

**축대칭 방위각 와도 방정식** (streamfunction $\Psi$를 통해):

자오 순환의 유선 함수 $\Psi(r, \theta)$를 정의:

$$\bar{\rho}\langle v_r \rangle = \frac{1}{r^2\sin\theta}\frac{\partial\Psi}{\partial\theta}, \quad \bar{\rho}\langle v_\theta \rangle = -\frac{1}{r\sin\theta}\frac{\partial\Psi}{\partial r}$$

방위각 와도의 시간 진화는 다음 힘들에 의해 결정됩니다:
- **Coriolis force**: $2\Omega_0 \frac{\partial(\bar{\rho}\lambda\langle\Omega\rangle)}{\partial z}$ — 회전의 $z$-방향 기울기가 자오 순환을 구동
- **Buoyancy**: 위도 방향 엔트로피 기울기 → 부력 토크
- **Reynolds stress curl**: 난류 응력의 회전(curl)
- **Lorentz force**: 자기 응력의 기여
- **Viscous diffusion**

핵심 통찰: 자오 순환은 차등 회전, 부력, 난류 응력의 **기계적 균형**의 결과물이지, 독립적으로 존재하는 흐름이 아닙니다. 따라서 자오 순환의 구조(단일 세포 vs 다중 세포)는 다른 모든 과정에 의존합니다.

#### 4.5 Solar Dynamo / 태양 다이나모

차등 회전이 자기장 발생과 어떻게 연결되는지의 기본 프레임워크:

**$\Omega$-effect (differential rotation → toroidal field)**:

차등 회전의 전단(shear)이 poloidal 자기장을 감아서(winding) toroidal 자기장을 생성합니다. 유도 방정식의 toroidal 성분에서:

$$\frac{\partial \langle B_\phi \rangle}{\partial t} \sim \lambda (\langle \mathbf{B}_P \rangle \cdot \nabla)\Omega$$

여기서 $\langle \mathbf{B}_P \rangle$는 축대칭 poloidal 자기장. 타코클라인의 강한 전단($\partial\Omega/\partial r$)이 이 과정의 가장 효율적인 장소입니다.

**$\alpha$-effect (helical convection → poloidal field)**:

나선형(helical) 대류가 toroidal 자기장으로부터 poloidal 자기장을 재생성합니다. 평균장 전기역학(mean-field electrodynamics)에서 turbulent electromotive force (EMF):

$$\boldsymbol{\varepsilon} = \langle \mathbf{v}' \times \mathbf{B}' \rangle = \alpha \langle \mathbf{B} \rangle - \eta_t \nabla \times \langle \mathbf{B} \rangle + \ldots$$

- $\alpha$는 kinetic helicity $\langle \mathbf{v}' \cdot (\nabla \times \mathbf{v}') \rangle$와 관련 — Coriolis force가 대류에 나선성을 부여
- $\eta_t$는 turbulent magnetic diffusivity — 자기장을 확산시키지만, 동시에 자기 재연결(reconnection)을 통한 극성 변환에도 기여

**Babcock-Leighton mechanism**: $\alpha$-effect의 대안으로, 표면에 출현한 쌍극 활동 영역의 Joy's law 기울기가 극향 자속 수송을 통해 poloidal 자기장을 재생성. 이 메커니즘은 flux-transport dynamo 모델의 핵심이며, 자오 순환에 의한 자속 수송에 크게 의존합니다.

---

### Section 5: Modeling Solar Convection / 태양 대류 모델링

#### 5.1 The Challenge / 도전

태양 대류를 수치적으로 모델링하는 것이 왜 극도로 어려운지를 정량화합니다:

- **Reynolds number**: $\mathrm{Re} = vL/\nu \sim 10^{14}$ — 지구상 어떤 난류 실험보다도 수 orders of magnitude 높음
- **점성 소산 규모**: $d_\nu \sim L\,\mathrm{Re}^{-3/4} \sim 1$ cm (Kolmogorov 미세 규모)
- **대류층 크기**: $L \sim 200$ Mm = $2 \times 10^{10}$ cm
- **동적 범위**: $L/d_\nu \sim 10^{10}$ — 이것을 해상하려면 격자점이 $(10^{10})^3 = 10^{30}$개 필요!

현재 가장 큰 시뮬레이션도 $\sim 10^9$ 격자점 정도이므로, **모든 태양 대류 시뮬레이션은 Large-Eddy Simulation (LES)**입니다 — 분해되지 않는 작은 규모의 효과를 subgrid-scale (SGS) 모델로 대체합니다.

**Rayleigh number**: 실제 태양 $\mathrm{Ra} \sim 10^{22}$이지만, 시뮬레이션은 $\mathrm{Ra} \sim 10^4$–$10^6$ — 약 16 orders of magnitude 부족

#### 5.2 ASH Code / ASH 코드

Anelastic Spherical Harmonic (ASH) code는 이 리뷰의 주요 시뮬레이션 도구입니다:

- **수치 방법**: 수평 방향 — 구면 조화 함수(pseudospectral), 수직 방향 — Chebyshev 다항식, 시간 — 반암묵적(semi-implicit, Crank-Nicolson for linear + Adams-Bashforth for nonlinear)
- **계산 영역**: 구각(spherical shell), 보통 $0.62\,R_\odot \leq r \leq 0.96\,R_\odot$ — 타코클라인 상부부터 표면 아래까지
- **비탄성 근사**: Section 4.1의 방정식 사용 — 음파 제거, 밀도 성층 유지
- **SGS 모델**: 표준적으로 enhanced diffusivities 사용 — 실효 확산 계수가 분자 값보다 $\sim 10^{10}$배 큼

**해상도**: 일반적으로 $\ell_{\max} \sim 85$–$340$ (구면 조화 차수), Chebyshev $N_r \sim 48$–$98$. 이것은 약 100–200 km 수평 해상도에 해당 — supergranulation 규모 정도

---

### Section 6: Global Simulation Results / 전구 시뮬레이션 결과

이 섹션은 ASH 코드를 이용한 3D 시뮬레이션의 주요 결과와 한계를 상세히 분석합니다.

#### 6.1 Historical Context / 역사적 맥락

- **Gilman (1977, 1978, 1979)**: 최초의 3D 전구 대류 시뮬레이션. Boussinesq 근사. 뚜렷한 **바나나 세포(banana cells)** — 적도에 정렬된 남북 방향 대류 세포. 차등 회전을 재현했지만, 과도하게 규칙적인 대류 패턴
- **Gilman & Miller (1986)**: 비탄성 근사로 확장. 밀도 성층의 효과를 처음 포함
- **Glatzmaier & Gilman (1982)**: MHD 확장. 자기장의 역효과(back-reaction) 포함
- **Brun & Toomre (2002), Miesch et al. (2000)**: **ASH 코드의 도입**. 높은 해상도에서의 난류적 대류 달성. 바나나 세포가 깨지고 복잡한 하강류 네트워크(downflow network) 출현

핵심 전환: **충분히 높은 Rayleigh number에서 대류의 성격이 질적으로 변합니다.** 층류적 바나나 세포 → 난류적 하강류 네트워크. 이 전환이 차등 회전의 생성 메커니즘을 근본적으로 변화시킵니다.

#### 6.2 Convection Structure / 대류 구조

높은 Ra 시뮬레이션에서의 대류 패턴:

**수평 구조**:
- **적도 근처**: 남-북 방향으로 정렬된 하강류 레인(lanes). Coriolis force에 의한 바나나 세포의 잔재이지만, 훨씬 불규칙적. 이 N-S 정렬이 **적도향 각운동량 수송의 핵심 메커니즘**
- **중위도**: 하강류 레인이 더 복잡해지고 교차하는 네트워크 형성
- **고위도**: 대류가 거의 등방적(isotropic) — 회전축과의 정렬 약화

**수직 구조**:
- **하강류(downflow)**: 좁고 강하며 긴 수명. 밀도 성층에 의해 깊이 갈수록 좁아짐 (mass flux conservation)
- **상승류(upflow)**: 넓고 약하며 짧은 수명. 하강류 네트워크 사이를 채움
- **비대칭성**: 하강류가 상승류보다 체계적으로 강함 — 대류의 근본적 비대칭성. 이것이 enthalpy flux와 kinetic energy flux 사이의 관계를 결정

**Rossby number의 역할**:

$$\mathrm{Ro} = \frac{v}{2\Omega_0 L}$$

$\mathrm{Ro} \ll 1$ (회전 우세): 바나나 세포, 원통형 등고선
$\mathrm{Ro} \gg 1$ (대류 우세): 등방 대류, 회전 효과 약화
태양: $\mathrm{Ro} \sim 1$ (표면 근처) ~ $\mathrm{Ro} \ll 1$ (대류층 하부) — 깊이에 따라 전이

#### 6.3 Differential Rotation / 차등 회전

**5가지 핵심 도전 과제 (Five Outstanding Challenges)**:

이것이 이 리뷰의 가장 중요한 목록 중 하나입니다. 시뮬레이션이 해결해야 할 관측과의 불일치:

1. **Polar vortex problem**: 시뮬레이션이 극에서 비현실적으로 강한 와류(vortex) — 빠른 극 회전 — 을 생성하는 경향. 관측에서는 극이 가장 느리게 회전. 원인: 고위도에서 Coriolis force에 의한 각운동량 집중

2. **Angular velocity contrast**: 적도-극 각속도 차이가 관측(~30%)보다 작게 나오는 경향. Reynolds 응력에 의한 적도향 수송이 불충분

3. **Nearly radial $\Omega$ contours**: Section 4.3.2의 thermal wind에서 논의한 것처럼, 관측된 거의 방사상 등고선을 재현하려면 적절한 위도 방향 엔트로피 기울기가 필요. 시뮬레이션에서 이 기울기가 자기일관적으로(self-consistently) 생성되는지는 아직 불완전

4. **Rotational shear layers**: 표면 전단층(NSSL)과 타코클라인을 동시에 재현하는 시뮬레이션은 아직 없음. 표면 전단층은 경계 조건과 근표면 역학에 민감하고, 타코클라인은 복사층과의 결합이 필요

5. **Torsional oscillations**: 11년 주기 비틀림 진동의 재현에는 태양 주기적 자기장이 필요하지만, 시뮬레이션이 아직 현실적인 태양 주기를 재현하지 못함

**각운동량 수지의 메커니즘**:

시뮬레이션에서 관측된 차등 회전의 유지 메커니즘:

- **적도향 각운동량 수송**: N-S 하강류 레인이 핵심. 이 레인들은 적도에서 방위각(φ) 방향으로 기울어져 있어 $\langle v'_\theta v'_\phi \rangle \neq 0$ (위도 방향 Reynolds 응력)를 생성. 이것이 각운동량을 적도 쪽으로 수송
- **자오 순환의 역할**: MC는 RS에 대해 반대 방향으로 작용 — 표면의 극향 흐름이 각운동량을 극 쪽으로 운반. 전체 균형은 RS가 MC보다 약간 우세하여 적도의 빠른 회전을 유지
- **Thermal wind**: 대류층 하부의 위도 엔트로피 기울기(극이 약간 더 뜨거움)가 Taylor-Proudman 균형에서의 이탈을 제공하여 방사상 $\Omega$ 등고선을 생성

#### 6.4 Meridional Circulation / 자오 순환

시뮬레이션에서의 자오 순환 특성:

- **다중 세포 구조 (Multi-cell structure)**: 단일 세포(표면 극향, 하부 적도향)가 아닌, 위도와 깊이에 따른 여러 개의 순환 세포. 이것은 flux-transport dynamo 모델의 기본 가정과 충돌
- **시간 변동성**: 자오 순환이 시간에 따라 크게 변동. 평균적인 패턴은 있지만, 순간적인 구조는 불규칙적
- **진폭**: 시뮬레이션에서 ~20 m/s로 관측과 비슷한 크기
- **오버슈트 영역에서의 적도향 흐름**: 대류층 아래의 오버슈트 영역에서 적도향 흐름이 나타남 — gyroscopic pumping (Coriolis force에 의한 축대칭 흐름의 구동)에 의한 것

**시뮬레이션과 flux-transport dynamo의 긴장**:
Flux-transport dynamo 모델(Dikpati & Charbonneau, 1999 등)은 깊은 대류층에서의 지속적이고 일관된 적도향 자오 순환에 의존하여 자기 자속을 적도로 운반합니다. 그러나 3D 시뮬레이션은 이러한 단순한 순환을 생성하지 않습니다. 이것이 solar dynamo 이론의 중요한 미해결 문제입니다.

#### 6.5 Dynamo / 다이나모

ASH 코드의 MHD 시뮬레이션(Case M 시리즈) 결과:

**Case M3** (Brun, Miesch, Toomre, 2004):
- **다이나모 유형**: $\alpha^2\Omega$-type — $\alpha$-effect와 $\Omega$-effect 모두 기여
- **변동 자기장 우세**: 총 자기 에너지의 ~98%가 변동(fluctuating) 성분, ~2%만 축대칭(mean) 성분. 평균장 이론(mean-field theory)의 가정과 큰 괴리
- **주기적 행동 부재**: 관측된 22년 자기 주기를 재현하지 못함. 자기장이 카오틱하게 변동
- **$\Omega$-effect**: 타코클라인의 전단이 포함되면 강한 toroidal 자기장 증폭이 가능하지만, 타코클라인을 자기일관적으로 포함하는 시뮬레이션은 아직 달성되지 않음
- **자기장의 역작용**: 자기장이 대류와 차등 회전에 역작용하여 약간 변형하지만, 대규모 구조를 질적으로 바꾸지는 않음 (현재 시뮬레이션 수준에서)

**핵심 한계**: 시뮬레이션의 magnetic Prandtl number $\mathrm{Pm} = \nu/\eta$가 ~1이지만 태양에서는 $\mathrm{Pm} \sim 10^{-2}$–$10^{-6}$. 이 불일치가 다이나모 거동에 질적인 차이를 야기할 수 있음.

---

### Section 7: Improvements Needed / 필요한 개선

#### 7.1 Resolution / 해상도

**수렴(convergence) 문제**: 해상도를 높이면 결과가 수렴하는 것이 아니라 **질적으로 변합니다**:
- 낮은 해상도: 바나나 세포, 강한 차등 회전
- 높은 해상도: 난류적 네트워크, 더 약한 차등 회전
- 더 높은 해상도에서는 어떻게 될지 불확실

이것은 단순히 "해상도를 높이면 해결된다"는 접근이 틀릴 수 있음을 시사합니다. 대류의 자기조직화(self-organization) 과정 — 예를 들어 inverse cascade, 역계단(anti-diffusive) 수송 등 — 이 충분히 분해되어야 비로소 태양과 유사한 행동이 나타날 수 있습니다.

#### 7.2 Subgrid-Scale (SGS) Modeling / 격자하 규모 모델링

현재 사용되는 SGS 모델의 한계:

- **Smagorinsky model**: $\nu_{\rm SGS} = (C_S \Delta)^2 |S|$ 형태의 확산. 항상 확산적(dissipative) — 역에너지 캐스케이드(inverse cascade)를 포착할 수 없음
- **Dynamic model**: $C_S$를 국소적으로 계산하여 음의 점성(backscatter)을 허용. 더 물리적이지만 수치적으로 불안정할 수 있음
- **비확산적 수송의 필요성**: Reynolds 응력의 비확산적(non-diffusive) 성분 — Λ-effect (turbulent angular momentum transport that is not proportional to the local gradient) — 이 차등 회전 유지에 중요할 수 있으나, 현재 SGS 모델에 포함되지 않음

#### 7.3 Boundary Conditions / 경계 조건

- **상부 경계**: 표면($r = R_\odot$)의 복잡한 물리(복사 냉각, 이온화, 자기 집중)를 포함하기 어려움. 보통 $r \sim 0.96\,R_\odot$에서 응력이 없는(stress-free) 조건 사용 → 표면 전단층(NSSL) 재현 불가
- **하부 경계**: 타코클라인과 복사층의 결합이 핵심. 단순한 불투과(impenetrable) 경계 → 타코클라인 역학 배제. 오버슈트를 허용하는 확장된 계산 영역이 필요

---

### Section 8: Tachocline Dynamics / 타코클라인 역학

이 리뷰의 후반부 핵심으로, 타코클라인의 물리를 깊이 있게 다룹니다.

#### 8.1 Convective Penetration / 대류 침투

대류가 안정 성층 영역으로 침투하는 두 가지 모드:

**Overshoot vs Penetration**:
- **Overshoot**: 대류 플룸이 안정층에 들어가지만 열적 구조를 바꾸지 않음 (단열 기울기 유지 안됨). 좁은 영역에 한정
- **Penetration**: 대류의 혼합이 충분히 강하여 안정층의 온도 기울기를 단열 기울기로 바꿈. 더 넓은 영역에 영향

**Stiffness parameter**: 침투 깊이를 결정하는 핵심 무차원 수:

$$S_t = \frac{N^2 d^2}{\kappa v}$$

여기서 $N$은 Brunt-Väisälä 진동수, $d$는 대류 규모, $\kappa$는 열확산 계수, $v$는 대류 속도. $S_t$가 클수록(안정 성층이 강할수록) 침투 깊이가 얕습니다.

**수치 시뮬레이션 결과**: 침투 깊이가 $S_t^{-1/4}$에 비례하는 스케일링이 관측됨. 태양의 $S_t \sim 10^7$에서는 매우 얕은 침투 — 일진학적 상한($d_{\rm ov} < 0.05\,H_P$)과 일관.

#### 8.2 Instabilities / 불안정성

타코클라인에서 작용할 수 있는 여러 불안정성:

**전단 불안정성 (Shear instabilities)**:
- **Richardson criterion**: 안정 성층된 전단 흐름의 불안정 조건:
$$\mathrm{Ri} = \frac{N^2}{(\partial v/\partial z)^2} < \frac{1}{4}$$
- 태양 타코클라인에서 $\mathrm{Ri} \sim 10^3$–$10^5$ — Richardson 기준으로는 안정! 그러나 확산(열확산, $\kappa \gg \nu$)이 안정화 효과를 약화시킬 수 있음

**자기 전단 불안정성 (Magneto-shear instabilities)**:
- **Gilman-Fox-Dikpati instability**: 타코클라인의 toroidal 자기장과 차등 회전의 위도 전단이 결합하여 비축대칭 불안정성 발생. 이것이 toroidal 자기장의 파괴적 재배치를 야기할 수 있음
- **Tipping instability** (Cally, 2001): 타코클라인에 갇힌 toroidal 자기장 관(flux tube)이 기울어져 불안정해지는 과정
- **MRI (Magnetorotational instability)**: $\partial\Omega^2/\partial\lambda < 0$인 영역에서 발생. 타코클라인의 일부 영역에서 가능하나 안정 성층이 억제

#### 8.3 Rotating Stratified Turbulence / 회전 성층 난류

타코클라인의 난류는 일반적인 3D 난류와 질적으로 다릅니다:

**2D 성격**: 안정 성층 + 빠른 회전 → 수직 운동 억제 → 거의 2D 흐름. 이것이 에너지 캐스케이드의 방향을 바꿉니다:
- 3D 난류: 에너지가 큰 규모 → 작은 규모 (forward cascade)
- 2D 난류: 에너지가 작은 규모 → 큰 규모 (**inverse cascade**)

**Rhines scale**: 난류 에너지의 역캐스케이드가 멈추는 규모:

$$L_\beta = \sqrt{\frac{U}{\beta}}$$

여기서 $\beta = d(2\Omega_0\cos\theta)/dy$는 행성 와도 기울기(planetary vorticity gradient), $U$는 특성 속도. 이 규모 이상에서는 Rossby wave로의 에너지 전환이 일어남.

**Rossby deformation radius**: 부력과 회전 효과가 경쟁하는 규모:

$$L_D = \frac{N\Delta_t}{2\Omega_0}$$

태양 타코클라인에서 $L_D \sim$ 수 Mm — 이 규모가 타코클라인 내부 구조의 특성 길이를 결정할 수 있음.

#### 8.4 Internal Waves / 내부파

대류 침투에 의해 여기된 중력파가 타코클라인과 복사 내부의 역학에 영향:

- **여기 메커니즘**: 오버슈팅 대류 플룸이 안정 성층에 부딪히면서 중력파(internal gravity waves) 발생
- **임계층 (Critical layers)**: 파동의 위상 속도가 평균 흐름 속도와 같아지는 층에서 파동이 흡수. 이것이 angular momentum의 비국소적 수송을 야기
- **QBO 유사체 (Quasi-Biennial Oscillation analogy)**: 지구 적도 성층권의 QBO는 내부 중력파의 임계층 흡수에 의해 구동되는 대류 진동(oscillating mean flow)입니다. 태양 타코클라인에서도 유사한 파동 구동 대류 흐름이 존재할 수 있음
- **파동 플럭스 추정**: 시뮬레이션에서 대류 침투에 의한 중력파의 운동량 플럭스가 추정되었으나, 타코클라인의 회전 프로파일을 설명하기에 충분한지는 아직 불확실

#### 8.5 Tachocline Confinement / 타코클라인 가둠

**이 리뷰의 가장 핵심적인 미해결 문제: 왜 타코클라인은 이렇게 얇은가?**

문제의 본질: 차등 회전의 전단이 점성·열확산에 의해 복사 내부로 퍼져야(spread) 합니다. 만약 아무것도 막지 않으면, 태양 나이(~$4.6 \times 10^9$ yr) 동안 전단이 태양 핵까지 퍼질 것입니다. 그런데 관측은 $\Delta_t \sim 0.04\,R_\odot$의 얇은 타코클라인을 보여줍니다.

**Spiegel & Zahn (1992) 모델 — 순수 유체역학적 가둠**:

비등방 난류 확산에 의한 가둠을 제안:
- 안정 성층에서의 2D 난류는 수평 확산이 수직 확산보다 훨씬 강함 ($\nu_H \gg \nu_V$)
- 수평 난류 확산이 위도 방향 차등 회전을 균일화하여 전단의 복사 내부 침투를 억제
- 예측되는 타코클라인 두께:

$$\boxed{\frac{\Delta_t}{r_t} \sim \left(\frac{\Omega}{N}\right)^{1/2}\left(\frac{\kappa_r}{\nu_H}\right)^{1/4}} \tag{Eq.~30}$$

여기서 $\kappa_r$은 방사 열확산, $\nu_H$는 수평 난류 점성, $N$은 Brunt-Väisälä 진동수.

**문제**: 이 모델의 타코클라인은 관측보다 두꺼운 경향이 있으며, 필요한 $\nu_H$ 값이 물리적으로 의문시됨. 또한, 2D 난류의 역캐스케이드(inverse cascade)가 오히려 차등 회전을 **생성**할 수 있어 균일화 가정이 깨질 수 있음.

**Gough & McIntyre (1998) 모델 — 자기장에 의한 가둠**:

복사 내부에 화석(fossil) poloidal 자기장이 존재하여 타코클라인의 확산을 억제:

- **물리적 메커니즘**: 복사 내부의 화석 자기장이 Ferraro의 법칙(등회전의 정리, $\mathbf{B} \cdot \nabla\Omega = 0$)에 의해 균일 회전을 강제. 대류층의 차등 회전은 이 자기장이 존재하는 곳까지만 침투 가능
- **Gyroscopic pumping**: Coriolis force에 의한 자오 순환이 대류층에서 복사 내부로 물질을 펌핑하여, 화석 자기장의 확산을 억제하는 역할
- 예측되는 타코클라인 두께:

$$\boxed{\frac{\Delta_t}{r_t} \sim \left(\frac{4\pi\bar{\rho}\nu\eta}{r_t^2 B_0^2}\right)^{1/4}} \tag{Eq.~31}$$

여기서 $\eta$는 자기 확산(Ohmic diffusivity), $B_0$는 화석 자기장 강도.

**필요한 자기장**: 관측된 $\Delta_t$를 재현하려면 $B_0 \sim 0.1$–$1$ G 정도면 충분. 더 보수적인 추정($\eta$ 대신 $\kappa_r$ 사용)에서도 $B_0 \geq 10^{-6}$ G로 극히 약한 자기장만 있으면 됨.

**두 모델의 비교**:

| 특성 | Spiegel & Zahn (1992) | Gough & McIntyre (1998) |
|------|----------------------|------------------------|
| 메커니즘 | 비등방 수평 난류 확산 | 화석 poloidal 자기장 |
| 핵심 매개변수 | $\nu_H$ (수평 점성) | $B_0$ (자기장 강도) |
| 필요 조건 | $\nu_H \gg \nu_V$ | $B_0 \geq 10^{-6}$ G |
| 문제점 | 역캐스케이드가 차등 회전 생성 가능 | 자기장 가둠(confinement) 자체의 유지 |
| 현재 평가 | 불충분할 가능성 | 더 유력한 후보 |

**미해결 과제**: Gough & McIntyre 모델도 자체적인 어려움이 있습니다:
- 화석 자기장이 대류층으로 확산하지 않도록 유지하는 메커니즘 필요 (자오 순환의 downwelling이 자기장을 가둠에 역할)
- 화석 자기장의 기원과 안정성 문제
- 2D 축대칭 모델에서의 검증이 아직 완전하지 않음

---

### Section 9: Conclusion / 결론

Miesch는 다음과 같이 주요 결론을 정리합니다:

1. **차등 회전의 유지**: Reynolds 응력(적도향 각운동량 수송)과 열풍(thermal wind, 위도 방향 엔트로피 기울기에 의한 Taylor-Proudman 이탈)의 조합으로 대략적으로 설명 가능. 그러나 5가지 세부 도전 과제가 남아 있음

2. **자오 순환**: 단순한 단일 세포가 아닌 다중 세포 구조이며 시간 변동성이 큼. 이것은 flux-transport dynamo 모델의 기본 가정에 심각한 의문을 제기

3. **타코클라인 가둠**: 순수 유체역학적 메커니즘(Spiegel & Zahn)보다는 화석 자기장(Gough & McIntyre)이 더 유력. 그러나 자기일관적(self-consistent) 모델은 아직 달성되지 않음

4. **결정적으로 부족한 것**: 대류층과 타코클라인을 동시에 포함하는 자기일관적 전구 시뮬레이션. 이것이 향후 연구의 핵심 목표

5. **해상도와 SGS 모델의 한계**: 현재 시뮬레이션은 태양의 복잡한 난류 역학을 충분히 포착하지 못하며, 수렴(convergence)이 보장되지 않음

---

## Key Takeaways / 핵심 시사점

1. **태양 내부 회전은 일진학(helioseismology)에 의해 정밀하게 결정되었다.** 적도-극 ~30% 각속도 대비, 중위도의 거의 방사상 등고선, $\Delta_t \sim 0.04\,R_\odot$의 얇은 타코클라인이 모든 내부 역학 모델이 재현해야 할 관측적 제약 조건이다.

2. **차등 회전은 Reynolds 응력과 자오 순환의 미묘한 균형으로 유지된다.** Reynolds 응력(적도에서 N-S 정렬된 하강류 레인에 의한 $\langle v'_\theta v'_\phi \rangle$)이 각운동량을 적도로 수송하고, 자오 순환은 이에 반대로 작용한다. 이 두 과정의 잔여 균형이 관측된 차등 회전을 결정한다.

3. **Taylor-Proudman 정리의 이탈이 열풍(thermal wind)으로 설명된다.** 핵심 방정식 $\Omega_0 \partial\langle v_\phi \rangle/\partial z \approx (g/2C_P\lambda r)\partial\langle S\rangle/\partial\theta$에서, 극-적도 ~10 K의 엔트로피 차이가 원통형 회전에서 방사상 회전으로의 전이를 설명한다. 이 작은 온도 차이가 대규모 회전 구조를 결정하는 것은 놀라운 사실이다.

4. **자오 순환은 단일 세포가 아닌 다중 세포 구조로, 시간 변동성이 크다.** 이것은 flux-transport dynamo 모델의 기본 가정(깊은 대류층에서의 지속적 적도향 흐름)과 정면으로 충돌하며, 태양 주기 예측 모델의 근본적 재검토를 요구한다.

5. **타코클라인의 얇음은 화석 자기장에 의해 유지될 가능성이 높다.** Spiegel & Zahn의 순수 유체역학적 모델은 역캐스케이드 문제로 불충분하며, Gough & McIntyre의 화석 poloidal 자기장 모델이 ~0.1–1 G의 약한 자기장만으로 관측을 설명할 수 있다.

6. **3D 전구 시뮬레이션은 아직 수렴하지 않았다.** Reynolds number가 태양보다 ~8–10 orders of magnitude 낮고, 해상도를 높이면 결과가 질적으로 변한다. 현재의 LES 결과를 태양에 직접 외삽하는 것은 위험하며, SGS 모델의 개선이 필수적이다.

7. **태양 다이나모 문제는 아직 미해결이다.** 시뮬레이션에서 자기 에너지의 98%가 변동 성분이고 주기적 행동이 없다. 현실적인 태양 주기를 재현하려면 대류층-타코클라인을 동시에 포함하는 자기일관적 MHD 시뮬레이션이 필요하다.

8. **이 리뷰는 태양 내부 역학의 "지도"를 제공한다.** 관측(Section 3), 이론(Section 4), 시뮬레이션(Section 6), 타코클라인 물리(Section 8)를 체계적으로 연결하여, 향후 20년간의 연구 방향을 설정했다. 이후 ASH 코드의 후속 개발과 다른 코드(Rayleigh, MPS/Boris 등)의 발전이 여기서 제기된 도전 과제들을 순차적으로 해결해 나가고 있다.

---

## Mathematical Summary / 수학적 요약

### Angular Momentum Balance Chain / 각운동량 균형 사슬

**Step 1 — Anelastic continuity (음파 필터링)**:
$$\nabla \cdot (\bar{\rho}\,\mathbf{v}) = 0$$

**Step 2 — Angular momentum transport equation (축대칭 평균)**:
$$\frac{\partial}{\partial t}\left(\bar{\rho}\lambda^2\langle\Omega\rangle\right) = -\nabla \cdot \left(\mathbf{F}_{\rm MC} + \mathbf{F}_{\rm RS} + \mathbf{F}_{\rm MS} + \mathbf{F}_{\rm MT} + \mathbf{F}_{\rm VD}\right)$$

where:
- $\mathbf{F}_{\rm MC} = \bar{\rho}\lambda\langle\mathbf{v}_M\rangle\lambda\langle\Omega\rangle$ (meridional circulation)
- $\mathbf{F}_{\rm RS} = \bar{\rho}\lambda\langle v'_i v'_\phi\rangle$ (Reynolds stress)
- $\mathbf{F}_{\rm MS} = -\frac{\lambda}{4\pi}\langle B'_i B'_\phi\rangle$ (Maxwell stress)
- $\mathbf{F}_{\rm MT} = -\frac{\lambda}{4\pi}\langle B_M\rangle\lambda\langle B_\phi\rangle$ (mean toroidal field)

**Step 3 — Thermal wind equation (Taylor-Proudman이탈)**:
$$\Omega_0 \frac{\partial \langle v_\phi \rangle}{\partial z} = \frac{g}{2C_P \lambda r}\frac{\partial \langle S \rangle}{\partial \theta}$$

$\partial\langle S\rangle/\partial\theta \neq 0$ → non-cylindrical $\Omega$ contours

**Step 4 — Tachocline confinement (Gough & McIntyre)**:
$$\frac{\Delta_t}{r_t} \sim \left(\frac{4\pi\bar{\rho}\nu\eta}{r_t^2 B_0^2}\right)^{1/4}$$

**Step 5 — Tachocline confinement (Spiegel & Zahn)**:
$$\frac{\Delta_t}{r_t} \sim \left(\frac{\Omega}{N}\right)^{1/2}\left(\frac{\kappa_r}{\nu_H}\right)^{1/4}$$

**Step 6 — Dynamo induction ($\Omega$-effect)**:
$$\frac{\partial \langle B_\phi \rangle}{\partial t} \sim \lambda(\langle\mathbf{B}_P\rangle \cdot \nabla)\Omega$$

**Step 7 — Mean-field EMF ($\alpha$-effect)**:
$$\boldsymbol{\varepsilon} = \langle\mathbf{v}' \times \mathbf{B}'\rangle = \alpha\langle\mathbf{B}\rangle - \eta_t\nabla \times \langle\mathbf{B}\rangle$$

---

## Paper in the Arc of History / 역사적 맥락의 타임라인

```
1942 Alfvén ──────── MHD 이론 확립, Alfvén waves
  │
1961 Leighton et al. ── 5분 진동 발견 → 일진학의 씨앗
  │
1977 Gilman ─────── 최초 3D 전구 대류 시뮬레이션 (Boussinesq)
  │                   → 바나나 세포, 차등 회전 재현
  │
1984 Duvall et al. ──── p-mode 회전 분열 → 내부 회전 최초 측정
  │
1985 Brown et al. ─── 적도 회전 프로파일 helioseismic 관측
  │
1988 Thompson (GONG) ── 전구 일진학 역산 → 내부 회전 매핑
  │
1992 Spiegel & Zahn ── ★ 타코클라인 명명, 비등방 난류 확산 가둠 모델
  │
1995 SOHO 발사 ───── MDI → 고정밀 일진학 데이터
  │
1998 Gough & McIntyre ── ★ 화석 자기장 가둠 모델
  │
1998 Schou et al. ───── MDI/GONG 내부 회전 정밀 매핑
  │
1999 Clune et al. ───── ASH 코드 소개 (비탄성 구면 조화)
  │
2000 Miesch et al. ──── ASH 난류 대류, 차등 회전 시뮬레이션
  │
2002 Brun & Toomre ── 높은 Ra에서 하강류 네트워크 → 바나나 세포 소멸
  │
2004 Brun, Miesch, Toomre ── ASH MHD 다이나모 (Case M3)
  │
2005 ★ 이 리뷰 (LRSP 2, 1) ★ ── 대류층+타코클라인 역학의 포괄적 정리
  │                               5가지 도전 과제 명문화
  │
2006 Miesch et al. ──── 타코클라인 포함 시뮬레이션 시도
  │
2011 Hotta et al. ───── 고해상도 ASH → polar vortex 완화
  │
2015 Hotta, Rempel, Yokoyama ── 극고해상도 (ℓ_max ~ 1500) 시뮬레이션
  │                               소규모 자기장의 역할 발견
  │
  ▼ (이후 연구)
Rayleigh 코드, MPS/Boris 코드, Pencil Code 등 다양한 전구 코드 발전
Featherstone & Miesch (2015), Matilsky (2020s) — 도전 과제 순차적 해결 시도
```

---

## Connections to Other Papers / 다른 논문과의 연결

| Paper | 연결 | 관계 |
|-------|------|------|
| Spiegel & Zahn (1992) | **전제/대비** | 타코클라인 명명 및 비등방 난류 확산 가둠 모델. 이 리뷰의 Section 8.5에서 비판적 검토 |
| Gough & McIntyre (1998) | **핵심** | 화석 자기장 가둠 모델. 현재 가장 유력한 타코클라인 가둠 메커니즘 |
| Gilman (1977, 1979) | **역사** | 최초 3D 전구 대류 시뮬레이션. ASH의 선조 |
| Brun & Toomre (2002) | **기반** | ASH 코드 고해상도 결과. 바나나 세포→난류 전이 |
| Brun, Miesch, Toomre (2004) | **기반** | ASH MHD 다이나모 (Case M3). Section 6.5의 핵심 |
| Schou et al. (1998) | **관측** | MDI/GONG 내부 회전 매핑. Section 3의 관측 기반 |
| Charbonneau et al. (1999) | **관측** | 타코클라인 두께·위치의 정밀 결정 |
| Dikpati & Charbonneau (1999) | **긴장** | Flux-transport dynamo 모델. 단일 세포 자오 순환 가정이 시뮬레이션 결과와 충돌 |
| Parker (1955, 1993) | **이론** | 태양 다이나모 이론 ($\alpha\Omega$-dynamo). Section 4.5의 기반 |
| Hotta, Rempel, Yokoyama (2015) | **후속** | 극고해상도 시뮬레이션. 이 리뷰의 도전 과제 일부 해결 |
| LRSP #1: Wood (2004) | **동료** | 같은 LRSP 시리즈. 항성풍-태양풍 연결 |
| Charbonneau (2010) — LRSP | **후속** | Solar dynamo 리뷰. 이 논문의 다이나모 논의를 대폭 확장 |

---

## References / 참고문헌

- Miesch, M.S., "Large-Scale Dynamics of the Convection Zone and Tachocline", *Living Rev. Solar Phys.*, 2, 1, 2005. DOI: 10.12942/lrsp-2005-1
- Spiegel, E.A., Zahn, J.-P., "The solar tachocline", *Astron. Astrophys.*, 265, 106–114, 1992.
- Gough, D.O., McIntyre, M.E., "Inevitability of a magnetic field in the Sun's radiative interior", *Nature*, 394, 755–757, 1998.
- Schou, J. et al., "Helioseismic Studies of Differential Rotation in the Solar Envelope by the Solar Oscillations Investigation Using the Michelson Doppler Imager", *Astrophys. J.*, 505, 390–417, 1998.
- Brun, A.S., Toomre, J., "Turbulent Convection under the Influence of Rotation: Sustaining a Strong Differential Rotation", *Astrophys. J.*, 570, 865–885, 2002.
- Brun, A.S., Miesch, M.S., Toomre, J., "Global-Scale Turbulent Convection and Magnetic Dynamo Action in the Solar Envelope", *Astrophys. J.*, 614, 1073–1098, 2004.
- Gilman, P.A., "Nonlinear Dynamics of Boussinesq Convection in a Deep Rotating Spherical Shell — I", *Geophys. Astrophys. Fluid Dyn.*, 8, 93–135, 1977.
- Charbonneau, P., Christensen-Dalsgaard, J., Henning, R., Larsen, R.M., Schou, J., Thompson, M.J., Tomczyk, S., "Helioseismic Constraints on the Structure of the Solar Tachocline", *Astrophys. J.*, 527, 445–460, 1999.
- Dikpati, M., Charbonneau, P., "A Babcock-Leighton Flux Transport Dynamo with Solar-Like Differential Rotation", *Astrophys. J.*, 518, 508–520, 1999.
- Parker, E.N., "Hydromagnetic Dynamo Models", *Astrophys. J.*, 122, 293–314, 1955.
- Hotta, H., Rempel, M., Yokoyama, T., "High-resolution Calculations of the Solar Global Convection with the Reduced Speed of Sound Technique", *Astrophys. J.*, 798, 51, 2015.
