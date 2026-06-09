---
title: "Solar Modulation of Cosmic Rays"
authors: [Marius S. Potgieter]
year: 2013
journal: "Living Reviews in Solar Physics"
doi: "10.12942/lrsp-2013-3"
topic: Living_Reviews_in_Solar_Physics
tags: [cosmic-rays, heliosphere, solar-modulation, solar-cycle, Parker-transport-equation, diffusion, drift, charge-sign-modulation, GMIR]
status: completed
date_started: 2026-04-23
date_completed: 2026-04-23
---

# 33. Solar Modulation of Cosmic Rays / 우주선의 태양 변조

---

## 1. Core Contribution / 핵심 기여

**English**: Potgieter (2013) provides a comprehensive Living Reviews article on the solar modulation of galactic cosmic rays (GCRs) and anomalous cosmic rays (ACRs) in the heliosphere — a half-century of progress since Parker (1965) formulated the transport equation. The paper distills the physics of modulation into four basic mechanisms encoded in the Parker transport equation (TPE): (1) outward convection by the solar wind, (2) diffusion along and across the turbulent heliospheric magnetic field (HMF) described by a full 3D diffusion tensor with parallel κ_∥ and perpendicular κ_⊥ components, (3) gradient, curvature, and current-sheet drifts (represented by the antisymmetric drift coefficient K_A) that produce the ∼22-year cycle and charge-sign dependent modulation, and (4) adiabatic cooling (∇·**V** > 0) that substantially reduces CR energies in the inner heliosheath. The review emphasises numerical modeling (1D/2D/3D steady-state to 3D time-dependent SDE models) and integrates observations from neutron monitors (11/22-year cycles), Ulysses (latitudinal gradients), Voyager 1/2 (TS and HP crossings), IBEX (asymmetric heliosphere/bow wave), and PAMELA (high-precision proton/electron/positron/anti-proton spectra during the deep 2007–2009 minimum).

**한국어**: Potgieter (2013)는 Parker(1965)가 수송방정식을 정립한 이후 약 50년간의 태양권 우주선 변조 연구를 집대성한 *Living Reviews* 리뷰이다. 본 논문은 Parker 수송방정식(TPE)에 내재된 네 가지 기본 변조 메커니즘을 정리한다: (1) 태양풍의 외향 대류(convection), (2) 난류 태양권 자기장(HMF)을 따르는/가로지르는 확산 — 평행 성분 κ_∥과 수직 성분 κ_⊥로 구성된 3D 확산 텐서로 기술, (3) 경사(gradient)·곡률(curvature)·전류층(current-sheet) 표류 — 반대칭 표류 계수 K_A로 표현되며 약 22년 주기와 전하 부호 의존 변조를 생성, (4) 단열 냉각(∇·**V** > 0) — 내부 태양권 외각(inner heliosheath)에서 CR 에너지를 크게 감소. 본 리뷰는 1D·2D·3D 정상상태 모델에서 3D 시간 의존 SDE 모델까지 수치 모델링을 강조하고, 중성자 모니터(11/22년 주기), Ulysses(위도 경사), Voyager 1/2(TS·HP 통과), IBEX(비대칭 태양권/활 파동), PAMELA(2007–2009 극소기 고정밀 양성자/전자/양전자/반양성자 스펙트럼)의 관측을 통합적으로 다룬다.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction (§1) and the Global Heliosphere (§2) / 서론과 전지구적 태양권

**English**: Cosmic rays (CRs) for this review are particles with *E* ≳ 1 MeV/nuc; modulation is considered to occur below ∼30 GeV/nuc. The heliosphere is bounded by the termination shock (TS), the heliosheath (the shocked solar wind region), and the heliopause (HP, the contact discontinuity with the local interstellar medium). Voyager 1 crossed the TS in December 2004 at 94 AU; Voyager 2 in August 2007 at 84 AU (asymmetric by ∼10 AU). By end-2013, V1 is at ∼126 AU and V2 at ∼103 AU. Voyager 1 likely crossed the HP (or something behaving like it) around August 2012 (Webber & McDonald, 2013) — a historic milestone. IBEX ENA observations (McComas et al., 2012) establish the interface as a *bow wave* rather than a bow shock and reveal an asymmetric heliosphere (nose-to-tail ratio ∼1:2).

The Parker spiral HMF has magnitude

$$B = B_0\left(\frac{r_0}{r}\right)^2\sqrt{1+\tan^2\psi}$$

with spiral angle

$$\tan\psi = \frac{\Omega(r-r_\odot)\sin\theta}{V}$$

Typical value at Earth: ψ ≈ 45°; beyond ∼10 AU in the equatorial plane, ψ → 90°. The HMF reverses polarity every ∼11 years (22-year Hale cycle). The heliospheric current sheet (HCS), separating opposite polarity hemispheres, has a wavy structure parameterised by tilt angle α: α = 3°–10° at solar minimum, α → 75° near maximum. A>0 epochs (1970s, 1990s, 2002–2014) have HMF pointing outward in the northern hemisphere; A<0 epochs (1980s, around 1965, 2009) have inward field.

**한국어**: 본 리뷰의 우주선은 *E* ≳ 1 MeV/nuc이며, 변조는 ∼30 GeV/nuc 이하에서 발생하는 것으로 간주한다. 태양권은 종단 충격(TS), 태양권 외각(heliosheath), 헬리오포즈(HP, 국부 성간 매질과의 접촉 불연속면)로 경계 지어진다. Voyager 1은 2004년 12월 94 AU에서 TS를 통과했고, Voyager 2는 2007년 8월 84 AU에서 통과했다(∼10 AU 비대칭). 2013년 말 기준 V1 ∼126 AU, V2 ∼103 AU이며, V1은 2012년 8월경 HP를 통과한 것으로 추정된다(Webber & McDonald, 2013) — 역사적 이정표. IBEX ENA 관측(McComas 외 2012)은 경계면을 *활 충격(bow shock)*이 아닌 *활 파동(bow wave)*으로 재정립했고, 태양권의 머리/꼬리 비율 ∼1:2의 비대칭성을 드러냈다.

Parker 나선형 HMF의 크기는 위 식과 같고, 나선각 tan ψ는 각속도 Ω, 태양풍 속도 V, 극각 θ에 의존한다. 지구에서 ψ ≈ 45°, 적도면 ∼10 AU 너머에서는 ψ → 90°. HMF는 약 11년마다 극성 반전(22년 Hale 주기). 반대 극성 반구를 분리하는 태양권 전류층(HCS)은 파상 구조이며, 기울기 각도 α로 매개화: 극소기 α = 3°–10°, 극대기 α → 75°. A>0 사이클(1970s, 1990s, 2002–2014)에서는 북반구 HMF가 외향이고, A<0 사이클(1980s, ∼1965, 2009)에서는 내향이다.

---

### Part II: Cosmic Rays in the Heliosphere (§3) / 태양권 내의 우주선

**English**: The paper distinguishes **galactic cosmic rays (GCRs)** — fully ionised nuclei, anti-protons, electrons, positrons produced outside the heliosphere — from **anomalous cosmic rays (ACRs)** — interstellar neutrals that become pickup ions and are accelerated (likely at or near the TS). ACRs dominate 10–100 MeV/nuc; GCRs dominate above ∼100 MeV/nuc. A critical input to modulation modeling is the **local interstellar spectrum (LIS)**: the unmodulated CR spectrum at/beyond the HP. LIS for *E* ≲ 1 GeV remain contentious because (i) they cannot be directly observed inside the heliosphere due to modulation, and (ii) the GALPROP propagation code gives different results depending on assumptions.

The dominant CR periodicity is the **11-year cycle**, evident in NM records since the 1950s. Figure 6 of the paper shows the Hermanus NM (4.6 GV cut-off rigidity) from 1960 to 2010: CRs vary from 100% in 1987 (A<0 maximum intensity) to 77.5% at the 1991 solar maximum — a ∼23% range. Different polarity cycles show distinct shapes: A<0 cycles (1965, 1987, 2009) display *sharp peaks* due to protons drifting inward through the wavy HCS; A>0 cycles (1976, 1997) show *flatter tops*. This is the hallmark of the **22-year cycle** and a direct signature of charge-sign dependent drifts.

Shorter periodicities: 25–27-day solar rotation; daily Earth rotation; Forbush decreases from CMEs; **corotating interaction regions (CIRs)** (< 1% effect); **global merged interaction regions (GMIRs)** formed from merged CMEs at large radial distance (Burlaga et al., 1993) — these produce step-like CR decreases during solar activity increases, contributing significantly to the 11-year cycle.

**한국어**: 본 논문은 **은하 우주선(GCR)** — 태양권 바깥에서 생성된 완전 이온화 원자핵·반양성자·전자·양전자 — 와 **이상 우주선(ACR)** — 성간 중성자에서 픽업 이온이 되어 TS 근처에서 가속되는 입자 — 를 구분한다. ACR은 10–100 MeV/nuc, GCR은 ∼100 MeV/nuc 이상에서 우세. 변조 모델의 핵심 입력은 **국부 성간 스펙트럼(LIS)** — HP 바깥의 비변조 CR 스펙트럼이다. *E* ≲ 1 GeV 영역 LIS는 (i) 변조로 인해 직접 관측 불가, (ii) GALPROP 전파 코드가 가정에 따라 다른 결과 제공으로 여전히 논쟁적이다.

지배적 CR 주기는 **11년 주기**로, 1950년대 이후 NM 기록에서 명백하다. 논문 Fig. 6은 Hermanus NM(차단 강성 4.6 GV)의 1960–2010 기록: 1987년(A<0 극대 강도) 100%에서 1991년 태양 극대기 77.5%로 약 23% 변동. 극성 사이클마다 모양이 다르다: A<0 사이클(1965, 1987, 2009)은 파동형 HCS를 통과하는 양성자 표류로 **뾰족한 피크**, A>0 사이클(1976, 1997)은 **평평한 정상**. 이는 **22년 주기**의 특징이자 전하 부호 의존 표류의 직접적 증거이다.

단기 주기: 25–27일(태양 자전), 일주기(지구 자전), CME에 의한 Forbush 감소, **공회전 상호작용 영역(CIR)**(<1% 효과), 큰 반지름 거리에서 CME가 병합된 **전지구적 병합 상호작용 영역(GMIR)**(Burlaga 외 1993) — 태양 활동 증가 시 계단형 CR 감소를 유발하여 11년 주기에 크게 기여.

---

### Part III: Solar Modulation Theory (§4) / 태양 변조 이론

#### §4.1 Parker transport equation (TPE) / Parker 수송방정식

**English**: The heliospheric TPE for the pitch-angle-averaged CR distribution function *f*(**r**, *P*, *t*) is:

$$\boxed{\;\underbrace{\frac{\partial f}{\partial t}}_{\text{(a)}} = -\underbrace{(\mathbf{V} + \langle \mathbf{v}_d\rangle)\cdot\nabla f}_{\text{(b), (c)}} + \underbrace{\nabla\cdot(\mathbf{K}_s\cdot\nabla f)}_{\text{(d)}} + \underbrace{\frac{1}{3}(\nabla\cdot\mathbf{V})\frac{\partial f}{\partial \ln P}}_{\text{(e)}}\;}$$

Term (a): time evolution of *f*. Steady-state solutions set ∂*f*/∂*t* = 0. Term (b): convection by solar wind velocity **V**. Term (c): drift convection by the pitch-angle-averaged drift velocity ⟨**v**_d⟩ = ∇×(K_A **e**_B). Term (d): diffusion via the symmetric diffusion tensor **K**_s. Term (e): adiabatic energy change; for ∇·**V** > 0 (solar wind expansion), CRs lose energy; for ∇·**V** < 0 (compression, e.g., at the TS), they gain energy; for ∇·**V** = 0 (constant-speed flow beyond the TS), no adiabatic change. When expanded in heliocentric spherical coordinates (*r*, θ, φ), the TPE gives Eq. (6) in the paper — a second-order linear PDE in 3 spatial + 1 momentum + 1 time dimension.

Useful relations:
- Rigidity: $P = pc/q$; for a proton, $P$ (GV) = $(1/Z)\sqrt{E(E+2E_0)}$ with $E_0 = 938$ MeV.
- Differential intensity: $j(T) = p^2 f(\mathbf{r}, p, t) = v\,U_p/(4\pi)$.
- $U_p = 4\pi p^2 f$ is the differential particle density.

**한국어**: 피치각 평균된 우주선 분포 함수 *f*(**r**, *P*, *t*)에 대한 태양권 TPE는 위 박스와 같다. (a) 시간 변화; 정상상태에서는 ∂*f*/∂*t* = 0. (b) 태양풍 속도 **V**에 의한 대류. (c) 피치각 평균 표류 속도 ⟨**v**_d⟩ = ∇×(K_A **e**_B)에 의한 표류 대류. (d) 대칭 확산 텐서 **K**_s를 통한 확산. (e) 단열 에너지 변화; ∇·**V** > 0(태양풍 팽창)일 때 에너지 손실, ∇·**V** < 0(압축, 예: TS에서) 에너지 이득, ∇·**V** = 0(TS 너머 등속 흐름) 변화 없음. 태양 중심 구면 좌표계에서 전개하면 논문 식 (6)이 되며, 이는 3개 공간 + 1개 운동량 + 1개 시간 차원의 2차 선형 PDE이다.

유용한 관계식:
- 강성: $P = pc/q$, 양성자에서 $P$ (GV) = $(1/Z)\sqrt{E(E+2E_0)}$, $E_0 = 938$ MeV.
- 차등 강도: $j(T) = p^2 f = v\,U_p/(4\pi)$.
- 차등 입자 밀도: $U_p = 4\pi p^2 f$.

#### §4.1.5 Force-field approximation / 힘장 근사

**English**: Gleeson & Axford (1967, 1968) derived an analytical steady-state solution to a reduced TPE (neglecting drifts and assuming a specific relation between the convective-diffusive streaming):

$$\boxed{\;J(T, r) = J_{\rm LIS}(T+\Phi)\cdot \frac{T(T+2m_p c^2)}{(T+\Phi)(T+\Phi+2m_p c^2)}\;}$$

Here $\Phi$ is the **modulation potential** (in MV or GV) that characterizes the integrated modulation between the observer and the modulation boundary. Typical values:
- $\Phi \approx 300$ MV (very quiet solar minimum, e.g., 2009)
- $\Phi \approx 500$–700 MV (typical solar minimum)
- $\Phi \approx 1000$–1200 MV (typical solar maximum)
- $\Phi \approx 1500$ MV (very active solar maximum, e.g., 1990 or 2003)

The force-field approximation is widely used for quick-look analysis of NM data and for practical radiation-environment calculations, although it fails to capture drifts, charge-sign dependence, and the 22-year cycle.

**한국어**: Gleeson & Axford(1967, 1968)는 표류를 무시하고 대류·확산 스트리밍의 특정 관계를 가정하여 TPE의 정상상태 해석해를 유도했다(위 박스). 여기서 $\Phi$는 **변조 포텐셜**(MV 또는 GV)로, 관측자와 변조 경계 사이 적분된 변조 정도를 특징짓는다. 대표값: 매우 조용한 극소기(2009) ∼300 MV, 일반 극소기 ∼500–700 MV, 일반 극대기 ∼1000–1200 MV, 매우 활동적 극대기(1990, 2003) ∼1500 MV. 힘장 근사는 NM 자료의 빠른 분석과 방사선 환경 계산에 널리 쓰이나, 표류·전하 부호 의존·22년 주기를 설명하지 못한다.

#### §4.2 Diffusion coefficients / 확산 계수

**English**: The 3D diffusion tensor in spherical coordinates has components

$$\begin{aligned}
K_{rr} &= K_\parallel\cos^2\psi + K_{\perp r}\sin^2\psi\\
K_{\theta\theta} &= K_{\perp\theta}\\
K_{\phi\phi} &= K_{\perp r}\cos^2\psi + K_\parallel\sin^2\psi\\
K_{\phi r} &= (K_{\perp r} - K_\parallel)\cos\psi\sin\psi
\end{aligned}$$

An empirical expression for the parallel diffusion coefficient:

$$K_\parallel = (K_\parallel)_0\,\beta\,\frac{B_0}{B_m}\left(\frac{P}{P_0}\right)^a \left[\frac{(P/P_0)^c + (P_k/P_0)^c}{1+(P_k/P_0)^c}\right]^{(b-a)/c}$$

with $(K_\parallel)_0 \sim 10^{22}$ cm² s⁻¹, $P_0 = 1$ GV, $b = 1.95$, $c = 3.0$, and $a$ (slope below $P_k$) varies between 2006 and 2009. Radial perpendicular diffusion is typically $K_{\perp r} = 0.02\,K_\parallel$ (a widely used rule of thumb). Polar perpendicular diffusion is enhanced near the poles:

$$K_{\perp\theta} = 0.02\,K_\parallel\,f_{\perp\theta}, \qquad f_{\perp\theta} = A^+ \mp A^-\tanh[8(\theta_A - 90° \pm \theta_F)]$$

with $d = 3.0$ enhancement factor.

**한국어**: 구면 좌표계 3D 확산 텐서의 성분은 위와 같다. 평행 확산 계수의 경험적 표현에서 $(K_\parallel)_0 \sim 10^{22}$ cm² s⁻¹, $P_0 = 1$ GV, $b = 1.95$, $c = 3.0$, $a$는 2006–2009에 따라 변화. 지름 수직 확산은 통상 $K_{\perp r} = 0.02\,K_\parallel$(관례). 극 수직 확산은 극 쪽으로 증폭되어 계수 $d = 3.0$로 모델링.

#### §4.3–4.4 Drifts / 표류

**English**: In weak scattering, the drift coefficient is

$$(K_d)_{ws} = (K_d)_0\,\frac{\beta P}{3 B_m}$$

and the drift velocity components are

$$\begin{aligned}
\langle v_d\rangle_r &= -\frac{A}{r\sin\theta}\frac{\partial}{\partial\theta}(\sin\theta\,K_{\theta r})\\
\langle v_d\rangle_\theta &= -\frac{A}{r}\left[\frac{1}{\sin\theta}\frac{\partial}{\partial\phi}(K_{\phi\theta})+\frac{\partial}{\partial r}(r K_{r\theta})\right]\\
\langle v_d\rangle_\phi &= -\frac{A}{r}\frac{\partial}{\partial\theta}(K_{\theta\phi})
\end{aligned}$$

with $A = \pm 1$ (A>0 or A<0 polarity). At low rigidities, drifts must be reduced from the weak-scattering expression:

$$K_d = (K_d)_{ws}\,\frac{(P/P_{d0})^2}{1+(P/P_{d0})^2}$$

This reduction is required to reproduce the small latitudinal gradients observed by Ulysses (Heber & Potgieter, 2006, 2008; De Simone et al., 2011).

**Key drift-modulation results** (paper §4.4):
1. Particles of opposite charge sample different heliospheric regions during the same polarity epoch.
2. A 22-year cycle is present in galactic CR modulation (not in other standard solar proxies) — confirmed in diurnal anisotropy direction (Potgieter & Moraal, 1985) and NM differential response functions.
3. The wavy HCS plays a central role in setting up the 22-year cycle.
4. CR radial and latitudinal gradients differ between A>0 and A<0 polarity cycles.
5. Below 500 MeV, A>0 solar minimum proton spectra are always higher than A<0 spectra; A<0 proton spectra cross A>0 spectra at a few GeV.
6. Drift effects are not equal in every 11-year cycle; the recent A<0 solar minimum (2009) was different from other A<0 cycles.

**한국어**: 약산란(weak scattering) 하 표류 계수는 $(K_d)_{ws} = (K_d)_0 \beta P/(3 B_m)$이고 성분별 표류 속도는 위와 같으며, $A = \pm 1$(A>0 또는 A<0 극성). 저강성에서는 표류를 감소시켜야 하는데(Ulysses 관측의 작은 위도 경사를 재현하려면 필수), 이를 위해 강성 의존 감소 인자 $K_d = (K_d)_{ws}(P/P_{d0})^2/[1+(P/P_{d0})^2]$를 도입.

**표류 변조의 주요 결과**: (1) 부호가 다른 입자는 동일 극성기에도 다른 태양권 영역을 경험, (2) GCR 변조에 22년 주기가 존재, (3) 파상 HCS가 22년 주기의 중심 역할, (4) A>0/A<0에서 지름·위도 경사 차이, (5) 500 MeV 이하에서 A>0 극소기 양성자 스펙트럼이 A<0보다 항상 높고 몇 GeV에서 교차, (6) 2009 A<0 극소기는 과거 A<0 사이클과 달랐다.

#### §4.6 Numerical modulation models / 수치 변조 모델

**English**: Key milestones:
- **Fisk (1971)**: first 1D (radial-only) steady-state solution of TPE.
- **Fisk (1976)**: 2D axisymmetric model without drifts.
- **Jokipii & Kopriva (1979); Moraal et al. (1979)**: first 2D drift models with flat HCS.
- **Potgieter & Moraal (1985); Burger & Potgieter (1989)**: first 2D drift models emulating wavy HCS.
- **Jokipii & Thomas (1981); Kóta & Jokipii (1983)**: 3D steady-state models with 3D wavy HCS.
- **Perko & Fisk (1983)**: first 1D time-dependent model.
- **Le Roux & Potgieter (1990)**: 2D time-dependent, including drifts and GMIRs.
- **Jokipii (1986); Potgieter & Moraal (1988)**: TS shock-acceleration in modulation models.
- **Haasbroek & Potgieter (1998)**: non-spherical heliospheric boundary.

**Stochastic differential equations (SDEs)**: Since the early 2000s, SDE-based models have become popular because they offer (i) unconditional numerical stability, (ii) independence from spatial grid size, (iii) natural parallelism, and (iv) the ability to extract additional physical insights (propagation times, energy loss distributions, pseudo-particle traces). Examples: Strauss et al. (2011a, 2012c), Kopp et al. (2012), Bobik et al. (2012).

**Specific SDE results**: Propagation times for 100 MeV galactic electrons from Earth to HP (at 140 AU): ∼240 days for A>0 cycle, ∼110 days for A<0 cycle, ∼400 days for no-drift case. The A<0 case is fastest because electrons escape easily through the poles (opposite sign to protons).

**한국어**: 주요 이정표 — Fisk 1971(1D 첫 해), Fisk 1976(2D 축대칭, 표류 없음), Jokipii & Kopriva 1979(2D 평면 HCS 표류), Potgieter & Moraal 1985(2D 파상 HCS 표류), Jokipii & Thomas 1981(3D 정상상태 파상 HCS), Perko & Fisk 1983(1D 시간 의존), Le Roux & Potgieter 1990(GMIR 포함 2D 시간 의존), Jokipii 1986 및 Potgieter & Moraal 1988(TS 충격 가속 포함), Haasbroek & Potgieter 1998(비구형 경계).

**SDE 방법** — 2000년대 초 이후 주류화. 장점: 무조건적 수치 안정성, 공간 격자 크기 독립, 자연스러운 병렬화, 물리적 통찰 추출(전파 시간, 에너지 손실, 의사 입자 궤적). 100 MeV 전자의 지구→HP(140 AU) 전파 시간: A>0 ∼240일, A<0 ∼110일, 표류 없음 ∼400일. A<0에서 가장 빠른 것은 전자가 극을 통해 쉽게 빠져나가기 때문(양성자와 부호 반대).

#### §4.7 Charge-sign dependent modulation / 전하 부호 의존 변조

**English**: Charge-sign dependent modulation is the smoking-gun signature of drifts. During A>0 cycles, positively charged CRs drift inward mainly from the heliospheric poles, while electrons enter mainly through the equatorial regions/HCS; during A<0 cycles, the reverse. This leads to:
- During A>0 cycles, the electron-to-proton ratio e⁻/p at Earth shows an **inverted V** shape around solar minimum.
- During A<0 cycles, the e⁻/p ratio shows an **upright V** shape around solar minimum.
- Below 500 MeV, proton spectra in A>0 minima exceed those in A<0 minima (consistent with the sharper CR peak in A<0 NM records).

**Observational tests**: PAMELA (2006–): first simultaneous high-precision proton, electron, positron, anti-proton measurements for modulation studies. PAMELA observed that proton intensities increased by a factor of ∼2.5 over 4.5 years (2006–2009) while electrons of the same rigidity increased by only ∼1.4 — direct confirmation of charge-sign dependent drifts. De Simone (2011): e⁻/e⁺ = 6 at 200 MeV in 2009 rose to e⁻/e⁺ = 15 at 8 GeV.

**한국어**: 전하 부호 의존 변조는 표류의 결정적 증거이다. A>0 사이클에서는 양의 전하 CR이 주로 극에서 내향 표류하고 전자는 적도/HCS를 통해 진입; A<0 사이클에서는 반대. 결과:
- A>0 사이클 극소기 e⁻/p 비율: **역 V자** 형태
- A<0 사이클 극소기 e⁻/p 비율: **정 V자** 형태
- 500 MeV 이하: A>0 극소기 양성자 스펙트럼 > A<0(A<0 NM 기록의 날카로운 피크와 일치)

**관측 검증**: PAMELA(2006–)는 양성자·전자·양전자·반양성자 최초 동시 고정밀 관측. 2006–2009(4.5년) 동안 양성자 강도는 ∼2.5배 증가, 동일 강성 전자는 ∼1.4배만 증가 — 전하 부호 의존 표류의 직접 확증. De Simone(2011): 2009년에 e⁻/e⁺ 비율이 200 MeV에서 6, 8 GeV에서 15.

#### §4.8 Main causes of the 11/22-year solar modulation cycles / 11/22년 태양 변조 주기의 주 원인

**English**: Modern understanding combines three effects:

1. **Drifts** dominate during solar minimum (up to 4 years of the 11-year cycle): they set up the 22-year cycle, charge-sign dependence, and small latitudinal gradients.

2. **Diffusion barrier variations** — diffusion coefficients change with the solar cycle. In the **compound modeling approach** (Potgieter & Ferreira, 2001; Ferreira & Potgieter, 2004), all diffusion coefficients change with time as $K \propto B(t)^{-n(P,t)}$ where $B(t)$ is the observed HMF magnitude near Earth (varies by ∼factor 2 over a cycle) and $n(P,t)$ is a rigidity- and tilt-angle-dependent function.

3. **Propagating barriers / GMIRs**: Large outward-propagating interaction regions (Burlaga et al., 1993) form ∼10–20 AU and beyond; they cause step-like CR decreases. A series of GMIRs can produce the full 11-year amplitude (Le Roux & Potgieter, 1995).

Ndiitwani et al. (2005): percentage of drifts needed in the compound model varies from ∼90% at solar minimum to ∼10–20% at solar maximum — during extreme maximum the heliosphere becomes **diffusion (non-drift) dominated**.

Percentage of total modulation occurring in the heliosheath (120 AU → 1 AU): at *E* ≲ 0.02 GeV, > 80% of the modulation occurs in the heliosheath for both polarity cycles. Negative percentages at high energies (A<0, α=10°) mean GCRs are actually *reaccelerated* at the TS.

**한국어**: 현대적 이해는 세 가지 효과의 결합이다. (1) **표류**: 극소기(11년 중 최대 4년)에 지배적; 22년 주기·전하 부호 의존·작은 위도 경사 설정. (2) **확산 장벽 변화**: 확산 계수가 태양 주기에 따라 변화. **Compound modeling** 접근(Potgieter & Ferreira 2001; Ferreira & Potgieter 2004)에서 $K \propto B(t)^{-n(P,t)}$, $B(t)$는 지구 근처 HMF 크기(주기 동안 약 2배 변동). (3) **전파 장벽/GMIR**: Burlaga 외(1993)의 대형 외향 전파 상호작용 영역; ∼10–20 AU 이상에서 형성; 계단형 감소 유발; GMIR 연쇄로 전체 11년 진폭 생성 가능(Le Roux & Potgieter 1995).

Ndiitwani 외(2005): compound 모델에서 필요한 표류 비율은 극소기 ∼90%에서 극대기 ∼10–20%로 변동 — 극단 극대기에는 태양권이 **확산 지배(비표류)** 상태가 된다.

헬리오시스(120 AU→1 AU)에서 발생하는 전체 변조의 비율: *E* ≲ 0.02 GeV에서 두 극성 모두 >80%. 고에너지 음수값(A<0, α=10°)은 GCR이 TS에서 실제로 *재가속*됨을 의미.

---

### Part IV: Observational Highlights (§5) / 관측 하이라이트

**English**:

**§5.1 The unusual solar minimum of 2007–2009**: Deepest and longest solar minimum of the space age. The HMF at Earth reached the lowest value since 1963. HCS tilt angle eventually reached minimum at end-2009. CRs with high rigidity reached **record-setting intensities**. PAMELA observed the highest proton spectrum ever, in December 2009. This was unexpected because during previous A<0 cycles, proton spectra were always *lower* than for A>0 cycles at *E* < few GeV. Adriani et al. (2013): absolute proton flux measurements down to 400 MV from July 2006 to end-2009.

**§5.2 Inner heliosphere (Ulysses)**:
- North-South asymmetry in GCR flux (McKibben et al., 1996).
- Small latitudinal gradients at solar minimum — LIS cannot be observed in inner polar regions.
- Latitudinal gradients for protons peak at ∼2 GV (maximum ∼0.4 %/degree for A>0, –0.2 %/degree for A<0).
- No latitudinal gradients at solar maximum (drifts absent).
- Recurrent particle events at high heliolatitudes without corresponding solar wind / HMF features.

**§5.3 Outer heliosphere (Voyagers)**:
- ACR intensity *did not* peak at TS crossing (2004.96) — contrary to standard diffusive shock acceleration prediction at TS. ACR intensity continued to rise in the heliosheath.
- TS observed to be weak, quasi-perpendicular shock; solar wind flow energy went mainly to pickup ions (Richardson & Stone, 2009).
- Voyager 1 galactic electron intensity (6–14 MeV) increased by a factor of 5 from 2004.96 to 2008.5.
- Aug 2012: sudden changes at V1 (121.7 AU) — 0.5 MeV proton intensity dropped > 90%, > 100 MeV proton count rate rose ∼30–50%. Webber & McDonald (2013) interpreted this as HP (or "heliocliff") crossing.

**한국어**:

**§5.1 2007–2009 이례적 극소기**: 우주시대 가장 깊고 긴 극소기. 지구에서 HMF 크기는 1963년 이후 최저. HCS 기울기는 2009년 말 최소 도달. 고강성 CR은 **기록적 강도** 달성. PAMELA는 2009년 12월 역대 최고 양성자 스펙트럼 관측 — 과거 A<0 사이클 양성자 스펙트럼이 A>0보다 낮았던 것과 정반대의 예상치 못한 결과. Adriani 외(2013): 2006년 7월–2009년 말 절대 양성자 플럭스 400 MV까지 측정.

**§5.2 내부 태양권(Ulysses)**: GCR 플럭스의 남북 비대칭성; 극소기 작은 위도 경사(극 쪽에서 LIS 관측 불가); 양성자 위도 경사는 ∼2 GV에서 피크(A>0 최대 ∼0.4 %/도, A<0 최대 –0.2 %/도); 극대기에 위도 경사 없음(표류 부재); 고위도 재발성 입자 이벤트.

**§5.3 외부 태양권(Voyager)**: TS 통과(2004.96)에서 ACR 강도가 피크 **아님** — 표준 확산 충격 가속 예측과 배치; 헬리오시스에서 계속 상승; TS는 약한 준수직 충격, 태양풍 에너지는 주로 픽업 이온으로 전달(Richardson & Stone 2009); V1의 6–14 MeV 은하 전자 강도는 2004.96→2008.5 동안 5배 증가; 2012년 8월 V1(121.7 AU)에서 급변 — 0.5 MeV 양성자 >90% 감소, >100 MeV 양성자 계수율 ∼30–50% 증가; Webber & McDonald(2013) HP(또는 "heliocliff") 통과로 해석.

---

### Part V: Constraints, Challenges, and Summary (§6, §7) / 제약·과제·요약

**English**: Key open problems:
1. **LIS determination**: needed at energies ≲ few GeV; adiabatic energy losses disguise LIS spectral shapes for ions/anti-protons below 10 GeV.
2. **Heliospheric geometry**: asymmetries in modulation volume; TS oscillation; role of the tail region; alignment of HMF and local interstellar magnetic field at HP.
3. **HCS**: calculation of tilt angle is model-dependent; unclear how waviness is preserved outward.
4. **HMF modeling**: Parker-type fields may be too simple near poles; Fisk-type fields are complex but observationally ambiguous.
5. **Diffusion theory**: ab initio theory cannot yet produce diffusion tensor elements consistent with observations; rigidity dependence of the two perpendicular components is poorly constrained.
6. **TS and beyond**: acceleration of ACRs is not fully understood (DSA, magnetic reconnection, stochastic acceleration).
7. **Time-dependent modeling**: what to use for time dependence of diffusion coefficients on top of wavy HCS variations.

**Summary (§7)**: Half a century after Parker's theory, the basic modulation paradigm is essentially correct. Major uncertainties remain in the *spatial, rigidity, and especially temporal dependence* of the diffusion coefficients. The field is "alive-and-well" with much progress still to be made.

**한국어**: 주요 미해결 과제: (1) **LIS 결정** — 몇 GeV 이하에서 필요; 10 GeV 이하 이온/반양성자는 단열 에너지 손실로 LIS 스펙트럼 모양이 변형. (2) **태양권 기하**: 변조 부피의 비대칭; TS 진동; 꼬리 영역의 역할; HP에서 HMF와 국부 성간 자기장의 정렬. (3) **HCS**: 기울기 계산은 모델 의존적; 외향으로 파상성 보존 메커니즘 불분명. (4) **HMF**: Parker형은 극 쪽에서 너무 단순; Fisk형은 복잡하나 관측 해석 모호. (5) **확산 이론**: ab initio 이론은 아직 관측과 일치하는 확산 텐서 생성 불가; 두 수직 성분의 강성 의존성 제약 약함. (6) **TS와 그 너머**: ACR 가속 미완성 이해(DSA, 자기 재결합, 확률 가속). (7) **시간 의존 모델**: 파상 HCS 변동 위에 확산 계수의 시간 의존성 설정 방식.

**요약(§7)**: Parker 이론 반세기 후, 기본 변조 패러다임은 본질적으로 옳다. 확산 계수의 *공간·강성·특히 시간* 의존성에 주요 불확실성 남아있다. 연구 분야는 "활력 충만"하며 많은 진전이 아직 필요하다.

---

## 3. Key Takeaways / 핵심 시사점

1. **Four mechanisms, one equation** — **네 메커니즘, 하나의 방정식**
   - **English**: All solar modulation is captured by the Parker transport equation, which contains exactly four physical terms: convection by the solar wind, diffusion (both parallel and perpendicular to the HMF), gradient/curvature/current-sheet drifts, and adiabatic energy change. These four terms compete over the solar cycle and together produce every observed feature of CR modulation.
   - **한국어**: 모든 태양 변조는 Parker 수송방정식에 담겨있으며, 이 방정식은 정확히 네 가지 물리 항 — 태양풍 대류, HMF 평행/수직 확산, 경사/곡률/HCS 표류, 단열 에너지 변화 — 을 포함한다. 이 네 항이 태양 주기 동안 경쟁하며 관측되는 모든 CR 변조 특징을 생성한다.

2. **The 22-year cycle is drift-driven and charge-sign dependent** — **22년 주기는 표류 구동·전하 부호 의존**
   - **English**: Because the HMF reverses polarity every ∼11 years, positively and negatively charged particles drift through different heliospheric regions during the same solar cycle. This produces a 22-year modulation cycle absent from other solar proxies — uniquely visible in CRs. A>0 cycles show flat-topped CR maxima; A<0 cycles show sharp peaks at Earth.
   - **한국어**: HMF가 ∼11년마다 극성 반전하므로, 같은 태양 주기 동안 양/음 전하 입자가 서로 다른 태양권 영역을 표류한다. 이는 다른 태양 지표에 없는 22년 변조 주기를 생성 — CR에서만 관측되는 고유 현상. A>0 사이클은 평평한 CR 정상, A<0 사이클은 날카로운 피크를 보인다.

3. **The force-field approximation is practical but incomplete** — **힘장 근사는 실용적이지만 불완전**
   - **English**: The Gleeson–Axford force-field formula $J(T) = J_{\rm LIS}(T+\Phi)[T(T+2m_pc^2)/((T+\Phi)(T+\Phi+2m_pc^2))]$ with a single parameter Φ works remarkably well for quick comparisons between NM data and LIS — but cannot capture drifts, charge-sign dependence, or 22-year effects. Φ varies from ∼300 MV (deep minimum) to ∼1500 MV (active maximum).
   - **한국어**: Gleeson–Axford 힘장 공식은 단일 매개변수 Φ로 NM 자료와 LIS 간 빠른 비교에 탁월 — 그러나 표류·전하 부호·22년 효과는 담을 수 없다. Φ는 깊은 극소기 ∼300 MV에서 활동적 극대기 ∼1500 MV까지 변동.

4. **Drifts dominate at minimum, diffusion at maximum** — **극소기는 표류, 극대기는 확산이 지배**
   - **English**: Compound modeling (Ndiitwani et al., 2005) shows that drift percentages needed to match e⁻/p observations vary from ∼90% at solar minimum to only ∼10–20% at solar maximum. During extreme maximum the heliosphere is effectively diffusion (non-drift) dominated — consistent with the observed disappearance of latitudinal gradients at maximum.
   - **한국어**: Compound 모델링(Ndiitwani 외 2005)은 e⁻/p 관측과 일치하는 데 필요한 표류 비율이 극소기 ∼90%에서 극대기 ∼10–20%로 변동함을 보인다. 극단 극대기에는 태양권이 실질적으로 확산(비표류) 지배 — 극대기에 위도 경사 소실 관측과 일치.

5. **GMIRs drive the step-like 11-year variation** — **GMIR이 계단형 11년 변동 구동**
   - **English**: Global Merged Interaction Regions (Burlaga et al., 1993) — formed by CMEs merging with CIRs at 10–20 AU — are large outward-propagating barriers that cause step-like CR decreases. A series of GMIRs during the increasing phase of the solar cycle can reproduce the full 11-year modulation amplitude (Le Roux & Potgieter, 1995).
   - **한국어**: 전지구적 병합 상호작용 영역(Burlaga 외 1993)은 10–20 AU에서 CME와 CIR 병합으로 형성되는 대형 외향 전파 장벽; 계단형 CR 감소 유발. 태양 활동 증가기 GMIR 연쇄로 전체 11년 변조 진폭 재현 가능(Le Roux & Potgieter 1995).

6. **The 2007–2009 minimum was anomalous** — **2007–2009 극소기는 이상 현상**
   - **English**: The deepest and longest solar minimum of the space age produced record CR intensities — particularly striking because during A<0 cycles proton spectra were historically *lower* than A>0 cycles, yet 2009 proton spectra exceeded all prior records. This forced re-examination of the interplay between drifts, diffusion, and propagating barriers over the extended cycle 23/24 minimum.
   - **한국어**: 우주시대 가장 깊고 긴 극소기가 기록적 CR 강도 생성 — 과거 A<0 사이클 양성자 스펙트럼이 A>0보다 낮았음에도 2009년이 역대 최고치. 확장된 23/24 주기 극소기 동안 표류·확산·전파 장벽의 상호작용 재검토를 강제.

7. **Most low-energy modulation occurs in the heliosheath** — **저에너지 변조 대부분은 헬리오시스에서 발생**
   - **English**: For CRs with *E* ≲ 0.02 GeV, > 80% of the total modulation (between HP at 120 AU and 1 AU) occurs inside the heliosheath for both polarity cycles. The heliosheath adds considerably to the modulation volume, even though the flow there is slower and less adiabatic.
   - **한국어**: *E* ≲ 0.02 GeV CR의 경우, 두 극성 모두에서 전체 변조(120 AU HP에서 1 AU)의 >80%가 헬리오시스 내부에서 발생. 헬리오시스의 흐름은 느리고 단열성이 약함에도 변조 부피에 상당 기여.

8. **SDE modeling enables new physical insights** — **SDE 모델링은 새로운 물리적 통찰을 가능하게 함**
   - **English**: Stochastic Differential Equation-based models (Strauss et al. 2011a, 2012c; Kopp et al. 2012) solve the TPE by integrating pseudo-particle trajectories backward in time, yielding propagation time distributions, energy loss histograms, and drift path visualisations — information unavailable from traditional finite-difference Crank–Nicholson/ADI schemes. Example: 100 MeV electrons from Earth to 140 AU HP take ∼240 days (A>0), ∼110 days (A<0), ∼400 days (no drift).
   - **한국어**: 확률 미분 방정식 기반 모델(Strauss 외 2011a, 2012c; Kopp 외 2012)은 의사 입자 궤적을 시간 역방향으로 적분하여 TPE를 풀어 전파 시간 분포, 에너지 손실 히스토그램, 표류 경로 시각화 등 — 기존 유한 차분 Crank–Nicholson/ADI 기법으로 얻을 수 없는 정보 — 를 제공. 예: 100 MeV 전자의 지구→140 AU HP 전파 시간은 A>0 ∼240일, A<0 ∼110일, 표류 없음 ∼400일.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 The Parker Transport Equation (TPE) / Parker 수송방정식

**English**: The fundamental equation of solar modulation:

$$\frac{\partial f}{\partial t} = -(\mathbf{V}+\langle\mathbf{v}_d\rangle)\cdot\nabla f + \nabla\cdot(\mathbf{K}_s\cdot\nabla f) + \frac{1}{3}(\nabla\cdot\mathbf{V})\frac{\partial f}{\partial \ln P}$$

Variables:
- $f(\mathbf{r}, P, t)$: pitch-angle-averaged CR distribution function
- $\mathbf{V}$: solar wind velocity (radial, $V\sim 400$ km/s at solar min equatorial plane)
- $\langle\mathbf{v}_d\rangle$: pitch-angle-averaged drift velocity = $\nabla\times(K_A\mathbf{e}_B)$
- $\mathbf{K}_s$: symmetric diffusion tensor (3×3)
- $P$: particle rigidity (GV)

### 4.2 Force-field approximation / 힘장 근사

$$J(T, r) = J_{\rm LIS}(T+\Phi)\cdot\frac{T(T+2m_pc^2)}{(T+\Phi)(T+\Phi+2m_pc^2)}$$

where $\Phi$ (units MV or GV) is the modulation potential; single-parameter description. Typical value ranges:
| Solar condition | $\Phi$ (MV) |
|---|---|
| Very deep minimum (2009) | ∼300 |
| Typical minimum | 500–700 |
| Typical maximum | 1000–1200 |
| Active maximum (1990, 2003) | 1400–1500 |

### 4.3 Diffusion tensor in Parker spiral / Parker 나선에서의 확산 텐서

$$\mathbf{K}_s = \begin{pmatrix}
K_\parallel\cos^2\psi+K_{\perp r}\sin^2\psi & 0 & (K_{\perp r}-K_\parallel)\cos\psi\sin\psi\\
0 & K_{\perp\theta} & 0\\
(K_{\perp r}-K_\parallel)\cos\psi\sin\psi & 0 & K_{\perp r}\cos^2\psi+K_\parallel\sin^2\psi
\end{pmatrix}$$

Typical relations: $K_{\perp r} = 0.02\,K_\parallel$; $K_{\perp\theta} = 0.02\,K_\parallel f_{\perp\theta}$ (enhanced by factor 3 toward poles).

### 4.4 Empirical parallel diffusion coefficient / 경험적 평행 확산 계수

$$K_\parallel = (K_\parallel)_0\,\beta\,\frac{B_0}{B_m}\left(\frac{P}{P_0}\right)^a \left[\frac{(P/P_0)^c+(P_k/P_0)^c}{1+(P_k/P_0)^c}\right]^{(b-a)/c}$$

with $(K_\parallel)_0 \sim 10^{22}$ cm² s⁻¹; $P_0=1$ GV; $b=1.95$; $c=3.0$; $a$ ∈ [varies 2006–2009].

### 4.5 Drift coefficient (weak scattering) / 표류 계수

$$(K_d)_{ws} = (K_d)_0\,\frac{\beta P}{3 B_m}$$

Reduced drift at low rigidity:

$$K_d = (K_d)_{ws}\,\frac{(P/P_{d0})^2}{1+(P/P_{d0})^2}$$

### 4.6 Parker spiral geometry / Parker 나선 기하

$$\mathbf{B} = B_0\left(\frac{r_0}{r}\right)^2(\mathbf{e}_r - \tan\psi\,\mathbf{e}_\phi)$$

$$\tan\psi = \frac{\Omega(r-r_\odot)\sin\theta}{V}$$

$$B = B_0\left(\frac{r_0}{r}\right)^2\sqrt{1+\tan^2\psi}$$

### 4.7 Rigidity-energy relations / 강성·에너지 관계

$$P = \frac{pc}{q} = \frac{A}{Ze}\sqrt{E(E+2E_0)}\quad\text{(GV)}$$

$$\beta = \frac{v}{c} = \frac{\sqrt{E(E+2E_0)}}{E+E_0}$$

$$j(T) = p^2 f(\mathbf{r}, p, t)$$

### 4.8 Heliospheric current sheet / 태양권 전류층

Tilt angle $\alpha$ ranges: 3°–10° (solar min) to ∼75° (solar max). Polar enhancement function:

$$f_{\perp\theta} = A^+ \mp A^-\tanh[8(\theta_A - 90° \pm \theta_F)]$$

with $A^\pm = (d\pm 1)/2$, $d = 3.0$, $\theta_F = 35°$.

**한국어**: 위 수식들은 모두 태양권 CR 수송의 표준 모델을 구성한다. TPE는 Fokker–Planck형 방정식이며 네 가지 물리 항(시간 변화·대류/표류·확산·단열 변화)을 담고 있다. 힘장 근사는 단일 변조 포텐셜 Φ로 스펙트럼 변형을 매개화한다. 확산 텐서는 Parker 나선 기하에서 4개의 독립 성분을 가지며, 표류 계수는 반대칭 성분으로 $\nabla\times(K_A\mathbf{e}_B)$를 통해 표류 속도를 생성한다.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1912 ── Hess: discovery of cosmic rays (balloon flights)
  │
1936 ── Forbush: CR decreases during magnetic storms
  │
1948 ── Parker & Tidman: earliest CR modulation ideas
  │
1957-58 ── International Geophysical Year: NMs deployed globally
  │
1958 ── Parker: solar wind theory, Archimedean HMF spiral predicted
  │
1965 ──┼── Parker: cosmic-ray transport equation (TPE) [foundational]
  │
1967 ── Gleeson & Axford: force-field approximation (analytic solution)
  │
1971 ── Fisk: first 1D numerical solution of TPE
  │
1977 ── Jokipii, Levy & Hubbard: gradient/curvature drifts → 22-year cycle
  │
1983 ── Perko & Fisk: first 1D time-dependent model
  │
1985 ── Potgieter & Moraal: first 2D drift model with wavy HCS
  │
1990 ── Ulysses launch: first polar orbit around Sun
  │
1992-93 ── Burlaga et al.: GMIRs as propagating CR barriers
  │
1996 ── Fisk: new HMF geometry from differential rotation
  │
2001 ── Potgieter & Ferreira: compound modeling approach
  │
2004 ── Voyager 1 crosses termination shock (94 AU)
  │
2007-09 ── Deepest solar minimum of space age
2009 ── Adriani et al. (PAMELA): first high-precision e⁻, e⁺, p, p̄
  │
2011-12 ── Strauss et al.: 3D SDE-based modulation models
  │
2012 ── Voyager 1 apparent heliopause crossing (~122 AU)
  │
2013 ──┼── POTGIETER REVIEW (this paper, Living Reviews)
  │
2015+ ── AMS-02 high-energy CR antimatter; Voyager LIS
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Parker (1958) "Solar Wind" | Predicted the Archimedean spiral HMF that forms the backbone of every CR modulation model / 모든 CR 변조 모델의 근간이 되는 Parker 나선 HMF를 예측 | Foundational — HMF geometry used throughout §2.2 and §4 / 토대 — §2.2와 §4 전반에 사용 |
| Parker (1965) "Passage of energetic charged particles through interplanetary space" | Derived the TPE (Eq. 5 of this review) / 본 리뷰 식 (5)인 TPE를 유도 | Direct: the paper reviewed here is essentially a 50-year celebration of Parker's TPE / 직접: 본 리뷰는 본질적으로 Parker TPE의 50주년 기념 |
| Gleeson & Axford (1967, 1968) "Cosmic Rays in the Interplanetary Medium" | Derived the force-field approximation used as analytic baseline / 해석적 기준인 힘장 근사를 유도 | Direct: Eq. (used in implementation notebook) / 직접: 구현 노트북에 사용된 수식 |
| Jokipii, Levy & Hubbard (1977) "Effects of particle drift on cosmic-ray transport" | Established gradient, curvature and current-sheet drifts as a dominant modulation mechanism / 경사·곡률·전류층 표류를 주된 변조 메커니즘으로 확립 | Direct: §4.3–4.4 builds entirely on this foundation / 직접: §4.3–4.4는 이 토대에 의존 |
| Fisk (1996) "Motion of the footpoints of heliospheric magnetic field lines at the Sun" | Proposed non-Parker HMF geometry from differential rotation | Modern alternative to Parker field, discussed in §2.2 and §6 / Parker 장의 현대적 대안, §2.2·§6에서 논의 |
| Burlaga et al. (1993) "A Magnetic Rope in the Solar Wind" | Introduced Global Merged Interaction Regions (GMIRs) as propagating CR barriers | Direct: §4.8 compound modeling is built on GMIR physics / 직접: §4.8 compound 모델은 GMIR 물리에 기반 |
| Heber & Potgieter (2006, 2008) "Cosmic rays at high heliolatitudes" | Ulysses observations constraining drift and latitudinal gradients | Observational anchor for §5.2 and drift-reduction formulae / §5.2와 표류 감소 공식의 관측 앵커 |
| Adriani et al. (2013) PAMELA | High-precision proton spectra during 2006–2009 minimum | Direct: key observational test case in §4.7 and §5.1 / 직접: §4.7과 §5.1의 핵심 관측 검증 사례 |
| Strauss et al. (2011a,c; 2012c) SDE models | Pseudo-particle propagation times, energy losses, drift paths / 의사 입자 전파 시간·에너지 손실·표류 경로 | Modern numerical technique featured throughout §4.6 / §4.6 전반에 걸친 현대 수치 기법 |
| McComas et al. (2012) IBEX | ENA maps revealing bow-wave (not bow-shock) and asymmetric heliosphere | Changed the global heliospheric picture in §2.1 / §2.1의 전지구적 태양권 그림을 변경 |

---

## 7. References / 참고문헌

- **Potgieter, M. S.**, "Solar Modulation of Cosmic Rays", *Living Rev. Solar Phys.*, **10**, (2013), 3. [DOI: 10.12942/lrsp-2013-3]
- Parker, E. N., "Dynamics of the interplanetary gas and magnetic fields", *ApJ*, **128**, 664 (1958).
- Parker, E. N., "The passage of energetic charged particles through interplanetary space", *Planet. Space Sci.*, **13**, 9 (1965).
- Gleeson, L. J. & Axford, W. I., "Cosmic rays in the interplanetary medium", *ApJ Lett.*, **149**, L115 (1967).
- Gleeson, L. J. & Axford, W. I., "Solar modulation of galactic cosmic rays", *ApJ*, **154**, 1011 (1968).
- Jokipii, J. R., Levy, E. H. & Hubbard, W. B., "Effects of particle drift on cosmic-ray transport", *ApJ*, **213**, 861 (1977).
- Fisk, L. A., "Solar modulation of galactic cosmic rays, 2", *JGR*, **76**, 221 (1971).
- Potgieter, M. S. & Moraal, H., "A drift model for the modulation of galactic cosmic rays", *ApJ*, **294**, 425 (1985).
- Burlaga, L. F. et al., "Large-scale structure of the solar wind for 1978 and 1986", *JGR*, **98**, 21003 (1993).
- Le Roux, J. A. & Potgieter, M. S., "The simulation of complete 11 and 22 year modulation cycles...", *ApJ*, **442**, 847 (1995).
- Heber, B. & Potgieter, M. S., "Cosmic rays at high heliolatitudes", *Space Sci. Rev.*, **127**, 117 (2006, 2008).
- Ferreira, S. E. S. & Potgieter, M. S., "Long-term cosmic-ray modulation in the heliosphere", *ApJ*, **603**, 744 (2004).
- Strauss, R. D. et al., "SDE-based modulation modeling" series (2011a,b,c; 2012a,b,c; 2013a,b).
- Adriani, O. et al. (PAMELA), "Time dependence of the proton flux measured by PAMELA during the 2006 July–2009 December solar minimum", *ApJ*, **765**, 91 (2013).
- McComas, D. J. et al., "The heliosphere's interstellar interaction: no bow shock", *Science*, **336**, 1291 (2012).
- Webber, W. R. & McDonald, F. B., "Recent Voyager 1 data indicate that on 25 August 2012...", *GRL*, **40**, 1665 (2013).
- Ndiitwani, D. C. et al., "Modeling cosmic ray intensities along the Ulysses trajectory", *Ann. Geophys.*, **23**, 1061 (2005).
- Fisk, L. A., "Motion of the footpoints of heliospheric magnetic field lines at the Sun", *JGR*, **101**, 15547 (1996).
- Kóta, J. & Jokipii, J. R., "Effects of drift on the transport of cosmic rays. IV", *ApJ*, **265**, 573 (1983).

(Full reference list: see pages 51–66 of the original review.)
