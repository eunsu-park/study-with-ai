---
title: "Interaction Between Convection and Pulsation"
authors: ["Günter Houdek", "Marc-Antoine Dupret"]
year: 2015
journal: "Living Reviews in Solar Physics"
doi: "10.1007/lrsp-2015-8"
topic: Living_Reviews_in_Solar_Physics
tags: [time-dependent-convection, pulsation, asteroseismology, mixing-length-theory, Reynolds-stress, p-modes, instability-strip, kappa-mechanism, surface-effect]
status: completed
date_started: 2026-04-23
date_completed: 2026-04-23
---

# 47. Interaction Between Convection and Pulsation / 대류와 맥동의 상호작용

---

## 1. Core Contribution / 핵심 기여

**한국어**: 이 리뷰 논문은 별의 맥동(pulsation)과 난류 대류(turbulent convection)의 상호작용을 기술하는 1차원 시간 의존 대류(Time-Dependent Convection, TDC) 모델에 대한 현대적 이해를 종합한다. 저자들은 헬리오세이즘과 별지진학(asteroseismology)의 관점에서, Reynolds 분리 접근을 출발점으로 하여 평균 유체방정식과 섭동(fluctuation) 방정식을 유도한 뒤, 두 가지 주류 반해석적 TDC 형식론 — **Gough (1965, 1977a,b)**의 혼합거리(mixing-length) 기반 국소/비국소 모델과 **Unno (1967, 1977)**의 모델을 일반화한 **Grigahcène et al. (2005)**의 비방사 맥동용 모델 — 을 체계적으로 소개하고 비교한다. 이어서 대류가 맥동 진동수에 미치는 "surface effect"(태양에서 ~13 μHz), 선폭 Γ = 2η의 예측, 고전 불안정대의 적색 경계(red edge) 재현, δ Scuti/γ Doradus/Cepheid/RR Lyrae/Mira/roAp 별과 태양형/적색거성 별의 모드 안정성과 감쇠 계산 결과를 체계적으로 제시한다. 핵심 결론은 어떤 단일 TDC 모델도 모든 맥동 현상을 통일되게 설명하지 못하며, 3차원 유체역학 시뮬레이션과의 결합이 미래의 길이라는 것이다.

**English**: This review synthesizes the contemporary understanding of one-dimensional time-dependent convection (TDC) models describing the interaction between stellar pulsation and turbulent convection. Starting from the Reynolds-averaged hydrodynamical equations in the Boussinesq approximation, the authors systematically derive the mean and fluctuation equations and present two mainstream semi-analytical TDC formalisms: the mixing-length-based local and nonlocal model of **Gough (1965, 1977a,b)** and the generalization to nonradial pulsations by **Grigahcène et al. (2005)** built upon **Unno (1967, 1977)**. Applications include the convective "surface effect" on oscillation frequencies (~13 μHz in the Sun), predictions of linewidths Γ = 2η, the reproduction of the red edge of the classical instability strip, and mode-stability/damping computations for δ Scuti, γ Doradus, Cepheid, RR Lyrae, Mira, roAp stars, together with solar-type and red-giant oscillators. The central conclusion is that no single TDC model explains all observed pulsation phenomena, and the path forward lies in coupling 1D TDC models to 3D hydrodynamical simulations.

---

## 2. Reading Notes / 읽기 노트

### Part I: Hydrodynamical Foundations / 유체역학적 기초 (§2, pp.7–11)

**한국어**: 논문은 회전과 자기장을 무시하고 구형 대칭에서 질량·운동량·열에너지 보존 방정식(1)-(3)에서 출발한다. 각 변수 y를 평균 ȳ와 섭동 y'로 분리하는 Reynolds 접근(Eq.4)이 핵심이다. 속도장은 v = U + u로 분리되며, 층상(horizontal) 평균으로 ρū = 0 (Eq.6)을 요구한다. 이는 Unno의 교번 표기 U=⟨v⟩ ≠ 0, ū ≠ 0 조건 대신 쓰인다. Reynolds 응력 텐서는 등방 부분(turbulent pressure p_t = ρ⟨u_r²⟩)과 비등방 부분 σ_t로 분해된다(Eq.10). 이방성 매개변수 Φ := ⟨|u|²⟩/u_r² 를 정의해(Eq.11) 비등방 난류를 매개한다; Φ=3은 등방을 의미한다. 난류 운동에너지 방정식(Eq.13)은 (i) 평균 흐름에서의 난류 에너지 생성, (ii) 난류 에너지 플럭스의 발산, (iii) 점성 소산 ρε_t 항으로 구성된다. Boussinesq 근사(Eqs.15-17)에서는 밀도 섭동이 부력항에서만 유지되고, 압력 섭동은 무시되며, 평균 열에너지 방정식은 대류 열 플럭스 F_c = ρ c_p ⟨wT'⟩ (Eq.19)을 포함하게 된다. 최종적으로 방사상으로 맥동하는 별의 평균 운동량(Eq.22), 열에너지(Eq.23), 복사 이동(Eqs.24-25) 방정식들이 Lagrangian-Eulerian 혼합 좌표 (q_1,q_2,q_3) = (rdθ, r sinθ dφ, ρ̄dr)로 기술된다.

**English**: The paper starts from mass, momentum, and thermal energy conservation (Eqs.1-3) in spherical geometry, neglecting rotation and magnetic fields. Each variable y is split into a mean ȳ and fluctuation y' (Eq.4, Reynolds separation). The velocity field is decomposed v = U + u, with horizontal average ρū = 0 (Eq.6). The Reynolds stress tensor is split into an isotropic turbulent pressure p_t = ρ⟨u_r²⟩ and an anisotropic part σ_t (Eq.10). The anisotropy parameter Φ := ⟨|u|²⟩/u_r² (Eq.11) measures deviation from isotropic turbulence (Φ=3). The turbulent kinetic energy equation (Eq.13) has three physically meaningful terms: (i) production from mean flow, (ii) divergence of turbulent energy flux, and (iii) viscous dissipation ρε_t. In the Boussinesq limit (Eqs.15-17) density perturbations are retained only in the buoyancy term, acoustic energy flux is dropped, and the mean thermal equation contains the convective heat flux F_c = ρ c_p ⟨wT'⟩ (Eq.19). The final mean equations for a radially pulsating envelope — mean momentum (Eq.22), thermal energy (Eq.23), and grey Eddington radiative transfer (Eqs.24-25) — are written in the mixed Lagrangian-Eulerian coordinate system q_i.

Key equation: the mean radial momentum balance retaining turbulent pressure is
$$\frac{\partial}{\partial q}(\bar{p} + p_t) + (3 - \bar{\Phi})\frac{p_t}{r\bar{\rho}} = -\frac{Gm}{r^2} - \frac{\partial^2 r}{\partial t^2}. \tag{22}$$

### Part II: Time-Dependent Mixing-Length Models / 시간 의존 혼합거리 모델 (§3, pp.12–32)

**한국어**: Boussinesq 섭동 방정식(Eqs.32-38)은 발산 없는 속도장, 운동량 방정식에서 부력항 gδT'/T̄을 통해서만 밀도 섭동을 유지, 초단열 기울기 β := -(∂T̄/∂q - δ̄/c_p · ∂p̄/∂q) (Eq.38)을 구동항으로 가진다. **§3.2.2**에서 정적 분위기(static atmosphere)에 대한 국소 혼합거리 모델이 제시되며, 비선형 유체 이류 항은 혼합거리 ℓ 을 이용한 단순화로 폐쇄된다. **§3.2.3**에서는 맥동하는 분위기로 확장되어, 맥동 주기와 대류 시간 규모의 비를 나타내는 무차원 수 σ̃ = ω/σ_c가 들어간다. 수학적 핵심은 **에디 생존 확률**(eddy survival probability) P(r,t,t_0)이다:
$$\mathcal{P}(r,t,t_0) = \exp\left[-\int_{t_0}^t \frac{W(t';t_0)\,dt'}{\ell}\right]. \tag{144}$$
난류 플럭스는 P와 선형 성장률 σ_c, 에디 생성률 nm을 사용한 적분으로 표현된다: F_c = nm c_p ∫WΘP dt_0 (Eq.147), p_t = nm ∫W²P dt_0 (Eq.148). 통계적 정상 상태에서 nm ∫P dt_0 = ρ (Eq.149)이며, 교정인자 χ = 1/2가 플럭스 정규화에서 얻어진다.

**§3.3 (Gough 비국소 모델)**: 국소 이론은 대류 경계에서 운동에너지가 급격히 0이 되는 비물리적 거동을 주기에, Gough(1977b)는 두 개의 추가 변수 — 대류 운동에너지 플럭스 I와 운동에너지 밀도 K 에 대한 추가 방정식을 도입해 **비국소**(nonlocal) 모델을 만들었다. 핵심 매개변수는 α = ℓ/H_p (혼합거리 매개변수), a (F_c의 비국소 정도), b (난류 운동량 플럭스의 비국소 정도).

**§3.4 (Unno/Grigahcène 비방사)**: Gabriel et al.(1975) 와 Grigahcène et al.(2005)은 Unno 모델을 **비방사**(nonradial) 맥동으로 일반화했다. 여기서 대류 섭동 δ(보통 F_c, p_t의 섭동)의 **복소 축약 인자** β̂ (Eq.107)가 등장한다. β̂는 시간 의존 대류가 자기일관적으로 결합된 비단열 진폭비를 조정하며, 태양 데이터로 교정된다. **§3.5-3.6**에서는 Gough와 Grigahcène/Unno 모델의 근본적 차이를 표와 논의로 정리한다. 예를 들어 Gough는 성장률 σ_c가 에디 생존 확률에 내재된 반면, Unno는 지수 완화 근사 dF_c/dt = (F_c0 - F_c)/τ_c 로 출발한다.

**English**: The Boussinesq fluctuation equations (Eqs.32-38) assume divergence-free velocity, retain density fluctuations only in the buoyancy term, and identify the superadiabatic gradient β (Eq.38) as the driving term for convection. **§3.2.2** derives the local mixing-length model for a static atmosphere, closing the nonlinear advection by lumping it into a linear damping over the mixing length. **§3.2.3** extends to pulsating atmospheres, introducing the dimensionless ratio σ̃ = ω/σ_c of pulsation to convective-growth frequency. The mathematical centerpiece is the **eddy survival probability** P(r,t,t_0) (Eq.144): the probability that an eddy created at t_0 survives until t. The convective fluxes are then integrals over creation time: F_c = nm c_p ∫WΘP dt_0 (Eq.147), p_t = nm∫W²P dt_0 (Eq.148). Statistical steady state fixes nm∫P dt_0 = ρ, and a calibration χ = 1/2 emerges from matching the steady flux.

**§3.3 (nonlocal Gough model)**: Because the local theory produces unphysical discontinuities at convective boundaries, Gough (1977b) introduced nonlocal integrals by adding equations for the kinetic-energy flux I and kinetic-energy density K, governed by two dimensionless parameters a (flux smoothing), b (Reynolds stress smoothing), and α = ℓ/H_p.

**§3.4 (Unno/Grigahcène nonradial)**: Gabriel et al. (1975) and Grigahcène et al. (2005) extended the Unno formalism to **nonradial** pulsations, introducing a complex closure parameter β̂ (Eq.107) that governs the self-consistent coupling of nonadiabatic pulsation with time-varying convection; β̂ is usually calibrated against solar data. **§3.5-3.6** contrasts the two formalisms: Gough contains the convective growth rate σ_c inside the eddy survival probability, while Unno-type models start from an exponential relaxation dF_c/dt = (F_c0 - F_c)/τ_c.

### Part III: Reynolds Stress Models / Reynolds 응력 모델 (§4, pp.33–34)

**한국어**: 혼합거리 모델은 단일 경험 매개변수 α에 의존하지만, Xiong(1977, 1989, 2007) 와 Canuto(1992, 1993)가 제시한 Reynolds 응력 모델은 2차(때로는 3차) 모멘트들에 대한 개별 수송 방정식을 풀어 대류를 기술한다. 이 접근은 원리적으로 스칼라 난류 점성이나 평균 변형률에 대한 선형 의존성 가정을 넘어선다. Xiong 모델은 태양 p-mode 안정성에서 소규모 난류 점성에 의한 감쇠 W_ν (Eq.138)가 지배적임을 주장하는 반면, Gough/Grigahcène 모델은 각각 난류 압력 섭동 δp_t 또는 섭동된 대류 열 플럭스 δF_c를 감쇠의 주범으로 지목한다. 이는 현재 미해결 쟁점이다.

**English**: Whereas MLT hinges on a single empirical parameter α, Reynolds stress models (Xiong 1977, 1989, 2007; Canuto 1992, 1993) solve individual transport equations for second- (occasionally third-) order moments of the convective fluctuations, in principle freeing themselves from scalar turbulent-viscosity assumptions. Xiong's model attributes solar p-mode stability predominantly to small-scale turbulent viscosity damping W_ν (Eq.138); Gough assigns it to turbulent-pressure perturbations δp_t; Grigahcène et al. assign it to the perturbed convective heat flux δF_c. The physical origin of solar damping is thus still debated.

### Part IV: Convection Effects on Pulsation Frequencies / 맥동 진동수에 대한 대류 효과 (§5, pp.35–40)

**한국어**: 관측된 태양 진동수와 표준 모델 S (Christensen-Dalsgaard et al. 1996) 사이의 inertia-scaled 잔차는 ν ~ 2.5 mHz 이상에서 **~13 μHz**에 달하는 계통 오차를 보인다(Figure 2a). 이 "surface effect"는 거의 l (구면 차수) 독립적이며 ν에만 의존하므로 근표면층의 불완전 물리(대류, 복사 전달, 비단열)를 의미한다. 대류의 두 효과는: **(i) 평형 구조**에서 난류 압력 p_t의 기여 (Eq.22), **(ii) 맥동 계산**에서 δp_t와 δF_c의 섭동적 기여. Figure 3은 p_t가 태양에서 총 압력의 **최대 15%**까지 증가함을 보여준다. δ Scuti 별에서는 같은 비율이 **70%**에 도달할 수 있다 (Houdek 2000). Houdek(1996)는 세 가지 모델 L.a(국소, p_t 무시), NL.a(비국소, p_t 포함 단열), NL.na(비국소, p_t 포함 비단열 with δp_t + δF_c)를 비교했다. 주요 결과(Figure 4): p_t의 평형 구조 효과는 단열 진동수를 **~12 μHz** 낮추지만(NL.a-L.a), 비단열 효과는 **~9 μHz** 올려(NL.na-NL.a), 순 효과는 ~3 μHz 감소로, 관측 ~13 μHz의 상당 부분을 재현한다.

**Kjeldsen et al.(2008) 보정**: δν = a(ν/ν_0)^b 거듭제곱법칙이 Kepler 시대 표준 휴리스틱. 태양에서 b ~ 4.9. Christensen-Dalsgaard(2012)는 음속 임계 진동수 ν_ac에 근거한 개선된 함수꼴을 제시.

**English**: Observed-minus-model frequency residuals reach ~**13 μHz** above ν ~ 2.5 mHz for the Sun (Figure 2a), with nearly no dependence on degree l — a signature of incomplete near-surface physics. Convection modifies pulsation frequencies through **(i)** the mean turbulent pressure p_t in the equilibrium structure (Eq.22) and **(ii)** Lagrangian perturbations δp_t and δF_c of Reynolds stress and convective heat flux in the pulsation equations. Figure 3 shows p_t can reach **15%** of total pressure in the Sun; in δ Scuti envelopes it can exceed **70%** (Houdek 2000; Antoci et al. 2013). Houdek (1996) compared three model treatments: L.a (local, no p_t), NL.a (nonlocal, p_t included, adiabatic), NL.na (nonlocal, p_t with δp_t + δF_c perturbations). The Reynolds stress in the *equilibrium* structure depresses adiabatic frequencies by ~**12 μHz** (NL.a minus L.a), while nonadiabatic δp_t + δF_c raises them by ~**9 μHz** (NL.na minus NL.a), so the net surface effect is ~3 μHz depression — recovering a substantial fraction of the ~13 μHz observational residual.

**Kjeldsen et al. (2008) correction**: δν = a(ν/ν_0)^b is the workhorse empirical power-law (b~4.9 for the Sun). Christensen-Dalsgaard (2012) proposed an improved functional form referenced to the acoustic cutoff ν_ac.

### Part V: Driving and Damping Mechanisms / 구동과 감쇠 기작 (§6, pp.41–57)

**한국어**: **작업 적분** (work integral) W는 한 맥동 주기 동안 모드에 순 전달되는 (열+역학) 에너지의 부호. W > 0 → 내재적으로 불안정(self-excited); W < 0 → 감쇠. Eddington(1926)의 고전 관계 (Eq.128): W = ∫₀^M dm ∮ δT(∂δs/∂t) dt. 방사 맥동(radial pulsation)에서 성장률은
$$\frac{\omega_i}{\omega_r} = \frac{W_g + W_t + F}{2\pi\omega_r^2\int_{m_b}^M |\delta r|^2 dm} = -\hat{\eta}_g - \hat{\eta}_t + \hat{F}. \tag{132}$$

**§6.2 내재적 불안정 맥동성**:
- **Cepheid & RR Lyrae (§6.2.1)**: Baker & Gough(1979)가 Gough의 국소 TDC로 처음 RR Lyrae 적색 경계 재현. 이들은 **맥동 섭동된 난류 압력**이 주 안정화 기여자라고 결론.
- **Mira (§6.2.2)**: 2500-3500 K, P ≳ 80일. Gough(1966, 1967)는 δp_t가 **탈안정화**(구동) 한다고 주장. 저자별 결론이 엇갈리는 난제.
- **δ Scuti (§6.2.3)**: Dupret et al.(2005c)는 Grigahcène 모델로 붉은 경계 성공적 재현. 이 모델에서는 **섭동된 대류 열 플럭스 W_C**가 감쇠 주역 (Figure 6 right). 대조적으로 Houdek(2000)는 Gough 모델로 **W_t(난류 압력)**가 주역이라고 주장 (Figure 5). 서로 다른 모델이 *같은* 경계를 *다른* 이유로 재현한다는 점이 놀랍다. Xiong & Deng(2001)는 세 번째 관점(W_ν 난류 점성) 제시.
- **γ Doradus (§6.2.4)**: F형 g-모드 맥동성. **"Convective blocking"** (또는 "convective shunting")이 구동 기작: 대류가 동결되면 하부로부터의 flux가 쌓이고 높은 flux일 때 더 많이 투과 → g-모드 구동. Dupret et al.(2004b, 2005a,b)가 Grigahcène TDC로 재현.
- **roAp (§6.2.5)**: 6800-8400 K, 강한 쌍극 자기장으로 극지 대류가 억제되어 **κ-기작**이 수소 이온화 지역에서 고차 저차수 음향 모드를 구동. Balmforth et al.(2001).

**§6.3 태양형과 적색거성**: 태양 p-모드는 **확률적으로 여기**(stochastically excited)되고 **내재적으로 감쇠**. 선폭 Γ는 2η에 해당 (Eq.139). Figure 8은 η의 물리 기원을 η_dyn (운동량 균형) + η_g (열 균형)으로 분해. BiSON 관측 선폭은 이론 η_nl과 **수 μHz 수준**으로 일치 (Figure 10). ν~3 mHz에서 태양 Γ ~ **0.95 μHz**이 특징적 값; 이는 ~1.7일의 모드 수명에 해당. 선폭의 유효 온도 의존성은 Belkacem et al.(2012)이 Kepler 데이터로 **Γ ∝ T_eff^10.8**, Appourchaux et al.(2012)가 **Γ ∝ T_eff^13**, Houdek et al. (진행중) Gough 비국소로 **T_eff^7.5**를 보고.

**적색거성**: ξ Hydrae(Houdek & Gough 2002) 방사 p-모드 예측 수명 15-17일이 후속 CoRoT/Kepler 관측(Carrier et al. 2010; Huber et al. 2010)과 일치. Dupret et al.(2009)는 비방사 l=1,2 모드가 **혼합 모드**(p-core mixed)임을 보였고, g-지배 모드는 수명이 매우 길어 관측 window 내에서 미해결 스펙트럼 피크를 낳는다.

**English**: The **work integral** W is the sign of net (thermal + mechanical) energy transferred to the mode per cycle; W>0 → intrinsically unstable, W<0 → damped (Eddington 1926). For radial pulsations (Eq.132), ω_i/ω_r = (W_g + W_t + F) / (2πω_r² ∫|δr|² dm).

**§6.2 Intrinsically unstable pulsators**:
- **Cepheid & RR Lyrae**: Baker & Gough (1979) first reproduced the RR Lyr red edge with Gough's local TDC, attributing stability to pulsation-perturbed turbulent pressure.
- **Mira**: 2500–3500 K, P ≳ 80 d. Gough (1966, 1967) argued δp_t *destabilizes* long-period pulsations — an ongoing puzzle.
- **δ Scuti (§6.2.3)**: Dupret et al. (2005c), using Grigahcène's model, successfully reproduces the red edge with the **perturbed convective heat flux W_C** as the dominant damping contribution (Figure 6 right). Houdek (2000), using Gough's model, reaches the same red edge but with **W_t (turbulent pressure)** as the dominant stabilizer (Figure 5). Xiong & Deng (2001) propose yet a third interpretation via **turbulent viscosity W_ν**. That three different physical agents reproduce the *same* boundary highlights the underdetermination of current TDC physics.
- **γ Doradus**: F-type g-mode pulsators driven by "**convective blocking**" (more precisely "convective shunting") — at the base of a thin surface convection zone, the flux is redirected in phase with the pulsation, driving g modes. Successfully modeled by Dupret et al. (2004b, 2005a,b) with the Grigahcène TDC treatment.
- **roAp**: 6800–8400 K, strong dipolar magnetic fields suppress convection in polar spots, allowing the κ-mechanism in the H-ionization zone to drive high-order low-degree acoustic modes (Balmforth et al. 2001).

**§6.3 Solar-type and red-giant oscillators**: Solar p modes are **stochastically excited** and **intrinsically damped**. The Lorentzian linewidth Γ = 2η/(2π) (Eq.139) links theory to observation. Figure 8 decomposes η into η_dyn (momentum balance) + η_g (thermal balance). BiSON linewidths agree with theoretical η_nl to within a few percent across 1.5–4 mHz (Figure 10). At ν~3 mHz the Sun has Γ~**0.95 μHz**, corresponding to mode lifetimes of ~1.7 days. The surface-temperature scaling of Γ from Kepler data is Γ ∝ T_eff^10.8 (Belkacem et al. 2012) or ∝ T_eff^13 (Appourchaux et al. 2012); Houdek et al. (in prep.) with Gough nonlocal model obtain T_eff^7.5 — a mild tension still to be resolved.

**Red giants**: ξ Hydrae radial p-mode lifetimes of 15–17 days (Houdek & Gough 2002) match CoRoT/Kepler (Carrier et al. 2010; Huber et al. 2010). Dupret et al. (2009) show nonradial l=1,2 modes are **mixed modes** (p near surface, g in core); g-dominated modes have very long lifetimes, creating unresolved spectral features.

**Numerical comparison for ξ Hydrae (red giant)**:
- Characteristic frequency: ν_max ≈ 80 μHz
- Predicted radial-mode lifetime: τ_rad ≈ 15–17 days
- Initial measurement by Stello et al. (2004, 2006): τ ≈ 2–3 days (underestimate — spectral peaks not resolved in 150-day observing run)
- Resolution later from CoRoT ≥ 150 days run: τ ≈ 15 days — agrees with Houdek & Gough prediction

**Mode height-lifetime relation**: For stochastic mode H ∝ τ V²_rms when τ ≫ T, but H ∝ T V²_rms for τ ≪ T, so unresolved g-dominated mixed modes have small H and are nearly invisible in Kepler spectra.

**한**: 이 예시는 대류-맥동 상호작용 모델링이 단지 진동수(real part)뿐 아니라 선폭(imaginary part)과 모드 높이(진폭)의 모든 관측 관찰값을 통해 엄격히 시험됨을 보여준다. 혼합 모드는 별 중심부의 구조까지 진단 가능하게 하지만, TDC의 경계 처리가 여전히 정확한 수명 예측의 제한 요인이다.
**En**: This example illustrates how convection–pulsation modelling is stringently tested not only by frequencies (real part) but also by linewidths (imaginary part) and mode heights (amplitude). Mixed modes probe the stellar core structure, but TDC boundary treatments remain the limiting factor for accurate lifetime predictions.

### Part VI: Multi-colour Photometry and Mode Identification / 다색 광도측정과 모드 식별 (§7, pp.58–60)

**한국어**: 비방사 맥동의 mono-chromatic 광도 변화 δm_λ (Eq.140)는 spherical degree l의 Legendre 함수를 포함하므로, 다른 passband에서 관측된 진폭비와 위상차가 l을 식별하는 데 사용된다. 핵심 이론 입력은 **f_T**(국소 유효 온도 변화의 진폭)와 **ψ_T**(위상)이며, 이들은 비단열 계산에서만 엄밀히 얻어진다. δ Sct 별 FG Vir에서 Daszyńska-Daszkiewicz et al.(2005)는 Gough(1977b) 비국소 TDC와 frozen convection 모델을 비교해, **frozen convection은 완전히 실패**하고 TDC가 필수임을 보였다. γ Dor에서도 마찬가지로 frozen convection은 위상 불일치를 내지만 TDC는 관측과 일치한다.

**English**: The monochromatic light variation δm_λ (Eq.140) for a nonradial mode depends on the associated Legendre function of degree l, so amplitude ratios and phase differences across photometric passbands identify l. Key nonadiabatic theoretical inputs are **f_T** (amplitude of local effective-temperature variation) and **ψ_T** (phase), obtainable only from full nonadiabatic calculations. For the δ Sct star FG Vir, Daszyńska-Daszkiewicz et al. (2005) show that frozen-convection models give completely wrong phases, while the nonlocal Gough (1977b) TDC matches observations. The same holds for γ Dor stars: TDC is required, not optional.

### Part VII: Appendices A-E overview / 부록 A-E 개요 (pp.62–71)

**한국어**:
- **Appendix A (Gough's turbulent fluxes)**: 에디 생존 확률 P와 적분 표현 F_c, p_t의 유도. 생성률 nm = ρτ⁻¹ = ρσ_cχ 와 교정 χ=1/2의 도출. 이 교정은 정상 상태 (Eqs. 53-54)의 MLT 플럭스로부터 얻어진다.
- **Appendix B (perturbed convection coefficients, Gough)**: (151)-(175)의 W_10, W_11, W_12, Θ_10, Θ_11, Θ_12, Φ_10, Φ_11, κ_10, κ_11, κ_12 에 대한 완전한 표현. 복소 변수 σ̃=ω/σ_c의 함수. 이들이 앞서 본 식 (72-74)의 맥동 섭동 대류 플럭스 δF_c 에 들어가는 계수이다. F, G, H 함수 (171-173)는 감마 Γ, 디감마 F, 지수 적분 E_1 을 포함하는 특수함수 표현.
- **Appendix C (perturbation of mean structure in Grigahcène model)**: 맥동 섭동된 평균 구조(ρ, p, T, L_r)를 맥동 고유함수와 연결하는 관계식들. 복소 축약 매개변수 β̂에 의존.
- **Appendix D (Grigahcène convection equations)**: 평균 + 섭동 대류 방정식을 Grigahcène 형식론으로 재작성.
- **Appendix E (perturbed convection, Grigahcène)**: Gough 부록 B에 대응하는 Grigahcène 모델의 섭동 계수.

**English**:
- **Appendix A** derives Gough's eddy survival probability, integrated fluxes F_c, p_t, creation rate nm = ρσ_cχ and the calibration χ=1/2 from the steady-state MLT fluxes.
- **Appendix B** gives complete expressions (151-175) for the complex-valued perturbed-convection coefficients W_ij, Θ_ij, Φ_ij, κ_ij as functions of σ̃=ω/σ_c that enter δF_c and δp_t in the pulsation equations. Special functions F, G, H involve gamma Γ, digamma F, and exponential integral E_1.
- **Appendices C-E** do the analogous work for Grigahcène et al.'s nonradial model: the perturbed mean structure (ρ, p, T, L_r), the convection equations, and the coefficients of the perturbed fluxes. These appendices are the reference for anyone implementing either TDC formalism numerically.

### Part VIII: Brief Discussion and Prospects / 간단한 논의와 전망 (§8, pp.60–61)

**한국어**: 현재 3D 유체역학 시뮬레이션의 계산비용은 대류 박스(~10 Mm)에 제한되며 전체 별은 어렵지만, (Trampedach et al. 2014a 같은) 평균된 3D 결과로 1D 모델의 외곽층을 교체하거나 mixing length를 교정하는 **혼성 접근**(hybrid approach)이 유망하다. Mundprecht et al.(2015)은 2D Cepheid 시뮬레이션으로 1D 대류 매개변수가 맥동 주기 동안 **최대 7.5배** 변동함을 보였다. 즉 1D에서 상수로 가정된 α·Φ·β̂가 사실은 맥동 위상에 의존하는 함수여야 한다. 3D의 스케일 분리 (l_L / l_η ~ R_e^(3/4) ~ 10^9)는 태양 Reynolds 수 R_e~10^12 에서 10^27 격자점을 요구해 실제 시뮬레이션은 hyperviscosity/Smagorinsky 부분격자 모델에 의존하는 LES이다.

**English**: Current 3D hydrodynamical simulations are limited to small boxes (~10 Mm) and LES (large eddy simulations) with sub-grid models. The promising path is a **hybrid approach**: use spatially and temporally averaged 3D results to replace the outer layers of 1D models (Rosenthal et al. 1995, 1999; Trampedach et al. 2014a) and calibrate the mixing length. Mundprecht et al. (2015) show with 2D Cepheid simulations that "constant" 1D convection parameters in fact vary by up to a factor 7.5 over a pulsation cycle — a fundamental limitation of purely 1D TDC. The Reynolds-number scale separation l_L/l_η ~ R_e^(3/4) ~ 10^9 in the Sun (R_e~10^12) would require ~10^27 grid points for direct simulation, still some 15 orders of magnitude beyond today's computers.

---

## 3. Key Takeaways / 핵심 시사점

1. **Reynolds separation is the foundation of all semi-analytical TDC / Reynolds 분리는 모든 반해석적 TDC의 기초** — 평균 + 섭동 분해와 Boussinesq 근사를 통해 완전한 hydro 방정식을 해석 가능한 형태로 축소한다. 이로부터 난류 압력 p_t와 대류 열 플럭스 F_c의 평균과 섭동 양쪽이 자연스럽게 정의된다. The full Navier-Stokes-like system is reduced to tractable mean + fluctuation equations in which turbulent pressure p_t and convective heat flux F_c emerge as the essential macroscopic quantities linking convection to pulsation.

2. **Two dominant 1D TDC formalisms exist and they disagree on physical causes / 두 주류 1D TDC 형식론이 있고 물리적 원인에서 불일치한다** — Gough(1977a,b)는 에디 생존 확률에 기반한 혼합거리 모델; Unno(1967)/Grigahcène(2005)는 지수 완화 닫힘으로 출발하여 복소 매개변수 β̂로 보정. δ Sct 적색 경계는 양쪽 모델에서 재현되나, Gough는 δp_t, Grigahcène는 δF_c, Xiong은 난류 점성 W_ν를 주 감쇠 기작으로 지목 — 현재 TDC 물리는 *과소결정*(underdetermined) 상태이다. Gough's mixing-length model (eddy survival probability) and Grigahcène's generalization of Unno (exponential relaxation + complex closure β̂) both reproduce the δ Sct red edge but disagree on *why* — indicating TDC physics is still underdetermined.

3. **Turbulent pressure p_t is the convective signature in mean structure / 난류 압력 p_t는 평균 구조에서의 대류 서명** — 태양 외곽층에서 p_t는 총 압력의 ~15%, δ Sct 표면에서는 ~70%에 달한다. 이를 정수압 평형 방정식에 포함시키면 단열 p-모드 진동수가 ~12 μHz **하강**하며, 이것이 "surface effect"의 대부분을 설명한다. Turbulent pressure reaches ~15% of total pressure at the solar surface and ~70% in δ Sct stars. Including p_t in hydrostatic equilibrium depresses adiabatic p-mode frequencies by ~12 μHz in the Sun, accounting for most of the "surface effect".

4. **Linewidth Γ = 2η directly tests TDC physics / 선폭 Γ = 2η은 TDC 물리를 직접 시험한다** — 태양 p-모드의 Lorentzian 선폭 Γ~**0.95 μHz** at ν~3 mHz은 생명시간 τ~1.7일에 해당. CoRoT, Kepler의 수천 개 별에 대한 정밀 선폭 측정은 Γ ∝ T_eff^(10-13)을 시사하며, 이는 TDC 모델의 엄격한 시험대이다. Appourchaux et al.(2012)의 Γ ∝ T_eff^13과 이론의 T_eff^7.5~10.8 사이에는 아직 수 σ의 긴장 관계가 남아 있다. The solar p-mode linewidth Γ~0.95 μHz at 3 mHz (lifetime ~1.7 days) and its T_eff scaling provide stringent tests; current tension between theory (T_eff^7.5–10.8) and Kepler data (T_eff^13) remains unresolved.

5. **κ-mechanism vs convective blocking are distinct driving channels / κ-기작과 대류 차단은 별개의 구동 채널** — **κ-기작**은 부분 이온화 영역(H, He I, He II)의 opacity 범프가 뜨거운 위상에서 flux를 가두어 구동하며, 주로 p-모드 고전 맥동성(Cepheid, RR Lyr, δ Sct, roAp)에서 작동. **대류 차단**(convective blocking)은 γ Dor 같은 F-g 맥동성에서 대류가 주기 동안 동결되어 radiative flux가 압축 위상에서 축적/투과 차이를 내어 g-모드를 구동한다. The κ-mechanism drives p-mode pulsation via opacity bumps in partial-ionization zones (Cepheid, RR Lyr, δ Sct, roAp); convective blocking drives γ Dor g modes by flux redistribution at the base of a thin convective envelope — two physically distinct channels both requiring TDC to be computed correctly.

6. **Solar-like oscillations are stochastically excited and intrinsically damped / 태양형 진동은 확률적으로 여기되고 내재적으로 감쇠한다** — 대류가 난류 "noise"로 p-모드를 구동하는 동시에 동일한 대류가 감쇠를 제공한다. Figure 8의 η 분해(η_scatt + η_leak + η_t + η_rad + η_conv)는 Reynolds stress 섭동이 η_t에, 대류 열 플럭스 섭동이 η_conv에 기여함을 보여준다. "General κ-mechanism" (Balmforth 1992a)은 ν_max 근처에서 부분적 탈안정화를 제공해 p-모드 파워 엔벨로프의 국소 감쇠율 함몰을 만든다. Convection simultaneously excites (stochastic driving) and damps solar p modes; the η budget decomposes into contributions from Reynolds stress (η_t), convective flux (η_conv), radiative (η_rad), and non-computed processes (η_scatt, η_leak).

7. **"Frozen convection" fails for short-period modes / 단주기 모드에서 "frozen convection"은 실패한다** — 동결 대류(δF_c = δp_t = 0)는 수학적으로 간편하지만, 대류 시간 규모와 맥동 주기가 비슷한 모든 표면 대류가 있는 별(δ Sct, γ Dor, 태양형)에서 질적으로 틀린 결과(잘못된 위상, 적색 경계 실패, 선폭 오차)를 낸다. 이는 비단열 다색 광도측정 모드 식별에서 특히 극적으로 드러난다. Frozen convection (δF_c = δp_t = 0) is mathematically convenient but fails qualitatively for any star with a surface convection zone (δ Sct, γ Dor, solar-type), producing wrong phases in multi-colour photometry and missing the instability-strip red edge.

8. **Future lies in hybrid 3D-to-1D modelling / 미래는 혼성 3D-1D 모델링에 있다** — 전체 별에 대한 DNS는 R_e ~10^12 때문에 10^27 격자를 필요로 하여 불가능. 현실적 해답은 평균된 3D 박스 시뮬레이션(Stein & Nordlund, Trampedach, Magic)으로 1D TDC 모델을 교정하고, 외곽층을 교체하거나 mixing-length 매개변수를 맥동 위상에 따라 변화시키는 것. Mundprecht et al.(2015)의 2D Cepheid 결과는 "상수" α가 실제로 7.5배 변동함을 보였다. Full 3D simulations are out of reach (R_e~10^12 requires ~10^27 grid points); the pragmatic future is hybrid — using averaged 3D simulations to calibrate and replace the outer layers of 1D TDC models, with pulsation-phase-dependent α.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Mean Reynolds-averaged equations / 평균 Reynolds 방정식

Mass conservation:
$$\frac{\partial\bar{\rho}}{\partial t} + \nabla\cdot(\bar{\rho}\bm{U}) = 0. \tag{8}$$

Mean momentum with Reynolds stress:
$$\bar{\rho}\frac{d\bm{U}}{dt} = \bar{\rho}\bar{\bm{g}} - \nabla(\bar{p} + p_t) - \nabla\cdot\bm{\sigma}_t. \tag{9}$$

**한**: d/dt = ∂/∂t + U·∇는 맥동 흐름 따라 미분. p_t와 σ_t는 Reynolds 응력 텐서의 등방/비등방 부분.
**En**: d/dt = ∂/∂t + U·∇ is the Lagrangian derivative along the pulsation flow; p_t and σ_t split the Reynolds stress isotropic/anisotropic.

Turbulent kinetic energy (Boussinesq):
$$\frac{\bar{\rho}}{2}\frac{d\overline{|\bm{u}|^2}}{dt} = -\overline{\rho\bm{uu}}:\nabla\bm{U} - \overline{\bm{u}\cdot\nabla p} - \bar{\rho}\epsilon_t. \tag{15}$$

**한**: 좌변은 난류 운동에너지의 물질 미분; 우변은 생성 – 압력 일 – 점성 소산.
**En**: Material derivative of turbulent KE = production – pressure work – viscous dissipation.

### 4.2 Mixing-length convective flux / 혼합거리 대류 플럭스

Standard MLT (Böhm-Vitense 1958):
$$F_c = \bar{\rho}\,c_p\,\overline{w T'} = \frac{1}{2}\bar{\rho}\,c_p\,(g\hat{\delta})^{1/2}\,\ell^2\,(\beta)^{3/2}\,H_p^{-1/2}.$$

**한**: 초단열 기울기 β = ∇̄ - ∇̄_ad를 구동력으로 하며, 이동 거리 ℓ 동안 fluid parcel이 평형으로부터 얻는 과잉 온도가 wT' ~ ℓ β w를 만들어낸다. β^(3/2) 스케일링은 대류가 초단열성에 매우 민감함을 뜻한다.
**En**: The superadiabatic gradient β drives the flux; over mixing length ℓ a parcel acquires excess temperature ~ℓ β, leading to wT' ~ ℓβw. The β^(3/2) scaling means convection is highly sensitive to superadiabaticity.

### 4.3 Eddy survival probability & fluxes (Gough's model) / 에디 생존 확률과 플럭스

Eddy survival probability:
$$\mathcal{P}(r, t, t_0) = \exp\left[-\frac{\widehat{W_0}\,e^{\sigma_c(t-t_0)}}{\sigma_c\,\ell}\right]. \tag{146}$$

**한**: 시각 t_0에 생성된 eddy가 t까지 살아남을 확률. 성장률 σ_c과 혼합거리 ℓ로 특징지어진다.
**En**: Probability that an eddy created at t_0 survives to t. Characterized by linear growth rate σ_c and mixing length ℓ.

Integrated fluxes:
$$F_c = n\,m\,c_p \int_{-\infty}^t W \Theta \mathcal{P}\,dt_0, \quad p_t = n\,m \int_{-\infty}^t W^2 \mathcal{P}\,dt_0. \tag{147-148}$$

Statistical steady state: nm ∫P dt_0 = ρ (Eq.149), calibration χ = 1/2.

### 4.4 Linear pulsation + TDC (Gough formalism)

The linearized pulsation equation for displacement ξ_r = δr, in complex frequency ω = ω_r + i ω_i:
$$\omega^2 \xi_r = \frac{1}{\bar{\rho}}\frac{d}{dq}\delta(\bar{p} + p_t) + \delta g + (3 - \Phi)\frac{\delta p_t}{r\bar{\rho}} + \cdots$$

Perturbed convection equations (see Appendix B, Eqs. 151-170) couple δF_c, δp_t, δℓ, δσ, δχ to the pulsation variables through complex coefficients W_{10}, W_{11}, Θ_{10}, ...

### 4.5 Work integral and growth rate / 작업 적분과 성장률

Eddington's general expression:
$$W = \int_0^M dm \oint \delta T\,\frac{\partial \delta s}{\partial t}\,dt. \tag{128}$$

Growth rate from the imaginary part of the eigenfrequency:
$$\frac{\omega_i}{\omega_r} = \frac{W_g + W_t + F}{2\pi\,\omega_r^2\int_{m_b}^M |\delta r|^2\,dm} \equiv \hat{\eta}_g + \hat{\eta}_t + \hat{F}. \tag{132}$$

**한**: W_g = ∫dm π/ρ² Im(δp·δρ*); W_t = 난류 압력 기여. ω_i > 0 이면 모드는 성장 (불안정), ω_i < 0 이면 감쇠.
**En**: W_g is gas-pressure work, W_t is turbulent-pressure work. ω_i > 0 means growing (unstable), ω_i < 0 means damped.

### 4.6 p-mode Lorentzian line profile / p-모드 Lorentzian 선 프로파일

$$P(\nu) = \frac{H}{1 + [(\nu - \nu_0)/(\Gamma/2)]^2}, \quad \Gamma = \frac{2\eta}{2\pi} = \pi^{-1}\eta_{nl}, \quad H = 2\tau V_{\rm rms}^2, \quad \tau = \eta^{-1}. \tag{139}$$

**한**: 확률적으로 여기된 모드의 파워 스펙트럼은 Lorentzian 모양. FWHM Γ은 선형 감쇠율 η의 2배(각진동수 단위). 높이 H는 관측 시간 T보다 수명 τ가 짧을 때 mode inertia 독립적.
**En**: Stochastically excited modes produce Lorentzian power-spectrum peaks of FWHM Γ = 2η, height H = 2τ V²_rms. For T > τ, H is independent of mode inertia.

### 4.7 Numerical example — Sun / 수치 예시 — 태양

Using Gough's nonlocal TDC (Balmforth 1992a; Houdek et al. 1999a):
- Solar 5-minute oscillations: ν_max ≈ **3.0 mHz**, ν_ac ≈ **5.5 mHz**.
- At ν = 3 mHz: η ≈ **1.5 μHz** (half angular frequency; cyclic Γ = η/π ≈ 0.48 μHz). BiSON FWHM measurement Γ_obs ≈ **0.95 μHz** — good agreement to factor ~2.
- Mode lifetime τ = 1/(2πΓ) ≈ **1.7 days** at ν_max.
- Surface effect at ν = 4 mHz: Δν_obs-model ≈ **–13 μHz**; TDC nonlocal model (NL.na–L.a) predicts ≈ **–3 μHz** residual after adiabatic p_t correction of –12 μHz plus nonadiabatic correction +9 μHz.

### 4.8 Linearized convective perturbations (schematic) / 선형화된 대류 섭동 (개요)

Each perturbed quantity in Gough's model is written as a linear combination:
$$\frac{\delta F_c}{F_{c,0}} = \mathcal{F}\left(\frac{\delta W}{W_0}\right) + \mathcal{G}\left(\frac{\delta \Theta}{\Theta_0}\right) + \mathcal{H}\,(\text{kinematic terms})$$

where 𝓕, 𝓖, 𝓗 (Eqs. 171-173) contain
- 𝓕 = 𝓘 Γ(2 - iσ̃)
- 𝓖 = 𝓙/𝓘 + 𝓕(2 - iσ̃)
- 𝓘 = 107{E_1[2.88(1+iσ̃)] – 320 E_1[2.88(3+iσ̃)]}
- 𝓙 = 12/[(1+iσ̃)(3+iσ̃)] (5^{1/2} s / 2)^{iσ̃} with s = 0.05.

**한**: 성장률 σ_c가 명시적으로 나타나는 σ̃=ω/σ_c 의존성은 Gough 모델의 고유 서명. 특수 함수 E_1은 에디 생존 확률의 누적 적분에서 발생.
**En**: The explicit σ̃=ω/σ_c dependence is Gough's signature; the exponential integral E_1 arises from accumulating the eddy survival probability over past creation times.

### 4.9 Worked trace: δ Sct red-edge energetics / δ Sct 적색 경계 에너지 작업

For a 1.7 M_⊙ δ Scuti model at T_eff = 6813 K (near red edge), fundamental radial mode n=1 (Figure 5 right):
- W_g (gas pressure work) ≈ +0.3 × 10⁻⁸ (driving, positive at surface)
- W_t (turbulent pressure work) ≈ –0.7 × 10⁻⁸ (dominant damping)
- Total W = W_g + W_t < 0 → damped mode → outside red edge.

Cross the red edge by decreasing T_eff by ~100 K: W_t becomes less negative, W_g still positive, total W>0 → unstable → mode observable.

### 4.10 Worked trace: solar damping rate scaling / 태양 감쇠율 스케일링 계산

For a solar model with ν = 3 mHz (ν_max), the damping contributions at the damping-rate "hump" (Figure 9):
- η_g (radiative damping from perturbed convective flux): ≈ –0.5 μHz (destabilising near ν_max — the "general κ-mechanism")
- η_t (turbulent pressure perturbation): ≈ +2.0 μHz (stabilising)
- Total η = η_g + η_t ≈ +1.5 μHz
- Cyclic linewidth Γ = 2η/(2π) ≈ 0.48 μHz (one factor of 2π in each conversion)
- Observed BiSON Γ ≈ 0.95 μHz — agreement within factor ~2.

Mode lifetime: τ = 1/η ≈ 1/(2π × 1.5 μHz) ≈ 1.1 × 10⁵ s ≈ 1.3 days.

**한**: 이 수치 예시는 태양 p-모드의 감쇠가 (i) 난류 압력 섭동에 의한 주된 안정화 +2 μHz, (ii) 대류에 의한 복사 플럭스 변조(η_g)가 ν_max 근처에서 *탈안정화*하여 국소 함몰을 만드는 균형으로부터 나옴을 보인다.
**En**: This example shows solar p-mode damping is a delicate balance: turbulent-pressure perturbation dominates stabilisation (+2 μHz), but the radiative modulation by convection (η_g) becomes *destabilising* near ν_max, creating the local dip known as the "general κ-mechanism" that sets the amplitude maximum location.

### 4.11 Stochastic excitation — amplitude balance / 확률적 여기 — 진폭 균형

For a stochastically excited mode (solar p-mode), the rms velocity amplitude is set by:
$$V^2_{\rm rms} = \frac{P_{\rm dr}}{2\,I\,\eta}$$

where P_dr is the driving power from turbulent convection (Lighthill mechanism), I is the mode inertia, η the damping rate. The observed power-spectrum height:
$$H = \frac{2\,V^2_{\rm rms}}{\pi\,\Gamma} = \frac{P_{\rm dr}}{\pi\,I\,\eta^2}.$$

**한**: 확률적 여기(driving)와 선형 감쇠의 평형이 관측 가능한 진폭을 결정. τ=1/η → Γ = 1/(πτ).
**En**: The balance between stochastic driving power and linear damping determines the observable amplitude. The mode height H and width Γ together carry direct information about both driving and damping physics.

### 4.12 Damping decomposition (Figure 8) / 감쇠 분해

The total linear damping rate η for a stochastically excited mode decomposes (Houdek et al. 1999a; Figure 8):
$$\eta = \eta_{\rm dyn} + \eta_g = \underbrace{\eta_{\rm scatt}}_{\text{incoherent scatter}} + \underbrace{\eta_{\rm leak}}_{\text{atmos leak}} + \underbrace{\eta_t}_{\text{turb momentum}} + \underbrace{\eta_{\rm rad}}_{\text{radiative}} + \underbrace{\eta_{\rm conv}}_{\text{convective heat}}$$

- η_scatt: incoherent scattering in the horizontally inhomogeneous superadiabatic boundary layer (Goldreich & Murray 1994); not computed in Houdek et al. (1999a).
- η_leak: transmission of waves into atmosphere (Balmforth & Gough 1990b); not computed in most stability calculations.
- η_t: modulation of turbulent momentum flux p_t by pulsation (Gough 1977).
- η_rad: radiative damping due to nonadiabatic departures from radiative equilibrium (Christensen-Dalsgaard & Frandsen 1983a).
- η_conv: modulation of turbulent heat flux F_c by the pulsation (Gough 1977); the *destabilising* branch of the "general κ-mechanism".

**한**: 태양에서 η_t와 η_conv의 경쟁이 ν_max 근처 "general κ-mechanism" 함몰을 만든다. Kepler 별에서는 η의 T_eff 의존성이 관측 Γ 스케일링을 결정.
**En**: In the Sun, η_t and η_conv compete near ν_max to produce the "general κ-mechanism" dip. In Kepler stars, the T_eff dependence of each component determines the observed Γ scaling.

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
┌──────────────────────────────────────────────────────────────────────┐
│              Theory of Convection–Pulsation Interaction              │
└──────────────────────────────────────────────────────────────────────┘

1877 ──── Boussinesq: eddy viscosity concept
           │
1915-1925 ─ Taylor, Prandtl: mixing-length idea
           │
1932-1935 ─ Biermann, Siedentopf: apply MLT to stars
           │
1958 ───── Böhm-Vitense: standard stellar MLT
           │
1962 ───── Baker & Kippenhahn: linear Cepheid instability (frozen)
           │
1965 ───── Gough (thesis): first TDC for Cepheids
           │
1966-1967 ─ Gough: Mira pulsation with TDC [δp_t destabilises]
           │
1967,1977 ─ Unno: convection model with exponential relaxation
           │
1977 ───── Gough (1977a,b): local & nonlocal mixing-length TDC ★★
           │
1979 ───── Baker & Gough: RR Lyrae red edge with local TDC
           │
1990 ───── Balmforth et al.: nonlocal TDC for Mira envelopes
           │
1992 ───── Balmforth: solar p-mode damping with Gough TDC ★
           │
1995 ───── Rosenthal et al.: turbulent pressure + hydro simulation
           │
1996 ───── Houdek: L.a, NL.a, NL.na decomposition of solar surface effect ★
           │
2000 ───── Houdek: δ Sct stability, W_t dominates red edge
           │
2002 ───── Houdek & Gough: ξ Hydrae red-giant lifetime prediction
           │
2004-2005 ─ Dupret, Grigahcène et al.: nonradial generalization of Unno
           │                              → γ Dor driving ★★
           │
2005 ───── Chaplin et al.: BiSON solar linewidth comparison
           │
2008 ───── Kjeldsen et al.: empirical surface-correction power law
           │
2009-2014 ─ CoRoT / Kepler: precision linewidths for thousands of stars
           │   - Baudin 2011, Appourchaux 2012: Γ ∝ T_eff^13
           │   - Belkacem 2012: T_eff^10.8
           │
2012 ───── Christensen-Dalsgaard: improved surface-correction form
           │
2014 ───── Ball & Gizon: scaling relation from mode inertia
           │
2015 ───── Houdek & Dupret: THIS REVIEW ★★★
           │
2020+ ──── Hybrid 3D-to-1D TDC (Trampedach, Magic, Mundprecht)
           │
Future ─── PLATO, Roman — tighter asteroseismic constraints
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Böhm-Vitense (1958) | Provides standard static MLT that Gough (1977) and Unno (1967) extend to time-dependent form. / Gough와 Unno가 시간 의존 대류로 확장한 정적 MLT 제공 | Foundational — without MLT there is no closed 1D convection theory |
| Gough (1977a,b) | Central formalism of this review. Eddy survival probability + local/nonlocal mixing-length TDC. / 본 리뷰의 중심 형식론. 에디 생존 확률 + 국소/비국소 혼합거리 TDC | Core — ~30% of review explains or applies Gough's model |
| Unno (1967, 1977) / Grigahcène et al. (2005) | The second main TDC formalism. Grigahcène et al. generalize Unno's model to nonradial pulsations, enabling γ Dor studies. / 두 번째 주류 TDC 형식론. 비방사 맥동 일반화로 γ Dor 구동 가능 | Core — §3.4, §3.6, §6.2.4 |
| Baker & Gough (1979) | First successful cool-edge reproduction of RR Lyrae using Gough's local TDC. / Gough의 국소 TDC로 RR Lyrae 저온 경계 재현 | Classical validation of TDC in radial pulsators |
| Balmforth (1992a,b) | Applied Gough nonlocal TDC to solar p-modes, predicting linewidths. / Gough의 비국소 TDC로 태양 p-모드 선폭 예측 | Benchmark for solar p-mode damping studies |
| Rosenthal et al. (1995, 1999) | Introduced 3D-simulation-based surface layers, establishing "hybrid" approach. / 3D 시뮬레이션 기반 외곽층으로 "혼성" 접근 확립 | Precursor to modern 3D-to-1D techniques |
| Houdek (1996, 2000); Houdek et al. (1999a) | Definitive calculation of solar/δ Sct surface effect and damping rates with Gough TDC. / Gough TDC로 태양/δ Sct 표면 효과와 감쇠율의 결정적 계산 | Heavily cited throughout §5-6 |
| Dupret et al. (2004b, 2005a,b,c) | Grigahcène TDC applied to γ Dor and δ Sct driving; established convective blocking for γ Dor. / Grigahcène TDC를 γ Dor와 δ Sct 구동에 적용; γ Dor의 대류 차단 확립 | Dupret co-author; defines §6.2.3-6.2.4 |
| Chaplin et al. (2005, 2009) | BiSON solar linewidth benchmarks; tested Gough nonlocal TDC against data. / BiSON 태양 선폭 기준; Gough 비국소 TDC 데이터 시험 | Primary observational constraint |
| Belkacem et al. (2012); Appourchaux et al. (2012) | Kepler linewidth scaling Γ ∝ T_eff^{10-13} — direct test of TDC predictions. / Kepler 선폭 스케일링 — TDC 예측 직접 시험 | Key empirical tension |
| Kjeldsen et al. (2008) | Empirical surface-effect power-law correction, routinely used. / 경험적 표면 효과 거듭제곱 보정, 일상 사용 | Workhorse tool in asteroseismology |
| Houdek & Gough (2002); Dupret et al. (2009) | Red-giant mode lifetimes; nonradial mixed modes. / 적색거성 모드 수명; 비방사 혼합 모드 | §6.3.2 |
| Trampedach et al. (2014a); Magic et al. (2013, 2015) | 3D-averaged stellar surface grids used to calibrate 1D TDC. / 3D 평균 별 표면 격자로 1D TDC 교정 | Future direction |

---

## 7. References / 참고문헌

- Houdek, G. & Dupret, M.-A., "Interaction Between Convection and Pulsation", *Living Rev. Solar Phys.*, 12, 8 (2015). DOI: 10.1007/lrsp-2015-8
- Gough, D. O., "Mixing-length theory for pulsating stars", *Astrophys. J.*, 214, 196 (1977a).
- Gough, D. O., "The current state of stellar mixing-length theory", in *Problems of Stellar Convection*, Lect. Notes Phys. 71, 15 (1977b).
- Unno, W., "Stellar Radial Pulsation Coupled with the Convection", *Publ. Astron. Soc. Jpn.*, 19, 140 (1967).
- Grigahcène, A., Dupret, M.-A., Gabriel, M., Garrido, R. & Scuflaire, R., "Convection-pulsation coupling. I. A mixing-length perturbative theory", *Astron. Astrophys.*, 434, 1055 (2005).
- Böhm-Vitense, E., "Über die Wasserstoffkonvektionszone in Sternen verschiedener Effektivtemperaturen und Leuchtkräfte", *Z. Astrophys.*, 46, 108 (1958).
- Baker, N. & Gough, D. O., "Pulsations of model RR Lyrae stars", *Astrophys. J.*, 234, 232 (1979).
- Balmforth, N. J., "Solar pulsational stability — I. Pulsation-mode thermodynamics", *Mon. Not. R. Astron. Soc.*, 255, 603 (1992a).
- Rosenthal, C. S., Christensen-Dalsgaard, J., Nordlund, Å., Stein, R. F. & Trampedach, R., "Convective contributions to the frequencies of solar oscillations", *Astron. Astrophys.*, 351, 689 (1999).
- Houdek, G., Balmforth, N. J., Christensen-Dalsgaard, J. & Gough, D. O., "Amplitudes of stochastically excited oscillations in main-sequence stars", *Astron. Astrophys.*, 351, 582 (1999a).
- Houdek, G., "Pulsations in Delta Scuti stars", Doctoral Thesis, University of Vienna (1996).
- Dupret, M.-A. et al., "Time-dependent convection seismic study of δ Scuti stars", *Astron. Astrophys.*, 435, 927 (2005c).
- Chaplin, W. J. et al., "BiSON mode linewidths", *Mon. Not. R. Astron. Soc.*, 360, 859 (2005).
- Belkacem, K. et al., "Seismic diagnostics of solar-type stars: line widths from Kepler", *Astron. Astrophys.*, 540, L7 (2012).
- Appourchaux, T. et al., "Oscillation mode linewidths and heights of 23 main-sequence stars observed by Kepler", *Astron. Astrophys.*, 537, A134 (2012).
- Kjeldsen, H., Bedding, T. R. & Christensen-Dalsgaard, J., "Correcting stellar oscillation frequencies for near-surface effects", *Astrophys. J. Lett.*, 683, L175 (2008).
- Houdek, G. & Gough, D. O., "Modelling pulsations of the subgiant β Hydri", *Mon. Not. R. Astron. Soc.*, 336, L65 (2002).
- Dupret, M.-A., Belkacem, K., Samadi, R. et al., "Theoretical amplitudes and lifetimes of non-radial solar-like oscillations in red giants", *Astron. Astrophys.*, 506, 57 (2009).
- Trampedach, R., Stein, R. F., Christensen-Dalsgaard, J., Nordlund, Å. & Asplund, M., "Improvements to stellar structure models, based on a grid of 3D convection simulations", *Mon. Not. R. Astron. Soc.*, 442, 805 (2014a).
- Christensen-Dalsgaard, J. et al., "The current state of solar modeling", *Science*, 272, 1286 (1996).
- Mundprecht, E., Muthsam, H. J. & Kupka, F., "Multidimensional simulations of Cepheid pulsations", *Mon. Not. R. Astron. Soc.*, 449, 2539 (2015).
- Xiong, D.-R., Cheng, Q. & Deng, L., "Turbulent convection and pulsational stability of variable stars", *Astrophys. J. Suppl.*, 108, 529 (1997).
- Canuto, V. M. & Mazzitelli, I., "Stellar turbulent convection: A new model and applications", *Astrophys. J.*, 370, 295 (1991).
- Gabriel, M., Scuflaire, R., Noels, A. & Boury, A., "Convection et instabilité des Céphéides", *Astron. Astrophys.*, 40, 33 (1975).
