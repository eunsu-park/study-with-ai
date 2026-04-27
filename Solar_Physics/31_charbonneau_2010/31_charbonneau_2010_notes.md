---
title: "Dynamo Models of the Solar Cycle — Reading Notes"
date: 2026-04-27
topic: Solar_Physics
tags: [solar dynamo, mean-field, alpha-omega, Babcock-Leighton, flux transport, butterfly diagram, Maunder Minimum]
paper: Charbonneau (2010), Living Reviews in Solar Physics 7, 3
doi: 10.12942/lrsp-2010-3
status: completed
---

# Dynamo Models of the Solar Cycle — Reading Notes
# 태양 주기의 다이나모 모델 — 읽기 노트

## 1. Core Contribution / 핵심 기여

**English.** Charbonneau (2010) is a comprehensive critical review of solar dynamo theory as it stood twenty-five years after the helioseismic revolution. The review surveys every major class of solar cycle model — classical $\alpha\Omega$ mean-field dynamos, interface dynamos, mean-field models incorporating meridional circulation, models based on hydrodynamical/MHD shear or buoyancy instabilities, models based on flux-tube instabilities, Babcock–Leighton flux-transport dynamos, and global MHD simulations — and assesses each against the observational benchmarks set by sunspot observations: the 11-year cycle, the 22-year Hale polarity cycle, the equatorward butterfly migration, the polar field reversal, hemispheric (anti)symmetry, and amplitude/phase fluctuations including Grand Minima. The author's central thesis is that no single model commands consensus, but that the post-helioseismic landscape strongly favors models in which the $\Omega$-effect is concentrated at the tachocline and the poloidal-source mechanism (whether $\alpha$-effect or Babcock–Leighton) operates either in the bulk convection zone or at the surface, with meridional circulation acting as a dynamical link.

**한국어.** Charbonneau (2010)은 일진동학(helioseismology) 혁명 이후 25년 시점의 태양 다이나모 이론을 포괄적·비판적으로 정리한 리뷰입니다. 이 리뷰는 고전적 $\alpha\Omega$ 평균장 다이나모, interface 다이나모, 자오선 순환을 포함한 평균장 모델, 유체역학/MHD 전단·부력 불안정에 기반한 모델, 자속관 불안정 모델, Babcock–Leighton 플럭스 수송 다이나모, 그리고 전역 MHD 시뮬레이션 등 주요 태양 주기 모델 전반을 다룹니다. 각 모델은 흑점 관측에서 도출된 관측적 기준 — 11년 주기, 22년 Hale 극성 주기, 적도 방향 버터플라이 이동, 극성 자기장 역전, 반구 대칭성, Grand Minima를 포함한 진폭/위상 변동 — 에 비추어 평가됩니다. 저자의 중심 주장은, 합의된 단일 모델은 없지만 일진동학 이후의 풍경은 $\Omega$-효과가 tachocline에 집중되고 폴로이달 원천(난류 $\alpha$ 또는 Babcock–Leighton)은 대류층 내부 또는 표면에서 작동하며 자오선 순환이 둘을 동역학적으로 연결하는 모델을 강하게 선호한다는 것입니다.

The review's pedagogical strength is its consistent presentation of the kinematic axisymmetric mean-field equations as a common framework into which every model class is mapped, allowing direct comparison of operating modes (steady vs. oscillatory), parity, period, and butterfly morphology across model families. The 2010 update over the 2005 version added turbulent pumping, expanded the discussion of MHD simulations of large-scale dynamo action, and added a new section on dynamo-based cycle prediction.

이 리뷰의 교육적 강점은 운동학적 축대칭 평균장 방정식을 공통 틀로 일관되게 제시하여, 모든 모델 부류를 그 틀에 매핑함으로써 작동 양식(steady vs. 진동), 패리티, 주기, 버터플라이 형태를 모델군 간 직접 비교할 수 있게 한다는 점입니다. 2010 업데이트는 2005 버전 대비 난류 pumping을 추가했고, 대규모 다이나모 작용에 대한 MHD 시뮬레이션 논의를 확장했으며, 다이나모 기반 주기 예측에 관한 새 절을 추가했습니다.

## 2. Reading Notes (Section by Section) / 읽기 노트 (절별)

### §1 Introduction (pp. 7–11) / 서론

**English.** §1.1–1.2 frame the scope: the review treats the cyclic regeneration of the Sun's *large-scale* magnetic field via inductive fluid flows, deliberately excluding flux-rope physics, near-surface small-scale dynamos, and stellar dynamos. A "model" is defined as a deliberately simplified theoretical construct. §1.3 surveys the historical line: Hale's polarity laws (1908–1924) → Larmor's inductive-flow conjecture (1919) → Cowling's antidynamo theorem (1933) → Parker's cyclonic-twist resolution (1955) → mean-field electrodynamics consolidation (Steenbeck, Krause, Rädler 1960s) → the four-way crisis of the 1980s (flux storage problem, simulations don't reproduce the Sun, helioseismology contradicts mean-field $\Omega(r,\theta)$ predictions, theoretical doubts on $\alpha$-effect operation), which fueled the modern Babcock–Leighton revival. §1.4 introduces the butterfly diagram as the primary observational benchmark: sunspots in $\sim\pm 30°$ bands drifting equatorward through the cycle to $\sim\pm 15°$ at maximum, with the polar cap flux ($\sim 10^{22}$ Mx) much smaller than active-region flux per cycle ($\sim 10^{25}$ Mx) — implying the interior field is toroidal-dominated.

**한국어.** §1.1–1.2는 범위를 설정합니다: 태양의 대규모 자기장이 유도성 유체 흐름으로 주기적으로 재생되는 과정을 다루며, 자속관 물리, 표면 부근 소규모 다이나모, 항성 다이나모는 의도적으로 제외됩니다. "모델"은 의도적으로 단순화된 이론적 구성물로 정의됩니다. §1.3은 역사적 흐름을 정리합니다: Hale 극성 법칙 → Larmor의 유도-흐름 추측 → Cowling 반다이나모 정리 → Parker의 cyclonic 비틀림 해결 → 평균장 전기역학의 정착 → 1980년대의 4중 위기 → 현대 Babcock–Leighton 부흥. §1.4는 버터플라이 다이어그램을 주된 관측적 기준으로 도입합니다: 흑점은 $\pm30°$ 띠에서 시작해 주기 진행에 따라 $\pm15°$로 적도방향 이동하며, 극관 자속($\sim 10^{22}$ Mx)은 주기당 활동영역 자속($\sim 10^{25}$ Mx)보다 훨씬 작아 내부 자기장이 토로이달 우세임을 시사합니다.

### §2 Making a Solar Dynamo Model (pp. 12–14) / 태양 다이나모 모델 만들기

**English.** §2.1 presents the MHD induction equation
$$\frac{\partial \mathbf{B}}{\partial t} = \nabla\times(\mathbf{u}\times\mathbf{B}) - \nabla\times(\eta\nabla\times\mathbf{B}).$$
§2.2 defines the dynamo problem: find a flow $\mathbf{u}$ such that this PDE has solutions with $|\mathbf{B}|$ growing or sustained against ohmic dissipation. §2.3 introduces the kinematic approximation — $\mathbf{u}$ is prescribed (not back-reacted upon by the Lorentz force), reducing the problem to a linear PDE for $\mathbf{B}$. §2.4 presents the axisymmetric poloidal–toroidal decomposition
$$\mathbf{B}(r,\theta,t) = \nabla\times\left[A(r,\theta,t)\,\hat{e}_\phi\right] + B(r,\theta,t)\,\hat{e}_\phi,$$
giving two coupled scalar PDEs for $A$ and $B$. §2.5 covers boundary conditions (matching to a current-free exterior at $r=R_\odot$) and parity: dipolar (antisymmetric, "A0/D") vs quadrupolar (symmetric, "S0/Q") families.

**한국어.** §2.1은 MHD 유도 방정식을 제시합니다. §2.2는 다이나모 문제를 정의합니다: 옴 소산에 대해 자기장이 성장 또는 유지되는 해를 갖도록 하는 흐름 $\mathbf{u}$를 찾는 것. §2.3 운동학적 근사: $\mathbf{u}$를 미리 지정(Lorentz 반작용 무시)해 PDE를 $\mathbf{B}$에 대해 선형으로 만듭니다. §2.4 축대칭 폴로이달–토로이달 분해는 $A$, $B$ 두 스칼라에 대한 결합 PDE 쌍을 줍니다. §2.5 경계 조건은 $r=R_\odot$에서 무전류 외부와의 정합이며, 패리티는 dipolar(반대칭)와 quadrupolar(대칭)로 구분됩니다.

### §3 Mechanisms of Magnetic Field Generation (pp. 15–18) / 자기장 생성 메커니즘

**English.** §3.1: poloidal $\to$ toroidal is unproblematic — a large-scale axisymmetric differential rotation $\Omega(r,\theta)$ shears poloidal field lines into toroidal, the $\Omega$-effect, with source term $r\sin\theta\,(\mathbf{B}_p\cdot\nabla\Omega)$ in the $B_\phi$ equation. §3.2: toroidal $\to$ poloidal is the hard direction, since Cowling's theorem forbids axisymmetric solutions. Four candidate mechanisms are surveyed:
- §3.2.1 *Mean-field $\alpha$-effect* — turbulent helicity produces $\mathcal{E} = \alpha\langle\mathbf{B}\rangle - \eta_T\nabla\times\langle\mathbf{B}\rangle$.
- §3.2.2 *Hydrodynamical shear instabilities* — non-axisymmetric instabilities of the latitudinal differential rotation in the tachocline produce $\alpha$-like effects.
- §3.2.3 *MHD instabilities* — magnetic shear instabilities (Tayler, Pitts–Tayler) in stratified rotating layers.
- §3.2.4 *Babcock–Leighton* — buoyant rise + Coriolis-induced tilt of bipolar magnetic regions, followed by surface diffusion and meridional advection of trailing-polarity flux to the poles, regenerates the global poloidal field.

**한국어.** §3.1: 폴로이달 → 토로이달은 어렵지 않습니다. 대규모 축대칭 미분 회전 $\Omega(r,\theta)$이 폴로이달 자기장을 토로이달로 전단합니다($\Omega$ 효과, 원천항 $r\sin\theta\,(\mathbf{B}_p\cdot\nabla\Omega)$). §3.2: 토로이달 → 폴로이달은 Cowling 정리 때문에 어려운 방향이며, 네 가지 후보 메커니즘이 검토됩니다 — 평균장 $\alpha$-효과, 유체역학적 전단 불안정, MHD 불안정, Babcock–Leighton 메커니즘. 후자는 부력 상승 + Coriolis 비틀림으로 기울어진 쌍극 자기 영역이 표면 확산 및 자오선 이류로 후행 극성 자속을 극으로 운반하여 전역 폴로이달장을 재생합니다.

### §4 A Selection of Representative Models (pp. 19–50) / 대표적 모델 선집

**English.** §4.1 fixes the modeling ingredients: $\Omega(r,\theta)$ from helioseismology (a near-rigid radiative interior, latitudinal differential rotation through the convection zone, joined by the $\sim 0.05R_\odot$ tachocline at $r/R_\odot\approx 0.7$); $\alpha$-effect prescription (typically $\alpha(r,\theta) = \alpha_0\,f(r)\,\cos\theta$); turbulent diffusivity profile $\eta_T(r)$; meridional circulation (single cell per hemisphere, surface poleward $\sim$ 20 m/s, return flow $\sim$ 1–2 m/s near base of CZ).

§4.2 *$\alpha\Omega$ mean-field models*. The dimensionless system, with $R_\odot$ as length unit and $\tau = R_\odot^2/\eta_T$ as time unit, becomes
$$\frac{\partial A}{\partial t} = \left(\nabla^2 - \frac{1}{\varpi^2}\right)A + C_\alpha\,\alpha B,$$
$$\frac{\partial B}{\partial t} = \left(\nabla^2 - \frac{1}{\varpi^2}\right)B + C_\Omega\,\varpi(\mathbf{B}_p\cdot\nabla\Omega) + \frac{1}{\varpi}\frac{d\eta}{dr}\frac{\partial(\varpi B)}{\partial r},$$
with $\varpi = r\sin\theta$, dimensionless numbers $C_\alpha = \alpha_0 R_\odot/\eta_T$, $C_\Omega = \Omega_0 R_\odot^2/\eta_T$, and dynamo number $D = C_\alpha C_\Omega$. Linear analysis yields oscillatory dynamo waves whose dispersion in the local Cartesian limit is $\sigma = \pm(1+i)\sqrt{D/2}/\sqrt{2}$ (roughly), so growth rate $\Re(\sigma) > 0$ requires $|D| > D_c$. The Parker–Yoshimura sign rule predicts equatorward propagation when $\alpha\,\partial\Omega/\partial r < 0$ in the northern hemisphere. §4.2.6's representative results: with helioseismic $\Omega(r,\theta)$ and $\alpha\propto\cos\theta$, the dominant cycle period and the latitude of the equatorward branch generally do *not* match observations — the $\alpha\Omega$ mean-field paradigm faces the "sign rule problem".

§4.3 *Interface dynamos* (Parker 1993): place the $\Omega$-effect in a high-conductivity layer (tachocline) and the $\alpha$-effect in a low-conductivity layer (convection zone) separated by a thin interface, which limits the field strength reaching the $\alpha$-region and thereby alleviates $\alpha$-quenching. Representative solutions reproduce equatorward butterflies with realistic peak field strengths of 1–10 kG in the $\Omega$-region.

§4.4 *Mean-field models with meridional circulation*. Adding the advective terms $\mathbf{u}_p\cdot\nabla A$ (and similarly for $B/\varpi$) to the kinematic equations changes the mode structure dramatically. When the magnetic Reynolds number $R_m = u_0 R_\odot/\eta_T \gtrsim 10^3$, the cycle period is set by the circulation turnover time $\sim L/u_0 \sim 11$ yr rather than by the diffusion time. The butterfly diagram becomes equatorward irrespective of the Parker–Yoshimura sign rule because the equatorward return flow at the base of the CZ advects toroidal flux equatorward.

§4.5 *Models based on shear instabilities*; §4.6 *Buoyant instabilities of sheared layers*; §4.7 *Models based on flux-tube instabilities* — each provides an alternative non-axisymmetric source replacing $\alpha$.

§4.8 *Babcock–Leighton models*. The poloidal-source term is non-local: a surface source $S(r,\theta;B(r_c,\theta,t))$ proportional to the toroidal field at the base of the CZ, with a delay set by the buoyant rise time. Coupling this to the $\Omega$-effect at the tachocline via meridional circulation gives a *flux-transport dynamo* whose period is again $\sim L/u_0$. These models naturally reproduce: (i) equatorward butterfly drift, (ii) polar field reversal at sunspot maximum, (iii) $\sim 90°$ phase lag between toroidal (sunspot) and poloidal (polar) cycles, (iv) hemispheric coupling through the equatorial diffusion. §4.9 surveys global MHD simulations (ASH, EULAG, etc.) which by 2010 had begun to produce cycling large-scale magnetic fields self-consistently.

**한국어.** §4.1은 모델링 재료를 고정합니다: 일진동학에서 얻은 $\Omega(r,\theta)$ (거의 강체에 가까운 복사층 내부, 대류층 전체에 걸친 위도 미분 회전, $r/R_\odot\approx 0.7$의 tachocline으로 연결); $\alpha$-효과 처방($\alpha = \alpha_0 f(r)\cos\theta$ 형태); 난류 확산도 $\eta_T(r)$; 자오선 순환(반구당 단일 cell, 표면 극방향 $\sim$20 m/s, 대류층 하부 반환류 $\sim$1–2 m/s).

A canonical helioseismic fit used throughout §4 is
$$\Omega(r,\theta) = \Omega_C + \frac{1}{2}\!\left[1+\mathrm{erf}\!\left(\frac{r-r_c}{w}\right)\right]\!\left[\Omega_S(\theta) - \Omega_C\right],$$
with $\Omega_S(\theta) = \Omega_{\rm eq}(1 - a_2\cos^2\theta - a_4\cos^4\theta)$, $\Omega_C/2\pi = 432$ nHz, $\Omega_{\rm eq}/2\pi = 460$ nHz, $a_2 = 0.17$, $a_4 = 0.08$, $r_c/R_\odot = 0.7$, $w/R_\odot = 0.025$. This is the function plotted in nearly every figure of §4 and is the *single most influential observational input* into modern solar dynamo modeling.

§4의 여러 그림과 거의 모든 처방에서 사용하는 정준적 일진동학 적합은 위 식이며, 매개변수 $\Omega_C/2\pi = 432$ nHz, $\Omega_{\rm eq}/2\pi = 460$ nHz, $a_2 = 0.17$, $a_4 = 0.08$, $r_c/R_\odot = 0.7$, $w/R_\odot = 0.025$를 사용합니다. 이는 현대 태양 다이나모 모델링에서 *가장 영향력 있는 단일 관측 입력*입니다.

§4.2 $\alpha\Omega$ 평균장 모델. 무차원화된 시스템에서 다이나모 수 $D = C_\alpha C_\Omega$가 임계값 $D_c$를 초과하면 진동 다이나모파가 성장합니다. Parker–Yoshimura 부호 규칙은 $\alpha\,\partial\Omega/\partial r < 0$일 때 적도방향 전파를 예측하지만, 일진동학적 $\Omega(r,\theta)$과 $\alpha\propto\cos\theta$ 조합은 흔히 관측과 부합하지 않아 "부호 규칙 문제"가 발생합니다.

§4.3 Interface 다이나모(Parker 1993): $\Omega$-효과를 고전도도 층(tachocline)에, $\alpha$-효과를 저전도도 층(대류층)에 분리해 $\alpha$-quenching을 완화합니다. §4.4 자오선 순환 포함 평균장 모델: 자기 Reynolds 수 $R_m \gtrsim 10^3$일 때 주기가 순환 회전 시간 $\sim L/u_0 \sim$ 11년으로 결정되며, 부호 규칙과 무관하게 적도방향 버터플라이가 만들어집니다. §4.5–4.7은 전단·부력·자속관 불안정 기반 대안 메커니즘들. §4.8 Babcock–Leighton 모델은 표면 비국소 원천항으로 작동하며 자오선 순환을 통해 tachocline의 $\Omega$-효과와 결합된 *플럭스 수송 다이나모*를 형성합니다. 적도방향 버터플라이, 흑점 극대기의 극성 역전, 토로이달–폴로이달 사이 $\sim 90°$ 위상차, 적도 확산을 통한 반구 결합 등을 자연스럽게 재현합니다. §4.9는 2010년 시점의 전역 MHD 시뮬레이션을 정리합니다.

### §5 Amplitude Fluctuations, Multiperiodicity, and Grand Minima (pp. 51–69) / 진폭 변동, 다중주기성, Grand Minima

**English.** §5.1 reviews the observational evidence: cycle-to-cycle amplitude variations of $\sim 50\%$, the Gnevyshev–Ohl rule (alternating odd/even cycle amplitudes), and Grand Minima (Maunder 1645–1715, Spörer, Dalton). §5.2 considers fossil interior fields plus the 22-yr cycle. §5.3 examines *dynamical nonlinearity* — Lorentz-force back-reaction on $\Omega$ and meridional circulation, plus dynamical $\alpha$-quenching where $\alpha = \alpha_0/(1 + (B/B_{\rm eq})^2)$. §5.4 *time-delay dynamics*: in B–L models the buoyant-rise + circulation-transit delays create a low-dimensional iterative map generically prone to period-doubling and chaos. §5.5 *stochastic forcing*: scatter in active-region tilt (Joy's law scatter) produces multiplicative noise on the surface source. §5.6 surveys candidate intermittency mechanisms (stochastic, nonlinear, threshold, time-delay) for Maunder-type minima. §5.7 introduces dynamo-based forecasting: schemes using the polar field at minimum (Schatten–Sofia–style "precursor") versus polar field assimilation through B–L flux-transport models (Dikpati & Gilman 2006 vs. Choudhuri et al. 2007).

**한국어.** §5.1 관측적 증거: 주기 간 진폭 변동 $\sim 50\%$, Gnevyshev–Ohl 규칙(홀짝 주기 진폭 교대), Grand Minima(Maunder 1645–1715, Spörer, Dalton). §5.2 화석 내부 자기장과 22년 주기. §5.3 동역학적 비선형성: $\Omega$와 자오선 순환에 대한 Lorentz 반작용, 동역학적 $\alpha$-quenching $\alpha = \alpha_0/(1 + (B/B_{\rm eq})^2)$. §5.4 시간지연 동역학: B–L 모델에서 부력 상승 + 순환 전송 지연으로 저차원 반복 사상이 만들어지고 주기배가 및 혼돈으로 이어질 수 있습니다. §5.5 확률 강제: Joy 법칙 산포로 인한 표면 원천의 곱셈 잡음. §5.6 간헐성 메커니즘들. §5.7 다이나모 기반 예측(Dikpati–Gilman 2006 vs. Choudhuri et al. 2007).

### §6 Open Questions and Current Trends (pp. 70–73) / 미해결 문제와 현재 동향

**English.** §6 lists eight critical questions: (1) which is the primary poloidal regeneration mechanism — turbulent $\alpha$ or B–L? (2) what limits the field amplitude — buoyancy losses, $\alpha$-quenching, or back-reaction on flow? (3) flux tubes vs. diffuse fields? (4) how constraining is the butterfly diagram, given that it samples only the strongest tube-forming flux? (5) is meridional circulation truly essential, or merely sufficient? (6) is the mean field truly axisymmetric? (7) what causes Maunder-type minima? (8) where do we go next — global MHD simulations or improved mean-field reductions?

**한국어.** §6은 여덟 가지 핵심 질문을 제시합니다: (1) 폴로이달 재생의 주된 메커니즘은 무엇인가 — 난류 $\alpha$인가 B–L인가? (2) 자기장 진폭을 제한하는 것은 무엇인가 — 부력 손실, $\alpha$-quenching, 흐름에 대한 반작용? (3) 자속관 vs. 확산 자기장? (4) 버터플라이 다이어그램이 얼마나 제약적인가(가장 강한 자속만 추적)? (5) 자오선 순환은 정말로 필수인가, 단지 충분한가? (6) 평균장은 정말 축대칭인가? (7) Maunder형 minima의 원인? (8) 다음 단계 — 전역 MHD 시뮬레이션 vs. 개선된 평균장 환원?

### Worked Example: Estimating the Cycle Period in a Flux-Transport Dynamo / 풀이 예제: 플럭스 수송 다이나모 주기 추정

**English.** Take a representative flux-transport dynamo with: surface poleward flow $u_0 = 20$ m/s, return flow at base of CZ $u_{\rm ret} = 1.5$ m/s, surface circulation arc $L_{\rm surf} = \pi R_\odot/2 \approx 1.09\times 10^9$ m, return arc length similar. Surface transit time $\tau_s = L_{\rm surf}/u_0 \approx 5.4\times 10^7$ s $\approx$ 1.7 yr; deep return time $\tau_d \approx L_{\rm circ}/u_{\rm ret} \approx 7.3\times 10^8$ s $\approx 23$ yr. Half cycle $\sim \tau_d/2$ since circulation completes half a hemispheric loop per cycle, predicting $T \approx 11$–12 yr — matching observation. This sensitivity to $u_{\rm ret}$ is the basis for "circulation-speed" cycle-prediction schemes (Dikpati–Gilman 2006).

**한국어.** 대표적인 플럭스 수송 다이나모를 가정합니다: 표면 극방향 흐름 $u_0 = 20$ m/s, 대류층 하부 반환류 $u_{\rm ret} = 1.5$ m/s. 표면 통과 시간 $\tau_s \approx 1.7$년, 심부 반환 시간 $\tau_d \approx 23$년. 주기당 반구 루프의 절반을 통과하므로 $T \approx 11$–12년 — 관측과 일치합니다. 이 $u_{\rm ret}$ 의존성이 "순환 속도" 기반 주기 예측(Dikpati–Gilman 2006)의 기초입니다.

### Worked Example: Parker–Yoshimura Sign Rule in Practice / 풀이 예제: Parker–Yoshimura 부호 규칙 적용

**English.** In the northern hemisphere mid-latitude tachocline, helioseismology gives $\partial\Omega/\partial r > 0$ (positive radial gradient). Standard mean-field theory prescribes $\alpha > 0$ in the northern hemisphere ($\alpha\propto\cos\theta$). The product $\alpha\,\partial\Omega/\partial r > 0$ implies *poleward* phase velocity by Parker–Yoshimura — opposite to observed equatorward butterfly migration. This sign-rule failure is *the* observational refutation of the simplest $\alpha\Omega$ dynamo and motivates either: (a) negative $\alpha$ in the bulk CZ near the equator, (b) interface dynamos, or (c) flux-transport dynamos in which meridional advection overrides the wave propagation direction.

**한국어.** 북반구 중위도 tachocline에서 일진동학은 $\partial\Omega/\partial r > 0$을 줍니다. 표준 평균장 이론은 북반구에서 $\alpha > 0$ ($\alpha\propto\cos\theta$)을 부여합니다. $\alpha\,\partial\Omega/\partial r > 0$이면 Parker–Yoshimura에 의해 *극방향* 위상 속도가 예측되며, 이는 관측된 적도방향 버터플라이 이동과 정반대입니다. 이 부호 규칙 실패가 단순 $\alpha\Omega$ 다이나모의 *결정적* 관측적 반증이며, (a) 적도 부근 대류층 내부에서 음의 $\alpha$, (b) interface 다이나모, 또는 (c) 자오선 이류가 파동 전파 방향을 압도하는 플럭스 수송 다이나모를 동기 부여합니다.

### Quantitative Observational Constraints / 정량적 관측 제약

**English.** The review repeatedly emphasizes the following numerical benchmarks any model must reproduce:
- Cycle period: $11.0 \pm 1.5$ yr (mean), with cycle-to-cycle scatter from 9 to 14 yr.
- Hale period: 22 yr (full magnetic cycle including polarity reversal).
- Sunspot latitudinal range: emergence band $\sim \pm 30°$ at cycle start, narrowing to $\sim \pm 8°$ at cycle end (butterfly wings).
- Polar field strength: $\sim 10$ G at minimum, reversing within $\pm 1$ year of sunspot maximum.
- Total polar cap unsigned flux: $\sim 10^{22}$ Mx.
- Total active-region flux per cycle: a few $\times 10^{25}$ Mx (1000× polar flux), implying toroidal-dominated interior.
- Toroidal-poloidal phase lag: $\approx 90°$ (quarter cycle).
- Hemispheric asymmetry: typically $< 30\%$ in cycle-integrated activity.
- Maunder Minimum: $\sim 70$ yr near-absence of spots (1645–1715), recurrence rate $\sim$ 2–3 per millennium from $^{10}$Be records.

**한국어.** 리뷰는 다음의 수치적 기준을 모든 모델이 반드시 재현해야 할 것으로 반복 강조합니다:
- 주기: 평균 $11.0 \pm 1.5$년, 주기 간 산포 9–14년.
- Hale 주기: 22년(극성 역전 포함).
- 흑점 위도 범위: 주기 시작 시 $\pm 30°$, 종료 시 $\pm 8°$.
- 극성 자기장: 극소기 $\sim 10$ G, 흑점 극대기 $\pm 1$년 이내 역전.
- 극관 무부호 자속: $\sim 10^{22}$ Mx.
- 주기당 활동영역 자속: 수 $\times 10^{25}$ Mx(극관의 1000배) → 내부 토로이달 우세.
- 토로이달–폴로이달 위상차: $\approx 90°$.
- 반구 비대칭: 주기 적분량의 $< 30\%$.
- Maunder Minimum: 1645–1715의 $\sim 70$년 흑점 거의 부재. $^{10}$Be 기록상 천년에 2–3회.

## 3. Key Takeaways / 핵심 시사점

1. **Cowling's theorem forces non-axisymmetry. / Cowling 정리는 비축대칭을 강제한다.**
   - EN: Any successful axisymmetric dynamo model must include a parametrized $\alpha$-effect (or B–L source) that *encodes* the unresolved non-axisymmetric physics — there is no axisymmetric escape.
   - KR: 성공적인 축대칭 다이나모 모델은 미해상 비축대칭 물리를 *부호화*하는 매개변수화된 $\alpha$-효과(또는 B–L 원천)를 반드시 포함해야 하며, 축대칭만으로의 탈출구는 없습니다.

2. **The dynamo number $D = C_\alpha C_\Omega$ controls criticality. / 다이나모 수 $D = C_\alpha C_\Omega$가 임계성을 결정한다.**
   - EN: For $|D| > D_c$ (typically $D_c \sim 10^2$–$10^3$), oscillatory dynamo modes grow; below it the field decays — independent of detailed geometry.
   - KR: $|D| > D_c$ (보통 $10^2$–$10^3$)이면 진동 다이나모 모드가 성장하고, 미만이면 자기장은 감쇠합니다 — 상세 기하학과 무관합니다.

3. **Helioseismic $\Omega(r,\theta)$ broke classical $\alpha\Omega$ models. / 일진동학적 $\Omega(r,\theta)$가 고전 $\alpha\Omega$ 모델을 깨뜨렸다.**
   - EN: With observed $\partial\Omega/\partial r > 0$ in low/mid-latitude tachocline, the Parker–Yoshimura sign rule predicts *poleward* migration unless $\alpha < 0$ in the northern hemisphere — contrary to most prescriptions, motivating B–L revival and meridional-circulation–dominated dynamos.
   - KR: 관측된 저·중위도 tachocline의 $\partial\Omega/\partial r > 0$에서 Parker–Yoshimura 부호 규칙은 북반구에서 $\alpha < 0$이 아닌 한 *극방향* 이동을 예측합니다. 이는 대부분의 처방과 모순되어 B–L 부흥과 자오선 순환 지배 다이나모로 이어졌습니다.

4. **Meridional circulation as conveyor belt sets the cycle period. / 자오선 순환이 컨베이어 벨트로 주기를 결정한다.**
   - EN: When advection dominates diffusion ($R_m \gtrsim 10^3$), the cycle period is $T \sim L_{\rm circ}/u_0 \sim 11$ yr, providing a natural quantitative match to observation.
   - KR: 이류가 확산을 압도할 때($R_m \gtrsim 10^3$) 주기는 $T \sim L_{\rm circ}/u_0 \sim 11$년으로 관측과 정량적으로 일치합니다.

5. **Babcock–Leighton localizes the source where it is observable. / B–L은 원천을 관측 가능한 곳에 둔다.**
   - EN: Surface flux transport from tilted bipolar regions is *directly observed*, making the B–L source term the most empirically constrained part of any solar dynamo model.
   - KR: 기울어진 쌍극 영역으로부터의 표면 자속 수송은 *직접 관측*되며, 따라서 B–L 원천항은 태양 다이나모 모델에서 가장 경험적으로 잘 제약된 구성요소입니다.

6. **Time delays generate intermittency. / 시간지연이 간헐성을 만든다.**
   - EN: B–L dynamos with explicit buoyant-rise and meridional-transit delays reduce to nonlinear iterative maps, producing chaos and Maunder-like quiescent epochs without invoking external forcing.
   - KR: 부력 상승과 자오선 전송의 명시적 지연을 갖는 B–L 다이나모는 비선형 반복 사상으로 환원되며, 외부 강제 없이 혼돈과 Maunder 유사 정적 시기를 생성합니다.

7. **Stochastic Joy-law scatter is unavoidable noise. / Joy 법칙 산포는 피할 수 없는 잡음이다.**
   - EN: Active-region tilt scatter ($\sim 30°$) makes the surface poloidal source *intrinsically* stochastic; this multiplicative noise alone can drive grand minima for plausible parameters.
   - KR: 활동 영역 기울기 산포($\sim 30°$)는 표면 폴로이달 원천을 *본질적으로* 확률적으로 만듭니다. 이 곱셈 잡음만으로도 합리적 파라미터에서 grand minima를 만들 수 있습니다.

8. **No consensus model — but the framework is converging. / 합의 모델은 없으나 틀은 수렴 중이다.**
   - EN: As of 2010 there is no single accepted solar dynamo model, but virtually all credible models share: tachocline $\Omega$-effect, surface or bulk-CZ poloidal source, meridional circulation, and turbulent diffusion — a "common backbone" the review crystallizes.
   - KR: 2010년 시점에 단일한 합의 모델은 없지만, 거의 모든 신뢰할 만한 모델은 tachocline의 $\Omega$-효과, 표면 또는 대류층 내부의 폴로이달 원천, 자오선 순환, 난류 확산이라는 "공통 척추"를 공유합니다. 이 리뷰가 이를 결정화합니다.

## 4. Mathematical Summary / 수학적 요약

### 4.1 MHD Induction Equation / MHD 유도 방정식
$$\frac{\partial \mathbf{B}}{\partial t} = \nabla\times(\mathbf{u}\times\mathbf{B}) - \nabla\times(\eta\,\nabla\times\mathbf{B}).$$
- $\mathbf{B}$: magnetic field (G). / 자기장.
- $\mathbf{u}$: flow velocity. / 흐름 속도.
- $\eta = (\mu_0\sigma)^{-1}$: magnetic diffusivity. / 자기 확산도.

### 4.2 Axisymmetric Poloidal–Toroidal Decomposition / 축대칭 폴로이달–토로이달 분해
$$\mathbf{B}(r,\theta,t) = \nabla\times\!\left[A(r,\theta,t)\hat{e}_\phi\right] + B(r,\theta,t)\,\hat{e}_\phi.$$
Components: $B_r = (r\sin\theta)^{-1}\partial_\theta(\sin\theta\,A)$, $B_\theta = -r^{-1}\partial_r(rA)$. The poloidal flux function $\Psi = \varpi A$ with $\varpi = r\sin\theta$.

### 4.3 Mean-Field $\alpha\Omega$ Equations / 평균장 $\alpha\Omega$ 방정식
$$\frac{\partial A}{\partial t} + \frac{1}{\varpi}(\mathbf{u}_p\cdot\nabla)(\varpi A) = \eta_T\!\left(\nabla^2 - \frac{1}{\varpi^2}\right)\!A + \alpha\,B,$$
$$\frac{\partial B}{\partial t} + \varpi\,\mathbf{u}_p\cdot\nabla\!\left(\frac{B}{\varpi}\right) = \eta_T\!\left(\nabla^2 - \frac{1}{\varpi^2}\right)\!B + \varpi(\mathbf{B}_p\cdot\nabla\Omega) + \frac{1}{\varpi}\frac{d\eta_T}{dr}\frac{\partial(\varpi B)}{\partial r}.$$
Each term explained:
- $\partial_t$: time evolution. / 시간 진화.
- $\mathbf{u}_p\cdot\nabla$: meridional advection. / 자오선 이류.
- $\eta_T(\nabla^2 - \varpi^{-2})$: turbulent diffusion. / 난류 확산.
- $\alpha B$: poloidal regeneration ($\alpha$-effect). / 폴로이달 재생.
- $\varpi(\mathbf{B}_p\cdot\nabla\Omega)$: $\Omega$-effect (toroidal source). / $\Omega$ 효과.

### 4.4 Dimensionless Numbers / 무차원 수
- $C_\alpha \equiv \alpha_0 R_\odot / \eta_T$.
- $C_\Omega \equiv \Omega_0 R_\odot^2 / \eta_T$.
- Magnetic Reynolds: $R_m \equiv u_0 R_\odot / \eta_T$.
- Dynamo number: $D \equiv C_\alpha C_\Omega$. Critical $D_c \sim 10^2$–$10^3$ for typical geometries.

### 4.5 Parker Dynamo Wave / Parker 다이나모 파동
For local plane-wave $A,B \propto \exp(ikx + iky y - i\omega t)$ with constant $\alpha$ and uniform shear $G = \partial\Omega/\partial r$:
$$(i\omega + \eta_T k^2)^2 = i\,\alpha G\,k_y\!\implies \omega = \pm\sqrt{|\alpha G k_y|/2}\,(1\pm i) - i\eta_T k^2.$$
- Real part = oscillation frequency (cycle period $T = 2\pi/\Re(\omega)$).
- Imaginary part: growth if $\Im(\omega) > 0$, i.e., $|\alpha G k_y|/2 > (\eta_T k^2)^2$.
- Phase velocity sign: equatorward when $\alpha\,G < 0$ in northern hemisphere (Parker–Yoshimura).

### 4.6 Babcock–Leighton Source Term / Babcock–Leighton 원천항
Surface, non-local in radius:
$$S_{\rm BL}(r,\theta,t) = \frac{\alpha_0}{2}\!\left[1+\mathrm{erf}\!\left(\frac{r-r_2}{d_2}\right)\right]\!\left[1-\mathrm{erf}\!\left(\frac{r-1}{d_1}\right)\right]\cos\theta\sin\theta\,f(B(r_c,\theta,t-\tau)),$$
with $f(B) = B/[1+(B/B_*)^2]$ saturating at $B \gtrsim B_*$ and a delay $\tau$ for buoyant rise.

### 4.7 Cycle Period in Flux-Transport Limit / 플럭스 수송 한계의 주기
$$T \approx \frac{L_{\rm circ}}{u_0} \approx \frac{2(R_\odot - r_c)}{u_0} \sim 10\text{–}15\,\text{yr},$$
for $u_0 \sim 1$–2 m/s at base of CZ — provides direct observational match.

### 4.8 Dynamical $\alpha$-Quenching / 동역학적 $\alpha$-quenching
$$\alpha(B) = \frac{\alpha_0}{1 + (B/B_{\rm eq})^2}, \quad B_{\rm eq} \approx \sqrt{4\pi\rho}\,u_{\rm rms} \sim 10^4\,\text{G at base of CZ}.$$
- $\alpha_0$: kinematic (linear) $\alpha$. / 운동학적 $\alpha$.
- $B_{\rm eq}$: equipartition field at which Lorentz force balances turbulent Reynolds stress. / Lorentz 힘이 난류 응력과 균형을 이루는 등분배 자기장.
- For $B \gg B_{\rm eq}$, the $\alpha$-effect saturates → nonlinear amplitude limitation. / 큰 $B$에서 $\alpha$가 포화 → 비선형 진폭 제한.

### 4.9 Time-Delay Iterative Map (B–L) / 시간지연 반복 사상 (B–L)
For Babcock–Leighton with explicit delays $\tau_1$ (rise) and $\tau_2$ (transit):
$$P_{n+1} = f(T_n) = a\,T_n / [1 + (T_n/T_*)^2], \quad T_n = b\,P_{n-1},$$
where $P_n$ is poloidal amplitude at cycle $n$ and $T_n$ toroidal amplitude. This iterated quadratic map exhibits period doubling for $a$ in a critical range and chaos beyond — explaining cycle-amplitude irregularity from purely deterministic dynamics.

## 5. Paper in the Arc of History / 역사 속의 논문

```
1843 ─ Schwabe: 11-year sunspot cycle discovered
1908 ─ Hale: sunspots are magnetic
1919 ─ Larmor: inductive flow conjecture
1933 ─ Cowling: antidynamo theorem (axisymmetric flows fail)
1955 ─ Parker: cyclonic-twist α-effect
1961 ─ Babcock: surface flux-transport mechanism
1966 ─ Steenbeck-Krause-Rädler: mean-field electrodynamics
1969 ─ Leighton: phenomenological B-L kinematic dynamo
1980s ─ Helioseismology reveals Ω(r,θ): four-way crisis
1991 ─ Choudhuri-Schüssler-Dikpati: meridional circulation revives B-L
1993 ─ Parker: interface dynamo
1995 ─ Charbonneau-MacGregor: tachocline α-effect
1999 ─ Dikpati-Charbonneau: flux-transport B-L dynamo
2005 ─ Charbonneau LRSP review v1
2006 ─ Dikpati-Gilman: cycle prediction (controversial)
2010 ─ ★ THIS REVIEW (v2: added pumping, MHD sims, predictions)
2014+ ─ EULAG, ASH, MHD simulations producing cyclic dynamos
```

**English.** This review sits at the consolidation point: helioseismology had revealed the rotation profile, Babcock–Leighton flux-transport models had matured into testable predictors, and the first large-scale MHD simulations were beginning to produce solar-like cycling — but no consensus existed. Charbonneau's role is to catalogue, compare, and critique without prematurely closing debates.

**한국어.** 이 리뷰는 통합의 시점에 위치합니다: 일진동학이 회전 프로파일을 밝혔고, Babcock–Leighton 플럭스 수송 모델은 검증 가능한 예측자로 성숙했으며, 최초의 대규모 MHD 시뮬레이션이 태양형 주기를 생성하기 시작했지만 합의는 없었습니다. Charbonneau의 역할은 논쟁을 성급히 닫지 않고 정리·비교·비판하는 것이었습니다.

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 |
|---|---|
| Hale (1908) — sunspot magnetism / 흑점 자기장 | Established the observational target (polarity laws) the dynamo must reproduce. / 다이나모가 재현해야 할 관측 대상 확립. |
| Cowling (1933) — antidynamo theorem / 반다이나모 정리 | Defines the *constraint* every model must circumvent. / 모든 모델이 우회해야 할 *제약* 정의. |
| Parker (1955) — cyclonic α-effect / 알파 효과 | Provides the foundational mechanism for §3.2.1 and §4.2. / §3.2.1, §4.2의 기초 메커니즘. |
| Babcock (1961) — surface flux transport / 표면 자속 수송 | Direct ancestor of §3.2.4 and §4.8. / §3.2.4, §4.8의 직접적 조상. |
| Steenbeck–Krause–Rädler (1966) | Quantitative mean-field electrodynamics underlying §4.2. / §4.2의 기초 수식. |
| Leighton (1969) — kinematic B–L | First explicit B–L model — reviewed in §4.8. / §4.8에서 검토되는 최초의 명시적 B–L 모델. |
| Spiegel–Zahn (1992) — tachocline | Defines the layer where §4.3 interface dynamo operates. / §4.3 interface 다이나모의 작동 층. |
| Dikpati–Charbonneau (1999) | Standard flux-transport B–L dynamo recipe. / 표준 플럭스 수송 B–L 다이나모 처방. |
| Dikpati–Gilman (2006) — cycle 24 forecast | Centerpiece of §5.7 prediction discussion. / §5.7 예측 논의의 중심. |
| Choudhuri et al. (2007) — alternative forecast | Counter-predicting weak cycle 24 — featured in §5.7. / 약한 주기 24 예측, §5.7. |

## 7. References / 참고문헌

- Charbonneau, P., "Dynamo Models of the Solar Cycle", *Living Reviews in Solar Physics*, 7, 3 (2010). [DOI: 10.12942/lrsp-2010-3]
- Parker, E. N., "Hydromagnetic Dynamo Models", *ApJ*, 122, 293 (1955).
- Babcock, H. W., "The Topology of the Sun's Magnetic Field and the 22-Year Cycle", *ApJ*, 133, 572 (1961).
- Leighton, R. B., "A Magneto-Kinematic Model of the Solar Cycle", *ApJ*, 156, 1 (1969).
- Steenbeck, M., Krause, F., Rädler, K.-H., "Berechnung der mittleren Lorentz-Feldstärke...", *Z. Naturforsch.*, 21a, 369 (1966).
- Cowling, T. G., "The magnetic field of sunspots", *MNRAS*, 94, 39 (1933).
- Dikpati, M., Charbonneau, P., "A Babcock–Leighton Flux Transport Dynamo with Solar-like Differential Rotation", *ApJ*, 518, 508 (1999).
- Dikpati, M., Gilman, P. A., "Simulating and Predicting Solar Cycles Using a Flux-Transport Dynamo", *ApJ*, 649, 498 (2006).
- Choudhuri, A. R., Chatterjee, P., Jiang, J., "Predicting Solar Cycle 24 With a Solar Dynamo Model", *Phys. Rev. Lett.*, 98, 131103 (2007).
- Yoshimura, H., "Solar-Cycle Dynamo Wave Propagation", *ApJ*, 201, 740 (1975).
- Hoyng, P., "The Field, the Mean, and the Meaning", in *Advances in Nonlinear Dynamos*, Eds. Ferriz-Mas & Núñez (2003).
- Ossendrijver, M., "The Solar Dynamo", *A&AR*, 11, 287 (2003).
- Käpylä, P. J., Korpi, M. J., Tuominen, I., "Solar dynamo simulations with mean-field $\alpha$ effect and turbulent pumping", *Astron. Nachr.*, 327, 884 (2006).
- Guerrero, G., de Gouveia Dal Pino, E. M., "Turbulent magnetic pumping in a Babcock–Leighton solar dynamo model", *A&A*, 485, 267 (2008).
- Petrovay, K., Szakály, G., "Transport effects in the evolution of the global solar magnetic field", *Solar Phys.*, 187, 1 (1999).
- Fan, Y., "Magnetic Fields in the Solar Convection Zone", *Living Rev. Solar Phys.*, 6, 4 (2009).

## 8. Appendix: Glossary of Operating Modes / 부록: 작동 양식 용어집

**English.** Mean-field dynamos can be classified by where each ingredient is physically located:

| Mode / 양식 | $\Omega$-effect location / 위치 | $\alpha$ / poloidal source | Period setter / 주기 결정 |
|---|---|---|---|
| Classical $\alpha\Omega$ / 고전 $\alpha\Omega$ | bulk CZ / 대류층 전체 | bulk $\alpha$ / 대류층 $\alpha$ | $\eta_T$ diffusion / 확산 |
| Interface / 인터페이스 | tachocline / 타코클라인 | thin overlying CZ / CZ 하부 얇은 층 | $\eta_T$ + interface / 확산 + 계면 |
| Flux-transport (B–L) / 플럭스 수송 | tachocline / 타코클라인 | surface (B–L) / 표면 | meridional circulation / 자오선 순환 |
| Distributed / 분산 | bulk CZ / 대류층 | bulk $\alpha$ / 대류층 $\alpha$ | mixed / 혼합 |
| Global MHD sim / 전역 MHD | self-consistent / 자기 일관 | self-consistent / 자기 일관 | emergent / 발현적 |

**한국어.** 평균장 다이나모는 각 구성요소의 물리적 위치에 따라 위와 같이 분류됩니다. 표에 정리한 다섯 양식이 §4의 핵심 비교 축입니다.

## 9. Appendix: Key Numerical Recipes / 부록: 핵심 수치 처방

**English.** A typical kinematic axisymmetric dynamo simulation requires:
1. Discretize $(r,\theta)$ on a 2D grid, e.g., $128\times 128$ (Chebyshev–Legendre or finite-difference).
2. Prescribe $\Omega(r,\theta)$ from helioseismic fits, e.g., $\Omega(r,\theta) = \Omega_c + \frac{1}{2}[1+\mathrm{erf}((r-r_c)/d)]\,(\Omega_S(\theta) - \Omega_c)$ with $\Omega_S(\theta) = \Omega_{\rm eq} - a_2\cos^2\theta - a_4\cos^4\theta$.
3. Prescribe $\alpha(r,\theta)$ profile concentrated near the surface (B–L) or in the CZ ($\alpha\Omega$).
4. Prescribe $\eta_T(r)$ — typically 1–2 orders larger in CZ than tachocline.
5. Prescribe $\mathbf{u}_p$ (single-cell stream function, mass-conserving).
6. Step the $A$ and $B$ PDEs with implicit-explicit (IMEX) time integration.
7. Diagnostics: cycle period, parity, butterfly diagram, polar-field reversal.

**한국어.** 전형적인 운동학적 축대칭 다이나모 시뮬레이션은 다음을 요구합니다: (1) $(r,\theta)$ 2D 격자, (2) 일진동학 적합으로부터 $\Omega(r,\theta)$ 처방, (3) $\alpha(r,\theta)$ 프로파일(B–L의 경우 표면 집중, $\alpha\Omega$의 경우 대류층 내부), (4) $\eta_T(r)$ 처방(대류층이 tachocline보다 1–2 자릿수 큼), (5) 질량 보존 자오선 순환 흐름 함수, (6) IMEX 시간 적분, (7) 진단: 주기, 패리티, 버터플라이, 극성 역전.

## 9b. ASCII Sketch: The Conveyor-Belt Flux-Transport Dynamo / ASCII 도해: 컨베이어 벨트 플럭스 수송 다이나모

```
         Surface (r = R_sun)
   ┌──────────────────────────────────────────┐
   │ ──→ ──→ ──→ poleward flow (20 m/s) ──→  │  ← B-L source
   │ ↑                                     ↓ │     (tilted BMRs decay)
   │ ↑                                     ↓ │
   │ ↑       Convection zone               ↓ │
   │ ↑                                     ↓ │
   │ ↑                                     ↓ │
   │ ←── ←── equatorward return (1.5 m/s) ←─ │  ← Ω-effect zone
   ├──────────────────────────────────────────┤  (tachocline, r ≈ 0.7 R)
   │           Radiative interior (rigid)     │
   └──────────────────────────────────────────┘
   Equator                                  Pole
```

**English.** The conveyor belt model: poleward surface flow advects newly-generated poloidal field (from B-L decay of bipolar regions) toward the poles; the equatorward return flow at the base of the CZ then advects toroidal flux (sheared from the descended poloidal field by the tachocline $\Omega$-effect) toward the equator, producing the equatorward butterfly.

**한국어.** 컨베이어 벨트 모델: 표면 극방향 흐름이 새로 생성된 폴로이달 자기장(쌍극영역 붕괴에서 생성)을 극으로 운반하고, 대류층 하부 적도방향 반환류는 tachocline의 $\Omega$-효과로 전단된 토로이달 자속을 적도방향으로 운반하여 적도방향 버터플라이를 만듭니다.

## 10. Self-Test Questions / 자가 진단 질문

**English.**
1. Why does Cowling's theorem not preclude solar dynamo action?
2. Derive the Parker dynamo wave dispersion relation in the local Cartesian limit.
3. Estimate the dynamo number for the Sun using $\alpha_0\sim 1$ m/s, $\Omega_0 \sim 3\times 10^{-6}$ rad/s, $\eta_T \sim 10^8$ m²/s.
4. In a flux-transport dynamo, what determines the cycle period to leading order? Give a numerical estimate.
5. What is the "sign rule problem" of mean-field $\alpha\Omega$ models, and how does the B–L mechanism evade it?
6. List four candidate mechanisms for Maunder-type Grand Minima.
7. Why does the toroidal-poloidal phase lag $\approx 90°$?

**한국어.**
1. Cowling 정리가 태양 다이나모 작용을 왜 배제하지 않는가?
2. 국소 직교 한계에서 Parker 다이나모 파동의 분산 관계를 유도하라.
3. $\alpha_0\sim 1$ m/s, $\Omega_0 \sim 3\times 10^{-6}$ rad/s, $\eta_T \sim 10^8$ m²/s로 태양의 다이나모 수를 추정하라.
4. 플럭스 수송 다이나모에서 주기를 1차로 결정하는 것은 무엇인가? 수치 추정값을 제시하라.
5. 평균장 $\alpha\Omega$ 모델의 "부호 규칙 문제"란 무엇이며, B–L 메커니즘이 어떻게 회피하는가?
6. Maunder형 Grand Minima의 후보 메커니즘 4가지를 열거하라.
7. 토로이달–폴로이달 위상차가 $\approx 90°$인 이유는?

**Answers (sketch) / 답안 (개요).**
1. EN: Cowling forbids axisymmetric solutions of the *full* induction equation; mean-field theory averages over non-axisymmetric turbulence to introduce an $\alpha$-source that *encodes* this physics. KR: Cowling은 *완전* 유도 방정식의 축대칭 해를 금지하지만, 평균장 이론은 비축대칭 난류를 평균하여 그 물리를 *부호화*하는 $\alpha$-원천을 도입합니다.
2. EN: Plug $A,B \propto e^{i(kx-\omega t)}$ into linearized $\alpha\Omega$ system → $(i\omega + \eta_T k^2)^2 = i\alpha G k_y$ → $\omega = \pm(1\pm i)\sqrt{|\alpha G k_y|/2} - i\eta_T k^2$. KR: 선형화된 $\alpha\Omega$에 평면파를 대입하여 위와 같이 풀어집니다.
3. EN: $C_\alpha = \alpha_0 R_\odot/\eta_T \approx 7$, $C_\Omega = \Omega_0 R_\odot^2/\eta_T \approx 1.5\times 10^4$, $D \approx 10^5 \gg D_c$. KR: 위 값을 대입.
4. EN: Meridional circulation: $T \sim 2(R_\odot - r_c)/u_{\rm ret} \sim 11$ yr for $u_{\rm ret}\sim 1.5$ m/s. KR: 자오선 순환: 위와 같이.
5. EN: Helioseismic $\partial\Omega/\partial r > 0$ + standard $\alpha > 0$ in N hemisphere → Parker–Yoshimura predicts poleward, contrary to observation. B–L sources poloidal field at surface independently of the wave-propagation rule. KR: 일진동학적 $\partial\Omega/\partial r > 0$ + 북반구 $\alpha > 0$ → 극방향 예측 (관측과 반대). B–L은 표면에서 폴로이달을 생성하므로 부호 규칙 무관.
6. EN: Stochastic noise, deterministic nonlinearity, threshold/parity modulation, time-delay chaos. KR: 확률적 잡음, 결정론적 비선형성, 임계/패리티 변조, 시간지연 혼돈.
7. EN: Toroidal field is the *integral in time* of poloidal shearing — $\partial_t B \propto B_p$ — yielding a Hilbert-transform-like 90° phase lag. KR: 토로이달은 폴로이달 전단의 *시간 적분*이므로 ($\partial_t B \propto B_p$) 90° 위상 지연이 발생합니다.
