---
title: "Flux Emergence (Theory)"
authors: [Mark C. M. Cheung, Hiroaki Isobe]
year: 2014
journal: "Living Reviews in Solar Physics"
doi: "10.12942/lrsp-2014-3"
topic: Living_Reviews_in_Solar_Physics
tags: [flux-emergence, magnetic-buoyancy, MHD, active-regions, sunspots, magnetoconvection, reconnection]
status: completed
date_started: 2026-04-23
date_completed: 2026-04-23
---

# 38. Flux Emergence (Theory) / 플럭스 출현 (이론)

---

## 1. Core Contribution / 핵심 기여

Cheung and Isobe (2014) deliver the definitive theoretical synthesis of how magnetic flux rises from the base of the convection zone, crosses the strongly stratified photosphere, and emerges into the corona. The review is organized around physical mechanisms rather than models: magnetic buoyancy provides the fundamental driver; convective stratification (density contrast ~10^6 from base of CZ to surface; ~10^8 more to corona) shapes the rise; magnetic buoyancy instabilities (Parker/undular and interchange) launch flux through the subadiabatic photospheric barrier; magnetoconvection corrugates fields into sea-serpent patterns and drains mass; magnetic twist provides coherence and the helicity that ultimately powers eruptions; reconnection and partial ionization govern coupling to the chromosphere and corona. The authors catalogue MHD simulation families (thin flux tube, anelastic, fully-compressible isothermal, fully-compressible radiative) with their limitations, and trace the arc of work from Parker (1955) to the 2010-era radiative MHD active-region-formation experiments (Cheung et al. 2010, Stein & Nordlund 2012).

Cheung과 Isobe (2014)는 자속이 대류층 바닥에서 솟아올라 강하게 층화된 광구를 뚫고 코로나로 출현하는 과정을 이론적으로 종합한 결정판 리뷰를 제시한다. 이 논문은 모델이 아닌 물리 메커니즘을 중심으로 구성된다. 자기 부력이 근본적 추진력이고, 대류 층화(대류층 바닥에서 표면까지 밀도 대비 ~10^6, 코로나까지 추가로 ~10^8)가 상승을 형성하며, 자기 부력 불안정성(Parker/물결, 교환)이 아디아바틱 광구 장벽을 뚫는 통로를 열고, 자기대류가 자기장을 바다뱀 모양으로 주름잡고 질량을 배수하며, 자기 뒤틀림이 응집과 나선도를 제공해 분출 에너지원이 되고, 재결합과 부분 전리가 채층·코로나 결합을 지배한다. 저자들은 MHD 시뮬레이션 계열(가는 자속관, anelastic, 완전 압축성 등온, 완전 압축성 복사)과 각각의 한계를 분류하고, Parker(1955)부터 2010년대 복사 MHD 활동 영역 형성 실험(Cheung et al. 2010, Stein & Nordlund 2012)까지 연구의 궤적을 추적한다.

A defining feature is the review's insistence that flux emergence is not a single problem but a family of problems connected by shared physics. The deep rise through the CZ (~10–30 days), the near-surface horizontal flattening (predicted by Spruit et al. 1987 and confirmed in 2010-era simulations), the photospheric breakthrough via magnetic buoyancy instability, and the coronal interaction with pre-existing field — each has its own dominant physical regime and its own class of numerical techniques. The review makes this structure explicit and surveys how observations from SDO/HMI and Hinode constrain each phase.

이 리뷰의 특징적인 주장은 플럭스 출현이 하나의 문제가 아니라 공통 물리로 연결된 여러 문제의 집합이라는 점이다. 대류층 깊은 곳을 통한 상승(~10–30일), 표면 근처의 수평 납작화(Spruit et al. 1987이 예측하고 2010년대 시뮬레이션이 확인), 자기 부력 불안정성을 통한 광구 돌파, 기존 자기장과의 코로나 상호작용 — 각 단계는 고유한 지배 물리와 고유한 수치 기법 부류를 가진다. 본 리뷰는 이 구조를 명시적으로 드러내고 SDO/HMI와 Hinode 관측이 각 단계를 어떻게 제약하는지 조사한다.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction and MHD Framework (§1–2) / 서론과 MHD 틀

The introduction motivates flux emergence through three questions: (1) how is magnetic flux transported from interior to atmosphere; (2) how does it energize or drain the corona; (3) what is the subsurface structure. The authors highlight coherent-bundle-vs-elementary-unit controversy (Zwaan 1978, 1985), the role of convective flows (Fan et al. 2003; Cheung et al. 2007a), and whether twist is necessary (Moreno-Insertis & Emonet 1996; Murray et al. 2006).

서론은 세 가지 질문으로 플럭스 출현을 동기 부여한다. (1) 자속이 어떻게 내부에서 대기로 이동하는가, (2) 어떻게 코로나를 에너지화하거나 탈에너지화하는가, (3) 지하 구조는 무엇인가. 저자들은 응집 다발 대 기본 단위 논쟁(Zwaan 1978, 1985), 대류 흐름의 역할, 뒤틀림의 필요성 등을 조명한다.

The MHD framework presented in Lagrangian form includes:

- **Mass conservation**: $D\rho/Dt + \rho \nabla\!\cdot\!\mathbf{v} = 0$ (Eq. 1)
- **Momentum**: $\rho D\mathbf{v}/Dt = \nabla\!\cdot\!\sigma + \mathbf{f}$ (Eq. 2) with stress tensor $\sigma$ including Maxwell stress $-B^2/8\pi\,\mathbf{I} + \mathbf{BB}/4\pi$.
- **Energy conservation** with radiative, conductive, Joule terms.
- **Induction equation** $\partial\mathbf{B}/\partial t = \nabla\!\times\!(\mathbf{v}\!\times\!\mathbf{B} - \eta\nabla\!\times\!\mathbf{B})$ (Eq. 12).

Constitutive relations (Eq. of state, Spitzer conductivity, optically thin or radiative-transfer losses, Ohm's law choices) determine which "family" a simulation belongs to.

MHD 체계는 라그랑지안 형태로 제시되며 질량·운동량·에너지·유도 방정식을 포함한다. 상태방정식, Spitzer 전도도, 광학적으로 얇은 혹은 복사 전달 손실, 옴의 법칙의 선택 같은 구성 관계가 시뮬레이션이 속하는 "계열"을 결정한다.

### Part II: Magnetic Buoyancy and Stratification (§3.1–3.2) / 자기 부력과 층화

**Buoyancy (§3.1)**: The foundational argument is total pressure balance between a magnetic structure and its surroundings:
$$p_i + \frac{B^2}{8\pi} = p_{\mathrm{amb}} \qquad (\text{Eq. 13})$$
Under thermal equilibrium (T_i = T_amb) this gives a density deficit
$$\Delta\rho/\rho \approx -\beta^{-1} \qquad (\text{Eq. 14})$$
and under isentropic conditions
$$\Delta\rho/\rho \approx -(\gamma_1 \beta)^{-1} \qquad (\text{Eq. 15})$$
Either way the structure experiences an upward buoyancy force of Δρ·g per unit volume.

부력의 기초 논증은 자기 구조와 주변의 총 압력 평형에서 출발한다. 열 평형에서 β^{-1}에 비례하는 밀도 결손이, 단열 상태에서도 비슷한 크기가 발생해 상승 부력을 만든다.

**Numerical example / 수치 예**: for a flux tube with B=10^4 G at depth where p = 10^14 dyn cm^{-2} (deep CZ), β = 8π·10^14 / 10^8 ≈ 2.5·10^7, so Δρ/ρ ≈ −4·10^{-8}. The buoyancy force is tiny per unit volume but the tube is long; rise time from base of CZ is ~10–30 days.

깊은 대류층에서 B=10^4 G일 때 β ≈ 2.5·10^7 이고 밀도 결손은 -4·10^{-8} 수준이다. 단위 체적당 부력은 작지만 자속관이 길어 상승 시간은 약 10–30일이다.

**Stratification and scaling (§3.2)**: The convection zone spans density contrast ~10^6 from z = -200 Mm to surface. Pressure scale height:
$$H_p = \left(\frac{d\ln p}{dz}\right)^{-1} \qquad (\text{Eq. 16})$$
is ~150 km at the photosphere and grows deeper in the CZ. Because H_p shrinks dramatically near the surface, a rising flux tube that maintains pressure equilibrium must expand; this expansion is predominantly horizontal (Spruit et al. 1987), producing the iconic "pancake" or sheet-like flattening seen in simulations (Figure 7).

척도 높이 H_p는 광구에서 약 150 km이며 대류층 깊은 곳으로 갈수록 커진다. 표면 근처에서 H_p가 급격히 작아지기 때문에, 압력 평형을 유지하며 상승하는 자속관은 팽창해야 하고, 이 팽창은 주로 수평으로 일어나 시뮬레이션에서 보이는 "팬케이크" 모양의 납작화를 만든다(그림 7).

Scaling between B and ρ during rise (§3.2.1):
$$B \propto \rho^{\kappa},\qquad \kappa = \frac{1+\epsilon}{2+\epsilon} \qquad (\text{Eq. 22})$$
where ε parameterizes the anisotropy of expansion. Isotropic (ε=1): κ=2/3; dominant horizontal (ε=0): κ=1/2. Pinto & Brun (2013) global anelastic simulations reported κ ≲ 1 for thin-tube-like rise while Cheung et al. (2010) surface AR formation found κ ≈ 1/2. A localized Ω-loop transitions between regimes.

상승 중 B와 ρ 사이 스케일링은 등방 팽창에서 B∝ρ^{2/3}, 수평 지배 팽창에서 B∝ρ^{1/2}이다. 깊은 가는 관 영역에서는 B∝ρ, 표면 근처에서는 B∝ρ^{1/2}로 전이된다.

**Numerical example 2 / 수치 예 2**: If a thin flux tube starts at the base of CZ with B_0 = 10^5 G (10 T, like convective intensification limit) and ρ_0 ≈ 0.2 g cm^{-3}, and emerges at τ=1 where ρ ≈ 3·10^{-7} g cm^{-3}, then for B ∝ ρ^{1/2}: B_final ≈ 10^5 × (3·10^{-7}/0.2)^{1/2} ≈ 10^5 × 1.2·10^{-3} ≈ 120 G. For ε = 1 (κ=2/3): B_final ≈ 10^5 × (1.5·10^{-6})^{2/3} ≈ 1.3 kG. The latter matches observed kG concentrations in pores and young AR.

대류층 바닥에서 B_0=10^5 G, ρ_0≈0.2 g/cm^3의 자속관이 표면(ρ≈3·10^{-7})에서 B∝ρ^{1/2}이면 약 120 G, B∝ρ^{2/3}이면 약 1.3 kG가 된다. 후자가 실제 관측되는 kG 농도와 부합한다.

**Convective stability (§3.2.2)**: The Schwarzschild criterion
$$\nabla > \nabla_{\mathrm{ad}} \quad \Leftrightarrow \quad \text{convectively unstable} \qquad (\text{Eqs. 25, 26})$$
controls whether a displaced parcel continues to rise. The photosphere has δ = ∇−∇_{ad} < 0 (subadiabatic) and acts as a barrier: a rising flux tube tends to stall just beneath the photosphere unless another agent takes over. Figure 11 in the paper shows ∇ crashing below ∇_{ad} in the photosphere.

Schwarzschild 기준은 변위된 유체 원소가 계속 상승할지 결정한다. 광구는 아디아바틱 이하(δ<0)이므로 장벽으로 작용해 상승하는 자속관은 광구 바로 아래에서 정체되기 쉽다.

### Part III: Magnetic Buoyancy Instabilities (§3.3) / 자기 부력 불안정성

Magnetic buoyancy instabilities break the photospheric stall. A horizontal flux sheet embedded in mechanical equilibrium satisfies
$$\frac{d}{dz}\!\left(p + \frac{B^2}{8\pi}\right) = -g\rho \qquad (\text{Eq. 27})$$
The magnetic pressure supports gas from below — a reservoir of free gravitational potential energy.

자기 부력 불안정성이 광구 정체를 깬다. 역학적 평형에 있는 수평 자속 판은 위 식을 만족한다. 자기 압력이 아래서 위로 가스를 지탱하며 자유 중력 위치에너지를 저장한다.

**Modes (§3.3.1, Fig. 14)**:
- **Interchange mode (k⊥B)**: field lines remain unbent; Rayleigh–Taylor-like. Criterion (Acheson 1979):
$$\frac{d}{dz}\ln\!\left(\frac{B}{\rho}\right) < -\frac{C_s^2}{g}\frac{N^2}{V_A^2} \qquad (\text{Eq. 28})$$
with N^2 the Brunt–Väisälä frequency and V_A = B/√(4πρ) the Alfvén speed.
- **Undular mode (k‖B, Parker instability)**: field lines bend; plasma slides down along field, crests rise, troughs sink. Criterion
$$\frac{d}{dz}\ln B < -\frac{C_s^2}{g}\!\left[k_\|^2\!\left(1 + \frac{k_z^2}{k_\perp^2}\right) + \frac{N^2}{V_A^2}\right] \qquad (\text{Eq. 31})$$
reducing to the simple sufficient criterion dB/dz < 0 (Eq. 33) in adiabatic stratification.

교환 모드(k⊥B)는 RT 불안정성과 유사하고, 물결 모드(k‖B, Parker 불안정성)는 자기선이 휘면서 봉우리가 솟고 골이 가라앉는다. 단열 층화에서는 dB/dz<0 조건만으로 물결 모드가 불안정하다.

**Linear growth rate (§3.3.2, Fig. 15)**: For a sheet without magnetic shear, the interchange mode grows monotonically with k; the pure undular mode peaks at k~0.5/H_p. With magnetic shear high-k modes are suppressed (Cattaneo & Hughes 1988; Nozawa 2005). Maximum undular growth at λ ~ 10 H_p, comparable to Ellerman bomb separations (Pariat et al. 2004).

전단이 없는 자속 판에서 교환 모드는 k에 따라 단조 증가하고 순수 물결 모드는 k~0.5/H_p에서 최대가 된다. 자기 전단은 고파수 모드를 억제한다.

**Nonlinear evolution (§3.3.3)**: Shibata et al. (1989a,b) 2D MHD: a horizontal photospheric flux sheet, initially perturbed at undular-mode wavelength, grows into Ω-loops. Rising velocity ~10–15 km/s, field-aligned downflow ~30–50 km/s (matching observed Ellerman bombs and moustache profiles, Chou & Zirin 1988; Otsuji et al. 2007). In 3D (Matsumoto & Shibata 1992; Matsumoto et al. 1993), interchange provides fine perpendicular structure while undular controls large-scale evolution. Convective intensification (Parker 1978; Spruit 1979) occurs at Ω-loop footpoints.

Shibata 등(1989)의 2D 시뮬레이션은 물결 모드 섭동이 Ω 루프로 성장함을 보였다. 상승 속도 10–15 km/s, 자기선 방향 하강 30–50 km/s는 Ellerman bomb 관측과 일치한다. 3D에서는 교환이 수직 미세구조를, 물결이 대규모 진화를 담당한다.

**Suppression of horizontal expansion (§3.3.4)**: An isolated untwisted tube fragments via horizontal expansion and counter-rotating vortex rolls. Twist (via magnetic tension) suppresses this (Moreno-Insertis & Emonet 1996; Emonet & Moreno-Insertis 1998) — criterion given by magnetic Weber number
$$We = \frac{v^2\rho}{B_t^2/4\pi} \lesssim 1 \qquad (\text{Eq. 42})$$
where v is rise velocity and B_t the transverse (azimuthal) field.

고립된 비꼬임관은 수평 팽창과 반대 방향 소용돌이 쌍으로 분열된다. 꼬임의 자기 장력이 이를 억제하며 그 기준은 자기 Weber 수 We≲1 이다.

### Part IV: Magnetoconvection (§3.4) / 자기대류

Convection reshapes emerging flux in several ways:

대류는 출현하는 자속을 여러 방식으로 재조형한다.

- **Upflow–downflow asymmetry**: granular upflows are broad and weak (~0.1–1 km/s); intergranular downflows are narrow and strong. Flux is pumped into downflow lanes.
- **Undulation by granular flows (§3.4.3)**: emerging horizontal field is pulled into U-loops in downdrafts and Ω-loops in upflow centers.
- **Mixed polarity / sea-serpent field (§3.4.4)**: field lines alternate between dipping into and rising out of the photosphere; opposite-polarity patches on small scales. Observed routinely in Hinode SOT-SP magnetograms of emerging ARs.
- **Convection-driven emergence (§3.4.5)**: Stein & Nordlund (2006, 2012) inject horizontal field in upflows at the bottom of a 20-Mm convection box; field organizes into small-scale bipoles at the surface with no imposed coherent tube.
- **Turbulent pumping (§3.4.6)**: stratified convection preferentially transports magnetic flux downward. Weak horizontal fields deposited at the surface are buried by downflows, consistent with the "quiet-Sun network" reservoir.

상승류-하강류 비대칭성, 입자 흐름에 의한 물결, 혼합 극성/바다뱀 구조, 대류 구동 출현(Stein & Nordlund 2012), 그리고 층화된 대류의 난류 하강 펌핑이 모두 출현 자속을 재조형한다.

Cheung et al. (2010) combined both: inject a semi-toroidal twisted tube through the bottom boundary of a radiative MHD box (at 7.5 Mm depth) — the tube flattens (§3.2), breaks up via magnetoconvection and buoyancy instability, and produces a realistic bipolar AR with coalescing pores (see Figure 9 in paper).

Cheung et al.(2010)은 7.5 Mm 깊이에서 반-토로이드 꼬임관을 주입해 납작화·자기대류·부력 불안정성이 결합되어 쌍극 AR과 합쳐지는 포어들이 형성됨을 보였다.

### Part V: Magnetic Twist (§3.6) / 자기 뒤틀림

**Idealized tube (§3.6.1)**: Axisymmetric tube with longitudinal B_l(r) and azimuthal B_θ(r). Common profile:
$$B_l(r) = B_0 \exp(-r^2/a^2),\qquad B_\theta(r) = q\,r\,B_l(r) \qquad (\text{Eqs. 40, 41})$$
with q a twist parameter and a the tube radius.

이상적 꼬임관은 축 방향 가우시안 B_l과 방위각 방향 B_θ=qr·B_l로 기술된다.

**Kink instability (§3.6.2)**: Linton et al. (1996) found critical twist
$$q_{\mathrm{cr}} = a^{-1} \qquad (\text{Eq. 43})$$
Above this, the tube axis writhes. Equivalently, total twist along a length L: Φ = q·L·B_l/B_l = qL. For a segment of length L = 2πa·N (N turns), Φ=2πN, and instability requires Φ > 2π (N > 1 turn over an Alfvén length). Matsumoto et al. (1998) and Fan et al. (1998) simulated kinked tubes rising into the atmosphere — the writhed configuration produces compact bipoles with polarity orientations deviating >90° from the axial orientation of the initial tube, reproducing delta-spot morphology.

Linton 등(1996)의 임계 꼬임 q_cr = 1/a, 즉 Alfvén 길이당 1회전 이상의 총 꼬임(Φ>2π)에서 축 휘틀림(writhe)이 발생한다. 관측되는 델타 흑점의 형성 메커니즘 후보다.

**Rotational/shearing motions (§3.6.3)**: A closed curve line integral of gas and magnetic pressure forces vanishes, but magnetic tension provides net torque:
$$t_z = \oint\!\left[\mathbf{r}\!\times\!\frac{1}{4\pi}(\mathbf{B}\!\cdot\!\nabla)\mathbf{B}\right]\!\cdot d\mathbf{l} \qquad (\text{Eq. 45})$$
This unbalanced torque drives sunspot rotation and shear flows along PILs observed by HMI.

폐곡선 적분에서 가스·자기 압력은 기여가 0이지만 자기 장력이 순 토크를 만들어 흑점 회전과 PIL 전단 흐름을 구동한다. HMI 관측과 일치.

**Helicity injection (§3.6.5)**:
$$\frac{dH_R}{dt} = 2\int\!\! \left[(\mathbf{A}_p\!\cdot\!\mathbf{B})v_n - (\mathbf{A}_p\!\cdot\!\mathbf{v})B_n\right] dS \qquad (\text{Eq. 53})$$
First term = emergence term (v_n = vertical flow), second term = shear term (v_t = horizontal flow). Magara & Longcope (2003) simulations show emergence term dominates early, shear term dominates late — consistent with Manchester (2004, 2007) interpretation of shear as propagating torsional Alfvén waves equilibrating current between sub- and suprasurface.

나선도 주입은 출현 항(수직 흐름)과 전단 항(수평 흐름)으로 나뉜다. 초기에는 출현 항, 후기에는 전단 항이 지배하며, 이는 표면 아래·위 전류 불균형을 균형 잡는 비틀림 Alfvén 파의 전파로 해석된다.

### Part VI: Resistivity, Partial Ionization, and Eruptions (§3.7–4) / 저항, 부분 전리, 분출

**Resistivity (§3.7.1)**: Spitzer conductivity gives Lundquist number S ~ 10^{-13} (T/10^6 K)^{-3/2}(L/10^9 cm)(V_A/10^8 cm/s) — enormously large. Numerical simulations cannot resolve; anomalous resistivity (Ugai & Tsuda 1977) introduces a threshold drift velocity above which η increases, producing Petschek-like fast reconnection.

**Partial ionization (§3.8)**: Hall and ambipolar diffusion appear in the generalized Ohm's law; ambipolar dominates in the lower chromosphere where neutrals slip relative to ions, effectively resistively dissipating perpendicular currents.

**Emergence as eruption trigger (§4)**: Chen & Shibata (2000) MHD simulation showed emergence of opposite-polarity flux beneath a pre-existing filament destabilizes the overlying arcade — canonical emergence-as-trigger scenario. Isobe et al. (2005) extended to cancellation/tether-cutting.

저항성 면에서 Lundquist 수는 매우 커서 수치 재결합을 강요하며, 이상 저항이 Petschek식 빠른 재결합을 만든다. 부분 전리 영역에서는 Hall·양극성 확산이 중요해지고, 양극성 확산이 채층 하부에서 수직 전류를 소산시킨다. 출현 자속이 분출 방아쇠 역할을 한다는 시나리오(Chen & Shibata 2000)는 정전기 상응 모델로 자리 잡았다.

### Part VII: Large-scale subsurface structure and jets (§3.5, §4) / 대규모 지하 구조와 제트

**Large-scale subsurface (§3.5)**: Thin flux tube and anelastic CZ models predict that coherent Ω-loops rise from the base of the CZ with tilt angles set by Coriolis deflection (Joy's law). Fan (2008), Weber et al. (2011) reproduce observed tilt-angle vs latitude by including rotational effects. Nelson et al. (2014) global dynamo simulations spontaneously form coherent flux bundles from turbulent dynamo field. A gap remains: global models capture the deep CZ but not the surface; local models capture the surface (top ~20 Mm) but not the deep rise. Coupling is the outstanding challenge.

**지하 대규모 구조 (§3.5)**: 가는 자속관과 anelastic 대류층 모델은 Coriolis 편향으로 기울기 각도가 설정된 응집 Ω 루프 상승을 예측한다(Joy 법칙). Nelson et al.(2014)의 전역 다이나모 시뮬레이션은 난류 다이나모 자기장에서 응집 자속 다발이 자발적으로 형성됨을 보였다. 전역 모델은 깊은 대류층, 국소 모델은 상부 20 Mm만을 잡아내며 이 둘의 결합이 미해결 과제다.

**Jets (§4.2)**: Emerging flux reconnects with pre-existing coronal field, producing coronal jets (X-ray and EUV). Yokoyama & Shibata (1995, 1996) canonical 2D model: emerging anemone bipole reconnects with overlying unipolar field; hot reconnection outflow accelerates plasma upward at sub-Alfvénic speeds (~100–200 km/s) while chromospheric evaporation fills the loop. Later work by Moreno-Insertis et al. (2008), Pariat et al. (2009) extended to 3D and to helical jets driven by kink-mode release of twist from emerged flux rope.

**제트 (§4.2)**: 출현 자속은 기존 코로나 자기장과 재결합해 X선·EUV 제트를 만든다. Yokoyama & Shibata(1995, 1996)의 정전 모델은 출현 아네모네 쌍극이 위 자기장과 재결합하여 100–200 km/s 속도의 상승 유출과 채층 증발을 만든다. 후속 연구는 3D와 꼬임 풀림 나선 제트로 확장했다.

**Eruptions of emerging flux ropes (§4.4)**: Some emergence events eject the entire flux rope as a CME. Archontis & Hood (2008, 2012), Leake et al. (2013, 2014) simulations show that when pre-existing coronal field has favorable orientation, reconnection between emerging and ambient field builds a new rope that becomes kink- or torus-unstable and erupts. This mechanism is especially relevant to "failed filament eruption" and "homologous flare" observations.

**출현 자속 로프의 분출 (§4.4)**: 일부 출현 사건은 자속 로프 전체를 CME로 방출한다. Archontis & Hood(2008, 2012), Leake et al.(2013, 2014) 시뮬레이션은 기존 코로나 자기장과 출현 자기장이 적절한 방향일 때 재결합으로 새 로프가 형성되고 꼬임·토러스 불안정성으로 분출하는 과정을 보였다. 실패한 필라멘트 분출과 동종 플레어 관측과 특히 관련된다.

### Part VIII: Data-driven and data-inspired models (§5) / 데이터 구동 모델

**One-way coupling (§5.1)**: Photospheric vector magnetograms drive the lower boundary of a coronal simulation via imposed v and B evolution. The corona adjusts passively. Schrijver & DeRosa (2003), Yeates et al. (2008) pioneered magneto-frictional models of the global corona driven by observed surface flux transport.

**단방향 결합 (§5.1)**: 광구 벡터 자기도가 코로나 시뮬레이션 하부 경계를 구동한다. 코로나는 수동적으로 조정되며, Schrijver & DeRosa(2003), Yeates et al.(2008)은 표면 자속 수송으로 구동되는 전역 코로나 자기마찰 모델을 선도했다.

**Current sheet formation (§5.2)**: Data-driven simulations of specific AR emergence events (Cheung & DeRosa 2012 used this approach for AR 11158) show where free energy accumulates into current sheets. The locations of predicted current sheets correlate with flare ribbon positions, validating the approach.

**전류판 형성 (§5.2)**: 특정 AR 출현 사건의 데이터 구동 시뮬레이션은 자유에너지가 전류판으로 축적되는 위치를 보여준다. 예측된 전류판 위치는 플레어 리본 위치와 상관되어 접근법을 검증한다.

**AR eruptions following emergence (§5.3)**: Kliem & Toeroek (2006) torus instability criterion requires the decay index n = −d ln B_ex/d ln z of the external field to exceed ~1.5. Data-driven models combining emergence with torus instability onset successfully predict CME timing for several M- and X-class events post-2014.

**출현 후 AR 분출 (§5.3)**: Kliem & Toeroek(2006) 토러스 불안정성 기준은 외부 자기장의 붕괴 지수 n>1.5를 요구한다. 출현과 토러스 불안정 개시를 결합한 데이터 구동 모델은 2014년 이후 여러 M·X급 사건에 대해 CME 시점을 예측해왔다.

---

## 3. Key Takeaways / 핵심 시사점

1. **Magnetic buoyancy is necessary but not sufficient** — buoyancy gives Δρ/ρ ≈ −β^{-1}, enough to rise through the CZ in ~10–30 days, but the subadiabatic photosphere halts pure buoyancy and requires an instability or convective agent to proceed.
   **자기 부력은 필요하지만 충분하지 않다** — 부력은 Δρ/ρ≈−β^{-1}의 결손을 만들어 대류층을 10–30일에 상승시키지만, 아디아바틱 광구는 순수 부력을 정지시키므로 불안정성이나 대류 요인이 필요하다.

2. **Horizontal flattening is universal near the surface** — because H_p drops from tens of Mm deep in the CZ to 150 km at the photosphere, any rising structure expands horizontally into a pancake; this robust prediction of Spruit et al. (1987) appears in every modern simulation.
   **표면 근처 수평 납작화는 보편적이다** — 척도 높이 H_p가 대류층 깊은 곳의 수십 Mm에서 광구의 150 km로 급감하므로, 상승 구조는 팬케이크 형태로 수평 팽창한다. Spruit et al.(1987)의 예측이 모든 현대 시뮬레이션에서 확인된다.

3. **B ∝ ρ^κ interpolates between tube-limit and pancake-limit** — deep thin-tube rise gives κ=1 (B∝ρ), isotropic expansion κ=2/3, horizontal pancake κ=1/2. An Ω-loop emerging from depth transitions between these as it rises.
   **B∝ρ^κ은 관 극한과 팬케이크 극한 사이를 보간한다** — 깊은 가는 관 상승은 κ=1, 등방 팽창은 κ=2/3, 수평 팬케이크는 κ=1/2. 출현하는 Ω 루프는 상승하며 전이한다.

4. **Parker (undular) and interchange instabilities break the photospheric barrier** — undular criterion dB/dz < 0 is satisfied by the field decrease above the photosphere; maximum growth at λ~10 H_p matches observed Ellerman bomb separations. Magnetic shear suppresses short wavelengths.
   **Parker(물결)와 교환 불안정성이 광구 장벽을 깬다** — dB/dz<0 조건은 광구 위 자기장 감소로 만족되며, 최대 성장 파장 λ~10H_p가 Ellerman bomb 간격과 일치한다. 자기 전단이 단파장을 억제한다.

5. **Twist is needed for coherent rise over ~Mm-scale tubes** — the magnetic Weber number We = v^2ρ/(B_t^2/4π) must be ≲1 for azimuthal field tension to prevent fragmentation into counter-rotating vortex rolls; this sets a minimum twist per unit length for observed ARs.
   **Mm 규모 자속관의 응집 상승에는 꼬임이 필요하다** — 자기 Weber 수 We≲1이어야 방위각 자기장 장력이 반대 소용돌이 쌍의 분열을 막는다.

6. **Kink instability at Φ > 2π produces delta spots** — total twist exceeding one full turn over an Alfvén length destabilizes the helical axis, causing writhe; horizontal polarity orientation deviates >90° from initial axial direction, reproducing delta-configuration observed in flare-productive ARs.
   **Φ>2π 꼬임 불안정성이 델타 흑점을 만든다** — Alfvén 길이당 1회전 초과 꼬임은 나선 축을 불안정화해 휘틀림(writhe)을 만들고, 극성 방향이 초기 축으로부터 90°>이상 회전해 델타 흑점을 재현한다.

7. **Magnetoconvection produces mixed polarity / sea-serpent field** — granular flows undulate emerging horizontal field into alternating polarity patches on ~1-Mm scales; the U-loop dips trap plasma that must drain via reconnection for flux to reach the corona.
   **자기대류가 혼합 극성/바다뱀 구조를 만든다** — 입자 흐름이 출현 수평 자속을 1 Mm 규모로 물결 치게 해 교번 극성 패치를 만들고, U 루프 골은 재결합 없이는 코로나로 자속이 도달하지 못하게 질량을 가둔다.

8. **Helicity injection is split between emergence and shear terms** — vertical flux transport (v_n B_t term) dominates early emergence; horizontal shearing (v_t B_n term) dominates later evolution. Long-duration shear contribution explains why ARs accumulate free energy days after flux stops emerging, consistent with the Longcope–Welsch current shunting model.
   **나선도 주입은 출현 항과 전단 항으로 나뉜다** — 초기에는 수직 자속 수송(v_n B_t)이 지배하고, 후기에는 수평 전단(v_t B_n)이 지배한다. 장기간 전단 기여는 AR이 자속 출현 종료 후에도 자유에너지를 축적하는 이유를 설명한다.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Buoyancy and pressure balance / 부력과 압력 평형

Total pressure balance:
$$p_i + \frac{B^2}{8\pi} = p_{\mathrm{amb}}$$
Thermal equilibrium density deficit:
$$\frac{\Delta\rho}{\rho} \approx -\beta^{-1},\qquad \beta = \frac{8\pi p}{B^2}$$
Buoyancy force per unit volume: $f_b = -g\Delta\rho = g\rho/\beta$. In a thin-flux-tube approximation the upward acceleration is
$$\frac{Dv_z}{Dt} = g\,\frac{\Delta\rho}{\rho} = \frac{g}{\beta}$$

### 4.2 Thin flux tube rise with drag / 가는 자속관 상승 (항력 포함)

A common engineering form used in 1D codes (Moreno-Insertis, Fan):
$$\frac{dv}{dt} = g\left(\frac{\Delta\rho}{\rho}\right) - \frac{C_D \rho_{\mathrm{amb}}}{\pi a \rho} v |v|$$
with a the tube radius and C_D~1. Terminal velocity when acceleration vanishes:
$$v_{\mathrm{term}} \sim \left(\frac{\pi a g \Delta\rho}{C_D \rho_{\mathrm{amb}}}\right)^{1/2}$$

### 4.3 Alfvén speed and rise velocity / Alfvén 속도와 상승 속도

$$v_A = \frac{B}{\sqrt{4\pi\rho}}$$
In thin flux tube approximation the buoyant rise velocity is bounded by v_A in the coherent regime:
$$v_{\mathrm{rise}} \sim v_A \left(\frac{a}{H_p}\right)^{1/2}$$
(Parker 1975; Moreno-Insertis 1983). For a=1 Mm at base of CZ (ρ~0.2 g/cm^3) with B=10^5 G: v_A ≈ 10^5/√(4π·0.2) = 63 km/s; with H_p ~ 50 Mm: v_rise ~ 9 km/s. Rise distance 200 Mm / 9 km/s ~ 26 days — consistent with observed AR emergence from reported subsurface detections.

### 4.4 Magnetic buoyancy instability criteria / 자기 부력 불안정성 기준

Interchange (k⊥B):
$$\frac{d}{dz}\ln\!\left(\frac{B}{\rho}\right) < -\frac{C_s^2}{g}\frac{N^2}{V_A^2}$$
Undular (k‖B):
$$\frac{d}{dz}\ln B < -\frac{C_s^2}{g}\!\left[k_\|^2\!\left(1+\frac{k_z^2}{k_\perp^2}\right) + \frac{N^2}{V_A^2}\right]$$
In adiabatic stratification (N^2=0) undular is simply:
$$\frac{dB}{dz} < 0$$
Maximum growth wavelength: λ_max ~ 10 H_p (for N^2 ~ g/H_p).

### 4.5 Kink instability (helical) / 꼬임 불안정성

For the twisted tube profile B_l = B_0 exp(-r^2/a^2), B_θ = q r B_l, Linton et al. (1996) found:
$$q_{\mathrm{cr}} = a^{-1}$$
Equivalently, critical total twist over Alfvén-length L_A:
$$\Phi_{\mathrm{cr}} = \int_0^{L_A} \frac{B_\theta}{rB_l}\,dl = q L_A > 2\pi$$
Writhe growth rate (order of magnitude): γ_kink ~ v_A/a.

### 4.6 Emergence time estimate / 출현 시간 추정

From base of CZ (d~200 Mm) to surface at v_rise ~ v_A (a/H_p)^{1/2}:
$$\tau_{\mathrm{emerge}} \sim \frac{d}{v_{\mathrm{rise}}} \sim \frac{200\,\mathrm{Mm}}{10\,\mathrm{km/s}} \sim 20\,\mathrm{days}$$
For B=10 kG (typical equipartition near base): τ~20–30 days. For B=100 kG (strong-tube models): τ~few days.

### 4.7 Convective pumping and turbulent diffusion / 대류 펌핑과 난류 확산

Mean-field EMF in stratified convection:
$$\mathbf{\mathcal{E}} = \alpha\,\mathbf{B} - \boldsymbol{\gamma}\!\times\!\mathbf{B} - \beta_t\,\nabla\!\times\!\mathbf{B}$$
where γ is the pumping velocity, antiparallel to gradient of turbulent kinetic energy. In the solar CZ γ_z < 0 (downward pumping). Weak horizontal flux deposited at the surface is transported down at ~10–100 m/s.

### 4.8 Helicity flux across photosphere / 광구 횡단 나선도 플럭스

$$\frac{dH_R}{dt} = 2\int\left[\underbrace{(\mathbf{A}_p\!\cdot\!\mathbf{B})v_n}_{\text{emergence}} - \underbrace{(\mathbf{A}_p\!\cdot\!\mathbf{v})B_n}_{\text{shear}}\right] dS$$
with A_p the vector potential of the potential field matching B_n on ∂V. Typical AR-scale helicity injection ~10^{42} Mx^2 over a few days.

### 4.9 Filling factor evolution / 충전율 진화

Filling factor f is the fraction of a resolution element covered by strong field. During emergence:
- Initial small-scale bipoles: f ~ 0.01–0.05 (scattered flux in granulation).
- Growing pore: f ~ 0.3–0.5 as flux converges into downflows.
- Mature sunspot umbra: f → 1 (continuous field).

The convective collapse mechanism (Parker 1978; Spruit 1979) locally intensifies equipartition field (~500 G) to kG after convective downflow evacuates the interior, consistent with observed G-band bright points and the unity filling factor of sunspots.

충전율 f는 분해 요소에서 강한 자기장이 차지하는 분율이다. 출현 시 초기 소규모 쌍극 ~0.01–0.05, 성장하는 포어 ~0.3–0.5, 성숙한 흑점 암부는 1에 도달한다. 대류 붕괴 메커니즘은 약 500 G를 kG로 국소 강화한다.

### 4.10 Ω-loop vs sea serpent geometry / Ω-루프 대 바다뱀 구조

**Ω-loop** (coherent emergence): a single arc reaches the photosphere with bipolar footpoints separating at horizontal flow speed ~1 km/s; filling factor f ~ 0.1 initially, → 1 in sunspot cores.

**Sea serpent** (magnetoconvection undulated): alternating U and Ω segments with wavelength ~1–5 Mm set by granulation + undular instability. Resolved mass drainage via reconnection between U-loop dips occurs before bulk flux reaches the corona.

ASCII geometry:
```
Ω-loop:                           Sea serpent:
       ___                         _   _   _
      /   \                       / \_/ \_/ \
     /     \                     /           \
----+-------+----  photosphere  -+-+-+-+-+-+-+-  photosphere
     \     /                     \/   \/   \/
      \___/                       U    U    U
     (N   S)                    (N  S  N  S  N  S)
```

### 4.11 Worked scenario — rise of an Ω-loop / 작동 예시 — Ω-루프 상승

Consider an initial horizontal twisted flux tube at depth z = -20 Mm with:
- Longitudinal field B_l0 = 2×10^4 G, radius a = 1 Mm.
- Density ρ_0 = 10^{-1} g cm^{-3}, ambient p_0 = 10^{13} dyn/cm^2.
- β_0 = 8π p_0/B_l0^2 ≈ 6×10^5.
- Localized density deficit Δρ/ρ = -β^{-1} ≈ -1.7×10^{-6} over 10 Mm-long segment.

Buoyant acceleration: a_b = g · Δρ/ρ ≈ 2.7×10^4 × 1.7×10^{-6} ≈ 0.046 cm/s^2.

Over 20 Mm rise distance (ignoring drag and κ rescaling), characteristic rise time τ = √(2d/a_b) = √(2·2×10^9/0.046) ≈ 3×10^5 s ≈ 3.5 days, consistent with observed emergence timescales after Ω formation triggers.

At surface (ρ = 3×10^{-7} g/cm^3), applying B ∝ ρ^{1/2}: B_final ≈ 2×10^4 × (3×10^{-7}/10^{-1})^{1/2} ≈ 35 G initial photospheric field — which then undergoes convective intensification to kG in downflows.

초기 수평 꼬임관(B_l0=2×10^4 G, a=1 Mm, 깊이 20 Mm)에서 β^{-1}≈1.7×10^{-6} 결손이 a_b≈0.046 cm/s^2 상승 가속도를 만들고, 표면까지 ~3.5일에 도달한다. 표면에서 B∝ρ^{1/2} 스케일로 약 35 G가 되며, 이후 대류 집중으로 kG로 강화된다.

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1955  Parker             Magnetic buoyancy: sunspots rise due to ρ deficit
1966  Parker             Undular (Parker) instability — discovered for ISM
1974  Parker             Twist/winding conservation in force-free evolution
1978  Spruit             Thin flux tube equations (1D Lagrangian)
1979  Acheson            General magnetic buoyancy instability criteria
1987  Spruit et al.      Horizontal flattening near photosphere predicted
1989  Shibata et al.     First 2D MHD emergence via undular mode
1993  Matsumoto et al.   First 3D MHD flux emergence
1996  Moreno-Insertis &  Twist suppresses vortex-roll fragmentation
        Emonet            (magnetic Weber number criterion)
1998  Fan et al.         Kink instability -> delta spots in simulation
2004  Archontis et al.   First CZ-to-corona flux emergence simulation
2006  Stein & Nordlund   Convection-driven emergence (no prescribed tube)
2010  Cheung et al.      Radiative MHD realistic AR formation
2011  Ilonidis et al.    Helioseismic detection of subsurface flux at 65 Mm
2014  CHEUNG & ISOBE     THIS REVIEW — theoretical synthesis
2014  Nelson et al.      Global dynamo simulations produce Ω-loops
2019  Chen et al.        Data-driven CME simulation (post-review)
2022  Inoue et al.       Data-constrained MHD X-class flare prediction
```

This review is the bridge between 50 years of theoretical development and the contemporary era of data-driven and data-constrained MHD modeling.

본 리뷰는 50년의 이론적 발전과 데이터 구동·데이터 제약 MHD 모델링의 현대를 잇는 다리 역할을 한다.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Parker (1955) ApJ 121 | Introduced magnetic buoyancy; direct ancestor. / 자기 부력 개념의 직계 조상. | Foundational; entire §3.1 builds on it. / 기초; §3.1 전체가 이 위에 세워짐. |
| Parker (1966) ApJ 145 | Undular (Parker) instability derivation, originally for interstellar medium. / 성간 매질 맥락에서 물결 불안정성 유도. | §3.3.1 instability criteria directly from this. / §3.3.1 기준은 바로 여기서 유래. |
| Spruit (1981) A&A | Thin flux tube equations. / 가는 자속관 방정식. | Underlies all §2.3 "diversity of models"; 1D rise simulations. / §2.3 모델 다양성의 기초. |
| Spruit et al. (1987) | Horizontal flattening conjecture. / 수평 납작화 추측. | Eq. 22 scaling and Figure 7 confirm this prediction. / 식 22 스케일링과 그림 7이 이 예측을 확인. |
| Fan (2009b) LRSP | Prior Living Review on flux emergence. / 기존 LRSP 플럭스 출현 리뷰. | This 2014 review extends & updates Fan (2009); covers surface radiative MHD era. / 2014 리뷰는 Fan(2009)를 확장·갱신. |
| Cheung et al. (2010) ApJ | Radiative MHD AR formation. / 복사 MHD AR 형성. | First author's own work; Figure 9 centerpiece of §3.2–3.4. / 저자 본인 작업; §3.2–3.4 중심. |
| Shibata et al. (1989) ApJ | First 2D MHD emergence. / 최초 2D MHD 출현. | §3.3.3 nonlinear Ω-loop rise example. / §3.3.3 비선형 Ω 루프 상승 예시. |
| Moreno-Insertis & Emonet (1996) | Twisted tube coherence. / 꼬임관 응집. | §3.3.4 horizontal expansion suppression; Weber number. / §3.3.4 수평 팽창 억제. |
| Linton et al. (1996) | Kink instability criterion q_cr=1/a. / 꼬임 불안정 기준. | §3.6.2 direct citation for delta spot formation. / §3.6.2 델타 흑점 형성 직접 인용. |
| Berger & Field (1984) | Relative helicity definition. / 상대 나선도 정의. | §3.6.5 helicity flux formula derives from this. / §3.6.5 나선도 플럭스 공식의 기원. |
| Stein & Nordlund (2012) | Convection-driven emergence. / 대류 구동 출현. | §3.4.5 alternative paradigm without imposed coherent tube. / §3.4.5 외부 주입 응집 관 없는 대안. |

---

## 7. References / 참고문헌

- Cheung, M. C. M., Isobe, H. "Flux Emergence (Theory)", Living Rev. Solar Phys. 11, 3 (2014). DOI: 10.12942/lrsp-2014-3
- Parker, E. N. "The Formation of Sunspots from the Solar Toroidal Field", ApJ 121, 491 (1955).
- Parker, E. N. "The Dynamical State of the Interstellar Gas and Field", ApJ 145, 811 (1966).
- Spruit, H. C. "Motion of Magnetic Flux Tubes in the Solar Convection Zone and Chromosphere", A&A 98, 155 (1981).
- Spruit, H. C., Title, A. M., van Ballegooijen, A. A. "Is There a Weak Mixed Polarity Background Field?", Solar Phys. 110, 115 (1987).
- Shibata, K., Tajima, T., Matsumoto, R., Horiuchi, T., Hanawa, T., Rosner, R., Uchida, Y. ApJ 338, 471 (1989).
- Matsumoto, R., Tajima, T., Shibata, K., Kaisig, M. ApJ 414, 357 (1993).
- Moreno-Insertis, F., Emonet, T. ApJ Lett. 472, L53 (1996).
- Linton, M. G., Longcope, D. W., Fisher, G. H. ApJ 469, 954 (1996).
- Fan, Y., Zweibel, E. G., Linton, M. G., Fisher, G. H. ApJ 505, L59 (1998).
- Archontis, V., Moreno-Insertis, F., Galsgaard, K., Hood, A., O'Shea, E. A&A 426, 1047 (2004).
- Cheung, M. C. M., Rempel, M., Title, A. M., Schüssler, M. ApJ 720, 233 (2010).
- Stein, R. F., Nordlund, Å. Solar Phys. 240, 211 (2006).
- Stein, R. F., Nordlund, Å. ApJ 753, L13 (2012).
- Berger, M. A., Field, G. B. J. Fluid Mech. 147, 133 (1984).
- Fan, Y. "Magnetic Fields in the Solar Convection Zone", Living Rev. Solar Phys. 6, 4 (2009).
- Acheson, D. J. Solar Phys. 62, 23 (1979).
- Chandrasekhar, S. "Hydrodynamic and Hydromagnetic Stability", Oxford (1961).
- Magara, T., Longcope, D. W. ApJ 586, 630 (2003).
- Manchester IV, W. B., Gombosi, T., DeZeeuw, D., Fan, Y. ApJ 610, 588 (2004).
- Nozawa, S. PASJ 57, 995 (2005).
- Murray, M. J., Hood, A. W., Moreno-Insertis, F., Galsgaard, K., Archontis, V. A&A 460, 909 (2006).
