---
title: "The Parker Problem: Existence of Smooth Force-Free Fields and Coronal Heating"
authors: David I. Pontin, Gunnar Hornig
year: 2020
journal: "Living Reviews in Solar Physics 17:5"
doi: "10.1007/s41116-020-00026-5"
topic: Living_Reviews_in_Solar_Physics
tags: [parker_problem, coronal_heating, force_free, current_sheets, nanoflares, flux_braiding, reduced_mhd, topological_dissipation]
status: completed
date_started: 2026-04-23
date_completed: 2026-04-23
---

# 69. The Parker Problem: Existence of Smooth Force-Free Fields and Coronal Heating / Parker 문제: 부드러운 힘없는 장의 존재와 코로나 가열

---

## 1. Core Contribution / 핵심 기여

Pontin and Hornig provide a comprehensive review of the "Parker problem" — the question first posed by Parker (1972) of whether an ideal-MHD coronal magnetic field, subjected to smooth but arbitrary footpoint shuffling at perfectly-conducting photospheric boundaries, will in general relax to a smooth force-free equilibrium or instead develop tangential magnetic field discontinuities (current sheets). They precisely formulate three versions of Parker's hypothesis (general, force-free, and weak), assemble decades of theoretical arguments for and against the existence of smooth equilibria (Parker's optical analogy, Low et al.'s topologically-untwisted fields, Ng–Bhattacharjee's reduced-MHD uniqueness theorem, Cowley et al.'s simply-connected tangential discontinuity impossibility proof, van Ballegooijen's error correction in Parker's original perturbation analysis), and synthesize computational evidence from ideal Lagrangian relaxation and continuously-driven flux braiding simulations. Their verdict: the strongest forms of Parker's hypothesis (Definitions 1 and 2) remain mathematically unproven, but the weak form (Definition 3) — that current layers thin exponentially in response to continued braiding — is firmly established. This weak form is sufficient to underpin the nanoflare mechanism for coronal heating.

Pontin과 Hornig는 "Parker 문제" — 완전 도전성 광구 경계에서 부드럽지만 임의적인 발자국 뒤섞임을 받는 이상 MHD 코로나 자기장이 일반적으로 부드러운 힘없는 평형으로 이완될 것인지, 아니면 접선 자기장 불연속(전류 시트)을 형성할 것인지에 대한 Parker(1972)의 질문 — 에 대한 종합적 리뷰를 제공한다. 그들은 Parker 가설의 세 가지 형태(일반형, 힘없는 형, 약한 형)를 정밀하게 공식화하고, 부드러운 평형 존재를 둘러싼 수십 년의 이론적 논증(Parker의 광학적 유추, Low 등의 위상적 비-꼬임 장, Ng–Bhattacharjee의 축소 MHD 유일성 정리, Cowley 등의 단순 연결 접선 불연속 불가능성 증명, van Ballegooijen의 Parker 원본 섭동 분석의 오류 수정)을 집대성하며, 이상 Lagrangian 이완 및 지속적으로 구동되는 자속 braiding 시뮬레이션으로부터의 계산적 증거를 종합한다. 그들의 판결: Parker 가설의 가장 강한 형태(정의 1, 2)는 수학적으로 증명되지 않았지만, 약한 형태(정의 3) — 계속되는 braiding에 반응하여 전류 층이 지수함수적으로 얇아진다 — 는 확고히 확립되었다. 이 약한 형태는 코로나 가열의 나노플레어 메커니즘을 뒷받침하기에 충분하다.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction and Statement of the Parker Hypothesis / 서론과 Parker 가설의 진술 (§1–2)

The coronal heating problem traces to the 1940s discovery by Edlén and Grotrian that the solar corona is ~10⁶ K — two orders of magnitude hotter than the photosphere. Parker (1972) argued that because photospheric convective flow speeds (v ∼ 1–10 km/s) are vastly slower than the coronal Alfvén speed (v_A ∼ 1000 km/s), the coronal magnetic field evolves quasi-statically through approximately force-free equilibria. With the low plasma-β of the corona (β ≲ 10⁻⁴), pressure gradients drop out and the equilibrium condition simplifies to

$$(\nabla\times\mathbf{B})\times\mathbf{B} = 0.$$

Parker then posed the question: as the field-line footpoints are shuffled by the photosphere, does the coronal field pass through a smooth sequence of such force-free equilibria, or does it encounter states where no smooth equilibrium exists — so that relaxation must produce discontinuities? He termed the latter process "topological dissipation" and identified it as the progenitor of "nanoflare" heating (Parker 1988).

코로나 가열 문제는 1940년대 Edlén과 Grotrian이 태양 코로나가 광구보다 두 자릿수 높은 ~10⁶ K임을 발견한 것으로 거슬러 올라간다. Parker(1972)는 광구 대류 흐름 속도(v ∼ 1–10 km/s)가 코로나 Alfvén 속도(v_A ∼ 1000 km/s)보다 훨씬 느리므로 코로나 자기장이 거의 힘없는 평형을 통해 준정적으로 진화한다고 주장했다. 저-β 코로나(β ≲ 10⁻⁴)에서 압력 구배는 사라지고 평형 조건은 (∇×B)×B = 0으로 단순화된다. Parker는 물었다: 장선 발자국이 광구에 의해 뒤섞일 때, 코로나 장은 그러한 힘없는 평형의 부드러운 연쇄를 통과하는가, 아니면 부드러운 평형이 존재하지 않는 상태를 만나 이완이 불연속을 생성해야 하는가? 그는 후자를 "위상적 소산"이라 명명하고 "나노플레어" 가열(Parker 1988)의 원천으로 확인했다.

The geometry of the Parker problem is explicit: an infinite volume 𝒱 between two parallel perfectly-conducting planes at z=0 and z=L, initially threaded by the uniform field **B** = B₀**e**_z. The plasma is perfectly conducting so it obeys

$$\frac{\partial\mathbf{B}}{\partial t} = \nabla\times(\mathbf{v}\times\mathbf{B}),$$

which enforces field-line topology conservation (the frozen-in theorem). Boundary flows v_t at z=0, z=L are smooth and tangential. After flows stop at t=τ, the field is left in a topologically tangled state (see Parker's Fig. 1), and is held with fixed footpoints while seeking a static equilibrium. The authors present three precise statements:

- **Definition 1 (general form, Parker 1994)**: For almost all possible boundary flows the field develops tangential discontinuities during relaxation to static equilibrium, due to the absence of any smooth static equilibrium.
- **Definition 2 (force-free form)**: The same claim applied specifically to the force-free equilibrium (valid for low β).
- **Definition 3 (weak form)**: For almost all boundary flows the magnetic field develops current layers; the width of these layers decreases and their strength increases exponentially with the complexity of the deformation — never becoming singular in finite time.

Parker 문제의 기하는 명시적이다: z=0과 z=L의 두 평행한 완전 도전성 평면 사이의 무한 부피 𝒱, 초기에 균일장 B = B₀**e**_z로 관통된다. 플라스마는 완전 도전성이어서 ∂B/∂t = ∇×(v×B)를 따른다. 이는 장선 위상 보존(자속-동결 정리)을 강제한다. 경계 흐름 v_t는 z=0, z=L에서 부드럽고 접선 방향이다. t=τ에서 흐름이 멈춘 후, 장은 위상적으로 얽힌 상태로 남고, 발자국은 고정된 채 정적 평형을 찾는다. 저자들은 세 가지 정밀한 진술을 제시한다: (1) 일반형 — 거의 모든 경계 흐름에 대해 장은 부드러운 정적 평형의 부재로 인해 접선 불연속을 형성; (2) 힘없는 형 — 힘없는 평형에 특화된 같은 주장; (3) 약한 형 — 층의 너비가 변형의 복잡도와 함께 지수함수적으로 감소하되 유한 시간에 특이해지지는 않음.

### Part II: Energy Injection / 에너지 주입 (§3)

Observational coronal heating requirements are: quiet Sun ∼ 100 W m⁻² (= 10⁵ erg cm⁻² s⁻¹), coronal holes ∼ 300 W m⁻², active regions ∼ 10⁴ W m⁻² (= 10⁷ erg cm⁻² s⁻¹) (Withbroe & Noyes 1977). Parker proposed the photospheric convection tangles coronal field lines, building up free magnetic energy that is then dissipated as heat. Parker (1988) estimated that for a shear-driven loop with B=100 G, v=0.5 km/s, L=100 Mm, the average perpendicular field component |B_⊥| should grow to about 25% of the axial field — giving a Parker angle of ∼14° between field lines and the vertical.

관측된 코로나 가열 요구량은: 조용한 태양 ∼ 100 W m⁻² (= 10⁵ erg cm⁻² s⁻¹), 코로나 홀 ∼ 300 W m⁻², 활동영역 ∼ 10⁴ W m⁻² (= 10⁷ erg cm⁻² s⁻¹)이다. Parker(1988)는 B=100 G, v=0.5 km/s, L=100 Mm의 전단-구동 루프에 대해 평균 수직 장 성분 |B_⊥|가 축 방향 장의 약 25%로 성장하며 — 장선과 수직 방향 사이에 ∼14°의 Parker 각도를 준다고 추정했다.

Energy injection is quantified by the Poynting flux through the photosphere. Starting from

$$\mathbf{F} = \frac{1}{\mu_0}\mathbf{E}\times\mathbf{B} = -\frac{(\mathbf{v}\cdot\mathbf{B})\mathbf{B}}{\mu_0} + \frac{B^2\mathbf{v}}{\mu_0},$$

and assuming v_n=0 at the photosphere, one obtains

$$F_P = -\int_S \frac{1}{\mu_0}(\mathbf{v}_t\cdot\mathbf{B}_t)\,B_n\,dA.$$

The integrand depends on the relative orientation of tangential flow **v**_t and tangential field **B**_t. Observational estimates using Fourier Local Correlation Tracking (FLCT) of plage magnetograms (Yeates et al. 2014; Welsch 2015) find net-upward Poynting fluxes of ∼5×10⁴ W m⁻², more than sufficient to balance quiet-Sun losses and comparable to active-region losses. Magneto-convection simulations (Shelyag et al. 2011, 2012) produce similar fluxes, identifying horizontal motions in intergranular magnetic flux concentrations as the dominant contribution.

에너지 주입은 광구를 통과하는 Poynting 플럭스로 정량화된다. 광구에서 v_n=0을 가정하면 F_P = -∫_S (1/μ₀)(v_t·B_t)B_n dA를 얻는다. 관측적 추정(Yeates 등 2014, Welsch 2015)은 plage magnetogram의 FLCT를 사용해 ∼5×10⁴ W m⁻²의 순상향 Poynting 플럭스를 발견한다. 자기-대류 시뮬레이션(Shelyag 등 2011, 2012)도 유사한 플럭스를 생성하며, intergranular 자속 집중에서의 수평 운동이 주된 기여임을 확인한다.

Nechaev (1999) provides a topological argument for energy buildup: by analogy with statistics of random braids, applying random footpoint motions makes increasing tangling far more likely than untangling. Since the tangling of field lines puts a lower bound on magnetic energy (Moffatt 1985), the stored free energy must steadily increase until a threshold triggers release.

Nechaev(1999)는 에너지 축적에 대한 위상학적 논증을 제공한다: 무작위 braid 통계와의 유추로, 무작위 발자국 운동은 얽힘 해소보다 얽힘 증가를 훨씬 더 가능성 있게 만든다. 장선 얽힘이 자기 에너지에 하한을 두므로(Moffatt 1985), 축적된 자유 에너지는 방출 임계치까지 꾸준히 증가해야 한다.

### Part III: Theoretical Arguments — Force-Free Properties / 이론적 논증: 힘없는 장 성질 (§4.1)

The force-free condition **J**×**B** = 0 is equivalent (where **B**≠0) to ∇×**B** = α(x)**B**, with the solenoidal constraint implying **B**·∇α = 0 — α is constant along field lines. These are "Beltrami" fields (Beltrami 1889) or "Trkalian" fields when α is globally constant (linear force-free).

힘없는 조건 J×B = 0은 (B≠0인 곳에서) ∇×B = α(x)B와 동등하며, solenoidal 제약은 B·∇α = 0 — α가 장선을 따라 일정함 — 을 함의한다. 이들은 Beltrami 장(Beltrami 1889) 또는 α가 전역적으로 일정할 때 Trkalian 장(선형 힘없는)이라 한다.

Topology of flux surfaces imposes strong constraints. Since α is constant on flux surfaces, and these surfaces must be (topologically) tori in 𝒱 with no nulls and no closed field lines (Arnold 1986 — only the torus is a closed compact surface of Euler characteristic 0 in ℝ³), ergodic field lines (commonly present in real non-symmetric MHS equilibria) cannot appear on surfaces of constant α. This excludes entire classes of fields — periodic cylinders cannot host smooth force-free fields with non-zero helicity (Vainshtein 1992). However, Parker's geometry is not periodic — the flux surfaces α=const end on the boundaries — so topological non-existence arguments do not directly apply.

자속 표면 위상은 강한 제약을 부과한다. α는 자속 표면 위에서 일정하고, 이 표면들은 𝒱 내에서 null이나 닫힌 장선 없이 위상적으로 토러스여야 하므로(Arnold 1986), ergodic 장선은 상수 α 표면에 나타날 수 없다. 이는 전체 종류의 장들을 배제한다 — 주기적 원통은 0이 아닌 helicity를 가진 부드러운 힘없는 장을 수용할 수 없다. 그러나 Parker의 기하는 주기적이지 않으므로 이러한 위상적 비-존재 논증은 직접 적용되지 않는다.

### Part IV: Parker's Perturbation Analysis and van Ballegooijen's Correction / Parker의 섭동 분석과 van Ballegooijen의 수정 (§4.2)

Parker (1972) analyzed small perturbations of **B** = B₀**e**_z with a series in ε ∼ |b|/B₀:

$$p = \sum_{n=0}^\infty \epsilon^n p_n,\qquad \mathbf{b}=\sum_{n=1}^\infty \epsilon^n \mathbf{b}_n,$$

where ∇·**b**_n = 0 and |**b**_n| ∼ B₀. Inserting into the MHS equation (1/4π)(∇×B)×B = ∇p (Gaussian units used here to match Parker's convention) and collecting powers of ε:

$$\nabla p_0 + \left[\nabla\!\left(p_1 + \frac{B_0 b_{1z}}{4\pi}\right) - \frac{B_0}{4\pi}\frac{\partial\mathbf{b}_1}{\partial z}\right]\epsilon + \mathcal{O}(\epsilon^2) = 0.$$

Parker set the first-order term to zero, giving

$$\nabla\!\left(p_1 + \frac{B_0 b_{1z}}{4\pi}\right) = \frac{B_0}{4\pi}\frac{\partial\mathbf{b}_1}{\partial z}.$$

Taking the divergence and using ∇·b₁=0 gives ∇²(p₁ + B₀b_{1z}/4π) = 0. Since variations in p₁+B₀b_{1z}/4π occur on the photospheric length scale l_⊥ while the loop length is l_z ≫ l_⊥, Parker argued the only bounded solution is p₁+B₀b_{1z}/4π = const, hence ∂b₁/∂z = 0 for all orders. Since boundary displacements generally differ at z=0 and z=L, no smooth equilibrium can satisfy ∂b/∂z = 0 throughout — leading Parker to conclude smooth equilibria do not exist.

Parker(1972)는 B = B₀**e**_z의 작은 섭동을 ε ∼ |b|/B₀의 급수로 분석했다. MHS 방정식에 삽입하고 ε의 거듭제곱을 모으면 일차 항을 0으로 설정해 Parker는 ∂b/∂z = 0이 모든 차수에서 성립한다고 결론지었다. 경계 변위가 z=0과 z=L에서 일반적으로 다르므로 부드러운 평형이 존재할 수 없다고 주장했다.

**Van Ballegooijen's (1985) correction**: The right-hand side term (B₀/4π)∂b₁/∂z is actually *first order in ε itself* (not zeroth), because b₁ carries a factor of ε already in the expansion. Properly accounting for this, the equation reduces to

$$\nabla\!\left(p_1 + \frac{B_0 b_{1z}}{4\pi}\right) = 0,$$

and ∂b₁/∂z ≠ 0 is allowed. The set of equilibrium solutions broadens to include z-dependent fields, and van Ballegooijen shows that for smooth boundary motions a corresponding smooth equilibrium exists. Zweibel & Li (1987) independently confirm this with a Lagrangian (fluid-displacement) approach, sidestepping the topology-specification difficulty. Craig & Sneyd (2005) use Fourier expansion to extend these results, though Low (2010a, b) has re-opened questions about their interpretation. Rosner & Knobloch (1982) and Arendt & Schindler (1988) give further examples of smooth equilibria for small perturbations. Bobrova & Syrovatskii (1979) provide a counter-example (1D force-free **B**=(cos(αz), sin(αz), 0)) where current becomes singular — but their geometry differs from Parker's (perfectly-conducting planes tangent to the field).

Van Ballegooijen(1985)의 수정: RHS 항 (B₀/4π)∂b₁/∂z는 실제로 ε 자체에서 일차(0차가 아니라)이다. b₁이 이미 전개에서 ε 인자를 가지기 때문이다. 이를 올바르게 고려하면 ∂b₁/∂z ≠ 0이 허용되고, 평형해 집합은 z-의존 장을 포함하도록 넓어진다. Zweibel & Li(1987)는 Lagrangian 접근으로 독립적으로 이를 확인한다.

### Part V: Arguments Against Smooth Equilibria — Parker's Optical Analogy / 부드러운 평형에 반대하는 논증: Parker의 광학적 유추 (§4.3.1)

A field line in a potential field satisfies

$$\frac{dx_i}{ds} = \frac{1}{n}\frac{\partial\psi}{\partial x_i},$$

where ψ is the magnetic potential (**B**=∇ψ) and n=|B|. This is identical to the optical ray equation with refractive index |B|. On a flux surface of ∇×**B** (where ∇×**B**·**n**=0), Stokes' theorem shows ∮**B**·d**l** = 0 for any closed curve. By analogy with Fermat's principle, field lines on the flux surface minimize the "optical path" ∫|B(s)|ds. Field lines avoid regions of high field strength — and if |B| is sufficiently strong somewhere on the flux surface, field lines can be completely excluded, creating a "hole" in the surface. Two bundles of flux separated by the surface (one on each side) come into contact across the hole; if their directions don't match, a tangential discontinuity must form (Fig. 3 of the paper).

잠재장에서 장선은 dx_i/ds = (1/n)∂ψ/∂x_i를 따른다. 이는 굴절률 |B|를 가진 광학 광선 방정식과 동일하다. Fermat 원리와의 유추로 장선은 "광학 경로" ∫|B(s)|ds를 최소화한다. 장선은 높은 장 강도 영역을 피하며, |B|가 충분히 강하면 장선이 완전히 배제되어 표면에 "구멍"이 생긴다. 표면 양쪽의 자속 다발이 구멍을 가로질러 접촉하며 방향이 맞지 않으면 접선 불연속이 형성되어야 한다.

However, this argument is persuasive but not rigorous: the field strength distribution is specified a priori (not consistent with a prescribed footpoint displacement), the analogy holds only locally on flux surfaces (global flux-surface existence is not assured), and the only concrete example (Parker 1990) does not conform to the Parker geometry because lateral boundaries supply extra pressure.

하지만 이 논증은 설득력은 있으나 엄밀하지 않다: 장 강도 분포가 선험적으로 지정되고, 유추는 자속 표면 위에서 국소적으로만 성립하며, 구체적 예제는 Parker 기하에 부합하지 않는다.

Low et al. (2006, 2007, 2009, 2010) consider "topologically untwisted" fields: fields with zero net circulation on flux tube cross-sections. They argue that perturbing such a field (e.g., by shrinking the cylinder length while maintaining B·n on boundaries) produces a new potential field whose topology is *different* from the initial one. Thus topology preservation requires current sheets. Aly & Amari (2010) refute this by providing counter-examples where the relaxed field is not topologically untwisted but carries distributed currents (Pontin & Huang 2012; Huang et al. 2009).

Low 등(2006, 2007, 2009, 2010)은 "위상적으로 비-꼬임(topologically untwisted)" 장 — 자속관 단면에서 순환이 0인 장 — 을 고려한다. 이러한 장을 섭동하면 초기와 *다른* 위상의 새로운 잠재장이 생성되므로 위상 보존에는 전류 시트가 필요하다고 주장한다. Aly & Amari(2010)는 분산 전류를 운반하는 이완된 장의 반례를 제공해 이를 반박한다.

Ng & Bhattacharjee (1998) exploit the parallel between reduced MHD (RMHD) and the 2D Euler equation in hydrodynamics to prove: "For any given footpoint mapping connected smoothly with the identity mapping, there is at most one smooth equilibrium". Loss of stability therefore forces relaxation toward a non-smooth state, the unique smooth state being unstable. Their proof relies on RMHD approximations; implications for full MHD with line-tying remain unclear.

Ng & Bhattacharjee(1998)는 축소 MHD와 2D Euler 방정식의 병렬을 이용해 "단위 사상과 부드럽게 연결된 주어진 발자국 사상에 대해 부드러운 평형은 최대 하나" 임을 증명한다. 안정성 상실은 이완을 비-부드러운 상태로 강제하며, 유일한 부드러운 상태는 불안정하다.

### Part VI: Arguments in Favour of Smooth Equilibria / 부드러운 평형을 지지하는 논증 (§4.4)

Bineau (1972) provides the strongest known existence result: for any potential field **B**⁰ and any boundary function σ̄ on S₁, there exists a family of force-free fields depending analytically on β in a neighborhood of β=0 (i.e., satisfying ∇×**B** = βσ**B** with small β). The iteration converges for sufficiently small β — but how small? The allowed α-range depends on field topology and is not given explicitly.

Bineau(1972)는 알려진 가장 강력한 존재 결과를 제공한다: 임의의 잠재장 B⁰와 경계함수 σ̄에 대해 β=0 근방에서 β에 해석적으로 의존하는 힘없는 장 족이 존재한다. 그러나 허용 α-범위는 장 위상에 의존하며 명시적으로 주어지지 않는다.

**Tangential discontinuity impossibility argument (van Ballegooijen 1985, 1988a; Longcope & Strauss 1994; Cowley et al. 1997)**: Suppose a simply-connected current sheet forms in a force-free field between line-tied boundaries. Consider two field lines 𝒞₁, 𝒞₂ immediately in front of and behind the sheet. They start infinitesimally close at z=0 (by smooth boundary flows from a smooth initial state) and must remain infinitesimally close at z=L. In a force-free field, the current flows entirely along the sheet and never leaves, so the total current around any closed loop on either surface is zero:

$$\oint \mathbf{B}\cdot d\mathbf{l} = 0.$$

Constructing a loop 𝒞₁ ∪ 𝒞₂' (where 𝒞₂' is a path along 𝒞₂ but displaced to the front side of the sheet), and noting that across a tangential discontinuity |**B**| is continuous (Landau & Lifshitz 1960), one obtains

$$\int_{\mathcal{C}_1} B_1(l)\,dl - \int_{\mathcal{C}_2'} B_1(l)\,\hat{\mathbf{B}}_2(l)\cdot\hat{\mathbf{B}}_1(l)\,dl = 0,$$

and similarly behind the sheet. Adding the two:

$$\oint_{\mathcal{C}_1+\mathcal{C}_2} B(l)\bigl[1-\cos\theta(l)\bigr]\,dl = 0,$$

where θ(l) is the angle between **B**₁ and **B**₂. The integrand is everywhere ≥ 0, so the only solution is θ(l) = 0 — contradicting the tangential discontinuity. Hence simply-connected tangential discontinuities cannot form under smooth boundary motions in an ideal plasma. Current sheets of branching (Y-type) or more complicated topology may still arise (Ng & Bhattacharjee 1998).

접선 불연속 불가능성 논증: 선 고정 경계 사이에 단순 연결 전류 시트가 형성된다고 가정하자. 힘없는 장에서 전류는 시트를 따라서만 흐르므로 어느 표면의 닫힌 루프에서든 ∮B·dl = 0이다. 시트를 둘러싼 두 루프를 구성하고 접선 불연속을 가로질러 |B|가 연속임을 이용하면 ∮B(l)[1-cos θ(l)]dl = 0을 얻는다. 피적분자가 모든 곳에서 ≥ 0이므로 유일한 해는 θ(l) = 0 — 접선 불연속과 모순. 따라서 이상 플라스마에서 부드러운 경계 운동 하에 단순 연결 접선 불연속은 형성될 수 없다.

This argument assumes no nulls, bald patches, or separatrix surfaces in the volume — precisely the assumptions of the Parker problem. More complex topologies (nulls, QSLs) can host current sheets; this is addressed in §7.

이 논증은 부피 내에 null, bald patch, 분리면이 없음을 가정한다 — 정확히 Parker 문제의 가정이다. 더 복잡한 위상(null, QSL)은 전류 시트를 수용할 수 있다.

**Ideal instabilities with line-tying**: The coalescence instability (Finn & Kaw 1977) and ideal kink (Kruskal et al. 1958; Hood & Priest 1979) form singular current sheets in periodic geometry. With line-tying, Longcope & Strauss (1994) showed that the coalescence instability produces intense but finite current concentrations (perpendicular length scale set by field-line mapping). The kink instability similarly develops smooth but narrow current layers in line-tied systems (Huang et al. 2010).

선 고정이 있는 이상 불안정성: coalescence와 ideal kink 불안정성은 주기적 기하에서 특이 전류 시트를 형성하지만, 선 고정 하에서는 강하지만 유한한 전류 집중으로 바뀐다.

### Part VII: Computational Approaches — Ideal Relaxation / 계산 접근: 이상 이완 (§5)

The computational challenge of testing Parker's hypothesis is preserving the exact field-line mapping during relaxation — Eulerian MHD codes suffer numerical diffusion ('slippage') that is exacerbated at thin structures.

Parker 가설 시험의 계산적 도전은 이완 중 정확한 장선 사상을 보존하는 것이다 — Eulerian MHD 코드는 수치 확산("미끄러짐")을 겪는다.

**Lagrangian magnetofrictional approach**: Starting from ideal induction D(**B**/ρ)/Dt = (**B**/ρ)·∇**v**, a Lagrangian mesh preserves field topology exactly. Fluid element positions **x**(**X**,t) follow the pull-back:

$$\mathbf{x}^*(\mathbf{B}(\mathbf{x},t),t) = \mathbf{B}(\mathbf{X},0),\qquad B_i(\mathbf{x},t) = \frac{1}{\Delta}\sum_j \frac{\partial x_i}{\partial X_j} B_j(\mathbf{X},0),$$

where Δ is the Jacobian determinant. Evolution via magneto-friction

$$\frac{D\mathbf{x}}{Dt} = \gamma\mathbf{F} = \gamma(\nabla\times\mathbf{B})\times\mathbf{B} - \gamma\nabla p$$

monotonically reduces magnetic energy toward force-free. Implementations: Craig & Sneyd (1986), Longbottom et al. (1998), Candelaresi et al. (2014) — the latter using a mimetic divergence-free derivative to improve accuracy. Candelaresi et al. (2015) compared evolution equations with/without inertia; no dependence on the choice was found.

라그랑주 자기마찰 접근: 이상 유도에서 시작해 라그랑주 격자가 장 위상을 정확히 보존한다. 자기마찰 Dx/Dt = γF = γ(∇×B)×B - γ∇p로 진화시키면 자기 에너지가 힘없는 상태로 단조감소한다.

**Variational integrator method (Zhou et al. 2014, 2018)**: Based on discrete exterior calculus applied to the Lagrangian for ideal MHD:

$$L(\mathbf{v},\rho,p,\mathbf{B}) = \int\!\left(\frac{1}{2}\rho v^2 - \frac{p}{\gamma-1} - \frac{B^2}{2}\right)d^3x,$$

giving the Euler-Lagrange equation

$$\rho_0\ddot{x}_i - B_{0j}\frac{\partial}{\partial x_{0j}}\!\left(\frac{x_{ik}B_{0k}}{\Delta}\right) + \frac{\partial\Delta}{\partial x_{ij}}\frac{\partial}{\partial x_{0j}}\!\left(\frac{p_0}{\Delta^\gamma} + \frac{x_{kl}x_{km}B_{0l}B_{0m}}{2\Delta^2}\right) = 0.$$

A friction term -νρẋ is added to damp motion. Zhou et al. (2018) applied this to the Parker geometry (RMHD variant) and found finite, smooth current layers for short system lengths but pathological solutions for lengths exceeding a critical value (unable to resolve the singular limit due to mesh distortion).

변분 적분기 방법: 이산 외미적분을 이용해 에너지를 더 충실히 보존한다. Zhou 등(2018)은 Parker 기하에 적용하여 짧은 계 길이에서는 유한하고 부드러운 전류 층을, 임계치를 초과하는 길이에서는 병리적 해를 발견했다.

### Part VIII: Progressively Thinning Current Layers / 점진적으로 얇아지는 전류 층 (§5.3)

Van Ballegooijen (1988a, b) showed that sequential shear flows of random strength and direction on one boundary produce an exponential decrease in length scales of β, γ (Euler potentials) in the field-line mapping. Equilibria obtained by energy minimization contain current layers that thin exponentially. Mikić et al. (1989) confirmed this with MHD simulations in a 64³ domain: filamentary current structures thin exponentially with time, peak current density grows exponentially (Fig. 6 of paper).

Van Ballegooijen(1988)은 한쪽 경계에서 무작위 강도/방향의 순차적 전단 흐름이 장선 사상의 길이 스케일의 지수함수적 감소를 생성함을 보였다. Mikić 등(1989)은 64³ 영역 MHD 시뮬레이션에서 이를 확인했다: 필라멘트형 전류 구조가 시간에 따라 지수함수적으로 얇아지고 피크 전류 밀도가 지수함수적으로 성장한다.

The reason (Pontin & Hornig 2015): in a force-free field with J = (1/μ₀)∇×B = αB, α is constant along field lines. The length scales of α must match the length scales of the field-line mapping (unless α is constant). Since J_∥ = α|B|, J_∥ inherits the length scales of α — hence of the mapping. As footpoints are shuffled, the mapping length scales shrink exponentially → current layers thin exponentially. This provides strong support for the weak form of Parker's hypothesis (Definition 3). Pontin & Hornig (2015) quantified this via the squashing factor Q (Titov 2007): as the twist parameter κ increases (more braided fields), Q develops progressively larger values in quasi-separatrix layers, matching the thinning of current layers with exponential scaling (Fig. 7 of paper).

Pontin & Hornig(2015)의 이유: 힘없는 장에서 α는 장선을 따라 일정하므로 α의 길이 스케일은 장선 사상의 길이 스케일과 일치해야 한다. 발자국이 뒤섞이면 사상 길이 스케일이 지수함수적으로 축소 → 전류 층이 지수함수적으로 얇아진다. 찌그러짐 인자 Q로 정량화된다.

### Part IX: Flux Braiding Simulations / 자속 Braiding 시뮬레이션 (§6)

Flux braiding simulations shift focus from "does a smooth equilibrium exist?" to "what happens when boundaries are continuously driven?". Typical setup: uniform B₀**e**_z in a rectangular domain, line-tied z-boundaries driven by time-dependent tangential flows (shears or vortices). Full MHD or RMHD equations are solved to a statistically steady state.

자속 braiding 시뮬레이션은 "부드러운 평형이 존재하는가"에서 "경계가 지속적으로 구동될 때 무슨 일이 일어나는가"로 초점을 전환한다.

**Full MHD results** (Galsgaard & Nordlund 1996; Hendrix & Van Hoven 1996; Rappazzo et al. 2007, 2008; Ng et al. 2012; Dahlburg et al. 2012, 2016, 2018):
- Thin ribbons of current form and dissipate throughout the domain, elongated along z
- Free magnetic energy (excess over potential) scales strongly with driving velocity — 1.5% of background for 2% Alfvén driver; 45% for 20% v_A
- Energy dissipation is essentially independent of resistivity (hyper-resistivity regime)
- The statistically steady state is *turbulent* (Hendrix & Van Hoven 1996) with magnetic energy spectrum E_m ∼ k_⊥⁻³/² (Kraichnan slope)

Full MHD 결과: 전류 리본이 형성되고 소산되며, 자유 자기 에너지는 구동 속도에 강하게 스케일하나 저항률과는 거의 독립적이다. 통계적 정상 상태는 난류적이다.

**RMHD results** (Longcope & Sudan 1994; Ng & Bhattacharjee 2008; Ng et al. 2012; Rappazzo et al. 2007, 2008):
- Total magnetic energy, kinetic energy, dissipation scale as η⁻¹/³ (approximately) for η > 10⁻³, but become independent of η for smaller η (when energy dissipation time τ_E > correlation time τ_c)
- Intermittency in dissipation: dN/dE ∼ E⁻¹·⁵ (Dmitruk & Gómez 1997) — compares favorably with observed flare statistics (Crosby et al. 1993; Wheatland 2008)
- Kraichnan turbulence with E_m ∼ k_⊥⁻³/² for τ_A/τ_p < 1; Kolmogorov E_m ∼ k_⊥⁻⁵/³ for τ_A/τ_p > 0.5

The heating rate scales with the dimensionless driving parameter

$$f = \frac{l_\perp v_A}{L u_{ph}},$$

giving (Rappazzo et al. 2008)

$$\varepsilon \sim l_\perp^2\,\rho\,v_A\,u_{ph}^2\left(\frac{l_\perp v_A}{L u_{ph}}\right)^{\frac{\alpha}{\alpha+1}},$$

where α ≈ 5/3 for strong turbulence.

RMHD 결과: 총 자기 에너지, 운동 에너지, 소산이 η⁻¹/³로 스케일하다가 작은 η에서는 독립적이 된다. 간헐성 분포는 관측된 플레어 통계와 유사하다.

**Resistive relaxation** (Pontin et al. 2011; Wilmot-Smith et al. 2010, 2011): Starting from an initially braided state (from ideal relaxation), resistive MHD evolution exhibits "unbraiding" — reconnection cascades produce simpler final fields. Only ~60% of stored magnetic energy is released; the rest remains as large-scale twist. This led Yeates et al. (2010) to discover additional topological constraints beyond Taylor's helicity constraint.

저항성 이완: 초기 braid 상태에서 저항성 MHD 진화는 "풀림(unbraiding)"을 보인다 — 재연결 연쇄가 더 단순한 최종 장을 생성한다. 약 60%의 자기 에너지만 방출되고 나머지는 대규모 비틀림으로 남는다.

**Plasma response** (Dahlburg et al. 2016, 2018; Pontin et al. 2017): When thermal conduction and optically-thin radiation are included, magnetic dissipation leads to a multi-thermal plasma with "strands" of hotter/cooler material. Synthesized emission (Fe XII 195 Å, Fe XV 284 Å) reproduces characteristics of observed active-region loops (Fig. 12). Strikingly, clear crossing of heated strands appears only sometimes despite underlying field lines being braided — reconciling observations of "well-combed" loops with the braiding hypothesis. Recent work (Pontin et al. 2020) shows non-thermal line broadening from braiding-induced turbulence is consistent with observed non-thermal widths.

플라스마 반응: 열 전도와 얇은 복사를 포함하면 자기 소산이 다중-열 플라스마를 생성한다. 합성된 방출은 관측된 활동영역 루프의 특성을 재현한다.

### Part X: Implications of Coronal Magnetic Complexity / 코로나 자기 복잡성의 함의 (§7)

The real corona hosts a web of null points, separatrix surfaces, separator field lines, and quasi-separatrix layers (QSLs) (Priest et al. 2002 "flux-tube tectonics"). Current sheet formation at 2D X-points (Dungey 1953) and 3D null points (Klapper et al. 1996; Pontin & Craig 2005) is well established. Bald patches (Titov et al. 1993) are natural sites. Low (1987, 1989, 1991, 1992), Vekstein et al. (1991) considered perturbations of potential fields with nulls → current sheets extending from photosphere to X-points. All of these sites exceed what is permitted in Parker's strict geometry but contribute to coronal heating.

실제 코로나는 null 점, 분리면, 분리자 장선, 준-분리면 층의 망을 가진다. 2D X-점과 3D null 점에서의 전류 시트 형성은 잘 확립되어 있다.

**Large-scale 3D MHD simulations** (Gudiksen & Nordlund 2002, 2005a, b; Bingert & Peter 2011, 2013; Kanella & Gudiksen 2018; Warnecke & Peter 2019): These simulations include realistic pressure/temperature, thermal conduction, radiative losses, and magneto-convection-like driving. They reproduce the million-degree corona with loop structures (Fig. 13, 14). Heating events follow a power-law distribution in energy; peak energy content near 10¹⁷ J consistent with Parker's nanoflares. Warnecke & Peter (2019) use observed magnetograms to drive the simulation and reproduce AIA 171 emission morphology. These models drastically under-resolve braiding but still reproduce coronal temperatures and fluxes — suggesting the plasma response is insensitive to the exact dissipation mechanism.

대규모 3D MHD 시뮬레이션: 현실적 압력/온도, 열 전도, 복사 손실, 자기-대류-유사 구동을 포함한다. 가열 사건은 Parker 나노플레어와 일치하는 멱함수 분포를 따른다.

### Part XI: Summary and Open Questions / 요약과 미해결 문제 (§8)

**Established**:
- Parker's original perturbation argument (1972) was flawed (van Ballegooijen 1985)
- Simply-connected tangential discontinuities cannot form from smooth boundary motions (Cowley et al. 1997)
- Current layers in equilibria under continued braiding thin exponentially (weak form, Definition 3)
- Flux-braiding simulations produce statistically steady heating sufficient to match coronal losses
- Plasma response to braiding-induced heating reproduces many observational characteristics

**Open**:
- Formal proof of non-existence of smooth equilibria for generic topologies (Definitions 1, 2)
- Scaling of dissipation with true coronal magnetic Reynolds number (R_m ∼ 10¹²–10¹⁴)
- Relative timescales of energy storage and release under true coronal parameters
- Effect of chromospheric magnetic structure and additional physics beyond MHD
- Role of flux-tube tectonics (Priest et al. 2002) compared to pure braiding

확립된 것: Parker의 원래 섭동 논증은 결함이 있었고, 단순 연결 접선 불연속은 형성될 수 없으며, 계속되는 braiding 하에 전류 층은 지수함수적으로 얇아진다.

미해결: 일반 위상에 대한 부드러운 평형의 부존재 형식적 증명, 진짜 코로나 R_m으로의 스케일링, 저장과 방출의 실제 시간 스케일, MHD 너머 물리의 역할.

---

## 3. Key Takeaways / 핵심 시사점

1. **The Parker problem has three precisely-stated forms, and each has a different status / Parker 문제는 세 가지로 정밀하게 진술되며 각기 다른 지위를 가진다** — Definitions 1 (general), 2 (force-free), 3 (weak) should not be conflated. Definition 3 is proven; Definitions 1 and 2 remain open. This framework organizes an otherwise diffuse 50-year debate. 정의 1(일반), 2(힘없는), 3(약한)을 혼동해서는 안 된다. 정의 3은 증명되었고, 정의 1, 2는 미해결이다. 이 틀은 달리 산만한 50년 논쟁을 정리한다.

2. **Van Ballegooijen's 1985 correction undermines Parker's original 1972 proof / Van Ballegooijen의 1985 수정은 Parker의 1972 원래 증명을 약화시킨다** — the RHS term in Parker's first-order equation is O(ε), not O(1), so the constraint ∂b₁/∂z = 0 does not follow. This does NOT invalidate Parker's intuition, but it removes the formal basis for the strong form of his hypothesis. Parker의 일차 방정식의 RHS 항은 O(ε)이지 O(1)이 아니므로 ∂b₁/∂z = 0이 따라오지 않는다. 이것은 Parker 직관을 무효화하지 않지만 가설 강한 형태의 형식적 근거를 제거한다.

3. **Simply-connected tangential discontinuities are impossible under smooth boundary motions / 단순 연결 접선 불연속은 부드러운 경계 운동 하에 불가능하다** — the Cowley-et-al. argument ∮B[1-cos θ(l)]dl = 0 is definitive for this class. Branching (Y-type) or more complex topologies are not excluded and may be the relevant current sheet geometry. Cowley 등의 논증 ∮B[1-cos θ(l)]dl = 0은 이 종류에 대해 결정적이다. 분기(Y형)나 더 복잡한 위상은 배제되지 않으며 관련 전류 시트 기하일 수 있다.

4. **Exponential thinning of current layers is proven and suffices for coronal heating / 전류 층의 지수함수적 얇아짐은 증명되었으며 코로나 가열에 충분하다** — both theoretically (via α being constant along field lines, inheriting mapping length scales) and computationally (Mikić et al. 1989; Pontin & Hornig 2015). This weak form provides the theoretical foundation for nanoflare heating without needing to settle the singular-equilibrium question. 이론적으로(α가 장선을 따라 일정해 사상 길이 스케일을 상속)도 계산적으로도 증명되었다. 약한 형태는 특이 평형 문제를 해결하지 않고도 나노플레어 가열에 이론적 기초를 제공한다.

5. **Flux braiding simulations reach a statistically steady state whose properties match coronal heating requirements / 자속 braiding 시뮬레이션은 코로나 가열 요구를 만족하는 통계적 정상 상태에 도달한다** — energy dissipation is independent of resistivity for R_m ≳ 10³; free magnetic energy exceeds kinetic by >10×; dN/dE ∼ E⁻¹·⁵ matches observed flare statistics. The driving pattern affects event characteristics but not the time-averaged heating. R_m ≳ 10³에서 에너지 소산은 저항률과 독립적이고, 자유 자기 에너지가 운동 에너지를 10배 초과하며, 사건 분포가 관측된 플레어 통계와 일치한다.

6. **Plasma response is surprisingly insensitive to detailed dissipation mechanism / 플라스마 반응은 세부 소산 메커니즘에 놀라울 만큼 둔감하다** — large-scale 3D simulations with RMHD-level (or coarser) resolution of braiding reproduce coronal emission morphology, temperature distributions, and DEM profiles (Warnecke & Peter 2019; Dahlburg et al. 2016). This makes observational discrimination between heating mechanisms challenging. 대규모 3D 시뮬레이션은 braiding의 RMHD 수준 해상도에서도 코로나 방출 형태, 온도 분포, DEM 프로파일을 재현한다. 이는 가열 메커니즘 간의 관측적 구별을 어렵게 만든다.

7. **Absence of visible "crossing strands" does NOT rule out braiding / 보이는 "교차 가닥"의 부재가 braiding을 배제하지 NOT** — Pontin et al. (2017) and Dahlburg et al. (2018) show synthesized emission from braided fields can appear as "well-combed" loops despite underlying topology being complex. Non-thermal line broadening (Pontin et al. 2020) is a more diagnostic signature. 합성 방출은 기저 위상이 복잡해도 "잘 빗겨진(well-combed)" 루프로 나타날 수 있다. 비-열적 선 넓어짐이 더 진단적 특징이다.

8. **The real corona has topological complexity (nulls, QSLs) beyond the Parker geometry / 실제 코로나는 Parker 기하를 넘는 위상적 복잡성(null, QSL)을 가진다** — Priest et al.'s "flux-tube tectonics" and magneto-frictional coronal-scale simulations suggest heating in the real Sun is a blend of Parker-style braiding dissipation plus current sheet formation at topological features. Both contribute; the relative importance remains observational/modeling frontier. 실제 태양의 가열은 Parker 스타일 braiding 소산 + 위상 특징에서의 전류 시트 형성의 혼합이다. 상대적 중요성은 관측/모델링 전선이다.

---

## 4. Mathematical Summary / 수학적 요약

### (A) Force-free field equations / 힘없는 장 방정식
$$\mathbf{J}\times\mathbf{B} = 0 \;\Leftrightarrow\; \nabla\times\mathbf{B} = \alpha(\mathbf{x})\,\mathbf{B},\qquad \mathbf{B}\cdot\nabla\alpha = 0.$$
- **J** = (1/μ₀)∇×**B**: current density / 전류 밀도
- α: force-free parameter, constant along each field line (from divergence of the above) / 힘없는 매개변수, 각 장선을 따라 일정
- α = const globally → "Beltrami" or linear force-free field / 전역 일정 → Beltrami 또는 선형 힘없는 장

### (B) Ideal induction equation / 이상 유도 방정식
$$\frac{\partial\mathbf{B}}{\partial t} = \nabla\times(\mathbf{v}\times\mathbf{B}).$$
Flux-freezing: field-line connectivity is preserved (Alfvén 1943). Central constraint in the Parker problem — topology must be maintained during relaxation.

자속-동결: 장선 연결이 보존된다(Alfvén 1943). Parker 문제의 중심 제약 — 이완 중 위상이 유지되어야 한다.

### (C) Magnetic energy / 자기 에너지
$$W = \int_V \frac{B^2}{8\pi}\,d^3x \quad\text{(Gaussian)} = \int_V \frac{B^2}{2\mu_0}\,d^3x \quad\text{(SI)}.$$
Free magnetic energy = W(current field) - W(potential field with same boundary). This excess drives heating when released.

자유 자기 에너지 = W(현재 장) - W(같은 경계의 잠재장). 방출될 때 이 초과분이 가열을 구동한다.

### (D) Poynting flux into the corona / 코로나로의 Poynting 플럭스
$$\mathbf{F} = -\frac{(\mathbf{v}\cdot\mathbf{B})\mathbf{B}}{\mu_0} + \frac{B^2\mathbf{v}}{\mu_0},\qquad F_P = -\int_S \frac{1}{\mu_0}(\mathbf{v}_t\cdot\mathbf{B}_t)\,B_n\,dA.$$
Observational value ~5×10⁴ W m⁻² = 5×10⁷ erg cm⁻² s⁻¹. Active region loss ~10⁷ erg cm⁻² s⁻¹ (= 10⁴ W m⁻²) is matched by this injection.

관측된 값 ~5×10⁴ W m⁻² = 5×10⁷ erg cm⁻² s⁻¹. 활동영역 손실 ~10⁷ erg cm⁻² s⁻¹은 이 주입으로 충족된다.

### (E) Parker's perturbation analysis (Gaussian units) / Parker 섭동 분석
$$\mathbf{B} = B_0\mathbf{e}_z + \mathbf{b},\qquad \mathbf{b}=\sum_{n\ge 1}\epsilon^n\mathbf{b}_n,\qquad p = \sum_{n\ge 0}\epsilon^n p_n.$$
First-order MHS equation:
$$\nabla\!\left(p_1 + \frac{B_0 b_{1z}}{4\pi}\right) - \frac{B_0}{4\pi}\frac{\partial\mathbf{b}_1}{\partial z} = 0.$$
Parker: RHS must vanish separately → ∂b₁/∂z = 0. Van Ballegooijen correction: RHS is O(ε), so it moves to next order →
$$\nabla\!\left(p_1 + \frac{B_0 b_{1z}}{4\pi}\right) = 0.$$
This allows z-dependent smooth equilibria.

Parker는 RHS가 따로 사라져야 한다고 주장(→ ∂b₁/∂z = 0). Van Ballegooijen 수정: RHS는 O(ε)이므로 차수 밀려남. 이는 z-의존 부드러운 평형을 허용한다.

### (F) Cowley-et-al. tangential discontinuity argument / Cowley 등 접선 불연속 논증
For a simply-connected current sheet in a force-free field between line-tied boundaries, constructing loops 𝒞₁∪𝒞₂' and 𝒞₂∪𝒞₁' on either side of the sheet, and using |**B**₁(l)|=|**B**₂(l)|=B(l) across a tangential discontinuity:
$$\oint_{\mathcal{C}_1+\mathcal{C}_2} B(l)\bigl[1-\cos\theta(l)\bigr]\,dl = 0.$$
Integrand ≥ 0 everywhere → θ(l) = 0 → no tangential discontinuity. Contradicts hypothesis.

피적분자 ≥ 0 → θ(l) = 0 → 접선 불연속 없음. 가설과 모순.

### (G) Lagrangian field representation (pull-back) / 라그랑주 장 표현(풀백)
$$\mathbf{x}^*(\mathbf{B}(\mathbf{x},t),t) = \mathbf{B}(\mathbf{X},0),\qquad B_i(\mathbf{x},t) = \frac{1}{\Delta}\sum_{j=1}^3\frac{\partial x_i}{\partial X_j}B_j(\mathbf{X},0).$$
Δ = det(∂x_i/∂X_j): Jacobian determinant. Field topology conserved exactly — no numerical slippage. Magneto-frictional evolution:
$$\frac{D\mathbf{x}}{Dt} = \gamma(\nabla\times\mathbf{B})\times\mathbf{B} - \gamma\nabla p.$$

장 위상이 정확히 보존되고 수치 미끄러짐이 없다.

### (H) Reduced MHD equations / 축소 MHD 방정식
Ordering: Bz ∼ 1, |B⊥| ∼ ε, ∂/∂z ∼ ε. **B** = B_{z0}**e**_z + **e**_z×∇_⊥ψ, **v** = **v**_⊥ = **e**_z×∇_⊥φ.
$$\rho_0\!\left(\frac{\partial\Omega}{\partial t} + \mathbf{v}_\perp\cdot\nabla\Omega\right) = \mathbf{B}\cdot\nabla J_z + \nu\nabla_\perp^2\Omega,$$
$$\frac{\partial\psi}{\partial t} + \mathbf{v}_\perp\cdot\nabla\psi = -B_{z0}\frac{\partial\phi}{\partial z} + \eta\nabla_\perp^2\psi,$$
where Ω = ∇_⊥²φ (vorticity), J_z = ∇_⊥²ψ (parallel current). Only shear Alfvén waves propagate in z. Parallel to 2D Euler equation.

코로나 루프의 대종횡비 근사. 2D Euler 방정식과 병렬.

### (I) Turbulence spectra in statistically steady state / 통계적 정상 상태의 난류 스펙트럼
- Kraichnan: $E_m(k_\perp) \sim k_\perp^{-3/2}$ (dominant in braiding simulations)
- Kolmogorov: $E_m(k_\perp) \sim k_\perp^{-5/3}$ (longer loops, stronger axial field)

Event distribution: $\dfrac{dN}{dE} \sim E^{-1.5}$ matches observed flare statistics.

### (J) Heating rate scaling (Rappazzo et al. 2008) / 가열률 스케일링
$$\varepsilon \sim l_\perp^2 \rho v_A u_{ph}^2\left(\frac{l_\perp v_A}{L u_{ph}}\right)^{\frac{\alpha}{\alpha+1}},\qquad \alpha \approx 5/3.$$
For typical coronal parameters (l_⊥ ∼ 10³ km, v_A = 1000 km/s, u_ph = 1 km/s, L = 10⁵ km), this gives heating rates of order 10⁷ erg cm⁻² s⁻¹, consistent with active-region requirements.

전형적 코로나 매개변수로 10⁷ erg cm⁻² s⁻¹ 수준의 가열률을 주며 활동영역 요구와 일치한다.

### (K) Worked numerical example — coronal heating requirement / 작업된 수치 예제 — 코로나 가열 요구
Active-region flux tube: B = 100 G = 0.01 T, l_⊥ = 10⁶ m, v_{ph} = 500 m/s, v_A = 10⁶ m/s, L = 10⁸ m, ρ ≈ 1.67×10⁻¹² kg/m³.
Poynting flux (Parker angle ~14°): F_P ∼ B_⊥ B_z v_ph/μ₀ ≈ (0.25)(0.01)²(500)/(4π×10⁻⁷) ≈ 10⁴ W m⁻² = 10⁷ erg cm⁻² s⁻¹ — matches active-region loss.

Braiding timescale for random footpoint walk of step δ ∼ v_ph τ_corr with τ_corr ∼ 5 min:
$$\tau_{\text{braid}} \sim \frac{l_\perp^2}{v_{ph}\delta} = \frac{(10^6)^2}{500 \cdot 500 \cdot 300} \approx 10^4 \text{ s} \sim 3 \text{ hr}.$$

Current sheet thickness at dissipation threshold (Sweet-Parker): δ_SP ∼ L/√S with S = L v_A/η. For S ∼ 10¹⁰ in the corona: δ_SP ∼ 10⁸/10⁵ m = 10³ m = 1 km — far below observational resolution.

활동영역 자속관 예: B = 100 G, l_⊥ = 10⁶ m, v_{ph} = 500 m/s. Poynting 플럭스 ∼ 10⁷ erg cm⁻² s⁻¹으로 활동영역 손실과 일치. Braiding 시간 척도 ∼ 3 시간. Sweet-Parker 전류 시트 두께 ∼ 1 km으로 관측 해상도 이하.

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1940s ─ Edlén, Grotrian discover million-degree corona
         │
1943  ─ Alfvén: frozen-in theorem (ideal MHD)
         │
1957  ─ Sweet-Parker reconnection model
         │
1958  ─ Kruskal et al.: hydromagnetic instability (kink)
         │
1965  ─ Kraichnan: k^{-3/2} MHD turbulence
         │
1972 ◆ Parker: topological dissipation hypothesis
         │    |                  "ApJ 174:499"
         │    └── sparks 50-year debate
         │
1974  ─ Strauss, Kadomtsev-Pogutse: Reduced MHD
         │
1977  ─ Withbroe & Noyes: coronal heating requirements
         │
1985 ● van Ballegooijen: ε-expansion error in Parker's proof [Paper #6]
         │
1986  ─ Craig & Sneyd: magneto-frictional relaxation method
         │
1988  ─ Parker: "nanoflare" terminology, 10^24-erg estimate
         │  van Ballegooijen: exponential thinning of current layers
         │
1989  ─ Parker: optical analogy (Fermat-like principle)
         │  Mikić et al.: filamentary current simulation (64^3)
         │
1994  ─ Parker: "Magnetostatic Theorem" monograph
         │  Longcope & Strauss: coalescence with line-tying
         │  Longcope & Sudan: first RMHD braiding simulation
         │
1996  ─ Galsgaard & Nordlund: first full-MHD braiding simulation
         │  Hendrix & Van Hoven: turbulent cascade in braiding
         │
1997  ─ Cowley et al.: ∮B[1-cos θ]dl=0 argument
         │  Dmitruk & Gómez: dN/dE ~ E^{-1.5}
         │
1998  ─ Ng & Bhattacharjee: RMHD uniqueness theorem
         │
2002  ─ Gudiksen & Nordlund: coronal-scale 3D MHD
         │  Priest et al.: flux-tube tectonics
         │
2007-08 ─ Rappazzo et al.: RMHD scaling laws
         │
2009  ─ Janse & Low; Wilmot-Smith et al.: braided relaxation
         │
2010  ─ Yeates et al.: topological constraints beyond Taylor
         │
2011  ─ Pontin et al.: resistive relaxation of braids
         │  Shelyag et al.: convective Poynting flux
         │
2014 ● Reale: Coronal Loops Living Review [Paper #39]
         │  Candelaresi et al.: mimetic Lagrangian relaxation
         │
2015  ─ Pontin & Hornig: exponential thinning via squashing factor
         │
2016  ─ Dahlburg et al.: braiding with thermodynamics
         │
2018  ─ Zhou et al.: variational integrator for Parker geometry
         │  Warnecke & Peter: data-driven coronal simulation
         │
2020 ◆ Pontin & Hornig: this review [Paper #69]
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| #6 van Ballegooijen (1985) | Directly corrects Parker's (1972) ε-expansion error; shows smooth equilibria exist for smooth small perturbations; provides Lagrangian framework later extended by Zweibel & Li (1987) / Parker의 섭동 오류를 직접 수정; 부드러운 작은 섭동에 대해 부드러운 평형이 존재함을 보임 | Pivotal — this is the single most important counter to Parker's original argument. Discussed in detail in §4.2. / 결정적 — Parker 원래 논증에 대한 가장 중요한 반박 |
| #39 Reale (2014) | Coronal Loops Living Review — compiles observed loop properties (temperatures, densities, lifetimes, emission characteristics) against which braiding-heating simulations must be validated / 코로나 루프 리뷰 — 관측된 루프 특성을 집대성 | Observational counterpart — Reale's review provides the constraints that plasma-response simulations in §6.6 must match. Pontin et al. (2017), Dahlburg et al. (2016) cited in §6.6 compare their synthetic emission to this framework. / 관측 대응물 — 플라스마 반응 시뮬레이션이 일치시켜야 할 제약 |
| Parker (1972, 1988, 1994) | The three foundational papers defining the problem, the nanoflare mechanism, and the "Magnetostatic Theorem" / 문제를 정의하고 나노플레어 메커니즘을 제안하며 "자기정적 정리"를 담은 세 기초 논문 | Foundational — this review is essentially a systematic evaluation of these three works after nearly 50 years / 기초적 — 본 리뷰는 이 세 작업에 대한 체계적 평가 |
| Cowley, Longcope & Sudan (1997) | Provides the ∮B[1-cos θ(l)]dl=0 argument proving simply-connected tangential discontinuities cannot form under smooth boundary motions / 단순 연결 접선 불연속이 부드러운 경계 운동 하에 형성될 수 없음을 증명하는 논증 제공 | Key theoretical result — the cleanest mathematical argument in the review; §4.4.2 / 핵심 이론 결과 — 리뷰에서 가장 깨끗한 수학적 논증 |
| Ng & Bhattacharjee (1998) | RMHD uniqueness theorem: at most one smooth equilibrium per footpoint mapping; if unstable, relaxation must produce non-smooth states / RMHD 유일성 정리 | Supports Parker's hypothesis in RMHD limit; applicability to full 3D MHD uncertain / RMHD 한계에서 Parker 가설 지지; 전체 3D MHD 적용성 불확실 |
| Rappazzo et al. (2007, 2008) | Standard reference for RMHD braiding scaling laws; heating rate formula ε ∼ f^{α/(α+1)} and turbulent spectra / RMHD braiding 스케일링 법칙의 표준 참고문헌 | Quantitative link between simulation and observation; §6.3 / 시뮬레이션과 관측 간의 정량적 연결 |
| Wilmot-Smith et al. (2009, 2010, 2011); Pontin et al. (2011) | Ideal braided initial state + resistive relaxation shows cascading reconnection releases ~60% of stored energy; additional topological invariants beyond Taylor's (Yeates et al. 2010) / 이상 braid 초기 상태 + 저항성 이완; Taylor 너머 위상 불변량 | Reveals the dynamics of energy release from a braided initial condition / braid 초기 조건으로부터 에너지 방출의 동역학을 드러냄 |
| Gudiksen & Nordlund (2002, 2005); Warnecke & Peter (2019) | Large-scale 3D MHD simulations driven by photospheric granulation-like flows produce observationally-realistic coronal loops and emission / 광구 과립-유사 흐름으로 구동되는 대규모 3D MHD | Proves that coarse-resolution braiding models can reproduce coronal observations; §7.2 / 거친 해상도 braiding 모델도 코로나 관측을 재현할 수 있음을 증명 |
| Priest, Heyvaerts & Title (2002) | Flux-tube tectonics: coronal heating via current sheet formation at the web of nulls, separators, and QSLs rather than pure braiding / 순수 braiding 대신 null, separator, QSL 망에서의 전류 시트 형성을 통한 코로나 가열 | Extension/complement to Parker's hypothesis; §7.1 / Parker 가설에 대한 확장/보완 |

---

## 7. References / 참고문헌

- Alfvén, H. (1943). "On the existence of electromagnetic-hydrodynamic waves". *Ark Mat Astron Fys* 29:1–7.
- Bineau, M. (1972). "Existence of force-free magnetic-fields". *Commun Pure Appl Math* 25(1):77.
- Candelaresi, S., Pontin, D. I., & Hornig, G. (2014). "Mimetic methods for Lagrangian relaxation of magnetic fields". *SIAM J Sci Comput* 36:B952–B968.
- Cowley, S. C., Longcope, D. W., & Sudan, R. N. (1997). "Current sheets in MHD turbulence". *Phys Rep* 283(1–4):227–251. DOI:10.1016/S0370-1573(96)00064-6
- Craig, I. J. D., & Sneyd, A. D. (1986). "A dynamic relaxation technique for determining the structure and stability of coronal magnetic fields". *Astrophys J* 311:451–459.
- Dahlburg, R. B., Einaudi, G., Taylor, B. D., Ugarte-Urra, I., Warren, H. P., Rappazzo, A. F., & Velli, M. (2016). "Observational signatures of coronal loop heating and cooling driven by footpoint shuffling". *Astrophys J* 817(1):47.
- Galsgaard, K., & Nordlund, Å. (1996). "Heating and activity of the solar corona. I". *J Geophys Res* 101:13445–13460.
- Gudiksen, B. V., & Nordlund, Å. (2005). "An ab initio approach to solar coronal loops". *Astrophys J* 618:1020.
- Low, B. C. (2010b). "The Parker magnetostatic theorem". *Astrophys J* 718:717–723.
- Mikić, Z., Schnack, D. D., & Van Hoven, G. (1989). "Creation of current filaments in the solar corona". *Astrophys J* 338:1148.
- Ng, C. S., & Bhattacharjee, A. (1998). "Nonequilibrium and current sheet formation in line-tied magnetic fields". *Phys Plasmas* 5:4028–4040.
- Parker, E. N. (1972). "Topological dissipation and the small-scale fields in turbulent gases". *Astrophys J* 174:499. DOI:10.1086/151510
- Parker, E. N. (1988). "Nanoflares and the solar X-ray corona". *Astrophys J* 330:474–479.
- Parker, E. N. (1994). *Spontaneous Current Sheets in Magnetic Fields*. Oxford University Press.
- Pontin, D. I., Candelaresi, S., Russell, A. J. B., & Hornig, G. (2016). "Braided magnetic fields: equilibria, relaxation and heating". *Plasma Phys Control Fusion* 58(5):054008.
- Pontin, D. I., & Hornig, G. (2015). "The structure of current layers and degree of field-line braiding in coronal loops". *Astrophys J* 805:47.
- Pontin, D. I., & Hornig, G. (2020). "The Parker problem: existence of smooth force-free fields and coronal heating". *Living Rev Sol Phys* 17:5. DOI:10.1007/s41116-020-00026-5 [this paper]
- Priest, E. R., Heyvaerts, J. F., & Title, A. M. (2002). "A flux-tube tectonics model for solar coronal heating". *Astrophys J* 576:533–551.
- Rappazzo, A. F., Velli, M., Einaudi, G., & Dahlburg, R. B. (2008). "Nonlinear dynamics of the Parker scenario for coronal heating". *Astrophys J* 677:1348.
- Reale, F. (2014). "Coronal loops: observations and modelling of confined plasma". *Living Rev Sol Phys* 11:4. [Paper #39]
- van Ballegooijen, A. A. (1985). "Force free fields and coronal heating part I. The formation of current sheets". *Astrophys J* 298:421. [Paper #6]
- van Ballegooijen, A. A. (1988a). "Force free fields and coronal heating part I". *Geophys Astrophys Fluid Dyn* 41(3–4):181–211.
- Warnecke, J., & Peter, H. (2019). "Data-driven model of the solar corona above an active region". *Astron Astrophys* 624:L12.
- Wilmot-Smith, A. L. (2015). "An overview of flux braiding experiments". *Philos Trans R Soc London, Ser A* 373:20140265.
- Withbroe, G. L., & Noyes, R. W. (1977). "Mass and energy flow in the solar chromosphere and corona". *Annu Rev Astron Astrophys* 15(1):363–387.
- Yeates, A. R., Bianchi, F., Welsch, B. T., & Bushby, P. J. (2014). "The coronal energy input from magnetic braiding". *Astron Astrophys* 564:A131–10.
- Zhou, Y., Huang, Y. M., Qin, H., & Bhattacharjee, A. (2018). "Constructing current singularity in a 3D line-tied plasma". *Astrophys J* 852:3.
- Zweibel, E. G., & Li, H. S. (1987). "The formation of current sheets in the solar atmosphere". *Astrophys J* 312:423–430.
