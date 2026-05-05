---
title: "Solar Force-Free Magnetic Fields"
authors: [Thomas Wiegelmann, Takashi Sakurai]
year: 2021
journal: "Living Reviews in Solar Physics"
doi: "10.1007/s41116-020-00027-4"
topic: Living_Reviews_in_Solar_Physics
tags: [force-free, NLFFF, coronal-magnetic-field, extrapolation, MHD, optimization, Grad-Rubin, magnetic-helicity]
status: completed
date_started: 2026-04-23
date_completed: 2026-04-23
---

# 72. Solar Force-Free Magnetic Fields / 태양 무력 자기장

---

## 1. Core Contribution / 핵심 기여

**English:** Wiegelmann & Sakurai (2021) is the updated *Living Review* on force-free magnetic-field modeling of the solar corona. The core contribution is a systematic synthesis of every method — analytic, semi-analytic, and numerical — used to extrapolate the coronal magnetic field $\mathbf{B}$ from photospheric vector-magnetogram boundary conditions under the assumption that the Lorentz force vanishes, $\mathbf{j}\times\mathbf{B}=0$. The review develops the hierarchy: potential fields (α = 0) → linear force-free fields (α = const) → nonlinear force-free fields (α = α(x,y,z), constant along field lines). It critically assesses five classes of NLFFF codes (upward integration, Grad-Rubin, MHD relaxation, optimization, boundary-element), covers the 180° azimuth ambiguity problem that afflicts photospheric data, reviews the consistency criteria (Aly 1989 integral relations) that force-free boundary conditions must satisfy, and presents the NLFFF consortium benchmarks that have guided the field since 2004.

**Korean:** Wiegelmann & Sakurai (2021)은 태양 코로나 무력(force-free) 자기장 모델링에 대한 *Living Reviews* 업데이트 판이다. 핵심 기여는 로렌츠 힘이 0이라는 가정 $\mathbf{j}\times\mathbf{B}=0$ 하에 광구 벡터 자력도 경계조건으로부터 코로나 자기장 $\mathbf{B}$를 외삽하는 모든 방법 — 해석적·반해석적·수치적 — 을 체계적으로 정리한 것이다. 저자들은 포텐셜장(α=0) → 선형 무력장(α=상수) → 비선형 무력장(α는 공간의 함수로서 field line 따라 일정)의 계층을 전개한다. NLFFF 코드 다섯 가지(upward integration, Grad-Rubin, MHD relaxation, optimization, boundary-element)를 비판적으로 평가하며, 광구 자료에 내재하는 180° 방위각 모호성 문제, 무력장 경계조건이 충족해야 할 일관성 조건(Aly 1989의 적분 관계), 2004년 이래 이 분야를 이끌어 온 NLFFF consortium 벤치마크 결과를 총망라한다. 2012년 초판 대비 약 70편의 추가 참고문헌과 6개 새 그림이 포함된 대규모 개정판이다.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction — Why Force-Free? / 서론 — 왜 무력장 근사인가?

**English:** §1 motivates the force-free assumption via the plasma beta parameter
$$
\beta = \frac{2\mu_0 p}{B^2}.
$$
Gary's (2001) plasma-β model (Fig. 2 of the paper) shows β has a sandwich structure: β > 1 in the photosphere and upper corona (> ~2.5 R☉), but β ≪ 1 between ~2 Mm and ~2 R☉ — the region of coronal interest. With β ≪ 1, magnetic pressure dominates gas pressure and gravity, and the leading-order force balance demands $\mathbf{j}\times\mathbf{B}=0$. The equivalent forms are
$$
(\nabla\times\mathbf{B})\times\mathbf{B}=0, \qquad \nabla\cdot\mathbf{B}=0.
$$
Solutions split into two branches: (i) $\nabla\times\mathbf{B}=0$ (current-free potential fields) or (ii) $\mathbf{B}\parallel\nabla\times\mathbf{B}$ (true force-free fields). A low-β value is *sufficient* but not *necessary* — if the pressure gradient is balanced by gravity, $\nabla p = -\rho\nabla\Psi$, force-free configurations can survive at high β (Neukirch 2005). Potential-field-source-surface (PFSS) models (Schatten et al. 1969) with a source surface at ~2.5 R☉ remain the most widely used global model, though they cannot capture active-region free energy. (pp. 4-7)

**Korean:** §1은 플라즈마 베타 $\beta=2\mu_0 p/B^2$로 force-free 가정의 정당성을 제시한다. Gary(2001) 모델에 따르면 β는 샌드위치 구조: 광구와 2.5 R☉ 이상 고고도 코로나에서는 β > 1이나 그 사이 ~2 Mm–2 R☉ 구간에서 β ≪ 1. 이 영역에서 자기압이 기체압·중력을 압도하여 선행 근사에서 $\mathbf{j}\times\mathbf{B}=0$이 성립한다. 해는 (i) 전류 없는 포텐셜장 또는 (ii) 진정한 force-free 장의 두 갈래로 나뉜다. 낮은 β는 *충분*조건이나 *필요*조건은 아니다 — 압력 경사가 중력과 평형이면 고-β에서도 force-free 가능. 전역 규모에서는 Schatten 등(1969)의 PFSS 모델(2.5 R☉ source surface에서 field line이 방사상 됨)이 여전히 표준이나 활성영역 자유 에너지는 담지 못한다.

### Part II: Linear Force-Free Fields / 선형 무력장 (§2)

**English:** When α = constant, the force-free equation becomes
$$
\nabla\times\mathbf{B}=\alpha\mathbf{B}, \quad \nabla\cdot\mathbf{B}=0.
$$
Taking the curl and using $\nabla\cdot\mathbf{B}=0$ gives the vector Helmholtz equation
$$
\Delta\mathbf{B}+\alpha^2\mathbf{B}=0,
$$
which admits separation-of-variables, Green's-function (Chiu & Hilton 1977), or Fourier solutions (Alissandrakis 1981). Seehafer (1978) wrote the Cartesian Fourier solution:
$$
B_z(x,y,z) = \sum_{m,n=1}^\infty C_{mn}\exp(-r_{mn}z)\sin\!\left(\tfrac{\pi m x}{L_x}\right)\sin\!\left(\tfrac{\pi n y}{L_y}\right),
$$
with $\lambda_{mn}=\pi^2(m^2/L_x^2+n^2/L_y^2)$ and $r_{mn}=\sqrt{\lambda_{mn}-\alpha^2}$. For $r_{mn}$ to remain real, $\alpha^2$ must not exceed $\alpha_{\max}^2=\pi^2(1/L_x^2+1/L_y^2)$. Normalizing by harmonic-mean length $L$ gives $-\sqrt{2}\pi < \alpha < \sqrt{2}\pi$ (in units of 1/L). The horizontal components $B_x, B_y$ follow from the curl relation. LFFF requires only the line-of-sight magnetogram $B_z(x,y)$ and a single global value of α.

How is α determined? Several strategies:
- **Photospheric fit**: averaged $\langle\alpha\rangle = \sum\mu_0 j_z \,{\rm sign}(B_z)/\sum|B_z|$ from the vertical current (Pevtsov et al. 1994, Hagino & Sakurai 2004).
- **Coronal-loop fit** (Carcedo et al. 2003): project model field lines onto EUV/XRT loop images and vary α to minimize the misalignment. Required input: identified loops with both footpoints (hard for EUV). Automated feature-recognition (Aschwanden 2008a, Inhester et al. 2008) partially addresses this.
- **Chromospheric α** (Gosain et al. 2014, AR11084): chromospheric α (~0.4) differs from coronal α (~0.23), so a single α cannot describe both layers → inherent breakdown of LFFF.
- **Neural networks** (Benson et al. 2019): train on pseudo-coronal-loop images from LFFF configurations; method works for synthetic but untested on real data.

**Key lesson**: if scatter in optimal α among field lines is small, LFFF is self-consistent; otherwise NLFFF is required. Malanushenko et al. (2009) showed quantities like twist and loop heights can be estimated to ~15% and ~5% accuracy using LFFF even when it is not truly self-consistent. (pp. 8-14)

**Korean:** α = 상수일 때 $\nabla\times\mathbf{B}=\alpha\mathbf{B}$의 curl을 취하면 벡터 헬름홀츠 방정식 $\Delta\mathbf{B}+\alpha^2\mathbf{B}=0$을 얻는다. Seehafer(1978)의 카르테시안 푸리에 해는 위와 같으며, $r_{mn}=\sqrt{\lambda_{mn}-\alpha^2}$이 실수여야 하므로 $\alpha^2 \le \alpha_{\max}^2=\pi^2(1/L_x^2+1/L_y^2)$로 제한. L을 조화평균 길이로 정규화 시 $-\sqrt{2}\pi<\alpha<\sqrt{2}\pi$. LFFF는 LOS 자력도 $B_z(x,y)$와 전역 α 하나만 있으면 된다. α 결정 방법: (1) 광구 수직전류로부터 평균 $\langle\alpha\rangle$ 계산, (2) 코로나 loop 영상에 field line 투영하여 fit(Carcedo 2003), (3) 채층 α와 코로나 α 비교(Gosain 2014; AR11084에서 0.4 vs 0.23 → LFFF 붕괴), (4) 신경망(Benson 2019, 아직 초기 단계). 핵심 교훈: field line별 α 분산이 작아야 LFFF 적용 가능. 그렇지 않아도 twist, loop 높이 등은 각각 ~15%, ~5% 오차로 추정 가능(Malanushenko 2009).

### Part III: Analytic Nonlinear Force-Free Solutions / 반해석적 비선형 해 (§3)

**English:** Exact 3D NLFFF solutions are nearly impossible; symmetry reductions give tractable cases. In 2D Cartesian geometry the force-free equation reduces to the Grad-Shafranov equation
$$
\Delta A = -\lambda^2 f(A),
$$
where $A$ is a flux function. For axisymmetric spherical geometry (Low & Lou 1990), $\mathbf{B} = (1/r\sin\theta)[(1/r)\,\partial A/\partial\theta\,\mathbf{e}_r - \partial A/\partial r\,\mathbf{e}_\theta] + Q\mathbf{e}_\varphi$, giving
$$
\frac{\partial^2 A}{\partial r^2} + \frac{1-\mu^2}{r^2}\frac{\partial^2 A}{\partial \mu^2} + Q\frac{dQ}{dA}=0, \quad \mu=\cos\theta.
$$
With the ansatz $Q(A)=\lambda A^{1+1/n}$ and $A(r,\theta)=P(\mu)/r^n$, a nonlinear ODE for $P(\mu)$ results, solved numerically as an eigenvalue problem (Wolfson 1995). The Low-Lou solution is the canonical NLFFF test case: rotate the symmetry axis, extract $\mathbf{B}$ on a planar cut, feed as boundary condition to the 3D code, and compare the reconstructed volume to the exact solution (Schrijver et al. 2006).

The Titov-Démoulin (1999) equilibrium constructs a toroidal current-carrying flux rope embedded in a potential background. A ring current $I$ with minor/major radii $a/R$, two monopoles of strength $\pm q$ at distance $L$ on the symmetry axis, and a line current $I_0$ along the axis combine into a stable or unstable equilibrium. The parameter
$$
B_0 \approx \frac{\mu_0 I_0}{2\pi R}
$$
controls stability: instability (kink) occurs for $R \gtrsim \sqrt{2}\,L$. This configuration is the standard test for eruption-onset studies (Török & Kliem 2005). (pp. 14-17)

**Korean:** 3D NLFFF 정확해는 거의 불가능하고 대칭성 환원으로만 가능하다. 2D Cartesian에서 force-free equation은 Grad-Shafranov 방정식으로 환원된다. 구면 축대칭에서 Low & Lou(1990)는 $Q(A)=\lambda A^{1+1/n}$, $A(r,\theta)=P(\mu)/r^n$ 가정으로 $P(\mu)$ ODE를 유도하여 수치적 고유값 문제로 푼다. 대칭축을 회전시켜 평면 자력도를 얻어 NLFFF 코드 벤치마크로 사용(Schrijver 2006, Fig. 6 원논문). Titov-Démoulin(1999)은 ring current $I$, 반지름 $a/R$, 쌍극 monopole $\pm q$, 축전류 $I_0$가 조합된 flux rope 평형으로 $R \gtrsim \sqrt{2}L$일 때 kink 불안정. CME 폭발 연구의 표준 테스트 (Török & Kliem 2005).

### Part IV: Azimuth Ambiguity & Consistency of Measurements / 방위각 모호성과 측정 일관성 (§4)

**English:** Vector magnetograms (Hinode/SOT-SP, SDO/HMI, NSO/SOLIS, Huairou, Mees, Mitaka) invert Stokes profiles (Unno 1956, Rachkovsky 1967) to produce three components. The LOS component is trustworthy; the horizontal components contain a **180° azimuth ambiguity** — only the azimuth squared is determined by linear polarization. Metcalf et al. (2006) compared 13 algorithms on a synthetic flux rope (Fan & Gibson 2004):

| Algorithm / 알고리즘 | Performance (% correct) |
|---|---|
| Acute-angle (potential reference) | 64-75% |
| Improved acute-angle (LFFF ref, Wang 1997) | 87% |
| Uniform shear method (Moon 2003) | 83% |
| Magnetic pressure gradient (Cuperman 1993) | 74% |
| Structure minimization (Georgoulis 2004) | 22% (worse than random!) |
| Non-potential method (Georgoulis 2005 / updated) | 70 / 90% |
| Pseudo-current (Gary & Démoulin 1995) | 78% |
| U. Hawai'i iterative | 97% |
| Minimum energy (Metcalf 1994) | 98% (linear) / 100% (nonlinear) |

Minimum-energy minimizes $E=\sum(|\nabla\cdot\mathbf{B}|+|\mathbf{j}|)^2$ via simulated annealing. It is the SDO/HMI disambiguation standard above 150 G in active regions (Hoeksema et al. 2014). Below that threshold, faster methods are used.

**Consistency criteria** (Aly 1989, Molodenskii 1969, Low 1985) are integral relations the photospheric vector field must satisfy for a force-free interior to exist. First moment (force balance):
$$
\int_S B_x B_z\,dS = \int_S B_y B_z\,dS = 0, \quad \int_S (B_x^2 + B_y^2)\,dS = \int_S B_z^2\,dS.
$$
Second moment (torque balance):
$$
\int_S x(B_x^2+B_y^2)\,dS = \int_S x B_z^2\,dS, \text{ etc.}
$$
Total magnetic energy from the virial theorem:
$$
E_{\rm tot}=\frac{1}{\mu_0}\int_S (xB_x+yB_y)B_z\,dx\,dy.
$$
A further flux-balance condition $\int_{S^+}f(\alpha)B_n\,dA = \int_{S^-}f(\alpha)B_n\,dA$ involves field-line connectivity which is unknown a priori. Normalized photospheric forces
$$
\frac{|F_x|}{F_p},\ \frac{|F_y|}{F_p},\ \frac{|F_z|}{F_p} \ll 1,\quad F_p = \tfrac{1}{8\pi}\int(B_x^2+B_y^2+B_z^2)\,dS
$$
indicate force-freeness. Metcalf et al. (1995) found photospheric forces (at $z=0$) of $\approx 0.3, 0.4, 0.6$, decreasing to < 0.1 at $z=400$ km (chromosphere). Liu et al. (2013) analyzed 925 magnetograms and found only 17-25% met $|F_z|/F_p<0.1$: **the majority of the photosphere is NOT force-free**. This motivates "preprocessing" (Wiegelmann et al. 2006b):
$$
L_{\rm prep}=\mu_1 L_1+\mu_2 L_2+\mu_3 L_3+\mu_4 L_4,
$$
with $L_1$ = force balance, $L_2$ = torque, $L_3$ = deviation from observations, $L_4$ = smoothness. The four $\mu_i$ are tuned per instrument. (pp. 17-31)

**Korean:** 벡터 자력도는 Stokes profile 반전에서 얻어지며 LOS 성분은 신뢰도 높지만 횡단 성분에는 **180° 방위각 모호성**이 있다. Metcalf 등(2006)은 13개 알고리즘을 합성 flux rope로 벤치마킹했다 (상단 표 참조). Minimum energy 방법(Metcalf 1994)이 simulated annealing으로 $E=\sum(|\nabla\cdot\mathbf{B}|+|\mathbf{j}|)^2$ 최소화하여 98-100% 정확도를 달성하며 SDO/HMI의 표준이다(>150 G). 일관성 조건(Aly 1989): 경계에서 알짜 자기력과 토크가 0이어야 하고, Virial 정리로 총 에너지가 계산 가능하다. 정규화된 광구 힘 $|F_i|/F_p \ll 1$ 판정. Metcalf(1995)는 $z=0$에서 0.3-0.6이던 힘이 $z=400$ km에서 0.1 이하로 감소 — 광구는 force-free가 아니지만 채층은 근사적으로 force-free. Liu(2013)의 925개 자력도 중 17-25%만 force-free 기준 통과. Wiegelmann 등(2006b)의 preprocessing $L_{\rm prep}=\mu_1 L_1+\mu_2 L_2+\mu_3 L_3+\mu_4 L_4$이 force/torque/data/smoothness 항을 결합하여 경계조건을 force-free 호환으로 조정한다.

### Part V: 3D Nonlinear Force-Free Fields / 3D NLFFF 물리 (§5)

**English:** Magnetic helicity $H_m=\int_V \mathbf{A}\cdot\mathbf{B}\,dV$ (Woltjer 1958) is conserved under ideal MHD and approximately conserved under resistive dynamics (Berger 1984). For a volume with a non-magnetic surface (photosphere), the gauge-invariant relative helicity (Finn & Antonsen 1985, Berger & Field 1984) is
$$
K=\int_V (\mathbf{A}+\mathbf{A}') \cdot (\mathbf{B}-\mathbf{B}')\,dV,
$$
with $\mathbf{B}',\mathbf{A}'$ a reference (often potential) field. A simpler proxy is current helicity $H_c=\int_V \mathbf{B}\cdot\nabla\times\mathbf{B}\,dV$, which shows a hemispheric rule (negative north, positive south; Pevtsov et al. 1995).

Sakurai (1989) enumerated energy principles for given $B_z$ boundary flux:
(a) Potential field = minimum energy state.
(b) Fixed $H_m$ → LFFF (stable or unstable).
(c) Field-line connectivity specified → NLFFF.

Using Euler potentials $\mathbf{B}=\nabla u\times\nabla v$ one shows $E_{\rm NLFFF}>E_{\rm LFFF}>E_{\rm potential}$.

**Aly-Sturrock conjecture** (Aly 1984, 1991; Sturrock 1991): the maximum energy of a simply-connected force-free field is that of the fully-open field (all lines reach infinity). Implication: opening lines during a CME cannot release energy. Choe & Cheng (2002) constructed force-free configurations with tangential discontinuities exceeding the open-field energy — debate ongoing.

**Stability**: Molodensky's (1974) energy criterion for force-free fields,
$$
W=\frac{1}{2\mu_0}\int_V\left[(\nabla\times\mathbf{A}_1)^2-\alpha\mathbf{A}_1\cdot\nabla\times\mathbf{A}_1\right]dV,
$$
with perturbed vector potential $\mathbf{A}_1=\xi\times\mathbf{B}_0$. Approximating $|\nabla\times\mathbf{A}_1|\sim|\mathbf{A}_1|/\ell$ gives the Shafranov-like stability limit
$$
|\alpha| \lesssim 1/\ell,
$$
i.e. twist length-scale must exceed the system size. Törok & Kliem (2005) showed the Titov-Démoulin unstable branch triggers erupting kink-unstable flux ropes matching TRACE observations. (pp. 32-37)

**Korean:** Helicity $H_m=\int\mathbf{A}\cdot\mathbf{B}\,dV$는 이상 MHD에서 보존되며 resistive에서도 근사 보존된다. 광구 같은 비자기적 경계에서는 상대 helicity $K$(Finn & Antonsen 1985)를 사용. 현류 helicity $H_c=\int\mathbf{B}\cdot\nabla\times\mathbf{B}\,dV$는 반구 법칙(북반구 음, 남반구 양; Pevtsov 1995) 존재.

에너지 원리(Sakurai 1989): (a) 포텐셜장=최소, (b) $H_m$ 고정→LFFF, (c) connectivity 지정→NLFFF. Euler potentials로 $E_{\rm NLFFF}>E_{\rm LFFF}>E_{\rm pot}$ 증명 가능. Aly-Sturrock 추측: force-free 상한은 open-field 에너지. Choe & Cheng(2002)은 tangential discontinuities로 이를 초과하는 구성을 제시 — 논쟁 중. 안정성: Molodensky(1974) 에너지 기준에서 Shafranov-like 한계 $|\alpha|\lesssim 1/\ell$ 도출. Törok & Kliem(2005)은 불안정 Titov-Démoulin에서 kink instability로 CME를 재현.

### Part VI: Numerical Methods for NLFFF / NLFFF 수치 방법 (§6)

**English:** Five main families of NLFFF codes:

**(1) Upward integration (Nakagawa 1974, Wu et al. 1985).** Integrate $\partial B_i/\partial z$ from the photospheric boundary upward using
$$
\frac{\partial B_{x0}}{\partial z}=\mu_0 j_{y0}+\frac{\partial B_{z0}}{\partial x},
$$
$$
\frac{\partial B_{y0}}{\partial z}=\frac{\partial B_{z0}}{\partial y}-\mu_0 j_{x0},
$$
$$
\frac{\partial B_{z0}}{\partial z}=-\frac{\partial B_{x0}}{\partial x}-\frac{\partial B_{y0}}{\partial y}.
$$
Ill-posed: small errors grow exponentially. Attempts at smoothing stabilization reduce but do not eliminate instability.

**(2) Grad-Rubin (1958, numerically Sakurai 1981, Amari et al. 1997).** The only mathematically well-posed approach: given $B_z$ on the full boundary and $\alpha$ on ONE polarity, iterate
$$
\mathbf{B}^{(k)}\cdot\nabla\alpha^{(k)}=0, \quad \alpha^{(k)}|_{S_\pm}=\alpha_{0\pm} \text{ (hyperbolic, along field lines)},
$$
$$
\nabla\times\mathbf{B}^{(k+1)}=\alpha^{(k)}\mathbf{B}^{(k)}, \quad \nabla\cdot\mathbf{B}^{(k+1)}=0 \text{ (elliptic, Ampère)}.
$$
Bineau (1972) proved existence/uniqueness for small α and weak nonlinearity. Either polarity gives a solution — and they differ for noisy real data, providing a consistency check. Extended by Wheatland & Régnier (2009) and Amari & Aly (2010) to blend both solutions.

**(3) MHD relaxation (Chodura & Schlüter 1981, Mikić et al. 1988).** Evolve reduced time-dependent MHD equations $\nu\mathbf{v}=(\nabla\times\mathbf{B})\times\mathbf{B}$, $\mathbf{E}+\mathbf{v}\times\mathbf{B}=0$, $\partial\mathbf{B}/\partial t = -\nabla\times\mathbf{E}$ with fictitious viscosity $\nu=|\mathbf{B}|^2/\mu$. This reduces to
$$
\frac{\partial\mathbf{B}}{\partial t}=\mu\,\mathbf{F}_{\rm MHD}, \quad \mathbf{F}_{\rm MHD}=\nabla\times\left(\frac{[(\nabla\times\mathbf{B})\times\mathbf{B}]\times\mathbf{B}}{B^2}\right).
$$

**(4) Optimization (Wheatland, Sturrock & Roumeliotis 2000).** Minimize the functional
$$
L=\int_V \left[B^{-2}|(\nabla\times\mathbf{B})\times\mathbf{B}|^2 + |\nabla\cdot\mathbf{B}|^2\right]dV.
$$
L = 0 ⟺ NLFFF. Functional derivative gives
$$
\frac{1}{2}\frac{dL}{dt}=-\int_V \frac{\partial\mathbf{B}}{\partial t}\cdot\mathbf{F}\,dV - \int_S \frac{\partial\mathbf{B}}{\partial t}\cdot\mathbf{G}\,dS,
$$
with F containing six terms (Eq. 79 of the paper). For vanishing surface terms, choosing $\partial\mathbf{B}/\partial t = \mu\mathbf{F}$ monotonically decreases L. Wiegelmann & Inhester (2010) extended with a measurement-error term
$$
\nu\int_S(\mathbf{B}-\mathbf{B}_{\rm obs})\cdot\mathbf{W}\cdot(\mathbf{B}-\mathbf{B}_{\rm obs})\,dS,
$$
where W encodes the measurement confidence. This is the most widely used NLFFF code.

**(5) Boundary-element (Yan & Sakurai 2000).** Based on the integral representation
$$
c_i\mathbf{B}_i=\oint_S \left(\overline{\mathbf{Y}}\frac{\partial\mathbf{B}}{\partial n}-\frac{\partial\overline{\mathbf{Y}}}{\partial n}\mathbf{B}_0\right)dS,
$$
with $\overline{\mathbf{Y}}={\rm diag}(\cos\lambda_i r/(4\pi r))$ and implicit $\lambda_i$ integrals. Mathematically appealing but computationally slow.

**Extensions**: Yin-Yang spherical grids (Jiang et al. 2012) for global NLFFF; VCA-NLFFF (Aschwanden 2013a-c) forward-fits superposed LFFFs to loop images (Eq. 87: $(\nabla\times\mathbf{B})\times\mathbf{B}=(\alpha_1-\alpha_2)\mathbf{B}_1\times\mathbf{B}_2$ for two LFFFs — quasi force-free if components are well separated); S-NLFFF (Chifu et al. 2017) adds a stereoscopic-loop misalignment term $L_4=\sum_i (1/\int_{c_i}ds)\int_{c_i}|\mathbf{B}\times\mathbf{t}_i|^2/\sigma_{c_i}^2\,ds$ to the optimization. (pp. 37-47)

**Korean:** NLFFF 수치 방법 5가지: (1) Upward integration — 불안정 (오차 지수적 성장), (2) Grad-Rubin — 수학적으로 well-posed, α를 한 극성에서 field line 따라 전파 후 Biot-Savart로 B 갱신 반복; 양 극성 해 비교로 일관성 점검, (3) MHD relaxation — 가상 점성으로 평형에 도달하는 시간의존 MHD, (4) Optimization (Wheatland 등 2000) — $L=\int[B^{-2}|(\nabla\times\mathbf{B})\times\mathbf{B}|^2+|\nabla\cdot\mathbf{B}|^2]\,dV$ 최소화; Wiegelmann & Inhester(2010)이 측정오차 항 $\nu\int(\mathbf{B}-\mathbf{B}_{\rm obs})\cdot\mathbf{W}\cdot(\mathbf{B}-\mathbf{B}_{\rm obs})dS$ 추가; 현재 가장 널리 쓰임, (5) Boundary-element — 적분 표현 사용, 느림. 확장: Yin-Yang 구면 그리드(Jiang 2012), VCA-NLFFF(Aschwanden — 국지 LFFF 중첩으로 quasi force-free), S-NLFFF(Chifu 2017 — stereoscopic loops 정렬 항 추가).

### Part VII: Effects, Limitations, Comparisons / 효과·한계·비교 (§7)

**English:** The NLFFF consortium (Schrijver, since 2004) produced key lessons from blind tests:

1. **Schrijver et al. (2006)** on analytic test fields — Wheatland optimization converges fastest, performs best where currents are strong.
2. **Metcalf et al. (2008)** on a solar-like reference (with finite photospheric Lorentz forces): preprocessing and chromospheric boundaries essential.
3. **Schrijver et al. (2008)** applied 14 NLFFF models to a flaring AR with Hinode/SOT-SP data. Wide variety of geometries, energy contents. Grad-Rubin best for current/field-line alignment consistency.
4. **DeRosa et al. (2009)**: NLFFF models diverge significantly in free-energy estimates on Hinode/SOT-SP data. Major issue: small FOV (~10%) misses connectivity.
5. **DeRosa et al. (2015)**: spatial resolution matters; free energy increases monotonically with resolution, while relative helicity scatters.

**Conditions for successful NLFFF** (DeRosa 2009 prescription):
1. Large model volumes with high resolution accommodating connectivity.
2. Measurement uncertainty accommodation in transverse field.
3. Preprocessing to approximate chromospheric (near-force-free) boundary.
4. Field lines must be compared with coronal observations for validation.

**Other effects investigated**:
- Size of computational domain (Tadesse et al. 2015): matters more for connected ARs.
- Spatial resolution (DeRosa 2015): higher resolution → more consistent, higher free energy.
- Finite β (Peter et al. 2015): if β ~ relative free energy, pressure gradient important.
- Instrumental (Thalmann 2013): Hinode vs SDO NLFFFs differ.
- Initial conditions (Kawabata 2020): matter only in weak-field regions for complex ARs.
- Additional measurements (Fleishman 2017, 2019): chromospheric/coronal data improve NLFFF even if partial.

**Flare/CME applications**:
- Bleybel et al. (2002): AR 7912 flare released free energy (Grad-Rubin). NLFFF best matches X-ray loops (Fig. 17).
- Thalmann 2008: NOAA 10540 pre-flare free energy 60% above potential.
- Jing et al. (2010): 75-sample statistical study — positive correlation between free magnetic energy and X-ray flare index.

**Coronal seismology cross-check** (Verwichte et al. 2013): two oscillating loops gave consistent Alfvén speeds from PFSS extrapolation + spectral DEM analysis — demonstrates magnetic extrapolations complement seismology. (pp. 47-55)

**Korean:** NLFFF consortium의 주요 교훈: (1) 해석적 테스트에서 Wheatland optimization이 가장 빠르고 정확 (Schrijver 2006), (2) 태양 유사 데이터에서 preprocessing과 채층 경계 필수 (Metcalf 2008), (3) 14개 모델이 활성영역에서 다양한 결과 (Schrijver 2008), (4) 작은 FOV가 주요 문제 (DeRosa 2009), (5) 해상도 증가 시 자유 에너지 단조 증가 (DeRosa 2015). 성공 조건: 큰 볼륨, 측정오차 수용, preprocessing, 코로나 영상 검증. 기타 효과: 계산 도메인 크기, 공간 해상도, 유한 β, 계기 차이, 초기 조건, 추가 측정 모두 영향. 플레어 응용: pre-flare에서 자유 에너지가 포텐셜 대비 60% 초과(Thalmann 2008), 75개 표본 통계에서 자유 에너지와 X선 flare index 양의 상관(Jing 2010). Verwichte(2013)는 코로나 seismology로 얻은 Alfvén 속도가 PFSS 외삽과 일치함을 보여 두 방법의 상보성 증명.

### Part VIII: MHS & MHD Extensions / MHS 및 MHD 확장 (§8)

**English:** Force-free is invalid in the photosphere/chromosphere. MHS codes generalize:
$$
\nabla p + \rho\nabla\Psi = \mathbf{j}\times\mathbf{B}.
$$
- MHS optimization: Wiegelmann & Neukirch (2006); refined Zhu & Wiegelmann (2018, 2019).
- MHS via MHD relaxation: Zhu et al. (2013, 2016) applied to chromospheric Hα fibrils; Miyoshi et al. (2020) 2D tests.
- MHS Grad-Rubin: Gilchrist & Wheatland (2013), Gilchrist et al. (2016).

MHS codes are more expensive, especially with mixed-β domains. For eruption dynamics, NLFFF serves as initial equilibrium for time-dependent MHD (Jiang 2013, Pagano 2014, Jiang 2017, Prasad 2018, Pagano 2018, Toriumi 2020). Questions remain: Does the field stay force-free during eruption? What role do thin current sheets play? (pp. 55-57)

**Korean:** 광구·채층은 force-free가 아니며 MHS 확장이 필요: $\nabla p + \rho\nabla\Psi = \mathbf{j}\times\mathbf{B}$. Optimization(Wiegelmann & Neukirch 2006), MHD relaxation(Zhu 2013-2016), Grad-Rubin(Gilchrist 2013-2016) 기반 MHS 코드 개발 중. 혼합-β 영역에서 계산 비용 큼. NLFFF는 CME/flare 시간 진화 MHD 시뮬레이션의 초기 평형으로 활용(Jiang 2013-2017, Toriumi 2020). 미해결 문제: 폭발 중 field가 force-free 유지되는가? Thin current sheet 역할?

---

## 3. Key Takeaways / 핵심 시사점

1. **Plasma-β hierarchy dictates the force-free regime / 플라즈마 β 계층이 force-free 유효 영역을 결정한다** — Force-free 근사는 β ≪ 1인 영역, 즉 상부 채층에서 약 2.5 R☉까지만 유효. 광구는 β ~ 1이므로 strictly force-free가 아니고 measurements가 consistency 조건을 완벽히 만족하지 못한다. Liu et al. (2013)에 따르면 925개 자력도 중 17-25%만 force-free 기준 통과. / The force-free approximation only holds in β ≪ 1 regions — the upper chromosphere through ~2.5 R☉. The photosphere has β ~ 1 and is not strictly force-free: only 17-25% of 925 analyzed magnetograms (Liu et al. 2013) meet the criterion $|F_z|/F_p<0.1$.

2. **LFFF is a useful compromise but fundamentally limited / LFFF는 유용하나 근본적 한계가 있다** — A single α (Seehafer 해 Fourier)로 빠르게 구현되나, AR11084 같은 실제 AR에서 chromospheric α ≈ 0.4와 coronal α ≈ 0.23은 일치하지 않는다. 여러 층을 하나의 LFFF로 설명할 수 없으므로 NLFFF가 필수. / LFFF with a single α has an analytic Fourier (Seehafer) solution, but real ARs such as AR11084 show chromospheric α ≈ 0.4 vs coronal α ≈ 0.23 — a global α cannot describe both. NLFFF is required.

3. **Grad-Rubin is the only mathematically well-posed NLFFF scheme / Grad-Rubin은 유일하게 수학적으로 well-posed인 NLFFF 방법** — Boundary conditions: $B_z$ on full boundary + α on ONE polarity. Existence/uniqueness proven (Bineau 1972). 두 극성에서 얻은 해의 차이가 measurement inconsistency의 진단 지표. / Grad-Rubin prescribes $B_z$ on the entire boundary plus α on a single polarity; existence and uniqueness proven by Bineau (1972). Solutions from ± polarities differ under noise — a useful consistency diagnostic.

4. **The Wiegelmann optimization method is the de-facto standard / Wiegelmann optimization 방법이 사실상 표준** — Functional $L=\int[B^{-2}|(\nabla\times\mathbf{B})\times\mathbf{B}|^2+|\nabla\cdot\mathbf{B}|^2]\,dV$의 gradient descent로 NLFFF 수렴. 측정오차 weighting $\mathbf{W}$ 추가(Wiegelmann & Inhester 2010)로 실제 HMI 데이터에 강건. SDO/HMI 운영 파이프라인에 채택. / The Wheatland-Sturrock-Roumeliotis (2000) functional minimized via steepest descent is the most widely used NLFFF solver. With the W-matrix extension (Wiegelmann & Inhester 2010) it handles HMI measurement errors robustly, and is deployed in operational pipelines.

5. **Preprocessing is indispensable for photospheric data / 광구 데이터에는 preprocessing이 필수** — $L_{\rm prep}=\mu_1 L_1+\mu_2 L_2+\mu_3 L_3+\mu_4 L_4$ (force + torque + data + smoothness) minimizes the Aly integral relations by orders of magnitude and produces a "chromosphere-like" boundary. Metcalf et al. (2008) 컨소시엄 벤치마크에서 결정적 요인으로 확인. / Preprocessing by minimizing $L_{\rm prep}=\mu_1 L_1+\mu_2 L_2+\mu_3 L_3+\mu_4 L_4$ (force balance + torque balance + data fidelity + smoothness) is essential to transform a forced photospheric magnetogram into a quasi-chromospheric, near-force-free boundary. Metcalf et al. (2008) confirmed this is decisive for consortium benchmarks.

6. **Azimuth ambiguity removal has a clear winner / 방위각 모호성 제거의 명확한 승자** — Metcalf (1994) minimum-energy method achieves 98-100% correct disambiguation; U. Hawai'i iterative method 97%; potential-field acute-angle only 64-75%. SDO/HMI adopts minimum-energy above 150 G. / The minimum-energy method of Metcalf (1994) achieves 98-100% on synthetic tests — adopted by SDO/HMI for |B|>150 G. Potential-field acute-angle is only 64-75% accurate but provides a fast initialization.

7. **Free magnetic energy correlates with flare productivity / 자유 자기에너지는 flare 생산성과 상관** — Jing et al. (2010)의 75-표본 통계로 free energy와 X-ray flare index 간 양의 상관 확인. Pre-flare에서 ~60% (Thalmann 2008, NOAA 10540, M-class) 초과 가능. NLFFF는 이 핵심 예측 지표를 제공. / Jing et al. (2010) on 75 ARs showed positive correlation between free magnetic energy and X-ray flare productivity. Pre-flare free energy can exceed potential-field energy by ~60% (Thalmann 2008, NOAA 10540 before M-class flare).

8. **NLFFF reconstructions have large systematic uncertainties — proper validation against coronal observations is non-negotiable / NLFFF 재구성에는 큰 계통오차가 있어 코로나 관측 검증이 필수** — DeRosa et al. (2009, 2015): 14개 코드가 동일 데이터에서 서로 다른 free energy를 산출. FOV, resolution, ambiguity, preprocessing, 초기 조건 모두 영향. 결과를 믿으려면 EUV/X-ray 영상과 field line 일치 확인 필수. / DeRosa et al. (2009, 2015) showed 14 NLFFF codes can give widely different free-energy estimates for identical data. FOV, resolution, disambiguation, preprocessing, and initial conditions all matter. Results should only be trusted when field lines agree with EUV/X-ray coronal loop observations.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Force-free condition / 무력장 조건

$$
\mathbf{j}\times\mathbf{B}=0 \iff (\nabla\times\mathbf{B})\times\mathbf{B}=0 \iff \nabla\times\mathbf{B}=\alpha\mathbf{B},\quad \mathbf{B}\cdot\nabla\alpha=0,\quad \nabla\cdot\mathbf{B}=0.
$$
- $\alpha$: force-free parameter / 무력장 매개변수; must be constant along each field line.
- $\mathbf{B}\cdot\nabla\alpha=0$ follows from $\nabla\cdot(\nabla\times\mathbf{B})=0$.

### 4.2 Potential field / 포텐셜장 (α = 0)

$$
\mathbf{B}=-\nabla\Phi,\quad \nabla^2\Phi=0,
$$
with Neumann boundary $\partial\Phi/\partial n = -B_n$ on the photosphere. 최소 에너지 상태 / Minimum-energy state.

### 4.3 Linear force-free field / 선형 무력장 (α = const)

$$
\Delta\mathbf{B}+\alpha^2\mathbf{B}=0,
$$
Seehafer (1978) Fourier solution:
$$
B_z(x,y,z)=\sum_{m,n} C_{mn}\,e^{-r_{mn}z}\sin\!\tfrac{\pi m x}{L_x}\sin\!\tfrac{\pi n y}{L_y},
\quad r_{mn}=\sqrt{\lambda_{mn}-\alpha^2},\ \lambda_{mn}=\pi^2\!\left(\tfrac{m^2}{L_x^2}+\tfrac{n^2}{L_y^2}\right).
$$
Constraint: $|\alpha|<\sqrt{2}\pi/L$ (harmonic-mean $L$). / 조건: $|\alpha|<\sqrt{2}\pi/L$.

### 4.4 Current density from vector magnetogram / 벡터 자력도로부터 전류 밀도

$$
\mu_0 j_{z0}=\frac{\partial B_{y0}}{\partial x}-\frac{\partial B_{x0}}{\partial y},\qquad \alpha(x,y)=\mu_0 \frac{j_{z0}}{B_{z0}}.
$$

### 4.5 Grad-Shafranov (axisymmetric NLFFF) / 축대칭 NLFFF

$$
\frac{\partial^2 A}{\partial r^2}+\frac{1-\mu^2}{r^2}\frac{\partial^2 A}{\partial \mu^2}+Q\frac{dQ}{dA}=0,\quad \mu=\cos\theta.
$$
Low-Lou ansatz: $Q=\lambda A^{1+1/n}$, $A=P(\mu)/r^n$ → ODE for $P(\mu)$ (eigenvalue problem).

### 4.6 Titov-Démoulin stability / T-D 안정성

$$
B_0\approx\frac{\mu_0 I_0}{2\pi R},\qquad \text{kink instability when } R\gtrsim\sqrt{2}\,L.
$$

### 4.7 Consistency criteria (Aly 1989) / 일관성 조건

$$
\int_S B_i B_z\,dS=0\ (i=x,y),\quad \int_S (B_x^2+B_y^2)\,dS=\int_S B_z^2\,dS.
$$
$$
\int_S x(B_x^2+B_y^2)\,dS=\int_S x B_z^2\,dS,\quad \text{etc.}
$$
$$
\int_S y B_x B_z\,dS=\int_S x B_y B_z\,dS.
$$

### 4.8 Virial theorem / 비리얼 정리

$$
\boxed{E_{\rm tot}=\frac{1}{\mu_0}\int_S (xB_x+yB_y)B_z\,dx\,dy.}
$$

### 4.9 Dimensionless photospheric forces / 무차원 광구 힘

$$
F_p=\tfrac{1}{8\pi}\int_S (B_x^2+B_y^2+B_z^2)\,dS,
$$
$$
F_x=-\tfrac{1}{4\pi}\int_S B_xB_z\,dS,\ F_y=-\tfrac{1}{4\pi}\int_S B_yB_z\,dS,\ F_z=\tfrac{1}{8\pi}\int_S (B_x^2+B_y^2-B_z^2)\,dS.
$$
Force-free if $|F_i|/F_p \ll 1$ (typically <0.1).

### 4.10 Preprocessing functional / 전처리 함수

$$
L_{\rm prep}=\mu_1 L_1+\mu_2 L_2+\mu_3 L_3+\mu_4 L_4,
$$
$$
L_1=\left[\left(\sum B_xB_z\right)^2+\left(\sum B_yB_z\right)^2+\left(\sum B_z^2-B_x^2-B_y^2\right)^2\right],
$$
$L_2$ = torque, $L_3=\sum(B_i-B_{i,\rm obs})^2$, $L_4=\sum|\Delta B_i|^2$ (smoothness).

### 4.11 Wiegelmann optimization / Wiegelmann 최적화

$$
\boxed{L=\int_V\left[B^{-2}|(\nabla\times\mathbf{B})\times\mathbf{B}|^2+|\nabla\cdot\mathbf{B}|^2\right]dV}
$$
$$
\frac{1}{2}\frac{dL}{dt}=-\int_V \frac{\partial\mathbf{B}}{\partial t}\cdot\mathbf{F}\,dV-\int_S \frac{\partial\mathbf{B}}{\partial t}\cdot\mathbf{G}\,dS.
$$
$$
\mathbf{F}=\nabla\times\left(\frac{[(\nabla\times\mathbf{B})\times\mathbf{B}]\times\mathbf{B}}{B^2}\right)+\left\{-\nabla\times\left(\frac{((\nabla\cdot\mathbf{B})\mathbf{B})\times\mathbf{B}}{B^2}\right)-\boldsymbol{\Omega}\times(\nabla\times\mathbf{B})-\nabla(\boldsymbol{\Omega}\cdot\mathbf{B})+\boldsymbol{\Omega}(\nabla\cdot\mathbf{B})+\Omega^2\mathbf{B}\right\},
$$
$$
\boldsymbol{\Omega}=B^{-2}\left[(\nabla\times\mathbf{B})\times\mathbf{B}-(\nabla\cdot\mathbf{B})\mathbf{B}\right].
$$
Choose $\partial\mathbf{B}/\partial t = \mu\mathbf{F}$ ⇒ $dL/dt \le 0$.

Wiegelmann & Inhester (2010) adds:
$$
\nu\int_S(\mathbf{B}-\mathbf{B}_{\rm obs})\cdot\mathbf{W}\cdot(\mathbf{B}-\mathbf{B}_{\rm obs})\,dS.
$$

### 4.12 MHD relaxation / MHD 이완

$$
\nu\mathbf{v}=(\nabla\times\mathbf{B})\times\mathbf{B},\ \mathbf{E}+\mathbf{v}\times\mathbf{B}=0,\ \frac{\partial\mathbf{B}}{\partial t}=-\nabla\times\mathbf{E},\ \nabla\cdot\mathbf{B}=0.
$$
With $\nu=|\mathbf{B}|^2/\mu$:
$$
\frac{\partial\mathbf{B}}{\partial t}=\mu\,\mathbf{F}_{\rm MHD},\quad \mathbf{F}_{\rm MHD}=\nabla\times\left(\frac{[(\nabla\times\mathbf{B})\times\mathbf{B}]\times\mathbf{B}}{B^2}\right).
$$

### 4.13 Grad-Rubin iteration / Grad-Rubin 반복

$$
\mathbf{B}^{(k)}\cdot\nabla\alpha^{(k)}=0,\quad \alpha^{(k)}|_{S_\pm}=\alpha_{0\pm},
$$
$$
\nabla\times\mathbf{B}^{(k+1)}=\alpha^{(k)}\mathbf{B}^{(k)},\ \nabla\cdot\mathbf{B}^{(k+1)}=0,\ B_z|_{S_\pm}=B_{z0},\ |\mathbf{B}|\to 0 \text{ as } |\mathbf{r}|\to\infty.
$$

### 4.14 Magnetic & current helicity / 자기·전류 helicity

$$
H_m=\int_V \mathbf{A}\cdot\mathbf{B}\,dV,\quad K=\int_V(\mathbf{A}+\mathbf{A}')\cdot(\mathbf{B}-\mathbf{B}')\,dV,\quad H_c=\int_V \mathbf{B}\cdot\nabla\times\mathbf{B}\,dV.
$$

### 4.15 Shafranov stability / Shafranov 안정성

Molodensky energy: $W=\frac{1}{2\mu_0}\int[(\nabla\times\mathbf{A}_1)^2-\alpha\mathbf{A}_1\cdot\nabla\times\mathbf{A}_1]\,dV$.
Stability criterion: $|\alpha|\lesssim 1/\ell$.

### 4.16 Worked numerical example — free energy of an AR / 활성영역 자유에너지 계산 예

**Setup:** AR of size $L\sim 100$ Mm with peak $B\sim 1000$ G. Volume $V\sim L^3=10^{24}\,{\rm cm}^3$. Magnetic energy density $B^2/(8\pi)$:
$$
\frac{B^2}{8\pi}\approx\frac{(1000\,{\rm G})^2}{8\pi}\approx 4\times 10^{4}\,{\rm erg/cm^3}.
$$
Total potential energy:
$$
E_{\rm pot}\sim 4\times 10^4\times 10^{24}\approx 4\times 10^{28}\,{\rm erg}.
$$
Typical Wiegelmann-optimization NLFFF recovers free energy fraction $\sim$ 5-30%:
$$
E_{\rm free}\sim 0.1\times E_{\rm pot}\approx 4\times 10^{27}\,{\rm erg},
$$
which is sufficient to power an X-class flare ($\sim 10^{31-32}$ erg total; free energy typically 10× the radiated energy in the loop volume). Virial theorem check: integrate $(xB_x+yB_y)B_z$ over the photospheric $L^2$ area; should match volumetric integral within a few percent for a consistent NLFFF. / 대표 α 값 (`α_typical ~ 10^{-8} cm^{-1} = 10 Mm^{-1}`): loop 길이 $\ell\sim 100$ Mm인 경우 $\alpha\ell\sim 1$로 Shafranov 한계에 근접.

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1908 ── Hale: Zeeman effect → first solar magnetic field measurements
         |
1950 ── Lundquist: force-free cylindrical equilibrium (Bessel)
1954 ── Chandrasekhar-Kendall: spherical force-free fields
         |
1958 ── Grad-Rubin: well-posed NLFFF BVP (for fusion plasmas)
1969 ── Schatten-Wilcox-Ness: PFSS model
         |
1974 ── Nakagawa: upward integration | Molodensky: stability criterion
1977 ── Chiu-Hilton: Green's function for LFFF
1978 ── Seehafer: Cartesian Fourier LFFF
1981 ── Sakurai: first numerical Grad-Rubin NLFFF
1984 ── Aly: Aly-Sturrock conjecture (open-field energy limit)
1988 ── Mikić et al.: MHD relaxation NLFFF
1989 ── Aly: consistency criteria (integral relations + virial)
1990 ── Low-Lou: analytic axisymmetric NLFFF (canonical benchmark)
1994 ── Metcalf: minimum-energy ambiguity removal
1997 ── Amari et al.: modern Grad-Rubin implementation
1999 ── Titov-Démoulin: flux rope equilibrium (kink-unstable)
2000 ── Wheatland-Sturrock-Roumeliotis: OPTIMIZATION METHOD
         Yan-Sakurai: boundary-element method
2004 ── Schrijver: NLFFF CONSORTIUM founded
2005 ── Török-Kliem: numerical TD kink simulation
2006 ── Hinode launch | Metcalf et al. ambiguity benchmark
         Wiegelmann et al.: preprocessing algorithm
2009 ── DeRosa et al.: critical NLFFF-Hinode assessment
2010 ── SDO/HMI first light → routine vector magnetograms
         Wiegelmann-Inhester: W-matrix extension (measurement errors)
2012 ── Wiegelmann-Sakurai 1st Living Review
         Jiang et al.: Yin-Yang global NLFFF grid
2013 ── Aschwanden: VCA-NLFFF forward-fit
2014 ── SDO/HMI vector magnetogram pipeline operational (Hoeksema)
2015 ── DeRosa et al. (resolution study)
2017 ── Chifu et al.: S-NLFFF stereoscopic extension
2018 ── Yeates et al.: ISSI global NLFFF workshops
2019 ── Benson et al.: neural-network α estimation
2020 ── DKIST first light | Toriumi et al.: data-driven MHD
         ╰── Wiegelmann-Sakurai Living Review (REVISED)  ◆ THIS PAPER (2021) ◆
         |
2021+ ─ Solar Orbiter out-of-ecliptic | MHS extensions mature
         NLFFF → MHD initial conditions for space weather
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Paper #6 Parker (1958, solar wind) | Coronal plasma β가 낮은 이유 (자기장이 주도)를 최초로 논증. Wiegelmann-Sakurai의 force-free 전제의 근간. / Established the low-β corona that justifies the force-free assumption. | ★★★ Foundational — the physical setting for all force-free modeling. |
| Paper #31 Parker (1988, nanoflare) | Free magnetic energy의 소산 메커니즘(nanoflare)을 제안; NLFFF는 이 free energy를 정량화하는 도구. / Proposed nanoflare dissipation of free magnetic energy; NLFFF quantifies this energy reservoir. | ★★★ NLFFF outputs feed Parker's nanoflare framework. |
| Sakurai (1981) | First numerical Grad-Rubin NLFFF — 본 리뷰의 §6.2 핵심 인용. / Foundational Grad-Rubin implementation cited throughout §6. | ★★★ Core method citation; Sakurai is coauthor. |
| Wheatland, Sturrock & Roumeliotis (2000) | Optimization 방법 원논문 — 현대 NLFFF의 표준. / Original optimization method paper — the modern NLFFF workhorse. | ★★★ Central to §6.4, most cited NLFFF code. |
| Low & Lou (1990) | 축대칭 반해석적 NLFFF — 모든 코드의 benchmark. / Semi-analytic axisymmetric NLFFF; universal benchmark. | ★★★ §3.1; test of any new NLFFF implementation. |
| Titov & Démoulin (1999) | Kink-unstable flux rope — eruption 트리거 메커니즘의 표준 모델. / Standard unstable flux rope for CME onset studies. | ★★★ §3.2; bridge to MHD simulations. |
| Aly (1984, 1989, 1991) | Consistency criteria (적분 관계) + Aly-Sturrock 추측. / Integral consistency criteria + Aly-Sturrock open-field energy conjecture. | ★★★ §4.6, §5.3; mathematical foundation. |
| Metcalf (1994) | Minimum-energy ambiguity removal — SDO/HMI 표준. / Minimum-energy method adopted by SDO/HMI. | ★★★ §4.3.8; operational disambiguation. |
| DeRosa et al. (2009, 2015) | NLFFF 방법 비교 critique — free energy uncertainty 강조. / Critical assessments of NLFFF divergence and resolution effects. | ★★☆ §7.1, §7.3.2. |

---

## 7. References / 참고문헌

- Wiegelmann, T. & Sakurai, T., "Solar Force-Free Magnetic Fields", *Living Reviews in Solar Physics* 18:1 (2021). https://doi.org/10.1007/s41116-020-00027-4
- Wheatland, M.S., Sturrock, P.A. & Roumeliotis, G., "An Optimization Approach to Reconstructing Force-Free Fields", *ApJ* 540, 1150 (2000).
- Wiegelmann, T. & Inhester, B., "How to Deal with Measurement Errors and Lacking Data in Nonlinear Force-Free Coronal Magnetic Field Modelling", *A&A* 516, A107 (2010).
- Low, B.C. & Lou, Y.Q., "Modeling Solar Force-Free Magnetic Fields", *ApJ* 352, 343 (1990).
- Titov, V.S. & Démoulin, P., "Basic Topology of Twisted Magnetic Configurations in Solar Flares", *A&A* 351, 707 (1999).
- Metcalf, T.R., "Resolving the 180-degree Ambiguity in Vector Magnetic Field Measurements: The Minimum Energy Method", *Sol. Phys.* 155, 235 (1994).
- Metcalf, T.R. et al., "An Overview of Existing Algorithms for Resolving the 180° Ambiguity in Vector Magnetic Fields", *Sol. Phys.* 237, 267 (2006).
- Aly, J.J., "On the Reconstruction of the Nonlinear Force-Free Coronal Magnetic Field from Boundary Data", *Sol. Phys.* 120, 19 (1989).
- Sakurai, T., "Calculation of Force-Free Magnetic Field with Non-Constant α", *Sol. Phys.* 69, 343 (1981).
- Grad, H. & Rubin, H., "Hydromagnetic Equilibria and Force-Free Fields", Proc. 2nd Int. Conf. on Peaceful Uses of Atomic Energy 31, 190 (1958).
- Amari, T., Boulmezaoud, T.Z. & Mikić, Z., "An Iterative Method for the Reconstruction of the Solar Coronal Magnetic Field. I. Method for Regular Solutions", *A&A* 350, 1051 (1999).
- DeRosa, M.L. et al., "A Critical Assessment of Nonlinear Force-Free Field Modeling of the Solar Corona for Active Region 10953", *ApJ* 696, 1780 (2009).
- Seehafer, N., "Determination of Constant α Force-Free Solar Magnetic Fields from Magnetograph Data", *Sol. Phys.* 58, 215 (1978).
- Woltjer, L., "A Theorem on Force-Free Magnetic Fields", *PNAS* 44, 489 (1958).
- Schatten, K.H., Wilcox, J.M. & Ness, N.F., "A Model of Interplanetary and Coronal Magnetic Fields", *Sol. Phys.* 6, 442 (1969).
- Wiegelmann, T., Inhester, B. & Sakurai, T., "Preprocessing of Vector Magnetograph Data for a Nonlinear Force-Free Magnetic Field Reconstruction", *Sol. Phys.* 233, 215 (2006b).
- Aschwanden, M.J., "A Nonlinear Force-Free Magnetic Field Approximation Suitable for Fast Forward-Fitting to Coronal Loops. I. Theory", *Sol. Phys.* 287, 323 (2013a).
- Chifu, I., Wiegelmann, T. & Inhester, B., "Nonlinear Force-Free Coronal Magnetic Stereoscopy", *ApJ* 837, 10 (2017).
