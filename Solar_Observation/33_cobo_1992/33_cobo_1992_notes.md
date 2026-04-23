---
title: "Inversion of Stokes Profiles"
authors: [Ruiz Cobo, B., del Toro Iniesta, J. C.]
year: 1992
journal: "The Astrophysical Journal"
doi: "10.1086/171862"
topic: Solar_Observation
tags: [stokes_inversion, radiative_transfer, magnetic_field, response_function, SIR, Marquardt, spectropolarimetry, DELO]
status: completed
date_started: 2026-04-23
date_completed: 2026-04-23
---

# 33. Inversion of Stokes Profiles / 스토크스 프로파일 인버전

---

## 1. Core Contribution / 핵심 기여

**EN.** Ruiz Cobo & del Toro Iniesta (1992) introduce **SIR** (Stokes Inversion based on Response functions), the first general-purpose nonlinear least-squares inversion code to recover the **depth stratification** of five physical magnitudes — temperature $T(\tau)$, magnetic field strength $B(\tau)$, inclination $\gamma(\tau)$, azimuth $\phi(\tau)$, line-of-sight velocity $v(\tau)$ — plus two depth-independent magnitudes — microturbulence $\xi_\mathrm{mic}$, macroturbulence $\xi_\mathrm{mac}$ — from observed full-Stokes profiles $(I, Q, U, V)$. Two foundational innovations drive the code's speed and robustness. **First**, SIR exploits the Sánchez Almeida (1992) analytical expression for response functions (RFs) together with the DELO formal solution of the polarized RTE (Rees, Murphy & Durrant 1989), so that a single forward pass through the atmosphere yields simultaneously the synthetic Stokes spectrum *and* every partial derivative $\partial I_k(\lambda)/\partial x_m(\tau)$ needed to build the Marquardt curvature matrix. **Second**, SIR parameterizes each depth-dependent magnitude by only $m$ values at equi-spaced nodes in $\log\tau$ (typically $m = 2, 5,$ or $9$), interpolated by cubic splines, reducing the free-parameter count from $5n + 2$ to $5m + 2$ while preserving full-atmosphere information through "equivalent RFs." A modified SVD stabilizes the inversion against disparate sensitivities of $I, Q, U, V$ to different physical magnitudes.

**KR.** Ruiz Cobo & del Toro Iniesta (1992)는 관측된 풀-스토크스 프로파일 $(I, Q, U, V)$ 로부터 다섯 개의 물리량 — 온도 $T(\tau)$, 자기장 세기 $B(\tau)$, 경사각 $\gamma(\tau)$, 방위각 $\phi(\tau)$, 시선속도 $v(\tau)$ — 의 **깊이 층상구조(depth stratification)** 와 두 개의 깊이-무관량 — 미시 난류속도 $\xi_\mathrm{mic}$, 거시 난류속도 $\xi_\mathrm{mac}$ — 를 동시에 복원하는 최초의 범용 비선형 최소자승 인버전 코드 **SIR**(Stokes Inversion based on Response functions)을 제시한다. 두 가지 근본적 혁신이 코드의 속도와 견고성을 만든다. **첫째**, Sánchez Almeida (1992)의 해석적 RF 표현과 편광 RTE의 DELO 형식해(Rees, Murphy & Durrant 1989)를 결합하여, 대기를 한 번 지나가는 정방향(forward) 계산만으로 합성 스토크스 스펙트럼과 Marquardt 곡률행렬 구성에 필요한 모든 편미분 $\partial I_k(\lambda)/\partial x_m(\tau)$ 을 동시에 얻는다. **둘째**, 각 깊이-의존 물리량을 $\log\tau$ 상 등간격 $m$ 개 노드의 값($m = 2, 5, 9$ 가 전형)으로만 기술하고 큐빅 스플라인으로 보간하여 자유 파라미터 수를 $5n+2$ 에서 $5m+2$ 로 감축하면서도 "등가 RF"(equivalent RF)를 통해 전 대기 정보를 유지한다. 수정된 SVD로 $I, Q, U, V$ 의 물리량별 민감도 차이에 의한 수치 불안정을 해소한다.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction — Synthesis vs Inversion / 합성과 인버전의 대비 (§1, pp. 375-376)

**EN.** The paper opens by situating Stokes inversion within solar spectroscopy's bigger project — **reliable determination of solar atmospheric physical magnitudes**. Zeeman-induced polarization in photospheric lines is a privileged diagnostic of the magnetic-field vector. Two methodological families compete:

1. **Synthesis methods (AHH, Auer, Heasley & House 1977).** Build trial atmospheres, compute synthetic Stokes, compare to observations, iterate by hand. Accurate because they handle the full RTE (including non-LTE: Rees, Murphy & Durrant 1989 and references) but prohibitively expensive for large data volumes because they demand whole-atmosphere knowledge and trial-and-error.

2. **Inversion methods.** Use nonlinear least-squares with an iterative optimizer — directly seek the atmosphere best fitting the observed Stokes profile. Two subclasses existed by 1992:
   - **AHH-style Milne-Eddington inversion** (Auer et al. 1977; Landolfi, Landi Degl'Innocenti & Arena 1984; Skumanich & Lites 1987; Lites et al. 1988 extending to non-LTE; Murphy 1991 for multi-line): analytic $\chi^2$-derivatives; parameters are atmospheric "averages" over the line-forming region; unrealistic but robust.
   - **KAL** (Keller, Solanki et al. 1990): MHD + horizontal pressure balance inside small-scale flux tubes; fits only selected points of the spectrum; good for flux tubes but fails on asymmetric profiles arising from *velocity gradients* along the line of sight.

**KR.** 논문은 스토크스 인버전을 태양 분광학의 더 큰 목표 — **태양 대기 물리량의 신뢰성 있는 결정** — 안에 위치시키며 시작한다. 광구 흡수선의 제만 유도 편광은 자기장 벡터의 특권적 진단 수단이다. 1992년 당시 두 방법론 계열이 경쟁 중이었다:

1. **합성(synthesis) 방법 (AHH: Auer, Heasley & House 1977).** 시험 대기를 세우고 합성 스토크스를 계산한 뒤 관측과 비교하여 수작업으로 반복. 전체 RTE(비-LTE 포함)를 다루므로 정확하지만, 대기 전체 지식과 시행착오를 요구하므로 대량 데이터에는 비용이 과도.

2. **인버전(inversion) 방법.** 비선형 최소자승 + 반복적 최적화기로 관측 스토크스에 맞는 대기를 직접 추적. 1992년까지 두 하위 범주:
   - **AHH 계열 Milne-Eddington 인버전** (Auer 1977; Landolfi et al. 1984; Skumanich & Lites 1987; Lites et al. 1988 비-LTE; Murphy 1991 다선): 해석적 $\chi^2$-도함수; 선형성 영역 평균 파라미터; 비현실적이나 견고.
   - **KAL** (Keller, Solanki et al. 1990): 소규모 flux tube 내부의 MHD + 수평 압력평형; 스펙트럼 선별 샘플만 적합; flux tube에 잘 맞으나 시선 속도 경사에서 오는 비대칭을 못 다룸.

**EN.** Asymmetries in Stokes profiles are ubiquitous — sunspots (Illing, Landman & Mickey 1975), faculae (Sánchez Almeida, Collados & del Toro Iniesta 1989), and virtually everywhere. They *demand* recovery of depth stratification. This is the gap SIR fills.

**KR.** 스토크스 프로파일의 비대칭성은 흑점(Illing et al. 1975), 백반(Sánchez Almeida et al. 1989), 사실상 어디에서나 관측된다. 따라서 깊이 층상구조의 복원이 *필수적*이다. 바로 이 공백을 SIR이 메운다.

### Part II: Response Functions as Derivatives / 응답함수 — $\chi^2$-도함수의 열쇠 (§2, pp. 376-377)

**EN.** The heart of any least-squares inversion is evaluating derivatives of the merit function with respect to each free parameter $a_j$. The merit function (eq. 1):

$$\chi^2 = \frac{1}{\nu} \sum_{k=1}^{4} \sum_{i=1}^{M} \left[ I_k^\mathrm{obs}(\lambda_i) - I_k^\mathrm{syn}(\lambda_i) \right]^2$$

where $\nu$ = degrees of freedom (observations − free parameters), $k$ indexes Stokes components, $i$ indexes wavelength samples. The model atmosphere $\mathbf{a}$ is an $n \times p + r$ vector: $n$ depth points, $p$ depth-varying magnitudes (here $p=5$: $T, B, \gamma, \phi, v$), and $r$ depth-constant ones ($r = 2$: $\xi_\mathrm{mic}, \xi_\mathrm{mac}$).

**KR.** 모든 최소자승 인버전의 핵심은 각 자유 파라미터 $a_j$ 에 대한 적합도 함수의 도함수를 평가하는 것이다. 적합도 함수 (식 1):

$$\chi^2 = \frac{1}{\nu} \sum_{k=1}^{4} \sum_{i=1}^{M} \left[ I_k^\mathrm{obs}(\lambda_i) - I_k^\mathrm{syn}(\lambda_i) \right]^2$$

여기서 $\nu$ 는 자유도(관측 수 − 자유 파라미터 수), $k$ 는 스토크스 성분, $i$ 는 파장 샘플 인덱스. 모델 대기 $\mathbf{a}$ 는 $n \times p + r$ 차원 벡터: $n$ 개의 깊이 점, $p$ 개의 깊이-의존 물리량($p=5$: $T, B, \gamma, \phi, v$), $r$ 개의 깊이-무관량($r=2$: $\xi_\mathrm{mic}, \xi_\mathrm{mac}$).

**EN.** Evaluating $\nabla_\mathbf{a}\chi^2$ by finite differences would require $5n+2$ RTE integrations per iteration — prohibitive. AHH sidestepped this with analytical Milne-Eddington derivatives (§2). SIR's alternative: use **response functions** (Landi Degl'Innocenti & Landi Degl'Innocenti 1977; Mein 1971 for unpolarized, Beckers & Milkey 1975 and Landolfi & Landi Degl'Innocenti 1982 generalizing). The RF arises from a first-order perturbative analysis of the RTE: a perturbation $\delta x(\tau)$ in a single physical magnitude yields (eq. 3):

$$\delta \mathbf{I}(\lambda) = \int_0^\infty \mathbf{R}(\lambda, \tau) \, \delta x(\tau) \, d\tau$$

**KR.** $\nabla_\mathbf{a}\chi^2$ 를 차분법으로 평가하면 반복마다 $5n+2$ 회의 RTE 적분이 필요 — 금지 수준. AHH는 해석적 M-E 도함수로 우회했다. SIR의 대안은 **응답함수(RF)** 를 쓰는 것이다. RF는 RTE의 1차 섭동 해석으로부터 유도되며, 단일 물리량의 섭동 $\delta x(\tau)$ 가 아래와 같이 방출 스펙트럼 변화를 유발한다 (식 3):

$$\delta \mathbf{I}(\lambda) = \int_0^\infty \mathbf{R}(\lambda, \tau) \, \delta x(\tau) \, d\tau$$

**EN.** For a discrete $n$-point atmosphere parameterized with coefficients $c_j$ (here $c_j = \Delta\log\tau \ln 10 \cdot \tau_j$ for equi-spaced log-$\tau$ grids), eq. (4):

$$\delta I_k(\lambda_i) = \Delta\log\tau \ln 10 \sum_{j=1}^{n} c_j \tau_j R_k(\lambda_i, \tau_j) \delta a_j$$

Substituting into $\delta\chi^2$ (eqs. 5-6):

$$\delta\chi^2 = \frac{2}{\nu} \Delta\log\tau \ln 10 \sum_{j=1}^{n} \left\{ \sum_{k=1}^{4}\sum_{i=1}^{M}[I_k^\mathrm{obs}(\lambda_i) - I_k^\mathrm{syn}(\lambda_i)]\, c_j\tau_j\, R_k(\lambda_i, \tau_j) \right\} \delta a_j$$

This is precisely the $\chi^2$-gradient; the bracketed quantity is $\partial\chi^2/\partial a_j$. Thus **RFs directly give** the entire Marquardt gradient without extra RTE solves.

**KR.** 이산화된 $n$-점 대기에서 계수 $c_j$ 로 매개변수화하면 (여기서 $c_j = \Delta\log\tau \ln 10 \cdot \tau_j$, 등간격 log-$\tau$ 격자), 식 (4):

$$\delta I_k(\lambda_i) = \Delta\log\tau \ln 10 \sum_{j=1}^{n} c_j \tau_j R_k(\lambda_i, \tau_j) \delta a_j$$

이를 $\delta\chi^2$ 에 대입하면 (식 5-6):

$$\delta\chi^2 = \frac{2}{\nu} \Delta\log\tau \ln 10 \sum_{j=1}^{n} \left\{ \sum_{k=1}^{4}\sum_{i=1}^{M}[I_k^\mathrm{obs}(\lambda_i) - I_k^\mathrm{syn}(\lambda_i)]\, c_j\tau_j\, R_k(\lambda_i, \tau_j) \right\} \delta a_j$$

괄호 안의 양이 $\partial\chi^2/\partial a_j$. 따라서 **RF를 구하면 추가 RTE 풀이 없이** Marquardt 경사 전체가 바로 얻어진다.

### Part III: DELO Method and Sánchez Almeida RF / DELO 법과 산체스 알메이다 RF (§2.1, pp. 377)

**EN.** For the polarized RTE (vector Unno-Rachkovsky equation), Rees, Murphy & Durrant (1989) introduced the **DELO method** (Diagonal Element Lambda Operator) evaluating the evolution operator $\mathbf{O}(\tau, \tau')$ — a $4\times4$ matrix — to express the formal solution (eq. 7):

$$\mathbf{I}(\tau) = -\int_{\tau_0}^{\tau} \mathbf{O}(\tau, \tau')\mathbf{K}(\tau')\mathbf{S}(\tau')d\tau' + \mathbf{O}(\tau, \tau_0)\mathbf{I}(\tau_0)$$

where $\mathbf{K}$ is the $4 \times 4$ propagation matrix (absorption + magneto-optical terms) and $\mathbf{S}$ is the source function vector.

**KR.** 편광 RTE(벡터 Unno-Rachkovsky 방정식)에 대해 Rees, Murphy & Durrant (1989)는 진화 연산자 $\mathbf{O}(\tau, \tau')$(4×4 행렬)를 평가하는 **DELO 방법**(Diagonal Element Lambda Operator)을 도입하여 아래와 같은 형식해를 얻는다 (식 7):

$$\mathbf{I}(\tau) = -\int_{\tau_0}^{\tau} \mathbf{O}(\tau, \tau')\mathbf{K}(\tau')\mathbf{S}(\tau')d\tau' + \mathbf{O}(\tau, \tau_0)\mathbf{I}(\tau_0)$$

여기서 $\mathbf{K}$ 는 4×4 전파행렬(흡수 + 자기광학 효과), $\mathbf{S}$ 는 source-function 벡터.

**EN.** Sánchez Almeida (1992) — within the Landi formalism — derived a general RF expression (eq. 8):

$$\mathbf{R}(\tau, \lambda) = -\mathbf{O}(0, \tau) \left\{ \frac{\partial\mathbf{K}}{\partial x}(\mathbf{I} - \mathbf{S}) - \mathbf{K}\frac{\partial\mathbf{S}}{\partial x} \right\}$$

Because this requires exactly the same $\mathbf{O}, \mathbf{K}, \mathbf{S}$ as the DELO forward solution, RF evaluation is essentially "free" once the RTE is solved — this is the crucial efficiency trick.

**KR.** Sánchez Almeida (1992)는 Landi 형식론 내에서 범용 RF 표현을 유도했다 (식 8):

$$\mathbf{R}(\tau, \lambda) = -\mathbf{O}(0, \tau) \left\{ \frac{\partial\mathbf{K}}{\partial x}(\mathbf{I} - \mathbf{S}) - \mathbf{K}\frac{\partial\mathbf{S}}{\partial x} \right\}$$

이 표현은 DELO 정방향 풀이와 동일한 $\mathbf{O}, \mathbf{K}, \mathbf{S}$ 를 요구하므로, RTE를 한 번 풀면 RF 계산은 사실상 "무료" — 이것이 핵심 효율성 트릭이다.

**EN. Figure 1 interpretation.** Normalized RFs for Fe I $\lambda$6302.5 at $\Delta\lambda = 120$ mÅ from line center, computed in the umbral model E of Maltby et al. (1986), sampled at $\log\tau = 1.0, 0, -1.0, \ldots, -4.5$ for the six quantities $T, B, \gamma, \phi, v, \xi_\mathrm{mic}$. Key observations:
1. RFs **span over a broad range** in $\log\tau$ — Stokes profiles are sensitive to perturbations in multiple atmospheric layers, enabling depth-stratification recovery.
2. Different Stokes parameters show **different RF shapes** — $I$ (solid), $Q$ (dotted), $U$ (short-dashed), $V$ (long-dashed).
3. Sensitivity **varies strongly by physical magnitude** — temperature RF is an order of magnitude larger than other RFs (hence temperature inverts first / is the most informative).
4. The shape of RFs differs greatly among parameters — $T$-RF for Stokes $I$ is bipolar; $B$-RF for $V$ peaks at different depths than $B$-RF for $Q$.

**KR. Figure 1 해석.** Maltby et al. (1986)의 본영 모형 E에서 Fe I $\lambda$6302.5, 선 중심으로부터 $\Delta\lambda = 120$ mÅ 에서 계산된, $\log\tau = 1.0, 0, -1.0, \ldots, -4.5$ 에서 샘플된 여섯 양 $T, B, \gamma, \phi, v, \xi_\mathrm{mic}$ 의 정규화 RF. 핵심 관찰:
1. RF가 $\log\tau$ 상 **넓은 범위에 걸쳐 있음** — 스토크스 프로파일이 여러 대기 층의 섭동에 민감 → 깊이 층상구조 복원 가능.
2. 스토크스 파라미터마다 **RF 모양이 다름** — $I$(solid), $Q$(dotted), $U$(short-dashed), $V$(long-dashed).
3. 물리량별로 **민감도 크기가 크게 다름** — 온도 RF가 다른 물리량 RF보다 한 자릿수 이상 큼 → 온도가 가장 먼저 잘 복원되는 양.
4. 파라미터별 RF 모양이 크게 달라 — Stokes $I$의 $T$-RF는 쌍극성, Stokes $V$의 $B$-RF는 Stokes $Q$의 $B$-RF와 다른 깊이에서 최대.

### Part IV: Description of Inversion Method — Marquardt + Node Parameterization + Modified SVD / 인버전 방법: Marquardt + 노드 매개변수화 + 수정 SVD (§3, pp. 377-378)

**EN.** The Marquardt-Levenberg iterative solver works via (eq. 9):

$$\nabla\chi^2(\mathbf{a}) + \mathbf{A}\,\delta\mathbf{a} = \mathbf{0}$$

where $\mathbf{A}$ is the curvature matrix — the Hessian of $\chi^2$ approximated by pairwise products of RFs (Press et al. 1986 §15.5). Diagonal elements are multiplied by $(1 + \lambda)$; the damping parameter $\lambda$ interpolates between steepest descent ($\lambda \to \infty$) and Gauss-Newton ($\lambda \to 0$).

**KR.** Marquardt-Levenberg 반복해는 (식 9):

$$\nabla\chi^2(\mathbf{a}) + \mathbf{A}\,\delta\mathbf{a} = \mathbf{0}$$

여기서 $\mathbf{A}$ 는 곡률행렬 — $\chi^2$ 의 헤시안을 RF 쌍별 곱으로 근사 (Press et al. 1986 §15.5). 대각 원소에 $(1+\lambda)$ 를 곱하며, 감쇠 파라미터 $\lambda$ 는 최급강하($\lambda \to \infty$)와 Gauss-Newton($\lambda \to 0$) 사이를 보간한다.

**EN. Two obstacles.**
1. **Large dimensions.** With an optical-depth grid of $n = 40$ and 5 depth-varying magnitudes + 2 constants, $\dim(\mathbf{A}) = 202 \times 202$. Inverting this is expensive and the matrix is ill-conditioned.
2. **Ill-conditioning.** Some magnitudes produce no meaningful modification of $\chi^2$ in some layers — e.g., magnetic field strength below $\tau = 1$. Then eq. (9) is indeterminate in those parameters; SVD might throw them away.

**SIR's remedies:**
- **Node parameterization** (problem 1). Approximate each physical magnitude's depth perturbation by cubic-spline interpolation among $m$ equi-spaced nodes with $m \ll n$. This reduces the free-parameter count from $n$ to $m$ per magnitude (factor $n/m$). Start with small $m$ (e.g., 2), converge, then increase $m$ (e.g., 5, 9) — progressive refinement.
- **Equivalent RFs at nodes.** Crucially, node parameterization does NOT discard atmospheric information: the RFs at nodes are linear combinations of full-atmosphere RFs weighted by cubic-spline coefficients (Appendix A, see Mathematical Summary below).
- **Modified SVD** (problem 2). Standard SVD discards the smallest singular values, which tend to correspond to the least-sensitive physical magnitudes. For a problem dominated by temperature sensitivity, this would mean $B, \gamma, \phi, v, \xi$ get wiped out. SIR's modification: ensure every physical magnitude contributes at least one diagonal element to the inversion — mix in more than one singular direction per magnitude.

**KR. 두 장애물.**
1. **거대 행렬.** $n = 40$, 깊이 가변 5 + 상수 2면 $\dim(\mathbf{A}) = 202 \times 202$. 역행렬화가 비싸고 ill-conditioned.
2. **비정칙(ill-conditioning).** 어떤 물리량은 특정 층에서 $\chi^2$ 를 거의 변화시키지 못함 — 예: $\tau = 1$ 아래의 자기장 세기. 그러면 식 (9)는 해당 파라미터에 대해 미정(indeterminate); 표준 SVD는 이를 폐기할 수 있음.

**SIR의 해법:**
- **노드 매개변수화**(문제 1). 각 물리량의 깊이 섭동을 $m$ 개 등간격 노드 사이 큐빅 스플라인 보간으로 근사 ($m \ll n$). 물리량당 자유 파라미터 수를 $n \to m$ 으로 축소 (인자 $n/m$). 작은 $m$(예: 2)에서 시작하여 수렴 후 증가(5, 9) — 점진적 정제.
- **노드에서의 등가 RF.** 결정적으로, 노드화는 대기 정보를 버리지 않는다. 노드 RF는 전 대기 RF의 선형결합이며 가중치는 큐빅 스플라인 계수 (부록 A; 아래 수학적 요약 참조).
- **수정 SVD**(문제 2). 표준 SVD는 최소 특이값(최소 민감 파라미터에 해당)을 버린다. 온도 민감도가 우세한 문제에서는 $B, \gamma, \phi, v, \xi$ 모두 소멸. SIR의 수정: 모든 물리량이 역행렬화 과정에 최소 한 개 대각 원소로 기여하도록 강제 — 물리량마다 한 개 이상의 특이 방향을 섞어 넣음.

### Part V: Numerical Code Summary / 수치 코드 요약 (§4, pp. 378-379)

**EN.** Practical setup:
- **Normalization.** Observed Stokes spectrum normalized to local continuum intensity of HSRA (Harvard-Smithsonian Reference Atmosphere, Gingerich et al. 1971); arbitrary but matches standard spectropolarimetric practice.
- **Optical-depth grid.** Equi-spaced in $\log\tau$; nodes are a subset of grid points; endpoints always included.
- **Number of nodes.** Experiments tested locating nodes where RFs are largest; equi-spaced nodes proved the best compromise in computer time (no significant improvement from other distributions). In practice: iterate with $m=2$ (linear interpolation), then $m=5$, then $m=9$. More than 3 groups rarely needed.
- **Initialization of $\mathbf{a}_0$.**
  - $T(\tau)$: prescribed start (e.g., Hénoux 1969 model $+0$ K, $-300$ K, $+300$ K, $+600$ K, $[+400+140\log\tau]$, $[+600+150\log\tau]$).
  - $B(\tau), \gamma(\tau), \phi(\tau)$: automatically from observed Stokes via weak-field approximation (Jefferies & Mickey 1991). Typical initial values $B_0 \simeq 2000$ G, $\gamma_0 \simeq 10°$, $\phi_0 \simeq 20°$.
  - $v(\tau)$: constant from mean position of Stokes $V$ zero-crossings. Typical $v \simeq 1$ km/s.
  - Pressure: hydrostatic equilibrium at each iteration.
  - $\xi_\mathrm{mic}$ initialized to $1$ km/s, $\xi_\mathrm{mac}$ to $1.5$ km/s.

**Special RF features.**
1. Both synthesized $\mathbf{I}^\mathrm{syn}$ and RFs are **convolved with the Gaussian macroturbulent profile**.
2. For the two magnetic angles $\gamma, \phi$, RFs are computed in $\tan(\gamma/2)$ and $\tan(\phi/4)$ to avoid instabilities from periodicity of sinusoidal functions.
3. RFs for microturbulence are computed at $n$ grid points then averaged (since $\xi_\mathrm{mic}$ is assumed single-valued).
4. RF for $\xi_\mathrm{mac}$ cannot be strictly defined; instead $\chi^2$-derivative w.r.t. $\xi_\mathrm{mac}$ is computed as convolution of $\mathbf{I}^\mathrm{syn}$ with $\partial G/\partial \xi_\mathrm{mac}$ (Gaussian derivative).

**KR.** 실제 설정:
- **정규화.** 관측 스토크스 스펙트럼을 HSRA(Gingerich et al. 1971)의 국소 연속광 세기로 정규화; 임의적 선택이나 표준 분광편광 관측 관행과 부합.
- **광학심도 격자.** $\log\tau$ 등간격; 노드는 격자 점의 부분집합; 끝점은 항상 포함.
- **노드 수.** RF 최대점에 노드를 두는 시도도 했으나 등간격이 계산비용 대비 가장 효율. 실제로는 $m=2$(선형 보간), 이어 $m=5$, $m=9$ 로 점진 증가; 3 그룹 이상은 거의 불필요.
- **$\mathbf{a}_0$ 초기화.**
  - $T(\tau)$: Hénoux (1969) 모형 $+0$ K, $-300$ K, $+300$ K, $+600$ K, $[+400+140\log\tau]$, $[+600+150\log\tau]$ 등으로 지정.
  - $B(\tau), \gamma(\tau), \phi(\tau)$: 약자기장 근사(Jefferies & Mickey 1991)로 자동. 전형적 $B_0 \simeq 2000$ G, $\gamma_0 \simeq 10°$, $\phi_0 \simeq 20°$.
  - $v(\tau)$: Stokes $V$ 영점 교차 평균 위치에서 상수. 전형 $v \simeq 1$ km/s.
  - 압력: 매 반복 정수학적 평형.
  - $\xi_\mathrm{mic}$ 초기값 1 km/s, $\xi_\mathrm{mac}$ 1.5 km/s.

**RF 계산의 특수 처리.**
1. 합성 $\mathbf{I}^\mathrm{syn}$ 과 RF 모두 **Gaussian 거시난류 프로파일과 컨볼루션**.
2. 자기장 각도 $\gamma, \phi$ 에 대해서는 주기성 불안정을 피하기 위해 $\tan(\gamma/2)$, $\tan(\phi/4)$ 에서 RF 계산.
3. 미시난류 RF는 $n$ 격자점에서 각각 계산 후 평균 (단일값 가정).
4. 거시난류 RF는 엄밀히 정의 불가; 대신 $\partial\chi^2/\partial\xi_\mathrm{mac}$ 를 $\mathbf{I}^\mathrm{syn}$ 과 $\partial G/\partial\xi_\mathrm{mac}$ (가우시안 도함수)의 컨볼루션으로 계산.

### Part VI: Numerical Results / 수치 결과 (§5, pp. 379-383)

#### §5.1 Reference Model Atmosphere

**EN.** Synthetic "observations" were generated from Maltby et al. (1986) umbral model E, sampled $\Delta\log\tau = 0.1$ from $\log\tau = 1.2$ down to $-3.6$. The depth run of the test atmosphere:

$$B(\tau) = \begin{cases} 1000 \text{ G}, & \log\tau < -2 \\ 3000 + 1000\log\tau \text{ G}, & -2 \le \log\tau \le 0 \\ 3000 \text{ G}, & \log\tau > 0 \end{cases}$$

$$\gamma(\tau) = \begin{cases} 75°, & \log\tau < -2 \\ 15 - 30\log\tau \text{ degrees}, & -2 \le \log\tau \le 0 \\ 15°, & \log\tau > 0 \end{cases}$$

$$\phi(\tau) = \begin{cases} 0°, & \log\tau < -2 \\ 90 + 45\log\tau \text{ degrees}, & -2 \le \log\tau \le 0 \\ 90°, & \log\tau > 0 \end{cases}$$

$$v(\tau) = \begin{cases} 0 \text{ km/s}, & \log\tau < -2 \\ (2 + \log\tau) \text{ km/s}, & -2 \le \log\tau \le 0 \\ 2 \text{ km/s}, & \log\tau > 0 \end{cases}$$

Micro- and macroturbulence fixed at 0.6 and 0.75 km/s. Smoothed to avoid derivative discontinuities. Figure 2 shows model-atmosphere lines (filled circles).

**KR.** Maltby et al. (1986) 본영 모형 E에서 합성 "관측"을 $\Delta\log\tau = 0.1$ 간격, $\log\tau = 1.2$ 에서 $-3.6$ 까지 샘플링하여 생성. 테스트 대기의 깊이 의존성:

$$B(\tau) = \begin{cases} 1000 \text{ G}, & \log\tau < -2 \\ 3000 + 1000\log\tau \text{ G}, & -2 \le \log\tau \le 0 \\ 3000 \text{ G}, & \log\tau > 0 \end{cases}$$

$$\gamma(\tau) = \begin{cases} 75°, & \log\tau < -2 \\ 15 - 30\log\tau \text{도}, & -2 \le \log\tau \le 0 \\ 15°, & \log\tau > 0 \end{cases}$$

$$\phi(\tau) = \begin{cases} 0°, & \log\tau < -2 \\ 90 + 45\log\tau \text{도}, & -2 \le \log\tau \le 0 \\ 90°, & \log\tau > 0 \end{cases}$$

$$v(\tau) = \begin{cases} 0 \text{ km/s}, & \log\tau < -2 \\ (2 + \log\tau) \text{ km/s}, & -2 \le \log\tau \le 0 \\ 2 \text{ km/s}, & \log\tau > 0 \end{cases}$$

미시/거시 난류는 0.6, 0.75 km/s 고정. 도함수 불연속 회피용 평활화.

#### §5.2–5.3 Starting Models and First Results

**EN.** Six starting model atmospheres (Hénoux 1969 umbral baseline perturbed as described in briefing, Fig. 3) were tried to check uniqueness. $B, \gamma, \phi, v, \xi$ all initialized to constant values. Six spectral lines (Table 1) used: $\lambda = 4574.22, 4798.73, 5127.68, 5253.03, 5522.45, 6302.51$ Å, with excitation potentials 0.05–4.21 eV, $\log(gf)$ from −0.58 to −6.06, $g_\mathrm{LS}$ from 1.0 to 2.5; sampled every 20 mÅ.

**Results (Fig. 2; noise-free case).** Open circles (mean over six initializations) vs filled circles (reference). All physical magnitudes recovered with high accuracy in the range $\log\tau \in [-2.6, 0.2]$:
- **Temperature**: rms deviation **6 K**
- **Magnetic field strength**: rms **25 G**
- **$\gamma$**: rms **0.°5**
- **$\phi$**: rms **2°**
- **Line-of-sight velocity**: rms **0.03 km/s**
- $\xi_\mathrm{mic}$ recovered as 0.63 km/s (reference 0.60)
- $\xi_\mathrm{mac}$ recovered as 0.73 km/s (reference 0.75)
- Output-vs-reference Stokes spectra differ by $1 \times 10^{-3}$ rms in continuum units.

**KR.** Hénoux (1969) 본영 기저를 브리핑에 기술한 대로 섭동한 6개 시작 모형(Fig. 3)으로 고유성 점검. $B, \gamma, \phi, v, \xi$ 는 모두 상수로 초기화. Table 1의 6개 스펙트럼 선 사용: $\lambda = 4574.22, 4798.73, 5127.68, 5253.03, 5522.45, 6302.51$ Å, 여기 포텐셜 0.05–4.21 eV, $\log(gf)$ −0.58 ~ −6.06, $g_\mathrm{LS}$ 1.0–2.5; 20 mÅ 간격 샘플.

**결과 (Fig. 2; 무노이즈).** $\log\tau \in [-2.6, 0.2]$ 에서 모든 물리량이 고정밀로 복원:
- **온도**: rms **6 K**
- **자기장 세기**: rms **25 G**
- **$\gamma$**: rms **0.°5**
- **$\phi$**: rms **2°**
- **시선속도**: rms **0.03 km/s**
- $\xi_\mathrm{mic}$: 0.63 km/s (참조 0.60)
- $\xi_\mathrm{mac}$: 0.73 km/s (참조 0.75)
- 출력 vs 참조 스토크스 차이는 연속광 단위 $1 \times 10^{-3}$ rms.

#### §5.4 Dependence on S/N Ratio

**EN.** White noise added to yield continuum-intensity S/N = 250 (lower bound for spectropolarimetric practice). Inversion of these noisy profiles (Fig. 4 shows Fe I $\lambda$6302.5 Stokes $I, Q, U, V$; dotted = reference, open circles = noisy input, solid = recovered). The recovered spectrum and output profiles are extremely similar to the reference; noise seems not to influence recovery. **rms deviations**:
- Output-vs-reference spectra: $1.4 \times 10^{-3}$
- $T$: **10 K**, $B$: **70 G**, $v$: **0.04 km/s**, $\gamma$: **3°**, $\phi$: **2°**
- $\xi_\mathrm{mic}$: 0.56 km/s, $\xi_\mathrm{mac}$: 0.75 km/s

**KR.** 연속광 S/N = 250 (실제 분광편광 관측의 최저 허용 수준)의 백색 노이즈를 추가. Fig. 4 는 Fe I $\lambda$6302.5의 $I, Q, U, V$ (점선 = 참조, 공원 = 노이즈 입력, 실선 = 복원). 복원 스펙트럼과 출력 프로파일이 참조와 거의 동일; 노이즈의 영향이 거의 없음. **rms 편차**:
- 출력 vs 참조 스펙트럼: $1.4 \times 10^{-3}$
- $T$: **10 K**, $B$: **70 G**, $v$: **0.04 km/s**, $\gamma$: **3°**, $\phi$: **2°**
- $\xi_\mathrm{mic}$: 0.56 km/s, $\xi_\mathrm{mac}$: 0.75 km/s

#### §5.5 Dependence on Number of Spectral Lines

**EN.** Testing with **only Fe I $\lambda\lambda$6301.5, 6302.5** (common spectropolarimetric pair). S/N = 250, results still excellent — rms in output-vs-reference spectra $2.2 \times 10^{-3}$. Recovered-vs-reference rms (same $\log\tau$ range): $T$: **30 K**, $B$: **100 G**, $v$: **0.15 km/s**, $\gamma$: **3°**, $\phi$: **7°**; $\xi_\mathrm{mic}$: 0.66 km/s, $\xi_\mathrm{mac}$: 0.74 km/s. Modestly worse than six lines but still usable.

**KR.** **Fe I $\lambda\lambda$6301.5, 6302.5** 쌍만으로 테스트 (분광편광 관측의 표준 쌍). S/N = 250. 여전히 우수한 결과 — 출력 vs 참조 스펙트럼 rms $2.2 \times 10^{-3}$. 복원 vs 참조 rms (동일 $\log\tau$ 범위): $T$: **30 K**, $B$: **100 G**, $v$: **0.15 km/s**, $\gamma$: **3°**, $\phi$: **7°**; $\xi_\mathrm{mic}$: 0.66 km/s, $\xi_\mathrm{mac}$: 0.74 km/s. 6선 대비 약간 열등하나 실사용 가능.

#### §5.6 Dependence on Number of Stokes Parameters — Only $I, V$

**EN.** Invert only Stokes $I$ and $V$ of Fe I $\lambda\lambda$6301.5, 6302.5 (S/N = 250). Field initialized to $B = 1500$ G, $\gamma = 10°$, $\phi = 1$ km/s for $v$. Fig. 5 shows recovered mean — the range for which the model is uniquely determined has slightly shrunk (log$\tau < -0.4$). But in that range the results are still good **even for $\phi$**, remarkably. rms: $T$: **13 K**, $B$: **110 G**, $v$: **0.1 km/s**, $\gamma$: **4°**, $\phi$: **15°**; $\xi_\mathrm{mic}$: 0.72 km/s, $\xi_\mathrm{mac}$: 0.70 km/s. Output-spectrum rms: $3 \times 10^{-3}$. **Key finding**: even though $\phi$ is traditionally believed to be carried by $Q, U$ (linear polarization), inverting only $I, V$ can recover $\phi$ to ~15° precision — because magneto-optical effects couple $\phi$ into $I, V$ when $B$ is strong.

**KR.** Fe I $\lambda\lambda$6301.5, 6302.5의 Stokes $I, V$ 만으로 인버전 (S/N = 250). 자기장은 $B = 1500$ G, $\gamma = 10°$, $v = 1$ km/s 로 초기화. Fig. 5 는 복원된 평균 — 유일 결정 범위가 log$\tau < -0.4$ 로 다소 축소. 그러나 그 범위에서 결과는 여전히 양호, **심지어 $\phi$ 에 대해서도**! rms: $T$: **13 K**, $B$: **110 G**, $v$: **0.1 km/s**, $\gamma$: **4°**, $\phi$: **15°**; $\xi_\mathrm{mic}$: 0.72 km/s, $\xi_\mathrm{mac}$: 0.70 km/s. 출력 스펙트럼 rms: $3 \times 10^{-3}$. **핵심 발견**: 전통적으로 $\phi$ 는 $Q, U$(선형편광)에만 정보가 있다고 믿어왔지만, $I, V$ 만으로도 ~15° 정도로 $\phi$ 복원 가능 — 강자기장에서 자기광학 효과가 $\phi$ 를 $I, V$ 로 커플링하기 때문.

### Part VII: Conclusions and Computing Time / 결론 및 계산시간 (§6, p. 383)

**EN.** Quoted benchmark: inverting $I, Q, U, V$ for 4 spectral lines × 250 wavelength points ($\nu = 4000$ data points) takes **~50 minutes of CPU time** on a Data General Eclipse 20000 (a common 1990-era workstation). Reduces to **~15 minutes** when inverting only $I, V$ for two lines. These times established SIR as practical for routine analysis on workstations.

**Key takeaways stated by the authors:**
1. Full RTE integrated by DELO — no approximation beyond LTE.
2. Modified-SVD inside Marquardt — robust against ill-conditioning.
3. RFs computed once per iteration step — fast convergence.
4. Results independent of starting atmosphere (six different starts gave same result).
5. Magneto-optical effects make $I, V$ alone sufficient for azimuth recovery to within a degree or two (in favorable conditions).

**KR.** 인용 벤치마크: Data General Eclipse 20000 워크스테이션에서 $I, Q, U, V$, 4개 선 × 250 파장점 ($\nu = 4000$) 인버전에 **~50분 CPU**. $I, V$, 2개 선만이면 **~15분**. 1990년대 초 워크스테이션에서 실용적.

**저자 결론:**
1. 전체 RTE를 DELO로 적분 — LTE 이외 근사 없음.
2. Marquardt 내부의 수정 SVD — 비정칙에 견고.
3. RF를 반복 단계당 한 번만 계산 — 빠른 수렴.
4. 시작 대기에 독립적 (6가지 시작 모두 동일 결과).
5. 자기광학 효과로 $I, V$ 만으로도 $\phi$ 복원 가능 (~몇 도 정밀도).

---

## 3. Key Takeaways / 핵심 시사점

1. **Response functions are the computational keystone of fast Stokes inversion / 응답함수는 고속 스토크스 인버전의 계산 핵심이다.** — By deriving $\partial I_k(\lambda)/\partial x_m(\tau)$ analytically from the DELO formal solution (Sánchez Almeida 1992), a single forward RTE pass delivers the full Marquardt gradient. This replaces $5n+2$ separate finite-difference RTE integrations per iteration, yielding orders-of-magnitude speedups. / DELO 형식해로부터 $\partial I_k(\lambda)/\partial x_m(\tau)$ 를 해석적으로 유도(Sánchez Almeida 1992)함으로써, 단일 정방향 RTE 풀이로 Marquardt 경사 전체를 얻을 수 있다. 반복마다 $5n+2$ 번의 차분법 RTE 적분이 필요했던 기존 방식 대비 수 자릿수 속도 향상.

2. **Node parameterization trades formal accuracy for ill-conditioning control / 노드 매개변수화는 정밀도 일부를 양보해 비정칙성을 통제한다.** — Reducing $n$ depth grid points to $m$ equi-spaced nodes ($m = 2, 5, 9$) with cubic-spline interpolation cuts the curvature matrix from $\sim 200 \times 200$ to $\sim 50 \times 50$ without losing information (equivalent RFs). Progressive $m$ refinement schedule (start $m=2$, converge, raise $m$) avoids overfitting. / $n$ 개 깊이 격자점을 $m$ 개 등간격 노드($m = 2, 5, 9$)와 큐빅 스플라인 보간으로 축소하면 곡률행렬이 $\sim 200 \times 200$ 에서 $\sim 50 \times 50$ 으로 감소 (등가 RF 덕에 정보 손실 없음). $m$ 을 점진적으로 늘리는 일정(2 → 5 → 9)이 과적합 회피.

3. **Modified SVD preserves all physical magnitudes against sensitivity disparity / 수정 SVD는 민감도 불균형 하에서 모든 물리량을 보존한다.** — Standard SVD would discard the smallest singular values, which correspond to least-sensitive magnitudes (e.g., $\xi_\mathrm{mac}$, $\phi$ at some depths). SIR forces each physical magnitude to retain at least one diagonal element — all parameters participate in the Marquardt step. / 표준 SVD는 최소 특이값(최소 민감도 물리량, 예: $\xi_\mathrm{mac}$ 또는 일부 층의 $\phi$ 에 해당)을 버린다. SIR은 각 물리량마다 최소 한 개 대각 원소가 역행렬화에 참여하도록 강제하여 모든 파라미터가 반복 단계에 기여.

4. **Depth stratification is recoverable in a finite "information-rich" range / 깊이 층상구조는 유한한 "정보-풍부" 범위에서 복원 가능하다.** — In the numerical tests, all five depth-varying magnitudes are tightly recovered in $\log\tau \in [-2.6, 0.2]$. Outside this range, the RFs die off and error bars diverge. For Fe I $\lambda$6301.5, 6302.5 alone, the useful range narrows to $\log\tau < -0.4$. **Practical lesson**: report stratification only in the RF-sensitive interval. / 수치 테스트에서 다섯 깊이-가변량 모두 $\log\tau \in [-2.6, 0.2]$ 에서 치밀하게 복원된다. 이 범위 밖에서는 RF가 소멸하고 오차 막대가 발산. Fe I $\lambda$6301.5, 6302.5 만 쓸 경우 범위가 $\log\tau < -0.4$ 로 축소. **실용적 교훈**: 층상구조는 RF-민감 구간에서만 보고하라.

5. **Uniqueness verified by multiple initializations / 다중 초기화로 고유성 검증된다.** — Starting from six very different atmospheres (Hénoux 1969 ± 300, 600 K etc.), SIR converges to the same result within the information-rich $\log\tau$ range. This empirically demonstrates the inversion is well-posed in that range. Error bars in Figures 2, 4, 5 show scatter across initializations. / 매우 다른 6개 시작 대기(Hénoux 1969 ± 300, 600 K 등)에서 출발해도 정보-풍부 $\log\tau$ 범위 내에서 동일 결과로 수렴. 해당 범위에서 인버전이 잘 정의(well-posed)임을 경험적 증명. Fig. 2, 4, 5 의 오차 막대가 초기화 간 산포를 보인다.

6. **$I, V$-only inversion recovers azimuth $\phi$ — contrary to folklore / $I, V$ 만으로도 방위각 $\phi$ 복원 가능 — 통설에 반하여.** — In strong magnetic fields the magneto-optical effect couples $\phi$ into Stokes $I$ and $V$ through the propagation matrix off-diagonal terms $\rho_U, \rho_V$. Even when $Q, U$ are unavailable (low S/N or 1D polarization), azimuth can be retrieved to ~15° precision. This challenges the dogma that "$\phi$ lives only in linear polarization." / 강자기장에서 자기광학 효과는 전파행렬 off-diagonal 항 $\rho_U, \rho_V$ 를 통해 $\phi$ 를 $I, V$ 로 커플링한다. $Q, U$ 가 없어도(저 S/N 혹은 1차원 편광계) $\phi$ 를 ~15° 정밀도로 복원 가능. "$\phi$ 는 선형편광에만 있다"는 통념을 반박.

7. **Computing time proved the method practical on 1990s hardware / 1990년대 하드웨어에서 실용성 입증.** — ~50 minutes for 4 lines × 4 Stokes × 250 wavelengths on a Data General Eclipse 20000, ~15 minutes for 2 lines × 2 Stokes. Modern processors (GHz vs ~MHz) achieve <10 seconds per pixel. For pixel-wise inversion of full-map data (e.g., Hinode SP rasters with $10^6$ pixels), this remains a 2–3 day problem; hence modern parallel SIR and ML surrogates. / 1990년대 Eclipse 20000에서 4선×4-스토크스×250파장에 ~50분, 2선×2-스토크스에 ~15분. 현대 GHz 프로세서에서는 픽셀당 10초 미만. Hinode SP 풀맵($10^6$ 픽셀) 인버전은 여전히 2–3일 문제 → 병렬 SIR과 ML 대리모델의 동기.

8. **SIR set the template that all modern inversion codes follow / SIR이 현대 인버전 코드의 표준을 세웠다.** — NICOLE (non-LTE), STiC (non-LTE + PRD), HAZEL (He I chromosphere), SPINOR, SIRJUMP, SIRGAUS all inherit: (i) RF-based gradient, (ii) node parameterization, (iii) Marquardt minimizer, (iv) SVD-like regularization. The 2016 LRSP review by del Toro Iniesta & Ruiz Cobo codifies the resulting "standard" framework. / NICOLE(비-LTE), STiC(비-LTE + PRD), HAZEL(He I 채층), SPINOR, SIRJUMP, SIRGAUS 모두 (i) RF 경사, (ii) 노드 매개변수화, (iii) Marquardt, (iv) SVD-류 정규화 를 계승. 2016년 LRSP 리뷰(del Toro Iniesta & Ruiz Cobo)가 이 "표준" 틀을 체계화.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 The Polarized Radiative Transfer Equation / 편광 복사전달방정식

The Unno-Rachkovsky equation for the four-component Stokes vector $\mathbf{I} = (I, Q, U, V)^T$:

$$\frac{d\mathbf{I}}{d\tau} = \mathbf{K}(\mathbf{I} - \mathbf{S})$$

where $\mathbf{K}$ is the $4 \times 4$ absorption/dispersion matrix:

$$\mathbf{K} = \begin{pmatrix} \eta_I & \eta_Q & \eta_U & \eta_V \\ \eta_Q & \eta_I & \rho_V & -\rho_U \\ \eta_U & -\rho_V & \eta_I & \rho_Q \\ \eta_V & \rho_U & -\rho_Q & \eta_I \end{pmatrix}$$

with $\eta_{I, Q, U, V}$ the polarized absorption coefficients and $\rho_{Q, U, V}$ the magneto-optical terms. $\mathbf{S} = (S_I, 0, 0, 0)^T$ in LTE (the Planck function at $T$).

**Formal solution via DELO evolution operator** (eq. 7):

$$\mathbf{I}(\tau) = -\int_{\tau_0}^{\tau} \mathbf{O}(\tau, \tau') \mathbf{K}(\tau') \mathbf{S}(\tau') d\tau' + \mathbf{O}(\tau, \tau_0) \mathbf{I}(\tau_0)$$

### 4.2 Response Functions / 응답함수

Definition: for a physical magnitude $x(\tau)$, the RF to Stokes $k$ at wavelength $\lambda$ is:

$$R_k(\lambda, \tau) = \frac{\partial I_k(\lambda)}{\partial x(\tau)}$$

Sánchez Almeida (1992) closed-form expression (eq. 8):

$$\mathbf{R}(\tau, \lambda) = -\mathbf{O}(0, \tau) \left\{ \left[\frac{\partial \mathbf{K}}{\partial x}\right](\tau) \left[\mathbf{I}(\tau) - \mathbf{S}(\tau)\right] - \mathbf{K}(\tau) \left[\frac{\partial \mathbf{S}}{\partial x}\right](\tau) \right\}$$

Interpretation: the RF traces how a local perturbation $\delta x(\tau)$ modifies the emergent Stokes spectrum through: (a) change in propagation matrix $\partial\mathbf{K}/\partial x$ acting on the local contrast $(\mathbf{I} - \mathbf{S})$, (b) change in source function $\partial\mathbf{S}/\partial x$ acting through the local absorption $\mathbf{K}$; the effect is then propagated from depth $\tau$ to the surface by the evolution operator $\mathbf{O}(0, \tau)$.

### 4.3 Merit Function / 적합도 함수

$$\chi^2(\mathbf{a}) = \frac{1}{\nu} \sum_{k=1}^{4} \sum_{i=1}^{M} \left[I_k^\mathrm{obs}(\lambda_i) - I_k^\mathrm{syn}(\lambda_i;\,\mathbf{a})\right]^2$$

$\nu$ = degrees of freedom = $4M - (\text{number of free parameters})$.

### 4.4 Linearization and $\chi^2$ Gradient / 선형화와 $\chi^2$ 경사

From eqs. (4)-(6), discretizing the atmosphere on an equi-spaced $\log\tau$ grid with coefficients $c_j$:

$$\delta I_k(\lambda_i) = \Delta\log\tau \cdot \ln 10 \sum_{j=1}^{n} c_j \tau_j R_k(\lambda_i, \tau_j)\, \delta a_j$$

$$\frac{\partial \chi^2}{\partial a_j} = -\frac{2}{\nu} \Delta\log\tau \cdot \ln 10 \sum_{k=1}^{4}\sum_{i=1}^{M} [I_k^\mathrm{obs}(\lambda_i) - I_k^\mathrm{syn}(\lambda_i)] \, c_j \tau_j\, R_k(\lambda_i, \tau_j)$$

### 4.5 Marquardt-Levenberg Step / Marquardt-Levenberg 단계

System of linear equations (eq. 9):

$$(\mathbf{A} + \lambda \text{diag}(\mathbf{A})) \, \delta\mathbf{a} = -\nabla_\mathbf{a}\chi^2$$

Curvature matrix elements:

$$A_{jj'} \approx \frac{2}{\nu} (\Delta\log\tau \cdot \ln 10)^2 \sum_{k,i} c_j \tau_j c_{j'} \tau_{j'} R_k(\lambda_i, \tau_j) R_k(\lambda_i, \tau_{j'})$$

Damping scalar $\lambda$: large $\lambda \to$ pure gradient descent; small $\lambda \to$ Gauss-Newton. Adapted each iteration: if $\chi^2$ improves, $\lambda \to \lambda/10$; else $\lambda \to 10\lambda$.

### 4.6 Node Parameterization and Equivalent RFs / 노드 매개변수화와 등가 RF (Appendix A)

Perturbations at $m$ equi-spaced nodes: $\delta y_l \equiv \delta a_q$, with $q = 1 + (l-1)(n-1)/(m-1)$, $l = 1, \ldots, m$. Cubic-spline interpolation:

$$\delta a_j \simeq \sum_{l=1}^{m} f_{j,l} \, \delta y_l$$

where $f_{j,l}$ are spline coefficients. Substituting into eq. (4):

$$\delta I_k(\lambda_i) \simeq \Delta\log\tau \cdot \ln 10 \sum_{l=1}^{m} \sum_{j=1}^{n} c_j \tau_j f_{j,l} R_k(\lambda_i, \tau_j)\, \delta y_l$$

Defining **equivalent RFs** at nodes (eq. 14):

$$\boxed{\mathcal{R}_k(\lambda_i, \tau_l) \equiv \Delta\log\tau \cdot \ln 10 \sum_{j=1}^{n} c_j \tau_j f_{j,l} R_k(\lambda_i, \tau_j)}$$

— a linear combination of full-atmosphere RFs weighted by spline coefficients. The modified merit-function variation (eq. 16):

$$\delta\chi^2 \simeq \frac{2}{\nu} \sum_{l=1}^{m} \sum_{k,i} [I_k^\mathrm{obs}(\lambda_i) - I_k^\mathrm{syn}(\lambda_i)] \, \mathcal{R}_k(\lambda_i, \tau_l) \, \delta y_l$$

**Degrees of freedom** change: $\nu' = 4M - (5m + 2)$, far smaller parameter count than $\nu = 4M - (5n + 2)$.

### 4.7 Modified SVD for Curvature Inversion / 곡률행렬 역행렬화용 수정 SVD (Appendix B)

Standard SVD factorization $\mathbf{M} = \mathbf{U} \mathbf{W} \mathbf{V}^T$ with singular values $w_{jj}$:

$$\mathbf{M}^{-1} = \mathbf{V}\, \text{diag}(1/w_{jj}) \, \mathbf{U}^T$$

Threshold tolerance: $1/w_{jj} \equiv 0$ if $w_{jj} \le \epsilon \max\{w_{jj}\}$, $\epsilon \sim 10^{-3}$ to $10^{-6}$.

**SIR modification**: since smallest singular values tend to align with least-sensitive magnitudes, standard truncation loses them. Because in eq. (20) the state vector is

$$\mathbf{x} = (x_1, \ldots, x_m, x_{m+1}, \ldots, x_{2m}, \ldots, x_{pm}, x_{pm+1}, \ldots, x_{pm+r})^T$$

SIR enforces: for every physical magnitude $p'$ (i.e., every $m$-block), at least one diagonal element must be inverted — every magnitude participates in the update step.

### 4.8 Typical Parameter Budget / 전형적 파라미터 예산

| Quantity / 물리량 | Symbol / 기호 | Depth-dependent? / 깊이-의존? | Typical nodes $m$ / 전형 노드 수 |
|---|---|---|---|
| Temperature / 온도 | $T(\tau)$ | Yes / 예 | 5–9 |
| Magnetic field strength / 자기장 세기 | $B(\tau)$ | Yes / 예 | 2–5 |
| Magnetic inclination / 자기장 경사각 | $\gamma(\tau)$ | Yes / 예 | 2–5 |
| Magnetic azimuth / 자기장 방위각 | $\phi(\tau)$ | Yes / 예 | 2–5 |
| Line-of-sight velocity / 시선속도 | $v(\tau)$ | Yes / 예 | 2–5 |
| Microturbulence / 미시난류속도 | $\xi_\mathrm{mic}$ | No / 아니오 | 1 |
| Macroturbulence / 거시난류속도 | $\xi_\mathrm{mac}$ | No / 아니오 | 1 |
| **Total** / **합** | | | $5m + 2 \approx 25$–$47$ |

### 4.9 Uncertainty Estimation / 불확실도 추정

After convergence, parameter covariance is approximately $\mathbf{C} \approx \mathbf{A}^{-1}$ (inverse curvature matrix, with $\lambda \to 0$):

$$\sigma_j^2 = C_{jj} = [\mathbf{A}^{-1}]_{jj}$$

In practice, standard errors are scaled by reduced $\chi^2$: $\sigma_j \to \sigma_j \sqrt{\chi^2_\mathrm{reduced}}$.

### 4.10 Worked Numerical Example / 구체 수치 예시

For the reference inversion of §5.3 (6 lines, no noise) of an umbral Fe I $\lambda$6302.5 line at $\log\tau = 0$:
- Reference: $T = 4500$ K, $B = 3000$ G, $\gamma = 15°$, $\phi = 90°$, $v = 2$ km/s
- Initialization (Hénoux +0 K): $T_0 \simeq 5800$ K, $B_0 \simeq 2000$ G, $\gamma_0 \simeq 10°$, $\phi_0 \simeq 20°$, $v_0 \simeq 1$ km/s
- After convergence: $T = 4506$ K (rms 6 K), $B = 3025$ G (rms 25 G), $\gamma = 15.5°$ (rms 0.5°), $\phi = 92°$ (rms 2°), $v = 2.03$ km/s (rms 0.03 km/s)
- Iterations: ~15–30 Marquardt steps, mostly at $\lambda$ between $10^{-3}$ and $10^1$.
- $\chi^2$ at convergence: $\chi^2 \simeq 10^{-6}$ (noise-free) or $\simeq 10^{-5}$ (S/N=250).

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
                  FOUNDATIONS
                  ===========

1971 ─ Mein — weighting functions for unpolarized RTE
   │
1975 ─ Beckers & Milkey — generalized RFs
   │
1977 ─ Landi Degl'Innocenti & Landi Degl'Innocenti — polarized RFs
   │
1977 ─ Auer, Heasley & House (AHH) — Milne-Eddington inversion
   │
1982 ─ Landi Degl'Innocenti & Landolfi — RFs for thermodynamic diagnostics
   │
1984 ─ Landolfi, Landi & Arena — extended AHH (damping, magneto-optical)
   │
1985 ─ Landi Degl'Innocenti — evolution operator formalism
   │
1986 ─ Press et al. — Numerical Recipes (SVD, Marquardt)
   │
1987 ─ Skumanich & Lites — AHH extension
   │
1989 ─ Rees, Murphy & Durrant — DELO method for polarized RTE
   │
1990 ─ Keller, Solanki et al. (KAL) — MHD flux-tube inversion
   │
1991 ─ Jefferies & Mickey — weak-field initialization
   │
1992 ─ Sánchez Almeida — general RF expression
   │
1992 ─┐
      │    ╔═══════════════════════════════════════════════════╗
      ├──> ║  THIS PAPER: Ruiz Cobo & del Toro Iniesta 1992    ║
      │    ║                 SIR CODE                          ║
      │    ╚═══════════════════════════════════════════════════╝
      │
1994 ─ del Toro Iniesta & Ruiz Cobo — user guide and applications
      │
1998 ─ Socas-Navarro, Trujillo Bueno, Ruiz Cobo — non-LTE extension (prelude to NICOLE)
      │
2000 ─ Socas-Navarro (SPINOR) — weak-field tests
      │
2007 ─ Tsuneta et al. — Hinode SP launch (drives pipeline demand)
      │
2015 ─ de la Cruz Rodríguez et al. — NICOLE public release
      │
2016 ─┐
      │    ┌───────────────────────────────────────────────────┐
      ├──> │  del Toro Iniesta & Ruiz Cobo, LRSP review        │  ← Paper #13 LRSP in this reading series
      │    └───────────────────────────────────────────────────┘
      │
2019 ─ de la Cruz Rodríguez et al. — STiC (non-LTE + PRD)
      │
2020 ─ Rimmele et al. — DKIST first light
      │
2021 ─ Asensio Ramos & Díaz Baso — machine-learning inverters built on SIR benchmarks
      │
  (present) — SIR still the reference code for routine SP data reduction
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Paper #13 LRSP — del Toro Iniesta & Ruiz Cobo 2016, "Inversion of Stokes Profiles" LRSP review | Direct successor / 직접적 계승 — the 2016 review generalizes and updates the 1992 SIR paper, covering NICOLE, HAZEL, STiC, ML inverters / 2016 LRSP 리뷰가 1992 SIR 논문을 일반화·갱신하며 NICOLE, HAZEL, STiC, ML 인버전을 포괄 | Essential pair reading; 1992 = foundation, 2016 = modern synthesis / 필수 세트; 1992 = 기초, 2016 = 현대 종합 |
| Tsuneta et al. 2008 (#14 SO) — Hinode/SP instrument | Instrument / 기기 — SIR became the default inverter for Hinode SP pixel-wise analysis / SIR이 Hinode SP 픽셀별 분석의 기본 인버전 | Every Hinode SP science paper uses SIR or descendants / 모든 Hinode SP 과학 논문이 SIR 혹은 후손 코드 사용 |
| Rimmele et al. 2020 (#23 SO) — DKIST | Instrument / 기기 — ViSP and DL-NIRSP data pipelines rely on SIR-family inverters for magnetic-field vector recovery / ViSP, DL-NIRSP 파이프라인이 자기장 벡터 복원에 SIR 계열 인버전 활용 | DKIST's $\mu$-arcsec-scale maps require pixel-wise SIR runs / DKIST의 μ-arcsec 규모 지도에 픽셀별 SIR 구동 |
| de Wijn et al. 2022 (#24 SO) — DKIST ViSP | Application / 응용 — ViSP Stokes data → SIR depth-stratified magnetic fields / ViSP 스토크스 데이터에서 SIR로 깊이 층상 자기장 추출 | Modern pipelines routinely invoke SIR in batch mode / 현대 파이프라인이 SIR을 배치 모드로 호출 |
| Jaeggli et al. 2022 (#25 SO) — DKIST DL-NIRSP | Application / 응용 — multi-line chromospheric inversion with SIR-like codes (HAZEL, STiC) / SIR 유사 코드(HAZEL, STiC)로 다중선 채층 인버전 | Shows SIR's RF paradigm extending to non-LTE / SIR의 RF 패러다임이 비-LTE로 확장됨 |
| Scharmer et al. 2003 (#3 SO) — SST concept paper | Instrument / 기기 — SST/CRISP Stokes data are SIR-inverted daily / SST/CRISP 스토크스 데이터가 매일 SIR로 인버전됨 | 1-m class ground-based observatories feed SIR with high-quality spectropolarimetry / 1 m급 지상망원경이 고품질 분광편광 데이터를 SIR에 공급 |
| Rimmele et al. 2011 (#20 SO) — ATST/DKIST concept | Instrument / 기기 — original design contemplated SIR-like inversion pipeline / 원설계에 SIR-류 인버전 파이프라인이 포함 | Dictates the wavelength sampling and S/N specs at DKIST (based on §5.4's S/N=250 threshold) / §5.4의 S/N=250 기준에 맞춰 DKIST 파장 샘플링·S/N 사양 결정 |
| Anderson et al. 2020 (#18 SO) / Rochus et al. 2020 (#19 SO) — Solar Orbiter EUI/SPICE | Complementary / 보완 — space instruments without polarimetry depend on ground-based SIR inversions for magnetic context / 편광 기능 없는 우주 장비가 지상 SIR 인버전 자기장 정보에 의존 | Demonstrates SIR's role even in joint multi-instrument campaigns / 다기기 공동 관측에서도 SIR의 역할 |

---

## 7. References / 참고문헌

### Primary Source / 주 참조

- Ruiz Cobo, B. & del Toro Iniesta, J. C., "Inversion of Stokes Profiles", *The Astrophysical Journal*, **398**, 375–385 (1992). DOI: [10.1086/171862](https://doi.org/10.1086/171862)

### Prerequisite Theory / 선행 이론

- Auer, L. H., Heasley, J. N., & House, L. L. (AHH), "Interpretation of vector magnetographs", *Solar Physics*, **55**, 47 (1977).
- Beckers, J. M. & Milkey, R. W., "The line response function of stellar atmospheres", *Solar Physics*, **43**, 289 (1975).
- Landi Degl'Innocenti, E. & Landi Degl'Innocenti, M., "Response functions for magnetic lines", *Astron. Astrophys.*, **56**, 111 (1977).
- Landi Degl'Innocenti, E. & Landolfi, M., "On the Diagnostic Use of Response Functions", *Solar Physics*, **77**, 13 (1982).
- Mein, P., "Inhomogeneous atmospheres, line profiles and weighting functions", *Solar Physics*, **20**, 3 (1971).
- Rees, D. E., Murphy, G. A., & Durrant, C. J., "Stokes profile analysis and vector magnetic fields. II. Formal numerical solutions of the Stokes transfer equations", *Astrophys. J.*, **339**, 1093 (1989).
- Sánchez Almeida, J., "Response function for the inversion of Stokes profiles", *Astrophys. J.*, **391**, 349 (1992).

### Algorithms and Numerical Methods / 알고리즘 및 수치법

- Press, W. H., Flannery, B. P., Teukolsky, S. A., & Vetterling, W. T., *Numerical Recipes*, Cambridge University Press (1986). [§2.9 SVD, §14-15 Marquardt].
- Jefferies, J. T. & Mickey, D. L., "On the inference of magnetic field vectors from Stokes profiles", *Astrophys. J.*, **372**, 694 (1991).

### Comparison Methods / 비교 방법

- Landolfi, M., Landi Degl'Innocenti, E., & Arena, P., "Inversion techniques for the determination of the magnetic field and the thermodynamical structure in small active regions", *Solar Physics*, **93**, 269 (1984).
- Lites, B. W., Skumanich, A., Rees, D. E., & Murphy, G. A., "Stokes profile analysis. V. Stokes V asymmetries and sunspots", *Astrophys. J.*, **330**, 493 (1988).
- Skumanich, A. & Lites, B. W., "Stokes profile analysis and vector magnetic fields. I. Inversion of photospheric lines", *Astrophys. J.*, **322**, 473 (1987).
- Murphy, G. A., "Stokes profile analysis and vector magnetic fields. V. A Stokes V analysis code with prescribed line-of-sight field gradients", PhD thesis, Univ. Sydney (1991).

### Reference Atmospheres and Lines / 참조 대기와 선

- Gingerich, O., Noyes, R. W., Kalkofen, W., & Cuny, Y., "The Harvard-Smithsonian Reference Atmosphere", *Solar Physics*, **18**, 347 (1971).
- Maltby, P., Avrett, E. H., Carlsson, M., Kjeldseth-Moe, O., Kurucz, R. L., & Loeser, R., "A new sunspot umbral model and its variation with the solar cycle", *Astrophys. J.*, **306**, 284 (1986).
- Thévenin, F., "Chemical composition of cool stars: II. Star catalog line list 4100–9000 Å", *Astron. Astrophys. Suppl. Ser.*, **77**, 137 (1989).
- Hénoux, J. C., "Empirical model for a sunspot umbra", *Astron. Astrophys.*, **2**, 288 (1969).

### Observational Motivation / 관측 동기

- Illing, R. M. E., Landman, D. A., & Mickey, D. L., "Broad-band circular polarization of sunspots: spectral dependence and theory", *Astron. Astrophys.*, **41**, 183 (1975).
- Sánchez Almeida, J., Collados, M., & del Toro Iniesta, J. C., "Line profile asymmetries of Stokes profiles", *Astron. Astrophys.*, **222**, 311 (1989).
- Grossmann-Doerth, U., Schüssler, M., & Solanki, S. K., "Unshifted, asymmetric Stokes V-profiles: possible solution of a riddle", *Astron. Astrophys.*, **221**, 338 (1989).

### Flux-Tube Inversion (KAL) / Flux tube 인버전

- Keller, C. U., Solanki, S. K., Steiner, O., & Stenflo, J. O., "Structure of solar magnetic fluxtubes from the inversion of Stokes spectra at disk center", *Astron. Astrophys.*, **233**, 583 (1990).

### Follow-up and Reviews / 후속 및 리뷰

- del Toro Iniesta, J. C. & Ruiz Cobo, B., "Inversion of the radiative transfer equation for polarized light", *Living Reviews in Solar Physics*, **13**, 4 (2016). (Paper #13 LRSP in this reading series.)
- Socas-Navarro, H., Ruiz Cobo, B., & Trujillo Bueno, J., "Non-LTE inversion of line profiles", *Astrophys. J.*, **507**, 470 (1998).
- de la Cruz Rodríguez, J., Leenaarts, J., Danilovic, S., & Uitenbroek, H., "STiC: A multi-atom non-LTE PRD inversion code for full-Stokes solar observations", *Astron. Astrophys.*, **623**, A74 (2019).

### Data Pipelines / 데이터 파이프라인

- Wittmann, A., "Computation of Stokes profiles", *Solar Physics*, **35**, 11 (1974). (The routines modified by Ruiz Cobo for LTE absorption/opacity.)
