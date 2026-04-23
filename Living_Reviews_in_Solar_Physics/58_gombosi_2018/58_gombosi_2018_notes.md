---
title: "Extended MHD Modeling of the Steady Solar Corona and the Solar Wind"
authors: ["Tamas I. Gombosi", "Bart van der Holst", "Ward B. Manchester IV", "Igor V. Sokolov"]
year: 2018
journal: "Living Reviews in Solar Physics"
doi: "10.1007/s41116-018-0014-4"
topic: Living_Reviews_in_Solar_Physics
tags: [MHD, AWSoM, SWMF, solar_wind, corona, Alfven_waves, turbulence, PFSS, Parker_wind, CME, transition_region, space_weather]
status: completed
date_started: 2026-04-23
date_completed: 2026-04-23
---

# 58. Extended MHD Modeling of the Steady Solar Corona and the Solar Wind / 정상 태양 코로나와 태양풍의 확장 MHD 모델링

---

## 1. Core Contribution / 핵심 기여

**English**: This *Living Reviews in Solar Physics* article is a 57-page comprehensive survey that traces how the heliospheric community arrived at the current state of the art in steady-state corona and solar-wind simulation. The historical thread begins with Carrington's 1859 discovery and runs through Biermann's comet-tail inference, Parker's 1958 hydrodynamic wind, Chamberlain's failed "solar breeze" alternative, Mariner-2 confirmation in 1962, the first Navier–Stokes corona (Scarf–Noble, 1965), the first two-fluid model (Sturrock–Hartle, 1966), PFSS (Altschuler–Newkirk, 1969), the first MHD helmet streamer (Pneuman–Kopp, 1971), the first 3-D MHD heliosphere codes of the 1990s, and culminates in the modern **Alfvén Wave Solar Model (AWSoM)** that self-consistently heats and accelerates the solar wind via dissipation of counter-propagating Alfvén-wave turbulence with no ad-hoc polytropic index and no unconstrained heating function. AWSoM incorporates (i) two-temperature plasma ($T_p,T_e$), (ii) anisotropic proton pressure ($T_{\parallel p},T_{\perp p}$), (iii) radiative losses $\Lambda(T)$ from CHIANTI, (iv) Spitzer–Härm parallel heat conduction, and (v) Alfvén wave energy densities $w_\pm$ coupled through reflection and a turbulence dissipation rate $\Gamma_\pm$. Results reproduce fast (700–800 km/s) and slow (300–450 km/s) solar wind bimodality, match Mariner-2 era in-situ speeds, and yield synthetic SDO/AIA, STEREO/EUVI EUV images consistent with observations (CR2107, 7 March 2011). The article closes with the latest **Threaded Field Line Model (TFLM, Sokolov et al. 2016)** that efficiently bridges the 10-Mm-thick transition region via 1-D threads rather than sub-kilometer grid refinement.

**Korean / 한국어**: 본 *Living Reviews in Solar Physics* 논문은 57 쪽 분량으로, 태양권 학계가 정상 상태 코로나 및 태양풍 시뮬레이션의 최신 수준에 도달하기까지의 경로를 포괄적으로 정리한 총설이다. 역사적 흐름은 1859년 Carrington 발견에서 시작하여, Biermann의 혜성 꼬리 추론, Parker(1958)의 유체역학적 태양풍, Chamberlain의 실패한 "solar breeze" 대안, Mariner-2의 1962년 확인, 최초의 Navier–Stokes 코로나(Scarf–Noble 1965), 최초의 2-유체 모델(Sturrock–Hartle 1966), PFSS(Altschuler–Newkirk 1969), 최초의 MHD helmet streamer(Pneuman–Kopp 1971), 1990년대 최초의 3차원 MHD 해류권 코드를 거쳐, 역방향 전파 알펜파 난류 소산으로 태양풍을 자기정합적으로 가열·가속하며 임시 폴리트로픽 지수와 제약 없는 가열 함수를 제거한 현대의 **Alfvén Wave Solar Model(AWSoM)** 에서 정점을 이룬다. AWSoM은 (i) 2-온도 플라즈마($T_p,T_e$), (ii) 이방성 양성자 압력($T_{\parallel p},T_{\perp p}$), (iii) CHIANTI의 복사 냉각 $\Lambda(T)$, (iv) Spitzer–Härm 자기장 평행 열전도, (v) 반사와 난류 소산율 $\Gamma_\pm$로 결합된 알펜파 에너지 밀도 $w_\pm$를 통합한다. 그 결과 고속(700–800 km/s)·저속(300–450 km/s) 바이모달 태양풍을 재현하고, Mariner-2 시기의 in-situ 속도와 일치하며, SDO/AIA·STEREO/EUVI 합성 EUV 이미지(CR2107, 2011년 3월 7일)를 관측과 일관되게 생성한다. 논문은 10 Mm 두께의 천이영역을 서브킬로미터 격자 세분화 대신 1차원 "thread" 집단으로 효율적으로 연결하는 최신 **Threaded Field Line Model(TFLM, Sokolov 외 2016)** 으로 마무리된다.

---

## 2. Reading Notes / 읽기 노트

### Part I: Early ideas (§§ 1–2) / 초기 아이디어 (pp. 2–12)

**English**: The story starts with Aristotle's quinta essentia, moves to Carrington's 1859 flare + geomagnetic storm observations, FitzGerald's 1892 "something like a comet's tail projected from the Sun" hypothesis, Chapman & Ferraro's 1931 cold plasma beam, Biermann (1951) inferring a continuous outflow with $n_{\rm sw}\approx 1000$ cm$^{-3}$ and $u_{\rm sw}\approx 1000$ km/s from Type I comet-tail accelerations of 30–300 cm/s², and Parker's 1958 seminal paper. Chapman preferred a static conducting corona heated to ≥10⁶ K but with zero bulk flow. Parker reconciled Biermann and Chapman with a **continuous hydrodynamic expansion**. He arrived at the cubic transcendental relation

$$\left[\frac{v^2}{v_m^2}-\ln\!\left(\frac{v^2}{v_m^2}\right)\right]=4\ln\!\left(\frac{r}{a}\right)+\left(\frac{v_{\rm esc}^2}{v_m^2}\right)\!\left(\frac{a}{r}\right)-4\left(\frac{v_{\rm esc}^2}{v_m^2}\right)-3+\ln 256$$

with $v_m^2=2kT_0/m_p$. For $T_0\approx 1$–$4\times 10^6$ K, supersonic solutions exist with terminal speeds around 400–700 km/s (Parker's original figure at p. 7). Parker also introduced the **Archimedean (Parker) spiral** for the carried field:

$$\frac{r}{R_s}-\ln\!\left(\frac{r}{R_s}\right)=1+\frac{v_s}{R_s\Omega_\odot}(\phi-\phi_s)$$

with $B_r\propto 1/r^2$ and $B_\phi\propto 1/r$ at large $r$. Opposition was strong (Chandrasekhar's reluctance, two hostile referees), but Mariner-2 (1962) confirmed $n\approx 5$–$20$ cm⁻³ and $v\approx 300$–$700$ km/s (Neugebauer & Snyder 1966), vindicating Parker.

**Korean / 한국어**: 이야기는 아리스토텔레스의 제5원소에서 시작하여, 1859년 Carrington의 플레어·지자기 폭풍 관측, FitzGerald(1892)의 "태양에서 혜성 꼬리 같은 것이 뻗어 나온다"는 가설, Chapman–Ferraro(1931)의 차가운 플라즈마 빔, Biermann(1951)이 Type I 혜성 꼬리의 30–300 cm/s² 가속도로부터 $n_{\rm sw}\approx 1000$ cm$^{-3}$, $u_{\rm sw}\approx 1000$ km/s 의 연속 유출을 추론한 것, 그리고 Parker(1958)의 기념비적 논문으로 이어진다. Chapman은 10⁶ K 이상으로 가열되었지만 bulk 흐름이 없는 정적 코로나를 선호하였다. Parker는 이를 **연속적 유체역학적 팽창**으로 통합하여 위의 3차 초월 관계식을 유도하였다. $T_0\approx 1$–$4\times 10^6$ K 에서 초음속 해가 존재하며 종단 속도는 400–700 km/s 이다(p. 7 원본 그림). Parker는 또한 운반 자기장에 대한 **Archimedean(Parker) 나선**을 도입했으며, 큰 $r$ 에서 $B_r\propto 1/r^2$, $B_\phi\propto 1/r$ 이다. 반대는 거셌지만(Chandrasekhar의 주저, 두 명의 적대적 referee), Mariner-2(1962)가 $n\approx 5$–$20$ cm⁻³, $v\approx 300$–$700$ km/s 를 확인하여 Parker가 옳았음을 입증하였다(Neugebauer & Snyder 1966).

### Part II: First numerical models (§ 3) / 최초 수치 모델 (pp. 12–19)

**English**: Scarf & Noble (1963, 1965) solved the spherically symmetric Navier–Stokes equations *including* heat conduction and viscosity, integrating inward from 1 AU ($n=3.4$ cm⁻³, $v=352$ km/s, $T=2.77\times 10^5$ K). They matched observed electron density $N_e(r)$ to within a factor of 2–3. Sturrock & Hartle (1966) published the first **two-fluid** solution, using the energy equation

$$\frac{3}{2}\frac{1}{T_s}\frac{dT_s}{dr}-\frac{1}{n}\frac{dn}{dr}=\frac{1}{\Phi k T_s}\frac{d}{dr}\!\left(r^2\kappa_s\frac{dT_s}{dr}\right)+\frac{3}{2}\frac{\nu_{ei}}{v}\frac{T_t-T_s}{T_s},\qquad (s=e,i;\ t=i,e)$$

with Spitzer conductivities $\kappa_e=6\times 10^{-7}T_e^{5/2}$ s⁻¹ and $\kappa_i=1.4\times 10^{-8}T_i^{5/2}$ s⁻¹, Chapman collision frequency $\nu_{ei}=8.5\times 10^{-2}nT_e^{-3/2}$, and flux factor $\Phi=nvr^2$. The resulting $v(r=1\,{\rm AU})=270$ km/s was too slow, $n=13$ cm⁻³ too high, but the 2-T framework was established. The **PFSS model** (Schatten et al. 1969; Altschuler & Newkirk 1969) expanded $\psi(r,\theta,\phi)$ in spherical harmonics with source-surface radius $R_s=2.5 R_\odot$:

$$\psi_s(r,\theta,\phi)=R_\odot\sum_{n=1}^{\infty}f_n(r)\sum_{m=0}^{n}\bigl[g_n^m\cos m\phi + h_n^m\sin m\phi\bigr]P_n^m(\theta)$$

with the crucial function

$$f_n(r)=\frac{(R_s/R_\odot)^{2n+1}\,(R_\odot/r)^{n+1}-1}{(R_s/R_\odot)^{2n+1}-1}\,\left(\frac{r}{R_\odot}\right)^n$$

ensuring $f_n(R_s)=0$ so $\mathbf{B}$ becomes radial at the source surface. Pneuman & Kopp (1971) combined PFSS-like topology with a hot, expanding Parker-like corona and obtained the first **helmet streamer** solution: a closed-field equatorial region where plasma is gravitationally confined, and a current sheet above it where the flow transitions from sub-Alfvénic to super-Alfvénic. This is the prototype of every modern global MHD corona.

**Korean / 한국어**: Scarf & Noble(1963, 1965)은 열전도와 점성을 포함한 구대칭 Navier–Stokes 방정식을 1 AU($n=3.4$ cm⁻³, $v=352$ km/s, $T=2.77\times 10^5$ K) 에서 안쪽으로 적분하여 풀었으며, 관측된 전자 밀도 $N_e(r)$ 와 2–3배 이내로 일치하였다. Sturrock & Hartle(1966)은 위의 에너지 방정식을 사용한 최초의 **2-유체** 해를 발표하였다. $v(r=1\,{\rm AU})=270$ km/s 는 너무 느리고 $n=13$ cm⁻³ 는 너무 높았지만 2-T 프레임워크는 확립되었다. **PFSS 모델**(Schatten 외 1969; Altschuler–Newkirk 1969)은 $\psi$ 를 $R_s=2.5 R_\odot$ 를 외부 경계로 하는 구면 조화 함수로 전개하였다. 위 $f_n(r)$ 함수는 $f_n(R_s)=0$ 이 되어 source surface 에서 자기장이 방사형이 되도록 한다. Pneuman–Kopp(1971)는 PFSS-형 위상과 Parker 유형 팽창 코로나를 결합하여 최초의 **helmet streamer** 해를 얻었다: 적도의 닫힌 자기장 영역은 중력 구속, 그 위 current sheet에서 sub-Alfvénic에서 super-Alfvénic으로 전이한다. 이것이 모든 현대 글로벌 MHD 코로나의 원형이다.

### Part III: Steady-state solar wind (§ 4) / 정상 태양풍 (pp. 19–30)

**English**: The "ambient + transient" paradigm is adopted: a quasi-steady Carrington-rotation-averaged corona/heliosphere provides the background for impulsive CMEs. The **bimodal wind** structure is: (a) open field coronal-hole flows produce fast (≈750 km/s) wind, (b) closed-field/streamer-belt regions produce slow (300–450 km/s) wind near the heliospheric current sheet. 2-D and quasi-3D models (Steinolfson 1978; Pizzo 1978, 1980) pioneered axial and meridional symmetry; full 3-D MHD heliosphere models appeared in the 1990s (Usmanov, Linker, Mikić). **3-D MHD coronal models** accept PFSS or MDI/HMI synoptic magnetograms as inputs and evolve the MHD equations with a polytropic equation of state $p\propto \rho^\gamma$ with reduced $\gamma\approx 1.05$–$1.1$ (mimicking heat deposition). Thermodynamic models (Lionello, Downs, van der Holst) go further by adding explicit heat conduction and radiative losses, resolving the transition region, and extending down to the upper chromosphere. The empirical **Wang–Sheeley–Arge (WSA)** relation $v(f_s)\propto f_s^{-\alpha}$ (with $f_s$ the flux expansion factor) is used as a boundary condition in many codes.

**Korean / 한국어**: "ambient + transient" 패러다임을 채택한다: 준정상 Carrington 회전 평균 코로나/해류권이 순간적 CME의 배경을 제공한다. **바이모달 태양풍** 구조: (a) 열린 자기장 coronal-hole 흐름은 고속풍(≈750 km/s)을 생성, (b) 닫힌 자기장·streamer belt 영역은 해류권 current sheet 근처에서 저속풍(300–450 km/s)을 생성한다. 2-D 및 준 3-D 모델(Steinolfson 1978; Pizzo 1978, 1980)이 축·자오면 대칭을 개척했고, 완전한 3-D MHD 해류권 모델은 1990년대에 등장했다(Usmanov, Linker, Mikić). **3-D MHD 코로나 모델**은 PFSS 또는 MDI/HMI 동기 자기사진을 입력으로 받아 폴리트로픽 상태방정식 $p\propto \rho^\gamma$ ($\gamma\approx 1.05$–$1.1$, 열 증착을 모사)으로 MHD 방정식을 진화시킨다. 열역학 모델(Lionello, Downs, van der Holst)은 명시적 열전도와 복사 냉각을 추가하고, 천이영역을 해상도로 잡으며, 상부 색구까지 확장한다. 실증적 **Wang–Sheeley–Arge(WSA)** 관계 $v(f_s)\propto f_s^{-\alpha}$ ($f_s$: 자속 팽창 인자)는 많은 코드에서 경계조건으로 사용된다.

### Part IV: Alfvén wave turbulence in MHD models (§ 5) / MHD 모델 내 알펜파 난류 (pp. 31–39)

**English**: The physical insight — first by Coleman (1968), Jacques (1977, 1978) — is that the **gradient of Alfvén wave pressure** accelerates the fast wind, and Alfvén wave **dissipation** via counter-propagating wave interaction heats the corona. AWSoM's governing system (Sokolov et al. 2013; van der Holst et al. 2014) evolves:

$$\frac{\partial \rho}{\partial t}+\nabla\cdot(\rho \mathbf{u})=0,\qquad \frac{\partial \mathbf{B}}{\partial t}+\nabla\cdot(\mathbf{uB}-\mathbf{Bu})=0$$

$$\frac{\partial(\rho\mathbf{u})}{\partial t}+\nabla\cdot\!\left(\rho\mathbf{uu}-\frac{\mathbf{BB}}{\mu_0}\right)+\nabla\!\left(p_i+p_e+\frac{B^2}{2\mu_0}+p_A\right)=-\frac{GM_\odot\rho\,\mathbf{r}}{r^3}$$

$$\frac{\partial}{\partial t}\!\left(\frac{p_e}{\gamma-1}\right)+\nabla\cdot\!\left(\frac{p_e}{\gamma-1}\mathbf{u}\right)+p_e\nabla\cdot\mathbf{u}=-\nabla\cdot\mathbf{q}_e+\frac{N_eN_ik_B\nu_{ei}}{(\gamma-1)N_i}(T_i-T_e)-Q_{\rm rad}+Q_e$$

The Alfvén wave pressure is $p_A=(w_++w_-)/2$, and the ion and electron heat equations exchange energy by the Coulomb collision frequency $\nu_{ei}/N_i=(2\sqrt{m_e}\Lambda_C (e^2/\epsilon_0)^2)/(3 m_p (2\pi k_B T_e)^{3/2})$. Radiative cooling uses CHIANTI: $Q_{\rm rad}=N_e N_i \Lambda(T_e)$. Heat flux is Spitzer–Härm:

$$\mathbf{q}_e=\kappa_\parallel\,\hat{\mathbf{b}}\,(\hat{\mathbf{b}}\cdot\nabla T_e),\qquad \kappa_\parallel\propto T_e^{5/2}$$

The Alfvén wave transport equation is (van der Holst et al. 2014):

$$\boxed{\frac{\partial w_\pm}{\partial t}+\nabla\cdot\bigl[(\mathbf{u}\pm\mathbf{V}_A)w_\pm\bigr]+\frac{w_\pm}{2}(\nabla\cdot\mathbf{u})=-\Gamma_\pm w_\pm\mp\mathcal{R}\sqrt{w_-w_+}}$$

with $\mathbf{V}_A=\mathbf{B}/\sqrt{\mu_0\rho}$, dissipation rate

$$\Gamma_\pm=\frac{2}{L_\perp}\sqrt{\frac{w_\mp}{\rho}},\qquad L_\perp\propto B^{-1/2}$$

and reflection coefficient

$$\mathcal{R}=\min\!\left\{\sqrt{(\hat{\mathbf{b}}\cdot[\nabla\times \mathbf{u}])^2+[(\mathbf{V}_A\cdot\nabla)\log V_A]^2},\ \max(\Gamma_\pm)\right\}\cdot[\text{imbalance factors}]$$

Wave heating is partitioned $Q_i=f_p(\Gamma_-w_-+\Gamma_+w_+)$ and $Q_e=(1-f_p)(\Gamma_-w_-+\Gamma_+w_+)$, with $f_p\approx 0.6$. The Poynting flux boundary condition at the photosphere is

$$\frac{\Pi_A}{B}=\text{const}\approx 1.1\times 10^6\ \text{W m}^{-2}\,\text{T}^{-1}$$

and the perpendicular correlation length scales as $L_\perp\sqrt{B}\in[100,300]$ km·T$^{1/2}$.

**Korean / 한국어**: 물리적 통찰—Coleman(1968), Jacques(1977, 1978)가 최초—은 **알펜파 압력 gradient** 가 고속풍을 가속하고, 역방향 전파파 상호작용을 통한 알펜파 **소산**이 코로나를 가열한다는 것이다. AWSoM의 지배 방정식 계(Sokolov 외 2013; van der Holst 외 2014)는 위와 같다: 연속·유도·운동량 방정식은 알펜파 압력 $p_A=(w_++w_-)/2$ 를 운동량 방정식에 포함한다. 전자 에너지 방정식은 Coulomb 충돌($\nu_{ei}/N_i$), 복사 냉각 $Q_{\rm rad}=N_e N_i\Lambda(T_e)$ (CHIANTI), 가열 분배 $Q_e$ 를 포함한다. 열 유속은 Spitzer–Härm $\mathbf{q}_e\propto T_e^{5/2}$ 형태이다. 알펜파 전달 방정식은 위 상자 식이며, 소산율 $\Gamma_\pm=(2/L_\perp)\sqrt{w_\mp/\rho}$, 상관 길이 $L_\perp\propto B^{-1/2}$, 그리고 반사 계수 $\mathcal{R}$ 는 알펜 속도 gradient와 vorticity로 제어된다. 파동 가열은 $Q_i=f_p(\cdot)$, $Q_e=(1-f_p)(\cdot)$ 으로 분배되며 $f_p\approx 0.6$ 이다. 광구 Poynting flux 경계조건 $\Pi_A/B\approx 1.1\times 10^6$ W m⁻² T⁻¹, 상관 길이 $L_\perp\sqrt{B}\in[100,300]$ km·T$^{1/2}$.

### Part V: Transition region (§§ 5.4, 7) / 천이영역 (pp. 35–47)

**English**: The TR spans $T_e\approx 2\times 10^4$ K (top of chromosphere) to $T_e\approx 4.5\times 10^5$ K over only ≈10 Mm, i.e., ≈$R_\odot/70$. This creates a temperature gradient ≈10⁴ K/km, requiring sub-kilometer resolution in a 3-D simulation — prohibitive. The 1-D **analytical TR solution** (Lionello et al. 2001) integrates

$$\frac{\partial}{\partial s}\!\left(\kappa_0 T_e^{5/2}\frac{\partial T_e}{\partial s}\right)+Q_h-N_e^2\Lambda(T_e)=0$$

and yields the jump condition

$$\bigl[\tfrac{1}{2}\kappa_0^2 T_e^5(\partial_s T_e)^2+\tfrac{2}{7}\kappa_0 Q_h T_e^{7/2}\bigr]_{T_{ch}}^{T_e}=(N_e T_e)^2\int_{T_{ch}}^{T_e}\kappa_0 T^{1/2}\Lambda(T)\,dT$$

Chromospheric boundary conditions: $T_{ch}=(2\text{–}5)\times 10^4$ K, $N_{ch}\approx 2\times 10^{16}$ m⁻³. The pressure scale length is $L=k_B T_{ch}/(m_i g)\approx T_{ch}\times(30\text{ m/K})$ with $g=274$ m/s². To avoid ultrafine TR meshing, Abbett (2007) and subsequent work use an **artificial broadening** of the TR by applying a factor $f\ge 1$:

$$\kappa_0\to f\kappa_0,\quad ds\to f\,ds,\quad \Gamma\to \Gamma/f,\quad Q_{\rm rad}\to Q_{\rm rad}/f$$

with $f=(T_m/T_e)^{5/2}$ at $T_{ch}\le T_e\le T_m$, choosing $T_m\approx 2.5\times 10^5$ K. The **Threaded Field Line Model (TFLM, Sokolov et al. 2016)** replaces the 3-D TR with 1-D "threads" along potential-field lines. Along a thread, $B\cdot A=$const, the 1-D continuity gives $\rho u/B=$const, momentum gives

$$\frac{\partial p}{\partial s}=-\frac{b_r GM_\odot\rho}{r^2}\Rightarrow p=p_{\rm TR}\exp\!\left[\int_{R_{\rm TR}}^r \frac{GM_\odot m_p}{2k_B T(r')}d\!\left(\frac{1}{r'}\right)\right]$$

and energy

$$\frac{2 N_i k_B}{B(\gamma-1)}\frac{\partial T}{\partial t}+\frac{2 k_B\gamma}{\gamma-1}\!\left(\frac{N_iu}{B}\right)\!\frac{\partial T}{\partial s}=\frac{\partial}{\partial s}\!\left(\frac{\kappa_\parallel}{B}\frac{\partial T}{\partial s}\right)+\frac{\Gamma_-w_-+\Gamma_+w_+-N_eN_i\Lambda(T)}{B}+\cdots$$

This makes the expensive TR tractable.

**Korean / 한국어**: 천이영역(TR)은 $T_e\approx 2\times 10^4$ K(색구 상부)에서 $T_e\approx 4.5\times 10^5$ K 까지를 ≈10 Mm 즉 ≈$R_\odot/70$ 안에 걸친다. 이로 인해 온도 gradient 가 ≈10⁴ K/km 에 달하여 3차원 시뮬레이션에서는 서브킬로미터 해상도가 필요 — 실행 불가능. 1-D **해석적 TR 해**(Lionello 외 2001)는 위 적분식을 풀어 점프 조건을 얻는다. 색구 경계: $T_{ch}=(2\text{–}5)\times 10^4$ K, $N_{ch}\approx 2\times 10^{16}$ m⁻³, 압력 scale length $L\approx T_{ch}\times 30$ m/K, $g=274$ m/s². 극미세 격자를 피하기 위해 Abbett(2007) 및 후속 연구는 인위적 factor $f\ge 1$ 을 적용하여 TR 을 인공적으로 넓힌다. **TFLM**(Sokolov 외 2016)은 3차원 TR 을 potential field 선을 따른 1-D thread 로 대체한다. Thread 를 따라 $B\cdot A=$const, 운동량 방정식은 위 지수적 압력 profile을 주며, 에너지 방정식은 1-D 형태로 간소화된다.

### Part VI: AWSoM and multi-temperature models (§ 6) / AWSoM 과 다중 온도 모델 (pp. 39–44)

**English**: AWSoM employs two Alfvén wave populations $w_\pm$ propagating parallel/antiparallel to $\mathbf{B}$ with a wave spectrum that is not resolved (only total $w_\pm$ and $L_\perp$). Partitioning between protons and electrons follows Chandran et al. (2011)'s **stochastic heating** formulation for perpendicular proton heating, and kinetic instability thresholds (firehose, mirror, cyclotron) limit anisotropy. In the 3-T version (van der Holst et al. 2014; Meng et al. 2015), $T_{\parallel p}$, $T_{\perp p}$, and $T_e$ are evolved separately. Figure 22 of the paper shows:
- Coronal holes (low $\beta$): most heat goes to $T_{\perp p}$ (perpendicular proton heating dominates)
- Streamer current sheet (high $\beta$): parallel proton heating dominates
- Intermediate $\beta$ (active-region margins): electron heating dominates

Fast CMEs (≥1000 km/s) can shock-heat protons while electrons cool adiabatically; beyond ≈2 $R_\odot$ Coulomb coupling is lost on the shock timescale. This three-temperature treatment matters for interpretation of in-situ ACE/WIND/Ulysses/Parker Solar Probe data.

**Korean / 한국어**: AWSoM은 자기장에 평행/반평행으로 전파하는 두 알펜파 집단 $w_\pm$ 를 사용하며, 파동 스펙트럼은 해상도로 잡지 않고 $w_\pm$ 총량과 $L_\perp$ 만 다룬다. 양성자와 전자 분배는 Chandran 외(2011)의 **stochastic heating** 공식을 따르며, 동역학적 불안정성 문턱(firehose, mirror, cyclotron)이 이방성을 제한한다. 3-T 버전(van der Holst 외 2014; Meng 외 2015)은 $T_{\parallel p}$, $T_{\perp p}$, $T_e$ 를 분리 진화시킨다. 논문 Fig. 22: 저-$\beta$ 영역(coronal hole)에서는 $T_{\perp p}$ 가열 우세, 고-$\beta$ 영역(streamer current sheet)에서는 평행 양성자 가열 우세, 중간 $\beta$(활성 영역 주변)에서는 전자 가열 우세. 고속 CME(≥1000 km/s)는 양성자를 충격 가열하되 전자는 단열 냉각될 수 있고, ≈2 $R_\odot$ 너머 Coulomb 결합이 충격 시간척도에서 상실된다. 3-온도 처리는 ACE/WIND/Ulysses/Parker Solar Probe in-situ 데이터 해석에 중요하다.

### Part VII: Model validation and mesh techniques (§§ 4.7–4.8) / 모델 검증과 격자 기법 (pp. 29–31)

**English**: Validation draws on (a) in-situ wind speed/density at 1 AU (ACE, WIND, OMNI), (b) remote-sensing EUV images (SDO/AIA 171/193/211 Å, STEREO/EUVI), (c) white-light pB images from LASCO, and (d) radio scintillation densities at outer corona. Jin et al. (2017) demonstrated AWSoM's ability to reproduce AIA 211 Å, EUVI 171 Å, and EUVI 195 Å for CR2107 (7 March 2011) with active regions AR1–AR7 correctly placed. AMR refinement is block-structured (BATSRUS) with levels typically 6–8 in the corona. Time stepping is explicit for outer blocks and **local time stepping** accelerates steady-state convergence (factor of ≈10–100).

**Korean / 한국어**: 검증은 (a) 1 AU 의 in-situ 태양풍 속도·밀도(ACE, WIND, OMNI), (b) 원격탐사 EUV 이미지(SDO/AIA 171/193/211 Å, STEREO/EUVI), (c) LASCO 백색광 pB 이미지, (d) 외부 코로나 전파 섬광 밀도에 의존한다. Jin 외(2017)는 AWSoM 이 CR2107(2011년 3월 7일)의 AIA 211 Å, EUVI 171 Å, EUVI 195 Å 를 활성 영역 AR1–AR7 을 올바르게 배치하여 재현함을 보였다. AMR 세분화는 블록 구조(BATSRUS)이며, 코로나에서 보통 6–8 레벨이다. 시간 적분은 외부 블록에서 explicit 이고, **local time stepping** 이 정상상태 수렴을 약 10–100배 가속한다.

### Part VIII: Model inputs and magnetogram sources (§ 4.6) / 모델 입력과 자기사진 원천 (pp. 27–29)

**English**: The synoptic magnetogram is the single most important external input. Common sources:
- **GONG** (Global Oscillation Network Group) — 1° lat/lon, zero-point referenced
- **SOLIS/VSM** — vector magnetogram, higher noise
- **MDI** (SOHO, 1996–2011) — historical baseline
- **HMI** (SDO, 2010–present) — 0.5″ resolution, polarization
- **ADAPT** (Air Force Data Assimilative Photospheric Flux Transport) — time-evolved far-side proxy

Far-side fields are unobservable directly; flux transport models (e.g., Schrijver & De Rosa) evolve them from prior near-side observations. Polar fields, which dominate coronal-hole area, are poorly observed due to line-of-sight projection and are typically corrected using a high-degree polynomial fit. These choices can shift predicted 1-AU solar wind speeds by ±100 km/s — a first-order systematic uncertainty that limits operational forecasting.

**Korean / 한국어**: 동기 자기사진은 단연 가장 중요한 외부 입력이다. 주요 원천은 다음과 같다:
- **GONG** — 위도/경도 1° 해상도, 영점 보정
- **SOLIS/VSM** — 벡터 자기사진, 잡음 높음
- **MDI** (SOHO, 1996–2011) — 역사적 기준선
- **HMI** (SDO, 2010–현재) — 0.5″ 해상도, 편광
- **ADAPT** — 공군 자료 동화 광구 자속 수송 모델, 시간 진화 원면 추정

원면 자기장은 직접 관측 불가; 자속 수송 모델(예: Schrijver & De Rosa)이 근면 관측에서 진화시킨다. Coronal-hole 면적을 지배하는 극지방 자기장은 시선 투영 때문에 관측이 빈약하여 고차 다항식 fit 로 보정한다. 이러한 선택이 예측된 1-AU 태양풍 속도를 ±100 km/s 이동시킬 수 있으며 — 운영 예보를 제한하는 1차 계통 불확실성이다.

### Part IX: Numerical mesh — AMR and BATSRUS (§ 4.8) / 수치 격자 — AMR 과 BATSRUS

**English**: The SWMF corona runs on the **BATSRUS** (Block-Adaptive-Tree Solar-wind Roe-type Upwind Scheme) code. The mesh consists of logically Cartesian blocks (8³ or 4³ cells) organized in an octree. AMR levels 0–8 typically give resolution ≈ $R_\odot/100$ ≈ 7000 km at the surface. Refinement criteria track:
- Current-sheet proximity ($|\mathbf{j}|/|\mathbf{B}|$ large)
- Alfvén-speed gradient (TR and streamer tops)
- Density gradient

Block-level parallelization enables scaling to 10⁵ cores. Implicit/semi-implicit schemes (Tóth et al. 2012) handle the stiff TR; explicit Riemann solvers (HLLC, Roe) handle the bulk wind. Divergence cleaning uses the 8-wave approach (Powell et al. 1999) or projection. The resulting 3-D steady solution typically takes 10⁴–10⁵ CPU-hours.

**Korean / 한국어**: SWMF 코로나는 **BATSRUS**(Block-Adaptive-Tree Solar-wind Roe-type Upwind Scheme) 코드로 실행된다. 격자는 octree 로 구성된 논리적 Cartesian 블록(8³ 또는 4³ 셀)이다. AMR 레벨 0–8 은 일반적으로 표면에서 ≈ $R_\odot/100$ ≈ 7000 km 해상도를 제공한다. 세분화 기준:
- Current sheet 근접($|\mathbf{j}|/|\mathbf{B}|$ 대)
- 알펜 속도 gradient (TR 및 streamer top)
- 밀도 gradient

블록 수준 병렬화로 10⁵ 코어까지 스케일링 가능. Implicit/semi-implicit 기법(Tóth 외 2012)으로 경직된 TR 을 처리하고, explicit Riemann solver(HLLC, Roe)가 bulk 태양풍을 처리한다. Divergence cleaning 은 8-wave 접근(Powell 외 1999) 또는 projection 을 사용한다. 3차원 정상 해는 일반적으로 10⁴–10⁵ CPU-시간이 소요된다.

### Part X: CME modeling and transient coupling / CME 모델링 및 과도 결합

**English**: Although this review focuses on steady-state corona, the steady solution is the launching platform for CME simulations. Standard procedure:
1. Relax the AWSoM corona to steady state (reach ~1–2% residual in outer wind)
2. Insert a flux rope — choices include **Titov–Démoulin** (1999), **Gibson–Low** (1998), or **spheromak** (Chané et al. 2008)
3. Re-evolve the MHD equations — the rope loses equilibrium and erupts
4. Track the CME through the inner heliosphere (SWMF-IH)
5. Compare arrival time, speed profile, shock structure at L1 with ACE

Manchester et al. (2004, 2008, 2014) and Jin et al. (2017) demonstrated arrival-time accuracy within ±6 h for several historical events. CME front speeds in simulations range 500–3000 km/s, consistent with the observed range.

**Korean / 한국어**: 본 총설이 정상 상태 코로나에 집중하지만, 정상 해는 CME 시뮬레이션의 출발 플랫폼이다. 표준 절차:
1. AWSoM 코로나를 정상 상태로 이완(외부 태양풍에서 ~1–2% 잔차 도달)
2. 자속관 삽입 — **Titov–Démoulin**(1999), **Gibson–Low**(1998), **spheromak**(Chané 외 2008) 중 선택
3. MHD 방정식 재진화 — 자속관이 평형을 잃고 폭발
4. 내부 해류권(SWMF-IH)에서 CME 추적
5. L1 에서 도달 시각·속도 profile·충격 구조를 ACE 와 비교

Manchester 외(2004, 2008, 2014)와 Jin 외(2017)는 여러 역사적 사건에서 ±6시간 이내의 도달 시각 정확도를 입증했다. 시뮬레이션 CME 전면 속도는 500–3000 km/s 범위로, 관측된 범위와 일치한다.

### Part XI: Worked quantitative example — slow wind at 1 AU / 정량 예제 — 1 AU 에서의 저속풍

**English**: Consider a slow-wind streamline originating from a streamer-belt footpoint at heliographic latitude 10°N. AWSoM parameters typical of the 2011-03-07 run:
- Base photospheric radial $B_r=5$ G, $T_{ch}=2.5\times 10^4$ K, $N_{ch}=2\times 10^{16}$ m⁻³
- Poynting flux input $\Pi_A\approx 5\times 10^5$ W/m² (for $B=0.05$ T)
- Correlation length $L_\perp \sqrt{B}=150$ km·T$^{1/2}$ → $L_\perp\approx 671$ km at the base
- Resulting Alfvén speed at 2 R⊙: $V_A\approx 800$ km/s
- Wave energy density at 2 R⊙: $w_+\approx 10^{-5}$ J/m³, giving $p_A\approx 5\times 10^{-6}$ Pa
- Temperature at 2 R⊙: $T_e\approx 1.4\times 10^6$ K, $T_p\approx 1.6\times 10^6$ K
- At 1 AU: $v_{\rm sw}\approx 380$ km/s, $n\approx 8$ cm⁻³, $T_p\approx 8\times 10^4$ K, $T_e\approx 1.2\times 10^5$ K

For contrast, a fast-wind streamline from a polar coronal hole at 60°N:
- $V_A$(2 R⊙)≈1500 km/s, $w_+\approx 3\times 10^{-5}$ J/m³
- $v_{\rm sw}$(1 AU)≈700 km/s, $n\approx 3$ cm⁻³, $T_p\approx 2.5\times 10^5$ K

These numbers match Ulysses polar passes and ACE near-ecliptic observations to within 20–30%.

**Korean / 한국어**: 위도 10°N streamer belt 족발에서 출발하는 저속풍 유선을 고려. 2011-03-07 실행의 전형적 AWSoM 매개변수:
- 광구 기저 $B_r=5$ G, $T_{ch}=2.5\times 10^4$ K, $N_{ch}=2\times 10^{16}$ m⁻³
- Poynting flux 입력 $\Pi_A\approx 5\times 10^5$ W/m² ($B=0.05$ T 기준)
- 상관 길이 $L_\perp\sqrt{B}=150$ km·T$^{1/2}$ → 기저에서 $L_\perp\approx 671$ km
- 2 R⊙ 에서 알펜 속도: $V_A\approx 800$ km/s
- 2 R⊙ 에서 파동 에너지 밀도: $w_+\approx 10^{-5}$ J/m³, $p_A\approx 5\times 10^{-6}$ Pa
- 2 R⊙ 에서 온도: $T_e\approx 1.4\times 10^6$ K, $T_p\approx 1.6\times 10^6$ K
- 1 AU 에서: $v_{\rm sw}\approx 380$ km/s, $n\approx 8$ cm⁻³, $T_p\approx 8\times 10^4$ K, $T_e\approx 1.2\times 10^5$ K

대조적으로, 위도 60°N 극지 coronal hole 에서의 고속풍 유선:
- $V_A$(2 R⊙)≈1500 km/s, $w_+\approx 3\times 10^{-5}$ J/m³
- $v_{\rm sw}$(1 AU)≈700 km/s, $n\approx 3$ cm⁻³, $T_p\approx 2.5\times 10^5$ K

이 수치들은 Ulysses 극지 passage 및 ACE ecliptic 관측과 20–30% 이내로 일치한다.

---

## 3. Key Takeaways / 핵심 시사점

1. **Parker's wind is still right, but incomplete** — Parker's isothermal hydrodynamic solution (Eq. 1 of the review) captures the qualitative essence of the solar wind; but the bimodal fast/slow wind and the high-temperature corona require **Alfvén-wave turbulence heating**, which Parker could not have anticipated with 1950s tools. / **Parker의 태양풍은 여전히 옳지만 불완전하다** — Parker 등온 유체역학 해는 태양풍의 정성적 본질을 포착하지만, 바이모달 고속/저속풍과 고온 코로나는 1950년대 도구로는 예측할 수 없었던 **알펜파 난류 가열**을 요구한다.

2. **PFSS + polytropic MHD was the 2nd-generation workhorse** — For 30 years (1970s–2000s), PFSS-driven 3-D MHD models with $\gamma\approx 1.05$ were the community standard. They reproduce topology well but bury the heating physics in a reduced polytropic index. / **PFSS + 폴리트로픽 MHD는 2세대의 주력** — 30년간(1970–2000) $\gamma\approx 1.05$ 의 PFSS 구동 3차원 MHD 모델이 표준이었다. 위상은 잘 재현하지만 가열 물리를 축소된 폴리트로픽 지수에 묻어 버린다.

3. **AWSoM removes the ad-hoc heating and closes the physics loop** — By evolving $w_\pm$ with explicit reflection and dissipation, AWSoM computes $Q_i,Q_e$ self-consistently from the Poynting flux boundary alone, not from a fitted function. This is the key conceptual advance. / **AWSoM 은 임시 가열을 제거하고 물리 루프를 닫는다** — 반사와 소산을 명시적으로 포함한 $w_\pm$ 진화를 통해 AWSoM 은 fit 함수가 아닌 Poynting flux 경계만으로 $Q_i,Q_e$ 를 자기정합적으로 계산한다. 이것이 핵심 개념적 진보이다.

4. **Two- and three-temperature physics matter** — Electrons dominate heat conduction; protons dominate bulk inertia; their $T_\parallel,T_\perp$ anisotropy reflects kinetic instability limits. A single-$T$ MHD model inevitably mis-identifies which species is heated where. / **2-온도·3-온도 물리는 중요하다** — 전자는 열전도를 지배하고, 양성자는 bulk 관성을 지배하며, $T_\parallel,T_\perp$ 이방성은 동역학적 불안정성 한계를 반영한다. 단일-$T$ MHD 모델은 필연적으로 어느 종이 어디서 가열되는지를 오판한다.

5. **Transition region is numerically brutal** — The 10-Mm TR with ≈10⁴ K/km gradient is the most numerically challenging element. Three solutions exist: (i) ultra-fine AMR (expensive), (ii) artificial broadening factor $f$ (standard since Abbett 2007), (iii) TFLM 1-D threads (Sokolov et al. 2016, the newest). / **천이영역은 수치적으로 가혹하다** — 10 Mm 두께, ≈10⁴ K/km gradient 의 TR 은 가장 수치적으로 어려운 요소이다. 세 해결책: (i) 극세 AMR(비쌈), (ii) Abbett 2007 이후 표준의 인공 broadening factor $f$, (iii) 최신 TFLM 1-D thread.

6. **Radiative losses and Spitzer conduction are non-negotiable** — Once the model extends below ≈1.05 $R_\odot$, $Q_{\rm rad}=N_eN_i\Lambda(T)$ and $\mathbf{q}_e\propto T^{5/2}\nabla T$ must be included; ignoring either destroys the coronal temperature profile and the EUV synthetic images lose observational fidelity. / **복사 냉각과 Spitzer 열전도는 타협 불가** — 모델이 ≈1.05 $R_\odot$ 아래로 내려오면 $Q_{\rm rad}=N_eN_i\Lambda(T)$, $\mathbf{q}_e\propto T^{5/2}\nabla T$ 가 필수이다. 둘 중 하나만 빠져도 코로나 온도 profile 이 파괴되고 EUV 합성 이미지가 관측 충실도를 잃는다.

7. **Boundary data (magnetograms) are the dominant uncertainty** — The input magnetogram (MDI/HMI/SOLIS/GONG) drives the whole simulation. Far-side uncertainty, line-of-sight saturation, and polar-field underestimation propagate throughout. Data-assimilation techniques (e.g., ADAPT) are increasingly essential. / **경계 데이터(자기사진)는 최대 불확실성원** — 입력 자기사진(MDI/HMI/SOLIS/GONG)이 전체 시뮬레이션을 구동한다. 원면 불확실성, 시선 포화, 극지방 자기장 과소평가가 전체로 전파된다. 자료 동화(예: ADAPT) 기법이 점점 필수가 되고 있다.

8. **The field is converging toward "digital twins" of the heliosphere** — SWMF, EUHFORIA, PsiMHD, and AWSoM-related codes are increasingly coupled to inner-heliosphere codes, providing an end-to-end chromosphere-to-Earth prediction pipeline. This is transforming space-weather forecasting from empirical to physics-based. / **해류권 "디지털 트윈" 으로 수렴 중** — SWMF, EUHFORIA, PsiMHD, AWSoM 관련 코드들이 내부 해류권 코드와 결합되어 색구~지구 end-to-end 예측 파이프라인을 제공한다. 이는 우주 날씨 예보를 경험적에서 물리 기반으로 변환시키고 있다.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Ideal MHD Equations (Eqs. 23–25) / 이상 MHD 방정식

**Continuity / 연속 방정식**:
$$\frac{\partial \rho}{\partial t}+\nabla\cdot(\rho \mathbf{u})=0$$
- $\rho$: mass density (kg m⁻³) / 질량 밀도
- $\mathbf{u}$: bulk velocity (same for ions and electrons in quasi-neutral plasma) / bulk 속도

**Momentum / 운동량 방정식**:
$$\frac{\partial(\rho\mathbf{u})}{\partial t}+\nabla\cdot\!\left(\rho\mathbf{uu}-\frac{\mathbf{BB}}{\mu_0}\right)+\nabla\!\left(p_i+p_e+\frac{B^2}{2\mu_0}+p_A\right)=-\frac{GM_\odot\rho\,\mathbf{r}}{r^3}$$
- $\mathbf{BB}/\mu_0$: Maxwell stress tensor / Maxwell 응력
- $B^2/(2\mu_0)$: magnetic pressure / 자기 압력
- $p_A=(w_++w_-)/2$: isotropic Alfvén wave pressure / 등방 알펜파 압력
- RHS: solar gravity / 태양 중력

**Induction / 유도 방정식**:
$$\frac{\partial \mathbf{B}}{\partial t}+\nabla\cdot(\mathbf{uB}-\mathbf{Bu})=0$$
ideal MHD frozen-in condition / 이상 MHD 동결 조건.

### 4.2 Two-temperature energy equations (Eqs. 26–27) / 2-온도 에너지 방정식

**Ion energy / 이온 에너지**:
$$\frac{\partial}{\partial t}\!\left(\frac{p_i}{\gamma-1}+\frac{\rho u^2}{2}+\frac{B^2}{2\mu_0}\right)+\nabla\cdot\bigl[\cdots\bigr]=-(\mathbf{u}\cdot\nabla)(p_e+p_A)+\frac{N_eN_ik_B\nu_{ei}}{(\gamma-1)N_i}(T_e-T_i)-\frac{GM_\odot\rho\mathbf{r}\cdot\mathbf{u}}{r^3}+Q_i$$

**Electron energy / 전자 에너지**:
$$\frac{\partial}{\partial t}\!\left(\frac{p_e}{\gamma-1}\right)+\nabla\cdot\!\left(\frac{p_e}{\gamma-1}\mathbf{u}\right)+p_e\nabla\cdot\mathbf{u}=-\nabla\cdot\mathbf{q}_e+\frac{N_eN_ik_B\nu_{ei}}{(\gamma-1)N_i}(T_i-T_e)-Q_{\rm rad}+Q_e$$

- Coulomb coupling equilibrates $T_i$ and $T_e$ where $\nu_{ei}$ is large. / Coulomb 결합이 $T_i$ 와 $T_e$ 를 같게 만든다 ($\nu_{ei}$ 가 클 때).
- $Q_{\rm rad}=N_eN_i\Lambda(T_e)$: CHIANTI-tabulated radiative loss / CHIANTI 복사 냉각
- Polytropic index $\gamma=5/3$.

### 4.3 Alfvén wave transport (Eqs. 31–36) / 알펜파 전달

$$\frac{\partial w_\pm}{\partial t}+\nabla\cdot\bigl[(\mathbf{u}\pm\mathbf{V}_A)w_\pm\bigr]+\frac{w_\pm}{2}(\nabla\cdot\mathbf{u})=-\Gamma_\pm w_\pm \mp \mathcal{R}\sqrt{w_-w_+}$$

- $\mathbf{V}_A=\mathbf{B}/\sqrt{\mu_0\rho}$: Alfvén velocity (m/s) / 알펜 속도
- Dissipation rate: $\Gamma_\pm=(2/L_\perp)\sqrt{w_\mp/\rho}$ / 소산율
- Reflection coefficient:
$$\mathcal{R}=\min\!\left\{\sqrt{(\hat{\mathbf{b}}\cdot[\nabla\times \mathbf{u}])^2+[(\mathbf{V}_A\cdot\nabla)\log V_A]^2},\ \max(\Gamma_\pm)\right\}\cdot[\text{imbalance}]$$
- Poynting flux boundary: $\Pi_A/B\approx 1.1\times 10^6$ W m⁻² T⁻¹
- Correlation length: $100 \le L_\perp\sqrt{B}\le 300$ km·T$^{1/2}$
- Wave heating split: $Q_i=f_p(\Gamma_-w_-+\Gamma_+w_+)$, $Q_e=(1-f_p)\cdot(\cdot)$, $f_p\approx 0.6$

### 4.4 Spitzer–Härm heat flux (Eq. 30) / Spitzer–Härm 열 유속

$$\mathbf{q}_e=\kappa_\parallel\,\hat{\mathbf{b}}\,(\hat{\mathbf{b}}\cdot\nabla T_e),\qquad \kappa_\parallel=\frac{3.2\cdot 6\pi}{\Lambda_C}\sqrt{\frac{2\pi\epsilon_0^2}{m_e e^2}}\,(k_B T_e)^{5/2}\,k_B$$

Heat transports only along field lines — key for the solar corona. / 열은 자기장 선을 따라서만 수송 — 태양 코로나의 핵심.

### 4.5 Transition-region jump condition (Eqs. 42–44) / 천이영역 점프 조건

$$\frac{\partial}{\partial s}\!\left(\kappa_0 T_e^{5/2}\frac{\partial T_e}{\partial s}\right)+Q_h-N_e^2\Lambda(T_e)=0$$

Multiplying by $\kappa_0 T_e^{5/2}(\partial T_e/\partial s)$ and integrating gives

$$(N_eT_e)=\sqrt{\frac{\tfrac{1}{2}\kappa_0^2 T_e^5(\partial_s T_e)^2+\tfrac{2}{7}\kappa_0 Q_h(T_e^{7/2}-T_{ch}^{7/2})}{\int_{T_{ch}}^{T_e}\kappa_0 T^{1/2}\Lambda(T)\,dT}}$$

Gravity-neglect condition: $L_g(T_e)\approx T_e\cdot(60\text{ m/K})\gg L_h\approx\sqrt{\kappa_0 T_e^{9/2}/[\Lambda(T_e)(N_eT_e)^2]}$.

### 4.6 Parker wind solution (Eq. 1) / Parker 태양풍 해

$$\left[\frac{v^2}{v_m^2}-\ln\!\left(\frac{v^2}{v_m^2}\right)\right]=4\ln\!\left(\frac{r}{a}\right)+\left(\frac{v_{\rm esc}^2}{v_m^2}\right)\!\left(\frac{a}{r}\right)-4\left(\frac{v_{\rm esc}^2}{v_m^2}\right)-3+\ln 256$$

Critical point at $r_c=a v_{\rm esc}^2/(4 v_m^2)$. Supersonic branch: $v_0\ll v_m$, $v(\infty)>v_m$.

**Worked example / 수치 예제**: For $T_0=1.5\times 10^6$ K, $v_m=\sqrt{2k_B T_0/m_p}=157$ km/s; $v_{\rm esc}(R_\odot)=618$ km/s. Take $a=R_\odot$ so $v_{\rm esc}^2/v_m^2\approx 15.5$, $r_c/a\approx 3.9$. The asymptotic wind speed is $v(\infty)\approx 480$ km/s — in agreement with the slow wind observed by Mariner-2. For $T_0=3\times 10^6$ K, $v(\infty)\approx 800$ km/s — matching Ulysses polar fast streams.

### 4.7 Polytropic MHD as simpler alternative / 단순 대안으로서의 폴리트로픽 MHD

Using $p=K\rho^\gamma$ with $\gamma\approx 1.05$–$1.1$ mimics heat deposition and avoids the full energy equation. Used in the "thermodynamic" pre-AWSoM models. Physically crude but numerically robust.

### 4.8 CME flux rope insertion (Titov–Démoulin / Gibson–Low) / CME 자속관 삽입

A twisted flux rope (typically Titov–Démoulin 1999 or Gibson–Low 1998 form) is inserted into the relaxed steady corona. The Lundquist-type analytic form in cylindrical $(r,\phi,z)$:

$$B_z=B_0 J_0(\alpha r),\quad B_\phi=B_0 J_1(\alpha r),\quad B_r=0$$

with constant-$\alpha$ force-free condition $\nabla\times\mathbf{B}=\alpha\mathbf{B}$. Injected CME speeds in AWSoM simulations range from 500 to 3000 km/s, spanning slow-to-extreme events.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1859 ─ Carrington & Hodgson flare (geomagnetic storm)
1892 ─ FitzGerald: "comet-tail-like emission"
1931 ─ Chapman & Ferraro: cold plasma beam
1942 ─ Alfvén: magnetohydrodynamic waves
1951 ─ Biermann: comet-tail acceleration → continuous outflow
1957 ─ Alfvén: frozen-in field in solar flow
1958 ─ Parker: solar wind hydrodynamic solution  ★
1960 ─ Chamberlain: "solar breeze" (wrong)
1962 ─ Mariner-2: confirmation (Neugebauer & Snyder 1966)
1965 ─ Scarf & Noble: first Navier–Stokes numerical solution
1966 ─ Sturrock & Hartle: first two-fluid model  ★
1968 ─ Coleman: Alfvén wave heating hypothesis
1969 ─ Altschuler & Newkirk: PFSS (Rs = 2.5 R⊙)  ★
1971 ─ Pneuman & Kopp: first 2-D MHD helmet streamer  ★
1977 ─ Jacques: Alfvén wave pressure driving wind
1978 ─ Steinolfson+: 2-D axisymmetric MHD corona
1990 ─ Wang & Sheeley: WSA empirical speed model
1994 ─ Usmanov: early 3-D MHD heliosphere
1999 ─ Matthaeus+: counter-propagating Alfvén turbulence
2000 ─ Usmanov+: first 2-D self-consistent Alfvén-wave corona
2001 ─ Lionello+: analytical TR solution
2005 ─ Suzuki & Inutsuka: 1-D Alfvén wave corona model
2007 ─ Abbett: artificial TR broadening
2010 ─ van der Holst+: 2-T AWSoM  ★
2013 ─ Sokolov+: formal AWSoM v1
2014 ─ van der Holst+: 3-T AWSoM with anisotropy  ★
2016 ─ Sokolov+: Threaded Field Line Model (TFLM)  ★
2017 ─ Jin+: EUV synthetic validation of AWSoM on CR2107
2018 ─ THIS REVIEW — Gombosi+ synthesis  ★
2018 ─ Parker Solar Probe launch (inner heliosphere era)
2020 ─ Solar Orbiter launch
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Parker (1958) "Dynamics of the interplanetary gas and magnetic fields" | Original hydrodynamic solar-wind solution; Eq. (1) of this review / 본 총설 Eq. (1) 의 원본 | Foundation — everything else extends this / 기초 — 다른 모든 것이 이를 확장 |
| Sturrock & Hartle (1966) | First 2-fluid solar-wind model; origin of $T_p\neq T_e$ / 최초 2-유체 태양풍 모델 | Precursor to AWSoM 2-T energetics / AWSoM 2-T 에너지학의 선구 |
| Altschuler & Newkirk (1969) | PFSS model; the magnetogram-to-coronal-field mapping / 자기사진→코로나 자기장 매핑 | Boundary condition for all modern MHD coronal codes / 모든 현대 MHD 코로나 코드의 경계조건 |
| Pneuman & Kopp (1971) | First 2-D MHD helmet streamer / 최초 2-D MHD helmet streamer | Conceptual template for 3-D streamer-belt / 3-D streamer belt 개념 템플릿 |
| Sokolov et al. (2013) "Magnetohydrodynamic waves and coronal heating: unifying empirical and MHD turbulence models" | Formal introduction of AWSoM / AWSoM 공식 도입 | The paper being reviewed here / 본 총설의 주제 |
| van der Holst et al. (2014) "AWSoM: A global solar wind model" | 3-T anisotropic AWSoM / 3-T 이방성 AWSoM | Most advanced version discussed / 가장 발전된 버전 |
| Cranmer & van Ballegooijen (2010) | Turbulent cascade and damping mechanism / 난류 계단·감쇠 기제 | Physics basis for $\Gamma_\pm$ in AWSoM / AWSoM $\Gamma_\pm$ 의 물리적 근거 |
| Chandran et al. (2011) | Stochastic heating of protons / 양성자 확률적 가열 | Justifies $f_p\approx 0.6$ proton/electron partition / $f_p$ 분배의 근거 |
| Matthaeus et al. (1999) | Counter-propagating Alfvén turbulence in corona / 역방향 전파 알펜 난류 | Direct theoretical parent of AWSoM wave equations / AWSoM 파동 방정식의 직접적 이론 부모 |
| Lionello et al. (2001) | Analytical TR solution / 해석적 TR 해 | Basis for TR jump condition used in AWSoM / AWSoM TR 점프 조건의 기초 |
| Manchester et al. (2004) | CME flux-rope injection in SWMF / SWMF의 CME 자속관 삽입 | How transient events are initialized / 과도 현상 초기화 방법 |
| Jin et al. (2017) | AWSoM validation against SDO/AIA and STEREO/EUVI / SDO/AIA, STEREO/EUVI 대비 AWSoM 검증 | Quantitative test of the method reviewed / 본 방법의 정량적 검증 |

---

## 7. References / 참고문헌

- Gombosi, T. I., van der Holst, B., Manchester, W. B., & Sokolov, I. V. (2018). "Extended MHD modeling of the steady solar corona and the solar wind." *Living Reviews in Solar Physics*, 15:4. DOI: 10.1007/s41116-018-0014-4. [Primary paper]
- Parker, E. N. (1958). "Dynamics of the Interplanetary Gas and Magnetic Fields." *Astrophysical Journal*, 128, 664. DOI: 10.1086/146579
- Biermann, L. (1951). "Kometenschweife und solare Korpuskularstrahlung." *Z. Astrophys.*, 29, 274.
- Neugebauer, M. & Snyder, C. W. (1966). "Mariner 2 observations of the solar wind." *JGR*, 71, 4469. DOI: 10.1029/JZ071i019p04469
- Sturrock, P. A. & Hartle, R. E. (1966). "Two-fluid model of the solar wind." *Phys. Rev. Lett.*, 16, 628. DOI: 10.1103/PhysRevLett.16.628
- Altschuler, M. D. & Newkirk, G. (1969). "Magnetic fields and the structure of the solar corona. I." *Solar Phys.*, 9, 131. DOI: 10.1007/BF00145734
- Pneuman, G. W. & Kopp, R. A. (1971). "Gas-magnetic field interactions in the solar corona." *Solar Phys.*, 18, 258. DOI: 10.1007/BF00145940
- Sokolov, I. V., van der Holst, B., Oran, R., et al. (2013). "Magnetohydrodynamic waves and coronal heating: unifying empirical and MHD turbulence models." *ApJ*, 764, 23. DOI: 10.1088/0004-637X/764/1/23
- van der Holst, B., Sokolov, I. V., Meng, X., et al. (2014). "Alfvén Wave Solar Model (AWSoM): coronal heating." *ApJ*, 782, 81. DOI: 10.1088/0004-637X/782/2/81
- Meng, X., van der Holst, B., Toth, G., Gombosi, T. I. (2015). "Alfvén wave solar model (AWSoM) with anisotropic proton temperature." *MNRAS*, 454, 3697. DOI: 10.1093/mnras/stv2249
- Cranmer, S. R. & van Ballegooijen, A. A. (2010). "Can the solar wind be driven by magnetic reconnection in the Sun's magnetic carpet?" *ApJ*, 720, 824.
- Chandran, B. D. G., Dennis, T. J., Quataert, E., Bale, S. D. (2011). "Incorporating kinetic physics into a two-fluid solar wind model." *ApJ*, 743, 197.
- Matthaeus, W. H., Zank, G. P., Oughton, S., et al. (1999). "Coronal heating by MHD turbulence driven by reflected low-frequency waves." *ApJ*, 523, L93.
- Lionello, R., Linker, J. A., Mikić, Z. (2001). "Including the transition region in models of the large-scale solar corona." *ApJ*, 546, 542.
- Sokolov, I. V., van der Holst, B., Manchester, W. B., et al. (2016). "Threaded-field-lines model for the low solar corona powered by the Alfvén wave turbulence." *ApJ*, 832, 94.
- Jin, M., Manchester, W. B., van der Holst, B., et al. (2017). "Data-constrained coronal mass ejections in a global magnetohydrodynamics model." *ApJ*, 834, 173.
- Spitzer, L. & Härm, R. (1953). "Transport phenomena in a completely ionized gas." *Phys. Rev.*, 89, 977.
- Wang, Y.-M. & Sheeley, N. R. (1990). "Solar wind speed and coronal flux-tube expansion." *ApJ*, 355, 726.
- Chamberlain, J. W. (1960). "Interplanetary gas. II. Expansion of a model solar corona." *ApJ*, 131, 47.
- Dere, K. P., Landi, E., Mason, H. E., et al. (1997). "CHIANTI — an atomic database for emission lines." *A&AS*, 125, 149.
- Titov, V. S. & Démoulin, P. (1999). "Basic topology of twisted magnetic configurations in solar flares." *A&A*, 351, 707.
- Gibson, S. E. & Low, B. C. (1998). "A time-dependent three-dimensional magnetohydrodynamic model of the coronal mass ejection." *ApJ*, 493, 460.
