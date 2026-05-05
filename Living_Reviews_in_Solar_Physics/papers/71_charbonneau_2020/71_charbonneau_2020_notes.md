---
title: "Dynamo Models of the Solar Cycle (2020 update)"
authors: Paul Charbonneau
year: 2020
journal: "Living Reviews in Solar Physics"
doi: "10.1007/s41116-020-00025-6"
topic: Living_Reviews_in_Solar_Physics
tags: [solar-cycle, dynamo, MHD, Babcock-Leighton, mean-field, flux-transport, Grand-Minima, stochastic]
status: completed
date_started: 2026-04-23
date_completed: 2026-04-23
---

# 71. Dynamo Models of the Solar Cycle (2020 update) / 태양 주기의 다이나모 모델 (2020 갱신판)

> **Relation to Paper #20 / Paper #20과의 관계**
>
> This 2020 review is a **major revision** of Paper #20 (Charbonneau 2010). The scaffolding (Sects. 1–4) is largely preserved — MHD induction equation, Cowling's theorem, α-effect, αΩ mean-field equations, Parker–Yoshimura sign rule — but three parts are **substantially updated**:
>
> 1. **Babcock–Leighton models** (Sect. 5) — promoted to a dedicated section, with new 3D / 2×2D non-axisymmetric models.
> 2. **Global MHD simulations** (Sect. 6) — entirely new; 2010-era simulations could not yet produce solar-like large-scale cycles.
> 3. **Amplitude fluctuations / Grand Minima** (Sect. 7) — reorganized around generic dynamical behaviors, and shortened to point at the technical literature.
>
> 본 2020년 리뷰는 Paper #20(Charbonneau 2010)의 **major revision**이다. 1–4장의 기본 골격(MHD 유도방정식, Cowling 정리, α 효과, αΩ 평균장 방정식, Parker–Yoshimura sign rule)은 대부분 유지되지만, 세 부분이 크게 업데이트되었다: (1) **Babcock–Leighton 모델**이 별도의 Sect. 5로 확장되고 3D/2×2D 비축대칭 모델이 추가됨; (2) **전지구 MHD 시뮬레이션**(Sect. 6)이 완전히 새로 추가됨; (3) **진폭 변동 및 Grand Minima**(Sect. 7)가 일반 거동 중심으로 재편되고 간결화됨.

---

## 1. Core Contribution / 핵심 기여

Charbonneau(2020)은 태양 주기를 hydromagnetic dynamo 과정으로 모델링하는 최근 10년간의 발전과 현재 논쟁을 포괄적으로 정리한 **living review**이다. 저자는 관측과 비교 가능한 수준의 세부성을 지닌 **(상대적으로) 단순한 mean-field / mean-field-like 모델**에 초점을 두면서도, 주요 관측 제약(polarity reversal with 11-year decadal half-period, 22-year magnetic cycle, sunspot butterfly diagram, poleward drift of diffuse surface field, polar field ~5 G, predominantly negative magnetic helicity in Northern hemisphere, antisymmetric equatorial parity, π/2 phase lag between poloidal and toroidal)과의 비교를 엄격하게 요구한다. Sect. 2–4는 MHD induction equation·Cowling's theorem·mean-field electrodynamics·αΩ 다이나모의 기초를 제공하고, Sect. 5는 **Babcock–Leighton 계열**(kinematic 2D, 3D flux emergence, 2×2D coupled SFT + interior dynamo, Lemerle & Charbonneau 2017)을 집중적으로 다룬다. Sect. 6는 **global MHD 시뮬레이션**(EULAG-MHD, ASH, Pencil Code)의 현재 성취—solar-like cycle period, equatorward migration without mean-field kinematics, α-tensor의 직접 측정, Grand-Minima-like 사건—를 요약한다. Sect. 7은 Hopf bifurcation, stochastic forcing, nonlinear modulation, time delays, thresholded modulation, on–off/in–out intermittency를 일반 거동으로 분류하고, Sect. 8은 8개의 열린 질문으로 마무리한다.

Charbonneau (2020) is a comprehensive *living review* of progress over the past decade in modeling the solar cycle as a hydromagnetic dynamo process. The author focuses on **(relatively) simple mean-field / mean-field-like models** that are nonetheless detailed enough to be compared against solar cycle observations. The target observational benchmarks are strict: decadal polarity reversals, 22-year magnetic cycle, sunspot butterfly, poleward surface drift, ~5 G polar fields, predominantly negative N-hemisphere magnetic helicity, antisymmetric equatorial parity, and the π/2 poloidal–toroidal phase lag. Sects. 2–4 cover MHD induction, Cowling's theorem, mean-field electrodynamics, and αΩ dynamos. Sect. 5 is a dedicated treatment of the **Babcock–Leighton family** — kinematic 2D, 3D flux emergence, and the 2×2D coupled SFT-plus-interior-dynamo approach of Lemerle & Charbonneau (2017). Sect. 6 reviews what **global MHD simulations** (EULAG-MHD, ASH, Pencil Code) now achieve: solar-like cycle periods, equatorward migration without any mean-field kinematic prescription, direct measurement of the α-tensor, and spontaneous Grand-Minima-like episodes. Sect. 7 classifies amplitude variability into generic dynamical behaviors (Hopf bifurcation, stochastic forcing, nonlinear modulation, time delays, thresholded modulation, on–off / in–out intermittency). Sect. 8 closes with eight open questions.

---

## 2. Reading Notes / 읽기 노트

### Part I: Setup and MHD framework (Sect. 1–2) / 설정과 MHD 프레임워크

#### 1.1–1.4 Scope, "model", historical survey, butterfly diagram

저자는 리뷰의 범위를 **dynamo models of the solar cycle**로 좁힌다. 다음 주제는 다루지 않는다: solar magnetic field observations 상세, magnetic flux tubes/ropes 물리, 태양 표면하 소규모 자기장 생성, solar cycle prediction, 다른 별의 자기장. 이는 Paper #15 (Hathaway 2015), Paper #18 (Fan 2009), Paper #38 (Cheung & Isobe 2014) 등이 각각 상세히 다룬다.

역사적으로: **1843** Schwabe의 11년 주기 발견 → **1908** Hale의 자성 증명 및 polarity law → **1919** Larmor의 hydromagnetic dynamo 아이디어 → **1934** Cowling's antidynamo theorem(축대칭 흐름은 축대칭 자기장을 dissipation에 대항해 유지할 수 없음) → **1955** Parker의 α-effect와 dynamo wave → **1960년대** Steenbeck–Krause–Rädler mean-field electrodynamics가 표준 이론으로 자리잡음 → **1970–80년대** magnetic buoyancy problem, α-quenching 계산, helioseismology가 정식 CZ에서의 α-effect 사용에 타격을 주면서 "three-way punch." 2010년판 이후 consensus는 여전히 없다.

The author narrows the scope strictly to **dynamo models of the solar cycle**. Excluded: observational surveys of solar magnetic fields, magnetic flux-tube physics, small-scale field generation, solar cycle prediction, and stellar dynamos. These are covered elsewhere (Paper #15 Hathaway 2015; Paper #18 Fan 2009; Paper #38 Cheung & Isobe 2014).

Historical arc: **1843** Schwabe discovers the 11-year cycle → **1908** Hale demonstrates magnetism and polarity laws → **1919** Larmor's hydromagnetic idea → **1934** Cowling's antidynamo theorem (axisymmetric flows cannot sustain axisymmetric fields against ohmic dissipation) → **1955** Parker's α-effect and dynamo wave → **1960s** Steenbeck–Krause–Rädler mean-field electrodynamics as mainstream → **1970–80s** the "three-way punch" (magnetic buoyancy problem, α-quenching calculations, helioseismology-derived Ω(r,θ) incompatible with mean-field dynamo solutions). As of 2020 there is still no consensus.

**Observational targets the sunspot butterfly imposes** (Fig. 2):
- Sunspots restricted to $|\lambda|\lesssim 30^\circ$.
- Emergence starts at ~15°, drifts toward the equator.
- Hemispheric synchrony.
- Polar field ~5 G reverses at sunspot maximum (Fig. 3).
- Polar cap flux $\sim 10^{22}$ Mx vs. total cycle emergence $\sim 10^{25}$ Mx → interior field dominated by toroidal component.

#### 2.1–2.4 MHD induction equation, dynamo problem, kinematic models, axisymmetric formulation

**MHD induction equation (paper Eq. 1)**:
$$\frac{\partial\mathbf{B}}{\partial t}=\nabla\times(\mathbf{u}\times\mathbf{B}-\eta\nabla\times\mathbf{B}),\quad \eta=c^2/(4\pi\sigma_e)$$
에너지 방정식을 스칼라 곱으로 얻으면(p. 9):
$$\frac{d}{dt}\int_V \frac{B^2}{8\pi}\,dV = -\oint_{\partial V}\mathbf{S}\cdot\hat{\mathbf{n}}\,dA-\int_V\frac{J^2}{\sigma_e}\,dV-\frac{1}{c}\int_V\mathbf{u}\cdot(\mathbf{J}\times\mathbf{B})\,dV$$
마지막 항의 음수화(즉, Lorentz force를 거스르는 일)가 dynamo 작동의 핵심. | The last term — flow doing work against the Lorentz force — is the essence of dynamo action.

**Rewriting the inductive term** (paper Eq. 5):
$$\nabla\times(\mathbf{u}\times\mathbf{B})=\underbrace{(\mathbf{B}\cdot\nabla)\mathbf{u}}_{\text{shearing}}-\underbrace{\mathbf{B}(\nabla\cdot\mathbf{u})}_{\text{compression}}-\underbrace{(\mathbf{u}\cdot\nabla)\mathbf{B}}_{\text{transport}}$$
전단·압축·수송이 유일한 증폭 경로; 강하게 아음속이라 압축은 무시 가능. | Shearing, compression, transport are the only amplification channels; compression negligible for subsonic flows.

**Axisymmetric decomposition (Sect. 2.4)**:
$$\mathbf{B}(r,\theta,t)=\nabla\times(A(r,\theta,t)\hat{\mathbf{e}}_\phi)+B(r,\theta,t)\hat{\mathbf{e}}_\phi$$
$$\mathbf{u}(r,\theta)=\mathbf{u}_p(r,\theta)+\varpi\Omega(r,\theta)\hat{\mathbf{e}}_\phi,\quad \varpi=r\sin\theta$$
결과는 A, B에 대한 두 결합 PDE (paper Eqs. 8–9):
$$\frac{\partial A}{\partial t}=\eta\!\left(\nabla^2-\tfrac{1}{\varpi^2}\right)\!A-\tfrac{\mathbf{u}_p}{\varpi}\cdot\nabla(\varpi A)$$
$$\frac{\partial B}{\partial t}=\eta\!\left(\nabla^2-\tfrac{1}{\varpi^2}\right)\!B+\tfrac{1}{\varpi}\tfrac{\partial(\varpi B)}{\partial r}\tfrac{\partial\eta}{\partial r}-\varpi\mathbf{u}_p\!\cdot\!\nabla\!\big(\tfrac{B}{\varpi}\big)-B\nabla\!\cdot\!\mathbf{u}_p+\varpi(\nabla\times(A\hat{\mathbf{e}}_\phi))\!\cdot\!\nabla\Omega$$
A 방정식에는 source term이 없음 → Cowling 정리의 수학적 표현. | A equation has no source term, the quantitative statement of Cowling's theorem.

### Part II: Regeneration mechanisms (Sect. 3) / 재생성 메커니즘

**P→T**: shearing term $(\mathbf{B}_p\cdot\nabla\Omega)t$은 solar-like differential rotation 하에서 10 G의 dipole을 10년에 1 kG toroidal로 만들 수 있다(paper Eq. 14). 문제 없음.

**T→P** 옵션 세 가지:
1. **Mean-field electrodynamics** (Sect. 3.2.1): 난류의 helical 성분이 α-effect. 유효 EMF $\mathcal{E}=\boldsymbol{\alpha}\langle\mathbf{B}\rangle+\boldsymbol{\gamma}\times\langle\mathbf{B}\rangle-\boldsymbol{\beta}\cdot(\nabla\times\langle\mathbf{B}\rangle)$. 2010년판 대비 큰 변화는 없으나, MHD 시뮬레이션에서 α 텐서를 직접 측정하는 방법이 성숙(Sect. 6).
2. **Babcock–Leighton mechanism** (Sect. 3.2.2 & Sect. 5): 기울어진 BMR의 trailing polarity가 확산·수송을 통해 global dipole에 기여. 관측(synoptic magnetogram)에서 직접 보임.
3. **HD/MHD 불안정성** (Sect. 3.2.3): tachocline의 shear 불안정성(Dikpati & Gilman 2001), sheared magnetic layer 불안정성(Thelen 2000b), toroidal flux tube buoyancy 불안정성(Ferriz-Mas et al. 1994)이 azimuthal EMF를 만듦.

### Part III: αΩ mean-field models (Sect. 4) / αΩ 평균장 모델

**4.2.7 The αΩ dynamo equations** (paper Eqs. 38–39):
$$\frac{\partial\langle A\rangle}{\partial t}=(\eta+\beta)\!\left(\nabla^2-\tfrac{1}{\varpi^2}\right)\!\langle A\rangle-\tfrac{\mathbf{u}_p}{\varpi}\!\cdot\!\nabla(\varpi\langle A\rangle)+\alpha\langle B\rangle$$
$$\frac{\partial\langle B\rangle}{\partial t}=(\eta+\beta)\!\left(\nabla^2-\tfrac{1}{\varpi^2}\right)\!\langle B\rangle+\varpi(\nabla\times(\langle A\rangle\hat{\mathbf{e}}_\phi))\!\cdot\!\nabla\Omega+\nabla\times[\alpha\nabla\times(\langle A\rangle\hat{\mathbf{e}}_\phi)]+\cdots$$
Dynamo numbers $C_\alpha=\alpha_0 R_\odot/\eta_0$, $C_\Omega=(\Delta\Omega)_0 R_\odot^2/\eta_0$, $D=C_\alpha\cdot C_\Omega$. Critical $D_\mathrm{crit}$ 초과 시 Hopf bifurcation.

**4.2.9 Parker–Yoshimura sign rule** (paper Eq. 45):
$$\mathbf{s}=\alpha\nabla\Omega\times\hat{\mathbf{e}}_\phi$$
Solar-like Ω(r,θ) (Fig. 4b) 하에서, 북반구 **positive α**는 **poleward** dynamo wave를, **negative α**는 **equatorward**를 준다. 이것이 태양 butterfly를 설명할 때 kinematic αΩ 모델에서 **negative α in low latitudes**가 필요한 이유. 또는 latitudinal shear (negative at mid-latitudes) 우위 → equatorward.

**4.2.11 Critical assessment**: equatorward drift를 dynamo wave로 설명하는 점이 성공. 그러나 α의 부호·위치·quenching이 ad hoc.

**Cycle period scaling**: dimensional period typically $\sim R_\odot^2/\eta_\mathrm{T}$; $\eta_\mathrm{T}\sim 10^{11}$–$10^{12}$ cm²/s → $\tau\sim$ 수 십 년. 조정으로 11년 재현 가능.

**4.3 Interface dynamos**: α와 Ω를 공간적으로 분리(tachocline에 Ω, CZ overshoot에 α). Strong α-quenching 회피. Parker(1993), MacGregor & Charbonneau(1997).

**4.4 Flux-transport dynamos**: meridional circulation이 P↔T 영역을 연결. Rm이 높을 때 meridional flow가 period를 결정 → solar-like 11년 period와 equatorward butterfly가 자연스럽게. Choudhuri et al. 1995, Dikpati & Charbonneau 1999, Küker et al. 2001. **약점**: single-cell, steady meridional flow 가정. 최근 helioseismic 역문제는 multi-cell 가능성 제시(Zhao et al. 2013) → dynamo 거동 크게 바뀔 수 있음.

**4.5 HD/MHD instabilities**: tachocline α-effect (Dikpati & Gilman 2001), buoyant flux tubes (Ferriz-Mas et al. 1994): stability diagram (Fig. 11)에서 60–150 kG 범위, 저위도에서만 unstable → 태양 low-latitude butterfly와 호환.

### Part IV: Babcock–Leighton models (Sect. 5) / Babcock–Leighton 모델 — **NEW / 크게 확장**

#### 5.1 Tilts of bipolar active regions

BMR dipole contribution per emergence (paper Eq. 49):
$$\delta D=\frac{3d\cos\lambda}{4\pi R^2}\Phi\sin\alpha,\qquad\sin\alpha\approx 0.5\sin\lambda\ (\text{Joy's law})$$
Φ는 unsigned flux, d는 pole separation. 관측 scatter 크며 이것이 dynamo stochasticity의 주요 원천. Pevtsov et al. 2014: tilt distribution이 약 15°의 표준편차를 가짐.

Flux-rope simulations는 $B_0\gtrsim 10^5$ G에서 tilt가 약해짐 → **tilt quenching**. 반대로 $B_0\lesssim 10^4$ G에서는 turbulent convection이 tilt를 방해 → BL mechanism의 **lower threshold**. 이것은 BL 다이나모가 self-excited가 아닌 이유.

#### 5.2 Surface flux transport (SFT)

**Fig. 12 (Lemerle et al. 2015)**: Cycle 21의 관측된 active region emergence를 입력으로 사용한 SFT 시뮬레이션이 polar field ~5 G 반전과 butterfly를 재현.

#### 5.3 Magnetic flux transport

BL 모델은 **advection-dominated** (meridional circulation > diffusion) vs. **diffusion-dominated**으로 분류. 기준은 magnetic Reynolds number $Rm=u_0 R_\odot/\eta_T$.

#### 5.4 Axisymmetric kinematic mean-field-like models

αΩ 형태의 kinematic equations에서 α항을 **non-local BL source term**으로 교체. Dikpati & Charbonneau 1999: surface 근처에 tachocline toroidal field의 값에 비례하는 source 배치.

**Fig. 13–14**: Charbonneau et al. 2005 meridional plane animation과 butterfly diagram. **Cycle period scaling** (paper Eq. 51):
$$P=56.8\,u_0^{-0.89}s_0^{-0.13}\eta_T^{0.22}\ [\text{years}]$$
→ advection-dominated에서 meridional flow speed $u_0\sim 10$–20 m/s가 period 지배.

**Turbulent pumping variation** (Guerrero & de Gouveia Dal Pino 2008, paper Eq. 52):
$$P=181.2\,u_0^{-0.12}\gamma_{r0}^{-0.51}\gamma_{\theta0}^{-0.05}\ [\text{years}]$$
→ radial pumping speed $\gamma_{r0}\sim 0.3$ m/s가 period 결정.

**약점**(5.4.3): steady single-cell meridional circulation은 helioseismology 최근 결과와 tension. 표면 polar field 과도하게 커짐. Polar branch dominant인 경향.

#### 5.5 Beyond 2D: non-axisymmetric models — **핵심 업데이트 / Key update**

- **Miesch & Dikpati 2014 / Miesch & Teweldebirhan 2016**: 내부 mean-field-like 2D dynamo에서 tilted 3D flux ring이 임계치 초과 시 배출. 자연스럽게 Joy's law 기울기.
- **Yeates & Muñoz-Jaramillo 2013b**: vortical upflow로 BMR emergence 구현. **Fig. 16**에 emergence 메커니즘.
- **Karak & Miesch 2017**: 장기간 3D BL 시뮬레이션. **Fig. 17**에 solar-like 주기와 반구 동기성. Tilt에 random scatter 도입 → Grand-Minima-like 사건 발생.
- **Lemerle & Charbonneau 2017 (2×2D)**: 2D SFT + 2D interior dynamo 결합. BMR 통계를 관측에서 추출(Lemerle et al. 2015, Jiang et al. 2011). **Fig. 18**: 한 rogue active region이 큰 cycle amplitude 감소 유발 → "rogue BMR" 이론이 관측된 cycle 23–24 transition을 재현.

#### 5.6 Surface dipole as precursor

BL-계 모델이 아닌 αΩ+MC 모델에서도 surface polar field가 다음 cycle amplitude 예측자 역할 가능(Fig. 19). 단, **meridional flow가 on**이어야 precursor 관계 성립. Correlation coefficient $r\approx 0.95$ between dipole-at-minimum and next-cycle-amplitude.

### Part V: Global MHD simulations (Sect. 6) / 전지구 MHD 시뮬레이션 — **ENTIRELY NEW**

Paper #20(2010)에 없던 내용. 약 10년 사이 ASH, EULAG-MHD, Pencil Code 시뮬레이션이 solar-like cycle을 자발적으로 생성.

**Governing equations (paper Eqs. 53–56)**:
$$\frac{\partial\rho}{\partial t}+\nabla\cdot(\rho\mathbf{u})=0$$
$$\frac{\partial\mathbf{u}}{\partial t}+(\mathbf{u}\cdot\nabla)\mathbf{u}=-\tfrac{1}{\rho}\nabla p-2\boldsymbol{\Omega}\times\mathbf{u}+\mathbf{g}+\tfrac{1}{4\pi\rho}(\nabla\times\mathbf{B})\times\mathbf{B}+\tfrac{1}{\rho}\nabla\cdot\boldsymbol{\tau}$$
$$\frac{\partial e}{\partial t}=(\gamma-1)e\nabla\cdot\mathbf{u}=\tfrac{1}{\rho}[\nabla\cdot((\chi+\chi_r)\nabla T)+\phi_u+\phi_B]$$
$$\frac{\partial\mathbf{B}}{\partial t}=\nabla\times(\mathbf{u}\times\mathbf{B}-\eta\nabla\times\mathbf{B})$$

**Key findings (Sect. 6.1–6.8)**:
- **6.1 Convection**: solar-like differential rotation 얻지만 torsional oscillation의 정확한 phase는 도전적.
- **6.2 Polarity reversals**: Käpylä et al. 2012, Ghizaru et al. 2010, Racine et al. 2011 — solar-like 22년 magnetic cycle 자연스럽게 발현. Equatorward branch도.
- **6.3 Field storage/amplification in tachocline**: rigid lower boundary 없이도 equipartition-strength toroidal field 형성 가능.
- **6.4 Turbulent induction and mean-field coefficients**: α-tensor와 β-tensor를 시뮬레이션에서 직접 측정. Diagonal α가 latitude·depth에 따라 부호 바꿈.
- **6.5 α-effect and turbulent diffusivity magnetic quenching**: Λ-quenching도 작동.
- **6.7 Formation of buoyant magnetic structures**: Nelson et al.(2013, 2014)에서 equipartition-strength 자성 자기관 자발 형성.
- **6.8 Lessons**: tachocline 없이도 solar-like cycle 가능 → **"tachocline is essential" consensus 흔들림**.

### Part VI: Amplitude fluctuations, multiperiodicity, Grand Minima (Sect. 7) / 진폭 변동·다중주기·Grand Minima

**7.1 Observational overview**:
- Maunder Minimum: 1645–1705 sunspot dearth.
- 유사 에피소드: Wolf Minimum(1282–1342), Spörer Minimum(1416–1534), Medieval Maximum(1100–1250) — cosmogenic radioisotope(¹⁴C, ¹⁰Be)에서 확인.
- Gleissberg cycle ~88 year modulation.
- Waldmeier Rule: amplitude–rise-time anti-correlation.
- Gnevyshev–Ohl Rule: 홀수 cycle이 짝수보다 큼(또는 반대).

**7.2 Generic behaviors**:
- **7.2.1 Going critical and Hopf-ing along** (Fig. 23): $D=D_\mathrm{crit}$에서 Hopf bifurcation. 임계 근방에서 D 변동이 amplitude 변동으로 크게 증폭.
- **7.2.2 Stochastic forcing (paper Eq. 64 — Cameron & Schüssler 2017b)**:
  $$\frac{dX}{dt}-(\beta+i\omega_0)X+(\gamma_r+i\gamma_i)|X|^2X=\sigma X\frac{dW}{dt}$$
  X는 magnetic field 측도, W는 Wiener 과정. 관측과 일치하는 SSN-like 시계열.
- **7.2.3 Nonlinear modulation / "surfing the wave"** (Figs. 24–26): magnetic field가 differential rotation에 Lorentz 반작용 → D 감소 → amplitude 감소 → differential rotation 회복 → 다시 증가. **Parity modulation**: symmetric(quadrupole) vs. antisymmetric(dipole) 모드 사이 전환. Type I/II modulation (Tobias et al. 1995).
- **7.2.4 Time delays**: BL 모델에서 meridional circulation이 poloidal→toroidal에 시간지연 도입. Durney 2000, Charbonneau 2001 — period doubling 캐스케이드와 혼돈.
- **7.2.5 Rattling the conveyor belt**: meridional flow 변동이 cycle amplitude 변동 유도. Nandy et al. 2011 — cycle 23–24 extended minimum을 circulation 감속으로 설명.

**7.3 Intermittency and Grand Minima/Maxima**:
- **On–off intermittency** (Fig. 28a): stochastic/deterministic forcing이 D를 주기적으로 subcritical로 밀어냄.
- **In–out intermittency** (Fig. 28b): self-excited 아님. Lower operating threshold 존재 → BL 다이나모에 해당. Charbonneau et al. 2004, Karak & Choudhuri 2013.
- **Fig. 29 (Olemskoy & Kitchatinov 2013)**: stochastic BL에서 Spörer-like 100년 Grand Minimum 발현. Inter-event waiting time exponential → memoryless process, cosmogenic 데이터와 일치.

**7.4 Thresholded amplitude modulation**: 같은 dynamo가 계속 작동하지만 sunspot 형성 threshold 아래로 amplitude가 떨어짐. Maunder Minimum 중 cyclic activity가 다른 proxy에 남아 있다는 관측(Beer et al. 1998)과 호환.

**7.5 Grand minima in MHD simulations**: Augustson et al. 2015 K3S 시뮬레이션에서 ~5 반주기 지속, magnetic energy 50% 감소 Grand Minimum 자발 발생.

**7.6 Fossil fields and 22-year cycle**: 방사층의 fossil field가 있다면 Gnevyshev–Ohl rule을 직접 설명 가능. 단 Mursula et al. 2001에 따르면 1700–1800과 1850–1990에서 even/odd 패턴이 뒤집혀 fossil field 단일 시나리오 반박.

### Part VI+: Machine learning and data assimilation / 머신러닝과 데이터 동화

논문 본문에서 ML은 명시적으로 장(章)으로 다뤄지지 않지만, 저자는 Sect. 5.6과 Sect. 8.5에서 다음 방향을 시사한다:
- **Upton & Hathaway (2014b)**: SFT 데이터 동화 기반 cycle prediction. Precursor로서 surface dipole 사용.
- **Hung et al. (2017)**: meridional flow의 helioseismic 데이터를 BL dynamo에 assimilate.
- **Dikpati et al. (2016)**: coronal/interplanetary magnetic field reconstruction을 위한 3D BL 모델.
- **Lemerle et al. (2015)**: BMR 통계 feed-in (2×2D model의 기반). Genetic-algorithm optimization으로 parameter tuning (Nandy & Choudhuri 2002 아이디어의 자동화).

The review does not dedicate a chapter to machine learning, but Sects. 5.6 and 8.5 point to data-assimilation approaches: Upton & Hathaway (2014b) for SFT-based cycle prediction using surface-dipole precursor; Hung et al. (2017) for helioseismic meridional-flow assimilation into BL dynamos; Dikpati et al. (2016) for 3D BL-based coronal/IMF reconstruction; and Lemerle et al. (2015) for injecting observed BMR statistics with genetic-algorithm parameter optimization. These foreshadow a ML-heavy next decade.

### Part VII: Open questions (Sect. 8) / 열린 질문

1. **8.1 Primary T→P mechanism?**: α-effect vs. BL vs. instability — 세 가지 viable, consensus 없음. "We actually have too many viable T→P mechanisms!"
2. **8.2 What limits amplitude?**: α-quenching, Λ-quenching, buoyant flux loss, Malkus–Proctor 중 무엇이 dominant? 모름.
3. **8.3 How constraining is butterfly?**: Flux rope 형성·부력 등 많은 단계가 불확실해 butterfly와 직접 비교의 정확도 한계.
4. **8.4 Is tachocline crucial?**: MHD 시뮬레이션이 tachocline 없이도 solar-like cycle 생성 + fully convective stars의 X-ray activity가 partial convective stars와 같은 rotation–activity relation → **아닐 수 있음**.
5. **8.5 Is meridional circulation crucial?**: Flux-transport 모델의 핵심 가정이지만 multi-cell / variability issue. Hung et al. 2017처럼 data assimilation이 해결책.
6. **8.6 Is mean field really axisymmetric?**: Active longitudes, 회전-based 주기성, corona 형태 — 비축대칭 성분 존재. High-Rm strong differential rotation은 axisymmetrize하는 경향.
7. **8.7 What causes Maunder-type Grand Minima?**: 트리거 및 회복 메커니즘 모름. Cosmogenic phase persistence가 thresholded modulation vs. true intermittency 구별 단서.
8. **8.8 Where do we go from here?**: mean-field/mean-field-like 모델은 여전히 연구 주력(long timescale, SFT evolution, prediction, stellar activity); global MHD는 계속 정교화.

---

## 3. Key Takeaways / 핵심 시사점

1. **Cowling's theorem is the pivot of the whole review** — 축대칭 흐름은 축대칭 자기장을 지속 못함. 따라서 α-effect, BL, 불안정성 등 **비축대칭 성분**이 필수. / Cowling's theorem is the pivot: axisymmetric flows cannot sustain axisymmetric fields, so every dynamo model must introduce non-axisymmetric processes (α-effect, BL, instabilities).

2. **Too many viable T→P mechanisms** — 저자 본인의 표현. α-effect, Babcock–Leighton, tachocline instabilities, buoyant flux tube instabilities 모두 설명력 있음. 관측만으로 구분하기 어려움. / The author states "we have too many viable T→P mechanisms!" Observations cannot yet discriminate among α-effect, BL, tachocline instabilities, and buoyant flux-tube instabilities.

3. **Babcock–Leighton models have matured dramatically** — 2010년판 이후 10년간 가장 큰 변화. Kinematic 2D → 3D flux emergence → 2×2D coupled SFT+interior (Lemerle & Charbonneau 2017). 관측 BMR 통계를 직접 주입 가능. Surface dipole as precursor는 이제 cycle prediction의 표준. / The biggest change since 2010: BL models now span 2D kinematic → 3D emergence → 2×2D coupled models; the surface-dipole-as-precursor relation has become the standard prediction scheme.

4. **Global MHD simulations can produce solar-like cycles without any mean-field prescription** — ASH, EULAG-MHD, Pencil Code가 equatorward migration, polarity reversal, Grand-Minima-like 사건을 자연스럽게 생성. 이는 "tachocline is essential" 합의를 흔든다. / Global MHD simulations now spontaneously produce solar-like cycles including equatorward migration, polarity reversals, and Grand-Minima-like episodes, challenging the tachocline-essential view.

5. **Stochastic forcing is probably the dominant driver of cycle variability** — BL tilt scatter, turbulent EMF fluctuation, meridional flow noise 등 모든 경로가 multiplicative noise로 작용. Cameron & Schüssler 2017b의 Hopf normal form SDE가 관측 SSN 통계 대부분을 재현. / Stochastic forcing — via BL tilt scatter, turbulent EMF fluctuations, and meridional flow noise — is most likely the dominant driver of cycle variability; Cameron & Schüssler (2017b)'s stochastic Hopf normal form reproduces most observed SSN statistics.

6. **Grand Minima arise from multiple possible mechanisms** — on-off intermittency(criticality 근방 stochastic pushing), in-out intermittency(lower operating threshold, BL-like), thresholded modulation(cycle 계속되지만 sunspot 형성 실패), parity modulation. 관측 phase persistence가 구별 열쇠. / Grand Minima can arise from on–off intermittency (stochastic pushing near criticality), in–out intermittency (operating threshold, as in BL), thresholded modulation, or parity modulation. Phase persistence during minima discriminates among them.

7. **Meridional circulation sets the cycle period in advection-dominated models, but its reality is contested** — 단일 cell steady flow가 표준 가정이지만 helioseismic 역문제에서 multi-cell 주장 등장. Turbulent pumping이 대안 period setter. / Meridional circulation sets the period in advection-dominated BL models ($P\propto u_0^{-0.89}$), but helioseismic inversions suggesting multi-cell structure threaten this; turbulent pumping is an alternative period-setter.

8. **Solar cycle prediction and stellar dynamos will be the next frontier** — BL 기반 예측 scheme이 성공적이나 예측 window 짧음(1 cycle). 완전대류 별의 magnetic activity가 partial-CZ 별과 같은 rotation-activity 관계를 보임 → 통합 다이나모 이론 필요. / Future frontiers: BL-based cycle prediction (now successful for 1-cycle forecasts) and unified dynamo theory that explains why fully convective and partially convective stars share the rotation–activity relation.

---

## 4. Mathematical Summary / 수학적 요약

### (A) MHD induction and decomposition / MHD 유도와 분해

$$\boxed{\frac{\partial\mathbf{B}}{\partial t}=\nabla\times(\mathbf{u}\times\mathbf{B}-\eta\nabla\times\mathbf{B})}$$
- $\mathbf{u}$: flow, $\mathbf{B}$: magnetic field, $\eta=c^2/(4\pi\sigma_e)$ magnetic diffusivity.
- Magnetic Reynolds number $\mathrm{Rm}=uL/\eta$: 태양에서는 $\sim 10^{9-10}$.

축대칭 분해: $\mathbf{B}=\nabla\times(A\hat{\mathbf{e}}_\phi)+B\hat{\mathbf{e}}_\phi$, $\mathbf{u}=\mathbf{u}_p+\varpi\Omega\hat{\mathbf{e}}_\phi$.

### (B) αΩ mean-field equations (Sect. 4.2.7)

$$\boxed{\frac{\partial\langle A\rangle}{\partial t}=(\eta+\beta)\!\left(\nabla^2-\tfrac{1}{\varpi^2}\right)\!\langle A\rangle-\tfrac{\mathbf{u}_p}{\varpi}\cdot\nabla(\varpi\langle A\rangle)+\alpha\langle B\rangle}$$

$$\boxed{\frac{\partial\langle B\rangle}{\partial t}=(\eta+\beta)\!\left(\nabla^2-\tfrac{1}{\varpi^2}\right)\!\langle B\rangle+\varpi(\nabla\times(\langle A\rangle\hat{\mathbf{e}}_\phi))\cdot\nabla\Omega}$$

- $\eta$: microscopic, $\beta$: turbulent diffusivity, $\eta_T=\eta+\beta\simeq\beta$.
- α term: provides T→P source (paper Eq. 19–23: $\mathcal{E}=\alpha\langle\mathbf{B}\rangle-\beta\nabla\times\langle\mathbf{B}\rangle$).
- Ω term: provides P→T source (shearing of $A$).
- Dynamo number $D=C_\alpha C_\Omega=\dfrac{\alpha_0(\Delta\Omega)_0 R_\odot^3}{\eta_0^2}$.

### (C) Parker–Yoshimura sign rule (dynamo wave direction)

$$\boxed{\mathbf{s}=\alpha\nabla\Omega\times\hat{\mathbf{e}}_\phi}$$
- Solar Ω(r,θ): positive $\partial\Omega/\partial r$ in equatorial tachocline, negative in polar tachocline, strong latitudinal shear in CZ.
- Equatorward butterfly requires (in kinematic αΩ): **negative α in N-hemisphere at low latitudes**, OR dominant negative latitudinal shear.

### (D) Stochastic Hopf normal form (Cameron & Schüssler 2017b)

$$\boxed{\frac{dX}{dt}-(\beta+i\omega_0)X+(\gamma_r+i\gamma_i)|X|^2 X=\sigma X\,\frac{dW}{dt}}$$
- $X$: complex magnetic amplitude.
- Without noise: limit cycle of amplitude $\sqrt{\beta/\gamma_r}$ and angular frequency $\omega_0-\gamma_i\beta/\gamma_r$.
- Cubic term $\propto|X|^2X$: flux loss by buoyancy.
- Multiplicative noise $\sigma X\,dW$: reflects that EMF fluctuation scales with B.

### (E) Babcock–Leighton dipole per BMR

$$\boxed{\delta D=\frac{3d\cos\lambda}{4\pi R^2}\Phi\sin\alpha},\quad \sin\alpha\approx 0.5\sin\lambda$$
- $\Phi$: BMR unsigned flux.
- $d$: pole separation.
- $\lambda$: emergence latitude, $\alpha$: tilt angle.
- Joy's law with 15° scatter → stochastic source for cycle amplitude.

### (F) Algebraic α-quenching

$$\boxed{\alpha(\langle B\rangle)=\frac{\alpha_0}{1+(\langle B\rangle/B_\mathrm{eq})^2}},\quad B_\mathrm{eq}^2/8\pi=\tfrac{1}{2}\rho u'^2$$

### (G) Flux-transport dynamo period scalings

$$\boxed{P_\mathrm{advect}\simeq 56.8\,u_0^{-0.89}s_0^{-0.13}\eta_T^{0.22}\ [\text{years}]}\quad \text{(Dikpati \& Charbonneau 1999)}$$

$$\boxed{P_\mathrm{pump}\simeq 181.2\,u_0^{-0.12}\gamma_{r0}^{-0.51}\gamma_{\theta0}^{-0.05}\ [\text{years}]}\quad \text{(Guerrero \& de Gouveia Dal Pino 2008)}$$

### (H) Worked numerical scenario / 구체 수치 예시

**Example 1: Linear amplification by differential rotation (Sect. 3.1)**. Starting with $B_p=10$ G dipole and solar differential rotation $\Delta\Omega/\Omega\sim 0.1$ over $\tau=10$ years:
$$B_\phi\sim(\mathbf{B}_p\cdot\nabla\Omega)\tau\sim 10\,\text{G}\times(0.3\,\mu\text{rad/s})\times(3\times 10^8\,\text{s})\sim 10^3\,\text{G}$$
→ $\sim 1$ kG toroidal in 10 years ✓ (matches paper Eq. 14 estimate of ~1 kG).

**Example 2: BL dipole balance**. Per cycle:
- Emerged flux ~$10^{25}$ Mx.
- Polar cap flux ~$10^{22}$ Mx.
- Required T→P conversion efficiency $\sim 10^{-3}$ — readily achieved by Joy's-law tilt + surface dispersal (Cameron & Schüssler 2015).

**Example 3: Grand-Minima recurrence**. Cosmogenic ¹⁴C/¹⁰Be data: ~2–3 Grand Minima per millennium, inter-event waiting time ~exponential → memoryless stochastic process compatible with on–off intermittency model (Fig. 29b of paper).

**Example 4: Cycle period from flux-transport scaling**. Take $u_0=15$ m/s, $\eta_T=5\times 10^{11}$ cm²/s, $s_0=1$ (normalized). Using Eq. (51): $P\simeq 56.8\times 15^{-0.89}\times 1^{-0.13}\times(5\times 10^{11})^{0.22}\simeq$ ~11 years ✓.

**Example 5: α-quenching saturation amplitude**. 태양 CZ 하부에서 $\rho\sim 0.1$ g/cm³, $u'\sim 100$ m/s = $10^4$ cm/s. Equipartition:
$$B_\mathrm{eq}=\sqrt{4\pi\rho u'^2}\sim\sqrt{4\pi\times 0.1\times 10^8}\sim 10^4\ \text{G}=1\ \text{kG}$$
이는 diffuse toroidal field 추정치와 양립. Flux-tube 수준의 $10^5$ G는 buoyancy가 concentrate한 결과. / Equipartition $B_\mathrm{eq}\sim 10^4$ G (1 kG) matches diffuse toroidal estimates; $10^5$ G flux-tube concentrations arise from buoyant concentration.

**Example 6: Maunder Minimum duration**. Observed 1645–1705 = ~60 years = ~5.5 cycle lengths; Augustson et al. (2015) simulation shows ~5 half-cycles at 50% magnetic-energy reduction → qualitatively consistent duration scale from parity modulation ~Type I. / Observed Maunder duration (60 yr ~ 5.5 cycles) is qualitatively reproduced by Augustson et al. (2015)'s Type-I parity modulation event (~5 half-cycles).

**Example 7: Meridional flow speed estimate**. Observed surface flow ~10–20 m/s poleward. Required return flow at base of CZ for mass conservation at density contrast $\rho(R_\odot)/\rho(0.7 R_\odot)\sim 10^{-3}$ plus area factor $(R_\odot/0.7 R_\odot)^2\sim 2$ gives return flow speed $\sim u_\mathrm{surf}/(10^{-3}\times 2)\sim$ ~0.01 m/s — extremely slow, hence sensitivity to perturbations. / Mass-conservation estimate gives deep return flow ~1 cm/s, explaining its sensitivity to perturbations and helioseismic difficulty.

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
TIMELINE OF SOLAR DYNAMO THEORY / 태양 다이나모 이론의 연대기

1843 Schwabe                     11-yr cycle discovered
1908 Hale                        Magnetic nature of sunspots
1919 Larmor                      Hydromagnetic dynamo idea
1934 Cowling (antidynamo)        Axisymmetric flows can't sustain
1955 Parker                      α-effect, dynamo wave ⭐ founding paper
1961 Babcock                     Surface flux transport model
1964 Leighton                    BL mechanism formalized
1966 Steenbeck/Krause/Rädler     Mean-field electrodynamics theory
1975 Yoshimura / 1976 Stix       Parker–Yoshimura sign rule
1981 Parker                      Interface dynamo concept
1983 Gilman                      First 3D MHD dynamo (too Jupiter-like)
1985 Glatzmaier                  First anelastic 3D MHD
1991 Wang, Sheeley, Nash         Post-helioseismic BL revival
1995 Choudhuri et al.            Flux-transport BL dynamo
1999 Dikpati & Charbonneau       Modern kinematic BL + MC
2003 Ossendrijver LRSP review    Mean-field dynamo review
2005 Brandenburg & Subramanian   Astrophys. Rep. review
   ────────────────────────
►2010 Charbonneau LRSP review    PAPER #20 (prior version)
   ────────────────────────
2011 Racine et al.               EULAG-MHD solar-like cycles
2012 Käpylä et al.               Pencil Code equatorward migration
2013 Yeates & Muñoz-Jaramillo    3D BL with vortical upflow
2014 Miesch & Dikpati            3D BL with flux ring injection
2015 Augustson et al.            Grand Minimum in MHD K3S simulation
2015 Hathaway LRSP review        Solar cycle observations (Paper #15)
2017 Lemerle & Charbonneau       2×2D coupled SFT + interior BL
2017 Karak & Miesch              Long-term 3D BL simulation
2017 Cameron & Schüssler         Stochastic Hopf SDE
►2020 Charbonneau LRSP UPDATE    THIS PAPER
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **#20 Charbonneau (2010)** — *Dynamo Models of the Solar Cycle* | 이 논문의 2010년판. Sects. 1–4는 대체로 보존, Sects. 5–6 크게 확장. Paper #71의 direct predecessor. | **Direct predecessor.** 2010년판을 읽은 독자는 Sect. 5 (BL 확장), Sect. 6 (global MHD, 완전 신규), Sect. 7 (재구성)에 집중. |
| **#04 Sheeley (2005)** — *Surface Evolution of the Sun's Magnetic Field* | BL 메커니즘의 관측적 근거(surface flux transport)를 제공. Sect. 5.2의 SFT 논의가 직접 연결. | Observational foundation for BL mechanism; cited throughout Sect. 5.2. |
| **#15 Hathaway (2015)** — *The Solar Cycle* | Sunspot cycle 관측(butterfly, amplitude variability, Waldmeier·Gnevyshev–Ohl rules). Sect. 1.4 및 Sect. 7.1에서 명시적으로 참조. | Observational benchmarks the review targets. |
| **#18 Fan (2009)** — *Magnetic Fields in the Solar Convection Zone* | Flux tube rise, Ω-loop 형성, Joy's law 기원. Sect. 5.1에서 tilt quenching 근거. | Physics of rising flux tubes, foundation for BL source term. |
| **#38 Cheung & Isobe (2014)** — *Flux Emergence* | BMR emergence의 표면 물리. 2×2D 및 3D BL 모델의 경계조건. | Surface emergence — connects interior dynamo to SFT. |
| **#05 Gizon & Birch (2005)** — *Local Helioseismology* | Differential rotation Ω(r,θ) 및 meridional circulation 정밀 측정. Sect. 2.1 및 4.4의 입력. | Provides Ω(r,θ) and $u_p$ that kinematic models take as given. |
| **#33 Potgieter (2013)** — *Solar Modulation of Cosmic Rays* | Grand Minima·cosmogenic radioisotope 연결. Sect. 7.1 및 7.3. | ¹⁴C / ¹⁰Be reconstructions, evidence for past Grand Minima. |
| **#11 Haigh (2007)** — *Sun and Earth's Climate* | 장기 활동 변동의 기후학적 중요성. Sect. 7 서두의 동기 부여. | Motivates caring about amplitude fluctuations and Grand Minima. |

---

## 7. References / 참고문헌

Primary:
- Charbonneau, P., "Dynamo models of the solar cycle", *Living Reviews in Solar Physics*, **17**, 4 (2020). DOI: 10.1007/s41116-020-00025-6
- Charbonneau, P., "Dynamo models of the solar cycle", *Living Reviews in Solar Physics*, **7**, 3 (2010). (Paper #20 — the 2010 edition this paper revises.)

Key references cited in notes:
- Babcock, H.W. (1961) "The topology of the Sun's magnetic field and the 22-year cycle", *Astrophys. J.*, **133**, 572.
- Leighton, R.B. (1969) "A magneto-kinematic model of the solar cycle", *Astrophys. J.*, **156**, 1.
- Parker, E.N. (1955) "Hydromagnetic dynamo models", *Astrophys. J.*, **122**, 293.
- Yoshimura, H. (1975) *Astrophys. J.*, **201**, 740; Stix, M. (1976) *Astron. Astrophys.*, **47**, 243.
- Dikpati, M. & Charbonneau, P. (1999) *Astrophys. J.*, **518**, 508.
- Choudhuri, A.R., Schüssler, M. & Dikpati, M. (1995) *Astron. Astrophys.*, **303**, L29.
- Käpylä, P.J. et al. (2012) *Astrophys. J.*, **755**, L22.
- Racine, É. et al. (2011) *Astrophys. J.*, **735**, 46.
- Ghizaru, M. et al. (2010) *Astrophys. J. Lett.*, **715**, L133.
- Augustson, K., Brun, A.S., Miesch, M. & Toomre, J. (2015) *Astrophys. J.*, **809**, 149.
- Miesch, M.S. & Dikpati, M. (2014) *Astrophys. J. Lett.*, **785**, L8.
- Yeates, A.R. & Muñoz-Jaramillo, A. (2013b) *Mon. Not. R. Astron. Soc.*, **436**, 3366.
- Karak, B.B. & Miesch, M. (2017) *Astrophys. J.*, **847**, 69.
- Lemerle, A. & Charbonneau, P. (2017) *Astrophys. J.*, **834**, 133.
- Cameron, R. & Schüssler, M. (2017b) *Astron. Astrophys.*, **599**, A52.
- Olemskoy, S.V. & Kitchatinov, L.L. (2013) *Astrophys. J.*, **777**, 71.
- Pevtsov, A.A. et al. (2014) *Space Sci. Rev.*, **186**, 285.
- Hathaway, D.H. (2015) "The Solar Cycle", *Living Reviews in Solar Physics*, **12**, 4.
- Usoskin, I.G. (2017) "A history of solar activity over millennia", *Living Reviews in Solar Physics*, **14**, 3.
- Fan, Y. (2009) "Magnetic fields in the solar convection zone", *Living Reviews in Solar Physics*, **6**, 4.
- Cheung, M.C.M. & Isobe, H. (2014) "Flux emergence (theory)", *Living Reviews in Solar Physics*, **11**, 3.
- Sheeley, N.R. Jr. (2005) "Surface evolution of the Sun's magnetic field", *Living Reviews in Solar Physics*, **2**, 5.
- Ossendrijver, M. (2003) *Astron. Astrophys. Rev.*, **11**, 287.
- Brandenburg, A. & Subramanian, K. (2005) *Phys. Rep.*, **417**, 1.
