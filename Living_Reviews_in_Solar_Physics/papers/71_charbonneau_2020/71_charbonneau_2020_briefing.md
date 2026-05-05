---
title: "Pre-Reading Briefing: Dynamo Models of the Solar Cycle (2020 update)"
paper_id: "71"
topic: Living_Reviews_in_Solar_Physics
date: 2026-04-23
type: briefing
---

# Dynamo Models of the Solar Cycle (2020 update): Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Charbonneau, P., "Dynamo models of the solar cycle", *Living Reviews in Solar Physics*, 17:4 (2020). DOI: 10.1007/s41116-020-00025-6
**Author(s)**: Paul Charbonneau (Université de Montréal)
**Year**: 2020

> **중요 / Important**: 이 논문은 **Paper #20 Charbonneau (2010)**의 **major revision / update**입니다. 본 브리핑과 노트는 2010년판에서 무엇이 업데이트되었는지를 명시적으로 추적합니다.
>
> This paper is a **major revision** of **Paper #20 Charbonneau (2010)**. This briefing and notes explicitly track what has been updated relative to the 2010 edition.

---

## 1. 핵심 기여 / Core Contribution

이 리뷰는 태양 주기(solar cycle)를 유체역학적 다이나모(dynamo) 과정으로 모델링하는 최근 10년간의 발전과 현재 논쟁을 정리한다. 저자는 단순하면서도 관측과 비교 가능한 수준의 상세함을 지닌 **mean-field / mean-field-like 모델**에 초점을 맞추고, MHD induction equation의 기초에서 출발하여 (i) 다이나모 문제의 수학적 정식화, (ii) poloidal→toroidal(Ω효과)과 toroidal→poloidal(α효과, Babcock–Leighton 메커니즘, MHD 불안정성) 재생성 메커니즘, (iii) αΩ 모델·interface 다이나모·flux transport 다이나모·Babcock–Leighton 모델의 대표 예시, (iv) 전지구 MHD 수치 시뮬레이션, (v) 진폭 변동·혼돈·간헐성·Grand Minima를 포괄적으로 다룬다. 2010년판 대비 가장 큰 변화는 **Babcock–Leighton 모델이 독자적 섹션(Sect. 5)으로 확장**되었고, **global MHD 시뮬레이션이 대규모 자기 주기 생성 수준에 도달**해 새 섹션(Sect. 6)이 추가되었으며, **확률적 forcing·혼돈·Grand Minima 논의**가 일반 거동(generic behavior) 중심으로 재편된 점이다.

This review surveys the past decade of progress and current debates in modeling the solar cycle as a hydromagnetic dynamo. The author focuses on **(relatively) simple mean-field / mean-field-like models** that are nonetheless detailed enough to be compared against solar cycle observations. Starting from the MHD induction equation, it covers (i) the mathematical formulation of the dynamo problem, (ii) poloidal→toroidal (Ω-effect) and toroidal→poloidal (α-effect, Babcock–Leighton mechanism, MHD instabilities) regeneration mechanisms, (iii) representative αΩ, interface, flux-transport, and Babcock–Leighton models, (iv) global MHD numerical simulations, and (v) amplitude fluctuations, chaos, intermittency and Grand Minima. Relative to the 2010 edition, the main updates are: **Babcock–Leighton models now have their own dedicated section (Sect. 5)**; **global MHD simulations have matured enough to generate solar-like large-scale cycles and receive a new section (Sect. 6)**; and **the treatment of stochastic forcing, chaos and Grand Minima is reorganized around generic behaviors** with pointers to the technical literature.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

2010년판 이후 10년간 태양 다이나모 연구는 세 가지 흐름에서 크게 전진했다. (1) **관측적 제약 강화**: SDO/HMI, 지속적 helioseismology, cosmogenic radioisotope(¹⁴C, ¹⁰Be) 재구성 정밀화, Joy's law·tilt scatter의 통계적 특성화. (2) **전지구 MHD 시뮬레이션 성숙**: EULAG-MHD, ASH, Pencil Code 기반 시뮬레이션이 solar-like 극성 반전과 equatorward 나비 다이어그램을 재현하기 시작(Ghizaru et al. 2010; Käpylä et al. 2012; Racine et al. 2011; Augustson et al. 2015). (3) **Babcock–Leighton(BL) 모델 중심의 데이터 동화/예측**: BMR 방출 통계를 BL 모델에 직접 주입한 **2×2D 모델(Lemerle & Charbonneau 2017)** 및 3D BL 시뮬레이션(Miesch & Dikpati 2014; Karak & Miesch 2017)이 등장. 또한 stochastic tilt scatter가 주기 진폭 변동과 Grand Minima의 주된 원인일 수 있다는 증거가 누적되었다.

The decade after the 2010 edition saw three major advances. (1) **Tighter observational constraints**: SDO/HMI, sustained helioseismology, improved cosmogenic (¹⁴C, ¹⁰Be) reconstructions, and statistical characterization of Joy's law tilt and its scatter. (2) **Maturation of global MHD simulations**: EULAG-MHD, ASH, and Pencil Code runs now reproduce solar-like polarity reversals and equatorward butterflies (Ghizaru et al. 2010; Käpylä et al. 2012; Racine et al. 2011; Augustson et al. 2015). (3) **Data-driven Babcock–Leighton modeling**: injection of observed BMR statistics into BL models, e.g. the **2×2D model of Lemerle & Charbonneau (2017)** and 3D BL simulations (Miesch & Dikpati 2014; Karak & Miesch 2017). Evidence has also accumulated that stochastic tilt scatter may be the dominant driver of cycle amplitude fluctuations and Grand Minima.

### 타임라인 / Timeline

```
1843  Schwabe: 11-year sunspot cycle
1908  Hale: magnetic nature of sunspots
1919  Larmor: hydromagnetic dynamo idea
1934  Cowling's antidynamo theorem (axisymmetric flows cannot sustain)
1955  Parker: mean-field α-effect, dynamo wave
1961  Babcock: surface flux transport model
1964  Leighton: Babcock–Leighton mechanism formalized
1969  Steenbeck, Krause & Rädler: mean-field electrodynamics theory
1975  Yoshimura / Stix: Parker–Yoshimura sign rule in spherical geometry
1985  First 3D MHD convection dynamo (Gilman, Glatzmaier)
1990s Helioseismology pins down differential rotation → trouble for MFE
1999  Dikpati & Charbonneau: modern flux-transport BL model
2003  Ossendrijver review; Brandenburg & Subramanian 2005
2010  Charbonneau LRSP review (Paper #20)  ← previous version
2011  Racine et al.: EULAG-MHD solar-like cycles
2014  Augustson et al.: Grand Minimum in MHD simulation
2014  Miesch & Dikpati: 3D BL model with explicit BMR emergence
2017  Lemerle & Charbonneau: 2×2D coupled SFT + interior dynamo
2017  Karak & Miesch: long-term 3D BL simulation
2020  THIS PAPER — Charbonneau LRSP update
```

---

## 3. 필요한 배경 지식 / Prerequisites

- **MHD 기본방정식 / Basic MHD**: induction equation $\partial_t\mathbf{B}=\nabla\times(\mathbf{u}\times\mathbf{B}-\eta\nabla\times\mathbf{B})$, Navier–Stokes + Lorentz force. | Induction equation, Navier–Stokes, Lorentz.
- **벡터 해석 / Vector calculus**: curl·divergence identities, poloidal–toroidal decomposition in spherical coordinates. | Poloidal–toroidal decomposition in spherical coords.
- **태양 내부 구조 / Solar structure**: radiative core, tachocline (r/R☉≈0.7), convection zone, photosphere; heliosesmic differential rotation profile. | Radiative core / tachocline / CZ; helioseismic Ω(r,θ).
- **Mean-field electrodynamics**: fluctuation averaging, turbulent EMF $\mathcal{E}=\langle\mathbf{u}'\times\mathbf{B}'\rangle=\alpha\langle\mathbf{B}\rangle-\beta\nabla\times\langle\mathbf{B}\rangle$. | Turbulent EMF, α and β tensors.
- **Dynamical systems**: Hopf bifurcation, limit cycles, stochastic differential equations (Wiener process), on–off / in–out intermittency. | Hopf bifurcation, SDEs, intermittency.
- **Observational solar cycle facts**: butterfly diagram, Hale's polarity laws, Joy's law tilt, 22-yr magnetic cycle, Waldmeier & Gnevyshev–Ohl rules, Maunder Minimum. | Butterfly, Hale, Joy, Waldmeier, GO rules, Maunder.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **Dynamo number** $D=C_\alpha C_\Omega$ | 무차원 다이나모 효율, 임계값 $D_\mathrm{crit}$ 초과 시 주기적 해(limit cycle) 발현. / Dimensionless efficiency; cyclic solutions exist for $D>D_\mathrm{crit}$. |
| **α-effect** | 헬리컬 난류 혹은 BMR 기울기가 만드는 평균 EMF 성분; Cowling 정리 우회. / Mean EMF from helical turbulence / BMR tilt; circumvents Cowling. |
| **Ω-effect** | Differential rotation에 의한 poloidal→toroidal 전환; shearing term. / Poloidal→toroidal conversion by differential rotation shear. |
| **Babcock–Leighton (BL) mechanism** | 기울어진 BMR의 표면 분산이 global dipole을 반전·재생성. / Decay of tilted BMRs reverses/regenerates the global dipole. |
| **Flux-transport dynamo** | 자오면 순환(meridional circulation)이 conveyor belt로 source 영역을 연결. / Meridional circulation acts as a conveyor belt between source regions. |
| **Parker–Yoshimura sign rule** | 다이나모 파 전파 방향 $\mathbf{s}=\alpha\nabla\Omega\times\hat{\mathbf{e}}_\phi$. / Propagation direction of dynamo waves. |
| **Tachocline** | CZ 바닥의 강한 shear 층, $r/R_\odot\approx 0.7$, 반두께 $w\approx0.05R_\odot$. / Shear layer at base of CZ. |
| **α-quenching** | 자기장 강도에 따라 α가 감소하는 비선형, $\alpha(B)=\alpha_0/(1+(B/B_\mathrm{eq})^2)$. / Nonlinear α suppression. |
| **Joy's law** | Tilt angle $\sin\alpha\approx 0.5\sin\lambda$, BL T→P 효율을 설정. / Sets BL T→P efficiency. |
| **Grand Minima** | Maunder·Spörer·Wolf 등 장기 활동 억제 구간. / Extended epochs of suppressed activity. |
| **On–off / In–out intermittency** | 임계 근방 stochastic/deterministic forcing에 의한 Grand Minima 기구. / Intermittency routes to Grand Minima. |
| **Magnetic Prandtl number** $\mathrm{Pm}=\nu/\eta$ | 점성 대 자기 확산의 비; Malkus–Proctor 기구 시간척도 결정. / Sets Malkus–Proctor timescale. |

---

## 5. 수식 미리보기 / Equations Preview

**(E1) MHD 유도 방정식 / MHD induction equation**
$$\frac{\partial \mathbf{B}}{\partial t}=\nabla\times(\mathbf{u}\times\mathbf{B}-\eta\nabla\times\mathbf{B})$$
축대칭 분해 $\mathbf{B}=\nabla\times(A\hat{\mathbf{e}}_\phi)+B\hat{\mathbf{e}}_\phi$와 $\mathbf{u}=\mathbf{u}_p+\varpi\Omega\hat{\mathbf{e}}_\phi$ 대입 시 A, B에 대한 두 결합 PDE가 나온다.
Inserting the axisymmetric split yields two coupled PDEs for the vector potential A and toroidal field B.

**(E2) αΩ mean-field 다이나모 방정식 / αΩ mean-field dynamo equations**
$$\frac{\partial\langle A\rangle}{\partial t}=(\eta+\beta)\!\left(\nabla^2-\tfrac{1}{\varpi^2}\right)\!\langle A\rangle-\tfrac{\mathbf{u}_p}{\varpi}\cdot\nabla(\varpi\langle A\rangle)+\alpha\langle B\rangle$$
$$\frac{\partial\langle B\rangle}{\partial t}=(\eta+\beta)\!\left(\nabla^2-\tfrac{1}{\varpi^2}\right)\!\langle B\rangle+\varpi\big(\nabla\times(\langle A\rangle\hat{\mathbf{e}}_\phi)\big)\cdot\nabla\Omega+\cdots$$
α항이 Cowling 정리를 우회하는 P↔T 결합을 제공. | The α-term provides the poloidal source that bypasses Cowling's theorem.

**(E3) Parker–Yoshimura sign rule**
$$\mathbf{s}=\alpha\nabla\Omega\times\hat{\mathbf{e}}_\phi$$
다이나모 파의 진행 방향; 북반구 양의 α와 양의 $\partial\Omega/\partial r$ 조합은 poleward 전파. 태양 butterfly의 equatorward drift를 재현하려면 북반구에서 **negative α** 또는 **negative radial shear**가 필요하다. / Direction of dynamo-wave propagation; equatorward butterfly requires negative α in N-hemisphere or negative radial shear.

**(E4) Babcock–Leighton dipole contribution per BMR**
$$\delta D=\frac{3d\cos\lambda}{4\pi R^2}\Phi\sin\alpha,\qquad \sin\alpha\approx 0.5\sin\lambda\ \text{(Joy)}$$
BMR 한 개가 글로벌 축대칭 쌍극자에 기여하는 양; 주기 평균 ~$10^{25}$ Mx 방출 중 ~$10^{22}$ Mx가 극관에 도달. / Contribution of one BMR to global axisymmetric dipole; of ~$10^{25}$ Mx emerged per cycle, ~$10^{22}$ Mx reach the poles.

**(E5) Algebraic α-quenching (amplitude saturation)**
$$\alpha(\langle\mathbf{B}\rangle)=\frac{\alpha_0}{1+(\langle\mathbf{B}\rangle/B_\mathrm{eq})^2}$$
Equipartition 근방에서 α를 꺾어 다이나모 포화. | Saturates dynamo at $B\sim B_\mathrm{eq}$.

**(E6) Stochastic forcing model (Cameron & Schüssler 2017b)**
$$\frac{dX}{dt}-(\beta+i\omega_0)X+(\gamma_r+i\gamma_i)|X|^2X=\sigma X\frac{dW}{dt}$$
Multiplicative 확률 forcing을 갖는 Hopf 정규형; 관측된 SSN 분포와 Gnevyshev–Ohl/Waldmeier-like 패턴을 재현. / Stochastic Hopf normal form reproducing observed SSN statistics.

---

## 6. 읽기 가이드 / Reading Guide

1. **먼저 Sect. 1–3을 빠르게**: 이미 2010판을 읽었다면 scope·dynamo problem·Cowling·α-effect·BL 메커니즘은 복습 수준. 변경이 크지 않다. | Skim Sect. 1–3 if you have read the 2010 edition: scope, dynamo problem, Cowling, α-effect, BL — little changed.
2. **Sect. 4 (mean-field models)는 선택적으로**: 4.2.7 αΩ 방정식과 4.2.9 Parker–Yoshimura sign rule, 4.4 flux-transport 항목만 엄밀히 읽어도 충분. | Focus on 4.2.7, 4.2.9, 4.4.
3. **Sect. 5 (Babcock–Leighton)는 핵심 업데이트**: 특히 5.5 non-axisymmetric 3D / 2×2D 모델(Miesch & Dikpati 2014, Yeates & Muñoz-Jaramillo 2013b, Karak & Miesch 2017, Lemerle & Charbonneau 2017)을 꼼꼼히. | **Must-read update**: Sect. 5, especially 5.5 on 3D / 2×2D models.
4. **Sect. 6 (global MHD)는 완전 신규**: simulation 코드별 성공·실패(ASH, EULAG-MHD, Pencil), equatorward 분기, tachocline 없는 다이나모, Grand-Minima-like 사건(Augustson et al. 2015)을 주목. | **Brand-new section**: global MHD; note Augustson Grand Minimum event.
5. **Sect. 7 (fluctuations, Grand Minima)은 개념 중심**: Hopf bifurcation → stochastic forcing → nonlinear modulation → time delays → conveyor-belt rattling → on–off/in–out intermittency 순서로 이해. | Conceptual tour: Hopf → stochastic → nonlinear → time delays → conveyor-belt → intermittency.
6. **Sect. 8 (open questions)**: 8개 질문을 점검 목록으로 활용. "primary T→P mechanism?", "what limits amplitude?", "is tachocline crucial?", "what causes Maunder Minima?" 등. | Treat the 8 questions as a checklist.

---

## 7. 현대적 의의 / Modern Significance

이 리뷰는 태양 주기 **예측**(Solar Cycle 25 forecast 논쟁), 우주날씨(space weather) 장기 전망, 항성 자기활동(stellar activity) 해석, 지구 기후(Sun–climate, Maunder Minimum)와 직결된 레퍼런스 포인트다. 특히 BL 메커니즘이 주기 예측 정확도의 증가 배경이라는 점, 글로벌 MHD가 solar-like 주기를 자연스럽게 만들 수 있다는 점은 2010년판 이후 가장 큰 관점 전환이다. 또한 "tachocline이 dynamo 작동에 essential한가?"라는 질문이 열리면서, 완전대류 별(fully convective stars)의 자기활동 scaling까지 포함하는 통합적 다이나모 이론의 필요성이 대두되었다. 본 논문은 향후 10년의 표준 참조 문헌이 될 것이다.

This review is a reference point for **solar cycle prediction** (Cycle 25 forecast debate), long-term space-weather outlook, stellar magnetic activity, and Sun–climate research. The two biggest perspective shifts since 2010 are: (i) BL-based schemes now underpin the most successful cycle prediction schemes, and (ii) global MHD simulations can spontaneously generate solar-like cycles, reopening the question of whether a tachocline is *essential* for dynamo action — a question whose answer must also encompass fully convective stars. This paper will be the standard reference for the next decade.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
