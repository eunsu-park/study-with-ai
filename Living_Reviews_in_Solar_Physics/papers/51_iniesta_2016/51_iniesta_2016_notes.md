---
title: "Inversion of the Radiative Transfer Equation for Polarized Light"
authors: [Jose Carlos del Toro Iniesta, Basilio Ruiz Cobo]
year: 2016
journal: "Living Reviews in Solar Physics, 13:4"
doi: "10.1007/s41116-016-0005-2"
topic: Living_Reviews_in_Solar_Physics
tags: [spectropolarimetry, inversion, radiative_transfer, Stokes_profiles, Milne_Eddington, SIR, response_functions, Levenberg_Marquardt]
status: completed
date_started: 2026-04-23
date_completed: 2026-04-23
---

# 51. Inversion of the Radiative Transfer Equation for Polarized Light / 편광된 빛에 대한 복사전달방정식의 역산

---

## 1. Core Contribution / 핵심 기여

This review is the definitive modern reference for the inversion of the polarized radiative transfer equation (RTE) in solar physics. The authors — both pioneers and practitioners of the principal inversion codes (SIR, MILOS) — provide a critical, 84-page exposition that (1) casts the forward problem of radiative transfer as a nonlinear mapping from atmospheric physical quantities to observable Stokes profiles, (2) exposes the ill-conditioned nature of the inverse mapping, and (3) surveys every major inversion technique in use: Milne-Eddington codes (HAO-ASP, VFISV, HELIX, MILOS), depth-varying codes (SIR, SPINOR, NICOLE, LILIA), database/PCA codes (FATIMA, CSIRO-Meudon), ANN, genetic algorithm, Bayesian, and the modern spatially-coupled, regularized-deconvolution, and sparse inversions. Along the way, the review tabulates the mathematical apparatus — the propagation matrix K, source vector S, the Unno-Rachkovsky analytic solution, response functions, χ² merit function, and the Levenberg-Marquardt normal equations — that underlies practically all of these codes.

이 리뷰는 태양물리학에서 편광된 복사전달방정식(RTE)의 역산에 관한 결정적인 현대 참고 자료이다. SIR와 MILOS 등 주요 inversion code의 선구자이자 실용가인 저자들은 84페이지에 걸쳐 비판적 서술을 제공하여, (1) 복사전달의 forward problem을 대기의 물리량에서 관측 가능한 Stokes 프로파일로의 비선형 사상으로 기술하고, (2) 역(inverse) 사상의 ill-conditioned 성격을 드러내며, (3) 사용 중인 모든 주요 inversion 기법을 개괄한다: Milne-Eddington 코드 (HAO-ASP, VFISV, HELIX, MILOS), 깊이 의존 코드 (SIR, SPINOR, NICOLE, LILIA), 데이터베이스/PCA 코드 (FATIMA, CSIRO-Meudon), ANN, 유전 알고리즘, Bayesian, 그리고 현대의 공간 결합(spatially-coupled), 정규화 deconvolution, sparse inversions. 이 과정에서 리뷰는 거의 모든 inversion code의 기초가 되는 수학적 장치—propagation matrix K, source vector S, Unno-Rachkovsky 해석해, response functions, χ² merit function, Levenberg-Marquardt 정규방정식—를 체계적으로 정리한다.

The overarching message is pragmatic: inversion is not a black box. Every retrieval depends on the assumptions (LTE/NLTE, depth-dependence, weak-field, MISMA, filling factor, node count, damping, initialization) that went into the code. The authors advocate an Occam-razor, step-by-step strategy where atmospheric complexity is increased only if the noise-limited residuals demand it, and they push back (with numerical evidence) against several widely-held misconceptions — notably the belief that Stokes V is proportional to the longitudinal field even in the weak regime, and the belief that visible Fe I 630 nm lines are useless for internetwork-field inference.

전체 메시지는 실용적이다: inversion은 블랙박스가 아니다. 모든 retrieval은 그 코드에 들어간 가정(LTE/NLTE, 깊이 의존성, weak-field, MISMA, filling factor, node 개수, damping, 초기화)에 달려 있다. 저자들은 대기 복잡도를 관측 noise로 정한 기준 이상으로는 올리지 말라는 Occam-razor적, 단계적 전략을 주창하며, 널리 받아들여진 여러 통념—특히 약한 자기장 영역에서도 Stokes V가 longitudinal field에 비례한다는 믿음과 가시광 Fe I 630 nm 선이 internetwork 필드 진단에 쓸모없다는 믿음—을 수치적 증거로 반박한다.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction / Section 1 — Astrophysics as inference / 서론: 추론으로서의 천체물리학

**EN.** The authors open with an epistemological claim: astrophysics, unlike laboratory physics, cannot run controlled experiments; its "third pillar" is always indirect inference from remote observables. The observer measures signal (Stokes parameters, here) plus noise, and maps it into physical quantities (T, B, γ, φ, v_LOS, etc.) via an assumed physics. Fig. 1 drives the point home: two Stokes V profiles that look identical can come from entirely different atmospheres (one cool, 2000 K below and with 800 G field; another 305 K warmer, 530 G field) — a direct illustration of non-uniqueness. The physics linking observables to quantities is radiative transfer, and the relevant mapping is the Fredholm first-kind integral equation I(0) = ∫₀^∞ O(0,τ) K(τ) S(τ) dτ (Landi Degl'Innocenti & Landi Degl'Innocenti 1985). "Inversion" means: automatically, reliably extract physical quantities from Stokes spectra by minimizing a distance in observable space.

**KR.** 저자는 인식론적 주장으로 시작한다: 천체물리학은 실험실 물리학과 달리 통제된 실험을 할 수 없으므로, "세 번째 기둥"은 언제나 원격 관측 가능량으로부터의 간접 추론이다. 관측자는 신호(여기서는 Stokes 파라미터)와 noise를 측정해 가정된 물리를 통해 물리량(T, B, γ, φ, v_LOS 등)으로 사상한다. Fig. 1은 이를 직관적으로 보여준다: 외형이 동일한 두 Stokes V 프로파일이 전혀 다른 대기에서 나올 수 있다(한 쪽은 2000 K 차가운 800 G 대기, 다른 쪽은 305 K 더 따뜻한 530 G 대기) — non-uniqueness의 직접적 예시. 관측 가능량과 물리량을 잇는 물리는 복사전달이며, 관련된 사상은 Fredholm 제1종 적분방정식 I(0) = ∫₀^∞ O(0,τ) K(τ) S(τ) dτ (Landi Degl'Innocenti & Landi Degl'Innocenti 1985)이다. "Inversion"이란 관측가능공간의 거리를 최소화하여 Stokes 스펙트럼으로부터 물리량을 자동적으로, 신뢰성 있게 추출하는 과정이다.

### Part II: Radiative transfer assumptions / Section 2 — RTE approximations / RTE 근사들

**EN.** The RTE for polarized light, using the Stokes pseudo-vector **I** = (I, Q, U, V)ᵀ, reads

$$\frac{d\mathbf{I}}{d\tau_c} = \mathbf{K}(\mathbf{I} - \mathbf{S}) \quad (\text{Eq.6})$$

where τ_c is continuum optical depth, K is the 4×4 propagation matrix (containing absorption η_I, pleochroism η_Q,U,V, and dispersion ρ_Q,U,V — the magneto-optical effects), and **S** is the source-function vector. The formal integral solution (Eq. 8) involves the evolution operator O(0, τ), which is in general not known analytically. Section 2.1 sketches the non-LTE problem: K and S depend on ρ_α(jm, j'm') — density matrix elements satisfying statistical equilibrium Eqs. (4)-(5) that couple to the radiation field, making the whole system non-local. Section 2.2 introduces LTE, where coherences vanish (populations are Boltzmann/Saha) and **S** = (B_ν(T), 0, 0, 0)ᵀ — a dramatic simplification visualized in Fig. 3. Section 2.3 introduces the Milne-Eddington (ME) approximation: **S** = (S₀ + S₁ τ_c) **e**₀ with constant K, giving the celebrated Unno-Rachkovsky analytic solution

$$\mathbf{I}(0) = (S_0 + \mathbf{K}^{-1} S_1)\mathbf{e}_0. \quad (\text{Eq.14})$$

Section 2.4 introduces the **weak-field approximation**, valid when g_eff · ΔλB / ΔλD ≪ 1 (Eq. 17), under which Stokes V becomes proportional to the LOS-projected field and the non-magnetic intensity derivative (Eq. 24, to be unpacked below). Section 2.5 discusses the MISMA (MIcro-Structured Magnetic Atmosphere) hypothesis of Sánchez Almeida et al. (1996): the atmosphere is stochastic on sub-photon-mean-free-path scales, which statistically reproduces the ubiquitous Stokes profile asymmetries.

**KR.** Stokes pseudo-vector **I** = (I, Q, U, V)ᵀ를 사용한 편광된 빛의 RTE는

$$\frac{d\mathbf{I}}{d\tau_c} = \mathbf{K}(\mathbf{I} - \mathbf{S}) \quad (\text{식 6})$$

이다. τ_c는 연속체 광학적 깊이, K는 흡수(η_I), pleochroism(편광 상태별 차별적 흡수 η_Q,U,V), dispersion(magneto-optical 효과 ρ_Q,U,V)을 담은 4×4 propagation matrix, **S**는 source-function 벡터이다. 형식적 적분해(식 8)는 evolution operator O(0, τ)를 포함하는데 일반적으로 해석해가 없다. 2.1절은 non-LTE 문제를 다룬다: K와 S가 통계 평형 방정식(식 4-5)을 만족하는 밀도행렬 원소 ρ_α(jm, j'm')에 의존하며, 이것이 복사장과 결합해 non-local 시스템이 된다. 2.2절은 LTE를 도입한다: coherence가 사라지고(population이 Boltzmann/Saha) **S** = (B_ν(T), 0, 0, 0)ᵀ로 극적으로 단순화된다 (Fig. 3). 2.3절은 Milne-Eddington (ME) 근사를 도입한다: **S** = (S₀ + S₁ τ_c) **e**₀이고 K는 상수일 때 유명한 Unno-Rachkovsky 해석해

$$\mathbf{I}(0) = (S_0 + \mathbf{K}^{-1} S_1)\mathbf{e}_0 \quad (\text{식 14})$$

가 성립한다. 2.4절은 **약장 근사(weak-field)**를 도입한다: g_eff · ΔλB / ΔλD ≪ 1 (식 17)일 때 Stokes V는 시선 방향 자기장 성분과 non-magnetic intensity의 파장 미분의 곱에 비례한다 (식 24). 2.5절은 Sánchez Almeida et al. (1996)의 MISMA 가설을 논의한다: 대기가 광자 평균자유행정 이하 스케일의 stochastic 구조를 가진다는 가설로, 통계적으로 Stokes profile의 asymmetry를 재현한다.

### Part III: Degrees of approximation in model atmospheres / Section 3 — 대기 모델의 근사 단계

**EN.** Section 3.1 considers atmospheres with constant physical quantities. §3.1.1 re-derives the 9-parameter ME atmosphere (B, γ, φ, v_LOS, η₀ = χ_line/χ_cont, ΔλD, damping a, S₀, S₁). Fig. 5 shows example Fe I 617.3 nm ME Stokes profiles for B = 1200 G (strong Zeeman splitting visible in all four Stokes) and B = 200 G (small Q, U). §3.1.2 examines the weak-field atmosphere: V ≃ −g_eff ΔλB cos γ ∂I_nm/∂λ (Eq. 24), the magnetographic equation. Fig. 6 quantifies its breakdown: fitting V_peak as V = CB for strengths up to 600 G with a 6 pm instrumental FWHM yields clear nonlinearities above ~300 G (fields saturate the approximation); the weak-field approximation already fails by > 3σ in typical polarimetry for B > 200-400 G. The Q ∝ B² sin²γ relation (right panel of Fig. 6) provides *independent* information on field strength even in the "weak" regime. §3.1.3 allows constant B, v_LOS but with realistic T(τ). Section 3.2 deals with physical quantities varying with depth: Mickey & Orrall (1974) discovered net circular polarization (NCP) in sunspots, which Auer & Heasley (1978) proved requires LOS-velocity gradients. §3.2.1 introduces the *node* parameterization used by SIR: physical quantities at a small number of τ-nodes (e.g., 2 for linear, 3 for parabolic stratifications) are the free parameters, interpolated by splines. §3.2.2 critically discusses MISMA's practical difficulties: too many free parameters per realization. §3.2.3 covers special-case atmospheres: *interlaced* (thin flux tubes piercing a background, with the Eq. 25 solution involving alternating evolution operators O_(−1)^(n−j)), *Gaussian-profile* penumbral models, and *jump-discontinuity* atmospheres (SIRJUMP).

**KR.** 3.1절은 물리량이 상수인 대기를 다룬다. §3.1.1에서 9-parameter ME 대기(B, γ, φ, v_LOS, η₀ = χ_line/χ_cont, ΔλD, damping a, S₀, S₁)를 재유도한다. Fig. 5는 Fe I 617.3 nm의 ME Stokes 프로파일 예를 보여주는데, B = 1200 G에서는 네 Stokes 모두 Zeeman splitting이 뚜렷하고, B = 200 G에서는 Q, U가 작다. §3.1.2는 weak-field 대기: V ≃ −g_eff ΔλB cos γ ∂I_nm/∂λ (식 24), magnetographic equation이다. Fig. 6은 이 근사의 파탄을 정량화한다: 기기 FWHM 6 pm로 V_peak = CB의 선형 맞춤을 600 G까지 할 때 ~300 G 이상에서 비선형성이 나타나고, B > 200-400 G에서는 이미 3σ 이상으로 깨진다. Q ∝ B² sin²γ 관계는(Fig. 6 오른쪽) "약한" 영역에서도 자기장 강도에 대한 *독립적* 정보를 제공한다. §3.1.3에서는 B, v_LOS는 상수지만 현실적 T(τ)를 가진 대기를 다룬다. 3.2절은 깊이에 따라 변하는 물리량을 다룬다: Mickey & Orrall (1974)은 sunspot에서 순환 편광의 net 성분(NCP)을 발견했고, Auer & Heasley (1978)는 이것이 LOS velocity gradient의 필요충분조건임을 증명했다. §3.2.1은 SIR의 *node* parameterization을 도입한다: 소수의 τ-node에서의 물리량(예: 선형은 2, parabolic은 3)을 자유 파라미터로 하고 spline으로 보간한다. §3.2.2는 MISMA의 실무적 어려움—realization당 free parameter가 너무 많음—을 비판한다. §3.2.3은 특수 대기들: *interlaced* (배경을 찌르는 thin flux tube, 식 25의 교대 evolution operator O_(−1)^(n−j)), *Gaussian 프로파일* penumbra 모델, *jump-discontinuity* 대기(SIRJUMP)를 다룬다.

### Part IV: Degrees of approximation in Stokes profiles / Section 4 — Stokes 프로파일의 근사

**EN.** Section 4 decomposes I_d ≡ 1 − I/I_c, Q, U, V into even and odd parts (Eqs. 27-28): S(x) = S₊(x) + S₋(x) with S_± = (S(x) ± S(−x))/2. Regular (no-velocity-gradient) Stokes I, Q, U are even while V is odd. Asymmetries (S_− for I, Q, U; S_+ for V) are diagnostics of velocity gradients. Fig. 8 shows Δ-profiles between two model atmospheres differing only in v_LOS gradient at the 10⁻⁴ I_c level — below typical "nominal" noise but detectable with modern instruments. Section 4 also explains PCA/eigenprofile decomposition (Rees et al. 2000, Fig. 9) and Hermite-function expansions (del Toro Iniesta & López Ariste 2003): Stokes profiles as elements of L² (square-integrable Hilbert space) admit orthonormal basis expansions that serve as compression or database lookup.

**KR.** 4절은 I_d ≡ 1 − I/I_c, Q, U, V를 짝/홀 함수로 분해한다(식 27-28): S(x) = S₊(x) + S₋(x), S_± = (S(x) ± S(−x))/2. velocity gradient가 없는 regular Stokes에서는 I, Q, U는 짝, V는 홀이다. Asymmetry(I, Q, U에서 S_−, V에서 S_+)는 velocity gradient의 진단 지표이다. Fig. 8은 v_LOS gradient만 다른 두 모델 대기의 Δ-profile을 10⁻⁴ I_c 수준에서 보여준다 — 일반적 "nominal" noise 아래이지만 현대 기기로는 검출 가능하다. 4절은 PCA/eigenprofile 분해(Rees et al. 2000, Fig. 9)와 Hermite-function 확장(del Toro Iniesta & López Ariste 2003)도 설명한다: Stokes 프로파일을 L² (square-integrable Hilbert 공간)의 원소로 보고 정규직교 기저로 확장하여 압축이나 데이터베이스 검색에 이용한다.

### Part V: Synthesis approach / Section 5 — 합성 접근

**EN.** Section 5 addresses the forward problem: given atmosphere, compute Stokes profiles. Section 5.1 revisits the ME constant-atmosphere synthesis, 5.2 tackles depth-dependent atmospheres (SIR, SPINOR, NICOLE), and 5.3 reviews Stokes synthesis from MHD simulations (Rempel 2012 sunspot cuts, Fig. 15). The takeaway is that forward synthesis is considered solved in practice; the non-trivial problem is identifying the inverse mapping (Section 7).

**KR.** 5절은 forward problem을 다룬다: 대기가 주어졌을 때 Stokes 프로파일을 계산. 5.1절은 ME 상수 대기 합성, 5.2절은 깊이 의존 대기(SIR, SPINOR, NICOLE), 5.3절은 MHD 시뮬레이션으로부터의 Stokes 합성(Rempel 2012 sunspot cut, Fig. 15)을 리뷰한다. 요점은 forward synthesis는 실무적으로 해결된 문제이며, 어려운 문제는 역사상(Section 7)의 식별이라는 것이다.

### Part VI: Response functions / Section 6 — 반응 함수

**EN.** **Core object.** A response function (RF) R_i(τ) is the functional derivative of the emergent Stokes vector with respect to a physical quantity x_i at optical depth τ. Formally (Eq. 29):

$$\delta \mathbf{I}(0) = \sum_{i=1}^{p+r} \int_0^{\infty} \mathbf{R}_i(\tau_c) \delta x_i(\tau_c) \, d\tau_c,$$

with (Eq. 30)

$$\mathbf{R}_i(\tau_c) \equiv \mathbf{O}(0,\tau_c) \left[\mathbf{K}(\tau_c) \frac{\partial \mathbf{S}}{\partial x_i} - \frac{\partial \mathbf{K}}{\partial x_i} (\mathbf{I} - \mathbf{S})\right].$$

**Physical meaning.** The two terms on the right are "sources" (∂S/∂x_i) and "sinks" (∂K/∂x_i) that balance to produce the net change in the emergent profile. RFs play the role of a Point-Spread-Function (PSF) in linear-system theory: the Stokes spectrum is (to first order) a linear functional of atmospheric perturbations with kernel R_i. RF signs matter — they can be positive or negative, and the automatic node-selection algorithm in SIR finds nodes at the zero-crossings of R_i(τ).

**Graphical examples.** Figs. 16-19 show RFs for Fe I 630.25 nm, computed in two model atmospheres (HSRA with B=2000 G, γ=30°, φ=60°, v_LOS=0 in model 1, and a perturbed model 2). Key features: RF of Stokes I to T (top of Fig. 16) is large around line center and log τ_c ≈ −1 to 0 (formation depth); RF of V to B is strongly peaked at the blue and red Zeeman components and decays above log τ_c ≈ 0; RF to v_LOS is odd in wavelength, peaking at line wings.

**Analytic RFs in ME.** Under ME, integrals disappear and RF reduces to a partial derivative (Eq. 34): R_i(λ) = ∂I(λ)/∂x_i. Fig. 20 shows RFs of I and V to v_LOS, B, and the filling factor α: RF_V^B and RF_V^α are nearly proportional, explaining why α and B are hard to separate from V alone — but Stokes I RFs to α and B differ, so including I breaks the degeneracy.

**EN Practical importance.** When forward-synthesizing Stokes spectra, the evolution operator O, propagation matrix K, and source S must all be computed anyway — deriving RFs alongside adds negligible extra cost (Ruiz Cobo & del Toro Iniesta 1992). This makes Levenberg-Marquardt with RF-provided Jacobians extremely efficient.

**KR.** **핵심 개체.** Response function(RF) R_i(τ)는 광학 깊이 τ에서 물리량 x_i에 대한 emergent Stokes 벡터의 범함수 미분이다. 형식적으로 (식 29):

$$\delta \mathbf{I}(0) = \sum_{i=1}^{p+r} \int_0^{\infty} \mathbf{R}_i(\tau_c) \delta x_i(\tau_c) \, d\tau_c,$$

여기서 (식 30)

$$\mathbf{R}_i(\tau_c) \equiv \mathbf{O}(0,\tau_c) \left[\mathbf{K}(\tau_c) \frac{\partial \mathbf{S}}{\partial x_i} - \frac{\partial \mathbf{K}}{\partial x_i} (\mathbf{I} - \mathbf{S})\right].$$

**물리적 의미.** 우변의 두 항은 "원천"(∂S/∂x_i)과 "흡수/소멸"(∂K/∂x_i)로, 이들의 균형이 emergent profile의 순변화를 만든다. RF는 선형 시스템 이론에서 Point-Spread-Function(PSF) 역할을 한다: Stokes 스펙트럼은 (1차 근사에서) 대기 섭동의 선형 범함수이며 커널은 R_i이다. RF의 부호는 중요하다 — 양/음이 될 수 있으며, SIR의 자동 node 선택 알고리즘은 R_i(τ)의 영점에서 node를 찾는다.

**그래픽 예시.** Fig. 16-19는 두 모델 대기(모델 1: HSRA, B=2000 G, γ=30°, φ=60°, v_LOS=0; 모델 2: 섭동 대기)에서 계산된 Fe I 630.25 nm의 RF를 보여준다. 핵심 특징: Stokes I의 T에 대한 RF(Fig. 16 위)는 line center 부근과 log τ_c ≈ −1 ~ 0(형성 깊이)에서 크다; V의 B에 대한 RF는 청/적 Zeeman 성분에서 강하게 피크를 이루며 log τ_c ≈ 0 이상에서 감쇠한다; v_LOS에 대한 RF는 파장에 대해 홀이며 line wing에서 피크.

**ME에서의 해석적 RF.** ME에서는 적분이 사라지고 RF는 단순 편미분이 된다(식 34): R_i(λ) = ∂I(λ)/∂x_i. Fig. 20은 I와 V의 v_LOS, B, filling factor α에 대한 RF를 보여준다: RF_V^B와 RF_V^α가 거의 비례하여 V만으로는 α와 B 분리가 어렵지만, I의 RF는 α와 B에 대해 다르므로 I를 포함하면 degeneracy가 깨진다.

**실무적 중요성.** Stokes 스펙트럼을 forward 합성할 때 어차피 evolution operator O, propagation matrix K, source S를 모두 계산하므로, 동시에 RF를 유도하는 데 드는 추가 비용은 미미하다(Ruiz Cobo & del Toro Iniesta 1992). 덕분에 Levenberg-Marquardt가 RF로 얻은 Jacobian을 사용할 때 극도로 효율적이다.

### Part VII: Inversion techniques / Section 7 — 역산 기법

**EN.** **χ² merit function (Eq. 35).**

$$\chi^2(\mathbf{x}) = \frac{1}{N_f} \sum_{s=0}^{3} \sum_{i=1}^{q} [I_s^{\text{obs}}(\lambda_i) - I_s^{\text{syn}}(\lambda_i; \mathbf{x})]^2 w_{s,i}^2$$

where the sum runs over four Stokes parameters and q wavelength samples, N_f = 4q − (np + r) is the number of degrees of freedom, and w_{s,i} are weights. The gradient and Hessian of χ² are expressible directly via RFs (Eqs. 36, 37): ∂χ²/∂x_m = (2/N_f) Σ [...] R_m, and ∂²χ²/∂x_m ∂x_k ≈ (2/N_f) Σ R_m R_k. Thus, computing RFs during synthesis is nearly all you need for second-order optimization.

**Levenberg-Marquardt (Section 7.2).** Near the χ² minimum, a parabolic expansion χ²(x + δx) ≈ χ²(x) + δxᵀ∇χ² + ½δxᵀH'δx (Eq. 38) gives the normal equations ∇χ² + H δx = 0 (Eq. 39), with the LM modification

$$2H_{ij} = \begin{cases} H'_{ij}(1+\lambda), & i=j, \\ H'_{ij}, & i \neq j. \end{cases}$$

Large λ → steepest descent; small λ → Gauss-Newton. Adaptive λ is updated every iteration. The post-convergence formal uncertainty is σ²_m ≈ (2/(np+r)) · [Σ (I_obs − I_syn)² w²] / [Σ R²_m w²] (Eq. 42): the larger the RF, the smaller the retrieval uncertainty.

**Nodes and SVD (§7.2.1).** With many depth grid points (say 20-30), the Hessian becomes 20×20 or bigger and can be quasi-singular because of wide dynamical range in Stokes sensitivities. SIR's twofold fix: (1) constrain depth-perturbations via a small number of nodes (1 = constant, 2 = linear, 3 = parabolic, etc.), so free parameters are y_m ≡ x_i at nodes; (2) invert the Hessian via Singular Value Decomposition (SVD), zeroing the inverses of diagonal elements below threshold. The "equivalent RF" \bar R at nodes arises from the interpolation (Eq. 43). §7.2.2 discusses SIR's automatic node-selection by looking for zero-crossings of ∂χ²/∂a_p — an Occam-razor implementation — while §7.2.3 covers NICOLE's non-LTE inversion with the "fixed departure coefficient" approximation.

**Other algorithms (§7.3-7.4).** §7.3 covers database/PCA inversions (FATIMA, CSIRO-Meudon): precompute an eigenprofile database, and the inversion reduces to a look-up in the low-dimensional expansion-coefficient space. §7.4 covers ANN (Carroll & Staude 2001; Socas-Navarro 2003; Carroll & Kopf 2008), genetic algorithms (Charbonneau 1995's PIKAIA used in HELIX and MPS codes), and Bayesian inversions (Asensio Ramos et al. 2007a, 2009). ANNs are trained on synthetic Stokes→physical-parameter pairs.

**Spatial inversions (§7.5).** §7.5.1: van Noort's (2012) spatially-coupled SIR jointly inverts all pixels in a 2D map with the telescope PSF as a coupling operator — a massive but consistent problem. §7.5.2: Ruiz Cobo & Asensio Ramos (2013) use PCA-regularized deconvolution of observed spectra before pixel-by-pixel SIR inversion. §7.5.3: Asensio Ramos & de la Cruz Rodríguez (2015) exploit sparsity — wavelet-represent the unknowns, solve for the few nonzero coefficients via proximal gradient methods. This factor-of-3-to-5 dimensional reduction yields results comparable to or better than pixel-to-pixel inversion.

**Summary table.** Table 1 lists ~30 inversion codes by identifier, reference, method (LM, PCA, ANN, GA, Bayesian, gradient descent), and "in use" status. LM dominates.

**KR.** **χ² merit function (식 35).**

$$\chi^2(\mathbf{x}) = \frac{1}{N_f} \sum_{s=0}^{3} \sum_{i=1}^{q} [I_s^{\text{obs}}(\lambda_i) - I_s^{\text{syn}}(\lambda_i; \mathbf{x})]^2 w_{s,i}^2$$

합은 네 Stokes와 q 파장 샘플에 대해 실행되고, N_f = 4q − (np + r)은 자유도, w_{s,i}는 가중치이다. χ²의 gradient와 Hessian은 RF로 직접 표현된다(식 36, 37): ∂χ²/∂x_m = (2/N_f) Σ [...] R_m, ∂²χ²/∂x_m ∂x_k ≈ (2/N_f) Σ R_m R_k. 따라서 synthesis 중 RF 계산만으로도 2차 최적화에 거의 충분하다.

**Levenberg-Marquardt (7.2절).** χ² 최소 근방에서 parabolic 확장 χ²(x + δx) ≈ χ²(x) + δxᵀ∇χ² + ½δxᵀH'δx (식 38)로부터 정규방정식 ∇χ² + H δx = 0 (식 39)이 나오고, LM 수정은

$$2H_{ij} = \begin{cases} H'_{ij}(1+\lambda), & i=j, \\ H'_{ij}, & i \neq j. \end{cases}$$

λ가 크면 steepest descent, 작으면 Gauss-Newton. 매 iteration마다 λ를 적응적으로 조정. 수렴 후 formal uncertainty는 σ²_m ≈ (2/(np+r)) · [Σ (I_obs − I_syn)² w²] / [Σ R²_m w²] (식 42): RF가 클수록 retrieval uncertainty가 작다.

**Node와 SVD (§7.2.1).** 깊이 grid point가 많으면(예: 20-30개) Hessian이 20×20 이상이 되고 Stokes 민감도의 넓은 동적 범위 때문에 quasi-singular가 된다. SIR의 해결책 두 가지: (1) 깊이 섭동을 소수의 node(1=상수, 2=선형, 3=parabolic 등)로 제약하여 자유 파라미터를 y_m ≡ node에서의 x_i로 둔다; (2) Singular Value Decomposition(SVD)으로 Hessian을 역산하여 임계값 이하 대각 원소의 역수를 영으로 설정. 보간으로부터 node에서의 "equivalent RF" \bar R이 정의된다(식 43). §7.2.2는 SIR의 ∂χ²/∂a_p의 영점 탐색에 기반한 자동 node 선택—Occam-razor 구현—을 논의하고, §7.2.3은 "fixed departure coefficient" 근사를 쓰는 NICOLE의 non-LTE inversion을 다룬다.

**기타 알고리즘 (§7.3-7.4).** §7.3은 database/PCA inversion(FATIMA, CSIRO-Meudon): eigenprofile 데이터베이스를 미리 계산하고, inversion을 저차원 확장계수 공간의 조회로 귀결. §7.4는 ANN(Carroll & Staude 2001; Socas-Navarro 2003; Carroll & Kopf 2008), genetic algorithm(Charbonneau 1995 PIKAIA, HELIX와 MPS 코드 사용), Bayesian inversion(Asensio Ramos et al. 2007a, 2009). ANN은 synthetic Stokes→물리 파라미터 쌍으로 학습한다.

**공간 inversion (§7.5).** §7.5.1: van Noort(2012)의 공간 결합 SIR은 telescope PSF를 결합 연산자로 사용해 2D 맵의 모든 pixel을 동시에 역산—거대하지만 consistent한 문제. §7.5.2: Ruiz Cobo & Asensio Ramos(2013)은 pixel별 SIR inversion 전에 PCA 정규화된 관측 스펙트럼 deconvolution. §7.5.3: Asensio Ramos & de la Cruz Rodríguez(2015)은 sparsity 활용—미지수를 wavelet으로 표현하고 proximal gradient method로 소수의 비영 계수를 풂. 3-5배 차원 축소로 pixel-to-pixel과 동등하거나 더 나은 결과.

**요약 표.** Table 1은 약 30개의 inversion 코드를 identifier, 참고문헌, 방법(LM, PCA, ANN, GA, Bayesian, gradient descent), "사용 중" 여부로 정리. LM이 지배적.

### Part VIII: Discussion — inversion results / Section 8 — 역산 결과 논의

**EN.** Section 8.1 presents a controlled experiment: take a Hinode/SP-like "observed model" (with realistic v_LOS, B, γ, φ gradients), synthesize Fe I 630.1/630.2 nm, convolve with Hinode PSF, add 10⁻³ I_c noise, and invert with six SIR "modes" of increasing complexity (mode 1 = ME-like, nodes = 1; mode 6 = automatic node selection). Fig. 25-26 show the retrieved stratifications and Stokes fits. Mode 1 (ME) gives decent fits (typical misfit ≤ 10%) but cannot reproduce asymmetries; modes 4-6 (n_B = 5, 7, auto) nearly exactly recover v_LOS, B, γ, φ gradients with uncertainty shaded regions that agree with the true stratifications in the [log τ_c ≈ −3, 0] range. Fig. 27 tracks Stokes V amplitude (δa) and area (δA) asymmetries as node count grows — mode 4+ reproduces both.

Section 8.2 tackles the perennial controversy about weak-field retrievals. They simulate Hinode/SP Fe I 630 nm profiles for 10,000 pixels with B ∈ {10, 15, 20, 25, 40, 50, 60, 75, 90, 100 G}, isotropic inclination, add S/N = 1000 or 3000 noise, invert à-la-ME with SIR. Fig. 29 shows: (1) fields weaker than 75 G (for S/N=1000) are slightly overestimated but never catastrophically; (2) the retrieved inclination PDFs cluster around 90° for weak fields, but this is just noise dominating Q, U — not a systematic failure of the visible lines. Fig. 30 reproduces input lognormal B PDF (Orozco Suárez et al. 2007b) in the retrievals; fields above ~60 G are faithfully recovered. The conclusion: visible Fe I lines are not useless for weak-field diagnostics — the key is S/N.

**KR.** 8.1절은 통제된 실험을 제시한다: 현실적인 v_LOS, B, γ, φ 그라디언트를 가진 Hinode/SP 유사 "관측 모델"에서 Fe I 630.1/630.2 nm을 합성, Hinode PSF로 convolve, 10⁻³ I_c noise 추가 후, 복잡도가 증가하는 SIR 여섯 "mode"(mode 1 = ME-like, nodes = 1; mode 6 = 자동 node 선택)로 역산. Fig. 25-26은 retrieved stratification과 Stokes fit을 보여준다. Mode 1(ME)은 괜찮은 fit(전형적 misfit ≤ 10%)을 주지만 asymmetry를 재현 못함; mode 4-6(n_B = 5, 7, auto)은 v_LOS, B, γ, φ 그라디언트를 거의 정확히 복원하며 uncertainty shaded region이 [log τ_c ≈ −3, 0] 범위에서 실제 stratification과 일치. Fig. 27은 node 수에 따른 Stokes V 진폭(δa)과 면적(δA) asymmetry를 추적—mode 4+ 가 둘 다 재현.

8.2절은 weak-field retrieval에 대한 오래된 논쟁을 다룬다. 10,000 pixel에 대해 B ∈ {10, 15, 20, 25, 40, 50, 60, 75, 90, 100 G}, 등방성 inclination, S/N = 1000 또는 3000의 noise를 추가한 Hinode/SP Fe I 630 nm 프로파일을 합성, SIR à-la-ME로 역산. Fig. 29: (1) 75 G 이하 자기장(S/N=1000)은 약간 과대평가되지만 결코 재앙적이지 않음; (2) 약한 필드의 retrieved inclination PDF가 90° 근방에 몰리는데, 이는 Q, U를 지배하는 noise 때문이지 가시광선의 체계적 실패는 아님. Fig. 30은 입력 lognormal B PDF(Orozco Suárez et al. 2007b)가 inversion으로 재현됨을 보여주며, ~60 G 이상은 충실히 복원됨. 결론: 가시광 Fe I 선은 약장 진단에 무용지물이 아니며—핵심은 S/N이다.

### Part IX: Conclusions and Appendix / Section 9 and Appendix — 결론 및 부록

**EN.** Section 9 distills the message: (1) inversion is a topological mapping between observable and physical spaces; its assumptions control its uncertainties; (2) the step-by-step approach (Occam) is superior — start with constant quantities, increase complexity only if noise allows; (3) weak-field is only a valid approximation for very weak fields and primarily for chromospheric broad lines — Stokes V is not strictly proportional to B_LOS; (4) Stokes I is a *better*, more precise measurement than Q, U, V because polarimetric efficiency is higher by √3; so use I together with V for field strength inference. (5) The MISMA hypothesis is conceptually and technically dispensable. The **Appendix** gives a practical initialization recipe: compute the longitudinal B via center-of-gravity (Semel 1967, Rees & Semel 1979)

$$B_{\text{LOS}} = \beta_B \frac{\lambda_+ - \lambda_-}{2}, \quad v_{\text{LOS}} = \beta_v \frac{\lambda_+ + \lambda_-}{2}$$

with β_B = 1/C, β_v = c/λ₀, C = 4.67·10⁻¹³ λ₀² g_eff, and λ_± = Σ S_±(λ_i) λ_i / Σ S_±(λ_i) (S_± = I ± V). Then use weak-field φ ≃ ½ arctan(U/Q) and tan²γ ≃ |4L g²_eff Δλ / (3 \bar G C B_LOS)| (Eqs. 67, 69) for γ, φ initial guesses. This initialization accelerates convergence dramatically (Fig. 23: an order of magnitude).

**KR.** 9절은 메시지를 요약한다: (1) inversion은 관측가능공간과 물리공간 간의 위상 사상이며, 그 가정이 불확실성을 지배한다; (2) 단계적 접근(Occam)이 우월—상수 양에서 시작해 noise가 허락할 때만 복잡도를 증가; (3) weak-field는 매우 약한 필드와 주로 chromospheric 넓은 선에서만 유효—Stokes V는 B_LOS에 엄밀히 비례하지 않는다; (4) Stokes I가 Q, U, V보다 더 *정밀*—polarimetric 효율이 √3배 높음; 따라서 자기장 강도 추론에 I와 V를 함께 사용; (5) MISMA 가설은 개념적/기술적으로 폐기 가능. **부록**은 실용적 초기화 레시피: center-of-gravity(Semel 1967, Rees & Semel 1979)로 종방향 B 계산

$$B_{\text{LOS}} = \beta_B \frac{\lambda_+ - \lambda_-}{2}, \quad v_{\text{LOS}} = \beta_v \frac{\lambda_+ + \lambda_-}{2}$$

여기서 β_B = 1/C, β_v = c/λ₀, C = 4.67·10⁻¹³ λ₀² g_eff, λ_± = Σ S_±(λ_i) λ_i / Σ S_±(λ_i) (S_± = I ± V). 다음 γ, φ 초기값은 weak-field φ ≃ ½ arctan(U/Q)과 tan²γ ≃ |4L g²_eff Δλ / (3 \bar G C B_LOS)| (식 67, 69). 이 초기화는 수렴을 극적으로 가속한다(Fig. 23: 한 자릿수 이상).

---

## 3. Key Takeaways / 핵심 시사점

1. **Inversion = ill-conditioned topological mapping / Inversion은 ill-conditioned 위상 사상이다** — The Stokes profile space and the physical-quantity space have different, in general non-unique, correspondences. Fig. 1 (two V profiles, two atmospheres) is the canonical proof. Every inversion retrieves *a* solution consistent with its assumptions, not *the* solution. 두 개의 identical한 Stokes V 프로파일이 2000 K의 T 차이와 270 G의 B 차이를 가진 두 대기에서 동일하게 나올 수 있다. 이는 inversion의 비유일성을 직접적으로 보여주며, 각 retrieval은 그 inversion이 가정한 물리 하에서의 한 solution일 뿐 유일한 정답이 아니다.

2. **Milne-Eddington is the workhorse approximation / Milne-Eddington은 주력 근사이다** — Nine parameters (B, γ, φ, v_LOS, η₀, ΔλD, a, S₀, S₁) and an analytic solution Eq. 14 make ME inversion fast and robust; the retrieved values correspond to Stokes-RF-weighted averages at log τ_c ≈ −1.5 to −2. The HAO-ASP, VFISV (HMI), HELIX, and MILOS codes are all ME-based. 9개 파라미터(B, γ, φ, v_LOS, η₀, ΔλD, a, S₀, S₁)와 해석해(식 14)로 ME inversion은 빠르고 안정적이다; 얻어진 값은 log τ_c ≈ −1.5 ~ −2에서의 Stokes-RF 가중 평균에 해당한다. HAO-ASP, VFISV(HMI), HELIX, MILOS 모두 ME 기반이다.

3. **Weak-field approximation is dangerous above ~200-400 G / 약장 근사는 ~200-400 G 이상에서 위험하다** — Despite textbook lore, V ∝ −g·ΔλB·cos γ · ∂I/∂λ fails beyond 200 G (6 pm FWHM) or 400 G (8.8 pm FWHM). Stokes I itself departs from I_nm earlier, breaking the assumption I = I_nm. Use Stokes I alongside V for strength inference. 교과서적 통념과 달리, V ∝ −g·ΔλB·cos γ · ∂I/∂λ는 6 pm FWHM에서는 200 G, 8.8 pm에서는 400 G 이상에서 파탄난다. Stokes I 자체가 I_nm에서 더 빨리 벗어나 I = I_nm 가정을 깨뜨린다. 자기장 강도 추론에 V뿐 아니라 I도 함께 써라.

4. **Response functions do triple duty / Response function은 세 가지 역할을 한다** — As (a) sensitivity diagnostics (which observable tracks which parameter at which depth), (b) Jacobians for LM (∇χ² = Σ R·Δ and H ≈ Σ R·R), and (c) uncertainty quantifiers (σ² ∝ 1/Σ R²). They can be computed alongside Stokes synthesis at negligible extra cost. (a) 민감도 진단 (어느 관측량이 어느 깊이의 어느 파라미터를 추적하는지), (b) LM의 Jacobian (∇χ² = Σ R·Δ, H ≈ Σ R·R), (c) 불확실성 정량화 (σ² ∝ 1/Σ R²). Stokes synthesis와 함께 추가 비용 거의 없이 계산 가능.

5. **SVD + nodes solves SIR's high-dimensional inversion / SVD + node가 SIR의 고차원 inversion을 해결한다** — Depth-varying codes face 20-100 dimensional Hessians that are numerically singular; SIR solves this via (a) node parameterization (1 = constant, 2 = linear, 3 = parabolic, ...) and (b) SVD-regularized Hessian inversion with small-eigenvalue zeroing. The result is a physics-principled Occam razor. 깊이 의존 code는 수치적으로 singular한 20-100차원 Hessian에 직면한다; SIR은 (a) node parameterization(1=상수, 2=선형, 3=parabolic, ...)과 (b) 작은 eigenvalue를 영으로 하는 SVD 정규화 Hessian inversion으로 해결. 결과는 물리 원리에 기반한 Occam razor.

6. **180° azimuth ambiguity is fundamental / 180° 방위각 ambiguity는 본질적이다** — Q and U depend on sin(2φ), cos(2φ), so φ and φ+180° yield identical linear polarization. Zeeman observations cannot resolve this alone. External constraints (minimum spatial field divergence, acute-angle continuity, force-free extrapolation) are required. Q, U는 sin(2φ), cos(2φ)에 의존하므로 φ와 φ+180°가 동일 선편광을 준다. Zeeman 관측만으로는 해결 불가. 외부 제약(공간적 발산 최소화, 예각 연속성, force-free 외삽) 필요.

7. **Noise sets the reasonable complexity / Noise가 합리적 복잡도를 정한다** — The step-by-step Occam strategy: add a node only if the residuals (χ² fit) exceed ~3σ. With S/N = 1000 Hinode/SP data and Fe I 630 nm, parabolic stratifications in B, γ, φ are reliably retrieved in [log τ_c ≈ −3, 0]; higher-order modes bring no information. 단계적 Occam 전략: 잔차(χ² fit)가 ~3σ를 초과할 때만 node를 추가하라. S/N=1000 Hinode/SP, Fe I 630 nm로는 B, γ, φ의 parabolic stratification이 [log τ_c ≈ −3, 0]에서 신뢰성 있게 복원된다; 더 높은 mode는 추가 정보 없음.

8. **Visible lines are not useless for weak fields / 가시광선은 약장 진단에 무용하지 않다** — Controversy repeatedly claimed Fe I 630 nm lines fail for B ≤ 100 G internetwork fields. The authors demonstrate (Fig. 29-30) that SIR à-la-ME correctly retrieves input B lognormal PDF above ~60 G, and even between 10-60 G the retrieval is only moderately biased. The issue is noise, not the lines. Fe I 630 nm이 B ≤ 100 G internetwork에 실패한다는 주장이 반복되었지만, 저자들은 Fig. 29-30에서 SIR à-la-ME가 ~60 G 이상에서 입력 B lognormal PDF를 정확히 복원하고, 10-60 G에서도 약간의 편향만 있음을 보인다. 문제는 noise이지 선 자체가 아니다.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Core polarized RTE / 핵심 편광 RTE

Differential form (Eq. 6): dI/dτ_c = K (I − S).

Formal integral (Eq. 8): I(0) = ∫₀^∞ O(0, τ_c) K(τ_c) S(τ_c) dτ_c.

### 4.2 Propagation matrix K / 전파 행렬 K (4×4, Pauli-like decomposition)

$$\mathbf{K} = \eta_I \mathbb{1} + \begin{pmatrix} 0 & \eta_Q & \eta_U & \eta_V \\ \eta_Q & 0 & \rho_V & -\rho_U \\ \eta_U & -\rho_V & 0 & \rho_Q \\ \eta_V & \rho_U & -\rho_Q & 0 \end{pmatrix}$$

The η's are absorption coefficients for each Stokes state (pleochroism); the ρ's are dispersion coefficients (magneto-optical rotation). Each is a sum of Voigt (η) and Faraday-Voigt (ρ) profiles over π, σ_+, σ_- Zeeman components:
$$\eta_I = \frac{\eta_0}{2}[\phi_p \sin^2\gamma + \frac{1}{2}(\phi_b + \phi_r)(1+\cos^2\gamma)], \quad \text{etc.}$$

### 4.3 Milne-Eddington atmospheric parameters / ME 대기 파라미터 (9개)

{B, γ, φ} (magnetic vector), v_LOS, η₀ = χ_line/χ_cont, ΔλD (Doppler width), a (damping), S₀ (source function at τ_c = 0), S₁ (gradient).

Source vector: S = (S₀ + S₁ τ_c) e₀, e₀ = (1, 0, 0, 0)ᵀ.

### 4.4 Unno-Rachkovsky analytic solution / ME 해석해 (Eq. 14)

$$\mathbf{I}(0) = (S_0 + \mathbf{K}^{-1} S_1) \mathbf{e}_0$$

### 4.5 Weak-field approximation / 약장 근사

$$V(\lambda) \simeq -g_{\text{eff}} \Delta\lambda_B \cos\gamma \frac{\partial I_{nm}}{\partial\lambda} \quad \text{(Eq.24)}$$

with Zeeman shift ΔλB = λ₀² e₀ B / (4πmc²) (Eq. 20) and effective Landé factor g_eff = ½(g_u + g_l) + ¼(g_u − g_l)[j_u(j_u+1) − j_l(j_l+1)] (Eq. 18).

Also (for linear polarization in the weak regime): Q ≃ (3/4)·Δλ_B²·\bar G·sin²γ·(1/Δλ)·(∂I_nm/∂λ) ∝ B² sin²γ.

### 4.6 Response function / 반응 함수 (Eqs. 29, 30)

$$\delta\mathbf{I}(0) = \sum_{i=1}^{p+r} \int_0^{\infty} \mathbf{R}_i(\tau_c) \, \delta x_i(\tau_c) \, d\tau_c$$

$$\mathbf{R}_i(\tau_c) = \mathbf{O}(0, \tau_c) \left[\mathbf{K}(\tau_c) \frac{\partial \mathbf{S}}{\partial x_i} - \frac{\partial \mathbf{K}}{\partial x_i}(\mathbf{I} - \mathbf{S})\right]$$

### 4.7 χ² merit function and derivatives / χ² 메리트 함수와 미분

Eq. 35: χ² = (1/N_f) Σ_s Σ_i [I_s^obs − I_s^syn]² w_{s,i}²

Gradient (Eq. 36): ∂χ²/∂x_m = (2/N_f) Σ_s Σ_i [I_s^obs − I_s^syn] w² R_{m,s}

Hessian (Eq. 37): ∂²χ²/∂x_m ∂x_k ≈ (2/N_f) Σ_s Σ_i w² R_{m,s} R_{k,s}

### 4.8 Levenberg-Marquardt normal equations / LM 정규방정식 (Eq. 39)

$$\nabla\chi^2 + \mathbf{H}\,\delta\mathbf{x} = \mathbf{0}$$
with diagonal damping: 2H_{ii} = H'_{ii}(1+λ), off-diagonal: 2H_{ij} = H'_{ij}.

λ adaptively tuned: decrease if χ² decreases, increase otherwise.

### 4.9 Formal uncertainty / 형식적 불확실성 (Eq. 42)

$$\sigma_m^2 \simeq \frac{2}{np+r} \cdot \frac{\sum_{s,i}[I_s^{\text{obs}}-I_s^{\text{syn}}]^2 w_{s,i}^2}{\sum_{s,i} R_{m,s}^2(\lambda_i) w_{s,i}^2}$$

Larger RF ⟹ smaller uncertainty.

### 4.10 Appendix initialization / 부록 초기화

Center-of-gravity (Semel 1967):
$$B_{\text{LOS}} = \beta_B \frac{\lambda_+ - \lambda_-}{2}, \quad v_{\text{LOS}} = \beta_v \frac{\lambda_+ + \lambda_-}{2}$$
with β_B = 1/C, β_v = c/λ₀, C = 4.67 × 10⁻¹³ λ₀² g_eff.

λ_± are centroids of S_± = I ± V:
$$\lambda_\pm = \frac{\sum_i S_\pm(\lambda_i) \lambda_i}{\sum_i S_\pm(\lambda_i)}$$

Weak-field azimuth: φ ≃ ½ arctan(U/Q) (Eq. 61).

Inclination from linear polarization L = √(Q² + U²):
$$\tan^2\gamma \simeq \left|\frac{4 L g_{\text{eff}}^2 \Delta\lambda}{3 \bar G C B_{\text{LOS}}}\right| \quad \text{(Eq.69)}$$

---

### 4.11 Worked numerical example: Fe I 630.25 nm ME inversion / 수치 예제

**Setup.** Consider an ME atmosphere with B = 1000 G, γ = 45°, φ = 30°, v_LOS = 0, η₀ = 5, ΔλD = 25 mÅ, a = 0.22, S₀ = 0.1, S₁ = 0.9, for the Fe I 630.25 nm line with g_eff = 2.5.

**Zeeman splitting.**
$$\Delta\lambda_B = \frac{\lambda_0^2 e_0 B}{4\pi m c^2} = 4.67\times 10^{-13} \cdot (6302.5)^2 \cdot 1000 \text{ Å} \approx 18.5 \text{ mÅ}$$
Ratio Δλ_B / Δλ_D ≈ 0.74, which times g_eff = 2.5 gives 1.85 — already in the intermediate/strong regime where weak-field fails.

**Expected Q/I and V/I amplitudes.** For this geometry, Stokes V/I_c peaks at ~0.2 (i.e., 20% of continuum) at ±Δλ_B ≈ ±18 mÅ from line center; Stokes Q/I_c peaks at ~0.03 (3%) with sin²γ = 0.5. Stokes U/I_c has the same magnitude as Q/I_c modulated by cos(2φ) − sin(2φ) factors.

**Weak-field limit check.** If we instead set B = 200 G, Δλ_B = 3.7 mÅ and V/I_c peak ≈ −g_eff Δλ_B cos γ (∂I_nm/∂λ)_peak. With (∂I_nm/∂λ)_peak ≈ 0.01 mÅ⁻¹, V_peak/I_c ≈ 2.5 × 3.7 × 0.707 × 0.01 ≈ 6.5% — consistent with Fig. 6 (200 G red V peak).

**Inclination uncertainty in Hinode/SP retrieval.** For B = 25 G at S/N = 1000 and isotropic inclination distribution, Fig. 29 right panel shows γ_output peaks near 90° (i.e., "most horizontal"). At B = 100 G the inclination PDF recovers the true isotropic shape. The uncertainty on γ for B = 25 G is σ_γ ~ 30°, vs. σ_γ ~ 10° for B = 100 G — a direct consequence of σ_γ ∝ 1/B_lin via Eq. 42.

**SIR run.** With SIR mode 4 (n_B = 5, n_T = 7) on 1000 synthetic profiles, convergence takes ~20 LM iterations starting from the CoG+weak-field appendix initialization (Fig. 23, red curve), an order-of-magnitude speedup over fixed initial guesses.

### 4.12 ASCII diagram: Stokes profile formation / Stokes 프로파일 형성 ASCII 도식

```
Observer (τ_c = 0)
     ↑  I(0)
     |────────────────────────────
     |  Upper photosphere     .  .  .   ← log τ_c ≈ -3
     |  (RFs of v_LOS, γ,    .  .  .      (chromospheric lines form here)
     |   φ peak here)        .  .  .
     |────────────────────────────
     |  Mid photosphere      ::::::   ← log τ_c ≈ -1.5
     |  (ME retrievals       ::::::     (most Fe I lines form here)
     |   correspond here)    ::::::
     |────────────────────────────
     |  Deep photosphere     ######   ← log τ_c ≈ 0
     |  (T sensitivity       ######     (continuum origin)
     |   dominates)          ######
     |────────────────────────────
         τ_c → ∞
         
      Stokes at observer = ∫ O·K·S dτ_c
```

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1852 ─────── Stokes (G.G.): Stokes parameters
              |
1908 ─────── Hale: first detection of sunspot magnetic fields via Zeeman splitting
              |
1953 ─────── Babcock: solar magnetograph (two-wavelength longitudinal B)
              |
1956 ─────── Unno: Milne-Eddington analytic RTE solution (absorption only)
              |
1962-67 ───── Rachkovsky: ME + magneto-optical (dispersion) effects
              |
1968 ─────── Mueller matrix formalism consolidated
              |
1972 ─────── Landi Degl'Innocenti: QED derivation of polarized RTE
              |
1972 ─────── Harvey, Livingston, Slaughter: "Solve for B by best fit" — SEMINAL INVERSION IDEA
              |
1977 ─────── Auer, Heasley, House: first proper synthetic-Stokes inversion method
              |
1984 ─────── Landolfi, Landi Degl'Innocenti, Arena: Florence inversion code
              |
1985 ─────── Landi Degl'Innocenti²: formal solution I(0) = ∫O·K·S dτ
              |
1987 ─────── Skumanich & Lites: HAO-ASP ME inversion code
              |
1990 ─────── Keller et al.: thin-flux-tube inversion code
              |
1992 ─────── Ruiz Cobo & del Toro Iniesta: SIR — depth-varying, RF-based, SVD, nodes
              |
1996 ─────── Sánchez Almeida et al.: MISMA hypothesis
              |
1997 ─────── Sánchez Almeida: IAC MISMA inversion
              |
1998 ─────── Frutiger & Solanki: SPINOR (SIR-like, thin flux tubes)
              |
2000 ─────── Socas-Navarro et al.: NICOLE (non-LTE), Bellot Rubio et al.: LILIA
              |
2000 ─────── Rees et al.: PCA database inversions (CSIRO-Meudon, FATIMA)
              |
2001 ─────── Carroll & Staude: artificial neural network inversions
              |
2003 ─────── Bellot Rubio: SIRGAUSS (Gaussian-profile penumbra)
              |
2004 ─────── Lagg et al.: HELIX (Hanle + Zeeman, He I 1083 nm), Asensio Ramos: molecular
              |
2007 ─────── Orozco Suárez & del Toro Iniesta: MILOS (IDL-based ME)
              |
2007a-07b ── Asensio Ramos et al.: Bayesian inversions, dimensionality
              |
2008 ─────── Asensio Ramos et al.: HAZEL (Hanle + Zeeman full slab)
              |
2009 ─────── Louis et al.: SIRJUMP (jump discontinuities)
              |
2011 ─────── Borrero et al.: VFISV for SDO/HMI
              |
2012 ─────── van Noort: spatially-coupled inversion with PSF
              |
2013 ─────── Ruiz Cobo & Asensio Ramos: regularized deconvolution
              |
2015 ─────── Asensio Ramos & de la Cruz Rodríguez: sparse (wavelet) inversion
              |
2015 ─────── Socas-Navarro et al.: NICOLE 2.0
              |
2016 ─────── THIS REVIEW — del Toro Iniesta & Ruiz Cobo: comprehensive synthesis
              |
2019 ─────── STiC (de la Cruz Rodríguez): multi-line non-LTE inversion
              |
2020+ ─────── DeSIRe, deep learning inversions (Milic & van Noort), Solar Orbiter/PHI uses MILOS,
              DKIST/ViSP uses SIR, Machine learning Bayesian inversions (Asensio Ramos et al.)
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Unno (1956) "Line formation of a normal Zeeman triplet" | Original ME analytic solution (absorption only) on which all ME codes build | Foundation. Eq. 14 of this review is Unno's formula |
| Rachkovsky (1962, 1967) | Added magneto-optical (dispersion) terms to K, completing the 7-element K | Completes the K matrix used by all modern inversion codes |
| Landi Degl'Innocenti (1972, 1985); Landi Degl'Innocenti & Landolfi (2004) monograph | QED derivation of RTE; formal integral solution I(0) = ∫O·K·S dτ; encyclopedic reference | Establishes the mathematical framework; Landi/Landolfi book is THE polarimetry textbook |
| Ruiz Cobo & del Toro Iniesta (1992) — SIR paper | The original SIR code, the flagship depth-varying inversion | This review's authors' own foundational work; most detailed operational discussion |
| Auer, Heasley & House (1977) | First inversion code based on Unno's theory with asymmetric profiles, gradients, magneto-optics | Prototype of the synthetic-Stokes fitting paradigm |
| Skumanich & Lites (1987) | HAO-ASP ME inversion, extensively used on ASP data | Most successful early ME code; template for VFISV, HELIX, MILOS |
| Socas-Navarro et al. (2000, 2015) — NICOLE | Only NLTE inversion code; uses fixed departure coefficients | Extension beyond LTE — critical for chromospheric inversions (Ca II 854.2 nm, Mg II h/k) |
| Borrero et al. (2011) — VFISV | Very Fast Inversion of Stokes Vector for SDO/HMI | Instrumental pipeline — every HMI vector magnetogram uses VFISV |
| van Noort (2012) — spatially coupled inversions | First 2D joint inversion accounting for telescope PSF | Prototype for modern spatial-regularization inversions (important for DKIST) |
| Asensio Ramos & de la Cruz Rodríguez (2015) — sparse inversion | Wavelet-sparse 2D ME inversion via proximal gradient | Modern approach to 2D/3D inversion, factor-3-5 dimensional reduction |
| Asensio Ramos et al. (2007a, 2009) — Bayesian inversions | MCMC, nested sampling for full posterior PDF | Alternative to LM — quantifies uncertainty rigorously |

---

## 7. References / 참고문헌

- del Toro Iniesta, J.C., Ruiz Cobo, B., "Inversion of the radiative transfer equation for polarized light", *Living Reviews in Solar Physics*, 13:4 (2016). DOI: 10.1007/s41116-016-0005-2 (**this paper**)
- Unno, W., "Line formation of a normal Zeeman triplet", *PASJ*, 8, 108 (1956)
- Rachkovsky, D.N., "Magnetic rotation effects in spectral lines", *Izv. Krym. Astrofiz. Obs.*, 27, 148 (1962); 37, 56 (1967)
- Landi Degl'Innocenti, E., Landi Degl'Innocenti, M., "Quantum electrodynamic approach to line formation theory", *Solar Physics*, 27, 319 (1972)
- Landi Degl'Innocenti, E., Landi Degl'Innocenti, M., "On the solution of the radiative transfer equations for polarized radiation", *Solar Physics*, 97, 239 (1985)
- Landi Degl'Innocenti, E., Landolfi, M., *Polarization in Spectral Lines*, Springer (2004) — the polarimetry bible
- Harvey, J., Livingston, W., Slaughter, C., "Inference of magnetic vector from observations of line profiles", in *Line Formation in the Presence of Magnetic Fields*, HAO (1972)
- Auer, L.H., Heasley, J.N., House, L.L., "Inference of magnetic and velocity fields from spectral line profile observations", *Solar Physics*, 55, 47 (1977)
- Skumanich, A., Lites, B.W., "Stokes profile analysis and vector magnetic fields. I. Inversion of photospheric lines", *ApJ*, 322, 473 (1987)
- Ruiz Cobo, B., del Toro Iniesta, J.C., "Inversion of Stokes profiles", *ApJ*, 398, 375 (1992) — original SIR paper
- Ruiz Cobo, B., del Toro Iniesta, J.C., "On the sensitivities of spectral lines to perturbations of the atmospheric parameters", *Solar Physics*, 164, 169 (1994)
- Frutiger, C., Solanki, S.K., "SPINOR — Stokes profile inversion using response functions", in *Space Solar Physics*, 368 (1998)
- Socas-Navarro, H., Trujillo Bueno, J., Ruiz Cobo, B., "Non-LTE inversion of line profiles", *ApJ*, 530, 977 (2000)
- Socas-Navarro, H., de la Cruz Rodríguez, J., Asensio Ramos, A., et al., "An open-source, massively parallel code for non-LTE synthesis and inversion of spectral lines and Zeeman-induced Stokes profiles", *A&A*, 577, A7 (2015) — NICOLE release
- Asensio Ramos, A., Trujillo Bueno, J., Landi Degl'Innocenti, E., "Advanced forward modeling and inversion of Stokes profiles resulting from the joint action of the Hanle and Zeeman effects", *ApJ*, 683, 542 (2008) — HAZEL
- Orozco Suárez, D., del Toro Iniesta, J.C., "The usefulness of analytic response functions", *A&A*, 462, 1137 (2007) — MILOS
- Borrero, J.M., Tomczyk, S., Kubo, M., et al., "VFISV: very fast inversion of the Stokes vector for the Helioseismic and Magnetic Imager", *Solar Physics*, 273, 267 (2011)
- van Noort, M., "Spatially coupled inversion of spectro-polarimetric image data", *A&A*, 548, A5 (2012)
- Ruiz Cobo, B., Asensio Ramos, A., "Returning magnetic flux in sunspot penumbrae", *A&A*, 549, L4 (2013)
- Asensio Ramos, A., de la Cruz Rodríguez, J., "Sparse inversion of Stokes profiles. I.", *A&A*, 577, A140 (2015)
- Asensio Ramos, A., Martínez González, M.J., Rubiño-Martín, J.A., "Bayesian analysis of spectropolarimetric observations", *A&A*, 476, 959 (2007a)
- Semel, M., "Contribution à l'étude des champs magnétiques dans les régions actives solaires", *Ann. Astrophys.*, 30, 513 (1967) — center-of-gravity method
- Rees, D.E., Semel, M.D., "Line formation in an unresolved magnetic element", *A&A*, 74, 1 (1979)
- Rees, D.E., López Ariste, A., Thatcher, J., Semel, M., "Fast inversion of polarization profiles using PCA", *A&A*, 355, 759 (2000)
- Rempel, M., "Numerical sunspot models", *ApJ*, 750, 62 (2012)
- Sánchez Almeida, J., Landi Degl'Innocenti, E., Martínez Pillet, V., Lites, B.W., "Line asymmetries and the microstructure of photospheric magnetic fields", *ApJ*, 466, 537 (1996) — MISMA
- Charbonneau, P., "Genetic algorithms in astronomy and astrophysics", *ApJS*, 101, 309 (1995) — PIKAIA
- Carroll, T.A., Staude, J., "The inversion of Stokes profiles with artificial neural networks", *A&A*, 378, 316 (2001)
- Socas-Navarro, H., "Measuring solar magnetic fields with artificial neural networks", *Neural Networks*, 16, 355 (2003)
- Stokes, G.G., "On the composition and resolution of streams of polarized light from different sources", *Trans. Camb. Philos. Soc.*, 9, 399 (1852)
- Hale, G.E., "On the probable existence of a magnetic field in sun-spots", *ApJ*, 28, 315 (1908)
- Mickey, D.L., Orrall, F.Q., "An observational test of Unno's theory", *A&A*, 31, 179 (1974) — NCP discovery
- Auer, L.H., Heasley, J.N., "The origin of broad-band circular polarization in sunspots", *A&A*, 64, 67 (1978)
- Orozco Suárez, D., Bellot Rubio, L.R., del Toro Iniesta, J.C., et al., "Quiet-Sun internetwork magnetic fields from the inversion of Hinode measurements", *ApJ*, 670, L61 (2007b)
- del Toro Iniesta, J.C., "On the discovery of the Zeeman effect on the Sun and in the laboratory", *Vistas Astron.*, 40, 241 (1996)
- del Toro Iniesta, J.C., *Introduction to Spectropolarimetry*, Cambridge Univ. Press (2003b)
