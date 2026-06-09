---
title: "Pre-Reading Briefing: Inversion of Stokes Profiles (SIR code)"
paper_id: "33"
topic: Solar_Observation
date: 2026-04-23
type: briefing
---

# Inversion of Stokes Profiles: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Ruiz Cobo, B. & del Toro Iniesta, J. C., "Inversion of Stokes Profiles", ApJ, 398, 375 (1992). DOI: 10.1086/171862
**Author(s)**: Basilio Ruiz Cobo; Jose Carlos del Toro Iniesta (Instituto de Astrofísica de Canarias, Tenerife, Spain)
**Year**: 1992

---

## 1. 핵심 기여 / Core Contribution

**EN.** The paper introduces **SIR** ("Stokes Inversion based on Response functions"), a general-purpose inversion code that recovers the depth stratification of temperature $T(\tau)$, magnetic field vector $[B(\tau), \gamma(\tau), \phi(\tau)]$, and line-of-sight velocity $v(\tau)$ — plus depth-constant micro- and macro-turbulence $\xi_\mathrm{mic}, \xi_\mathrm{mac}$ — from observed full-Stokes line profiles $I, Q, U, V$. The two key innovations are: (1) **simultaneous computation of response functions (RFs) with the full polarized RTE** (no separate finite-difference passes), exploiting the DELO formal solution (Rees, Murphy & Durrant 1989) and the Sánchez Almeida (1992) RF expression; (2) a **node-based parameterization** in which each physical quantity is described by $m$ values at $m$ equi-spaced nodes in $\log\tau$ and smoothed by cubic splines — drastically reducing the number of free parameters from $5n+2$ to $5m+2$ (with $m \ll n$), while using **equivalent RFs** that preserve full-atmosphere information. A **modified SVD** (weighting every physical magnitude equally) protects against ill-conditioning in the Marquardt curvature matrix.

**KR.** 본 논문은 관측된 풀-스토크스 프로파일 $I, Q, U, V$ 로부터 광학심도 축을 따라 온도 $T(\tau)$, 자기장 벡터 $[B(\tau), \gamma(\tau), \phi(\tau)]$, 시선속도 $v(\tau)$ 의 층상구조(stratification), 그리고 깊이-상수인 미시·거시 난류속도 $\xi_\mathrm{mic}, \xi_\mathrm{mac}$ 를 복원하는 범용 인버전 코드 **SIR**("Stokes Inversion based on Response functions")을 제시한다. 두 가지 핵심 혁신은 다음과 같다 — (1) **응답함수(RF)를 편광 복사전달방정식(RTE)의 형식해(DELO 방법)와 동시에 계산**하여 매 반복마다 $\nabla \chi^2$ 평가를 위해 RTE를 다시 풀 필요가 없게 한 것; (2) **노드 기반 매개변수화** — 각 물리량을 $\log\tau$ 상 등간격 $m$개의 노드 값으로만 표현하고 큐빅 스플라인으로 보간함으로써 자유 파라미터 수를 $5n+2$ 에서 $5m+2$ 로 급감시키면서도 "등가 RF"(전 대기에 걸친 RF의 선형결합)를 통해 전 대기 정보를 유지한 것. 곡률행렬의 특이성(ill-conditioning) 문제는 **수정 SVD** 기법으로 해결하여, 각 물리량별 민감도 차이를 균등화한다.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

**EN.** By the late 1980s, two schools of "model atmosphere determination from Stokes profiles" competed: (a) **synthesis** (Auer, Heasley & House 1977) — trial-and-error forward modeling; accurate but expensive; (b) **inversion** — nonlinear least-squares fit. Existing inversion codes either assumed analytical Milne-Eddington (AHH: Auer et al. 1977; extended by Landolfi, Landi Degl'Innocenti & Arena 1984; Skumanich & Lites 1987; Lites et al. 1988; Murphy 1991) giving single-valued atmospheric parameters, or fitted small-scale flux tubes with MHD+pressure balance (Keller, Solanki et al. 1990 = "KAL") fitting only selected spectrum samples. Neither class recovered **depth stratification** of multiple free physical quantities from asymmetric Stokes profiles — but velocity gradients are ubiquitous in sunspots (Illing, Landman & Mickey 1975) and faculae (Sánchez Almeida, Collados & del Toro Iniesta 1989). Ruiz Cobo & del Toro Iniesta stepped into this gap by combining an accurate DELO RTE solver with a fast RF-based gradient computation and node parameterization.

**KR.** 1980년대 후반까지 "스토크스 프로파일에서 대기 모델 결정"은 두 파로 나뉘어 있었다 — (a) **합성(synthesis) 기법**(Auer, Heasley & House 1977): 시행착오형 정방향 모델링, 정확하지만 큰 관측량 처리에 비용이 과도; (b) **인버전(inversion) 기법**: 비선형 최소자승법. 기존 인버전 코드는 해석적 Milne-Eddington 가정(AHH: Auer et al. 1977, 이후 Landolfi et al. 1984, Skumanich & Lites 1987, Lites et al. 1988, Murphy 1991로 확장) 하에서 단일값 대기 파라미터만 얻거나, 또는 소규모 flux tube를 MHD + 수평 압력평형과 함께 선별 샘플에만 적합(KAL: Keller, Solanki et al. 1990) 하는 방식에 국한되었다. 어느 쪽도 비대칭 스토크스 프로파일로부터 여러 물리량의 **깊이 층상구조(depth stratification)** 를 복원하지 못했다. 그러나 흑점의 시선속도 경사(Illing et al. 1975) 및 백반의 비대칭성(Sánchez Almeida et al. 1989)은 속도 경사(velocity gradients)가 태양 대기에 편재함을 보여주고 있었다. Ruiz Cobo & del Toro Iniesta는 정확한 DELO RTE solver + RF 기반 고속 gradient 계산 + 노드 매개변수화를 결합하여 이 공백을 메웠다.

### 타임라인 / Timeline

```
1971 ─── Mein: weighting functions (1D unpolarized RFs)
1975 ─── Beckers & Milkey: RF generalization
1977 ─── Auer, Heasley & House (AHH): Milne-Eddington inversion
1977 ─── Landi Degl'Innocenti & Landi Degl'Innocenti: RF for polarized case
1982 ─── Landi & Landolfi: diagnostic use of RFs
1984 ─── Landolfi et al.: AHH generalization
1985 ─── Landi Degl'Innocenti: DELO formalism precursor
1986 ─── Press et al.: Numerical Recipes — SVD, Marquardt
1989 ─── Rees, Murphy & Durrant: DELO method (polarized RTE)
1989 ─── Sánchez Almeida, Collados, del Toro Iniesta: facula asymmetries
1990 ─── Keller/Solanki et al. (KAL): MHD flux-tube inversion
1991 ─── Jefferies & Mickey: weak-field initialization
1992 ─── Sánchez Almeida: general RF expression (prerequisite for SIR)
1992 ─── ┌─────────────────────────────────────────────────┐
         │  THIS PAPER — SIR (Ruiz Cobo & del Toro Iniesta) │
         └─────────────────────────────────────────────────┘
   ↓
1994 ─── del Toro Iniesta & Ruiz Cobo: SIR user applications
2016 ─── del Toro Iniesta & Ruiz Cobo: LRSP inversion review (Paper #13 LRSP)
```

---

## 3. 필요한 배경 지식 / Prerequisites

**EN.**
1. **Polarized radiative transfer**. The Unno-Rachkovsky equation $\frac{d\mathbf{I}}{d\tau} = \mathbf{K}(\mathbf{I} - \mathbf{S})$ where $\mathbf{I} = (I, Q, U, V)^T$, $\mathbf{K}$ is the $4\times4$ propagation matrix (absorption + magneto-optical terms), $\mathbf{S}$ is the source-function vector.
2. **Milne-Eddington solution** (as baseline / initialization scheme).
3. **Marquardt-Levenberg algorithm** for nonlinear least-squares: $(\mathbf{A} + \lambda \mathbf{I})\,\delta\mathbf{a} = -\nabla\chi^2$.
4. **Response functions** $R_m(\tau, \lambda) = \partial I(\lambda)/\partial x_m(\tau)$ — Landi Degl'Innocenti & Landi Degl'Innocenti 1977.
5. **Singular Value Decomposition (SVD)** for pseudo-inverse of ill-conditioned matrices (Numerical Recipes, Press et al. 1986 §2.9).
6. **Cubic splines** and equi-spaced interpolation.
7. Basic sunspot atmospheric models (umbra: Maltby et al. 1986 "model E"; Henoux 1969).

**KR.**
1. **편광 복사전달이론**. Unno-Rachkovsky 방정식 $\frac{d\mathbf{I}}{d\tau} = \mathbf{K}(\mathbf{I} - \mathbf{S})$, $\mathbf{I} = (I, Q, U, V)^T$, $\mathbf{K}$ 는 흡수 + 자기광학 효과를 포함하는 $4\times4$ 전파행렬, $\mathbf{S}$ 는 source-function 벡터.
2. **Milne-Eddington 해** (기준 및 초기화용).
3. **Marquardt-Levenberg 알고리즘**: $(\mathbf{A} + \lambda \mathbf{I})\,\delta\mathbf{a} = -\nabla\chi^2$.
4. **응답함수** $R_m(\tau, \lambda) = \partial I(\lambda)/\partial x_m(\tau)$ — Landi Degl'Innocenti & Landi Degl'Innocenti (1977).
5. **특이값 분해(SVD)** — 특이행렬 유사역행렬 계산 (Press et al. 1986 §2.9).
6. **큐빅 스플라인** 과 등간격 보간.
7. 흑점 대기 모형 기초 (본영: Maltby et al. 1986 "model E"; Henoux 1969).

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| Stokes vector $(I, Q, U, V)$ | 완전한 편광 상태 기술 4-벡터 / Four-component vector fully describing the polarization state of light |
| Response Function (RF) $R_m(\tau, \lambda)$ | 어떤 물리량 $x_m(\tau)$ 의 미소 섭동 $\delta x_m$ 에 대한 방출 스펙트럼 $\delta I(\lambda)$ 의 편미분 커널 / Kernel $\partial I(\lambda)/\partial x_m(\tau)$ telling how observed spectrum responds to a perturbation at depth $\tau$ |
| DELO method | Diagonal Element Lambda Operator — 대각 원소 람다 연산자 방법으로 편광 RTE의 형식해를 효율적으로 수치 적분 (Rees et al. 1989) / Efficient numerical integrator for polarized RTE using the diagonal-element evolution operator |
| Merit function $\chi^2$ | 관측 $I_k^\mathrm{obs}$ 와 합성 $I_k^\mathrm{syn}$ 의 제곱합 잔차 / Squared-residual sum, eq. (1): $\chi^2 = (1/\nu)\sum_{k,i}[I_k^\mathrm{obs}(\lambda_i) - I_k^\mathrm{syn}(\lambda_i)]^2$ |
| Marquardt algorithm | Gauss-Newton + steepest descent 혼합, 감쇠 계수 $\lambda$ 로 혼합 비율 제어 / Damped Gauss-Newton method; mixing gradient descent and Hessian inverse via scalar $\lambda$ |
| Curvature matrix $\mathbf{A}$ | $\chi^2$ 의 (근사) 헤시안 $\approx$ RF 간 내적 행렬 / (Approximate) Hessian of $\chi^2$; built from pairwise RF products |
| Node parameterization | 각 물리량을 $\log\tau$ 상 $m$ 개 등간격 노드 값으로만 기술, 큐빅 스플라인 보간 / Describe each physical magnitude by $m$ values on equi-spaced $\log\tau$ nodes, interpolate with cubic splines |
| Equivalent RF $\mathcal{R}_k(\lambda, \tau_l)$ | 노드에서 샘플된 RF — 전 대기 RF 값들의 선형결합 (Appendix A eq. 14) / RFs sampled at nodes, linear combinations of full-atmosphere RFs through cubic-spline coefficients |
| Micro-/macroturbulence $\xi_\mathrm{mic}, \xi_\mathrm{mac}$ | 광학심도에 의존하지 않는 등방성 속도 분산 — 선폭(damping)과 가우시안 컨볼루션에 사용 / Depth-independent isotropic velocity dispersions entering line damping and Gaussian convolution |
| Modified SVD | 물리량별 민감도 차이를 보정하기 위해 각 자기장/온도 등에 대해 개별적으로 대각 원소 반전을 강제 / Our modification: invert at least one diagonal element per physical magnitude to avoid SVD discarding less-sensitive ones |
| HSRA | Harvard-Smithsonian Reference Atmosphere (Gingerich et al. 1971), 연속광 정규화 기준 / Reference quiet-Sun continuum used to normalize observations |
| Hénoux 1969 umbral model | 시작 모형 초기화 테스트용 참조 흑점 대기 / Reference umbral model used for starting-atmosphere perturbations |

---

## 5. 수식 미리보기 / Equations Preview

**(1) Merit function / 적합도 함수:**
$$\chi^2 = \frac{1}{\nu} \sum_{k=1}^{4} \sum_{i=1}^{M} \left[ I_k^\mathrm{obs}(\lambda_i) - I_k^\mathrm{syn}(\lambda_i) \right]^2$$

Here $\nu$ = degrees of freedom, $k$ indexes Stokes components, $i$ indexes wavelength samples.
여기서 $\nu$ 는 자유도, $k$ 는 스토크스 성분 인덱스, $i$ 는 파장 샘플 인덱스.

**(3) Linearized Stokes variation from perturbation / 섭동에 의한 스토크스 선형 변화:**
$$\delta \mathbf{I}(\lambda) = \int_0^\infty \mathbf{R}(\lambda, \tau) \, \delta x(\tau) \, d\tau$$

**(8) Sánchez Almeida (1992) RF expression / 산체스 알메이다 RF 표현:**
$$\mathbf{R}(\tau, \lambda) = -\mathbf{O}(0, \tau) \left\{ \left[ \frac{\partial \mathbf{K}}{\partial x}\right](\mathbf{I} - \mathbf{S}) - \mathbf{K}\left[\frac{\partial \mathbf{S}}{\partial x}\right] \right\}$$

where $\mathbf{O}(\tau_1, \tau_2)$ is the evolution operator entering the DELO formal solution $\mathbf{I}(\tau) = -\int_{\tau_0}^{\tau} \mathbf{O}(\tau, \tau')\mathbf{K}(\tau')\mathbf{S}(\tau')d\tau' + \mathbf{O}(\tau, \tau_0)\mathbf{I}(\tau_0)$.

**(9) Marquardt linear system / Marquardt 선형계:**
$$\nabla\chi^2(\mathbf{a}) + \mathbf{A}\,\delta\mathbf{a} = \mathbf{0}$$

with curvature matrix $\mathbf{A}$ approximated by pairwise RF products. Diagonal elements are multiplied by $(1 + \lambda)$; $\lambda$ controls interpolation between steepest descent ($\lambda \to \infty$) and Gauss-Newton ($\lambda \to 0$).

**(14) Equivalent RFs at nodes / 노드에서의 등가 RF:**
$$\mathcal{R}_k(\lambda_i, \tau_l) \equiv \Delta\log\tau \cdot \ln 10 \sum_{j=1}^{n} c_j \tau_j f_{j,l} R_k(\lambda_i, \tau_j)$$

where $f_{j,l}$ are cubic-spline interpolation coefficients; these compress the $n$-dimensional RF into $m$ node values ($m \ll n$), without losing whole-atmosphere information — the heart of SIR's efficiency.

---

## 6. 읽기 가이드 / Reading Guide

**EN.**
- **Read §1 and §2 carefully** — the motivation (synthesis vs inversion) and the core mathematical framework (merit function, RFs in Marquardt's equations).
- **Focus on eqs. (3)-(10)**: understand how a single RF evaluation feeds the $\chi^2$ gradient — this is the computational trick that gives SIR its speed.
- **Skim §3's discussion of SVD and Marquardt** unless you plan to reimplement; the key pragmatic point is that the problem is ill-conditioned and needs a modified SVD.
- **In §4-5 compare** the recovered atmospheres in Figures 2-6 against the reference — note how error bars widen outside $\log\tau \in [-2.6, 0.2]$, marking the "information-rich" layer.
- **Appendix A** is essential: the equivalent-RF derivation is the single-most important ingredient enabling node parameterization.
- **Appendix B** explains how standard SVD would fail and the modification needed.

**KR.**
- **§1, §2 를 꼼꼼히 읽기** — 합성 vs 인버전의 동기, 그리고 적합도 함수/RF/Marquardt 수식을 연결하는 핵심 틀.
- **식 (3)-(10) 에 집중** — 한 번의 RF 평가로 $\chi^2$ 경사를 얻는 구조를 이해하면, SIR의 속도 이점이 분명해진다.
- **§3의 SVD/Marquardt 논의**는 재구현 의도가 없다면 개요만 읽어도 된다. 핵심 메시지는 문제가 ill-conditioned이며 수정 SVD가 필요하다는 점.
- **§4-5의 수치실험**에서 Figures 2-6을 참조 대기와 비교하라 — $\log\tau \in [-2.6, 0.2]$ 바깥에서 오차 막대가 커지는 것이 "정보가 살아있는 층"의 경계를 보여준다.
- **부록 A**는 필수 — 등가 RF 유도는 노드 매개변수화를 가능케 하는 핵심 요소.
- **부록 B**는 표준 SVD가 왜 실패하는지, 그리고 어떤 수정이 필요한지를 설명.

---

## 7. 현대적 의의 / Modern Significance

**EN.** SIR became **the de facto standard code for solar magnetic-field inversion** for three decades. Nearly every Hinode/SP, SST/CRISP, GREGOR/GRIS, GST/NIRIS, DKIST/ViSP data reduction pipeline uses SIR or its descendants (SIRJUMP for flux tubes; SIRGAUS; SPINOR in Zurich; NICOLE adding non-LTE; STiC with non-LTE+PRD; HAZEL for chromospheric He I 1083 nm). The paper's **response-function framework is universal**: modern ML inversion (pixel-wise CNNs, deep-learning emulators by Asensio Ramos and others) all fit into the RF paradigm when sensitivity analysis is needed. The **node parameterization** concept proved prescient — it survives in every modern inverter and reappears as "basis coefficients" in machine-learning surrogates. For research using IRIS, Hinode, DKIST, and the upcoming EST (European Solar Telescope), SIR-level understanding is foundational. The 2016 LRSP review by del Toro Iniesta & Ruiz Cobo (Paper #13 LRSP in this reading series) updates and generalizes this 1992 foundation.

**KR.** SIR은 30년간 **태양 자기장 인버전 분야의 사실상 표준 코드**가 되었다. Hinode/SP, SST/CRISP, GREGOR/GRIS, GST/NIRIS, DKIST/ViSP의 거의 모든 데이터 처리 파이프라인이 SIR 또는 그 후손(SIRJUMP, SIRGAUS, 취리히의 SPINOR, 비-LTE를 추가한 NICOLE, 비-LTE+PRD를 포함하는 STiC, 채층의 He I 1083 nm용 HAZEL 등)을 사용한다. **응답함수 프레임워크는 범용적**이다 — 최신 ML 기반 인버전(픽셀별 CNN, Asensio Ramos 그룹 등의 딥러닝 에뮬레이터)도 민감도 분석이 필요할 때는 모두 RF 패러다임에 흡수된다. **노드 매개변수화** 개념은 선견지명이 있었다 — 모든 현대 인버전 코드에 살아남았고 ML 대리모델(surrogate)의 "기저 계수"로도 재등장한다. IRIS, Hinode, DKIST, 그리고 곧 가동될 EST(European Solar Telescope)를 활용한 연구에서 SIR 수준의 이해는 필수다. 본 시리즈의 Paper #13 (LRSP) 인 del Toro Iniesta & Ruiz Cobo 2016 리뷰가 이 1992 논문의 기초를 갱신·확장한다.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
