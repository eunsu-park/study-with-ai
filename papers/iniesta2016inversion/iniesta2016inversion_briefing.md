---
title: "Pre-Reading Briefing: Inversion of the Radiative Transfer Equation for Polarized Light"
paper_id: "51"
topic: Living_Reviews_in_Solar_Physics
date: 2026-04-23
type: briefing
---

# Inversion of the Radiative Transfer Equation for Polarized Light: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: del Toro Iniesta, J. C. & Ruiz Cobo, B., "Inversion of the radiative transfer equation for polarized light", Living Reviews in Solar Physics, 13:4 (2016). DOI: 10.1007/s41116-016-0005-2
**Author(s)**: Jose Carlos del Toro Iniesta (IAA-CSIC, Granada, Spain); Basilio Ruiz Cobo (IAC, Tenerife, Spain)
**Year**: 2016

---

## 1. 핵심 기여 / Core Contribution

이 Living Review는 편광된 빛(polarized light)에 대한 복사전달방정식(RTE, Radiative Transfer Equation)의 역산(inversion) 문제를 체계적으로 리뷰한 가장 종합적인 자료이다. 1970년대 초부터 발전해 온 다양한 inversion 기법—Milne-Eddington (ME), SIR, SPINOR, HAZEL, NICOLE, VFISV, MILOS, ANN, GA, PCA, Bayesian, sparse inversions 등—의 수학적 기초와 실무적 제약을 밝히고, 각 기법에 내재된 가정(LTE/NLTE, 상수/깊이 의존 물리량, weak-field 가정, MISMA 가설 등)을 명시한다. 핵심적 관점은 inversion을 관측 가능량의 공간(Stokes profile의 위상 공간)에서 물리량의 공간으로의 비선형, ill-conditioned 사상(mapping)으로 보는 위상적(topological) 접근이며, Levenberg-Marquardt 알고리즘과 response function(RF)이 어떻게 자연스럽게 이 문제의 핵심 도구가 되는지를 설명한다.

This Living Review is the most comprehensive reference on the inversion of the Radiative Transfer Equation (RTE) for polarized light. It systematically reviews inversion techniques developed since the early 1970s—including Milne-Eddington (ME), SIR, SPINOR, HAZEL, NICOLE, VFISV, MILOS, ANN, GA, PCA, Bayesian, and sparse inversions—laying bare the mathematical foundations and practical limitations of each, and making explicit the assumptions (LTE/NLTE, constant vs. depth-varying quantities, weak-field approximation, MISMA hypothesis, etc.) baked into every method. Its unifying perspective treats inversion as a nonlinear, ill-conditioned mapping between the space of observables (Stokes profiles) and that of physical quantities, with Levenberg-Marquardt optimization and response functions emerging as the natural tools.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

1970년대까지 태양 자기장 진단은 magnetograph (Babcock 1953)에서 한두 개 파장 샘플만을 이용한 longitudinal B 추정에 그쳤다. Unno (1956)가 Milne-Eddington 분위기에서 편광된 RTE의 해석적 해를 제시하고, Rachkovsky (1962, 1967)가 magneto-optical 효과를 포함시킨 이후, Landi Degl'Innocenti의 양자전기역학적 유도(1972)로 편광 RTE 자체가 확립되었다. Auer et al. (1977)이 최초의 본격적 합성 Stokes 기반 inversion을 제안했고, Harvey et al. (1972)의 seminal work에서 관측-이론 맞춤에 의한 B 추정이 시도되었다. 이후 1990년대 SIR (Ruiz Cobo & del Toro Iniesta 1992), SPINOR (Frutiger & Solanki 1998), ASP용 ME inversion (Skumanich & Lites 1987)이 차례로 등장했다. Hinode/SP, SUNRISE/IMaX, SDO/HMI, Solar Orbiter/PHI 등 현대 관측이 4 Stokes 파라미터의 고정밀 고해상도 분광편광측정(spectropolarimetry)을 가능케 하면서 inversion은 태양 대기의 3D 자기-열역학 구조를 추론하는 필수 도구로 자리잡았다.

Until the 1970s, solar magnetic diagnostics relied on magnetographs (Babcock 1953) that yielded only a longitudinal field estimate from one or two wavelength samples. After Unno (1956) provided an analytic Milne-Eddington solution and Rachkovsky (1962, 1967) added magneto-optical (dispersion) effects, Landi Degl'Innocenti's quantum-electrodynamic derivation (1972) firmly established the polarized RTE. Auer et al. (1977) proposed the first genuine synthetic-Stokes inversion, while Harvey et al. (1972) had already pioneered "solving for B by best fit". The 1990s brought SIR (Ruiz Cobo & del Toro Iniesta 1992), SPINOR (Frutiger & Solanki 1998), and the ME code by Skumanich & Lites (1987) for the HAO Advanced Stokes Polarimeter. Modern instruments—Hinode/SP, SUNRISE/IMaX, SDO/HMI, Solar Orbiter/PHI—deliver the high-precision, high-resolution spectropolarimetry of all four Stokes parameters that makes inversion the indispensable tool for reconstructing the 3D magneto-thermodynamic state of the solar atmosphere.

### 타임라인 / Timeline

```
1852 ─ Stokes parameters (G.G. Stokes)
1908 ─ Hale: sunspot magnetic fields (Zeeman)
1956 ─ Unno: Milne-Eddington analytic solution
1962 ─ Rachkovsky: magneto-optical effects
1972 ─ Landi Degl'Innocenti: QED derivation of RTE
1972 ─ Harvey et al.: best-fit B inference
1977 ─ Auer et al.: first inversion method
1984 ─ Landolfi et al.: Florence code
1985 ─ Landi Degl'Innocenti²: formal solution I(0)=∫OKS dτ
1987 ─ Skumanich & Lites: HAO-ASP ME code
1992 ─ Ruiz Cobo & del Toro Iniesta: SIR
1998 ─ Frutiger & Solanki: SPINOR
2000 ─ Socas-Navarro et al.: NICOLE (NLTE)
2007 ─ Orozco Suárez & del Toro Iniesta: MILOS
2008 ─ Asensio Ramos et al.: HAZEL (Hanle + Zeeman)
2011 ─ Borrero et al.: VFISV (SDO/HMI)
2012 ─ van Noort: spatially-coupled inversions
2013 ─ Ruiz Cobo & Asensio Ramos: regularized deconvolution
2015 ─ Asensio Ramos & de la Cruz Rodríguez: sparse inversions
2016 ─ This review
```

---

## 3. 필요한 배경 지식 / Prerequisites

- **RTE (복사전달)**: scalar RTE의 해석적 해(dI/dτ = I - S; Milne-Eddington 선형 source function), Eddington-Barbier 관계
- **Polarization formalism**: Stokes vector I = (I, Q, U, V)ᵀ, Mueller matrix, Pauli-like 4×4 생성자
- **Zeeman effect**: normal/anomalous splitting, Landé factor g, π/σ 성분
- **Hanle effect**: 편광 각도와 편광도의 자기장 의존성 (chromospheric lines에 유용)
- **Line formation**: LTE vs. NLTE, Boltzmann/Saha 법칙, source function, opacity
- **Linear algebra**: eigenvalue problem, SVD, Jacobian/Hessian, 행렬 지수
- **Optimization**: χ² minimization, gradient descent, Gauss-Newton, Levenberg-Marquardt
- **Inverse problems**: ill-posed problem, regularization, Tikhonov, cross-talk/degeneracy
- **Numerical methods**: Voigt/Faraday-Voigt profile, cubic splines, quadrature
- **Solar atmosphere**: photosphere/chromosphere 구조, HSRA/VAL/FAL 모델, flux tubes, granulation
- **Zeeman**: weak-field approx, ΔλB = (λ₀² e₀ B)/(4π m c²)

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| Stokes vector (I, Q, U, V) | 빛의 총 강도 + 3개 편광 성분 / Total intensity + 3 polarization components; I is unpolarized total, Q/U are linear polarization along ±Q/±U axes, V is circular polarization |
| Propagation matrix K | 4×4 흡수 행렬, absorption/pleochroism/dispersion 포함 / 4×4 matrix containing absorption, pleochroism (differential polarization absorption), and dispersion (magneto-optical effects) |
| Source function vector S | 편광된 emissivity vector / Emissivity vector; reduces to (B_ν, 0, 0, 0)ᵀ in LTE |
| Milne-Eddington (ME) atmosphere | 깊이 독립 상수 K와 S = S₀ + S₁τ의 선형 source로 RTE의 해석적 해가 가능 / Constant K and linear source in τ allow analytic I(0) = (S₀ + K⁻¹ S₁) e₀ |
| Unno-Rachkovsky solution | ME 해석해 (Unno 1956, Rachkovsky 1962, 67) with all seven K elements / Analytic ME solution with absorption + dispersion |
| Weak-field approximation | g_eff·ΔλB / ΔλD ≪ 1; Stokes V ∝ -g·ΔλB·cos γ · ∂I/∂λ / Stokes V proportional to longitudinal B times line-profile derivative |
| MISMA | MIcro-Structured Magnetic Atmosphere; photon 평균자유행정 이하 스케일의 stochastic 구조 / Stochastic sub-photon-mean-free-path structuring hypothesized to explain Stokes asymmetries |
| Response function (RF) | R_i(τ) = ∂I/∂x_i(τ): 대기 파라미터 x_i의 광학적 깊이 τ 근방의 단위 섭동에 대한 emergent I의 변화 / Partial derivative of emergent Stokes spectrum w.r.t. atmospheric parameter at given depth |
| Levenberg-Marquardt (LM) | Gauss-Newton과 steepest descent의 적응적 혼합 / Adaptive blend of Gauss-Newton and gradient descent for nonlinear χ² minimization |
| χ² (merit function) | Σ (I_obs - I_syn)² w²; 관측가능공간의 거리 / Observational-space distance, normalized by degrees of freedom |
| SIR | Stokes Inversion based on Response functions, 1992 / SIR code: node-based depth-varying LM inversion with SVD |
| Ambiguity (180°) | Zeeman 관측만으로는 φ와 φ+180°를 구별 못함 / Azimuth from linear polarization has intrinsic 180° ambiguity; resolved externally (divergence-free, Chi-squared minimization with neighbors, Δφ) |
| Response function PSF | RFs는 선형 시스템 이론의 점확산함수 역할 / In linear-system theory, RFs play the role of a PSF relating perturbations to spectral response |

---

## 5. 수식 미리보기 / Equations Preview

**1. Polarized RTE / 편광 복사전달방정식**
$$\frac{d\mathbf{I}}{d\tau_c} = \mathbf{K}(\mathbf{I} - \mathbf{S})$$
Stokes 벡터의 광학적 깊이 미분은 흡수와 emission의 경쟁. The Stokes vector derivative along the LOS is a competition between absorption (K) and emission (S).

**2. Formal integral solution / 형식적 적분 해**
$$\mathbf{I}(0) = \int_0^{\infty} \mathbf{O}(0, \tau_c) \mathbf{K}(\tau_c) \mathbf{S}(\tau_c) \, d\tau_c$$
evolution operator O를 커널로 하는 Fredholm first-kind 적분방정식 (이 때문에 inversion이라 부름). A first-kind Fredholm integral equation with evolution operator O as kernel, the reason this is called "inversion".

**3. Milne-Eddington analytic solution / ME 해석해**
$$\mathbf{I}(0) = (S_0 + \mathbf{K}^{-1} S_1)\mathbf{e}_0, \quad \mathbf{e}_0 = (1,0,0,0)^T$$
constant K와 선형 source (S₀ + S₁τ_c)일 때 경우의 9-parameter 해석해. With constant K and S = S₀ + S₁τ_c, a 9-parameter analytic solution exists.

**4. Weak-field magnetographic equation / 약장 근사**
$$V(\lambda) \simeq -g_{\text{eff}} \Delta\lambda_B \cos\gamma \frac{\partial I_{nm}}{\partial \lambda}$$
$$\Delta\lambda_B = \frac{\lambda_0^2 e_0 B}{4\pi m c^2}$$
약한 자기장 극한에서 Stokes V는 non-magnetic I의 파장 미분에 비례. In the weak-field limit, V is proportional to the wavelength derivative of the non-magnetic I.

**5. Response function / 반응 함수**
$$\delta \mathbf{I}(0) = \sum_i \int_0^{\infty} \mathbf{R}_i(\tau_c) \delta x_i(\tau_c) \, d\tau_c$$
$$\mathbf{R}_i(\tau_c) \equiv \mathbf{O}(0, \tau_c) \left[\mathbf{K}(\tau_c) \frac{\partial \mathbf{S}}{\partial x_i} - \frac{\partial \mathbf{K}}{\partial x_i}(\mathbf{I}-\mathbf{S})\right]$$
물리량 x_i의 각 τ 근처 섭동이 emergent I에 어떻게 전달되는지 결정. Determines how a perturbation in x_i at optical depth τ propagates to the emergent spectrum.

**6. Levenberg-Marquardt update / LM 업데이트**
$$\nabla \chi^2 + \mathbf{H}\delta\mathbf{x} = \mathbf{0}, \quad 2H_{ij} = \begin{cases} H'_{ij}(1+\lambda) & i=j \\ H'_{ij} & i \neq j \end{cases}$$
λ가 크면 steepest descent, 작으면 Gauss-Newton. Large λ → steepest descent, small λ → Gauss-Newton; adaptively tuned per iteration.

---

## 6. 읽기 가이드 / Reading Guide

- **Section 1-2**: inversion이 왜 비선형, ill-conditioned 문제인지 개념적 이해. 저자의 "mapping between observable space and physical space" 관점을 꼭 포착.
- **Section 2 (RTE assumptions)**: NLTE/LTE/ME/weak-field/MISMA의 계층 구조를 파악. Eq. (6)과 (13)이 중심.
- **Section 3 (model atmospheres)**: 상수 물리량 vs. 깊이 의존 물리량의 선택이 code 선택을 결정.
- **Section 4 (Stokes profiles)**: parity (even/odd) 분해가 velocity gradient와 어떻게 연결되는지.
- **Section 5 (synthesis)**: forward problem 해결법 — spectral synthesis와 MHD simulations.
- **Section 6 (Response functions)**: **매우 중요**. Eq. (29)-(34)와 Fig. 16-20을 주의 깊게. RFs는 inversion의 Jacobian 역할.
- **Section 7 (Inversion techniques)**: 대부분 LM 기반. 7.1 (χ²), 7.2 (LM + SVD + nodes), 7.3 (PCA), 7.4 (ANN/GA/Bayes), 7.5 (spatial coupling/sparsity).
- **Section 8 (Discussion)**: 실무 예시. Hinode/SP 관측에 대한 SIR run과 weak-field retrieval의 신뢰도.
- **Appendix**: center-of-gravity + weak-field 초기화. 실제 구현 시 유용.

읽을 때 중점 / Focus while reading:
- Eq. (6), (13), (14), (24), (29), (30), (35)-(42) 반드시 소화.
- RFs가 왜 Hessian을 자연스럽게 제공하는지.
- SIR의 "node" 개념과 SVD 정규화의 필연성.
- weak-field 근사가 실제로 언제 깨지는지 (Fig. 6-7).

Before reading, be sure to internalize Eqs. (6), (13), (14), (24), (29), (30), (35)-(42). Note why RFs naturally furnish Hessian entries, understand the SIR "node" concept and the necessity of SVD regularization, and see in Fig. 6-7 when the weak-field approximation actually fails.

---

## 7. 현대적 의의 / Modern Significance

이 리뷰는 현재 운용 중이거나 곧 운용될 모든 주요 태양 분광편광 관측기—Hinode/SP, SDO/HMI, IRIS, SUNRISE/IMaX, DKIST, Solar Orbiter/PHI—의 데이터 해석 표준을 정의한다. 2016년 이후에도 HMI의 VFISV, Solar Orbiter/PHI의 MILOS, DKIST/ViSP의 SIR 등은 이 review에 기술된 방법론을 직접 사용한다. 또한 machine learning 기반 inversion (ANN의 현대적 확장, Bayesian inversion with normalizing flows, transformer-based spectropolarimetric inference) 역시 여기서 제시된 topological 관점과 response function 이론에 뿌리를 둔다. Sparse inversion과 spatial coupling은 2D/3D 대기 재구성의 길을 열었으며, 2020년대 3D non-LTE inversion 연구(STiC, DeSIRe)의 기반이 되었다. 이 논문은 대학원생과 태양 물리학자 모두에게 "왜 특정 inversion code를 선택해야 하고, 그 결과를 얼마나 신뢰할 수 있는가"에 대한 결정적 참고문헌이다.

This review defines the data-interpretation standard for every major solar spectropolarimeter currently in operation or soon to be—Hinode/SP, SDO/HMI, IRIS, SUNRISE/IMaX, DKIST, Solar Orbiter/PHI. Post-2016 pipelines such as HMI's VFISV, Solar Orbiter/PHI's MILOS, and DKIST/ViSP's SIR directly use the methodology described here. Machine-learning-based inversions (modern ANN extensions, Bayesian inversions with normalizing flows, transformer-based spectropolarimetric inference) likewise root themselves in the topological perspective and response-function theory presented in this paper. Sparse and spatially-coupled inversions opened the door to true 2D/3D atmospheric reconstruction, and the 2020s-era 3D non-LTE inversion codes (STiC, DeSIRe) build directly on these foundations. For graduate students and working solar physicists alike, this paper is the definitive reference for deciding which inversion code to use and how far to trust its output.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
