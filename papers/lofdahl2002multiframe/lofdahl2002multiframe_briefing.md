---
title: "Pre-Reading Briefing: Multi-Frame Blind Deconvolution with Linear Equality Constraints"
paper_id: "32"
topic: Solar_Observation
date: 2026-04-23
type: briefing
---

# Multi-Frame Blind Deconvolution with Linear Equality Constraints: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Löfdahl, M. G. (2002). "Multi-frame blind deconvolution with linear equality constraints." *Proc. SPIE* **4792-21**, Image Reconstruction from Incomplete Data II, Seattle, WA, USA, July 2002. DOI: 10.1117/12.451791
**Author(s)**: Mats G. Löfdahl (Institute for Solar Physics, Royal Swedish Academy of Sciences)
**Year**: 2002

---

## 1. 핵심 기여 / Core Contribution

**English.** Löfdahl unifies a family of seemingly distinct image restoration techniques — Phase Diversity (PD), Phase Diverse Speckle (PDS), plain Multi-Frame Blind Deconvolution (MFBD), and even Shack–Hartmann (SH) wavefront sensing — into a single mathematical framework: MFBD with Linear Equality Constraints (MFBD–LEC). The key insight is that every data-collection scheme simply imposes linear relations on the wavefront expansion coefficients $\alpha_{jm}$. By expressing these as matrix constraints $\mathbf{C}\boldsymbol{\alpha} = \mathbf{d}$ and solving the resulting reduced-dimension optimization via a null-space basis $\mathbf{Q}_2$, one code can handle all scenarios — the "physics" of the experiment enters as data rather than being hard-coded. This opens the door to hybrid data sets (mixed diversity, multi-wavelength, SPDS, polarimetric) that previously demanded custom algorithms.

**Korean.** Löfdahl은 겉보기에 서로 다른 영상 복원 기법들 — 위상 다양성(PD), 위상 다양성 스펙클(PDS), 일반 다중프레임 블라인드 디컨볼루션(MFBD), 심지어 Shack–Hartmann(SH) 파면 센싱까지 — 을 하나의 수학적 틀로 통합한다: 선형 등식 제약조건이 있는 MFBD(MFBD–LEC). 핵심 통찰은 모든 데이터 수집 방식이 파면 확장 계수 $\alpha_{jm}$에 대한 선형 관계로 환원된다는 것이다. 이를 행렬 제약조건 $\mathbf{C}\boldsymbol{\alpha} = \mathbf{d}$로 표현하고 영공간 기저 $\mathbf{Q}_2$를 통해 차원이 축소된 최적화 문제로 풀면, 하나의 코드가 모든 시나리오를 다룰 수 있다 — 실험의 "물리"가 코드가 아니라 데이터로 입력된다. 이는 이전에는 개별 알고리즘이 필요했던 혼합 데이터셋(다양한 다이버시티, 다파장, SPDS, 편광계)까지 처리할 길을 연다.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

**English.** By 2002, ground-based solar imaging was in transition. Speckle methods (Weigelt 1977, Labeyrie 1970) had matured, phase diversity (Gonsalves 1982; Paxman, Schulz & Fienup 1992) had been adapted to the solar photosphere by Löfdahl & Scharmer (1994), and phase-diverse speckle (Paxman, Schulz & Fienup 1992; Paxman et al. 1996) combined the two paradigms. Adaptive Optics (AO) systems for solar work were being commissioned (Scharmer et al. 2000; Rimmele et al.). However, each technique required its own gradient/Hessian derivations and its own software — a heavy burden when new data-collection schemes (mixed wavelengths, polarization switching, SH combined with high-resolution imagery) were proliferating. Löfdahl's unifying LEC framework arrived just as the Swedish Solar Telescope (SST, 1m, first light 2002) was coming online, promising 0.1-arcsec resolution imagery that would stress existing restoration pipelines.

**Korean.** 2002년 당시 지상 기반 태양 관측은 전환기에 있었다. 스펙클 방법(Weigelt 1977, Labeyrie 1970)은 성숙했고, 위상 다양성(Gonsalves 1982; Paxman, Schulz & Fienup 1992)은 Löfdahl & Scharmer(1994)에 의해 태양 광구에 적용되었으며, 위상 다양성 스펙클(Paxman, Schulz & Fienup 1992; Paxman 등 1996)은 두 패러다임을 결합했다. 태양용 적응 광학계(AO)도 운용되기 시작했다(Scharmer 등 2000; Rimmele 등). 그러나 각 기법은 고유의 경사도/헤시안 유도와 전용 소프트웨어가 필요했고, 새로운 데이터 수집 방식(다파장 혼합, 편광 스위칭, 고해상도 영상과 결합된 SH)이 늘어나면서 이는 큰 부담이었다. Löfdahl의 통합 LEC 프레임워크는 SST(1m, 2002년 초기 관측) 가동을 앞두고 등장했으며, 0.1-arcsec 해상도 영상은 기존 복원 파이프라인을 한계까지 시험할 것이었다.

### 타임라인 / Timeline

```
1970    Labeyrie — speckle interferometry
1977    Weigelt — speckle masking
1982    Gonsalves — phase diversity for AO (one pair of images)
1990    Roddier — Zernike atmospheric wavefront simulation
1992    Paxman, Schulz & Fienup — joint object/aberration PD theory; PDS introduced
1993    Schulz — MFBD for astronomical images
1994    Löfdahl & Scharmer — PD adapted to solar photosphere
1996    Paxman et al. — PDS evaluated for solar imaging
1998    Vogel, Chan, Plemmons — fast PD algorithms with regularization
2000    Löfdahl, Scharmer & Wei — deformable mirror calibration by PD
2001    Löfdahl, Berger & Seldin — multi-wavelength PDS sequences
2002    *** THIS PAPER — MFBD–LEC unifying framework ***
2002    Tritschler & Schmidt — sunspot PD photometry
2002    SST 1m first light
2005+   MOMFBD (van Noort, Rouppe van der Voort, Löfdahl 2005) — full implementation
```

---

## 3. 필요한 배경 지식 / Prerequisites

**English.**
- **Fourier optics**: Generalized pupil function $P = A e^{i\phi}$, point spread function $s = |\mathcal{F}^{-1}\{P\}|^2$, optical transfer function (OTF) as Fourier transform of PSF.
- **Image formation (isoplanatic)**: $d_j = f * s_j + n_j$ in real space, $D_j = F \cdot S_j + N_j$ in Fourier space.
- **Wavefront expansion**: Zernike polynomials and Karhunen–Loève modes for Kolmogorov turbulence statistics.
- **Maximum likelihood estimation** under Gaussian noise; the closed-form Wiener-filter object estimate.
- **Numerical linear algebra**: QR/SVD factorization, null space of a matrix, reduced Newton/Gauss–Newton methods.
- **Constrained optimization**: elimination via null-space parameterization (from Kahaner–Moler–Nash textbook).
- **Prior papers in this track** (#20, #21 on seeing/AO basics): statistical description of atmospheric turbulence and tilt correction.

**Korean.**
- **푸리에 광학**: 일반화된 동공 함수 $P = A e^{i\phi}$, 점 확산 함수 $s = |\mathcal{F}^{-1}\{P\}|^2$, PSF의 푸리에 변환인 광학 전달 함수(OTF).
- **영상 형성 (등평면)**: 실공간 $d_j = f * s_j + n_j$, 푸리에 공간 $D_j = F \cdot S_j + N_j$.
- **파면 확장**: Kolmogorov 난류 통계를 위한 Zernike 다항식과 Karhunen–Loève 모드.
- **Gaussian 잡음 가정하 최대 우도 추정**과 폐쇄형 Wiener 필터 물체 추정.
- **수치 선형대수**: QR/SVD 분해, 행렬의 영공간, 축소 뉴턴/가우스–뉴턴법.
- **제약조건이 있는 최적화**: 영공간 모수화를 통한 제거(Kahaner–Moler–Nash 교재).
- **이 트랙의 선행 논문**(#20, #21 시잉/AO 기초): 대기 난류의 통계적 기술과 틸트 보정.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| MFBD | Multi-Frame Blind Deconvolution — joint estimation of a single object and multiple PSFs from a series of degraded frames / 다중 프레임에서 하나의 물체와 여러 PSF를 공동 추정 |
| PD | Phase Diversity — adds intentionally defocused frames so the algorithm has more information on aberrations / 의도적으로 초점을 벗어난 프레임을 추가해 수차 정보를 늘리는 기법 |
| PDS | Phase-Diverse Speckle — PD applied to many short-exposure realizations of seeing / 다수의 단시간 노출 시잉 실현에 적용된 PD |
| SH WFS | Shack–Hartmann wavefront sensor — microlens array creates sub-images each sampling a pupil region / 마이크로렌즈 배열이 동공 영역별 부-영상을 생성 |
| LEC | Linear Equality Constraint — relation $\sum_j c_j \alpha_{jm} = d$ among wavefront coefficients / 파면 계수 간 선형 제약조건 |
| Generalized pupil | $P = A e^{i\phi}$; amplitude mask $A$ and wavefront phase $\phi$ / 진폭 마스크와 파면 위상 |
| Zernike polynomial | Orthogonal basis on unit disk; standard decomposition for optical aberrations / 단위 원판 상의 직교 기저 |
| Karhunen–Loève (KL) mode | Orthogonal basis diagonalising the Kolmogorov covariance; optimal for atmospheric phase / Kolmogorov 공분산을 대각화하는 기저; 대기 위상에 최적 |
| Null space ($\mathbf{Q}_2$) | Subspace where $\mathbf{C}\mathbf{x} = 0$; parameterizes constrained solutions / $\mathbf{C}\mathbf{x} = 0$인 부분공간; 제약 해를 모수화 |
| Wiener filter | MMSE linear filter; closed-form object estimate given OTFs / MMSE 선형 필터; OTF가 주어진 물체의 폐쇄형 추정 |
| Isoplanatic patch | Region of the sky over which the PSF is approximately constant / PSF가 근사적으로 일정한 영역 |
| Diversity channel | An imaging channel whose phase differs from the reference by a known (or structurally similar) amount / 기준 대비 위상이 알려진 양만큼 다른 채널 |

---

## 5. 수식 미리보기 / Equations Preview

**(1) Generalized pupil / 일반화 동공 함수**

$$
P_j = A_j \exp\{i\phi_j\}
$$

**English.** Complex-valued pupil: binary amplitude mask $A_j$ times phase factor $e^{i\phi_j}$. PSF then $s_j = |\mathcal{F}^{-1}\{P_j\}|^2$.
**Korean.** 복소 동공: 이진 진폭 마스크 $A_j$와 위상 인자 $e^{i\phi_j}$의 곱. PSF는 $s_j = |\mathcal{F}^{-1}\{P_j\}|^2$.

**(2) Phase expansion / 위상 확장**

$$
\phi_j = \theta_j + \sum_{m=1}^{M} \alpha_{jm}\psi_m
$$

**English.** $\theta_j$ is a known part (e.g., diversity defocus), $\psi_m$ are basis functions (KL or Zernike), $\alpha_{jm}$ are unknowns to estimate.
**Korean.** $\theta_j$는 알려진 부분(예: 다이버시티 디포커스), $\psi_m$은 기저 함수(KL 또는 Zernike), $\alpha_{jm}$은 추정할 미지수.

**(3) ML metric (Löfdahl Eq. 6) / 최대 우도 척도**

$$
L(\boldsymbol\alpha) = \sum_u \!\left[\sum_{j=1}^{J} |D_j|^2 - \frac{\left|\sum_j D_j^{*} S_j\right|^2}{Q}\right] + \frac{\gamma_{\rm wf}}{2}\sum_m \frac{1}{\lambda_m}\sum_j|\alpha_{jm}|^2
$$

**English.** The object $F$ has been eliminated via Wiener filter, leaving a metric only in the wavefront coefficients. $Q = \gamma_{\rm obj} + \sum_j |S_j|^2$ stabilises the inversion; $\gamma_{\rm wf}/\lambda_m$ is the mode-variance-aware wavefront prior.
**Korean.** 물체 $F$는 Wiener 필터로 제거되어 척도는 파면 계수만의 함수가 된다. $Q = \gamma_{\rm obj} + \sum_j |S_j|^2$은 역변환을 안정화하고, $\gamma_{\rm wf}/\lambda_m$은 모드 분산을 반영한 파면 사전분포.

**(4) Linear equality constraint / 선형 등식 제약**

$$
\mathbf{C}\cdot\boldsymbol{\alpha} - \mathbf{d} = 0
$$

**English.** All coupling between frames — PD, PDS, SH, polarization, dual-wavelength — expressed as one or more rows of $\mathbf{C}$. Solutions form affine subspace $\boldsymbol\alpha = \bar{\boldsymbol\alpha} + \mathbf{Q}_2\boldsymbol\beta$.
**Korean.** 프레임 간 모든 결합(PD, PDS, SH, 편광, 이중 파장)이 $\mathbf{C}$의 행으로 표현된다. 해는 아핀 부분공간 $\boldsymbol\alpha = \bar{\boldsymbol\alpha} + \mathbf{Q}_2\boldsymbol\beta$.

**(5) Reduced normal equations / 축소 정규방정식**

$$
\mathbf{Q}_2^{\top}\mathbf{A}^{\rm MFBD}\mathbf{Q}_2\cdot\delta\boldsymbol\beta - \mathbf{Q}_2^{\top}\mathbf{b}^{\rm MFBD} \simeq 0
$$

**English.** Constrained minimization becomes an unconstrained one in the lower-dimensional $\boldsymbol\beta$. The $\mathbf{Q}_2$ matrix comes from a QR (or SVD) factorization of $\mathbf{C}^{\top}$.
**Korean.** 제약 최적화는 낮은 차원의 $\boldsymbol\beta$에 대한 무제약 최적화로 변환된다. $\mathbf{Q}_2$는 $\mathbf{C}^{\top}$의 QR(또는 SVD) 분해에서 얻는다.

---

## 6. 읽기 가이드 / Reading Guide

**English.**
1. **Sections 1–2.1**: Pay attention to how the Wiener-filter object estimate reduces the problem to one in $\alpha$ only. Note the two regularization parameters $\gamma_{\rm obj}$ and $\gamma_{\rm wf}$, and the role of KL variances $\lambda_m$.
2. **Sections 2.2–2.3**: Compare the PD gradient/Hessian (Eqs. 8–10) with the MFBD versions (Eqs. 13–16); notice the block-diagonal structure and the $\sqrt{J}$, $J$ factors that arise from splitting $\alpha_m \mapsto \alpha_{jm}$.
3. **Section 3.1**: This is the conceptual heart — constrained optimization via null-space basis $\mathbf{Q}_2$. Keep Eqs. (17)–(21) handy.
4. **Sections 3.2–3.4**: Work through the sample constraint matrices in Figs. 1–4. Try to predict the null-space structure before reading the explanation.
5. **Section 3.5 (SH)**: Notice how SH is just MFBD with different $A_j$ for each imaging channel — a beautiful re-interpretation.
6. **Section 4**: Read as a menu of future applications (multi-object, multi-wavelength, variable $K$).
7. **Section 5 (Discussion)**: Focus on the future-direction comment about finding sparser $\mathbf{Q}_2$.

**Korean.**
1. **1–2.1절**: Wiener 필터 물체 추정이 어떻게 문제를 $\alpha$만의 것으로 축소하는지 주목. 두 개의 정규화 모수 $\gamma_{\rm obj}$, $\gamma_{\rm wf}$와 KL 분산 $\lambda_m$의 역할에 주목.
2. **2.2–2.3절**: PD 경사도/헤시안(식 8–10)과 MFBD 버전(식 13–16)을 비교하고 블록 대각 구조, $\alpha_m \mapsto \alpha_{jm}$ 분할에서 생기는 $\sqrt{J}$, $J$ 인자를 살펴볼 것.
3. **3.1절**: 개념의 핵심 — 영공간 기저 $\mathbf{Q}_2$를 통한 제약 최적화. 식 (17)–(21)을 가까이 두고 읽을 것.
4. **3.2–3.4절**: 그림 1–4의 샘플 제약 행렬을 따라가기. 설명을 읽기 전에 영공간 구조를 먼저 예측해 볼 것.
5. **3.5절 (SH)**: SH가 각 채널마다 다른 $A_j$를 가진 MFBD에 불과하다는 재해석의 아름다움.
6. **4절**: 향후 응용의 메뉴(다물체, 다파장, 가변 $K$)로 읽을 것.
7. **5절 (토론)**: 더 성긴 $\mathbf{Q}_2$를 찾는 향후 방향 논평에 집중.

---

## 7. 현대적 의의 / Modern Significance

**English.** This 2002 formulation is the theoretical skeleton of MOMFBD (Multi-Object Multi-Frame Blind Deconvolution) — now the workhorse post-processing pipeline at SST, CRISP/CHROMIS, GREGOR, DKIST, and for simulated DKIST/SUNRISE data. The unified code base drastically lowered the barrier to combining new observables (spectro-polarimetric scans, narrow-band filtergrams, tip-tilt aux data). Today every diffraction-limited ground-based solar image in the literature — sub-0.1" granulation, penumbra filaments, chromospheric fibrils — has very likely passed through a descendant of the MFBD–LEC algorithm. The paper's deeper message, that the "physics of the data collection" belongs in a constraint matrix rather than in hard-coded gradient expressions, also anticipates modern probabilistic programming and differentiable-forward-model approaches in computational imaging.

**Korean.** 2002년의 이 수식화는 MOMFBD(다물체 다중프레임 블라인드 디컨볼루션)의 이론적 골격으로, 현재 SST, CRISP/CHROMIS, GREGOR, DKIST 및 시뮬레이션된 DKIST/SUNRISE 데이터에서 주력 후처리 파이프라인이 되었다. 통합 코드 기반은 새로운 관측량(분광편광 스캔, 협대역 필터그램, 틸트 보조 데이터)을 결합하는 진입 장벽을 급격히 낮췄다. 오늘날 문헌에 등장하는 회절 한계 지상 태양 영상(0.1" 이하 입상, 반영 필라멘트, 채층 피브릴)은 거의 모두 MFBD–LEC 후손 알고리즘을 거쳤을 가능성이 매우 높다. "데이터 수집의 물리"가 하드코딩된 경사도 식이 아니라 제약 행렬에 속한다는 이 논문의 심층적 메시지는 현대의 확률적 프로그래밍과 계산 영상에서의 미분 가능 전방 모델 접근법도 예견한다.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
