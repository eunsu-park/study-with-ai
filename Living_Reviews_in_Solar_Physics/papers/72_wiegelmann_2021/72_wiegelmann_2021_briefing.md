---
title: "Pre-Reading Briefing: Solar Force-Free Magnetic Fields"
paper_id: "72"
topic: Living_Reviews_in_Solar_Physics
date: 2026-04-23
type: briefing
---

# Solar Force-Free Magnetic Fields: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Wiegelmann, T. & Sakurai, T., "Solar Force-Free Magnetic Fields", *Living Reviews in Solar Physics*, 18:1 (2021). DOI: 10.1007/s41116-020-00027-4
**Author(s)**: Thomas Wiegelmann (Max-Planck-Institut für Sonnensystemforschung), Takashi Sakurai (National Astronomical Observatory of Japan)
**Year**: 2021

---

## 1. 핵심 기여 / Core Contribution

**Korean:** 이 리뷰 논문은 태양 코로나 자기장을 모델링하는 표준 프레임워크인 "force-free magnetic field" (무력장, 로렌츠 힘이 0인 자기장) 이론과 수치 계산 기법을 포괄적으로 정리한다. 플라즈마 베타(β)가 1보다 훨씬 작은 코로나에서 로렌츠 힘 $\mathbf{j}\times\mathbf{B}=0$ 조건은 자기장과 전류 밀도가 평행함을 의미하며, 이는 비례상수 α(force-free parameter)가 위치의 함수 또는 상수(LFFF vs NLFFF)인지에 따라 선형·비선형 문제로 나뉜다. 저자들은 (1) 포텐셜장 및 선형 force-free장의 해석적 해, (2) Low & Lou(1990), Titov-Démoulin(1999) 등 반해석적 비선형 해, (3) 광구 벡터 자력도의 방위각 모호성 제거 알고리즘, (4) NLFFF 재구성을 위한 5가지 수치 방법(upward integration, Grad-Rubin, MHD relaxation, optimization, boundary element), (5) 코로나 영상에 의해 유도되는 확장 기법(VCA-NLFFF, S-NLFFF), (6) 방법 비교 및 free magnetic energy, helicity 등 도출 물리량의 신뢰성을 체계적으로 다룬다.

**English:** This review paper provides a comprehensive synthesis of the theoretical foundations and numerical techniques for modeling the solar coronal magnetic field under the "force-free" (vanishing Lorentz force) assumption — the dominant paradigm for coronal field reconstruction. Because the coronal plasma beta $\beta = 2\mu_0 p/B^2$ is much less than unity above active regions, magnetic forces dominate and the equilibrium condition reduces to $\mathbf{j}\times\mathbf{B}=0$, equivalent to $\nabla\times\mathbf{B}=\alpha\mathbf{B}$. Depending on whether α is spatially constant (LFFF) or a function of position that is constant along field lines (NLFFF), the mathematical problem ranges from linear Helmholtz-like PDEs to nonlinear mixed elliptic-hyperbolic systems. The review covers (1) analytic potential and LFFF solutions, (2) semi-analytic NLFFF equilibria (Low–Lou, Titov–Démoulin), (3) 180° azimuth ambiguity removal algorithms, (4) five families of numerical extrapolation schemes (upward integration, Grad–Rubin, MHD relaxation, optimization, boundary-element), (5) new approaches using coronal images (VCA-NLFFF, S-NLFFF neural-net methods), and (6) systematic comparisons, limitations (photospheric inconsistency, noise, resolution, finite-β), and derived quantities such as free magnetic energy and magnetic helicity. This is a major revision of the 2012 Living Review, adding ~70 references and six figures to cover the 2012–2020 literature.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

**Korean:** 태양 자기장의 광구 측정은 20세기 초 Hale의 젬만 효과 발견까지 거슬러 올라가지만, 코로나 자기장은 직접 측정이 극히 어려워 반드시 광구 경계조건을 외삽(extrapolation)하여 추정해야 한다. 1950년대 Lundquist와 Chandrasekhar가 force-free 상태를 연구하며 이론 토대를 마련했고, 1958년 Grad & Rubin이 핵융합 플라즈마 맥락에서 수치 접근법을 처음 제시했다. 1970년대 Nakagawa(1974)의 upward integration, 1978년 Seehafer의 LFFF Fourier 해, 1981년 Sakurai의 첫 수치 Grad-Rubin 구현이 이어졌다. 2000년 Wheatland, Sturrock & Roumeliotis가 optimization 방법을 제안하며 현대적 NLFFF 시대가 개막되었고, Hinode(2006)와 SDO/HMI(2010) 등 고품질 광구 벡터 자력도 관측이 가용해지면서 활성영역 3D 재구성이 실용적 단계에 진입했다. 2004년부터 Schrijver 주도의 NLFFF consortium이 방법론 벤치마크를 지속해왔다.

**English:** Photospheric magnetic field measurements date back to Hale's discovery of the Zeeman effect in sunspots (1908), but the coronal magnetic field is essentially impossible to measure directly — so it must be extrapolated from photospheric boundary conditions. The mathematical framework developed gradually: Lundquist (1950) and Chandrasekhar studied force-free states; Grad & Rubin (1958) introduced a well-posed iterative scheme in the fusion-plasma context; Nakagawa (1974) proposed the "upward integration" approach; Seehafer (1978) wrote the Fourier-series LFFF solution; Sakurai (1981) produced the first numerical Grad–Rubin coronal field. The modern NLFFF era opened with Wheatland, Sturrock & Roumeliotis (2000), who introduced the optimization functional. High-quality vector magnetograms from Hinode/SOT (2006) and SDO/HMI (2010) made 3D reconstruction of active regions operationally feasible, and the NLFFF consortium led by Schrijver (since 2004) has run repeated benchmark tests on analytic, synthetic, and real data. This 2021 review is a major update of the original 2012 Living Review, reflecting a decade of SDO data and many new papers.

### 타임라인 / Timeline

```
1908 ── Hale: 흑점의 젬만 효과 / Zeeman effect in sunspots
1950 ── Lundquist: axial-symmetric LFFF (Bessel function) 해
1954 ── Chandrasekhar & Kendall: 구대칭 force-free 장
1958 ── Grad & Rubin: well-posed NLFFF 경계값 문제
1969 ── Schatten et al.: PFSS (potential-field source-surface) 모델
1974 ── Nakagawa: upward integration 방법
1978 ── Seehafer: LFFF Fourier 해 (Cartesian)
1981 ── Sakurai: 첫 수치 Grad-Rubin NLFFF
1984 ── Aly / Sturrock conjecture: open field 에너지 상한
1988 ── Mikić et al.: MHD relaxation 시도
1990 ── Low & Lou: 반해석적 구형 NLFFF
1999 ── Titov & Démoulin: flux-rope 모델
2000 ── Wheatland et al.: optimization 방법
2000 ── Yan & Sakurai: boundary-element 방법
2004 ── NLFFF consortium 시작 (Schrijver 등)
2006 ── Hinode 발사 / Metcalf et al. 모호성 제거 벤치마크
2010 ── SDO/HMI 벡터 자력도 정식 제공
2012 ── Wiegelmann & Sakurai, 첫 Living Review 판
2015 ── VCA-NLFFF (Aschwanden), S-NLFFF (Chifu et al.)
2019 ── 신경망 기반 α 추정 (Benson et al.)
2020 ── Toriumi et al.: data-driven MHD 비교
2021 ── ◆ 본 리뷰 (이 논문) / This review ◆
```

---

## 3. 필요한 배경 지식 / Prerequisites

**Korean:**
- **벡터 해석 & 편미분방정식**: $\nabla\times$, $\nabla\cdot$, 라플라스/헬름홀츠 방정식, 경계값 문제 (elliptic/hyperbolic 혼합형 이해)
- **전자기학**: Maxwell 방정식, Ampère 법칙 $\mathbf{j}=\nabla\times\mathbf{B}/\mu_0$
- **플라즈마 물리**: 플라즈마 베타 $\beta=2\mu_0 p/B^2$, 로렌츠 힘, MHD 기초
- **태양물리**: 광구/채층/코로나 구조, 활성영역, 자속관(flux tube), 자기헬리시티, CME/flare 개요
- **수치해석**: 유한차분법, simulated annealing, 최적화(gradient descent), Fast Fourier Transform
- **선행 논문**: Paper #6 (Parker 1958, solar wind), Paper #31 (Parker 1988, nanoflare — helicity와 free energy 맥락), 태양 자기 재연결 기본 개념

**English:**
- **Vector calculus & PDEs**: curl, divergence, Laplace and Helmholtz equations, mixed elliptic-hyperbolic boundary value problems
- **Electromagnetism**: Maxwell's equations, Ampère's law $\mathbf{j}=\nabla\times\mathbf{B}/\mu_0$
- **Plasma physics**: plasma beta, Lorentz force, MHD basics
- **Solar physics**: photosphere/chromosphere/corona, active regions, flux tubes, magnetic helicity, overview of CME/flare
- **Numerical methods**: finite differences, simulated annealing, gradient descent, FFT
- **Prior papers**: Paper #6 (Parker 1958, solar wind), Paper #31 (Parker 1988, nanoflare — helicity and free-energy context), basics of magnetic reconnection

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| Force-free field / 무력장 | $\mathbf{j}\times\mathbf{B}=0$, 즉 로렌츠 힘이 0인 자기장; $\nabla\times\mathbf{B}=\alpha\mathbf{B}$로 표현. Coronal 플라즈마(β≪1)의 지배 근사. |
| Plasma beta (β) | $\beta = 2\mu_0 p/B^2$, 플라즈마 압력/자기 압력 비. β≪1이면 자기장이 모든 것을 지배 → force-free 근사 유효. |
| Potential field / 포텐셜장 | α=0인 경우: $\mathbf{B}=-\nabla\Phi$, $\nabla^2\Phi=0$. 자유 자기에너지가 없는 최소 에너지 상태. |
| LFFF (Linear force-free field) | α가 공간적으로 상수인 경우. 벡터 헬름홀츠 방정식 $\Delta\mathbf{B}+\alpha^2\mathbf{B}=0$. Seehafer Fourier 해. |
| NLFFF (Nonlinear force-free field) | α가 공간의 함수이지만 자기력선 따라 일정. 혼합형 PDE, 수치 해만 가능. |
| α (force-free parameter) | $\mathbf{j}=\alpha\mathbf{B}/\mu_0$에서 비례상수. $\mathbf{B}\cdot\nabla\alpha=0$ (field line 따라 보존). |
| PFSS (Potential-Field Source-Surface) | 대규모 전역 모델; ~2.5 R☉ source surface에서 자기력선이 방사상이 된다고 가정. |
| Vector magnetogram / 벡터 자력도 | 광구 3성분 ($B_x$, $B_y$, $B_z$) 측정. Stokes profile inversion으로 얻음. |
| 180° azimuth ambiguity / 방위각 모호성 | 횡단 자기장 측정에서 본질적으로 180° 부호 모호성 존재; 제거 알고리즘 필요. |
| Free magnetic energy / 자유 자기에너지 | $E_{\rm free}=E_{\rm NLFFF}-E_{\rm pot}$; flare/CME 구동 가능 에너지. |
| Magnetic helicity | $H_m=\int \mathbf{A}\cdot\mathbf{B}\, dV$; 자기장 꼬임(twist/linkage) 위상적 측도. |
| Virial theorem | $E_{\rm tot}=\frac{1}{\mu_0}\int_S (xB_x+yB_y)B_z\,dxdy$. 경계 데이터만으로 총 에너지 계산. |
| Grad-Rubin method | α를 one polarity에서 field line 따라 전파, Biot-Savart로 B 갱신 반복. Well-posed. |
| Optimization method | Functional $L=\int_V[|(\nabla\times\mathbf{B})\times\mathbf{B}|^2+|\nabla\cdot\mathbf{B}|^2]/B^2\,dV$ 최소화. |
| MHD relaxation | 점성 $\nu$를 갖는 시간의존 MHD 방정식으로 평형 도달. |
| Preprocessing / 전처리 | 힘 있는 광구 데이터를 force-free 근사에 적합하게 조정 (integral relations 최소화). |
| Magneto-hydro-statics (MHS) | Force-free 확장: $\nabla p + \rho\nabla\Psi = \mathbf{j}\times\mathbf{B}$, 플라즈마 압력·중력 포함. |

---

## 5. 수식 미리보기 / Equations Preview

### Force-free condition / 무력장 조건

$$
\mathbf{j}\times\mathbf{B}=0 \iff \nabla\times\mathbf{B}=\alpha\mathbf{B}, \quad \mathbf{B}\cdot\nabla\alpha=0.
$$

로렌츠 힘이 0이려면 전류가 자기장에 평행해야 한다. α는 field line 따라 상수. / The Lorentz force vanishes iff the current density is field-aligned; α is conserved along field lines.

### Potential field / 포텐셜장

$$
\mathbf{B}=-\nabla\Phi, \quad \nabla^2\Phi=0.
$$

α=0인 단순한 경우; Laplace 방정식. / Simplest case with α=0; Laplace equation.

### Linear force-free field Helmholtz equation / LFFF 헬름홀츠 방정식

$$
\Delta\mathbf{B}+\alpha^2\mathbf{B}=0, \quad \alpha=\text{const}.
$$

$\nabla\times(\nabla\times\mathbf{B}=\alpha\mathbf{B})$를 취하고 $\nabla\cdot\mathbf{B}=0$ 적용 시 유도. Fourier 해 가능 (Seehafer 1978). / Derived by taking the curl of the force-free equation; admits Fourier-series solutions.

### Wiegelmann optimization functional / Wiegelmann 최적화 함수

$$
L=\int_V \frac{|(\nabla\times\mathbf{B})\times\mathbf{B}|^2 + |\nabla\cdot\mathbf{B}|^2}{B^2}\,dV.
$$

L=0이면 NLFFF. Functional derivative로 $\partial\mathbf{B}/\partial t = \mu\mathbf{F}$ 갱신 ($\mathbf{F}$에 여러 항 포함). / L=0 implies NLFFF equilibrium; the iteration evolves B along the functional gradient.

### Virial theorem total energy / 비리얼 정리 총 에너지

$$
E_{\rm tot}=\frac{1}{\mu_0}\int_S (xB_x+yB_y)B_z\,dx\,dy.
$$

경계 데이터만으로 총 자기에너지 계산. Force-free 가정의 consistency check로 사용. / Total magnetic energy from boundary integrals alone — used as a consistency check.

---

## 6. 읽기 가이드 / Reading Guide

**Korean:**
1. **§1-2 (Intro, LFFF)**: 플라즈마 베타와 force-free 정의를 확실히 이해하라. Fig. 2 (플라즈마 β 대 높이)와 Seehafer Fourier 해 (Eqs. 18-20)가 핵심.
2. **§3 (Analytic NLFFF)**: Low-Lou와 Titov-Démoulin은 모든 NLFFF 코드의 벤치마크이므로 Eqs. 22-27 꼼꼼히.
3. **§4 (Ambiguity & Consistency)**: Fig. 8의 성능 비교를 훑고, Metcalf(1994) minimum energy 방법 (Eq. 31)과 preprocessing functional (Eqs. 45-49)의 $L_1, L_2$ 항이 force/torque balance라는 점을 이해.
4. **§5 (NLFFF 3D Properties)**: Helicity, Aly-Sturrock conjecture, Shafranov limit $|α|≲1/\ell$ (Eq. 57) — 물리적 직관을 잡는 섹션.
5. **§6 (Numerical methods)** — 이 리뷰의 핵심: 5가지 방법을 순서대로 비교하라.
   - Upward integration (불안정) → Grad-Rubin (안정, well-posed) → MHD relaxation → **Optimization (Wheatland 2000, 현재 가장 널리 사용)** → Boundary-element.
   - Eq. 77 (optimization functional)과 Eqs. 69-75 (MHD relaxation) 비교.
6. **§7 (Comparisons)**: NLFFF consortium 결과가 무엇을 발견했는지 — "일관된 경계조건이 있어야만 NLFFF가 잘 작동한다"는 결론이 핵심.
7. **§8 (MHS, MHD extensions)**: 미래 방향; finite β 효과 처리.

**English:**
1. **§1-2 (Intro, LFFF)**: Solidify your understanding of plasma beta and the force-free definition. Fig. 2 (β vs. height) and the Seehafer Fourier solution (Eqs. 18-20) are key.
2. **§3 (Analytic NLFFF)**: Low-Lou and Titov-Démoulin serve as benchmarks for all NLFFF codes — study Eqs. 22-27 carefully.
3. **§4 (Ambiguity & Consistency)**: Skim the performance comparison in Fig. 8. Understand Metcalf's (1994) minimum-energy method (Eq. 31) and that the $L_1, L_2$ terms in preprocessing (Eqs. 45-49) encode force and torque balance.
4. **§5 (NLFFF 3D Properties)**: Helicity, Aly-Sturrock conjecture, Shafranov-type stability limit $|α| \lesssim 1/\ell$ (Eq. 57) — physical intuition section.
5. **§6 (Numerical methods)** — the core of the review: compare the five methods in order.
   - Upward integration (unstable) → Grad-Rubin (stable, well-posed) → MHD relaxation → **Optimization (Wheatland 2000, currently most widely used)** → Boundary-element.
   - Compare Eq. 77 (optimization functional) with Eqs. 69-75 (MHD relaxation).
6. **§7 (Comparisons)**: What the NLFFF consortium found — "NLFFF only works well with consistent boundary conditions" is the central lesson.
7. **§8 (MHS, MHD extensions)**: Future directions, handling finite-β effects.

---

## 7. 현대적 의의 / Modern Significance

**Korean:** NLFFF 외삽은 현재 우주기상(space weather) 실시간 운영에 이르기까지 핵심 도구이다. SDO/HMI 벡터 자력도가 매 12분 제공되면서 Wiegelmann optimization, Grad-Rubin, MHD relaxation 기반 NLFFF 코드가 일상적으로 활성영역 3D 자기장을 재구성한다. 이로부터 계산되는 free magnetic energy, helicity, twist, shear 등이 flare/CME 예보 지표로 쓰인다. DKIST(2020 운영 시작)의 고해상도 광구 관측과 Solar Orbiter의 공간 다지점 관측은 방위각 모호성 제거 및 보조 측점을 제공하며, 최근 신경망 기반 α 추정(Benson et al. 2019) 및 data-driven MHD 시뮬레이션(Toriumi et al. 2020)이 등장해 NLFFF를 시간의존 물리 모델의 초기 조건으로 통합하는 방향으로 진화하고 있다. 또한 MHS(magnetohydrostatics) 확장은 finite-β 광구/채층까지 모델링 영역을 확장하는 차세대 과제다. 이 리뷰는 이 모든 사조를 한 곳에서 조망할 수 있는 표준 참고문헌으로 자리잡고 있다.

**English:** NLFFF extrapolation is now a cornerstone tool for space-weather operations. With SDO/HMI vector magnetograms available every 12 minutes, optimization, Grad-Rubin, and MHD-relaxation codes routinely reconstruct 3D active-region fields, and derived quantities — free magnetic energy, helicity, twist, shear — feed directly into flare and CME prediction metrics. DKIST's high-resolution photospheric observations (operations starting 2020) and Solar Orbiter's out-of-ecliptic vantage point provide new constraints for ambiguity removal and multi-point NLFFF. Recent machine-learning approaches (Benson et al. 2019 for α estimation) and data-driven MHD simulations (Toriumi et al. 2020) are evolving NLFFF from a static extrapolation into the initial-condition generator for full time-dependent coronal models. The magnetohydrostatic (MHS) extension — covered in §8 — pushes the modeling domain down into the finite-β photosphere and chromosphere. This review is the canonical single-source reference that consolidates all these strands.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
