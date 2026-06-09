---
title: "Pre-reading Briefing: Solar Force-Free Magnetic Fields (Wiegelmann & Sakurai 2012)"
date: 2026-04-27
topic: Solar_Physics
paper_number: 35
authors: Thomas Wiegelmann, Takashi Sakurai
year: 2012
journal: Living Reviews in Solar Physics
doi: 10.12942/lrsp-2012-5
tags: [force-free, NLFFF, coronal-magnetic-field, extrapolation, optimization, vector-magnetogram, review]
---

# Pre-reading Briefing / 사전 읽기 브리핑

## Why This Paper Matters / 이 논문이 중요한 이유

**English.** The solar corona's structure and dynamics are governed almost entirely by the magnetic field, yet routine high-accuracy magnetic measurements only exist at the photosphere. To understand the 3D coronal field, we must extrapolate upward from photospheric vector magnetograms under suitable assumptions. The dominant assumption — justified because the corona is low-$\beta$ — is the *force-free* condition $\mathbf{j}\times\mathbf{B}=\mathbf{0}$, which leads to $\nabla\times\mathbf{B}=\alpha(\mathbf{r})\mathbf{B}$. This Living Reviews article by Wiegelmann & Sakurai provides the canonical, comprehensive survey of force-free field theory and the numerical methods (potential, linear, nonlinear) used to reconstruct coronal magnetic fields from observations. It is a one-stop reference that defines the language, equations, ambiguity-removal pipelines, and benchmark tests every modern coronal-field study cites.

**한국어.** 태양 코로나의 구조와 동역학은 거의 전적으로 자기장에 의해 지배되지만, 정밀 자기장 관측은 광구에서만 가능합니다. 따라서 3차원 코로나 자기장을 알기 위해서는 광구 벡터 자기도를 경계조건으로 위쪽으로 외삽해야 하며, 코로나의 낮은 플라즈마 베타($\beta\ll 1$) 덕분에 *무력장(force-free)* 가정 $\mathbf{j}\times\mathbf{B}=\mathbf{0}$, 즉 $\nabla\times\mathbf{B}=\alpha(\mathbf{r})\mathbf{B}$ 형태가 채택됩니다. Wiegelmann & Sakurai의 본 Living Reviews 논문은 무력장 이론과 코로나 자기장 재구성을 위한 수치적 방법(potential, linear FF, nonlinear FF)을 종합적으로 정리한 표준 참고서입니다. 본 논문은 현대 코로나 자기장 연구의 언어, 방정식, ambiguity 제거 절차, 벤치마크 테스트의 기준을 제공합니다.

## Prerequisites / 선수 지식

| Topic | Why needed (EN) | 필요한 이유 (KR) |
|-------|-----------------|------------------|
| Maxwell's equations (magnetostatics) | Derivation of $\nabla\times\mathbf{B}=\mu_0\mathbf{j}$, $\nabla\cdot\mathbf{B}=0$ | $\mathbf{B}$ 장의 기본 방정식 |
| Plasma $\beta$ concept | Justifies dropping pressure/gravity terms | 비자기력 무시 정당화 |
| Vector calculus (curl, divergence, Laplacian) | Force-free PDEs | 무력장 편미분방정식 풀이 |
| Boundary value problems (elliptic/hyperbolic) | NLFFF system is mixed-type | NLFFF는 elliptic-hyperbolic 혼합형 |
| Fourier analysis / Green's functions | Potential & LFFF solutions | 퍼텐셜 및 선형무력장 해법 |
| Magnetogram observations (Stokes profiles, Zeeman effect) | Origin of vector $\mathbf{B}$ data | 벡터 자기도 측정 원리 |
| Optimization (gradient descent / variational) | Wiegelmann's NLFFF approach | NLFFF 최적화 기법 |
| Prior papers #10 (Parker), #33 (Schou et al. HMI) | Coronal heating & HMI instrument context | 코로나 가열 및 HMI 관측 맥락 |

## Historical Context / 역사적 맥락

**English.** The force-free idea dates to Lust & Schlüter (1954) and Chandrasekhar & Kendall (1957). Lundquist (1950) gave the constant-$\alpha$ Bessel-function solution for an infinite cylinder; Low & Lou (1990) provided the standard axisymmetric NLFFF benchmark; Titov & Démoulin (1999) introduced the flux-rope equilibrium widely used to test stability. The modern era of 3D NLFFF extrapolation began with Sakurai (1981) boundary-element work, the Grad–Rubin scheme (Amari et al. 1997, 2006), the optimization principle (Wheatland, Sturrock & Roumeliotis 2000) further developed by Wiegelmann (2004), and MHD relaxation / magnetofrictional methods (Yang, Sturrock & Antiochos 1986; van Ballegooijen 2004). The launches of Hinode/SP (2006) and SDO/HMI (2010) made full-disk vector magnetograms routinely available, intensifying the demand for robust NLFFF codes — which the NLFFF Consortium (Schrijver et al. 2006, 2008; DeRosa et al. 2009) benchmarked side by side.

**한국어.** 무력장 개념은 Lust & Schlüter (1954)와 Chandrasekhar & Kendall (1957)에서 출발했으며, Lundquist (1950)은 무한 원통에서의 상수-$\alpha$ Bessel 함수 해를 제시했습니다. Low & Lou (1990)는 표준적인 축대칭 NLFFF 벤치마크를, Titov & Démoulin (1999)은 안정성 테스트에 자주 쓰이는 flux-rope 평형 모델을 제시했습니다. 3D NLFFF 외삽의 현대기는 Sakurai (1981)의 boundary-element 방법, Grad–Rubin 방식 (Amari 등 1997, 2006), Wheatland 등 (2000)이 제안하고 Wiegelmann (2004)이 발전시킨 최적화 방법, 그리고 MHD relaxation/magnetofrictional 방법 (Yang 등 1986; van Ballegooijen 2004)으로 확립되었습니다. Hinode/SP (2006)와 SDO/HMI (2010)의 발사로 풀-디스크 벡터 자기도가 상시 제공되면서 NLFFF 코드의 견고성에 대한 수요가 폭증했고, NLFFF Consortium (Schrijver 등 2006, 2008; DeRosa 등 2009)이 다양한 코드를 벤치마크했습니다.

## Key Vocabulary / 핵심 용어

| Term | Definition (EN) | 정의 (KR) |
|------|-----------------|-----------|
| Force-free field | $\mathbf{j}\times\mathbf{B}=\mathbf{0}$, equivalently $\nabla\times\mathbf{B}\parallel\mathbf{B}$ | 자기력선과 전류가 평행하여 Lorentz 힘이 0 |
| Plasma $\beta$ | $\beta=2\mu_0 p/B^2$, ratio of gas to magnetic pressure | 가스압 대 자기압 비 |
| Potential field | $\nabla\times\mathbf{B}=0$, $\mathbf{B}=-\nabla\phi$, $\Delta\phi=0$ | 무전류 자기장; Laplace 방정식 |
| Linear FFF (LFFF) | $\alpha=$ const everywhere; vector Helmholtz equation | $\alpha$가 공간상에서 상수인 무력장 |
| Nonlinear FFF (NLFFF) | $\alpha=\alpha(\mathbf{r})$ varying along/across field lines | $\alpha$가 자기력선에 따라 변하는 무력장 |
| $\alpha$ (force-free parameter) | Constant along each field line: $\mathbf{B}\cdot\nabla\alpha=0$ | 자기력선 위에서 일정한 비틀림 모수 |
| Vector magnetogram | $(B_x,B_y,B_z)$ map at the photosphere | 광구의 3차원 자기장 측정도 |
| 180° azimuth ambiguity | Stokes-Q,U give $B_\perp$ direction modulo 180° | 횡자기장 방향이 180°로 모호 |
| Preprocessing | Conditioning photospheric data toward force-free consistency | 광구 데이터를 무력장 호환으로 조정 |
| Grad–Rubin method | Iterative solver: alternate update of $\alpha$ and $\mathbf{B}$ | $\alpha$와 $\mathbf{B}$를 교대 갱신 |
| Optimization (Wiegelmann) | Minimize $L=\int(B^{-2}|(\nabla\times\mathbf{B})\times\mathbf{B}|^2 + |\nabla\cdot\mathbf{B}|^2)\,dV$ | 무력장 위반량 적분의 최소화 |
| MHD relaxation / magnetofrictional | Pseudo-time evolution toward $\mathbf{j}\times\mathbf{B}=0$ | 가상 시간 진화로 무력장에 수렴 |
| Lundquist solution | Constant-$\alpha$ Bessel-function flux rope | Bessel 함수 형태의 비틀린 자속관 |
| Low–Lou solution | Axisymmetric NLFFF benchmark from Grad–Shafranov | 축대칭 NLFFF 벤치마크 |
| Titov–Démoulin equilibrium | Toroidal current ring equilibrium | 토로이드 전류환 평형 모델 |
| PFSS | Potential-Field Source-Surface global model | 글로벌 퍼텐셜장 모델 |

## Q&A / 질문과 답변

### Q1. Why is the corona "force-free" but the photosphere is not? / 왜 코로나는 무력장이고 광구는 그렇지 않은가?
**EN.** The plasma $\beta$ parameter is $\sim 1$ at the photosphere and rises again in the upper corona, but in between (chromosphere → low corona, $\sim 2$–$2000$ Mm) $\beta\ll 1$. There the magnetic pressure dominates pressure gradient, gravity, and inertia, so to leading order the Lorentz force must vanish to maintain quasi-static balance. At the photosphere $\beta\sim 1$ and pressure gradients/gravity matter; thus measured photospheric fields are *not* exactly force-free, motivating the "preprocessing" step.

**KR.** 플라즈마 베타는 광구에서 약 1이고 상부 코로나에서 다시 1 이상이지만 그 사이의 채층-저코로나 영역(약 2~2000 Mm)에서는 $\beta\ll 1$입니다. 이 영역에서는 자기압이 압력 기울기·중력·관성보다 압도적이므로 정역학 균형을 위해 Lorentz 힘이 0에 가까워야 합니다. 광구에서는 $\beta\sim 1$이라 무력장 가정이 정확히 성립하지 않으므로, 측정된 자기도를 무력장 호환으로 만들어주는 preprocessing 과정이 필요합니다.

### Q2. What is the 180° azimuth ambiguity and why is it a big deal? / 180° azimuth ambiguity란? 왜 중요한가?
**EN.** Linear polarization Stokes Q, U give $B_\perp$ direction modulo 180° (a flip leaves Q, U unchanged). Uncorrected, the ambiguity yields wrong vertical currents $J_z=\partial_x B_y-\partial_y B_x$ and wrong $\alpha=\mu_0 J_z/B_z$, ruining the NLFFF boundary. Algorithms like the acute-angle method (vs. potential), pseudo-current method, structure minimization, and minimum-energy method (Metcalf 1994) are surveyed; minimum-energy is generally the most reliable but expensive.

**KR.** 선형 편광(Stokes Q, U)은 횡자기장 $B_\perp$의 방향을 180° 모호성을 가지고 줍니다(180° 뒤집어도 Q, U 동일). 이 모호성을 풀지 않으면 수직 전류 $J_z=\partial_x B_y-\partial_y B_x$와 $\alpha=\mu_0 J_z/B_z$가 잘못 계산되어 NLFFF 경계조건이 망가집니다. 본 리뷰는 acute-angle 방법(퍼텐셜과 비교), pseudo-current 방법, structure minimization, minimum-energy 방법(Metcalf 1994) 등을 다루며, minimum-energy가 가장 신뢰할 만하지만 계산비용이 큽니다.

### Q3. Why not just use linear force-free with a single $\alpha$? / 왜 단일 $\alpha$로 충분하지 않은가?
**EN.** Empirically (e.g., Pevtsov, Wheatland), different field lines in the same active region have different best-fit $\alpha$ — a clear contradiction to LFFF's premise. Marsch et al. (2004) found a single $\alpha$ fit some loops; Wiegelmann & Neukirch (2002) found scatter incompatible with LFFF. Thus LFFF gives only crude global estimates of twist (~15%) and loop heights (~5%); it misses concentrated currents and free energy.

**KR.** Pevtsov, Wheatland 등이 같은 활동영역 내 자기력선들마다 최적 $\alpha$ 값이 다름을 보였는데, 이는 LFFF 가정과 모순입니다. Marsch 등(2004)은 단일 $\alpha$가 일부 loop에 잘 맞다고 보고했지만, Wiegelmann & Neukirch (2002)는 LFFF와 일관되지 않는 산포를 발견했습니다. 따라서 LFFF는 비틀림(~15%)과 loop 높이(~5%) 정도만 거칠게 추정 가능하고 집중 전류·자유 에너지를 놓칩니다.

### Q4. What are the main NLFFF numerical methods and how do they differ? / 주요 NLFFF 수치 기법과 차이점은?
**EN.** (i) **Upward integration** — directly integrates $\nabla\times\mathbf{B}=\alpha\mathbf{B}$ vertically; numerically unstable (mixed-type PDE). (ii) **Grad–Rubin** — alternately updates $\alpha$ along characteristics and $\mathbf{B}$ via Biot–Savart; uses one polarity's $\alpha$. (iii) **MHD relaxation / magnetofrictional** — evolves $\partial_t\mathbf{B}=\nabla\times(\mathbf{v}\times\mathbf{B})$ with $\mathbf{v}\propto\mathbf{j}\times\mathbf{B}$ to suppress force. (iv) **Optimization (Wiegelmann)** — minimizes the volume integral of force-free violations and divergence. (v) **Boundary-element / Green's** (Yan & Sakurai 2000) — surface integral formulation. The NLFFF Consortium benchmarks show consistent solutions exist but require carefully prepared boundaries and large enough volumes.

**KR.** (i) **Upward integration**: $\nabla\times\mathbf{B}=\alpha\mathbf{B}$를 위로 직접 적분, 수치적 불안정. (ii) **Grad–Rubin**: $\alpha$를 자기력선을 따라, $\mathbf{B}$를 Biot–Savart로 교대 갱신, 한 극성의 $\alpha$만 사용. (iii) **MHD relaxation / magnetofrictional**: $\partial_t\mathbf{B}$를 진화시키며 $\mathbf{v}\propto\mathbf{j}\times\mathbf{B}$로 힘을 0으로 유도. (iv) **Optimization (Wiegelmann)**: 무력장 위반량과 발산의 부피 적분을 최소화. (v) **Boundary-element/Green** (Yan & Sakurai 2000): 표면 적분식. NLFFF Consortium 벤치마크는 신중한 경계조건과 충분히 큰 볼륨에서만 일관된 해가 나옴을 보였습니다.

### Q5. What is "preprocessing" of vector magnetograms? / 벡터 자기도 "preprocessing"이란?
**EN.** Wiegelmann, Inhester & Sakurai (2006) — minimize a functional that drives the photospheric data toward (a) zero net force ($\sum B_x B_z = \sum B_y B_z = 0$), (b) zero net torque, (c) smoothness, and (d) closeness to observation. The result is a slightly modified, force-free-consistent boundary that NLFFF codes can integrate from, mimicking a chromospheric (low-$\beta$) layer.

**KR.** Wiegelmann, Inhester & Sakurai (2006)이 제안한 절차로, 광구 자기도 데이터를 (a) 알짜 힘이 0 ($\sum B_x B_z=\sum B_y B_z=0$), (b) 알짜 토크가 0, (c) 매끄러움, (d) 관측에 근접 — 의 네 조건을 만족하도록 약간 수정합니다. 결과적으로 NLFFF 코드가 출발점으로 쓰기 적합한 채층-수준($\beta\ll 1$) 경계조건이 얻어집니다.

### Q6. Why measure the free magnetic energy? / 자유 자기 에너지를 왜 측정하는가?
**EN.** $E_{\rm free}=E_{\rm NLFFF}-E_{\rm potential}$ is the upper bound on energy releasable in flares & CMEs (Aly–Sturrock conjecture). Reliable NLFFF gives observational predictions of eruption budgets — critical for space weather forecasting. The corona's free energy is small ($\lesssim 50\%$ of potential), so accurate NLFFF requires very small relative errors, hence the need for benchmarks.

**KR.** $E_{\rm free}=E_{\rm NLFFF}-E_{\rm potential}$은 플레어·CME에서 방출 가능한 에너지의 상한 (Aly–Sturrock 추측)입니다. 신뢰할 만한 NLFFF는 폭발 에너지 예산을 관측 기반으로 추정하게 해 우주기상 예보의 핵심 자료가 됩니다. 자유 에너지는 퍼텐셜장의 50% 이하 정도라 작아서 정확한 NLFFF는 매우 작은 상대 오차를 요구하며, 그래서 벤치마크 테스트가 필수입니다.

## Reading Strategy / 읽기 전략

**English.** (1) Skim Sec. 1 for motivation and the $\beta$-vs-height figure. (2) Master Sec. 2 (LFFF) — the Seehafer Fourier method and Helmholtz equation give the cleanest math. (3) Read Sec. 3 (Low–Lou, Titov–Démoulin) carefully — these are the benchmarks every code is tested on. (4) Sec. 4 is dense — focus on minimum-energy ambiguity removal and preprocessing rationale; treat the algorithm survey as a glossary. (5) Sec. 5 (helicity, energy, stability) — note the Aly–Sturrock argument. (6) Sec. 6 — pick the optimization method (Wiegelmann) for deep reading; understand the functional and how it is descended; skim other methods. (7) Sec. 6.6 NLFFF Consortium results — what determines code agreement?

**한국어.** (1) Sec. 1은 동기 부여와 $\beta$-vs-높이 그림 위주로 빠르게. (2) Sec. 2 (LFFF)는 Seehafer Fourier 방법과 Helmholtz 방정식을 제대로 익히기 — 가장 깔끔한 수학. (3) Sec. 3 (Low–Lou, Titov–Démoulin)은 모든 코드 테스트의 표준이므로 정독. (4) Sec. 4는 빽빽함 — minimum-energy ambiguity 제거와 preprocessing 동기를 핵심으로, 나머지는 용어집 수준으로. (5) Sec. 5 (helicity, energy, stability) — Aly–Sturrock 논증을 정리. (6) Sec. 6은 optimization 기법(Wiegelmann)을 정독하고 functional과 하강 절차를 이해; 다른 방법은 개관. (7) Sec. 6.6의 NLFFF Consortium 결과에서 코드 일치도를 결정하는 요인을 정리.

## References / 참고문헌
- Wiegelmann, T. & Sakurai, T., "Solar Force-free Magnetic Fields", Living Reviews in Solar Physics, 9, 5 (2012). DOI: 10.12942/lrsp-2012-5
- Low, B.C. & Lou, Y.Q., "Modeling solar force-free magnetic fields", ApJ, 352, 343 (1990).
- Wheatland, M.S., Sturrock, P.A., Roumeliotis, G., "An optimization approach to reconstructing force-free fields", ApJ, 540, 1150 (2000).
- Wiegelmann, T., "Optimization code with weighting function for the reconstruction of coronal magnetic fields", Solar Phys., 219, 87 (2004).
- Schrijver, C.J. et al., "Nonlinear force-free modeling of coronal magnetic fields", Solar Phys., 235, 161 (2006).
- DeRosa, M.L. et al., "A critical assessment of nonlinear force-free field modeling", ApJ, 696, 1780 (2009).
