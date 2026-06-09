---
title: "Pre-Reading Briefing: Thermal Diagnostics with AIA on SDO: DEM Inversions"
paper_id: "30"
topic: Solar_Observation
date: 2026-04-23
type: briefing
---

# Thermal Diagnostics with AIA on SDO: A Validated Method for DEM Inversions: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Cheung, M. C. M., Boerner, P., Schrijver, C. J., Testa, P., Chen, F., Peter, H., Malanushenko, A., "Thermal Diagnostics with the Atmospheric Imaging Assembly onboard the Solar Dynamics Observatory: A Validated Method for Differential Emission Measure Inversions", ApJ, 807, 143 (2015). DOI: 10.1088/0004-637X/807/2/143
**Author(s)**: Mark C. M. Cheung, Paul Boerner, Carolus J. Schrijver, Paola Testa, Feng Chen, Hardi Peter, Anna Malanushenko
**Year**: 2015

---

## 1. 핵심 기여 / Core Contribution

이 논문은 SDO/AIA의 6개 협대역 EUV 채널(94, 131, 171, 193, 211, 335 Å) 관측에서 미분 방출 측도(DEM, Differential Emission Measure) 분포를 역산하는 새로운 방법을 제안한다. 핵심 아이디어는 "희소성(sparsity)" 개념에 기반한 basis pursuit(기저 추구) 선형계획 문제로 DEM 역문제를 정식화하는 것이다. 이 방법은 양(positive-semidefinite) 해를 보장하며, 기존 χ² 최소화 방식과 달리 정규화 후 음수 제거 같은 후처리가 필요 없다. 또한 계산 속도가 매우 빨라(초당 10^4 해 이상) AIA 데이터의 높은 시간·공간 분해능에 걸맞은 DEM 맵 대량 생산이 가능하다. 저자들은 log-normal DEM, 3D NLFFF 활동영역 모델(AR 11158), 그리고 시간종속 MHD 시뮬레이션이라는 세 가지 검증용 모델로 역산 방법을 엄격히 검증한다.

This paper proposes a new DEM inversion method for SDO/AIA narrowband EUV observations (six channels: 94, 131, 171, 193, 211, 335 Å). The key idea is to formulate the DEM inverse problem as a basis pursuit linear program grounded in the concept of sparsity. The method guarantees positive-semidefinite solutions without the negativity fixes required by χ² minimization approaches, and is extremely fast (>10^4 solutions/second), enabling routine production of DEM maps at AIA's full cadence and spatial resolution. The authors rigorously validate the method against three classes of thermal models of increasing realism: log-normal Gaussian DEMs, quasi-steady loop atmospheres on an NLFFF model of NOAA AR 11158, and a fully-compressible 3D MHD simulation of AR corona formation. They also demonstrate how augmenting AIA with XRT's Be-thin channel further improves inversions, opening joint multi-instrument DEM analysis.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

DEM 역산은 태양 코로나 진단의 고전 문제이다. 1960년대 Phillips, Jefferies, Craig & Brown이 Fredholm 제1종 적분방정식으로서 수학적 난점(비유일성, 불안정성, 양성 조건)을 정리했고, 이후 40년간 스펙트럼선 비율, 정규화(smoothness/zeroth-order), MCMC, 파라메트릭 inversion 등 다양한 방법이 개발되었다. 2010년 SDO 발사로 AIA가 매 12초마다 7개 EUV 채널의 full-disk 이미지를 제공하게 되면서, "pixel-by-pixel DEM map을 고정밀 실시간으로 만들 수 있는가"가 핵심 과제로 부상했다. Hannah & Kontar (2012), Plowman et al. (2013)이 zeroth-order regularization 방식을 발전시켰고, Aschwanden & Boerner (2011)은 Gaussian 파라메트릭 inversion을 사용했다. Testa et al. (2012)과 Guennou et al. (2012a,b)은 AIA의 6-채널 한계를 체계적으로 평가했다. 본 논문은 이 흐름에서 "압축 센싱(compressed sensing, Candès & Tao 2006)"의 아이디어를 DEM 문제에 도입한 첫 주요 작업이다.

DEM inversion is a classic diagnostic problem in solar coronal physics. In the 1960s–70s, Phillips, Jefferies, and Craig & Brown formalized its mathematical pathologies (non-uniqueness, instability, positivity) as a Fredholm integral equation of the first kind. Over the following four decades, many approaches emerged: spectral line ratios, smoothness/zeroth-order regularization, MCMC, and parametric inversion. The launch of SDO in 2010, with AIA delivering seven-channel full-disk images every 12 seconds, made "can we produce high-fidelity DEM maps pixel-by-pixel at AIA cadence?" a central question. Hannah & Kontar (2012) and Plowman et al. (2013) advanced zeroth-order regularization; Aschwanden & Boerner (2011) used Gaussian parametric inversion; Testa et al. (2012) and Guennou et al. (2012a,b) systematically probed the six-channel limitation. The present paper is the first major application of compressed-sensing ideas (Candès & Tao 2006) to the DEM problem.

### 타임라인 / Timeline

```
1953 ── Courant & Hilbert: Fredholm equations (mathematical framework)
1962 ── Phillips: regularization of ill-posed inverse problems
1976 ── Craig & Brown: DEM inversion pathologies catalogued
1986 ── Craig & Brown book: "Inverse Problems in Astronomy"
1992 ── Monsignori Fossi & Landini: smoothness-regularized DEMs
1998 ── Chen, Donoho, Saunders: basis pursuit algorithm
2006 ── Candès & Tao: compressed sensing theory
2010 ── SDO/AIA launch: 7-channel 12s cadence EUV imaging
2011 ── Aschwanden & Boerner: Gaussian parametric AIA DEMs
2012 ── Hannah & Kontar: zeroth-order regularized inversion (AIA)
2012 ── Testa et al., Guennou et al.: AIA DEM limitations study
2013 ── Plowman, Kankelborg, Martens: fast AIA DEM (SVD + positivity)
2015 ── Cheung et al. (this paper): sparse/basis-pursuit DEM inversion
```

---

## 3. 필요한 배경 지식 / Prerequisites

- **Optically-thin plasma emission / 광학적으로 얇은 플라스마 방출**: EUV 코로나 복사는 $n_e n_H G(T)$에 비례하는 선 방출의 선적분으로 표현된다는 사실. EUV coronal radiation is a line-of-sight integral of $n_e n_H G(T)$.
- **Emission measure (EM) and DEM / 방출 측도**: $\mathrm{EM} = \int n_e^2 \, dz$, $\mathrm{DEM}(T) = n_e^2 \, dz/dT$. Classic definitions relating plasma density to observable emission.
- **Fredholm integral equations of the first kind / 제1종 프레드홀름 적분방정식**: ill-posed inverse problem; kernel $K_i(T)$ maps unknown function into observations.
- **Linear algebra for underdetermined systems / 부정 선형계**: when $m < n$, solution is not unique; use regularization or sparsity.
- **Linear programming & simplex method / 선형계획과 심플렉스 방법**: Dantzig (1955); used here to solve L1-norm minimization with inequality constraints.
- **Compressed sensing basics / 압축 센싱 기초**: Candès & Tao; recovery of sparse signals from underdetermined measurements via L1 minimization.
- **Tikhonov / zeroth-order regularization**: L2 penalty on solution norm or its derivatives to stabilize inversion.
- **AIA instrument / AIA 장비**: 7 EUV filters (94, 131, 171, 193, 211, 304, 335 Å); 1.5" resolution; 12s cadence; passband temperature response functions from CHIANTI.
- **Iron ionization stages & CHIANTI / 철 이온화 단계**: each AIA passband peaks at dominant Fe ion (Fe IX/X/XI/XII/XIV/XVI/XVIII) emission temperatures.
- **Python/IDL scientific computing / 수치계산 기초**: ability to read matrix-form equations and simplex solutions.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| DEM (Differential Emission Measure) | $\xi(T) = n_e^2 \, dz/d\log T$ (or $dz/dT$); plasma density-squared column distribution as a function of temperature. 플라스마 온도별 밀도 제곱 기둥 분포. |
| Temperature response function $R_i(T)$ | $i$-th channel signal per unit DEM at temperature $T$, units DN s$^{-1}$ cm$^5$ pixel$^{-1}$. 각 채널의 온도별 감응 함수. |
| Basis pursuit | L1-norm minimization $\min\|x\|_1$ s.t. $Dx=y$; convex surrogate for L0 sparsity. 희소해를 찾는 L1 최소화 기법. |
| Compressed sensing | Theory of recovering sparse signals from far fewer measurements than the signal dimension. 희소 신호를 소수 측정으로 복구하는 이론. |
| Tikhonov regularization | $\min \|Dx-y\|^2 + \lambda\|Lx\|^2$; L2 penalty stabilizing ill-posed inversions. L2 패널티 기반 정규화. |
| Zeroth-order regularization | Regularization term $\lambda\|x\|_2^2$ penalizing total solution norm (used by Hannah & Kontar 2012). 해 자체의 L2 크기를 페널티. |
| MCMC inversion (Kashyap & Drake) | Markov-chain Monte Carlo sampling of DEM posterior; slow but provides uncertainties. DEM 사후분포 샘플링 방법. |
| NLFFF (Nonlinear Force-Free Field) | Extrapolated coronal magnetic field satisfying $\mathbf{J}\times\mathbf{B}=0$; used to thread quasi-steady loop atmospheres. 비선형 무력장 외삽. |
| AR 11158 | Active region that produced the first X-class flare of SC24 (2011-02-15 X2.2); validation target in this paper. 본 논문에서 분석된 활동영역. |
| Be-thin XRT channel | Hinode/XRT broadband X-ray filter, peak response $\log T/K \sim 7.0$; complements AIA at hot temperatures. Hinode/XRT의 고온 감응 채널. |
| Basis functions (Dirac + Gaussians) | 84-column dictionary $\mathbf{B}$ combining 21 delta functions with truncated Gaussians of widths $a=0.1,0.2,0.6$. 델타+가우시안 기저 84개. |
| Simplex algorithm | Classical LP solver traversing polytope vertices; used here via IDL simplex routine. 선형계획의 고전 해법. |

(12 terms)

---

## 5. 수식 미리보기 / Equations Preview

**Eq. 1 — Forward problem / 정방향 문제**
$$
y_i = \int_0^\infty K_i(T)\, \mathrm{DEM}(T)\, dT
$$
$y_i$는 채널 $i$의 exposure-normalized pixel count (DN s$^{-1}$ pixel$^{-1}$), $K_i(T)$는 온도 응답 함수 (DN cm$^5$ s$^{-1}$ pixel$^{-1}$), $\mathrm{DEM}(T)$는 구하려는 미분 방출 측도. 이것이 본 논문의 중심 적분방정식이며 Fredholm 제1종에 해당한다.

$y_i$ is the exposure-normalized pixel value for channel $i$; $K_i(T)$ is the temperature response function; $\mathrm{DEM}(T)$ is the unknown. This Fredholm first-kind integral equation is the forward problem.

**Eq. 2 — Matrix form / 행렬 형식**
$$
\vec{y} = \mathbf{D}\vec{x}
$$
이산화 후 $m \times n$ (또는 $m\times l$) 행렬 방정식으로 환원. $m=6$ AIA 채널, $n=21$ 온도 bin 또는 $l=84$ 기저 함수.

After quadrature, inversion reduces to a linear system. For AIA: $m=6$ channels, $n=21$ temperature bins, $l=84$ basis functions.

**Eq. 7–12 — Basis pursuit LP1 / 희소 해 선형계획**
$$
\text{LP1: } \min_{\vec{x}} \sum_{j=1}^n x_j \ \text{ s.t. } \ \mathbf{D}\vec{x}\le \vec{y}+\vec{\eta},\ \mathbf{D}\vec{x}\ge\max(\vec{y}-\vec{\eta},0),\ \vec{x}\ge 0
$$
부정 시스템에서 L0 희소성 문제를 L1 근사로 바꾸고, $\vec x \ge 0$ 덕분에 $\|\vec x\|_1 = \sum x_j$로 단순화된다. $\vec\eta$는 채널별 AIA 불확도.

The L0 sparsity problem is relaxed to L1; because $\vec x\ge 0$, $\|\vec x\|_1$ reduces to the linear sum. $\vec\eta$ encodes per-channel AIA uncertainties.

**Eq. 13 — Log-normal validation DEM**
$$
\xi(T,T_c,\sigma) = \frac{\mathrm{EM}_0}{\sigma\sqrt{2\pi}}\exp\!\left[-\frac{(\log T - \log T_c)^2}{2\sigma^2}\right]
$$
검증에 사용된 Gaussian-in-log-T DEM. 피크 온도 $T_c$와 폭 $\sigma$를 파라미터로 스캔.

Validation DEMs: Gaussian in log-T with peak $T_c$ and width $\sigma$.

**Eq. 14–16 — Fidelity metrics / 충실도 지표**
$$
\mathrm{EM} = \sum_j \mathrm{EM}_j,\quad \log T_\mathrm{EM} = \mathrm{EM}^{-1}\sum_j \mathrm{EM}_j \log T_j,\quad W_\mathrm{EM}^2 = \mathrm{EM}^{-1}\sum_j \mathrm{EM}_j (\log T_j - \log T_\mathrm{EM})^2
$$
0차/1차/2차 모멘트로 total EM, EM-가중 온도, DEM의 열 폭을 정의하여 inversion과 truth 비교.

Zeroth/first/second moments of the EM distribution: total EM, EM-weighted log-temperature, and thermal width.

---

## 6. 읽기 가이드 / Reading Guide

- **섹션 2 (Statement of the Problem)**: Fredholm 적분방정식과 DEM inversion의 4가지 병적 성격(존재, 유일성, 안정성, 양성)을 파악하라. AIA는 $m=6$, $n\gg 6$이므로 근본적으로 underdetermined이다.
- **Section 2 covers the four pathologies of the Fredholm first-kind problem and the underdetermined nature of six-channel AIA DEM inversions.**

- **섹션 3 (A New Method Based on Sparsity)**: 저자들이 L0 → L1 → LP1으로 옮겨가는 논리를 따라가라. 양성 조건 $\vec x \ge 0$과 tolerance $\vec\eta$의 역할에 주목.
- **Section 3 traces the L0 → L1 → LP1 reformulation. Pay attention to positivity and tolerance bands.**

- **섹션 4 (Validation Tests)**: 3가지 검증 모델을 각 모델의 물리적 현실성 증가 순서로 이해. Fig. 2–9가 핵심 그림.
- **Section 4: validation against increasingly realistic models. Figs. 2–9 are the key.**

- **섹션 5 (Application to AIA Data)**: AR 11158의 두 번의 solar rotation에 대한 DEM 맵; 코어 loop vs fan loop의 열 구조 구별을 확인.
- **Section 5 applies the method to AR 11158; note how core vs. fan loops separate thermally.**

- **섹션 6 (Joint AIA+XRT)**: XRT Be-thin 채널 추가 시 고온 DEM 복원 정확도 향상을 보이는 핵심 확장.
- **Section 6 shows how adding XRT's Be-thin channel sharpens hot-plasma recovery.**

- **Appendix A (Quadrature Scheme)**: 84-basis dictionary matrix 구성을 반드시 확인. 실제 코드 재현에 필수.
- **Appendix A details the dictionary construction — essential for reproducing the code.**

---

## 7. 현대적 의의 / Modern Significance

이 논문의 희소 DEM inversion 코드는 현재 `aia_sparse_em_init` / `aia_sparse_em_solve`로 SolarSoft에 포함되어 일상적으로 사용된다. 또한 flare 연구(nanoflare heating 가설 검증, flare decay phase 해석), AR thermal evolution 연구, 그리고 AIA-XRT 또는 AIA-EIS 결합 DEM inversion의 표준 방법이 되었다. Su et al. (2018), Warren et al. (2020) 등 많은 후속 연구가 이 방법을 사용한다. 머신러닝 기반 DEM inversion(Wang et al. 2024 등)의 benchmark로도 기능한다. 미래 임무인 MUSE(Multi-slit Solar Explorer)와 EUV spectroscopy 데이터에 적용 가능하도록 확장되고 있다.

The sparse DEM code is now bundled in SolarSoft as `aia_sparse_em_init`/`aia_sparse_em_solve` and has become a standard tool. It underpins nanoflare-heating tests, AR thermal evolution studies, and serves as the backbone of joint AIA–XRT and AIA–EIS DEM analyses. It is a recurring benchmark for emerging ML-based DEM inversion, and the method is being extended for the upcoming MUSE multi-slit spectrograph and for EIS-style spectroscopy.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
