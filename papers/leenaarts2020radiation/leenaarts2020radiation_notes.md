---
title: "Radiation Hydrodynamics in Simulations of the Solar Atmosphere"
authors: Jorrit Leenaarts
year: 2020
journal: "Living Reviews in Solar Physics 17:3"
doi: "10.1007/s41116-020-0024-x"
topic: Living_Reviews_in_Solar_Physics
tags: [radiative-transfer, MHD, chromosphere, corona, photosphere, non-LTE, multi-group, BIFROST, MURaM]
status: completed
date_started: 2026-04-23
date_completed: 2026-04-23
---

# 70. Radiation Hydrodynamics in Simulations of the Solar Atmosphere / 태양 대기 시뮬레이션의 복사 유체역학

---

## 1. Core Contribution / 핵심 기여

**English.** Leenaarts' Living Review is a comprehensive survey of the approximations used to compute radiation-matter energy exchange in multi-dimensional radiation-MHD simulations of the solar atmosphere. The review is organized around the four atmospheric regimes (convection zone/photosphere, chromosphere, transition region, corona) because the physics of radiative energy exchange differs qualitatively between them. In the photosphere, LTE source functions and multi-group (opacity-binned) radiative transfer work well; in the optically-thin corona the coronal approximation (statistical equilibrium + no absorption) reduces radiation losses to a tabulated function $\Lambda(T)$; the chromosphere is the hardest because it is optically thin in the continuum but optically thick in a few strong resonance lines that require non-LTE with partial redistribution (PRD), while hydrogen and helium ionisation is out of equilibrium. Leenaarts catalogues the major production codes (BIFROST, MURaM, CO5BOLD, STAGGER, StellarBox, MANCHA3D, RAMENS, ANTARES, RadMHD), identifies where each falls on the approximation spectrum (Fig. 4's 8 options), and identifies five directions for improvement: testing Skartlien's scattering scheme, updating Carlsson & Leenaarts 2012 empirical recipes with PRD, critically assessing non-equilibrium ionisation methods, validating Judge's escape probability approach, and extending non-equilibrium ionisation to elements responsible for TR/coronal radiative losses.

**한국어.** Leenaarts의 Living Review는 다차원 복사-MHD 태양 대기 시뮬레이션에서 복사-물질 에너지 교환을 계산하는 데 사용되는 근사 방법들의 종합적 개관이다. 이 리뷰는 복사 에너지 교환의 물리가 영역마다 질적으로 다르기 때문에 네 가지 대기 영역(대류층/광구, 채층, 천이영역, 코로나)을 중심으로 구성된다. 광구에서는 LTE source function과 multi-group(opacity-binned) 복사 전달이 잘 작동하고; optically thin인 코로나에서는 coronal approximation(통계 평형 + 흡수 무시)이 복사 손실을 tabulated 함수 $\Lambda(T)$로 축약하며; 채층은 continuum에서는 optically thin이지만 몇 개의 강한 공명선에서 optically thick이어서 PRD를 포함하는 non-LTE 처리가 필요하고, 수소와 헬륨 이온화가 평형에서 벗어나기 때문에 가장 다루기 어렵다. Leenaarts는 주요 프로덕션 코드들(BIFROST, MURaM, CO5BOLD, STAGGER, StellarBox, MANCHA3D, RAMENS, ANTARES, RadMHD)을 정리하고, 각 코드가 근사 스펙트럼(Fig. 4의 8가지 옵션)의 어디에 위치하는지 식별하며, 다섯 가지 개선 방향 — Skartlien 산란 기법 검증, Carlsson & Leenaarts 2012 경험적 recipe를 PRD로 업데이트, 비평형 이온화 방법의 정확도 비판적 평가, Judge의 escape probability 방법 검증, 천이영역/코로나 복사 손실을 담당하는 원소에 대한 비평형 이온화 확장 — 을 제시한다.

---

## 2. Reading Notes / 읽기 노트

### Part I: Fundamentals (§2) / 기초

**English.** The review opens by writing down the MHD equations of continuity, momentum, and energy including radiation terms:

$$\frac{\partial \rho}{\partial t} = -\nabla\cdot(\rho\mathbf{v})$$
$$\frac{\partial \mathbf{p}}{\partial t} = -\nabla\cdot(\mathbf{v}\otimes\mathbf{p}-\boldsymbol{\tau}) - \nabla P + \mathbf{J}\times\mathbf{B} + \rho\mathbf{g} - \nabla\mathbf{P}_\mathrm{rad}$$
$$\frac{\partial e}{\partial t} = -\nabla\cdot(e\mathbf{v}) - P\nabla\cdot\mathbf{v} + Q + Q_\mathrm{rad}$$

The gas pressure is $P$, $\boldsymbol{\tau}$ the stress tensor, $\mathbf{J}$ the current density, $\mathbf{B}$ the magnetic field, $\mathbf{g}$ gravitational acceleration, $\mathbf{P}_\mathrm{rad}$ the radiation pressure tensor, and $Q_\mathrm{rad}$ the radiation heating/cooling term. Table 1 of the review compares the radiative energy density $E_\mathrm{rad}=(4\sigma/c)T^4$ to the gas energy density $e=nk_BT$:

| Region / 영역 | $E_\mathrm{rad}$ [J m$^{-3}$] | $e$ [J m$^{-3}$] | $E_\mathrm{rad}/e$ |
|---|---|---|---|
| Photosphere | 0.84 | $1.2\times10^4$ | $7.0\times10^{-5}$ |
| Chromosphere | 0.37 | 17 | $2.2\times10^{-2}$ |
| Corona | 0.22 | 0.21 | 1.1 |

Radiation pressure at $T=5700$ K is ~0.27 Pa vs gas pressure $\sim10^4$ Pa, confirming that radiation pressure is negligible in the solar atmosphere. However, the radiative flux $F_\mathrm{rad}=\sigma T^4 \approx 6.3\times 10^7$ W m$^{-2}$ represents an enormous energy throughput that must be correctly balanced by $Q_\mathrm{rad}$.

Because light-crossing times (seconds) are much shorter than hydrodynamic timescales (minutes in photosphere, ~1 min in chromosphere, few min in corona), the $\frac{1}{c}\partial_t I$ term in the transfer equation is dropped, leaving the static transfer equation $\mathbf{n}\cdot\nabla I_\nu = j_\nu - \alpha_\nu I_\nu$. The radiative flux divergence is then

$$\nabla\cdot\mathbf{F} = \int_0^\infty 4\pi\kappa_\nu\rho(S_\nu - J_\nu)\,d\nu \quad (\mathrm{Eq.\ 11})$$

with $S_\nu = j_\nu/(\kappa_\nu\rho)$ the source function and $J_\nu$ the angle-averaged intensity. In the diffusion approximation (optical depth $\gg 1$):

$$\mathbf{F}\approx -\frac{16\sigma T^3}{3\kappa_R}\nabla T$$
$$\frac{1}{\kappa_R} = \frac{\int_0^\infty (1/\kappa_\nu)(dB_\nu/dT)\,d\nu}{\int_0^\infty (dB_\nu/dT)\,d\nu}$$

Any numerical method that solves the transfer equation must converge to this limit at depth.

**한국어.** 리뷰는 복사 항을 포함한 연속, 운동량, 에너지의 MHD 방정식을 쓰는 것으로 시작한다 (위 영어 섹션의 식 참조). 여기서 $Q_\mathrm{rad}$은 복사 가열/냉각 항이며 이 리뷰의 핵심 계산 대상이다. Table 1은 복사 에너지 밀도 $E_\mathrm{rad}=(4\sigma/c)T^4$를 기체 에너지 밀도 $e=nk_BT$와 비교한다: 광구에서 $E_\mathrm{rad}/e \sim 7\times10^{-5}$, 채층에서 $\sim 2\times10^{-2}$, 코로나에서 $\sim 1$이다. $T=5700$ K에서 복사압은 $\sim 0.27$ Pa인 반면 기체압은 $\sim 10^4$ Pa로, 태양 대기에서 복사압은 무시할 수 있음을 확인한다. 그러나 복사 flux $F_\mathrm{rad}\approx 6.3\times10^7$ W m$^{-2}$은 엄청난 에너지 흐름이므로 $Q_\mathrm{rad}$으로 정확히 균형을 맞춰야 한다.

광속 통과 시간(초)이 수력학적 timescale(광구 수 분, 채층 ~1 분, 코로나 수 분)보다 훨씬 짧으므로 전달 방정식에서 $\frac{1}{c}\partial_t I$ 항을 떨어뜨려 정적 전달 방정식 $\mathbf{n}\cdot\nabla I_\nu = j_\nu - \alpha_\nu I_\nu$을 얻는다. 깊은 대기에서는 확산 근사가 적용되며 flux는 $T^3 \nabla T$에 비례하고, 올바른 평균 opacity는 Rosseland mean이다. 모든 수치 방법은 깊은 곳에서 이 극한에 수렴해야 한다.

### Part II: Photospheric Multi-group RT (§3) / 광구 Multi-group RT

**English.** The frequency integral in Eq. 11 cannot be evaluated pointwise in 3D; Nordlund (1982) replaced it by a sum over $N$ radiation groups:

$$\int_0^\infty 4\pi\kappa\rho(S_\nu-J_\nu)\,d\nu \approx \sum_{i=1}^N 4\pi\kappa_i\rho(S_i-J_i)\quad(\mathrm{Eq.\ 19})$$

The derivation rests on the observation that the $\Lambda$-operator depends only on opacity, so frequencies with identical opacities everywhere (same $\kappa_\nu$ at every spatial point) yield identical mean intensities. Two frequencies with identical opacities give $J_{\nu_1}+J_{\nu_2} = \Lambda_{\nu_1}[S_{\nu_1}] + \Lambda_{\nu_2}[S_{\nu_2}] = \Lambda_\kappa[S_{\nu_1}+S_{\nu_2}]$, so the bin-integrated source function $S_i = \sum_{j(i)} w_j S_j$ can be used once frequencies are sorted into bins by opacity.

**τ-sorting.** The de-facto method: at each frequency $\nu_j$, compute the vertical Rosseland optical depth $\tau_R(\tau_\nu=1)$ in a 1D reference atmosphere; define border values $\tau_R^k$ equidistantly in $\log\tau_R$; assign frequency to bin $i$ if $\tau_R^{k-1}<\tau_R(\tau_\nu=1)\le\tau_R^k$. Fig. 1 illustrates: the solid line is $\tau_R(\tau_\nu=1)$ vs frequency, horizontal lines are bin borders, and frequencies falling in each horizontal band are assigned to that bin.

**Bin mean opacity.** Two forms: (a) the bin Rosseland opacity ensures diffusion limit:
$$\frac{1}{\kappa_i^R} = \frac{\sum_{j(i)} w_j (1/\kappa_j)(dB_j/dT)}{\sum_{j(i)} w_j (dB_j/dT)}\quad(\mathrm{Eq.\ 28})$$
(b) the Planck mean is appropriate at low optical depth:
$$\kappa_i^B = \frac{\sum_{j(i)} w_j \kappa_j B_j}{\sum_{j(i)} w_j B_j}\quad(\mathrm{Eq.\ 29})$$
A smooth transition is made via $\kappa_i = W\kappa_i^R + (1-W)\kappa_i^B$ with $W = \exp(-\tau_i/\tau_0)$ or $W=\exp(-l_i/(\kappa_i\rho))$.

**LTE source function.** In photospheric deep layers (roughly $-100\,\mathrm{km}<z<200\,\mathrm{km}$, where $z=0$ is where $\tau_{500\,\mathrm{nm}}=1$) the source function is nearly Planckian. Then $S_i = \sum_{j(i)} w_j B_j$ depending only on temperature, pre-computed as a 1D lookup table.

**Non-LTE source function (Skartlien 2000).** In the upper photosphere and chromosphere this breaks down; resonance lines have photon destruction probability $\epsilon = C_{ul}/(A_{ul}+C_{ul}+B_\nu B_{ul}) < 10^{-4}$. With coherent two-level scattering $S_\nu = (1-\epsilon)J_\nu + \epsilon B_\nu$, so the flux divergence is

$$\frac{\nabla\cdot\mathbf{F}_\nu^\mathrm{NLTE}}{\rho} = 4\pi\epsilon\kappa_\nu(B_\nu - J_\nu)\quad(\mathrm{Eq.\ 36})$$

For given $B_\nu - J_\nu$ the non-LTE energy exchange can be orders of magnitude smaller than LTE. Skartlien's group form is $S_i = \epsilon_i J_i + t_i$, with group-averaged scattering probability $\epsilon_i$ and group thermal production $t_i$, requiring per-simulation 1D reference recalibration.

**Solving the transfer equation.** Two methods: short characteristics (SC) compute $I(\tau)=I(\tau=0)e^{-\tau}+\int_0^\tau S(t)e^{t-\tau}dt$ along segments between grid cells (orange arrows in Fig. 3), simple and parallelisable but diffusive; long characteristics (LC) trace rays from bottom to top of the domain (red lines), non-diffusive but require inter-subdomain communication. The combination options in Fig. 4 give 8 types: LTE vs non-LTE × SC vs LC × solve-for-K vs solve-for-I. Code assignments: Nordlund (1982) Type 5; MURaM, RAMENS, MANCHA3D, StellarBox Type 6; CO5BOLD Types 2 and 6; STAGGER Type 8; BIFROST Type 4.

**Heating rate numerical stability.** Deep in the atmosphere, computing $Q = 4\pi\kappa\rho(J-S)$ suffers from cancellation because $S\approx J$; instead use flux divergence $Q=-\nabla\cdot\mathbf{F}$. In upper layers the reverse: flux divergence is noisy, $S-J$ split is stable. Bruls et al. (1999) switching scheme: use flux divergence below $\tau\sim 0.1$, switch to $S-J$ above. Alternatively solve for $K^I = S - I$ directly via $dK^I/d\tau = dS/d\tau - K^I$, which avoids the cancellation.

**Example results.** Figs. 5–6 show a 4-group BIFROST computation: bins 1 and 2 produce images reminiscent of continuum granulation, bins 3 and 4 resemble Ca II H/K wings. Heating per mass unit in the chromosphere is independent of mass density and depends only on $\kappa_i(S_i-J_i)$. Fig. 7 shows individual group errors can reach 50% (group 3 at 0.3 Mm) but the total heating has only few-percent error. Fig. 8: LTE vastly overestimates chromospheric cooling (because $\epsilon = 1$ in LTE), while non-LTE yields much smaller, more realistic amplitudes.

**한국어.** 식 (11)의 주파수 적분은 3D에서 점별 계산이 불가능하므로 Nordlund (1982)는 $N$개의 복사 group에 대한 합으로 대체하였다 (식 19). 유도의 핵심은 $\Lambda$-연산자가 opacity에만 의존하므로 모든 공간점에서 동일한 opacity를 갖는 주파수들이 동일한 $J$를 갖는다는 관찰이다. 같은 opacity를 갖는 두 주파수에서 $J_{\nu_1}+J_{\nu_2} = \Lambda_\kappa[S_{\nu_1}+S_{\nu_2}]$이므로, 주파수를 opacity별로 bin에 분류한 뒤 bin-적분 source function $S_i = \sum_{j(i)} w_j S_j$을 사용할 수 있다.

**τ-sorting.** 사실상 표준 방법: 각 주파수 $\nu_j$에 대해 1D 기준 대기에서 $\tau_\nu=1$이 되는 높이의 Rosseland optical depth $\tau_R$를 계산하고, 경계값 $\tau_R^k$을 $\log\tau_R$에서 등간격으로 놓고, $\tau_R^{k-1} < \tau_R(\tau_\nu=1) \le \tau_R^k$이면 주파수를 bin $i$에 할당. Fig. 1은 이를 도식화한다.

**Bin 평균 opacity.** 깊은 곳에서 확산 극한을 보장하려면 bin Rosseland opacity (식 28), 바깥쪽에서는 Planck mean opacity (식 29)가 적절하며, $W=\exp(-\tau_i/\tau_0)$로 부드럽게 전환한다.

**LTE source function.** 깊은 광구($-100\,\mathrm{km}<z<200\,\mathrm{km}$)에서 source function은 거의 Planck 분포이므로 $S_i = \sum_{j(i)} w_j B_j$는 온도에만 의존하는 1D lookup table로 사전계산된다.

**Non-LTE source function (Skartlien 2000).** 상부 광구와 채층에서는 이 가정이 무너진다. 공명선의 광자 파괴 확률은 $\epsilon < 10^{-4}$로, coherent 2-레벨 산란에서 $S_\nu = (1-\epsilon)J_\nu + \epsilon B_\nu$, flux 발산은 식 36과 같다. 주어진 $B_\nu - J_\nu$에 대해 non-LTE 에너지 교환은 LTE보다 수 자릿수 작을 수 있다. Skartlien의 group 형태는 $S_i = \epsilon_i J_i + t_i$로, 시뮬레이션별 1D 기준 대기 재보정이 필요하다.

**전달 방정식 풀이.** Short characteristics (SC)는 격자 셀 사이 짧은 세그먼트를 따라 공식적 해를 계산 (식 38), 단순하고 병렬화 쉬우나 확산적; Long characteristics (LC)는 대기 전체를 관통하는 ray를 추적, 비확산적이지만 서브도메인 간 통신 필요. Fig. 4의 8가지 타입 조합: LTE/non-LTE × SC/LC × solve-for-K/solve-for-I. 코드 분류: Nordlund (1982) Type 5; MURaM, RAMENS, MANCHA3D, StellarBox Type 6; CO5BOLD Type 2·6; STAGGER Type 8; BIFROST Type 4.

**가열률 수치 안정성.** 깊은 대기에서 $Q=4\pi\kappa\rho(J-S)$은 $S\approx J$로 인한 cancellation이 문제; flux 발산 $Q=-\nabla\cdot\mathbf{F}$ 사용. 상부에서는 반대로 flux 발산이 노이지하고 $S-J$가 안정. Bruls et al. (1999) 스위칭 방식: $\tau\sim 0.1$ 아래에서는 flux 발산, 위에서는 $S-J$. 대안으로 $K^I=S-I$를 직접 풀이하여 cancellation 회피.

**예시 결과.** Fig. 5-6 4-group BIFROST 계산: bin 1-2는 continuum granulation, bin 3-4는 Ca II H/K 날개를 닮음. 채층 가열률은 질량 밀도와 무관. Fig. 7: 개별 group 오차는 50%까지 (group 3 at 0.3 Mm) 올라가나 총 가열의 오차는 수% 수준. Fig. 8: LTE는 채층 냉각을 크게 과대평가 ($\epsilon=1$), non-LTE는 훨씬 작고 현실적.

### Part III: Radiative Losses in TR and Corona (§4) / 천이영역 및 코로나 복사 손실

**English.** The corona is optically thin at all wavelengths except in the radio regime. In the transition region (TR, temperatures from $10^4$ to $10^5$ K where hydrogen is ionized), radiation losses are dominated by EUV lines of lower ionisation stages. For most lines and most regions the TR is optically thin; Kerr et al. (2019) showed Si IV 140 nm can be thick during flares.

**Coronal approximation.** Ignore non-local radiative transfer ($J_\nu = 0$ in rate equations), assume statistical equilibrium ($\partial n_i/\partial t = 0$). Then for a bound-bound transition in element $X$, ionisation stage $m$, upper level $j$, lower level $i$:

$$Q_{ij} = h\nu_{ij}\frac{A_{ji}}{n_e}\frac{n_{j,m}}{n_m}\frac{n_m}{n_X}\frac{n_X}{n_H}n_e n_H \equiv G(T,n_e)n_e n_H\quad(\mathrm{Eqs.\ 48-49})$$

Here $A_{ji}$ is the Einstein spontaneous emission coefficient, $n_{j,m}/n_m$ is the fraction of ions in stage $m$ in level $j$, $n_m/n_X$ is the fraction of atoms in stage $m$, $n_X/n_H$ is the abundance of element $X$. Because upper level populations are dominated by collisional excitation from the ground state, $n_e n_m \sim A_{ji} n_{j,m}$, giving $G \approx$ linear in $n_e$.

Summing $Q_{ij}$ over all levels, ionisation stages, and elements (plus continuum processes) gives the total loss function $\Lambda(T,n_e)$. Fig. 9 (Landi & Landini 1999) shows $\Lambda$ depends on $n_e$ by only ~20% over $10^{14}$–$10^{20}$ m$^{-3}$, so $\Lambda$ can be pre-computed at a fixed $n_e = 10^{16}$ m$^{-3}$. The radiative cooling is then:

$$Q = -\Lambda(T) n_e n_H\quad(\mathrm{Eq.\ 50})$$

**Abundance sensitivity.** The coronal loss function is dominated by C, Si, O, Fe lines. Fig. 10 shows that using coronal abundances (Feldman et al. 1992) versus photospheric abundances (Grevesse & Sauval 1998) changes $\Lambda$ by factors of 2–3 at coronal temperatures. This is likely the dominant source of uncertainty since coronal abundances have not been systematically reinvestigated since 1992.

**Implementation details.** CHIANTI atomic database (Dere et al. 1997, 2019) supplies the atomic data and ships with IDL/Python packages. To avoid spurious contributions from cooler layers, a cutoff is applied: BIFROST uses $\exp(-P/P_0)$ with $P_0$ at the top of the chromosphere; MURaM uses a hard cutoff at $T=20\,000$ K. Rempel (2017) pointed out that for large grid spacings the narrow TR is undersampled, causing inaccurate $Q$; he proposed a subgrid interpolation scheme.

**한국어.** 코로나는 radio 영역을 제외한 모든 파장에서 optically thin이다. 천이영역(TR, $10^4$–$10^5$ K, 수소 완전 이온화)에서는 저전리 단계의 EUV 선들이 복사 손실을 주도한다. 대부분의 선과 대부분의 영역에서 TR은 optically thin이며, Kerr et al. (2019)는 플레어 중 Si IV 140 nm가 optically thick이 될 수 있음을 보였다.

**Coronal approximation.** 비국소 복사 전달 무시($J_\nu=0$), 통계 평형 가정($\partial n_i/\partial t=0$). 그러면 bound-bound 전이에 대해 식 48, 49처럼 $Q_{ij}=G(T,n_e)n_e n_H$로 쓸 수 있다. 상위 level이 주로 ground state로부터의 충돌 excitation으로 채워지므로 $G$는 $n_e$에 거의 선형.

모든 level, 이온화 단계, 원소(continuum 과정 포함)에 대한 합이 총 loss function $\Lambda(T,n_e)$. Fig. 9: $\Lambda$의 $n_e$ 의존성은 $10^{14}$–$10^{20}$ m$^{-3}$에서 ~20%에 불과하므로 고정된 $n_e=10^{16}$ m$^{-3}$에서 사전계산 가능. 복사 냉각은 $Q=-\Lambda(T)n_e n_H$ (식 50).

**Abundance 민감도.** 코로나 loss function은 C, Si, O, Fe 선이 지배. Fig. 10: Feldman et al. (1992)의 코로나 abundance vs Grevesse & Sauval (1998)의 광구 abundance 사용 시 코로나 온도에서 $\Lambda$가 2-3배 차이. 1992년 이후 코로나 abundance가 체계적으로 재조사되지 않아, 이것이 주요 불확실성 원천일 가능성.

**구현 세부.** CHIANTI 원자 데이터베이스 (Dere et al. 1997, 2019) 사용. 저온층 오염 방지 cutoff: BIFROST는 $\exp(-P/P_0)$ (연속), MURaM은 $T=20\,000$ K에서 hard cutoff. Rempel (2017)는 격자 간격이 클 때 좁은 TR의 undersampling으로 $Q$ 부정확, subgrid interpolation 제안.

### Part IV: Chromospheric Radiative Transfer (§5) / 채층 복사 전달

**English.** The photospheric approximations break down in the chromosphere: it has low opacity except in strong spectral lines (H I Ly-α, Ca II H&K and infrared triplet, Mg II h/k lines, UV continua below 160 nm, sub-mm continua above 160 μm). See Fig. 12 showing net cooling is dominated by five Ca II lines and two Mg II lines between $z=700$–2120 km; at higher altitudes Ly-α dominates. These lines are severely underestimated by bin-averaged opacity; the source function is not Planckian; CRD is inaccurate — PRD must be used.

A full 3D non-LTE statistical equilibrium with PRD is in principle possible but computationally crushing: ~10 s of CPU time per grid cell for one atom (Sukhorukov & Leenaarts 2017), compared to ~5 μs per grid cell per timestep for the radiation-MHD simulation itself. That's a factor of $2\times10^6$ too slow.

**Carlsson & Leenaarts 2012 empirical recipes.** They describe the net radiative cooling as the product of three empirically calibrated quantities:

$$Q_{X_m} = -L_{X_m}E_{X_m}(\tau)\frac{n_{X_m}}{n_X} A_X \frac{n_H}{\rho}n_e\rho\quad(\mathrm{Eq.\ 51})$$

- $L_{X_m}$: optically thin loss function per electron per particle (fitted as function of $T$, computed from net downward rates);
- $E_{X_m}(\tau)$: photon escape probability (function of vertical column mass);
- $n_{X_m}/n_X$: ionisation fraction (function of $T$);
- $A_X$: elemental abundance.

These are tabulated from a 2D BIFROST radiation-MHD simulation with MULTI3D including PRD. For H I the tables come from a 1D RADYN simulation including non-equilibrium ionisation. Fig. 13 shows the JPDFs of $L_{\mathrm{Ca\,II}}$, $E_{\mathrm{Ca\,II}}$, and Ca II fraction vs temperature/column mass, together with the adopted red fit curves. The spread in these quantities means the recipe will have large errors in individual grid cells, though on average it reproduces the detailed calculation well (Fig. 14).

**Coronal radiation absorption.** Half of coronal UV radiation is emitted downward and absorbed in the chromosphere. Using He I continuum representative opacity $\kappa_\mathrm{HeI}$ and $\eta = -Q_\mathrm{cor}/4\pi$, solve a simplified transfer equation $dI/ds = \eta - \kappa_\mathrm{HeI}\rho I$, giving $Q_\mathrm{abs} = 4\pi\kappa_\mathrm{HeI}\rho J_\mathrm{cor}$.

Fig. 15 combines photospheric (bin 1), chromospheric (Carlsson-Leenaarts), coronal (CHIANTI), and total losses in a 3D BIFROST simulation. The photospheric cooling is largest in the photosphere; coronal losses are relevant throughout the corona and largest in the TR; chromospheric losses peak just below the TR due to Ly-α. The combined picture shows the largest radiative losses per unit mass occur in the TR.

**한국어.** 광구의 근사는 채층에서 무너진다: 채층은 강한 공명선(H I Ly-α, Ca II H&K 및 IR triplet, Mg II h/k, 160 nm 이하 UV continuum, 160 μm 이상 sub-mm continuum) 외에서는 opacity가 낮다. Fig. 12: $z=700$-2120 km에서 순냉각은 Ca II 5개 선 + Mg II 2개 선이 지배; 더 높이 올라가면 Ly-α가 지배. 이 선들은 bin 평균 opacity로 크게 과소평가되고, source function은 Planck 분포가 아니며, CRD는 부정확 — PRD가 필요.

완전한 3D non-LTE 통계 평형 + PRD는 원리적으로 가능하지만 계산적으로 극도로 비싸다: 단일 원자 단일 해에 격자 셀당 ~10 초 CPU (Sukhorukov & Leenaarts 2017), 복사-MHD 시뮬레이션 자체는 셀당 timestep당 ~5 μs이므로 $2\times10^6$배 차이.

**Carlsson & Leenaarts 2012 경험적 recipe.** 순복사 냉각을 세 경험적 계량의 곱으로 기술 (식 51): optically thin loss function $L_{X_m}$, photon escape probability $E_{X_m}(\tau)$, ionisation fraction $n_{X_m}/n_X$, abundance $A_X$. PRD 포함 2D BIFROST + MULTI3D 시뮬레이션에서 표화; H I의 경우 RADYN 1D 비평형 이온화 계산에서. Fig. 13: JPDF와 fit curve. 분산 탓에 개별 격자 셀에서 큰 오차 가능하지만 평균적으로 상세 계산을 잘 재현 (Fig. 14).

**코로나 복사 흡수.** 코로나 UV의 절반은 아래로 방출되어 채층에서 흡수. He I continuum 대표 opacity $\kappa_\mathrm{HeI}$, $\eta = -Q_\mathrm{cor}/4\pi$을 사용해 간단한 전달 방정식 $dI/ds=\eta - \kappa_\mathrm{HeI}\rho I$을 풀어 $Q_\mathrm{abs}=4\pi\kappa_\mathrm{HeI}\rho J_\mathrm{cor}$을 얻는다.

Fig. 15: 3D BIFROST 시뮬레이션에서 광구(bin 1), 채층(Carlsson-Leenaarts), 코로나(CHIANTI), 총 손실. 광구 냉각은 광구에서 최대, 코로나 손실은 전체 코로나에 걸쳐 있으며 TR에서 최대, 채층 손실은 Ly-α로 TR 바로 아래에서 피크. 합성하면 가장 큰 질량당 복사 손실은 TR에서 발생.

### Part V: Non-equilibrium Ionisation and Equation of State (§6) / 비평형 이온화와 상태방정식

**English.** Non-equilibrium ionisation matters when MHD timescales are short compared to atomic relaxation. The combined continuity + rate equation is

$$\frac{\partial n_i}{\partial t} + \nabla\cdot(n_i\mathbf{v}) = \sum_{j=1,j\neq i}^N n_j P_{ji} - n_i\sum_{j=1,j\neq i}^N P_{ij}\quad(\mathrm{Eq.\ 54})$$

Ignoring advection and assuming constant $P_{ij}$ gives $\partial\mathbf{n}/\partial t = P\mathbf{n}$, solved as $\mathbf{n}(t) = \sum_i c_i\mathbf{a}_i e^{\lambda_i t}$ with eigenvalues $\lambda_i$ of $P$. One eigenvalue is zero (equilibrium); others are negative. The relaxation timescale is $\tau = 1/\min(|\lambda_i|)$ for $\lambda_i\neq 0$. For a two-level atom Carlsson & Stein (2002) derived

$$\tau = \frac{1}{C_{21}+C_{12}+R_{21}\left[1 - (n_1 R_{12})/(n_2 R_{21})\right]}\quad(\mathrm{Eq.\ 58})$$

With $C$ collisional and $R$ radiative rates. Fig. 16: hydrogen ionisation/recombination timescale in a RADYN simulation reaches ~$10^5$ s in the chromosphere, where $T$ and $n_e$ are low and strong transitions are close to detailed balance. For helium, Golding et al. (2014) found $10^2$–$10^3$ s — also out of equilibrium.

These timescales far exceed the chromospheric/TR hydrodynamic timescales (~1 min). Consequences:
- The EOS cannot be computed from Saha–Boltzmann; instead Eq. 54 must be solved with energy conservation $e = \frac{3}{2}k_BT(\sum_{ijk}n_{ijk}+n_e) + \sum n_{ijk}E_{ijk}$ (Eq. 60) and charge conservation $n_e = \sum(j-1)n_{ijk}$.
- Ionisation no longer acts as an energy buffer. In LTE, internal energy increases go into ionising H and He before temperature can rise; in non-equilibrium, $T$ rises directly, and $T$ decreases are stronger because ionisation energy cannot be quickly released.
- This increases the amplitude of temperature jumps in acoustic shocks (Carlsson & Stein 2002; Leenaarts et al. 2007).
- "Preferred temperature" bands at $\sim$6, 10, 22 kK in JPDFs of LTE simulations (Fig. 18) — associated with Saha transitions of H I, He I, He II — vanish when non-equilibrium is turned on.

**Sollum (1999) approximations.** The chromospheric radiation field in H transitions can be approximated by a constant value above some height and by the local Planck function at larger depth, with a smooth transition. This allows Eq. 54 to be solved without solving the transfer equation, at 3–5× the cost of LTE. BIFROST uses this method (Leenaarts et al. 2007). Golding et al. (2016) extended to helium, needing 7 radiation bins to approximate the He continua and He II 30.4 nm line. Non-equilibrium also affects ambipolar diffusion efficiency (Martinez-Sykora et al. 2017; Khomenko et al. 2018), because ambipolar diffusivity depends explicitly on neutral/ion densities.

**한국어.** 비평형 이온화는 MHD timescale이 원자 이완 timescale보다 짧을 때 중요. 연속 + 비율 방정식 (식 54)에서 advection 무시, $P_{ij}$ 일정 가정하면 $\partial\mathbf{n}/\partial t = P\mathbf{n}$이 되고, 해는 $P$의 고유값 $\lambda_i$ 합. 하나의 고유값은 0(평형), 나머지는 음. 이완 timescale은 $\tau = 1/\min(|\lambda_i|)$. Two-level atom의 경우 Carlsson & Stein (2002)이 식 58 유도. Fig. 16: RADYN 시뮬레이션에서 수소 이온화/재결합 timescale이 채층에서 ~$10^5$ s에 달함 ($T$, $n_e$ 낮고 강한 전이가 detailed balance에 가까움). 헬륨의 경우 Golding et al. (2014)에서 $10^2$–$10^3$ s.

이 timescale은 채층/TR 수력학적 timescale(~1 min)을 크게 초과. 결과:
- EOS는 Saha-Boltzmann으로 계산 불가; 식 54 + 에너지 보존 (식 60) + 전하 보존 (식 59) 연립풀이 필요.
- 이온화가 더 이상 에너지 buffer 역할을 못함. LTE에서는 내부 에너지 증가가 먼저 H, He 이온화에 들어가 $T$ 상승을 억제; 비평형에서는 $T$가 직접 상승하고, $T$ 하강 시 이온화 에너지를 빨리 방출하지 못해 하강폭도 커짐.
- 음향 shock에서 $T$ jump 진폭 증가 (Carlsson & Stein 2002; Leenaarts et al. 2007).
- LTE 시뮬레이션의 JPDF에서 ~6, 10, 22 kK의 "선호 온도" 띠 (Fig. 18, H I, He I, He II Saha 전이) — 비평형 켜면 사라짐.

**Sollum (1999) 근사.** H 전이에서 채층 복사장을 채층 내 일정값 + 깊은 곳에서 국소 Planck 함수로 근사 (부드러운 전환). 전달 방정식 해결 없이 식 54 풀이 가능, LTE 대비 3-5배 비용. BIFROST에서 사용 (Leenaarts et al. 2007). Golding et al. (2016)이 헬륨으로 확장, He continuum 및 He II 30.4 nm 근사에 7개 radiation bin 필요. 비평형은 ambipolar diffusion 효율에도 영향 (Martinez-Sykora et al. 2017; Khomenko et al. 2018) — ambipolar diffusivity는 중성/이온 밀도에 직접 의존.

### Part VI: Other Developments (§7) / 기타 발전

**English.** **Abbett & Fisher (2012) fast photospheric RT.** Treat each column as independent plane-parallel atmosphere. The angle-averaged mean intensity becomes $J_\nu(\tau_\nu) = \frac{1}{2}\int_0^\infty S_\nu(\tau'_\nu)E_1(|\tau_\nu-\tau'_\nu|)d\tau'_\nu$. The exponential integral $E_1(x)$ peaks at $x=0$, so $J_\nu \approx \frac{1}{2}S_\nu(\tau_\nu)(1-E_2(\tau_\nu)/2)$. Assuming LTE, the frequency integral yields $Q \approx -2\kappa^B\rho\sigma T^4 E_2(\tau^B)$ with $\kappa^B$ Planck-averaged opacity and $\tau^B$ its optical depth. Extremely fast (single 2D lookup + column integration) with reasonable granulation morphology (Fig. 19 from RadMHD).

**Judge (2017) escape probability method.** Replace the repeated transfer equation solutions by a single vertical integral for $\bar{J}$:

$$\frac{d}{d\tau}(S-\bar{J}) = q^{1/2}\frac{d}{d\tau}(q^{1/2}S)\quad(\mathrm{Eq.\ 67})$$

where $q$ is an escape probability function depending only on vertical optical depth. Approximations: source function varies slowly along optical path, horizontal structure is ignored. Solving the statistical equilibrium non-LTE RT in a MURaM test atmosphere is ~100× faster than full method. Can also solve non-equilibrium problems (Eqs. 54, 59, 60 simultaneously). Not yet tested in radiation-MHD simulations — one of Leenaarts' five suggested improvements.

**한국어.** **Abbett & Fisher (2012) 빠른 광구 RT.** 각 열을 독립 plane-parallel 대기로 취급. 각도 평균 평균 강도 $J_\nu \approx \frac{1}{2}S_\nu(\tau_\nu)(1-E_2(\tau_\nu)/2)$. LTE 가정 + 주파수 적분 후 $Q\approx -2\kappa^B\rho\sigma T^4 E_2(\tau^B)$. 극도로 빠름(2D lookup + 열별 적분), 합리적 granulation 형태 재현 (Fig. 19 RadMHD).

**Judge (2017) escape probability.** 반복적 전달 방정식 해를 $\bar{J}$에 대한 단일 수직 적분으로 대체 (식 67). 근사: source function이 광경로 따라 천천히 변함, 수평 구조 무시. MURaM 테스트 대기에서 통계 평형 non-LTE RT가 완전 방법 대비 ~100배 빠름. 비평형 문제(식 54, 59, 60 동시)에도 적용 가능. 복사-MHD 시뮬레이션에서 아직 미검증 — Leenaarts의 5개 개선 과제 중 하나.

### Part VII: Conclusions and Outlook (§8) / 결론과 전망

**English.** Photospheric radiative energy exchange is the best developed. Pereira et al. (2013) showed that 11-group LTE multi-group RT with accurate abundances and opacities reproduces center-to-limb variation, absolute flux spectrum, hydrogen line wings, and continuum intensity distribution. Safe for virtually all purposes.

TR/coronal losses are less accurate: statistical equilibrium + no absorption + fixed abundances give factor-of-two errors from non-equilibrium effects (especially in flares, factor of 2 in $\Lambda$), and abundance uncertainty gives factor-of-3 errors at high temperatures.

The chromosphere is the least studied and the methods most inaccurate. A long sequence of approximations is needed for computational feasibility. Leenaarts argues that the chromosphere adapts its thermodynamic state so that heating and cooling balance — radiative losses scale as $T^4$ (LTE) or exponentially (coronal approximation), but heating mechanisms (viscosity $T^{1/2}$, resistivity $\ln T/T^{3/2}$) do not. So the actual temperature balance depends sensitively on the radiative loss model. A simplified MURaM simulation with single-bin gray LTE + coronal loss function produces chromospheric structure resembling reality (Fig. 20), though with wrong temperatures.

Leenaarts' five improvements:
1. Test Skartlien's multi-bin scattering in mid/upper chromosphere.
2. Update Carlsson & Leenaarts (2012) Ca II, Mg II tables using 3D non-LTE PRD (Sukhorukov & Leenaarts 2017).
3. Critically assess non-equilibrium ionisation methods, especially the Ly-α treatment.
4. Test and develop the escape probability method further (Sect. 7.2).
5. Implement non-equilibrium ionisation for TR/coronal elements in a computationally efficient fashion.

**한국어.** 광구 복사 에너지 교환이 가장 잘 정립됨. Pereira et al. (2013): 정확한 abundance와 opacity를 사용한 11-group LTE multi-group RT가 중심-limb 변화, 절대 flux 스펙트럼, 수소 line 날개, continuum 강도 분포를 재현. 거의 모든 용도에 충분.

TR/코로나 손실은 덜 정확: 통계 평형 + 흡수 무시 + 고정 abundance로 비평형 효과에서 2배 오차 (특히 플레어에서 $\Lambda$가 2배), abundance 불확실성은 고온에서 3배 오차.

채층은 가장 덜 연구되었고 방법이 가장 부정확. 계산 가능성을 위해 긴 근사 시퀀스 필요. Leenaarts의 논지: 채층은 가열과 냉각이 균형 이루도록 열역학 상태를 조정 — 복사 손실은 $T^4$(LTE) 또는 지수적(coronal approximation)으로 스케일, 그러나 가열 메커니즘(viscosity $T^{1/2}$, resistivity $\ln T/T^{3/2}$)은 그렇지 않음. 따라서 실제 온도 균형은 복사 손실 모델에 민감하게 의존. 단일 bin gray LTE + coronal loss function의 단순 MURaM 시뮬레이션도 현실과 유사한 채층 구조(Fig. 20), 단 온도는 틀림.

Leenaarts의 5가지 개선 과제:
1. Skartlien multi-bin 산란을 채층 중·상부에서 검증.
2. Carlsson & Leenaarts (2012) Ca II, Mg II 표를 3D non-LTE PRD (Sukhorukov & Leenaarts 2017)로 업데이트.
3. 비평형 이온화 방법, 특히 Ly-α 처리를 비판적으로 평가.
4. Escape probability 방법 검증·개발 (§7.2).
5. TR/코로나 원소에 대해 계산적으로 효율적인 비평형 이온화 구현.

---

## 3. Key Takeaways / 핵심 시사점

1. **Radiation is essential but computationally prohibitive / 복사는 필수이나 계산적으로 감당 불가** — The specific intensity $I_\nu$ depends on 7 parameters (3 space, 2 angle, 1 frequency, 1 time), making full RT in 3D MHD impractical. Approximations are mandatory, and the quality of a simulation is largely set by the quality of its radiation treatment. / 비휘도는 7개 매개변수(공간 3, 각도 2, 주파수 1, 시간 1)에 의존하므로 3D MHD에서 완전한 RT는 불가능. 근사는 필수이며, 시뮬레이션의 품질은 대체로 복사 처리의 품질에 의해 결정됨.

2. **Multi-group RT works in the photosphere / Multi-group RT는 광구에서 잘 작동** — Nordlund's (1982) binning of frequencies by opacity, together with LTE source functions, accurately reproduces granulation and photospheric line wings. Four groups suffice for radiative losses; 11 groups match observations in detail (Pereira et al. 2013). The total flux divergence has only few-percent error even when individual bins err by 50%. / Nordlund (1982)의 opacity 기반 주파수 분류와 LTE source function은 granulation 및 광구 line 날개를 정확히 재현. 복사 손실에 4개 group으로 충분; 11개 group으로 관측과 세밀히 일치 (Pereira et al. 2013). 개별 bin이 50% 오차여도 총 flux 발산은 수% 오차.

3. **Non-LTE source function changes chromospheric cooling by factors of $10^4$ / Non-LTE source function은 채층 냉각을 $10^4$배 바꿈** — Resonance lines have $\epsilon < 10^{-4}$, so LTE overestimates the cooling rate enormously. Skartlien's (2000) multi-group scattering scheme remedies this but requires per-simulation 1D calibration. / 공명선은 $\epsilon < 10^{-4}$이므로 LTE는 냉각률을 극도로 과대평가. Skartlien (2000) multi-group 산란 기법이 이를 해결하지만 시뮬레이션별 1D 보정 필요.

4. **Coronal approximation gives an easily pre-computed loss function / Coronal approximation은 쉽게 사전계산 가능한 loss function 제공** — Because upper levels are collisionally populated from the ground state and the $n_e$ dependence is weak (~20% over 6 orders of magnitude), the coronal loss function reduces to $Q = -\Lambda(T)n_e n_H$, a 1D lookup. But abundance uncertainty gives factors of 2–3 variations in $\Lambda$. / 상위 level이 ground state로부터 충돌 excitation되고 $n_e$ 의존성이 약하므로 ($10^{14}$–$10^{20}$ m$^{-3}$에서 ~20%), coronal loss function은 $Q=-\Lambda(T)n_e n_H$으로 축약되어 1D lookup. 그러나 abundance 불확실성으로 $\Lambda$에 2-3배 변동.

5. **Hydrogen ionisation timescale is $10^5$ s in the chromosphere / 채층 수소 이온화 timescale은 $10^5$ s** — Much longer than hydrodynamic timescales (~1 min), so ionisation is severely out of equilibrium. This means LTE equation of state is inadequate; ionisation can no longer buffer energy changes; temperature jumps in acoustic shocks are amplified; JPDFs of $T$ lose the LTE "preferred temperature" bands. / 수력학적 timescale(~1 분)보다 훨씬 길어 이온화는 심각한 비평형 상태. LTE 상태방정식은 부적절; 이온화가 더 이상 에너지 buffer 역할 못함; 음향 shock에서 온도 jump 증폭; $T$의 JPDF에서 LTE "선호 온도" 띠 사라짐.

6. **Chromospheric radiative transfer is the weakest link / 채층 복사 전달이 가장 약한 고리** — Strong resonance lines (H I Ly-α, Ca II H/K, Mg II h/k) dominate cooling but require non-LTE with PRD. Full 3D calculation is $2\times 10^6$× too slow. Carlsson & Leenaarts (2012) empirical recipes (loss × escape probability × ionisation × abundance) are fast but were calibrated on a single 2D simulation and need updating with modern PRD. / 강한 공명선 (H I Ly-α, Ca II H/K, Mg II h/k)이 냉각을 지배하지만 PRD 포함 non-LTE 필요. 완전한 3D 계산은 $2\times10^6$배 너무 느림. Carlsson & Leenaarts (2012) 경험적 recipe (loss × escape probability × ionisation × abundance)는 빠르지만 단일 2D 시뮬레이션에서 보정된 것이라 현대 PRD로 업데이트 필요.

7. **Eight combinatorial options structure all photospheric RT codes / 8가지 조합 옵션이 모든 광구 RT 코드를 구조화** — LTE vs non-LTE source function × SC vs LC × solve-for-K vs solve-for-I (Fig. 4). Leenaarts maps each production code to one option: BIFROST uses non-LTE + SC + solve-for-I (Type 4), MURaM uses LTE + LC + solve-for-I (Type 6), STAGGER uses non-LTE + LC + solve-for-I (Type 8). This taxonomy enables informed code selection and comparison. / LTE vs non-LTE source function × SC vs LC × solve-for-K vs solve-for-I (Fig. 4). Leenaarts는 각 프로덕션 코드를 하나의 옵션에 매핑: BIFROST는 non-LTE + SC + solve-for-I (Type 4), MURaM은 LTE + LC + solve-for-I (Type 6), STAGGER는 non-LTE + LC + solve-for-I (Type 8). 이 분류가 정보 있는 코드 선택과 비교를 가능케 함.

8. **Chromosphere "self-adjusts" to balance heating and cooling / 채층은 가열-냉각 균형으로 "자기조정"** — In MHD the only heating mechanisms are viscosity and resistivity (temperature-independent in practice), while radiative losses scale strongly with temperature. So the chromosphere finds a temperature where losses match heating. An imperfect radiation treatment gives wrong temperatures (~1000-2000 K off) but nearly correct density/velocity structure and dissipation rate. This explains why simpler codes still produce chromospheres "looking right" (Fig. 20). / MHD에서 가열 메커니즘은 viscosity와 resistivity만 있으며 실무상 온도 무관, 반면 복사 손실은 온도에 강하게 의존. 따라서 채층은 손실이 가열과 같은 온도를 찾음. 불완전한 복사 처리는 틀린 온도(~1000-2000 K 오차)를 주지만 거의 올바른 밀도/속도 구조와 dissipation rate를 준다. 단순한 코드도 "그럴듯한" 채층을 만드는 이유 (Fig. 20).

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Radiative transfer fundamentals / 복사 전달 기초

Time-independent transfer equation (Eq. 14):
$$\mathbf{n}\cdot\nabla I_\nu = j_\nu - \alpha_\nu I_\nu$$
where $\alpha_\nu = \kappa_\nu\rho$ is extinction per volume, $\kappa_\nu$ is opacity per mass, $j_\nu$ is emissivity. Source function $S_\nu = j_\nu/\alpha_\nu$. Formal solution along a ray (Eq. 38):
$$I(\tau) = I(\tau=0)e^{-\tau} + \int_0^\tau S(t)e^{t-\tau}dt$$

Radiative flux divergence (Eq. 11):
$$\nabla\cdot\mathbf{F} = \int_0^\infty 4\pi\kappa_\nu\rho(S_\nu - J_\nu)\,d\nu$$

### 4.2 Diffusion and Rosseland mean / 확산과 Rosseland 평균

Deep-atmosphere approximation (Eq. 15):
$$J_\nu \approx S_\nu + \frac{1}{3}\frac{d^2 S_\nu}{d\tau_\nu^2}$$

Rosseland mean opacity (Eq. 18):
$$\frac{1}{\kappa_R} = \frac{\int_0^\infty (1/\kappa_\nu)(dB_\nu/dT)\,d\nu}{\int_0^\infty (dB_\nu/dT)\,d\nu}$$

Total flux in diffusion limit (Eq. 17):
$$\mathbf{F} \approx -\frac{16\sigma T^3}{3\kappa_R}\nabla T$$

### 4.3 Multi-group formulation / Multi-group 공식

Replace frequency integral by N groups (Eq. 19):
$$\int_0^\infty 4\pi\kappa\rho(S_\nu - J_\nu)\,d\nu \approx \sum_{i=1}^N 4\pi\kappa_i\rho(S_i - J_i)$$

Bin Rosseland opacity (Eq. 28):
$$\frac{1}{\kappa_i^R} = \frac{\sum_{j(i)} w_j (1/\kappa_j)(dB_j/dT)}{\sum_{j(i)} w_j (dB_j/dT)}$$

Bin Planck opacity (Eq. 29):
$$\kappa_i^B = \frac{\sum_{j(i)} w_j \kappa_j B_j}{\sum_{j(i)} w_j B_j}$$

Smooth transition (Eqs. 30, 31):
$$\kappa_i = W\kappa_i^R + (1-W)\kappa_i^B,\quad W = e^{-\tau_i/\tau_0}$$

### 4.4 Non-LTE source function / Non-LTE source function

Photon destruction probability (Eq. 33):
$$\epsilon = \frac{C_{ul}}{A_{ul}+C_{ul}+B_\nu B_{ul}}$$

Two-level atom source function (Eq. 35):
$$S_\nu = (1-\epsilon)J_\nu + \epsilon B_\nu$$

Non-LTE flux divergence (Eq. 36):
$$\frac{\nabla\cdot\mathbf{F}_\nu^\mathrm{NLTE}}{\rho} = 4\pi\epsilon\kappa_\nu(B_\nu - J_\nu)$$

Skartlien's group source function (Eq. 37):
$$S_i = \epsilon_i J_i + t_i$$

### 4.5 Opacity sources / 흡수 원천

Principal opacity contributions in the photosphere and low chromosphere (not explicit in the review but implied):

- **H^- bound-free**: peaks near 850 nm, dominates continuum opacity from 400 nm to 1.6 μm in the photosphere. $\kappa_{\mathrm{H}^-}^\mathrm{bf} \propto n_e P_g$ via detailed photoionisation cross-section.
- **H^- free-free**: dominates infrared continuum beyond 1.6 μm. $\kappa_{\mathrm{H}^-}^\mathrm{ff} \propto n_e P_g T^{-3/2}$.
- **Thomson scattering**: frequency-independent $\kappa_T = \sigma_T n_e / \rho$ with $\sigma_T = 6.65\times 10^{-29}$ m$^2$; relevant at low densities (chromosphere and above).
- **Line opacity**: resonance lines (H I, Ca II H&K, Mg II h&k) contribute enormously to chromospheric opacity with $\kappa_\mathrm{line}\propto n_\ell f_{\ell u}\phi_\nu$.

### 4.6 Coronal approximation / Coronal approximation

Rate equations with $\partial_t = 0$, $J_\nu = 0$ (Eq. 46):
$$0 = \sum_{j,j\neq i} n_j P_{ji} - n_i\sum_{j,j\neq i} P_{ij}$$

Bound-bound cooling (Eq. 47): $Q_{ij} = h\nu_{ij}A_{ji}n_j$.

Total coronal loss (Eq. 50):
$$Q = -\Lambda(T)n_e n_H$$

### 4.7 Chromospheric empirical recipe / 채층 경험적 recipe

Carlsson & Leenaarts (2012), per species $X$ in stage $m$ (Eq. 51):
$$Q_{X_m} = -L_{X_m}E_{X_m}(\tau)\frac{n_{X_m}}{n_X}A_X\frac{n_H}{\rho}n_e\rho$$

### 4.8 Non-equilibrium ionisation / 비평형 이온화

Continuity + rate equation (Eq. 54):
$$\frac{\partial n_i}{\partial t} + \nabla\cdot(n_i\mathbf{v}) = \sum_{j\neq i} n_j P_{ji} - n_i\sum_{j\neq i} P_{ij}$$

Eigenvalue solution (Eq. 56): $\mathbf{n}(t) = \sum_i c_i\mathbf{a}_i e^{\lambda_i t}$, relaxation timescale $\tau = 1/\min(|\lambda_i|)$.

Two-level relaxation (Eq. 58):
$$\tau = \left\{C_{21}+C_{12}+R_{21}\left[1 - \frac{n_1 R_{12}}{n_2 R_{21}}\right]\right\}^{-1}$$

Charge conservation (Eq. 59): $n_e = \sum_{i,j,k} (j-1)n_{ijk}$.

Energy conservation (Eq. 60):
$$e = \frac{3}{2}k_BT\left(\sum_{ijk} n_{ijk} + n_e\right) + \sum_{ijk} n_{ijk}E_{ijk}$$

### 4.9 Generalized Ohm's law and ambipolar diffusion / 일반화된 Ohm 법칙과 ambipolar diffusion

For partially ionized plasma the induction equation includes Hall, ambipolar, and Biermann battery terms:
$$\frac{\partial \mathbf{B}}{\partial t} = \nabla\times\left[(\mathbf{v}\times\mathbf{B}) - \eta_\mathrm{Ohm}\mathbf{J} - \eta_\mathrm{Hall}(\mathbf{J}\times\mathbf{B}) - \eta_\mathrm{amb}((\mathbf{J}\times\mathbf{B})\times\mathbf{B}) + \frac{\nabla P_e}{e n_e}\right]$$

The ambipolar diffusivity $\eta_\mathrm{amb}$ depends explicitly on neutral and ion densities (Khomenko et al. 2014):
$$\eta_\mathrm{amb} = \frac{(\rho_n/\rho)^2}{\sum_i \rho_i \nu_{in}}$$
where $\nu_{in}$ is the ion-neutral collision frequency. This is why non-equilibrium ionisation directly impacts ambipolar heating (Martinez-Sykora et al. 2017).

### 4.10 Photon suction / 광자 석션

Photon suction is a non-LTE effect: in optically thick lines, a population surplus in the upper level (created by e.g. recombination cascades) enhances downward radiative rates, "sucking" population down the levels. The net radiative bracket $[1 - (n_1 R_{12})/(n_2 R_{21})]$ in Eq. 58 encodes this effect — when it is negative (lower level under-populated relative to Saha-Boltzmann), the relaxation timescale becomes especially long and the two-level atom is close to radiative detailed balance, minimising the effective collisional relaxation.

### 4.11 Abbett-Fisher fast method / Abbett-Fisher 빠른 방법

Column-based formal solution (Eq. 61):
$$J_\nu(\tau_\nu) = \frac{1}{2}\int_0^\infty S_\nu(\tau'_\nu)E_1(|\tau_\nu - \tau'_\nu|)d\tau'_\nu$$

Local source approximation (Eq. 63): $J_\nu \approx S_\nu(\tau_\nu)(1 - E_2(\tau_\nu)/2)$.

LTE + Planck-mean opacity (Eq. 65):
$$Q \approx -2\kappa^B\rho\sigma T^4 E_2(\tau^B)$$

### 4.12 Numerical Example: H ionization NLTE vs LTE / 수치 예제: H 이온화 NLTE vs LTE

Consider a chromospheric fluid element at $T = 6000$ K, $n_H = 10^{17}$ m$^{-3}$. In LTE:
- Saha equation gives $n_\mathrm{HII}/n_\mathrm{HI} \approx (2 \times (2\pi m_e k_B T)^{3/2}/h^3) (g_\mathrm{HII}/g_\mathrm{HI})(1/n_e)\exp(-13.6\,\mathrm{eV}/k_BT)$.
- With $k_BT = 0.517$ eV, the Boltzmann factor is $e^{-26.3}\approx 3.7\times 10^{-12}$, so hydrogen is essentially neutral.

In non-LTE with radiation temperature $T_\mathrm{rad}(\mathrm{Ly\alpha}) = 7000$ K (slightly hotter than local $T$):
- The Ly-α radiation pumps the upper levels faster than collisions can thermalise them.
- Radiative ionisation from $n=2$ keeps hydrogen partially ionised even at $T=6000$ K.
- The relaxation time to Saha equilibrium at these conditions is $\tau \sim 10^5$ s, so if a shock passed by 10 min earlier and temporarily heated to 20 kK (full ionisation), the gas retains much of that ionisation for many minutes.

Numerically, $n_e/n_H$ in a RADYN simulation can exceed $10^{-3}$ in the mid-chromosphere at $T \sim 6000$ K, whereas LTE predicts $n_e/n_H \sim 10^{-7}$ — a factor of $10^4$ difference with major consequences for electron density-dependent ambipolar diffusivity and line opacity.

### 4.13 Numerical Example: CRD vs PRD / 수치 예제: CRD vs PRD

For Mg II k (279.6 nm):
- **CRD**: absorbed and emitted photon frequencies are uncorrelated; line profile in emission follows the Voigt profile weighted by $(1-\epsilon)J + \epsilon B$ evaluated at line center.
- **PRD**: outside the Doppler core ($|x|>3$, where $x = (\nu-\nu_0)/\Delta\nu_\mathrm{Dopp}$), emitted photon frequency is correlated with absorbed frequency — Doppler-redistribution with small shift.
- Observationally, the wings of Mg II k show frequency coherence; modelling with CRD smears the line wings and overestimates line-center brightness by factors of 1.5-2.
- Sukhorukov & Leenaarts (2017) 3D PRD calculation costs ~10 s CPU per grid cell per atom — $2\times 10^6\times$ slower than the host MHD step.

### 4.14 Numerical Example: 3D MHD simulation cost / 수치 예제: 3D MHD 시뮬레이션 비용

A typical BIFROST run: $768^3$ grid = $4.5\times 10^8$ cells. At ~5 μs per cell per timestep, one timestep takes ~38 min CPU × N_cores. Adding PRD 3D non-LTE RT (10 s × 4.5e8 / N_cores ≈ 5e7 CPU-hours per timestep) is clearly infeasible. Reducing to 1 s per cell would still require 1.3×10^5 CPU-hours per step — unaffordable. Hence the empirical recipe approach.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1970 ─ Mihalas: "Stellar Atmospheres" — theoretical foundation
1982 ─ Nordlund — 3D granulation + multi-group RT (opacity binning invented)
1984 ─ Mihalas & Mihalas — "Foundations of Radiation Hydrodynamics" textbook
1992 ─ Carlsson & Stein — RADYN, 1D non-LTE radiation hydrodynamics
1992 ─ Ludwig thesis — multi-group refinement (Rosseland bin opacity)
1999 ─ Sollum — chromospheric hydrogen radiation approximation
2000 ─ Skartlien — multi-group scattering for non-LTE source function in 3D
2002 ─ Carlsson & Stein — H non-equilibrium ionisation timescale 10^5 s
2004 ─ Vogler et al. — MURaM code: 3D sunspot simulations
2005 ─ Vogler et al. — MURaM technical paper
2009 ─ Leenaarts & Carlsson — MULTI3D non-LTE RT code
2011 ─ Gudiksen et al. — BIFROST radiation-MHD code
2012 ─ Carlsson & Leenaarts — chromospheric loss empirical recipe
2012 ─ Martinez-Sykora — ambipolar diffusion in 2D solar simulation
2012 ─ Abbett & Fisher — RadMHD fast RT
2012 ─ Freytag et al. — CO5BOLD
2013 ─ Pereira et al. — 11-group LTE multi-group benchmarks observations
2014-16 ─ Golding et al. — non-equilibrium helium ionisation
2017 ─ Sukhorukov & Leenaarts — 3D PRD radiative transfer
2017 ─ Judge — escape probability non-LTE method (100× speedup)
2017 ─ Rempel — extended MURaM for corona
2020 ─ Leenaarts — THIS REVIEW (comprehensive survey)
2020s ─ DKIST era: demands on models continue to rise
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Nordlund (1982) | Original multi-group RT + 3D granulation | Foundational — invented the technique Leenaarts devotes §3 to |
| Carlsson & Stein (1992, 2002) | 1D non-LTE RHD with RADYN | Calibrates chromospheric recipes; establishes non-equilibrium H ionisation |
| Vernazza, Avrett, Loeser (1981) | VAL3C semi-empirical chromospheric model | Still referenced for line cooling balance (Fig. 12) |
| Skartlien (2000) | Multi-group scattering scheme | Extension of Nordlund that underlies non-LTE source treatment in 3D codes |
| Carlsson & Leenaarts (2012) | Chromospheric empirical loss recipes | Centerpiece of §5; authorised approximation for chromospheric cooling |
| Gudiksen et al. (2011) | BIFROST code paper | One of the main codes benchmarked throughout |
| Vogler et al. (2005) | MURaM code paper | Another main code reference |
| Martinez-Sykora et al. (2017) | Ambipolar diffusion enables spicule generation | Ambipolar diffusion term connects to non-equilibrium ionisation (§6) |
| Sukhorukov & Leenaarts (2017) | 3D PRD in MULTI3D | Benchmark for future chromospheric transfer — Leenaarts calls for updating recipes using this |
| Pereira et al. (2013) | 11-group LTE benchmarks solar observations | Demonstrates that photospheric multi-group RT is quantitatively accurate |
| Golding et al. (2014, 2016) | Non-equilibrium helium ionisation | Extends Carlsson & Stein hydrogen method to helium — relevant to §6 |
| Khomenko et al. (2014, 2018) | Multi-fluid partial ionisation | Beyond-single-fluid treatment complementing single-fluid ambipolar description |
| Judge (2017) | Escape probability non-LTE | Topic of §7.2 and one of Leenaarts' five improvement directions |
| CHIANTI (Dere et al. 1997, 2019) | Atomic database | Underlies all coronal loss function calculations (§4) |
| Rempel (2017) | MURaM extension to corona | Addresses TR subgrid issue and grid-resolution-dependent $Q$ |

---

## 7. References / 참고문헌

- Leenaarts, J., "Radiation hydrodynamics in simulations of the solar atmosphere", Living Reviews in Solar Physics (2020) 17:3. https://doi.org/10.1007/s41116-020-0024-x
- Nordlund, Å., "Numerical simulations of the solar granulation. I.", A&A 107:1 (1982).
- Mihalas, D. & Mihalas, B.W., "Foundations of Radiation Hydrodynamics", Oxford University Press (1984).
- Carlsson, M. & Stein, R.F., "Non-LTE radiating acoustic shocks and Ca II K2V bright points", ApJ 397:L59 (1992). https://doi.org/10.1086/186544
- Carlsson, M. & Stein, R.F., "Dynamic hydrogen ionization", ApJ 572:626–635 (2002). https://doi.org/10.1086/340293
- Vogler, A., Shelyag, S., Schussler, M., Cattaneo, F., Emonet, T., Linde, T., "Simulations of magneto-convection in the solar photosphere. MURaM code", A&A 429:335–351 (2005).
- Gudiksen, B.V. et al., "The stellar atmosphere simulation code Bifrost", A&A 531:A154 (2011). https://doi.org/10.1051/0004-6361/201116520
- Freytag, B. et al., "Simulations of stellar convection with CO5BOLD", J. Comput. Phys. 231:919–959 (2012).
- Skartlien, R., "A multigroup method for radiation with scattering in three-dimensional hydrodynamic simulations", ApJ 536:465–480 (2000).
- Carlsson, M. & Leenaarts, J., "Approximations for radiative cooling and heating in the solar chromosphere", A&A 539:A39 (2012). https://doi.org/10.1051/0004-6361/201118366
- Sukhorukov, A.V. & Leenaarts, J., "Partial redistribution in 3D non-LTE radiative transfer in solar-atmosphere models", A&A 597:A46 (2017).
- Abbett, W.P. & Fisher, G.H., "Radiative cooling in MHD models of the quiet Sun convection zone and corona", Sol. Phys. 277:3–20 (2012).
- Judge, P.G., "Efficient radiative transfer for dynamically evolving stratified atmospheres", ApJ 851:5 (2017). https://doi.org/10.3847/1538-4357/aa96a9
- Golding, T.P., Carlsson, M., Leenaarts, J., "Detailed and simplified nonequilibrium helium ionization in the solar atmosphere", ApJ 784:30 (2014).
- Golding, T.P., Leenaarts, J., Carlsson, M., "Non-equilibrium helium ionization in an MHD simulation of the solar atmosphere", ApJ 817:125 (2016).
- Martinez-Sykora, J., De Pontieu, B., Hansteen, V.H., "Two-dimensional radiative magnetohydrodynamic simulations of the importance of partial ionization in the chromosphere", ApJ 753:161 (2012).
- Martinez-Sykora, J. et al., "On the generation of solar spicules and Alfvenic waves", Science 356:1269–1272 (2017).
- Khomenko, E., Vitas, N., Collados, M., de Vicente, A., "Numerical simulations of quiet Sun magnetic fields seeded by the Biermann battery", A&A 604:A66 (2017).
- Pereira, T.M.D. et al., "How realistic are solar model atmospheres?", A&A 554:A118 (2013).
- Vernazza, J.E., Avrett, E.H., Loeser, R., "Structure of the solar chromosphere. III.", ApJS 45:635–725 (1981).
- Landi, E. & Landini, M., "Radiative losses of optically thin coronal plasmas", A&A 347:401–408 (1999).
- Dere, K.P. et al., "CHIANTI — an atomic database for emission lines", A&AS 125:149–173 (1997); updates through 2019 (Version 9).
- Rempel, M., "Extension of the MURaM radiative MHD code for coronal simulations", ApJ 834:10 (2017).
- Bruls, J.H.M.J., Vollmoller, P., Schussler, M., "Computing radiative heating on unstructured spatial grids", A&A 348:233–248 (1999).
- Kunasz, P. & Auer, L.H., "Short characteristic integration of radiative transfer problems", J. Quant. Spectrosc. Radiat. Transf. 39:67–79 (1988).
- Sollum, E., "Dynamic hydrogen ionization", Master's thesis, University of Oslo (1999).
- Leenaarts, J., Carlsson, M., Hansteen, V., Rutten, R.J., "Non-equilibrium hydrogen ionization in 2D simulations of the solar atmosphere", A&A 473:625–632 (2007).
- Ludwig, H.-G., "Nichtgrauer Strahlungstransport in numerischen Simulationen stellarer Konvektion", PhD thesis, Christian-Albrechts-Universitat Kiel (1992).
- Heinemann, T. et al., "Radiative transfer in decomposed domains", A&A 448:731–737 (2006).
