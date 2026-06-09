---
title: "Pre-Reading Briefing: Solar UV and X-ray Spectral Diagnostics"
paper_id: "59"
topic: Living_Reviews_in_Solar_Physics
date: 2026-04-23
type: briefing
---

# Solar UV and X-ray Spectral Diagnostics: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Del Zanna, G., & Mason, H. E. (2018). Solar UV and X-ray spectral diagnostics. *Living Reviews in Solar Physics*, 15:5. DOI: 10.1007/s41116-018-0015-3
**Author(s)**: Giulio Del Zanna, Helen E. Mason (DAMTP, Cambridge)
**Year**: 2018

---

## 1. 핵심 기여 / Core Contribution

이 논문은 태양 코로나 및 전이영역에서 optically thin X-ray/극자외선(EUV)/자외선(UV) 방출선을 이용한 플라스마 진단(spectral diagnostic) 기법을 278쪽에 걸쳐 종합적으로 정리한 Living Review이다. 저자들은 (1) 고해상도 XUV 분광계(OSO부터 SDO·IRIS까지)의 역사, (2) 광학적으로 얇은 방출선의 형성 이론 (contribution function G(T,N_e), collisional excitation, 이온화 평형), (3) 전자 밀도 N_e·전자 온도 T_e·미분방출측정(DEM)·원소 풍부도(abundance) 진단법, (4) CHIANTI 원자 데이터베이스와 관련 원자 계산 기법, (5) 조용한 태양·코로나 홀·활동영역·플레어 각각에 대한 관측 결과들을 통합적으로 다룬다.

This review provides a comprehensive treatise on spectroscopic plasma diagnostics using optically thin X-ray, EUV, and UV emission lines from the solar corona and transition region. The authors cover: (1) a historical tour of XUV high-resolution spectrometers from rocket flights through OSO, Skylab, SMM, SoHO, Hinode, SDO, and IRIS; (2) theoretical foundations of line formation (contribution function G(T,N_e), collisional rates with Maxwellian distributions, level population equations, ion charge-state balance); (3) diagnostics of electron densities (line-ratio methods using metastable levels), electron temperatures, differential emission measure (DEM), and elemental abundances; (4) the CHIANTI atomic database and underlying atomic-structure/scattering codes; and (5) observational results across quiet Sun, coronal holes, active regions, and flares, with emphasis on modern single-ion diagnostics.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

1960년대 로켓 실험으로 시작된 태양 XUV 분광 관측은 대기 흡수 때문에 반드시 우주로 나가야 한다. Edlén(1942)의 녹색 코로나선(Fe XIV 5303 Å) 동정 이후 "코로나 온도가 수백만 K"라는 사실이 확립되었고, 1960~70년대 Skylab·OSO 시리즈가 대규모 XUV 스펙트럼을 제공했다. 1995년 SoHO, 2006년 Hinode/EIS, 2010년 SDO, 2013년 IRIS가 차례로 고분해능 분광과 영상 진단을 현대화했다.

XUV observations must be done from space because Earth's atmosphere blocks those wavelengths. Following Edlén's (1942) identification of the Fe XIV 5303 Å green coronal line as highly ionized iron, rocket flights of the 1960s and the Skylab/OSO era (1970s) collected the first high-quality XUV spectra. SoHO (1995), Hinode/EIS (2006), SDO/AIA (2010), and IRIS (2013) progressively improved spatial, spectral, and temporal resolution. The paper reviews the full arc — instruments, atomic data, and diagnostic methods — as they co-evolved over ~60 years.

### 타임라인 / Timeline

```
1942 -- Edlén: Fe XIV 5303 Å (hot corona confirmed)
1960s -- First XUV rocket flights
1973 -- Skylab ATM: first extensive XUV spectroscopy (OV, Si XII, Mg X)
1980 -- SMM (BCS, FCS, UVSP, XRP)
1991 -- Yohkoh (SXT, BCS)
1995 -- SoHO (CDS, SUMER, UVCS)
1996 -- CHIANTI v1 atomic database public
2006 -- Hinode/EIS (high-resolution EUV imaging spectroscopy)
2010 -- SDO/AIA (7 EUV channels, continuous imaging)
2013 -- IRIS (UV TR spectroscopy at sub-arcsec)
2018 -- This review (Del Zanna & Mason)
```

---

## 3. 필요한 배경 지식 / Prerequisites

**Atomic physics / 원자물리**: Einstein A/B coefficients, statistical weights, collision strength Ω(E), Maxwell–Boltzmann velocity distribution, LS/jj coupling, metastable levels. Reading: Rybicki & Lightman (*Radiative Processes*), Condon & Shortley (*Theory of Atomic Spectra*).

**Plasma physics / 플라스마 물리**: Coronal approximation (excitation from ground + spontaneous decay dominate), ionization equilibrium (collisional ionization, radiative and dielectronic recombination), optically thin radiative transfer.

**Mathematical tools / 수학적 도구**: Fredholm integral of the first kind (DEM inversion), regularization (Tikhonov, MCMC, spline), numerical integration of G(T)·DEM.

**Solar atmosphere / 태양 대기**: Chromosphere → transition region → corona temperature stratification, magnetic structures (quiet Sun, coronal holes, active regions, flares), typical T∼10^4 K (chromosphere) → 10^5–10^6 K (TR) → 10^6–10^7 K (corona/flares).

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| XUV | X-ray (5–150 Å) + EUV (150–900 Å) + FUV/UV (900–2000 Å) 합친 영역 |
| Optically thin / 광학적으로 얇음 | Self-absorption negligible; emitted photons escape freely |
| Contribution function G(T,N_e) | Encodes atomic physics for a line: ion fraction × level population × A-value × abundance |
| Emission measure EM | ∫ N_e N_H dh (column) or ∫ N_e² dV (volume); amount of emitting plasma |
| DEM ξ(T) or DEM(T) | dEM/dT: how plasma is distributed in temperature along line of sight |
| Collision strength Υ(T) | Thermally averaged dimensionless excitation cross-section |
| Metastable level | Long-lived upper level whose population competes with ground → density-sensitive lines |
| Density-sensitive line ratio | Ratio involving metastable decay; e.g., Fe XII 186.89/195.12 Å |
| Temperature-sensitive line ratio | Ratio of two lines with very different excitation energies; e.g., Be-like resonance/intercombination |
| Coronal model approximation | Only collisional excitation from ground + spontaneous decay considered (low density limit) |
| CHIANTI | Open atomic database with energies, A-values, effective collision strengths for XUV diagnostics |
| Ionization equilibrium | Steady state between ionization and recombination; gives N(Z^{+r})/N(Z) vs T |

---

## 5. 수식 미리보기 / Equations Preview

**(1) Line intensity (optically thin):**
$$I(\lambda_{ji}) = \frac{h\nu_{ji}}{4\pi}\int N_j(Z^{+r}) A_{ji}\, ds \quad [\text{erg cm}^{-2}\text{s}^{-1}\text{sr}^{-1}]$$
Radiance from an emission line is the line-of-sight integral of the upper-level population times spontaneous-decay rate. 업준위 밀도·A-값의 시선 적분.

**(2) Contribution function formulation:**
$$I(\lambda_{ji}) = \int_s G(N_e, T, \lambda_{ji})\, N_e N_H\, ds$$
with
$$G(N_e, T, \lambda_{ji}) = \mathrm{Ab}(Z) A_{ji} \frac{h\nu_{ij}}{4\pi} \frac{N_j(Z^{+r})}{N_e N(Z^{+r})} \frac{N(Z^{+r})}{N(Z)}.$$
Separates atomic physics G(T,N_e) from plasma properties (N_e, T, element abundance). 원자물리와 플라스마 상태의 분리.

**(3) Maxwellian rate coefficient:**
$$C^e_{ij}(T_e) = \frac{8.63\times 10^{-6}}{T_e^{1/2}}\frac{\Upsilon_{ij}(T_e)}{g_i}\exp\!\left(-\frac{\Delta E_{ij}}{k T_e}\right)\;[\text{cm}^3\text{s}^{-1}]$$
Collisional excitation rate given thermally-averaged collision strength Υ. Maxwell 분포 하의 충돌 여기율.

**(4) Differential emission measure and intensity integral:**
$$\mathrm{DEM}(T) = N_e N_H \frac{dh}{dT}\;[\text{cm}^{-5}\text{K}^{-1}], \qquad I(\lambda_{ij}) = \mathrm{Ab}(Z)\int_T C(\lambda_{ij}, N_e)\,\mathrm{DEM}(T)\, dT.$$
DEM은 시선 방향 온도 분포; 관측된 여러 선의 강도를 통해 Fredholm 1종 적분을 역산해 구한다.

**(5) Two-level density diagnostic limits:**
$$\frac{N_m}{N_g} = \frac{N_e C^e_{g,m}}{N_e C^e_{m,g} + A_{m,g}}\;\Rightarrow\;\begin{cases}I_{m,g}\propto N_e^2 & N_e\ll A/C^e \\ I_{m,g}\propto N_e & N_e\gg A/C^e.\end{cases}$$
At low density metastable population ∝ N_e² (like allowed line); at high density ∝ N_e (Boltzmann saturation) — this gives the S-curve of a density-sensitive line ratio.

---

## 6. 읽기 가이드 / Reading Guide

278쪽 리뷰이므로 순서대로 다 읽기보다 목적에 맞게 탐색하는 것이 효율적이다. 최초 읽기 권장 순서:

1. **§1–2 (~25 pp)**: Introduction + 기기 역사. Skim 위주.
2. **§3 (pp 26–50)**: Spectral-line formation — 본 리뷰의 이론적 핵심. 모든 수식 꼼꼼히.
3. **§7 (pp 115–124)**: EM·DEM 정의와 inversion 기법.
4. **§9.1–9.2 (pp 129–134)**: Ne line-ratio 원리 + L-function/emissivity-ratio 방법.
5. **§11.1 (pp 178–187)**: Te from same-ion line ratios (He-like G-ratio, Be-like resonance/intercomb.).
6. **§14 (pp 214–233)**: FIP effect와 abundance 측정법.
7. **§15**: 남은 미해결 문제 (Li-like/Na-like anomaly, He-line enhancement, non-thermal distributions).

The 278-page review is best read by topic rather than linearly. Treat §3 (line formation) and §7 (EM/DEM) as required theoretical reading; the density (§9) and temperature (§11) chapters are reference tables you look up per ion. The conclusions (§15) and non-equilibrium (§6) sections highlight still-open problems that motivate future work.

---

## 7. 현대적 의의 / Modern Significance

**Why this review matters today / 현재의 의의**:

- **CHIANTI ubiquity**: 거의 모든 태양 분광 연구가 CHIANTI를 사용; 이 리뷰는 CHIANTI v8 기준의 진단 결과를 체계적으로 정리한 참고서이다.
- **Solar Orbiter / SPICE, Aditya-L1, MAGIXS**: 2020년대의 새 분광기들이 관측할 EUV/X-ray 영역과 연결되는 진단 지침을 제공.
- **Coronal heating / wind acceleration**: 전자 밀도·온도·FIP bias 측정은 코로나 가열과 태양풍 기원 규명을 위한 최우선 관측이며, 이 리뷰가 해당 방법론의 표준을 정립.
- **Non-equilibrium / κ-distribution**: IRIS와 flare 관측에서 떠오른 이슈로, 향후 하드 X-ray/RHESSI-류 자료와 연계될 때 이 리뷰의 프레임이 출발점이 된다.
- **Machine-learning DEM**: 본 리뷰가 정리한 regularized inversion은 최근 딥러닝 기반 DEM 추정 연구의 baseline 역할을 한다.

For anyone doing solar XUV spectroscopy today — from Hinode/EIS, IRIS, SDO/AIA analysis to Solar Orbiter/SPICE planning — this review is the modern standard reference that supersedes the older Mariska (1992), Phillips/Feldman/Landi (2008) textbooks by updating every density/temperature diagnostic with the latest atomic data (R-matrix calculations; CHIANTI v8). It also serves as a gateway for machine-learning DEM inversions and for interpreting future sub-MK EUV observations with SPICE.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
