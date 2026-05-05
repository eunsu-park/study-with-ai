---
title: "Solar UV and X-ray Spectral Diagnostics"
authors: [Giulio Del Zanna, Helen E. Mason]
year: 2018
journal: "Living Reviews in Solar Physics"
doi: "10.1007/s41116-018-0015-3"
topic: Living_Reviews_in_Solar_Physics
tags: [spectroscopy, XUV, EUV, corona, transition-region, DEM, CHIANTI, diagnostics]
status: completed
date_started: 2026-04-23
date_completed: 2026-04-23
---

# 59. Solar UV and X-ray Spectral Diagnostics / 태양 UV·X선 분광 진단

---

## 1. Core Contribution / 핵심 기여

이 278쪽 Living Review는 태양 외곽대기(전이영역·코로나)에서 관측되는 optically thin XUV (X-ray 5–150 Å + EUV 150–900 Å + UV 900–2000 Å) 방출선을 이용한 플라스마 진단의 방법론·원자 데이터·관측 결과를 종합한다. Del Zanna와 Mason은 (1) 로켓 실험부터 IRIS·SDO·Hinode까지 XUV 분광기의 역사, (2) optically thin 조건에서의 방출선 형성 이론 (contribution function G(T,N_e), Maxwell 충돌 여기율, 이온화 평형), (3) metastable 준위를 이용한 전자 밀도 N_e 진단과 two-line 비(ratio) 기반 전자 온도 T_e 진단, (4) DEM(T) inversion의 다양한 기법(χ² spline, MCMC, Tikhonov 정칙화, EM loci), (5) CHIANTI v8 원자 데이터베이스와 그 기반이 되는 R-matrix/distorted-wave 계산 기법, (6) 조용한 태양·코로나 홀·활동영역·플레어에 대한 구체적 관측·진단 결과, 그리고 (7) 원소 풍부도 측정과 FIP(First Ionization Potential) 효과를 아우른다. 이 리뷰의 핵심 기여는 2018년 시점의 "최고의 원자 데이터"를 기반으로 수십 개 이온 시퀀스(He-like, Li-like, Be-like, B-like, C-like, ... Fe coronal ions)의 density/temperature 진단 비를 표와 그림으로 정리해, 태양 분광 연구자가 참조할 "표준 핸드북" 역할을 하도록 만든 점이다.

This 278-page Living Review synthesizes the methodology, atomic data, and observational results of optically thin XUV (5–150 Å X-ray + 150–900 Å EUV + 900–2000 Å UV) plasma diagnostics for the solar corona and transition region. Del Zanna and Mason cover: (1) the history of XUV spectrometers from early rocket flights through IRIS/SDO/Hinode; (2) the theory of line formation in optically thin plasmas — the contribution function G(T,N_e), Maxwellian collisional excitation rates, level-population equations, and ion charge-state balance; (3) electron-density N_e diagnostics via line ratios that exploit metastable levels; (4) electron-temperature T_e diagnostics via ratios of lines with different excitation energies within the same ion; (5) methods for differential emission measure (DEM) inversion (χ² spline, MCMC with Bayesian priors, Tikhonov regularization, EM-loci); (6) the CHIANTI v8 atomic database and its underlying R-matrix and distorted-wave scattering calculations; (7) observational results for quiet Sun, coronal holes, active regions, and flares; and (8) elemental abundance measurements and the FIP effect. The core contribution is a unified, up-to-date reference work that tabulates and benchmarks density- and temperature-sensitive ratios across all isoelectronic sequences relevant to the solar XUV spectrum, serving as the standard handbook for current and next-generation spectroscopic missions (Solar Orbiter/SPICE, Aditya-L1, MAGIXS).

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction and Historical Perspective (§§1–2, pp 5–25) / 서론과 기기 역사

태양 코로나는 주로 optically thin 방출을 내며 1 MK 이상의 고온 플라스마이다. 가시광 영역에는 K-corona(자유전자 Thomson scattering)와 F-corona(먼지 산란)가 있고, 금지선 Fe XIV 5303 Å(녹색선)과 Fe X 6374 Å(붉은선)이 식에서 관측된다. XUV 영역(5–2000 Å)은 대기 흡수로 지상 관측이 불가능해 1960년대 로켓 실험으로 시작되었다. Skylab ATM(1973–74)이 최초의 본격적 XUV 분광을, SoHO(1995–)의 CDS·SUMER·UVCS가 고해상도·장기 관측을, Hinode/EIS(2006–)가 고분해능 EUV 영상 분광을, SDO/AIA(2010–)가 7개 EUV 채널의 연속 영상을, IRIS(2013–)가 sub-arcsec UV 전이영역 관측을 제공한다. 저자들은 Table 1과 여러 표로 각 기기의 파장대·분해능·FOV·검교정 상태를 정리한다.

The solar corona emits mostly optically-thin radiation at ≥1 MK. In visible light the K-corona (Thomson scattering by free electrons) and F-corona (dust scattering) are seen in eclipses, along with forbidden lines like Fe XIV 5303 Å and Fe X 6374 Å. The XUV range (5–2000 Å) requires space observations. After 1960s rockets, Skylab ATM (1973–74) provided the first extensive XUV spectra; SoHO (1995–) with CDS/SUMER/UVCS gave high-resolution long-term coverage; Hinode/EIS (2006–) brought high-resolution EUV imaging spectroscopy; SDO/AIA (2010–) provides continuous 7-channel EUV imaging; IRIS (2013–) reaches sub-arcsec UV transition-region spectroscopy. Tables summarize wavelength range, δλ, FOV and calibration state for each instrument.

### Part II: The formation of the XUV spectrum (§3, pp 26–50) / XUV 스펙트럼의 형성

**§3.1 Optically thin line intensity.** 광학적으로 얇은 가정 하에서 관측 intensity I(λ_{ji})는 upper-level 밀도 N_j와 spontaneous decay rate A_{ji}의 시선 적분이다:

$$I(\lambda_{ji}) = \frac{h\nu_{ji}}{4\pi}\int N_j(Z^{+r})\, A_{ji}\, ds \quad [\text{erg cm}^{-2}\text{s}^{-1}\text{sr}^{-1}]. \tag{1}$$

Upper-level population N_j는 다음과 같이 인수분해된다:

$$N_j(Z^{+r}) = \underbrace{\frac{N_j(Z^{+r})}{N(Z^{+r})}}_{\text{level pop.}}\cdot\underbrace{\frac{N(Z^{+r})}{N(Z)}}_{\text{ion fraction}}\cdot\underbrace{\frac{N(Z)}{N_H}}_{\text{abundance }Ab(Z)}\cdot\underbrace{\frac{N_H}{N_e}}_{\sim 0.83}\cdot N_e. \tag{4}$$

이로부터 contribution function이 정의된다:

$$G(N_e, T, \lambda_{ij}) = Ab(Z)\, A_{ji}\, \frac{h\nu_{ij}}{4\pi}\, \frac{N_j(Z^{+r})}{N_e N(Z^{+r})}\, \frac{N(Z^{+r})}{N(Z)}. \tag{6}$$

대부분의 이온에서 G(T)는 좁은 온도 범위에서 날카로운 피크를 보여, 관측된 한 방출선이 사실상 한 온도의 플라스마를 "찍는" 진단 역할을 한다.

In optically thin conditions, observed intensity is the line-of-sight integral of N_j·A_{ji}. The upper-level density is factored into level population, ion fraction, elemental abundance, and electron density (Eq. 4). This yields the contribution function G(N_e,T,λ) (Eq. 6), which packages all the atomic physics (A-values, collision strengths, ion balance, abundance) into one function that peaks sharply in T, so each line effectively samples a narrow temperature slice.

**§3.2 Maxwellian rates.** 코로나 전자는 Maxwell–Boltzmann 분포를 따른다고 가정:
$$f(v) = 4\pi v^2 \left(\frac{m}{2\pi k T_e}\right)^{3/2} e^{-mv^2/2kT_e}. \tag{10}$$
이로부터 충돌 여기율 계수:
$$C^e_{ij}(T_e) = \frac{8.63\times 10^{-6}}{T_e^{1/2}} \frac{\Upsilon_{ij}(T_e)}{g_i}\, e^{-\Delta E_{ij}/kT_e}\;[\text{cm}^3\text{s}^{-1}], \tag{15}$$
여기서 Υ_{ij}(T)는 thermally-averaged collision strength (Burgess 변수 기준 dimensionless). De-excitation rate는 detailed-balance로부터:
$$C^d_{j,i} = \frac{g_i}{g_j}\, C^e_{i,j}\, e^{\Delta E_{i,j}/kT_e}. \tag{19}$$

**§3.4 Level population.** 다준위 statistical equilibrium 방정식(Eq. 20)을 dN_j/dt=0으로 풀어 level populations를 얻는다. Coronal-model 근사(지면준위↑→↑excited level, 그 후 spontaneous decay)는 저밀도 (N_e ≲ 10^8–10^12 cm^{-3}) 조건에서 유효하다. Metastable levels (A ~ 10^0–10^2 s^{-1})는 collisional de-excitation (N_e C^e ~ A)에서 ground와 경쟁해 population이 축적되며, 여기서 density-sensitive line ratio가 생긴다.

**§3.5 Ion charge state.** 이온화 평형 (collisional ionization vs radiative+dielectronic recombination)은 N(Z^{+r})/N(Z) vs T 분포를 결정한다. Fe XII, Fe XIII 같은 iron coronal ions은 ~10^6.2 K에서 peak abundance.

**§3.6–3.7 Optical depth & continuum.** Lyman α, He II 304 Å, Fe XV 284 Å 등 일부 강한 선은 opacity 효과로 분기 비(branching ratio)가 이론값에서 벗어난다. 연속체(free-free, free-bound, two-photon)는 X-ray 온도 진단과 electron density 추정(free-free ∝ N_e²)에 사용.

The coronal plasma is assumed Maxwellian; the rate coefficient C^e (Eq. 15) uses Υ_{ij}(T). De-excitation follows from detailed balance (Eq. 19). Level populations come from the statistical equilibrium equations; the coronal-model approximation (ground excitation + spontaneous decay) is valid at low density. Metastable levels, with A ~ 10^0–10^2 s^{-1}, allow collisional de-excitation to compete with radiative decay — this is the physical origin of density-sensitive line ratios. Ion charge-state balance combines collisional ionization with radiative and dielectronic recombination to give N(Z^{+r})/N(Z) vs T.

### Part III: Satellite lines and atomic data (§§4–5, pp 51–104) / 위성선과 원자 데이터

**§4** Dielectronic capture로 생기는 satellite line(공명선에 가까운 파장)은 이온 population에 무관하게 전자 온도에 민감한 T 진단을 제공한다 — 특히 He-like ion(예: Fe XXV)의 satellite structure. 또한 Li-like ions은 inner-shell excitation으로 satellite line을 만든다. 비Maxwellian 분포를 측정할 수 있는 간접 진단도 가능.

**§5** CHIANTI 패키지는 atomic structure (SUPERSTRUCTURE, AUTOSTRUCTURE, GRASP), electron-ion scattering (R-matrix suites: RMATRX1, UCL·Belfast codes, Breit–Pauli R-matrix, IRON Project), ionization/recombination rates를 통합한 오픈 데이터베이스. 저자들은 isoelectronic sequence별로 (H-like, He-like, Li-like, ..., Ca-like Fe VII까지) 원자 데이터의 정확도·불확실성·벤치마킹(ratio 측정 vs 계산)을 논의한다. Ions의 A-value 불확실성이 10–20% 수준까지 줄어든 반면, forbidden line의 collision strength는 R-matrix 대규모 계산 유무에 따라 2–3배 차이가 난다.

Satellite lines from dielectronic capture (which form very close to resonance lines) provide temperature diagnostics independent of ion balance. High-resolution spectra are required. CHIANTI integrates modern atomic-structure (SUPERSTRUCTURE, AUTOSTRUCTURE, GRASP) and scattering (R-matrix/distorted-wave) calculations. The authors benchmark ion-by-ion, noting that modern APAP-team R-matrix work has reduced uncertainties to 10–20% for strong lines, but factor-of-2–3 differences can persist for some forbidden transitions.

### Part IV: Non-equilibrium effects (§6, pp 105–114) / 비평형 효과

플라스마 동역학 timescale이 이온화/재결합 timescale보다 짧으면 이온화 평형이 깨진다(특히 TR과 플레어 impulsive phase). 또한 coronal 전자 분포는 때때로 Maxwell이 아니라 κ-분포(고에너지 꼬리가 있는) 형태일 수 있다. Figure 51은 O IV 1401 Å와 Si IV 1403 Å의 contribution function이 κ가 감소할수록 저온 쪽으로 이동함을 보여, IRIS 관측의 line-ratio 해석에 영향을 준다. Dudík et al. (2014a, 2015)의 Hinode/EIS transient loop에서 κ=2~3을 시사하는 결과 (Fig. 52) 가 인용된다.

If plasma evolution is faster than ionization/recombination, equilibrium breaks down (especially in TR and flare impulsive phase). Non-Maxwellian electron distributions (κ-distributions with a high-energy tail) can also occur. Figure 51 shows G(T) curves shift toward lower T as κ decreases; Figure 52 presents EIS observations of a transient loop favoring κ=2–3.

### Part V: Emission measure diagnostics (§§7–8, pp 115–128) / 방출측정 진단

**§7.1 Definitions.** If a unique N_e(T) relationship exists (e.g., constant-pressure loop), the column DEM is defined:
$$\mathrm{DEM}(T) \equiv N_e N_H \frac{dh}{dT}\;[\text{cm}^{-5}\text{K}^{-1}], \qquad \int_T \mathrm{DEM}(T)\,dT = \int_h N_e N_H\, dh = EM. \tag{89}$$
Volume EM과 column EM을 구분: $EM_V = \int_V N_e^2 dV$ (cm^{-3}), column $EM = \int_h N_e N_H dh$ (cm^{-5}). Intensity integral becomes:
$$I(\lambda_{ij}) = Ab(Z)\int_T C(\lambda_{ij}, N_e)\, \mathrm{DEM}(T)\, dT. \tag{90}$$
이는 제1종 Fredholm 적분 방정식이며 DEM 역산이 ill-posed 문제가 되는 이유이다.

**§7.2 EM-loci 방법.** 각 선의 I_obs/G(T)를 T의 함수로 그리면 EM_L 상한선을 얻는다. 곡선들이 한 점에서 교차하면 플라스마는 거의 isothermal. 많은 활동영역 core (~3 MK)나 코로나 홀은 near-isothermal distribution을 보인다 (Fig. 54, 93, 96).

**§7.3 Anomalous EM.** Li-like, Na-like ions의 EM이 같은 온도의 다른 isoelectronic sequence 결과보다 ~5배 크게 나오는 문제. 원인으로 (i) charge-state distribution의 고밀도 효과, (ii) Maxwell 분포의 편차 (고에너지 꼬리), (iii) ambipolar diffusion, (iv) 비평형 이온화 등이 제시되지만 완전히 설명 안 됨. He EUV 선도 5배 정도 enhanced인 유사 문제.

**§7.4 DEM inversion 방법.** (i) Withbroe(1975)의 반복법, (ii) MCMC/Bayesian (Kashyap & Drake 1998, PINTofALE), (iii) χ² + spline (XRT_DEM, MPFIT), (iv) maximum entropy (Lagrange multiplier with cubic spline), (v) Tikhonov regularization + GSVD (Hannah & Kontar 2012 DATA2DEM_REG), (vi) Gaussian basis superposition (Del Zanna 2013a). Fig. 56은 MCMC vs spline vs XRT_DEM 비교로 대부분 잘 제약된 T 영역 (log T=6.0–6.6)에서 일치함을 보여준다.

**§8** 실제 관측 결과: Quiet Sun의 DEM은 log T≈5.2에 최소, ~1 MK에 peak, ~2 MK에서 급강하 (Fig. 57). 활동영역 core는 3 MK isothermal. "Warm" 1 MK loops는 near-isothermal cross-section. Moss는 AR loop의 다리 영역 (0.7–1 MK). Fan loops는 sunspot 위에서 cooler (0.7–1 MK).

The column DEM (Eq. 89) makes the intensity integral a Fredholm equation of the first kind (Eq. 90), making inversion ill-posed. EM-loci plots I_ob/G(T) give upper bounds; crossing of curves signals isothermal plasma. Inversion methods include MCMC (PINTofALE), χ²-spline (XRT_DEM), maximum entropy, Tikhonov regularization (DATA2DEM_REG), and Gaussian superposition. Quiet Sun DEM peaks at ~1 MK with a minimum at log T≈5.2; AR cores are near-isothermal at ~3 MK; "warm" 1 MK loops are isothermal in cross-section.

### Part VI: Electron density diagnostics (§§9–10, pp 129–176) / 전자밀도 진단

**§9.1 Line-ratio principle.** Forbidden transitions have A ~ 10^0–10^2 s^{-1}, so collisional de-excitation becomes important when N_e C^e_{m,g} ~ A_{m,g}. Two-level model:
$$N_m = \frac{N_g N_e C^e_{g,m}}{N_e C^e_{m,g} + A_{m,g}}. \tag{95}$$

Limits give the famous S-curve:
- $N_e \to 0$: $I_{m,g} \propto N_e^2$ (like allowed lines, Eq. 96)
- $N_e \to \infty$: $N_m/N_g \to (g_m/g_g)\exp(-\Delta E/kT)$ (Boltzmann), $I_{m,g}\propto N_e$ (Eq. 98)
- Intermediate: $I_{m,g} \propto N_e^\beta$, $1<\beta<2$.

Ratio of forbidden (metastable-fed) to allowed (ground-fed) line is the classical density diagnostic (I^F/I^A).

**Fe XIV 5303 Å example.** Ground config 3s²3p has ²P_{1/2}, ²P_{3/2}. Radiative decay between them is the green coronal line with A=60 s^{-1}. At N_e < 10^8 cm^{-3} the population is nearly all in ²P_{1/2}; at higher N_e the ²P_{3/2} population grows and the 334.2 Å / 353.8 Å ratio changes (Fig. 59).

**§9.2 L-function and emissivity-ratio methods.** When multiple lines from the same ion are available, plot I_ob/G(T,N_e) vs N_e for each; all curves should coincide at the correct density. The emissivity-ratio method:
$$R_{ji} = \frac{I_{ob} N_e \lambda_{ji}}{N_j(N_e,T_e)\, A_{ji}}\cdot\mathrm{Const} \tag{102}$$
plotted vs N_e gives direct measure of consistency. Fig. 61 shows Fe XII lines (186.88, 192.39, 193.51, 195.12, 196.65 Å) converging at log N_e ≈ 8.5–10.

**§9.4 Specific ions.**
- **He-like (C V, O VII, Ne IX, Mg XI, Si XIII, ..., Fe XXV)**: R = z/(x+y) ratio where z is the forbidden ¹S₀→³S₁ decay, x+y are intercombination ³P→¹S₀ (see Table 6). Fig. 63 shows O VII R vs N_e up to 10^12 cm^{-3}, sensitive for flare densities.
- **Be-like (C III, O V, ... Fe XXIII)**: Strongest UV lines. C III 1175 Å multiplet vs 977 Å resonance gives N_e ~ 10^{8-10}; O V 1218/1371 Å; Mg IX 368/706 (Table 7).
- **B-like (N III, O IV, ... Fe XXII)**: Strong diagnostics from 2s²2p–2s2p² transitions. Si X 356/347 Å, S XII 218/227 Å used with CDS/Hinode EIS.
- **C-like, N-like, O-like, ...**: Tables 10–14 exhaustively list transitions, typical log T of peak ion abundance, and the log N_e range where each ratio is useful.
- **O IV 1400 Å multiplet**: IRIS "plasma bombs" reach densities 10^{13} cm^{-3} from S IV lines, but O IV saturates at 10^{11} (Fig. 67). Emissivity-ratio analysis assuming isothermal O IV + S IV (Fig. 68) gives consistent log N_e ≈ 11.7 in flare footpoints.
- **Fe XII 186.88/195.12 Å**: The workhorse Hinode/EIS ratio (Fig. 61) covering log N_e 8–10.
- **Iron coronal ions Fe IX–Fe XIII**: Many transitions in Table 14, with careful blend identification.

**§10 Observations.** Coronal hole N_e ≈ 10^8 cm^{-3}; quiet Sun log N_e ≈ 8.5–9; AR log N_e ≈ 9–10; AR loop feet log N_e ≈ 10–11; flare loop tops log N_e up to 10^{12}. Fig. 89 shows flare footpoints reaching log N_e = 13 in IRIS S IV observations.

Forbidden-line ratios give the density through the balance between radiative decay and collisional de-excitation of metastable levels. The two-level model predicts the classic S-shaped I^F/I^A curve: I ∝ N_e² at low density, I ∝ N_e at high. The L-function and emissivity-ratio methods generalize to many lines. Tables 6–14 exhaustively cover He-like through Al-like ions and Fe IX–Fe XIII. Typical density ranges: coronal holes ~10^8 cm^{-3}, quiet Sun 10^{8.5–9}, active regions 10^{9–10}, loop feet 10^{10–11}, flare footpoints up to 10^{12–13}.

### Part VII: Electron temperature diagnostics (§§11–12, pp 177–213) / 전자온도 진단

**§11.1 Same-ion ratios.** Two lines from the same ion but with different excitation energies Δ_{g,j}, Δ_{g,k}:
$$\frac{I_{g,j}}{I_{g,k}} = \frac{\Delta E_{g,j}\,\Upsilon_{g,j}}{\Delta E_{g,k}\,\Upsilon_{g,k}}\exp\!\left[\frac{\Delta E_{g,k}-\Delta E_{g,j}}{k_B T}\right]. \tag{105}$$
Sensitivity requires $(\Delta E_{g,k}-\Delta E_{g,j})/k_B T \gg 1$.

- **He-like G-ratio**: G = (x+y+z)/w where w is resonance, x,y intercombination, z forbidden. Fig. 90 shows O VII G(T) decreasing monotonically from ~1.4 at log T=6.0 to ~0.3 at log T=7.0. Also "P-ratio" = (3p→1s)/(2p→1s) and higher-n ratios (Fig. 91 for Si XIII) are alternative T diagnostics.
- **Be-like (Table 17)**: Ratio of resonance 2s²→2s2p ¹P₁ to intercombination 2s²→2s2p ³P₁ — e.g., Mg IX 368/706 Å ratio giving 1 MK quiet-Sun T; O V 629/1218 Å for TR temperatures. These lines are very close in wavelength and insensitive to density.
- **Iron ions (Table 19)**: Fe VIII–Fe XIII EUV diagnostics using pairs like Fe XII 193.51/364.47 Å or Fe XIII 202/10749 Å; forbidden/allowed ratios are potentially very sensitive but lines are at very different wavelengths requiring multi-instrument co-observation.

**§11.4 T from satellite lines.** Ratio of dielectronic-capture satellite to parent resonance line is a clean T diagnostic independent of ion balance — but needs very high spectral resolution.

**§11.5 T from X-ray continuum.** Slope of free-free continuum is directly sensitive to T (GOES X-ray flux ratio gives isothermal T in flares); RESIK observations (Fig. 93) show EM-loci of continuum crossing at ~10 MK in flare peak.

**§12 Observational results.** Coronal hole T ≈ 0.8–1.3 MK (Mg IX 706 Å); quiet Sun near-isothermal at log T ≈ 6.0 (Mg IX off-limb at 1.35 MK); AR core ~3 MK (isothermal, Fig. 96); AR warm loops ~1 MK; flare loops 10–30 MK (Fe XVI, Fe XVIII, Fe XXIII, Fe XXV).

Same-ion line-ratio temperature diagnostics follow Eq. 105: the ratio is sensitive when (ΔE_k−ΔE_j) >> k_BT. The He-like G-ratio and Be-like resonance/intercombination ratios are canonical diagnostics. Satellite lines give T independent of ion balance. X-ray continuum slope measures T directly. Typical values: coronal holes 0.8–1.3 MK, quiet Sun ~1 MK, AR cores ~3 MK, flares 10–30 MK.

### Part VIII: Line widths, abundances, and conclusions (§§13–15, pp 196–234) / 선폭·풍부도·결론

**§13 Line widths.** Observed width combines natural + collisional (Voigt) + thermal (Doppler) + non-thermal turbulence. FWHM of a Gaussian thermal line is $\Delta\lambda_{FWHM} = \lambda (8 k_B T \ln 2 / M c^2)^{1/2}$. Non-thermal velocities ξ in quiet Sun TR lines ~20 km/s; coronal AR outflows show excess widths 40–60 km/s; flare lines up to 200 km/s non-thermal (MHD turbulence or unresolved flows).

**§14 Abundances and FIP effect.** Low-FIP elements (< 10 eV: Fe, Si, Mg, Ca) are enhanced by factor ~3–4 in slow solar wind and AR cores ("coronal abundances"); high-FIP elements (> 10 eV: O, Ne, Ar) retain photospheric abundances. Fast solar wind has near-photospheric composition. Diagnostic methods: (i) Pottasch (1963) approximation for mean G(T), (ii) Widing–Feldman (1989) ratio, (iii) DEM-based simultaneous inversion of abundances + DEM. Results: AR cores FIP bias ≈ 3; quiet Sun variable; coronal holes near-photospheric; flares also near-photospheric (with some variation).

**§15 Conclusions.** Progress in atomic data (10–20% accuracy for strong lines via APAP R-matrix) and in joint imaging-spectroscopy (Hinode + SDO + IRIS) has been transformational. Outstanding issues: (i) Li-/Na-like anomaly, (ii) EUV He-line enhancement, (iii) non-Maxwellian distributions, (iv) time-dependent ionization in dynamic TR, (v) absolute abundance variability between SS/AR/CH/flare, (vi) need for new X-ray spectrometers (MAGIXS sounding rocket, 6–20 Å) since SMM era.

Line widths combine thermal Doppler, natural, collisional, and non-thermal turbulent contributions. Non-thermal velocities are 20 km/s (quiet Sun TR), 40–60 km/s (AR outflows), up to 200 km/s (flares). The FIP effect enhances low-FIP elements (Fe, Si, Mg) by ~3–4 in slow solar wind and AR cores; fast solar wind and coronal holes retain near-photospheric composition. Outstanding issues include the Li/Na-like emission-measure anomaly, EUV He enhancement, non-Maxwellian κ-distributions, and the need for new X-ray spectrometers.

### Part IX: Quantitative result highlights / 정량 결과 요약

**Quiet Sun (QS) / 조용한 태양**:
- EM distribution: minimum at log T=5.2, peak at 1 MK, log EM ≈ 27.5 at peak (Fig. 57)
- N_e: 10^{8.5–9.0} cm^{-3} in corona, 10^{9–10} in TR footpoints
- Non-thermal width ξ ≈ 20 km/s (Si IV, C IV, O V)
- DEM near-isothermal at ~1 MK off-limb

**Coronal holes (CH) / 코로나 홀**:
- N_e ≈ 0.5–1 × 10^8 cm^{-3} off-limb at 1.1 R_☉ (50% lower than QS)
- T ≈ 0.8–1.3 MK, gradient drops with height
- Composition: near-photospheric (low-FIP elements NOT enhanced)
- Plume base: T ≈ 0.78 MK, isothermal

**Active regions (AR) / 활동영역**:
- Core loops (log T=6.5): isothermal at 3 MK, FIP bias = 3–4
- Warm loops (1 MK): near-isothermal cross-section, N_e ≈ 10^{9.5}
- Moss (loop footpoints): T ≈ 0.7–1 MK, N_e ≈ 10^{10}
- Fan loops (sunspot): cooler, T ≈ 0.7 MK

**Flares / 플레어**:
- Loop-top: T up to 30 MK (Fe XXIV, Fe XXV), N_e up to 10^{12}
- Footpoints (IRIS): N_e up to 10^{13} from S IV 1406/1404 Å ratio
- Composition: near-photospheric
- Non-thermal widths: up to 200 km/s (MHD turbulence or unresolved flows)

### Part X: How to use this review / 이 리뷰의 사용법

For a practical researcher, the review's most-used items are:
1. **Tables 6, 7, 8, 10, 11, 12, 13, 14**: density-diagnostic line pairs per isoelectronic sequence, with usable log N_e range and log T of peak abundance.
2. **Table 17, 18, 19**: temperature-diagnostic ratios for Be-like, C-like through Ni-like ions, and Fe coronal ions.
3. **Figs. 59 (Fe XIV), 61 (Fe XII), 63 (O VII), 66 (CDS composite), 67 (O IV/S IV)**: reference density-sensitivity curves.
4. **Fig. 57 (QS DEM), 54, 56 (DEM comparisons)**: canonical DEM shapes.

CHIANTI v8 underlies all calculations in the review. For numerical work, the *chiantipy* Python package (and IDL SolarSoft CHIANTI) exposes emiss_calc, g_of_t, dem_fit routines to reproduce every figure in the review.

The quick-reference decision tree: (i) If you want T, look for two same-ion allowed lines with very different ΔE; (ii) if you want N_e, look for a ratio involving a metastable-fed line; (iii) if you want DEM, observe ≥ 10 lines over a broad T range and invert; (iv) if you want abundance, use DEM simultaneously or use Pottasch's G(T_max) approximation.

---

## 3. Key Takeaways / 핵심 시사점

1. **Contribution function G(T,N_e) is the central object.** G packages all atomic physics (abundance, A-value, collision strength, ion balance) into a function of plasma parameters, enabling the clean separation of atomic from plasma problems. Because G peaks sharply in T for most lines (FWHM Δlog T ≈ 0.2–0.3), each observed XUV line effectively samples a narrow temperature slice — this is *why* XUV spectroscopy is such a precise T diagnostic.
   **기여함수 G(T,N_e)가 분광 진단의 핵심 개체**다. G는 원소 풍부도·A-값·충돌 강도·이온화 분율을 하나의 함수로 묶어 원자물리와 플라스마 상태를 분리 가능하게 만들고, 대부분 이온에서 T 방향으로 좁게 피크를 이루므로 각 방출선이 특정 온도의 "slice"를 대표한다.

2. **Density diagnostics rely on metastable levels (A ~ 1–100 s^{-1}).** When N_e C^e_{m,g} approaches A_{m,g}, collisional de-excitation competes with radiative decay; the metastable population ratio N_m/N_g changes with density, and this shows up in the observed line-ratio S-curve. No metastable level → no density sensitivity. Classic examples: Fe XIV 334/353 Å (³P), Fe XII 186.88/195.12 Å, O IV 1401/1404 Å, S IV 1406/1404 Å.
   **밀도 진단은 metastable 준위의 존재에 의존**한다. A 값이 작은 metastable 준위가 있어야 N_e C^e ~ A 조건에서 collisional de-excitation이 경쟁하고 line ratio가 N_e에 민감해진다. 대표 예: Fe XIV 334/353, Fe XII 186.88/195.12, O IV 1401/1404.

3. **Temperature diagnostics exploit (ΔE_k−ΔE_j)/k_BT >> 1.** Two same-ion allowed lines with very different excitation energies give exp((ΔE_k−ΔE_j)/k_BT) sensitivity. The He-like G-ratio = (x+y+z)/w and Be-like resonance/intercombination ratio (Mg IX 368/706 Å) are the canonical examples. Satellite-line ratios are ion-balance-independent and cleaner but require resolution ≤ 5 mÅ.
   **온도 진단은 (ΔE_k−ΔE_j)/k_BT>>1 조건을 이용**한다. 같은 이온에서 여기 에너지가 크게 다른 두 선의 비가 T에 지수 민감도를 보인다. He-like G-ratio와 Be-like Mg IX 368/706 Å이 대표적.

4. **DEM inversion is an ill-posed Fredholm problem.** The intensity integral I = Ab∫C(T,N_e)·DEM(T)dT requires inverting a smooth kernel C against a positive solution DEM. Common approaches: MCMC (Bayesian), χ²-spline (XRT_DEM), Tikhonov regularization (GSVD, DATA2DEM_REG), maximum entropy, Gaussian superposition. Methods agree in well-constrained T regions (log T ≈ 6.0–6.6) but can diverge in poorly-constrained regions (Fig. 56).
   **DEM 역산은 ill-posed Fredholm 문제**이다. MCMC·spline·Tikhonov·최대엔트로피 방법이 각기 다른 regularization을 제공하며, 잘 제약된 T 영역에서는 일치하지만 경계 영역에서는 발산할 수 있다.

5. **Solar-atmospheric parameters span huge ranges.** Quiet corona log T ≈ 6.0 (1 MK), N_e ≈ 10^{8–9} cm^{-3}, log EM ≈ 27–28 cm^{-5}. AR cores log T ≈ 6.5 (3 MK), N_e ≈ 10^{9–10}. Flare loops log T up to 7.3 (20 MK), N_e up to 10^{12}. TR (10^5 K) high-density plasma bombs reach N_e ≈ 10^{13}. Any single diagnostic works only over a limited N_e or T range — the review's tables flag usable ranges per ratio.
   **태양 대기 파라미터 범위가 매우 넓다.** 조용한 코로나 ~1 MK, 10^8–10^9 cm^{-3}; 활동영역 ~3 MK, 10^9–10^10; 플레어 loop top ~20 MK, 10^{12}. 진단 비마다 사용 가능한 범위가 제한되므로 Tables 6–19가 매 이온 별로 usable range를 명시한다.

6. **CHIANTI and R-matrix atomic data underpin everything.** The review's diagnostic results only became reliable after modern R-matrix calculations (Burgess & Tully 1992; APAP team; Badnell; Liang & Badnell) reduced A-value and collision-strength uncertainties to 10–20%. Previous diagnostics using older data often gave inconsistent densities/temperatures; Section 5 benchmarks each isoelectronic sequence against observations.
   **CHIANTI와 R-matrix 원자 계산이 모든 진단의 토대**다. 현대 R-matrix 계산으로 A-값과 collision strength 불확실성이 10–20%로 줄어 이전 진단의 불일치가 대부분 해결되었다.

7. **Non-equilibrium effects complicate TR and flare diagnostics.** Dynamic TR has timescales shorter than ionization equilibration (107 s for C IV); κ-distributions (non-Maxwellian tails) shift contribution functions toward lower T by Δlog T ≈ 0.2–0.4 as κ decreases from 10 to 2 (Fig. 51). The Li-/Na-like "anomaly" (EM enhanced by ×5) likely reflects this, and IRIS observations of TR "bombs" require κ-distribution interpretation.
   **비평형 효과(시간의존 이온화, κ-분포)가 TR/플레어 진단을 복잡하게 한다.** κ 값이 낮을수록 contribution function이 저온 쪽으로 이동하며, Li-/Na-like 이상 현상은 이러한 효과의 흔적으로 보인다.

8. **FIP effect fingerprints plasma origin.** Low-FIP elements (Fe, Si, Mg) enhanced by ×3–4 in slow solar wind and AR cores mark "coronal abundance"; fast wind and coronal holes keep photospheric composition. Direct measurement via spectral-line ratios + DEM provides a remote-sensing complement to in-situ Ulysses/ACE/Parker Probe composition data. Different diagnostic methods (Pottasch, Widing–Feldman, DEM-based) can disagree by factor of 2, so consistency checks are essential.
   **FIP 효과는 플라스마 기원의 지문**이다. 저-FIP 원소(Fe, Si, Mg)는 slow solar wind와 활동영역 core에서 ×3–4 증가하고, fast wind와 coronal hole은 광구 조성에 가깝다. 이는 in-situ 관측과 원격 분광 관측을 연결하는 핵심 진단이다.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Line intensity and contribution function / 방출선 강도와 기여함수

Optically thin line radiance:
$$\boxed{\, I(\lambda_{ji}) = \frac{h\nu_{ji}}{4\pi}\int N_j(Z^{+r})\, A_{ji}\, ds \,}. \tag{1}$$

Factoring the upper-level density:
$$N_j(Z^{+r}) = \frac{N_j(Z^{+r})}{N(Z^{+r})}\cdot\frac{N(Z^{+r})}{N(Z)}\cdot\frac{N(Z)}{N_H}\cdot\frac{N_H}{N_e}\cdot N_e. \tag{4}$$

Definition of G:
$$G(N_e,T,\lambda_{ij}) = Ab(Z)\, A_{ji}\,\frac{h\nu_{ij}}{4\pi}\,\frac{N_j(Z^{+r})}{N_e N(Z^{+r})}\,\frac{N(Z^{+r})}{N(Z)}, \tag{6}$$
so that
$$I(\lambda) = \int_s G(N_e,T,\lambda)\, N_e N_H\, ds. \tag{5}$$

**Interpretation / 해석**: Ab(Z) — 원소 풍부도 (e.g., Fe/H≈3.2×10^{-5}); A_{ji} — 자발방출 확률 (allowed ~10^{10} s^{-1}, forbidden ~1 s^{-1}); hν_{ij}/4π — 한 전이당 에너지·고립각; N_j/N(Z^{+r}) — upper level 분율 (coronal model 근사에서 N_e 의존); N(Z^{+r})/N(Z) — 이온화 평형 (T 의존). G는 대부분 log T에서 FWHM ~ 0.2–0.3의 날카로운 피크를 갖는다.

### 4.2 Collisional excitation / 충돌 여기

Rate coefficient for Maxwellian electrons:
$$\boxed{\, C^e_{ij}(T_e) = \frac{8.63\times 10^{-6}}{T_e^{1/2}}\frac{\Upsilon_{ij}(T_e)}{g_i}\exp\!\left(-\frac{\Delta E_{ij}}{kT_e}\right)\,\text{cm}^3\text{s}^{-1} \,}, \tag{15}$$

with thermally-averaged collision strength
$$\Upsilon_{i,j}(T_e) = \int_0^\infty \Omega_{i,j}(E_j)\, e^{-E_j/kT_e}\, d(E_j/kT_e). \tag{16}$$

Detailed balance de-excitation:
$$C^d_{j,i} = \frac{g_i}{g_j}\, C^e_{i,j}\, e^{\Delta E_{i,j}/kT_e}. \tag{19}$$

**Numerical example / 수치 예.** Fe XII 195.12 Å (Δλ≈6.35×10^{-11} cm, E≈63.5 eV ≈ 7.4×10^5 K). At T_e=1.5 MK (log T=6.2, near peak ionization), Υ≈4.5, g_i=4, so
$$C^e \approx \frac{8.63\times 10^{-6}}{(1.5\times 10^6)^{1/2}}\frac{4.5}{4}\exp\!\left(-\frac{7.4\times 10^5}{1.5\times 10^6}\right) \approx 7\times 10^{-9}\cdot 1.12\cdot 0.61 \approx 4.8\times 10^{-9}\;\text{cm}^3\text{s}^{-1}.$$
For N_e=10^9 cm^{-3}, excitation rate N_e C^e ≈ 4.8 s^{-1}, comparable to strong metastable A-values, giving the observed density sensitivity.

### 4.3 Level population and two-level density diagnostic / 준위 개체수와 두 준위 밀도 진단

Statistical equilibrium:
$$\frac{dN_j}{dt} = \sum_{k>j} N_k A_{k,j} + \sum_{k>j}N_k N_e C^d_{k,j} + \sum_{i<j}N_i N_e C^e_{i,j} + \cdots - N_j(\cdot\cdot\cdot) = 0. \tag{20}$$

For a two-level ground (g) + metastable (m) system:
$$\boxed{\, \frac{N_m}{N_g} = \frac{N_e C^e_{g,m}}{N_e C^e_{m,g} + A_{m,g}} \,}. \tag{95}$$

Limits:
- $A_{m,g} \gg N_e C^e_{m,g}$: $N_m \propto N_g N_e$, so $I_{m,g} = h\nu N_m A_{m,g} \propto N_e^2$.
- $A_{m,g} \ll N_e C^e_{m,g}$: $N_m/N_g = (g_m/g_g)\exp(-\Delta E/kT)$ (Boltzmann), $I_{m,g}\propto N_e$.

**Critical density / 임계 밀도**:
$$N_e^{crit} = A_{m,g}/C^e_{m,g}(T_e).$$
Fe XIV ²P_{3/2} has A=60 s^{-1}; at T=2 MK, C^e ≈ 3×10^{-8} cm³/s → N_e^{crit} ≈ 2×10^9 cm^{-3}, consistent with Fig. 59 which shows the 334/353 Å ratio turning over near log N_e = 9.5.

### 4.4 Density-sensitive ratio curves / 밀도 민감 비 곡선

Ratio of forbidden (metastable-fed) to allowed (ground-fed):
$$\frac{I^F_{m\to g}}{I^A_{k\to g}} = \frac{A_{m,g}\, h\nu_{m,g}\, N_m}{A_{k,g}\, h\nu_{k,g}\, N_g} = f\!\left(\frac{N_e C^e_{m,g}}{A_{m,g}}\right).$$

**Fe XII 186.89/195.12 ratio**: Both lines are 3s²3p³ → 3s²3p²3d transitions. 186.89 Å is populated partly from metastable ⁴P_{5/2,3/2} (fed by ground excitation); 195.12 Å from ground ⁴S_{3/2}. Ratio increases from ~0.1 (log N_e=8) to ~0.5 (log N_e=10), plateauing at log N_e > 11. See Fig. 61.

**Density ratio at two temperatures.** Since both Υ_{g,m} and Υ_{g,k} depend weakly on T while Boltzmann factor is shared, the ratio is largely T-insensitive — advantageous. But if one line is fed through a higher-excited metastable, Boltzmann factor differs → extra T dependence.

### 4.5 Temperature-sensitive ratio / 온도 민감 비

From Eq. 15, two allowed lines from the ground to different excited levels j,k:
$$\boxed{\, \frac{I_{g,j}}{I_{g,k}} = \frac{\Delta E_{g,j}\,\Upsilon_{g,j}}{\Delta E_{g,k}\,\Upsilon_{g,k}}\exp\!\left[\frac{\Delta E_{g,k}-\Delta E_{g,j}}{k_B T}\right]\,}. \tag{105}$$

Sensitivity: d(ln ratio)/d(ln T) = (ΔE_k−ΔE_j)/k_BT ≫ 1 needed. If ΔE_k−ΔE_j = 10 eV = 1.16×10^5 K and T=10^6 K, factor = 0.12; ratio changes by factor of e^0.12·Δln T — modest. For T=10^5 K (TR), factor = 1.16, giving strong sensitivity.

**Fe XII 195 Å vs Fe XIII 202 Å cross-ion T diagnostic** (combining different ions assumes ionization equilibrium): ratio I(Fe XIII 202)/I(Fe XII 195) as function of T peaks at log T = 6.2–6.3 boundary between the two ion fractions.

### 4.6 Emission measure and DEM / 방출측정과 미분방출측정

Column EM:
$$EM = \int_h N_e N_H\, dh\;[\text{cm}^{-5}]. \tag{92}$$

DEM definition:
$$\mathrm{DEM}(T) = N_e N_H\frac{dh}{dT}\;[\text{cm}^{-5}\text{K}^{-1}], \qquad EM = \int_T \mathrm{DEM}(T)\, dT. \tag{89, 92}$$

Intensity equation:
$$\boxed{\, I(\lambda_{ij}) = Ab(Z)\int_T C(\lambda_{ij}, N_e)\, \mathrm{DEM}(T)\, dT \,}. \tag{90}$$

With M observed lines and the DEM discretized on a T grid (N points), this is an N-unknown linear system of M equations — often underdetermined, requiring regularization. One common approach: Tikhonov:
$$\hat{\mathrm{DEM}} = \arg\min_{\mathrm{DEM}\geq 0}\left\{\sum_{i=1}^M\left(\frac{I_i^{obs}-\int C_i(T)\mathrm{DEM}(T)dT}{\sigma_i}\right)^2 + \lambda\int\left(\frac{d^2\mathrm{DEM}}{dT^2}\right)^2 dT\right\}.$$

**Effective temperature**:
$$\log T_{\text{eff}} = \frac{\int G(T,N)\,\mathrm{DEM}(T)\,\log T\, dT}{\int G(T)\,\mathrm{DEM}(T)\, dT}. \tag{93}$$

**Typical values.** Quiet Sun log(N_e N_H dh) ≈ 27–28 (cm^{-5}); AR cores 28–29; flare loops 29–30. Volume EM for a 1 MK AR loop of N_e=10^9, volume 10^{27} cm^3: EM_V = 10^{18}·10^{27} = 10^{45} cm^{-3}.

### 4.7 EM-loci diagnostic / EM-loci 진단

For each observed line i, EM-loci curve:
$$EM_L^i(T) = \frac{I^i_{obs}}{Ab(Z) \cdot C_i(T, N_e)}.$$

This is an upper bound on the emission measure at each T (assuming all emission comes from that T). If multiple lines give curves crossing at one T, the plasma is isothermal at that T. If curves define an envelope, the DEM is the lower envelope (schematically).

### 4.8 Ionization equilibrium / 이온화 평형

Steady state in charge state r:
$$N_r[\alpha^I_r + \alpha^{PI}_r] = N_{r-1}[\alpha^I_{r-1}] + N_{r+1}[\alpha^R_{r+1} + \alpha^D_{r+1}],$$
where α^I, α^{PI}, α^R, α^D are rate coefficients for collisional ionization, photoionization, radiative recombination, dielectronic recombination. Solution gives N(Z^{+r})/N(Z) vs T. For iron, peak abundance temperatures log T: Fe IX=5.85, Fe X=6.0, Fe XI=6.1, Fe XII=6.2, Fe XIII=6.25, Fe XIV=6.3, Fe XV=6.35, Fe XVI=6.45, Fe XVII=6.75, Fe XVIII=6.85, Fe XXIII=7.1, Fe XXIV=7.15, Fe XXV=7.5.

**Non-equilibrium timescale / 비평형 시간척도.**
$$\tau_{\text{ion,rec}} = \frac{1}{N_e(\alpha^I + \alpha^R + \alpha^D)}.$$
For C IV at T=10^5 K with N_e=10^{10}: τ_{ion} ≈ 2×10^{-3} s (very fast), τ_{rec} ≈ 88 s (slow). If the plasma cools or heats on timescales < τ_{rec}, charge-state distribution lags the temperature, and one can see, e.g., Fe XIV emission at temperatures that equilibrium would assign to Fe XII.

**Photoionization vs collisional balance**: In the quiet lower corona (≥10^8 cm^{-3}), collisional ionization dominates over photoionization for most ions. In coronal holes above 1.5 R_☉, electron densities drop below 10^7 cm^{-3}, and photoexcitation from the disk becomes non-negligible (~W(r)·I_ν term in Eq. 25) — the next Living Review extension promises to cover this.

### 4.9 κ-distributions / κ 분포 (Non-Maxwellian)

When the electron distribution has a high-energy tail characterized by κ > 3/2:
$$f_\kappa(E) = A_\kappa\frac{2}{\sqrt{\pi}(k_BT)^{3/2}}\frac{E^{1/2}}{(1+E/[\kappa-3/2]k_BT)^{\kappa+1}},$$
which reduces to Maxwellian as κ→∞. The rate coefficient C^e_{ij} increases relative to Maxwellian for transitions where ΔE_{ij} >> k_BT — i.e., for resonance lines of highly ionized species excited by the tail. Low-κ (e.g., κ=2) shifts G(T) curves of Li-like and Na-like ions toward lower T by Δlog T ≈ 0.2–0.4, potentially explaining the Li/Na anomaly noted in §7.3.

### 4.10 Worked example: Fe XIII 202/203 density ratio / 작동 예: Fe XIII 밀도 진단

Fe XIII ground configuration 3s²3p² has ³P_{0,1,2} levels. The 202.04 Å line (3p²→3p3d ³D) is mainly fed from ground ³P_0; the 203.83 Å self-blend (³P_{1,2} → 3p3d) is fed from metastable ³P_1 and ³P_2. At N_e→0: metastable populations are low, ratio 203/202 small. At N_e→∞: Boltzmann with g_{³P1}=3, g_{³P2}=5, g_{³P0}=1, so population ratio (³P_1+³P_2)/³P_0 approaches (3+5)/1 = 8 (times Boltzmann factor). Thus the 203.83/202.04 ratio grows by more than an order of magnitude across log N_e = 8–11. At T=2 MK, this ratio equals ~1 at log N_e ≈ 9.5, making it an excellent Hinode/EIS density diagnostic for active-region corona. Tables 10 and 14 of the review tabulate the useful ranges and required wavelength corrections for blends.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1942  Edlén                           Fe XIV 5303 Å hot-corona identification
1961  Pottasch                        DEM concept from solar UV
1963  Pottasch                        Coronal abundance method
1969  Gabriel & Jordan                He-like G-ratio density diagnostic
1972  Mariska                         TR structure review
1976  Athay                           Solar chromosphere-corona textbook
1978  Withbroe                        First iterative DEM inversion
1981  Feldman                         FIP-effect discovery in solar wind
1982  Gabriel & Mason                 Density-sensitive ratio principles
1985  Meyer                           Solar abundance compilation
1992  Arnaud & Raymond                Ionization balance for iron
1992  Mariska                         TR spectroscopy textbook
1994  Mason & Monsignori-Fossi        Previous major review
1996  Dere et al.                     CHIANTI v1 public release
1998  Kashyap & Drake                 MCMC DEM (PINTofALE)
2006  Hinode/EIS launch               High-resolution EUV spectroscopy
2008  Phillips, Feldman, Landi        UV and X-ray spectroscopy textbook
2012  Landi et al.                    CHIANTI v7
2013  Del Zanna et al.                Hinode/EIS atomic-data benchmarking
2013  Young et al.                    IRIS launch, sub-arcsec UV spectra
2014  Bradshaw & Raymond              Non-equilibrium ionization review
★ 2018  Del Zanna & Mason              THIS REVIEW (Living Reviews)
2018  CHIANTI v9 release              Updated ion data
2020  Solar Orbiter launch            SPICE for XUV spectroscopy
2024  MAGIXS sounding rocket (planned) X-ray 6-20 Å spectroscopy
```

This review extends and updates four prior standard references: Mason & Monsignori-Fossi (1994), Mariska (1992), Phillips/Feldman/Landi (2008), and the original Gabriel & Mason (1982) diagnostics paper. It consolidates 60 years of XUV spectroscopy and sets the baseline for Solar Orbiter/SPICE (launched 2020) and future X-ray missions.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Edlén (1942) | Identification of Fe XIV 5303 Å as highly-ionized iron established coronal temperatures of ~10^6 K | Foundational — all XUV spectroscopy is downstream of hot-corona recognition |
| Pottasch (1963) | Introduced emission measure method for solar abundances | Direct ancestor of DEM formalism in §7 |
| Gabriel & Jordan (1969) | He-like G-ratio diagnostic (z/w, (x+y+z)/w) for temperature | Section 11.1.1 expands this to modern atomic data |
| Gabriel & Mason (1982) | Principles of density-sensitive line ratios via metastable levels | Section 9 is the modern tabulation of their framework |
| Mariska (1992), *The Solar Transition Region* | Comprehensive TR spectroscopy textbook | Pre-SoHO counterpart; Del Zanna & Mason update every diagnostic with CHIANTI v8 |
| Mason & Monsignori-Fossi (1994) | Previous A&A Review: *Spectroscopic diagnostics in the VUV for solar and stellar plasmas* | Direct predecessor — this 2018 Living Review supersedes it |
| Dere et al. (1997); Young et al. (2003); Del Zanna et al. (2015) | CHIANTI database versions 1, 5, 8 | Underlying atomic data; this review uses v8 throughout |
| Arnaud & Raymond (1992); Bryans et al. (2009) | Ionization equilibrium for Fe and other elements | Section 3.5.5 Fig. gives N(Z^{+r})/N(Z) vs T |
| Phillips, Feldman & Landi (2008), *Ultraviolet and X-ray Spectroscopy of the Solar Atmosphere* | Textbook-level reference | Complementary; this review focuses on diagnostics while Phillips et al. covers broader theory |
| Bradshaw & Raymond (2014) | Review of non-equilibrium ionization in corona | Extends §6 discussion; relevant for dynamic TR and flares |
| Asplund, Grevesse, Sauval & Scott (2009) | Solar chemical composition reference | Used for reference photospheric abundances in §14 FIP analysis |
| Dudík et al. (2014a, 2015) | κ-distribution diagnostics with IRIS and Hinode/EIS | Source for §6.2 non-Maxwellian figures 51–52 |

---

## 7. References / 참고문헌

- Del Zanna, G., & Mason, H. E. (2018). Solar UV and X-ray spectral diagnostics. *Living Reviews in Solar Physics*, 15:5. DOI: 10.1007/s41116-018-0015-3
- Mason, H. E., & Monsignori-Fossi, B. C. (1994). Spectroscopic diagnostics in the VUV for solar and stellar plasmas. *A&A Review*, 6, 123–179.
- Mariska, J. T. (1992). *The Solar Transition Region*. Cambridge University Press.
- Phillips, K. J. H., Feldman, U., & Landi, E. (2008). *Ultraviolet and X-ray Spectroscopy of the Solar Atmosphere*. Cambridge University Press.
- Gabriel, A. H., & Mason, H. E. (1982). Solar physics (spectroscopic diagnostics). In *Applied Atomic Collision Physics* (Vol. 1), Academic Press.
- Dere, K. P., Landi, E., Mason, H. E., Monsignori-Fossi, B. C., & Young, P. R. (1997). CHIANTI — an atomic database for emission lines. *A&AS*, 125, 149–173.
- Del Zanna, G., Dere, K. P., Young, P. R., Landi, E., & Mason, H. E. (2015). CHIANTI — An atomic database for emission lines. Version 8. *A&A*, 582, A56.
- Withbroe, G. L. (1975). The analysis of XUV emission lines. *Solar Physics*, 45, 301–317.
- Kashyap, V., & Drake, J. J. (1998). Markov-Chain Monte Carlo reconstruction of emission measure distributions. *ApJ*, 503, 450–466.
- Hannah, I. G., & Kontar, E. P. (2012). Differential emission measures from the regularized inversion of Hinode and SDO data. *A&A*, 539, A146.
- Gabriel, A. H., & Jordan, C. (1969). Interpretation of solar helium-like ion line intensities. *MNRAS*, 145, 241–248.
- Pottasch, S. R. (1963). The lower solar corona: interpretation of the ultraviolet spectrum. *ApJ*, 137, 945–966.
- Edlén, B. (1942). Die Deutung der Emissionslinien im Spektrum der Sonnenkorona. *Zeitschrift für Astrophysik*, 22, 30.
- Bradshaw, S. J., & Raymond, J. (2014). Non-equilibrium radiation and ionization. In *Radiative Processes in Optically Thin Plasmas*. Springer.
- Asplund, M., Grevesse, N., Sauval, A. J., & Scott, P. (2009). The chemical composition of the Sun. *ARA&A*, 47, 481–522.
- Dudík, J., et al. (2014a, 2015). Non-Maxwellian distributions and IRIS/EIS observations. *ApJ* and *A&A* series.
