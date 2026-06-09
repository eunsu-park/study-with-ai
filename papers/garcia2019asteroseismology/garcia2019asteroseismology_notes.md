---
title: "Asteroseismology of Solar-Type Stars"
authors: ["Rafael A. García", "Jérôme Ballot"]
year: 2019
journal: "Living Reviews in Solar Physics"
doi: "10.1007/s41116-019-0020-1"
topic: Living_Reviews_in_Solar_Physics
tags: [asteroseismology, solar-type-stars, p-modes, Kepler, CoRoT, scaling-relations, stellar-rotation, magnetic-activity]
status: completed
date_started: 2026-04-23
date_completed: 2026-04-23
---

# 65. Asteroseismology of Solar-Type Stars / 태양형 별의 성진학

---

## 1. Core Contribution / 핵심 기여

이 Living Review는 태양형 주계열 냉각 왜성(F, G, K 스펙트럼형)을 대상으로 한 성진학 분야의 관측·이론·데이터 분석을 종합한다. 저자들은 태양이라는 "가장 잘 연구된 별"에서 검증된 헬리오지진학 방법론이 어떻게 CoRoT, Kepler, K2, TESS 우주망원경을 통해 원거리 별로 확장되었는지를 추적하고, 확률적으로 여기되는 p-mode(음향 모드) 스펙트럼의 모델링 및 물리적 해석을 체계적으로 설명한다. 핵심 요소는 (i) 파워 스펙트럼 밀도의 구조(포톤 잡음, granulation, activity, rotation peaks, acoustic p-mode hump), (ii) p/g/mixed 모드의 cavity 이론과 turning-point 근사, (iii) 스케일링 관계를 통한 질량·반지름 결정 ($\Delta\nu \propto \sqrt{\langle\rho\rangle}$, $\nu_{\max}\propto g/\sqrt{T_{\rm eff}}$), (iv) 회전 분열에서 내부 회전을 추출하는 방법, 그리고 (v) p-mode 주파수 이동(frequency shift)을 이용해 태양 외 별의 자기 활동 주기를 탐지하는 방법이다.

This Living Review synthesises the observational techniques, oscillation theory, and spectral-analysis tools that define modern asteroseismology of cool solar-type dwarfs (spectral types F, G, K). The authors trace how helioseismology, validated on the Sun, has been extended to hundreds-to-thousands of distant stars by the CoRoT, Kepler, K2, and TESS space missions. Five central strands are developed: (i) the architecture of the observed power spectrum (photon noise, granulation, stellar activity, rotational peaks, and the acoustic p-mode envelope); (ii) the cavity theory of p-, g-, and mixed modes, built from the Brunt–Väisälä and Lamb frequencies; (iii) solar-calibrated scaling relations between the large separation Δν, the frequency of maximum power ν_max, and the effective temperature, yielding stellar masses and radii to a few percent; (iv) the measurement of internal rotation from rotational splittings of multiplets; and (v) the detection of magnetic-activity cycles in other stars via temporal variations of p-mode frequencies, heights, and widths. The Sun is repeatedly shown to be a "normal" star in the stellar-asteroseismology context, setting the stage for the PLATO era.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction and Solar-Like Oscillations in a Helioseismic Context (§1–§2) / 도입과 헬리오지진학적 맥락

저자들은 "태양형 별"을 고전 불안정대(classical instability strip)의 붉은 가장자리 아래에 위치한 냉각 주계열 왜성으로 정의한다 (Fig. 1, Kiel diagram). 이 별들은 표면 대류층(surface convection zone)이 확률적으로 음향 모드를 여기·감쇠시키는 공통 특성을 공유한다. Goldreich & Keeley (1977), Goldreich & Kumar (1988), Samadi & Goupil (2001), Belkacem+ (2008)이 turbulent convection이 음향 모드의 확률적 여기원임을 확립했다.

The authors restrict "solar-type stars" to cool main-sequence dwarfs below the red edge of the classical instability strip (Fig. 1). These stars share stochastic driving of acoustic modes by turbulent surface convection — not the heat-engine (κ) mechanism of classical pulsators. This makes their spectra universal in form: a Gaussian-like p-mode hump on top of a granulation background.

태양은 이 분야의 정답지이다. SoHO (1995–)의 GOLF(Doppler), MDI, VIRGO/SPM(intensity) 덕분에 태양 내부 음속, 밀도, 회전 프로파일, 대류층 바닥, 헬륨 존재비가 정밀하게 결정되어 있다 (§2). 태양에서는 개별 g-mode 후보가 일부 보고되었으나 (García+2007 GOLF; Fossat+2017) 여전히 논쟁적이다 (Schunker+2018). 태양형 주계열 별에서는 순수 g-mode도 mixed mode도 MS stage에선 검출되지 않았다.

The Sun is the ground truth: decades of GOLF (Doppler) and VIRGO/SPM (intensity) on SoHO yield its sound-speed, density, rotation profile, base of convection zone, and He abundance with high precision. Individual solar g-modes remain controversial (García+2007; Fossat+2017; counter: Schunker+2018). Neither pure g-modes nor mixed modes have been unambiguously detected in MS solar-type stars, so all stellar inferences rely on p-modes.

### Part II: Asteroseismic Observations (§3) / 성진학 관측

**Observational requirements** (§3): 긴 연속 관측, 수분 이하의 sampling (모든 acoustic cutoff > Nyquist 보장, 태양형의 경우 > 500–8000 μHz), minimal gaps(정규 gap은 Dirac comb을 만들어 sidebands 발생). 지상 single-site는 day/night 주기로 24h sidebands를 피할 수 없어 BiSON, SONG 같은 네트워크가 필요. 관측 방식은 Doppler velocity (높은 S/N, 예: GOLF 약 300 factor for modes) 혹은 intensity photometry (낮은 S/N ≈ 30, 그러나 우주망원경이 다수 목표를 동시 관측 가능).

Observational requirements: long uninterrupted time series, sampling faster than ~1 min (so the acoustic cut-off is below Nyquist), and minimal gaps (regular gaps inject sidebands via the window-function convolution). Observables: (a) Doppler velocity (GOLF, BiSON, SONG ground network) with superior mode S/N (factor ≈300) but photon-hungry; (b) photometry (VIRGO/SPM, CoRoT, Kepler, K2, TESS) with lower mode S/N (≈30) but simultaneous multi-star observation of hundreds of targets.

**Historical firsts**: α Cen A (Schou & Buzasi 2000/2001 WIRE; Bouchy & Carrier 2001/2002 Doppler), Procyon (controversy resolved by Arentoft+2008 multi-site), η Boo (Kjeldsen+1995). 공간 관측 시대는 CoRoT (2006, ~12 solar-like targets)부터 본격화되었고 Kepler (2009, 512 short-cadence targets 동시 관측)가 혁명을 완성.

**Mission tally** — WIRE, MOST, CoRoT, Kepler/K2, TESS. TESS는 아직 태양형 MS에서는 제한적 (밝은 별 위주, π Men, TOI-197). 향후 PLATO (2026 예정)가 주력.

### §3.1 Power Spectrum Structure / 파워 스펙트럼 구조

Fig. 5는 CoRoT의 HD 52265 PSD를 보여준다. 주파수별 물리:
- $\nu<10\,\mu\mathrm{Hz}$: rotation peaks + harmonics (starspot modulation)
- $10$–$1000\,\mu\mathrm{Hz}$: stellar activity slope + granulation continuum (Harvey law)
- p-mode envelope: Gaussian-like bump 중심이 $\nu_{\max}$
- high-ν: photon noise flat floor

Fig. 5 (HD 52265) anatomizes the PSD: (i) below ~10 μHz, rotation peaks and their harmonics; (ii) a power-law activity slope blending into the granulation continuum (Harvey law); (iii) the Gaussian-like p-mode hump centred on ν_max; (iv) a flat photon-noise floor at high frequency.

Fig. 6 (16 Cyg A) zooms into the p-mode region showing the regular comb pattern — pairs of (ℓ=0, ℓ=2) and (ℓ=1, ℓ=3) modes separated by Δν, with the tiny small separation δν₀₂ inside each pair.

### Part III: Theory of Oscillations (§4) / 진동 이론

**Eigenvalue problem**: 태양형 별은 구형 대칭으로 가정되어 oscillations are described by eigenfrequencies $\omega_{n,\ell,m}$ and eigenmodes characterised by three integers:
- $n$: radial order (p-mode는 양수, g-mode는 음수)
- $\ell$: angular degree (0=radial, 1=dipole, 2=quadrupole, 3=octupole)
- $m$: azimuthal order, $-\ell\le m\le +\ell$

구형 대칭 상태에서는 $\nu_{n,\ell,m}=\nu_{n,\ell}$ ($m$에 degenerate). 회전이 이 degeneracy를 깬다 (§4.2).

**Turning-point wave equation (Eq. 1)**:
$$\frac{d^2\xi_r}{dr^2}+K(r)\xi_r=0,\quad K(r)=\frac{\omega^2}{c^2}\left(\frac{N^2}{\omega^2}-1\right)\left(\frac{S_\ell^2}{\omega^2}-1\right)$$
- $N$ = Brunt–Väisälä frequency (buoyancy)
- $S_\ell = \sqrt{\ell(\ell+1)}c/r$ = Lamb frequency (horizontal acoustic)
- $c$ = sound speed

Two propagation regimes:
- $\omega > N$ AND $\omega > S_\ell$ → **p-mode cavity** (acoustic, high-frequency, outer envelope)
- $\omega < N$ AND $\omega < S_\ell$ → **g-mode cavity** (buoyancy, low-frequency, radiative interior)

§4.1 p-modes는 태양형 별에서 유일하게 관측되는 모드. Inner turning point $r_t$는 $S_\ell(r_t)=\omega$를 만족하며 $r_t=c(r_t)L/\omega_{n,\ell}$ ($L=\ell+1/2$). 낮은 ℓ 모드는 더 깊이, 높은 n 모드도 더 깊이 침투 (Fig. 9).

§4.1.1 p-modes are acoustic waves, stochastically excited, stable against non-adiabatic processes such as the κ mechanism. Asymptotically ($n\gg\ell$), Tassoul (1980) predicted the comb pattern. Outer turning point set by acoustic cutoff $\omega_c=c_s/(2H)$ (isothermal approximation), ≈5600 μHz for the Sun; modes above are not reflected and produce pseudo-modes / HIPS (García+1998 GOLF, Karoff 2007 ground, Jiménez+2015 Kepler).

§4.1.2 g-modes exist only where $N^2>0$ (radiative zones). For the Sun, expected g-mode periods are >35 min. They are evanescent in the convective envelope, so their surface amplitudes are tiny.

§4.1.3 Mixed modes appear in subgiants and red giants where the evanescent barrier between cavities thins; they have g-character in the core and p-character in the envelope (Scuflaire 1974; η Boo — Kjeldsen+1995). They are uniquely sensitive to the core. MS solar-type stars do not show them.

### §4.2 Rotation and Splittings / 회전과 분열

회전이 $m$-degeneracy를 깬다. 1차 섭동 (Eq. 5):
$$\delta\omega_{n,\ell,m}=m\iint K_{n,\ell,m}(r,\theta)\,\Omega(r,\theta)\,dr\,d\theta$$
$K_{n,\ell,m}$은 회전 kernel이며 eigenmode에 의해 결정. 강체 회전(solid-body) 가정하에서 Ledoux (1951):
$$\delta\omega_{n,\ell,m}=m(C_{n,\ell}-1)\Omega \quad (\text{Eq. 6})$$
태양형 p-mode 고차 ($n\gtrsim 10$)의 경우 $C_{n,\ell}\lesssim 10^{-2}\approx 0$. g-mode는 $C_{n,\ell}\approx 1/[\ell(\ell+1)]$. 관측 가능한 splitting (Eq. 7):
$$\nu_{n,\ell,m}=\nu_{n,\ell}-m\nu_s,\quad \nu_s\equiv\Omega/(2\pi)$$

Rotation lifts the $m$-degeneracy. Rotational kernels $K_{n,\ell,m}$ weight $\Omega(r,\theta)$ in the mode cavity. For solid-body rotation the Ledoux formula simplifies to Eq. (6). Pure p-modes have $C_{n,\ell}\approx 0$ for high $n$, so the observed splitting directly equals the rotation frequency (Eq. 7). Mixed modes in subgiants give access to core rotation because g-like portions have non-zero Ledoux coefficients.

**Asymmetries**: large latitudinal differential rotation (Gizon & Solanki 2004), very fast rotation (Deheuvels+2017), or magnetic fields (Gough & Thompson 1990; Kiefer & Roth 2018) can generate asymmetric splittings and amplitude asymmetries.

### §4.3 Mode Visibility / 모드 가시도

Disk-integration makes high-ℓ modes cancel. The observed amplitude (Eq. 8) $a_{n,\ell,m}=r_{\ell,m}(i)V_\ell A$ factorises into a geometric inclination-dependent part and an ℓ-dependent visibility:
$$V_\ell^2=(2\ell+1)\pi\left[\int_0^1 P_\ell(\mu)W(\mu)\mu\,d\mu\right]^2 \quad (\text{Eq. 9})$$
with $W(\mu)$ limb-darkening/weighting. In practice ℓ=0, 1, 2 dominate, some ℓ=3 is visible, ℓ≥4 is negligible (Fig. 10).

Disk-integration washes out spatial small-scale modes. Only ℓ=0,1,2 (and sometimes ℓ=3) are usefully measured. The inclination factor (Eq. 10)
$$r_{\ell,m}^2(i)=\frac{(\ell-|m|)!}{(\ell+|m|)!}[P_\ell^{|m|}(\cos i)]^2$$
determines which m-components are seen. Pole-on ($i=0$): only $m=0$. Equator-on ($i=90°$): only even $\ell+m$ components. This is why the joint fit of $(i, \nu_s)$ exhibits a "banana"-shape likelihood (Fig. 14).

### §4.4 Frequency Separations / 주파수 분리

**Large separation** (Eq. 11):
$$\Delta\nu_\ell(n)\equiv \nu_{n,\ell}-\nu_{n-1,\ell}$$
First-order asymptotic (Tassoul 1980, Eq. 12): $\nu_{n,\ell}\approx\Delta\nu(n+\ell/2+1/4+\varepsilon)$.
$$\Delta\nu=\left[2\int_0^R\frac{dr}{c}\right]^{-1} \quad (\text{Eq. 13})$$
acoustic travel time across the star.

**Small separation** (Eq. 14):
$$\delta\nu_{\ell,\ell+2}(n)=\nu_{n,\ell}-\nu_{n-1,\ell+2}$$
Second-order asymptotic (Eq. 15):
$$\delta\nu_{\ell,\ell+2}(n)\simeq -(4\ell+6)\frac{\Delta\nu_\ell(n)}{4\pi^2\nu_{n,\ell}}\int_0^R\frac{dc}{dr}\frac{dr}{r}$$
small separation is dominated by the sound-speed gradient near the core ($1/r$ weight), so it probes central composition (hydrogen burning).

δν₀₂/Δν ratio (Roxburgh & Vorontsov 2003) is especially valued because near-surface systematic effects largely cancel.

### Part IV: Spectral Analysis (§5) / 스펙트럼 분석

**Échelle diagram (§5.1)**: plot ν vs. (ν mod Δν). Modes of the same ℓ line up on nearly vertical ridges. Fig. 13 shows three Kepler stars: KIC 6603624 (clean MS), KIC 3656476 (slightly evolved), KIC 11026764 (subgiant with bumped ℓ=1 mixed mode at ~900 μHz).

에셸 다이어그램은 ℓ-identification의 표준 도구이며 subgiant에서 mixed mode가 만드는 "꺾임(bump)"이 진화 단계의 지표가 된다.

**Modelled spectrum (§5.2)**:
$$S(\nu)=B(\nu)+P(\nu)$$
- $B(\nu)$: background (photon noise $W$ + Harvey components, Eq. 20)
$$H_i(\nu)=\frac{\xi_i\sigma_i^2\tau_i}{1+(2\pi\nu\tau_i)^{\alpha_i}}$$
- $P(\nu)=\sum_{n,\ell}M_\ell(\nu;H,\Gamma,\nu_s,i)$, where $M_\ell$ is a sum of Lorentzians (Eq. 22):
$$L(\nu;\nu_0,\Gamma,H)=\frac{H}{1+\left(\frac{2(\nu-\nu_0)}{\Gamma}\right)^2}$$

The observed PSD follows a 2-dof χ² distribution about the limit spectrum because Fourier real/imaginary parts are Gaussian (central limit theorem).

**Mode height and width** (Eqs. 23, 24):
$$H=\frac{\langle|F(\nu)|^2\rangle}{16\pi^2\eta^2\nu_0^2},\quad \Gamma=\frac{\eta}{\pi}$$
Mode lifetime $\tau_{\rm mode}=1/(\pi\Gamma)$.

**MLE (§5.3)**: the likelihood (Eq. 30) is
$$\mathcal{L}(Y;\mathbf{p})=\prod_{i=1}^n\frac{1}{S(\nu_i;\mathbf{p})}\exp\left[-\frac{Y_i}{S(\nu_i;\mathbf{p})}\right]$$
Minimised via modified Newton. Hessian gives Cramér–Rao lower-bound errors. For a radial mode, Libbrecht (1992) (Eq. 31):
$$\sigma_\nu=\sqrt{f(\beta)\frac{\Gamma}{4\pi T}}$$
$T$=observation time, $\beta=B/H$ background-to-height ratio.

**Bayesian methods (§5.4)**: posterior via Bayes' theorem (Eq. 32) $p(\mathbf{p}|Y,I)\propto p(\mathbf{p}|I)p(Y|\mathbf{p},I)$. Priors encode physics (e.g., inclination $p(i)di=\sin i\,di$ for isotropy). MCMC with Metropolis–Hastings (Eq. 35) and parallel tempering (Eq. 36) are standard for multi-modal posteriors. Corsaro & De Ridder (2014) apply nested sampling.

**Local vs global fits (§5.5)**: local fits single pairs (ℓ=0,2) in narrow windows; global fits all orders simultaneously, sharing parameters (e.g., single $i$ and $\nu_s$). Global is preferred for precision (Appourchaux+2008, Mathur+2013a).

**Model comparison (§5.6)**: Wilks (1938) test for nested models — $\Lambda=2(\ln\mathcal{L}_1-\ln\mathcal{L}_0)\sim\chi^2(\Delta\text{dof})$. Bayesian odds ratio $O_{ij}=p(M_i|Y)/p(M_j|Y)$ (Eq. 39).

### §5.7 Global Seismic Parameters / 전역 성진학 매개변수

정밀 모드 fitting 대신 두 개의 전역 파라미터만을 빠르게 추출하는 파이프라인이 발달. $\nu_{\max}$는 p-mode 엔벨로프에 Gaussian fit. $\Delta\nu$는 (i) PSD의 autocorrelation의 최대 lag, 또는 (ii) PSD의 Fourier transform에서 $\tau=2/\Delta\nu$ 피크. $A_{\max}$는 엔벨로프 내 모드 총 power에서 얻는다.

Fast global pipelines (A2Z, COR, OCT, SYD) extract $(\Delta\nu, \nu_{\max}, A_{\max})$ for thousands of stars in minutes. Essential for population studies.

### Part V: Inferences on Stellar Structure (§6) / 항성 구조 추론

**§6.1 Scaling relations for mass and radius**: solar-calibrated (Kjeldsen & Bedding 1995):

$$\boxed{\frac{\Delta\nu}{\Delta\nu_\odot}\approx\left(\frac{M}{M_\odot}\right)^{1/2}\left(\frac{R}{R_\odot}\right)^{-3/2}} \quad (\text{Eq. 43})$$

$$\boxed{\frac{\nu_{\max}}{\nu_{\max,\odot}}\approx\left(\frac{M}{M_\odot}\right)\left(\frac{R}{R_\odot}\right)^{-2}\left(\frac{T_{\rm eff}}{T_{\rm eff,\odot}}\right)^{-1/2}} \quad (\text{Eq. 44})$$

with $\Delta\nu_\odot=135.1\pm 0.1\,\mu\mathrm{Hz}$, $\nu_{\max,\odot}=3090\pm 30\,\mu\mathrm{Hz}$, $T_{\rm eff,\odot}=5770\,\mathrm{K}$ (Huber+2011 using 21 yrs of VIRGO).

Inverting:
$$\frac{M}{M_\odot}\approx\left(\frac{\nu_{\max}}{\nu_{\max,\odot}}\right)^3\left(\frac{\Delta\nu}{\Delta\nu_\odot}\right)^{-4}\left(\frac{T_{\rm eff}}{T_{\rm eff,\odot}}\right)^{3/2}$$
$$\frac{R}{R_\odot}\approx\left(\frac{\nu_{\max}}{\nu_{\max,\odot}}\right)\left(\frac{\Delta\nu}{\Delta\nu_\odot}\right)^{-2}\left(\frac{T_{\rm eff}}{T_{\rm eff,\odot}}\right)^{1/2}$$

**Combined relation (Eq. 45)**: $\Delta\nu\propto M^{-1/4}T_{\rm eff}^{3/8}\nu_{\max}^{3/4}$. Weak M dependence → universal $\Delta\nu$–$\nu_{\max}$ power law with $b\approx 0.75$ (Eq. 46, Stello+2009a; Fig. 19 Kepler 1700 stars).

**Physical basis**: Δν scales as the inverse acoustic radius, $\Delta\nu\propto\sqrt{\langle\rho\rangle}$ by homology (Belkacem+2013). $\nu_{\max}$ scales as the acoustic cutoff $\omega_c\propto g/\sqrt{T_{\rm eff}}$ (Brown+1991; justified theoretically by Belkacem+2011).

**§6.2 Model-independent inversions**: Reese+2012 achieve 0.5% mean density via inversion. Buldgen+2016a,b reduce 16 Cyg A/B to 2% mass, 1% radius, 3% age.

**§6.3 Model-dependent grid fitting**: MESA/CESAM grids varying $(M, [Fe/H], Y, \alpha_{\rm MLT})$ fit to Δν, δν, ν_max, Teff, L yielding ~1% radii, ~4% masses, ~10–15% ages.

**§6.4 Ensemble asteroseismology**: Kepler's "LEGACY sample" (Lund+2017) of 66 stars gave a gold-standard grid. Chaplin+2011, 2014a characterized ~500 and ~1000 MS solar-type stars respectively.

**§6.5 C–D diagram**: $\langle\Delta\nu\rangle$ vs $\langle\delta\nu_{02}\rangle$ — tracks age (main-sequence evolution decreases δν₀₂ as core H is consumed).

**§6.6 Near-surface effects**: 1D models poorly handle the super-adiabatic layer, causing systematic frequency offsets. Corrections: Kjeldsen+2008 power-law, Ball & Gizon 2014 cubic+inverse, Sonoi+2015 Lorentzian.

**§6.7 Constraints on internal structure**: glitches (sharp features at BCE and He II zone) appear as oscillations in second frequency differences (Monteiro+1994, Verma+2017). Acoustic depths localize features (Fig. 28: KIC 6116048, BCE at $\tau\sim 3000$ s, He II at $\tau\sim 800$ s).

### Part VI: Stellar Rotation (§7) / 항성 회전

**§7.1 Photometric rotation**: starspot modulation of light curve → $P_{\rm rot}$. Methods: low-freq PSD peak (Barban+2009), autocorrelation function (McQuillan+2013), time–period wavelet (Mathur+2010b). Caution: harmonics can mimic true period (Fig. 30–31 KIC 4918333). ACF robust because harmonics alternate in height.

광곡선 회전 주기 추출은 세 방법 (PSD 피크, ACF, 웨이블릿)을 교차검증하는 것이 표준. Data calibration이 다른 파이프라인 간 비교 (García+2013a, 2014a)로 기기 잡음을 배제해야 한다.

**§7.2 Internal rotation from seismology**: HD 52265 (Gizon+2013 CoRoT) — first clean MS asteroseismic rotation, $\Omega/(2\pi)\approx 1\,\mu\mathrm{Hz}$, $\sin i\approx 0.6$, internally consistent with starspot and spectroscopy (Fig. 29). Kernels $K_{n,\ell}(r)$ in MS stars have most power in outer layers; dipole and quadrupole kernels integrated above $r\approx 0.15 R_\odot$ are nearly linear in radius (Fig. 33) → modes probe radiative+convective envelope with similar weighting.

**Two-zone model (Benomar+2015, Eqs. 58–61)**: separate $f_{\rm rad}$ and $f_{\rm conv}$. Assume $f_{\rm conv}\approx f_{\rm surf}$, then
$$\langle f_{\rm rad}\rangle=f_{\rm surf}+\frac{f_{\rm seis}-f_{\rm surf}}{\langle I_{\rm rad}\rangle}$$
Benomar+2015 found nearly uniform internal rotation in 22 CoRoT/Kepler targets (1.07–1.56 $M_\odot$) — **the Sun's flat radiative-zone rotation is typical**. Only KIC 9139163 showed marginal radial differential rotation.

### Part VII: Stellar Magnetic Activity (§8) / 자기 활동

**§8.1 From light curves**: $S_{\rm ph}$ proxy (Mathur+2014a,b) = rms of light curve on timescales $5\times P_{\rm rot}$. Defined for velocity too ($S_{\rm vel}$). Butterfly-like behavior: KIC 3733735 (Mathur+2014a, Fig. 37) rotation shifts from 3 d (minimum) to 2.54 d (maximum) — spots migrate in latitude, analog to solar differential rotation butterfly.

**§8.2 From asteroseismology**: p-mode **frequency shifts** track magnetic cycles. Sun: δν ≈ 0.4 μHz over 11-yr cycle, increasing with frequency (high-ν modes have outer turning points, so the perturbation is confined near the photosphere). First extra-solar detection: García+2010 on HD 49933 (CoRoT F-type) — shift ~2 μHz, 4× solar, anti-correlated with mode amplitude (Fig. 40, 41). Mode heights decrease and widths increase with activity; total energy is conserved (Pallé+1990, Chaplin+2000).

Salabert+2011b fitted individual ℓ=0, ℓ=1 modes of HD 49933 independently and confirmed the frequency dependence. NARVAL Mount-Wilson-S-index monitoring confirmed an ongoing cycle (Fig. 42).

**§8.2.1 Metallicity**: KIC 8006161 (Karoff+2018) — solar analog with $[Fe/H]\approx 0.3$ (2× solar). Larger surface differential rotation and larger frequency shifts than Sun → supports the chain "higher Z → larger opacities → deeper convection zone → stronger dynamo" (Brun+2017, Bessolaz & Brun 2011).

**§8.2.2 Frequency-shift vs $T_{\rm eff}$/age relations**: Salabert+2016a, 2018; Kiefer+2017; Santos+2018 — shift amplitude correlates with stellar activity level.

**§8.2.3 Variation with frequency**: frequency shift increases with ν due to mode-inertia weighting (Libbrecht & Woodard 1990; Goldreich+1991; Basu+2012).

**Böhm-Vitense (2007) A and I branches** (Fig. 39): activity cycle period vs. rotation period shows two branches, "Active" (shorter cycles, faster rotators) and "Inactive" (longer cycles, slower rotators). Sun sits near the intersection. KIC 8006161 fits I-branch. ι Hor, HD 49933, HD 76151 sit on A-branch.

---

## 3. Key Takeaways / 핵심 시사점

1. **Universal power-spectrum anatomy / PSD의 보편적 해부** — 모든 태양형 별의 PSD는 photon noise + Harvey granulation + activity slope + rotation peaks + Gaussian p-mode bump의 동일 구조를 보인다. 태양(Fig. 4 GOLF vs VIRGO)은 이 구조의 모든 성분을 교정할 수 있는 유일한 기준점이다. All solar-type PSDs share the same architecture (noise floor, granulation, activity, rotation, p-mode hump); the Sun is the unique reference where every component is independently calibrated.

2. **Scaling relations give 1–3% radii and 4–5% masses / 스케일링 관계의 정량적 위력** — $\Delta\nu_\odot=135.1\,\mu\mathrm{Hz}$, $\nu_{\max,\odot}=3090\,\mu\mathrm{Hz}$를 기준으로 (M, R)를 대수적으로 푼다. Kepler 1700개 별의 $\Delta\nu\propto\nu_{\max}^{0.75}$ 관계(Stello+2009a)가 이를 확인. 이는 외계행성 반지름 정밀도의 병목을 해소했다. Solar-calibrated Δν and ν_max give a closed-form algebraic inversion for $(M, R)$ that has revolutionized exoplanet host-star characterisation.

3. **p-modes are stochastically excited damped oscillators / 확률적으로 여기된 감쇠 진동자** — 각 모드는 2-dof $\chi^2$로 변동하는 Lorentzian 프로파일. 이로 인해 MLE와 Bayesian/MCMC가 단일 spectrum으로도 원리적으로 정당화된다. Each mode is a stochastically driven damped harmonic oscillator with a Lorentzian limit spectrum realised as a 2-dof $\chi^2$; this statistical foundation validates MLE/Bayesian fitting on a single Fourier realisation.

4. **Mixed modes are the gateway to cores / 혼합 모드는 내핵의 관문** — 주계열 태양형에선 검출 불가능하나 subgiant·RG에선 p/g coupling으로 core rotation과 진화 단계(RGB vs. clump)를 판정 (Bedding+2011). Mixed modes don't exist in MS solar-type stars, but once stars evolve they become the primary tool for core physics.

5. **Internal rotation in MS solar-type stars is near-uniform / 주계열 태양형 내부 회전은 거의 균일** — Benomar+2015 두-영역 모델로 분석한 22개 CoRoT/Kepler 별은 $f_{\rm rad}/f_{\rm surf}\approx 1$. 태양은 예외가 아니라 전형이다. 이는 강력한 각운동량 수송 메커니즘(magnetic, gravity-wave, or transport)을 요구한다. Solar-type main-sequence interior rotation is mostly uniform (Benomar+2015), demanding efficient angular momentum transport — the Sun is a typical, not exceptional, case.

6. **Magnetic cycles leave seismic fingerprints / 자기 주기의 성진학적 지문** — 태양에서 δν ≈ 0.4 μHz (11-yr). HD 49933은 4× 더 큰 δν ≈ 2 μHz. 이 신호는 frequency에 따라 증가하여 표면 근처 교란임을 보여준다. 모드 높이 감소, 폭 증가, energy supply는 보존. Asteroseismic frequency shifts trace magnetic cycles in other stars; HD 49933 shows cycles four times larger than the Sun's, with the same frequency dependence.

7. **The Böhm-Vitense two-branch diagram / 두-분기 활동 다이어그램** — activity-cycle period vs. rotation period에서 A(active)와 I(inactive) 두 분기. 태양은 경계 근처. Metallicity(KIC 8006161)가 분기에 영향을 줄 가능성. The Sun's position in the Böhm-Vitense $P_{\rm cyc}$–$P_{\rm rot}$ plane places it near the A/I boundary; metallicity plausibly modulates which branch a star follows.

8. **Near-surface and surface effects dominate systematic errors / 표면효과가 계통오차를 지배** — 1D stellar evolution 코드는 super-adiabatic surface layer를 부정확하게 모델링해 주파수 offset 생성 (Kjeldsen+2008; Ball & Gizon 2014). δν/Δν 비율(Roxburgh-Vorontsov 2003)이 이를 대폭 상쇄. The dominant systematic in mode frequencies is the near-surface layer that 1D models handle poorly; frequency-ratio diagnostics mitigate this.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 The eigenvalue problem

구면대칭 별의 선형 섭동: 각 eigenmode $\xi_{n,\ell,m}(r,\theta,\phi,t)=\xi_r(r)Y_\ell^m(\theta,\phi)e^{-i\omega t}$. Mode inertia (Eq. 2):
$$\mathcal{I}_{n,\ell,m}=\int_V \rho|\xi_{n,\ell,m}|^2 dV$$

Radial wave equation with $K(r)$ (Eq. 1):
$$\frac{d^2\xi_r}{dr^2}+\frac{\omega^2}{c^2}\left(\frac{N^2}{\omega^2}-1\right)\left(\frac{S_\ell^2}{\omega^2}-1\right)\xi_r=0$$
- $N^2=g\left(\frac{1}{\Gamma_1}\frac{d\ln p}{dr}-\frac{d\ln\rho}{dr}\right)$ (Brunt–Väisälä)
- $S_\ell=\sqrt{\ell(\ell+1)}\,c/r$ (Lamb)
- $c=\sqrt{\Gamma_1 p/\rho}$ (sound speed)

### 4.2 Asymptotic comb pattern

Tassoul (1980) first-order (Eq. 12):
$$\nu_{n,\ell}\simeq\Delta\nu\left(n+\frac{\ell}{2}+\frac{1}{4}+\varepsilon\right)$$
- $\Delta\nu=\left[2\int_0^R dr/c\right]^{-1}$ : large separation (acoustic radius inverse)
- $\varepsilon$: phase offset, ≈1.4 for the Sun

Implication: échelle ridges for ℓ and ℓ+2 fall near the same (ν mod Δν) column because $\ell/2$ shifts by integer.

Second-order (Tassoul 1990, Vorontsov 1991):
$$\delta\nu_{\ell,\ell+2}(n)\simeq -(4\ell+6)\frac{\Delta\nu_\ell(n)}{4\pi^2\nu_{n,\ell}}\int_0^R\frac{dc}{dr}\frac{dr}{r}$$
The $1/r$ weighting makes δν strongly sensitive to the inner core sound-speed gradient (i.e., composition via mean molecular weight).

### 4.3 Rotation

Ledoux splitting (Eq. 6):
$$\delta\omega_{n,\ell,m}=m(C_{n,\ell}-1)\Omega$$
For p-modes with $n\gtrsim 10$: $C_{n,\ell}\approx 0$, so
$$\nu_{n,\ell,m}=\nu_{n,\ell}-m\nu_s,\quad \nu_s=\Omega/(2\pi)$$
For g-modes: $C_{n,\ell}\approx 1/[\ell(\ell+1)]$.

Inclination-dependent amplitudes (Eq. 10):
$$r_{\ell,m}^2(i)=\frac{(\ell-|m|)!}{(\ell+|m|)!}[P_\ell^{|m|}(\cos i)]^2$$
For ℓ=1: $r_{1,0}^2=\cos^2 i$, $r_{1,\pm 1}^2=\sin^2 i/2$. Sum-rule $\sum_m r_{\ell,m}^2=1$.

### 4.4 Spectrum model

$$S(\nu)=B(\nu)+P(\nu)$$

Background (Harvey-like, Eq. 19):
$$B(\nu)=\sum_i\frac{\xi_i\sigma_i^2\tau_i}{1+(2\pi\nu\tau_i)^{\alpha_i}}+W$$
with $W$ photon noise. Typically 1–2 granulation components for MS stars.

p-mode sum (Eq. 28):
$$P(\nu)=\sum_{n,\ell}M_\ell(\nu;H_{n,\ell},\Gamma_{n,\ell},\nu_{n,\ell},\nu_s,i)$$
$$M_\ell(\nu)=\sum_{m=-\ell}^{m=\ell}a_{\ell,m}^2(i)\frac{H_{n,\ell}}{1+\left[\frac{2(\nu-\nu_{n,\ell}+m\nu_s)}{\Gamma_{n,\ell}}\right]^2}$$

Observed PSD realisation $Y(\nu_i)=S(\nu_i)\times X_i$ with $X_i\sim\chi^2_2/2$ IID.

### 4.5 Scaling relations (solar-calibrated)

$$\Delta\nu \approx \Delta\nu_\odot\left(\frac{M}{M_\odot}\right)^{1/2}\left(\frac{R}{R_\odot}\right)^{-3/2}$$

$$\nu_{\max}\approx\nu_{\max,\odot}\left(\frac{M}{M_\odot}\right)\left(\frac{R}{R_\odot}\right)^{-2}\left(\frac{T_{\rm eff}}{T_{\rm eff,\odot}}\right)^{-1/2}$$

Solving (inverse form):
$$\frac{R}{R_\odot}=\left(\frac{\nu_{\max}}{\nu_{\max,\odot}}\right)\left(\frac{\Delta\nu}{\Delta\nu_\odot}\right)^{-2}\left(\frac{T_{\rm eff}}{T_{\rm eff,\odot}}\right)^{1/2}$$

$$\frac{M}{M_\odot}=\left(\frac{\nu_{\max}}{\nu_{\max,\odot}}\right)^3\left(\frac{\Delta\nu}{\Delta\nu_\odot}\right)^{-4}\left(\frac{T_{\rm eff}}{T_{\rm eff,\odot}}\right)^{3/2}$$

**Worked example (the Sun)**: $\Delta\nu=135.1$, $\nu_{\max}=3090$, $T_{\rm eff}=5770$ → $R/R_\odot=1.00$, $M/M_\odot=1.00$ by construction.

**Worked example (16 Cyg A)**: $\Delta\nu\approx 103.4\,\mu\mathrm{Hz}$, $\nu_{\max}\approx 2200\,\mu\mathrm{Hz}$, $T_{\rm eff}\approx 5825$ K:
- $R/R_\odot=(2200/3090)\times(103.4/135.1)^{-2}\times(5825/5770)^{0.5}\approx 0.712\times 1.707\times 1.0048\approx 1.22$
- $M/M_\odot=(2200/3090)^3\times(103.4/135.1)^{-4}\times(5825/5770)^{1.5}\approx 0.361\times 2.915\times 1.014\approx 1.07$

Consistent with detailed-model values ($R=1.22 R_\odot$, $M=1.08 M_\odot$; Metcalfe+2012) to ~1%.

### 4.6 Frequency precision (Libbrecht 1992)

$$\sigma_\nu=\sqrt{f(\beta)\frac{\Gamma}{4\pi T}},\quad f(\beta)=(1+\beta)^{1/2}[(1+\beta)^{1/2}+\beta^{1/2}]^3$$
$T$=observation time, $\beta=B/H$. $\sigma_\nu\propto T^{-1/2}$: factor of 2 precision requires 4× longer observation.

### 4.7 Two-zone rotation (Benomar+2015)

$$\langle\delta\nu_{n,\ell}\rangle=I_{\rm rad}f_{\rm rad}+I_{\rm conv}f_{\rm conv},\quad I_{\rm rad}+I_{\rm conv}=1$$
$$I_{\rm rad}=\int_0^{r_{\rm bcz}}K_{n,\ell}(r)\,dr,\quad I_{\rm conv}=\int_{r_{\rm bcz}}^R K_{n,\ell}(r)\,dr$$

Assuming $f_{\rm conv}\approx f_{\rm surf}$:
$$\langle f_{\rm rad}\rangle=f_{\rm surf}+\frac{f_{\rm seis}-f_{\rm surf}}{\langle I_{\rm rad}\rangle}$$

---

## 5. Paper in the Arc of History / 역사 속의 논문 (ASCII timeline)

```
1962  Leighton, Noyes, Simon — 태양 5분 진동 발견
      discovery of solar 5-min oscillation
 │
1975  Deubner — k–ω diagram, p-mode 확립
 │
1980  Tassoul — p-mode asymptotic theory (Δν, δν) ─────┐
 │                                                     │
1985  Christensen-Dalsgaard+ — p-mode inversion (CZ)  │ (theoretical scaffold)
 │                                                     │
1990  Brown+ — Procyon ground-based attempts          │
 │                                                     │
1995  SoHO/GOLF/MDI/VIRGO launch                      │
1995  Kjeldsen & Bedding — scaling relations          │
1995  Kjeldsen — η Boo (first MS star p-mode)         │
 │                                                     │
2000  α Cen A (WIRE + Doppler)                        │
2006  CoRoT launch ───────┬─► new epoch               │
2009  Kepler launch ──────┤                           │
2010  García+ — HD 49933 magnetic cycle detected      │
2011  Chaplin+ — Kepler 500-star ensemble             │
2013  Gizon+ — HD 52265 internal rotation             │
2015  Benomar+ — 22 stars nearly uniform rotation     │
2017  Lund+ — LEGACY sample 66 stars                  │
2018  TESS launch                                     │
2019  ★ García & Ballot — THIS REVIEW ★  ◄────────────┘ (synthesis)
 │
2026  PLATO launch (expected)
```

### 관련 Living Reviews 및 참고 논문 / Related papers in this project:

- Paper #5: Gizon & Birch (2005) — Local helioseismology (prerequisite for methodology)
- Paper #49: Basu (2016) — Global helioseismology (prerequisite for inversion theory and methods)
- Paper #30: Solar differential rotation (context for internal rotation inferences)
- Paper #52: Solar cycle (context for magnetic-activity seismic signatures)

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **Paper #5 Gizon & Birch (2005) — Local Helioseismology** | Provides the sibling-methodology review for the Sun; asteroseismology's tools descend from global+local helioseismology as applied to disk-integrated light | Very high — defines the method heritage |
| **Paper #49 Basu (2016) — Global Helioseismology** | Inversion theory for sound-speed and rotation profiles that inspired stellar-kernel methods (e.g., Benomar+2015 two-zone model) | Very high — identical eigenvalue-problem framework |
| **Paper #30 Solar Differential Rotation** | Context for interpreting stellar internal-rotation profiles and surface differential rotation (§7) | High — comparison base for asteroseismic $f_{\rm rad}/f_{\rm surf}$ |
| **Paper #52 Solar Cycle / Magnetic Activity** | Underpins interpretation of stellar frequency shifts, $S_{\rm ph}$ proxies, and Böhm-Vitense diagram (§8) | High — provides the Sun's benchmark 11-yr signal |
| **Tassoul (1980) — Asymptotic Theory** | Source of $\nu_{n,\ell}\approx\Delta\nu(n+\ell/2+1/4+\varepsilon)$ and δν expression (Eqs. 12, 15) | Foundational for §4.4 |
| **Kjeldsen & Bedding (1995) — Scaling Relations** | Eqs. 43, 44 originate here; establish solar-calibrated mass/radius inference | Foundational for §6.1 |
| **Aerts, Christensen-Dalsgaard & Kurtz (2010) textbook** | Comprehensive theoretical treatment that complements this review's methodological focus | Complementary reference text |
| **Chaplin & Miglio (2013) — Asteroseismology of Solar-Type and Red-Giant Stars (ARA&A)** | Earlier companion review focused on red giants; this paper is the solar-type-star counterpart | Complementary scope |
| **Lund+ (2017) — LEGACY Kepler Sample** | Gold-standard 66-star dataset underlying §6.4 ensemble analysis | Data benchmark |
| **Belkacem+ (2011, 2013)** | Theoretical justification of ν_max and Δν scaling (acoustic cutoff, homology) | §6.1 physical basis |

---

## 7. References / 참고문헌

- García, R. A. & Ballot, J., "Asteroseismology of solar-type stars", *Living Reviews in Solar Physics*, 16:4 (2019). DOI: 10.1007/s41116-019-0020-1

### Supporting references cited / 논문 내 주요 참고문헌 (selection)

- Aerts, C., Christensen-Dalsgaard, J., & Kurtz, D. W., *Asteroseismology*, Springer (2010).
- Appourchaux, T., Gizon, L., & Rabello-Soares, M.-C., "Peak-bagging of the low-degree solar modes", A&AS 132, 107 (1998).
- Basu, S., "Global seismology of the Sun", *Living Reviews in Solar Physics*, 13:2 (2016). [Project paper #49]
- Bedding, T. R., Mosser, B., Huber, D., et al., "Gravity modes as a way to distinguish between hydrogen- and helium-burning red giant stars", Nature 471, 608 (2011).
- Belkacem, K., Goupil, M. J., Dupret, M. A., et al., "The underlying physical meaning of the ν_max–ν_c relation", A&A 530, A142 (2011).
- Benomar, O., Takata, M., Shibahashi, H., Ceillier, T., & García, R. A., "Nearly uniform internal rotation of solar-like main-sequence stars revealed by space-based asteroseismology and spectroscopic measurements", MNRAS 452, 2654 (2015).
- Brown, T. M., Gilliland, R. L., Noyes, R. W., & Ramsey, L. W., "Detection of possible p-mode oscillations on Procyon", ApJ 368, 599 (1991).
- Buldgen, G., Reese, D. R., & Dupret, M. A., "Constraints on the structure of 16 Cyg A and 16 Cyg B using inversion techniques", A&A 585, A109 (2016a).
- Chaplin, W. J., Kjeldsen, H., Christensen-Dalsgaard, J., et al., "Ensemble asteroseismology of solar-type stars with the NASA Kepler mission", Science 332, 213 (2011).
- Chaplin, W. J. & Miglio, A., "Asteroseismology of solar-type and red-giant stars", ARA&A 51, 353 (2013).
- García, R. A., Mathur, S., Salabert, D., et al., "CoRoT reveals a magnetic activity cycle in a Sun-like star", Science 329, 1032 (2010).
- Gizon, L. & Birch, A. C., "Local helioseismology", *Living Reviews in Solar Physics*, 2:6 (2005). [Project paper #5]
- Gizon, L., Ballot, J., Michel, E., et al., "Seismic constraints on rotation of Sun-like star and mass of exoplanet", PNAS 110, 13267 (2013).
- Huber, D., Bedding, T. R., Stello, D., et al., "Testing scaling relations for solar-like oscillations from the main sequence to red giants using Kepler data", ApJ 743, 143 (2011).
- Kjeldsen, H. & Bedding, T. R., "Amplitudes of stellar oscillations: the implications for asteroseismology", A&A 293, 87 (1995).
- Kjeldsen, H., Bedding, T. R., & Christensen-Dalsgaard, J., "Correcting stellar oscillation frequencies for near-surface effects", ApJ 683, L175 (2008).
- Ledoux, P., "The nonradial oscillations of gaseous stars and the problem of Beta Canis Majoris", ApJ 114, 373 (1951).
- Libbrecht, K. G., "On the ultimate accuracy of solar oscillation frequency measurements", ApJ 387, 712 (1992).
- Lund, M. N., Silva Aguirre, V., Davies, G. R., et al., "Standing on the shoulders of Dwarfs: the Kepler asteroseismic LEGACY sample. I. Oscillation mode parameters", ApJ 835, 172 (2017).
- Mathur, S., García, R. A., Ballot, J., et al., "Magnetic activity of F stars observed by Kepler", A&A 562, A124 (2014a).
- Metcalfe, T. S., Chaplin, W. J., Appourchaux, T., et al., "Asteroseismology of the solar analogs 16 Cyg A and B from Kepler observations", ApJ 748, L10 (2012).
- Roxburgh, I. W. & Vorontsov, S. V., "The ratio of small to large separations of acoustic oscillations as a diagnostic of the interior of solar-like stars", A&A 411, 215 (2003).
- Salabert, D., Régulo, C., Ballot, J., García, R. A., & Mathur, S., "About the p-mode frequency shifts in HD 49933", A&A 530, A127 (2011b).
- Stello, D., Chaplin, W. J., Basu, S., Elsworth, Y., & Bedding, T. R., "The relation between Δν and ν_max for solar-like oscillations", MNRAS 400, L80 (2009a).
- Tassoul, M., "Asymptotic approximations for stellar nonradial pulsations", ApJS 43, 469 (1980).
