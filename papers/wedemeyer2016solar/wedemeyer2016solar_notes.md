---
title: "Solar Science with the Atacama Large Millimeter/Submillimeter Array — A New View of Our Sun"
authors: [Wedemeyer, S., Bastian, T., Brajša, R., Hudson, H., Fleishman, G., Loukitcheva, M., et al.]
year: 2016
journal: "Space Science Reviews"
doi: "10.1007/s11214-015-0229-9"
topic: Solar_Observation
tags: [ALMA, chromosphere, millimeter, radio-astronomy, free-free, interferometry, solar-physics]
status: completed
date_started: 2026-04-23
date_completed: 2026-04-23
---

# 27. Solar Science with ALMA — A New View of Our Sun / ALMA로 본 태양 — 태양의 새로운 시각

---

## 1. Core Contribution / 핵심 기여

**English**: This landmark review by Wedemeyer, Bastian, Brajša and 36 co-authors of the SSALMON consortium establishes the scientific foundation for solar observations with the Atacama Large Millimeter/submillimeter Array (ALMA). The authors systematically treat (i) the physics of millimeter/submillimeter radiation formation in the solar atmosphere — dominated by thermal free-free (electron-ion and H⁻) emission with minor contributions from recombination lines, molecular CO rotational lines, gyroresonance, and gyrosynchrotron radiation; (ii) ALMA's technical capabilities — 66 antennas (fifty 12-m plus twelve 7-m plus four 12-m Total Power), baselines up to 16 km, 10 receiver bands spanning 35–950 GHz (wavelengths 8.6–0.3 mm), primary beam ≈ 19″×(λ/1 mm), with sub-arcsecond synthesized resolution; and (iii) a comprehensive catalog of solar science cases including quiet-Sun thermal structure, sunspot umbrae, flares including the enigmatic sub-THz component, prominences, coronal rain, chromospheric heating, wave propagation, and chromospheric magnetometry via free-free polarization.

The paper's central message is that ALMA acts as a nearly linear thermometer of the solar chromosphere because in the Rayleigh-Jeans regime the observed brightness temperature T_b is directly proportional to the local gas temperature in the continuum-forming layer. Combined with its high cadence (< 1 s for snapshot imaging), ALMA's multi-band capability lets observers sample different chromospheric heights (~500–2000 km above the photosphere) simply by changing the frequency. This is a paradigm shift after decades of ~10″ single-dish or small-array limitations (BIMA, Nobeyama, OVSA). The paper also prescribes how to use 3D MHD simulations (Bifrost, CO⁵BOLD) forward-modeled with radiative transfer codes (LINFOR3D, PHOENIX/3D) to interpret ALMA maps correctly.

**한국어**: Wedemeyer, Bastian, Brajša 및 SSALMON 컨소시엄의 36명 공저자가 작성한 이 획기적 리뷰는 Atacama Large Millimeter/submillimeter Array (ALMA)를 이용한 태양 관측의 과학적 기반을 정립한다. 저자들은 (i) 태양 대기에서 mm/sub-mm 복사의 형성 물리 — 열적 free-free (전자-이온 및 H⁻) 방출이 주를 이루고, 재결합선, CO 회전 분자선, gyroresonance, gyrosynchrotron이 부차적으로 기여함, (ii) ALMA의 기술 사양 — 66개 안테나(12 m × 50 + 7 m × 12 + 12 m TP × 4), 기저선 최대 16 km, 10개 수신기 대역 35–950 GHz (파장 8.6–0.3 mm), 주빔 ≈ 19″×(λ/1 mm), 서브초 합성 분해능, (iii) 조용한 태양의 열 구조, 흑점 암부, 플레어(미스터리한 sub-THz 성분 포함), 홍염, 코로나 rain, 채층 가열, 파동 전파, free-free 편광을 이용한 채층 자기장 측정 등 포괄적 과학 응용을 체계적으로 다룬다.

이 논문의 핵심 메시지는 Rayleigh-Jeans 영역에서 관측 휘도 온도 T_b가 연속체 형성층의 국소 기체 온도에 직접 비례하기 때문에 ALMA가 태양 채층의 '거의 선형 온도계' 역할을 한다는 것이다. 1초 미만의 snapshot cadence와 다파장 관측 능력을 결합하면, 주파수만 바꿈으로써 채층의 서로 다른 높이(광구 위 약 500–2000 km)를 샘플링할 수 있다. 이는 수십 년간 10″ 수준의 단일 접시 또는 소규모 배열(BIMA, Nobeyama, OVSA)의 한계를 극복한 패러다임 전환이다. 논문은 또한 3D MHD 시뮬레이션(Bifrost, CO⁵BOLD)을 복사전달 코드(LINFOR3D, PHOENIX/3D)로 전방 모델링하여 ALMA 맵을 올바르게 해석하는 방법론을 제시한다.

본 논문은 2016년에 출판되었지만, 이후 수년간 태양 ALMA 관측 커뮤니티의 표준 참고 문헌이 되었으며, SSALMON 네트워크(http://ssalmon.uio.no)를 통해 시뮬레이션-관측 연결을 이끌었다. 2016년 말 ALMA Cycle 4부터 정기 태양 관측이 시작되어 Shimojo et al. (2017), White et al. (2017), Bastian et al. (2017)의 후속 논문들이 이 리뷰의 기대를 확인하거나 갱신했다.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction and Motivation (§1) / 서론과 동기 (§1)

**English**: The chromosphere — a complex, dynamic, partially ionized layer between the photosphere and corona — plays a pivotal role in energy and matter transport and ultimately in coronal heating. Advances in visible/IR instrumentation (CRISP at SST, Hinode, IRIS) improved chromospheric diagnostics in Ca II 854.2 nm, Hα, Mg II h & k, but the interpretation of these optically-thick, non-LTE lines remains complicated (Fig. 1 shows representative active region, sunspot, coronal hole, and network scenes). Radiation at mm wavelengths has long had great diagnostic potential but was hampered by (a) small telescope apertures yielding poor spatial resolution for single dishes and (b) small interferometric arrays (BIMA, 10″ beams; Fig. 2 shows representative BIMA 3.5 mm images from Loukitcheva et al. 2009, 2014). ALMA, operating from the Chajnantor plateau at 5000 m altitude, brings an enormous leap: it achieves spatial resolution close to the visible/IR diffraction limits of 1-m class solar telescopes, with high temporal resolution and broad band-switching capability.

**한국어**: 광구와 코로나 사이에 위치한 채층은 복잡하고 역동적이며 부분적으로 이온화된 층으로, 에너지/물질 수송과 궁극적으로 코로나 가열에 핵심적 역할을 한다. 가시/IR 계측기(SST의 CRISP, Hinode, IRIS)의 발전으로 Ca II 854.2 nm, Hα, Mg II h&k에서 채층 진단이 개선되었으나, 이들 optically-thick, non-LTE 선의 해석은 여전히 복잡하다 (Fig. 1: 활동영역, 흑점, 코로나홀, 네트워크). mm 파장 복사는 오랫동안 큰 진단 잠재력이 있었으나 (a) 단일접시의 낮은 공간 분해능과 (b) 소규모 간섭계(BIMA 10″)에 의해 제한받았다 (Fig. 2). ALMA는 Chajnantor 5000 m 고원에서 운영되어, 1 m 급 태양 망원경의 회절 한계에 근접한 공간 분해능과 높은 시간 분해능 및 광대역 전환 능력을 제공하여 이 한계를 도약적으로 극복한다.

### Part II: Formation of Millimeter Radiation (§2) / mm 복사의 형성 (§2)

**English**: The dominant opacity sources at mm/sub-mm wavelengths (in quiet-Sun absence of strong B) are: (1) electron-ion free-free absorption (H⁺ + e⁻ + γ → H⁺ + e⁻*, i.e., inverse bremsstrahlung), and (2) H⁻ free-free (important at cooler, photospheric heights). Following Dulk (1985) and after Karzas & Latter (1961):

$$\chi_{\text{ions,ff}} \approx \frac{1}{3c}\left(\frac{2}{\pi}\right)^{1/2} \frac{\nu_p^2}{\nu^2} \frac{4\pi e^4}{\sqrt{m_e}(k_BT)^{3/2}} \sum_i Z_i^2 n_i \frac{\pi}{\sqrt{3}} g_{\text{ions,ff}}(T,\nu)$$

Numerically (cgs):
$$\chi_{\text{ions,ff}} \approx 9.78 \times 10^{-3}\, \frac{n_e}{\nu^2 T^{3/2}} \sum_i Z_i^2 n_i\, (17.9 + \ln T^{3/2} - \ln \nu)\quad [\text{cm}^{-1}]$$

Simplified (Kundu 1965): χ = ξ(T,e)·n_e²/(n·ν²·T^{3/2}), with ξ ≈ 0.12 (chromosphere) and ≈ 0.2 (corona). Since opacity is governed by particle interactions, LTE holds and the source function is Planckian. In the RJ limit (valid for T ≳ 4500 K at mm wavelengths):

$$B_\nu(T) = \frac{2h\nu^3}{c^2}\left(e^{h\nu/k_B T}-1\right)^{-1} \approx \frac{2 k_B T}{c^2}\nu^2$$

Emergent specific intensity:
$$I_\nu = \int_0^{\tau_{\max}(\nu)} B_\nu(T(\bar\tau))\, e^{-\bar\tau}\, d\bar\tau$$

Brightness temperature:
$$T_b = \frac{c^2}{2 k_B \nu^2} I_\nu$$

**Fig. 4 interpretation**: At λ = 1 mm, electron-ion free-free dominates across T = 1 kK–100 kK, normalized to the Thomson cross-section σ = 6.65×10⁻²⁹ m². H⁻ contributes negligibly above the temperature minimum.

**Fig. 5 interpretation** — the contribution functions at 0.3, 1.0, 3.0, 9.0 mm peak at average heights of 490, 730, 960, 1170 km (Wedemeyer-Böhm 2007), or 700, 1000, 1550, 1700, 2000 km (Loukitcheva 2015 enhanced network), with FWHMs 220, 360, 440, 480 km. **The longer the wavelength, the higher the formation layer.** Spatial inhomogeneities give standard deviations of z_m ~ 15% of the mean.

**한국어**: mm/sub-mm 파장에서의 주요 불투명도 원인은 (1) 전자-이온 free-free 흡수(역제동복사), (2) H⁻ free-free (광구 근처에서 중요)이다. Dulk (1985) 및 Karzas & Latter (1961)에 따라 χ_{ions,ff} ∝ n_e²·Σ Z_i²n_i / (ν²·T^{3/2})·g_ff 형태를 가진다. 수치적으로 χ ≈ 0.12·n_e²/(ν²T^{3/2}) (채층). 입자 상호작용이 지배하므로 LTE가 성립하고 원천 함수는 Planck식이다. RJ 극한(mm 파장 T ≳ 4500 K에서 유효)에서 B_ν ≈ 2k_B T ν²/c², 따라서 T_b = c²I_ν/(2k_B ν²).

**Fig. 4**: λ = 1 mm에서 전자-이온 free-free가 T = 1 kK–100 kK에 걸쳐 H⁻를 압도.

**Fig. 5**: 기여함수는 0.3, 1.0, 3.0, 9.0 mm에서 평균 형성 높이가 490, 730, 960, 1170 km (Wedemeyer-Böhm 2007)이며, FWHM 220, 360, 440, 480 km. **파장이 길수록 형성층이 높다.**

#### Recombination Lines (§2.2)

A Rydberg transition between levels (n+Δn) → n of an ion of charge Z, mass M has frequency:
$$\nu \approx 2 c R_M Z^2 \frac{\Delta n}{n^3}, \quad R_M = \frac{R_\infty}{1 + (1822.88848\, M - Z)^{-1}}$$

JCMT observations detected H I n = 20→19 at 888.047 GHz and 22→21 at 662.404 GHz (Clark 2000a,b) — ALMA will dramatically improve sensitivity, spectral resolution, and line-profile fidelity.

#### Molecular CO Lines (§2.3)

CO rotational transitions J = 6→5 at 691 GHz and J = 7→6 at 806 GHz should produce depressions of ~20 K at 3500 K. The CO "COmosphere" (Ayres et al.) — cold pockets at ~1000 K in the upper photosphere / lower chromosphere — can be directly probed. PHOENIX/3D synthetic spectra for ALMA Band 6 (Fig. 6) show a ~±400 K range across the simulation with line depths of ~100 K average, up to 250 K.

#### Gyroresonance and Gyrosynchrotron (§2.4)

Gyroresonance (harmonics s = 2–5 of ν_B = 2.8×10⁶ Hz × B[G]) falls below 30 GHz even for 3570 G sunspots, outside ALMA. However, non-thermal **gyrosynchrotron** from relativistic electrons (γ² times larger than ν_B) does fall within ALMA bands, particularly during flares. The Razin effect ν_peak ≈ 20 n_e/B sets the spectral peak. Spectral indices: α_lf = 2–3 (optically thick), α_hf (optically thin) from the electron power-law index.

**Figure 6 interpretation**: PHOENIX/3D synthetic spectra for ALMA Band 6 (1.1–1.4 mm). Grey area shows brightness temperature range across 33×33 simulation columns. Black line: average spectrum. Dashed lines: ±1σ. The continuum varies significantly (±400 K, even ±600 K for extreme columns), but individual spectral lines are shallow (~100 K, max 250 K) due to pressure broadening. This prediction motivates ALMA spectral line searches.

#### Polarization and Magnetic Fields (§2.5)

The slope of the free-free brightness spectrum:
$$\zeta = \frac{d\log T_b(\nu)}{d\log \nu}$$

gives the temperature gradient in the formation layer and controls circular polarization:
$$\mathcal{P} = \zeta \frac{\nu_B}{\nu} \cos\theta$$

For an optically thin, isothermal slab ζ = 2, giving Kundu's classical result:
$$\mathcal{P} \simeq 2 \frac{\nu_B}{\nu} \cos\theta$$

Numerically: 𝒫 (%) ≈ 1.85×10⁻³ × λ(mm) × B(G) × cos θ. For 1 kG sunspot at 1 mm, 𝒫 ≈ 2%. The longitudinal field from a single-frequency measurement:
$$B_l = |\mathbf{B}|\cos\theta = \frac{\mathcal{P}\nu}{\zeta}(2.8\times 10^6)^{-1}\ \text{Hz}^{-1}\ (\text{G})$$

By observing at multiple frequencies, ALMA reconstructs B_l as a function of height — revolutionary chromospheric magnetometry.

### Part III: ALMA Technical Capabilities (§3) / ALMA 기술 사양 (§3)

**English**: Section 3.1 surveys pre-ALMA mm/cm facilities (WSRT, VLA, Nobeyama NoRH, SSRT, Nançay, RATAN-600, OVSA, BIMA). Section 3.2 details ALMA itself:
- **12-m Array**: 50 movable antennas, baselines up to 16 km
- **ACA**: twelve 7-m antennas (baselines 8.5–33 m) plus four 12-m TP
- **10 receiver bands** from 35 to 950 GHz (λ 8.6 to 0.3 mm)
- Two polarization channels, 8 GHz bandwidth per polarization, divided into 2 GHz basebands
- FDM mode: up to 32 sub-bands, 62.5 MHz each, max 3840 channels, min channel spacing 3.8 kHz → velocity resolution 0.02 km/s at 110 GHz
- Primary beam θ ≈ 1.13 λ/D ≈ 19″ × (λ/1 mm) for D = 12 m
- Maximum recoverable size ϑ_max = 0.6λ/L_min = 37100/(L_min·ν) arcsec; for ACA 7-m (L_min = 8.4 m, NS): ϑ_max ≈ 44″ at 100 GHz
- Full-disk scan in 8 minutes with 12-m TP (see Fig. 3 for a 230 GHz map)
- Solar-specific measures: roughened antenna surfaces, solar filters or detector detuning to avoid saturation, attenuated sensitivity (T_sys 100–1000 K)

**한국어**: §3.1은 pre-ALMA 시설 개괄. §3.2는 ALMA 자체:
- **12-m 배열**: 이동가능 안테나 50대, 기저선 최대 16 km
- **ACA**: 7-m × 12 + TP 12-m × 4
- **수신기 10개 대역**: 35–950 GHz (λ 8.6–0.3 mm)
- 편광 2채널, 대역폭 8 GHz, 2 GHz baseband × 4
- FDM 모드: 최대 3840 채널, 분해능 3.8 kHz (110 GHz에서 0.02 km/s)
- 주빔 θ ≈ 19″ × (λ/1 mm) [D = 12 m]
- 최대 회수 각도 ϑ_max = 37100/(L_min·ν) arcsec; 7-m 배열 8.4 m 최단 기저선에서 100 GHz에 44″
- 12-m TP로 전체 디스크 스캔 8분
- 태양 특화 대책: 표면 거칠기 처리, 태양 필터/디튜닝, T_sys 100–1000 K

**Section 3.3 diagnostic potential**: Interferometric imaging of the Sun is challenging because the chromosphere fills the FOV with emission on all spatial scales, contrasts of up to 1000 K against a 4000–7000 K background. ALMA's three critical features: (i) many dishes (62 for interferometry), (ii) ACA 7-m for short spacings, (iii) 12-m TP for zero-spacing and fast full-disk scans. Spatial resolution ≈ 0.3″–0.4″ × (λ/1 mm); at 1 mm and below, this matches today's optical solar telescopes.

**§3.3.2 High cadence**: For interferometric snapshot imaging, sensitivity is not a problem because the Sun is very bright. The u-v coverage required for the target structure determines cadence. Integrations of minutes to 40 min are feasible for slowly evolving features, but phenomena like 3-min chromospheric oscillations or QPP (<1 s) require snapshot mode. ALMA can achieve < 1 s cadence for bright compact sources.

**§3.3.3 Spectral capabilities**: With up to 3840 channels at 3.8 kHz spacing, ALMA records ionic recombination lines (H I, C III, O IV, O V, O VI, Ne VII, Si XI, Fe XV) and molecular rotational lines (CO in particular). Some recombination lines originate in the corona, providing coronal diagnostics alongside the chromospheric continuum. CO is the most promising molecule because of its extensive rotational spectrum in the ALMA range. / ALMA는 3.8 kHz 채널 간격으로 재결합선과 CO 회전선을 동시 관측 가능.

**§3.3.4 Polarization & chromospheric B**: The free-free polarization method measures the longitudinal component B_l via Eq. 17. The height at which B is determined corresponds to the formation height of the observed frequency, so multi-frequency observations give B_l(z). Technical requirement: 0.1% polarization accuracy after calibration — achievable on-axis and feasible at beam centers (FWHM/3 for 0.3% off-axis). For sunspots with kG fields, few % polarization is easy; for quiet Sun, 0.5% signal needs careful calibration. / 다주파 관측으로 B_l(z) 도출; 흑점은 쉬움, 조용한 태양은 정밀 보정 필요.

**§3.3.5 Coordinated observations**: ALMA alone is powerful, but ALMA + IRIS + SDO + DKIST + EST + Solar-C coordination is essential for connecting chromospheric diagnostics across UV, visible, IR, and radio. Fig. 8 shows a simulated coordinated dataset: ALMA 1 mm + 3 mm + IRIS Mg II k core/wing + photospheric UV continuum + magnetogram, all on the same physical structure.

### Part IV: Science Cases (§4) / 과학 응용 (§4)

**English**:
- **§4.1 Grand Challenges**: Coronal/chromospheric heating; solar flares; prominences; space weather. ALMA provides 3D thermal structure and dynamics.
- **§4.2 Quiet Sun**: Thermal structure & dynamics (§4.2.1): Carlsson & Stein (1994) propagating-shock simulation reproduces Ca II K "K-grains"; the dynamic coexistence of hot shocks and cold post-shock CO pockets is a key ALMA test. Table 1 lists models A (Bifrost, L15), B–E (CO⁵BOLD, W07/W12/W13), with synthetic 1 mm T_b: means 3936–5087 K, std 822–1085 K, max up to 11094 K for Bifrost enhanced network.
- **§4.2.2 Numerical predictions**: Fig. 10 shows four model maps and T_b histograms. Non-equilibrium H ionization (model D) narrows the formation height distribution, decreasing mean T_b from 4476 K (LTE) to 3936 K (NLTE) and std from 998 K to 933 K. Spatial resolution critically affects T_b histograms (Fig. 11): contrast drops from 24% (native) to ~10% at 1″ resolution, following δT_b,rms/⟨T_b⟩ ∝ exp(−Δα/D).
- **§4.2.3 Magnetic fields**: Fig. 12 shows photospheric B_z vs. reconstructed 3-mm map at 1500 km height. Simulated circular polarization at 3 mm is ±0.5%, giving B_l within ±100 G. For quiet Sun this exceeds ALMA's 0.1% polarization requirement.
- **§4.2.4 Vortex flows / chromospheric swirls**: SST-observed swirls (12.7 ± 4.0 min lifetime, 4 ± 1.4″ diameter). ALMA provides direct thermal diagnostics.
- **§4.2.6 Center-to-limb & spicules**: Less limb brightening than expected — ALMA will clarify absorbing spicule contribution.
- **§4.3 Active regions & sunspots**: Fig. 13 shows simulated AR 12158 in Bands 3 (114.8 GHz) and 6 (238.6 GHz): umbra appears bright (hot) at low frequency, dark (cool) at high frequency — reveals temperature stratification.
- **§4.3.3 Umbral oscillations**: Magnetoacoustic gravity waves (3 min); ALMA snapshot cadence perfect.
- **§4.4 Flares**: Microwave flare emission is gyrosynchrotron (peaks 5–20 GHz), but the **sub-THz component** (Fig. 16: 212 GHz + 405 GHz SST excess, Kaufmann et al. 2009b) remains mysterious. Candidate mechanisms: synchrotron from positrons, diffusive radiation, Cherenkov. ALMA imaging spectroscopy across the full sub-THz range will be decisive. Positron signatures: polarization reversal at high frequency (Fig. 17).
- **§4.4.2 Micro/nanoflares**: ALMA's sensitivity will access ×10⁻⁹ M-class energies.
- **§4.4.4 QPP (quasi-periodic pulsations)**: Periods from fractions of a second to minutes; ALMA resolves spatial transverse structure.
- **§4.5–§4.6 Chromospheric heating & waves**: Acoustic, Alfvén, magneto-acoustic shocks, resonant absorption, Kelvin-Helmholtz instabilities in flux tubes.
- **§4.7 Coronal loops & coronal rain**: Strand widths ~100 km — ALMA thermometer of cool ~2000 K coronal rain in active region loops.
- **§4.8 Prominences and filaments**: ALMA determines T of cool prominence plasma, essential for energy balance. Multi-band tomography of the PCTR (prominence-corona transition region).

**§4.2.5 Ion-neutral collisions**: In the middle/upper chromosphere, ion-neutral collision times (~10⁻⁵–10⁻⁷ s in lower, up to ~1 s in upper) approach ALMA observational timescales. The validity of single-fluid MHD must be tested by comparing simulations with ALMA. This motivates multi-fluid MHD for Alfvén wave propagation.

**§4.2.6 Center-to-limb variation**: Observations in mm/cm show less limb brightening than predicted from the T(z) rise with height. The discrepancy is typically attributed to absorbing spicules (cooler than transition region). ALMA with TP system measures network and inter-network brightness simultaneously, providing diagnostic. Spicule lifetimes (~15 min for normal, ~2 min for type II) challenge u-v coverage.

**§4.2.7 Polar brightenings**: NoRH 17 GHz observations show polar regions brighter than rest of quiet Sun (diffuse ~1500 K + compact ~3500 K excess). Nature of compact sources unknown — no UV counterparts. Long-term variation correlates with solar cycle polar magnetic field. ALMA with high resolution can elucidate the compact sources.

**한국어**: (주요 내용 요약)
- **§4.2 조용한 태양**: Carlsson & Stein (1994) 전파 충격파 시뮬레이션으로 Ca II K-grains 재현; 뜨거운 충격파와 충격 후 차가운 CO 포켓의 공존 테스트. Table 1: 모델 A–E, 1 mm 평균 T_b = 3936–5087 K.
- **§4.2.2**: Non-equilibrium 수소 이온화 (모델 D): 평균 T_b 4476 K → 3936 K (LTE→NLTE). Fig. 11: 공간 분해능 저하 시 대비 24% → 10%.
- **§4.2.3 자기장**: Fig. 12 — 3 mm 맵에서 B_l ± 100 G 복원.
- **§4.3 활동 영역/흑점**: Fig. 13 — 흑점 암부가 저주파(Band 3)에서 밝음/고주파(Band 6)에서 어두움 → 온도 구조 드러냄.
- **§4.4 플레어**: sub-THz 성분 (Fig. 16)의 정체 규명이 핵심.
- **§4.5 가열**: 음향파, Alfvén 파, 자기음향 충격, Kelvin-Helmholtz 불안정성.
- **§4.7 코로나 loop 및 rain**: ~100 km strand; ALMA로 2000 K 코로나 비 진단.
- **§4.8 홍염**: PCTR 다파장 단층촬영.

### Part V: Solar Flare sub-THz Mystery (§4.4) in Depth / sub-THz 플레어 미스터리 심층

**English**: Section 4.4 deserves special attention. Microwave flare emission is normally interpreted as gyrosynchrotron from mildly relativistic electrons, peaking at 5–20 GHz (Guidice & Castelli 1975) and decreasing toward higher frequencies. However, Kaufmann et al. (2001, 2004, 2009a) discovered for the 6 Dec 2006 flare (Fig. 16) a second, increasing sub-THz spectral component at 212 and 405 GHz with peak flux ~5×10⁴ sfu at 3 mm (100 GHz). The spectrum rises with frequency — an entirely new flare spectral behavior. Multiple candidate mechanisms:
- **Synchrotron from relativistic positrons** (Fleishman & Kontar 2010): positrons from nuclear reactions can produce the rising spectrum; polarization reverses at the transition frequency (Fig. 17, red arrow).
- **Diffusive radiation** (Fleishman): bremsstrahlung-like emission from scattering off density fluctuations.
- **Cherenkov emission**: requires further investigation.
- **Thermal free-free from evaporated dense plasma**: flat continuum consistent with optically thin RJ emission.

The positron signature is distinctive: positrons preferentially emit ordinary-mode (o-mode) radiation, while electrons favor extraordinary-mode (x-mode). The polarity reversal in circular polarization (Fig. 17 inset) around 10–20 MeV is the "smoking gun". ALMA's broadband imaging spectroscopy across 35–950 GHz combined with SDO/HMI vector magnetograms should definitively distinguish these mechanisms and quantify the relativistic positron component — currently "almost unexplored diagnostics". After thermalization, positrons form positronium, detectable via the 203 GHz 1s–2s transition (Ellis & Bland-Hawthorn 2009).

**한국어**: §4.4는 특별한 주목을 요한다. 마이크로파 플레어 방출은 보통 경상대론적 전자의 gyrosynchrotron으로 5–20 GHz 피크(Guidice & Castelli 1975)를 가지며 고주파로 감소한다. 그러나 Kaufmann et al. (2001, 2004, 2009a)은 2006년 12월 6일 플레어(Fig. 16)에서 212, 405 GHz에서 증가하는 두 번째 sub-THz 성분을 발견(100 GHz에서 5×10⁴ sfu 피크). 주파수에 따라 증가하는 스펙트럼 — 완전히 새로운 플레어 행태. 후보 메커니즘:
- **상대론적 양전자 synchrotron** (Fleishman & Kontar 2010): 핵반응 생성 양전자가 증가 스펙트럼 생성; 전환 주파수에서 편광 반전(Fig. 17).
- **Diffusive 복사**: 밀도 요동 산란에 의한 제동복사 유사 방출.
- **Cherenkov 방출**: 추가 연구 필요.
- **증발된 고밀도 플라즈마의 열 free-free**: 광학 얇은 RJ와 일치하는 평탄 연속체.

양전자 서명은 독특하다: 양전자는 ordinary mode (o-mode)를, 전자는 extraordinary mode (x-mode)를 선호. 10–20 MeV 부근 원편광 반전(Fig. 17 삽입)이 결정적 단서. ALMA의 35–950 GHz 광대역 영상분광과 SDO/HMI 벡터 자기도 결합으로 이들 메커니즘을 구별하고 상대론적 양전자 성분을 정량화. 열화 후 양전자는 positronium을 형성, 203 GHz 전이로 검출 가능(Ellis & Bland-Hawthorn 2009).

### Part VI: Prominences, Coronal Rain, and Waves (§4.7–§4.8) / 홍염, 코로나 비, 파동 (§4.7–§4.8)

**English**: **Prominences** are cool (~7000 K), dense plasma suspended in a hotter, rarefied corona by magnetic fields. ALMA's unique capability: measuring prominence plasma temperatures with high spatial/temporal resolution. Since the chromospheric prominence plasma is optically thin at ALMA wavelengths, brightness temperatures directly give kinetic temperatures after simple radiative transfer. Multi-band observations scan the prominence thermal structure from cool cores to hotter Prominence-Corona Transition Region (PCTR). Previous studies (Bastian 1993a, Harrison 1993, Irimajiri 1995) used low-resolution interferometers — ALMA resolves fine structures of ~100 km.

**Coronal rain** (§4.7.2) consists of cool (2000–10000 K) chromospheric-transition-temperature clumps falling back along coronal loop field lines. The strand-like substructure with widths ~100 km or less (Fig. 19) was previously inferred but not resolved. ALMA's thermometer capability is ideal: coronal rain is a unique probe of coronal heating distribution (footpoint-concentrated heating produces thermal non-equilibrium cycles; loops overload and catastrophically cool). The existence of rain at 2000 K (near H recombination regime) confirms extreme temperature inhomogeneity.

**Waves and oscillations** (§4.6) — acoustic, magneto-acoustic shock, Alfvén, magneto-acoustic kink waves — propagate through the chromosphere carrying energy. Phase spectra between different formation heights yield phase speeds and thus wave modes. The vanishing phase lag between Ca II 854.2 nm and 849.8 nm was misinterpreted as standing waves (Mein 1971); later correction (Skartlien 1994) showed they form at similar heights. ALMA's multi-band sampling at truly different heights (Δz = 500–1500 km across bands) enables unambiguous phase speed measurements. Alfvén waves damped by ion-neutral collisions: Soler et al. (2015) found a critical wavelength interval of strong dissipation; ALMA can test these predictions by observing temperature response to wave passage.

**한국어**: **홍염**은 뜨겁고 희박한 코로나에 자기장으로 지지되는 차갑고(~7000 K) 조밀한 플라즈마. ALMA의 독특한 능력: 홍염 플라즈마 온도의 고해상도 측정. ALMA 파장에서 홍염 플라즈마는 광학 얇기 때문에 휘도 온도가 단순한 복사전달 후 운동 온도를 직접 제공. 다파장 관측으로 냉각 중심부부터 뜨거운 PCTR까지 열 구조 단층촬영.

**코로나 비**(§4.7.2): 코로나 loop 자기장선을 따라 떨어지는 차가운(2000–10000 K) 채층/전이영역 온도 덩어리. Strand 폭 ~100 km(Fig. 19)은 이전에 추론만 되었으나 미해결. ALMA의 온도계 기능이 이상적: 코로나 비는 코로나 가열 분포의 독특한 탐침(발자국 집중 가열이 열 비평형 순환 생성). 2000 K(수소 재결합 영역) 존재는 극단적 온도 비균질성 확인.

**파동과 진동**(§4.6): 음향파, 자기음향 충격파, Alfvén, 자기음향 kink 파동이 채층을 통해 에너지 수송. 서로 다른 형성 높이 간 위상 스펙트럼으로 위상 속도와 파동 모드 도출. Ca II 854.2 nm과 849.8 nm 사이 소멸 위상 지연을 정상파로 오해(Mein 1971), 후에 같은 높이에서 형성됨을 발견(Skartlien 1994). ALMA의 다파장 관측은 실제 다른 높이(Δz = 500–1500 km)를 샘플링하여 위상 속도 측정 가능.

### Part VII: Concluding Observations / 결론

**English**: ALMA will transform solar chromospheric physics by providing, for the first time, (1) direct, near-linear thermometer at sub-arcsecond resolution, (2) multi-band height tomography, (3) polarimetric chromospheric magnetometry, (4) high cadence (< 1 s snapshot) for wave propagation, (5) sub-THz flare diagnostics. The SSALMON network coordinates simulation-based preparation to interpret ALMA observations. Regular solar observing begins Cycle 4 (late 2016). Coordinated observations with DKIST (4-m, 2020), IRIS, Hinode, SDO, EST, and Solar-C will multiply ALMA's scientific output.

**한국어**: ALMA는 (1) 서브초각 분해능의 선형 온도계, (2) 다파장 높이 단층촬영, (3) 편광 채층 자기장 측정, (4) < 1 s snapshot 고 cadence, (5) sub-THz 플레어 진단을 처음으로 가능케 한다. SSALMON 네트워크가 시뮬레이션 기반 해석을 조율. Cycle 4 (2016년 말)부터 정기 태양 관측 개시. DKIST(4 m, 2020), IRIS, Hinode, SDO, EST, Solar-C와의 조율 관측이 ALMA의 과학적 산출을 증폭할 것이다.

---

## 3. Key Takeaways / 핵심 시사점

1. **ALMA is a nearly linear chromospheric thermometer / ALMA는 거의 선형인 채층 온도계이다** — In the Rayleigh-Jeans regime, T_b = (c²/2k_Bν²)·I_ν is directly proportional to the local gas temperature in the optically thick continuum-forming layer; unlike Ca II or Mg II, no non-LTE fitting is required. / RJ 영역에서 T_b ∝ T_gas이며, Ca II/Mg II와 달리 non-LTE 복잡 피팅이 불필요하다.

2. **Wavelength samples height / 파장이 높이를 샘플링한다** — The free-free opacity scales as χ ∝ n_e²/(ν²T^{3/2}), so longer wavelengths become optically thick at greater heights. Formation heights: 0.3 mm → 490 km, 1 mm → 730 km, 3 mm → 960 km, 9 mm → 1170 km (Wedemeyer-Böhm 2007 quiet Sun); enhanced network (Loukitcheva 2015a) formation heights are 40–70% higher. ALMA's 10 bands give multi-height tomography. / 파장이 길수록 형성층이 높다. ALMA 10개 대역으로 다층 단층촬영 가능.

3. **Unprecedented spatial resolution for mm observations / mm 관측의 전례 없는 공간 분해능** — ALMA's synthesized beam reaches 0.3″–0.4″ × (λ/1 mm), comparable to 1-m optical solar telescopes and a factor of ~30 improvement over BIMA's 10″. Primary beam is ~19″ × (λ/1 mm). / 합성 빔 0.3″×(λ/1mm)으로 BIMA 대비 30배 향상.

4. **Chromospheric magnetometry via free-free polarization / free-free 편광을 통한 채층 자기장 측정** — Circular polarization 𝒫 = ζ·(ν_B/ν)·cos θ allows direct measurement of B_l. For 1 kG sunspot at 1 mm, 𝒫 ≈ 2%; for quiet Sun at 3 mm, 𝒫 ≈ 0.5% giving B_l within ±100 G, a breakthrough because previous methods (Ca II IR, He I 1083 nm) suffered weak signal and non-LTE complications. / 원편광도에서 종방향 자기장 B_l 직접 측정; 흑점에서 2%, 조용한 태양에서 0.5%.

5. **High cadence opens new physics / 고 cadence로 새로운 물리학 개방** — < 1 s snapshot imaging enables chromospheric seismology: capture of 3-min chromospheric oscillations, quasi-periodic pulsations in flares (fractions of s to min), umbral oscillations, propagating shocks, wave propagation across formation heights. / snapshot < 1 s로 3분 채층 진동, QPP, 암부 진동, 충격파 실시간 추적.

6. **Non-equilibrium hydrogen ionization is essential / 비평형 수소 이온화는 필수적이다** — The LTE assumption for H ionization can fail in the dynamic chromosphere. Non-eq treatment (Leenaarts & Wedemeyer-Böhm 2006) reduces mean 1-mm T_b from 4476 K to 3936 K, narrows formation layer, and alters histogram shape. Interpretation of ALMA maps requires 3D MHD + NLTE hydrogen. / 수소 이온화 LTE 가정은 동적 채층에서 실패; 3D MHD + NLTE 수소가 ALMA 해석에 필수.

7. **Solar mm/sub-mm flare sub-THz component is a major unsolved mystery / 태양 플레어의 sub-THz 성분은 중대한 미해결 수수께끼** — Unexplained increasing spectrum toward THz (Kaufmann et al. 2009b, Fig. 16) may be synchrotron from positrons, diffusive radiation, or Cherenkov emission. ALMA's broadband imaging spectroscopy in the sub-THz range is the decisive tool. / sub-THz 증가 스펙트럼의 기원 (상대론적 양전자? 확산 복사? Cherenkov?) 미해결.

8. **Chromospheric fine-structure requires short baselines (ACA) / 채층 미세 구조에는 짧은 기저선(ACA)이 필수** — Interferometric arrays miss emission on scales larger than λ/L_min; the 7-m ACA plus 12-m TP are essential because the chromosphere fills the FOV. Without them, the background 4000–7000 K disk is filtered out. / 채층 구조는 전 공간 규모에 걸쳐 있어, 짧은 기저선 ACA와 단일 디스크 TP 없이는 배경 4000–7000 K disk가 제거됨.

9. **3D MHD + radiative-transfer forward modeling is indispensable / 3D MHD + 복사전달 전방모델링은 필수** — Interpreting ALMA maps requires synthetic brightness temperature from models like Bifrost (Gudiksen 2011), CO⁵BOLD (Freytag 2012) with transfer codes LINFOR3D, RH, PHOENIX/3D. Static 1D VAL/FAL models cannot reproduce the intermittent thermal structure revealed by ALMA. SSALMON coordinates this international effort. / ALMA 맵 해석은 Bifrost, CO⁵BOLD 3D MHD와 LINFOR3D/RH/PHOENIX 복사전달 합성이 필수. 1D VAL/FAL 정적 모델은 ALMA가 드러내는 간헐적 열 구조를 재현 불가.

10. **CO molecular thermometer extends the diagnostics / CO 분자 온도계가 진단 확장** — CO pure rotational J = 6→5 (691 GHz) and 7→6 (806 GHz) lines can probe the "COmosphere" — cold pockets at ~3500 K (potentially down to 2000 K) in the upper photosphere / low chromosphere that persist in 3D shock dynamics. ALMA will resolve the spatial distribution of these cold regions. / ALMA 691/806 GHz에서 CO 회전 전이로 ~3500 K 저온 포켓(COmosphere) 진단.

---

## 4. Mathematical Summary / 수학적 요약

### (a) Rayleigh-Jeans Brightness Temperature / RJ 휘도 온도

$$B_\nu(T) = \frac{2h\nu^3}{c^2}\left(e^{h\nu/k_B T}-1\right)^{-1} \xrightarrow{h\nu\ll k_B T} \frac{2 k_B T \nu^2}{c^2}$$

$$\boxed{T_b \equiv \frac{c^2}{2 k_B \nu^2} I_\nu}$$

*Variables / 변수*: B_ν Planck source function [erg s⁻¹ cm⁻² Hz⁻¹ sr⁻¹]; c speed of light; h Planck; k_B Boltzmann; ν frequency; T kinetic gas temperature; I_ν specific intensity. In optically thick, isothermal layer: T_b = T_gas. **의미 / Meaning**: RJ 한계에서 T_b ≈ 기체 온도.

### (b) Free-Free Absorption Coefficient / Free-free 흡수 계수

Full semi-classical expression (Dulk 1985, Karzas-Latter Gaunt factor):
$$\chi_{\text{ions,ff}} \approx \frac{1}{3c}\left(\frac{2}{\pi}\right)^{1/2}\frac{\nu_p^2}{\nu^2}\frac{4\pi e^4}{\sqrt{m_e}(k_BT)^{3/2}}\sum_i Z_i^2 n_i \frac{\pi}{\sqrt{3}} g_{\text{ions,ff}}(T,\nu)$$

Gaunt factor:
$$g_{\text{ions,ff}}(T,\nu) = \frac{\sqrt{3}}{\pi}\ln\!\left(\frac{(2 k_B T)^{3/2}}{2\pi\Gamma\nu\sqrt{m_e}e^2}\right)$$

Plasma frequency:
$$\nu_p = \sqrt{\frac{e^2 n_e}{\pi m_e}}$$

Simplified form (Kundu 1965):
$$\boxed{\chi \approx \xi(T,e)\, \frac{n_e^2}{n\,\nu^2 T^{3/2}},\quad \xi \simeq 0.12\ (\text{chromo}),\ 0.2\ (\text{corona})}$$

**의미 / Meaning**: 불투명도 ∝ n_e²/(ν²T^{3/2}). 저주파에서 불투명도 증가 → 더 높은 층.

### (c) Radiative Transfer / 복사 전달

$$I_\nu = \int_0^{\tau_{\max}(\nu)} B_\nu(T(\bar\tau))\, e^{-\bar\tau}\, d\bar\tau, \quad \tau_{\max} = \infty\ (\text{optically thick})$$

For isothermal slab at T: I_ν = B_ν(T)·(1 − e^{−τ}). Optically thick: I_ν → B_ν(T), T_b → T.

### (d) ALMA Primary Beam (Diffraction) / ALMA 주빔

$$\boxed{\theta_{PB} \approx 1.13\frac{\lambda}{D} \approx 19''\times\frac{\lambda}{1\,\text{mm}} \quad (D = 12\,\text{m})}$$

Maximum recoverable scale:
$$\vartheta_{\max} = \frac{0.6\lambda}{L_{\min}}\,\text{rad} = \frac{37100}{L_{\min}[\text{m}]\,\nu[\text{GHz}]}\,\text{arcsec}$$

For ACA 7-m (L_min = 8.4 m): ϑ_max ≈ 44″ at 100 GHz.

Synthesized (diffraction-limited) resolution scales as ≈ 0.3″–0.4″ × (λ/1 mm), set by the longest baseline.

### (e) Circular Polarization & Magnetic Field / 원편광과 자기장

Free-free brightness slope:
$$\zeta = \frac{d\log T_b(\nu)}{d\log\nu}$$

Polarization:
$$\boxed{\mathcal{P} = \zeta\frac{\nu_B}{\nu}\cos\theta, \quad \nu_B = \frac{e B}{2\pi m_e c} = 2.8\times 10^6\,\text{Hz}\times B[\text{G}]}$$

Optically thin, isothermal: ζ = 2 → 𝒫 = 2(ν_B/ν)cos θ; numerically 𝒫(%) ≈ 1.85×10⁻³·λ(mm)·B(G)·cos θ. Longitudinal B:
$$B_l = |\mathbf{B}|\cos\theta = \frac{\mathcal{P}\nu}{\zeta}\cdot (2.8\times 10^6)^{-1}\ (\text{G})$$

### (f) Contribution Function and Formation Height / 기여함수와 형성 높이

The contribution function C(z, ν) weights the Planck function:
$$I_\nu = \int B_\nu(T(z))\, C(z,\nu)\, dz, \quad C(z,\nu) = \frac{d\tau_\nu}{dz}\, e^{-\tau_\nu(z)}$$

Effective formation height z_m(ν) = max of C(z,ν). For quiet Sun (Wedemeyer-Böhm 2007): z_m = 490, 730, 960, 1170 km at λ = 0.3, 1.0, 3.0, 9.0 mm. FWHM: 220, 360, 440, 480 km. Spatial standard deviation Δz_m/z_m,avg ≈ 15%.

### (g) Worked Numerical Example / 구체적 수치 예시

**Example 1 — Optical depth at 3 mm**: For quiet-Sun chromospheric layer at λ = 3 mm (ν = 100 GHz), assume T = 6500 K, n_e = 5×10¹⁰ cm⁻³:

χ ≈ 0.12 × (5×10¹⁰)² / (10²⁰ × 6500^{1.5}) ≈ 0.12 × 2.5×10²¹ / (10²⁰ × 5.24×10⁵) = 0.12 × 4.77×10⁻⁵ ≈ 5.7×10⁻⁶ cm⁻¹.

Scale height H ~ 200 km = 2×10⁷ cm → τ ≈ χ·H ≈ 114, i.e., optically thick, confirming T_b ≈ T. Observed T_b range 4000–8000 K (bright network) consistent with gas temperature 4500–8500 K. / 광학 두께 114 → optically thick 확인.

**Example 2 — Polarization in plage**: at ν = 100 GHz, ν_B = 2.8 MHz for B = 1 G; for plage B = 100 G, ν_B = 280 MHz. With ζ = 2 (optically thin slab approximation): 𝒫 ≈ 2 × 2.8×10⁸/10¹¹ = 0.56% — detectable by ALMA (requirement 0.1%). For strong sunspot (B = 1 kG) at 1 mm (300 GHz): 𝒫 ≈ 2×2.8×10⁹/3×10¹¹ ≈ 1.9%.

**Example 3 — Mosaic timing**: to cover full Sun (2000″) at 1 mm with 19″ beam requires (2000/19)² ≈ 11000 pointings; with TP fast scanning, 8 min per scan (Fig. 3 shows 230 GHz full disk in 10 min double-circle pattern).

**Example 4 — Band-by-band resolution table**: ALMA Band / λ (mm) / ν (GHz):
- Band 3: 2.6–3.6 / 84–116 → synth. resolution 0.06″ (16 km baseline)
- Band 4: 1.8–2.4 / 125–163 → 0.04″
- Band 5: 1.4–1.8 / 163–211 → 0.03″
- Band 6: 1.1–1.4 / 211–275 → 0.025″
- Band 7: 0.8–1.1 / 275–373 → 0.02″
- Band 9: 0.4–0.5 / 602–720 → 0.01″

**Example 5 — Brightness temperature from RJ**: at ν = 240 GHz and T = 7000 K chromospheric source, B_ν = 2k_B T ν²/c² = 2 × 1.38×10⁻²³ × 7000 × (2.4×10¹¹)²/(3×10⁸)² = 1.23×10⁻¹¹ W m⁻² Hz⁻¹ sr⁻¹. With 1″ beam solid angle 2.35×10⁻¹¹ sr, flux ~0.29 mJy per beam — within ALMA sensitivity limits.

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
Year    Milestone
────────────────────────────────────────────────────
1959 ── Kundu: first solar interferometric radio obs.
1961 ── Karzas & Latter: quantum Gaunt factor (used in Eq. 1)
1965 ── Kundu: simplified free-free expression (Eq. 5)
1975 ── WSRT (6, 20 cm) first 2D imaging
1981 ── Vernazza, Avrett, Loeser: VAL semi-empirical models
1985 ── Dulk: free-free absorption semi-classical form (Eq. 1)
1993 ── Fontenla et al.: FAL atmospheric models
1994 ── Carlsson & Stein: 1D dynamic shock chromosphere simulation
1998 ── Bastian et al. (PAPER #26): radio solar emission review
2002 ── Bastian: solar ALMA proposal era begins
2004 ── Loukitcheva et al.: BIMA vs. simulation comparison
2007 ── Wedemeyer-Böhm et al.: 3D contribution functions (Fig. 5)
2009 ── BIMA solar images (Loukitcheva 2009)
2011 ── ALMA commences (no solar mode)
2012 ── Wedemeyer-Böhm et al.: chromospheric tornado
2014 ── First ALMA solar commissioning data, SSALMON formed
2015 ── Solar observing modes tested
2016 ── THIS PAPER (Wedemeyer et al. 2016) published; Cycle 4 begins
2017 ── Shimojo et al.: first ALMA quiet-Sun T measurements
2017 ── White et al.: ALMA chromospheric mosaic
2020+─ Coordinated ALMA + DKIST (4-m, 2020) + IRIS + Solar-C campaigns
```

**Historical arc commentary / 역사적 해설**:

*English*: The paper sits at a pivotal transition point in solar physics. Before 2016, mm/sub-mm solar observations were a specialty curiosity — the BIMA images in Fig. 2 were state of the art but fundamentally limited to ~10″ resolution. The chromosphere was studied primarily through Ca II K, Hα, Mg II — all complicated non-LTE diagnostics. After 2016, ALMA made mm-wave solar observing mainstream, offering sub-arcsecond resolution and a near-linear thermometer. The paper's prescient predictions — for example, that observing 1 mm with 0.2″ resolution yields chromospheric contrast ~24%, dropping to ~10% at 1″ — have been largely borne out by the early ALMA campaigns.

*한국어*: 이 논문은 태양물리학의 결정적 전환점에 위치한다. 2016년 이전의 mm/sub-mm 태양 관측은 특수 분야였고 BIMA 영상은 ~10″로 제한되었다. 채층은 주로 Ca II K, Hα, Mg II로 연구되었으나 모두 복잡한 non-LTE 진단이다. 2016년 이후 ALMA는 mm 파장 태양 관측을 주류로 만들어 서브각초 분해능과 거의 선형인 온도계를 제공한다. 논문의 선견지명적 예측들(예: 0.2″ 분해능에서 1 mm 채층 대비 ~24%, 1″에서 ~10%로 감소)은 초기 ALMA 관측에서 대부분 확인되었다.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Bastian et al. 1998 (Paper #26) | General radio-frequency solar emission mechanisms review — free-free, gyroresonance, gyrosynchrotron, plasma emission / 전파 방출 메커니즘 리뷰 | Direct theoretical foundation; ALMA paper extends to mm/sub-mm regime specifically / ALMA 논문의 직접적 이론 기반 |
| Dulk 1985 (Annu. Rev.) | Simple formulas for plasma emission and absorption; used for Eq. 1–5 / 플라즈마 방출/흡수 공식 | Free-free absorption coefficient derivation / free-free 흡수계수 유도 |
| Wedemeyer-Böhm et al. 2007 | 3D radiative-MHD model; contribution functions at mm wavelengths (Fig. 5) / 3D RMHD 모델, 기여함수 | Provides the quiet-Sun formation heights used throughout / 형성 높이 데이터 제공 |
| Vernazza, Avrett, Loeser 1981 (VAL) | Classical 1D semi-empirical quiet-Sun model / 고전 1D 준경험 모델 | Reference atmosphere against which 3D models and ALMA are compared / 비교 기준 대기 |
| Carlsson & Stein 1994, 1995 | 1D propagating shock chromosphere simulation; explains Ca II K-grains / 전파 충격파 채층 시뮬레이션 | Key heating mechanism ALMA will test / ALMA로 검증할 가열 메커니즘 |
| Loukitcheva et al. 2004, 2015a | Synthetic mm brightness maps from 3D simulations; enhanced network model / 합성 mm 밝기 맵 | Direct predictions compared with ALMA observations (Fig. 10) / ALMA 관측과 직접 비교 |
| Karzas & Latter 1961 | Quantum mechanical Gaunt factor; used in Eq. 2 / 양자 Gaunt factor | Fundamental opacity calculation / 기초 불투명도 계산 |
| Kaufmann et al. 2009b | Sub-THz flare component discovery (212 + 405 GHz, Fig. 16) / sub-THz 플레어 성분 발견 | Motivates ALMA flare observations; unresolved mechanism / ALMA 플레어 관측 동기 |
| Gudiksen et al. 2011 (Bifrost) | 3D stellar atmosphere simulation code / 3D 항성 대기 시뮬레이션 코드 | Provides Model A (Table 1) used for all synthetic maps / Table 1 모델 A 제공 |
| Leenaarts & Wedemeyer-Böhm 2006 | Non-equilibrium hydrogen ionization treatment / 비평형 수소 이온화 처리 | Model D vs. E comparison (Fig. 10d); reduces T_b from 4476 K to 3936 K / NLTE 보정 기법 |
| Kaufmann et al. 2001, 2004, 2009a | Original discovery papers of sub-THz component / sub-THz 성분 발견 원논문 | Observational basis of the sub-THz mystery / sub-THz 수수께끼 관측적 근거 |
| Fleishman & Kontar 2010 | Diffusive radiation mechanism for sub-THz / sub-THz 확산 복사 메커니즘 | Alternative to positron synchrotron / 양전자 synchrotron 대안 |
| Fleishman et al. 2015 (GX Simulator) | 3D modeling of active region mm emission / 활동영역 mm 방출 3D 모델링 | Produces Fig. 13 AR 12158 predictions / Fig. 13 예측 생성 |

---

## 7. References / 참고문헌

- Wedemeyer, S., Bastian, T., Brajša, R., et al., "Solar science with the Atacama Large Millimeter/submillimeter Array — A new view of our Sun", *Space Science Reviews*, 200, 1–73 (2016). DOI: 10.1007/s11214-015-0229-9
- Bastian, T. S., Benz, A. O., Gary, D. E., "Radio emission from solar flares", *Annu. Rev. Astron. Astrophys.*, 36, 131 (1998). [Paper #26]
- Dulk, G. A., "Radio emission from the sun and stars", *Annu. Rev. Astron. Astrophys.*, 23, 169 (1985).
- Karzas, W. J., Latter, R., "Electron radiative transitions in a Coulomb field", *Astrophys. J. Suppl.*, 6, 167 (1961).
- Vernazza, J. E., Avrett, E. H., Loeser, R., "Structure of the solar chromosphere. III.", *Astrophys. J. Suppl.*, 45, 635 (1981).
- Fontenla, J. M., Avrett, E. H., Loeser, R., "Energy balance in the solar transition region. III.", *Astrophys. J.*, 406, 319 (1993).
- Carlsson, M., Stein, R. F., "Non-LTE radiating acoustic shocks and CA II K2V bright points", *Astrophys. J. Lett.*, 440, L29 (1995).
- Wedemeyer-Böhm, S., Ludwig, H.-G., Steffen, M., Leenaarts, J., Freytag, B., "Inter-network regions of the Sun at millimetre wavelengths", *Astron. Astrophys.*, 471, 977 (2007).
- Loukitcheva, M., Solanki, S. K., Carlsson, M., Stein, R. F., "Millimeter observations and chromospheric dynamics", *Astron. Astrophys.*, 419, 747 (2004).
- Loukitcheva, M., Solanki, S. K., White, S. M., "Imaging the solar chromosphere in the millimetre range", *Astron. Astrophys.*, 575, A15 (2015a).
- Kaufmann, P., Trottet, G., Giménez de Castro, C. G., et al., "Sub-terahertz, microwaves and high energy emissions during the 6 December 2006 flare", *Solar Phys.*, 255, 131 (2009b).
- Kundu, M. R., *Solar Radio Astronomy*, Interscience (1965).
- Brajša, R., Benz, A. O., Temmer, M., Jurdana-Šepić, R., Šaina, B., Wöhl, H., "An interpretation of the coronal holes' visibility in the millimeter wavelength range", *Solar Phys.*, 245, 167 (2007).
- Gudiksen, B. V., Carlsson, M., Hansteen, V. H., et al., "The stellar atmosphere simulation code Bifrost", *Astron. Astrophys.*, 531, A154 (2011).
- Fleishman, G. D., Kuznetsov, A. A., "Fast gyrosynchrotron codes", *Astrophys. J.*, 721, 1127 (2010).
- Shimojo, M., Bastian, T. S., Hales, A. S., et al., "Observing the Sun with the Atacama Large Millimeter/submillimeter Array (ALMA): Fast-scan single-dish mapping", *Solar Phys.*, 292, 87 (2017).
- White, S. M., Iwai, K., Phillips, N. M., et al., "Observing the Sun with ALMA — receiving and calibration methods for the Solar SV campaign", *Solar Phys.*, 292, 88 (2017).
- Leenaarts, J., Wedemeyer-Böhm, S., "Time-dependent hydrogen ionisation in 3D simulations of the solar chromosphere", *Astron. Astrophys.*, 460, 301 (2006).
- Fleishman, G. D., Kontar, E. P., "Sub-THz radiation mechanisms in solar flares", *Astrophys. J. Lett.*, 709, L127 (2010).
- Gudiksen, B. V., Carlsson, M., Hansteen, V. H., Hayek, W., Leenaarts, J., Martínez-Sykora, J., "The stellar atmosphere simulation code Bifrost", *Astron. Astrophys.*, 531, A154 (2011).
- Freytag, B., Steffen, M., Ludwig, H.-G., et al., "Simulations of stellar convection with CO⁵BOLD", *J. Comput. Phys.*, 231, 919 (2012).
- Hauschildt, P. H., Baron, E., "A 3D radiative transfer framework. I. Non-local operator splitting and continuum scattering", *Astron. Astrophys.*, 509, A41 (2010).
- Kaufmann, P., Raulin, J.-P., de Castro, C. G. G., et al., "A new solar burst spectral component emitting only in the terahertz range", *Astrophys. J. Lett.*, 603, L121 (2004).
- Loukitcheva, M., Solanki, S. K., White, S. M., "Prospects for solar studies with interferometric arrays at millimeter/submillimeter wavelengths", *Astron. Astrophys.*, 497, 273 (2009).
- Wedemeyer, S., et al., "Advances in solar simulations and mm-wave diagnostics — SSALMON white paper", *arXiv* (2015).
