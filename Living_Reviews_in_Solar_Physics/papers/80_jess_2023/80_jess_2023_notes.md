---
title: "Waves in the Lower Solar Atmosphere: The Dawn of Next-Generation Solar Telescopes"
authors: [Jess, Jafarzadeh, Keys, Stangalini, Verth, Grant]
year: 2023
journal: "Living Reviews in Solar Physics"
doi: "10.1007/s41116-022-00035-6"
topic: Living_Reviews_in_Solar_Physics
tags: [waves, photosphere, chromosphere, MHD, sunspot, mode-conversion, DKIST, review]
status: completed
date_started: 2026-04-23
date_completed: 2026-04-23
---

# 80. Waves in the Lower Solar Atmosphere: The Dawn of Next-Generation Solar Telescopes / 하층 태양 대기의 파동 — 차세대 태양 망원경 시대의 서막

---

## 1. Core Contribution / 핵심 기여

Jess et al. (2023) deliver a 170-page Living Review with a deliberately two-pronged structure. The first half (Sect. 2) is a pedagogical handbook on wave-analysis methodology: Fourier and three-dimensional Fourier techniques, confidence-level estimation, Lomb–Scargle, wavelets (Morlet, Paul), Empirical Mode Decomposition, Proper Orthogonal Decomposition (POD) and Dynamic Mode Decomposition (DMD), k-ω and B-ω diagrams, the effect of spatial resolution, and the theoretical identification of MHD wave modes in homogeneous unbounded plasmas and in the magnetic cylinder geometry (including elliptical cross-section extensions). The second half (Sect. 3) is a literature review of a decade of observational/theoretical progress, partitioned into global wave modes (3.1), large-scale magnetic structures such as sunspots and pores (3.2–3.3), and small-scale magnetic structures including MBPs, fibrils and spicules (3.4). A Sect. 4 is devoted to future directions — DKIST post-focus instruments (ViSP, VBI, VTF, DL-NIRSP, Cryo-NIRSP), balloon-borne Sunrise-III SUSI/SCIP, Solar Orbiter/PHI, Solar-C, and the upcoming NLST/FRANCIS fibre-ferrule spectropolarimeter.

Jess 등(2023)은 의도적으로 이원 구조로 설계된 170쪽 Living Review이다. 전반부(2절)는 파동 분석 방법론의 교본이다 — 1D/3D Fourier 기법, 신뢰 수준 추정, Lomb–Scargle, wavelet(Morlet·Paul), Empirical Mode Decomposition(EMD), Proper Orthogonal Decomposition(POD)·Dynamic Mode Decomposition(DMD), k-ω·B-ω 다이어그램, 공간 분해능 효과, 그리고 균질·무경계 플라스마와 자기 실린더(타원 단면 확장 포함) 형상에서의 MHD 모드 식별을 다룬다. 후반부(3절)는 지난 10년의 관측·이론 진전을 (3.1) 전역 파동 모드, (3.2–3.3) 흑점·포어 등 대규모 자기 구조, (3.4) MBP·fibril·spicule 등 소규모 자기 구조로 분할해 정리한다. 4절은 DKIST 후초점 관측기기(ViSP·VBI·VTF·DL-NIRSP·Cryo-NIRSP), 풍선 탑재 Sunrise-III SUSI/SCIP, Solar Orbiter/PHI, Solar-C, 차세대 인도 NLST/FRANCIS 광섬유 분광편광계 등 향후 시설에 대한 로드맵을 제시한다.

The review's unifying thesis is that the lower solar atmosphere must be treated as a single continuous MHD waveguide where the plasma-β varies by several orders of magnitude over ~1500 km of height, and that mode conversion/transmission across the β≈1 equipartition layer is the central physical event that determines how wave energy is funnelled upward. This framing replaces the older, feature-by-feature compartmentalisation (treating sunspots, network, plage and quiet Sun as separate chapters) and is made possible only by modern multi-wavelength, multi-height, high-cadence imaging and spectropolarimetric inversion capabilities.

본 리뷰의 통합 테제는 하층 대기를 β가 1500 km 남짓의 고도 범위에서 수 차수만큼 변하는 단일 연속 MHD 도파관으로 보아야 하며, β≈1 등분배층에서의 모드 변환/투과가 코로나로의 에너지 유입 경로를 결정하는 중심 물리 사건이라는 것이다. 이 관점은 흑점·네트워크·플레이지·조용한 태양을 분리해 다루던 과거 방식을 대체하며, 현대의 다파장·다고도·고시간분해능 영상과 분광편광 역산 기법이 있기에 가능해졌다.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction and Motivation (Sect. 1, pp. 2–8) / 서론과 동기

The review opens with the two classical heating mechanisms: DC (reconnection, nano/microflares of ~10²⁴–10²⁷ erg) and AC (wave-driven, Schwarzschild 1948). It argues that neither has been definitively identified as the dominant coronal heater, but the lower atmosphere (photosphere + chromosphere) is where the magnetic footpoints live and therefore where wave generation is most directly diagnosable. The flaring energy distribution follows a power law (Shimizu & Tsuneta 1997; Parnell & Jupp 2000), with the most numerous events being the smallest — nanoflares at ~10²⁴ erg and microflares at ~10²⁷ erg — but their cumulative heating contribution remains uncertain because they typically lie near or below noise levels of current facilities (Terzo+2011; Jess+2014, 2019). Technological milestones cited: adaptive optics (Rimmele & Marino 2011), speckle reconstruction (Wöger+2008), MOMFBD (van Noort+2005). Figure 1 juxtaposes DST construction (1969, Sacramento Peak) with DKIST construction (2019, Haleakalā), emphasising a ~50-year generational gap. Early wave detections (Leighton 1960; Leighton, Noyes & Simon 1962; Sheeley & Bhatnagar 1971) are revisited through Fig. 2, showing the classical Doppler-velocity map of a sunspot region with 300-s, ~0.6 km/s oscillations.

Historical interpretive confusion is worth noting: Osterbrock (1961) and Mein & Mein (1976) had reported inconsistencies between measured phase velocities of waves between two atmospheric heights and a purely acoustic wave interpretation. It was not yet appreciated that the 5-min oscillations are evanescent rather than propagating, and that magnetic fields fundamentally modify wave character (Michalitsanos 1973; Nakagawa 1973; Stein & Leibacher 1974). The field of MHD was thus introduced, linking observed wave signatures to underlying magnetic configurations where field strengths can reach ~6000 G in sunspot umbrae (Livingston+2006; Okamoto & Sakurai 2018), producing wave modes highly modified from purely acoustic counterparts.

리뷰는 두 가지 고전적 가열 기작으로 시작한다 — DC(재결합, nano/microflare ~10²⁴–10²⁷ erg)와 AC(파동, Schwarzschild 1948). 두 기작 중 어느 쪽도 코로나의 지배적 가열원으로 확정되지 않았으나, 하층 대기는 자기 기저부가 놓인 지역으로 파동 생성을 직접 진단할 수 있다. 플레어 에너지는 멱법칙 분포를 따르며(Shimizu & Tsuneta 1997; Parnell & Jupp 2000), 수가 가장 많은 사건은 가장 작은 nanoflare(~10²⁴ erg)·microflare(~10²⁷ erg)이지만 이들의 누적 기여는 현재 시설의 잡음 수준에 근접해 불확실하다. Osterbrock(1961)·Mein & Mein(1976)이 두 고도 사이 위상속도가 순수 음파 해석과 일치하지 않음을 보고했을 때, 5분 진동이 전파하지 않는 증발형이며 자기장이 파동 성격을 근본적으로 바꾼다는 점을 당시에는 인식하지 못했다 — 이후 MHD의 도입으로 ~6000 G 흑점 umbra의 강자기장(Livingston+2006; Okamoto & Sakurai 2018)이 순수 음파와는 크게 다른 모드를 낳는다는 사실이 정착됐다.

리뷰는 두 가지 고전적 코로나 가열 기작 — DC(재결합, nano/microflare ~10²⁴–10²⁷ erg)와 AC(파동 구동, Schwarzschild 1948) — 으로 시작한다. 두 기작 중 어느 쪽도 코로나의 주된 가열원으로 확정되지 않았으나, 하층 대기는 자기 기저부이므로 파동 생성 진단이 가장 직접적이라고 논증한다. 기술적 이정표로 적응광학(Rimmele & Marino 2011), speckle 복원(Wöger+2008), MOMFBD(van Noort+2005)가 언급된다. 그림 1은 DST(1969)와 DKIST(2019) 건설 장면을 병치하여 50년의 세대 격차를 드러낸다. 초기 파동 검출(Leighton 1960; Sheeley & Bhatnagar 1971)은 그림 2에서 300 s·~0.6 km/s 흑점 영역 Doppler 속도 지도로 재조명된다.

Plasma-β is introduced via
$$\beta = \frac{2\mu_0 p_0}{B_0^2} = \frac{8\pi n_H k_B T}{B_0^2}$$
(Eqs. 1–2 of the paper). Sunspots (B≳1000 G) and pores are "low-β throughout the lower atmosphere" and act as wave guides from photosphere to corona (Aschwanden+2016; Grant+2018). MBPs, however, lose their low-β status as their fields expand with height, so β>1 can occur by low-chromospheric heights.

플라스마 β는 식 (1–2)로 정의된다. 흑점(B≳1000 G)과 포어는 하층 대기 전 구간에서 저β이므로 광구부터 코로나까지 이어지는 도파관 역할을 한다. MBP는 고도에 따라 자기장이 팽창하여 저층 채층에서 β>1로 전이한다.

### Part II: Wave Analysis Tools (Sect. 2, pp. 8–67) / 파동 분석 도구

**2.1 Observations / 관측 사례** — Two benchmark datasets are used throughout: (i) HARDcam Hα at the DST, 10 Dec 2011, 20 fps, 0.138″/pix, 75-min sequence of NOAA 11366 (Jess+2012a). (ii) SuFI/Sunrise 9 Jun 2009, Ca II H and 300 nm, balloon-borne at 37 km altitude (Solanki+2017). These are reused as worked examples in every subsequent sub-section. / 두 벤치마크 데이터셋이 전 절에 걸쳐 재사용된다 — HARDcam Hα(DST, 2011-12-10, 20 fps, 0.138″/pix)와 SuFI/Sunrise(2009-06-09, Ca II H 및 300 nm).

**2.2 One-dimensional Fourier analysis / 1D Fourier 분석** — Classical PSD via FFT. "Common misconceptions" (2.2.1) warns about (a) zero-padding inflating apparent resolution, (b) windowing choice affecting leakage, (c) the need for the Nyquist criterion f_Nyq = 1/(2Δt). For HARDcam at Δt=1.78 s, f_Nyq ≈ 281 mHz. Confidence levels (2.2.2) are computed against red-noise (AR-1) backgrounds per Torrence & Compo (1998). Lomb–Scargle (2.2.3) handles irregular sampling. One-dimensional Fourier filtering and phase-lag analysis close the subsection. / 2.2절은 고전 FFT PSD, 영-padding과 windowing의 함정, Nyquist 기준(HARDcam의 경우 281 mHz), Torrence & Compo(1998) 방식의 red-noise 대비 신뢰 수준, Lomb–Scargle, 1D Fourier 필터링과 phase-lag 분석을 다룬다.

**2.3–2.4 3D Fourier and wavelets / 3차원 Fourier와 웨이블릿** — The 3D PSD P(k_x, k_y, ω) is central for isolating spatial structure of waves. k-ω slices (Sect. 2.7) reveal p-mode ridges. Wavelet methods give time-frequency localisation; phase velocities are derived via wavelet cross-spectrum phase (Sect. 2.4.1).

**2.5–2.6 EMD and POD/DMD** — Empirical Mode Decomposition splits a signal into intrinsic mode functions, useful for non-stationary signals. POD and DMD, imported from fluid dynamics, project 2D spatial data onto orthonormal/dynamic modes; exceptionally effective for identifying spatially coherent wave eigenmodes in sunspot umbrae (Albidah+2021; Stangalini+2022, see Figs. 57–60 of the paper). POD decomposes the data tensor $X(x,y,t)$ into spatial eigenfunctions $\phi_i(x,y)$ and time-varying amplitudes $a_i(t)$ that are mutually orthogonal and ranked by energy. DMD further assumes each mode oscillates at a single complex frequency, yielding $X(x,y,t) = \sum_i \phi_i(x,y) \exp(\lambda_i t)$ with $\lambda_i = \sigma_i + i\omega_i$ encoding growth rate and frequency. For a circular-umbra sunspot, DMD recovers the expected m=0, 1, 2, ... eigenmodes; for ellipsoidal or irregular umbrae, DMD produces modes that reflect the real geometry rather than idealised cylinders.

2.5-2.6절의 EMD·POD·DMD: EMD는 비정상 신호를 내재 모드 함수로 분해한다. POD는 데이터 텐서 $X(x,y,t)$를 상호 직교 공간 고유함수 $\phi_i(x,y)$와 시간 진폭 $a_i(t)$로 분해하고 에너지 순으로 순위를 매긴다. DMD는 각 모드가 단일 복소 주파수 $\lambda_i = \sigma_i + i\omega_i$로 진동한다고 가정하여 성장률과 주파수를 동시에 추정한다. 원형 umbra에서는 m=0,1,2,... 의 이상적 모드를 재현하지만, 타원 또는 불규칙 umbra에서는 실제 형상을 반영하는 모드를 생성한다.

**2.7 k-ω diagrams / k-ω 다이어그램** — Obtained by integrating the 3D PSD over the azimuthal angle in k-space. p-mode ridges trace dispersion relations $\omega = \omega(k_h)$ for each radial order. The acoustic cutoff $\omega_c$ separates trapped from propagating regimes:
$$\omega^2 = c_s^2 k_z^2 + \omega_c^2, \qquad \omega_c = \frac{c_s}{2H}.$$
For a photospheric isothermal atmosphere T≈5800 K → c_s≈8.5 km/s, H≈150 km, ω_c/(2π) ≈ 5 mHz. Waves below ω_c are evanescent; 5-min p-modes (3 mHz) are therefore trapped inside the Sun. In the chromosphere, T increases but gravity is the same, H grows, and ω_c drops to ~3–4 mHz, allowing 3-min waves to propagate upward — the physical basis of the "3-min chromospheric window".

2.7절의 k-ω 다이어그램은 3D PSD를 k 평면 방위각으로 적분해 얻는다. p-mode 능선은 방사 차수별 분산 관계를 시각화한다. 음향 차단 주파수에 의해 갇힌 영역과 전파 영역이 분리되며(식 위 참조), 등온 광구(T≈5800 K, c_s≈8.5 km/s, H≈150 km)에서 ω_c/2π ≈ 5 mHz이다. 5분 p-mode(3 mHz)는 태양 내부에 갇히고, 채층에서는 온도 상승으로 ω_c가 ~3–4 mHz로 낮아져 3분 파동의 상승 전파를 허용한다.

**2.8 Effect of spatial resolution / 공간 분해능 효과** — The 1-m SST sees twice the acoustic energy flux of the 0.7-m VTT (Bello González+2010 vs 2009). Under-resolved pixels mix oscillatory signatures within the resolution element, destroying phase coherence. This directly motivates DKIST's 4-m aperture. The diffraction limit $\theta \approx 1.22 \lambda/D$ for D=4 m at 500 nm gives $\theta \approx 0.031''$ ≈ 22 km on the Sun — roughly an order of magnitude below typical pressure scale heights in the photosphere (~150 km) and comparable to the expected widths of elementary flux tubes (~50–100 km). Seeing and AO performance also interact with temporal cadence: instrumental "false periodicities" can arise from thermal drift, pointing jitter, and AO lock transients (Sect. 2). The review stresses that scientific calibration (removing slow trends, checking for fixed-pattern noise, verifying the Nyquist condition) must precede any wave analysis.

**2.9 MHD wave mode identification / MHD 모드 식별**

*2.9.1 Homogeneous unbounded plasma.* Three modes: slow, fast, Alfvén. Dispersion relations:
$$\omega^2 = k^2 v_A^2 \cos^2\theta \qquad \text{(Alfvén)}$$
$$\omega^4 - \omega^2 k^2 (c_s^2 + v_A^2) + k^4 c_s^2 v_A^2 \cos^2\theta = 0 \qquad \text{(slow/fast)}$$
where θ is the angle between **k** and **B**. Alfvén speed $v_A = B/\sqrt{\mu_0 \rho}$ and sound speed $c_s = \sqrt{\gamma p/\rho}$. Slow and fast mode branches follow from the biquadratic. Fast is nearly isotropic; slow is "only approximately anisotropic" — propagation along B is unrestricted but strictly perpendicular propagation is forbidden. The Alfvén wave is strictly field-aligned.

*2.9.2 Magnetic cylinder model (Edwin & Roberts 1983).* Flux tube of radius R, internal density ρ_i, external density ρ_e, uniform axial field. Wave numbers: axial k_z, azimuthal m. The azimuthal integer m classifies modes: m=0 sausage, m=1 kink, m≥2 fluting. Radial behaviour classifies as body (oscillatory inside, evanescent outside) or surface (evanescent on both sides, peaked at boundary).

*Characteristic speeds in photospheric conditions* v_A > c_e > c_0 > v_{Ae} produce two trapped bands: slow [c_T, c_0] and fast [c_0, c_e], with tube speed
$$c_T = \frac{c_0 v_A}{\sqrt{c_0^2 + v_A^2}}. \qquad (14)$$
Torsional Alfvén waves exist for any m and propagate strictly along the tube axis (unlike the homogeneous plasma case where Alfvén propagates at any angle to B). For elliptical cross-sections (Aldhafeeri+2021; Figs. 39–41), fluting modes (m=2,3) with even phase symmetry about the major axis can coalesce as eccentricity ε=√(1−b²/a²) increases — the cylindrical classification breaks down for real sunspot umbrae (ε=0.58 and 0.76 observed).

2.9.2절의 자기 실린더 모델은 Edwin & Roberts(1983)를 따라 내부 밀도 ρ_i, 외부 ρ_e의 자속관을 다룬다. 방위각 모드 번호 m으로 sausage(m=0)·kink(m=1)·fluting(m≥2)을 분류하고, 방사 거동에 따라 body/surface 모드로 나눈다. 광구 조건 v_A > c_e > c_0 > v_{Ae}에서 slow 대역 [c_T, c_0]과 fast 대역 [c_0, c_e]의 두 갇힌 대역이 존재하며, c_T는 식 (14)로 정의된다. Torsional Alfvén 파는 균질 플라스마와 달리 자속관 축을 따라서만 전파한다. 타원 단면(Aldhafeeri+2021)의 경우 이심률 ε 증가에 따라 fluting 모드들이 융합하여 단순 실린더 분류가 깨진다 — 실제 흑점은 ε=0.58–0.76로 관측됐다.

### Part III: Global Wave Modes (Sect. 3.1, pp. 68–74) / 전역 파동 모드

Classical helioseismology partitions solar oscillations into p-modes (pressure-driven, 0≤ℓ≤10³), g-modes (buoyancy, confined to radiative interior/upper atmosphere), and f-modes (surface gravity). Photospheric 5-min oscillation at ~3 mHz is the signature of trapped p-modes (Leighton+1962; Ulrich 1970). In the chromosphere, 3-min oscillations (~5 mHz) dominate sunspot umbrae; this is because the effective acoustic cutoff is exceeded there.

Kayshap+2018 using IRIS multi-line (Mn I 2801 Å, Mg II k, C II) identified upward-propagating p-modes with periods 1.6–4.0 min and downward propagation for periods >4.5 min. Internal gravity waves carry ~2 kW/m² (Vigeesh+2021) and have f < 2 mHz. f-mode frequencies are used to measure the "seismic solar radius" (Schou+1997; Dziembowski+2001).

Power halos around magnetic network appear at photospheric/low-chromospheric heights while 3-min magnetic shadows mark the upper chromosphere. The halos may arise from fast-wave reflection at the magnetic canopy (Khomenko & Calvo 2012). The ramp effect (Bloomfield+2007; Jess+2013) allows frequencies below 5.3 mHz to propagate when the field is sufficiently inclined. Figure 42 (Samanta+2016, SST/CRISP) maps 3-, 5-, 7-min period distributions at five heights through Hα. Jafarzadeh+2021 found lack of 3-min global oscillations in ALMA mm-wave brightness temperatures of magnetic chromospheric regions.

전역 p-mode(3 mHz)는 광구를 지배하고, 3분 진동(5 mHz)은 흑점 채층을 지배한다. Kayshap+2018은 IRIS 다중선(Mn I 2801, Mg II k, C II)으로 1.6–4.0분 주기의 상승 p-mode와 4.5분 초과 주기의 하강 전파를 식별했다. 내부 중력파는 ~2 kW/m² 에너지 유량(Vigeesh+2021)을 운반하며 f<2 mHz이다. 자기 네트워크 주변 파워 halo와 고채층 3분 magnetic shadow는 각각 자기 캐노피에서의 fast파 반사(Khomenko & Calvo 2012)와 모드 변환(Moretti+2007)으로 해석된다. Ramp effect(Bloomfield+2007; Jess+2013)는 자기장 경사가 클수록 5.3 mHz 미만 저주파의 상승 전파를 허용한다.

**Samanta+2016 multi-height cascade.** Figure 42 of the review shows a five-layer SST/CRISP Hα "period distribution map" sampling the photosphere (wide-band filtergram) through the upper chromosphere (Hα core). The dominant period shifts from ~5 min in the low photosphere to ~3 min in the mid chromosphere and back to 5–7 min at the very top. The 3-min band is visibly absent in the uppermost Hα core layer — a "green-colour drop-out" — attributed either to dense long fibrils acting as umbrellas (Rutten 2017) obscuring Doppler signals from below, or to genuine mode conversion/dissipation eliminating the 3-min band above the canopy. The ambiguity illustrates why multi-wavelength, multi-technique cross-checks are mandatory.

Samanta+2016의 다층 관측(그림 42)은 SST/CRISP Hα로 광구·저-·중-·고 채층의 다섯 층을 샘플링한 지배 주기 지도이다. 주기는 저광구 ~5분 → 중 채층 ~3분 → 최상층 5–7분으로 변화한다. 최상 Hα core 층에서 3분 대역이 "녹색 색상"으로 결여되는 현상은 (i) 고밀도 장 fibril이 "우산" 역할로 아래 Doppler 신호를 가리거나(Rutten 2017), (ii) 캐노피 위에서의 모드 변환/소멸로 3분 대역이 제거되는 것으로 해석된다 — 이 모호성은 다파장·다기법 교차 검증의 필요성을 보여준다.

### Part IV: Large-Scale Magnetic Structures (Sect. 3.2, pp. 74–88) / 대규모 자기 구조

**3.2.1 Magnetoacoustic waves in large-scale structures.** Sunspots and pores show 5-min (~3 mHz) power at photosphere and 3-min (~5 mHz) at chromosphere (Centeno+2006a,b). Atmospheric stratification changes c_s, v_A by orders of magnitude across the atmosphere — Fig. 36 (Bruls & Solanki 1993 NC5 model + Vernazza VAL-A atmosphere) plots Alfvén, fast, kink, slow, sound, tube speeds versus height from 200 to 1500 km. In the ≈800 km transition region, mode conversion becomes geometrically and energetically important.

**Mode conversion.** At β≈1 (equivalently c_s ≈ v_A), mode labels break down. Cally (2001, 2007) quantify transmission and conversion. The fast-to-slow *transmission* coefficient is
$$T = \exp\!\left[-\pi k h_s \sin^2\alpha\right] \qquad (20)$$
where k is the wavenumber, h_s is the thickness of the conversion layer, and α is the attack angle between k and B. Energy conservation gives T+|C|=1 with C the fast-to-fast conversion coefficient. Larger frequency → larger T (more transmitted as slow). Schunker & Cally (2006) demonstrated that the ramp effect plus mode conversion produces an acoustic flux strongly dependent on field geometry.

**Acoustic resonator.** Jess+2020 using IBIS/DST showed that the sunspot atmosphere acts as an acoustic resonator, producing enhanced 3-min (~5.8 mHz) power via a cavity bounded by the photosphere and transition region temperature gradients. Felipe+2020 independently confirmed. Figure 53 (Jess+2020) shows vertical stack of IBIS Ca II 8542 Å narrowband images with spectral energy peaking at ≈5.8 mHz.

흑점의 광구 파워는 5분(~3 mHz), 채층은 3분(~5 mHz)이 지배한다. 고도에 따른 c_s·v_A의 수 차수 변화가 그림 36(Bruls & Solanki 1993 NC5 + VAL-A) 플롯으로 요약된다. β≈1에서 모드 변환이 일어나며 투과 계수 T는 식 (20)로 기술된다. Jess+2020이 IBIS/DST 관측으로 흑점 대기가 음향 공명공동 역할을 한다는 증거(≈5.8 mHz 파워 피크)를 제시했고 Felipe+2020이 독립 확인했다.

**Umbral flashes and running penumbral waves.** UFs are chromospheric brightenings caused by upward slow MHD waves steepening into shocks as they traverse multiple density scale heights (Beckers & Tallant 1969). Figure 51 (Grant+2018) cartoon shows three regions — umbra centre (shocks), mode-conversion zones (left: resonantly amplified Alfvén; right: sinusoidal magnetoacoustic → elliptical Alfvén). RPWs propagate outward across penumbra at 10–40 km/s apparent horizontal phase speeds, interpreted as slow magnetoacoustic waves propagating along magnetic field lines inclined to horizontal (Bloomfield+2007; Tziotziou+2006). Alfvén-driven shocks (Grant+2018) efficiently contribute to chromospheric energy budget via resonant amplification.

Umbral flash는 상승하는 slow MHD 파가 밀도 층 스케일 여러 개를 통과하며 충격파로 급경사화되어 채층에서 밝기 증가로 관측된다(Beckers & Tallant 1969). 그림 51(Grant+2018)은 흑점 대기 단면도로 umbra 중심(충격파)·좌우 모드 변환 영역(공명 증폭된 Alfvén·타원 Alfvén)을 보여준다. Running penumbral wave는 penumbra를 10–40 km/s 수평 위상속도로 바깥으로 퍼지며, 경사 자기력선을 따라 전파하는 slow 모드로 해석된다. Grant+2018의 Alfvén 구동 충격파는 공명 증폭을 통해 채층 에너지 예산에 효과적으로 기여한다.

**Quantitative umbral flash parameters.** Observed UF periods cluster at 140–200 s (De la Cruz Rodríguez+2013; Houston+2018). Shock velocity amplitudes reach 6–8 km/s in Ca II 8542 Å Doppler diagnostics. Anan+2019 measured total shock heating rates using IFU spectroscopy of integral field slit units, finding ~10⁶ erg cm⁻² s⁻¹ = 1 kW/m² — a substantial but insufficient fraction of the ~20 kW/m² chromospheric radiative losses. The gap motivates Alfvén-mediated heating pathways (Grant+2018). Higher time-resolution DKIST ViSP observations should resolve the shock-steepening front in finer detail, potentially reconciling energy budgets.

**Running penumbral wave nature.** Apparent horizontal phase speeds of 10–40 km/s in the chromosphere are consistent with slow magnetoacoustic waves propagating along field lines inclined ≳70° from vertical. The "running" appearance across penumbra is a projection effect — waves are propagating along field lines that fan outward at increasing inclinations, and the crossing of field-line footprints with the observed height surface produces the apparent horizontal motion. Bloomfield+2007 explicitly showed the apparent speed is frequency-dependent and decreases with distance from the umbra — a signature of the ramp effect varying with local field inclination.

Umbral flash의 정량적 특성: 주기는 140–200 s에 집중되고(De la Cruz Rodríguez+2013; Houston+2018), Ca II 8542 Å Doppler 충격파 속도 진폭은 6–8 km/s이다. Anan+2019는 IFU 분광법으로 충격 가열율 ~10⁶ erg cm⁻² s⁻¹ ≈ 1 kW/m²를 측정했는데, 이는 채층 복사 손실 ~20 kW/m²의 일부에 그쳐 Alfvén 매개 가열 경로(Grant+2018)의 필요성을 시사한다. Running penumbral wave는 70° 이상 경사된 자기력선을 따라 전파하는 slow magnetoacoustic 파의 투영 효과이며, Bloomfield+2007은 주파수와 거리에 따라 겉보기 속도가 변하는 ramp effect 신호를 명시했다.

### Part V: Eigenmodes of Large-Scale Structures (Sect. 3.3, pp. 88–97) / 대규모 구조의 고유모드

Jess+2017 applied 3D k-ω filtering to HARDcam Hα line-core intensities of a near-circular sunspot, isolating coherent horizontal ridges interpreted as m=1 slow body modes rotating within the chromospheric umbra. Keys+2018 analysed ROSA data of multiple pores and identified sausage (m=0) body versus surface modes via their radial power distributions (Fig. 56): body modes peak at tube centre, surface modes peak at tube boundary. Estimated energy fluxes: surface 22±10 kW/m², body 11±5 kW/m² (Moreels framework). Stangalini+2018 used phase-lag between circular polarisation (CP) and intensity to identify propagating magnetic fluctuations along the umbra-penumbra boundary — a spatially coherent surface mode of the sunspot flux tube not consistent with pure opacity effects.

Jess+2017는 HARDcam Hα 세기에 3D k-ω 필터링을 적용해 거의 원형인 흑점에서 채층 umbra를 회전하는 m=1 slow body mode를 분리했다. Keys+2018은 ROSA로 포어들의 sausage(m=0) body/surface 모드를 방사 파워 분포로 구별했다 — body는 관 중심에서, surface는 경계에서 최대(그림 56). 에너지 유량 추정치: surface 22±10 kW/m², body 11±5 kW/m². Stangalini+2018은 원편광(CP)과 세기 사이의 phase-lag 분석으로 umbra-penumbra 경계의 전파하는 자기 요동을 식별했다.

Albidah+2022 and Stangalini+2022 used POD/DMD on sunspot umbrae to identify multiple slow body modes at photospheric and chromospheric heights. For elliptical umbrae (ε=0.58), the exact umbral-shape model reproduces the observed even-m=1 kink overtone far better than either the circular or elliptical cylinder models. Stangalini+2022 (Figs. 59–60) reconstructed the observed Doppler signal from the first 50 eigenfunctions with nine dominant modes — a reconstruction impossible with idealised cylinder geometries.

Albidah+2022, Stangalini+2022는 POD/DMD를 흑점 umbra에 적용해 광구·채층에서 다수의 slow body mode를 식별했다. 이심률 ε=0.58의 타원 umbra에서는 실제 umbra 형상 모델이 원형/타원 실린더 모델보다 관측된 짝수 m=1 kink 오버톤을 더 잘 재현했다. Stangalini+2022는 처음 50개 고유함수(9개 지배 모드)로 관측된 Doppler 신호를 재구성했다.

### Part VI: Small-Scale Magnetic Structures (Sect. 3.4, pp. 97–117) / 소규모 자기 구조

**3.4.1 Excitation, propagation, dissipation.** MBPs and internetwork magnetic elements occupy intergranular lanes with β close to or exceeding 1 (Keys+2020). Wave excitation mechanisms: (1) granular buffeting → kink modes (Spruit 1976), (2) compression by converging granules → sausage modes, (3) rotating flows → torsional Alfvén (Spruit 1982). Jafarzadeh+2013 identified super-sonic pulse-like kicks in Sunrise MBPs with energy fluxes carried upward as kink waves. Bate+2022 reported high-frequency kink waves in chromospheric spicules with 1.8 × 10⁵ J/m²/s energy flux.

Typical periods across small-scale structures: MBP kink oscillations 126–700 s (Stangalini+2013); fibril transverse oscillations 30–500 s with amplitudes 1–2 km/s (Jafarzadeh+2017a,b); spicule swaying periods 25–60 s (De Pontieu+2007; McIntosh+2011). Spicule Alfvénic-wave energy flux estimates range 100–1000 W/m², sufficient to heat the quiet-Sun chromosphere/corona at the order-of-magnitude level.

MBP와 internetwork 자기 요소는 β≈1 근처 입계선에 분포한다(Keys+2020). 파동 여기 기작은 (1) 입자적 granular buffeting → kink, (2) 수렴하는 입자 압축 → sausage, (3) 회전 흐름 → torsional Alfvén이다(Spruit 1982). Jafarzadeh+2013은 Sunrise로 초음속 펄스형 킥을 검출했고, Bate+2022는 채층 스피큘에서 1.8×10⁵ J/m²/s의 고주파 kink 파 에너지 유량을 보고했다. 전형적 주기: MBP kink 126–700 s, fibril 30–500 s, 스피큘 swaying 25–60 s.

**3.4.2 Magnetic-field perturbations in small-scale structures.** High-precision spectropolarimetric inversions (SPINOR, NICOLE, STiC) enable direct detection of oscillating magnetic field components, disentangling true magnetic waves from opacity-induced "fake" magnetic signatures. Fujimura & Tsuneta (2009) phase-relation framework (v_r vs B_r for kink, v_z vs B_z for sausage) underlies most identifications.

**DKIST first-light observations.** VBI white-light bursts resolve 20 km features; ViSP/DL-NIRSP enable simultaneous multi-line spectropolarimetry at 4-m diffraction limit; Cryo-NIRSP reaches the coronal He I 10830 Å line. FRANCIS (NLST prototype at DST, 2022 commissioning) uses a 20×20 fibre array in a 1.10×1.13 mm ferrule — approximately 35 fibres on umbra, 170 on penumbra, 195 on quiet Sun per 40×40″ field. Figure 67 shows the inlet ferrule with 390/393 nm (continuum/Ca II K) sample images.

**Future repositories.** A Level-2 data repository (analogous to JSOC) and the WaLSA (Waves in the Lower Solar Atmosphere) code library at https://www.WaLSA.team/ are endorsed as community standards.

DKIST first-light 관측기기로 VBI(20 km 분해 수준의 백색광 burst), ViSP/DL-NIRSP(4 m 회절한계 분광편광), Cryo-NIRSP(He I 10830 Å 코로나선)가 가동됐다. FRANCIS는 20×20 광섬유 배열(1.10×1.13 mm ferrule)로 umbra 35·penumbra 170·quiet Sun 195개 섬유를 할당한다(그림 67). 커뮤니티 표준으로 Level-2 데이터 저장소와 WaLSA 공개 코드 라이브러리가 권장된다.

**Multi-mode identification challenges.** In small-scale structures, it is often impossible to resolve the flux-tube cross-section, so eigenmode classification must rely on indirect diagnostics: (a) the ratio of area to intensity oscillations (sausage mode signature — anti-phase between area and intensity for compressive mode; Edwin & Roberts 1983; Moreels+2013), (b) width–intensity phase relations for sausage vs kink discrimination, (c) Doppler-velocity × intensity phase lag analysis for slow magnetoacoustic identification. Wave-mode mixing can further obscure the picture: Morton+2012 demonstrated simultaneous compressible and incompressible modes in the same magnetic feature. High-frequency (>50 mHz) waves in MBPs, if detected reliably, carry disproportionate energy flux per period and may bridge the heating gap (Jafarzadeh+2013; Bate+2022). However, high-frequency power is often contaminated by photon noise; confidence level estimation using AR-1 red-noise backgrounds (Torrence & Compo 1998) is essential.

소규모 구조에서는 자속관 단면을 분해하기 어려워 간접 진단에 의존해야 한다 — (a) 면적·세기 진동 비율(sausage 특성: 압축 모드에서 면적과 세기가 반위상; Edwin & Roberts 1983; Moreels+2013), (b) 폭-세기 위상 관계로 sausage/kink 구별, (c) Doppler 속도×세기 phase-lag로 slow magnetoacoustic 식별. Morton+2012가 같은 자기 구조에 압축·비압축 모드가 공존함을 보인 바, 모드 혼합이 해석을 더 복잡하게 만든다. MBP의 >50 mHz 고주파 파동은 주기당 에너지 유량이 크나 광자 잡음에 오염되기 쉬워 AR-1 red-noise 대비 신뢰 수준 평가(Torrence & Compo 1998)가 필수이다.

### Part VII: Future Directions & Conclusions (Sect. 4–5, pp. 117–130) / 향후 방향과 결론

Sunrise-III (2024 flight) carries SUSI (near-UV 300–430 nm spectropolarimetry) and SCIP (near-IR 765–855 nm). Solar Orbiter/PHI (HRT) resolves ~200 km at perihelion 0.28 AU. The charge-caching camera concept (C³Po; Keller 2004) reaches 10⁻⁵ polarimetric precision at kHz rates. The Fast SpectroPolarimeter (FSP, Iglesias+2016) with ferroelectric liquid crystal modulator achieves 400 fps at 5 e⁻ read noise — 4× better than CRISP/IBIS. The conclusions explicitly encourage future researchers to download and extend the WaLSA codebase rather than re-invent analyses from scratch.

Sunrise-III(2024년 비행)는 SUSI(근자외 300–430 nm)와 SCIP(근적외 765–855 nm)를 탑재한다. Solar Orbiter/PHI는 근일점 0.28 AU에서 ~200 km 분해한다. Keller 2004의 C³Po 개념은 kHz 속도로 10⁻⁵ 편광 정밀도를 달성한다. Iglesias+2016의 FSP는 강유전성 액정 변조기로 400 fps·5 e⁻ 읽기잡음에 도달하여 CRISP/IBIS 대비 4배 개선됐다. 결론부는 WaLSA 코드베이스 활용과 확장을 장려한다.

---

## 3. Key Takeaways / 핵심 시사점

1. **Plasma-β is the single most important parameter for wave physics in the lower atmosphere.** It varies by orders of magnitude in ~1500 km, controlling which mode labels apply and where mode conversion occurs. — 하층 대기에서 β는 가장 중요한 파라미터이며, 1500 km 고도 내에서 수 차수로 변하여 모드 명명과 변환 지점을 결정한다.

2. **The 5-min p-mode is trapped, the 3-min chromospheric oscillation is propagating.** The distinction arises from the acoustic cutoff ω_c=c_s/(2H): ~5 mHz at photosphere, ~3–4 mHz at chromosphere. The 3-min oscillation dominates umbrae because its frequency exceeds the local cutoff and it couples to the resonant cavity between photosphere and transition region. — 5분 p-mode는 갇히고 3분 채층 진동은 전파한다. 차단 주파수 ω_c=c_s/(2H)가 광구 5 mHz·채층 3–4 mHz로 낮아지며, 3분 진동은 흑점 공명공동 증폭으로 umbra에서 지배적이다.

3. **Mode conversion at β=1 determines upward energy throughput.** Transmission coefficient T=exp(−πkh_s sin²α) is maximised for small attack angles α. The ramp effect (inclined fields) effectively lowers the cutoff and opens low-frequency windows. — β=1 층에서의 모드 변환이 상승 에너지 처리량을 결정한다. 투과 계수 T=exp(−πkh_s sin²α)는 충돌각 α가 작을수록 최대화되고, ramp effect로 저주파 창이 열린다.

4. **The magnetic cylinder classification (sausage/kink/fluting; body/surface) is a useful first approximation but breaks down for real umbrae.** Observed eccentricities ε=0.58–0.76 cause fluting modes to coalesce; exact-shape POD/DMD analysis is required. — 실린더 모드 분류는 1차 근사로 유용하나 실제 umbra(ε=0.58–0.76)에서 깨지며, 정확한 형상 기반 POD/DMD가 필요하다.

5. **Umbral flashes are non-linear slow-mode shocks, not acoustic oscillations.** Grant+2018 showed Alfvén-driven resonant amplification contributes substantially to chromospheric heating of sunspots. — Umbral flash는 음파 진동이 아닌 비선형 slow 모드 충격파이며, Alfvén 구동 공명 증폭이 흑점 채층 가열에 크게 기여한다.

6. **Spatial resolution is a first-order scientific variable, not a nuisance.** The 1-m SST sees ~2× the acoustic energy flux of the 0.7-m VTT — not because physics differs, but because under-resolution mixes signals. DKIST's 4-m aperture is therefore a *qualitative* upgrade. — 공간 분해능은 일차적 과학 변수이다. 1 m SST는 0.7 m VTT의 2배 음향 에너지 유량을 관측하며, DKIST 4 m는 질적 도약이다.

7. **Methodology pluralism is mandatory.** Single-technique wave studies are no longer defensible: Fourier + wavelet + POD/DMD + phase-lag + B-ω must be cross-checked to disentangle opacity, mode conversion, and multi-mode superposition. — 단일 기법 파동 연구는 더 이상 방어 가능하지 않다 — Fourier·wavelet·POD/DMD·phase-lag·B-ω의 교차 검증이 필수이다.

8. **Open-source, community-maintained code (WaLSA) and Level-2 data repositories are the scientific infrastructure for the DKIST decade.** The review explicitly positions analysis-software sharing as a first-class research output. — WaLSA 공개 코드와 Level-2 데이터 저장소가 DKIST 시대의 과학 인프라이며, 분석 소프트웨어 공유는 일급 연구 산출물로 규정된다.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Acoustic cutoff frequency / 음향 차단 주파수
For an isothermal atmosphere with scale height $H = k_B T/(\bar{m}g)$ and sound speed $c_s = \sqrt{\gamma k_B T/\bar{m}}$:
$$\omega_c = \frac{c_s}{2H} = \frac{\gamma g}{2c_s}.$$
Solar photosphere ($T=5778\,\text{K}$, $\gamma=5/3$, $g=274\,\text{m/s}^2$): $c_s \approx 8.5\,\text{km/s}$, $H \approx 150\,\text{km}$, $\omega_c/(2\pi) \approx 5\,\text{mHz}$. Chromosphere ($T=10\,000\,\text{K}$): $\omega_c/(2\pi)$ falls to 3–4 mHz.

### 4.2 Isothermal dispersion relation / 등온 분산 관계
For vertical propagation in a stratified isothermal atmosphere:
$$\omega^2 = c_s^2 k_z^2 + \omega_c^2.$$
Below $\omega_c$: $k_z^2<0$ → evanescent (trapped). Above $\omega_c$: $k_z^2>0$ → propagating. The 5-min p-mode (3 mHz) is below cutoff → trapped; the 3-min chromospheric oscillation (5.5 mHz) is above → propagating.

### 4.3 Plasma-β (two equivalent forms) / β 두 형태
$$\beta = \frac{2\mu_0 p_0}{B_0^2} \quad \text{(SI)} \qquad \beta = \frac{8\pi n_H k_B T}{B_0^2} \quad \text{(cgs)}.$$
Sunspot umbra ($B=3000\,\text{G}$, $T=4000\,\text{K}$, $n_H=10^{17}\,\text{cm}^{-3}$): $\beta \approx 10^{-2}$. Quiet-Sun intergranular ($B=100\,\text{G}$, $T=6000\,\text{K}$, $n_H=10^{17}\,\text{cm}^{-3}$): $\beta \approx 10$.

### 4.4 MHD wave speeds and dispersion / MHD 파속
Alfvén: $v_A = B/\sqrt{\mu_0 \rho}$. Sound: $c_s = \sqrt{\gamma p/\rho}$. Alfvén dispersion:
$$\omega = k v_A \cos\theta.$$
Slow/fast biquadratic:
$$\omega^4 - \omega^2 k^2 (c_s^2+v_A^2) + k^4 c_s^2 v_A^2 \cos^2\theta = 0.$$
Solutions:
$$\omega_\pm^2 = \frac{k^2}{2}\!\left[(c_s^2+v_A^2) \pm \sqrt{(c_s^2+v_A^2)^2 - 4 c_s^2 v_A^2 \cos^2\theta}\right].$$
Upper sign → fast; lower sign → slow. At $\theta=0$: $\omega_+=k\max(c_s,v_A)$, $\omega_-=k\min(c_s,v_A)$. At $\theta=\pi/2$: $\omega_+=k\sqrt{c_s^2+v_A^2}$ (fast), $\omega_-=0$ (slow forbidden).

### 4.5 Tube speed / 관속도
$$c_T = \frac{c_0 v_A}{\sqrt{c_0^2 + v_A^2}}.$$
Lower bound of slow trapped band in a magnetic cylinder under photospheric conditions $v_A > c_e > c_0 > v_{Ae}$.

### 4.6 Mode transmission at β=1 / 모드 투과 계수
$$T = \exp\!\left(-\pi k h_s \sin^2\alpha\right), \qquad T + |C| = 1.$$
α = attack angle between **k** and **B**; h_s = thickness of conversion layer; k = wavenumber; C = fast-to-fast conversion coefficient.

### 4.7 Wave energy flux / 파동 에너지 유량
Acoustic: $F = \rho \, \langle v^2 \rangle \, c_s$. Alfvén: $F = \rho \, \langle v^2 \rangle \, v_A$. Kink: $F = \rho \, \langle v^2 \rangle \, c_k$ where $c_k = \sqrt{(\rho_i v_{A,i}^2 + \rho_e v_{A,e}^2)/(\rho_i + \rho_e)}$.

### 4.8 Worked example — chromospheric Alfvén flux / 채층 Alfvén 유량
$\rho = 10^{-10}\,\text{kg/m}^3$, $v=1\,\text{km/s}=10^3\,\text{m/s}$, $v_A=100\,\text{km/s}=10^5\,\text{m/s}$:
$$F = 10^{-10} \times (10^3)^2 \times 10^5 = 10^{4}\,\text{W/m}^2 = 10\,\text{kW/m}^2,$$
comparable to the observed chromospheric radiative losses.

### 4.9 Worked example — p-mode wave period / p-mode 주기 계산
$\omega/(2\pi) = 3\,\text{mHz} \Rightarrow$ period $= 1/(3\times 10^{-3}) \approx 333\,\text{s} \approx 5.5\,\text{min}$. For $c_s = 8.5\,\text{km/s}$ and $\omega^2 > \omega_c^2$ only if $c_s^2 k_z^2 > \omega_c^2 - \omega^2$. Since $\omega < \omega_c$, $k_z^2 < 0$ → evanescent → trapped p-mode.

### 4.10 Worked example — umbral 3-min chromospheric wave / 흑점 3분 채층 파동
Period $P = 180$ s → $f = 1/P \approx 5.56$ mHz. At chromospheric height ($T=10\,000$ K, $H \approx 300$ km for $\bar{m}/m_H = 1.3$):
$$c_s \approx \sqrt{\gamma R T/\mu} \approx 11.7\,\text{km/s}, \quad \omega_c/(2\pi) \approx 3.1\,\text{mHz}.$$
Since 5.56 mHz > 3.1 mHz, the wave propagates upward. For $\omega^2 = c_s^2 k_z^2 + \omega_c^2$:
$$k_z = \sqrt{\omega^2 - \omega_c^2}/c_s.$$
Evaluating: $k_z \approx \sqrt{(2\pi\cdot 5.56)^2 - (2\pi\cdot 3.1)^2}\times 10^{-3}/11.7 \approx 2.4 \times 10^{-6}\,\text{m}^{-1}$, corresponding to a vertical wavelength $\lambda_z = 2\pi/k_z \approx 2600$ km.

### 4.11 Worked example — running penumbral wave phase speed / 페넘브라 파동 위상속도
Observed outward apparent horizontal phase speed 20 km/s at 500 km above photosphere in a sunspot penumbra where magnetic field is inclined at 70° from vertical. The true propagation along the inclined field yields an apparent horizontal speed $v_{h,app} = c_{s,true} / \cos\theta_B$ where $\theta_B$ is field inclination from horizontal. With $\cos(20°) = 0.94$: $c_{s,true} \approx v_{h,app} \times \cos(20°) \approx 19$ km/s, consistent with slow-mode propagation along inclined field. This is the observational signature that running penumbral waves are slow magnetoacoustic waves projected onto the sky plane.

### 4.12 Worked example — mode transmission at β=1 / β=1 모드 투과
At attack angle $\alpha = 30°$ with $k = 2\pi/(500\,\text{km}) = 1.26\times 10^{-5}\,\text{m}^{-1}$ and conversion layer thickness $h_s = 100$ km:
$$T = \exp(-\pi \cdot 1.26\times 10^{-5} \cdot 10^5 \cdot \sin^2 30°) = \exp(-0.99) \approx 0.37.$$
So 37% of fast-wave energy is transmitted as slow mode; 63% converted (reflected/refracted as fast mode with altered character). Smaller α (wave more aligned with B) → larger T. This is why mode conversion efficiency depends critically on the magnetic field geometry at β=1.

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1942 ─ Alfvén — MHD waves predicted (Nature 150, 405)
1948 ─ Schwarzschild — AC heating hypothesis
1960 ─ Leighton — discovery of 5-min photospheric oscillation
1969 ─ Beckers & Tallant — umbral flashes (3-min)
1970 ─ Ulrich — p-modes as resonant cavity modes
1972 ─ Zirin & Stein — running penumbral waves discovered
1975 ─ Deubner — k-ω ridges resolve p-mode spectrum
1981 ─ Spruit — thin flux tube theory
1983 ─ Edwin & Roberts — magnetic cylinder dispersion (foundation of Sect. 2.9.2)
1993 ─ Bruls & Solanki — NC5 flux tube model (basis of Fig. 36)
2005 ─ Nakariakov & Verwichte — waves review (LRSP; Paper #3)
2006 ─ Hinode launch; Khomenko & Calvo sunspot osc. review
2007 ─ De Pontieu — spicule Alfvén waves (Science)
2010 ─ SDO launch (AIA/HMI)
2013 ─ IRIS launch; Jess — ramp effect paper
2015 ─ Jess chromosphere review; Khomenko & Collados sunspots review (Paper #45)
2017 ─ Jess — 3D k-ω identification of m=1 slow body modes in sunspot umbra
2018 ─ Keys — pore body/surface modes; Grant — Alfvén-driven shocks
2019 ─ DKIST first light
2020 ─ Jess — acoustic resonator confirmation; Solar Orbiter launch
2021 ─ Aldhafeeri — elliptical cylinder MHD modes
2022 ─ Stangalini, Albidah — POD/DMD sunspot eigenmodes
2023 ─ Jess+ (this review) — synthesis for DKIST decade
2024+ ─ Sunrise-III, Solar-C, NLST/FRANCIS expected
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Paper #3 — Nakariakov & Verwichte (2005) "Coronal Waves and Oscillations" | Canonical corona-focused wave review; Jess+2023 is its lower-atmosphere counterpart; both cite Edwin & Roberts (1983) cylinder dispersion as foundational. / 코로나 중심 파동 리뷰의 하층 대기 대응본. | Direct thematic complement — together they cover photosphere → corona wave physics. / 광구-코로나 전 구간 파동 물리의 상호 보완. |
| Paper #45 — Khomenko & Collados (2015) "Oscillations and Waves in Sunspots" | Deep review of sunspot oscillation physics; extensively cited in Sect. 3.2 of Jess+2023 (mode conversion, 3-min resonance, magnetic canopy reflection). / 흑점 파동 리뷰, 3.2절에서 광범위 인용. | Sunspot-focused subset of Jess+2023's scope; Jess+2023 adds post-2015 POD/DMD + DKIST-era observations. / 흑점 한정 하위집합, DKIST 시대 관측 추가. |
| Edwin & Roberts (1983) "Wave Propagation in a Magnetic Cylinder" | Foundational theory for Sect. 2.9.2; defines slow/fast bands, tube speed c_T. / 실린더 이론의 토대, slow/fast 대역과 c_T 정의. | All large-/small-scale structure analyses rest on this framework. / 대/소규모 구조 분석의 공통 기반. |
| Schwarzschild (1948) "On Noise from the Sun" | Original AC heating hypothesis. / AC 가열 가설 원전. | Motivates entire wave-heating research programme. / 파동-가열 연구 전반의 동기. |
| Grant+2018 "Alfvén-wave Dissipation in Sunspot Atmospheres" | Key result in Sect. 3.2; Figure 51 cartoon adapted from this work. / 3.2절 핵심 결과, 그림 51 출처. | Observational proof of Alfvén-mediated chromospheric heating in sunspots. / 흑점 채층의 Alfvén 매개 가열 관측 증거. |
| Stangalini+2022, Albidah+2022 POD/DMD studies | Section 3.3's climax; demonstrates irregular umbral shape matters. / 3.3절 정점, 불규칙 umbra 형상의 중요성. | Establishes POD/DMD as standard post-2020 eigenmode analysis tool. / 2020년대 표준 고유모드 분석 기법 확립. |
| Cally (2001, 2007) mode-conversion theory | Eq. (20) transmission coefficient foundation. / 식 (20) 투과 계수의 이론 토대. | Quantifies energy leakage across β=1 layer. / β=1 층 에너지 누설 정량화. |

---

## 4.9b Mode Identification Checklist / 모드 식별 체크리스트

관측에서 특정 파동 모드를 식별하기 위한 진단 체크:

Observational diagnostics to identify a wave mode:

| Mode / 모드 | Doppler | Line width | Intensity | Phase relation |
|---|---|---|---|---|
| Slow (longitudinal) | Oscillates | - | Oscillates with V | In phase |
| Fast (compressive) | Oscillates (perp) | Oscillates | Oscillates | In phase |
| Alfvén (torsional) | - | Oscillates | - | π/2 with v_LOS |
| Kink (transverse) | Oscillates | - | - | - |

## 4.9c Power Spectrum Analysis Methods / 파워 스펙트럼 분석 기법

- **Periodogram**: FFT-based, assumes stationarity
- **Wavelet analysis**: Time-resolved frequency content (Torrence & Compo 1998)
- **Empirical Mode Decomposition (EMD)**: Hilbert-Huang transform, nonstationary
- **POD (Proper Orthogonal Decomposition)**: Spatial eigenmode extraction
- **DMD (Dynamic Mode Decomposition)**: Spatio-temporal mode extraction

DKIST 관측에서는 sub-second cadence + high spatial resolution으로 전통적 FFT보다 wavelet/POD/DMD가 선호된다.

DKIST observations with sub-second cadence + high spatial resolution favor wavelet/POD/DMD over traditional FFT.

## 4.10 Alfvén Wave Energy Budget / Alfvén 파 에너지 수지

채층/코로나 가열에 필요한 에너지 흐름은 약 $10^7$ erg/cm²/s (quiet sun)에서 $10^8$ erg/cm²/s (active region)까지 요구된다. Alfvén 파 에너지 유량:

Energy flux requirements: ~$10^7$ erg/cm²/s (quiet sun) to $10^8$ erg/cm²/s (AR). Alfvén wave flux:

$$
F_A = \rho \langle \delta v^2 \rangle v_A
$$

photospheric kG tube에서 ρ≈10⁻⁷ g/cm³, v_A≈10 km/s, δv≈1 km/s → F_A≈10⁵ erg/cm²/s. 코로나로 올라가면서 ρ 감소로 δv²가 증가하여 더 큰 에너지 유량 가능.

At photospheric kG tubes, ρ≈10⁻⁷ g/cm³, v_A≈10 km/s, δv≈1 km/s → F_A≈10⁵ erg/cm²/s. As ρ drops with height, δv² increases → larger fluxes possible.

## 4.11 Torsional Alfvén Wave Identification / 비틀림 Alfvén 파 식별

DKIST급 관측에서 line width oscillation과 Doppler shift의 위상 차이로 torsional (axisymmetric twist) vs kink (translational) 파 구분 가능:

At DKIST-class resolution, phase relation between line-width oscillation and Doppler shift distinguishes torsional (axisymmetric twist) from kink (translational) modes:

- Torsional: line-width oscillates, Doppler velocity centered, π/2 phase lag
- Kink: Doppler velocity oscillates, line-width constant

## 4.12 Phase-Mixing and Damping / 위상 혼합 감쇠

Inhomogeneous Alfvén speed profile causes phase mixing:
$$
\tau_{PM} \sim \left(\frac{6 \eta k_\perp^2 \omega}{v_A^2 L_A^2}\right)^{-1/3}
$$

비균질 Alfvén 속도로 인한 위상 혼합이 파동을 빠르게 감쇠시킨다. 코로나 루프의 kink oscillation 감쇠의 주요 메커니즘.

Phase mixing from inhomogeneous $v_A$ damps waves rapidly. This is the dominant damping mechanism for coronal loop kink oscillations.

## 4.12b Resonant Absorption Damping / 공명 흡수 감쇠

공명 흡수(resonant absorption)는 kink 모드 감쇠의 주요 메커니즘:
Resonant absorption is a major kink-mode damping mechanism:
$$
\tau_d / P = \frac{2}{\pi}\left(\frac{\zeta + 1}{\zeta - 1}\right)\frac{1}{\epsilon}
$$
where $\zeta = \rho_i/\rho_e$ and $\epsilon = \ell/R$ (boundary thickness). 관측된 $\tau_d/P \sim 2-4$ 값은 $\epsilon \sim 0.3$을 암시하여 얇은 경계 근사를 검증.

Observed $\tau_d/P \sim 2-4$ imply $\epsilon \sim 0.3$, validating thin-boundary approximation.

## 4.12c Chromospheric Heating by Magneto-Acoustic Shocks / 자기-음향 충격파에 의한 채층 가열

Photospheric oscillations propagating upward steepen into shocks at ~1 Mm height:
- Linear regime: $\delta v \ll c_s$
- Nonlinear: $\delta v \sim 0.3-0.5 c_s$ with $\tau_{steep} \sim H/c_s \sim 30$ s
- Shock heating rate: $\Phi_{shock} \sim \rho c_s^3 M^3/L$ where $M$ is Mach number

Umbral flashes ($T_{brightening} \sim 1000-2000 K$)이 shock heating의 직접 증거.

Umbral flashes ($T_{brightening} \sim 1000-2000$ K) are direct evidence of shock heating.

## 4.13 Numerical Examples Summary Table / 수치 예제 요약 표

| Phenomenon / 현상 | Period | Amplitude | Reference |
|---|---|---|---|
| Photospheric p-mode | 5 min (3.3 mHz) | ~300 m/s | Leighton 1962 |
| 3-min chromospheric | 180 s (5.5 mHz) | ~5 km/s | Beckers & Tallant 1969 |
| Umbral flashes | 180 s | 20% intensity | Rouppe van der Voort+2003 |
| RPW | 180-300 s | 20-30 km/s phase | Zirin & Stein 1972 |
| Kink loop oscillation | 3-5 min | ~5-30 km/s | Aschwanden+1999 |
| Torsional Alfvén | ~5 min | ~30 km/s | Jess+2009 |

## 4.14 Observational Cadence Requirements / 관측 캐던스 요구사항

- High-frequency p-mode (8 mHz): need cadence < 20 s
- Alfvén wave in corona (10-100 mHz): cadence < 2-5 s
- Chromospheric shock (3 min period): cadence ~15 s sufficient

DKIST 주요 기기 (VBI, ViSP, DL-NIRSP)는 sub-second cadence를 제공하여 고주파 파동 검출 가능.

DKIST instruments (VBI, ViSP, DL-NIRSP) provide sub-second cadence, enabling high-frequency wave detection.

---

## 7. References / 참고문헌

- Jess, D. B., Jafarzadeh, S., Keys, P. H., Stangalini, M., Verth, G., & Grant, S. D. T., "Waves in the Lower Solar Atmosphere: The Dawn of Next-Generation Solar Telescopes", *Living Reviews in Solar Physics*, 20:1 (2023). DOI: [10.1007/s41116-022-00035-6](https://doi.org/10.1007/s41116-022-00035-6)
- Alfvén, H., "Existence of electromagnetic-hydrodynamic waves", *Nature* 150, 405 (1942).
- Beckers, J. M. & Tallant, P. E., "Chromospheric inhomogeneities in sunspot umbrae", *Solar Phys.* 7, 351 (1969).
- Bruls, J. H. M. J. & Solanki, S. K., "The chromospheric temperature rise in solar magnetic flux tube models", *Astron. Astrophys.* 273, 293 (1993).
- Cally, P. S., "Note on an exact solution for magnetoatmospheric waves", *Astrophys. J.* 548, 473 (2001).
- Cally, P. S., "What to look for in the seismology of solar active regions", *Astron. Nachr.* 328, 286 (2007).
- Deubner, F.-L., "Observations of low wavenumber nonradial eigenmodes of the Sun", *Astron. Astrophys.* 44, 371 (1975).
- Edwin, P. M. & Roberts, B., "Wave propagation in a magnetic cylinder", *Solar Phys.* 88, 179 (1983).
- Grant, S. D. T. et al., "Alfvén-wave dissipation in the solar chromosphere", *Nature Physics* 14, 480 (2018).
- Jess, D. B. et al., "Multiwavelength studies of MHD waves in the solar chromosphere", *Space Sci. Rev.* 190, 103 (2015).
- Keys, P. H. et al., "Photospheric observations of surface and body modes in solar magnetic pores", *Astrophys. J.* 857, 28 (2018).
- Khomenko, E. & Collados, M., "Oscillations and waves in sunspots", *Living Rev. Solar Phys.* 12, 6 (2015).
- Leighton, R. B., Noyes, R. W., & Simon, G. W., "Velocity fields in the solar atmosphere", *Astrophys. J.* 135, 474 (1962).
- Nakariakov, V. M. & Verwichte, E., "Coronal waves and oscillations", *Living Rev. Solar Phys.* 2, 3 (2005).
- Schwarzschild, M., "On noise arising from the solar granulation", *Astrophys. J.* 107, 1 (1948).
- Spruit, H. C., "Propagation speeds and acoustic damping of waves in magnetic flux tubes", *Solar Phys.* 75, 3 (1982).
- Stangalini, M. et al., "Large scale MHD wave eigen-modes in solar umbrae", *Astrophys. J.* 940, 5 (2022).
- Ulrich, R. K., "The five-minute oscillations on the solar surface", *Astrophys. J.* 162, 993 (1970).
- WaLSA team, code repository: https://www.WaLSA.team/
