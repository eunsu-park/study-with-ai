---
title: "Observations of Cool-Star Magnetic Fields"
authors: Ansgar Reiners
year: 2012
journal: "Living Reviews in Solar Physics"
doi: "10.12942/lrsp-2012-1"
topic: Living_Reviews_in_Solar_Physics
tags: [stellar-magnetism, Zeeman-effect, Stokes-parameters, ZDI, LSD, M-dwarfs, dynamo, rotation-activity, Rossby-number, brown-dwarfs]
status: completed
date_started: 2026-04-23
date_completed: 2026-04-23
---

# 28. Observations of Cool-Star Magnetic Fields / 차가운 별의 자기장 관측

---

## 1. Core Contribution / 핵심 기여

**EN**: Reiners (2012) delivers the definitive observational review of magnetic fields in cool stars (spectral type F and later, including M dwarfs, pre-main-sequence stars, brown dwarfs, and some giants). The review performs three interlocking tasks. First, it builds the methodological foundation from the Zeeman effect upward: Landé factors, absorption-line splitting Δλ = 46.67 g λ₀² B, the Stokes I / Q / U / V polarization formalism, the degeneracy between field strength B and filling factor f (only the product Bf is directly observable), equivalent-width and Doppler-imaging methods, Zeeman Doppler Imaging (ZDI) for vector-field reconstruction, and Least Squares Deconvolution (LSD) as the key signal-boosting technique. Second, it compiles essentially all published magnetic-field measurements in cool stars into seven master tables organized by stellar type (sun-like, M dwarf, pre-MS, giants) and observable (Stokes I integrated fields vs. Stokes V longitudinal/ZDI fields), totaling several hundred stars. Third, it synthesizes the physics: the rotation–magnetic-field–activity triad governed by Rossby number Ro = P_rot / τ_conv, saturation at Ro ≲ 0.1 where Bf plateaus near a few kG, the transition at spectral type M3/M4 from partially convective (Sun-like) to fully convective interiors, the unified Christensen et al. (2009) energy-flux dynamo scaling uniting planets and stars, and the puzzle of young T Tauri star fields that may carry a fossil component.

**KR**: Reiners(2012)는 차가운 별(F형 이후, M형 왜성·전주계열성·갈색왜성·일부 거성 포함)의 자기장에 대한 관측적 종합 리뷰의 결정판이다. 본 리뷰는 세 가지 과제를 서로 연결지어 수행한다. 첫째, Zeeman 효과에서 출발하여 방법론적 기초를 구축한다: Landé 인자, 흡수선 분리 Δλ = 46.67 g λ₀² B, Stokes I/Q/U/V 편광 형식, 자기장 세기 B와 채움 인자 f의 축퇴(곱 Bf만이 직접 관측 가능), 등가폭 및 Doppler Imaging 기법, 벡터장 재구성을 위한 Zeeman Doppler Imaging(ZDI), 핵심 신호 증강 기법인 Least Squares Deconvolution(LSD). 둘째, 발표된 거의 모든 차가운 별의 자기장 측정치를 항성 유형(태양형·M 왜성·전주계열성·거성)과 관측량(Stokes I 통합장 vs Stokes V 종방향/ZDI 장)별로 7개의 마스터 표에 정리하여 수백 개 별을 수록한다. 셋째, 물리를 종합한다: Rossby 수 Ro = P_rot / τ_conv로 지배되는 회전–자기장–활동도 삼각 관계, Ro ≲ 0.1에서 Bf가 수 kG에 고정되는 포화, M3/M4에서 부분대류(태양형)에서 완전대류로의 구조 전이, 행성과 별을 잇는 Christensen 외(2009)의 에너지 플럭스 다이나모 스케일링, 그리고 화석(fossil) 성분을 가질 수 있는 어린 T Tauri 별의 수수께끼.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction (§1, pp. 7–8) / 서론

**EN**: The review opens with the fundamental observational difficulty: magnetic fields are invisible, detectable only through their indirect effects on spectra, polarization, or non-thermal emission. The Sun (Figure 1, SOHO/MDI magnetogram) shows that even a "well-understood" star is confusing when reduced to a single disk-integrated flux number: Zeeman diagnostics give ~10 G average unsigned flux, but Hanle effect analysis (Trujillo Bueno et al. 2004) gives values an order of magnitude higher. This illustrates a central theme: different techniques probe different field populations. The review restricts "cool stars" to types later than early F (where rotational broadening becomes prohibitive for field detection), running through sun-like (G–K), M dwarfs, pre-main-sequence stars, and brown dwarfs (M < 0.08 M☉).

**KR**: 리뷰는 근본적 관측적 난제로 시작한다: 자기장은 눈에 보이지 않으며, 스펙트럼·편광·비열적 방출을 통한 간접 효과로만 검출된다. 태양(그림 1, SOHO/MDI 자기도)은 "잘 이해된" 별조차 원반 통합 플럭스 하나로 축약하면 혼란스럽다는 것을 보여준다: Zeeman 진단은 평균 부호 없는 플럭스 ~10 G를 주지만, Hanle 효과 분석(Trujillo Bueno 외 2004)은 한 자릿수 더 큰 값을 준다. 이는 중심 주제를 드러낸다: 각기 다른 기법은 각기 다른 자기장 모집단을 탐침한다. 본 리뷰는 "cool star"를 자전 넓힘이 자기장 검출을 가로막는 초기 F형보다 늦은 별로 한정하며, 태양형(G–K), M 왜성, 전주계열성, 갈색왜성(M < 0.08 M☉)을 다룬다.

### Part II: Methodology — Zeeman Effect (§2.1.1–2.1.3, pp. 9–12) / Zeeman 효과

**EN**: In the presence of a magnetic field B, an atomic level with total angular momentum J splits into (2J+1) sublevels, with energy difference proportional to B·g, where g is the Landé factor (Eq. 1). Dipole transitions obey ΔM = −1, 0, +1, giving three groups: π (ΔM=0, linearly polarized along B) and σ_blue, σ_red (ΔM=±1, shifted to blue/red). The effective Landé g of a transition is Eq. (2). The observable wavelength shift of σ-components is

$$\Delta\lambda = 46.67 \, g \, \lambda_0^2 \, B \quad \text{(mÅ, λ₀ in μm, B in kG)} \quad \text{(Eq. 3)}$$

equivalently as a velocity displacement

$$\Delta v = 1.4 \, \lambda_0 \, g \, B \quad \text{(km/s)} \quad \text{(Eq. 4)}$$

The critical insight is the λ² dependence: at visible wavelengths a 1 kG field produces only ~1 km/s — smaller than typical stellar line widths (a few km/s) and spectrograph resolution (R ~ 100,000 corresponds to 3 km/s). Infrared measurements benefit from the linear λ scaling in velocity. The three Zeeman components have distinct polarization: π is always linearly polarized (invisible when the line of sight is parallel to B), while σ-components are circularly polarized if B is longitudinal, and linearly polarized if B is transverse. The Stokes vectors (Stokes 1852) describe the polarization state: I = total intensity, Q and U = two orthogonal linear polarizations, V = circular polarization.

**KR**: 자기장 B가 있을 때, 총 각운동량 J인 원자 준위는 (2J+1)개의 부준위로 분리되며, 에너지 차이는 B·g에 비례한다(g는 Landé 인자, 식 1). 쌍극자 전이는 ΔM = −1, 0, +1을 따르므로 세 그룹이 생긴다: π(ΔM=0, B 방향 선편광)와 σ_blue, σ_red(ΔM=±1, 청/적 이동). 전이의 유효 Landé g는 식 (2). σ 성분의 관측 파장 이동은

$$\Delta\lambda = 46.67 \, g \, \lambda_0^2 \, B \quad \text{(mÅ, λ₀ μm, B kG)}$$

속도 이동으로는

$$\Delta v = 1.4 \, \lambda_0 \, g \, B \quad \text{(km/s)}$$

핵심 통찰은 λ² 의존성이다: 가시광에서 1 kG 자기장은 ~1 km/s만 만들며, 이는 일반 항성 선폭(수 km/s)과 분광기 분해능(R ~ 100,000은 3 km/s에 해당)보다 작다. 적외선 측정은 속도에서 λ에 선형 비례하여 이득이 있다. 세 Zeeman 성분은 구별되는 편광을 가진다: π는 항상 선편광(시선 방향이 B와 평행이면 보이지 않음), σ 성분은 B가 종방향이면 원편광, 횡방향이면 선편광. Stokes 벡터(Stokes 1852)는 편광 상태를 기술한다: I = 총 세기, Q, U = 두 직교 선편광, V = 원편광.

### Part III: Reconstruction from Stokes Vectors (§2.1.4–2.1.5, pp. 14–17) / Stokes 벡터로부터 재구성

**EN**: Figures 3 and 4 present simulations illustrating three idealized field geometries (pure single polarity; two equal opposite-polarity regions; asymmetric opposite polarities) observed in transverse (Fig. 3) and longitudinal (Fig. 4) orientations. Critical lessons:

1. **Stokes I** (integrated light): Zeeman broadening of ~10% of the line; detects total field regardless of sign, but degenerate with other broadening mechanisms.
2. **Stokes V** (circular polarization): Typical signal ~1% of intensity for kG fields; sensitive to *net* longitudinal field — opposite polarities cancel, making Stokes V a *lower limit*.
3. **Stokes Q/U** (linear polarization): Typical signal ~0.1% (factor 10 weaker than V) for kG fields; sensitive to transverse field; opposite orientations at 90° also cancel.
4. **Observational noise floor**: Wade et al. (2000) Ap/Bp observations show circular polarization at ~1–2 × 10⁻². Kochukhov et al. (2011) reach 5 × 10⁻⁵ circular and ~10⁻⁶ linear in active K dwarf HR 1099 — the state-of-the-art precision. Cool-star Stokes V/I ≲ 10⁻⁴ is routinely challenging.

The **field/flux/filling-factor degeneracy** is fundamental (§2.1.5). A star 50% covered at B = 1 kG looks similar in Stokes I to a star 25% covered at B = 2 kG. Papers usually quote the product Bf, with the crucial clarifying note that *Bf is really an average flux density* (not a flux in the strict sense, since f is a relative fraction). A key historical puzzle: Gray (1985) noticed Bf ≈ 500 G appears essentially constant across G/K stars — the "magnetic conservation" conundrum that turned out to reflect systematic methodological biases.

**KR**: 그림 3과 4는 세 가지 이상화된 자기장 기하(단일 극성; 같은 크기 반대 극성 두 영역; 비대칭 반대 극성)를 횡방향(그림 3)과 종방향(그림 4)에서 관측한 시뮬레이션을 제시한다. 핵심 교훈:

1. **Stokes I**(통합광): 선의 ~10% 수준 Zeeman 넓힘; 부호와 무관한 총 자기장을 검출하지만 다른 넓힘 메커니즘과 축퇴.
2. **Stokes V**(원편광): kG 자기장에 대해 전형적 신호는 세기의 ~1%; 알짜 종방향 자기장에 민감 — 반대 극성은 상쇄되므로 Stokes V는 *하한*.
3. **Stokes Q/U**(선편광): kG 자기장에 대해 전형 신호 ~0.1%(V보다 10배 약함); 횡방향 자기장에 민감; 90° 반대 방향도 상쇄.
4. **관측 잡음 하한**: Wade 외(2000) Ap/Bp 관측은 원편광 ~1–2 × 10⁻². Kochukhov 외(2011)는 활성 K 왜성 HR 1099에서 원편광 5 × 10⁻⁵, 선편광 ~10⁻⁶ 도달 — 최신 정밀도. Cool star Stokes V/I ≲ 10⁻⁴는 일상적으로 도전적.

**장·플럭스·채움 인자 축퇴**는 근본적이다(§2.1.5). 50%가 B = 1 kG로 덮인 별은 25%가 B = 2 kG로 덮인 별과 Stokes I에서 유사하게 보인다. 논문들은 보통 Bf 곱을 인용하며, *Bf는 사실 평균 플럭스 밀도*(엄격한 의미의 플럭스가 아님, f는 상대 비율)라는 중요한 주의가 따른다. 주요 역사적 수수께끼: Gray(1985)는 G/K 별에서 Bf ≈ 500 G가 본질적으로 일정해 보임을 지적 — "자기 보존" 수수께끼는 방법론적 체계 편향을 반영한 것으로 판명.

### Part IV: Equivalent Widths, Doppler Imaging, LSD (§2.1.6–2.1.8, pp. 17–23) / 등가폭, Doppler Imaging, LSD

**EN**: **Equivalent widths** (§2.1.6): For a *saturated* spectral line, Zeeman splitting widens the line while keeping the core depth constant, so the equivalent width grows with B. Basri et al. (1992) and Mullan & Bell (1976) exploited this via comparisons of Zeeman-sensitive versus insensitive lines.

**Doppler Imaging** (§2.1.7): For a rotating star, each point on the visible hemisphere has a unique projected Doppler velocity, so wavelength position across a broadened spectral line maps to spatial position across the stellar disk (Vogt & Penrod 1983, Deutsch 1958). **Zeeman Doppler Imaging (ZDI)** (Semel 1989) extends this to polarized light: as the star rotates through different phases, field components invisible at one phase can appear large in Stokes V/Q/U at others. Figure 6 (Kochukhov & Piskunov 2002) shows a reconstruction of two 8-kG radial spots using all four Stokes parameters vs Stokes I + V alone — the latter suffers crosstalk between radial and meridional field components. Kochukhov & Piskunov (2002) report that Stokes I + V alone can underestimate spot area at low latitudes by factors of a few.

**Least Squares Deconvolution** (LSD; §2.1.8, Donati et al. 1997, Semel 1989): Cool-star polarization signals in single lines are below the noise floor of current instruments. LSD combines hundreds to thousands of lines via the assumption that each line is a convolution of a common broadening function with a weighted delta at the rest wavelength. Under the **weak-field approximation** (Unno 1956, Stenflo 1994) — valid when Zeeman splitting ≪ Doppler width — the Stokes V signal for any line obeys

$$V(v) \propto g_i \, B \, \frac{\partial I(v)}{\partial v} \quad \text{(Eq. 5)}$$
$$Q(v) \propto g_i^2 \, B^2 \, \frac{\partial^2 I(v)}{\partial v^2} \quad \text{(Eq. 6)}$$

This reduces many weak signals to a single high-S/N average V profile weighted by each line's Landé factor. Donati & Brown (1997) claim LSD remains useful up to ~5 kG, though formally the approximation breaks down above ~1.2 kG.

**KR**: **등가폭**(§2.1.6): *포화된* 흡수선에 대해, Zeeman 분리는 코어 깊이를 유지하면서 선을 넓히므로 등가폭이 B와 함께 증가한다. Basri 외(1992)와 Mullan & Bell(1976)은 Zeeman에 민감한 선과 둔감한 선을 비교하여 이를 활용했다.

**Doppler Imaging**(§2.1.7): 자전하는 별에서 가시 반구의 각 점은 고유한 투영 Doppler 속도를 가지므로, 넓어진 스펙트럼 선에서의 파장 위치는 항성 원반의 공간 위치에 대응한다(Vogt & Penrod 1983, Deutsch 1958). **Zeeman Doppler Imaging(ZDI)**(Semel 1989)은 이를 편광으로 확장한다: 별이 여러 위상을 자전하면서, 한 위상에서 보이지 않던 자기장 성분이 다른 위상에서 Stokes V/Q/U에 크게 나타날 수 있다. 그림 6(Kochukhov & Piskunov 2002)은 8 kG 반경 방향 두 반점을 네 Stokes 전부 vs Stokes I+V만으로 복원한 결과를 비교 — 후자는 반경·자오선 성분 간 누화(crosstalk)가 발생. Kochukhov & Piskunov(2002)는 Stokes I+V만으로는 저위도에서 반점 면적이 수배 과소평가된다고 보고.

**Least Squares Deconvolution**(LSD; §2.1.8, Donati 외 1997, Semel 1989): Cool star의 단일 선 편광 신호는 현재 장비의 잡음 하한 아래다. LSD는 각 선이 공통 넓힘 함수와 정지 파장의 가중 델타 함수의 콘볼루션이라는 가정으로 수백~수천 개 선을 결합한다. **약장 근사(weak-field approximation)**(Unno 1956, Stenflo 1994) — Zeeman 분리가 Doppler 폭보다 훨씬 작을 때 유효 — 하에서 임의의 선에 대한 Stokes V 신호는

$$V(v) \propto g_i \, B \, \frac{\partial I(v)}{\partial v}$$
$$Q(v) \propto g_i^2 \, B^2 \, \frac{\partial^2 I(v)}{\partial v^2}$$

이는 많은 약한 신호를 각 선의 Landé 인자로 가중한 단일 고-S/N 평균 V 프로파일로 축약한다. Donati & Brown(1997)은 LSD가 ~5 kG까지 유용하다고 주장하지만, 공식적으로는 ~1.2 kG 이상에서 근사가 무너진다.

### Part V: Broad-Band Polarization and Indirect Diagnostics (§2.2–2.3, pp. 23–26) / 광대역 편광·간접 진단

**EN**: **Broad-band methods**: Leroy (1962), Koch & Pfeiffer (1976), Mullan & Bell (1976) proposed broad-band linear polarization from saturation differences between π and σ components. Huovelin & Saar (1991) modeled this including Rayleigh/Thomson scattering, concluding scattering typically dominates. Bagnulo et al. (2002) use the weak-field slope of V vs. dI/dλ to get field strengths in hot stars; for HD 94660 this yields B ≈ 2 kG (Figure 9).

**Indirect diagnostics** (§2.3): Indirect magnetic indicators include chromospheric Ca II H&K, Hα emission, coronal X-rays, UV C IV, and radio emission. The Schrijver et al. (1989) Ca II relation gives (Eq. 7)

$$\frac{I_C - 0.13}{I_W} = 0.008 \, \langle fB \rangle^{0.6}$$

Schrijver (1990) finds F(C IV) ∝ ⟨fB⟩^0.7 (Eq. 8). The Pevtsov et al. (2003) X-ray–magnetic-flux relation (Eq. 9, Figure 11) spans ten orders of magnitude:

$$L_X \propto \Phi^{1.15}$$

holding from quiet Sun through active stars and T Tauri stars. The radio gyrofrequency (Eq. 11) is ν_c = eB / (2π m_e c) ≈ 2.8 × 10⁶ B (Hz, B in Gauss); detection of electron cyclotron maser emission at 8.5 GHz (Hallinan et al. 2008) implies B ≥ 3 kG on the emitting brown dwarfs.

**KR**: **광대역 방법**: Leroy(1962), Koch & Pfeiffer(1976), Mullan & Bell(1976)은 π와 σ 성분의 포화 차이에서 오는 광대역 선편광을 제안. Huovelin & Saar(1991)은 Rayleigh/Thomson 산란을 포함하여 모형화하며 일반적으로 산란이 우세하다고 결론. Bagnulo 외(2002)는 V vs dI/dλ의 약장 기울기로 뜨거운 별의 자기장 세기를 얻으며, HD 94660에서는 B ≈ 2 kG(그림 9).

**간접 진단**(§2.3): 간접 자기 지표로는 채층 Ca II H&K, Hα 방출, 코로나 X-선, UV C IV, 전파 방출이 있다. Schrijver 외(1989) Ca II 관계는 식 (7):

$$\frac{I_C - 0.13}{I_W} = 0.008 \, \langle fB \rangle^{0.6}$$

Schrijver(1990)은 F(C IV) ∝ ⟨fB⟩^0.7 (식 8). Pevtsov 외(2003) X-선–자기 플럭스 관계(식 9, 그림 11)는 10자리수에 걸친다:

$$L_X \propto \Phi^{1.15}$$

조용한 태양부터 활성 별, T Tauri 별까지 모두 성립. 전파 gyrofrequency(식 11)는 ν_c = eB / (2π m_e c) ≈ 2.8 × 10⁶ B (Hz, B는 Gauss); Hallinan 외(2008)의 8.5 GHz 전자 사이클로트론 메이저 방출 검출은 방출 갈색왜성에서 B ≥ 3 kG를 의미.

### Part VI: Stokes I Measurements in Cool Stars (§3.1, pp. 27–40) / 차가운 별의 Stokes I 측정

**EN**: **Sun-like stars** (§3.1.1): Robinson (1980) introduced the Fourier-transform comparison of magnetically sensitive/insensitive lines. Saar (1988) added proper radiative transfer and line-blend corrections. Basri et al. (1990) introduced a two-component atmospheric treatment. Valenti et al. (1995) were first to use high-R (103,000) IR spectra at 1.56 μm and 2.22 μm for the K2 dwarf ε Eri, finding Bf = 130 G — dramatically lower than earlier optical-based estimates of ~800 G. This famously demonstrated that optical Zeeman analysis in sun-like stars was biased upward. Anderson et al. (2010) re-examined 59 Vir at R = 220,000, SNR 400 — their two-component fits cannot exclude B = 0 (upper limit 300 G). Figure 13 shows that starspot temperature spots can mimic Zeeman broadening at visible wavelengths, driving the measured Bf. Table 1 (36 stars) collects the "best" sun-like star measurements; IR data exist for only 6 K dwarfs.

**M-type stars** (§3.1.2): M dwarfs (0.1–0.6 M☉, radii ~ half solar) have narrower intrinsic line widths, slower rotation at given activity level, and cooler temperatures — together making Zeeman detection much easier. Saar & Linsky (1985) achieved the first M-dwarf Zeeman detection in AD Leo (M3.5) at 2.22 μm with Bf = 2.8 kG. Johns-Krull & Valenti (1996, 2000) used Fe I 8468 Å, measuring Bf = 3.9 kG in EV Lac (M3.5, Figure 14). The Fe I line is embedded in a TiO forest, complicating modeling. Reiners & Basri (2006, 2007) developed the FeH (Wing–Ford band, 1 μm) template-interpolation method (Figure 15), yielding Bf = 2.2 kG for Gl 729. Table 2 lists ~80 M dwarfs. Figure 16 plots Bf vs spectral type: mid- to late-M dwarfs routinely reach 2–4 kG. The implication: if 50% of a star with mean field 4 kG is "quiet" photosphere, the other half must host ~8 kG — two to three orders of magnitude stronger than the Sun.

**Pre-main-sequence stars and young brown dwarfs** (§3.1.3): Basri et al. (1992) via equivalent widths gave the first T Tauri detections. Johns-Krull et al. (1999b, 2007) used Ti I 2.22 μm, measuring Bf ≈ 2–4 kG in 14 T Tauri stars (BP Tau has 2.2 kG). Yang & Johns-Krull (2011) added 14 young stars in Orion. Table 3 records ~40 pre-MS stars with Bf typically 1–4 kG. A striking result from Reiners et al. (2009b): *none* of four young accreting brown dwarfs has a detectable Zeeman signal — fields are below the detection threshold that is reached by almost every older M dwarf.

**KR**: **태양형 별**(§3.1.1): Robinson(1980)은 자기 민감·둔감 선의 Fourier 변환 비교를 도입. Saar(1988)는 적절한 복사 전달과 선 혼합 보정을 추가. Basri 외(1990)는 2성분 대기 처리를 도입. Valenti 외(1995)는 K2 왜성 ε Eri에 대해 최초로 고분해능(R = 103,000) 적외선 스펙트럼(1.56 μm, 2.22 μm)을 사용하여 Bf = 130 G를 얻었으며, 이는 이전 가시광 기반 ~800 G보다 극적으로 낮다. 이는 태양형 별의 가시광 Zeeman 분석이 위쪽으로 편향되어 있었음을 유명하게 보여준 사례이다. Anderson 외(2010)는 R = 220,000, SNR 400으로 59 Vir를 재조사 — 2성분 적합은 B = 0을 배제하지 못함(상한 300 G). 그림 13은 starspot 온도 반점이 가시광 Zeeman 넓힘을 흉내낼 수 있음을 보여주며, 측정된 Bf를 주도할 수 있다. 표 1(36개 별)은 "최상" 태양형 별 측정치를 수록; 적외선 데이터는 K 왜성 6개뿐.

**M형 별**(§3.1.2): M 왜성(0.1–0.6 M☉, 반경 ~태양의 절반)은 내재 선폭이 좁고, 주어진 활동도에서 자전이 느리며, 온도가 낮다 — 이 조합이 Zeeman 검출을 훨씬 쉽게 만든다. Saar & Linsky(1985)는 2.22 μm에서 AD Leo(M3.5)의 최초 M 왜성 Zeeman 검출(Bf = 2.8 kG). Johns-Krull & Valenti(1996, 2000)는 Fe I 8468 Å를 사용해 EV Lac(M3.5)에서 Bf = 3.9 kG(그림 14). Fe I 선은 TiO 숲에 묻혀 있어 모형화가 까다롭다. Reiners & Basri(2006, 2007)는 FeH(Wing–Ford 대역, 1 μm) 템플릿 보간법을 개발(그림 15), Gl 729에서 Bf = 2.2 kG를 얻었다. 표 2는 ~80개 M 왜성을 수록. 그림 16은 Bf vs 분광형: 중·후기 M 왜성은 일상적으로 2–4 kG에 이른다. 함의: 평균 4 kG 자기장의 별에서 50%가 "조용한" 광구라면, 다른 반은 ~8 kG — 태양보다 2–3자리 수 더 강하다.

**전주계열성과 어린 갈색왜성**(§3.1.3): Basri 외(1992)는 등가폭으로 최초의 T Tauri 검출. Johns-Krull 외(1999b, 2007)는 Ti I 2.22 μm를 사용, 14개 T Tauri 별에서 Bf ≈ 2–4 kG 측정(BP Tau는 2.2 kG). Yang & Johns-Krull(2011)은 Orion의 14개 어린 별 추가. 표 3은 Bf 1–4 kG를 갖는 ~40개 전주계열성. Reiners 외(2009b)의 충격적 결과: 네 개의 강착 중인 어린 갈색왜성 *어느 것도* 검출 가능한 Zeeman 신호가 없음 — 장이 거의 모든 더 나이 든 M 왜성이 도달하는 검출 한계 아래에 있다.

### Part VII: Stokes V Measurements and ZDI (§3.2, pp. 41–44) / Stokes V 측정·ZDI

**EN**: **Dwarfs and subgiants** (§3.2.1, Table 5, ~50 stars): First successful longitudinal-field detection in a low-mass star came from RS CVn binaries (Donati et al. 1990, 1992 on HR 1099). LSD then opened Stokes V measurements in many stars. For F–K sun-like stars, average ⟨B⟩ from ZDI is typically tens of Gauss, with local field strengths in Doppler maps up to several hundred Gauss. For M dwarfs, ⟨B⟩ from Stokes V reaches up to ~1.5 kG (Morin et al. 2008, 2010) — significantly larger than in hotter sun-like stars observed with Stokes V, but still a small fraction of Stokes I field values. Kochukhov et al. (2011) achieved the first detection of linear polarization (Q, U) in sun-like stars on HR 1099: circular polarization 5 × 10⁻⁵, linear 10 × weaker.

**Giants** (§3.2.2, Table 6): Active giants like FK Com show ZDI maps with field strengths up to several hundred Gauss; most giants have ⟨B⟩ ≲ 1 G (Arcturus, Pollux, Betelgeuse, EK Boo).

**Young stars** (§3.2.3, Table 7): Pre-MS stars show ZDI fields up to a few hundred Gauss in photospheric lines, but emission lines (He I 5876 Å) formed in the accretion shock show up to 3 kG — a geometric effect where the accretion column reveals an almost uncanceled bundle of fluxtubes.

**KR**: **왜성·준거성**(§3.2.1, 표 5, ~50개 별): 저질량 별에서 최초의 종방향 자기장 검출은 RS CVn 쌍성에서 왔다(Donati 외 1990, 1992, HR 1099). 이후 LSD가 많은 별에서 Stokes V 측정을 가능케 했다. F–K 태양형 별의 경우 ZDI에서의 평균 ⟨B⟩는 보통 수십 Gauss, Doppler 지도의 국소 장 세기는 수백 Gauss까지. M 왜성의 Stokes V ⟨B⟩는 최대 ~1.5 kG(Morin 외 2008, 2010) — Stokes V로 관측한 더 뜨거운 태양형 별보다 훨씬 크지만 Stokes I 장 값의 작은 분수에 불과. Kochukhov 외(2011)는 HR 1099에서 태양형 별 최초의 선편광(Q, U) 검출: 원편광 5 × 10⁻⁵, 선편광이 10배 약함.

**거성**(§3.2.2, 표 6): FK Com 같은 활성 거성의 ZDI 지도는 수백 Gauss까지; 대부분의 거성은 ⟨B⟩ ≲ 1 G (Arcturus, Pollux, Betelgeuse, EK Boo).

**어린 별**(§3.2.3, 표 7): 전주계열성의 광구 선 ZDI는 수백 Gauss까지; 그러나 강착 충격 영역에서 형성되는 방출선(He I 5876 Å)은 최대 3 kG — 거의 상쇄되지 않은 flux tube 다발을 드러내는 강착 column의 기하적 효과.

### Part VIII: Rotation–Magnetic-Field–Activity Relation (§4, pp. 45–50) / 회전–자기장–활동도 관계

**EN**: Dynamo theory predicts activity scales with the dynamo number D, related to the α-effect and shear. The **Rossby number** Ro = P_rot / τ_conv encodes the Coriolis influence on convection. Pizzolato et al. (2003) showed a tight rotation-activity relation (Figure 18):

$$\log(L_X / L_{\rm bol}) \sim -2.5 \times \log(Ro) + \text{const}, \quad Ro \gtrsim 0.1$$
$$\log(L_X / L_{\rm bol}) \approx -3 \text{ (saturated)}, \quad Ro \lesssim 0.1$$

Saar (1996a, 2001) found Bf scales with Ro^(~-1.5). Figure 19 (crosses = sun-like from Saar, squares = M dwarfs from Reiners et al. 2009a) plots Bf vs Ro over Ro ~ 0.003 to 1: a rising relation until saturation at a few kG for Ro ≲ 0.1.

An important caveat: at very low masses (M7–M9, Reiners & Basri 2010), the correlation between v sin i and Bf weakens, suggesting the rotation-activity relation may break down. Atmospheric neutrality (ionization fraction drops below ~3000 K, Meyer & Meyer-Hofmeister 1999; Mohanty et al. 2002) may weaken coupling between magnetic fields and stellar atmospheres.

Section 4.2 discusses Hα activity vs Bf (Figure 20): in early/mid-M (≤M6), Hα saturates around Bf = 2000 G. The implication: saturation of *activity* at Ro ~ 0.1 reflects saturation of the *field itself*, not just of the heating mechanism.

Section 4.3: A rough approximation from Figure 21 gives Bf ≈ 50 v_eq (Gauss, km/s). Combining with Eq. 4, the ratio of Zeeman shift to rotational broadening is

$$\frac{\Delta v_{\rm Zeeman}}{v_{\rm eq}} \approx 0.07 \, \lambda_0 \, g \quad \text{(Eq. 13, λ₀ in μm)}$$

At optical wavelengths (λ₀ = 0.5 μm) this is only ~3% for g=1 — below 10% of rotational broadening. IR observations are essential for sun-like Zeeman detection.

**KR**: 다이나모 이론은 활동도가 α 효과와 전단에 관련된 dynamo number D에 스케일함을 예측한다. **Rossby 수** Ro = P_rot / τ_conv는 대류에 대한 Coriolis 영향을 부호화한다. Pizzolato 외(2003)는 밀착된 회전-활동 관계를 보였다(그림 18):

$$\log(L_X / L_{\rm bol}) \sim -2.5 \log(Ro) + \text{const}, \quad Ro \gtrsim 0.1$$
$$\log(L_X / L_{\rm bol}) \approx -3 \text{ (포화)}, \quad Ro \lesssim 0.1$$

Saar(1996a, 2001)는 Bf가 Ro^(~-1.5)로 스케일함을 발견. 그림 19(십자 = Saar의 태양형, 사각형 = Reiners 외 2009a의 M 왜성)는 Ro ~ 0.003 ~ 1 범위에서 Bf vs Ro를 도시: Ro ≲ 0.1에서 수 kG로 포화되기까지 상승하는 관계.

중요한 주의: 매우 낮은 질량(M7–M9, Reiners & Basri 2010)에서 v sin i와 Bf 상관관계가 약해져 회전-활동 관계가 무너질 수 있음을 시사. 대기 중성화(~3000 K 이하에서 이온화 분수 감소, Meyer & Meyer-Hofmeister 1999; Mohanty 외 2002)가 자기장과 항성 대기의 결합을 약화시킬 수 있다.

§4.2는 Hα 활동도 vs Bf(그림 20): 초·중기 M(≤M6)에서 Hα는 Bf ≈ 2000 G 부근에서 포화. 함의: Ro ~ 0.1의 *활동도* 포화는 가열 메커니즘만이 아니라 *자기장 자체*의 포화를 반영.

§4.3: 그림 21의 대략적 근사는 Bf ≈ 50 v_eq (Gauss, km/s). 식 4와 결합하면 Zeeman 이동 대 자전 넓힘의 비는

$$\frac{\Delta v_{\rm Zeeman}}{v_{\rm eq}} \approx 0.07 \, \lambda_0 \, g \quad \text{(λ₀ μm 단위)}$$

가시광(λ₀ = 0.5 μm)에서 g=1일 때 ~3% — 자전 넓힘의 10% 미만. 태양형 Zeeman 검출에 적외선 관측이 필수.

### Part IX: Equipartition, Geometry, Beyond Rotation (§5–7, pp. 51–59) / 에너지 등분배·기하·회전을 넘어

**EN**: **Equipartition** (§5): Magnetic pressure B²/8π must be balanced by gas pressure. Saar (1990) estimates equipartition fields of 1–2 kG (G type), 2–3 kG (K), 3.5–4 kG (early-mid M). The *maximum average* field observed in M dwarfs (~4 kG) closely matches these predictions, though the *local* field strength in magnetic regions can exceed equipartition.

**Geometries** (§6): Figure 22 shows the Sun observed by SDO during the August 2010 Great Eruption, with extrapolated coronal field lines. Figure 23 (Donati 2011, Morin et al. 2010) plots large-scale magnetic geometry vs rotation period and stellar mass: symbol size = field strength, shape = axisymmetry, color = poloidal (red) vs toroidal (blue). Key findings:
- More rapidly rotating stars produce stronger, more axisymmetric, poloidal fields.
- Very-low-mass rapid rotators (M ≲ 0.1 M☉) show *different* geometries than early-M, with more complex, non-axisymmetric fields.
- Stokes V maps detect only ~10% of the full field (Figure 24), implying >90% cancellation between opposite polarities.

**Beyond rotation** (§7): Christensen et al. (2009) proposed a scaling law based on internal energy flux:

$$\frac{B^2}{2\mu_0} \propto f_{\rm ohm} \, \rho^{1/3} \, (F q_0)^{2/3}$$

This unifies T Tauri stars, old M dwarfs, brown dwarfs, Jupiter, and Earth on a single relation (Figure 25). It predicts kG fields on rapidly-rotating, evolved brown dwarfs. Brown dwarfs (§7.2): all young (≲10 Myr) brown dwarfs investigated to date show no Zeeman detection, yet radio observations indicate kG fields on some old L-type brown dwarfs. Fossil vs dynamo fields in young stars (§7.3): Yang & Johns-Krull (2011) propose fossil fields decaying over Myr, but Chabrier & Küker (2006) predict fossil decay in ≤1000 yr for fully convective stars — hence dynamos must be operating in T Tauri stars.

**KR**: **에너지 등분배**(§5): 자기 압력 B²/8π은 기체 압력과 균형을 이뤄야 한다. Saar(1990)는 등분배 장을 G형 1–2 kG, K형 2–3 kG, 초·중기 M형 3.5–4 kG로 추정. M 왜성에서 관측된 *최대 평균* 자기장(~4 kG)은 이 예측과 잘 일치하지만, 자기 영역의 *국소* 장 세기는 등분배를 초과할 수 있다.

**기하**(§6): 그림 22는 2010년 8월 대분출 때 SDO가 관측한 태양에 외삽된 코로나 자기력선을 보여준다. 그림 23(Donati 2011, Morin 외 2010)은 대규모 자기 기하를 자전 주기와 항성 질량의 함수로 도시: 심볼 크기 = 장 세기, 모양 = 축대칭성, 색 = 폴로이달(적)/토로이달(청). 핵심 결과:
- 더 빠르게 자전하는 별은 더 강하고, 더 축대칭적이며, 폴로이달 자기장을 생성.
- 매우 저질량 빠른 회전자(M ≲ 0.1 M☉)는 초기 M형과 *다른* 기하를 보이며, 더 복잡하고 비축대칭적.
- Stokes V 지도는 전체 자기장의 ~10%만 검출(그림 24), 반대 극성 간 >90% 상쇄를 의미.

**회전을 넘어**(§7): Christensen 외(2009)는 내부 에너지 플럭스에 기반한 스케일링 법칙을 제안:

$$\frac{B^2}{2\mu_0} \propto f_{\rm ohm} \, \rho^{1/3} \, (F q_0)^{2/3}$$

이는 T Tauri 별, 오래된 M 왜성, 갈색왜성, 목성, 지구를 단일 관계로 통합(그림 25). 빠르게 자전하는 진화된 갈색왜성에서 kG 자기장을 예측. 갈색왜성(§7.2): 지금까지 조사된 모든 어린(≲10 Myr) 갈색왜성은 Zeeman 검출이 없지만, 전파 관측은 일부 오래된 L형 갈색왜성에서 kG 자기장을 시사. 어린 별의 화석 vs 다이나모 자기장(§7.3): Yang & Johns-Krull(2011)은 Myr 규모로 감쇠하는 화석 자기장을 제안하지만, Chabrier & Küker(2006)는 완전 대류 별에서 ≤1000 yr의 화석 감쇠를 예측 — 따라서 T Tauri 별에서 다이나모가 작동해야 함.

---

## 3. Key Takeaways / 핵심 시사점

1. **The Bf degeneracy is fundamental** — 관측량은 Bf뿐이다.
   **EN**: Integrated-light Zeeman broadening measures the *product* of field strength B and filling factor f, never B alone. A star covered 50% with B = 1 kG and a star covered 25% with B = 2 kG produce nearly identical Stokes I signatures. This makes Bf an "average flux density" rather than a flux in the strict sense, and different techniques (equivalent widths, profile fitting, polarization) weight different parts of the distribution.
   **KR**: 통합광 Zeeman 넓힘은 B와 f의 *곱*만 측정하며, B 자체는 결코 측정하지 못한다. B = 1 kG에 50% 덮인 별과 B = 2 kG에 25% 덮인 별은 Stokes I에서 거의 동일한 서명을 준다. 이는 Bf를 엄격한 플럭스가 아니라 "평균 플럭스 밀도"로 만들며, 서로 다른 기법(등가폭, 프로파일 적합, 편광)은 분포의 서로 다른 부분을 가중한다.

2. **Infrared is vastly more sensitive than optical for Zeeman detection** — 적외선이 가시광보다 훨씬 민감하다.
   **EN**: The λ² dependence of Zeeman wavelength shift (Δλ = 46.67 g λ₀² B) means a 1 kG field at λ = 12,000 Å produces ~27 mÅ shift, while at 5000 Å it gives only ~5 mÅ. In velocity terms (Δv ∝ λ), IR gains a factor of 2.4. This is why the K2 dwarf ε Eri's field was revised from ~800 G (optical) to 130 G (IR) by Valenti et al. (1995).
   **KR**: Zeeman 파장 이동(Δλ = 46.67 g λ₀² B)의 λ² 의존성 때문에, 1 kG 자기장이 12,000 Å에서는 ~27 mÅ 이동을 만들지만 5000 Å에서는 ~5 mÅ에 불과하다. 속도로는(Δv ∝ λ) 적외선이 2.4배 이득. 이 때문에 Valenti 외(1995)는 K2 왜성 ε Eri의 자기장을 ~800 G(가시광)에서 130 G(적외선)로 하향 조정했다.

3. **M dwarfs harbor fields 2–3 orders of magnitude stronger than the Sun** — M 왜성은 태양보다 2–3자리 수 강한 자기장을 품는다.
   **EN**: Mid/late-M dwarfs routinely show Bf ≈ 2–4 kG via Stokes I Zeeman broadening, compared to the Sun's disk-averaged ~10 G (Zeeman) to 100 G (Hanle). If 50% of an M dwarf with mean 4 kG is "quiet", the active half must carry ~8 kG. This has severe implications for habitability of exoplanets around M dwarfs.
   **KR**: 중·후기 M 왜성은 Stokes I Zeeman 넓힘으로 일상적으로 Bf ≈ 2–4 kG를 보이며, 태양의 원반 평균 ~10 G(Zeeman) ~ 100 G(Hanle)와 대조된다. 평균 4 kG M 왜성의 50%가 "조용한" 광구라면, 활성 반은 ~8 kG를 운반해야 한다. 이는 M 왜성 주변 외계행성의 거주 가능성에 심각한 함의를 가진다.

4. **The Rossby-number scaling unifies rotation, magnetic field, and activity** — Rossby 수 스케일링이 회전·자기장·활동도를 통합한다.
   **EN**: Normalized X-ray luminosity log(L_X / L_bol) vs Ro = P_rot / τ_conv shows a rising relation up to Ro ~ 0.1 then a saturation plateau. Bf scales approximately as Ro^(-1.5) in the unsaturated regime (Saar 1996a). Saturation at Ro ~ 0.1 reflects saturation of the *field itself*, not merely activity. The relation spans sun-like stars through fully convective mid-M dwarfs, suggesting a common dynamo principle.
   **KR**: 정규화된 X-선 광도 log(L_X / L_bol) vs Ro = P_rot / τ_conv는 Ro ~ 0.1까지 상승한 뒤 포화 평탄면을 보인다. 비포화 영역에서 Bf는 대략 Ro^(-1.5)로 스케일(Saar 1996a). Ro ~ 0.1의 포화는 단순 활동도가 아니라 *자기장 자체*의 포화를 반영. 태양형 별에서 완전 대류 중기 M 왜성까지 관계가 지속되며 공통 다이나모 원리를 시사한다.

5. **Stokes V maps reveal only ~10% of the true magnetic flux** — Stokes V 지도는 실제 자기 플럭스의 ~10%만 드러낸다.
   **EN**: Comparing average Stokes V field ⟨B_V⟩ with Stokes I field ⟨B_I⟩ in M dwarfs (Figure 24), the ratio ⟨B_V⟩/⟨B_I⟩ is typically ≲ 10%. The magnetic energy ratio ⟨B²_V⟩/⟨B²_I⟩ is 0.3–15%. Cancellation between opposite polarities renders most of the field invisible to circular polarization, with implications for the completeness of ZDI reconstructions.
   **KR**: M 왜성에서 평균 Stokes V 장 ⟨B_V⟩와 Stokes I 장 ⟨B_I⟩를 비교하면(그림 24), 비율 ⟨B_V⟩/⟨B_I⟩은 보통 ≲ 10%. 자기 에너지 비율 ⟨B²_V⟩/⟨B²_I⟩은 0.3–15%. 반대 극성 간 상쇄가 자기장 대부분을 원편광에 보이지 않게 만들며, ZDI 재구성의 완전성에 함의를 가진다.

6. **The M3/M4 fully-convective boundary does NOT produce an obvious dynamo discontinuity in field strength** — M3/M4 완전대류 경계에서 자기장 세기는 급격한 불연속을 보이지 않는다.
   **EN**: Despite solar-dynamo theory assigning a privileged role to the tachocline (absent in fully convective stars), Stokes I measurements show no sharp break in Bf at M3/M4 — both partial and fully convective M dwarfs reach kG fields when rapidly rotating. However, ZDI *geometries* differ: early-M and very-low-mass rapid rotators show qualitatively different topologies, suggesting different dynamo mechanisms may coexist or a small-scale dynamo may complement the tachoclinic one.
   **KR**: 태양 다이나모 이론이 tachocline(완전대류 별에는 부재)에 특권적 역할을 부여함에도 불구하고, Stokes I 측정은 M3/M4에서 Bf의 급격한 단절을 보이지 않는다 — 빠르게 자전하는 부분·완전대류 M 왜성 모두 kG 자기장에 도달. 그러나 ZDI *기하*는 다르다: 초기 M형과 매우 저질량 빠른 회전자는 질적으로 다른 위상을 보이며, 서로 다른 다이나모 메커니즘의 공존 또는 소규모 다이나모가 tachocline 다이나모를 보완할 수 있음을 시사.

7. **Young brown dwarfs show no detectable magnetic fields in Zeeman despite having accretion disks** — 어린 갈색왜성은 강착원반을 가짐에도 Zeeman 검출이 없다.
   **EN**: Reiners et al. (2009b) investigated four young accreting brown dwarfs and detected no Zeeman signatures — in stark contrast to pre-MS stars of similar age (Bf ≈ 2 kG) and older brown dwarfs where radio emission implies ≥ 3 kG fields. Possible causes: accretion-disk regulation of magnetism, larger radii of young brown dwarfs diluting surface flux, or an ineffective dynamo at these masses.
   **KR**: Reiners 외(2009b)는 강착 중인 어린 갈색왜성 4개를 조사했으나 Zeeman 서명을 검출하지 못했다 — 비슷한 나이의 전주계열성(Bf ≈ 2 kG) 및 전파 방출이 ≥ 3 kG 자기장을 시사하는 오래된 갈색왜성과 극적 대조. 가능 원인: 강착원반의 자기장 조절, 어린 갈색왜성의 더 큰 반경에 의한 표면 플럭스 희석, 또는 이 질량대에서 비효율적인 다이나모.

8. **Equipartition sets a natural cap near a few kG for average main-sequence fields** — 에너지 등분배가 주계열 평균 자기장에 자연스러운 상한(수 kG)을 설정한다.
   **EN**: Magnetic pressure B²/8π must be balanced by gas pressure. Saar (1990) estimates equipartition fields of 1–2 kG (G), 2–3 kG (K), 3.5–4 kG (early-mid M). These closely match the *maximum observed average* M-dwarf fields. Local field strengths in magnetic regions may exceed equipartition, but the global average on an ensemble of fluxtubes plus field-free photosphere cannot.
   **KR**: 자기 압력 B²/8π은 기체 압력과 균형을 이뤄야 한다. Saar(1990)는 등분배 장을 G형 1–2 kG, K형 2–3 kG, 초·중기 M형 3.5–4 kG로 추정. 이는 M 왜성의 *최대 관측 평균* 자기장과 잘 맞는다. 자기 영역의 국소 장 세기는 등분배를 초과할 수 있으나, 플럭스 튜브 집합체 + 자기장이 없는 광구의 전역 평균은 초과할 수 없다.

---

## 4. Mathematical Summary / 수학적 요약

### A. Zeeman effect and Landé factor / Zeeman 효과와 Landé 인자

The Landé factor for a single level (LS coupling):
$$g_i = \frac{3}{2} + \frac{S_i(S_i+1) - L_i(L_i+1)}{2 J_i (J_i+1)} \quad \text{(Eq. 1)}$$

The **effective Landé factor** for a transition between levels (u, l):
$$g = \frac{1}{2}(g_u + g_l) + \frac{1}{4}(J_u - J_l)(g_u - g_l)(J_u + J_l + 1) \quad \text{(Eq. 2)}$$

**EN**: g = 1 for a "normal" triplet; typical astrophysical lines have g between ~0.5 (Zeeman-insensitive) and ~3 (highly sensitive). Fe I 8468 Å has g = 2.5; FeH lines span g ≈ 0.5 to 2.5.
**KR**: "정상" 삼중항은 g = 1; 천체물리 선은 보통 g가 ~0.5(Zeeman 둔감)에서 ~3(매우 민감) 사이. Fe I 8468 Å는 g = 2.5; FeH 선은 g ≈ 0.5 ~ 2.5.

### B. Zeeman splitting / Zeeman 분리

Wavelength shift (σ-component displacement from line center):
$$\boxed{\Delta\lambda = 46.67 \, g \, \lambda_0^2 \, B} \quad \text{(Eq. 3, mÅ; λ₀ μm; B kG)}$$

Velocity shift:
$$\boxed{\Delta v = 1.4 \, \lambda_0 \, g \, B} \quad \text{(Eq. 4, km/s; λ₀ μm; B kG)}$$

Equivalently with Δλ in Å, λ₀ in Å, B in Gauss:
$$\Delta\lambda_B = 4.67 \times 10^{-13} \, g \, \lambda_0^2 \, B$$

**Numerical examples / 수치 예제**:

| λ₀ | g | B | Δλ_B | Δv |
|---|---|---|---|---|
| 5000 Å | 1.0 | 1 kG | 11.7 mÅ | 0.7 km/s |
| 5000 Å | 2.5 | 1 kG | 29.2 mÅ | 1.75 km/s |
| 12000 Å | 1.0 | 1 kG | 67.2 mÅ | 1.68 km/s |
| 12000 Å | 2.5 | 1 kG | 168 mÅ | 4.20 km/s |
| 22000 Å | 2.5 | 1 kG | 566 mÅ | 7.7 km/s |

**EN**: For a 1 kG field in a g=2.5 line, optical Zeeman splitting is marginally below typical spectrograph resolution, whereas at 2.2 μm it is comfortably resolved.
**KR**: g=2.5 선에서 1 kG 자기장의 가시광 Zeeman 분리는 일반 분광기 분해능 직하이지만, 2.2 μm에서는 충분히 분해된다.

### C. Stokes vectors and weak-field approximation / Stokes 벡터와 약장 근사

Stokes parameter definitions:
- **I** = total intensity (↕ + ↔)
- **Q** = ↕ − ↔ (linear polarization, reference frame)
- **U** = ↗ − ↘ (linear polarization, 45° rotated)
- **V** = ↻ − ↺ (circular polarization)

**Weak-field approximation** (Unno 1956, Stenflo 1994), valid when Δv_Zeeman ≪ Doppler width:

$$\boxed{V(v) = -\frac{1}{c} \, g_{\rm eff} \, \lambda_0 \, B_{\parallel} \, \frac{\partial I(v)}{\partial v}} \quad \text{(Eq. 5)}$$

$$\boxed{Q(v) = \frac{1}{4 c^2} \, g_{\rm eff}^2 \, \lambda_0^2 \, B_{\perp}^2 \, \frac{\partial^2 I(v)}{\partial v^2}} \quad \text{(Eq. 6)}$$

**EN**: These relations are the foundation of LSD: all lines share the profile derivative, scaled by g and intrinsic line depth, so their information can be co-added. V ∝ B_∥ (linear in longitudinal field), Q ∝ B_⊥² (quadratic in transverse field) — explaining why Q/U are factor ~10–20 weaker than V.
**KR**: 이 관계식이 LSD의 기반이다: 모든 선은 동일한 프로파일 미분을 공유하며 g와 내재 선 깊이로 스케일되므로, 정보를 누적할 수 있다. V ∝ B_∥(종방향 자기장에 선형), Q ∝ B_⊥²(횡방향 자기장에 제곱) — Q/U가 V보다 ~10–20배 약한 이유.

### D. Field, flux, filling factor / 장·플럭스·채움 인자

Observable product:
$$Bf \equiv \langle B \rangle = \text{average unsigned flux density}$$

Total magnetic flux:
$$\mathcal{F} = 4 \pi R^2 \, (Bf) \propto (Bf) \, R^2$$

**EN**: Two stars with identical Bf can have very different total flux if their radii differ — important when comparing T Tauri stars (large R) to field M dwarfs (small R).
**KR**: 두 별의 Bf가 같더라도 반경이 다르면 총 플럭스는 크게 다를 수 있다 — T Tauri 별(큰 R)과 field M 왜성(작은 R) 비교 시 중요.

### E. Rotation–activity scaling / 회전-활동 스케일링

Rossby number:
$$\boxed{Ro = \frac{P_{\rm rot}}{\tau_{\rm conv}}}$$

where τ_conv is the convective turnover time at the base of the convective zone (τ_conv ~ 12–50 d for the Sun, higher for low-mass stars).

Rotation-activity relation (Pizzolato et al. 2003, Noyes et al. 1984):
$$\log \frac{L_X}{L_{\rm bol}} = \begin{cases}
C_1 - 2 \log Ro & Ro \gtrsim 0.1 \quad \text{(unsaturated)} \\
C_2 \approx -3 & Ro \lesssim 0.1 \quad \text{(saturated)}
\end{cases}$$

Magnetic-field scaling (Saar 1996a, 2001):
$$Bf \approx 70 \, Ro^{-1.5} \quad \text{(Gauss)}$$

Rotation-velocity version:
$$Bf \approx 50 \, v_{\rm eq} \quad \text{(Gauss; v in km/s)}$$

Zeeman-to-rotational-broadening ratio:
$$\boxed{\frac{\Delta v_{\rm Zeeman}}{v_{\rm eq}} \approx 0.07 \, \lambda_0 \, g \quad \text{(Eq. 13, λ₀ μm)}}$$

**Numerical example / 수치 예제**:
- Sun (v_eq ≈ 2 km/s, Ro ≈ 1.5): predicted Bf ≈ 70 × 1.5^(-1.5) ≈ 38 G — consistent with observed disk-averaged solar flux 10–100 G.
- Active M dwarf (v_eq = 15 km/s, Ro ≈ 0.05): predicted Bf ≈ 70 × 0.05^(-1.5) ≈ 6200 G — but saturated at ~4 kG in reality.
- Young T Tauri (Ro ≈ 0.01): Bf formula gives 70,000 G but observed values are ~2 kG (equipartition limited).

### F. Equipartition / 에너지 등분배

Magnetic pressure:
$$P_{\rm mag} = \frac{B^2}{8\pi}$$

Equipartition condition:
$$\frac{B^2}{8\pi} \lesssim P_{\rm gas}$$

Numerical estimates (Saar 1990):
| Spectral type | B_equi |
|---|---|
| G | 1–2 kG |
| K | 2–3 kG |
| early/mid M | 3.5–4 kG |

### G. Christensen et al. (2009) unified scaling / Christensen 외(2009) 통합 스케일링

$$\boxed{\frac{B^2}{2\mu_0} = c \, f_{\rm ohm} \, \rho^{1/3} \, (F q_0)^{2/3}}$$

where f_ohm is the fraction of convective power dissipated as ohmic heating, ρ is density, F is a geometric factor, and q_0 is the convective heat flux. This scaling fits Earth, Jupiter, M dwarfs, and T Tauri stars on a single line (Figure 25).

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1897 ─── Zeeman discovers atomic level splitting in B-field
              │
1924 ─── Hanle effect (scattering polarization)
              │
1947 ─── Babcock: first stellar magnetic field (Ap stars, Zeeman)
              │
1971 ─── Preston: photoelectric Zeeman analyzer
              │
1980 ─── Robinson: Fourier-transform Zeeman in sun-like stars (ξ Boo A, 70 Oph A, 61 Vir)
              │
1985 ─── Saar & Linsky: first M-dwarf Zeeman (AD Leo, Ti I 2.22 μm) — 2.8 kG
              │
1988 ─── Saar: improved radiative transfer, line-blend corrections
              │
1989 ─── Semel: ZDI concept — polarized Doppler Imaging
              │
1995 ─── Valenti et al.: ε Eri field revised DOWN from 800→130 G using IR
              │
1996 ─── Johns-Krull & Valenti: FeI 8468 Å method, EV Lac 3.9 kG
              │
1997 ─── Donati et al.: Least Squares Deconvolution (LSD)
              │
2000 ─── Wade et al.: all-four-Stokes in Ap/Bp stars (Q, U baseline for cool stars)
              │
2003 ─── Pizzolato et al.: X-ray/Rossby rotation-activity relation
              │
2006 ─── Reiners & Basri: FeH 1 μm empirical method for M dwarfs
              │
2008 ─── Donati et al., Morin et al.: Stokes-V ZDI maps of M dwarfs up to 1.5 kG
              │
2009 ─── Christensen et al.: unified dynamo scaling across planets, BDs, stars
              │
2009 ─── Reiners et al.: no Zeeman detection in young accreting brown dwarfs
              │
2010 ─── Anderson et al.: 59 Vir re-analysis — Bf consistent with 0 at optical!
              │
2011 ─── Kochukhov et al.: first linear-polarization detection in cool stars (HR 1099)
              │
────────► 2012 ★ REINERS — "Observations of Cool-Star Magnetic Fields"
              │         definitive observational review; ~150 references
              │
2015+ ─── Continuing: SPIRou, CARMENES, CRIRES+, PEPSI
              │         refine observations; near-IR opens new windows
              │
2018+ ─── ESPaDOnS / Narval BCool survey produces ZDI catalog (~200 stars)
              │
2023 ─── First ZDI of Proxima Centauri (closest M dwarf, habitable-zone planet)
```

**EN**: The paper is a keystone: it stands at the end of a decade of rapid observational progress (FeH method, ZDI maps, unified scalings) and is the most-cited reference by subsequent M-dwarf/exoplanet-host stellar-magnetism papers.
**KR**: 본 논문은 쐐기돌이다: 빠른 관측적 진보의 10년(FeH 기법, ZDI 지도, 통합 스케일링) 끝에 위치하며, 이후의 M 왜성/외계행성 숙주 항성 자기장 논문들에서 가장 많이 인용된다.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **Zeeman (1897)** — Original discovery of line splitting in magnetic fields | Theoretical foundation of every technique in this review / 본 리뷰의 모든 기법의 이론적 기반 | ★★★ Direct — without Zeeman there is no stellar magnetometry / 직접적 — Zeeman 없이는 항성 자기 측정 자체가 없음 |
| **Babcock (1947)** — First stellar magnetic field detected in 78 Vir (Ap star) | First application of Zeeman to stars other than Sun / 태양 외 항성에 Zeeman을 최초 적용 | ★★★ Historical anchor — cool-star work extends Ap-star precedents to far subtler signals / 역사적 정박점 — cool star 작업은 Ap 별 선례를 훨씬 미묘한 신호로 확장 |
| **Semel (1989)** — Zeeman Doppler Imaging proposal | Inversion technique for vector field mapping / 벡터 자기장 지도화의 역변환 기법 | ★★★ Core method — §2.1.7 devoted to ZDI; Tables 5–7 list ZDI maps / 핵심 기법 — §2.1.7 전체가 ZDI; 표 5–7이 ZDI 지도 수록 |
| **Donati et al. (1997)** — Least Squares Deconvolution (LSD) | Multi-line co-addition boosts polarimetric S/N | ★★★ Enabling technique — nearly all post-1997 Stokes V detections use LSD / 가능 기법 — 1997년 이후 거의 모든 Stokes V 검출이 LSD 사용 |
| **Saar (1988, 1996a)** — Improved Zeeman analysis for cool stars | Framework for radiative transfer + filling factor + blends | ★★★ Foundational — Reiners explicitly builds on Saar's compilations in §3.1.1 and §4 / 기초 — §3.1.1과 §4에서 Saar의 자료집을 명시적으로 토대로 함 |
| **Valenti et al. (1995)** — IR Zeeman of ε Eri revises field downward | Demonstrated optical bias; motivated IR push | ★★★ Key empirical caveat — referenced throughout §3.1.1 / 핵심 실증적 주의 — §3.1.1 전반에서 인용 |
| **Johns-Krull & Valenti (1996, 2000)** — Fe I 8468 Å M-dwarf Zeeman | First robust M-dwarf Bf measurements at optical | ★★★ Core M-dwarf dataset in Table 2 / 표 2의 핵심 M 왜성 데이터셋 |
| **Noyes et al. (1984); Pizzolato et al. (2003)** — Rotation-activity relation via Rossby number | Empirical rotation-activity connection, saturation at Ro~0.1 | ★★★ Central framework of §4 / §4의 핵심 framework |
| **Christensen et al. (2009)** — Unified energy-flux dynamo scaling | Connects brown dwarfs, M dwarfs, planets | ★★ Featured in §7.1, Figure 25 / §7.1과 그림 25에 수록 |
| **Morin et al. (2008, 2010)** — Stokes-V ZDI in M dwarfs | Recent high-impact M-dwarf ZDI survey | ★★ Primary reference for §3.2.1 and geometry section / §3.2.1 및 기하 절의 주요 참고 |
| **Kochukhov & Piskunov (2002)** — 4-Stokes ZDI crosstalk | Demonstrates Stokes-I+V limitations in ZDI | ★★ Figure 6 reproduction; methodological caution in §2.1.7 / 그림 6 재현; §2.1.7의 방법론적 주의 |
| **Paper #27 Solanki et al. (2006)** — *The solar magnetic field* (if reviewed) | The "ground truth" stellar dynamo benchmark | ★★ Solar reference for comparing cool-star measurements / 차가운 별 측정 비교의 태양 기준 |
| **Paper #29 (likely) Brun & Browning / Charbonneau** — Solar dynamo theory | Provides theoretical framework for rotation-activity observations | ★★ Complementary — Reiners provides data, dynamo theorists interpret / 상보적 — Reiners는 데이터, 다이나모 이론가들이 해석 |
| **Living Reviews solar activity / chromosphere / corona** | Indirect diagnostics (Ca II, X-ray) used in §2.3 | ★ Contextual connection for indirect methods / 간접 기법을 위한 맥락적 연결 |

---

## 7. References / 참고문헌

### Primary citation / 주 인용
- Reiners, A. (2012), "Observations of Cool-Star Magnetic Fields", *Living Reviews in Solar Physics*, **8**, 1. DOI: 10.12942/lrsp-2012-1. URL: http://www.livingreviews.org/lrsp-2012-1

### Foundational Zeeman / 기초 Zeeman
- Zeeman, P. (1897), "On the Influence of Magnetism on the Nature of the Light Emitted by a Substance", *Ap.J.*, **5**, 332.
- Hanle, W. (1924), *Z. Phys.*, **30**, 93.
- Condon, E. U. & Shortley, G. H. (1963), *The Theory of Atomic Spectra*, Cambridge University Press.
- Beckers, J. M. (1969), *Solar Phys.*, **9**, 372; *Solar Phys.*, **10**, 262.
- Landi Degl'Innocenti, E. & Landolfi, M. (2004), *Polarization in Spectral Lines*, Kluwer.

### Stellar magnetic field methodology / 항성 자기장 방법론
- Stokes, G. G. (1852), *Trans. Cambridge Philos. Soc.*, **9**, 399.
- Robinson, R. D., Jr. (1980), *Ap.J.*, **239**, 961.
- Saar, S. H. (1988), *Ap.J.*, **324**, 441.
- Saar, S. H. (1996b), *IAU Symposium 176*, 237.
- Semel, M. (1989), *A&A*, **225**, 456.
- Donati, J.-F. et al. (1997), *MNRAS*, **291**, 658.
- Unno, W. (1956), *Publ. Astron. Soc. Japan*, **8**, 108.
- Stenflo, J. O. (1994), *Solar Magnetic Fields: Polarized Radiation Diagnostics*, Kluwer.
- Kochukhov, O. & Piskunov, N. (2002), *A&A*, **388**, 868.
- Donati, J.-F. & Brown, S. F. (1997), *A&A*, **326**, 1135.
- Donati, J.-F. & Landstreet, J. D. (2009), *ARA&A*, **47**, 333.

### Sun-like star measurements / 태양형 별 측정
- Preston, G. W. (1971), *Ap.J.*, **164**, 309.
- Vogt, S. S. (1980), *Ap.J.*, **240**, 567.
- Gray, D. F. (1984), *Ap.J.*, **277**, 640; (1985), *PASP*, **97**, 719.
- Basri, G., Marcy, G. W. & Valenti, J. A. (1990, 1992), *Ap.J.*
- Valenti, J. A., Marcy, G. W. & Basri, G. (1995), *Ap.J.*, **439**, 939.
- Anderson, R. I., Reiners, A. & Solanki, S. K. (2010), *A&A*, **522**, A81.
- Marsden, S. C. et al. (2006a, 2006b), *MNRAS*.
- Petit, P. et al. (2008), *MNRAS*, **388**, 80.
- Kochukhov, O. et al. (2011), *Ap.J. Lett.*, **732**, L19.

### M-dwarf, pre-MS, brown-dwarf measurements / M 왜성·전주계열성·갈색왜성 측정
- Saar, S. H. & Linsky, J. L. (1985), *Ap.J. Lett.*, **299**, L47.
- Johns-Krull, C. M. & Valenti, J. A. (1996), *Ap.J. Lett.*, **459**, L95; (2000), *ASP Conf. Ser.*, **198**, 371.
- Reiners, A. & Basri, G. (2006), *Ap.J.*, **644**, 497; (2007), *Ap.J.*, **656**, 1121; (2010), *Ap.J.*, **710**, 924.
- Reiners, A., Basri, G. & Browning, M. (2009a), *Ap.J.*, **692**, 538.
- Reiners, A. et al. (2009b), *Ap.J.*, **697**, 373.
- Morin, J. et al. (2008), *MNRAS*, **390**, 567; (2010), *MNRAS*, **407**, 2269.
- Johns-Krull, C. M. (2007), *Ap.J.*, **664**, 975.
- Yang, H. & Johns-Krull, C. M. (2011), *Ap.J.*, **729**, 83.
- Phan-Bao, N. et al. (2006, 2009), *Ap.J.*
- Kochukhov, O. et al. (2009, 2011), *A&A*.
- Shulyak, D. et al. (2010, 2011), *A&A*.

### Rotation–activity, dynamos, scaling / 회전-활동·다이나모·스케일링
- Noyes, R. W. et al. (1984), *Ap.J.*, **279**, 763.
- Pizzolato, N. et al. (2003), *A&A*, **397**, 147.
- Pevtsov, A. A. et al. (2003), *Ap.J.*, **598**, 1387.
- Schrijver, C. J. et al. (1989), *Ap.J.*, **337**, 964.
- Saar, S. H. (1990), *IAU Symp. 138*, 427; (2001), *ASP Conf. Ser.*, **223**, 292.
- Christensen, U. R., Holzwarth, V. & Reiners, A. (2009), *Nature*, **457**, 167.
- Charbonneau, P. (2010), *Living Rev. Solar Phys.*, **7**, 3.
- Ossendrijver, M. (2003), *A&A Rev.*, **11**, 287.
- Kiraga, M. & Stępień, K. (2007), *Acta Astron.*, **57**, 149.
- Barnes, S. A. & Kim, Y.-C. (2010), *Ap.J.*, **721**, 675.
- Kim, Y.-C. & Demarque, P. (1996), *Ap.J.*, **457**, 340.

### Indirect diagnostics and radio / 간접 진단·전파
- Güdel, M. (2002, 2004), *Ap.J.*; *ARA&A*.
- Berdyugina, S. V. (2005), *Living Rev. Solar Phys.*, **2**, 8.
- Hall, J. C. (2008), *Living Rev. Solar Phys.*, **5**, 2.
- Trujillo Bueno, J., Shchukina, N. & Asensio Ramos, A. (2004), *Nature*, **430**, 326.
- Hallinan, G. et al. (2008), *Ap.J.*, **684**, 644.
- Berger, E. et al. (2005, 2009), *Ap.J.*.
- Bagnulo, S. et al. (2002), *A&A*, **394**, 1023.
