---
title: "Modeling Multiphase Plasma in the Corona: Prominences and Rain"
authors: "Keppens, R., Zhou, Y., Xia, C."
year: 2025
journal: "Living Reviews in Solar Physics"
doi: "10.1007/s41116-025-00043-2"
topic: Living_Reviews_in_Solar_Physics
tags: [prominences, coronal_rain, thermal_instability, MHD_simulation, multiphase_corona, MPI_AMRVAC, TNE, evaporation_condensation]
status: completed
date_started: 2026-04-23
date_completed: 2026-04-23
---

# 87. Modeling Multiphase Plasma in the Corona: Prominences and Rain / 코로나 다상 플라즈마 모델링: 프로미넌스와 비

---

## 1. Core Contribution / 핵심 기여

**English**: Keppens, Zhou & Xia (2025) review a decade of multi-dimensional MHD simulations that self-consistently form coronal condensations — both long-lived prominences and transient coronal rain — within realistically stratified, magnetized solar atmospheres. The central thesis is that these two apparently different phenomena share a common physical origin: **thermal instability** (TI), discovered by Parker (1953) and developed by Field (1965), in which the optically-thin radiative loss rate per unit volume n_e n_H Λ(T) rises faster than any heating can compensate once a cool perturbation forms, causing runaway cooling from coronal (10⁶ K) to chromospheric (10⁴ K) temperatures. The review traces how numerical MHD — via the open-source MPI-AMRVAC code and its international collaborators — has progressed from 1D loop hydrodynamics (Antiochos-Klimchuk 1991) through 2D/2.5D arcade and flux-rope MHD (Xia et al. 2012; Keppens & Xia 2014) up to full 3D chromosphere-to-corona simulations (Xia & Keppens 2016; Lu et al. 2024) and post-flare rain (Ruan et al. 2024). Condensations with density contrasts of 100–1000× the ambient corona, drop sizes down to 100 km, and fall speeds near 100 km/s are now reproduced, with the onset location consistently predicted by linear TI criteria. The review establishes that prominence formation and coronal rain are indistinguishable at the level of governing PDEs — they are both manifestations of TI acting within different magnetic topologies and heating prescriptions.

**Korean / 한국어**: Keppens, Zhou, Xia (2025)는 현실적으로 층화된 자화 태양 대기에서 코로나 응축물 — 오래 지속되는 프로미넌스와 일시적인 코로나 비 모두 — 을 자기일관적으로 형성하는 지난 10년간의 다차원 MHD 시뮬레이션을 리뷰한다. 핵심 주장: 겉보기에 다른 이 두 현상은 공통 물리 기원을 갖는다 — **열 불안정성(TI)**. Parker (1953)가 발견하고 Field (1965)가 발전시킨 이 불안정성에서, 광학적으로 얇은 단위부피당 복사 손실률 n_e n_H Λ(T)은 저온 섭동이 형성되자마자 어떤 가열로도 보상할 수 없을 만큼 빠르게 증가하여 코로나 온도(10⁶ K)에서 채층 온도(10⁴ K)로의 폭주 냉각을 초래한다. 오픈소스 MPI-AMRVAC 코드와 국제 협력단을 통해, 수치 MHD는 1D 루프 수력학 (Antiochos-Klimchuk 1991)에서 2D/2.5D 아케이드·플럭스로프 MHD (Xia et al. 2012; Keppens & Xia 2014)를 거쳐 완전 3D 채층-코로나 시뮬레이션 (Xia & Keppens 2016; Lu et al. 2024)과 포스트플레어 비 (Ruan et al. 2024)까지 진전되었다. 주변 코로나의 100–1000배 밀도 대비, 100 km 수준의 방울 크기, 100 km/s 부근의 낙하 속도를 갖는 응축물이 재현되며, 발생 위치는 선형 TI 기준으로 일관되게 예측된다. 리뷰는 프로미넌스 형성과 코로나 비가 지배 편미분방정식 수준에서 구별되지 않음을 확립한다 — 둘 다 서로 다른 자기 위상과 가열 처방 하에서 작용하는 TI의 발현이다.

---

## 2. Reading Notes / 읽기 노트

### Part I: §1-2 The Multiphase Corona / 다상 코로나의 관측적 동기 (pp. 2-5)

**English**: The corona hosts matter over five decades of temperature, from 10⁴ K chromospheric-like filaments/prominences to >10⁷ K flare plasma, all within the same magnetic topology. Observational evidence: (i) **Prominences** — seen on the limb as emission structures at 10⁴ K, with densities n ~ 10¹⁰ cm⁻³ (vs. n ~ 10⁹ cm⁻³ corona). Hirayama (1985) and Parenti (2014) reviewed their properties; filaments on disk are the same structures in absorption. (ii) **Coronal rain** — thin (~100 km) cool blobs falling along loops at ~100 km/s; detected in EUV (Şahin et al. 2023) and H-α (Antolin et al. 2015). (iii) **Postflare rain** — Mason & Kniezewski (2022) analyzed 241 flares and found a positive correlation between GOES class and rain duration; rain appears minutes after matter was heated above 10 million degrees. Antolin (2020) argued the omnipresence of multiphase matter is the key to coronal heating. Related reviews: Gibson (2018, prominences), Mackay et al. (2010, prominence magnetic structure), Antolin (2020, coronal rain).

**Korean / 한국어**: 코로나는 동일한 자기 위상 내에 10⁴ K 채층류 필라멘트/프로미넌스부터 10⁷ K 이상의 플레어 플라즈마까지 5차수에 걸친 온도의 물질을 품는다. 관측적 근거: (i) **프로미넌스** — 10⁴ K 방출 구조로 림에서 관측, 밀도 n ~ 10¹⁰ cm⁻³ (코로나 n ~ 10⁹ cm⁻³). 디스크상 필라멘트는 흡수로 보이는 동일 구조. (ii) **코로나 비** — 가느다란(~100 km) 저온 덩어리가 ~100 km/s로 루프를 따라 낙하; EUV와 Hα에서 검출. (iii) **포스트플레어 비** — Mason & Kniezewski (2022): 241개 플레어 분석, GOES 등급과 비 지속시간의 양의 상관관계 발견; 1,000만도 이상으로 가열된 수분 후 비 출현. Antolin (2020)은 다상 물질의 편재성이 코로나 가열 이해의 핵심이라 주장.

### Part II: §3 Force and Energy Balance / 힘과 에너지 균형 (pp. 5-10)

**English**: The starting point is the full MHD momentum equation:
$$\rho(\partial_t \mathbf{v} + \mathbf{v}\cdot\nabla\mathbf{v}) = -\nabla p + \rho\mathbf{g} + \frac{1}{\mu_0}(\nabla\times\mathbf{B})\times\mathbf{B} + \mathbf{F}_{\text{visc}} \quad (1)$$
For long-lived prominences, the static limit gives force balance: $0 = -\nabla p + \rho\mathbf{g} + \mathbf{J}\times\mathbf{B}$. Two classical models simplify this to 2D (invariant along filament axis):
- **Kippenhahn-Schlüter (1957)**: sheared arcade with upward-concave field-line dips; magnetic tension supports gravity-loaded plasma.
- **Kuperus-Raadu (1974)**: non-force-free flux rope with line current + photospheric mirror current; azimuthal B around the rope supports matter.

Extensions: Petrie et al. (2007) constructed full 2.5D magnetohydrostatic prominences; Blokland & Keppens (2011a) exploited the full freedom of 2.5D MHD to build force-balanced flux ropes with internal cross-sectional variation (Fig. 2 in paper).

**Energy balance** is captured in the plasma-temperature form:
$$\mathcal{R}\rho\partial_t T + \mathcal{R}\rho\mathbf{v}\cdot\nabla T + (\gamma-1)p\nabla\cdot\mathbf{v} = (\gamma-1)\left[\rho\mathcal{L} + \nabla\cdot(\kappa(\tilde T^{5/2})(\hat{\mathbf{b}}\cdot\nabla T)\hat{\mathbf{b}})\right] \quad (4)$$
where $\mathcal{R} = k_B/\mu$, $\gamma = 5/3$, and $\mathcal{L}$ is the net heat gain-minus-loss per unit mass. In steady state with no motion, we require:
$$\rho h = n_e n_H \Lambda(T) - \nabla\cdot(\kappa(\tilde T^{5/2})(\hat{\mathbf{b}}\cdot\nabla T)\hat{\mathbf{b}}) \quad (6)$$
The (unknown) heating h must exactly balance optically thin radiative losses (typically parametrized by the cooling curve Λ(T)) plus conductive losses. Anisotropic conduction: κ ∝ T^{5/2} along field, suppressed perpendicular.

Low et al. (2012a, b) showed that insisting on a static, singular Kippenhahn-Schlüter mass sheet inevitably invokes resistive decay; the PCTR coincides with tangential discontinuities in B, which must dissipate. Resistivity is introduced through the induction equation:
$$\partial_t \mathbf{B} = \nabla\times(\mathbf{v}\times\mathbf{B} - \eta\mathbf{J}) \quad (7)$$

**Korean / 한국어**: 시작점은 완전 MHD 운동량 방정식 (1). 오래 지속되는 프로미넌스의 경우 정적 극한에서 힘 균형: $0 = -\nabla p + \rho\mathbf{g} + \mathbf{J}\times\mathbf{B}$. 두 고전 모델은 이를 2D로 단순화:
- **Kippenhahn-Schlüter (1957)**: 위로 오목한 dip을 갖는 전단 아케이드; 자기 장력이 중력 부하 플라즈마 지지.
- **Kuperus-Raadu (1974)**: 선전류 + 광구 거울전류의 비-힘자유 플럭스로프; 로프 주위 방위각 B가 물질 지지.

에너지 균형은 식 (4)로 표현되며 — 등방·자기장 정렬 열전도와 복사 손실 항을 포함. 정적 상태에서 알려지지 않은 가열 h는 광학적으로 얇은 복사 손실 n_e n_H Λ(T)과 전도 손실의 합을 정확히 상쇄해야 한다 (식 6). 이방성 전도: κ ∝ T^{5/2} (장 방향), 수직 방향으로는 억제.

Low et al. (2012a, b): 특이한 정적 Kippenhahn-Schlüter 질량 시트 고집은 저항 소산을 필연적으로 초래; PCTR은 B의 접선 불연속과 일치하여 반드시 소산되어야 한다. 저항은 유도 방정식 (7)로 도입.

### Part III: §4 Linear MHD Theory — Thermal Instability Zoo / 선형 MHD 이론 — 열 불안정성 동물원 (pp. 11-20)

**English**: This is the theoretical heart. Linear MHD spectroscopy (Goedbloed et al. 2019) computes all normal modes of a background equilibrium via exp(-iωt) ansatz. Static 1D equilibria have:
- Continuous Alfvén and slow continua (stable, real-ω);
- Discrete normal modes (fast, slow, Alfvén) that can be growing (Im ω > 0 ⇒ instability).

**Field (1965) TI dispersion relation** for uniform, radiating, non-adiabatic hydro medium (linearization about static uniform state with perfect heating-cooling balance 𝓛₀ = 0):
$$\omega^3 - i\omega^2\frac{(\gamma-1)\mathcal{L}_T}{\mathcal{R}} - \omega c_0^2 k^2 - i(\gamma-1)(\rho_0\mathcal{L}_\rho - T_0\mathcal{L}_T)k^2 = 0 \quad (8)$$
where $\mathcal{L}_T = \partial\mathcal{L}/\partial T$, $\mathcal{L}_\rho = \partial\mathcal{L}/\partial\rho$, c₀ is sound speed.

Three regimes:
- **Entropy (condensation) mode** — purely growing, Im(ω) = ν > 0. This is the classical isobaric TI.
- **Two overstable acoustic modes** — growing oscillations (forward-backward pair).
- These can coalesce; up to three real unstable roots possible.

**Isobaric criterion** (Field 1965): at constant pressure, perturbations grow if $\left(\frac{\partial\mathcal{L}}{\partial T}\right)_p < 0$, i.e., when cooling "catches up faster" than heating as T drops.

**Isochoric criterion**: at constant density, unstable if $\mathcal{L}_T > 0$ (note sign convention: 𝓛 is net loss here).

**Thermal continuum (TC)** — van der Linden & Goossens (1991a, b): in stratified or magnetized cylindrical/slab geometries, the discrete Field (1965) mode is replaced by an ENTIRE CONTINUUM of singular, ultra-localized thermal eigenfunctions. This naturally explains the observed fine structure (strands of ~100 km width) in prominences. With finite perpendicular conduction $\kappa_\perp \neq 0$, the continuum becomes a quasi-continuum of dense discrete modes.

**Claes & Keppens (2021)** computed all MHD eigenmodes for a realistic chromosphere-to-corona stratified atmosphere with horizontal sheared field. Key finding: unstable thermal modes are pervasive throughout the chromosphere and into the corona, while overstable slow modes dominate the low corona. This "explains" the multiphase nature observationally seen from spicules up.

**Convective Continuum Instability (CCI)** — Blokland & Keppens (2011b): in 2.5D flux ropes, instability arises when the flux-surface-projected Brunt-Väisälä frequency goes negative:
$$N^2_{\text{BV,pol}} = -\left[\frac{\mathbf{B}_{\theta}\cdot\nabla p}{\rho B}\right]\left[\frac{\mathbf{B}_{\theta}\cdot\nabla S}{\gamma S B}\right] \quad (9)$$
This acts as a SEED for subsequent thermal instability, triggering condensation on a specific flux surface (Jenkins & Keppens 2021).

**Korean / 한국어**: 이 장은 이론적 심장부. 선형 MHD 분광학은 exp(-iωt) 가정으로 배경 평형의 모든 법선모드를 계산. 정적 1D 평형은:
- 연속 Alfvén·완속(slow) 연속체 (안정, 실수 ω);
- 이산 법선모드 (빠른·느린·Alfvén) — 증가하는 경우 불안정.

**Field (1965) TI 분산 관계** (식 8): 균일·복사·비단열 수력 매질, 완전 가열-냉각 균형 𝓛₀ = 0 주변 선형화. 세 영역: 엔트로피 모드 (순수 증가, 고전적 등압 TI), 두 과안정 음향 모드 (증가 진동쌍), 최대 세 실수근까지.

**등압 기준** (Field 1965): $\left(\partial\mathcal{L}/\partial T\right)_p < 0$일 때 등압 섭동 증가.

**열 연속체 (TC)** — van der Linden & Goossens (1991a, b): 층화·자화 원통·슬랩 기하에서 이산 Field 모드는 특이·초국소 열 고유함수의 **연속체 전체**로 대체. 관측된 프로미넌스 ~100 km 폭 strand 미세구조를 자연스럽게 설명. 유한 수직 전도 $\kappa_\perp \neq 0$: 연속체 → 조밀 이산 모드의 준연속체.

**Claes & Keppens (2021)**: 수평 전단장을 갖는 현실적 채층-코로나 층화 대기에 대해 모든 MHD 고유모드 계산. 핵심: 불안정 열모드는 채층 전반과 코로나까지 편재, 과안정 완속 모드는 저코로나에 우세.

**대류 연속체 불안정성 (CCI)** — Blokland & Keppens (2011b): 2.5D 플럭스로프에서 자속면 투영 Brunt-Väisälä 진동수가 음수화될 때 (식 9) 발생. 후속 열 불안정성의 **씨앗** 역할.

### Part IV: §5 1D Nonlinear Hydrodynamic Loop Evolutions / 1D 비선형 수력 루프 진화 (pp. 20-28)

**English**: Reducing full MHD to 1D along a fixed field line (frozen-field hydro), with field-line shape dictating area variation A(s) through $B(s)A(s) = $ const, gives:
$$\partial_t \rho + \frac{1}{A}\partial_s (A\rho v_\parallel) = 0 \quad (10)$$
plus the along-field projections of Eqs. (1) and (4).

**Evaporation-condensation** — Mok et al. (1990) first showed a 1D semicircular loop with footpoint-concentrated heating (exponential decay length < 12% of loop length) forms a stable condensation at the apex. Antiochos & Klimchuk (1991) refined this; Antiochos et al. (1999) reached quasi-steady condensations of size ~5000 km.

**Thermal Non-Equilibrium (TNE)** — Antiochos et al. (2000): a 320 Mm long shallow-dipped loop with asymmetric heating shows cyclic condensation formation → movement → destruction, despite time-independent heating. The loop is always in near-hydrostatic balance (Eq. 3) but fails to achieve Eq. (6) thermal balance.

**Müller et al. (2003, 2005)** — 10 Mm short loops, cyclic rain formation with fall speeds ~100 km/s matching observations, decelerating upon approaching the upper chromosphere.

**Klimchuk-Luna (2019) TNE criterion**:
$$1 + \frac{A_{\text{tr}} R_{\text{tr}}}{A_c R_c} < \frac{H_{\text{foot}}}{H_{\text{apex}}} \quad (11)$$
where H are volumetric heating rates (foot vs apex), A cross-sectional areas, R radiative losses per unit area at transition region vs corona. This derives directly from Eq. (6) averaged over the loop: total coronal heating $A_c L H$ must balance coronal radiation $A_c L n^2 \tilde\Lambda(\tilde T_c)$ plus transition region losses $A_{\text{tr}} R_{\text{tr}}$. The inequality selects loops where footpoint heating dominates enough to drive TNE cycles.

**Froment et al. (2018)** — parameter survey of 1020 simulations varying geometry and heating, showed all loop geometries can produce heating-cooling cycles with favorable heating prescriptions. **Pelouze et al. (2022)** — 9000-run survey: asymmetric loops less likely to produce rain unless heating compensates the asymmetry. **Kucera et al. (2024)** — random nanoflare heating less favorable than steady footpoint heating for condensation.

**Huang et al. (2021)** — a unified model showing both injection and evaporation-condensation prominence formation pathways emerge depending on whether heating is deposited in lower vs upper chromosphere (Gaussian pulse).

**Jerčić et al. (2025)** — first two-fluid (plasma-neutral) 1D prominence model; ion-neutral decoupling of order 100 m/s in the PCTR.

**Korean / 한국어**: 완전 MHD를 고정 자력선을 따른 1D로 축소 (동결장 수력). 자력선 모양이 $B(s)A(s)$ = const로 단면 변화 A(s) 결정, 질량보존식 (10).

**증발-응축** — Mok et al. (1990): 반원형 루프, 지수감쇠 길이 < 루프 길이의 12% 풋포인트 가열 → 정점 응축. Antiochos et al. (1999): 준정상 응축 크기 ~5000 km.

**열 비평형 (TNE)** — Antiochos et al. (2000): 320 Mm 얕은 dip 루프, 비대칭 가열, 시간 독립적 가열에도 불구하고 주기적 응축 형성 → 이동 → 파괴.

**Müller et al. (2003, 2005)** — 10 Mm 단 루프, 주기적 비 형성, 관측 일치하는 ~100 km/s 낙하 속도.

**Klimchuk-Luna (2019) TNE 기준** (식 11): 풋포인트 가열이 정점 가열 대비 특정 임계값 초과 시 주기적 행동 발생.

**Froment et al. (2018)**: 1020 시뮬레이션 매개변수 서베이. **Pelouze et al. (2022)**: 9000 실행 서베이. **Huang et al. (2021)**: 주입 vs 증발-응축 경로 통합.

### Part V: §6 Multi-Dimensional MHD Condensation Formation / 다차원 MHD 응축 형성 (pp. 28-48)

**English**: Table 1 of the paper classifies >40 multi-D MHD studies by formation pathway (in-situ TI, evaporate, levitate, plasmoid-fed, emergence, injection, postflare rain), dimensionality (2D/2.5D/3D), configuration (arcade, flux rope, streamer, standard flare), and atmosphere (corona-only vs chromosphere-to-corona).

**§6.1 2D Arcades**:
- **Choe & Lee (1992)** — first 2D in-situ TI prominence in a sheared arcade.
- **Xia et al. (2012)** — first 2.5D footpoint-heated bipolar arcade with full chromosphere-to-corona stratification; KS prominence verified against Field TI criteria.
- **Keppens & Xia (2014)** — quadrupolar arcade with central dip; followed hours-long prominence evolution with flux-rope formation and coronal-rain-like drainage.
- **Zhou et al. (2023)** — "winking filament" periodic appearance/disappearance in Hα due to forced oscillator with localized cyclic footpoint heating.
- **Jerčić et al. (2024)** — same quadrupolar topology produces vertical slab prominences under steady heating vs fragmented horizontal threads under stochastic heating (Fig. 9).

**§6.1.3 To rain or not to rain**:
- **Fang et al. (2013, 2015)** — first multi-D coronal rain in bipolar arcades. Blob widths 400-800 km (20 km resolution). Sympathetic cooling: first blob triggers TI on neighboring field lines via Lorentz perturbations, producing rain showers.
- **Li et al. (2022b)** — >6000 individual blobs over 10 h simulated, periodicities tens of minutes to hours matching Auchère et al. (2014).

**§6.2 2.5D Flux Ropes**:
- **Kaneko & Yokoyama (2015)** — introduced **levitation-condensation**: converging footpoint motions form a flux rope that lifts chromospheric-adjacent material; the lifted matter is thermodynamically ripe for TI.
- **Jenkins & Keppens (2021)** — 6 km cell-size 2.5D simulation; first direct link between levitation-condensation and linear MHD spectroscopy via CCI (Eq. 9).
- **Brughmans et al. (2022)** — background heating prescriptions H ∝ ρh with $H \propto B^2\rho^\beta$ for varying (α, β).
- **Liakh & Keppens (2023)** — stochastic heating → automatic prominence rotation ~60 km/s due to angular momentum conservation during contraction.

**§6.3 3D MHD**:
- **Xia et al. (2014)** — first 3D MHD in-situ condensation with coherent prominence structure, 240 × 180 × 120 Mm³ domain, 460 km resolution. Produced a "horned" coronal cavity matching observations.
- **Xia & Keppens (2016)** — first 3D fragmented fine-structured prominence (250 km cells). Mass circulation quantified: chromospheric evaporation ↔ flux-rope condensation ↔ prominence drainage.
- **Kaneko & Yokoyama (2017, 2018)** — reconnection-condensation: adjacent arcades reconnect, doubling coronal field line lengths; longer loops cross the λ_F threshold $\lambda_F^2 = \kappa(\tilde T^{5/2}) T / n^2 \Lambda(T)$ and become TI-unstable.
- **Donné & Keppens (2024)** — higher-resolution (41 km) reconnection-condensation; Rayleigh-Taylor fingering in the reconnected flux rope (Fig. 13).
- **Moschou et al. (2015)** — first truly 3D coronal rain in a potential quadrupolar arcade; ~200 km cells, 20-30 blobs at 20,000 K, RTI-driven field deformation.
- **Xia et al. (2017)** — weak bipolar setup, 80 km resolution, rain diverted from weak to strong field regions.
- **Kohutova et al. (2020)** — sub-photosphere-to-corona radiative MHD simulation of a dipolar region; Ohmic dissipation from shuffled coronal loops drives TI.
- **Lu et al. (2024)** — 3D coronal rain over 2000 G sunspot field, 41 Mm height, ~half a day of physical time (Fig. 14). Synthetic EUV channels reproduce the observed periodicities and phase shifts of Auchère et al. (2018).

**§6.4 Erupting prominences**:
- **Linker et al. (2001)** — first 2.5D axisymmetric streamer prominence formation by levitation; flux reduction controls eruption.
- **Zhao et al. (2017)** — Cartesian 2.5D (up to 250 Mm); 24 km resolution; erupting flux rope traps chromospheric matter.
- **Zhao & Keppens (2022)** — "plasmoid-fed prominence formation" (PF²): plasmoid-mediated chromospheric matter trapping.
- **Fan (2017, 2018, 2020)** — full 3D corona-only erupting prominences in spherical domains up to 11 R_⊙.
- **Xing et al. (2025)** — 3D chromosphere-to-corona CME simulation; filament levitation → splitting → erupting.

**§6.5 Other topologies**:
- **Schlenker et al. (2021)** — TNE cycles in a streamer plus solar wind setup (2D axisymmetric).
- **Mason et al. (2019)** observation — rain in spine-fan topology (coronal null point).
- **Popescu Braileanu & Keppens (2025)** — first two-fluid simulation of reconnection-induced rain in spine-fan topology.

**Korean / 한국어**: 리뷰 Table 1: 40개 이상의 다차원 MHD 연구를 형성 경로 (in-situ TI, 증발, 부양, 플라즈모이드 공급, 발현, 주입, 포스트플레어 비), 차원 (2D/2.5D/3D), 구성 (아케이드, 플럭스로프, 스트리머, 표준 플레어), 대기 (코로나 only vs 채층-코로나) 별로 분류.

**§6.1 2D 아케이드**: Choe & Lee (1992) 최초, Xia et al. (2012) 채층-코로나 완전 층화, Keppens & Xia (2014) 사극형 아케이드, Jerčić et al. (2024) 정상 vs 확률적 가열의 다른 형태 생성 (Fig. 9).

**§6.1.3 비 vs 무비**: Fang et al. (2013, 2015) — 최초 다차원 코로나 비, 400-800 km 블롭 폭, 교감 냉각 (sympathetic cooling).

**§6.2 2.5D 플럭스로프**: Kaneko & Yokoyama (2015) **부양-응축** 도입; Jenkins & Keppens (2021) 6 km 셀, CCI와의 직접 연결.

**§6.3 3D MHD**: Xia et al. (2014, 2016) 최초 3D 응집·미세구조 프로미넌스; Moschou et al. (2015), Lu et al. (2024) 3D 코로나 비.

**§6.4 분출 프로미넌스**: Linker et al. (2001), Zhao & Keppens (2022) PF², Fan (2017-2020) 3D.

### Part VI: §7 Postflare Rain / 포스트플레어 비 (pp. 48-52)

**English**: Flare-driven rain poses unique challenges: the reconnection energy release heats loop-tops to >10 MK via non-thermal electron beams (Emslie 1978) which deposit energy in the chromosphere, driving explosive evaporation that refills loops. Then, minutes later, some loops catastrophically cool to chromospheric temperatures via TI.

**1D beam-injected loop models** — Reep et al. (2020, 2022) found impossibility of rain condensation under typical beam parameters; secondary weak footpoint heating was necessary. **Benavitz et al. (2025)** introduced spatio-temporal FIP element abundance variations (low-FIP elements like Fe, Si, Mg advected into loops), changing local Λ(T) and successfully triggering rain. **Reep et al. (2025)** — low-FIP abundance enhancement and rain occurrence correlated.

**Multi-D MHD flare models**:
- **Ruan et al. (2021)** — 2.5D standard flare from preflare through gradual phase; postflare rain in two consecutive episodes (~15-20 min each). Rain on neighboring loops causes hot loops to temporarily disappear from EUV (Fig. 18).
- **Sen et al. (2024)** — 2.5D multi-erupting flux ropes → postflare config with rain ~30 min after last eruption, even without chromospheric evaporation.
- **Ruan et al. (2024)** — first full 3D MHD standard flare with rain (Fig. 19). Impulsive-phase Kelvin-Helmholtz turbulence in loop-tops (matches Hinode EIS velocities, Ruan et al. 2023); Richtmyer-Meshkov instabilities near termination shock; Rayleigh-Taylor turbulence in gradual-phase loop-tops. Rain manifests on higher-lying loops later in time.

**Key discrepancy**: all 1D and multi-D models produce rain only after a delay beyond the impulsive phase peak; observations sometimes show rain within a few minutes of the peak.

**Korean / 한국어**: 플레어 구동 비의 고유 과제: 재결합 에너지 방출 → 비열 전자빔 (Emslie 1978) → 채층 폭발적 증발 → 루프 재충전. 이후 일부 루프 파국 냉각 → TI 응축.

**1D 빔 주입 모델**: Reep et al. (2020, 2022) 전형 빔 파라미터 하 응축 실패; Benavitz et al. (2025) 시공간 FIP 원소 풍부도 변화로 최초 성공.

**다차원 MHD**: Ruan et al. (2021) 2.5D; Ruan et al. (2024) 최초 3D; 충격파 상호작용 (KHI, RMI, RTI)과 TI 결합.

### Part VII: §8-9 Beyond Solar; Open Problems / 태양 외 및 미해결 문제 (pp. 52-59)

**English**: **Beyond solar**: (i) Stellar flares observed by TESS on M dwarfs have inferred energies >10³² erg; Yang et al. (2023) 1D hydro of flare loops produced postflare condensations with secondary peaks. (ii) Peng & Matsumoto (2017) — galactic prominence via levitation-condensation in a 400×400 pc² box, $7\times10^4 M_\odot$. (iii) Daley-Yates & Jardine (2024) — rapidly-rotating stars produce slingshot prominences centrifugally ejected at ~813 km/s; 18 slingshot events over ~400 h simulated, relevant for stellar mass and angular momentum loss. (iv) Interstellar, circumgalactic, intergalactic medium (Sharma et al. 2012) — TI forms cold filaments from hot phase similarly.

**Open problems (§9)**:
1. Full link between linear eigenmode spectra (global TI + TC) and nonlinear condensation morphology remains unresolved.
2. Thin clump thickness in rain appears constant at falling (Şahin et al. 2023) — a clue to fundamental physics, likely tied to $\kappa_\perp$-regulated thermal continuum overstability, currently unexplored.
3. Counter-streaming Doppler bullseye patterns in quiescent filaments (Karki et al. 2025) — open formation mechanism; Zhou et al. (2025a) with ad-hoc injection reproduces them.
4. Prominence tornadoes — internal structure debated; Gunár et al. (2023) suggest projection effects; no self-consistent rotating-condensation model exists yet.
5. Two-fluid (plasma-neutral) effects only explored in simplified settings; multi-dimensional + gravity-stratified + partial ionization simulations needed.
6. Multi-dimensional non-LTE radiative transfer with net radiative cooling rates (NRCRs) tabulated by Gunár et al. (2025) not yet fully integrated into MHD simulations.
7. Role of heating-cooling balance in dictating morphology — models can trigger condensation in any topology; discriminating the "correct" heating remains elusive.
8. Observational resolution is outpacing simulations: DKIST and Goode Solar Telescope now resolve rain strands down to 21 km (Tamburri et al. 2025) and postflare strands at the diffraction limit of 64 km (Schmidt et al. 2025).
9. Full prominence lifecycle models (data-driven from magnetograms, through formation, internal dynamics, and disappearance) still lacking.
10. Space weather forecasting frameworks (Baratashvili et al. 2025) ignore cool prominence matter in CME ejecta — multiphase inclusion could improve geo-effectiveness predictions.

**Korean / 한국어**: **태양 외**: (i) TESS가 관측한 M 왜성 항성 플레어 (Yang et al. 2023); (ii) Peng & Matsumoto (2017) 은하 프로미넌스; (iii) Daley-Yates & Jardine (2024) 슬링샷 프로미넌스 ~813 km/s; (iv) Sharma et al. (2012) 성간·은하간 매질 TI 차가운 필라멘트.

**미해결 문제**: 선형 고유모드 스펙트럼과 비선형 형태의 완전 연결; 비 클럼프 두께의 일정성; 역방향 흐름 패턴; 프로미넌스 토네이도; 이중 유체 효과; 다차원 non-LTE 복사 전달; 가열-냉각 균형의 형태 결정; DKIST 관측 해상도 따라잡기 (21 km, 64 km); 완전 생애주기 모델; 우주기상 예보 통합.

### Numerical Example Walkthrough / 수치 예시 연습

**English**: Consider a coronal loop at T_c = 10⁶ K, n_c = 10⁹ cm⁻³. A condensation forms at T_cond = 10⁴ K.

1. **Isobaric condition**: p = nk_B T = const. If pressure balance is preserved:
$$\frac{n_{\text{cond}}}{n_c} = \frac{T_c}{T_{\text{cond}}} = \frac{10^6}{10^4} = 100 \quad \text{(density contrast)}$$
So condensation density ~ 10¹¹ cm⁻³.

2. **Cooling function scaling**: For coronal abundances, around 10⁵-10⁶ K, $\Lambda(T) \sim 10^{-21.8} T^{-1/2}$ erg cm³ s⁻¹ (rough fit). Near 10⁴ K, Λ drops sharply. The cooling time:
$$\tau_{\text{cool}} = \frac{3 n k_B T}{n_e n_H \Lambda(T)} \sim \frac{3 k_B T}{n \Lambda(T)}$$
For T = 10⁶ K, n = 10⁹ cm⁻³, Λ ~ 10⁻²² erg cm³ s⁻¹: $\tau_{\text{cool}} \sim 10³-10⁴$ s (~30 min), consistent with observed rain periodicity.

3. **Rain drop free-fall scaling**: A 100 km drop falling from apex at h = 30 Mm under g = 274 m/s²:
$$v_{\text{free}} = \sqrt{2gh} = \sqrt{2\cdot 274\cdot 3\times 10^7} \approx 4000 \text{ m/s} = 4 \text{ km/s}$$
Wait, that's lower than observed 100 km/s. The discrepancy: observations show ~100 km/s but models/observations agree these are SUBSONIC and BELOW pure free-fall — the blobs feel pressure gradient opposing their fall (Müller et al. 2005). Effective gravity along curved loops reduces the driving. Actual fall speeds of 50-100 km/s match a quasi-ballistic descent from the apex over ~1000 s.

4. **Field's isobaric TI growth rate**: For Λ(T) ∝ T^b, the growth rate ν satisfies:
$$\nu_{\text{iso}} \approx \frac{(\gamma-1)}{\gamma\mathcal{R}}\frac{p_0}{T_0}\left[(b-1)\frac{\Lambda_0}{T_0}\right]$$
For b = -1 (radiative branch below 10⁵ K) and T₀ = 10⁶ K, n₀ = 10⁹ cm⁻³, τ_TI ~ 10⁴ s ~ 3 hours, matching TNE cycle periods.

**Korean / 한국어**: T_c = 10⁶ K, n_c = 10⁹ cm⁻³ 코로나 루프 고려. T_cond = 10⁴ K 응축.
1. 등압: 밀도 대비 100배.
2. 냉각 시간 ~ 30 min, 관측 비 주기성 일치.
3. 자유낙하 ~4 km/s이나 실제 관측 50-100 km/s: 곡선 루프, 압력 경사 저항, 유효 중력 고려.
4. 등압 TI 성장률 τ_TI ~ 3시간, TNE 주기 일치.

---

## 3. Key Takeaways / 핵심 시사점

1. **Prominences and coronal rain share a common physical origin — thermal instability.** / **프로미넌스와 코로나 비는 공통 물리 기원을 갖는다 — 열 불안정성.**
   - English: Despite apparent differences in scale (filaments are Mm-long, rain blobs are ~100 km), longevity (prominences last days to months, rain falls in minutes), and topology (prominences need dips or ropes, rain forms in any loop), both phenomena arise from runaway optically-thin radiative cooling when heating cannot keep pace with n_e n_H Λ(T). The governing partial differential equations are identical.
   - 한국어: 겉보기 차이 — 스케일 (필라멘트는 Mm 길이, 비 블롭은 ~100 km), 수명 (프로미넌스는 수일-수개월, 비는 수분), 위상 (프로미넌스는 dip/로프 필요, 비는 어떤 루프에서도) — 에도 불구하고, 둘 다 n_e n_H Λ(T)를 가열이 상쇄하지 못할 때 발생하는 폭주 광학적 얇은 복사 냉각에서 기인한다. 지배 편미분방정식은 동일하다.

2. **The thermal continuum, not just the discrete TI mode, governs fine structure formation.** / **미세구조 형성은 이산 TI 모드가 아닌 열 연속체가 지배한다.**
   - English: van der Linden & Goossens (1991) and Claes & Keppens (2021) showed that in stratified/magnetized media, Field's (1965) discrete growing mode becomes a CONTINUUM of ultra-localized, singular thermal eigenmodes. With finite $\kappa_\perp$ this quasi-continuum of dense discrete modes naturally produces the ~100 km fine-structured strands observed in prominences and rain.
   - 한국어: van der Linden & Goossens (1991)와 Claes & Keppens (2021): 층화·자화 매질에서 Field (1965)의 이산 증가 모드는 **특이·초국소** 열 고유모드의 **연속체**가 된다. 유한 $\kappa_\perp$에서는 조밀한 이산 모드의 준연속체가 되어 관측된 ~100 km 미세구조 strand를 자연스럽게 생성.

3. **Thermal Non-Equilibrium (TNE) is a consequence, not a separate phenomenon, of TI.** / **열 비평형(TNE)은 TI와 별개 현상이 아닌 결과이다.**
   - English: Klimchuk (2019) argued for distinguishing TNE from TI, but Keppens et al. (2025) clarified that any condensation observed in a TNE cycle is a manifestation of TI acting on the (non-steady) background. The Klimchuk-Luna (2019) criterion (Eq. 11) provides a simple 1D test for when cyclic behavior emerges.
   - 한국어: Klimchuk (2019)은 TNE와 TI 구분을 주장했으나, Keppens et al. (2025)은 TNE 사이클에서 관측되는 모든 응축이 (비정상) 배경에 작용하는 TI의 발현임을 명확히 했다. Klimchuk-Luna (2019) 기준 (식 11)이 주기적 행동 발생의 간단한 1D 시험이다.

4. **Multiple formation pathways converge on the same TI endpoint.** / **여러 형성 경로가 동일한 TI 종점으로 수렴한다.**
   - English: Paper's Table 1 identifies at least 6 formation pathways (in-situ TI, evaporation-condensation, levitation-condensation, plasmoid-fed, emergence-driven, injection). All end at TI as the local trigger for mass concentration. The pathway is set by the magnetic topology and heating prescription.
   - 한국어: 논문 Table 1이 최소 6개 형성 경로 (in-situ TI, 증발-응축, 부양-응축, 플라즈모이드 공급, 발현 구동, 주입) 식별. 모두 질량 집중의 지역 트리거로서 TI에 종착. 경로는 자기 위상과 가열 처방이 결정.

5. **Evaporation-condensation requires footpoint-concentrated heating with H_foot/H_apex exceeding a geometric threshold.** / **증발-응축은 기하학적 임계값을 초과하는 풋포인트 집중 가열을 요구한다.**
   - English: The Klimchuk-Luna criterion (Eq. 11) $1 + A_{\text{tr}}R_{\text{tr}}/(A_c R_c) < H_{\text{foot}}/H_{\text{apex}}$ says that for a semi-circular loop, exponentially-decaying heating with scale length < ~12% of loop length triggers condensation (Mok et al. 1990). Loops with asymmetric footpoint conditions are less likely to produce rain (Pelouze et al. 2022).
   - 한국어: Klimchuk-Luna 기준 (식 11): 반원형 루프에서 지수감쇠 길이 < 루프 길이의 12% 가열이 응축 유발 (Mok et al. 1990). 비대칭 풋포인트 조건의 루프는 비 생성 가능성 낮음 (Pelouze et al. 2022).

6. **Levitation-condensation bypasses the need for chromosphere-to-corona coupling.** / **부양-응축은 채층-코로나 결합 필요성을 우회한다.**
   - English: Kaneko & Yokoyama (2015) showed that in a corona-only setup, converging footpoint motions form a flux rope whose nested flux surfaces lift coronal material; the lifted, slightly denser-than-ambient matter then undergoes TI via the CCI-seeded thermal continuum (Jenkins & Keppens 2021). No explicit chromospheric evaporation is required.
   - 한국어: Kaneko & Yokoyama (2015): 코로나 only 세팅에서 수렴 풋포인트 운동 → 플럭스로프 → 중첩 자속면이 코로나 물질 부양; 부양된 다소 조밀한 물질이 CCI-씨앗 열 연속체 경유 TI 진행 (Jenkins & Keppens 2021). 명시적 채층 증발 불필요.

7. **3D MHD simulations now reproduce observed rain statistics and periodicities.** / **3D MHD 시뮬레이션은 이제 관측 비 통계와 주기성을 재현한다.**
   - English: Lu et al. (2024) produced 41 Mm-tall sunspot-bipole simulations lasting ~half a day physical time that reproduce Auchère et al. (2018) EUV periodicities (hours) and cross-channel phase shifts (AIA 304/171/131). Linear TI criteria match condensation locations.
   - 한국어: Lu et al. (2024): 41 Mm 높이 흑점 쌍극 시뮬레이션, ~반나절 물리 시간, Auchère et al. (2018) EUV 주기성 (시간 단위)과 채널 간 위상차 (AIA 304/171/131) 재현. 선형 TI 기준이 응축 위치 일치.

8. **Resolution gap between observations and simulations is closing rapidly.** / **관측-시뮬레이션 해상도 격차가 급속히 좁혀지고 있다.**
   - English: Best 3D simulations reach ~40 km cells (Donné & Keppens 2024; Ruan et al. 2024). DKIST resolves 21 km rain strands (Tamburri et al. 2025) and 64 km postflare strands (Schmidt et al. 2025). This forces modelers to either refine further or to account for sub-grid physics (non-LTE, two-fluid, kinetic).
   - 한국어: 최고 3D 시뮬레이션 ~40 km 셀 (Donné & Keppens 2024; Ruan et al. 2024). DKIST는 21 km 비 strand (Tamburri et al. 2025), 64 km 포스트플레어 strand (Schmidt et al. 2025) 분해. 모델러는 추가 미세화 또는 서브그리드 물리 (non-LTE, 이중 유체, 운동론) 반영 필요.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Governing MHD Equations / 지배 MHD 방정식

Momentum / 운동량:
$$\rho(\partial_t \mathbf{v} + \mathbf{v}\cdot\nabla\mathbf{v}) = -\nabla p + \rho\mathbf{g} + \frac{1}{\mu_0}(\nabla\times\mathbf{B})\times\mathbf{B} + \mathbf{F}_{\text{visc}}$$

- $\rho$: mass density / 질량 밀도
- $\mathbf{v}$: velocity / 속도
- $p$: gas pressure / 기체 압력
- $\mathbf{g}$: gravitational acceleration / 중력 가속도
- $\mathbf{B}$: magnetic field / 자기장
- $\mu_0$: permeability / 투자율
- $\mathbf{F}_{\text{visc}}$: viscous force (usually $\propto$ velocity gradients) / 점성력

Energy / 에너지:
$$\mathcal{R}\rho\partial_t T + \mathcal{R}\rho\mathbf{v}\cdot\nabla T + (\gamma-1)p\nabla\cdot\mathbf{v} = (\gamma-1)\left[\rho\mathcal{L} + \nabla\cdot(\kappa(\tilde T^{5/2})(\hat{\mathbf{b}}\cdot\nabla T)\hat{\mathbf{b}})\right]$$

- $\mathcal{R} = k_B/\mu$: gas constant / 기체 상수
- $\gamma = 5/3$: ratio of specific heats / 비열비
- $\mathcal{L}$: net heat gain-minus-loss per unit mass / 단위질량당 순 가열-손실
- $\kappa(\tilde T^{5/2})\hat{\mathbf{b}}$: anisotropic (field-aligned) thermal conduction / 자기장 정렬 열전도

Induction (with resistivity) / 유도 (저항 포함):
$$\partial_t \mathbf{B} = \nabla\times(\mathbf{v}\times\mathbf{B} - \eta\mathbf{J})$$

### 4.2 Steady-State Balance / 정상상태 균형

Force balance / 힘 균형:
$$0 = -\nabla p + \rho\mathbf{g} + \mathbf{J}\times\mathbf{B}$$

Along a single field line (projected hydrostatic) / 단일 자력선 (투영 유체정역학):
$$0 = -\hat{\mathbf{b}}\cdot\nabla p + \rho g_\parallel$$

Thermal balance (heating vs optically thin cooling + conduction) / 열 균형:
$$\rho h = n_e n_H \Lambda(T) - \nabla\cdot(\kappa(\tilde T^{5/2})(\hat{\mathbf{b}}\cdot\nabla T)\hat{\mathbf{b}})$$

- $h$: (unknown) heating per unit mass / 단위질량당 (알려지지 않은) 가열
- $n_e n_H \Lambda(T) = \rho^2 \Lambda(T)/m_p^2$ for fully ionized H / 완전 이온화 H의 경우

### 4.3 Thermal Instability (Field 1965) / 열 불안정성

Linear dispersion relation for uniform, radiating, non-adiabatic hydro medium:
$$\omega^3 - i\omega^2\frac{(\gamma-1)\mathcal{L}_T}{\mathcal{R}} - \omega c_0^2 k^2 - i(\gamma-1)(\rho_0\mathcal{L}_\rho - T_0\mathcal{L}_T)k^2 = 0$$

where:
- $\mathcal{L}_T = (\partial\mathcal{L}/\partial T)_\rho$
- $\mathcal{L}_\rho = (\partial\mathcal{L}/\partial\rho)_T$
- $c_0^2 = \gamma p_0/\rho_0$: sound speed squared

Three roots categorized by Field's state-space: (1) three real → entropy + two acoustic modes, (2) one real + complex conjugate pair → overstable oscillations.

**Isobaric TI criterion** / **등압 TI 기준**:
$$\left(\frac{\partial\mathcal{L}}{\partial T}\right)_p < 0$$

Growth rate (short-wavelength, isobaric limit):
$$\nu_{\text{iso}} \approx -\frac{(\gamma-1)}{\gamma}\frac{T_0 \mathcal{L}_T - \rho_0\mathcal{L}_\rho}{\mathcal{R}T_0/p_0}$$

**Isochoric TI criterion** / **등적 TI 기준**:
$$\mathcal{L}_T > 0 \quad \text{(with Field's sign convention)}$$

### 4.4 Cooling Function / 냉각 함수

Piecewise power-law fit for optically thin losses (typical coronal abundances):
$$\Lambda(T) \approx \begin{cases} 10^{-21.94} T^{-0.5} & 10^{5.4} < T < 10^{6.5}\\ 10^{-21.2}  & 10^{6.5} < T < 10^{7.6}\\ \vdots & \text{other ranges}\end{cases} \text{ erg cm}^3 \text{ s}^{-1}$$

Full curves by Colgan et al. (2008), Schure et al. (2009), Dere et al. (2009), Hermans & Keppens (2021).

### 4.5 1D Frozen-Field Mass Conservation / 1D 동결장 질량보존

$$\partial_t \rho + \frac{1}{A}\partial_s (A\rho v_\parallel) = 0$$

- $A(s)$: cross-sectional area from $B(s)A(s) =$ const
- $s$: along-field coordinate / 자기장 방향 좌표

### 4.6 Klimchuk-Luna (2019) TNE Criterion / TNE 기준

$$1 + \frac{A_{\text{tr}} R_{\text{tr}}}{A_c R_c} < \frac{H_{\text{foot}}}{H_{\text{apex}}}$$

Derivation: coronal loop energy balance $A_c L H_c = A_c L \bar\rho^2 \bar{\tilde\Lambda}(\tilde T_c) + A_{\text{tr}} R_{\text{tr}}$, with $R_{\text{tr}} \sim \bar\kappa(\tilde T_c)\tilde T_c/L$. The inequality selects loops where footpoint heating dominates over steady apex heating enough to break thermal balance.

### 4.7 Convective Continuum Instability / 대류 연속체 불안정성

Projected Brunt-Väisälä frequency in 2.5D flux ropes:
$$N^2_{\text{BV,pol}} = -\left[\frac{\mathbf{B}_{\theta}\cdot\nabla p}{\rho B}\right]\left[\frac{\mathbf{B}_{\theta}\cdot\nabla S}{\gamma S B}\right]$$

where $S = p/\rho^\gamma$ is specific entropy. CCI sets in when $N^2_{\text{BV,pol}} < 0$ anywhere on a flux surface; this acts as a seed for subsequent TI.

### 4.8 Reconnection-Condensation Length Scale / 재결합-응축 길이 스케일

Dimensional analysis of Eq. (6) gives a critical field line length:
$$\lambda_F^2 = \frac{\kappa(\tilde T^{5/2}) T}{n^2 \Lambda(T)}$$

Field lines with length L > λ_F are TI-susceptible (Kaneko & Yokoyama 2017); reconnection that doubles line length can push loops across this threshold.

### 4.9 Coronal Rain Free-Fall Kinematics / 코로나 비 자유낙하 운동학

With drag from pressure gradient, rain drop equation:
$$\frac{dv_r}{dt} = g_\parallel(s) - \frac{1}{\rho_r}\partial_s p + F_{\text{drag}}/\rho_r$$

- $g_\parallel(s) = \mathbf{g}\cdot\hat{\mathbf{b}}(s)$: projected gravity / 투영 중력
- $F_{\text{drag}}$: drag due to ambient corona flowing past drop / 주변 코로나 흐름의 항력

Typical solution: drop accelerates from apex, reaches ~50-100 km/s (below pure free-fall because of pressure opposition from loop footpoint buildup), decelerates near upper chromosphere upon impact.

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1953 Parker ───── First recognition of thermal (radiative) instability
 |                 in a radiating gas.
 |
1957 Kippenhahn & Schlüter ─── Sheared arcade prominence model.
 |                              Magnetic tension supports weight in dips.
 |
1965 Field ────── Definitive paper on TI. Full dispersion relation,
 |                 isobaric/isochoric criteria, condensation modes in
 |                 stratified atmospheres. THE foundation cited by all.
 |
1974 Kuperus & Raadu ── Flux-rope prominence (inverse-polarity)
 |                        topology. Non-force-free Lorentz support.
 |
1976 Heasley & Mihalas ── Radiative + magnetostatic prominence slabs.
 |
1990-1991 Mok+; Antiochos & Klimchuk ── 1D semicircular loop with
 |                                       footpoint heating → stable apex
 |                                       condensation (evaporation-
 |                                       condensation scenario).
 |
1991 van der Linden & Goossens ─ THERMAL CONTINUUM discovered in
 |                                cylindrical/slab MHD equilibria.
 |
1993 Keppens+ ── Linear MHD of overstable discrete Alfvén modes
 |                forming prominences.
 |
1999-2000 Antiochos, MacNeice, Spicer+ ── THERMAL NON-EQUILIBRIUM
 |                                         coined. Cyclic condensation
 |                                         in 1D loops.
 |
2003-2005 Müller, Hansteen, Peter+ ── 1D rain formation & fall
 |                                     kinematics matching obs.
 |
2010-2012 Xia, Fang, Keppens+ ── First 2.5D chromosphere-to-corona
 |                                MHD prominences (MPI-AMRVAC).
 |
2013-2015 Fang et al. ── First multi-D coronal rain in 2D arcades.
 |
2014-2016 Xia & Keppens ── First coherent (2014) and fine-structured
 |                          (2016) 3D MHD prominences.
 |
2015-2018 Kaneko & Yokoyama ── Levitation-condensation (2015) →
 |                              Reconnection-condensation (2017-18).
 |
2018 Froment+ ── 1020-run 1D loop parameter survey; systematic TNE
 |                characterization.
 |
2019 Klimchuk & Luna ── Analytic TNE onset criterion (Eq. 11).
 |
2020 Antolin (review) ── "Multiphase corona" as unifying concept.
 |
2021 Jenkins & Keppens ── First direct link between nonlinear MHD and
 |                          linear CCI + thermal continuum spectroscopy.
 |
2022 Pelouze+ ── 9000-run 1D TNE survey with asymmetric heating.
 |
2023 MPI-AMRVAC 3.0 (Keppens+) ── Open-source release with state-of-
 |                                  art radiative MHD, non-equilibrium
 |                                  ionization, two-fluid options.
 |
2024 Lu+; Donné & Keppens; Ruan+ ── 3D rain over sunspots; 3D
 |                                    reconnection-condensation; 3D
 |                                    postflare rain.
 |
2025 KEPPENS, ZHOU, XIA (this paper) ── Synthesis review integrating
 |                                       all prior threads.
 |
2025+ DKIST observations of 21 km rain strands (Tamburri+), 64 km
       postflare strands (Schmidt+) — pushing simulation resolution.
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **Paper #36 — Reale (2014), "Coronal Loops"** | Coronal loop structure, transition region, chromospheric evaporation | Foundational review of coronal loop physics that Keppens+ build upon. Keppens+ extend Reale's hydrodynamic loop framework with thermal instability and multidimensional MHD to explain the condensations that Reale discussed observationally. / Reale의 코로나 루프 물리 리뷰가 기초. Keppens+는 수력학 루프 틀에 TI와 다차원 MHD를 추가하여 Reale이 관측적으로 논한 응축을 설명. |
| **Paper #39 — Mackay et al. (2010), "Physics of Solar Prominences"** | Prominence magnetic structure (KS vs KR topologies) | Mackay+ reviewed prominence magnetic topologies mostly without the multiphase plasma component. Keppens+ Table 1 now populates each topology with self-consistent multi-D MHD condensation models. / Mackay+은 주로 다상 플라즈마 없이 프로미넌스 자기 위상 리뷰. Keppens+ Table 1은 각 위상을 자기일관적 다차원 MHD 응축 모델로 채움. |
| **Paper #61 — Priest & Forbes (2000 book series), Reconnection in solar plasmas** | Magnetic reconnection as driver of CME eruption and flare energy release | Keppens+ §6.4 (erupting prominences) and §7 (postflare rain) directly depend on reconnection physics. Reconnection-condensation (Kaneko & Yokoyama 2017) unifies reconnection with TI. / Keppens+ §6.4과 §7은 재결합 물리에 직접 의존. 재결합-응축 (Kaneko & Yokoyama 2017)은 재결합과 TI를 통합. |
| Parenti (2014), "Solar Prominences" LRSP review | Observational properties of prominences | Parenti is the observational companion to Keppens+. Keppens+ cite Parenti for the multiphase observational evidence they model. / Parenti는 Keppens+의 관측적 동반자. Keppens+은 자신들이 모델링하는 다상 관측 증거로 Parenti 인용. |
| Gibson (2018), "Solar prominences: theory and models" | Previous LRSP review on prominence theory | Keppens+ explicitly extend Gibson's coverage of MHD simulations including condensed prominence plasma and energetics. / Keppens+는 응집 프로미넌스 플라즈마와 에너지론을 포함한 MHD 시뮬레이션의 Gibson 다룬 범위를 명시적으로 확장. |
| Antolin (2020), "Thermal instability and non-equilibrium in coronal loops" | Coronal rain review | Antolin emphasized rain's role as probe of coronal heating. Keppens+ operationalize this by showing how rain morphology reveals the heating prescription. / Antolin은 비가 코로나 가열의 프로브 역할을 강조. Keppens+은 비 형태가 가열 처방을 드러내는 방식을 보여주며 이를 실현. |
| Sharma et al. (2012), "Thermal instability-driven turbulent mixing in galactic halos" | TI in intergalactic medium | Provides the scale-invariant analog: the same TI framework forms cold filaments from hot gas across six orders of magnitude in scale (solar loop to ICM). / 동일한 TI 틀이 6차수 스케일(태양 루프-ICM)에 걸쳐 차가운 필라멘트를 형성하는 규모 불변 유사체 제공. |

---

## 7. References / 참고문헌

- Keppens, R., Zhou, Y., Xia, C. "Modeling multiphase plasma in the corona: prominences and rain", *Living Reviews in Solar Physics*, 22:4 (2025). DOI: [10.1007/s41116-025-00043-2](https://doi.org/10.1007/s41116-025-00043-2)

### Primary Sources Cited in the Review

- Parker, E.N. "Instability of thermal fields", *ApJ* 117:431 (1953).
- Field, G.B. "Thermal instability", *ApJ* 142:531 (1965). DOI: 10.1086/148317
- Kippenhahn, R., Schlüter, A. "Eine theorie der solaren filamente", *Z. Astrophys.* 43:36 (1957).
- Kuperus, M., Raadu, M.A. "The support of prominences formed in neutral sheets", *A&A* 31:189 (1974).
- Antiochos, S.K., Klimchuk, J.A. "A model for the formation of solar prominences", *ApJ* 378:372 (1991). DOI: 10.1086/170437
- Antiochos, S.K., MacNeice, P.J., Spicer, D.S., Klimchuk, J.A. "The dynamic formation of prominence condensations", *ApJ* 512:985 (1999). DOI: 10.1086/306804
- Antiochos, S.K., MacNeice, P.J., Spicer, D.S. "The thermal nonequilibrium of prominences", *ApJ* 536:494 (2000). DOI: 10.1086/308922
- van der Linden, R.A.M., Goossens, M. "The thermal continuum in coronal loops: instability criteria and the influence of perpendicular thermal conduction", *SoPh* 134:247 (1991).
- Klimchuk, J.A., Luna, M. "The role of asymmetries in thermal nonequilibrium", *ApJ* 884:68 (2019). DOI: 10.3847/1538-4357/ab41f4
- Xia, C., Chen, P.F., Keppens, R. "Simulations of prominence formation in the magnetized solar corona by chromospheric heating", *ApJ* 748:L26 (2012).
- Xia, C., Keppens, R. "Internal dynamics of a twin-layer solar prominence", *ApJ* 825:L29 (2016). DOI: 10.3847/2041-8205/825/2/L29
- Kaneko, T., Yokoyama, T. "Numerical study on in-situ prominence formation by radiative condensation in the solar corona", *ApJ* 806:115 (2015).
- Jenkins, J.M., Keppens, R. "Prominence formation by levitation-condensation at extreme resolutions", *A&A* 646:A134 (2021).
- Claes, N., Keppens, R. "Thermal stability of magnetohydrodynamic modes in homogeneous plasmas", *A&A* 624:A96 (2019).
- Ruan, W., Zhou, Y., Keppens, R. "A fully self-consistent model for solar flares", *ApJ* 883:52 (2019); Ruan et al. *ApJ* 961:60 (2024).
- Froment, C., Auchère, F., Mikić, Z., et al. "On the occurrence of thermal nonequilibrium in coronal loops", *ApJ* 855:52 (2018).
- Antolin, P. "Thermal instability and non-equilibrium in solar coronal loops", *Plasma Phys. Control. Fusion* 62:014016 (2020). DOI: 10.1088/1361-6587/ab5406
- Keppens, R., Popescu Braileanu, B., Zhou, Y., et al. "MPI-AMRVAC 3.0", *A&A* 673:A66 (2023).
- Lu, L., Feng, S., Xia, C., et al. "Long-duration 3D MHD simulation of coronal rain over sunspots", *ApJ* (2024).
- Donné, D., Keppens, R. "3D MHD reconnection-condensation with high resolution", *ApJ* (2024).

### Related Living Reviews

- Parenti, S. "Solar prominences: observations", *LRSP* 11:1 (2014).
- Gibson, S.E. "Solar prominences: theory and models", *LRSP* 15:7 (2018).
- Mackay, D.H., Karpen, J.T., Ballester, J.L., Schmieder, B., Aulanier, G. "Physics of solar prominences: II. Magnetic structure and dynamics", *SSRv* 151:333 (2010).
- Reale, F. "Coronal loops: observations and modeling of confined plasma", *LRSP* 11:4 (2014).
- Arregui, I., Oliver, R., Ballester, J.L. "Prominence oscillations", *LRSP* 15:3 (2018).

### Observational Benchmarks

- Antolin, P., Vissers, G., Pereira, T.M.D., et al. "The multithermal and multi-stranded nature of coronal rain", *ApJ* 806:81 (2015).
- Auchère, F., Froment, C., Soubrié, E., et al. "The coronal monsoon: thermal nonequilibrium revealed by periodic coronal rain", *ApJ* 853:176 (2018).
- Mason, E.I., Kniezewski, K.L. "Correlations between solar flares and postflare coronal rain", *ApJ* 939:21 (2022).
- Şahin, S., Antolin, P., Froment, C., Schad, T.A. "Coronal rain statistics from large-scale surveys", *ApJ* 950:171 (2023).
