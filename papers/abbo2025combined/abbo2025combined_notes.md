---
title: "Combined coronal observations of the streamer belt with Metis and EUI instruments on Solar Orbiter"
authors: L. Abbo, R. Susino, S. Parenti, F. Auchère, V. Andretta, D. Spadaro, M. Romoli, S. Fineschi, R. Lionello, S. Giordano, V. Da Deppo, C. Grimani, P. Heinzel, G. Naletto, G. Nicolini, M. Stangalini, L. Teriaca, M. Uslenghi, Y. De Leo, F. Landini, G. Jerse, M. Pancrazzi, C. Sasso
year: 2025
journal: "Astronomy & Astrophysics, 702, A254 (10 pp.)"
doi: "10.1051/0004-6361/202347599"
topic: Solar Observation / Solar Orbiter — Coronal Diagnostics
tags: [Solar Orbiter, Metis, EUI/FSI, coronagraph, streamer belt, electron temperature, emission measure, polarized brightness, Thomson scattering, CHIANTI]
status: completed
date_started: 2026-04-28
date_completed: 2026-04-28
---

# 61. Combined coronal observations of the streamer belt with Metis and EUI instruments on Solar Orbiter / Solar Orbiter Metis와 EUI 결합 관측을 통한 streamer belt의 코로나 관측

---

## 1. Core Contribution / 핵심 기여

### 한국어
이 논문은 ESA-NASA Solar Orbiter에 탑재된 두 코로나 관측기 — **Metis 코로나그래프** (가시광 편광 밝기 pB + HI Lyα 121.6 nm) 와 **EUI/FSI** (Extreme Ultraviolet Imager / Full Sun Imager, 17.4 nm Fe IX/X 대역) — 의 동시 관측을 **처음으로 결합**하여, 외부 코로나(~4–4.5 R☉)의 streamer belt에서 **전자 온도 T_e**를 영상만으로 추정하는 새로운 진단법을 제시한다. 핵심 절차는 (1) Metis pB의 Thomson 산란 역산으로 전자 밀도 n_e(r) 도출, (2) 시선 방향 적분으로 column emission measure EM 계산, (3) FSI 17.4 nm 응답함수 R(T_e)와 결합해 예상 카운트율을 T_e의 함수로 산출, (4) 측정 카운트와 비교하여 T_e 역산 — 응답함수의 종형(bell-shaped) 특성으로 cold/hot 두 해가 도출된다. 2021년 3월 21일 관측에서 동/서 적도 streamer는 cold ≈ (5.3⁺²·⁰₋₁.₅, 5.7⁺¹·⁹₋₁.₄) × 10⁵ K, hot ≈ 1.4⁺⁰·³₋₀.₂ × 10⁶ K 의 두 해를 산출했다. 이 결과는 UVCS·LASCO·MLSO·일식 관측 기반 기존 결과와 정량적으로 일치하며, 외부 코로나 T_e를 **분광관측 없이 다중대역 영상만으로** 진단할 수 있는 가능성을 처음으로 입증한 의의가 있다.

### English
This paper presents the **first combined analysis** of two Solar Orbiter coronal instruments — the **Metis coronagraph** (visible-light polarized brightness pB + HI Lyα at 121.6 nm) and **EUI/FSI** (Extreme Ultraviolet Imager / Full Sun Imager, 17.4 nm Fe IX/X band) — to derive the **electron temperature T_e** in the outer-corona streamer belt (~4–4.5 R☉) using imaging alone. The methodology proceeds as: (1) invert Metis pB by Thomson-scattering theory to derive electron density n_e(r); (2) integrate n_e² along the line-of-sight to compute the column emission measure EM; (3) combine with the FSI 17.4 nm response function R(T_e) to compute expected count rates as a function of T_e; (4) compare with measured counts to invert for T_e — the bell-shaped response function yields two solutions (cold/hot). From observations on 21 March 2021, the eastern and western equatorial streamers gave cold solutions of (5.3⁺²·⁰₋₁.₅, 5.7⁺¹·⁹₋₁.₄) × 10⁵ K and a hot solution of ~1.4⁺⁰·³₋₀.₂ × 10⁶ K. These results are quantitatively consistent with prior estimates from UVCS, LASCO, MLSO, and eclipse observations, and the paper establishes for the first time the feasibility of diagnosing outer-corona T_e **using multi-band imaging alone, without spectroscopy**.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction (§1, p. 1–2) / 서론

#### 한국어
서론은 **외부 코로나 T_e 측정의 본질적 어려움**을 제기한다. 코로나 가열 메커니즘과 태양풍 가속을 이해하려면 T_e 분포가 필수지만, 직접 측정은 매우 제한적이다.

- **분광선 강도비 방법**(line-ratio diagnostics)은 충돌여기로 형성된 UV 분광선이 필요한데, ≳2 R☉부터는 너무 어두워 측정이 어렵다 (Raymond+ 1997, David+ 1998, Li+ 1998, Parenti+ 2000, Marocchi+ 2001, Bemporad+ 2003, Uzzo+ 2004, 2007).
- **UVCS/SOHO**(1995–2012)는 UV 분광편광 코로나그래프로 ~3.5 R☉까지 T_e를 직접 측정했지만(Kohl+ 1995, 1997; Antonucci+ 2006, 2012, 2020a; Cranmer+ 2017, 2019), 후속 임무가 없었다.
- **간접 추정** — 코로나그래프 pB로 n_e를 얻고 정수압 평형(Munro & Jackson 1977; Gibson+ 1999) 또는 유체역학(Lemaire & Stegen 2016)을 가정해 T_e 도출.
- **협대역 코로나 영상** — 가시광/근적외선 (Gopalswamy+ 2021), 또는 일식 (Habbal+ 2021; Boe+ 2022, 2023; Del Zanna+ 2023) 으로 ~3–4 R☉까지 T_e 측정.

이 논문은 **Solar Orbiter 시대의 새로운 가능성**을 활용한다. Solar Orbiter (Müller+ 2020) 의 두 관측기:
- **Metis** (Antonucci+ 2020b; Fineschi+ 2020) — VL (580–640 nm) pB + UV (HI Lyα 121.6 nm) 동시 관측 코로나그래프, 1.7–9 R☉.
- **EUI/FSI** (Rochus+ 2020) — 17.4 nm (Fe IX/X) Full Sun Imager. 일반 모드는 디스크 영상, 그러나 **coronagraphic mode** (Auchère+ 2023a) — 진입구 앞에 movable occulting disk를 두어 stray light 억제 — 가 ~7.4 R☉까지 EUV 영상을 가능케 한다. **단, 이 모드는 S/C가 0.45 AU 이하일 때만 작동.**

논문 구조:
- §2: 관측 데이터 설명
- §3: 두 방출 메커니즘의 물리
- §4: T_e 도출 방법
- §5: 결과 해석과 가정의 한계 토론

#### English
The introduction frames the **fundamental difficulty of measuring T_e in the outer corona**. Understanding coronal heating and solar-wind acceleration requires the T_e distribution, but direct measurement is severely limited.

- **Line-ratio diagnostics** require collisionally excited UV lines, which become too faint above ~2 R☉ (Raymond+ 1997, David+ 1998, Li+ 1998, Parenti+ 2000, Marocchi+ 2001, Bemporad+ 2003, Uzzo+ 2004, 2007).
- **UVCS/SOHO** (1995–2012) directly measured T_e up to ~3.5 R☉ via UV spectropolarimetry (Kohl+ 1995, 1997; Antonucci+ 2006, 2012, 2020a; Cranmer+ 2017, 2019), with no successor since.
- **Indirect estimates** — coronagraph pB inversion of n_e under hydrostatic equilibrium (Munro & Jackson 1977; Gibson+ 1999) or hydrodynamics (Lemaire & Stegen 2016).
- **Narrow-band coronal imaging** — VL/IR (Gopalswamy+ 2021), or eclipses (Habbal+ 2021; Boe+ 2022, 2023; Del Zanna+ 2023) up to ~3–4 R☉.

This paper exploits **the new possibilities of the Solar Orbiter era** (Müller+ 2020):
- **Metis** (Antonucci+ 2020b; Fineschi+ 2020) — simultaneous VL (580–640 nm) pB + UV (HI Lyα 121.6 nm) coronagraph, 1.7–9 R☉.
- **EUI/FSI** (Rochus+ 2020) — 17.4 nm (Fe IX/X) Full Sun Imager. Normally a disk imager, but a **coronagraphic mode** (Auchère+ 2023a) — using a movable occulting disk at the entrance to suppress stray light — enables EUV imaging up to ~7.4 R☉. **This mode only operates at S/C distances < 0.45 AU.**

Paper structure: §2 observations; §3 physics of two emission mechanisms; §4 T_e derivation method; §5 results & assumption limits.

### Part II: Metis and EUI/FSI Observations (§2, p. 2) / 관측

#### 한국어
**관측일**: 2021년 3월 21일, Solar Orbiter cruise phase. S/C는 태양으로부터 0.68 AU, 일심경도 ~110°E.

**FSI 데이터 (00:45:45 UT)**:
- 코로나그래프 모드, 노출시간 640 s
- Plate scale 4.46″/pixel ≈ 2200 km/pixel (POS, 0.68 AU)
- FOV: 9.7 R☉ (적도면), full FOV 3.8°
- 공간해상도 = 2 × plate scale
- 데이터 처리: dark-subtracted, despiked (Auchère+ 2023a 절차)
- DOI: EUI Data Release 6 (Kraaikamp+ 2023)
- **코로나그래프 모드는 S/C–Sun 거리 ≤ 0.45 AU에서만 가능** — 광구가 너무 가까우면 occulter가 외부 코로나만 비추므로

**Metis 데이터 (03:00 UT 시작)**:
- VL pB (580–640 nm): 14 프레임 × 노출 30 s × 4 편광각, 총 effective 420 s
- UV HI Lyα (121.6 nm): 60 s
- VL plate scale 10.14″/pixel, UV plate scale 20.4″/pixel
- 모든 영상 4×4 binning → ~20,000 km/pixel (VL), ~40,000 km/pixel (UV)
- Metis FOV: ~4 R☉ (occulter 가장자리) 부터 ~7.4 R☉ (적도면)
- 처리·보정: Romoli+ 2021, Andretta+ 2021, De Leo+ 2023, 2025
- 14프레임 평균 후 Müller formalism으로 단일 pB 영상 생성, effective 1680 s

**시간 차이 처리**: FSI와 Metis 사이 2.25시간 차이 → CME나 회전·궤도 효과 무시 (3월 20일 23:50까지의 CME 이후 안정), 두 관측이 거의 동일 코로나 구조를 보는 것으로 간주.

**중요한 사전 사건**: 3월 20일 01:25 UT부터 CME 발생 — SOHO/LASCO, STEREO/SECCHI에서 관측됨. Metis 외부 도어가 23:30 UT(3/17)부터 23:50 UT(3/20)까지 닫혀있어 직접 관측은 안 됨. 그러나 11:00 UT(3/21) 이후 큰 분출이 없어 quasi-static 상태로 회복.

#### English
**Observation date**: 21 March 2021, Solar Orbiter cruise phase, 0.68 AU from Sun, heliographic longitude ~110°E.

**FSI data** (00:45:45 UT): Coronagraphic mode, 640 s exposure. Plate scale 4.46″/pixel ≈ 2200 km/pixel (POS at 0.68 AU). FOV 9.7 R☉ along equator. Spatial resolution = 2 × plate scale. Processed (dark-subtracted, despiked) per Auchère+ 2023a; EUI Data Release 6 (Kraaikamp+ 2023). **The coronagraphic mode only works at S/C–Sun distance ≤ 0.45 AU.**

**Metis data** (starting 03:00 UT): VL pB (580–640 nm) — 14 frames × 30 s exposure × 4 polarization angles, total effective 420 s. UV HI Lyα (121.6 nm) — 60 s. VL plate scale 10.14″/pixel, UV 20.4″/pixel. All images 4×4 binned → ~20,000 km/pixel (VL), ~40,000 km/pixel (UV). FOV ~4–7.4 R☉ along equator. Processing/calibration per Romoli+ 2021, Andretta+ 2021, De Leo+ 2023, 2025. The 14 frames averaged via Müller formalism into a single effective pB image (effective 1680 s).

**Time-gap handling**: 2.25 h gap → no eruptions, rotational/orbital effects negligible.

**CME caveat**: A CME began 01:25 UT on 20 March (seen by SOHO/LASCO, STEREO/SECCHI). Metis external door was closed 23:30 UT 17 Mar – 23:50 UT 20 Mar. After 11:00 UT 21 Mar, no further major eruptions — corona considered quasi-static.

**Figure 1 (overview)**: FSI 17.4 nm at 02:46:43 UT (disk + coronagraphic <1.85 R☉ and 1.85–4.45 R☉) + Metis pB and Lyα at 03:00 UT (>4.45 R☉). White dotted circles mark FSI inner (4.05 R☉) and outer (4.45 R☉) FOV in the overlap region. **Two angular sectors marking the streamers used for analysis** are delimited by dotted lines.

**Figure 2**: Same data with WOW (Wavelet-Optimized Whitening) filter (Auchère+ 2023b) applied for fine structure. Right panel: 3D MHD field-line extrapolation (PSI MAS model, Mikić+ 2018) using HMI synoptic boundaries (CR 2242) — confirms streamer belt geometry.

### Part III: Formation of VL and EUV Band Emissions (§3, p. 3–4) / 두 대역 방출의 형성

#### 한국어

**Metis pB의 물리 — Thomson 산란 / K-corona**:
가시광(580–640 nm)으로 측정한 코로나의 K-corona는 광구에서 나온 빛이 자유 전자에 산란된 빛. **편광되어 있고**, 광학적으로 얇으며, **밀도에 선형**이다. van de Hulst (1950) 적분식 (Eq. 1):

$$
I_\mathrm{pB} \propto \int_\rho^\infty n_e(r)\, [A(r) - B(r)]\, \frac{\rho^2\, dr}{r\sqrt{r^2 - \rho^2}} \tag{1}
$$

- ρ: plane-of-sky 충격 매개변수
- r: 실제 3D 반경
- A(r), B(r): Thomson 산란 위상함수 기하인자 (광구 밝기·산란각의 함수)
- 적분 핵심: 광경로가 LOS 따라 전자 밀도를 누적하지만, ρ 근처에서 가중치가 가장 크다 (1/√(r²−ρ²) 인자)

**구대칭 가정** 하에서 이 식을 *역변환*하면 측정 pB(ρ)에서 3D n_e(r)을 얻을 수 있다. 적도면 streamer 코어처럼 LOS 방향이 거의 균질한 영역에서는 적합한 근사.

**FSI 17.4 nm의 물리 — Fe IX/X 충돌여기**:
EUV 17.4 nm 대역은 Fe IX (171.07 Å)와 Fe X (174.5, 175.3, 177.2 Å) 분광선이 지배. 코로나는 EUV에서 광학적으로 얇으므로 분광선 강도는 (Eq. 2):

$$
I = \int_\mathrm{l.o.s.} A_\mathrm{Fe}\, G(T_e, n_e)\, n_e^2\, dx \tag{2}
$$

- A_Fe: 철 elemental abundance
- G(T_e, n_e): contribution function — 이온화 평형의 Fe IX/X 분율 × 충돌여기율 × 분광선 가지비 (CHIANTI v10.1 계산; Dere+ 1997, 2023)
- T_e의 강한 의존성: 이온화 평형이 log T_e ≈ 5.8–6.0에서 피크
- n_e의 약한 의존성: 충돌여기율의 일부

추가로 **공명산란 기여**가 있을 수 있음(Schrijver & McMullen 2000) — Fe IX 17.1 nm는 광학적으로 얇은 코로나에서 부분적으로 중요.

**HI Lyα의 물리 — 공명산란 + Doppler dimming**:
121.6 nm Lyα는 주로 광구·채층의 광원 광자가 코로나 중성 수소에 **공명산란**되어 나오는 빛 (Gabriel 1971). 산란 강도는:
- 전자 밀도 n_e (양성자와 결합)
- 전자 온도 T_e (이온화 균형 → 중성 H 분율)
- 채층 Lyα 광원 강도와 윤곽
- **Doppler dimming** — 산란 이온이 광원 방향으로 빠르게 흘러나가면 흡수 단면적이 감소 (Hyder & Lites 1970; Withbroe+ 1982; Noci+ 1987)

**중요**: 이 논문에서 Lyα는 *비교용*으로만 사용. T_e 도출에는 사용되지 않음. Doppler dimming 분석으로 LOS 적분(Romoli+ 2021; Antonucci+ 2023)은 본 논문 범위 밖.

**Figure 3 — 위도별 분포 비교**: 4.05–4.45 R☉ 높이에서 polar angle (PA, 북극 반시계)에 따른 강도. 세 대역(Metis pB 빨강, Lyα 파랑, FSI 17.4 nm 회색) 모두 **PA ≈ 90° (동쪽 streamer)** 와 **PA ≈ 265°, 292.5° (서쪽 streamer 두 봉우리)** 에서 피크 — 동일한 코로나 구조를 보고 있음을 확인.

**Figure 4 — 반경별 비교**: 두 streamer 중심에서 강도 vs 거리. 핵심 통찰: **FSI 강도의 √I (검은 굵은 선)** 가 Metis pB (빨간 선)와 *유사한 기울기*를 보임. EUV는 n_e²에 비례하므로 √I ∝ n_e ∝ pB. **이 일치는 FSI 방출이 주로 충돌여기에 의한 것임을 시사** (n_e²-의존성 우세). 공명산란 기여는 ≳4 R☉에서 무시 가능 — *분석의 핵심 가정*.

#### English

**Metis pB physics — Thomson scattering / K-corona**:
The visible-light K-corona is photospheric light scattered off coronal free electrons; it is **polarized**, optically thin, and **linear in n_e**. Van de Hulst (1950) integral (Eq. 1):

$$
I_\mathrm{pB} \propto \int_\rho^\infty n_e(r)\, [A(r) - B(r)]\, \frac{\rho^2\, dr}{r\sqrt{r^2 - \rho^2}}
$$

ρ = plane-of-sky impact parameter, r = 3D heliocentric radius, A(r) and B(r) = Thomson phase-function geometric factors. The kernel weights points near r ≈ ρ most strongly. Under spherical symmetry, *inverting* gives 3D n_e(r) from measured pB(ρ) — adequate for streamer cores.

**FSI 17.4 nm physics — Fe IX/X collisional excitation**:
The 17.4 nm band is dominated by Fe IX (171.07 Å) and Fe X (174.5, 175.3, 177.2 Å) lines. In an optically thin corona, intensity is (Eq. 2):

$$
I = \int_\mathrm{l.o.s.} A_\mathrm{Fe}\, G(T_e, n_e)\, n_e^2\, dx
$$

A_Fe = iron abundance, G = contribution function (ionic fraction × collisional excitation rate × branching ratios; computed via CHIANTI v10.1, Dere+ 1997, 2023). G peaks at log T_e ≈ 5.8–6.0; depends weakly on n_e. There may also be a **resonant-scattering contribution** for Fe IX 17.1 nm (Schrijver & McMullen 2000).

**HI Lyα physics — resonant scattering + Doppler dimming**:
121.6 nm Lyα is mostly chromospheric photons resonantly scattered by neutral coronal H (Gabriel 1971). Intensity depends on n_e (via protons), T_e (ionization balance → neutral H fraction), Lyα chromospheric intensity/profile, and **Doppler dimming** (Hyder & Lites 1970; Withbroe+ 1982; Noci+ 1987).

**Important**: Lyα is used only for comparison, not in the T_e derivation.

**Figure 3 — latitudinal distribution**: At 4.05–4.45 R☉, intensities vs PA. All three bands peak at PA ≈ 90° (east streamer) and PA ≈ 265°, 292.5° (west streamer twin peaks) — confirms the three bands trace the same coronal structures.

**Figure 4 — radial profiles**: At streamer centers, the **√I_FSI** (thick black) closely matches the **Metis pB** (red), since EUV ∝ n_e² implies √I ∝ n_e ∝ pB. This near-coincidence indicates **FSI emission is dominated by collisional excitation**, with negligible resonant scattering above ~4 R☉ — the critical assumption for the analysis.

### Part IV: Data Analysis (§4, p. 4–7) / 자료 분석

#### §4.1 Electron density / 전자 밀도

##### 한국어
**기법**: van de Hulst (1950) 표준 역산 — 측정된 Metis pB(ρ)에서 구대칭(local spherical symmetry) 가정 하 3D n_e(r) 추출.
**가정**: streamer 핵심 영역과 수직방향으로의 LOS 변동이 streamer의 반경 변동보다 작음.
**결과**: Figure 5 (top) — 동/서 streamer를 둘러싼 영역 (PA = 90°, 292.5°) 의 n_e(r, θ) 지도, 4.0–7.5 R☉, 적도 근방.
- 4.25 R☉ 부근 동 streamer 코어: log n_e ≈ 5.4 (n_e ≈ 2.5 × 10⁵ cm⁻³)
- 4.25 R☉ 서 streamer 코어: 비슷한 수준
- streamer 외곽으로 갈수록 점진적 감소

**Figure 5 (bottom)** — 다른 문헌 모델과 비교:
- Romoli+ 2021 (Metis 2020년 5월 첫광 적도 streamer; 녹색)
- Saito+ 1977 (태양 활동 최저기, 점선)
- Withbroe 1988 (quiet-corona, dashed)
- Guhathakurta+ 1999 (heliospheric current sheet, small-dotted)
- Gibson+ 1999 (MLSO/Mk3 + LASCO/C2 적도 streamer 2–4 R☉ 외삽; long-dashed)
- Hayes+ 2001 (LASCO/C2 near-equatorial streamer; dash-dotted)
- Frazin+ 2003 (LASCO/C2 토모그래피; solid)
- Vásquez+ 2003 (적도 밀도 프로파일; dash-triple-dotted)

**비교 결과**: 본 논문의 결과는 절대값과 반경 의존성 모두에서 기존 문헌과 일치 (Withbroe 1988, Hayes+ 2001, Frazin+ 2003 등). 문헌 간 분산이 ×2 이상인 점은 *최저기 코로나 streamer T_e의 본질적 변동성*을 보여줌.

**불확도 처리**: pB 측정 불확도를 이용한 최소·최대 pB로 두 번 역산 → n_e 하한·상한. Fig 5의 빨간·주황 곡선의 오차막대.

##### English
**Technique**: Standard van de Hulst (1950) inversion — extract 3D n_e(r) from measured Metis pB(ρ) under local spherical symmetry.
**Assumption**: LOS variation perpendicular to the streamer is smaller than the radial gradient.
**Result**: Figure 5 (top) — n_e(r, θ) maps in regions surrounding east and west streamers, 4.0–7.5 R☉. Near 4.25 R☉ both streamer cores: log n_e ≈ 5.4 (n_e ≈ 2.5 × 10⁵ cm⁻³).

**Figure 5 (bottom)**: Compared with literature models — Romoli+ 2021 (Metis first light), Saito+ 1977 (solar minimum), Withbroe 1988 (quiet corona), Guhathakurta+ 1999 (HCS), Gibson+ 1999 (MLSO/Mk3 + LASCO/C2 equatorial streamer 2–4 R☉ extrapolated), Hayes+ 2001 (LASCO/C2), Frazin+ 2003 (LASCO/C2 tomography), Vásquez+ 2003. **Match**: this paper's profiles agree in absolute value and radial trend; the >2× spread among literature shows intrinsic streamer variability.

**Uncertainty handling**: Min/max pB from pB-uncertainty → two inversions → n_e lower/upper bounds (error bars in Fig 5 bottom).

#### §4.2 Emission measure and electron temperature / 방출 측정량과 전자 온도

##### 한국어

**Emission Measure 계산**:
plane-of-sky의 각 점에 대해 ±10 R☉ LOS 적분으로 column EM 계산 (Eq. 3):

$$
\mathrm{EM} = \int_\mathrm{l.o.s.} n_e^2(x)\, dx \tag{3}
$$

- 가정: POS 면에서 LOS 방향 n_e(x)는 그 점의 반경 거리에서의 n_e(r)와 같다 (구대칭 적분 → 반경 외삽)
- Metis pB 데이터는 ~7.4 R☉까지 → 그 이상은 외삽
- 검증: 4.25 R☉에서 ±10 R☉ 적분이 무한대 적분의 ≥99% 포착

**Figure 6 (top)** — EM 지도. 동/서 streamer 코어에서 log EM ≈ 22 (단위 cm⁻⁵) 정도까지 도달.

**FSI 카운트율 - T_e 관계**:
이론적 카운트율은 (Eq. 4):

$$
C_\mathrm{FSI} = \frac{1}{4\pi}\int_\mathrm{l.o.s.} R(T_e, n_e)\, n_e^2\, dx \;\approx\; \frac{1}{4\pi}\, R(T_e)\, \mathrm{EM} \tag{4}
$$

여기서 R(T_e, n_e)는 **FSI 17.4 nm 응답함수**로:
- 17.4 nm 대역의 모든 분광선의 contribution function 합산
- 철 abundance × 이온화 평형 (CHIANTI v10.1)
- FSI 분광 응답함수 × 검출기 효율 가중

**중요한 단순화**: R(T_e, n_e) ≈ R(T_e). n_e 의존성은 약함 — 적분 밖으로 인수화 가능.

**철 abundance 처리** (논문 결과의 핵심 불확도):
- Asplund+ 2021 광구 abundance
- Asplund+ 2021 × 10⁰·⁵ (FIP < 10 eV 원소의 first ionization potential bias 보정; CHIANTI 기본값; Dere+ 2023)

이 두 값이 이온화 평형 분율과 결합해 R(T_e)의 두 곡선 → 빨간 띠의 폭

**Figure 6 (bottom)** — 핵심 도식. C_FSI vs log T_e:
- **빨간 띠**: 예상 카운트율 (밀도 불확도 + abundance 불확도 조합); R(T_e) × EM 형태의 *종형* 곡선
- **회색 가로띠**: 측정 카운트 (스트리머 중심에서 평균)
- **두 교차점** = cold solution (저온 측, 상승부) + hot solution (고온 측, 하강부)

응답 함수 R(T_e)가 종형이므로 한 측정값에 두 T_e 해. 이는 **고립된 EUV 협대역 측정의 본질적 모호성** — 단일 대역 영상으로는 cold/hot을 구별 불가.

**이온화 평형 가정**: R(T_e, n_e) 계산은 collisional ionization equilibrium 가정 (§5에서 검증).

**Figure 7 — streamer 중심으로부터 각거리별 T_e**:
- 동 streamer (주황): cold ≈ 5.3 × 10⁵ K, hot ≈ 1.4 × 10⁶ K
- 서 streamer (빨강): cold ≈ 5.7 × 10⁵ K, hot ≈ 1.4 × 10⁶ K
- 실선 = cold, 파선 = hot
- ±15° 각도 범위에서 **거의 평탄** — streamer 내부 밀집한 플라즈마는 균일한 온도

**불확도 출처**:
- Statistical (Poissonian) 카운트 오차
- Calibration: Metis VL ~7%, Metis UV ~15%, FSI ~30% (EUI 팀 추정)
- 밀도 도출 오차 (§4.1)
- Iron abundance 두 값의 차이

cold 해의 큰 불확도는 R(T_e)의 log T_e ≈ 5.4 부근 *plateau* 때문 (Fig 6 bottom 가로띠 연장).

##### English

**EM computation**:
Column EM via ±10 R☉ LOS integration (Eq. 3): EM = ∫ n_e²(x) dx. Assumes LOS variation = radial profile n_e(r) at each impact-point distance. ±10 R☉ captures ≥ 99 % of the asymptotic integral at 4.25 R☉. **Fig 6 (top)**: EM map; both streamer cores reach log EM ≈ 22 cm⁻⁵.

**FSI count rate - T_e relation** (Eq. 4):
C_FSI = (1/4π) ∫ R(T_e, n_e) n_e² dx ≈ (1/4π) R(T_e) EM. R is the FSI 17.4 nm response: sum of contribution functions of all 17.4 nm lines × Fe abundance × ionization balance × FSI spectral response × detector efficiency. Key simplification: R weakly depends on n_e — factor outside integral.

**Iron-abundance treatment**: Two values — Asplund+ 2021 photospheric, and Asplund+ 2021 × 10⁰·⁵ (FIP correction for elements with FIP < 10 eV, CHIANTI default; Dere+ 2023). Sets the width of the red band in Fig 6.

**Figure 6 (bottom)** — the key diagram. Plots C_FSI vs log T_e:
- Red band: expected count rate (density + abundance uncertainties), R(T_e) × EM with bell-shape
- Grey horizontal band: measured count rate at streamer centers
- Two intersections = cold (lower-T branch, rising) + hot (higher-T branch, falling) solutions

This **two-fold ambiguity is intrinsic to a single EUV narrow-band measurement** — cannot be broken without an additional band.

**Ionization-equilibrium assumption**: R(T_e, n_e) computation assumes collisional ionization equilibrium (validated in §5).

**Figure 7** — T_e vs angular distance from streamer center, ±15°:
- East streamer (orange): cold ≈ 5.3 × 10⁵ K, hot ≈ 1.4 × 10⁶ K
- West streamer (red): cold ≈ 5.7 × 10⁵ K, hot ≈ 1.4 × 10⁶ K
- Solid = cold, dashed = hot
- **Nearly flat** within ±15° — dense streamer plasma is roughly isothermal across its angular extent

**Uncertainty sources**: Poisson errors, calibration (Metis VL ~7%, Metis UV ~15%, FSI ~30%), density-inversion errors, iron-abundance choice. Cold-solution large uncertainty traces to the R(T_e) plateau near log T_e ≈ 5.4.

### Part V: Discussion (§5, p. 8–9) / 토의

#### 한국어

**5.1 문헌과의 비교 (Figure 8)**:
4.25 R☉에서 본 논문의 두 해(cold/hot)를 다른 연구들과 직접 비교:
- 빨간 동그라미 = 서 streamer; 주황 = 동 streamer; 채워진 = hot, 빈 = cold
- Gibson+ 1999 (적도, hydrostatic; 검은 실선)
- Vásquez+ 2003 (적도; dashed-triple-dotted)
- Vásquez+ 2003 (적도, 정적; dotted)
- Spadaro+ 2007 (중위도 streamer 축; dash-dotted)
- Susino+ 2008 (streamer/coronal-hole 경계; dash-dotted grey)
- Gopalswamy+ 2021 BITSE 측정 (적도; black filled circle)

**비교 통찰**:
- **Cold 해 (~5.3–5.7 × 10⁵ K)** ↔ Spadaro+ 2007과 Susino+ 2008의 streamer 축 프로파일과 일치
- **Hot 해 (~1.4 × 10⁶ K)** ↔ 일반적 적도 streamer 값과 일치 (Gibson+ 1999; Vásquez+ 2003)
- 다른 연구들은 2.7–2.8 R☉에서 ~1 MK까지 측정. 4 R☉까지 외삽하면 본 논문의 hot 해보다 약간 낮음 — 정수압이 아닌 약간 감소하는 T_e(r)을 가정하면 일치. UVCS/SOHO Lyα profile (Fineschi+ 1998) 도 유사한 streamer T_e 측정.

**5.2 이온화 평형 가정의 검증 (Figure 9)**:
**핵심 도식**. Fe IX (어두운 색)와 Fe X (밝은 색)의 시간 척도들을 4.25 R☉에서 비교:
- **τ_ion** = 1/(n_e α_ion) — 이온화 시간 (파란선/시안선)
- **τ_rec** = 1/(n_e α_rec) — 재결합 시간 (보라선)
- **τ_coll** = 충돌여기 시간 (노란선)
- **τ_exp** — 코로나 확장 시간; v_outflow ∈ {20, 50, 100 km/s} 의 세 점 (녹색, 주황, 빨간)
- 세로 막대 = 본 논문의 cold (≈ 0.5 MK) 와 hot (≈ 1.4 MK) 해

**핵심 결론**:
- 4.25 R☉에서 τ_exp ~ τ_ion ~ τ_rec — **방법이 적용 가능한 경계 조건**
- 빠른 흐름(100 km/s)에서는 frozen-in 가능성 — 논문은 *최악의 경우*에도 ionization equilibrium이 first approximation으로는 합리적이라고 결론
- frozen-in 상태일 때, 도출된 T_e는 *지역 전자 온도가 아닌* freeze-in 높이의 Fe IX/X 이온화·재결합 평형 온도 — 즉 *상한*으로 해석

**비교**: 일식 협대역 관측 (Habbal+ 2010, 2011, 2021; Boe+ 2018, 2023) 에서 Fe X, XI, XIV freeze-in 거리는 일반적으로 1.25–2.2 R☉ 정도 (코로나 홀과 helmet streamer 모두). 그러나 Shen+ 2018 시뮬레이션 → 적도 streamer belt 풍 이온은 5 R☉까지 진화 가능.

**5.3 Isothermal LOS 가정**:
column EM은 등온 LOS 가정 — 실제로는 LOS에 따라 코로나 구조와 온도가 다양하므로 위험. 하지만 두 streamer 코어를 보고 있으므로:
- 대부분의 EUV 방출이 **streamer 내부의 밀집한 플라즈마**에서 옴
- 미세 구조의 다양한 온도 기여 + streamer 외부 저밀도 영역의 기여는 무시 가능

또한 경험적 모델(Withbroe 1988, Vásquez+ 2003, Lemaire & Stegen 2016) 과 일식 측정(Habbal+ 2021, Boe+ 2023, Del Zanna+ 2023) 모두 외부 코로나에서 T_e의 반경 감소율이 일반적으로 *느림* — LOS 내 등온성은 합리적 first approximation.

**5.4 He II 304 nm 채널 활용 가능성**:
같은 방법을 다른 데이터셋으로 적용할 때 **EUI/FSI He II 304 nm 협대역 채널** 추가 → 세 대역 (Metis VL, Metis UV, FSI 174, FSI 304) → cold/hot 모호성 해결 (Guennou+ 2012의 loci method 변형). 현재 데이터셋에는 304 nm 동시 관측이 없음.

**5.5 4 R☉ 이하 적용**:
이 방법을 두 기기 FOV가 코로나 더 낮은 영역에서 겹치는 데이터셋에 적용하면 (이온화 평형 가정이 더 잘 맞는 ~ < 4 R☉) 추가 검증 가능.

#### English

**5.1 Comparison with literature (Figure 8)**:
Direct comparison at 4.25 R☉:
- **Cold solutions (~5.3–5.7 × 10⁵ K)** match Spadaro+ 2007 (mid-latitude streamer axis) and Susino+ 2008 (streamer / coronal-hole boundary) cold profiles
- **Hot solutions (~1.4 × 10⁶ K)** match typical equatorial streamer values (Gibson+ 1999; Vásquez+ 2003)
- Other studies measure ~1 MK at 2.7–2.8 R☉; extrapolating to 4 R☉ gives slightly lower than the hot solution — consistent with non-hydrostatic, slowly declining T_e(r). UVCS Lyα profiles (Fineschi+ 1998) measured similar streamer T_e.

**5.2 Validity of ionization-equilibrium assumption (Figure 9 — the decisive figure)**:
At 4.25 R☉, τ_ion, τ_rec (Fe IX, Fe X) compared with τ_coll and τ_exp (for outflow speeds 20, 50, 100 km/s).
- **τ_exp ~ τ_ion ~ τ_rec at 4.25 R☉** — **the method operates at the boundary of validity**
- For fast outflows (100 km/s), frozen-in is possible → the inferred T_e is then the freeze-in (ionization/recombination equilibrium) temperature of Fe IX/X ions, considered as an *upper limit* of the local electron temperature
- Eclipse Fe X, XI, XIV freeze-in distances are typically 1.25–2.2 R☉ (Habbal+ 2010, 2011, 2021; Boe+ 2018, 2023). However, Shen+ 2018 simulations indicate equatorial streamer belt wind ions can evolve up to 5 R☉.

**5.3 Isothermal-LOS assumption**:
EM in Eq. (3) treats LOS as isothermal — risky in general, but for streamer cores most EUV emission comes from the dense streamer plasma, so the contribution from low-density external regions and small-scale temperature inhomogeneities can be neglected. Empirical models (Withbroe 1988; Vásquez+ 2003; Lemaire & Stegen 2016) and eclipse measurements (Habbal+ 2021; Boe+ 2023; Del Zanna+ 2023) all indicate slowly declining outer-corona T_e — supporting the LOS-isothermal approximation as a first cut.

**5.4 He II 304 nm channel possibility**:
Adding the EUI/FSI He II 304 nm narrow-band channel to future combined-observation datasets would yield three EUV bands → resolve the cold/hot ambiguity by a loci-method variant (Guennou+ 2012). Not available in this dataset.

**5.5 Application below 4 R☉**:
Applying the method where the FOV overlap covers heights < 4 R☉ — where ionization equilibrium holds more securely — would further validate.

### Part VI: Conclusions / 결론

#### 한국어
- Solar Orbiter EUI/FSI 코로나그래프 모드 + Metis VL pB 결합으로 4.25 R☉에서 streamer T_e 영상 진단 *처음 시연*
- 두 streamer 모두 cold ≈ 5.5 × 10⁵ K, hot ≈ 1.4 × 10⁶ K — 기존 진단과 일치
- 이온화 평형과 LOS 등온 가정이 핵심. 4.25 R☉에서 작동하지만 *경계 조건*. 이 방법으로 도출된 두 해는 *T_e 가능한 범위*로 해석 가능
- He II 304 nm 추가 또는 < 4 R☉ 적용으로 추후 검증 가능
- 차세대 우주 코로나 영상기 임무 (Vigil 등) 의 다중대역 진단 설계에 유의미한 시범

#### English
- First demonstration of imaging-only T_e diagnostics for streamers at 4.25 R☉ using Solar Orbiter Metis VL pB + EUI/FSI coronagraphic mode
- Both streamers: cold ≈ 5.5 × 10⁵ K, hot ≈ 1.4 × 10⁶ K — consistent with prior diagnostics
- Ionization equilibrium and LOS isothermality are the key assumptions; method works at 4.25 R☉ as a *boundary case*. The two solutions can be interpreted as *the range of possible T_e*
- Future validation via He II 304 nm or below 4 R☉
- A methodological prototype for next-generation multi-band coronal imaging missions (Vigil, etc.)

---

## 3. Key Takeaways / 핵심 시사점

1. **Imaging-only outer-corona T_e diagnostic / 영상만으로 외부 코로나 T_e 진단** —
   UVCS 종료(2012) 이후 첫번째로 ~4 R☉ 외부 코로나의 T_e를 *분광 없이 영상만으로* 측정. Metis pB → n_e → EM, 그리고 FSI 17.4 nm → T_e의 단순한 흐름. 차세대 임무 설계 (Vigil, PUNCH 등) 에 직접적 영향. /
   First post-UVCS measurement of T_e at ~4 R☉ using *imaging only, no spectroscopy*. The pipeline Metis pB → n_e → EM, FSI 17.4 nm → T_e is simple and reproducible — a methodological prototype for next-generation missions.

2. **Two solutions are intrinsic, not a flaw / 두 해는 결함이 아닌 본질적 모호성** —
   FSI 응답함수 R(T_e)의 종형 모양 때문에 한 측정값에 cold/hot 두 해. 단일 EUV 협대역 영상의 본질적 한계 — 본 분석에서는 *T_e 가능범위*로 해석. He II 304 nm 등 추가 대역으로 해결 가능. /
   The bell-shaped R(T_e) gives two solutions per measurement — a fundamental property of single-band EUV imaging, not an analysis flaw. Treated as the *range of possible T_e*; can be resolved by adding He II 304 nm band.

3. **EM은 두 기기를 잇는 다리 / Emission Measure bridges two instruments** —
   Metis는 n_e (선형, n_e¹), FSI는 n_e²·G(T_e) (이차). EM = ∫ n_e² dx 는 Metis의 출력을 FSI의 입력 형식으로 변환. *이런 종류의 영상 융합에서 항상 등장할 패턴*. /
   Metis traces n_e (linear), FSI traces n_e²·G(T_e) (quadratic). EM = ∫ n_e² dx converts Metis's output to FSI's input — a recurring pattern in such imaging-fusion analyses.

4. **이온화 평형은 4 R☉에서 경계 조건 / Ionization equilibrium is at the limit at 4 R☉** —
   Figure 9에서 τ_exp ~ τ_ion ~ τ_rec — 가정이 위태로운 경계. 만약 frozen-in 상태면 도출된 T_e는 freeze-in 높이의 이온화 평형 온도(상한). 이 한계를 *솔직하게* 보고한 점이 논문의 지적 정직성. /
   At 4.25 R☉, τ_exp ~ τ_ion ~ τ_rec — assumption holds *barely*. If frozen-in, the derived T_e is the freeze-in equilibrium temperature (an upper bound). The honest reporting of this caveat is the paper's intellectual integrity.

5. **두 streamer는 거의 등온 ±15° / Two streamers are nearly isothermal across ±15°** —
   Fig 7에서 streamer 중심 ±15°에서 T_e 변동 미미 — 밀집한 streamer 플라즈마는 거의 균일 온도. 이는 streamer 내부 자기 가둠(magnetic confinement)의 결과로 해석 가능 — 닫힌 자기 루프 안에서 가열 균형이 빠르게 평형. /
   Within ±15° of streamer center, T_e is nearly flat — dense plasma in closed magnetic loops reaches uniform thermal balance.

6. **Iron abundance 선택이 띠 폭의 주요 원인 / Iron abundance choice dominates uncertainty bands** —
   Asplund+ 2021 vs FIP-corrected (×10⁰·⁵) 차이가 R(T_e)와 빨간 띠 폭의 주요 결정자. T_e 자체의 통계적 정밀도보다 abundance의 *선택*이 더 큰 systematic 불확도. abundance 자체가 streamer 안에서 일정한지도 추가 연구 주제. /
   The choice between Asplund+ 2021 photospheric and ×10⁰·⁵ FIP-corrected abundance dominates the red-band width — a *systematic* uncertainty larger than statistical T_e precision. Whether abundance is uniform within streamers itself merits further work.

7. **n_e 결과는 기존 문헌과 견고하게 일치 / n_e results robustly match literature** —
   Fig 5에서 본 논문의 n_e 프로파일이 Withbroe 1988, Hayes+ 2001, Frazin+ 2003 등과 절대값·반경 의존성에서 일치. 이는 *방법 검증* — 만약 n_e가 틀렸다면 EM, T_e 모두 무너짐. 이 일치가 cold/hot T_e 추정의 신뢰성을 떠받친다. /
   n_e profiles match Withbroe 1988, Hayes+ 2001, Frazin+ 2003 in absolute value and radial trend — validates the inversion. If n_e were wrong, EM and T_e would both fail; this consistency supports the cold/hot results.

8. **이 연구는 Solar Orbiter 시대의 다중대역 영상 시너지 시범 / Methodological prototype for the Solar Orbiter era of multi-band imaging synergy** —
   분광에서 협대역 영상으로 이동하는 큰 흐름(DKIST CRYO-NIRSP, ALMA, EUV imager 어레이)의 *모범 사례*. 두 영상기를 결합해 분광 없이도 진단 가능함을 보임 — Vigil, MUSE 등 차세대 임무 설계에 직접적 함의. /
   Solar physics is moving from spectroscopy toward narrow-band imaging (DKIST CRYO-NIRSP, ALMA, EUV imager arrays). This paper exemplifies how imager combinations can substitute for spectroscopy — direct implications for next-generation missions like Vigil and MUSE.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 The pipeline / 처리 흐름

```
Metis pB(ρ)  ──[Eq. 1, van de Hulst inversion]──>  n_e(r)
                                                      │
                                                      ▼
                                          [Eq. 3, LOS integration]
                                                      │
                                                      ▼
                                                  EM(POS point)
                                                      │
                                                      ▼
                          [Eq. 4, with R(T_e) from CHIANTI]
                                                      │
                                                      ▼
                                          C_FSI_predicted(T_e)
                                                      │
                              [Match with measured C_FSI_observed]
                                                      │
                                                      ▼
                                            T_e (cold + hot)
```

### 4.2 Equations in order / 수식 정리

#### Eq. 1 — pB (Thomson scattering)
$$
I_\mathrm{pB}(\rho) \propto \int_\rho^\infty n_e(r)\, [A(r) - B(r)]\, \frac{\rho^2\, dr}{r\sqrt{r^2 - \rho^2}}
$$
- **변수 / Variables**: ρ = POS 충격 매개변수 (impact parameter); r = 3D 반경; A(r), B(r) = Thomson scattering phase-function 기하인자 (van de Hulst 1950)
- **물리 / Physics**: K-corona는 광구 빛의 자유전자 산란 → 광학적 얇음, 편광, n_e 선형
- **역산 / Inversion**: 측정된 pB(ρ)에서 구대칭 가정 하에 n_e(r) 추출

#### Eq. 2 — Optically thin EUV line
$$
I_\mathrm{line} = \int_\mathrm{l.o.s.} A_\mathrm{Fe}\, G(T_e, n_e)\, n_e^2\, dx
$$
- **변수 / Variables**: A_Fe = 철 abundance (수소 대비); G(T_e, n_e) = contribution function (CHIANTI v10.1)
- **G의 구성 / G components**: G = (이온화 분율 of Fe IX/X) × (충돌여기율 to upper level) × (분광선 가지비)
- **n_e 의존성 / n_e dependence**: G는 n_e에 약함 (≪ n_e²)

#### Eq. 3 — Column emission measure
$$
\mathrm{EM}(\rho, \mathrm{POS}) = \int_\mathrm{l.o.s.} n_e^2(x)\, dx
$$
- **단위 / Units**: cm⁻⁵
- **적분 범위 / Integration range**: ±10 R☉ around POS — at 4.25 R☉, captures ≥99% of asymptotic value
- **연결 / Bridge**: optically thin EUV emission ∝ EM × G(T_e)

#### Eq. 4 — FSI count rate (the key equation / 핵심 식)
$$
C_\mathrm{FSI} = \frac{1}{4\pi}\int_\mathrm{l.o.s.} R(T_e, n_e)\, n_e^2\, dx \;\approx\; \frac{1}{4\pi}\, R(T_e)\, \mathrm{EM}
$$
- **R(T_e, n_e) = FSI 응답함수 / response function**:
  - R(T_e, n_e) = ∫ R_inst(λ) · [Σ_lines G_line(T_e, n_e) · A_Fe] dλ
  - R_inst(λ) = FSI 분광 응답 (광학+검출기)
  - 합산 = 17.4 nm 대역의 모든 Fe IX/X 분광선의 contribution function
- **단순화 / Simplification**: R의 n_e 의존성이 약함 → factor outside integral
- **종형 / Bell-shape**: R(T_e)는 log T_e ≈ 5.8–6.0에서 피크 (Fe IX/X 이온화 분율의 함수). 측정 카운트 ↔ T_e 매핑이 *2-to-1* — 이것이 cold/hot 두 해의 기원.

### 4.3 Auxiliary timescales / 보조 시간 척도

이온화 평형 가정의 검증 (§5):

$$
\tau_\mathrm{ion} = \frac{1}{n_e \alpha_\mathrm{ion}(T_e)}, \quad
\tau_\mathrm{rec} = \frac{1}{n_e \alpha_\mathrm{rec}(T_e)}, \quad
\tau_\mathrm{exp} = \left[\frac{v_\mathrm{out}}{n_e}\frac{dn_e}{dr}\right]^{-1}
$$

- α_ion, α_rec = 충돌이온화·재결합 비율 계수 (CHIANTI atomic data)
- v_out = 태양풍 outflow 속도 (논문에서 20, 50, 100 km/s 시나리오)
- **조건 / Condition**: τ_ion·τ_rec ≪ τ_exp → ionization equilibrium 성립
- **본 논문 / This paper**: 4.25 R☉에서 모두 비교 가능 (Fig 9) → "boundary case"

### 4.4 Numerical results table / 결과 표

| Streamer | Position (PA) | n_e at 4.25 R☉ | EM | T_e cold | T_e hot |
|---|---|---|---|---|---|
| East | 90° | ~2.5 × 10⁵ cm⁻³ | log EM ≈ 22 cm⁻⁵ | (5.3⁺²·⁰₋₁.₅) × 10⁵ K | (1.4⁺⁰·³₋₀.₂) × 10⁶ K |
| West | 292.5° | ~2.5 × 10⁵ cm⁻³ | log EM ≈ 22 cm⁻⁵ | (5.7⁺¹·⁹₋₁.₄) × 10⁵ K | (1.4⁺⁰·²₋₀.₃) × 10⁶ K |

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1950 ──── van de Hulst, "Electron density of solar corona"
              ↳ pB inversion formula (Eq. 1) — STILL the standard
              ↳ K-corona Thomson scattering theory + Lick eclipse data
                                │
1971 ──── Gabriel, "Resonance scattering theory in the corona"
              ↳ HI Lyα coronal emission → Lyα diagnostic foundation
                                │
1977 ──── Munro & Jackson, ApJ 213, 874
              ↳ First systematic n_e from pB + hydrostatic T_e
                                │
1988 ──── Withbroe, ApJ 325, 442 — quiet-corona n_e/T_e empirical model
                                │
1993 ──── Yohkoh launched (X-ray imaging spectrometer)
                                │
1995 ──── ★ SOHO launched — UVCS + LASCO + EIT
              Kohl+ 1995 — UVCS instrument paper
                                │
1997 ──── Kohl+, Sol. Phys. 175 — first UVCS coronal-hole T_e
              Dere+ 1997 — CHIANTI v1 atomic database (used here!)
                                │
1999 ──── Gibson+, J. Geophys. Res. 104 — equatorial streamer n_e + T_e
                                │
2003 ──── Frazin+ — LASCO/C2 tomographic n_e
              Vásquez+ — equatorial T_e profile from LASCO
                                │
2007–08 ── Spadaro+, Susino+ — mid-latitude streamer + boundary T_e
                                │
2010 ──── SDO launched — AIA + HMI
                                │
2010–18 ── Habbal+, Boe+ — eclipse Fe X/XI/XIV freeze-in distances
                                │
2018 ──── Parker Solar Probe launched
              Del Zanna & Mason — LRSP review on UV/X-ray spectral diagnostics
                                │
2020 ──── Solar Orbiter launched (ESA-NASA)
              Antonucci+ — Metis instrument paper (#57 in this archive)
              Rochus+ — EUI instrument paper
                                │
2021 ──── Mar 21: ★ DATA OF THIS PAPER (Solar Orbiter cruise phase)
              Romoli+ — Metis radiometric calibration
              Andretta+ — Lyα Doppler dimming
              Antonucci+ — first Metis science results
              Gopalswamy+ — BITSE narrow-band T_e at 4 R☉
                                │
2023 ──── Auchère+ — FSI coronagraphic mode description (decisive enabler)
              Boe+ — eclipse equatorial T_e at 1.4–2.8 R☉
              Dere+ — CHIANTI v10.1 (used in R(T_e) here)
                                │
2025 ──── ★ THIS PAPER — Abbo+, A&A 702, A254
              First Metis + FSI combined T_e at 4.25 R☉
                                │
20??─── Future:
              He II 304 nm channel addition → resolve cold/hot
              Vigil, PUNCH, MUSE — multi-band imaging missions
              ML-based inversion / Bayesian T_e estimation
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **#57 Antonucci+ 2020** (Metis instrument, A&A 642, A10) | **Direct data source / 직접적 자료 출처**. The Metis pB and Lyα channels described in this paper produce all VL/UV input to #61's pipeline. / 본 논문의 모든 VL/UV 입력 자료를 만든 기기 논문. | ★★★★★ Required prerequisite / 필수 선행 |
| **van de Hulst 1950** (BAN 11, 135) | **Method foundation / 방법론 기반**. Eq. (1) of #61 IS this paper's inversion formula — used directly to produce n_e(r). / 본 논문 식 (1)의 출처 — n_e(r) 도출에 직접 사용. | ★★★★★ Foundational / 기초 |
| **Rochus+ 2020** (EUI instrument, A&A 642, A8) | **Direct data source / 직접적 자료 출처**. Describes the FSI 17.4 nm channel; coronagraphic mode (Auchère+ 2023a) is the enabler. / FSI 17.4 nm 채널과 coronagraphic mode의 기본 기기 논문. | ★★★★★ Required prerequisite / 필수 선행 |
| **Auchère+ 2023a** (A&A 674, A127) | **Operational enabler / 운영 가능화**. Describes how FSI is used in coronagraphic mode — without this, EUV imaging at 1.85–4.45 R☉ would not exist. / 본 논문의 EUV 외부 코로나 영상 가능화의 운영적 토대. | ★★★★★ Critical / 핵심 |
| **Dere+ 1997 / Dere+ 2023** (CHIANTI) | **Atomic data source / 원자 데이터 출처**. The R(T_e) response function in Eq. (4) is computed from CHIANTI v10.1 — ionization balance, collisional excitation rates, and line wavelengths/strengths all from CHIANTI. / R(T_e) 응답함수 계산의 원자 데이터 모두 CHIANTI에서. | ★★★★ Computational backbone / 계산 골격 |
| **Kohl+ 1997** (UVCS, Sol. Phys. 175) | **Predecessor diagnostic / 선행 진단**. UVCS/SOHO measured T_e at < 3.5 R☉ via UV spectropolarimetry; #61 picks up where UVCS left off, using imaging instead. / UVCS가 한 일을 영상으로 이어받음. | ★★★★ Historical context / 역사적 맥락 |
| **Romoli+ 2021** (Metis first light, A&A 656, A32) | **Calibration foundation / 보정 토대**. The Metis pB radiometric calibration used in #61. / 본 논문의 Metis 보정 파이프라인의 기초. | ★★★ Calibration / 보정 |
| **#59 Del Zanna & Mason 2018** (LRSP 13, 5) | **Methodological review / 방법론 리뷰**. Comprehensive review of EM, contribution functions, and CHIANTI diagnostics — provides the textbook background for #61. / EM, contribution function, CHIANTI 진단의 교과서적 배경 제공. | ★★★★ Methodological context / 방법론 맥락 |
| **Withbroe 1988 / Hayes+ 2001 / Frazin+ 2003 / Vásquez+ 2003** | **n_e validation / n_e 검증**. #61 Fig 5 compares its n_e against these models — agreement validates the inversion. / 본 논문 n_e 결과의 검증 비교 대상. | ★★★ Cross-check / 교차검증 |
| **Spadaro+ 2007 / Susino+ 2008** | **T_e literature comparison / T_e 문헌 비교**. The cold-solution (~5 × 10⁵ K) of #61 matches these mid-latitude streamer profiles. / cold 해 (~5 × 10⁵ K) 와 일치. | ★★★ Cross-check / 교차검증 |
| **Habbal+ 2021 / Boe+ 2018, 2023** | **Eclipse complementary / 일식 상보**. Eclipse-based Fe ion freeze-in distances inform the §5 discussion of ionization-equilibrium boundary. / 이온화 평형 가정 검증의 외부 증거. | ★★ Discussion / 논의 |
| **#36 Müller+ 2020** (Solar Orbiter mission) | **Mission context / 임무 맥락**. The 0.68 AU vantage point used here is enabled by Solar Orbiter's near-Sun orbit. / 본 논문 관측 기하의 토대인 임무. | ★★★ Mission context / 임무 맥락 |
| **Guennou+ 2012** (loci method) | **Future extension / 향후 확장**. Multi-band loci method that #61 §5.4 proposes adding (with He II 304 nm) to break cold/hot ambiguity. / 304 nm 추가 시 cold/hot 모호성 해결을 위한 방법론 출처. | ★★ Future direction / 향후 방향 |

---

## 7. References / 참고문헌

### Primary instrument papers / 1차 기기 논문
- Müller, D., et al. *The Solar Orbiter mission: Science overview*. **A&A** 642, A1 (2020). DOI: 10.1051/0004-6361/202038467
- Antonucci, E., et al. *Metis: the Solar Orbiter visible light and ultraviolet coronal imager*. **A&A** 642, A10 (2020). DOI: 10.1051/0004-6361/201935338
- Rochus, P., et al. *The Solar Orbiter EUI instrument: The Extreme Ultraviolet Imager*. **A&A** 642, A8 (2020). DOI: 10.1051/0004-6361/201936663
- Fineschi, S., Naletto, G., Romoli, M., et al. *Optical design of the Multi-Element Telescope for Imaging and Spectroscopy (METIS) coronagraph on the Solar Orbiter mission*. **Exp. Astron.** 49, 239 (2020).

### Calibration & operational mode / 보정·운영 모드
- Auchère, F., Berghmans, D., Dumesnil, C., et al. *EUI/FSI coronagraphic operations*. **A&A** 674, A127 (2023a). DOI: 10.1051/0004-6361/202244665
- Auchère, F., Soubrié, E., Pelouze, G., & Buchlin, É. *Wavelet-Optimized Whitening filter (WOW)*. **A&A** 670, A66 (2023b).
- Romoli, M., Antonucci, E., Andretta, V., et al. *First light of Metis*. **A&A** 656, A32 (2021). DOI: 10.1051/0004-6361/202140980
- Andretta, V., De Leo, Y., Telloni, D., et al. *Lyα Doppler dimming with Metis*. **A&A** 656, L14 (2021).
- De Leo, Y., et al. *Metis radiometric calibration*. **A&A** 676, A45 (2023); **A&A** 697, A73 (2025).
- Kraaikamp, E., et al. *SolO/EUI Data Release 6 2023-01*. ROB (2023). DOI: 10.24414/z818-4163

### Foundational physics / 기초 물리
- van de Hulst, H. C. *The electron density of the solar corona*. **Bull. Astron. Inst. Netherl.** 11, 135 (1950).
- Gabriel, A. H. *Resonance scattering*. **Sol. Phys.** 21, 392 (1971).
- Withbroe, G. L., Kohl, J. L., Weiser, H., & Munro, R. H. *Probing the solar wind acceleration*. **Space Sci. Rev.** 33, 17 (1982).
- Noci, G., Kohl, J. L., & Withbroe, G. L. *Doppler dimming*. **ApJ** 315, 706 (1987).

### CHIANTI / 원자 데이터베이스
- Dere, K. P., Landi, E., Mason, H. E., Monsignori Fossi, B. C., & Young, P. R. *CHIANTI — an atomic database for emission lines I*. **A&AS** 125, 149 (1997). DOI: 10.1051/aas:1997368
- Dere, K. P., Del Zanna, G., Young, P. R., & Landi, E. *CHIANTI XVII. Version 10.1*. **ApJS** 268, 52 (2023). DOI: 10.3847/1538-4365/acec79
- Del Zanna, G., Dere, K. P., Young, P. R., & Landi, E. *CHIANTI v10*. **ApJ** 909, 38 (2021).

### Predecessor diagnostics / 선행 진단
- Kohl, J. L., Esser, R., Gardner, L. D., et al. *UVCS instrument*. **Sol. Phys.** 162, 313 (1995); **Sol. Phys.** 175, 613 (1997).
- Munro, R. H. & Jackson, B. V. *Coronal n_e from pB*. **ApJ** 213, 874 (1977).
- Withbroe, G. L. *Quiet corona empirical model*. **ApJ** 325, 442 (1988).
- Gibson, S. E., Fludra, A., Bagenal, F., et al. *Equatorial streamer*. **J. Geophys. Res.** 104, 9691 (1999).
- Hayes, A. P., Vourlidas, A., & Howard, R. A. *LASCO/C2 streamer profiles*. **ApJ** 548, 1081 (2001).
- Frazin, R. A., Cranmer, S. R., & Kohl, J. L. *Tomographic streamer densities*. **ApJ** 597, 1145 (2003).
- Vásquez, A. M., van Ballegooijen, A. A., & Raymond, J. C. *Equatorial T_e*. **ApJ** 598, 1361 (2003).
- Spadaro, D., Susino, R., Ventura, R., et al. *Mid-latitude streamer T_e*. **A&A** 475, 707 (2007).
- Susino, R., Ventura, R., Spadaro, D., Vourlidas, A., & Landi, E. *Streamer/coronal-hole boundary*. **A&A** 488, 303 (2008).

### Eclipse and freeze-in studies / 일식 및 freeze-in 연구
- Habbal, S. R., Druckmüller, M., Morgan, H., et al. *Eclipse Fe ion freeze-in*. **ApJ** 708, 1650 (2010); **ApJ** 734, 120 (2011); **ApJ** 911, L4 (2021).
- Boe, B., Habbal, S. R., Downs, C., & Druckmüller, M. *Freeze-in distances*. **ApJ** 859, 155 (2018); **ApJ** 935, 173 (2022); **ApJ** 951, 55 (2023).
- Del Zanna, G., Samra, J., Monaghan, A., et al. *Eclipse T_e*. **ApJ** 909, 38 (2021).
- Gopalswamy, N., Newmark, J., Yashiro, S., et al. *BITSE narrow-band T_e*. **Sol. Phys.** 296, 15 (2021).

### Methodology and validation / 방법론과 검증
- Schrijver, C. J. & McMullen, R. *Resonant scattering vs collisional excitation*. **ApJ** 531, 1121 (2000).
- Mikić, Z., Downs, C., Linker, J. A., et al. *PSI MAS 3D MHD model*. **Nat. Astron.** 2, 913 (2018).
- Scherrer, P. H., et al. *HMI*. **Sol. Phys.** 275, 207 (2012).
- Asplund, M., Amarsi, A. M., & Grevesse, N. *Photospheric abundances*. **A&A** 653, A141 (2021).
- Lemaire, P. & Stegen, K. *Hydrodynamic streamer model*. **Sol. Phys.** 291, 3659 (2016).
- Guennou, C., Auchère, F., Soubrié, E., et al. *Multi-band loci method*. **ApJS** 203, 25 (2012).

### This paper / 이 논문
- Abbo, L., Susino, R., Parenti, S., Auchère, F., Andretta, V., Spadaro, D., Romoli, M., Fineschi, S., et al. *Combined coronal observations of the streamer belt with Metis and EUI instruments on Solar Orbiter*. **A&A** 702, A254 (2025). DOI: 10.1051/0004-6361/202347599
