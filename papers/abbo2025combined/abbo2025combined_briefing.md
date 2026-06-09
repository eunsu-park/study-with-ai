---
title: "Pre-Reading Briefing: Combined coronal observations of the streamer belt with Metis and EUI instruments on Solar Orbiter"
paper_id: "61_abbo_2025"
topic: Solar_Observation
date: 2026-04-28
type: briefing
---

# Combined coronal observations of the streamer belt with Metis and EUI instruments on Solar Orbiter
## Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Abbo, L., Susino, R., Parenti, S., Auchère, F., Andretta, V., Spadaro, D., Romoli, M., Fineschi, S., et al. (2025). *Combined coronal observations of the streamer belt with Metis and EUI instruments on Solar Orbiter*. **Astronomy & Astrophysics, 702, A254.** DOI: [10.1051/0004-6361/202347599](https://doi.org/10.1051/0004-6361/202347599)
**First author**: L. Abbo (INAF – Astrophysical Observatory of Torino)
**Year**: 2025 (Received 28 July 2023 / Accepted 31 August 2025)

---

## 1. 핵심 기여 / Core Contribution

### 한국어
이 논문은 Solar Orbiter에 탑재된 두 관측기 — **Metis 코로나그래프** (가시광 편광 밝기 pB + UV HI Lyα) 와 **EUI/FSI** (Extreme Ultraviolet Imager / Full Sun Imager, 17.4 nm Fe IX/X 방출선) — 의 관측을 **처음으로 결합**하여, 외부 코로나(~4–4.5 R☉)에서 streamer belt의 **전자 온도 T_e**를 추정하는 새로운 방법을 제시한다. 핵심 아이디어는 다음과 같다:

1. **Metis pB**의 Thomson 산란 영상을 역산해 전자 밀도 n_e(r)을 얻고
2. 이를 시선(LOS) 적분해 **방출 측정량 EM = ∫ n_e² dx**를 계산하고
3. **FSI 17.4 nm 응답 함수 R(T_e, n_e)** 를 이용해 예상 카운트율을 T_e의 함수로 계산한 뒤
4. 측정된 FSI 카운트와 비교해 **T_e를 역산**한다.

응답 함수의 종형(bell-shaped) 모양 때문에 각 측정값은 **두 개의 해(cold/hot)** 를 갖는다. 2021년 3월 21일 관측에서 동/서 두 적도 streamer에 대해 cold ≈ 5.3–5.7 × 10⁵ K, hot ≈ 1.4 × 10⁶ K를 얻었으며, 기존 UVCS·LASCO·MLSO 기반 결과와 일관된다. 이는 외부 코로나의 T_e를 **분광관측 없이 다중대역 영상만으로** 진단하는 새로운 가능성을 보여준다.

### English
This paper presents the **first combined analysis** of two Solar Orbiter coronal instruments — the **Metis coronagraph** (visible-light polarized brightness pB + UV HI Lyα) and **EUI/FSI** (Extreme Ultraviolet Imager / Full Sun Imager, 17.4 nm Fe IX/X line) — to infer the **coronal electron temperature T_e** in the streamer belt at heliocentric heights of ~4–4.5 R☉. The methodology is:

1. Invert **Metis pB** Thomson-scattering imaging to derive electron density n_e(r);
2. Integrate along the line of sight (LOS) to obtain the **column emission measure EM = ∫ n_e² dx**;
3. Use the **FSI 17.4 nm response function R(T_e, n_e)** to compute expected count rates as a function of T_e;
4. Compare to the measured FSI counts to **infer T_e**.

Because of the bell-shaped FSI response, each measurement yields **two solutions (cold/hot)**. From observations on 21 March 2021, the eastern/western equatorial streamers give cold ≈ 5.3–5.7 × 10⁵ K and hot ≈ 1.4 × 10⁶ K, consistent with prior UVCS, LASCO, and MLSO-based estimates. The work demonstrates a novel **imaging-only diagnostic** of T_e in the outer corona, complementing traditional spectroscopic methods.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

#### 한국어
**외부 코로나 온도 측정의 난제.** 코로나의 전자 온도는 코로나 가열 메커니즘과 태양풍 가속을 이해하는 핵심 매개변수다. 그러나 ~3–8 R☉ 사이의 외부 코로나에서는 직접 측정이 매우 어렵다. 그 이유:

- **분광선 강도비(line ratio)** 방법은 충돌여기로 만들어진 UV 선이 필요하지만, ≳2 R☉부터는 너무 어둡다 (Raymond+ 1997, David+ 1998, Parenti+ 2000).
- **UVCS/SOHO** (1995–2012)는 분광편광 코로나그래프로 ~3.5 R☉까지 T_e를 측정했지만(Kohl+ 1995, 1997; Antonucci+ 2006), 이후 후속 분광 코로나그래프 임무가 없다.
- 대안으로는 코로나그래프 pB의 n_e 프로파일을 정수압 평형 또는 유체역학으로 가정해 T_e를 *간접* 추정한다 (Munro & Jackson 1977; Gibson+ 1999; Lemaire & Stegen 2016).

**Solar Orbiter의 등장.** 2020년 발사된 ESA-NASA의 Solar Orbiter는 0.28 AU까지 접근하며, 두 가지 코로나 영상기를 탑재했다 — **Metis** (가시광 + UV Lyα 코로나그래프, 1.7–9 R☉) 와 **EUI/FSI** (17.4 nm Fe IX/X 영상). 특히 EUI/FSI는 일반적으로 디스크 영상기지만 2021–2022년 **occulter 모드(coronagraphic mode)** 가 시험되어 ~7.4 R☉까지 EUV 영상을 가능케 했다 (Auchère+ 2023a). 이로써 처음으로 **VL + UV + EUV** 다중대역 외부 코로나 동시 관측이 가능해졌다.

이 논문은 그 가능성을 활용한 **최초의 결합 분석**이다.

#### English
**The challenge of outer-corona temperature.** Electron temperature in the corona is a key parameter for understanding heating and solar-wind acceleration. However, *direct* measurement between ~3 and 8 R☉ is very difficult:

- **Line-ratio diagnostics** require collisionally excited UV lines, which become too faint above ≳2 R☉ (Raymond+ 1997, David+ 1998, Parenti+ 2000).
- **UVCS/SOHO** (1995–2012) measured T_e up to ~3.5 R☉ via UV spectropolarimetry (Kohl+ 1995, 1997; Antonucci+ 2006), but no successor spectroscopic coronagraph has flown since.
- Alternatives derive T_e *indirectly* from coronagraph pB inversions of n_e under hydrostatic or hydrodynamic equilibrium (Munro & Jackson 1977; Gibson+ 1999; Lemaire & Stegen 2016).

**Enter Solar Orbiter.** Launched in 2020, the ESA–NASA Solar Orbiter approaches the Sun to 0.28 AU and carries two coronal imagers — **Metis** (VL + UV Lyα coronagraph, 1.7–9 R☉) and **EUI/FSI** (17.4 nm Fe IX/X imager). Crucially, FSI was operated in an **occulter (coronagraphic) mode** in 2021–2022 (Auchère+ 2023a), enabling EUV imaging up to ~7.4 R☉. For the first time, **VL + UV + EUV** simultaneous multi-band outer-corona observations are possible.

This paper exploits that capability with the **first combined analysis**.

### 타임라인 / Timeline

```
1950 ──── van de Hulst publishes pB inversion formula for K-corona n_e
             │
1971 ──── Gabriel — HI Lyα resonant scattering theory in the corona
             │
1977 ──── Munro & Jackson — first systematic n_e from pB / hydrostatic T_e
             │
1995 ──── SOHO launched / Kohl+ UVCS instrument paper
             │
1997 ──── Kohl+ first UVCS T_e in coronal holes (line ratios)
             │
1999 ──── Gibson+ MLSO/LASCO-based n_e and T_e of streamer belt
             │
2003 ──── Vásquez+ tomographic n_e + T_e from LASCO/C2
             │
2018 ──── Boe+ first eclipse Fe^10+/Fe^13+ freeze-in distance estimates
             │
2020 ──── Solar Orbiter launched (ESA/NASA)
                Antonucci+ Metis instrument paper (A&A 642, A10)
                Rochus+ EUI instrument paper (A&A 642, A8)
             │
2021 ──── March 21: Combined Metis + EUI/FSI coronagraphic-mode observations
                used in this paper
                Romoli+ Metis radiometric calibration; Andretta+ Lyα
             │
2023 ──── Auchère+ describes FSI coronagraphic mode (A&A 674, A127)
             │
2025 ──── ★ This paper: Abbo+ first combined Metis + FSI T_e in streamers
```

---

## 3. 필요한 배경 지식 / Prerequisites

### 한국어

**필수 (반드시 알아야 함):**
1. **Thomson 산란과 K-corona** — 광구 가시광이 코로나 자유전자에 의해 편광 산란되어 만드는 K-corona. 강도는 n_e의 *선형* 함수.
2. **Polarized brightness (pB) 역산** — van de Hulst (1950)의 적분 공식. 구대칭 가정 하에 pB(ρ) → n_e(r). 이 논문 식 (1).
3. **광학적으로 얇은 EUV 방출선** — 코로나는 EUV에서 광학적으로 얇으므로 강도는 n_e²·G(T_e, n_e)의 LOS 적분. 이 논문 식 (2).
4. **방출 측정량 (Emission Measure)** — EM = ∫ n_e²(x) dx. 강도와 함께 T_e/n_e를 분리하는 핵심 진단.
5. **Contribution function G(T_e, n_e)** — 원자 데이터(이온화 평형 + 충돌여기율 + 분광선 발생 확률)로부터 계산. CHIANTI 데이터베이스 사용.
6. **이온화 평형 (Ionization equilibrium)** — 충돌 이온화율과 재결합율이 균형. Fe IX/X의 경우 log T_e ≈ 5.8–6.0에서 피크.

**유익 (있으면 좋음):**
7. **HI Lyα 공명 산란과 Doppler dimming** (Gabriel 1971; Hyder & Lites 1970; Withbroe+ 1982) — 이 논문에서는 비교용으로만 사용.
8. **Solar Orbiter 미션 개요** — 0.28 AU 근접, 황도면 외부 궤도 (Müller+ 2020).
9. **Metis 기기 구조** — VL pB (580–640 nm) + UV (HI Lyα 121.6 nm) 동시 관측 (Antonucci+ 2020b; #57 Solar_Observation reading list).
10. **EUI/FSI 코로나그래프 모드** — movable occulting disk를 사용해 stray light를 억제, 0.45 AU 이하에서만 가능 (Auchère+ 2023a; Rochus+ 2020).
11. **Streamer (helmet streamer)** — 닫힌 자기 구조로 적도면 위에 있는 밝은 코로나 광원. 느린 태양풍의 기원.

**선행 논문:**
- **#57** Antonucci+ 2020 — Metis 기기 논문 (필수)
- **#36** Müller+ 2020 — Solar Orbiter 미션 개요
- **#45** Rochus+ 2020 — EUI 기기 논문
- **#50** Howard+ 2020 — SoloHI (헬리오스피어 영상)

### English

**Required (must know):**
1. **Thomson scattering and the K-corona** — photospheric VL polarization-scattered by coronal free electrons; intensity is *linear* in n_e.
2. **Polarized-brightness (pB) inversion** — van de Hulst (1950) integral formula; under spherical symmetry, pB(ρ) → n_e(r). Eq. (1) of the paper.
3. **Optically thin EUV emission lines** — corona is optically thin in EUV, so intensity = LOS integral of n_e²·G(T_e, n_e). Eq. (2).
4. **Emission Measure (EM)** — EM = ∫ n_e²(x) dx; combined with intensity, separates T_e from n_e.
5. **Contribution function G(T_e, n_e)** — computed from atomic data (ionization balance + collisional excitation rates + branching ratios). Computed via the CHIANTI database.
6. **Ionization equilibrium** — balance of collisional ionization and recombination rates. For Fe IX/X, peaks near log T_e ≈ 5.8–6.0.

**Helpful (nice to have):**
7. **HI Lyα resonant scattering & Doppler dimming** (Gabriel 1971; Hyder & Lites 1970; Withbroe+ 1982) — used here only for comparison.
8. **Solar Orbiter mission overview** — perihelion 0.28 AU, out-of-ecliptic orbit (Müller+ 2020).
9. **Metis instrument design** — simultaneous VL pB (580–640 nm) and UV HI Lyα (121.6 nm) imaging (Antonucci+ 2020b; #57 in Solar_Observation list).
10. **EUI/FSI coronagraphic mode** — uses a movable occulting disk to suppress stray light; only feasible at S/C distances < 0.45 AU (Auchère+ 2023a; Rochus+ 2020).
11. **(Helmet) streamers** — bright closed magnetic structures over the equatorial belt, the source of the slow solar wind.

**Prerequisite papers:**
- **#57** Antonucci+ 2020 — Metis instrument paper (essential)
- **#36** Müller+ 2020 — Solar Orbiter mission overview
- **#45** Rochus+ 2020 — EUI instrument paper
- **#50** Howard+ 2020 — SoloHI (heliospheric imager)

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **pB (Polarized Brightness)** | 코로나의 K-corona 편광 밝기 성분. Thomson 산란 강도 — 광학적으로 얇아 n_e에 선형. Stokes Q,U로부터 추출. / Polarized brightness from K-corona Thomson scattering — optically thin, linear in n_e; extracted from Stokes Q,U. |
| **K-corona / F-corona** | K = 자유전자 산란(편광됨, n_e 진단), F = 황도면 먼지 산란(비편광, 외삽으로 차감). / K = electron-scattered (polarized, traces n_e), F = dust-scattered zodiacal light (unpolarized, subtracted). |
| **Coronagraphic mode** | 디스크 영상이 아닌 occulting disk로 광구를 차단해 약한 코로나 신호를 노출시키는 관측 모드. / Imaging mode using an occulting disk to block the photospheric disk and reveal faint corona. |
| **EM (Emission Measure)** | EM = ∫ n_e² dx (단위 cm⁻⁵). 충돌여기 EUV 강도가 LOS의 n_e² 분포를 누적한 양. / Column EM = ∫ n_e² dx (cm⁻⁵). Sum of n_e² along LOS that drives collisional EUV emissivity. |
| **Contribution function G(T_e, n_e)** | 단위 EM당 분광선 방출률. 이온화 평형, 충돌여기, 분광선 발생 확률을 모두 포함. CHIANTI로 계산. / Line emissivity per unit EM, combining ionization balance, excitation rates, and atomic transition probabilities. Computed from CHIANTI. |
| **Response function R(T_e, n_e)** | 기기 응답까지 포함한 G(T_e, n_e) — 대역의 모든 분광선과 검출기 효율 가중. 식 (4)의 핵심. / Instrument-weighted G(T_e, n_e), summing all spectral lines in the passband × detector efficiency. The core of Eq. (4). |
| **Streamer belt** | 적도면 주변의 닫힌 자기루프 구조로 만들어진 밝은 코로나 영역. 느린 태양풍의 발원지. / Bright corona above closed magnetic loops near the equator; source of slow solar wind. |
| **Heliocentric distance ρ vs r** | ρ = 천구면(POS)에서의 충격 매개변수, r = 실제 3D 반경. 적분 시 r²−ρ²의 변환 필요. / ρ = plane-of-sky impact parameter, r = true 3D heliocentric radius. Integral conversion uses r²−ρ². |
| **Loci method** | EM curves (다중 대역)을 (T_e, EM) 평면에 그려 교차점을 T_e로 추정. 이 논문은 변형판 사용. / Plot EM curves from multiple bands in (T_e, EM) plane; intersection = T_e. This paper uses a variant. |
| **Ionization equilibrium** | 이온의 생성/소멸이 평형. 만약 태양풍 흐름이 빨라 시간 척도가 expansion보다 길면 "frozen-in" 상태로 깨짐. / Balance of ion production/loss; broken into "frozen-in" state if solar-wind expansion is faster than ionization timescales. |
| **Cold / Hot solution** | FSI 응답 함수가 종형이라 한 강도값에 두 T_e 해 (저온측·고온측)가 존재. / Bell-shaped FSI response yields two T_e solutions (lower/higher branch) per measured intensity. |
| **Doppler dimming** | 산란 이온이 광원 방향으로 빠르게 흐르면 공명 흡수 단면적이 줄어 산란강도가 감소. 태양풍 속도 진단. / Fast outflowing scatterers lose resonant absorption efficiency, reducing intensity; used as a solar-wind speed diagnostic. |
| **WOW filter** | Wavelet-Optimized Whitening filter (Auchère+ 2023b). 코로나 미세구조 강조용 영상처리. / Wavelet-optimized whitening image-enhancement filter for coronal fine structure. |
| **PSI MAS model** | Predictive Science Inc.의 3D MHD 솔라 코로나 모델. HMI 자기지도를 경계조건으로 사용. / 3D MHD coronal model from PSI, with HMI synoptic field as the photospheric boundary. |

---

## 5. 수식 미리보기 / Equations Preview

### Eq. (1) — Polarized Brightness from Thomson scattering / 편광 밝기

$$
I_\mathrm{pB} \propto \int_\rho^\infty n_e(r)\, [A(r) - B(r)]\, \frac{\rho^2\, dr}{r\sqrt{r^2 - \rho^2}}
$$

**의미 / Meaning**: 천구면(POS)에서 충격 매개변수 ρ인 시선을 따라 적분. A(r), B(r)은 Thomson 산란의 위상함수 기하인자 (van de Hulst 1950). 이 식의 *역산*으로 측정 pB(ρ)에서 3D n_e(r)를 얻는다 — 단, 구대칭 가정이 필요. 적도 streamer 근방에서는 충분히 적합.

**Meaning**: LOS integral at impact parameter ρ on the plane of the sky. A(r), B(r) are Thomson-scattering phase-function geometric factors (van de Hulst 1950). *Inverting* this gives 3D n_e(r) from measured pB(ρ), under spherical symmetry — adequate near equatorial streamers.

### Eq. (2) — EUV line emissivity / EUV 분광선 방출률

$$
I = \int_\mathrm{l.o.s.} A_\mathrm{Fe}\, G(T_e, n_e)\, n_e^2\, dx
$$

**의미 / Meaning**: 광학적으로 얇은 코로나에서 분광선 강도. A_Fe는 철 원소 abundance. G(T_e, n_e)는 contribution function — 이온화 평형(Fe IX/X의 분율)과 충돌여기율을 결합. n_e²의 의존성이 EM과 연결되는 출발점.

**Meaning**: Optically thin coronal line intensity. A_Fe = iron abundance. G(T_e, n_e) is the contribution function combining the ionic fraction (from ionization balance) and collisional excitation rate. The n_e² dependence is what couples this to EM.

### Eq. (3) — Column emission measure / 시선 방출 측정량

$$
\mathrm{EM} = \int_\mathrm{l.o.s.} n_e^2(x)\, dx
$$

**의미 / Meaning**: Metis pB로 얻은 n_e(r)을 plane-of-sky 점에서 시선 방향 ±10 R☉까지 적분. 4.25 R☉에서는 ±10 R☉ 적분이 무한대 적분의 ≥99%를 포착함을 검증. 이 EM이 EUV 측정과 연결되는 다리.

**Meaning**: Integrate n_e(r) from Metis along ±10 R☉ around the POS. At 4.25 R☉, this captures ≥99 % of the asymptotic integral. EM is the bridge between Metis and EUV.

### Eq. (4) — FSI count rate as a T_e function / FSI 카운트율

$$
C_\mathrm{FSI} = \frac{1}{4\pi}\int_\mathrm{l.o.s.} R(T_e, n_e)\, n_e^2\, dx \;\approx\; \frac{1}{4\pi}\, R(T_e)\, \mathrm{EM}
$$

**의미 / Meaning**: 핵심 공식. R(T_e, n_e)는 17.4 nm 대역 모든 분광선의 contribution function, 철 abundance, FSI 분광응답을 결합한 *기기 응답함수*. R이 n_e에 약하게 의존한다는 점을 활용해 R(T_e) ≈ R(T_e)로 인수화하면 — 측정 C_FSI와 EM(이미 Metis로 알려짐) 으로부터 T_e를 직접 역산할 수 있다. R(T_e)가 종형이므로 두 해(cold/hot)가 나옴.

**Meaning**: The key formula. R(T_e, n_e) is the *instrument response* combining contribution functions of all 17.4 nm lines, the iron abundance, and FSI's spectral response. Since R depends only weakly on n_e, factoring it out gives a direct inversion of T_e from the measured C_FSI and the already-known EM (from Metis). The bell-shape of R(T_e) yields two solutions (cold/hot).

### Auxiliary — Ionization timescale / 이온화 시간 척도

$$
\tau_\mathrm{ion} = \frac{1}{n_e\, \alpha_\mathrm{ion}}, \qquad \tau_\mathrm{rec} = \frac{1}{n_e\, \alpha_\mathrm{rec}}, \qquad \tau_\mathrm{exp} = \left[\frac{v}{n_e}\frac{dn_e}{dr}\right]^{-1}
$$

**의미 / Meaning**: 이온화 평형 가정의 타당성 검증용. τ_exp (확장 시간)이 τ_ion·τ_rec보다 짧으면 frozen-in 상태가 되어 가정이 깨진다. 4.25 R☉ 근처에서는 비교 가능 — 이 논문은 *경계 조건*에서 작동한다.

**Meaning**: Used to test the ionization-equilibrium assumption. When τ_exp (expansion) is shorter than τ_ion·τ_rec, the plasma becomes frozen-in and equilibrium fails. Near 4.25 R☉ they are comparable — the method operates *at the limit* of validity.

---

## 6. 읽기 가이드 / Reading Guide

### 한국어

**전체 길이**: 10페이지. 비교적 짧은 method paper. 30분~1시간 정도면 1차 통독 가능.

**섹션별 추천 읽기 순서:**

1. **Abstract → §1 Introduction (pp. 1–2)**: Context-Aims-Methods-Results-Conclusions 흐름이 abstract에 잘 나와 있다. Introduction은 외부 코로나 T_e 측정의 역사적 난제와 Solar Orbiter의 특수성을 설명. **여기서 "왜 이 결합이 새로운가?"를 잡고 가자.**

2. **§2 Metis and EUI/FSI observations (p. 2)**: 데이터 셋과 관측 모드 설명. 핵심: 3월 21일 2021년 관측, 0.68 AU 거리, FSI coronagraphic mode (~1.85–4.45 R☉ 범위), Metis pB+Lyα (~4 R☉ 이상). **Fig. 1, 2를 정독 — 두 기기의 FOV 중첩 영역이 이 분석의 무대.**

3. **§3 Formation of the VL and EUV band emissions (pp. 3–4)**: 두 방출 메커니즘 설명. 식 (1), (2) 등장. Lyα는 비교용이며 *분석에서는 사용 안 함* — 이 점을 놓치지 말 것. **여기서 모든 물리가 들어 있다.**

4. **§4 Data analysis (pp. 4–7)**: 본론. 구조:
   - **§4.1 Electron density**: pB → n_e 역산. Fig. 5의 동/서 streamer 비교. 다른 문헌 모델과의 비교 (Withbroe 1988, Vásquez+ 2003 등)는 sanity check.
   - **§4.2 Emission measure and electron temperature**: 핵심 방법. Fig. 6 (top) EM 지도, Fig. 6 (bottom) 카운트율 vs T_e. **Fig. 6의 빨간 띠와 회색 띠 교차점이 cold/hot 해.** Fig. 7은 streamer 중심으로부터 각거리 vs T_e — *streamer 내부에서 거의 균일*한 점에 주목.

5. **§5 Discussion (pp. 8–9)**: 가정의 한계 검증.
   - **Fig. 8**: 다른 문헌 T_e 결과와 비교. cold 해는 Spadaro+ 2007, Susino+ 2008과 일치, hot 해는 더 일반적 streamer 값과 일치.
   - **Fig. 9**: 이온화/재결합 시간 vs 확장 시간. 결정적 그림 — *이 방법이 적용 가능한 경계 조건*을 시각화.
   - 두 가지 핵심 가정 — (1) 이온화 평형, (2) 등온 LOS — 가 어디서 깨질 수 있는지 솔직하게 논의.

6. **References (p. 10)**: 외부 코로나 진단의 *교과서*. UVCS (Kohl+), pB 역산 (van de Hulst 1950), CHIANTI (Dere+ 1997, 2023), 일식 진단 (Habbal+, Boe+) 등 모두 포함.

**시간 절약 팁:**
- §3의 모든 식을 처음에는 *유도하지 말고 인수의 의미만* 파악하라. 노트 작성 단계에서 다시 정리.
- Fig. 4, 5의 비교 그림은 *방법의 sanity check*이지 새 결과가 아니므로 빠르게.
- §5의 freeze-in 이온 논의(Boe+ 2018)는 별도 깊은 주제 — 1차 통독에서는 결론만.

**주목할 디테일:**
- Section 4.2의 "two values of iron abundance" — Asplund+ 2021 photospheric vs CHIANTI default(FIP factor 10⁰·⁵). 이것이 ε bands의 폭을 결정.
- Fig. 6 bottom panels의 회색 띠 (관측 카운트)가 빨간 띠(예상 카운트 vs T_e) 의 *상승부와 하강부 모두*를 가로지르는지 확인 — 이것이 두 해의 기하학적 해석.

### English

**Total length**: 10 pages. A relatively short methods paper. 30 min – 1 h for a first read.

**Recommended reading order:**

1. **Abstract → §1 Introduction (pp. 1–2)**: The Context–Aims–Methods–Results–Conclusions structure is clear in the abstract. The introduction lays out why outer-corona T_e is hard to measure and why Solar Orbiter is special. **Make sure to grasp "what is new about this combination?"**

2. **§2 Metis and EUI/FSI observations (p. 2)**: Dataset and observing modes. Key facts: March 21 2021, 0.68 AU, FSI coronagraphic mode (~1.85–4.45 R☉), Metis pB + Lyα (~ ≥4 R☉). **Read Fig. 1, 2 carefully — the FOV overlap is the stage of the whole analysis.**

3. **§3 Formation of the VL and EUV band emissions (pp. 3–4)**: Two emission mechanisms; Eqs. (1), (2). Note that Lyα is used *only* for comparison and *not* in the inversion. **All the physics is here.**

4. **§4 Data analysis (pp. 4–7)**: The core.
   - **§4.1 Electron density**: pB → n_e inversion. Fig. 5 compares east/west streamers; comparisons with literature models (Withbroe 1988, Vásquez+ 2003, etc.) are a sanity check.
   - **§4.2 Emission measure and electron temperature**: The core method. Fig. 6 (top) EM map; Fig. 6 (bottom) count rate vs T_e. **The intersection of red and grey bands in Fig. 6 are the cold/hot solutions.** Fig. 7 plots T_e vs angular distance from the streamer center — note the *near-uniformity inside the streamer*.

5. **§5 Discussion (pp. 8–9)**: Honesty about the assumptions.
   - **Fig. 8**: T_e compared with literature. Cold solutions match Spadaro+ 2007 and Susino+ 2008; hot solutions match more typical streamer values.
   - **Fig. 9**: Ionization/recombination timescales vs expansion time — the *decisive* figure showing the boundary of validity.
   - The two key assumptions — (1) ionization equilibrium, (2) isothermal LOS — and where they may break.

6. **References (p. 10)**: A textbook list for outer-corona diagnostics — UVCS (Kohl+), pB inversion (van de Hulst 1950), CHIANTI (Dere+ 1997, 2023), eclipse diagnostics (Habbal+, Boe+).

**Time-saving tips:**
- For §3 equations, on first pass, don't derive — *just identify what each factor means*. Re-derive in the notes step.
- The comparison plots in Figs. 4, 5 are sanity checks, not new results — read quickly.
- The freeze-in discussion (Boe+ 2018) in §5 is a deep side topic — get the conclusion only on first pass.

**Details to watch for:**
- "Two values of iron abundance" in §4.2 — Asplund+ 2021 photospheric vs CHIANTI default (FIP factor 10⁰·⁵). This sets the width of the red bands.
- In the bottom panels of Fig. 6, check that the grey band (observed counts) crosses *both* the rising and falling branches of the red band (expected vs T_e) — this is the geometric meaning of the two solutions.

---

## 7. 현대적 의의 / Modern Significance

### 한국어

**(1) 외부 코로나 T_e 진단의 부활.**
UVCS 종료 (2012) 이후 ~3 R☉ 너머의 T_e 측정은 일식 관측 (Habbal+, Boe+) 외에는 정체였다. 이 논문은 **상시 우주 관측**으로 streamer T_e를 측정할 수 있는 길을 열었다. Solar Orbiter는 2026년대 후반까지 운영 예정이므로, 본 방법은 향후 수년간 *streamer 진화* 와 *태양 활동 주기* 를 따라 T_e를 추적하는 데 사용될 수 있다.

**(2) 다중대역 영상 시너지의 모범 사례.**
태양 물리학은 점점 *spectroscopy* (분광)에서 *narrow-band imaging* (협대역 영상) 로 무게 중심이 이동하고 있다 — DKIST/CRYO-NIRSP, ALMA, EUV imager arrays 등. 이 논문은 두 영상기 (VL + EUV) 의 결합으로 분광이 못 하는 영역에서도 진단이 가능함을 보여, 이 흐름의 *방법론적 시범*이 된다.

**(3) Solar wind 가속·기원 연구와의 연결.**
Streamer는 느린 태양풍의 주요 기원. T_e는 코로나 가열 과정과 풍 가속의 핵심 매개변수. PSP (Parker Solar Probe), Solar Orbiter, 향후 Vigil/PUNCH 등 차세대 임무가 streamer-내부 in situ 측정과 영상 결합 분석으로 가속 메커니즘을 푸는 데 이 진단은 직접 입력이 된다.

**(4) 머신러닝/역산 기법과의 결합 가능성.**
EM과 응답함수에 기반한 역산은 Bayesian inference, neural-network surrogate model 등으로 확장될 수 있다. He II 304 nm 추가 채널 (논문 §5에서 언급)을 사용하면 cold/hot 모호성 해결이 가능 — 차세대 임무 설계에 직접적 함의.

**(5) 우리 학습 트랙에서의 위치.**
Solar_Observation Phase 4 (Solar Orbiter SPICE/EUI)의 응용 사례. #57 Metis 기기 논문, #45 EUI 기기 논문이 *기기 자체*를 다루었다면, 본 논문은 *그것들을 사용해 무엇을 할 수 있는가* 의 첫 사례. 향후 streamer/태양풍 연구로 확장되는 가교.

### English

**(1) Revival of outer-corona T_e diagnostics.**
After UVCS ended (2012), measuring T_e beyond ~3 R☉ stagnated except for eclipse observations (Habbal+, Boe+). This paper opens a route to measure streamer T_e from **routine space observations**. Solar Orbiter operates into the late 2020s; this method can track *streamer evolution* and *solar-cycle dependence* of T_e for years.

**(2) A model for multi-band imaging synergy.**
Solar physics is shifting from *spectroscopy* to *narrow-band imaging* — DKIST/CRYO-NIRSP, ALMA, EUV imager arrays. This paper shows that combining two imagers (VL + EUV) can diagnose where spectroscopy cannot reach — a *methodological prototype* for the broader trend.

**(3) Connection to solar-wind acceleration / origin.**
Streamers are the main source of the slow solar wind. T_e is a key parameter for heating and wind acceleration. With PSP, Solar Orbiter, and upcoming Vigil/PUNCH-class missions performing in-situ + imaging studies of streamer interiors, this diagnostic is a direct input to acceleration-mechanism studies.

**(4) Potential synergy with ML / inversion techniques.**
EM- and response-function-based inversion naturally extends to Bayesian inference and neural-network surrogate models. Adding the He II 304 nm channel (mentioned in §5) would resolve the cold/hot ambiguity — a direct implication for next-generation mission design.

**(5) Position in our study track.**
An applications-of paper for Solar_Observation Phase 4 (Solar Orbiter SPICE/EUI). Whereas #57 (Metis) and #45 (EUI) describe the *instruments themselves*, this paper is the first example of *what we can do with them*. A bridge into streamer/solar-wind research.

---

## 사전 학습 권장 사항 / Pre-Reading Recommendations

### 한국어
1. **van de Hulst (1950)** 의 pB 역산 공식 - 식 (1)이 어디서 나오는지 1페이지 정도 읽고 시작 (다른 코로나그래프 논문과 공통).
2. **CHIANTI 데이터베이스** 개요 (Dere+ 1997, 2023) - contribution function이 어떻게 계산되는지 *블랙박스로 받아들이지 말 것*. 한 페이지짜리 overview면 충분.
3. **#57 Metis paper (Antonucci+ 2020)** 의 §2 (instrument design) 만이라도 다시 살펴 — 이 논문의 pB와 Lyα 데이터가 어떻게 만들어졌는지 머릿속 그림이 필요.
4. **EM 개념** — Mason & Monsignori Fossi 1994 또는 Del Zanna & Mason 2018 (Living Reviews) 의 EM 섹션을 5분 훑기.

### English
1. **van de Hulst (1950)** pB inversion — read ~1 page on where Eq. (1) comes from (common to most coronagraph papers).
2. **CHIANTI database** overview (Dere+ 1997, 2023) — don't treat the contribution function as a black box; a one-page overview is enough.
3. **#57 Metis paper (Antonucci+ 2020)** — at least re-skim §2 (instrument design) to picture how the pB and Lyα data of this paper are produced.
4. **EM concept** — 5-min skim of the EM section in Mason & Monsignori Fossi 1994 or Del Zanna & Mason 2018 (Living Reviews).

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
