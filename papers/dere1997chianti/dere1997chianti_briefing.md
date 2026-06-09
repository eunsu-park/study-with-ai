---
title: "Pre-Reading Briefing: CHIANTI — an atomic database for emission lines. I. Wavelengths greater than 50 Å"
paper_id: "63_dere_1997"
topic: Solar_Observation
date: 2026-04-28
type: briefing
---

# CHIANTI — an atomic database for emission lines. I. Wavelengths greater than 50 Å: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: K. P. Dere, E. Landi, H. E. Mason, B. C. Monsignori Fossi, and P. R. Young, "CHIANTI — an atomic database for emission lines. I. Wavelengths greater than 50 Å", *Astronomy and Astrophysics Supplement Series*, **125**, 149–173 (1997). DOI: 10.1051/aas:1997368
**Author(s)**: K. P. Dere (NRL), E. Landi (Florence), H. E. Mason (Cambridge DAMTP), B. C. Monsignori Fossi (Arcetri), P. R. Young (Cambridge DAMTP)
**Year**: 1997

---

## 1. 핵심 기여 / Core Contribution

### 한국어
이 논문은 **CHIANTI** (이탈리아 와인 이름에서 유래한 코드네임이자 지명) 원자 데이터베이스의 **창립 논문**이다. CHIANTI는 광학적으로 얇은(optically thin) 천체 플라즈마—태양 코로나, 천이영역, 항성 대기, 활동성 은하핵 등—의 EUV/UV/X-ray 방출선 스펙트럼을 합성(synthesize)하기 위해 필요한 **에너지 준위(energy levels), 파장(wavelengths), 복사 천이 확률(A 계수), 전자 충돌 여기율(collision strengths Ω, effective collision strengths Υ)**을 한 곳에 모은 종합 데이터베이스이다. 첫 버전은 50 Å 이상 파장 영역에서 ~80여 개 이온(H I부터 Ni XXVIII까지 우주 원소)을 포함하며, Burgess & Tully (1992)의 스케일링 방법으로 모든 충돌 강도 데이터를 시각적으로 검사·내삽 가능한 표준 형태로 저장한다. 또한 **IDL 루틴**(레벨 인구 계산, 합성 스펙트럼, 밀도·온도 진단)을 함께 배포하며, **익명 FTP**로 누구나 자유롭게 내려받아 사용·갱신할 수 있도록 설계되었다. 이 점은 1990년대 중반 천체분광학 연구의 재현성과 표준화를 한 단계 도약시켰다.

### English
This is the **founding paper** of the **CHIANTI** atomic database (named after the Tuscan wine and region, reflecting the Italy–UK–US collaboration). CHIANTI is a comprehensive database that bundles **energy levels, wavelengths, radiative transition probabilities (A values), and electron collision excitation data (collision strengths Ω, thermally averaged effective collision strengths Υ)** required to synthesize EUV/UV/X-ray emission line spectra of optically thin astrophysical plasmas — solar corona, transition region, stellar atmospheres, AGN, and so on. Version 1 covers wavelengths longer than 50 Å for ~80 ions (H I through Ni XXVIII for cosmically abundant elements), and stores every collision strength after applying the **Burgess & Tully (1992) scaling**, allowing visual inspection, interpolation, and proper extrapolation to threshold and infinite energy. The package ships with **IDL routines** (level populations, synthetic spectra, density/temperature diagnostics) and is freely distributed by **anonymous FTP**. This combination of curated data, a single transparent format, and open distribution was a watershed moment for reproducibility and standardisation in astrophysical spectroscopy.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

#### 한국어
1990년대 초중반은 태양·천체 분광학의 황금기였다. **Yohkoh** (1991, X-ray/SXT, BCS), **SOHO** (1995 발사, CDS·SUMER·UVCS) 임무와 **Hubble** (특히 GHRS, STIS), **EUVE** (1992)가 동시에 가동되며 EUV/UV/X-ray 방출선 데이터가 폭증했다. 그러나 이를 해석하기 위한 원자 데이터는 여러 곳에 흩어져 있었고, 코드별로 형식이 달라 결과를 상호 비교하기 어려웠다 — Landini & Monsignori Fossi (1970, 1990), Mewe (1972), Mewe & Gronenschild (1981), Raymond & Smith (1977), Kato (1976), Stern et al. (1978), Gaetz & Salpeter (1983) 등 다수의 spectral code가 존재했지만 각자 자체 원자 데이터를 사용했다. 동시에 R-matrix(QUB Belfast), Distorted Wave(UCL), Iron Project(Hummer et al. 1993), Bhatia & Doschek 등 충돌 산란 계산이 급격히 정확해지고 있었다. CHIANTI는 이 분산된 데이터를 **하나의 표준화된 IDL 친화적 파일 구조**로 모으고, 1992년 Abingdon에서 열린 **SOHO CDS/SUMER 워크숍**(편집: Lang 1994)에서 이뤄진 데이터 평가 작업을 기반으로 한다.

#### English
The early-to-mid 1990s was a golden era for solar and astrophysical spectroscopy. **Yohkoh** (1991, SXT/BCS), the **SOHO** mission (launched 1995, CDS/SUMER/UVCS), **Hubble** (notably GHRS, STIS), and **EUVE** (1992) were producing an explosion of EUV/UV/X-ray emission line data. However, the atomic data needed to interpret these observations were scattered across many sources, and each spectral code (Landini & Monsignori Fossi 1970/1990; Mewe 1972; Mewe & Gronenschild 1981; Raymond & Smith 1977; Kato 1976; Stern et al. 1978; Gaetz & Salpeter 1983; Arnaud & Rothenflug 1985 for ionisation balance) used its own internal atomic dataset, making cross-comparison difficult. Simultaneously, electron–ion scattering calculations were becoming dramatically more accurate via R-matrix (Belfast/QUB), Distorted Wave (UCL), the Iron Project (Hummer et al. 1993), and the Bhatia/Doschek/Sampson programmes. CHIANTI was conceived to **collect this scattered data in one standardised IDL-friendly file structure**, building on the data assessment performed at the **1992 Abingdon SOHO CDS/SUMER workshop** (proceedings edited by Lang, ADNDT 1994).

### 타임라인 / Timeline

```
1962 ── Van Regemorter: g-bar approximation for collision strengths
1970 ── Landini & Monsignori Fossi: first comprehensive coronal spectral code
1972 ── Mewe: spectral code with polynomial collision-strength fits
1981 ── Mewe & Gronenschild: extended Mewe formulation
1985 ── Arnaud & Rothenflug: standard ionisation equilibrium tables
1987 ── DARC (Dirac-Fock R-matrix) developed by Grant and co-workers
1991 ── Yohkoh launched (SXT, BCS)
1992 ── EUVE launched; Burgess & Tully publish their scaling method
1992 ── Abingdon SOHO CDS/SUMER atomic data workshop
1993 ── Iron Project consortium begins (Hummer et al.)
1994 ── ADNDT special volume on iso-electronic sequences (ed. Lang)
1995 ── NIST critical compilation of energy levels (Martin et al.)
1995 ── SOHO launched (CDS, SUMER, UVCS, EIT)
1996 ── Mason: comparative review of plasma emission codes
1997 ── ★ Dere et al.: CHIANTI v1 published — this paper
... ──
2003 ── CHIANTI v4 (Young et al.): proton rates, photoexcitation
2009 ── CHIANTI v6 (Dere et al.): full ionisation/recombination
2015 ── CHIANTI v8: improved Fe coronal lines for Hinode/EIS, IRIS
2021 ── CHIANTI v10 (Del Zanna et al.)
2023 ── CHIANTI v10.1 (Dere et al.) — used by paper #61 Abbo et al. 2025
```

---

## 3. 필요한 배경 지식 / Prerequisites

### 한국어
- **원자물리 기초**: 전자 배치(configuration), 항(term), 준위(level), LSJ 결합 vs. 중간 결합(IC), 미세 구조(fine structure), iso-electronic sequence(같은 전자 수 → 비슷한 구조).
- **분광학 표기**: H-like, He-like, Li-like ... 시리즈; Fe XII = Fe$^{11+}$ (스펙트럼 표기는 이온 단계 + 1).
- **방출 메커니즘**: 광학적으로 얇은 플라즈마, 충돌 여기 → 자발 복사 천이(spontaneous radiative decay), 통계 평형(statistical equilibrium).
- **기본 천이 확률 개념**: Einstein A 계수, 발진자 강도(oscillator strength) $f$, 가중 발진자 강도 $gf$, 분지비(branching ratio).
- **충돌 산란 이론 (개념 수준)**: Distorted Wave (DW), Coulomb-Bethe (CBe), Close-Coupling (CC), R-matrix; 공명(resonance)이 충돌 강도에 미치는 영향.
- **이온화 평형**: 정상상태(steady-state) 충돌-이온화 평형(coronal equilibrium), $N(X^{+m})/N(X)$.
- **수치 도구**: IDL 기본 사용법(SOHO 시대 표준 분석 언어), 익명 FTP/CDS 데이터 배포 개념.
- **선행 논문**: Mason & Monsignori Fossi (1994) — 이 논문이 따르는 방출선 형식주의의 정수.

### English
- **Atomic physics basics**: electron configuration, term, level, LSJ vs. intermediate coupling (IC), fine structure, iso-electronic sequence (ions with the same electron count share structure).
- **Spectroscopic notation**: H-like, He-like, Li-like sequences; Fe XII = Fe$^{11+}$ (spectroscopic numeral = charge + 1).
- **Emission mechanisms**: optically thin plasma, collisional excitation → spontaneous radiative decay, statistical equilibrium.
- **Transition probability concepts**: Einstein A coefficient, oscillator strength $f$, weighted oscillator strength $gf$, branching ratio.
- **Scattering theory (conceptual)**: Distorted Wave (DW), Coulomb-Bethe (CBe), Close-Coupling (CC), R-matrix; the role of resonances in enhancing collision strengths.
- **Ionisation equilibrium**: steady-state coronal (collisional) equilibrium, ion fractions $N(X^{+m})/N(X)$.
- **Tools**: basic IDL fluency (the lingua franca of the SOHO era), familiarity with anonymous FTP / CDS data distribution.
- **Companion reading**: Mason & Monsignori Fossi (1994) — the formalism this paper crystallises into a database.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **Atomic energy level** / 원자 에너지 준위 | An eigenstate of the bound-electron Hamiltonian, labelled by configuration + term + $J$. CHIANTI takes observed energies from NIST (Martin et al. 1995) and supplements with theoretical (SSTRUCT) values when needed. NIST 관측값을 우선 사용하고, 없으면 UCL SSTRUCT로 계산한 이론값으로 보완한다. |
| **A-coefficient (Einstein $A_{ji}$)** / 자발 복사 천이 확률 | Spontaneous radiative decay rate from upper level $j$ to lower level $i$ in s$^{-1}$. Determines line emissivity together with upper-level population. 윗준위 인구와 함께 방출선의 세기를 결정. CHIANTI stores E1, E2, M1, M2 transitions; transitions with branching ratio $<10^{-5}$ are pruned. |
| **Oscillator strength $f_{ij}$ / weighted $gf$** / 발진자 강도 | Dimensionless strength of an absorption line, related to $A$ via $A_{ji} = (8\pi^2 e^2 / m_e c \lambda^2)(g_i / g_j) f_{ij}$. CHIANTI stores $gf$ values for dipole transitions. |
| **Collision strength $\Omega_{ij}$** / 충돌 강도 | Symmetric, dimensionless quantity related to the electron impact excitation cross section by $\sigma_{ij} = \pi a_0^2 (\Omega_{ij}/\omega_i)/E$. Computed from theory (DW, CBe, CC, R-matrix). 대칭성: $\Omega_{ij} = \Omega_{ji}$. |
| **Effective collision strength $\Upsilon_{ij}(T_e)$** / 유효 충돌 강도 | Maxwellian average of $\Omega$ over electron velocity distribution at temperature $T_e$. The quantity actually needed for excitation rate coefficients in collisionally ionised plasmas. CHIANTI stores 5-point spline fits to scaled $\Upsilon$. |
| **Burgess & Tully (1992) scaling** / Burgess–Tully 스케일링 | A method to map $(E, \Omega)$ or $(T, \Upsilon)$ over the entire $[0, \infty)$ range onto a finite scaled interval $[0,1]$, using a transition-type-dependent transformation. Ensures correct high-energy/temperature limit (Bethe form for dipole) and allows visual quality control. CHIANTI의 모든 충돌 데이터를 5-point spline으로 압축 저장하는 핵심 수단. |
| **Ionisation fraction** / 이온화 비율 | $N(X^{+m})/N(X)$ as a function of $T_e$ in steady-state coronal equilibrium; CHIANTI v1 uses the Arnaud & Rothenflug (1985) calculation. |
| **Optically thin plasma** / 광학적으로 얇은 플라즈마 | A regime where emitted photons escape without re-absorption; line intensity is the volume integral of emissivity. Valid for most of the corona and transition region for EUV/X-ray lines. |
| **Iso-electronic sequence** / 동전자열 | Set of ions with the same number of electrons (e.g., H-like = 1 e$^-$). CHIANTI organises §4 of the paper by this principle because atomic structure scales smoothly along a sequence. |
| **Coronal (collisional) equilibrium** / 코로나 평형 | Low-density limit where collisional excitation rate $\gg$ ionisation/recombination rate, so level populations decouple from ionisation balance. Valid for $N_e \lesssim 10^{15}$ cm$^{-3}$ for the species in CHIANTI v1. |
| **Contribution function $G(T_e, n_e)$** / 기여 함수 | The product of ion fraction, level population, and atomic constants that maps a Differential Emission Measure (DEM) to a line intensity. The principal output of CHIANTI for instrument response calculations (e.g., FSI 17.4 nm in Abbo+ 2025). |
| **CHIANTI directory structure** / CHIANTI 디렉터리 구조 | One directory per ion (e.g., `fe_12/`) containing files for energy levels (`.elvlc`), wavelengths/A values (`.wgfa`), and 5-point spline fits to scaled effective collision strengths (`.splups`). Transparent ASCII format → easy update by anyone. |
| **IDL routines / SolarSoft** | Companion package supplying level-population solver, synthetic spectrum generator, density/temperature diagnostics, and integration with SOHO/CDS analysis (Pike & Del Zanna). 이후 SolarSoftWare(SSW)로 통합. |
| **BURLY** | The IDL routines for **B**urgess & T**u**lly scaling (`BURgess..tulLY`) — interactive graphical inspection of every collision data set. |

---

## 5. 수식 미리보기 / Equations Preview

### 5.1 광학적으로 얇은 단일 선의 emissivity / Optically thin line emissivity (Eq. 1)

$$
\boxed{\;\epsilon_{i,j} \;=\; N_j(X^{+m})\, A_{j,i}\, \frac{hc}{\lambda_{i,j}}\;}
\quad \text{[erg cm}^{-3}\text{ s}^{-1}\text{]}
$$

#### 한국어
- $\epsilon_{i,j}$: 단위 부피·시간당 방출 에너지 (단일 천이 $j \to i$).
- $N_j(X^{+m})$: 이온 $X^{+m}$의 윗준위 $j$에 있는 원자수 밀도 [cm$^{-3}$].
- $A_{j,i}$: Einstein 자발 복사 계수 [s$^{-1}$].
- $hc/\lambda_{i,j}$: 광자 한 개의 에너지.
- 광학적으로 얇은 플라즈마에서는 자기흡수 보정이 필요 없다.

#### English
- $\epsilon_{i,j}$ is the power emitted per unit volume in the single transition $j \to i$.
- $N_j$ is the upper-level population density of ion $X^{+m}$.
- $A_{j,i}$ is the spontaneous emission coefficient.
- $hc/\lambda$ is the photon energy.
- In the optically thin limit no self-absorption correction is required.

### 5.2 윗준위 인구의 인수분해 / Factorisation of upper-level population (Eq. 2)

$$
\boxed{\;N_j(X^{+m}) \;=\; \underbrace{\frac{N_j(X^{+m})}{N(X^{+m})}}_{\text{level pop. fraction}} \cdot \underbrace{\frac{N(X^{+m})}{N(X)}}_{\text{ion fraction}} \cdot \underbrace{\frac{N(X)}{N(H)}}_{\text{abundance}} \cdot \underbrace{\frac{N(H)}{N_e}}_{\sim 0.83} \cdot N_e\;}
$$

#### 한국어
이 분해는 CHIANTI 사용자 인터페이스의 철학을 보여준다. 각 인수가 별도의 모듈에서 계산된다.
1. **준위 인구 비율** — 통계 평형 풀이(다음 식). $T_e, N_e$의 함수.
2. **이온 비율** — Arnaud & Rothenflug 1985. $T_e$의 함수.
3. **원소 풍부도** — 사용자가 자유롭게 지정 (광구, 코로나 abundance set).
4. **수소/전자 비** — 약 0.83 (완전 이온화 H+He 플라즈마).

#### English
This factorisation reflects the modular philosophy of the CHIANTI interface: each factor is computed by a separate module.
1. **Level population fraction** — solved by statistical equilibrium (next equation); function of $T_e, N_e$.
2. **Ion fraction** — Arnaud & Rothenflug (1985) in v1; function of $T_e$.
3. **Element abundance** — user-selectable (photospheric, coronal, FIP-biased sets).
4. **Hydrogen-to-electron ratio** — $\sim 0.83$ for a fully ionised H+He plasma.

### 5.3 통계 평형 방정식 / Statistical equilibrium (Eq. 4)

$$
\boxed{\;
N_j \!\left[ N_e \!\sum_{i} C^{e}_{j,i} + N_p \!\sum_{i} C^{p}_{j,i} + \!\!\sum_{i>j} R_{j,i} + \!\!\sum_{i<j} A_{j,i} \right]
=
\sum_{i} N_i \!\left[ N_e C^{e}_{i,j} + N_p C^{p}_{i,j} \right] + \!\!\sum_{i>j} N_i A_{i,j} + \!\!\sum_{i<j} N_i R_{i,j}
\;}
$$

#### 한국어
좌변 = 준위 $j$를 떠나는 모든 비율, 우변 = 준위 $j$로 들어오는 모든 비율. 정상상태에서 좌·우변이 같다.
- $C^{e}, C^{p}$: 전자·양성자 충돌 여기/소거 율 계수 [cm$^3$ s$^{-1}$].
- $R_{j,i}$: 흑체에 의한 유도 흡수율 (대부분 천체에서 무시).
- $A_{j,i}$: 자발 복사.
- v1에서는 양성자 율은 미포함(곧 추가될 예정이라 명시) — fine structure 천이에서 중요.

#### English
LHS = total rate out of level $j$, RHS = total rate into level $j$; in steady state these balance.
- $C^{e}, C^{p}$: electron / proton collisional rate coefficients [cm$^3$ s$^{-1}$].
- $R_{j,i}$: stimulated absorption from the radiation field (negligible for most low-density plasmas).
- $A_{j,i}$: spontaneous radiation.
- v1 omits proton rates (slated for inclusion); these are important for fine-structure transitions.

### 5.4 Maxwellian-평균 충돌 여기율 / Maxwellian-averaged collision rate (Eqs. 5, 6)

$$
\boxed{\;
C^{e}_{i,j}(T_e) \;=\; \frac{8.63 \times 10^{-6}}{T_e^{1/2}}\; \frac{\Upsilon_{i,j}(T_e)}{\omega_i}\; \exp\!\left(-\frac{E_{i,j}}{k T_e}\right)
\quad [\text{cm}^3\,\text{s}^{-1}]\;}
$$

with the thermally averaged effective collision strength

$$
\boxed{\;\Upsilon_{i,j}(T_e) \;=\; \int_{0}^{\infty} \Omega_{i,j}\, \exp\!\left(-\frac{E_j}{k T_e}\right) \,\mathrm{d}\!\left(\frac{E_j}{k T_e}\right)\;}
$$

#### 한국어
- $\omega_i$: 아래준위 $i$의 통계적 가중치 ($2J_i+1$).
- $E_{i,j}$: 두 준위 사이 에너지 차.
- 계수 $8.63 \times 10^{-6}$는 $\sqrt{2\pi/m_e}$ 등에서 유래한 표준 수치.
- $\Omega_{i,j}$를 Maxwellian으로 적분해 얻는 $\Upsilon$가 핵심 데이터. CHIANTI는 이를 5-point spline으로 압축 저장.

#### English
- $\omega_i$: statistical weight of the lower level ($2J_i+1$).
- $E_{i,j}$: energy difference between the two levels.
- The constant $8.63 \times 10^{-6}$ comes from $\sqrt{2\pi \hbar^4/(k m_e^3)}$ collected with $a_0^2$.
- $\Upsilon$ is the integral of $\Omega$ over a Maxwellian; this is the quantity CHIANTI stores as a 5-point spline fit in `.splups` files.

### 5.5 Van Regemorter $\bar{g}$ 근사 / The g-bar approximation (Eq. 7)

$$
\Omega_{i,j} \;=\; \frac{8\pi}{\sqrt{3}}\, \omega_j\, f_{i,j}\, \frac{I_H}{E_{i,j}}\, \bar{g}
$$

#### 한국어
1962년 Van Regemorter의 반경험식으로, 발진자 강도 $f_{ij}$만 알면 충돌 강도를 추정할 수 있다. 그러나 정확도는 25% 정도(Younger & Wiese 1979)이며 forbidden, intercombination 천이에는 부적합. CHIANTI는 가능한 한 이 근사를 R-matrix/CC/DW 결과로 대체했다.

#### English
The 1962 semi-empirical formula relating $\Omega$ to the oscillator strength $f$ via an effective Gaunt factor $\bar{g}$ — Coulomb-Bethe based, accurate to $\sim 25\%$ for $\Delta n=0$ allowed transitions but unreliable for $\Delta n \neq 0$, forbidden, or intercombination transitions. CHIANTI replaces this with R-matrix/CC/DW calculations whenever possible.

### 5.6 Burgess & Tully 스케일링 / Burgess–Tully scaling (for dipole transitions)

For an allowed (dipole) transition,

$$
x \;=\; 1 - \frac{\ln C}{\ln(X - 1 + C)}, \qquad y \;=\; \Omega \, \ln(X - 1 + e),
$$

where $X = E/E_{\text{th}}$ is the impact energy in threshold units and $C$ is an adjustable scale parameter. The scaled energy $x \in [0,1]$ maps threshold $\to$ infinity onto a finite interval, and $y$ approaches the Bethe limit $4 \omega_i f_{i,j}/E_{i,j}$ at $x = 1$. This permits a 5-point spline to capture $\Omega$ over the full energy range.

#### 한국어
허용 천이를 예로 든 BT scaling: 입사 에너지 $X = E/E_{\text{th}}$ (threshold 단위)을 $x \in [0,1]$로 사상하고, 충돌 강도를 $y = \Omega \ln(X-1+e)$로 변환한다. $x=1$ (무한대 에너지)에서 $y$가 Bethe 극한값 $4\omega_i f_{i,j}/E_{i,j}$에 점근하므로, 누락된 high-energy/temperature 한계가 자동으로 보장된다. **5-point spline**으로 압축해도 $\Upsilon$ 평균 편차가 1% 미만(Mg X 예: 0.2%, N III: 0.5%).

#### English
For an allowed transition: scaled energy $x = 1 - \ln(C)/\ln(X - 1 + C) \in [0,1]$, and scaled collision strength $y = \Omega \ln(X - 1 + e)$. As $x \to 1$ ($E \to \infty$), $y \to 4\omega_i f_{i,j}/E_{i,j}$ (the Bethe limit), so the high-energy behaviour is correctly imposed. A 5-point spline reproduces the original $\Upsilon$ to better than 1% in CHIANTI's tests (e.g., 0.2% for Mg X, 0.5% for N III).

### 5.7 contribution function $G(T_e, n_e)$ — CHIANTI 사용자 출력 / The user-facing observable

Even though this paper does not write it down explicitly, the *contribution function* $G$ is the natural composite of equations (1) and (2):

$$
G(T_e, n_e) \;=\; \frac{hc}{\lambda} \, \frac{A_{j,i}\, N_j(X^{+m})/N(X^{+m})}{N_e} \, \frac{N(X^{+m})}{N(X)} \, \frac{N(X)}{N(H)} \, \frac{N(H)}{N_e}
$$

so that the line intensity along the line of sight reduces to

$$
I(\lambda_{i,j}) \;=\; \frac{1}{4\pi} \int G(T_e, n_e)\, n_e^2 \, \mathrm{d}\ell
\;=\; \frac{1}{4\pi} \int G(T_e)\, \mathrm{DEM}(T_e)\, \mathrm{d}T_e ,
$$

after re-arranging into the **Differential Emission Measure (DEM)** $\mathrm{DEM}(T_e) = n_e^2 \, \mathrm{d}\ell/\mathrm{d}T_e$.

#### 한국어
이 식이 바로 #61 Abbo+ 2025에서 FSI 17.4 nm response function $R(T_e)$를 만들 때 합산되는 모든 방출선의 핵심 항이다. CHIANTI의 IDL 함수 `g_of_t.pro`가 이를 계산한다.

#### English
This is precisely the per-line term that paper #61 (Abbo et al. 2025) sums when constructing the FSI 17.4 nm response function $R(T_e)$. The CHIANTI IDL function `g_of_t.pro` computes it.

---

## 6. 읽기 가이드 / Reading Guide

### 한국어
이 논문은 **참조용 데이터베이스 논문**이지 일반적인 의미의 실험·이론 논문이 아니다. 따라서 처음부터 끝까지 직선적으로 읽기보다는 다음 전략을 권한다.

1. **Abstract + §1 Introduction (반드시 정독)** — CHIANTI의 5가지 설계 목표 (업데이트 용이성, 배포 용이성, 사용자 투명성, 정확성 유지, 프로그램 개발에 적합한 데이터·프로그래밍 구조)와 핵심 기여를 파악.
2. **§2 Emission line intensities (반드시 정독, 수식 손으로 따라가기)** — 식 (1)–(4). 위 5절에서 미리 본 emissivity, level pop. factorisation, statistical equilibrium 식의 출처. 양성자 율은 v1에 미포함이라는 점, $N_e \lesssim 10^{15}$ cm$^{-3}$ 한계 등 적용 범위를 확인.
3. **§3 Atomic data (반드시 정독)** — 데이터의 출처와 평가 방법.
   - §3.1 에너지 준위 (NIST 우선, 없으면 SSTRUCT)
   - §3.2 복사 데이터 ($A$ 값, branching ratio $<10^{-5}$ 가지치기)
   - §3.3 전자 충돌 데이터 (DW, CBe, CC, R-matrix 비교; 식 5–7)
   - §3.4 충돌 데이터 평가 (Burgess–Tully 스케일링의 핵심)
4. **§4 The CHIANTI database (스킴)** — iso-electronic sequence별로 어떤 이온이 어떤 source에서 왔는지 나열. 본인이 관심 있는 이온(Fe IX–XIV가 SDO/AIA·EUI에 핵심)만 골라 읽는다. **§4.1 He II, §4.2 Li-like, §4.3 Be-like (특히 §4.3.1 C III), §4.5 B-like (Fe XXII)** 정도가 EUV에서 가장 자주 인용된다.
5. **Tables 1–2 (이온 목록)** — H부터 Ni까지 어떤 이온화 단계가 포함되었는지 한눈에 보자.
6. **Fig. 1 (Mg X 예시)** — Burgess–Tully 스케일링이 어떻게 raw data → scaled $\Omega, \Upsilon$로 변환되는지의 그림 예시. 한 번 천천히 살펴보자.
7. **(생략 가능)** §4의 모든 iso-electronic sequence를 다 읽을 필요 없음 — 필요할 때 reference로 돌아오면 된다.

#### 시간 예산 (제안)
- 정독 부분 (§1–§3, Abstract): 90분
- 표·그림 훑기: 20분
- §4 핵심 이온만 (Fe IX–XIV 위주): 30분
- 총 ~2시간 30분

### English
This is a **reference / database paper**, not a typical experimental or theoretical research paper, so do not read it linearly cover to cover. Instead:

1. **Abstract + §1 Introduction (read deeply)** — internalise the five design goals (easy to update, easy to distribute, transparent to the user, accuracy maintained by visual inspection, data structure suitable for program development).
2. **§2 Emission line intensities (read deeply, write out the equations by hand)** — Eqs. (1)–(4). These are the emissivity, level-population factorisation, and statistical-equilibrium equations previewed in Section 5 above. Note the v1 caveats: proton rates omitted; valid for $N_e \lesssim 10^{15}$ cm$^{-3}$.
3. **§3 Atomic data (read deeply)** — the *how* of the database.
   - §3.1 Energy levels (NIST preferred, SSTRUCT supplements).
   - §3.2 Radiative data ($A$ values; branching ratios $< 10^{-5}$ pruned from the distributed file).
   - §3.3 Electron collisional data (DW, CBe, CC, R-matrix compared; Eqs. 5–7).
   - §3.4 Assessment (the heart of the Burgess–Tully scaling philosophy).
4. **§4 The CHIANTI database (skim)** — lists which ions, drawn from which sources, organised by iso-electronic sequence. Read in full only the sequences you care about. For SDO/AIA, Hinode/EIS, and Solar Orbiter EUI, the key blocks are **§4.1 (He II), §4.2 (Li-like), §4.3 (Be-like, especially §4.3.1 C III), §4.5 (B-like, including Fe XXII)** and any Fe IX–XIV mentions.
5. **Tables 1–2 (ion inventory)** — at-a-glance map of which ionisation stages from H to Ni are included.
6. **Fig. 1 (Mg X example)** — illustrates Burgess–Tully scaling: raw $\Omega(E)$ → scaled $\Omega(x)$ → $\Upsilon(T)$ → scaled $\Upsilon(x)$. Linger on this figure.
7. **(Optional)** Skip the rest of §4's ion-by-ion details until you need them as a reference.

#### Suggested time budget
- Deep read (Abstract, §1–§3): 90 min
- Tables & figure: 20 min
- §4 selected ions (Fe IX–XIV focus): 30 min
- Total ≈ 2 h 30 min

---

## 7. 현대적 의의 / Modern Significance

### 한국어
CHIANTI는 1997년 v1 출간 이후 25년 이상 끊임없이 갱신되어, 오늘날 태양·천체 EUV/UV/X-ray 분광 분석의 **사실상 국제 표준**이 되었다. 그 영향은 다음 세 갈래로 정리된다.

1. **CHIANTI 자체의 진화 (#64 Dere+ 2023, v10.1과의 연속선)**
   - v4 (2003): 양성자 여기율, photoexcitation 추가.
   - v6 (2009): 전체 ionisation/recombination 일원화.
   - v8 (2015): Fe coronal 라인 정밀화 (Hinode/EIS, IRIS 시대).
   - v10/v10.1 (2021/2023): 새로운 R-matrix 자료, 더 많은 이온 (예: Ne, Si, S 미세 구조), 비-Maxwellian 분포 지원. **이 v10.1이 paper #61 Abbo+ 2025의 R(T_e) 계산에 사용된다.**
2. **Solar Orbiter / FSI 17.4 nm의 R(T_e) 합성 (#61과의 연결)** — Abbo+ (2025)는 FSI 17.4 nm 통과대역 안에 들어오는 모든 spectral line의 contribution function $G(T_e)$를 CHIANTI로 합산해 R(T_e)를 만든다. 본 1997 논문의 식 (1)–(4)가 정확히 그 합산의 단위이다. Fe IX, X, XI, XII (특히 Fe IX 171 Å, Fe X 174–177 Å) 이 17.4 nm 부근의 주요 기여자.
3. **모든 주요 EUV/X-ray 임무의 분석 파이프라인에 내장**
   - **SOHO/CDS, SUMER, UVCS** (이 논문의 시초 동기)
   - **Yohkoh/BCS, Hinode/EIS, XRT**
   - **TRACE, SDO/AIA**: 6개 EUV 채널의 temperature response 곡선이 CHIANTI 기반.
   - **IRIS**: chromosphere–TR (Si IV, C II, Mg II h&k) 진단.
   - **Solar Orbiter/SPICE & EUI/FSI**: 본 #61, 그리고 후속 분석.
   - **Hinode/EIS DEM 분석**, AGN/X-ray binary 광이온화 모델, 항성 코로나 분석.

또한 이 논문은 **천체물리 데이터의 공개 배포(open distribution)**의 모범 사례이다. 익명 FTP, 투명한 ASCII 형식, 사용자가 직접 갱신 가능한 구조 — 이 모든 요소가 오늘날의 *FAIR* 데이터(Findable, Accessible, Interoperable, Reusable) 원칙의 1990년대판 선구이다. ChiantiPy(Python 포팅)와 SunPy 생태계 통합으로 IDL → Python 시대에도 계속 살아있다.

### English
Since v1 in 1997, CHIANTI has been continuously updated for over 25 years and is now the **de-facto international standard** for solar and astrophysical EUV/UV/X-ray spectroscopic analysis. Its modern impact has three strands.

1. **Evolution of CHIANTI itself (continuous arc to #64, Dere+ 2023, v10.1)**
   - v4 (2003): proton excitation rates and photoexcitation added.
   - v6 (2009): unified ionisation/recombination treatment.
   - v8 (2015): much improved Fe coronal lines for the Hinode/EIS and IRIS era.
   - v10/v10.1 (2021/2023): new R-matrix data, more ions, support for non-Maxwellian electron distributions. **v10.1 is the version used by paper #61 (Abbo et al. 2025) to compute the FSI 17.4 nm response.**
2. **Solar Orbiter / FSI 17.4 nm $R(T_e)$ synthesis (link to #61)** — Abbo et al. (2025) sum the contribution functions $G(T_e)$ of every spectral line within the FSI 17.4 nm bandpass to build $R(T_e)$. Equations (1)–(4) of this 1997 paper are *literally* the per-line ingredient of that sum. Fe IX, X, XI, XII (notably Fe IX 171 Å and Fe X 174–177 Å) dominate near 17.4 nm.
3. **Embedded in essentially every major EUV/X-ray mission analysis pipeline**
   - **SOHO/CDS, SUMER, UVCS** (the original drivers).
   - **Yohkoh/BCS, Hinode/EIS, XRT.**
   - **TRACE, SDO/AIA**: temperature response curves for the six EUV channels are computed from CHIANTI.
   - **IRIS**: chromosphere–TR diagnostics (Si IV, C II, Mg II h&k).
   - **Solar Orbiter/SPICE & EUI/FSI**: paper #61 and ongoing work.
   - **Hinode/EIS DEM analyses**, AGN/X-ray binary photoionisation models, stellar corona analyses.

The paper is also a model case for **open distribution of astrophysical data**. Anonymous FTP, transparent ASCII format, and a structure that any user can update prefigure today's *FAIR* data principles (Findable, Accessible, Interoperable, Reusable) by more than a decade. The Python port **ChiantiPy** and integration with the SunPy ecosystem keep CHIANTI alive in the post-IDL era.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
