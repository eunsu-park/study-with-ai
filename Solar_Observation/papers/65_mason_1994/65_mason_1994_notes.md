---
title: "Spectroscopic diagnostics in the VUV for solar and stellar plasmas"
authors: H. E. Mason, B. C. Monsignori Fossi
year: 1994
journal: "The Astronomy and Astrophysics Review, Vol. 6, pp. 123–179"
doi: "10.1007/BF01208253"
topic: Solar Observation / Spectroscopic diagnostics review
tags: [VUV, EUV, optically thin plasma, contribution function, DEM, density-sensitive ratio, FIP effect, line ratio, ionization equilibrium, dielectronic recombination, CHIANTI, SOHO, Skylab, SMM, HRTS, UVCS]
status: completed
date_started: 2026-04-28
date_completed: 2026-04-28
---

# 65. Spectroscopic diagnostics in the VUV for solar and stellar plasmas / VUV 분광 진단법: 태양 및 항성 플라즈마

---

## 1. Core Contribution / 핵심 기여

### 한국어
Mason & Monsignori Fossi 1994는 SOHO 발사 직전(1995년 12월)에 출판된 **VUV(100–3000 Å) 분광 진단법의 체계적 종설**로, 광학적으로 얇은 태양 및 항성 플라즈마의 물리 매개변수(전자 밀도 $n_e$, 전자 온도 $T_e$, differential emission measure DEM, 원소 함량, 비열적 속도 $\xi$)를 분광 관측에서 추출하는 *방법론 전체*를 한 권에 종합했다. 5개 대섹션 — (§2) 원자 과정과 원자 데이터, (§3) 진단 기법 카탈로그, (§4) 태양 관측의 결과와 미래 임무, (§5) 항성 대기 — 를 통해 line emissivity, contribution function $G(T_e, n_e)$, density-sensitive line ratio (forbidden/intersystem vs allowed), temperature-sensitive line ratio (다른 여기 에너지를 가진 두 분광선), 부피·미분 emission measure (EM, DEM), FIP(First Ionization Potential) 효과, line profile 분석을 통일된 수학적 틀로 정리했다. 이 종설은 H. E. Mason의 24년 후속 종설(#59 Del Zanna & Mason 2018, LRSP)의 직접적 선구자이며, **#61 Abbo+ 2025**가 사용한 개념 — 응답함수 $R(T_e)$, contribution function, EM, FIP-수정 abundance, 이온화 평형의 한계 — 가 모두 본 종설의 §2.4와 §3에서 처음 체계화되었다.

### English
Mason & Monsignori Fossi 1994 is a **systematic review of VUV (100–3000 Å) spectroscopic diagnostics**, published just before the SOHO launch (Dec 1995), that consolidates the *entire methodology* for extracting physical parameters (electron density $n_e$, electron temperature $T_e$, differential emission measure DEM, element abundances, non-thermal velocity $\xi$) of optically thin solar and stellar plasmas from spectroscopic observations. Across five large sections — (§2) atomic processes and atomic data, (§3) diagnostic techniques, (§4) solar observations and future missions, (§5) stellar atmospheres — it organises line emissivity, the contribution function $G(T_e, n_e)$, density-sensitive line ratios (forbidden/intersystem vs allowed), temperature-sensitive line ratios (lines with different excitation energies), volume and differential emission measure (EM, DEM), the FIP (First Ionization Potential) effect, and line-profile analysis into a single coherent mathematical framework. This review is the direct precursor to H. E. Mason's 24-year successor review (#59 Del Zanna & Mason 2018, LRSP) and contains the foundational treatment of every concept used in **#61 Abbo+ 2025** — the response function $R(T_e)$, contribution function, EM, FIP-corrected abundance, and limits of the ionization-equilibrium assumption — all systematised in §2.4 and §3 of this paper.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction (§1, p. 123) / 서론

#### 한국어
서론은 VUV 분광이 지난 25년간 OSO, Skylab, SMM, Spacelab, IUE, EXOSAT, ROSAT, HST, EUVE 등 일련의 우주 임무를 통해 풍부한 데이터를 산출했음을 정리한다. 본 종설의 *집중 영역*은 100–2000 Å — 천이 영역(transition region, $T_e \geq 2\times 10^4$ K), 코로나($T_e \geq 10^6$ K), 플레어($T_e \geq 3\times 10^6$ K)의 분광선이 지배하는 파장대. 더 긴 파장(2000–3000 Å)은 광구·채층 기여가 우세. 본 종설의 구조는 §2 원자 과정, §3 진단 기법, §4 태양 관측, §5 항성 대기로 명시.

#### English
The introduction summarises the wealth of VUV data produced by OSO, Skylab, SMM, Spacelab, IUE, EXOSAT, ROSAT, HST, EUVE missions over the past 25 years. The review's *focus* is 100–2000 Å — dominated by transition-region ($T_e \geq 2\times 10^4$ K), coronal ($T_e \geq 10^6$ K), and flare ($T_e \geq 3\times 10^6$ K) lines. Longer wavelengths (2000–3000 Å) are dominated by photospheric/chromospheric features. The paper's structure is: §2 atomic processes, §3 diagnostic techniques, §4 solar results, §5 stellar atmospheres.

### Part II: Atomic Processes (§2, pp. 124–134) / 원자 과정

#### §2.1 Emission lines / 방출선

##### 한국어
**§2.1.1 Bound-bound emission**: 들뜬 상태의 이온이 자발 방출하면서 광자 방출. 단위 부피·시간당 방출되는 power는

$$P_{i,j} = N_j(X^{+m})\, A_{j,i}\, \frac{hc}{\lambda_{i,j}} \quad \text{erg cm}^{-3}\text{sec}^{-1} \tag{2}$$

여기서 $A_{j,i}$는 Einstein 자발방출 계수. 레벨 인구 $N_j$는 다음 단계별 인수분해(Eq. 3)로 분해:

$$N_j(X^{+m}) = \frac{N_j(X^{+m})}{N(X^{+m})}\cdot \frac{N(X^{+m})}{N(X)}\cdot \frac{N(X)}{N(H)}\cdot \frac{N(H)}{N_e}\cdot N_e \tag{3}$$

각 인수의 물리적 의미:
- $N_j(X^{+m})/N(X^{+m})$: 이온 $X^{+m}$의 레벨 $j$ 인구 분율 (statistical equilibrium)
- $N(X^{+m})/N(X)$: 이온 $X^{+m}$의 이온화 분율 (이온화 평형, 주로 $T_e$ 함수)
- $N(X)/N(H)$: 원소 X의 수소 대비 함량
- $N(H)/N_e \approx 0.8$: 완전 이온화 플라즈마의 수소 함량 비율

지구에서의 line flux (Eq. 4):
$$I(\lambda_{i,j}) = \frac{1}{4\pi R^2}\int_V P_{i,j}\, dV \quad \text{erg cm}^{-2}\text{sec}^{-1}\text{sr}^{-1} \tag{4}$$

**Statistical equilibrium (Eq. 5)**: 레벨 인구 계산의 일반식. 모든 충돌·복사 여기·탈여기를 포함:
$$N_j\left(N_e\Sigma_i C^e_{j,i} + N_p\Sigma_i C^p_{j,i} + \Sigma_{i>j} R_{j,i} + \Sigma_{i<j} A_{j,i}\right) = \Sigma_i N_i\left(N_e C^e_{i,j} + N_p C^p_{i,j}\right) + \Sigma_{i>j} N_i A_{i,j} + \Sigma_{i<j} N_i R_{i,j} \tag{5}$$

여기서 $C^e_{j,i}, C^p_{j,i}$ = 전자·양성자 충돌여기율 (cm³ s⁻¹), $R_{j,i}$ = 자극흡수율 (s⁻¹), $A_{j,i}$ = 자발방출률 (s⁻¹).

**§2.1.2 Coronal model approximation**: 광학적으로 *허용된* 전기 쌍극자 전이의 경우, 들뜬 상태 인구가 무시 가능하고 ($N_j(X^{+m})/N(X^{+m}) \simeq 1$), 두 레벨 시스템으로 풀 수 있다:
$$N_g(X^{+m})N_e C^e_{g,j} = N_j \Sigma_k A_{j,k} \tag{6}$$

이로부터 emissivity:
$$P_{g,j} = \frac{N(X^{+m})}{N(X)}\cdot \frac{N(X)}{N(H)}\cdot \frac{N(H)}{N_e}\cdot C^e_{g,j}\cdot \frac{hc}{\lambda_{g,j}}\cdot N_e^2\cdot B_{j,g} \tag{7}$$

$B_{j,g} = A_{j,g}/\Sigma_k A_{j,k}$ = 가지비(branching fraction). emissivity가 $N_e^2$에 의존 — *모든 광학적으로 얇은 충돌여기 방출선의 본질적 특성*.

**Contribution function (Eq. 10)**:
$$G(T, \lambda_{g,j}) = \frac{N(X^{+m})}{N(X)}\cdot \frac{N(H)}{N_e}\cdot C^e_{g,j}\cdot B_{j,g} \tag{10}$$

**Note**: $G(T, n_e)$는 $T_e$의 함수로 강하게 피크가 진다 (이온화 분율이 종형). 이것이 본 종설(과 #61 Abbo+ 2025의 R(T_e))의 모든 진단의 핵심.

**§2.1.3 Maxwellian collisional excitation (Eq. 11)**:
$$C^e_{i,j} = \frac{8.63\times 10^{-6}\, \Upsilon_{i,j}(T_e)}{T_e^{1/2}\, \omega_i}\, \exp\!\left(-\frac{\Delta E_{i,j}}{kT_e}\right) \tag{11}$$

$\omega_i$ = 통계 가중치, $\Upsilon_{i,j}$ = 열 평균 충돌 강도:
$$\Upsilon_{i,j}(T_e) = \int_0^\infty \Omega_{i,j}\, \exp\!\left(-\frac{E_j}{kT_e}\right)\, d\!\left(\frac{E_j}{kT_e}\right) \tag{12}$$

$\Omega_{i,j}$ = 충돌 강도 (대칭, 무차원, 산란 단면적 관련).

##### English
**§2.1.1 Bound-bound emission** — power per volume·time emitted by spontaneous decay (Eq. 2). The level population $N_j$ is decomposed as (Eq. 3) into the population fraction × ionization fraction × elemental abundance × hydrogen fraction × $N_e$ — each factor reflecting different physics. The flux at Earth (Eq. 4) is the volume integral.

**Statistical equilibrium (Eq. 5)** is the general balance equation for level populations including all collisional and radiative pumping/depumping mechanisms.

**§2.1.2 Coronal model approximation** — for optically allowed dipole transitions, the ground state holds nearly all the population. Eq. 6 reduces statistical equilibrium to a two-level system; emissivity scales as $N_e^2$ (Eq. 7) — the universal hallmark of optically thin collisional emission. The **contribution function** $G(T, \lambda)$ (Eq. 10) bundles the ion fraction, hydrogen abundance fraction, collision rate, and branching ratio. $G$ is sharply peaked in $T_e$ (driven by the bell-shaped ionic fraction), the cornerstone of all diagnostics in this review.

**§2.1.3** — the **Maxwellian-averaged collision rate** (Eq. 11) involves the **thermally-averaged collision strength** $\Upsilon$ (Eq. 12), itself derived from the symmetric collision strength $\Omega$.

#### §2.2 Electron scattering calculations / 전자 산란 계산

##### 한국어
- **Distorted Wave (DW)** — 가장 단순; 산란 전자가 중심장 퍼텐셜에서 운동. 정확도 ~25% (UCL DW 코드, Eissner & Seaton 1972). 다전자 이온화 단계에 적합.
- **Coulomb Bethe (CBe)** — 고에너지 부분파에 유효; 산란 전자가 표적 전자에 침투하지 않는다고 가정 (Burgess & Shoerey 1974).
- **Close-Coupling (CC)** — 가장 정확 (~10%). 표적과 산란 전자의 결합 채널을 모두 풀어야 함 (IMPACT, RMATX 코드). 계산 비용 큼.
- **Effective Gaunt factor / Van Regemorter (1962) (Eq. 13)** — 반경험 공식: $\Omega_{i,j} = (8\pi/\sqrt{3})\, \omega_j\, f_{i,j}\, (I_H/\Delta E_{i,j})\, \bar{g}$. **본 종설은 Sampson & Zhang 1992의 비판을 인용해 사용 *중단을 권고***.
- **Burgess–Tully scaling (1992)** — 충돌 산란 데이터의 비판적 평가법 (Fig. 1: Mg X 2s–2p 전이).

##### English
DW (~25% accuracy), CBe (high-energy partial waves), CC (~10%, most accurate but expensive). The **Van Regemorter $\bar{g}$ formula** (Eq. 13) should be **abandoned** per Sampson & Zhang (1992). **Burgess–Tully scaling** (1992) is the recommended critical-evaluation tool.

#### §2.3 Continuum emission / 연속체 방출

##### 한국어
- **§2.3.1 Free-free (Eq. 14, 15)** — thermal bremsstrahlung. 완전 이온화 플라즈마의 총 free-free emitted power: $P_{ff}(T) = 2.4\times 10^{-27}\, T^{1/2}\, N_e^2$ erg cm⁻³ s⁻¹. $T > 10^7$ K에서 주된 복사 손실.
- **§2.3.2 Free-bound (Eq. 16)** — 자유 전자가 이온에 포획 (radiative recombination). 이온화 한계에서 불연속 발생.
- **§2.3.3 Two-photon continuum** — hydrogenic·helium-like 이온의 metastable 2s 상태에서 2광자 붕괴. $T \leq 3\times 10^4$ K에서 중요.
- **§2.3.4 Total continuum** — Gronenschild & Mewe 1978; Mewe et al. 1986; Burgess & Summers 1987 분석.

##### English
Free-free dominates at $T > 10^7$ K with $P_{ff} \propto T^{1/2} N_e^2$ (Eq. 15). Free-bound shows discontinuities at ionization thresholds. Two-photon emission matters at low $T$.

#### §2.4 Ionization balance / 이온화 평형

##### 한국어
**핵심 식 (Eq. 17)**:
$$N^{+m}(q_{col} + q_{au} + q_{ct}) = N^{+m+1}(\alpha_r + \alpha_d + \alpha_{ct}) \tag{17}$$

생성 과정: 충돌이온화 ($q_{col}$), excitation-autoionization ($q_{au}$), charge transfer ($q_{ct}$).
소멸 과정: radiative recombination ($\alpha_r$), dielectronic recombination ($\alpha_d$), charge transfer ($\alpha_{ct}$).

- **§2.4.1 RR + photoionization** (Eq. 18): TOPBASE 데이터베이스 (Opacity Project, Seaton 1987). Aldrovandi & Pequignot 1976 근사.
- **§2.4.2 Dielectronic recombination + autoionization** (Eq. 19, 20, 21): 자유 전자가 doubly excited state로 포획되어 안정화. **고온에서 RR보다 인자 ≥20 우세** (Burgess 1964). Burgess 1965 일반 공식. Field ionization effect (Krylstedt et al. 1990).
- **§2.4.3 Direct collisional ionization** (Eq. 22): Burgess & Chidichimo 1983 일반 공식 (실험과 23% 일치). Arnaud & Rothenflug 1985, Arnaud & Raymond 1992 컴파일.
- **§2.4.4 Ionization equilibrium**: Arnaud & Raymond 1992의 철 이온 새 계산 — Fe⁺⁵–Fe⁺⁹은 더 *낮은 온도*에서 피크, Fe⁺¹⁵–Fe⁺²³은 더 *높은 온도*로 이동 (Fig. 2). Dickson et al. 1994의 ADAS 기반 독립 계산도 합의.
- **§2.4.5 Non-equilibrium processes**: 시간 척도가 원자 과정보다 짧으면 시간 의존 방정식 필요 — **#61 Abbo+ 2025 §5의 frozen-in 논의의 출발점**. Mewe et al. 1985b의 1D MHD 플레어 모델, Shoub 1983의 비-Maxwellian 효과, Tworkowski 1975의 확산 효과.

##### English
**Eq. 17** is the key balance equation. The dominant processes are collisional ionization (with autoionization), radiative recombination, and dielectronic recombination. **DR exceeds RR by factor ≥20 at high $T$** (Burgess 1964). Arnaud & Raymond 1992 is the standard iron ionization-equilibrium reference. **§2.4.5 is the conceptual source of the frozen-in / non-equilibrium discussion in #61 Abbo+ 2025 §5.**

#### §2.5 Future atomic physics work + ADAS / 향후 원자 물리 연구

##### 한국어
- **Iron Project** (Hummer et al. 1993): n=3 구조 포함 모든 철 이온의 CC 충돌여기율 새 계산.
- **ADAS** (Atomic Data and Analysis Structure, Summers et al.) — 핵융합 연구용으로 개발된 종합 원자 데이터 시스템. 통합된 collisional-radiative 모델 — 평형/비평형 조건 모두 처리. **CHIANTI(#63, #64)의 사실상 경쟁자이자 보완자**.

##### English
The Iron Project aims to compute new CC excitation rates for all Fe ions. **ADAS** is a comprehensive atomic-data system from JET fusion research, the de-facto competitor and complement to CHIANTI (#63, #64).

### Part III: Plasma Diagnostics (§3, pp. 134–141) / 플라즈마 진단

#### §3.1 UV-EUV spectral line identifications / 분광선 동정
- Edlén 1940년대 코로나 가시 영역 선 동정 (이 종설의 시작점)
- Lund 대학 Martinson 그룹의 고이온화 원자 분광 — review (Curtis & Martinson 1990)
- 다수의 line list/atlas: Malinovsky & Heroux 1973 (50–300 Å), Behring et al. 1972/1976, Dere 1978 (171–630 Å), Feldman et al. 1987 (170–625 Å) 등.

#### §3.2 Synthetic spectra / 합성 스펙트럼
- 코드 비교 (10⁴–10⁸ K): Raymond & Smith 1977, Mewe 1985a, Landini & Monsignori-Fossi 1990, Doschek & Cowan 1984 — *최대 30% 차이*.

#### §3.3 Radiative cooling curves / 복사 냉각 곡선
- Cox & Tucker 1969, Sutherland & Dopita 1993의 평형/비평형 종합. 코드 간 ~30% 일치.

#### §3.4 Electron density diagnostics / 전자 밀도 진단

##### 한국어 (가장 중요한 섹션 — **#61 R(T_e) 진단 및 다양한 streamer 진단의 공통 토대**)

원리: 동일 이온의 두 분광선 비율은 부피·함량에 무관하므로 강력한 진단. 분광선을 그룹화:
- **Allowed (electric dipole)**: 충돌여기 → 빠른 복사 붕괴 (coronal model § 2.1.2)
- **Forbidden / intersystem**: metastable 레벨 $m$에서 시작; $A_{m,g} \sim 10^0–10^2$ s⁻¹ — 충돌탈여기 $N_e C^e_{m,g}$가 비교 가능해질 수 있음

**Two-level metastable model**:
- 저밀도 한계 ($N_e \to 0$, $A_{m,g} \gg N_e C^e_{m,g}$): $I_{m,g} \propto N_e^2$ (Eq. 24)
- 고밀도 한계 ($N_e \to \infty$, $N_e C^e_{m,g} \gg A_{m,g}$, Boltzmann 평형): $I_{m,g} \propto N_e$ (Eq. 26)
- 중간 영역: $I_{m,g} \propto N_e^\beta$, $1 < \beta < 2$ (Eq. 27)
- 더 복잡한 (allowed - from metastable) 비: $I_{k,m} \propto N_e^\beta$, $2 < \beta < 3$ (Eq. 28)

**구체적 진단 비 / Concrete diagnostic ratios**:
- C III [1909/977] (천이 영역, intersystem/allowed)
- O V [1218/629] (천이 영역)
- Fe IX [242/245] (코로나, $T_e \approx 8\times 10^5$ K)
- O IV [1407.39/1404.81] (천이 영역, $T_e$-비민감)
- Si III [1301/1312], Si IV [1393/1402]
- Fe XIV [211/219] (코로나)

**중요한 도식 (Fig. 6, p. 145)**: Doschek 1985에서 가져온 그림. C III, N III/IV, O III/IV/V, S IV/V의 metastable 레벨 인구 vs $N_e$ ($10^8–10^{14}$ cm⁻³). Plateau (저밀도 → $N_j/N_e N_Z$ 일정) 와 declining 영역 ($A_{j,i}$ 의존)의 전이가 진단 가능 영역. 대부분의 천이 영역 분광선은 $N_e > 10^{10}$ cm⁻³까지 민감.

##### English
Same-ion ratios are powerful because they are independent of volume and abundance. Lines split into **allowed** (collisional excitation, fast decay) and **forbidden/intersystem** (from metastable level $m$ with small $A_{m,g}$, allowing density-dependent depopulation). The two-level metastable model gives $I \propto N_e^2$ at low density, $I \propto N_e$ at high density, and the intermediate behaviour is the diagnostic regime. Concrete ratios include C III [1909/977], O V [1218/629], Fe IX [242/245] (coronal), O IV [1407/1405] (T-insensitive), Si III/IV pairs, and Fe XIV [211/219]. **Fig. 6 (Doschek 1985)** plots metastable populations vs $N_e$ for many ions and is a master reference.

#### §3.4.1 Radiative decay rates for forbidden/intersystem transitions / 금지·intersystem 전이의 복사 붕괴율
- $A_{m,j}$의 정확도가 ratio 진단의 핵심. Harvard SAO ion-trapping 측정 — C II 2330 Å, C III 1909 Å 등.

#### §3.5 Electron temperature diagnostics / 전자 온도 진단
$T_e$에 민감한 ratio (Eq. 29):
$$\frac{I_{g,j}}{I_{g,k}} = \frac{\Delta E_{g,k}\, \Upsilon_{g,j}}{\Delta E_{g,j}\, \Upsilon_{g,k}}\, \exp\!\left(\frac{\Delta E_{g,k} - \Delta E_{g,j}}{kT_e}\right) = F(T_e) \tag{29}$$

**조건**: $(\Delta E_{g,k} - \Delta E_{g,j})/kT_e \gg 1$ — 같은 이온의 두 다른 여기 에너지 분광선, 같은 등온 영역, 같은 $N_e$. 같은 분광기에서 함께 관측 가능한 가까운 파장이 이상적.

**예시 / Examples**: Li-like ions [2s-2p / 2s-3p] (McWhirter 1976) — but $G(T)$ broad. O V [629/172] (Flower & Nussbaumer 1975), O V [1218/629], O VI [1032/173]. Na-like Si IV [1129/1393], Al III [1612/1855] (Doschek & Feldman 1987).

#### §3.6 Emission measure analysis / Emission measure 분석

##### 한국어
**부피 emission measure**:
$$EM = \int_V N_e^2\, dV \tag{30}$$

**Pottasch (1964) 방법**: contribution function이 $T_{max}$ 부근에서만 기여한다고 가정 — 등온 평균 $\langle EM \rangle$ 추출:
$$I^{obs}_{\lambda_{g,j}} = \beta\, \frac{N(X)}{N(H)}\, \langle EM \rangle\, \int_{\Delta T} G(T, \lambda_{g,j})\, dT \tag{31}$$

이는 다른 $T_{max}$ 분광선들로 $\langle EM \rangle$의 *온도 분포*를 도출 (Fig. 3: active region, quiet Sun, coronal hole의 $T \cdot DEM$ 곡선).

**일반 형식 (DEM, Eq. 32)**:
$$I^{obs}_{\lambda_{g,j}} = \beta\, \frac{N(X)}{N(H)}\, \int G(T, \lambda_{g,j})\, \phi(T)\, dT \tag{32}$$

여기서 $\phi(T) = N_e^2 \, dV/dT$ = differential emission measure (DEM). 적분 방정식의 *역문제* — 다양한 분광선 강도로부터 $\phi(T)$ 추출.

**SOHO CDS-SUMER DEM 비교 연구** (Harrison & Thompson 1992): 알려진 DEM을 시험 케이스로 다양한 inversion 기법 비교. **주요 문제**: $\phi(T)$의 *smoothness* 제약 명세, 원자 데이터 오차 전파, 좁은 $G(T)$ + 넓은 온도 범위의 분광선 선택.

##### English
**Volume EM** (Eq. 30) and the Pottasch isothermal approximation (Eq. 31). Fig. 3 shows $T \cdot DEM$ for active region, quiet Sun, and coronal hole. The general DEM (Eq. 32) requires solving an integral equation — an *ill-posed inverse problem* requiring smoothness constraints. The SOHO CDS-SUMER DEM comparison study (Harrison & Thompson 1992) benchmarks methods.

#### §3.7 Abundance determination / 함량 결정

##### 한국어
DEM을 이용한 iterative 절차로 원소 함량 결정 (Pottasch 1964, Monsignori Fossi et al. 1994c). Meyer 1985, 1990, 1993의 종설 — **태양풍에서 측정한 코로나 함량은 광구 함량과 다르며, FIP에 따라 다름**. **FIP 효과**:
- FIP < 10 eV (Mg, Si, Fe, Ca, Na, K, Al) — 코로나에서 광구보다 인자 ~3–4 *증가*
- FIP > 10 eV (H, He, C, N, O, Ne, Ar) — 광구 값 그대로
- 다른 자기 구조 (closed/open field lines) 에서 다른 패턴 (Fig. 4: [Ne/Mg] for diverging fields, polar plumes, AR, transition zone, prominences, impulsive flares — 인자 30 이상의 변동)

**#61 Abbo+ 2025의 핵심 systematic 불확도**: 두 abundance 값(Asplund+ 2021 광구 vs CHIANTI 기본 FIP-corrected ×$10^{0.5}$)의 차이가 R(T_e) 띠 폭의 주요 결정자 — **본 §3.7의 FIP 논의가 그 출발점**.

##### English
Iterative DEM-based abundance determination. **The FIP effect** — coronal abundances differ from photospheric, with low-FIP elements (Mg, Si, Fe, etc.) enhanced ~3–4× and high-FIP elements (H, He, C, N, O, Ne, Ar) at photospheric values. **Fig. 4** shows [Ne/Mg] across solar feature types. **This is the conceptual origin of the iron-abundance choice that drives the systematic R(T_e) uncertainty in #61 Abbo+ 2025.**

#### §3.8 Spectral line profiles / 분광선 프로파일
Gaussian profile (Eq. 33):
$$I_\lambda = \frac{I}{\sqrt{2\pi}\sigma}\exp\!\left[-(\lambda-\lambda_0)^2/(2\sigma^2)\right] \tag{33}$$

분산 (Eq. 34):
$$\sigma^2 = \frac{\lambda^2}{2c^2}\!\left(\frac{2kT}{M} + \xi^2\right) + \sigma_I^2 \tag{34}$$

세 성분: 열 운동 (~$T$), 비열적 속도 ($\xi$, *most probable non-thermal velocity = ntv*), 기기 폭 ($\sigma_I$). 천이 영역 광폭 $\xi \sim 20$ km/s — 가열 메커니즘 제약. 코로나 외부의 광폭 — 파동 전파.

### Part IV: Solar Physics — Observations (§4, pp. 141–172) / 태양 관측 결과

#### §4.1 Early observations / 초기 관측

##### 한국어
- **§4.1.1 OSO** — OSO-5 GSFC grating spec.: Fe XXI [128.73/145.66] for $N_e \geq 10^{11}$ cm⁻³ (Kastner et al. 1974, Mason et al. 1984, Conlon et al. 1992). OSO-4/-6 HCO 300–1400 Å. OSO-7 spectroheliograph 190–300 Å — Fe IX–Fe XVI ($6\times 10^5$–$2.5\times 10^6$ K) (Kastner et al. 1974, 1976, 1978). OSO-8 1000–2000 Å (Hansen & Schaffner 1977; Bruner & McWhirter 1979 — quiet Sun C IV 1548 Å, **Alfvén wave heating으로 결론**).
- **§4.1.2 Skylab** (1973) — NRL S082A "overlapogram" (170–630 Å, $\Delta\lambda = 0.1$ Å, $2''$ 해상도, Feldman et al. 1987 atlas), HCO S0555 (280–1350 Å), NRL S082B (970–3940 Å). **Feldman 1992의 burst model** — 비-평형 가설; FIP 효과 측정 (Widing & Feldman 1992; 1994; Feldman 1992b; Doschek et al. 1991).
- **§4.1.3 SMM (UVSP)** — 1150–3600 Å. O IV 1.7 (10⁵K) multiplet ~ 1400 Å, Si IV 1402.77 Å allowed line. C IV 1548 Å sunspot, loop flow studies (Gurman & Athay 1983; Kopp et al. 1985; Gebbie et al. 1981; Henze & Engvold 1992 redshifts 4–8 km/s). **Fe XXI 1354.1 Å** (10⁷ K) — 임펄스 위상의 blueshift (Mason et al. 1986).

##### English
**OSO** (1962-onwards), **Skylab** (1973 — S082A/S082B/HCO instruments), **SMM-UVSP** (1980-89, 1150–3600 Å). Foundational for iron diagnostics, FIP measurements, transition-region downflows, and flare blueshifts.

#### §4.2 Recent observations (1985–1994) / 최근 관측
- **§4.2.1 CHASE** (Spacelab-2, 1985) — He II 304 Å / Lyα 1218 Å로 corona 헬륨 함량 N(He)/N(H) = 0.079±0.011.
- **§4.2.2 HRTS** — NRL의 1170–1710 Å, $\Delta\lambda = 0.05$ Å, $1''$ 공간 분해능. C IV 1548 Å power spectrum studies (Dere 1989) → 코로나는 미세 구조 (filling factor $5\times 10^{-3}$ at $10^5$ K, 0.4 at $3\times 10^4$ K). C IV redshift 일관됨. 폭발적 사건 (explosive events) — magnetic reconnection과 상관 (Dere et al. 1991). **Fig. 7**: ntv vs $T$ — active region에서 quiet region보다 큼.
- **§4.2.3 Spartan UVCS** (1993) — 1.3–3.5 R☉ HI Lyα 1216 Å + O VI 1032/1037 Å. **#61 Abbo+ 2025의 직접 선구자**: Lyα 프로파일의 두 성분(1 Å 좁은 + 50 Å 넓은) — 코로나 H 운동 온도, Doppler dimming으로 *outflow speed* 측정. Withbroe et al. 1982, Noci et al. 1987의 Doppler dimming 이론. **이 종설의 §4.2.3은 본질적으로 #61이 사용한 Metis Lyα 진단의 *원형***.
- **§4.2.4 Multilayer optics** — normal incidence X-ray imaging. Walker et al. 1987 (44 Å), 1988 (171–175 Å Fe IX/X = #61의 EUI/FSI 17.4 nm 대역의 *직계 조상*!), 256 Å (He II + Fe XXIV). NIXT (Golub et al. 1990, 63.5 Å Mg X). MSSTA (1991, sub-arcsecond Lyα + Fe XII 193 Å).
- **§4.2.5 SERTS** (GSFC, 1989-) — EUV imaging spectrograph 235–450 Å + 170–225 Å (2nd order). $4.7 \leq \log T_e \leq 6.8$. 흑점 outflow Mg IX 368 Å (14 km/s, Neupert et al. 1992c). **SDO/AIA·SPICE의 직접 선구자**.
- **§4.2.6 SPDE** (Lockheed, 1992, 1994) — 동시 EUV/UV/soft X-ray + 65/130/170/304 Å + 1216/1550/1580 Å + 1200–1420 Å + 265–309 Å. ~1″ 해상도.

(이후 §4.2.7–§4.4: SOHO 미션 예고, 미래 임무 — 이 종설이 SOHO 발사 전에 작성되었음을 기억)

### Part V: Stellar Atmospheres (§5) / 항성 대기

##### 한국어
태양 분광 진단 기법을 항성 대기에 적용. IUE, EXOSAT, ROSAT, HST, EUVE, ASCA로 관측한 후기형(M dwarf), 활동성, RS CVn binaries 등. **항성 대기에서 DEM 유도**, 항성 대기의 dynamic하고 inhomogeneous한 본질, 항성 플레어의 $T_e$/$N_e$ 진단. 본 학습 트랙(태양 중심)에서는 *주변적*이지만 분광 진단법의 보편성을 보여주는 챕터.

##### English
Application of solar diagnostic techniques to stellar atmospheres. IUE, EXOSAT, ROSAT, HST, EUVE, ASCA observations of M dwarfs, RS CVn binaries, and other active stars. Stellar DEM derivation, dynamic/inhomogeneous nature, stellar-flare $T_e$/$N_e$ diagnostics. Peripheral to the solar-focused study track but demonstrates the universality of these methods.

---

## 3. Key Takeaways / 핵심 시사점

1. **Contribution function $G(T_e, n_e)$는 모든 진단의 출발점 / The contribution function is the starting point of all diagnostics** —
   $G$는 ionization fraction × hydrogen abundance × collision rate × branching ratio를 단일 함수로 통합 (Eq. 10). $T_e$에 강하게 피크가 진다는 점이 진단 가능성의 본질. **#61의 R(T_e)는 G의 다대역 가중 합산**. /
   $G$ unifies ionic fraction × H abundance × collision rate × branching ratio (Eq. 10), sharply peaked in $T_e$. **#61's R(T_e) is a multi-band weighted sum of G's.**

2. **밀도 진단은 metastable 레벨에 의존 / Density diagnostics rely on metastable levels** —
   Forbidden/intersystem 분광선의 $A_{m,g}$가 $N_e C^e_{m,g}$와 비교 가능해질 때만 진단 가능 (Eq. 24–28). 이로부터 $I \propto N_e^\beta$, $1 < \beta < 3$. 같은 이온의 ratio라 부피·함량 무관 — 강력한 진단. /
   Forbidden/intersystem lines from a metastable level give $I \propto N_e^\beta$ ($1<\beta<3$, Eq. 24-28); same-ion ratios are volume- and abundance-independent — a powerful diagnostic.

3. **온도 진단은 같은 이온의 두 다른 여기 에너지 / Temperature from two different excitation energies of the same ion** —
   Eq. 29: $(\Delta E_k - \Delta E_j)/kT_e \gg 1$ 조건. 같은 이온, 같은 등온 영역, 같은 $N_e$. $G(T)$가 좁은 분광선이 더 정확. **#61의 응답함수 R(T_e)는 같은 원리의 *대역 평균* 버전**. /
   Eq. 29 with $\Delta E$ split. Same ion, same isothermal region, same $N_e$. Lines with narrow $G(T)$ are more precise. **#61's R(T_e) is the band-averaged version of the same principle.**

4. **DEM은 inverse problem이며 ill-posed / DEM is an inverse problem and ill-posed** —
   Eq. 32의 적분 방정식 inversion. 원자 데이터 오차, 좁은 G(T) 라인의 부족, smoothness 제약의 specification 모두가 도전. **#61에서는 streamer가 등온이라고 가정해 이 어려움 회피** — 이 가정의 합리성이 §5의 토의 핵심. /
   The integral inversion in Eq. 32 is plagued by atomic data errors, narrow-G(T) line scarcity, and smoothness constraint specification. **#61 sidesteps this by assuming streamer isothermality** — the validity of which is the key §5 discussion.

5. **이온화 평형 가정에는 본질적 한계 / Ionization equilibrium has intrinsic limits** —
   §2.4.5: 시간 척도가 원자 과정보다 짧으면(태양풍 흐름이 빠르면) 비평형. **#61 Abbo+ 2025의 Fig. 9 — τ_exp ~ τ_ion ~ τ_rec at 4.25 R☉ — 가 본 종설 §2.4.5의 직접적 후속**. /
   §2.4.5 — when timescales are shorter than atomic processes (fast solar wind), equilibrium fails. **#61's Fig. 9 (τ_exp ~ τ_ion ~ τ_rec at 4.25 R☉) is the direct continuation of this §2.4.5 discussion.**

6. **FIP 효과는 코로나 함량의 30+배 변동을 일으킴 / The FIP effect causes 30+× variation in coronal abundances** —
   §3.7 + Fig. 4: 자기 구조에 따라 [Ne/Mg]가 광구의 0.1배 ~ 광구의 30배. 저-FIP 원소 (Mg, Si, Fe) ~3–4배 enhancement. **#61의 두 abundance 선택 (Asplund 2021 vs FIP-corrected ×$10^{0.5}$)이 R(T_e) 띠 폭을 결정 — 본 §3.7이 그 출발**. /
   §3.7 + Fig. 4 — [Ne/Mg] varies 0.1×–30× across magnetic features. Low-FIP (Mg, Si, Fe) ~3-4× enhanced. **#61's two abundance choices (Asplund vs FIP×$10^{0.5}$) determine the R(T_e) band width — origin of which is this §3.7.**

7. **분광 진단의 정확도는 원자 데이터에 의해 제한됨 / Diagnostic accuracy is bounded by atomic data quality** —
   합성 스펙트럼 코드 간 ~30% 차이 (§3.2). DW vs CC 충돌율 ~25% 차이 (§2.2). **이것이 25년 후의 CHIANTI v10.1 (#64)이 여전히 점진적 개선이 필요한 이유** — 원자 데이터 향상은 지속 진행 중. /
   ~30% spread among synthetic-spectrum codes (§3.2); ~25% DW vs CC differences (§2.2). **This is why CHIANTI v10.1 (#64) — 25 years later — still requires incremental improvement.**

8. **본 1994년 종설은 SOHO 시대의 *anticipation* / This 1994 review captures the *anticipation* of the SOHO era** —
   §4.2.3에서 Spartan UVCS는 1993년에 1.3–3.5 R☉의 HI Lyα + O VI 분광 — *Solar Orbiter Metis가 영상으로 하는 일을 분광으로 시연한 미니 임무*. §4.2.4의 multilayer optics (Walker et al. 1988, 171–175 Å Fe IX/X)는 **EUI/FSI 17.4 nm의 직계 조상**. SOHO 발사 1년 전, 30년 후 Solar Orbiter의 모든 핵심 기술이 *예고*되어 있다. /
   §4.2.3 (Spartan UVCS, 1.3–3.5 R☉ Lyα + O VI) is a mini-mission demonstrating spectroscopically what Solar Orbiter Metis does imaging-wise. §4.2.4 multilayer optics (Walker 1988, 171–175 Å Fe IX/X) is the **direct ancestor of EUI/FSI 17.4 nm**. One year before SOHO launch, every key technology of 30-years-later Solar Orbiter is *anticipated*.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Hierarchy of equations / 식 계층

```
Bound-bound emission (Eq. 2)
        │
        ▼  via (Eq. 3) population factorization
Volume integral → Earth flux (Eq. 4)
        │
        ▼  Statistical equilibrium (Eq. 5)  [or coronal model Eqs. 6-9]
        │
        ▼  Define contribution function (Eq. 10)
Emissivity = G(T_e, n_e) × n_e²
        │
        ▼  Maxwellian collision rates (Eqs. 11-12)
        │
        ▼  Ion fraction from ionization equilibrium (Eq. 17)
        │
        ▼  Volume EM (Eq. 30) → Pottasch (Eq. 31) → DEM (Eq. 32)
        │
        ▼  Density diagnostics (Eqs. 23-28)
        │
        ▼  Temperature diagnostics (Eq. 29)
        │
        ▼  Profile analysis (Eqs. 33-34)
```

### 4.2 Core equations table / 핵심 수식 모음

| Eq. | Name / 이름 | Form |
|---|---|---|
| 2 | Bound-bound emissivity | $P_{i,j} = N_j A_{j,i} (hc/\lambda)$ |
| 3 | Population factorisation | $N_j = (N_j/N^{+m})(N^{+m}/N_X)(N_X/N_H)(N_H/N_e) N_e$ |
| 5 | Statistical equilibrium | Σ excitation rates = Σ de-excitation rates |
| 7 | Coronal model emissivity | $P \propto G(T,n) N_e^2$ |
| 10 | Contribution function | $G = (N^{+m}/N_X)(N_H/N_e) C^e B$ |
| 11 | Maxwellian collision rate | $C^e = (8.63\times 10^{-6} \Upsilon)/(T^{1/2}\omega) e^{-\Delta E/kT}$ |
| 13 | Van Regemorter (deprecated) | $\Omega = (8\pi/\sqrt 3) \omega f (I_H/\Delta E) \bar g$ |
| 15 | Free-free total | $P_{ff} = 2.4\times 10^{-27} T^{1/2} N_e^2$ |
| 17 | Ionization balance | $N^{+m}(q_{col}+q_{au}+q_{ct}) = N^{+m+1}(\alpha_r+\alpha_d+\alpha_{ct})$ |
| 24 | Density-low limit | $I_{m,g} \propto N_e^2$ |
| 26 | Density-high limit | $I_{m,g} \propto N_e$ |
| 27 | Density intermediate | $I_{m,g} \propto N_e^\beta, 1<\beta<2$ |
| 28 | Allowed-from-metastable | $I_{k,m} \propto N_e^\beta, 2<\beta<3$ |
| 29 | Temperature diagnostic | $I_j/I_k = (\Delta E_k \Upsilon_j)/(\Delta E_j \Upsilon_k) e^{(\Delta E_k-\Delta E_j)/kT}$ |
| 30 | Volume EM | $EM = \int N_e^2 dV$ |
| 31 | Pottasch isothermal | $I = \beta (N_X/N_H) \langle EM\rangle \int_{\Delta T} G dT$ |
| 32 | DEM general | $I = \beta (N_X/N_H) \int G(T) \phi(T) dT$, $\phi = N_e^2 dV/dT$ |
| 33-34 | Gaussian profile + width | $\sigma^2 = (\lambda^2/2c^2)(2kT/M + \xi^2) + \sigma_I^2$ |

### 4.3 Concrete numerical examples / 구체적 수치 예

**Example 1 — O IV [1407.39/1404.81] density-sensitive ratio**:
- O IV 천이 영역 ion ($T_{max} \approx 1.5\times 10^5$ K)
- 1407.39 Å (intersystem) / 1404.81 Å (allowed)
- ratio가 $T_e$에 비민감 (좁은 $G(T)$)
- $N_e \in [10^{10}, 10^{12}]$ cm⁻³ 범위에서 가장 민감
- HRTS 관측에서 active region filling factor $5\times 10^{-3}$ 도출 (Dere et al. 1987)

**Example 2 — Active region DEM (Fig. 3)**:
- Active region: $T \cdot DEM$ at $\log T = 6.0 \approx 10^{27}$ K cm⁻⁵
- Quiet Sun: $\log T = 6.0$에서 $\sim 10^{26}$ K cm⁻⁵
- Coronal hole: $\sim 10^{25}$ K cm⁻⁵
- 천이 영역 ($\log T = 4.5$)에서 모두 비슷한 값 ($\sim 10^{27.5}$ K cm⁻⁵)으로 수렴
- **이 EM curve 형태가 #61이 streamer에서 도출한 EM = $4\times 10^{22}$ cm⁻⁵에 정량적 맥락 제공**

**Example 3 — Iron ionization equilibrium shift (Fig. 2)**:
- Fe⁺¹⁵ peak at $\log T_e \approx 6.5$ (Arnaud & Raymond 1992) vs Fe⁺¹⁵ peak at $\log T_e \approx 6.4$ (Arnaud & Rothenflug 1985)
- Fe⁺⁹: AR'92 shifts to *lower* T (pushed toward $\log T \approx 5.95$)
- 이 shift가 #61의 R(T_e) 종형 곡선의 정확한 위치를 결정 — *이온화 평형 데이터의 미세 변경이 streamer 온도 추정의 ~5% 변화를 일으킴*

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1940s ── Edlén — 코로나 가시 영역 분광선 동정 (Fe XIV, Ca XV 등)
              ↳ 코로나 분광학의 출발점
                   │
1962 ──── First OSO launched
                   │
1964 ──── Pottasch — isothermal EM analysis (Eq. 31의 출처)
                   │
1971 ──── Gabriel — coronal Lyα resonance scattering
                   │
1973 ──── Skylab Apollo Telescope Mount
              S082A (170-630 Å NRL), S0555 HCO, S082B (970-3940 Å NRL)
                   │
1976-77 ── Mewe; Raymond & Smith — first synthetic spectra codes
                   │
1980-89 ── SMM-UVSP (1150-3600 Å) + HXIS + XRP
                   │
1985 ──── CHASE on Spacelab-2 — He II/Lyα coronal helium abundance
              Mewe spectral code
                   │
1985-93 ── HRTS (NRL rocket) — high-resolution UV spectra
                   │
1988 ──── Walker et al. — first multilayer optics image at 171-175 Å (Fe IX/X)
              ↳ EUI/FSI 17.4 nm 직계 조상
                   │
1990 ──── Landini & Monsignori-Fossi atomic database (CHIANTI 전신)
                   │
1992 ──── Arnaud & Raymond — new Fe ionization equilibrium
              Burgess & Tully — collision-data critical evaluation method
                   │
1993 ──── Spartan UVCS — 1.3-3.5 R☉ Lyα + O VI spectroscopy
              ↳ Solar Orbiter Metis의 분광 선구자
                   │
1994 ──── ★ THIS PAPER — Mason & Monsignori Fossi review
                   │
1995 ──── ★ SOHO launched — UVCS, CDS, SUMER, EIT
              CHIANTI v1 (Dere+ 1997 = #63 in this archive)
                   │
2006 ──── Hinode (EIS, SOT, XRT)
                   │
2010 ──── SDO (AIA, HMI, EVE)
                   │
2013 ──── IRIS
                   │
2018 ──── ★ #59 Del Zanna & Mason 2018 (LRSP)
              ↳ 본 종설의 24년 후 갱신판 (같은 H.E. Mason 저자)
                   │
2020 ──── Solar Orbiter launched (Metis, EUI, SPICE, SoloHI, ...)
                   │
2023 ──── #64 Dere+ 2023 — CHIANTI v10.1
                   │
2025 ──── ★ #61 Abbo+ 2025 — Solar Orbiter Metis + EUI/FSI streamer T_e
              ↳ 본 종설 §3 (diagnostics)·§3.6 (EM)·§3.7 (FIP)·§4.2.3-4 (Lyα + multilayer)의
                직접적 응용
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **#59 Del Zanna & Mason 2018** (LRSP 13, 5) | **직접적 후속 (24년 후) / Direct successor 24 years later** — 같은 H.E. Mason 저자. 본 종설의 §2-§3 방법론을 SDO/Hinode/IRIS/Solar Orbiter 시대로 갱신. 같은 구조, 같은 식, 새 데이터·코드. | ★★★★★ Direct lineage / 직계 |
| **#61 Abbo+ 2025** (A&A 702, A254) | **본 종설의 가장 직접적인 응용 / Most direct application of this review** — §2.1 contribution function, §2.4.5 non-equilibrium ionization, §3.6 EM, §3.7 FIP-corrected abundance, §4.2.3 Lyα Doppler dimming, §4.2.4 multilayer Fe IX/X imaging — 모든 도구가 #61에 직접 사용. | ★★★★★ Foundation / 기초 |
| **#63 Dere+ 1997 CHIANTI I** | **공동 저자 / Co-authorship** — H.E. Mason과 B.C. Monsignori Fossi가 모두 CHIANTI I의 저자. 본 종설의 §2-§3 방법론을 *데이터베이스로 코드화*한 것이 CHIANTI v1. | ★★★★★ Same authors → CHIANTI |
| **#64 Dere+ 2023 CHIANTI v10.1** | **본 종설의 진단법을 26년 갱신된 원자 데이터로 재실현 / Realisation of this review's diagnostics with 26-years-updated atomic data** — Burgess–Tully scaling, 향상된 R-matrix 충돌 데이터, Asplund 2021 abundance, 새 이온화 평형. | ★★★★ Implementation / 구현 |
| **van de Hulst 1950 (#62 in archive)** | **상호보완 / Complementary** — van de Hulst는 K-corona pB로부터 $n_e$ 도출 (광학적 얇은 *연속체*). 본 종설은 *분광선*에서 $n_e$, $T_e$, EM 도출. 두 진단의 결합이 #61의 핵심. | ★★★★ Complementary diagnostics |
| **Pottasch 1964** | **EM 분석법의 근원 / Origin of EM analysis** — 본 종설 Eq. 31 (isothermal EM)이 Pottasch 1964의 isothermal 근사를 직접 사용. | ★★★★ Methodological root |
| **Arnaud & Raymond 1992** | **이온화 평형의 표준 / Standard for ionization equilibrium** — 본 종설 Fig. 2에서 인용된 Fe 이온화 평형 새 계산. CHIANTI 초기 버전의 ionization data 출처. | ★★★ Atomic-data input |
| **Burgess & Tully 1992** | **충돌 데이터 평가법 / Collision-data evaluation method** — 본 종설이 Van Regemorter $\bar{g}$를 *abandoned*로 권고하면서 추천한 대체법. CHIANTI에서 표준으로 채택. | ★★★ Critical-evaluation method |
| **Walker et al. 1988** (multilayer 171-175 Å) | **EUI/FSI 17.4 nm의 기술 조상 / Technological ancestor of EUI/FSI 17.4 nm** — 본 종설 §4.2.4가 인용. 이후 SDO/AIA 171 Å, Solar Orbiter EUI/FSI 17.4 nm로 발전. | ★★★ Technology lineage |
| **Withbroe et al. 1982; Noci et al. 1987** | **Doppler dimming 이론 / Doppler dimming theory** — 본 종설 §4.2.3 + §3.8. #61의 Lyα Doppler dimming 비교 분석에 사용. | ★★★ Lyα diagnostic |
| **Asplund et al. 2021** (광구 abundance) | **본 종설 §3.7 FIP 논의의 현대 abundance / Modern abundance underlying §3.7** — #61이 사용한 두 abundance 값 중 하나. | ★★★ Abundance reference |
| **Harrison & Thompson 1992** (DEM 비교) | **SOHO CDS-SUMER 사전 시험 / SOHO CDS-SUMER pre-launch DEM benchmarking** — 본 종설 §3.6의 SOHO 직전 DEM 방법론 평가. | ★★ DEM methodology |

---

## 7. References / 참고문헌

### Atomic data and ionization equilibrium / 원자 데이터 및 이온화 평형
- Arnaud, M. & Rothenflug, R. *An updated evaluation of recombination and ionization rates*. **A&AS** 60, 425 (1985).
- Arnaud, M. & Raymond, J. *Iron ionization and recombination rates and ionization equilibrium*. **ApJ** 398, 394 (1992).
- Burgess, A. *Dielectronic recombination and the temperature of the solar corona*. **ApJ** 139, 776 (1964).
- Burgess, A. & Tully, J. A. *On the analysis of collision strengths and rate coefficients*. **A&A** 254, 436 (1992).
- Burgess, A. & Chidichimo, M. C. *Electron impact ionization of complex ions*. **MNRAS** 203, 1269 (1983).
- Mewe, R., Lemen, J. R. & van den Oord, G. H. J. *Calculated X-radiation from optically thin plasmas. V*. **A&AS** 65, 511 (1986).
- Aldrovandi, S. M. V. & Pequignot, D. *Radiative and dielectronic recombination coefficients for complex ions*. **A&A** 47, 321 (1976).

### Spectral diagnostics / 분광 진단
- Pottasch, S. R. *On the interpretation of the solar ultraviolet emission line spectrum*. **Space Sci. Rev.** 3, 816 (1964).
- McWhirter, R. W. P. *Spectral intensities*. In *Plasma Diagnostic Techniques*, eds. Huddlestone & Leonard (Academic Press 1965).
- Dere, K. P. & Mason, H. E. *Spectroscopic diagnostics of the active region transition zone and corona*. In *Solar Active Regions*, ed. F. Q. Orrall (Colorado AP 1981).
- Doschek, G. A. *Diagnostics of solar and astrophysical plasmas dependent on dielectronic recombination*. **ApJ** 296, 244 (1985).
- Mason, H. E. & Bhatia, A. K. *Coronal Mg VI/Si IX/S XI*. **MNRAS** 184, 423 (1978).

### Atomic codes and databases / 원자 코드 및 데이터베이스
- Eissner, W. & Seaton, M. J. *Computer programs for the calculation of electron–atom collision cross sections*. **J. Phys. B** 5, 2187 (1972).
- Berrington, K. A., et al. *RMATX*. **Comput. Phys. Commun.** 14, 367 (1978).
- Landini, M. & Monsignori Fossi, B. C. *The X-UV spectrum of thin plasmas*. **A&AS** 82, 229 (1990).
- Raymond, J. C. & Smith, B. W. *Soft X-ray spectrum of a hot plasma*. **ApJS** 35, 419 (1977).
- Mewe, R., Gronenschild, E. H. B. M. & van den Oord, G. H. J. *X-radiation. III*. **A&AS** 62, 197 (1985a).
- Summers, H. P. — ADAS package, JET fusion research.
- Sutherland, R. S. & Dopita, M. A. *Cooling functions for low-density astrophysical plasmas*. **ApJS** 88, 253 (1993).

### Solar observations / 태양 관측
- **Skylab**: Feldman, U., Behring, W. E., Curdt, W., et al. *A coronal spectral atlas (170–625 Å)* (1987); Doschek, G. A. & Cowan, R. D. (1984).
- **SMM**: Cheng, C.-C., et al. *Fe XXI 1354 Å in flares*. **ApJ** 233, 736 (1979); Mason, H. E., et al. (1986).
- **HRTS**: Brueckner, G. E. & Bartoe, J.-D. F. *HRTS instrument*. **ApJ** 272, 329 (1983); Dere, K. P., et al. *Power spectrum of C IV transition region*. **Sol. Phys.** 123, 41 (1989); Dere, K. P., et al. (1991) — explosive events.
- **Spartan UVCS**: Kohl, J. L., et al. (1994); Withbroe, G. L., et al. (1982); Noci, G., et al. (1987).
- **CHASE (Spacelab-2)**: Breeveld, A. A., et al. (1988); Gabriel, A. H., et al. (1994).
- **SERTS**: Neupert, W. M., et al. (1992); Thomas, R. J. & Neupert, W. M. (1994); Brosius, J. W., et al. (1994).
- **Multilayer**: Walker, A. B. C., et al. (1988); Golub, L., et al. (1990); Davila, J. M., et al. (1992); Walker, A. B. C., et al. *MSSTA* (1993).

### Abundance and FIP / 함량 및 FIP
- Meyer, J.-P. *The baseline composition of solar energetic particles*. **ApJS** 57, 151 (1985); reviews in 1990, 1993.
- Widing, K. G. & Feldman, U. *On the abundance variations in the corona*. **ApJ** 392, 401 (1992); 1994.
- Feldman, U. *Elemental abundances in the upper solar atmosphere*. **Phys. Scripta** 46, 202 (1992); reviews in 1992b, 1993.

### Profile analysis / 프로파일 분석
- Dere, K. P. & Mason, H. E. *Nonthermal velocities in the solar transition region*. **Sol. Phys.** 144, 217 (1993).
- Hassler, D. M., et al. *Coronal nonthermal broadening Mg X*. **ApJ** 348, L77 (1990).

### This paper / 이 논문
- Mason, H. E. & Monsignori Fossi, B. C. *Spectroscopic diagnostics in the VUV for solar and stellar plasmas*. **The Astronomy and Astrophysics Review** 6, 123–179 (1994). DOI: 10.1007/BF01208253

---

*※ 본 노트는 #61 Abbo+ 2025의 사전 학습 자료로 작성되었으며, 본 종설의 §2-§3에 등장하는 모든 핵심 진단 도구가 #61에서 어떻게 응용되는지를 명시적으로 강조했다. /*
*※ These notes were written as foundational background for #61 Abbo+ 2025, with explicit emphasis on how every key diagnostic tool from §2-§3 of this review is applied in #61.*
