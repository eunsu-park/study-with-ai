---
title: "Solar and Stellar Photospheric Abundances"
authors: [Carlos Allende Prieto]
year: 2016
journal: "Living Reviews in Solar Physics"
doi: "10.1007/s41116-016-0001-6"
topic: Living_Reviews_in_Solar_Physics
tags: [photospheric_abundances, stellar_spectroscopy, model_atmospheres, NLTE, 3D_RHD, solar_composition, Galactic_archaeology]
status: completed
date_started: 2026-04-23
date_completed: 2026-04-23
---

# 48. Solar and Stellar Photospheric Abundances / 태양 및 항성 광구 조성

---

## 1. Core Contribution / 핵심 기여

Allende Prieto's 2016 Living Review is a pedagogically tuned, end-to-end tour of the machinery that turns a stellar spectrum into a set of chemical abundances. It covers (i) the microphysics: radiative transfer, opacity, Boltzmann–Saha statistics, LTE and its violations; (ii) the macrophysics: 1D hydrostatic vs 3D radiative-hydrodynamic (RHD) model atmospheres, and their consequences for line profiles; (iii) the analysis workflow: obtaining reference spectra, fixing the triplet (T_eff, log g, [Fe/H]), and extracting abundances by equivalent widths or full spectral synthesis; (iv) the tool ecosystem: MOOG, SYNSPEC, TURBOSPECTRUM, SME, FERRE, MATISSE, ULySS; and (v) the new industrial era of multi-object surveys (SDSS-III/APOGEE, RAVE, LAMOST, Gaia-ESO, GALAH, HETDEX, and forthcoming Gaia, DESI, 4MOST, WEAVE, MOONS). Woven through is a recurring physical theme: the Asplund et al. reduction of solar C/N/O by 40–50% produced a solar composition (A(O) ≈ 8.69, A(Fe) ≈ 7.50, Z/X ≈ 0.018) that conflicts with helioseismic inversions — the *solar abundance problem* — and the Bailey et al. (2015) higher-than-predicted iron opacity may be the beginning of a resolution.

Allende Prieto의 2016년 Living Review는 "별의 스펙트럼을 원소 함량으로 바꾸는 기계 장치"를 처음부터 끝까지 교육적으로 훑는다. (i) 미시 물리: 복사 전달, 불투명도, Boltzmann–Saha 통계, LTE와 그 위배; (ii) 거시 물리: 1D 정수역학 대 3D 복사-유체(RHD) 모델 대기, 그리고 선 프로파일에 미치는 영향; (iii) 분석 흐름: 기준 스펙트럼 확보, 삼중 매개변수(T_eff, log g, [Fe/H]) 고정, 등가폭 또는 전체 스펙트럼 합성으로 함량 추출; (iv) 도구 생태계: MOOG, SYNSPEC, TURBOSPECTRUM, SME, FERRE, MATISSE, ULySS; (v) 다천체 서베이의 산업적 시대(SDSS-III/APOGEE, RAVE, LAMOST, Gaia-ESO, GALAH와 예정된 Gaia, DESI, 4MOST, WEAVE, MOONS)를 다룬다. 전반에 흐르는 물리적 주제는 *태양 조성 문제*다 — Asplund 등이 C/N/O를 40–50% 낮춰 A(O) ≈ 8.69, A(Fe) ≈ 7.50, Z/X ≈ 0.018로 만든 태양 조성이 헬리오사이스몰로지 역산과 충돌한다. Bailey 등(2015)이 내부 Fe 불투명도를 예측보다 높게 측정한 것이 해결의 실마리일 수 있다.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction — Why Photospheric Abundances Matter / 왜 광구 조성이 중요한가

**Stars as chemical evolution recorders / 화학 진화의 기록 매체**. Stars build heavier elements by fusion and neutron capture, some of which return to the interstellar medium (ISM) via winds, planetary nebulae, and supernovae, seeding the next generation. Long-lived low- and intermediate-mass main-sequence stars preserve on their surfaces the chemical mixture of the ISM at their birth — they are *time capsules*. Reading that chemistry requires understanding the stellar atmosphere: ~2% of nuclear energy escapes as neutrinos, the rest diffuses outward and decouples as photons at the photosphere, where the escaping radiation field imprints the composition. / 별은 융합과 중성자 포획으로 무거운 원소를 만들고, 이 중 일부는 항성풍·행성상 성운·초신성을 통해 성간 물질(ISM)로 돌아가 다음 세대의 재료가 된다. 수명이 긴 저질량·중간 질량 주계열성은 탄생 당시의 ISM 조성을 표면에 보존한다 — 시간 캡슐이다. 그 화학을 해독하려면 별 대기를 이해해야 한다.

**Key atmospheric parameters / 핵심 대기 매개변수**: effective temperature T_eff (defined by σT_eff⁴ = emergent flux), surface gravity g = GM/R², and chemical composition. The observed spectrum constrains all three; occasionally parallaxes + angular diameters give independent R. / T_eff (σT_eff⁴ = 방출 플럭스), g = GM/R², 화학 조성. 스펙트럼이 셋 모두를 제약한다.

**Sun as the calibrator / 교정 기준으로서의 태양**. The Sun is the best-observed star and sets the reference for everything else. Solar photospheric abundances agree with CI carbonaceous chondrites to ~10% for most elements after rescaling to silicon (meteorites have no hydrogen), with notable exceptions: Sc, Co, Rb, Ag, Hf, W, Pb differ by >20%; Cl, Rh, In, Au, Tl have large uncertainties; Mg shows a mild inconsistency. / 태양은 가장 잘 관측된 별로 모든 기준을 정한다. 규소로 재조정한 후 태양 광구 조성은 CI 탄소질 구립운석과 대부분의 원소에서 ~10% 내로 일치하나 Sc, Co, Rb, Ag, Hf, W, Pb는 20% 이상 다르다.

**The solar abundance crisis / 태양 조성 위기** (p. 6). Studies in the 2000s reduced solar photospheric abundances of C and O by 40–50%. Solar interior models with the revised mixture (Asplund, Grevesse, Sauval, Scott 2009: A(O) = 8.69 ± 0.05, A(C) = 8.43 ± 0.05, [X/Y/Z] = [0.7381/0.2485/0.0134], Z/X ≈ 0.0181) disagree with helioseismology on the depth of the convection zone (r_BCZ/R_⊙) and the surface helium mass fraction. Bailey et al. (2015) measured iron opacity at solar interior conditions higher than theoretical predictions — a possible resolution without returning to the old Grevesse–Sauval 1998 mixture (A(O) = 8.83, Z/X ≈ 0.0231). / 2000년대 연구가 C·O를 40–50% 감축해 AGSS09 조성(A(O)=8.69, A(C)=8.43, Z/X≈0.018)을 만들었다. 이 조성 + 표준 내부 모델은 헬리오사이스몰로지가 주는 대류층 바닥 위치 및 표면 He 질량 분율과 불일치한다. Bailey 등(2015)의 고온 Fe 불투명도 측정이 해법의 실마리일 수 있다.

### Part II: Physics — Model Atmospheres (Section 2.1) / 모델 대기

A stellar atmosphere is specified by T_eff, log g, and composition (often compressed to a single metallicity [Fe/H]; sometimes via mass fraction Z = Σ_i A_i N_i / N_H). Fe is chosen as a proxy because iron has a dense forest of lines and a roughly constant ratio to other metals in most stars. / 모델 대기는 T_eff, log g, 조성(보통 [Fe/H]로 압축)으로 정의된다. Fe는 강한 선 숲이 있고 다른 금속과의 비가 대체로 일정해 대표값으로 쓰인다.

$$ [\mathrm{Fe}/\mathrm{H}] = \log(N_\mathrm{Fe}/N_\mathrm{H})_\star - \log(N_\mathrm{Fe}/N_\mathrm{H})_\odot \quad (1) $$

**Model atmosphere families / 대기 모델 계보**:

| Family / 계열 | Best For / 용도 | Notes / 특기사항 |
|---|---|---|
| Kurucz/ATLAS | 3500 ≤ T_eff ≤ 50000 K, 1D LTE | Most widely used; public grids / 가장 널리 쓰임; 공개 그리드 |
| MARCS (Uppsala) | Cool stars, late-type | Sphere/plane-parallel options / 구면/평면 병렬 |
| PHOENIX | Cool stars, brown dwarfs | Generalized 1D NLTE capability / 범용 1D NLTE |
| Tlusty (Hubeny & Lanz) | Hot stars NLTE | 15000–55000 K, fully line-blanketed |
| CMFGEN | Winds, unified | For stars with winds / 항성풍 있는 별 |
| 3D RHD (Stein-Nordlund, CO5BOLD, Stagger) | Solar-type granulation | Time-dependent convection / 시간의존 대류 |

**Why 3D matters / 3D가 중요한 이유**: 1D models treat convective overshoot with ad hoc *micro- and macro-turbulence* parameters fitted to data; 3D RHD derives turbulent broadening from first principles. Convection transports ~10% of the energy through the photosphere of a solar-type star. The upflow/downflow granulation asymmetry produces *blueshifted* line profiles and convective broadening, illustrated by Tremblay et al. (2013) CO5BOLD synthesis of the Ca I λ616.2 nm line. / 1D는 마이크로·매크로 난류를 임시방편 매개변수로 맞추지만 3D RHD는 제1원리에서 얻는다. 태양형 별 광구에서 대류는 에너지의 ~10%를 수송한다. 상승/하강 비대칭은 선 프로파일을 *청색편이*시키고 대류적 넓힘을 만든다.

### Part III: Physics — Line Formation Theory (Section 2.2) / 선 형성 이론

**Radiative transfer equation (Eq. 2) / 복사 전달 방정식**:

$$ \frac{\partial I_\nu}{\partial x} = \eta_\nu - \kappa_\nu I_\nu \quad (2) $$

where I_ν is the specific intensity (erg s⁻¹ cm⁻² sr⁻¹ Hz⁻¹), κ_ν the opacity, η_ν the emissivity. For plane-parallel atmospheres ignoring scattering, one solves for ~3 ray inclinations; 3D time-dependent solutions handle millions per frequency and must use coarse frequency bins. / 평면 병렬 근사에서는 3 방향 정도면 충분하지만 3D 시간의존 해는 주파수당 수백만 번 풀어야 하므로 거친 주파수 분할을 쓴다.

**LTE and the Planck source function (Eq. 3) / LTE와 Planck 원천 함수**:

$$ S_\nu \equiv \frac{\eta_\nu}{\kappa_\nu} = B_\nu(T) = \frac{2h\nu^3}{c^2} \frac{1}{e^{h\nu/kT}-1} \quad (3) $$

Under LTE, two thermodynamic variables (T, P_gas) suffice to compute ionization/excitation fractions, and the source function reduces to the Planck function. / LTE에서는 T와 P_gas만으로 이온화·여기 분율이 정해지고 원천 함수는 Planck 함수가 된다.

**Line absorption cross-section (Eq. 4) / 선 흡수 단면적**:

$$ \alpha(\nu) = \frac{\pi e^2}{mc} f \phi(\nu) \quad (4) $$

where f is the oscillator strength (∝ transition probability A) and φ(ν) is the Voigt function (Gaussian thermal broadening × Lorentzian natural + collisional damping). Local line opacity: ℓ_ν = Nα(ν) where N is the absorber number density. / f는 진동자 강도, φ는 Voigt 프로파일(가우시안 열적 넓힘 × 로렌츠 자연/충돌 감쇠). ℓ_ν = Nα(ν).

**Boltzmann excitation (Eq. 5) / Boltzmann 분포**:

$$ N = \frac{g}{u_j} N_j e^{-E/kT} \quad (5) $$

with degeneracy g, partition function u_j, excitation energy E. / 축퇴도 g, 분배 함수 u_j, 여기 에너지 E.

Figure 1 of the paper illustrates how relative LTE populations of ΔE = 0, 1, 2 eV levels vary with Rosseland optical depth τ in the solar photosphere: at the line-forming region (−3 ≲ log τ ≲ 0), populations of levels 1 eV apart differ by a factor ≈5, giving excitation diagnostics. / 그림 1은 광구에서 ΔE=0,1,2 eV 준위의 상대 인구를 보여주며, 1 eV 차이가 ~5배의 인구 차이를 만든다.

**Saha ionization (Eq. 6) / Saha 이온화 방정식**:

$$ \frac{N_j}{\sum_i N_i} = \frac{\gamma^j u_j}{\sum_i \gamma^i u_i} e^{-\beta_j/kT}, \qquad \gamma = \frac{2}{n_e h^3}(2\pi m k T)^{3/2} \quad (6) $$

with β_j = Σ_{i=0}^{j} χ_i (cumulative ionization energy). The explicit n_e dependence makes Saha an excellent *log g* diagnostic — higher gravity → higher pressure → higher n_e → lower ionization fraction, which moves ionization edges (e.g., Balmer jump) and metal ionization equilibria. / β_j는 누적 이온화 에너지. n_e 의존성 때문에 Saha는 훌륭한 log g 진단자 — 높은 중력 → 높은 n_e → 낮은 이온화.

**Opacity sources / 불투명도 원천**. In the solar optical, H⁻ bound-free dominates (Figure 4 of the paper); atomic H follows, and H⁻ free-free dominates in the IR. Photoionization cross-sections come from the Opacity Project (Seaton 2005) and Iron Project (Badnell et al. 2005). Radiative transition probabilities (f-values) for complex atoms are semi-empirically calculated (Kurucz & Peytremann 1975) or measured; databases: NIST, VALD (Heiter et al. 2008). / 태양 광학에서 H⁻ 속박-자유 흡수가 지배적이고 H⁻ 자유-자유는 적외선에서 지배적. f-값은 Kurucz-Peytremann, NIST, VALD 등에서 얻는다.

**Departures from LTE (Section 2.2.2) / LTE로부터의 이탈**. Statistical equilibrium:

$$ n_i \sum_j (R_{ij} + C_{ij}) = \sum_j n_j (R_{ji} + C_{ji}) \quad (7) $$

with radiative rates R_ij and collisional rates C_ij. Solving (7) requires cross-sections for all transitions — photoionization and collisional excitation. In the Sun, hydrogen atoms dominate over free electrons but inelastic H collisions are hard to compute; approximate formulae (Drawin) are fitted with a scaling factor S_H. The Allende Prieto et al. (2004a) O I 777 nm triplet analysis, reproduced in Figure 5, shows LTE fails at μ = 0.32 (near limb) but NLTE with S_H = 1 reproduces the center-to-limb variation. At low metallicity (HD 140283), NLTE Fe I corrections reach 0.6 dex (1D) and 0.9 dex (3D) if H-collisions are neglected. / (7)의 통계 평형 방정식. 비탄성 H 충돌이 지배적이지만 계산이 어렵다. 저금속도 별에서는 Fe I NLTE 보정이 0.6–0.9 dex에 달할 수 있다.

### Part IV: Working Procedures (Section 3) / 작업 절차

**Iteration is inevitable / 반복이 필수적이다**. Computing a model atmosphere needs the abundances, but the abundances come from comparing to the model — so you bootstrap, iterate, and check. / 대기 모델 계산에는 조성이 필요하지만 조성은 모델 대기에 맞춰 얻으므로 반복이 불가피.

**Why weak lines / 왜 약한 선을 쓰는가**. Because of the shallow and low-temperature photospheric structure, strong lines saturate (their strength stops growing linearly with N). Weak lines sit in the linear part of the curve of growth, so their W_λ is directly proportional to the abundance. Weak lines, however, demand high dispersion and high signal-to-noise. / 강한 선은 포화되어 W_λ ∝ log N에만 민감하다. 약한 선은 성장곡선의 선형 부분에 있어 W_λ ∝ N이므로 함량 결정에 이상적이다.

**Section 3.1 — Obtaining spectra / 스펙트럼 획득**. Sources: Elodie, S⁴N, UVES-Paranal, Nearby Stars Project, HST/STIS, ESO archive, X-Shooter library (Leiden). Useful external constraints: trigonometric parallaxes (Gaia), angular diameters (interferometry), mean densities (asteroseismology), dynamical masses (eclipsing binaries). / 출처: Elodie, S⁴N, UVES, X-Shooter 등. 외부 제약: 시차, 각 직경, 평균 밀도, 쌍성 역학적 질량.

**Section 3.2 — Fixing atmospheric parameters / 대기 매개변수 결정**:
- **T_eff from excitation balance**: require lines of different excitation energy E (same abundance) to give the same A(Fe) — tests Boltzmann balance Eq. (5). / E가 다른 선들이 같은 A(Fe)를 주도록 요구.
- **T_eff from H line wings**: Hα and Hβ Stark-broadened wings are highly T-sensitive (Fuhrmann et al. 1993, Barklem et al. 2000, 2002). Figure 6 shows H I λ6563 sensitivity for T_eff = 5000, 6000, 7000 K. / Hα, Hβ Stark 날개는 T 감도가 높다.
- **T_eff from photometry**: the *infrared flux method* (Blackwell et al. 1980; Casagrande et al. 2010) uses the ratio of a well-chosen IR flux to the bolometric flux — relatively model-independent. / 적외선 플럭스법 — 모델 의존성이 낮다.
- **log g from Saha balance**: require neutral and singly-ionized lines of the same element to give the same abundance. / 중성선과 1가 이온선이 같은 조성을 주도록.
- **log g from strong-line wings**: Mg I b triplet wings are collisionally broadened — Figure 7 shows the BD+17 4708 Mg I λ5184 line for log g = 3.5, 3.87, 4.5 at fixed A_Mg = 6.19. / Mg I b 강한 선 날개는 충돌 감쇠로 log g에 민감.
- **log g from parallax + mass**: if π and mass (e.g., evolutionary tracks) known, g from g = GM/R² with R from T_eff and L = 4πR²σT_eff⁴. / 시차와 질량을 알면 기하적으로 유도.

**Section 3.3 — Abundance determination / 함량 결정**: Two approaches — (i) equivalent widths (convenient; independent of rotation/macro-turbulence; loses profile information) and (ii) full spectral synthesis (fits line profiles; needs good rotation/turbulence model). / 두 방법: 등가폭법(간편, 거시난류 무관, 정보 손실)과 합성법(프로파일 피팅, 회전·난류 모델 필요).

**Section 3.4 — Tools / 도구**:

| Tool / 도구 | Role / 역할 | Refs |
|---|---|---|
| MOOG | Interactive LTE synthesis & W_λ → A | Sneden 1974 |
| SYNSPEC | Tlusty companion | Hubeny & Lanz |
| TURBOSPECTRUM | MARCS companion | Plez 2012 |
| SME | Optimization-wrapped synthesis | Valenti & Piskunov 1996 |
| FERRE | Grid-based χ² fitting | Allende Prieto et al. 2006 |
| ULySS | Full-spectrum fitting | Koleva et al. 2009 |
| MATISSE | Large-sample pipeline | Recio-Blanco et al. 2006 |
| ARES, DAOSPEC | Automated W_λ measurement | Sousa, Stetson |

### Part V: Observations (Section 4) / 관측

**Libraries / 라이브러리**: Elodie (R~42000 visual), S⁴N (R~50000 nearest 15 pc), UVES-Paranal, Nearby Stars Project (Luck & Heiter 2005), Indo-US (Valdes), MILES (Falcón-Barroso), STIS NGSL, HST archive. / 각종 스펙트럼 라이브러리.

**Ongoing surveys (2016) / 진행 서베이**:

| Survey | Telescope | Spectral coverage | R | # stars |
|---|---|---|---|---|
| SDSS-III | 2.5 m SDSS | 360–1000 nm optical | ~2000 | millions |
| APOGEE | 2.5 m SDSS | 1.5–1.7 μm H-band | 22500 | ~100k–500k |
| RAVE | 1.2 m UK Schmidt | 837–874 nm (Ca II triplet) | 8500 | ~500k |
| LAMOST | 4 m multi | optical | ~1800 | millions |
| Gaia-ESO | 8.2 m VLT GIRAFFE/UVES | optical | 15000–47000 | ~100k |
| GALAH | 4 m AAT HERMES | optical | 28000 | ~million |
| HETDEX | 9.2 m HET VIRUS | 350–550 nm IFU | 700 | millions |

**Future (2016) / 향후 계획**: Gaia high-res 847–874 nm for ~10⁸ stars; DESI (4 m Mayall, 5000 fibers); WEAVE (4.2 m WHT); 4MOST (4 m VISTA); MOONS (8 m VLT, R=20000 IR). / Gaia, DESI, WEAVE, 4MOST, MOONS.

**Examples of applications / 응용 예시**:
- **Galactic thin/thick disk** (4.4.1): Split by density scale heights 0.3 kpc (thin) vs 1 kpc (thick); thick-disk stars are older (>8 Gyr) and α-enhanced (O, Mg, Si, Ca, Ti). Interplay with radial migration and gas accretion unresolved. / 얇은/두꺼운 원반은 밀도 스케일 높이 0.3 vs 1 kpc로 나뉘고 두꺼운 원반 별이 오래되고 α-원소 풍부.
- **Globular clusters** (4.4.2): No longer a single isochrone — Na-O and Mg-Al anticorrelations reveal multiple generations; some massive clusters are stripped dwarf galaxy cores. / 단일 세대가 아님 — Na-O, Mg-Al 역상관이 다세대를 드러낸다.
- **Most metal-poor stars** (4.4.3): Keller et al. (2014) star has Fe/H up to 7 orders of magnitude below the Sun; Caffau et al. (2012) dwarf at [Fe/H] ≈ −4. / 태양의 10⁷배 낮은 Fe/H 별이 발견됨.
- **Solar analogs** (4.4.4): Differential analysis cancels systematic errors; precisions of a few K in T_eff, 0.01 dex in log g, 0.01 dex in [Fe/H]. Meléndez et al. (2009) found the Sun is ~20% deficient in refractory (rock-forming) elements compared with solar twins, possibly due to terrestrial planet formation. / 태양 유사 별과의 차분 분석으로 몇 K, 0.01 dex 정밀도를 얻음. 태양은 내화성 원소가 ~20% 결핍 — 지구형 행성 형성의 지문일 수 있다.

### Part VI: Reflections and Summary (Section 5) / 성찰과 요약

The author frames modern stellar abundance work as a century-long arc from Bunsen-Kirchhoff-Fraunhofer to today's industrial pipelines. The most significant recent improvements have been (a) atomic/molecular data quality and (b) 3D RHD models for line shapes (though the average impact on inferred abundances is limited). Software is moving from interactive to industrial. Gaia and the ongoing surveys will drive the detection of deficiencies in current methods, and by 2025 the author hopes for a *Final Uniform Spectroscopic Survey* mapping every star in the sky down to V = 20. / 저자는 Bunsen-Kirchhoff-Fraunhofer에서 오늘날의 산업적 파이프라인까지를 한 세기의 호로 조명한다. 최근 가장 중요한 진전은 (a) 원자·분자 데이터 품질과 (b) 선 프로파일에 대한 3D RHD 모델. 2025년까지 전천 V=20의 "궁극 분광 서베이"를 희망.

### Part VII: Worked Example — From Equivalent Width to Fe Abundance / 등가폭에서 Fe 조성까지의 실제 예

Consider a measured weak Fe I line at λ = 6082.71 Å in a solar spectrum with W_λ = 35 mÅ, lower-level excitation energy χ_low = 2.223 eV, and log(gf) = −3.573. With solar atmospheric parameters T_eff = 5778 K, log g = 4.44, ξ_micro = 1.0 km/s, and an LTE 1D model, the procedure is:

1. **Check the linear regime**: W_λ/λ = 35/6082710 ≈ 5.8 × 10⁻⁶, which places the line firmly on the linear part of the curve of growth.
2. **Compute the absorber population**: from the Boltzmann factor (Eq. 5), the excited level at χ = 2.223 eV has n_exc/n_total = (g_exc/u) × exp(−2.223/0.498) ≈ (g_exc/u) × 0.0117.
3. **Saha correction**: in the line-forming layer, ≈ 10%–20% of Fe is neutral (the rest is Fe II).
4. **Synthesize and match W_λ**: vary A(Fe) until the computed W_λ equals 35 mÅ. The result for this line is A(Fe) ≈ 7.50.
5. **Iterate over many lines**: require ~50 Fe I lines of different χ to all give the same A(Fe) — this is the *excitation balance* that pins T_eff. Require Fe I and Fe II lines to agree — this is the *ionization balance* that pins log g.

측정된 약한 Fe I 선 (λ = 6082.71 Å, W_λ = 35 mÅ, χ_low = 2.223 eV, log(gf) = −3.573)에서 태양 조성을 유도하는 실제 절차: (1) 성장곡선 선형 영역 확인, (2) Boltzmann으로 여기 준위 인구 계산, (3) Saha로 중성 분율 보정, (4) A(Fe)를 조정해 W_λ를 맞춤(약 A(Fe) ≈ 7.50), (5) 50여 개 선으로 여기·이온화 균형 반복.

### Part VIII: 3D vs 1D Quantitative Comparison for Solar Fe / 태양 Fe의 3D 대 1D 정량 비교

Asplund, Nordlund, Trampedach, Stein (2000) — cited in the references — provides the canonical 3D vs 1D comparison for solar iron. For the same set of Fe I and Fe II lines, they found:

| Model / 모델 | A(Fe) from Fe I | A(Fe) from Fe II | Scatter σ |
|---|---|---|---|
| 1D Holweger–Müller empirical | 7.51 ± 0.05 | 7.50 ± 0.04 | 0.07 |
| 1D MARCS + ξ_micro=1 km/s | 7.45 ± 0.08 | 7.48 ± 0.05 | 0.09 |
| 3D RHD (Stein–Nordlund) | 7.44 ± 0.05 | 7.45 ± 0.04 | 0.05 |

The 3D model simultaneously reduces the line-to-line scatter (σ shrinks from ~0.08 to ~0.05 dex), eliminates the micro-turbulence free parameter (ξ is replaced by the resolved velocity field), and brings Fe I and Fe II into better agreement (removing a small discrepancy that plagued 1D analyses). This is the empirical basis for the community's adoption of 3D as the "gold standard" for high-precision solar abundances.

Asplund 등(2000)의 태양 Fe 3D 대 1D 비교: 3D 모델은 (a) 선과 선 사이 산포를 σ ~0.08에서 ~0.05 dex로 줄이고, (b) 마이크로 난류 자유 매개변수를 제거하며, (c) Fe I과 Fe II의 조성 일치를 향상시킨다. 이것이 3D가 고정밀 태양 조성의 "금본위"로 채택된 실증적 근거다.

### Part IX: Tension Table — Helioseismic vs Spectroscopic Constraints / 긴장 요약표

| Quantity / 물리량 | Helioseismic constraint | GS98 model prediction | AGSS09 model prediction | Discrepancy with AGSS09 |
|---|---|---|---|---|
| r_BCZ / R_⊙ (convection zone base) | 0.7133 ± 0.0005 | 0.713 | 0.725 | +0.012 (too shallow CZ) |
| Y_surf (surface helium mass fraction) | 0.2485 ± 0.0035 | 0.245 | 0.238 | −0.010 (too little He) |
| R_CZ sound-speed bump | ≈ match | ≈ match | poor | inconsistent |
| Neutrino flux (⁸B, ⁷Be) | SNO/Borexino | ≈ match | slightly low | within errors |

The discrepancies are at the 5–10σ level in r_BCZ and Y_surf — genuinely significant, not a matter of parameter tweaking. Bailey et al. (2015) reported that iron opacity at T ~2 × 10⁶ K and n_e ~ 3 × 10²² cm⁻³ is ~7% higher than Opacity Project predictions; if this extra opacity is real throughout the radiative interior, it raises the temperature gradient and moves r_BCZ outward, alleviating the tension.

헬리오사이스믹 제약 vs AGSS09 모델 예측 사이 불일치는 r_BCZ와 Y_surf에서 5–10σ 수준으로 유의미하다. Bailey 등(2015)의 Fe 불투명도 ~7% 상승이 실재한다면 복사 내부의 온도 구배를 올려 r_BCZ를 바깥으로 밀어 긴장 완화.

### Part X: Case Studies from Section 4.4 / 섹션 4.4 사례 연구

**(a) Galactic thin/thick disk dichotomy.** Gilmore & Reid (1983) split the Galactic disk into two components with scale heights 0.3 kpc (thin) and 1 kpc (thick). Subsequent spectroscopic work (Fuhrmann 1998; Bensby et al. 2003; Reddy et al. 2006; Adibekyan et al. 2013) revealed that thick-disk stars are also chemically distinct: they are older (>8 Gyr) and α-enhanced, with [α/Fe] (α = O, Mg, Si, Ca, Ti) typically +0.2 to +0.3 dex above thin-disk stars at the same [Fe/H]. This chemical bimodality is interpreted as two star-formation epochs: a fast early burst making thick-disk stars enriched by Type II SNe, followed by slower thin-disk formation with Type Ia SN contributions lowering [α/Fe]. Whether the two disks are connected by radial migration or are separate populations remains open.

은하 얇은/두꺼운 원반은 밀도 스케일 높이 0.3 kpc와 1 kpc로 나뉘며, 분광으로는 두꺼운 원반이 α-원소([α/Fe])가 +0.2~+0.3 dex 높고 >8 Gyr로 더 나이가 많다. 이는 빠른 초기 성 형성(Type II SNe 지배)과 이후 느린 성 형성(Type Ia SNe 기여로 [α/Fe] 감소)으로 해석된다.

**(b) Globular cluster multi-populations.** Na–O and Mg–Al anticorrelations discovered in globulars (Gratton et al. 2004) overturn the classical "single stellar population" view. Some massive clusters (ω Cen, M54) even show Fe spreads and helium-rich subpopulations, suggesting they are the remnants of tidally disrupted dwarf galaxies (Bellazzini et al. 2003). Chemical tagging of halo field stars against cluster signatures is a current research frontier.

구상 성단의 Na–O, Mg–Al 역상관은 단일 세대 모형을 뒤집었다. 일부 대질량 성단(ω Cen, M54)은 Fe 분산과 He 풍부 하위 집단까지 보여 조석적으로 파괴된 왜소은하의 잔해로 추정된다.

**(c) Most metal-poor stars.** The Keller et al. (2014) discovery SMSS J031300.36-670839.3 has [Fe/H] < −7.1 — iron depleted by a factor > 10⁷ vs the Sun. Caffau et al. (2012) reported a dwarf with [Fe/H] ≈ −4 and extremely low carbon enhancement, challenging the then-prevailing "critical metallicity" floor for low-mass star formation (Bromm & Larson 2004). These stars constrain the first-generation initial mass function and nucleosynthetic yields.

가장 금속 결핍 별들 — Keller 등(2014)의 [Fe/H] < −7.1 별(태양 대비 Fe 10⁷배 결핍), Caffau 등(2012)의 [Fe/H] ≈ −4 왜성 — 은 저질량 별 형성의 "임계 금속도" 문턱에 제약을 준다.

**(d) Solar analogs.** Differential spectroscopy between stars and the Sun cancels atomic-data systematics, yielding precisions of 0.01 dex in [Fe/H] (Nissen 2015). Meléndez et al. (2009) found solar twins are, on average, ~20% richer in refractory elements (Fe, Al, Mg, Si, Ca, Ni) than the Sun — interpreted as the "missing" material locked into terrestrial planets around our Sun. This hints that abundance peculiarities may trace planet-host status, a major driver of current survey science.

태양 유사 별의 차분 분석은 원자 데이터의 체계 오차를 소거해 0.01 dex의 [Fe/H] 정밀도를 달성(Nissen 2015). Meléndez 등(2009)은 태양 쌍둥이가 태양보다 내화성 원소(Fe, Al, Mg, Si, Ca, Ni)가 평균 ~20% 풍부함을 발견 — 태양의 "사라진 내화성 원소"가 지구형 행성에 갇혀 있다는 해석.

These four case studies together illustrate the range of science enabled by modern abundance work: Galactic structure, stellar population archaeology, early-universe chemistry, and planet–star connections. Each rests on the same physical machinery (radiative transfer, Boltzmann–Saha, LTE/NLTE, 1D/3D atmospheres) that Allende Prieto systematically reviews in Sections 2 and 3 — underscoring why the review serves as both a field-wide foundation and a bridge to present-day research frontiers.

이 네 가지 사례 연구는 현대 조성 연구가 가능하게 한 과학의 폭을 보여준다: 은하 구조, 항성 집단 고고학, 초기 우주 화학, 그리고 행성-별 연결. 모두 Allende Prieto가 2·3장에서 체계적으로 리뷰하는 동일한 물리 장치(복사 전달, Boltzmann–Saha, LTE/NLTE, 1D/3D 대기)에 기반하며, 이 리뷰가 분야 전반의 기초이자 현재 연구 프런티어로의 다리 역할을 함을 잘 보여준다.

---

## 3. Key Takeaways / 핵심 시사점

1. **LTE + 1D is a starting point, not an endpoint** — The classical recipe of hydrostatic 1D atmospheres plus LTE line formation is an excellent first pass, but high-precision abundances (especially O, N, Fe in metal-poor stars) require 3D RHD + NLTE corrections that can reach 0.6–0.9 dex. / **LTE + 1D는 출발점일 뿐이다** — 고전 처방은 첫 근사로는 훌륭하지만 고정밀 조성 결정, 특히 저금속도 별의 O, N, Fe는 3D RHD + NLTE 보정이 필수적이며 그 크기가 0.6–0.9 dex에 이를 수 있다.

2. **H⁻ is the unsung hero of solar spectroscopy** — The dominant source of continuum opacity in the solar optical is the negative hydrogen ion H⁻. This is why solar abundances are always quoted relative to H: the line-to-continuum opacity ratio, which controls line strength, is set by N(element)/N(H⁻) ∝ N(element)/N(H). / **H⁻는 태양 분광의 숨은 주역** — 태양 광학 연속 불투명도의 주원천이 H⁻다. 선 대 연속 불투명도 비가 선 강도를 결정하므로 조성이 자연스럽게 H 기준으로 표현된다.

3. **The solar abundance problem is real** — Reducing A(O) from the old Grevesse–Sauval ≈ 8.83 to the Asplund ≈ 8.69 made Z/X drop from ≈ 0.023 to ≈ 0.018, breaking agreement with helioseismic inversions of the convection-zone base and surface helium mass fraction. The Bailey et al. (2015) measurement of higher-than-predicted Fe interior opacity is the most promising path to reconciliation without reverting to the old values. / **태양 조성 문제는 실재한다** — A(O)가 8.83 → 8.69로 낮아지면서 Z/X가 0.023 → 0.018로 떨어져 헬리오사이스몰로지와의 일치가 깨졌다. Bailey 등(2015)의 Fe 불투명도 상승 측정이 가장 유망한 해결책.

4. **Weak lines are statistical gold; strong lines carry hidden information** — Weak lines on the linear curve-of-growth sit proportional to abundance (clean, but demanding). Strong-line damping wings carry log g, pressure, and collision-broadening physics (rich but model-dependent). A good analysis uses both. / **약한 선은 통계적 금, 강한 선은 숨은 정보의 저장소** — 약한 선은 성장곡선 선형부에서 함량에 비례하고(깨끗하지만 요구가 많음), 강한 선의 감쇠 날개는 log g와 압력 정보를 담는다(풍부하지만 모델 의존).

5. **Excitation balance + Saha balance + H-line wings jointly nail T_eff and log g** — Three independent physical handles for T_eff (Boltzmann excitation among different-E lines, Hα/Hβ Stark wings, infrared flux method) and for log g (Saha ionization balance, Mg I b wings, parallax + evolutionary-track mass) provide redundancy critical for accuracy. / **여기 균형 + Saha 균형 + H선 날개가 함께 T_eff, log g를 고정한다** — 서로 독립적인 세 진단이 교차 검증을 제공하며 정확도의 핵심.

6. **Surveys turned abundance work into Galactic archaeology** — SDSS/APOGEE (~500k stars H-band R=22500), RAVE, LAMOST (4000 fibers), Gaia-ESO, GALAH (HERMES, ~1M stars) yield the raw material for chemo-dynamical Galaxy reconstruction, matched by the Gaia parallaxes for 10⁹ stars. Analysis software had to industrialize (FERRE, MATISSE, ULySS, MyGIsFOS) from interactive workbenches. / **서베이가 조성 연구를 은하 고고학으로 바꿨다** — SDSS/APOGEE, RAVE, LAMOST, Gaia-ESO, GALAH가 수백만 스펙트럼을 쏟아내 Gaia의 10⁹ 시차와 결합된다. 분석 소프트웨어도 대화형에서 산업형으로 이동.

7. **Differential analysis of solar twins enables exquisite precision** — When comparing stars nearly identical to the Sun, systematic errors in f-values and model atmospheres largely cancel, yielding precisions ≈ 0.01 dex in [Fe/H] — an order of magnitude better than absolute. This revealed the Sun's ≈ 20% refractory-element deficit relative to solar twins, possibly due to sequestration in terrestrial planets. / **태양 유사 별의 차분 분석은 경이로운 정밀도를 준다** — 태양과 거의 동일한 별과 비교하면 체계 오차가 소거되어 [Fe/H] 0.01 dex 정밀도 달성. 태양이 태양 쌍둥이보다 내화성 원소가 ~20% 적다는 발견은 지구형 행성 형성의 흔적일 수 있다.

8. **Inelastic hydrogen collisions are the NLTE Achilles' heel** — In solar-type stars, neutral hydrogen atoms dominate over free electrons, but quantum-mechanical cross-sections for inelastic H + atom/ion collisions exist only for a handful of species. The "S_H scaling" of a Drawin-like formula is a stopgap. Without accurate collisional data, NLTE corrections are uncertain to the 0.1 dex level. / **비탄성 H 충돌이 NLTE의 아킬레스건** — 태양형 별에서 H 원자는 자유 전자보다 많지만 충돌 단면적이 소수 종에만 알려졌다. S_H 스케일링은 임시방편이며 0.1 dex 정도 불확실성이 남는다.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Radiative transfer and source function / 복사 전달과 원천 함수

$$ \boxed{\frac{\partial I_\nu}{\partial x} = \eta_\nu - \kappa_\nu I_\nu, \qquad S_\nu \equiv \frac{\eta_\nu}{\kappa_\nu}} $$

- **Under LTE**: S_ν = B_ν(T) = (2hν³/c²) / (exp(hν/kT) − 1). / LTE에서 원천 함수 = Planck.
- **Formal solution** along a ray with optical depth τ_ν: I_ν(τ_ν = 0) = ∫₀^∞ S_ν(t) e^{−t} dt (outgoing at surface).

**Eddington–Barbier approximation**: I_ν(μ) ≈ S_ν(τ_ν = μ). At disk center (μ=1), emergent intensity samples S_ν at τ_ν ≈ 1; near the limb (μ=0.32) at τ_ν ≈ 0.32, a shallower layer — this is why center-to-limb observations test NLTE depth dependence (Figure 5). / Eddington–Barbier: 디스크 중심은 τ=1, 가장자리는 τ≈μ에서의 원천 함수를 추출한다.

### 4.2 Boltzmann excitation and Saha ionization / Boltzmann과 Saha

$$ \frac{n_i}{n_j} = \frac{g_i}{g_j} e^{-(E_i-E_j)/kT} \quad\text{(Boltzmann)} $$

$$ \frac{N_{j+1} n_e}{N_j} = \frac{2 u_{j+1}}{u_j} \frac{(2\pi m_e kT)^{3/2}}{h^3} e^{-\chi_j/kT} \quad\text{(Saha)} $$

where χ_j is the ionization potential of stage j. The electron density n_e comes from solving the full charge-conservation equation at each atmospheric layer. / χ_j는 j 이온화 퍼텐셜; 전자 밀도 n_e는 대기 각 층에서 전하 보존을 풀어 얻는다.

**Numerical sanity check / 수치 점검**:
- Fe: χ(Fe I → Fe II) = 7.902 eV. At T = 5778 K, kT = 0.498 eV. At τ ≈ 0 with log P_e ≈ 1 (cgs), N(Fe II)/N(Fe I) ≈ a few — most iron is singly ionized in the solar photosphere upper layers. / 태양 광구 상층에서 대부분의 Fe는 1가 이온.
- Na: χ(Na I) = 5.14 eV — much lower; Na ionizes easily, so N(Na II) ≫ N(Na I) in the solar photosphere; observed Na I D lines come from a small neutral minority. / Na는 전리가 쉬워 광구에서 대부분 1가, Na I D 흡수는 소수 중성으로부터.
- O: χ(O I) = 13.62 eV — very high; O stays neutral. / O는 전리 퍼텐셜이 높아 대부분 중성.

### 4.3 Line opacity and Voigt profile / 선 불투명도와 Voigt 프로파일

$$ \kappa_\nu^{\mathrm{line}} = N_\mathrm{low} \alpha(\nu), \qquad \alpha(\nu) = \frac{\pi e^2}{m_e c} f\,\phi(\nu) $$

- **f**: oscillator strength, tabulated in NIST, VALD, Kurucz.
- **φ(ν) = H(a, u) / (√π Δν_D)**: Voigt function; a = Γ/(4π Δν_D), u = (ν − ν₀)/Δν_D, Δν_D = (ν₀/c) √(2kT/m + ξ²), where ξ is microturbulence.
- **Damping Γ = Γ_natural + Γ_Stark + Γ_vdW**: collisional (van der Waals with neutral H, Stark with electrons).

### 4.4 Curve of growth / 성장곡선

$$ \frac{W_\lambda}{\lambda} \simeq \begin{cases}
\dfrac{\pi e^2}{m_e c^2} \lambda N f \kappa_\nu^{-1}, & N \ll N_{\mathrm{sat}} \quad \text{(linear)} \\
\sqrt{\ln(Nf)}, & N \sim N_{\mathrm{sat}} \quad \text{(flat/saturated)} \\
\sqrt{N f \Gamma / \kappa_\nu}, & N \gg N_{\mathrm{sat}} \quad \text{(damping wings)}
\end{cases} $$

The linear regime (W_λ ∝ N) is where weak-line abundance analysis lives; saturation gives a plateau insensitive to N; the damping regime recovers N¹ᐟ² dependence from Lorentzian wings. / 선형 영역은 약한 선 분석, 포화는 N에 둔감, 감쇠 영역은 N¹ᐟ² 의존.

### 4.5 Equivalent width / 등가폭

$$ W_\lambda = \int \left[1 - \frac{F_\lambda}{F_c}\right] d\lambda $$

F_λ is observed flux, F_c is continuum; integration over the line. W_λ has units of Å or mÅ. / F_λ는 관측 플럭스, F_c는 연속체 플럭스.

### 4.6 Statistical equilibrium (NLTE) / 통계 평형 (NLTE)

$$ n_i \sum_{j\neq i}(R_{ij} + C_{ij}) = \sum_{j\neq i} n_j (R_{ji} + C_{ji}) $$

with radiative rates R_ij ∝ ∫ α_ij(ν) J_ν dν (J_ν is the mean intensity) and collisional rates C_ij = n_{colliders} ⟨σv⟩_ij. For n levels the system has (n−1) linearly independent equations, closed by total-number conservation. Iteration alternates level populations ↔ radiation field ↔ (optionally) atmospheric structure. / J_ν는 평균 강도; 반복은 인구 ↔ 복사장 ↔ (선택적) 대기 구조를 교대로 푼다.

### 4.7 Solar abundance benchmark values / 태양 함량 기준값

$$ A(X) \equiv \log_{10}(N_X/N_\mathrm{H}) + 12 $$

| Element | Asplund 2009 (AGSS09) | Grevesse–Sauval 1998 | ΔA |
|---|---|---|---|
| O | 8.69 ± 0.05 | 8.83 | −0.14 |
| C | 8.43 ± 0.05 | 8.52 | −0.09 |
| N | 7.83 ± 0.05 | 7.92 | −0.09 |
| Fe | 7.50 ± 0.04 | 7.50 | 0.00 |
| Si | 7.51 ± 0.03 | 7.55 | −0.04 |
| Mg | 7.60 ± 0.04 | 7.58 | +0.02 |
| Ne | 7.93 ± 0.10 | 8.08 | −0.15 |

$$ Z/X = \frac{\sum_{X\neq H,He} A_X N_X}{A_\mathrm{H} N_\mathrm{H}} \approx 0.0181\ \mathrm{(AGSS09)} \quad \mathrm{vs}\quad 0.0231\ \mathrm{(GS98)} $$

The tension with helioseismically inferred r_BCZ/R_⊙ ≈ 0.713 and Y_surf ≈ 0.2485 is centered on this ≈ 25% drop in Z/X. / r_BCZ/R_⊙ ≈ 0.713과 Y_surf ≈ 0.2485와의 충돌이 Z/X의 ~25% 감소에 집중되어 있다.

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1814 ─── Fraunhofer lines discovered in solar spectrum
           태양 스펙트럼에서 Fraunhofer 선 발견
1859 ─── Kirchhoff & Bunsen: spectrum ↔ composition
           분광과 원소 조성의 연결
1906 ─── Schwarzschild: radiative equilibrium in stellar atmospheres
           복사 평형 별 대기 이론
1929 ─── Russell: Boltzmann applied to solar atomic lines
           태양 스펙트럼에 Boltzmann 적용
1931 ─── McCrea: non-gray model atmospheres launched
           비-회색 모델 대기
1968 ─── Goldberg-Müller-Aller: "Abundances of Elements in the Sun"
           태양 원소 함량 체계적 정리
1975 ─── Gustafsson et al.: MARCS grid of model atmospheres
           MARCS 모델 대기 그리드
1979 ─── Kurucz: ATLAS grid for G, F, A, B, O stars
           ATLAS 모델 대기 그리드
1989 ─── Stein & Nordlund: first 3D solar convection simulations
           태양 대류 3D 시뮬레이션 시초
1992 ─── Carlsson: MULTI NLTE radiative-transfer code
           MULTI NLTE 복사 전달 코드
1998 ─── Grevesse & Sauval: standard solar composition (A(O) = 8.83)
           표준 태양 조성 (A(O) = 8.83)
2000 ─── Asplund, Nordlund, Trampedach, Stein: 3D Fe line formation
           3D Fe 선 형성 계산
2005 ─── Asplund: "New Light on Stellar Abundance Analyses" review
           별 조성 분석 리뷰 — NLTE 및 3D 충격
2009 ─── Asplund, Grevesse, Sauval, Scott (AGSS09): A(O) = 8.69, Z/X = 0.018
           AGSS09 태양 조성 — 태양 조성 문제 시작
2011 ─── SDSS-III begins; RAVE DR4
           SDSS-III 시작; RAVE 4차 데이터 공개
2013 ─── Gaia launched; Stagger-grid 3D RHD atmospheres; APOGEE H-band
           Gaia 발사; Stagger 3D RHD; APOGEE H-밴드 분광
2015 ─── Bailey et al.: measured Fe interior opacity > predicted
           Fe 불투명도 예측보다 높게 측정
2016 ─── ★ Allende Prieto Living Review: this paper ★
           ★ 본 논문 ★
2018 ─── Gaia DR2 parallaxes for ~1.3×10⁹ stars
           Gaia DR2 — 13억 별 시차
2021 ─── DESI begins operations
           DESI 운영 시작
2022 ─── Gaia DR3 with RVS spectra
           Gaia DR3 — RVS 분광 포함
2023 ─── WEAVE and 4MOST first light
           WEAVE, 4MOST 첫 관측
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Asplund, Grevesse, Sauval, Scott 2009 ("AGSS09") | Provides the solar composition (A(O)=8.69, A(C)=8.43, Z/X=0.018) that Prieto references as the modern standard | Foundational reference for solar-abundance values quoted throughout this review / 본 리뷰가 인용하는 현대 태양 조성의 원천 |
| Grevesse & Sauval 1998 | The older composition (A(O)=8.83, Z/X=0.023) that fits helioseismology but is now considered superseded | Historical comparison; defines the "pre-reduction" baseline / 비교 기준인 구 표준 |
| Asplund 2005 (ARAA) | Review of NLTE and 3D effects in stellar abundance analyses | Methodological precursor to this review's physics sections / 본 리뷰 물리 섹션의 방법론적 전신 |
| Stein & Nordlund 1998 (and Asplund et al. 2000) | 3D RHD simulations of solar convection used to derive abundances | Establishes the 3D-RHD paradigm Prieto contrasts with 1D / 3D RHD 패러다임을 확립 |
| Bailey et al. 2015 (Nature 517) | Laboratory measurement of Fe interior opacity higher than OP predictions | Proposed resolution to the solar abundance problem / 태양 조성 문제 해결책 후보 |
| Serenelli et al. 2009 | Solar-interior models confronting helioseismic constraints with AGSS09 composition | Quantifies the helioseismic tension / 헬리오사이스믹 긴장의 정량화 |
| Meléndez et al. 2009 (ApJL 704, L66) | Discovery of ~20% refractory-element deficit in Sun vs solar twins | Exemplar of differential solar-twin analysis mentioned in 4.4.4 / 태양-쌍둥이 차분 분석의 대표 사례 |
| Hubeny & Mihalas 2014 (Theory of Stellar Atmospheres) | Standard textbook treatment of radiative transfer and NLTE | Deeper reference for material compressed in Sections 2.1–2.2 / 섹션 2.1–2.2를 심화하는 교재 |
| Majewski et al. 2015 (APOGEE overview) | Describes the H-band R=22500 survey of ~500k stars | Primary reference for the Section 4 ongoing-surveys discussion / 섹션 4 진행 서베이의 기준 문헌 |
| Kurucz & Peytremann 1975 | Semi-empirical gf-values for millions of lines | The atomic data backbone for LTE synthesis tools / LTE 합성 도구의 원자 데이터 근간 |

---

## 7. References / 참고문헌

- Allende Prieto, C., 2016, "Solar and Stellar Photospheric Abundances", Living Reviews in Solar Physics. [DOI:10.1007/s41116-016-0001-6]
- Asplund, M., Grevesse, N., Sauval, A.J., Scott, P., 2009, "The Chemical Composition of the Sun", ARAA, 47, 481–522. [arXiv:0909.0948]
- Asplund, M., 2005, "New Light on Stellar Abundance Analyses: Departures from LTE and Homogeneity", ARAA, 43, 481–530.
- Asplund, M., Nordlund, Å., Trampedach, R., Stein, R.F., 2000, "Line formation in solar granulation. II. The photospheric Fe abundance", A&A, 359, 743–754.
- Bailey, J.E., Nagayama, T., Loisel, G.P. et al., 2015, "A higher-than-predicted measurement of iron opacity at solar interior temperatures", Nature, 517, 56–59.
- Barklem, P.S., Piskunov, N., O'Mara, B.J., 2000, "Self-broadening in Balmer line wing formation in stellar atmospheres", A&A, 363, 1091–1105.
- Caffau, E., Bonifacio, P., François, P. et al., 2012, "A primordial star in the heart of the Lion", A&A, 542, A51.
- Casagrande, L., Ramírez, I., Meléndez, J., Bessell, M., Asplund, M., 2010, "An absolutely calibrated T_eff scale from the infrared flux method", A&A, 512, A54.
- Grevesse, N., Sauval, A.J., 1998, "Standard Solar Composition", Space Sci. Rev., 85, 161–174.
- Gustafsson, B., Edvardsson, B., Eriksson, K., Jørgensen, U.G., Nordlund, Å., Plez, B., 2008, "A grid of MARCS model atmospheres for late-type stars", A&A, 486, 951–970.
- Hubeny, I., Mihalas, D., 2014, *Theory of Stellar Atmospheres*, Princeton University Press.
- Keller, S.C. et al., 2014, "A single low-energy, iron-poor supernova as the source of metals in the star SMSS J031300.36-670839.3", Nature, 506, 463–466.
- Kurucz, R.L., Peytremann, E., 1975, *A table of semiempirical gf values*, SAO Special Report.
- Lodders, K., 2003, "Solar System Abundances and Condensation Temperatures of the Elements", ApJ, 591, 1220–1247.
- Majewski, S.R. et al., 2015, "The Apache Point Observatory Galactic Evolution Experiment (APOGEE)", arXiv:1509.05420.
- Meléndez, J., Asplund, M., Gustafsson, B., Yong, D., 2009, "The Peculiar Solar Composition and Its Possible Relation to Planet Formation", ApJL, 704, L66–L70.
- Serenelli, A.M., Basu, S., Ferguson, J.W., Asplund, M., 2009, "New Solar Composition: The Problem with Solar Models Revisited", ApJL, 705, L123.
- Stein, R.F., Nordlund, Å., 1998, "Simulations of Solar Granulation. I. General Properties", ApJ, 499, 914.
- Tremblay, P.-E. et al., 2013, "Spectroscopic analysis of DA white dwarfs with 3D model atmospheres", A&A, 552, A13.
- Valenti, J.A., Piskunov, N., 1996, "Spectroscopy Made Easy: A New Tool for Fitting Observations with Synthetic Spectra", A&AS, 118, 595.
