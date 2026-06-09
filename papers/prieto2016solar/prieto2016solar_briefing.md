---
title: "Pre-Reading Briefing: Solar and Stellar Photospheric Abundances"
paper_id: "48"
topic: Living_Reviews_in_Solar_Physics
date: 2026-04-23
type: briefing
---

# Solar and Stellar Photospheric Abundances: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Carlos Allende Prieto, "Solar and Stellar Photospheric Abundances", Living Reviews in Solar Physics, 2016. DOI: 10.1007/s41116-016-0001-6
**Author(s)**: Carlos Allende Prieto (Instituto de Astrofísica de Canarias)
**Year**: 2016

---

## 1. 핵심 기여 / Core Contribution

This Living Review provides a comprehensive, pedagogical account of how stellar (and in particular solar) photospheric chemical abundances are extracted from spectra. Allende Prieto walks through (1) the physics that underlies stellar-atmosphere and line-formation calculations, (2) the working procedures — gathering spectra, fixing atmospheric parameters (T_eff, log g, [Fe/H]), and deriving abundances — and (3) the explosive growth of multi-object spectroscopic surveys (SDSS/APOGEE, RAVE, LAMOST, Gaia-ESO, GALAH, Gaia) that now deliver spectra for millions of Milky Way stars. The review functions as both a textbook chapter and a state-of-the-field snapshot circa 2016.

이 Living Review는 별(특히 태양)의 광구 화학 조성을 분광 관측으로부터 유도하는 방법을 체계적이고 교육적으로 서술한다. 저자는 (1) 별 대기 및 선 형성 계산을 떠받치는 물리(복사 전달, LTE/NLTE, 불투명도, Boltzmann–Saha 방정식), (2) 스펙트럼 획득 → 대기 매개변수(T_eff, log g, [Fe/H]) 결정 → 조성 도출의 실제 작업 절차, (3) SDSS/APOGEE, RAVE, LAMOST, Gaia-ESO, GALAH, Gaia 등 수백만 개 별의 분광 데이터를 쏟아내는 다천체 분광 서베이의 현황을 통합적으로 정리한다. 2016년 시점의 분야 전체를 조망하는 "교과서 겸 현황 보고서"로 기능한다.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

By 2016, stellar spectroscopy stood at a transition point. The classical theoretical foundations laid by Schwarzschild, Milne, Eddington, and Unsöld in the early 20th century had matured into a mature industry of 1D LTE model atmospheres (ATLAS/Kurucz, MARCS/Uppsala, PHOENIX, Tlusty). Starting around 2000, two seismic shifts collided with this classical picture: (i) 3D radiative-hydrodynamic (RHD) simulations of stellar granulation (Stein-Nordlund, CO5BOLD, Stagger-grid) replaced the hydrostatic 1D + micro/macro-turbulence prescription, and (ii) careful NLTE corrections, combined with better atomic/molecular data, lowered the solar metallicity. Asplund and co-workers (2000, 2005, 2009) reduced the solar C, N, O abundances by 40–50%, producing the "solar abundance problem": helioseismic inversions demanded Z/X ≈ 0.023–0.025 but spectroscopy now gave ≈ 0.018. At the same time, multi-object fiber spectrographs (SDSS, APOGEE, RAVE, LAMOST, Gaia-ESO) turned chemistry into a Galaxy-scale survey science.

2016년 즈음 별 분광학은 거대한 전환기에 있었다. 20세기 초 Schwarzschild, Milne, Eddington, Unsöld가 세운 고전적 이론은 1D LTE 모델 대기(ATLAS/Kurucz, MARCS, PHOENIX, Tlusty)라는 성숙한 산업으로 발전했다. 2000년 무렵부터 두 가지 지진이 이 고전적 그림과 충돌했다: (i) 별 입자운동을 직접 계산하는 3D 복사-유체 시뮬레이션(Stein-Nordlund, CO5BOLD, Stagger-grid)이 1D 정수역학 + 마이크로/매크로 난류 처방을 대체했고, (ii) 정교한 NLTE 보정과 개선된 원자/분자 데이터가 태양의 금속 함량을 낮췄다. Asplund 등(2000, 2005, 2009)이 태양의 C, N, O 함량을 40–50% 줄이자, 태양 내부 헬리오사이스몰로지가 요구하는 Z/X ≈ 0.023–0.025와 분광값 ≈ 0.018 사이에 불일치("태양 조성 문제")가 발생했다. 동시에 SDSS, APOGEE, RAVE, LAMOST, Gaia-ESO 같은 다천체 섬유 분광기가 화학을 은하 규모의 서베이 과학으로 바꿨다.

### 타임라인 / Timeline

```
1814 ─ Fraunhofer: dark lines in solar spectrum / 태양 흡수선 발견
1859 ─ Kirchhoff-Bunsen: spectrum ↔ composition / 분광과 조성의 연결
1929 ─ Russell: Boltzmann excitation in Sun / 태양 내 Boltzmann 여기
1931 ─ McCrea: non-gray model atmospheres / 비-회색 모델 대기
1968 ─ Unsöld & Goldberg-Müller-Aller reviews of solar abundances
1970s ─ Kurucz ATLAS, MARCS (Gustafsson et al. 1975) — 1D LTE standards
1989 ─ Stein & Nordlund: first 3D convection of solar photosphere
2000 ─ Asplund et al.: 3D LTE Fe abundance of the Sun
2005 ─ Asplund: "New Light" review — solar C/N/O reductions
2009 ─ Asplund, Grevesse, Sauval, Scott (AGSS09) composition; helioseismic problem
2011–2015 ─ SDSS-III/APOGEE, RAVE, LAMOST, Gaia-ESO in full operation
2013 ─ Gaia launched (Dec 2013); Stagger-grid / CO5BOLD 3D RHD grids
2015 ─ Bailey et al.: measured Fe interior opacity higher than predicted
2016 ─ THIS REVIEW: state of solar & stellar abundance analysis
```

---

## 3. 필요한 배경 지식 / Prerequisites

- **Stellar atmospheres**: hydrostatic equilibrium, effective temperature T_eff (σT_eff⁴ = emergent flux), surface gravity g = GM/R², optical depth τ. / 정수평형, T_eff, 표면 중력, 광학 깊이의 개념.
- **Radiative transfer**: equation dI_ν/dx = η_ν − κ_ν I_ν, source function S_ν = η_ν/κ_ν, Eddington–Barbier relation I_ν(μ) ≈ S_ν(τ_ν = μ). / 복사 전달 방정식과 Eddington–Barbier 근사.
- **Statistical mechanics**: Boltzmann distribution n_i/n_j ∝ (g_i/g_j) exp(−ΔE/kT), Saha ionization equation. / Boltzmann 분포와 Saha 이온화 방정식.
- **Spectral line basics**: oscillator strength f, Voigt profile (Gaussian × Lorentzian), equivalent width W_λ, curve of growth (linear → saturation → damping). / 진동자 강도, Voigt 프로파일, 등가폭, 성장곡선.
- **Solar abundances notation**: A(X) = log₁₀(N_X/N_H) + 12, [Fe/H] = log(N_Fe/N_H)_* − log(N_Fe/N_H)_⊙. / 태양 함량 표기법.
- **Basic numerical/ML literacy**: χ² fitting, least squares, the flavor of automated pipelines. / 최적화와 자동 파이프라인에 대한 기초 지식.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| Photospheric abundance / 광구 조성 | The elemental composition derived from absorption lines formed in the stellar photosphere (the visible surface layer). / 별의 가시 표면층(광구)에서 형성되는 흡수선으로부터 유도되는 원소 함량. |
| LTE (Local Thermodynamic Equilibrium) / 국소 열평형 | Approximation that level populations follow Boltzmann/Saha at the local gas temperature; source function = Planck function. / 에너지 준위 분포가 국소 기체 온도의 Boltzmann/Saha를 따른다는 근사; 원천 함수 = Planck 함수. |
| NLTE / 비-LTE | Statistical equilibrium with full radiative + collisional rates; needed when photon mean free path > local gradient scale. / 복사 및 충돌 과정의 통계 평형을 풀어야 하며, 광자 평균 자유행로가 국소 구배보다 길 때 필요. |
| Curve of growth / 성장곡선 | log(W_λ/λ) vs log(Nf λ) showing linear → saturated → damping regimes. / log(W_λ/λ) 대 log(Nf λ) 그래프로, 선형 → 포화 → 감쇠(damping) 영역을 보여줌. |
| Equivalent width W_λ / 등가폭 | Width of an equivalent rectangular absorption with the same integrated flux deficit as the line. / 동일한 적분 플럭스 결손을 갖는 직사각형 흡수의 폭. |
| Opacity κ_ν / 불투명도 | Absorption coefficient per unit length at frequency ν; in the solar optical, H⁻ bound-free dominates the continuum. / 주파수 ν에서 단위 길이당 흡수 계수; 태양 광학 영역에서는 H⁻ 속박-자유 흡수가 연속체 불투명도를 지배. |
| H⁻ ion | Negative hydrogen ion responsible for most of the solar continuum opacity in the optical. / 태양 광학 영역 연속체 불투명도의 주된 원천인 음의 수소 이온. |
| Metallicity [Fe/H] / 금속 함량 | log(N_Fe/N_H)_star − log(N_Fe/N_H)_Sun; proxy for overall heavy-element content. / 별과 태양의 Fe/H 비율의 상용로그 차; 전체 중원소 함량의 대리 지표. |
| A(X) notation | A(X) = log(N_X/N_H) + 12; solar values: A(H) ≡ 12, A(Fe) ≈ 7.50, A(O) ≈ 8.69. / A(X) = log(N_X/N_H) + 12; 태양 기준 A(H) ≡ 12, A(Fe) ≈ 7.50, A(O) ≈ 8.69. |
| Grevesse–Sauval scale / GS98 척도 | 1998 standard solar abundance table with A(O) ≈ 8.83, used in early helioseismic models. / 1998년 표준 태양 조성표 (A(O) ≈ 8.83), 초기 헬리오사이스몰로지 모델에서 사용. |
| Asplund reduction / Asplund 감축 | 3D + NLTE analyses that lowered A(O) to ≈ 8.69 (AGSS09), creating the solar abundance problem. / 3D + NLTE 분석을 통해 A(O)를 8.69로 낮추고 태양 조성 문제를 발생시킨 사건. |
| Helioseismic Z/X / 헬리오사이스믹 Z/X | Metal/hydrogen mass-fraction ratio at the surface, constrained by solar oscillations (≈ 0.018 with new abundances, ≈ 0.023 with old). / 태양 진동으로 제약되는 표면 금속/수소 질량비. |
| 3D RHD atmosphere / 3D 복사-유체 대기 | Time-dependent 3D simulation of near-surface convection (Stein-Nordlund, CO5BOLD, Stagger-grid). / 표면 대류를 시간의존 3D로 푸는 시뮬레이션. |
| Equivalent width method / 등가폭법 | Abundance derivation from measured W_λ independent of macro-turbulence/rotation. / 측정된 W_λ로부터 거시난류·회전에 무관하게 조성을 유도하는 방법. |

---

## 5. 수식 미리보기 / Equations Preview

**(1) Radiative transfer equation / 복사 전달 방정식**

$$ \frac{\partial I_\nu}{\partial x} = \eta_\nu - \kappa_\nu I_\nu $$

I_ν는 복사 강도, η_ν는 방출률, κ_ν는 불투명도. Under LTE, source function S_ν ≡ η_ν/κ_ν reduces to the Planck function B_ν(T). / LTE 하에서 원천 함수는 Planck 함수로 환원된다.

**(2) Boltzmann excitation / Boltzmann 분포**

$$ N = \frac{g}{u_j} N_j e^{-E/kT} $$

이온화 상태 j, 축퇴도 g, 분배 함수 u_j, 에너지 E. 주어진 온도에서 특정 준위의 개수 밀도를 결정.

**(3) Saha ionization equation / Saha 이온화 방정식**

$$ \frac{N_j}{\sum_i N_i} = \frac{\gamma^j u_j}{\sum_i \gamma^i u_i} e^{-\beta_j / kT}, \quad \gamma = \frac{2}{n_e h^3} (2\pi m k)^{3/2} T^{3/2} $$

전자 밀도 n_e, 이온화 에너지 χ_i의 함수로 이온화 분율을 준다. / Gives ionization fractions as a function of n_e, T, and χ.

**(4) Metallicity / 금속 함량**

$$ [\mathrm{Fe}/\mathrm{H}] = \log(N_\mathrm{Fe}/N_\mathrm{H})_\star - \log(N_\mathrm{Fe}/N_\mathrm{H})_\odot $$

태양에 대한 로그 비율. / Logarithmic ratio relative to the Sun.

**(5) Line absorption cross-section / 선 흡수 단면적**

$$ \alpha(\nu) = \frac{\pi e^2}{m c} f \phi(\nu) $$

진동자 강도 f와 Voigt 프로파일 φ(ν)의 곱; 국소 선 불투명도는 ℓ_ν = Nα(ν). / Local line opacity ℓ_ν = Nα(ν).

---

## 6. 읽기 가이드 / Reading Guide

- **Section 1 (Introduction)**: Skim for the big-picture motivation — stars as tracers of Galactic chemical evolution, meteoritic vs photospheric abundance agreement (~10% except Sc, Co, Rb, Ag, Hf, W, Pb), and the solar abundance problem (C, N, O lowered 40–50%; helioseismic tension). / 섹션 1은 동기 부여를 위한 큰 그림 — 별은 은하 화학 진화의 추적자, 광구/운석 조성 일치(~10%), 태양 조성 문제.
- **Section 2 (Physics)**: The essential theory block. Read 2.1 (model atmospheres: 1D vs 3D), 2.2 (line formation — this is where all the key equations live: opacity, Boltzmann, Saha, LTE assumption), 2.2.2 (NLTE — the full statistical equilibrium problem, collisional vs radiative rates, H-collisions as the main uncertainty in late-type stars). / 섹션 2는 이론의 핵심; 수식은 여기에 몰려 있다.
- **Section 3 (Working Procedures)**: The "how to do it" section. 3.1 data, 3.2 atmospheric parameters (excitation balance, Balmer wings for T_eff, Saha balance for log g), 3.3 abundance derivation (equivalent widths vs full synthesis), 3.4 tools (MOOG, SME, FERRE, ULySS, MATISSE). / 섹션 3은 실무 편람.
- **Section 4 (Observations)**: Survey census — SDSS/APOGEE (100k H-band R=22500 spectra of ~half a million stars), RAVE, LAMOST (4000 fibers), Gaia-ESO, GALAH. Useful as reference but skimmable. / 대규모 서베이 현황 요약; 참고용.
- **Section 5 (Reflections)**: Author's outlook — Gaia + spectroscopy will transform Galactic archaeology over the next decade. / 저자의 전망.
- **Tip**: The paper is long on surveys but short on quantitative abundance tables; supplement with Asplund et al. (2009) for the AGSS solar composition numbers. / 팁: 이 논문 자체에는 수치 표가 적으니 AGSS09을 함께 보라.

---

## 7. 현대적 의의 / Modern Significance

Photospheric abundances remain the *reference frame* for Galactic chemical evolution: every [α/Fe] vs [Fe/H] diagram that separates thin/thick disk stars, every chemical-tagging claim, every exoplanet-host-star refractory-element analysis ultimately traces back to the procedures reviewed here. Since 2016, Gaia DR2/DR3 (2018, 2022) delivered the predicted parallaxes for billions of stars, DESI started operations (2021), 4MOST and WEAVE came online (2023–2024), and the Bailey-et-al. (2015) opacity measurement has indeed been developed as a partial resolution to the solar abundance problem — but the tension is not fully closed. Machine-learning abundance pipelines (The Cannon, The Payne, data-driven transfer between spectra) now complement FERRE/SME-style physical fitting. This review remains the canonical entry point for anyone learning how a number like A(Fe) = 7.50 comes out of a stellar spectrum.

광구 조성은 은하 화학 진화의 *기준 좌표계*이다. 얇은/두꺼운 원반을 분리하는 모든 [α/Fe]–[Fe/H] 도표, 화학 태깅, 외계 행성 모체 별의 내화성 원소 분석은 결국 이 리뷰에서 정리된 절차로 회귀한다. 2016년 이후 Gaia DR2/DR3가 수십억 별의 시차를 제공했고, DESI(2021), 4MOST·WEAVE(2023–2024)가 가동되었으며, Bailey 등(2015)의 Fe 불투명도 측정은 태양 조성 문제의 부분적 해결책으로 발전했다(완전히 해소되지는 않음). 기계학습 기반 파이프라인(The Cannon, The Payne 등)도 FERRE/SME 같은 물리 피팅을 보완한다. 이 리뷰는 "별 분광에서 A(Fe) = 7.50 같은 숫자가 어떻게 나오는가?"를 이해하려는 모든 사람의 표준 진입점으로 남아 있다.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
