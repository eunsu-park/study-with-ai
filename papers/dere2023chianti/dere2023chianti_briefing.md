---
title: "Pre-Reading Briefing: CHIANTI XVII — Version 10.1: Revised Ionization and Recombination Rates"
paper_id: "64_dere_2023"
topic: Solar_Observation
date: 2026-04-28
type: briefing
---

# CHIANTI XVII — Version 10.1: Revised Ionization and Recombination Rates and Other Updates: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Dere, K. P., Del Zanna, G., Young, P. R., Landi, E., "CHIANTI — An Atomic Database for Emission Lines. XVII. Version 10.1: Revised Ionization and Recombination Rates and Other Updates," *The Astrophysical Journal Supplement Series*, **268**, 52 (17 pp), 2023. DOI: [10.3847/1538-4365/acec79](https://doi.org/10.3847/1538-4365/acec79)
**Author(s)**: Kenneth P. Dere (George Mason Univ.), Giulio Del Zanna (Cambridge DAMTP), Peter R. Young (NASA GSFC / Northumbria), Enrico Landi (Univ. of Michigan)
**Year**: 2023 (Received May 24; Accepted Jul 31; Published Sep 28)

---

## 1. 핵심 기여 / Core Contribution

### 한국어
이 논문은 천체 플라스마의 광학적 얇은(optically thin) 발광 스펙트럼을 모델링하는 표준 도구인 **CHIANTI 원자 데이터베이스**의 **버전 10.1** 릴리스 노트(release note)에 해당한다. CHIANTI 10(Del Zanna et al. 2021) 이후 누적된 새로운 실험·이론 결과를 반영하여 (i) **13개 이온의 이온화 단면적·속도 계수**를 새 측정치(Hahn, Bernhardt, Fogle 등의 저장 링/충돌 빔 실험)에 맞춰 재적합하고, (ii) **인(P) 등전자 수열 전체에 대한 새로운 복사 재결합(RR)·이중전자 재결합(DR) 속도**(Bleda et al. 2022)를 통합하며, (iii) 이러한 갱신된 속도들로부터 **새로운 이온화 평형(ionization equilibrium) 파일**을 재계산한다. 동시에 (iv) **N·O 등전자 수열 8개 이온**에 대해 R-matrix ICFT 계산 기반의 새 전자 충돌·복사 데이터셋을 추가하고, (v) 6개 이온의 에너지 준위·파장을 갱신했으며, (vi) Asplund et al. 2021 광구 원소 함량을 반영한 새 abundance 파일과, FIP bias 10^0.5을 적용한 새 코로나 abundance 파일을 제공한다. 본 논문은 새로운 물리 결과를 제시하는 것이 아니라, **태양·천체 분광 진단의 분모(denominator)에 해당하는 원자 데이터의 정밀도 향상**을 정량적으로 문서화한다는 점에서 의의가 있다.

### English
This paper is the **release note** for **version 10.1** of the **CHIANTI atomic database**, the de-facto standard for modelling optically thin emission from astrophysical plasmas. Since CHIANTI 10 (Del Zanna et al. 2021), the team accumulated (i) new laboratory ionization cross-section measurements (Hahn, Bernhardt, Fogle and collaborators using heavy-ion storage rings and crossed-beam experiments) for 13 commonly observed ions, refit using the Burgess–Tully ionization (BTI) scaling, (ii) **complete radiative- and dielectronic-recombination rates for the phosphorus iso-electronic sequence** from Bleda et al. (2022), and (iii) **a recomputed default ionization-equilibrium file** built from the revised rates. In parallel, the release adds (iv) new electron-collision and radiative datasets — based on R-matrix ICFT calculations of Mao et al. (2020, 2021) — for **8 ions in the N- and O-like sequences**, (v) updated energy levels and wavelengths for 6 other ions, and (vi) two new abundance files: a photospheric file based on Asplund et al. (2021) and a representative coronal file derived from it by applying a uniform FIP-bias factor of 10^0.5 to low-FIP elements. Rather than presenting new physics, the paper rigorously documents the *denominator* of nearly every spectroscopic diagnostic — the atomic data — and quantifies how much it has changed since v10.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

#### 한국어
CHIANTI는 1996년 Dere, Landi, Mason, Monsignori-Fossi, Young이 SOHO 시대의 EUV 분광기(CDS, SUMER)를 지원하기 위해 시작한 오픈 소스 프로젝트로, 1997년 첫 릴리스(논문 #63) 이후 약 25년간 **17편의 논문**을 통해 점진적으로 갱신되어 왔다. 2007년 Dere가 H부터 Zn까지 모든 이온의 이온화 단면적을 종합한 작업이 v6의 토대가 되었고, Bleda et al. (2003)이 시작한 등전자 수열 단위 RR/DR 계산 프로그램이 v7~v10에 걸쳐 H 수열부터 Si 수열까지 차례로 통합되었다. v10(Del Zanna et al. 2021)은 모든 Fe 이온의 R-matrix 계산을 도입한 큰 도약이었지만, 그 이후로도 (a) Hahn et al.의 TSR/CRYRING 저장 링 측정, (b) Bleda et al. (2022)의 P 수열 RR/DR 완성, (c) Mao et al.의 N·O 수열 R-matrix 결과, (d) Asplund et al. (2021)의 새 광구 함량 등이 누적되었다. v10.1은 이 누적분을 통합한 **점진적 마이너 릴리스(minor release)**다.

#### English
CHIANTI was launched in 1996 by Dere, Landi, Mason, Monsignori-Fossi, and Young to support EUV spectrometers of the SOHO era (CDS, SUMER); since the first release paper in 1997 (paper #63 in this study), the project has produced **17 release/update papers** over roughly 25 years. Dere (2007) compiled ionization cross sections for every ion from H to Zn, forming the basis of v6, while the iso-electronic recombination programme initiated by Badnell et al. (2003) and extended by Bleda and colleagues progressively delivered RR/DR rates for the H-like through Si-like sequences across v7–v10. CHIANTI 10 (Del Zanna et al. 2021) was a major step forward — introducing R-matrix calculations for all Fe ions — but new laboratory measurements (Hahn et al. at the Heidelberg TSR storage ring; Bernhardt et al.; Fogle et al. at Oak Ridge) and theoretical work continued to accumulate. v10.1 is the **incremental minor release** that folds in (a) those storage-ring ionization data, (b) Bleda et al. (2022)'s completion of the P-sequence RR/DR rates, (c) Mao et al.'s R-matrix results for N- and O-like ions, and (d) the new Asplund et al. (2021) photospheric abundance compilation.

### 타임라인 / Timeline

```
1992 ─── Burgess & Tully scaling for collision strengths (BT scaling)
1996 ─── CHIANTI project launched (SOHO support)
1997 ─── CHIANTI v1.0 (Dere et al., paper #63 in this study)
1998 ─── Mazzotta et al. ionization equilibrium (long-time baseline)
2003 ─── Badnell et al. systematic DR programme begins
2007 ─── Dere ionization cross sections H–Zn (basis of v6)
2009 ─── CHIANTI v6 (Dere et al.)
2010-16 ── Hahn, Bernhardt, Fogle storage-ring measurements accumulate
2016 ─── CHIANTI v8 (Del Zanna et al.); Young et al. modern overview
2020 ─── Mao et al. R-matrix ICFT for N-like ions (O II – Zn XXIV)
2021 ─── CHIANTI v10 (Del Zanna et al.) — major Fe R-matrix update
         Mao et al. R-matrix ICFT for O-like ions (Ne III – Zn XXIII)
         Asplund et al. new photospheric abundances
2022 ─── Bleda et al. RR/DR for entire phosphorus sequence
2023 ─── CHIANTI v10.1 (this paper, #64) — incremental update
2025 ─── Abbo et al. (#61) uses CHIANTI v10.1 for R(T_e) curves
```

---

## 3. 필요한 배경 지식 / Prerequisites

### 한국어

**원자 물리학 (Atomic physics)**
- **이온화 단면적 σ(E)**: 단위 에너지 E의 자유 전자가 이온을 한 단계 더 이온화시킬 미시적 확률 단면적. 단위 cm². 이온화 퍼텐셜 IP 이상에서 0이 아니다.
- **이온화 속도 계수 R(T)**: σ(E)를 Maxwell–Boltzmann 분포로 적분해서 얻는 거시적 속도 계수 [cm³ s⁻¹]. 코로나 이온화 평형의 핵심 입력.
- **직접 이온화(DI)와 자동이온화 여기(EA)**: 두 가지 이온화 채널. DI는 한 번에 외각 전자를 떼어내고, EA는 내각 전자를 들뜬 상태로 여기시킨 뒤 자동이온화로 전자를 잃는다. EA는 IP보다 훨씬 높은 임계값(threshold)을 가지며 단면적에 계단형(step-like) 구조를 만든다.
- **복사 재결합(RR)과 이중전자 재결합(DR)**: 자유 전자의 이온 포획 과정. RR은 광자를 방출, DR은 자동이온화 준위를 거쳐 안정화된다. 저온에서는 ΔN=0 코어 전자 여기형 DR이 지배적.

**플라스마 분광 (Plasma spectroscopy)**
- **충돌 평형(coronal equilibrium)**: 전자 충돌 이온화와 전자 충돌 재결합이 균형을 이루는 정상 상태. 태양 코로나의 중간/상부 전이 영역에서 표준 가정.
- **이온화 분율(ionization fraction)** $f_{Z,z}(T_e)$: 원소 Z의 전체 이온 중 z번 이온화된 것의 분율. T_e에 대해 종 모양(bell-shaped) 곡선을 그린다.
- **방출 함수(contribution function) G(T_e, n_e)**: 한 스펙트럼선의 단위 부피·단위 방출 측도(Emission Measure)당 광도. CHIANTI가 출력하는 핵심 양.

**수치/방법론 (Numerical / methodological)**
- **Burgess–Tully 스케일링(BTI scaling)**: 단면적·충돌 강도를 [0,1] 구간의 무차원 변수로 압축하여 부드러운 스플라인 적합을 가능하게 하는 변환 (식 1, 2).
- **Flexible Atomic Code (FAC)**: Gu (2002)의 상대론적 원자 구조·충돌 계산 코드. 측정이 없는 이온의 단면적 추정에 사용.
- **R-matrix 방법**: 전자-이온 산란을 표적 이온 내부 영역과 외부 영역으로 분리하여 풀이하는 양자역학적 산란 이론. 공명 구조까지 포함.
- **AUTOSTRUCTURE / GRASP2K / Breit–Pauli**: 원자 구조 코드들. 이번 릴리스에서 일부 이온의 A 값은 Breit–Pauli 또는 MCDHF 결과로 교체됨.

**선행 지식 / 논문**
- **#63 Dere et al. 1997**: CHIANTI v1.0 — 데이터베이스 구조와 G(T_e, n_e) 정의의 원형
- **#61 Abbo et al. 2025**: v10.1을 사용하여 O VI/Ne VIII 도플러 변광 R(T_e) 진단을 수행 → 본 논문의 직접적 사용 사례
- **#65 Mason & Monsignori-Fossi 1994**: 코로나 분광 진단의 일반 리뷰

### English

**Atomic physics**
- **Ionization cross section σ(E)**: microscopic probability area for a free electron of energy E to ionize an ion by one stage. Units cm²; non-zero only above the ionization potential (IP).
- **Ionization rate coefficient R(T)**: macroscopic rate [cm³ s⁻¹] obtained by integrating σ(E) over a Maxwell–Boltzmann distribution. The principal input to coronal ionization equilibrium.
- **Direct ionization (DI) vs. excitation-autoionization (EA)**: two channels. DI removes an outer electron; EA excites an inner electron to an autoionizing level which then sheds an electron. EA thresholds sit well above the IP and produce step-like features in σ(E).
- **Radiative (RR) and dielectronic (DR) recombination**: the inverse processes. RR emits a photon; DR captures the electron into an autoionizing level that is stabilized radiatively. At low T, ΔN = 0 core excitations dominate DR.

**Plasma spectroscopy**
- **Coronal (collisional) equilibrium**: steady state in which electron-impact ionization balances electron + radiative recombination — the textbook assumption for the mid/upper solar atmosphere.
- **Ionization fraction** $f_{Z,z}(T_e)$: fraction of element Z in ionization stage z; bell-shaped in T_e.
- **Contribution function G(T_e, n_e)**: emissivity per unit emission measure of a single line — the CHIANTI output that connects atomic physics to observed line intensities.

**Numerical / methodological**
- **Burgess–Tully ionization (BTI) scaling**: maps E and σ to dimensionless variables on [0,1] so smooth splines can be fit (Eqs. 1, 2 of the paper).
- **Flexible Atomic Code (FAC)**: Gu (2002) relativistic atomic-structure / collision code, used to estimate cross sections for ions lacking measurements.
- **R-matrix method**: quantum scattering technique that splits electron–ion scattering into an internal (target) region and an external region, capturing resonance structure.
- **AUTOSTRUCTURE / GRASP2K / Breit–Pauli**: atomic-structure codes; v10.1 swaps several A-values to Breit–Pauli or MCDHF results.

**Prior reading**
- **#63 Dere et al. 1997**: CHIANTI v1.0 — database architecture and the original G(T_e, n_e) definition.
- **#61 Abbo et al. 2025**: uses v10.1 for the O VI / Ne VIII Doppler-dimming R(T_e) diagnostic — the direct downstream consumer of this release.
- **#65 Mason & Monsignori-Fossi 1994**: general review of coronal spectroscopic diagnostics.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **Ionization fraction / 이온화 분율** | $f_{Z,z}(T_e)$: 원소 Z의 전체 핵 중 전하 상태 z에 있는 분율. 코로나 평형에서 T_e만의 함수. — Fraction of an element's nuclei in charge state z; in coronal equilibrium it is a function of T_e only. |
| **Ionization equilibrium / 이온화 평형** | 모든 전하 상태에 대해 이온화 입속(in-flow)과 재결합 출속(out-flow)이 평형을 이루는 정상 상태. CHIANTI는 이를 별도 파일로 배포. — Steady-state condition in which ionization in-flow balances recombination out-flow for every stage; CHIANTI ships it as a precomputed file. |
| **Iso-electronic sequence / 등전자 수열** | 동일한 전자 수를 가지는 이온들의 집합 (예: P-like = 15 electrons → P I, S II, Cl III, …). 원자 구조 계산을 수열 단위로 묶을 수 있음. — Set of ions sharing the same electron count; allows structure calculations to proceed sequence-by-sequence. |
| **Radiative recombination (RR) / 복사 재결합** | $X^{z+} + e^{-} \to X^{(z-1)+} + h\nu$. 자유–속박 광자 방출. 모든 T에서 작동. — Free–bound recombination with photon emission; operates at all T. |
| **Dielectronic recombination (DR) / 이중전자 재결합** | 자유 전자가 코어 전자를 들뜨게 하면서 자동이온화 준위로 포획되고, 후속 복사 전이로 안정화되는 2단계 과정. 일반적으로 코로나 온도에서 RR보다 큼. — Two-step capture into an autoionizing level (with simultaneous core excitation) followed by radiative stabilization; usually dominates over RR at coronal temperatures. |
| **Direct ionization (DI) / 직접 이온화** | 충돌 전자가 표적 전자를 직접 떼어내는 1단계 이온화. — Single-step removal of a target electron by the impact electron. |
| **Excitation–autoionization (EA) / 들뜸–자동이온화** | 내각 전자가 들뜬 자동이온화 준위로 여기된 뒤 전자를 방출하는 2단계 이온화. 단면적에 계단형 구조 생성. — Two-step ionization via an autoionizing intermediate; produces step-like features in σ(E). |
| **Distorted wave (DW) calculation / 왜곡파 계산** | 입사·산란 전자를 표적 퍼텐셜 안에서 왜곡된 평면파로 근사하는 충돌 계산법. 빠르지만 공명 효과를 놓침. — Collision approximation that distorts the projectile wave by the target potential; fast but misses resonances. |
| **R-matrix method / R-matrix 방법** | 표적 내부와 외부 영역을 분리하여 풀어 공명까지 포함하는 정밀 산란 계산법. ICFT는 그 중간결합(intermediate coupling) 변형. — Accurate scattering technique that splits internal/external regions and captures resonances; ICFT is its intermediate-coupling frame transformation variant. |
| **Autoionization / 자동이온화** | 결합 에너지를 가지는 들뜬 상태가 광자 방출 없이 자발적으로 전자를 방출하는 과정. EA·DR의 핵심 단계. — Spontaneous electron ejection from a bound excited state above an ionization threshold; central to both EA and DR. |
| **Burgess–Tully scaling / BT 스케일링** | 충돌 강도를 [0,1] 구간 무차원 변수로 압축하여 스플라인 적합과 보간을 용이하게 함. — Maps collision strengths to a finite interval enabling smooth spline interpolation. |
| **Maxwellian rate coefficient / 맥스웰 속도계수** | $R(T)=\int v\,\sigma(E)\,f(v,T)\,dv$. 단면적을 열적 분포로 평균. — Thermal average of $v\sigma$ over a Maxwell–Boltzmann distribution. |
| **Version-controlled atomic database / 버전 관리 원자 데이터베이스** | 모든 출력이 명시적 버전 태그(예: v10.1)와 결합되어, 후속 논문이 정확한 데이터셋을 인용·재현할 수 있게 하는 운영 모델. — Operational model in which every output carries an explicit version tag (e.g. v10.1) so downstream papers can cite a reproducible dataset. |
| **FIP bias / FIP 편향** | 첫 이온화 퍼텐셜이 낮은 (≤10 eV) 원소가 코로나에서 광구 대비 ~3–4배 풍부해지는 현상. v10.1 코로나 abundance 파일은 일률적으로 10^0.5 인수를 적용. — Enhancement of low-FIP elements (≤10 eV) in the corona relative to the photosphere by factors of 3–4; the new coronal abundance file applies a uniform 10^0.5 boost. |

---

## 5. 수식 미리보기 / Equations Preview

### 5.1 BTI scaled energy and cross section / BTI 스케일된 에너지·단면적 (식 1, 2)

$$
U \;=\; 1 \;-\; \frac{\ln f}{\ln(u - 1 + f)}, \qquad
\Sigma \;=\; \frac{u\,\sigma\,I^{2}}{\ln u + 1}
$$

- $u = E/I$: 이온화 퍼텐셜 단위로 환산한 입사 전자 에너지 / incident electron energy in units of the ionization potential.
- $f$: 데이터 평가자가 곡선의 모양을 가장 잘 펴기 위해 선택하는 스케일 파라미터 / data-assessor-chosen scaling parameter that flattens the curve for spline fitting.
- 이 변환은 임계값(threshold) 근처와 무한 에너지 한계 사이의 단면적을 [0,1] 구간에 매핑하여 **9개 이상의 스플라인 노드**로 정확한 재현을 가능하게 한다. v10.1에서 13개 이온의 DI 단면적이 이 형태로 새로 적합되었다. / This transform compresses σ(E) onto [0,1] so that **≥9 spline nodes** suffice for an accurate fit. v10.1 refits DI cross sections for 13 ions in this form.

### 5.2 Maxwellian ionization rate coefficient / 맥스웰 이온화 속도 계수 (식 3)

$$
R(T) \;=\; \int_{v_{IP}}^{\infty} v\,\sigma(E)\,f(v,T)\,dv,
\qquad E = \tfrac{1}{2} m v^{2}
$$

- 적분 하한 $v_{IP}$: 운동에너지가 IP와 같은 전자 속도. 그 아래에서는 이온화 불가능 / cutoff at the velocity whose kinetic energy equals the IP.
- $f(v, T)$: 전자 온도 T에서의 Maxwell–Boltzmann 속도 분포 / Maxwell–Boltzmann velocity distribution at electron temperature T.
- **이 식이 곧 #61 Abbo et al. 2025의 R(T_e) 곡선을 만드는 첫 번째 입력**이다. v10.1의 Table 1은 13개 이온에 대한 신·구 R(T_max) 비를 0.56–1.17 범위로 보고한다. / **This integral feeds the R(T_e) curves used in #61 Abbo et al. 2025.** Table 1 of v10.1 quotes new/old rate ratios at T_max ranging from 0.56 (Fe VIII) to 1.17 (Fe XII).

### 5.3 Total recombination rate / 총 재결합 속도

$$
\alpha_{\text{tot}}(T_e) \;=\; \alpha_{\text{RR}}(T_e) \;+\; \alpha_{\text{DR}}(T_e)
$$

- v10.1은 **인(P) 등전자 수열 전체**(Z = 16부터 30+)에 대해 Bleda et al. (2022)의 새 RR + DR 적합 매개변수를 도입한다. / v10.1 ingests Bleda et al. (2022)'s RR + DR fit parameters for the **entire P iso-electronic sequence** (Z = 16 to 30+).
- 저온에서 DR이 ΔN = 0 코어 여기로 지배되며, 이전(Mazzotta et al. 1998)의 LS-coupling 코드들은 이 영역을 과소평가했음을 본 논문이 지적한다. / At low T, DR is dominated by ΔN = 0 core excitations; older LS-coupling tabulations (Mazzotta et al. 1998; Shull & van Steenberg 1982) under-estimated this regime.

### 5.4 Coronal ionization equilibrium / 코로나 이온화 평형

각 전하 상태 z의 이온화 분율은 정상 상태 평형식에서 결정된다 / each charge-state fraction $f_{Z,z}$ is fixed by the steady-state balance:

$$
n_{Z,z-1}\,R^{\text{ion}}_{z-1\to z} \;=\; n_{Z,z}\,\alpha^{\text{rec}}_{z\to z-1}
$$

- 모든 $z$에 대해 동시에 풀면 $f_{Z,z}(T_e) = n_{Z,z}/n_Z$를 얻는다 / Solved for all z gives $f_{Z,z}(T_e)$.
- v10.1의 Section 2.16과 Figure 18은 Fe X–XIV에 대해 새 곡선이 v10 곡선 대비 어떻게 이동했는지를 보여준다. **11개 이온은 log T_max가 0.05 dex 이동**하고, **7개 이온(V VIII, IX; Sc VI, VII; Ca VI; K V)은 피크 분율이 10% 이상 변한다**. / Section 2.16 and Fig. 18 show the shifts for Fe X–XIV; 11 ions have log T_max shifted by 0.05 dex, and 7 ions (V VIII, IX; Sc VI, VII; Ca VI; K V) have peak fractions changed by >10%.

### 5.5 Contribution function from CHIANTI components / CHIANTI 성분으로 구성한 방출 함수

$$
G_{ji}(T_e, n_e) \;=\; \frac{A_{ji}\,n_j(T_e, n_e)}{n_{Z,z}(T_e, n_e)}\,\frac{n_{Z,z}(T_e)}{n_Z}\,\frac{n_Z}{n_H}\,\frac{1}{n_e}
$$

- $A_{ji}$: 상위 준위 j → 하위 준위 i의 자발 방출(Einstein) 계수 / spontaneous emission coefficient.
- $n_j/n_{Z,z}$: 전자 충돌 들뜸·자발 방출로 결정되는 **상위 준위 인구분포** (CHIANTI의 충돌 강도 파일) / level population from collisional excitation + spontaneous decay (uses CHIANTI collision-strength files).
- $n_{Z,z}/n_Z = f_{Z,z}(T_e)$: **이온화 분율** (이번 릴리스에서 갱신) / ionization fraction (updated in this release).
- $n_Z/n_H$: **원소 함량비** (이번 릴리스의 새 abundance 파일) / elemental abundance (new abundance files in this release).
- 이 4성분 곱이 곧 #61이 사용하는 R(T_e) ≡ G_{O VI}(T_e)/G_{Ne VIII}(T_e) 비의 분자·분모를 구성한다. **v10.1 → v10 릴리스 사이의 변화는 이 곱의 모든 인수에 영향을 준다**. / The product of these four factors is what enters R(T_e) ≡ G_{O VI}/G_{Ne VIII} as used by #61. **The v10 → v10.1 update modifies every factor in the product.**

---

## 6. 읽기 가이드 / Reading Guide

### 한국어

이 논문은 **릴리스 노트(release note) 형식**이다. 새 물리 결과를 발표하는 논문이 아니라, 데이터베이스의 변경 사항을 항목별로 정리한 문서다. 따라서 처음부터 끝까지 같은 깊이로 읽을 필요는 없다. 다음 우선순위로 읽기를 권장한다.

**우선순위 1 — 반드시 정독 (must-read)**
- **§1 Introduction (p. 1)** — CHIANTI 프로젝트 전체의 역사와 v10.1의 위치. Dere 1997, Del Zanna 2021 등 선행 릴리스를 명시적으로 인용한다.
- **§2.1 Approach (p. 2)** — Dere (2007) 이래 사용된 BTI 스케일링과 단면적 적합 절차. 식 (1), (2)의 정의가 후속 절들의 모든 그림(Figs. 3–15)을 이해하는 열쇠.
- **§2.14 Ionization Rate Coefficients (p. 11)** — 식 (3)의 R(T) 정의와 Fig. 16, 17의 해석. **#61의 R(T_e) 곡선이 직접 의존하는 부분.** Table 1의 13개 이온 신/구 비율이 핵심.
- **§2.16 A Revised Ionization Balance (p. 12)** — 새 이온화 평형 파일에 대한 정량적 변화. Fig. 18(Fe X–XIV)을 보고 #61에서 사용된 종(O VI, Ne VIII)이 영향권에 있는지 확인할 것.
- **§4 Elemental Abundances (p. 15)** — Asplund et al. (2021) 광구 함량 파일과 새 코로나 함량 파일. **Ne 함량이 +35% 증가**한 것이 핵심 — Ne VIII 진단을 사용하는 #61에 직접적 함의.

**우선순위 2 — 중요 (skim with care)**
- **§2.15 Recombination Rate Coefficients for the P Sequence (p. 11)** — Bleda et al. (2022)의 RR/DR 적합. v10.1에서 가장 큰 단일 이론 갱신.
- **§3.4–3.5 Nitrogen / Oxygen Iso-electronic Sequences (pp. 13–14)** — Mao et al. (2020, 2021)의 R-matrix ICFT 결과로 교체된 8개 이온. **O VI는 N 수열이 아니지만, O 수열 이온들은 #61의 분광 환경에서 등장**.
- **§5 Conclusions (p. 16)** — 갱신 사항 요약 체크리스트.

**우선순위 3 — 가볍게 훑기 (skim only)**
- **§2.2–2.13** — 이온별 단면적 적합 세부 사항. 13개 이온 각각에 대한 그림(Figs. 3–15)이 비슷한 형태로 반복된다. 자신이 다루는 이온이 목록에 있다면 그 절만 정독하고, 나머지는 그림의 정성적 변화만 확인.
- **§3.1–3.3, 3.6–3.8** — 6개 이온에 대한 에너지 준위·파장 갱신. 특정 진단선을 사용 중이 아니라면 깊이 들어가지 않아도 됨.

**읽는 동안 손에 들고 있을 질문 / Questions to keep in hand**
1. **#61의 R(T_e) 곡선에 직접 영향을 주는 이온(O VI, Ne VIII)이 13개 갱신 이온 목록에 있는가?** (Table 1을 확인)
2. **Ne 함량이 +35% 변한 것이 R(T_e) 비의 분모(Ne VIII)를 어느 방향으로 이동시키는가?**
3. **Fig. 18의 이온화 평형 변화 폭(0.05 dex)이 #61의 도플러 변광 진단의 형성 온도 가정에 의미 있는 수준인가?**
4. **버전 인용의 중요성**: v10인지 v10.1인지에 따라 R(T_e) 분석 결과가 달라질 수 있다. 후속 논문은 반드시 정확한 버전을 명시해야 함을 본 논문이 보여줌.

### English

This paper is in the **release-note genre** — it does not present new physics, but documents what changed in the database. Read it stratified by priority:

**Priority 1 — must-read**
- **§1 Introduction (p. 1)**: CHIANTI's history and where v10.1 sits. Names every prior release paper, including Dere et al. 1997 and Del Zanna et al. 2021.
- **§2.1 Approach (p. 2)**: the BTI scaling and cross-section fitting protocol inherited from Dere (2007). Eqs. (1) and (2) — the key to interpreting every cross-section figure that follows.
- **§2.14 Ionization Rate Coefficients (p. 11)**: definition of R(T) (Eq. 3) and reading Figs. 16–17. **This is the section that #61's R(T_e) curves depend on directly.** Table 1 quantifies the new/old ratios for the 13 updated ions.
- **§2.16 A Revised Ionization Balance (p. 12)**: how the new equilibrium file changed. Inspect Fig. 18 (Fe X–XIV) and check whether the species used in #61 (O VI, Ne VIII) sit in the affected list.
- **§4 Elemental Abundances (p. 15)**: the Asplund et al. (2021) photospheric file and the new coronal file. **The Ne abundance is +35% higher** — a direct implication for any Ne VIII–based diagnostic.

**Priority 2 — important, skim carefully**
- **§2.15 P-sequence RR/DR (p. 11)**: the single largest theoretical addition.
- **§3.4–3.5 N- and O-like sequences (pp. 13–14)**: 8 ions replaced with Mao et al. R-matrix data. O VI itself is Li-like, but O-sequence ions surface in #61's spectroscopic context.
- **§5 Conclusions (p. 16)**: summary checklist.

**Priority 3 — light skim**
- **§§2.2–2.13**: ion-by-ion cross-section refits. Figs. 3–15 follow the same template; read the section only for ions you actively work with.
- **§§3.1–3.3, 3.6–3.8**: energy-level / wavelength updates for 6 ions; relevant only if you use those particular diagnostic lines.

**Questions to keep in hand while reading**
1. **Are the ions that drive #61's R(T_e) (O VI, Ne VIII) on the list of 13 updated ions?** (Check Table 1.)
2. **In which direction does the +35% Ne abundance change push the R(T_e) ratio's denominator (Ne VIII)?**
3. **Are the 0.05-dex shifts in ionization-balance T_max (Fig. 18) significant for the formation-temperature assumptions in the #61 Doppler-dimming diagnostic?**
4. **Version citation matters**: results can change between v10 and v10.1; this paper itself is the proof that downstream work must cite the exact version used.

---

## 7. 현대적 의의 / Modern Significance

### 한국어

#### 7.1 #63 Dere et al. 1997 → #64로 이어지는 25년의 계보 / 25-year lineage from #63 to #64

CHIANTI v1.0은 SOHO/CDS, SUMER 분광기 분석을 지원하기 위해 1997년에 공개되었다. 그 안에 정의된 데이터베이스 구조(에너지 준위 파일 + 파장·gf·A값 파일 + 충돌 강도 파일)와 G(T_e, n_e) 형식주의는 그대로 v10.1에 살아있다. 본 논문은 같은 저자(K. P. Dere)가 25년 후 같은 데이터베이스를 갱신한 결과로, **천체 물리 인프라의 장기 유지 관리(long-term maintenance)가 어떻게 작동하는지를 보여주는 모범 사례**다. 단면적 한 이온, A 값 하나의 갱신이 누적되어 매 5년 단위로 메이저 릴리스(v8, v9, v10)가 나오고, 사이사이에 이러한 마이너 릴리스(v10.1)가 끼어든다.

#### 7.2 #61 Abbo et al. 2025와의 직접 연결 / Direct link to #61 Abbo et al. 2025

#61은 Metis/SoHO와 UVCS 도플러 변광 관측을 비교하기 위해 **CHIANTI v10.1로 계산한 R(T_e) ≡ G_{O VI}(T_e)/G_{Ne VIII}(T_e) 비**를 사용한다. 본 논문(#64)은 그 R(T_e) 곡선의 4가지 입력 — (a) 충돌 강도(Mao et al. 2021의 O 수열 R-matrix), (b) 이온화·재결합 속도(Table 1의 13개 이온), (c) 이온화 평형(Fig. 18), (d) Ne 함량(+35%) — 모두에 손을 댄 릴리스다. 따라서 #61의 R(T_e) 분석을 비판적으로 평가하려면 #64를 반드시 알아야 하며, **#61이 v10이 아닌 v10.1을 명시적으로 인용한 것은 이 차이를 인지하고 있었음을 의미**한다.

#### 7.3 태양 분광 진단의 정확도 / Accuracy of solar spectroscopic diagnostics

EUV·X-ray 분광 관측에서 도출되는 거의 모든 양 — 형성 온도(formation temperature), 미분 방출 측도(DEM), 원소 함량, 비열적 속도(non-thermal velocity) — 은 G(T_e, n_e)의 정확도에 의존한다. v10.1은 일부 이온에서 **이온화 속도 계수가 ~10% 이상 변하고**(Table 1), Ne의 광구 함량이 ~35% 변하는 갱신이다. 이는 다음 세대의 분광기 — Solar Orbiter/SPICE, EUVST, MUSE — 의 데이터 분석에 직접적인 영향을 미친다. **버전을 명시적으로 인용하고, 데이터베이스가 새 버전으로 갱신될 때마다 핵심 결과를 재계산하는 관행**이 정착될 필요가 있음을 본 논문이 강조한다.

#### 7.4 오픈 소스 천체 물리 인프라의 모범 / Exemplar of open-source astrophysics infrastructure

CHIANTI는 25년 넘게 CC-BY 라이선스 하에 데이터·소프트웨어를 모두 공개해 왔다. 본 논문이 보여주듯, 새 측정·새 이론 결과가 학계에서 발표되면 그것이 곧 데이터베이스에 통합되고, 통합 사실 자체가 ApJS 같은 동료심사 학술지에 명시적으로 보고된다. 이는 **재현 가능한 천체 물리(reproducible astrophysics)**의 작동 원리를 정확히 구현한다.

### English

#### 7.1 25-year lineage from #63 to #64
CHIANTI v1.0 was released in 1997 to support SOHO/CDS and SUMER analyses. Its database architecture (energy-level files + wavelength / gf / A-value files + collision-strength files) and G(T_e, n_e) formalism survive unchanged into v10.1. This paper, with the same lead author (K. P. Dere) updating the same database 25 years later, is **a textbook example of long-term maintenance of astrophysical infrastructure**. Updates accumulate one cross section, one A-value at a time; major releases (v8, v9, v10) appear every ~5 years, with minor releases like v10.1 in between.

#### 7.2 Direct link to #61 Abbo et al. 2025
#61 compares Metis/SoHO and UVCS Doppler-dimming observations using **R(T_e) ≡ G_{O VI}(T_e)/G_{Ne VIII}(T_e) computed with CHIANTI v10.1**. v10.1 (this paper) modifies all four inputs of that R(T_e) curve: (a) collision strengths (Mao et al. 2021 O-sequence R-matrix), (b) ionization / recombination rates (Table 1's 13 ions), (c) the ionization balance (Fig. 18), and (d) the Ne abundance (+35%). Critically evaluating #61's R(T_e) analysis therefore requires reading #64. **The fact that #61 cites v10.1 explicitly (not just "CHIANTI") signals awareness of this distinction.**

#### 7.3 Accuracy of solar spectroscopic diagnostics
Essentially every quantity derived from EUV/X-ray spectroscopy — formation temperature, differential emission measure (DEM), elemental abundances, non-thermal velocities — depends on the accuracy of G(T_e, n_e). v10.1 changes ionization rate coefficients by >10% for several ions (Table 1) and changes the photospheric Ne abundance by ~35%. These shifts propagate directly into the analysis of next-generation spectrometers — Solar Orbiter/SPICE, EUVST, MUSE. The paper is a standing reminder that **explicit version citation and re-computation of headline results when the database updates** must become standard practice.

#### 7.4 Exemplar of open-source astrophysics infrastructure
For 25+ years CHIANTI has released both data and software under a CC-BY licence. As this paper demonstrates, new measurements and new theoretical work in the literature get folded into the database, and the act of folding-in is itself reported in a peer-reviewed journal (ApJS). This is precisely how **reproducible astrophysics** operates in practice.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
