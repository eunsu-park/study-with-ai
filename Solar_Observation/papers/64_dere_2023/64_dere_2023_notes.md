---
title: "CHIANTI — An Atomic Database for Emission Lines. XVII. Version 10.1: Revised Ionization and Recombination Rates and Other Updates"
authors: [Kenneth P. Dere, Giulio Del Zanna, Peter R. Young, Enrico Landi]
year: 2023
journal: "The Astrophysical Journal Supplement Series, 268, 52 (17 pp)"
doi: "10.3847/1538-4365/acec79"
topic: Solar Observation / Atomic database / CHIANTI
tags: [CHIANTI v10.1, ionization equilibrium, dielectronic recombination, atomic data update, isoelectronic sequence, BTI scaling, R-matrix ICFT, phosphorus sequence, Asplund 2021, FIP bias]
status: completed
date_started: 2026-04-28
date_completed: 2026-04-28
---

# 64. CHIANTI XVII — Version 10.1: Revised Ionization and Recombination Rates and Other Updates / CHIANTI XVII — 버전 10.1: 갱신된 이온화·재결합 속도와 기타 업데이트

---

## 1. Core Contribution / 핵심 기여

### English
This paper is the **release note** for **CHIANTI v10.1**, the de-facto standard atomic database for modelling optically thin emission from astrophysical plasmas. CHIANTI v10 (Del Zanna et al. 2021) had completed the major Fe R-matrix overhaul, but new laboratory and theoretical results continued to accumulate. v10.1 folds in five distinct updates: (1) **Refit ionization cross sections for 13 commonly observed ions** (S XIII, O V, Mg VIII, Fe VIII–XII, Fe XIII–XV, Fe XVII, Fe XVIII) using the Burgess–Tully ionization (BTI) scaling, based on storage-ring measurements by Hahn and collaborators (Heidelberg TSR), Bernhardt et al. and Fogle et al. (Oak Ridge); (2) **Complete RR + DR rates for the entire phosphorus iso-electronic sequence** (Z = 16–30+) from Bleda et al. (2022), filling the last major gap in the systematic Badnell programme (H- through Si-like sequences were already incorporated in v7–v10); (3) **A recomputed default ionization-equilibrium file** built from the new rates, with 11 Fe-related ions showing log T_max shifted by 0.05 dex and 7 ions (V VIII–IX, Sc VI–VII, Ca VI, K V) showing peak ion-fraction changes >10%; (4) **R-matrix ICFT electron-collision and radiative datasets** for 8 ions in the N-like (O II, Si VIII, Ar XII, Ca XIV) and O-like (Si VII, S IX, Ar XI, Ca XIII) sequences from Mao et al. (2020, 2021); (5) **New abundance files**: the photospheric file `Sun_photospheric_2021_asplund.abund` based on Asplund et al. (2021) — featuring a +35% increase in neon — plus a uniformly FIP-corrected coronal file `Sun_coronal_2021_chianti.abund` derived by multiplying low-FIP (≤10 eV) abundances by 10^0.5. The paper does not present new physics; it documents the *denominator* of nearly every spectroscopic diagnostic — atomic data — and quantifies how much it has changed since v10.

### 한국어
이 논문은 천체 플라스마의 광학적 얇은(optically thin) 발광을 모델링하는 표준 도구인 **CHIANTI 원자 데이터베이스 v10.1**의 **릴리스 노트(release note)**다. CHIANTI v10(Del Zanna et al. 2021)에서 모든 Fe 이온의 R-matrix 계산이라는 큰 도약이 완료되었지만, 그 후로도 새로운 실험·이론 결과가 누적되어 왔다. v10.1은 다섯 가지 갱신을 통합한다: (1) **13개 상시 관측 이온의 이온화 단면적 재적합** — S XIII, O V, Mg VIII, Fe VIII–XII, Fe XIII–XV, Fe XVII, Fe XVIII를 Burgess–Tully 이온화(BTI) 스케일링으로 재적합. Hahn 등(Heidelberg TSR 저장 링), Bernhardt 등, Fogle 등(Oak Ridge)의 측정에 기반함; (2) **인(P) 등전자 수열 전체(Z = 16–30+)에 대한 RR + DR 속도 완성** — Bleda et al. (2022)의 결과로, Badnell의 체계적 프로그램에서 마지막 주요 공백(H 수열부터 Si 수열은 이미 v7–v10에 통합됨)을 메움; (3) **새 속도들로 재계산된 기본 이온화 평형 파일** — Fe 관련 11개 이온에서 log T_max가 0.05 dex 이동하고, 7개 이온(V VIII–IX, Sc VI–VII, Ca VI, K V)에서 피크 이온 분율이 10% 이상 변함; (4) **N 등전자(O II, Si VIII, Ar XII, Ca XIV)·O 등전자(Si VII, S IX, Ar XI, Ca XIII) 8개 이온에 대한 R-matrix ICFT 충돌·복사 데이터셋** — Mao et al. (2020, 2021); (5) **새 abundance 파일** — Asplund et al. (2021)에 기반한 광구 파일 `Sun_photospheric_2021_asplund.abund`(네온 +35%)와, 저-FIP(≤10 eV) 원소 함량에 일률적으로 10^0.5 인수를 곱해 유도한 코로나 파일 `Sun_coronal_2021_chianti.abund`. 본 논문은 새 물리를 제시하지 않는다. 대신, 거의 모든 분광 진단의 **분모(denominator)**에 해당하는 원자 데이터를 정량적으로 문서화하고 v10 대비 얼마나 변했는지를 기록한다.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction & Methodology (§1–2.1) / 도입과 방법론

#### English
**§1 Introduction (p. 1)** sets the scene: CHIANTI has been an open-source project since 1996, with the first release paper in 1997 (Dere et al. 1997 — paper #63 in this study). The current public site is `https://chiantidatabase.org`. The previous release was CHIANTI 10 (Del Zanna et al. 2021), and Young et al. (2016) and Del Zanna & Young (2020) are the modern overviews. The database contains a *minimal* set of ionization/recombination rates for all atoms and ions of all elements between H and Zn, plus *detailed* atomic data (energy levels, wavelengths, gf and A values, electron collision strengths, optional autoionization / level-resolved RR / proton rates) for the 283 ions shown in the periodic chart of Figure 1. Two physical processes drive ionization (DI: direct ionization; EA: excitation–autoionization) and two drive recombination (RR: radiative; DR: dielectronic). Dere (2007) compiled cross sections for every ion from H to Zn using measurements where available and FAC distorted-wave calculations elsewhere; v10.1 updates that compilation. Recombination has been the focus of the Badnell programme (Badnell et al. 2003) producing iso-electronic-sequence DR/RR data — Bleda et al. (2022) finished the P sequence, which is the major addition in §2.15.

**§2.1 Approach (p. 2)** explains the BTI fitting pipeline. Measured cross sections (or FAC results when no measurement exists) are first scaled with the Burgess–Tully ionization transformation (Eqs. 1–2), then fit with a spline of typically ≥9 nodes. The scaled energy `U` runs from 0 (at threshold) to 1 (infinite energy), and the assessor-chosen scaling parameter `f` controls how the energy axis is stretched so the curve is easy to fit. The resulting parameters are written to per-ion `.diparams` files (e.g. `fe_24.diparams`). High-energy behaviour at U = 1 is anchored by the Bethe limit; the EA component (Dere 2007) computed with FAC is hand-edited to threshold/amplitude consistent with the new measurements. Section 2.1 also notes Dufresne & Del Zanna (2019) and Dufresne et al. (2020) recently computed C and O cross sections from metastable levels — those will appear in a future release.

#### 한국어
**§1 서론(p. 1)**은 무대를 설정한다: CHIANTI는 1996년 이래의 오픈 소스 프로젝트로, 첫 릴리스 논문은 1997년(Dere et al. 1997 — 본 연구의 #63번)이다. 현재 공개 사이트는 `https://chiantidatabase.org`. 이전 릴리스는 CHIANTI 10(Del Zanna et al. 2021)이고, Young et al. (2016)과 Del Zanna & Young (2020)이 현대적 개관이다. 데이터베이스는 H–Zn까지 모든 원소·이온에 대한 *최소* 이온화/재결합 속도 집합과, Figure 1 주기율표에 표시된 283개 이온에 대한 *상세* 원자 데이터(에너지 준위, 파장, gf·A 값, 충돌 강도, 선택적 자동이온화/준위 분해 RR/양성자 속도)를 포함한다. 이온화는 두 과정(DI: 직접 이온화; EA: 들뜸–자동이온화), 재결합도 두 과정(RR: 복사 재결합; DR: 이중전자 재결합)으로 진행된다. Dere (2007)는 H부터 Zn까지 모든 이온의 단면적을 (측정이 있으면 측정, 없으면 FAC 왜곡파 계산으로) 종합했고, v10.1은 그 종합본을 갱신한다. 재결합은 Badnell 프로그램(Badnell et al. 2003)의 등전자 수열 단위 RR/DR 데이터 생산이 중심이었으며, Bleda et al. (2022)이 P 수열을 마무리하여 §2.15의 주요 추가가 된다.

**§2.1 접근법(p. 2)**은 BTI 적합 파이프라인을 설명한다. 측정된 단면적(또는 측정이 없으면 FAC 결과)을 먼저 Burgess–Tully 이온화 변환(식 1–2)으로 스케일하고, 일반적으로 9개 이상의 노드를 가진 스플라인으로 적합한다. 스케일된 에너지 `U`는 임계값(threshold)에서 0, 무한 에너지에서 1로 가며, 평가자가 선택하는 스케일 파라미터 `f`가 곡선이 쉽게 적합되도록 에너지 축의 늘림을 제어한다. 결과 파라미터는 이온별 `.diparams` 파일(예: `fe_24.diparams`)에 기록된다. U = 1 고에너지 거동은 Bethe 극한에 고정되고, FAC로 계산된 EA 성분(Dere 2007)은 임계값·진폭이 새 측정과 일치하도록 수동 편집된다. §2.1은 또한 Dufresne & Del Zanna (2019)와 Dufresne et al. (2020)이 최근 C·O의 메타스테이블 준위에서의 단면적을 계산했음을 언급하며, 이는 추후 릴리스에 반영될 예정이다.

### Part II: 13-Ion Cross-Section Refits (§2.2–2.13) / 13개 이온 단면적 재적합

#### English
The paper devotes §§2.2–2.13 to *each ion individually*, in iso-electronic-sequence order. The structure is repetitive: a comparison plot of "Hahn / Fogle / Bernhardt" (red) vs. "2007" (blue, the previous Dere fit) vs. "present" (black, the new fit). Below I summarise the key per-ion findings; specific numerical changes feed into the rate-coefficient ratios in Table 1 (§2.14).

- **§2.2 Be sequence: O V (Fig. 3).** Crossed-beam measurements by Fogle et al. (2008) at Oak Ridge. The 2007 cross section was ~10–15% too high at the peak; the present fit reduces it by 0.84 in the EA region near 310 eV. Earlier data (Crandall+ 1979; Falk+ 1983; Loch+ 2003) suffered from beam contamination by metastables (~30–50% above threshold). Dufresne+ 2020 R-matrix calculations agree well with Fogle.
- **§2.2 Be sequence: S XIII (Figs. 2, 4).** Hahn et al. (2012a) at the heavy-ion storage ring; the team used the ³⁵S isotope where hyperfine mixing of ³P₀–³P₁ levels shortens the metastable lifetime so that the metastables decay before measurement. The new fit reproduces the Hahn DI cross section within a few percent; the original FAC EA components at 2412 and 2786 eV were scaled down by factors of 0.75 and 0.8 respectively. Peak cross-section ratio (present/2007) is 1.03 at 1600 eV.
- **§2.3 B sequence: Mg VIII (Fig. 5).** Hahn et al. (2010) at Max-Planck TSR. New fit uses the measurements just above threshold; at 1000 eV present/2007 ratio is ~1.10.
- **§2.4 F sequence: Fe XVIII (Fig. 6).** Hahn et al. (2013) at TSR. The scattered structure near 600, 1400, 1600 eV is run-to-run background variation. At 700 eV the present cross section is ~0.92× the 2007 FAC value; at 300 eV (low) it is ~1.3× the FAC.
- **§2.5 Ne sequence: Fe XVII (Fig. 7).** Critical for X-ray spectra in 10–18 Å and EUV 200–300 Å. Original 2007 used DI from 2p only; present extends to 2s and includes EA threshold adjustments. At 2000 eV present/2007 = 1.13; ratio = 1 at 3200 eV; 0.94 at 5000 eV.
- **§2.6 Mg sequence: Fe XV (Fig. 8).** Bernhardt et al. (2014) measurements show both DI and EA. Present DI fit is to Bernhardt below 793 eV; EA component to n=3 reduced by 0.92. At 1000 eV ratio = 0.99; at 2000 eV ratio = 0.95.
- **§2.7 Al sequence: Fe XIV (Fig. 9).** EA threshold for n=3 raised; n=4 amplitude scaled by 0.8. Just above threshold ratio = 2.0; at 700 eV ratio = 0.99; above 780 eV ratio = 0.8.
- **§2.8 Si sequence: Fe XIII (Fig. 10).** New fit uses Hahn et al. (2011a, 2012b). EA contributions to n=4, 5 removed. Threshold ratio = 2.5; at 600 eV = 1.08; above 1000 eV = 0.8.
- **§2.9 P sequence: Fe XII (Fig. 11).** Hahn et al. (2011b). EA thresholds adjusted; rate coefficient at log T = 6.2 changes by ~17% (largest single ion change).
- **§2.10 S sequence: Fe XI (Fig. 12).** Hahn et al. (2012c). EA threshold for lowest-energy component moved from 882 eV → 721 eV; magnitudes scaled ×1.3, ×4.0, ×2.0 for the three EA components. Threshold ratio = 1.6; at 1000 eV = 0.97.
- **§2.11 Cl sequence: Fe X (Fig. 13).** DI cross section reduced by ~40% below 700 eV; EA threshold at 650 eV (n=2 → 3d). At threshold ratio = 1.5; at 600 eV = 0.72; above 700 eV = 0.85.
- **§2.12 Ar sequence: Fe IX (Fig. 14).** Hahn et al. (2016). 2007 FAC was a factor of ~1.5–2 too high; present fit follows the storage-ring data closely. Ratio = 0.72 at maximum (the largest reduction in Table 1).
- **§2.13 K sequence: Fe VIII (Fig. 15).** Hahn et al. (2015). EA components (5 in total, 156–200 eV) appear as steps. Hahn beam had only ~6% metastables.

#### 한국어
논문은 §§2.2–2.13을 *이온별로 하나씩* 등전자 수열 순서로 다룬다. 구조는 반복적이다: "Hahn / Fogle / Bernhardt"(빨강) vs. "2007"(파랑, 이전 Dere 적합) vs. "present"(검정, 새 적합)의 비교 플롯. 아래에 이온별 핵심 결과를 요약한다. 구체적인 수치 변화는 §2.14의 Table 1에 있는 속도 계수 비율로 전파된다.

- **§2.2 Be 수열: O V (Fig. 3).** Fogle et al. (2008)의 Oak Ridge 교차 빔 측정. 2007 단면적은 피크에서 ~10–15% 과대였고, 새 적합은 EA 영역(310 eV 근처)에서 0.84배로 줄였다. 이전 데이터(Crandall+ 1979; Falk+ 1983; Loch+ 2003)는 메타스테이블에 의한 빔 오염(임계값 위에서 ~30–50%) 문제가 있었다. Dufresne+ 2020 R-matrix 계산은 Fogle와 잘 일치.
- **§2.2 Be 수열: S XIII (Figs. 2, 4).** Hahn et al. (2012a) 중이온 저장 링. ³⁵S 동위원소의 초미세 ³P₀–³P₁ 혼합으로 메타스테이블 수명이 짧아져 측정 전에 붕괴함. 새 적합이 Hahn DI 단면적을 수 % 이내로 재현; 2412, 2786 eV 두 EA 성분(원래 FAC)을 각각 0.75, 0.8배로 축소. 1600 eV에서 피크 비율(present/2007) = 1.03.
- **§2.3 B 수열: Mg VIII (Fig. 5).** Hahn et al. (2010) Max-Planck TSR. 임계값 바로 위 측정 사용; 1000 eV에서 present/2007 = ~1.10.
- **§2.4 F 수열: Fe XVIII (Fig. 6).** Hahn et al. (2013) TSR. 600, 1400, 1600 eV 부근 구조는 측정-측정 간 배경 변화. 700 eV에서 ~0.92×; 300 eV(저)에서 ~1.3×.
- **§2.5 Ne 수열: Fe XVII (Fig. 7).** X-선 10–18 Å, EUV 200–300 Å에 핵심. 2007은 2p에서의 DI만 고려; 현재는 2s까지 확장 + EA 임계값 조정. 2000 eV에서 present/2007 = 1.13; 3200 eV에서 1; 5000 eV에서 0.94.
- **§2.6 Mg 수열: Fe XV (Fig. 8).** Bernhardt et al. (2014)이 DI·EA 모두 측정. 793 eV 아래 Bernhardt에 적합; n=3 EA 성분을 0.92로 축소. 1000 eV에서 비율 = 0.99; 2000 eV에서 0.95.
- **§2.7 Al 수열: Fe XIV (Fig. 9).** n=3 EA 임계값 상향; n=4 진폭 0.8배. 임계값 바로 위 비율 = 2.0; 700 eV = 0.99; 780 eV 위 = 0.8.
- **§2.8 Si 수열: Fe XIII (Fig. 10).** Hahn et al. (2011a, 2012b). n=4, 5 EA 기여 제거. 임계값에서 비율 = 2.5; 600 eV = 1.08; 1000 eV 위 = 0.8.
- **§2.9 P 수열: Fe XII (Fig. 11).** Hahn et al. (2011b). EA 임계값 조정; log T = 6.2에서 속도 계수 ~17% 변화 (Table 1에서 단일 이온 최대 변화).
- **§2.10 S 수열: Fe XI (Fig. 12).** Hahn et al. (2012c). 가장 낮은 EA 임계값 882 eV → 721 eV 이동; 세 EA 성분 진폭에 ×1.3, ×4.0, ×2.0 스케일. 임계값 비율 = 1.6; 1000 eV = 0.97.
- **§2.11 Cl 수열: Fe X (Fig. 13).** 700 eV 아래 DI 단면적 ~40% 축소; 650 eV에 EA 임계값(n=2 → 3d). 임계값 비율 = 1.5; 600 eV = 0.72; 700 eV 위 = 0.85.
- **§2.12 Ar 수열: Fe IX (Fig. 14).** Hahn et al. (2016). 2007 FAC가 ~1.5–2배 과대였음 — 새 적합이 저장 링 데이터를 정확히 따라감. 최대값에서 비율 = 0.72 (Table 1에서 최대 감소).
- **§2.13 K 수열: Fe VIII (Fig. 15).** Hahn et al. (2015). 5개 EA 성분(156–200 eV)이 계단형 구조로 나타남. Hahn 빔의 메타스테이블 분율은 ~6%에 불과.

### Part III: Ionization Rate Coefficients & Equilibrium (§2.14–2.16) / 이온화 속도 계수와 평형

#### English
**§2.14 Ionization Rate Coefficients (p. 11)** integrates the cross sections over a Maxwell–Boltzmann distribution (Eq. 3, see §4 below). Figure 16 shows Fe XIV: the top panel plots present (black) vs. 2007 (blue) rate coefficients on a log-log plot from 10⁶–4×10⁶ K; the bottom panel shows the present/2007 ratio dropping from 0.99 at 10⁶ K to 0.94 at 4×10⁶ K — i.e. a 6% reduction at the high end. Although the cross section at high energies dropped by 20–30%, the rate coefficient barely moves because the MB distribution at 1.91 MK does not have many electrons at energies where σ has changed (Figure 17 visualises this overlap explicitly: the green MB curve falls off well before the EA energies above 700 eV). **Table 1** lists the present/2007 ratio at T_max for the 13 refit ions:

| Ion | T (10⁶ K) | Ratio | Ion | T (10⁶ K) | Ratio |
|-----|-----------|-------|-----|-----------|-------|
| S XIII | 2.7 | 1.11 | Fe XIII | 1.7 | 1.09 |
| O V | 0.24 | 0.96 | Fe XII | 1.6 | 1.17 |
| Mg VIII | 0.79 | 1.09 | Fe XI | 1.3 | 1.09 |
| Fe XVIII | 7.9 | 1.09 | Fe X | 1.1 | 1.14 |
| Fe XVII | 5.6 | 1.06 | Fe IX | 0.79 | 0.72 |
| Fe XV | 2.2 | 0.99 | Fe VIII | 0.56 | 1.00 |
| Fe XIV | 1.9 | 0.97 | | | |

The largest reduction is Fe IX (0.72) and the largest increase is Fe XII (1.17). All others sit within ±15%.

**§2.15 RR/DR for the P Iso-electronic Sequence (p. 11)** ingests the Bleda et al. (2022) results for P-like ions Z = 16 (S II) through 30+. Bleda's calculations were validated against Novotný et al. (2012) measurements of S-like → P-like recombination (Fe XII → Fe XI), with "quite good" overall agreement. Compared to v10.0, the changes are usually small at high T (>2×10⁴ K), but at low T Bleda's rates exceed Mazzotta et al. (1998) significantly because Mazzotta often used Shull & van Steenberg (1982) and Jacobs et al. (1977) tabulations computed in LS coupling, which under-estimate the dominant low-T DR channel — the ΔN = 0 core excitation followed by ΔN = 0 radiative stabilisation. For photoionised plasmas (high-T DR with ΔN > 0), the differences matter most: e.g. Ca VI new DR is ~50% lower than Shull & van Steenberg's. Ni XIV shows large differences. **The largest single change is +30% for S I.**

**§2.16 A Revised Ionization Balance (p. 12)** uses the new ionization (§2.14) and recombination (§2.15) rates to recompute the steady-state ionization-equilibrium file. **Figure 18** plots Fe X–XIV ion fractions as functions of T (full lines = v10.1, dashed = v10.0). Across the full Fe sequence, **11 ions have log T_max shifted by 0.05 dex** and **7 ions (V VIII–IX, Sc VI–VII, Ca VI, K V) have peak fractions changed by >10%**, the latter driven by the updated DR rates from §2.15.

#### 한국어
**§2.14 이온화 속도 계수(p. 11)**는 단면적을 Maxwell–Boltzmann 분포로 적분한다(식 3, 아래 §4 참조). Figure 16은 Fe XIV: 위 패널은 10⁶–4×10⁶ K 범위에서 present(검정) vs. 2007(파랑) 속도 계수를 log-log로 표시; 아래 패널은 present/2007 비율이 10⁶ K에서 0.99, 4×10⁶ K에서 0.94로 — 즉 고온단에서 6% 감소를 보여준다. 단면적은 고에너지에서 20–30% 떨어졌지만, 1.91 MK MB 분포가 σ가 변한 에너지 영역에 전자를 거의 보내지 않기 때문에 속도 계수는 거의 움직이지 않는다(Figure 17은 이 중첩을 명시적으로 시각화한다 — 녹색 MB 곡선은 700 eV 위 EA 에너지에 도달하기 한참 전에 떨어진다). **Table 1**은 13개 재적합 이온의 T_max에서의 present/2007 비율을 보여준다:

| 이온 | T (10⁶ K) | 비율 | 이온 | T (10⁶ K) | 비율 |
|-----|-----------|-------|-----|-----------|-------|
| S XIII | 2.7 | 1.11 | Fe XIII | 1.7 | 1.09 |
| O V | 0.24 | 0.96 | Fe XII | 1.6 | 1.17 |
| Mg VIII | 0.79 | 1.09 | Fe XI | 1.3 | 1.09 |
| Fe XVIII | 7.9 | 1.09 | Fe X | 1.1 | 1.14 |
| Fe XVII | 5.6 | 1.06 | Fe IX | 0.79 | 0.72 |
| Fe XV | 2.2 | 0.99 | Fe VIII | 0.56 | 1.00 |
| Fe XIV | 1.9 | 0.97 | | | |

최대 감소는 Fe IX(0.72), 최대 증가는 Fe XII(1.17). 나머지는 모두 ±15% 안.

**§2.15 P 등전자 수열 RR/DR(p. 11)**은 P 유사 이온 Z = 16(S II)부터 30+까지의 Bleda et al. (2022) 결과를 도입한다. Bleda의 계산은 Novotný et al. (2012)이 측정한 S 유사 → P 유사 재결합(Fe XII → Fe XI)과 "꽤 잘" 일치한다. v10.0 대비 변화는 고온(>2×10⁴ K)에서는 보통 작지만, 저온에서는 Bleda의 속도가 Mazzotta et al. (1998)을 크게 상회한다. Mazzotta는 Shull & van Steenberg (1982)와 Jacobs et al. (1977)의 LS 결합 표를 자주 썼는데, 이는 저온의 지배적 DR 채널 — ΔN = 0 코어 여기 + ΔN = 0 복사 안정화 — 을 과소평가한다. 광이온화 플라스마(고온 DR, ΔN > 0)에서는 차이가 가장 중요하다. 예: Ca VI의 새 DR은 Shull & van Steenberg보다 ~50% 낮다. Ni XIV는 큰 차이를 보인다. **단일 최대 변화는 S I의 +30%**.

**§2.16 갱신된 이온화 평형(p. 12)**은 새 이온화(§2.14)·재결합(§2.15) 속도로 정상 상태 이온화 평형 파일을 재계산한다. **Figure 18**은 Fe X–XIV 이온 분율을 T의 함수로 표시(실선 = v10.1, 점선 = v10.0). 전체 Fe 수열에서 **11개 이온이 log T_max가 0.05 dex 이동**하고, **7개 이온(V VIII–IX, Sc VI–VII, Ca VI, K V)이 피크 분율이 10% 이상 변한다**. 후자는 §2.15에서 갱신된 DR 속도가 주된 원인이다.

### Part IV: Updated Atomic Models (§3) / 갱신된 원자 모델

#### English
**§3.1 H sequence: C VI** — corrects an error in the proton rate coefficients (originally Zygelman & Dalgarno 1987). The Z³ scaling was multiplied instead of divided when ingested into v6, leading to rates too large by a factor 4.7 × 10⁴. With correct rates, two-photon decay (rather than proton excitation) becomes the dominant depopulation mechanism for the 2s²S_{1/2} metastable level. The two-photon continuum for C VI at 1 MK and n_e = 10¹⁰ cm⁻³ is **>1000× stronger** than previously thought; the populations of 2p ²P_{1/2,3/2} are enhanced by 44% and 10% respectively; the C VI Lyα line (a self-blend of 2p → 1s) is **18% weaker** in v10.1.

**§3.2 B sequence: O IV** — energy levels 27, 28, 51 had been swapped with levels 29, 30, 52. Fixed; new wavelengths derived from updated energies.

**§3.3 C sequence: Ca XV** — collision strengths had been omitted in the v10 update; now restored.

**§3.4 N sequence (p. 13)** — Mao et al. (2020) presented R-matrix ICFT calculations of effective collision strengths for N-like ions O II to Zn XXIV. v10.1 incorporates new models for **O II, Si VIII, Ar XII, Ca XIV** (only the bound states retained, since these ions don't produce strong satellite lines). For O II the effective collision strengths for the first five states of the ground configuration were replaced with Tayal (2007) up to 10⁵ K (more accurate at low T). AUTOSTRUCTURE was used for the target. For O II, Si VIII, and Ar XII, the AS A values for transitions within the 2s²2p³ and 2s2p⁴ configurations were replaced with the Breit–Pauli calculations of Tachiev & Froese Fischer (2002); for Ca XIV with the MBPT calculations of Wang et al. (2016). AS radiative data differed from MCDHF/Breit–Pauli by 10–20% only.

**§3.5 O sequence (p. 14)** — Mao et al. (2021) R-matrix ICFT for O-like ions Ne III to Zn XXIII (target: 630 fine-structure levels up to nl = 5d). v10.1 incorporates new models for **Si VII, S IX, Ar XI, Ca XIII** (which had limited DW-based models in CHIANTI). EUV/UV/near-IR line intensities differ significantly with the new data. For Si VII and S IX the Tachiev & Froese Fischer (2002) Breit–Pauli A values are used for n = 2 transitions (typically 10–20% shift). For Ar XI and Ca XIII the Song et al. (2021) GRASP2K values are used (~10% differences with AS).

**§3.6 Si sequence: Fe XIII** — Zhang et al. (2021) MCDHF structure calculation gives theoretical energies within "a few hundred cm⁻¹" of experiment. Confirms Del Zanna (2012) identifications except for one strong n = 4 → n = 3 line: Del Zanna proposed 76.507 Å but Zhang's ab-initio MCDHF differs by 0.35 Å, while the alternative 76.867 Å differs by only 0.015 Å — so v10.1 adopts the new identification. Several other tentative IDs were not adopted pending more experimental work.

**§3.7 Ar sequence: Fe IX** — Ryabtsev et al. (2022) provided new experimental energies for 3p⁴3d² and 3p⁵4f configurations. CHIANTI 10 had experimental energies for all 12 of the 3p⁵4f fine-structure levels from O'Dwyer et al. (2012); these are updated. Agreement is good for nine levels; larger differences (160 and 560 cm⁻¹) for the remaining three are due to identification differences. Comparing Del Zanna et al. (2014) A values to Ryabtsev: the ¹F₃/³F₃ and ¹G₄/³F₄ labels needed to be swapped; afterwards A values agree very well. For the 3p⁴3d² configuration with 111 fine-structure levels, only 5 had previous experimental energies — 16 new and 5 updated were added from Ryabtsev. Young & Landi (2009) had identified 7 lines as Fe IX without atomic transitions; Ryabtsev provides transition info for two, agreeing with Del Zanna (2009) at 192.63 Å but differing at 194.80 Å.

**§3.8 Ca sequence: Fe VII** — energy levels updated using Kramida et al. (2022). Many levels are highly mixed (especially 3p⁵3d³); not always possible to match Kramida labels with the existing CHIANTI model. Decisions documented in Young (2023b). New model has 17 levels with newly assigned experimental energies and 10 with updates ≥100 cm⁻¹. Wavelengths in v10.1 reflect new energies. However, intensity inconsistencies between strongest predicted lines and Hinode/EIS observations (Del Zanna 2009) remain — the full solution requires further calculations and assessments.

#### 한국어
**§3.1 H 수열: C VI** — 양성자 속도 계수(원래 Zygelman & Dalgarno 1987)의 오류를 수정. v6에 도입할 때 Z³ 스케일링을 나누는 대신 곱한 결과, 속도가 4.7 × 10⁴배 과대했다. 올바른 속도로 양성자 들뜸 대신 two-photon 붕괴가 2s²S_{1/2} 메타스테이블 준위의 지배적 탈인구화 메커니즘이 된다. 1 MK, n_e = 10¹⁰ cm⁻³에서 C VI two-photon 연속체는 이전 추정보다 **>1000배 강함**; 2p ²P_{1/2,3/2} 인구는 각각 44%, 10% 증가; C VI Lyα 선(2p → 1s 자기 블렌드)은 v10.1에서 **18% 약해짐**.

**§3.2 B 수열: O IV** — 에너지 준위 27, 28, 51이 29, 30, 52와 뒤바뀌어 있던 오류 수정. 새 파장 도출.

**§3.3 C 수열: Ca XV** — v10 업데이트에서 충돌 강도가 누락되었던 것을 복원.

**§3.4 N 수열(p. 13)** — Mao et al. (2020)이 N 유사 이온 O II–Zn XXIV에 대해 R-matrix ICFT로 유효 충돌 강도를 계산. v10.1은 **O II, Si VIII, Ar XII, Ca XIV**에 대한 새 모델을 도입한다(이들은 강한 위성선을 생성하지 않으므로 속박 상태만 유지). O II는 바닥 배치의 첫 5개 상태에 대한 유효 충돌 강도를 10⁵ K까지 Tayal (2007)로 교체(저온에서 더 정확). 표적은 AUTOSTRUCTURE 사용. O II, Si VIII, Ar XII에서는 2s²2p³ 및 2s2p⁴ 배치 내 전이의 AS A 값을 Tachiev & Froese Fischer (2002) Breit–Pauli로, Ca XIV에서는 Wang et al. (2016) MBPT로 교체. AS 복사 데이터는 MCDHF/Breit–Pauli와 10–20%만 차이.

**§3.5 O 수열(p. 14)** — Mao et al. (2021) R-matrix ICFT로 O 유사 이온 Ne III–Zn XXIII (표적: nl = 5d까지 630 미세구조 준위). v10.1은 **Si VII, S IX, Ar XI, Ca XIII**에 새 모델 도입(이들은 CHIANTI에 DW 기반 제한적 모델만 있었음). EUV/UV/근적외선 선 강도가 새 데이터로 크게 달라진다. Si VII, S IX는 n = 2 전이에 Tachiev & Froese Fischer (2002) Breit–Pauli A 값(보통 10–20% 변화); Ar XI, Ca XIII는 Song et al. (2021) GRASP2K 값(AS와 ~10% 차이).

**§3.6 Si 수열: Fe XIII** — Zhang et al. (2021)의 MCDHF 구조 계산이 실험과 "수백 cm⁻¹ 이내"의 이론 에너지 제공. Del Zanna (2012)의 동정을 확인하되, 강한 n = 4 → n = 3 선 하나만 예외: Del Zanna는 76.507 Å을 제안했지만 Zhang의 ab-initio MCDHF는 0.35 Å 차이; 대안 76.867 Å은 0.015 Å만 차이 — 따라서 v10.1은 새 동정을 채택. 다른 일부 잠정 동정은 추가 실험 전까지 채택 보류.

**§3.7 Ar 수열: Fe IX** — Ryabtsev et al. (2022)이 3p⁴3d² 및 3p⁵4f 배치의 새 실험 에너지 제공. CHIANTI 10은 3p⁵4f 미세구조 준위 12개 모두 O'Dwyer et al. (2012)에서 실험 에너지를 가졌고, 이들이 갱신됨. 9개는 잘 일치, 나머지 3개에서 (160, 560 cm⁻¹) 차이는 동정 차이 때문. Del Zanna et al. (2014)의 A 값과 Ryabtsev 비교: ¹F₃/³F₃와 ¹G₄/³F₄ 라벨이 바뀌어야 했고, 수정 후 A 값은 매우 잘 일치. 111개 미세구조 준위를 가진 3p⁴3d² 배치는 이전에 5개만 실험 에너지가 있었음 — Ryabtsev에서 16개 신규 + 5개 갱신 추가. Young & Landi (2009)는 7개 선을 원자 전이 정보 없이 Fe IX로 동정했었는데, Ryabtsev가 두 개에 전이 정보 제공 — 192.63 Å는 Del Zanna (2009)와 일치, 194.80 Å는 차이.

**§3.8 Ca 수열: Fe VII** — Kramida et al. (2022)로 에너지 준위 갱신. 많은 준위가 강하게 혼합되어 있어(특히 3p⁵3d³) Kramida 라벨을 기존 CHIANTI 모델과 매칭하기 어려운 경우 다수. 결정 사항은 Young (2023b)에 문서화. 새 모델에는 새로 실험 에너지가 부여된 17개 준위와 ≥100 cm⁻¹ 갱신된 10개 준위가 있음. 그러나 가장 강한 예측 선과 Hinode/EIS 관측(Del Zanna 2009) 사이의 강도 불일치는 여전히 남아 있어, 추가 계산이 필요.

### Part V: Elemental Abundances (§4) / 원소 함량

#### English
**§4 (p. 15)** introduces the new abundance files. v10 used `Sun_photospheric_2015_scott.abund` (Asplund et al. 2009, supplemented with Scott et al. 2015a, 2015b and Grevesse et al. 2015 for some elements). v10.1 introduces `Sun_photospheric_2021_asplund.abund` containing data from the Asplund et al. (2021) compilation. That compilation is a comprehensive review of photospheric abundances of all elements up to U, with improved values obtained mostly from coupling photospheric observed spectra with 3D NLTE hydrodynamical simulations of the outer convection zone and atmosphere, plus improved atomic data. Additional inputs come from meteorite analyses, the Genesis return samples, helioseismology, and sunspot observations.

Compared to the v10.0 default set, **Asplund et al. 2021 differ by 10% or more for four elements: Li (−19%), Ne (+35%), Cl (−19%), Ti (+10%)**. The most significant change for ionised-plasma spectral modelling is the Ne abundance increase. The new Ne/O ratio is 0.23, larger than the previous 0.17, and consistent with recent spectroscopic re-evaluations in the transition region (Young 2018) and the corona during solar minimum (Landi & Testa 2015). The Ne/O ratio and absolute Ne and O abundances are central to resolving the discrepancy between helioseismic models of the solar interior and accurate-abundance-based opacities (Christensen-Dalsgaard 2021).

The earlier coronal abundance files (`Sun_coronal_1992_feldman.abund`, `Sun_coronal_2012_schmelz.abund`) were largely based on solar active region data, where the ratio of low-FIP / high-FIP elements is enhanced — the FIP bias — by averaged factors ~3–4. Those coronal files were obtained by averaging and applying empirical corrections to *older* photospheric data; they are inconsistent with the new photospheric file. v10.1 therefore introduces `Sun_coronal_2021_chianti.abund`, derived from the new Asplund et al. 2021 photospheric file by **multiplying low-FIP (FIP ≤ 10 eV) elements by a uniform factor 10^0.5** ≈ 3.16. This is provided as a "representative" coronal file; users should be aware that the FIP bias actually varies among solar structures and with temperature. Old abundance files (both photospheric and coronal) remain available in `abundance/archive`.

#### 한국어
**§4 (p. 15)**는 새 abundance 파일을 소개한다. v10은 `Sun_photospheric_2015_scott.abund`(Asplund et al. 2009 + 일부 원소에 Scott et al. 2015a,b 및 Grevesse et al. 2015 보충)을 사용했다. v10.1은 Asplund et al. (2021) 종합본을 담은 `Sun_photospheric_2021_asplund.abund`를 도입한다. 이 종합본은 U까지 모든 원소의 광구 함량에 대한 포괄적 리뷰로, 외부 대류층·대기의 3D NLTE 유체역학 시뮬레이션과 광구 관측 스펙트럼의 결합 + 개선된 원자 데이터로 도출되었다. 추가 입력은 운석 분석, Genesis 시료 회수, 태양진동학, 흑점 관측에서 옴.

v10.0 기본 집합 대비 **Asplund et al. 2021은 4개 원소에서 10% 이상 차이: Li (−19%), Ne (+35%), Cl (−19%), Ti (+10%)**. 이온화 플라스마 분광 모델링에서 가장 중요한 변화는 Ne 함량 증가다. 새 Ne/O 비율은 0.23으로 이전 0.17보다 크며, 전이 영역(Young 2018)과 태양 극소기 코로나(Landi & Testa 2015)의 최근 분광 재평가와 일치한다. Ne/O 비율과 Ne·O 절대 함량은 태양 내부의 태양진동학적 모델과 정밀 함량 기반 opacity 사이의 불일치를 해결하는 데 핵심이다(Christensen-Dalsgaard 2021).

이전 코로나 abundance 파일(`Sun_coronal_1992_feldman.abund`, `Sun_coronal_2012_schmelz.abund`)은 주로 활동 영역 데이터에 기반했으며, 거기서 저-FIP/고-FIP 원소 비율이 평균 ~3–4배 향상된다(FIP bias). 이 파일들은 *옛* 광구 데이터에 평균치와 경험적 보정을 적용해 얻은 것이라, 새 광구 파일과 일관되지 않는다. 따라서 v10.1은 새 Asplund et al. 2021 광구 파일에서 **저-FIP(FIP ≤ 10 eV) 원소에 일률적으로 10^0.5 ≈ 3.16배를 곱해** 유도한 `Sun_coronal_2021_chianti.abund`를 도입한다. 이는 "대표" 코로나 파일로 제공된 것이며, 사용자는 FIP bias가 실제로는 태양 구조와 온도에 따라 변한다는 점에 유의해야 한다. 옛 파일들(광구·코로나 모두)은 `abundance/archive`에 보존된다.

### Part VI: Conclusions (§5) / 결론

#### English
**§5 (p. 16)** is a checklist summary: (i) ionization rates updated for 13 ions; (ii) complete DR data added for the P sequence; (iii) core data sets — energy levels, A values, electron excitation rates — updated for 15 ions in total (including the N-, O-sequence ions discussed in §3 plus the Fe XIII, Fe IX, Fe VII updates and miscellaneous fixes); (iv) new ionization-equilibrium file built from the revised rates; (v) new abundance files. The release date is 2023 July 17. Acknowledgments thank Dr. Stefan Schippers for the machine-readable Bleda data; funding from NASA grants (KPD, PRY, EL) and STFC consolidated grant (GDZ at DAMTP).

#### 한국어
**§5 (p. 16)**는 체크리스트 요약: (i) 13개 이온 이온화 속도 갱신; (ii) P 수열 DR 데이터 완성; (iii) 핵심 데이터셋 — 에너지 준위, A 값, 전자 들뜸 속도 — 총 15개 이온 갱신(§3의 N·O 수열 이온 + Fe XIII, Fe IX, Fe VII 갱신 + 잡다한 수정 포함); (iv) 새 이온화 평형 파일; (v) 새 abundance 파일. 릴리스 날짜는 2023년 7월 17일. 감사 사항: Bleda 데이터의 기계 판독 형식 제공에 Stefan Schippers; 자금 출처는 NASA 보조금(KPD, PRY, EL)과 STFC 통합 보조금(DAMTP의 GDZ).

---

## 3. Key Takeaways / 핵심 시사점

1. **CHIANTI v10.1 is an "incremental but pervasive" minor release.** / **CHIANTI v10.1은 "점진적이지만 전방위적인" 마이너 릴리스다.**
   - **EN**: Although billed as a minor release, v10.1 touches every link in the spectroscopic chain — cross sections, rate coefficients, ionization equilibrium, atomic models, and abundances — meaning that nearly any G(T_e, n_e) computed with v10 will shift, even if individual changes are <20%. Citing "CHIANTI" without a version is no longer adequate.
   - **KR**: 마이너 릴리스로 분류되지만 v10.1은 분광 사슬의 모든 고리 — 단면적, 속도 계수, 이온화 평형, 원자 모델, 함량 — 를 건드린다. 따라서 v10으로 계산된 거의 모든 G(T_e, n_e)는 이동할 것이고, 개별 변화가 <20%여도 마찬가지다. 버전 없이 "CHIANTI"만 인용하는 것은 더 이상 충분하지 않다.

2. **Storage-ring measurements have outstripped FAC predictions for most key Fe ions.** / **저장 링 측정이 대부분의 핵심 Fe 이온에서 FAC 예측을 능가한다.**
   - **EN**: Hahn and collaborators' Heidelberg TSR / Oak Ridge campaign produced laboratory cross sections for Fe VIII–XVIII over ~2010–2016. Where 2007 FAC overshot (Fe IX by 30–40%) or undershot (Fe XII by 17%), the new fits track the data within a few percent. Future releases will shift further as Dufresne et al. extend ab initio calculations to metastables.
   - **KR**: Hahn 등의 Heidelberg TSR/Oak Ridge 캠페인이 Fe VIII–XVIII의 단면적을 2010–2016년에 걸쳐 실험적으로 생산했다. 2007 FAC가 과대(Fe IX, 30–40%)·과소(Fe XII, 17%)였던 곳에서, 새 적합은 데이터를 수 % 이내로 따라간다. Dufresne 등이 메타스테이블에 ab initio 계산을 확장하면 추후 릴리스가 더 이동할 것이다.

3. **Cross-section changes do not always propagate to rate-coefficient changes.** / **단면적 변화가 항상 속도 계수 변화로 이어지진 않는다.**
   - **EN**: Figure 17 makes the point graphically: even when σ shifts by 20–30% above 700 eV, the Fe XIV rate at 1.91 MK barely moves (ratio 0.97) because the MB distribution at that temperature has too few electrons in that energy range. The lesson: thermally averaged quantities are insensitive to high-energy details, but extremely sensitive to threshold accuracy.
   - **KR**: Figure 17이 이를 시각적으로 보여준다 — σ가 700 eV 위에서 20–30% 이동해도, 1.91 MK에서 Fe XIV 속도는 거의 움직이지 않는다(비율 0.97). 그 온도의 MB 분포가 해당 에너지 영역에 전자를 거의 보내지 않기 때문. 교훈: 열평균량은 고에너지 디테일에 둔감하지만 임계값 정확도에는 매우 민감하다.

4. **The phosphorus-sequence DR completion plugs the last big iso-electronic gap.** / **인 수열 DR 완성으로 마지막 큰 등전자 공백이 메워졌다.**
   - **EN**: Bleda et al. (2022) closed a 20-year programme that began with Badnell et al. (2003). For low-temperature DR (which dominates in photoionised plasmas and the cool transition region), the new rates can exceed the legacy Mazzotta/Shull–van Steenberg values significantly, because the older LS-coupling tabulations under-counted ΔN = 0 channels. For solar-corona temperatures the changes are smaller but non-zero (e.g. S I +30%, Ca VI −50% in the photoionised regime).
   - **KR**: Bleda et al. (2022)이 Badnell et al. (2003)에서 시작된 20년 프로그램을 마무리했다. 저온 DR(광이온화 플라스마와 차가운 전이 영역에서 지배적)에 대해, 새 속도는 Mazzotta/Shull–van Steenberg의 레거시 값을 크게 상회할 수 있다 — 옛 LS 결합 표가 ΔN = 0 채널을 과소평가했기 때문. 태양 코로나 온도에서는 변화가 작지만 0은 아니다(예: 광이온화 영역에서 S I +30%, Ca VI −50%).

5. **The +35% Ne abundance increase is the single most consequential change for solar UV/EUV diagnostics.** / **Ne 함량의 +35% 증가는 태양 UV/EUV 진단에서 가장 큰 영향을 미치는 단일 변화다.**
   - **EN**: Asplund et al. (2021) raise the photospheric Ne from log = 7.93 (2009) to 8.06 (2021). Any line ratio involving Ne (Ne VII, Ne VIII, Ne IX) — including Doppler-dimming diagnostics like O VI / Ne VIII — is directly affected. The new Ne/O ratio (0.23 vs. 0.17) is also relevant to the long-standing solar-interior opacity problem in helioseismology (Christensen-Dalsgaard 2021).
   - **KR**: Asplund et al. (2021)은 광구 Ne를 log = 7.93(2009)에서 8.06(2021)으로 올렸다. Ne를 포함하는 모든 선 비율(Ne VII, Ne VIII, Ne IX) — O VI / Ne VIII 도플러 변광 진단 포함 — 이 직접 영향받는다. 새 Ne/O 비율(0.23 vs. 0.17)은 태양진동학의 오랜 태양 내부 opacity 문제와도 관련된다(Christensen-Dalsgaard 2021).

6. **Coronal abundances should now be derived consistently from the new photospheric file.** / **코로나 함량은 이제 새 광구 파일에서 일관되게 유도되어야 한다.**
   - **EN**: Older Feldman 1992 / Schmelz 2012 coronal files mixed older photospheric values with empirical FIP corrections, producing internal inconsistencies. v10.1's `Sun_coronal_2021_chianti.abund` applies a uniform 10^0.5 boost to all FIP ≤ 10 eV elements on top of Asplund 2021. This is "representative" only — the actual FIP bias varies with structure (active region vs. quiet Sun vs. coronal hole) and with formation temperature.
   - **KR**: 옛 Feldman 1992 / Schmelz 2012 코로나 파일은 옛 광구 값과 경험적 FIP 보정을 섞어 내적 불일치를 만들었다. v10.1의 `Sun_coronal_2021_chianti.abund`는 Asplund 2021 위에 FIP ≤ 10 eV 원소 모두에 일률적으로 10^0.5 인수를 적용한다. 이는 "대표"일 뿐이며, 실제 FIP bias는 구조(활동 영역 vs. 조용한 태양 vs. 코로나 구멍)와 형성 온도에 따라 달라진다.

7. **The C VI proton-rate fix shows that legacy bugs lurk for years.** / **C VI 양성자 속도 버그 수정은 레거시 버그가 수년간 잠재함을 보여준다.**
   - **EN**: A factor-of-Z⁶ error (4.7 × 10⁴ for Z = 6) in the ingestion of Zygelman & Dalgarno (1987) survived from CHIANTI v6 (2009) until 2023. Consequences cascade: C VI two-photon continuum >1000× stronger; Lyα 18% weaker; level populations shifted by tens of percent. Any user of those quantities pre-v10.1 should re-run their analyses.
   - **KR**: Zygelman & Dalgarno (1987) 도입 시의 Z⁶ 인수 오류(Z = 6에 대해 4.7 × 10⁴)가 CHIANTI v6(2009)부터 2023년까지 살아남았다. 결과는 연쇄적: C VI two-photon 연속체 >1000배 강해짐; Lyα 18% 약해짐; 준위 인구 수십 % 이동. v10.1 이전의 해당 양을 쓴 모든 사용자는 분석을 재실행해야 한다.

8. **Reproducible astrophysics requires explicit version-tagged citation of atomic databases.** / **재현 가능한 천체 물리는 원자 데이터베이스의 명시적 버전 인용을 요구한다.**
   - **EN**: This release-note paper itself is the proof: the same project, same authors, same database can yield different results between v10.0 (2021) and v10.1 (2023). Modern best practice — followed by paper #61 (Abbo et al. 2025) which cites "CHIANTI v10.1" — is to fix the version, ideally with a Zenodo DOI or `chiantipy.__version__` log, in the methods section of any paper that reports physical conclusions derived from the database.
   - **KR**: 이 릴리스 노트 자체가 증거다 — 동일 프로젝트, 동일 저자, 동일 데이터베이스가 v10.0(2021)과 v10.1(2023) 사이에서 다른 결과를 낼 수 있다. 현대적 모범 — 논문 #61 Abbo et al. 2025가 "CHIANTI v10.1"을 명시한 것 — 은 데이터베이스 기반 물리 결론을 보고하는 모든 논문의 방법 섹션에서 버전(가능하면 Zenodo DOI 또는 `chiantipy.__version__` 로그)을 고정하는 것이다.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Burgess–Tully Ionization (BTI) Scaling / BTI 스케일링 (Eqs. 1–2)

The fitting transformation that compresses cross sections onto a finite interval [0, 1] for spline interpolation:

$$
U \;=\; 1 \;-\; \frac{\ln f}{\ln(u - 1 + f)}
$$

$$
\Sigma \;=\; \frac{u\,\sigma\,I^{2}}{\ln u + 1}
$$

**Term-by-term / 용어 설명:**

| Symbol | English | 한국어 |
|--------|---------|--------|
| $u = E/I$ | Incident electron energy in units of the ionization potential | 이온화 퍼텐셜 단위 입사 전자 에너지 |
| $E$ | Free-electron kinetic energy | 자유 전자 운동에너지 |
| $I$ | Ionization potential of the ion | 이온의 이온화 퍼텐셜 |
| $\sigma(E)$ | Ionization cross section [cm²] | 이온화 단면적 |
| $f$ | Assessor-chosen scaling parameter (typically ~1–10) | 평가자가 선택하는 스케일 파라미터 |
| $U$ | Scaled energy on [0, 1]; U = 0 at threshold (u = 1), U = 1 at u → ∞ | 스케일된 에너지 [0, 1]; 임계값에서 0, 무한대에서 1 |
| $\Sigma$ | Scaled cross section; finite at U = 0 and U = 1 (the latter set by the Bethe limit) | 스케일된 단면적; U = 0과 U = 1에서 유한(후자는 Bethe 극한이 결정) |

**Why it works / 왜 작동하는가:**
- **EN**: $\sigma(E)$ has a sharp threshold rise and a slow $\sim \ln(u)/u$ Bethe tail at high energy. Direct spline-fitting in $E$ is poor because the curve spans 4+ decades. The transformation maps $E \in [I, \infty)$ to $U \in [0, 1]$, and divides out the asymptotic Bethe behaviour, leaving a smooth $\Sigma(U)$ that ≥9 spline nodes can reproduce to within experimental error.
- **KR**: $\sigma(E)$는 임계값에서 급격히 상승하고 고에너지에서 $\sim \ln(u)/u$의 느린 Bethe 꼬리를 가진다. $E$에서 직접 스플라인 적합하면 곡선이 4+ 데케이드에 걸쳐 어렵다. 이 변환은 $E \in [I, \infty)$를 $U \in [0, 1]$로 매핑하고 점근적 Bethe 거동을 나눠주어, ≥9 스플라인 노드로 실험 오차 이내 재현이 가능한 매끄러운 $\Sigma(U)$를 남긴다.

### 4.2 Maxwellian Ionization Rate Coefficient / 맥스웰 이온화 속도 계수 (Eq. 3)

$$
R(T) \;=\; \int_{v_{\rm IP}}^{\infty} v\,\sigma(E)\,f(v, T)\,dv,
\qquad E = \tfrac{1}{2} m v^{2}
$$

**Term-by-term / 용어 설명:**

| Symbol | English | 한국어 |
|--------|---------|--------|
| $R(T)$ | Maxwell-averaged ionization rate coefficient [cm³ s⁻¹] | 맥스웰 평균 이온화 속도 계수 |
| $T$ | Electron temperature [K] | 전자 온도 |
| $v$ | Electron speed | 전자 속도 |
| $v_{\rm IP}$ | Velocity such that $\tfrac{1}{2} m v_{\rm IP}^{2} = I$ — the cutoff below which ionization is impossible | 운동에너지가 IP와 같은 전자 속도 — 그 아래는 이온화 불가능 |
| $\sigma(E)$ | Cross section, evaluated at $E = \tfrac{1}{2} m v^{2}$ | 단면적, $E = \tfrac{1}{2} m v^{2}$에서 평가 |
| $f(v, T)$ | Maxwell–Boltzmann velocity distribution at electron temperature $T$ | 전자 온도 $T$에서의 맥스웰–볼츠만 속도 분포 |

**Operational note / 운영 메모:**
- **EN**: The integrand $v \sigma(E) f(v, T)$ is the product of three sharply peaked or rolling-off functions (Figure 17): the cross section rises from 0 at $v_{\rm IP}$ then peaks; the velocity factor $v$ grows linearly; the MB distribution falls super-exponentially. The overlap region is narrow, which is why a 20–30% change in σ at high energies can still leave $R(T_{\max})$ nearly unchanged.
- **KR**: 적분항 $v \sigma(E) f(v, T)$는 세 함수의 곱이다(Figure 17): 단면적은 $v_{\rm IP}$에서 0으로 시작해 피크; 속도 인수 $v$는 선형 증가; MB 분포는 초지수적으로 감소. 중첩 영역은 좁고, 그래서 고에너지에서 σ가 20–30% 변해도 $R(T_{\max})$는 거의 변하지 않을 수 있다.

### 4.3 Total Recombination Rate / 총 재결합 속도

$$
\alpha_{\rm tot}(T_e) \;=\; \alpha_{\rm RR}(T_e) \;+\; \alpha_{\rm DR}(T_e)
$$

- **EN**: RR (radiative recombination) and DR (dielectronic recombination) act in parallel. v10.1 ingests Bleda et al. (2022)'s RR + DR fit parameters for the entire P sequence (Z = 16–30+). Standard fitting form in CHIANTI is the Badnell parametrisation:
$$
\alpha_{\rm DR}(T) = T^{-3/2} \sum_{i} c_{i}\,\exp(-E_{i} / k T)
$$
with the $\{c_i, E_i\}$ stored per ion. RR uses the Verner–Ferland fit
$$
\alpha_{\rm RR}(T) = A \left[\sqrt{T/T_{0}} (1 + \sqrt{T/T_{0}})^{1-B} (1 + \sqrt{T/T_{1}})^{1+B}\right]^{-1}
$$
with parameters $A, B, T_0, T_1$ tabulated.
- **KR**: RR(복사 재결합)과 DR(이중전자 재결합)은 병렬로 작동. v10.1은 Bleda et al. (2022)의 RR + DR 적합 파라미터를 P 수열 전체(Z = 16–30+)에 도입. CHIANTI 표준 적합 형식은 Badnell 파라미터화:
$$
\alpha_{\rm DR}(T) = T^{-3/2} \sum_{i} c_{i}\,\exp(-E_{i} / k T)
$$
$\{c_i, E_i\}$는 이온별 저장. RR은 Verner–Ferland 적합 사용:
$$
\alpha_{\rm RR}(T) = A \left[\sqrt{T/T_{0}} (1 + \sqrt{T/T_{0}})^{1-B} (1 + \sqrt{T/T_{1}})^{1+B}\right]^{-1}
$$
파라미터 $A, B, T_0, T_1$ 표화.

### 4.4 Coronal Ionization Equilibrium / 코로나 이온화 평형

For each charge state $z$ of element $Z$, balance ionization in-flow against recombination out-flow:

$$
n_{Z, z-1}\,R^{\rm ion}_{z-1 \to z}(T_e) \;=\; n_{Z, z}\,\alpha^{\rm rec}_{z \to z-1}(T_e)
$$

**Solving the chain / 사슬 풀기:**

$$
\frac{n_{Z, z}}{n_{Z, z-1}} \;=\; \frac{R^{\rm ion}_{z-1 \to z}}{\alpha^{\rm rec}_{z \to z-1}}
$$

iterating up the chain and normalising $\sum_z n_{Z, z} = n_Z$ gives the **ionization fraction**:

$$
f_{Z, z}(T_e) \;\equiv\; \frac{n_{Z, z}}{n_Z}
$$

- **EN**: This is what `Sun_chianti_v10.1.ioneq` (the file CHIANTI ships) tabulates. Section 2.16 / Figure 18 quantify the v10 → v10.1 changes for Fe X–XIV.
- **KR**: 이것이 `Sun_chianti_v10.1.ioneq`(CHIANTI 배포 파일)에 표화된 양이다. §2.16 / Figure 18이 Fe X–XIV에 대한 v10 → v10.1 변화를 정량화한다.

### 4.5 Contribution Function (CHIANTI's headline output) / 방출 함수 (CHIANTI 핵심 출력)

For an emission line from upper level $j$ to lower level $i$:

$$
G_{ji}(T_e, n_e) \;=\; \frac{A_{ji}\,n_j(T_e, n_e)}{n_{Z,z}(T_e, n_e)}\;\frac{n_{Z,z}(T_e)}{n_Z}\;\frac{n_Z}{n_H}\;\frac{1}{n_e}
$$

**Term-by-term / 용어 설명:**

| Symbol | English | 한국어 |
|--------|---------|--------|
| $A_{ji}$ | Einstein spontaneous emission coefficient | 자발 방출 계수 |
| $n_j / n_{Z,z}$ | Upper-level population fraction (from collisional excitation + radiative decay; uses CHIANTI's collision-strength files) | 상위 준위 인구 분율 (충돌 들뜸 + 복사 붕괴; CHIANTI 충돌 강도 파일 사용) |
| $n_{Z,z} / n_Z = f_{Z,z}(T_e)$ | Ionization fraction (updated in v10.1) | 이온화 분율 (v10.1에서 갱신됨) |
| $n_Z / n_H$ | Elemental abundance relative to hydrogen (new abundance file in v10.1) | 수소 대비 원소 함량 (v10.1의 새 abundance 파일) |
| $1 / n_e$ | Normalisation per unit electron density | 단위 전자 밀도당 정규화 |

The line emissivity per unit emission measure is then $\varepsilon_{ji} = G_{ji}(T_e, n_e) \cdot n_e n_H$.

- **EN**: **All four right-hand-side factors changed in v10.1** for at least some ions: collision strengths via §3, ionization fractions via §2.16, and abundances via §4. Even when the cross-section refits in §§2.2–2.13 produce only modest rate-coefficient changes (Table 1), the cumulative G(T_e, n_e) shift can be larger.
- **KR**: **v10.1에서 우변 4개 인수 모두 일부 이온에서 변경됨**: §3을 통한 충돌 강도, §2.16을 통한 이온화 분율, §4를 통한 함량. §§2.2–2.13의 단면적 재적합이 모달한 속도 계수 변화만 만들어도(Table 1), 누적 G(T_e, n_e) 이동은 더 클 수 있다.

### 4.6 Worked Numerical Example: Fe IX/X Formation Temperature Shift / 수치 예: Fe IX/X 형성 온도 이동

**The setup / 설정:**
The Fe IX → Fe X ionization stage transition determines the formation temperature of two lines critical to solar EUV: the Fe IX 171 Å line (SDO/AIA's hottest narrowband channel, peaking near log T = 5.9) and the Fe X 174 Å line. Their ratio is a workhorse temperature diagnostic in the upper transition region / lower corona.

**Step 1 — Ionization rate change / 이온화 속도 변화:**
From Table 1, at T_max:
- Fe IX (T_max = 0.79 × 10⁶ K): $R^{\rm ion}_{\rm Fe\,IX \to Fe\,X}$(v10.1) / $R^{\rm ion}$(v10.0) = **0.72** (28% reduction).
- Fe X (T_max = 1.1 × 10⁶ K): $R^{\rm ion}_{\rm Fe\,X \to Fe\,XI}$(v10.1) / $R^{\rm ion}$(v10.0) = **1.14** (14% increase).

**Step 2 — Effect on Fe IX equilibrium fraction / Fe IX 평형 분율에의 영향:**
In coronal equilibrium $f_{\rm Fe\,IX} \propto R^{\rm ion}_{\rm Fe\,VIII \to Fe\,IX} / R^{\rm ion}_{\rm Fe\,IX \to Fe\,X}$. With the denominator dropping by a factor 0.72 (i.e. multiplied by 1/0.72 ≈ 1.39), Fe IX **persists to higher temperatures** than in v10.0 — Fe IX no longer ionises away as efficiently. The peak of $f_{\rm Fe\,IX}(T)$ is therefore expected to **shift towards higher T_max by roughly $\Delta \log T_{\max} \approx +0.05$ dex** (consistent with the Section 2.16 statement that "11 ions have log T_max shifted by 0.05 dex").

**Step 3 — Effect on Fe X equilibrium fraction / Fe X 평형 분율에의 영향:**
Fe X is the upper of the Fe IX → Fe X transition. With Fe IX → Fe X rate dropping (×0.72) and Fe X → Fe XI rate rising (×1.14), the Fe X peak fraction is **suppressed** relative to v10.0. Estimate: at T = 1.1 × 10⁶ K,
$$
f_{\rm Fe\,X}(\text{v10.1}) / f_{\rm Fe\,X}(\text{v10.0}) \;\approx\; \frac{0.72}{1.14} \;\approx\; 0.63
$$
in the simple two-stage approximation — a ~37% reduction in peak Fe X fraction, in qualitative agreement with the 7-ion list of >10% changes in Section 2.16 (which lists V VIII–IX, Sc VI–VII, Ca VI, K V; Fe X is not in that explicit list but Figure 18 plots Fe X–XIV as the headline visualisation).

**Step 4 — Propagation to G(T) / G(T)로의 전파:**
For the Fe X 174 Å line, $G_{174}(T_e) \propto f_{\rm Fe\,X}(T_e)$ (other factors are weak functions of T). Therefore the contribution function in v10.1 is **suppressed by ~30–40%** at log T = 6.0 relative to v10.0. The peak temperature is shifted towards lower T because the "shoulder" formed by the rising Fe X stage now rolls over earlier.

**Step 5 — Downstream impact on R(T_e) diagnostics / 하류 R(T_e) 진단으로의 영향:**
Although the Doppler-dimming diagnostic of paper #61 (Abbo et al. 2025) uses the Li-like ions O VI and Ne VIII (not Fe IX/X), the same mechanism applies: any line whose ionization-fraction peak shifts can change the inferred $T_e$ for a fixed line ratio. For #61's R(T_e) ≡ G_{O VI}(T_e) / G_{Ne VIII}(T_e), the v10 → v10.1 update shifts: (i) ionization equilibrium of O VI and Ne VIII (Section 2.16, though these are not in the Section 2.14 13-ion list, they are still affected through the new RR/DR rates); (ii) Ne abundance +35% (Section 4) — this scales G_{Ne VIII} upward by 1.35, decreasing R(T_e) by ~26% at all T. This single abundance shift is *larger* than the rate-coefficient shifts and is the dominant v10 → v10.1 change for the #61 diagnostic.

**The lesson / 교훈:**
**EN**: A "5% change in the ionization rate" can produce a "30% change in the contribution function" at a specific T because the chain $R^{\rm ion} \to f(T) \to G(T)$ amplifies through the steeply-falling tails of the ion-fraction curves. Re-running #61's R(T_e) analysis with v10.0 vs. v10.1 should yield different inferred temperatures, and the difference — though probably small in absolute terms — should be reported as part of the systematic-error budget.
**KR**: 특정 T에서 "이온화 속도의 5% 변화"가 "방출 함수의 30% 변화"를 만들 수 있다 — $R^{\rm ion} \to f(T) \to G(T)$ 사슬이 이온 분율 곡선의 가파른 꼬리를 통해 증폭하기 때문. #61의 R(T_e) 분석을 v10.0 vs. v10.1로 재실행하면 다른 추론 온도가 나올 것이고, 그 차이는 절대적으로는 작을지라도 시스템 오차 예산의 일부로 보고되어야 한다.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1932 ─── Bethe high-energy ionization cross-section formula
1962 ─── Lotz semi-empirical ionization formulas (long the standard)
1969 ─── Burgess: dielectronic recombination identified as dominant in coronae
1982 ─── Shull & van Steenberg DR rates (LS coupling, used for ~25 years)
1987 ─── Zygelman & Dalgarno H-like proton excitation rates
1992 ─── Burgess & Tully: BT scaling for collision strengths (parent of BTI)
1996 ─── CHIANTI project launched (SOHO support)
1997 ─── ░░░ #63 Dere et al. 1997: CHIANTI v1.0 ░░░
1998 ─── Mazzotta et al. ionization equilibrium (long-time baseline)
2002 ─── Gu: Flexible Atomic Code (FAC) — workhorse for theoretical σ
2003 ─── Badnell et al. systematic DR programme begins (H-sequence)
2007 ─── Dere ionization cross sections H–Zn (basis of CHIANTI v6)
2008 ─── Fogle et al. crossed-beam Be-sequence (C III, N IV, O V)
2009 ─── CHIANTI v6 (Dere et al.); Asplund et al. photospheric abundances
2010 ─── Hahn et al. begin Heidelberg TSR Fe-ion ionization campaign
2010-16 ─ Hahn / Bernhardt / Fogle storage-ring measurements accumulate
2014 ─── Bernhardt et al. Fe XV measurements
2016 ─── CHIANTI v8 (Del Zanna et al.); Young et al. modern overview
2020 ─── ░░░ Mao et al. R-matrix ICFT for N-like ions ░░░
         Dufresne et al. C and O metastable ionization cross sections
2021 ─── ░░░ CHIANTI v10 (Del Zanna et al.) — major Fe R-matrix update ░░░
         Mao et al. R-matrix ICFT for O-like ions
         Asplund et al. new photospheric abundances (3D NLTE)
         Zhang et al. MCDHF Fe XIII structure
2022 ─── Bleda et al. RR/DR for entire phosphorus sequence
         Ryabtsev et al. Fe IX experimental energies
         Kramida et al. NIST ASD v5.10
2023 ─── ★★★ #64 CHIANTI v10.1 (this paper) — incremental update ★★★
2025 ─── #61 Abbo et al. uses CHIANTI v10.1 for R(T_e) diagnostic
       ─── Future: v11 — Dufresne metastables, more N/O sequence ions
```

**Where v10.1 sits / v10.1의 위치:**
- **EN**: v10.1 is a *consolidation* release: it does not introduce a new computational technique (R-matrix was the v10 leap); rather, it folds in two years of accumulated measurements (Hahn et al.), one major theoretical completion (Bleda P sequence), one continuation of an ongoing programme (Mao et al. N/O), and one external community update (Asplund et al. abundances). This pattern — major release every ~5 years, minor releases in between — is the operational rhythm of a long-term astrophysical infrastructure project.
- **KR**: v10.1은 *통합* 릴리스다 — 새 계산 기법을 도입하지 않는다(R-matrix는 v10의 도약). 대신 2년 동안 누적된 측정(Hahn 등), 하나의 큰 이론적 완성(Bleda P 수열), 하나의 진행 중 프로그램의 연장(Mao 등 N/O), 하나의 외부 공동체 업데이트(Asplund 등 함량)를 통합한다. 이 패턴 — ~5년마다 메이저 릴리스, 그 사이에 마이너 릴리스 — 이 장기 천체 물리 인프라 프로젝트의 운영 리듬이다.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **#63 Dere et al. 1997** (CHIANTI v1.0) | **Direct lineage** — same lead author, same database architecture, 25 years apart. v10.1 is the 17th release in the sequence #63 began. The G(T_e, n_e) formalism defined in 1997 is unchanged. / **직접 계보** — 같은 주저자, 같은 데이터베이스 구조, 25년 차이. v10.1은 #63이 시작한 시퀀스의 17번째 릴리스. 1997년에 정의된 G(T_e, n_e) 형식주의가 변하지 않음. | **Foundational** / **기초적**. Reading #63 then #64 traces the entire arc of CHIANTI's evolution. / #63 다음 #64를 읽으면 CHIANTI 진화 전체 호를 추적. |
| **#61 Abbo et al. 2025** (Doppler-dimming R(T_e)) | **Direct downstream consumer** — explicitly cites "CHIANTI v10.1" for computing the O VI / Ne VIII R(T_e) curve. The Ne abundance change (+35%, Section 4) and the ionization-equilibrium revision (Section 2.16) both feed directly into #61's diagnostic. / **직접 하류 소비자** — O VI / Ne VIII R(T_e) 곡선 계산에 "CHIANTI v10.1"을 명시. Ne 함량 변화(+35%, §4)와 이온화 평형 개정(§2.16)이 모두 #61 진단에 직접 반영됨. | **Critical** / **결정적**. To critically assess #61's R(T_e), one must read #64. / #61의 R(T_e)를 비판적으로 평가하려면 #64 필독. |
| **#65 Mason & Monsignori-Fossi 1994** (coronal diagnostics review) | **Co-authorship lineage** — H. E. Mason is one of the original CHIANTI co-founders (Dere, Landi, **Mason**, Monsignori-Fossi, Young). The diagnostic framework reviewed in #65 (line ratios, density/temperature diagnostics) is the *use case* for which CHIANTI v1.0 (#63) and all subsequent releases were built. / **공저자 계보** — H. E. Mason은 CHIANTI 원공동창립자(Dere, Landi, **Mason**, Monsignori-Fossi, Young) 중 한 명. #65에서 리뷰한 진단 프레임워크(선 비율, 밀도/온도 진단)가 CHIANTI v1.0(#63)과 이후 모든 릴리스의 *사용 사례*. | **Conceptual prerequisite** / **개념적 전제**. The diagnostics #65 describes are precisely what v10.1's improved atomic data sharpens. / #65가 설명하는 진단이 v10.1의 개선된 원자 데이터로 정밀화되는 것. |
| **Asplund et al. 2021** (photospheric abundances) | **External input** — the new `Sun_photospheric_2021_asplund.abund` file is built from this compilation. The +35% Ne change is the single most consequential change in v10.1 for solar UV/EUV diagnostics. / **외부 입력** — 새 `Sun_photospheric_2021_asplund.abund` 파일이 이 종합본에서 구축됨. Ne의 +35% 변화는 태양 UV/EUV 진단에서 v10.1의 가장 큰 변화. | **High**. Any spectroscopic abundance analysis using CHIANTI is directly linked. / CHIANTI를 쓴 모든 분광 함량 분석은 직접 연결됨. |
| **Mao et al. 2020, 2021** (R-matrix ICFT for N- and O-like ions) | **External input** — Section 3.4 (N-like: O II, Si VIII, Ar XII, Ca XIV) and Section 3.5 (O-like: Si VII, S IX, Ar XI, Ca XIII) ingest these calculations. This is the continuation of the iso-electronic R-matrix programme that produced the v10 Fe overhaul. / **외부 입력** — §3.4(N 유사: O II, Si VIII, Ar XII, Ca XIV)와 §3.5(O 유사: Si VII, S IX, Ar XI, Ca XIII)가 이 계산을 도입. v10 Fe 정비를 만든 등전자 R-matrix 프로그램의 연장. | **Medium-high**. Affects 8 specific ions used in coronal diagnostics. / 코로나 진단에 쓰이는 8개 특정 이온에 영향. |
| **Bleda et al. 2022** (P-sequence RR + DR) | **External input** — Section 2.15. Closes a 20-year iso-electronic recombination programme that began with Badnell et al. 2003. Affects ions from S II up through Cu XIV. / **외부 입력** — §2.15. Badnell et al. 2003에서 시작된 20년 등전자 재결합 프로그램을 마무리. S II부터 Cu XIV까지 영향. | **Medium-high**. Largest single theoretical addition in v10.1; primarily affects photoionised plasma analyses. / v10.1의 단일 최대 이론 추가; 주로 광이온화 플라스마 분석에 영향. |
| **Hahn et al. 2010, 2011, 2012, 2013, 2015, 2016** (storage-ring ionization) | **External input** — drives the cross-section refits in §§2.2–2.13 for 11 of the 13 updated ions. / **외부 입력** — §§2.2–2.13의 13개 갱신 이온 중 11개의 단면적 재적합을 추동. | **Medium-high**. The single largest experimental input to v10.1. / v10.1의 단일 최대 실험 입력. |
| **Del Zanna et al. 2021** (CHIANTI v10) | **Predecessor** — v10.1 explicitly defines itself as the increment over v10. All ionization-equilibrium plots in Figure 18 compare v10.1 to v10. / **선행** — v10.1이 v10에 대한 증분으로 명시적으로 정의됨. Figure 18의 모든 이온화 평형 플롯이 v10.1을 v10과 비교. | **Critical**. v10 defines the baseline against which v10.1 is measured. / v10이 v10.1이 측정되는 기준선을 정의. |

---

## 7. References / 참고문헌

**Primary paper / 주 논문:**
- Dere, K. P., Del Zanna, G., Young, P. R., Landi, E., "CHIANTI — An Atomic Database for Emission Lines. XVII. Version 10.1: Revised Ionization and Recombination Rates and Other Updates," *The Astrophysical Journal Supplement Series*, **268**, 52 (17 pp), 2023. DOI: [10.3847/1538-4365/acec79](https://doi.org/10.3847/1538-4365/acec79)

**Earlier CHIANTI release papers / 이전 CHIANTI 릴리스 논문:**
- Dere, K. P., Landi, E., Mason, H. E., Monsignori-Fossi, B. C., Young, P. R., "CHIANTI — an atomic database for emission lines," *A&AS*, **125**, 149, 1997. (= paper #63 in this study)
- Dere, K. P., Landi, E., Young, P. R., et al., "CHIANTI — an atomic database for emission lines. IX. Ionization rates, recombination rates, ionization equilibria for the elements hydrogen through zinc and updated atomic data," *A&A*, **498**, 915, 2009.
- Del Zanna, G., Dere, K. P., Young, P. R., Landi, E., "CHIANTI — An Atomic Database for Emission Lines. XVI. Version 10," *ApJ*, **909**, 38, 2021.
- Young, P. R., Dere, K. P., Landi, E., Del Zanna, G., Mason, H. E., "The CHIANTI atomic database," *JPhB*, **49**, 074009, 2016.
- Del Zanna, G., Mason, H. E., "Solar UV and X-ray spectral diagnostics," *LRSP*, **15**, 5, 2018.
- Del Zanna, G., Young, P. R., "Atomic Data for Plasma Spectroscopy: The CHIANTI Database," *Atoms*, **8**, 46, 2020.

**Methodology / 방법론:**
- Burgess, A., Tully, J. A., "On the analysis of collision strengths and rate coefficients," *A&A*, **254**, 436, 1992.
- Dere, K. P., "Ionization rate coefficients for the elements hydrogen through zinc," *A&A*, **466**, 771, 2007.
- Gu, M. F., "The flexible atomic code: applications," *ApJL*, **579**, L103, 2002.
- Badnell, N. R., O'Mullane, M. G., Summers, H. P., et al., "Dielectronic recombination data for dynamic finite-density plasmas. I.," *A&A*, **406**, 1151, 2003.
- Badnell, N. R., "A Breit–Pauli distorted wave implementation for AUTOSTRUCTURE," *CoPhC*, **182**, 1528, 2011.

**Storage-ring / lab measurements / 저장 링·실험 측정:**
- Hahn, M., Bernhardt, D., Lestinsky, M., et al., 2010, *ApJ*, **712**, 1166 (Mg VIII).
- Hahn, M., Grieser, M., Krantz, C., et al., 2011a, *ApJ*, **735**, 105.
- Hahn, M., Bernhardt, D., Grieser, M., et al., 2011b, *ApJ*, **729**, 76 (Fe XII).
- Hahn, M., Bernhardt, D., Grieser, M., et al., 2012a, *PhRvA*, **85**, 042713 (S XIII).
- Hahn, M., Becker, A., Grieser, M., et al., 2012b, *ApJ*, **760**, 80.
- Hahn, M., Becker, A., Bernhardt, D., et al., 2012c, *ApJ*, **761**, 79 (Fe XI).
- Hahn, M., Becker, A., Bernhardt, D., et al., 2013, *ApJ*, **767**, 47 (Fe XVII, XVIII).
- Hahn, M., Becker, A., Bernhardt, D., et al., 2015, *ApJ*, **813**, 16 (Fe VIII).
- Hahn, M., Becker, A., Bernhardt, D., et al., 2016, *JPhB*, **49**, 084006 (Fe IX).
- Bernhardt, D., Becker, A., Grieser, M., et al., 2014, *PhRvA*, **90**, 012702 (Fe XV).
- Fogle, M., Bahati, E. M., Bannister, M. E., et al., 2008, *ApJS*, **175**, 543 (Be-sequence).

**Theoretical / 이론:**
- Bleda, E. A., Altun, Z., Badnell, N. R., 2022, *A&A*, **668**, A72 (P-sequence RR/DR).
- Mao, J., Badnell, N. R., Del Zanna, G., 2020, *A&A*, **643**, A95 (N-sequence ICFT).
- Mao, J., Badnell, N. R., Del Zanna, G., 2021, *A&A*, **653**, A81 (O-sequence ICFT).
- Dufresne, R. P., Del Zanna, G., 2019, *A&A*, **626**, A123.
- Dufresne, R. P., Del Zanna, G., Badnell, N. R., 2020, *MNRAS*, **497**, 1443.
- Tachiev, G. I., Froese Fischer, C., 2002, *A&A*, **385**, 716 (Breit–Pauli).
- Tayal, S. S., 2007, *ApJS*, **171**, 331 (O II).
- Wang, K., Si, R., Dang, W., et al., 2016, *ApJS*, **223**, 3 (Ca XIV MBPT).
- Song, C. X., Zhang, C. Y., Wang, K., et al., 2021, *ADNDT*, **138**, 101377 (GRASP2K).
- Zhang, X. H., Del Zanna, G., Wang, K., et al., 2021, *ApJS*, **257**, 56 (Fe XIII MCDHF).
- Zygelman, B., Dalgarno, A., 1987, *PhRvA*, **35**, 4085 (proton rates).

**Ionization equilibrium baselines / 이온화 평형 기준:**
- Mazzotta, P., Mazzitelli, G., Colafrancesco, S., Vittorio, N., 1998, *A&AS*, **133**, 403.
- Shull, J. M., van Steenberg, M., 1982, *ApJS*, **48**, 95.
- Jacobs, V. L., Davis, J., Kepple, P. C., Blaha, M., 1977, *ApJ*, **211**, 605.
- Savin, D. W., Laming, J. M., 2002, *ApJ*, **566**, 1166.

**Abundances / 함량:**
- Asplund, M., Amarsi, A. M., Grevesse, N., 2021, *A&A*, **653**, A141 (new photospheric default).
- Asplund, M., Grevesse, N., Sauval, A. J., Scott, P., 2009, *ARA&A*, **47**, 481 (previous default).
- Scott, P., Asplund, M., Grevesse, N., et al., 2015a, *A&A*, **573**, A26.
- Scott, P., Grevesse, N., Asplund, M., et al., 2015b, *A&A*, **573**, A25.
- Grevesse, N., Scott, P., Asplund, M., Sauval, A. J., 2015, *A&A*, **573**, A27.
- Landi, E., Testa, P., 2015, *ApJ*, **800**, 110 (coronal Ne/O).
- Young, P. R., 2018, *ApJ*, **855**, 15 (TR Ne/O).
- Christensen-Dalsgaard, J., 2021, *LRSP*, **18**, 2 (helioseismology).
- Laming, J. M., 2015, *LRSP*, **12**, 2 (FIP review).

**Spectral identifications / 분광 동정:**
- Ryabtsev, A. N., Kononov, E. Y., Young, P. R., 2022, *ApJ*, **936**, 60 (Fe IX).
- O'Dwyer, B., Del Zanna, G., Badnell, N. R., Mason, H. E., Storey, P. J., 2012, *A&A*, **537**, A22 (Fe IX).
- Young, P., 2023a, "Updating Fe IX energy levels for CHIANTI 10.1, v1.0," Zenodo, doi:10.5281/zenodo.7803672.
- Young, P., 2023b, "Updating Fe VII energy levels for CHIANTI 10.1, v1.0," Zenodo, doi:10.5281/zenodo.7799540.
- Kramida, A., Ralchenko, Yu., Reader, J. & NIST ASD Team, 2022, NIST Atomic Spectra Database (v5.10).

**Software / 소프트웨어:**
- CHIANTI public site: https://chiantidatabase.org
- ChiantiPy: Python interface to the CHIANTI database.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
