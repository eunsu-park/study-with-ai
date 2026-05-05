# Plasma Spectroscopy & Diagnostics / 플라즈마 분광학 및 진단

## Overview / 개요
A **methodology-focused** track on inferring physical conditions of optically thin astrophysical plasmas from spectroscopic observations — emission-line intensities, line ratios, broadband filter response, and continuum spectra. Covers the foundations of optically thin radiative transfer; atomic processes (collisional excitation, ionization equilibrium, dielectronic recombination); reference databases (CHIANTI, AtomDB); diagnostics for electron temperature, density, flows, and ionization state; **differential emission measure (DEM) inversion** principles and algorithms; element abundances and the FIP effect. Applies to solar corona EUV/X-ray observations, stellar coronae, AGN, supernova remnants — anywhere optically thin plasma is observed.

**광학적으로 얇은 천체 플라즈마의 물리 조건을 분광 관측 (분광선 강도, line ratio, 광대역 필터 응답, 연속 스펙트럼) 으로부터 추론하는 방법론**을 학습하는 트랙. 광학적으로 얇은 복사 전달 기초, 원자 과정(충돌여기·이온화 평형·dielectronic recombination), 참조 데이터베이스(CHIANTI, AtomDB), 전자 온도·밀도·유동·이온화 상태 진단, **DEM 역산 원리·알고리즘**, 원소 abundance·FIP 효과. 태양 코로나 EUV/X-ray, 항성 코로나, AGN, supernova remnant 등 광학적으로 얇은 플라즈마 모두에 적용.

## Learning Roadmap / 학습 로드맵

### Phase 1: Foundations — Optically Thin Radiative Transfer / 광학적으로 얇은 복사 전달 기초 (1960s–1970s)
- Pottasch 1963 — emission measure analysis / 방출 측정 분석
- Jefferies-Orrall-Zirker 1972 — bivariate DEM / 이변량 DEM
- Withbroe 1975 — modern DEM formalism / 현대 DEM 형식론
- Craig & Brown 1976 — ill-posed inverse problem / ill-posed 역문제

### Phase 2: Atomic Physics & Reference Databases / 원자 물리학·참조 데이터베이스 (1990s–2020s)
- CHIANTI v1 (Dere+ 1997) and updates (Dere+ 2023 v10.1) / CHIANTI 데이터베이스
- AtomDB / APED / AtomDB·APED
- Mason 1994 VUV diagnostics review / Mason 1994 VUV 진단 종설
- Del Zanna & Mason 2018 LRSP review / Del Zanna-Mason 2018 LRSP 리뷰

### Phase 3: Differential Emission Measure (DEM) — Principles & Algorithms / DEM — 원리·알고리즘 (1980–2023)
- Withbroe-Sylwester iterative method / Withbroe-Sylwester 반복법
- Kashyap & Drake MCMC (PINTofALE) / MCMC DEM
- Hannah & Kontar regularized inversion (demreg) / 정규화 역산
- Cheung 2015 sparse DEM, Plowman 2013 basis pursuit / sparse DEM, basis pursuit
- Aschwanden 2013 single-Gaussian, Morgan & Pickering 2019 SITES / 단일-Gaussian, SITES
- Massa 2023 RML / Regularized maximum likelihood

### Phase 4: Density / Temperature / Flow Diagnostics / 밀도·온도·유동 진단 (modern)
- Line ratios for n_e / 밀도용 line ratio
- Forbidden vs allowed lines / 금지선·허용선
- Doppler shifts and non-thermal broadening / Doppler 천이·비열적 폭
- EM-loci method / EM-loci 기법

### Phase 5: Element Abundances & FIP Effect / 원소 abundance·FIP 효과 (1970s–2020s)
- FIP effect discovery (Pottasch 1963) / FIP 효과 발견
- Coronal vs photospheric abundances / 코로나 vs 광구 abundance
- Laming 2015 LRSP review / Laming 2015 LRSP 종설
- Modern non-Maxwellian and out-of-equilibrium effects / 비-Maxwell·비평형 효과

## Directory Structure / 디렉토리 구조
```
Plasma_Spectroscopy_Diagnostics/
├── papers/          # Reading list and per-paper notes / 리딩 리스트 및 논문별 노트
├── notes/           # Theory notes (Saha, Boltzmann, contribution functions) / 이론 노트
├── notebooks/       # Implementations (CHIANTI calls, demreg, sparse DEM) / 구현
├── scripts/         # Utilities (line-ratio calculators, abundance fitting) / 유틸리티
├── data/            # Sample datasets (Hinode/EIS, SDO/AIA, RESIK) / 샘플 데이터셋
└── README.md
```

## Source Topics for Initial Migration / 초기 이관 출처
- `Solar_Observation/` — **the major source**: #30 Cheung 2015, #63 Dere 1997 CHIANTI I, #64 Dere 2023 CHIANTI v10.1, #65 Mason 1994 VUV diagnostics, #66–80 (Phase 8 DEM 15 papers)
- `Living_Reviews_in_Solar_Physics/` — #41 Laming 2015 (FIP), #59 Del Zanna & Mason 2018 (UV/X-ray spectroscopy)

After migration, Solar_Observation Phase 7–8 will be substantially trimmed and refocused on instruments + observation techniques.

## Status / 상태
**Active** — Initial scaffolding 2026-05-01. Largest Phase B migration target (~25 papers).
