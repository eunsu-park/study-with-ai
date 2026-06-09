# Numerical MHD Simulation Paper Reading List / 수치 MHD 시뮬레이션 논문 읽기 목록

A methodology-focused track on numerical MHD simulation of the solar atmosphere, interior, and heliosphere. Phase B migration completed 2026-05-01: 1 entry migrated from `Solar_Physics/`; LRSP MHD-simulation reviews cross-referenced (LRSP source-track policy). New entries (community code method papers) to be curated.

태양 대기·내부·헬리오스피어 수치 MHD 시뮬레이션 방법론 트랙. 2026-05-01 Phase B 이관 — `Solar_Physics/` 1편 이관, LRSP MHD-simulation 종설 cross-reference (LRSP source-track 정책). 커뮤니티 코드 method paper 신규 큐레이션 예정.

---

## Phase 1: MHD Equations & Classical Numerical Schemes / MHD 방정식·고전 수치 기법 (1980s–2000s)

> _New entries to curate:_
> - Brio & Wu 1988 — MHD Riemann solver (J. Comput. Phys. 75, 400)
> - Powell+ 1999 — 8-wave scheme, ∇·B handling (J. Comput. Phys. 154, 284)
> - Tóth 2000 — ∇·B = 0 in numerical MHD (J. Comput. Phys. 161, 605)
> - Dedner+ 2002 — hyperbolic divergence cleaning (J. Comput. Phys. 175, 645)
> - Stone+ 2008 — Athena code (ApJS 178, 137)
> - Mignone+ 2007 — PLUTO code (ApJS 170, 228)
> - Keppens+ 2012 / 2023 — MPI-AMRVAC

(Method papers to be curated.)

---

## Phase 2: Solar Convection & Magnetoconvection Simulations / 태양 대류·magnetoconvection (1989–2009)

### 1. Penumbral Structure and Outflows in Simulated Sunspots
- **Authors**: M. Rempel, M. Schüssler, R. H. Cameron, M. Knölker
- **Year**: 2009
- **DOI**: 10.1126/science.1173798
- **Why it matters**: First fully self-consistent 3D radiative MHD simulation of an entire sunspot — umbra, penumbra, and surrounding granulation — that spontaneously reproduced filamentary penumbral structure and the Evershed flow. Closed a decades-long gap between observation and theory of sunspot fine structure. / 흑점 전체(암부, 반암부, 주변 입상 조직) 의 최초 자기 일관적 3D 복사 MHD 시뮬레이션으로, 섬유상 반암부 구조와 Evershed 흐름을 자발적으로 재현. 흑점 미세 구조의 관측과 이론 사이 수십 년 간격을 줄임.
- **Prerequisites**: Radiative MHD, magnetoconvection / 복사 MHD, 자기대류
- **Status**: [ ] (migrated from Solar_Physics #55)

### Cross-references from Living Reviews in Solar Physics / LRSP 교차 참조
> - LRSP #16 Nordlund+ 2009 — Solar Surface Convection
> - LRSP #18 Fan 2009 — Magnetic Fields in the Solar Convection Zone
> - LRSP #30 Stein 2012 — Solar Surface Magneto-Convection
> - LRSP #38 Cheung & Isobe 2014 — Flux Emergence (Theory)
> - LRSP #55 Rempel 2009 — N/A (this is migrated entry SP #55, in this topic)

> _New entries to curate:_
> - Vögler+ 2005 — MURaM code first description (A&A 429, 335)
> - Gudiksen+ 2011 — Bifrost code (A&A 531, A154)
> - Stein & Nordlund 1989 — first realistic granulation (ApJ 342, L95)

---

## Phase 3: Coronal MHD & NLFFF Extrapolation / 코로나 MHD·NLFFF 외삽 (2000–2020)

### Cross-references from Living Reviews in Solar Physics / LRSP 교차 참조
> - LRSP #20 Charbonneau 2010 — Dynamo Models of the Solar Cycle
> - LRSP #31 Mackay & Yeates 2012 — The Sun's Global Photospheric and Coronal Magnetic Fields
> - LRSP #35 Wiegelmann & Sakurai 2012 — Solar Force-Free Magnetic Fields
> - LRSP #71 Charbonneau 2020 — Dynamo Models of the Solar Cycle (update)
> - LRSP #72 Wiegelmann+ 2021 — Solar Force-Free Magnetic Fields (update)
> - LRSP #76 Fan 2021 — Magnetic Fields in the Solar Convection Zone (update)

> _New entries to curate:_
> - Rempel 2017 — MURaM-Corona simulations (ApJ 834, 10)
> - Schrijver+ 2008 — NLFFF benchmark (ApJ 675, 1637)
> - Bingert & Peter 2011 — coronal heating simulations (A&A 530, A112)

---

## Phase 4: Reconnection-Resolved & Plasmoid Simulations / 재결합·plasmoid 시뮬레이션 (2009–Present)

### Cross-references from Living Reviews in Solar Physics / LRSP 교차 참조
> - LRSP #69 Pontin & Priest 2020 — Magnetic Reconnection: MHD Theory and Modelling
> - LRSP #77 Pontin 2022 — magnetic topology updates

> _New entries to curate:_
> - Loureiro+ 2007 — plasmoid instability (Phys. Plasmas 14, 100703)
> - Bhattacharjee+ 2009 — fast plasmoid reconnection (Phys. Plasmas 16, 112102)
> - Daughton+ 2011 — kinetic plasmoid (Nat. Phys. 7, 539)

---

## Phase 5: Multi-Physics Frontier — Partial Ionization, Multi-fluid, GPU / Multi-physics 프론티어 (2015–Present)

### Cross-references from Living Reviews in Solar Physics / LRSP 교차 참조
> - LRSP #45 Khomenko 2015 — Partial Ionization in the Solar Atmosphere
> - LRSP #87 Keppens 2025 — Modeling Multiphase Plasma in the Corona: Prominences and Rain

> _New entries to curate:_
> - Martínez-Sykora+ 2017 — Bifrost two-fluid (ApJ 847, 36)
> - Felipe+ 2010 — magneto-acoustic propagation
> - GPU MHD code papers (e.g., GAMERA, K-Athena)

---

## Legend / 범례
- `[ ]` not started / 시작 전
- `[~]` in progress / 진행 중
- `[x]` completed / 완료

## Migration Log / 이관 로그
**2026-05-01 Phase B**: Migrated 1 entry from `Solar_Physics/papers/reading_list.md` (#55 Rempel 2009 sunspot simulation). Extensive LRSP cross-references added (LRSP entries retained in source). 14+ new method papers to curate (community codes, recent advances).

이 토픽은 Phase A 스캐폴드 후 method paper 신규 큐레이션이 가장 많이 필요. Phase B 이관은 Rempel 2009 1편으로 제한적 — 대부분 LRSP cross-reference 와 신규 큐레이션으로 채워질 예정.
