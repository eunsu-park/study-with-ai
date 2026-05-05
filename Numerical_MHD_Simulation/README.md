# Numerical MHD Simulation / 수치 MHD 시뮬레이션

## Overview / 개요
A **methodology-focused** track on numerical magnetohydrodynamic (MHD) simulation of the solar atmosphere, interior, and heliosphere — covering the equations and conservation laws, finite-volume / finite-difference / spectral schemes, divergence-cleaning techniques, adaptive mesh refinement (AMR), and the major community codes (MURaM, Bifrost, Stagger, MPI-AMRVAC, Athena, PLUTO, Bifrost). Covers solar magnetoconvection simulations, flux emergence, NLFFF magnetic-field extrapolation, reconnection-resolved simulations, and partially-ionized / multi-fluid extensions. Cross-cuts with Solar_Physics (theory) and Magnetic_Reconnection_Eruption (target phenomena).

태양 대기·내부·헬리오스피어의 **수치 MHD 시뮬레이션 방법론** 학습 트랙. 방정식·보존 법칙, 유한체적/차분/스펙트럼 기법, divergence cleaning, AMR(adaptive mesh refinement), 주요 커뮤니티 코드(MURaM, Bifrost, Stagger, MPI-AMRVAC, Athena, PLUTO 등) 를 다룸. 태양 magnetoconvection, flux emergence, NLFFF 자기장 외삽, reconnection-resolved 시뮬레이션, 부분 이온화·multi-fluid 확장 포함. Solar_Physics(이론) 와 Magnetic_Reconnection_Eruption(현상) 과 횡단.

## Learning Roadmap / 학습 로드맵

### Phase 1: MHD Equations & Classical Numerical Schemes / MHD 방정식·고전 수치 기법 (1980s–1990s)
- Ideal/resistive MHD equations / 이상·저항 MHD 방정식
- Conservation form, Riemann solvers / 보존형, Riemann solver
- Constrained transport for ∇·B = 0 / ∇·B = 0 보장 기법
- Operator splitting, divergence cleaning / 연산자 분할, divergence cleaning

### Phase 2: Solar Convection & Magnetoconvection Simulations / 태양 대류·magnetoconvection 시뮬레이션 (1995–2015)
- Stein & Nordlund convection simulations / Stein-Nordlund 대류
- MURaM / Bifrost realistic atmospheres / MURaM·Bifrost 현실 대기
- Flux emergence simulations (Cheung & Isobe 2014 LRSP) / 플럭스 출현
- Sunspot and pore formation / 흑점·pore 형성

### Phase 3: Coronal MHD & NLFFF Extrapolation / 코로나 MHD·NLFFF 외삽 (2000–2020)
- Force-free, magnetostatic, and full MHD coronal models / Force-free, magnetostatic, 풀 MHD 코로나 모델
- NLFFF extrapolation methods (Wiegelmann reviews) / NLFFF 외삽 방법
- Coronal MHD with realistic radiation (MURaM-Corona) / 현실 복사 코로나 MHD

### Phase 4: Reconnection-Resolved & Plasmoid-Resolving Simulations / 재결합 해상·plasmoid 해상 시뮬레이션 (2010s)
- Plasmoid instability simulations / Plasmoid 불안정성 시뮬레이션
- Sheet instability and Sweet-Parker-Lundquist scaling / sheet 불안정성, Sweet-Parker-Lundquist 스케일링
- Pontin 2020/2022 LRSP reviews / Pontin 2020·2022 LRSP 리뷰

### Phase 5: Multi-Physics Frontier — Partial Ionization, Multi-fluid, GPU / Multi-physics 프론티어 — 부분 이온화·multi-fluid·GPU (2015–Present)
- Khomenko 2015 partial-ionization LRSP / Khomenko 2015 부분 이온화 LRSP
- Two-fluid and multi-fluid solar atmosphere / 이체·다체 유체 태양 대기
- GPU-accelerated MHD codes / GPU 가속 MHD 코드
- Keppens 2025 multiphase prominences and rain / Keppens 2025 multiphase prominence·rain

## Directory Structure / 디렉토리 구조
```
Numerical_MHD_Simulation/
├── papers/          # Reading list and per-paper notes / 리딩 리스트 및 논문별 노트
├── notes/           # Theory & numerics notes (CT, AMR, Riemann solvers) / 이론·수치 노트
├── notebooks/       # Implementations (1D Sod, Brio-Wu, simple MHD) / 구현
├── scripts/         # Code-runner wrappers (MPI-AMRVAC, Athena++) / 코드 래퍼
├── data/            # Sample simulation outputs / 샘플 시뮬레이션 출력
└── README.md
```

## Source Topics for Initial Migration / 초기 이관 출처
- `Solar_Physics/` — Parker dynamo simulations, force-free field papers (e.g., #21 Parker 1988), Rempel 2009 sunspot
- `Living_Reviews_in_Solar_Physics/` — #16 Nordlund 2009, #20 Charbonneau 2010 dynamo, #30 Stein 2012 magnetoconvection, #31 Mackay 2012 flux ropes, #35 Wiegelmann 2012 NLFFF, #38 Cheung-Isobe 2014 flux emergence, #45 Khomenko 2015 partial ionization, #69 Pontin 2020, #71 Charbonneau 2020, #72 Wiegelmann 2021, #76 Fan 2021, #87 Keppens 2025
- New entries: simulation method papers (Stone Athena, Mignone PLUTO, Vögler MURaM, Gudiksen Bifrost) — to be curated.

Phase B 이관 예정. 코드별 method paper 추가 큐레이션 필요.

## Status / 상태
**Active** — Initial scaffolding 2026-05-01.
