# Helioseismology & Asteroseismology / 일진학 및 항성진동학

## Overview / 개요
A study track on **acoustic and gravity oscillations of the Sun and stars** as probes of stellar interiors — from Leighton's discovery of the 5-minute oscillation (1962) through GONG and SOHO/MDI's global helioseismology, time-distance and ring-diagram local helioseismology, to space-based asteroseismology with CoRoT, *Kepler*, TESS, and PLATO. Covers ridge-fitting, inversion of mode frequencies and splittings for internal sound speed and rotation, near-surface effects, and the extension of solar techniques to red giants, subdwarfs, and main-sequence stars.

태양과 항성의 **음향·중력 진동을 항성 내부의 탐침**으로 사용하는 학습 트랙. Leighton 의 5분 진동 발견(1962)부터 GONG·SOHO/MDI 의 글로벌 일진학, time-distance·ring-diagram 국지 일진학, CoRoT·Kepler·TESS·PLATO 우주 임무의 항성진동학까지. ridge fitting, mode 주파수·splitting 역산을 통한 내부 음속·회전 결정, near-surface effect, 태양 기법의 적색거성·subdwarf·주계열성 확장 등을 다룸.

## Learning Roadmap / 학습 로드맵

### Phase 1: Discovery and Theory of Solar Oscillations / 태양 진동의 발견과 이론 (1962–1975)
- 5-minute oscillation discovery (Leighton 1962) / 5분 진동 발견
- Ulrich's identification with trapped acoustic modes (1970) / Ulrich 의 갇힌 음향 모드 동정
- Deubner's k–ω diagram (1975) / Deubner k–ω 다이어그램

### Phase 2: Global Helioseismology — Internal Structure & Rotation / 글로벌 일진학 — 내부 구조 및 회전 (1995–2010)
- GONG, BiSON, SOHO/MDI networks / 글로벌 관측망
- Internal sound speed and rotation profile inversion / 내부 음속·회전 프로파일 역산
- Tachocline and convection zone base / 타코클라인과 대류층 바닥

### Phase 3: Local Helioseismology / 국지 일진학 (2000–2015)
- Time-distance helioseismology / 시간-거리 일진학
- Ring-diagram analysis / Ring-diagram 분석
- Subsurface flow and active region detection / 표면 아래 유동 및 활동 영역 검출

### Phase 4: Asteroseismology — Stellar Oscillations / 항성진동학 — 항성 진동 (2010–Present)
- *Kepler*-era main-sequence and subgiant asteroseismology / *Kepler* 시대 주계열성·subgiant 항성진동학
- Red-giant mixed modes and core rotation / 적색거성 혼합 모드와 핵 회전
- TESS and PLATO ensemble seismology / TESS·PLATO 앙상블 항성진동학

### Phase 5: Modern Frontiers — Convection, Magnetism, and Numerical Forward Modelling / 현대 프론티어 — 대류, 자기장, 수치 forward 모델링 (2015–Present)
- Magnetic effects on mode frequencies / 자기장의 모드 주파수 효과
- Convective noise and stochastic excitation / 대류 잡음과 확률적 여기
- Forward-modelling with realistic simulations / 현실적 시뮬레이션 기반 forward 모델링

## Directory Structure / 디렉토리 구조
```
Helioseismology_Asteroseismology/
├── papers/          # Curated paper reading list and per-paper notes / 논문 리딩 리스트 및 논문별 노트
├── notes/           # Theory and concept notes / 이론 및 개념 노트
├── notebooks/       # Practice and implementation / 실습 및 구현
├── scripts/         # Standalone Python/IDL scripts / 독립 실행 스크립트
├── data/            # Sample datasets (mode frequency tables, time-series) / 샘플 데이터셋
└── README.md
```

## Source Topics for Initial Migration / 초기 이관 출처
Papers will be migrated in Phase B from:
- `Solar_Physics/` — Leighton 1962, Ulrich 1970, Deubner 1975, Christensen-Dalsgaard 1996, Schou 1998
- `Living_Reviews_in_Solar_Physics/` — Gizon 2005, Howe 2009, Hanasoge 2012, Christensen-Dalsgaard 2021, Hanasoge 2022, Basu 2016, Houdek 2015, García 2019, Hall 2008
- New asteroseismology entries (Kepler, TESS, PLATO) to be curated.

Phase B 에서 이관 예정. 각 논문은 정식 entry 가 본 토픽으로 이동하고, 출처 토픽에는 cross-reference 만 남김.

## Status / 상태
**Active** — Initial scaffolding 2026-05-01; reading list to be populated in migration Phase B.
