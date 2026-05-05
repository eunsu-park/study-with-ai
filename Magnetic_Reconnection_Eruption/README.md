# Magnetic Reconnection & Solar Eruptions / 자기 재결합 및 태양 폭발현상

## Overview / 개요
A study track on **magnetic reconnection** and the **eruptive phenomena it powers** — solar flares and coronal mass ejections (CMEs). From Sweet (1958) and Parker (1957) Sweet–Parker theory through Petschek (1964) fast reconnection, the CSHKP standard flare model (Carmichael, Sturrock, Hirayama, Kopp & Pneuman), CME initiation models (breakout, flux-rope, torus instability), to modern 3D reconnection, plasmoid instability, turbulent reconnection, and Magnetospheric Multiscale (MMS) and PSP/STIX observations. Cross-cuts between solar atmospheric physics, in-situ heliospheric observations, and magnetospheric reconnection.

**자기 재결합**과 그것이 구동하는 **폭발 현상** — 태양 플레어와 코로나 질량 방출(CME) — 을 학습하는 트랙. Sweet(1958)·Parker(1957) Sweet–Parker 이론부터 Petschek(1964) 빠른 재결합, CSHKP 표준 플레어 모델, CME 시작 모델(breakout, flux-rope, torus instability), 현대의 3D 재결합·plasmoid 불안정성·난류 재결합, MMS·PSP/STIX 관측까지. 태양 대기 물리, in-situ 헬리오스피어 관측, 자기권 재결합을 가로지름.

## Learning Roadmap / 학습 로드맵

### Phase 1: Reconnection Foundations / 재결합 기초 (1957–1964)
- Sweet–Parker (Sweet 1958, Parker 1957) / Sweet–Parker 이론
- Petschek fast reconnection (1964) / Petschek 빠른 재결합
- Vasyliunas review (1975) / Vasyliunas 종설

### Phase 2: Standard Flare Model — CSHKP / 표준 플레어 모델 — CSHKP (1964–1990)
- Carmichael (1964), Sturrock (1966), Hirayama (1974), Kopp & Pneuman (1976) / 4인의 표준 모델
- Hard X-ray and EUV signatures / 경 X선·EUV 시그니처
- Two-ribbon flare and post-flare loops / 양가닥 플레어와 post-flare 루프

### Phase 3: CME Initiation Models / CME 시작 모델 (1995–2010)
- Aulanier & Démoulin flux-rope formation (1998) / 플럭스 로프 생성
- Antiochos breakout model (1999) / Antiochos breakout 모델
- Kliem & Török torus instability (2006) / Kliem-Török torus 불안정성
- Forbes catastrophe / Forbes 파국 모델

### Phase 4: 3D Reconnection, Plasmoids, and Turbulence / 3D 재결합, plasmoid, 난류 (2009–Present)
- Loureiro & Uzdensky plasmoid instability / Loureiro-Uzdensky plasmoid 불안정성
- 3D reconnection at QSLs and null points / QSL·null point 3D 재결합
- Turbulent reconnection (Lazarian & Vishniac) / 난류 재결합

### Phase 5: Modern Observations & Numerical Simulations / 현대 관측 및 수치 시뮬레이션 (2010–Present)
- MMS in-situ reconnection (asymmetric, kinetic) / MMS in-situ 재결합 (비대칭, 동역학)
- PSP/STIX flare diagnostics / PSP/STIX 플레어 진단
- 3D MHD flare simulations / 3D MHD 플레어 시뮬레이션
- Reconnection-driven heliospheric structures / 재결합 구동 헬리오스피어 구조

## Directory Structure / 디렉토리 구조
```
Magnetic_Reconnection_Eruption/
├── papers/          # Reading list and per-paper notes / 리딩 리스트 및 논문별 노트
├── notes/           # Theory notes (CSHKP, Sweet-Parker derivations, etc.) / 이론 노트
├── notebooks/       # Implementations (current sheet, plasmoid simulations) / 구현
├── scripts/         # Utilities / 유틸리티
├── data/            # Sample datasets (RHESSI/STIX flare events, MMS reconnection events) / 샘플 데이터셋
└── README.md
```

## Source Topics for Initial Migration / 초기 이관 출처
- `Solar_Physics/` — Sweet 1958 (#?), Parker 1957 (#19), Petschek 1964 (#20), Aulanier 1998 (#23), Antiochos 1999 (#25), Kliem-Török 2006 (#26)
- `Living_Reviews_in_Solar_Physics/` — Pontin 2020, Pontin 2022, Webb 2012 (CMEs), Chen 2011 (CMEs), Shibata-Magara 2011 (flares)
- New entries: Carmichael 1964, Sturrock 1966, Hirayama 1974, Kopp & Pneuman 1976, Loureiro+ 2007 plasmoid instability.

Phase B 이관 예정. CME 관측 (Gosling 1993, Howard 2008/STEREO) 은 `Heliosphere_Solar_Wind` 와의 경계가 모호 — Phase B 시작 시 결정.

## Status / 상태
**Active** — Initial scaffolding 2026-05-01.
