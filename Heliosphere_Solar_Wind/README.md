# Heliosphere & Solar Wind / 헬리오스피어 및 태양풍

## Overview / 개요
A study track on the **solar wind, the heliosphere, and the in-situ measurements that probe them** — from Biermann's comet-tail evidence (1951) and Parker's prediction of supersonic outflow (1958), through Helios, Ulysses, Wind, ACE, and STEREO, to the close-in revolution with **Parker Solar Probe** and **Solar Orbiter**. Covers fast/slow wind acceleration mechanisms, magnetic switchbacks, heliospheric current sheet, co-rotating interaction regions, the heliopause and termination shock crossings by Voyager 1 & 2, and the extension to stellar winds and astrospheres.

**태양풍, 헬리오스피어, 그리고 이를 탐사하는 in-situ 관측**을 학습하는 트랙. Biermann 의 혜성 꼬리 증거(1951), Parker 의 초음속 유출 예측(1958)부터 Helios·Ulysses·Wind·ACE·STEREO, 그리고 **Parker Solar Probe**·**Solar Orbiter** 의 근접 관측 혁명까지. fast/slow 태양풍 가속 메커니즘, magnetic switchback, 헬리오스피어 전류시트, co-rotating interaction region (CIR), Voyager 1·2 의 heliopause·termination shock 통과, 항성풍·astrosphere 까지 확장.

## Learning Roadmap / 학습 로드맵

### Phase 1: Solar Wind Discovery & Theory / 태양풍 발견과 이론 (1951–1962)
- Biermann comet tails (1951) / Biermann 혜성 꼬리
- Parker's hydrodynamic solar wind (1958) / Parker 유체역학적 태양풍
- Mariner 2 confirmation (Neugebauer & Snyder 1962) / Mariner 2 확인

### Phase 2: Solar Wind Structure & Composition / 태양풍 구조와 조성 (1970s–2000s)
- Helios 1/2 inner heliosphere / Helios 1/2 내부 헬리오스피어
- Ulysses high-latitude polar wind / Ulysses 고위도 극풍
- FIP effect, charge state diagnostics / FIP 효과, charge state 진단
- Co-rotating interaction regions (CIRs) / Co-rotating interaction region

### Phase 3: Heliosphere Boundaries — Termination Shock & Heliopause / 헬리오스피어 경계 — termination shock 및 heliopause (1990s–2014)
- Voyager 1 & 2 outer heliosphere / Voyager 1·2 외부 헬리오스피어
- IBEX ENA mapping / IBEX ENA 매핑
- Termination shock and heliopause crossings / Termination shock·heliopause 통과
- Ribbon and global heliosphere shape / Ribbon과 헬리오스피어 형태

### Phase 4: Modern Era — PSP & Solar Orbiter / 현대 — PSP·Solar Orbiter (2018–Present)
- Magnetic switchbacks (PSP discovery) / 자기 switchback
- Alfvén critical surface / Alfvén 임계면
- Sub-Alfvénic wind sampling / Sub-Alfvénic 태양풍 샘플링
- Combined remote-sensing and in-situ science / 원격관측·in-situ 결합

### Phase 5: Stellar Winds & Astrospheres / 항성풍 및 astrosphere (2015–Present)
- Stellar mass-loss measurements / 항성 질량 손실 측정
- Astrosphere observations of nearby stars / 근접 항성의 astrosphere 관측
- Exoplanet space weather / 외계행성 우주기상

## Directory Structure / 디렉토리 구조
```
Heliosphere_Solar_Wind/
├── papers/          # Reading list and per-paper notes / 리딩 리스트 및 논문별 노트
├── notes/           # Theory notes (Parker equations, Alfvén surface, ENA imaging) / 이론 노트
├── notebooks/       # Solar wind data analysis (PSP/SWEAP, Wind/3DP) / 태양풍 데이터 분석
├── scripts/         # Utilities (CDAweb fetchers, OMNI parser) / 유틸리티
├── data/            # Sample datasets / 샘플 데이터셋
└── README.md
```

## Source Topics for Initial Migration / 초기 이관 출처
- `Solar_Physics/` — Biermann 1951 (#11), Parker 1958 (#12), Gosling 1993 (#22), Kasper 2021 PSP switchbacks (#38)
- `Space_Weather/` — solar wind structure papers; PSP & Solar Orbiter mission overviews; Voyager outer heliosphere
- `Living_Reviews_in_Solar_Physics/` — Bruno 2013 (turbulence), Marsch 2006 (kinetic), Verscharen 2019 (kinetic), Vidotto 2021 (stellar winds), Sheeley 2005 (Lockwood structures)
- `Solar_Observation/` — PSP/FIELDS, PSP/SWEAP, PSP/IS☉IS, PSP/WISPR, Solar Orbiter (cross-reference)

Phase B 이관 예정.

## Status / 상태
**Active** — Initial scaffolding 2026-05-01.
