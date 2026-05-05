---
title: "Pre-Reading Briefing: The SOHO Mission: An Overview"
paper_id: "08_domingo_1995"
topic: Solar Observation
date: 2026-04-16
type: briefing
---

# The SOHO Mission: An Overview — Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Domingo, V., Fleck, B., and Poland, A.I. (1995). "The SOHO Mission: An Overview." *Solar Physics*, Vol. 162, pp. 1–37.
**Author(s)**: Vicente Domingo (ESA SOHO Project Scientist), Bernhard Fleck (ESA), Arthur I. Poland (NASA GSFC)
**Year**: 1995

---

## 1. 핵심 기여 / Core Contribution

이 논문은 **SOHO(Solar and Heliospheric Observatory)**의 과학적 목표, 기기 구성, 궤도 설계, 운용 개념을 종합적으로 기술합니다. SOHO는 ESA와 NASA의 공동 프로젝트로, 태양-지구 라그랑주 L1 점에 배치된 최초의 포괄적 태양 관측소입니다. 12개의 과학 기기를 탑재하여 태양 내부(일진학), 태양 대기(코로나), 태양풍의 세 가지 핵심 영역을 동시에 연속 관측합니다. SOHO는 1995년 12월 발사 후 30년 가까이 운용되며 태양 물리학의 거의 모든 분야를 변혁했습니다.

This paper comprehensively describes the scientific objectives, instrument complement, orbital design, and operational concept of **SOHO (Solar and Heliospheric Observatory)**. A joint ESA-NASA project, SOHO was the first comprehensive solar observatory placed at the Sun-Earth Lagrange L1 point. It carries 12 scientific instruments covering three key domains simultaneously and continuously: solar interior (helioseismology), solar atmosphere (corona), and solar wind. Launched in December 1995, SOHO has operated for nearly 30 years and transformed virtually every field of solar physics.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

1990년대 초, 태양 물리학은 몇 가지 핵심적 미해결 문제에 직면해 있었습니다:

In the early 1990s, solar physics faced several fundamental unsolved problems:

1. **태양 중성미자 문제 (Solar Neutrino Problem)**: 이론 예측의 1/3만 관측 → 태양 내부 구조 또는 중성미자 물리가 잘못됨. 일진학(helioseismology)으로 태양 내부를 독립적으로 제약할 필요.
   Only 1/3 of predicted neutrinos observed → either solar interior models or neutrino physics wrong. Helioseismology needed to independently constrain solar interior.

2. **코로나 가열 문제 (Coronal Heating Problem)**: 왜 코로나(~10⁶ K)가 광구(~5800 K)보다 200배 뜨거운가? 파동 가열 vs 나노플레어 — 기존 관측으로는 구분 불가.
   Why is the corona (~10⁶ K) 200× hotter than the photosphere (~5800 K)? Wave heating vs nanoflares — existing observations could not distinguish.

3. **태양풍 가속 메커니즘**: Parker (1958)의 이론은 태양풍 존재를 예측했지만, 정확한 가속 메커니즘과 빠른 태양풍(~800 km/s)의 기원은 미해결.
   Parker's theory predicted the solar wind, but the exact acceleration mechanism and origin of fast wind (~800 km/s) remained unknown.

이전 우주 미션(Skylab, SMM, Ulysses)은 특정 영역만 관측했고, 지상 일진학 네트워크(GONG, BiSON — 이 시리즈 #5, #6)는 duty cycle 한계(~80%)에 부딪혔습니다. SOHO는 이 모든 한계를 극복하기 위해 설계되었습니다.

Previous missions (Skylab, SMM, Ulysses) only observed specific domains, and ground-based helioseismology networks (GONG, BiSON — series #5, #6) were limited to ~80% duty cycle. SOHO was designed to overcome all these limitations.

### 타임라인 / Timeline

```
1980  ── SMM (Solar Maximum Mission) — 최초의 태양 전용 우주 관측소
         First dedicated solar space observatory
         │
1982  ── SOHO 개념 제안 (ESA Phase A study)
         SOHO concept proposed
         │
1985  ── ESA Horizon 2000 프로그램에 SOHO 선정
         SOHO selected for Horizon 2000 ("Cornerstone" mission)
         │
1989  ── ESA-NASA 양해각서(MOU) 체결
         ESA-NASA MOU signed (ESA: spacecraft + 9 instruments,
         NASA: launch + 3 instruments + operations)
         │
1990  ── GONG 관측 시작 [이 시리즈 #5]
         │
1992  ── Ulysses 목성 궤도 조우, 태양 극궤도 진입
         │
1994  ── 기기 통합 완료, 시험
         Instrument integration and testing
         │
1995  ── ★ Domingo et al., Solar Physics: SOHO 미션 개요 출판 ★
      │  12월 2일: Atlas II-AS로 SOHO 발사
      │  Dec 2: SOHO launched on Atlas II-AS
         │
1996  ── 2월: L1 헤일로 궤도 도착, 과학 관측 시작
      │  Feb: Arrival at L1 halo orbit, science operations begin
         │
1998  ── 6월: 자세 제어 상실 사고 → 9월 복구 (근접 사망 경험)
         June: Attitude control loss → Sept recovery
         │
2003  ── 마지막 자이로 고장 → 자이로 없는(gyroless) 운용 모드 전환
         Last gyro failure → gyroless operations
         │
2010  ── SDO 발사 — SOHO의 과학적 후계자
         │
2025  ── SOHO 여전히 운용 중 (발사 30주년)
         SOHO still operational (30th anniversary)
```

---

## 3. 필요한 배경 지식 / Prerequisites

### 3.1 라그랑주 L1 점과 헤일로 궤도 / L1 Point and Halo Orbit

태양-지구 라그랑주 L1 점은 지구에서 태양 방향으로 약 150만 km(지구-태양 거리의 ~1%) 떨어진 지점으로, 태양과 지구의 중력이 궤도 원심력과 균형을 이루는 곳입니다. 여기에 위치하면:

The Sun-Earth L1 Lagrange point is ~1.5 million km sunward of Earth (~1% of Sun-Earth distance), where solar and terrestrial gravity balance orbital centrifugal force. Advantages:

- **태양 연속 관측**: 지구 그림자에 들어가지 않음 → 24시간 365일 관측 (duty cycle ~100%)
- **안정된 열 환경**: 태양-우주선 거리가 거의 일정
- **태양풍 직접 측정**: L1은 태양풍이 지구에 도달하기 ~1시간 전에 통과하는 곳

SOHO는 L1 주위를 도는 **헤일로 궤도(halo orbit)**에 배치됩니다. 이것은 불안정한 궤도이므로 주기적인 궤도 유지 기동(station-keeping)이 필요합니다.

SOHO is placed in a **halo orbit** around L1. This is an unstable orbit requiring periodic station-keeping maneuvers.

### 3.2 일진학 기초 / Helioseismology Basics

태양은 수백만 개의 음파(p-모드)로 진동하며, 이 진동의 주파수 패턴이 태양 내부 구조(밀도, 온도, 음속, 회전)를 결정합니다:

The Sun oscillates in millions of acoustic modes (p-modes). The frequency pattern encodes the internal structure (density, temperature, sound speed, rotation):

$$\nu_{n,\ell} \approx \Delta\nu \left(n + \frac{\ell}{2} + \epsilon\right) - \delta\nu_{\ell}$$

- 대간격(large separation) $\Delta\nu \approx 135\,\mu\text{Hz}$: 태양 평균 밀도에 민감
- 소간격(small separation) $\delta\nu$: 태양 핵 구조에 민감

SOHO의 일진학 기기(GOLF, VIRGO, MDI)는 지상 관측의 duty cycle 한계를 극복하여 연속적이고 gap-free한 관측을 제공합니다.

### 3.3 코로나 관측 기법 / Coronal Observation Techniques

코로나는 광구보다 약 100만 배 어두워서, 태양 원반을 차폐(occulting)해야 관측할 수 있습니다:

The corona is ~10⁶ times fainter than the photosphere, requiring occulting of the solar disk:

- **코로나그래프(Coronagraph)**: 태양 원반을 인공적으로 차폐하여 코로나 관측 (SOHO의 LASCO)
- **EUV 영상**: 고온 코로나 플라즈마가 방출하는 극자외선을 직접 촬영 (SOHO의 EIT)
- **분광**: 방출선 프로파일에서 온도, 밀도, 속도 측정 (SOHO의 CDS, SUMER, UVCS)

### 3.4 이전 논문과의 연결 / Connection to Previous Papers

- **#5 Harvey et al. (1996) — GONG**: 지상 일진학 네트워크. SOHO의 MDI/GOLF/VIRGO가 우주에서 이를 보완. GONG의 ~80% duty cycle을 SOHO가 ~100%로 개선.
- **#6 Chaplin et al. (1996) — BiSON**: 저차수 p-모드 관측. SOHO의 GOLF가 L1에서 같은 관측을 연속 수행.
- **#7 Tomczyk et al. (2016) — COSMO**: COSMO는 코로나 자기장 측정을 목표로 하는 지상 기기. SOHO/LASCO의 백색광 코로나 관측과 상보적. K-Cor가 LASCO C2 아래 공백을 메움.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **L1 (First Lagrange Point)** | 태양-지구 사이의 중력 평형점. 태양 연속 관측에 이상적. / Gravitational equilibrium point between Sun and Earth. Ideal for uninterrupted solar observation. |
| **Halo Orbit** | L1 점 주위를 도는 3차원 궤도. SOHO는 6개월 주기, 진폭 ~200,000 km × 600,000 km. / 3D orbit around L1. SOHO's has ~6-month period, amplitude ~200,000 × 600,000 km. |
| **Helioseismology** | 태양 진동(p-모드)을 분석하여 내부 구조를 추론하는 학문. / Study of solar oscillations (p-modes) to infer internal structure. |
| **p-mode** | 압력(pressure) 복원력에 의한 음파 진동. 주기 ~5분, $\ell = 0$~수천. / Acoustic oscillations restored by pressure. Period ~5 min, $\ell = 0$ to thousands. |
| **Coronagraph** | 태양 원반을 인공적으로 차폐하여 코로나를 관측하는 기기. / Instrument that blocks the solar disk to observe the corona. |
| **EUV (Extreme Ultraviolet)** | 파장 10–121 nm의 극자외선. 코로나 온도($10^5$–$10^7$ K) 플라즈마 관측에 핵심. / Wavelength 10–121 nm. Key for observing coronal-temperature plasmas. |
| **In-situ measurement** | 태양풍 입자와 자기장을 우주선 위치에서 직접 측정. / Direct measurement of solar wind particles and fields at the spacecraft location. |
| **CME (Coronal Mass Ejection)** | 코로나에서 대량의 플라즈마와 자기장이 폭발적으로 방출되는 현상. / Explosive release of plasma and magnetic field from the corona. |
| **MDI (Michelson Doppler Imager)** | SOHO의 핵심 일진학 기기. 전일면 도플러 영상으로 태양 진동과 자기장 측정. / Key SOHO helioseismology instrument. Full-disk Doppler imaging for oscillations and magnetic fields. |
| **LASCO (Large Angle Spectroscopic Coronagraph)** | SOHO의 3중 코로나그래프(C1, C2, C3). 1.1–30 $R_\odot$ 범위 코로나 관측. / SOHO's triple coronagraph. Covers 1.1–30 $R_\odot$. |
| **SUMER/CDS/UVCS** | SOHO의 자외선/EUV 분광기 3종. 코로나 플라즈마의 온도, 밀도, 속도 측정. / Three UV/EUV spectrometers measuring coronal plasma temperature, density, velocity. |
| **Telemetry** | 우주선에서 지상으로 과학 데이터를 전송하는 시스템. SOHO: ~200 kbit/s. / System for transmitting science data from spacecraft to ground. SOHO: ~200 kbit/s. |

---

## 5. 수식 미리보기 / Equations Preview

### 5.1 L1 점 위치 / L1 Point Location

$$r_{L1} \approx R \left(\frac{M_\oplus}{3 M_\odot}\right)^{1/3} \approx 1.5 \times 10^6 \text{ km}$$

- $R$: 지구-태양 거리 (~1 AU = 1.496×10⁸ km)
- $M_\oplus / M_\odot \approx 3 \times 10^{-6}$

### 5.2 p-모드 점근 관계 / p-Mode Asymptotic Relation

$$\nu_{n,\ell} \approx \Delta\nu \left(n + \frac{\ell}{2} + \epsilon\right) - D_0 \ell(\ell+1)$$

- $\Delta\nu \approx 135\,\mu\text{Hz}$: 대간격 (태양 반경과 평균 음속에 의존)
- $D_0$: 핵 구조 민감 항
- $n$: 방사 차수(radial order), $\ell$: 각차수(angular degree)

### 5.3 코로나그래프 산란광 수준 / Coronagraph Stray Light Level

우주 코로나그래프의 핵심 장점 — 지구 대기 산란광 제거:

$$\frac{B_\text{sky}}{B_\odot} \sim 10^{-6} \text{ (ground)} \quad \rightarrow \quad \sim 10^{-10} \text{ (space, LASCO C2)}$$

이것이 LASCO가 K-Cor(지상, $\sim 10^{-9}$)보다 넓은 시야에서 더 어두운 코로나를 관측할 수 있는 이유입니다.

### 5.4 Thomson 산란 편광 밝기 / Thomson Scattering pB

$$pB(r) \propto \int n_e(l) \sin^2\chi \, dl$$

LASCO의 백색광 코로나 관측은 이 원리를 이용하여 코로나 전자 밀도를 측정합니다 (이 시리즈 #7에서 K-Cor 맥락으로 이미 구현).

### 5.5 도플러 속도 측정 / Doppler Velocity Measurement

MDI의 Michelson 간섭계 방식:

$$v = \frac{c}{\lambda} \Delta\lambda = \frac{c}{\lambda} \cdot \frac{\delta I}{dI/d\lambda}$$

Ni I 676.8 nm 흡수선의 파장 이동에서 시선 방향 속도를 측정합니다.

---

## 6. 읽기 가이드 / Reading Guide

### 구조 / Structure

이 논문은 37페이지의 긴 미션 개요 논문입니다. 주요 섹션:

This is a 37-page mission overview paper. Main sections:

1. **Introduction (§1)**: 과학적 동기 — 빠르게 읽되, 세 가지 핵심 질문(태양 내부, 코로나 가열, 태양풍)을 기억
2. **Scientific Objectives (§2)**: 핵심 — 각 목표가 어떤 기기에 매핑되는지 주목
3. **The Payload (§3)**: 가장 긴 섹션 — 12개 기기를 세 그룹으로 분류:
   - **일진학**: GOLF, VIRGO, MDI
   - **코로나/태양 대기**: CDS, EIT, LASCO, SUMER, UVCS, SWAN
   - **태양풍 in-situ**: CELIAS, COSTEP, ERNE
4. **Spacecraft (§4)**: 우주선 설계 — pointing 안정성과 열 환경에 주목
5. **Orbit and Operations (§5)**: L1 궤도와 운용 개념

### 읽기 전략 / Reading Strategy

1. **1차 읽기 (30분)**: §1–2를 주의 깊게 읽고, §3는 기기별 Table/Figure를 중심으로 스캔. 각 기기의 "무엇을 측정하는가"에 집중.
   First pass (30 min): Read §1–2 carefully, scan §3 focusing on Tables/Figures. Focus on "what does each instrument measure."

2. **2차 읽기 (20분)**: §3에서 핵심 3기기(MDI, LASCO, EIT)를 상세히 읽기. §4–5는 가볍게.
   Second pass (20 min): Read MDI, LASCO, EIT in detail. Skim §4–5.

3. **주의할 점**: 이 논문은 1995년 **발사 전** 작성되었으므로 계획과 기대치를 기술합니다. 실제 성과는 이후 논문들에서 다뤄집니다.
   Note: Written **before launch** (1995), so describes plans and expectations. Actual results came in later papers.

### 핵심 Figure 목록 / Key Figures

- **Fig. 1**: SOHO 우주선 외관 — 기기 배치 파악
- **Fig. 3–4**: L1 헤일로 궤도 — 궤도 설계 이해
- **Tables**: 각 기기의 사양 요약 — 나중에 참조할 중요한 데이터

---

## 7. 현대적 의의 / Modern Significance

SOHO는 태양 물리학 역사상 가장 성공적인 미션 중 하나이며, 2025년 현재까지도 운용 중입니다:

SOHO is one of the most successful missions in solar physics history, still operational as of 2025:

1. **일진학 혁명**: SOHO/MDI의 연속 관측이 태양 내부 차동 회전(tachocline), 음속 프로파일, 대류대 구조를 밀리헤르츠 정밀도로 밝혀냄 → 태양 중성미자 문제가 태양 모델이 아닌 중성미자 물리의 문제임을 확인 (→ 중성미자 진동 발견, 2002 노벨 물리학상).
   MDI's continuous observations revealed tachocline, sound speed profile, and convection zone structure → confirmed the solar neutrino problem was in neutrino physics, not solar models (→ neutrino oscillation discovery, 2002 Nobel Prize).

2. **CME 과학의 탄생**: LASCO가 20,000개 이상의 CME를 관측하여 CME 통계학, 형태학, 지자기 폭풍 예보의 기초를 수립.
   LASCO observed >20,000 CMEs, establishing CME statistics, morphology, and geomagnetic storm forecasting.

3. **혜성 발견**: LASCO로 4,000개 이상의 혜성을 발견 — 역사상 가장 성공적인 혜성 발견 기기.
   LASCO discovered >4,000 comets — the most prolific comet discoverer in history.

4. **후속 미션의 모범**: SDO(2010), Solar Orbiter(2020), Parker Solar Probe(2018) 등 모든 후속 태양 미션이 SOHO의 과학적 틀 위에 구축됨.
   All subsequent solar missions (SDO, Solar Orbiter, Parker Solar Probe) built on SOHO's scientific framework.

5. **이 시리즈에서의 위치**: SOHO는 지상 관측(#1–#7)에서 우주 관측으로의 전환점. 이후 #9(EIT), #10(LASCO), #11(MDI), #12(AIA), #13(HMI) 등이 SOHO의 개별 기기나 후속 기기를 상세히 다룹니다.
   SOHO marks the transition from ground-based (#1–#7) to space-based observation in this series. Papers #9–#13 cover individual SOHO instruments and successors in detail.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
