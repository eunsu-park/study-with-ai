---
title: "Pre-Reading Briefing: The Solar Dynamics Observatory (SDO)"
paper_id: "35_pesnell_2012"
topic: Solar Observation
date: 2026-04-16
type: briefing
---

# The Solar Dynamics Observatory (SDO) — Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Pesnell, W.D., Thompson, B.J., and Chamberlin, P.C. (2012). "The Solar Dynamics Observatory (SDO)." *Solar Physics*, Vol. 275, pp. 3–15.
**Author(s)**: W. Dean Pesnell (NASA GSFC, SDO Project Scientist), B.J. Thompson (NASA GSFC), P.C. Chamberlin (NASA GSFC)
**Year**: 2012

---

## 1. 핵심 기여 / Core Contribution

이 논문은 NASA의 **SDO(Solar Dynamics Observatory)** 미션의 전체 개요를 기술합니다. SDO는 NASA Living With a Star (LWS) 프로그램의 첫 번째 미션으로, 2010년 2월 11일 Atlas V로 발사되었습니다. 3개의 기기 — **AIA**(EUV/UV 영상), **HMI**(자기장/도플러), **EVE**(EUV 분광 조도) — 를 탑재하여 태양 변동성의 물리적 원인을 이해하고 우주 날씨를 예측하는 것을 목표로 합니다. SDO는 **경사 지구 정지 궤도(inclined geosynchronous orbit)**에 위치하여 전용 지상국(White Sands, NM)을 통해 **~150 Mbit/s**의 데이터를 거의 24시간 연속 전송합니다 — 이것은 SOHO의 40 kbit/s 대비 약 3,750배입니다.

This paper provides the mission-level overview of NASA's **SDO (Solar Dynamics Observatory)**, the first mission of the Living With a Star (LWS) program. Launched on 11 February 2010 on an Atlas V, SDO carries three instruments — **AIA** (EUV/UV imaging), **HMI** (magnetic field/Doppler), and **EVE** (EUV spectral irradiance) — to understand the physical causes of solar variability and improve space weather prediction. SDO is in an **inclined geosynchronous orbit**, transmitting ~150 Mbit/s nearly 24/7 via a dedicated ground station at White Sands, NM — approximately 3,750× the SOHO data rate.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

2000년대 초, SOHO(1995~)와 TRACE(1998~)가 태양 물리학을 변혁했지만 한계가 분명했습니다:

- **SOHO**: L1에서 연속 관측이 가능하지만, EIT의 2.6″ 분해능과 ~12분 케이던스가 느림. 텔레메트리 40 kbit/s로 데이터 제한.
- **TRACE**: 0.5″ 고분해능이지만 8.5′ FOV(태양의 1/4)만 관측. LEO에서 3개월/년 일식 시즌.
- **자기장**: SOHO/MDI는 광구 자기장을 측정했지만, 분해능과 케이던스가 코로나 변동을 따라가기에 불충분.

SDO는 이 모든 한계를 동시에 극복하도록 설계되었습니다: 전일면 + 고분해능 + 고케이던스 + 대용량 텔레메트리.

### SOHO vs SDO 비교 / SOHO vs SDO Comparison

| 항목 | SOHO (1995) | SDO (2010) |
|------|-------------|------------|
| 궤도 | L1 (150만 km) | GEO-sync (36,000 km) |
| 기기 수 | 12 | 3 |
| EUV 영상 | EIT (1024², 2.6″, 4밴드, 12분) | AIA (4096², 0.6″, 10채널, 12초) |
| 자기장 | MDI (1024², 4″, ~60분) | HMI (4096², 1″, 45초) |
| 텔레메트리 | 40 kbit/s | ~150 Mbit/s |
| 일일 데이터 | ~3.5 Gbit | ~1.5 TB |
| 데이터 비율 | ×1 | ×3,750 |

### 타임라인 / Timeline

```
1995  ── SOHO 발사 [#8] — L1 최초 태양 관측소
         │
1998  ── TRACE 발사 [#11] — 고분해능 EUV
         │
2001  ── LWS 프로그램 시작 — SDO 선정
         │
2004  ── SDO Phase B (정의 단계)
         │
2008  ── 기기 통합 및 시험 완료
         │
2010  ── ★ SDO 발사 (Feb 11, Atlas V, Cape Canaveral) ★
      │  GEO-sync 궤도 진입, 첫 빛 (Mar 27)
         │
2012  ── ★ Pesnell et al.: SDO 미션 개요 출판 ★
      │  AIA (#12), HMI (#13), EVE 기기 논문 동시 출판
         │
2025  ── SDO 15년째 운용 중
```

---

## 3. 필요한 배경 지식 / Prerequisites

### 3.1 GEO-synchronous 궤도 vs L1 / GEO vs L1

| 특성 | L1 (SOHO) | GEO-sync (SDO) |
|------|-----------|----------------|
| 거리 | ~1.5×10⁶ km | ~36,000 km |
| 데이터율 | ~40 kbit/s (DSN) | ~150 Mbit/s (전용 지상국) |
| 관측 연속성 | ~100% (일식 없음) | ~95% (일식 시즌 제외) |
| 지상국 | DSN 공유 (3곳 분산) | White Sands 전용 (2곳) |
| 신호 지연 | ~5초 편도 | ~0.12초 편도 |

SDO가 GEO를 선택한 핵심 이유: **데이터율**. AIA+HMI+EVE가 생성하는 ~150 Mbit/s를 L1에서는 전송할 수 없습니다.

### 3.2 Living With a Star (LWS) 프로그램

NASA의 LWS 프로그램은 태양-태양권-지구 연결을 이해하여 우주 날씨의 사회적 영향을 경감하는 것을 목표로 합니다. SDO는 LWS의 첫 번째 미션이며, 이후 Van Allen Probes(2012), IRIS(2013) 등이 뒤따랐습니다.

### 3.3 이전 논문과의 연결 / Connection to Previous Papers

- **#8 Domingo (SOHO)**: SDO는 SOHO의 직접적 후계자. SOHO의 12개 기기를 3개로 통합하되, 각각이 훨씬 강력.
- **#12 Lemen (AIA)**: 이미 AIA 기기를 상세히 학습함. SDO 논문은 미션 맥락 제공.
- **#11 Handy (TRACE)**: TRACE의 기술 유산이 AIA에 직접 계승.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **LWS (Living With a Star)** | NASA의 태양-태양권 연구 프로그램. SDO는 LWS 첫 번째 미션. / NASA's heliophysics program. SDO is LWS-1. |
| **GEO-synchronous orbit** | 지구 자전 주기와 같은 궤도. SDO는 경도 102°W, 경사각 28.5°. / Orbit with Earth's rotation period. SDO at 102°W, 28.5° inclination. |
| **Ka-band** | 26.5-40 GHz 주파수 대역. SDO의 고속 다운링크에 사용. / High-frequency band for SDO's high-rate downlink. |
| **EVE (EUV Variability Experiment)** | SDO의 3번째 기기. 태양 EUV 분광 조도를 측정. / SDO's third instrument measuring EUV spectral irradiance. |
| **JSOC (Joint SDO Operations Center)** | Stanford 대학의 SDO 데이터 처리/아카이브 센터. / SDO data processing center at Stanford. |
| **Eclipse season** | SDO가 지구 그림자에 들어가는 기간. 춘분/추분 근처 ~3주씩, 연 2회. / Period when SDO enters Earth's shadow. ~3 weeks near equinoxes, twice yearly. |

---

## 5. 수식 미리보기 / Equations Preview

### 5.1 GEO 궤도 고도 / GEO Orbital Altitude

$$r_{\text{GEO}} = \left(\frac{GM_\oplus T^2}{4\pi^2}\right)^{1/3} \approx 42,164 \text{ km (중심에서)}$$

고도: $h = r_{\text{GEO}} - R_\oplus \approx 35,786$ km

### 5.2 데이터율 비교 / Data Rate Comparison

$$\frac{\text{SDO}}{\text{SOHO}} = \frac{150 \text{ Mbit/s}}{0.04 \text{ Mbit/s}} = 3,750\times$$

### 5.3 일일 데이터 볼륨 / Daily Data Volume

$$V = 150 \text{ Mbit/s} \times 86400 \text{ s/day} \times 0.95 = 12.3 \text{ Tbit/day} \approx 1.5 \text{ TB/day}$$

---

## 6. 읽기 가이드 / Reading Guide

이 논문은 13페이지의 비교적 짧은 미션 개요입니다:

1. **§1 Introduction**: SDO/LWS 미션 목표
2. **§2 Instrument Suite**: AIA, HMI, EVE 각각의 역할 (상세는 개별 논문에)
3. **§3 Spacecraft**: 궤도 설계, GEO의 장단점, 전력/열/자세 제어
4. **§4 Ground System**: White Sands 지상국, Ka-band 다운링크, JSOC 아카이브
5. **§5 Operations**: 관측 모드, 일식 시즌 대응

**읽기 전략**: SOHO(#8)와 비교하며 읽기 — 같은 과학 목표를 15년 후의 기술로 어떻게 다르게 달성하는지에 주목.

---

## 7. 현대적 의의 / Modern Significance

SDO는 태양 물리학의 "빅데이터 시대"를 열었습니다:

1. **~1.5 TB/day**: 발사 당시 NASA 최대 데이터율 미션. 15년간 누적 ~7 PB.
2. **머신러닝 혁명**: AIA의 방대한 데이터셋이 태양 물리 ML/AI 연구의 기반.
3. **실시간 우주 날씨**: AIA/HMI 데이터가 NOAA SWPC의 플레어/CME 예보에 핵심 입력.
4. **시민 과학**: Helioviewer.org 등을 통해 공개 데이터에 전 세계가 접근.
5. **15년+ 운용**: 태양 주기 24 전체 + 주기 25 상승기 커버.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
