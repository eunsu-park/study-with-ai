---
title: "Pre-Reading Briefing: The Transition Region and Coronal Explorer"
paper_id: "11_handy_1999"
topic: Solar Observation
date: 2026-04-16
type: briefing
---

# The Transition Region and Coronal Explorer (TRACE) — Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Handy, B.M., et al. (1999). "The Transition Region and Coronal Explorer." *Solar Physics*, Vol. 187, pp. 229–260.
**Author(s)**: Barry M. Handy (Lockheed Martin) + multiple co-authors
**Year**: 1999

---

## 1. 핵심 기여 / Core Contribution

이 논문은 NASA SMEX(Small Explorer) 미션인 **TRACE(Transition Region and Coronal Explorer)**의 설계, 성능, 초기 관측 결과를 기술합니다. TRACE는 **0.5″ 공간 분해능**으로 EUV/UV 영상을 촬영하는 소형 위성으로, EIT(#9)와 같은 다층 코팅 수직 입사 광학계를 사용하지만 공간 분해능이 ~5배 높습니다. 30 cm Cassegrain 망원경에 1024×1024 CCD를 탑재하여 171 Å, 195 Å, 284 Å (EUV) 및 1216 Å, 1550 Å, 1600 Å, 1700 Å (UV), white light 채널을 제공합니다. TRACE는 코로나 루프의 미세 구조, "moss" 방출, 코로나 비(coronal rain), 나노플레어 등을 최초로 시각화하여 코로나 가열 연구를 변혁했습니다.

This paper describes the design, performance, and early results of **TRACE (Transition Region and Coronal Explorer)**, a NASA SMEX mission. TRACE achieves **0.5″ spatial resolution** EUV/UV imaging using a 30-cm Cassegrain telescope with normal-incidence multilayer-coated optics, similar to EIT (#9) but with ~5× higher spatial resolution. It provides channels at 171 Å, 195 Å, 284 Å (EUV), 1216 Å, 1550 Å, 1600 Å, 1700 Å (UV), and white light. TRACE was the first to visualize coronal loop fine structure, "moss" emission, coronal rain, and nanoflares, transforming coronal heating research.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

1990년대 중반, SOHO/EIT(#9)가 4개 EUV 밴드로 태양 전면 영상을 제공했지만, 2.6″/pixel의 공간 분해능으로는 코로나의 미세 구조를 분해할 수 없었습니다. 코로나 가열 문제의 핵심 — 에너지가 어디서, 어떻게 방출되는지 — 은 개별 코로나 루프와 그 내부 구조를 관측해야 답할 수 있었습니다.

In the mid-1990s, SOHO/EIT provided full-disk EUV images at 2.6″/pixel, insufficient to resolve coronal fine structure. The core of the coronal heating problem — where and how energy is released — required observing individual coronal loops and their internal structure.

TRACE는 "작지만 날카로운" 접근: 전일면 대신 **8.5′×8.5′ FOV** (태양 직경의 ~1/4)에 집중하되, 분해능을 0.5″까지 높여 코로나 물리학의 새로운 체제(regime)를 열었습니다.

TRACE took the "small but sharp" approach: instead of full-disk coverage, it focused on an **8.5′×8.5′ FOV** (~1/4 of solar diameter) with 0.5″ resolution, opening a new regime of coronal physics.

### 타임라인 / Timeline

```
1991  ── Yohkoh/SXT — 연 X선 코로나 영상 (~5″)
         │
1995  ── SOHO/EIT [#9] — EUV 전일면 영상 (2.6″)
      │  SOHO/LASCO [#10] — 코로나그래프
         │
1998  ── ★ TRACE 발사 (1998년 4월 2일, Pegasus XL) ★
      │  Sun-synchronous LEO (600-650 km, 97.8° inclination)
         │
1999  ── ★ Handy et al.: TRACE 기기 논문 출판 ★
      │  코로나 루프 미세 구조 최초 관측
      │  "Moss" 방출 발견 (Berger et al., 1999)
         │
2003  ── TRACE + RHESSI 공동 플레어 관측
         │
2010  ── SDO/AIA [#12] — TRACE 분해능 + EIT FOV 결합 (후계자)
         │  TRACE 운용 종료
```

---

## 3. 필요한 배경 지식 / Prerequisites

### 3.1 EIT와의 핵심 차이 / Key Differences from EIT

| 항목 | EIT (#9) | TRACE |
|------|----------|-------|
| 구경 / Aperture | 12 cm | 30 cm |
| 초점 거리 / Focal length | 165 cm | 850 cm |
| 픽셀 / Pixel | 2.6″ (21 μm) | 0.5″ (21 μm) |
| FOV | 45′ (전일면) | 8.5′ (부분) |
| CCD | 1024² 후면 조사 | 1024² 후면 조사 (lumogen 코팅) |
| EUV 밴드 | 171, 195, 284, 304 | 171, 195, 284 |
| UV/가시광 | 없음 | 1216, 1550, 1600, 1700, WL |
| 거울 설계 | 4분할 | 4분할 (동일 개념) |

TRACE는 EIT의 4분할 거울 개념을 계승하되, 구경을 2.5배 키워 분해능을 5배 높이고, UV 채널을 추가했습니다.

### 3.2 코로나 루프 물리학 / Coronal Loop Physics

코로나는 자기장에 의해 구조화되며, 기본 단위는 **코로나 루프(coronal loop)**입니다. 루프는 광구의 반대 극성 자기장 사이를 연결하는 아치형 플라즈마 관입니다. 코로나 가열 문제의 핵심은:
- 루프의 온도 구조: 균일한가, 다중 열적 가닥(strand)인가?
- 가열은 루프 꼭대기인가, 발(footpoint)인가?
- 정상 상태인가, 나노플레어(간헐적)인가?

TRACE가 0.5″로 루프를 분해함으로써 이 질문들에 직접 답할 수 있게 되었습니다.

### 3.3 이전 논문과의 연결 / Connection to Previous Papers

- **#9 EIT**: TRACE의 직접적 기술 선조. 같은 4분할 다층 코팅 개념, 같은 3개 EUV 밴드 (171/195/284).
- **#8 SOHO**: TRACE는 SOHO와 동시 운용되어 상보적: SOHO/EIT가 전체 맥락, TRACE가 선택 영역 고분해능.
- **#10 LASCO**: TRACE는 디스크/저코로나 관측, LASCO는 1.1+ R☉ 코로나.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **SMEX (Small Explorer)** | NASA의 소형 우주과학 미션 프로그램. 빠른 개발, 낮은 비용. TRACE는 SMEX-4. / NASA's small space science mission program. |
| **Cassegrain telescope** | 주경+부경 반사 망원경. TRACE는 30cm 구경 Cassegrain. / Primary+secondary mirror reflector. |
| **Coronal loop** | 광구 자기장 사이를 연결하는 아치형 플라즈마 구조. 코로나의 기본 구성 단위. / Arch-shaped plasma structure connecting photospheric magnetic fields. |
| **Moss emission** | 활동 영역 하부에서 관측되는 미세한 밝은 점 구조. TRACE가 최초 발견. / Fine bright point structures at active region bases. First discovered by TRACE. |
| **Nanoflare** | 코로나를 가열하는 것으로 제안된 극히 작은 에너지 방출 사건 ($\sim 10^{24}$ erg). / Very small energy release events proposed to heat the corona. |
| **Lumogen coating** | CCD 표면에 도포된 형광 물질. UV 광자를 가시광으로 변환하여 QE 향상. / Fluorescent material on CCD surface converting UV to visible photons. |
| **Sun-synchronous orbit** | 태양과 일정한 각도를 유지하는 극궤도. ~8개월/년 연속 관측 가능. / Polar orbit maintaining constant angle with Sun. ~8 months/year continuous observation. |
| **Pixel-limited resolution** | 광학 분해능이 픽셀 크기보다 좋아서, 분해능이 픽셀에 의해 결정되는 상태. / When optical resolution exceeds pixel size, resolution is set by the pixel. |

---

## 5. 수식 미리보기 / Equations Preview

### 5.1 공간 분해능 / Spatial Resolution

$$\theta_{\text{pixel}} = \frac{21\,\mu\text{m}}{8500\,\text{mm}} = 0.509'' \approx 0.5''$$

EIT 대비: $2.6''/0.5'' = 5.2\times$ 향상

### 5.2 회절 한계 / Diffraction Limit

$$\theta_{\text{diff}} = 1.22 \frac{\lambda}{D} = 1.22 \times \frac{171\,\text{\AA}}{30\,\text{cm}} = 0.14''$$

TRACE의 회절 한계(0.14″)는 픽셀 크기(0.5″)보다 훨씬 작으므로, 분해능은 픽셀에 의해 결정됩니다.

### 5.3 FOV 계산

$$\text{FOV} = 1024 \times 0.5'' = 512'' \approx 8.5'$$

태양 직경(~32′ = 1920″)의 약 27%.

---

## 6. 읽기 가이드 / Reading Guide

### 구조 / Structure

약 32페이지의 기기+초기 결과 논문:

1. **Introduction (§1)**: 과학 목표, SMEX 프로그램 맥락
2. **Instrument Description (§2)**: 망원경, 다층 코팅, CCD, 필터, 궤도
3. **Performance (§3)**: 분해능, 감도, 케이던스, 온도 응답
4. **Early Results (§4)**: 코로나 루프, moss, 동적 현상 — 이 섹션이 과학적 핵심
5. **Operations (§5)**: 관측 모드, 데이터 처리

### 읽기 전략 / Reading Strategy

1. **§2 기기 설계**: EIT와의 차이점에 주목 — 같은 4분할 개념이지만 구경/초점 거리가 다름
2. **§4 초기 결과**: 가장 중요한 섹션 — TRACE가 어떤 새로운 물리를 보여주었는지
3. **§3 성능**: 0.5″ 분해능이 실제로 달성되었는지 확인

---

## 7. 현대적 의의 / Modern Significance

TRACE는 "고분해능 EUV 영상"의 시대를 열었습니다:

1. **코로나 루프 미세 구조**: 개별 루프가 ~1″ 폭의 가는 가닥으로 구성됨을 최초 관측 → 다중 열적 가닥(multi-thermal strand) 모델의 관측적 기초.
2. **Moss 발견**: 활동 영역 하부의 밝은 미세 구조 → 고온 루프의 발(footpoint)에서의 전도 가열 증거.
3. **SDO/AIA의 직접적 선조**: AIA는 TRACE의 분해능(0.6″)과 EIT의 FOV(전일면)를 결합한 기기. TRACE가 없었다면 AIA의 과학적 설계가 불가능.
4. **코로나 가열 논쟁의 관측적 전환점**: 정상 상태 가열 vs 나노플레어 간헐적 가열 논쟁에 결정적 관측 증거 제공.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
