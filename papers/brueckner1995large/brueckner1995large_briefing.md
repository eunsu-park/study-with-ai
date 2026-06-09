---
title: "Pre-Reading Briefing: The Large Angle Spectroscopic Coronagraph (LASCO)"
paper_id: "10_brueckner_1995"
topic: Solar Observation
date: 2026-04-16
type: briefing
---

# The Large Angle Spectroscopic Coronagraph (LASCO) — Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Brueckner, G.E., Howard, R.A., Koomen, M.J., et al. (1995). "The Large Angle Spectroscopic Coronagraph (LASCO)." *Solar Physics*, Vol. 162, pp. 357–402.
**Author(s)**: G.E. Brueckner (PI, Naval Research Laboratory) + 29 co-authors
**Year**: 1995

---

## 1. 핵심 기여 / Core Contribution

이 논문은 SOHO에 탑재된 **LASCO(Large Angle Spectroscopic Coronagraph)**의 설계, 광학 원리, 기기 사양, 운용 계획을 상세히 기술합니다. LASCO는 세 개의 코로나그래프(C1, C2, C3)로 구성되어 태양 반경 1.1–30 $R_\odot$ 범위의 코로나를 연속 관측하는 시스템입니다. C1은 내부 차폐(internally occulted) 반사 코로나그래프로 Fabry-Perot 분광 기능을 갖추고 있고, C2와 C3는 외부 차폐(externally occulted) 코로나그래프입니다. LASCO는 우주에서 운용된 최초의 백색광 코로나그래프로서, 40,000개 이상의 CME를 발견하고 4,000개 이상의 혜성을 발견하는 등 태양-태양권 물리학을 변혁했습니다.

This paper describes the design, optical principles, instrument specifications, and operational plans of **LASCO (Large Angle Spectroscopic Coronagraph)** on SOHO. LASCO consists of three coronagraphs (C1, C2, C3) providing continuous coverage of the corona from 1.1 to 30 $R_\odot$. C1 is an internally occulted reflecting coronagraph with Fabry-Perot spectroscopic capability, while C2 and C3 are externally occulted coronagraphs. As the first white-light coronagraphs operated from space with modern CCD detectors, LASCO has discovered over 40,000 CMEs and 4,000+ comets, transforming solar-heliospheric physics.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

코로나그래프는 1930년 Bernard Lyot가 발명한 이후 태양 물리학의 핵심 도구입니다. 태양 원반(corona의 ~10⁶배 밝음)을 인공적으로 차폐하여 희미한 코로나를 관측합니다. 지상 코로나그래프는 지구 대기의 산란광(~10⁻⁶ $B_\odot$)에 의해 제한되지만, 우주에서는 이 한계가 극적으로 개선됩니다.

The coronagraph, invented by Bernard Lyot in 1930, has been a key tool in solar physics. It artificially blocks the solar disk (~10⁶× brighter than corona) to observe the faint corona. Ground-based coronagraphs are limited by atmospheric scattered light (~10⁻⁶ $B_\odot$), but this limit improves dramatically from space.

LASCO 이전의 우주 코로나그래프:
- **Skylab ATM (1973-74)**: 최초의 우주 백색광 코로나그래프. 사진 필름 사용, 제한된 관측 기간.
- **SMM/Coronagraph (1980-89)**: 전자 검출기 사용, 하지만 LEO의 지구 식(eclipse) 간섭.
- **P78-1/Solwind (1979-85)**: 미 공군 위성의 코로나그래프.

이들은 모두 LEO에서 운용되어 지구 식에 의한 관측 중단이 빈번했습니다. LASCO는 L1에서 운용되어 이 문제를 해결합니다.

### 타임라인 / Timeline

```
1930  ── Lyot: 최초의 코로나그래프 발명 (Pic du Midi)
         │
1973  ── Skylab/ATM: 최초의 우주 코로나그래프 (사진 필름)
         │  최초의 우주 CME 관측
         │
1979  ── P78-1/Solwind: 최초의 전자 검출기 우주 코로나그래프
         │
1980  ── SMM/C-P: Solar Maximum Mission 코로나그래프 (LEO)
         │  HAO Mk-III K-coronameter (Mauna Loa)
         │
1988  ── SOHO 기기 선정 — LASCO 선정 (PI: Brueckner, NRL)
         │
1993  ── K-Cor 개념 개발 시작
         │
1995  ── ★ Brueckner et al.: LASCO 기기 논문 출판 ★
      │  12월: SOHO 발사
         │
1996  ── LASCO 첫 빛 — 최초의 L1 코로나그래프 영상
      │  CME 일상 관측 시작
         │
1998  ── SOHO 자세 상실/복구 — LASCO C1 이후 재가동 불가
         │  (C2, C3는 정상 운용 계속)
         │
2006  ── STEREO/COR1+COR2: 다시점 코로나그래프
         │
2013  ── K-Cor (Mauna Loa) 배치 [이 시리즈 #7]
         │
2025  ── LASCO C2/C3 여전히 운용 중 (30년째)
         │  PUNCH, PROBA-3 등 차세대 코로나그래프 계획
```

---

## 3. 필요한 배경 지식 / Prerequisites

### 3.1 코로나그래프 차폐 방식 / Coronagraph Occulting Methods

코로나그래프의 핵심은 태양 원반의 직접광을 차폐하는 것입니다. 두 가지 주요 방식:

**내부 차폐 (Internally occulted)** — LASCO C1:
- 대물렌즈/거울이 태양 전체 빛을 받은 후, 초점면에서 태양 상(image)을 차폐
- 장점: 차폐 경계가 날카로움 → 림에 매우 가까운 관측 가능 (C1: 1.1 $R_\odot$)
- 단점: 대물렌즈에서 산란광 발생 → 더 높은 산란광 수준

**외부 차폐 (Externally occulted)** — LASCO C2, C3:
- 대물렌즈 앞에 차폐 원판(occulting disk)을 배치하여 직접광 자체를 차단
- 장점: 산란광이 극적으로 낮음 (C2: ~10⁻¹⁰ $B_\odot$)
- 단점: 차폐 경계가 덜 날카로움 → 림에서 더 먼 곳부터만 관측 가능 (C2: 1.5 $R_\odot$)

### 3.2 Thomson 산란과 편광 밝기 / Thomson Scattering and pB

코로나의 백색광은 자유 전자에 의한 Thomson 산란입니다:

$$pB(r) \propto \int n_e(l) \sin^2\chi \, dl$$

- $pB$: 편광 밝기 (polarization brightness) — 전자 기둥 밀도(column density)에 비례
- $n_e$: 전자 밀도
- $\chi$: 산란각

이 원리는 #7(COSMO/K-Cor)에서 이미 구현했습니다. LASCO C2/C3는 이 원리로 코로나 전자 밀도를 측정합니다.

### 3.3 Fabry-Perot 간섭계 / Fabry-Perot Interferometer

LASCO C1의 고유한 특징은 **Fabry-Perot 에탈론**을 내장한 분광 코로나그래프라는 점입니다:

$$T(\lambda) = \frac{1}{1 + F \sin^2\left(\frac{2\pi n d \cos\theta}{\lambda}\right)}$$

- $F = 4R/(1-R)^2$: finesse coefficient
- $d$: 에탈론 간격 (gap)
- $n$: 굴절률
- 분광 분해능: $\lambda/\Delta\lambda \approx 700$ mÅ at 6000 Å

이것으로 Fe XIV 5303 Å, Fe X 6374 Å 등 코로나 금지선의 강도, 도플러 이동, 선폭을 측정할 수 있습니다.

### 3.4 이전 논문과의 연결 / Connection to Previous Papers

- **#7 Tomczyk (COSMO)**: K-Cor가 LASCO C2 아래 공백(1.05–1.5 $R_\odot$)을 메움. LASCO C1과 유사한 내부 차폐 방식.
- **#8 Domingo (SOHO)**: LASCO는 SOHO 12개 기기 중 하나. EIT와 전자장비(LEB) 공유.
- **#9 Delaboudinière (EIT)**: EIT와 LASCO가 LEB를 공유 (합산 5.2 kbit/s).

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **Internally occulted** | 초점면에서 태양 상을 차폐. 림 가까이 관측 가능 (1.1 $R_\odot$). C1 방식. / Blocking solar image at focal plane. Can observe close to limb. |
| **Externally occulted** | 대물렌즈 앞에서 직접광 차단. 매우 낮은 산란광. C2, C3 방식. / Blocking direct light before objective. Very low stray light. |
| **Occulting disk** | 태양 직접광을 차단하는 원형 차폐판. / Circular disk that blocks direct sunlight. |
| **Lyot stop** | 회절광을 추가로 차단하는 구경. / Aperture that blocks additional diffracted light. |
| **pB (polarization brightness)** | Thomson 산란의 편광 성분. 전자 밀도에 비례. / Polarized component of Thomson scattering. Proportional to electron density. |
| **F-corona** | 행성간 먼지에 의한 산란광. K-corona와 분리 필요. / Scattered light from interplanetary dust. Must be separated from K-corona. |
| **K-corona** | 자유 전자에 의한 Thomson 산란 코로나. 과학적 관심 대상. / Thomson-scattered corona from free electrons. The scientific target. |
| **Fabry-Perot etalon** | 두 평행 반사면 사이의 다중 반사로 좁은 파장 선택. C1의 분광 장치. / Multiple reflections between parallel surfaces for narrow wavelength selection. C1's spectrometer. |
| **Vignetting** | 시야각 주변부에서 빛이 부분적으로 차단되어 밝기가 감소하는 현상. / Partial light blockage at field edges causing brightness reduction. |
| **CME (Coronal Mass Ejection)** | 코로나에서 대량의 플라즈마와 자기장이 폭발적으로 방출. LASCO의 핵심 관측 대상. / Explosive release of plasma and magnetic field from the corona. LASCO's primary target. |
| **Halo CME** | 태양-지구 방향으로 전파되는 CME. LASCO에서 태양 주위를 둘러싸는 밝은 고리로 관측. / CME propagating toward/away from Earth. Appears as bright ring around the Sun in LASCO. |

---

## 5. 수식 미리보기 / Equations Preview

### 5.1 Thomson 산란 편광 밝기 / Thomson Scattering pB

$$pB = \frac{\pi \sigma_T \bar{B}_\odot}{2} \int n_e \sin^2\chi \, dl$$

- $\sigma_T = 6.65 \times 10^{-29}$ m²: Thomson 산란 단면적
- 이미 #7에서 구현. LASCO는 이것을 1.5–30 $R_\odot$ 범위에서 적용.

### 5.2 Fabry-Perot 투과 함수 / Fabry-Perot Transmission

$$T(\lambda) = \frac{1}{1 + F \sin^2\left(\frac{2\pi n d}{\lambda}\right)}, \quad F = \frac{4R}{(1-R)^2}$$

- C1의 분광 분해능: ~700 mÅ at ~5300 Å → $\lambda/\Delta\lambda \approx 7500$

### 5.3 코로나그래프 산란광 수준 / Coronagraph Stray Light

외부 차폐 코로나그래프의 산란광 $B_{stray}$는 주로 회절에 의해 결정:

$$B_{stray} \approx \left(\frac{\lambda}{D}\right)^2 B_\odot$$

C2 ($D = 20$ cm): $B_{stray} \sim 10^{-10} B_\odot$ at 2.5 $R_\odot$

### 5.4 FOV와 공간 분해능 / FOV and Spatial Resolution

| | C1 | C2 | C3 |
|---|---|---|---|
| FOV (inner) | 1.1 $R_\odot$ | 1.5 $R_\odot$ | 3.7 $R_\odot$ |
| FOV (outer) | 3.0 $R_\odot$ | 6.0 $R_\odot$ | 30 $R_\odot$ |
| Pixel | 5.6″ | 11.4″ | 56″ |
| Detector | 1024×1024 | 1024×1024 | 1024×1024 |

---

## 6. 읽기 가이드 / Reading Guide

### 구조 / Structure

이 논문은 46페이지의 상세한 기기 논문입니다:

1. **Introduction (§1)**: 코로나그래프 역사와 LASCO 과학 목표
2. **C1 Coronagraph (§2)**: 내부 차폐 반사 코로나그래프 + Fabry-Perot 분광
3. **C2 Coronagraph (§3)**: 외부 차폐 코로나그래프 (주력 CME 모니터)
4. **C3 Coronagraph (§4)**: 광각 외부 차폐 코로나그래프
5. **Camera and Electronics (§5)**: CCD, LEB, 데이터 처리
6. **Operations (§6)**: 관측 모드, 텔레메트리, 데이터 압축
7. **Calibration (§7)**: 지상 교정

### 읽기 전략 / Reading Strategy

1. **핵심 집중**: §2(C1)과 §3(C2)가 가장 중요. C1의 Fabry-Perot 분광 기능과 C2의 외부 차폐 설계에 집중.
2. **비교하며 읽기**: C1/C2/C3의 차이점 — 차폐 방식, FOV, 분해능, 산란광 수준.
3. **§5 카메라**: EIT와 공유하는 LEB 구조 이해.
4. **§6 데이터 압축**: EIT 논문(#9)에서 나온 ADCT/Rice 압축이 LASCO에도 동일 적용.

---

## 7. 현대적 의의 / Modern Significance

LASCO는 우주 코로나그래프의 황금 표준(gold standard)입니다:

1. **CME 과학의 기초**: 40,000+ CME 카탈로그 (CDAW catalog)가 CME 통계학, 형태학, 운동학의 기초 데이터.
   Foundation of CME science: 40,000+ CME catalog is the basis for CME statistics, morphology, and kinematics.

2. **우주 날씨 예보**: Halo CME의 실시간 감지가 지자기 폭풍 예보의 핵심 입력.
   Space weather forecasting: real-time halo CME detection is critical input for geomagnetic storm prediction.

3. **혜성 발견**: 4,000+ sungrazing 혜성 발견 — 역사상 가장 성공적인 혜성 발견 기기. 시민 과학 프로젝트로도 활용.
   Comet discovery: 4,000+ sungrazing comets — most prolific comet discoverer in history.

4. **30년 운용**: C2/C3는 1996년부터 2025년 현재까지 거의 30년간 연속 운용 중.
   30 years of operation: C2/C3 have operated continuously from 1996 to present.

5. **후속 기기의 기준**: STEREO/COR1+COR2, Solar Orbiter/Metis, PROBA-3/ASPIICS 등이 LASCO 설계를 기반으로 발전.
   Standard for successors: STEREO/COR, Solar Orbiter/Metis, PROBA-3/ASPIICS all evolved from LASCO designs.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
