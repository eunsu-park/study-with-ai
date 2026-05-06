---
title: "Pre-Reading Briefing: A Catalog of White Light Coronal Mass Ejections Observed by the SOHO Spacecraft"
paper_id: "34_yashiro_2004"
topic: Low_SNR_Imaging
date: 2026-05-06
type: briefing
---

# A Catalog of White Light Coronal Mass Ejections Observed by the SOHO Spacecraft: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: S. Yashiro, N. Gopalswamy, G. Michalek, O. C. St. Cyr, S. P. Plunkett, N. B. Rich, R. A. Howard, "A Catalog of White Light Coronal Mass Ejections Observed by the SOHO Spacecraft," *J. Geophys. Res.*, 109, A07105, 2004. DOI: 10.1029/2003JA010282
**Author(s)**: S. Yashiro, N. Gopalswamy, G. Michalek, O. C. St. Cyr, S. P. Plunkett, N. B. Rich, R. A. Howard
**Year**: 2004

---

## 1. 핵심 기여 / Core Contribution

본 논문은 SOHO/LASCO C2/C3 코로나그래프가 1996–2002 년 (태양주기 23 의 전체 상승부 + 극대 + 초기 하강부) 동안 관측한 약 6907 개 CME 의 *온라인 카탈로그 (CDAW)* 를 공식적으로 기술하고 그 통계적 특성을 정리한다. 핵심 내용: (1) running-difference movie 위에서 *수동* 으로 CME 를 식별, (2) 각 CME 의 첫 번째 출현 시각, central position angle (CPA), apparent angular width $W$, height-time 곡선과 그 1·2차 다항식 적합으로부터 평균 속도와 가속도를 측정. 보고된 통계: normal CME 의 평균 폭이 $47^\circ$ (1996) 에서 $61^\circ$ (1999) 로 증가; 평균 속도가 $300\,\text{km s}^{-1}$ (극소) 에서 $500\,\text{km s}^{-1}$ (극대) 로 증가; halo CME 평균 속도 $957\,\text{km s}^{-1}$ vs normal $428\,\text{km s}^{-1}$; 느린 CME ($V\le250$) 는 가속, 빠른 CME ($V>900$) 는 감속 — 태양풍과의 항력성 상호작용 시사.

This paper formally describes the *online CDAW catalog* of all ~6907 CMEs detected by SOHO/LASCO C2/C3 between 1996 and 2002 (full ascending phase, maximum and early declining phase of cycle 23) and summarizes their statistical properties. Key elements: (1) *manual* CME identification on running-difference movies, (2) measurement of first-appearance time, central position angle (CPA), apparent angular width $W$ and height-time profile fitted by 1st- and 2nd-order polynomials to derive mean speed and acceleration. Reported statistics: mean width of normal CMEs grew from $47^\circ$ (1996) to $61^\circ$ (1999); mean speed rose from $300\,\text{km s}^{-1}$ (solar minimum) to $500\,\text{km s}^{-1}$ (maximum); halo CME mean speed $957\,\text{km s}^{-1}$ vs normal $428\,\text{km s}^{-1}$; slow CMEs ($V\le 250$) accelerate, fast CMEs ($V>900$) decelerate, suggesting drag-like interaction with the solar wind.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

CME 의 개념은 1971 년 OSO-7 코로나그래프에 의해 처음 발견되었고 (Tousey 1973), 그 후 Skylab/ATM (1973), Solwind (P78-1, 1979), SMM/C-P (1980) 가 누적적으로 약 2000 개의 CME 를 카탈로그화했다. 1995 년 SOHO/LASCO 의 발사로 관측 가용성과 시야 (32 R$_\odot$ 까지) 가 크게 향상되어, 단일 임무가 이전 모든 임무를 합친 것보다 더 많은 CME 를 기록할 수 있게 되었다. 그러나 LASCO 운영 초기에는 통일된 카탈로그가 없었고 (St. Cyr et al. 2000 이 1996–1998 년 841 CME 를 수동 카탈로그화), 본 논문은 그 노력을 cycle 23 전체로 확장하고 community-standard 한 *living catalog* 로 만든 결과이다.

The CME concept was first established with the OSO-7 coronagraph in 1971 (Tousey 1973). Skylab/ATM (1973), Solwind (P78-1, 1979) and SMM/C-P (1980) cumulatively cataloged about 2000 CMEs. The 1995 launch of SOHO/LASCO drastically improved coverage and field-of-view (out to 32 R$_\odot$); a single mission could now record more CMEs than all previous missions combined. Early LASCO operations lacked a unified catalog (St. Cyr et al. 2000 manually cataloged 841 CMEs in 1996–1998). This paper extends that effort across all of cycle 23 and establishes the community-standard *living catalog*.

### 타임라인 / Timeline

```
1971 ─ OSO-7 first CME detection (Tousey)
1973 ─ Skylab/ATM coronagraph (110 CMEs)
1980 ─ Solwind & SMM/C-P (~2000 CMEs combined)
1995 ─ SOHO launch; LASCO C1/C2/C3 commissioning
2000 ─ St. Cyr et al.: first 841 LASCO CMEs (1996–1998)
2003 ─ Halo CME classification (Gopalswamy)
2004 ─ THIS PAPER: 6907 CMEs, 1996–2002 (CDAW catalog)
2009 ─ STEREO/SECCHI launches; CME stereoscopy
2018+ ─ Automatic CME catalogs (CACTus, SEEDS, CORIMP)
```

---

## 3. 필요한 배경 지식 / Prerequisites

- **SOHO/LASCO instrument / SOHO/LASCO 기기**: C2 (2.1–6 R$_\odot$), C3 (4–32 R$_\odot$), pixel sizes 11.2 / 56.0 arcsec, 512×512 MVI compressed images.
- **Coronagraph background subtraction / 코로나그래프 배경 차감**: Running-difference vs temporal-median background; F-corona and K-corona components.
- **Position angle convention / 위치각 정의**: PA measured CCW from solar North; CPA = midangle between two edges; halo CMEs cannot have a CPA.
- **CME morphology / CME 형태학**: Three-part structure (bright leading edge + dark cavity + bright core); halo, partial halo, narrow / normal / wide classifications.
- **Height-time analysis / 높이-시간 분석**: Linear fit (mean speed) vs quadratic fit (mean acceleration) to projected leading edge.
- **Solar cycle proxy / 태양주기 지표**: Sunspot number, F10.7 — for cycle-rate correlation (Sec. 4 of paper).

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| CME (Coronal Mass Ejection) | 코로나로부터 ~$10^{15}$–$10^{16}$ g 의 플라스마와 자기장이 분출되는 사건 / Eruption of plasma + magnetic field, ~$10^{15}$–$10^{16}$ g, from corona |
| LASCO C2 / C3 | SOHO 의 외부 코로나그래프; FOV 2.1–6 R$_\odot$, 4–32 R$_\odot$ / Outer coronagraphs on SOHO |
| Running-difference movie | 인접 프레임 차분 영상 — 약한 운동 구조 강조 / Frame-to-frame difference, highlighting moving features |
| MVI format | LASCO 의 운영용 압축 영상 (512×512) / Operational compressed image format (512×512) |
| Apparent (sky-plane) speed | 천구면에 투영된 속도 — 진짜 속도의 하한 / Speed projected onto sky plane, lower bound of true speed |
| CPA (Central Position Angle) | CME 두 가장자리의 중앙 PA / Midangle between two CME edges |
| MPA (Measurement Position Angle) | 가장 빠르게 움직이는 가장자리의 PA (halo CME 의 경우) / PA of the fastest-moving edge (used for halos) |
| Halo CME | 차폐 디스크를 둘러싸는 CME (Earth-/anti-Earth-directed) / CME surrounding the occulting disk (Earth- or anti-Earth-directed) |
| Narrow / Normal / Wide | $W\le20^\circ$ / $20^\circ<W\le120^\circ$ / $W>120^\circ$ |
| Quality index | leading-edge 추적의 신뢰도 (0=poor … 5=excellent) / Reliability rating of leading-edge tracking |
| CDAW catalog | NASA GSFC Coordinated Data Analysis Workshops 가 호스팅하는 LASCO CME 카탈로그 / NASA GSFC-hosted LASCO CME catalog |

---

## 5. 수식 미리보기 / Equations Preview

Height-time 1차 적합 (linear fit, mean speed):

$$
h(t) = a + b\,t,\qquad b = \langle v\rangle
$$

2차 적합 (quadratic fit, mean acceleration):

$$
h(t) = a + b\,t + c\,t^2,\qquad 2c = \langle a\rangle
$$

평균 폭의 오차 (Sec. 3.1):

$$
\sigma_{\bar W}^2 = \frac{1}{n^2}\sum_{i=1}^{n}\sigma_{W_i}^2
$$

CPA 정의 (개념적):

$$
\text{CPA} = \tfrac{1}{2}\,(\text{PA}_{\text{edge}_1} + \text{PA}_{\text{edge}_2})
$$

CME 의 운동방정식 (drag interpretation, Sec. 4.2):

$$
m_{\text{CME}}\,\ddot h = F_{\text{prop}}(h) - F_{\text{drag}}(h,\dot h - v_{\text{sw}})
$$

(Math delimiters: `$...$` inline, `$$...$$` block — never `\(...\)` or `\[...\]`)

---

## 6. 읽기 가이드 / Reading Guide

- **Sec. 1 (Intro)** — CME 의 역사와 LASCO 의 위치, 카탈로그의 동기.
- **Sec. 2 (Catalog)** — 식별 절차, 측정 절차, 카탈로그의 attribute (Fig. 1 의 catalog screenshot).
- **Sec. 3.1 (Apparent Width)** — Fig. 2 의 7년치 폭 분포 히스토그램. 1996/1997 의 단일-피크 ($\sim40^\circ$) → 1998 이후 이중-피크 ($15^\circ + 50^\circ$) → 2001 이후 단일-피크 ($20^\circ$–$35^\circ$). Table 2 의 narrow/normal/wide 비율.
- **Sec. 3.2 (Latitudes)** — Fig. 3, Fig. 4. Solar minimum 에는 적도 $\pm20^\circ$, maximum 에는 모든 위도; high-latitude CME 의 N-S 비대칭.
- **Sec. 3.3 (Speeds)** — Fig. 5 의 7년치 속도 히스토그램. 평균 속도 281 → 521 km s$^{-1}$, 분포의 피크가 250 → 400 km s$^{-1}$. Table 3 의 narrow/normal/wide 별 평균 속도. Fig. 6 의 CME width-speed 산점도, $r=0.44$.
- **Sec. 3.4 (Acceleration)** — Fig. 7 의 측정 오차, Fig. 8 의 4 개 속도 구간별 가속도 분포. 핵심: 느린 CME 가속 / 빠른 CME 감속.
- **Sec. 4 (Discussion)** — St. Cyr 등과의 차이 (불일치율 7%); CME 이동에 대한 propelling vs drag 해석; 미래 계획 (halo / fast / SEP-associated 부분 카탈로그).

- **Sec. 1 (Intro)** — History of CMEs and where LASCO fits; catalog motivation.
- **Sec. 2 (Catalog)** — Identification, measurement and catalog attributes (see Fig. 1 screenshot).
- **Sec. 3.1 (Apparent Width)** — Fig. 2 width histograms over 7 years: single peak (~$40^\circ$) in 1996/1997 → bimodal ($15^\circ + 50^\circ$) from 1998 → single peak ($20^\circ$–$35^\circ$) from 2001. Table 2 narrow/normal/wide fractions.
- **Sec. 3.2 (Latitudes)** — Figs. 3–4. Equatorial $\pm20^\circ$ at minimum vs all latitudes at maximum; N-S asymmetry at high latitude.
- **Sec. 3.3 (Speeds)** — Fig. 5: mean speed 281 → 521 km s$^{-1}$, peak shifts 250 → 400 km s$^{-1}$. Table 3 by class. Fig. 6 width-speed scatter, $r=0.44$.
- **Sec. 3.4 (Acceleration)** — Fig. 7 measurement error analysis; Fig. 8 acceleration vs speed bin. Slow CMEs accelerate / fast decelerate.
- **Sec. 4 (Discussion)** — Differences vs St. Cyr (7% disagreement); propelling vs drag interpretation; future special-population catalogs.

---

## 7. 현대적 의의 / Modern Significance

CDAW 카탈로그는 우주환경 (space weather) 연구의 사실상 표준 입력으로 자리잡았다. 이후의 거의 모든 CME-flare 연관성 연구, halo CME 와 지자기폭풍 연관성 연구 (Gopalswamy 2003), CME 도착시간 예측 모델 (ENLIL, Drag-Based Model) 의 *훈련 데이터* 가 본 카탈로그에서 비롯된다. 또한 본 논문이 도입한 *수동 식별 + 시간 중앙값/running-difference background* 작업 흐름은 이후의 *자동* 카탈로그 (CACTus, SEEDS, CORIMP) 의 검증 기준점이 된다. Low-SNR Imaging 분야에서 본 논문이 갖는 의미는 두 가지: (i) running-difference 와 temporal-median 배경 차감이 약한 dynamic feature 검출에 어떻게 작용하는지의 community 표준 사례, (ii) 6907 개의 ground-truth label 을 통해 ML/automatic detection 알고리듬의 학습/평가에 직접 활용 가능한 데이터 자원.

The CDAW catalog has become the *de facto* standard input for space-weather research. Almost every subsequent study of CME-flare relations, halo-CME geomagnetic-storm correlations (Gopalswamy 2003) and CME arrival-time prediction models (ENLIL, Drag-Based Model) draws training data from this catalog. The manual-identification + temporal-median / running-difference workflow defined here also serves as a benchmark for later *automatic* catalogs (CACTus, SEEDS, CORIMP). For Low-SNR Imaging this paper matters in two ways: (i) it is the community-standard demonstration of how running-difference and temporal-median background subtraction enable detection of faint dynamic features; (ii) the 6907 ground-truth labels are a directly usable training/evaluation resource for ML and automatic-detection algorithms.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
