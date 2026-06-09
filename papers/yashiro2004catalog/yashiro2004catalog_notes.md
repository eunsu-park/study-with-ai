---
title: "A Catalog of White Light Coronal Mass Ejections Observed by the SOHO Spacecraft"
authors: "S. Yashiro, N. Gopalswamy, G. Michalek, O. C. St. Cyr, S. P. Plunkett, N. B. Rich, R. A. Howard"
year: 2004
journal: "Journal of Geophysical Research, 109, A07105"
doi: "10.1029/2003JA010282"
topic: Low_SNR_Imaging
tags: [CME, LASCO, SOHO, catalog, space-weather, coronagraph, running-difference]
status: completed
date_started: 2026-05-06
date_completed: 2026-05-06
---

# 34. A Catalog of White Light Coronal Mass Ejections Observed by the SOHO Spacecraft / SOHO 관측 백색광 CME 카탈로그

---

## 1. Core Contribution / 핵심 기여

본 논문은 SOHO/LASCO C2/C3 코로나그래프가 1996 년 1월 부터 2002 년 12 월 (태양주기 23 의 상승부 + 극대 + 초기 하강부) 까지 기록한 **6907 개의 코로나 질량분출 (CME)** 을 통일된 카탈로그 (CDAW Catalog) 로 정리하고 그 통계적 성질을 종합 분석한다. 작업 흐름은 (1) running-difference movie 위에서 *수동* CME 식별, (2) C2 (혹은 C2 가 없으면 C3) 영상에서 두 가장자리의 PA 측정 → CPA = 두 PA 의 중각, $W$ = 두 PA 의 차, (3) leading edge 의 height-time 추적과 1-차/2-차 다항식 적합으로 평균 속도 $b$ 와 평균 가속도 $2c$ 도출. 도출된 통계는 (a) 평균 폭이 $47^\circ$ (1996, 극소) → $61^\circ$ (1999, 극대 초기) → $53^\circ$ (2002), (b) CME 위도 분포가 극소 시 적도 $\pm20^\circ$, 극대 시 모든 위도, (c) 평균 속도가 $281\to 521\,\text{km s}^{-1}$ (극소→극대), (d) halo CME 평균 속도 $957$ vs normal $428\,\text{km s}^{-1}$, (e) 느린 CME ($V\le250$) 가속, 빠른 CME ($V>900$) 감속.

This paper presents the **CDAW catalog of 6907 CMEs** detected by SOHO/LASCO C2/C3 from January 1996 through December 2002 (cycle-23 ascending + maximum + early declining). Workflow: (1) *manual* identification on running-difference movies, (2) measurement on C2 (else C3) of the two-edge position angles, giving CPA = midangle, $W$ = edge-PA difference, (3) leading-edge height-time tracking with 1st-/2nd-order polynomial fits to derive mean speed $b$ and mean acceleration $2c$. Findings: (a) mean width grew $47^\circ$ (1996, min) → $61^\circ$ (1999, max-onset) → $53^\circ$ (2002), (b) latitudes confined to $\pm20^\circ$ at minimum vs all latitudes at maximum, (c) mean speed rose $281\to 521\,\text{km s}^{-1}$ (min→max), (d) halo CMEs $957\,\text{km s}^{-1}$ vs normal $428\,\text{km s}^{-1}$, (e) slow CMEs ($V\le 250$) accelerate while fast CMEs ($V>900$) decelerate.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction (Sec. 1, p. 1) / 도입

CME 의 역사적 발견은 1971 년 OSO-7 코로나그래프 (Tousey 1973) 까지 거슬러간다. 이후 Skylab/ATM (1973, 110 CMEs), Solwind (P78-1, 1980), SMM/C-P (1980) 가 ~2000 개의 CME 를 누적했다. 1995 년 SOHO/LASCO 의 발사로 단일 임무가 이전 모든 임무보다 많은 CME 를 기록하게 되었고 (이 카탈로그 시점에서 7000 개 초과), 이는 통계 분석의 양적·질적 도약을 가져왔다. 본 논문은 St. Cyr et al. (2000) 의 1996-1998 년 841 CME 카탈로그를 cycle 23 전체로 확장하고 community-standard *living catalog* 으로 정착시킨다.

CMEs were first detected by the OSO-7 coronagraph in 1971 (Tousey 1973). Later Skylab/ATM (1973, 110 CMEs), Solwind (P78-1, 1980) and SMM/C-P (1980) cumulatively recorded ~2000 CMEs. SOHO/LASCO (1995) by itself surpassed the entire previous record (>7000 CMEs by this paper). The work extends the 841-CME St. Cyr et al. (2000) list across all of cycle 23 and establishes a community-standard *living catalog*.

### Part II: The Online CME Catalog (Sec. 2, pp. 1-3) / 온라인 카탈로그

**2.1 CME identification.** 카탈로그는 LASCO C2 (FOV 2.1–6 R$_\odot$) 와 C3 (4–32 R$_\odot$) 의 모든 CME 를 포함한다 (C1 은 첫 5 년만 동작 → 제외). 영상은 1024×1024 (pixel size 11.2", 56.0") 이지만 운영용으로는 512×512 MVI 로 압축. 식별 기준: "두 개 이상의 연속 LASCO 영상에서 새로운 백색광 밝기 강조가 바깥쪽으로 이동하면 CME". 단일 영상에서도 *세 부분 구조* (밝은 leading edge + 어두운 cavity + 밝은 core) 가 명확하면 카탈로그에 포함. 식별은 *수동* 이며 SolarSoft IDL 도구로 running-difference movie 를 본다.

**2.2 CME measurements.** 각 CME 에 대해 (i) C2 첫 출현 시각, (ii) leading edge 높이 (디스크 중심 기준, *not* 태양 limb 기준) 의 시간 시퀀스, (iii) PA — leading edge 가 가장 빠르게 이동하는 PA 를 *MPA* (Measurement PA) 라 부르고, 두 가장자리의 PA 의 중각을 *CPA* (Central PA) 라 한다. Halo CME 는 occulting disk 를 둘러싸므로 CPA 정의 불가 → MPA 만 기록. 폭 $W$ = 두 가장자리 PA 의 차. 카탈로그의 attribute: 첫 출현 시각, CPA, $W$, linear-fit speed, 2nd-order-fit speed (last & 20 R$_\odot$), acceleration, MPA, daily plot, daily movie 링크. URL: http://cdaw.gsfc.nasa.gov/CME_list/. 약 4% 의 CME 는 측정점 부족 또는 데이터 갭으로 속도 측정 불가.

**2.1 Identification.** Catalog covers C2 (2.1–6 R$_\odot$) and C3 (4–32 R$_\odot$); C1 only ran for the first 5 years and is excluded. Original images are 1024×1024 (11.2"/56.0" pixels) but compressed to 512×512 MVI for operations. Identification rule: "a new white-light brightness enhancement moving outward in at least two consecutive LASCO images." Single images may suffice if a clear three-part CME structure (bright leading edge + dark cavity + bright core) is visible. Identification is *manual*, using SolarSoft IDL on running-difference movies.

**2.2 Measurements.** Per CME: (i) first C2 appearance time, (ii) leading-edge height time-series (relative to disk center, *not* limb), (iii) PAs — *MPA* (the PA where the leading edge moves fastest) and *CPA* (the midangle of the two edges). Halo CMEs surround the occulting disk so CPA is undefined → only MPA is recorded. Width $W$ = difference between the two edge PAs. Attributes: first time, CPA, $W$, linear-fit speed, 2nd-order-fit speed (last & at 20 R$_\odot$), acceleration, MPA, daily plot, daily movie. URL: http://cdaw.gsfc.nasa.gov/CME_list/. ~4% of CMEs lack speed estimates due to missing measurement points or data gaps.

### Part III: CME Properties (Sec. 3, pp. 4-9) / CME 특성

**3.1 Apparent Width.** Fig. 2 가 7 년치 폭 분포 히스토그램. 1996 (204 CMEs), 1997 (351), 1998 (697), 1999 (957), 2000 (1580), 2001 (1466), 2002 (1652). 분포는 1996/1997 의 단일 피크 ($\sim40^\circ$) → 1998-2000 의 *이중 피크* ($15^\circ + 50^\circ$) → 2001-2002 의 단일 피크 ($20^\circ$–$35^\circ$). Table 1: Average (Median) width of *normal* CMEs (20°<W≤120°): 47°(43°)→58°(53°)→56°(53°)→61°(58°)→57°(52°)→56°(52°)→53°(49°). 평균 폭 오차

$$
\sigma_{\bar W}^2 = \frac{1}{n^2}\sum_i\sigma_{W_i}^2
$$

→ 1996 의 30% (50%) 측정 오차 가정 시 $\sigma_{\bar W}\approx 1.2^\circ\;(2.0^\circ)$, 2000 에서는 $0.6^\circ\;(0.9^\circ)$. 즉 표본 크기 덕분에 평균 폭의 *통계적* 오차는 매우 작고, 태양주기 변동은 유의함. Table 2 의 narrow($W\le20^\circ$)/normal/wide($W>120^\circ$) 비율: narrow 가 1996 의 16% 에서 2002 의 23% 로 단조 증가; wide 가 6% → 13% → 10%.

**3.2 Apparent Latitudes.** CPA→latitude 변환 ($0^\circ\to90^\circ$, $90^\circ\to 0^\circ$, $180^\circ\to-90^\circ$, $270^\circ\to0^\circ$). Fig. 3 의 1996-2002 위도 분포: 극소 (1996-1997) 에는 거의 모든 CME 가 $\pm20^\circ$ 내 (streamer belt 와 일치); 1998 에는 폭이 $\pm60^\circ$ 로 확장; 1999-2000 (극대) 에 모든 위도. Critical latitude $\phi$ (전체 CME 의 80% 가 이 위도 내) 는 1996 의 $\pm20^\circ$ 부터 2000 의 $-61^\circ/65^\circ$ 까지 변동, *N-S 비대칭* 도 보임 (Sec. 3.2 17 단락). Fig. 4 의 width-latitude scatterplot: 명확한 위도-폭 상관 없음, 다만 high-latitude (>80°) CME 의 median width 가 약간 더 큰 경향 (projection effect 으로 추정).

**3.3 Apparent Speeds.** Fig. 5 의 1996-2002 속도 히스토그램. 평균 속도가 281 → 320 → 421 → 499 → 502 → 481 → 521 km s$^{-1}$, 분포의 피크가 $\sim250\to\sim400$ km s$^{-1}$. 2001 의 평균 속도 423 km s$^{-1}$ 는 2000 (452) 과 2002 (468) 보다 *낮음* — 이는 2001 의 narrow CME 비율이 13% 로 떨어진 것 (나머지 normal/wide) 과 관련, 측정/기록 효과로 해석. Table 3: average speed by class. *Halo* CME 평균 속도 957 km s$^{-1}$ — normal CME (428) 의 *2배 이상*. Fig. 6 의 width-speed scatterplot: $W<60^\circ$ 에서는 거의 무상관 (평균 속도 $508\to398$ km s$^{-1}$ 로 $W$ 가 $0\to70^\circ$ 변화), $W>60^\circ$ 에서 양의 상관, $W=360^\circ$ (halo) 에서 $957$ km s$^{-1}$. Fast CMEs ($V>900$) 의 width-speed 상관계수 $r=0.44$.

**3.4 Apparent Acceleration.** 5 점 이상의 height-time 측정점이 있는 3058 CME 에 대해 quadratic fit 으로 가속도 도출. Fig. 7 의 오차 분석: 3 점 측정 시 마지막 점이 0.1 R$_\odot$ 어긋나면 $V$ 의 오차 ~4%, $a$ 의 오차 $\pm22\,\text{m s}^{-2}$ (큼); 5 점 시 $a$ 의 오차 $\pm3\,\text{m s}^{-2}$ (훨씬 작음). Fig. 8 의 4 개 속도 구간 가속도 분포:
- $V\le 250$ km s$^{-1}$ (475 CMEs): 피크 $\sim+5\,\text{m s}^{-2}$ — *대부분 가속*
- $250<V\le 450$ (872): 피크 $\sim 0$ — 거의 등속
- $450<V\le 900$ (1288): 피크 $\sim -5\,\text{m s}^{-2}$
- $V>900$ (423): 피크 $\sim -15\,\text{m s}^{-2}$ — *대부분 감속*

해석: CME 와 태양풍의 *항력* (drag) 상호작용이 LASCO C2/C3 FOV (2-32 R$_\odot$) 에서 trajectory 를 결정. 태양풍보다 빠른 CME 는 감속, 느린 CME 는 끌려들어가 가속. 이는 *Gopalswamy et al.* (2000, 2001b) 의 1-AU 도착시간 모델과 정합.

**3.1 Apparent Width.** Fig. 2 shows 7 years of width histograms, 204 CMEs (1996) up to 1652 (2002). The shape evolves: single peak (~$40^\circ$) at minimum → bimodal ($15^\circ + 50^\circ$) in 1998-2000 → single peak ($20^\circ$–$35^\circ$) by 2001-2002. Table 1: average (median) widths of *normal* CMEs grow $47^\circ(43^\circ)$ to $61^\circ(58^\circ)$ and back. The mean-width error formula $\sigma_{\bar W}^2 = (1/n^2)\sum\sigma_{W_i}^2$ gives only $1.2^\circ$ in 1996 and $0.6^\circ$ in 2000 — sample size makes the cycle variation highly significant. Table 2 narrow fraction grows 16% → 23%.

**3.2 Apparent Latitudes.** CPAs are converted to apparent latitude. Fig. 3: minimum CMEs hug the equator $\pm20^\circ$ (matching the streamer belt); width grows to $\pm60^\circ$ in 1998; all latitudes during 1999-2000 maximum. Critical latitude (containing 80% of CMEs) widens $\pm20^\circ\to-61^\circ/+65^\circ$; visible N-S asymmetry in high latitudes. Fig. 4 width-latitude scatter shows no strong correlation, only a slight tendency for high-latitude (>80°) CMEs to be wider (projection effect).

**3.3 Apparent Speeds.** Fig. 5: average speed 281 → 521 km s$^{-1}$ across 1996-2002, peak shifts $\sim 250\to 400$ km s$^{-1}$. 2001 dips ($423$ km s$^{-1}$) anomalously, linked to lower narrow-CME fraction. Halo CMEs average $957$ km s$^{-1}$, *over twice* normal CMEs (428). Fig. 6 width-speed: weak for $W<60^\circ$, positive correlation for $W>60^\circ$ ($r=0.44$ for fast CMEs).

**3.4 Apparent Acceleration.** 3058 CMEs with ≥5 measurements get a quadratic fit. Fig. 7 error analysis: 3 points → $\Delta V\sim 4\%$ but $\Delta a\sim\pm 22\,\text{m s}^{-2}$ (large); 5 points → $\Delta a\sim\pm 3\,\text{m s}^{-2}$. Fig. 8 acceleration distributions by speed bin: slow CMEs peak at $+5$, intermediate at 0, fast at $-5$ to $-15\,\text{m s}^{-2}$. Interpretation: drag-like solar-wind interaction in 2-32 R$_\odot$ FOV — CMEs faster than wind decelerate, slower CMEs accelerate.

### Part IV: Discussion (Sec. 4, pp. 9-10) / 토의

**4.1 Number of CMEs.** St. Cyr et al. (2000) 와의 비교: 1996-1998 년 동기간에 St. Cyr 은 841, 본 카탈로그는 1083 — 차이 265 CME. 그 중 215 가 식별 기준 차이로 설명됨 (e.g. *coronal anomaly* 가 St. Cyr 에서는 CME 가 아니지만 본 카탈로그에서 110 개를 CME 로 분류; St. Cyr 은 동일 PA 의 jet-like CME 의 첫 번째만 카운트하지만 본 카탈로그는 모두 카운트, 86 개 추가). 결과: 전체 *불일치율* 7%. 즉 CME 수동 식별은 본질적으로 *주관적* 이지만 합리적인 카탈로그끼리는 90% 이상 일치한다.

**4.2 CME Trajectory.** 빠른 CME 의 감속과 느린 CME 의 가속은 propelling force (자기 부력 + Lorentz) 가 $r<2\,R_\odot$ 에서 지배하고, $r>2\,R_\odot$ 에서는 drag 가 지배함을 시사. 빠른 CME ($V>1000$) 에서 가속하는 CME 가 한 사례도 없는 것은 모든 fast CME 가 $r=2\,R_\odot$ 이전에 *가속을 끝냄* 을 보여준다. 따라서 성공적 CME 모델은 (i) $<2\,R_\odot$ 의 propelling force, (ii) $>2\,R_\odot$ 의 drag 를 *모두* 다뤄야 한다.

**4.3 Future Plan.** 미래 작업: (i) halo CME 부분 카탈로그 (geoeffective), (ii) fast CME 부분 카탈로그 (SEP-associated), (iii) interacting CME 카탈로그.

**4.1 Number of CMEs.** Compared to St. Cyr et al. (2000) over 1996-1998: 1083 (this catalog) vs 841 (St. Cyr); 265 difference. 215 traced to identification-criterion differences (110 "coronal anomalies" counted as CMEs here but not by St. Cyr; 86 successive jet-like CMEs at the same PA counted individually here). Net 7% disagreement — manual identification is *subjective* but reasonable catalogs agree at the 90%+ level.

**4.2 CME Trajectory.** Slow → accelerate / fast → decelerate is consistent with propelling forces (magnetic buoyancy + Lorentz) dominating below 2 R$_\odot$ and drag dominating above. No fast ($V>1000$) CMEs accelerate — all of them finished accelerating before $r=2\,R_\odot$. A complete CME model needs both <2 R$_\odot$ propelling and >2 R$_\odot$ drag.

**4.3 Future Plan.** Halo, fast, and interacting CME sub-catalogs are planned.

### Part IV.5: Specific Acceleration Numbers and Their Interpretation / 가속도 수치와 해석

논문 Sec. 3.4 의 Fig. 8 데이터를 정량적으로 정리:

| Speed bin (km s$^{-1}$) | N CMEs | Acceleration peak (m s$^{-2}$) | Interpretation |
|---|---|---|---|
| $V \le 250$ | 475 | $\sim +5$ | Slow CMEs accelerated by faster solar wind drag |
| $250 < V \le 450$ | 872 | $\sim 0$ | Approximately co-moving with the solar wind |
| $450 < V \le 900$ | 1288 | $\sim -5$ | Mildly faster than wind; mild deceleration |
| $V > 900$ | 423 | $\sim -15$ | Substantially faster than wind; strong drag |

이 단조로운 추세 — 느릴수록 가속, 빠를수록 감속 — 은 LASCO C2/C3 FOV 가 *항력 영역* (drag regime) 임을 강력히 시사하며, propelling force (자기 부력) 가 이미 $r<2\,R_\odot$ 에서 종료됨을 함의한다. Fast CME ($V>1000$ km s$^{-1}$) 중 가속하는 사례가 *전무* 하다는 관찰이 이 결론의 핵심 증거.

This monotonic trend is the strongest single piece of evidence in the paper that LASCO's 2-32 R$_\odot$ FOV is the *drag regime*: propelling forces (magnetic buoyancy + Lorentz) finish their work below 2 R$_\odot$, and from there outward CMEs are passively dragged by the ambient solar wind. The complete absence of accelerating fast CMEs ($V>1000$ km s$^{-1}$) is the smoking gun.

**Comparison to slow solar wind.** The slow solar wind (streamer-belt component) has speed $\sim 350\text{–}400$ km s$^{-1}$ at 1 AU; below 32 R$_\odot$ it is even slower (~200-300 km s$^{-1}$). CMEs in the $V<250$ bin are therefore *slower than ambient wind* at LASCO altitudes — they get *pushed* outward by the wind ram pressure, hence the positive acceleration. CMEs at $V>900$ are super-Alfvénic and faster than even the fast wind ($\sim 700$ km s$^{-1}$), hence strong drag deceleration.

**느린 태양풍과의 비교.** 느린 태양풍은 1 AU 에서 350-400 km s$^{-1}$, LASCO 고도에서는 200-300 km s$^{-1}$. $V<250$ 구간 CME 는 *주변 풍보다 느림* → 태양풍 ram pressure 에 의해 가속. $V>900$ 구간은 super-Alfvénic 으로 빠른 태양풍 (~700) 보다도 빨라 강한 drag.

### Part V: Summary (Sec. 5, p. 10) / 요약

본 논문 핵심 결론 7 가지:
1. 카탈로그 ~7000 CME, 1996-2002.
2. Width 의 cycle 변동 ($47^\circ\to61^\circ$, normal CMEs).
3. Bimodal width 분포는 1998-2000 (early max) 에만 명확.
4. 위도 분포의 cycle 변동 (적도 → 모든 위도).
5. 평균 속도 cycle 변동 ($300\to500$ km s$^{-1}$).
6. Halo CME 의 평균 속도 (957) 가 normal (428) 의 2 배 이상.
7. Slow CME 가속, fast CME 감속 — drag 시사.

Seven take-aways: (1) ~7000 CMEs catalog, (2) width grows cycle min→max, (3) bimodal width only at early max, (4) latitudes spread with cycle, (5) speeds rise cycle min→max, (6) halo CMEs >2× faster than normal, (7) slow accelerate / fast decelerate.

---

## 3. Key Takeaways / 핵심 시사점

1. **수동 식별 + running-difference 가 카탈로그의 *ground truth* 를 정의했다 / Manual identification + running-difference defined the catalog's ground truth** — 7% 의 카탈로그 간 불일치율은 *수동* 라벨이 갖는 본질적 주관성을 드러낸다. 이후 자동 알고리듬의 평가 기준점. The 7% inter-catalog disagreement reveals manual labeling's intrinsic subjectivity — yet it became the reference for later automatic catalogs.

2. **태양주기 23 의 통계적 특성이 명확히 드러난다 / Cycle-23 statistical properties are clearly revealed** — 평균 폭이 $47\to61\to53^\circ$, 평균 속도가 $281\to521$ km s$^{-1}$ 로 단조에 가까운 cycle dependence — 이전 carbon-paper-era 카탈로그 (~2000 CMEs) 로는 불가능했던 정밀 측정. Width $47\to61\to53^\circ$ and speed $281\to521\,\text{km s}^{-1}$ trace cycle phase precisely — only possible with the LASCO sample size.

3. **Halo CME 는 더 빠르고 더 넓은 *집단* 이다 / Halo CMEs are intrinsically faster and wider** — 957 vs 428 km s$^{-1}$ 평균 속도 — geoeffective CME (지자기 폭풍의 90% 이상이 halo CME) 의 위험도가 단순한 LOS effect 가 아닌 *물리적* 차이임을 시사. 957 vs 428 km s$^{-1}$ — geoeffective halos are physically (not just LOS-projected) distinct.

4. **Drag-dominated CME trajectory in 2-32 R$_\odot$** — Fig. 8 의 가속도-속도 회귀: 솔라윈드보다 빠른 CME 는 감속, 느린 CME 는 가속, intermediate CME 는 거의 등속. CME 도착시간 예측 모델 (Drag-Based Model, ENLIL) 의 *경험적* 핵심 입력. Solar-wind drag drives the 2-32 R$_\odot$ trajectory; the empirical foundation of Drag-Based and ENLIL arrival models.

5. **이중모달 폭 분포는 *one-time* 현상이 아니다 / Bimodal width distribution is cycle-phase-specific** — 1998-2000 (early max) 에만 보이는 narrow ($\sim15^\circ$) + normal ($\sim50^\circ$) 두 봉우리 구조. CME 발생 메커니즘이 cycle 단계에 따라 *다중* (active region eruption + jet-like flux emergence) 일 수 있음을 시사. The narrow+normal twin peaks only appear in 1998-2000 — suggesting multiple production mechanisms active in early max.

6. **5-점 측정 요구가 acceleration 의 신뢰도를 결정 / The 5-point requirement is what makes acceleration reliable** — 3 점이면 $\Delta a\approx\pm22\,\text{m s}^{-2}$ (rms), 5 점이면 $\pm3\,\text{m s}^{-2}$. 7 배 이상의 정밀도 차이. 카탈로그가 3058/6907 (~44%) 만 가속도를 보고하는 이유. 3-point fit yields $\pm22\,\text{m s}^{-2}$ vs 5-point $\pm3\,\text{m s}^{-2}$ — explains the 3058/6907 acceleration coverage.

7. **Living catalog 패러다임의 정착 / The "living catalog" paradigm is established** — St. Cyr 의 권고에 따라 카탈로그는 *최종* 산출물이 아닌 *지속 개정* 자원으로 운영. 이후 Stanford LMSAL HEK, JSOC SHARP, HMI vector synoptics 등 모두 이 패러다임 채택. The catalog runs as a continuously revised resource, a paradigm later adopted by HEK, SHARP and others.

8. **수동 카탈로그가 자동 알고리듬의 학습 데이터 / Manual catalog as training data for automatic algorithms** — CACTus (Robbrecht & Berghmans 2004), SEEDS (Olmedo et al. 2008), CORIMP (Byrne et al. 2012), 이후 deep-learning 기반 검출기 (e.g. CMEs from STEREO/SECCHI) 의 학습/평가 데이터로 본 카탈로그가 사실상 ground truth. CACTus, SEEDS, CORIMP, and subsequent deep-learning detectors all use this catalog as ground truth.

---

## 4. Mathematical Summary / 수학적 요약

**Linear height-time fit (mean speed, Sec. 2.2):**

$$
h(t) = a + b\,t,\qquad b = \langle v\rangle\;\;[\text{km s}^{-1}]
$$

**Quadratic height-time fit (mean acceleration):**

$$
h(t) = a + b\,t + c\,t^2,\qquad 2c = \langle a\rangle\;\;[\text{m s}^{-2}]
$$

— $h$ is leading-edge heliocentric height, measured from disk center. The catalog reports both fits; *initial* and *final* speeds come from the quadratic fit at endpoints; speed at 20 R$_\odot$ is also tabulated.

**Position angle definitions:**

$$
\text{CPA} = \tfrac{1}{2}(\text{PA}_{\text{edge}_1}+\text{PA}_{\text{edge}_2}),\qquad W = |\text{PA}_{\text{edge}_2}-\text{PA}_{\text{edge}_1}|
$$

— PA measured CCW from solar North. Halo CMEs have $W=360^\circ$ and undefined CPA — only MPA is recorded.

**Mean-width statistical error (Sec. 3.1):**

$$
\sigma_{\bar W}^2 = \frac{1}{n^2}\sum_{i=1}^{n}\sigma_{W_i}^2
$$

— assumes independent measurement errors; with 30% per-CME width error and $n=204$ in 1996, $\sigma_{\bar W}\approx1.2^\circ$.

**CME trajectory equation of motion (Sec. 4.2 interpretation):**

$$
m_{\text{CME}}\,\ddot h = F_{\text{prop}}(h) - F_{\text{drag}}(h,\dot h - v_{\text{sw}}(h))
$$

— $F_{\text{prop}}$ dominates $r<2\,R_\odot$, $F_{\text{drag}}$ dominates $r>2\,R_\odot$. The drag term is the empirical basis of the Drag-Based Model (DBM):

$$
\ddot h = -\gamma(\dot h - v_{\text{sw}})|\dot h - v_{\text{sw}}|
$$

with empirical $\gamma\sim 10^{-7}\text{ km}^{-1}$.

**Worked numerical example (one CDAW catalog event).** From Fig. 1c-d: CME first at C2 06:54:05 UT on 2000-01-01, PA=11°, width=76°. Linear fit gives $V=337$ km s$^{-1}$ (slope of $h$-vs-$t$ over 18:00-24:00 UT, 12-25 R$_\odot$ range). Quadratic fit yields $V_{\text{2nd}}=531$ km s$^{-1}$, $V_{20\,R_\odot}=470$ km s$^{-1}$, $a=8.8\,\text{m s}^{-2}$. Interpretation: this is a slow CME (linear $V<450$) that *accelerates* (positive $a$), consistent with Sec. 3.4 — slow CMEs are still being driven outward by drag from the faster background solar wind. Catalog row: `2000/01/01, 06:54:05, 21, 76, 337, 531, 470, 8.8, 11`.

**Drag-Based Model derivation from this catalog.** The catalog's Fig. 8 shows acceleration $a$ as a function of speed bin. Vršnak et al. (2013) formalized this empirically: assume the only force on the CME body in 2-32 R$_\odot$ is hydrodynamic drag,

$$
m_{\text{CME}}\,\frac{dv}{dt} = -C_d\,A\,\rho_{\text{sw}}\,(v - v_{\text{sw}})\,|v - v_{\text{sw}}|
$$

Defining the *drag parameter* $\gamma = C_d A \rho_{\text{sw}}/m_{\text{CME}}$ and treating $\gamma$ as approximately constant in the LASCO FOV gives

$$
\frac{dv}{dt} = -\gamma\,(v - v_{\text{sw}})\,|v - v_{\text{sw}}|
$$

with analytic solution $v(r)=v_{\text{sw}} + (v_0-v_{\text{sw}})/(1\pm \gamma|v_0-v_{\text{sw}}|(r-r_0))$. Fitting CDAW data gives $\gamma\sim 0.2\text{–}2\times 10^{-7}\text{ km}^{-1}$ and $v_{\text{sw}}\sim 400\text{ km s}^{-1}$. This is now the operational basis of NOAA SWPC's CME arrival forecasts.

**Drag-Based Model 유도.** Fig. 8 의 가속도-속도 관계를 Vršnak et al. (2013) 가 형식화: 항력만 고려하면 $\dot v = -\gamma(v-v_{\text{sw}})|v-v_{\text{sw}}|$, 해석해는 $v(r)=v_{\text{sw}}+(v_0-v_{\text{sw}})/(1\pm\gamma|v_0-v_{\text{sw}}|(r-r_0))$. CDAW 데이터로부터 $\gamma\sim 0.2\text{–}2\times10^{-7}\text{ km}^{-1}$. NOAA SWPC 의 CME 도착 예측의 운영 기반이다.

**Speed projection correction.** The catalog reports *apparent* sky-plane speed $v_{\text{app}}$. For a CME with true 3-D speed $v_{\text{true}}$ at heliographic longitude $\phi$ from the limb,

$$
v_{\text{app}} = v_{\text{true}}\,\sin\phi
$$

— the catalog speed underestimates the true speed by $\sin\phi$. For halo CMEs (Earth-directed, $\phi\to 0$) the catalog speed is severely underestimated; cone-model fits (Michalek et al. 2003, Xie et al. 2004) deproject these. For limb CMEs ($\phi=90^\circ$) $v_{\text{app}}\approx v_{\text{true}}$, which is why the width-speed correlation $r=0.44$ in Fig. 6 is computed mostly on quasi-limb events.

**투영 보정.** 카탈로그 속도는 *겉보기* 속도 $v_{\text{app}}=v_{\text{true}}\sin\phi$. Halo ($\phi\to0$) 는 심각히 과소추정 → cone-model 보정 필요. Limb ($\phi=90^\circ$) 는 $v_{\text{app}}\approx v_{\text{true}}$.

**Width classification rule:**

$$
\text{class}(W) = \begin{cases}\text{narrow} & W\le 20^\circ\\ \text{normal} & 20^\circ<W\le 120^\circ\\ \text{wide} & W>120^\circ\\ \text{halo} & W=360^\circ\end{cases}
$$

**Solar-cycle correlation (qualitative).** Sunspot number $R_z$ and CME daily rate $N_{\text{CME}}$ are both proxies of cycle phase. From Table 1 + monthly $R_z$:

$$
N_{\text{CME}}(t) \approx \alpha + \beta R_z(t)
$$

with $\beta\sim 0.025\text{ CME day}^{-1}\text{ SSN}^{-1}$ (computed from $\sim 0.6$/day in 1996 vs $\sim 4.4$/day in 2000 against $R_z\approx 8\to 120$).

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1971 Tousey — first CME, OSO-7
   │
1973 MacQueen et al. — Skylab/ATM (110 CMEs)
   │
1980 Sheeley et al. — Solwind/SMM (~2000 CMEs)
   │
1984 Hundhausen — first CME statistical synthesis
   │
1995 Brueckner et al. — LASCO instrument launch
   │
2000 St. Cyr et al. — first 841 LASCO CMEs (1996-1998)
   │
2003 Gopalswamy — halo CME classification
   │
2004 ★ THIS PAPER — 6907 CME catalog (1996-2002)
   │
2004 Robbrecht & Berghmans — CACTus auto-detection
   │
2008 Olmedo et al. — SEEDS auto-detection
   │
2012 Byrne et al. — CORIMP multi-scale CME tracking
   │
2018+ Bobra & Mason / others — ML for CME / SEP forecast
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Brueckner et al. 1995 (LASCO instrument) | 본 카탈로그가 사용하는 관측 기기의 설명 / Defines the observing instrument | High — without LASCO, no catalog |
| St. Cyr et al. 2000 (first LASCO catalog) | 본 논문이 *직접* 확장하는 선행 카탈로그 / Directly preceding catalog this paper extends | High — Sec. 4.1 explicitly compares |
| Gopalswamy et al. 2003 (halo CME classification) | Halo CME 의 type F/A/P 분류와 geo-effectiveness / Halo classification used here | High — halo speed analysis (957 km s$^{-1}$) cites this |
| Howard et al. 1985 (SMM CME catalog) | Pre-SOHO 의 ~1200 CME 카탈로그 (비교 대상) / Pre-SOHO comparison catalog | Medium — used for cycle-22 vs cycle-23 comparisons |
| Robbrecht & Berghmans 2004 (CACTus) | 본 카탈로그의 자동화 후속 / Automated successor (and primary alternative) | High — modern users typically compare CDAW vs CACTus |
| Morgan & Druckmüller 2014 (MGN) | LASCO 영상의 *enhancement* — 본 카탈로그의 가시성 향상 | Medium — improves the imagery the catalog is built on |
| Vourlidas et al. 2010 (CME mass) | CDAW 의 width/speed 와 결합되어 mass 통계 / Mass attribute added later to CDAW | Medium — extends the catalog parameters |
| Drag-Based Model (Vršnak et al. 2013) | Fig. 8 의 acc-vs-speed 결과를 모델로 형식화 / Formalizes the empirical drag relation | High — direct theoretical descendant |

---

## 5.5 Reproduced Statistics from the Catalog / 카탈로그에서 재현되는 통계

기억할 핵심 숫자 (직접 인용 가능):

- **Total CMEs (1996-2002):** 6907 detected; 6599 with measurable speeds; 3058 with measurable acceleration.
- **Annual count: 204, 351, 697, 957, 1580, 1466, 1652** (1996-2002).
- **Mean width of normal CMEs:** $47^\circ$ (1996, min) → $61^\circ$ (1999, max-onset) → $53^\circ$ (2002).
- **Mean speed:** $281$ km s$^{-1}$ (1996) → $521$ km s$^{-1}$ (2002); peak shift $250\to 400$ km s$^{-1}$.
- **Halo CME mean speed: $957$ km s$^{-1}$** vs **normal mean $428$ km s$^{-1}$** — factor of 2.24.
- **Critical latitude $\phi$:** $\pm 20^\circ$ (1996) → $\pm 65^\circ$ (2000).
- **Bimodal width peaks:** $\sim15^\circ$ (narrow) + $\sim 50^\circ$ (normal), only 1998-2000.
- **Fast CME width-speed correlation:** $r=0.44$ (Fig. 6c).
- **Inter-catalog disagreement (vs St. Cyr 2000):** 7% net.
- **Fastest CME in catalog:** 2604 km s$^{-1}$ on 12 May 2000.
- **Acceleration error:** $\pm 22$ m s$^{-2}$ (3 points) → $\pm 3$ m s$^{-2}$ (5 points).

These ten numbers are the most-cited deliverables of the paper.

이 10 개의 수치는 본 논문에서 가장 자주 인용되는 결과이며, 이후 모든 CME 통계 비교의 *baseline* 이 된다.

---

## 6.5 Practical Notes for Catalog Users / 카탈로그 이용자 실무 노트

**Caveats from manual identification / 수동 식별의 한계.**
1. **Subjectivity / 주관성** — Sec. 4.1 의 7% disagreement rate 는 *상한값* 이 아니라 *대표값*. CME 속성 (특히 width, leading edge 정의) 에 따라 카탈로그 간 차이가 더 클 수 있다. ML 학습 시 single-catalog ground truth 의 noise floor 로 인식해야 한다.
2. **Faint event under-counting / 약한 이벤트 누락** — narrow CME ($W\le 20^\circ$) 의 비율이 1996 년 16% 에서 2002 년 23% 로 단조 증가하는 것은 *물리적* 변화가 아닐 가능성이 큼 (관측자 학습효과, jet-like CME 의 식별 기준 변화). 시간 의존 분석은 narrow 비율 변화를 보정해야 한다.
3. **Halo CME selection bias / Halo 편향** — halo CME 는 극대 시 더 자주 식별되지만 LOS effect 도 있음. 통계 분석 시 halo + non-halo 를 분리할 것.
4. **Speed = linear-fit by default / 속도는 기본 1차 적합** — 카탈로그의 *대표* 속도는 1차 적합값. 가속하는 CME (slow) 는 평균 속도 < 도착 속도; 감속하는 CME (fast) 는 반대. 도착시간 예측에는 *2차 적합 + 20 R$_\odot$* 속도 사용 권장.

**Manual-identification caveats:** 7% disagreement is *typical*; narrow-CME fraction (16% → 23%) likely reflects observer learning, not physics; halo selection is biased toward maximum; arrival-time prediction should use the 2nd-order-fit speed at 20 R$_\odot$.

**Catalog format / 카탈로그 형식.**
- HTML matrix at https://cdaw.gsfc.nasa.gov/CME_list/
- Per-event JavaScript movie + height-time PNG plots
- Text-only version with initial speeds (machine-readable)
- Daily log with quality index (0=poor … 5=excellent)
- 후속 확장: halo list, fast list, SEP-associated list (Sec. 4.3 future plan).

**Modern automated alternatives (post-paper) / 자동화 대안.**

| Catalog | Reference | Method | Strength | Weakness |
|---|---|---|---|---|
| CDAW (this paper) | Yashiro 2004 | Manual + RD | Highest reliability, longest record | Subjective; updates lag |
| CACTus | Robbrecht & Berghmans 2004 | (t, height) Hough transform | Real-time, reproducible | Misses faint CMEs |
| SEEDS | Olmedo 2008 | Region growing on RD | Faint CME sensitivity | More false positives |
| CORIMP | Byrne 2012 | Multiscale + dynamic separation | CME tracking through 32 R$_\odot$ | Compute-heavy |

**Recommended use pattern.** For statistics use CDAW + CACTus *intersection* (high reliability events). For ML training use CDAW labels with 7% label-noise prior. For time-series modeling correct for the narrow-CME fraction trend.

**권장 사용 패턴.** 통계 분석에는 CDAW $\cap$ CACTus, ML 훈련에는 CDAW 라벨 (7% noise prior), 시계열 모델링에는 narrow CME 비율 보정.

---

## 7. References / 참고문헌

- S. Yashiro, N. Gopalswamy, G. Michalek, O. C. St. Cyr, S. P. Plunkett, N. B. Rich, R. A. Howard, "A catalog of white light coronal mass ejections observed by the SOHO spacecraft," *J. Geophys. Res.*, 109, A07105, 2004. DOI: 10.1029/2003JA010282
- G. E. Brueckner et al., "The Large Angle Spectroscopic Coronagraph (LASCO)," *Solar Phys.*, 162, 357-402, 1995.
- O. C. St. Cyr et al., "Properties of coronal mass ejections: SOHO LASCO observations from January 1996 to June 1998," *J. Geophys. Res.*, 105, 18169-18185, 2000.
- R. A. Howard, D. J. Michels, N. R. Sheeley Jr., M. J. Koomen, "The observation of a coronal transient directed at Earth," *Astrophys. J.*, 263, L101-L104, 1982.
- R. A. Howard, N. R. Sheeley Jr., D. J. Michels, M. J. Koomen, "Coronal mass ejections — 1979-1981," *J. Geophys. Res.*, 90, 8173-8191, 1985.
- N. Gopalswamy et al., "Prominence eruptions and coronal mass ejection: A statistical study using microwave observations," *Astrophys. J.*, 586, 562-578, 2003.
- A. J. Hundhausen, "Sizes and locations of coronal mass ejections — SMM observations from 1980 and 1984-1989," *J. Geophys. Res.*, 98, 13177, 1993.
- E. Robbrecht and D. Berghmans, "Automated recognition of coronal mass ejections (CMEs) in near-real-time data," *Astron. Astrophys.*, 425, 1097-1106, 2004.
- J. P. Byrne, H. Morgan, S. R. Habbal, P. T. Gallagher, "Automatic detection and tracking of coronal mass ejections in coronagraph time series," *Astrophys. J.*, 752, 145, 2012.
- B. Vršnak et al., "Propagation of interplanetary coronal mass ejections: The drag-based model," *Solar Phys.*, 285, 295-315, 2013.
- R. Tousey, "The solar corona," in *Space Research XIII*, Akademie-Verlag, Berlin, 1973.
- H. Morgan and M. Druckmüller, "Multi-scale Gaussian normalization for solar image processing," *Solar Phys.*, 289, 2945-2955, 2014.
- CDAW SOHO/LASCO CME Catalog: https://cdaw.gsfc.nasa.gov/CME_list/
- E. Olmedo, J. Zhang, H. Wechsler, A. Poland, K. Borne, "Automatic detection and tracking of coronal mass ejections in coronagraph time series," *Solar Phys.*, 248, 485-499, 2008.
- C. Xie, L. Ofman, G. Lawrence, "Cone model for halo CMEs: Application to space weather forecasting," *J. Geophys. Res.*, 109, A03109, 2004.
- G. Michalek, N. Gopalswamy, S. Yashiro, "A new method for estimating widths, velocities, and source location of halo coronal mass ejections," *Astrophys. J.*, 584, 472-478, 2003.
- D. F. Webb, E. W. Cliver, N. U. Crooker, O. C. St. Cyr, B. J. Thompson, "Relationship of halo coronal mass ejections, magnetic clouds, and magnetic storms," *J. Geophys. Res.*, 105, 7491-7508, 2000.
- N. Gopalswamy, A. Lara, S. Yashiro, M. L. Kaiser, R. A. Howard, "Predicting the 1-AU arrival times of coronal mass ejections," *J. Geophys. Res.*, 106, 29207-29217, 2001b.
- S. W. Kahler, "The correlation between solar energetic particle peak intensities and speeds of coronal mass ejections," *J. Geophys. Res.*, 106, 20947-20955, 2001.
