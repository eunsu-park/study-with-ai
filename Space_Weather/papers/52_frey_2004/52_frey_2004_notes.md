---
title: "Substorm Onset Observations by IMAGE-FUV"
authors: [Frey, Mende, Angelopoulos, Donovan]
year: 2004
journal: "Journal of Geophysical Research: Space Physics"
doi: "10.1029/2004JA010607"
topic: Space_Weather
tags: [substorm, onset, aurora, IMAGE-FUV, statistics, MLT, magnetic-latitude, IMF]
status: completed
date_started: 2026-04-25
date_completed: 2026-04-25
---

# 52. Substorm Onset Observations by IMAGE-FUV / IMAGE-FUV로 관측한 substorm onset

> Frey, H. U., Mende, S. B., Angelopoulos, V., and Donovan, E. F. (2004),
> *J. Geophys. Res.*, **109**, A10304, doi:10.1029/2004JA010607.

---

## 1. Core Contribution / 핵심 기여

**EN.** Frey et al. (2004) compile the first community-scale catalog of substorm onsets from continuous global UV imaging: **2437 onsets** identified from IMAGE-FUV WIC + SI-13 images between **May 2000 and December 31, 2002**, an order of magnitude larger than the previous best (Polar UVI, 648 onsets). They define an objective onset rule — local brightening + ≥ 20-min azimuthal expansion + ≥ 30-min separation from the previous onset — and tabulate, for each event, the date, time, instrument used, IMAGE geocentric distance, brightness, pixel coordinates, and geographic and AACGM (magnetic) coordinates of the brightest onset pixel. Their headline result establishes the canonical IMAGE-FUV substorm climatology: average onset MLT = **23:00 (median) / 22:30 (mean) ± 01:21** and AACGM latitude = **66.4° ± 2.86°**.

**KR.** Frey et al. (2004)는 IMAGE-FUV 위성의 WIC + SI-13 채널이 **2000년 5월부터 2002년 12월 31일까지** 관측한 글로벌 자외선 영상에서 **2437개의 substorm onset**을 식별한 최초의 학계 규모 카탈로그를 발표하였다. 이는 직전 최대 통계인 Polar UVI(648개)의 약 4배에 달하는 표본이다. 저자들은 객관적 onset 기준을 정의하였다: 국지적 밝아짐 + 20분 이상의 방위각 방향 확장 + 이전 onset 후 30분 이상 경과. 각 사건에 대해 날짜·시각·사용 instrument·IMAGE의 지심 거리·밝기·픽셀 좌표·지리적 좌표·AACGM(자기) 좌표를 기록하였다. 핵심 결과: 평균 onset MLT = **23:00 (중앙값) / 22:30 (평균) ± 01:21**, AACGM 자기위도 = **66.4° ± 2.86°**.

**EN. (continued).** Beyond the headline numbers, the paper documents quantitatively how onset location and brightness depend on IMF Bz, IMF By, IMF Bx, solar wind dynamic pressure, and season. They confirm Liou et al. (2001)'s seasonal MLT shift (summer onsets ≈ 1 hour earlier than winter), the equatorward shift of onset latitude under southward IMF, and the strong anti-correlation between dynamic pressure and onset latitude. Crucially, they also publish the entire onset list electronically, transforming substorm research from a series of small-N case studies into a large-N statistical and machine-learning enterprise.

**KR. (계속).** 헤드라인 수치를 넘어, 이 논문은 onset 위치와 밝기가 IMF Bz·By·Bx, 태양풍 동압력, 계절에 어떻게 정량적으로 의존하는지 정리한다. Liou et al. (2001)의 계절성 MLT 이동(여름 onset이 겨울보다 약 1시간 빠름), 남향 IMF에서의 onset 위도 적도쪽 이동, 동압력과 onset 위도 사이의 강한 음의 상관관계를 모두 확인한다. 결정적으로 전체 onset 리스트를 전자적으로 공개하여 substorm 연구를 소표본 사례 연구의 연속에서 대표본 통계·기계학습 연구로 전환시켰다.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction & Motivation / 서론과 동기

**EN.** Substorms are the most spectacular signature of solar wind–magnetosphere–ionosphere coupling: hundreds of GW released in the magnetotail, intense plasma flows, strong field-aligned currents, electromagnetic waves, and bright dynamic auroras. Akasofu (1964) framed the four-phase picture (growth–onset–expansion–recovery), and McPherron (1972) tied the growth phase to tail flux loading. Two competing onset models drive much of the controversy:
- **Current Disruption (CD)** model — onset begins near Earth at $r < 8 R_E$ via cross-field current instability (Lui et al. 1991), with auroral breakup occurring almost simultaneously.
- **Near-Earth Neutral Line (NENL)** model — reconnection occurs at $r \sim 15$–$25 R_E$ (Hones 1979), and auroral breakup follows when fast flows reach the inner magnetosphere (Shiokawa et al. 1998).

Distinguishing the two requires accurate, large-N onset timing relative to ground and in-situ data — exactly what this paper provides.

**KR.** Substorm은 태양풍-자기권-전리권 결합의 가장 화려한 표지로, 자기꼬리에서 수백 GW의 에너지 방출, 강력한 plasma flow, field-aligned current, 전자기파, 그리고 밝고 역동적인 오로라를 만든다. Akasofu(1964)가 4단계 모델(growth-onset-expansion-recovery)을 제안하였고, McPherron(1972)이 growth phase를 tail flux 축적으로 연결하였다. onset의 두 경쟁 모델:
- **CD (Current Disruption) 모델**: $r < 8 R_E$의 근지구 영역에서 cross-field current instability(Lui et al. 1991)로 onset이 시작되고 오로라 breakup이 거의 동시에 발생.
- **NENL (Near-Earth Neutral Line) 모델**: $r \sim 15$–$25 R_E$에서 reconnection이 발생하고(Hones 1979), 빠른 흐름이 내자기권에 도달했을 때 비로소 오로라 breakup이 발생(Shiokawa et al. 1998).

두 모델을 구별하려면 지상·in-situ 자료에 대한 정확한 대표본 onset 시각이 필요한데, 본 논문이 이를 제공한다.

**EN.** Earlier statistical samples were small: DE-1 (68 onsets, Craven & Frank 1991), Viking (133 onsets, Henderson & Murphree 1995), Polar UVI (648 onsets, Liou et al. 2001), IMAGE-FUV winter subset (78 onsets, Gérard et al. 2004), and IMAGE-FUV proton/electron subset (91 onsets, Mende et al. 2003). The Liou et al. study identified systematic IMF and seasonal effects — but with only ~100 events per quadrant, error bars were large.

**KR.** 기존 통계 표본은 작았다: DE-1(68개, Craven & Frank 1991), Viking(133개, Henderson & Murphree 1995), Polar UVI(648개, Liou et al. 2001), IMAGE-FUV 겨울 부분집합(78개, Gérard et al. 2004), IMAGE-FUV 양성자/전자 비교 부분집합(91개, Mende et al. 2003). Liou et al.은 IMF와 계절 효과를 발견했으나 사분위당 100개 정도 사례로는 오차가 컸다.

### Part II: Instrumentation & Coordinates / 기기와 좌표계

**EN.** IMAGE was launched in March 2000 into a highly elliptical polar orbit (1000 × 45,600 km, 14:14 h period) with apogee initially over the North Pole. The FUV instrument package (Mende et al. 2000) contains three imagers; this paper uses the **Wideband Imaging Camera (WIC)** (140–180 nm, primarily LBH N₂ + 135.6 nm OI, 50 km pixel from apogee, electron-precipitation aurora) and the **Spectrographic Imager 135.6 nm channel (SI-13)** (100 km pixel, also electron aurora). Both image the aurora once per 2-minute spin. The SI-12 proton channel is *not* used because Mende et al. (2003) showed that proton and electron onsets occupy the same MLT/latitude statistically.

**KR.** IMAGE는 2000년 3월에 1000 × 45,600 km, 14시간 14분 주기의 극궤도로 발사되었고, 발사 시 apogee는 북극 위였다. FUV 장비(Mende et al. 2000)는 세 개의 imager로 구성되며 본 논문은 **WIC**(140–180 nm, 주로 LBH N₂ + 135.6 nm OI, apogee 기준 50 km 픽셀, 전자 강하 오로라)와 **SI-13**(135.6 nm OI 채널, 100 km 픽셀, 전자 오로라)를 사용한다. 두 카메라 모두 2분 spin period마다 한 번씩 오로라를 촬영한다. SI-12(양성자 채널)는 사용하지 않는데, Mende et al.(2003)이 양성자와 전자 onset의 MLT/위도 분포가 통계적으로 동일함을 보였기 때문이다.

**EN.** Each image is geo-referenced using bright UV stars crossing the field of view (Frey et al. 2003). The residual pointing error is up to 4 pixels in the spin plane and 2 pixels perpendicular, giving largest MLT uncertainty in summer/winter and largest latitude uncertainty in spring/fall. Pixel positions are converted to geographic latitude/longitude and then to AACGM (Altitude-Adjusted Corrected GeoMagnetic) coordinates assuming an emission altitude of 110 km.

**KR.** 각 영상은 밝은 자외선 별이 시야를 가로지르는 것을 이용하여 위치보정된다(Frey et al. 2003). 잔차 pointing 오차는 spin 평면에서 최대 4 픽셀, 수직 방향에서 최대 2 픽셀이며, 결과적으로 여름·겨울에는 MLT 오차가, 봄·가을에는 위도 오차가 최대가 된다. 픽셀 위치는 110 km의 발광 고도를 가정하여 지리 좌표로 변환된 뒤 AACGM(고도 보정된 보정 자기) 좌표로 변환된다.

**EN.** Operational caveats: FUV is turned off during radiation-belt passes, leaving 8–10 hours of useful imaging per orbit. Near apogee around the equator (post-2003), aurora appears near the Earth's limb and location accuracy degrades — this becomes important for the 2006 follow-up paper but does not affect the 2000–2002 sample of this paper.

**KR.** 운용 제약: 방사선 벨트 통과 시 FUV는 OFF되므로 궤도당 유효 관측 시간은 8–10시간이다. 적도 부근 apogee(2003년 이후)에서는 오로라가 limb 근처에 위치하여 위치 정확도가 저하되지만, 이는 후속 2006년 논문에서 중요해지며 본 논문의 2000–2002 표본에는 영향을 주지 않는다.

### Part III: Onset Identification Algorithm / onset 식별 알고리즘

**EN.** The authors search through the ~3-year FUV image stream and accept an event as a *substorm onset* only if **all three criteria** are satisfied:

1. **Local brightening** — A clear, localized increase in WIC/SI-13 intensity must occur over a small auroral patch (typically a few hundred km wide).
2. **Azimuthal expansion ≥ 20 min** — The brightened region must spread along the auroral oval (in MLT) for at least 20 minutes; this filters out pseudo-breakups and small intensifications.
3. **30-min separation** — At least 30 minutes must have elapsed since the previous accepted onset; this avoids double-counting intensifications within a single expansion phase.

Within the first onset image, an analyst visually marks the center of the auroral bulge, after which a computer routine finds the brightest pixel near that point and records its geographic and AACGM coordinates. The output is a tabular catalog (one row per onset).

**KR.** 저자들은 약 3년치 FUV 영상 스트림을 검색하여, **세 조건이 모두 충족되는** 사건만 substorm onset으로 받아들인다:

1. **국지적 밝아짐** — WIC/SI-13 강도가 작은 오로라 패치(보통 수백 km 폭)에서 명확하고 국지적으로 증가해야 한다.
2. **20분 이상의 방위각 방향 확장** — 밝아진 영역이 오로라 oval을 따라(MLT 방향) 최소 20분간 확장되어야 한다. 이 조건은 pseudo-breakup이나 작은 intensification을 걸러낸다.
3. **30분 분리** — 이전 onset 이후 최소 30분이 경과해야 한다. 이는 단일 expansion phase 내의 intensification을 별개 onset으로 중복 집계하는 것을 방지한다.

첫 onset 영상 내에서 분석자가 auroral bulge의 중심을 시각적으로 표시하면, 컴퓨터 루틴이 그 근방의 가장 밝은 픽셀을 찾아 지리·AACGM 좌표를 기록한다. 결과물은 onset 한 개당 한 행인 표 형식 카탈로그이다.

**EN.** Each row in the published list contains: date and UT time of onset, instrument used (WIC/SI-13), IMAGE geocentric distance (R_E), brightness in instrument counts, image pixel (x, y), geographic latitude/longitude, and AACGM latitude/MLT. The full file is hosted at http://sprg.ssl.berkeley.edu/image/ and is searchable by, e.g., high latitude, late MLT, proximity to a specific ground station, or proximity to IMAGE.

**KR.** 공개 리스트의 각 행은 다음 항목을 포함한다: onset 날짜와 UT 시각, 사용 instrument(WIC/SI-13), IMAGE 지심 거리 (R_E), instrument count 단위 밝기, 영상 픽셀 좌표 (x, y), 지리 위도/경도, AACGM 위도/MLT. 전체 파일은 http://sprg.ssl.berkeley.edu/image/에 호스팅되며 고위도·늦은 MLT·특정 지상 관측소 인근·IMAGE 위성 인근 등으로 검색 가능하다.

### Part IV: Statistical Results / 통계 결과

#### IV.1 MLT and Magnetic Latitude Distribution / MLT와 자기위도 분포

**EN.** The histograms of all 2437 onsets show:
- **MLT**: distribution peaks at 23:00, with FWHM of about 4 hours (roughly 21:00–01:00 contains the bulk of events). Median MLT = **23:00**, mean = **22:30**, std = **01:21 (1.35 h)**.
- **Magnetic latitude**: roughly Gaussian, peaked at 67° with a long equatorward tail. Median = **66.4°**, mean = **66.1°**, std = **2.86°**. Almost no events poleward of 72° or equatorward of 60°.
- **Geomagnetic longitude**: roughly flat (no preferred longitude), confirming that onsets are organized by *magnetic local time*, not by geographic location.

**KR.** 2437개 onset의 히스토그램은 다음을 보여준다:
- **MLT**: 분포는 23:00에서 정점을 보이며 FWHM은 약 4시간(대략 21:00–01:00 내에 대부분 포함). 중앙값 = **23:00**, 평균 = **22:30**, 표준편차 = **01:21 (1.35시간)**.
- **자기위도**: 대략 가우시안 분포, 67°에서 정점이며 적도쪽으로 긴 꼬리. 중앙값 = **66.4°**, 평균 = **66.1°**, 표준편차 = **2.86°**. 72° 극쪽이나 60° 적도쪽에는 사례가 거의 없다.
- **자기 경도**: 거의 평탄(특정 경도 선호 없음). 이는 onset이 *MLT*로 조직되고 지리적 위치로는 조직되지 않음을 확인.

**EN.** The Frey et al. (2004) numbers agree remarkably well with all earlier studies (Table 1 below), which is itself a key result — it shows that despite different instruments, sensitivities, and small samples, the *average* substorm onset location is robust.

**KR.** Frey et al.(2004)의 수치는 이전 모든 연구와 매우 잘 일치하며(아래 표 1 참조), 이 일치 자체가 핵심 결과이다 — 서로 다른 기기·감도·작은 표본에도 불구하고 *평균* substorm onset 위치가 강건함을 보여준다.

| Satellite / 위성 | # Onsets / onset 수 | Median MLT (mean) / MLT 중앙값 (평균) | Median MLAT (mean) / 자기위도 중앙값 (평균) | Reference / 참고 |
|---|---:|:---:|:---:|---|
| DE-1 | 68 | 22:50 (22:48) | 65° (?) | Craven & Frank 1991 |
| Viking | 133 | 23:05 (22:48) | 66.7° (65.8°) | Henderson & Murphree 1995 |
| Polar UVI | 648 | 22:30 (22:42) | 67.0° (66.6°) | Liou et al. 2001 |
| IMAGE (winter only) | 78 | 23:24 | 65.6° | Gérard et al. 2004 |
| **IMAGE (this paper)** | **2437** | **23:00 (23:00)** | **66.4° (66.1°)** | **Frey et al. 2004** |

#### IV.2 Seasonal Modulation / 계절 변조

**EN.** Following Liou et al. (2001), Frey et al. (2004) bin onsets by season:
- **Summer (May–Aug.)**: onset MLT shifts **~1 hour earlier** (toward dusk, ≈ 22:00) and onset latitude shifts **~1.5°–2° poleward**. Reason: summer hemisphere has higher ionospheric conductivity from solar EUV, reducing the field-aligned voltage required to close the substorm current wedge, so the unstable region maps to a more poleward, earlier-MLT footpoint.
- **Winter (Nov.–Feb.)**: onset MLT is closer to magnetic midnight (≈ 23:30–00:00) and onset latitude is **~1.5°–2° lower** (more equatorward).
- **Equinoxes**: intermediate values, often the noisiest because of orbital geometry.

**KR.** Liou et al.(2001)을 따라 Frey et al.(2004)은 onset을 계절별로 분류한다:
- **여름(5–8월)**: onset MLT가 **약 1시간 이른** 시각(저녁쪽, 약 22:00)으로 이동하고, onset 위도는 **약 1.5°–2° 극쪽**으로 이동. 원인: 여름 반구는 태양 EUV로 인해 전리권 전도도가 높아 substorm current wedge를 닫는 데 필요한 field-aligned 전압이 줄어들고, 따라서 불안정 영역의 footpoint가 더 극쪽·이른 MLT로 매핑된다.
- **겨울(11–2월)**: onset MLT가 자기 자정에 가깝고(약 23:30–00:00), onset 위도는 **약 1.5°–2° 낮다**(더 적도쪽).
- **춘추분**: 중간 값. 궤도 기하 때문에 가장 noisy하다.

#### IV.3 IMF Bz Dependency / IMF Bz 의존성

**EN.** Onset latitude vs. IMF Bz (1-hour averaged before onset):
- **Bz < 0 (southward)**: onsets concentrate at lower latitudes (60°–66°). Median latitude shifts **~1°–3°** equatorward as Bz decreases from 0 to –10 nT. This is consistent with the standard picture: southward IMF drives reconnection, opens flux, inflates the polar cap, and pushes the auroral oval equatorward.
- **Bz > 0 (northward)**: onsets occur at higher latitudes (67°–70°), often associated with quiet conditions and pseudo-breakup-like events that nonetheless meet the 20-min/30-min criteria.
- **Bz ≈ 0**: tightest cluster around 66°–67°.

**KR.** onset 위도 vs. IMF Bz (onset 직전 1시간 평균):
- **Bz < 0 (남향)**: onset이 저위도(60°–66°)에 몰린다. Bz가 0에서 –10 nT로 감소함에 따라 중앙 위도가 **약 1°–3°** 적도쪽으로 이동. 이는 표준 그림과 일치한다: 남향 IMF가 reconnection을 구동하여 자속을 열고 polar cap을 팽창시키며, 오로라 oval을 적도쪽으로 밀어낸다.
- **Bz > 0 (북향)**: onset이 고위도(67°–70°)에서 발생. 흔히 조용한 조건이나 pseudo-breakup 유사 사건이 20분/30분 기준을 만족하는 경우.
- **Bz ≈ 0**: 66°–67° 부근에 가장 좁게 집중.

#### IV.4 IMF By and Bx Dependency / IMF By와 Bx 의존성

**EN.** IMF By controls onset MLT through magnetic-tension-induced asymmetry in tail-lobe pressure:
- **By > 0 (positive)**: onsets shift **~30 min earlier in MLT** (toward dusk).
- **By < 0 (negative)**: onsets shift **~30 min later in MLT** (toward dawn).

IMF Bx — although weakly coupled in classical Dungey-cycle pictures — also shows a significant effect, consistent with Liou et al. (2001):
- **Bx > 0**: onset latitude is *lower* than for Bx < 0.
- **Bx < 0**: onset latitude is *higher*.

The Bx effect arises because IMF Bx tilts the bow-shock and distorts the open–closed field-line boundary asymmetrically.

**KR.** IMF By는 자기 장력으로 인한 tail lobe 압력 비대칭을 통해 onset MLT를 조절한다:
- **By > 0 (양)**: onset이 **MLT 기준 약 30분 이른** 쪽(저녁쪽)으로 이동.
- **By < 0 (음)**: onset이 **MLT 기준 약 30분 늦은** 쪽(새벽쪽)으로 이동.

IMF Bx는 고전적인 Dungey cycle 그림에서는 결합이 약하지만, Liou et al. (2001)과 일치하는 유의미한 효과를 보인다:
- **Bx > 0**: onset 위도가 Bx < 0일 때보다 *낮다*.
- **Bx < 0**: onset 위도가 *높다*.

Bx 효과는 IMF Bx가 bow shock을 기울여 open-closed field line 경계를 비대칭적으로 왜곡시키기 때문에 발생한다.

#### IV.5 Solar Wind Dynamic Pressure / 태양풍 동압력

**EN.** Following Gérard et al. (2004), Frey et al. (2004) confirm a strong **anti-correlation between $P_{dyn}$ and onset latitude**:
$$
\Lambda_{\text{onset}} \approx 67° - 1.5\log_{10}(P_{dyn}/1\text{ nPa})
$$
A 10× increase in dynamic pressure (1 → 10 nPa) lowers onset latitude by roughly 1.5° (e.g., from 67° to 65.5°). The interpretation: high $P_{dyn}$ compresses the magnetosphere, pushing the inner edge of the plasma sheet earthward and mapping the onset region equatorward.

**KR.** Gérard et al.(2004)을 따라 Frey et al.(2004)은 **$P_{dyn}$과 onset 위도 사이의 강한 음의 상관관계**를 확인한다:
$$
\Lambda_{\text{onset}} \approx 67° - 1.5\log_{10}(P_{dyn}/1\text{ nPa})
$$
동압력이 10배 증가하면(1 → 10 nPa) onset 위도가 약 1.5° 낮아진다(예: 67° → 65.5°). 해석: 높은 $P_{dyn}$이 자기권을 압축하여 plasma sheet 내연을 지구쪽으로 밀고, onset 영역의 footprint를 적도쪽으로 매핑한다.

#### IV.6 Substorm Rate / Substorm 발생률

**EN.** From 2437 onsets over ~32 months of FUV operation (with effective duty cycle ~30% accounting for radiation-belt outage and limb geometry), the average substorm rate works out to **≈ 4–5 substorms per day** when integrated over both hemispheres — in good agreement with ground-magnetometer-based AE-index counts.

**KR.** 약 32개월의 FUV 운영 기간 동안 2437개 onset(방사선 벨트 outage와 limb 기하를 고려한 유효 duty cycle 약 30%)으로부터, 양반구 통합 평균 substorm 발생률은 **하루에 약 4–5개**로 계산되며 지상 자력계 기반 AE-index 통계와 잘 일치한다.

### Part V: Discussion & Database Release / 논의와 데이터베이스 공개

**EN.** Frey et al. emphasize three discussion points:
1. **Robust mean location** — Despite IMF-, season-, and pressure-driven shifts of individual onsets, the population mean (23:00 MLT, 66.4°) is reproduced by every previous mission. This suggests internal magnetospheric processes (Harang discontinuity location, plasma-sheet inner edge) set the average.
2. **External vs internal control** — IMF and solar wind clearly modulate individual onsets, but only at the ±1 hour MLT and ±2° latitude level. The bulk distribution is "internally controlled."
3. **Conjugacy** — The 2000–2002 IMAGE dataset is northern-only; the 2006 ICS-8 follow-up adds the southern hemisphere and shows the same average location, confirming statistical (but not event-by-event) conjugacy.

**KR.** Frey et al.은 세 가지 논점을 강조한다:
1. **강건한 평균 위치** — 개별 onset이 IMF·계절·압력 변화에 따라 이동하지만, 모집단 평균(23:00 MLT, 66.4°)은 모든 이전 mission에서 재현된다. 이는 내부 자기권 과정(Harang discontinuity 위치, plasma sheet 내연)이 평균을 결정함을 시사한다.
2. **외부 vs 내부 제어** — IMF와 태양풍은 개별 onset을 명확히 변조하지만, ±1시간 MLT와 ±2° 위도 수준에 그친다. 분포 전체는 "내부 제어"된다.
3. **Conjugacy** — 2000–2002 IMAGE 자료는 북반구뿐이지만, 2006 ICS-8 후속 연구가 남반구를 추가하여 동일한 평균 위치를 확인하며, 통계적(사건별이 아닌) conjugacy를 입증한다.

**EN.** The published electronic list is the paper's most enduring contribution. As the authors note, it can be searched for high-MLAT events, late-MLT events, ground-station-conjunction events, or events close to IMAGE for high-resolution case studies. By 2024 the list has been cited in hundreds of papers and used as ground-truth for THEMIS, SuperDARN, GOES, Cluster, MMS, and ML-based onset detectors.

**KR.** 공개된 전자 리스트는 본 논문의 가장 오래 지속되는 기여이다. 저자들이 언급한 대로, 고MLAT 사건·늦은 MLT 사건·지상 관측소 conjunction 사건·IMAGE 인근 고해상도 사례 연구용 사건 등 다양하게 검색 가능하다. 2024년 기준 본 리스트는 수백 편의 논문에서 인용되었고 THEMIS, SuperDARN, GOES, Cluster, MMS, ML 기반 onset 검출기의 ground truth로 사용되고 있다.

---

## 3. Key Takeaways / 핵심 시사점

1. **Largest IMAGE-FUV onset catalog (2437 events) / 최대 규모 IMAGE-FUV onset 카탈로그 (2437개)** — EN. Quadrupled the previous best (Polar UVI, 648 onsets), enabling the first high-statistics binning by IMF and season simultaneously. KR. 직전 최대(Polar UVI, 648개)의 약 4배. IMF와 계절을 동시에 분할해도 통계가 충분한 최초의 데이터셋.

2. **Canonical onset location: 23:00 MLT, 66.4° MLAT / 표준 onset 위치: 23:00 MLT, 66.4° 자기위도** — EN. This pre-midnight, sub-auroral pair has become the literature reference, reproduced by every mission within ±10 min of MLT and ±1° of latitude. KR. 이 pre-midnight·sub-auroral 위치는 학계 기준값이 되었으며, 모든 이후 mission에서 MLT 기준 ±10분, 위도 기준 ±1° 이내로 재현된다.

3. **Three-rule onset detection algorithm / 3-rule onset 검출 알고리즘** — EN. Local brightening + ≥ 20-min azimuthal expansion + ≥ 30-min separation provides a reproducible criterion that has been adopted by SuperMAG (Newell & Gjerloev 2011) and ML-based detectors as a definition standard. KR. 국지적 밝아짐 + 20분 이상 방위각 방향 확장 + 30분 분리는 재현 가능한 기준으로, SuperMAG(Newell & Gjerloev 2011)와 ML 기반 검출기들이 정의 표준으로 채택하였다.

4. **IMF Bz controls latitude, IMF By controls MLT / IMF Bz는 위도, IMF By는 MLT 제어** — EN. Southward Bz pushes onsets ~1°–3° equatorward; positive By shifts onsets ~30 min earlier in MLT. These trends scale linearly with the IMF magnitude in the tested range (±10 nT). KR. 남향 Bz는 onset을 약 1°–3° 적도쪽으로 밀고, 양의 By는 MLT 기준 약 30분 이른 쪽으로 이동시킨다. 시험 범위(±10 nT)에서 IMF 크기에 선형 scale.

5. **Dynamic-pressure anti-correlation / 동압력 음의 상관관계** — EN. $\Lambda_{\text{onset}} \propto -1.5\log_{10}(P_{dyn})$. A 10× pressure increase lowers latitude by ~1.5°, confirming Gérard et al. (2004). KR. $\Lambda_{\text{onset}} \propto -1.5\log_{10}(P_{dyn})$. 압력이 10배 증가하면 위도가 약 1.5° 낮아진다 (Gérard et al. 2004 확인).

6. **Seasonal MLT shift of ~1 hour / 약 1시간의 계절성 MLT 이동** — EN. Summer onsets occur 1 hour earlier in MLT and 1.5° more poleward than winter onsets, consistent with conductivity-modulated current-wedge geometry. KR. 여름 onset은 겨울보다 MLT 기준 1시간 빠르고 1.5° 더 극쪽이며, 전도도 변조 current wedge 기하와 일치.

7. **Internal control dominates the population mean / 모집단 평균은 내부 제어가 지배** — EN. External (IMF, solar wind) influences are real but average out: the population mean is set by internal magnetospheric processes (Harang discontinuity, plasma-sheet inner-edge mapping). KR. 외부(IMF, 태양풍) 영향은 실재하지만 평균 시 상쇄된다: 모집단 평균은 내부 자기권 과정(Harang discontinuity, plasma sheet 내연 매핑)이 결정.

8. **Public database catalyzed substorm research / 공개 데이터베이스가 substorm 연구를 가속** — EN. Hosting the full list at sprg.ssl.berkeley.edu transformed substorm science from small case studies to large-N ML-ready research, including THEMIS conjunction planning and modern CNN/LSTM onset detectors. KR. 전체 리스트를 sprg.ssl.berkeley.edu에 호스팅함으로써 substorm 과학을 소규모 사례 연구에서 대표본 ML-ready 연구로 전환시켰다(THEMIS conjunction 계획, 최신 CNN/LSTM onset 검출기 포함).

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Onset detection rule (formalized) / onset 검출 규칙 (수식화)

Let $I(\vec{x}, t)$ be the WIC/SI-13 image intensity at pixel $\vec{x}$ and time $t$. Define a "bright pixel set" via a sliding-window threshold:

$$
B(t) = \left\{ \vec{x}\,\big|\, I(\vec{x}, t) > I_{\text{bg}}(\vec{x}, t) + n\sigma \right\}
$$

where $I_{\text{bg}}$ is a 30-minute-running-mean background and $n \sim 3$.

A candidate onset at time $t_0$ requires:

$$
\begin{aligned}
\text{(i)} \quad &\exists\, \vec{x}_0 \in \text{nightside oval} : I(\vec{x}_0, t_0) > I_{\text{bg}} + n\sigma, \\
\text{(ii)} \quad &\Delta\phi_{B(t)} \big|_{t_0}^{t_0 + 20\,\text{min}} \geq \Delta\phi_{\min}, \\
\text{(iii)} \quad &t_0 - t_{\text{prev. onset}} \geq 30\,\text{min},
\end{aligned}
$$

where $\Delta\phi_{B(t)}$ is the MLT extent of $B(t)$ and $\Delta\phi_{\min}$ is a few hours of MLT.

### 4.2 Brightest-pixel localization / 가장 밝은 픽셀 결정

Within a manually selected ROI around the onset bulge:

$$
\vec{x}^* = \arg\max_{\vec{x}\in\text{ROI}} I(\vec{x}, t_0)
$$

The (geographic, magnetic) coordinates of $\vec{x}^*$ are recorded as the onset location.

### 4.3 Coordinate transforms / 좌표 변환

Pixel → geographic:
$$
(x_{\text{pix}}, y_{\text{pix}}) \xrightarrow{\text{IMAGE attitude}} (\lambda_{\text{geo}}, \phi_{\text{geo}})
$$

Geographic → AACGM:
$$
(\lambda_{\text{geo}}, \phi_{\text{geo}}) \xrightarrow{\text{IGRF + emission alt. 110 km}} (\Lambda_{\text{AACGM}}, \text{MLT})
$$

### 4.4 Statistical descriptors / 통계 기술

Let $\{(\Lambda_i, \text{MLT}_i)\}_{i=1}^{N=2437}$ be the catalog. The paper reports:

$$
\widetilde{\Lambda} = 66.4°,\quad \overline{\Lambda} = 66.1°,\quad \sigma_\Lambda = 2.86°
$$

$$
\widetilde{\text{MLT}} = 23\!:\!00,\quad \overline{\text{MLT}} = 22\!:\!30,\quad \sigma_{\text{MLT}} = 01\!:\!21
$$

(tilde = median, bar = mean)

### 4.5 Linear regressions found in the paper / 논문에서 보고된 선형 회귀

| Predictor / 예측 변수 | Response / 반응 변수 | Approximate slope / 근사 기울기 |
|---|---|---|
| IMF Bz (nT) | $\Lambda$ (deg) | $\partial\Lambda/\partial B_z \approx +0.2°/\text{nT}$ for Bz > 0; ≈ +0.3°/nT for Bz < 0 |
| IMF By (nT) | MLT (h) | $\partial \text{MLT}/\partial B_y \approx -3$ min/nT (positive By → earlier MLT) |
| $\log_{10}(P_{dyn}/\text{nPa})$ | $\Lambda$ (deg) | $\partial\Lambda/\partial \log P_{dyn} \approx -1.5°$ |
| Season (months from summer solstice) | MLT (h) | $\partial \text{MLT}/\partial \text{season} \approx +0.17$ h/month (summer earlier) |

### 4.6 Worked numerical example / 수치 예제

**Question.** Predict the onset MLT and latitude for a substorm in **January** (winter), with IMF Bz = –4 nT, By = +3 nT, $P_{dyn}$ = 4 nPa.

**Solution / 풀이.**

Start from baseline: $\overline{\Lambda} = 66.4°$, $\overline{\text{MLT}} = 23\!:\!00$.

- Bz contribution: $\Delta\Lambda_{Bz} \approx 0.3 \times (-4) = -1.2°$.
- $P_{dyn}$ contribution: $\Delta\Lambda_P \approx -1.5 \log_{10}(4/1) \approx -1.5 \times 0.602 \approx -0.9°$.
- Predicted latitude: $\Lambda \approx 66.4 - 1.2 - 0.9 = \mathbf{64.3°}$.

- By contribution: $\Delta\text{MLT}_{By} \approx -3 \times 3 = -9$ min (earlier).
- Winter season: roughly +0.5 h later than annual mean.
- Predicted MLT: $\text{MLT} \approx 23\!:\!00 - 0\!:\!09 + 0\!:\!30 \approx \mathbf{23\!:\!21}$.

**Interpretation.** A storm-time southward Bz with elevated dynamic pressure pushes the substorm onset down to ~64° (auroral oval expansion). Winter shifts MLT slightly later, while positive By shifts it slightly earlier, partially canceling.

**해석.** 폭풍 시 남향 Bz와 높은 동압력은 onset을 약 64°까지 적도쪽으로 밀어 내린다(오로라 oval 팽창). 겨울은 MLT를 약간 늦게 만들고, 양의 By는 약간 이르게 만들어 부분적으로 상쇄된다.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1964 ─ Akasofu: 4-phase substorm (growth/onset/expansion/recovery)
1972 ─ McPherron: growth phase, tail flux loading
1979 ─ Hones: NENL model
1991 ─ Lui et al.: Current Disruption (CD) instability
1991 ─ Craven & Frank (DE-1): 68-onset statistics
1995 ─ Henderson & Murphree (Viking): 133-onset statistics
2000 ─ IMAGE launch; FUV starts operation
2001 ─ Liou et al. (Polar UVI): 648 onsets, IMF + seasonal effects
2003 ─ Mende, Frey et al.: proton vs electron substorm
2004 ─ ★ FREY et al. (this paper): 2437 IMAGE-FUV onsets, public catalog
2004 ─ Gérard et al.: dynamic-pressure control of onset
2004 ─ Østgaard et al.: conjugate-hemisphere asymmetry from IMAGE+Polar
2005 ─ Wang et al.: solar illumination effect
2006 ─ Frey & Mende (ICS-8): southern-hemisphere extension, +1755 onsets
2007 ─ THEMIS launch; uses Frey list for conjunctions
2011 ─ Newell & Gjerloev: SuperMAG SML index calibrated against Frey
2018+ ─ ML onset detectors (CNN/LSTM) trained on Frey + SuperMAG lists
2020s ─ Frey list still cited as ground truth in modern substorm climatology
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Akasofu (1964) | Defines the 4-phase substorm framework that this paper's onset criteria implement (the 30-min separation is roughly Akasofu's expansion-phase duration). / 본 논문의 onset 기준이 구현하는 4단계 substorm 틀을 정의. (30분 분리는 Akasofu의 expansion phase 지속시간에 해당.) | Foundational / 기반 |
| Liou et al. (2001) | Polar UVI 648-onset study; first to find IMF Bz/By/Bx and seasonal effects. Frey et al. confirms and refines all of these with 4× the events. / Polar UVI 648개 onset 연구; IMF Bz/By/Bx 및 계절 효과를 처음 발견. Frey et al.이 4배 이상의 사례로 모두 확인·정밀화. | Direct predecessor / 직접 선행 |
| Mende et al. (2003) | Justifies neglecting the proton (SI-12) channel by showing electron and proton onsets occupy the same MLT/MLAT statistically. / 양성자(SI-12) 채널을 무시하는 근거를 제공: 양성자와 전자 onset이 통계적으로 동일한 MLT/MLAT를 차지함을 보임. | Method justification / 방법론 정당화 |
| Gérard et al. (2004) | Establishes the strong $P_{dyn}$ → onset latitude anti-correlation; Frey et al. confirms with the larger sample. / 강한 $P_{dyn}$ → onset 위도 음의 상관관계 확립; Frey et al.이 더 큰 표본으로 확인. | Companion paper / 동반 논문 |
| Østgaard et al. (2004) | Uses 5 IMAGE+Polar conjugate events to measure inter-hemispheric asymmetry; complements Frey et al.'s statistical view. / IMAGE+Polar 5쌍 conjugate 사건으로 양반구 비대칭 측정; Frey et al.의 통계적 시각을 보완. | Complementary / 보완적 |
| Frey & Mende (2006, ICS-8) | Direct follow-up: adds 1755 southern-hemisphere onsets and confirms population-mean reproducibility. / 직접 후속 연구: 남반구 onset 1755개 추가; 모집단 평균 재현성 확인. | Direct sequel / 직접 후속 |
| Newell & Gjerloev (2011) | SuperMAG SME/SML index calibration uses Frey list as ground truth for substorm onset definition. / SuperMAG SME/SML 지수 보정에서 Frey 리스트를 substorm onset 정의의 ground truth로 사용. | Modern application / 현대적 응용 |
| Wang et al. (2005) | Uses Frey list to investigate solar-illumination control on onset latitude. / Frey 리스트를 사용하여 onset 위도에 대한 태양광 조사 효과를 분석. | Downstream user / 후속 활용 |

---

## 7. References / 참고문헌

- **Frey, H. U., Mende, S. B., Angelopoulos, V., and Donovan, E. F.**, "Substorm onset observations by IMAGE-FUV," *J. Geophys. Res.*, **109**, A10304, 2004. [doi:10.1029/2004JA010607]
- Akasofu, S.-I., "The development of the auroral substorm," *Planet. Space Sci.*, **12**, 273, 1964.
- Craven, J. D. and Frank, L. A., "Diagnosis of auroral dynamics using global auroral imaging…," in *Auroral Physics*, Cambridge Univ. Press, pp. 273–297, 1991.
- Frey, H. U. and Mende, S. B., "Substorm onsets as observed by IMAGE-FUV," *Int. Conf. Substorms-8*, 71–75, 2006.
- Gérard, J.-C. et al., "Solar wind control of auroral substorm onset locations observed with the IMAGE-FUV imagers," *J. Geophys. Res.*, **109**, A03208, 2004. [doi:10.1029/2003JA010129]
- Henderson, M. G. and Murphree, J. S., "Comparison of Viking onset locations with the predictions of the thermal catastrophe model," *J. Geophys. Res.*, **100**, 1857, 1995.
- Hones, E. W., "Plasma flows in the magnetotail and its implications for substorm theories," in *Dynamics of the Magnetosphere*, D. Reidel, pp. 545–562, 1979.
- Liou, K., Newell, P. T., Sibeck, D. G., Meng, C.-I., Brittnacher, M., and Parks, G., "Observation of IMF and seasonal effects in the location of auroral substorm onset," *J. Geophys. Res.*, **106**, 5799, 2001.
- Lui, A. T. Y. et al., "A cross-field current instability for substorm expansion," *J. Geophys. Res.*, **96**, 11389, 1991.
- McPherron, R. L., "Substorm related changes in the geomagnetic tail: the growth phase," *Planet. Space Sci.*, **20**, 1521, 1972.
- Mende, S. B. et al., "Far ultraviolet imaging from the IMAGE spacecraft," *Space Sci. Rev.*, **91**, 287, 2000.
- Mende, S. B., Frey, H. U., Morsony, B. J., and Immel, T. J., "Statistical behavior of proton and electron auroras during substorms," *J. Geophys. Res.*, **108**, 1339, 2003. [doi:10.1029/2002JA009751]
- Newell, P. T. and Gjerloev, J. W., "Evaluation of SuperMAG auroral electrojet indices as indicators of substorms and auroral power," *J. Geophys. Res.*, **116**, A12211, 2011.
- Østgaard, N. et al., "Interplanetary magnetic field control of the location of substorm onset and auroral features in the conjugate hemispheres," *J. Geophys. Res.*, **109**, A07204, 2004. [doi:10.1029/2003JA010370]
- Shiokawa, K., Haerendel, G., and Baumjohann, W., "Azimuthal pressure gradient as driving force of substorm currents," *Geophys. Res. Lett.*, **25**, 959, 1998.
- Wang, H., Lühr, H., Ma, S. Y., and Ritter, P., "Statistical study of the substorm onset: its dependence on solar wind parameters and solar illumination," *Ann. Geophys.*, **23**, 2069, 2005.
