---
title: "Improvement in the Prediction of Solar Wind Conditions Using Near-Real Time Solar Magnetic Field Updates"
authors: C. Nick Arge, Victor J. Pizzo
year: 2000
journal: "Journal of Geophysical Research"
doi: "10.1029/1999JA000262"
topic: Space_Weather
tags: [WSA model, solar wind prediction, PFSS, flux tube expansion, synoptic maps, space weather forecasting]
status: completed
date_started: 2026-04-17
date_completed: 2026-04-17
---

# 18. Improvement in the Prediction of Solar Wind Conditions Using Near-Real Time Solar Magnetic Field Updates / 근 실시간 태양 자기장 업데이트를 이용한 태양풍 조건 예측 개선

---

## 1. Core Contribution / 핵심 기여

이 논문은 Wang-Sheeley (WS) 태양풍 예측 모델에 여러 가지 중요한 수정을 가하여, 현재 **WSA(Wang-Sheeley-Arge) 모델**로 알려진 개선된 경험적 태양풍 예측 체계를 확립합니다. 핵심 개선 사항은 다음과 같습니다: (1) flux tube expansion factor ($f_s$)와 태양풍 속도를 연결하는 **연속적 경험 함수**를 도입하여 source surface에서의 속도를 직접 지정 (기존 WS 모델은 지구에서의 속도를 기반으로 fitting), (2) source surface에서 지구까지 태양풍을 **방사형 스트림 전파** 방식으로 보내면서 스트림 간 상호작용을 고려하는 단순 scheme 도입, (3) WSO magnetogram의 품질을 평가하는 **품질 관리 시스템** ($Q$ factor) 개발, (4) **solar $b$ angle에 따른 극 자기장 보정**, (5) **일일 갱신(daily updated) synoptic map** 구축 기법 개발. 3년간의 WIND 위성 데이터와 비교 검증한 결과, 가장 개선된 방법(MDU)은 태양풍 속도를 10-15% 이내로 예측하고, IMF 극성을 ~75% 정확도로 예측함을 보였습니다.

This paper establishes the improved empirical solar wind prediction scheme now known as the **WSA (Wang-Sheeley-Arge) model** by making several significant modifications to the original Wang-Sheeley (WS) model. The key improvements are: (1) a **continuous empirical function** relating flux tube expansion factor ($f_s$) to solar wind velocity, applied at the source surface rather than at Earth (as in the original WS model), (2) a simple **radial stream propagation scheme** from the source surface to Earth that accounts for stream-stream interactions, (3) a **quality control system** ($Q$ factor) for identifying and removing problematic WSO magnetograms, (4) **polar field corrections for solar $b$ angle variations**, and (5) development of **daily updated synoptic map** construction techniques. Validated against 3 years of WIND satellite data, the most improved method (MDU) predicts solar wind speed to within 10-15% and IMF polarity with ~75% accuracy.

---

## 2. Reading Notes / 읽기 노트

### Section 1: Introduction / 서론

논문은 Skylab 시대 이래 coronal holes과 고속 태양풍 스트림의 관계가 확립된 역사를 개관합니다. PFSS 모델을 이용한 coronal 자기장 매핑에서 시작하여, Wang & Sheeley (1990)의 핵심 발견 — flux tube expansion factor와 태양풍 속도의 반비례 관계 — 을 소개합니다.

The paper reviews the history from the Skylab era, when coronal holes were linked to high-speed solar wind streams. Starting from coronal magnetic field mapping using the PFSS model, it introduces the key finding of Wang & Sheeley (1990): the inverse relationship between flux tube expansion factor and solar wind speed.

주요 선행 연구:
- Nolte et al. (1976): coronal holes ↔ 고속 태양풍 스트림 연결 / coronal holes ↔ fast wind connection
- Levine et al. (1977): 자기장 divergence가 작은 open-field 영역 ↔ 고속 스트림 / small magnetic divergence ↔ fast streams
- Wang & Sheeley (1990): $f_s$-$v$ 반비례 관계 발견, CC = 0.57 (3개월 평균) / inverse $f_s$-$v$ relation, CC = 0.57 (3-month averages)
- Wintoft & Lundstedt (1997): ANN + PFSS 결합, CC = 0.58, 일일 예측에서 CC ~ 0.4 / ANN + PFSS, CC = 0.58, daily CC ~ 0.4

Prior related work:
- Nolte et al. (1976): coronal hole ↔ fast wind link
- Levine et al. (1977): small magnetic divergence in open-field regions ↔ fast streams
- Wang & Sheeley (1990): discovered inverse $f_s$-$v$ relation, CC = 0.57 (3-month averages)
- Wintoft & Lundstedt (1997): ANN + PFSS combination, CC = 0.58, daily CC ~ 0.4

### Section 2: Overview / 개요

WS 모델에 대한 두 가지 범주의 수정을 소개합니다:

Two categories of modifications to the WS model are introduced:

**입력 데이터 개선 / Input data improvements:**
1. Synoptic map을 이전 rotation의 것이 아닌 **매일 갱신** / Synoptic maps updated **daily** rather than using the previous rotation's map
2. 문제 있는 magnetogram을 식별·제거하고, projection 효과, solar $b$ angle, 극 자기장 불확실성을 보정 / Identify and remove problematic magnetograms; correct for projection effects, solar $b$ angle, and polar field uncertainties

**모델링 방법 개선 / Modeling method improvements:**
1. $f_s$와 속도를 연결하는 연속 경험 함수를 source surface에서 적용 / Continuous empirical function relating $f_s$ to velocity applied at the source surface
2. Source surface에서 지구까지 방사형 스트림 전파 + 스트림 상호작용 고려 / Radial stream propagation from source surface to Earth with stream interaction scheme

### Section 3: Input to the Model / 모델 입력

#### 3.1 Full-Rotation (FR) Synoptic Maps / 전체 자전 싱옵틱 맵

WSO에서 매일 1-2개의 저해상도 magnetogram을 촬영합니다 (경도 110°, 중앙 자오선 중심). Carrington 좌표로 보간하면 각 magnetogram은 $23 \times 30$ 개의 셀(경도 5°, 사인 위도 1/15)로 구성됩니다. 태양은 하루에 ~13° 회전하므로 연속 magnetogram의 중심 자오선 경도 차이는 ~10°-15°입니다.

WSO takes 1-2 low-resolution magnetograms per day (110° longitude extent centered on central meridian). Interpolated to Carrington coordinates, each magnetogram has $23 \times 30$ cells (5° longitude, 1/15 sine latitude). The Sun rotates ~13°/day, so consecutive magnetogram central meridian longitudes differ by ~10°-15°.

FR synoptic map의 특징: 경도 방향 거리에 따른 가중 평균으로 합성. 중앙 자오선 근처 관측에 높은 가중치. 한 셀에 최대 ~16개 magnetogram이 겹침 (이론적). 완성에 ~8일 소요.

FR synoptic maps: assembled via weighted mean favoring observations near central meridian. Up to ~16 overlapping magnetograms per cell (theoretical). Takes ~8 days to complete.

#### 3.2 Daily Updated (DU) Synoptic Maps / 일일 갱신 싱옵틱 맵

가장 최근 magnetogram의 leading edge에서 시작하여 최근 360°의 관측을 포함하는 synoptic map입니다. FR map과의 차이점: 동쪽 가장자리(leading edge)는 최근 1개 magnetogram에만 의존하므로 데이터 품질에 매우 민감합니다.

Synoptic maps starting from the leading edge of the most recent magnetogram, covering 360° of the most recent observations. Key difference from FR maps: the eastern edge (leading edge) relies on as few as one magnetogram, making it very sensitive to data quality.

핵심 장점: 지구를 향한 방향의 자기장이 가장 최근 관측으로 구성 → 예측에 유리 / Key advantage: Earth-facing field composed of most recent observations → better for predictions

핵심 단점: projection 효과(동쪽 가장자리에서 LOS 자기장이 실제의 ~1/2)에 매우 취약 / Key disadvantage: very susceptible to projection effects (LOS field ~1/2 of true value at eastern edge)

#### 3.3 Modifications to Daily Updated Maps / 일일 갱신 맵 수정

**3.3.1 문제 있는 magnetogram 식별 / Problematic magnetogram identification:**

극 자기장($B$)의 세 가지 특성(평균 $\langle B \rangle$, 표준편차 $\sigma_B$, 최대 spread $|B_\text{max} - B_\text{min}|$)을 62개의 "good" 참조 magnetogram과 비교하여 품질 계수 $Q$를 계산합니다:

Three properties of the polar field ($B$): mean $\langle B \rangle$, standard deviation $\sigma_B$, and maximum spread $|B_\text{max} - B_\text{min}|$, are compared against 62 "good" reference magnetograms to compute the quality factor $Q$:

$$Q = \sqrt{\left(\frac{\langle B \rangle - \langle B_\text{ref} \rangle}{\langle \sigma \rangle_{B(\text{ref})}}\right)^2 + \left(\frac{\sigma_B}{\sigma_{B(\text{ref})}}\right)^2} \times \left(\frac{|B_\text{max} - B_\text{min}|}{\langle \sigma \rangle_{B(\text{ref})}}\right) \tag{1}$$

- $Q < 6.5$: magnetogram 사용 가능 / magnetogram accepted
- $Q > 6.5$: 문제 있음 — 양쪽 끝 5°씩 경도를 잘라내고 재계산 / problematic — trim 5° from each end and recalculate
- 25° 이상 잘라내거나 여전히 $Q > 6.5$이면 magnetogram 폐기 / if >25° trimmed or still $Q > 6.5$, discard entirely
- 전체 magnetogram의 약 12%가 폐기됨, 41%는 15° 미만 trimming / ~12% discarded, ~41% trimmed less than 15°

**3.3.2 Projection 효과 보정 / Projection effect corrections:**

WSO magnetogram은 시선 방향(LOS) 자기장 성분 $B_l$을 측정합니다:

WSO magnetograms measure the line-of-sight (LOS) magnetic field component $B_l$:

$$B_l = B_r \sin\theta \cos(\phi - \phi_o) + B_\theta \cos\theta \cos(\phi - \phi_o) - B_\phi \sin(\phi - \phi_o) \tag{2}$$

여기서 $\phi_o$는 관측 시 중앙 자오선의 Carrington 경도, $\theta$는 colatitude입니다. 광구 자기장이 거의 방사형($B_\theta \approx 0$, $B_\phi \approx 0$)이라는 관측적 증거에 기반하여 단순화합니다:

Where $\phi_o$ is the Carrington longitude of the central meridian at observation time, $\theta$ is the colatitude. Observational evidence that the photospheric field is nearly radial ($B_\theta \approx 0$, $B_\phi \approx 0$) simplifies this to:

$$B_l = B_r \sin\theta \cos(\phi - \phi_o) \tag{3}$$

보정 과정: (1) 먼저 경도 방향 projection을 $\cos(\phi - \phi_o)$로 나누어 보정, (2) 그 다음 위도 방향 projection을 보정 (solar $b$ angle에 의한 계절적 변동 포함).

Correction procedure: (1) first correct longitudinal projection by dividing by $\cos(\phi - \phi_o)$, (2) then correct latitudinal projection (including seasonal variation due to solar $b$ angle).

**3.3.3 Solar $b$ angle에 따른 극 자기장 보정 / Polar field corrections for solar $b$ angle:**

태양 자전축의 황도면에 대한 7.25° 기울기 때문에 solar $b$ angle이 연간 $\pm 7.25°$로 변합니다. $b = +7.25°$일 때 북극은 잘 보이지만 남극은 잘 안 보이고, 반년 후에는 반대입니다. 이로 인해 극 자기장 측정에 연간 변동이 생깁니다 (Figure 3).

The 7.25° tilt of the solar rotation axis relative to the ecliptic causes the solar $b$ angle to vary $\pm 7.25°$ annually. At $b = +7.25°$, the north pole is well observed but the south pole is not, and vice versa half a year later. This creates annual variations in polar field measurements (Figure 3).

보정 방법: 사인 위도 $\pm 14.5/15.0$ ($\approx \pm 75°$)과 $\pm 13.5/15.0$ ($\approx \pm 64°$)에서, $|b| > 5°$인 가장 신뢰할 만한 보정된 극 자기장 데이터에 2차 다항식을 fitting합니다. 이 다항식으로 각 새 magnetogram의 극 자기장을 정규화합니다.

Correction method: at sine latitudes $\pm 14.5/15.0$ ($\approx \pm 75°$) and $\pm 13.5/15.0$ ($\approx \pm 64°$), a second-order polynomial is fit to the most reliable corrected polar field data from magnetograms with $|b| > 5°$. This polynomial normalizes the polar fields of each new magnetogram.

#### 3.4 Gaps in Synoptic Maps / 싱옵틱 맵의 공백

FR map: 큰 공백은 이전 rotation의 데이터로 채움, 작은 공백($\leq 15°$)은 보간. DU map: 큰 공백은 채우지 않음 — 공백이 있으면 예측하지 않음. 작은 공백은 보간. 채우지 못한 공백의 비율이 클수록 모델 성능 저하 (Section 5에서 확인).

FR maps: large gaps filled with previous rotation data, small gaps ($\leq 15°$) interpolated. DU maps: large gaps NOT filled — no predictions made when gaps exist. Small gaps interpolated. Higher unfilled gap fractions correlate with worse model performance (confirmed in Section 5).

### Section 4: Modifications to the Wang-Sheeley Model / WS 모델 수정

#### 4.1 Expansion Factor–Velocity Relationship / 팽창 계수–속도 관계

PFSS를 표준적으로 적용합니다: source surface 반지름 $R_s = 2.5 R_\odot$, spherical harmonics를 $\ell = 30$까지 전개합니다. 각 source surface 위의 점 $P$에서 자기력선을 광구까지 추적하여 expansion factor를 계산합니다:

PFSS is applied in the standard manner: source surface radius $R_s = 2.5 R_\odot$, spherical harmonics expanded to $\ell = 30$. At each point $P$ on the source surface, field lines are traced back to the photosphere to compute the expansion factor:

$$f_s = \left(\frac{R_\odot}{R_s}\right)^2 \frac{B^P(R_\odot)}{B^P(R_s)} \tag{implicit}$$

**핵심 개선: 새로운 $v(f_s)$ 경험 관계식** / **Key improvement: new $v(f_s)$ empirical relation:**

$$v(f_s) = 267.5 + \left[\frac{410}{(f_s)^{2/5}}\right] \tag{4}$$

이 관계식은 저위도 in situ 데이터에서 도출되었으며, **source surface에서의 속도**를 지정합니다 (기존 WS 모델은 지구에서의 속도를 기반으로 fitting). 이것은 중요한 차이점입니다: Arge & Pizzo의 $v$-$f_s$ 관계는 source surface에서 경험적으로 반복 조정하여 지구에서 관측값과 합리적으로 일치하는 속도를 얻을 때까지 fitting한 것입니다.

This relation was derived from low-latitude in situ data and specifies the **velocity at the source surface** (the original WS model fitting was based on velocities at Earth). This is a crucial difference: Arge & Pizzo's $v$-$f_s$ relation is empirically iterated at the source surface until predicted velocities at Earth reasonably match observations.

WS $v$-$f_s$ 플롯에서 scatter가 큰 이유:
1. 등속 전파 가정 — 실제로 태양풍 스트림은 등속으로 전파되지 않고 상호작용함
2. CME 같은 transient 이벤트가 배경 태양풍에 섞여 있음

Reasons for large scatter in WS $v$-$f_s$ plots:
1. Constant speed propagation assumption — in reality solar wind streams interact
2. Transient events like CMEs are mixed into the background solar wind data

#### 4.2 Propagation of Solar Wind to Earth / 태양풍의 지구 전파

태양풍을 source surface에서 지구까지 방사형으로 전파하는 단순 scheme:

Simple scheme for radially propagating solar wind from source surface to Earth:

1. Ecliptic line을 따라 각 $5° \times 5°$ 셀(72개)의 중앙 자오선 통과 시간을 기록하고 $f_s$로부터 속도를 부여
2. 각 요소를 1/8 AU씩 등속으로 전파
3. 인접한 요소가 만나면(빠른 요소가 느린 요소를 따라잡으면) 가중 평균 속도로 병합:

1. Record central meridian passage time for each $5° \times 5°$ cell (72 cells) along the ecliptic line, assign velocity from $f_s$
2. Propagate each element by 1/8 AU at constant velocity
3. When adjacent elements meet (fast catches slow), merge with weighted average velocity:

$$v_i = \sqrt{\frac{2}{(1/v_i^2) + (1/v_{i+1}^2)}} \tag{5}$$

이 가중 함수는 단순 산술 평균보다 물리적으로 적절한 **조화 평균에 가까운** 형태입니다. 빠른 요소의 도착 시간은 느린 요소의 도착 시간에 맞춰 조정됩니다 (빠른 요소가 1/8 AU 지점에서 느린 요소를 따라잡으면 거기서 합쳐진 속도로 다시 전파).

This weighting function is close to a **harmonic mean**, more physically appropriate than an arithmetic mean. The arrival time of the faster element is adjusted to match the slower element's arrival time (if the fast element catches the slow one at 1/8 AU, they merge and propagate together at the combined speed).

시간 해상도: 72개 셀이 ecliptic을 따라 배열되어 ~1/3 일의 시간 해상도로 예측. 4일 전 예측이 관측과 가장 잘 일치 (source surface → 지구 평균 전파 시간이 ~4일이므로).

Time resolution: 72 cells along the ecliptic yield ~1/3 day time resolution. 4-day advance predictions best match observations (mean propagation time from source surface to Earth is ~4 days).

**IMF 극성 예측**: 속도 예측과 유사하지만, 가중 함수 대신 겹치는 IMF 요소들의 평균을 사용합니다 (극성만 예측, 크기는 예측하지 않음). IMF 극성은 자기장 값이 지구에 도달한 후에야 할당됩니다.

**IMF polarity prediction**: similar to velocity prediction, but uses an average of overlapping IMF elements instead of the weighting function (predicts polarity only, not magnitude). IMF polarity is assigned only after magnetic field values arrive at Earth.

### Section 5: Predictions Using Three Methods / 세 가지 방법을 이용한 예측

3년간(1994년 후반~1997년 후반) WSO magnetogram과 WIND 위성 데이터를 비교합니다. 세 가지 예측 방법:

3-year comparison (late 1994 to late 1997) between WSO magnetograms and WIND satellite data. Three prediction methods:

| 방법 / Method | 설명 / Description |
|---|---|
| **FR (Full Rotation)** | 이전 rotation의 FR synoptic map 사용 — 전통적 방법 / Previous rotation's FR synoptic map — traditional approach |
| **DU (Daily Updated)** | 매일 갱신하되 magnetogram 보정 없음 / Daily updates, no individual magnetogram corrections |
| **MDU (Modified Daily Updated)** | 매일 갱신 + $Q$ factor 품질 관리 + 극 자기장 보정 + solar $b$ angle 보정 / Daily updates + $Q$ factor QC + polar field corrections + solar $b$ angle corrections |

#### 5.1 Instructive Examples / 예시

**CR 1899 (1995년 8월, 태양 극소기 전):**

| 방법 / Method | CC | AFD |
|---|---|---|
| FR | 0.678 | 0.12 |
| DU | 0.796 | 0.11 |
| MDU | 0.813 | 0.11 |

이 시기에는 고속풍 스트림이 잘 발달해 있어 세 방법 모두 속도 변화를 추적하지만, DU/MDU가 최신 데이터를 사용하므로 CC가 더 높습니다. AFD 0.11은 400 km/s 태양풍에서 ~45 km/s 편차에 해당합니다.

Well-developed fast wind streams during this period; all methods track speed variations, but DU/MDU show higher CC due to more recent data. AFD of 0.11 corresponds to ~45 km/s deviation for 400 km/s wind.

**CR 1911 (1996년 7월, 태양 극소기):**

| 방법 / Method | CC | AFD |
|---|---|---|
| FR | 0.065 | 0.11 |
| DU | 0.118 | 0.12 |
| MDU | 0.24 | 0.096 |

태양 극소기의 flat dipole 자기장 구성에서는 CC가 매우 낮지만 AFD는 여전히 작습니다 (~45 km/s). 이는 속도가 주로 ~400 km/s 이하의 저속풍이기 때문입니다. DU 방법에서 7월 13-14일경 **false alarm**(실제로 발생하지 않은 고속풍 예측)이 나타남 — 이는 매우 품질이 나쁜 magnetogram(Figure 2)이 synoptic map에 포함되었기 때문. MDU에서는 이 magnetogram이 제거되어 false alarm이 사라짐.

During solar minimum's flat dipole magnetic field configuration, CC is very low but AFD remains small (~45 km/s) because velocity is mostly slow wind (~400 km/s or less). DU method shows a **false alarm** around July 13-14 (predicted fast wind that didn't occur) — caused by a very poor quality magnetogram (Figure 2) included in the synoptic map. In MDU, this magnetogram is excluded and the false alarm disappears.

**핵심 교훈**: CC는 항상 모델 성능의 최선의 척도가 아닙니다. 예보에서는 작은 AFD(빠른/느린 풍속을 구별할 수 있는 정도)가 큰 CC보다 더 중요할 수 있습니다.

**Key lesson**: CC is not always the best measure of model performance. For forecasting, small AFD (ability to distinguish fast/slow wind) may be more important than large CC.

#### 5.1.2 IMF Polarity Comparison / IMF 극성 비교

CR 1911에서의 IMF 극성 예측 (FCPP: Fraction of Correct Polarity Predictions):
- FR: 0.86, DU: 0.91, MDU: 0.86
- 태양 극소기에 current sheet이 flat하므로 극성 예측이 상대적으로 용이 / Solar minimum's flat current sheet makes polarity prediction relatively easier

For CR 1911, FCPP (Fraction of Correct Polarity Predictions):
- FR: 0.86, DU: 0.91, MDU: 0.86
- Flat current sheet during solar minimum facilitates polarity prediction

#### 5.2 Long-Term Evaluation / 장기 평가 (3년)

Figure 7은 37개의 슬라이딩 시간 bin(각 3 Carrington rotation, 1 CR씩 이동)에 대한 AFD, CC, FCPP를 보여줍니다.

Figure 7 shows AFD, CC, and FCPP for 37 sliding time bins (3 Carrington rotations each, shifted by 1 CR).

**속도 예측 — AFD (Figure 7a):**
- 세 방법 모두 AFD ~0.15 범위에서 변동 / All three methods fluctuate around AFD ~0.15
- 3년 전체 AFD: FR 0.159, DU 0.157, MDU 0.150
- 채우지 못한 공백 비율(dashed line)이 클 때 AFD도 커짐 / Large unfilled gap fraction correlates with large AFD
- 1996년 8-10월: solar $b$ angle이 크고 current sheet이 flat → 극 자기장 불확실성 증가 → MDU가 극 보정으로 가장 좋은 성능 / Aug-Oct 1996: large $b$ angle + flat current sheet → polar field uncertainty → MDU performs best with polar corrections
- 400 km/s 태양풍 기준 AFD 0.10-0.15 → 40-60 km/s 이내 예측 / For 400 km/s wind, AFD 0.10-0.15 → predictions within 40-60 km/s

**속도 예측 — CC (Figure 7b):**
- 3년 전체 CC: MDU 0.389, DU 0.363, FR 0.343
- 통계적으로 유의한 CC(우연 확률 < 1%)가 대부분이지만, 공백 비율이 클 때 유의하지 않은 CC도 있음 / Most CCs are statistically significant (chance probability < 1%), but insignificant CCs occur when gap fractions are large
- Solar 극소기(1996년 5월경) 전후로 모든 방법의 CC가 낮아짐 — flat dipole 구성 때문 / All methods show lower CC around solar minimum (May 1996) due to flat dipole configuration

**IMF 극성 — FCPP (Figure 7c):**
- 3년 전체 FCPP: DU 0.764, FR 0.746, MDU 0.751
- Solar $b$ angle이 크고 current sheet이 flat할 때 FCPP가 높아지는 경향 (subearth point이 current sheet 위나 아래에 명확히 위치) / FCPP tends to be higher when solar $b$ angle is large and current sheet is flat (subearth point clearly above or below current sheet)
- 지속성(persistence)을 고려하면(IMF 극성은 ~13.64일 지속) 유효 샘플 크기는 ~145-174개로 줄지만, 이 FCPP가 우연일 확률은 여전히 "many orders of magnitude less than 1%" / After accounting for persistence (~13.64 days), effective sample sizes reduce to ~145-174, but probability of FCPP occurring by chance is still "many orders of magnitude less than 1%"

**모델 실패 요인 / Model failure factors:**
1. Synoptic map의 채우지 못한 공백 / Unfilled gaps in synoptic maps
2. Transient 이벤트 (CME) — 모델이 예측 불가 / Transient events (CMEs) — model cannot predict
3. Ecliptic plane과 current sheet의 기울기 / Inclination of ecliptic to current sheet
4. 개별 magnetogram의 측정 품질 / Quality of individual magnetogram measurements
5. 극 자기장 측정의 불확실성 / Polar field measurement uncertainties
6. 전구 자기장 구성의 변화 속도 / Rate of change of global magnetic field configuration

### Section 6: Summary and Conclusions / 요약 및 결론

MDU 방법이 가장 일관되게 좋은 배경 태양풍 속도 예측을 제공하며, AFD는 ~0.10-0.15 범위(40-60 km/s 이내)로 빠른 풍속과 느린 풍속을 구별할 수 있습니다. CC도 세 방법 중 가장 높습니다(0.389). IMF 극성은 세 방법 모두 ~75%의 정확도로 예측합니다.

The MDU method provides the most consistently good background solar wind speed predictions, with AFD in the ~0.10-0.15 range (within 40-60 km/s), enabling distinction between fast and slow wind. CC is also highest among the three methods (0.389). IMF polarity is predicted with ~75% accuracy by all three methods.

향후 활용: SOHO/MDI, GONG 같은 고해상도 magnetogram을 사용하면 일일 갱신이 더 효과적일 것이며, 이 태양풍 예측은 kinematic/MHD 모델의 현실적 경계 조건을 제공하여 CME 도달 시간, stream 구조, geoeffectiveness 예측에 기여할 수 있습니다.

Future applications: higher-resolution magnetograms from SOHO/MDI and GONG would make daily updating more effective. These solar wind predictions can provide realistic boundary conditions for kinematic/MHD models, contributing to CME arrival time, stream structure, and geoeffectiveness predictions.

---

## 3. Key Takeaways / 핵심 시사점

1. **Source surface에서 속도 지정이 핵심 개선** — 기존 WS 모델은 지구에서의 관측 속도와 $f_s$를 직접 비교했지만, WSA 모델은 source surface에서 속도를 부여한 후 스트림 상호작용을 고려하여 전파합니다. 이것이 scatter를 줄이는 근본적 개선입니다.
Assigning velocity at the source surface is the key improvement — the original WS model directly compared observed velocities at Earth with $f_s$, but WSA assigns velocity at the source surface and then propagates with stream interaction effects. This fundamentally reduces scatter.

2. **입력 데이터 품질이 모델 물리만큼 중요** — 논문의 상당 부분(Section 3)이 magnetogram 품질 관리, projection 보정, solar $b$ angle 보정에 할애됩니다. MDU와 DU의 차이는 오직 이 데이터 품질 처리에 있으며, 이것만으로 false alarm을 제거하고 성능을 개선합니다.
Input data quality is as important as model physics — a significant portion of the paper (Section 3) is devoted to magnetogram quality control, projection corrections, and solar $b$ angle corrections. The only difference between MDU and DU is this data quality treatment, and it alone eliminates false alarms and improves performance.

3. **$Q$ factor: 정량적 데이터 품질 관리의 선구적 사례** — 극 자기장의 평균, 표준편차, 최대 spread를 참조 세트와 비교하는 이 품질 계수는 자동화된 magnetogram 선별의 초기 사례이며, ~12%의 불량 magnetogram을 걸러냅니다.
The $Q$ factor is a pioneering example of quantitative data quality control — comparing polar field mean, standard deviation, and maximum spread against a reference set. This automated magnetogram screening filters out ~12% of bad data.

4. **CC vs AFD: 예보에서 무엇이 중요한가** — CR 1911에서 CC는 0.065(FR)로 거의 무의미하지만 AFD는 0.11로 여전히 ~45 km/s 이내 예측입니다. 예보에서는 고속/저속 풍속을 구별할 수 있는 AFD가 CC보다 더 실용적인 성능 지표입니다.
CC vs AFD: what matters for forecasting — in CR 1911, CC is 0.065 (FR), nearly meaningless, but AFD is 0.11, still predicting within ~45 km/s. For forecasting, AFD (ability to distinguish fast/slow wind) is more practical than CC.

5. **스트림 상호작용의 단순 모델링** — 1/8 AU 단위로 전파하면서 조화 평균에 가까운 가중 함수로 인접 요소를 병합하는 방식은 놀라울 정도로 효과적입니다. 이 방법은 CIR(corotating interaction region)의 형성을 근사적으로 포착합니다.
Simple modeling of stream interactions — propagating in 1/8 AU steps with a near-harmonic-mean weighting function to merge adjacent elements is surprisingly effective. This approach approximately captures CIR (corotating interaction region) formation.

6. **Solar $b$ angle이 극 자기장과 current sheet 예측에 결정적 영향** — 태양 자전축의 7.25° 기울기가 만드는 연간 변동은 단순한 관측 artifact가 아니라 모델 성능에 직접적 영향을 줍니다. 특히 current sheet이 flat할 때 (태양 극소기) $b$ angle 효과가 극대화됩니다.
Solar $b$ angle critically affects polar field and current sheet predictions — the annual variation from the Sun's 7.25° axial tilt is not just an observational artifact but directly impacts model performance, especially when the current sheet is flat (solar minimum).

7. **CME는 이 모델의 한계** — WSA는 배경(ambient) 태양풍만 예측하며 CME 같은 transient 이벤트를 포함하지 않습니다. 3년 연구 기간의 마지막 시기(1997년)에 halo CME가 급증하면서 모든 방법의 성능이 저하됩니다.
CMEs are a fundamental limitation of this model — WSA predicts only the ambient solar wind and does not include transient events like CMEs. Performance of all methods degrades near the end of the 3-year study period (1997) as halo CME activity increases.

8. **운용 예보의 기반 확립** — 이 논문은 현재 NOAA/SWPC WSA-ENLIL 운용 시스템의 직접적 선구자입니다. Source surface에서의 속도/IMF 예측은 ENLIL의 내부 경계 조건이 됩니다.
Foundation for operational forecasting established — this paper is the direct precursor to the current NOAA/SWPC WSA-ENLIL operational system. Velocity/IMF predictions at the source surface serve as ENLIL's inner boundary conditions.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 PFSS 모델 / PFSS Model

코로나 자기장은 Laplace 방정식으로 계산됩니다:
The coronal magnetic field is computed via Laplace's equation:

$$\nabla^2 \Phi = 0 \quad \text{for} \quad R_\odot \leq r \leq R_s$$

- 경계조건: 광구에서 $B_r = -\partial\Phi/\partial r$ (magnetogram 입력), source surface ($R_s = 2.5 R_\odot$)에서 순수 방사형
- Spherical harmonics 전개를 $\ell = 30$까지 수행
- Boundary conditions: $B_r = -\partial\Phi/\partial r$ at photosphere (magnetogram input), purely radial at source surface ($R_s = 2.5 R_\odot$)
- Spherical harmonics expansion to $\ell = 30$

### 4.2 Flux Tube Expansion Factor / 플럭스 관 팽창 계수

$$f_s = \left(\frac{R_\odot}{R_s}\right)^2 \frac{B^P(R_\odot)}{B^P(R_s)}$$

- $B^P(R_\odot)$: 광구에서의 자기장 강도 (field line의 footpoint) / Magnetic field strength at the photosphere (field line footpoint)
- $B^P(R_s)$: source surface에서의 자기장 강도 / Magnetic field strength at the source surface
- $f_s = 1$: 순수 방사형 팽창 (최소 팽창) / Purely radial expansion (minimum expansion)
- $f_s \gg 1$: 초방사형 팽창 (coronal hole 경계, streamer belt 근처) / Super-radial expansion (near coronal hole boundaries, streamer belt)

### 4.3 Expansion Factor → Velocity (핵심 경험 관계식) / Core Empirical Relation

$$v(f_s) = 267.5 + \left[\frac{410}{(f_s)^{2/5}}\right] \quad \text{[km/s]} \tag{4}$$

| $f_s$ | $v$ [km/s] | 물리적 위치 / Physical location |
|---|---|---|
| 1 | 677.5 | Coronal hole 중심 (최소 팽창) / CH center (minimum expansion) |
| 5 | 487.5 | Coronal hole 내부 / Inside CH |
| 10 | 430.7 | CH 경계 근처 / Near CH boundary |
| 50 | 332.3 | Streamer belt 근처 / Near streamer belt |
| 100 | 302.5 | Closed field 경계 / Closed field boundary |

**기존 WS 모델과의 핵심 차이**: 이 속도는 **source surface에서** 지정됩니다 (WS는 지구에서 관측된 속도에 직접 fitting). Source surface에서 지정 후 스트림 상호작용을 통해 전파하면 지구에서의 속도 scatter가 줄어듭니다.

**Key difference from original WS**: this velocity is assigned **at the source surface** (WS directly fitted to observed velocities at Earth). Assigning at the source surface and propagating with stream interactions reduces scatter at Earth.

### 4.4 Magnetogram 품질 계수 / Magnetogram Quality Factor

$$Q = \sqrt{\left(\frac{\langle B \rangle - \langle B_\text{ref} \rangle}{\langle \sigma \rangle_{B(\text{ref})}}\right)^2 + \left(\frac{\sigma_B}{\sigma_{B(\text{ref})}}\right)^2} \times \left(\frac{|B_\text{max} - B_\text{min}|}{\langle \sigma \rangle_{B(\text{ref})}}\right) \tag{1}$$

- $\langle B_\text{ref} \rangle$: 62개 참조 magnetogram의 극 자기장 평균 / Mean polar field of 62 reference magnetograms
- $\sigma_{B(\text{ref})}$: 참조 세트의 표준편차 / Standard deviation of reference set
- $\langle \sigma \rangle_{B(\text{ref})}$: 참조 세트 표준편차의 평균 / Average of reference set standard deviations
- 각 극(N, S)에 대해 독립적으로 계산 / Computed independently for each pole (N, S)
- $Q < 6.5$ → 수용, $Q > 6.5$ → 양쪽 끝 5°씩 trim 후 재계산 / $Q < 6.5$ → accept, $Q > 6.5$ → trim 5° from each end and recalculate

### 4.5 LOS 자기장과 방사 자기장 / LOS vs Radial Magnetic Field

완전한 LOS 성분 / Full LOS component:

$$B_l = B_r \sin\theta \cos(\phi - \phi_o) + B_\theta \cos\theta \cos(\phi - \phi_o) - B_\phi \sin(\phi - \phi_o) \tag{2}$$

방사형 가정($B_\theta \approx 0$, $B_\phi \approx 0$) 하에서 단순화 / Simplified under radial assumption:

$$B_l = B_r \sin\theta \cos(\phi - \phi_o) \tag{3}$$

보정 절차 / Correction procedure:
1. 경도 projection 보정: $B_l / \cos(\phi - \phi_o)$ / Longitudinal projection: divide by $\cos(\phi - \phi_o)$
2. 위도 projection 보정: solar $b$ angle에 의한 효과 포함, $\sin\theta$로 보정 / Latitudinal projection: correct by $\sin\theta$, including solar $b$ angle effects

### 4.6 스트림 상호작용 가중 함수 / Stream Interaction Weighting Function

$$v_i = \sqrt{\frac{2}{(1/v_i^2) + (1/v_{i+1}^2)}} \tag{5}$$

이 함수는 두 속도의 **역제곱의 산술 평균의 역수의 제곱근**, 즉 이차 조화 평균(quadratic harmonic mean)에 해당합니다. 산술 평균보다 느린 속도에 더 큰 가중치를 부여하여, 물리적으로 빠른 스트림이 느린 스트림을 따라잡을 때 느린 쪽의 영향이 더 크게 반영됩니다.

This is the **square root of the inverse of the arithmetic mean of the inverse squares** of the two velocities — a quadratic harmonic mean. It weights slower speeds more heavily than an arithmetic mean, physically reflecting that when a fast stream catches a slow one, the slower stream's influence dominates.

### 4.7 통계 지표 / Statistical Metrics

**평균 분수 편차 (AFD)** / Average Fractional Deviation:

$$\text{AFD} = \left\langle \frac{v_p - v_o}{v_o} \right\rangle$$

- $v_p$: 예측 속도, $v_o$: 관측 속도 / predicted velocity, observed velocity
- AFD = 0.15 → 400 km/s에서 ±60 km/s / for 400 km/s, ±60 km/s

**상관 계수 (CC)** / Correlation Coefficient:
- 지속성(persistence) 보정: 태양풍 스트림은 최소 2일 지속 → 5.28개의 예측 데이터 포인트가 2일 bin에 해당 → 유효 샘플 크기 = 실제 비교 횟수 / 5.28
- Persistence correction: solar wind streams persist ≥2 days → 5.28 predicted data points per 2-day bin → effective sample size = actual comparisons / 5.28

**정확 극성 예측 비율 (FCPP)** / Fraction of Correct Polarity Predictions:
- 9시간 평균 WIND IMF 데이터에서 9개 시간값 중 2/3 이상이 같은 부호면 해당 극성 할당, 그렇지 않으면 "mixed" (0)으로 표기 → mixed 제외하고 FCPP 계산
- From 9-hour averaged WIND IMF data: if ≥2/3 of 9 hourly values share the same sign, assign that polarity; otherwise mark "mixed" (0) → FCPP calculated excluding mixed periods

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1958  Parker               태양풍 존재 이론적 예측
                            Theoretical prediction of solar wind (Paper #4)
      │
1962  Mariner 2            태양풍 최초 직접 확인
                            First direct solar wind confirmation
      │
1969  Altschuler &          PFSS 모델 개발 — 코로나 자기장 근사
      Newkirk               PFSS model — coronal field approximation
      │
1976  Nolte et al.          Coronal holes ↔ 고속 태양풍 연결
                            Coronal holes ↔ fast solar wind link
      │
1977  Levine et al.         Open-field divergence ↔ 태양풍 속도
                            Open-field divergence ↔ solar wind speed
      │
1982  Pizzo                 3D corotating stream 모델
                            3D corotating stream model
      │
1987  Fry & Akasofu         자기 위도 기반 태양풍 속도 경험 공식
                            Latitude-based empirical solar wind speed
      │
1990  Wang & Sheeley        f_s ↔ v 반비례 관계 최초 발견
                            First f_s ↔ v inverse relation (CC=0.57)
      │
1992  Wang & Sheeley        PFSS 기반 태양풍 모델 발전
                            PFSS-based solar wind model development
      │
1995  Zhao & Hoeksema       일일 평균 IMF 극성 예측
                            Daily averaged IMF polarity prediction
      │
1997  Wintoft &             ANN + PFSS 결합 모델, CC=0.58
      Lundstedt             ANN + PFSS combined model, CC=0.58
      │
>>>  2000  Arge & Pizzo     ★ WSA 모델: v(f_s) at source surface,
                            stream interaction, QC system  <<<< 이 논문
      │
2003  Arge et al.           WSA 모델 추가 개선 및 검증
                            Further WSA improvements and validation
      │
2003  Odstrcil              ENLIL 3D MHD heliospheric 모델
                            ENLIL 3D MHD heliospheric model
      │
2004  Arge et al.           WSA + θ_b 매개변수 추가
                            WSA + θ_b parameter added
      │
현재  NOAA/SWPC             WSA-ENLIL 운용 우주기상 예보 시스템
present                     WSA-ENLIL operational forecasting system
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **#4 Parker (1958)** — 태양풍 존재 예측 | Parker가 예측한 태양풍의 **속도 구조**를 경험적으로 예측하는 모델 / Empirically predicts the **speed structure** of the solar wind Parker predicted | WSA는 Parker의 태양풍 위에 구축된 예측 도구 / WSA is a prediction tool built upon Parker's solar wind |
| **#11 Burton et al. (1975)** — Dst 경험 공식 | Burton 방정식의 입력인 태양풍 매개변수를 WSA가 예측 / WSA predicts the solar wind parameters that are input to Burton's equation | WSA → Dst 예보 체인의 첫 고리 / WSA is the first link in the WSA → Dst forecasting chain |
| **Wang & Sheeley (1990)** — $f_s$-$v$ 관계 발견 | WSA의 직접적 전신. $f_s$-$v$ 반비례 관계를 발견했지만 source surface가 아닌 지구에서 fitting | Direct predecessor. Discovered $f_s$-$v$ relation but fitted at Earth, not source surface |
| **Altschuler & Newkirk (1969)** — PFSS 모델 | WSA의 핵심 구성 요소인 PFSS 코로나 자기장 모델의 원조 / Original PFSS coronal field model, a core component of WSA | PFSS가 없으면 $f_s$ 계산이 불가능 / Without PFSS, $f_s$ cannot be computed |
| **Pizzo (1982)** — 3D corotating stream 모델 | 이 논문의 공동저자(Pizzo)가 개발한 더 정교한 스트림 상호작용 모델. WSA는 그 단순화 버전 사용 | More sophisticated stream interaction model by the paper's co-author (Pizzo). WSA uses a simplified version |
| **#17 Allen (1989)** — solar $b$ angle과 geomagnetic activity | Allen이 분석한 지구-태양 기하학적 관계와 solar $b$ angle 효과가 WSA의 극 자기장 보정과 직결 | The Earth-Sun geometric relationships and solar $b$ angle effects Allen analyzed are directly relevant to WSA's polar field corrections |
| **Odstrcil (2003)** — ENLIL 모델 | WSA의 source surface 출력을 내부 경계 조건으로 사용하는 3D MHD 모델 / 3D MHD model using WSA's source surface output as inner boundary condition | WSA-ENLIL 운용 시스템의 결합 / Coupling of WSA-ENLIL operational system |

---

## 7. References / 참고문헌

- Altschuler, M. A. and G. Newkirk Jr., "Magnetic fields and the structure of the solar corona," *Sol. Phys.*, 9, 131-149, 1969.
- Arge, C. N. and V. J. Pizzo, "Improvement in the prediction of solar wind conditions using near-real time solar magnetic field updates," *J. Geophys. Res.*, 105(A5), 10465-10479, 2000. [DOI: 10.1029/1999JA000262]
- Burton, R. K., R. L. McPherron, and C. T. Russell, "An empirical relationship between interplanetary conditions and Dst," *J. Geophys. Res.*, 80, 4204-4214, 1975. [Paper #11]
- Fry, C. D. and S.-I. Akasofu, "Latitudinal dependence of solar wind speed," *Planet. Space Sci.*, 35(7), 913-920, 1987.
- Gosling, J. T., V. Pizzo, M. Neugebauer, and C. W. Snyder, "Twenty-seven-day recurrences in the solar wind: Mariner 2," *J. Geophys. Res.*, 77, 2744-2751, 1972.
- Hoeksema, J. T., J. M. Wilcox, and P. H. Scherrer, "Structure of the heliospheric current sheet in the early portion of sunspot cycle 21," *J. Geophys. Res.*, 87, 10331-10338, 1982.
- Hoeksema, J. T., "Structure and evolution of the large scale solar and heliospheric magnetic fields," Ph.D. thesis, Stanford Univ., 1984.
- Levine, R. H., M. D. Altschuler, and J. W. Harvey, "Solar sources of the interplanetary magnetic field and solar wind," *J. Geophys. Res.*, 82, 1061-1065, 1977.
- Nolte, J. T. et al., "Coronal holes as sources of solar wind," *Sol. Phys.*, 46, 303-322, 1976.
- Parker, E. N., "Dynamics of the interplanetary gas and magnetic fields," *Astrophys. J.*, 128, 664-676, 1958. [Paper #4]
- Pizzo, V. J., "A three-dimensional model of corotating streams in the solar wind 3: Magnetohydrodynamic streams," *J. Geophys. Res.*, 87, 4374-4394, 1982.
- Schatten, K. H., J. M. Wilcox, and N. F. Ness, "A model of interplanetary and coronal magnetic fields," *Sol. Phys.*, 9, 442-455, 1969.
- Taylor, J. R., *An Introduction to Error Analysis*, 270 pp., Univ. Sci. Books, Mill Valley, Calif., 1982.
- Wang, Y.-M. and N. R. Sheeley, "Solar wind speed and coronal flux-tube expansion," *Astrophys. J.*, 355, 726-732, 1990.
- Wang, Y.-M. and N. R. Sheeley, "Why fast solar wind originates from slowly expanding coronal flux tubes," *Astrophys. J.*, 372, L45-L48, 1991.
- Wang, Y.-M. and N. R. Sheeley, "On potential field models of the solar corona," *Astrophys. J.*, 392, 310-319, 1992.
- Wang, Y.-M. and N. R. Sheeley, "Solar implications of ULYSSES interplanetary field measurements," *Astrophys. J.*, 447, L143-L146, 1995a.
- Wang, Y.-M. and N. R. Sheeley, "Empirical relationship between the magnetic field and the mass and energy flux in the source regions of the solar wind," *Astrophys. J.*, 449, L157-L160, 1995b.
- Wang, Y.-M., S. H. Hawley, and N. R. Sheeley, "The magnetic nature of coronal holes," *Science*, 271, 464-469, 1996.
- Wintoft, P. and H. Lundstedt, "Prediction of daily average solar wind velocity from solar magnetic field observations using hybrid intelligent systems," *Phys. Chem. Earth*, 22(7-8), 617, 1997.
- Zhao, X. and J. T. Hoeksema, "Unique determination of model coronal magnetic fields using photospheric observations," *Sol. Phys.*, 143, 41-48, 1993.
- Zhao, X. and J. T. Hoeksema, "Prediction of the interplanetary magnetic field strength," *J. Geophys. Res.*, 100, 19-33, 1995.
