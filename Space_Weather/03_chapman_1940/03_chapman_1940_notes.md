---
title: "Geomagnetism"
authors: Sydney Chapman, Julius Bartels
year: 1940
journal: "Oxford University Press (Clarendon Press), 2 vols., pp. xxviii+1049"
topic: Space Weather / Geomagnetism
tags: [geomagnetism, spherical harmonics, Gauss coefficients, Kp index, Dst index, magnetic storm, diurnal variation, secular variation, geomagnetic indices, dipole field]
status: completed
date_started: 2026-04-06
date_completed: 2026-04-07
---

# Geomagnetism (1940)
# 지자기학 (1940)

---

## 핵심 기여 / Core Contribution

Chapman과 Bartels의 "Geomagnetism"은 **지자기학 분야의 백과사전적 종합서**로, 1940년 이전까지 축적된 지구 자기장에 관한 모든 관측 데이터, 수학적 분석 방법, 물리적 해석을 하나의 체계로 통합했습니다. 이 모노그래프의 기여는 세 축으로 나뉩니다: (1) **Gauss의 구면 조화 분석(spherical harmonic analysis)**을 지구 자기장에 체계적으로 적용하여, 관측 데이터로부터 **내부 기원(internal origin, 외핵 대류)**과 **외부 기원(external origin, 전리층·자기권 전류)**의 자기장 성분을 수학적으로 분리하는 표준 방법론을 확립했습니다. (2) Bartels가 개발한 **Kp 지수**를 포함한 지자기 활동 지수 체계를 도입하여, 전 세계 관측소 데이터를 **하나의 정량적 척도**로 환산하는 방법을 제시했습니다 — 이는 지자기 폭풍의 강도 비교, 태양 활동과의 상관 분석, 나아가 우주기상 예보의 기초가 됩니다. (3) 자기장의 **시간 변동(temporal variations)**을 시간 스케일별로 체계적으로 분류했습니다: 영년 변화(secular variation, 수백 년), 태양흑점 주기(~11년), 27일 재현(태양 자전), 일변화(diurnal variation, Sq), 자기 폭풍(수시간~수일). 이러한 분류와 분석 틀은 우주 시대(Space Age) 이전에 **지상 관측만으로** 구축된 것으로, 이후 위성 관측 시대에도 기본 참조 체계로 기능했습니다.

Chapman and Bartels' "Geomagnetism" is an **encyclopedic synthesis** of all observational data, mathematical analysis methods, and physical interpretations of Earth's magnetic field accumulated before 1940. Its contributions fall along three axes: (1) Systematic application of **Gauss's spherical harmonic analysis** to establish the standard methodology for mathematically separating **internally-originated** (outer core convection) and **externally-originated** (ionospheric/magnetospheric currents) field components from observational data. (2) Introduction of a geomagnetic activity index system, including Bartels' **Kp index**, providing a method to convert worldwide observatory data into **a single quantitative measure** — foundational for storm intensity comparison, solar-geomagnetic correlation analysis, and ultimately space weather forecasting. (3) Systematic classification of magnetic field **temporal variations** by timescale: secular variation (centuries), solar cycle (~11 years), 27-day recurrence (solar rotation), diurnal variation (Sq), and magnetic storms (hours to days). This classification and analytical framework was built entirely from **ground-based observations alone** before the Space Age, and continued to serve as the basic reference system even in the satellite era.

---

## 읽기 노트 / Reading Notes

### Part I. 지구 자기장의 기본 요소 / Fundamental Elements of the Geomagnetic Field

#### 1.1 자기장 요소의 정의 / Definitions of Magnetic Field Elements

지구 자기장은 벡터이므로, 한 지점에서 세 개의 독립적인 성분으로 완전히 기술됩니다. Chapman과 Bartels는 다음 **7가지 요소(elements)**를 정의하는데, 이 중 세 개만 독립적입니다:

The geomagnetic field is a vector, fully described by three independent components at any point. Chapman and Bartels define seven **elements**, of which only three are independent:

| 기호 / Symbol | 이름 / Name | 정의 / Definition |
|---|---|---|
| $F$ | Total intensity / 전자기장 강도 | 자기장 벡터의 크기 / Magnitude of the field vector |
| $H$ | Horizontal intensity / 수평 성분 | 수평면 위 자기장 성분 / Horizontal component of the field |
| $Z$ | Vertical intensity / 수직 성분 | 연직 방향 성분 (아래 양) / Vertical component (positive downward) |
| $X$ | North component / 북향 성분 | 지리적 북쪽 방향 성분 / Geographic northward component |
| $Y$ | East component / 동향 성분 | 지리적 동쪽 방향 성분 / Geographic eastward component |
| $D$ | Declination / 편각 | $H$와 지리적 북쪽의 각도 / Angle between $H$ and geographic north |
| $I$ | Inclination (Dip) / 복각 | $\vec{F}$와 수평면의 각도 / Angle between $\vec{F}$ and horizontal plane |

관계식 / Relations:

$$H = F\cos I, \quad Z = F\sin I, \quad \tan I = Z/H$$
$$X = H\cos D, \quad Y = H\sin D, \quad \tan D = Y/X$$
$$F^2 = H^2 + Z^2 = X^2 + Y^2 + Z^2$$

Chapman과 Bartels는 **세 요소의 독립적 측정 조합**으로 다양한 실험을 구성할 수 있음을 보여줍니다. 예를 들어 $(D, H, Z)$, $(D, I, F)$, $(X, Y, Z)$ 등.

#### 1.2 등자기선도 / Isomagnetic Charts

자기장 요소가 같은 값을 갖는 지점들을 지도 위에 연결한 것이 **등자기선도(isomagnetic chart)**입니다:

- **Isogonic lines (등편각선)**: $D$ = 일정. 특히 $D = 0$인 선을 **agonic line(무편각선)**이라 합니다 — 이 선 위에서는 나침반이 정확히 지리적 북쪽을 가리킵니다.
- **Isoclinic lines (등복각선)**: $I$ = 일정. $I = 0$인 선이 **magnetic equator(자기 적도)**입니다.
- **Isodynamic lines (등역선)**: $F$, $H$, 또는 $Z$ = 일정.

이 지도들은 지구 자기장의 공간적 구조를 직관적으로 보여줍니다. **자기 극(magnetic poles)**은 $I = \pm 90°$인 지점이며, 지리적 극과 일치하지 않습니다 (약 11° 기울어져 있음).

Isomagnetic charts connect points of equal magnetic element values on a map. The **agonic line** ($D = 0$) shows where compasses point to true north. The **magnetic equator** ($I = 0$) differs from the geographic equator. Magnetic poles ($I = \pm 90°$) are offset ~11° from geographic poles.

#### 1.3 자기 쌍극자 근사 / The Dipole Approximation

지구 자기장의 가장 기본적인 모델은 **중심 쌍극자(centered dipole)**입니다. 지구 중심에 자기 쌍극자가 있다고 가정하면:

The most basic model is a **centered magnetic dipole**. The scalar potential for a magnetic dipole at the center:

$$V = -\frac{M \cos\theta}{r^2}$$

여기서 $M$은 쌍극자 모멘트, $\theta$는 쌍극자 축에서의 여위도(colatitude), $r$은 지구 중심으로부터의 거리.

자기장 성분:

$$B_r = -\frac{\partial V}{\partial r} = -\frac{2M\cos\theta}{r^3}$$

$$B_\theta = -\frac{1}{r}\frac{\partial V}{\partial \theta} = -\frac{M\sin\theta}{r^3}$$

$$B_\phi = 0 \quad \text{(축대칭이므로 / due to axial symmetry)}$$

지표면($r = a$)에서의 값:

$$H = \frac{M\sin\theta}{a^3}, \quad Z = \frac{2M\cos\theta}{a^3}$$

$$\tan I = \frac{Z}{H} = 2\cot\theta = 2\tan\lambda_m$$

이 마지막 관계식이 중요합니다 — **복각 $I$로부터 자기 위도 $\lambda_m$을 추정할 수 있습니다**. 이 관계는 현재도 고지자기학(paleomagnetism)에서 널리 사용됩니다.

The last relation — $\tan I = 2\tan\lambda_m$ — allows estimation of **geomagnetic latitude from inclination** and is still widely used in paleomagnetism.

그러나 쌍극자 근사에는 명백한 한계가 있습니다:
- 실제 지구에는 쌍극자가 아닌 **non-dipole** 성분이 약 10% 존재
- 자기 극은 정확히 지리적 극의 대척점(antipodal)이 아님
- 자기 적도는 지리적 적도와 어긋남

However, the dipole approximation has clear limitations:
- ~10% of Earth's field is **non-dipole** components
- Magnetic poles are not exactly antipodal to each other
- The magnetic equator deviates from the geographic equator

이러한 편차를 정밀하게 기술하기 위해 구면 조화 분석이 필요합니다.

Spherical harmonic analysis is needed to precisely describe these deviations.

---

### Part II. 구면 조화 분석 / Spherical Harmonic Analysis

이 부분이 모노그래프의 **수학적 핵심**입니다. Gauss (1839)가 도입한 방법을 Chapman과 Bartels가 체계적으로 정리하고 확장합니다.

This is the **mathematical core** of the monograph. Chapman and Bartels systematize and extend the method introduced by Gauss (1839).

#### 2.1 자기 퍼텐셜의 일반 전개 / General Expansion of the Magnetic Potential

지구 표면 근처에서 자기장은 **소용돌이가 없으므로(curl-free)** 스칼라 퍼텐셜 $V$로 기술됩니다:

Near Earth's surface, the field is **curl-free** (no local currents), so it can be described by a scalar potential $V$:

$$\vec{B} = -\nabla V$$

$V$는 라플라스 방정식 $\nabla^2 V = 0$을 만족하며, 구면 좌표에서 일반해는:

$$V(r, \theta, \phi) = a \sum_{n=1}^{\infty} \sum_{m=0}^{n} \left[\left(\frac{a}{r}\right)^{n+1}\left(g_n^m \cos m\phi + h_n^m \sin m\phi\right) + \left(\frac{r}{a}\right)^n\left(q_n^m \cos m\phi + s_n^m \sin m\phi\right)\right] P_n^m(\cos\theta)$$

여기서:
- $(a/r)^{n+1}$ 항 → **내부 기원(internal origin)** — $r$ 증가 시 감소 → 지구 내부의 전류/자화가 원인
- $(r/a)^n$ 항 → **외부 기원(external origin)** — $r$ 증가 시 증가 → 전리층/자기권의 전류가 원인
- $g_n^m$, $h_n^m$ = 내부 기원의 **Gauss 계수**
- $q_n^m$, $s_n^m$ = 외부 기원의 Gauss 계수
- $P_n^m(\cos\theta)$ = 배속 르장드르 함수 (associated Legendre functions)
- $n$ = 차수(degree), $m$ = 위수(order)

#### 2.2 내부/외부 분리의 원리 / Principle of Internal/External Separation

**Gauss의 핵심 통찰**: 지표면($r = a$)에서의 관측만으로 내부/외부 기원을 분리할 수 있습니다.

방법: 지표면에서 $V$를 구면 조화 함수로 전개하면 $g_n^m + q_n^m$과 $h_n^m + s_n^m$의 합만 결정됩니다. 그러나 **수직 성분 $Z$** 또는 **$\partial V / \partial r$**을 측정하면:

$$-\frac{\partial V}{\partial r}\bigg|_{r=a} = Z = \sum_{n,m} \left[(n+1)(g_n^m \cos m\phi + h_n^m \sin m\phi) - n(q_n^m \cos m\phi + s_n^m \sin m\phi)\right] P_n^m$$

내부 항 앞에는 계수 $(n+1)$이, 외부 항 앞에는 계수 $-n$이 붙으므로, $H$와 $Z$의 조합으로 두 기원을 **분리**할 수 있습니다.

**Gauss's key insight**: observations at the surface ($r = a$) alone can separate internal and external sources. The radial derivative of $V$ gives different coefficients — $(n+1)$ for internal and $-n$ for external terms — enabling separation.

Gauss의 1839년 분석 결과: **주 자기장의 99% 이상이 내부 기원**이며, 외부 기원은 변동 성분(자기 폭풍, 일변화 등)에만 유의미합니다.

Gauss's 1839 result: **>99% of the main field is internal in origin**. External sources contribute significantly only to variation fields (storms, diurnal variation, etc.).

#### 2.3 Gauss 계수의 물리적 의미 / Physical Meaning of Gauss Coefficients

각 $(n, m)$ 항의 물리적 의미:

| 차수 $n$ | 위수 $m$ | 성분 / Component | 물리적 의미 / Physical Meaning |
|---|---|---|---|
| 1 | 0 | $g_1^0$ | 축 쌍극자 (axial dipole) — 자기장 에너지의 ~90% |
| 1 | 1 | $g_1^1$, $h_1^1$ | 적도 쌍극자 (equatorial dipole) — 축의 기울기를 결정 |
| 2 | 0 | $g_2^0$ | 축 사중극자 (axial quadrupole) |
| 2 | 1 | $g_2^1$, $h_2^1$ | 사중극자의 기울기 성분 |
| 2 | 2 | $g_2^2$, $h_2^2$ | 적도면 사중극자 |
| $n \geq 3$ | — | — | 고차 다중극자 (higher multipoles) |

**쌍극자 축의 방향** 결정:

$$\tan\alpha = \frac{h_1^1}{g_1^1}, \quad \cos\theta_0 = \frac{g_1^0}{\sqrt{(g_1^0)^2 + (g_1^1)^2 + (h_1^1)^2}}$$

여기서 $\alpha$는 쌍극자 축의 경도, $\theta_0$는 여위도입니다. 1940년 당시 값: 쌍극자 축은 지리적 축에서 약 11.5° 기울어져 있었습니다.

The **dipole axis direction** is determined from the $n=1$ coefficients. In 1940, the tilt was ~11.5° from the geographic axis.

#### 2.4 Non-dipole 자기장 / The Non-Dipole Field

$n \geq 2$ 항들의 합이 **non-dipole field(비쌍극자장)**이며, 이를 등역선도(isodynamic chart)에서 쌍극자 성분을 제거하면 시각화할 수 있습니다. Chapman과 Bartels는 이 잔여 자기장이:

- 수 개의 **큰 규모 이상 영역(anomalies)**으로 구성됨 (남대서양, 시베리아, 캐나다 등)
  Composed of several **large-scale anomalies** (South Atlantic, Siberia, Canada, etc.)
- 쌍극자장보다 훨씬 빠르게 **영년 변화(secular variation)**함
  Varies secularly much faster than the dipole field
- 외핵 표면 근처의 **국소적 대류 패턴**을 반영할 가능성이 높음
  Likely reflects **localized convection patterns** near the outer core surface

---

### Part III. 자기장의 시간 변동 / Temporal Variations of the Magnetic Field

Chapman과 Bartels는 자기장 변동을 **시간 스케일**에 따라 체계적으로 분류합니다. 이 분류 체계는 현대 지자기학에서도 기본 참조 틀로 사용됩니다.

Chapman and Bartels classify field variations systematically by **timescale**. This classification remains the basic reference framework in modern geomagnetism.

#### 3.1 영년 변화 / Secular Variation

**시간 스케일**: 수십~수백 년 / Timescale: decades to centuries

Gauss 계수의 시간 변화율 $\dot{g}_n^m$, $\dot{h}_n^m$ (secular variation coefficients):

$$\frac{dg_1^0}{dt} \approx -15 \text{ nT/yr (1940년 당시)}$$

주요 현상 / Key phenomena:
- **자기 쌍극자 모멘트의 감소**: 1830년 이후 약 5% 감소 — 현대까지 이어지는 추세
  **Dipole moment decay**: ~5% decrease since 1830 — a trend continuing to the present
- **자기 극의 이동(magnetic pole wandering)**: 연간 수 km 이동
  **Magnetic pole wandering**: several km per year
- **서향 이동(westward drift)**: non-dipole 자기장 패턴이 연간 약 0.2°씩 서쪽으로 이동 — 외핵 표면의 차동 회전(differential rotation)을 시사
  **Westward drift**: non-dipole patterns drift ~0.2°/yr westward — suggesting differential rotation of the outer core surface
- **자기 제크(magnetic jerks)**: 영년 변화의 가속도에서 갑작스런 변화 (이 현상은 Chapman과 Bartels 당시에는 아직 체계적으로 인식되지 않았음)
  **Magnetic jerks**: abrupt changes in secular variation acceleration (not yet systematically recognized in Chapman & Bartels' time)

#### 3.2 조용한 날의 태양 일변화 / Solar Quiet-Day Variation ($S_q$)

**시간 스케일**: 24시간 주기 / Timescale: 24-hour period

Chapman과 Bartels는 자기적으로 가장 조용한 5일(International Quiet Days)의 관측 데이터를 평균하여 **$S_q$ 변동**을 추출합니다.

$S_q$ variation is extracted by averaging the 5 magnetically quietest days per month.

$S_q$의 특성 / Characteristics of $S_q$:
- **진폭**: 수십 nT (위도에 따라 변화)
  **Amplitude**: tens of nT (latitude-dependent)
- **원인**: 태양 UV에 의해 이온화된 전리층(E 영역, ~110 km)의 **조석 바람(tidal wind)**에 의한 다이나모 전류
  **Cause**: dynamo currents driven by **tidal winds** in the ionospheric E-region (~110 km), ionized by solar UV
- **패턴**: 각 반구에서 태양을 향한 쪽에 하나의 전류 소용돌이(current vortex)가 형성
  **Pattern**: one current vortex forms on the sunward side in each hemisphere

Chapman과 Bartels의 $S_q$ 전류 체계 모델:

$$S_q(t, \lambda, \phi) = \sum_{m=1}^{4} \sum_{n=m}^{N} \left[\alpha_n^m \cos m\tau + \beta_n^m \sin m\tau\right] P_n^m(\cos\theta)$$

여기서 $\tau = 2\pi t / 24$ (local time을 각도로 변환).

이 전류 체계는 지표면에서 약 **100,000 A**의 전류를 수반합니다 — 이것은 순수하게 **외부 기원**이며, Gauss 분석에서 외부 항($q_n^m$, $s_n^m$)으로 나타납니다.

The $S_q$ current system involves ~100,000 A of current in a vortex pattern in the ionospheric E-region, driven by tidal winds in the solar-UV-ionized conducting layer.

#### 3.3 자기 폭풍 / Magnetic Storms

Chapman과 Bartels는 자기 폭풍의 **형태학적 분류(morphological classification)**를 수립합니다. Paper #2(Chapman & Ferraro, 1931)에서 물리적 메커니즘을 다루었다면, 여기서는 **관측 데이터의 통계적 패턴**에 집중합니다.

While Paper #2 addressed physical mechanisms, here the focus is on **statistical patterns of observational data**.

**자기 폭풍의 3단계 구조 / Three-Phase Structure**:

```
H (수평 성분)
│
│    ┌──┐
│    │  │ Initial phase
│    │  │ (SC, 수십 분)
│────┘  │
│       │
│       └──────┐
│              │ Main phase
│              │ (수 시간)
│              │
│              └──────────────────────── Recovery phase
│                                        (수 시간~수일)
│
└──────────────────────────────────────── t
```

1. **Initial phase (초기 위상)**: Sudden Commencement(SC)로 시작 — $H$가 **갑자기 증가** (수십 nT, 수 분). Paper #2에서 설명한 자기장 압축 효과.
   Begins with SC — $H$ **suddenly increases** (tens of nT, minutes). Field compression explained in Paper #2.
2. **Main phase (주 위상)**: $H$가 크게 **감소** (수십~수백 nT, 수 시간). 환전류(ring current)의 형성에 의함 — 이 메커니즘은 이 모노그래프 시점에서는 아직 완전히 이해되지 않았음.
   $H$ **decreases** strongly (tens to hundreds of nT, hours). Due to ring current formation — not yet fully understood at the time of this monograph.
3. **Recovery phase (회복 위상)**: $H$가 서서히 원래 수준으로 **회복** (수 시간~수일). 환전류 입자의 점진적 소실에 의함.
   $H$ gradually **recovers** to baseline (hours to days). Due to gradual loss of ring current particles.

Chapman과 Bartels는 자기 폭풍을 **교란 일변화(disturbance daily variation, $S_D$)**와 **비주기적 교란(storm-time variation, $D_{st}$)**으로 분해합니다:

$$\Delta H(t, \lambda) = D_{st}(t) \cos\lambda + S_D(\tau, \lambda)$$

여기서:
- $D_{st}(t)$: 시간의 함수만으로 기술되는 **축대칭(axially symmetric)** 교란 — **환전류** 효과
- $S_D(\tau, \lambda)$: local time $\tau$에 의존하는 교란 — **부분 환전류(partial ring current)**, 전리층 전류 등
- $\cos\lambda$ 인자: 환전류의 자기장이 적도에서 가장 강하고 위도에 따라 감소

이 분해가 바로 현대 **Dst 지수**의 수학적 기초입니다.

This decomposition is the mathematical foundation of the modern **Dst index** — separating the axisymmetric ring-current effect from local-time-dependent disturbances.

#### 3.4 27일 재현 경향 / 27-Day Recurrence Tendency

Bartels의 독보적 기여 중 하나: 지자기 교란이 약 **27일 주기**로 반복되는 경향이 있다는 발견.

One of Bartels' distinctive contributions: geomagnetic disturbances tend to recur with a ~27-day period.

- **27일**: 태양의 자전 주기 (회합 주기, synodic period)
- 의미: 태양 표면의 특정 활성 영역이 태양 자전에 의해 반복적으로 지구를 향하면서 교란을 유발
- Bartels는 이를 시각화하기 위해 **Bartels rotation chart**를 발명 — 자기 데이터를 27일 간격으로 줄 바꿈하여 수직으로 정렬하면, 재현 패턴이 **수직 줄무늬**로 나타남
- 27일 재현이 **태양흑점이 아닌 곳**에서도 발생하는 경우가 있음 → Bartels는 이를 보이지 않는 **"M-regions"**이라 명명 — 이것은 훗날 **코로나 홀(coronal hole)**로 밝혀짐 (1970년대)

The 27-day recurrence maps to the solar synodic rotation period. Bartels invented **rotation charts** for visualization and identified invisible "M-regions" (later revealed as **coronal holes** in the 1970s) as sources of recurring activity unassociated with sunspots.

---

### Part IV. 지자기 지수 체계 / Geomagnetic Index Systems

이 부분이 Bartels의 **가장 실용적인 기여**입니다. 전 세계 수십 개 관측소의 서로 다른 자기 기록을 하나의 **표준화된 활동도 지표**로 변환하는 방법론을 수립합니다.

This is Bartels' **most practical contribution** — methodology for converting diverse magnetic records from dozens of worldwide observatories into a **standardized activity measure**.

#### 4.1 K 지수 / The K Index

각 관측소에서 3시간 간격으로 측정하는 **국소 활동 지수**:

A **local activity index** measured at each observatory every 3 hours:

**측정 방법**:
1. 3시간 구간(00-03, 03-06, ..., 21-24 UT)에서 $H$와 $D$ 성분의 변동 범위(range)를 측정
2. 조용한 날의 $S_q$ 변동을 먼저 제거(차감)
3. 교란 범위를 **준대수 척도(quasi-logarithmic scale)** 0–9로 변환

$K$ 척도의 설계 철학 — 왜 준대수인가? / Design philosophy — why quasi-logarithmic?

1. **자기 교란의 분포가 로그 정규(log-normal)**에 가깝기 때문 — 선형 척도에서는 대부분의 날이 K=0–1에 몰림
   Geomagnetic disturbance amplitudes follow approximately **log-normal distributions** — on a linear scale, most days would cluster at K=0–1
2. 약한 교란과 강한 교란 모두를 **동등한 분해능**으로 포착하기 위함
   To capture both weak and strong disturbances with **equal resolution**
3. 관측소마다 다른 배경 변동 수준을 **표준화**하기 용이함
   Facilitates **standardization** across stations with different background variation levels

각 관측소마다 $K$ 값과 실제 진폭의 대응표가 다릅니다(관측소의 지자기 위도에 따라). 고위도 관측소는 같은 $K$ 값에 대해 더 큰 진폭 범위를 가집니다.

Each station has its own K-to-amplitude lookup table, calibrated for its geomagnetic latitude. Higher-latitude stations have larger amplitude ranges for the same K value.

#### 4.2 Kp 지수 / The Kp (Planetary K) Index

Bartels의 핵심 혁신 — **전 지구적(planetary)** 활동 지수:

Bartels' key innovation — a **planetary** activity index:

**산출 과정 / Calculation process**:
1. 전 세계에 분포한 ~13개 중위도 관측소에서 각각 $K$ 값을 산출
   Compute $K$ values at ~13 mid-latitude stations worldwide
2. 각 관측소의 $K$를 **표준화(standardize)** — $K_s$ (standardized K)
   **Standardize** each station's $K$ → $K_s$, using station-specific conversion tables to correct for latitude/longitude/local anomaly differences
3. 표준화된 $K_s$ 값들의 **평균** → $K_p$
   **Average** the standardized $K_s$ values → $K_p$

$$K_p = \frac{1}{N} \sum_{i=1}^{N} K_{s,i}$$

$K_p$는 0에서 9까지의 값을 가지며, 1/3 단위 세분화: $0_o, 0_+, 1_-, 1_o, 1_+, ..., 9_-, 9_o$ (총 28단계).

**왜 중위도 관측소만 사용하는가? / Why only mid-latitude stations?**
- 고위도: 오로라 전류의 국소적 교란이 너무 커서 전 지구적 활동을 대표하지 못함
  High latitude: local auroral current disturbances too large to represent global activity
- 저위도: 적도 전류제트(equatorial electrojet)의 영향이 과다
  Low latitude: excessive equatorial electrojet influence
- 중위도(약 $40°$–$60°$ 지자기 위도): 전 지구적 교란을 가장 잘 대표
  Mid-latitude (~$40°$–$60°$ geomagnetic): best represents global disturbance

#### 4.3 ap 및 Ap 지수 / The ap and Ap Indices

$K_p$는 준대수 척도이므로 **산술 연산(평균, 합산)에 적합하지 않습니다**. 이를 보완하기 위해 **ap 지수**를 정의합니다:

$K_p$ uses a quasi-logarithmic scale, unsuitable for arithmetic operations. The **ap index** provides a linear equivalent:

$$K_p \rightarrow a_p \text{ (변환 표 사용)}$$

| $K_p$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| $a_p$ (nT) | 0 | 3 | 7 | 15 | 27 | 48 | 80 | 132 | 207 | 400 |

일일 평균:

$$A_p = \frac{1}{8} \sum_{i=1}^{8} a_{p,i}$$

$A_p$는 **해당 일의 전체 지자기 활동 수준**을 하나의 숫자로 나타냅니다.

#### 4.4 Dst 지수의 수학적 기초 / Mathematical Foundation of the Dst Index

Chapman과 Bartels는 자기 폭풍의 **축대칭(axially symmetric)** 교란을 분리하는 방법을 제시합니다. 이것이 이후 **Sugiura (1964)**가 정식화한 Dst 지수의 원형입니다:

$$D_{st}(t) = \frac{\sum_{i=1}^{N} \Delta H_i(t) / \cos\lambda_i}{N}$$

여기서 / where:
- $\Delta H_i(t)$ = $i$번째 관측소에서 관측된 $H$ 변동 (조용한 날 기준값 차감) / Observed $H$ variation at station $i$ (quiet-day baseline subtracted)
- $\lambda_i$ = $i$번째 관측소의 지자기 위도 / Geomagnetic latitude of station $i$
- $\cos\lambda_i$ 보정: 쌍극자 자기장 하에서 수평 성분은 $\cos\lambda$에 비례하므로, 이를 보정하여 적도에서의 등가 교란으로 환산 / Correction: $H$ scales as $\cos\lambda$ under dipole field, normalizing to equatorial equivalent disturbance

$D_{st}$의 물리적 해석: 음의 $D_{st}$는 지구를 감싸는 **서향 환전류(westward ring current)**의 존재를 의미합니다. 이 전류가 만드는 자기장이 지구 쌍극자 자기장을 **약화**시키기 때문입니다.

Negative $D_{st}$ indicates a **westward ring current** encircling Earth, whose field weakens the dipole field at the surface. The $\cos\lambda$ correction normalizes to equatorial equivalent disturbance.

---

### Part V. 태양 활동과 지자기 활동의 관계 / Solar-Geomagnetic Correlations

#### 5.1 태양흑점 주기와 지자기 활동 / Sunspot Cycle and Geomagnetic Activity

Chapman과 Bartels는 광범위한 통계 분석을 통해 다음을 확립합니다:

Through extensive statistical analysis, Chapman and Bartels establish the following:

- 지자기 활동($A_p$, 폭풍 발생 빈도)은 **태양흑점 수(sunspot number, $R$)**와 양의 상관관계
  Geomagnetic activity ($A_p$, storm frequency) is positively correlated with **sunspot number ($R$)**
- 그러나 상관관계는 **불완전** — 태양 극대기에도 조용한 시기가 있고, 극소기에도 때로 폭풍이 발생
  However, the correlation is **imperfect** — quiet periods occur even at solar maximum, and storms occasionally occur at minimum
- 자기 활동 극대는 태양흑점 극대보다 약간 **지연** (1–2년)
  Geomagnetic activity maximum is slightly **delayed** (1–2 years) relative to sunspot maximum
- 태양 활동 하강기에 **27일 재현 경향**이 강해짐 (M-region 효과)
  **27-day recurrence tendency** strengthens during the declining phase of solar activity (M-region effect)

이러한 불완전한 상관관계는 이후 **코로나 질량 방출(CME)**과 **코로나 홀(coronal hole)**이라는 두 가지 서로 다른 태양풍 구동원의 발견으로 설명됩니다.

The imperfect correlation was later explained by the discovery of two distinct solar wind drivers: **CMEs** (correlated with sunspot cycle) and **coronal holes** (more prominent in declining phase).

#### 5.2 자기 폭풍과 태양 플레어 / Magnetic Storms and Solar Flares

Carrington (1859)의 관측 이후, 태양 플레어와 자기 폭풍의 연관성이 알려져 있었습니다. Chapman과 Bartels는 이 연관성을 통계적으로 분석합니다:

- 모든 큰 플레어가 자기 폭풍을 유발하지는 않음
  Not all large flares cause magnetic storms
- 태양 원반의 **중앙 부근**에서 발생한 플레어가 지자기 효과가 강함 (지구를 향한 방향)
  Flares near the **center of the solar disk** have stronger geomagnetic effects (Earth-directed)
- 플레어 발생 후 자기 폭풍까지의 **지연 시간**: ~1–2일 → 태양-지구 거리를 이 시간으로 나누면 약 1000 km/s (태양 물질의 이동 속도 추정)
  **Delay time** from flare to storm: ~1–2 days → dividing the Sun-Earth distance by this time gives ~1000 km/s (estimated transit speed of solar material)

이 지연 시간 분석은 이후 Parker (1958)의 태양풍 이론을 지지하는 관측적 증거가 됩니다.

This delay-time analysis became observational evidence supporting Parker's (1958) solar wind theory.

---

## 핵심 시사점 / Key Takeaways

1. **구면 조화 분석은 지자기학의 로제타석이다** — 구면 위의 관측만으로 자기장의 **내부/외부 기원을 분리**할 수 있다는 것은 Gauss의 천재적 통찰이며, Chapman과 Bartels는 이를 실용적 방법론으로 완성했습니다. 이 방법은 현재 IGRF(International Geomagnetic Reference Field) 모델의 기초입니다.
   Spherical harmonic analysis is the Rosetta Stone of geomagnetism — separating internal/external sources from surface observations alone. Chapman and Bartels completed Gauss's insight into a practical methodology, now the basis of IGRF models.

2. **Kp 지수는 데이터 과학의 선구적 사례이다** — 서로 다른 위치, 감도, 배경 잡음을 가진 관측소들의 데이터를 **하나의 표준화된 지표**로 통합하는 Bartels의 방법론은 현대 데이터 과학의 정규화(normalization)와 앙상블(ensemble) 기법을 선취합니다.
   The Kp index is a pioneering case of data science — Bartels' methodology for integrating data from diverse observatories into a single standardized metric anticipates modern normalization and ensemble techniques.

3. **준대수 척도의 선택은 물리에 기반한 통계적 결정이다** — 지자기 교란의 로그-정규 분포를 반영하여 준대수 척도를 채택한 것은, 데이터의 통계적 특성을 이해한 위에 측정 체계를 설계한 모범 사례입니다.
   The quasi-logarithmic scale reflects the log-normal distribution of geomagnetic disturbances — a model of designing measurement systems grounded in statistical understanding of the data.

4. **27일 재현과 M-region은 20년을 앞선 예측이다** — Bartels가 태양 표면에서 보이지 않는 "M-region"이 반복적 지자기 교란의 원인이라고 추론한 것은, 1970년대 Skylab에 의해 발견된 **코로나 홀(coronal hole)**의 사실상 예측입니다.
   Bartels' inference of invisible "M-regions" on the Sun causing recurring activity was effectively a prediction of **coronal holes**, discovered 30 years later by Skylab.

5. **자기 폭풍의 3단계 구조 분류는 영구적이다** — Initial phase (SC) → Main phase → Recovery phase의 분류는 이 모노그래프에서 관측적으로 확립되었으며, 80년이 지난 현재까지도 자기 폭풍 연구의 표준 프레임워크입니다.
   The three-phase classification (SC → Main → Recovery) was observationally established here and remains the standard framework for magnetic storm research 80+ years later.

6. **$D_{st}$와 $S_D$의 분리는 물리적 원인의 분리이다** — 축대칭 교란($D_{st}$, 환전류)과 local-time 의존 교란($S_D$, 부분 환전류·전리층 전류)의 분리는 단순한 수학적 조작이 아니라, **서로 다른 물리적 전류 체계**의 기여를 구별하는 것입니다.
   The separation of $D_{st}$ (axisymmetric, ring current) from $S_D$ (local-time-dependent, partial ring current + ionospheric currents) distinguishes contributions from **different physical current systems**, not just a mathematical decomposition.

7. **이 모노그래프는 우주 시대 이전의 최고 걸작이다** — 위성도 없고, 태양풍 개념도 없고, 자기권이라는 단어조차 없던 시대에, **지상 관측만으로** 이토록 정교한 분석 체계를 구축한 것은 경이로운 성취입니다. 이후 위성 관측이 이 체계의 물리적 원인을 밝혔을 때, Chapman-Bartels의 수학적 틀은 거의 수정 없이 그대로 사용되었습니다.
   Built entirely from ground observations — without satellites, solar wind concept, or even the word "magnetosphere" — this monograph's analytical framework required almost no modification when satellite observations later revealed the physical causes.

8. **Chapman과 Bartels의 역할 분담은 이론-관측 협업의 모범이다** — Chapman은 수학적 분석과 물리적 해석을, Bartels는 통계적 방법론과 지수 체계를 담당했습니다. 이론가와 관측/통계 전문가의 **보완적 전문성**이 만들어낸 시너지의 전형입니다.
   Chapman contributed mathematical analysis and physical interpretation; Bartels contributed statistical methodology and index systems — a textbook case of complementary expertise between theorist and observationalist/statistician.

---

## 수학적 요약 / Mathematical Summary

### 지구 자기장의 핵심 방정식 / Core Equations of Geomagnetism

**1. 구면 조화 전개 (자기 퍼텐셜)**:

$$V = a \sum_{n=1}^{\infty} \sum_{m=0}^{n} \left[\left(\frac{a}{r}\right)^{n+1} (g_n^m \cos m\phi + h_n^m \sin m\phi) + \left(\frac{r}{a}\right)^n (q_n^m \cos m\phi + s_n^m \sin m\phi)\right] P_n^m(\cos\theta)$$

**2. 자기장 성분 (구면 좌표)**:

$$B_r = -\frac{\partial V}{\partial r}, \quad B_\theta = -\frac{1}{r}\frac{\partial V}{\partial \theta}, \quad B_\phi = -\frac{1}{r\sin\theta}\frac{\partial V}{\partial \phi}$$

**3. 쌍극자 자기장**:

$$B_r = -\frac{2M\cos\theta}{r^3}, \quad B_\theta = -\frac{M\sin\theta}{r^3}, \quad \tan I = 2\cot\theta$$

**4. 쌍극자 모멘트**:

$$M = a^3 \sqrt{(g_1^0)^2 + (g_1^1)^2 + (h_1^1)^2}$$

**5. 각 차수의 자기장 에너지 스펙트럼 (Lowes-Mauersberger spectrum)**:

$$R_n = (n+1) \sum_{m=0}^{n} \left[(g_n^m)^2 + (h_n^m)^2\right]$$

**6. Dst 지수**:

$$D_{st}(t) = \frac{1}{N}\sum_{i=1}^{N} \frac{\Delta H_i(t)}{\cos\lambda_i}$$

**7. Kp → ap 변환 (준대수 → 선형)**:

$$a_p = f(K_p) \quad \text{(변환 표 사용)}$$

**8. $S_q$ 변동의 구면 조화 전개**:

$$S_q(\theta, \tau) = \sum_{m=1}^{4}\sum_{n=m}^{N} (\alpha_n^m \cos m\tau + \beta_n^m \sin m\tau) P_n^m(\cos\theta)$$

**9. 자기 폭풍 분해**:

$$\Delta H(t, \lambda, \tau) = D_{st}(t)\cos\lambda + S_D(\tau, \lambda)$$

**10. 영년 변화 (1차 근사)**:

$$g_n^m(t) = g_n^m(t_0) + \dot{g}_n^m \cdot (t - t_0)$$

---

## 역사적 맥락의 타임라인 / Paper in the Arc of History

```
1600  Gilbert ─ De Magnete (지구 = 자석)
  │
1701  Halley ─ 최초의 등편각선도 (declination chart) 작성
  │
1838  Gauss ─ 구면 조화 분석으로 지구 자기장의 내/외부 기원 분리
  │          $g_n^m$, $h_n^m$ 최초 결정
  │
1852  Sabine ─ 지자기 활동 ↔ 태양흑점 주기 상관관계 발견
  │
1859  Carrington ─ 태양 플레어 → 자기 폭풍 최초 관측
  │
1882  Balfour Stewart ─ 일변화의 원인: 지구 외부 전류 체계 제안
  │
1908  ★ Birkeland ─ Norwegian Aurora Polaris Expedition (#1)
  │
1919  Lindemann ─ 태양 방출물 = 전하 중성 이온화 가스
  │
1931  ★ Chapman & Ferraro ─ 자기 폭풍의 정량적 이론 (#2)
  │        유도 전류 → 자기장 차폐 → cavity (자기권의 예측)
  │
1940  ★★★ Chapman & Bartels ─ "Geomagnetism" (#3) ← 이 모노그래프
  │        지자기학의 종합적 체계화
  │        구면 조화 분석의 표준화
  │        Kp, Dst 지수의 확립
  │        자기 폭풍 3단계 분류
  │        27일 재현과 M-region 발견
  │
1942  Alfvén ─ MHD 파동 이론 (Alfvén wave)
  │
1957  IGY ─ 국제 지구물리의 해 (전 지구 관측 네트워크 확장)
  │
1958  Parker ─ 태양풍 이론 (#4 다음 논문)
  │        M-region의 물리적 실체 해명의 시작
  │
1958  Van Allen ─ 방사선대 발견 (#5)
  │
1964  Sugiura ─ Dst 지수의 현대적 정식화
  │        Chapman-Bartels의 수학적 기초 위에 구축
  │
1968  IAGA ─ IGRF 최초 공식 발표
  │        Gauss 계수의 국제 표준 결정
  │
1975  Burton et al. ─ Dst-태양풍 경험적 관계식 (#11)
  │
1989  Tsyganenko ─ Kp 매개변수 자기장 모델 (T89) (#14)
  │        Bartels의 Kp를 자기장 모델링의 입력으로 사용
  │
2019  IGRF-13 ─ 차수 n=13까지의 Gauss 계수 결정
  │        Chapman-Bartels의 방법론이 80년간 유지
  │
현재  Swarm 위성 → IGRF-14 준비, 실시간 Kp/Dst 서비스 운영
```

---

## 다른 논문과의 연결 / Connections to Other Papers

| Paper / 논문 | Relationship / 관계 |
|---|---|
| **Birkeland (1908)** — #1 | Birkeland의 극지 관측 데이터가 이 모노그래프의 오로라 및 자기 교란 분석의 출발점. Birkeland이 관측한 현상에 Chapman과 Bartels가 수학적 분석 틀을 제공 |
| **Chapman & Ferraro (1931)** — #2 | 자기 폭풍의 물리적 메커니즘(유도 전류, 자기장 차폐). 이 모노그래프에서는 같은 폭풍을 **관측/통계적 관점**에서 분류하고 지수화함 — 물리와 관측의 양면을 완성 |
| **Parker (1958)** — #4 다음 논문 | 태양풍 이론. Chapman-Bartels의 27일 재현과 M-region을 설명하는 물리적 기초 제공. 태양풍이 연속 흐름임을 밝혀 자기권이 상시 구조임을 확립 |
| **Van Allen (1958)** — #5 | 방사선대 발견. Chapman-Bartels의 구면 조화 분석과 쌍극자 모델이 입자 포획 영역 예측의 수학적 기초를 제공 |
| **Burton et al. (1975)** — #11 | Chapman-Bartels의 $D_{st}$ 분해를 바탕으로, 태양풍 매개변수와 Dst의 경험적 관계식을 도출. 지자기 폭풍 예보의 시작 |
| **Tsyganenko (1989)** — #14 | Bartels의 Kp 지수를 입력 매개변수로 사용한 최초의 경험적 자기장 모델(T89). Kp가 자기권 상태의 프록시로 기능함을 입증 |
| **Gonzalez et al. (1994)** — #15 | Chapman-Bartels의 자기 폭풍 분류를 현대화 — Dst 임계값에 의한 정량적 폭풍 등급화 (moderate: -50 ~ -100 nT, intense: -100 ~ -250 nT, super: < -250 nT) |
| **Gauss (1839)** | 구면 조화 분석의 창시자. Chapman과 Bartels가 이 방법을 체계화하고 현대적 형태로 완성 |
| **Sugiura (1964)** | Chapman-Bartels의 $D_{st}$ 수학적 기초 위에 현대적 Dst 지수를 정식화. 실시간 산출 방법 확립 |

---

## 부록: 지자기 지수 종합 해설 / Appendix: Comprehensive Guide to Geomagnetic Indices

Chapman & Bartels (1940)에서 수학적 기초가 확립된 지자기 지수들과, 이후 우주 시대에 발전된 현대 지수들을 정리합니다.

This appendix summarizes the geomagnetic indices whose mathematical foundations were established in Chapman & Bartels (1940), along with modern indices developed in the Space Age.

### A.1 K 지수 / K Index

**개발자 / Developer**: Julius Bartels (1938)
**목적 / Purpose**: 개별 관측소에서의 **국소 지자기 교란 수준**을 정량화 / Quantify **local geomagnetic disturbance level** at individual observatories

**정의 / Definition**: 3시간 간격으로 자기장의 수평 성분($H$, $D$)에서 **조용한 날 변동($S_q$)을 제거한 후**, 잔여 교란의 최대-최소 범위(range)를 측정하여 0~9의 정수로 변환한 값.

The residual disturbance range (max minus min) of the horizontal components ($H$, $D$) over each 3-hour interval, after removing the quiet-day variation ($S_q$), converted to an integer 0–9.

**산출 과정 / Calculation process**:

1. 하루를 8개 구간으로 나눔 (00-03, 03-06, ..., 21-24 UT)
   Divide the day into 8 intervals (00-03, 03-06, ..., 21-24 UT)
2. 각 구간에서 $S_q$ (quiet-day solar variation)를 차감
   Subtract $S_q$ (quiet-day solar variation) from each interval
3. 잔여 교란의 range 측정:
   Measure the range of the residual disturbance:

$$R = \max(\Delta H) - \min(\Delta H) \quad \text{(3시간 구간 내 / within 3-hour interval)}$$

4. $R$을 **준대수 척도(quasi-logarithmic scale)**로 변환:
   Convert $R$ to a **quasi-logarithmic scale**:

| $K$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|-----|---|---|---|---|---|---|---|---|---|---|
| 하한 / Lower limit (nT) | 0 | 5 | 10 | 20 | 40 | 70 | 120 | 200 | 330 | 500 |

> **주의 / Note**: 위 임계값은 **중위도 표준 관측소** 기준입니다. 각 관측소는 자신의 지자기 위도에 맞게 **개별 변환표**를 가집니다. 고위도 관측소는 같은 $K$에 대해 더 큰 진폭 범위를 가집니다.
>
> The thresholds above are for a **mid-latitude standard station**. Each station has its own **individual conversion table** calibrated for its geomagnetic latitude. Higher-latitude stations have larger amplitude ranges for the same $K$.

**왜 준대수 척도인가? / Why quasi-logarithmic?**
- 지자기 교란의 진폭이 **로그-정규 분포(log-normal distribution)**에 가깝기 때문
  Geomagnetic disturbance amplitudes follow approximately **log-normal distributions**
- 선형 척도에서는 대부분의 날이 $K=0$~$1$에 몰려 분해능이 낮아짐
  On a linear scale, most days would cluster at $K=0$–$1$, reducing resolution
- 각 단계마다 진폭이 약 2배씩 증가하여, 약한 교란과 강한 교란을 **동등한 분해능**으로 포착
  Amplitude roughly doubles at each step, capturing weak and strong disturbances with **equal resolution**

---

### A.2 Kp 지수 / Kp Index (Planetary K Index)

**개발자 / Developer**: Julius Bartels (1949)
**목적 / Purpose**: **전 지구적(planetary)** 지자기 활동 수준을 하나의 숫자로 표현 / Express **global (planetary)** geomagnetic activity as a single number

**정의 / Definition**: 전 세계 ~13개 **중위도** 관측소의 $K$ 값을 표준화(standardize)한 후 평균한 값.

The average of standardized $K$ values from ~13 **mid-latitude** observatories worldwide.

**산출 과정 / Calculation process**:

1. 각 관측소 $i$에서 $K_i$ 산출
   Compute $K_i$ at each station $i$
2. 관측소별 **표준화 변환표**를 적용하여 $K_{s,i}$ (standardized K) 산출 — 위도, 경도, 지역 지질 이상에 의한 관측소 간 차이를 보정
   Apply station-specific **standardization tables** → $K_{s,i}$, correcting for inter-station differences due to latitude, longitude, and local geological anomalies
3. 표준화된 값들의 평균:
   Average the standardized values:

$$K_p = \frac{1}{N} \sum_{i=1}^{N} K_{s,i}$$

4. 결과를 **1/3 단위**로 세분화: $0_o, 0_+, 1_-, 1_o, 1_+, ..., 9_-, 9_o$ (총 28단계)
   Results are subdivided into **1/3 steps**: $0_o, 0_+, 1_-, 1_o, 1_+, ..., 9_-, 9_o$ (28 levels total)

**왜 중위도 관측소만 사용하는가? / Why only mid-latitude stations?**
- **고위도 / High latitude**: auroral electrojet의 국소 교란이 너무 커서 전 지구적 활동을 대표하지 못함 / Local auroral electrojet disturbances too large to represent global activity
- **저위도 / Low latitude**: equatorial electrojet의 영향이 과다 / Excessive equatorial electrojet influence
- **중위도 / Mid-latitude** (~40°-60° 지자기 위도): 전 지구적 교란을 가장 잘 대표 / Best represents global disturbance

**현재 운영 / Current operation**: GFZ Potsdam (독일 / Germany)

---

### A.3 ap 지수 및 Ap 지수 / ap and Ap Indices

**목적 / Purpose**: $K_p$는 준대수 척도라 **산술 연산(평균, 합산)이 물리적으로 의미 없음**. 이를 선형(linear) 등가로 변환한 것이 $a_p$.

$K_p$ uses a quasi-logarithmic scale, making **arithmetic operations (averaging, summing) physically meaningless**. The $a_p$ index provides a linear equivalent.

**변환표 / Conversion table**:

| $K_p$ | 0 | 0+ | 1- | 1 | 1+ | 2- | 2 | 2+ | 3- | 3 | 3+ | 4- | 4 |
|-------|---|----|----|----|----|----|---|----|----|----|----|----|---|
| $a_p$ (nT) | 0 | 2 | 3 | 4 | 5 | 6 | 7 | 9 | 12 | 15 | 18 | 22 | 27 |

| $K_p$ | 4+ | 5- | 5 | 5+ | 6- | 6 | 6+ | 7- | 7 | 7+ | 8- | 8 | 8+ | 9- | 9 |
|-------|----|----|---|----|----|----|----|----|---|----|----|----|----|----|---|
| $a_p$ (nT) | 32 | 39 | 48 | 56 | 67 | 80 | 94 | 111 | 132 | 154 | 179 | 207 | 236 | 300 | 400 |

**Ap 지수 / Ap Index** (일일 평균 / daily average):

$$A_p = \frac{1}{8} \sum_{j=1}^{8} a_{p,j}$$

해당 일의 **전체 지자기 활동 수준**을 하나의 숫자(nT 단위)로 나타냅니다.

Represents the **overall geomagnetic activity level** of the day as a single number in nT.

**활동 등급 / Activity classification**:

| $A_p$ 범위 / Range | 활동 수준 / Activity Level |
|-----------|--------------------------|
| 0–7 | Quiet / 조용 |
| 8–15 | Unsettled / 약간 불안정 |
| 16–29 | Active / 활동적 |
| 30–49 | Minor storm / 소폭풍 |
| 50–99 | Major storm / 대폭풍 |
| 100–400 | Severe storm / 극심한 폭풍 |

---

### A.4 Dst 지수 / Dst Index (Disturbance Storm Time)

**개발자 / Developer**: 수학적 기초 Chapman & Bartels (1940), 현대적 정식화 Sugiura (1964)
Mathematical foundation by Chapman & Bartels (1940), modern formalization by Sugiura (1964)
**목적 / Purpose**: **환전류(ring current)** 강도의 척도 — 자기 폭풍의 강도를 정량화 / Measure **ring current** intensity — quantify magnetic storm strength

**정의 / Definition**: 저위도 관측소들에서 측정한 자기장 수평 성분($H$)의 축대칭(axially symmetric) 교란을 $\cos\lambda$ 보정하여 평균한 값.

The average of $\cos\lambda$-corrected axially symmetric disturbance in the horizontal component ($H$) from low-latitude observatories.

**산출 공식 / Formula**:

$$D_{st}(t) = \frac{1}{N} \sum_{i=1}^{N} \frac{\Delta H_i(t)}{\cos \lambda_i}$$

여기서 / where:
- $\Delta H_i(t) = H_{\text{obs},i}(t) - H_{\text{quiet},i}(t)$ : 관측값에서 조용한 날 기준값을 뺀 교란 / Observed value minus quiet-day baseline
- $\lambda_i$ : $i$번째 관측소의 지자기 위도 / Geomagnetic latitude of station $i$
- $\cos\lambda_i$ 보정: 쌍극자 자기장 하에서 환전류의 수평 성분 효과가 $\cos\lambda$에 비례하므로, 적도 등가 교란으로 환산 / Correction: ring current's horizontal effect scales as $\cos\lambda$, normalizing to equatorial equivalent

**물리적 의미 / Physical meaning**:
- **$D_{st} > 0$**: 자기장 압축 (태양풍 동압 증가, SC) / Field compression (solar wind ram pressure increase, SC)
- **$D_{st} < 0$**: 환전류 강화 → 지표면 자기장 약화 (main phase) / Ring current enhancement → surface field weakening
- **$D_{st} \approx 0$**: 조용한 상태 / Quiet conditions

**폭풍 분류 / Storm classification** (Gonzalez et al., 1994):

| $D_{st}$ 최솟값 / Minimum | 분류 / Classification |
|----------------|----------------------|
| $-30$ ~ $-50$ nT | Weak storm / 약한 폭풍 |
| $-50$ ~ $-100$ nT | Moderate storm / 중간 폭풍 |
| $-100$ ~ $-250$ nT | Intense storm / 강한 폭풍 |
| $< -250$ nT | Super storm / 초강력 폭풍 |

**사용 관측소 / Observatories used**: 4개 저위도 관측소 / 4 low-latitude stations
- Honolulu (HON, 21°N), San Juan (SJG, 18°N), Hermanus (HER, 34°S), Kakioka (KAK, 36°N)

**시간 해상도 / Time resolution**: 1시간 / 1 hour
**현재 운영 / Current operation**: WDC for Geomagnetism, Kyoto (일본 교토대학 / Kyoto University, Japan)

---

### A.5 SYM-H 및 ASY-H 지수 / SYM-H and ASY-H Indices

**개발자 / Developer**: Iyemori (1990)
**목적 / Purpose**: Dst의 **고시간분해능 버전** (1분) 및 비대칭 성분 분리 / **High-time-resolution version** of Dst (1-min) and asymmetric component separation

**SYM-H**: Dst와 개념적으로 동일하나 **1분 해상도** / Conceptually identical to Dst but with **1-minute resolution**

$$\text{SYM-H}(t) = \frac{1}{N} \sum_{i=1}^{N} \frac{\Delta H_i(t)}{\cos\lambda_i}$$

**ASY-H**: 환전류의 **비대칭(asymmetric) 성분** — local time에 따른 편차 / **Asymmetric component** of the ring current — deviation with local time

$$\text{ASY-H}(t) = \max_i\left(\frac{\Delta H_i}{\cos\lambda_i}\right) - \min_i\left(\frac{\Delta H_i}{\cos\lambda_i}\right)$$

- ASY-H가 크면 **부분 환전류(partial ring current)**가 강한 것
  Large ASY-H indicates a strong **partial ring current**
- Main phase 초기에 ASY-H가 크고, recovery phase에서 감소 → 환전류가 대칭화
  ASY-H is large early in the main phase, decreasing in recovery → ring current symmetrization

**관계 / Relation**: $\text{SYM-H} \approx D_{st}$ (약간의 차이는 관측소 선택과 기준값 방법의 차이에서 기인 / slight differences arise from observatory selection and baseline methods)

---

### A.6 AE, AU, AL 지수 / AE, AU, AL Indices (Auroral Electrojet Indices)

**개발자 / Developer**: Davis & Sugiura (1966)
**목적 / Purpose**: **오로라대 전류(auroral electrojet)**의 강도 측정 — substorm 활동의 척도 / Measure **auroral electrojet** intensity — proxy for substorm activity

**산출 방법 / Calculation method**:

1. 오로라대(지자기 위도 ~65°-70°)에 위치한 **12개 관측소**에서 $H$ 성분 측정
   Measure $H$ component at **12 stations** in the auroral zone (~65°-70° geomagnetic latitude)
2. 각 관측소에서 조용한 날 기준값 차감 → $\Delta H_i(t)$
   Subtract quiet-day baseline at each station → $\Delta H_i(t)$
3. 모든 관측소 중에서:
   From all stations:

$$AU(t) = \max_i \left[\Delta H_i(t)\right] \quad \text{(상한 포락선 / upper envelope)}$$

$$AL(t) = \min_i \left[\Delta H_i(t)\right] \quad \text{(하한 포락선 / lower envelope)}$$

$$AE(t) = AU(t) - AL(t)$$

$$AO(t) = \frac{AU(t) + AL(t)}{2}$$

**물리적 의미 / Physical meaning**:
- **AU > 0**: **동향(eastward) electrojet** — 오후 측 전리층 전류 / Afternoon-side ionospheric current
- **AL < 0**: **서향(westward) electrojet** — 자정 측 전리층 전류 (substorm과 직결) / Midnight-side ionospheric current (directly linked to substorms)
- **AE**: 전체 auroral electrojet 활동의 척도 / Measure of total auroral electrojet activity
- **AO**: 등가 영역(zonal) 전류 / Equivalent zonal current

**substorm과의 관계 / Relation to substorms**:
- Substorm expansion phase 시작 시 $AL$이 급격히 감소 (수백~수천 nT)
  $AL$ drops sharply (hundreds to thousands of nT) at substorm expansion phase onset
- $AE$의 급격한 증가 = substorm onset의 지표
  Sharp increase in $AE$ = indicator of substorm onset

**시간 해상도 / Time resolution**: 1분 / 1 minute
**현재 운영 / Current operation**: WDC Kyoto

---

### A.7 aa 지수 / aa Index

**개발자 / Developer**: Mayaud (1972)
**목적 / Purpose**: **가장 긴 시간 범위의 지자기 활동 기록** (1868년~현재, 150년+) / **Longest continuous geomagnetic activity record** (1868–present, 150+ years)

**정의 / Definition**: 지자기적으로 대략 대척점(antipodal)에 위치한 **2개 관측소**의 $K$ 값을 평균한 3시간 지수.

A 3-hourly index averaging $K$ values from **2 approximately antipodal observatories**.

$$aa = \frac{a_{K,\text{north}} + a_{K,\text{south}}}{2}$$

**관측소 / Observatories**:
- 북반구 / Northern hemisphere: Hartland (영국 / UK), 이전 Greenwich → Abinger → Hartland
- 남반구 / Southern hemisphere: Canberra (호주 / Australia), 이전 Melbourne → Toolangi → Canberra

**장점 / Advantage**: 관측소가 2개뿐이라 단순하지만, 1868년까지 소급 가능 → **태양 주기와 장기 추세 분석**에 핵심적

Only 2 stations keeps it simple, but the record extends back to 1868 → essential for **solar cycle and long-term trend analysis**

---

### A.8 PC 지수 / PC Index (Polar Cap Index)

**개발자 / Developer**: Troshichev & Andrezen (1985)
**목적 / Purpose**: **극관(polar cap)**을 통과하는 에너지 유입의 척도 / Measure of energy input through the **polar cap**

**정의 / Definition**: 극지방 관측소(Thule, Vostok)에서 측정한 자기장 변동을 **최적 방향(optimal direction)**으로 투사한 값.

Magnetic field variations at polar stations (Thule, Vostok) projected onto the **optimal direction**.

$$PC = \frac{\Delta F_{\text{proj}} - \beta}{\alpha}$$

여기서 / where:
- $\Delta F_{\text{proj}}$ : 최적 방향으로 투사된 자기장 변동 / Field variation projected onto optimal direction
- $\alpha$, $\beta$ : 통계적으로 결정된 정규화 계수 / Statistically determined normalization coefficients

**물리적 의미 / Physical meaning**: 태양풍의 **merging electric field** ($E_m$)와 높은 상관관계를 가지므로, 태양풍-자기권 결합의 프록시로 사용됩니다.

Highly correlated with the solar wind **merging electric field** ($E_m$), serving as a proxy for solar wind-magnetosphere coupling.

$$E_m = V_{sw} \cdot B_T \cdot \sin^2(\theta_c / 2)$$

여기서 / where: $V_{sw}$ = 태양풍 속도 / solar wind speed, $B_T$ = IMF 횡단 성분 / IMF transverse component, $\theta_c$ = clock angle

---

### A.9 지수 간 관계 요약 / Summary of Inter-Index Relationships

```
태양풍 / Solar wind → 자기권 / Magnetosphere → 전리층 / Ionosphere → 지상 관측 / Ground obs.
  │                      │                        │                     │
  │                 Ring current              Electrojet               │
  │                      │                        │                     │
  PC ←──────────── Dst/SYM-H              AE/AU/AL               K/Kp/ap
  │                      │                        │                     │
  └── 에너지 유입          └── 폭풍 강도             └── substorm 활동      └── 전체 활동 수준
      Energy input           Storm intensity          Substorm activity      Overall activity
```

| 지수 / Index | 측정 대상 / What it measures | 시간 분해능 / Resolution | 위도 범위 / Latitude |
|------|---------------------------|-----------|---------|
| **K, Kp** | 전체 지자기 교란 수준 / Overall disturbance level | 3시간 / 3 hours | 중위도 / Mid-latitude |
| **ap, Ap** | Kp의 선형 등가 / Linear equivalent of Kp | 3시간·1일 / 3 hours·daily | 중위도 / Mid-latitude |
| **Dst** | 환전류 강도 (폭풍 세기) / Ring current intensity (storm strength) | 1시간 / 1 hour | 저위도 / Low-latitude |
| **SYM-H** | Dst의 1분 버전 / 1-min Dst equivalent | 1분 / 1 min | 저위도 / Low-latitude |
| **ASY-H** | 환전류 비대칭 / Ring current asymmetry | 1분 / 1 min | 저위도 / Low-latitude |
| **AE** | Auroral electrojet 전체 활동 / Total electrojet activity | 1분 / 1 min | 오로라대 / Auroral zone |
| **AU** | 동향 electrojet / Eastward electrojet | 1분 / 1 min | 오로라대 / Auroral zone |
| **AL** | 서향 electrojet (substorm) / Westward electrojet | 1분 / 1 min | 오로라대 / Auroral zone |
| **aa** | 장기 활동 기록 / Long-term activity record | 3시간 / 3 hours | 중위도 2개소 / Mid-lat. (2 stations) |
| **PC** | 극관 에너지 유입 / Polar cap energy input | 1분 / 1 min | 극관 / Polar cap |

이 지수들 중 **K, Kp, Dst의 수학적 기초**가 바로 이 모노그래프 — Chapman & Bartels (1940)에서 확립되었으며, 나머지는 우주 시대(1960년대 이후) 위성 관측과 함께 발전했습니다. Paper #11 Burton et al. (1975)에서는 Dst와 태양풍 매개변수의 경험적 관계식을, Paper #14 Tsyganenko (1989)에서는 Kp를 입력으로 사용한 자기장 모델을 다루게 됩니다.

Among these, the **mathematical foundations of K, Kp, and Dst** were established in this monograph — Chapman & Bartels (1940). The rest developed alongside satellite observations in the Space Age (post-1960s). Paper #11 Burton et al. (1975) derives the empirical Dst–solar wind relation, and Paper #14 Tsyganenko (1989) uses Kp as input to a magnetic field model.

---

## 참고문헌 / References

- Chapman, S. and Bartels, J., *Geomagnetism*, 2 vols., Oxford University Press (Clarendon Press), 1940.
- Gauss, C.F., "Allgemeine Theorie des Erdmagnetismus," in *Resultate aus den Beobachtungen des magnetischen Vereins im Jahre 1838*, pp. 1–57, 1839.
- Chapman, S. and Ferraro, V.C.A., "A New Theory of Magnetic Storms," *Terrestrial Magnetism and Atmospheric Electricity*, Vol. 36, pp. 77–97, 171–186, 1931.
- Bartels, J., "Terrestrial-magnetic activity and its relations to solar phenomena," *Terrestrial Magnetism and Atmospheric Electricity*, Vol. 37, pp. 1–52, 1932.
- Sugiura, M., "Hourly values of equatorial Dst for the IGY," *Annals of the International Geophysical Year*, Vol. 35, pp. 9–45, 1964.
- Parker, E.N., "Dynamics of the Interplanetary Gas and Magnetic Fields," *Astrophysical Journal*, Vol. 128, pp. 664–676, 1958.
- Lowes, F.J., "Spatial power spectrum of the main geomagnetic field, and extrapolation to the core," *Geophysical Journal International*, Vol. 36, pp. 717–730, 1974.
- Mauersberger, P., "Das Mittel der Energiedichte des geomagnetischen Hauptfeldes an der Erdoberfläche und seine säkulare Änderung," *Gerlands Beiträge zur Geophysik*, Vol. 65, pp. 207–215, 1956.
- Thébault, E. et al., "International Geomagnetic Reference Field: the 13th generation," *Earth, Planets and Space*, Vol. 67, 79, 2015.
- Bartels, J., "The technique of scaling indices K and Q of geomagnetic activity," *Annals of the International Geophysical Year*, Vol. 4, pp. 215–226, 1957.
- Davis, T.N. and Sugiura, M., "Auroral electrojet activity index AE and its universal time variations," *Journal of Geophysical Research*, Vol. 71, pp. 785–801, 1966.
- Gonzalez, W.D. et al., "What is a geomagnetic storm?," *Journal of Geophysical Research*, Vol. 99, pp. 5771–5792, 1994.
- Iyemori, T., "Storm-time magnetospheric currents inferred from mid-latitude geomagnetic field variations," *Journal of Geomagnetism and Geoelectricity*, Vol. 42, pp. 1249–1265, 1990.
- Mayaud, P.N., "The aa indices: A 100-year series characterizing the magnetic activity," *Journal of Geophysical Research*, Vol. 77, pp. 6870–6874, 1972.
- Troshichev, O.A. and Andrezen, V.G., "The relationship between interplanetary quantities and magnetic activity in the southern polar cap," *Planetary and Space Science*, Vol. 33, pp. 415–419, 1985.
