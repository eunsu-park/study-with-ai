# Pre-Reading Briefing: Chapman & Bartels (1940) / 사전 읽기 브리핑

**Paper**: "Geomagnetism" (Monograph, 2 volumes)
**Authors**: Sydney Chapman, Julius Bartels
**Year**: 1940
**Publisher**: Oxford University Press (Clarendon Press)
**Pages**: ~1049 (Vol. I: Geomagnetic and Related Phenomena; Vol. II: Analysis and Physical Interpretation)

---

## 1. 핵심 기여 / Core Contribution

Chapman과 Bartels의 "Geomagnetism"은 **지자기학 분야의 바이블**로, 20세기 후반까지 이 분야의 결정적 참고문헌으로 군림한 모노그래프입니다. 이 저작은 세 가지 핵심 기여를 합니다: (1) **지자기 지수(geomagnetic indices)**의 체계화 — 특히 Bartels가 개발한 **Kp 지수**와 이후 **Dst 지수**의 수학적 기초를 확립하여, 지자기 활동을 전 세계적으로 표준화된 방식으로 정량화할 수 있게 했습니다. (2) **구면 조화 분석(spherical harmonic analysis)**을 지구 자기장에 체계적으로 적용하여, 지구 내부 기원의 주 자기장(main field)과 외부 기원의 변동 자기장(variation field)을 수학적으로 분리하는 틀을 제공했습니다. (3) 지자기 변동 현상 — 일변화(daily variation), 자기 폭풍(magnetic storms), 태양흑점 주기와의 상관관계 — 을 관측 데이터에 기반하여 **통계적으로 분석하는 방법론**을 확립했습니다.

Chapman and Bartels' "Geomagnetism" is the **bible of geomagnetism**, the definitive reference for the field until the late 20th century. It makes three key contributions: (1) **Systematization of geomagnetic indices** — especially the **Kp index** developed by Bartels, and the mathematical foundation for the **Dst index**, enabling standardized global quantification of geomagnetic activity. (2) **Systematic application of spherical harmonic analysis** to Earth's magnetic field, providing the mathematical framework for separating the internally-originated main field from externally-originated variation fields. (3) Establishing **statistical methodology** for analyzing geomagnetic variations — diurnal variation, magnetic storms, and solar cycle correlations — based on observational data.

---

## 2. 역사적 맥락 / Historical Context

```
1600  Gilbert — "De Magnete" (지구 자체가 거대한 자석 / Earth itself is a giant magnet)
  │
1839  Gauss — 구면 조화 분석으로 지구 자기장의 내/외부 기원 분리
  │         Separated internal/external sources via spherical harmonics
  │
1908  Birkeland — 오로라와 전류 체계 탐험 (Paper #1)
  │
1931  Chapman & Ferraro — 자기 폭풍의 정량적 이론 (Paper #2)
  │
1940  ★ Chapman & Bartels — "Geomagnetism" ← 지금 읽을 모노그래프
  │     지자기학의 모든 지식을 집대성, Kp 지수 체계화
  │     Encyclopedic synthesis of geomagnetism, Kp index systematized
  │
1958  Parker — 태양풍 예측 (Paper #4)
  │
1958  Van Allen — 방사선대 발견 (Paper #5)
  │
1961  Dungey — 자기 재결합 (Paper #6)
  │
1975  Burton et al. — Dst와 태양풍의 경험적 관계 (Paper #11)
```

- 이 모노그래프는 **Gauss (1839)**가 시작한 구면 조화 분석 전통을 계승하고 완성합니다.
  This monograph inherits and completes the spherical harmonic analysis tradition started by **Gauss (1839)**.
- Chapman은 Paper #2에서 자기 폭풍의 물리 이론을 세운 인물이며, 여기서는 **관측과 수학적 분석**에 집중합니다.
  Chapman built the physical theory of magnetic storms in Paper #2; here he focuses on **observations and mathematical analysis**.
- Bartels는 **통계적 지자기학(statistical geomagnetism)**의 아버지로, Kp 지수를 비롯한 활동도 지표를 개발했습니다.
  Bartels is the father of **statistical geomagnetism**, developing activity indices including the Kp index.
- 이 책은 우주 시대(Space Age) **이전**에 쓰여졌으므로, 자기권(magnetosphere)이나 태양풍(solar wind) 개념이 등장하지 않습니다. 대신 지상 관측만으로 놀라울 정도로 정교한 분석 체계를 구축합니다.
  Written **before** the Space Age, this book contains no magnetosphere or solar wind concepts. Instead, it builds remarkably sophisticated analysis systems from ground observations alone.

---

## 3. 필요한 배경 지식 / Prerequisites

### 3.1 이전 논문에서 배운 개념 / From Previous Papers

| 논문 / Paper | 핵심 개념 / Key Concept |
|---|---|
| #1 Birkeland (1908) | 오로라와 전류 체계, terrella 실험, 극지 관측 / Aurora and current systems, terrella experiments, polar observations |
| #2 Chapman & Ferraro (1931) | 자기 폭풍의 물리적 메커니즘, 유도 전류, 자기 공동(cavity) / Physical mechanism of magnetic storms, induced currents, magnetic cavity |

### 3.2 새로 필요한 수학 / New Math Needed

1. **구면 조화 함수 / Spherical Harmonics** ($Y_n^m(\theta, \phi)$)
   - 구면 위의 함수를 분해하는 직교 기저 함수
   - Orthogonal basis functions for decomposing functions on a sphere
   - Fourier 분석의 구면 버전이라고 이해하면 됨 (spherical analog of Fourier analysis)

2. **르장드르 다항식 / Legendre Polynomials** ($P_n(\cos\theta)$)
   - 구면 조화 함수의 위도 방향 성분
   - Latitudinal component of spherical harmonics
   - $P_0 = 1$, $P_1 = \cos\theta$, $P_2 = \frac{1}{2}(3\cos^2\theta - 1)$, ...

3. **시계열 분석 기초 / Time-Series Analysis Basics**
   - 조화 분석 (harmonic analysis): 주기적 신호를 sin/cos 성분으로 분해
   - 자기상관 (autocorrelation), 추세 분석 (trend analysis)

4. **벡터 미적분 / Vector Calculus**
   - 자기 포텐셜 (magnetic potential), 그래디언트, 발산, 회전
   - Magnetic potential, gradient, divergence, curl

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 직관적 설명 / Intuitive Explanation |
|---|---|
| **Main field (주 자기장)** | 지구 내부(외핵의 대류)에서 기원하는 강하고 느리게 변하는 자기장. 대략적으로 쌍극자(dipole)로 근사됨. The strong, slowly varying field originating from Earth's interior (outer core convection). Approximately a dipole. |
| **Secular variation (영년 변화)** | 주 자기장이 수십~수백 년에 걸쳐 서서히 변하는 것. 자기 극의 이동, 쌍극자 모멘트의 변화 포함. Slow changes in the main field over decades to centuries, including magnetic pole drift and dipole moment changes. |
| **Diurnal variation (일변화, $S_q$)** | 하루 주기로 반복되는 자기장 변동. 태양 UV에 의해 이온화된 전리층의 전류 체계가 원인. Daily-repeating field variations caused by ionospheric current systems driven by solar UV ionization. |
| **Magnetic storm (자기 폭풍)** | 수시간~수일에 걸친 강한 자기장 교란. Initial phase (증가) → main phase (급격한 감소) → recovery phase (회복)의 3단계 구조. Strong field disturbance over hours to days with 3 phases. |
| **Kp index (Kp 지수)** | Bartels가 개발한 **3시간 간격**의 준대수적(quasi-logarithmic) 지자기 활동 지수 (0 ~ 9). 전 세계 관측소의 자기장 교란 진폭을 평균하여 산출. A 3-hour quasi-logarithmic geomagnetic activity index (0–9) developed by Bartels, averaging disturbance amplitudes from worldwide stations. |
| **Dst index (Dst 지수)** | 적도 부근 관측소에서 측정한 자기장의 수평 성분 교란. 환전류(ring current) 강도의 척도. Equatorial disturbance in the horizontal component; a measure of ring current intensity. |
| **Spherical harmonic analysis (구면 조화 분석)** | 지구 표면의 자기장을 수학적으로 분해하여 **내부 기원 (n, internal)**과 **외부 기원 (e, external)** 성분을 분리하는 기법. Gauss가 1839년에 도입. Mathematical decomposition separating **internal** and **external** field sources, introduced by Gauss in 1839. |
| **Geomagnetic coordinates (지자기 좌표)** | 지구의 자기 쌍극자 축을 기준으로 한 좌표계. 지리적 좌표와 다름. Coordinate system based on Earth's magnetic dipole axis, different from geographic coordinates. |
| **Quiet day / Disturbed day ($S_q$ / $S_D$)** | 자기적으로 조용한 날($S_q$: quiet-day solar variation)과 교란된 날($S_D$: disturbance variation)을 구분하여 각각의 전류 체계를 분석. Separating magnetically quiet and disturbed days to analyze their respective current systems. |
| **Isomagnetic charts (등자기선도)** | 자기장의 특정 요소(편각, 복각, 수평성분 등)가 같은 값을 갖는 지점을 연결한 지도. Maps connecting points with equal values of a magnetic element (declination, inclination, horizontal intensity, etc.). |

---

## 5. 수식 미리보기 / Equations Preview

### 5.1 지구 자기장의 구면 조화 전개 / Spherical Harmonic Expansion of Earth's Field

지구 자기장의 스칼라 포텐셜 $V$를 구면 조화 함수로 전개합니다:

The scalar potential $V$ of Earth's magnetic field is expanded in spherical harmonics:

$$V(r, \theta, \phi) = a \sum_{n=1}^{\infty} \left[\left(\frac{a}{r}\right)^{n+1} S_n^i + \left(\frac{r}{a}\right)^n S_n^e\right]$$

여기서 / where:
- $a$ = 지구 반지름 / Earth's radius
- $r$ = 중심으로부터의 거리 / distance from center
- $\theta$ = 여위도 (colatitude)
- $\phi$ = 경도 (longitude)
- $S_n^i$ = **내부 기원** 성분 (internal origin) — $(a/r)^{n+1}$로 감소하므로 $r > a$에서 유효
- $S_n^e$ = **외부 기원** 성분 (external origin) — $(r/a)^n$으로 증가하므로 $r < a$에서 유효

**핵심 통찰**: $r$에 대한 의존성이 다르므로, 지상 관측만으로도 자기장의 내부/외부 기원을 수학적으로 분리할 수 있습니다!

**Key insight**: The different $r$-dependence allows mathematical separation of internal vs. external field sources from ground observations alone!

### 5.2 내부 기원 항의 전개 / Internal Source Terms

$$S_n^i(\theta, \phi) = \sum_{m=0}^{n} \left(g_n^m \cos m\phi + h_n^m \sin m\phi\right) P_n^m(\cos\theta)$$

- $g_n^m$, $h_n^m$ = **Gauss 계수 (Gauss coefficients)** — 지구 자기장의 "지문"
- $P_n^m(\cos\theta)$ = 배속 르장드르 함수 (associated Legendre functions)
- $n = 1$ 항이 **쌍극자(dipole)**, $n = 2$가 **사중극자(quadrupole)**, ...
- 쌍극자 항이 전체 자기장 에너지의 **~90%**를 차지

The $n=1$ term is the **dipole** (~90% of total field energy), $n=2$ is the **quadrupole**, etc.

### 5.3 쌍극자 근사 / Dipole Approximation

지자기 쌍극자의 자기장 성분:

The magnetic field components of the geomagnetic dipole:

$$B_r = -\frac{\partial V}{\partial r} = -2 \frac{M}{r^3} \cos\theta$$

$$B_\theta = -\frac{1}{r}\frac{\partial V}{\partial \theta} = -\frac{M}{r^3} \sin\theta$$

여기서 $M$은 지구의 자기 쌍극자 모멘트 ($\approx 8.0 \times 10^{22}$ A·m²)

where $M$ is Earth's magnetic dipole moment ($\approx 8.0 \times 10^{22}$ A·m²)

### 5.4 Kp 지수의 계산 / Kp Index Calculation

Bartels의 Kp 지수 산출 과정:

1. 각 관측소에서 3시간 간격으로 자기장 교란 범위(range) $K$를 측정 (0–9의 준대수 척도)
   Measure the 3-hourly range $K$ of field disturbance at each station (quasi-logarithmic scale 0–9)

2. 각 관측소의 $K$를 표준화(standardize)하여 관측소 간 차이 보정
   Standardize each station's $K$ to correct for inter-station differences

3. 표준화된 값들의 평균 → **$K_p$** (planetary K index)
   Average the standardized values → **$K_p$**

$K$ 값과 실제 교란 진폭 $a_K$ (nT)의 대응:

Correspondence between $K$ values and actual disturbance amplitude $a_K$ (nT):

| $K$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|-----|---|---|---|---|---|---|---|---|---|---|
| $a_K$ (nT) | 0–5 | 5–10 | 10–20 | 20–40 | 40–70 | 70–120 | 120–200 | 200–330 | 330–500 | >500 |

**준대수 척도**: 각 단계마다 진폭 범위가 대략 2배씩 증가 → 매우 넓은 범위의 활동도를 9단계로 압축

**Quasi-logarithmic scale**: amplitude range roughly doubles at each step → compresses a very wide activity range into 9 levels

### 5.5 Dst 지수의 수학적 기초 / Mathematical Foundation of Dst

자기 폭풍의 수평 성분 교란 $D_{st}$:

Horizontal component disturbance during magnetic storms:

$$D_{st}(t) = \overline{\Delta H(\lambda, t)} \cdot \frac{1}{\cos\lambda}$$

여기서 / where:
- $\Delta H$ = 관측된 수평 성분에서 조용한 날 기준값을 뺀 값 / Observed horizontal component minus quiet-day baseline
- $\lambda$ = 지자기 위도 / geomagnetic latitude
- $\cos\lambda$ 보정: 적도에서 멀어질수록 환전류의 자기장 효과가 감소하는 것을 보정 / Correction for decreasing ring current effect with latitude

이 지수는 이후 **Sugiura (1964)**에 의해 현대적 형태로 정식화되지만, 수학적 기초는 Chapman & Bartels에서 시작됩니다.

This index was later formalized in modern form by **Sugiura (1964)**, but the mathematical foundation starts here.

---

## 6. 읽기 전략 / Reading Strategy

이 모노그래프는 626페이지(Vol. I)의 방대한 저작이므로, 전체를 처음부터 끝까지 읽기보다는 핵심 장(章)에 집중하는 것을 권합니다:

This is a massive 626-page monograph (Vol. I), so focusing on key chapters is recommended:

### 최우선 장 / Must-Read Chapters
1. **Chapter 1–3**: 지구 자기장의 기본 요소와 관측 방법 (자기 요소: $D$, $I$, $H$, $Z$, $F$ 정의)
   Basic elements of the geomagnetic field and observation methods
2. **Chapter 9–10**: 구면 조화 분석과 Gauss 계수 — 이 책의 수학적 핵심
   Spherical harmonic analysis and Gauss coefficients — mathematical core
3. **Chapter 11–13**: 일변화($S_q$)와 교란 변동($S_D$) — 전류 체계의 추론
   Diurnal variation and disturbance variation — inferring current systems
4. **Chapter 14–15**: 자기 폭풍의 형태학적 분류와 통계
   Morphological classification and statistics of magnetic storms

### 참고용 장 / Reference Chapters
5. **Chapter 16–18**: 지자기 지수 (K, Kp 등) — Bartels의 핵심 기여
   Geomagnetic indices — Bartels' key contribution
6. **Chapter 20–22**: 태양 활동과 지자기 활동의 상관관계
   Correlations between solar and geomagnetic activity

---

## 7. 이 모노그래프를 읽으면서 주목할 질문들 / Questions to Keep in Mind

1. **쌍극자 근사의 한계**: 지구 자기장이 완벽한 쌍극자가 아니라는 것은 어떤 관측 증거에서 드러나는가?
   What observational evidence reveals that Earth's field is not a perfect dipole?

2. **내부 vs 외부 분리**: 지상 관측만으로 자기장의 기원(내부/외부)을 어떻게 구별할 수 있는가? (Gauss의 방법)
   How can ground observations alone distinguish internal from external field sources?

3. **Kp 지수의 설계 철학**: 왜 선형 척도가 아닌 준대수 척도를 사용했는가? 왜 3시간 간격인가?
   Why quasi-logarithmic rather than linear? Why 3-hour intervals?

4. **자기 폭풍의 패턴**: Chapman & Bartels는 자기 폭풍의 3단계(initial, main, recovery)를 어떻게 통계적으로 정의했는가?
   How did they statistically define the 3 phases of magnetic storms?

5. **태양풍 이전의 세계관**: 태양풍과 자기권 개념 없이, 외부 기원 자기장 변동을 어떻게 설명했는가?
   Without solar wind / magnetosphere concepts, how did they explain external field variations?

---

## 8. 핵심 인물 소개 / Key Figures

### Sydney Chapman (1888–1970)
- 영국의 수학자이자 지구물리학자. Paper #2에서 자기 폭풍 이론을 세운 인물.
- British mathematician and geophysicist. Built magnetic storm theory in Paper #2.
- 대기 상층의 전류 체계(Sq 전류), Chapman 층(이온화 모델) 등 다수의 기여.
- Also contributed Chapman layer (ionization model), Sq current systems, and more.

### Julius Bartels (1899–1964)
- 독일의 지구물리학자. **통계적 지자기학의 선구자**.
- German geophysicist. **Pioneer of statistical geomagnetism**.
- **Kp 지수**를 포함한 여러 지자기 활동 지수를 개발.
- Developed multiple geomagnetic activity indices including the **Kp index**.
- "27-day recurrence" — 지자기 활동이 태양 자전 주기(~27일)에 따라 반복됨을 발견.
- Discovered "27-day recurrence" of geomagnetic activity following the solar rotation period.

---

## 9. 다음 논문과의 연결 / Connection to Next Papers

| 이 모노그래프의 기여 / This monograph's contribution | 이후 논문에서의 발전 / Later development |
|---|---|
| Kp 지수 체계화 / Kp index systematization | #14 Tsyganenko (1989): Kp로 매개변수화된 자기장 모델 / Kp-parameterized field model |
| Dst의 수학적 기초 / Mathematical foundation of Dst | #11 Burton et al. (1975): Dst와 태양풍의 경험적 관계식 / Empirical Dst-solar wind relation |
| 구면 조화 분석 / Spherical harmonic analysis | IGRF (국제 지자기 참조장): Gauss 계수의 현대적 결정 / Modern determination of Gauss coefficients |
| 자기 폭풍 분류 / Magnetic storm classification | #15 Gonzalez et al. (1994): 폭풍/substorm 구별, Dst 임계값 / Storm/substorm distinction, Dst thresholds |
| 27일 재현성 / 27-day recurrence | #4 Parker (1958): 태양풍으로 설명 / Explained by solar wind — spiral structure from solar rotation |
