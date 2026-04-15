---
title: "What Is a Geomagnetic Storm?"
authors: Walter D. Gonzalez, Joselyn A. Joselyn, Yohsuke Kamide, Herbert W. Kroehl, Gordon Rostoker, Bruce T. Tsurutani, Vytenis M. Vasyliunas
year: 1994
journal: "Journal of Geophysical Research, 99(A4), 5771–5792"
topic: Space_Weather
tags: [geomagnetic storm, Dst index, ring current, substorm, IMF Bz, solar wind coupling, storm classification, space weather]
status: completed
date_started: 2026-04-15
date_completed: 2026-04-15
---

# 15. What Is a Geomagnetic Storm? / 지자기 폭풍이란 무엇인가?

---

## 1. Core Contribution / 핵심 기여

이 논문은 "지자기 폭풍이란 무엇인가?"라는 근본적인 질문에 대해 우주기상 분야 최고 전문가 7인의 합의된 답을 제시한 결정적인 리뷰 논문이다. 1990년대 초반까지 지자기 폭풍의 정의는 연구자마다 달랐고, substorm과의 관계도 불분명했다. 이 논문은 (1) Dst 지수에 기반한 정량적 폭풍 강도 분류(weak/moderate/intense/super-intense), (2) 폭풍의 3단계(initial phase, main phase, recovery phase)의 명확한 정의, (3) 태양풍 구동 조건(장시간 남향 IMF $B_z$와 $VB_s$ coupling), (4) 폭풍과 substorm의 근본적 차이를 체계적으로 정리했다. 특히 "폭풍은 substorm의 단순 합이 아니다"라는 결론은 이후 수십 년간의 storm–substorm 논쟁에 방향을 제시했다. 또한 실용적 관점에서 기존 지자기 지수(Dst, AE, Kp)의 한계를 지적하고 7가지 구체적 개선 권고를 제시하여, 현대 우주기상 모니터링 시스템의 발전 방향을 설정했다.

This paper provides the definitive community-consensus answer to the fundamental question "What is a geomagnetic storm?" Written by seven leading experts (Gonzalez, Joselyn, Kamide, Kroehl, Rostoker, Tsurutani, Vasyliunas), it resolves longstanding ambiguities that had plagued storm research through the early 1990s. The paper systematically establishes: (1) quantitative Dst-based storm intensity classification (weak: −30 to −50 nT; moderate: −50 to −100 nT; intense: < −100 nT; super-intense: < −250 nT); (2) clear definitions of the three canonical storm phases (initial, main, recovery); (3) solar wind driving conditions (sustained southward IMF $B_z$ and $VB_s$ coupling as the dominant control); and (4) the fundamental distinction between storms and substorms. The central conclusion that "storms are NOT merely superpositions of substorms" set the direction for decades of subsequent storm–substorm debate. The paper also critically evaluates existing geomagnetic indices and provides seven concrete recommendations for improving storm monitoring, thereby shaping the development of modern space weather observing systems.

---

## 2. Reading Notes / 읽기 노트

### Section 1: Introduction / 소개 (pp. 5771–5772)

저자들은 지자기 폭풍 연구의 중요성을 두 가지 측면에서 소개한다. 첫째는 학술적 측면으로, 지자기 폭풍은 지구물리학의 핵심 연구 대상이다. 둘째는 실용적 측면으로, 지자기 폭풍이 전력망, 통신, 위성 등에 피해를 줄 수 있다. 30년 이상 연구자들이 "unit of fundamental energy injection"을 이해하려 해왔으나, 정의에 대한 합의가 부족했다.

The authors motivate the paper from two angles: academic (storms as a central problem in geophysics) and practical (storms causing damage to power grids, communications, and satellites). Despite 30+ years of research, the community still lacked consensus on what constitutes a storm. The paper originated from a National Institute for Space Research (INPE) meeting at São José dos Campos, Brazil, November 5–8, 1991, and subsequent discussions at the International Geophysical Year.

논문의 구조: Section 2는 폭풍의 역사적 정의와 자기권 매개변수, Section 3은 IMF $B_z$의 역할, Section 4는 storm/substorm 관계, Section 5는 추가 메커니즘, Section 6은 요약, Section 7은 관측 개선 권고.

The paper structure: Section 2 covers historical definitions and magnetospheric parameters; Section 3 addresses the IMF $B_z$ role; Section 4 discusses storm/substorm relationships; Section 5 covers additional mechanisms; Section 6 provides summary concepts; Section 7 gives recommendations for improved observations.

### Section 2: Geomagnetic Storm / 지자기 폭풍 (pp. 5772–5776)

#### 2.1 Historical Development and Critique of Existing Definitions / 기존 정의의 역사적 발전과 비판

지자기 폭풍은 1800년대 중반에 처음 "storm"으로 명명되었으며, 핵심 특징은 수평 자기장 성분(H)의 감소와 후속 회복이다. Chapman and Bartels [1940]는 이를 포획 자기권 입자(trapped magnetospheric particles)의 증가로 설명했다. Sugiura and Chapman [1960]은 적도 관측소 데이터를 기반으로 약(weak), 중(moderate), 강(great)의 3단계 분류를 제안했지만, 정량적 임계값은 불명확했다.

Geomagnetic storms were first named in the mid-1800s, characterized by an unmistakable decrease of horizontal magnetic intensity (H) and subsequent recovery. Chapman and Bartels [1940] attributed this to enhanced trapped magnetospheric particle population. Sugiura and Chapman [1960] proposed a three-tier classification (weak, moderate, great) using equatorial station data, but without firm quantitative thresholds.

Dst 지수는 Sugiura [1964]가 처음 도입했으며, 적도 위도 관측소 4개의 H 성분 변동을 시간별 평균한 값이다. 하지만 Dst에는 여러 전류 시스템(magnetopause current, field-aligned current, tail current 등)이 기여하므로, 순수한 ring current만을 반영하지는 않는다. Kp, AE 등 다른 지수도 사용되지만, 각각 한계가 있다.

The Dst index was introduced by Sugiura [1964] as an hourly average of H-component variations at four equatorial stations. However, Dst receives contributions from multiple current systems (magnetopause, field-aligned, tail currents), not purely the ring current. Other indices (Kp, AE) are also used but each has limitations. The paper notes that Kp index variations are difficult to interpret physically because they can be caused by any geophysical current system.

#### 2.2 Dst Index and Magnetospheric Parameters / Dst 지수와 자기권 매개변수 (pp. 5773–5774)

폭풍의 핵심 물리적 특성은 ring current의 강화이다. Ring current은 양성자(H⁺)와 산소 이온(O⁺)이 10–300 keV 에너지 범위에서 적도면 약 2–7 $R_E$에서 서쪽으로 흐르는 전류이다. Dessler-Parker-Sckopke (DPS) 관계에 의해:

The principal defining property of a magnetic storm is the creation of an enhanced ring current, formed by ions (most notably protons and oxygen ions) and electrons in the 10–300 keV energy range, located usually between 2 to 7 $R_E$. The Dessler-Parker-Sckopke (DPS) relationship gives:

$$Dst^*(t)/B_0 \approx 2E(t)/3E_m$$

여기서 $Dst^*$은 ring current에 의한 자기장 변화, $B_0$는 적도면 자기장, $E(t)$는 ring current 입자의 총 에너지, $E_m$은 쌍극자 자기장의 총 에너지이다. $Q(t) = 2.5 \times 10^{21} E(t)$ (Gaussian 단위).

Where $Dst^*$ is the field decrease due to the ring current, $B_0$ is the average equatorial surface field, $E(t)$ is the total energy of the ring current particles, and $E_m$ ($= 8 \times 10^{24}$ ergs) is the total magnetic energy of the geomagnetic field outside the Earth.

Burton equation (Burton et al., 1975)으로 표현하면:

The Burton equation gives the energy balance:

$$\frac{dDst^*}{dt} = Q(t) - \frac{Dst^*}{\tau}$$

여기서 $Q(t)$는 에너지 주입률, $\tau$는 ring current 소산 시간상수이다. 에너지 주입이 없을 때($Q = 0$), recovery phase의 해는:

Where $Q(t)$ is the energy injection rate and $\tau$ is the decay time. When there is no energy input ($Q = 0$), the recovery phase solution is:

$$E(t) = E_0 e^{-t/\tau}$$

$\tau$의 값은 논쟁의 대상이다. Burton et al. [1975]는 약 7.7시간을 제안했지만, 최근 연구는 $\tau$가 Dst 크기에 따라 변할 수 있으며, recovery phase에서 5–10시간, main phase의 강한 폭풍에서는 0.5–1시간까지 짧아질 수 있다고 제시한다.

The value of $\tau$ remains debated. Burton et al. [1975] suggested ~7.7 hours, but recent work shows $\tau$ may vary with Dst magnitude — values of 5 to 10 hours are common during recovery, but decay times as short as ~1 hour may be needed at the peak of the main phase of intense storms.

#### 2.3 Classification of Storms by Intensity: The Question of Threshold / 강도별 폭풍 분류: 임계값 문제 (pp. 5774)

저자들은 정량적 분류 기준을 제시한다:

The authors establish quantitative classification criteria:

| 분류 / Category | Dst 기준 / Dst Criterion | 통계 (1975–1986) / Statistics |
|---|---|---|
| Weak (약) | −30 to −50 nT | 전체 폭풍의 약 25% / ~25% of all storms |
| Moderate (중) | −50 to −100 nT | 전체 폭풍의 약 63% / ~63% of all storms |
| Intense (강) | < −100 nT | 전체 폭풍의 약 11% / ~11% of all storms |
| Super-intense (초강) | < −250 nT (−500 이하) | 전체의 약 1.5% 미만 / <1.5% of all storms |

1975–1986 기간 동안, Dst < −100 nT인 intense storm은 전체의 약 11–12%를 차지했다. Dst < −30 nT를 하한으로 설정한 이유는 그 이하는 물리적 의미가 약하고, 실용적 관점에서도 "storm"이라 칭할 근거가 부족하기 때문이다.

During 1975–1986, approximately 25% of all storms were more negative than −30 nT, approximately 88% were more negative than −50 nT, and approximately 1% were more negative than −100 nT. The −30 nT lower threshold was chosen because below this level the term storm has no physical basis. Less than 1% of the Kp values were 7 or larger, and less than 10% were 5 or larger.

### Section 3: Role of the Interplanetary Medium / 행성간 매질의 역할 (pp. 5774–5780)

#### 3.1 Origins of $B_s$ / 남향 자기장의 기원 (pp. 5774–5776)

행성간 자기장(IMF)의 남향 성분 $B_s$가 폭풍의 핵심 구동자이다. $B_s$의 기원은 크게 두 가지:

The southward component of the IMF ($B_s$) is the key storm driver. Its origins are twofold:

**1. Sheath fields (피복 자기장)**: CME 앞의 충격파 뒤에 형성되는 압축된 영역. 태양풍의 ambient IMF가 shock에 의해 압축되어 남향 성분이 강화된다. Shocked heliospheric current sheet나 draped magnetic field도 남향 $B_s$를 생성할 수 있다.

**1. Sheath fields**: The compressed region behind the shock ahead of a CME. Ambient IMF is compressed by the shock, enhancing southward components. Shocked heliospheric current sheets and draped magnetic fields can also produce southward $B_s$.

**2. Driver gas fields (구동 가스 자기장)**: CME의 본체, 특히 magnetic cloud 구조. Magnetic cloud는 강하고 회전하는 자기장을 가지며, N-S 방향 변동이 크다. 약 10%의 driver gas만이 큰 N-S 방향 변동을 보인다. Klein and Burlaga [1982]의 magnetic cloud 모델과 Marubashi [1986]의 force-free flux rope 모델이 대표적이다.

**2. Driver gas fields**: The CME body itself, especially magnetic cloud structures. Magnetic clouds have strong, rotating magnetic fields with large N-S variations. Only about 10% of driver gases have large N-S directional variations. Klein and Burlaga's [1982] magnetic cloud model and Marubashi's [1986] force-free flux rope configuration are representative.

핵심 발견: ISEE 3 데이터(1978년 8월–1979년 12월) 분석에서 intense storm (Dst < −100 nT)을 유발하는 10개 사건 중 대부분이 sheath fields 또는 magnetic cloud와 관련되었다. HILDCAA(High-Intensity Long-Duration Continuous AE Activity) 사건도 substorm과 유사한 특성을 보이지만, 폭풍보다는 약한 강도이다.

Key finding: Analysis of ISEE 3 data (August 1978–December 1979) showed that most of the 10 intense storms (Dst < −100 nT) were associated with either sheath fields or magnetic clouds. HILDCAA events show substorm-like characteristics but with shorter durations and weaker intensity.

#### 3.2 Solar Wind–Magnetosphere Interaction During Magnetic Storms / 폭풍 동안의 태양풍–자기권 상호작용 (pp. 5776–5779)

자기 reconnection이 태양풍 에너지를 자기권으로 전달하는 주요 메커니즘이다. 에너지 전달 효율은 약 1% 수준이지만, 이것만으로도 대규모 폭풍을 유발하기에 충분하다.

Magnetic reconnection is the primary energy transfer mechanism from solar wind to magnetosphere. The energy transfer efficiency is of the order of 10% during intense magnetic storms, though typically only ~1% during quiet times.

**Coupling functions (결합 함수)**: 다양한 에너지 결합 함수가 제안되었으며, Table 2에 정리되어 있다:

**Coupling functions**: Various energy coupling functions have been proposed, summarized in Table 2:

| Electric Field Related | Power Related | Simple |
|---|---|---|
| $vB_z$ | $\epsilon = vL_0B^2\sin^4(\theta/2)$ | $B_z$ |
| $vB_T$ | $(\rho v^2)^{1/2}vB_z$ | $B_zv^2, Bv^2$ |
| $vB_T\sin(\theta/2)$ | $(\rho v^2)^{-1/2}vB_T^2\sin^4(\theta/2)$ | $B_z^2v, B^2v$ |
| $vB_T\sin^2(\theta/2)$ | $(\rho v^2)^{1/4}vB_T\sin^4(\theta/2)$ | |
| $vB_T\sin^4(\theta/2)$ | | |

이 중 지배적인 매개변수는 $B_s$와 $V$(태양풍 속도)이다. 가장 널리 사용되는 것은:
- $VB_s$ (electric field coupling): Burton et al. [1975]
- $\epsilon = vL_0^2 B^2 \sin^4(\theta/2)$ (Perreault-Akasofu epsilon): 총 에너지 전달률

Among these, the dominant parameters are $B_s$ and $V$ (solar wind speed). The most widely used are $VB_s$ (electric field coupling, Burton et al. [1975]) and the Perreault-Akasofu $\epsilon$ function for total energy transfer rate.

Intense storm의 조건: peak Dst < −100 nT을 유발하려면 $B_z > 10$ nT(남향)이 3시간 이상 지속되어야 한다. 이러한 조건은 주로 large $B_z$ excursion을 가진 ICME에 의해 충족된다. 고속 태양풍 stream만으로는 moderate storm 수준까지만 도달한다.

Intense storm condition: To produce peak Dst < −100 nT, southward $B_z > 10$ nT must be sustained for more than 3 hours. Such conditions are primarily met by ICMEs with large $B_z$ excursions. High-speed streams alone typically produce only moderate storms.

#### 3.3 Seasonal and Solar-Cycle Distribution of Storms / 폭풍의 계절·태양주기 분포 (pp. 5779–5780)

지자기 활동은 춘·추분(equinoxes)에 극대를 보인다(Russell-McPherron effect). Intense storm은 태양주기 동안 이중 피크(dual-peak) 분포를 보인다: 하나는 태양극대기 근처, 다른 하나는 극대기 후 2–3년에 나타난다.

Geomagnetic activity shows maxima at the equinoxes (Russell-McPherron effect). Intense storms show a dual-peak distribution during the solar cycle: one near solar maximum and another 2–3 years after maximum.

Figure 7은 태양주기 21(solar cycle 21)의 intense storm 이중 피크 분포를 보여주며, $B_s$ 사건의 태양주기 분포와 유사한 패턴을 보인다. $B_s > 10$ nT이고 지속시간 > 3시간인 사건의 분포가 intense storm 분포와 일치한다는 것은 남향 IMF가 폭풍의 핵심 구동자임을 재확인한다.

Figure 7 shows the dual-peak distribution of intense storms for solar cycle 21, with a corresponding distribution of $B_s$ events. The match between events with $B_s > 10$ nT and duration > 3 hours and the intense storm distribution confirms southward IMF as the primary storm driver.

### Section 4: Relationships of Storms and Substorms / 폭풍과 서브스톰의 관계 (pp. 5780–5782)

#### 4.1 A Question of Definition / 정의의 문제

이 섹션은 논문에서 가장 논쟁적이고 중요한 부분이다. 핵심 질문: "지자기 폭풍은 intense substorm의 집합인가, 아니면 근본적으로 다른 현상인가?"

This is the most debated and important section of the paper. The central question: "Is a geomagnetic storm a collection of intense substorms, or a fundamentally different phenomenon?"

Chapman [1962]의 초기 관점: 폭풍은 substorm의 빈번한 발생에 의해 발전한다. 그러나 저자들은 이에 대해 비판적 재검토를 수행한다.

Chapman's [1962] early view: storms develop as a result of frequent occurrence of substorms. The authors critically re-examine this view.

**Substorm 정의 (Akasofu [1968])**: 자기권 substorm은 자기 폭풍과 독립적으로 발생할 수 있다. Substorm은 야간 측에서 태양풍–자기권 상호작용으로부터 파생된 에너지가 오로라 전리층과 자기권에 침적되는 일시적 과정이다.

**Substorm definition (Akasofu [1968])**: A magnetospheric substorm can occur independently of a magnetic storm. It is a transient process initiated on the night side of the Earth in which a significant amount of energy derived from solar wind–magnetosphere interaction is deposited in the auroral ionosphere and magnetosphere.

**Storm ≠ Substorm의 증거들**:

1. 많은 isolated substorm은 유의미한 Dst 하강을 일으키지 않는다. 단, 가장 intense한 substorm은 main phase 기간에 발생한다.
2. 폭풍 없는 substorm이 관측된다 (no magnetic storm observed despite substorm activity).
3. 자기 활동이 높은 시기(AE > 1000)라도 항상 폭풍을 동반하지는 않는다.

**Evidence that Storm ≠ Substorm**:

1. There is an apparent difference between storm associated and substorms occurring at times of no significant Dst enhancement, except that the most intense substorms are usually found within the main phase of storms.
2. No magnetic storms have been observed in the absence of substorms — but storms without substorms have not been observed either (this cuts both ways).
3. When geomagnetic activity is exceptionally high (AE > 1000 nT), it always involves substorms, but this does not necessarily mean the substorms cause the storm.

#### 4.2 Storm Conditions / 폭풍 조건

Burton equation을 사용하여, ring current 진화가 substorm activity (AE로 측정)와 비례한다고 가정하면 Dst 변화를 재현할 수 있다. 그러나 이것이 storm의 main phase가 substorm의 선형 중첩이라는 것을 의미하지는 않는다.

Using the Burton equation, if one assumes that Q is simply proportional to substorm activity (AE), the storm main phase can be described as a linear superposition of substorm disturbances. However, the logic behind this does not demand that the auroral electrojets directly generate the ring current. Rather, it suggests there is an energy reservoir in the solar wind/magnetosphere/ionosphere system.

Figure 8 (Kamide and Fukushima [1971])는 이 가정하에 예측된 Dst와 관측된 Dst를 비교한다. 합리적 일치를 보이지만, 효율 $\alpha_i$가 시간에 따라 변하며 main phase의 후반에 최대가 된다는 점이 중요하다:

Figure 8 compares predicted and observed Dst under this assumption. Reasonable agreement is found, but the efficiency $\alpha_i$ varies with time, being largest during the early main phase:

$$\text{storm} = \sum \alpha_i (\text{substorm})_i$$

여기서 $\alpha_i$는 $i$번째 substorm의 효율이다. 매개변수 $a$ ($0 < a < c < 1$)이 ring current 성장과 substorm의 상관관계를 나타내며, intense storm에서는 ring current와 substorm 사이의 상관이 더 강하지만, moderate storm에서는 decoupling이 나타난다.

Where $\alpha_i$ is the efficiency of the $i$-th substorm. A parameter $a$ ($0 < a < c < 1$) represents the efficiency of the ring current growth relative to the corresponding substorm. For intense storms, the correlation between ring current and substorm activity is stronger, but for moderate storms, decoupling appears.

**AE–Dst 관계**: Saba et al. [1994]의 분석에서, 22개 intense storm (1974, 1978, 1979)에 대해 AE의 10시간 적분과 peak Dst 사이에 상관계수 0.81을 얻었다. 그러나 moderate storm에서는 상관이 현저히 낮아져, storm과 substorm의 decoupling을 시사한다.

**AE–Dst relationship**: Saba et al. [1994] found a correlation coefficient of 0.81 between the 10-hour integral of AE and peak Dst for 22 intense storms. However, for moderate storms, the correlation was considerably lower (0.68), suggesting storm/substorm decoupling. Figure 10 shows AE–Dst relationships at several levels of storm intensity, with linear behavior for moderate storms but saturation at higher AE values for intense storms.

### Section 5: Discussion / 논의 (pp. 5782–5786)

#### 5.1 Origin of Dst Through Injection of Energetic Particles / Dst의 기원: 고에너지 입자 주입 (pp. 5782–5784)

Ring current 강화의 물리적 메커니즘에 대한 논의이다. 핵심 질문: 주입된 고에너지 입자가 어떻게 ring current에 기여하는가?

Discussion of the physical mechanisms behind ring current enhancement. Key question: how do injected energetic particles contribute to the ring current?

두 가지 시나리오:

Two scenarios:

**1. 직접 주입 (direct injection)**: 태양풍의 대류 전기장(convection electric field)이 충분히 강하면 ($E > 5$ mV/m), 입자가 직접 inner magnetosphere까지 도달하여 symmetric ring current를 형성한다. 이는 intense storm에서 주로 관찰된다.

**1. Direct injection**: When the convection electric field is sufficiently strong ($E > 5$ mV/m), particles can reach the inner magnetosphere directly, forming a symmetric ring current. This is primarily observed during intense storms.

**2. Substorm 관련 주입**: Substorm expansion phase에서 방출된 입자가 ring current으로 이동한다. Takahashi et al. [1990]은 단색 입자 trajectory tracing으로 closed ring current 형성을 시뮬레이션했다.

**2. Substorm-related injection**: Particles released during substorm expansion phase drift into the ring current. Takahashi et al. [1990] simulated closed ring current formation through monochromatic particle trajectory tracing.

Ring current의 비대칭성: Main phase 동안 ring current은 비대칭(asymmetric)이며, 야간 측에 집중된다. 이 비대칭 ring current은 strong isotropic and field-aligned current으로 유지된다.

Ring current asymmetry: During the main phase, the ring current is asymmetric, concentrated on the nightside. This asymmetric ring current is maintained by strong isotropic and field-aligned currents.

#### 5.2 Ring Current Loss Processes / 환전류 손실 과정 (pp. 5784)

Ring current 소산의 주요 메커니즘:

Major ring current loss mechanisms:

1. **전하 교환 (charge exchange)**: ring current 이온이 geocoronal 중성 수소와 충돌하여 에너지를 잃는 과정. 가장 중요한 손실 과정이다.
2. **Coulomb 산란 (Coulomb scattering)**: 열 플라즈마와의 상호작용으로 인한 산란.
3. **전자기 이온 cyclotron 파동 (electromagnetic ion cyclotron waves)**: Cornwall [1977]이 제안. Storm main phase에서 ring current이 지구에 가까워질 때 중요해질 수 있다.

1. **Charge exchange**: Ring current ions collide with geocoronal neutral hydrogen, losing energy. The most important loss process.
2. **Coulomb scattering**: Scattering by interaction with thermal plasma.
3. **Electromagnetic ion cyclotron waves**: Proposed by Cornwall [1977]. May become important during storm main phase when ring current moves close to Earth.

#### 5.3 Relationship of Dst to Other Geomagnetic Indices / Dst와 다른 지자기 지수의 관계 (pp. 5785)

Dst와 다른 지수 사이의 관계에 대한 기존 연구가 제한적이다. Campbell [1973]은 Dst/AE 관계에서 계절 변동을 발견했다. Saba et al. [1994]의 pair correlation 연구에서, Dst와 AE, AL 사이의 상관은 태양 주기, 계절, 폭풍 강도에 따라 크게 달라진다.

Studies of Dst relationships to other indices are limited. Campbell [1973] found seasonal variation in Dst/AE relationships. Saba et al.'s [1994] pair correlation study showed that Dst–AE and Dst–AL correlations vary significantly with solar cycle, season, and storm intensity.

### Section 6: Summary Concepts / 요약 (pp. 5786–5787)

저자들의 핵심 합의:

The authors' key consensus:

1. **지자기 폭풍의 정의**: 중·저위도 지자기 변동이 ring current의 강화에 의해 발생하는 현상. 정량적 기준은 Dst 임계값으로 설정.

2. **Storm과 substorm은 모두 태양풍 에너지를 자기권–전리권 시스템에서 물리적 과정으로 재분배하는 현상**이지만, 그 proportions은 다르다.

3. **Substorm은 폭풍의 필요조건이 아닐 수 있다**: Substorm 없이도 convection electric field만으로 ring current이 형성될 수 있다 (intense storm의 경우).

4. **정의**: 태양-지구 상호작용의 강화된 기간에서 ring current이 성장하여 Dst가 특정 임계값(이 논문에서는 −30 nT)을 넘는 것을 폭풍으로 정의. 이 임계값은 운용적 필요에 따라 더 낮게 설정할 수 있다.

The authors' key consensus points:

1. **Storm definition**: A geomagnetic storm is a phenomenon of middle- and low-latitude geomagnetic variations, identified by the intensification of the ring current and quantified by the Dst index.

2. **Both storms and substorms** originate in physical processes by which energy from the solar wind is redistributed in the magnetosphere–ionosphere system. The proportions of the substorms are in some way related to the properties of the ring current.

3. **Substorms may not be necessary for storms**: The convection electric field alone can lead to ring current formation without substorms (during intense storms).

4. **Formal definition**: A storm is defined as an interval of enhanced solar-terrestrial interaction featuring ring current growth leading to a Dst exceeding the specified threshold (−30 nT in this paper, though the threshold being −30 nT seems to depend on the behavior of the convection electric field).

Figure 11은 storm–substorm 관계의 도식적 요약을 제공한다. AE와 Dst 지수로 세 가지 활동 수준(substorm, HILDCAA, storm)을 구분하고, 관련된 $B_s$ field의 행동과 기원을 보여준다.

Figure 11 provides a schematic summary of the storm–substorm relationship, distinguishing three activity levels (substorm, HILDCAA, storm) using AE and Dst indices, along with the associated $B_s$ field behavior and origins.

### Section 7: Recommendations / 권고 사항 (pp. 5787–5789)

#### 7.1 Geomagnetic Index Improvements / 지자기 지수 개선

**Dst 지수의 문제점**:
- 관측소가 4개뿐이어서 극관(polar cap)이 적절히 반영되지 않음
- AE 관측소가 적도 쪽으로 약 180° 이격되어야 하지만 실제로는 균등하지 않음
- Kp 지수는 물리적 해석이 어려움

**Problems with Dst index**:
- Only 4 stations, so polar cap not adequately represented
- AE stations should cover ~180° latitude span but actual coverage is uneven
- Kp index is difficult to interpret physically

**7가지 구체적 권고**:

1. Dst에 기여하는 AE 관측소 범위를 적도 남북으로 확장
2. AE 관측소를 Z 성분 포함하여 재배치
3. 극관 경계 평가를 위한 관측소 추가
4. Dst 관측소를 증설하여 적도 전류의 종방향 분포 파악
5. 태양풍 모니터링을 실시간으로 확대 (L1 point)
6. 실시간 Dst 산출 능력 확보
7. 태양풍과 자기권 자기장 데이터를 실시간으로 통합

**7 specific recommendations**:

1. Improve coverage of AE stations contributing to Dst, extending equatorward
2. Reconfigure AE stations to include Z component data
3. Add stations for polar cap boundary evaluation
4. Increase Dst stations to resolve longitudinal distribution of equatorial currents
5. Expand real-time solar wind monitoring (L1 point)
6. Establish real-time Dst computation capability
7. Integrate solar wind and magnetospheric data in real-time

#### 7.2 Solar Wind Monitoring / 태양풍 모니터링

ISTP(International Solar-Terrestrial Physics) 프로그램 하에서 실시간 태양풍 모니터링의 필요성을 강조. 태양풍 플라즈마와 행성간 자기장 매개변수를 지구 상류(upstream)에서 실시간 모니터링하여, 데이터를 국제적으로 공유할 것을 권고한다. 이는 이후 ACE 위성(1997)과 DSCOVR 위성(2015)의 L1 모니터링으로 실현되었다.

The paper emphasizes the need for continuous real-time solar wind monitoring under the ISTP program. It recommends monitoring solar wind plasma and interplanetary magnetic field parameters upstream of Earth in real-time, with data distributed internationally. This was subsequently realized through the ACE satellite (1997) and DSCOVR satellite (2015) at L1.

---

## 3. Key Takeaways / 핵심 시사점

1. **Dst 기반 정량적 분류가 표준이 되었다** — Weak (−30~−50 nT), moderate (−50~−100 nT), intense (< −100 nT), super-intense (< −250 nT)의 4단계 분류는 30년이 지난 현재까지 우주기상 커뮤니티의 표준으로 사용된다. NOAA의 G-scale도 이 프레임워크에서 발전했다.
   The Dst-based four-tier classification became the community standard, still in use 30 years later. NOAA's operational G-scale evolved from this framework.

2. **폭풍의 핵심 물리는 ring current 강화이다** — Dst 지수가 ring current 에너지의 직접적 대리 변수(DPS relation)라는 것이 이 논문의 정의적 프레임워크의 기반이다. Ring current의 형성, 유지, 소산 과정이 폭풍의 모든 단계를 결정한다.
   The core physics of a storm is ring current enhancement. The DPS relation connecting Dst to ring current energy underlies the paper's entire definitional framework.

3. **Storm ≠ Substorm — 이것이 가장 중요한 결론이다** — 폭풍은 substorm의 단순 집합이 아니라 독립적 현상이다. Intense storm에서는 convection electric field에 의한 직접 입자 주입이 주된 ring current 형성 메커니즘일 수 있으며, substorm이 필수 조건이 아닐 수 있다.
   Storms are NOT mere superpositions of substorms. During intense storms, direct particle injection by the convection electric field may be the primary ring current formation mechanism, and substorms may not be a necessary condition.

4. **남향 IMF $B_z$가 폭풍의 핵심 구동자이다** — 10 nT 이상의 남향 $B_z$가 3시간 이상 지속되면 intense storm이 발생한다. 이 조건은 주로 ICME (magnetic cloud 또는 sheath field)에 의해 충족된다.
   Sustained southward IMF $B_z > 10$ nT for > 3 hours drives intense storms. This condition is primarily met by ICMEs (magnetic clouds or sheath fields).

5. **Coupling function의 다양성과 $VB_s$의 우위** — Table 2에 정리된 다양한 coupling function 중 $VB_s$가 가장 단순하고 효과적인 에너지 결합 매개변수이다. 더 복잡한 함수($\epsilon$, $F_A$ 등)도 유용하지만, 핵심은 태양풍 속도와 남향 IMF의 곱이다.
   Among the various coupling functions in Table 2, $VB_s$ is the simplest and most effective. More complex functions ($\epsilon$, $F_A$) are useful, but the essential control is the product of solar wind speed and southward IMF.

6. **Ring current의 소산 시간 $\tau$는 고정값이 아니다** — Burton et al.의 7.7시간은 평균값이지만, 실제로 $\tau$는 Dst 크기, 폭풍 단계, 에너지 주입률에 따라 0.5–10시간까지 변한다. 이는 현대의 비선형 Dst 예측 모델 발전의 근거가 되었다.
   The decay time $\tau$ is not a fixed constant. While Burton et al.'s 7.7 hours is an average, $\tau$ actually varies from 0.5 to 10 hours depending on Dst magnitude, storm phase, and energy injection rate. This motivated development of modern nonlinear Dst prediction models.

7. **ICME vs CIR 구동 폭풍의 차이** — ICME는 intense storm의 주요 원인이고 (Dst < −100 nT), CIR(고속 스트림)은 moderate storm까지만 유발한다. 그러나 CIR은 반복적이므로 장기적 누적 효과가 중요하다. Half of intense storms are caused by sheath fields and half by magnetic clouds.
   ICMEs drive intense storms while CIRs typically produce only moderate storms. However, CIRs are recurrent, so their cumulative long-term effects are significant.

8. **이 논문의 7가지 권고가 현대 우주기상 시스템을 형성했다** — L1 point의 실시간 태양풍 모니터링(ACE, DSCOVR), SYM-H(1분 해상도 Dst), 실시간 Dst 산출 등 현재의 우주기상 인프라 대부분이 이 논문의 권고에서 시작되었다.
   The paper's seven recommendations shaped modern space weather infrastructure: real-time solar wind monitoring at L1 (ACE, DSCOVR), SYM-H (1-min resolution Dst), and real-time Dst computation all trace back to recommendations in this paper.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Dessler-Parker-Sckopke (DPS) Relation / DPS 관계식

Ring current 에너지와 지표 자기장 변화의 직접적 연결:

Direct connection between ring current energy and surface magnetic field change:

$$Dst^*/B_0 \approx \frac{2E(t)}{3E_m}$$

- $Dst^*$: ring current에 의한 자기장 변화 (nT) / Magnetic field change due to ring current
- $B_0$: 적도면 지표 자기장 (~31,000 nT) / Equatorial surface field
- $E(t)$: ring current 입자의 총 운동에너지 / Total kinetic energy of ring current particles
- $E_m = 8 \times 10^{24}$ ergs: 쌍극자 자기장의 총 에너지 / Total dipole field energy

**물리적 의미**: Dst가 −100 nT이면, ring current 에너지는 지구 쌍극자 에너지의 약 0.5%에 해당한다.

**Physical meaning**: When Dst = −100 nT, the ring current energy is approximately 0.5% of the Earth's dipole field energy.

### 4.2 Burton Equation / Burton 방정식

Ring current의 에너지 균형:

Energy balance of the ring current:

$$\frac{dDst^*}{dt} = Q(t) - \frac{Dst^*}{\tau}$$

- $Q(t)$: 에너지 주입률 (nT/h), $VB_s$에 비례 / Energy injection rate, proportional to $VB_s$
- $\tau$: 소산 시간상수 / Decay time constant
  - 평균 ~7.7 h (Burton et al., 1975)
  - 실제 범위: 0.5–10 h (storm phase와 intensity에 따라 변동)

**Pressure-corrected Dst**: 실제 Dst에서 magnetopause current 기여를 제거:

**Pressure-corrected Dst**: Removing magnetopause current contribution from observed Dst:

$$Dst^* = Dst - b\sqrt{p} + c$$

여기서 $p$는 태양풍 동압(dynamic pressure), $b \approx 15.8$ nT/nPa$^{1/2}$, $c \approx 20$ nT.

Where $p$ is solar wind dynamic pressure, $b \approx 15.8$ nT/nPa$^{1/2}$, $c \approx 20$ nT.

**Recovery phase 해** ($Q = 0$):

$$Dst^*(t) = Dst^*(t_0) \cdot e^{-t/\tau}$$

**일반해** (Q가 알려진 경우):

$$Dst^*(t) = e^{-t/\tau}\left[Dst^*(0) + \int_0^t Q(t')e^{t'/\tau}dt'\right]$$

### 4.3 Energy Coupling Functions / 에너지 결합 함수

**Perreault-Akasofu Epsilon Function**:

$$\epsilon = vL_0^2 B^2 \sin^4\left(\frac{\theta}{2}\right)$$

- $v$: 태양풍 속도 (km/s) / Solar wind speed
- $L_0 = 7 R_E$: 유효 자기권 스케일 / Effective magnetosphere scale length
- $B$: IMF 크기 (nT) / IMF magnitude
- $\theta$: IMF clock angle = $\arctan(B_y/B_z)$ in GSM coordinates

**수치 예시**: $v = 500$ km/s, $B = 20$ nT, $\theta = 180°$ (순수 남향)일 때:

$$\epsilon = 500 \times (7 \times 6371)^2 \times 20^2 \times \sin^4(90°) \approx 4 \times 10^{12} \text{ W}$$

**Numerical example**: For $v = 500$ km/s, $B = 20$ nT, $\theta = 180°$ (purely southward):
$\epsilon \approx 4 \times 10^{12}$ W — sufficient to drive an intense storm.

**$VB_s$ (Simple Electric Field Coupling)**:

$$E_y = VB_s \quad (\text{mV/m})$$

여기서 $B_s = |B_z|$ when $B_z < 0$, otherwise $B_s = 0$.

Intense storm 조건: $VB_s > 5$ mV/m이 3시간 이상 지속.

Intense storm condition: $VB_s > 5$ mV/m sustained for > 3 hours.

### 4.4 Storm–Substorm Efficiency / 폭풍–서브스톰 효율

$$Q(t) = \alpha(t) \cdot AL(t)$$

여기서 $\alpha(t)$는 시간 변동 효율 매개변수로, substorm activity (AL로 측정)가 ring current에 기여하는 비율을 나타낸다. 효율이 main phase 동안 크고 recovery phase에서 작아지는 것은 energy partitioning이 폭풍 단계에 따라 달라짐을 의미한다.

Where $\alpha(t)$ is a time-varying efficiency parameter representing the fraction of substorm activity (measured by AL) that contributes to the ring current. The efficiency being large during main phase and small during recovery indicates that energy partitioning varies with storm phase.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1800s  지자기 폭풍 최초 관측 (geomagnetic "storms" first named)
  │
1940  Chapman & Bartels — 지자기 이론 체계화 (Geomagnetism)
  │
1957  IGY — 국제 지구물리학의 해, 체계적 관측 시작
  │
1961  Dessler & Parker — DPS 관계 도출 (ring current ↔ Dst)
  │
1964  Sugiura — Dst 지수 도입
  │     Akasofu — substorm 개념 확립 (Paper #8)
  │
1973  McPherron et al. — substorm 위성 관측, NENL 모델 (Paper #10)
  │
1975  Burton et al. — Dst 예측 경험적 방정식 (Paper #11)
  │
1978  Perreault & Akasofu — ε coupling function
  │
1989  Quebec 정전 — 우주기상의 실용적 중요성 부각
  │
1991  ISTP 프로그램 시작; INPE workshop (이 논문의 기원)
  │
1994  ★ Gonzalez et al. — 지자기 폭풍의 정의와 분류 ★
  │     Dst 기반 표준 분류, storm ≠ substorm, 7가지 권고
  │
1997  ACE 위성 발사 — L1 실시간 태양풍 모니터링 실현
  │     Kamide et al. — 폭풍 예보 능력 재검토
  │
1999  NOAA G-scale 도입 (G1–G5 폭풍 등급)
  │
2003  Halloween storms — Dst < −400 nT, 현대 우주기상 최대 사건
  │
2015  DSCOVR 위성 발사 — ACE 후속 L1 모니터링
  │
현재  SYM-H (1분 해상도), 실시간 Dst, AI 기반 폭풍 예측
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| #8 Akasofu (1964) — Auroral Substorm | Substorm의 형태학적 정의를 제공. 이 논문에서 storm과 substorm을 구별하는 핵심 참조. / Provided morphological definition of substorms. Key reference for distinguishing storms from substorms. | 직접적 선행 / Direct prerequisite |
| #10 McPherron et al. (1973) — Satellite Substorm | NENL 모델과 위성 기반 substorm 관측. 이 논문에서 substorm 물리를 이해하는 기반. / NENL model and satellite-based substorm observations. Foundation for understanding substorm physics in this paper. | 직접적 선행 / Direct prerequisite |
| #11 Burton et al. (1975) — Dst Empirical Relation | Burton equation이 이 논문의 핵심 수학적 프레임워크. Dst 예측과 ring current 에너지 균형의 기반. / Burton equation is the core mathematical framework of this paper. Foundation for Dst prediction and ring current energy balance. | 핵심 기반 / Core foundation |
| #3 Chapman & Ferraro (1931) — Magnetosphere Theory | 자기권 이론의 시초. SSC와 initial phase의 물리적 기반을 제공. / Origin of magnetosphere theory. Provides physical basis for SSC and initial phase. | 역사적 기반 / Historical foundation |
| Perreault & Akasofu (1978) — ε Function | 에너지 결합 함수의 대표. 태양풍–자기권 에너지 전달률 추정의 표준. / Representative energy coupling function. Standard for estimating solar wind–magnetosphere energy transfer rate. | 수학적 도구 / Mathematical tool |
| Kamide & Fukushima (1971) — Storm Prediction | AE–Dst 관계를 이용한 폭풍 예측 시도. Figure 8의 근거. / Storm prediction attempt using AE–Dst relationship. Basis for Figure 8. | 방법론적 연결 / Methodological link |
| Saba et al. (1994) — AE–Dst Correlation | 22개 intense storm에 대한 AE–Dst 상관 분석. Storm–substorm decoupling의 정량적 증거. / AE–Dst correlation analysis for 22 intense storms. Quantitative evidence for storm–substorm decoupling. | 동시대 검증 / Contemporary validation |

---

## 7. References / 참고문헌

- Gonzalez, W. D., J. A. Joselyn, Y. Kamide, H. W. Kroehl, G. Rostoker, B. T. Tsurutani, and V. M. Vasyliunas, "What is a geomagnetic storm?", *J. Geophys. Res.*, 99(A4), 5771–5792, 1994. [doi:10.1029/93JA02867]
- Burton, R. K., R. L. McPherron, and C. T. Russell, "An empirical relationship between interplanetary conditions and Dst", *J. Geophys. Res.*, 80, 4204, 1975.
- Akasofu, S.-I., "The development of the auroral substorm", *Planet. Space Sci.*, 12, 273, 1964.
- McPherron, R. L., C. T. Russell, and M. P. Aubry, "Satellite studies of magnetospheric substorms on August 15, 1968", *J. Geophys. Res.*, 78, 3131, 1973.
- Dessler, A. J., and E. N. Parker, "Hydromagnetic theory of geomagnetic storms", *J. Geophys. Res.*, 64, 2239, 1959.
- Sckopke, N., "A general relation between the energy of trapped particles and the disturbance field near the Earth", *J. Geophys. Res.*, 71, 3125, 1966.
- Chapman, S., and V. C. A. Ferraro, "A new theory of magnetic storms", *Terr. Magn.*, 36, 77, 1931.
- Perreault, P., and S.-I. Akasofu, "A study of geomagnetic storms", *Geophys. J. R. Astron. Soc.*, 54, 547, 1978.
- Sugiura, M., "Hourly values of equatorial Dst for the IGY", *Ann. Int. Geophys. Year*, 35, 9, 1964.
- Kamide, Y., and T. Fukushima, "Analysis of magnetic storms with Dst indices for the equatorial ring current field", *Rep. Ionos. Space Res. Jpn.*, 25, 125, 1971.
- Saba, M. M. F., W. D. Gonzalez, and A. L. Clúa de Gonzalez, "Relationships between the AE, ap and Dst indices near solar minimum (1974) and at solar maximum (1979)", *Ann. Geophys.*, 15, 1265, 1997.
- Tsurutani, B. T., and W. D. Gonzalez, "The cause of high-intensity long-duration continuous AE activity (HILDCAA): Interplanetary Alfvén wave trains", *Planet. Space Sci.*, 35, 405, 1987.
- Takahashi, S., T. Iyemori, and M. Takeda, "A simulation of the storm-time ring current", *Planet. Space Sci.*, 38, 1133, 1990.
- Cornwall, J. M., "On the role of charge exchange in generating unstable waves in the ring current", *J. Geophys. Res.*, 82, 1188, 1977.
- Klein, L. W., and L. F. Burlaga, "Interplanetary magnetic clouds at 1 AU", *J. Geophys. Res.*, 87, 613, 1982.
- Marubashi, K., "Structure of the interplanetary magnetic clouds and their solar origins", *Adv. Space Res.*, 6(6), 335, 1986.
- Russell, C. T., and R. L. McPherron, "Semiannual variation of geomagnetic activity", *J. Geophys. Res.*, 78, 92, 1973.
