---
title: "Ueber das Gleichgewicht der Sonnenatmosphäre (On the Equilibrium of the Solar Atmosphere)"
authors: Karl Schwarzschild
year: 1906
journal: "Nachrichten von der Königlichen Gesellschaft der Wissenschaften zu Göttingen, Math.-phys. Klasse, 1906, pp. 41–53"
topic: Solar Physics / Radiative Transfer
tags: [radiative equilibrium, adiabatic equilibrium, limb darkening, radiative transfer, solar atmosphere, temperature structure, optical depth, Stefan-Boltzmann law, Schwarzschild]
status: completed
date_started: 2026-04-08
date_completed: 2026-04-08
---

# On the Equilibrium of the Solar Atmosphere (1906)
# 태양 대기의 평형에 관하여 (1906)

**Karl Schwarzschild**

---

## Core Contribution / 핵심 기여

Karl Schwarzschild는 이 논문에서 태양 대기의 에너지 전달 메커니즘에 대한 근본적 질문 — **단열 평형(대류에 의한)인가 복사 평형(복사의 흡수·방출 균형에 의한)인가** — 을 정량적으로 분석합니다. 그는 Kirchhoff의 복사 법칙과 Stefan-Boltzmann 법칙을 결합하여, 복사 평형 하에서의 태양 대기 온도-깊이 관계를 **광학적 깊이(optical depth, $m$)**의 함수로 유도하고, 이로부터 태양 원반의 **주변감광(limb darkening)** — 태양 가장자리가 중심보다 어둡게 보이는 현상 — 을 이론적으로 예측합니다. 복사 평형 모델의 주변감광 예측 $F(i) = (1+2\cos i)/3$이 관측 데이터와 단열 평형 예측 $F(i) = \cos i$보다 훨씬 잘 일치함을 보여, 태양 외층이 복사 평형에 가깝다고 결론지었습니다. 이 논문은 **복사 전달(radiative transfer) 이론의 출발점**이며, 현대 항성 대기 물리학의 기초를 놓은 역사적 업적입니다. 또한 복사 평형의 안정성 조건을 분석하여, 태양 대기 외층은 복사 평형으로 안정하지만 내부는 대류 불안정이 발생할 수 있음을 예측하였는데, 이것은 현대에 알려진 태양의 **대류층(convection zone)**과 **복사층(radiative zone)** 구분의 최초 이론적 예견이기도 합니다.

Karl Schwarzschild quantitatively analyzes the fundamental question of energy transport in the solar atmosphere — **adiabatic equilibrium (by convection) vs. radiative equilibrium (by balance of radiation absorption and emission)**. Combining Kirchhoff's radiation law with the Stefan-Boltzmann law, he derives the temperature-depth relation under radiative equilibrium as a function of **optical depth** $m$, and from this theoretically predicts the Sun's **limb darkening**. He shows that the radiative equilibrium prediction $F(i) = (1+2\cos i)/3$ fits observational data far better than the adiabatic prediction $F(i) = \cos i$, concluding that the Sun's outer layers are close to radiative equilibrium. This paper is the **starting point of radiative transfer theory** and laid the foundation for modern stellar atmosphere physics. His stability analysis further predicted that the outer solar atmosphere is stable under radiative equilibrium while the interior may be convectively unstable — the first theoretical foreshadowing of the Sun's **convection zone** and **radiative zone** distinction.

---

## Reading Notes / 읽기 노트

### Part I: The Central Question — Adiabatic or Radiative? / 핵심 질문 — 단열인가 복사인가?

#### 논문의 출발점 / The Starting Point

Schwarzschild는 논문을 태양 표면의 다양한 현상 — 입자(granulation), 흑점(Sonnenflecken), 홍염(Protuberanzen) — 을 언급하며 시작합니다. 이러한 격렬한 현상의 물리적 조건을 이해하기 위해, 먼저 **평균적인 정상 상태(stationärer Zustand)**를 수학적으로 기술해야 한다고 주장합니다. 이것이 "기계적 평형(mechanisches Gleichgewicht)"입니다.

Schwarzschild opens by noting the Sun's turbulent phenomena — granulation, sunspots, prominences — and argues that to understand these, one must first mathematically describe the **mean stationary state**. This is the "mechanical equilibrium."

그런데 태양 대기의 평형에는 두 가지 극단적 형태가 있습니다:

There are two extreme forms of equilibrium in the solar atmosphere:

**단열 평형(Adiabatisches Gleichgewicht)**: 지구 대기와 유사하게, 상승하고 하강하는 가스 덩어리(대류)가 대기를 완전히 혼합하는 상태. 각 지점의 온도는 그 지점에서 단열적으로 팽창/압축한 기체 덩어리가 가질 온도와 같습니다. 에너지는 **물질의 운동(대류)**에 의해 전달됩니다.

**Adiabatic equilibrium**: Similar to Earth's atmosphere, where ascending and descending gas parcels (convection) thoroughly mix the atmosphere. Temperature at each point equals what an adiabatically expanding/compressing gas parcel would have. Energy is transported by **material motion (convection)**.

**복사 평형(Strahlungsgleichgewicht)**: 강하게 복사를 흡수·방출하는 대기에서, 대류가 없어도 복사의 흡수와 방출의 균형만으로 에너지가 전달되는 상태. 각 층이 흡수한 만큼 정확히 방출하며, 에너지는 **빛(복사)**에 의해 전달됩니다.

**Radiative equilibrium**: In a strongly absorbing and emitting atmosphere, energy is transported purely by **light (radiation)**, with each layer emitting exactly as much as it absorbs, even without convection.

Schwarzschild는 이 두 극단 중 태양에 더 적합한 것이 어느 쪽인지를 결정하려 합니다. 그의 전략: 두 모델 각각에서 **태양 원반의 밝기 분포(주변감광, limb darkening)**를 예측하고, 실제 관측과 비교하는 것입니다.

Schwarzschild aims to determine which extreme better fits the Sun. His strategy: predict **limb darkening** from each model and compare with actual observations.

#### 핵심 가정들 / Key Assumptions

Schwarzschild는 논문의 한계를 분명히 밝힙니다. 다음을 **무시**합니다:

Schwarzschild clearly states the paper's limitations. He **neglects**:

1. **빛의 산란(Streuung)**: Schuster(1905)가 중요하다고 지적한 대기 입자에 의한 빛의 산란을 무시. 이것은 나중에 Milne, Chandrasekhar 등이 보완합니다.
Light scattering by atmospheric particles — later addressed by Milne, Chandrasekhar.

2. **굴절(Refraction)**: 대기를 통과하는 빛의 굴절을 무시.
Refraction of light passing through the atmosphere.

3. **파장별 흡수 차이**: 흡수 계수 $a$가 파장에 무관하다고 가정 (회색 대기, grey atmosphere). 실제로는 파장에 따라 크게 다르지만, 첫 근사로 충분합니다.
Wavelength-dependent absorption — assumes absorption coefficient $a$ is wavelength-independent (grey atmosphere).

4. **중력의 높이 의존성과 구면 기하**: 중력이 일정하고 대기가 평면 평행(plane-parallel)이라 가정.
Height-dependence of gravity and spherical geometry — assumes constant gravity and plane-parallel atmosphere.

Schwarzschild 자신이 "이 분석은 결코 결정적이거나 강제적이지 않으며(keineswegs als abschliessend oder zwingend gelten)... 간단한 생각을 가장 단순한 형태로 제시(einen einfachen Gedanken zunächst in einfachster Form ausführt)"한다고 겸손하게 밝힙니다. 그러나 이 "간단한 생각"이 20세기 전체 항성 대기 이론의 초석이 되었습니다.

Schwarzschild himself modestly states this analysis is "by no means conclusive or compelling... presenting a simple idea in the simplest form." Yet this "simple idea" became the cornerstone of all 20th-century stellar atmosphere theory.

---

### Part II: Mathematical Framework — Three Types of Equilibrium / 수학적 체계 — 세 가지 평형

#### 기본 변수와 단위 체계 / Basic Variables and Unit System

Schwarzschild는 독특한 단위 체계를 도입합니다:

Schwarzschild introduces a unique unit system:

- $p$: 압력 (센티시말 도 — 섭씨가 아닌 **절대 온도** / pressure)
- $t$: 절대 온도 (absolute temperature, **not** Celsius)
- $\rho$: 밀도, 수소 원자 대비 (density, relative to hydrogen)
- $M$: 분자량 (molecular weight, relative to hydrogen)
- $g$: 중력 가속도, 지구 표면 대비 (gravity, relative to Earth surface) — 태양에서 $g = 27.7$
- $h$: 깊이, "균질 대기"(8 km) 단위로 안쪽을 양수 (depth, in units of "homogeneous atmosphere" = 8 km, positive inward)

이상 기체 법칙은:

The ideal gas law becomes:

$$\rho t = \frac{p \cdot M}{R}, \qquad R = 0.106 \tag{1}$$

정역학적 평형(hydrostatic equilibrium) — 각 층에서 중력과 압력 경도가 균형:

Hydrostatic equilibrium — gravity balances pressure gradient at each layer:

$$dp = \rho g \, dh \tag{2}$$

(1)과 (2)를 결합하면:

Combining (1) and (2):

$$\frac{dp}{p} = \frac{M}{R} \cdot \frac{g}{t} \, dh \tag{3}$$

이것은 세 가지 평형 유형 모두의 출발점입니다. 차이는 $t$와 $p$의 관계(즉, **상태 방정식**)에서 발생합니다.

This is the starting point for all three equilibrium types. The difference arises in the relationship between $t$ and $p$ (the **equation of state**).

#### a) 등온 평형 / Isothermal Equilibrium

가장 단순한 경우: 온도 $t$가 일정. 식 (3)을 적분하면:

Simplest case: constant temperature $t$. Integrating (3):

$$p = p_0 \, e^{\frac{Mg}{Rt}h}, \qquad \rho = \rho_0 \, e^{\frac{Mg}{Rt}h} \tag{4}$$

태양에서 $g = 27.7$, $T \approx 6000$K이므로, 압력과 밀도가 공기 기준 14.7 km마다, 수소 기준 212 km마다 10배씩 증가합니다. 태양 반지름에서 1초각(arcsecond)이 약 725 km에 해당하므로, 태양 가장자리가 매우 날카롭게 보이는 이유가 설명됩니다.

On the Sun, with $g = 27.7$ and $T \approx 6000$K, pressure and density increase tenfold every 14.7 km for air and 212 km for hydrogen. Since 1 arcsecond at the Sun corresponds to ~725 km, this explains why the solar limb appears sharp.

#### b) 단열 평형 / Adiabatic Equilibrium

기체가 단열적으로 팽창/압축될 때의 Poisson 관계식:

Poisson relations for adiabatic expansion/compression:

$$\frac{p}{p_0} = \left(\frac{\rho}{\rho_0}\right)^k = \left(\frac{t}{t_0}\right)^{\frac{k}{k-1}} \tag{5}$$

여기서 $k = c_p / c_v$는 비열비입니다: 단원자 기체 $k = 5/3$, 이원자 $k = 7/5$, 삼원자 $k = 4/3$, 다원자 기체는 $k \to 1$.

where $k = c_p/c_v$ is the ratio of specific heats: monatomic $k = 5/3$, diatomic $k = 7/5$, triatomic $k = 4/3$, polyatomic $k \to 1$.

(3)과 (5)를 결합하여 적분하면 **선형 온도 경사**를 얻습니다:

Combining (3) and (5) and integrating gives a **linear temperature gradient**:

$$t - t_0 = \frac{k-1}{k} \cdot \frac{Mg}{R} \cdot (h - h_0) \tag{6}$$

**핵심 결과**: 단열 평형에서 온도는 깊이에 따라 **일정한 비율로 선형 증가**합니다. 지구 대기에서 이것은 100 m당 약 1°C(건조 단열 감률)입니다. 태양에서는 이 경사가 27.7배 더 가파르며, 공기 기준으로 3.63 m당 1°, 수소 기준으로 52 m당 1°입니다.

**Key result**: In adiabatic equilibrium, temperature increases **linearly** with depth at a constant rate. On Earth this is ~1°C per 100 m (dry adiabatic lapse rate). On the Sun this gradient is 27.7× steeper: 1° per 3.63 m for air, 1° per 52 m for hydrogen.

또 다른 중요한 특성: 단열 대기에는 **유한한 외부 경계**가 있습니다. $t = \rho = p = 0$이 되는 지점이 존재합니다. 온도 6000°의 층에서 이 경계까지의 거리는 공기 기준 22 km, 수소 기준 300 km입니다.

Another important feature: an adiabatic atmosphere has a **finite outer boundary** where $t = \rho = p = 0$. The distance from the 6000° layer to this boundary is 22 km for air, 300 km for hydrogen.

#### c) 복사 평형 — 이 논문의 핵심 / Radiative Equilibrium — The Core of This Paper

여기서 Schwarzschild의 진정한 기여가 시작됩니다. 태양의 바깥 부분이 점점 뜨거워지고 밀도가 높아지는 가스 덩어리의 연속적 전이를 이룬다고 가정합니다. 각 층은 **동시에 흡수체이자 방출체**이며, 대류가 없을 때 어떤 온도 분포를 가져야 거대한 에너지 흐름이 정상적으로(stationary) 유지되는지를 묻습니다.

Here Schwarzschild's true contribution begins. He assumes the Sun's outer parts form a continuous transition of increasingly hot and dense gas. Each layer is **simultaneously an absorber and emitter**, and he asks what temperature distribution is needed to maintain the enormous energy flow in a steady state without convection.

**흡수 계수(Absorptionsvermögen) $a$**: 높이 $dh$의 얇은 층이 통과하는 빛의 $a \cdot dh$ 비율을 흡수합니다.

**Absorption coefficient $a$**: A thin layer of height $dh$ absorbs fraction $a \cdot dh$ of passing light.

**Kirchhoff의 법칙 적용**: 같은 층은 온도 $t$에서의 흑체 방출 $E = c \cdot t^4$ (Stefan-Boltzmann)에 비례하는 에너지 $E \cdot a \, dh$를 방출합니다.

**Applying Kirchhoff's law**: The same layer emits energy $E \cdot a \, dh$ proportional to the blackbody emission $E = c \cdot t^4$ (Stefan-Boltzmann) at temperature $t$.

이제 두 방향의 복사를 정의합니다:

Now define radiation in two directions:

- $A$: 바깥으로(위로) 향하는 복사 에너지 (outward-directed radiation)
- $B$: 안쪽으로(아래로) 향하는 복사 에너지 (inward-directed radiation)

각각에 대한 **복사 전달 방정식(radiative transfer equations)**:

**Radiative transfer equations** for each:

$$\frac{dB}{dh} = a(E - B) \tag{7}$$

$$\frac{dA}{dh} = -a(E - A) \tag{8}$$

식 (7)의 물리적 의미: 안쪽으로 향하는 복사 $B$가 $dh$만큼 더 깊이 들어가면, 층의 자체 방출($aE \cdot dh$)이 더해지고 흡수($aB \cdot dh$)가 빠지므로, 순 변화는 $a(E-B) \cdot dh$입니다.

Physical meaning of (7): As inward radiation $B$ penetrates deeper by $dh$, the layer's own emission ($aE \cdot dh$) is added and absorption ($aB \cdot dh$) is removed, so the net change is $a(E-B) \cdot dh$.

이제 핵심 변수 변환: **광학적 질량(optische Masse)** $m$을 도입합니다:

Now the key variable change: introduce **optical mass** $m$:

$$m = \int_{\infty}^{h} a \, dh \tag{9}$$

이것은 대기 바깥($\infty$)에서 깊이 $h$까지의 누적 흡수량입니다. 현대 천체물리학의 **optical depth** $\tau$와 본질적으로 같은 개념으로, Schwarzschild가 여기서 최초로 도입한 것입니다.

This is the cumulative absorption from the atmosphere's exterior ($\infty$) to depth $h$. Essentially the same concept as modern astrophysics' **optical depth** $\tau$ — first introduced by Schwarzschild here.

$m$으로 변수를 바꾸면 전달 방정식이 단순해집니다:

Changing variables to $m$ simplifies the transfer equations:

$$\frac{dB}{dm} = E - B, \qquad \frac{dA}{dm} = A - E \tag{10}$$

**복사 평형 조건**: 정상 상태에서 각 층이 흡수한 만큼 정확히 방출해야 하므로:

**Radiative equilibrium condition**: In steady state, each layer must emit exactly as much as it absorbs:

$$aA + aB = 2aE \qquad \Rightarrow \qquad A + B = 2E$$

보조 변수 $\gamma = A - E = E - B$를 도입하면, 식 (10)의 덧셈과 뺄셈으로:

Introducing auxiliary variable $\gamma = A - E = E - B$, addition and subtraction of (10) gives:

$$\frac{d\gamma}{dm} = 0, \qquad \frac{dE}{dm} = \gamma$$

따라서 $\gamma = \text{const}$이고, $E = E_0 + \gamma(m-1) = E_0 + \gamma m$. 경계 조건(대기 바깥 $m=0$에서 안쪽으로 향하는 복사 $B = 0$)을 적용하면:

So $\gamma = \text{const}$ and $E$ is linear in $m$. Applying boundary conditions ($B = 0$ at $m = 0$, meaning no inward radiation from outside):

$$\boxed{E = \frac{A_0}{2}(1 + m), \qquad A = \frac{A_0}{2}(2 + m), \qquad B = \frac{A_0}{2} m} \tag{11}$$

여기서 $A_0$는 대기 밖으로 나가는 관측 가능한 총 에너지입니다.

where $A_0$ is the observable total energy leaving the atmosphere.

**이 결과의 의미**: 흑체 방출 $E = ct^4$가 광학적 깊이 $m$에 따라 **선형으로** 증가합니다. $m=0$(표면)에서 $E = A_0/2$, 즉 $t^4$가 관측되는 유효 온도의 절반에 해당합니다. 이것이 유명한 **경계 온도(Grenztemperatur)** $\tau$입니다:

**Meaning of this result**: Blackbody emission $E = ct^4$ increases **linearly** with optical depth $m$. At $m=0$ (surface), $E = A_0/2$, meaning $t^4$ is half the effective temperature's value. This gives the famous **boundary temperature** $\tau$:

$$\tau = \frac{T}{\sqrt{2}} \approx \frac{6000}{\sqrt{2}} \approx 5050 \text{ K}$$

즉, 태양 대기의 최외곽 온도는 유효 온도의 $1/\sqrt{2}$ 배입니다. 이것은 현대 천체물리학에서 **Eddington-Barbier relation**의 원형입니다.

The outermost temperature of the solar atmosphere is $1/\sqrt{2}$ times the effective temperature. This is the prototype of the modern **Eddington-Barbier relation**.

#### 온도-깊이 관계의 수치적 결과 / Numerical Temperature-Depth Results

흡수 계수가 밀도에 비례한다고 가정($a = \rho k$)하면 $m = kp/g$이 되고, 식 (11)에서:

Assuming absorption coefficient proportional to density ($a = \rho k$), then $m = kp/g$, and from (11):

$$t^4 = \frac{1}{2} T^4 \left[1 + \frac{k}{g} p\right] \tag{14}$$

공기의 흡수 계수 $k \approx 0.6$을 사용한 수치 결과 (p. 47의 표):

Numerical results using air absorption coefficient $k \approx 0.6$ (table from p. 47):

| 깊이 $h$ / Depth | 온도 $t$ / Temperature | 광학적 깊이 $m$ | 밀도 $\rho$ |
|------|------|------|------|
| $-\infty$ (바깥) | 5050° | 0.000 | 0.00 |
| $-36.9$ km | 5060° | 0.008 | 0.02 |
| $-19.1$ | 5300° | 0.215 | 0.51 |
| 0.0 (기준면) | 7570° | 4.06 | 6.8 |
| $+12.0$ | 10100° | 15.0 | 18.7 |
| $+55.7$ | 20200° | 255.0 | 159.4 |

**등온 평형과의 비교**: 복사 평형의 온도는 깊이가 증가하면 등온 상태($\tau = 5050$°)에서 점점 벗어나 급격히 상승합니다. 대기 바깥으로 갈수록 복사 평형은 등온 평형에 수렴합니다 — 이것은 물리적으로 자연스럽습니다: 대기가 희박해지면 각 층이 거의 투명해져 복사가 자유롭게 통과하므로, 온도 변화가 작아집니다.

**Comparison with isothermal**: Radiative equilibrium temperature departs from isothermal ($\tau = 5050$°) with increasing depth. Toward the exterior, radiative equilibrium converges to isothermal — physically natural: as the atmosphere thins, layers become nearly transparent and temperature variation diminishes.

---

### Part III: Stability of Radiative Equilibrium / 복사 평형의 안정성

Schwarzschild는 복사 평형이 단열 평형보다 안정한지를 분석합니다. 핵심 논증:

Schwarzschild analyzes whether radiative equilibrium is more stable than adiabatic. Key argument:

**단열 온도 경사** (식 6에서):

**Adiabatic temperature gradient** (from equation 6):

$$\frac{dt}{dh}\bigg|_{\text{adiab}} = \frac{k-1}{k} \cdot \frac{Mg}{R}$$

**복사 평형 온도 경사** (식 14에서):

**Radiative equilibrium temperature gradient** (from equation 14):

$$\frac{dt}{dh}\bigg|_{\text{rad}} = \frac{1}{4}\left(1 - \frac{\tau^4}{t^4}\right) \cdot \frac{Mg}{R}$$

**안정성 조건**: 복사 평형의 온도 경사가 단열 경사보다 **작으면** 대기는 안정합니다. 만약 복사 평형의 경사가 더 크면, 가스 덩어리가 상승할 때 주위보다 뜨거워져 계속 상승하게 되므로 대류가 시작됩니다 (대류 불안정).

**Stability condition**: If the radiative gradient is **less steep** than the adiabatic gradient, the atmosphere is stable. If steeper, rising gas parcels become hotter than their surroundings and keep rising — convection begins (convective instability).

$$1 - \frac{\tau^4}{t^4} < 4 \cdot \frac{k-1}{k} \tag{17}$$

**결론**: $k > 4/3$ (삼원자 이상의 기체)일 때, 이 부등식은 **모든 깊이**에서 만족됩니다. 왼쪽은 최대 1에 접근하고, 오른쪽은 $4(k-1)/k$로 $k > 4/3$이면 항상 1 이상입니다.

**Conclusion**: For $k > 4/3$ (triatomic or higher gases), this inequality is satisfied at **all depths**. The left side approaches 1 at most, while the right side is $4(k-1)/k \geq 1$ for $k > 4/3$.

$$\boxed{\text{복사 평형은 } k > 4/3 \text{인 기체(삼원자 이상)에서 항상 안정}}$$
$$\boxed{\text{Radiative eq. is always stable for } k > 4/3 \text{ (triatomic or higher gases)}}$$

그러나 단원자(k=5/3)나 이원자(k=7/5) 기체에서는 **깊은 곳**(높은 $t$)에서 불안정이 발생할 수 있습니다. Schwarzschild는 이로부터 다음을 추론합니다:

However, for monatomic ($k=5/3$) or diatomic ($k=7/5$) gases, instability can occur at **great depths** (high $t$). Schwarzschild infers:

> "태양 대기의 바깥 껍질은 안정한 복사 평형에 있고, 내부는 아마 단열 평형에 가까운 대류 영역이 존재하여, 에너지원에서 에너지를 빼내는 역할을 할 것이다."

> "The outer shell of the solar atmosphere is in stable radiative equilibrium, while the interior likely contains a convective zone close to adiabatic equilibrium, responsible for extracting energy from its sources."

이것은 현대에 알려진 태양의 **복사층(radiative zone, 0.25–0.7 $R_\odot$)**과 **대류층(convection zone, 0.7–1.0 $R_\odot$)** 구분의 최초 이론적 예견입니다!

This is the first theoretical foreshadowing of the Sun's **radiative zone** (0.25–0.7 $R_\odot$) and **convection zone** (0.7–1.0 $R_\odot$)!

---

### Part IV: Limb Darkening — The Observational Test / 주변감광 — 관측적 검증

#### 각도별 복사 전달 / Angular Radiative Transfer

지금까지의 분석은 수직 방향의 복사만 고려했습니다. 실제로 태양 원반의 밝기 분포를 예측하려면, **비스듬한 각도**로 통과하는 복사를 고려해야 합니다.

The analysis so far only considered vertical radiation. To predict the solar disk's brightness distribution, radiation passing at **oblique angles** must be considered.

수직에서 각도 $i$만큼 기울어진 방향의 복사 $F(i)$에 대해:

For radiation $F(i)$ at angle $i$ from the vertical:

$$\frac{dF}{dh} = -\frac{\alpha}{\cos i}(E - F) \tag{18}$$

여기서 $\alpha$는 수직 방향 흡수 계수입니다. $\cos i$로 나누는 이유: 비스듬한 경로는 같은 층 $dh$를 통과하는 거리가 $dh/\cos i$로 더 길기 때문입니다.

where $\alpha$ is the vertical absorption coefficient. Division by $\cos i$ because the oblique path through the same layer $dh$ is longer: $dh/\cos i$.

광학적 깊이 $\mu = \int \alpha \, dh$를 정의하면:

Defining optical depth $\mu = \int \alpha \, dh$:

$$F(i) = \int_0^\infty E \, e^{-\mu \sec i} \, \frac{d\mu}{\cos i} \cdot \sec i \tag{20 simplified}$$

실용적 근사: 작은 $i$에서 $F(i)$가 거의 일정하고 큰 $i$에서만 급격히 변하므로, $a \approx 2\alpha$ 관계를 사용할 수 있습니다. 그러면 $m = 2\mu$이고:

Practical approximation: $F(i)$ is nearly constant for small $i$ and changes rapidly near $i = 90°$, so $a \approx 2\alpha$ can be used. Then $m = 2\mu$ and:

$$F(i) = \int_0^\infty E \, e^{-\frac{m}{2} \sec i} \, \frac{dm}{2} \cdot \sec i \tag{22}$$

#### 주변감광 법칙의 유도 / Derivation of the Limb Darkening Law

**복사 평형**의 경우, 식 (11)에서 $E = \frac{A_0}{2}(1+m)$을 (22)에 대입하면:

For **radiative equilibrium**, substituting $E = \frac{A_0}{2}(1+m)$ from (11) into (22):

$$F(i) = \frac{A_0}{2}(1 + 2\cos i)$$

중심 밝기($i=0$)를 1로 정규화하면:

Normalizing to center brightness ($i=0$) = 1:

$$\boxed{F(i) = \frac{1 + 2\cos i}{3}} \tag{28}$$

**단열 평형**의 경우, 삼원자 기체($k = 4/3$)를 가정하면:

For **adiabatic equilibrium**, assuming triatomic gas ($k = 4/3$):

$$\boxed{F(i) = \cos i} \tag{29}$$

#### 관측 데이터와의 비교 — 결정적 순간 / Comparison with Observations — The Decisive Moment

Schwarzschild는 G. Müller의 "Photometrie der Gestirne" (p. 323)에서 thermosäulen(열전대)과 bolometer(복사계)로 측정한 태양 원반의 밝기 분포 데이터를 인용합니다. $r/R$은 태양 원반 중심에서 가장자리까지의 분율적 거리이며, $\sin i = r/R$입니다.

Schwarzschild cites G. Müller's brightness distribution data from "Photometrie der Gestirne" (p. 323), measured with thermopiles and bolometers.

| $r/R$ | 관측 / Observed | 복사 평형 / Radiative Eq. | 단열 평형 / Adiabatic Eq. |
|------|------|------|------|
| 0.0 | 1.00 | 1.00 | 1.00 |
| 0.2 | 0.99 | 0.99 | 0.98 |
| 0.4 | 0.97 | 0.95 | 0.92 |
| 0.6 | 0.92 | 0.87 | 0.80 |
| 0.7 | 0.87 | 0.81 | 0.71 |
| 0.8 | 0.81 | 0.73 | 0.60 |
| 0.9 | 0.70 | 0.63 | 0.44 |
| 0.96 | 0.59 | 0.52 | 0.28 |
| 0.98 | 0.49 | 0.47 | 0.20 |
| 1.0 | (0.40) | 0.33 | 0.00 |

**분석**: 
- 단열 평형은 가장자리에서 밝기가 **0으로** 떨어지지만, 관측에서는 가장자리에서도 상당한 밝기(~0.40)가 남아 있습니다.
- 복사 평형은 가장자리에서도 $1/3$의 밝기를 예측하며, 관측과 훨씬 잘 일치합니다.
- 관측값이 복사 평형보다 약간 높은 것은 Schwarzschild가 무시한 산란, 파장 의존성 등의 효과 때문입니다.

**Analysis**:
- Adiabatic equilibrium predicts brightness dropping to **zero** at the limb, but observations show significant brightness (~0.40) remains.
- Radiative equilibrium predicts $1/3$ brightness at the limb, matching observations far better.
- Observed values slightly exceeding the radiative prediction reflect scattering, wavelength dependence, and other effects Schwarzschild neglected.

**Schwarzschild의 결론**: "복사 평형 모델이 관측과 더 잘 일치한다(Die Formeln (28) und (29) sind mit der Beobachtung zu vergleichen)." 태양 대기의 외층은 **복사 평형**에 가깝습니다.

**Schwarzschild's conclusion**: The radiative equilibrium model fits observations better. The Sun's outer layers are close to **radiative equilibrium**.

---

## Key Takeaways / 핵심 시사점

1. **태양 대기는 복사 평형에 가깝다 / The solar atmosphere is close to radiative equilibrium**: Schwarzschild는 관측적 증거(주변감광)를 통해, 태양 외층의 에너지 전달이 대류(물질 운동)가 아니라 복사(빛의 흡수와 방출)에 의해 지배됨을 보여주었습니다. 이것은 태양 물리학의 근본적 사실입니다.
Schwarzschild showed through observational evidence (limb darkening) that energy transport in the Sun's outer layers is dominated by radiation, not convection. This is a fundamental fact of solar physics.

2. **광학적 깊이 개념의 탄생 / Birth of the optical depth concept**: 대기의 물리적 깊이 $h$ 대신 누적 흡수량 $m = \int a \, dh$를 사용하면, 복사 전달 방정식이 극적으로 단순해집니다. 이 변수 변환은 현대 천체물리학에서 **optical depth** $\tau$로 표준화되어, 항성 대기, 성간 매질, 행성 대기 등 모든 복사 전달 문제의 기본 변수가 되었습니다.
Replacing physical depth $h$ with cumulative absorption $m = \int a \, dh$ dramatically simplifies the radiative transfer equations. This variable is now standardized as optical depth $\tau$, the fundamental variable in all radiative transfer problems.

3. **경계 온도 $\tau = T/\sqrt{2}$ — 복사 평형의 상징적 결과 / Boundary temperature $\tau = T/\sqrt{2}$ — iconic result of radiative equilibrium**: 태양 대기의 최외곽 온도가 유효 온도의 $1/\sqrt{2}$배라는 결과는, 복사 전달 이론의 가장 유명한 결과 중 하나입니다. 현대적 표현으로 "$\tau = 2/3$에서의 온도가 유효 온도"라는 Eddington-Barbier 근사의 원형입니다.
The result that the outermost atmospheric temperature is $1/\sqrt{2}$ times the effective temperature is one of the most famous results of radiative transfer theory — the prototype of the modern Eddington-Barbier approximation.

4. **대류층과 복사층의 최초 이론적 예견 / First theoretical prediction of convective and radiative zones**: 안정성 분석에서 "바깥 껍질은 복사 평형으로 안정, 내부는 대류 불안정 가능"이라는 결론은, 현대에 확인된 태양의 복사층(0.25–0.7 $R_\odot$)과 대류층(0.7–1.0 $R_\odot$) 구분을 70년 앞서 예견한 것입니다.
The stability analysis conclusion — "outer shell stable in radiative equilibrium, interior possibly convectively unstable" — anticipated the Sun's radiative/convection zone boundary by 70 years.

5. **주변감광은 대기 구조의 직접적 관측 증거이다 / Limb darkening is direct observational evidence of atmospheric structure**: 태양 가장자리가 어두운 것은 단순한 광학적 효과가 아니라, 태양 대기의 온도가 깊이에 따라 증가한다는 **직접적 증거**입니다. Schwarzschild는 이 관측 현상을 이론적 예측과 정량적으로 연결한 최초의 과학자입니다.
Limb darkening is not a mere optical effect but **direct evidence** that temperature increases with depth in the solar atmosphere. Schwarzschild was the first to quantitatively connect this observation to theoretical predictions.

6. **"간단한 생각을 가장 단순한 형태로" — 강력한 근사의 힘 / "A simple idea in the simplest form" — the power of good approximations**: Schwarzschild는 산란, 파장 의존성, 구면 기하 등을 모두 무시하고도, 회색 대기(grey atmosphere) + 평면 평행(plane-parallel) 근사만으로 관측을 상당히 잘 설명했습니다. 이것은 물리학에서 "올바른 물리를 포착하는 가장 단순한 모델"이 얼마나 강력한지를 보여주는 교과서적 사례입니다.
Despite neglecting scattering, wavelength dependence, and spherical geometry, Schwarzschild's grey, plane-parallel approximation explained observations remarkably well — a textbook example of how the simplest model capturing the right physics can be powerful.

7. **Kirchhoff의 법칙(Paper #3)이 정량적 도구가 되다 / Kirchhoff's law (Paper #3) becomes a quantitative tool**: Paper #3에서 확립된 "좋은 방출체 = 좋은 흡수체"라는 정성적 원리가, 이 논문에서 복사 전달 방정식의 형태로 **정량적 예측 도구**로 변환됩니다. 이것이 과학의 진보 방식입니다 — 관찰에서 법칙으로, 법칙에서 정량적 모델로.
The qualitative principle "good emitter = good absorber" from Paper #3 is transformed into a **quantitative predictive tool** as the radiative transfer equation. This is how science progresses — from observation to law to quantitative model.

8. **Schwarzschild의 비극적 천재성 / Schwarzschild's tragic genius**: 이 논문을 쓴 같은 과학자가 9년 후 제1차 세계대전 동부전선에서 복무하면서 Einstein 장방정식의 최초 정확해(블랙홀 해)를 유도합니다. 42세에 사망한 그의 짧은 생애에서, 복사 전달 이론과 일반상대론 모두에 기초적 기여를 했다는 것은 경이적입니다.
The same scientist who wrote this paper derived the first exact solution to Einstein's field equations (the Schwarzschild black hole solution) while serving on the WWI Eastern Front 9 years later. That he made foundational contributions to both radiative transfer and general relativity in a life spanning only 42 years is extraordinary.

---

## Mathematical Summary / 수학적 요약

### 정역학적 평형 / Hydrostatic Equilibrium

$$dp = \rho g \, dh, \qquad \rho t = \frac{pM}{R}$$

### 등온 평형 / Isothermal Equilibrium ($t = \text{const}$)

$$p = p_0 \, e^{\frac{Mg}{Rt}h}, \qquad \rho = \rho_0 \, e^{\frac{Mg}{Rt}h}$$

### 단열 평형 / Adiabatic Equilibrium

$$\frac{p}{p_0} = \left(\frac{t}{t_0}\right)^{\frac{k}{k-1}}, \qquad t - t_0 = \frac{k-1}{k} \cdot \frac{Mg}{R}(h - h_0)$$

### 복사 전달 방정식 / Radiative Transfer Equations

$$\frac{dB}{dm} = E - B, \qquad \frac{dA}{dm} = A - E, \qquad m = \int a \, dh$$

### 복사 평형 해 / Radiative Equilibrium Solution

$$E = \frac{A_0}{2}(1+m), \quad A = \frac{A_0}{2}(2+m), \quad B = \frac{A_0}{2}m$$

### 경계 온도 / Boundary Temperature

$$\tau = \frac{T}{\sqrt{2}} \approx 5050 \text{ K} \quad (T \approx 6000 \text{ K})$$

### 온도-깊이 관계 / Temperature-Depth Relation

$$t^4 = \frac{1}{2}T^4\left[1 + \frac{k}{g}p\right] = \tau^4\left[1 + \frac{k}{g}p\right]$$

### 주변감광 법칙 / Limb Darkening Laws

**복사 평형 / Radiative equilibrium**: $F(i) = \dfrac{1 + 2\cos i}{3}$

**단열 평형 / Adiabatic equilibrium** ($k = 4/3$): $F(i) = \cos i$

### 안정성 조건 / Stability Condition

$$1 - \frac{\tau^4}{t^4} < 4 \cdot \frac{k-1}{k}$$

항상 만족 for $k > 4/3$. Always satisfied for $k > 4/3$.

---

## Paper in the Arc of History / 역사적 맥락의 논문

```
1860  Kirchhoff — 복사 법칙: 방출 = 흡수 (#3) / Radiation law: emission = absorption
  │
1879  Stefan — E = σT⁴ (실험) / (empirical)
  │
1884  Boltzmann — E = σT⁴ (이론적 유도) / (theoretical derivation)
  │
1900  Planck — 흑체 복사 법칙 / Blackbody radiation law
  │
1905  Schuster — 태양 대기 산란 모델 / Solar atmosphere scattering model
  │
1906  ────► SCHWARZSCHILD — 복사 평형 vs 단열 평형 ◄────
  │         ★ 복사 전달 이론의 출발점 / Birth of radiative transfer ★
  │         ★ 주변감광 이론적 예측 / Limb darkening prediction ★
  │         ★ 광학적 깊이 개념 도입 / Optical depth introduced ★
  │
1907  Schwarzschild — 항성 복사 전달 확장 / Extended to stars
  │
1908  Hale — 흑점 자기장 발견 (#5) / Sunspot magnetic field
  │
1914  Eddington — 항성 내부 복사 모델 / Stellar interior radiation model
  │
1916  Schwarzschild — 일반상대론 정확해 / GR exact solution (black holes)
  │
1921  Milne — 복사 평형의 엄밀한 수학 / Rigorous radiative eq. math
  │
1930  Chandrasekhar — 복사 전달 이론 완성 / Radiative transfer theory completed
  │
1950s 일진학 발전에 의한 태양 내부 구조 확인 / Helioseismology confirms solar structure
  │
1998  Schou et al. — 타코클라인 발견 (#17) / Tachocline discovery
```

---

## Connections to Other Papers / 다른 논문과의 연결

| 논문 / Paper | 연결 / Connection |
|---|---|
| **#3 Kirchhoff & Bunsen (1860)** | Kirchhoff의 복사 법칙("좋은 방출체 = 좋은 흡수체")이 이 논문의 핵심 가정. 정성적 법칙을 정량적 전달 방정식으로 변환한 관계. / Kirchhoff's radiation law is this paper's core assumption. Transforms a qualitative law into quantitative transfer equations. |
| **#2 Fraunhofer (1814)** | Fraunhofer 선(흡수선)의 형성 조건 — 태양 대기의 온도-깊이 구조 — 을 이 논문이 최초로 정량적으로 모델링. / This paper first quantitatively models the temperature-depth structure that produces Fraunhofer absorption lines. |
| **#5 Hale (1908)** | 태양 대기의 물리적 구조(온도, 밀도, 광학적 깊이)를 이해하게 됨으로써, Hale가 흑점 스펙트럼에서 Zeeman 효과를 해석할 맥락 제공. / Understanding the physical structure of the solar atmosphere provided context for Hale's interpretation of the Zeeman effect in sunspot spectra. |
| **#9 Babcock (1961)** | 태양 자기 주기 모델에서, 대류층과 복사층의 경계(타코클라인)가 핵심 역할. Schwarzschild가 최초로 이 경계의 존재를 이론적으로 예견. / In the solar magnetic cycle model, the convective/radiative zone boundary (tachocline) plays a key role — first theoretically predicted by Schwarzschild. |
| **#17 Schou et al. (1998)** | 일진학(helioseismology)으로 태양 내부의 차등 회전을 매핑하여 타코클라인 발견. Schwarzschild의 대류/복사 영역 구분 예측을 관측적으로 확인. / Helioseismology mapped internal differential rotation and discovered the tachocline — observational confirmation of Schwarzschild's convective/radiative zone prediction. |
| **Eddington (1914)** | Schwarzschild의 복사 평형 개념을 항성 내부로 확장. 항성의 광도-질량 관계 유도. / Extended Schwarzschild's radiative equilibrium concept to stellar interiors, deriving the luminosity-mass relation. |
| **Chandrasekhar (1930s)** | 복사 전달 이론을 산란, 비등방성, 편광까지 포함하여 완성. Schwarzschild의 방정식이 출발점. / Completed radiative transfer theory including scattering, anisotropy, and polarization — starting from Schwarzschild's equations. |

---

## References / 참고문헌

- Schwarzschild, K., "Ueber das Gleichgewicht der Sonnenatmosphäre", *Nachrichten von der Königlichen Gesellschaft der Wissenschaften zu Göttingen, Math.-phys. Klasse*, 1906, pp. 41–53.
- Schuster, A., "Radiation through a Foggy Atmosphere", *Astrophysical Journal*, Vol. 21, pp. 1–22, 1905.
- Müller, G., *Die Photometrie der Gestirne*, Wilhelm Engelmann, Leipzig, 1897.
- Chandrasekhar, S., *Radiative Transfer*, Oxford University Press, 1950 (Dover reprint, 1960).
- Mihalas, D., *Stellar Atmospheres*, W.H. Freeman, 2nd ed., 1978.
- Rybicki, G.B. and Lightman, A.P., *Radiative Processes in Astrophysics*, Wiley, 1979.
