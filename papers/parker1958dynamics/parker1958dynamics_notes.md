---
title: "Dynamics of the Interplanetary Gas and Magnetic Fields"
authors: Eugene N. Parker
year: 1958
journal: "The Astrophysical Journal, Vol. 128, pp. 664–676"
doi: "10.1086/146579"
topic: Space Weather / Solar Wind Theory
tags: [solar wind, hydrodynamic expansion, coronal heating, Parker spiral, interplanetary magnetic field, supersonic flow, critical point, hydrostatic equilibrium]
status: completed
date_started: 2026-04-08
date_completed: 2026-04-08
---

# Dynamics of the Interplanetary Gas and Magnetic Fields (1958)
# 행성간 가스와 자기장의 역학 (1958)

---

## 핵심 기여 / Core Contribution

Eugene Parker는 Biermann (1951)이 혜성 꼬리 관측에서 추론한 "태양에서 지속적으로 가스가 유출된다"는 가설의 **유체역학적 필연성**을 증명했습니다. 논문의 논증 구조는 세 단계로 이루어집니다: (1) **정수압 평형의 실패 증명** — ~$10^6$ K의 코로나를 정수압 평형으로 가정하면, 무한대에서 유한한 압력($p(\infty) \approx 10^{-3} p_0$)이 남는데, 이를 상쇄할 성간 압력이 없으므로 정적 해는 물리적으로 불가능합니다. (2) **초음속 정상 팽창의 유도** — 정수압 평형 대신 운동 방정식과 연속 방정식을 결합하면, 코로나 가스가 태양 근처에서 아음속으로 출발하여 임계 반경($r_c = \lambda a/2$, 음속 도달 지점)을 지나 **초음속으로 가속**되는 유일한 물리적 해를 얻습니다. 코로나 온도 $T_0 = 1$–$3 \times 10^6$ K에서 지구 궤도에서의 속도는 ~200–900 km/s로, Biermann의 관측 추정치(500–1500 km/s)와 정량적으로 일치합니다. (3) **행성간 자기장의 나선 구조(Parker spiral) 예측** — 태양풍이 자기장선을 방사 방향으로 끌고 나가면서 태양 자전에 의해 나선 형태가 되며, 자기장은 $B_r \propto 1/r^2$, $B_\phi \propto 1/r$로 감소하여 먼 거리에서는 방위각 성분이 지배적입니다. 지구 궤도(1 AU)에서 나선각은 ~45°입니다.

Eugene Parker proved the **hydrodynamic inevitability** of Biermann's (1951) hypothesis that gas continuously streams from the Sun (inferred from comet tail observations). The argument proceeds in three stages: (1) **Proof of hydrostatic failure** — assuming hydrostatic equilibrium for a ~$10^6$ K corona yields finite pressure at infinity ($p(\infty) \approx 10^{-3} p_0$), with no interstellar pressure to counterbalance it, making the static solution physically impossible. (2) **Derivation of supersonic steady expansion** — combining the equation of motion and continuity equation instead gives a unique physical solution where coronal gas starts subsonic near the Sun, passes through a critical radius ($r_c = \lambda a/2$, where it reaches the sound speed), and **accelerates to supersonic velocities**. For coronal temperatures $T_0 = 1$–$3 \times 10^6$ K, the velocity at Earth's orbit is ~200–900 km/s, quantitatively matching Biermann's observational estimates (500–1500 km/s). (3) **Prediction of the spiral interplanetary magnetic field (Parker spiral)** — the solar wind drags field lines radially outward while solar rotation wraps them into spirals, with $B_r \propto 1/r^2$ and $B_\phi \propto 1/r$, so the azimuthal component dominates at large distances. The spiral angle at Earth's orbit (1 AU) is ~45°.

---

## 읽기 노트 / Reading Notes

### §I. Introduction / 서론

Parker는 Biermann (1951, 1952, 1957a)의 관측적 결론에서 출발합니다:

Parker begins from Biermann's (1951, 1952, 1957a) observational conclusions:

- 혜성의 **이온 꼬리(ion tail)**가 항상 태양 반대 방향을 가리킴 — 태양 복사압만으로는 설명 불가능
  Comet **ion tails** always point away from the Sun — unexplainable by solar radiation pressure alone
- 가스가 태양에서 **모든 방향으로** 500–1500 km/s로 유출됨
  Gas streams from the Sun **in all directions** at 500–1500 km/s
- 지구 궤도에서의 밀도: ~500 수소 원자/cm³
  Density at Earth's orbit: ~500 hydrogen atoms/cm³
- 질량 손실률: $10^{14}$–$10^{15}$ g/sec
  Mass loss rate: $10^{14}$–$10^{15}$ g/sec

Parker는 두 가지 질문을 제기합니다:

Parker raises two questions:

1. **어떤 메커니즘이 이 유출을 만드는가?** — 열 속도만으로는 부족. $T = 3 \times 10^6$ K에서도 수소 이온의 열 속도는 260 km/s에 불과하고, 태양 중력을 탈출하려면 500 km/s 이상이 필요
   What mechanism drives the outflow? Thermal velocity alone is insufficient — even at $3 \times 10^6$ K, hydrogen thermal velocity is only 260 km/s, while escape requires 500+ km/s

2. **유출이 태양 자기장을 어떻게 변형하는가?** — 이온화 가스는 자기장선에 동결되어 있으므로, 유출이 자기장을 끌고 나갈 것
   How does the outflow deform the solar magnetic field? Ionized gas is frozen to field lines, so the outflow should drag them outward

Parker의 핵심 전략: 가열 메커니즘의 세부를 모르더라도, 코로나가 ~$10^6$ K로 가열되어 있다는 **관측적 사실**만으로 유출의 역학을 다룰 수 있음.

Parker's key strategy: regardless of the unknown heating mechanism details, the **observed fact** that the corona is heated to ~$10^6$ K is sufficient to determine the outflow dynamics.

---

### §II. Static Equilibrium / 정수압 평형

이 절은 논문의 **논리적 기초**입니다 — Chapman (1957)의 정적 코로나 모델을 **반증**합니다.

This section is the **logical foundation** — it **refutes** Chapman's (1957) static corona model.

#### 2.1 기압 방정식 / Barometric Equation

완전 이온화 수소(이온 + 전자)의 총 압력 $p = 2NkT$에 대한 정수압 평형 (eq. 1):

Hydrostatic equilibrium for fully ionized hydrogen (ion + electron, total pressure $p = 2NkT$) (eq. 1):

$$0 = \frac{d}{dr}(2NkT) + \frac{GM_\odot M N}{r^2}$$

#### 2.2 온도 프로파일 / Temperature Profile

Spitzer (1947)의 열전도도 $\kappa(T) \cong 5 \times 10^{-7} T^n$ ergs/cm² sec °K를 사용합니다:

Using Spitzer's (1947) thermal conductivity $\kappa(T) \cong 5 \times 10^{-7} T^n$ ergs/cm² sec °K:

- 이온화 수소: $n = 5/2$ → $\kappa \propto T^{5/2}$
  Ionized hydrogen: $n = 5/2$
- 중성 수소: $n = 1/2$
  Neutral hydrogen: $n = 1/2$

정상 상태 열전도 방정식 $\nabla \cdot [\kappa(T)\nabla T] = 0$의 해 (eq. 3):

Steady-state heat conduction equation solution (eq. 3):

$$T(r) = T_0 \left(\frac{a}{r}\right)^{1/(n+1)}$$

이온화 수소($n = 5/2$)의 경우 $T \propto r^{-2/7}$ — **매우 느리게 감소**합니다. 이것이 핵심: 코로나의 높은 온도가 먼 거리까지 유지됩니다.

For ionized hydrogen ($n = 5/2$), $T \propto r^{-2/7}$ — **very slow decrease**. This is crucial: the corona's high temperature persists to large distances.

#### 2.3 밀도 해와 무한대에서의 압력 / Density Solution and Pressure at Infinity

정수압 평형의 밀도 해 (eq. 4):

$$N(r) = N_0 \left(\frac{r}{a}\right)^{1/(n+1)} \exp\left\{\frac{\lambda(n+1)}{n}\left[\left(\frac{a}{r}\right)^{n/(n+1)} - 1\right]\right\}$$

여기서 **$\lambda$**는 논문의 가장 중요한 무차원 매개변수입니다:

where **$\lambda$** is the paper's most important dimensionless parameter:

$$\lambda = \frac{GM_\odot M}{2akT_0}$$

$\lambda$의 물리적 의미: **중력 에너지와 열 에너지의 비율**. 코로나 기저($r = a$)에서 입자 하나의 중력 결합 에너지($GM_\odot M/a$)를 열 에너지($2kT_0$)로 나눈 값.

Physical meaning of $\lambda$: **ratio of gravitational to thermal energy** — gravitational binding energy per particle ($GM_\odot M/a$) divided by thermal energy ($2kT_0$) at the coronal base.

**핵심 결과 — 무한대에서의 압력 (eq. 9)**:

**Key result — pressure at infinity (eq. 9)**:

$$p(\infty) = p_0 \exp\left[\frac{-\lambda(n+1)}{n}\right]$$

수치적으로 / Numerically:
- 이온화 수소 ($n = 5/2$, $T_0 = 1.5 \times 10^6$ K): $\lambda = 5.35$ → $p(\infty) = 0.55 \times 10^{-3} p_0$
- 중성 수소 ($n = 0.5$): $p(\infty) = 10^{-7} p_0$

**Parker의 결정적 논증**: $p(\infty) \neq 0$이지만, 이 압력을 상쇄할 성간 매질의 압력이 없습니다. 성간 수소 ~10 atoms/cm³ at 100 K → $p_{\text{ISM}} \approx 1.4 \times 10^{-13}$ dynes/cm² ≪ $p(\infty)$. **따라서 정수압 평형은 불가능**하며, 코로나는 반드시 바깥으로 팽창해야 합니다.

**Parker's decisive argument**: $p(\infty) \neq 0$, but there is no interstellar pressure to counterbalance it. Interstellar hydrogen ~10 atoms/cm³ at 100 K → $p_{\text{ISM}} \approx 1.4 \times 10^{-13}$ dynes/cm² ≪ $p(\infty)$. **Therefore hydrostatic equilibrium is impossible**, and the corona must expand outward.

---

### §III. Stationary Expansion / 정상 팽창

정수압 평형이 불가능하므로, Parker는 정상 상태(시간 독립) 팽창 해를 구합니다.

Since hydrostatic equilibrium is impossible, Parker seeks a steady-state (time-independent) expansion solution.

#### 3.1 기본 방정식 / Governing Equations

구면 대칭 정상 흐름의 두 방정식:

Two equations for spherically symmetric steady flow:

**운동 방정식 / Equation of motion** (eq. 10):

$$NMv\frac{dv}{dr} = -\frac{d}{dr}(2NkT) - GM_\odot MN \frac{1}{r^2}$$

**연속 방정식 / Continuity equation** (eq. 11):

$$\frac{d}{dr}(r^2 Nv) = 0 \quad \Rightarrow \quad N(r)v(r) = N_0 v_0 \left(\frac{a}{r}\right)^2$$

연속 방정식의 의미: 단위 시간당 구면을 통과하는 질량 $4\pi r^2 NMv = \text{const}$. 밀도 × 속도는 $1/r^2$로 감소합니다.

Continuity equation meaning: mass flux through any spherical surface $4\pi r^2 NMv = \text{const}$. Density × velocity decreases as $1/r^2$.

#### 3.2 무차원화 / Non-Dimensionalization

Parker는 세 개의 무차원 변수를 도입합니다 (eq. 12–13):

Parker introduces three dimensionless variables:

$$\xi = r/a, \quad \tau = T(r)/T_0, \quad \psi = \frac{Mv^2}{2kT_0}$$

$\psi$의 물리적 의미: 운동 에너지를 열 에너지로 나눈 것. **$\psi = \tau$이면 $v = c_s$ (음속)**입니다.

Physical meaning of $\psi$: kinetic energy divided by thermal energy. **$\psi = \tau$ means $v = c_s$ (sound speed)**.

무차원 운동 방정식 (eq. 13):

$$\frac{d\psi}{d\xi}\left(1 - \frac{\tau}{\psi}\right) = -2\xi^2 \frac{d}{d\xi}\left(\frac{\tau}{\xi^2}\right) - \frac{2\lambda}{\xi^2}$$

#### 3.3 등온 코로나의 해 / Isothermal Corona Solution

온도가 $r = a$에서 $r = b$까지 $T_0$로 일정하고, $r > b$에서는 $T \approx 0$이라고 가정합니다.

Assume temperature is constant $T_0$ from $r = a$ to $r = b$, and $T \approx 0$ beyond $r = b$.

$r < b$ ($\tau = 1$) 영역에서 eq. 13을 적분하면 (eq. 14):

Integrating eq. 13 in the region $r < b$ ($\tau = 1$) gives (eq. 14):

$$\psi - \ln\psi = \psi_0 - \ln\psi_0 + 4\ln\xi - 2\lambda\left(1 - \frac{1}{\xi}\right)$$

#### 3.4 임계점과 유일한 물리적 해 / Critical Point and Unique Physical Solution

eq. 14는 **$Y = 4\ln\xi - 2\lambda(1 - 1/\xi)$와 $Z = \psi - \ln\psi$의 관계**입니다.

eq. 14 is a relationship between **$Y = 4\ln\xi - 2\lambda(1 - 1/\xi)$ and $Z = \psi - \ln\psi$**.

$Y$는 $\xi = \lambda/2$에서 최솟값을 가지고, $Z$는 $\psi = 1$ (음속)에서 최솟값을 가집니다. **$\psi$가 모든 $\xi$에서 실수이고 양수이려면**, $Y$와 $Z$가 동시에 최솟값을 가져야 합니다. 즉:

$Y$ has a minimum at $\xi = \lambda/2$, and $Z$ has a minimum at $\psi = 1$ (sound speed). For **$\psi$ to be real and positive for all $\xi$**, $Y$ and $Z$ must reach their minima simultaneously:

$$\psi = 1 \quad \text{at} \quad \xi = \frac{\lambda}{2}$$

이것이 **임계점(critical point)** 조건입니다: 흐름이 음속에 도달하는 반경 $r_c = \lambda a / 2$.

This is the **critical point** condition: the radius where the flow reaches the sound speed, $r_c = \lambda a / 2$.

임계점 조건으로부터 초기 속도 $\psi_0$가 유일하게 결정됩니다 (eq. 16):

The critical point condition uniquely determines the initial velocity $\psi_0$ (eq. 16):

$$\psi_0 - \ln\psi_0 = 2\lambda - 3 - 4\ln\frac{\lambda}{2}$$

**de Laval 노즐 유사성**: 이 임계점 전이는 로켓 노즐(de Laval nozzle)에서 아음속 → 초음속 전이와 수학적으로 동일합니다. 중력이 "노즐 목"의 역할을 합니다 — 압력 구배가 가스를 밀어내고, 중력이 저항하며, 임계점에서 둘이 균형을 이루는 순간 음속을 돌파합니다.

**De Laval nozzle analogy**: This critical point transition is mathematically identical to the subsonic → supersonic transition in a rocket nozzle. Gravity acts as the "nozzle throat" — pressure gradient pushes gas out, gravity resists, and at the critical point where they balance, the flow breaks through the sound speed.

#### 3.5 수치적 결과 — Figure 1 / Numerical Results — Figure 1

Parker의 Figure 1은 논문의 핵심 결과입니다 — 등온 코로나의 팽창 속도 $v(r)$ vs. $\xi = r/a$:

Parker's Figure 1 is the paper's central result — expansion velocity $v(r)$ vs. $\xi = r/a$ for an isothermal corona:

| 코로나 온도 $T_0$ | $\lambda$ | 임계점 $r_c$ | 지구 궤도(215$a$)에서의 $v$ |
|---|---|---|---|
| $0.5 \times 10^6$ K | 16.0 | $8a$ (~$5.5 R_\odot$) | ~100 km/s |
| $1.0 \times 10^6$ K | 8.0 | $4a$ (~$2.7 R_\odot$) | ~300 km/s |
| $1.5 \times 10^6$ K | 5.35 | $2.7a$ (~$1.8 R_\odot$) | ~500 km/s |
| $2.0 \times 10^6$ K | 4.0 | $2a$ (~$1.4 R_\odot$) | ~650 km/s |
| $3.0 \times 10^6$ K | 2.67 | $1.3a$ | ~900 km/s |

**핵심 관찰**: $T_0 = 1.5 \times 10^6$ K이면 지구 궤도에서 ~500 km/s — Biermann의 관측값과 정확히 일치합니다!

**Key observation**: $T_0 = 1.5 \times 10^6$ K gives ~500 km/s at Earth's orbit — matching Biermann's observations precisely!

#### 3.6 3차원 vs 1차원 팽창 / 3D vs 1D Expansion

Parker는 팽창의 차원성이 중요함을 보여줍니다 (eq. 18):

Parker shows dimensionality of expansion matters (eq. 18):

- **3차원 (구면)**: $v \to \infty$ as $r \to \infty$ — 속도가 무한히 증가 (ln $\xi$ 항 때문)
  **3D (spherical)**: velocity increases without limit
- **1차원**: $v$가 열 속도 이하로 제한됨 — 초음속 팽창 불가
  **1D**: velocity limited to below thermal speed — no supersonic expansion

**결론**: 초음속 태양풍은 **구면 팽창의 기하학적 효과** 덕분에 가능합니다.

**Conclusion**: Supersonic solar wind is possible thanks to the **geometric effect of spherical expansion**.

---

### §IV. Coronal Heating and Mass Loss / 코로나 가열과 질량 손실

#### 4.1 에너지 요구량 / Energy Requirements

태양풍의 운동 에너지 플럭스: $I(r) = \frac{1}{2}MNv^3 \cdot 4\pi r^2 \approx 1.5 \times 10^{29}$ ergs/sec

Solar wind kinetic energy flux: ~$1.5 \times 10^{29}$ ergs/sec

이것은 코로나의 열 복사 에너지($3 \times 10^{27}$ ergs/sec)의 **~50배**입니다. 태양풍을 유지하려면 코로나 가열이 단순한 열전도 손실 보충을 넘어 훨씬 더 많은 에너지를 공급해야 합니다.

This is **~50 times** the corona's thermal radiation energy ($3 \times 10^{27}$ ergs/sec). Maintaining the solar wind requires far more energy than merely replenishing conduction losses.

Parker는 가열 메커니즘으로 **hydromagnetic waves**(자기유체역학 파동, 후에 Alfvén파로 확인)를 제안합니다.

Parker suggests **hydromagnetic waves** (later identified as Alfvén waves) as the heating mechanism.

#### 4.2 질량 손실률 / Mass Loss Rate

$$\frac{dM_\odot}{dt} = 4\pi a^2 N_0 M v_0$$

$T_0 = 3 \times 10^6$ K, $N_0 = 3 \times 10^7$ cm⁻³: 질량 손실 ~$10^{14}$ g/sec → Biermann의 추정과 일치.

Mass loss ~$10^{14}$ g/sec → matches Biermann's estimate.

---

### §V. General Solar Magnetic Field / 태양 전반 자기장

이 절에서 Parker는 핵심 질문을 던집니다: 태양풍이 태양 자기장선에 **동결(frozen-in)**되어 있다면, 유출이 자기장을 어떻게 변형하는가?

Here Parker asks the key question: if the solar wind is **frozen-in** to solar field lines, how does the outflow deform the field?

#### 5.1 자기장선의 방사상 확장 / Radial Stretching of Field Lines

Parker의 논증:

Parker's argument:

1. 태양 표면에 자기장 없는 영역(field-free region)이 관측되지 않음 (Babcock & Babcock, 1955)
   No field-free regions observed on the solar surface
2. 따라서 유출 가스는 모두 자기장선을 관통하며, 동결 조건에 의해 자기장선을 끌고 나감
   Therefore all outflowing gas threads field lines and drags them outward by the frozen-in condition
3. $r > b$ (가열 영역 바깥)에서는 방사 방향으로 확장 → **$B_r \propto 1/r^2$** ($\nabla \cdot B = 0$에 의해)
   Beyond $r > b$, field lines stretch radially → **$B_r \propto 1/r^2$** (by $\nabla \cdot B = 0$)

---

### §VI. Interplanetary Magnetic Field and Retardation of Solar Rotation / 행성간 자기장과 태양 자전 감속

이 절에서 **Parker spiral**이 도출됩니다.

This section derives the **Parker spiral**.

#### 6.1 나선 자기장의 유도 / Derivation of the Spiral Field

태양과 함께 회전하는 좌표계에서 ($\omega \simeq 2.7 \times 10^{-6}$ rad/s), $r > b$에서 가스 속도:

In the co-rotating frame ($\omega \simeq 2.7 \times 10^{-6}$ rad/s), gas velocity for $r > b$:

$$v_r = v_m, \quad v_\theta = 0, \quad v_\phi = \omega(r - b)\sin\theta$$

(eq. 24) 여기서 $v_m$은 $r > b$에서의 일정한 방사 속도.

유선(streamline)의 방정식 (eq. 25):

$$\frac{r}{b} - 1 - \ln\left(\frac{r}{b}\right) = \frac{v_m}{b\omega}(\phi - \phi_0)$$

동결 조건($\vec{B} \parallel \vec{v}$, $\nabla \cdot \vec{B} = 0$)으로부터 자기장 성분 (eq. 26):

From frozen-in condition and $\nabla \cdot \vec{B} = 0$, field components (eq. 26):

$$B_r(r, \theta, \phi) = B(\theta, \phi_0)\left(\frac{b}{r}\right)^2$$

$$B_\theta = 0$$

$$B_\phi(r, \theta, \phi) = B(\theta, \phi_0)\frac{\omega}{v_m}(r - b)\left(\frac{b}{r}\right)^2 \sin\theta$$

#### 6.2 나선각과 1 AU에서의 기하학 / Spiral Angle and Geometry at 1 AU

$B_\phi / B_r$의 비율:

$$\frac{B_\phi}{B_r} = \frac{\omega(r - b)}{v_m}\sin\theta$$

이 비율은 $r$에 비례하므로, **먼 거리에서 자기장은 주로 방위각 방향**(나선)입니다.

This ratio increases with $r$, so at **large distances the field is primarily azimuthal** (spiral).

$B_\phi = B_r$이 되는 (나선각 45°) 반경 (eq. 27):

The radius where $B_\phi = B_r$ (spiral angle 45°) (eq. 27):

$$r = \frac{v_m}{\omega}\sin\theta$$

적도면($\theta = \pi/2$)에서 $v_m = 500$ km/s이면:
At the equator ($\theta = \pi/2$) with $v_m = 500$ km/s:

$$r \approx \frac{500 \text{ km/s}}{2.7 \times 10^{-6} \text{ rad/s}} \approx 1.9 \times 10^{13} \text{ cm} \approx 1.2 \text{ AU}$$

지구 궤도(1 AU)에서 나선각은 약 **45°** — 이것은 1963년 위성 관측으로 확인되었습니다!

At Earth's orbit (1 AU), the spiral angle is approximately **45°** — confirmed by satellite observations in 1963!

$v_m = 1000$ km/s이면 45° 반경은 ~2.5 AU (화성 궤도 부근).

For $v_m = 1000$ km/s, the 45° radius is ~2.5 AU (near Mars orbit).

#### 6.3 자기 토크와 태양 자전 감속 / Magnetic Torque and Solar Spin-Down

자기장이 태양에 가하는 토크:

Torque exerted on the Sun by the magnetic field:

$$L(r) = \frac{2}{15}b^4 \frac{\omega}{v_m}B_0^2\left(1 - \frac{b}{r}\right)$$

$v_m = 1000$ km/s, $b = 2 \times 10^{11}$ cm일 때: $L(\infty) = 5.8 \times 10^{30}$ dynes·cm

특성 감속 시간 $I\omega/L = 10^{18}$ 초 = $3 \times 10^{10}$ 년 — 태양 나이($5 \times 10^9$ 년)보다 길므로 자기 토크에 의한 태양 자전 감속은 무시 가능.

Characteristic spin-down time $I\omega/L = 10^{18}$ s = $3 \times 10^{10}$ yr — longer than the Sun's age ($5 \times 10^9$ yr), so magnetic torque spin-down is negligible.

---

### §VII. Plasma Instability and the Interplanetary Magnetic Shell / 플라즈마 불안정과 행성간 자기 껍질

#### 7.1 자기장 에너지 vs 운동 에너지 / Magnetic vs Kinetic Energy

태양 근처($r \lesssim 1$–2 AU): 자기장 에너지 밀도 $B^2/8\pi$와 운동 에너지 밀도 $\frac{1}{2}NMv^2$이 비슷한 크기.

Near the Sun ($r \lesssim 1$–2 AU): magnetic energy density $B^2/8\pi$ is comparable to kinetic energy density $\frac{1}{2}NMv^2$.

그러나 먼 거리에서: $B^2/8\pi \propto r^{-4}$ (방사 성분) while $\frac{1}{2}NMv^2 \propto r^{-2}$ → **운동 에너지가 지배**.

But at large distances: $B^2/8\pi \propto r^{-4}$ (radial component) while $\frac{1}{2}NMv^2 \propto r^{-2}$ → **kinetic energy dominates**.

#### 7.2 플라즈마 불안정 / Plasma Instability

1–2 AU 부근에서 플라즈마의 열적 압력이 이방성(anisotropic)이 됩니다: 자기장에 수직인 압력 $p_\perp$이 평행한 압력 $p_\parallel$보다 작아짐. 이 조건에서 **hydromagnetic wave**의 전파 속도가 허수가 되어, 진폭이 지수적으로 성장합니다.

Near 1–2 AU, thermal pressure becomes anisotropic: $p_\perp < p_\parallel$ relative to $\vec{B}$. Under this condition, the propagation speed of **hydromagnetic waves** becomes imaginary, causing exponential amplitude growth.

결과: 태양 내부의 질서있는 자기장이 1 AU 근처에서 **무질서한 자기 껍질(disordered magnetic shell)**로 변환됩니다 (~$10^{-5}$ gauss). 이 예측은 이후 우주선 관측으로 확인되었습니다.

Result: the ordered inner solar magnetic field transforms into a **disordered magnetic shell** (~$10^{-5}$ gauss) near 1 AU. This prediction was later confirmed by spacecraft observations.

---

## 핵심 시사점 / Key Takeaways

1. **정수압 평형의 실패가 태양풍의 존재를 필연적으로 만든다** — $10^6$ K 코로나의 정적 해는 무한대에서 유한한 압력을 가지며, 이를 상쇄할 성간 압력이 없으므로 팽창은 **물리적 필연**입니다. 이것은 "왜 태양풍이 존재하는가?"라는 질문을 "왜 코로나가 뜨거운가?"로 환원시킵니다.
   Hydrostatic failure makes the solar wind's existence **physically inevitable** — the static solution gives finite pressure at infinity with no counterbalancing interstellar pressure. This reduces "why does the solar wind exist?" to "why is the corona hot?"

2. **임계점은 de Laval 노즐과 동치이다** — 음속 전이의 수학적 구조가 유체역학의 초음속 노즐과 정확히 동일합니다. 중력이 노즐 목의 역할을 하며, 이 유사성은 이후 항성풍 이론의 기초가 됩니다.
   The critical point is equivalent to a de Laval nozzle — the mathematical structure of the sonic transition is identical. Gravity acts as the nozzle throat, and this analogy became foundational for stellar wind theory.

3. **코로나 온도가 태양풍의 모든 것을 결정한다** — $T_0$가 주어지면 $\lambda$가 결정되고, $\lambda$로부터 임계점 위치, 초기 속도, 최종 속도, 질량 손실률이 모두 계산됩니다. Fig. 1의 온도별 속도 곡선은 이 의존성을 극명하게 보여줍니다.
   Coronal temperature determines everything about the solar wind — given $T_0$, $\lambda$ is fixed, from which the critical point, initial velocity, terminal velocity, and mass loss rate all follow. Fig. 1 strikingly demonstrates this dependence.

4. **구면 팽창의 기하학이 초음속 흐름을 가능하게 한다** — 1차원에서는 흐름이 열 속도를 넘을 수 없지만, 3차원 구면 팽창에서는 ln $\xi$ 항 덕분에 무한한 가속이 가능합니다. 차원성이 물리의 본질을 바꾸는 아름다운 예입니다.
   Spherical geometry enables supersonic flow — in 1D, flow cannot exceed thermal velocity, but in 3D spherical expansion, the ln $\xi$ term allows unlimited acceleration. A beautiful example of dimensionality changing the physics.

5. **Parker spiral은 동결 자기장의 직접적 결과이다** — 방사상 태양풍 + 태양 자전 + 동결 조건 → 아르키메데스 나선. $B_r \propto 1/r^2$, $B_\phi \propto 1/r$이므로 먼 거리에서 방위각 성분이 지배하며, 1 AU에서 나선각 ~45°. 이것은 IMF $B_z$ 성분의 부호 변화를 통해 자기권-태양풍 결합의 핵심이 됩니다.
   The Parker spiral is a direct consequence of the frozen-in condition — radial wind + solar rotation + frozen-in → Archimedean spiral. Since $B_\phi \propto 1/r$ dominates over $B_r \propto 1/r^2$ at large distances, the spiral angle at 1 AU is ~45°. This becomes central to magnetosphere-solar wind coupling through IMF $B_z$ sign changes.

6. **Chapman-Ferraro의 "간헐적 덩어리"가 "연속 흐름"으로 대체된다** — Paper #2의 자기 폭풍 이론은 태양에서 간헐적으로 방출된 플라즈마 덩어리를 가정했지만, Parker는 태양풍이 **상시** 불고 있음을 보여줍니다. 이것은 자기권이 영구적 구조이며, 자기 폭풍은 태양풍의 **강화**(속도/밀도 증가)에 의한 것임을 의미합니다.
   Chapman-Ferraro's "intermittent clouds" are replaced by "continuous flow" — Paper #2's storm theory assumed intermittent plasma clouds, but Parker shows the solar wind blows **continuously**. This means the magnetosphere is a permanent structure, and storms result from solar wind **enhancements** (speed/density increases).

7. **이 논문은 과학적 용기의 모범이다** — 심사자 2명이 모두 거부한 논문을 Chandrasekhar가 출판한 것은 과학사의 전설적 일화입니다. Chapman 자신이 정적 코로나 모델의 대가였기에, 그의 모델을 정면으로 반박하는 이 논문은 상당한 학계 저항에 직면했습니다. 4년 후 Mariner 2가 태양풍을 직접 관측하면서 Parker의 이론은 극적으로 입증되었습니다.
   This paper exemplifies scientific courage — both reviewers rejected it, and Chandrasekhar published it by overruling them. Since Chapman himself was the authority on the static corona model, this direct rebuttal faced significant resistance. Four years later, Mariner 2's direct detection of the solar wind dramatically vindicated Parker.

8. **코로나 가열 문제는 여전히 열려 있다** — Parker 자신이 인정하듯, 이 논문은 코로나가 ~$10^6$ K라는 관측적 사실을 **가정**하고 그 결과를 도출합니다. 코로나를 어떻게 가열하는가는 별개의 문제이며, 60년이 넘도록 완전히 해결되지 않았습니다. 2018년 발사된 Parker Solar Probe의 핵심 과학 목표 중 하나입니다.
   The coronal heating problem remains open — as Parker himself acknowledges, this paper **assumes** the coronal temperature as an observational fact and derives consequences. How the corona is heated is a separate problem, still not fully solved after 60+ years. It is a key science objective of the Parker Solar Probe (launched 2018).

---

## 수학적 요약 / Mathematical Summary

### Parker 태양풍 모델의 핵심 방정식 / Core Equations of the Parker Solar Wind Model

**1. 정수압 평형 / Hydrostatic equilibrium** (eq. 1):
$$0 = \frac{d}{dr}(2NkT) + \frac{GM_\odot MN}{r^2}$$

**2. 열전도 온도 프로파일 / Thermal conduction temperature profile** (eq. 3):
$$T(r) = T_0\left(\frac{a}{r}\right)^{1/(n+1)}, \quad n = 5/2 \text{ (ionized H)}$$

**3. 무한대 압력 (정수압의 실패) / Pressure at infinity (hydrostatic failure)** (eq. 9):
$$p(\infty) = p_0 \exp\left[\frac{-\lambda(n+1)}{n}\right] \neq 0$$

**4. 무차원 중력 매개변수 / Dimensionless gravity parameter**:
$$\lambda = \frac{GM_\odot M}{2akT_0}$$

**5. 운동 방정식 / Equation of motion** (eq. 10):
$$NMv\frac{dv}{dr} = -\frac{d}{dr}(2NkT) - GM_\odot MN\frac{1}{r^2}$$

**6. 연속 방정식 / Continuity equation** (eq. 11–12):
$$N(r)v(r) = N_0 v_0 \left(\frac{a}{r}\right)^2$$

**7. 등온 팽창의 적분 / Isothermal expansion integral** (eq. 14):
$$\psi - \ln\psi = \psi_0 - \ln\psi_0 + 4\ln\xi - 2\lambda\left(1 - \frac{1}{\xi}\right)$$

**8. 임계점 조건 / Critical point condition** (eq. 16):
$$\psi_0 - \ln\psi_0 = 2\lambda - 3 - 4\ln\frac{\lambda}{2}$$

**9. Parker spiral 유선 / Parker spiral streamline** (eq. 25):
$$\frac{r}{b} - 1 - \ln\frac{r}{b} = \frac{v_m}{b\omega}(\phi - \phi_0)$$

**10. Parker spiral 자기장 / Parker spiral magnetic field** (eq. 26):
$$B_r = B_0\left(\frac{b}{r}\right)^2, \quad B_\phi = B_0\frac{\omega(r-b)}{v_m}\left(\frac{b}{r}\right)^2\sin\theta$$

---

## 역사적 맥락의 타임라인 / Paper in the Arc of History

```
1859  Carrington ─ 태양 플레어 → 자기 폭풍 최초 관측
  │                First flare-storm observation
  │
1908  ★ Birkeland ─ 태양 하전 입자 → 오로라 (#1)
  │                Solar particles → aurora
  │
1931  ★ Chapman & Ferraro ─ 간헐적 플라즈마 → 자기 폭풍 (#2)
  │                Intermittent plasma → magnetic storms
  │
1940  ★ Chapman & Bartels ─ 27일 재현, M-region (#3)
  │                27-day recurrence, M-regions
  │
1948  Biermann ─ 혜성 이온 꼬리 → 태양 복사압으로 설명 불가
  │             Comet ion tails → radiation pressure insufficient
  │
1951  Biermann ─ 태양에서 연속적 입자 방출 제안
  │             Continuous corpuscular radiation from Sun proposed
  │
1957  Chapman ─ 정적 코로나 모델 (정수압 평형)
  │             Static corona model (hydrostatic equilibrium)
  │
1958  ★★★ Parker ─ "Dynamics of the Interplanetary Gas" (#4) ← 이 논문
  │         정수압 평형 불가능 → 초음속 팽창 필연
  │         Hydrostatic equilibrium impossible → supersonic expansion inevitable
  │         Parker spiral 자기장 예측
  │         Parker spiral field predicted
  │
1958  Van Allen ─ 방사선대 발견 (#5 다음 논문)
  │               Radiation belt discovery
  │
1959  Parker ─ "solar wind" 용어 최초 사용
  │            Term "solar wind" coined
  │
1961  Dungey ─ 자기 재결합 → 열린 자기권 (#6)
  │            Magnetic reconnection → open magnetosphere
  │
1962  Mariner 2 ─ ★ 태양풍 직접 관측 확인 (Neugebauer & Snyder)
  │               Direct solar wind observation confirmed
  │
1963  Ness et al. ─ Parker spiral 구조 확인
  │                 Parker spiral structure confirmed
  │
1975  Burton et al. ─ Dst-태양풍 관계식 (#11)
  │                   Dst-solar wind relation
  │
2018  Parker Solar Probe ─ Parker 이름을 딴 최초의 NASA 미션
  │                        First NASA mission named after a living scientist
  │
현재  Parker Solar Probe가 태양 코로나 직접 진입 → 가열 메커니즘 규명 중
      PSP entering corona directly → investigating heating mechanism
```

---

## 다른 논문과의 연결 / Connections to Other Papers

| Paper / 논문 | Relationship / 관계 |
|---|---|
| **Birkeland (1908)** — #1 | Birkeland의 "태양 하전 입자" 가설이 Parker에 의해 **연속적 초음속 흐름**으로 정밀화됨. 간헐적 방출이 아닌 상시 흐름 / Birkeland's "solar charged particles" refined by Parker into a **continuous supersonic flow** — not intermittent |
| **Chapman & Ferraro (1931)** — #2 | Chapman-Ferraro의 "간헐적 플라즈마 덩어리"가 **연속 태양풍**으로 대체됨. 자기권이 상시 구조가 됨. 자기 폭풍은 태양풍 강화에 의한 것 / "Intermittent clouds" replaced by **continuous solar wind**. Magnetosphere becomes permanent; storms result from wind enhancement |
| **Chapman & Bartels (1940)** — #3 | 27일 재현과 M-region의 물리적 설명 제공 — 코로나 홀에서 나오는 고속 태양풍이 태양 자전에 따라 반복적으로 지구를 스침 / Physical explanation for 27-day recurrence and M-regions — fast solar wind from coronal holes sweeps Earth repeatedly with solar rotation |
| **Van Allen (1958)** — #5 다음 논문 | 연속 태양풍이 자기권에 에너지/입자를 공급하여 방사선대를 유지 / Continuous solar wind supplies energy/particles to maintain radiation belts |
| **Dungey (1961)** — #6 | Parker spiral의 $B_z$ 성분이 남향일 때 자기 재결합 → 태양풍 에너지가 자기권에 유입. IMF 방향이 우주기상의 핵심 / When $B_z$ of Parker spiral is southward → reconnection → solar wind energy enters magnetosphere. IMF direction is key to space weather |
| **Burton et al. (1975)** — #11 | Parker의 태양풍 매개변수(속도, 밀도, IMF)를 Dst에 연결하는 경험적 관계식 / Empirical relation connecting Parker's wind parameters (speed, density, IMF) to Dst |
| **Biermann (1951)** | 혜성 꼬리에서 태양 가스 유출을 추론 — Parker 이론의 관측적 출발점 / Inferred solar gas outflow from comet tails — observational starting point for Parker's theory |
| **Chapman (1957)** | 정적 코로나 모델 — Parker가 정면 반박한 대상. 정수압 평형의 한계를 드러냄 / Static corona model — directly refuted by Parker, revealing the limits of hydrostatic equilibrium |

---

## 참고문헌 / References

- Parker, E.N., "Dynamics of the Interplanetary Gas and Magnetic Fields," *Astrophysical Journal*, Vol. 128, pp. 664–676, 1958. [DOI: 10.1086/146579]
- Biermann, L., "Kometenschweife und solare Korpuskularstrahlung," *Zeitschrift für Astrophysik*, Vol. 29, pp. 274–286, 1951.
- Biermann, L., "Solar corpuscular radiation and the interplanetary gas," *Observatory*, Vol. 107, pp. 109–110, 1957.
- Chapman, S., "Notes on the solar corona and the terrestrial ionosphere," *Smithsonian Contributions to Astrophysics*, Vol. 2, pp. 1–12, 1957.
- Chapman, S. and Ferraro, V.C.A., "A New Theory of Magnetic Storms," *Terrestrial Magnetism and Atmospheric Electricity*, Vol. 36, pp. 77–97, 1931.
- Chapman, S. and Bartels, J., *Geomagnetism*, Oxford University Press, 1940.
- Neugebauer, M. and Snyder, C.W., "Solar Plasma Experiment," *Science*, Vol. 138, pp. 1095–1097, 1962.
- Spitzer, L., "The temperature of interstellar matter, I," *Astrophysical Journal*, Vol. 107, pp. 6–33, 1948.
