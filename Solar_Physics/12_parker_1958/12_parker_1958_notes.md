---
title: "Dynamics of the Interplanetary Gas and Magnetic Fields"
authors: Eugene N. Parker
year: 1958
journal: "The Astrophysical Journal, 128, 664–676"
topic: Solar Physics / Solar Wind
tags: [solar wind, corona, hydrodynamic expansion, supersonic flow, Parker spiral, interplanetary magnetic field, hydrostatic equilibrium, critical point, mass loss, coronal heating, magnetic torque, plasma instability]
status: completed
date_started: 2026-04-12
date_completed: 2026-04-13
---

# Dynamics of the Interplanetary Gas and Magnetic Fields (1958)
# 행성간 가스와 자기장의 역학 (1958)

**Eugene N. Parker** — Enrico Fermi Institute for Nuclear Studies, University of Chicago

---

## Core Contribution / 핵심 기여

Parker는 Biermann(1951)이 혜성 꼬리 관측으로 추론한 태양의 연속적 입자 방출에 대한 **완전한 유체역학적 이론**을 제공했다. 논증은 세 단계로 구성된다. **첫째**, 태양 코로나가 $\sim 10^6$ K의 고온이라면 정역학적 평형(hydrostatic equilibrium)이 불가능하다는 것을 증명했다 — 열전도에 의해 온도가 $T(r) \propto r^{-2/7}$로 천천히 감소하므로, 무한대 거리에서 압력이 유한값($p(\infty) \sim 10^{-5}$ dyne/cm²)에 수렴하는데, 이 값은 성간 매질의 압력($\sim 10^{-13}$ dyne/cm²)보다 $10^8$배 높다. 평형이 불가능하므로 코로나는 반드시 바깥으로 팽창해야 한다. **둘째**, 구대칭 정상 상태 팽창의 유체역학 방정식을 무차원화하여 풀었다. 핵심 결과는 열 에너지와 중력 에너지의 비율 $\lambda = GM_\odot M / 2kT_0 a$에 의해 유일한 해가 결정되며, 이 해는 코로나 기저부에서 아음속으로 시작하여 임계점 $r_c = \lambda a / 2$에서 음속을 돌파하고 이후 무한히 가속되는 **trans-sonic** 유출이라는 것이다. $T_0 = 1$–$4 \times 10^6$ K에서 1 AU 도달 속도는 Biermann이 요구한 500–1000 km/sec와 정확히 일치한다 (Fig. 1). **셋째**, 이 초음속 유출이 태양의 쌍극 자기장과 결합하면 — frozen-in 조건에 의해 — 자기력선이 태양 자전 때문에 나선형으로 감겨 **Archimedean spiral** (Parker spiral)을 형성함을 유도했다 (Eq. 25–26, Fig. 6). 적도면에서 $B_\phi = B_r$이 되는 거리는 ~2.5 AU ($v_m = 1000$ km/s)이며, 이 이후 $B_\phi$가 지배적이다. Parker는 또한 이 나선 자기장이 태양에 가하는 토크를 계산하여 태양 회전 감속 시간이 ~$3 \times 10^{10}$년으로 무시할 수 있음을 보였고, 1–2 AU 사이에서 플라즈마 불안정성에 의해 자기장이 혼란스러운 껍질(tangled shell)을 형성할 것이라 예측했다.

Parker provided a **complete hydrodynamic theory** for Biermann's (1951) inferred continuous solar particle emission. The argument has three parts. **First**, he proved that hydrostatic equilibrium is impossible for a $\sim 10^6$ K corona — thermal conduction gives $T(r) \propto r^{-2/7}$, so pressure at infinity converges to a finite value ($p(\infty) \sim 10^{-5}$ dyne/cm²), which is $10^8$ times above interstellar pressure. Since equilibrium is impossible, the corona must expand outward. **Second**, he solved the spherically symmetric steady-state hydrodynamic equations in dimensionless form. The key result: a unique solution is determined by the ratio of gravitational to thermal energy $\lambda = GM_\odot M / 2kT_0 a$, and this solution is a **trans-sonic** outflow starting subsonic at the coronal base, crossing the sound speed at the critical point $r_c = \lambda a / 2$, and accelerating indefinitely. For $T_0 = 1$–$4 \times 10^6$ K, velocities at 1 AU match Biermann's required 500–1000 km/sec exactly (Fig. 1). **Third**, he showed that this supersonic outflow combined with the solar dipole field — via the frozen-in condition — winds magnetic field lines into an **Archimedean spiral** (Parker spiral) (Eq. 25–26, Fig. 6). At the equator, $B_\phi = B_r$ at ~2.5 AU ($v_m = 1000$ km/s), beyond which $B_\phi$ dominates. Parker also calculated the magnetic torque on the Sun (spin-down time ~$3 \times 10^{10}$ yr, negligible) and predicted that plasma instabilities between 1–2 AU would create a tangled magnetic shell.

---

## Reading Notes / 읽기 노트

### §I: Introduction / 서론 (pp. 664–665)

Parker는 Biermann의 결론을 명시적으로 출발점으로 삼는다:

Parker explicitly takes Biermann's conclusions as his starting point:

> "Biermann (1951, 1952, 1957a) has pointed out that the observed motions of comet tails would seem to require gas streaming outward from the sun."

Biermann의 추정값을 정리하면: / Biermann's estimates summarized:

| 양 / Quantity | 값 / Value |
|---|---|
| 유출 속도 / Outflow velocity | 500–1500 km/sec |
| 지구 궤도 밀도 / Density at Earth | ~500 H atoms/cm³ |
| 질량 손실률 / Mass loss rate | $10^{14}$ gm/sec |
| 태양 질량 대비 / Relative to Sun | $\sim 10^{-14}$ $M_\odot$/yr |

Parker는 두 가지 핵심 질문을 제기한다:

Parker poses two key questions:

1. **태양에서 어떤 메커니즘이 $10^{14}$ gm/sec의 수소를 1000 km/sec로 방출하는가?** — Schlüter의 melon-seed 과정 등은 음속 이하로만 가속 가능 / What mechanism ejects hydrogen at these rates?
2. **유출하는 가스가 태양 쌍극 자기장에 어떤 영향을 미치는가?** — frozen-in이라면 자기장이 끌려 나갈 것 / What happens to the solar dipole field?

**에너지 논증**: 태양풍의 운동 에너지 flux: $I = \frac{1}{2}NMv^3 \times 4\pi r^2 \approx 1.5 \times 10^{29}$ ergs/sec (at 1 AU). 이것은 코로나의 열전도($2 \times 10^4$ ergs/cm² sec)와 복사($10^4$ ergs/cm² sec) 에너지 손실의 합보다 **$10^2$배** 크다. 따라서 코로나 가열이 단순한 부산물이 아니라 근본적 구동력일 수 있다.

**Energy argument**: Solar wind kinetic energy flux ($1.5 \times 10^{29}$ ergs/sec) is **$10^2$ times** larger than coronal thermal conduction + radiation losses combined. This suggests coronal heating may be the fundamental driver, not a side effect.

Parker는 이를 뒤집어 본다: 코로나 가열이 $10^6$ K를 유지하는 것이 **원인**이고, 태양풍은 그 **필연적 결과**라고 제안한다.

Parker inverts the logic: coronal heating maintaining $10^6$ K is the **cause**, and solar wind is the **inevitable consequence**.

---

### §II: Static Equilibrium — The Impossibility Proof / 정역학 평형 — 불가능성 증명 (pp. 665–666)

이것이 논문의 가장 중요한 절이다. Parker는 정역학 평형이 코로나에 대해 불가능함을 깔끔하게 증명한다.

This is the paper's most important section. Parker cleanly proves hydrostatic equilibrium is impossible for the corona.

**정역학 평형 방정식 (Eq. 1):**

$$0 = \frac{d}{dr}(2NkT) + \frac{GM_\odot MN}{r^2}$$

완전 이온화 수소이므로 $p = 2NkT$ (이온 + 전자 모두 기여). $M$은 수소 원자 질량.

For fully ionized hydrogen, $p = 2NkT$ (ions + electrons). $M$ is hydrogen atom mass.

**열전도에 의한 온도 분포 (Eq. 3):**

열전도도 $\kappa(T) \simeq 5 \times 10^{-7} T^n$ ergs/cm sec K. 열유속 방정식 $\nabla \cdot [\kappa(T) \nabla T] = 0$의 구대칭 해:

$$T(r) = T_0 \left(\frac{a}{r}\right)^{1/(n+1)}$$

- 이온화 수소: $n = 5/2$ → $T \propto r^{-2/7}$ (매우 천천히 감소!)
- 중성 수소: $n = 1/2$ → $T \propto r^{-2/3}$

**밀도와 압력 해 (Eq. 4, 7):**

$$N(r) = N_0 \left(\frac{r}{a}\right)^{1/(n+1)} \exp\left\{\left[\frac{\lambda(n+1)}{n}\right]\left[\left(\frac{a}{r}\right)^{n/(n+1)} - 1\right]\right\}$$

$$p(r) = p_0 \exp\left\{\left[\frac{\lambda(n+1)}{n}\right]\left[\left(\frac{a}{r}\right)^{n/(n+1)} - 1\right]\right\}$$

여기서 **$\lambda$가 핵심 무차원 매개변수**: / Where $\lambda$ is the key dimensionless parameter:

$$\lambda = \frac{GM_\odot M}{2kT_0 a}$$

물리적 의미: 코로나 기저부에서의 **중력 에너지 / 열 에너지** 비율.

Physical meaning: ratio of **gravitational energy / thermal energy** at the coronal base.

**무한대에서의 압력 (Eq. 9):**

$$p(\infty) = p_0 \exp\left[\frac{-\lambda(n+1)}{n}\right]$$

**수치 대입** ($a = 10^6$ km, $T_0 = 1.5 \times 10^6$ K, $N_0 = 3 \times 10^7$ cm$^{-3}$):

| 경우 / Case | $n$ | $\lambda$ | $p(\infty)$ | 성간 압력 대비 |
|---|---|---|---|---|
| 이온화 수소 / Ionized H | 5/2 | 5.35 | $0.55 \times 10^{-3} p_0 \approx 7 \times 10^{-6}$ dyne/cm² | **$10^7$배 초과** |
| 중성 수소 / Neutral H | 1/2 | 5.35 | $10^{-7} p_0$ | 여전히 초과 |

성간 매질 압력: $p_{ISM} \sim 10$ H atoms/cm³ at 100 K → $1.4 \times 10^{-13}$ dyne/cm²

**결론**: 어떤 합리적인 $n$ 값에서도 $p(\infty)$가 성간 압력보다 훨씬 크다. 따라서 **정역학 평형은 불가능**하고, 코로나는 반드시 팽창해야 한다.

**Conclusion**: For any reasonable $n$, $p(\infty)$ far exceeds interstellar pressure. Hydrostatic equilibrium is **impossible** — the corona must expand.

이 논증의 우아함은 놀라울 정도이다: 열전도가 온도를 천천히 감소시키기 때문에, 먼 거리에서도 "뜨거운" 가스가 남아 있고, 이 가스의 압력이 성간 매질에 대해 균형을 이룰 수 없다는 것이다. **코로나의 높은 온도 자체가 태양풍의 원인**이다.

The elegance is remarkable: thermal conduction causes temperature to decrease so slowly that "hot" gas persists at large distances, and this gas pressure cannot balance against the interstellar medium. **The high coronal temperature itself causes the solar wind**.

---

### §III: Stationary Expansion — The Parker Equation / 정상 상태 팽창 — Parker 방정식 (pp. 667–669)

정역학 평형이 불가능하므로, Parker는 정상 상태 팽창(stationary expansion)을 분석한다.

Since hydrostatic equilibrium is impossible, Parker analyzes stationary expansion.

**운동 방정식 (Eq. 10):**

$$NMv\frac{dv}{dr} = -\frac{d}{dr}(2NkT) - GM_\odot MN\frac{1}{r^2}$$

왼쪽: 관성력(유체 가속). 오른쪽: 압력 기울기 + 중력.

Left: inertial force. Right: pressure gradient + gravity.

**연속 방정식 (Eq. 11):**

$$\frac{d}{dr}(r^2 Nv) = 0 \quad \Rightarrow \quad N(r)v(r) = N_0 v_0 \left(\frac{a}{r}\right)^2$$

구대칭 팽창에서 질량 보존.

**무차원화**: $\xi = r/a$, $\tau = T/T_0$, $\psi = Mv^2/2kT_0$ (속도의 무차원 형태)

**Parker 방정식 (Eq. 13):**

$$\frac{d\psi}{d\xi}\left(1 - \frac{\tau}{\psi}\right) = -2\xi^2 \frac{d}{d\xi}\left(\frac{\tau}{\xi^2}\right) - \frac{2\lambda}{\xi^2}$$

왼쪽의 $(1 - \tau/\psi)$는 **핵심 인자**: $\psi = \tau$ (즉, $v = c_s$ 음속)일 때 **특이점(singularity)** 발생 → 이것이 trans-sonic 천이점.

The factor $(1 - \tau/\psi)$ on the left is **crucial**: singularity when $\psi = \tau$ (i.e., $v = c_s$ sound speed) → this is the trans-sonic transition point.

**등온 근사** ($\tau = 1$, $r < b$에서 일정 온도 유지): Parker는 $r = a$에서 $r = b$까지 코로나 가열이 온도를 $T_0$로 유지한다고 가정. $r > b$에서는 가열 중단, $\tau \approx 0$.

**Isothermal approximation**: Parker assumes heating maintains $T_0$ from $r = a$ to $r = b$; beyond $r = b$, heating stops and $\tau \approx 0$.

**등온 영역의 적분 (Eq. 14):**

$$\psi - \ln\psi = \psi_0 - \ln\psi_0 + 4\ln\xi - 2\lambda\left(1 - \frac{1}{\xi}\right)$$

**임계점 조건 유도**: 이 방정식의 해가 모든 $\xi \geq 1$에서 실수(양수)이려면, 우변의 두 함수 $Y = 4\ln\xi - 2\lambda(1-1/\xi)$와 $Z = \psi - \ln\psi$가 **같은 $\xi$에서** 최솟값을 가져야 한다. $Y$는 $\xi = \lambda/2$에서 최소, $Z$는 $\psi = 1$에서 최소. 따라서:

**Critical point condition derivation**: For real solutions at all $\xi \geq 1$, functions $Y$ and $Z$ must reach their minima at the same $\xi$. $Y$ is minimized at $\xi = \lambda/2$, $Z$ at $\psi = 1$. Therefore:

$$\psi_0 - \ln\psi_0 = 2\lambda - 3 - 4\ln\frac{\lambda}{2} \quad \cdots (16)$$

이것이 **고유값 조건(eigenvalue condition)**: 주어진 $T_0$ (따라서 $\lambda$)에 대해 초기 속도 $v_0$ (따라서 $\psi_0$)가 **유일하게** 결정된다!

This is the **eigenvalue condition**: for a given $T_0$ (hence $\lambda$), the initial velocity $v_0$ (hence $\psi_0$) is **uniquely** determined!

**임계점의 물리적 의미**: / **Physical meaning of the critical point**:

$$r_c = \frac{\lambda a}{2} = \frac{GM_\odot M}{4kT_0}$$

이것은 중력 에너지 = 열 에너지인 거리이다. 이 점에서 유속이 **정확히 음속**($v = c_s$)이며, 이 안쪽은 아음속, 바깥쪽은 초음속이다. de Laval 노즐의 목(throat)과 정확히 동일한 물리학이다.

This is the distance where gravitational energy = thermal energy. Flow speed equals the sound speed **exactly** here — subsonic inside, supersonic outside. Identical physics to a de Laval nozzle throat.

**Fig. 1의 핵심 결과** (p. 668): 다양한 코로나 온도에서의 팽창 속도:

| $T_0$ | $r_c$ (임계 반경) | $v$ at $r/a = 90$ (~1 AU) |
|---|---|---|
| $0.5 \times 10^6$ K | 매우 큼 | ~100 km/s |
| $1.0 \times 10^6$ K | ~50$a$ | ~300 km/s |
| $1.5 \times 10^6$ K | ~36$a$ | ~500 km/s |
| $2.0 \times 10^6$ K | ~25$a$ | ~600 km/s |
| $3.0 \times 10^6$ K | ~17$a$ | ~750 km/s |
| $4.0 \times 10^6$ K | ~12$a$ | ~900 km/s |

Biermann이 요구한 500 km/sec는 $T_0 \approx 1.5 \times 10^6$ K에서 자연스럽게 나온다 — 이것은 관측된 코로나 온도와 **정확히 일치**!

Biermann's required 500 km/sec emerges naturally at $T_0 \approx 1.5 \times 10^6$ K — **exactly matching** the observed coronal temperature!

**차원성의 중요성**: Parker는 팽창이 구면(3차원)이기 때문에 작동한다고 지적한다. 1차원에서는 $\ln\xi$ 항이 사라져 무한 가속이 불가능하다 (§III, p. 669). 3차원에서는 $4\ln\xi$가 무한히 커져 $\psi \to \infty$를 보장한다.

**Importance of dimensionality**: Parker notes expansion works because it's spherical (3D). In 1D, the $\ln\xi$ term vanishes and unlimited acceleration is impossible.

---

### §IV: Coronal Heating and Mass Loss / 코로나 가열과 질량 손실 (pp. 670–672)

Parker는 정상 상태 태양풍을 유지하기 위해 필요한 코로나 가열 에너지를 계산한다.

Parker calculates the coronal heating energy required to maintain steady-state solar wind.

**에너지 수송 속도 $w$와 열 속도 $u$의 비율 (Eq. 22):**

$$\frac{w}{u} = \left(\frac{2}{3}\right)^{1/2} \psi^{1/2} \left[\frac{1}{2}(\psi_m - \psi) - 1\right]$$

여기서 $\psi_m$은 $v = v_m = 500$ km/sec일 때의 $\psi$ 값.

**Fig. 4** (p. 671)에서 $w/u$를 그리면: $T_0 = 2$–$3 \times 10^6$ K에서 $w/u$가 1을 넘지 않는다. 이것은 기계적 에너지 수송 속도가 열 속도를 넘지 않아도 된다는 것 — 물리적으로 합리적.

**질량 손실률 (Eq. 23):**

$$\frac{dM_\odot}{dt} = 4\pi a^2 N_0 M v_0$$

**Fig. 5** (p. 672): $T_0 = 3 \times 10^6$ K, $N_0 = 3 \times 10^7$ cm$^{-3}$이면:
- $v_0 \approx 160$ km/sec
- $dM_\odot/dt = 10^{14}$ gm/sec — Biermann의 추정값과 **정확히 일치**!

Parker의 핵심 결론: 코로나 온도가 $2$–$3 \times 10^6$ K이면 Biermann이 요구한 모든 수치 — 500 km/sec 속도, $10^{14}$ gm/sec 질량 손실 — 가 자연스럽게 나온다. 코로나 가열의 기원이 궁극적 문제이며, Parker는 각주에서 Fermi 과정에 의한 hydromagnetic wave 가열을 제안한다.

Parker's conclusion: coronal temperature of $2$–$3 \times 10^6$ K naturally produces all of Biermann's numbers. Coronal heating origin is the ultimate question — Parker suggests hydromagnetic wave heating via Fermi acceleration in a footnote.

---

### §V: General Solar Magnetic Field — The Parker Spiral / 태양 일반 자기장 — Parker 나선 (pp. 672–674)

이 절은 논문의 **두 번째 대발견**이다. Parker는 놀라울 정도로 간단한 기하학적 논증으로 행성간 자기장의 나선 구조를 유도한다.

This section contains the paper's **second great discovery**. Parker derives the spiral structure of the interplanetary magnetic field with remarkably simple geometry.

**핵심 가정**: 태양에 자기장이 없는 영역(field-free region)이 없다. 모든 가스가 자기력선에 관통되어 있으므로, 유출하는 가스가 자기장을 끌고 나간다 (frozen-in).

**Key assumption**: No field-free regions in the Sun. All gas is threaded by field lines, so outflowing gas drags the field outward (frozen-in).

**유선(streamline) 방정식** — $r = b$ 이상에서 $v_r = v_m = \text{const}$, $v_\theta = 0$, $v_\phi = \omega(r-b)\sin\theta$:

$$\frac{r}{b} - 1 - \ln\frac{r}{b} = \frac{v_m}{b\omega}(\phi - \phi_0) \quad \cdots (25)$$

이것은 **Archimedean spiral** ($r \gg b$일 때)이다.

**자기장 성분 (Eq. 26):**

$\nabla \cdot \mathbf{B} = 0$과 정상 상태 조건에서 자기장이 유선과 일치하므로:

$$B_r(r, \theta, \phi) = B(\theta, \phi_0)\left(\frac{b}{r}\right)^2$$

$$B_\theta = 0$$

$$B_\phi(r, \theta, \phi) = B(\theta, \phi_0)\left(\frac{\omega}{v_m}\right)(r-b)\left(\frac{b}{r}\right)^2 \sin\theta$$

**나선각 $\Psi$ (자기장과 방사 방향 사이의 각도):**

$$\tan\Psi = \frac{B_\phi}{B_r} = \frac{\omega(r-b)\sin\theta}{v_m}$$

**45° 나선각 도달 거리** (적도면, $\sin\theta = 1$):

$$r_{45°} = b + \frac{v_m}{\omega} \approx \frac{v_m}{\omega}$$

$v_m = 1000$ km/sec, $\omega = 2.7 \times 10^{-6}$ rad/sec이면:

$$r_{45°} = \frac{10^8}{2.7 \times 10^{-6}} \approx 3.7 \times 10^{13} \text{ cm} \approx 2.5 \text{ AU}$$

**지구 궤도(1 AU)에서의 나선각**: $\tan\Psi = \omega \cdot 1\text{AU} / v_m$

| $v_m$ (km/s) | $\Psi$ at 1 AU |
|---|---|
| 400 (slow wind) | ~45° |
| 700 (fast wind) | ~28° |
| 1000 | ~20° |

Parker의 Fig. 6 (p. 674)은 적도면에 투영된 자기력선을 보여주며, 이것이 바로 **Parker spiral**이다 — 현대 heliospheric physics의 가장 기본적인 구조.

Parker's Fig. 6 (p. 674) shows field lines projected onto the equatorial plane — this is the **Parker spiral**, the most fundamental structure in modern heliospheric physics.

---

### §VI: Interplanetary Magnetic Field and Retardation of Solar Rotation / 행성간 자기장과 태양 회전 감속 (pp. 674–675)

Parker는 나선 자기장이 태양에 가하는 토크를 계산한다.

Parker calculates the torque exerted on the Sun by the spiral magnetic field.

**자기 에너지 밀도 vs 운동 에너지 밀도**:

$$\frac{B^2/8\pi}{\frac{1}{2}NMv^2} \propto r^{-4} \text{ (for } r < r_{45°}\text{)} \text{ vs } r^{-2}$$

자기 에너지는 $r^{-4}$로 감소하지만 운동 에너지는 $r^{-2}$로만 감소하므로, 먼 거리에서 운동 에너지가 지배적이다. 따라서 자기장이 유출하는 가스의 운동에 큰 영향을 미치지 않는다.

**토크 계산 결과:**

$$L(\infty) = 5.8 \times 10^{30} \text{ dynes/cm}$$

태양의 관성 모멘트 $I \approx 2 \times 10^{54}$ gm cm²이므로:

$$t_{\text{spin-down}} = \frac{I\omega}{L} = \frac{I}{L/\omega} \approx 3 \times 10^{10} \text{ yr}$$

이것은 태양 나이($4.6 \times 10^9$ yr)보다 길어서 **무시할 수 있다**. 그러나 젊은 별에서는 이 메커니즘이 중요할 수 있다 — 현대의 "magnetic braking" 이론의 선구.

Spin-down time exceeds the Sun's age, so it's **negligible**. But for young stars this mechanism could be important — a precursor to modern "magnetic braking" theory.

---

### §VII: Plasma Instability and the Interplanetary Magnetic Shell / 플라즈마 불안정성과 행성간 자기 껍질 (p. 675)

Parker는 마지막으로 행성간 공간에서의 자기장 구조의 불안정성을 논의한다.

Parker finally discusses magnetic field structure instability in interplanetary space.

1–2 AU에서 자기장 에너지가 운동 에너지보다 빠르게 감소하므로, 자기력선은 방사 방향으로 "늘어나게" 된다. 이때 열 운동이 비등방적이 되면 ($p_\perp > p_\parallel$, 여기서 $\perp$과 $\parallel$은 자기장에 대해), hydromagnetic wave의 전파가 순허(purely imaginary)가 되어 **불안정성**이 발생한다.

결과: 약 1 AU 바깥에서 깔끔한 나선 구조가 혼란스러운 **tangled field shell** ($\sim 10^{-5}$ gauss)로 변한다. Parker는 이것이 우주선(cosmic ray) 관측에서 이미 추론된 바와 일치한다고 지적한다 (Meyer, Parker & Simpson 1956; Simpson 1957).

Result: Beyond ~1 AU, the clean spiral structure transitions to a **tangled field shell** ($\sim 10^{-5}$ gauss). Parker notes this is consistent with cosmic ray observations.

---

## Key Takeaways / 핵심 시사점

1. **정역학 평형의 불가능성이 태양풍을 증명한다**: Parker의 가장 근본적인 기여는 태양풍의 존재를 **관측이 아닌 이론으로** 예측한 것이다. $10^6$ K 코로나 + 열전도 → $p(\infty) > p_{ISM}$ → 평형 불가능 → 팽창 필연. 이 3단계 논증은 과학사에서 가장 우아한 이론적 예측 중 하나이며, 코로나 온도라는 **단일 관측 사실**에서 태양풍의 존재를 필연적으로 유도한다. / Parker's most fundamental contribution is predicting the solar wind from **theory, not observation**. The three-step argument — hot corona + conduction → finite $p(\infty)$ → equilibrium impossible → expansion inevitable — derives the solar wind's existence from a single observed fact: coronal temperature.

2. **Trans-sonic 해의 유일성과 고유값 문제**: Parker 방정식의 해는 무수히 많지만, 물리적으로 유의미한 해 — 아음속에서 시작하여 초음속으로 전이하는 것 — 는 **단 하나**뿐이다. 주어진 $T_0$에 대해 $v_0$가 유일하게 결정되는 이 고유값 문제의 구조는 de Laval 노즐과 동일하며, 이것은 우연이 아니라 수렴-확산하는 유효 단면적($r^2$가 역할)의 물리학에서 비롯된다. / The Parker equation has infinitely many solutions, but only **one** physically meaningful trans-sonic solution exists. This eigenvalue structure is identical to a de Laval nozzle — not coincidental but from converging-diverging effective cross-section physics.

3. **차원성이 핵심이다**: 1차원 팽창에서는 $\ln\xi$ 항이 없어 무한 가속이 불가능하고 아음속 바람만 가능하다. 3차원(구면) 팽창에서만 $4\ln\xi$가 등장하여 초음속 가속이 가능하다. 이것은 왜 평면 대기(지구)는 증발(evaporation)만 가능하고 구면 코로나(태양)는 바람(wind)을 일으키는지를 설명한다. / In 1D, the $\ln\xi$ term vanishes and only subsonic breeze is possible. Only 3D (spherical) expansion produces the $4\ln\xi$ term enabling supersonic acceleration — explaining why flat atmospheres evaporate while spherical coronae produce winds.

4. **Parker spiral의 단순함과 보편성**: 나선 구조는 단 두 가지 효과 — 방사 유출 + 태양 자전 — 의 조합으로부터 기하학적으로 **필연적**이다. $\tan\Psi = \omega r \sin\theta / v_m$이라는 간단한 공식이 현대 heliospheric physics의 기초이며, 1965년 Ness의 관측으로 정확히 확인되었다. 이 공식의 보편성은 자전하는 모든 별에 적용 가능하며, 현대 astrospheric physics의 시작이기도 하다. / The spiral structure follows geometrically from just two effects — radial outflow + rotation. The simple formula $\tan\Psi = \omega r \sin\theta / v_m$ is the foundation of modern heliospheric physics, confirmed by Ness (1965), and applies universally to all rotating stars.

5. **Biermann-Parker 축: 관측에서 이론으로의 완벽한 다리**: Biermann(1951)이 혜성 꼬리에서 태양풍의 **존재**를 추론했다면, Parker(1958)는 코로나의 열역학에서 태양풍의 **필연성**을 증명했다. 이 두 논문은 과학적 발견의 이상적 패턴 — 관측 증거 → 이론적 정당화 → 예측 → 실험 확인(Mariner 2, 1962) — 을 완벽히 구현한다. / Biermann (1951) inferred solar wind **existence** from comet tails; Parker (1958) proved its **inevitability** from coronal thermodynamics. Together they exemplify the ideal pattern: observational evidence → theoretical justification → prediction → experimental confirmation.

6. **논문 거부의 역사적 교훈**: 심사자 2명 모두 이 논문을 거부했다. Chapman은 자신의 정적 코로나 모델을 강하게 지지했고, 당시 학계 대다수가 회의적이었다. 편집자 Chandrasekhar가 개인적으로 게재를 결정했다. 이것은 패러다임 전환이 초기에 저항을 받는 전형적인 사례이며, 과학적 동료 심사의 한계를 보여준다. 4년 후 Mariner 2의 직접 관측이 Parker의 예측을 확인하면서 논쟁이 종결되었다. / Both referees rejected this paper. Chandrasekhar personally overruled them. This is a classic case of paradigm-shift resistance, resolved when Mariner 2 confirmed Parker's predictions in 1962.

7. **현대 태양풍 물리학의 씨앗**: 이 논문에서 Parker가 미해결로 남긴 문제들 — 코로나 가열 메커니즘, 자기장의 유출에 대한 역효과, 다차원 효과, 태양 주기에 따른 변동 — 은 60년 넘게 현대 태양물리학의 핵심 연구 주제로 남아 있다. 2018년 발사된 Parker Solar Probe는 그의 이름을 딴 최초의 NASA 탐사선이며, 이 미해결 문제들에 답하기 위해 태양에 역사상 가장 가까이 접근하고 있다. / The unsolved problems Parker left — coronal heating mechanism, back-reaction of magnetic fields, multi-dimensional effects, solar cycle variations — have remained core research topics for 60+ years. Parker Solar Probe (2018) carries his name and approaches closer to the Sun than ever to answer these questions.

---

## Mathematical Summary / 수학적 요약

### Parker 이론의 완전한 수학적 구조 / Complete Mathematical Structure

**1단계: 불가능성 증명 (§II)**

$$\text{Hydrostatic: } 0 = \frac{d}{dr}(2NkT) + \frac{GM_\odot MN}{r^2}$$
$$\text{Heat conduction: } T(r) = T_0(a/r)^{2/7} \text{ (ionized H)}$$
$$\Rightarrow p(\infty) = p_0 \exp[-\lambda(n+1)/n] \gg p_{ISM}$$
$$\therefore \text{ Hydrostatic equilibrium is IMPOSSIBLE}$$

**2단계: 팽창 방정식 (§III)**

$$\text{Momentum: } NMv\frac{dv}{dr} = -\frac{d}{dr}(2NkT) - \frac{GM_\odot MN}{r^2}$$
$$\text{Continuity: } N(r)v(r) = N_0 v_0 (a/r)^2$$

무차원화 ($\xi = r/a$, $\psi = Mv^2/2kT_0$, $\lambda = GM_\odot M/2kT_0 a$):

$$\text{등온 해: } \psi - \ln\psi = \psi_0 - \ln\psi_0 + 4\ln\xi - 2\lambda(1-1/\xi)$$

$$\text{고유값: } \psi_0 - \ln\psi_0 = 2\lambda - 3 - 4\ln(\lambda/2)$$

$$\text{임계점: } r_c = \lambda a/2, \quad v(r_c) = c_s = \sqrt{2kT_0/M}$$

**3단계: Parker Spiral (§V)**

$$\text{유선: } r/b - 1 - \ln(r/b) = (v_m/b\omega)(\phi-\phi_0)$$

$$B_r = B_0(b/r)^2, \quad B_\phi = B_0(\omega/v_m)(r-b)(b/r)^2\sin\theta$$

$$\tan\Psi = \omega(r-b)\sin\theta / v_m$$

### 물리적 흐름도 / Physical Flow Chart

```
태양 코로나 / Solar Corona
  T₀ ~ 10⁶ K, N₀ ~ 3×10⁷ cm⁻³
  │
  ├── 정역학 평형 시도? / Try hydrostatic equilibrium?
  │   → p(∞) = 10⁻⁵ dyne/cm² ≫ p_ISM = 10⁻¹³ dyne/cm²
  │   → 불가능! / IMPOSSIBLE!
  │
  ├── 열전도가 온도를 천천히 감소시킴
  │   T(r) ∝ r⁻²/⁷ (very slowly!)
  │   → 먼 거리에서도 "뜨거운" 가스 존재
  │
  ▼
정상 상태 팽창 / Stationary Expansion
  │
  ├── r < r_c: 아음속 (subsonic), 가속 중
  │   v < c_s, 열 에너지 > 중력 에너지의 감소
  │
  ├── r = r_c = λa/2: 임계점 (critical point)
  │   v = c_s (음속), trans-sonic 천이
  │   de Laval 노즐의 throat와 동일한 물리학
  │
  ├── r > r_c: 초음속 (supersonic), 계속 가속
  │   v > c_s, 무한대에서 v → ∞ (등온 근사)
  │
  ▼
1 AU에서의 태양풍 / Solar Wind at 1 AU
  v ~ 400-800 km/s, n ~ 5-10 cm⁻³
  │
  ├── Frozen-in: 자기장이 플라즈마와 함께 이동
  │
  ├── 태양 자전 (ω = 2.7×10⁻⁶ rad/s)
  │   + 방사 유출 (v_m)
  │   → Parker spiral: tan Ψ = ωr sinθ / v_m
  │
  ├── ~2.5 AU: B_ϕ = B_r (45° 나선각)
  │
  └── > 1 AU: plasma instability → tangled field shell
```

---

## Paper in the Arc of History / 역사 속의 논문

```
1908  Hale ──────────── 흑점 자기장 발견
      │
1931  Chapman & Ferraro ── 자기 폭풍과 태양 입자 구름
      │
1942  Alfvén ────────── MHD, frozen-in 조건
      │
1951  Biermann ─────── 혜성 꼬리 → 연속적 corpuscular radiation (관측 증거)
      │
1957  Chapman ─────── 정적 코로나 모델 (열전도로 확장)
      │                  Parker가 반박하는 대상
      │
      ▼
╔═══════════════════════════════════════════════════════════════════════╗
║  1958  PARKER — 태양풍 이론                                          ║
║         §II: 정역학 평형 불가능 → 팽창 필연                             ║
║         §III: 등온 코로나의 trans-sonic 팽창 해                        ║
║         §V: Parker spiral (행성간 자기장의 나선 구조)                    ║
║         "solar wind" 명명                                            ║
║         Dynamics of the Interplanetary Gas and Magnetic Fields       ║
╚═══════════════════════════════════════════════════════════════════════╝
      │
      ├── 1962  Mariner 2 ──── 태양풍 직접 관측 확인 (v ~ 400 km/s)
      │                        Parker의 예측 확인 → 논쟁 종결
      │
      ├── 1965  Ness ────────── Parker spiral 관측 확인 (IMP-1)
      │
      ├── 1967  Weber & Davis ── MHD 태양풍 모델 (자기장의 역효과 포함)
      │
      ├── 1977  Voyager 1 & 2 ── 행성간 공간 탐사
      │                          Parker spiral 정밀 확인
      │
      ├── 1990  Ulysses ────── 극 궤도에서 태양풍 구조 관측
      │                        Fast wind (극) vs Slow wind (적도)
      │
      ├── 1995  SOHO ────────── 코로나 가열 메커니즘 연구
      │
      └── 2018  Parker Solar Probe ── Parker의 이름을 딴 최초 NASA 탐사선
                                      태양에 역사상 가장 가까이 접근
                                      sub-Alfvénic 태양풍 최초 진입 (2021)
```

---

## Connections to Other Papers / 다른 논문과의 연결

| 논문 / Paper | 연결 / Connection |
|---|---|
| **Biermann (1951)** [SP #11] | **직접적 선행 논문**. Parker가 이론화하는 관측 증거의 원천. "It is the purpose of this paper to explore some of the grosser dynamic consequences of Biermann's conclusions" / Direct predecessor providing observational evidence that Parker formalizes |
| **Alfvén (1942)** [SP #8] | Frozen-in 조건 — Parker spiral 유도의 물리적 기반. 자기력선이 플라즈마와 함께 운동한다는 가정이 §V의 핵심 / Frozen-in condition is the physical basis for Parker spiral derivation |
| **Babcock (1961)** [SP #9] | 태양 쌍극 자기장의 구조 — Parker가 §V에서 유출하는 가스가 끌고 나가는 자기장의 원천 / Solar dipole field structure — the field carried outward by the solar wind |
| **Leighton (1969)** [SP #10] | Babcock-Leighton 다이나모 — 태양 자기 주기가 태양풍의 장기 변동을 지배 / Solar dynamo governs long-term solar wind variations |
| **Chapman & Ferraro (1931)** [SW #2] | 자기 폭풍과 태양 입자 — Parker가 간헐적 방출이 아닌 연속적 유출임을 이론적으로 보임 / Parker shows continuous outflow rather than intermittent emission |
| **Parker (1958)** [SW #4] | **같은 논문**, Space Weather 관점: 태양풍이 지구 자기권에 미치는 영향의 출발점 / Same paper from Space Weather perspective: starting point for magnetospheric coupling |
| **Ness (1965)** [SW #9] | Parker spiral의 **관측적 확인**. IMP-1 위성이 행성간 자기장의 나선 구조를 직접 측정 / **Observational confirmation** of Parker spiral by IMP-1 satellite |
| **Dungey (1961)** [SW #6] | 자기 재결합 — Parker spiral의 $B_z$ 성분이 Dungey의 열린 자기권 모델의 핵심 입력 / Magnetic reconnection — Parker spiral's $B_z$ component is key input for Dungey's open magnetosphere |

---

## References / 참고문헌

- Parker, E.N., "Dynamics of the Interplanetary Gas and Magnetic Fields," *The Astrophysical Journal*, 128, 664–676, 1958. [DOI: 10.1086/146579]
- Biermann, L., "Kometenschweife und solare Korpuskularstrahlung," *Zeitschrift für Astrophysik*, 29, 274, 1951.
- Chapman, S., "Notes on the Solar Corona and the Terrestrial Ionosphere," *Smithsonian Contributions to Astrophysics*, 2, 1, 1957.
- Schlüter, A., *Memo High Altitude Observatory*, 1954.
- van de Hulst, H.C., *The Sun*, ed. G.P. Kuiper (Chicago: University of Chicago Press), 1953.
- Meyer, P., Parker, E.N., & Simpson, J.A., "Solar Cosmic Rays of February 1956 and Their Propagation through Interplanetary Space," *Physical Review*, 104, 768, 1956.
- Ness, N.F., "The Earth's Magnetic Tail," *Journal of Geophysical Research*, 70, 2989, 1965.
- Weber, E.J. & Davis, L., "The Angular Momentum of the Solar Wind," *The Astrophysical Journal*, 148, 217, 1967.
