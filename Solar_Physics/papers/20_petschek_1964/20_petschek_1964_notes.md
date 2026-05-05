---
title: "Magnetic Field Annihilation"
authors: Harry E. Petschek
year: 1964
journal: "AAS-NASA Symposium on the Physics of Solar Flares, NASA SP-50, pp. 425-439"
doi: "1964NASSP..50..425P"
topic: Solar_Physics
tags: [reconnection, fast-reconnection, slow-shock, MHD, solar-flare, Petschek]
status: completed
date_started: 2026-04-19
date_completed: 2026-04-19
---

# 20. Magnetic Field Annihilation / 자기장 소멸

---

## 1. Core Contribution / 핵심 기여

### English
Petschek attacks the central embarrassment of Sweet-Parker reconnection: its rate is 10²–10⁴ times too slow to explain solar flares that release $10^{32}$ erg in $10^2$ s. He shows that **previous analyses overlooked standing magnetohydrodynamic waves as a mechanism for converting magnetic energy into plasma energy**. In Sweet-Parker's picture, finite conductivity diffusion has to do all the work, and the resulting current sheet must be as long as the entire flare region. Petschek instead treats the annihilation as a two-region problem: (1) a *tiny* central diffusion region of length $y^* \ll L$ where resistivity matters, and (2) a much larger *external flow* region where the magnetic energy is converted via **four standing slow-mode MHD waves (switch-off shocks in the compressible case)** propagating outward from the diffusion region at the Alfvén speed based on the normal field component. The switch-off shocks turn off the tangential component of $B$ across them, directly converting magnetic energy to heat and directed Alfvén-speed outflow. Using Laplace's equation for the external flow and a linearized matching at the boundary layer edge, Petschek derives the key result $M_{o,\mathrm{max}} = \pi/[4\ln(2 M_{o,\mathrm{max}}^2 R_m)]$ — a reconnection rate that depends only **logarithmically** on the magnetic Reynolds number. For typical solar corona $R_m \sim 10^{10}$, this gives $M_o \sim 0.01$–$0.1$, bringing annihilation times from Parker's $5 \times 10^4$ s down to $10^2$ s, matching observations.

### 한국어
Petschek은 Sweet-Parker 재결합의 치명적 약점—태양 플레어($10^{32}$ erg가 $10^2$ 초 안에 방출됨)에 비해 $10^2$–$10^4$배 느린 속도—을 정면으로 겨냥한다. 그의 핵심 통찰은 **이전 분석들이 자기 에너지를 플라즈마 에너지로 변환하는 메커니즘으로서 정상 자기유체역학파(standing MHD wave)를 간과했다**는 것이다. Sweet-Parker 그림에서는 유한 전도도 확산만이 모든 일을 하기 때문에, 전류층이 전체 플레어 영역만큼 길어야 한다. Petschek은 재결합 영역을 두 영역으로 나눈다: (1) 저항이 중요한 **매우 작은 중심 확산 영역** ($y^* \ll L$), (2) 훨씬 큰 **외부 흐름 영역**에서 중심에서 바깥으로 Alfvén 속도(수직 자기장 성분 기준)로 전파되는 **네 개의 정상 저속 모드 MHD 파(압축성 경우에는 switch-off 충격파)**에 의해 자기 에너지가 변환된다. Switch-off 충격파는 접선 자기장 성분을 꺼버리며, 자기 에너지를 직접 열 및 Alfvén 속도의 지향 흐름으로 변환한다. 외부 흐름에 대해 Laplace 방정식을, 경계층 가장자리에서 선형 매칭을 수행하여 Petschek은 핵심 결과 $M_{o,\mathrm{max}} = \pi/[4\ln(2 M_{o,\mathrm{max}}^2 R_m)]$을 유도한다. 이 속도는 자기 Reynolds 수에 **로그적으로만 의존**한다. 태양 코로나 $R_m \sim 10^{10}$에서 $M_o \sim 0.01$–$0.1$이 되어, 소멸 시간이 Parker의 $5 \times 10^4$ 초에서 $10^2$ 초로 줄어들어 관측과 일치한다.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction — The Slow-Rate Problem / 느린 속도 문제 (p. 425)

#### English
Petschek opens by surveying the state of the field. Sweet (1958) first proposed reconnection for flares; Parker (1957, 1963) performed a quantitative analysis. All such analyses predict release times 10–100× longer than observed. Even accounting for partially ionized plasma with ambipolar diffusion (Jaggi 1963) is insufficient. Furth, Killeen, and Rosenbluth (1963) showed that the Parker-Sweet configuration is **linearly unstable**, suggesting reconnection proceeds through turbulence (their References 6, 7). Petschek takes a different tack: he argues that Parker missed a distinct mechanism — **conversion of magnetic energy via wave propagation** rather than via resistive diffusion. Parker's solution is one special case; a different steady configuration with much faster rates is possible.

The introduction includes the famous **Figure 50-1** comparing the two conversion mechanisms. On the left (diffusion), the boundary thickness grows as $\delta = \sqrt{c^2 t/(4\pi\sigma)}$ — the ordinary skin depth, which decreases with increasing conductivity. On the right (wave propagation), the Alfvén waves propagate outward at $\delta = (B_n/\sqrt{4\pi\rho}) t$ — the Alfvén speed, which is **independent of conductivity**. In high-conductivity media (solar corona, where $\sigma$ is enormous), the wave mechanism must dominate at late times.

#### 한국어
Petschek은 분야의 현황을 개관하며 시작한다. Sweet(1958)가 플레어 재결합을 처음 제안했고, Parker(1957, 1963)가 정량 분석을 수행했다. 이 분석들은 관측보다 10–100배 긴 방출 시간을 예측한다. 부분 전리 플라즈마와 양극확산(Jaggi 1963)을 고려해도 충분치 않다. Furth, Killeen, Rosenbluth(1963)는 Parker-Sweet 배치가 **선형 불안정**함을 보였고, 재결합이 난류를 통해 진행된다는 제안으로 이어졌다(논문 내 참고문헌 6, 7). Petschek은 다른 접근을 취한다: Parker가 뚜렷한 한 가지 메커니즘—**저항 확산이 아닌 파동 전파를 통한 자기 에너지 변환**—을 놓쳤다고 주장한다. Parker의 해는 하나의 특수해일 뿐이며, 훨씬 빠른 속도를 주는 다른 정상 구성이 가능하다.

서론에는 두 가지 변환 메커니즘을 비교하는 유명한 **Figure 50-1**이 나온다. 왼쪽(확산)에서 경계층 두께는 $\delta = \sqrt{c^2 t/(4\pi\sigma)}$로 자라는데, 이는 일반 피부 깊이로 전도도가 커지면 감소한다. 오른쪽(파동 전파)에서 Alfvén 파는 $\delta = (B_n/\sqrt{4\pi\rho}) t$, 즉 Alfvén 속도로 바깥으로 전파되며, 이는 **전도도와 무관**하다. 고전도도 매질(태양 코로나, $\sigma$가 엄청남)에서는 후기 시간에 파동 메커니즘이 지배할 수밖에 없다.

---

### Part II: Diffusion Model — Sweet-Parker Recap / 확산 모델 (pp. 427–428)

#### English
Before presenting his new model, Petschek carefully rederives Sweet-Parker so he can compare apples to apples. The key geometry (Figure 50-2) has fluid moving in toward a central boundary of length $2L$ and thickness $2\delta$, with an X-type neutral point at center.

Mass conservation (Eq. 1):
$$u_{xo} L = v \delta$$

Bernoulli along the boundary (Eq. 2): $\rho v^2/2 = p - p_o$. Combined with pressure balance across the boundary (Eq. 3): $p - p_o = B_{yo}^2/(8\pi)$. Hence $v = B_{yo}/\sqrt{4\pi\rho} = V_A$ — the outflow is Alfvénic.

Ohm's law and Faraday's law give (Eq. 7):
$$u_{xo} = \frac{c^2}{4\pi\sigma\delta}$$

Combining Eqs. 1, 7, and using the Alfvén speed (Eq. 9), and defining $M_o = u_{xo}/V_A$ (Eq. 10) and $R_m = 4\pi\sigma V_A L/c^2$ (Eq. 11), one obtains (Eq. 12):
$$\boxed{M_o = R_m^{-1/2}}$$

For astronomical applications $R_m \gg 1$, and this gives a very slow annihilation rate — the Sweet-Parker result Petschek aims to surpass.

#### 한국어
새 모델을 제시하기 전에 Petschek은 동일한 조건에서 비교할 수 있도록 Sweet-Parker를 조심스럽게 재유도한다. 핵심 기하(Figure 50-2)는 중심에 X자 중성점을 가진 길이 $2L$, 두께 $2\delta$ 경계를 향해 유체가 유입되는 구조다.

질량 보존 (Eq. 1): $u_{xo} L = v \delta$.

경계 방향 Bernoulli 방정식 (Eq. 2): $\rho v^2/2 = p - p_o$. 경계 가로 압력 균형 (Eq. 3): $p - p_o = B_{yo}^2/(8\pi)$과 결합하면 $v = B_{yo}/\sqrt{4\pi\rho} = V_A$ — 유출은 Alfvén 속도다.

Ohm 법칙과 Faraday 법칙에서 (Eq. 7): $u_{xo} = c^2/(4\pi\sigma\delta)$.

Eq. 1, 7을 Alfvén 속도 (Eq. 9), $M_o = u_{xo}/V_A$ (Eq. 10), $R_m = 4\pi\sigma V_A L/c^2$ (Eq. 11)와 결합하면 (Eq. 12): $M_o = R_m^{-1/2}$.

천체물리 응용에서 $R_m \gg 1$이므로 매우 느린 소멸 속도를 준다 — Petschek이 넘어서려는 Sweet-Parker 결과다.

---

### Part III: Incompressible Model — The New Geometry / 비압축성 모델: 새로운 기하 (pp. 428–431)

#### English
Petschek now presents his modification. **Figure 50-3** shows the new geometry: instead of a single long current sheet of length $L$, the current is concentrated at the origin (diffusion region of length $2y^* \ll 2L$), and four "standing waves" (dark solid lines) fan outward in a V-shape. The light lines (external field) bend sharply across the heavy lines (standing waves). The boundary layer is sharply defined for $y > y^*$ by these waves; inside $y < y^*$, diffusion still rules.

A critical philosophical point: unlike Sweet-Parker, Petschek's analysis yields a **range of consistent steady solutions** parameterized by the far-upstream velocity $u_\infty = M_o V_A$. The actual rate is determined by external driving ("throttle placed elsewhere in the flow field, for example motions of the feet of the magnetic field lines in the solar surface"). Petschek computes the *maximum* allowed rate.

**Subdivision of the analysis** — three parts:
1. Boundary layer (inside the V)
2. External flow field (outside the V)
3. Maximum flow velocity for which a consistent pattern exists

**(i) Boundary layer** — Mass conservation analogous to Eq. 1 but locally (Eq. 13): $u_{xo} y = v \delta$. Momentum balance in y-direction (Eq. 14): $d/dy(\rho v^2 \delta) = -B_{yo} B_x/(4\pi)$. Combining (Eq. 15): $M_o^2 d/dy(y^2/\delta) = -b_x$ where $b_x = B_x/B_{yo}$.

For wave-dominated region ($y > y^*$), setting flow speed = wave propagation speed gives (Eq. 16): $M_o = |b_x|$. Constant $b_x$ along the boundary implies $b_x$ jumps sign discontinuously at the neutral point — which is "of course, unreasonable" — signaling that near the origin, *diffusion* must take over. From Eq. 15 with $M_o = |b_x|$: boundary layer thickness $\delta = M_o |y|$ (Eq. 17) — thickness grows linearly with $y$, giving the V-shape.

For diffusion-dominated region ($y < y^*$): (Eq. 18) $M_o = c^2/(4\pi\sigma V_A \delta)$, which with Eq. 15 yields (Eq. 19): $b_x = -(M_o^3 \cdot 8\pi\sigma V_A/c^2) y$ — $b_x$ varies linearly near origin. Matching the linear interior to the constant exterior at $y = y^*$ gives the **diffusion region length** (Eq. 20):
$$\boxed{y^* = \frac{c^2}{8\pi\sigma V_A M_o^2}}$$

Note Eqs. 20 and 12 are identical (up to a factor of 2) with $y^*$ and $L$ interchanged: Petschek's diffusion region is a *tiny Sweet-Parker sheet* nested at the heart of his geometry.

**(ii) External flow field** — Outside the boundary layer, $\mathbf{j}/\sigma$ is negligible, so Ohm's law reduces to the frozen-flux condition (Eq. 21): $\mathbf{E} + \mathbf{u} \times \mathbf{B}/c = 0$. Linearizing with $\mathbf{u} = u_{xo}(\hat{\mathbf{i}} + \mathbf{u}')$, $\mathbf{B} = B_{yo}(\hat{\mathbf{j}} + \mathbf{B}')$ (Eq. 22), taking curl twice, one finds that vorticity is conserved and zero at infinity, so both $\mathbf{B}'$ and $\mathbf{u}'$ are curl-free. Hence they satisfy **Laplace's equation**. The boundary layer defines the source: across the edge, $B_z' + d\delta/dy = b_x$ (Eq. 24). Using the known $b_x$ and $d\delta/dy$ (Eq. 17), the boundary values are (Eqs. 25, 26):
- Wave region ($y > y^*$): $B_x' = -2 M_o (y/|y|)$
- Diffusion region ($y < y^*$): $B_x' = -2 M_o (y/y^*)$

The external field is obtained via the dipole-like integral (Eq. 27):
$$\mathbf{B}'(\mathbf{r}) = \frac{1}{\pi} \int_{-L}^{L} \frac{B_x'(\eta)(\mathbf{r}-\boldsymbol{\eta})}{|\mathbf{r}-\boldsymbol{\eta}|^2} d\eta$$

Evaluating at the origin (Eq. 28):
$$B_y'(0) = -\frac{2 M_o}{\pi} \ln(L/y^*)$$

This is the central field perturbation "leaking" from the boundary layer back into the upstream flow. Here is Petschek's deepest physical insight: the perturbation grows **logarithmically** with $L/y^*$, and for the linear analysis to hold this perturbation must stay small.

**(iii) Maximum rate** — The linear analysis breaks down if $|B_y'(0)| \gtrsim 1$. Setting the limiting value somewhat arbitrarily to $B_y'(0) = -1/2$ (Eq. 29):
$$\boxed{M_{o,\mathrm{max}} = \frac{\pi}{4\ln(L/y^*)}} \quad \text{(Eq. 30)}$$

Using Eq. 20 for $y^*$ and eliminating:
$$\boxed{M_{o,\mathrm{max}} = \frac{\pi}{4\ln(2 M_{o,\mathrm{max}}^2 R_m)}} \quad \text{(Eq. 31)}$$

This is **the** Petschek formula. Because the dependence on $R_m$ is logarithmic, the result is remarkably insensitive to the poorly known conductivity — a "fortunate circumstance." For $R_m = 10^{10}$ (solar corona), iterating: $M_o \approx 0.03$–$0.05$, giving annihilation times ~ $10^2$–$10^3$ s, matching observations.

#### 한국어
Petschek은 이제 자신의 수정안을 제시한다. **Figure 50-3**이 새로운 기하를 보여준다: 길이 $L$의 단일 긴 전류층 대신 전류가 원점(길이 $2y^* \ll 2L$의 확산 영역)에 집중되고, 네 개의 "정상 파"(굵은 실선)가 V자 형태로 바깥으로 뻗어 나간다. 가는 선(외부 자기장)은 굵은 선(정상 파)을 가로지르며 급격히 꺾인다. 경계층은 $y > y^*$에서 이 파들에 의해 뚜렷이 정의되고, $y < y^*$에서는 여전히 확산이 지배한다.

중요한 철학적 차이: Sweet-Parker와 달리 Petschek의 분석은 **일관된 정상해의 범위**를 주며, 원거리 유입 속도 $u_\infty = M_o V_A$로 매개변수화된다. 실제 속도는 외부 구동("흐름장 다른 곳의 throttle, 예를 들어 태양 표면의 자기장 발 운동")에 의해 결정된다. Petschek은 *최대* 허용 속도를 계산한다.

**분석 세분화** — 세 부분:
1. 경계층 (V자 내부)
2. 외부 흐름장 (V자 외부)
3. 일관된 패턴이 존재하는 최대 흐름 속도

**(i) 경계층** — Eq. 1과 유사하지만 국소적인 질량 보존 (Eq. 13): $u_{xo} y = v \delta$. y 방향 운동량 균형 (Eq. 14): $d/dy(\rho v^2 \delta) = -B_{yo} B_x/(4\pi)$. 결합 (Eq. 15): $M_o^2 d/dy(y^2/\delta) = -b_x$, $b_x = B_x/B_{yo}$.

파 지배 영역 ($y > y^*$)에서 흐름 속도 = 파 전파 속도로 놓으면 (Eq. 16): $M_o = |b_x|$. 경계를 따라 상수 $b_x$는 중성점에서 부호가 불연속 점프함을 뜻하고, 이것은 "물론 비합리적"이며, 원점 근처에서 *확산*이 인수인계해야 함을 신호한다. Eq. 15에서 $M_o = |b_x|$를 대입하면 경계층 두께 $\delta = M_o |y|$ (Eq. 17) — 두께가 $y$에 선형 비례, V자 모양을 만든다.

확산 지배 영역 ($y < y^*$): (Eq. 18) $M_o = c^2/(4\pi\sigma V_A \delta)$, Eq. 15와 결합해 (Eq. 19) $b_x = -(M_o^3 \cdot 8\pi\sigma V_A/c^2) y$ — $b_x$가 원점 근처에서 선형 변화. 내부 선형부와 외부 상수부를 $y = y^*$에서 매칭하면 **확산 영역 길이** (Eq. 20): $y^* = c^2/(8\pi\sigma V_A M_o^2)$.

Eq. 20과 Eq. 12는 ($y^*$와 $L$을 맞바꾸면) 인수 2까지 동일하다. 즉, Petschek의 확산 영역은 그의 기하 중심에 둥지 튼 **작은 Sweet-Parker 전류층**이다.

**(ii) 외부 흐름장** — 경계층 바깥에서 $\mathbf{j}/\sigma$는 무시할 수 있어 Ohm 법칙이 freeze-in 조건 (Eq. 21): $\mathbf{E} + \mathbf{u} \times \mathbf{B}/c = 0$로 환원된다. $\mathbf{u} = u_{xo}(\hat{\mathbf{i}} + \mathbf{u}')$, $\mathbf{B} = B_{yo}(\hat{\mathbf{j}} + \mathbf{B}')$ (Eq. 22)로 선형화하고 curl을 두 번 취하면 소용돌이가 보존되며 무한원에서 0이므로 $\mathbf{B}'$과 $\mathbf{u}'$ 모두 curl-free, 따라서 **Laplace 방정식**을 만족한다. 경계층이 소스를 정의: 가장자리를 가로질러 $B_z' + d\delta/dy = b_x$ (Eq. 24). 알려진 $b_x$, $d\delta/dy$ (Eq. 17)를 사용해 경계값 (Eqs. 25, 26):
- 파 영역 ($y > y^*$): $B_x' = -2 M_o (y/|y|)$
- 확산 영역 ($y < y^*$): $B_x' = -2 M_o (y/y^*)$

외부장은 쌍극자 유사 적분 (Eq. 27)으로 얻는다: $\mathbf{B}'(\mathbf{r}) = (1/\pi)\int_{-L}^{L} B_x'(\eta)(\mathbf{r}-\boldsymbol{\eta})/|\mathbf{r}-\boldsymbol{\eta}|^2 \, d\eta$.

원점에서 평가 (Eq. 28): $B_y'(0) = -(2 M_o/\pi) \ln(L/y^*)$.

이것이 경계층에서 상류 흐름으로 "새어나온" 중심 자기장 섭동이다. Petschek의 가장 깊은 물리 통찰은 여기에 있다: 섭동이 $L/y^*$에 대해 **로그적으로** 자라며, 선형 분석이 성립하려면 이 섭동이 작아야 한다.

**(iii) 최대 속도** — 선형 분석은 $|B_y'(0)| \gtrsim 1$이 되면 깨진다. 한계값을 다소 임의로 $B_y'(0) = -1/2$로 (Eq. 29): $M_{o,\mathrm{max}} = \pi/[4\ln(L/y^*)]$ (Eq. 30). Eq. 20을 사용해 $y^*$ 소거: $M_{o,\mathrm{max}} = \pi/[4\ln(2 M_{o,\mathrm{max}}^2 R_m)]$ (Eq. 31).

이것이 **Petschek 공식**이다. $R_m$ 의존성이 로그적이므로, 결과는 잘 알려지지 않은 전도도에 놀라울 정도로 둔감하다 — "다행스러운 상황." $R_m = 10^{10}$(태양 코로나)에서 반복 대입: $M_o \approx 0.03$–$0.05$로, 소멸 시간 $\sim 10^2$–$10^3$ 초 — 관측과 일치.

---

### Part IV: Compressible Model — Switch-Off Shocks / 압축성 모델: Switch-off 충격파 (pp. 433–436)

#### English
In solar flares, gas pressure is small compared to magnetic pressure, so compressibility matters. Petschek argues the compressible geometry is nearly identical, with one crucial reinterpretation: the standing waves of the incompressible case become **switch-off shock waves** in the compressible case. A switch-off shock is defined as "a shock wave behind which the magnetic field is normal to the wave front (the tangential component is switched off across the shock)" — which is just a **slow-mode shock** in the limit of strong field. Since these waves make only a small angle with the y-axis, the field on the downstream side is essentially normal, matching the switch-off condition.

Critically, switch-off shocks propagate at $V_A$ based on the **normal** component of the field — the same speed as the Alfvén wave in the incompressible case. This is why the reconnection rates in the two cases are nearly identical.

The compressible analysis introduces the density ratio $\alpha = \rho_o/\rho_{bl}$ (Eq. 32) where $\rho_{bl}$ is density inside the boundary layer. For energy conservation in a monatomic gas at zero upstream pressure, $\alpha = 2/5$. With more degrees of freedom or radiative cooling, $\alpha < 2/5$.

Modified equations: mass conservation (13'): $\alpha u_{xo} y = v \delta$. Momentum (14'): $d/dy(\rho_o v^2 \delta) = -\alpha B_{yo} B_x/(4\pi)$. Boundary thickness (17'): $\delta = \alpha M_o |y|$. Diffusion length (20'): $y^* = c^2/(8\pi\sigma V_A \alpha M_o^2)$.

The maximum rate (31'):
$$M_{o,\mathrm{max}} = \frac{\pi}{2(1+\alpha) \ln(2 M_{o,\mathrm{max}}^2 \alpha R_m)}$$

Since $0 < \alpha < 1$, the difference from incompressible is at most a factor of 2 — the reconnection rate is remarkably robust across compressibility regimes.

#### 한국어
태양 플레어에서는 기체 압력이 자기 압력보다 작으므로 압축성이 중요하다. Petschek은 압축성 기하가 거의 동일하다고 주장하되, 한 가지 결정적 재해석을 한다: 비압축성의 정상 파동이 압축성에서는 **switch-off 충격파**가 된다. Switch-off 충격파는 "충격파 뒤에서 자기장이 파면에 수직한(접선 성분이 충격을 건너며 꺼지는) 충격파"로 정의되는데, 이는 강한 자기장 극한의 **저속 모드 충격파**에 해당한다. 이 파들이 y축과 작은 각만을 이루므로 하류 쪽 자기장은 본질적으로 수직이며 switch-off 조건과 일치한다.

중요한 점: switch-off 충격파는 자기장의 **수직** 성분 기준 $V_A$로 전파한다 — 비압축성의 Alfvén 파와 같은 속도다. 이 때문에 두 경우의 재결합 속도가 거의 동일하다.

압축성 분석에서 밀도 비 $\alpha = \rho_o/\rho_{bl}$ (Eq. 32)을 도입한다, $\rho_{bl}$은 경계층 내부 밀도. 영의 상류 압력의 단원자 기체에서 에너지 보존 시 $\alpha = 2/5$. 더 많은 자유도나 복사 냉각이 있으면 $\alpha < 2/5$.

수정 방정식: 질량 보존 (13') $\alpha u_{xo} y = v \delta$. 운동량 (14') $d/dy(\rho_o v^2 \delta) = -\alpha B_{yo} B_x/(4\pi)$. 경계 두께 (17') $\delta = \alpha M_o |y|$. 확산 길이 (20') $y^* = c^2/(8\pi\sigma V_A \alpha M_o^2)$.

최대 속도 (31'): $M_{o,\mathrm{max}} = \pi/[2(1+\alpha)\ln(2 M_{o,\mathrm{max}}^2 \alpha R_m)]$.

$0 < \alpha < 1$이므로 비압축성과의 차이는 인수 2 이내 — 재결합 속도는 압축성 체제에 대해 놀라울 정도로 강건하다.

---

### Part V: Application to Solar Flares / 태양 플레어 응용 (p. 436)

#### English
Petschek applies his formula to Parker's assumed flare conditions: $B = 500$ G, $L = 10^4$ km, $n = 2 \times 10^{11}$ cm⁻³. To release all the stored magnetic energy in the observed $10^2$–$10^3$ s:

**TABLE 50-1** (reproduced):

| Results From | Conductivity | Annihilation Time |
|---|---|---|
| Parker ($T = 10^4$ K, partial ionization) | fully ionized | $5 \times 10^4$ s |
| Parker ($T = 10^4$ K, ambipolar diffusion) | reduced | $6 \times 10^3$ s |
| **Present (Petschek)** ($T = 10^6$ K) | fully ionized | **$10^2$ s** |
| Observations | — | $10^2$–$10^3$ s |

The reduction from Parker's $5 \times 10^4$ s to Petschek's $10^2$ s — a factor of 500 — brings theory into agreement with observation for the first time.

Petschek notes a further subtlety: Eq. 18 gives boundary layer thickness at origin of $10^{-4}$ cm, which is *unphysical* because the current would require current densities exceeding what is possible within an electron gyroradius (~ 1 cm). Kinetic effects therefore reduce effective conductivity, thickening the diffusion region — but because the dependence is only logarithmic, this changes the annihilation time by just 30%.

**Coda on field geometry**: The analysis assumed exactly antiparallel fields. Adding a perpendicular (guide) field leaves the incompressible analysis unchanged (since such a field is uniform everywhere). In the compressible case, the perpendicular component is amplified inside the boundary layer, reducing the effective compressibility — but this matters little.

**Conclusion**: "the rate at which magnetic energy can be converted into plasma energy is sufficiently rapid to account for the observed times in solar flares. The rate at which energy can be released is therefore not a valid criticism of the suggestion that solar flares result from the release of magnetically stored energy."

#### 한국어
Petschek은 Parker가 가정한 플레어 조건에 자신의 공식을 적용한다: $B = 500$ G, $L = 10^4$ km, $n = 2 \times 10^{11}$ cm⁻³. 저장된 자기 에너지 전부를 관측된 $10^2$–$10^3$ 초 안에 방출하기 위해:

**TABLE 50-1** (재현):

| 결과 출처 | 전도도 | 소멸 시간 |
|---|---|---|
| Parker ($T = 10^4$ K, 부분 전리) | 완전 전리 | $5 \times 10^4$ s |
| Parker ($T = 10^4$ K, 양극확산) | 감소된 값 | $6 \times 10^3$ s |
| **Petschek** ($T = 10^6$ K) | 완전 전리 | **$10^2$ s** |
| 관측 | — | $10^2$–$10^3$ s |

Parker의 $5 \times 10^4$ 초에서 Petschek의 $10^2$ 초로의 감소 — 인수 500 — 은 처음으로 이론을 관측과 일치시켰다.

Petschek은 추가 섬세함을 지적한다: Eq. 18은 원점에서 경계층 두께 $10^{-4}$ cm를 주지만, 이는 *비물리적*이다. 요구되는 전류 밀도가 전자 회전 반지름(~1 cm) 안에서 흐를 수 있는 수준을 초과하기 때문. 따라서 운동론적 효과가 유효 전도도를 줄여 확산 영역을 두껍게 하지만, 의존성이 로그일 뿐이므로 소멸 시간은 30%만 변한다.

**자기장 기하에 대한 부기**: 분석은 정확히 반평행 자기장을 가정했다. 수직(가이드) 성분을 추가해도 비압축성 분석은 변하지 않는다(이 성분이 모든 곳에서 균일하므로). 압축성 경우에서 수직 성분은 경계층 내부에서 증폭되어 유효 압축성을 줄이지만 큰 영향은 없다.

**결론**: "자기 에너지가 플라즈마 에너지로 변환되는 속도는 태양 플레어에서 관측된 시간을 설명하기에 충분히 빠르다. 에너지가 방출될 수 있는 속도는 따라서 태양 플레어가 자기적으로 저장된 에너지의 방출로 발생한다는 제안에 대한 유효한 비판이 아니다."

---

### Part VI: Discussion Section — Peers Respond / 토론 섹션: 동료들의 반응 (pp. 437–439)

#### English
The published discussion after the talk gives rare insight into how the community received the idea.

- **Parker**: Asks for clarification that the reduction of *width* of the diffusion strip is what enhances diffusion rates. Petschek agrees: it's the reduction of both width *and* length that allows faster transit.
- **Meyer**: Skeptical — worries the V-shape is not as sharp as drawn, and that standing waves might actually travel inward and reduce to the old picture. Petschek defends: the fluid inside the boundary is escaping at $V_A$; waves propagate outward relative to the fluid but stand still in the lab frame because they are blown back by the flow.
- **Sweet** (the original!): Overwhelmingly supportive. "Your solution struck me at once as the solution for which we have been seeking." He raises only a technical worry about perpendicular field components, which Petschek addresses.
- **Wentzel**: Points out that stationary flow theories provide only one timescale, whereas flares have both rise and decay times. Petschek: the rise time is set by "the time it takes to flow through" = $L/V_A$, setting up the flow pattern.

#### 한국어
발표 후 공개 토론은 커뮤니티가 이 아이디어를 어떻게 받아들였는지 드문 통찰을 준다.

- **Parker**: 확산 스트립의 *폭* 감소가 확산 속도를 강화한다는 해석을 확인 요청. Petschek 동의: 폭과 길이 *둘 다*의 감소가 더 빠른 통과를 가능하게 함.
- **Meyer**: 회의적 — V자가 그려진 것만큼 선명하지 않을 것이고 정상 파가 실제로 내부로 이동해 옛 그림으로 환원될 것이라 우려. Petschek 반박: 경계 내부 유체는 $V_A$로 빠져나가며, 파는 유체 기준 바깥으로 전파되지만 실험실 틀에서는 흐름에 의해 뒤로 밀려 정지 상태에 있음.
- **Sweet** (원작자!): 압도적으로 지지. "당신의 해는 우리가 찾고 있던 해처럼 즉시 내게 다가왔다." 수직 자기장 성분에 대한 기술적 우려만 제기했고 Petschek이 답변.
- **Wentzel**: 정상 흐름 이론은 한 시간 척도만 제공하나, 플레어는 상승 시간과 감쇠 시간 둘 다 가진다고 지적. Petschek: 상승 시간은 "흐름이 통과하는 시간" = $L/V_A$로, 흐름 패턴을 설정하는 시간.

---

## 3. Key Takeaways / 핵심 시사점

1. **Two mechanisms, not one, convert magnetic to plasma energy / 자기 → 플라즈마 에너지 변환에는 두 메커니즘이 있다** — Sweet-Parker only used diffusion. Petschek realized that **wave propagation** (Alfvén waves; switch-off shocks in compressible case) is a distinct and, at high $R_m$, faster mechanism. Conductivity controls diffusion rate ($\sigma^{-1/2}$) but NOT wave rate (Alfvén speed is $\sigma$-independent). / Sweet-Parker는 확산만을 사용했다. Petschek은 **파동 전파**(Alfvén 파; 압축성에서 switch-off 충격파)가 뚜렷한 별개 메커니즘이며 고 $R_m$에서 더 빠르다는 것을 간파했다. 전도도는 확산 속도($\sigma^{-1/2}$)를 제어하지만 파동 속도는 제어하지 **않는다**(Alfvén 속도는 $\sigma$ 무관).

2. **The diffusion region shrinks, it does not vanish / 확산 영역은 작아지지만 사라지지 않는다** — Reconnection still needs resistivity at the very center (to violate the frozen-flux condition), but only in a *tiny* region $y^* \sim L / (M_o^2 R_m)$. Outside this, the MHD ideal description is fine and shocks do the energy conversion. The diffusion region is a "nested Sweet-Parker" inside the larger Petschek geometry. / 재결합은 여전히 중심에서 저항이 필요하지만(freeze-in 조건을 깨기 위해) 그 영역은 $y^* \sim L/(M_o^2 R_m)$로 매우 작다. 이 바깥에서는 MHD 이상 기술이 충분하고 충격파가 에너지를 변환한다. 확산 영역은 Petschek 기하 안에 둥지 튼 "내장된 Sweet-Parker"다.

3. **Consistent solutions exist for a range of rates / 일관된 해가 속도의 범위에서 존재** — Unlike Sweet-Parker's unique $M_o = R_m^{-1/2}$, Petschek's configuration admits a *family* of solutions parameterized by the driver. The actual rate is externally controlled, up to a **maximum** imposed by nonlinear breakdown at $|B_y'(0)| \sim 1$. This shifts reconnection from "intrinsic rate problem" to "driver problem." / Sweet-Parker의 유일한 $M_o = R_m^{-1/2}$와 달리 Petschek의 배치는 구동자로 매개변수화된 해의 *족(族)*을 허용한다. 실제 속도는 외부에서 제어되며 비선형 붕괴가 일어나는 $|B_y'(0)| \sim 1$에서 주어지는 **최대**까지 가능하다. 이는 재결합을 "내재 속도 문제"에서 "구동 문제"로 옮긴다.

4. **Logarithmic dependence on $R_m$ is the magic / $R_m$에 대한 로그 의존성이 마법** — $M_o^\mathrm{SP} = R_m^{-1/2}$ (power-law) versus $M_o^\mathrm{Pet} \sim 1/\ln R_m$ (logarithm). For $R_m = 10^{10}$: Sweet-Parker gives $10^{-5}$, Petschek gives $\sim 0.03$ — a factor of 3000. The logarithm also makes the answer insensitive to the exact value of $\sigma$ (which is genuinely uncertain in flare plasmas), a "fortunate circumstance." / Sweet-Parker는 거듭제곱, Petschek는 로그. $R_m = 10^{10}$: Sweet-Parker는 $10^{-5}$, Petschek는 $\sim 0.03$ — 인수 3000 차이. 로그 덕분에 답이 정확한 $\sigma$ 값(플레어 플라즈마에서 실제로 불확실)에 둔감해짐.

5. **Switch-off shocks are the workers / Switch-off 충격파가 일꾼** — Most of the magnetic energy is converted to heat + Alfvénic outflow *at the four slow-mode shocks*, not at the diffusion region. This is why reconnection signatures in solar/space observations manifest as shock-heated plasma and Alfvénic jets, not as steady resistive sheets. MMS and Cluster have directly detected such slow shocks in Earth's magnetotail. / 자기 에너지 대부분은 확산 영역이 아니라 *네 개의 저속 모드 충격파*에서 열 + Alfvén 유출로 변환된다. 이것이 태양·우주 관측에서 재결합 징표가 정상 저항층이 아닌 충격 가열 플라즈마와 Alfvén 제트로 나타나는 이유다. MMS와 Cluster 위성이 지자기 꼬리에서 이러한 저속 충격을 직접 관측했다.

6. **Solar flare problem resolved (at Petschek's level) / 태양 플레어 문제가 해결됨 (Petschek 수준에서)** — Table 50-1 shows Parker's $5 \times 10^4$ s compared to Petschek's $10^2$ s, against observed $10^2$–$10^3$ s. This is not merely a factor-of-two improvement; it's a qualitative shift that removed the primary objection to the magnetic-energy-release hypothesis for flares. This paper effectively established flares as reconnection events. / Table 50-1은 Parker의 $5 \times 10^4$ 초와 Petschek의 $10^2$ 초를 관측값 $10^2$–$10^3$ 초와 대비한다. 인수 2의 개선이 아니라 질적 전환이다 — 자기 에너지 방출 가설에 대한 주요 반대를 제거했다. 이 논문은 사실상 플레어를 재결합 사건으로 확립했다.

7. **Compressibility is a factor-of-two effect / 압축성은 인수 2 정도 효과** — Eq. 31' vs. 31 differ only by $(1+\alpha)$ in the prefactor. This robustness is crucial: flare plasma is highly compressible ($\beta \ll 1$) yet Petschek's conclusion carries over. The reason is that switch-off shocks in compressible flow travel at the same speed (Alfvén speed based on $B_n$) as the Alfvén waves in incompressible flow. / Eq. 31'와 31은 $(1+\alpha)$ 인수만큼만 차이. 이 강건성이 결정적: 플레어 플라즈마는 고압축성($\beta \ll 1$)이지만 Petschek의 결론이 그대로 이어진다. 이유: 압축성에서 switch-off 충격파는 비압축성의 Alfvén 파와 같은 속도($B_n$ 기준 Alfvén 속도)로 전파.

8. **What Petschek left unresolved / Petschek가 미해결로 남긴 것** — (a) The internal structure of the diffusion region is a black box. (b) The linearization at the boundary edge was not rigorously justified, and Eq. 29 ($B_y'(0) = -1/2$) is somewhat arbitrary. (c) Subsequent numerical MHD simulations (Biskamp 1986) found Petschek's geometry is unstable under uniform $\eta$ and collapses to Sweet-Parker — restored only under anomalous or locally enhanced resistivity. Modern understanding: the Petschek-like *fast* rate is recovered in nature via either (i) kinetic/Hall physics in thin sheets, or (ii) plasmoid instability tearing Sweet-Parker sheets into many X-points. The *mechanism* Petschek proposed is partly wrong, but the *rate* he predicted is right. / (a) 확산 영역 내부 구조는 블랙박스. (b) 경계 가장자리 선형화는 엄밀히 정당화되지 않았고, Eq. 29($B_y'(0) = -1/2$)는 다소 임의. (c) 후속 수치 MHD 시뮬레이션(Biskamp 1986)은 균일 $\eta$에서 Petschek 기하가 불안정하고 Sweet-Parker로 붕괴함을 발견 — 이상 저항이나 국소 증강 저항에서만 복원. 현대 이해: Petschek-유사 *빠른* 속도는 (i) 얇은 전류층의 운동론/Hall 물리, (ii) Sweet-Parker 전류층을 여러 X점으로 찢는 플라즈모이드 불안정성을 통해 자연에서 복원. Petschek이 제안한 *메커니즘*은 부분적으로 틀렸지만 예측한 *속도*는 맞다.

---

## 4. Mathematical Summary / 수학적 요약

### Sweet-Parker Baseline (for comparison) / Sweet-Parker 기준

$$
\underbrace{u_{xo} L = v\delta}_{\text{mass}}, \quad
\underbrace{v = V_A}_{\text{Bernoulli}}, \quad
\underbrace{u_{xo} = \frac{c^2}{4\pi\sigma\delta}}_{\text{Ohm/Faraday}}
\;\Rightarrow\;
\boxed{M_o = R_m^{-1/2}} \quad (\text{Eq. 12})
$$

For $R_m = 10^{10}$: $M_o \sim 10^{-5}$, annihilation time $L/u_{xo} \sim 5 \times 10^4$ s (too slow for flares).

### Petschek Geometry: Boundary Layer / 경계층

Mass conservation in the thin layer (Eq. 13):
$$u_{xo} y = v\delta$$

Momentum balance across $y$ (Eq. 14):
$$\frac{d}{dy}(\rho v^2 \delta) = -\frac{B_{yo} B_x}{4\pi}$$

Combined (Eq. 15, with $b_x = B_x/B_{yo}$, $M_o = u_{xo}/V_A$):
$$M_o^2 \frac{d}{dy}\!\left(\frac{y^2}{\delta}\right) = -b_x$$

**Wave-dominated region ($y > y^*$)**: flow speed equals wave speed $\Rightarrow M_o = |b_x|$ (Eq. 16), yielding

$$\boxed{\delta = M_o |y|} \quad (\text{Eq. 17})$$

— the V-shape of the boundary layer.

**Diffusion-dominated region ($y < y^*$)**: diffusion velocity (Eq. 18): $M_o = c^2/(4\pi\sigma V_A \delta)$, leading to

$$b_x = -\frac{M_o^3 \cdot 8\pi\sigma V_A}{c^2} y \quad (\text{Eq. 19})$$

Matching at $y = y^*$:

$$\boxed{y^* = \frac{c^2}{8\pi\sigma V_A M_o^2} = \frac{L}{2 M_o^2 R_m}} \quad (\text{Eq. 20})$$

### Petschek Geometry: External Flow / 외부 흐름

Linearization: $\mathbf{u} = u_{xo}(\hat{\mathbf{i}} + \mathbf{u}')$, $\mathbf{B} = B_{yo}(\hat{\mathbf{j}} + \mathbf{B}')$. Frozen-flux + double-curl: $\nabla \times \mathbf{B}' = 0$, $\nabla \times \mathbf{u}' = 0$, hence both satisfy Laplace's equation.

Boundary source (Eq. 24): $B_z' + d\delta/dy = b_x$. Evaluating with Eq. 17:
$$B_x' = -2 M_o \cdot \mathrm{sign}(y) \; (y>y^*); \quad B_x' = -2 M_o (y/y^*) \; (y<y^*)$$

Dipole-like integral solution (Eq. 27):
$$\mathbf{B}'(\mathbf{r}) = \frac{1}{\pi} \int_{-L}^{L} \frac{B_x'(\eta)(\mathbf{r} - \boldsymbol{\eta})}{|\mathbf{r} - \boldsymbol{\eta}|^2} \, d\eta$$

Center-line perturbation (Eq. 28):
$$\boxed{B_y'(0) = -\frac{2 M_o}{\pi} \ln(L/y^*)}$$

### Maximum Annihilation Rate / 최대 소멸 속도

Linearization fails when $|B_y'(0)| \gtrsim 1$. Choosing $B_y'(0) = -1/2$ (Eq. 29):

$$\boxed{M_{o,\mathrm{max}} = \frac{\pi}{4 \ln(L/y^*)} = \frac{\pi}{4 \ln(2 M_{o,\mathrm{max}}^2 R_m)}} \quad (\text{Eqs. 30, 31})$$

This is transcendental but iterable. Worked example, $R_m = 10^{10}$:
- Initial guess $M_o = 0.1$: RHS = $\pi/[4 \ln(2 \cdot 0.01 \cdot 10^{10})] = \pi/[4 \ln(2 \times 10^8)] \approx \pi/(4 \cdot 19.1) \approx 0.041$.
- Second iteration $M_o = 0.041$: RHS = $\pi/[4 \ln(2 \cdot 0.00168 \cdot 10^{10})] \approx \pi/(4 \cdot 17.2) \approx 0.046$.
- Converges near $M_o \approx 0.045$.

Annihilation time: $\tau = L/u_{xo} = L/(M_o V_A) \approx (10^9 \,\mathrm{cm})/(0.045 \cdot 10^8 \,\mathrm{cm/s}) \approx 220 \,\mathrm{s}$.

### Compressible Correction / 압축성 보정

Introducing $\alpha = \rho_o/\rho_{bl}$ (Eq. 32; $\alpha = 2/5$ for monatomic perfect gas, energy conserved):

$$M_{o,\mathrm{max}} = \frac{\pi}{2(1+\alpha) \ln(2 M_{o,\mathrm{max}}^2 \alpha R_m)} \quad (\text{Eq. 31'})$$

For $\alpha = 2/5$: prefactor becomes $\pi/2.8$ vs. incompressible $\pi/4$ — incompressible gives larger rate. The difference is **less than a factor of 2**.

### Comparison Table / 비교표

| Quantity / 양 | Sweet-Parker | Petschek (incompressible) |
|---|---|---|
| Current-sheet length | $L$ (global) | $y^* = L/(2 M_o^2 R_m)$ (tiny) |
| Sheet thickness | $\delta = L R_m^{-1/2}$ | $\delta = M_o y$ (linear) |
| Inflow Mach $M_o$ | $R_m^{-1/2}$ | $\pi/[4 \ln(2 M_o^2 R_m)]$ |
| $R_m = 10^{10}$ | $10^{-5}$ | $\sim 0.045$ |
| Flare time for $L = 10^4$ km, $V_A = 10^3$ km/s | $10^7$ s | $220$ s |

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
 1946     1956/58       1957     1962      1963       1964         1975            1986            2001 ──── 2009 ──── 2020s
  │         │             │        │         │          │            │               │              │         │          │
Giovanelli Sweet       Parker  Dungey   Furth-       PETSCHEK      Vasyliunas    Biskamp:       Shibata  Bhattacharjee  MMS mission
flare X-   steady-     computes open      Killeen-     (THIS)        improves    Petschek     flare       plasmoid       directly
point      state recon slow     magneto-  Rosenbluth   fast-recon    internal    unstable     observations instability    observes
[concept   (Sweet-     rate of  sphere    tearing                    structure   under        as reconn.  revives         Petschek
of recon]  Parker      SP        [1st     mode                       of diffusion uniform η;   (Masuda    fast recon      slow shocks
           model       model     recon in instability                 region      recovers      cusp at   at Sweet-       at Earth's
           estab-      [slow]    space]  [another                                SP geom.     flare top] Parker rate    magnetopause/
           lished]                        problem                                                                         magnetotail
                                          with SP]
```

### 한국어 타임라인 설명
- **1946 Giovanelli**: X자 중성점에서 플레어 에너지 방출 개념.
- **1956/58 Sweet-Parker**: 정상상태 재결합 모델, 그러나 속도가 너무 느림.
- **1957 Parker**: 느린 속도를 정량화.
- **1962 Dungey**: 지자기권 재결합 제안.
- **1963 Furth-Killeen-Rosenbluth**: Sweet-Parker가 tearing mode 불안정.
- **1964 Petschek (본 논문)**: Switch-off 충격파를 도입한 fast reconnection 모델.
- **1975 Vasyliunas**: Petschek 확산 영역 내부 구조 정교화.
- **1986 Biskamp**: 균일 $\eta$에서 Petschek 기하가 불안정해 Sweet-Parker로 붕괴한다는 수치 실험.
- **2001 Shibata et al.**: Yohkoh 위성으로 플레어 정상부 재결합 구조 관측.
- **2009 Bhattacharjee et al.**: Plasmoid instability로 Sweet-Parker가 빠른 재결합을 복원.
- **2020s MMS**: 자기권계면·꼬리에서 Petschek 형 저속 충격파 직접 관측.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **Sweet (1958) / Parker (1957) — paper #19** | Direct target. Petschek's paper is explicitly framed as fixing Sweet-Parker's too-slow rate. Equation 12 is their result; the rest of the paper departs from it. / 직접적 대상. Petschek 논문은 Sweet-Parker의 느린 속도를 고치는 것으로 명시적 위치 설정. Eq. 12가 그들의 결과. | **Foundational** — reading paper #19 first is required. / 필수 선행 독서. |
| **Furth, Killeen, Rosenbluth (1963)** | FKR showed Sweet-Parker is tearing-mode unstable, motivating the search for alternatives. Petschek cites them as a separate line (turbulence-based) complementary to his laminar fast-reconnection solution. / FKR은 Sweet-Parker가 tearing 모드 불안정함을 보였고, 대안 탐색을 동기부여. Petschek은 그들을 난류 기반의 별도 노선으로 인용. | **Parallel track** — both criticize SP but reach different conclusions. / 둘 다 SP를 비판하지만 다른 결론. |
| **Sherleif (1960)** | Cited as the standard reference for MHD shocks, used to justify the switch-off shock identification. / MHD 충격파의 표준 참고문헌으로 인용, switch-off 충격파 식별의 근거. | **Technical basis** for the compressible analysis. / 압축성 분석의 기술적 기반. |
| **Vasyliunas (1975)** | First major refinement of Petschek: resolves the internal structure of the diffusion region (Petschek left it a black box). / Petschek의 첫 주요 정교화: 확산 영역 내부 구조(Petschek이 블랙박스로 남긴)를 해결. | **Direct successor** — cited in nearly every reconnection paper since. / 거의 모든 후속 재결합 논문에서 인용. |
| **Biskamp (1986)** | Numerical MHD simulations found Petschek's geometry *unstable* under uniform resistivity, collapsing back to Sweet-Parker. Upended the paper's direct applicability. / 수치 MHD 시뮬레이션으로 Petschek 기하가 균일 저항에서 *불안정*하고 Sweet-Parker로 붕괴함을 발견. 논문의 직접 적용 가능성에 도전. | **Major caveat** — forces a distinction between Petschek's *rate* and *mechanism*. / Petschek의 *속도*와 *메커니즘*의 구별 강요. |
| **Bhattacharjee et al. (2009), Uzdensky et al. (2010)** | Plasmoid instability: Sweet-Parker sheets above critical length rip apart into chains of plasmoids, restoring a fast rate $\sim 0.01 V_A$ — close to Petschek's number but by a different mechanism. / 플라즈모이드 불안정성: 임계 길이 이상의 Sweet-Parker 전류층이 plasmoid 사슬로 찢어져 빠른 속도 $\sim 0.01 V_A$ 복원 — Petschek 수치에 가깝지만 다른 메커니즘. | **Vindication of the conclusion**: Petschek's rate is right even if his steady geometry isn't. / 결론의 정당화: Petschek의 속도는 정상 기하가 아니어도 옳다. |
| **Shibata et al. (1995, 2001) / Masuda cusp** | Yohkoh X-ray observations of flare loop-top hard X-ray sources → direct observational support for a reconnection outflow jet terminating at a looptop shock. The Petschek outflow geometry is the simplest picture. / 요코 X선 플레어 루프 꼭대기 경단 X선 관측 → 루프 꼭대기 충격에서 끝나는 재결합 유출 제트의 직접 관측 지지. Petschek 유출 기하가 가장 단순한 그림. | **Observational link** to solar flares. / 태양 플레어 관측 연결. |
| **MMS mission (2015–)** | First direct in-situ detection of slow-mode (Petschek-like) shocks in the magnetotail and at the magnetopause. Validates the shock picture even if the geometric details differ from idealized Petschek. / 지자기 꼬리와 자기권계면에서 저속 모드(Petschek형) 충격의 최초 직접 현장 관측. 이상화된 Petschek과 기하 세부는 다르지만 충격 그림을 검증. | **Experimental confirmation** in space plasma. / 우주 플라즈마에서의 실험적 확인. |

---

## 7. References / 참고문헌

- Petschek, H. E., "Magnetic Field Annihilation," in *AAS-NASA Symposium on the Physics of Solar Flares*, NASA SP-50, ed. W. N. Hess, pp. 425–439, 1964. [ADS: 1964NASSP..50..425P]
- Sweet, P. A., "The Neutral Point Theory of Solar Flares," in *Electromagnetic Phenomena in Cosmical Physics*, IAU Symposium No. 6, p. 123, 1958.
- Parker, E. N., "Sweet's Mechanism for Merging Magnetic Fields in Conducting Fluids," *J. Geophys. Res.* **62**, 509, 1957.
- Parker, E. N., *Astrophys. J. Suppl.* Ser. X, 8:177, 1963.
- Furth, H. P., Killeen, J., Rosenbluth, M. N., "Finite-Resistivity Instabilities of a Sheet Pinch," *Phys. Fluids* **6**, 459, 1963.
- Jaggi, R. K., "Flare Energy Dissipation by Ambipolar Diffusion," *J. Geophys. Res.* **68**, 4429, 1963.
- Shercliff, J. A., *J. Fluid Mech.* **9**, 481, 1960.
- Vasyliunas, V. M., "Theoretical Models of Magnetic Field Line Merging," *Rev. Geophys. Space Phys.* **13**, 303, 1975.
- Biskamp, D., "Magnetic Reconnection via Current Sheets," *Phys. Fluids* **29**, 1520, 1986.
- Bhattacharjee, A. et al., "Fast Reconnection in High-Lundquist-Number Plasmas due to the Plasmoid Instability," *Phys. Plasmas* **16**, 112102, 2009.
- Uzdensky, D. A. et al., "Fast Magnetic Reconnection in the Plasmoid-Dominated Regime," *Phys. Rev. Lett.* **105**, 235002, 2010.
