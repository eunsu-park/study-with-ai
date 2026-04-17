---
title: "Sweet's Mechanism for Merging Magnetic Fields in Conducting Fluids"
authors: Eugene N. Parker
year: 1957
journal: "Journal of Geophysical Research"
doi: "10.1029/JZ062i004p00509"
topic: Solar_Physics
tags: [magnetic reconnection, Sweet-Parker model, current sheet, MHD, solar flares, magnetic merging, resistive diffusion]
status: completed
date_started: 2026-04-17
date_completed: 2026-04-17
---

# 19. Sweet's Mechanism for Merging Magnetic Fields in Conducting Fluids / 전도성 유체에서 자기장 병합을 위한 스윗의 메커니즘

---

## 1. Core Contribution / 핵심 기여

Eugene Parker는 1956년 IAU 심포지엄에서 Peter Sweet가 정성적으로 제안한 반평행 자기장 병합(merging) 아이디어를 최초로 정량적인 수학적 모델로 발전시켰다. Parker는 두 개의 반평행(oppositely directed) 자기장이 외부 힘에 의해 서로 밀릴 때, 그 사이에 형성되는 매우 얇은 전이층(transition layer)에서 자기 확산(resistive diffusion)이 급격히 일어나 자기장선이 재결합(reconnect)할 수 있음을 보였다. 핵심 결과는 병합 속도 $v_m$이 특성 길이 $L$, 전도도 $\sigma$, 수자기 속도(hydromagnetic velocity) $C$에 대해 $v_m \approx 0.26\, c(C/\sigma L)^{1/2}$ 로 주어진다는 것이다. 이 속도는 순수한 옴 확산 속도 $c(c/\sigma L)$보다 훨씬 빠르지만, 태양 코로나에서 관측되는 플레어의 시간 척도를 설명하기에는 여전히 "너무 느리다." 이 "느린 재결합 문제(slow reconnection problem)"의 발견은 이후 Petschek(1964)의 빠른 재결합 모델과 현대 plasmoid instability 연구의 직접적 동기가 되었으며, Sweet-Parker 모델은 오늘날까지 모든 자기 재결합 이론의 기준점으로 남아 있다.

Eugene Parker provided the first quantitative mathematical formalization of Peter Sweet's qualitative idea (proposed at the 1956 IAU Symposium) for the merging of antiparallel magnetic fields. Parker demonstrated that when two oppositely directed magnetic fields are pressed together by external forces, a very thin transition layer forms between them where resistive diffusion becomes rapid enough for field lines to reconnect. The key result is that the merging velocity scales as $v_m \approx 0.26\, c(C/\sigma L)^{1/2}$, where $L$ is the characteristic length, $\sigma$ the electrical conductivity, $C$ the hydromagnetic (Alfvén) velocity, and $c$ the speed of light (CGS). This velocity is much faster than pure ohmic diffusion ($\sim c^2/\sigma L$) but still far too slow to account for observed solar flare timescales in the corona. The discovery of this "slow reconnection problem" directly motivated Petschek's (1964) fast reconnection model and modern plasmoid instability research. The Sweet-Parker model remains the benchmark against which all magnetic reconnection theories are measured to this day.

---

## 2. Reading Notes / 읽기 노트

### Section I: Introduction / 서론 (pp. 509–511)

Parker는 Sweet(1956)의 핵심 관찰에서 출발한다: 규모 $L$의 반평행 자기장이 전도성 매질에서 서로 밀리면, 일반적인 확산 시간 $L^2\sigma/c^2$에 비해 매우 짧은 시간에 상호확산(interdiffusion)이 일어날 수 있는 상황이 발생한다.

Parker begins with Sweet's (1956) key observation: when oppositely directed magnetic fields of scale $L$ in a highly conducting medium are shoved against each other, an interesting situation arises where interdiffusion can occur in times far shorter than the usual diffusion time $L^2\sigma/c^2$.

**차원 분석 (Dimensional Analysis)**:

- 흑점 자기장의 일반적인 확산 시간: $L \cong 10^9$ cm인 경우, 온도 $10^4$ K에서 $\sigma \cong 1.8 \times 10^{13}$ esu이면, 확산 시간 $\sim 2 \times 10^{10}$ 초 ≈ 600년
  Normal diffusion time for a sunspot field: with $L \cong 10^9$ cm, $T = 10^4$ K, $\sigma \cong 1.8 \times 10^{13}$ esu, the decay time is $\sim 2 \times 10^{10}$ sec ≈ 600 years
- Sweet의 메커니즘으로는 약 2주 만에 병합이 가능 — **약 $10^4$배 빠르다**
  With Sweet's mechanism, merging could occur in about two weeks — **roughly $10^4$ times faster**

이 극적인 차이의 핵심은 반평행 자기장이 밀릴 때 자기장선 사이에 형성되는 **중성면(neutral surface)**에 있다. 이 면에서 자기장이 0이 되고, 자기장 기울기가 무한히 커져 확산항이 중요해진다.

The key to this dramatic difference lies in the **neutral surface** that forms between antiparallel field lines when they are pressed together. On this surface the field vanishes and the field gradient becomes so steep that the diffusion term becomes important.

**물리적 과정 (Physical Process — Fig. 1)**:

Parker는 Figure 1에서 태양 광구면에 있는 두 쌍극 흑점군의 병합 과정을 3단계로 보여준다:

Parker illustrates in Figure 1 the merging process for two bipolar sunspot groups on the solar photosphere in three stages:

1. **(a)** 두 쌍극 흑점군이 같은 태양 위도에서 서로 멀리 떨어져 있음 — 자기장선이 서로 연결되지 않음
   Two bipolar sunspot groups widely separated at the same solar latitude — field lines not interconnected

2. **(b)** 광구면 아래의 물질 운동이 두 흑점군을 서로 밀어넣음 — 자기장이 왜곡되고, 중성면에서 기울기가 증가하며, 전도성이 높아 자기장선 연결이 방해됨
   Dense gas motions beneath the photosphere shove the groups together — fields distort, gradient across neutral plane increases, high conductivity prevents interconnection

3. **(c)** 약 1주일 후 급격한 상호확산이 일어나 자기장선이 재결합됨
   After about a week, rapid interdiffusion occurs and field lines reconnect

**유도 방정식 (Induction Equation)**:

기본 수자기 방정식은 다음과 같다 (CGS 단위):

The basic hydromagnetic equation in CGS units:

$$\frac{\partial \mathbf{B}}{\partial t} = \nabla \times (\mathbf{v} \times \mathbf{B}) \quad \cdots (1)$$

전도도가 아무리 크더라도, 중성면에서의 자기장 기울기가 무한히 증가하면 확산항 $(c^2/4\pi\mu\sigma)\nabla^2 \mathbf{B}$가 대류항 $\nabla \times (\mathbf{v} \times \mathbf{B})$에 필적하게 된다. 이때 자기장선 사이의 유체가 빠져나가면서 두 반평행 자기장이 더 가까이 접근하고, 상호확산이 빠르게 일어난다.

No matter how large the conductivity, if the field gradient across the neutral surface increases without limit, the diffusion term $(c^2/4\pi\mu\sigma)\nabla^2 \mathbf{B}$ becomes comparable to the dynamical term $\nabla \times (\mathbf{v} \times \mathbf{B})$. The efflux of fluid from between the fields allows them to approach and interdiffuse rapidly.

**핵심 스케일링 (Key Scaling from Introduction)**:

- 전이층 두께 $l$에서의 확산 시간: $l^2\sigma/c^2$
  Diffusion time across transition layer of thickness $l$: $l^2\sigma/c^2$
- 병합 속도: $u \cong c^2/l\sigma$
  Merging velocity: $u \cong c^2/l\sigma$
- 유출 속도: $v \cong uL/l$ (질량 보존)
  Outflow velocity: $v \cong uL/l$ (mass conservation)
- 자기 압력에 의한 유출 가속: $\frac{1}{2}\rho v^2 \cong B^2/8\pi$ → $v \cong C_0$ (수자기 속도)
  Magnetic pressure drives outflow: $\frac{1}{2}\rho v^2 \cong B^2/8\pi$ → $v \cong C_0$ (hydromagnetic velocity)
- 따라서 / Therefore: $u \cong c(C_0/L\sigma)^{1/2}$, $l/L \cong c/(C_0 L \sigma)^{1/2}$

**수치 예시 (Numerical Example)**:

$B = 1000$ gauss, $L \cong 10^9$ cm, $\sigma \cong 1.8 \times 10^{13}$ esu, $\rho = 10^{-8}$ g/cm³일 때:

- $C_0 \cong 100$ km/sec (Alfvén 속도)
- $u \cong 7$ m/sec (병합 속도)
- $l/L \cong 0.7 \times 10^{-4}$ — 전이층이 **전체 규모의 $10^{-4}$배**로 매우 얇음
  The transition layer is **$10^{-4}$ times the total scale** — extremely thin

---

### Section II: Expulsion of Fluid / 유체의 방출 (pp. 512–514)

Parker는 Sweet의 메커니즘에 관련된 물리적 과정을 이해하기 위해, 두 자기장 사이에 갇힌 유체가 어떻게 짜내져(squeezed out) 나가는지를 분석한다.

Parker analyzes the physical process of how fluid caught between two magnetic fields is squeezed out, to understand the mechanism underlying Sweet's idea.

**모델 설정 (Model Setup — Fig. 2)**:

- 두 개의 초전도 판(infinitely conducting sheets)이 $x = \pm \epsilon$에 위치
  Two infinitely conducting sheets at $x = \pm \epsilon$
- 이 판들 사이에 전도성 비점성 유체의 얇은 층이 채워져 있음
  A thin layer of conducting inviscid fluid fills the space between them
- $x = \pm a$ ($a \gg \epsilon$) 너머에는 전도성 유체
  Conducting fluid beyond $x = \pm a$

자기장 $\mathbf{B}$는 $xy$-평면에 평행하며, 스칼라 포텐셜 $\psi$의 기울기로 표현됨:

$$\mathbf{B} = -\nabla\psi \quad \cdots (2)$$

**자기장 포텐셜 (Magnetic Field Potential)**:

초전도 판의 경계 조건을 적용하면:

$$\psi(x, y) = \pm \frac{B_0 b^2}{4\sqrt{\pi}} \int_{-\infty}^{+\infty} \frac{dk \sin ky}{\sinh ka} \cosh k(x \pm \epsilon) \exp(-k^2 b^2/4) \quad \cdots (3)$$

여기서 $\pm$는 $-a < x < -\epsilon$ (양), $+\epsilon < x < +a$ (음)에 해당한다. 경계에서의 자기장 밀도:

$$B_x(\pm\epsilon, y) = \mp \frac{B_0 b^2}{4\sqrt{\pi}} \int_{-\infty}^{+\infty} \frac{dk\, k \cos ky}{\sinh ka} \exp(-k^2 b^2/4)$$

$a \ll b$이고 $\sinh ka \approx ka$로 근사하면:

$$B_x(\pm\epsilon, y) \sim (B_0 b/2a^2) \exp(-y^2/b^2) \times \{1 - \tfrac{1}{3}(a^2/b^2)(1-2y^2/b^2) + O^4(a/b)\} \quad \cdots (4)$$

**유체 운동 (Fluid Motion)**:

초전도 판에 작용하는 자기 압력은 $p = B_x^2(\pm\epsilon, y)/8\pi$이다. 이 압력이 유체를 $y$-방향으로 가속시킨다:

$$\frac{d^2 Y}{dt^2} = -\frac{1}{\rho}\frac{\partial p}{\partial y}$$

에너지 보존을 적용하고, 비압축성 유체($\rho$ = const)를 가정하면:

$$\frac{dY(t)}{dt} = \frac{C_0 b}{2a} \left\{ \exp\left[-\frac{2Y^2(0)}{b^2}\right] - \exp\left[-\frac{2Y^2(t)}{b^2}\right] \right\}^{1/2} \left\{1 + O^2(a/b)\right\}$$

여기서 $C_0 = B_0/(4\pi\rho)^{1/2}$은 수자기 속도이다.

$C_0 t/a$의 급수 전개로 위치를 계산하면 (Eq. 5):

$$Y(t) = Y(0)\left\{1 + \left(\frac{C_0 t}{a}\right)^2 \exp\left[-\frac{2Y^2(0)}{b^2}\right] + \cdots\right\}$$

**유출 속도 (Outflow Velocity — Eq. 8)**:

$v_y = dY(t)/dt$, $y = Y(t)$로 쓰면:

$$v_y = 2C_0(y/a)(C_0 t/a) \exp(-2y^2/b^2) + O^3(C_0 t/a) \quad \cdots (8)$$

**전류 시트 두께의 진화 (Evolution of Layer Thickness — Eq. 10)**:

$l(y, t)$를 전이층의 $x$-방향 폭이라 하면, 연속 방정식 적분에서:

$$\frac{\partial l}{\partial t} + l \frac{\partial v_y}{\partial y} = 0 \quad \cdots (9)$$

결과:

$$l(y, t) = l(y, 0) \left\{1 - \left(\frac{C_0 t}{a}\right)^2 \left(1 - 4\frac{y^2}{b^2}\right) \exp\left(-\frac{2y^2}{b^2}\right) + O^4(C_0 t/a)\right\} \quad \cdots (10)$$

$x$축 근처($y^2 \ll b^2$)에서는 $l(y,t)/l(y,0)$이 $y$에 거의 무관하여, 전이층이 두께를 줄이면서 균일하게 유지됨을 보여준다. $y^2 > b^2$이면 두께가 증가하지만 지수 인자에 의해 팽창은 크지 않다.

Near the $x$-axis ($y^2 \ll b^2$), $l(y,t)/l(y,0)$ is essentially independent of $y$, showing the layer remains uniform as it thins. For $y^2 > b^2$ the thickness increases, but the exponential factor keeps inflation modest.

---

### Section III: Merging of Fields / 자기장의 병합 (pp. 514–519)

이 절은 논문의 핵심 부분으로, Parker가 Sweet의 병합 메커니즘의 정량적 모델을 구축한다.

This is the core section where Parker constructs the quantitative model of Sweet's merging mechanism.

**정상 상태 가정 (Steady-State Assumption)**:

정상 상태 조건($\partial/\partial t = 0$)에서, 필드가 충분히 압축되어($a \ll b$) $y^2 > b^2$ 영역에서 유체 운동이 균일하고 질서 정연하다고 가정한다. 이 영역에서 $v_x/v_y$, $B_x/B_y$, $(\partial/\partial y)/(\partial/\partial x)$는 모두 $O(a/b)$로 작다.

Under steady-state conditions ($\partial/\partial t = 0$), with fields sufficiently compressed ($a \ll b$), the fluid motion in the region $y^2 > b^2$ is orderly and uniform. In this region, $v_x/v_y$, $B_x/B_y$, and $(\partial/\partial y)/(\partial/\partial x)$ are all small, $O(a/b)$.

**유체역학 방정식 (Hydrodynamic Equation — Eq. 11)**:

$$\frac{\partial \mathbf{v}}{\partial t} + (\mathbf{v} \cdot \nabla)\mathbf{v} = -\frac{1}{\rho}\nabla\left[p + \frac{B^2}{8\pi}\right] + \frac{1}{4\pi\rho}(\mathbf{B} \cdot \nabla)\mathbf{B} \quad \cdots (11)$$

$x$-성분의 적분(전이층 $x^2 > l^2$ 바깥에서 필드가 천천히 변하므로 $|\mathbf{B}|$를 $x$-독립으로 취급):

$$p(x, y) + B^2(x, y)/8\pi \cong p_0 + B_x^2(\epsilon, y)/8\pi \quad \cdots (13)$$

즉, **총 압력(가스 압력 + 자기 압력)이 전이층 안과 밖에서 같다**.

That is, **total pressure (gas + magnetic) is the same inside and outside the transition layer**.

**유출 속도 유도 (Outflow Velocity Derivation — Eq. 14)**:

비선형 항 $v_x \partial v_x/\partial x$의 직접 적분이 불가능하므로, Parker는 Section II의 정성적 결과를 활용한다: $y^2 < b^2$ 영역에서 유출이 질서 정연하게 진행되며, $\partial v_x/\partial y$가 초과 압력 $p - p_0$에 비례한다고 가정:

$$\frac{\partial v_x}{\partial y} = \left(\frac{p - p_0}{\rho}\right)^{1/2} \frac{1}{L} \quad \cdots (\text{assumed})$$

(13)을 사용하면:

$$\frac{\partial v_x}{\partial y} = \left[\frac{B_x^2(\epsilon, y) - B^2(x, y)}{8\pi L^2 \rho}\right]^{1/2} \quad \cdots (14)$$

여기서 $L$은 $O(b)$의 비례 상수이다.

**수자기 방정식과 병합 속도 (Hydromagnetic Equation & Merging Velocity)**:

정상 상태 수자기 방정식:

$$\frac{\partial \mathbf{B}}{\partial t} = \nabla \times (\mathbf{v} \times \mathbf{B}) + \frac{c^2}{4\pi\sigma}\nabla^2\mathbf{B}$$

적분하면 ($B_x$와 $\partial B_x/\partial y$가 $y = 0$에서 소멸):

$$v_x B_y = \frac{c^2}{4\pi\sigma}\left[\frac{\partial B_y}{\partial x} - \left(\frac{\partial B_y}{\partial x}\right)_0\right] \quad \cdots (17)$$

전이층 바깥에서 $\partial B_y/\partial x \cong B/L$이고 무시 가능. 따라서 자기장이 병합하는 속도 $v_m$은:

$$v_m = \frac{c^2}{4\pi\sigma B_x(\epsilon, y)} \left(\frac{\partial B_y}{\partial x}\right)_0 \quad \cdots (18)$$

**핵심 미분 방정식 (Master ODE — Eq. 26)**:

Parker는 무차원 변수를 도입한다:
- $\phi \equiv B(x, y)/B_x(\epsilon, y)$ — 정규화된 자기장 (Eq. 23)
- $f(\phi) \equiv (\partial B/\partial x)/(\partial B/\partial x)_0$ — 정규화된 기울기 (Eq. 24)
- $\xi \equiv B_x(\epsilon, y)/\beta(y)$ (Eq. 25)
- $\Delta \equiv B_m/(\partial B/\partial y)_0$, $\lambda \equiv B_m/(\partial B/\partial x)_0$ — 특성 길이

$\phi$를 독립 변수로 하면:

$$f\frac{df}{d\phi} - \frac{f(f-1)}{\phi}\left(\frac{1+\xi^2\phi^2}{1-\xi^2\phi^2}\right) + \frac{\Delta}{\lambda}\frac{\phi(1-\phi^2)^{1/2}}{(1-\xi^2\phi^2)^{1/2}} = 0 \quad \cdots (26)$$

이것은 $\phi$에 대한 비선형 상미분 방정식으로, 경계 조건 $f(1) = 0$ ($\partial B/\partial y$가 전이 영역을 벗어나면 소멸)을 가진다.

This is a nonlinear ODE in $\phi$ with boundary condition $f(1) = 0$ (since $\partial B/\partial y$ vanishes when we leave the transition region).

**급수 해 (Series Solutions)**:

$\phi = 0$ 근처($B \to 0$, 전이층 중심):

$$f(\phi) = 1 - (\Delta/\lambda)\phi^2 + \tfrac{1}{4}(\Delta/\lambda)(1+14\xi^2/3 - 97\xi^4/3 + \cdots)\phi^4 + \cdots \quad \cdots (27)$$

$\phi = 1$ 근처(전이층 경계, $B \to B_m$)에서 $f(\phi)$를 일반 급수 전개할 수 없다 ($\phi = 1$은 정칙점이 아님). Parker는 반복법(reiteration)으로 풀어:

$$f^{(0)}(\phi) = \left(\frac{2\Delta}{3\lambda}\right)^{1/2}(1-\phi^2)^{3/4}\left\{1 + \frac{\xi^2}{20}(2+3\phi^2) + O^4(\xi)\right\} \quad \cdots (29)$$

**매칭 조건 (Matching Condition)**:

$\phi = 0$ 근처의 급수와 $\phi = 1$ 근처의 급수를 $\phi = 3/4$에서 매칭하여 $\Delta/\lambda$ 비율을 결정한다:

- 비압축성($\xi = 0$): $\Delta/\lambda = 0.820$
- 약간 압축성($\xi = 0.316$): $\Delta/\lambda = 0.772$

Figure 3은 두 경우의 $f(\phi)$를 비교하여 보여준다. 압축성 효과는 크지 않다.

Figure 3 compares $f(\phi)$ for both cases. Compressibility effects are modest.

**최종 병합 속도 (Final Merging Velocity — Eqs. 32, 33)**:

(18), (20)–(22), (25)를 결합하면:

$$v_m = \frac{\xi^{1/2}}{2\sqrt{\pi}} \left(\frac{\lambda}{\Delta}\right)^{1/2} \left[\left(\frac{p_0/\rho_0}{L\sigma}\right)^{1/2}\right]^{1/2}$$

최종적으로:

- **비압축성 ($\xi = 0$)**: $v_m = 0.263\, c(C/\sigma L)^{1/2} \quad \cdots (32)$
- **압축성 ($\xi = 0.316$)**: $v_m = 0.253\, c(C/\sigma L)^{1/2} \quad \cdots (33)$

여기서 $C = B_x(\epsilon, y)/(4\pi\rho_0)^{1/2}$는 전이 영역 바깥의 수자기 속도이다.

Where $C = B_x(\epsilon, y)/(4\pi\rho_0)^{1/2}$ is the hydromagnetic velocity outside the transition region.

압축성의 효과: $v_m$이 약간 감소하지만 순 효과가 미미하다. 기울기 $(\partial B/\partial y)_0$이 압축에 의해 증가하지만, 밀도 증가 때문에 유체 방출이 느려져 상쇄된다.

Effect of compressibility: $v_m$ decreases slightly, but the net effect is minor. The gradient $(\partial B/\partial y)_0$ is enhanced by compression, but the increased density slows fluid expulsion, nearly canceling out.

---

### Section IV: Conclusion / 결론 (pp. 519–520)

Parker는 Sweet의 메커니즘의 천체물리학적 의미를 정리한다:

Parker summarizes the astrophysical implications of Sweet's mechanism:

**Sweet-Parker 속도 vs 순수 확산 속도**:

- 순수 옴 확산: $v_{\text{diff}} = c(c/\sigma L)$ → 속도항 $\nabla \times (\mathbf{v} \times \mathbf{B})$ 없이 확산 방정식만으로 얻어짐
  Pure ohmic diffusion: obtained from diffusion equation alone without the velocity term
- Sweet-Parker: $v_m \sim c(C/\sigma L)^{1/2}$ → 확산과 유체 운동의 결합
  Sweet-Parker: coupling of diffusion and fluid motion

Sweet-Parker 속도와 순수 확산 속도의 비율: $(C/c)^{1/2}(\sigma L/c)^{1/2}$ — **대규모 전도도, 대규모 길이, 큰 수자기 속도에서 매우 크다**.

The ratio of Sweet-Parker to pure diffusion velocity: $(C/c)^{1/2}(\sigma L/c)^{1/2}$ — **very large for large conductivity, large scale, and large hydromagnetic velocity**.

**필요 조건 (Requirements)**:
1. 두 자기장이 반평행(antiparallel)이어야 한다 — 완전히 반평행이 아니어도 됨
   Fields must be antiparallel — need not be perfectly so
2. 큰 전기 전도도, 큰 규모, 큰 수자기 속도
   Large electrical conductivity, large scale, large hydromagnetic velocity

**천체물리학적 응용 (Astrophysical Applications)**:

- **태양 플레어**: Sweet의 메커니즘이 자기장선 재결합을 일으켜 불안정한 배치(configuration)를 만들고, 자기 에너지를 운동 에너지로 변환할 수 있다. Parker(1957)에서 태양 플레어의 예시를 제시했다고 언급.
  Solar flares: Sweet's mechanism may produce reconnection, creating unstable configurations and converting magnetic energy to kinetic energy.

- **지구 자기장**: 지구 외부 자기장의 침투. $\sigma = 10^{13}$ esu, $L = 10^9$ cm, $\rho = 10^{-21}$ g/cm³이면 병합 속도 $\sim 0.1$ km/sec. Chapman-Ferraro 고리 전류 모델에 대한 도전.
  Earth's magnetic field penetration: with $\sigma = 10^{13}$ esu, $L = 10^9$ cm, $\rho = 10^{-21}$ g/cm³, merging velocity $\sim 0.1$ km/sec. Challenges the Chapman-Ferraro ring-current model.

- **MHD 난류(turbulence)**: Sweet의 메커니즘이 수자기 난류에서 자기장의 확산과 소산을 변형시킬 수 있을지 추측.
  MHD turbulence: speculation that Sweet's mechanism might modify diffusion and dissipation of magnetic fields in hydromagnetic turbulence.

**Figure 4 — 수직 자속관의 병합**:

Parker는 마지막으로 두 개의 **수직(perpendicular)** 자속관(flux tubes)이 만나는 경우를 도식적으로 보여준다(Fig. 4). 접촉 영역에서 유체가 자기력선을 따라 빠져나가고, 자기장선이 끊어져 재결합한 후 자기 장력(tension)에 의해 더 짧은 경로로 당겨진다.

Parker concludes by illustrating (Fig. 4) the merging of two **perpendicular** flux tubes. Fluid squeezes out of the contact region along field lines, field lines sever and reconnect, and tension in the reconnected lines pulls them to a shorter path.

---

## 3. Key Takeaways / 핵심 시사점

1. **Sweet-Parker 모델은 최초의 정량적 자기 재결합 모델이다** — Sweet가 제안한 정성적 아이디어를 Parker가 수학적으로 정량화하여, 병합 속도 $v_m \sim c(C/\sigma L)^{1/2}$를 유도했다. 이 공식은 입력 매개변수(자기장, 밀도, 전도도, 규모)만 알면 재결합 속도를 예측할 수 있게 해준다.
   The Sweet-Parker model is the first quantitative magnetic reconnection model — Parker formalized Sweet's qualitative idea, deriving the merging velocity $v_m \sim c(C/\sigma L)^{1/2}$. This formula predicts the reconnection rate from input parameters (field, density, conductivity, scale) alone.

2. **순수 확산보다 훨씬 빠르지만, 태양 플레어를 설명하기에는 여전히 느리다** — Sweet-Parker 속도는 순수 옴 확산 $c^2/\sigma L$보다 $(C\sigma L/c)^{1/2}$배 빠르지만, 태양 코로나 조건에서 $R_m \sim 10^{12}$이므로 재결합 시간은 수백 년이 아닌 수주이지만, 관측된 플레어(수분~수시간)보다 여전히 느리다.
   Much faster than pure diffusion but still too slow for solar flares — the Sweet-Parker velocity is $(C\sigma L/c)^{1/2}$ times faster than pure ohmic diffusion, but with coronal $R_m \sim 10^{12}$, the reconnection time is weeks rather than centuries, yet still far slower than observed flares (minutes to hours).

3. **전이층의 극단적 박리가 핵심 물리이다** — $l/L \sim 10^{-4}$, 즉 $10^9$ cm 규모의 자기장 구조에서 전이층 두께는 $\sim 10^5$ cm (1 km) 에 불과하다. 이 극단적인 기하학적 비율이 느린 재결합의 근본 원인이다: 좁은 통로를 통해 모든 유체가 빠져나가야 하므로 병목 현상이 발생한다.
   The extreme thinning of the transition layer is the key physics — $l/L \sim 10^{-4}$: for a $10^9$ cm field structure, the layer is only $\sim 10^5$ cm (1 km) thick. This extreme aspect ratio is the root cause of slow reconnection: all fluid must escape through a narrow bottleneck.

4. **Parker의 접근법은 유체 방출과 자기 확산을 결합한다** — Section II에서 유체가 자기 압력에 의해 방출되는 과정을 분석하고, Section III에서 이를 자기 확산 방정식과 결합하여 자기 일관적(self-consistent) 해를 구한다. 이 "결합(coupling)" 아이디어는 이후 모든 재결합 모델의 기본 구조가 되었다.
   Parker's approach couples fluid expulsion with magnetic diffusion — Section II analyzes fluid expulsion by magnetic pressure, Section III combines this with the magnetic diffusion equation to obtain a self-consistent solution. This "coupling" idea became the basic structure of all subsequent reconnection models.

5. **압축성(compressibility)의 효과는 미미하다** — 비압축성($\xi = 0$)과 압축성($\xi = 0.316$) 경우의 병합 속도가 각각 $0.263$, $0.253$으로 단 4% 차이. 이는 밀도 증가와 기울기 증가가 거의 상쇄되기 때문이다.
   Compressibility effects are negligible — merging velocities for incompressible ($\xi = 0$) and compressible ($\xi = 0.316$) cases are $0.263$ and $0.253$ respectively, only a 4% difference, because density increase and gradient enhancement nearly cancel.

6. **이 논문은 "느린 재결합 문제"를 발견한 논문이다** — Parker가 계산한 속도가 태양 플레어를 설명하기에 불충분하다는 인식은, 이후 Petschek(1964)의 빠른 재결합 모델과 현대 plasmoid instability 연구의 직접적 동기가 되었다. 문제를 정확하게 정의한 것 자체가 큰 기여이다.
   This paper discovered the "slow reconnection problem" — Parker's realization that his calculated velocity is insufficient for solar flares directly motivated Petschek's (1964) fast reconnection model and modern plasmoid instability research. Precisely defining the problem was itself a major contribution.

7. **"Merging"에서 "Reconnection"으로의 용어 진화** — Parker는 일관되게 "merging"이라는 용어를 사용한다. 오늘날 "reconnection"이 표준 용어가 되었지만, Parker의 원래 물리적 묘사는 변하지 않았다. 자기장선이 끊어지고 재결합하는 것보다는 "두 반평행 자기장이 서로 합쳐진다"는 관점.
   Terminology evolution from "merging" to "reconnection" — Parker consistently uses "merging." While "reconnection" became the standard term, Parker's original physical description remains unchanged — the perspective of two antiparallel fields "merging together" rather than field lines "breaking and reconnecting."

8. **논문의 구조가 교과서적이다** — (1) 차원 분석으로 직관 제공, (2) 이상화된 모델로 물리 과정 이해, (3) 엄밀한 수학적 유도, (4) 천체물리학적 응용. 이 구조는 이론 물리학 논문의 모범으로 꼽힌다.
   The paper's structure is textbook-quality — (1) dimensional analysis for intuition, (2) idealized model for physical understanding, (3) rigorous mathematical derivation, (4) astrophysical applications. This structure is considered a model for theoretical physics papers.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 기본 방정식 체계 / Basic Equation System (CGS)

**유도 방정식 (Induction Equation)**:

$$\frac{\partial \mathbf{B}}{\partial t} = \nabla \times (\mathbf{v} \times \mathbf{B}) + \frac{c^2}{4\pi\sigma}\nabla^2\mathbf{B} \quad \cdots (1)$$

- $\mathbf{B}$: 자기장 / magnetic field
- $\mathbf{v}$: 유체 속도 / fluid velocity
- $c$: 광속 / speed of light
- $\sigma$: 전기 전도도 / electrical conductivity

첫 번째 항: 대류(convection) — 자기장을 유체와 함께 운반
두 번째 항: 확산(diffusion) — 자기장을 유체에 대해 확산

First term: convection — transports field with fluid.
Second term: diffusion — diffuses field relative to fluid.

**유체 운동 방정식 (Momentum Equation)**:

$$\frac{\partial \mathbf{v}}{\partial t} + (\mathbf{v} \cdot \nabla)\mathbf{v} = -\frac{1}{\rho}\nabla\left[p + \frac{B^2}{8\pi}\right] + \frac{1}{4\pi\rho}(\mathbf{B} \cdot \nabla)\mathbf{B} \quad \cdots (11)$$

- $p$: 가스 압력 / gas pressure
- $B^2/8\pi$: 자기 압력 / magnetic pressure
- $(\mathbf{B} \cdot \nabla)\mathbf{B}/4\pi\rho$: 자기 장력 / magnetic tension

### 4.2 Sweet-Parker 층의 기하학 / Sweet-Parker Layer Geometry

```
         B↑          B↑          B↑
         |           |           |
   ======|===========|===========|======  ← 전이층 상부 경계
    → → v_out   전이층 (두께 l)   v_out → →
   ======|===========|===========|======  ← 전이층 하부 경계
         |           |           |
         B↓          B↓          B↓

         ←————————— L ——————————→
         ↕ l (l ≪ L)
```

- $L$: 전이층 길이 (전체 규모, $\sim b$) / Layer length (global scale)
- $l$: 전이층 두께 / Layer thickness
- $v_m$: 유입 속도 (병합 속도) / Inflow (merging) velocity
- $v_{\text{out}} \sim C$: 유출 속도 (Alfvén 속도) / Outflow velocity (Alfvén speed)

### 4.3 핵심 유도 과정 / Key Derivation

**Step 1: 질량 보존 / Mass Conservation**

$$\rho\, v_m\, L = \rho\, v_{\text{out}}\, l \quad \Rightarrow \quad \frac{v_m}{v_{\text{out}}} = \frac{l}{L}$$

**Step 2: 유출 속도 = 수자기 속도 / Outflow = Hydromagnetic Velocity**

자기 압력 $B^2/8\pi$가 유체를 가속:

$$\frac{1}{2}\rho v_{\text{out}}^2 \cong \frac{B^2}{8\pi} \quad \Rightarrow \quad v_{\text{out}} \cong C_0 = \frac{B}{(4\pi\rho)^{1/2}}$$

**Step 3: 확산-대류 균형 / Diffusion-Convection Balance**

전이층 내에서 자기장이 확산에 의해 소멸되는 속도와 대류에 의해 밀려오는 속도가 균형:

$$v_m \cong \frac{c^2}{4\pi\sigma\, l} \quad \Rightarrow \quad l \cong \frac{c^2}{4\pi\sigma\, v_m}$$

**Step 4: 결합 → 병합 속도 / Combine → Merging Velocity**

Step 1과 Step 3에서 $l$을 소거:

$$v_m = v_{\text{out}} \cdot \frac{l}{L} = C_0 \cdot \frac{c^2}{4\pi\sigma\, v_m\, L}$$

$$v_m^2 = \frac{C_0\, c^2}{4\pi\sigma\, L}$$

$$\boxed{v_m = c\left(\frac{C_0}{\sigma L}\right)^{1/2} \cdot \frac{1}{(4\pi)^{1/2}}}$$

Parker의 엄밀한 계산 결과(수치 계수 포함):

$$v_m = 0.263\, c\left(\frac{C}{\sigma L}\right)^{1/2} \quad \text{(비압축성)} \quad \cdots (32)$$

$$v_m = 0.253\, c\left(\frac{C}{\sigma L}\right)^{1/2} \quad \text{(압축성, } \xi = 0.316\text{)} \quad \cdots (33)$$

### 4.4 SI 단위 변환 / SI Unit Conversion

현대적 SI 단위로 변환하면 ($\eta = c^2/(4\pi\sigma)$ → $\eta = 1/(\mu_0\sigma)$):

In modern SI units ($\eta = c^2/(4\pi\sigma)$ → $\eta = 1/(\mu_0\sigma)$):

$$v_m = \frac{v_A}{\sqrt{R_m}}, \quad R_m = \frac{v_A L}{\eta}$$

- $v_A = B/\sqrt{\mu_0\rho}$: Alfvén 속도
- $\eta = 1/(\mu_0\sigma)$: 자기 확산 계수 / magnetic diffusivity
- $R_m$: 자기 레이놀즈 수 / magnetic Reynolds number

### 4.5 전이층 두께 / Transition Layer Thickness

$$\frac{l}{L} = \frac{1}{\sqrt{R_m}}$$

태양 코로나에서 $R_m \sim 10^{12}$ → $l/L \sim 10^{-6}$

In the solar corona with $R_m \sim 10^{12}$ → $l/L \sim 10^{-6}$

### 4.6 비선형 마스터 ODE / Nonlinear Master ODE (Eq. 26)

$$f\frac{df}{d\phi} - \frac{f(f-1)}{\phi}\left(\frac{1+\xi^2\phi^2}{1-\xi^2\phi^2}\right) + \frac{\Delta}{\lambda}\frac{\phi(1-\phi^2)^{1/2}}{(1-\xi^2\phi^2)^{1/2}} = 0$$

- $\phi = B/B_m$: 정규화된 자기장 세기 ($0 \le \phi \le 1$)
  Normalized field strength
- $f = (\partial B/\partial x)/(\partial B/\partial x)_0$: 정규화된 $x$-기울기
  Normalized $x$-gradient
- $\xi = B_x(\epsilon,y)/\beta(y)$: 압축성 매개변수
  Compressibility parameter
- $\Delta/\lambda$: 두 특성 길이의 비율 — 해의 고유값(eigenvalue)
  Ratio of two characteristic lengths — eigenvalue of the solution

경계 조건: $f(0) = 1$ (전이층 중심에서 기울기 최대), $f(1) = 0$ (전이층 밖에서 기울기 소멸)

Boundary conditions: $f(0) = 1$ (maximum gradient at layer center), $f(1) = 0$ (gradient vanishes outside layer)

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1942  Alfvén        — MHD파 발견, 자기장 동결 조건 / MHD waves, frozen-in theorem
  │
1946  Giovanelli    — 자기 중성점에서 플레어 이론 / Flare theory via neutral point [Paper #18]
  │
1950  Alfvén        — "Cosmical Electrodynamics" 출판 / Published textbook
  │
1953  Cowling       — 옴 확산 시간 ~300년으로 너무 느림 / Ohmic diffusion timescale ~300 yr too slow
  │
1953  Dungey        — X-형 중성점에서 토폴로지 변화 제안 / Topology change at X-type neutral points
  │
1956  Sweet         — 반평행 자기장 병합 아이디어 (IAU) / Antiparallel field merging idea (IAU)
  │
1957  Parker ★      — Sweet-Parker 모델 정량화 / Quantitative S-P model ← 이 논문 / THIS PAPER
  │
1958  Parker        — 태양풍 이론 / Solar wind theory
  │
1961  Dungey        — 지구 자기권 재결합 / Magnetospheric reconnection
  │
1964  Petschek      — 빠른 재결합 모델 (v ~ v_A/ln R_m) / Fast reconnection model
  │
1966  Sturrock      — 플레어의 자기 에너지 저장 모델 / Magnetic energy storage model for flares
  │
1974  Carmichael,   — CSHKP 표준 플레어 모델 / Standard flare model
      Sturrock,
      Hirayama,
      Kopp-Pneuman
  │
2007+ Loureiro      — Plasmoid instability: Sweet-Parker 시트 불안정성 / S-P sheet instability
      et al.           → 빠른 재결합으로 천이 / transition to fast reconnection
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Giovanelli (1946) — Paper #18 | 자기 중성점에서의 전자 가속으로 플레어 설명; Parker의 논문은 중성점에서 자기장 자체가 어떻게 에너지를 방출하는지 정량화 / Explained flares via electron acceleration at neutral points; Parker quantified how the field itself releases energy at such points | 직접적 전신: Giovanelli의 "자기 중성점"이 Parker의 "전이층"의 물리적 기원 / Direct predecessor: Giovanelli's "neutral point" is the physical origin of Parker's "transition layer" |
| Sweet (1956) — IAU Proceedings | Parker가 정량화한 물리적 아이디어의 원천 / Source of the physical idea that Parker formalized | 직접적 전신: Sweet의 정성적 그림 + Parker의 수학 = Sweet-Parker 모델 / Direct predecessor: Sweet's qualitative picture + Parker's mathematics = Sweet-Parker model |
| Cowling (1953) | 흑점 자기장의 옴 확산 시간이 ~600년으로, 관측된 자기장 변화를 설명할 수 없음을 보임 / Showed ohmic diffusion time ~600 yr for sunspot fields, unable to explain observed field changes | 동기 부여: "확산이 너무 느리다"는 문제 제기가 Sweet-Parker 메커니즘의 필요성을 확립 / Motivation: the "diffusion is too slow" problem established the need for the Sweet-Parker mechanism |
| Dungey (1953) | X-형 중성점에서 자기장 토폴로지 변화 가능성을 처음 제안 / First proposed topology change at X-type neutral points | 개념적 기초: 자기장 토폴로지가 변할 수 있다는 아이디어를 Parker가 정량적 모델로 발전 / Conceptual foundation: Parker developed the idea of topology change into a quantitative model |
| Petschek (1964) | Sweet-Parker의 "느린 재결합" 문제를 해결하기 위해 느린 충격파(slow shocks)를 도입, $v_m \sim v_A/\ln R_m$ / Introduced slow shocks to solve the "slow reconnection" problem, $v_m \sim v_A/\ln R_m$ | 직접적 후속: Parker가 제기한 문제를 풀기 위한 첫 번째 시도; 전류 시트를 짧게 만들어 병목을 해소 / Direct successor: first attempt to solve Parker's problem; shortened the current sheet to relieve the bottleneck |
| Loureiro et al. (2007) | Sweet-Parker 전류 시트가 $R_m > 10^4$에서 tearing mode에 의해 불안정하여 plasmoid chain으로 분열 / Sweet-Parker sheets unstable to tearing at $R_m > 10^4$, fragmenting into plasmoid chains | 현대적 해결: Sweet-Parker 시트는 실제로 존재할 수 없으며, 자발적으로 빠른 재결합으로 천이함 / Modern resolution: Sweet-Parker sheets cannot actually exist; they spontaneously transition to fast reconnection |

---

## 7. References / 참고문헌

- Parker, E. N., "Sweet's Mechanism for Merging Magnetic Fields in Conducting Fluids," *J. Geophys. Res.*, 62(4), 509–520, 1957. [DOI: 10.1029/JZ062i004p00509]
- Sweet, P. A., "The Neutral Point Theory of Solar Flares," *Proceedings of the IAU Symposium on Electromagnetic Phenomena in Cosmical Physics*, Stockholm, 1956.
- Parker, E. N. and Krook, M., *Astrophys. J.*, 124, 214, 1956.
- Parker, E. N., *Phys. Rev.*, 107, 830, 1957.
- Storey, L. R. O., *Phil. Trans. R. Soc.*, A, 246, 113, 1954.
- Parker, E. N., *J. Geophys. Res.*, 61, 625, 1956.
- Giovanelli, R. G., "A Theory of Chromospheric Flares," *Nature*, 158, 81–82, 1946.
- Cowling, T. G., *The Sun* (ed. Kuiper), Chapter 8, University of Chicago Press, 1953.
- Dungey, J. W., "Conditions for the Occurrence of Electrical Discharges in Astrophysical Systems," *Phil. Mag.*, 44, 725–738, 1953.
- Petschek, H. E., "Magnetic Field Annihilation," *NASA Spec. Publ.*, SP-50, 425–439, 1964.
- Loureiro, N. F., Schekochihin, A. A., and Cowley, S. C., "Instability of current sheets and formation of plasmoid chains," *Phys. Plasmas*, 14, 100703, 2007.
