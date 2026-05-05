# Pre-reading Briefing / 사전 읽기 브리핑

**Paper**: Dynamics of the Interplanetary Gas and Magnetic Fields
**Author**: Eugene N. Parker
**Year**: 1958
**Journal**: The Astrophysical Journal, 128, 664–676
**DOI**: 10.1086/146579

---

## 핵심 기여 / Core Contribution

Parker는 Biermann(1951)이 혜성 꼬리 관측으로부터 추론한 태양의 연속적 입자 방출에 대한 **이론적 정당화**를 제공했다. 핵심 논증은 놀랍도록 우아하다: (1) 태양 코로나가 $\sim 10^6$ K의 고온이므로 정역학적 평형(hydrostatic equilibrium)을 가정하면 무한대 거리에서 압력이 0이 아닌 유한값($p(\infty) \sim 10^{-5}$ dyne/cm²)에 수렴하는데, (2) 성간 매질의 압력은 이보다 훨씬 낮으므로($\sim 10^{-13}$ dyne/cm²), (3) 정역학적 평형은 **불가능**하고 코로나는 **반드시 바깥으로 팽창**해야 한다. Parker는 구대칭 정상 상태 유체역학 방정식을 풀어 코로나가 **아음속에서 초음속으로 천이하는 연속적 유출** — 이것을 "solar wind"라 명명 — 이 필연적임을 보였다. 또한 태양 자전과 방사 유출의 결합이 행성간 자기장을 **나선(spiral)** 구조로 만든다는 것을 유도했다. 이 "Parker spiral"은 1962년 Mariner 2에 의해 확인되었으며, 현대 heliospheric physics의 기초가 되었다.

Parker provided the **theoretical justification** for Biermann's (1951) inference of continuous solar particle emission from comet tail observations. The core argument is elegantly simple: (1) the solar corona at $\sim 10^6$ K, if assumed in hydrostatic equilibrium, yields a non-vanishing pressure at infinity ($p(\infty) \sim 10^{-5}$ dyne/cm²); (2) interstellar pressure is far lower ($\sim 10^{-13}$ dyne/cm²); (3) therefore hydrostatic equilibrium is **impossible** and the corona **must expand outward**. Parker solved the spherically symmetric steady-state hydrodynamic equations to show that the corona inevitably undergoes a **subsonic-to-supersonic transition** — naming this continuous outflow the "solar wind." He further derived that the combination of solar rotation and radial outflow shapes the interplanetary magnetic field into a **spiral** structure. This "Parker spiral" was confirmed by Mariner 2 in 1962 and became the foundation of modern heliospheric physics.

---

## 역사적 맥락 / Historical Context

### 이전 배경 / Prior Background

- **Chapman (1957)**: 정적 코로나 모델 — 코로나가 열전도에 의해 먼 거리까지 확장되지만 정역학적 평형을 유지한다고 가정. Parker가 정면으로 반박하는 대상. / Static corona model — assumed hydrostatic equilibrium extending to large distances via thermal conduction. Parker's direct target of refutation.
- **Biermann (1951)**: 혜성 꼬리 관측으로 연속적 태양 입자 방출 추론. Parker의 출발점. / Inferred continuous solar particle emission from comet tails. Parker's starting point.
- **Schlüter (1954)**: 태양에서의 입자 유출 메커니즘으로 melon-seed 과정 제안. Parker가 불충분하다고 판단. / Proposed melon-seed process for particle ejection. Parker deemed insufficient.

### 이 논문의 위치 / Where This Paper Fits

```
1931  Chapman & Ferraro ── 자기 폭풍과 태양 입자 구름
      │
1942  Alfvén ──────────── MHD, frozen-in 조건
      │
1951  Biermann ─────────── 혜성 꼬리 → 연속적 corpuscular radiation (관측 증거)
      │
1957  Chapman ──────────── 정적 코로나 모델 (열전도로 확장)
      │
      ▼
╔═══════════════════════════════════════════════════════════════════════╗
║  1958  PARKER — 태양풍 이론                                          ║
║         정역학 평형 불가능 → 코로나의 초음속 팽창 필연적                   ║
║         "solar wind" 명명                                            ║
║         Parker spiral 유도 (행성간 자기장의 나선 구조)                   ║
║         Dynamics of the Interplanetary Gas and Magnetic Fields       ║
╚═══════════════════════════════════════════════════════════════════════╝
      │
      ├── 1962  Mariner 2 ──── 태양풍 직접 관측 확인
      │
      ├── 1965  Ness ────────── Parker spiral 관측 확인
      │
      ├── 1967  Weber & Davis ── MHD 태양풍 모델 (자기장 포함)
      │
      └── 2018  Parker Solar Probe ── Parker의 이름을 딴 탐사선 발사
```

### 논문 수용의 역사 / Reception History

이 논문은 처음에 **격렬한 반대**에 직면했다. 심사자 2명 모두 거부했으며, 편집자 Chandrasekhar가 개인적으로 게재를 결정했다. Chapman은 정적 코로나 모델을 강하게 지지했고, 많은 천문학자들이 Parker의 결론을 의심했다. 1962년 Mariner 2의 직접 관측이 Parker의 예측을 확인하면서 논쟁이 종결되었다.

The paper initially faced **fierce opposition**. Both referees rejected it; editor Chandrasekhar personally overruled them. Chapman strongly defended his static corona model. The debate ended when Mariner 2's in-situ measurements in 1962 confirmed Parker's predictions.

---

## 필요한 배경 지식 / Prerequisites

### 유체역학 / Hydrodynamics
- **연속 방정식(continuity equation)**: 질량 보존, $\nabla \cdot (\rho \mathbf{v}) = 0$
- **오일러 방정식(Euler equation)**: 압력 기울기 + 중력에 의한 유체 운동
- **정역학 평형(hydrostatic equilibrium)**: 압력 기울기 = 중력 (정지 상태)
- **아음속/초음속 천이(subsonic/supersonic transition)**: de Laval 노즐 유사 — 유체가 아음속에서 초음속으로 전이할 때의 조건

### 열역학 / Thermodynamics
- **이상 기체 상태 방정식**: $p = 2NkT$ (완전 이온화 수소, 이온+전자)
- **열전도(thermal conduction)**: Spitzer 열전도도 $\kappa(T) \propto T^n$
- **열유속 방정식(heat-flow equation)**: $\nabla \cdot [\kappa(T) \nabla T] = 0$

### MHD 기초 / Basic MHD
- **Frozen-in 조건**: 완전 전도 플라즈마에서 자기력선이 유체와 함께 운동
- **자기 응력(magnetic stress)**: $B_r B_\phi / 4\pi$ — 태양 회전 감속의 원인

### 이전 논문과의 연결 / Connection to Prior Papers
- **Biermann (1951)** [SP #11]: 관측 증거 — Parker가 이론화하는 대상
- **Alfvén (1942)** [SP #8]: Frozen-in 조건 — Parker spiral 유도의 물리적 기반

---

## 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **Solar wind** (태양풍) | Parker가 이 논문에서 명명. 태양 코로나에서 초음속으로 팽창하여 행성간 공간으로 흘러나가는 이온화된 가스. 이전에는 "corpuscular radiation"이라 불림 / Named in this paper. Ionized gas expanding supersonically from the corona into interplanetary space |
| **Hydrostatic equilibrium** (정역학 평형) | 중력과 압력 기울기가 균형을 이루어 유체가 정지해 있는 상태. Parker는 코로나에 대해 이것이 **불가능**함을 보임 / Balance between gravity and pressure gradient. Parker showed this is **impossible** for the corona |
| **$\lambda$ (lambda)** | 무차원 매개변수: $\lambda = GM_\odot M / 2kT_0 a$. 중력 에너지와 열 에너지의 비율. $\lambda > 2$이면 초음속 팽창이 발생 / Dimensionless parameter: ratio of gravitational to thermal energy. Supersonic expansion occurs when $\lambda > 2$ |
| **$\xi$ (xi)** | 무차원 거리: $\xi = r/a$ (코로나 기저부로부터의 거리). Parker 방정식의 독립 변수 / Dimensionless distance from coronal base |
| **$\psi$ (psi)** | 무차원 속도 변수: $\psi = \frac{1}{2}Mv^2/kT_0$. Parker 방정식의 종속 변수 / Dimensionless velocity variable: ratio of kinetic to thermal energy |
| **Critical point** (임계점) | $\xi = \lambda/2$ ($r = r_c$)에서 유속이 아음속에서 초음속으로 전이하는 점. de Laval 노즐의 목(throat)에 해당 / Point where flow transitions from subsonic to supersonic, analogous to a de Laval nozzle throat |
| **Parker spiral** (Parker 나선) | 태양 자전과 방사 유출의 결합으로 형성되는 행성간 자기장의 나선 구조. Eq. 25–27에서 유도 / Spiral structure of interplanetary magnetic field from combined solar rotation and radial outflow |
| **$\psi_0$ (psi-zero)** | 코로나 기저부($r = a$)에서의 초기 속도 매개변수. 정상 상태 해의 고유값(eigenvalue) — $T_0$에 의해 결정 / Initial velocity parameter at coronal base. Eigenvalue of steady-state solution, determined by $T_0$ |

---

## 수식 미리보기 / Equations Preview

### 1. 정역학 평형과 그 실패 / Hydrostatic Equilibrium and Its Failure

**정역학 평형 방정식 (Eq. 1):**

$$0 = \frac{d}{dr}(2NkT) + \frac{GM_\odot MN}{r^2}$$

이온화 수소의 압력은 $p = 2NkT$ (이온 + 전자). 온도가 열전도에 의해 결정된다고 가정하면:

$$T(r) = T_0 \left(\frac{a}{r}\right)^{1/(n+1)}$$

여기서 열전도도 $\kappa(T) \propto T^n$. 이온화 수소: $n = 5/2$, 중성 수소: $n = 1/2$.

이를 적분하면 무한대에서의 압력 (Eq. 9):

$$p(\infty) = p_0 \exp\left[\frac{-\lambda(n+1)}{n}\right]$$

**핵심**: $\lambda = GM_\odot M / 2kT_0 a \approx 5.35$ (이온화 수소, $T_0 = 1.5 \times 10^6$ K)이면 $p(\infty) = 0.55 \times 10^{-3} p_0$. 이 값은 성간 매질 압력보다 **수 자릿수 높아** 정역학 평형이 불가능!

**Key**: With $\lambda \approx 5.35$, $p(\infty) = 0.55 \times 10^{-3} p_0 \approx 10^{-5}$ dyne/cm², which is **orders of magnitude above** interstellar pressure — hydrostatic equilibrium is impossible!

### 2. 정상 상태 팽창 방정식 / Stationary Expansion Equations

**운동 방정식 (Eq. 10):**

$$NMv\frac{dv}{dr} = -\frac{d}{dr}(2NkT) - GM_\odot MN\frac{1}{r^2}$$

**연속 방정식 (Eq. 11):**

$$\frac{d}{dr}(r^2 Nv) = 0 \quad \Rightarrow \quad N(r)v(r) = N_0 v_0 \left(\frac{a}{r}\right)^2$$

### 3. Parker 방정식 (무차원) / Parker Equation (Dimensionless)

무차원 변수: $\xi = r/a$, $\tau = T(r)/T_0$, $\psi = \frac{1}{2}Mv^2/kT_0$

$$\frac{d\psi}{d\xi}\left(1 - \frac{\tau}{\psi}\right) = -2\xi^2 \frac{d}{d\xi}\left(\frac{\tau}{\xi^2}\right) - \frac{2\lambda}{\xi^2}$$

등온($\tau = 1$) 가정 시 적분하면 (Eq. 14):

$$\psi - \ln\psi = \psi_0 - \ln\psi_0 + 4\ln\xi - 2\lambda\left(1 - \frac{1}{\xi}\right)$$

### 4. 임계점 조건 / Critical Point Condition

정상 상태 해가 존재하려면 $Y$와 $Z$가 **같은 $\xi$에서** 동시에 최솟값을 가져야 함 → $\psi = 1$ at $\xi = \lambda/2$:

$$\psi_0 - \ln\psi_0 = 2\lambda - 3 - 4\ln\frac{\lambda}{2}$$

이것이 **고유값 조건**: 주어진 $T_0$ (따라서 $\lambda$)에 대해 $v_0$가 유일하게 결정됨!

This is the **eigenvalue condition**: for a given $T_0$ (hence $\lambda$), $v_0$ is uniquely determined!

### 5. Parker spiral — 행성간 자기장 / Interplanetary Magnetic Field (Eq. 25–26)

유선(streamline) 방정식 ($r = b$ 이상에서 $v = v_m = \text{const}$):

$$\frac{r}{b} - 1 - \ln\frac{r}{b} = \frac{v_m}{b\omega}(\phi - \phi_0)$$

자기장 성분 (Eq. 26):

$$B_r = B(θ, \phi_0)\left(\frac{b}{r}\right)^2$$

$$B_\phi = B(θ, \phi_0)\left(\frac{\omega}{v_m}\right)(r - b)\left(\frac{b}{r}\right)^2 \sin\theta$$

$B_\phi / B_r = \omega(r-b)\sin\theta / v_m$ → 먼 거리에서 $B_\phi$가 지배적 → **나선!**

### 6. 45° 나선각 도달 거리 / Radius Where Spiral Reaches 45°

$$B_\phi = B_r \quad \text{when} \quad r - b = \frac{v_m}{\omega}\sin\theta$$

적도면($\theta = \pi/2$)에서, $v_m = 1000$ km/s일 때:

$$r_{45°} \approx \frac{v_m}{\omega} = \frac{10^8}{2.7 \times 10^{-6}} \approx 2.5 \text{ AU}$$

---

## 논문 구조 미리보기 / Paper Structure Preview

| 섹션 / Section | 내용 / Content |
|---|---|
| I. Introduction (p. 664–665) | Biermann의 결론 소개, 문제 제기: 무엇이 입자를 방출하는가? / Introduces Biermann's conclusions, poses the question |
| II. Static Equilibrium (p. 665–666) | 정역학 평형의 **불가능성** 증명 — 핵심 논증 / Proves **impossibility** of hydrostatic equilibrium |
| III. Stationary Expansion (p. 667–669) | Parker 방정식 유도, 등온 해, Fig. 1 (속도 곡선) / Derives Parker equation, isothermal solutions |
| IV. Coronal Heating and Mass Loss (p. 670–672) | 가열 에너지 예산, 질량 손실률 / Heating energy budget, mass loss rate |
| V. General Solar Magnetic Field (p. 672–673) | Parker spiral 유도 (Eq. 24–27), Fig. 6 / Derives Parker spiral |
| VI. IMF and Solar Rotation (p. 674–675) | 자기 토크, 태양 회전 감속 / Magnetic torque, solar spin-down |
| VII. Plasma Instability (p. 675) | 행성간 자기장의 불안정성 → 혼란 자기장 껍질 / Instability → tangled field shell |

---

## 읽기 팁 / Reading Tips

1. **§II가 논문의 핵심 논증**입니다. 정역학 평형의 실패를 이해하면 나머지는 자연스럽게 따라옵니다. / §II is the paper's core argument. Understanding why hydrostatic equilibrium fails makes everything else follow.

2. **Fig. 1** (p. 668)을 주의 깊게 보세요 — 다양한 코로나 온도에서의 속도 곡선이 Parker의 주요 예측입니다. / Study Fig. 1 carefully — velocity curves for various coronal temperatures are Parker's main predictions.

3. **§V의 Parker spiral** (Eq. 25, Fig. 6)은 이 논문의 두 번째 대발견입니다. 간단한 기하학에서 나오는 심오한 결과에 주목하세요. / The Parker spiral in §V (Eq. 25, Fig. 6) is the paper's second great discovery. Note the profound result from simple geometry.

4. 이 논문은 Biermann(1951) [SP #11]의 **직접적 후속**입니다. Biermann이 관측에서 이론으로의 다리를 놓았다면, Parker는 그 다리 위에 정량적 이론을 세웠습니다. / This paper is a direct sequel to Biermann (1951) [SP #11].

5. 논문이 **거부당한 역사**를 기억하세요. 편집자 Chandrasekhar의 결단이 없었다면 현대 태양물리학의 역사가 달라졌을 것입니다. / Remember the paper was initially **rejected** by both referees.
