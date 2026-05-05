# Pre-Reading Briefing: Parker (1958) / 사전 읽기 브리핑

**Paper**: "Dynamics of the Interplanetary Gas and Magnetic Fields"
**Author**: Eugene N. Parker
**Year**: 1958
**Journal**: The Astrophysical Journal, Vol. 128, pp. 664–676
**DOI**: 10.1086/146579

---

## 1. 핵심 기여 / Core Contribution

Eugene Parker는 태양 코로나가 ~$10^6$ K의 고온이라면, **정수압 평형(hydrostatic equilibrium)이 불가능**하며 코로나 가스가 필연적으로 **초음속으로 바깥을 향해 팽창**해야 한다는 것을 유체역학적으로 증명했습니다. 이것이 바로 **태양풍(solar wind)**의 최초 이론적 예측입니다. Parker는 Biermann (1951)이 혜성 꼬리의 운동에서 추론한 "태양에서 지속적으로 가스가 방출된다"는 관측적 가설에서 출발하여, 간단한 구면 대칭 유체역학 방정식으로부터 코로나 온도에 따른 팽창 속도를 정량적으로 계산했습니다. 또한 태양 자전에 의해 행성간 자기장이 **아르키메데스 나선(Parker spiral)** 형태를 가져야 한다고 예측했습니다. 이 논문은 처음에 강한 반대에 직면했지만(심사자 2명이 거부, 편집장 Chandrasekhar의 결단으로 출판), 1962년 Mariner 2 탐사선의 관측으로 극적으로 확인되었습니다.

Eugene Parker proved hydrodynamically that if the solar corona is at ~$10^6$ K, **hydrostatic equilibrium is impossible** and coronal gas must inevitably **expand supersonically outward** — the first theoretical prediction of the **solar wind**. Starting from Biermann's (1951) observational hypothesis that gas continuously streams from the Sun (inferred from comet tail motions), Parker quantitatively computed expansion velocities as a function of coronal temperature using simple spherically symmetric hydrodynamic equations. He also predicted that the interplanetary magnetic field must form an **Archimedean spiral (Parker spiral)** due to solar rotation. Initially met with strong opposition (two reviewers rejected it; editor Chandrasekhar overruled them), the paper was dramatically confirmed by Mariner 2 observations in 1962.

---

## 2. 역사적 맥락 / Historical Context

```
1908  Birkeland ─ 태양 하전 입자가 오로라를 유발 (Paper #1)
  │         Solar charged particles cause aurora
  │
1931  Chapman & Ferraro ─ 간헐적 플라즈마 덩어리가 자기 폭풍 유발 (Paper #2)
  │         Intermittent plasma clouds cause magnetic storms
  │
1940  Chapman & Bartels ─ 지자기학 종합, 27일 재현, M-region (Paper #3)
  │         Geomagnetism synthesis, 27-day recurrence, M-regions
  │
1948  Biermann ─ 혜성 이온 꼬리가 태양 복사압만으로 설명 불가
  │         Comet ion tails unexplainable by solar radiation pressure alone
  │
1951  Biermann ─ 태양에서 연속적인 "corpuscular radiation" 제안
  │         Proposed continuous corpuscular radiation from the Sun
  │
1957  Chapman ─ 정적 코로나 모델 (태양풍 없이)
  │         Static corona model (no solar wind) — 매우 뜨거운 코로나가 태양계 전체로 확장
  │
1958  ★ Parker ─ "Dynamics of the Interplanetary Gas" ← 이 논문
  │         정수압 평형 불가능 → 초음속 팽창 필연 → "태양풍" 예측
  │         Hydrostatic equilibrium impossible → supersonic expansion inevitable → "solar wind"
  │         Parker spiral 예측
  │
1959  Parker ─ "solar wind" 용어 최초 사용 (후속 논문에서)
  │         First use of the term "solar wind"
  │
1962  Mariner 2 ─ 태양풍의 직접 관측 확인! (Neugebauer & Snyder)
  │         Direct observational confirmation of solar wind!
  │
1963  Ness et al. ─ Parker spiral 자기장 구조 확인
  │         Parker spiral magnetic field structure confirmed
```

- **Chapman (1957)**은 정적 코로나 모델을 제안했습니다: 고온의 코로나는 열전도에 의해 매우 먼 거리까지 뜨겁게 유지되므로, 정수압 평형 상태로 존재할 수 있다고 주장.
  Chapman proposed a static corona model: the hot corona extends far out via thermal conduction, maintaining hydrostatic equilibrium.
- **Parker의 핵심 반론**: Chapman의 정수압 해(解)를 무한대에서 평가하면 **유한한 압력**이 남는데, 이를 상쇄할 **성간 압력(interstellar pressure)**이 없으므로 정적 평형은 불가능. 반드시 팽창해야 함.
  Parker's key rebuttal: Chapman's hydrostatic solution gives **finite pressure at infinity**, with no counterbalancing **interstellar pressure** — so static equilibrium is impossible.

---

## 3. 필요한 배경 지식 / Prerequisites

### 3.1 이전 논문에서 배운 개념 / From Previous Papers

| 논문 / Paper | 핵심 개념 / Key Concept |
|---|---|
| #1 Birkeland (1908) | 태양에서 하전 입자 방출 → 오로라 / Solar charged particle emission → aurora |
| #2 Chapman & Ferraro (1931) | 플라즈마 덩어리의 지구 자기장 압축, 자기권 공동 / Plasma cloud compresses Earth's field, magnetospheric cavity |
| #3 Chapman & Bartels (1940) | 27일 재현, M-region (코로나 홀의 전조), Kp/Dst 지수 / 27-day recurrence, M-regions (coronal hole precursor), Kp/Dst indices |

### 3.2 새로 필요한 물리 / New Physics Needed

1. **유체역학 기초 / Basic Fluid Dynamics**
   - 연속 방정식 (continuity equation): 질량 보존 / Mass conservation
   - 운동 방정식 (equation of motion): 압력 구배 + 중력 / Pressure gradient + gravity
   - 정상 상태 흐름 (steady-state flow) / Steady-state flow

2. **기압 방정식 / Barometric Equation**
   - 정수압 평형: 압력 구배가 중력을 정확히 상쇄 / Hydrostatic balance: pressure gradient exactly balances gravity
   - $dp/dr = -\rho g$ 의 구면 대칭 버전 / Spherically symmetric version

3. **음속과 초음속 흐름 / Sound Speed and Supersonic Flow**
   - 이상 기체의 음속: $c_s = \sqrt{2kT/M}$ (수소의 경우 / for hydrogen)
   - 아음속 → 초음속 전이 (de Laval 노즐 유사) / Subsonic → supersonic transition (analogous to de Laval nozzle)

4. **열전도 / Thermal Conduction**
   - Spitzer 열전도도: $\kappa \propto T^{5/2}$ (완전 이온화 수소 / fully ionized hydrogen)
   - 코로나의 높은 온도가 먼 거리까지 유지되는 이유 / Why coronal high temperature persists to large distances

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 직관적 설명 / Intuitive Explanation |
|---|---|
| **Solar wind (태양풍)** | 태양 코로나에서 지속적으로 방출되는 초음속 플라즈마 흐름. 이 논문에서 "solar wind"라는 용어 자체는 사용되지 않으나 개념이 확립됨. A continuous supersonic plasma flow from the solar corona. The term itself is not used in this paper, but the concept is established. |
| **Hydrostatic equilibrium (정수압 평형)** | 압력 구배력이 중력을 정확히 상쇄하여 가스가 정지 상태를 유지하는 것. Chapman의 정적 코로나 모델의 기초. Pressure gradient exactly balances gravity, keeping gas stationary — basis of Chapman's static corona model. |
| **Stationary expansion (정상 팽창)** | 시간에 무관한 (정상 상태) 바깥 방향 흐름. Parker가 정수압 평형 대신 제안한 해. Time-independent (steady-state) outward flow — Parker's alternative to hydrostatic equilibrium. |
| **Critical point (임계점)** | 흐름이 아음속에서 초음속으로 전이하는 반경. de Laval 노즐의 목(throat)에 해당. The radius where flow transitions from subsonic to supersonic — analogous to a nozzle throat. |
| **Parker spiral (파커 나선)** | 태양 자전으로 인해 행성간 자기장선이 아르키메데스 나선 형태를 가지는 것. 태양풍이 방사 방향으로 나가면서 자기장선이 끌려가므로 나선이 형성됨. Interplanetary field lines form Archimedean spirals due to solar rotation — the radially outflowing wind drags the frozen-in field lines. |
| **Coronal temperature ($T_0$)** | 태양 코로나의 온도 (~$1$–$3 \times 10^6$ K). 이 값이 태양풍의 최종 속도를 결정하는 핵심 매개변수. Solar corona temperature — the key parameter determining final solar wind velocity. |
| **$\lambda$ (무차원 중력 매개변수)** | $\lambda = GM_\odot M / 2akT_0$ — 중력 에너지와 열 에너지의 비율. $\lambda > 2$이면 초음속 팽창이 가능. Ratio of gravitational to thermal energy. Supersonic expansion is possible when $\lambda > 2$. |
| **Frozen-in field (동결 자기장)** | 이상적 MHD에서 플라즈마와 자기장이 함께 움직이는 것. 태양풍이 자기장선을 바깥으로 끌고 나감. In ideal MHD, plasma and magnetic field move together — solar wind drags field lines outward. |
| **Interplanetary magnetic shell (행성간 자기 껍질)** | Parker가 예측한 태양계 내부를 감싸는 무질서한 자기장 영역 (~$10^{-5}$ gauss). 플라즈마 불안정에 의해 형성. Disordered magnetic field region (~$10^{-5}$ gauss) enclosing the inner solar system, formed by plasma instability. |

---

## 5. 수식 미리보기 / Equations Preview

### 5.1 정수압 평형과 그 실패 / Hydrostatic Equilibrium and Its Failure

정수압 평형 (eq. 1):

$$0 = \frac{d}{dr}(2NkT) + \frac{GM_\odot M N}{r^2}$$

온도 프로파일이 열전도에 의해 $T(r) = T_0(a/r)^{1/(n+1)}$로 주어지면, 밀도 해(eq. 4):

$$N(r) = N_0 \left(\frac{r}{a}\right)^{1/(n+1)} \exp\left\{\frac{\lambda(n+1)}{n}\left[\left(\frac{a}{r}\right)^{n/(n+1)} - 1\right]\right\}$$

여기서 / where $\lambda = GM_\odot M / 2akT_0$ (무차원 중력 매개변수 / dimensionless gravity parameter).

**핵심 문제 / The key problem**: $r \to \infty$에서 압력이 **유한한 값** $p(\infty) = p_0 \exp[-\lambda(n+1)/n]$으로 수렴합니다. 이온화 수소($n = 5/2$)의 경우 $p(\infty) \approx 10^{-3} p_0$ — 이를 상쇄할 성간 압력이 없으므로 **정수압 평형은 불가능**합니다.

At $r \to \infty$, the pressure converges to a **finite value**. For ionized hydrogen ($n = 5/2$), $p(\infty) \approx 10^{-3} p_0$ — with no interstellar pressure to counterbalance this, **hydrostatic equilibrium is impossible**.

### 5.2 정상 팽창의 운동 방정식 / Equation of Motion for Stationary Expansion

운동 방정식 (eq. 10):

$$NMv\frac{dv}{dr} = -\frac{d}{dr}(2NkT) - GM_\odot MN \frac{1}{r^2}$$

연속 방정식 (eq. 11):

$$\frac{d}{dr}(r^2 N v) = 0 \quad \Rightarrow \quad N(r)v(r) = N_0 v_0 \left(\frac{a}{r}\right)^2$$

무차원 변수 도입 ($\xi = r/a$, $\tau = T/T_0$, $\psi = Mv^2/2kT_0$) 후 무차원 운동 방정식 (eq. 13):

$$\frac{d\psi}{d\xi}\left(1 - \frac{\tau}{\psi}\right) = -2\xi^2 \frac{d}{d\xi}\left(\frac{\tau}{\xi^2}\right) - \frac{2\lambda}{\xi^2}$$

### 5.3 임계점 조건 / Critical Point Condition

정상 해가 존재하려면 **임계점**($\psi = 1$, 즉 $v = c_s$)에서 분자와 분모가 동시에 0이어야 합니다. 등온($\tau = 1$) 코로나의 경우, 임계점은 $\xi_c = \lambda/2$에서:

For a steady solution, the **critical point** ($\psi = 1$, i.e., $v = c_s$) requires numerator and denominator to vanish simultaneously. For an isothermal ($\tau = 1$) corona, the critical point is at $\xi_c = \lambda/2$:

$$\psi_0 - \ln\psi_0 = 2\lambda - 3 - 4\ln\frac{\lambda}{2} \quad \text{(eq. 16 — 초기 속도 결정 / determines initial velocity)}$$

### 5.4 Parker Spiral 자기장 / Parker Spiral Magnetic Field

태양풍 속도 $v_m$과 태양 자전 각속도 $\omega$에 의해, 자기장선의 유선(eq. 25):

$$\frac{r}{b} - 1 - \ln\left(\frac{r}{b}\right) = \frac{v_m}{b\omega}(\phi - \phi_0)$$

자기장 성분 (eq. 26):

$$B_r = B(\theta, \phi_0)\left(\frac{b}{r}\right)^2, \quad B_\theta = 0, \quad B_\phi = B(\theta, \phi_0)\frac{\omega}{v_m}(r-b)\left(\frac{b}{r}\right)^2 \sin\theta$$

$B_\phi / B_r$이 $r$에 비례하여 증가 → 먼 거리에서 자기장은 주로 **방위각 방향(나선)**. 자기장선이 반경 벡터와 45° 각도를 이루는 거리 (eq. 27):

$$r = \frac{v_m}{\omega}\sin\theta \approx 1 \text{ AU (지구 궤도 부근)}$$

---

## 6. 논문의 구조 / Paper Structure

| 절 / Section | 내용 / Content | 중요도 / Priority |
|---|---|---|
| I. Introduction | Biermann의 관측적 증거, 문제 설정 / Biermann's evidence, problem setup | ★★★ |
| II. Static Equilibrium | 정수압 평형의 실패 증명 / Proof that hydrostatic equilibrium fails | ★★★ |
| III. Stationary Expansion | 정상 팽창 해 도출, 임계점, Fig. 1 / Steady expansion solution, critical point | ★★★ |
| IV. Coronal Heating and Mass Loss | 코로나 가열 에너지와 질량 손실률 / Coronal heating energy and mass loss rate | ★★ |
| V. General Solar Magnetic Field | 태양풍이 자기장선을 끌고 나감 / Solar wind drags field lines outward | ★★★ |
| VI. Interplanetary Magnetic Field | Parker spiral 도출, 토크 / Parker spiral derivation, torque on Sun | ★★★ |
| VII. Plasma Instability | 행성간 자기 껍질, 불안정 / Interplanetary magnetic shell, instability | ★★ |

---

## 7. 읽으면서 주목할 질문들 / Questions to Keep in Mind

1. **정수압 평형이 왜 실패하는가?** — 무한대에서 유한한 압력이 남는다는 것이 물리적으로 무엇을 의미하는가?
   Why does hydrostatic equilibrium fail? What does finite pressure at infinity physically mean?

2. **임계점의 물리적 의미**: 흐름이 음속을 넘는 반경은 어디이며, 이것은 de Laval 노즐과 어떻게 유사한가?
   What is the physical meaning of the critical point? Where does flow become supersonic, and how is this analogous to a de Laval nozzle?

3. **코로나 온도와 태양풍 속도의 관계**: Fig. 1에서 $T_0$가 높을수록 속도가 빠른 이유는?
   Why does higher $T_0$ give faster wind speed in Fig. 1?

4. **Parker spiral의 기하학**: 지구 궤도(1 AU)에서 자기장선은 반경 방향과 약 45°를 이루는데, 이것이 의미하는 바는?
   At Earth's orbit, field lines make ~45° with the radial direction — what does this imply?

5. **이 논문이 Chapman-Ferraro 이론을 어떻게 변형하는가?** — Paper #2의 "간헐적 플라즈마 덩어리"가 "연속적 흐름"으로 바뀌면 자기권의 성질이 어떻게 달라지는가?
   How does this paper modify the Chapman-Ferraro picture? What changes when "intermittent clouds" become "continuous flow"?

---

## 8. 핵심 인물 / Key Figure

### Eugene N. Parker (1927–2022)
- 미국의 천체물리학자, 시카고 대학 교수 (1955–1995)
  American astrophysicist, University of Chicago professor
- **태양풍(solar wind)** 이론의 창시자 — 이 용어 자체를 만든 사람
  Founder of **solar wind** theory — coined the term itself
- 이 논문은 처음에 **2명의 심사자가 모두 거부**했으나, Astrophysical Journal 편집장 **Subrahmanyan Chandrasekhar**가 심사자들을 무시하고 출판을 결정
  Both reviewers rejected this paper, but editor **Chandrasekhar** overruled them
- 2018년 NASA **Parker Solar Probe** 발사 — 살아있는 과학자의 이름을 딴 최초의 NASA 미션
  NASA's **Parker Solar Probe** (2018) — first NASA mission named after a living scientist
- 2020년 **Crafoord Prize**, 태양물리학의 "노벨상"
  2020 **Crafoord Prize**, the "Nobel of solar physics"

---

## 9. 다음 논문과의 연결 / Connection to Next Papers

| 이 논문의 기여 / This paper's contribution | 이후 논문에서의 발전 / Later development |
|---|---|
| 태양풍의 존재 예측 / Solar wind existence predicted | #5 Van Allen (1958): 방사선대 발견 — 태양풍이 자기권에 갇힌 입자를 공급 / Radiation belts — solar wind supplies trapped particles |
| 자기권이 상시 구조 / Magnetosphere is permanent | #6 Dungey (1961): 연속적 태양풍 하에서 자기 재결합 → 열린 자기권 / Magnetic reconnection under continuous solar wind → open magnetosphere |
| Parker spiral 자기장 / Parker spiral field | #11 Burton et al. (1975): IMF $B_z$ (나선의 수직 성분)가 Dst를 결정 / IMF $B_z$ (vertical component of spiral) determines Dst |
| 27일 재현의 물리적 설명 / Physical explanation of 27-day recurrence | 코로나 홀 → 고속 태양풍 → 재현 / Coronal holes → fast solar wind → recurrence |
| 코로나 가열 문제 제기 / Coronal heating problem raised | 현재까지 미해결 — Parker Solar Probe의 핵심 과학 목표 / Still unsolved — key science objective of Parker Solar Probe |
