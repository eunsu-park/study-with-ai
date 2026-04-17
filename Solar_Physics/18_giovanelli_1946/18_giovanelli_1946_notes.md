---
title: "A Theory of Chromospheric Flares"
authors: Ronald G. Giovanelli
year: 1946
journal: "Nature"
doi: "10.1038/158081a0"
topic: Solar_Physics
tags: [solar flares, magnetic reconnection, neutral point, chromosphere, sunspots, MHD]
status: completed
date_started: 2026-04-17
date_completed: 2026-04-17
---

# 18. A Theory of Chromospheric Flares / 채층 플레어 이론

---

## 1. Core Contribution / 핵심 기여

Giovanelli는 태양 채층 플레어의 에너지원에 대한 최초의 자기장 기반 이론을 제시했다. 그는 흑점이 성장하면서 유도되는 전기장이 자기 중성점(magnetic neutral point) 근처에서 전자를 가속시키고, 가속된 전자가 수소 원자와 충돌하여 여기 및 이온화를 일으킴으로써 플레어의 밝기 증가가 발생한다고 제안했다. 이것은 후에 "magnetic reconnection"이라 불리게 되는 개념의 기원이며, 현대 태양 물리학에서 플레어와 CME(코로나 질량 방출)를 설명하는 가장 근본적인 메커니즘이다.

Giovanelli presented the first magnetic-field-based theory for the energy source of solar chromospheric flares. He proposed that as sunspots grow, the induced electric fields accelerate electrons near magnetic neutral points — locations where the magnetic field vanishes. These accelerated electrons then collide with hydrogen atoms, causing excitation and ionization that produce the observed brightening. This was the conceptual origin of what would later be called "magnetic reconnection," now recognized as the most fundamental mechanism explaining flares and coronal mass ejections (CMEs) in modern solar physics.

---

## 2. Reading Notes / 읽기 노트

### Part I: Observational Basis / 관측적 근거 (p. 81, col. 2, para. 1–2)

Giovanelli는 이미 확립된 관측 사실에서 출발한다:
Giovanelli begins from well-established observational facts:

- 채층 플레어는 흑점과 밀접히 연관된다 — 흑점군 근처에서 발생 확률이 높고, 흑점군의 크기가 클수록 확률이 높다
  Chromospheric flares are closely associated with sunspots — the probability of a flare increases with the size of the sunspot group
- 자기적으로 복잡한 $\beta\gamma$- 및 $\gamma$-형 흑점군에서 더 자주 발생하며, 단순한 $\alpha$- 및 $\beta$-형보다 빈도가 높다
  More frequent in magnetically complex $\beta\gamma$- and $\gamma$-type groups than simpler $\alpha$- and $\beta$-types
- 플레어는 수명이 짧고(약 30분), 상당히 국소적이며, 태양 표면 위 또는 표면을 가로질러 이동하는 속도를 보이지 않는다
  Flares are short-lived (~30 min), quite localized, and show no velocity either in height or across the solar surface
- 이러한 관측 특성은 플레어가 **자기장과 관련된 에너지 방출 과정**이라는 것을 강하게 시사한다
  These observational characteristics strongly suggest that flares are a **magnetic-field-related energy release process**

### Part II: The Mechanism — Induced Electric Fields / 메커니즘 — 유도 전기장 (p. 81, col. 2, para. 3–end)

Giovanelli의 핵심 아이디어는 다음과 같다:
Giovanelli's key idea is as follows:

1. **흑점이 성장하면** 자기선속(magnetic flux)이 증가하고, 이에 따라 **유도 전기장**이 발생한다
   As a sunspot **grows**, the magnetic flux increases, producing an **induced electric field**

2. 흑점 주변에서 유도되는 전기장과 자기장을 계산하기 위해, 흑점을 **전류 코일(current coil)**로 모델링한다
   To calculate the induced electric and magnetic fields, the sunspot is modeled as a **current coil**

3. 구체적 수치: 흑점이 50시간에 걸쳐 균일하게 성장하여 직경 $7 \times 10^9$ cm, 최대 자기장 2,000 gauss에 도달한다고 가정
   Specific values: a sunspot growing uniformly over 50 hours to a diameter of $7 \times 10^9$ cm with a maximum field of 2,000 gauss

4. 계산 결과를 표로 제시:
   Results presented in a table:

| 중심으로부터 거리 / Distance from centre | 자기장 / Magnetic field (gauss) | 유도 전기장 / Electric field (volt/cm) |
|---|---|---|
| $3.5 \times 10^9$ cm | 8 | $1.55 \times 10^{-3}$ |
| $1.75 \times 10^9$ cm | 64 | $6.2 \times 10^{-3}$ |

5. Chapman과 Cowling의 결과를 인용: 교차된 전기장과 자기장에서 하전 입자는 전기장 방향으로 drift velocity를 갖는다:
   Citing Chapman and Cowling: in crossed electric and magnetic fields, charged particles have a drift velocity in the direction of the electric field:

$$v = \frac{Ee\tau}{m(1 + \omega^2\tau^2)}$$

여기서 $\tau$는 충돌 간 평균 시간, $\omega$는 자이로 주파수이다.
where $\tau$ is the mean time between collisions and $\omega$ is the gyro-frequency.

### Part III: Energy Conditions for Excitation / 여기를 위한 에너지 조건 (p. 82, col. 1, para. 1–5)

전자가 수소 원자의 첫 번째 이온화 포텐셜에 해당하는 에너지를 얻으려면 다음 조건을 충족해야 한다:
For electrons to acquire energy equal to the first ionization potential of hydrogen:

$$\frac{E^2 \lambda^2}{1 + 8.8 \times 10^{-3} H^2 \lambda^2} \geq 2 \times 10^{15}$$

여기서:
where:
- $\lambda$: 평균자유행로 (mean free path) — E.M.U. 단위
- $E$: 전기장 (E.M.U.)
- $H$: 자기장 (gauss)

**핵심 결과**: $H = 0$ (중성점)이면 $\lambda \geq 4.5 \times 10^7 / E$이다. $E = 10^{-3}$ volt/cm일 때 $\lambda \geq 450$ cm — 이는 채층 중간 높이(약 6,000 km)의 평균자유행로에 해당한다.
**Key result**: If $H = 0$ (neutral point), then $\lambda \geq 4.5 \times 10^7 / E$. For $E = 10^{-3}$ volt/cm, $\lambda \geq 450$ cm — this corresponds to the mean free path at mid-chromosphere (~6,000 km).

반면 $H$가 큰 경우 ($8.8 \times 10^{-3} H^2 \lambda^2 \gg 1$), 훨씬 약한 자기장에서만 여기가 가능하다: $H \leq E / (1.32 \times 10^6) \leq 7.5 \times 10^{-2}$ gauss.
If $H$ is large, excitation requires much weaker fields: $H \leq 7.5 \times 10^{-2}$ gauss.

**결론**: 유도 전기장에 의한 전자 여기는 **자기장이 0인 중성점** 근처에서만 효과적으로 일어난다.
**Conclusion**: Electron excitation by the induced electric field is effective only near **neutral points where the magnetic field is zero**.

### Part IV: Chromospheric Conductivity / 채층 전도도 (p. 82, col. 1–2)

Giovanelli는 Cowling의 채층 전도도 계산을 인용한다:
Giovanelli cites Cowling's calculations of chromospheric conductivity:

$$\sigma^l + i\sigma^{II} = \{6.8 \times 10^{13} \bar{Z} T^{-3/2} - i \, 8.6 \times 10^3 HT / \rho_e\}^{-1} \text{ E.M.U.}$$

여기서:
where:
- $\bar{Z}$: 평균 이온화도 (mean degree of ionization)
- $T$: 전자 온도 (electron temperature)
- $\rho_e$: 전자 압력 (electron pressure)
- $\sigma^l$, $\sigma^{II}$: 직접 및 횡방향 전도도 (direct and transverse conductivities)

채층 전체에서 직접 전도도는 약 $10^{-8}$ E.M.U. 수준이다. 매우 작은 자기장(0.1 gauss 이상)에서도 두 전도도가 비슷해지면서 무시할 수 있게 된다.
Throughout the chromosphere, the direct conductivity is of the order of $10^{-8}$ E.M.U. For magnetic fields greater than ~0.1 gauss, both conductivities become comparable and negligible.

**핵심 함의**: 자기장이 0이 아닌 곳에서는 전도도가 자기장선 방향으로만 유의미하다. 따라서 중성점 근처에서만 전자가 자유롭게 가속될 수 있다.
**Key implication**: Where the field is non-zero, conductivity is significant only along field lines. Electrons can be freely accelerated only near neutral points.

### Part V: Neutral Point Physics / 중성점 물리 (p. 82, col. 2, para. 1–4)

외부 자기장이 흑점 자기장에 대해 경사져 있으면, 중성점은 흑점 위 또는 아래에 위치한다 (흑점과 외부 장의 극성 관계에 따라).
If the external magnetic field is inclined to the sunspot field, the neutral point lies either above or below the spot (depending on the polarity relationship).

중성점이 채층에 있으면:
If the neutral point is in the chromosphere:

1. 전자는 전기장의 영향 아래 자기장 방향으로 이동하도록 구속됨
   Electrons are constrained to move along the magnetic field under the influence of the electric field
2. 중성점에서 벗어나면 자기장이 충분히 강해져 전자를 자기력선에 속박
   Moving away from the neutral point, the field becomes strong enough to confine electrons to field lines
3. 전자는 결국 **역전층(reversing layer)**에 도달 — 전도도가 높아 공간 전하 축적을 방지하는 영역
   Electrons eventually reach the **reversing layer** — a region where conductivity is high enough to prevent space charge accumulation

자기력선을 따라 흑점으로 들어가는 전자에 의해 축적된 편극 장(polarization fields)은 전기 세기를 증가시키는 경향이 있으며, 중성점 근처에서의 전기장 세기 감소를 일으키지 않는다.
Polarization fields built up by electrons flowing into the sunspot along field lines tend to increase the electric intensity and cause no blockage to the mechanism near the neutral point.

### Part VI: Flare Properties Explained / 설명되는 플레어 특성 (p. 82, col. 2, para. 5–end)

Giovanelli는 자신의 이론이 플레어의 주요 관측 특성을 설명할 수 있다고 주장한다:
Giovanelli argues his theory explains the key observed properties of flares:

1. **위치 (Location)**: 채층에서의 국소화된 밝기 증가 — 중성점이 채층에 위치할 때 발생
   Localized brightening in the chromosphere — occurs when the neutral point is located in the chromosphere

2. **정상성 (Stationarity)**: 이동 속도를 보이지 않음 — 중성점은 기하학적으로 결정되는 고정된 위치
   No velocity observed — the neutral point is a geometrically determined fixed location

3. **흑점과의 연관 (Association with sunspots)**: 흑점 근처에서 발생 — 흑점의 자기장이 메커니즘의 핵심
   Occurs near sunspots — the sunspot's magnetic field is central to the mechanism

4. **일시성 (Transient nature)**: 여러 가능한 원인 — 흑점 성장률의 일시적 증가, 자기장의 변화, 중성점 위치의 이동
   Multiple possible causes — temporary increases in sunspot growth rate, changes in magnetic field, movement of neutral point location

5. **$\gamma$-형 선호 ($\gamma$-type preference)**: $\gamma$-형은 $\beta$-형보다 중성점의 수가 더 많으므로 플레어가 더 빈번
   $\gamma$-types have more neutral points than $\beta$-types, hence more frequent flares

---

## 3. Key Takeaways / 핵심 시사점

1. **자기 에너지가 플레어의 에너지원이다** — Giovanelli는 열적·운동학적 에너지가 아닌, 흑점의 자기장에 저장된 에너지가 플레어를 구동한다는 것을 최초로 명확히 제안했다. 이는 현대 태양 물리학의 기본 패러다임이 되었다.
   **Magnetic energy is the energy source for flares** — Giovanelli was the first to clearly propose that the energy stored in sunspot magnetic fields, not thermal or kinetic energy, drives flares. This became the foundational paradigm of modern solar physics.

2. **중성점은 에너지 방출의 특별한 장소이다** — 자기장이 0인 중성점에서만 전자가 자유롭게 가속될 수 있다. 자기장이 있는 곳에서는 전자가 자기력선에 속박되어 충분한 에너지를 얻지 못한다.
   **Neutral points are special sites for energy release** — Only at neutral points (where $B = 0$) can electrons be freely accelerated. Where a magnetic field exists, electrons are confined to field lines and cannot gain sufficient energy.

3. **전도도의 비등방성이 핵심 역할을 한다** — 채층에서 자기장이 있으면 전도도가 자기장선 방향으로만 유의미하다. 이 비등방성이 중성점을 특별하게 만든다.
   **Anisotropy of conductivity plays a key role** — In the chromosphere, the presence of a magnetic field makes conductivity significant only along field lines. This anisotropy makes neutral points special.

4. **관측 특성과의 일관성** — Giovanelli의 이론은 플레어의 위치, 정상성, 일시성, 흑점과의 연관성을 모두 자연스럽게 설명한다. 이는 이론의 설득력을 높인다.
   **Consistency with observational properties** — The theory naturally explains flare location, stationarity, transience, and sunspot association, increasing its persuasiveness.

5. **번개 방전과의 유추** — Giovanelli는 플레어를 대기 중 번개의 전기 방전에 비유했다. 이 유추는 에너지 축적-방출의 기본 구조를 이해하는 데 유용하지만, 실제 메커니즘은 MHD 과정으로 훨씬 복잡하다.
   **Lightning discharge analogy** — Giovanelli analogized flares to atmospheric lightning. This analogy is useful for understanding the energy accumulation-release structure, but the actual mechanism involves much more complex MHD processes.

6. **"reconnection"이라는 용어 없이 재결합을 제안** — Giovanelli는 이 논문에서 반대 극성 자기장이 만나는 중성점에서의 에너지 방출을 기술했지만, "reconnection"이라는 용어는 사용하지 않았다. 이 용어는 7년 후 Dungey (1953)에 의해 도입된다.
   **Proposed reconnection without the term** — Giovanelli described energy release at neutral points where opposite-polarity fields meet, but did not use the word "reconnection." The term was introduced by Dungey seven years later (1953).

7. **짧지만 영향력이 거대한 논문** — 단 2페이지의 Nature letter이지만, 태양 물리학의 가장 근본적인 질문 "플레어는 어떻게 에너지를 얻는가?"에 대한 패러다임 전환을 일으켰다.
   **Short but enormously influential** — A mere 2-page Nature letter, yet it triggered a paradigm shift on solar physics' most fundamental question: "How do flares get their energy?"

8. **후속 연구의 토대** — 이 논문은 Sweet-Parker (1957–58), Petschek (1964), 그리고 현대의 collisionless reconnection 연구에 이르는 자기 재결합 이론의 출발점이 되었다.
   **Foundation for subsequent research** — This paper became the starting point for reconnection theory, leading to Sweet-Parker (1957–58), Petschek (1964), and modern collisionless reconnection studies.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 하전 입자의 Drift Velocity / Drift Velocity of Charged Particles

교차된 전기장 $E$와 자기장 $H$ 하에서 하전 입자의 전기장 방향 drift velocity:
Drift velocity of charged particles in the direction of the electric field under crossed $E$ and $H$ fields:

$$v = \frac{Ee\tau}{m(1 + \omega^2\tau^2)}$$

- $e$: 전자 전하 / electron charge
- $\tau$: 충돌 간 평균 시간 / mean time between collisions
- $\omega = eH/mc$: 자이로(사이클로트론) 주파수 / gyro(cyclotron) frequency
- $m$: 전자 질량 / electron mass

**물리적 의미**: $H = 0$ (중성점)이면 $\omega = 0$이므로 $v = Ee\tau/m$ — 전자는 전기장 방향으로 자유롭게 가속된다. $H$가 크면 $\omega\tau \gg 1$이 되어 drift가 억제된다.
**Physical meaning**: At $H = 0$ (neutral point), $\omega = 0$ so $v = Ee\tau/m$ — electrons accelerate freely along the electric field. For large $H$, $\omega\tau \gg 1$ and the drift is suppressed.

### 4.2 수소 이온화 조건 / Hydrogen Ionization Condition

전자가 충돌 사이에 수소의 첫 번째 이온화 에너지를 얻기 위한 조건:
Condition for an electron to acquire the first ionization energy of hydrogen between collisions:

$$\frac{E^2 \lambda^2}{1 + 8.8 \times 10^{-3} H^2 \lambda^2} \geq 2 \times 10^{15}$$

- $\lambda$: 평균자유행로 (E.M.U. 단위) / mean free path (in E.M.U.)
- $E$: 전기장 (E.M.U.) / electric field (E.M.U.)
- $H$: 자기장 (gauss) / magnetic field (gauss)

**두 가지 극한 (Two limiting cases)**:

(a) $H = 0$ (중성점 / neutral point):

$$\lambda \geq \frac{4.5 \times 10^7}{E}$$

$E = 10^{-3}$ volt/cm ($10^5$ E.M.U./cm) 일 때 $\lambda \geq 450$ cm — 채층 중간(~6,000 km)의 평균자유행로에 해당.
For $E = 10^{-3}$ volt/cm, $\lambda \geq 450$ cm — the mean free path at mid-chromosphere (~6,000 km).

(b) $8.8 \times 10^{-3} H^2 \lambda^2 \gg 1$ (강한 자기장 / strong magnetic field):

$$H \leq \frac{E}{1.32 \times 10^{6}} \leq 7.5 \times 10^{-2} \text{ gauss}$$

→ 사실상 0에 가까운 자기장에서만 가능. 흑점 근처의 수백~수천 gauss에서는 불가능.
→ Only possible for fields nearly zero. Impossible near sunspots with hundreds to thousands of gauss.

### 4.3 채층 전도도 / Chromospheric Conductivity

Cowling (1945)의 채층 전도도 공식:
Cowling's (1945) formula for chromospheric conductivity:

$$\sigma^l + i\sigma^{II} = \left\{6.8 \times 10^{13} \bar{Z} T^{-3/2} - i \, 8.6 \times 10^3 \frac{HT}{\rho_e}\right\}^{-1} \text{ E.M.U.}$$

- $\sigma^l$: 직접 전도도 (전기장 방향) / direct conductivity (along $E$)
- $\sigma^{II}$: 횡방향 전도도 (전기장에 수직) / transverse conductivity (perpendicular to $E$)
- $\bar{Z}$: 평균 이온화도 / mean degree of ionization
- $T$: 전자 온도 / electron temperature
- $\rho_e$: 전자 압력 / electron pressure

채층에서 직접 전도도 $\sim 10^{-8}$ E.M.U. 자기장 $> 0.1$ gauss에서는 횡방향 전도도와 비슷해지며 무시 가능.
In the chromosphere, direct conductivity $\sim 10^{-8}$ E.M.U. For fields $> 0.1$ gauss, both conductivities become comparable and negligible.

### 4.4 흑점의 유도 전기장 / Induced Electric Field of a Sunspot

흑점을 전류 코일로 모델링하여 유도 전기장 계산:
Modeling the sunspot as a current coil to calculate the induced electric field:

- 흑점 직경: $7 \times 10^9$ cm / Sunspot diameter: $7 \times 10^9$ cm
- 성장 시간: 50시간 (균일 성장) / Growth time: 50 hours (uniform growth)
- 최대 자기장: 2,000 gauss / Maximum magnetic field: 2,000 gauss
- 코일 반경의 5배 거리에서: $E \sim 10^{-3}$ volt/cm, $H \sim$ 수 gauss
  At 5× coil radius: $E \sim 10^{-3}$ volt/cm, $H \sim$ a few gauss

이 계산은 흑점 근처에서 중성점이 존재하는 영역(2.5–5× 반경)에서의 전기장이 전자를 여기시키기에 충분함을 보여준다.
This calculation shows that the electric field in the region where neutral points exist (2.5–5× radius from the sunspot) is sufficient to excite electrons.

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1859  ─── Carrington & Hodgson: 최초의 백색광 플레어 관측
          First white-light flare observation
          │
1908  ─── Hale (#5): 흑점 자기장 발견 (Zeeman 효과)
          Discovery of sunspot magnetic fields
          │
1939  ─── Giovanelli: 채층 플레어의 흑점 연관성 통계 연구
          Statistical study of flare-sunspot association
          │
1942  ─── Alfvén (#7): MHD 파동 이론
          MHD wave theory
          │
1942  ─── Hey: 태양 전파 방출 발견
          Discovery of solar radio emission
          │
1943  ─── Chapman & Cowling: 비균일 기체의 수학적 이론
          Mathematical Theory of Non-Uniform Gases
          │
1945  ─── Cowling: 태양 대기 전도도 계산
          Calculation of solar atmospheric conductivity
          │
>>>>  ─── 1946: Giovanelli (#18) — 자기 중성점에서의 플레어 이론 ◀ THIS PAPER
          Theory of flares at magnetic neutral points
          │
1947  ─── Giovanelli: 후속 논문 — 이론 확장 및 정교화
          Follow-up papers — theory extension and refinement
          │
1950  ─── Giovanelli: 전자 및 이온 속도 분포 연구
          Study of electron and ion velocity distributions
          │
1953  ─── Dungey: "magnetic reconnection" 용어 도입, 수학적 정립
          Introduction of "magnetic reconnection" term, mathematical formulation
          │
1957  ─── Sweet: 재결합 기하학의 정량적 모델
          Quantitative model of reconnection geometry
          │
1958  ─── Parker: Sweet-Parker 재결합 모델
          Sweet-Parker reconnection model
          │
1964  ─── Petschek: 빠른 재결합 모델
          Fast reconnection model
          │
2015  ─── NASA MMS: 자기권에서 reconnection 직접 관측
          Direct observation of reconnection in magnetosphere
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| #5 Hale (1908) — 흑점 자기장 발견 | Giovanelli의 이론은 Hale이 발견한 흑점 자기장을 플레어의 에너지원으로 직접 활용한다 / Giovanelli directly uses Hale's discovered sunspot magnetic fields as the energy source for flares | 선수 논문 — 흑점의 자기적 성질이 없으면 이 이론은 존재할 수 없다 / Prerequisite — without the magnetic nature of sunspots, this theory could not exist |
| #7 Alfvén (1942) — MHD 파동 | Alfvén의 MHD 이론은 플라즈마에서 자기장의 거동을 기술하는 기본 틀을 제공한다 / Alfvén's MHD theory provides the basic framework for magnetic field behavior in plasma | 이론적 배경 — Giovanelli는 MHD가 막 태동하던 시기에 자기장 기반 이론을 제안했다 / Theoretical background — Giovanelli proposed a magnetic-field-based theory when MHD was just emerging |
| Dungey (1953) — Magnetic reconnection | Giovanelli의 중성점 개념을 수학적으로 정립하고 "reconnection"이라는 용어를 도입 / Mathematically formalized Giovanelli's neutral point concept and introduced the term "reconnection" | 직접적 후속 — Giovanelli의 물리적 직관에 수학적 엄밀성을 부여 / Direct successor — gave mathematical rigor to Giovanelli's physical intuition |
| Sweet (1958) & Parker (1957) — Sweet-Parker model | 최초의 정량적 재결합 모델. 재결합 속도를 계산했으나 관측보다 느림 / First quantitative reconnection model. Calculated reconnection rate but too slow compared to observations | 이론 발전 — Giovanelli가 제안한 개념을 정량적으로 발전시킴 / Theory development — quantitatively advanced the concept proposed by Giovanelli |
| Petschek (1964) — Fast reconnection | 느린 충격파를 도입하여 빠른 재결합 모델 제시. 관측에 더 부합 / Introduced slow-mode shocks for a fast reconnection model, better matching observations | 문제 해결 — Sweet-Parker 모델의 "느린 재결합" 문제를 해결 / Problem resolution — solved the "slow reconnection" problem of the Sweet-Parker model |
| Carrington (1859) — 백색광 플레어 | 최초의 플레어 관측. Giovanelli가 설명하려 한 현상의 발견 / First flare observation. The phenomenon Giovanelli sought to explain | 관측적 출발점 / Observational starting point |

---

## 7. References / 참고문헌

- Giovanelli, R. G., "A Theory of Chromospheric Flares," *Nature*, 158, 81–82, 1946. [DOI: 10.1038/158081a0]
- Giovanelli, R. G., *Astrophys. J.*, 89, 555, 1939.
- Chapman, S., *Mon. Not. Roy. Ast. Soc.*, 103, 1117, 1943.
- Chapman, S. and Cowling, T. G., *"The Mathematical Theory of Non-Uniform Gases,"* Cambridge University Press, 1939.
- Cillié, G. G. and Menzel, D. H., Harvard College Observatory Circular 410, 1935.
- Cowling, T. G., *Proc. Roy. Soc.*, 183, 453, 1945.
- Thiessen, *Observatory*, 66, 230, 1946.
- Dungey, J. W., "Conditions for the Occurrence of Electrical Discharges in Astrophysical Systems," *Phil. Mag.*, 44, 725–738, 1953.
- Sweet, P. A., "The Neutral Point Theory of Solar Flares," *IAU Symposium 6*, 123–134, 1958.
- Parker, E. N., "Sweet's Mechanism for Merging Magnetic Fields in Conducting Fluids," *J. Geophys. Res.*, 62, 509–520, 1957.
- Petschek, H. E., "Magnetic Field Annihilation," *NASA SP-50*, 425–439, 1964.
