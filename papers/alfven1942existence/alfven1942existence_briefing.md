---
title: "Pre-reading Briefing: Existence of Electromagnetic-Hydrodynamic Waves"
paper: "08_alfven_1942"
authors: Hannes Alfvén
year: 1942
journal: "Nature, Vol. 150, pp. 405–406"
type: briefing
date: 2026-04-09
---

# Pre-reading Briefing / 사전 읽기 브리핑

## Existence of Electromagnetic-Hydrodynamic Waves
**Hannes Alfvén (1942)** — *Nature*, Vol. 150, pp. 405–406

---

## 핵심 기여 / Core Contribution

Hannes Alfvén은 이 짧은 2페이지 Nature 단신에서, 전도성 유체(conducting fluid) 속에서 자기장 선을 따라 전파되는 새로운 종류의 파동이 존재함을 이론적으로 예측했습니다. 이 파동은 자기장의 장력(magnetic tension)이 복원력 역할을 하며, 마치 기타 줄의 진동처럼 자기장 선을 따라 횡파(transverse wave)로 전파됩니다. 이후 "Alfvén wave"라 명명된 이 파동은 태양 코로나, 태양풍, 지구 자기권, 핵융합 플라즈마 등 거의 모든 우주 플라즈마 환경에서 관측되며, 현대 자기유체역학(MHD)의 기초가 되었습니다. Alfvén은 이 업적을 포함하여 1970년 노벨 물리학상을 수상했습니다.

In this short 2-page Nature letter, Hannes Alfvén theoretically predicted the existence of a new type of wave propagating along magnetic field lines in conducting fluids. These waves are driven by magnetic tension as the restoring force, propagating as transverse waves along field lines — much like vibrations on a guitar string. Later named "Alfvén waves," they are observed in nearly every cosmic plasma environment: the solar corona, solar wind, Earth's magnetosphere, and fusion plasmas. They became foundational to modern magnetohydrodynamics (MHD). Alfvén received the 1970 Nobel Prize in Physics in part for this work.

---

## 역사적 맥락 / Historical Context

### 1942년의 태양 물리학 / Solar Physics in 1942

1940년대 초, 태양 물리학은 중대한 전환점에 서 있었습니다:

In the early 1940s, solar physics stood at a critical juncture:

| 시기 / Period | 발견 / Discovery | 의미 / Significance |
|---|---|---|
| 1908 (Paper #5) | Hale — 흑점 자기장 발견 / Sunspot magnetic fields | 태양이 강한 자기장을 가짐을 증명 / Proved the Sun has strong magnetic fields |
| 1925 (Paper #7) | Hale & Nicholson — 극성 법칙 / Polarity law | 22년 자기 주기 발견 / Discovered the 22-year magnetic cycle |
| 1930s | Cowling — 자기장 소멸 시간 / Field decay time | 태양 자기장이 단순 확산보다 오래 지속됨 / Solar magnetic fields persist longer than simple diffusion predicts |
| **1942** | **Alfvén — MHD 파동 예측** / **MHD wave prediction** | **자기장과 플라즈마의 동적 상호작용** / **Dynamic interaction between fields and plasma** |

핵심 문제는: 태양 내부의 전도성 유체에서 자기장은 어떻게 행동하는가? 당시 물리학자들은 전자기학과 유체역학을 별개의 분야로 다루었습니다. Alfvén은 이 두 분야를 통합하여 전도성 유체 속 자기장의 역학을 기술하는 새로운 물리학 — magnetohydrodynamics (MHD) — 을 창시했습니다.

The key question was: how do magnetic fields behave in the Sun's conducting fluid interior? At the time, physicists treated electromagnetism and fluid dynamics as separate disciplines. Alfvén unified them, founding a new physics — magnetohydrodynamics (MHD) — describing the dynamics of magnetic fields in conducting fluids.

### "Frozen-in" 개념의 탄생 / Birth of the "Frozen-in" Concept

Alfvén의 핵심 통찰은 **"frozen-in" 조건**입니다: 전기 전도도가 매우 높은 유체(완전 전도체에 가까운)에서 자기장 선은 유체에 "얼어붙어(frozen-in)" 유체와 함께 움직입니다. 이것이 MHD 파동이 존재할 수 있는 물리적 기반입니다.

Alfvén's key insight was the **"frozen-in" condition**: in a highly conducting fluid (near-perfect conductor), magnetic field lines are "frozen into" the fluid and move with it. This is the physical basis allowing MHD waves to exist.

---

## 필요한 배경 지식 / Prerequisites

### 1. 전자기학 기초 / Electromagnetism Basics

**Maxwell 방정식 (관련 부분):**

$$\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t} \quad \text{(Faraday's law)}$$

$$\nabla \times \mathbf{B} = \mu_0 \mathbf{J} \quad \text{(Ampère's law, quasi-static)}$$

**Ohm의 법칙 (이동 전도체에서) / Ohm's law in a moving conductor:**

$$\mathbf{J} = \sigma(\mathbf{E} + \mathbf{v} \times \mathbf{B})$$

여기서 $\sigma$는 전기 전도도, $\mathbf{v}$는 유체 속도, $\mathbf{B}$는 자기장입니다.
Where $\sigma$ is electrical conductivity, $\mathbf{v}$ is fluid velocity, $\mathbf{B}$ is magnetic field.

### 2. 유체역학 기초 / Fluid Dynamics Basics

**운동 방정식 / Momentum equation:**

$$\rho \frac{\partial \mathbf{v}}{\partial t} = -\nabla p + \mathbf{J} \times \mathbf{B}$$

$\mathbf{J} \times \mathbf{B}$ 항이 Lorentz 힘으로, 자기장이 유체에 미치는 힘입니다.
The $\mathbf{J} \times \mathbf{B}$ term is the Lorentz force — the magnetic field's force on the fluid.

### 3. 자기압과 자기장력 / Magnetic Pressure and Tension

자기장의 응력 텐서(Maxwell stress tensor)에서 두 가지 힘이 나옵니다:

From the Maxwell stress tensor, two forces emerge:

- **자기압 / Magnetic pressure**: $p_B = \frac{B^2}{2\mu_0}$ — 자기장 선에 수직 방향으로 작용 / Acts perpendicular to field lines
- **자기장력 / Magnetic tension**: $\frac{B^2}{\mu_0 R}$ — 휘어진 자기장 선을 곧게 펴려는 힘 (R은 곡률 반경) / Force that straightens curved field lines (R is radius of curvature)

**Alfvén 파의 복원력은 바로 이 자기장력(magnetic tension)입니다!**

**The restoring force for Alfvén waves is precisely this magnetic tension!**

### 4. 이전 논문과의 연결 / Connection to Previous Papers

- **Paper #5 (Hale, 1908)**: 흑점의 자기장 — Alfvén 파가 전파될 "매체"가 존재함을 확인 / Sunspot magnetic fields — confirmed the "medium" for Alfvén wave propagation exists
- **Paper #7 (Hale & Nicholson, 1925)**: 자기 주기 — 태양 자기장이 체계적으로 변화함 / Magnetic cycle — solar magnetic fields change systematically

---

## 핵심 용어 / Key Vocabulary

| 용어 / Term | 직관적 설명 / Intuitive Explanation |
|---|---|
| **Magnetohydrodynamics (MHD)** | 전도성 유체와 자기장의 상호작용을 기술하는 물리학. "magneto(자기)" + "hydro(유체)" + "dynamics(역학)"의 합성어. / Physics describing the interaction of conducting fluids and magnetic fields. |
| **Alfvén wave** | 자기장 선을 따라 전파되는 횡파. 기타 줄의 진동과 비슷 — 장력(magnetic tension)이 복원력. / Transverse wave propagating along magnetic field lines. Like a vibrating guitar string — tension is the restoring force. |
| **Frozen-in condition** | 전도도가 높은 유체에서 자기장 선이 유체에 "얼어붙어" 함께 움직이는 현상. / In highly conducting fluids, field lines are "frozen into" the fluid and move with it. |
| **Conducting fluid** | 전류가 흐를 수 있는 유체. 태양 내부의 플라즈마, 지구 외핵의 액체 금속 등. / A fluid that can carry electric current: solar plasma, liquid metal in Earth's outer core, etc. |
| **Magnetic tension** | 휘어진 자기장 선이 곧게 펴지려는 힘. 고무줄의 탄성력과 유사. / Force that straightens curved field lines — analogous to elastic tension in a rubber band. |
| **Magnetic pressure** | 자기장이 주변을 밀어내는 압력 ($B^2/2\mu_0$). 가스 압력과 유사한 역할. / Pressure exerted by the magnetic field ($B^2/2\mu_0$), analogous to gas pressure. |
| **Alfvén speed** | Alfvén 파의 전파 속도: $v_A = B/\sqrt{\mu_0 \rho}$. 자기장이 강할수록, 밀도가 낮을수록 빠름. / Propagation speed of Alfvén waves. Faster with stronger fields and lower density. |
| **Incompressible perturbation** | 밀도 변화 없이 유체가 변형되는 섭동. Alfvén 파는 비압축성. / A perturbation without density changes. Alfvén waves are incompressible. |
| **Transverse wave** | 파동 전파 방향에 수직으로 진동하는 파동. / Wave oscillating perpendicular to the propagation direction. |

---

## 수식 미리보기 / Equations Preview

### 핵심 결과: Alfvén 파 속도 / Key Result: Alfvén Wave Speed

논문의 가장 중요한 결과는 Alfvén 파의 전파 속도입니다:

The paper's most important result is the Alfvén wave propagation speed:

$$\boxed{v_A = \frac{B}{\sqrt{\mu_0 \rho}}}$$

여기서:
- $B$ = 자기장 세기 (T) / magnetic field strength
- $\mu_0$ = 진공 투자율 ($4\pi \times 10^{-7}$ H/m) / permeability of free space
- $\rho$ = 유체 밀도 (kg/m³) / fluid density

### 유도 과정 미리보기 / Derivation Preview

**Step 1**: 균일한 자기장 $\mathbf{B}_0 = B_0 \hat{z}$와 정지 유체를 가정합니다.

Assume a uniform magnetic field $\mathbf{B}_0 = B_0 \hat{z}$ and a fluid at rest.

**Step 2**: 작은 섭동(perturbation)을 가합니다: $\mathbf{v} = \delta\mathbf{v}$, $\mathbf{B} = \mathbf{B}_0 + \delta\mathbf{B}$

Apply small perturbations: $\mathbf{v} = \delta\mathbf{v}$, $\mathbf{B} = \mathbf{B}_0 + \delta\mathbf{B}$

**Step 3**: 선형화된 운동 방정식과 유도 방정식:

Linearized momentum and induction equations:

$$\rho_0 \frac{\partial \delta\mathbf{v}}{\partial t} = \frac{1}{\mu_0}(\mathbf{B}_0 \cdot \nabla)\delta\mathbf{B}$$

$$\frac{\partial \delta\mathbf{B}}{\partial t} = (\mathbf{B}_0 \cdot \nabla)\delta\mathbf{v}$$

**Step 4**: 이 두 식을 결합하면 파동 방정식을 얻습니다:

Combining these gives the wave equation:

$$\frac{\partial^2 \delta\mathbf{v}}{\partial t^2} = v_A^2 \frac{\partial^2 \delta\mathbf{v}}{\partial z^2}$$

이것은 속도 $v_A = B_0/\sqrt{\mu_0 \rho_0}$로 $z$ 방향(자기장 방향)을 따라 전파되는 횡파의 표준 파동 방정식입니다!

This is the standard wave equation for a transverse wave propagating along the $z$-direction (field direction) at speed $v_A = B_0/\sqrt{\mu_0 \rho_0}$!

### 물리적 직관 / Physical Intuition

**기타 줄 비유 / Guitar String Analogy:**

| 기타 줄 / Guitar string | Alfvén 파 / Alfvén wave |
|---|---|
| 줄의 장력 $T$ / String tension | 자기장력 $B^2/\mu_0$ / Magnetic tension |
| 줄의 선밀도 $\mu$ / Linear mass density | 유체 밀도 $\rho$ / Fluid density |
| 파속 $v = \sqrt{T/\mu}$ / Wave speed | $v_A = B/\sqrt{\mu_0\rho}$ / Alfvén speed |
| 줄을 튕김 / Plucking the string | 자기장 선을 변위시킴 / Displacing the field line |

자기장 선을 "탄성이 있는 줄"처럼 생각하면, 이 줄을 옆으로 당겼다 놓으면 장력에 의해 진동이 줄을 따라 전파됩니다. 이것이 바로 Alfvén 파입니다.

Think of magnetic field lines as "elastic strings." If you pluck one sideways and release it, the tension causes the vibration to propagate along the string. That's an Alfvén wave.

### 태양에서의 Alfvén 속도 / Alfvén Speed in the Sun

| 태양 영역 / Solar region | $B$ (T) | $\rho$ (kg/m³) | $v_A$ (km/s) |
|---|---|---|---|
| 광구 / Photosphere | 0.1–0.3 | ~$10^{-4}$ | ~10–30 |
| 코로나 / Corona | $10^{-3}$–$10^{-2}$ | ~$10^{-12}$ | ~1,000–10,000 |
| 흑점 / Sunspot (umbra) | 0.2–0.4 | ~$10^{-4}$ | ~20–40 |
| 태양풍 (1 AU) / Solar wind | ~$5 \times 10^{-9}$ | ~$10^{-20}$ | ~50 |

코로나에서 Alfvén 속도가 매우 빠른 것에 주목하세요. 이것이 Alfvén 파가 코로나 가열의 유력한 후보인 이유 중 하나입니다.

Note the very fast Alfvén speed in the corona. This is one reason Alfvén waves are a leading candidate for coronal heating.

---

## 읽기 포인트 / Reading Points

이 논문을 읽을 때 주목할 점들:

Key things to watch for when reading:

1. **Alfvén의 핵심 가정**: 완전 전도체(perfect conductor)에서의 frozen-in 조건이 어떻게 파동의 존재를 이끌어내는지
   How the frozen-in condition in a perfect conductor leads to wave existence

2. **비압축성**: Alfvén 파가 왜 유체의 밀도를 변화시키지 않는 횡파인지
   Why Alfvén waves are transverse waves that don't change fluid density

3. **에너지 전달**: 자기장 에너지와 운동 에너지가 파동에서 어떻게 등분배되는지
   How magnetic and kinetic energy are equipartitioned in the wave

4. **태양 물리학 응용**: Alfvén이 태양 흑점과의 연관성을 어떻게 언급하는지
   How Alfvén mentions the connection to sunspots

5. **역사적 맥락**: 이 논문이 발표될 당시 학계의 회의적 반응 (특히 Chapman과 Cowling의 반대)
   The skeptical reception at the time (especially from Chapman and Cowling)

---

## 논문의 역사적 수용 / Historical Reception

흥미롭게도, Alfvén의 논문은 처음에 학계에서 **강한 저항**에 부딪혔습니다:

Interestingly, Alfvén's paper initially met **strong resistance** from the community:

- **Sydney Chapman** (당시 가장 영향력 있는 지구물리학자)과 **Thomas Cowling**은 MHD 파동의 존재를 수년간 인정하지 않았습니다.
  Chapman (the most influential geophysicist at the time) and Cowling did not accept MHD waves for years.

- Alfvén의 논문은 처음에 여러 저널에서 거절당했고, 결국 Nature의 짧은 단신으로만 출판되었습니다.
  The paper was initially rejected by several journals and was ultimately published only as a short Nature letter.

- 1948년 Alfvén이 Fermi를 만나 직접 설명한 후에야 Fermi가 "물론 맞다(of course)"라고 인정했고, 그 후 학계가 빠르게 수용했습니다.
  Only after Alfvén met Fermi in 1948 and explained it in person did Fermi say "of course," after which the community rapidly accepted it.

이 역사는 과학에서 패러다임 변화가 얼마나 어려운지를 보여주는 좋은 사례입니다.

This history is a great example of how difficult paradigm shifts can be in science.

---

## 다음 단계 / Next Steps

논문을 읽으신 후 질문이 있으시면 언제든 물어보세요. Q&A 내용은 이 파일에 추가됩니다.

After reading the paper, feel free to ask any questions. Q&A content will be appended to this file.
