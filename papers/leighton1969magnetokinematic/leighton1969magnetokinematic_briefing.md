---
title: "Pre-reading Briefing: A Magneto-Kinematic Model of the Solar Cycle"
paper: "10_leighton_1969"
authors: Robert B. Leighton
year: 1969
journal: "The Astrophysical Journal, 156, 1–26"
type: briefing
date: 2026-04-10
---

# Pre-reading Briefing / 사전 읽기 브리핑

## A Magneto-Kinematic Model of the Solar Cycle
**Robert B. Leighton (1969)** — *The Astrophysical Journal*, 156, 1–26

---

## 핵심 기여 / Core Contribution

Leighton은 Babcock(1961)의 현상학적 태양 다이나모 모델을 **정량적 수학 모델**로 발전시켰습니다. Babcock이 "이런 일이 일어난다"고 서술적으로 기술한 것을, Leighton은 편미분방정식으로 정식화하여 컴퓨터로 풀었습니다. 핵심 혁신은 두 가지입니다: (1) **초립자 확산(supergranular diffusion)**: 태양 표면의 초립자 대류 셀(~30,000 km)이 자기장을 무작위 보행(random walk)시켜 확산시킨다는 발견을 모델에 도입. 확산 계수 $\kappa \sim 770$–$1540$ km²/s. (2) **자오선 흐름(meridional circulation)**: 적도에서 극으로의 느린 표면 흐름이 BMR의 후행 극성 flux를 극 방향으로 수송. 이 두 메커니즘을 Babcock의 차등 회전 + 자기 부력과 결합하여, 태양 자기 주기를 재현하는 최초의 **수치 시뮬레이션**을 수행했습니다. 이 모델은 "Babcock-Leighton 다이나모"로 불리며, 현대 flux transport dynamo 모델의 직접적 선조입니다.

Leighton developed Babcock's (1961) phenomenological model into a **quantitative mathematical model**. Where Babcock described "this happens," Leighton formulated it as PDEs and solved them numerically. Two key innovations: (1) **Supergranular diffusion**: the random walk of magnetic flux by supergranular convection cells (~30,000 km), with diffusivity $\kappa \sim 770$–$1540$ km²/s. (2) **Meridional circulation**: slow surface flow from equator to poles transporting following-polarity flux poleward. Combining these with Babcock's differential rotation and magnetic buoyancy, he performed the first **numerical simulation** reproducing the solar magnetic cycle. This "Babcock-Leighton dynamo" is the direct ancestor of modern flux transport dynamo models.

---

## 역사적 맥락 / Historical Context

| 시기 / Period | 발견 / Discovery | 의미 / Significance |
|---|---|---|
| 1961 (SP #9) | Babcock — 5단계 다이나모 모델 / 5-stage dynamo | 현상학적 모델, 정량적 수식 부족 / Phenomenological, lacking quantitative equations |
| 1962 (SP #13) | Leighton, Noyes & Simon — 5분 진동 발견 | 일진학의 시작 / Birth of helioseismology |
| 1964 | Leighton — 초립자 확산 발견 | 자기장 수송 메커니즘 / Magnetic flux transport mechanism |
| **1969** | **Leighton — 운동학적 다이나모** | **Babcock 모델의 수학적 정량화 / Mathematical quantification** |

Leighton은 이미 1962년에 태양 5분 진동을 발견한 관측 천문학자로(SP #13), 1964년에는 초립자 대류에 의한 자기장의 무작위 보행(random walk diffusion)을 발견했습니다. 이 관측 경험이 Babcock 모델에 빠져 있던 핵심 물리 — **flux가 어떻게 수송되는가** — 를 제공합니다.

Leighton, already known for discovering solar 5-minute oscillations (1962, SP #13), discovered random walk diffusion of magnetic flux by supergranular convection in 1964. This observational insight provided the missing physics in Babcock's model — **how flux is transported**.

---

## 필요한 배경 지식 / Prerequisites

### 1. Babcock 모델 (SP #9) — 복습 / Review

5단계: Poloidal → (차등 회전) → Toroidal → (자기 부력) → BMR → (Joy's law) → 극성 반전 → 반전된 Poloidal

핵심 미해결: **Stage 4에서 flux가 극 방향으로 "어떻게" 이동하는가?**
Babcock: "migration implies the existence of meridional flow" — 추측일 뿐.

Key unsolved: **"How" does flux migrate poleward in Stage 4?**
Babcock only speculated about meridional flow.

### 2. 초립자 대류 / Supergranulation

태양 표면의 대류 패턴:
- **입자(granulation)**: ~1,000 km, 수명 ~10분 / ~1,000 km, lifetime ~10 min
- **초립자(supergranulation)**: ~30,000 km, 수명 ~1일 / ~30,000 km, lifetime ~1 day

초립자 셀의 가장자리에 자기 flux가 축적됨(chromospheric network). 각 셀이 소멸하고 새로 형성될 때마다 flux가 무작위로 이동 → **random walk diffusion**.

Magnetic flux accumulates at supergranular cell boundaries (chromospheric network). As cells die and reform, flux undergoes random walk → **diffusion**.

**확산 계수 / Diffusion coefficient:**

$$\kappa = \frac{l^2}{4\tau}$$

여기서 $l \sim 30{,}000$ km (셀 크기), $\tau \sim 1$ day (수명).

$$\kappa \sim \frac{(3 \times 10^4)^2}{4 \times 86400} \approx 2600 \text{ km}^2/\text{s}$$

(Leighton의 논문에서는 $\kappa \sim 770$–$1540$ km²/s로 약간 낮은 값을 사용)

### 3. 축대칭 자기장의 분해 / Axisymmetric Field Decomposition

Leighton은 자기장을 두 성분으로 분해합니다:

- **$A(\theta, t)$**: Poloidal 장의 vector potential — 자오선 면의 자기장 / Poloidal field
- **$B(\theta, t)$**: Toroidal 장 — 동-서 방향 자기장 / Toroidal field

핵심 방정식:
$$\frac{\partial A}{\partial t} = \kappa \nabla^2 A + S_A \quad \text{(poloidal: diffusion + source)}$$
$$\frac{\partial B}{\partial t} = \text{(differential rotation)} \cdot \nabla A + \kappa \nabla^2 B + S_B \quad \text{(toroidal: winding + diffusion + source)}$$

### 4. 이전 논문과의 연결 / Connection to Previous Papers

- **SP #8 (Alfvén, 1942)**: Frozen-in + MHD — 모델의 물리적 기반 / Physical foundation
- **SP #9 (Babcock, 1961)**: 5단계 모델 — Leighton이 정량화하는 대상 / What Leighton quantifies
- **SP #13 (Leighton et al., 1962)**: 5분 진동 — 같은 저자의 관측 업적 / Same author's observational work

---

## 핵심 용어 / Key Vocabulary

| 용어 / Term | 직관적 설명 / Intuitive Explanation |
|---|---|
| **Kinematic dynamo** | 유체 운동(속도장)을 주어진 것으로 가정하고, 그 안에서 자기장이 어떻게 진화하는지만 계산하는 모델. 자기장이 유체 운동에 미치는 역효과(back-reaction)를 무시. / Model that takes fluid motion as given and computes magnetic field evolution, ignoring back-reaction. |
| **Supergranular diffusion** | 초립자 대류 셀에 의한 자기 flux의 무작위 보행 확산. 확산 계수 $\kappa \sim 10^{2}$–$10^{3}$ km²/s. / Random walk diffusion of magnetic flux by supergranular cells. |
| **Flux transport** | 확산 + 자오선 흐름에 의한 자기 flux의 표면 수송. Babcock-Leighton 다이나모의 핵심 요소. / Surface transport of flux by diffusion + meridional flow. |
| **Source term** | BMR 출현을 나타내는 수학적 항. Joy's law 기울기를 포함. / Mathematical term representing BMR emergence with Joy's law tilt. |
| **Poloidal source** | toroidal → poloidal 변환 (Babcock-Leighton 메커니즘). $\alpha$-효과와 다른 방식. / Toroidal → poloidal conversion via BMR tilt (not α-effect). |
| **$\Omega$-effect** | 차등 회전에 의한 poloidal → toroidal 변환. $\Omega$는 회전 각속도. / Poloidal → toroidal conversion by differential rotation. |

---

## 수식 미리보기 / Equations Preview

### Leighton의 핵심 방정식 / Leighton's Key Equations

**1. 표면 자기장의 확산-이류 방정식 / Surface field diffusion-advection:**

$$\frac{\partial B_r}{\partial t} = \frac{\kappa}{R^2}\left[\frac{1}{\sin\theta}\frac{\partial}{\partial\theta}\left(\sin\theta\frac{\partial B_r}{\partial\theta}\right)\right] - \frac{1}{R\sin\theta}\frac{\partial}{\partial\theta}(v_\theta B_r \sin\theta) + S(\theta, t)$$

**2. Toroidal 장 생성 ($\Omega$-effect):**

$$\frac{\partial B_\phi}{\partial t} = R\sin\theta\, B_r \frac{\partial\Omega}{\partial r} + R\sin\theta\, B_\theta \frac{1}{r}\frac{\partial\Omega}{\partial\theta}$$

**3. 확산 계수 추정 / Diffusivity estimate:**

$$\kappa = \frac{l^2}{4\tau} \sim 770\text{–}1540 \text{ km}^2/\text{s}$$

---

## 읽기 포인트 / Reading Points

1. **Babcock과의 차이**: 어디에서 Babcock을 넘어서는지 — 수식, 수치 해, 정량적 예측
2. **초립자 확산의 도입**: $\kappa$의 물리적 근거와 수치 추정
3. **자오선 흐름**: 극 방향 flux 수송에서의 역할 — Babcock의 "추측"을 정량화
4. **수치 시뮬레이션 결과**: 나비 다이어그램, 극성 반전 타이밍이 관측과 일치하는지
5. **모델의 한계**: Leighton이 인정하는 문제점 — kinematic 가정, back-reaction 무시

---

## 다음 단계 / Next Steps

PDF를 다운로드하세요: `sci-hub.se/10.1086/149822`

다운로드 후 `Solar_Physics/papers/10_leighton_1969/` 폴더에 `10_leighton_1969_paper.pdf`로 저장해주세요.
