---
title: "A Magneto-Kinematic Model of the Solar Cycle"
authors: Robert B. Leighton
year: 1969
journal: "The Astrophysical Journal, 156, 1–26"
topic: Solar Physics / Solar Dynamo
tags: [Babcock-Leighton dynamo, kinematic dynamo, supergranular diffusion, random walk, flux transport, meridional flow, differential rotation, butterfly diagram, Spörer's law, solar cycle, numerical simulation, poloidal field, toroidal field]
status: completed
date_started: 2026-04-10
date_completed: 2026-04-10
---

# A Magneto-Kinematic Model of the Solar Cycle (1969)
# 태양 주기의 자기-운동학적 모델 (1969)

**Robert B. Leighton**

---

## Core Contribution / 핵심 기여

Leighton은 Babcock(1961)의 현상학적 태양 다이나모 모델을 **편미분방정식 시스템**으로 정량화하고 **컴퓨터 수치 시뮬레이션**으로 풀어 태양 자기 주기를 재현한 최초의 연구를 수행했다. 모델은 세 가지 자기장 성분($B_r$, $B_\theta$, $B_\phi$)을 위도($\mu = \cos\theta$)와 시간의 함수로 추적하며, 핵심 물리 과정은: (1) **차등 회전**($\Omega$-effect)이 poloidal → toroidal 변환을 수행 (Eq. 3), (2) toroidal 장이 임계값 $B_c$를 초과하면 **자기 부력**으로 분출하여 BMR을 형성하고, Joy's law 기울기에 의해 자오선 자기 쌍극자 모멘트를 생성 (Eq. 6), (3) 분출된 $B_r$이 **초립자 확산**(random walk, $T_D \approx 20$ yr)에 의해 위도 방향으로 확산 (Eq. 5). 9개 조절 가능 매개변수($\alpha, \beta, n, \tau, F, G, B_c, \epsilon, T_D$) 중 핵심은 $F$(tilt factor)와 $\tau$(분출 시간 상수)로, 이들이 22년 진동 주기를 결정한다. "표준 모델"($\alpha=0, \beta=10, n=8, \epsilon=1$)은 Spörer 법칙, 나비 다이어그램, 극 자기장의 극이동 반전, 극소-극대 비대칭(4.6 vs 6.4년) 등을 정량적으로 재현한다. 더 나아가 분출율에 **무작위 변동**을 도입하여 관측된 태양 주기의 진폭·위상 불규칙성까지 재현했다. 이 모델은 현대 "Babcock-Leighton flux transport dynamo"의 직접적 원형이다.

Leighton performed the first **computer numerical simulation** of the solar magnetic cycle by formulating Babcock's (1961) phenomenological model as a **system of PDEs**. The model tracks three field components ($B_r$, $B_\theta$, $B_\phi$) as functions of latitude ($\mu = \cos\theta$) and time. Key physics: (1) **differential rotation** ($\Omega$-effect) converts poloidal → toroidal (Eq. 3), (2) when toroidal field exceeds critical $B_c$, **magnetic buoyancy** erupts BMRs with Joy's law tilt producing meridional magnetic dipole moments (Eq. 6), (3) erupted $B_r$ is dispersed by **supergranular diffusion** (random walk, $T_D \approx 20$ yr) (Eq. 5). Of 9 adjustable parameters, the key ones are $F$ (tilt factor) and $\tau$ (eruption time constant), which determine the 22-year oscillation period. The "standard model" ($\alpha=0, \beta=10, n=8, \epsilon=1$) quantitatively reproduces Spörer's law, butterfly diagrams, poleward migration of polar field reversal, and minimum-maximum asymmetry (4.6 vs 6.4 yr). By introducing **random fluctuations** in eruption rate, it also reproduces observed amplitude and phase irregularities of solar cycles. This model is the direct prototype of the modern "Babcock-Leighton flux transport dynamo."

---

## Reading Notes / 읽기 노트

### Section I: Introduction / 도입 (pp. 1–2)

Leighton은 Babcock 모델의 핵심 약점을 정확히 지적한다:

Leighton precisely identifies Babcock's key weakness:

> "Babcock's model assumes that, during the amplification stage, the time history of the field at each latitude is governed solely by the differential rotation and is otherwise independent of the fields at other latitudes."

즉, Babcock은 **위도 간 결합(cross-coupling)**을 무시했다. BMR의 확장·이동이 다른 위도의 자기장에 영향을 미치고, 이것이 다시 차등 회전에 의한 증폭을 수정한다. 이 피드백 루프를 포함하지 않으면 정량적 모델이 불가능하다.

Babcock ignored **cross-latitude coupling**: BMR expansion/migration affects fields at other latitudes, which modifies amplification. Without this feedback, a quantitative model is impossible.

**Leighton의 목표**: "a simple, quantitative, *closed* kinematical model" — 관측이든 이론이든 잘 알려진 물리에만 기반한 **닫힌(self-consistent)** 모델.

---

### Section II: The Model — 8 Assumptions and Key Equations / 모델 — 8개 가정과 핵심 방정식 (pp. 2–6)

Leighton은 모델을 8개 가정으로 구축한다:

**가정 1**: 축대칭 평균장 — $B_r(\mu, t)$, $B_\theta(\mu, t)$, $B_\phi(\mu, t)$는 위도와 시간만의 함수. 경도 방향 평균. / Axisymmetric mean fields, functions of latitude and time only.

**가정 2**: 차등 회전 — 위도·깊이의 함수:

$$\Omega = \Omega_s + (\alpha + \beta\sin^n\theta)\frac{R-r}{H} \quad \cdots (1)$$

여기서 $\alpha, \beta$는 반경 방향 속도 기울기, $n$은 위도 의존성 매개변수, $H$는 shear layer 두께. 표면에서 $\Omega_s = 18\sin^2\theta$ rad/yr (Newton & Nunn).

**가정 3**: $\Omega$-effect — 차등 회전이 $B_r$과 $B_\theta$에 작용하여 $B_\phi$를 생성:

$$\frac{\partial B_\phi}{\partial t} = \sin\theta\left(B_\theta\frac{\partial\Omega}{\partial\theta} + RB_r\frac{\partial\Omega}{\partial r}\right) \quad \cdots (2)$$

두 가지 경우:
- **Case A** ($\epsilon = 0$): shear layer가 매우 얇음 → 위도 기울기만 작용 / Very thin shear layer
- **Case B** ($\epsilon = 1$): shear layer 두께 = 침투 깊이 → 반경 기울기도 작용 / Shear layer = penetration depth

$$\frac{\partial' B_\phi}{\partial t} = \sin\theta\left[-(a + \beta\sin^n\theta)\frac{R}{H}B_r\right] \quad \text{(Case A)} \quad \cdots (3a)$$

$$\frac{\partial' B_\phi}{\partial t} = \sin\theta\left[-(a + \beta\sin^n\theta)\frac{R}{H}B_r + \left(36 + \frac{n\beta}{2}\sin^{n-2}\theta\right)\sin\theta\cos\theta B_\theta\right] \quad \text{(Case B)} \quad \cdots (3b)$$

**가정 4**: 임계 자기장 $B_c$ — $|B_\phi| \geq B_c$일 때 분출(eruption) 발생. 조절 가능 매개변수.

**가정 5**: **초립자 확산** — $B_r$이 random walk에 의해 위도 방향으로 확산:

$$\frac{\partial'' B_r}{\partial t} = \frac{1}{T_D}\frac{\partial}{\partial\mu}\left[(1-\mu^2)\frac{\partial B_r}{\partial\mu}\right] \quad \cdots (5)$$

여기서 $T_D = R^2/\kappa \approx 20$ yr (초립자 확산 시간 상수). $\kappa \sim 770$–$1540$ km²/s.

이것이 Leighton의 **핵심 혁신**: Babcock이 정성적으로만 언급한 flux 수송을 정량적 확산 방정식으로 표현.

**가정 6**: BMR 분출 시 Joy's law 기울기로 인한 자오선 자기 쌍극자 모멘트 생성:

$$\frac{\partial'' B_\phi}{\partial t} = -a_0|B_\phi|B_\phi / 2\pi R B_c \tau \quad \cdots (6)$$

여기서 $\tau$는 특성 분출 시간, $a_0/2\pi R \approx 1/100$. 분출된 flux의 기울기가 $B_r$에 소스를 제공 → **toroidal → poloidal 변환** (Babcock-Leighton 메커니즘).

**가정 7**: 분출이 $B_\phi$를 감소시킴 — 분출된 만큼 차감.

**가정 8**: Ad hoc 보정 — $B_\phi$의 50년 감쇠, 비분출 $B_r$의 소스(fraction $G$), 총 flux 보존 보정.

### 최종 연립 방정식 / Final System of Equations (pp. 5–6)

$$B_\theta = \frac{R}{H\sin\theta}\int_{-1}^{\mu}(B_r + B_s)\,d\mu \quad \cdots (7)$$

$$\frac{\partial B_\phi}{\partial t} = \sin\theta\left[-(a+\beta\sin^n\theta)\frac{R}{H}(B_r+B_s) + \epsilon\left(36+\frac{n\beta}{2}\sin^{n-2}\theta\right)\sin\theta\cos\theta B_\theta\right] - \delta\frac{|B_\phi|B_\phi}{100B_c\tau} - \frac{B_\phi}{50} \quad \cdots (8)$$

$$\frac{\partial B_r}{\partial t} = -\delta\frac{FH}{80R\tau}\frac{\partial}{\partial\mu}(\mu B_\phi) + \frac{1}{T_D}\frac{\partial}{\partial\mu}\left[(1-\mu^2)\frac{\partial B_r}{\partial\mu}\right] \quad \cdots (9)$$

$$\frac{\partial B_s}{\partial t} = -\frac{GH}{80R\tau}\frac{\partial}{\partial\mu}(\mu B_\phi) - \frac{B_s}{50} \quad \cdots (10)$$

여기서 $\delta = 1$ if $|B_\phi| \geq B_c$, $\delta = 0$ otherwise. $F$는 tilt factor, $B_s$는 비분출 방사장.

---

### Section III: Preliminary Discussion / 예비 논의 (pp. 6–7)

**9개 매개변수의 역할:**

| 매개변수 / Parameter | 역할 / Role | 관측 제약 / Observational constraint |
|---|---|---|
| $T_D$ (~20 yr) | 초립자 확산 시간 | 초립자 크기·수명에서 ~50% 이내 확정 |
| $H$ | Shear layer 두께 | $B_r$과 $B_\phi$의 상대 스케일만 결정, 중요하지 않음 |
| $B_c$ | 임계 분출 자기장 | 진동 진폭만 결정 |
| $G$ (~0.003) | 비분출 $B_r$ 비율 | 매우 작음, 정상 상태 수렴 촉진 |
| $\alpha, \beta$ | 반경 방향 $\Omega$ 기울기 | 핵심! 내부가 외부보다 빠르게 회전 시 효과적 |
| $n$ | $\Omega$의 위도 의존성 | 나비 다이어그램 형태 결정 |
| $F$ | Tilt factor (Joy's law) | **핵심**: $F > F_m$이어야 진동 유지 |
| $\tau$ | 분출 시간 상수 | $F$와 함께 22년 주기 결정 |

**핵심 통찰**: "If the solution is to be oscillatory, the erupted flux must produce sufficient axial dipole moment to not only cancel the previously existing moment but also to establish an equal one of opposite sign, all in the presence of the dispersive effects of the random walk."

즉, $F$에는 최소값 $F_m$이 존재 — $F < F_m$이면 진동이 소멸. $F_m$이 작을수록 모델이 물리적으로 그럴듯하다.

---

### Section V: Results / 결과 (pp. 7–20)

#### (a) Case $\alpha = \beta = 0$ — 반경 속도 기울기 없음 / No radial velocity gradient

- $F_m \approx 6$: 진동 유지에 필요한 최소 tilt factor가 꽤 큼 → 비현실적
- 그러나 나비 다이어그램, 극이동 반전 등 정성적 특징은 잘 재현
- Fig. 1: **문자 기반 컴퓨터 출력** — 0–9(양), A–I(음)로 자기장 세기 표현. 역사적으로 중요한 초기 수치 시뮬레이션 출력
- Fig. 2: $B_\phi$와 $B_r$의 contour plot — 나비 다이어그램 형태 확인
- **극소-극대 비대칭**: 4.6년(상승) vs 6.4년(하강) — 관측값 4.50 vs 6.56년과 놀랍게 일치!
- 극 자기장 반전: $d\mu/dt = 0.085$ yr⁻¹ → 극대 후 ~3년에 극에 도달 (관측: ~3.6년)

#### (b) Case $\alpha = 0, \beta = 18$ — 반경 속도 기울기 있음 / With radial gradient

- **핵심 발견**: 반경 방향 $\Omega$ 기울기는 위도 방향 기울기보다 **10배 효과적**
- $F_m \approx 0.6$으로 급감! 물리적으로 훨씬 그럴듯
- 이것은 **태양 내부가 표면보다 빠르게 회전**해야 함을 의미 (후에 일진학으로 확인)
- Fig. 3: Spörer 법칙과의 비교 — 다양한 $n$ 값에서 관측과 양적 일치
- Fig. 4: $n = 2, 6, 8, 10$에서의 나비 다이어그램 비교

#### (e) Symmetric Modes / 대칭 모드

- 흥미로운 발견: 반대칭 모드(dipole-like, 태양의 실제 주기)가 지배적이지만, 대칭 모드(quadrupole-like)도 가능
- 두 모드가 "beat"하여 반구 비대칭 설명 가능

#### (f) "Standard Case" 채택 / Adoption

$$\alpha = 0, \quad \beta = 10, \quad n = 8, \quad \epsilon = 1$$

이 매개변수 조합이 관측과 가장 잘 일치. Fig. 6에 contour plot 제시.

#### (g) Introduction of Randomness / 무작위성 도입 (pp. 15–20)

**이것이 Leighton 논문의 가장 혁신적 부분 중 하나이다.**

$\tau$에 로그 정규 분포(log-normal)의 무작위 변동을 도입:
- 평균 분출 주기: 11년
- RMS 변동: 1–2년
- $\tau$ 범위: 0.133–2.12년
- $F = 2$, $\tau_0 = 0.60$ yr, $\sigma = 1.0$

**결과** (Fig. 7):
- 20개 연속 11년 주기를 시뮬레이션 → 관측된 18개 주기와 비교
- 주기 간 진폭 변동, 상승/하강 비대칭, 반구 간 위상차 등이 자연스럽게 출현
- **Table 2**: 모델과 관측의 통계 비교 — 놀라운 정량적 일치

| 물리량 / Quantity | 모델 / Model | 관측 / Observed |
|---|---|---|
| 평균 주기 / Mean period | 11.0 yr | 11.0 yr |
| 주기 표준편차 / Period σ | 1.5 yr | 1.2 yr |
| 진폭 변동 / Amplitude variation | 관측과 유사 | — |
| 상승/하강 비율 / Rise/decline ratio | 0.65 | 0.68 |

---

### Section VI: Discussion / 논의 (pp. 20–25)

Leighton은 모델의 성과와 한계를 균형 있게 논의한다:

**성과:**
1. Spörer 법칙 정량적 재현
2. 나비 다이어그램의 폭과 형태
3. 극 자기장 극이동 반전 타이밍
4. 극소-극대 비대칭 (4.6 vs 6.4년)
5. 무작위 변동에 의한 주기 불규칙성
6. 반구 비대칭의 자연적 출현
7. 특정 경도대 재발의 정성적 설명

**한계:**
1. **Kinematic 가정**: 자기장이 유체 운동에 미치는 역효과(back-reaction) 무시 → 진정한 자기유체역학(MHD) 모델이 아님
2. **축대칭 가정**: 경도 방향 구조(활동 경도대 등) 기술 불가
3. **얕은 층 가정**: 현대에는 tachocline (0.7 $R_\odot$)이 핵심으로 밝혀짐
4. **$F$ factor의 물리적 해석**: $F > 1$이 필요한 경우 — Joy's law 기울기만으로는 불충분할 수 있음
5. **자오선 흐름의 명시적 부재**: 확산만으로 극이동 설명 — 현대 모델에서는 자오선 흐름이 필수

---

## Key Takeaways / 핵심 시사점

1. **최초의 태양 다이나모 수치 시뮬레이션**: 1969년에 컴퓨터로 22년 자기 주기를 재현한 것은 놀라운 성취이다. Fig. 1의 문자 기반 출력은 초기 전산 과학의 역사적 유물이자, 물리학에서 수치 시뮬레이션의 시작을 상징한다. / The first numerical simulation of a 22-year solar dynamo cycle in 1969 is a remarkable achievement. Fig. 1's character-based output is a historical artifact symbolizing the dawn of computational physics.

2. **초립자 확산의 핵심 역할**: Leighton이 1964년에 발견한 초립자 random walk 확산($\kappa \sim 10^3$ km²/s)이 Babcock 모델의 가장 큰 결함 — flux가 "어떻게" 수송되는가 — 를 해결했다. 관측자가 이론을 완성한 드문 사례. / Supergranular diffusion discovered by Leighton himself (1964) solved Babcock's biggest gap — "how" flux is transported. A rare case of observer completing theory.

3. **$F_m$ (최소 tilt factor)의 의미**: 이 값이 물리적으로 허용 가능한 범위 내여야 모델이 유효하다. 반경 방향 $\Omega$ 기울기를 포함하면 $F_m$이 6에서 0.6으로 급감 → 내부 차등 회전의 중요성을 1969년에 이미 예측. 1998년 Schou et al.(SP #17)이 일진학으로 확인. / The dramatic drop of $F_m$ from 6 to 0.6 with radial $\Omega$ gradient predicted the importance of internal differential rotation in 1969 — confirmed by helioseismology in 1998.

4. **무작위성에서 오는 주기 불규칙성**: 단순한 로그 정규 분포의 분출율 변동만으로 관측된 주기 진폭·위상 불규칙성을 재현한 것은 심오하다. 이것은 태양 주기가 **확정적(deterministic) 진동자 + 확률적(stochastic) 교란**의 조합임을 시사한다. 현대 태양 주기 예측의 핵심 개념. / Reproducing cycle irregularities with simple stochastic fluctuations in eruption rate suggests the solar cycle is a **deterministic oscillator + stochastic perturbation** — a key concept in modern prediction.

5. **Babcock-Leighton 메커니즘의 정의**: 이 논문에서 확립된 "toroidal → poloidal 변환이 BMR의 Joy's law 기울기에 의해 일어난다"는 메커니즘은 mean-field 다이나모의 $\alpha$-effect와 근본적으로 다르다. $\alpha$-effect는 소규모 난류에 의한 것이고, Babcock-Leighton 메커니즘은 **대규모 BMR의 표면 효과**이다. 현대에는 후자가 태양 다이나모에 더 적합한 것으로 받아들여진다. / The Babcock-Leighton mechanism (toroidal → poloidal via BMR tilt) differs fundamentally from mean-field α-effect (small-scale turbulence). Modern consensus favors the former for the solar dynamo.

6. **"닫힌(closed)" 모델의 의미**: Leighton의 모델은 외부 입력 없이 자기장이 스스로 진동하는 self-consistent 시스템이다. 한번 시작되면 22년 주기로 영구히 진동한다(무작위 교란이 없으면). 이것은 태양 다이나모가 **자기 유지(self-sustaining) 시스템**임을 수학적으로 보여준 최초의 증명이다. / The first mathematical proof that the solar dynamo is a **self-sustaining** oscillating system.

7. **현대 flux transport dynamo와의 연결**: Leighton의 모델에 자오선 흐름을 명시적으로 추가하면 현대의 "flux transport dynamo"가 된다. Dikpati & Charbonneau (1999), Wang & Sheeley (1991) 등이 이 확장을 수행했으며, 이것이 Solar Cycle 24, 25 예측의 기초가 되었다. / Adding explicit meridional flow to Leighton's model yields the modern "flux transport dynamo" used for Solar Cycle predictions.

---

## Mathematical Summary / 수학적 요약

### Leighton 모델의 완전한 방정식 시스템 / Complete Equation System

**좌표**: $\mu = \cos\theta$ (colatitude cosine), $\sin\theta = \sqrt{1-\mu^2}$, 시간 $t$ (years)

| Eq. | 수식 / Formula | 물리적 의미 / Physical Meaning |
|---|---|---|
| (1) | $\Omega = \Omega_s + (\alpha + \beta\sin^n\theta)\frac{R-r}{H}$ | 차등 회전 (위도 + 깊이) |
| (3a) | $\partial'B_\phi/\partial t = -\sin\theta(a+\beta\sin^n\theta)\frac{R}{H}B_r$ | $\Omega$-effect (Case A, 얕은 층) |
| (3b) | $\partial'B_\phi/\partial t = (3a) + \epsilon(\cdots)\sin\theta\cos\theta B_\theta$ | $\Omega$-effect (Case B, 두꺼운 층) |
| (5) | $\partial''B_r/\partial t = \frac{1}{T_D}\frac{\partial}{\partial\mu}[(1-\mu^2)\frac{\partial B_r}{\partial\mu}]$ | 초립자 확산 ($T_D \approx 20$ yr) |
| (6) | $\partial''B_\phi/\partial t = -a_0\|B_\phi\|B_\phi/(2\pi RB_c\tau)$ | BMR 분출 → $B_\phi$ 감소 |
| (7) | $B_\theta = \frac{R}{H\sin\theta}\int_{-1}^{\mu}(B_r+B_s)\,d\mu$ | $\nabla\cdot\mathbf{B}=0$에서 유도 |
| (9) | $\frac{\partial B_r}{\partial t} = -\delta\frac{FH}{80R\tau}\frac{\partial}{\partial\mu}(\mu B_\phi) + \frac{1}{T_D}\frac{\partial}{\partial\mu}[(1-\mu^2)\frac{\partial B_r}{\partial\mu}]$ | **핵심**: 방사장 진화 (분출 소스 + 확산) |
| (10) | $\frac{\partial B_s}{\partial t} = -\frac{GH}{80R\tau}\frac{\partial}{\partial\mu}(\mu B_\phi) - \frac{B_s}{50}$ | 비분출 방사장 (안정화 항) |

### 물리적 흐름도 / Physical Flow Chart

```
B_r (poloidal/radial)
  │
  │ ← Supergranular diffusion (Eq. 5, T_D ~ 20 yr)
  │ ← BMR eruption source with Joy's law tilt (Eq. 9, factor F)
  │
  ▼
B_θ (poloidal/meridional) ← derived from B_r via ∇·B = 0 (Eq. 7)
  │
  │ Differential rotation (Ω-effect, Eq. 3)
  │ ┌─ Latitude gradient: Ω_s = 18 sin²θ
  │ └─ Radial gradient: α + β sin^n θ
  ▼
B_ϕ (toroidal)
  │
  │ When |B_ϕ| ≥ B_c: eruption (magnetic buoyancy)
  │ → produces B_r source via Joy's law tilt
  │ → reduces B_ϕ (Eq. 6)
  │
  └──→ back to B_r (closes the loop!)
```

### "표준 모델" 매개변수 / "Standard Model" Parameters

| 매개변수 | 값 | 물리적 의미 |
|---|---|---|
| $\alpha$ | 0 | 반경 $\Omega$ 기울기 (위도 무관 부분) |
| $\beta$ | 10 rad/yr | 반경 $\Omega$ 기울기 (위도 의존 부분) |
| $n$ | 8 | 위도 의존성 지수 |
| $\epsilon$ | 1 (Case B) | Shear layer = 침투 깊이 |
| $T_D$ | 20 yr | 초립자 확산 시간 ($\kappa \sim 770$ km²/s) |
| $F$ | ~2 | Tilt factor (Joy's law 효율) |
| $\tau$ | ~0.6 yr | 분출 시간 상수 |
| $G$ | 0.003$F$ | 비분출 fraction |
| $B_c$ | 20$R/H$ | 임계 분출 자기장 |

---

## Paper in the Arc of History / 역사 속의 논문

```
1908  Hale ──────────── 흑점 자기장 발견
      │
1925  Hale & Nicholson ── 극성 법칙, 22년 자기 주기
      │
1942  Alfvén ────────── MHD + frozen-in 조건
      │
1961  Babcock ─────── 5단계 현상학적 다이나모 모델
      │                  Phenomenological, no equations
      │
1962  Leighton et al. ── 5분 진동 발견 (일진학의 시작)
      │
1964  Leighton ─────── 초립자 확산 발견 (random walk)
      │                  Supergranular diffusion
      │
      ▼
╔══════════════════════════════════════════════════════════════════════════╗
║  1969  LEIGHTON — 자기-운동학적 다이나모 모델                               ║
║         Babcock 모델을 PDE + 수치 시뮬레이션으로 정량화                       ║
║         초립자 확산 + 차등 회전 + Joy's law → 닫힌 모델                      ║
║         무작위 변동 → 주기 불규칙성 재현                                      ║
║         A Magneto-Kinematic Model of the Solar Cycle                     ║
╚══════════════════════════════════════════════════════════════════════════╝
      │
      ├── 1991  Wang & Sheeley ── 자오선 흐름 추가 → flux transport dynamo
      │
      ├── 1995  Dikpati & Choudhuri ── mean-field + B-L 결합 모델
      │
      ├── 1998  Schou et al. ──── tachocline 발견 → 내부 차등 회전 확인
      │                            (Leighton의 1969 예측 검증!)
      │
      ├── 1999  Dikpati & Charbonneau ── 현대 flux transport dynamo
      │
      └── 2020s ── Solar Cycle 25 예측 (Babcock-Leighton 기반)
```

---

## Connections to Other Papers / 다른 논문과의 연결

| 논문 / Paper | 연결 / Connection |
|---|---|
| **Hale & Nicholson (1925)** [SP #7] | 극성 법칙 — 모델이 "자동으로" 재현해야 할 관측. Leighton의 모델에서 toroidal 장 부호가 반구별로 반대 → 극성 법칙 만족 / Polarity law automatically reproduced |
| **Alfvén (1942)** [SP #8] | Frozen-in 조건 — 완전 전도도 가정의 물리적 기반 ("Perfect conductivity and laminar flow are assumed") / Physical basis for perfect conductivity assumption |
| **Babcock (1961)** [SP #9] | **직접적 선행 논문**. Leighton이 정량화하는 대상. "Babcock's (1961) topological model... seems to account in a natural way for some of the most characteristic qualitative features" / Direct predecessor that Leighton quantifies |
| **Leighton et al. (1962)** [SP #13] | 5분 진동 발견 — 같은 저자. 관측 경험이 이론 구축에 기여 / Same author's observational work |
| **Schou et al. (1998)** [SP #17] | Tachocline 발견 — Leighton이 1969년에 "내부가 빠르게 회전"해야 함을 예측 → 일진학으로 확인 / Tachocline confirmed Leighton's 1969 prediction |
| **Parker (1958)** [SP #12] | 태양풍 이론 — 극 자기장(Leighton 모델의 출력)이 heliospheric field의 원천 / Polar field from Leighton's model sources the heliosphere |

---

## References / 참고문헌

- Leighton, R.B., "A Magneto-Kinematic Model of the Solar Cycle," *The Astrophysical Journal*, 156, 1–26, 1969. [DOI: 10.1086/149822]
- Babcock, H.W., "The Topology of the Sun's Magnetic Field and the 22-Year Cycle," *The Astrophysical Journal*, 133, 572–587, 1961.
- Leighton, R.B., "Transport of Magnetic Fields on the Sun," *The Astrophysical Journal*, 140, 1547, 1964.
- Leighton, R.B., Noyes, R.W. & Simon, G.W., "Velocity Fields in the Solar Atmosphere. I.," *The Astrophysical Journal*, 135, 474, 1962.
- Wang, Y.-M. & Sheeley, N.R., "Magnetic Flux Transport and the Sun's Dipole Moment," *The Astrophysical Journal*, 375, 761, 1991.
- Dikpati, M. & Charbonneau, P., "A Babcock-Leighton Flux Transport Dynamo with Solar-like Differential Rotation," *The Astrophysical Journal*, 518, 508, 1999.
