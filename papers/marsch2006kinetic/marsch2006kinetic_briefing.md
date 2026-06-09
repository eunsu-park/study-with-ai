---
title: "Kinetic Physics of the Solar Corona and Solar Wind — Pre-reading Briefing"
paper: "Marsch, E. (2006), Kinetic Physics of the Solar Corona and Solar Wind, Living Rev. Solar Phys., 3, 1"
date: 2026-04-09
type: briefing
---

# 사전 읽기 브리핑: Kinetic Physics of the Solar Corona and Solar Wind
# Pre-reading Briefing: Kinetic Physics of the Solar Corona and Solar Wind

**저자 / Author**: Eckart Marsch (Max-Planck-Institut für Sonnensystemforschung)
**출판 / Published**: Living Reviews in Solar Physics, 3, 1 (2006)
**DOI**: 10.12942/lrsp-2006-1
**분량 / Length**: ~85 pages (review article)

---

## 핵심 기여 / Core Contribution

이 리뷰는 태양 코로나와 태양풍의 **운동론적(kinetic) 물리학**을 포괄적으로 다루며, MHD(자기유체역학) 수준의 유체 기술만으로는 포착할 수 없는 입자 수준의 물리 현상을 체계적으로 정리한다. 특히 Helios, SOHO, Ulysses 우주선의 *in situ* 측정으로 밝혀진 태양풍 입자들의 **속도 분포 함수(VDF)**가 Maxwell 분포에서 크게 벗어난다는 관측적 사실을 중심으로, 이러한 비평형 상태를 설명하는 운동론적 이론 — Vlasov-Boltzmann 방정식, 파동-입자 상호작용, 준선형 이론(QLT), exospheric 모델 — 을 상세히 리뷰한다. 코로나 가열과 태양풍 가속이라는 태양물리학의 핵심 미해결 문제에 대해 운동론적 관점에서의 해답을 모색하는 핵심 참고문헌이다.

This review comprehensively covers the **kinetic (particle-level) physics** of the solar corona and solar wind, systematically organizing phenomena that cannot be captured by MHD-level fluid descriptions alone. Centering on the observational fact — revealed by *in situ* measurements from Helios, SOHO, and Ulysses — that solar wind particle **velocity distribution functions (VDFs)** deviate strongly from Maxwellian, it reviews the kinetic theories explaining these non-equilibrium states: the Vlasov-Boltzmann equation, wave-particle interactions, quasilinear theory (QLT), and exospheric models. This is a key reference for seeking kinetic-perspective answers to the central unsolved problems of solar physics: coronal heating and solar wind acceleration.

---

## 역사적 맥락 / Historical Context

```
1958  Parker — 태양풍의 유체역학적 모델 (초음속 팽창)
       Parker — Hydrodynamic model of solar wind (supersonic expansion)
              |
1970s Helios 미션 — 0.3 AU까지 in situ 측정, 비Maxwell 분포 발견
       Helios mission — in situ measurements to 0.3 AU, non-Maxwellian VDFs discovered
              |
1979  Scudder & Olbert — exospheric 모델의 초기 형태
       Scudder & Olbert — early exospheric model
              |
1982  Marsch et al. — 양성자 VDF의 core 온도 비등방성 및 beam 구조 발견
       Marsch et al. — proton VDF core temperature anisotropy and beam structure
              |
1990s Ulysses — 고위도 태양풍 관측, SOHO — 코로나 원격 관측
       Ulysses — high-latitude solar wind, SOHO — coronal remote sensing
              |
2003  Lamy, Zouganelis et al. — 현대 exospheric 모델 (kappa 분포 기반)
       Lamy, Zouganelis et al. — modern exospheric models (kappa-based)
              |
>>>  2006  Marsch — 이 리뷰: 운동론의 포괄적 종합 <<<
>>>  2006  Marsch — this review: comprehensive kinetic synthesis <<<
              |
2018  Parker Solar Probe 발사 — 코로나 직접 탐사의 시작
       Parker Solar Probe launch — beginning of direct coronal exploration
```

이 논문은 Parker의 유체 모델(1958)과 Helios의 관측적 발견(1970s-80s) 사이의 간극을 메우는 운동론적 틀을 제공한다. LRSP #3 (Nakariakov & Verwichte, 2005)의 MHD 파동 리뷰와 상보적이며, #4 (Sheeley, 2005)의 자기 flux transport를 미시적 수준에서 재해석하는 관점을 제공한다.

This paper bridges the gap between Parker's fluid model (1958) and Helios' observational discoveries (1970s-80s) with a kinetic framework. It complements LRSP #3's MHD wave review and reinterprets #4's magnetic flux transport from a microscopic perspective.

---

## 필요한 배경 지식 / Prerequisites

### 물리학 기초 / Physics Fundamentals
- **통계역학 / Statistical Mechanics**: 분포 함수, Maxwell-Boltzmann 분포, 위상 공간(phase space) / Distribution functions, Maxwell-Boltzmann distribution, phase space
- **전자기학 / Electromagnetism**: Maxwell 방정식, Lorentz 힘, 전자기파 편광 / Maxwell's equations, Lorentz force, EM wave polarization
- **플라즈마 물리학 기초 / Plasma Physics Basics**: Debye 차폐, 플라즈마 진동수, 자이로주파수, 플라즈마 beta / Debye shielding, plasma frequency, gyrofrequency, plasma beta

### 수학 도구 / Mathematical Tools
- **편미분방정식 / PDEs**: 특히 수송 방정식 / especially transport equations
- **Fourier 분석**: 분산 관계 유도에 필수 / essential for deriving dispersion relations
- **텐서 대수 / Tensor Algebra**: 압력 텐서, 응력 텐서 / pressure tensor, stress tensor
- **특수함수 / Special Functions**: Bessel 함수, 감마 함수 / Bessel functions, Gamma function

### 이전 LRSP 논문 / Prior LRSP Papers
- **#3 Nakariakov & Verwichte (2005)**: MHD 파동 — 이 논문은 MHD를 넘어서는 운동론적 파동을 다룸 / MHD waves — this paper goes beyond MHD to kinetic waves
- **#5 Gizon & Birch (2005)**: 일진학의 파동 이론 배경 / helioseismology wave theory background

---

## 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **VDF (Velocity Distribution Function)** | 속도 공간에서 입자의 분포를 나타내는 함수. Maxwell 분포가 열평형 상태이지만, 태양풍에서는 크게 벗어남 / Function describing particle distribution in velocity space. Maxwellian is thermal equilibrium, but solar wind deviates strongly |
| **Vlasov-Boltzmann equation** | 충돌항을 포함한 운동론적 방정식. 위상 공간에서 VDF의 시간 진화를 기술 / Kinetic equation with collision term. Describes time evolution of VDF in phase space |
| **Kappa ($\kappa$) distribution** | Maxwell 분포의 일반화. $\kappa \to \infty$이면 Maxwell, 작은 $\kappa$는 고에너지 꼬리(suprathermal tail)를 가짐 / Generalization of Maxwellian. $\kappa \to \infty$ gives Maxwellian; small $\kappa$ has enhanced high-energy tails |
| **Strahl** | 자기장 방향을 따라 좁은 pitch-angle로 이동하는 초열 전자 빔. 열 흐름의 주요 운반체 / Suprathermal electron beam traveling along **B** with narrow pitch-angle. Primary carrier of heat flux |
| **Core-halo structure** | 전자 VDF의 구조: 저온의 core와 고온의 halo로 구성. Strahl은 halo의 비등방적 성분 / Electron VDF structure: cold core + hot halo. Strahl is anisotropic component of halo |
| **Cyclotron resonance** | 파동의 Doppler-shifted 주파수가 입자의 자이로주파수와 일치할 때 발생하는 공명. $\omega - k_\parallel v_\parallel = \pm \Omega_j$ / Resonance when Doppler-shifted wave frequency matches particle gyrofrequency |
| **Landau damping** | 파동의 위상 속도와 비슷한 속도를 가진 입자가 파동 에너지를 흡수하는 비충돌적 감쇠 / Collisionless damping where particles traveling near the wave phase speed absorb wave energy |
| **Quasilinear Theory (QLT)** | 파동-입자 상호작용의 이론적 프레임워크. VDF와 파동 에너지가 느리게 변하면서 서로 영향 / Theoretical framework for wave-particle interactions. VDF and wave energy slowly co-evolve |
| **Exospheric model** | 코로나를 충돌 없는 외기권으로 모델링. 입자가 중력 + 전기장 + 자기장 속에서 탄도적으로 운동 / Models corona as collisionless exosphere. Particles move ballistically in gravity + electric + magnetic fields |
| **Plasma beta ($\beta$)** | 열압력 대 자기 압력의 비율. $\beta = 8\pi n k_B T / B_0^2$. 코로나에서는 $\beta \ll 1$, 태양풍에서는 $\beta \sim 1$ / Ratio of thermal to magnetic pressure. $\beta \ll 1$ in corona, $\beta \sim 1$ in solar wind |
| **Proton beam** | 양성자 VDF에서 core와 별도로 자기장 방향으로 빠르게 이동하는 2차 성분. $v_d \sim V_A$ / Secondary proton component drifting along **B** at roughly Alfvén speed |
| **Temperature anisotropy** | $T_\perp \neq T_\parallel$ — 자기장에 수직/평행 방향의 온도가 다름. 파동-입자 상호작용의 증거 / $T_\perp \neq T_\parallel$ — different temperatures perpendicular/parallel to **B**. Evidence of wave-particle interactions |
| **Fokker-Planck operator** | Coulomb 충돌을 VDF에 대한 확산 방정식으로 기술하는 연산자 / Operator describing Coulomb collisions as diffusion equation acting on VDF |
| **Coronal funnel** | 코로나 홀 바닥에서 chromospheric network에 뿌리를 둔 깔때기 모양 자기장 구조. 태양풍의 기원 / Funnel-shaped magnetic field structure rooted in chromospheric network at coronal hole base. Origin of solar wind |

---

## 수식 미리보기 / Equations Preview

### 1. Vlasov-Boltzmann 방정식 / Vlasov-Boltzmann Equation

태양풍 운동론의 핵심 방정식. 위상 공간에서 종(species) $j$의 VDF $f_j$의 시간 진화를 기술한다.
The fundamental equation of solar wind kinetics. Describes time evolution of species $j$ VDF in phase space.

$$\left[\frac{\partial}{\partial t} + \mathbf{v} \cdot \frac{\partial}{\partial \mathbf{x}} + \left(\mathbf{g} + \frac{e_j}{m_j}(\mathbf{E} + \frac{1}{c}\mathbf{v} \times \mathbf{B})\right) \cdot \frac{\partial}{\partial \mathbf{v}}\right] f_j = \left[\frac{d}{dt}f_j\right]_{c,w}$$

- 좌변: 위상 공간에서의 VDF 수송 (자유 이동 + 중력 + 전자기력) / LHS: VDF transport in phase space (free streaming + gravity + EM forces)
- 우변: 충돌 및 파동-입자 상호작용 항 / RHS: collision and wave-particle interaction terms

### 2. Kappa 분포 / Kappa Distribution

비열적 초열 꼬리를 가진 VDF의 일반적 표현. Exospheric 모델과 관측 VDF 피팅에 핵심적.
General expression for VDFs with suprathermal tails. Key for exospheric models and observed VDF fitting.

$$f(v) = \frac{n}{(\pi \kappa v_\kappa^2)^{3/2}} \frac{\Gamma(\kappa+1)}{\Gamma(\kappa-1/2)} \left[1 + \frac{v^2}{\kappa v_\kappa^2}\right]^{-(\kappa+1)}, \quad v_\kappa = \left(\frac{2\kappa - 3}{\kappa} \frac{k_B T_\kappa}{m}\right)^{1/2}$$

- $\kappa \to \infty$: Maxwell 분포로 수렴 / converges to Maxwellian
- $\kappa \sim 2$-4: 강한 고에너지 꼬리 (관측에서 흔함) / strong high-energy tail (common in observations)
- 고에너지 멱법칙 꼬리: $f(v) \sim v^{-2(\kappa+1)}$ / power-law tail

### 3. Bernoulli 에너지 보존 / Bernoulli Energy Conservation

코로나 팽창의 polytropic 모델에서의 에너지 보존. 태양풍 속도의 상한을 결정.
Energy conservation in polytropic model of coronal expansion. Sets upper bound on solar wind speed.

$$\frac{1}{2}V^2 = \frac{\gamma}{\gamma-1}\frac{2k_B T_C}{m_p} - \frac{GM_\odot}{R_\odot}$$

- $V$: 종단 태양풍 속도 / terminal solar wind speed
- $T_C$: 코로나 온도 / coronal temperature
- 탈출 속도: $V_\infty = (2GM_\odot/R_\odot)^{1/2} = 618$ km s$^{-1}$ / escape speed

### 4. Cyclotron 공명 조건 / Cyclotron Resonance Condition

파동-입자 에너지 교환이 일어나는 조건. 코로나 가열의 핵심 메커니즘.
Condition for wave-particle energy exchange. Key mechanism for coronal heating.

$$w_\parallel = w_j^\pm = \frac{\tilde{\omega}'(k_\parallel) \pm \Omega_j}{k_\parallel}$$

- $\tilde{\omega}'$: Doppler-shifted 주파수 / Doppler-shifted frequency
- $\Omega_j$: 종 $j$의 자이로주파수 / gyrofrequency of species $j$
- $+$: 좌편광 파동과의 공명, $-$: 우편광 파동과의 공명 / $+$: left-hand resonance, $-$: right-hand resonance

### 5. 온도 비등방성 임계값 / Temperature Anisotropy Threshold

양성자 core의 $T_\perp/T_\parallel$ 비율을 제한하는 cyclotron 불안정성 조건.
Cyclotron instability condition constraining proton core $T_\perp/T_\parallel$ ratio.

$$\frac{T_{\perp p}}{T_{\parallel p}} - 1 = \frac{S_p}{\beta_{\parallel p}^{\alpha_p}}$$

- $S_p \approx 1$, $\alpha_p \approx 0.4$: 피팅 파라미터 / fitting parameters
- Helios 관측과 잘 일치: 태양풍이 한계 안정성 근처에서 유지됨을 의미 / Good agreement with Helios: solar wind maintains near marginal stability

### 6. 준선형 확산 방정식 / Quasilinear Diffusion Equation

파동 장(field)에 의한 VDF의 pitch-angle 확산을 기술. 파동-입자 상호작용의 정량적 프레임워크.
Describes pitch-angle diffusion of VDF by wave fields. Quantitative framework for wave-particle interactions.

$$\frac{\partial}{\partial t} f_j(v_\parallel, v_\perp, t) = \int \frac{d^3k}{(2\pi)^3} \sum_M \hat{\mathcal{B}}_M(\mathbf{k}) \frac{1}{v_\perp} \frac{\partial}{\partial \alpha} \left(v_\perp \nu_{j,M}(\mathbf{k}; v_\parallel, v_\perp) \frac{\partial}{\partial \alpha} f_j\right)$$

- $\hat{\mathcal{B}}_M$: 정규화된 자기장 요동 스펙트럼 / normalized magnetic fluctuation spectrum
- $\nu_{j,M}$: 이온-파동 완화율 / ion-wave relaxation rate
- $\partial/\partial\alpha$: pitch-angle 미분 / pitch-angle derivative

---

## 논문 구조 개요 / Paper Structure Overview

| 장 / Ch. | 제목 / Title | 핵심 내용 / Key Content |
|---|---|---|
| 1 | Introduction | 논문 범위, 운동론의 중요성, 태양풍 유형 (fast/slow/transient) / Scope, importance of kinetics, solar wind types |
| 2 | Particle Velocity Distributions | 전자 (core-halo-strahl), 양성자 (비등방성, beam), 알파 입자, 중이온의 VDF 관측 / Observed VDFs of electrons, protons, alphas, heavy ions |
| 3 | Kinetic Description | 코로나 에너지론, 충돌 조건, exospheric 모델, Vlasov-Boltzmann 이론 / Coronal energetics, collisionality, exospheric models, V-B theory |
| 4 | Transport | 고전적 수송 이론의 한계, 전자 열유속 문제, multi-moment 유체 방정식 / Classical transport breakdown, electron heat flux, multi-moment fluid equations |
| 5 | Plasma Waves & Microinstabilities | 분산 관계, Landau/cyclotron 감쇠, kinetic Alfvén waves, 비선형 결합 / Dispersion, Landau/cyclotron damping, KAWs, nonlinear coupling |
| 6 | Wave-Particle Interactions | QLT, pitch-angle 확산, kinetic shell 모델, 온도 비등방성 조절, beam 안정성 / QLT, pitch-angle diffusion, kinetic shell, anisotropy regulation, beam stability |
| 7 | Kinetic Modelling | Vlasov 방정식의 수치 풀이: 전자+양성자 모델, 코로나 이온/전자 모델 / Numerical Vlasov solutions: e+p models, coronal ion/electron models |
| 8 | Summary & Conclusions | 요약 및 향후 전망 (Parker Solar Probe 등) / Summary and future perspectives |

---

## 읽기 전략 / Reading Strategy

1. **Section 2 (VDF 관측)부터 시작**: 나머지 이론의 동기부여가 되는 관측적 기반. 그림 1-5를 주의 깊게 살펴볼 것 / Start with Section 2 (VDF observations) — motivational observational basis. Study Figures 1-5 carefully.

2. **Section 3.6-3.7 (Vlasov-Boltzmann)에 집중**: 이후 모든 이론적 논의의 수학적 기반 / Focus on Section 3.6-3.7 — mathematical foundation for all subsequent theory.

3. **Section 5-6 (파동 & 파동-입자 상호작용)이 핵심**: 코로나 가열 문제의 운동론적 해답. 이 부분이 논문의 가장 중요한 기여 / Sections 5-6 are the heart — kinetic answers to coronal heating. This is the paper's most important contribution.

4. **Table 1 (충돌 조건)과 Table 2 (감쇠 영역)는 반드시 이해할 것** / Must understand Table 1 (collisionality) and Table 2 (damping regimes).

---

## 이전 논문과의 연결 / Connections to Previous Papers

| 논문 / Paper | 연결 / Connection |
|---|---|
| LRSP #1 Wood (2004) | Astrosphere — 항성풍의 운동론적 측면을 이 논문이 태양풍에 적용 / Stellar wind kinetic aspects applied to solar wind here |
| LRSP #2 Miesch (2005) | 내부 역학의 유체적 기술 vs. 이 논문의 코로나/태양풍 운동론적 기술 / Fluid interior dynamics vs. kinetic corona/wind description |
| LRSP #3 Nakariakov & Verwichte (2005) | MHD 파동의 코로나 진단 → 이 논문은 MHD 너머 운동론적 파동으로 확장 / MHD coronal diagnostics → this paper extends to kinetic waves beyond MHD |
| LRSP #6 Longcope (2005) | 자기장 위상과 에너지 방출 → 운동론적 에너지 소산의 미시적 과정과 연결 / Magnetic topology and energy release → connects to microscopic kinetic dissipation |
