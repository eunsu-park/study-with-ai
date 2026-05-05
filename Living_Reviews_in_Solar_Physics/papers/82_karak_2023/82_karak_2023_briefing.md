---
title: "Pre-Reading Briefing: Models for the Long-Term Variations of Solar Activity"
paper_id: "82"
topic: Living_Reviews_in_Solar_Physics
date: 2026-04-23
type: briefing
---

# Models for the Long-Term Variations of Solar Activity: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Karak, B. B. (2023). Models for the long-term variations of solar activity. *Living Reviews in Solar Physics*, 20, 3. DOI: 10.1007/s41116-023-00037-y
**Author(s)**: Bidya Binay Karak (IIT Banaras Hindu University)
**Year**: 2023

---

## 1. 핵심 기여 / Core Contribution

**한국어**
본 논문은 태양 자기 활동의 11년 주기를 넘어서는 장기 변동 — 그랜드 미니마(grand minima), 그랜드 맥시마(grand maxima), 그네비셰프–올 규칙(Gnevyshev–Ohl rule), 글라이스버그 주기(Gleissberg cycle, ~90–100 yr), 수에스/드 브리(Suess/de Vries) 주기(~205–210 yr) — 을 설명하는 다이나모 모델 전반을 체계적으로 정리한 리뷰이다. Karak은 모든 변동을 세 가지 근본 원인 — (1) 유동에 대한 자기 피드백 (로렌츠 힘, Λ-quenching 등), (2) 확률적 강제(stochastic forcing, BMR tilt scatter·α-effect 요동), (3) 다이나모 과정에서의 시간 지연(time delay) — 으로 축약하고, Babcock–Leighton 플럭스 수송 다이나모(FTD)를 중심 플랫폼으로 삼아 각 메커니즘을 평가한다. 결론은 태양 다이나모가 약하게 초임계(weakly supercritical) 영역에서 작동하며, 장기 변동의 주요 원인이 확률적 효과라는 것이다.

**English**
This review systematically surveys dynamo models explaining the long-term modulations of solar magnetic activity that go beyond the 11-year cycle: grand minima, grand maxima, the Gnevyshev–Ohl (Even–Odd) rule, the Gleissberg cycle (~90–100 yr) and the Suess/de Vries cycle (~205–210 yr). Karak condenses every mechanism proposed in the literature into three root causes — (1) magnetic feedback on large-scale flows (Lorentz force, Λ-quenching), (2) stochastic forcing (scatter in Babcock–Leighton (BL) source terms, fluctuations in α), and (3) time delays intrinsic to spatially segregated poloidal and toroidal source regions. Using Babcock–Leighton flux-transport dynamos (FTD) as the central framework, he argues that the solar dynamo is likely operating only slightly above criticality and that stochastic effects — not strong nonlinear feedback — are the dominant cause of grand minima and long-term cycle amplitude modulation.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

**한국어**
- 1976년 Eddy의 Maunder Minimum 재발견 이후 "태양 주기는 주기적이지 않다"는 인식이 고착되었다.
- 1990년대–2000년대: Parker(1955), Babcock(1961), Leighton(1964) 이후 발전한 평균장 다이나모 이론이 확률적 강제(Choudhuri 1992, Hoyng 1993)와 결합하여 장기 변동을 설명하기 시작.
- 2000년 Charbonneau & Dikpati의 2D FTD 모델에 α 요동 도입 이후 BL 프로세스 요동이 주류가 되었다.
- 2010–2020년대: 11,400년 ¹⁴C 재구성(Usoskin 등), 3D BL 다이나모(Miesch & Dikpati 2014, Karak & Miesch 2017), 2x2D (Lemerle & Charbonneau 2017)가 가능해지며 BMR tilt scatter의 중요성이 부각됨.
- 본 논문은 2023년 시점에서 해당 분야의 종합 정리이다.

**English**
- After Eddy's 1976 rediscovery of the Maunder Minimum, irregularity became a central feature of the solar cycle.
- 1990s–2000s: Mean-field dynamos (Parker 1955, Babcock 1961, Leighton 1964) combined with stochastic forcing (Choudhuri 1992, Hoyng 1993) began to naturally reproduce long-term variations.
- After Charbonneau & Dikpati 2000 introduced α-fluctuations into a 2D BL FTD, stochastic BL process became the canonical framework.
- In the 2010s–2020s, 11,400-yr ¹⁴C reconstructions (Usoskin et al.), 3D BL models (Miesch & Dikpati 2014; Karak & Miesch 2017) and 2x2D models (Lemerle & Charbonneau 2017) elevated BMR tilt-scatter to centre stage.
- This 2023 review synthesises the state of the field.

### 타임라인 / Timeline

```
1610 ─ Galileo sunspot observations begin
1645 ─ Maunder Minimum starts (~70 yr)
1715 ─ End of Maunder Minimum
1844 ─ Schwabe discovers 11-yr cycle
1908 ─ Hale's law (polarity)
1919 ─ Joy's law (Hale, Ellerman, Nicholson)
1939 ─ Gleissberg cycle (~90 yr)
1948 ─ Gnevyshev–Ohl rule (Even–Odd)
1955 ─ Parker αΩ dynamo
1961 ─ Babcock phenomenological model
1964 ─ Leighton model
1976 ─ Eddy rediscovers Maunder Minimum
1988 ─ Hoyng: α-fluctuations
1992 ─ Choudhuri: stochastic BL
2000 ─ Charbonneau & Dikpati: stochastic 2D FTD
2004 ─ Usoskin: 11,000-yr reconstruction
2010 ─ Karak: meridional flow drop → grand min.
2014 ─ Miesch & Dikpati: first 3D BL dynamo
2017 ─ Lemerle & Charbonneau 2x2D; Karak & Miesch 3D BL
2018 ─ Karak & Miesch: recovery from grand minima via pumping
2023 ─ Karak review (this paper) ── WE ARE HERE
```

---

## 3. 필요한 배경 지식 / Prerequisites

**한국어**
- **MHD 방정식**: induction equation $\partial_t \mathbf{B} = \nabla \times (\mathbf{v}\times\mathbf{B} - \eta\nabla\times\mathbf{B})$ 과 Navier–Stokes 방정식.
- **평균장 이론**: Reynolds 분해, mean EMF $\overline{\mathcal{E}} = \overline{v'\times B'} = \alpha\overline{B} - \eta_t \nabla\times\overline{B}$.
- **축대칭 다이나모**: poloidal–toroidal 분해, $\overline{B}_p = \nabla\times(A\hat{\phi})$.
- **Babcock–Leighton 과정**: 기울어진 BMR의 확산으로 poloidal field 생성 (Joy's law, tilt angle).
- **Flux transport dynamo**: meridional circulation이 equatorward return flow로 toroidal belt를 이동시킴.
- **확률 과정**: Wiener process, Poisson/exponential 분포, 로그정규 분포, Ornstein–Uhlenbeck 과정.
- **비선형 동역학**: Hopf bifurcation, period-doubling, chaotic iterative map.

**English**
- **MHD**: induction equation and Navier–Stokes equation.
- **Mean-field theory**: Reynolds decomposition, mean EMF $\overline{\mathcal{E}} = \alpha\overline{B} - \eta_t\nabla\times\overline{B}$.
- **Axisymmetric dynamo**: poloidal–toroidal decomposition, vector potential $A(r,\theta,t)$.
- **Babcock–Leighton process**: poloidal field from tilted BMR decay (Joy's law).
- **Flux-transport dynamo**: meridional circulation return flow advects the toroidal belt equatorward.
- **Stochastic processes**: Wiener process, Poisson/exponential waiting-time distributions, log-normal distribution, Ornstein–Uhlenbeck process.
- **Nonlinear dynamics**: Hopf bifurcation, period-doubling, chaotic iterative maps.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **Grand minimum** | 1–3 주기 이상 지속되는 현저한 저활성 에피소드 (Maunder ≈ 70 yr). 11,400년 중 ~17% 비중, Spörer형(>100 yr)과 Maunder형(30–90 yr). Extended low-activity episodes; Sun spent ≈17% of last 11,400 yr in grand minima. |
| **Grand maximum** | 반대로 현저한 고활성 (Modern Maximum ~1960). 11,000년 중 23회, 총 12% 비중. Elevated-activity episodes, e.g. Modern Maximum (~1960). |
| **Gnevyshev–Ohl rule** | Even cycle + 이후 odd cycle 쌍에서 odd cycle의 sunspot sum이 더 큼. 절대 법칙은 아님 (pairs 4–5, 22–23 위반). Even–Odd effect: odd cycle stronger than preceding even cycle. |
| **Gleissberg cycle** | ~90–100 yr 변조. Smoothed detrended amplitude에서 발견. ~90–100 yr modulation of cycle amplitudes. |
| **Suess/de Vries cycle** | ~205–210 yr 주기, cosmogenic ¹⁰Be/¹⁴C에서 검출. ~205–210 yr modulation detected in cosmogenic isotopes. |
| **Babcock–Leighton (BL) process** | Joy's law로 기울어진 BMR의 확산에 의한 poloidal field 생성. 1 BMR당 ~5×10²¹ Mx net flux. Poloidal field generation via decay of tilted BMRs. |
| **Flux transport dynamo (FTD)** | Toroidal belt의 적도 방향 이동이 dynamo wave 대신 meridional circulation return flow에 의해 일어나는 모델. Dynamo in which return meridional flow drives equatorward toroidal migration. |
| **α-quenching** | $\alpha \to \alpha_0/[1+(B/B_{eq})^2]$. 다이나모를 포화시키는 비선형성. Nonlinear saturation of α by magnetic back-reaction. |
| **Tilt scatter / Joy's law σ** | Joy's law 평균 기울기 주위의 분산 ($\sigma \sim 15$–$20°$). BMR당 포함 poloidal flux 랜덤화. Scatter in BMR tilt around Joy's law mean, driving stochasticity. |
| **Rogue BMR** | Joy's law에 역행하거나 크게 기울어진 BMR로 단일 사건이 다이나모 궤적을 바꿀 수 있음. Anomalous BMR that can single-handedly alter cycle outcome (Nagy et al. 2017). |
| **Iterative map** | 시간 지연 BL 다이나모를 이산 반복 $p_{n+1} = \alpha f(p_n) p_n$로 축약. Discrete 1D reduction capturing delay + nonlinearity. |
| **Weakly supercritical / near-critical** | $D/D_c \approx 2$ 부근. Long-term cycle memory · grand minima 발현에 유리. Regime just above dynamo bifurcation, favouring grand minima. |
| **Magnetic pumping** | 하향 convective pumping으로 poloidal field를 CZ 내부로 수송. Downward transport of poloidal flux by convective pumping. |
| **Dynamo number D** | $D = \alpha_0\Delta\Omega R_\odot^3/\eta_0^2$. 다이나모 효율의 무차원 척도. Dimensionless measure of dynamo efficiency. |

(13 terms provided / 13개 용어 수록)

---

## 5. 수식 미리보기 / Equations Preview

**한국어 / English**

**(1) Babcock–Leighton FTD 기본 방정식 / Core FTD equations**

$$\frac{\partial A}{\partial t} + \frac{1}{s}(\mathbf{v}_m\cdot\nabla)(sA) = \eta_t\left(\nabla^2-\frac{1}{s^2}\right)A + \alpha B$$

$$\frac{\partial B}{\partial t} + \frac{1}{r}\left[\frac{\partial(rv_rB)}{\partial r}+\frac{\partial(v_\theta B)}{\partial \theta}\right] = \eta_t\left(\nabla^2-\frac{1}{s^2}\right)B + s(\mathbf{B}_p\cdot\nabla)\Omega + \frac{1}{r}\frac{d\eta_t}{dr}\frac{\partial(rB)}{\partial r}$$

여기서 $s = r\sin\theta$. 두 번째 항이 poloidal field advection, $\alpha B$ 항이 BL source, $s(\mathbf{B}_p\cdot\nabla)\Omega$가 Ω effect. / Here $s = r\sin\theta$; the $\alpha B$ term is the BL source and $s(\mathbf{B}_p\cdot\nabla)\Omega$ represents the Ω effect.

**(2) 확률적 α-effect / Stochastic α (Schmitt et al. 1996 형식)**

$$\alpha \to [1 + s\,\sigma(t)]\,\alpha, \quad \sigma(t) \in [-1,1]$$

$s=1$일 때 ~200% 요동. 이후 $\alpha \to \alpha_0/[1+(B/B_0)^2]$로 quenching. / With $s=1$ the stochastic term gives ≈200% fluctuation; quenching $\alpha\to\alpha_0/[1+(B/B_0)^2]$ saturates the field.

**(3) Iterative map / 반복 사상 (Charbonneau 2001)**

$$p_{n+1} = \alpha\, f(p_n)\, p_n$$

$f(p_n) = \frac{1}{4}[1+\mathrm{erf}((p_n-p_1)/w_1)][1-\mathrm{erf}((p_n-p_2)/w_2)]$로 설정하면 period-doubling과 간헐성이 자연스럽게 발생. / The double-erf nonlinearity yields period doubling and intermittency.

**(4) Cameron–Schüssler stochastic normal-form model**

$$dX = \left(\beta + i\omega_0 - (\gamma_r + i\gamma_i)|X|^2\right) X\, dt + \sigma X\, dW_c$$

$X$는 복소장(real/imag = toroidal/poloidal), $\beta$가 supercriticality, $\sigma$가 잡음 강도. / X is a complex field (real = toroidal, imag = poloidal); β sets supercriticality, σ sets noise amplitude.

**(5) Buoyancy time delay in BL source (Jouve et al. 2010)**

$$\text{Source} = \frac{\alpha B(0.7R_\odot,\theta,t-\tau_B)}{1 + [B(0.7R_\odot,\theta,t-\tau_B)/B_0]^2}, \quad \tau_B \sim \tau_0/B^2$$

자기장이 강할수록 flux tube가 빨리 올라와 $\tau_B$가 짧음. / Stronger tubes rise faster, so delay $\tau_B$ shortens with field strength.

---

## 6. 읽기 가이드 / Reading Guide

**한국어**
1. **Sect. 2 관측 정리** (2 페이지): 그랜드 미니마/맥시마, 주기 타입을 외울 것. Figs. 1 & 2의 ¹⁴C 재구성이 표적 관측이다.
2. **Sect. 3–4 다이나모 개요** (6 페이지): 이미 Parker/Babcock/Leighton 논문을 읽었다면 빠르게 통과. Eqs. (3)–(12)는 평균장 FTD의 기본 세트.
3. **Sect. 5 세 가지 원인** (4 페이지): "flow feedback / stochastic forcing / time delay" 삼분법이 전체 논문의 틀. 이 장에서 개념적 뼈대 확보.
4. **Sect. 6 평균장 FTD 모델** (17 페이지): 최대 분량. 6.2(요동) 6.3(특정 비선형성) 6.4(시간 지연)는 핵심. Fig. 12의 $\sigma_\delta$ 시나리오 비교가 압권.
5. **Sect. 7 MHD 시뮬레이션** (4 페이지): convection 기반 결과. BMR 부재가 큰 약점.
6. **Sect. 8 열린 질문** (2 페이지): bimodal distribution (Fig. 25), Gnevyshev–Ohl 원인, Gleissberg 원인.
7. **Sect. 9 요약**: 약하게 초임계이며 확률적 효과가 우세하다는 저자의 결론.
8. **그림 우선순위**: Fig. 2 (¹⁴C), Fig. 4 (Joy's law scatter), Fig. 6 (meridional flow drop), Fig. 9 (grand min 통계), Fig. 12 (tilt scatter 효과), Fig. 17 (critical vs supercritical), Fig. 25 (bimodal PDF).

**English**
1. **Sect. 2 Observations** (~2 pp): memorise types of grand minima/maxima; Figs. 1 & 2 are the observational targets.
2. **Sect. 3–4 Dynamo overview** (~6 pp): skim if familiar with Parker/Babcock/Leighton; Eqs. (3)–(12) set the mean-field FTD framework.
3. **Sect. 5 Three causes** (~4 pp): the flow-feedback / stochastic / time-delay trichotomy is the skeleton of the whole review.
4. **Sect. 6 Mean-field BL/FTD models** (~17 pp, the bulk): focus on 6.2 (fluctuations), 6.3 (specific nonlinearities), 6.4 (time-delay). Fig. 12 (tilt-scatter $\sigma_\delta$ panels) is the centrepiece.
5. **Sect. 7 MHD simulations** (~4 pp): global convection results; note the absence of explicit BMRs.
6. **Sect. 8 Open questions** (~2 pp): bimodal PDF (Fig. 25), origin of Gnevyshev–Ohl, Gleissberg.
7. **Sect. 9 Summary**: author's conclusion — weakly supercritical, stochastic-driven.
8. **Figure priority**: Fig. 2 (¹⁴C), Fig. 4 (Joy's law scatter), Fig. 6 (meridional drop triggers Maunder), Fig. 9 (grand-min stats), Fig. 12 (tilt-scatter effect), Fig. 17 (critical vs supercritical), Fig. 25 (bimodal PDF).

---

## 7. 현대적 의의 / Modern Significance

**한국어**
- **Cycle 25 예측**: 이 리뷰는 cycle 24가 기록상 가장 약한 주기 중 하나였음을 회고하며, BL dynamo 기반 polar precursor가 cycle 25 amplitude를 예측하는 원리를 뒷받침한다.
- **우주 기후(space climate)**: Maunder Minimum급 에피소드가 복사·우주선 플럭스·기후에 미치는 영향은 GPS, 전력망, 승무원 피폭과 직결된다(Temmer 2021).
- **항성 다이나모**: Baliunas HK 프로그램, Metcalfe 2016, Baum 2022 등 sun-like star의 grand-minimum 후보(HD 4915, HD 166620)는 이 리뷰의 "약한 초임계 시나리오"와 일관.
- **기계 학습/데이터 동화**: Hung et al. 2017의 variational data assimilation이 meridional flow 변동을 추정한 사례처럼, 확률적 FTD는 Bayesian/ML 예보에 가장 적합한 "단순하지만 풍부한" 모델을 제공한다.
- **Bimodal 상태**: "regular vs grand-minimum mode"의 이중 분포는 dual-poloidal source(classical α + BL) 개념으로 이어지며 태양 다이나모의 근본 대칭성 문제를 제기.

**English**
- **Cycle 25 forecasting**: the review contextualises the unusually weak Cycle 24, and supports BL-based polar-precursor prediction methods for Cycle 25 amplitude.
- **Space climate**: Maunder-type episodes directly affect radiation, cosmic-ray flux and Earth's climate — relevant to GPS, power grids, astronaut exposure (Temmer 2021).
- **Stellar dynamos**: grand-minimum candidates in sun-like stars (HD 4915, HD 166620; Baum et al. 2022) align with Karak's weakly-supercritical scenario for the Sun.
- **Data assimilation / ML**: stochastic FTD is the simplest framework rich enough for Bayesian / machine-learning forecasting (e.g. Hung et al. 2017 variational assimilation of meridional flow).
- **Bimodal state**: the "regular vs grand-minimum mode" distribution motivates dual-poloidal-source models (classical α + BL), probing deeper symmetry properties of the solar dynamo.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
