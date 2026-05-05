---
title: "Dynamo Models of the Solar Cycle"
authors: Paul Charbonneau
year: 2010
journal: "Living Reviews in Solar Physics, 7, 3"
doi: "10.12942/lrsp-2010-3"
topic: Living Reviews in Solar Physics / Solar Dynamo
tags: [solar-dynamo, MHD, mean-field-theory, Babcock-Leighton, flux-transport, alpha-effect, Omega-effect, tachocline, grand-minima, solar-cycle]
status: completed
date_started: 2026-04-19
date_completed: 2026-04-19
---

# 20. Dynamo Models of the Solar Cycle / 태양 주기의 다이나모 모델

---

## 1. Core Contribution / 핵심 기여

**한국어:**
이 논문은 태양이 **스스로 ~11년 주기의 자기장을 생성·유지하는 메커니즘**을 설명하는 모든 주요 이론을 한 자리에 정리한 결정판 리뷰다. Charbonneau는 자기유체역학(MHD) 유도 방정식에서 출발해, 축대칭 분해를 통한 포로이달/토로이달 이원 표현, 평균장 전기역학의 α-효과·Ω-효과·난류 확산율, Babcock-Leighton 표면 플럭스 메커니즘, 그리고 자오선 순환이 주기 조절을 담당하는 flux-transport dynamo까지를 체계적으로 유도한다. 이 리뷰의 진정한 기여는 **각 모델의 정의, 지배 방정식, 관측 제약에 대한 성공/실패, 그리고 서로 간의 관계**를 하나의 일관된 형식으로 비교한 점에 있다. 또한 비선형 포화(α-quenching, Malkus-Proctor 효과), grand minima 같은 주기의 장기 변동성, 예측 가능성, 3D 글로벌 MHD 시뮬레이션을 다루어 **2010년 당시 태양 다이나모 연구의 현재 수준**을 고정시켰다.

**English:**
This is the definitive review of all major theories explaining how the Sun **self-generates and sustains its ~11-year magnetic cycle**. Starting from the MHD induction equation, Charbonneau systematically derives the axisymmetric toroidal/poloidal decomposition, the mean-field electrodynamics framework with its α-effect, Ω-effect, and turbulent diffusivity, the Babcock-Leighton surface-flux mechanism, and the flux-transport dynamo in which meridional circulation sets the cycle period. The review's real contribution lies in comparing **each model's definition, governing equations, successes and failures against observational constraints, and mutual relationships** within a single consistent formalism. It also covers nonlinear saturation (α-quenching, Malkus-Proctor effect), long-term variability such as grand minima, predictability, and 3D global MHD simulations — together fixing the "state of the art" of solar dynamo theory as of 2010.

---

## 2. Reading Notes / 읽기 노트

### §1. Introduction / 서론

**한국어:**
Charbonneau는 태양 주기 모델링의 **핵심 역설**로 시작한다. 태양 주기는 놀랄 만큼 규칙적(11년)이면서도 실질적으로 불규칙하다(주기 길이 9–14년, 진폭 ~2배 변동, Maunder Minimum 같은 grand minima). 따라서 성공적인 다이나모 이론은 (a) **주기성**과 (b) **변동성**을 동시에 설명해야 한다. 저자는 "solar cycle"을 **태양의 대규모 내부 자기장 성분의 준주기적 극성 반전**으로 정의하고, 흑점은 그 **표면적 발현(surface manifestation)**일 뿐임을 강조한다. 이 구분이 중요한 이유는, 일부 모델은 표면 장을 직접 다루고(Babcock-Leighton), 일부 모델은 심부 toroidal 장을 다루기(mean-field αΩ) 때문이다.

**English:**
Charbonneau opens with the **central paradox** of solar-cycle modeling: the cycle is remarkably regular (11 years) yet practically irregular (period range 9–14 years, amplitude varying ~2×, grand minima like Maunder). A successful dynamo theory must therefore explain both (a) **periodicity** and (b) **variability**. He defines the "solar cycle" as the **quasi-periodic polarity reversal of the Sun's large-scale internal magnetic field**, and stresses that sunspots are only the **surface manifestation** of this deeper process. The distinction matters because some models deal with surface fields (Babcock-Leighton) while others deal with deep toroidal fields (mean-field αΩ).

### §2. Observations / 관측적 제약

**한국어:**
모든 다이나모 모델이 맞춰야 할 **"열한 가지 관측 사실"**이 정리된다:

1. **11-year Schwabe cycle** — 흑점 수의 평균 주기
2. **22-year Hale cycle** — 자기장 극성을 포함한 완전한 주기
3. **Equatorward drift (Spörer's law)** — 흑점 대역이 주기 동안 ±30° → ±5°로 이동 (butterfly diagram)
4. **Hale's polarity law** — 반구 내 BMR의 leading/following polarity가 한 주기 내 고정
5. **Joy's law** — BMR의 축이 적도 방향으로 약간 기울어짐 (위도 의존 기울기)
6. **Poloidal field reversal at solar maximum** — 극지 자기장은 흑점 최대기 근처에 부호를 바꿈
7. **Differential rotation** — 표면: 적도 460 nHz, 극 320 nHz; 내부: tachocline에서 강체 회전으로 전환
8. **Meridional circulation** — 표면 ~20 m/s 적도→극 흐름, 깊이 ~10 m/s 극→적도 반류
9. **Grand minima** — Maunder Minimum(1645–1715) 등 10–11% 시간 동안 활동 정지
10. **Gleissberg(~90년)·Suess(~210년) 변동** — 장기 진폭 변동
11. **Waldmeier effect** — 주기 상승 속도 ∝ 주기 진폭 (큰 주기일수록 빨리 상승)

Figure 4에서 SOHO/MDI helioseismology로 얻은 **내부 차등 회전 프로파일**이 등장한다. tachocline(~0.7 $R_\odot$)에서 **강한 반경 방향 전단** $\partial\Omega/\partial r$가 발생 — 여기가 Ω-효과의 주요 무대로 지목된다.

**English:**
The **"eleven observational facts"** all dynamo models must match are laid out:

1. **11-year Schwabe cycle** — mean period of sunspot number
2. **22-year Hale cycle** — full period including magnetic polarity
3. **Equatorward drift (Spörer's law)** — sunspot band migrates from ±30° to ±5° over one cycle (butterfly diagram)
4. **Hale's polarity law** — leading/following polarity of BMRs fixed per hemisphere during a cycle
5. **Joy's law** — BMR axis tilted slightly toward equator (latitude-dependent tilt)
6. **Poloidal reversal at solar maximum** — polar field flips near sunspot maximum
7. **Differential rotation** — surface: 460 nHz equator, 320 nHz pole; interior rigid rotation below the tachocline
8. **Meridional circulation** — ~20 m/s surface poleward flow; ~10 m/s deep equatorward return
9. **Grand minima** — e.g. Maunder Minimum (1645–1715), near-shutdown for ~10–11% of the time
10. **Gleissberg (~90 yr) & Suess (~210 yr) variations** — long-term amplitude modulations
11. **Waldmeier effect** — cycle rise-rate ∝ cycle amplitude (larger cycles rise faster)

Figure 4 shows the **internal differential rotation** from SOHO/MDI helioseismology. A strong radial shear $\partial\Omega/\partial r$ appears in the tachocline (~0.7 $R_\odot$), identifying it as the main stage for the Ω-effect.

### §3. Mean-field Electrodynamics / 평균장 전기역학

**한국어:**
본격적인 수식 전개가 시작된다. MHD 유도 방정식:

$$
\frac{\partial \mathbf{B}}{\partial t} = \nabla \times (\mathbf{u}\times \mathbf{B}) - \nabla\times(\eta \nabla\times\mathbf{B})
$$

속도와 자기장을 **평균 + 요동**으로 분해 ($\mathbf{u} = \langle\mathbf{u}\rangle + \mathbf{u}'$, 동일하게 $\mathbf{B}$). 평균을 취하면:

$$
\frac{\partial \langle\mathbf{B}\rangle}{\partial t} = \nabla\times(\langle\mathbf{u}\rangle\times\langle\mathbf{B}\rangle + \boldsymbol{\mathcal{E}}) - \nabla\times(\eta\nabla\times\langle\mathbf{B}\rangle)
$$

여기서 $\boldsymbol{\mathcal{E}} = \langle\mathbf{u}'\times\mathbf{B}'\rangle$는 **난류 전기장(mean electromotive force)**. 이것을 평균장 $\langle\mathbf{B}\rangle$과 그 기울기로 전개:

$$
\boldsymbol{\mathcal{E}}_i = \alpha_{ij}\langle B\rangle_j - \beta_{ijk}\partial_j\langle B\rangle_k
$$

등방 근사 하에서 $\alpha_{ij} = \alpha\delta_{ij}$, $\beta_{ijk} = \eta_T\epsilon_{ijk}$가 되어 **α-효과와 난류 확산율 $\eta_T$**가 나타난다. 물리적 의미:

- $\alpha \sim -\frac{1}{3}\tau_c\langle\mathbf{u}'\cdot(\nabla\times\mathbf{u}')\rangle$ — 난류의 헬리시티(helicity)에서 유래. 북반구에서 (+), 남반구에서 (−).
- $\eta_T \sim \frac{1}{3}\tau_c\langle u'^2\rangle$ — 난류 혼합에 의한 효과적 확산.

이 수식 전개가 다이나모 이론의 **핵심 철학**을 보여준다: 작은 스케일 난류가 **대칭 파괴자(α)**와 **확산자(η_T)**로 요약된다.

**English:**
The full mathematical machinery is set up. The MHD induction equation is decomposed into mean + fluctuation ($\mathbf{u} = \langle\mathbf{u}\rangle + \mathbf{u}'$, similarly $\mathbf{B}$). Averaging yields

$$
\frac{\partial \langle\mathbf{B}\rangle}{\partial t} = \nabla\times(\langle\mathbf{u}\rangle\times\langle\mathbf{B}\rangle + \boldsymbol{\mathcal{E}}) - \nabla\times(\eta\nabla\times\langle\mathbf{B}\rangle),
$$

where $\boldsymbol{\mathcal{E}} = \langle\mathbf{u}'\times\mathbf{B}'\rangle$ is the **mean electromotive force**. Expanding in the mean field gives

$$
\boldsymbol{\mathcal{E}}_i = \alpha_{ij}\langle B\rangle_j - \beta_{ijk}\partial_j\langle B\rangle_k.
$$

Under the isotropic approximation $\alpha_{ij}=\alpha\delta_{ij}$, $\beta_{ijk}=\eta_T\epsilon_{ijk}$, yielding the **α-effect and turbulent diffusivity**:

- $\alpha \sim -\frac{1}{3}\tau_c\langle\mathbf{u}'\cdot(\nabla\times\mathbf{u}')\rangle$ — arises from turbulent helicity; positive in the northern hemisphere, negative in the south.
- $\eta_T \sim \frac{1}{3}\tau_c\langle u'^2\rangle$ — effective diffusivity from turbulent mixing.

This derivation captures the **philosophy** of dynamo theory: small-scale turbulence is summarized into a **symmetry-breaker (α)** and a **diffuser (η_T)**.

### §4. Model Classes / 모델 분류

**한국어:**
이 섹션이 리뷰의 심장부다. 축대칭 분해:

$$
\mathbf{B}(r,\theta,t) = \nabla\times[A(r,\theta,t)\hat{\phi}] + B(r,\theta,t)\hat{\phi}
$$

를 적용하면 평균장 다이나모 방정식은 **두 개의 스칼라 PDE 쌍**이 된다:

$$
\frac{\partial A}{\partial t} = \eta_T\left(\nabla^2 - \frac{1}{\varpi^2}\right)A + \alpha B \qquad (\text{poloidal})
$$

$$
\frac{\partial B}{\partial t} = \eta_T\left(\nabla^2 - \frac{1}{\varpi^2}\right)B + \varpi(\nabla\times A\hat{\phi})\cdot\nabla\Omega \qquad (\text{toroidal})
$$

여기서 $\varpi = r\sin\theta$. 소스 항에서:
- $\alpha B$ 항 → α-효과: toroidal을 poloidal로 재생
- $\varpi\nabla A \cdot \nabla\Omega$ 항 → Ω-효과: 차등 회전이 poloidal을 toroidal로 늘림

모델들은 이 두 소스 항 중 **어떤 것이 지배적인지**로 분류:

| 모델 유형 | α 위치 | Ω 위치 | 주기 조절자 |
|---|---|---|---|
| **αΩ dynamo** | convection zone 전역 | radial shear | 확산 타임스케일 |
| **α²Ω dynamo** | 강한 α+Ω 양쪽 | — | 혼합 |
| **Interface dynamo** | tachocline 바로 위 | tachocline | 저항 |
| **Flux-transport dynamo** | 표면(Babcock-Leighton) | tachocline | 자오선 순환 |
| **Babcock-Leighton** | 표면 BMR 붕괴 | tachocline | 확산 or 이류 |

**Parker-Yoshimura 법칙**이 핵심 제약이다. αΩ 다이나모의 파동 전파 방향:

$$
s_P = \alpha\,\frac{\partial \Omega}{\partial r}
$$

$s_P > 0$이면 극방향, $s_P < 0$이면 적도방향 전파. 태양은 **적도방향(butterfly)** 관측 → 북반구에서 $\alpha > 0$, tachocline에서 $\partial\Omega/\partial r > 0$ (적도), $< 0$ (중위도)이므로 부호 조건 만족이 **미묘한 문제**로 남는다.

**§4.3 Interface dynamo (Parker 1993)**: α-효과(convection zone)와 Ω-효과(tachocline)를 **공간적으로 분리**. 두 영역이 얇은 경계층으로 연결되며, 분리가 강한 α-quenching(약한 영역 α, 강한 영역 Ω)을 자연스럽게 만든다.

**§4.4 Flux-transport dynamo**: 자오선 순환을 대류층 바닥까지 내려보내 **컨베이어 벨트 역할**을 시킨다. 적도방향 반류가 tachocline의 toroidal 장을 적도로 운반해 butterfly 모양을 재현. **주기 길이는 순환 속도에 반비례** — 관측된 ~20 m/s면 ~10–15년 주기가 자연스럽게 나온다.

**§4.5 Babcock-Leighton**: α-효과를 **표면 BMR 붕괴 + Joy's law 기울기**로 대체. Leighton(1969)의 surface flux transport + Babcock(1961)의 주기 개요를 결합. 물리적으로 관측 가능한 α이므로 현대 다이나모의 선두주자.

**English:**
This section is the heart of the review. Applying the axisymmetric decomposition

$$
\mathbf{B}(r,\theta,t) = \nabla\times[A(r,\theta,t)\hat{\phi}] + B(r,\theta,t)\hat{\phi}
$$

converts the mean-field dynamo into **a pair of scalar PDEs**:

$$
\frac{\partial A}{\partial t} = \eta_T\!\left(\nabla^2 - \frac{1}{\varpi^2}\right)A + \alpha B,
$$

$$
\frac{\partial B}{\partial t} = \eta_T\!\left(\nabla^2 - \frac{1}{\varpi^2}\right)B + \varpi(\nabla\times A\hat{\phi})\cdot\nabla\Omega.
$$

Source terms: $\alpha B$ regenerates poloidal from toroidal; $\varpi\nabla A\cdot\nabla\Omega$ shears poloidal into toroidal. Models are classified by which source dominates (see table above).

The **Parker-Yoshimura rule** $s_P = \alpha\,\partial_r\Omega$ is a key constraint: positive $s_P$ → poleward propagation; negative → equatorward. Matching the observed butterfly requires negative $s_P$ in the dynamo region — a subtle sign problem in the Sun.

- **Interface dynamo (Parker 1993)**: α in the convection zone and Ω in the tachocline, **spatially separated** by a thin interface. Natural α-quenching emerges from the separation.
- **Flux-transport dynamo**: uses meridional circulation as a **conveyor belt** from surface to base of convection zone; the equatorward return flow advects toroidal field at the tachocline equatorward, reproducing the butterfly. **Cycle period is inversely proportional to the flow speed** — ~20 m/s gives ~10–15 yr naturally.
- **Babcock-Leighton**: replaces classical α with **surface BMR decay + Joy's law tilt**. Combines Leighton's (1969) surface flux transport with Babcock's (1961) cycle outline. Provides a physically observable α, making it the leading modern paradigm.

### §5. Nonlinear Saturation / 비선형 포화

**한국어:**
선형 다이나모는 $|D| > D_{\rm crit}$에서 **지수적 성장**만 한다 — 실제 태양은 유한한 진폭에서 포화. 주요 메커니즘:

1. **Algebraic α-quenching**:
   $$
   \alpha(\mathbf{B}) = \frac{\alpha_0}{1 + (B/B_{\rm eq})^2}
   $$
   가장 단순. $B_{\rm eq}$는 자기장 에너지가 난류 에너지와 같아지는 값 ($\sim 10^4$ G at tachocline).

2. **Malkus-Proctor mechanism**: Lorentz 힘이 차등 회전을 **변형**시켜 Ω-효과를 약화. 비선형 back-reaction의 진짜 물리적 형태.

3. **Magnetic buoyancy flux loss**: tachocline의 toroidal 장이 임계치 넘으면 부력으로 떠올라 표면에서 빠져나감 — 자연스러운 진폭 제한.

4. **Time-delay models**: flux-transport dynamo에서 순환 시간만큼의 지연이 비선형 결합 시 혼돈(chaos)과 간헐적 진폭 변조 유발.

**Figure 19**가 α-quenched αΩ 다이나모의 시간-위도 도표를 보여주며, butterfly 재현이 가능함을 시연.

**English:**
Linear dynamos only grow exponentially for $|D|>D_{\rm crit}$ — the real Sun saturates. Key mechanisms:

1. **Algebraic α-quenching** $\alpha = \alpha_0/[1+(B/B_{\rm eq})^2]$ with $B_{\rm eq}\sim 10^4$ G in the tachocline.
2. **Malkus-Proctor mechanism**: Lorentz feedback **deforms differential rotation**, weakening Ω. A physically genuine back-reaction.
3. **Magnetic buoyancy flux loss**: toroidal field above a threshold rises buoyantly and leaves the system — a natural amplitude limiter.
4. **Time-delay models**: the finite circulation time in flux-transport dynamos, combined with nonlinearities, produces chaos and intermittent amplitude modulation.

Figure 19 demonstrates that an α-quenched αΩ dynamo can reproduce the butterfly diagram.

### §6. Grand Minima and Long-Term Variability / grand minima와 장기 변동성

**한국어:**
Maunder Minimum(1645–1715)은 단순 결정론 다이나모로는 설명하기 어렵다. 후보 메커니즘:

- **확률적 α-변동**: BMR emergence가 이산적·랜덤하므로 α가 노이즈를 가짐. 충분한 노이즈가 다이나모를 임계치 이하로 끌어내리면 일시적 정지.
- **Parity 전환**: 쌍극자(dipolar)와 사극자(quadrupolar) 모드 사이 스위칭. Maunder Minimum 이후 비대칭 복귀 관측과 일치.
- **혼돈 동역학(chaos)**: 비선형 다이나모가 본질적 혼돈이므로 긴 "낮은 상태" 시간이 자연스럽게 등장.
- **확산-이류 경쟁**: flux-transport dynamo에서 $\eta_T$와 자오선 속도의 비율이 임계선 근처면 간헐성 발생.

Figure 27에서 대표적 grand-minima 모형(Choudhuri et al. 2004)의 동역학이 시연됨.

**English:**
The Maunder Minimum is hard to explain with a purely deterministic dynamo. Candidate mechanisms:

- **Stochastic α-fluctuations** from discrete BMR-emergence noise; sufficient noise can drive the dynamo below critical and temporarily shut it off.
- **Parity flips** between dipolar and quadrupolar modes (consistent with the asymmetric recovery observed after the Maunder Minimum).
- **Chaotic dynamics**: nonlinear dynamos are intrinsically chaotic; long "low states" emerge naturally.
- **Diffusion-advection competition** in flux-transport dynamos near a critical ratio of $\eta_T$ to meridional speed.

Figure 27 demonstrates a representative grand-minima model (Choudhuri et al. 2004).

### §7. Cycle Predictions / 주기 예측

**한국어:**
유명한 **Dikpati-Choudhuri 논쟁**(2006): 두 flux-transport dynamo 그룹이 Cycle 24에 대해 정반대 예측 — Dikpati는 "강함", Choudhuri는 "약함". 후자가 옳았다(Cycle 24는 100년 만에 가장 약). 핵심 차이:

- 과거 polar field → 다음 주기 toroidal: 확산 or 이류?
- Charbonneau는 **관측 가능한 polar field precursor**가 가장 강한 예측 지표임을 강조.

이 논쟁은 다이나모 이론이 **순수 이론에서 예측 과학으로 전환**하는 순간을 상징.

**English:**
The famous **Dikpati-Choudhuri controversy (2006)**: two flux-transport dynamo groups gave opposite predictions for Cycle 24 — Dikpati "strong", Choudhuri "weak". The latter was correct (Cycle 24 was the weakest in a century). The key difference is whether past polar field is carried to the next cycle by diffusion or advection. Charbonneau emphasizes the **polar-field precursor** as the strongest available indicator. The controversy marked the transition of dynamo theory **from pure theory to predictive science**.

### §8. Global MHD Simulations / 글로벌 MHD 시뮬레이션

**한국어:**
Brun, Miesch 등의 ASH 시뮬레이션은 완전 3D 대류 다이나모를 목표로 하나, 2010년 기준 **주기적 반전을 안정적으로 만드는 데는 제한적**. 주된 난관:

- 태양 수준의 Reynolds 수 구현 불가 (10^10+ vs. 10^4 시뮬)
- 관측된 차등 회전 형태 재현이 어려움
- **주기 반전 문제**: 대부분 시뮬이 영구 극성 장만 만듦

이 섹션은 리뷰가 쓰인 시점의 **미해결 전선**임을 솔직히 인정. 2011년 이후 Ghizaru-Charbonneau EULAG-MHD 시뮬이 주기 반전에 성공.

**English:**
Simulations like ASH (Brun, Miesch et al.) aim at full 3D convective dynamos, but as of 2010 **struggle to produce stable periodic reversals**. Main obstacles: unreachable solar Reynolds numbers ($10^{10+}$ vs. ~$10^4$), difficulty matching the observed differential-rotation profile, and especially the **reversal problem** (most simulations yield only permanent polar fields). The section honestly identifies this as the **unsolved frontier** at the time; EULAG-MHD (Ghizaru-Charbonneau) achieved cyclic reversal shortly after (2011).

### §9. Outlook / 전망

**한국어:**
저자는 세 가지 장기 과제를 제시: (1) **α-효과의 미시 기원**(convection simulation의 helicity), (2) **tachocline의 정확한 역할**(α가 어디에? toroidal 저장소인가?), (3) **grand minima의 진정한 원인**(noise vs. chaos vs. parity).

**English:**
Three long-term challenges: (1) the **microscopic origin of α** (helicity from convection simulations), (2) the **precise role of the tachocline** (where is α located? is it the toroidal reservoir?), and (3) the **true cause of grand minima** (noise vs. chaos vs. parity).

---

## 3. Key Takeaways / 핵심 시사점

1. **모든 다이나모 모델은 MHD 유도 방정식 + 평균화의 특수한 경우다 / Every dynamo model is a special case of the MHD induction equation + averaging** — 축대칭 분해와 mean-field 전기역학은 거의 모든 태양 다이나모 연구의 공통 수학 기반이다. 모델의 차이는 **α와 η_T를 어디에, 얼마로 두느냐**일 뿐이다. / The axisymmetric decomposition and mean-field electrodynamics provide the common mathematical foundation. Models differ only in **where α and η_T live, and at what magnitude**.

2. **Cowling 정리는 α-효과의 존재 이유다 / Cowling's theorem is why α-effect exists** — 축대칭 유동만으로는 축대칭 자기장을 유지할 수 없다. α-효과는 3D 난류 운동의 헬리시티를 통해 이 금지를 우회하는 수학적 장치다. / Axisymmetric motions cannot sustain an axisymmetric field. The α-effect is the mathematical device that bypasses this ban via the helicity of 3D turbulent motion.

3. **Ω-효과의 위치는 helioseismology가 결정했다 / Helioseismology fixed the location of the Ω-effect** — tachocline(~0.7 $R_\odot$)의 강한 반경 방향 전단이 toroidal 장 생성의 주 무대로 식별된 것은 1990년대 내부 회전 관측 덕분이다. 이전 대류층 내부 Ω 모델은 사실상 폐기. / The strong radial shear at the tachocline (~0.7 $R_\odot$) was identified as the main stage of toroidal-field production only after 1990s helioseismology. Earlier convection-zone Ω models were effectively retired.

4. **Flux-transport dynamo는 주기 길이의 물리적 기원을 제공한다 / Flux-transport dynamos provide a physical origin for cycle length** — 주기가 $\tau_{\rm diff}$(확산 시간)이 아니라 **자오선 순환 일주 시간**으로 결정되므로, 관측 가능한 속도(~20 m/s)에서 ~11년이 자연스럽게 나온다. 이것이 2000년대 이후 주류가 된 이유. / Cycle period is set by the **meridional circulation time**, not diffusion time — giving ~11 yr naturally from observed ~20 m/s flows. This is why it became the dominant paradigm after 2000.

5. **Babcock-Leighton은 관측 가능한 α-효과다 / Babcock-Leighton provides an observable α-effect** — 고전 mean-field α는 convection zone 난류의 통계로 간접 추정하지만, BMR 기울기(Joy's law)는 magnetogram으로 직접 관측 가능. 따라서 BL 다이나모는 **경험적 교정이 가능한 α**를 준다. / Classical mean-field α is inferred indirectly from turbulence statistics, but BMR tilts (Joy's law) are observable in magnetograms. BL dynamos thus provide an **empirically calibratable α**.

6. **Parker-Yoshimura 부호 규칙은 모든 αΩ 모델의 시험대 / The Parker-Yoshimura sign rule tests every αΩ model** — $s_P = \alpha\,\partial_r\Omega$의 부호가 태양의 적도방향 butterfly와 일치해야 한다. 이 조건이 α와 Ω의 위치를 강하게 제약하고, interface dynamo와 flux-transport dynamo가 등장하게 된 동기가 되었다. / The sign of $s_P$ must match the observed equatorward butterfly, strongly constraining where α and Ω sit — motivating the interface and flux-transport models.

7. **Grand minima 설명은 본질적으로 확률적 / 혼돈적이다 / Explaining grand minima is intrinsically stochastic or chaotic** — 순수 결정론 PDE로는 Maunder Minimum의 70년 정지를 재현할 수 없다. BMR emergence noise, 비선형 혼돈, parity mode 스위칭 중 어느 것이 진짜 원인인지는 여전히 미해결. / No purely deterministic PDE reproduces a 70-year shutdown; whether BMR noise, nonlinear chaos, or parity switching is the true cause remains unsolved.

8. **태양 다이나모 이론은 예측 과학으로 전환 중 / Solar dynamo theory is transitioning to predictive science** — Dikpati-Choudhuri 논쟁이 상징하듯, 이론은 더 이상 단순 "해설"이 아니라 다음 주기 진폭/시작 시점을 정량적으로 예측해야 한다. polar field precursor가 현재 가장 강력한 지표. / As symbolized by the Dikpati-Choudhuri controversy, the theory must now quantitatively predict the next cycle's amplitude and timing — the polar-field precursor is the strongest current indicator.

---

## 4. Mathematical Summary / 수학적 요약

### A. Core equations / 핵심 방정식

**MHD 유도 방정식 / MHD induction equation:**
$$
\frac{\partial \mathbf{B}}{\partial t} = \nabla\times(\mathbf{u}\times\mathbf{B}) - \nabla\times(\eta\nabla\times\mathbf{B})
$$

**자기 Reynolds 수 / Magnetic Reynolds number:**
$$
R_m = \frac{UL}{\eta}
$$
태양 convection zone에서 $R_m \gtrsim 10^6$ → 유도가 확산을 압도.

**축대칭 분해 / Axisymmetric decomposition:**
$$
\mathbf{B} = B\hat{\phi} + \nabla\times(A\hat{\phi}), \quad \mathbf{u} = \mathbf{u}_p + \varpi\Omega\hat{\phi}
$$

### B. Mean-field equations / 평균장 방정식

**일반 형태 / General form:**
$$
\frac{\partial A}{\partial t} + \frac{1}{\varpi}(\mathbf{u}_p\cdot\nabla)(\varpi A) = \eta_T\!\left(\nabla^2 - \frac{1}{\varpi^2}\right)A + \alpha B
$$
$$
\frac{\partial B}{\partial t} + \varpi\mathbf{u}_p\cdot\nabla\!\left(\frac{B}{\varpi}\right) + B\nabla\cdot\mathbf{u}_p = \eta_T\!\left(\nabla^2 - \frac{1}{\varpi^2}\right)B + \varpi(\nabla\times A\hat{\phi})\cdot\nabla\Omega
$$

여기서 $\varpi = r\sin\theta$ (cylindrical radius), $\mathbf{u}_p$ = 자오선 순환, $\Omega(r,\theta)$ = 차등 회전. 오른쪽 항:
- $\alpha B$: α-effect (poloidal regeneration)
- $\varpi(\nabla\times A\hat{\phi})\cdot\nabla\Omega$: Ω-effect (toroidal generation)

### C. Mean-field closures / 평균장 닫힘 관계

**EMF 전개 / Mean EMF expansion:**
$$
\boldsymbol{\mathcal{E}} = \alpha\langle\mathbf{B}\rangle - \eta_T\nabla\times\langle\mathbf{B}\rangle
$$

**α의 Second-Order Correlation Approximation(SOCA) / α in SOCA:**
$$
\alpha = -\frac{\tau_c}{3}\langle\mathbf{u}'\cdot(\nabla\times\mathbf{u}')\rangle
$$
(난류 헬리시티 비례 / proportional to turbulent helicity)

**난류 확산율 / Turbulent diffusivity:**
$$
\eta_T = \frac{\tau_c}{3}\langle u'^2\rangle
$$

### D. Dynamo numbers / 다이나모 수

$$
C_\alpha = \frac{\alpha_0 R_\odot}{\eta_T}, \qquad C_\Omega = \frac{(\Delta\Omega) R_\odot^2}{\eta_T}, \qquad D = C_\alpha C_\Omega
$$

- **αΩ 근사**: $|C_\alpha| \ll |C_\Omega|$ → α-효과의 toroidal 생성 무시.
- **Parker-Yoshimura 부호 규칙**:
  $$
  s_P = \alpha\,\frac{\partial \Omega}{\partial r}
  $$
  $s_P > 0$ poleward, $s_P < 0$ equatorward wave.

### E. α-quenching (nonlinear saturation) / α-담금질

$$
\alpha(B) = \frac{\alpha_0}{1 + (B/B_{\rm eq})^2}, \quad B_{\rm eq}^2 = \mu_0\rho\langle u'^2\rangle
$$

### F. Babcock-Leighton α / BL α-항

BMR 붕괴에 의한 표면 소스:
$$
S_{\rm BL}(r,\theta,t) = s(r)\sin\theta\cos\theta\,f(B_{\rm tach})\,\langle B\rangle_{\rm tach}
$$
$f(B)$는 $B_1 < B < B_2$에서만 1인 임계 함수 (emergence threshold).

### G. Flux-transport cycle period estimate / 플럭스 수송 주기 추정

순환 시간으로부터:
$$
T_{\rm cycle} \sim 2\frac{L_{\rm mer}}{u_{\rm mer}} \approx 2\cdot\frac{R_\odot}{20\,{\rm m/s}} \approx 11\text{ yr}
$$
(관측된 ~20 m/s 순환이 ~11년 주기로 직결.)

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1844  Schwabe — 11년 흑점 주기 발견 / discovery of 11-yr cycle
  │
1908  Hale — 흑점의 Zeeman effect / sunspot Zeeman effect
  │
1919  Hale polarity law + Larmor 다이나모 아이디어 / Hale's law + Larmor's dynamo idea
  │
1933  Cowling — 반다이나모 정리 / antidynamo theorem
  │     ← "축대칭 다이나모 불가능" 장벽
1955  Parker — cyclonic convection & α-효과 씨앗 / α-effect seed
  │
1961  Babcock — 표면 플럭스 기반 주기 모델 / surface-flux cycle model
  │
1966  Steenbeck-Krause-Rädler — 평균장 MHD 체계화 / mean-field MHD formalism
  │
1969  Leighton — 확산 전달 다이나모 / diffusive transport dynamo
  │
1970s αΩ 다이나모 1세대: Stix, Yoshimura, Köhler / first-generation αΩ models
  │
1980s Solar interior helioseismology: differential rotation profile / 내부 회전 관측
  │
1993  Parker — interface dynamo / 인터페이스 다이나모
  │
1995+ Choudhuri, Schüssler, Dikpati — flux-transport dynamo 확립 / FTD established
  │
2004  Choudhuri et al. — stochastic α, grand minima 설명 / stochastic α & grand minima
  │
2006  Dikpati vs Choudhuri — Cycle 24 예측 논쟁 / cycle prediction controversy
  │
▶ 2010  ★ Charbonneau — "Dynamo Models of the Solar Cycle" (이 논문 / THIS PAPER)
  │       전 시대의 종합 / synthesis of the era
  │
2011  Ghizaru, Charbonneau, Smolarkiewicz — EULAG-MHD: 주기 반전 성공
  │                                        / first stable cyclic reversal in global sim
2014  Charbonneau — LRSP 개정판 / revised LRSP review
  │
2017  Cameron & Schüssler — surface-BL observational dynamo / BL 관측 기반 다이나모
  │
2020s Machine-learning hybrid prediction / ML 혼합 예측, PSP/SO in-situ probing
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **LRSP #2 — Hathaway, *The Solar Cycle* (2015)** | 관측 측면의 자매 리뷰 — 11개 관측 제약의 상세 / Observational companion — detailed treatment of the 11 observational constraints | 필수 선행 / Prerequisite |
| **LRSP #15 — Fan, *Magnetic Fields in the Solar Convection Zone* (2021)** | flux tube emergence = BL α-효과의 미시 물리 / flux-tube emergence provides micro-physics of BL α | 직접 연결 / Direct link |
| **Parker (1955) *ApJ 122, 293*** | α-효과 개념의 원조 / original α-effect concept | 이 리뷰 §3의 기초 / foundation of §3 |
| **Steenbeck, Krause, Rädler (1966)** | 평균장 전기역학의 형식화 / formalization of mean-field electrodynamics | §3 수식의 뿌리 / mathematical root of §3 |
| **Babcock (1961) & Leighton (1969)** | BL 메커니즘의 두 기둥 / twin pillars of BL mechanism | §4.5 주제 / main subject of §4.5 |
| **Parker (1993) — interface dynamo** | §4.3의 중심 모델 / central model of §4.3 | α-Ω 공간 분리의 원조 / origin of spatial separation |
| **Dikpati & Gilman (2006) / Choudhuri et al. (2007)** | Cycle 24 예측 논쟁 / Cycle 24 prediction debate | §7 주제 / subject of §7 |
| **LRSP on Helioseismology (Howe 2009)** | tachocline 차등 회전 데이터 / tachocline differential rotation data | Figure 4의 데이터 출처 / data source of Fig 4 |
| **Ghizaru, Charbonneau, Smolarkiewicz (2011)** | 3D global MHD cyclic reversal / 3D 글로벌 MHD 주기 반전 | §8 직후의 돌파구 / breakthrough immediately after §8 |

---

## 7. References / 참고문헌

- Charbonneau, P., "Dynamo Models of the Solar Cycle", *Living Reviews in Solar Physics*, 7, 3, 2010. [DOI: 10.12942/lrsp-2010-3]
- Babcock, H. W., "The Topology of the Sun's Magnetic Field and the 22-YEAR Cycle", *ApJ*, 133, 572, 1961.
- Choudhuri, A. R., Schüssler, M., Dikpati, M., "The solar dynamo with meridional circulation", *A&A*, 303, L29, 1995.
- Choudhuri, A. R., Chatterjee, P., Jiang, J., "Predicting Solar Cycle 24 with a Solar Dynamo Model", *PRL*, 98, 131103, 2007.
- Cowling, T. G., "The magnetic field of sunspots", *MNRAS*, 94, 39, 1933.
- Dikpati, M., Gilman, P. A., "Simulating and Predicting Solar Cycles Using a Flux-Transport Dynamo", *ApJ*, 649, 498, 2006.
- Hale, G. E., "On the Probable Existence of a Magnetic Field in Sun-spots", *ApJ*, 28, 315, 1908.
- Hale, G. E., Ellerman, F., Nicholson, S. B., Joy, A. H., "The Magnetic Polarity of Sun-Spots", *ApJ*, 49, 153, 1919.
- Leighton, R. B., "A Magneto-Kinematic Model of the Solar Cycle", *ApJ*, 156, 1, 1969.
- Parker, E. N., "Hydromagnetic Dynamo Models", *ApJ*, 122, 293, 1955.
- Parker, E. N., "A Solar Dynamo Surface Wave at the Interface between Convection and Nonuniform Rotation", *ApJ*, 408, 707, 1993.
- Steenbeck, M., Krause, F., Rädler, K.-H., "Berechnung der mittleren Lorentz-Feldstärke für ein elektrisch leitendes Medium in turbulenter, durch Coriolis-Kräfte beeinflußter Bewegung", *Z. Naturforsch*, 21a, 369, 1966.
- Yoshimura, H., "Solar-cycle dynamo wave propagation", *ApJ*, 201, 740, 1975.
