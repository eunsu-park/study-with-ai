---
title: "Magnetic Fields in the Solar Convection Zone"
authors: Yuhong Fan
year: 2009
journal: "Living Reviews in Solar Physics"
doi: "10.12942/lrsp-2009-4"
topic: Living_Reviews_in_Solar_Physics
tags: [solar-interior, flux-tube, magnetic-buoyancy, active-region, MHD, dynamo, joys-law]
status: completed
date_started: 2026-04-17
date_completed: 2026-04-17
---

# 18. Magnetic Fields in the Solar Convection Zone / 태양 대류층의 자기장

---

## 1. Core Contribution / 핵심 기여

### English

Fan's 2009 *Living Reviews* article is the canonical synthesis of how **toroidal magnetic flux generated in the tachocline is stored, destabilized, and transported as coherent Ω-loops through the solar convection zone (CZ) to form bipolar active regions**. The review organizes roughly fifteen years of theoretical work around three complementary pillars: (1) the **thin flux tube (TFT) approximation**, which reduces a magnetic rope to a 1-D Lagrangian filament governed by buoyancy, magnetic tension, aerodynamic drag, and the Coriolis force; (2) **2-D and 3-D MHD simulations** of rising tubes in stratified, rotating plasma, which reveal the essential role of field-line twist in preserving tube coherence against vortex shedding; and (3) **confrontation with observed asymmetries** of bipolar magnetic regions (BMRs) — Joy's-law tilt, leading-polarity separation, and Hale-cycle reversal. The paper's central quantitative conclusion is that reproducing the observed latitude band (±30°) and Joy's-law tilt requires initial toroidal fields of $B_0 \sim 3\times 10^4$–$10^5$ G at the base of the CZ — **one to two orders of magnitude above equipartition with convection**, and far stronger than the field that the mean-field dynamo would naively predict. Fan also surveys storage mechanisms (overshoot-layer stability, flux pumping by downdrafts), explosive flattening near the photosphere, and the influence of giant-cell convection, setting the agenda for the next decade's global convective-dynamo simulations.

### 한국어

Fan의 2009년 *Living Reviews* 리뷰는 **태양 타코클라인(tachocline)에서 생성된 토로이달 자기 플럭스가 어떻게 대류층 바닥에 저장되고, 불안정해지며, Ω-루프 형태로 광구까지 일관되게 부상하여 양극성 활동 영역(BMR)을 형성하는가**에 관한 15여 년간의 이론적 발전을 체계적으로 종합합니다. 논문은 세 개의 축으로 전개됩니다: (1) **얇은 플럭스 튜브(TFT) 근사** — 자기 로프를 1차원 라그랑지안 필라멘트로 환원하여 부력·장력·항력·코리올리력의 상호작용을 계산; (2) **2D 및 3D MHD 시뮬레이션** — 부상 중 튜브가 vortex shedding으로 쪼개지지 않으려면 임계 이상의 꼬임(twist)이 필요함을 정량화; (3) **관측된 BMR 비대칭성**과의 대조 — Joy's law 기울기, leading/following 극성 분리, Hale 주기 역전. 핵심 결론은 위도 분포(±30°)와 기울기 각을 재현하려면 대류층 바닥 초기 자기장이 $B_0 \sim 3\times 10^4$–$10^5$ G로 **대류와의 등분배(equipartition) 값보다 1–2차수 강해야 한다**는 것이며, 이는 평균장 다이나모의 직관적 예측을 상회하는 값입니다. 저자는 또한 저장 메커니즘(오버슛층 안정성, 대류 하강류에 의한 플럭스 펌핑), 광구 근처의 폭발적 평탄화, 거대 세포 대류의 영향을 정리하며 이후 10년간의 global convective dynamo 연구 방향을 제시합니다.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction and Observational Motivation / 서론과 관측적 동기 (§1)

**English**
Fan begins by cataloguing the observational facts that any emergence theory must reproduce:
- Sunspots appear within a ±30° latitude band (the "royal zone") — never at poles.
- **Joy's law**: the tilt angle $\gamma$ of a BMR with respect to the east–west direction scales roughly as $\gamma \approx 0.5°\,\sin\lambda$ (Wang & Sheeley 1989; Howard 1991). Leading polarity sits equatorward.
- **Hale's polarity law**: east-west polarity order is fixed within a hemisphere and reverses every ~11 years.
- Leading-following asymmetry: leading polarity is more compact and moves faster.
- Typical BMR flux $\Phi \sim 10^{21}$–$10^{22}$ Mx; emergence time ~days.

He introduces the **flux-tube paradigm**: a toroidal strand generated deep in the CZ becomes buoyant, portions rise as Ω-loops, and the two legs puncture the photosphere as a pair of spots. The rest of the review develops the physics under this paradigm.

**한국어**
논문은 관측적 제약을 먼저 나열합니다:
- 흑점은 적도로부터 ±30° 띠("royal zone") 안에서만 출현하며, 극지에는 나타나지 않습니다.
- **Joy의 법칙**: BMR의 동서 방향에 대한 기울기 각 $\gamma$가 위도 $\lambda$에 대해 $\gamma \approx 0.5° \sin\lambda$로 근사됩니다 (Wang & Sheeley 1989). 선행 극성(leading polarity)이 적도 쪽에 위치합니다.
- **Hale 법칙**: 같은 반구 내 동서 극성 순서가 고정되어 있고 ~11년 주기로 반전됩니다.
- 선행-후행 비대칭성: 선행 극성이 더 조밀하고 빠르게 이동합니다.
- 대표 BMR 플럭스는 $\Phi \sim 10^{21}$–$10^{22}$ Mx, 부상 시간은 며칠 수준입니다.

Fan은 이러한 현상을 설명하는 **플럭스 튜브 패러다임**을 소개합니다: 대류층 깊은 곳에서 생성된 토로이달 자기 가닥이 부력을 얻어 일부가 Ω-루프로 상승하고, 양쪽 다리(leg)가 광구를 관통해 한 쌍의 흑점으로 나타납니다.

### Part II: Storage at the Base of the Convection Zone / 대류층 바닥에서의 저장 (§2)

**English**
The problem: convective flows have timescale $\tau_c \sim H_p/v_c \sim$ 1 month, but the ~11-year dynamo needs to *keep* flux in place between cycles. Two sub-problems:

**(2.1) Why the overshoot layer?** — Parker (1975) showed that flux in a convectively unstable layer experiences buoyancy on a timescale short compared to the cycle. The solution proposed by van Ballegooijen (1982) is to store flux in the **overshoot layer** just below the CZ base, where the stratification is weakly subadiabatic (convective plumes overshoot by inertia but the mean gradient is stable). Thickness is ~0.01–0.1 $H_p$ (~10³–10⁴ km).

**(2.2) Stability of an isolated flux tube** — A horizontal toroidal tube can be held in mechanical equilibrium if lateral pressure balance and curvature/buoyancy effects cancel. The *undular* (Parker) mode destabilizes wave-like perturbations along the tube. Linear stability analyses (Ferriz-Mas & Schüssler 1993, 1995) give a "stability diagram" in $(B_0, \lambda)$ space: tubes with $B \lesssim 10^5$ G at low latitudes are neutrally stable to undular modes for the overshoot-layer stratification. **Equipartition-strength fields ($\sim 10^4$ G) are too buoyant to store** — they would escape in months, not years.

**(2.3) Flux pumping** — 3-D simulations of penetrative convection (Tobias et al. 1998, 2001) show that strong asymmetric downdrafts drag horizontal flux *below* the CZ base and compress it against the stable layer. This provides an active transport mechanism complementing passive overshoot storage.

**한국어**
저장 문제의 핵심: 대류 시간 규모는 $\tau_c \sim H_p/v_c \sim 1$ 개월인데, 다이나모는 11년 주기 동안 플럭스를 그대로 유지해야 합니다. 두 개의 하위 문제가 있습니다:

**(2.1) 왜 오버슛층인가?** — Parker (1975)는 대류 불안정 층의 플럭스가 주기에 비해 훨씬 짧은 시간에 자기 부력으로 빠져나간다는 것을 보였습니다. van Ballegooijen (1982)의 해결책은 대류층 바로 아래 약하게 안정한 **오버슛층**(두께 $\sim 0.01$–$0.1 H_p$, 약 $10^3$–$10^4$ km)에 플럭스를 저장하는 것입니다. 이 층은 대류 플룸이 관성으로 침투하는 동안 평균 성층은 sub-adiabatic으로 안정합니다.

**(2.2) 고립된 플럭스 튜브의 안정성** — 수평 토로이달 튜브는 측면 압력 평형과 곡률/부력이 상쇄될 때 기계적 평형을 유지합니다. *파상(undular, Parker) 모드*는 튜브 축 방향의 파동 섭동을 불안정화합니다. Ferriz-Mas & Schüssler (1993, 1995)의 선형 안정성 분석은 $(B_0, \lambda)$ 공간에서 "안정성 도표"를 제시하는데, **저위도에서 $B \lesssim 10^5$ G 인 튜브만 오버슛층 성층에서 중립 안정**합니다. **등분배 자기장($\sim 10^4$ G)은 너무 부력이 강해 저장이 불가능**하며, 몇 개월 내에 빠져나갑니다.

**(2.3) 플럭스 펌핑** — Tobias 외 (1998, 2001)의 관통 대류 3D 시뮬레이션은 비대칭적 강한 하강류가 수평 플럭스를 대류층 바닥 *아래로* 끌어내려 안정층에 압축시키는 능동적 수송 메커니즘을 보여줍니다. 이는 수동적 오버슛 저장을 보완합니다.

### Part III: Thin Flux Tube Theory / 얇은 플럭스 튜브 이론 (§3)

**English**
The **TFT approximation** treats a flux rope as a 1-D Lagrangian curve when the tube radius $a$ is much smaller than the pressure scale height $H_p$. Each cross section is in instantaneous lateral pressure balance with the surroundings:

$$
p_e = p_i + \frac{B^2}{8\pi}.
$$

The equation of motion per unit mass of a tube element is (eq. 3.4 of the paper):

$$
\rho \frac{d\mathbf{v}}{dt} = -\nabla p_i + \rho\mathbf{g} + \frac{1}{4\pi}(\mathbf{B}\cdot\nabla)\mathbf{B} - C_d \frac{\rho_e |\mathbf{v}_\perp|\mathbf{v}_\perp}{\pi a} - 2\rho\,\mathbf{\Omega}\times\mathbf{v}.
$$

Expanding $(\mathbf{B}\cdot\nabla)\mathbf{B}/4\pi = \partial_s(B^2/8\pi)\hat{l} + (B^2/4\pi)\mathbf{k}$, where $\hat{l}$ is the unit tangent and $\mathbf{k} = d\hat{l}/ds$ is the curvature vector, separates pressure gradient along the tube from magnetic tension perpendicular to it. Mass and flux conservation along the tube close the system:

$$
\rho \, A \, \delta s = \text{const}, \qquad B \, A = \text{const}.
$$

**Key results from TFT computations** (D'Silva & Choudhuri 1993; Fan, Fisher & Deluca 1993; Caligari, Moreno-Insertis & Schüssler 1995):

- **Latitude of emergence**: Coriolis force deflects rising tubes poleward. Reproducing the ±30° royal zone requires $B_0 \approx (3\text{–}10) \times 10^4$ G. Weaker fields ($10^4$ G) emerge too far poleward (λ > 40°); much stronger ($3\times 10^5$ G) emerge too close to the equator.
- **Joy's law tilt**: Coriolis force on the diverging legs during rise imparts an east-west tilt. Matching the observed slope $\tan\gamma \propto \sin\lambda$ requires similar $B_0$.
- **Asymmetric rise speed**: In the rotating frame, retrograde zonal flow develops in the tube apex → leading leg is more vertical, follower leans backward. This reproduces the observed leading-polarity compactness.

Caligari et al. (1995) further showed that tubes must start in **mechanical equilibrium** (not neutral buoyancy) — they begin neutrally buoyant and grow unstable to undular modes with wavelength $L \sim \pi R_\odot$, consistent with the observed active-region size distribution.

**한국어**
**TFT 근사**는 튜브 반지름 $a$가 압력 스케일 높이 $H_p$보다 매우 작을 때, 자기 로프를 1차원 라그랑지안 곡선으로 취급합니다. 각 단면은 주변과 순간적인 측면 압력 평형을 유지합니다:

$$
p_e = p_i + \frac{B^2}{8\pi}.
$$

튜브 요소 단위 질량당 운동 방정식(논문 식 3.4)은 다음과 같습니다:

$$
\rho \frac{d\mathbf{v}}{dt} = -\nabla p_i + \rho\mathbf{g} + \frac{1}{4\pi}(\mathbf{B}\cdot\nabla)\mathbf{B} - C_d \frac{\rho_e |\mathbf{v}_\perp|\mathbf{v}_\perp}{\pi a} - 2\rho\,\mathbf{\Omega}\times\mathbf{v}.
$$

자기 텐서를 $(\mathbf{B}\cdot\nabla)\mathbf{B}/4\pi = \partial_s(B^2/8\pi)\hat{l} + (B^2/4\pi)\mathbf{k}$로 분해하면 튜브 축을 따르는 압력 구배와 수직 방향 **자기 장력**이 구분됩니다. 여기서 $\hat{l}$은 단위 접선, $\mathbf{k} = d\hat{l}/ds$는 곡률 벡터입니다. 질량·플럭스 보존이 시스템을 닫습니다:

$$
\rho \, A \, \delta s = \text{const}, \qquad B \, A = \text{const}.
$$

**TFT 계산의 주요 결과**(D'Silva & Choudhuri 1993; Fan 외 1993; Caligari 외 1995):

- **부상 위도**: 코리올리력이 상승하는 튜브를 극 쪽으로 편향시킵니다. 관측된 ±30° 띠를 재현하려면 $B_0 \approx (3\text{–}10) \times 10^4$ G가 필요합니다. 약한 장($10^4$ G)은 너무 극 가까이(λ > 40°)에서 부상하고, 강한 장($3 \times 10^5$ G)은 너무 적도 가까이에서 부상합니다.
- **Joy의 법칙 기울기**: 상승 중 발산하는 양 다리에 작용하는 코리올리력이 동서 기울기를 만듭니다. $\tan\gamma \propto \sin\lambda$ 관측 기울기를 맞추려면 같은 범위의 $B_0$가 필요합니다.
- **비대칭 부상 속도**: 회전 좌표계에서 튜브 꼭대기에 역방향 zonal flow가 발달 → 선행 다리는 더 수직, 후행 다리는 뒤로 기울어져 관측된 선행 극성 조밀함을 재현합니다.

Caligari 외 (1995)는 또한 튜브가 **기계적 평형**(중립 부력이 아님)에서 출발해야 함을 보였고, 파장 $L \sim \pi R_\odot$의 파상 모드로 성장하여 관측된 활동 영역 크기 분포와 일치한다는 것을 보였습니다.

### Part IV: 2-D and 3-D MHD Simulations / 2D 및 3D MHD 시뮬레이션 (§4)

**English**
TFT is 1-D and cannot capture fragmentation. Full MHD reveals a crucial constraint:

**(4.1) The twist requirement.** Schüssler (1979) and Longcope et al. (1996) showed that an *untwisted* straight tube rising through a stratified medium sheds two counter-rotating vortices (due to the curl of the induced drag), which split the tube into two fragments. Emonet & Moreno-Insertis (1998), in 2-D simulations, demonstrated that a minimum twist parameter

$$
q = \frac{B_\phi}{r\,B_z} > q_\mathrm{cr} \sim \frac{1}{a}\sqrt{\frac{H_p}{H_\rho}}\cdot\frac{1}{\sqrt{\beta_c}}
$$

is required for the tube to remain coherent. Physically, the azimuthal field component provides tension that holds the cross section together against the vortex pair.

**(4.2) 3-D Ω-loop simulations.** Fan (2001, 2008), Abbett et al. (2001) integrated the fully compressible MHD equations in Cartesian boxes spanning a fraction of the CZ. Key findings:
- An initially horizontal tube with an imposed undular perturbation develops a rising apex while anchored legs remain near the base.
- Tubes with sufficient twist preserve their identity; coherent Ω-loops emerge with morphology matching cartoon expectations.
- Rise speeds $v_\mathrm{rise} \sim v_A/\sqrt{\beta}$ (slower than Alfvén by $\sqrt{\beta} \sim 300$).
- Coriolis action on the expanding legs reproduces Joy's-law tilt self-consistently, not just as a 1-D parameter.

**(4.3) Spherical-shell simulations.** Jouve & Brun (2007, 2009) placed rising tubes in a full spherical CZ with solar-like differential rotation. The meridional circulation and latitudinal rotation gradient leave imprints on emergence latitude, though the results are qualitatively consistent with Cartesian predictions.

**한국어**
TFT는 1D이므로 파편화를 포착하지 못합니다. 완전 MHD는 결정적인 제약을 드러냅니다:

**(4.1) 꼬임 요구조건.** Schüssler (1979), Longcope 외 (1996)는 *꼬임이 없는* 직선 튜브가 성층 매질 속에서 상승할 때, 유도 항력의 curl에 의해 두 개의 반대 방향 소용돌이를 생성하여 튜브가 둘로 쪼개진다는 것을 보였습니다. Emonet & Moreno-Insertis (1998)은 2D 시뮬레이션으로 최소 꼬임 파라미터

$$
q = \frac{B_\phi}{r\,B_z} > q_\mathrm{cr} \sim \frac{1}{a}\sqrt{\frac{H_p}{H_\rho}}\cdot\frac{1}{\sqrt{\beta_c}}
$$

이 필요함을 보였습니다. 물리적으로, 방위각 자기장 성분이 제공하는 장력이 소용돌이 쌍의 분열 작용을 상쇄합니다.

**(4.2) 3D Ω-루프 시뮬레이션.** Fan (2001, 2008), Abbett 외 (2001)는 대류층의 일부를 덮는 직교 격자에서 완전 압축성 MHD 방정식을 적분했습니다. 주요 발견:
- 수평 방향 초기 튜브에 파상 섭동이 가해지면 꼭대기는 상승, 다리는 바닥 근처에 고정된 Ω-구조가 형성됩니다.
- 충분한 꼬임을 가진 튜브는 정체성을 보존하며, 카툰 기대와 일치하는 일관된 Ω-루프가 부상합니다.
- 부상 속도는 $v_\mathrm{rise} \sim v_A/\sqrt{\beta}$ (알펜 속도보다 $\sqrt{\beta} \sim 300$ 배 느림).
- 팽창하는 양 다리에 작용하는 코리올리력이 Joy's law 기울기를 1D 파라미터가 아닌 자기일관적으로 재현합니다.

**(4.3) 구형 쉘 시뮬레이션.** Jouve & Brun (2007, 2009)은 태양과 유사한 차등 회전을 갖는 완전 구형 대류층에 부상 튜브를 배치했습니다. 자오면 순환과 위도별 자전 구배가 부상 위도에 영향을 미치지만, 결과는 직교 격자 예측과 질적으로 일치합니다.

### Part V: Interaction with Convection / 대류와의 상호작용 (§5)

**English**
Earlier TFT and MHD box simulations assumed a quiet, hydrostatic background. Reality: the CZ is vigorously convecting with $v_c \sim 100$ m/s at depth. Fan surveys the implications:
- **Drag dominates for weak tubes**: if $B \lesssim B_\mathrm{eq} \sim 10^4$ G, convective drag exceeds buoyancy and the tube is carried along with the flow — emergence becomes chaotic, the latitude constraint is lost.
- **Strong tubes punch through**: for $B \gtrsim 3\times 10^4$ G, magnetic tension dominates over convective drag (the ratio scales as $(B/B_\mathrm{eq})^2$), restoring the ordered TFT behavior. This is a second argument for strong-field storage.
- **Giant cells**: Fan notes that coherent giant convective cells (Miesch et al. 2008) could systematically steer rising tubes. Definitive simulations require global scale and were computationally prohibitive at the time of the review.

**한국어**
이전 TFT와 MHD 박스 시뮬레이션은 정적인 배경을 가정했습니다. 실제 대류층은 깊은 곳에서 $v_c \sim 100$ m/s의 격렬한 대류가 존재합니다. Fan은 그 영향을 정리합니다:
- **약한 튜브는 항력이 지배**: $B \lesssim B_\mathrm{eq} \sim 10^4$ G이면 대류 항력이 부력을 능가하여 튜브가 흐름에 실려가고, 부상이 무질서해져 위도 제약이 사라집니다.
- **강한 튜브는 관통**: $B \gtrsim 3 \times 10^4$ G이면 자기 장력이 대류 항력을 능가하여($(B/B_\mathrm{eq})^2$에 비례) TFT의 질서 있는 거동이 회복됩니다. 이는 강자기장 저장을 지지하는 두 번째 논거입니다.
- **거대 세포**: Miesch 외 (2008)가 제안한 일관된 거대 대류 세포가 부상 튜브를 체계적으로 유도할 수 있습니다. 확정적 시뮬레이션은 당시 global 규모 계산 자원이 부족하여 미해결로 남아 있었습니다.

### Part VI: Near-Surface Emergence / 광구 근처 부상 (§6)

**English**
As the tube apex rises above depth $\sim -20$ Mm, the density scale height shrinks rapidly. The TFT approximation breaks down because $a \to H_p$, and a qualitatively new regime appears:
- **Explosive expansion ("flux explosion")**: Moreno-Insertis et al. (1995) found that above a critical depth the interior density drops so fast that the tube flattens and disperses laterally. Cheung et al. (2007, 2008) performed realistic surface emergence MHD with radiative transfer and showed that the apex splits into finger-like undulations (the "sea-serpent" pattern) while the legs remain cohesive.
- **Horizontal fields first**: in photospheric observations, emerging active regions first show horizontal fields granulating between the poles — a signature of the apex passing through the surface.
- **Umbral field strength**: surface measurements give $B \approx 2$–$3$ kG in fully-developed sunspots, much less than the $10^5$ G at the CZ base. The reduction is set by conservation of $B/\rho$ along the rising tube and the drop in $\rho$ from $\sim 0.2$ to $\sim 10^{-7}$ g/cm³.

**한국어**
튜브 꼭대기가 깊이 $\sim -20$ Mm 위로 상승하면 밀도 스케일 높이가 급격히 감소합니다. $a \to H_p$가 되어 TFT 근사가 깨지고, 질적으로 새로운 영역이 나타납니다:
- **폭발적 팽창("flux explosion")**: Moreno-Insertis 외 (1995)는 임계 깊이 위에서 튜브 내부 밀도가 너무 빨리 떨어져 튜브가 수평으로 퍼진다는 것을 보였습니다. Cheung 외 (2007, 2008)는 복사 전달을 포함한 현실적 표면 부상 MHD로 꼭대기가 손가락 모양의 물결("sea-serpent" 패턴)로 분열되는 동안 다리는 응집력을 유지하는 것을 보였습니다.
- **수평 장이 먼저**: 광구 관측에서 부상하는 활동 영역은 두 극 사이 입자 경계(granulation) 위로 수평 자기장이 먼저 나타납니다 — 꼭대기가 표면을 통과한 signature입니다.
- **흑점 본영 자기장 강도**: 표면 측정은 발달된 흑점에서 $B \approx 2$–$3$ kG로, 대류층 바닥의 $10^5$ G보다 훨씬 작습니다. 이 감소는 상승 튜브를 따라 $B/\rho$가 보존되고 $\rho$가 $\sim 0.2$에서 $\sim 10^{-7}$ g/cm³까지 떨어지는 것으로 설명됩니다.

### Part VII: Summary and Open Questions / 요약과 미해결 문제 (§7)

**English**
Fan closes by listing the open questions as of 2009:
1. Is the field really ~10⁵ G? Alternative "weak-field" scenarios driven by convection remain viable if small-scale dynamos dominate.
2. Where does the twist come from? Options: Σ-effect in the tachocline, helical turbulence, or Coriolis spin-up during rise. Each gives different hemispheric helicity rules.
3. How does giant-cell convection quantitatively steer rising tubes?
4. What connects the interior TFT picture to the surface flux rope that eventually erupts as a CME?

**한국어**
Fan은 2009년 시점의 미해결 문제를 정리하며 마칩니다:
1. 자기장이 정말 $\sim 10^5$ G인가? 작은 규모 다이나모가 지배한다면 대류 주도의 "약한 장" 시나리오도 여전히 가능합니다.
2. 꼬임의 기원은? 타코클라인의 Σ-효과, 헬리컬 난류, 또는 부상 중 코리올리 spin-up 중 하나이며, 각각이 반구별 헬리시티 법칙에 다른 예측을 줍니다.
3. 거대 세포 대류가 부상 튜브를 정량적으로 어떻게 유도하는가?
4. 내부 TFT 그림과 최종적으로 CME로 폭발하는 표면 플럭스 로프 사이의 연결은?

---

## 3. Key Takeaways / 핵심 시사점

1. **Strong-field scenario is dominant** — 강자기장 시나리오가 표준이다. TFT 계산은 위도 ±30° 띠와 Joy's law 기울기를 동시에 재현하려면 대류층 바닥 초기 자기장이 $B_0 \approx (3\text{–}10) \times 10^4$ G여야 한다는 것을 일관되게 도출한다. 이는 등분배 값($\sim 10^4$ G)보다 1–2차수 강하다.

2. **Overshoot layer solves Parker's storage problem** — 오버슛층이 Parker의 저장 문제를 해결한다. 대류층 내부에서는 플럭스가 주기보다 훨씬 짧은 시간에 부상해버리지만, 바닥 아래 약하게 안정한 오버슛층은 $\sim 10^5$ G의 플럭스를 11년 주기 내내 유지할 수 있는 유일한 위치이다.

3. **Twist is a physical necessity, not an aesthetic choice** — 꼬임은 물리적 필연이지 선택이 아니다. 꼬이지 않은 튜브는 vortex shedding으로 단 몇 $H_p$ 내에 쪼개져 버린다. $q > q_\mathrm{cr}$ 조건이 관측된 helicity의 반구 법칙(음수 북반구/양수 남반구) 기원에 대한 첫 제약을 제공한다.

4. **Coriolis action explains Joy's law self-consistently** — 코리올리 작용이 Joy의 법칙을 자기일관적으로 설명한다. 상승 중 양 다리가 각운동량 보존에 의해 편향되어 리딩 극성이 적도 쪽, 후행 극성이 극 쪽에 놓인다. 위도 의존성 $\sin\lambda$는 코리올리 항의 위도 구성 성분 $\Omega \sin\lambda$에서 직접 나온다.

5. **Convection can be overcome only by strong fields** — 강자기장만이 대류를 극복할 수 있다. 대류 항력과 자기 장력의 비율이 $(B/B_\mathrm{eq})^2$로 스케일되므로, $B = 3 B_\mathrm{eq}$만 되어도 장력이 항력을 한 차수 능가하여 튜브가 질서 있게 상승한다.

6. **Surface emergence is a different regime** — 표면 부상은 다른 영역이다. 깊이 $-20$ Mm 위로 TFT 근사가 깨지며 튜브는 "sea-serpent" 파동 구조로 분열된다. 흑점의 $2$–$3$ kG는 상승 중 $B/\rho$ 보존의 자연스러운 귀결이며, 내부 자기장 강도 추정과 모순되지 않는다.

7. **The review defines the pre-global-simulation benchmark** — 이 리뷰는 global 시뮬레이션 이전 시대의 기준을 정의한다. 이후 Ghizaru (2010), Käpylä (2012), Hotta (2016) 등의 global convective dynamo 연구는 모두 Fan 2009의 정량적 예측(위도 분포, 기울기 각, 필요 자기장 강도)과 대조된다.

8. **Open questions still drive current research** — 미해결 문제들이 여전히 연구를 주도한다. 강/약 자기장 논쟁, 꼬임의 기원, 거대 세포의 영향, CME 플럭스 로프 기원 — 모두 2026년 현재도 활발한 연구 주제이며, HMI와 DKIST 관측이 결정적 제약을 제공하고 있다.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Lateral Pressure Balance and Magnetic Buoyancy / 측면 압력 평형과 자기 부력

Starting from the condition that a tube element is in lateral pressure balance:
$$
p_e = p_i + \frac{B^2}{8\pi}.
$$
If temperatures equilibrate ($T_i = T_e$), the ideal-gas law gives
$$
\frac{\rho_e - \rho_i}{\rho_e} = \frac{1}{\beta} \equiv \frac{B^2}{8\pi p_e},
$$
producing a buoyancy force per unit volume $(\rho_e - \rho_i)g \approx \rho_e g / \beta$. **주변과 내부의 온도가 같을 때 어떤 자기 튜브라도 밀도가 낮아져 부력을 받는다**는 결정적 결론.

### 4.2 Thin Flux Tube Momentum Equation / 얇은 플럭스 튜브 운동 방정식

Per unit mass, in the inertial frame first and then with rotation added:
$$
\rho \frac{d\mathbf{v}}{dt} = -\nabla p_i + \rho\mathbf{g} + \underbrace{\partial_s\!\left(\frac{B^2}{8\pi}\right)\hat{l}}_{\text{magnetic pressure along }s} + \underbrace{\frac{B^2}{4\pi}\mathbf{k}}_{\text{magnetic tension}} - \underbrace{C_d \frac{\rho_e |v_\perp| v_\perp}{\pi a}}_{\text{aerodynamic drag}} - 2\rho\,\mathbf{\Omega}\times\mathbf{v}.
$$

Each term: gas-pressure gradient → establishes hydrostatic balance; gravity → gives buoyancy once pressure balance inserted; magnetic pressure along tube → accelerates material along tube from high-$B$ to low-$B$ regions; **magnetic tension** → restoring force at curved portions, dominant near Ω-apex; **drag** → couples tube to external flow when $v_\perp \neq 0$; **Coriolis** → bends northward-rising legs westward (leads to Joy's law).

### 4.3 Mass and Flux Conservation / 질량·플럭스 보존

Along the tube, Lagrangian conservation reads:
$$
\frac{d}{dt}(\rho A \,\delta s) = 0, \qquad \frac{d}{dt}(B A) = 0.
$$
Combining: $B/\rho \propto \delta s$ (along a tube element). **부상 중 튜브가 늘어나면 $B$는 증가, 밀도가 감소하면 $B$도 감소**한다.

### 4.4 Critical Twist / 임계 꼬임

From the Moreno-Insertis & Emonet (1996), Longcope et al. (1996) analysis of vortex shedding:
$$
q_\mathrm{cr}^2 \sim \frac{\rho_e v_\perp^2}{B^2/4\pi} \cdot \frac{1}{a^2} \quad\Rightarrow\quad q_\mathrm{cr} \sim \frac{1}{a}\frac{v_\perp}{v_A}.
$$
Physically: the azimuthal tension force $B_\phi^2/4\pi a$ must balance the dynamical pressure of the induced flow against the cross section.

### 4.5 Rise Time / 부상 시간

Terminal buoyant rise speed from balance of buoyancy and drag (or buoyancy and tension for coherent Ω-loop):
$$
v_\mathrm{rise} \sim \frac{v_A}{\sqrt{\beta}} \sim \sqrt{\frac{g H_p}{\beta}}.
$$
For $B_0 = 10^5$ G at the CZ base ($\rho \approx 0.2$ g/cm³), $v_A \approx 630$ cm/s and $\beta \approx 10^5$, giving $v_\mathrm{rise} \sim 2$ cm/s — but the tube expands as it rises, so the apex traverses $\sim 200$ Mm in roughly **a month**.

### 4.6 Joy's Law Tilt / Joy의 법칙 기울기

From linearized Coriolis deflection during rise (Fan 1993; D'Silva & Choudhuri 1993):
$$
\tan\gamma \approx \frac{\Omega \,\tau_\mathrm{rise}\, v_\perp}{v_\mathrm{rise}}\sin\lambda = f(B_0, \Phi)\,\sin\lambda.
$$
Matching the observed $\gamma \approx 0.5°\sin\lambda$ constrains the proportionality coefficient and therefore $B_0 \sim (3\text{–}10)\times 10^4$ G for $\Phi \sim 10^{22}$ Mx.

### 4.7 Worked Example: A Representative Rising Tube / 대표 부상 튜브 수치 예

Take $B_0 = 10^5$ G, $\Phi = 10^{22}$ Mx, CZ-base density $\rho_0 = 0.2$ g/cm³:
- Cross-sectional area $A_0 = \Phi/B_0 = 10^{17}$ cm² ⇒ radius $a_0 = \sqrt{A_0/\pi} \approx 5600$ km.
- Scale height $H_p \approx 5\times 10^4$ km ⇒ $a_0/H_p \approx 0.1$ (TFT just valid).
- Alfvén speed $v_A = B_0/\sqrt{4\pi\rho_0} \approx 630$ cm/s.
- Density deficit $\Delta\rho/\rho = 1/\beta = B_0^2/(8\pi p_0) \approx 10^{-5}$ ⇒ buoyant acceleration $g \cdot 10^{-5} \approx 3$ cm/s².
- Rise time across 200 Mm ≈ $\sqrt{2 L/g_\mathrm{eff}} \approx 4\times 10^6$ s ≈ **45 days**.
- Coriolis parameter at $\lambda = 20°$: $2\Omega\sin\lambda \approx 10^{-6}$ s⁻¹ ⇒ tilt $\sim \Omega\tau_\mathrm{rise}\sin\lambda \approx 0.35\sin\lambda$ rad ≈ $20° \sin\lambda$ — in the right ballpark after realistic drag.

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1908 ─── Hale: Zeeman effect in sunspots / 흑점 제만 효과
1919 ─── Hale polarity law / 헤일 극성 법칙
1955 ─── Parker: magnetic buoyancy / 자기 부력
1961 ─── Babcock: flux-transport dynamo cartoon
1966 ─── Weiss: buoyant flux tubes simulation
1975 ─── Parker: flux loss from the CZ / 대류층 플럭스 손실 문제
1979 ─── Schüssler: untwisted tubes fragment / 꼬임 없는 튜브 쪼개짐
1981 ─── Spruit: thin flux tube equations
1982 ─── van Ballegooijen: storage in overshoot layer
1989 ─── Wang & Sheeley: modern Joy's law statistics
1991 ─── Choudhuri & Gilman: Coriolis deflection computed
1993 ─── D'Silva & Choudhuri: 10⁵ G required ⇐ watershed
1994 ─── Fan et al.: TFT + solar stratification
1995 ─── Caligari et al.: full TFT equations, latitude predictions
1996 ─── Longcope et al.; Moreno-Insertis & Emonet: twist threshold
1998 ─── Fisher et al.: leading-follower asymmetry explained
2001 ─── Fan; Abbett et al.: first 3D MHD Ω-loop rise
2005 ─── Miesch: LRSP CZ dynamics review (prerequisite)
2007 ─── Cheung et al.: near-surface emergence MHD
2008 ─── Jouve & Brun: spherical-shell Ω-loop simulations
2009 ─── FAN: THIS REVIEW / 본 리뷰
2010 ─── Ghizaru et al.: global convective dynamo era begins
2012 ─── Stein & Nordlund: weak-field alternative scenario
2016 ─── Hotta et al.: high-resolution global dynamo reproduces buoyancy
2019 ─── Schunker et al.: helioseismic emergence signatures (weak)
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **Miesch 2005 (LRSP #2)** | Sets the stratified, differentially rotating CZ that TFT tubes live in / TFT 튜브가 부상하는 성층·차등 회전 배경을 제공 | Essential prerequisite — defines $\rho(r)$, $\Omega(r,\theta)$, tachocline / 필수 선수; 성층·회전·타코클라인 정의 |
| **Charbonneau 2010 (LRSP "Dynamo Models")** | Fan's buoyancy is the rise-stage of the solar dynamo / Fan의 부력이 다이나모의 상승 단계 | Tachocline-generated toroidal field of Charbonneau is Fan's initial condition / Charbonneau의 토로이달 장이 Fan의 초기조건 |
| **Chen 2011 / 2017 (LRSP "CMEs")** | Emerged flux ropes become the progenitor structures of CMEs / 부상 완료된 플럭스 로프가 CME 전구체 | Twist and helicity set during CZ rise are inherited by erupting ropes / 부상 중 결정된 꼬임·헬리시티가 분출 로프에 계승됨 |
| **Cheung & Isobe 2014 (LRSP)** | Continues Fan's story from photosphere upward / Fan 이야기를 광구 위로 이어감 | Near-surface and coronal emergence, uses Fan as deep-CZ input / 표면 이상 영역 부상, Fan을 깊은 대류층 입력으로 사용 |
| **McIntosh et al. 2014 ("extended cycle")** | Observed low-latitude horizontal-field patches hint at pre-emergence tubes / 관측된 저위도 수평장 패치는 부상 전 튜브를 암시 | Potential observational tracers of the TFT storage population / TFT 저장 집단의 잠재적 관측 추적자 |
| **Ghizaru et al. 2010; Hotta et al. 2016** | Global convective-dynamo simulations test Fan's predictions in situ / global 대류-다이나모 시뮬레이션이 Fan의 예측을 직접 테스트 | They include buoyancy self-consistently rather than inserting tubes by hand / 튜브를 수동 삽입이 아닌 자기일관적 부력으로 처리 |

---

## 7. References / 참고문헌

- Fan, Y. "Magnetic Fields in the Solar Convection Zone", Living Reviews in Solar Physics, 6, 4 (2009). [DOI: 10.12942/lrsp-2009-4]
- Parker, E. N. "The Formation of Sunspots from the Solar Toroidal Field", Astrophysical Journal, 121, 491 (1955).
- Spruit, H. C. "Motion of Magnetic Flux Tubes in the Solar Convection Zone and Chromosphere", Astronomy & Astrophysics, 98, 155 (1981).
- van Ballegooijen, A. A. "The Overshoot Layer at the Base of the Solar Convective Zone and the Problem of Magnetic Flux Storage", Astronomy & Astrophysics, 113, 99 (1982).
- Ferriz-Mas, A. & Schüssler, M. "Instabilities of Magnetic Flux Tubes in a Stellar Convection Zone: I. Equatorial Flux Rings in Differentially Rotating Stars", Geophys. Astrophys. Fluid Dynamics, 72, 209 (1993).
- D'Silva, S. & Choudhuri, A. R. "A Theoretical Model for Tilts of Bipolar Magnetic Regions", Astronomy & Astrophysics, 272, 621 (1993).
- Fan, Y., Fisher, G. H., & DeLuca, E. E. "The Origin of Morphological Asymmetries in Bipolar Active Regions", Astrophysical Journal, 405, 390 (1993).
- Caligari, P., Moreno-Insertis, F., & Schüssler, M. "Emerging Flux Tubes in the Solar Convection Zone. I. Asymmetry, Tilt, and Emergence Latitude", Astrophysical Journal, 441, 886 (1995).
- Longcope, D. W., Fisher, G. H., & Arendt, S. "The Evolution and Fragmentation of Rising Magnetic Flux Tubes", Astrophysical Journal, 464, 999 (1996).
- Emonet, T. & Moreno-Insertis, F. "The Physics of Twisted Magnetic Tubes Rising in a Stratified Medium", Astrophysical Journal, 492, 804 (1998).
- Fan, Y. "The Emergence of a Twisted Ω-tube into the Solar Atmosphere", Astrophysical Journal, 554, L111 (2001).
- Abbett, W. P., Fisher, G. H., & Fan, Y. "The Three-dimensional Evolution of Rising, Twisted Magnetic Flux Tubes in a Gravitationally Stratified Model Convection Zone", Astrophysical Journal, 546, 1194 (2001).
- Cheung, M. C. M., Schüssler, M., & Moreno-Insertis, F. "Magnetic Flux Emergence in Granular Convection: Radiative MHD Simulations and Observational Signatures", Astronomy & Astrophysics, 467, 703 (2007).
- Jouve, L. & Brun, A. S. "On the Role of Meridional Flows in Flux Transport Dynamo Models", Astronomy & Astrophysics, 474, 239 (2007).
- Miesch, M. S. "Large-Scale Dynamics of the Convection Zone and Tachocline", Living Reviews in Solar Physics, 2, 1 (2005). [DOI: 10.12942/lrsp-2005-1]
- Wang, Y.-M. & Sheeley, N. R. "Average Properties of Bipolar Magnetic Regions during Sunspot Cycle 21", Solar Physics, 124, 81 (1989).
