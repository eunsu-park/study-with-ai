---
title: "Coronal Mass Ejections: Models and Their Observational Basis"
authors: P. F. Chen
year: 2011
journal: "Living Reviews in Solar Physics, 8, 1"
doi: "10.12942/lrsp-2011-1"
topic: Living Reviews in Solar Physics / Coronal Mass Ejections
tags: [CME, flux-rope, magnetic-breakout, tether-cutting, catastrophe, kink-instability, torus-instability, sigmoid, coronal-dimming, space-weather]
status: completed
date_started: 2026-04-19
date_completed: 2026-04-19
---

# 22. Coronal Mass Ejections: Models and Their Observational Basis / 코로나 질량 방출: 모델과 관측적 근거

---

## 1. Core Contribution / 핵심 기여

**한국어:**
Chen(2011)은 **CME(Coronal Mass Ejection)**의 개시·가속·전파 메커니즘을 **관측 증거에 기반해** 체계적으로 정리한 결정판 리뷰다. 저자는 "어느 모델이 맞는가?"라는 교조적 물음을 피하고, (a) **에너지 저장**(자기 자유 에너지, 광구 twist/flux emergence/shear), (b) **트리거**(격변, breakout, tether-cutting, kink/torus 불안정성), (c) **전파**(drag, ICME 연결) 세 단계로 각 모델의 가정·예측·관측 증거를 비교한다. 관측 측면에서는 3-part CME(bright front + dark cavity + dense core), sigmoid, coronal dimming, EUV wave, post-eruption arcade 등을 모델별로 어떻게 재현하는지 명시한다. 이 리뷰는 SDO가 발사된 직후 시점에 쓰여 **고해상도 관측 시대의 CME 이론 해석틀**을 설정했으며, 이후 Parker Solar Probe의 CME 내부 통과 관측까지 연결되는 공통 언어가 되었다.

**English:**
Chen (2011) is the definitive, observation-grounded review of the physical initiation, acceleration, and propagation of **Coronal Mass Ejections (CMEs)**. Rather than championing a single model, the author organizes the theory into three stages: (a) **energy storage** (magnetic free energy via photospheric twist, flux emergence, shear), (b) **triggers** (catastrophe, breakout, tether-cutting, kink/torus instabilities), and (c) **propagation** (aerodynamic drag, ICME linkage). For each he compares assumptions, predictions, and observational evidence. On the observational side, iconic features — three-part CME (bright front + dark cavity + dense core), sigmoids, coronal dimming, EUV waves, post-eruption arcades — are mapped onto each model's predicted signature. Written just after SDO launched, the review established the **interpretive framework for the high-resolution CME era** and remains the common language for in-situ observations by Parker Solar Probe (from 2018).

---

## 2. Reading Notes / 읽기 노트

### §1. Introduction / 서론

**한국어:**
Chen은 CME의 **4대 특징**을 제시: (1) 대규모(~$10^{10}$ km³), (2) 고질량(~$10^{12}$–$10^{13}$ kg), (3) 고속(50–3000 km/s, 중앙값 ~450 km/s), (4) 자기장 운반자(행성간 자기 구름, interplanetary magnetic cloud). 그리고 **"왜 CME 이론이 어려운가"**의 핵심을 제시 — 관측 영상(코로나그래프)은 Thomson scattering으로 **전자 밀도의 시선 적분**만 보여주므로 실제 3D 자기 구조는 직접 관측되지 않는다. 따라서 모델 선호도는 직접 관측이 아닌 **2차 지표**(sigmoid, 필라멘트 궤적, dimming 위치 등)에 의존한다.

**English:**
Chen opens with CMEs' **four defining features**: (1) huge volume (~$10^{10}$ km³), (2) massive ($10^{12}$–$10^{13}$ kg), (3) fast (50–3000 km/s, median ~450 km/s), (4) carriers of magnetic field (interplanetary magnetic clouds). He identifies the **core difficulty**: coronagraphs image only Thomson-scattered line-of-sight electron density, not the 3D magnetic structure — so model preference relies on **indirect signatures** (sigmoids, filament trajectories, dimming locations).

### §2. Observational Overview / 관측 개요

**한국어:**
관측 특성을 정리:

1. **Speed distribution**: 고속(>800 km/s)과 저속(<400 km/s)의 이중 모드 분포. 고속은 flare 동반, 저속은 filament 분출 동반이 많음.
2. **3-part structure**: LASCO에서 ~1/3의 CME가 **bright front + dark cavity + dense core**의 고전적 3층 구조. 이것이 **플럭스 로프 + 가열된 sheath + 필라멘트 코어** 모델의 근거.
3. **Sigmoid**: 분출 전 sigmoid(S 또는 역S 모양 EUV/X-ray 구조)를 가진 active region의 ~80%가 48시간 내 분출.
4. **Coronal dimming**: CME 발생 후 EUV에서 밝기 감소 영역, 분출 footpoint를 직접 지시.
5. **EUV waves (대형파)**: 태양 표면을 가로지르는 waves, CME 팽창의 결과로 해석(fast-mode wave + field-line stretching의 조합).
6. **Post-eruption arcade**: 분출 후 reconnection으로 형성되는 환 구조, post-flare loops.
7. **Statistical properties**: 태양활동주기와 강한 상관, 최대기에 ~5/day, 최소기에 ~0.5/day.

Figure 4(LASCO 3-part CME)와 Figure 11(sigmoid eruption sequence)이 이 섹션의 중심.

**English:**
Observational properties:

1. **Bimodal speed distribution**: fast (>800 km/s, flare-associated) vs. slow (<400 km/s, filament-associated).
2. **Three-part structure**: ~1/3 of LASCO CMEs show the classical **bright front + dark cavity + dense core** — evidence for **flux rope + heated sheath + filament core**.
3. **Sigmoids**: ~80% of active regions hosting a pre-eruption sigmoid erupt within 48 h.
4. **Coronal dimming**: post-eruption EUV depression marks the eruption footpoints.
5. **EUV waves**: large-scale waves across the solar surface — mix of fast-mode wave and field-line stretching driven by CME expansion.
6. **Post-eruption arcade**: loop arcade formed by reconnection after the eruption.
7. **Statistics**: strong cycle dependence, ~5/day at maximum, ~0.5/day at minimum.

### §3. Energy Storage and Pre-eruption State / 에너지 저장과 분출 전 상태

**한국어:**
CME의 **에너지원은 자기 자유 에너지**(free magnetic energy). Grad-Shafranov 방정식의 force-free 해는 코로나 플라스마가 $\beta\ll 1$이라서 성립:

$$
\nabla\times\mathbf{B} = \alpha(\mathbf{r})\,\mathbf{B}, \quad \mathbf{B}\cdot\nabla\alpha = 0
$$

자유 에너지 $E_{\rm free} = E_{\rm total} - E_{\rm potential}$은 **비포텐셜 자기장**(twist, shear)에 저장. 축적 메커니즘:

1. **Flux emergence**: 대류층에서 광구로 떠오른 자기 튜브가 이미 twist 지님
2. **Photospheric motion**: 차등 회전과 광구 흐름이 field line을 shear·twist
3. **Flux cancellation**: 반대 극성 경계에서 flux 소멸 → helicity 축적

**Flux rope**가 코로나에 존재할 수 있는지 여부가 두 이론 학파를 가름:
- **"Pre-existing flux rope" models** (breakout, torus): 분출 전 이미 형성
- **"Sheared arcade" models** (tether-cutting, 일부 catastrophe): reconnection으로 **분출 중에** 형성

관측적으로 sigmoid와 filament channel은 pre-existing flux rope를 강하게 시사하지만, 모든 CME에서 그런지는 논쟁 중.

**English:**
CME energy = **magnetic free energy** $E_{\rm free} = E_{\rm total} - E_{\rm potential}$, stored in non-potential fields (twist, shear). Storage mechanisms: flux emergence, photospheric shearing, flux cancellation. The key theoretical split is whether a **flux rope exists before eruption** (breakout, torus schools) or forms **during eruption** via reconnection (tether-cutting and some catastrophe scenarios). Sigmoids and filament channels observationally favor pre-existing ropes, but the universality is contested.

### §4. Trigger Mechanisms / 트리거 메커니즘

**한국어 / English (섹션 전체):** 이 섹션이 리뷰의 심장부. 4대 모델을 비교:

#### §4.1 Catastrophe model (Forbes & Priest 1991)

**한국어:** 이차원 축대칭 평형에서, 광구의 **점진적 수렴 운동**이 배경 field을 압축. 시스템이 평형 곡선의 "접는 점(fold point)"을 넘으면 **격변적 상실**(loss of equilibrium) → flux rope 상승 → 아래쪽 current sheet 형성 → reconnection. 핵심 지표: 수렴 운동과 임계점 도달까지의 **느린 선행 단계**.

**English:** In a 2D axisymmetric equilibrium, slow photospheric **convergence** compresses the background field. When the system crosses a "fold point" of the equilibrium curve, it undergoes a **loss of equilibrium** — the rope rises, a current sheet forms below, and reconnection starts. The hallmark is a **slow precursor phase** of convergence.

#### §4.2 Magnetic breakout model (Antiochos et al. 1999)

**한국어:** **다중극성 (multipolar)** 자기 구조가 필수. 중앙 arcade 위에 오버라잉 arcade가 있고, 둘 사이 null point. 중앙 arcade가 shearing되어 부풀어 오르면 null에서 **외부 reconnection**이 일어나 오버라잉 field가 제거됨 → "tether"가 풀려 내부 flux rope 급가속 → 결국 **내부 reconnection**(flare reconnection)까지 이어짐. 핵심: **두 단계 reconnection** (먼저 외부, 그 다음 내부).

**English:** Requires a **multipolar** topology: a central arcade, an overlying arcade, and a null point between. Shearing of the central arcade inflates it against the overlying field. At the null, **external reconnection** removes the overlying field (the "breakout"), freeing the inner flux rope to accelerate. Later **internal reconnection** produces the flare. Key: **two-stage reconnection** (external first, then internal).

#### §4.3 Tether-cutting model (Moore et al. 2001)

**한국어:** Sigmoid의 **J-shaped footpoint 영역**에서 **내부 reconnection**이 먼저 발생. 반대 극성의 J가 만나 짧은 고리(flare loop)와 긴 고리(flux rope)를 만듦. flux rope가 상승하면 **2차 reconnection**(flare)이 활성화. 핵심: **내부 reconnection이 먼저** (breakout과 반대).

**English:** **Internal reconnection** starts first at the **J-shaped footpoints** of a sigmoid. Opposite-polarity J's reconnect to form a short flare loop and a long flux rope. As the rope rises, a **second reconnection** powers the flare. Opposite ordering to breakout: internal first.

#### §4.4 Kink and torus instabilities

**한국어:**
- **Kink instability**: 직선 flux tube의 twist 각 $\Phi$가 임계값 $\Phi_{\rm crit}\approx 2.5\pi$를 넘으면 나선 형태로 불안정. 실제 코로나의 휘어진 flux rope에서는 $\Phi_{\rm crit}$가 달라짐.
- **Torus instability**: 토러스 flux rope가 외부 감쇠장의 **급감** 하에서 팽창 방향 양성 피드백. 임계 조건:
$$n \equiv -\frac{d\ln B_{\rm ext}}{d\ln h} > n_{\rm crit} \approx 1.5$$
여기서 $h$ = 로프 높이. $n$은 **decay index**라 불리며, CME 개시의 정량적 예측 지표.

두 불안정성 모두 **이상 MHD**(ideal MHD) 내에서 작동 — reconnection 없이도 분출 시작 가능(그러나 flare 생성을 위해 reconnection은 동반).

**English:**
- **Kink instability**: a straight flux tube becomes helical when twist $\Phi$ exceeds $\Phi_{\rm crit}\approx 2.5\pi$. In curved coronal ropes, the threshold is different.
- **Torus instability**: a toroidal rope expands unstably if the external field decays faster than
$$n \equiv -\frac{d\ln B_{\rm ext}}{d\ln h} > n_{\rm crit} \approx 1.5.$$
$n$ is the **decay index**, a quantitative CME-initiation predictor. Both are **ideal MHD** instabilities — no reconnection needed to begin eruption (though reconnection accompanies the flare).

#### §4.5 Comparison and relation to flares

**한국어:** 모든 4모델이 공통적으로 **flare-CME 연결**을 설명할 수 있다. flare는 **reconnection의 에너지 방출**(열, 비열), CME는 **플라스마와 자기장의 질량 방출**. 하지만 flare-only 사건(confined flare)과 CME-only 사건(filament 분출 without flare)도 존재 → 인과 관계는 단순하지 않음.

**한국어:** All four can in principle explain **flare-CME association**: flares dissipate reconnection energy as heat and non-thermal particles, while CMEs carry mass and field into space. However, confined flares (flare-only) and eruptive filaments (CME without flare) also exist — so the causal relation is not one-directional.

### §5. CME Propagation / CME 전파

**한국어:**
코로나를 벗어난 CME는 **배경 태양풍과의 항력(drag)**으로 속도가 조절됨. 간단한 운동 방정식:

$$
\frac{du_{\rm CME}}{dt} = -\gamma\,(u_{\rm CME} - u_{\rm SW})\,|u_{\rm CME} - u_{\rm SW}|
$$

$\gamma \sim C_d A\rho_{\rm SW} / m_{\rm CME}$는 공기역학적 항력 계수. **고속 CME는 감속**, **저속 CME는 가속**되어 1 AU에서 속도가 ~400–600 km/s로 수렴. ENLIL과 EUHFORIA 같은 **우주 기상 운영 모델**이 이 drag 공식을 CME-flux rope 확장 모델과 결합해 지구 도달 시간을 예측.

**ICME**(Interplanetary CME)는 CME가 in-situ에서 관측될 때의 명칭. **자기 구름(magnetic cloud)** 특징:
- 강한 자기장 ($> 10$ nT at 1 AU)
- 부드러운 회전(smooth field rotation) — 플럭스 로프 서명
- 낮은 양성자 온도 ($T_p / T_{\rm expected} < 0.5$)

**English:**
Beyond the corona, CMEs evolve under **aerodynamic drag** from the ambient solar wind. Fast CMEs decelerate, slow ones accelerate — speeds converge to ~400–600 km/s at 1 AU. Operational models (ENLIL, EUHFORIA) combine this drag law with flux-rope expansion to predict Earth arrival. **ICMEs** (Interplanetary CMEs) show **magnetic cloud** signatures: strong field ($>10$ nT at 1 AU), smooth rotation (flux rope), low proton temperature ($T_p/T_{\rm expected} < 0.5$).

### §6. Conclusion and Open Problems / 결론과 미해결 문제

**한국어:**
Chen이 정리하는 5대 미해결 문제:
1. **Flux rope의 존재 시점**: 분출 전인가 중인가?
2. **Trigger의 보편성**: 어느 모델이 어떤 CME 사건에 적용되는가?
3. **3D 비축대칭 효과**: 대부분의 이론이 2D/axisymmetric. 3D 시뮬레이션이 얼마나 다른가?
4. **Particle acceleration**: CME-driven shock이 어떻게 고에너지 입자(SEP)를 가속하는가?
5. **Predictive forecasting**: 어느 active region이 얼마나 빨리 CME를 만들지 지금도 정량 예측 어려움.

**English:**
Five open problems:
1. **When does the flux rope form** — before or during eruption?
2. **Which trigger applies to which event?**
3. **3D non-axisymmetric effects**: most theory is 2D/axisymmetric — how different are 3D simulations?
4. **Particle acceleration** at CME-driven shocks producing Solar Energetic Particles (SEPs).
5. **Predictive forecasting**: we still cannot quantitatively predict which active region will erupt and when.

---

## 3. Key Takeaways / 핵심 시사점

1. **CME의 에너지원은 오직 자기 자유 에너지 / CME energy source is exclusively magnetic free energy** — 열압력·중력·회전 에너지는 CME의 운동에너지($10^{31}$–$10^{32}$ erg)를 전혀 공급할 수 없다. 유일한 저장소는 **비포텐셜 자기장(twist, shear)**. / Thermal, gravitational, and rotational energies are insufficient; only **non-potential magnetic fields** can supply the $10^{31}$–$10^{32}$ erg kinetic energy.

2. **4대 트리거 모델은 상호 배타가 아니다 / The four trigger models are not mutually exclusive** — catastrophe, breakout, tether-cutting, kink/torus는 **서로 다른 CME 사건**에 적용될 수 있다. 실제 태양은 "하나의 메커니즘"이 아니라 **여러 경로의 앙상블**을 보여준다. / Different mechanisms may apply to different events; the Sun shows an ensemble of pathways.

3. **Decay index $n>1.5$가 현대 CME 예측의 정량 기준 / Decay index $n>1.5$ is the modern quantitative predictor** — torus instability 임계 조건이 머신러닝 CME 예측 모델의 핵심 feature. PIL(polarity inversion line) 근처에서 계산 가능. / The torus-instability criterion is now a standard ML feature, computable near the polarity inversion line.

4. **Sigmoid는 분출 예고의 가장 강력한 단일 지표 / Sigmoids are the single strongest pre-eruption indicator** — sigmoid 보유 active region의 ~80%가 48시간 내 분출. 비포텐셜 twist의 시각적 증거. / ~80% of sigmoid-hosting active regions erupt within 48 h.

5. **Breakout과 tether-cutting은 "reconnection 순서"로 구별된다 / Breakout vs. tether-cutting differ in reconnection ordering** — breakout은 **외부 → 내부**, tether-cutting은 **내부 → 외부**. 관측된 첫 brightening의 위치(중앙 또는 주변)로 구별 가능. / Breakout: external → internal; tether-cutting: internal → external. Distinguishable by where the first brightening appears.

6. **CME drag는 속도를 수렴시킨다 / CME drag converges speeds** — 고속 CME 감속, 저속 CME 가속. 1 AU에서 ~400–600 km/s로 수렴. 운영 예보 모델의 핵심 가정. / Fast CMEs decelerate, slow ones accelerate, converging to 400–600 km/s at 1 AU — the operational forecasting assumption.

7. **Magnetic cloud는 flux rope의 in-situ 사인 / Magnetic clouds are the in-situ flux-rope signature** — 강한 B, 부드러운 회전, 낮은 $T_p$. 1970년대 Burlaga가 분류 이후 CME의 flux rope 가설 결정적 증거. / Strong B, smooth rotation, low $T_p$ — established by Burlaga (1970s) as definitive evidence for the flux-rope picture.

8. **3D 비축대칭이 미래 연구의 전선 / 3D non-axisymmetric effects are the frontier** — 2010년 이후 Jiang, Torok 등이 3D MHD로 CME를 재현. 관측 AR의 비대칭성을 반영하는 MHD 시뮬레이션이 데이터 주도 예보의 핵심. / Jiang, Török and others have produced 3D MHD CMEs since 2010; data-driven 3D MHD is central to modern forecasting.

---

## 4. Mathematical Summary / 수학적 요약

### A. Force-free coronal field / 코로나 force-free 장

$$
\nabla\times\mathbf{B} = \alpha(\mathbf{r})\,\mathbf{B}, \quad \mathbf{B}\cdot\nabla\alpha = 0
$$

### B. Magnetic free energy / 자기 자유 에너지

$$
E_{\rm free} = \frac{1}{2\mu_0}\int |\mathbf{B}|^2\,dV - \frac{1}{2\mu_0}\int |\mathbf{B}_{\rm pot}|^2\,dV
$$

### C. Magnetic helicity / 자기 helicity

$$
H_m = \int_V \mathbf{A}\cdot\mathbf{B}\,dV
$$

ideal MHD 불변량. CME가 코로나 helicity를 우주로 내보내는 주 통로.

### D. Torus instability decay index / 토러스 불안정 decay index

$$
n(h) = -\frac{d\ln B_{\rm ext}(h)}{d\ln h}, \quad \text{unstable if } n > n_{\rm crit}\approx 1.5
$$

$B_{\rm ext}$ = flux rope 위치에서 외부 감쇠 자기장, $h$ = 로프 높이. PIL 위에서 potential field extrapolation으로 계산.

### E. Kink instability criterion / 킹크 불안정 조건

$$
\Phi = \oint \frac{B_\phi}{RB_z}\,d\ell > \Phi_{\rm crit}
$$

직선 튜브: $\Phi_{\rm crit} = 2\pi$ (Kruskal-Shafranov). 아치형: $\Phi_{\rm crit}\approx 2.5\pi$ (Török & Kliem 2005).

### F. CME drag equation / CME 항력 방정식

$$
\frac{du_{\rm CME}}{dt} = -\gamma\,(u_{\rm CME} - u_{\rm SW})\,|u_{\rm CME} - u_{\rm SW}|
$$

$$
\gamma = \frac{C_d\,A_{\rm CME}\,\rho_{\rm SW}}{m_{\rm CME} + m_{\rm virtual}}
$$

$C_d \sim 1$ (drag coefficient), $A$ = CME 단면적, $m_{\rm virtual}$ = 가상 질량(휩쓸린 태양풍).

### G. Three-phase CME kinematics / CME 3단계 운동학

$$
h(t) = \begin{cases}
h_0 + v_0(t-t_0) & \text{(slow rise, pre-eruption)} \\
h_1 + v_0(t-t_1) + \frac{1}{2}a_{\rm imp}(t-t_1)^2 & \text{(impulsive, } t_1<t<t_2\text{)} \\
h_2 + v_{\rm cruise}(t-t_2) - \frac{1}{2}|a_{\rm drag}|(t-t_2)^2 & \text{(propagation, } t>t_2\text{)}
\end{cases}
$$

각 단계가 다른 물리(느린 재평형화, 불안정성 활성화, 행성간 drag).

### H. Flux rope magnetic flux and twist / 플럭스 로프의 자기 flux와 twist

축(axial) flux: $\Phi_{\rm axial} = \int B_z\,dA$
twist / unit length: $T = B_\phi / (R\,B_z)$
총 twist: $\Phi_{\rm total} = \int T\,dz$

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1971  OSO-7 — 첫 CME 관측 / first CME observation
  │
1973  Skylab ATM — 체계적 CME 관측 시작 / systematic CME observations
  │
1978  Burlaga — magnetic cloud 정의 / defined magnetic clouds
  │
1989  Hundhausen — CME 질량·에너지 통계 / CME mass/energy statistics
  │
1991  Forbes & Priest — catastrophe 모델 / catastrophe model
  │
1995  SOHO/LASCO — 매일 CME 관측 / daily LASCO CME monitoring
  │
1999  Antiochos, DeVore, Klimchuk — magnetic breakout 모델 / breakout model
  │
2001  Moore et al. — tether-cutting 모델 / tether-cutting model
  │
2005  Török & Kliem — kink & torus 3D 시뮬 / kink & torus 3D simulations
  │
2006  STEREO launch — 3D CME 재구성 / 3D reconstruction
  │
2010  SDO launch — AIA/HMI 고해상도 / high-res initiation obs
  │
▶ 2011  ★ Chen — "CMEs: Models and Observational Basis" (이 논문 / THIS PAPER)
  │
2014  Kliem et al. — data-driven MHD CME 시뮬 / data-driven MHD CME sim
  │
2018  Parker Solar Probe — in-situ CME 내부 / in-situ through CMEs
2020  Solar Orbiter — 고위도 + 원격 / high-lat + remote combo
2022  ML CME prediction (Bobra, SHARP features) — ML 예측 / ML forecasting
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **LRSP #9 — Priest, *The Magnetohydrostatic Sun* (2014)** | 코로나 자기장 평형과 불안정성 이론의 기초 / foundation for coronal equilibria | 필수 선행 / Prerequisite |
| **LRSP #20 — Charbonneau, *Dynamo Models of the Solar Cycle* (2010)** | helicity 주입의 근원(다이나모)을 제공 / dynamo supplies helicity reservoir for CMEs | 상류 연결 / Upstream link |
| **LRSP #21 — Ofman, *Wave Modeling of the Solar Wind* (2010)** | ICME이 전파하는 배경 태양풍 / ambient solar wind into which ICMEs propagate | 전파 단계 / Propagation stage |
| **Forbes & Priest (1991)** | catastrophe 모델 원전 / original catastrophe model | §4.1 대상 / subject of §4.1 |
| **Antiochos, DeVore, Klimchuk (1999)** | magnetic breakout 모델 원전 / original breakout model | §4.2 대상 / subject of §4.2 |
| **Moore et al. (2001)** | tether-cutting 모델 원전 / original tether-cutting model | §4.3 대상 / subject of §4.3 |
| **Török & Kliem (2005) ApJ** | kink & torus 불안정 3D 시뮬 / kink/torus 3D sims | §4.4 대상 / subject of §4.4 |
| **Burlaga (1978)** | magnetic cloud 최초 분류 / original magnetic cloud classification | §5 기반 / basis of §5 |
| **Kliem & Török (2006) PRL** | torus instability 정량화 / torus criterion $n>1.5$ | §4.4 핵심 / core of §4.4 |

---

## 7. References / 참고문헌

- Chen, P. F., "Coronal Mass Ejections: Models and Their Observational Basis", *Living Reviews in Solar Physics*, 8, 1, 2011. [DOI: 10.12942/lrsp-2011-1]
- Antiochos, S. K., DeVore, C. R., Klimchuk, J. A., "A Model for Solar Coronal Mass Ejections", *ApJ*, 510, 485, 1999.
- Burlaga, L. F., "Magnetic clouds and force-free fields with constant alpha", *JGR*, 93, 7217, 1988.
- Forbes, T. G., Priest, E. R., "Photospheric magnetic field evolution and eruptive flares", *ApJ*, 446, 377, 1995.
- Hundhausen, A. J., "Sizes and locations of coronal mass ejections: SMM observations from 1980 and 1984–1989", *JGR*, 98, 13177, 1993.
- Kliem, B., Török, T., "Torus Instability", *PRL*, 96, 255002, 2006.
- Low, B. C., "Coronal mass ejections, magnetic flux ropes, and solar magnetism", *JGR*, 106, 25141, 2001.
- Moore, R. L., Sterling, A. C., Hudson, H. S., Lemen, J. R., "Onset of the Magnetic Explosion in Solar Flares and Coronal Mass Ejections", *ApJ*, 552, 833, 2001.
- Török, T., Kliem, B., "Confined and Ejective Eruptions of Kink-unstable Flux Ropes", *ApJ*, 630, L97, 2005.
- Yashiro, S., Gopalswamy, N., Michalek, G., et al., "A catalog of white light CMEs observed by SOHO/LASCO", *JGR*, 109, A07105, 2004.
- Zhang, J., Dere, K. P., Howard, R. A., Kundu, M. R., White, S. M., "On the temporal relationship between coronal mass ejections and flares", *ApJ*, 559, 452, 2001.
