---
title: "Wave Modeling of the Solar Wind"
authors: Leon Ofman
year: 2010
journal: "Living Reviews in Solar Physics, 7, 4"
doi: "10.12942/lrsp-2010-4"
topic: Living Reviews in Solar Physics / Solar Wind
tags: [solar-wind, MHD-waves, Alfven-waves, ion-cyclotron, turbulence, multifluid-MHD, WKB, corona, coronal-hole, heliosphere]
status: completed
date_started: 2026-04-19
date_completed: 2026-04-19
---

# 21. Wave Modeling of the Solar Wind / 태양풍의 파동 모델링

---

## 1. Core Contribution / 핵심 기여

**한국어:**
이 리뷰는 **태양풍이 어떻게 그렇게 빠르고 뜨거운가**라는 반세기의 난제에 대해, Parker(1958)의 고전 열압력 구동 모델의 실패를 출발점으로 삼아 **MHD 파동이 공급하는 운동량과 에너지**로 문제를 해결하려는 모든 현대 모델을 한 자리에 정리한다. Ofman은 (a) **WKB 1D 파동-유체 모델**(Hollweg 계열), (b) **2.5D 다유체 MHD 시뮬레이션**, (c) **이온 사이클로트론 공명 가열**(SOHO/UVCS의 O⁵⁺ 비등방 관측이 요구), (d) **Kolmogorov / Iroshnikov-Kraichnan 난류 캐스케이드**, (e) **PIC/하이브리드 kinetic 접근**을 체계적으로 비교한다. 각 모델을 Helios·Ulysses·SOHO·WIND 관측과 정량적으로 대조하고, 어떤 관측 제약을 만족하고 어떤 것을 놓치는지 명확히 기록한다.

**English:**
Starting from the failure of Parker's (1958) classical thermal-pressure wind at explaining **why the solar wind is so fast and so hot**, this review assembles all modern models that fill the gap with **momentum and energy delivered by MHD waves**. Ofman systematically compares (a) **WKB 1D wave-fluid models** (the Hollweg lineage), (b) **2.5D multifluid MHD simulations**, (c) **ion-cyclotron resonant heating** (forced by SOHO/UVCS's O⁵⁺ anisotropy), (d) **Kolmogorov / Iroshnikov-Kraichnan turbulent cascades**, and (e) **PIC/hybrid kinetic approaches**. Each class is quantitatively compared against Helios, Ulysses, SOHO, and WIND observations, with an honest ledger of what is explained and what remains open.

---

## 2. Reading Notes / 읽기 노트

### §1. Introduction / 서론

**한국어:**
Ofman은 두 가지 문제를 대비시킨다. (1) **코로나 가열 문제**(왜 코로나는 광구의 200배 더 뜨거운가?)와 (2) **태양풍 가속 문제**(왜 빠른 바람이 그렇게 빠른가?). 두 문제는 **같은 에너지 저장소(자기장 + 파동)**를 공유하므로 분리할 수 없다. Parker 1958의 순수 열압력 구동 모델은 $T\sim 10^6$ K에서 음속점 통과 후 ~400 km/s까지만 주는 반면, 극관 coronal hole의 바람은 ~750 km/s에 달한다. 따라서 **추가 에너지원이 필요**하며, 그 후보가 이 리뷰의 주제 — MHD 파동(특히 Alfvén파)와 난류다.

**English:**
Ofman juxtaposes two problems: (1) the **coronal heating problem** (why is the corona 200× hotter than the photosphere?) and (2) the **solar wind acceleration problem** (why is the fast wind so fast?). They share the **same reservoir** (magnetic field + waves) and cannot be decoupled. Parker's (1958) pure thermal model at $T\sim 10^6$ K reaches only ~400 km/s, while polar coronal holes accelerate wind to ~750 km/s. **An extra energy source is required** — the subject of this review: MHD waves (especially Alfvénic) and their turbulent cascade.

### §2. Observational Constraints / 관측적 제약

**한국어:**
태양풍 파동 모델이 만족해야 할 관측 사실이 정리된다:

1. **고속/저속 바람의 이중성 (Ulysses)**: 극관 open field에서 $V \sim 700$–800 km/s, streamer belt에서 $V \sim 300$–500 km/s. Fast 바람은 **밀도가 낮고 He 비율이 높으며 변동성이 작다**.
2. **Helios in-situ 관측**: $r \gtrsim 0.3$ AU에서 양성자 온도 $T_p \sim 10^5$ K, 알파 입자 온도 $T_\alpha \gtrsim 4T_p$. 이온은 반경에 따라 **비단열** 감소 — 지속적 가열 증거.
3. **Alfvén 파 스펙트럼 (Belcher & Davis 1971)**: $\delta\mathbf{B}\perp\mathbf{B}_0$, Elsässer $|z^+| \gg |z^-|$ (외향 지배), 스펙트럼 $\propto f^{-5/3}$가 이온 스케일 위까지.
4. **SOHO/UVCS coronal hole 관측**: O⁵⁺ 수직 온도 $T_\perp \gtrsim 10^8$ K, 평행 온도 $T_\parallel \lesssim 10^6$ K — **선택적, 비등방 가열**. 양성자 질량의 16배인 이온이 수직 방향으로만 매우 뜨겁다.
5. **Temperature anisotropy $T_\perp/T_\parallel > 1$** (양성자, 이온): Helios 관측에서 1 AU까지 지속.

Figure 2(Ulysses 속도-위도 dial plot)와 Figure 8(UVCS 이온 온도)이 이 섹션의 "상징 그림"이다.

**English:**
The observations any wave model must match:

1. **Bimodal wind (Ulysses)**: ~700–800 km/s from polar open fields, ~300–500 km/s from the streamer belt. Fast wind is **less dense, more He-rich, less variable**.
2. **Helios in-situ**: at $r \gtrsim 0.3$ AU, $T_p \sim 10^5$ K, $T_\alpha \gtrsim 4T_p$. Ions cool **non-adiabatically** — continuous heating is required.
3. **Alfvénic spectrum (Belcher & Davis 1971)**: $\delta\mathbf{B}\perp\mathbf{B}_0$, outward-dominated ($|z^+|\gg|z^-|$), with $f^{-5/3}$ inertial-range scaling.
4. **SOHO/UVCS coronal-hole observations**: O⁵⁺ perpendicular temperature $\gtrsim 10^8$ K, parallel $\lesssim 10^6$ K — **selective, anisotropic heating**. A 16× proton-mass ion is enormously hot only perpendicular to B.
5. **Proton/ion anisotropy $T_\perp/T_\parallel > 1$** persists to 1 AU.

### §3. Parker's Theory and its Limits / Parker 이론과 그 한계

**한국어:**
고전 Parker 이중성 유체 방정식(등온):

$$
\frac{1}{u}\frac{du}{dr}\left(u^2 - c_s^2\right) = \frac{2c_s^2}{r} - \frac{GM_\odot}{r^2}
$$

음속 임계점 $r_c = GM_\odot/(2c_s^2)$에서 물리적 바람 해가 등장. $T=10^6$ K에서 $r_c \approx 6 R_\odot$, 바람 속도는 1 AU에서 ~400 km/s로 포화. Ofman은 이 모델의 **3대 한계**를 명시:
- 빠른 바람(750 km/s) 미달
- coronal hole 이온 비등방 가열 설명 불가
- 비단열 이온 온도 프로파일 설명 불가

"추가 구동력"의 가장 강력한 증거는 파동 스펙트럼이 바람의 주요 에너지원 스케일에 이미 존재한다는 점이다.

**English:**
The classical isothermal Parker equation has a sonic critical point at $r_c = GM_\odot/(2c_s^2)$. At $T=10^6$ K, $r_c\approx 6R_\odot$ and 1-AU wind saturates near 400 km/s. Ofman lists three **failures**: fast wind speed, anisotropic ion heating, and non-adiabatic ion cooling. The strongest evidence for extra driving is that the wave spectrum is already observed at the relevant scales.

### §4. WKB Wave-Driven 1D Models / WKB 파동 구동 1D 모델

**한국어:**
**운동량 방정식**에 파동 압력 항을 추가:

$$
\rho u\frac{du}{dr} = -\frac{dp}{dr} - \frac{\rho GM_\odot}{r^2} + F_{\rm wave}, \quad F_{\rm wave} = -\frac{1}{2}\frac{d}{dr}\!\left(\frac{\langle\delta B^2\rangle}{\mu_0}\right)
$$

파동이 광구에서 주입되고 **WKB 근사**(amplitude가 배경보다 느리게 변함)로 전파하면 파동 작용이 보존됨:

$$
\frac{d}{dr}\!\left[\langle\delta B^2\rangle\frac{(u+v_A)^2}{v_A}A(r)\right] = 0
$$

여기서 $A(r)$은 자기 흐름관 단면적. 파동 진폭은 밀도 감소로 **커지며**, ponderomotive force로 바람을 가속한다. Hollweg, Leer, Holzer, Isenberg 등이 1970s–90s에 개발. **결과**: WKB 모델은 fast 바람의 속도에 도달할 수 있지만, 이온 비등방 가열은 여전히 설명 못함 — **추가 감쇠/소산 메커니즘 필요**.

**English:**
Adding **wave pressure** to the momentum equation, with wave action conserved under **WKB** propagation, gives an amplitude that grows outward (density drop) and a ponderomotive force that accelerates the flow. Hollweg, Leer, Holzer, and Isenberg developed this framework in the 1970s–90s. **Result**: WKB models can reach fast-wind speeds but **cannot** explain anisotropic ion heating — a **dissipation mechanism** beyond WKB is needed.

### §5. Multifluid MHD Simulations / 다유체 MHD 시뮬레이션

**한국어:**
§4의 1D 한계를 넘어 Ofman 본인의 팀을 포함한 여러 그룹이 **2.5D 다유체 MHD**(proton + electron + α 등 각 종이 독립 유체)를 코로나 hole에 적용. 주요 결과:

- **양성자와 알파의 속도 차이**($V_\alpha > V_p$ at 1 AU): 파동 에너지가 α에 우선적으로 전달되는 이온-사이클로트론 효과로 설명.
- **비등방 온도 발생**: 파동 구동 비등방성이 수치적으로 재현.
- **자기장 기하구조의 중요성**: super-radial 팽창 geometry가 바람 속도에 크게 영향.

Figure 14 (Ofman 2004 시뮬레이션)가 UVCS의 O⁵⁺ 관측과 정성적으로 일치하는 비등방 온도 지도를 보여준다.

**English:**
Beyond §4's 1D limits, multiple groups (including Ofman's) apply **2.5D multifluid MHD** (proton + electron + α as separate fluids) to coronal holes. Key results: alpha-proton speed difference ($V_\alpha > V_p$ at 1 AU), anisotropic temperature generation, and strong sensitivity to super-radial flux-tube geometry. Figure 14 (from Ofman 2004) reproduces the UVCS anisotropy qualitatively.

### §6. Ion-Cyclotron Resonant Heating / 이온-사이클로트론 공명 가열

**한국어:**
SOHO/UVCS의 비등방 가열은 **파동-입자 공명**을 가장 강력하게 시사한다. 공명 조건:

$$
\omega - k_\parallel v_\parallel = n\Omega_i, \quad \Omega_i = \frac{q_iB}{m_ic}, \quad n=\pm 1, \pm 2, ...
$$

left-hand polarized Alfvén/이온-사이클로트론 파동이 이온의 자이로 운동과 공명하여 **수직 방향으로 에너지를 이전**. 따라서 $T_\perp \uparrow, T_\parallel$는 거의 변화 없음 — 관측된 비등방성과 일치.

**문제점**: 저주파 Alfvén 파가 이온 사이클로트론 주파수까지 **어떻게 캐스케이드**되는가? 자연 주입 주파수는 $\sim 10^{-3}$ Hz, 양성자 자이로 주파수는 코로나에서 $\sim 10^3$ Hz — 6 decade 차이. **난류 캐스케이드**가 답이어야 한다. 하지만 준평행 난류는 이온 스케일에서 **느리게 캐스케이드**(critical balance), 따라서 에너지가 제시간에 충분히 도달할지가 핵심 미해결 문제.

**English:**
UVCS anisotropy strongly implies **wave-particle resonance**. Left-hand-polarized Alfvén/ion-cyclotron waves deliver energy perpendicular to B, giving $T_\perp \uparrow$ while $T_\parallel$ barely changes — matching observations. **Problem**: injected Alfvén frequencies (~$10^{-3}$ Hz) are ~6 decades below the proton gyrofrequency (~$10^3$ Hz in the corona). A **turbulent cascade** must bridge the gap, but parallel cascades are slow (critical balance) — whether enough energy reaches ion scales in time is a central open question.

### §7. Turbulent Cascade / 난류 캐스케이드

**한국어:**
Alfvén 파동은 유한 진폭에서 비선형 상호작용으로 **에너지를 작은 스케일로 이송**. Elsässer 변수 $\mathbf{z}^\pm = \mathbf{u}\pm\mathbf{b}/\sqrt{\mu_0\rho}$가 핵심 언어:

$$
\partial_t z^\pm + (\mathbf{u}\mp\mathbf{v}_A)\cdot\nabla z^\pm = -\rho^{-1}\nabla p_T + \nu\nabla^2 z^\pm
$$

**외향 지배**($|z^+|\gg|z^-|$)는 반사(reflection) 난류 — 바람이 감속하는 영역 또는 flux tube 꺾임에서 $z^+$가 $z^-$로 일부 반사되어야 캐스케이드가 유지됨. Matthaeus & Zhou, Cranmer & van Ballegooijen(2005)의 **reflection-driven turbulence** 모델이 주류.

**스펙트럼 논쟁:**
- **Kolmogorov** $E(k)\propto k^{-5/3}$ — 강한 난류 한계
- **Iroshnikov-Kraichnan** $E(k)\propto k^{-3/2}$ — Alfvén 상호작용 약한 한계
- 관측은 대체로 −5/3에 가깝지만, 영역/스케일 의존성 있음.

Figure 20에서 SOHO/UVCS와 함께 풍 가열 프로파일이 reflection-turbulence 모델로 잘 fit됨을 보여준다.

**English:**
Finite-amplitude Alfvén waves nonlinearly cascade to small scales. Elsässer variables $\mathbf{z}^\pm$ track co/counter-propagating modes. Observed outward dominance ($|z^+|\gg|z^-|$) requires **reflection turbulence**: $z^+$ partly reflects to $z^-$ at wind-deceleration regions or flux-tube flexures. The **Matthaeus-Zhou / Cranmer-van Ballegooijen (2005)** reflection-driven turbulence model is the current paradigm. Spectral debate: Kolmogorov ($k^{-5/3}$, strong turbulence) vs. Iroshnikov-Kraichnan ($k^{-3/2}$, weak wave turbulence) — observations tend toward $k^{-5/3}$ but scale-dependent.

### §8. Kinetic / PIC Approaches / 운동론 접근

**한국어:**
MHD는 이온 자이로반지름(~km) 아래를 기술할 수 없다. 따라서 코로나의 kinetic 소산 스케일은 **PIC(Particle-In-Cell)**나 **하이브리드 kinetic**(이온은 PIC, 전자는 유체) 시뮬레이션이 필요. 2010년 현재 **계산 비용의 제약**이 주된 장벽이나, Markovskii, Vasquez, Gary 등의 simulation이 Alfvén 캐스케이드가 이온 사이클로트론 주파수 근처에서 감쇠하여 **준수직 가열**을 만드는 과정을 재현함. 양자화된 결과보다는 **메커니즘 시연** 단계.

**English:**
MHD cannot describe scales below the ion gyroradius (~km). Coronal kinetic dissipation requires **PIC** or **hybrid-kinetic** simulations. As of 2010 computational cost remains the main bottleneck, but work by Markovskii, Vasquez, and Gary reproduces how Alfvénic cascades damp near $\Omega_i$ and produce predominantly perpendicular heating. Still demonstration-level rather than quantitative.

### §9. Summary and Outlook / 요약과 전망

**한국어:**
Ofman은 세 가지 결론을 내린다:
1. **WKB 모델 단독으로는 부족하지만**, 여전히 빠른 바람의 "운동량 계산기"로 유용.
2. **이온-사이클로트론 공명**이 비등방 가열의 가장 유력한 후보이나, 주파수 갭을 채울 **캐스케이드 경로가 결정적 문제**.
3. **장기 과제**: reflection-driven turbulence와 kinetic 소산을 연결하는 **multi-scale 시뮬레이션**, 그리고 곧 발사될(2018) **Parker Solar Probe의 in-situ 관측**이 모델을 결정적으로 검증할 것.

**English:**
Three conclusions: (1) WKB alone is insufficient but still useful as a "momentum calculator"; (2) ion-cyclotron resonance remains the leading candidate for anisotropic heating, with the **cascade pathway to ion scales** the critical unsolved piece; (3) the near future requires **multi-scale simulations** linking reflection-driven turbulence to kinetic dissipation, with the upcoming **Parker Solar Probe** offering decisive tests.

---

## 3. Key Takeaways / 핵심 시사점

1. **Parker 열압력 구동 모델은 고속 바람을 설명하지 못한다 / Parker's thermal model cannot explain fast wind** — $T\sim 10^6$ K에서 1 AU 속도가 ~400 km/s로 포화. 관측된 ~750 km/s 극관 바람은 반드시 **추가 에너지원**이 필요. / At $T\sim 10^6$ K the solution saturates at ~400 km/s; the observed ~750 km/s polar wind needs an **extra energy source**.

2. **Alfvén 파가 운동량과 에너지의 주요 공급자 / Alfvén waves are the leading momentum and energy supplier** — Belcher & Davis(1971) in-situ 관측 이후 외향 지배 Alfvén 스펙트럼은 상시 존재. ponderomotive 힘이 바람을 가속, 감쇠가 플라스마를 가열. / Outward-dominated Alfvén spectra are universally observed; ponderomotive force accelerates, damping heats.

3. **SOHO/UVCS의 O⁵⁺ 비등방 가열이 파동-입자 공명 증거 / UVCS O⁵⁺ anisotropy is evidence for wave-particle resonance** — $T_\perp \gtrsim 10^8$ K, $T_\parallel \lesssim 10^6$ K는 이온-사이클로트론 공명만이 자연스럽게 설명. / Only ion-cyclotron resonance naturally produces $T_\perp\gg T_\parallel$.

4. **저주파-고주파 캐스케이드가 핵심 미해결 문제 / The low-to-high frequency cascade is the central unsolved problem** — 광구 주입 주파수 ~$10^{-3}$ Hz와 양성자 자이로 주파수 ~$10^3$ Hz 사이 6 decade를 어떻게 넘느냐. 난류 캐스케이드가 유일한 후보. / Bridging 6 decades between photospheric injection and proton gyrofrequency remains open; turbulent cascade is the only candidate.

5. **Reflection-driven turbulence가 현대적 표준 / Reflection-driven turbulence is the modern standard** — 외향 지배 관측($|z^+|\gg|z^-|$)을 유지하면서 캐스케이드를 만드는 유일한 자기일관 메커니즘. Cranmer-van Ballegooijen(2005)가 정량 모델화. / Only reflection turbulence simultaneously preserves outward dominance and produces a cascade (Cranmer-van Ballegooijen 2005).

6. **다유체 MHD가 필수적 / Multifluid MHD is essential** — 양성자·알파·이온 온도와 속도의 **종간 차이**를 재현하려면 단일 유체 MHD로는 불가능. $V_\alpha > V_p$와 $T_\alpha > 4T_p$를 자연스럽게 설명. / Single-fluid MHD cannot reproduce species differences like $V_\alpha > V_p$ and $T_\alpha>4T_p$.

7. **자기 flux tube 기하가 빠른 바람 속도를 결정 / Flux-tube geometry sets fast-wind speed** — super-radial 팽창은 Alfvén 속도 프로파일을 급변화시켜 ponderomotive 가속을 제어. 파동만큼이나 중요한 기하 요소. / Super-radial expansion shapes the Alfvén-speed profile and thus ponderomotive acceleration — as important as the wave physics itself.

8. **kinetic 효과가 궁극적 소산을 담당 / Kinetic effects handle ultimate dissipation** — 이온 자이로반지름 아래의 소산은 MHD로 불가능. PIC/hybrid kinetic이 코로나 가열 연결고리. 미래 계산 자원의 핵심 수혜자. / Dissipation below the ion gyroradius is outside MHD; PIC/hybrid-kinetic codes will be the main beneficiary of future computing.

---

## 4. Mathematical Summary / 수학적 요약

### A. Parker's isothermal wind / Parker 등온 태양풍

$$
\frac{1}{u}\frac{du}{dr}\left(u^2 - c_s^2\right) = \frac{2c_s^2}{r} - \frac{GM_\odot}{r^2}, \quad r_c = \frac{GM_\odot}{2c_s^2}
$$

### B. Wave-driven momentum equation / 파동 포함 운동량 방정식

$$
\rho u\frac{du}{dr} = -\frac{dp}{dr} - \frac{\rho GM_\odot}{r^2} - \frac{1}{2}\frac{d}{dr}\!\left(\frac{\langle\delta B^2\rangle}{\mu_0}\right)
$$

### C. WKB wave-action conservation / WKB 파동 작용 보존

$$
\frac{d}{dr}\!\left[\frac{\langle\delta B^2\rangle}{8\pi}\frac{(u+v_A)^2}{v_A}A(r)\right] = 0
$$

$A(r)$ = flux tube 단면적. 감쇠가 없을 때 파동 플럭스의 보존 형태.

### D. Alfvén speed / Alfvén 속도

$$
v_A(r) = \frac{B(r)}{\sqrt{\mu_0\rho(r)}}
$$

### E. Ion-cyclotron resonance / 이온-사이클로트론 공명

$$
\omega - k_\parallel v_\parallel = n\,\Omega_i, \quad \Omega_i = \frac{q_iB}{m_ic}
$$

$n=\pm 1, \pm 2, ...$. $n=+1$이 가장 일반적 공명 모드.

### F. Elsässer variables and MHD turbulence / Elsässer 변수와 MHD 난류

$$
\mathbf{z}^\pm = \mathbf{u}\pm\frac{\mathbf{b}}{\sqrt{\mu_0\rho}}
$$

$$
\partial_t \mathbf{z}^\pm + (\mathbf{u}\mp\mathbf{v}_A)\cdot\nabla\mathbf{z}^\pm = -\rho^{-1}\nabla p_T + \text{(reflection + dissipation)}
$$

### G. Turbulent cascade spectra / 난류 캐스케이드 스펙트럼

- Kolmogorov (strong): $E(k) = C_K\,\epsilon^{2/3}k^{-5/3}$
- Iroshnikov-Kraichnan (weak wave): $E(k) = C_{\rm IK}(\epsilon\,v_A)^{1/2}k^{-3/2}$

### H. Reflection coefficient / 반사 계수

반사 난류에서 $z^-$가 $z^+$로부터 생성되는 효율:

$$
R(r) = \frac{|z^-|}{|z^+|} \approx \frac{|\partial_r v_A|}{|\omega|}
$$

### I. Heating rate per unit mass / 단위 질량당 가열율

$$
Q = \rho\,\langle z^+z^-\rangle\left(\frac{|z^+|+|z^-|}{2\lambda_\perp}\right) \quad \text{(Hossain et al. 1995)}
$$

$\lambda_\perp$ = 수직 상관 길이. 두 Elsässer 성분의 **상호 존재**가 가열의 필수 조건.

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1942  Alfvén — 자기유체 파동 / hydromagnetic waves
  │
1951  Biermann — 혜성 꼬리 증거 / comet-tail evidence
  │
1958  Parker — 태양풍 이론 예측 / theoretical prediction
  │
1962  Mariner 2 — 첫 in-situ / first in-situ measurement
  │
1971  Belcher & Davis — Alfvén 변동 관측 / Alfvén fluctuations observed
  │
1975+ Hollweg — WKB 1D 파동 구동 / WKB wave-driven wind
  │
1988  Parker — nanoflare 가설 / nanoflare hypothesis (alt. channel)
  │
1990s Ulysses — 고위도 bimodal 풍 / bimodal wind over poles
  │
1998  SOHO/UVCS — O⁵⁺ 비등방 가열 / anisotropic ion heating
  │
2000  Matthaeus-Zhou — reflection turbulence 개념 / concept of reflection turbulence
  │
2005  Cranmer & van Ballegooijen — self-consistent reflection-driven model
  │
2007  STEREO — 3D 태양 고리 / 3D view of coronal loops
  │
▶ 2010  ★ Ofman — "Wave Modeling of the Solar Wind" (이 논문 / THIS PAPER)
  │
2018  Parker Solar Probe — 20 $R_\odot$ 내부 진입 / first in-situ inside 20 $R_\odot$
2019  Bale et al. — switchback 발견 / discovery of switchbacks
2020  Solar Orbiter — 고위도 + 원격 조합 / high-lat + remote combo
2021  Berghmans et al. — EUI "campfires" nanoflare 후보 / campfire nanoflare candidates
2023+ PSP perihelion <10 $R_\odot$ — 코로나 진입 / entering the corona
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **LRSP #3 — Marsch, *Kinetic Physics of the Solar Wind Plasma* (2006)** | in-situ 이온 분포 함수 관측 — §6 비등방성의 근거 / in-situ ion distribution functions, basis for §6 | 필수 선행 / Prerequisite |
| **LRSP #8 — Cranmer, *Coronal Heating & Solar Wind* (2009)** | 자매 리뷰, 코로나 가열 문제 중심 / companion review, emphasis on coronal heating | 보완적 / Complementary |
| **Parker (1958) *ApJ 128, 664*** | 고전 열압력 바람 이론의 원조 / original thermal wind | §3 출발점 / §3 starting point |
| **Alfvén (1942) *Nature 150, 405*** | Alfvén 파 자체의 기원 / origin of Alfvén waves | 전체 개념의 뿌리 / root of all |
| **Belcher & Davis (1971)** | in-situ Alfvén 변동의 정량적 관측 / quantitative Alfvén fluctuations | §2 실증 / §2 empirical basis |
| **Kohl et al. (1998) SOHO/UVCS** | O⁵⁺ 비등방 온도 발견 / discovery of O⁵⁺ anisotropy | §6 동기 / motivation of §6 |
| **Cranmer & van Ballegooijen (2005) *ApJS 156, 265*** | reflection-driven turbulence 정량 모델 / quantitative reflection-turbulence model | §7 핵심 모델 / core model of §7 |
| **Hollweg (1978) *GRL 5, 731*** | WKB 1D wave-driven fluid model | §4 근간 / foundation of §4 |
| **Matthaeus & Goldstein (1982)** | Elsässer 난류 스펙트럼 분석 / Elsässer spectrum analysis | §7 도구 / §7 toolkit |
| **Bale et al. (2019) PSP switchbacks** | 이 리뷰 이후 결정적 in-situ 관측 / decisive post-review in-situ | 후속 검증 / follow-up validation |

---

## 7. References / 참고문헌

- Ofman, L., "Wave Modeling of the Solar Wind", *Living Reviews in Solar Physics*, 7, 4, 2010. [DOI: 10.12942/lrsp-2010-4]
- Alfvén, H., "Existence of electromagnetic-hydrodynamic waves", *Nature*, 150, 405, 1942.
- Belcher, J. W., Davis, L., "Large-amplitude Alfvén waves in the interplanetary medium, 2", *JGR*, 76, 3534, 1971.
- Biermann, L., "Kometenschweife und solare Korpuskularstrahlung", *Zeitschr. Astrophys.*, 29, 274, 1951.
- Cranmer, S. R., van Ballegooijen, A. A., "On the generation, propagation and reflection of Alfvén waves from the solar photosphere to the distant heliosphere", *ApJS*, 156, 265, 2005.
- Hollweg, J. V., "Some physical processes in the solar wind", *Rev. Geophys. Space Phys.*, 16, 689, 1978.
- Kohl, J. L., Noci, G., Antonucci, E., et al., "UVCS/SOHO empirical determinations of anisotropic velocity distributions in the solar corona", *ApJ*, 501, L127, 1998.
- Markovskii, S. A., Vasquez, B. J., "Soliton-like structures on the outer-scale separatrix in anisotropic two-dimensional magnetohydrodynamic turbulence", *ApJ*, 739, 22, 2010.
- Marsch, E., "Kinetic Physics of the Solar Corona and Solar Wind", *Living Reviews in Solar Physics*, 3, 1, 2006.
- Matthaeus, W. H., Goldstein, M. L., "Measurement of the rugged invariants of magnetohydrodynamic turbulence in the solar wind", *JGR*, 87, 6011, 1982.
- Parker, E. N., "Dynamics of the Interplanetary Gas and Magnetic Fields", *ApJ*, 128, 664, 1958.
- Ulysses mission papers (Smith et al. 1995, McComas et al. 2000) — solar wind bimodal structure over heliographic latitude.
