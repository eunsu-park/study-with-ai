---
title: "Solar Flares: Magnetohydrodynamic Processes"
authors: ["Kazunari Shibata", "Tetsuya Magara"]
year: 2011
journal: "Living Reviews in Solar Physics"
doi: "10.12942/lrsp-2011-6"
topic: Living_Reviews_in_Solar_Physics
tags: [solar-flare, magnetic-reconnection, MHD, CSHKP, plasmoid, tearing, chromospheric-evaporation, flux-emergence]
status: completed
date_started: 2026-04-23
date_completed: 2026-04-23
---

# 27. Solar Flares: Magnetohydrodynamic Processes / 태양 플레어: 자기유체역학 과정

---

## 1. Core Contribution / 핵심 기여

**English.** Shibata and Magara deliver a comprehensive, pedagogical, and historically-grounded review of the MHD mechanisms that power solar flares. Their central claim is unambiguous: a flare is the explosive dissipation of coronal electric current that has been quasi-statically built up as field-aligned current (free magnetic energy) via flux emergence and photospheric shear, and this dissipation proceeds through magnetic reconnection inside thin current sheets. The review is organised along the **energy chain** of a flare — energy build-up (Section 3), energy release (Section 4), energy transport (Section 5) — and at each stage it sets the observational facts (Yohkoh / SOHO / TRACE / RHESSI / Hinode era) against the full family of theoretical models (CSHKP phenomenology, Sweet-Parker and Petschek reconnection, tearing and plasmoid instabilities, breakout and emerging-flux-trigger eruptions). The synthesis produced is the **plasmoid-induced reconnection / fractal-current-sheet paradigm** that explains (a) why reconnection is fast despite large Lundquist number, (b) why multiple time- and length-scales coexist in a flare, and (c) how a single underlying physics generates the full observed taxonomy — from the smallest X-ray jets and microflares to the largest LDE flares, giant arcades, and stellar superflares.

**한국어.** Shibata와 Magara는 태양 플레어를 구동하는 MHD 메커니즘에 대한 포괄적이고 교육적이며 역사적 근거가 있는 리뷰를 제공한다. 그들의 핵심 주장은 분명하다: 플레어란 자속 방출과 광구 shear를 통해 자기장 정렬 전류 (자유 자기에너지)로 준정적으로 축적된 코로나 전류의 폭발적 소산이며, 이 소산은 얇은 current sheet 내부의 자기 재결합을 통해 진행된다는 것이다. 리뷰는 플레어의 **에너지 사슬**에 따라 구성되어 있다 — 에너지 축적 (3장), 에너지 해방 (4장), 에너지 수송 (5장) — 각 단계에서 관측 사실 (Yohkoh / SOHO / TRACE / RHESSI / Hinode 시대)을 이론 모델 전체 계열 (CSHKP 현상론, Sweet-Parker와 Petschek 재결합, tearing 및 플라즈모이드 불안정성, breakout 및 emerging-flux-trigger 분출)에 대조한다. 도출된 종합은 **플라즈모이드 유도 재결합 / 프랙털 current sheet 패러다임**이며, 이는 (a) 높은 Lundquist 수에도 재결합이 빠른 이유, (b) 플레어에서 여러 시간·길이 규모가 공존하는 이유, (c) 하나의 근원적 물리가 어떻게 관측된 전체 분류 — 가장 작은 X-선 제트와 마이크로플레어로부터 가장 큰 LDE 플레어, 거대 아케이드, 항성 슈퍼플레어까지 — 를 생성하는지 설명한다.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction & Observational Taxonomy / 서론 및 관측 분류 (pp. 5 – 22, §1 – §2)

**English.** The paper opens (§1, p. 5) with the energetic scales: flare energies 10^28 – 10^32 erg released over 10^3 – 10^4 s, loop heights 10^4 – 10^5 km. The foundational equation (their Eq. 1) estimates the magnetic energy stored in a typical sunspot:

$$E_\text{mag} \simeq \frac{B^2}{8\pi} L^3 \simeq 10^{33} \left(\frac{B}{10^3 \text{ G}}\right)^2 \left(\frac{L}{3\times 10^9 \text{ cm}}\right)^3 \text{ erg}.$$

Only a fraction of this — the *free* energy above the potential-field minimum — is actually available. The classical reconnection-time-scale puzzle is set up via the Spitzer diffusion time (their Eq. 2 – 3):

$$t_\text{dif} \simeq L^2/\eta \simeq 10^{14} (L/10^9 \text{ cm})^2 (T/10^6 \text{ K})^{3/2} \text{ s},$$

which exceeds a flare's lifetime by ten orders of magnitude — motivating the need for dynamical reconnection models.

The authors (§1, p. 7 – 8) credit Giovanelli (1946) and Hoyle (1949) with proposing that neutral points / X-points dissipate field energy; Sweet (1958) and Parker (1957) supplied the first MHD model including plasma flow; Petschek (1964) added standing slow shocks to give a *fast* rate. The paper then walks through the four phenomenological progenitors of the CSHKP standard model (Figure 2, p. 7): Carmichael (1964) single-ribbon sketch, Sturrock (1966) Y-type neutral point, Hirayama (1974) rising prominence + shock compression, Kopp & Pneuman (1976) helmet-streamer reconnection. Švestka & Cliver (1992) coined "CSHKP"; Shibata (1999) modified it to include the rising plasmoid and reconnection jet (Figure 3b, p. 9).

§2 (pp. 12 – 22) classifies the observational zoo: **LDE flares** with cusp-shaped SXR loops (Feb 21 1992, Tsuneta 1992a; Figure 4), **giant arcades** > 1 solar radius (McAllister 1996; Figure 6), **impulsive / compact flares** with loop-top HXR source (Masuda 1994; Figure 8), and **transient brightenings / X-ray jets** (Shibata 1992, 1994). Table 1 (p. 21) lists flare families by length scale (10^4 – 10^5 km), and Table 4 (p. 22) presents the unified picture — all of these have the same basic CSHKP geometry, differing only in size, field strength, and mass-ejection style.

**한국어.** 논문은 (§1, p. 5) 에너지 규모로 시작한다: 10^3 – 10^4 s에 걸쳐 방출되는 10^28 – 10^32 erg 플레어 에너지, 10^4 – 10^5 km 루프 높이. 기초 식 (Eq. 1)은 전형적인 흑점에 저장된 자기에너지를 추정한다:

$$E_\text{mag} \simeq \frac{B^2}{8\pi} L^3 \simeq 10^{33} \left(\frac{B}{10^3 \text{ G}}\right)^2 \left(\frac{L}{3\times 10^9 \text{ cm}}\right)^3 \text{ erg}.$$

이 중 일부 — 포텐셜 장 최소값 위의 *자유* 에너지 — 만 실제로 사용 가능하다. 고전적 재결합 시간 규모 퍼즐은 Spitzer 확산 시간 (Eq. 2 – 3)으로 제시된다:

$$t_\text{dif} \simeq L^2/\eta \simeq 10^{14} (L/10^9 \text{ cm})^2 (T/10^6 \text{ K})^{3/2} \text{ s},$$

이는 플레어 수명을 10차수 초과하며 — 동역학적 재결합 모델의 필요성을 동기 부여한다.

저자들은 (§1, p. 7 – 8) Giovanelli (1946)와 Hoyle (1949)가 중성점 / X점이 자기장 에너지를 소산한다고 제안한 공로를 인정한다; Sweet (1958)와 Parker (1957)는 플라즈마 흐름을 포함한 최초의 MHD 모델을 제공했다; Petschek (1964)는 정상 느린 충격파를 추가하여 *빠른* 속도를 주었다. 이어서 논문은 CSHKP 표준 모델의 네 현상론적 전구체 (Figure 2, p. 7)를 다룬다: Carmichael (1964) 단일 ribbon 스케치, Sturrock (1966) Y형 중성점, Hirayama (1974) 상승 prominence + 충격 압축, Kopp & Pneuman (1976) 헬멧 streamer 재결합. Švestka & Cliver (1992)가 "CSHKP"를 명명했다; Shibata (1999)는 상승 플라즈모이드와 reconnection jet을 포함하도록 이를 수정했다 (Figure 3b, p. 9).

§2 (pp. 12 – 22)는 관측적 동물원을 분류한다: cusp 형 SXR 루프를 갖는 **LDE 플레어** (1992년 2월 21일, Tsuneta 1992a; Figure 4), > 1 태양 반경의 **거대 아케이드** (McAllister 1996; Figure 6), 루프톱 HXR source를 갖는 **임펄시브 / 컴팩트 플레어** (Masuda 1994; Figure 8), **transient brightenings / X-선 제트** (Shibata 1992, 1994). Table 1 (p. 21)은 길이 규모 (10^4 – 10^5 km)별로 플레어 가족을 나열하며, Table 4 (p. 22)는 통합된 그림을 제시한다 — 이 모두가 같은 기본 CSHKP 기하를 가지며, 크기, 자기장 세기, 질량 방출 방식만 다르다.

### Part II: Energy Build-up / 에너지 축적 (pp. 23 – 44, §3)

**English.** §3.1 (Flux emergence, p. 23) is the mechanism by which free energy is *pumped* into the corona. Sub-photospheric twisted flux ropes rise buoyantly (Parker 1955 instability, magnetic Rayleigh-Taylor), cross the photosphere, and expand rapidly into the corona carrying their twist and shear. §3.1.1 reviews morphology (tongues, arch-filament-systems, U-loops). §3.1.2 reviews dynamics: Shibata et al. (1989) 2D simulations show that an emerging flux tube expands self-similarly and reconnects with overlying coronal field if the polarities are favorable (Yokoyama & Shibata 1995, 1996; Figure 30). §3.1.3 surveys 3D simulations (Magara 2001, Manchester et al. 2004, Archontis 2004; Figure 35) and the self-consistent generation of shear and converging flows.

§3.2 (Magnetic structure) describes the quasi-static coronal states: **filaments/prominences** (cool T ~ 10^4 K plasma suspended in 10^6 K corona, held by the tension of twisted/dipped field lines), **sigmoids** (S-shaped SXR structures associated with CME productivity; Canfield, Hudson, McKenzie 1999), **force-free fields** (J × B = 0 so curl(B) = alpha(r) B, with alpha = const for linear force-free, alpha = alpha(field-line) for non-linear). The **Aly-Sturrock conjecture** (§3.2.4) states that a simply-connected force-free field with fixed normal B at the boundary cannot have more energy than the fully-open field — providing an upper energy bound and a puzzle: how can an eruption expel the field against this bound? Resolutions include disconnection via reconnection, multipolar topologies, and partial openings.

§3.3 (Magnetic helicity) is the topological invariant:

$$H = \int_V \mathbf{A}\cdot\mathbf{B}\, dV,$$

gauge-invariant if measured as relative helicity (Berger & Field 1984). Because H is nearly conserved under reconnection (it can only dissipate on resistive timescales), it must be *shed* from active regions into the heliosphere — which is the role of CMEs (Démoulin 2007). The measurement of helicity injection through the photospheric boundary constrains flare/CME productivity.

**한국어.** §3.1 (자속 방출, p. 23)은 자유 에너지가 코로나로 *주입*되는 메커니즘이다. 광구 아래 twisted flux rope이 부력으로 상승하여 (Parker 1955 불안정성, 자기 Rayleigh-Taylor), 광구를 가로지르고, twist와 shear를 운반하며 코로나로 급격히 확장한다. §3.1.1은 형태 (혀 모양, arch-filament-systems, U-loops)를 검토한다. §3.1.2는 동역학을 검토한다: Shibata et al. (1989) 2D 시뮬레이션은 emerging flux tube이 자기 닮은꼴로 확장하며 극성이 유리하면 위의 코로나 자기장과 재결합함을 보인다 (Yokoyama & Shibata 1995, 1996; Figure 30). §3.1.3은 3D 시뮬레이션 (Magara 2001, Manchester et al. 2004, Archontis 2004; Figure 35)과 shear 및 수렴 흐름의 자기일관적 생성을 개관한다.

§3.2 (자기 구조)는 준정적 코로나 상태를 기술한다: **필라멘트/홍염** (10^6 K 코로나에 매달린 차가운 T ~ 10^4 K 플라즈마, twisted/dipped 자기력선의 tension으로 유지), **시그모이드** (CME 생산성과 연관된 S형 SXR 구조; Canfield, Hudson, McKenzie 1999), **force-free field** (J × B = 0이므로 curl(B) = alpha(r) B, 선형 force-free에서 alpha = const, 비선형에서 alpha = alpha(field-line)). **Aly-Sturrock 추측** (§3.2.4)은 단순연결 force-free 자기장이 경계에서 고정된 법선 B를 가질 때 완전 열린 자기장보다 많은 에너지를 가질 수 없다고 한다 — 이는 에너지 상한과 퍼즐을 동시에 제공한다: 어떻게 분출이 이 상한에 저항하여 자기장을 밀어낼 수 있는가? 해결책은 재결합을 통한 disconnection, 다극 위상, 부분적 열림을 포함한다.

§3.3 (자기 helicity)는 위상적 불변량이다:

$$H = \int_V \mathbf{A}\cdot\mathbf{B}\, dV,$$

상대 helicity로 측정하면 게이지 불변 (Berger & Field 1984). H는 재결합 하에서 거의 보존되므로 (저항 시간 규모에서만 소산 가능), 활동 영역에서 헬리오스피어로 *방출*되어야 한다 — 이것이 CME의 역할이다 (Démoulin 2007). 광구 경계를 통한 helicity 주입 측정은 플레어/CME 생산성을 제약한다.

### Part III: Energy Release — Magnetic Reconnection / 에너지 해방 — 자기 재결합 (pp. 45 – 60, §4)

**English.** This is the mathematical core of the paper (§4.1, pp. 45 – 49). A steady 2D current sheet (Figure 26, p. 46) has length L, width w, inflow B_i, v_i, diffusion region resistivity eta. Faraday's law and Ohm's law give (Eq. 16):

$$d\Phi/dt = cE = 4\pi c^{-1} \eta j = v_i B_i = \text{constant}.$$

Dividing by v_A B_i defines the non-dimensional reconnection rate as the inflow Alfvén Mach number (Eq. 17):

$$M_A = v_i/v_A.$$

The reconnection time (Eq. 18) is t_rec = L/v_i = t_A/M_A. With t_A = L/v_A ~ 10 – 100 s for coronal scales, the question becomes: **what sets M_A?**

**Sweet-Parker (1957-58; Eq. 19).** Conservation of mass and energy in a long thin sheet (L >> w) gives

$$M_A \simeq R_m^{-1/2}, \qquad R_m = v_A L/\eta.$$

With coronal R_m ~ 10^14, M_A ~ 10^(-7), giving t_rec ~ 10^8 – 10^9 s — far too slow.

**Petschek (1964; Eq. 21).** By shortening the diffusion region to a tiny patch and letting most of the inflow be deflected through two pairs of *standing slow-mode shocks*, the reconnection rate becomes

$$M_A \simeq \frac{\pi}{8 \ln R_m} \sim 0.01\text{–}0.1 ,$$

fast enough to match flare timescales. Plug into Eq. 23 with typical coronal numbers: v_i = 100 km/s, B_i = 100 G, L = 2 × 10^9 cm → dE/dt ~ 4 × 10^28 erg/s, matching Yohkoh impulsive-phase luminosities (Masuda et al. 1994).

**Locally enhanced resistivity (§4.1.4).** Numerical simulations (Sato & Hayashi 1979; Ugai & Tsuda 1977; Biskamp 1986; Scholer 1989) show that Petschek reconnection is achieved only if eta is **locally enhanced** inside the diffusion region — uniform eta collapses to Sweet-Parker. Driven-type reconnection (external inflow) and spontaneous-type (triggered by micro-instabilities) are both possible, but in both cases the *nature of the current sheet* — its thinning to ion-kinetic scales — is what switches on anomalous resistivity.

**Tearing instability and fractal reconnection (§4.1.5; Furth, Killeen, Rosenbluth 1963).** A Sweet-Parker sheet is unstable to long-wavelength perturbations that grow on a timescale intermediate between t_A and t_R:

$$\gamma \sim t_A^{-2/5} t_R^{-3/5},$$

breaking the sheet into magnetic islands. Secondary tearing inside the thinning inter-island current sheets produces a **fractal current sheet** (Shibata & Tanuma 2001; Figures 27 – 28) — a hierarchy of islands at progressively smaller scales until the kinetic (ion-Larmor) scale is reached. The reconnection rate effectively decouples from resistivity.

**Plasmoid-induced reconnection (§4.1.6).** Shibata (1999) argues that plasmoid ejection is not a byproduct of reconnection but an *active driver*: a stationary plasmoid chokes inflow; its ejection strongly draws in new flux, accelerating the rate. The positive feedback (plasmoid ejection → enhanced inflow → more magnetic flux → more reconnection → faster plasmoid ejection) generates Alfvénic-speed plasmoid motion, which the authors connect to observed plasmoid speeds (Ohyama & Shibata 1998, Figure 5 of the paper).

**Current-sheet formation (§4.2).** Two paths: (1) interaction of separate flux domains (Figure 29 — emerging bipole meets overlying oblique field, producing an interface current sheet; Yokoyama & Shibata 1995, 1996), and (2) internal shear development in a single flux domain (Mikić et al. 1988, Choe & Lee 1996, Figure 34 — footpoint shearing produces a vertical current sheet above the polarity inversion line).

**Energy-release modeling (§4.3).** For bipolar (single-domain) systems: **mass-loaded** (Low 1996), **flux-cancellation** (Linker et al. 2003, Figure 37a), **tether-cutting** (Moore et al. 2001, Figure 37b), **kink instability** (Fan & Gibson 2003, Figure 37c), **loss-of-equilibrium** (Forbes & Isenberg 1991, Figure 37e), and **torus instability** (Kliem & Török 2006). For multi-polar systems: **breakout** (Antiochos et al. 1999, Figure 38 — overlying field reconnects first, freeing inner flux rope) and **emerging-flux trigger** (Chen & Shibata 2000, Figure 39 — new flux emerges near inversion line). These are all consistent: they differ in the *trigger* but share the same engine (current-sheet reconnection).

**한국어.** 이것이 논문의 수학적 핵심이다 (§4.1, pp. 45 – 49). 정상 2D current sheet (Figure 26, p. 46)은 길이 L, 폭 w, 유입 B_i, v_i, 확산 영역 저항도 eta를 갖는다. Faraday 법칙과 Ohm 법칙은 (Eq. 16):

$$d\Phi/dt = cE = 4\pi c^{-1} \eta j = v_i B_i = \text{상수}.$$

v_A B_i로 나누면 무차원 재결합 속도가 유입 Alfvén 마하수로 정의된다 (Eq. 17): M_A = v_i/v_A. 재결합 시간 (Eq. 18)은 t_rec = L/v_i = t_A/M_A이다. 코로나 규모에서 t_A = L/v_A ~ 10 – 100 s이므로, 질문은 **무엇이 M_A를 결정하는가?**가 된다.

**Sweet-Parker (1957-58; Eq. 19).** 긴 얇은 sheet (L >> w)에서 질량·에너지 보존은 M_A ≃ R_m^(-1/2)를 준다. 코로나 R_m ~ 10^14로 M_A ~ 10^(-7), t_rec ~ 10^8 – 10^9 s — 너무 느림.

**Petschek (1964; Eq. 21).** 확산 영역을 작은 패치로 짧게 하고 유입의 대부분을 두 쌍의 *정상 느린모드 충격파*로 굴절시키면, 재결합 속도는 M_A ≃ π/(8 ln R_m) ~ 0.01 – 0.1로 플레어 시간 규모에 부합한다. Eq. 23에 전형 값 v_i = 100 km/s, B_i = 100 G, L = 2 × 10^9 cm를 대입 → dE/dt ~ 4 × 10^28 erg/s, Yohkoh 임펄시브 단계 광도와 일치 (Masuda et al. 1994).

**국소적으로 향상된 저항도 (§4.1.4).** 수치 시뮬레이션 (Sato & Hayashi 1979; Ugai & Tsuda 1977; Biskamp 1986; Scholer 1989)은 Petschek 재결합이 확산 영역 내부에서 eta가 **국소적으로 향상**될 때만 달성됨을 보인다 — 균일 eta는 Sweet-Parker로 붕괴한다. Driven-type (외부 유입)과 spontaneous-type (미시 불안정성 유발) 모두 가능하지만, 두 경우 모두 *current sheet의 본성* — 이온-kinetic 규모로의 박화 — 이 이상(異常) 저항도를 켜는 것이다.

**Tearing 불안정성과 프랙털 재결합 (§4.1.5; Furth, Killeen, Rosenbluth 1963).** Sweet-Parker sheet는 t_A와 t_R 사이의 시간 규모로 성장하는 장파장 섭동에 대해 불안정하다:

$$\gamma \sim t_A^{-2/5} t_R^{-3/5},$$

이는 sheet를 자기 섬들로 쪼갠다. 박화되는 섬간 current sheet 내부의 2차 tearing은 **프랙털 current sheet**를 생성한다 (Shibata & Tanuma 2001; Figures 27 – 28) — kinetic (이온-Larmor) 규모에 도달할 때까지 점점 작은 규모에서의 섬들의 계층 구조. 재결합 속도는 실질적으로 저항도로부터 분리된다.

**플라즈모이드 유도 재결합 (§4.1.6).** Shibata (1999)는 플라즈모이드 방출이 재결합의 부산물이 아니라 *능동적 구동자*라고 주장한다: 정지한 플라즈모이드는 유입을 막고, 그 방출은 새 자속을 강하게 끌어당겨 속도를 가속한다. 양성 피드백 (플라즈모이드 방출 → 향상된 유입 → 더 많은 자속 → 더 많은 재결합 → 더 빠른 플라즈모이드 방출)은 Alfvén 속도 플라즈모이드 운동을 생성하며, 저자들은 이를 관측된 플라즈모이드 속도와 연결한다 (Ohyama & Shibata 1998).

**Current sheet 형성 (§4.2).** 두 경로: (1) 별개 flux 영역의 상호작용 (Figure 29 — 방출 bipole이 위쪽 경사 자기장을 만나 계면 current sheet 생성; Yokoyama & Shibata 1995, 1996), (2) 단일 flux 영역 내의 내부 shear 발달 (Mikić et al. 1988, Choe & Lee 1996, Figure 34 — 발자국 shear가 극성 반전선 위에 수직 current sheet를 생성).

**에너지 해방 모델링 (§4.3).** 단극성 (단일 영역) 시스템: **mass-loaded** (Low 1996), **flux-cancellation** (Linker et al. 2003), **tether-cutting** (Moore et al. 2001), **kink 불안정성** (Fan & Gibson 2003), **loss-of-equilibrium** (Forbes & Isenberg 1991), **torus 불안정성** (Kliem & Török 2006). 다극성 시스템: **breakout** (Antiochos et al. 1999 — 위 자기장이 먼저 재결합하여 내부 flux rope 해방)과 **emerging-flux trigger** (Chen & Shibata 2000 — 반전선 근처에서 새 자속 방출). 모두 일관적이다: 이들은 *trigger*가 다르지만 같은 엔진 (current-sheet 재결합)을 공유한다.

### Part IV: Energy Transport / 에너지 수송 (pp. 61 – 71, §5)

**English.** §5.1 (Radiation). The canonical flare light-curve (Figure 41): precursor → impulsive phase (HXR > 30 keV, microwave, Hα kernels all peak together; Neupert effect — SXR luminosity approximately proportional to the time-integral of HXR) → gradual phase (SXR loops and Hα ribbons rise gradually as reconnection propagates to higher field lines). Figure 42 contrasts impulsive-phase geometry (deep SXR loop with HXR loop-top source and footpoint kernels — Masuda 1994) vs. gradual-phase geometry (shrinking cusp with isothermal slow shock and conduction front above it).

§5.2 (Mass ejection). Reconnection jets are bidirectional flows at the Alfvén speed v_A. Their width scales as w_jet ~ M_A × L. For Sweet-Parker M_A ~ 10^(-7) gives w_jet ~ 100 cm (laughably thin); for Petschek M_A ~ 0.01 – 0.1 gives w_jet ~ 100 – 1000 km — comparable to observed jet widths. Plasmoid ejection speed grows exponentially then saturates; Shibata & Tanuma (2001) Eq. 26 – 27 give:

$$v = \frac{v_A e^{\omega t}}{e^{\omega t} - 1 + v_A/v_0}, \qquad \omega = \rho_0 v_A / (\rho_p L).$$

§5.3 (Shock formation). Slow MHD shocks extend from the diffusion region (Petschek geometry) and heat coronal plasma to

$$T_\text{slow shock} \sim T_\text{corona}/\beta, \qquad \beta = 2nkT/(B^2/8\pi) \sim 0.01,$$

suggesting heating from 1 MK to ~ 100 MK — overestimated because thermal conduction relaxes the temperature (Yokoyama & Shibata 1997 confirm in MHD simulation: adiabatic slow shock splits into conduction front + isothermal slow shock).

§5.3.2 (Chromospheric evaporation). Thermal conduction from the super-hot loop top drives ablation of chromospheric plasma up into the loop. In steady state:

$$\kappa_0 T^{7/2}/L^2 \sim (5/2) p v_\text{evap},$$

with evaporation speed of order the sound speed, v_evap ~ c_s ~ 500 (T/10^7 K)^(1/2) km/s. Combining with reconnection heating Q ~ B^2 v_A/(4 pi L) gives the famous scaling law (Eq. 38; Yokoyama & Shibata 1998):

$$T_\text{loop} \sim (B^2 v_A L / (4\pi \kappa_0))^{2/7} \sim 4\times 10^7 (B/100\text{ G})^{6/7} (n/10^{10} \text{ cm}^{-3})^{-1/7} (L/10^9 \text{ cm})^{2/7} \text{ K}.$$

This is one of the most important predictions of the reconnection-driven flare paradigm because it directly links magnetic field strength, density, and loop length to the observed loop temperature.

§5.3.3 (Fast shock). Downward reconnection jet impinges on the top of the SXR loop, producing a fast MHD shock where HXR loop-top sources may be generated:

$$T_\text{fast shock} \sim m_i v_\text{jet}^2/(6 k_B) \sim 2\times 10^8 (B/100\text{ G})^2 (n_e/10^{10} \text{ cm}^{-3})^{-1} \text{ K}.$$

§5.4 (Wave propagation). **Moreton waves** (chromospheric H-alpha signature of coronal fast-mode shock, Uchida 1968), **EIT waves** (EUV dimming wavefronts, Thompson 1998). Chen et al. (2002) explain both as parts of a single expanding magnetic structure.

§5.5 (Particle acceleration). The convective electric field in the reconnection inflow:

$$E = (v_i/c) B_i \sim 3\times 10^3 (M_A/0.1)(B/100\text{ G})^2 (n_\text{jet}/10^{10})^{-1/2} \text{ V m}^{-1},$$

accelerates protons and electrons to relativistic energies. Only a tiny fraction of the flare volume is accelerator-scale, but it produces the observed GeV ions and hundreds-of-keV electrons.

**한국어.** §5.1 (복사). 정준 플레어 광도곡선 (Figure 41): precursor → 임펄시브 단계 (HXR > 30 keV, 마이크로파, Hα kernels가 함께 피크; Neupert 효과 — SXR 광도가 HXR의 시간 적분에 근사적으로 비례) → 점진 단계 (SXR 루프와 Hα ribbons가 재결합이 더 높은 자기력선으로 전파함에 따라 점진적으로 상승). Figure 42는 임펄시브 단계 기하 (깊은 SXR 루프 + HXR 루프톱 source + 발자국 kernels — Masuda 1994)와 점진 단계 기하 (수축하는 cusp + 등온 느린 충격파 + 위쪽 전도 전선)를 대비한다.

§5.2 (질량 방출). Reconnection jet은 Alfvén 속도 v_A의 양방향 흐름이다. 폭은 w_jet ~ M_A × L로 스케일된다. Sweet-Parker M_A ~ 10^(-7)은 w_jet ~ 100 cm (우스꽝스러울 정도로 얇음); Petschek M_A ~ 0.01 – 0.1은 w_jet ~ 100 – 1000 km — 관측된 jet 폭과 비슷하다. 플라즈모이드 방출 속도는 지수함수적으로 증가 후 포화; Shibata & Tanuma (2001) Eq. 26 – 27은:

$$v = \frac{v_A e^{\omega t}}{e^{\omega t} - 1 + v_A/v_0}, \qquad \omega = \rho_0 v_A / (\rho_p L).$$

§5.3 (충격파 형성). 느린 MHD 충격파는 확산 영역 (Petschek 기하)에서 뻗어나와 코로나 플라즈마를 T_slow shock ~ T_corona/β, β = 2nkT/(B^2/8π) ~ 0.01로 가열한다 — 1 MK에서 ~ 100 MK로의 가열을 시사하지만 열전도가 온도를 완화하므로 과대평가된다 (Yokoyama & Shibata 1997 MHD 시뮬레이션에서 확인: 단열 느린 충격파가 전도 전선 + 등온 느린 충격파로 분리).

§5.3.2 (채층 증발). 초고온 루프 top으로부터의 열전도가 채층 플라즈마를 루프로 끌어올린다. 정상 상태에서 κ_0 T^(7/2)/L^2 ~ (5/2) p v_evap로, 증발 속도는 음속 크기 v_evap ~ c_s ~ 500 (T/10^7 K)^(1/2) km/s. 재결합 가열 Q ~ B^2 v_A/(4 π L)와 결합하면 유명한 스케일링 법칙 (Eq. 38; Yokoyama & Shibata 1998):

$$T_\text{loop} \sim (B^2 v_A L / (4\pi \kappa_0))^{2/7} \sim 4\times 10^7 (B/100\text{ G})^{6/7} (n/10^{10})^{-1/7} (L/10^9)^{2/7} \text{ K}.$$

이는 자기장 세기, 밀도, 루프 길이를 관측된 루프 온도와 직접 연결하므로 재결합 구동 플레어 패러다임의 가장 중요한 예측 중 하나이다.

§5.3.3 (빠른 충격파). 하향 reconnection jet이 SXR 루프 top에 충돌하여 HXR 루프톱 source가 생성될 수 있는 빠른 MHD 충격파를 생성: T_fast shock ~ m_i v_jet^2/(6 k_B) ~ 2 × 10^8 (B/100 G)^2 (n_e/10^10)^(-1) K.

§5.4 (파동 전파). **Moreton 파동** (코로나 빠른모드 충격파의 채층 H-alpha 서명, Uchida 1968), **EIT 파동** (EUV dimming wavefront, Thompson 1998). Chen et al. (2002)는 둘 다 단일 확장 자기 구조의 부분으로 설명한다.

§5.5 (입자 가속). 재결합 유입의 대류 전기장 E = (v_i/c) B_i ~ 3 × 10^3 (M_A/0.1)(B/100 G)^2 (n_jet/10^10)^(-1/2) V/m은 양성자와 전자를 상대론적 에너지로 가속한다. 플레어 부피의 작은 부분만이 가속기 규모이지만, 관측되는 GeV 이온과 수백-keV 전자를 생성한다.

### Part V: Stellar Flares and Concluding Remarks / 항성 플레어 및 결론 (pp. 71 – 75, §6 – §7)

**English.** §6 applies the flare scaling laws to stellar flares. Observed stellar flares span T = 10^7 – 10^8 K and E = 10^29 – 10^37 erg — much larger than the Sun. The universal emission-measure vs temperature correlation (Feldman 1995; Shibata & Yokoyama 1999, 2002) is derived from the combination of the reconnection temperature scaling (Eq. 38), hydrostatic balance 2nkT = B^2/(8 pi), and EM = n^2 L^3 to give (Eq. 43):

$$EM \simeq 10^{48} (B/50\text{ G})^{-5} (n_0/10^9)^{3/2} (T/10^7 \text{ K})^{17/2} \text{ cm}^{-3}.$$

All solar microflares, flares, and stellar flares lie along lines of nearly constant B (30 – 150 G), but with loop length varying from 10^8 cm (solar microflare) to 10^{12} cm (stellar superflare). The ability of a single scaling to cover 10^9 orders in energy is the strongest evidence that all flares share the same MHD mechanism.

§7 (Concluding remarks) is summarized in **Figure 47**, a one-page schematic: preflare (emerging flux + shear/converging flows + instability) → current-sheet formation → reconnection → (thermalization producing SXR loop via chromospheric evaporation) + (particle acceleration producing HXR footpoints and loop-top) + (mass ejection producing jets, plasmoids, CMEs). This single flow-chart is the take-away: the flare is a dissipation of coronal current-sheet free energy via reconnection, branching into thermal, particle, and kinetic channels.

**한국어.** §6은 플레어 스케일링 법칙을 항성 플레어에 적용한다. 관측된 항성 플레어는 T = 10^7 – 10^8 K와 E = 10^29 – 10^37 erg에 걸쳐 있다 — 태양보다 훨씬 크다. 보편적 방출량 대 온도 상관 (Feldman 1995; Shibata & Yokoyama 1999, 2002)은 재결합 온도 스케일링 (Eq. 38), 정역학 평형 2nkT = B^2/(8π), EM = n^2 L^3의 결합으로 유도된다 (Eq. 43):

$$EM \simeq 10^{48} (B/50\text{ G})^{-5} (n_0/10^9)^{3/2} (T/10^7 \text{ K})^{17/2} \text{ cm}^{-3}.$$

모든 태양 마이크로플레어, 플레어, 항성 플레어는 거의 일정한 B (30 – 150 G)선 위에 놓이나, 루프 길이는 10^8 cm (태양 마이크로플레어)에서 10^{12} cm (항성 슈퍼플레어)까지 변한다. 하나의 스케일링이 10^9 에너지 차수를 포괄한다는 것은 모든 플레어가 같은 MHD 메커니즘을 공유한다는 가장 강력한 증거이다.

§7 (결론)은 **Figure 47**, 한 페이지 개요도로 요약된다: preflare (flux 방출 + shear/수렴 흐름 + 불안정성) → current sheet 형성 → 재결합 → (채층 증발을 통한 SXR 루프 생성하는 열화) + (HXR 발자국과 루프톱 생성하는 입자 가속) + (jet, 플라즈모이드, CME 생성하는 질량 방출). 이 단일 흐름도가 핵심 요지이다: 플레어는 재결합을 통한 코로나 current sheet 자유 에너지의 소산이며, 열/입자/운동 채널로 분기된다.

---

## 3. Key Takeaways / 핵심 시사점

1. **A flare = explosive dissipation of coronal current / 플레어 = 코로나 전류의 폭발적 소산** — The flare engine is the rapid conversion of field-aligned current (free magnetic energy) into heat + kinetic energy + particles + radiation through magnetic reconnection in a thin current sheet. Everything else is a consequence. 플레어 엔진은 얇은 current sheet에서의 자기 재결합을 통해 자기장 정렬 전류 (자유 자기에너지)를 열 + 운동 에너지 + 입자 + 복사로 급속히 변환하는 것이다. 나머지는 모두 결과이다.

2. **The Sweet-Parker vs Petschek tension defines the reconnection puzzle / Sweet-Parker 대 Petschek 긴장이 재결합 퍼즐을 정의** — With R_m ~ 10^14 in the corona, Sweet-Parker gives t_rec ~ 10^9 s (too slow by six orders), while Petschek gives t_rec ~ 10^3 s (right order) — but steady Petschek requires locally enhanced resistivity. 코로나 R_m ~ 10^14에서 Sweet-Parker는 t_rec ~ 10^9 s (6차수 느림), Petschek는 t_rec ~ 10^3 s (맞는 차수)를 주지만 — 정상 Petschek는 국소적으로 향상된 저항도를 필요로 한다.

3. **Plasmoid instability resolves the puzzle / 플라즈모이드 불안정성이 퍼즐을 해결** — Above a critical Lundquist number S_c ~ 10^4, Sweet-Parker sheets are violently unstable, spontaneously producing a cascade of plasmoids and a fractal current sheet whose effective rate is independent of eta. This turns the 40-year Sweet-Parker/Petschek tension into a non-problem. S_c ~ 10^4 이상에서 Sweet-Parker sheet는 격렬히 불안정하여 플라즈모이드 연쇄와 유효 속도가 eta에 무관한 프랙털 current sheet를 자발적으로 생성한다. 이는 40년된 Sweet-Parker/Petschek 긴장을 무(無)문제로 만든다.

4. **CSHKP is the universal 2D skeleton / CSHKP는 보편적 2D 골격** — The Carmichael-Sturrock-Hirayama-Kopp-Pneuman picture (rising plasmoid, reconnection sheet below, cusp SXR loop, H-alpha ribbons) captures the geometry of LDE flares, impulsive flares, X-ray jets, giant arcades — and with appropriate scale changes, even stellar superflares. CSHKP 그림 (상승 플라즈모이드, 아래 재결합 sheet, cusp SXR 루프, H-alpha ribbons)은 LDE 플레어, 임펄시브 플레어, X-선 제트, 거대 아케이드의 기하를 포착하며 — 적절한 규모 변환으로 항성 슈퍼플레어까지도.

5. **Flux emergence + shear is the universal trigger pipeline / 자속 방출 + shear가 보편적 trigger 파이프라인** — Free energy is pumped in by buoyant rise of twisted sub-photospheric tubes; shear or converging photospheric flows concentrate the current; eventually an instability (tearing, kink, torus, loss-of-equilibrium) or a topological change (breakout, emerging-flux trigger) forms a thin current sheet and reconnection takes over. 자유 에너지는 부력으로 상승하는 twisted sub-광구 튜브에 의해 주입된다; shear 또는 수렴 광구 흐름이 전류를 집중시킨다; 결국 불안정성 (tearing, kink, torus, loss-of-equilibrium) 또는 위상 변화 (breakout, emerging-flux trigger)가 얇은 current sheet를 형성하고 재결합이 맡는다.

6. **Chromospheric evaporation sets the SXR loop temperature / 채층 증발이 SXR 루프 온도를 설정** — The scaling T_loop ~ (B^2 v_A L/(4 pi kappa_0))^(2/7) ~ 4 × 10^7 (B/100 G)^(6/7) (L/10^9 cm)^(2/7) K links reconnection heating rate, conduction, and evaporation, and quantitatively predicts T ~ 10^7 K observed in post-flare loops. T_loop ~ (B^2 v_A L/(4π κ_0))^(2/7) 스케일링은 재결합 가열률, 전도, 증발을 연결하며, post-flare 루프에서 관측되는 T ~ 10^7 K를 정량 예측한다.

7. **The universal EM-T relation unifies 9 orders of energy / 보편적 EM-T 관계가 9차수 에너지를 통합** — EM ~ 10^48 B^(-5) n^(3/2) T^(17/2) cm^(-3) along 30 – 150 G lines explains solar microflares, solar flares, and stellar superflares as the same phenomenon on different L. EM ~ 10^48 B^(-5) n^(3/2) T^(17/2) cm^(-3)의 30 – 150 G 선은 태양 마이크로플레어, 태양 플레어, 항성 슈퍼플레어를 다른 L에서의 같은 현상으로 설명한다.

8. **Reconnection converts energy AND changes topology / 재결합은 에너지를 변환하고 위상을 바꾼다** — Beyond its rate, reconnection is topologically essential: it reconnects field lines, producing closed loops from open ones (or vice versa), enabling CMEs (which carry off magnetic helicity) and plasmoid ejections. Without topology change, the corona would accumulate helicity without bound. 속도를 넘어, 재결합은 위상적으로 본질적이다: 자기력선을 재결합하여 열린 선에서 닫힌 루프 (또는 그 반대)를 만들어 CME (자기 helicity 운반)와 플라즈모이드 방출을 가능하게 한다. 위상 변화 없이는 코로나가 helicity를 한없이 축적할 것이다.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Magnetic energy budget / 자기 에너지 예산

$$E_\text{mag} = \frac{B^2}{8\pi} L^3 \simeq 10^{33}\left(\frac{B}{10^3 \text{G}}\right)^2\!\left(\frac{L}{3\times 10^9 \text{cm}}\right)^3 \text{erg}.$$

**Terms / 항목.** *B* — active-region photospheric field (kilogauss in a sunspot); *L* — linear size of stored-energy volume. Only the free energy *above the potential minimum* is releasable — typically 10 – 30 %. 저장 부피의 선형 크기 L, 활동 영역 광구 자기장 B; 포텐셜 최소값 위의 자유 에너지만 방출 가능 — 일반적으로 10 – 30 %.

### 4.2 Spitzer diffusion time / Spitzer 확산 시간

$$t_\text{dif} \simeq \frac{L^2}{\eta} \simeq 10^{14}\left(\frac{L}{10^9\text{cm}}\right)^2\!\left(\frac{T}{10^6\text{K}}\right)^{3/2} \text{s}, \qquad \eta \simeq 10^4 (T/10^6\text{K})^{-3/2} \text{cm}^2 \text{s}^{-1}.$$

**Terms.** *eta* — Spitzer magnetic diffusivity from electron-ion Coulomb collisions; *L* — current-sheet scale. This 10^14 s is *hopelessly* longer than any flare; the MHD machinery must evade it. Spitzer 자기 확산도 eta는 전자-이온 Coulomb 충돌에서; current sheet 규모 L. 이 10^14 s는 어떤 플레어보다 *절망적으로* 길다; MHD 기계가 이를 회피해야 한다.

### 4.3 Sweet-Parker reconnection / Sweet-Parker 재결합

$$M_A = v_i/v_A \simeq R_m^{-1/2}, \qquad R_m = v_A L/\eta, \qquad t_\text{rec} = t_A \sqrt{R_m}.$$

**Terms.** *M_A* — inflow Alfvén Mach number; *R_m* — Lundquist number; *t_A = L/v_A* — Alfvén transit time. Derivation idea: mass conservation rho v_i L = rho v_A w, induction v_i B_i = eta B_i / w → w = sqrt(eta L / v_A) → v_i = v_A sqrt(eta/(v_A L)) = v_A R_m^(-1/2). 질량 보존 rho v_i L = rho v_A w, 유도 v_i B_i = eta B_i / w → w = sqrt(eta L / v_A) → v_i = v_A R_m^(-1/2).

### 4.4 Petschek reconnection / Petschek 재결합

$$M_A \simeq \frac{\pi}{8 \ln R_m} \sim 0.01\text{--}0.1 .$$

**Terms.** The *logarithmic* rather than power-law dependence on R_m is the source of Petschek's speed. Physically: most of the inflow is diverted sideways through two pairs of slow-mode shocks standing away from a tiny central diffusion patch of size delta << L. R_m에 대한 *로그* 의존성 (멱법칙이 아님)이 Petschek 속도의 원천이다. 물리적으로: 유입의 대부분이 크기 delta << L인 작은 중심 확산 패치에서 떨어진 두 쌍의 느린모드 충격파를 통해 옆으로 굴절된다.

### 4.5 Coronal Alfvén velocity / 코로나 Alfvén 속도

$$v_A = \frac{B}{\sqrt{4\pi\rho}} \sim 1000\left(\frac{B}{50\text{ G}}\right)\!\left(\frac{n}{10^{10}\text{cm}^{-3}}\right)^{-1/2} \text{km/s}.$$

**Example.** For a typical active region (B = 100 G, n = 10^{10} cm^(-3)): v_A ~ 2000 km/s. Reconnection inflow v_i = M_A v_A for Petschek M_A = 0.05 gives v_i = 100 km/s — consistent with direct observations by SOHO/Hinode (Yokoyama et al. 2001). 전형 활동 영역에서 Petschek M_A = 0.05로 v_i = 100 km/s — SOHO/Hinode의 직접 관측과 일치.

### 4.6 Tearing-mode growth rate / Tearing 모드 성장률

$$\gamma \sim \tau_A^{-2/5}\,\tau_R^{-3/5} \sim \tau_A^{-1} R_m^{-3/5}.$$

**Terms.** *tau_A = delta/v_A* — Alfvén time across sheet half-width delta; *tau_R = delta^2/eta* — resistive time across delta. The 2/5 – 3/5 FKR scaling is intermediate between ideal (gamma ~ 1/tau_A) and pure resistive (gamma ~ 1/tau_R) — tearing exploits both flow and diffusion. tau_A는 sheet 반폭 delta를 가로지르는 Alfvén 시간, tau_R는 delta를 가로지르는 저항 시간; 2/5 – 3/5 FKR 스케일링은 이상 (gamma ~ 1/tau_A)과 순수 저항 (gamma ~ 1/tau_R) 사이의 중간이다.

### 4.7 Plasmoid instability criterion / 플라즈모이드 불안정성 임계값

$$S > S_c \simeq 10^4 \quad \Rightarrow \quad \text{Sweet-Parker sheet disintegrates.}$$

**Terms.** *S* = Lundquist number computed on the Sweet-Parker width delta_SP = L/sqrt(R_m). The critical S_c ~ 10^4 was found numerically by Shibata & Tanuma (2001) and analytically by Loureiro et al. (2007). Above S_c, the number of plasmoids grows as N ~ S^(3/8) and the effective reconnection rate saturates at M_A ~ 10^(-2) independent of S. S는 Sweet-Parker 폭 delta_SP = L/sqrt(R_m)에서 계산된 Lundquist 수. 임계값 S_c ~ 10^4은 Shibata & Tanuma (2001) 수치, Loureiro et al. (2007) 해석으로 발견. S_c 위에서 플라즈모이드 수는 N ~ S^(3/8)로 증가하고 유효 재결합 속도는 M_A ~ 10^(-2)로 S에 무관하게 포화한다.

### 4.8 Reconnection energy-release rate / 재결합 에너지 해방률

$$\frac{dE}{dt} \simeq \frac{L^2 B_i^2 v_i}{4\pi} \sim 4\times 10^{28}\!\left(\frac{v_i}{100\text{km/s}}\right)\!\left(\frac{B_i}{100\text{G}}\right)^2\!\left(\frac{L}{2\times 10^9\text{cm}}\right)^2 \text{erg/s}.$$

**Concrete numerical example.** A mid-size impulsive flare has L = 2 × 10^9 cm, B = 100 G, n = 10^{10} cm^(-3). Then v_A = 2000 km/s; Petschek v_i = 0.05 v_A = 100 km/s; dE/dt = 4 × 10^28 erg/s; over a 10^3 s impulsive phase this gives E ~ 4 × 10^31 erg — right in the observed M/X-class range (Masuda 1994). 중간 규모 임펄시브 플레어에서 L = 2 × 10^9 cm, B = 100 G, n = 10^{10} cm^(-3); v_A = 2000 km/s, Petschek v_i = 100 km/s, dE/dt = 4 × 10^28 erg/s, 10^3 s 임펄시브 단계 동안 E ~ 4 × 10^31 erg — 관측된 M/X-class 범위.

### 4.9 Yokoyama-Shibata loop-top scaling / Yokoyama-Shibata 루프톱 스케일링

$$T_\text{loop} \simeq \left(\frac{B^2 v_A L}{4\pi \kappa_0}\right)^{2/7} \simeq 4\times 10^7\!\left(\frac{B}{100\text{G}}\right)^{6/7}\!\left(\frac{n}{10^{10}\text{cm}^{-3}}\right)^{-1/7}\!\left(\frac{L}{10^9\text{cm}}\right)^{2/7} \text{K}.$$

**Terms.** kappa_0 = 10^(-6) cgs (Spitzer conductivity), Q = B^2 v_A/(4 pi L) = reconnection volumetric heating. Derivation: conduction kappa_0 T^(7/2)/L^2 = Q. This scaling is *the* predictive triumph of MHD flare theory; it quantitatively matches SXR loop temperatures without free parameters. kappa_0 = 10^(-6) cgs (Spitzer 전도도), Q = 재결합 부피 가열. 유도: 전도 kappa_0 T^(7/2)/L^2 = Q. 이 스케일링은 MHD 플레어 이론의 *예측적* 승리이다; 자유 매개변수 없이 SXR 루프 온도와 정량적으로 일치한다.

### 4.10 Numerical comparison Sweet-Parker vs Petschek / 수치 비교 Sweet-Parker 대 Petschek

**Worked example at coronal conditions.** Take B = 100 G, n = 10^{10} cm^(-3), L = 10^4 km = 10^9 cm, T = 10^7 K.

| Quantity / 양 | Formula / 공식 | Value / 값 |
|---|---|---|
| v_A | B/sqrt(4 pi rho) | 2.2 × 10^3 km/s |
| eta (Spitzer) | 10^4 (T/10^6)^(-3/2) | 0.3 cm^2/s |
| R_m = v_A L / eta | | 7 × 10^{14} |
| t_A = L/v_A | | 4.5 s |
| M_A (Sweet-Parker) | R_m^(-1/2) | 3.8 × 10^(-8) |
| t_rec (Sweet-Parker) | t_A/M_A | 1.2 × 10^8 s = 3.8 yr |
| M_A (Petschek) | pi/(8 ln R_m) | 0.012 |
| t_rec (Petschek) | t_A/M_A | 370 s = 6 min |

**Conclusion / 결론.** Sweet-Parker gives a reconnection time comparable to the solar cycle, while Petschek's 6-minute timescale matches the observed impulsive-phase duration of a typical flare. Plasmoid instability makes Petschek-like rates automatic. Sweet-Parker는 태양 주기와 비슷한 재결합 시간을 주고, Petschek의 6분 시간 규모는 전형적 플레어의 관측된 임펄시브 단계 지속 시간과 일치한다. 플라즈모이드 불안정성이 Petschek 유사 속도를 자동으로 만든다.

### 4.11 Harris current-sheet equilibrium / Harris current sheet 평형

$$B_x(z) = B_0 \tanh(z/a), \qquad j_y(z) = \frac{c B_0}{4\pi a}\text{sech}^2(z/a), \qquad n(z) = n_0 \text{sech}^2(z/a) + n_\infty,$$

with total pressure B_0^2/(8 pi) + n k T held constant across the sheet. **Terms.** *a* — sheet half-thickness, *B_0* — asymptotic field. This is the canonical equilibrium perturbed by tearing instability. a는 sheet 반두께, B_0은 점근 자기장; 이것이 tearing 불안정성이 교란하는 정준 평형이다.

### 4.12 Chromospheric evaporation speed / 채층 증발 속도

$$v_\text{evap} \simeq c_s \sim 500\left(\frac{T}{10^7\text{K}}\right)^{1/2} \text{km/s}, \qquad \kappa_0 T^{7/2}/L^2 \sim \frac{5}{2} p v_\text{evap}.$$

**Terms.** *c_s* — isothermal sound speed at the chromospheric top; *kappa_0 T^(7/2)/L^2* — downward heat flux from super-hot loop-top region; *p* — chromospheric pressure; *5/2 p v* — enthalpy flux of the evaporation flow. This equation expresses the energy balance that connects coronal heating (from reconnection) to the observed SXR loop filling. c_s는 채층 최상부의 등온 음속; kappa_0 T^(7/2)/L^2는 초고온 루프톱에서의 하향 열 플럭스; p는 채층 압력; 5/2 p v는 증발 흐름의 엔탈피 플럭스. 이 식은 코로나 가열 (재결합으로부터)을 관측된 SXR 루프 채움과 연결하는 에너지 평형을 표현한다.

### 4.13 Convective electric field for particle acceleration / 입자 가속을 위한 대류 전기장

$$E = (v_i/c) B_i \sim 3\times 10^3\!\left(\frac{M_A}{0.1}\right)\!\left(\frac{B}{100\text{G}}\right)^2\!\left(\frac{n_\text{jet}}{10^{10}\text{cm}^{-3}}\right)^{-1/2} \text{V/m}.$$

**Terms.** Inflow velocity v_i crossed with inflow field B_i gives the motional electric field (E = -v × B / c in Gaussian units). *M_A* = v_i/v_A; a Petschek M_A ~ 0.1 with B = 100 G gives E of order kV/m — sufficient to accelerate protons to > GeV energies across a meter-scale diffusion region. 유입 속도 v_i와 유입 자기장 B_i의 외적은 운동 전기장을 준다 (Gaussian 단위 E = -v × B / c). M_A = v_i/v_A; Petschek M_A ~ 0.1와 B = 100 G는 kV/m 크기의 E를 준다 — 미터 규모 확산 영역에서 양성자를 > GeV 에너지로 가속하기에 충분.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1859 ────────── Carrington / Hodgson — first white-light flare
   │
1946-49 ─────── Giovanelli / Hoyle — neutral-point dissipation hypothesis
   │
1957-58 ─────── Sweet / Parker — diffusive reconnection (too slow)
   │
1963 ────────── Furth-Killeen-Rosenbluth — tearing-mode instability
   │
1964 ────────── Petschek — fast reconnection via slow shocks
   │
1964-76 ─────── Carmichael / Sturrock / Hirayama / Kopp-Pneuman — CSHKP model
   │
1973-74 ─────── Skylab — SXR corona, cusp structures
   │
1991-2001 ───── Yohkoh — SXR cusps, Masuda HXR loop-top
   │                     (revolutionized flare physics)
   │
1995 ────────── SOHO — EUV dimming, CME connections
   │
2001 ────────── Shibata-Tanuma — fractal current sheet,
   │              Loureiro / Bhattacharjee will later prove this analytically
   │
2002 ────────── RHESSI — HXR imaging spectroscopy
   │
2007-09 ─────── Loureiro / Bhattacharjee — plasmoid instability S_c ~ 10^4
   │
2011 ────────── ★ Shibata & Magara LRSP review (this paper)
   │
2015-present ── IRIS, Hinode, Parker Solar Probe, MMS — kinetic/in-situ
   │
2020-present ── Kepler/TESS superflares on Sun-like stars
```

**English.** Shibata & Magara's review sits at the exact moment when plasmoid instability was being established as the resolution of the Sweet-Parker-Petschek tension. Before 2007, fast reconnection required somewhat ad-hoc "locally enhanced resistivity"; by 2011, it was becoming clear that any large-enough Sweet-Parker sheet would spontaneously fragment into plasmoids, producing an effective rate near Petschek. The paper therefore represents the *synthesis* of the classical CSHKP framework with the new plasmoid-dominated fast reconnection picture. It has become one of the standard graduate references on solar flare physics.

**한국어.** Shibata & Magara의 리뷰는 플라즈모이드 불안정성이 Sweet-Parker-Petschek 긴장의 해결로 확립되던 바로 그 순간에 위치한다. 2007년 이전에 빠른 재결합은 다소 임시방편적인 "국소적으로 향상된 저항도"를 필요로 했다; 2011년까지 충분히 큰 Sweet-Parker sheet가 자발적으로 플라즈모이드로 조각나서 Petschek에 가까운 유효 속도를 생성한다는 것이 분명해졌다. 따라서 이 논문은 고전적 CSHKP 틀과 새로운 플라즈모이드 지배 빠른 재결합 그림의 *종합*을 나타낸다. 태양 플레어 물리학의 표준 대학원 참고문헌 중 하나가 되었다.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Parker (1957), Sweet (1958) | Original diffusive reconnection model | Provides the slow-rate baseline (M_A ~ R_m^(-1/2)) against which all faster models are measured / 확산 재결합 원본 모델 — 느린 기준 (M_A ~ R_m^(-1/2)), 더 빠른 모든 모델이 이에 비교된다 |
| Petschek (1964) | Fast reconnection with slow-mode shocks | The M_A ~ pi/(8 ln R_m) rate, directly reproduced in Eq. 21 of Shibata & Magara. Foundation of the "CSHKP engine" / 느린모드 충격파에 의한 빠른 재결합 — Eq. 21에서 직접 재현됨. "CSHKP 엔진"의 기초 |
| Furth, Killeen, Rosenbluth (1963) | Tearing-mode instability theory | Foundational for §4.1.5 fractal current sheet; provides the gamma ~ tau_A^(-2/5) tau_R^(-3/5) scaling / tearing 모드 불안정성 이론 — §4.1.5 프랙털 current sheet의 기초; gamma ~ tau_A^(-2/5) tau_R^(-3/5) 스케일링 |
| Carmichael (1964), Sturrock (1966), Hirayama (1974), Kopp-Pneuman (1976) | Four CSHKP progenitor papers | The geometric/phenomenological skeleton (Figure 2) that Shibata & Magara unpack and extend throughout §2 and §4 / 네 CSHKP 전구체 논문 — Shibata & Magara가 §2와 §4에서 풀어내고 확장하는 기하/현상론적 골격 (Figure 2) |
| Masuda et al. (1994) | Yohkoh HXR loop-top source | The key observation that HXR emission comes from above the SXR loop — direct evidence for reconnection ABOVE the loop (not inside) / Yohkoh HXR 루프톱 source — HXR 방출이 SXR 루프 위에서 온다는 핵심 관측; 루프 *위* (내부가 아닌) 재결합의 직접 증거 |
| Shibata et al. (1995) | Plasmoid-ejection flare model | This paper's Figure 3b is taken from Shibata et al. 1995; introduces the plasmoid + reconnection-jet geometry / 플라즈모이드 방출 플레어 모델 — 본 논문 Figure 3b; 플라즈모이드 + reconnection jet 기하 도입 |
| Shibata & Tanuma (2001) | Fractal current sheet, plasmoid-induced reconnection | Numerical demonstration that Sweet-Parker sheets cascade into plasmoid hierarchies; foundation of §4.1.5 – 6 / 프랙털 current sheet, 플라즈모이드 유도 재결합 — Sweet-Parker sheet가 플라즈모이드 계층으로 연쇄한다는 수치 시연; §4.1.5 – 6의 기초 |
| Yokoyama & Shibata (1998) | MHD simulation with thermal conduction | Derives the T_loop ~ (B^2 v_A L / kappa_0)^(2/7) scaling law (Eq. 38) — quantitative prediction of flare-loop temperature / 열전도 포함 MHD 시뮬레이션 — T_loop ~ (B^2 v_A L / kappa_0)^(2/7) 스케일링 법칙 (Eq. 38) 유도; 플레어 루프 온도 정량 예측 |
| Shibata & Yokoyama (1999, 2002) | Universal EM-T scaling for solar and stellar flares | Equation 43 unifies microflares to stellar superflares on a single line; direct consequence of Eq. 38 / 태양·항성 플레어의 보편 EM-T 스케일링 — Eq. 43이 마이크로플레어에서 항성 슈퍼플레어까지 단일 선으로 통합; Eq. 38의 직접 결과 |
| Antiochos et al. (1999) | Magnetic breakout model | A key multi-polar eruption mechanism discussed in §4.3.2; alternative to single-domain instabilities / 자기 breakout 모델 — §4.3.2에서 논의된 핵심 다극 분출 메커니즘; 단일 영역 불안정성의 대안 |

---

## 7. References / 참고문헌

- Shibata, K., & Magara, T., "Solar Flares: Magnetohydrodynamic Processes", *Living Reviews in Solar Physics*, **8**, 6 (2011). DOI: [10.12942/lrsp-2011-6](https://doi.org/10.12942/lrsp-2011-6)
- Sweet, P.A., "The Neutral Point Theory of Solar Flares", *IAU Symposium*, **6**, 123 (1958).
- Parker, E.N., "Sweet's Mechanism for Merging Magnetic Fields in Conducting Fluids", *JGR*, **62**, 509 (1957).
- Petschek, H.E., "Magnetic Field Annihilation", *NASA Special Publication*, **SP-50**, 425 (1964).
- Furth, H.P., Killeen, J., & Rosenbluth, M.N., "Finite-Resistivity Instabilities of a Sheet Pinch", *Phys. Fluids*, **6**, 459 (1963).
- Carmichael, H., "A Process for Flares", in *AAS-NASA Symposium on the Physics of Solar Flares* (1964).
- Sturrock, P.A., "Model of the High-Energy Phase of Solar Flares", *Nature*, **211**, 695 (1966).
- Hirayama, T., "Theoretical Model of Flares and Prominences", *Solar Phys.*, **34**, 323 (1974).
- Kopp, R.A., & Pneuman, G.W., "Magnetic Reconnection in the Corona and the Loop Prominence Phenomenon", *Solar Phys.*, **50**, 85 (1976).
- Masuda, S., Kosugi, T., Hara, H., Tsuneta, S., & Ogawara, Y., "A Loop-Top Hard X-Ray Source in a Compact Solar Flare as Evidence for Magnetic Reconnection", *Nature*, **371**, 495 (1994).
- Shibata, K., et al., "Hot-Plasma Ejections Associated with Compact-Loop Solar Flares", *ApJ*, **451**, L83 (1995).
- Shibata, K., & Tanuma, S., "Plasmoid-Induced Reconnection and Fractal Reconnection", *Earth Planets Space*, **53**, 473 (2001).
- Yokoyama, T., & Shibata, K., "Magnetic Reconnection as the Origin of X-ray Jets and Hα Surges on the Sun", *Nature*, **375**, 42 (1995).
- Yokoyama, T., & Shibata, K., "A Two-Dimensional MHD Simulation of the Solar Flare", *ApJ*, **494**, L113 (1998).
- Shibata, K., & Yokoyama, T., "Origin of the Universal Correlation Between the Flare Temperature and Emission Measure", *ApJ*, **526**, L49 (1999).
- Antiochos, S.K., DeVore, C.R., & Klimchuk, J.A., "A Model for Solar Coronal Mass Ejections", *ApJ*, **510**, 485 (1999).
- Loureiro, N.F., Schekochihin, A.A., & Cowley, S.C., "Instability of current sheets and formation of plasmoid chains", *Phys. Plasmas*, **14**, 100703 (2007).
- Bhattacharjee, A., Huang, Y.-M., Yang, H., & Rogers, B., "Fast reconnection in high-Lundquist-number plasmas", *Phys. Plasmas*, **16**, 112102 (2009).
- Priest, E.R., & Forbes, T.G., *Magnetic Reconnection: MHD Theory and Applications* (Cambridge University Press, 2000).
- Aschwanden, M.J., *Physics of the Solar Corona* (Springer, 2004).
