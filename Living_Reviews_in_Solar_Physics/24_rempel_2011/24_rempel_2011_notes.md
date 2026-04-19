---
title: "Sunspot Modeling: From Simplified Models to Radiative MHD Simulations"
authors: Matthias Rempel, Rolf Schlichenmaier
year: 2011
journal: "Living Reviews in Solar Physics"
doi: "10.12942/lrsp-2011-3"
topic: Living_Reviews_in_Solar_Physics
tags: [sunspot, MHD, magneto-convection, penumbra, umbra, Evershed, flux-emergence, MURaM, helioseismology]
status: completed
date_started: 2026-04-19
date_completed: 2026-04-19
---

# 24. Sunspot Modeling: From Simplified Models to Radiative MHD Simulations / 흑점 모델링 — 단순 모델에서 복사 MHD 시뮬레이션까지

---

## 1. Core Contribution / 핵심 기여

### 한국어
이 리뷰는 **흑점 모델링의 한 세기 역사와 2005-2010년 radiative MHD 시뮬레이션 돌파구**를 총체적으로 정리한 논문이다. 저자 Rempel은 MURaM 코드로 세계 최초의 3D 전체 흑점 시뮬레이션을 수행한 당사자이고, Schlichenmaier는 penumbra 이론 및 관측 분야의 권위자다. 두 사람은 (1) 1940년대 Biermann의 **"자기장이 대류를 억제한다"** 는 이론에서 시작된 흑점 에너지 수송 문제, (2) 1990년대 Jahn & Schmidt의 **tripartite 자기정역학 모델**(umbra + penumbra + quiet Sun을 current sheet로 분리), (3) penumbra 미세구조 설명을 위한 **field-free gap 모델**(Parker, Spruit & Scharmer)과 **moving flux tube 모델**(Schlichenmaier)의 성공과 한계, (4) 2006년 Schüssler & Vögler의 umbral dot 시뮬레이션, 2007년 Heinemann et al.의 penumbra slab, 2009년 Rempel et al.의 **완전한 3D radiative MHD 흑점 시뮬레이션** — 이 세 논문이 열어젖힌 "시뮬레이션 시대"를 다룬다. 핵심 메시지는 **"흑점 미세구조의 공통 원리는 경사진 자기장 배경에서 발생하는 overturning magneto-convection이다"** — umbral dot(거의 수직 자기장의 upflow plume), penumbral filament(경사진 자기장의 elongated convection cell), Evershed flow(penumbra 깊은 경계층의 Lorentz-force 구동 수평 outflow)가 모두 **단일 magneto-convective 프로세스의 다른 국면**임을 자기일관적으로 설명한다. 또한 (5) flux emergence(100 kG 초기 자기장 → 태양 표면에서 active region 형성), (6) penumbra 형성 및 moat flow, (7) helioseismology가 드러낸 ~2 Mm 깊이의 **얕은(shallow) 파속도 교란** 같은 최신 결과들을 종합한다. 결론은 분명하다: **"흑점은 정적 자기 구조가 아니라, 난류 magneto-convection이 작은 스케일에서 만들어내는 동적 평형의 대규모 표현이다."**

### English
This review surveys the century-long history of sunspot modeling and the **2005–2010 breakthrough in 3D radiative MHD simulations**. Rempel is the architect of MURaM's full-sunspot simulations; Schlichenmaier is a leading authority on penumbra observations and theory. Together they cover (1) the energy-transport puzzle seeded by Biermann's 1941 **"suppressed convection"** theory; (2) Jahn & Schmidt's (1994) **tripartite magnetostatic model** separating umbra, penumbra, and quiet Sun via two current sheets; (3) the successes and limits of idealized penumbral models — the **field-free gap model** (Parker; Spruit & Scharmer) and the **moving flux tube model** (Schlichenmaier); (4) the three papers that opened the "simulation era": Schüssler & Vögler (2006) for umbral dots, Heinemann et al. (2007) for penumbra slabs, and **Rempel et al. (2009a, b) for complete 3D radiative MHD sunspot simulations**. The central message is that **all sunspot fine structure arises from overturning magneto-convection in an inclined magnetic field** — umbral dots (upflow plumes in near-vertical field), penumbral filaments (elongated convection cells in inclined field), and the Evershed flow (Lorentz-force-driven horizontal outflow in a deep boundary layer) are different manifestations of a **single magneto-convective process**. The review also synthesizes (5) flux emergence (~100 kG initial field giving active regions at the surface), (6) penumbra formation and moat flow, and (7) helioseismic evidence of a **shallow (~2 Mm)** wave-speed perturbation. The definitive conclusion: **"A sunspot is not a static magnetic construct but the large-scale manifestation of a dynamic equilibrium produced by small-scale turbulent magneto-convection."**

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction (§1, p. 5)

**한국어**: 태양 자기장은 대류층에서 생성되어 부분적으로 외기층(광구, 채층, 코로나)으로 수송된다. 흑점은 그중 가장 두드러진 광구 현상 — 수 주 수명, 30 granule 크기, **$B = 1000$–$3000$ G, 자속 ~$10^{22}$ Mx**. 이 리뷰의 관점: **(a) global structure, (b) fine structure (dynamic), (c) formation/evolution, (d) helioseismology** 네 각도에서 흑점을 조명. 이전 리뷰(Solanki 2003, Thomas & Weiss 2004 등)에 비해 **2006년 이후 radiative MHD 시뮬레이션 혁명**이 이 리뷰의 신 메시지.

**English**: Solar magnetic fields are generated in the convection zone and partly transported into the outer layers. Sunspots are the most prominent photospheric manifestation — lifetimes of weeks, sizes of ~30 granules, **$B = 1000$–$3000$ G, total flux ~$10^{22}$ Mx**. The review examines sunspots from four perspectives: **(a) global structure, (b) dynamic fine structure, (c) formation/evolution, (d) helioseismic constraints**. The novel content over prior reviews (Solanki 2003; Thomas & Weiss 2004, 2008) is the **post-2006 radiative-MHD simulation revolution**.

### Part II: Global Sunspot Structure (§2, pp. 6–11)

#### §2.1 Time scales / 시간 스케일
- Dynamical time scale (Alfvén crossing): ~1 hr for $D=30,000$ km, $v_A\sim60$ km/s in umbra photosphere
- Lifetime: weeks → 500× dynamical time
- **핵심 퍼즐**: dynamic fine structure (umbral dots, filaments) ↔ globally stable structure. "작은 스케일 동역학이 큰 스케일 안정성을 만든다."

#### §2.2 Sunspot darkness and energy transport / 흑점 어둠과 에너지 수송
네 이론이 순차적으로 제시됨:

1. **Hale (1908a)의 "tornado" 이론** — 오답. Evershed (1909)가 수평 outflow 발견으로 번복.
2. **Biermann (1941), Alfvén (1942) — suppressed convection**: 자기장이 대류를 억제해 $F_{\text{umbra}} = 0.23\,F_{\text{QS}}$로 감소. $T_u\sim4000$ K vs $T_{\text{QS}}\sim 6060$ K → $(T_u/T_{\text{QS}})^4 \approx 19\%$ (Eq., Stefan-Boltzmann).
3. **Hoyle (1949), Jahn (1992) — heat flux dilution by funneling**: $\beta = 8\pi p/B^2$ 가 깊이에 따라 증가하므로, flux 보존을 위해 자속관 단면적이 깊이에 따라 감소. 따라서 umbra 지름이 깊이에 따라 **2배 증가**(factor 4 in area) → 상승하는 열 플럭스가 희석됨.
4. **Deinzer (1965), Meyer et al. (1974) — modified convection**: 완전 억제는 불가능 (umbra가 여전히 4000 K). Magnetoconvection이 overstable oscillation으로 에너지 수송. **2000 km** 깊이 위에서는 $\eta < \kappa$로 overstable.

**$ \beta $ 심도 의존성**:
$$\beta = \frac{8\pi p}{B^2}$$
표면($\tau=1$): $\beta\sim1$; $z=-2$ Mm: $\beta\sim 10$–$100$; 코로나: $\beta\ll1$.

#### §2.2.4 No bright ring 문제
Funnel이 열을 차단한다면 흑점 주변에 **bright ring**이 있어야 하는데, 관측은 0.1–1%만 발견. Spruit (1977)이 해결: 대류층의 **큰 열전도도**가 온도 교란을 확산시킴 ("kitchen analogy": 구리판 위 절연체 주변 온도 상승 없음). 현대 MHD 시뮬레이션도 bright ring 없음 (Rempel 2011c).

#### §2.3 Subsurface morphology / 서브포토스피어 구조
- **Monolithic model**: 단일 자속관. 장점 — 장기 안정성, mean-field 기술 용이. 단점 — umbral dots, bright filaments 설명이 까다로움.
- **Spaghetti/cluster/jelly-fish model** (Parker 1979, Spruit 1981): 많은 작은 자속관 다발. Field-free 기체가 상승해 umbral dots 생성. **문제**: 자속관을 묶어두는 힘은 무엇인가? 흑점 중심으로 갈수록 B가 증가하는 관측은 왜?
- **Meyer et al. (1974)**: 2000 km 이상 깊이에서 $\eta>\kappa$이므로 magnetoconvection이 플라즈마를 섞는다 → **monolithic 선호**. 저자들: "fine structure 존재 자체는 spaghetti의 증거가 아니다."

#### §2.4 Stability of monolithic models / monolithic 모델의 안정성
**Interchange (fluting) instability**: 자속관 경계에서 수평 자기장이 휘어지려는 경향. Jahn & Schmidt (1994) 분석: 상층 5 Mm에서는 magnetopause 경사가 작고 buoyancy가 안정화; 깊은 층(>5 Mm)에서는 불안정. Schüssler & Rempel (2005)는 깊은 층에서 $B$가 사실상 감소해 "disconnection"이 일어난다고 제안 → **표면 ~5 Mm 아래는 monolithic이 아닐 수 있다**. 떠 있는 상부만 단일 자속관.

### Part III: Sunspot Fine Structure (§3, pp. 12–33) — 핵심 섹션

#### §3.1 Umbral dots / 암부 점
- 크기 0.5" (~350 km), 수 분 수명
- Danielson (1964) 발견
- 관측: **자기장이 주변 umbra보다 약함**, 수백 m/s upflow (Socas-Navarro 2004, Rimmele 2004, 2008, Bharti 2007)
- 해석: overturning magneto-convection의 **upflow plume**. 아래(주변) vs 위(관측) 의 field 감소 정도 — 깊은 곳에서는 거의 0, line-formation region에서는 덜 감소.

#### §3.2 Penumbral observations / 반암부 관측 — 가장 조밀한 섹션
**Intensity pattern (§3.2.1)**:
- Bright/dark filaments, penumbral grains
- Inner penumbra에서 0.1" 스케일 미세구조 (Scharmer et al. 2002), mid/outer에서 0.35"
- **Bright filament은 dark core를 가짐** — 핵심 발견 (Scharmer 2002)
- Inclined dark stripes ("striations", Ichimoto 2007b) — fluting instability의 corrugation?

**Evershed flow, uncombed field, NCP (§3.2.2)**: 가장 복잡한 섹션
- **Disk-center spot** (Fig. 3 좌): LOS 속도는 수직 유동 → upflow patches (blue, 1.5 km/s) in inner/mid penumbra, downflow patches (red, >5 km/s) in outer penumbra. "opposite polarity" patches가 downflow와 co-spatial.
- **$\theta=47°$ spot** (Fig. 3 우): horizontal (Evershed) flow 지배 → center-side blueshift, limb-side redshift, clipped ±3.5 km/s.
- **Flow field**: filamentary, azimuthal 평균 수평 + 약간의 upflow(inner)/downflow(outer). Dark core 내에 강한 Evershed. 관측 속도 3–8 km/s (평균 6.5 km/s), 국소 피크 >10 km/s.
- **Magnetic field**: Stokes V가 3개 lobe → 한 성분으로 fit 불가능. **"Uncombed"** = interlocking of 두 성분: (1) 덜 경사진 배경장 (30°→60°), (2) 거의 수평이고 Doppler shift된 Evershed-carrying component. 후자가 **magnetized**임이 확인됨.
- **Canopy**: penumbra 외부 inclined field가 chromosphere로 연장. Evershed flow 대부분은 downflow 영역으로 사라짐 (<10%만 canopy로).
- **NCP** $\int V(\lambda)d\lambda$: LOS 속도/자기장 gradient에서만 non-zero. Horizontal flow channel in ambient field로 설명 가능 (Schlichenmaier 2002). 최근 관측: flow channel field가 **더 강함** (Tritschler 2007, Ichimoto 2008).

#### §3.3–3.4 Penumbral modeling: idealized models

**§3.4.1 Field-free gap / convective roll**:
- Parker (1979), Spruit & Scharmer (2006), Scharmer & Spruit (2006)
- 자기장이 거의 없는 gap에서 overturning convection이 열을 운반
- 기울어진 penumbral field에서 gap이 **elongated** → bright filaments with dark cores
- Evershed flow는 gap 내 radial outflow — 문제: 관측된 Evershed는 magnetized인데 gap은 field-free
- Danielson (1961) convective roll도 유사 geometry

**§3.4.2 Moving flux tube / dynamic flux tube model (Schlichenmaier 1998)**:
- Thin flux tube 근사. 2D tripartite 배경 위에서 단일 자속관의 동역학 추적.
- 시나리오: (a) tube가 magnetopause에 초기에 놓임 → (b) quiet Sun에서 복사 가열 → (c) 팽창 → (d) convectively unstable stratification으로 상승 → (e) upflow 발생 → (f) 광구에서 복사 냉각 → (g) supercritical outflow (Evershed).
- Penumbral grains = tube footpoint, horizontal outflow = Evershed, uncombed field = tube + background.
- **Serpentine flow**: overshoot → 파동 패턴 → downflow로 잠수. Sainz Dalda & Bellot Rubio (2008)의 관측적 증거.
- **Heat transport 문제**: single tube로는 penumbra brightness 설명 부족. 많은 tube가 필요.

#### §3.5 Idealized magneto-convection / 이상화된 자기대류 시뮬레이션
Proctor & Weiss (1982), Hurlburt et al. (1996, 2000), Weiss (2002): Chandrasekhar number $\zeta = \eta/\kappa$.
- $\zeta < 1$: oscillatory convection (umbra 상부 2 Mm에서 실현)
- $\zeta > 1$: steady overturning (대부분 대류층)
- Hurlburt 2D/3D: 경사 자기장에서 **traveling wave convection** — penumbra 구조/flow 모사
- Thomas et al. (2002) — **turbulent magnetic pumping**이 penumbra edge에서 field를 누름: 시험가설.

#### §3.6 Radiative magneto-convection / 복사 자기대류 — **두 번째 핵심 섹션**

##### §3.6.1 Numerical challenges
- Large domain + sufficient resolution + radiative transfer + partial ionization 동시 만족 필요
- **큰 Alfvén velocity** (수 kG / 낮은 $\rho$) → timestep 극히 작음 ($v_A > 1000$ km/s)
- 해결책 1: 작은 domain에 focus
- 해결책 2: 저-$\beta$ 영역에서 **Lorentz force artificially limit** — 2 order of magnitude 계산비용 절약 (Rempel et al. 2009b appendix)

##### §3.6.2 Umbral dots (Schüssler & Vögler 2006)
- Domain: 5.76 × 1.4 Mm. $\tau_R=1$ 위 400 km부터 시작. $B_z = 2500$ G.
- **결과**: field-free upflow plume이 자연스럽게 monolithic 2500 G 영역 내에서 형성 (Fig. 5).
- Plume이 광구 근처 밀도 급강 → 팽창 → 자기장 약화 → overturning convection 개시 → ~500 km downward extent → ~30 min lifetime
- **중심 dark lane**: stagnation point 위 밀도/압력 증가 → $\tau_R=1$ 면이 상승 → 낮은 온도 층에서 line formation → 어두움
- **철학적 함의**: umbral dot은 **cluster model 증거가 아님**. Monolithic 구조 내 local magneto-convective fluctuation.

##### §3.6.3 Umbra/penumbra transition, slab geometry
- Heinemann et al. (2007): 직사각형 슬랩(slab) ~4 Mm 지름의 mini-sunspot. 최초의 필라멘트 형성 시뮬레이션.
- Rempel et al. (2009b): 20 Mm 지름 슬랩 → **inner penumbra filament 2-3 Mm, 명확한 dark core** (Fig. 6, 7).
- **Filament 단면** (Fig. 7): 중심 upflow $v_z\sim2$ km/s, 가장자리 downflow, 강한 central horizontal outflow (filament axis를 따라) ~2 km/s, 약한 roll convection 흔적, 자기장 inclination >80°.

##### §3.6.4 Full sunspots (Rempel et al. 2009a)
- **Domain**: 98 × 49 × 6 Mm. 해상도: 32 km(수평)/16 km(수직). **1.8 billion grid points**. Opposite polarity sunspot pair, 각 $1.6\times10^{22}$ Mx.
- Fig. 8, 9: 완성된 penumbra + umbral dots + moat flow의 초기 신호.
- **핵심 결과 (Rempel 2011a)**:
  - Extended outer penumbra filling factor ≈ 1, Evershed 평균 5 km/s (피크 14 km/s)
  - **Inclination > 45°**에서 coherent outflow 발생 (Fig. 10)
  - 밝은 filament = 중간 정도 강한 radial field + 강하게 감소한 vertical field → inclination 변동 큼
  - Inner penumbra: bright filament ↔ outflow 상관 (+); outer: 상관 거의 0
  - Brightness–overturning velocity 관계: $I \propto \sqrt{v_z^{\text{RMS}}(\tau=1)}$
  - Fast horizontal outflow 구동 원리: **$\tau=1$ 바로 아래 좁은 경계층에서 Lorentz force가 upflow의 가압 energy를 radial 가속으로 전환**. 속도 피크 ≈ Alfvén velocity (deep photosphere).

- Fig. 13: 현재까지 최고 해상도 (16/12 km) — 사실적 penumbra.

##### §3.6.5 Unified picture of magneto-convection — **리뷰의 신 메시지**
**모든 fine structure = 경사진 자기장 배경에서 발생하는 overturning magneto-convection의 다른 국면**:
- 수직 자기장 (umbra): upflow plume → **umbral dot**
- 약간 경사 (umbra/penumbra): elongated UD → peripheral UD
- 경사 (inner penumbra): **dark-cored filament**
- 강한 경사 (outer penumbra): 수평 outflow 지배 — **Evershed flow**

공통 원리: **깊은 곳(큰 열용량)에 연결된 upflow plume이 photosphere brightness 유지**; inclined field가 anisotropy를 부여해 elongated 구조 + **Alfvén 속도급 radial outflow** 생성.

#### §3.7 Critical assessment
- 해상도 여전히 부족 — filament 내 turbulence 미해결 (laminar로 강제됨)
- 초기조건 arbitrary — self-consistent sunspot formation sim은 아직
- **Overturning convection 관측적 증거는 논쟁 중**: Sánchez Almeida (2007), Rimmele (2008)는 지지; Bellot Rubio (2010), Franz & Schlichenmaier (2009)는 부재. Bharti et al. (2011): C I 538.0 nm이 가장 유망한 탐지선. Scharmer (2011), Joshi (2011): stray-light 보정 후 수직 1.2 km/s RMS velocity 관측 → MHD 예측과 일치하는 방향이지만 확정적이지 않음.

### Part IV: Sunspot Formation and Evolution (§4, pp. 34–40)

#### §4.1 Flux emergence in lower CZ
- Solar CZ density contrast $10^6$ → scale height 50 Mm(base) to 100 km(surface)
- **Thin flux tube approximation**: anelastic, subsonic 가정 유효
- 초기 $B$ 추정: 100–150 kG (tilt angle, low-latitude emergence 관측과 일치)
- Weber et al. (2011): 3D global convection zone에서 drag force를 통한 상승 → **50 kG만 필요** (대류와 상호작용으로 2-3× 감소)
- **Fragmentation 문제**: untwisted tube는 coherent 상승 불가 (Schüssler 1979). Twist >특정값 필요 (Fan 2008).

#### §4.2 Flux emergence in upper CZ
- 최종 ~10 Mm는 fully compressible MHD + partial ionization + radiative transfer 필요
- Cheung et al. (2007, 2008): $10^{18}$–$10^{20}$ Mx emergence — active region 크기 미만
- **Cheung et al. (2010)**: Rempel의 코드 개선 + bottom BC로 half-torus 자속관 주입 → $7\times10^{21}$ Mx, 2 pores 형성, umbral dots + light bridges 관찰 (Fig. 14)
- Stein et al. (2011): 수평 자기장 bottom BC → 복잡한 active region 형성

#### §4.3 Formation of a penumbra / 반암부 형성 (여전히 open!)
- 대부분 flux emergence sim은 penumbra 형성을 **자연스럽게 보이지 않음** — grid 해상도 부족 + 부족한 자속량 + BC 문제
- Rempel et al. (2009a): extended penumbra는 **가까운 opposite polarity spot 사이**에서만
- Rempel (2011b): 상부 경계에서 inclination 인위적 증가 → periodic domain 단일 sunspot의 extended penumbra 확보
- **관측** (Fig. 15, Schlichenmaier 2010b): proto-spot에 flux patches가 합쳐지며 penumbra가 flux emergence 반대편에 형성.

#### §4.4 Sunspot evolution past emergence
- Active phase (polarity 분리) → passive phase
- **Disconnection 가설** (Schüssler & Rempel 2005): 형성 직후 깊은 root에서 $B$가 equipartition 이하로 drop → turbulent motion에 의해 "분리됨" → shallow sunspot(~5-10 Mm)
- Rempel (2011c): 16 Mm 깊이 domain에서 bottom BC 제약 없이 1-2 일 수명. 50 Mm 깊이 extrapolation → 10일 수명 → **관측 수 주와 모순** → deeper anchor 필요.

#### §4.5 Moat flows / 모트 흐름
- 흑점 주변 수평 outflow, ~500 m/s, 2× sunspot radius 폭
- Moving Magnetic Features (MMFs, Harvey & Harvey 1973): dipolar, sunspot polarity flux 수송
- **Gizon et al. (2000, 2009, 2010)**: f-mode helioseismology → 표면 Moat flow MMF tracking과 일치, 깊이 4.5 Mm까지 outflow 검출. Featherstone et al. (2011): 2-component — superficial + 5 Mm peak.
- Meyer et al. (1974) 제안: supergranular downflow vertex + penumbra heat blockage → reversal → moat outflow.
- 최근 MHD sim (Rempel 2011c): outflow 지배적, collar flow 없음 — pore simulations과 다름.

### Part V: Helioseismic Constraints (§5, p. 41)

- 흑점 → quiet Sun 대비 **작은 교란이 아님** → 선형 inversion 한계
- Kosovichev et al. (2000): time-distance → **2-layer 구조**: 상부 4 Mm wave speed 감소, 아래 10 Mm까지 증가. 5% 교란 → 2000 K 또는 ~20 kG.
- Moradi et al. (2010), Gizon et al. (2009): 대부분의 travel-time shift는 **얕은 상부 2 Mm**에서 기원. Braun & Birch (2006) frequency-dependent analysis 확인.
- Cameron et al. (2011): forward modeling → shallow sunspot model이 f, p1, p2 mode 관측을 잘 재현.
- **결론**: sunspot은 실제로 얕은 구조일 가능성 → Schüssler & Rempel (2005) disconnection 시나리오와 부합.

### Part VI: Summary (§6, pp. 42–43)

**2개 simplified model class**:
- (a) Flux tube: Evershed와 line asymmetry 성공; overall downflow와 penumbra brightness 전체 에너지 수송 설명 불충분
- (b) Gappy penumbra: field-free gap + deep overturning convection이 energy 공급; Evershed 자체 설명 부재

**Radiative MHD sim**: 경사각에 따른 magneto-convection 체제의 **전 스펙트럼**을 포착:
- Umbral dot = field-free upflow plume (gappy와 유사)
- Inner penumbra filament = **magnetized** (1 kG) — gap 모델과 달리 자기장 남음
- Outer penumbra = anisotropic magneto-convection, 강한 horizontal outflow (flux tube 유사성)

**Open problems**:
- Overturning convection의 **명확한 관측 확증**
- Penumbra 형성 trigger (magnetic? convective?)
- Sunspot decay 정량적 메커니즘
- 4D MHD sim의 spectral-line forward modeling으로 직접 관측 비교 (진행 중)

---

## 3. Key Takeaways / 핵심 시사점

1. **흑점은 정적이 아니다** — 작은 스케일에서의 **동적 magneto-convection**이 대규모 안정 구조를 만든다. Alfvén 시간(1시간)의 500배 수명은 **정역학적 균형이 아니라 난류적 재구성의 평균 결과**. / A sunspot is not static: small-scale **dynamic magneto-convection** produces a large-scale stable structure. The 500× ratio of lifetime to Alfvén time is **not static equilibrium but time-averaged turbulent reconfiguration**.

2. **흑점 어둠의 복합 원인** — 단일 원리가 아니라 (a) 자기장의 대류 억제(Biermann), (b) 기하학적 funneling(Hoyle/Jahn), (c) 수정된 magneto-convection (Deinzer/Meyer)의 삼중 기여. Umbra 광도 ~19% ($T_u=4000$ K, Stefan-Boltzmann 4제곱), penumbra heat flux 25% 감소. / Sunspot darkness is not from a single cause but from (a) magnetic suppression of convection, (b) geometric heat-flux funneling, and (c) modified magneto-convection. Umbra luminosity ~19%, penumbral heat flux reduced by 25%.

3. **Penumbra는 uncombed magnetic field의 filamentary 구조** — 두 성분의 **interlocking**: 덜 경사진 배경장(inner 30° → outer 60°)과 거의 수평인 Evershed-carrying component. NCP의 비영(non-zero)은 LOS gradient의 직접적 지표. / The penumbra is a filamentary structure with an **uncombed magnetic field** — interlocking of two components: a less-inclined background and a nearly horizontal Evershed-carrying flow channel. Nonzero NCP is the direct signature of LOS gradients.

4. **Radiative MHD가 흑점의 "통합 이론"** — **단일 원리: 경사 자기장 배경에서의 overturning magneto-convection**이 umbral dot(수직 plume), penumbral filament(경사 elongated cell), Evershed flow(Alfvén급 수평 outflow)를 **모두 자기일관적으로** 만들어낸다. 이 unified picture는 flux tube와 gappy penumbra라는 2개 단순 모델을 **부분적으로 포괄**. / Radiative MHD is the **unified theory** — a single principle (overturning magneto-convection in an inclined-field background) self-consistently produces umbral dots (vertical plumes), penumbral filaments (inclined elongated cells), and the Evershed flow (Alfvénic horizontal outflow). This picture **subsumes** both flux-tube and gappy-penumbra models.

5. **Evershed flow의 물리적 기원** — $\tau=1$ 바로 아래의 **narrow boundary layer**에서 Lorentz force가 upflow의 pressure-driven energy를 horizontal 가속으로 전환. 피크 속도 ≈ deep photosphere Alfvén velocity → 10 km/s 이상 가능. / Origin of the Evershed flow: in a **narrow boundary layer below $\tau=1$**, the Lorentz force channels the pressure-driven energy of upflows into horizontal acceleration. Peak speed $\approx$ deep-photospheric Alfvén velocity, easily exceeding 10 km/s.

6. **흑점은 "얕다 (shallow)"** — Helioseismology (Moradi 2010; Cameron 2011)는 wave speed 교란의 대부분이 상위 2 Mm에 집중되어 있음을 보인다. Schüssler & Rempel (2005)의 disconnection 시나리오 — 형성 직후 깊은 root가 turbulent motion으로 잠식되어 $B$가 equipartition 이하로 감소 — 와 부합. / Sunspots are **shallow**: helioseismology (Moradi 2010; Cameron 2011) shows most of the wave-speed perturbation is within the top 2 Mm, consistent with the Schüssler–Rempel (2005) disconnection scenario (deep roots disrupted by turbulent motion after formation).

7. **Flux emergence의 "초기 자기장" 문제가 해결되고 있음** — Thin flux tube 근사는 100-150 kG를 요구했지만, Weber et al. (2011)의 **3D global convection 상호작용** 시뮬레이션은 50 kG만으로도 관측된 active region 특성 재현. 대류가 자속관을 "끌어올리므로" 덜 강한 초기장으로 충분. / Flux emergence's "initial-field" problem is being resolved: thin-tube approximations required 100–150 kG, but Weber et al. (2011) with **3D global-convection interaction** show that 50 kG suffices — the convection effectively drags the tube upward, reducing the required initial field strength by a factor of 2–3.

8. **관측과 시뮬레이션이 2010년대에 직접 비교 가능해졌다** — 이 리뷰가 쓰인 2011년에는 아직 간접적이었지만, MURaM 출력의 synthetic Stokes profile을 Hinode/DKIST 관측과 직접 비교하는 것이 이후 10년의 표준 방법론. 따라서 이 리뷰는 **"simulation-based sunspot physics" 시대의 시작점**으로 기능한다. / Observations and simulations became directly comparable in the 2010s. Though still indirect in 2011, synthesizing Stokes profiles from MURaM and comparing them to Hinode/DKIST data became the standard methodology of the following decade — this review marks the **start of the simulation-based era of sunspot physics**.

---

## 4. Mathematical Summary / 수학적 요약

### (A) Plasma $\beta$ 와 magnetic pressure

$$\beta \equiv \frac{8\pi p}{B^2},\qquad p_{\text{total}} = p_{\text{gas}} + \frac{B^2}{8\pi}$$

- Umbra photosphere ($\tau=1$): $\beta\sim1$
- $z=-2$ Mm: $\beta\sim10$–$100$
- Corona: $\beta\ll1$
- **함의**: mono/spaghetti 전환은 $\beta$와 연결 — 표면에서는 자기 압력이 field-free gap을 누를 수 있지만 깊은 곳에서는 기체 압력이 자기장을 흩뜨린다.

### (B) Stefan-Boltzmann 흑점 광도

$$\frac{L_u}{L_{\text{QS}}} = \left(\frac{T_u}{T_{\text{QS}}}\right)^4 \approx \left(\frac{4000}{6060}\right)^4 \approx 0.19$$

Penumbra: $T_p=5275$ K → $L_p/L_{\text{QS}}\approx 0.58$. Heat flux 감소 — umbra 77%, penumbra 25%.

### (C) Magnetostatic equilibrium (Jahn-Schmidt tripartite 모델)

$$\nabla p - \rho\vec g + \frac{1}{4\pi}(\nabla\times\vec B)\times\vec B = 0$$

$$\nabla\!\left(p + \frac{B^2}{8\pi}\right) = \rho\vec g + \frac{(\vec B\cdot\nabla)\vec B}{4\pi}$$

Umbra-penumbra 경계 **current sheet**: 수평 pressure balance. Magnetic tension이 곡면 flux tube 유지.

### (D) Alfvén 속도와 dynamical timescale

$$v_A = \frac{B}{\sqrt{4\pi\rho}},\qquad \tau_A = \frac{D}{v_A}$$

Umbra photosphere: $B=3000$ G, $\rho\sim2\times10^{-7}$ g/cm³ → $v_A\sim60$ km/s, $\tau_A(D=30\,000\text{ km})\sim8$ min. 흑점 수명 weeks → **$\sim500\tau_A$**.

### (E) Moving flux tube: centrifugal vs. magnetic curvature balance

$$\kappa\rho v^2 = \frac{\kappa B^2}{4\pi}\quad\Rightarrow\quad v = v_A$$

Tube overshoot → $B$ 감소 → $v_A$ 감소 → quasi-stationary **serpentine flow**. Footpoint sufficient 시 stable.

### (F) Chandrasekhar number for magneto-convection

$$\zeta = \frac{\eta}{\kappa}$$

- $\zeta<1$: oscillatory convection (umbra 상부 2 Mm) → umbral dots oscillation 해석 시도
- $\zeta>1$: steady overturning (대부분 solar CZ)

실제로 radiative MHD sim은 $\zeta$ 상관없이 바로 overturning으로 이행 — idealized studies의 한계.

### (G) Radiative MHD 지배 방정식 (MURaM 등)

$$\frac{\partial\rho}{\partial t} + \nabla\cdot(\rho\vec v) = 0$$

$$\frac{\partial(\rho\vec v)}{\partial t} + \nabla\cdot\!\left(\rho\vec v\otimes\vec v + p_{\text{total}}\mathbb{I} - \frac{\vec B\vec B}{4\pi}\right) = \rho\vec g$$

$$\frac{\partial\vec B}{\partial t} = \nabla\times(\vec v\times\vec B) - \nabla\times(\eta\nabla\times\vec B)$$

$$\frac{\partial E}{\partial t} + \nabla\cdot[(E+p_{\text{total}})\vec v - \vec v\cdot\vec B\,\vec B/4\pi] = \rho\vec g\cdot\vec v + Q_{\text{rad}} + Q_{\text{visc}} + Q_{\text{ohm}}$$

$$Q_{\text{rad}} = \rho\kappa(4\pi J - 4\sigma T^4)$$

Rempel et al. (2009): 4096×2048×? grid, 1.8 billion points, 32/16 km 해상도, 수백만 CPU-hours.

### (H) Brightness–velocity 관계 (Rempel 2011a)

$$I \propto \sqrt{v_z^{\text{RMS}}(\tau=1)}$$

Penumbra brightness는 **overturning convection velocity의 제곱근**에 비례. 구조적 예측 — 관측으로 검증 진행 중.

### (I) Flux tube model — initial field at CZ base

- Thin flux tube approximation: $B\gtrsim 100$ kG (to avoid fragmentation without twist)
- 3D global convection interaction (Weber 2011): $B\approx 50$ kG sufficient
- Active region flux: $10^{21}$–$10^{22}$ Mx

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1610 ┃ Galileo — first telescopic sunspot observations
     ┃
1908 ┃ Hale — Zeeman-effect magnetic field; "tornado" theory (wrong)
1909 ┃ Evershed — radial penumbral outflow discovered
     ┃
1941 ┃ Biermann — "suppressed convection" (foundation of modern theory)
1942 ┃ Alfvén — MHD, frozen-in flux
1949 ┃ Hoyle — heat-flux funneling concept
     ┃
1964 ┃ Danielson — umbral dots discovered
1965 ┃ Deinzer — modified convection; pure suppression impossible
1972 ┃ Sheeley — moat flow first identified
1973 ┃ Harvey & Harvey — Moving Magnetic Features (MMFs)
1974 ┃ Meyer et al. — overstable magnetoconvection; moat collar-flow idea
1976 ┃ Spruit — monolithic vs cluster debate opens
1977 ┃ Spruit — resolves "no bright ring" puzzle via CZ heat conductivity
1979 ┃ Parker — field-free gap concept; rising flux tube ideas
     ┃
1988 ┃ von der Lühe — (solar AO correlation tracker — relevant to fine-structure observation)
     ┃
1992 ┃ Jahn — umbral dots / field-free gaps
1993 ┃ Solanki & Montavon — uncombed penumbral field concept
1994 ┃ Jahn & Schmidt — definitive 2D tripartite magnetostatic model
     ┃
1998 ┃ Schlichenmaier, Jahn & Schmidt — moving flux tube model
     ┃
2002 ┃ Scharmer et al. — dark cores in penumbral filaments (SST)
2003 ┃ Solanki — "Magnetic structure of sunspots" review
     ┃
2005 ┃ Schüssler & Rempel — flux disconnection scenario
2006 ┃ Schüssler & Vögler — first 3D MHD sim of umbral dot (Fig. 5)
2006 ┃ Spruit & Scharmer — gappy penumbra model
2007 ┃ Heinemann et al. — first 3D radiative MHD penumbra slab
2008 ┃ Thomas & Weiss — "Sunspots and Starspots" book
     ┃
2009 ┃ Rempel et al. — FIRST FULL 3D RADIATIVE MHD SUNSPOT PAIR (Figs. 8-10)
2010 ┃ Cheung et al. — flux emergence → pores with umbral dots (Fig. 14)
2010 ┃ Moradi et al. — helioseismic case for shallow sunspots
2011 ┃ Rempel — best-resolved 16 km sunspot sim (Fig. 13)
     ┃
★2011┃ ← THIS REVIEW (Rempel & Schlichenmaier, Living Reviews)
     ┃   Synthesis at the transition point
     ┃
2012 ┃ GREGOR first light
2015+┃ Hinode SP-driven inversions + MURaM forward modeling routine
2020 ┃ DKIST first light (4 m) — sub-20 km penumbra observations
2023 ┃ DKIST MCAO commissioning; direct sim-obs comparisons
2026 ┃ ML-accelerated inversions train on MURaM outputs
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **#23 Rimmele & Marino 2011 "Solar Adaptive Optics"** (LRSP) | AO가 가능하게 한 0.1" 해상도 Scharmer 2002 dark-core 발견이 이 리뷰 §3.2의 핵심 관측 근거 | "관측 기술 → 모델링 요구" 직접 연결. 본 리뷰는 AO가 드러낸 미세구조를 설명하는 이론. |
| **Solanki 2003 "The magnetic and dynamic structure of sunspots"** (A&ARv) | 이전 세대의 정의적 리뷰 (pre-radiative-MHD). §2, §3의 관측적 배경 | 본 리뷰는 Solanki 이후 8년간의 MHD 시뮬레이션 진보를 집약. |
| **Thomas & Weiss 2008 "Sunspots and Starspots"** (Cambridge) | Penumbra 이론의 책 분량 정리. 본 리뷰 §3.3-3.4 (idealized models)의 교과서 | Flux tube/gappy 모델의 상세 수식은 이 책에 있음. 본 리뷰는 이를 MHD sim과 비교. |
| **Scharmer et al. 2002 "Dark cores in penumbral filaments"** (Nature) | 본 리뷰의 가장 자주 인용되는 관측. 모든 penumbra 모델의 벤치마크 | 0.1" dark core 발견이 radiative MHD sim의 구체적 타겟. |
| **Schüssler & Vögler 2006 "Magnetoconvection in a sunspot umbra"** (ApJ) | §3.6.2의 출발점. Fig. 5의 umbral dot 시뮬레이션 | Radiative MHD 시대의 첫 페이지. |
| **Rempel et al. 2009a, b "Penumbral structure and outflows..."** (Science/ApJ) | 저자 Rempel의 대표 결과. §3.6.4 전체, Fig. 8-12 | Full sunspot sim. 이 리뷰가 자기 인용이 많지만 실제로 혁명적 결과. |
| **Schüssler & Rempel 2005 "The dynamical disconnection..."** (A&A) | §2.4, §4.4의 disconnection 시나리오 | 흑점이 왜 얕은가에 대한 theoretical scenario — helioseismology 결과와 부합. |
| **Fan 2009 "Magnetic fields in the solar convection zone"** (LRSP) | §4.1 flux emergence의 참고 리뷰 | Thin flux tube emergence 이론의 정리. 본 리뷰는 간단히 언급만. |
| **Gizon & Birch 2005 "Local helioseismology"** (LRSP) | §5의 helioseismology 기초 | 흑점 내부 구조 탐사의 방법론. |
| **Nordlund, Stein & Asplund 2009 "Solar surface convection"** (LRSP) | Quiet Sun granulation MHD sim의 3-decade 역사 | 본 리뷰는 이 성공을 흑점으로 확장하는 "next step". |

---

## 7. References / 참고문헌

**본 논문 / This paper**:
- Rempel, M. & Schlichenmaier, R., "Sunspot Modeling: From Simplified Models to Radiative MHD Simulations", *Living Reviews in Solar Physics*, **8**, 3 (2011). [DOI: 10.12942/lrsp-2011-3]

**핵심 관측 / Key observations**:
- Hale, G. E., "On the probable existence of a magnetic field in sun-spots", *ApJ*, **28**, 315 (1908).
- Evershed, J., "Radial movement in sun-spots", *MNRAS*, **69**, 454 (1909).
- Danielson, R. E., "The structure of sunspot penumbras. II.", *ApJ*, **139**, 45 (1964).
- Scharmer, G. B. et al., "Dark cores in sunspot penumbral filaments", *Nature*, **420**, 151 (2002).
- Bellot Rubio, L. R. et al., "Vector spectropolarimetry of dark-cored penumbral filaments with Hinode", *ApJ Lett.*, **668**, L91 (2007).
- Ichimoto, K. et al., "Twisting motions of sunspot penumbral filaments", *Science*, **318**, 1597 (2007).

**역사적 이론 / Historical theory**:
- Biermann, L., "Der gegenwärtige Stand der Theorie konvektiver Sonnenmodelle", *Viertel. Astron. Ges.*, **76**, 194 (1941).
- Alfvén, H., "Existence of Electromagnetic-Hydrodynamic Waves", *Nature*, **150**, 405 (1942).
- Hoyle, F., *Some Recent Researches in Solar Physics*, Cambridge Univ. Press (1949).
- Deinzer, W., "On the magneto-hydrostatic theory of sunspots", *ApJ*, **141**, 548 (1965).
- Meyer, F., Schmidt, H. U., Weiss, N. O., Wilson, P. R., "The growth and decay of sunspots", *MNRAS*, **169**, 35 (1974).
- Parker, E. N., "Sunspots and the physics of magnetic flux tubes. I-III", *ApJ*, **230**, 905 (1979).
- Spruit, H. C., "Heat flow near obstacles in the solar convection zone", *Solar Phys.*, **55**, 3 (1977).

**Simplified models**:
- Jahn, K. & Schmidt, H. U., "Thick penumbra in a magnetostatic sunspot model", *A&A*, **290**, 295 (1994).
- Schlichenmaier, R., Jahn, K. & Schmidt, H. U., "A dynamical model for the penumbral fine structure and the Evershed effect in sunspots", *ApJ Lett.*, **493**, L121 (1998).
- Spruit, H. C. & Scharmer, G. B., "Fine structure, magnetic field and heating of sunspot penumbrae", *A&A*, **447**, 343 (2006).
- Scharmer, G. B. & Spruit, H. C., "Magnetostatic penumbra models with field-free gaps", *A&A*, **460**, 605 (2006).

**Radiative MHD simulations**:
- Schüssler, M. & Vögler, A., "Magnetoconvection in a sunspot umbra", *ApJ Lett.*, **641**, L73 (2006).
- Heinemann, T., Nordlund, Å., Scharmer, G. B., Spruit, H. C., "MHD simulations of penumbra fine structure", *ApJ*, **669**, 1390 (2007).
- Rempel, M., Schüssler, M., Knölker, M., "Radiative magnetohydrodynamic simulation of sunspot structure", *ApJ*, **691**, 640 (2009a).
- Rempel, M., Schüssler, M., Cameron, R. H., Knölker, M., "Penumbral structure and outflows in simulated sunspots", *Science*, **325**, 171 (2009b).
- Rempel, M., "Penumbral fine structure and driving mechanisms of large-scale flows in simulated sunspots", *ApJ*, **729**, 5 (2011a).
- Kitiashvili, I. N. et al., "Traveling waves of magneto-convection and the origin of the Evershed effect in sunspots", *ApJ Lett.*, **700**, L178 (2009).

**Flux emergence & formation**:
- Schüssler, M. & Rempel, M., "The dynamical disconnection of sunspots from their magnetic roots", *A&A*, **441**, 337 (2005).
- Cheung, M. C. M. et al., "Simulation of the formation of a solar active region", *ApJ*, **720**, 233 (2010).
- Stein, R. F. et al., "Solar flux emergence simulations", *Solar Phys.*, **268**, 271 (2011).
- Weber, M. A., Fan, Y., Miesch, M. S., "The rise of active region flux tubes in the turbulent solar convective envelope", *ApJ*, **741**, 11 (2011).
- Fan, Y., "Magnetic fields in the solar convection zone", *Living Rev. Solar Phys.*, **6**, 4 (2009).

**Helioseismology**:
- Gizon, L. & Birch, A. C., "Local helioseismology", *Living Rev. Solar Phys.*, **2**, 6 (2005).
- Kosovichev, A. G., Duvall Jr., T. L., Scherrer, P. H., "Time-distance inversion methods and results", *Solar Phys.*, **192**, 159 (2000).
- Moradi, H. et al., "Modeling the subsurface structure of sunspots", *Solar Phys.*, **267**, 1 (2010).
- Cameron, R. H., Gizon, L., Schunker, H., Pietarila, A., "Constructing semi-empirical sunspot models for helioseismology", *Solar Phys.*, **268**, 293 (2011).

**Reviews cited**:
- Solanki, S. K., "Sunspots: An overview", *A&A Rev.*, **11**, 153 (2003).
- Thomas, J. H. & Weiss, N. O., *Sunspots and Starspots*, Cambridge University Press (2008).
- Nordlund, Å., Stein, R. F., Asplund, M., "Solar surface convection", *Living Rev. Solar Phys.*, **6**, 2 (2009).
