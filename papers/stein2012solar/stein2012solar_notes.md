---
title: "Solar Surface Magneto-Convection"
authors: Robert F. Stein
year: 2012
journal: "Living Reviews in Solar Physics"
doi: "10.12942/lrsp-2012-4"
topic: Living_Reviews_in_Solar_Physics
tags: [magneto-convection, radiative-MHD, granulation, sunspots, surface-dynamo, flux-emergence]
status: completed
date_started: 2026-04-23
date_completed: 2026-04-23
---

# 30. Solar Surface Magneto-Convection / 태양 표면 자기대류

---

## 1. Core Contribution / 핵심 기여

**English**: Stein's 2012 Living Reviews article consolidates a decade of "realistic" radiative-magnetohydrodynamic (R-MHD) simulations that have transformed our understanding of how convection and magnetic fields couple in the top ~20 Mm of the solar convection zone and the overlying photosphere. The review compares two complementary simulation strategies: "idealized" (ideal-gas, diffusive radiation, anelastic) studies that cleanly isolate physics, and "realistic" (tabular EOS, non-grey multi-group radiative transfer, fully compressible) studies that can be directly compared to spectra, polarimetry, and G-band imagery. Stein's central argument is that solar magneto-convection is so non-linear and non-local that analytical mixing-length arguments fail — only full 3D simulations reproduce the observed granulation, flux-emergence hierarchy, and pore/sunspot properties. The paper walks through four physical regimes: (i) turbulent convection and surface dynamo action (Sec. 4.1), (ii) subsurface rise and emergence of magnetic flux (Sec. 4.2), (iii) small-scale flux concentrations including G-band bright points and faculae (Sec. 4.3), and (iv) pores and sunspots (Sec. 4.4).

**한국어**: Stein(2012) Living Reviews 논문은 태양 대류층 최상부 ~20 Mm와 그 위 광구에서 대류와 자기장이 결합되는 방식을 완전히 새로 쓴 지난 10년의 "현실적" 복사-자기유체역학(R-MHD) 시뮬레이션을 집대성한다. 저자는 두 가지 상호보완적 시뮬레이션 전략을 비교한다. "이상화된" 연구(이상기체, 확산 복사, anelastic)는 물리를 깨끗이 분리해 볼 수 있고, "현실적" 연구(표 형태 EOS, 비회색 다중 그룹 복사전달, 완전 압축성)는 스펙트럼·편광·G-band 영상과 직접 비교 가능하다. 핵심 논지는 태양 자기대류가 비선형·비국소적이어서 혼합길이 이론 같은 해석적 논의로는 재현되지 않고, 오직 완전한 3D 시뮬레이션만이 관측된 입상반, 자속 출현 계층, 포어·흑점의 특성을 재현한다는 것이다. 논문은 네 가지 물리 영역을 순서대로 다룬다. (i) 난류 대류와 표면 다이나모(4.1), (ii) 표면하 자속 상승과 출현(4.2), (iii) G-band bright point·광반(facula)을 포함한 소규모 자속 집중(4.3), (iv) 포어와 흑점(4.4).

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction — Why Magneto-Convection Matters / 1부: 자기대류가 중요한 이유

**English** (pp. 5–6): Stein motivates the review by noting that the solar convection zone is the ultimate mechanical energy source for all chromospheric and coronal activity, mediated through magnetic fields. The topology of convection is controlled by mass conservation in a stratified atmosphere: density scale height $H_\rho = -(d\ln\rho/dr)^{-1}$ sets the over-turning length. A warm fluid parcel rising a distance $\Delta r$ becomes overdense relative to its new surroundings by $\Delta\rho/\rho = -(d\ln\rho/dr)\Delta r$, which forces it to turn over into lanes within one scale height. This asymmetry — broad laminar upflows and narrow turbulent downflows — is central to every subsequent argument. Convection morphology cascades outward: granules (~1 Mm, ~5 min lifetime) are fed by mesogranules (~5 Mm) which in turn live inside supergranules (~30 Mm). Magnetic features span from ~70 km (SST resolution) to 100 Mm (active regions). Three orders of magnitude more flux emerges in the quiet Sun than in active regions (Thornton & Parnell 2011).

**한국어** (5–6쪽): Stein은 태양 대류층이 모든 색층·코로나 활동의 기계적 에너지원이며, 그 전달 매개가 자기장임을 상기시키며 리뷰를 시작한다. 대류의 위상은 성층 대기의 질량보존에 의해 결정된다. 밀도 스케일 높이 $H_\rho = -(d\ln\rho/dr)^{-1}$이 뒤집힘 길이를 규정한다. 따뜻한 유체 조각이 $\Delta r$만큼 상승하면 주변에 비해 $\Delta\rho/\rho = -(d\ln\rho/dr)\Delta r$만큼 과밀해지므로, 한 스케일 높이 안에서 레인으로 뒤집힌다. 이 비대칭(넓은 층류 상승류, 좁은 난류 하강류)은 이후 모든 논증의 중심이다. 대류 형태는 바깥으로 계층화된다. 입상반(~1 Mm, 수명 ~5분)은 메조입상반(~5 Mm)에 들어 있고, 다시 초입상반(~30 Mm)에 포함된다. 자기 구조는 ~70 km(SST 해상도)에서 100 Mm(활동영역)까지 아우른다. 정적태양에서 출현하는 자속은 활동영역보다 3자릿수 더 많다(Thornton & Parnell 2011).

Key mixing-length estimate / 주요 혼합길이 추정:
- Convective velocity near surface: $u_{\text{conv}} \sim (F_\odot/\rho)^{1/3} \approx 2\text{–}3\,\text{km/s}$ for $F_\odot = 6\times 10^{10}\,\text{erg/cm}^2\text{/s}$, $\rho\sim 3\times10^{-7}\,\text{g/cm}^3$.
- Turnover time: $\tau \sim H_\rho/u_{\text{conv}} \sim 150\,\text{km}/3\,\text{km/s}\sim 50\,\text{s}$ for pressure scale height near the surface.
- Equipartition field: $B_{\text{eq}}=(4\pi\rho)^{1/2}u_{\text{conv}}\sim 400\text{–}500\,\text{G}$ near τ=1.

### Part II: Equations (§2, pp. 7–9) / 2부: 방정식 (7–9쪽)

**English**: The R-MHD system couples five conservation equations plus an EOS and radiative transfer. Eq. (1) is continuity; Eq. (2) momentum with Lorentz and Coriolis; Eqs. (4–5) kinetic and internal energy; Eq. (8) magnetic energy balance; Eq. (10) induction with Eq. (11) Ohm's law. The key physical insights:

1. The Lorentz force in Eq. (2) inhibits motion perpendicular to $\mathbf{B}$, killing convective transport.
2. The viscous stress tensor (Eq. 3), $\tau_{ij}=\mu(\partial_i u_j+\partial_j u_i - \tfrac{2}{3}\delta_{ij}\nabla\cdot\mathbf{u})$, is subgrid in practice — numerical dissipation supplies it.
3. Radiative heating/cooling (Eq. 6) $Q_{\text{rad}}=\int\!\!\int \rho\kappa_\nu(I_\nu-S_\nu)\,d\Omega\,d\nu$ cannot be treated by diffusion near τ=1 nor by escape probability — one must solve the radiative transfer equation (Eq. 12) explicitly along rays.
4. Ionization energy accounts for ~2/3 of the energy transported near the surface (Stein & Nordlund 1998), so the EOS must include LTE ionization of H, He, and abundant elements.
5. The Hall and electron pressure terms in Ohm's law are usually neglected but the Hall term becomes important in the weakly ionized photosphere.

**한국어**: R-MHD 시스템은 다섯 개의 보존방정식과 EOS·복사전달로 구성된다. 식(1)은 연속, (2)는 로런츠·코리올리 포함 운동량, (4–5)는 운동·내부에너지, (8)은 자기에너지 수지, (10)은 유도(식(11) 옴의 법칙과 함께)이다. 물리적 요점은:

1. 식(2)의 로런츠 힘은 $\mathbf{B}$에 수직인 운동을 억제해 대류수송을 차단한다.
2. 점성 응력텐서(식 3) $\tau_{ij}=\mu(\partial_i u_j+\partial_j u_i - \tfrac{2}{3}\delta_{ij}\nabla\cdot\mathbf{u})$는 실제로는 격자 이하이며 수치 소산이 대신 공급한다.
3. 복사 가열/냉각(식 6) $Q_{\text{rad}}=\int\!\!\int \rho\kappa_\nu(I_\nu-S_\nu)\,d\Omega\,d\nu$은 τ=1 부근에서 확산이나 탈출확률로 처리할 수 없고, 식(12)의 복사전달을 광선을 따라 명시적으로 풀어야 한다.
4. 이온화 에너지는 표면 근처 수송 에너지의 ~2/3를 담당하므로(Stein & Nordlund 1998), EOS에는 H·He 및 풍부 원소의 LTE 이온화가 포함되어야 한다.
5. 옴의 법칙의 홀 항과 전자압 항은 보통 무시하지만, 약하게 이온화된 광구에서는 홀 항이 중요해질 수 있다.

### Part III: Observations (§3, pp. 10–13) / 3부: 관측

**English**: Stein summarizes the observational input against which simulations must be tested:
- **Spatial hierarchy**: features exist from ~70 km (SST limit) to ~100 Mm (Fig. 1 Parnell et al. 2009).
- **Flux distribution**: a power law with slope –1.85 down to ~$10^{17}$ Mx; no characteristic scale is evident (Fig. 2).
- **Magnetic butterfly diagram** (Fig. 3): Hale's polarity law, Joy's law tilt, poleward transport over 22-yr cycle.
- **Strong vertical vs. weak horizontal**: kilogauss vertical fields inhabit the network and intergranular lanes, whereas horizontal fields permeate granule interiors with average $\langle B_h\rangle = 55\text{–}60$ G, vertical average only 11 G (Lites et al. 2008).
- **G-band intensity**: bright points in intergranular lanes (Fig. 4) trace kG concentrations; the "hot-wall" effect explains facular limb brightening (Fig. 23).
- **Emerging Ω-loops**: appear as horizontal fields first (Stokes Q,U) followed by vertical polarities at endpoints (Fig. 5, Martínez González & Bellot Rubio 2009; Centeno et al. 2007).

**한국어**: Stein은 시뮬레이션이 맞춰야 할 관측 입력을 정리한다.
- **공간 계층**: ~70 km(SST 한계)에서 ~100 Mm(활동영역)까지 존재(Fig. 1, Parnell et al. 2009).
- **자속 분포**: ~$10^{17}$ Mx까지 기울기 –1.85 멱법칙, 특징적 스케일 없음(Fig. 2).
- **자기 나비 다이어그램**(Fig. 3): Hale 극성 법칙, Joy 경사, 22년 주기의 극 방향 수송.
- **강한 수직 vs. 약한 수평**: kG 수직장은 네트워크·입상반 간 레인에 위치, 수평장은 입상반 내부에 퍼져 있음. 평균 $\langle B_h\rangle=55\text{–}60$ G, 수직 평균은 11 G에 불과(Lites et al. 2008).
- **G-band 세기**: 입상반 간 레인의 bright point(Fig. 4)가 kG 집중을 추적; "hot-wall" 효과가 주변부(limb) 광반의 밝기 상승을 설명(Fig. 23).
- **출현하는 Ω-loop**: 처음엔 수평장(Stokes Q,U)이 보이고, 끝점에서 수직극이 나타남(Fig. 5, Martínez González & Bellot Rubio 2009; Centeno et al. 2007).

### Part IV: Simulations §4.1 — Turbulent Convection & Dynamo (pp. 14–17) / 4부: 난류 대류와 다이나모

**English**: Surface dynamo action was demonstrated conclusively by Vögler & Schüssler (2007) and Pietarila Graham et al. (2010) using shallow, high-resolution magneto-convection simulations. Pietarila Graham et al. showed that in the kinematic (linear growth) phase:
- 95% of magnetic energy generation comes from **stretching** (turbulent flow working against magnetic tension) at sub-granule scales (0.1–1 Mm) in downdrafts, producing still smaller-scale field (20–200 km).
- 5% comes from **compression** against magnetic pressure.
- There is also a cascade of magnetic energy from dynamo-generated scales to smaller scales (Fig. 6).
In the saturated phase, stretching is almost balanced by compressive cascade + MHD waves. Schüssler & Vögler (2008) and Abbett (2007) confirmed that such dynamos produce a **preponderance of horizontal field** (Fig. 9) because loops connecting opposite sides of granules are wider than a scale height: $\langle B_H\rangle/\langle B_V\rangle \approx L/h$ (Steiner 2010).

**한국어**: 표면 다이나모 작용은 Vögler & Schüssler (2007)와 Pietarila Graham et al. (2010)이 얕고 고해상도인 자기대류 시뮬레이션으로 결정적으로 입증했다. Pietarila Graham et al.은 운동학적(선형 성장) 단계에서:
- 자기에너지 생성의 95%가 하강류 내 sub-granule 스케일(0.1–1 Mm)에서 **신장**(난류가 자기장력에 대항해 일)으로부터 오며 더 작은 스케일(20–200 km)의 자기장을 만들고,
- 5%가 자기압에 대한 **압축**에서 오며,
- 다이나모 생성 스케일에서 더 작은 스케일로의 자기에너지 캐스케이드도 존재함을 보였다(Fig. 6).
포화 단계에서 신장은 압축성 캐스케이드 + MHD 파 생성과 거의 균형을 이룬다. Schüssler & Vögler (2008), Abbett (2007)은 이런 다이나모가 **수평장의 우세**를 낳음을 확인했다(Fig. 9). 입상반 양쪽을 잇는 고리가 스케일 높이보다 넓기 때문이다: $\langle B_H\rangle/\langle B_V\rangle \approx L/h$ (Steiner 2010).

### Part V: §4.2 — Subsurface Rise & Emergence (pp. 17–27) / 5부: 표면하 상승과 출현

**English**: Three initialization strategies produce emerging flux: (i) coherent twisted flux tubes forced through the bottom boundary (Cheung et al. 2007, 2010), (ii) minimally structured horizontal field advected by inflows (Stein et al. 2010ab), (iii) locally produced by dynamo action (Abbett 2007). Despite these differences, common features emerge:
- **Hierarchy of Ω- and U-loops** (Figs. 10, 12) — small loops ride piggy-back on larger ones in serpentine fashion (Cheung et al. 2007).
- **Expansion**: for horizontal flux tube with $\alpha=\partial v_x/\partial x$ and $\partial v_z/\partial z = \epsilon\alpha$, the field evolves as $D\ln B/Dt = -(1+\epsilon)\alpha$ and density as $D\ln\rho/Dt = -(2+\epsilon)\alpha$, giving $B\propto\rho^{(1+\epsilon)/(2+\epsilon)}$. Isotropic expansion ($\epsilon=1$): $B\propto\rho^{2/3}$; mostly horizontal ($\epsilon\ll 1$): $B\propto\rho^{1/2}$ (Cheung et al. 2010).
- **"Pepper-and-salt" pattern**: small bipoles appear with mixed polarities, then collect into unipolar regions on supergranule boundaries.
- **Turbulent pumping**: downflow–upflow asymmetry pumps flux downward on average (Drobyshevski et al. 1980; Nordlund et al. 1992).
- **Flux tubes are leaky**: Cattaneo et al. (2006) showed that without symmetries there are no flux surfaces separating a concentration from its surroundings — the "flux tube" is a local surface phenomenon (Fig. 19).

**한국어**: 자속 출현을 만드는 세 가지 초기화 전략이 있다. (i) 하단 경계로 결맞는 꼬인 자속관을 강제 주입(Cheung et al. 2007, 2010), (ii) 미세구조 없는 수평장을 유입류로 이류(Stein et al. 2010ab), (iii) 국소 다이나모(Abbett 2007). 접근이 달라도 공통 특징이 나타난다.
- **Ω/U-loop 계층**(Fig. 10, 12) — 작은 루프가 큰 루프에 올라타는 사행(serpentine) 구조(Cheung et al. 2007).
- **팽창**: 수평 자속관에 $\alpha=\partial v_x/\partial x$, $\partial v_z/\partial z = \epsilon\alpha$이면 $D\ln B/Dt = -(1+\epsilon)\alpha$, $D\ln\rho/Dt = -(2+\epsilon)\alpha$ → $B\propto\rho^{(1+\epsilon)/(2+\epsilon)}$. 등방 팽창($\epsilon=1$): $B\propto\rho^{2/3}$, 거의 수평($\epsilon\ll 1$): $B\propto\rho^{1/2}$.
- **"Pepper-and-salt" 패턴**: 혼합 극성 작은 bipole이 먼저 나오고 나중에 초입상반 경계의 단극(unipolar) 영역으로 모임.
- **난류 펌핑**: 하강류-상승류 비대칭이 평균적으로 자속을 아래로 끌어내림.
- **자속관은 새는 구조**: Cattaneo et al. (2006)은 대칭이 없으면 집중 영역과 주변을 가르는 자속면이 없음을 보였다 — "자속관"은 국소 표면 현상일 뿐(Fig. 19).

Concrete numerical example / 구체적 예: Cheung et al. (2010)은 7.5 Mm 깊이에서 중심장 21 kG, 총 자속 $7.6\times 10^{21}$ Mx의 half-torus를 주입하여 3.0 hr에 혼합 극성 출현을 거쳐 22.1 hr에 활동영역급 양극성 포어 쌍을 형성한다(Fig. 14).

### Part VI: §4.3 — Small-Scale Flux Concentrations (pp. 27–31) / 6부: 소규모 자속 집중

**English**: The life cycle of a kilogauss concentration:
1. Divergent upflows sweep field into intergranular lanes on granule timescales (~5 min), with strongest concentrations at lane vertices (Fig. 15).
2. Lorentz force then **inhibits transverse plasma motion** inside the concentration → convective heat transport to the surface is suppressed → radiative cooling continues from the top → density scale height decreases.
3. **Convective intensification / convective collapse** (Parker 1978; Spruit 1979): plasma drains out of the concentration, raising $B$ until $p_{\text{in}} + B^2/8\pi = p_{\text{out}}$.
4. Resulting B ≫ equipartition (often 1.5–2 kG at surface), far exceeding $B_{\text{eq}}\sim 400\text{–}500$ G.
5. **Wilson depression**: evacuated column → reduced opacity → τ=1 surface sinks by ~200–350 km (Fig. 22, Maltby 2000).
6. **Narrow** concentrations heated by hot sidewalls → appear bright (G-band points, Fig. 21).
7. **Wide** concentrations not heated → appear dark (pores, sunspots).

Figure 22 (Carlsson et al. 2004) shows the temperature, density, and magnetic field in a vertical slice. G-band mean formation height is ~54 km above τ=1. G-band brightness correlates with strong field (Fig. 24) but not all strong-field regions are bright (larger concentrations are dark).

**한국어**: kG 집중의 생애주기:
1. 발산 상승류가 입상반 시간(~5 min)에 자기장을 입상반 간 레인으로 쓸어 넣고, 레인 꼭짓점에 가장 강하게 집중시킨다(Fig. 15).
2. 로런츠 힘이 집중 내부 **횡방향 운동을 억제** → 표면으로의 대류 열수송이 차단 → 상부에서 복사 냉각은 계속 → 밀도 스케일 높이 감소.
3. **대류적 강화 / 대류적 붕괴**(Parker 1978; Spruit 1979): 플라즈마가 집중에서 빠져나와 $B$ 상승, $p_{\text{in}} + B^2/8\pi = p_{\text{out}}$ 성립.
4. 결과적으로 B ≫ 등분배값(표면에서 보통 1.5–2 kG), $B_{\text{eq}}\sim 400\text{–}500$ G를 훨씬 초과.
5. **윌슨 함몰**: 배기된 기둥 → 불투명도 감소 → τ=1 면이 ~200–350 km 가라앉음(Fig. 22, Maltby 2000).
6. **좁은** 집중은 측벽 가열로 밝음(G-band point, Fig. 21).
7. **넓은** 집중은 가열 불충분 → 어두움(포어, 흑점).

Fig. 22(Carlsson et al. 2004)의 수직 슬라이스가 온도·밀도·자기장을 보여준다. G-band 평균 생성 고도는 τ=1 위 ~54 km. G-band 밝기는 강한 자기장과 상관이 있으나(Fig. 24), 모든 강자기장 영역이 밝은 것은 아니다(큰 집중은 어둡다).

### Part VII: §4.4 — Pores & Sunspots (pp. 31–40) / 7부: 포어와 흑점

**English**: Recent "realistic" R-MHD simulations have produced all the main sunspot features:
- **Micropores** (Bercik 2002) form spontaneously at intergranular lane vertices where several lanes meet — an upflow reverses, the granule disappears, and strong field moves in.
- **Pores**: develop spontaneously in emerging Ω-loop simulations as rising flux leaves vertical concentrations behind (Stein & Nordlund 2006). The pore in Fig. 27 has $2.4\times 10^{20}$ Mx flux, area 6 Mm², field $\sim$2 kG, lifetime >8–12 hr. Pores are **edge-brightened** because τ=1 at their rim sits higher in temperature (Fig. 27).
- **Sunspots with penumbrae** (Rempel et al. 2009; Rempel 2011): a pair of axisymmetric self-similar flaring magnetic funnels is imposed; penumbrae form only when initial inclination exceeds ~45°. Cheung et al. (2010) grew a pair of sunspots from an emerging Ω-loop (Fig. 14).
- **Umbral dots** (Schüssler & Vögler 2006): narrow upflow plumes of hot plasma flanked by narrow cool downflows (Fig. 32, plume velocity up to 2.7 km/s). As plasma piles up near τ=1 and expands laterally, it reduces field strength, increases opacity, and produces the central dark lane through the bright dot.
- **Evershed outflow** (Figs. 33–34, Rempel 2011): in inclined penumbral fields, pressure pushes the upflow, overturning horizontal flow is channeled along field lines, and the Lorentz force turns the flow horizontal.

**한국어**: 최근 "현실적" R-MHD 시뮬레이션은 흑점의 모든 주요 특징을 재현해냈다.
- **마이크로포어**(Bercik 2002)는 여러 레인이 만나는 꼭짓점에서 자발적으로 형성됨 — 상승류가 역전되고, 입상반이 사라지고, 강한 자기장이 이동해 들어옴.
- **포어**: 출현하는 Ω-loop 시뮬레이션에서 상승 자속이 수직 집중을 남기며 자발적으로 발달(Stein & Nordlund 2006). Fig. 27의 포어는 자속 $2.4\times 10^{20}$ Mx, 면적 6 Mm², 자기장 ~2 kG, 수명 >8–12시간. 포어는 **가장자리가 밝은데(edge-brightened)**, 테두리에서 τ=1이 더 높은 온도에 위치하기 때문이다(Fig. 27).
- **반영(penumbra)을 가진 흑점**(Rempel et al. 2009; Rempel 2011): 축대칭 자기상사(self-similar) flaring 퍼널 쌍을 부과. 초기 경사가 ~45°를 초과할 때만 반영이 형성된다. Cheung et al. (2010)은 출현 Ω-loop에서 흑점 쌍을 자라게 했다(Fig. 14).
- **Umbral dots**(Schüssler & Vögler 2006): 좁은 고온 플라즈마 상승 플룸과 좁은 저온 하강류가 인접(Fig. 32, 플룸 속도 2.7 km/s까지). 플라즈마가 τ=1 근처에서 쌓여 측방 확장 → 자기장 약화, 불투명도 증가 → 밝은 dot 중앙에 어두운 띠 생성.
- **Evershed 유출**(Fig. 33–34, Rempel 2011): 경사진 반영 자기장에서 압력이 상승류를 밀고, 뒤집히는 수평 흐름이 자기선을 따라 이동하며, 로런츠 힘이 흐름을 수평 방향으로 돌려 Evershed 유출을 만든다.

### Part VIII: §5 — The Future (p. 41) / 8부: 향후 과제

**English**: Stein lists open problems: ab-initio sunspot formation from a fully dynamo-generated field, the role of non-LTE effects in chromospheric layers, quantitative matching of solar abundance determinations to 3D models, and coupling photospheric drivers to full coronal simulations (BIFROST being the first attempt).

**한국어**: Stein은 열린 문제를 나열한다. 완전히 다이나모 기원인 자기장으로부터 ab-initio 흑점 형성, 색층에서 비LTE 효과의 역할, 태양 원소 존재량 결정의 3D 모델 정량적 일치, 그리고 광구 구동자와 코로나 시뮬레이션 커플링(BIFROST가 첫 시도).

### Part IX: Figures worth remembering / 9부: 기억할 만한 그림들

**English**: Several figures anchor the review and are worth memorizing:
- **Fig. 1** (Parnell et al. 2009): Hinode Stokes-V image of a sunspot + quiet Sun at 108 km pixel, the "data" that simulations must reproduce.
- **Fig. 6** (Pietarila Graham et al. 2010): the 95/5 dynamo energy-transfer diagram — stretching dominates over compression.
- **Fig. 10**: time still of rising magnetic flux — shows the hierarchy of U- and Ω-loops and filamentary rise.
- **Fig. 13** (Cheung et al. 2007): twisted flux tube rising from z=−1.5 Mm, expanding and fragmenting as it approaches the surface over ~13 min.
- **Fig. 14** (Cheung et al. 2010): 22.1 hr time sequence showing an entire active region form from an emerging half-torus.
- **Fig. 15** (Stein & Nordlund 2006): sweeping of 30-G horizontal seed field into intergranular lanes in 30 min.
- **Fig. 22** (Carlsson et al. 2004): temperature, density, field slice showing Wilson depression and G-band formation height.
- **Fig. 25** (Bercik): flux tube pressure balance and evacuation signature.
- **Fig. 27**: spontaneously formed simulated pore with edge brightening.
- **Figs. 31, 33, 34** (Rempel 2011): sunspot with penumbra, Evershed outflow velocity, and energy conversion map.

**한국어**: 리뷰의 뼈대를 이루는 몇 그림은 기억해 둘 가치가 있다.
- **Fig. 1**(Parnell et al. 2009): 108 km 픽셀 히노데 Stokes-V 흑점 + 정적태양 — 시뮬레이션이 맞춰야 할 "데이터".
- **Fig. 6**(Pietarila Graham et al. 2010): 95/5 다이나모 에너지 전송 다이어그램 — 신장이 압축보다 우세.
- **Fig. 10**: 상승하는 자속 스틸 — U/Ω-loop 계층과 필라멘트성 상승 시각화.
- **Fig. 13**(Cheung et al. 2007): z=−1.5 Mm에서 상승하는 꼬인 자속관이 ~13분 동안 팽창·조각화.
- **Fig. 14**(Cheung et al. 2010): 22.1시간 시퀀스로 half-torus 출현으로부터 활동영역 전체 형성.
- **Fig. 15**(Stein & Nordlund 2006): 30 G 수평 초기장이 30분 내 입상반 간 레인으로 쓸림.
- **Fig. 22**(Carlsson et al. 2004): 온도·밀도·자기장 슬라이스 — Wilson 함몰과 G-band 형성 고도.
- **Fig. 25**(Bercik): 자속관 압력 균형과 배기 흔적.
- **Fig. 27**: 자발적으로 형성된 모의 포어와 가장자리 밝아짐.
- **Fig. 31, 33, 34**(Rempel 2011): 반영을 가진 흑점, Evershed 유출 속도, 에너지 변환 맵.

### Part X: Concrete Numerical Examples / 10부: 구체적 수치 예시

**English**: Across the review Stein embeds many quantitative anchors. Collecting them:
- **Granule size**: ~1 Mm width, ~300 s lifetime, velocity 2–3 km/s.
- **Mesogranule**: ~5 Mm, hours lifetime — observational only, no direct convective driver.
- **Supergranule**: ~30 Mm, ~24 hr lifetime, hosts network fields.
- **Intergranular lane B**: ~1 kG vertical, up to 1.7 kG (p_in + B²/8π = p_out limit).
- **Quiet-Sun horizontal field** (Lites et al. 2008): $\langle B_h\rangle = 55$–60 G; vertical $\langle B_v\rangle = 11$ G.
- **Flux distribution**: power law slope –1.85 from $10^{17}$ to $10^{23}$ Mx (Parnell et al. 2009).
- **Emerging flux rate**: $\sim 3\times 10^{-10}$ Mx km$^{-2}$ s$^{-1}$ in quiet Sun (Thornton & Parnell 2011); three orders of magnitude more than active regions.
- **Dynamo scales**: 95% stretching at 0.1–1 Mm → 20–200 km; 5% compression; MHD wave leakage.
- **Wilson depression**: ~200 km (network), ~500 km (sunspot umbra).
- **G-band formation height**: ~54 km above τ=1 (log τ=1 ≈ −0.48).
- **Facular "hot-wall" effect**: excess brightness from a ~30-km-thick density-gradient layer.
- **Sunspot umbral dots** (Schüssler & Vögler 2006): plume velocities up to 2.7 km/s.
- **Penumbra threshold**: inclination >45° needed for Evershed-flow filaments (Rempel 2011).
- **Domain sizes**: Cheung et al. (2010) used 92×49 Mm wide, Rempel (2011) sunspot at 98×49×6 Mm.

**한국어**: 리뷰 전반에 Stein은 많은 정량 기준점을 박아 두었다. 정리하면:
- **입상반 크기**: ~1 Mm 너비, 수명 ~300초, 속도 2–3 km/s.
- **메조입상반**: ~5 Mm, 수명 수 시간 — 관측상으로만 식별, 직접 대류 구동자는 없음.
- **초입상반**: ~30 Mm, 수명 ~24시간, 네트워크 자기장 호스트.
- **입상반 간 레인 B**: 수직 ~1 kG, 최대 1.7 kG(p_in + B²/8π = p_out 한계).
- **정적태양 수평장**(Lites et al. 2008): $\langle B_h\rangle = 55$–60 G, 수직 $\langle B_v\rangle = 11$ G.
- **자속 분포**: $10^{17}$–$10^{23}$ Mx 범위에서 기울기 –1.85 멱법칙.
- **출현 자속률**: 정적태양에서 $\sim 3\times 10^{-10}$ Mx km$^{-2}$ s$^{-1}$; 활동영역보다 3자릿수 많음.
- **다이나모 스케일**: 0.1–1 Mm에서 신장 95% → 20–200 km, 압축 5%, MHD 파 누출.
- **Wilson 함몰**: 네트워크 ~200 km, 흑점 umbra ~500 km.
- **G-band 형성 고도**: τ=1 위 ~54 km (log τ=1 ≈ −0.48).
- **광반 "hot-wall"**: 두께 ~30 km의 밀도 경사층으로부터 초과 밝기.
- **흑점 umbral dot**(Schüssler & Vögler 2006): 플룸 속도 최대 2.7 km/s.
- **반영 임계**: Evershed 유출 필라멘트 형성에 경사 >45° 필요(Rempel 2011).
- **도메인 크기**: Cheung et al. (2010) 92×49 Mm 너비, Rempel (2011) 흑점 98×49×6 Mm.

---

## 3. Key Takeaways / 핵심 시사점

1. **Realistic simulations have crossed a threshold** — **현실적 시뮬레이션은 임계점을 넘었다**. With tabular EOS + non-grey radiative transfer + full compressibility, R-MHD simulations now reproduce granulation contrast, Stokes profiles, flux-emergence hierarchies, and sunspot morphology to quantitative accuracy, shifting magneto-convection from a theoretical sketch to a predictive tool. / 표 EOS + 비회색 복사 + 완전 압축성으로 구성된 R-MHD 시뮬레이션은 입상반 대비, Stokes 프로파일, 자속 출현 계층, 흑점 형태를 정량적으로 재현하게 되었고, 이로써 자기대류는 이론적 스케치에서 예측 도구로 전환되었다.

2. **Stratification asymmetry is fundamental** — **성층 비대칭은 근본적이다**. Broad laminar upflows and narrow turbulent downflows (necessitated by mass conservation in a stratified atmosphere) seed every phenomenon Stein discusses: dynamo action in downdrafts, flux pumping downward, intergranular concentration, micropore nucleation at lane vertices. / 성층 대기의 질량보존이 만드는 넓은 층류 상승류 + 좁은 난류 하강류의 비대칭이 이 논문의 모든 현상(하강류 내 다이나모, 자속 펌핑, 입상반 간 집중, 레인 꼭짓점의 마이크로포어 핵생성)의 씨앗이다.

3. **Kilogauss concentrations are super-equipartition** — **kG 집중은 등분배를 초과한다**. Convective intensification amplifies B far beyond $B_{\text{eq}}\sim 400\text{–}500$ G near τ=1. The end state is set by total pressure balance $p_{\text{in}}+B^2/8\pi=p_{\text{out}}$, not by flux freezing or kinetic balance — a genuinely thermodynamic equilibrium. / 대류적 강화는 τ=1 근처의 $B_{\text{eq}}\sim 400\text{–}500$ G를 훨씬 초과하도록 B를 증폭한다. 최종 상태는 자속 동결이나 운동 균형이 아닌 전체 압력 균형 $p_{\text{in}}+B^2/8\pi=p_{\text{out}}$으로 결정되는 진정한 열역학적 평형이다.

4. **Wilson depression links dynamics to photometry** — **윌슨 함몰은 동역학과 측광을 연결한다**. Evacuated columns depress the τ=1 surface by ~200–500 km. Narrow concentrations ("hot-wall" heated) appear bright as G-band points; wide concentrations are dark as pores/sunspots. The same physics explains facular limb brightening. / 배기된 기둥은 τ=1 면을 ~200–500 km 가라앉힌다. 좁은 집중("hot-wall" 가열)은 G-band point로 밝게, 넓은 집중은 포어·흑점으로 어둡게 나타난다. 같은 물리가 광반 주변부 밝기 상승도 설명한다.

5. **Surface dynamo is driven by stretching** — **표면 다이나모는 신장으로 구동된다**. Pietarila Graham et al. (2010) pinned down that 95% of kinematic dynamo energy input comes from turbulent stretching at sub-granule scales (0.1–1 Mm) in downdrafts, only 5% from compression. This explains why the quiet Sun contains three orders of magnitude more emerging flux than active regions — dynamo action is ubiquitous in downflows. / Pietarila Graham et al. (2010)은 운동학적 다이나모 에너지 입력의 95%가 하강류 내 sub-granule 규모(0.1–1 Mm)의 난류 신장에서 오며, 5%만 압축에서 옴을 확정했다. 이것이 정적태양에서 활동영역보다 3자릿수 더 많은 자속이 출현하는 이유를 설명한다 — 하강류 어디에나 다이나모가 있다.

6. **Flux emergence is inherently filamentary** — **자속 출현은 본질적으로 필라멘트성이다**. A rising coherent tube fragments as it meets downdrafts on smaller and smaller scales, producing the "pepper-and-salt" mixed-polarity pattern before collecting into unipolar regions. The Ω/U-loop hierarchy is not optional — it is the topological consequence of stratified convection draping over buoyant flux. / 상승하는 결맞는 자속관은 점점 작은 스케일의 하강류를 만나며 조각나 "pepper-and-salt" 혼합 극성 패턴을 만든 뒤 단극 영역으로 모인다. Ω/U-loop 계층은 선택이 아니라 부력 자속 위로 드리워진 성층 대류의 위상학적 귀결이다.

7. **Pores and sunspots need no ab-initio origin** — **포어와 흑점은 ab-initio 기원이 필수는 아니다**. Cheung et al. (2010) grew a pair of sunspots from an emerging twisted half-torus; Rempel (2011) produced penumbrae with Evershed flow from axisymmetric self-similar initial conditions. But no simulation has yet produced a sunspot starting from zero large-scale flux — the ab-initio sunspot remains an open challenge. / Cheung et al. (2010)은 출현하는 꼬인 half-torus로 흑점 쌍을 키웠고, Rempel (2011)은 축대칭 자기상사 초기조건으로 Evershed 유출을 가진 반영을 만들어냈다. 그러나 대규모 자속 없이 0부터 흑점을 만드는 ab-initio 시뮬레이션은 아직 존재하지 않는다 — 여전히 열린 과제다.

8. **Simulations are now indispensable observational interpreters** — **시뮬레이션은 이제 관측의 필수 해석자다**. Stokes inversion codes are calibrated on simulation snapshots (Khomenko et al. 2005; Shelyag et al. 2007); helioseismic inversions use simulation kernels (Zhao et al. 2007; Braun et al. 2012); abundance determinations depend on 3D convection models. The separation between "theory" and "observation" has largely dissolved. / Stokes 역해석 코드는 시뮬레이션 스냅샷으로 보정되고(Khomenko et al. 2005; Shelyag et al. 2007), 태양 지진학 역해석은 시뮬레이션 커널을 사용하며(Zhao et al. 2007; Braun et al. 2012), 원소 존재량 결정은 3D 대류 모델에 의존한다. "이론"과 "관측"의 구분은 거의 사라졌다.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 R-MHD Conservation Laws / R-MHD 보존 법칙

**Continuity / 연속방정식**:
$$\frac{\partial \rho}{\partial t} = -\nabla\cdot(\rho\mathbf{u})$$
- $\rho$: mass density [g cm$^{-3}$] / 질량 밀도
- $\mathbf{u}$: velocity [cm s$^{-1}$] / 속도

**Momentum / 운동량**:
$$\frac{\partial(\rho\mathbf{u})}{\partial t} = -\nabla\cdot(\rho\mathbf{u}\mathbf{u}) - \nabla P - \rho\mathbf{g} + \mathbf{J}\times\mathbf{B} - 2\rho\boldsymbol{\Omega}\times\mathbf{u} - \nabla\cdot\boldsymbol{\tau}_{\text{visc}}$$
- $P$: thermal pressure / 열역학 압력
- $\mathbf{J}=\nabla\times\mathbf{B}/\mu$: current / 전류
- $\mathbf{J}\times\mathbf{B}$: Lorentz force, which can be split into tension $(\mathbf{B}\cdot\nabla)\mathbf{B}/\mu$ and magnetic pressure $-\nabla(B^2/2\mu)$ / 로런츠 힘 = 자기장력 + 자기압 gradient

**Viscous stress / 점성응력**:
$$\tau_{ij} = \mu\left(\frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i} - \frac{2}{3}\nabla\cdot\mathbf{u}\,\delta_{ij}\right)$$

**Internal energy / 내부에너지**:
$$\frac{\partial e}{\partial t} = -\nabla\cdot(e\mathbf{u}) - P(\nabla\cdot\mathbf{u}) + Q_{\text{rad}} + Q_{\text{visc}} + \eta J^2$$

**Radiative heating / 복사 가열**:
$$Q_{\text{rad}} = \int_\nu\!\!\int_\Omega \rho\kappa_\nu(I_\nu - S_\nu)\,d\Omega\,d\nu$$
- $\kappa_\nu$: opacity / 불투명도
- $S_\nu = \epsilon_\nu/\kappa_\nu$: source function / 원천함수

**Radiative transfer / 복사전달**:
$$\frac{\partial I_\nu}{\partial \ell} = \epsilon_\nu - \chi_\nu I_\nu$$

**Induction & Ohm's law / 유도방정식과 옴의 법칙**:
$$\frac{\partial \mathbf{B}}{\partial t} = -\nabla\times\mathbf{E}, \qquad \mathbf{E} = -\mathbf{u}\times\mathbf{B} + \eta\mathbf{J} + \frac{1}{en_e}(\mathbf{J}\times\mathbf{B} - \nabla P_e)$$

Substituting (neglecting Hall/pressure): $\partial_t\mathbf{B} = \nabla\times(\mathbf{u}\times\mathbf{B}) - \nabla\times(\eta\nabla\times\mathbf{B})$.

### 4.2 Characteristic Dimensionless Numbers / 특성 무차원 수

**Mach number / 마하수**:
$$\mathrm{Ma} = \frac{u}{c_s}, \qquad c_s = \sqrt{\frac{\gamma P}{\rho}}$$
Near τ=1: $u\sim 2\text{–}3$ km/s, $c_s\sim 7$ km/s → Ma $\sim 0.3$–0.4 (weakly supersonic in downdrafts).

**Reynolds number (effective) / 유효 레이놀즈 수**:
$$\mathrm{Re} = \frac{uL}{\nu}\sim 10^{11}\text{–}10^{12}\text{ (solar)}, \quad \mathrm{Re}_{\text{sim}}\sim 10^{3}\text{–}10^{4}$$

**Magnetic Reynolds number / 자기 레이놀즈 수**:
$$\mathrm{Rm} = \frac{uL}{\eta_m}\sim 10^{9}\text{ (solar)}, \quad \mathrm{Rm}_{\text{sim}}\sim 10^{3}\text{–}10^{4}$$

**Prandtl numbers / Prandtl 수**:
$$\mathrm{Pr} = \nu/\kappa_{\text{th}}, \qquad \mathrm{Pm} = \nu/\eta_m$$

### 4.3 Flux Expulsion & Frozen-Flux / 자속 축출과 동결

In the high-conductivity limit ($\eta\to 0$):
$$\frac{D}{Dt}\left(\frac{\mathbf{B}}{\rho}\right) = \left(\frac{\mathbf{B}}{\rho}\cdot\nabla\right)\mathbf{u}$$

For 2D axisymmetric converging flow $\mathbf{u} = -\alpha r\,\hat{r}$ with vertical seed field $B_z$:
$$\frac{dB_z}{dt} = -B_z\nabla\cdot\mathbf{u}_\perp = 2\alpha B_z \Rightarrow B_z(t) = B_{z,0}e^{2\alpha t}$$
(exponential amplification until Lorentz back-reaction saturates the flow; Galloway & Weiss 1981).

### 4.4 Flux-Tube Pressure Balance / 자속관 압력 균형

$$p_{\text{in}} + \frac{B^2}{8\pi} = p_{\text{out}}$$

For a fully evacuated tube at τ=1 ($p_{\text{out}}\sim 1.2\times 10^5$ dyne cm$^{-2}$), neglecting $p_{\text{in}}$:
$$B_{\max} = \sqrt{8\pi p_{\text{out}}}\sim 1.7\text{ kG}$$
This matches observed kG network fields. Wilson depression (τ=1 depth shift) follows from hydrostatic rescaling with reduced opacity.

### 4.5 Expansion of Rising Flux Tube / 상승 자속관의 팽창

With $\alpha=\partial_x v_x = \partial_y v_y$ and $\partial_z v_z = \epsilon\alpha$:
$$\frac{D\ln B}{Dt} = -(1+\epsilon)\alpha, \qquad \frac{D\ln\rho}{Dt} = -(2+\epsilon)\alpha$$
$$\Rightarrow B\propto \rho^{(1+\epsilon)/(2+\epsilon)}$$
Isotropic ($\epsilon=1$): $B\propto\rho^{2/3}$. Horizontal-dominated ($\epsilon\ll 1$): $B\propto\rho^{1/2}$.

### 4.6 Kink Instability Criterion / Kink 불안정성 조건

A twisted flux tube of length $L$, radius $a$, and axial field $B_z$ with twist $\Phi=B_\phi L/(aB_z)$ is kink-unstable (ideal MHD) when:
$$\Phi > \Phi_c \approx 2\pi$$
equivalently when the number of twists per $L$ exceeds unity. For rising flux tubes, Linton et al. (1996) gave:
$$q = \frac{B_\phi(a)}{aB_z(0)} > q_c \sim \frac{1}{a}$$
Thin tubes kink easily; fat tubes are more stable.

### 4.7 Mixing-Length Convective Velocity / 혼합길이 대류 속도

Enthalpy flux balance near surface:
$$F_\odot = \rho u_{\text{conv}} c_p \Delta T \approx \rho u_{\text{conv}}^3 \cdot f$$
with $f$ an order-unity factor when $\Delta T \sim T (u/c_s)^2$. Taking $F_\odot = 6.3\times 10^{10}$ erg cm$^{-2}$ s$^{-1}$, $\rho = 3\times10^{-7}$ g cm$^{-3}$:
$$u_{\text{conv}} \approx (F_\odot/\rho)^{1/3} \approx 2.7\text{ km/s}$$

### 4.8 Equipartition Field / 등분배 자기장

$$\frac{B_{\text{eq}}^2}{8\pi} = \frac{1}{2}\rho u_{\text{conv}}^2 \Rightarrow B_{\text{eq}} = (4\pi\rho)^{1/2} u_{\text{conv}}$$
With the above values: $B_{\text{eq}}\sim 400$–500 G.

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1896 ──── Janssen photographs granulation
            (입상반의 사진 기록)
1938 ──── Biermann: convective energy transport in Sun
1958 ──── Parker: flux tube concept, magnetic buoyancy
1961 ──── Leighton: chromospheric network (Ca II K)
1966 ──── Weiss: first magneto-convection simulation (2D, Boussinesq)
            (최초의 자기대류 시뮬레이션)
1978 ──── Parker: convective intensification / flux tube collapse
1979 ──── Spruit: flux tube equilibrium & evacuation
1981 ──── Galloway & Weiss: flux expulsion dynamics
1982 ──── Nordlund: first 3D realistic solar granulation simulation
            (최초의 3D 현실적 태양 입상반)
1989 ──── Stein & Nordlund: topology of stratified convection
1998 ──── Stein & Nordlund: ionization energy drives solar convection
2002 ──── Bercik: micropore formation in R-MHD
2005 ──── Vögler et al. (MURaM): realistic solar surface magneto-convection
2006 ──── Stein & Nordlund: advected horizontal field → pepper-and-salt
            Schüssler & Vögler: umbral dot simulation
2007 ──── Cheung et al.: twisted flux tube emergence
            Vögler & Schüssler: solar surface dynamo confirmed
            Abbett: local dynamo across full convection zone
2008 ──── Lites et al. (Hinode): quiet-Sun horizontal field 55 G
            Schüssler & Vögler: shallow surface dynamo saturation
2009 ──── Parnell et al.: power-law flux distribution slope –1.85
            Martínez González & Bellot Rubio: quiet-Sun Ω-loop imaging
            Nordlund, Stein & Asplund: Living Reviews on solar convection
2010 ──── Stein et al. (Nature): advected flux emergence in large domain
            Pietarila Graham et al.: 95% stretching / 5% compression dynamo
            Cheung et al.: sunspot pair from emerging half-torus
2011 ──── Thornton & Parnell: log-normal flux distribution
            Rempel: self-similar sunspot + penumbra + Evershed flow
2012 ──── STEIN: THIS REVIEW ★
            (Living Reviews — synthesis of realistic magneto-convection)
─────────── after this paper ──────────
2014 ──── Rempel: full sunspot R-MHD with penumbral fine structure
2016 ──── Cheung & Isobe: flux emergence reviews
2020s ─── BIFROST / MURaM-corona coupling; machine learning emulators
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **Paper #16 Rempel & Schlichenmaier (2011)** — "Sunspot Modelling: From Simplified Models to Radiative MHD Simulations" | Companion review focused specifically on sunspots; Stein cites it for sunspot details. / 흑점에 특화된 자매 리뷰. Stein이 흑점 세부사항에 대해 직접 인용. | Very high / 매우 높음 — directly complementary scope |
| **Nordlund, Stein & Asplund (2009)** — "Solar Surface Convection" (LRSP) | The non-magnetic convection review that this paper explicitly updates with magnetic content. / Stein이 자기 내용을 보완하는 비자기 대류 리뷰. | Very high / 매우 높음 — direct predecessor |
| **Fan (2009)** — "Magnetic Fields in the Solar Convection Zone" (LRSP) | Reviews deep-interior rise of flux; Stein focuses on last scale-height. / 심부 자속 상승에 관한 리뷰. Stein은 마지막 스케일 높이에 집중. | High / 높음 — complementary depth regimes |
| **Charbonneau (2010)** — "Dynamo Models of the Solar Cycle" (LRSP) | Global dynamo theory; Stein covers local surface dynamo. / 전역 다이나모 이론. Stein은 국소 표면 다이나모 담당. | High / 높음 — two halves of dynamo problem |
| **Miesch (2005)** — "Large-Scale Dynamics of the Convection Zone and Tachocline" (LRSP) | Deep convection zone dynamics (rotation, differential rotation). / 심부 대류층 동역학(자전, 차등 자전). | Medium / 중간 — sets the deep boundary condition |
| **Hathaway (2010)** — "The Solar Cycle" (LRSP) | Provides butterfly diagram, Hale/Joy laws that surface magneto-convection must be consistent with. / 나비 다이어그램, Hale·Joy 법칙 제공 — 표면 자기대류가 일관되어야 함. | Medium / 중간 — observational context |
| **Gizon & Birch (2005)** — "Local Helioseismology" (LRSP) | Uses simulation outputs for inversion kernels. / 역해석 커널에 시뮬레이션 출력 사용. | Medium / 중간 — observational diagnostic |
| **de Wijn et al. (2009)** — "Small-Scale Solar Magnetic Fields" | Observational review; cited heavily for Sec. 3 observations. / 3장 관측 내용에 대거 인용된 관측 리뷰. | High / 높음 — observational underpinning |
| **Parker (1978) / Spruit (1979)** — Convective intensification | Analytical foundation for kG flux tube physics. / kG 자속관 물리의 해석적 기초. | High / 높음 — original theory |

---

## 7. References / 참고문헌

- Stein, R. F., "Solar Surface Magneto-Convection", Living Reviews in Solar Physics, **9**, 4 (2012). DOI: 10.12942/lrsp-2012-4
- Nordlund, Å., Stein, R. F., Asplund, M., "Solar Surface Convection", Living Rev. Solar Phys. **6**, 2 (2009).
- Vögler, A., Shelyag, S., Schüssler, M., Cattaneo, F., Emonet, T., Linde, T., "Simulations of magneto-convection in the solar photosphere. Equations, methods, and results of the MURaM code", A&A **429**, 335 (2005).
- Stein, R. F., Nordlund, Å., "Solar small-scale magnetoconvection", ApJ **642**, 1246 (2006).
- Cheung, M. C. M., Schüssler, M., Tarbell, T. D., Title, A. M., "Solar surface emerging flux regions: A comparative study of radiative MHD modeling and Hinode SOT observations", ApJ **687**, 1373 (2008).
- Cheung, M. C. M., Rempel, M., Title, A. M., Schüssler, M., "Simulation of the formation of a solar active region", ApJ **720**, 233 (2010).
- Rempel, M., Schüssler, M., Cameron, R. H., Knölker, M., "Penumbral structure and outflows in simulated sunspots", Science **325**, 171 (2009).
- Rempel, M., "Penumbral fine structure and driving mechanisms of large-scale flows in simulated sunspots", ApJ **729**, 5 (2011).
- Rempel, M., Schlichenmaier, R., "Sunspot Modeling: From Simplified Models to Radiative MHD Simulations", Living Rev. Solar Phys. **8**, 3 (2011).
- Pietarila Graham, J., Cameron, R., Schüssler, M., "Turbulent Small-scale Dynamo Action in Solar Surface Simulations", ApJ **714**, 1606 (2010).
- Schüssler, M., Vögler, A., "Magnetoconvection in a sunspot umbra", ApJ **641**, L73 (2006).
- Parker, E. N., "Hydraulic concentration of magnetic fields in the solar photosphere", ApJ **221**, 368 (1978).
- Spruit, H. C., "Convective collapse of flux tubes", Solar Phys. **61**, 363 (1979).
- Stein, R. F., Nordlund, Å., "Simulations of Solar Granulation. I. General properties", ApJ **499**, 914 (1998).
- Parnell, C. E., DeForest, C. E., Hagenaar, H. J., Johnston, B. A., Lamb, D. A., Welsch, B. T., "A power-law distribution of solar magnetic fields over more than five decades in flux", ApJ **698**, 75 (2009).
- Thornton, L. M., Parnell, C. E., "Small-scale flux emergence observed using Hinode/SOT", Solar Phys. **269**, 13 (2011).
- Lites, B. W. et al., "The horizontal magnetic flux of the quiet-Sun internetwork as observed with the Hinode spectro-polarimeter", ApJ **672**, 1237 (2008).
- Martínez González, M. J., Bellot Rubio, L. R., "Emergence of small-scale magnetic loops through the quiet solar atmosphere", ApJ **700**, 1391 (2009).
- Charbonneau, P., "Dynamo Models of the Solar Cycle", Living Rev. Solar Phys. **7**, 3 (2010).
- Fan, Y., "Magnetic Fields in the Solar Convection Zone", Living Rev. Solar Phys. **6**, 4 (2009).
- Galloway, D. J., Weiss, N. O., "Convection and magnetic fields in stars", ApJ **243**, 945 (1981).
