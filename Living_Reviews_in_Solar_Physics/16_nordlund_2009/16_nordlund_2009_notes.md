---
title: "Solar Surface Convection"
authors: Åke Nordlund, Robert F. Stein, Martin Asplund
year: 2009
journal: "Living Reviews in Solar Physics"
doi: "10.12942/lrsp-2009-2"
topic: Living_Reviews_in_Solar_Physics
tags: [solar convection, granulation, supergranulation, radiative hydrodynamics, spectral line formation, solar abundances, helioseismology, MHD, coronal heating]
status: completed
date_started: 2026-04-16
date_completed: 2026-04-16
---

# 16. Solar Surface Convection / 태양 표면 대류

---

## 1. Core Contribution / 핵심 기여

이 리뷰 논문은 태양 표면 대류(solar surface convection)에 관한 포괄적 총설로, 가시 태양 표면(광구)에서 직접 관측 가능한 대류 현상의 물리학을 체계적으로 정리합니다. 논문은 온도 극소 영역에서 약 20 Mm 깊이까지의 범위를 다루며, 이 영역에서 밀도와 압력이 각각 $10^6$배, $10^8$배 변하는 극한 조건을 분석합니다. 핵심적으로, 3D 복사-유체역학(radiative-hydrodynamic) 시뮬레이션이 관측된 스펙트럼 선 프로파일, 과립 패턴, 속도장을 놀라울 정도로 잘 재현함을 보여줍니다. 이를 기반으로 태양 화학 조성(C, N, O 풍부도)의 대폭적 하향 수정이라는 역사적으로 중대한 결과를 제시하며, helioseismology, 자기장 상호작용, 코로나 가열 등 광범위한 응용 분야를 다룹니다.

This comprehensive review systematically covers the physics of solar convection directly observable at the solar surface (photosphere), from the temperature minimum down to about 20 Mm below. In this range, density and pressure vary by factors of $10^6$ and $10^8$ respectively — an extreme regime where ionization, molecular dissociation, and the transition from convective to radiative energy transport all play crucial roles. The central achievement demonstrated in this review is that 3D radiative-hydrodynamic supercomputer simulations match observational constraints with remarkable fidelity — reproducing spectral line profiles, granulation patterns, and velocity fields without any adjustable parameters like micro/macroturbulence. Building on this success, the review presents the historically significant downward revision of solar C, N, and O abundances (~0.15–0.25 dex lower than previous values), which triggered the "solar abundance crisis" — a still-unresolved conflict with helioseismology. The review also covers applications to helioseismology (wave propagation, p-mode excitation and frequencies), the interaction of convection with magnetic fields (faculae, pores, sunspots, flux emergence), and convection as a driver of chromospheric and coronal heating via Poynting flux.

---

## 2. Reading Notes / 읽기 노트

### §1 Introduction / 도입 (pp. 7–8)

태양 대류는 태양의 구조와 외형 모두에 중심적 중요성을 가집니다. 이 리뷰는 태양 표면에서 관측 가능한 것에 직접 영향을 미치는 층, 즉 온도 극소 영역에서 가시 표면 아래 약 20 Mm까지를 다룹니다. 이 깊이 20 Mm은 태양 반경의 약 3%, 대류층의 약 10%에 해당하지만, 압력/밀도 범위의 약 절반을 포함합니다.

Convection is of central importance to both the structure and appearance of the Sun. This review focuses on layers from the temperature minimum down to ~20 Mm below the visible surface. While this corresponds to only ~3% of the solar radius and ~10% of the convection zone depth, it spans roughly half the range of pressure or density. In these surface and near-surface layers, the equation of state is strongly influenced by ionization and molecular dissociation — unlike the deeper interior where the gas is nearly fully ionized and follows a nearly ideal polytrope.

태양은 정량적 비교를 위한 이상적 대상입니다: 시간, 공간, 파장 영역에서 5차원적 관측이 가능하며, 근접성과 겉보기 광도 덕분에 극자외선부터 원적외선까지 스펙트럼을 완전히 분해할 수 있습니다.

The Sun is an ideal object for *quantitative and accurate* comparisons between numerical simulations and observations — arguably in five dimensions: time, spatial domain over a large range of scales, and wavelength from extreme UV to far infrared.

### §2 Hydrodynamics of Solar Convection / 태양 대류의 유체역학 (pp. 9–14)

#### §2.1 Mass Conservation / 질량 보존

연속 방정식의 Euler 형태와 Lagrange 형태:

$$\frac{\partial \rho}{\partial t} = -\nabla \cdot (\rho \mathbf{u}) \quad \text{(Eulerian)} \tag{1}$$

$$\frac{D \ln \rho}{Dt} = -\nabla \cdot (\mathbf{u}) \quad \text{(Lagrangian)} \tag{2}$$

Lagrange 형태에서: 유체 덩어리가 여러 밀도 눈금높이를 상승하면 그에 상응하는 팽창이 필요합니다. Euler 형태에서: 국소 밀도가 시간에 따라 크게 변하지 않으면, 높이에 따른 평균 밀도 감소를 빠른 수평 팽창으로 보상할 수 있습니다.

From the Lagrangian form: a fluid parcel ascending over several density scale heights must expand correspondingly. From the Eulerian form: if local density doesn't change much with time, the rapid decrease of mean density with height can be balanced by rapid sideways expansion. This fundamental constraint dictates the topology of solar convection.

**Anelastic approximation** (Gough, 1969): $\frac{\partial \rho}{\partial t} \approx 0$으로 놓으면, $\rho \approx \rho(z,t)$이고:

$$u_z \frac{\partial \ln \rho}{\partial z} + \frac{\partial u_z}{\partial z} \approx -\frac{\partial u_x}{\partial x} - \frac{\partial u_y}{\partial y} \tag{4}$$

$\ln \rho$가 높이에 따라 급변하면 상승/하강 유체는 빠르게 팽창/수축해야 합니다.

#### §2.2 Equations of Motion / 운동 방정식

$$\frac{\partial (\rho \mathbf{u})}{\partial t} = -\nabla \cdot (\rho \mathbf{u}\mathbf{u}) - \nabla P - \rho \nabla \Phi - \nabla \cdot \tau_{\text{visc}} \tag{5}$$

Lagrange 형태:

$$\frac{D\mathbf{u}}{Dt} = -\frac{P}{\rho}\nabla \ln P - \nabla \Phi - \frac{1}{\rho}\nabla \cdot \tau_{\text{visc}} \tag{6}$$

**정역학적 평형(hydrostatic equilibrium)**에서:

$$-\frac{P}{\rho}\frac{\partial \ln P}{\partial z} = \frac{\partial \Phi}{\partial z} \equiv g_z \tag{7}$$

일정한 $P/\rho$ (일정 온도)에서 압력은 지수적으로 감소:

$$P = P_0 e^{-z/H_P}, \quad H_P = \frac{P}{\rho g_z} \tag{8, 9}$$

중요한 결과: 수평 압력 기울기 $\nabla_\perp \ln P \approx \nabla_\perp \ln P_0(x,y)$는 높이에 거의 무관합니다. 이는 수평 속도장이 높이에 따라 천천히 변하는 이유를 설명합니다.

An important consequence: horizontal pressure gradients are essentially independent of height, which explains why horizontal velocity fields vary slowly with height.

Anelastic 근사에서 압력에 대한 Poisson 방정식:

$$\nabla^2 P = \nabla \cdot [-\rho \mathbf{u} \cdot \nabla \mathbf{u} - \rho \nabla \Phi - \nabla \cdot (\tau_{\text{visc}})] \tag{12}$$

이 근사에서 압력은 단위 부피당 힘에 의해 순간적으로 결정되며, 구속 조건 $\nabla \cdot (\rho \mathbf{u}) = 0$을 강제합니다.

#### §2.3 Kinetic Energy, Buoyancy Work, Gas Pressure Work / 운동 에너지, 부력 일, 기체 압력 일

운동 에너지 방정식:

$$\frac{\partial E_{\text{kin}}}{\partial t} = -\nabla \cdot (E_{\text{kin}} \mathbf{u}) - \mathbf{u} \cdot \nabla P - \rho \mathbf{u} \cdot \nabla \Phi + \text{viscous terms} \tag{13}$$

**부력 일(buoyancy work)**: $\rho' u_z' g_z$. 평균보다 무거운 유체가 하강하고 가벼운 유체가 상승하면 양의 부력 일이 생깁니다. 그러나 질량 보존 $0 = \langle \rho u_z \rangle$에 의해:

$$\langle \rho \rangle \langle u_z \rangle = -\langle \rho' u_z' \rangle \tag{16}$$

따라서 전체 부력 일은 0으로 사라집니다! 이 역설의 해결: 부력은 대류 셀에 양의 일을 하지만, 동시에 평균 흐름에 대한 중력의 음의 일이 정확히 상쇄합니다. 실질적으로 점성 소산을 균형잡는 것은 **기체 압력 일**입니다:

$$\langle -\mathbf{u} \cdot \nabla P \rangle = \langle P \nabla \cdot \mathbf{u} \rangle = \langle P \nabla \cdot \mathbf{u} \rangle \tag{17}$$

가스가 상승할 때(팽창) 압력이 하강할 때(압축)보다 높으므로 양의 일이 됩니다.

The buoyancy work paradox: total buoyancy work vanishes due to mass conservation. The resolution is that buoyancy does positive work on convective motions, but there is an equally large negative work by gravity on the mean flow. The net work against viscous dissipation is balanced by **gas pressure work** — gas expands at higher pressure (on the way up) than it is compressed (on the way down).

#### §2.4 Energy Transport / 에너지 수송

내부 에너지 진화 방정식:

$$\frac{\partial E}{\partial t} = -\nabla \cdot (E\mathbf{u}\mathbf{u}) - P(\nabla \cdot \mathbf{u}) + Q_{\text{rad}} + Q_{\text{visc}} \tag{18}$$

여기서 $P(\nabla \cdot \mathbf{u})$는 PdV 일(단열 가열/냉각)입니다.

**복사 에너지 전달**: 복사 가열/냉각률:

$$Q_{\text{rad}} = -\nabla \cdot \mathbf{F}_{\text{rad}} = \int_\nu \int_\Omega \rho \kappa_\nu (I_\nu - S_\nu) \, d\mathbf{\Omega} \, d\nu \tag{25, 29}$$

$I_\nu < S_\nu$인 곳에서 냉각이 일어납니다 — 특히 광학적 깊이 단위 근처의 표면층에서 강합니다. **Opacity binning** (Nordlund, 1982): 주파수 적분을 효율적으로 근사하는 핵심 수치 기법.

Radiative cooling is strongest near optical depth unity, where the outgoing intensity is similar to the source function but incoming intensity is much lower (seeing a "dark sky" from the optically thin atmosphere above). The opacity binning method groups frequencies into bins based on opacity level, enabling efficient computation.

**상태 방정식**: 표면 근처에서 수소와 헬륨의 이온화/해리가 중요하여, 이상 기체 근사가 적용되지 않습니다. $P = P(\rho, e)$와 $\kappa_\nu = \kappa_\nu(\rho, e)$를 테이블로 미리 계산하여 사용합니다.

**에너지 플럭스**: 총 에너지 보존에서:

$$\frac{\partial (E + E_{\text{kin}})}{\partial t} = -\nabla \cdot (\mathbf{F}_{\text{conv}} + \mathbf{F}_{\text{kin}} + \mathbf{F}_{\text{rad}} + \mathbf{F}_{\text{visc}}) \tag{32}$$

- $\mathbf{F}_{\text{conv}} = (E+P)\mathbf{u}$ — 대류(엔탈피) 플럭스
- $\mathbf{F}_{\text{kin}} = \frac{1}{2}\rho u^2 \mathbf{u}$ — 운동 에너지 플럭스
- 표면 근처에서 내부 에너지가 이온화 에너지에 의해 지배되므로 대류 플럭스가 우세

At the solar surface there is a very rapid transition from primarily convective to primarily radiative energy transport.

### §3 Granulation / 과립 (pp. 15–34)

#### §3.1 Observational Constraints / 관측적 제약

태양 과립 패턴은 1801년 Herschel이 최초 관측했으며, Nasmyth(1865)는 "버드나무 잎" 패턴으로 묘사, Dawes(1864)가 "granule" 용어를 도입, Janssen(1896)의 사진으로 논란이 종결되었습니다. 과립은 약 1000 km (1 Mm) 크기이며 열 수송에 의한 대류를 나타냅니다.

관측적 제약의 핵심 역설: **가장 좋은 관측적 제약은 공간적으로 완전히 분해되지 않은 관측에서 옵니다** — 스펙트럼 선의 폭, 형태, 세기가 수치 모델과의 정밀 비교를 가능하게 합니다. 직접 이미지는 기기 분해능, 대기 효과, 산란광에 의해 피할 수 없는 열화를 겪습니다.

The best observational constraints paradoxically come from spatially *unresolved* observations — spectral line widths, shapes, and strengths allow detailed quantitative comparisons with 3D models. Direct imaging is unavoidably affected by instrumental resolution, atmospheric blurring, and scattered light.

수치 모델들의 rms intensity fluctuation은 연속광 강도의 1–2%로 일치합니다. Hinode와의 비교도 본질적으로 일관됩니다 (Danilovic et al., 2008).

#### §3.2 Convective Driving / 대류 구동

대류는 주로 태양 가시 표면의 얇은 열적 경계층에서의 **복사 냉각**에 의해 구동됩니다 (표면 아래에서 올라오는 가열이 아닌!). 밝은 과립은 상승하는 뜨거운 플라즈마, 어두운 intergranular lane은 하강하는 차가운 플라즈마입니다. 과립의 직경은 약 1 Mm입니다.

핵심 메커니즘 (Stein & Nordlund, 1998):
1. 플라즈마가 표면에 도달하면 광자가 열 에너지를 방출 → 냉각
2. 수소 이온이 전자를 포획하여 중성이 됨 → 이온화 에너지도 방출
3. 엔트로피가 급격히 감소 → 밀도가 증가하여 과밀 유체 생성
4. 과밀 유체가 중력에 의해 하강 → **부력 일이 주로 차가운 하강류에서 발생**

Convection is driven primarily by **radiative cooling** at a thin thermal boundary layer at the visible surface. The key process: plasma reaches the surface → photons carry away thermal energy → hydrogen ions recombine releasing ionization energy → entropy drops sharply → overdense fluid sinks. Most buoyancy work occurs in the cool downflows, not the warm upflows (Figure 4). At greater depths, buoyancy work increasingly occurs in cool downflows rather than warm upflows.

#### §3.3 Scale Selection / 크기 선택

밀도의 급격한 변화가 대류 스케일을 결정합니다. 질량 보존에 의해, 유체 덩어리가 하나의 밀도 눈금높이를 오르내릴 때 $e$배의 팽창/수축이 필요합니다.

과립의 수평 크기:

$$r = 2H(v_H / v_z) \tag{37}$$

여기서 $H$는 눈금높이, $v_H$는 수평 속도, $v_z$는 수직 속도. 표면에서의 최소 수직 속도 추정:

$$\sigma T_{\text{eff}}^4 \approx \rho V_z \left(\frac{5}{2}kT + x\chi\right) \tag{38}$$

$x \approx 0.1$ (수소 이온화 비율), $\chi$ (수소 이온화 에너지)로 $V_z \sim 2$ km/s가 필요합니다. 수평 속도는 음속(~7 km/s)을 넘을 수 없으므로, 과립의 상한 크기는 $2r \sim 4$ Mm입니다 (관측과 일치). 깊이가 증가하면 눈금높이도 증가하여 더 큰 대류 셀이 나타납니다 — 20–30 Mm 깊이에서 convective cellular structures가 나타나며, mesogranular (4 Mm) 및 supergranular (8 Mm) 스케일도 관측됩니다.

The horizontal scale of granules is set by mass conservation: $r \approx 2H(v_H/v_z)$. With minimum vertical velocity ~2 km/s (needed to carry the solar flux) and maximum horizontal velocity ~7 km/s (sound speed), the upper size limit is $2r \sim 4$ Mm. At greater depths, scale heights increase, producing larger convective cells.

표면 100 km 이내에서 상승류가 면적의 약 2/3, 하강류가 1/3을 차지합니다.

#### §3.4 Horizontal Patterns and Evolution / 수평 패턴과 진화

과립의 온도와 속도 패턴은 강한 밀도 층화를 반영합니다. 표면에 도달한 플라즈마는 밀도가 급격히 감소하여 바깥쪽으로 퍼집니다 (Figure 8). 이 과정에서:

- 더 깊은 층의 작은 셀들의 하강류가 합쳐져 더 큰 셀의 filamentary downflows를 형성 → 계층적 구조 (Figure 7)
- 과립 표면의 대류는 대략적으로 superadiabatic 층화를 유지
- Schwarzschild 판정 기준에 의해 더 큰 스케일은 약간의 superadiabatic 성층에 의한 대류 불안정으로 구동

#### §3.5 Exploding Granules / 폭발하는 과립

과립이 확장하면서 상승류 속도가 감소하고, 중심부에서 복사 냉각이 충분히 강해져 어두운 중심이 형성됩니다. 이것이 "폭발하는 과립"입니다 — 중심에서 하강류가 시작되어 과립이 분열합니다. 이 현상은 시뮬레이션에서 자연스럽게 재현됩니다.

#### §3.6 Surface Entropy Jump / 표면 엔트로피 점프

표면에서의 엔트로피 점프는 태양 대류의 가장 특징적인 현상 중 하나입니다. 유체가 광학적 깊이 $\tau \sim 100$을 통과할 때 급격히 냉각되기 시작하며, $\tau = 1$을 통과하면서 엔트로피와 에너지가 급격히 감소합니다 (Figure 3):

- 광학적 깊이 $\tau \sim 100$에서 냉각 시작
- 가스가 재결합하면서 엔트로피 급감
- 밀도 증가로 표면 위를 지나면서 약간의 복사 재가열
- 다시 $\tau = 1$ 아래로 내려가면서 추가 냉각
- 내부로 향할수록 단열 압축과 확산에 의한 에너지 교환으로 가열

#### §3.7–3.8 Temperature Fluctuations & Average Structure / 온도 요동과 평균 구조

수평 평균 구조에서:
- 표면 근처 수평 평균 온도 기울기는 1D 모델보다 더 얕음 (corrugated $\tau = 1$ surface 효과)
- 표면 위에서는 복사 가열에 의해 1D 모델보다 더 따뜻함
- 평균 기울기 차이가 p-mode 주파수에 영향

#### §3.9 Vorticity / 와도

주로 baroclinic 생성 ($\nabla \rho \times \nabla P$ 항)에 의해 생성. 하강류에서 와도가 크며, turbulent downflows에서의 vortex stretching에 의해 증폭됩니다. 상승류에서는 상대적으로 층류적(laminar)입니다.

#### §3.10 Shocks / 충격파

수렴하는 intergranular 하강류의 수평 속도가 때때로 음속에 도달하여 약한 충격파가 형성됩니다. 이 충격파는 vorticity를 생성합니다.

#### §3.11 Energy Fluxes / 에너지 플럭스

에너지 플럭스의 수직 분포를 분석하면:
- 대류 엔탈피 플럭스와 운동 에너지 플럭스의 합이 표면 아래에서 지배적
- 표면을 지나면서 복사 플럭스로 급격히 전환
- 운동 에너지 플럭스는 하방을 향함 (무거운 하강류가 지배적이므로)
- Acoustic 플럭스는 매우 작음 (~전체의 0.01%)

#### §3.12 Connections with Mixing Length Recipes / 혼합 길이 이론과의 관계

혼합 길이 이론(MLT)은 1D 모델에서 대류를 기술하는 전통적 방법이지만, 3D 시뮬레이션과의 비교를 통해 그 한계가 명확해졌습니다:

- MLT는 국소적, 시간 비의존적, 1D 모델
- 대류 에너지 수송의 비대칭성 (넓고 느린 상승류 vs. 좁고 빠른 하강류)을 포착하지 못함
- 복사장과 가스의 3D 에너지 교환 무시
- 그러나 적절한 $\alpha$ 매개변수 선택으로 평균 층화를 합리적으로 근사 가능

MLT captures the average stratification reasonably with an appropriate $\alpha$ parameter, but fundamentally cannot represent the asymmetry between broad, gentle upflows and narrow, fast downflows, nor the 3D energy exchange between the radiation field and gas.

### §4 Larger Scale Flows and Multi-Scale Convection / 대규모 흐름과 다중 스케일 대류 (pp. 35–38)

전통적으로 태양 표면 흐름은 세 스케일로 분류됩니다:
- **Mesogranulation**: 5–10 Mm (November 1980에서 최초 감지)
- **Supergranulation**: 20–50 Mm (Hart 1956, Leighton et al. 1962)
- **Giant cells**: >100 Mm

그러나 현재 증거는 이들이 **별개의 대류 스케일이 아닌 연속적 속도 스펙트럼**의 일부임을 보여줍니다. SOHO/MDI 관측과 시뮬레이션 모두에서 속도 진폭은 파수에 대해 거의 선형적으로 감소하며 (velocity spectrum $V(k) = \sqrt{kP(k)}$, Eq. 42), 과립 이상의 스케일에서 뚜렷한 특징이 없습니다 (Figure 22).

Current evidence strongly suggests a continuous velocity spectrum rather than distinct scales. Both SOHO/MDI observations and simulations show velocity amplitudes decreasing approximately linearly with horizontal wavenumber on scales larger than granulation, with no distinct features (Figure 22). The velocity spectrum is approximately linear in $k$: a few km/s on granular scales, several hundred m/s on mesogranular scales, and 100–200 m/s on supergranular scales.

**자기 관측 스케일의 기원**: "meso-" 와 "supergranulation"의 구분은 관측 기법의 유효 "필터"에 의한 folding 효과일 가능성이 높습니다. 작은 스케일의 대류 운동이 더 큰 스케일을 advect하여 패턴을 생성합니다.

**Multi-scale self-similarity**: Gaussian 필터로 400, 200, 100, 50 Mm 크기에서 같은 수의 해상도 요소를 남기면, 각 스케일의 패턴이 매우 유사합니다 (Figures 24–25). 이는 scale-free 행동을 보여줍니다.

### §5 Spectral Line Formation / 스펙트럼 선 형성 (pp. 39–59)

이 섹션은 논문의 **가장 영향력 있는 결과**를 담고 있습니다.

#### §5.1 Spatially Resolved Lines / 공간 분해 선

태양 표면의 공간 분해 스펙트럼 선은 놀라운 다양성을 보입니다 (Figure 26):
- 상승류에서: 연속광 강도 높음, 청색 편이, 강한 선
- 하강류에서: 연속광 강도 낮음, 적색 편이, 약한 선
- 개별 line bisector는 공간 평균 프로파일의 비대칭성과 전혀 다름

3D 시뮬레이션은 각 선의 고유한 "fingerprint"를 재현합니다: 등가 폭, 선 세기, 깊이, 편이, 비대칭성이 과립 패턴에 따른 연속광 강도와의 상관 관계를 매우 잘 맞춥니다 (Figure 27). 대부분의 선에서 LTE 가정으로도 훌륭한 일치를 보입니다 (Li I 670.8 nm선은 non-LTE 효과의 예외).

#### §5.2 Spatially Averaged Lines / 공간 평균 선

1D 모델의 근본적 한계:
- 광구 속도장을 예측하지 않음
- 스펙트럼 선이 관측보다 좁음 (Figure 28)
- 이를 보정하기 위해 **micro/macroturbulence**라는 fudge parameter 도입 필요

3D 모델의 혁명:
- 자기 일관적(self-consistent) 속도장으로부터 선 폭 자연스럽게 재현
- **fudge parameter 불필요** (Figure 29)
- 관측된 선 프로파일과의 잔차가 1–2% 수준

The width of spatially averaged photospheric lines is much wider than predicted from natural and thermal broadening alone. 1D models require micro/macroturbulence fudge parameters to compensate; 3D models reproduce the widths naturally through self-consistent velocity fields, with residuals at the 1–2% level (Figure 29).

태양의 광구 스펙트럼 선은 특징적인 **C-자 형태의 bisector**를 가집니다 — 상승류(청색 편이)가 하강류보다 더 넓은 면적을 차지하기 때문입니다. 3D 시뮬레이션은 Fe I과 Fe II 선의 비대칭성을 잘 재현합니다 (Figure 30). 약한 Fe I 선은 $\approx -500$ m/s의 선 편이를 보입니다.

#### §5.3 Solar Abundance Analysis / 태양 풍부도 분석

3D 모델의 성공은 태양 화학 조성 결정의 혁명을 가져왔습니다. 1D 모델과 비교한 3D 모델의 장점:
- 혼합 길이 매개변수 불필요
- Micro/macroturbulence 불필요
- 대기 비균질성과 대류 속도의 자기 일관적 처리

**탄소 (C)**: $\log \epsilon_C = 8.39 \pm 0.05$ (3D) — 이전 값보다 ~0.17 dex 낮음. 다양한 진단 지표([C I], C I, CH, C₂ Swan, CO)가 3D에서만 일관된 결과를 줌 (Table 1). 탄소 동위비 $^{12}$C/$^{13}$C = 87 ± 4 (CO 선에서)는 지구값과 일치.

**질소 (N)**: $\log \epsilon_N = 7.80 \pm 0.05$ (3D) — 이전보다 ~0.25 dex 낮음. N I, NH, CN 지표 모두 일관.

**산소 (O)**: $\log \epsilon_O = 8.66 \pm 0.05$ (3D, Asplund et al. 2004) — 이전 값(8.93, Anders & Grevesse 1989)보다 ~0.27 dex 낮음. 이 원소가 가장 논란이 됨:
- [O I] 630 nm 선이 Ni I 블렌드의 영향을 받음 (Figure 31)
- O I 777 nm 삼중선은 non-LTE 효과에 민감 (Figure 32)
- OH 선은 온도 구조에 매우 민감
- 산소 동위비 $^{16}$O/$^{18}$O = 479 ± 29 (3D, CO 선)는 지구값(499 ± 1)과 일치 (1D 모델은 불일치)

**철 (Fe)**: $\log \epsilon_{\text{Fe I}} = 7.44$, $\log \epsilon_{\text{Fe II}} = 7.45$ (3D) — 1D 모델의 microturbulence 의존성 제거. H 충돌에 의한 non-LTE 효과는 아직 불확실.

**기타 원소**: Si, Ni, Zr, Eu, Hf, Th 등의 3D 기반 풍부도도 결정됨. 대부분 Holweger–Müller 1D 모델보다 0.02–0.1 dex 낮음.

#### §5.3.6 Implications and Reliability / 함의와 신뢰성

**좋은 소식**: 새로운 낮은 풍부도는 전체 광구 금속량을 $Z = 0.0122$로 감소시켜, 인근 OB형 별과 성간 매질의 관측과 일치시킵니다. 또한 은하 화학 진화 모델과 일관됩니다.

**나쁜 소식**: C, O, Ne가 태양 내부의 주요 불투명도 원천이므로, 감소된 풍부도는 태양 내부 모델의 온도/밀도 구조를 변화시킵니다. 이로 인해 **helioseismology로 추론된 대류층 깊이와 불일치**가 발생합니다 — 이것이 **"태양 풍부도 위기(solar abundance crisis)"**입니다.

제안된 해결책들: 빠진 불투명도, 저금속성 가스의 부착, 내부 중력파, Ne 풍부도 재평가 등 — 아직 만족스러운 해결이 없습니다.

The "solar abundance crisis": the reduced C, N, O abundances change the predicted temperature and density structure of the solar interior, making the predicted depth of the convection zone inconsistent with helioseismology. Proposed solutions include missing opacity, accretion of low-metallicity gas, internal gravity waves, and revised Ne abundances — none fully satisfactory.

### §6 Applications to Helioseismology / Helioseismology 응용 (pp. 60–70)

#### §6.1 Wave Propagation in the Convection Zone / 대류층에서의 파동 전파

태양 내부에서 음파와 표면 중력파가 전파합니다. Green 함수:

$$\left[\frac{\partial^2}{\partial t^2} - c^2 \nabla^2\right] G(\mathbf{r}, t) = S(\mathbf{r}, t)\delta(\mathbf{r} - \mathbf{r}_0)\delta(t - t_0) \tag{43}$$

대류 시뮬레이션을 사용한 파동 전파 연구는 time-distance helioseismology의 정확도를 검증하는 데 사용됩니다. Ray-tracing 역산으로 추론한 수평 속도는 시뮬레이션의 실제 흐름과 4 Mm 깊이까지 잘 일치하나, 수직 속도 역산은 부정확합니다 (Figure 34).

#### §6.2 Excitation of p-modes / p-mode 여기

p-mode는 대류층에서의 엔트로피 요동과 Reynolds 응력에 의해 여기됩니다. 여기율 (PdV 일 적분):

$$\frac{\Delta \langle E_\omega \rangle}{\Delta t} = \frac{\omega^2 \left|\int_z dz \delta P_\omega^* \frac{\partial \xi_\omega}{\partial z}\right|^2}{8\Delta\nu E_\omega} \tag{50}$$

여기서 $\delta P_\omega^* = \delta P_t + \delta P_g^{\text{non-ad}}$ (turbulent pressure + non-adiabatic gas pressure). 시뮬레이션에서 계산한 여기율은 SOHO GOLF 관측과 잘 일치합니다 (Figure 36).

핵심 물리:
- 저주파에서 여기 감소: mode mass 증가 + mode compression 감소
- 고주파에서 여기 감소: 대류가 저주파 과정이므로 고주파 압력 요동이 약함
- 여기는 표면 근처(표면 아래 수 100 km 이내)에서 집중됨 (Figure 41)
- 자기장이 있는 활동 영역 주변에서 고주파 p-mode 여기 증가 → "acoustic halo" 현상

P-mode excitation occurs close to the solar surface where turbulent velocities and entropy fluctuations are largest. The excitation rate from simulations agrees well with SOHO GOLF observations (Figure 36).

#### §6.3 p-mode Frequencies / p-mode 주파수

1D 모델의 p-mode 고유 주파수와 관측 주파수 사이에 불일치가 있습니다. 대류가 두 가지 효과로 이 불일치를 줄입니다:

1. **Turbulent pressure**: 대류층 상부 가스 압력의 ~15%를 차지하여 대기를 약 반 눈금높이 상승시킴
2. **뜨거운 가스의 은닉**: H⁻ 불투명도의 온도 의존성으로 인해 뜨거운 가스가 보이지 않음 → 수평 평균 온도가 1D effective temperature보다 높음 → 대기가 또 약 반 눈금높이 상승

총 효과: 대기가 약 1 scale height 확장 → 공명 공동(resonant cavity) 증가 → 주파수 감소. 이것이 1D 모델과 관측 사이의 불일치를 줄입니다 (Figure 43). f-mode 잔차는 변하지 않지만 (정역학 구조에 거의 무관), p-mode 잔차는 크게 감소합니다.

Convection extends the atmosphere by about one scale height (turbulent pressure + hiding of hot gas), enlarging the resonant cavity and reducing p-mode frequencies — thereby reducing the discrepancy with observations (Figure 43).

#### §6.4 p-mode Line Profiles / p-mode 선 프로파일

진동과 대류/복사의 상호작용은 mode spectra의 비대칭성과 속도/강도 사이의 비대칭 반전을 야기합니다. 복사 전달 효과가 이 위상 관계를 지배합니다.

### §7 Interaction with Magnetic Fields / 자기장과의 상호작용 (pp. 71–88)

#### §7.1 Effects of Magnetic Fields on Convection / 자기장이 대류에 미치는 효과

표면 근처 자기장은 대략적 압력 평형 상태입니다 — 자기 집중 내부의 자기+가스 압력이 주변 가스 압력과 같습니다. 대류 흐름이 자기 플럭스를 intergranular lane으로 쓸어모읍니다.

자기장 세기에 따른 과립 변화 (Figure 45, Vögler 2005):
- **0 G**: 정상 과립 패턴
- **200 G**: intergranular lane이 자기장으로 채워지고 넓어짐, micropore 형성
- **800 G**: 소수의 무자기장 과립 섬이 넓은 자기장 레인에 둘러싸임

자기 플럭스 분포: 약한 자기장(<500 G)은 log-normal 분포, 강한 자기장(1–10%)만이 Zeeman splitting으로 관측 가능하지만 자기 에너지의 절반 이상을 차지합니다. 가장 가능성 높은 자기장 값은 ~10 G입니다 (Figure 48).

**Flux expulsion vs. advective concentration**: 대류 운동은 자기장을 대류 셀 내부에서 밀어내어(expulsion) 경계에 집중시킵니다(advective transport). 자기장선은 자유 표면을 관통하여 효율적으로 집중되고, 강한 자기장에 의한 대류 억제로 더욱 강화됩니다.

#### §7.2 Center-to-Limb Behavior / Center-to-Limb 행동

디스크 중심에서 림(limb)으로 관측하면 여러 변화가 있습니다:
- 과립이 3D "베개(pillow)" 형태로 보임
- 디스크 중심: 작은 자기 집중은 밝은 점(bright points), 큰 것은 어두움
- 림 근처: 자기 집중의 낮은 밀도/불투명도를 통해 뒤쪽 뜨거운 과립 벽이 보임 → **faculae** (Figure 54)

**"Hot wall" 효과** (Spruit 1976, 1977): 강한 자기장 영역에서 intergranular lane이 최대 350 km 아래로 함몰됩니다. 림 방향에서 보면, 이 빈 공간을 통해 뜨거운 과립 벽이 보여 밝은 faculae로 관측됩니다. G-band에서 밝은 점은 강한 자기장의 좋은 proxy이지만, 많은 강한 자기장 영역은 G-band에서 어둡게 나타납니다 (더 넓은 면적, Figure 52).

#### §7.3 Magnetic Flux Emergence / 자기 플럭스 출현

자기 플럭스 출현의 두 메커니즘:
1. **부력(buoyancy)**: 강한 자기장을 가진 집중이 압력 평형을 위해 밀도가 낮아 부력을 받아 상승
2. **이류(advection)**: 대류 상승류에 의해 운반

**"Magnetic flux pumping"**: 상승류/하강류의 비대칭성 (진폭과 위상)이 자기 플럭스의 하방 수송 경향을 만듭니다 — 대류층 깊은 곳에 상당한 자기 플럭스를 저장할 수 있습니다.

시뮬레이션에서 twisted flux tube의 상승이 재현됩니다 (Figure 55): 표면에 도달하면 과립 흐름이 자기장을 intergranular lane으로 분산시키고 패턴이 정상화됩니다. 또한 bipole 출현과 침강이 모두 관측됩니다 (Figure 56).

**Local dynamo**: Emonet & Cattaneo (2001)는 회전 없이도 난류 매질에서 다이나모 작용이 일어남을 보였습니다. 태양 표면에서도 전지구적 다이나모와 함께 국소 다이나모가 작용할 수 있습니다.

#### §7.4 Convection as a Driver of Chromospheric and Coronal Heating / 채층 및 코로나 가열의 구동원

태양 대류층은 채층과 코로나 활동의 궁극적 에너지원입니다. 에너지 전달은 주로 **Poynting flux** (전자기 에너지 플럭스)를 통해 이루어집니다.

핵심 물리: Poynting flux의 크기를 결정하는 핵심 요인은 자기장선과 표면의 각도입니다:
- 비틀림에 대한 저항이 강하면 → 역방향 힘 발생 → 큰 Poynting flux
- 저항이 없으면 → 자기장선이 직선 유지 → Poynting flux 거의 없음

**중요한 결과**: 저항률이 증가하면 에너지 소산이 *감소*합니다 — 더 큰 저항률은 자기장선이 더 빨리 확산하여 경사가 줄어들기 때문입니다. 따라서 주어진 저항률에서의 소산은 매우 낮은 저항률에서의 소산의 **하한**입니다 — 수치 실험이 정확한 예측을 할 수 있다는 희망을 줍니다.

Gudiksen & Nordlund (2005)는 이 메커니즘만으로 3D 코로나 모델을 구축하여 합성 진단이 관측과 잘 일치함을 보였습니다.

### §8 Current Status and Future Directions / 현황과 미래 방향 (p. 89)

- 표면 근처 유체역학은 정성적, 정량적으로 잘 이해됨
- 자기-유체역학(MHD)은 아직 불완전 — 흑점과 태양 활동 주기 포함
- 채층/코로나의 MHD 연구가 현재 주요 연구 방향
- 미래 과제: 복사, 해상도, 확산, non-LTE, 이온화 등의 개선 필요
- Particle-in-cell 코드, 하이브리드 유체/입자 코드 등 새로운 도구 개발 필요

### §9 Summary / 요약 (p. 90)

태양 대류는 태양의 자기장, 폭발 현상, 행성간 매질, 지구 날씨와 우주 날씨를 궁극적으로 제어하는 구동원입니다. 관측과 수치 시뮬레이션의 결합이 이 물리 과정에 대한 상세한 통찰을 제공합니다.

---

## 3. Key Takeaways / 핵심 시사점

1. **대류는 표면 냉각에 의해 구동된다** — 내부의 가열이 아닌, 표면에서의 복사 냉각이 엔트로피를 급격히 감소시켜 과밀 하강류를 생성하는 것이 대류의 주된 구동 메커니즘입니다. 부력 일의 대부분은 차가운 하강류에서 발생합니다.
Convection is driven by radiative cooling at the surface, not by heating from below. The sharp entropy drop creates overdense downflows — most buoyancy work occurs in these cool downflows, not warm upflows.

2. **3D 모델은 fudge parameter 없이 관측을 재현한다** — micro/macroturbulence라는 인위적 매개변수 없이도, 3D 복사-유체역학 시뮬레이션이 관측된 스펙트럼 선의 폭, 형태, 편이, 비대칭성을 1–2% 정확도로 재현합니다. 이는 1D 모델에 대한 패러다임 전환입니다.
3D models reproduce observed spectral line widths, shapes, shifts, and asymmetries at the 1–2% level without any fudge parameters — a paradigm shift from 1D models requiring micro/macroturbulence.

3. **태양 C, N, O 풍부도가 대폭 하향 수정되었다** — 3D 모델 기반 분석에서 $\log \epsilon_C = 8.39$, $\log \epsilon_N = 7.80$, $\log \epsilon_O = 8.66$으로, 이전 표준값보다 0.15–0.27 dex 낮습니다. 전체 금속량은 $Z = 0.0194$에서 $Z = 0.0122$로 감소합니다.
3D-based analysis yields $\log \epsilon_C = 8.39$, $\log \epsilon_N = 7.80$, $\log \epsilon_O = 8.66$ — 0.15–0.27 dex lower than previous standard values. The overall metallicity drops from $Z = 0.0194$ to $Z = 0.0122$.

4. **태양 풍부도 위기가 발생했다** — 감소된 C, N, O 풍부도는 태양 내부 모델의 불투명도를 변화시켜, helioseismology로 측정된 대류층 깊이와 불일치를 야기합니다. 이 문제는 2009년 시점에서 미해결이며, 현재까지도 완전히 해결되지 않았습니다.
The reduced abundances alter solar interior opacity, creating a discrepancy with the convection zone depth measured by helioseismology — the "solar abundance crisis" that remains unresolved.

5. **대류 스케일은 연속 스펙트럼을 형성한다** — Mesogranulation과 supergranulation은 별개의 대류 모드가 아니라, 과립에서 전지구적 스케일까지 매끄럽게 연결되는 속도 스펙트럼의 일부입니다. 속도 진폭은 파수에 반비례하여 감소합니다.
Meso- and supergranulation are not distinct convective modes but part of a smooth velocity spectrum from granular to global scales, with amplitudes decreasing roughly inversely with wavenumber.

6. **대류가 p-mode를 여기하고 주파수를 수정한다** — 대류에 의한 엔트로피 요동과 Reynolds 응력이 p-mode를 여기하며, 시뮬레이션 여기율은 SOHO GOLF 관측과 일치합니다. 또한 turbulent pressure와 hot gas hiding이 대기를 ~1 scale height 확장하여 p-mode 주파수 불일치를 줄입니다.
Convection excites p-modes via entropy fluctuations and Reynolds stresses (matching SOHO GOLF observations), and extends the atmosphere by ~1 scale height, reducing p-mode frequency discrepancies.

7. **자기장은 대류에 의해 intergranular lane에 집중된다** — Flux expulsion과 advective transport에 의해 자기 플럭스가 대류 셀 경계에 집중됩니다. 약한 자기장이 표면의 대부분을 차지하지만, 강한 자기장(1–10% 면적)이 자기 에너지의 절반 이상을 차지합니다.
Magnetic flux is concentrated in intergranular lanes by flux expulsion and advective transport. Weak fields cover most of the surface, but strong fields (1–10% of area) contain more than half the magnetic energy.

8. **대류가 코로나 가열을 구동한다** — Poynting flux가 대류에서 코로나로의 주된 에너지 전달 메커니즘이며, 저항률 증가가 오히려 소산을 감소시킨다는 반직관적 결과가 수치 실험의 예측력에 희망을 줍니다.
Poynting flux is the main energy transport mechanism from convection to corona. Counter-intuitively, increased resistivity *decreases* energy dissipation — giving hope that numerical experiments can provide accurate predictions.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 기본 유체 방정식 / Governing Fluid Equations

| 방정식 / Equation | 수식 / Formula | 의미 / Meaning |
|---|---|---|
| 질량 보존 (Eq. 1) | $\frac{\partial \rho}{\partial t} = -\nabla \cdot (\rho \mathbf{u})$ | 밀도 변화 = 질량 플럭스 발산의 음수 / Density change = negative divergence of mass flux |
| 운동 방정식 (Eq. 6) | $\frac{D\mathbf{u}}{Dt} = -\frac{P}{\rho}\nabla \ln P - \nabla \Phi - \frac{1}{\rho}\nabla \cdot \tau_{\text{visc}}$ | 유체 가속 = 압력 기울기 + 중력 + 점성 / Fluid acceleration = pressure gradient + gravity + viscous |
| 정역학 평형 (Eq. 7) | $-\frac{P}{\rho}\frac{\partial \ln P}{\partial z} = g_z$ | 수직 압력 기울기-중력 균형 / Vertical pressure-gravity balance |
| 압력 눈금높이 (Eq. 9) | $H_P = P / (\rho g_z)$ | 압력이 $e$배 감소하는 높이 / Height for $e$-fold pressure decrease |

### 4.2 에너지 관련 방정식 / Energy Equations

| 방정식 / Equation | 수식 / Formula | 의미 / Meaning |
|---|---|---|
| 내부 에너지 (Eq. 18) | $\frac{\partial E}{\partial t} = -\nabla \cdot (E\mathbf{u}\mathbf{u}) - P(\nabla \cdot \mathbf{u}) + Q_{\text{rad}} + Q_{\text{visc}}$ | 에너지 진화: 이류 + PdV + 복사 + 점성 / Energy evolution |
| 복사 전달 (Eq. 27) | $\frac{\partial I_\nu}{\partial \tau_\nu} = S_\nu - I_\nu$ | 광선을 따른 강도 변화 / Intensity change along ray |
| 복사 가열 (Eq. 29) | $Q_{\text{rad}} = \int_\nu \int_\Omega \rho \kappa_\nu (I_\nu - S_\nu) \, d\Omega \, d\nu$ | $I_\nu < S_\nu$이면 냉각 / Cooling when $I_\nu < S_\nu$ |
| 총 에너지 플럭스 (Eq. 32) | $\frac{\partial(E+E_{\text{kin}})}{\partial t} = -\nabla \cdot (\mathbf{F}_{\text{conv}} + \mathbf{F}_{\text{kin}} + \mathbf{F}_{\text{rad}} + \mathbf{F}_{\text{visc}})$ | 4가지 에너지 플럭스의 보존 / Conservation of 4 energy fluxes |

### 4.3 스케일 선택과 대류 구동 / Scale Selection & Convective Driving

| 방정식 / Equation | 수식 / Formula | 의미 / Meaning |
|---|---|---|
| 과립 크기 (Eq. 37) | $r = 2H(v_H / v_z)$ | 과립 반경 ∝ 눈금높이 × 속도비 / Granule radius |
| 표면 플럭스 균형 (Eq. 38) | $\sigma T_{\text{eff}}^4 \approx \rho V_z (\frac{5}{2}kT + x\chi)$ | 복사 손실 = 대류 엔탈피 플럭스 / Radiative loss = convective enthalpy flux |
| 속도 스펙트럼 (Eq. 42) | $V(k) = \sqrt{kP(k)}$ | 단위 $\ln k$ 당 속도 진폭 / Velocity amplitude per unit $\ln k$ |

### 4.4 Helioseismology / Helioseismology

| 방정식 / Equation | 수식 / Formula | 의미 / Meaning |
|---|---|---|
| Green 함수 (Eq. 43) | $\left[\frac{\partial^2}{\partial t^2} - c^2 \nabla^2\right] G = S\delta(\mathbf{r}-\mathbf{r}_0)\delta(t-t_0)$ | 음파 전파의 Green 함수 / Acoustic Green's function |
| p-mode 여기율 (Eq. 50) | $\frac{\Delta \langle E_\omega \rangle}{\Delta t} = \frac{\omega^2 |\int dz \delta P_\omega^* \frac{\partial \xi_\omega}{\partial z}|^2}{8\Delta\nu E_\omega}$ | PdV 일에 의한 mode 에너지 증가율 / Mode energy increase via PdV work |
| 대류 속도 스펙트럼 (Eq. 52) | $P_{\text{vel}}(\nu) = A(\nu^2 + w(k)^2)^{-n(k)}$ | 대류의 시간 주파수 스펙트럼 / Convective temporal frequency spectrum |

### 4.5 수치적 예시 / Numerical Example

**과립 크기 추정**: 표면에서 $H_P \approx 150$ km, $v_z \sim 2$ km/s, $v_H \sim 7$ km/s (음속)이면:
$$r = 2 \times 150 \times (7/2) = 1050 \text{ km} \approx 1 \text{ Mm}$$

이것이 관측된 과립 크기 ~1 Mm과 일치합니다. 상한(음속 제한)에서 $2r \sim 4$ Mm으로, 관측된 최대 과립 크기와 일치합니다.

**태양 풍부도 비교**:

| 원소 | 3D 모델 (이 논문) | Anders & Grevesse 1989 | 차이 |
|---|---|---|---|
| C | $\log \epsilon = 8.39$ | $\log \epsilon = 8.56$ | −0.17 dex |
| N | $\log \epsilon = 7.80$ | $\log \epsilon = 8.05$ | −0.25 dex |
| O | $\log \epsilon = 8.66$ | $\log \epsilon = 8.93$ | −0.27 dex |
| Fe | $\log \epsilon = 7.45$ | $\log \epsilon = 7.67$ | −0.22 dex |

전체 금속량: $Z = 0.0122$ (3D) vs. $Z = 0.0194$ (이전) — 약 37% 감소.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1801 ─── Herschel: 과립 최초 관측 / First granulation observation
  │
1864 ─── Dawes: "granule" 용어 도입 / Coined "granule"
  │
1896 ─── Janssen: 최초 양질 과립 사진 / First quality photographs
  │
1956 ─── Hart: Supergranulation 최초 관측 / First supergranulation detection
  │
1958 ─── Böhm-Vitense: MLT 정립 / Mixing length theory formalized
  │
1962 ─── Leighton et al.: Supergranulation 상세 연구 / Detailed supergranulation study
  │
1976 ─── Vernazza et al.: VAL3C 반경험적 모델 대기 / Semi-empirical model atmosphere
  │
1982 ─── Nordlund: Opacity binning method / 불투명도 비닝법 도입
  │
1984 ─── Nordlund: 최초 현실적 3D 대류 시뮬레이션 / First realistic 3D simulations
  │
1989 ─── Anders & Grevesse: 표준 태양 풍부도 (높은 값) / Standard solar abundances
  │      Stein & Nordlund: 대류 구동 메커니즘 규명 / Convective driving mechanism
  │
1998 ─── Stein & Nordlund: 표면 냉각 구동 입증, 과립 물리 / Surface cooling driving
  │
2000 ─── Asplund et al.: 3D 스펙트럼 선 재현 성공 / 3D line profile matching
  │
2004 ─── Asplund et al.: 태양 C, N, O 풍부도 하향 수정 / Abundance revision
  │      ★★★ "태양 풍부도 위기" 시작 / Solar abundance crisis begins ★★★
  │
2005 ─── Vögler: 3D MHD 과립 시뮬레이션 / 3D MHD granulation
  │      Gudiksen & Nordlund: 3D 코로나 모델 / 3D corona model
  │
2006 ─── Stein et al.: Supergranulation 규모 시뮬레이션 / Supergranulation-scale sims
  │
2009 ─── ★ Nordlund, Stein & Asplund: 이 리뷰 논문 ★ / This review
  │
2021 ─── Asplund et al.: 풍부도 재분석 (소폭 상향) / Abundance re-analysis
  │
2022 ─── Magg et al.: 대안적 풍부도 분석 (논쟁 지속) / Alternative analysis (debate continues)
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Böhm-Vitense (1958), "Mixing Length Theory" | 3D 시뮬레이션이 대체하는 전통적 1D 대류 모델 / Traditional 1D convection model superseded by 3D simulations | MLT의 한계와 3D 모델의 우월성을 이해하는 비교 기준 / Baseline for understanding MLT limitations |
| Anders & Grevesse (1989), "Solar Abundances" | 3D 분석으로 하향 수정된 기존 표준 태양 풍부도 / Previous standard abundances revised downward by 3D analysis | 풍부도 위기의 출발점 / Starting point of the abundance crisis |
| Stein & Nordlund (1998), "Simulations of Solar Granulation" | 이 리뷰의 핵심 시뮬레이션 — 대류 구동, 과립 물리 / Core simulations underpinning this review | 표면 냉각 구동, 에너지 플럭스, 부력 일 분석의 원천 / Source of driving mechanism, energy flux, buoyancy work analysis |
| Asplund et al. (2004, 2005), "Solar C/N/O Abundances" | 이 리뷰에서 종합된 3D 풍부도 결정의 원논문 / Original papers on 3D abundance determinations summarized here | 태양 풍부도 위기를 촉발한 핵심 결과 / Key results triggering the abundance crisis |
| Vögler (2005), "3D MHD Simulations" | 자기장-대류 상호작용의 핵심 시뮬레이션 / Key simulations of magnetic field-convection interaction | §7의 자기장 효과 논의의 주요 출처 / Main source for §7 magnetic field discussion |
| Gudiksen & Nordlund (2005), "3D Corona Model" | Poynting flux에 의한 코로나 가열 모델 / Coronal heating model via Poynting flux | 대류→코로나 에너지 전달의 정량적 증거 / Quantitative evidence for convection→corona energy transfer |
| Christensen-Dalsgaard et al. (1996), "Solar Model S" | p-mode 주파수 비교의 표준 1D 태양 모델 / Standard 1D solar model for p-mode frequency comparison | §6.3의 주파수 잔차 분석의 기준 / Reference for frequency residual analysis in §6.3 |
| Goldreich et al. (1994), "p-mode Excitation" | p-mode 여기의 이론적 기초 / Theoretical foundation for p-mode excitation | §6.2의 여기 메커니즘 비교 대상 / Comparison for excitation mechanism in §6.2 |

---

## 7. References / 참고문헌

- Nordlund, Å., Stein, R. F., & Asplund, M., "Solar Surface Convection", *Living Rev. Solar Phys.*, 6, 2, 2009. [DOI: 10.12942/lrsp-2009-2]
- Stein, R. F. & Nordlund, Å., "Simulations of Solar Granulation. I. General Properties", *ApJ*, 499, 914, 1998.
- Asplund, M., Grevesse, N., Sauval, A. J., & Scott, P., "The Chemical Composition of the Sun", *ARA&A*, 47, 481, 2009.
- Asplund, M., Grevesse, N., & Sauval, A. J., "The Solar Chemical Composition", *ASP Conf. Ser.*, 336, 25, 2005.
- Asplund, M., et al., "Line formation in solar granulation. IV. [O I], O I and OH lines and the photospheric O abundance", *A&A*, 417, 751, 2004.
- Anders, E. & Grevesse, N., "Abundances of the elements: Meteoritic and solar", *Geochim. Cosmochim. Acta*, 53, 197, 1989.
- Böhm-Vitense, E., "Über die Wasserstoffkonvektionszone in Sternen verschiedener Effektivtemperaturen und Leuchtkräfte", *Z. Astrophys.*, 46, 108, 1958.
- Vögler, A., "On the effect of a small-scale dynamo on the structure of the solar photosphere", *Mem. S.A.It.*, 76, 842, 2005.
- Gudiksen, B. V. & Nordlund, Å., "An Ab Initio Approach to the Solar Coronal Heating Problem", *ApJ*, 618, 1020, 2005.
- Nordlund, Å., "Numerical simulations of the solar granulation. I. Basic equations and methods", *A&A*, 107, 1, 1982.
- Christensen-Dalsgaard, J., et al., "The Current State of Solar Modeling", *Science*, 272, 1286, 1996.
- Goldreich, P., Murray, N., & Kumar, P., "Excitation of solar p-modes", *ApJ*, 424, 466, 1994.
- Grevesse, N. & Sauval, A. J., "Standard Solar Composition", *Space Sci. Rev.*, 85, 161, 1998.
