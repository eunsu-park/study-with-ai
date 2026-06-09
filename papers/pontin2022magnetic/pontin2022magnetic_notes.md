---
title: "Magnetic Reconnection: MHD Theory and Modelling"
authors: [David I. Pontin, Eric R. Priest]
year: 2022
journal: "Living Reviews in Solar Physics"
doi: "10.1007/s41116-022-00032-9"
topic: Living_Reviews_in_Solar_Physics
tags: [magnetic_reconnection, MHD, solar_corona, plasmoid_instability, null_points, QSL, 3D_reconnection, Sweet_Parker, Petschek]
status: completed
date_started: 2026-04-23
date_completed: 2026-04-23
---

# 77. Magnetic Reconnection: MHD Theory and Modelling / 자기재결합: MHD 이론과 모델링

---

## 1. Core Contribution / 핵심 기여

Pontin과 Priest의 202쪽 *Living Reviews* 논문은 태양 코로나의 자기재결합에 대한 MHD 관점을 망라한다. 저자들은 먼저 자기 위상학(null points, separatrices, QSLs)과 자속/장선 보존 개념을 정립한 뒤, 2D 고전 모델(Sweet-Parker, Petschek, tearing mode)과 플라즈모이드 불안정성을 설명하고, 이어서 3D 재결합의 근본적으로 다른 성질—flux velocity 부재, $\int E_\parallel \, dl$로 정의되는 재결합률, spine-fan/torsional/separator/quasi-separator 등 다양한 모드—을 제시한다. 마지막으로 태양 플레어, 코로나 가열, 태양풍, 지구 자기권에 대한 응용을 짧게 다룬다. 핵심 메시지는 **3D 재결합은 2D의 단순 확장이 아니라 질적으로 다른 현상**이라는 점과, 현대 코로나 관측의 대부분은 단순한 2D X-point 그림이 아닌 QSL/HFT, braided field, null-point 구조에서의 3D 재결합으로 이해해야 한다는 점이다.

Pontin and Priest's 202-page *Living Reviews* article provides a comprehensive MHD perspective on magnetic reconnection in the solar corona. The authors first establish magnetic topology (null points, separatrices, QSLs) and flux/field-line conservation concepts, then explain classical 2D models (Sweet-Parker, Petschek, tearing mode) and the plasmoid instability, before presenting the fundamentally different nature of 3D reconnection—the non-existence of a flux velocity, the rate defined by $\int E_\parallel \, dl$, and the variety of modes (spine-fan, torsional spine/fan, separator, quasi-separator). Applications to solar flares, coronal heating, the solar wind, and the magnetosphere are briefly surveyed. The central message is that **3D reconnection is qualitatively different** from 2D, and most coronal observations must be understood through QSL/HFT, braided field, and null-point structures rather than simple 2D X-point cartoons.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction and Historical Overview (Sect. 1) / 서론과 역사적 개관

**Sect. 1.1 — Four phases of reconnection theory / 재결합 이론의 네 단계**

저자들은 재결합 이론의 발전을 네 단계로 요약한다 (p. 6):

The authors summarise the theoretical development in four phases (p. 6):

1. **Sweet-Parker (1958) + Petschek (1964)**: 최초의 steady 2D 모델. 전자는 느리고($M_A \sim S^{-1/2}$), 후자는 빠르다($M_A \sim \pi/(8\ln S)$). First steady 2D models; SP is slow while Petschek is fast.
2. **Biskamp (1986) + Priest & Forbes (1986)**: 수치실험이 Petschek에 의문을 제기했으나, "Almost-Uniform" family가 발견되어 Petschek와 Biskamp 해를 특수경우로 포함시킴. Numerical experiments questioned Petschek, but the "Almost-Uniform" family subsumed Petschek and Biskamp as special cases.
3. **Collisionless & impulsive bursty reconnection**: Hall 효과로 전자·이온 관성 길이에서의 diffusion region과, tearing으로 인한 bursty 재결합 발견. Hall-scale diffusion region and bursty tearing-driven reconnection discovered.
4. **3D reconnection (Priest et al. 2003 ~ 현재/present)**: Schindler et al. (1988)의 $\int E_\parallel\, ds \neq 0$ 조건이 핵심. Schindler et al. (1988)'s $\int E_\parallel \, ds \neq 0$ condition is the key criterion; reconnection can occur without null points.

**Four ways reconnection changes the system / 재결합이 시스템을 변화시키는 네 방식** (p. 5): (a) 강한 전기장·전류·충격파가 고에너지 입자 가속, (b) 전류의 ohmic 소산이 자기에너지를 열로 전환, (c) $\mathbf{j}\times\mathbf{B}$ 힘이 플라즈마를 가속, (d) 장선 연결성 변화가 입자·열 수송 경로를 바꿈. (a) generates strong $\mathbf{E}$, currents, shocks (accelerating particles); (b) ohmic dissipation converts magnetic energy to heat; (c) Lorentz force drives high-speed flows; (d) connectivity changes alter particle/heat transport.

**Five modes of 3D reconnection identified (p. 7) / 식별된 3D 재결합의 다섯 모드**:

- **Torsional spine or torsional fan reconnection** — rotational motions near a null / null 주변 회전 운동
- **Spine-fan reconnection** — shearing motions near a null / null 주변 shearing 운동
- **Separator reconnection** — at the intersection of two separatrix surfaces / 두 분리면의 교차선에서
- **Quasi-separator or HFT reconnection** — at the intersection of two QSLs / 두 QSL의 교차선에서
- **Braid reconnection** — in a braided flux tube (no nulls, no separatrices) / 매듭진 자속관 (null도 separatrix도 없음)

이러한 모드 분류는 2022년 리뷰의 가장 명확한 조직화 틀 중 하나이며, 독자들이 각 절을 찾아 읽을 수 있게 한다. This classification is one of the clearest organisational frames of the 2022 review, letting readers navigate between sections.

### Part II: Topology and Null Points (Sect. 2) / 위상학과 영점

**Sect. 2.1 — 2D null points / 2D 영점**

2D null 근처 선형 자기장은 $\mathbf{B} = (B_0/r_0)[y, \bar\alpha^2 x]$ (Eq. 3)로 표현되며, $\bar\alpha^2 < 0$이면 O-type (circular), $\bar\alpha^2 > 0$이면 X-type (hyperbolic). 전류 밀도는 $j_z = (B_0/\mu r_0)(\bar\alpha^2 - 1)$. X-type null은 외부 경계가 자유로울 때 **local collapse** 불안정성이 있어 Dungey(1953) 이후 수많은 선형 분석이 수행됨. 저항률이 $\eta$일 때 붕괴 과정에서의 재결합률은 의외로 $1/\ln\eta$로 스케일링(빠르다!).

Near a 2D null the linear field is $\mathbf{B} = (B_0/r_0)[y, \bar\alpha^2 x]$, O-type for $\bar\alpha^2 < 0$ and X-type for $\bar\alpha^2 > 0$. Current density $j_z = (B_0/\mu r_0)(\bar\alpha^2-1)$. X-type nulls are locally unstable to collapse when the boundary is free; linear analyses (Dungey 1953 onwards) show surprisingly the reconnection rate during collapse scales as $1/\ln\eta$.

**Sect. 2.2 — 3D null points / 3D 영점**

3D null의 가장 단순한 예: $(B_x, B_y, B_z) = (x, y, -2z)$ (Eq. 5). 장선은 $y = Cx$, $z = K/x^2$를 만족하여 하나의 **spine curve**(z축)와 하나의 **fan surface**(xy-평면)를 이룬다. 일반 linear null은 네 개의 독립 상수 $(a, b, j_\|, j_\perp)$로 매개되고 (Eq. 6), $j_\perp \neq 0$이면 **oblique null**, $j_\|$가 충분히 크면 eigenvalue가 복소수가 되어 **spiral null**이 된다 (Fig. 4). 태양 코로나에는 광구 source 10개당 약 1개의 coronal null이 존재(Schrijver & Title 2002)—통계적으로 활동 영역마다 다수의 null이 있음.

The simplest 3D null has $\mathbf{B} = (x, y, -2z)$ (Eq. 5), with field lines satisfying $y = Cx$, $z = K/x^2$, giving a spine curve (z-axis) and a fan surface (xy-plane). A general linear null is parametrised by four constants $(a, b, j_\|, j_\perp)$ (Eq. 6); $j_\perp \neq 0$ gives an oblique null, and sufficiently large $j_\|$ gives a spiral null. In the corona, roughly one coronal null exists per ten photospheric sources.

**Sect. 2.6 — QSLs and squashing factor / 준분리층과 찌그러짐 인자**

QSL 개념(Priest & Démoulin 1995): nulls 없이도 강한 전류가 쌓일 수 있는, 장선 대응사상이 연속이지만 기울기가 매우 큰 곳. Titov(2007)의 **squashing factor**:

$$Q_\pm = \frac{-N_\pm^2}{B_{z\pm}/B_{z\mp}}, \quad N_\pm^2 = \left(\frac{\partial X_\pm}{\partial x_\pm}\right)^2 + \left(\frac{\partial X_\pm}{\partial y_\pm}\right)^2 + \left(\frac{\partial Y_\pm}{\partial x_\pm}\right)^2 + \left(\frac{\partial Y_\pm}{\partial y_\pm}\right)^2 \qquad (\text{Eq. 7})$$

$Q$ 성질: (i) 방향 무관 $Q_+ = Q_-$, (ii) separatrix 위에서 $Q \to \infty$, (iii) $Q \gg 2$이면 QSL. 두 QSL의 교선을 **Hyperbolic Flux Tube (HFT)** 또는 quasi-separator라 한다 (Titov et al. 2002).

QSL concept (Priest & Démoulin 1995): locations where field-line mapping is continuous but has extremely steep gradients, allowing strong currents to accumulate even without nulls. Titov's squashing factor: $Q$ is direction-independent, $Q\to\infty$ on separatrices, and $Q\gg 2$ identifies QSLs. The intersection of two QSLs forms a Hyperbolic Flux Tube (HFT) or quasi-separator.

### Part III: Flux Conservation and the Nature of 3D Reconnection (Sects. 3-4) / 자속 보존과 3D 재결합의 본질

이상 MHD ($R_m \gg 1$)에서 $\mathbf{E} + \mathbf{v}\times\mathbf{B} = 0$이 성립하여 flux velocity, field-line velocity, plasma velocity의 수직 성분이 모두 $\mathbf{w}_\perp = \mathbf{v}_\perp = \mathbf{E}\times\mathbf{B}/B^2$로 같다. 자속, 장선, topology가 모두 보존된다.

In ideal MHD, $\mathbf{E}+\mathbf{v}\times\mathbf{B}=0$ and the perpendicular components of plasma, flux, and field-line velocities all equal $\mathbf{E}\times\mathbf{B}/B^2$. Flux, field lines, and topology are all conserved.

비이상(non-ideal) 항 $\mathbf{N} = \mathbf{E} + \mathbf{v}\times\mathbf{B}$가 있을 때, $\mathbf{B}\times(\nabla\times\mathbf{N}) = 0$이면 장선은 보존되지만 자속은 보존되지 않고, $\nabla\times\mathbf{N} = 0$이면 자속 보존된다. $\mathbf{N} = \mathbf{u}\times\mathbf{B}+\nabla\Phi$ 형식이면 2D 재결합 유형; 그렇지 않으면 2.5D/3D. **핵심 차이**: 3D에서는 flux velocity가 유일하게 존재하지 않고, 확산 영역 안에서 장선들이 지속적으로 연결을 바꾼다. 2D처럼 "X-point에서만 연결이 바뀐다"는 개념이 성립하지 않는다.

For non-ideal $\mathbf{N}$: if $\mathbf{B}\times(\nabla\times\mathbf{N}) = 0$, field lines are conserved but not flux; if $\nabla\times\mathbf{N}=0$, flux is conserved. When $\mathbf{N} = \mathbf{u}\times\mathbf{B}+\nabla\Phi$, we have 2D-type reconnection; otherwise 2.5D/3D. **Key distinction**: in 3D no unique flux velocity exists, and field lines continuously change connections inside the diffusion region, unlike the 2D picture where only the X-point changes connectivity.

**3D reconnection rate** (Schindler, Hesse & Birn 1988):
$$\int_\text{field line through D} E_\parallel \, dl \neq 0$$
확산 영역(D)을 관통하는 장선을 따라 이 적분의 **모든 장선에 대한 최대값**이 재결합률을 결정한다. Nulls 없이도 가능. Evaluated along a field line through diffusion region D, the maximum over all such lines gives the reconnection rate; possible without nulls.

### Part IV: 2D Reconnection Models (Sects. 7-8) / 2D 재결합 모델

#### 7.1 Sweet-Parker (1958) / 스위트-파커 모델

**가정 / Assumptions**: 길이 $2L$, 두께 $2l$의 정상상태 전류 시트로 $\pm B_i$ 자기장이 속도 $v_i$로 유입 (Fig. 36).

**유도 / Derivation**:
- **확산-유입 균형 / Diffusion-inflow balance**: 확산 시간 $\tau_d \sim l^2/\eta$가 유입 시간 $l/v_i$와 같아야 하므로
  $$v_i = \eta/l \qquad (\text{Eq. 63})$$
- **질량 보존 / Mass conservation** (incompressible): $L v_i = l v_o$ (Eq. 64)
- **출력 속도 = 유입 Alfvén 속도 / Outflow at inflow Alfvén speed**: $\mathbf{j}\times\mathbf{B}$로 플라즈마를 $v_o = v_{Ai} \equiv B_i/\sqrt{\mu\rho}$까지 가속 (Eq. 65)
- 세 식으로부터 $l$ 소거:
  $$\boxed{M_i \equiv \frac{v_i}{v_{Ai}} = \frac{1}{\sqrt{R_{mi}}} = \frac{1}{\sqrt{S}}} \qquad (\text{Eq. 66})$$

**태양 코로나 수치 / Coronal numbers** (p. 66):
- 코로나에서 $R_{me} = L_e v_{Ae}/\eta \sim 10^8$–$10^{12}$
- Sweet-Parker 비 $\sim 10^{-4}$–$10^{-6}$ × Alfvén 속도 → **100초 플레어에는 너무 느림**
- 확산 시간 $\tau_d = L^2/\eta = 10^{-9} L^2 T^{3/2}$ (Eq. 63 인접 텍스트). For $L = 10^7$ m, $T = 10^6$ K: $\tau_d \sim 10^{14}$ s — **huge compared to $\sim 100$ s flare time-scale**

**세 가지 중요한 에너지 관점 / Three important energetics considerations** (p. 66-67):
1. **에너지 분할 / Energy partition**: 유입 자기에너지의 정확히 절반이 열, 나머지 절반이 운동에너지로 전환되어 equipartition 조건 성립. Half of the inflowing magnetic energy is converted to thermal, half to kinetic, giving equipartition.
2. **압력 보정 / Pressure correction**: 유출 압력 $p_o$가 중성점 압력 $p_N$을 초과하면 유출이 감속하여 재결합률이 $(1 + \frac{1}{2}\beta_i(1-p_o/p_i))^{1/4}/\sqrt{R_{mi}}$로 보정됨. When outflow pressure $p_o > p_N$, the outflow slows and the rate is reduced.
3. **압축성 보정 / Compressibility correction**: $\rho_o > \rho_i$일 때 비율이 $(\rho_o/\rho_i)^{1/2}$만큼 증가. Compressibility enhances the rate by $(\rho_o/\rho_i)^{1/2}$ when $\rho_o > \rho_i$.

#### 7.2 Petschek (1964) / 페체크 모델

**핵심 아이디어 / Key idea**: 확산 영역을 전체 시스템 길이 $L_e$가 아니라 훨씬 짧은 $L \ll L_e$로 만들고, 대부분의 에너지 변환을 네 개의 slow-mode shocks에서 수행 (Fig. 37). 유입 에너지의 $2/5$가 열, $3/5$가 운동에너지로 변환.

**Almost-uniform 근사에서 / In the almost-uniform approximation**:
유입 영역에서 $\nabla^2 A_1 = 0$을 풀면
$$B_i = B_e\left(1 - \frac{4M_e}{\pi}\log\frac{L_e}{L}\right) \qquad (\text{Eq. 71})$$
최대 재결합률 ($B_i = B_e/2$ 가정):
$$\boxed{M_e^* \approx \frac{\pi}{8\log R_{me}}} \qquad (\text{Eq. 72})$$

**수치 예 / Numerical example**: $R_{me} = 10^{12}$: $M_e^* \approx \pi/(8 \times 27.6) \approx 0.014$. $R_{me} = 10^8$: $M_e^* \approx 0.021$. → **Sweet-Parker보다 약 $\sqrt{R_{me}}/\ln R_{me}$배 빠름** (예: $10^{12}$에서 $\sim 10^5$배).

**주의 / Caveat**: 균일 저항률에서는 Petschek이 유지되지 않음(Biskamp 1986). Baty et al. (2006, 2009)이 국소적으로 증강된 저항률(Eq. 73: $\eta(x) = (\eta_0 - \eta_1)\exp[-(x/l_x)^2 - (y/l_y)^2]+\eta_1$)에서 Petschek 해를 재현하여 "marginally stable" 결론.

**Key idea**: Make the diffusion region much shorter than the system ($L \ll L_e$), with most energy converted at four slow-mode shocks. Using Laplace's equation in the inflow region, the maximum rate is Petschek's classic $M_e^* \approx \pi/(8\log R_{me})$. For $R_{me} \sim 10^{12}$ this gives $\sim 0.014$—about $10^5$ times faster than Sweet-Parker. Petschek is not maintained with truly uniform resistivity (Biskamp 1986), but Baty et al. (2006) showed it is stable with slightly nonuniform (enhanced at the X-point) resistivity.

**Almost-Uniform family and other regimes** (Sect. 7.3): Priest & Forbes (1986)가 Petschek의 경계 조건을 변화시켜 **Almost-Uniform Reconnection** 일반화. 매개변수 $b$에 따라 (Fig. 40b):
- $b = 0$: 고전 Petschek ($M_i \approx $ 일정, 수평 흐름 작음)
- $b < 0$: slow-mode compression, converging flow (non-potential)
- $0 < b < 1$: Petschek 체제
- $b > 1$: **flux pile-up regime** — diffusion region에 접근할수록 $B$ 증가; 중앙 전류 시트가 Petschek보다 훨씬 길어짐
- $b = 2/\pi$: 수평 흐름 없음 (Petschek의 원래 경우)

Priest & Forbes (1986) generalised Petschek's boundary conditions to the Almost-Uniform family. Depending on parameter $b$: $b=0$ is classic Petschek, $b<0$ gives convergent slow-mode compression, $b>1$ gives flux pile-up with a much longer central current sheet. Different regimes apply depending on whether driving or spontaneous reconnection dominates.

#### 8.3 Plasmoid instability / 플라즈모이드 불안정성

Loureiro, Schekochihin & Cowley (2007)가 Sweet-Parker 시트가 $L/l \sim \sqrt{S}$의 종횡비로 tearing 불안정성에 노출됨을 보임. 성장률과 가장 빠른 wavenumber:

$$\boxed{\gamma_{\max}\tau_A \sim S^{1/4}, \quad k_{\max}L \sim S^{3/8}}$$

여기서 $\tau_A = L/v_A$. 임계 Lundquist number:
$$\boxed{S_c \sim 10^4}$$
(정확히는 $S_c \approx 10^4$–$10^5$, SP 시트 종횡비 $\sim 100$에서의 onset; Loureiro et al. 2005, Samtaney et al. 2009 시뮬레이션 확인)

**비선형 단계의 놀라운 결과 / Striking nonlinear result**: 비선형 포화 후 평균 재결합률이 **$S$에 거의 무관** — 이는 현재 시트 계층구조의 최소 크기가 $S_c$에 의해 결정되고, 그 크기의 SP 재결합률 $\sim 1/\sqrt{S_c}$가 전체 비율을 결정하기 때문 (Uzdensky et al. 2010, Bhattacharjee et al. 2009). 즉 **고전 Sweet-Parker의 "너무 느린" 문제를 해결!**

**Fractal plasmoid cascade / 프랙탈 플라즈모이드 계단구조** (Shibata & Tanuma 2001): 플라즈모이드 분포 $f(\Psi) \sim \Psi^{-2}$, 크기 분포 $f(w) \sim w^{-2}$ (수치실험에서는 $-1$에서 $-2$ 사이). 드물게 중앙에서 "monster plasmoid"가 형성되어 거대한 폭발성 에너지 방출에 연결됨.

**Plasmoid instability in 3D**: Huang & Bhattacharjee (2016), Leake et al. (2020)의 3D 시뮬레이션은 2D와 달리 잘 정의된 플라즈모이드 대신 **난류적 행동**으로 전환됨을 보임—3D에서는 twisted flux ropes가 secondary kinking으로 연쇄 붕괴.

**Tearing mode precursor theory (Sect. 8.1) / 뜯어내기 모드 선구자 이론**

Furth, Killeen & Rosenbluth (FKR 1963)의 선형 분석은 1D 전류 시트 또는 shear 자기장에 세 가지 저항성 불안정성—**gravitational, rippling, tearing**—을 발견. 모든 모드의 시간 척도는 $\tau_d^{(1-\lambda)}\tau_A^\lambda$ ($0 < \lambda < 1$)로 $\tau_d$ 단독보다 훨씬 빠름. Tearing mode는 magnetic islands (2D) 또는 flux ropes (3D)를 형성.

Furth-Killeen-Rosenbluth (FKR 1963) discovered three resistive instabilities in a 1D current sheet: gravitational, rippling, and tearing modes. All have growth time-scales $\tau_d^{(1-\lambda)}\tau_A^\lambda$, much faster than pure diffusion. Tearing creates magnetic islands (2D) or flux ropes (3D).

**Pucci-Velli "ideal tearing" question (Sect. 8.3.1) / Pucci-Velli "이상 뜯어내기" 문제**

Pucci & Velli (2013), Uzdensky & Loureiro (2016)은 플라즈모이드 불안정성의 성장률이 $S\to\infty$에서 발산하는 문제를 지적. Resolution: 시트가 $S^{-1/2}$ SP 종횡비까지 형성되기 **전에** 먼저 tearing이 발동하여, 시트 종횡비가 $S^{-1/3}$ 정도일 때 이미 "ideal tearing"이 시작됨. 이는 시뮬레이션에서 종종 관찰되는 "current sheet가 결코 full SP 종횡비에 도달하지 않는" 현상을 설명.

Pucci-Velli and Uzdensky-Loureiro resolved the divergent growth-rate issue by noting that tearing triggers *before* the sheet reaches SP aspect ratio—"ideal tearing" kicks in at aspect ratio $\sim S^{-1/3}$, explaining why simulations often show sheets never reaching full SP thinning.

Loureiro, Schekochihin & Cowley (2007) showed that Sweet-Parker sheets with aspect ratio $L/l \sim \sqrt{S}$ are tearing-unstable with $\gamma_{\max}\tau_A \sim S^{1/4}$ and $k_{\max}L \sim S^{3/8}$. The critical Lundquist number is $S_c \sim 10^4$. The striking result is that the nonlinear reconnection rate becomes nearly independent of $S$—governed by SP reconnection at the smallest sheet in the hierarchy, with rate $\sim 1/\sqrt{S_c}$. This resolves the Sweet-Parker "too slow" problem. Plasmoids follow a power-law distribution $f(\Psi) \sim \Psi^{-2}$, with rare "monster plasmoids" causing explosive energy release. In 3D the clean plasmoid picture is replaced by a turbulent transition mediated by twisted flux ropes undergoing secondary kinking.

### Part V: 3D Null-Point Reconnection (Sect. 10) / 3D 영점 재결합

초기 kinematic 모델 (Priest & Titov 1996): 두 가지 기본 모드—**spine reconnection** (유입이 fan을 가로지름)과 **fan reconnection** (유입이 spine을 가로지름). 둘 다 kinematic 단계에서 특이점(singularity) 있음. 이후 동적 MHD 시뮬레이션은 실제로는 이 둘이 **분리된 형태로는 거의 일어나지 않고**, 대신 세 가지 일반 모드가 현실적으로 나타남:

**Sect. 10.2.1 — Spine-fan reconnection / 스파인-팬 재결합** (가장 흔함; Pontin et al. 2007a): Shearing 구동으로 spine과 fan이 함께 "plane of collapse"에서 붕괴하며 둘 사이 각도가 좁혀짐 (Fig. 60). 전류 시트는 spine과 fan 중간 각도로 기울어진 localized sheet. Flux가 fan 표면과 spine을 **모두** 가로질러 전달됨. Power-law scaling: reconnection rate $\sim \eta^{0.25}$, peak current $\sim \eta^{-0.6}$ to $\eta^{-0.8}$.

**Sect. 10.2.2 — Torsional spine reconnection / 비틀림 스파인 재결합**: Spine line을 축으로 하는 회전 구동이 spine 주변에 감긴 current tube 형성 (Fig. 62a). Field lines는 spine 주변에서 **counter-rotational slippage**.

**Sect. 10.2.3 — Torsional fan reconnection / 비틀림 팬 재결합**: Spine 근처 회전 흐름이 fan 쪽으로 전파되어 fan 평면 내 current sheet + vortex sheet 형성; Kelvin-Helmholtz 불안정성에 취약하여 복잡한 비선형 진화.

**Oscillatory reconnection** (Sect. 10.3): Null collapse 후 reverse current가 back-pressure로 재반전을 유도하여 주기적으로 재결합 방향 뒤집음. Solar flare의 quasi-periodic pulsations 해석에 사용됨 (McLaughlin et al. 2018).

**3D null point collapse tearing** (Wyper & Pontin 2014a,b, Fig. 47): Current sheet aspect ratio $\sim 50$에서 tearing onset. 비선형 단계에 **다중 3D null**과 twisted flux ropes 형성 — 3D 플라즈모이드 계단구조의 시작.

Spine-fan reconnection (the most common mode) occurs when shearing drives spine and fan to collapse together in a "plane of collapse", forming a current sheet at an intermediate angle that transfers flux across both structures. Torsional spine/fan reconnection is driven by rotational flows: torsional spine wraps current around the spine with counter-rotational slippage, while torsional fan forms a current+vortex sheet across the fan. Oscillatory reconnection arises when back-pressure reverses the collapse cyclically, offering a mechanism for flare quasi-periodic pulsations. Wyper & Pontin (2014) found tearing in 3D null current sheets at aspect ratio $\sim 50$, generating multiple 3D nulls and twisted flux ropes.

### Part VI: Applications (Sects. 11-17) / 응용

- **Separator reconnection** (Sect. 11): 두 null을 잇는 장선에서의 재결합; 광구 자기그램에서 statistical analysis로 발견됨 (Platten et al. 2014).
- **Quasi-separator/HFT reconnection** (Sect. 12): "slip-running" 또는 "slipping" 재결합 (Aulanier et al. 2006)—장선이 연속적으로 미끄러짐. Titov et al. (2009)의 **slip-forth/slip-back squashing factors** $Q_{sf}, Q_{sb}$로 진단.
- **Flare 3D paradigm** (Sect. 13): "CSHKP" 2D 모델에서 3D 확장으로—flare ribbons는 QSL 발자취로 해석.
- **Coronal heating by reconnection** (Sect. 14): Parker(1972, 1983)의 braid-based nanoflare 아이디어가 "flux tube tectonics" (Priest et al. 2002)로 발전. Magnetogram resolution이 높아질수록 더 많은 separators와 nulls 발견—코로나 가열의 상당 부분이 작은 규모 재결합에서 기인 가능.
- **Solar wind interchange reconnection** (Sect. 15.2): 열린-닫힌 자기장 사이 재결합이 slow solar wind 공급에 기여 (Fisk et al. 1998; Antiochos et al. 2011).
- **Plasmoid instability in low atmosphere** (Sect. 8.3.2, end): Rouppe van der Voort et al. (2017)이 chromospheric UV bursts에서 플라즈모이드 관측. Peter et al. (2019) 시뮬레이션: low-$\beta$에서 재결합 효율적, 예상 peak $T \approx 0.2$ MK.

- Separator reconnection (along a field line joining two nulls): detected statistically in photospheric magnetograms (Platten et al. 2014)
- Quasi-separator/HFT reconnection gives "slip-running" reconnection, diagnosed by slip-forth/slip-back squashing factors
- 3D flare paradigm: flare ribbons are footprints of QSLs
- Coronal heating: Parker's braid-based nanoflare developed into "flux tube tectonics"; higher-resolution magnetograms reveal more nulls and separators
- Interchange reconnection supplies slow solar wind
- Chromospheric UV bursts show plasmoid-mediated reconnection with predicted $T \sim 0.2$ MK

### Part VII: Open Questions and Outlook (Sect. 18) / 미해결 문제와 전망

저자들은 결론에서 다음과 같은 미해결 문제를 지적한다:

The authors close by highlighting open questions:

1. **3D 플라즈모이드의 비선형 진화 / Nonlinear evolution of 3D plasmoid instability**: 2D는 잘 이해되었으나 3D는 난류로 가는 전이가 부분적으로만 연구됨. 3D is only partially understood—the transition to turbulence needs more work.

2. **Onset 조건 / Onset conditions**: 재결합이 current sheet 형성 즉시 시작하는가, 아니면 어떤 임계값이 있는가? 이는 solar flare prediction의 핵심. Is there a threshold for reconnection onset once a current sheet forms? Critical for flare prediction.

3. **Partial ionisation in the lower atmosphere / 하층 대기의 부분 이온화**: Ambipolar diffusion의 재결합에 대한 완전한 영향은 아직 해명되지 않음. Full account of ambipolar diffusion effects on reconnection has not yet been given.

4. **Turbulence-driven reconnection / 난류 주도 재결합**: Lazarian & Vishniac (1999) 이후 난류가 재결합률을 증강시키는 메커니즘은 여전히 논의중. Turbulent enhancement of the reconnection rate remains controversial.

5. **Particle acceleration / 입자 가속**: 직접 전기장 가속, shocks, 난류 중 어느 것이 flare의 비열 입자를 주로 만드는가? Which mechanism dominates nonthermal particle acceleration?

6. **Universal reconnection rate / 보편 재결합비**: 왜 모든 체제(collisional, collisionless, bursty)가 비슷한 $(0.01$–$0.1)v_A$를 주는가? Priest et al. (2021)의 "ideal-region-dominated" 가설이 아직 완전히 검증되지 않음. Why do all regimes give similar rates?

---

## 3. Key Takeaways / 핵심 시사점

1. **Sweet-Parker is too slow by orders of magnitude / 스위트-파커는 자릿수 차이로 너무 느림** — $M_A = 1/\sqrt{S}$에서 코로나 $S \sim 10^{12}$를 대입하면 재결합률이 Alfvén 속도의 $10^{-6}$ 수준이어서 플레어의 $\sim 100$초 에너지 방출을 결코 설명할 수 없다. 이것이 20세기 중반부터 fast reconnection 메커니즘이 핵심 연구 주제였던 이유다. Plugging $S \sim 10^{12}$ into $M_A = 1/\sqrt{S}$ gives $\sim 10^{-6} v_A$—utterly insufficient for flare time-scales and the primary motivation for faster mechanisms.

2. **Petschek achieves fast reconnection through geometry / 페체크는 기하학으로 빠른 재결합 달성** — 핵심은 $L \ll L_e$인 작은 확산 영역 + 네 개의 slow-mode shocks이다. $M_e^* \approx \pi/(8\ln S)$는 $\ln$ 의존이라 $S$가 매우 커도 $0.01$–$0.1$ 수준. Petschek의 타당성 논쟁(균일 저항률에서 유지되지 않음)은 Baty et al. (2006)이 국소적으로 증강된 저항률에서 해결. Petschek's rate $M_e^* \approx \pi/(8\ln S)$ depends only logarithmically on $S$; the Biskamp (1986) critique was resolved by Baty et al. (2006) with slightly nonuniform resistivity.

3. **The plasmoid instability resolves SP bottleneck / 플라즈모이드 불안정성이 SP 병목을 해결** — Sweet-Parker 시트가 $S > S_c \sim 10^4$에서 tearing으로 분열하여, 재결합률이 $S$와 거의 무관하게 되고 고전 SP 예측 대비 수 자릿수 빨라진다. 성장률 $\gamma \tau_A \sim S^{1/4}$, wavenumber $kL \sim S^{3/8}$, monster plasmoids가 폭발적 방출을 유발. Sweet-Parker sheets tear for $S > S_c \sim 10^4$, giving an almost $S$-independent effective rate and resolving the classic bottleneck.

4. **3D reconnection is fundamentally different from 2D / 3D 재결합은 2D와 근본적으로 다름** — (a) Flux velocity가 유일하게 존재하지 않음; (b) 재결합률은 $\int E_\parallel \, dl$의 확산 영역 관통 장선들에 대한 최대값; (c) Null이 없어도 재결합 가능(QSL, HFT에서). "X-point에서만 연결이 바뀐다"는 2D 직관은 3D에서 완전히 깨진다. The single flux velocity disappears, the rate is the maximum of $\int E_\parallel \, dl$, and reconnection occurs at QSLs/HFTs without nulls.

5. **QSLs and HFTs are real sites of reconnection / QSL과 HFT는 실제 재결합 장소** — Squashing factor $Q \gg 2$로 진단되는 영역이 관측된 flare ribbons와 잘 일치. Slip-running reconnection은 QSL을 따라 장선이 연속적으로 연결을 바꾸는 현상. 활동 영역에서 흔히 발견되는 구조. QSL regions with $Q \gg 2$ match observed flare ribbons; slip-running reconnection along QSLs is common in active regions.

6. **Spine-fan reconnection is the generic 3D null mode / 스파인-팬은 일반적 3D null 모드** — Priest & Titov(1996)의 idealised spine/fan 모드는 실제로는 분리되어 일어나지 않고, shearing driving은 거의 항상 둘이 함께 "plane of collapse"로 붕괴하는 spine-fan 모드를 만든다. 이 모드에서 flux는 spine과 fan을 모두 가로질러 전달된다. Idealised pure spine and pure fan modes rarely occur; shearing drives spine and fan to collapse together, transferring flux across both.

7. **Nanoflare heating revisited via braiding and tectonics / Nanoflare 가열 재해석: braid와 tectonics** — Parker(1983)의 braided field 아이디어는 "flux tube tectonics" (Priest et al. 2002)와 3D braided-field 시뮬레이션으로 정량화되고 있다. 고해상도 magnetogram이 많은 null과 separator를 드러내며, 작은 규모 재결합의 코로나 가열 기여가 점점 더 중요해 보임. Parker's braid-based nanoflare concept is quantified by modern braid simulations and "flux tube tectonics", with more nulls/separators found as magnetogram resolution increases.

8. **The $(0.01-0.1)v_A$ "universal rate" arises from ideal-region dominance / $(0.01$-$0.1)v_A$ "보편 비율"은 이상 영역이 지배하기 때문** — 충돌형(Petschek), 충돌없는(Hall), bursty(plasmoid) 모두 비슷한 대규모 재결합률 $(0.01$–$0.1)v_A$를 준다. Priest et al. (2021)은 이것이 재결합률이 중심 전류 시트 주변의 **이상 영역(ideal region)** 으로 주로 결정되며 microphysics와 시트 길이에 약한 의존성만 갖기 때문이라고 설명. Collisional, collisionless, and bursty regimes all give $(0.01$–$0.1)v_A$ because the ideal region around the diffusion region dominates, depending only weakly on microphysics.

---

## 4. Mathematical Summary / 수학적 요약

**1. Induction equation (basis of MHD reconnection) / MHD 재결합의 출발점 유도 방정식:**
$$\frac{\partial \mathbf{B}}{\partial t} = \nabla\times(\mathbf{v}\times\mathbf{B}) + \frac{\eta}{\mu_0}\nabla^2\mathbf{B}$$
- 첫 항: advection (frozen-in flux). The advection term gives frozen-in flux in ideal MHD.
- 둘째 항: diffusion — $\eta$가 유한할 때만 연결성 변화. The diffusion term breaks connectivity when $\eta > 0$.
- 비차원화: $R_m = S = L v_A/\eta$. Non-dimensionalisation defines $R_m = S$.

**2. Lundquist number / 런드퀴스트 수:**
$$S = \frac{L v_A}{\eta}$$
코로나 / Corona: $S \sim 10^{12}$–$10^{14}$; 지구 자기권 / magnetosphere: $S \sim 10^{13}$; 실험실 / lab: $S \sim 10^3$–$10^5$.

**3. Sweet-Parker rate (Eqs. 63-66) / 스위트-파커 비:**
$$\boxed{M_A = \frac{v_i}{v_{Ai}} = \frac{1}{\sqrt{S}}, \qquad l = \frac{L}{\sqrt{S}}, \qquad v_o = v_{Ai}}$$
유도는 (i) 확산-유입 균형 $v_i = \eta/l$, (ii) 질량 보존 $Lv_i = lv_o$, (iii) 출력 Alfvén 속도 $v_o = v_{Ai}$의 세 방정식에서. Derived from (i) diffusion-advection balance, (ii) mass conservation, (iii) outflow at Alfvén speed.

**4. Petschek maximum rate (Eq. 72) / 페체크 최대 비:**
$$\boxed{M_e^* \approx \frac{\pi}{8 \log R_{me}}}$$
- 유입 영역 Laplace 방정식 해 (Eq. 71): $B_i = B_e(1 - (4M_e/\pi)\log(L_e/L))$
- $B_i = B_e/2$에서 확산 영역이 "자기 차단" 되는 지점
- Solution of Laplace's eq in inflow region; self-choking when $B_i = B_e/2$

**5. 3D reconnection rate (Schindler et al. 1988) / 3D 재결합비:**
$$\boxed{\text{rate} = \max_{\text{field lines through D}} \int E_\parallel \, dl}$$
- $E_\parallel = (\mathbf{E}\cdot\mathbf{B})/B$
- D = 확산 영역 / diffusion region
- Null points not required — flux changes continuously

**6. Squashing factor (Eq. 7) / 찌그러짐 인자:**
$$Q = \frac{N^2}{|B_{n-}/B_{n+}|}, \quad N^2 = \sum_{i,j}\left(\frac{\partial X_i^+}{\partial x_j^-}\right)^2$$
- 장선 발자국 대응사상 $(x^-,y^-) \to (X^+, Y^+)$. Field-line footpoint mapping.
- 원이 타원으로 매핑될 때 비율 $\approx Q$ (large $Q$ limit). Aspect ratio of mapped circle is $\approx Q$ when $Q$ large.
- QSL: $Q \gg 2$; separatrix: $Q \to \infty$

**7. Plasmoid instability scaling / 플라즈모이드 불안정성 스케일링:**
$$\boxed{\gamma_{\max}\tau_A \sim S^{1/4}, \qquad k_{\max} L \sim S^{3/8}, \qquad S_c \sim 10^4}$$
- $\tau_A = L/v_A$
- Critical SP sheet aspect ratio $L/l \sim 100$
- Nonlinear rate: $M_A^{\rm eff} \sim 1/\sqrt{S_c} \sim 0.01$ (nearly $S$-independent)

**8. Null-point field structures / 영점 자기장 구조:**
- 2D null (Eq. 3): $\mathbf{B} = (B_0/r_0)(y, \bar\alpha^2 x)$
- 3D proper radial null (Eq. 5): $\mathbf{B} = (x, y, -2z)$
- Field lines: $y = Cx$, $z = K/x^2$ (spine = z-axis, fan = xy-plane)
- General linear null (Eq. 6): 4 parameters $(a, b, j_\|, j_\perp)$

**9. Worked example: coronal Sweet-Parker time / 작업 예제: 코로나 SP 시간**
- $L = 10^7$ m (typical active-region length)
- $v_A = 10^6$ m/s (Alfvén speed)
- $\eta = 1$ m²/s (Spitzer, $T = 10^6$ K)
- $S = Lv_A/\eta = 10^{13}$
- Sweet-Parker time $\tau_{SP} = L/v_i = L\sqrt{S}/v_A = 10 \sqrt{10^{13}} \approx 3 \times 10^7$ s $\approx 1$ year
- Observed flare: $\sim 100$ s → need $\sim 3\times 10^5$ times faster
- Petschek gives $M_e^* \approx \pi/(8\cdot 30) \approx 0.013$ → $\tau_P \approx L/(0.013\, v_A) \approx 800$ s ✓ flare-consistent
- Plasmoid: effective rate $\sim 0.01$ → similar time-scale ✓

**10. Worked example: plasmoid cascade in coronal sheet / 작업 예제: 코로나 시트 내 플라즈모이드 계단구조**

시작 조건 / Starting condition: Sweet-Parker sheet with $L = 10^7$ m, $S = 10^{13}$.

- SP 시트 두께 / SP sheet thickness: $l_{SP} = L/\sqrt{S} = 10^{7}/\sqrt{10^{13}} \approx 3$ m
- 시트 종횡비 / Aspect ratio: $L/l_{SP} = \sqrt{S} \approx 3\times 10^{6} \gg 100$ → **severely unstable**
- Fastest growing mode: $k_{\max}L \sim S^{3/8} = (10^{13})^{3/8} \approx 10^{4.9} \approx 8\times 10^4$ → wavelength $\lambda \approx L/(8\times 10^4) \approx 125$ m
- Growth rate: $\gamma_{\max} \tau_A \sim S^{1/4} \approx 5600$, with $\tau_A = L/v_A = 10$ s → $\gamma_{\max} \sim 560$ s$^{-1}$, growth time $\sim 2$ ms(!)
- Number of plasmoids at onset: $\sim k_{\max}L \sim 10^5$
- Cascade terminates at level where local $S \sim S_c \sim 10^4$; at each level plasmoid sheets are $\sim \sqrt{S_c}\approx 100$ 배 smaller than parent
- Effective rate: $\sim 1/\sqrt{S_c} \approx 0.01\, v_A$ ✓ consistent with observed fast reconnection

실제 코로나에서 플라즈모이드는 SDO/AIA와 RHESSI로 "blobs"로 관측됨 (Lin et al. 2005, Takasao et al. 2011). Plasmoids observed as "blobs" by SDO/AIA and RHESSI.

**11. QSL location in active region (Sect. 2.6.2 example) / 활동 영역에서의 QSL 위치**

Titov & Démoulin (1999)의 모델: force-free flux rope (major radius $R$, minor radius $a$, line current $I_0$) + 두 개의 subphotospheric charges $(-q, q)$ at $\pm L$. 결과적으로 photospheric $Q$ map이 "fishhook" 모양의 매우 얇은 QSL을 보이며, 이는 관측된 flare ribbons의 형태와 매우 일치 (Fig. 15a). Photospheric $Q$ map shows fishhook-shaped thin QSLs matching observed flare ribbons.

이 모델은 저자들이 "classic eruption scenario"로 사용하며, 2D CSHKP 모델의 직접적 3D 확장. Used as the "classic eruption scenario", a direct 3D extension of the 2D CSHKP flare model.

**12. Sweet-Parker vs Petschek comparison table / SP vs Petschek 비교표**

| 항목 / Item | Sweet-Parker | Petschek |
|---|---|---|
| Rate / 재결합비 | $1/\sqrt{S}$ | $\pi/(8\ln S)$ |
| $S=10^{12}$ value | $10^{-6}$ | $\sim 0.014$ |
| Ratio / 비율 | — | $\sim 10^{5}\times$ faster |
| Diffusion region length | $L_e$ (system-wide) | $L \ll L_e$ (small) |
| Energy conversion / 에너지 변환 | Entire sheet | 4 slow-mode shocks |
| Resistivity requirement / 저항률 조건 | Uniform OK | Nonuniform (enhanced at X-point) |
| Observational match to flares | No (too slow) | Yes |

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
Year  | Development / 발전
------+-------------------------------------------------------------
1947  | Giovanelli: neutral-point E-fields accelerate flare particles
1953  | Dungey: X-point collapse instability
1957  | Parker: dynamical dissipation (Paper #6)
1958  | Sweet-Parker: first quantitative model (too slow)
1961  | Dungey: magnetospheric reconnection
1963  | Furth-Killeen-Rosenbluth: tearing mode instability
1964  | Petschek: fast reconnection with slow shocks
1983  | Parker: nanoflare/braiding (Paper #27)
1986  | Biskamp: numerical challenge to Petschek
1988  | Schindler-Hesse-Birn: 3D general magnetic reconnection condition
1990  | Lau-Finn: 3D null spine/fan classification
1995  | Priest-Démoulin: quasi-separatrix layers
1996  | Priest-Titov: spine & fan reconnection at 3D nulls
2002  | Priest-Heyvaerts-Title: flux tube tectonics (coronal heating)
2006  | Baty-Forbes-Priest: Petschek with nonuniform resistivity
2007  | Loureiro-Schekochihin-Cowley: plasmoid instability
       | Titov: slip-squashing factors Q_sf, Q_sb
2009  | Bhattacharjee et al.: nonlinear plasmoid simulations
2014  | Wyper-Pontin: 3D null tearing
2017  | Rouppe van der Voort: observed chromospheric plasmoids
2022  | Pontin-Priest: THIS REVIEW — comprehensive MHD synthesis
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **#6 Parker (1957) "Sweet's mechanism for merging"** | Sweet-Parker의 물리적 토대를 Parker가 정식화. Pontin-Priest Sect. 7.1의 직접적 역사적 뿌리. | Pontin-Priest Section 7.1 builds directly on Parker's formalisation of Sweet's merging picture. |
| **#27 Parker (1983) "Magnetic neutral sheets in evolving fields"** | Braided field의 자발적 current sheet 형성 → Pontin-Priest Sect. 5.5 (braided fields) + Sect. 14 (coronal heating) | Parker's braiding argument underlies the "flux tube tectonics" framework of Sects. 5.5 & 14. |
| **Loureiro, Schekochihin & Cowley (2007)** | 플라즈모이드 불안정성의 원 논문; Pontin-Priest Sect. 8.3의 중심. | Original plasmoid instability paper — central reference of Sect. 8.3. |
| **Schindler, Hesse & Birn (1988)** | $\int E_\parallel \, ds \neq 0$ 3D 재결합 조건; Sect. 4의 출발점. | Provides the 3D reconnection condition — starting point of Sect. 4. |
| **Priest & Forbes (2000) "Magnetic Reconnection"** | 이전 표준 교과서. Pontin-Priest는 이를 출발점으로 3D 및 플라즈모이드 발전을 추가. | Previous standard textbook; this review updates it with 3D and plasmoid developments. |
| **Priest (2014) textbook** | 직접적 전신. Pontin-Priest는 Priest(2014) 이후의 새로운 발전을 강조. | Direct predecessor text; this review emphasises post-2014 developments. |
| **Aulanier et al. (2006) "Slip-running reconnection"** | QSL을 따라 미끄러지는 재결합; Sect. 12의 기반. | Slip-running reconnection along QSLs — basis of Sect. 12. |

---

## 7. References / 참고문헌

**Primary reference / 주 참고문헌:**
- Pontin, D. I., & Priest, E. R. (2022). Magnetic reconnection: MHD theory and modelling. *Living Reviews in Solar Physics*, 19(1), 1. DOI: [10.1007/s41116-022-00032-9](https://doi.org/10.1007/s41116-022-00032-9)

**Classic foundational papers / 고전 토대 논문:**
- Parker, E. N. (1957). Sweet's mechanism for merging magnetic fields in conducting fluids. *JGR*, 62, 509. (Paper #6 in this collection)
- Parker, E. N. (1983). Magnetic neutral sheets in evolving fields. *ApJ*, 264, 635. (Paper #27)
- Sweet, P. A. (1958). The neutral point theory of solar flares. *IAU Symp.* 6, 123.
- Petschek, H. E. (1964). Magnetic field annihilation. *NASA Spec. Publ.* 50, 425.
- Furth, H. P., Killeen, J., & Rosenbluth, M. N. (1963). Finite-resistivity instabilities of a sheet pinch. *Phys. Fluids*, 6, 459.
- Schindler, K., Hesse, M., & Birn, J. (1988). General magnetic reconnection, parallel electric fields, and helicity. *JGR*, 93, 5547.
- Priest, E. R., & Démoulin, P. (1995). Three-dimensional magnetic reconnection without null points. *JGR*, 100, 23443.
- Priest, E. R., & Titov, V. S. (1996). Magnetic reconnection at three-dimensional null points. *Phil. Trans. Roy. Soc. A*, 354, 2951.
- Loureiro, N. F., Schekochihin, A. A., & Cowley, S. C. (2007). Instability of current sheets and formation of plasmoid chains. *Phys. Plasmas*, 14, 100703.
- Bhattacharjee, A., Huang, Y.-M., Yang, H., & Rogers, B. (2009). Fast reconnection in high-Lundquist-number plasmas due to the plasmoid instability. *Phys. Plasmas*, 16, 112102.
- Baty, H., Forbes, T. G., & Priest, E. R. (2006). The effect of nonuniform resistivity in Petschek reconnection. *Phys. Plasmas*, 13, 022312.
- Titov, V. S. (2007). Generalised squashing factors for covariant description of magnetic connectivity in the solar corona. *ApJ*, 660, 863.
- Wyper, P. F., & Pontin, D. I. (2014b). Non-linear tearing of 3D null-point current sheets. *Phys. Plasmas*, 21, 082114.
- Priest, E. R. (2014). *Magnetohydrodynamics of the Sun*. Cambridge University Press.

**Additional cited works / 추가 인용 문헌:**
- Giovanelli, R. G. (1947). Magnetic and electric phenomena in the Sun's atmosphere associated with sunspots. *MNRAS*, 107, 338.
- Dungey, J. W. (1953). Conditions for the occurrence of electrical discharges in astrophysical systems. *Phil. Mag.*, 44, 725.
- Dungey, J. W. (1961). Interplanetary magnetic field and the auroral zones. *Phys. Rev. Lett.*, 6, 47.
- Biskamp, D. (1986). Magnetic reconnection via current sheets. *Phys. Fluids*, 29, 1520.
- Priest, E. R., & Forbes, T. G. (1986). New models for fast steady state magnetic reconnection. *JGR*, 91, 5579.
- Titov, V. S., Hornig, G., & Démoulin, P. (2002). Theory of magnetic connectivity in the solar corona. *JGR*, 107, 1164.
- Aulanier, G., Pariat, E., Démoulin, P., & DeVore, C. R. (2006). Slip-running reconnection in quasi-separatrix layers. *Solar Phys.*, 238, 347.
- Uzdensky, D. A., Loureiro, N. F., & Schekochihin, A. A. (2010). Fast magnetic reconnection in the plasmoid-dominated regime. *Phys. Rev. Lett.*, 105, 235002.
- Pucci, F., & Velli, M. (2013). Reconnection of quasi-singular current sheets: The "ideal" tearing mode. *ApJL*, 780, L19.
- Priest, E. R., Chitta, L. P., & Syntelis, P. (2021). A cancellation nanoflare model for solar chromospheric and coronal heating II — 2D theory. *ApJ*, 910, 49.
