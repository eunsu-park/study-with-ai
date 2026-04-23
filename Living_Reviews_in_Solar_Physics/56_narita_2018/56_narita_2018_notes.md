---
title: "Space-Time Structure and Wavevector Anisotropy in Space Plasma Turbulence"
authors: Narita, Y.
year: 2018
journal: "Living Reviews in Solar Physics"
doi: "10.1007/s41116-017-0010-0"
topic: Living_Reviews_in_Solar_Physics
tags: [turbulence, wavevector-anisotropy, MHD, kinetic-waves, dispersion-relation, solar-wind, multi-spacecraft, MMS, Cluster, Taylor-hypothesis, critical-balance, Goldreich-Sridhar]
status: completed
date_started: 2026-04-23
date_completed: 2026-04-23
---

# 56. Space-Time Structure and Wavevector Anisotropy in Space Plasma Turbulence / 우주 플라즈마 난류에서의 시공간 구조와 파수벡터 이방성

---

## 1. Core Contribution / 핵심 기여

**English**: Narita (2018) is a comprehensive review that recasts space plasma turbulence as an intrinsically **space-time (k-ω domain)** phenomenon rather than a purely spectral one. Traditional solar-wind turbulence studies analyze a one-dimensional energy spectrum: either E(ω) from single-spacecraft time series or E(k) from numerical simulation snapshots. Narita shows these two 1D spectra are just projections of a higher-dimensional spectrum E(k, ω) over the wavenumber-frequency plane. The review builds a hierarchy of models — from the hydrodynamic random-sweeping model of Kraichnan (1964) with Doppler shift k·U_0 and Doppler broadening σ = k·δU, through magnetohydrodynamic (MHD) extensions with Alfvén waves forward and backward, up to the kinetic wave modes (ion cyclotron, whistler, kinetic Alfvén, ion Bernstein) — and links these to the observed spectral slope k^(-5/3) or k^(-3/2). In parallel, the review surveys wavevector anisotropy models (two-component slab+2D, Goldreich-Sridhar critical balance k_∥ ∝ k_⊥^(2/3), elliptic, non-elliptic, asymmetric) and explains why single-spacecraft spectra with flow direction parallel to B_0 give slope ≈ -2 while perpendicular flow gives ≈ -5/3 (Horbury et al. 2008).

**한국어**: Narita (2018)는 우주 플라즈마 난류를 단순 스펙트럼이 아닌 **본질적으로 시공간(k-ω 영역)** 현상으로 재정의하는 포괄적 리뷰이다. 기존 태양풍 난류 연구는 1차원 에너지 스펙트럼 — 단일 위성 시계열에서의 E(ω) 또는 수치 시뮬레이션 스냅샷에서의 E(k) — 에 집중해 왔다. Narita는 이 두 1D 스펙트럼이 고차원 스펙트럼 E(k, ω)의 파수-주파수 평면으로의 투영일 뿐임을 보인다. 리뷰는 Kraichnan(1964)의 유체 random-sweeping 모델 (Doppler shift k·U_0, Doppler broadening σ = k·δU) 에서 시작하여, 전방·후방 전파 Alfvén 파를 포함한 MHD 확장, 나아가 이온-cyclotron, whistler, kinetic Alfvén, ion Bernstein 과 같은 운동학적(kinetic) 파동 모드까지 모델 계층을 구축하며, 관측되는 스펙트럼 기울기 k^(-5/3) 또는 k^(-3/2) 와 연결한다. 한편 파수 벡터 이방성 모델들 — 2-성분 (slab + 2D), Goldreich-Sridhar 임계 균형 k_∥ ∝ k_⊥^(2/3), elliptic, non-elliptic, 비대칭 — 을 정리하며, 단일 위성에서 흐름 방향이 B_0 와 평행할 때 기울기가 ≈ -2, 수직일 때 ≈ -5/3 이 되는 이유 (Horbury et al. 2008) 를 설명한다.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction (Section 1, pp. 2-7) / 서론

**English**: The review opens by listing turbulent plasma environments: solar photospheric convection, solar wind, planetary magnetospheres, shock-upstream/downstream regions. Two characteristic properties of plasma turbulence make it harder than fluid turbulence:

1. **Coupling with electromagnetic fields** — plasmas are electrically conducting, so turbulent energy transport is mediated by waves (Alfvén, fast, slow) in addition to eddies.
2. **Collisionlessness** — in the solar wind the mean free path is of order the system size, so energy dissipation requires wave-particle interactions (Landau, cyclotron resonance), not binary collisions.

Reynolds number estimates for astrophysical plasmas: Earth's outer core Re ~ 10^8, solar convection zone Re ~ 10^10, galaxy Re ~ 10^11. Solar-wind spectral indices are observed to lie in the range −1.5 to −1.7 at MHD scales (Coleman 1968; Matthaeus-Goldstein 1982), with a spectral break near 0.1-1 Hz (spacecraft frame) followed by steeper slopes in the kinetic range (Leamon et al. 1998; Alexandrova et al. 2009; Sahraoui et al. 2009).

The paper highlights Fig. 1 — a schematic of how observers get E(ω) from time series (spacecraft), while simulators get E(k) from spatial snapshots. The two can be connected via Taylor's frozen-in-flow hypothesis (1938), but only when the advective convection dominates over intrinsic wave propagation and over large-scale sweeping velocity.

**한국어**: 리뷰 서론은 난류 플라즈마 환경 — 태양 광구 대류, 태양풍, 행성 자기권, 충격파 상·하류 — 을 나열한다. 플라즈마 난류를 유체 난류보다 어렵게 만드는 두 가지 특성:

1. **전자기장과의 결합** — 플라즈마는 전도성이므로, 난류 에너지 수송이 에디(eddy)뿐 아니라 파동(Alfvén, fast, slow)을 매개로 한다.
2. **무충돌성(collisionlessness)** — 태양풍에서 평균 자유 경로가 시스템 크기 수준이므로, 에너지 소산이 이항 충돌이 아닌 파동-입자 상호작용(Landau, cyclotron 공명)에 의존한다.

천체물리적 플라즈마의 Reynolds 수 추정치: 지구 외핵 Re ~ 10^8, 태양 대류층 Re ~ 10^10, 은하 Re ~ 10^11. 태양풍의 관측 스펙트럼 지수는 MHD 스케일에서 −1.5 에서 −1.7 범위이며 (Coleman 1968; Matthaeus-Goldstein 1982), 위성 좌표계 기준 0.1-1 Hz 근처에서 스펙트럼 꺾임 (break) 이 발생하고 운동학 영역에서 더 가파른 기울기로 이어진다 (Leamon et al. 1998; Alexandrova et al. 2009; Sahraoui et al. 2009).

논문은 Fig. 1 — 관측자는 시계열로 E(ω)를, 시뮬레이터는 공간 스냅샷으로 E(k)를 얻는다는 개략도 — 을 강조한다. 둘은 Taylor (1938)의 frozen-in-flow 가설로 연결되나, 이는 대류가 내재 파동 속도와 대규모 sweeping 속도를 압도할 때만 유효하다.

### Part II: Space-Time Structure (Section 2, pp. 7-27) / 시공간 구조

#### 2.1 Hydrodynamic picture — Random sweeping model (pp. 7-11)

**English**: The key setup: total velocity field is **U_0 + δU + u**, where U_0 is the constant mean flow, δU is large-scale flow variation, and u is the fully-developed small-scale turbulence. Ideal convection equation:

$$\left(\frac{\partial}{\partial t} + (\mathbf{U}_0 + \delta\mathbf{U})\cdot\nabla\right)\mathbf{u} = 0 \qquad \text{(Eq. 6)}$$

Solving in k-space: u(k, t) = exp[-i k·(U_0 + δU) t] u(k, 0). Ensemble averaging with an isotropic Gaussian δU gives the random-sweeping energy spectrum:

$$E(\mathbf{k}, \omega) = \frac{E(\mathbf{k})}{\sqrt{2\pi k^2 (\delta U)^2}} \exp\left[-\frac{(\omega - \mathbf{k}\cdot\mathbf{U}_0)^2}{2 k^2 (\delta U)^2}\right] \qquad \text{(Eq. 10)}$$

This is a Gaussian in ω centered at the Doppler shift ω = k·U_0 with standard deviation σ = k δU (Doppler broadening). In the vanishing-sweeping limit δU → 0, F(k, ω) → δ(ω − k·U_0), recovering Taylor's hypothesis (Eq. 13). For the streamwise k_flow:

$$E(k_{flow}, \omega) = E(k_{flow})\,\delta(\omega - k_{flow} U_0) \qquad \text{(Eq. 13)}$$

The spectral index is invariant between frequency and wavenumber domains for an infinitely long inertial range:
$$E(k) = C_K \epsilon^{2/3} k^{-5/3}, \quad E(\omega) = C(U_0, \delta U) C_K \epsilon^{2/3} |\omega|^{-5/3} \qquad \text{(Eqs. 14, 15)}$$

The Lagrangian frequency spectrum steepens to ω^(-2) (Tennekes 1975):
$$E(\omega) \propto \omega^{-2} \quad \text{(Lagrangian)} \qquad \text{(Eq. 4)}$$

**Validity of Taylor's hypothesis**. Narita (2017a) defines the validity parameter I by integrating over a Doppler-shift bin:
$$I = \text{erf}(\Delta\tau), \quad \Delta\tau \simeq \frac{1}{\sqrt{2}}\frac{U_0}{\delta U}\frac{\Delta\omega}{\omega} \qquad \text{(Eqs. 19, 23)}$$
Taylor is valid (I ≈ 1) when U_0 / δU is large; when δU is large the spectrum spreads over both positive and negative ω (backward sweeping motion).

**한국어**: 기본 설정: 전체 속도장은 **U_0 + δU + u** (U_0: 평균 흐름, δU: 대규모 변동, u: 소규모 발전된 난류). 이상 대류 방정식 (Eq. 6) 을 k-공간에서 풀어 u(k, t) = exp[-i k·(U_0 + δU) t] u(k, 0) 를 얻고, 등방 Gaussian δU 로 앙상블 평균하면 random-sweeping 스펙트럼 (Eq. 10) 이 얻어진다.

이는 Doppler shift ω = k·U_0 를 중심으로 표준편차 σ = k δU 의 Gaussian (Doppler broadening) 이다. sweeping 소멸 극한 δU → 0 에서 F → δ 함수가 되어 Taylor 가설이 복원된다 (Eq. 13). 관성 영역이 무한히 길 경우 스펙트럼 지수는 ω 영역과 k 영역에서 동일 (Eqs. 14, 15). Lagrangian 주파수 스펙트럼은 더 가팔라져 ω^(-2) (Tennekes 1975).

Taylor 가설의 유효 기준은 Narita(2017a) 의 I = erf(Δτ) (Eq. 19-23) 로 정량화되며, U_0 / δU 가 클 때 I ≈ 1 이고, δU 가 크면 스펙트럼이 양·음 주파수에 퍼져 sweeping 에 의한 역류 운동이 나타난다.

#### 2.2 MHD picture — Strong vs weak, wave approach (pp. 11-13)

**English**: Two regimes:
- **Strong turbulence**: fluctuation amplitudes alter the mean field. Treated via mixing-length, eddy viscosity, Alfvén time, k-ε models (Biskamp 2003; Yokoi 2006).
- **Weak turbulence**: fluctuations are much smaller than the mean field, and the turbulence is composed of linear-mode waves with amplitudes appearing only as dispersion-relation corrections:
$$\omega = \omega_0(\mathbf{k}) + \frac{\partial\omega}{\partial(A^2)}\bigg|_{\omega_0} + \mathcal{O}(A^4) \qquad \text{(Eq. 24)}$$

Wave approach. For counter-propagating Alfvén waves in a mean flow:
$$\omega_\pm = \mathbf{k}\cdot\mathbf{U}_0 \pm \mathbf{k}\cdot\mathbf{V}_A \qquad \text{(Eqs. 25, 27)}$$

When V_A is large (V_A ≈ 0.5 U_0), the two Alfvén branches are clearly resolved in E(k, ω) (Fig. 4 top). When V_A is small (V_A ≈ 0.2 U_0), the branches merge into an enhanced Doppler-broadened peak (Fig. 4 bottom). The energy spectrum for random MHD sweeping is:
$$E(\mathbf{k}, \omega) = \left(F^{(+)}(\mathbf{k}, \omega) + F^{(-)}(\mathbf{k}, \omega)\right) E(\mathbf{k}) \qquad \text{(Eq. 29)}$$
with Gaussian widths (σ^(±))^2 = |k·(δU ± δV_A)|^2 (Eq. 31).

The cross helicity h_c = ⟨δU·δV_A⟩ (Eq. 32) measures imbalance between forward/backward Alfvén waves — typically high (|h_c| near 1) near the Sun, decreasing with heliocentric distance.

**한국어**: 두 영역:
- **강한 난류(strong turbulence)**: 변동이 평균장을 변형시킴. 혼합 길이, 에디 점성, Alfvén 시간, k-ε 모델 (Biskamp 2003; Yokoi 2006) 로 다룸.
- **약한 난류(weak turbulence)**: 변동이 평균장보다 훨씬 작고, 난류가 선형 모드 파동으로 구성되며, 진폭은 분산 관계의 보정 항으로만 나타남 (Eq. 24).

파동 접근. 평균 흐름 속의 역전파 Alfvén 파 분산 관계 (Eqs. 25, 27). V_A 가 크면 (V_A ≈ 0.5 U_0) 두 Alfvén 분기가 E(k, ω) 에서 명확히 분리 (Fig. 4 상단), V_A 가 작으면 (V_A ≈ 0.2 U_0) 확장된 Doppler-broadened 피크로 병합 (Fig. 4 하단). random MHD sweeping 스펙트럼 (Eq. 29) 과 σ^(±) 폭 (Eq. 31). Cross helicity h_c (Eq. 32) 는 전방·후방 Alfvén 파의 불균형 — 태양 근처에서 높고 (|h_c| ≈ 1), 거리에 따라 감소.

#### 2.3 Kinetic waves (pp. 13-19)

**English**: On scales ~ ion gyro-radius or inertial length (~ 100-1000 km at 1 AU), linear modes become **kinetic**: dispersive and dissipative via wave-particle interactions (Landau, cyclotron, pitch-angle scattering — see Fig. 5). Parallel-propagating whistler: ω ∝ k at very low frequencies (MHD fast-mode behavior), but ω ∝ k^2 at ion gyro-frequency and above. Phase speed v_ph = ω/k = δE/δB rises with frequency, so electric-field amplitude grows relative to magnetic.

The ion gyro-radius and inertial length are O(100-1000 km) at 1 AU; the electron counterparts are O(10-100 km). Three wave families (Fig. 6):

**2.3.1 Alfvén mode family** — Ion cyclotron mode (parallel): left-handed, resonance at ω = Ω_i; kinetic Alfvén wave (KAW, oblique): carries perpendicular electric-field at k_⊥ ρ_i > 1. Whistler is the parallel Alfvén branch at k > k_ion.

**2.3.2 Fast mode family** — Whistler for quasi-parallel; oblique extensions at higher k.

**2.3.3 Slow mode family** — Ion-acoustic for quasi-parallel; obliquely propagating kinetic slow mode.

At k = k_ion = Ω_i / V_A (ion inertial wavenumber), the MHD branches split into six kinetic branches (Fig. 6 right): whistler (W), kinetic Alfvén (KA), kinetic slow (KS), fundamental ion Bernstein (IB1), second harmonic (IB2).

**한국어**: 이온 gyro-radius 또는 관성 길이 (1 AU 에서 ~ 100-1000 km) 스케일에서 선형 모드는 **운동학적(kinetic)** 이 됨: 분산성(dispersive) 이고 소산성(dissipative); 파동-입자 상호작용 (Landau, cyclotron, pitch-angle 산란 — Fig. 5) 에 의함. 평행 전파 whistler 는 저주파에서 ω ∝ k (MHD fast-mode), 이온 gyro-frequency 이상에서 ω ∝ k^2; 위상 속도 v_ph = δE/δB 가 주파수와 함께 증가하여 전기장 진폭이 자기장 대비 커짐.

이온 gyro-radius·관성 길이 1 AU 에서 ~ 100-1000 km; 전자 대응은 ~ 10-100 km. 세 파동 가족 (Fig. 6):

**2.3.1 Alfvén 가족** — 이온 cyclotron (평행): 좌선회, ω = Ω_i 공명; 운동학적 Alfvén 파 (KAW, 경사): k_⊥ ρ_i > 1 에서 수직 전기장.

**2.3.2 Fast 가족** — quasi-parallel 에서 whistler, 높은 k 경사 확장.

**2.3.3 Slow 가족** — quasi-parallel 에서 이온 음향, 경사 전파 kinetic slow mode.

k = k_ion = Ω_i / V_A (이온 관성 파수) 에서 MHD 분기가 6 개 운동학적 분기로 쪼개짐 (Fig. 6 우측).

#### 2.4-2.6 Zero-frequency mode, sideband waves, coherent structures (pp. 19-22)

**English**:
- **Zero-frequency mode**: entropy mode / pressure-balanced structures with ω ≈ 0 in the plasma frame but moved by convection.
- **Sideband waves**: nonlinear 3-wave coupling produces additional branches offset from the main Doppler-shifted dispersion line, visible in E(k, ω) as secondary peaks.
- **Coherent structures**: current sheets, flux ropes, Alfvén vortices — localized in x-space, broadband in k-space, contributing to non-Gaussian statistics and intermittency.

**한국어**:
- **영주파수(zero-frequency) 모드**: 플라즈마 좌표계에서 ω ≈ 0 의 엔트로피 모드 / 압력 균형 구조, 대류에 의해 이동.
- **Sideband 파**: 비선형 3-파 결합이 주 Doppler 분산선에서 벗어난 추가 분기를 만들어 E(k, ω) 의 이차 피크로 나타남.
- **Coherent 구조**: current sheet, flux rope, Alfvén vortex — x-공간 국소화, k-공간 광대역; 비-Gaussian 통계와 간헐성(intermittency) 기여.

#### 2.7 Lessons from observations (pp. 22-27)

**English**: Three categories:
- **Eulerian picture** — spacecraft-frame frequency spectra; Taylor hypothesis maps ω to k_flow.
- **Catalog of dispersion diagrams** — Cluster and Themis wave telescope / k-filtering results showing MHD, ion-cyclotron, whistler, KAW, mirror-mode, and Bernstein branches.
- **Statistical dispersion diagram** — stacking many events to build a statistical E(k, ω) for solar-wind turbulence.

**한국어**: 세 범주: Eulerian 관점 (위성 좌표계 주파수 스펙트럼, Taylor 가설로 k_flow 로 매핑); 분산 다이어그램 카탈로그 (Cluster, Themis 의 wave telescope / k-filtering 결과, MHD·이온 cyclotron·whistler·KAW·mirror·Bernstein 분기); 통계적 분산 다이어그램 (다수 이벤트 스택으로 태양풍 난류의 통계적 E(k, ω) 생성).

### Part III: Wavevector Anisotropy (Section 3, pp. 27-36) / 파수벡터 이방성

#### 3.1 Impact of the large-scale magnetic field (p. 27)

**English**: The mean B_0 breaks isotropy. Parallel and perpendicular directions to B_0 become distinct; Alfvén waves propagate only along B_0, so energy transfer perpendicular is via nonlinear cascades of fluctuations, not wave propagation. This creates the characteristic k_⊥ >> k_∥ anisotropy.

**한국어**: 평균 B_0 가 등방성을 깸. B_0 에 평행·수직 방향이 구별됨; Alfvén 파는 B_0 를 따라서만 전파되므로, 수직 방향 에너지 전달은 파동 전파가 아닌 비선형 캐스케이드에 의존 → 특성 이방성 k_⊥ >> k_∥.

#### 3.2 Two-component model (p. 28)

**English**: Matthaeus et al. (1990): spectrum is the sum of **slab** (k_∥ only, wavevector parallel to B_0) and **2D** (k_⊥ only, wavevector perpendicular to B_0). Observational fits: ~ 20% slab + ~ 80% 2D in the solar wind.
$$E(\mathbf{k}) = E_{slab}(k_\parallel)\delta(k_\perp) + E_{2D}(k_\perp)\delta(k_\parallel)$$

**한국어**: Matthaeus et al. (1990) 2-성분 모델: 스펙트럼 = **slab** (k_∥ 만, 파수 벡터가 B_0 에 평행) + **2D** (k_⊥ 만, 파수 벡터가 B_0 에 수직). 태양풍 관측 적합: slab ~ 20% + 2D ~ 80%.

#### 3.3 Critical balance model (pp. 30-32)

**English**: Goldreich & Sridhar (1995): nonlinear eddy-turnover time τ_NL ~ 1/(k_⊥ δu_⊥) balances the Alfvén wave propagation time τ_A ~ 1/(k_∥ V_A). Setting τ_NL ~ τ_A:
$$k_\parallel V_A \sim k_\perp \delta u_\perp$$

Combined with Kolmogorov perpendicular cascade δu_⊥ ~ (ε / k_⊥)^(1/3):
$$k_\parallel \propto k_\perp^{2/3}$$

Eddies elongate along B_0 as the cascade proceeds to smaller scales. The perpendicular spectrum: E(k_⊥) ∝ k_⊥^(-5/3); parallel spectrum along a field line: E(k_∥) ∝ k_∥^(-2). This explains the Horbury et al. (2008) observation of slope varying from −5/3 (flow ⊥ B_0) to −2 (flow ∥ B_0) — see Fig. 2.

**한국어**: Goldreich & Sridhar (1995): 비선형 에디 회전 시간 τ_NL ~ 1/(k_⊥ δu_⊥) 과 Alfvén 파 전파 시간 τ_A ~ 1/(k_∥ V_A) 이 균형. τ_NL ~ τ_A 로부터 k_∥ V_A ~ k_⊥ δu_⊥, 수직 Kolmogorov 캐스케이드 δu_⊥ ~ (ε / k_⊥)^(1/3) 과 결합하여 **k_∥ ∝ k_⊥^(2/3)**. 작은 스케일로 갈수록 에디는 B_0 를 따라 길어짐. 수직 스펙트럼 E(k_⊥) ∝ k_⊥^(-5/3), 평행 E(k_∥) ∝ k_∥^(-2). 이는 Horbury et al. (2008) 의 흐름-필드 각도별 기울기 변화 −5/3 ↔ −2 관측을 설명 (Fig. 2).

#### 3.4 Elliptic anisotropy model (p. 32)

**English**: The energy spectrum in (k_∥, k_⊥) space is modeled as an ellipse:
$$E(k_\parallel, k_\perp) \propto (k_\parallel^2 / a^2 + k_\perp^2 / b^2)^{-\alpha/2}$$
with ratio b/a characterizing the anisotropy. In the solar wind at MHD scales, b/a ≈ 3-10 (perpendicular cascade preferred).

**한국어**: (k_∥, k_⊥) 공간에서 스펙트럼을 타원으로 모델링; 비율 b/a 가 이방성 정도를 규정. 태양풍 MHD 스케일에서 b/a ≈ 3-10 (수직 캐스케이드 선호).

#### 3.5 Non-elliptic anisotropy model (p. 33)

**English**: Observations at sub-ion scales and in magnetosheaths show contours that are not perfect ellipses. Non-elliptic extensions introduce angle-dependent spectral indices α(θ_kB). This captures the "two-slope" behavior where the ⊥ spectrum is Kolmogorov-like and the ∥ spectrum is steeper.

**한국어**: 이온 이하 스케일과 magnetosheath 관측은 완전한 타원이 아닌 등고선을 보임. 비-타원 확장은 각도 의존 스펙트럼 지수 α(θ_kB) 를 도입; 수직은 Kolmogorov 유사, 평행은 가파른 "two-slope" 행동을 포착.

#### 3.6 Asymmetries (p. 35)

**English**: Additional breaks in symmetry:
- **k_∥ → −k_∥ asymmetry** via cross-helicity (imbalanced Alfvén waves).
- **k → −k asymmetry** via magnetic helicity (circular polarization preference).
- **Reflectional asymmetry** in the perpendicular plane (e.g., anisotropy axis rotated from B_0 by a small angle).

**한국어**: 추가 대칭성 파괴: k_∥ → −k_∥ (cross helicity 에 의한 Alfvén 파 불균형); k → −k (자기 나선도에 의한 원편광 선호); 수직면 반사 비대칭 (이방성 축이 B_0 에서 약간 회전).

#### 3.7 Lessons from observations (p. 36)

**English**: Multi-spacecraft k-filtering on Cluster (baselines 100-10000 km) and MMS (baselines 10-1000 km) reveals:
- Magnetosheath: strong anisotropy b/a ≈ 3-5, quasi-2D dominant.
- Solar wind: intermediate anisotropy, mixed slab+2D.
- Foreshock: coherent wave modes (ULF, whistlers).
- MMS at 10 km tetrahedron accesses electron-scale anisotropy for the first time.

**한국어**: Cluster (기준선 100-10000 km) 와 MMS (10-1000 km) 의 다중 위성 k-filtering 결과:
- Magnetosheath: 강한 이방성 b/a ≈ 3-5, quasi-2D 우세.
- 태양풍: 중간 이방성, slab + 2D 혼합.
- Foreshock: coherent 파동 모드 (ULF, whistler).
- MMS 의 10 km 사면체는 최초로 전자 스케일 이방성에 접근.

### Part IV: Outlook (Section 4, pp. 37-41) / 전망

**English**: Open problems and future missions:
- **PSP (2018+)** and **Solar Orbiter (2020+)** will sample the Alfvénic zone near the Sun, testing whether turbulence becomes Kolmogorov or IK.
- **10-spacecraft clusters** proposed to resolve full 3D k-anisotropy simultaneously at multiple scales.
- Theoretical development of **strong MHD turbulence dynamical alignment** (Boldyrev 2006: k_⊥ ∝ k_∥^(2/3) anisotropy plus alignment angle scaling) remains active.
- Connecting turbulence cascade to **reconnection** and **coronal heating** requires the k-ω picture, not 1D slopes.

**한국어**: 열린 문제와 향후 임무:
- **PSP (2018+)** 및 **Solar Orbiter (2020+)** 는 태양 근처 Alfvén 영역을 탐사하여 난류가 Kolmogorov 인지 IK 인지 검증.
- **10-위성 클러스터** 가 제안되어 여러 스케일에서 동시에 3D k-이방성을 해상.
- **강한 MHD 난류의 동적 정렬(dynamic alignment)** 이론 (Boldyrev 2006: k_⊥ ∝ k_∥^(2/3) 이방성 + 정렬 각도 스케일링) 이 활발히 연구 중.
- 난류 캐스케이드를 **재연결** 및 **코로나 가열** 과 연결하려면 1D 기울기가 아닌 k-ω 이 그림이 필요.

---

## 3. Key Takeaways / 핵심 시사점

1. **Turbulence is fundamentally a space-time phenomenon / 난류는 본질적으로 시공간 현상이다**
   - **English**: The 1D spectrum E(k) (simulation) and E(ω) (observation) are two different projections of a single higher-dimensional spectrum E(k, ω). Equating them uncritically — via Taylor's frozen-in-flow hypothesis — is only justified when U_0 >> δU, V_A.
   - **한국어**: 1D 스펙트럼 E(k) (시뮬레이션) 과 E(ω) (관측) 은 단일 고차원 스펙트럼 E(k, ω) 의 서로 다른 두 투영이다. Taylor 가설로 이 둘을 아무 조건 없이 동일시하는 것은 U_0 >> δU, V_A 일 때만 정당화된다.

2. **Doppler shift and broadening determine the ω-k mapping / Doppler 천이와 넓힘이 ω-k 매핑을 결정한다**
   - **English**: The random-sweeping spectrum is a Gaussian centered at ω = k·U_0 with width σ = k δU. In the solar wind (U_0 ≈ 400 km/s, δU ≈ 30-80 km/s), σ/ω ≈ δU/U_0 ≈ 0.1, so Taylor holds reasonably well at MHD scales but can break at kinetic scales where V_A / U_0 → 1.
   - **한국어**: random-sweeping 스펙트럼은 ω = k·U_0 를 중심으로 σ = k δU 폭의 Gaussian. 태양풍(U_0 ≈ 400 km/s, δU ≈ 30-80 km/s) 에서 σ/ω ≈ 0.1 이므로 MHD 스케일에서 Taylor 가 합리적으로 유효하나, V_A / U_0 → 1 인 운동학 스케일에서는 깨질 수 있다.

3. **Wavevector anisotropy is ubiquitous and unavoidable / 파수 벡터 이방성은 보편적이고 피할 수 없다**
   - **English**: The mean field B_0 breaks isotropy. The critical-balance scaling k_∥ ∝ k_⊥^(2/3) (Goldreich-Sridhar 1995) is the theoretical backbone; single-spacecraft observations of spectral index varying with flow-field angle from −5/3 (⊥) to −2 (∥) (Horbury et al. 2008) directly confirm it.
   - **한국어**: 평균장 B_0 가 등방성을 깬다. 임계 균형 스케일링 k_∥ ∝ k_⊥^(2/3) (Goldreich-Sridhar 1995) 이 이론적 중추이며, 흐름-장 각도에 따라 −5/3 (⊥) 에서 −2 (∥) 로 변하는 단일 위성 스펙트럼 지수 관측 (Horbury et al. 2008) 이 이를 직접 확인한다.

4. **Multi-spacecraft k-filtering is the gold standard for 3D wavevector spectra / 다중 위성 k-필터링이 3D 파수 벡터 스펙트럼의 표준이다**
   - **English**: With N ≥ 4 spacecraft in a non-coplanar tetrahedron, one can Fourier-transform in both t and x simultaneously. The accessible k-range is set by the inter-spacecraft distance: Cluster (100-10000 km) covers MHD-to-ion scales; MMS (10-1000 km) reaches ion-to-electron scales.
   - **한국어**: 비-동일평면 사면체에 배치된 N ≥ 4 위성으로 시간과 공간을 동시에 Fourier 변환할 수 있다. 접근 가능한 k 범위는 위성 간 거리로 결정됨: Cluster (100-10000 km) 는 MHD-이온 스케일, MMS (10-1000 km) 는 이온-전자 스케일.

5. **Spectral slope -5/3 vs -3/2 reflects underlying closure / 기울기 -5/3 대 -3/2 는 기저의 폐쇄(closure) 를 반영한다**
   - **English**: Kolmogorov k^(-5/3) follows dimensional analysis with energy flux alone. Iroshnikov-Kraichnan k^(-3/2) assumes Alfvén-wave weak-interaction closure. Modern high-resolution solar-wind data often show slopes near −5/3 (Kolmogorov-like), but transitional regions and highly Alfvénic streams can approach −3/2 (Boldyrev with dynamic alignment).
   - **한국어**: Kolmogorov k^(-5/3) 는 에너지 flux 만의 차원 분석. Iroshnikov-Kraichnan k^(-3/2) 는 Alfvén 파 약한 상호작용 폐쇄 가정. 현대 고해상도 태양풍 데이터는 −5/3 (Kolmogorov 유사) 에 가까우나, 전이 영역과 고-Alfvénic 흐름에서 −3/2 (동적 정렬을 포함한 Boldyrev) 에 근접할 수 있다.

6. **Kinetic waves populate six branches below k_ion / k_ion 이하에서 운동학 파동은 6 개 분기를 가진다**
   - **English**: Below k_ion = Ω_i/V_A, MHD has 3 modes (A, F, S) × 2 propagation directions = 6 branches. Above k_ion, modes bifurcate into kinetic Alfvén, kinetic slow, whistler, ion-cyclotron, ion Bernstein 1st and 2nd harmonic, etc. Each branch has its own dispersion curvature and polarization, providing a fingerprint in E(k, ω).
   - **한국어**: k_ion = Ω_i/V_A 이하에서 MHD 는 3 개 모드 (A, F, S) × 2 방향 = 6 분기. k_ion 이상에서 kinetic Alfvén, kinetic slow, whistler, 이온-cyclotron, ion Bernstein (1·2차 조화) 등으로 분기. 각 분기는 고유한 분산 곡률과 편극을 가져 E(k, ω) 의 지문이 된다.

7. **Cross-helicity imbalance distinguishes "near-Sun" from "mature" turbulence / Cross helicity 불균형이 "태양 근방" 과 "성숙" 난류를 구별한다**
   - **English**: Near the Sun (< 0.3 AU), |h_c| ≈ 1: outward-propagating Alfvén waves dominate (pristine solar outflow). At 1 AU, |h_c| ≈ 0.1-0.3: counter-propagating waves are needed for the cascade, so turbulence has "matured" via reflection/parametric decay/mixing.
   - **한국어**: 태양 근방 (< 0.3 AU) 에서 |h_c| ≈ 1: 외향 Alfvén 파 우세 (원시 태양 유출). 1 AU 에서 |h_c| ≈ 0.1-0.3: 캐스케이드를 위해 역전파 파동 필요하므로, 반사·매개변수 붕괴·혼합을 통해 난류가 "성숙".

8. **The k-ω picture is the bridge between simulation and observation / k-ω 그림이 시뮬레이션과 관측의 다리이다**
   - **English**: Numerical MHD/PIC simulations output full E(k, ω) and can be projected onto E(ω) by observer-motion sampling. Single-spacecraft observations give E(ω) and, under Taylor, E(k_flow). Multi-spacecraft give E(k). The k-ω diagram is the common ground where all methods meet and where linear-mode dispersion relations provide a sanity check on both sides.
   - **한국어**: MHD/PIC 수치 시뮬레이션은 전체 E(k, ω) 를 출력하며, 관측자 운동으로 샘플링하여 E(ω) 로 투영할 수 있다. 단일 위성 관측은 E(ω) 를, Taylor 하에서 E(k_flow) 를 준다. 다중 위성은 E(k). k-ω 다이어그램이 모든 방법의 공통 지반이며, 선형 모드 분산 관계가 양쪽 모두의 타당성 점검 기준이 된다.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Kolmogorov inertial-range spectrum / Kolmogorov 관성 영역 스펙트럼

$$E^{(1D)}(k) = C_K \epsilon^{2/3} k^{-5/3}$$

**English**: C_K ≈ 1.6 is the Kolmogorov constant, ε the energy dissipation rate (L^2 T^(-3)), k the wavenumber (L^(-1)). Dimensional analysis: [E] = L^3 T^(-2); [ε^(2/3) k^(-5/3)] = (L^(4/3) T^(-2))(L^(5/3)) = L^3 T^(-2). ✓

**한국어**: C_K ≈ 1.6 는 Kolmogorov 상수, ε 는 에너지 소산률 (L^2 T^(-3)), k 는 파수 (L^(-1)). 차원 분석 검증: [E] = L^3 T^(-2) 와 [ε^(2/3) k^(-5/3)] = L^3 T^(-2) 일치. ✓

### 4.2 Taylor's frozen-in-flow hypothesis / Taylor 가설

$$\omega = \mathbf{k}\cdot\mathbf{U}_0$$

**English**: Valid when the convection speed U_0 dominates over intrinsic wave propagation (V_A) and over large-scale flow variation (δU). In that case the 1D frequency spectrum equals the streamwise 1D wavenumber spectrum up to a Jacobian factor U_0.

**한국어**: 대류 속도 U_0 가 내재 파동 속도 V_A 와 대규모 흐름 변동 δU 를 압도할 때 유효. 이 경우 1D 주파수 스펙트럼은 흐름 방향 1D 파수 스펙트럼과 Jacobian 인자 U_0 를 제외하고 동일.

### 4.3 Random-sweeping 2D spectrum / 랜덤 스위핑 2D 스펙트럼

$$E(\mathbf{k}, \omega) = \frac{E(\mathbf{k})}{\sqrt{2\pi k^2 (\delta U)^2}} \exp\left[-\frac{(\omega - \mathbf{k}\cdot\mathbf{U}_0)^2}{2 k^2 (\delta U)^2}\right]$$

**English**: Gaussian in ω at fixed k, centered at the Doppler shift k·U_0, width σ = k δU. As δU → 0, recovers a Dirac delta — Taylor's hypothesis. The Eulerian 1D frequency spectrum obtained by integration over k:
$$E(\omega) = \int d\mathbf{k}\, E(\mathbf{k}, \omega)$$

**한국어**: 고정 k 에서 ω 에 대한 Gaussian, Doppler shift k·U_0 중심, 폭 σ = k δU. δU → 0 에서 Dirac delta 복원 → Taylor 가설. k 적분으로 Eulerian 1D 주파수 스펙트럼을 얻음.

### 4.4 Doppler-shifted Alfvén dispersion / Doppler 천이된 Alfvén 분산

$$\omega_\pm = \mathbf{k}\cdot\mathbf{U}_0 \pm \mathbf{k}\cdot\mathbf{V}_A$$

**English**: The plus sign is forward-propagating Alfvén (relative to B_0), minus is backward. In the slow-flow limit V_A > U_0, two distinct dispersion lines are observable in E(k, ω); otherwise they merge into a Doppler-broadened peak.

**한국어**: 더하기 부호는 전방 전파 Alfvén (B_0 에 대해), 빼기는 후방. 느린 흐름 극한 V_A > U_0 에서 E(k, ω) 에 두 분산선이 구별되어 관찰; 아닐 경우 Doppler-broadened 피크로 병합.

### 4.5 Goldreich-Sridhar critical balance / Goldreich-Sridhar 임계 균형

$$\tau_{NL} \sim \tau_A \Rightarrow \frac{1}{k_\perp \delta u_\perp} \sim \frac{1}{k_\parallel V_A}$$
$$\delta u_\perp \sim (\epsilon/k_\perp)^{1/3} \Rightarrow k_\parallel \propto k_\perp^{2/3}$$

**English**: The anisotropy is scale-dependent: eddies elongate along B_0 as one goes to smaller scales. Perpendicular spectrum E(k_⊥) ∝ k_⊥^(-5/3); parallel spectrum E(k_∥) ∝ k_∥^(-2).

**한국어**: 이방성이 스케일 의존적: 작은 스케일로 갈수록 에디가 B_0 방향으로 길어짐. 수직 스펙트럼 E(k_⊥) ∝ k_⊥^(-5/3); 평행 E(k_∥) ∝ k_∥^(-2).

### 4.6 Iroshnikov-Kraichnan MHD spectrum / IK MHD 스펙트럼

$$E(k) = C_{IK} (\epsilon V_A)^{1/2} k^{-3/2}$$

**English**: Alternative closure with Alfvén-wave weak interaction. Extra V_A factor reflects slower cascade rate τ_cas ~ τ_NL^2 / τ_A.

**한국어**: Alfvén 파 약한 상호작용 폐쇄의 대안. 추가 V_A 인자는 더 느린 캐스케이드 속도 τ_cas ~ τ_NL^2 / τ_A 를 반영.

### 4.7 Structure function / 구조 함수

$$S_p(\mathbf{r}) = \langle |\delta\mathbf{u}(\mathbf{x} + \mathbf{r}) - \delta\mathbf{u}(\mathbf{x})|^p \rangle$$

**English**: For Kolmogorov K41, S_p(r) ∝ r^(p/3) in the inertial range. For MHD with critical balance, the exponent depends on whether r is parallel or perpendicular to B_0. Intermittency corrections (She-Leveque 1994) modify the exponents.

**한국어**: Kolmogorov K41 에서 관성 영역 S_p(r) ∝ r^(p/3). 임계 균형을 갖는 MHD 에서는 r 이 B_0 에 평행인지 수직인지에 따라 지수가 달라짐. 간헐성 보정 (She-Leveque 1994) 이 지수를 수정.

### 4.8 Multi-spacecraft k-filtering / 다중 위성 k-필터링

**English**: For N spacecraft at positions {x_α} and signals {B_α(t)}, the wave-telescope (Capon) estimator is:
$$P(\mathbf{k}, \omega) = \frac{1}{\mathbf{H}^\dagger(\mathbf{k}) \mathbf{M}^{-1}(\omega) \mathbf{H}(\mathbf{k})}$$
where H(k) is the array steering vector H_α(k) = exp(−i k·x_α) and M(ω) is the cross-spectral-density matrix between spacecraft. The peak of P(k, ω) at fixed ω identifies the wavevector of that frequency component. Requires N ≥ 4 in 3D, non-coplanar configuration.

**한국어**: N 개 위성 위치 {x_α} 와 신호 {B_α(t)} 에 대해 wave-telescope (Capon) 추정기: P(k, ω) = 1 / (H^† M^(-1) H). 여기서 H(k) 는 어레이 조향 벡터 H_α = exp(−i k·x_α), M(ω) 는 위성 간 교차 스펙트럼 밀도 행렬. 고정 ω 에서 P(k, ω) 의 피크가 해당 주파수 성분의 파수 벡터를 식별. 3D 에서 비-동일평면 N ≥ 4 필요.

### 4.9 Worked example: Taylor validity in MMS magnetosheath / 수치 예제: MMS magnetosheath 에서 Taylor 유효성

**English**: MMS in the magnetosheath: U_0 ≈ 200 km/s, δU ≈ 100 km/s, V_A ≈ 80 km/s. Inter-spacecraft distance 10 km → maximum wavenumber k_max ≈ 2π/10 ≈ 0.63 km^(-1). Taylor validity: I = erf(Δτ) with Δτ ≈ (U_0/δU)(Δω/ω)/√2 = (2)(0.1)/1.41 ≈ 0.14 → I = erf(0.14) ≈ 0.15. **Taylor's hypothesis is NOT well satisfied in the magnetosheath**, so k-filtering must be used instead of simple ω = k·U_0 mapping.

In the solar wind at 1 AU: U_0 ≈ 400 km/s, δU ≈ 30 km/s, V_A ≈ 50 km/s. Δτ ≈ (13.3)(0.1)/1.41 ≈ 0.94 → I ≈ 0.82. Taylor is reasonably valid for MHD scales, but fails at kinetic scales where V_A becomes comparable to U_0 in the parallel direction.

**한국어**: MMS magnetosheath 에서 U_0 ≈ 200 km/s, δU ≈ 100 km/s, V_A ≈ 80 km/s. 위성 간 거리 10 km → 최대 파수 k_max ≈ 2π/10 ≈ 0.63 km^(-1). Taylor 유효성 I = erf(Δτ) 로, Δτ ≈ 0.14 → I ≈ 0.15. **Magnetosheath 에서 Taylor 가설은 잘 성립하지 않음** — 단순 ω = k·U_0 매핑 대신 k-filtering 이 필요.

1 AU 태양풍: U_0 ≈ 400 km/s, δU ≈ 30 km/s, V_A ≈ 50 km/s. Δτ ≈ 0.94 → I ≈ 0.82. MHD 스케일에서 Taylor 합리적, 그러나 V_A 가 U_0 에 근접하는 운동학 스케일 평행 방향에서는 실패.

### 4.10 MMS tetrahedron geometry / MMS 사면체 기하

**English**: Four spacecraft at vertices of a quasi-regular tetrahedron, inter-spacecraft distance d. Accessible wavenumber range: k_min ≈ 2π/L_sys, k_max ≈ π/d (Nyquist-like). For MMS with d = 10 km, k_max ≈ 0.31 km^(-1), suitable for electron scales (k_e ≈ 0.1 km^(-1)). For d = 1000 km, k_max ≈ 3.1 × 10^(-3) km^(-1), covering ion scales (k_i ≈ 0.01 km^(-1) at 1 AU).

Anisotropy ratio at magnetosheath (MMS observations): b/a ≈ 3-5 (perpendicular wavenumber preferred by factor 3-5 over parallel). Solar wind at 1 AU (Cluster): b/a ≈ 2-4, less extreme because the anisotropy has "relaxed" via mixing.

**한국어**: quasi-regular 사면체의 4 개 위성, 위성 간 거리 d. 접근 가능 파수 범위: k_min ≈ 2π/L_sys, k_max ≈ π/d (Nyquist 유사). MMS d = 10 km 시 k_max ≈ 0.31 km^(-1), 전자 스케일 (k_e ≈ 0.1 km^(-1)) 에 적합. d = 1000 km 시 k_max ≈ 3.1 × 10^(-3) km^(-1), 이온 스케일 (k_i ≈ 0.01 km^(-1) at 1 AU) 를 포괄.

Magnetosheath 이방성 비 (MMS 관측): b/a ≈ 3-5. 1 AU 태양풍 (Cluster): b/a ≈ 2-4, 혼합에 의해 "이완"되어 덜 극단적.

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1915 ─── G.I. Taylor: turbulence statistical theory / Taylor 난류 통계 이론
           │
1926 ─── Richardson: energy cascade picture / 에너지 캐스케이드 개념
           │
1938 ─── Taylor's frozen-in-flow hypothesis / Taylor 가설
           │
1941 ─── Kolmogorov K41: E(k) ∝ k^(-5/3) / Kolmogorov 스펙트럼
           │
1964 ─── Kraichnan: random sweeping model / 랜덤 스위핑 모델
           │
1965 ─── Iroshnikov-Kraichnan MHD: E(k) ∝ k^(-3/2) / IK MHD 스펙트럼
           │
1968 ─── Coleman: Mariner 2 solar-wind turbulence / 태양풍 난류 관측
           │
1975 ─── Tennekes: Lagrangian E(ω) ∝ ω^(-2) / Lagrangian 스펙트럼
           │
1983 ─── Shebalin-Matthaeus-Montgomery: 2D anisotropy in MHD / MHD 2D 이방성
           │
1990 ─── Matthaeus et al.: slab + 2D two-component model / slab+2D 2-성분 모델
           │
1995 ─── Goldreich-Sridhar: critical balance k_∥ ∝ k_⊥^(2/3) / 임계 균형
           │
2000 ─── Cluster mission launch: 4-spacecraft k-filtering / Cluster 발사
           │
2006 ─── Boldyrev: dynamic alignment E(k_⊥) ∝ k_⊥^(-3/2) / 동적 정렬
           │
2008 ─── Horbury et al.: angle-dependent spectral index / 각도별 지수
           │
2015 ─── MMS mission launch: 10 km tetrahedron / MMS 발사
           │
2018 ─── Narita review (this paper): k-ω + anisotropy synthesis ★
           │
2018 ─── Parker Solar Probe launch / PSP 발사
           │
2020 ─── Solar Orbiter launch / Solar Orbiter 발사
           │
Future ─ 10-spacecraft clusters for full 3D k-anisotropy / 10-위성 클러스터
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **Paper #32 (Tsurutani et al., Living Reviews series — magnetic decreases / interplanetary discontinuities)** | Discontinuities are coherent structures that contribute to the non-Gaussian tail of turbulence and to the k-space broadband content discussed in Section 2.6. / 불연속성은 coherent 구조로서 난류의 비-Gaussian 꼬리와 2.6 절의 k-공간 광대역 기여에 해당. | High / 높음 — Narita's "coherent structures" directly overlap with magnetic decreases/discontinuities. / Narita 의 "coherent 구조" 가 자기 감소/불연속성과 직접 중첩. |
| **Matthaeus et al. (1990) — two-component slab + 2D model** | Foundation of Section 3.2. Observationally constrained ~20% slab + ~80% 2D in the solar wind. / 3.2 절의 근간. 태양풍 관측 제약 ~20% slab + ~80% 2D. | High / 높음 — the first-principle decomposition of wavevector anisotropy. / 파수 벡터 이방성의 1차 원리 분해. |
| **Goldreich & Sridhar (1995) — critical-balance theory** | Section 3.3 backbone. Provides k_∥ ∝ k_⊥^(2/3) scaling and explains the observed angle-dependent spectral index. / 3.3 절의 중추. k_∥ ∝ k_⊥^(2/3) 를 제공하고 관측된 각도별 지수를 설명. | Critical / 결정적 — modern theoretical standard for MHD turbulence anisotropy. / MHD 난류 이방성의 현대 이론 표준. |
| **Horbury et al. (2008) — wavelet angle-dependent spectra** | Observationally confirms critical balance by showing slope varies from −5/3 (⊥) to −2 (∥). Fig. 2 of this review. / 흐름-장 각도별 기울기 −5/3 ↔ −2 변화 관측으로 임계 균형을 확인 (Fig. 2). | Critical / 결정적 — the key single-spacecraft evidence for anisotropy. / 단일 위성에서 이방성의 핵심 증거. |
| **Coleman (1968) — Mariner 2 solar-wind turbulence** | Historical origin of solar-wind turbulence studies. Index near −5/3 established the Kolmogorov paradigm at 1 AU. / 태양풍 난류 연구의 역사적 기원. 1 AU 에서 −5/3 지수로 Kolmogorov 패러다임 확립. | High / 높음 — baseline observation all subsequent work builds on. / 이후 모든 연구의 기준 관측. |
| **Boldyrev (2006) — dynamic alignment** | Alternative to Goldreich-Sridhar predicting E(k_⊥) ∝ k_⊥^(-3/2) via alignment of δu and δB. / δu, δB 정렬을 통한 E(k_⊥) ∝ k_⊥^(-3/2) 를 예측하는 Goldreich-Sridhar 의 대안. | High / 높음 — explains observed −3/2 slopes in highly Alfvénic solar-wind streams. / 고-Alfvénic 태양풍 흐름에서 관측되는 −3/2 를 설명. |

---

## 7. References / 참고문헌

- Narita, Y. (2018). Space-time structure and wavevector anisotropy in space plasma turbulence. *Living Reviews in Solar Physics*, 15, 2. DOI: 10.1007/s41116-017-0010-0
- Kolmogorov, A.N. (1941). The local structure of turbulence in incompressible viscous fluid for very large Reynolds numbers. *Dokl. Akad. Nauk SSSR*, 30, 301.
- Taylor, G.I. (1938). The spectrum of turbulence. *Proc. Roy. Soc. London A*, 164, 476.
- Kraichnan, R.H. (1964). Kolmogorov's hypotheses and Eulerian turbulence theory. *Phys. Fluids*, 7, 1723.
- Iroshnikov, P.S. (1963). Turbulence of a conducting fluid in a strong magnetic field. *Astron. Zh.*, 40, 742.
- Coleman, P.J. (1968). Turbulence, viscosity, and dissipation in the solar-wind plasma. *ApJ*, 153, 371.
- Tennekes, H. (1975). Eulerian and Lagrangian time microscales in isotropic turbulence. *J. Fluid Mech.*, 67, 561.
- Shebalin, J.V., Matthaeus, W.H., Montgomery, D. (1983). Anisotropy in MHD turbulence due to a mean magnetic field. *J. Plasma Phys.*, 29, 525.
- Matthaeus, W.H., Goldstein, M.L., Roberts, D.A. (1990). Evidence for the presence of quasi-two-dimensional nearly incompressible fluctuations in the solar wind. *JGR*, 95, 20673.
- Goldreich, P., Sridhar, S. (1995). Toward a theory of interstellar turbulence II. Strong Alfvénic turbulence. *ApJ*, 438, 763.
- Boldyrev, S. (2006). Spectrum of magnetohydrodynamic turbulence. *PRL*, 96, 115002.
- Horbury, T.S., Forman, M., Oughton, S. (2008). Anisotropic scaling of magnetohydrodynamic turbulence. *PRL*, 101, 175005.
- Osman, K.T., Horbury, T.S. (2009). Multi-spacecraft measurement of anisotropic correlation functions in solar wind turbulence. *ApJ*, 693, L175.
- Alexandrova, O. et al. (2009). Universality of solar-wind turbulent spectrum. *PRL*, 103, 165003.
- Sahraoui, F. et al. (2009). Evidence of a cascade and dissipation of solar-wind turbulence. *PRL*, 102, 231102.
- Carbone, V., Veltri, P., Mangeney, A. (1995). Coherent structure formation and magnetic field line reconnection in magnetohydrodynamic turbulence. *Phys. Fluids A*, 7, 3287.
- Narita, Y. (2017a). Error estimate of Taylor's frozen-in-flow hypothesis. *Ann. Geophys.*, 35, 325.
- Biskamp, D. (2003). *Magnetohydrodynamic Turbulence*. Cambridge University Press.
- Bruno, R., Carbone, V. (2013). The solar wind as a turbulence laboratory. *LRSP*, 10, 2.
- Tu, C.-Y., Marsch, E. (1995). MHD structures, waves, and turbulence in the solar wind. *Space Sci. Rev.*, 73, 1.
