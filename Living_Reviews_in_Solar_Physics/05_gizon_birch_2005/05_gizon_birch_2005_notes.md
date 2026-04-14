---
title: "Local Helioseismology"
authors:
  - Laurent Gizon
  - Aaron C. Birch
year: 2005
journal: "Living Reviews in Solar Physics, 2, 6"
topic: "Living Reviews in Solar Physics / Local Helioseismology"
tags:
  - helioseismology
  - local helioseismology
  - time-distance
  - ring-diagram
  - holography
  - solar oscillations
  - p-modes
  - f-modes
  - inverse problem
  - Born approximation
  - sensitivity kernels
  - supergranulation
  - differential rotation
  - meridional flow
status: completed
date_started: 2026-04-09
date_completed: 2026-04-09
---

# Local Helioseismology
## 국소 태양진동학

---

## 핵심 기여 / Core Contribution

이 논문은 국소 태양진동학(local helioseismology)의 이론적 기반과 관측 기법, 그리고 과학적 성과를 130페이지 이상에 걸쳐 포괄적으로 정리한 리뷰 논문이다. 전통적인 전구 태양진동학(global helioseismology)이 태양 고유진동수(eigenfrequencies)만을 해석하는 데 반해, 국소 태양진동학은 **전체 파동장(full wavefield)**을 해석하여 태양 내부의 3차원 구조와 흐름을 복원한다. 이 리뷰는 다섯 가지 주요 기법 — Fourier-Hankel 분광법, ring-diagram 분석, time-distance 태양진동학, 태양진동 holography, 직접 모델링 — 의 이론적 기초를 통일된 수학적 틀 안에서 제시하고, 각 기법의 감도 커널(sensitivity kernel), 역문제(inverse problem) 풀이 방법, 그리고 잡음 특성을 체계적으로 비교한다. 과학적 성과로는 차등 회전(differential rotation), 자오면 흐름(meridional flow), 초과립(supergranulation)의 깊이 구조, 흑점(sunspot) 아래의 3차원 유동 구조, 그리고 태양 뒷면 영상(far-side imaging)까지 포괄한다. 특히 Born 근사(Born approximation)에 기반한 감도 커널의 체계적 도출이 이 리뷰의 이론적 핵심이며, 이는 이전의 광선 근사(ray approximation)를 넘어서는 중요한 진전이다.

This paper is a comprehensive 130+ page review that systematically organizes the theoretical foundations, observational techniques, and scientific results of local helioseismology. While traditional global helioseismology interprets only solar eigenfrequencies, local helioseismology interprets the **full wavefield** to reconstruct three-dimensional structures and flows inside the Sun. The review presents the theoretical basis of five major techniques — Fourier-Hankel spectral method, ring-diagram analysis, time-distance helioseismology, helioseismic holography, and direct modeling — within a unified mathematical framework, systematically comparing the sensitivity kernels, inverse-problem approaches, and noise properties of each method. Scientific results span differential rotation, meridional flow, depth structure of supergranulation, 3D flow structures beneath sunspots, and far-side imaging. The systematic derivation of sensitivity kernels based on the Born approximation is the theoretical centerpiece of this review, representing a significant advance beyond the earlier ray approximation.

---

## 읽기 노트 / Reading Notes

### 1. 도입과 개관 (§1) / Introduction and Outline (§1)

"국소 태양진동학(local helioseismology)"이라는 용어는 Lindsey et al. (1993)에 의해 처음 사용되었다. 이 분야의 핵심 아이디어는 태양 표면에서 관측되는 파동장이 태양 내부의 물리적 성질(음속, 밀도, 흐름 속도, 자기장 등)에 의해 변형되므로, 파동장을 분석하면 내부 구조를 역으로 추론할 수 있다는 것이다.

The term "local helioseismology" was first used by Lindsey et al. (1993). The core idea of this field is that the wavefield observed at the solar surface is modified by the physical properties of the solar interior (sound speed, density, flow velocity, magnetic field, etc.), so analyzing the wavefield allows inverse inference of internal structure.

전구 태양진동학과 국소 태양진동학의 근본적 차이는 다음과 같다:

The fundamental difference between global and local helioseismology is as follows:

| 비교 항목 / Comparison | 전구 태양진동학 / Global | 국소 태양진동학 / Local |
|---|---|---|
| 해석 대상 / Interpreted quantity | 고유진동수 $\omega_{nlm}$ / Eigenfrequencies | 전체 파동장 $\Phi(\mathbf{x},t)$ / Full wavefield |
| 공간 분해능 / Spatial resolution | 구면 조화 함수로 제한 / Limited by spherical harmonics | 지역적 패치 분석 가능 / Can analyze local patches |
| 차원 / Dimensionality | 1D (반경 방향) 또는 2D / 1D (radial) or 2D | 3D 복원 가능 / 3D reconstruction possible |
| 시간 분해능 / Temporal resolution | ~수개월 (긴 시계열 필요) / ~months (long time series needed) | ~수일 (짧은 시계열로도 가능) / ~days (short time series possible) |
| 적용 범위 / Applicability | 구 대칭 구조 / Spherically symmetric structure | 비구면, 국소 구조 / Non-spherical, local structures |

국소 태양진동학이 전구 태양진동학을 대체하는 것이 아니라 **보완**하는 관계임을 강조할 필요가 있다. 전구 태양진동학이 태양 내부의 평균적 구조(sound speed profile, rotation rate의 위도-깊이 의존성 등)를 정밀하게 결정하는 데 탁월한 반면, 국소 태양진동학은 흑점, 초과립, 국소 흐름 등 공간적으로 비균일한 구조를 탐사하는 데 강점을 지닌다.

It is important to emphasize that local helioseismology does not replace global helioseismology but **complements** it. While global helioseismology excels at precisely determining the average internal structure (sound speed profile, latitude-depth dependence of rotation rate, etc.), local helioseismology is strong at probing spatially inhomogeneous structures such as sunspots, supergranulation, and local flows.

---

### 2. 태양 진동의 관측 (§2) / Observations of Solar Oscillations (§2)

#### 2.1 관측 장비와 데이터 / Instruments and Data

이 분야의 발전은 고품질 관측 데이터의 가용성에 크게 의존해 왔다. 주요 관측 장비는 다음과 같다:

The development of this field has been strongly dependent on the availability of high-quality observational data. The major instruments are:

- **TON (Taiwan Oscillation Network)**: Ca II K 선 관측. 1분 간격(cadence). 주로 intensity 관측에 사용.
  / Ca II K line observations. 1-minute cadence. Primarily used for intensity observations.

- **GONG (Global Oscillation Network Group)**: 6개 지상 관측소로 구성된 네트워크. Ni I 6768 Å 흡수선 사용. 1분 간격. 약 90% duty cycle 달성. 네트워크 구성 덕분에 거의 연속적인 관측이 가능하다.
  / Network of 6 ground-based stations. Uses Ni I 6768 Å absorption line. 1-minute cadence. Achieves approximately 90% duty cycle. Network configuration enables nearly continuous observation.

- **MDI/SOHO (Michelson Doppler Imager on Solar and Heliospheric Observatory)**: Ni I 6768 Å 흡수선 사용. 1분 간격. 1024×1024 CCD. 우주에서 관측하므로 대기 효과가 없고 100% duty cycle에 가깝다. 그러나 telemetry 제한으로 전체 디스크 데이터는 간헐적으로만 전송 가능하다.
  / Ni I 6768 Å absorption line. 1-minute cadence. 1024×1024 CCD. Space-based observation eliminates atmospheric effects and achieves near-100% duty cycle. However, telemetry limitations restrict full-disk data transmission to intermittent periods.

관측 가능한 물리량(observable)은 주로 **시선 방향 도플러 속도(line-of-sight Doppler velocity)** $\Phi(\mathbf{x},t) = \mathcal{F}\{\hat{l} \cdot \mathbf{v}(\mathbf{x}, z_{\text{obs}}, t)\}$ 이다. 여기서 $\hat{l}$은 시선 방향 단위 벡터, $z_{\text{obs}}$는 관측 높이(광구 부근), $\mathcal{F}$는 관측 장비의 필터 함수이다. Intensity 관측도 사용되지만 도플러 관측이 신호 대 잡음비가 더 우수하다.

The primary physical observable is the **line-of-sight Doppler velocity** $\Phi(\mathbf{x},t) = \mathcal{F}\{\hat{l} \cdot \mathbf{v}(\mathbf{x}, z_{\text{obs}}, t)\}$, where $\hat{l}$ is the line-of-sight unit vector, $z_{\text{obs}}$ is the observation height (near the photosphere), and $\mathcal{F}$ is the instrument filter function. Intensity observations are also used but Doppler observations offer better signal-to-noise ratio.

#### 2.2 태양 진동의 기본 성질 / Fundamental Properties of Solar Oscillations

태양의 진동은 구면 조화 함수(spherical harmonics)로 분해되는 고유 모드(eigenmodes)로 기술된다. 변위(displacement)는 다음과 같이 전개된다:

Solar oscillations are described as eigenmodes that decompose into spherical harmonics. The displacement is expanded as:

$$\delta \mathbf{r} = \sum_{n,l,m} a_{nlm} \, \boldsymbol{\xi}_{nl}(r) \, Y_l^m(\theta, \phi) \, e^{i\omega_{nlm} t}$$

여기서 $n$은 radial order(반경 차수), $l$은 angular degree(각도 차수), $m$은 azimuthal order(방위 차수), $\boldsymbol{\xi}_{nl}(r)$은 radial eigenfunction, $Y_l^m$은 구면 조화 함수, $\omega_{nlm}$은 고유 진동수이다.

Here $n$ is the radial order, $l$ is the angular degree, $m$ is the azimuthal order, $\boldsymbol{\xi}_{nl}(r)$ is the radial eigenfunction, $Y_l^m$ is the spherical harmonic, and $\omega_{nlm}$ is the eigenfrequency.

주요 모드 유형은 다음과 같다:

The main mode types are:

- **f-mode ($n=0$)**: 표면 중력파(surface gravity wave). 분산 관계: $\omega^2 = gk$, 여기서 $g$는 표면 중력 가속도, $k$는 수평 파수. 유체 역학에서 깊은 물의 표면파와 물리적으로 동등하다. 복원력은 중력이며, 파동은 표면 근처에 국한된다.
  / Surface gravity wave. Dispersion relation: $\omega^2 = gk$, where $g$ is surface gravitational acceleration and $k$ is horizontal wavenumber. Physically equivalent to deep-water surface waves in fluid mechanics. The restoring force is gravity, and the wave is confined near the surface.

- **p-mode ($n \geq 1$)**: 음파(acoustic wave). 복원력은 압력(pressure). 낮은 $l$과 높은 $n$을 가진 모드는 더 깊이 침투한다. 이는 음속이 깊이에 따라 증가하므로 파동이 내부에서 굴절되어 되돌아오기 때문이다. 구체적으로, 모드가 표면 아래로 침투할 수 있는 하한점(lower turning point)은 $r_t$이며, 이 지점에서 수평 위상 속도가 국소 음속과 같아진다: $c(r_t)/r_t = \omega/\sqrt{l(l+1)}$.
  / Acoustic wave. Restoring force is pressure. Modes with low $l$ and high $n$ penetrate deeper. This occurs because sound speed increases with depth, causing waves to refract and turn back inside. Specifically, the lower turning point $r_t$ where a mode can penetrate below the surface is where the horizontal phase speed equals the local sound speed: $c(r_t)/r_t = \omega/\sqrt{l(l+1)}$.

- **Acoustic cutoff frequency**: 약 5.3 mHz. 이 진동수 이상에서는 파동이 태양 내부에 포획(trap)되지 않고 대기로 빠져나간다. 따라서 이 진동수 이상의 파동은 뚜렷한 고유 모드를 형성하지 않는다.
  / Approximately 5.3 mHz. Above this frequency, waves are not trapped inside the Sun and escape into the atmosphere. Therefore, waves above this frequency do not form well-defined eigenmodes.

#### 2.3 파워 스펙트럼의 구조 / Structure of the Power Spectrum

$l$-$\nu$ 도표(angular degree vs. frequency diagram)에서 태양 진동은 뚜렷한 능선(ridge) 구조를 보인다. 각 능선은 특정 radial order $n$에 대응한다. f-mode 능선은 가장 낮은 진동수에 위치하며, 그 위로 $n=1, 2, 3, \ldots$의 p-mode 능선들이 순서대로 나타난다.

In the $l$-$\nu$ diagram (angular degree vs. frequency), solar oscillations show clear ridge structures. Each ridge corresponds to a specific radial order $n$. The f-mode ridge is at the lowest frequency, with $n=1, 2, 3, \ldots$ p-mode ridges appearing in order above it.

$l > 150$ 영역에서는 파동 감쇠(wave damping)가 심해져 능선이 점차 연속적인 구조로 변한다. 이는 고차 모드가 대류에 의한 산란과 흡수로 인해 수명이 짧아지기 때문이다. 이 영역에서는 모드들이 개별적으로 분해되지 않아, 전구 태양진동학의 모드별 분석이 어렵다. 이것이 바로 국소 태양진동학이 필요한 이유 중 하나이다 — 고유 모드의 개별 분해에 의존하지 않고 파동장 자체를 직접 분석할 수 있기 때문이다.

For $l > 150$, wave damping becomes significant and ridges gradually become continuous structures. This is because higher-degree modes have shorter lifetimes due to scattering and absorption by convection. In this regime, modes are not individually resolved, making mode-by-mode analysis of global helioseismology difficult. This is precisely one reason local helioseismology is needed — it can directly analyze the wavefield itself without relying on individual mode resolution.

---

### 3. 태양 진동의 모델 (§3) / Models of Solar Oscillations (§3)

이 장은 리뷰의 이론적 핵심이다. 국소 태양진동학의 모든 기법은 결국 이 장에서 정립된 파동 이론에 기반한다.

This section is the theoretical core of the review. All local helioseismology techniques are ultimately based on the wave theory established in this section.

#### 3.1 선형 파동 방정식 / Linear Wave Equation

배경 상태(background state)는 다음의 힘 균형(force balance)을 만족한다:

The background state satisfies the following force balance:

$$\rho_0 \mathbf{v}_0 \cdot \nabla \mathbf{v}_0 = -\nabla p_0 + \rho_0 \mathbf{g}_0$$

여기서 $\rho_0$는 배경 밀도, $\mathbf{v}_0$는 배경 유동(대류, 회전, 자오면 흐름 등), $p_0$는 배경 압력, $\mathbf{g}_0$는 중력이다. 이 배경 위에 작은 변위 $\boldsymbol{\xi}$가 중첩되면, 선형화된 파동 방정식은 다음과 같다:

Here $\rho_0$ is background density, $\mathbf{v}_0$ is background flow (convection, rotation, meridional flow, etc.), $p_0$ is background pressure, and $\mathbf{g}_0$ is gravity. Superimposing a small displacement $\boldsymbol{\xi}$ on this background yields the linearized wave equation:

$$\mathcal{L}\boldsymbol{\xi} = \mathbf{S}$$

여기서 $\mathcal{L}$은 파동 연산자, $\mathbf{S}$는 확률적 소스(stochastic source)이다. 비강제(unforced) 경우 $\mathcal{L}\boldsymbol{\xi} = 0$이다. 연산자 $\mathcal{L}$의 구체적 형태는 다음과 같다:

Here $\mathcal{L}$ is the wave operator and $\mathbf{S}$ is a stochastic source. In the unforced case, $\mathcal{L}\boldsymbol{\xi} = 0$. The explicit form of operator $\mathcal{L}$ is:

$$\mathcal{L}\boldsymbol{\xi} = -\rho_0 \frac{d_0^2 \boldsymbol{\xi}}{dt^2} + \nabla[\gamma p_0 \nabla \cdot \boldsymbol{\xi} + \boldsymbol{\xi} \cdot \nabla p_0] - (\nabla \cdot \boldsymbol{\xi})\nabla p_0 - \boldsymbol{\xi} \cdot \nabla(\nabla p_0)$$

여기서 $d_0/dt = \partial/\partial t + \mathbf{v}_0 \cdot \nabla$는 배경 흐름에 대한 물질 도함수(material derivative)이다. 이 연산자에는 여러 물리적 효과가 포함되어 있다: 첫째 항은 관성, 둘째 항은 압력 섭동에 의한 복원력, 셋째와 넷째 항은 부력(buoyancy)과 관련된 효과이다. 배경 흐름 $\mathbf{v}_0$는 물질 도함수를 통해 파동에 영향을 준다 — 흐름 방향으로 전파하는 파동은 빨라지고, 반대 방향은 느려진다.

Here $d_0/dt = \partial/\partial t + \mathbf{v}_0 \cdot \nabla$ is the material derivative following the background flow. This operator contains several physical effects: the first term is inertia, the second is the restoring force from pressure perturbation, and the third and fourth are buoyancy-related effects. The background flow $\mathbf{v}_0$ affects waves through the material derivative — waves propagating in the flow direction speed up, and those against the flow slow down.

#### 3.2 파동 여기 / Wave Excitation

태양 진동은 대류층 상부의 난류 대류(turbulent convection)에 의해 확률적으로 여기된다 (Goldreich et al. 1994). 두 가지 주요 소스가 있다:

Solar oscillations are stochastically excited by turbulent convection in the upper convection zone (Goldreich et al. 1994). There are two main sources:

1. **Reynolds 응력 (난류 압력) / Reynolds stress (turbulent pressure)**: 난류 속도 요동의 비선형 상호작용이 음파를 생성한다. 이것이 지배적인 여기 메커니즘이다.
   / Nonlinear interaction of turbulent velocity fluctuations generates acoustic waves. This is the dominant excitation mechanism.

2. **엔트로피 요동 / Entropy fluctuations**: 대류에 의한 온도/엔트로피 비균일성도 파동을 여기한다. 그 기여는 Reynolds 응력보다 작지만 무시할 수 없다.
   / Temperature/entropy inhomogeneities from convection also excite waves. Their contribution is smaller than Reynolds stress but not negligible.

소스의 통계적 성질은 공분산 행렬(source covariance matrix) $M_{ij}(\mathbf{r}, \mathbf{r}', t-t') = \langle S_i(\mathbf{r},t) S_j^*(\mathbf{r}',t') \rangle$로 기술된다. 이 확률적 여기 모델은 국소 태양진동학에서 관측량의 기대값과 잡음 특성을 이해하는 데 핵심적이다 — 관측되는 파워 스펙트럼의 형태(능선 폭, 비대칭성 등)와 cross-covariance의 통계적 성질이 모두 이 소스 모델에서 도출된다.

The statistical properties of the source are described by the source covariance matrix $M_{ij}(\mathbf{r}, \mathbf{r}', t-t') = \langle S_i(\mathbf{r},t) S_j^*(\mathbf{r}',t') \rangle$. This stochastic excitation model is essential for understanding expected values and noise properties of observables in local helioseismology — the shape of the observed power spectrum (ridge width, asymmetry, etc.) and the statistical properties of cross-covariances are all derived from this source model.

#### 3.3 Green 함수 / Green's Functions

Green 함수 $G^i_j(\mathbf{r},t;\mathbf{r}',t')$는 위치 $\mathbf{r}'$에서 시각 $t'$에 $j$ 방향으로 가한 충격력(impulse)에 의해 위치 $\mathbf{r}$에서 시각 $t$에 나타나는 $i$ 방향 변위 응답이다. 이는 파동 문제의 가장 기본적인 구성 요소이다 — 임의의 소스에 대한 파동장을 Green 함수와 소스의 합성곱(convolution)으로 구할 수 있다.

The Green's function $G^i_j(\mathbf{r},t;\mathbf{r}',t')$ is the $i$-direction displacement response at position $\mathbf{r}$ and time $t$ due to an impulse applied in the $j$-direction at position $\mathbf{r}'$ and time $t'$. This is the most fundamental building block of the wave problem — the wavefield for any source can be obtained as a convolution of the Green's function with the source.

Green 함수를 구하는 세 가지 접근 방법은 다음과 같다:

Three approaches to computing the Green's function are:

**(1) 직접 수치 해법 / Direct numerical solution**: Fourier 영역에서 $(\mathcal{L}\mathbf{G})_j^i = \delta_{ij}\delta(\mathbf{r}-\mathbf{r}')$를 수치적으로 풀어 Green 함수를 직접 구한다. 가장 정확하지만 계산 비용이 높다.
/ Numerically solve $(\mathcal{L}\mathbf{G})_j^i = \delta_{ij}\delta(\mathbf{r}-\mathbf{r}')$ in the Fourier domain to directly obtain the Green's function. Most accurate but computationally expensive.

**(2) Normal-mode 급수 전개 / Normal-mode summation**: Green 함수를 고유 모드의 급수로 전개한다:
/ Expand the Green's function as a series of eigenmodes:

$$G_j^i(\mathbf{r},t;\mathbf{r}',t') = \sum_\beta s_j^\beta(\mathbf{r}) \, s_i^\beta(\mathbf{r}') \, \sin[\omega_\beta(t-t')]$$

여기서 $s_j^\beta$는 모드 $\beta$의 고유 함수, $\omega_\beta$는 해당 고유 진동수이다. 이 방법은 저차 모드가 지배적인 경우에 효율적이나, 많은 모드를 합산해야 할 경우 수렴이 느릴 수 있다.
/ Here $s_j^\beta$ is the eigenfunction of mode $\beta$ and $\omega_\beta$ is the corresponding eigenfrequency. This method is efficient when low-order modes dominate, but convergence can be slow when many modes must be summed.

**(3) 평면 평행 근사 / Plane-parallel approximation**: 국소적으로 태양을 평면으로 근사하면, Green 함수를 수직 성분 $G_z$와 수평 성분 $G_h$로 분리할 수 있다. 이 근사는 수평 파장이 태양 반경보다 훨씬 작은 경우($l \gg 1$)에 유효하며, 국소 태양진동학에서 가장 흔히 사용된다.
/ If the Sun is locally approximated as a plane, the Green's function can be separated into vertical $G_z$ and horizontal $G_h$ components. This approximation is valid when horizontal wavelengths are much smaller than the solar radius ($l \gg 1$), and is most commonly used in local helioseismology.

**관측 가능량에 대한 Green 함수 / Green's function for observables**: 실제 관측량은 변위나 속도 그 자체가 아니라 시선 방향 도플러 신호이므로, 관측 가능량에 대한 Green 함수를 별도로 정의해야 한다:

Since the actual observable is not displacement or velocity itself but the line-of-sight Doppler signal, the Green's function for the observable must be separately defined:

$$\Phi(\mathbf{x},t) = \int d^3\mathbf{r}' \int dt' \, \mathcal{G}(\mathbf{x},t;\mathbf{r}',t') \cdot \mathbf{S}(\mathbf{r}',t')$$

여기서 $\mathcal{G}$는 소스로부터 관측 가능량까지의 Green 함수이다.
/ Here $\mathcal{G}$ is the Green's function from source to observable.

#### 3.4 0차 문제와 파워 스펙트럼 / Zero-Order Problem and Power Spectrum

0차 문제(zero-order problem)는 균일한 배경 매질에서의 파동 전파를 기술한다: $\mathcal{L}^0 \boldsymbol{\xi}^0 = \mathbf{S}^0$. 관측 가능량의 파워 스펙트럼은 다음과 같이 정의된다:

The zero-order problem describes wave propagation in a uniform background medium: $\mathcal{L}^0 \boldsymbol{\xi}^0 = \mathbf{S}^0$. The power spectrum of the observable is defined as:

$$P(\mathbf{k}, \omega) = \frac{(2\pi)^3}{AT} \, \mathbb{E}[|\Phi(\mathbf{k}, \omega)|^2]$$

여기서 $A$는 관측 영역의 면적, $T$는 관측 시간, $\mathbb{E}$는 앙상블 평균(소스의 실현에 대한 기대값)이다. 이 파워 스펙트럼은 $l$-$\nu$ 도표에서 보이는 능선 구조를 재현하며, 각 능선의 폭은 모드의 감쇠율(damping rate)에, 비대칭성은 소스의 위치(관측 높이에 대한 상대적 위치)에 의해 결정된다.

Here $A$ is the observation area, $T$ is the observation time, and $\mathbb{E}$ is the ensemble average (expectation over source realizations). This power spectrum reproduces the ridge structures seen in the $l$-$\nu$ diagram, where each ridge's width is determined by the mode damping rate and asymmetry by the source location (relative to the observation height).

#### 3.5 1차 섭동과 Born 근사 — 역문제의 기초 / First-Order Perturbations and the Born Approximation — Foundation of the Inverse Problem

이 절은 국소 태양진동학 역문제의 이론적 핵심이다. 배경 매질에 작은 섭동(perturbation) $\delta q_\alpha(\mathbf{r})$이 있을 때 (예: 음속, 밀도, 흐름 속도의 변화), 관측 가능량 $d_i$(예: travel time, frequency shift)에 대한 효과는 Born 근사(1차 섭동)로 다음과 같이 기술된다:

This section is the theoretical core of the local helioseismology inverse problem. When there are small perturbations $\delta q_\alpha(\mathbf{r})$ in the background medium (e.g., changes in sound speed, density, flow velocity), the effect on observable $d_i$ (e.g., travel time, frequency shift) is described in the Born approximation (first-order perturbation) as:

$$\boxed{\delta d_i = \sum_\alpha \int d^3\mathbf{r} \, K_\alpha^i(\mathbf{r}) \, \delta q_\alpha(\mathbf{r})}$$

이것이 국소 태양진동학 역문제의 **근본 방정식**이다. 여기서:

This is the **fundamental equation** of the local helioseismology inverse problem. Here:

- $\delta d_i$: 관측량의 섭동 (travel time 변화, 진동수 이동 등) / Perturbation in the observable (travel time change, frequency shift, etc.)
- $K_\alpha^i(\mathbf{r})$: 감도 커널(sensitivity kernel). 위치 $\mathbf{r}$에서 물리량 $\alpha$의 단위 섭동이 관측량 $d_i$에 미치는 영향을 나타낸다 / Sensitivity kernel. Represents the effect of a unit perturbation of quantity $\alpha$ at position $\mathbf{r}$ on observable $d_i$
- $\delta q_\alpha(\mathbf{r})$: 물리량 $\alpha$의 3차원 섭동 / 3D perturbation of physical quantity $\alpha$

이 방정식의 물리적 의미는 명확하다: 관측량의 변화는 내부 구조의 변화에 대한 **가중 적분(weighted integral)**이며, 가중 함수가 바로 감도 커널이다. 역문제(inverse problem)란 측정된 $\delta d_i$로부터 미지의 $\delta q_\alpha(\mathbf{r})$를 복원하는 것이다.

The physical meaning of this equation is clear: the change in the observable is a **weighted integral** over changes in internal structure, where the weighting function is the sensitivity kernel. The inverse problem is to recover the unknown $\delta q_\alpha(\mathbf{r})$ from the measured $\delta d_i$.

#### 3.6 Born 근사의 유효성 검증 / Tests of the Born Approximation

Born 근사의 유효성은 섭동의 크기와 공간 규모에 따라 달라진다:

The validity of the Born approximation depends on the size and spatial scale of perturbations:

- **약한 섭동, 큰 규모**: Born 근사가 우수하게 작동한다. 예를 들어 차등 회전, 자오면 흐름, 음속의 작은 변화 등.
  / Born approximation works excellently. For example, differential rotation, meridional flow, small sound-speed changes.

- **약한 섭동, 작은 규모**: 섭동의 크기가 제1 Fresnel 영역(first Fresnel zone)보다 작은 경우, **광선 근사(ray approximation)는 실패**하지만 Born 근사는 여전히 유효하다. 이는 Born 근사가 파동의 회절 효과를 올바르게 포착하기 때문이다.
  / When the perturbation size is smaller than the first Fresnel zone, the **ray approximation fails** but the Born approximation remains valid. This is because the Born approximation correctly captures wave diffraction effects.

- **강한 섭동**: Born 근사가 실패할 수 있다. 특히 흑점 내부 표면 근처에서는 자기장에 의한 강한 파동-자기장 상호작용이 일어난다.
  / Born approximation may fail. Particularly near the surface inside sunspots, strong wave-magnetic field interactions occur.

이러한 유효성 범위의 이해는 실용적으로 매우 중요하다. 역문제를 풀 때, 해석의 신뢰성은 forward model(Born 근사)의 정확성에 직접 의존하기 때문이다.

Understanding this validity range is practically very important. When solving the inverse problem, the reliability of the interpretation directly depends on the accuracy of the forward model (Born approximation).

#### 3.7 강한 섭동: 자기 플럭스 튜브와 흑점 / Strong Perturbations: Magnetic Flux Tubes and Sunspots

Born 근사가 실패하는 영역에서는 보다 정교한 모델이 필요하다. 자기 플럭스 튜브(magnetic flux tubes)는 독자적인 파동 모드를 지원한다:

In regimes where the Born approximation fails, more sophisticated models are needed. Magnetic flux tubes support their own wave modes:

- **Alfvén 모드 / Alfvén modes**: 자기장 방향을 따라 전파하는 횡파. $v_A = B/\sqrt{4\pi\rho}$.
  / Transverse waves propagating along the magnetic field direction. $v_A = B/\sqrt{4\pi\rho}$.

- **Sausage 모드 / Sausage modes**: 플럭스 튜브의 축대칭 수축/팽창. 튜브 단면이 균일하게 팽창하고 수축한다.
  / Axisymmetric contraction/expansion of the flux tube. The tube cross-section uniformly expands and contracts.

- **Kink 모드 / Kink modes**: 플럭스 튜브의 횡방향 변위. 튜브 전체가 옆으로 흔들린다.
  / Transverse displacement of the flux tube. The entire tube sways sideways.

특히 중요한 것은 **모드 변환(mode conversion)**이다: p-mode가 자기장이 강한 영역에 들어가면, 음파 에너지의 일부가 자기 음파(magneto-acoustic wave) 또는 Alfvén 파로 변환된다. 이것이 흑점에 의한 p-mode 흡수(absorption)의 물리적 메커니즘 중 하나이다. 변환된 에너지는 자기장을 따라 상층 대기로 전달되어 관측되지 않으므로, 결과적으로 p-mode 에너지가 "흡수"된 것처럼 보인다.

Particularly important is **mode conversion**: when p-modes enter a region of strong magnetic field, part of the acoustic energy is converted into magneto-acoustic waves or Alfvén waves. This is one of the physical mechanisms of p-mode absorption by sunspots. The converted energy propagates along the magnetic field into the upper atmosphere and becomes unobservable, so it appears as if p-mode energy has been "absorbed."

---

### 4. 국소 태양진동학의 기법 (§4) — 리뷰의 핵심 / Methods of Local Helioseismology (§4) — The Core of the Review

이 장은 36페이지에 달하는 리뷰의 핵심 부분으로, 다섯 가지 주요 기법의 이론과 구현을 상세히 기술한다.

This section, spanning 36 pages, is the core of the review, detailing the theory and implementation of five major techniques.

#### 4.1 Fourier-Hankel 분광법 (Braun et al. 1987) / Fourier-Hankel Spectral Method (Braun et al. 1987)

**기본 아이디어 / Basic idea**: 흑점이나 활동 영역 주변의 고리(annulus)에서 파동장을 안쪽으로 향하는(ingoing) 성분과 바깥으로 향하는(outgoing) 성분으로 분해한다. 이를 위해 파동장을 Hankel 함수 $H_m^{(1)}$, $H_m^{(2)}$ (원통 좌표에서 안쪽/바깥쪽 전파파에 대응)로 전개한다.

Decompose the wavefield in an annulus around a sunspot or active region into ingoing and outgoing components. For this, expand the wavefield in Hankel functions $H_m^{(1)}$, $H_m^{(2)}$ (corresponding to inward/outward propagating waves in cylindrical coordinates).

**흡수 계수 / Absorption coefficient**: 안쪽 진폭 $A_m$과 바깥 진폭 $B_m$의 비로 정의된다:

Defined as the ratio of ingoing amplitude $A_m$ and outgoing amplitude $B_m$:

$$\alpha_m(L, \nu) = 1 - \frac{|B_m|^2}{|A_m|^2}$$

**핵심 발견 / Key finding**: 흑점은 p-mode 파워의 약 **50%**를 흡수한다. 이 발견은 Braun et al. (1987)의 역사적 결과이며, 국소 태양진동학의 시작을 알린 중요한 관측이었다. 흡수 메커니즘은 아직 완전히 규명되지 않았으나, 앞서 언급한 모드 변환이 주요 후보이다.

Sunspots absorb approximately **50%** of incoming p-mode power. This finding is the historical result of Braun et al. (1987) and was an important observation marking the beginning of local helioseismology. The absorption mechanism is not yet fully understood, but the mode conversion mentioned above is the leading candidate.

**위상 이동 / Phase shifts**: 안쪽 파와 바깥 파 사이의 위상 차이는 산란(scattering)에 민감하다. 이 정보는 흡수 계수와 결합하여 산란체의 물리적 성질(크기, 강도 등)에 대한 제약을 제공한다.

The phase difference between ingoing and outgoing waves is sensitive to scattering. This information, combined with the absorption coefficient, provides constraints on the physical properties (size, strength, etc.) of the scatterer.

**모드 혼합(mode mixing)**: 보다 일반적으로, 산란 행렬(scattering matrix) $S_{ij}$가 안쪽 모드와 바깥 모드를 연결한다: $B_i = \sum_j S_{ij} A_j$. 이는 한 모드에서 다른 모드로의 에너지 전달을 기술하며, 흡수와 위상 이동을 통일된 틀에서 다룬다.

More generally, the scattering matrix $S_{ij}$ connects ingoing and outgoing modes: $B_i = \sum_j S_{ij} A_j$. This describes energy transfer from one mode to another and treats absorption and phase shifts in a unified framework.

#### 4.2 Ring-Diagram 분석 (Hill 1988) / Ring-Diagram Analysis (Hill 1988)

**기본 아이디어 / Basic idea**: 태양 표면의 15°×15° 패치(patch)를 추적(track)하면서 그 영역의 국소적인 $\mathbf{k}$-$\omega$ 파워 스펙트럼을 구한다. 일정한 진동수에서의 파워 분포를 ($k_x$, $k_y$) 공간에 그리면, p-mode 능선들이 **고리(ring)** 형태로 나타난다 — 이것이 방법의 이름이 된 유래이다.

Track a 15°×15° patch on the solar surface and compute the local $\mathbf{k}$-$\omega$ power spectrum for that region. Plotting the power distribution at a constant frequency in ($k_x$, $k_y$) space reveals p-mode ridges as **rings** — hence the method's name.

**흐름에 의한 고리 이동 / Ring shift by flows**: 수평 흐름이 있으면 도플러 효과로 인해 진동수가 이동하고, 고리가 흐름 방향으로 치우친다. 구체적으로, 파워 프로필은 다음 Lorentz 함수로 적합(fit)된다:

Horizontal flows shift the frequency via the Doppler effect, causing the ring to displace in the flow direction. Specifically, the power profile is fit with the following Lorentzian:

$$P_{\text{fit}} = \frac{A}{1 + \frac{(\omega - \omega_0 - k U_x \cos\psi - k U_y \sin\psi)^2}{\gamma^2}} + B k^{-3}$$

여기서:
/ Where:

- $\omega_0$: 섭동 없는 중심 진동수 / Unperturbed central frequency
- $U_x, U_y$: 수평 흐름 성분 / Horizontal flow components
- $\gamma$: 선폭(모드 감쇠율) / Linewidth (mode damping rate)
- $\psi$: 파수 벡터의 방위각 / Azimuth of the wavenumber vector
- $Bk^{-3}$: 배경 잡음 / Background noise

적합된 $U_x$, $U_y$는 해당 패치 내의 깊이-평균된 수평 흐름을 나타낸다. 깊이별 정보를 얻기 위해서는 다른 모드($n$, $l$)의 결과를 결합하여 역문제를 풀어야 한다.

The fitted $U_x$, $U_y$ represent the depth-averaged horizontal flows within that patch. To obtain depth-resolved information, results from different modes ($n$, $l$) must be combined and an inverse problem must be solved.

**깊이 역산(depth inversion)**: 진동수 이동을 깊이 함수로 분해하기 위해 다음의 적분 관계를 사용한다:

For decomposing frequency shifts as a function of depth, the following integral relation is used:

$$\frac{\delta\omega_{nl}}{\omega_{nl}} = \int dz \, K_{c,\rho}^{nl}(z) \, \frac{\delta c}{c} + \int dz \, K_{\rho,c}^{nl}(z) \, \frac{\delta\rho}{\rho} + \frac{F(\omega_{nl})}{I_{nl}}$$

여기서 $K_{c,\rho}^{nl}$과 $K_{\rho,c}^{nl}$은 구조 커널(structural kernel), $F/I$는 표면 효과(surface term)이다. 역산 방법으로는 **RLS(Regularized Least Squares)**와 **OLA(Optimally Localized Averages)**가 사용된다.

Here $K_{c,\rho}^{nl}$ and $K_{\rho,c}^{nl}$ are structural kernels and $F/I$ is the surface term. Inversion methods include **RLS (Regularized Least Squares)** and **OLA (Optimally Localized Averages)**.

- **RLS**: 정규화된 최소자승법. 해의 매끄러움(smoothness)에 대한 제약을 부과하여 불안정성을 억제한다. $\min_{\delta q} \left[ \|\delta\mathbf{d} - \mathbf{K}\delta\mathbf{q}\|^2 + \lambda \|\mathbf{L}\delta\mathbf{q}\|^2 \right]$.
  / Regularized least-squares. Imposes smoothness constraints on the solution to suppress instability.

- **OLA**: 최적 국소화 평균. 특정 목표 위치에서의 해가 가능한 한 국소적이 되도록 가중치를 선택한다. 공간 분해능과 잡음 증폭 사이의 트레이드오프(trade-off)를 명시적으로 제어할 수 있다.
  / Optimally localized averages. Chooses weights so the solution at a specific target location is as localized as possible. Explicitly controls the trade-off between spatial resolution and noise amplification.

**장점과 한계 / Strengths and limitations**: Ring-diagram 분석은 상대적으로 구현이 간단하고 빠르며, 많은 패치를 동시에 분석하여 전구적 지도를 만들 수 있다. 그러나 15° 패치의 공간 분해능으로 인해 미세 구조(흑점 내부 구조 등)의 해석에는 한계가 있다.

Ring-diagram analysis is relatively simple to implement and fast, allowing simultaneous analysis of many patches to create global maps. However, the spatial resolution of 15° patches limits the interpretation of fine structures (such as internal sunspot structure).

#### 4.3 Time-Distance 태양진동학 (Duvall et al. 1993) / Time-Distance Helioseismology (Duvall et al. 1993)

이 기법은 아마도 국소 태양진동학에서 가장 널리 사용되는 방법이다.

This is perhaps the most widely used method in local helioseismology.

**기본 아이디어 / Basic idea**: 두 점 $\mathbf{x}_1$, $\mathbf{x}_2$ 사이의 교차 공분산(cross-covariance) $C(\mathbf{x}_1, \mathbf{x}_2, t)$를 태양 지진파(seismogram)로 해석한다. 이 함수는 한 점에서 출발한 파동이 태양 내부를 통과하여 다른 점에 도달하는 데 걸리는 시간(travel time)에 대한 정보를 담고 있다.

Interpret the cross-covariance $C(\mathbf{x}_1, \mathbf{x}_2, t)$ between two points as a solar seismogram. This function contains information about the travel time for waves departing from one point, passing through the solar interior, and arriving at another point.

$$C(\mathbf{x}_1, \mathbf{x}_2, t) = \frac{1}{T} \int_0^T dt' \, \Phi(\mathbf{x}_1, t') \, \Phi(\mathbf{x}_2, t'+t)$$

이 교차 공분산은 시간 지연(time lag) $t$의 함수로서, 양의 $t$와 음의 $t$에서 봉우리(peak)를 보인다. 양의 시간 지연은 $\mathbf{x}_1$에서 $\mathbf{x}_2$로의 전파를, 음의 시간 지연은 반대 방향의 전파를 나타낸다.

This cross-covariance, as a function of time lag $t$, shows peaks at both positive and negative $t$. Positive time lag represents propagation from $\mathbf{x}_1$ to $\mathbf{x}_2$, and negative time lag represents propagation in the opposite direction.

**Fourier 필터링 / Fourier filtering**: 파동의 깊이별 민감도를 선택하기 위해 위상 속도 필터(phase-speed filter)를 적용한다:

Phase-speed filters are applied to select depth-specific sensitivity:

$$F_i(k, \omega) = \exp\left[-\frac{(\omega/k - v_i)^2}{2\delta v_i^2}\right]$$

이 필터는 특정 위상 속도 $v_i$ 주변의 모드만 통과시킨다. 위상 속도가 높을수록 더 깊이 침투하는 파동을 선택하므로, 필터의 선택에 따라 탐사 깊이를 제어할 수 있다.

This filter passes only modes around a specific phase speed $v_i$. Higher phase speeds select waves penetrating deeper, so the probing depth can be controlled by filter selection.

**Travel time 측정 / Travel time measurement**: 교차 공분산으로부터 travel time $\tau_+$ (양의 방향)와 $\tau_-$ (음의 방향)를 추출하는 방법에는 두 가지 주요 접근이 있다:

Two main approaches exist for extracting travel times $\tau_+$ (positive direction) and $\tau_-$ (negative direction) from the cross-covariance:

1. **Wavelet 적합법 / Wavelet fitting**: 교차 공분산의 봉우리를 wavelet(예: Gabor 함수)로 적합하여 봉우리 위치를 결정한다.
   / Fit the cross-covariance peaks with a wavelet (e.g., Gabor function) to determine peak positions.

2. **Gizon-Birch (2004) 정의 / Gizon-Birch (2004) definition**: 보다 체계적인 정의로, 참조 교차 공분산(reference cross-covariance) $C^0$에 대한 시간 이동을 선형적으로 추정한다. 이 정의의 장점은 잡음의 통계적 성질(공분산)을 해석적으로 계산할 수 있다는 것이다.
   / A more systematic definition that linearly estimates the time shift relative to a reference cross-covariance $C^0$. The advantage of this definition is that the statistical properties (covariance) of noise can be analytically computed.

**Travel time의 물리적 해석 / Physical interpretation of travel times**:

- **Travel time 차이(difference)**: $\tau_{\text{diff}} = \tau_+ - \tau_-$. 이것은 **흐름(flows)**에 민감하다. 흐름이 있으면 순방향 전파와 역방향 전파의 시간이 달라지기 때문이다. 예를 들어, 한 방향으로 10 m/s의 흐름이 있으면 순방향 파동은 빨라지고 역방향은 느려져 travel time 차이가 발생한다.
  / This is sensitive to **flows**. When flows exist, forward and backward propagation times differ. For example, a 10 m/s flow in one direction speeds up the forward wave and slows the backward wave, producing a travel-time difference.

- **Travel time 평균(mean)**: $\tau_{\text{mean}} = (\tau_+ + \tau_-)/2$. 이것은 **음속(sound speed)**과 **구조적 섭동**에 민감하다. 흐름의 효과는 상쇄되고, 경로를 따른 음속 변화에 의한 시간 변화가 남는다.
  / This is sensitive to **sound speed** and **structural perturbations**. Flow effects cancel out, leaving time changes due to sound-speed variations along the path.

**잡음 특성 / Noise properties**: Travel time의 주된 잡음 원인은 확률적 여기에 의한 "실현 잡음(realization noise)"이다. 신호 대 잡음비(SNR)는 $\sqrt{T}$에 비례한다 — 즉 관측 시간이 4배가 되면 잡음은 반으로 줄어든다. 또한 travel time의 공간적 상관 길이는 지배적인 파동의 반파장(half dominant wavelength)에 해당한다. 이 상관 길이보다 가까이 있는 점들의 travel time은 통계적으로 독립이 아니므로, 역문제를 풀 때 이 상관을 고려해야 한다.

The main noise source in travel times is "realization noise" from stochastic excitation. The signal-to-noise ratio (SNR) scales as $\sqrt{T}$ — so quadrupling the observation time halves the noise. The spatial correlation length of travel times corresponds to half the dominant wavelength. Travel times at points closer than this correlation length are not statistically independent, so this correlation must be accounted for in the inverse problem.

**감도 커널 / Sensitivity kernels**: Travel time을 내부 구조의 섭동과 연결하는 커널에는 세 가지 수준의 근사가 있다:

Three levels of approximation exist for kernels connecting travel times to internal structure perturbations:

**(1) 광선 근사(ray approximation)**: 가장 단순한 근사. Travel time을 광선 경로(ray path)를 따른 선적분으로 나타낸다:
/ Simplest approximation. Represents travel time as a line integral along the ray path:

$$\delta\tau \approx -\int_{\text{ray}} \frac{\delta c}{c^2} \, ds$$

여기서 $ds$는 광선 경로를 따른 호 길이이다. 기하 광학의 한계로 인해 Fresnel 영역보다 작은 구조에 대해서는 부정확하다.
/ Here $ds$ is the arc length along the ray path. Due to geometric optics limitations, it is inaccurate for structures smaller than the Fresnel zone.

**(2) Fresnel 영역 근사 / Fresnel zone approximation**: 광선 근사를 확장하여 광선 주변의 Fresnel 영역에 분포한 커널을 사용한다. 1차 파동 효과를 근사적으로 포함한다.
/ Extends the ray approximation by using kernels distributed over the Fresnel zone around the ray. Approximately includes first-order wave effects.

**(3) Born 근사 — "바나나-도넛(banana-doughnut)" 커널**: 가장 정확한 근사. Born 근사로 도출된 3D 감도 커널은 매우 특이한 형태를 보인다: 광선 경로를 따라 **속이 빈** 바나나 모양이다. 즉, 커널의 민감도가 광선 경로 바로 위에서는 0이고, 그 주변에서 최대값을 가진다. 이것이 "바나나-도넛" 커널이라 불리는 이유이며, 이 결과는 지진학(seismology)에서 먼저 발견된 것이 태양진동학에 도입된 것이다.

The most accurate approximation. 3D sensitivity kernels derived from the Born approximation have a very peculiar shape: a **hollow** banana shape along the ray path. The kernel sensitivity is zero right on the ray path and reaches its maximum around it. This is why they are called "banana-doughnut" kernels, a result first discovered in seismology and later introduced to helioseismology.

이 결과의 물리적 직관은 다음과 같다: 광선 경로 바로 위의 섭동은 파동의 진폭만 변화시키고 위상은 변화시키지 않으므로, travel time(위상 기반 측정)에 대한 민감도가 0이다. 반면 광선 경로에서 약간 벗어난 곳의 섭동은 산란파와 직접파 사이의 간섭을 통해 위상(즉 travel time)을 변화시킨다.

The physical intuition behind this result is: perturbations right on the ray path change only the wave amplitude, not the phase, so the sensitivity to travel time (a phase-based measurement) is zero. Perturbations slightly off the ray path, however, change the phase (i.e., travel time) through interference between scattered and direct waves.

**역산(Inversions)**: Travel time 데이터로부터 3D 구조를 복원하는 데 RLS 또는 SOLA(Subtractive Optimally Localized Averages)가 사용된다. 3D 커널을 사용한 역산은 계산 비용이 높지만, 광선 근사 역산에 비해 훨씬 정확한 결과를 제공한다.

RLS or SOLA (Subtractive Optimally Localized Averages) are used to recover 3D structure from travel-time data. Inversion with 3D kernels is computationally expensive but provides much more accurate results than ray-theory inversions.

#### 4.4 태양진동 Holography (Lindsey & Braun 1990) / Helioseismic Holography (Lindsey & Braun 1990)

**기본 아이디어 / Basic idea**: 광학 holography에서 영감을 받은 기법. 표면에서 관측된 파동장을 시간적으로 "되돌려서(backpropagate)" 태양 내부의 특정 깊이에서의 파동장을 추정한다. 이는 지구 물리학의 seismic migration과 유사한 개념이다.

A technique inspired by optical holography. "Backpropagates" the wavefield observed at the surface to estimate the wavefield at specific depths inside the Sun. This concept is similar to seismic migration in geophysics.

**Egression과 Ingression**: 두 가지 기본 물리량이 정의된다:

Two fundamental quantities are defined:

- **Egression $H^+$**: 동공(pupil) $P$ 영역에서 관측된 **발산파(diverging waves)**로부터 초점(focus) 위치 $\mathbf{r}$에서의 파동장을 추정한다:
  / Estimates the wavefield at focus position $\mathbf{r}$ from **diverging waves** observed in the pupil $P$:

$$H_+^P(\mathbf{r}, \omega) = \int_P d^2\mathbf{x}' \, G_+^{\text{holo}}(\mathbf{x}-\mathbf{x}', z, \omega) \, \Phi(\mathbf{x}', \omega)$$

- **Ingression $H^-$**: 동공 $P$ 영역에서 관측된 **수렴파(converging waves)**로부터 초점 위치에서의 파동장을 추정한다. Egression과 유사한 형태이나 $G_-^{\text{holo}}$를 사용한다.
  / Estimates the wavefield at the focus from **converging waves** observed in the pupil $P$. Similar form to egression but uses $G_-^{\text{holo}}$.

**Holographic Green 함수**: 이 역전파에 사용되는 Green 함수에는 두 가지 선택이 있다:

Two choices exist for the Green's functions used in backpropagation:

- **광선 이론(ray theory)**: Eq. 71. 계산이 빠르지만 파동 효과를 무시한다.
  / Fast to compute but ignores wave effects.
- **파동 이론(wave theory)**: Eqs. 72-73. 더 정확하지만 계산 비용이 높다.
  / More accurate but computationally expensive.

**다양한 holography 변형 / Various holography variants**:

**(1) 국소 제어 상관(local control correlations)**:
$$C_\pm = \langle \Phi \cdot H_\mp^* \rangle$$

이것은 표면에서의 관측과 ingression/egression의 상관으로, 특정 초점 아래의 물리적 성질에 민감하다.
/ Correlations between surface observations and ingression/egression, sensitive to physical properties below the specific focus.

**(2) 음파 파워 holography(acoustic power holography)**:
$$\mathcal{P}(\mathbf{r}, \omega) = |H_+(\mathbf{r}, \omega)|^2$$

이것은 초점 위치에서의 파동 여기 강도를 추정한다. 태양 플레어(flare)에 의한 음파 방출을 감지하는 데 사용된다.
/ Estimates wave excitation intensity at the focus. Used to detect acoustic emission from solar flares.

**(3) 위상 민감 holography(phase-sensitive holography)**: 두 동공 $P$, $P'$를 사용하여:
$$C_{P,P'} = \langle H_-^P \cdot H_+^{P' *} \rangle$$

대칭 위상(symmetric phase) $\phi_s$는 음속에, 비대칭 위상(antisymmetric phase) $\phi_a$는 흐름에 민감하다. 이는 time-distance 기법의 $\tau_{\text{mean}}$과 $\tau_{\text{diff}}$에 각각 대응한다.

The symmetric phase $\phi_s$ is sensitive to sound speed, and the antisymmetric phase $\phi_a$ is sensitive to flows. These correspond to $\tau_{\text{mean}}$ and $\tau_{\text{diff}}$ in the time-distance technique, respectively.

**(4) 뒷면 영상(far-side imaging)**: 이 기법의 가장 극적인 응용 중 하나. 태양 뒷면(지구에서 보이지 않는 면)의 활동 영역을 감지할 수 있다. 원리는 다음과 같다:

One of the most dramatic applications of this technique. Can detect active regions on the far side of the Sun (invisible from Earth). The principle is:

- **2-skip 기하(geometry)**: 파동이 태양 전면 → 내부 → 뒷면 → 내부 → 전면으로 두 번 반사하는 경로. 뒷면의 활동 영역은 반사 특성을 변화시켜 travel time을 변화시킨다.
  / Waves reflect twice on a path: front → interior → back → interior → front. Active regions on the back side change reflection properties, altering travel times.

- **1-skip/3-skip 기하**: 1번 반사 경로와 3번 반사 경로를 비교하여 뒷면의 위상 이동을 검출한다.
  / Compare 1-bounce and 3-bounce paths to detect phase shifts on the far side.

뒷면 영상은 Stanford 대학에서 운영 모드(operational mode)로 일상적으로 수행되며, 태양 뒷면에서 지구 쪽으로 회전해 올 대형 활동 영역을 미리 감지하여 우주 기상 예보에 기여한다.

Far-side imaging is routinely performed in operational mode at Stanford University, contributing to space weather forecasting by detecting large active regions rotating toward Earth from the far side.

#### 4.5 직접 모델링 (Woodard 2002) / Direct Modeling (Woodard 2002)

**기본 아이디어 / Basic idea**: 다른 기법들과 달리, 중간 단계(travel time, ring fit 등)를 거치지 않고 관측된 Fourier 진폭으로부터 직접 내부 구조를 역산한다.

Unlike other techniques, inverts directly from observed Fourier amplitudes to internal structure without intermediate steps (travel time, ring fit, etc.).

**수학적 틀 / Mathematical framework**: 파수 $\mathbf{q}$를 가진 흐름은 Fourier 영역에서 $|\mathbf{k}-\mathbf{k}'| = q$를 만족하는 모드들 사이에만 상관(correlation)을 유도한다. 즉, 흐름의 공간 규모가 직접적으로 모드 간 결합의 선택 규칙을 결정한다. 이 성질을 이용하여 forward 문제와 inverse 문제를 **선형 최소 자승법(linear least-squares)**으로 풀 수 있다.

A flow with wavenumber $\mathbf{q}$ induces correlations only between modes satisfying $|\mathbf{k}-\mathbf{k}'| = q$ in the Fourier domain. That is, the spatial scale of the flow directly determines the selection rule for mode coupling. Using this property, the forward and inverse problems can be solved as **linear least-squares** problems.

이 기법의 장점은 travel time이나 frequency shift 같은 중간 측정량의 정의에 의존하지 않으므로, 정보 손실이 최소화된다는 것이다. 단점은 계산 비용이 매우 높고, 잡음 모델이 복잡하다는 것이다.

The advantage of this technique is that it does not depend on the definition of intermediate measurements such as travel times or frequency shifts, minimizing information loss. The disadvantage is very high computational cost and a complex noise model.

#### 4.6 기법 비교 요약 / Summary Comparison of Techniques

| 기법 / Method | 입력 / Input | 측정량 / Measured Quantity | 커널 근사 / Kernel Approx. | 강점 / Strength | 한계 / Limitation |
|---|---|---|---|---|---|
| Fourier-Hankel | 고리 영역 파동장 / Annular wavefield | 흡수, 위상 / Absorption, phase | — | 산란 진단 / Scattering diagnostics | 축대칭 구조에 제한 / Limited to axisymmetric |
| Ring-diagram | 15° 패치 / 15° patch | 진동수, 선폭, 흐름 / Freq., width, flows | 1D 커널 / 1D kernels | 빠르고 전구 지도 가능 / Fast, global mapping | 낮은 공간 분해능 / Low spatial resolution |
| Time-distance | 점 간 교차 공분산 / Point-pair cross-covariance | Travel time $\tau$ | Ray, Fresnel, Born | 높은 분해능, 3D / High resolution, 3D | 잡음 상관 복잡 / Complex noise correlation |
| Holography | 동공-초점 역전파 / Pupil-focus backpropagation | 위상, 진폭 / Phase, amplitude | Ray or wave | 뒷면 영상 / Far-side imaging | Green 함수 정확도 의존 / Depends on Green's fn. accuracy |
| Direct modeling | Fourier 진폭 / Fourier amplitudes | 모드 간 상관 / Mode correlations | Born | 정보 손실 최소 / Minimal info loss | 계산 비용 극히 높음 / Extremely high cost |

---

### 5. 과학적 성과 (§5) / Scientific Results (§5)

이 장은 49페이지에 걸쳐 국소 태양진동학의 주요 과학적 발견을 정리한다.

This section, spanning 49 pages, summarizes the major scientific findings of local helioseismology.

#### 5.1 전구 규모의 결과 / Global-Scale Results

##### 5.1.1 차등 회전 / Differential Rotation

Time-distance 기법과 ring-diagram 분석의 차등 회전 결과는 전구 태양진동학의 역산 결과와 잘 일치한다. 이 일치는 국소 태양진동학 기법의 신뢰성을 검증하는 중요한 벤치마크이다.

Differential rotation results from time-distance and ring-diagram analysis agree well with global helioseismology inversion results. This agreement is an important benchmark validating the reliability of local helioseismology techniques.

특히 **표면 근처 전단(near-surface shear)**이 확인되었다: 낮은 위도에서 회전 각속도가 깊이에 따라 증가한다. 즉, 태양 표면은 바로 아래의 층보다 약간 느리게 회전한다. 이 전단층의 깊이는 약 30-35 Mm이며, 초과립 대류(supergranulation)의 깊이와 관련이 있을 수 있다.

In particular, **near-surface shear** was confirmed: at low latitudes, the rotation angular velocity increases with depth. That is, the solar surface rotates slightly slower than the layer immediately below. The depth of this shear layer is approximately 30-35 Mm and may be related to the depth of supergranular convection.

##### 5.1.2 비틀림 진동 / Torsional Oscillations

약 $\pm 10$ m/s의 빠른/느린 회전 대역이 적도를 향해 이동하는 패턴이 발견되었다. 이 "비틀림 진동(torsional oscillations)"은 자기 활동 주기와 밀접하게 연관되어 있으며, 활동 대역의 적도 이동을 따라간다.

Bands of fast/slow rotation of approximately $\pm 10$ m/s migrating equatorward were discovered. These "torsional oscillations" are closely linked to the magnetic activity cycle, following the equatorward migration of the activity belt.

국소 태양진동학의 중요한 기여는 이 비틀림 진동이 **대류층 깊숙이까지** 지속됨을 보인 것이다. 이는 표면 현상이 아니라 대류층 전체에 걸친 조직화된 흐름임을 의미하며, 태양 다이나모(solar dynamo) 이론에 대한 중요한 제약 조건을 제공한다.

An important contribution of local helioseismology was showing that these torsional oscillations persist **deep into the convection zone**. This means they are not surface phenomena but organized flows spanning the entire convection zone, providing important constraints on solar dynamo theory.

##### 5.1.3 자오면 흐름 / Meridional Flow

자오면 흐름은 표면 근처에서 약 **10-20 m/s**의 속도로 극 방향(poleward)으로 흐른다. 국소 태양진동학을 통해 이 흐름이 약 12-15 Mm 깊이까지 측정되었다.

Meridional flow moves poleward near the surface at approximately **10-20 m/s**. Through local helioseismology, this flow has been measured to a depth of approximately 12-15 Mm.

그러나 2005년 시점에서 **복귀 흐름(return flow)**은 아직 검출되지 않았다. 질량 보존에 의해 어딘가에서 적도 방향으로의 복귀 흐름이 존재해야 하지만, 깊은 층에서의 관측이 어려워 그 깊이와 구조가 미확인 상태였다. 이 복귀 흐름의 위치와 구조는 flux-transport dynamo 모델의 핵심 매개변수이므로, 이를 결정하는 것이 국소 태양진동학의 중요한 미해결 과제이다.

However, as of 2005, the **return flow** had not yet been detected. Mass conservation requires an equatorward return flow somewhere, but the depth and structure remained unconfirmed due to difficulty observing deep layers. The location and structure of this return flow is a key parameter for flux-transport dynamo models, making its determination an important unsolved problem for local helioseismology.

태양 주기에 따른 자오면 흐름의 변화도 감지되었다. 활동 극대기에 흐름 패턴이 변화하며, 이는 활동 영역으로의 유입(inflow) 때문인 것으로 해석된다.

Variations in meridional flow with the solar cycle were also detected. The flow pattern changes at activity maximum, interpreted as due to inflows toward active regions.

##### 5.1.4 수직 흐름 / Vertical Flows

수직 흐름의 측정은 수평 흐름에 비해 훨씬 어렵다. 도플러 관측은 시선 방향 속도만 측정하므로, 태양 디스크 중심에서의 수직 흐름은 시선 속도와 일치하지만, 림(limb) 근처에서는 수평 흐름과 혼재된다. 여러 연구의 결과가 서로 **상충**하며, 2005년 시점에서는 아직 합의가 이루어지지 않았다.

Measuring vertical flows is much more difficult than horizontal flows. Since Doppler observations measure only line-of-sight velocity, vertical flows coincide with line-of-sight velocity at disk center but mix with horizontal flows near the limb. Results from different studies **conflict** with each other, and no consensus had been reached as of 2005.

#### 5.2 활동 영역과 흑점 / Active Regions and Sunspots

##### 5.2.1 활동 영역으로의 유입 흐름 / Inflows Toward Active Regions

활동 영역 복합체(complexes of activity) 주변에서 **20-50 m/s**의 수렴 흐름(converging flows)이 발견되었다. 이 유입 흐름은 활동 영역 주변의 약 10-15° 범위에 걸쳐 존재하며, 활동 영역의 자기 플럭스와 관련이 있다. 이 흐름은 지표면 부근에 국한된 것으로 보이며, 깊이에 따라 감소한다.

**20-50 m/s** converging flows were discovered around complexes of activity. These inflows extend over a range of approximately 10-15° around active regions and are related to the magnetic flux of the active region. These flows appear to be confined near the surface and decrease with depth.

이 유입 흐름은 flux-transport 모델에서 중요한 역할을 할 수 있다: 자기 플럭스가 적도 방향으로 끌려가는 메커니즘을 제공하여, 선행 극성(leading polarity)의 교차 적도 소멸(cross-equatorial cancellation)을 촉진할 수 있다.

These inflows may play an important role in flux-transport models: they provide a mechanism for dragging magnetic flux equatorward, facilitating cross-equatorial cancellation of leading polarity.

##### 5.2.2 흑점의 3차원 구조 / Three-Dimensional Structure of Sunspots

Time-distance 태양진동학을 통해 흑점 아래의 3차원 유동 구조가 측정되었다:

Three-dimensional flow structures beneath sunspots were measured through time-distance helioseismology:

- **표면 근처 (~0-5 Mm 깊이)**: 강한 유입 흐름(inflow). 흑점 표면에서 물질이 안쪽으로 흘러 들어간다.
  / Strong inflows. Material flows inward at the sunspot surface.

- **더 깊은 층 (~5-10 Mm 이상)**: 유출 흐름(outflow). 이것은 표면에서 관측되는 moat flow(흑점 주변의 바깥 방향 흐름)와 연결된다.
  / Outflows. These connect to the moat flow (outward flow around the sunspot) observed at the surface.

그러나 흑점의 더 깊은 구조(>10 Mm)에 대해서는 결과가 불확실하다. Born 근사의 유효성이 흑점 내부의 강한 자기장 근처에서 의심되며, 역산 결과의 해석에 주의가 필요하다.

Results for deeper sunspot structure (>10 Mm) are uncertain. The validity of the Born approximation is questionable near the strong magnetic field inside sunspots, and care is needed in interpreting inversion results.

##### 5.2.3 음파 흡수와 위상 이동 / Acoustic Absorption and Phase Shifts

앞서 Fourier-Hankel 분석(§4.1)에서 언급했듯이, 흑점은 p-mode 파워의 약 **50%**를 흡수한다. 이 흡수율은 모드의 진동수와 구면조화 차수에 의존한다.

As mentioned in the Fourier-Hankel analysis (§4.1), sunspots absorb approximately **50%** of p-mode power. This absorption rate depends on mode frequency and spherical harmonic degree.

파동 속도 섭동(wave-speed perturbation)도 흑점 아래에서 검출된다. 표면 근처에서는 음속이 감소하고(차가운 우산 효과), 더 깊은 곳에서는 증가하는 패턴이 보고되었다. 그러나 이 깊은 층의 음속 증가가 실제 물리적 효과인지 역산 아티팩트(artifact)인지에 대해서는 논쟁이 있다.

Wave-speed perturbations are also detected beneath sunspots. Near the surface, sound speed decreases (cool umbrella effect), and at greater depth, an increase is reported. However, there is debate about whether this deep sound-speed increase is a real physical effect or an inversion artifact.

##### 5.2.4 뒷면 영상 / Far-Side Imaging

태양진동 holography의 가장 인상적인 응용이다. Stanford 대학에서 일상적으로(routinely) 수행되며, 태양 뒷면의 대형 활동 영역을 감지한다.

The most impressive application of helioseismic holography. Routinely performed at Stanford University, detecting large active regions on the Sun's far side.

원리적 한계는 공간 분해능이다: 현재의 기술로는 중간 규모 이상의 활동 영역만 감지할 수 있으며, 작은 활동 영역이나 조용한 태양(quiet Sun) 특징은 검출이 불가능하다. 그럼에도 우주 기상 예보의 관점에서 매우 귀중한 도구이다 — 지구를 향해 회전해 올 대형 활동 영역을 2주 전에 미리 알 수 있기 때문이다.

The fundamental limitation is spatial resolution: current technology can only detect medium-to-large active regions; small active regions or quiet Sun features cannot be detected. Nevertheless, it is an extremely valuable tool for space weather forecasting — it can provide 2-week advance warning of large active regions rotating toward Earth.

#### 5.3 초과립 / Supergranulation

##### 5.3.1 기본 성질 / Basic Properties

초과립(supergranulation)은 태양 표면 대류의 한 규모로, 다음과 같은 특성을 가진다:

Supergranulation is a scale of solar surface convection with the following characteristics:

- **수평 유출 흐름 / Horizontal outflows**: ~300-500 m/s. 초과립 세포의 중심에서 경계로 향하는 바깥 방향 흐름이다.
  / ~300-500 m/s. Outward flows from the center of supergranular cells toward the boundaries.

- **경계에서의 약한 하강 흐름 / Weak downflows at boundaries**: 초과립 경계에서 물질이 아래로 가라앉는다. 이 하강 흐름이 색구층 자기 네트워크(chromospheric magnetic network)를 형성한다.
  / Material sinks downward at supergranular boundaries. These downflows form the chromospheric magnetic network.

- **전형적 규모 / Typical scale**: ~30 Mm (태양 반경의 약 4%)
  / ~30 Mm (approximately 4% of the solar radius)

- **수명 / Lifetime**: ~1-2일
  / ~1-2 days

##### 5.3.2 깊이 구조 — 미해결 논쟁 / Depth Structure — Unresolved Debate

초과립의 깊이 구조는 2005년 시점에서 활발한 논쟁의 대상이었다:

The depth structure of supergranulation was subject to active debate as of 2005:

- **얕은 모델 (shallow model, <5 Mm)**: 일부 time-distance 연구에서 초과립 흐름이 매우 표면적인 현상이라는 결과를 보고했다. 이 관점에서 초과립은 표면 근처의 대류 과정에 의해 구동된다.
  / Some time-distance studies reported that supergranular flows are a very superficial phenomenon. In this view, supergranulation is driven by convective processes near the surface.

- **깊은 모델 (deep model, >10 Mm)**: 다른 연구에서는 초과립의 흐름 패턴이 10 Mm 이상의 깊이까지 연장된다는 결과를 보고했다. 이 관점에서 초과립은 더 깊은 대류 과정과 연결된다.
  / Other studies reported that supergranular flow patterns extend to depths exceeding 10 Mm. In this view, supergranulation is connected to deeper convective processes.

이 불일치의 원인은 역산 기법의 차이, 잡음 처리 방법, 사용된 커널의 정확도 등에 있을 수 있다. 이 문제의 해결은 태양 대류의 이해에 중요한 의미를 가진다.

The causes of this disagreement may lie in differences in inversion techniques, noise treatment methods, and accuracy of kernels used. Resolving this issue has important implications for understanding solar convection.

##### 5.3.3 회전 유도 와도 / Rotation-Induced Vorticity

매우 흥미로운 발견은 초과립에서의 **사이클론 패턴(cyclonic pattern)**의 관측이다: 북반구에서는 반시계 방향(counter-clockwise), 남반구에서는 시계 방향(clockwise)의 와도(vorticity)가 관측되었다. 이는 Coriolis 효과에 의한 것으로, 지구의 기상 시스템에서 관측되는 사이클론 회전 방향과 동일하다.

A very interesting discovery was the observation of a **cyclonic pattern** in supergranulation: counter-clockwise vorticity in the northern hemisphere and clockwise in the southern hemisphere. This is due to the Coriolis effect, the same cyclone rotation direction observed in Earth's weather systems.

이 관측은 국소 태양진동학만이 제공할 수 있는 독특한 과학적 결과이다. 전구 태양진동학이나 표면 관측만으로는 이러한 미세한 회전 패턴을 검출하기 어렵다.

This observation is a unique scientific result that only local helioseismology can provide. Such subtle rotation patterns are difficult to detect with global helioseismology or surface observations alone.

##### 5.3.4 패턴 진화와 파동적 성분 / Pattern Evolution and Wave-Like Component

초과립은 **순행 방향(prograde direction)**으로 이동하는 것처럼 보인다는 관측 결과가 있다. 이는 단순한 대류 세포의 이류(advection)로 설명되지 않으며, 초과립에 파동적 성분(wave-like component)이 있을 수 있음을 시사한다. 이 "파동적 초과립" 해석은 매우 논쟁적이며, 초과립의 본질이 대류인지 파동인지, 또는 그 혼합인지에 대한 질문을 제기한다.

Observations suggest supergranules appear to travel in the **prograde direction**. This is not explained by simple advection of convective cells and suggests there may be a wave-like component to supergranulation. This "wave-like supergranulation" interpretation is highly controversial, raising questions about whether the nature of supergranulation is convection, waves, or a mixture.

---

## 핵심 요점 / Key Takeaways

1. **국소 태양진동학의 독자성 / Uniqueness of local helioseismology**: 전구 태양진동학이 고유진동수만 사용하는 데 반해, 국소 태양진동학은 전체 파동장을 해석하여 3차원 정보를 추출한다. 이 차이가 흑점 아래 구조, 초과립, 뒷면 영상 등의 연구를 가능하게 한다.
   / While global helioseismology uses only eigenfrequencies, local helioseismology interprets the full wavefield to extract 3D information. This difference enables studies of subsurface sunspot structure, supergranulation, far-side imaging, etc.

2. **Born 근사의 중심적 역할 / Central role of the Born approximation**: $\delta d_i = \sum_\alpha \int K_\alpha^i \, \delta q_\alpha \, d^3r$ 이 근본 방정식은 모든 역문제의 기초이며, Born 근사의 정확도가 결과의 신뢰도를 결정한다.
   / This fundamental equation $\delta d_i = \sum_\alpha \int K_\alpha^i \, \delta q_\alpha \, d^3r$ underlies all inverse problems, and the accuracy of the Born approximation determines result reliability.

3. **바나나-도넛 커널의 발견 / Discovery of banana-doughnut kernels**: Born 근사로 도출된 travel time 감도 커널이 광선 경로를 따라 속이 빈 형태라는 발견은 직관에 반하지만 물리적으로 옳다. 이는 광선 근사의 한계를 명확히 보여준다.
   / The finding that Born-approximation travel-time sensitivity kernels are hollow along the ray path is counterintuitive but physically correct. This clearly shows the limitations of the ray approximation.

4. **5가지 기법의 상보성 / Complementarity of 5 techniques**: 각 기법은 고유한 장단점이 있으며, 독립적인 검증과 상호 보완이 가능하다. Time-distance는 높은 분해능, ring-diagram은 빠른 전구 지도, holography는 뒷면 영상에 강점을 지닌다.
   / Each technique has unique strengths and weaknesses, enabling independent verification and mutual complementation. Time-distance excels at high resolution, ring-diagram at rapid global mapping, holography at far-side imaging.

5. **표면 근처 전단과 비틀림 진동의 심층 확장 / Near-surface shear and deep extension of torsional oscillations**: 이 두 발견은 국소 태양진동학이 전구 태양진동학의 결과를 확인하고 확장하는 능력을 잘 보여준다.
   / These two discoveries well demonstrate local helioseismology's ability to confirm and extend the results of global helioseismology.

6. **자오면 흐름의 깊이 측정과 미해결 복귀 흐름 / Meridional flow depth measurement and unresolved return flow**: 표면 근처의 극 방향 흐름(10-20 m/s)은 잘 측정되지만, 질량 보존에 의해 반드시 존재해야 하는 적도 방향 복귀 흐름은 2005년 시점에서 미검출. 이는 다이나모 이론의 핵심 제약 조건이다.
   / The near-surface poleward flow (10-20 m/s) is well measured, but the equatorward return flow required by mass conservation was undetected as of 2005. This is a key constraint for dynamo theory.

7. **흑점에서의 Born 근사의 한계 / Limitation of Born approximation at sunspots**: 강한 자기장 영역에서 모드 변환과 비선형 파동-자기장 상호작용으로 인해 Born 근사가 실패할 수 있다. 이는 흑점 심층 구조 연구의 근본적 도전이다.
   / Mode conversion and nonlinear wave-magnetic field interactions in strong magnetic field regions can cause the Born approximation to fail. This is a fundamental challenge for deep sunspot structure studies.

8. **우주 기상 예보에의 기여 / Contribution to space weather forecasting**: 뒷면 영상은 지구로 회전해 올 활동 영역을 2주 전에 미리 감지할 수 있는 실용적 도구이며, 국소 태양진동학의 가장 직접적인 사회적 응용이다.
   / Far-side imaging is a practical tool that can detect active regions 2 weeks before they rotate toward Earth, the most direct societal application of local helioseismology.

---

## 수학적 요약 / Mathematical Summary

| 기호 / Symbol | 의미 / Meaning | 정의/관계 / Definition/Relation |
|---|---|---|
| $\omega_{nlm}$ | 고유 진동수 / Eigenfrequency | $\delta\mathbf{r} = \sum a_{nlm} \boldsymbol{\xi}_{nl} Y_l^m e^{i\omega_{nlm}t}$ |
| $\omega^2 = gk$ | f-mode 분산 관계 / f-mode dispersion | 표면 중력파 / Surface gravity wave |
| $\mathcal{L}\boldsymbol{\xi} = \mathbf{S}$ | 파동 방정식 / Wave equation | $\mathcal{L}$: 파동 연산자, $\mathbf{S}$: 확률적 소스 |
| $G_j^i$ | Green 함수 / Green's function | 충격 응답 / Impulse response |
| $P(\mathbf{k},\omega)$ | 파워 스펙트럼 / Power spectrum | $(2\pi)^3/(AT) \cdot \mathbb{E}[|\Phi|^2]$ |
| $\delta d_i = \sum_\alpha \int K_\alpha^i \delta q_\alpha \, d^3r$ | **Born 근사 근본 방정식** / Born approx. fundamental eq. | 역문제의 기초 / Foundation of inverse problem |
| $\alpha_m = 1 - |B_m|^2/|A_m|^2$ | 흡수 계수 / Absorption coefficient | Fourier-Hankel 기법 |
| $\tau_{\text{diff}} = \tau_+ - \tau_-$ | Travel time 차이 / Travel time difference | 흐름에 민감 / Sensitive to flows |
| $\tau_{\text{mean}} = (\tau_+ + \tau_-)/2$ | Travel time 평균 / Travel time mean | 음속에 민감 / Sensitive to sound speed |
| $C(\mathbf{x}_1,\mathbf{x}_2,t)$ | 교차 공분산 / Cross-covariance | 태양 지진파 / Solar seismogram |
| $H_\pm^P$ | Egression/Ingression | Holographic 역전파 추정 / Holographic backpropagation estimate |

---

## 역사적 타임라인 / Historical Timeline

```
1960s ─── 전구 태양진동학의 시작 / Global helioseismology begins
  │
1987 ─── Braun et al.: Fourier-Hankel 방법으로 흑점의 p-mode 흡수 발견
  │       / p-mode absorption by sunspots discovered
  │
1988 ─── Hill: Ring-diagram 분석법 도입
  │       / Ring-diagram analysis introduced
  │
1990 ─── Lindsey & Braun: 태양진동 holography 제안
  │       / Helioseismic holography proposed
  │
1993 ─── Duvall et al.: Time-distance 태양진동학 도입
  │       / Time-distance helioseismology introduced
  │       Lindsey et al.: "국소 태양진동학" 용어 처음 사용
  │       / Term "local helioseismology" first used
  │
1994 ─── Goldreich et al.: 확률적 여기 이론 정립
  │       / Stochastic excitation theory established
  │
1996 ─── MDI/SOHO 발사 → 고품질 연속 데이터 시대 개막
  │       / MDI/SOHO launched → era of high-quality continuous data
  │
2000 ─── Lindsey & Braun: 뒷면 영상 최초 시연
  │       / First far-side imaging demonstration
  │
2002 ─── Woodard: 직접 모델링 기법 도입
  │       / Direct modeling technique introduced
  │
2004 ─── Gizon & Birch: Travel time의 체계적 정의
  │       / Systematic travel time definition
  │       바나나-도넛 커널 도입 / Banana-doughnut kernels introduced
  │
2005 ─── 이 리뷰 출판 / This review published
  │       Stanford에서 뒷면 영상 일상 운영 시작
  │       / Far-side imaging operational at Stanford
```

---

## 다른 주제와의 연결 / Connections to Other Topics

| 관련 분야 / Related Field | 연결 / Connection |
|---|---|
| 전구 태양진동학 / Global helioseismology | 국소 태양진동학의 전신이자 상호 검증 대상. 차등 회전 역산 결과의 일치가 두 분야의 신뢰성을 보증한다. / Predecessor and mutual validation target. Agreement on differential rotation inversions ensures reliability of both fields. |
| 태양 다이나모 이론 / Solar dynamo theory | 자오면 흐름, 비틀림 진동, 차등 회전의 깊이별 구조가 다이나모 모델의 핵심 입력/제약 조건이다. / Depth structures of meridional flow, torsional oscillations, and differential rotation are core inputs/constraints for dynamo models. |
| Flux-transport 모델 / Flux-transport model | 자오면 흐름의 측정이 flux-transport 모델의 핵심 매개변수를 결정한다 (Sheeley 2005, LRSP 2, 5). / Measurement of meridional flow determines key parameters of flux-transport models (Sheeley 2005, LRSP 2, 5). |
| 우주 기상 / Space weather | 뒷면 영상이 지구로 회전해 올 활동 영역을 미리 감지하여 예보에 기여한다. / Far-side imaging detects active regions rotating toward Earth for early warning. |
| 태양 대류 / Solar convection | 초과립의 깊이 구조, 대류층 내 흐름 패턴이 대류 이론의 관측적 제약이다. / Depth structure of supergranulation, flow patterns in the convection zone are observational constraints for convection theory. |
| 지진학 / Seismology | 바나나-도넛 커널, Born 근사, 역문제 기법이 지구 지진학에서 차용/공유된다. / Banana-doughnut kernels, Born approximation, inverse problem techniques are borrowed from/shared with terrestrial seismology. |
| MHD 파동 이론 / MHD wave theory | 흑점 내부의 모드 변환(p-mode → Alfvén, magneto-acoustic)은 MHD 파동 이론의 직접적 응용이다 (Nakariakov & Verwichte 2005, LRSP 2, 3). / Mode conversion inside sunspots is a direct application of MHD wave theory (Nakariakov & Verwichte 2005, LRSP 2, 3). |
| 태양 자기장 관측 / Solar magnetic field observations | 활동 영역의 자기적 특성과 국소 태양진동학으로 측정된 유동 구조의 상관관계가 자기-대류 상호작용의 이해에 기여한다. / Correlations between magnetic properties of active regions and flow structures measured by local helioseismology contribute to understanding magneto-convective interactions. |

---

## 참고 문헌 / References

- Gizon, L. and Birch, A.C., "Local Helioseismology", Living Reviews in Solar Physics, 2, 6 (2005). [DOI: 10.12942/lrsp-2005-6]
- Braun, D.C., Duvall, T.L., and LaBonte, B.J., "Acoustic absorption by sunspots", Astrophys. J., 319, L27-L31 (1987).
- Hill, F., "Rings and Trumpets — Three-Dimensional Power Spectra of Solar Oscillations", Astrophys. J., 333, 996-1013 (1988).
- Lindsey, C. and Braun, D.C., "Helioseismic imaging of sunspots at their antipodes", Solar Phys., 126, 101-115 (1990).
- Duvall, T.L., Jefferies, S.M., Harvey, J.W., and Pomerantz, M.A., "Time-distance helioseismology", Nature, 362, 430-432 (1993).
- Lindsey, C., Braun, D.C., Jefferies, S.M., Woodard, M.F., Fan, Y., Gu, Y., and Redfield, S., "Local Helioseismology of subsurface structure", in GONG 1992, (Ed.) Brown, T., ASP Conference Series, 42, 81-84 (1993).
- Goldreich, P., Murray, N., and Kumar, P., "Excitation of solar p-modes", Astrophys. J., 424, 466-479 (1994).
- Woodard, M.F., "Solar Subsurface Flow Assessed by a Large Correlation Ensemble", Astrophys. J., 565, 634-639 (2002).
- Gizon, L. and Birch, A.C., "Time-Distance Helioseismology: Noise Estimation", Astrophys. J., 614, 472-489 (2004).
- Christensen-Dalsgaard, J., "Helioseismology", Rev. Mod. Phys., 74, 1073-1129 (2002).
