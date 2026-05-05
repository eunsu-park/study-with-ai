---
title: "Solar Interior Rotation and its Variation"
authors: Rachel Howe
year: 2009
journal: "Living Reviews in Solar Physics"
doi: "10.12942/lrsp-2009-1"
topic: Living_Reviews_in_Solar_Physics
tags: [helioseismology, solar rotation, tachocline, torsional oscillation, differential rotation, inversion, convection zone]
status: completed
date_started: 2026-04-15
date_completed: 2026-04-15
---

# 15. Solar Interior Rotation and its Variation / 태양 내부 자전과 그 변동

---

## 1. Core Contribution / 핵심 기여

This review provides the most comprehensive observational account of the Sun's internal rotation as revealed by helioseismology over approximately forty years. Starting from the 1960s oblateness controversy — where Dicke's measurements of the Sun's shape raised questions about core rotation and tests of General Relativity — the article traces the development of helioseismic techniques through the golden age of solar cycle 23, when GONG and SOHO/MDI provided essentially uninterrupted data. The review establishes the modern picture of solar interior rotation: (1) approximately rigid rotation of the radiative interior at ~430 nHz, (2) a thin tachocline shear layer at ~0.69 $R_\odot$ connecting the rigid interior to the differentially rotating convection zone, (3) differential rotation throughout the convection zone with constant-rotation contours tilted at ~25° to the rotation axis (not cylindrical as predicted by Taylor-Proudman), (4) a near-surface shear layer above ~0.95 $R_\odot$, (5) the torsional oscillation — migrating bands of faster/slower zonal flow tracking the solar cycle throughout the convection zone depth, and (6) possible temporal variations near the tachocline including a controversial 1.3-year signal. For each feature, Howe carefully examines the relationship between observations and theoretical models, while maintaining the perspective of an observer and acknowledging systematic uncertainties.

이 리뷰는 일진학이 약 40년에 걸쳐 밝혀낸 태양 내부 자전의 관측적 이해를 가장 포괄적으로 정리한 논문입니다. 1960년대 Dicke의 편평도 측정이 촉발한 핵 자전 논쟁에서 출발하여, 태양 주기 23 동안 GONG과 SOHO/MDI가 제공한 연속 데이터로 확립된 현대적 그림까지를 추적합니다. 핵심 발견은 다음과 같습니다: (1) 복사 내부의 강체 자전(~430 nHz), (2) ~0.69 $R_\odot$의 얇은 tachocline 전단층, (3) 대류층의 차등 자전(등자전 윤곽이 ~25° 기울어진 형태), (4) ~0.95 $R_\odot$ 이상의 표면 근처 전단층, (5) 태양 주기에 연동하여 대류층 전체 깊이로 침투하는 비틀림 진동, (6) 논쟁 중인 tachocline 1.3년 신호. 각 구조에 대해 관측과 이론 모델의 관계를 관측자 관점에서 신중하게 검토하며 체계적 불확실성을 인정합니다.

---

## 2. Reading Notes / 읽기 노트

### Section 1: Introduction / 도입

The Sun rotates approximately once every 27 days, but this rotation is non-uniform: the equator rotates substantially faster than the poles. This surface differential rotation was known from sunspot tracking since the 17th century. However, only within the last ~30 years (before 2009) has it been possible to observe the rotation profile in the solar interior via helioseismology.

태양은 약 27일 주기로 자전하지만, 이 자전은 균일하지 않습니다. 적도가 극보다 상당히 빠르게 회전합니다. 이 표면 차등 자전은 17세기부터 흑점 추적으로 알려져 있었으나, 내부 자전 프로파일의 관측은 일진학을 통해서만 가능해졌습니다.

Howe identifies four major features of the time-invariant rotation profile (Figure 1):
1. **Radiative interior and core**: approximately solid-body rotation, though the innermost core may differ
2. **Tachocline**: thin shear zone between the differentially rotating convection zone and rigid radiative interior — believed crucial for the solar dynamo
3. **Differential rotation in the bulk of the convection zone**: latitude-dependent rotation
4. **Subsurface shear layer**: between the fastest-rotating layer at ~0.95 $R_\odot$ and the surface

시간-불변 자전 프로파일의 4가지 주요 구조: (1) 복사 내부/핵의 강체 자전, (2) 얇은 tachocline 전단층, (3) 대류층의 차등 자전, (4) 표면 근처 전단층.

The review also covers the time-varying part: the torsional oscillation (Section 9) and possible variations at the base of the convection zone (Section 10).

### Section 2: Acoustic Modes / 음향 모드

#### 2.1 Introduction / 도입

Helioseismic data consist of measurements of photospheric Doppler velocity or intensity, taken at ~1-minute cadence over months to years. The velocity variations at the solar surface originate in acoustic modes (p-modes) propagating in a cavity bounded above by the solar surface and below by a wavelength-dependent depth. These modes are classified by:
- **Radial order $n$**: number of radial nodes
- **Spherical harmonic degree $l$**: total number of nodal lines on the surface
- **Azimuthal order $m$**: number of nodal lines crossing the equator ($-l \le m \le l$)

일진학의 원시 데이터는 광구 Doppler 속도 또는 강도 측정으로, 약 1분 간격으로 수 개월~수 년간 수집됩니다. 모드는 세 양자수 $(n, l, m)$으로 분류됩니다.

The radial displacement of a fluid element is expressed as:

$$\delta r(r,\theta,\phi,t) = \sum_{m=-l}^{l} a_{nlm} \xi_{nl}(r) Y_l^m(\theta,\phi) e^{i\omega_{nlm} t} \tag{1}$$

**Mode penetration depth** is a key concept: the lower turning point radius $r_t$ is a monotonic function of the phase speed $\nu/L$, where $L = \sqrt{l(l+1)} \approx l + 1/2$. Lower-degree modes penetrate deeper — $l = 0$ reaches the core (but gives no rotational information), while $l \ge 200$ probes only a few Mm below the surface. This varying penetration depth makes it possible to deduce the rotation as a function of depth (Figure 5, 6).

**모드 침투 깊이**: 하한 전환점 반경 $r_t$는 위상 속도 $\nu/L$의 단조 함수. 낮은 $l$ 모드가 더 깊이 침투하여 깊이별 자전 정보를 추출할 수 있게 합니다.

Power in the modes peaks at ~3 mHz (5-minute period); useful measurements span 1.5-5 mHz.

#### 2.2 Differential Rotation and Rotational Splitting / 차등 자전과 자전 분리

Solar rotation lifts the degeneracy between modes of the same $l$ and different $m$, resulting in "rotational splitting":

$$\delta\nu_{m,l} \equiv \nu_{-m,l} - \nu_{+m,l}$$

To first order, the splitting is proportional to the rotation rate multiplied by $m$. Because modes of different $m$ sample different latitude ranges — sectoral ($|m| = l$) modes are confined near the equator, zonal ($m = 0$) modes reach the poles — we can measure rotation as a function of latitude (Figure 8).

자전 분리 $\delta\nu_{m,l}$는 1차 근사에서 자전율 $\times m$에 비례합니다. 다른 $m$ 값을 가진 모드가 다른 위도 범위를 탐사하므로, 위도별 자전을 측정할 수 있습니다.

Rather than fitting all $2l+1$ individual frequencies, the $m$-dependence is commonly expressed as a polynomial expansion:

$$\nu_{nlm} = \nu_{nl} + \sum_{j=1}^{j_{\max}} a_j(n,l) \mathcal{P}_j^{(l)}(m) \tag{2}$$

where $\mathcal{P}_j^{(l)}(m)$ are polynomials related to Clebsch-Gordan coefficients. **Odd-order coefficients** ($a_1, a_3, a_5, \ldots$) encode the rotational asymmetry; **even-order** encode structural asphericity. Roughly:
- $a_1$: average rotation rate over all latitudes
- $a_3$ and higher: differential rotation

홀수 차수 계수는 자전 비대칭, 짝수 차수 계수는 구조 비구대칭 정보를 담고 있습니다. $a_1$은 전체 위도 평균 자전율, $a_3$ 이상은 차등 자전을 기술합니다.

#### 2.3 Spherical Harmonics and Leakage / 구면 조화와 누설

Spherical harmonic masks separate contributions from modes of different degree and azimuthal order. However, because only part of the solar disk is visible, the masks are not perfectly orthogonal and modes "leak" into neighboring spectra. The leakage matrix element is:

$$s(l,m,l',m') = \frac{1}{\pi} \int_{-1}^{1} \int_{-\pi/2}^{\pi/2} P_l^{|m|}(x) P_{l'}^{|m'|}(x) \cos(m\phi) \cos(m'\phi) V(\rho) A(\rho) dx d\phi \tag{6}$$

This leakage is especially problematic for $l = 1$ splittings (from $m$-leaks at the $\delta l = 0, \delta m = \pm 2$ component) and for high-degree modes where adjacent-$l$ ridges overlap.

구면 조화 마스크의 불완전한 직교성으로 모드 간 "누설"이 발생하며, 이는 특히 $l = 1$ splitting과 고차수 모드에서 문제가 됩니다.

#### 2.4 Estimating Rotation from Coefficients / 계수에서 자전 추정

Without full inversion, approximate expressions relate splitting coefficients to rotation at specific latitudes:

$$\Omega_0^{nl} \approx a_1^{nl} + a_3^{nl} + a_5^{nl} \tag{7}$$
$$\Omega_{30}^{nl} \approx a_1^{nl} - \frac{a_3^{nl}}{4} - \frac{19a_5^{nl}}{16} \tag{8}$$
$$\Omega_{45}^{nl} \approx a_1^{nl} - \frac{3a_5^{nl}}{2} - \frac{3a_5^{nl}}{4} \tag{9}$$
$$\Omega_{60}^{nl} \approx a_1^{nl} - \frac{11a_3^{nl}}{4} + \frac{37a_5^{nl}}{16} \tag{10}$$

(subscripts indicate latitude in degrees). Sorted by turning-point radius (i.e., $\nu/L$), these reveal the near-surface shear, convection zone differential rotation, and the transition at the tachocline (Figure 9).

역산 없이도 splitting 계수에서 특정 위도의 자전을 근사적으로 추정할 수 있습니다. 전환점 반경(즉 $\nu/L$)으로 정렬하면 표면 전단, 대류층 차등 자전, tachocline 전이가 드러납니다.

### Section 3: Inversion Basics / 역산 기초

#### 3.1 The Inversion Problem / 역산 문제

The basic 2-dimensional rotation inversion can be stated as: given $M$ observations $d_i$, infer the rotation profile $\Omega(r, \theta)$. Each datum is a spatially weighted average:

$$d_i = \int_0^{R_\odot} \int_0^{\pi} K_i(r,\theta) \Omega(r,\theta) \, dr \, d\theta + \epsilon_i \tag{11}$$

where $K_i$ is the **kernel** — a model-dependent spatial weighting function. For the 2D rotation inversion, the kernel expression (Schou et al. 1994) is:

$$K_{nlm}(r,\theta) = \frac{m}{I_{nl}} \left\{ \xi_{nl}(r)\left[\xi_{nl}(r) - \frac{2}{L}\eta_{nl}(r)\right] P_l^m(x)^2 + \frac{\eta_{nl}(r)^2}{L^2}\left[\left(\frac{dP_l^m}{dx}\right)^2 (1-x^2) - 2P_l^m \frac{dP_l^m}{dx} x + \frac{m^2}{1-x^2} P_l^m(x)^2\right] \right\} \rho(r) r \sin\theta \tag{12}$$

where $I_{nl} = \int_0^{R_\odot} [\xi_{nl}(r)^2 + \eta_{nl}(r)^2] \rho(r) r^2 dr$ (Eq. 13), $\xi_{nl}$ is radial displacement, $L^{-1}\eta_{nl}$ is horizontal displacement, and $\rho(r)$ is density.

2차원 자전 역산 문제: $M$개의 관측 $d_i$로부터 자전 프로파일 $\Omega(r,\theta)$를 추론합니다. 각 관측값은 커널 $K_i$로 가중된 자전 프로파일의 공간 적분입니다. 커널은 모드의 고유함수와 밀도 프로파일에서 결정됩니다.

The inversion aims to find:

$$\bar{\Omega}(r_0,\theta_0) = \sum_{i=1}^{M} c_i(r_0,\theta_0) d_i \tag{14}$$

The choice of coefficients $c_i$ defines the inversion method.

#### 3.2 Averaging Kernels / 평균 커널

Substituting Eq. (11) into Eq. (14):

$$\bar{\Omega}(r_0,\theta_0) = \int_0^{R_\odot} \int_0^{\pi} \mathcal{K}(r_0,\theta_0;r,\theta) \Omega(r,\theta) \, dr \, d\theta + \epsilon_i \tag{15}$$

where the **averaging kernel** is:

$$\mathcal{K}(r_0,\theta_0;r,\theta) \equiv \sum_{i=1}^{M} c_i(r_0,\theta_0) K_i(r,\theta) \tag{16}$$

The averaging kernel is independent of the data values but depends on which modes are available. It is a powerful tool for assessing the reliability of an inversion inference.

**평균 커널** $\mathcal{K}$는 추론된 자전율이 실제 자전 프로파일의 어떤 가중 평균인지를 보여줍니다. 데이터 값에 무관하지만 사용 가능한 모드에 의존하며, 역산 결과의 신뢰성을 평가하는 핵심 도구입니다.

#### 3.3 Inversion Errors / 역산 오차

The formal uncertainty on the inferred rotation rate:

$$\sigma^2[\Omega(r_0,\theta_0)] = \sum_i [c_i(r_0,\theta_0) \sigma_i]^2 \tag{17}$$

The "error magnification" for uniform errors:

$$\Lambda(r_0,\theta_0) = \left[\sum_i c_i(r_0,\theta_0)^2\right]^{1/2} \tag{19}$$

Errors at different points are correlated even when input errors are uncorrelated (Eq. 20). The finite width of averaging kernels introduces systematic errors — e.g., a thin shear layer may be under/overestimated on either side.

역산 오차의 형식적 불확실성(Eq. 17)과 오차 증폭(Eq. 19). 입력 오차가 비상관이더라도 추론된 프로파일의 오차는 상관될 수 있습니다. 유한 폭의 평균 커널은 체계적 오차를 야기합니다.

#### 3.4 Regularized Least Squares (RLS) / 정규화 최소제곱법

RLS finds the model profile that best fits the data, subject to a smoothness penalty:

$$\sum_i \frac{[d_i - \int_0^R \int_0^{\pi} \Omega(r,\theta) K_i(r,\theta) dr d\theta]^2}{(\sigma_i/\bar{\sigma})^2} + \mu_r^2 \int_0^R \int_0^{\pi} \left(\frac{\partial^2 \Omega}{\partial r^2}\right)^2 dr d\theta + \mu_\theta^2 \int_0^R \int_0^{\pi} \left(\frac{\partial^2 \Omega}{\partial \theta^2}\right)^2 dr d\theta \tag{21}$$

where $\mu_r$ and $\mu_\theta$ are radial and latitudinal tradeoff parameters. **Advantages**: computationally inexpensive, always provides an estimate everywhere. **Disadvantage**: averaging kernels are not guaranteed to be well-localized (Figures 11).

RLS는 데이터 적합도와 매끄러움 사이의 균형을 찾습니다. 계산이 빠르고 모든 위치에서 추정이 가능하지만, 평균 커널의 국소화가 보장되지 않습니다.

#### 3.5 Optimally Localized Averaging (OLA/SOLA) / 최적 국소 평균법

The SOLA approach minimizes the difference between the averaging kernel and a target kernel $\mathcal{T}$ (e.g., Gaussian):

$$\int_0^R \int_0^{\pi} [\mathcal{T}(r_0,\theta_0;r,\theta) - \mathcal{K}(r_0,\theta_0;r,\theta)]^2 \, r \, dr \, d\theta + \lambda \sum_{i=1}^{M} [\sigma_i c_i(r_0,\theta_0)]^2 \tag{22}$$

The tradeoff parameter $\lambda$ balances resolution against error amplification. **Advantage**: clearer interpretation — the averaging kernel tells you exactly what you're measuring (Figure 12). **Disadvantage**: computationally more expensive; target kernel must be appropriately chosen.

SOLA는 평균 커널을 목표 함수(Gaussian 등)에 최대한 가깝게 만듭니다. 해석이 더 명확하지만 계산 비용이 높습니다. $\lambda$가 해상도와 오차 증폭 사이의 균형을 제어합니다.

#### 3.7 Limitations / 한계

Key limitations of the inversion process:
- Resolution limited by the deepest and shallowest turning-point radii of available modes
- Latitudinal resolution is poor at high latitudes (small $|m|/l$ modes have few nodes near equator)
- Below the convection zone, only low-degree ($l \le 20$) modes penetrate → poor latitudinal and radial resolution
- Inversions measure only the **north-south symmetric** part; hemisphere asymmetry is averaged out
- Inversions are insensitive to **meridional motions**

역산의 핵심 한계: 고위도에서 위도 해상도가 떨어지고, 대류층 아래에서는 저차수 모드만 도달하여 해상도가 나빠지며, 남북 대칭 부분만 측정 가능하고, 자오면 운동에 둔감합니다.

### Section 4: Observations — A Brief Historical Overview / 관측의 역사적 개요

Systematic helioseismic observations span nearly 30 years (by 2009). Key milestones (Figure 13):

- **1976-1978**: Early South Pole summer observations
- **1979**: Birmingham group identifies global low-degree modes (Claverie et al.)
- **1981**: First widely-separated multi-site observations (Birmingham/Tenerife)
- **1986**: BBSO 100-day observations (Libbrecht)
- **1989-2003**: French-based IRIS network
- **1992**: Six-station BiSON network complete
- **1993-1996**: Taiwanese Oscillations Network (TON)
- **1994-2004**: LOWL-ECHO project
- **1995**: GONG network begins — 6-station worldwide network for medium-degree p-modes
- **1996**: SOHO spacecraft (MDI, LOI, GOLF) begins operation

Together, GONG and SOHO/MDI provided essentially complete coverage of solar cycle 23 — an unprecedented dataset for studying solar interior rotation and its solar-cycle changes.

GONG(1995)과 SOHO/MDI(1996)가 태양 주기 23 전체를 아우르는 전례 없는 데이터셋을 제공했습니다.

### Section 5: The Core and Radiative Interior / 핵과 복사 내부

#### 5.1 The Oblateness Controversy / 편평도 논쟁

Interest in the deep solar interior rotation predates helioseismology. The solar oblateness $\Delta r$ is related to the quadrupole moment $J_2$:

$$\frac{\Delta r}{r_0} = \frac{3}{2} J_2 \tag{23}$$

Dicke (1964) recognized that fast internal rotation could affect the Sun's gravitational potential, potentially destroying the agreement between GR predictions and Mercury's perihelion precession. Dicke & Goldenberg (1967) reported $J_2 = 5 \times 10^{-5}$, implying a fast-rotating core, sparking intense controversy.

Dicke의 편평도 측정($J_2 = 5 \times 10^{-5}$)은 빠르게 자전하는 핵을 암시했고, 이는 수성 근일점 세차와 일반 상대성 이론 검증에 영향을 미쳐 큰 논쟁을 불러일으켰습니다.

Subsequent measurements by Hill & Stebbins (1975) found $J_2 = 9.6 \times 10^{-6}$, much smaller. By the 1990s, Dicke et al. (1986, 1987) repeated measurements found significantly smaller values with some solar-cycle variation. Modern values: $J_2 \sim 1.8 \times 10^{-7}$, $J_4 \sim 9.8 \times 10^{-7}$ (Lydon & Sofia, 1996). The focus shifted from oblateness to helioseismology.

이후 측정들은 훨씬 작은 $J_2$ 값을 보고했고, 초점은 편평도에서 일진학으로 이동했습니다.

#### 5.2-5.7 Low-degree Helioseismic Results / 저차수 일진학 결과

The $l = 1$ rotational splitting has been the primary probe of core rotation. Table 1 summarizes measurements from 1988-2002, showing convergence from initial values of ~0.75 $\mu$Hz (Tenerife, 1988) toward ~0.43 $\mu$Hz (GOLF, 2002) — a value consistent with either rigid rotation or a slight downturn in the core.

$l = 1$ 자전 분리는 핵 자전의 주요 탐침입니다. 1988-2002년 측정값은 초기 ~0.75 $\mu$Hz에서 ~0.43 $\mu$Hz로 수렴하며, 이는 강체 자전 또는 핵에서의 약간의 감소와 일치합니다.

Key challenges with low-degree splittings:
1. The $l = 1$ doublet components are extremely close (<1 $\mu$Hz apart), resolved only below ~2.2 mHz
2. Low-degree modes spend most time in outer layers → not very sensitive to core
3. Combining low-degree with medium-degree data introduces cross-instrument systematics

저차수 splitting의 핵심 어려움: $l=1$ 이중선이 극히 가까워 2.2 mHz 이하에서만 분해 가능, 모드가 대부분의 시간을 외층에서 보내 핵에 둔감, 다른 기기 데이터 결합 시 체계적 오차 발생.

#### 5.8-5.9 Summary and Gravity Modes / 요약 및 중력 모드

**Best evidence**: Rotation between ~0.2 $R_\odot$ and tachocline is approximately constant with radius and spherically symmetric. No evidence from p-modes for a different inner core rotation rate, but it cannot be ruled out.

**$g$-modes**: Would be much more sensitive to core rotation (amplitudes peak in the interior), but surface amplitudes are extremely small. Garcia et al. (2007) report a possible detection corresponding to core rotation 3-5 times surface rate, but this remains unconfirmed. Upper limit on amplitudes: ~6 mm/s (Gabriel et al. 2002).

최적 증거: ~0.2 $R_\odot$에서 tachocline까지 자전율이 반경에 대해 거의 일정하고 구대칭. 가장 안쪽 핵의 자전은 $p$-mode로는 결정 불가. $g$-mode 검출 시도는 아직 미확인 상태.

### Section 6: The Tachocline / 타코클라인

#### 6.1 Observations / 관측

The tachocline is the thin transition layer between the differentially rotating convection zone and the rigidly rotating radiative interior, located at ~0.69 $R_\odot$. The term was introduced by Spiegel & Zahn (1992), crediting D.O. Gough's correction of the earlier "tachycline" (Spiegel, 1972).

타코클라인은 대류층의 차등 자전과 복사 내부의 강체 자전 사이의 얇은 전이층으로, ~0.69 $R_\odot$에 위치합니다.

Key observational parameters from Table 2:

| Reference | $r/R_\odot$ | $\Gamma/R_\odot$ (width) | Project |
|-----------|-------------|--------------------------|---------|
| Kosovichev (1996) | 0.692 | 0.09 | BBSO |
| Basu (1997) | 0.705 | 0.048 | GONG |
| Antia et al. (1998) | 0.6947 | 0.033 | GONG |
| Corbard et al. (1999) | 0.691 | 0.01 | LOWL |
| Charbonneau et al. (1999) | 0.693 | 0.039 | LOWL |
| Elliott & Gough (1999) | 0.697 | 0.019 | MDI |
| Basu & Antia (2003) | 0.6916 | 0.0162 | MDI, GONG |

**Consensus**: centroid slightly below the seismically-determined base of the convection zone (0.713 $R_\odot$), width ~0.02-0.05 $R_\odot$. Charbonneau et al. (1999) found it slightly prolate (shallower at latitude 60° by ~0.024 $R_\odot$); Basu & Antia (2003) found slightly thicker and shallower at high latitudes.

**합의**: 중심은 대류층 하단(0.713 $R_\odot$)보다 약간 아래, 두께 ~0.02-0.05 $R_\odot$. 고위도에서 약간 두껍고 얕을 수 있음.

The discovery of the tachocline (Brown et al. 1989) solved the puzzle of the apparent absence of a radial rotation gradient in the convection zone — the dynamo must operate in the tachocline rather than throughout the convection zone.

tachocline의 발견은 대류층에서 방사상 자전 기울기가 없는 수수께끼를 풀었습니다 — 다이나모가 대류층 전체가 아닌 tachocline에서 작동해야 함을 시사합니다.

#### 6.2 Models and the Tachocline / 모델

Three candidate mechanisms for confining the tachocline:
1. **Turbulent flows** (Spiegel & Zahn, 1992)
2. **"Fossil" magnetic fields** (Gough & McIntyre, 1998)
3. **Gravity waves** ($g$-modes) (Zahn et al., 1997)

All have problems as sole mechanisms. 3D simulations have not yet reproduced a self-sustaining tachocline.

tachocline 유지 메커니즘 후보: 난류, 화석 자기장, 중력파. 모두 단독 메커니즘으로는 문제가 있으며, 3D 시뮬레이션으로 자기 유지 tachocline이 아직 재현되지 못했습니다.

### Section 7: Rotation in the Bulk of the Convection Zone / 대류층 내부의 자전

#### 7.1 Observational History / 관측 역사

Pre-helioseismology models predicted rotation constant on cylinders (Taylor-Proudman constraint). Early observations (Duvall & Harvey, 1984; Brown, 1985; Brown & Morrow, 1987) from the South Pole and Fourier Tachometer hinted at little radial differential rotation in the convection zone and little differential rotation below.

일진학 이전 모델은 원통형 등자전(Taylor-Proudman 제약)을 예측했으나, 초기 관측들은 대류층 내 반경 방향 차등 자전이 거의 없음을 시사했습니다.

The landmark result came from Schou et al. (1998), who analyzed the first 144 days of MDI data using four different inversion techniques (2dRLS, 2dSOLA, 1d×1dSOLA, 1.5dRLS — Figure 18). **Key findings**:
- Consistent and robust results from surface to ~0.5 $R_\odot$ at low latitudes
- Rotation in the bulk, below 0.95 $R_\odot$, increases slowly with radius at most latitudes
- **Definitively incompatible with rotation on cylinders**

Schou et al. (1998)의 MDI 144일 분석이 획기적 결과: 대류층 내 자전이 원통형이 아님을 확정적으로 보여주었습니다.

#### 7.4 Slanted Contours / 기울어진 등자전 윤곽

Gilman & Howe (2003) and Howe et al. (2005) showed that the contours of constant rotation in the convection zone are tilted at about **25° to the rotation axis** (Figure 19). This is intermediate between cylindrical (0°) and radial (90°) configurations (Figure 20). This discovery provides a key constraint for convection zone models.

대류층의 등자전 윤곽이 자전축에 대해 약 **25° 기울어져** 있음이 발견되었습니다. 이는 원통형(0°)과 반경 방향(90°) 사이의 중간값으로, 대류층 모델의 핵심 제약 조건입니다.

#### 7.5 Polar Rotation / 극 자전

An interesting feature from early GONG/MDI observations: while the surface rotation rate is well-described by the three-term fit $\Omega(\theta) = A + B\cos^2\theta + C\cos^4\theta$, the rotation rate close to the poles is significantly slower than predicted. This may be due to drag from the solar wind. The inferred high-latitude rate does speed up during solar maximum.

극지방의 자전율은 표면 3항 적합보다 상당히 느립니다. 태양풍에 의한 항력 때문일 수 있으며, 태양 극대기에 속도가 빨라집니다.

#### 7.6 Models / 모델

Early dynamo models (Glatzmaier, 1985; Gilman & Miller, 1986) produced rotation on cylinders, inconsistent with observations. Modern 3D simulations (Miesch et al. 2008 — Figure 21) based on giant convection cells, after temporal averaging, produce rotation patterns that look quite solar-like, with slanted contours and differential rotation.

초기 다이나모 모델은 원통형 자전을 예측했지만, 최신 3D 시뮬레이션(Miesch et al. 2008)은 시간 평균 후 기울어진 등자전 윤곽과 차등 자전을 포함한 태양과 유사한 패턴을 재현합니다.

### Section 8: The Near-Surface Shear / 표면 근처 전단

A persistent puzzle: Doppler measurements give slower rotation than magnetic feature tracking:

$$\frac{\Omega_m}{2\pi} = 462 - 74\mu^2 - 53\mu^4 \text{ nHz (magnetic features)} \tag{25}$$
$$\frac{\Omega_p}{2\pi} = 452 - 49\mu^2 - 84\mu^4 \text{ nHz (plasma)} \tag{26}$$

This discrepancy arises because magnetic tracers (sunspots) are anchored in a faster-rotating layer deeper down. Rhodes et al. (1979) first detected the subsurface shear using high-degree modes probing the upper 20 Mm.

자기 추적자(흑점)가 더 빠르게 회전하는 하층에 고정되어 있기 때문에, Doppler 측정이 자기장 추적보다 느린 자전을 보여줍니다. Rhodes et al. (1979)이 고차수 모드로 표면 아래 전단을 최초로 감지했습니다.

With GONG and MDI, Schou et al. (1998) found clear evidence of near-surface shear in global inversions. The near-surface shear is also accessible to local helioseismology (ring-diagram analysis) — Howe et al. (2006a) found good agreement between local and global results at latitudes $\le 30°$ (Figure 22).

Corbard & Thompson (2002) studied the shear in detail using $f$-modes: slope close to $-400$ nHz/$R_\odot$ at low latitudes, decreasing to near zero at ~30° and possibly reversing sign at higher latitudes.

Corbard & Thompson (2002): 저위도에서 기울기 약 $-400$ nHz/$R_\odot$, ~30°에서 거의 0으로 감소, 고위도에서 부호 반전 가능.

### Section 9: The Torsional Oscillation / 비틀림 진동

#### 9.1 Discovery (Before Helioseismology) / 발견

Howard & LaBonte (1980) first described the torsional oscillation using 12 years (1966-1978) of full-disk Doppler observations from Mount Wilson: a pattern of flow bands migrating equatorward, with the greatest concentration of active regions at the poleward edge of the main equatorward-moving band. The high-latitude variations were interpreted as bands starting at the poles and taking a full 22-year Hale cycle to drift to the equator.

Howard & LaBonte(1980)이 Mount Wilson 12년 Doppler 관측으로 최초 기술: 적도 방향으로 이동하는 흐름 대상 패턴으로, 활동 영역의 극향 가장자리에서 가장 밀집.

The bands extend over about 10° in latitude, with zonal velocities a few m/s faster or slower than the surroundings — corresponding to $\lesssim 0.5\%$ of the overall rotation, or a few nanohertz.

대상 폭 ~10° 위도, 대상 속도 수 m/s (전체 자전의 ~0.5%, 수 nHz).

#### 9.2 Early Helioseismic Measurements / 초기 일진학 측정

Early hints in BBSO data (Woodard & Libbrecht, 1993). Kosovichev & Schou (1997) found evidence of flows a few m/s faster than the general rotation profile in $f$-mode measurements. Howe et al. (2000c) first reported radially-resolved evidence of zonal flow migration, combining MDI and GONG data: the equatorward-migrating part penetrated to at least 0.92 $R_\odot$ (56 Mm below the surface) at latitudes below ~40°.

Howe et al. (2000c)이 MDI/GONG 결합 데이터로 대상 흐름 이동의 반경 방향 분해 증거를 최초 보고: 적도 방향 이동 패턴이 적어도 0.92 $R_\odot$까지 침투.

#### 9.3 Recent Results / 최근 결과

Vorontsov et al. (2002) used 11-year sinusoids to characterize the pattern at each location — a key innovation making the poleward propagation obvious even with less than half a cycle of data. The high-latitude region of changing rotation involves the whole convection zone depth. The equatorward branch also penetrates through much of the convection zone, with a phase displacement at greater depths.

Vorontsov et al. (2002)이 11년 사인파를 도입하여 반 주기 미만의 데이터에서도 극향 전파를 명확하게 보여주는 혁신적 방법을 사용했습니다. 적도 방향 가지도 대류층 상당 부분을 관통하며, 깊은 곳에서 위상이 변위됩니다.

Figures 25-27 show the rotation variations at multiple depths and latitudes from GONG and MDI (both RLS and OLA), with 11-year sine functions fitted to the variations (Figure 28). The equatorward branch follows the ~25° slant of the rotation contours, being displaced in phase from the surface pattern at greater depths.

#### Key Observations about the Torsional Oscillation Pattern:

1. **Appearance depends on background subtraction**: $f$-mode results (Figure 24, smooth 3-term expansion subtracted) look different from $p$-mode inversions (Figure 25, temporal mean subtracted)
2. **Each equatorward-migrating band exists ~18 years**: emerging at mid-latitudes soon after solar maximum, disappearing at the equator a couple of years after the following cycle minimum
3. **Poleward branch lasts ~9 years**: appearing a year or so after solar minimum and moving to the pole before the next minimum
4. **New equatorward branch appears several years before** new-cycle active regions begin to erupt
5. **Amplitude is small compared to the near-surface shear** but the fractional change in the shear is much greater than the fractional change in the rotation rate

비틀림 진동의 핵심 관측: (1) 배경 빼기 방법에 따라 외관이 달라짐, (2) 적도 방향 이동 대상은 ~18년 존재, (3) 극향 대상은 ~9년, (4) 새 주기 활동 영역 출현 수 년 전에 이미 나타남, (5) 진폭은 작지만 전단의 분수 변화는 큼.

#### 9.4 Local Helioseismology / 국소 일진학

Ring-diagram analysis (Haber et al. 2002) and time-distance technique (Zhao & Kosovichev, 2004) confirmed the surface zonal flow pattern and revealed associated **meridional flow** modulation — converging/diverging flows around activity belts forming circulation cells.

링 다이어그램과 시간-거리 기법으로 표면 대상 흐름 패턴을 확인하고, 활동 영역 주위의 자오면 순환 세포가 연관된 자오면 흐름 변조를 발견했습니다.

#### 9.5 Models / 모델

Models for the torsional oscillation include:
- **Lorentz force from dynamo waves** (Schüssler 1981; Yoshimura 1981) — surface-only, equatorward-only
- **Reynolds stress response** to time-dependent dynamo magnetic field (Kitchatinov 1996) — weak poleward branch
- **Lorentz force of dynamo-generated field on angular velocity** (Covas et al. 2000) — approximately solar-like, somewhat sensitive to boundary conditions
- **Geostrophic flow from thermal forcing** (Spruit 2003) — greatest amplitude at surface, falls to 1/3 at 0.92 $R_\odot$; hard to explain depth-dependent phase
- **Mean-field flux-transport dynamo** (Rempel 2007) — poleward branch from periodic mid-latitude forcing; equatorward branch needs thermal forcing

모델들: Lorentz 힘에 의한 다이나모 파동, Reynolds 응력, 열적 강제에 의한 지균풍 등. 아직 모든 관측 특성을 완전히 재현하는 단일 모델은 없습니다.

### Section 10: Tachocline Variations / 타코클라인 변동

#### 10.1 The 1.3-Year Signal / 1.3년 신호

Howe et al. (2000b) reported variations of the equatorial rotation rate near the tachocline with a **1.3-year period** during 1995-1999, from both GONG and MDI observations:
- Strongest at $0.72 R_\odot$ (just above tachocline)
- Weaker, anticorrelated signal at $0.63 R_\odot$ (below tachocline)
- Also an apparent 1-year periodicity at higher latitudes

This attracted considerable interest because of the coincidence with the ~1.3-year period seen in some heliospheric and geomagnetic observations.

Howe et al. (2000b)이 GONG/MDI 데이터에서 tachocline 근처(0.72 $R_\odot$)의 적도 자전율에 **1.3년 주기** 변동을 보고. 이는 태양권/지자기 관측에서도 유사한 주기가 보여 큰 관심을 끌었습니다.

**However**: Antia & Basu (2000) and Basu & Antia (2001) found no significant variations with slightly different analysis. Moreover, **the periodic signal disappears in post-2001 data** even in the original authors' analysis (Howe et al. 2007 — Figure 32). It seems likely the high-latitude 1-year period was an artifact. The intermittency in short-period variations does not itself imply the phenomenon was not real — it will be interesting to see whether the oscillation reappears in the new solar cycle.

**그러나**: 다른 분석에서 유의미한 변동이 발견되지 않았고, 2001년 이후 데이터에서 원저자 분석에서도 주기 신호가 사라졌습니다. 단주기 변동의 간헐성이 현상의 비실재를 의미하지는 않지만, 확증은 아직 없습니다.

#### 10.2-10.3 Tachocline Jets and Angular Momentum / Tachocline 제트와 각운동량

Christensen-Dalsgaard et al. (2004) found possible evidence of jets near the tachocline migrating equatorward by ~30° in two years, but significance and meaning remain unestablished.

Antia et al. (2008) found temporal variations of solar kinetic energy and angular momentum on the solar-cycle timescale (but not the 1.3-year cycle), with some discrepancies between MDI and GONG results.

### Section 11: Summary and Discussion / 요약 및 논의

Helioseismology has provided several key insights:
1. Approximately rigid rotation of the radiative interior
2. Differential rotation throughout the convection zone
3. The thin tachocline
4. Extension of the surface torsional oscillation throughout the convection zone

These discoveries have repeatedly overturned theoretical expectations, inspiring modelers to improve their simulations.

**Open questions** identified at the time of writing:
- Strength of near-surface shear at high latitudes
- Rotation of the inner core
- Any inhomogeneities and changes in the tachocline
- A complete numerical model of the solar dynamo with long-term predictive capability

일진학이 밝혀낸 핵심 통찰: (1) 복사 내부 강체 자전, (2) 대류층 차등 자전, (3) 얇은 tachocline, (4) 비틀림 진동의 대류층 전체 확장. 미해결 문제: 고위도 표면 전단, 핵 자전, tachocline 변동, 예측 가능한 다이나모 모델.

---

## 3. Key Takeaways / 핵심 시사점

1. **태양 내부 자전은 4개의 뚜렷한 영역으로 구성됩니다** — 강체 자전하는 복사 내부, 얇은 tachocline 전이층, 차등 자전하는 대류층, 그리고 표면 근처 전단층. 이 구조는 17세기 이래 알려진 표면 차등 자전보다 훨씬 복잡합니다.
   **The solar interior rotation comprises four distinct regions** — rigidly rotating radiative interior, thin tachocline transition, differentially rotating convection zone, and near-surface shear layer. This structure is far more complex than the surface differential rotation known since the 17th century.

2. **대류층의 등자전 윤곽은 원통형이 아니라 ~25° 기울어져 있습니다** — Taylor-Proudman 정리의 예측(원통형 자전)과 반대되는 이 관측은 대류층 역학에서 Coriolis 힘, 열 구배, Reynolds 응력의 상호작용이 복잡함을 보여줍니다.
   **Constant-rotation contours are tilted ~25° rather than cylindrical** — contradicting the Taylor-Proudman prediction, revealing complex interplay of Coriolis forces, thermal gradients, and Reynolds stresses in convection zone dynamics.

3. **Tachocline은 극히 얇은 전단층($\Gamma \sim 0.02 R_\odot$)으로, 태양 다이나모의 핵심 위치입니다** — tachocline의 발견은 다이나모가 대류층 전체가 아닌 이 좁은 영역에서 작동한다는 패러다임 전환을 가져왔습니다. 그러나 이 층이 어떻게 유지되는지(난류, 자기장, 중력파)는 아직 미해결입니다.
   **The tachocline is an extremely thin shear layer ($\Gamma \sim 0.02 R_\odot$), the key site for the solar dynamo** — its discovery was a paradigm shift. How it is maintained remains an open question.

4. **비틀림 진동은 표면 현상이 아니라 대류층 전체 깊이에 걸친 현상입니다** — GONG/MDI 데이터가 보여준 것처럼, 대상 흐름 패턴이 대류층 상당 깊이까지 침투하며 ~25° 기울어진 등자전 윤곽을 따라 위상이 변합니다. 이는 다이나모-자전 피드백의 직접적 증거입니다.
   **The torsional oscillation penetrates through the full convection zone depth** — not merely a surface phenomenon. Flow bands follow the 25° slant of rotation contours, providing direct evidence of dynamo-rotation feedback.

5. **핵 자전은 ~40년 연구에도 불구하고 여전히 불확실합니다** — $p$-mode 저차수 splitting 값은 ~0.43 $\mu$Hz로 수렴했지만, 이는 강체 자전과 핵에서의 약간의 감소 모두와 양립합니다. $g$-mode 검출이 결정적이겠지만 아직 미확인 상태입니다.
   **Core rotation remains uncertain despite ~40 years of research** — low-degree $p$-mode splittings converge to ~0.43 $\mu$Hz, consistent with either rigid rotation or slight downturn in the core. $g$-mode detection would be decisive but remains unconfirmed.

6. **역산 방법의 선택이 결과에 중요한 영향을 미칩니다** — RLS와 OLA/SOLA는 다른 평균 커널을 생성하며, 특히 해상도가 떨어지는 영역(고위도, 심부)에서 결과가 달라질 수 있습니다. 여러 방법의 비교가 결과 신뢰성 검증에 필수적입니다.
   **Choice of inversion method significantly affects results** — RLS and OLA/SOLA produce different averaging kernels, leading to potentially different results where resolution is poor. Comparing multiple methods is essential for validation.

7. **다중 독립 관측 시설의 중요성이 반복적으로 입증되었습니다** — GONG과 MDI 사이의 불일치(예: "polar jet" 아티팩트, 데이터 파이프라인 차이)가 발견될 때마다, 진정한 태양 특성과 체계적 오차를 구별하는 것이 핵심 과제였습니다. 독립적 교차 검증이 없으면 아티팩트가 진짜 물리학으로 잘못 해석될 위험이 있습니다.
   **The importance of multiple independent observing facilities has been repeatedly demonstrated** — discrepancies between GONG and MDI (the "polar jet" artifact, pipeline differences) highlighted the challenge of distinguishing real solar features from systematic errors.

8. **Tachocline 1.3년 신호는 과학적 검증의 좋은 사례입니다** — 초기 보고가 큰 관심을 끌었지만, 다른 그룹의 분석에서는 확인되지 않았고, 추가 데이터에서 신호가 사라졌습니다. 이는 시계열 분석에서 짧은 데이터에 기반한 주기 검출의 위험성을 보여줍니다.
   **The tachocline 1.3-year signal is an instructive case of scientific verification** — initially exciting, unconfirmed by other groups, and disappearing in extended data. It demonstrates risks of period detection from short time series.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Mode Classification and Displacement / 모드 분류 및 변위

유체 요소의 방사 변위 / Radial displacement of a fluid element:
$$\delta r(r,\theta,\phi,t) = \sum_{m=-l}^{l} a_{nlm} \xi_{nl}(r) Y_l^m(\theta,\phi) e^{i\omega_{nlm} t} \tag{1}$$

- $n$: radial order (방사 차수)
- $l$: spherical harmonic degree (구면 조화 차수) — 표면 nodal line 수
- $m$: azimuthal order (방위각 차수, $-l \le m \le l$)
- $\xi_{nl}(r)$: radial eigenfunction (방사 고유함수)
- $Y_l^m(\theta,\phi)$: spherical harmonic (구면 조화 함수)

Lower turning point: $r_t$ determined by $\nu/L$ where $L = \sqrt{l(l+1)} \approx l + 1/2$.
하한 전환점: $\nu/L$에 의해 결정, 낮은 $l$ → 깊은 침투.

### 4.2 Frequency Expansion and Splitting Coefficients / 주파수 전개 및 분리 계수

$$\nu_{nlm} = \nu_{nl} + \sum_{j=1}^{j_{\max}} a_j(n,l) \mathcal{P}_j^{(l)}(m) \tag{2}$$

$$\mathcal{P}_j^{(l)}(m) = \frac{l\sqrt{(2l-j)!(2l+j+1)!}}{(2l)!\sqrt{2l+1}} C_{j0lm}^{lm} \tag{3}$$

- $a_j$: splitting coefficients (분리 계수)
- $\mathcal{P}_j^{(l)}$: Ritzwoller-Lavely polynomials (Clebsch-Gordan 계수와 관련)
- **Odd $j$** ($a_1, a_3, a_5$): rotation information (자전 정보)
- **Even $j$** ($a_2, a_4, a_6$): structural asphericity (구조 비구대칭)
- $a_1 \approx$ average rotation; $a_3$ and higher $\approx$ differential rotation

### 4.3 The Forward Problem / 순방향 문제

Observation as weighted integral of rotation:
$$d_i = \int_0^{R_\odot} \int_0^{\pi} K_i(r,\theta) \Omega(r,\theta) \, dr \, d\theta + \epsilon_i \tag{11}$$

Rotation kernel:
$$K_{nlm}(r,\theta) = \frac{m}{I_{nl}} \left\{ \xi_{nl}\left[\xi_{nl} - \frac{2}{L}\eta_{nl}\right] P_l^m(x)^2 + \frac{\eta_{nl}^2}{L^2}\left[\left(\frac{dP_l^m}{dx}\right)^2(1-x^2) - 2P_l^m\frac{dP_l^m}{dx}x + \frac{m^2}{1-x^2}P_l^m(x)^2\right]\right\}\rho(r)r\sin\theta \tag{12}$$

where $I_{nl} = \int_0^{R_\odot} [\xi_{nl}^2 + \eta_{nl}^2]\rho(r) r^2 dr$ (Eq. 13), $x = \cos\theta$.

### 4.4 The Inverse Problem / 역방향 문제

Inversion solution:
$$\bar{\Omega}(r_0,\theta_0) = \sum_{i=1}^{M} c_i(r_0,\theta_0) d_i \tag{14}$$

Averaging kernel:
$$\mathcal{K}(r_0,\theta_0;r,\theta) = \sum_{i=1}^{M} c_i(r_0,\theta_0) K_i(r,\theta) \tag{16}$$

The inferred $\bar{\Omega}$ is a spatial average of the true $\Omega$ weighted by $\mathcal{K}$.

### 4.5 RLS Inversion / RLS 역산

Minimize:
$$\sum_i \frac{[d_i - \int\int \Omega K_i \, dr\,d\theta]^2}{(\sigma_i/\bar{\sigma})^2} + \mu_r^2 \int\int \left(\frac{\partial^2\Omega}{\partial r^2}\right)^2 dr\,d\theta + \mu_\theta^2 \int\int \left(\frac{\partial^2\Omega}{\partial\theta^2}\right)^2 dr\,d\theta \tag{21}$$

- $\mu_r, \mu_\theta$: tradeoff parameters (반경/위도 방향 매끄러움과 데이터 적합도 사이의 균형)

### 4.6 SOLA Inversion / SOLA 역산

Minimize:
$$\int_0^R \int_0^\pi [\mathcal{T} - \mathcal{K}]^2 \, r\,dr\,d\theta + \lambda \sum_{i=1}^{M} [\sigma_i c_i]^2 \tag{22}$$

- $\mathcal{T}$: target kernel (목표 커널, 예: Gaussian)
- $\lambda$: resolution-error tradeoff parameter

### 4.7 Error Estimates / 오차 추정

Formal uncertainty:
$$\sigma^2[\Omega(r_0,\theta_0)] = \sum_i [c_i(r_0,\theta_0)\sigma_i]^2 \tag{17}$$

Error magnification (for uniform errors):
$$\Lambda(r_0,\theta_0) = \left[\sum_i c_i(r_0,\theta_0)^2\right]^{1/2} \tag{19}$$

Error correlation between two points:
$$C(r_0,r_1) = \frac{\sum c_i(r_0) c_i(r_1) \sigma_i^2}{[\sum c_i^2(r_0)\sigma_i^2]^{1/2}[\sum c_i^2(r_1)\sigma_i^2]^{1/2}} \tag{20}$$

### 4.8 Surface Differential Rotation / 표면 차등 자전

Magnetic features: $\frac{\Omega_m}{2\pi} = 462 - 74\mu^2 - 53\mu^4$ nHz (Eq. 25)

Surface plasma: $\frac{\Omega_p}{2\pi} = 452 - 49\mu^2 - 84\mu^4$ nHz (Eq. 26)

where $\mu = \sin(\text{latitude})$. The ~10 nHz equatorial difference reflects the near-surface shear.

### 4.9 Solar Oblateness / 태양 편평도

$$\frac{\Delta r}{r_0} = \frac{3}{2}J_2 \tag{23}$$

$$\frac{\Delta r - \delta r}{r_0} = \frac{3}{2}J_2 \tag{24}$$

where $\delta r$ is the additional contribution from surface rotation. Units of $\delta r$ and $\Delta r$ are milliarcseconds.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1960s    Dicke의 편평도 측정 — 핵 자전 논쟁 시작
         Dicke oblateness — core rotation controversy begins
            |
1975     Deubner: 5분 p-mode 진동 발견
         Deubner: 5-min p-mode oscillation discovery
            |
1979     Birmingham group: 전구 저차수 모드 발견
         Birmingham group: global low-degree modes
            |
1980     Howard & LaBonte: 비틀림 진동 발견 (Mt Wilson Doppler)
         Howard & LaBonte: torsional oscillation discovery
            |
1984     Duvall & Harvey: 남극 관측 → 내부 자전 최초 추론
         Duvall & Harvey: South Pole → first interior rotation
            |
1986     Libbrecht: BBSO 100일 관측, 중간차수 모드
         Libbrecht: BBSO 100-day, medium-degree modes
            |
1989     Brown et al.: 대류층 자전 + tachocline 발견
         Brown et al.: convection zone rotation + tachocline
            |
1992     Spiegel & Zahn: "tachocline" 용어 명명, 이론적 모델
         Spiegel & Zahn: coin "tachocline", theoretical model
            |
1995     GONG 가동 시작 / GONG begins
            |
1996     SOHO/MDI 가동 시작 / SOHO/MDI begins
            |
1998     Schou et al.: MDI 144일 역산 — 결정적 자전 프로파일
         Schou et al.: MDI 144-day — definitive rotation profile
            |
2000     Howe et al.: tachocline 1.3년 신호 / 1.3-yr signal
         Howe et al.: torsional oscillation 반경 분해 / radially resolved
            |
2005     Howe et al.: 25° 기울어진 등자전 윤곽 / 25° slanted contours
            |
2008     Miesch et al.: 태양과 유사한 3D 자전 시뮬레이션
         Miesch et al.: solar-like 3D rotation simulation
            |
2009  ★ Howe: "Solar Interior Rotation and its Variation" ★
         — 40년 관측적 이해의 종합 리뷰
            |
2010     SDO/HMI 발사 / SDO/HMI launch
            |
2010s    HMI 데이터로 태양 주기 24 자전 변동 연구 계속
         Continued rotation variation studies with HMI through cycle 24
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **LRSP #5: Gizon & Birch (2005)** "Local Helioseismology" | 국소 일진학 기법 (ring diagram, time-distance)의 기초. 이 리뷰에서 다루는 torsional oscillation의 국소적 측정(Section 9.4)과 표면 전단(Section 8)의 핵심 방법론. / Foundation of local helioseismology techniques used for torsional oscillation measurement (Sec. 9.4) and near-surface shear (Sec. 8). | 직접 선수 논문 / Direct prerequisite |
| **Schou et al. (1998)** "Helioseismic Studies of Differential Rotation..." | MDI 144일 데이터의 4가지 역산 방법 비교. 대류층 자전이 원통형이 아님을 확정적으로 증명한 획기적 논문. Figure 18의 원천. / Definitive proof that convection zone rotation is not cylindrical. Source of Figure 18. | 이 리뷰의 핵심 관측 결과 / Core observational result of this review |
| **Brown et al. (1989)** "Inferring the Sun's internal angular velocity..." | 대류층 자전 프로파일의 최초 정량적 추론과 tachocline의 존재를 최초로 확인한 논문. / First quantitative inference of convection zone rotation and first confirmation of tachocline existence. | Tachocline 발견 / Tachocline discovery |
| **Spiegel & Zahn (1992)** "The solar tachocline" | "Tachocline" 용어를 명명하고 난류에 의한 유지 메커니즘을 제안한 이론 논문. / Named the "tachocline" and proposed turbulent maintenance mechanism. | Tachocline 이론의 출발점 / Starting point of tachocline theory |
| **Howard & LaBonte (1980)** "The Sun is observed to be a torsional oscillator..." | 비틀림 진동의 최초 발견. 이 리뷰 Section 9의 출발점. / First discovery of torsional oscillation. Starting point of Section 9. | 비틀림 진동 발견 / Torsional oscillation discovery |
| **Howe et al. (2000b)** "Dynamic Variations at the Base of the Solar Convection Zone" | Tachocline 근처의 1.3년 변동 보고. 이 리뷰 Section 10의 핵심 주제이자 과학적 검증의 사례. / Report of 1.3-year variations near tachocline. Key subject of Section 10 and a case study in scientific verification. | Tachocline 시간 변동 / Tachocline temporal variations |
| **Miesch et al. (2008)** "Structure and Evolution of Giant Cells..." | 3D 구면 대류 시뮬레이션에서 시간 평균 후 태양과 유사한 자전 패턴(기울어진 윤곽, 차등 자전)을 재현. / 3D simulation reproducing solar-like rotation pattern after temporal averaging. | 관측과 모델의 연결 / Observation-model connection |
| **LRSP series: other rotation/dynamo reviews** | 이 리뷰는 Miesch (2005) "Large-Scale Dynamics of the Convection Zone and Tachocline"와 상보적. 관측자 vs. 모델러 관점. / Complementary to Miesch (2005) review — observer vs. modeler perspective. | 상보적 리뷰 / Complementary reviews |

---

## 7. References / 참고문헌

- Howe, R., "Solar Interior Rotation and its Variation", *Living Rev. Solar Phys.*, **6**, 1 (2009). DOI: 10.12942/lrsp-2009-1
- Schou, J., et al., "Helioseismic Studies of Differential Rotation in the Solar Envelope by the Solar Oscillations Investigation Using the Michelson Doppler Imager", *Astrophys. J.*, **505**, 390-417 (1998)
- Brown, T.M., et al., "Inferring the Sun's internal angular velocity from observed p-mode frequency splittings", *Astrophys. J.*, **343**, 526-546 (1989)
- Spiegel, E.A. & Zahn, J.-P., "The solar tachocline", *Astron. Astrophys.*, **265**, 106-114 (1992)
- Howard, R. & LaBonte, B.J., "The Sun is observed to be a torsional oscillator with a period of 11 years", *Astrophys. J.*, **239**, L33-L36 (1980)
- Howe, R., et al., "Dynamic Variations at the Base of the Solar Convection Zone", *Science*, **287**, 2456-2460 (2000)
- Miesch, M.S., et al., "Structure and Evolution of Giant Cells in Global Models of Solar Convection", *Astrophys. J.*, **673**, 557-575 (2008)
- Gizon, L. & Birch, A.C., "Local Helioseismology", *Living Rev. Solar Phys.*, **2**, 6 (2005)
- Christensen-Dalsgaard, J., et al., "The Current State of Solar Modeling", *Science*, **272**, 1286-1292 (1996)
- Howe, R., et al., "Deeply Penetrating Banded Zonal Flows in the Solar Convection Zone", *Astrophys. J.*, **634**, 1405-1415 (2005)
- Howe, R., et al., "Solar Convection-Zone Dynamics, 1995-2004", *Astrophys. J.*, **649**, 1155-1168 (2006a)
- Corbard, T. & Thompson, M.J., "The subsurface radial gradient of solar angular velocity from MDI f-mode observations", *Solar Phys.*, **205**, 211-229 (2002)
- Vorontsov, S.V., et al., "Helioseismic measurement of solar torsional oscillations", *Science*, **296**, 101-103 (2002)
- Basu, S. & Antia, H.M., "Changes in Solar Dynamics from 1995 to 2002", *Astrophys. J.*, **585**, 553-565 (2003)
- Rempel, M., "Origin of Solar Torsional Oscillations", *Astrophys. J.*, **655**, 651-659 (2007)
