---
title: "Solar Adaptive Optics"
authors: Thomas R. Rimmele, Jose M. Marino
year: 2011
journal: "Living Reviews in Solar Physics, Vol. 8, Article 2"
doi: "10.12942/lrsp-2011-2"
topic: Solar Observation
tags: [adaptive-optics, wavefront-sensing, solar-telescope, MCAO, Shack-Hartmann, correlation-tracking, Kolmogorov-turbulence, high-resolution-imaging]
status: completed
date_started: 2026-04-19
date_completed: 2026-04-19
---

# 20. Solar Adaptive Optics / 태양 적응광학

---

## 1. Core Contribution / 핵심 기여

이 논문은 **태양 적응광학(Solar Adaptive Optics, SAO)** 분야의 이론적 기반, 공학적 구현, 운용 사례, 그리고 미래 확장(MCAO/GLAO)을 한데 모은 종합 리뷰이다. 지상 태양망원경이 회절 한계 해상도에 도달하기 위해서는 대기 난류(seeing)에 의한 파면 왜곡을 실시간으로 보정해야 하는데, 태양은 **확장된 저대비 연속 광원**이기 때문에 밤하늘 AO의 점광원 기반 센트로이드 방법을 적용할 수 없다. 저자들은 이 핵심 난제를 해결한 **상관 추적 Shack–Hartmann(correlation-tracking SH)** 파면 감지기의 원리와 성능을 상세히 기술하고, Kolmogorov 난류 통계(, , , )에서 출발해 변형 거울(DM) 선택, 실시간 제어 루프 설계, 폐루프 성능 예산(error budget)까지 전 과정을 연결한다. 또한 DST, SST, VTT, GREGOR에서의 운용 성과를 Strehl ratio와 과학적 발견 관점에서 정리하고, DKIST(당시 ATST)와 EST를 위한 차세대 **Multi-Conjugate AO(MCAO)** 와 **Ground-Layer AO(GLAO)** 시스템의 원리·초기 실험 결과를 제시한다.

This review consolidates the theoretical foundations, engineering implementation, operational experience, and future extensions of **Solar Adaptive Optics (SAO)**. Ground-based solar telescopes can reach diffraction-limited resolution only by correcting atmospheric-turbulence wavefront aberrations in real time, but the Sun is an **extended, low-contrast source** that defeats the centroid-based reference-star wavefront sensing used in nighttime AO. The authors detail how **correlation-tracking Shack–Hartmann (SH) wavefront sensing** solves this core problem, and they weave together the full system: Kolmogorov statistics ( $r_0$, $\theta_0$, $f_G$, $\tau_0$ ), deformable mirror (DM) selection, real-time control loop design, and closed-loop error-budget analysis. Performance results from the **DST, SST, VTT, and GREGOR** are summarized through Strehl metrics and the science they have enabled, and the review previews **Multi-Conjugate AO (MCAO)** and **Ground-Layer AO (GLAO)** prototypes that will serve the next-generation 4-m class telescopes (DKIST/ATST, EST).

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction & Motivation (§1) / 서론과 동기

태양 물리학의 최전선(미세 자기 플럭스 튜브, 과립 내부 동역학, 채층 파동, 플레어 촉발 현장)은 모두 **~100 km 공간 스케일**을 요구하며, 이는 지구에서 ~0.15 arcsec에 해당한다. 가시광(500 nm)에서 1 m 구경이 회절 한계 0.13 arcsec를 제공하지만, 일반 seeing은 1–3 arcsec 수준이어서 회절 한계가 전혀 활용되지 못한다. **AO 없이 지상 태양 관측은 무의미**하다는 것이 SAO 개발의 출발점.

At the solar-physics frontier (small-scale magnetic flux tubes, intra-granular dynamics, chromospheric waves, flare trigger sites), the required spatial scale is ~100 km on the Sun ≈ **0.15 arcsec** from Earth. A 1-m aperture yields a 0.13″ diffraction limit at 500 nm, but typical solar seeing is 1–3″, so the diffraction limit is wasted without AO. **SAO is therefore not an enhancement but a prerequisite** for ground-based high-resolution solar physics.

---

### Part II: Atmospheric Turbulence (§2) / 대기 난류

**Kolmogorov 난류 이론**이 기반이다. 굴절률 요동은 에너지 캐스케이드의 수동 스칼라(passive scalar)로 취급되어 구조 함수가

$$
D_n(r) = C_n^2\, r^{2/3}
$$

형태를 갖는다. 여기서 $C_n^2$은 굴절률 구조 상수(단위 $\text{m}^{-2/3}$), outer scale $L_0$(~수 m ~ 수십 m)와 inner scale $l_0$(~mm) 사이에서 유효하다.

The theory rests on **Kolmogorov turbulence**, treating refractive-index fluctuations as a passive scalar in an energy cascade. The structure function scales as $D_n(r) = C_n^2 r^{2/3}$ between the inner scale $l_0$ (~mm) and the outer scale $L_0$ (~meters to tens of meters).

**Fried parameter** $r_0$는 파면 위상의 통합 효과를 단일 길이로 요약한다:

$$
r_0 = \left[ 0.423 \left(\frac{2\pi}{\lambda}\right)^{\!2} \sec\zeta \int_0^\infty C_n^2(h)\,dh \right]^{-3/5}
$$

- $\lambda^{6/5}$로 증가 → **적외선이 가시광보다 AO가 쉬움** (원적외선에서는 seeing이 거의 문제되지 않음).
- 구경 $D$ 위에서 비보정 파면 분산은 $\sigma_\phi^2 = 1.03 (D/r_0)^{5/3}$ rad² 이다.

**Fried parameter** $r_0$ collapses integrated turbulence into a single length. It scales as $\lambda^{6/5}$ — infrared observing is much easier than visible. Uncorrected phase variance over aperture $D$ is $\sigma_\phi^2 = 1.03 (D/r_0)^{5/3}$.

**Greenwood frequency** $f_G$ — 제어 대역폭 요건:

$$
f_G = \left[ 0.102 \left(\frac{2\pi}{\lambda}\right)^{\!2} \sec\zeta \int_0^\infty C_n^2(h)\, v^{5/3}(h)\, dh \right]^{3/5}
$$

AO 서보 폐루프 대역폭이 $f_G$보다 충분히 커야 한다. 낮 태양 관측은 $f_G \sim 50–200$ Hz, 즉 수 kHz의 샘플링이 필요.

**Greenwood frequency** $f_G$ sets the required servo bandwidth; daytime solar values of 50–200 Hz translate to a kHz-scale loop rate.

**Isoplanatic angle** $\theta_0$ — 고고도($h^{5/3}$ 가중) 난류가 지배:

$$
\theta_0 = 2.914 k^2 \sec^{8/3}\zeta \int_0^\infty C_n^2(h)\, h^{5/3}\, dh \Big]^{-3/5}
$$

태양(낮)에서 가시광 $\theta_0 \simeq 5\text{–}10$ arcsec 수준이다. 태양 활동 영역(AR) 하나의 시야(~60″)는 $\theta_0$보다 훨씬 크기 때문에 **단일 DM 보정만으로는 활동 영역 전체를 보정할 수 없다** → MCAO 동기.

$\theta_0$ is dominated by high-altitude layers (weight $h^{5/3}$). In daytime at visible wavelengths it is only 5–10″. Active-region FOVs (~60″) are therefore much larger than $\theta_0$, motivating MCAO.

**낮 seeing의 특이점 / Daytime seeing specifics** — 지표면 가열로 **ground layer 난류가 지배적**. 대형 돔의 열 관리, 경면 냉각, 사이트 선택(고지대, 해안 풍상)이 중요. Sac Peak, La Palma, Haleakalā가 전형적인 양질 사이트.

Daytime seeing is dominated by **ground-layer turbulence** driven by surface heating. Dome thermal control, mirror cooling, and site selection (high elevation, upwind of ocean) are critical. Sac Peak, La Palma, and Haleakalā are canonical sites.

---

### Part III: Principles of Adaptive Optics (§3) / 적응광학의 원리

**폐루프 AO 시스템**은 3요소로 구성된다:
1. **파면 감지기(Wavefront Sensor, WFS)** — Shack–Hartmann, Curvature, Pyramid 중 태양은 거의 SH 사용.
2. **변형 거울(Deformable Mirror, DM)** — 압전(PZT), bimorph, MEMS 중 태양은 PZT continuous facesheet가 표준.
3. **실시간 제어기(RTC)** — 기울기 측정 → 재구축(reconstruction) → DM 명령, 지연 < 1 ms 목표.

A closed-loop AO system has three elements: **WFS**, **DM**, and **Real-Time Controller (RTC)**. Solar AO almost exclusively uses SH sensors and PZT continuous-facesheet DMs, with loop latencies below 1 ms.

**파면 재구축(Reconstruction)** — 측정된 국부 기울기 벡터 $\mathbf{s}$로부터 DM 액추에이터 명령 $\mathbf{a}$를 구한다:

$$
\mathbf{a} = R\, \mathbf{s}, \qquad R = (H^T C_n^{-1} H)^{-1} H^T C_n^{-1}
$$

여기서 $H$는 영향 함수(interaction matrix), $C_n$은 측정 잡음 공분산, $R$은 제어 행렬(reconstruction matrix). 정칙화(Tikhonov) 또는 SVD 절단으로 잡음 증폭을 억제.

**Reconstruction** maps measured slopes $\mathbf{s}$ to actuator commands $\mathbf{a}$ via $\mathbf{a} = R\mathbf{s}$, with $R$ built from the interaction matrix $H$ and noise covariance $C_n$, regularized by Tikhonov or SVD truncation to limit noise amplification.

**폐루프 제어** — 가장 단순한 적분기(integrator):

$$
\mathbf{a}_{k+1} = \mathbf{a}_k + g R\, \mathbf{s}_k
$$

이득 $g \in (0,1)$. 더 진화된 제어기(LQG, predictive)는 temporal error를 추가로 감소시킨다.

**Closed-loop** typically uses a simple integrator $\mathbf{a}_{k+1} = \mathbf{a}_k + g R \mathbf{s}_k$ with gain $g$; advanced controllers (LQG, predictive) reduce temporal error further.

**에러 예산(Error Budget)** — 잔차 분산은 각 오차원의 제곱 합:

$$
\sigma_{\rm tot}^2 = \sigma_{\rm fit}^2 + \sigma_{\rm temp}^2 + \sigma_{\rm noise}^2 + \sigma_{\rm aniso}^2 + \sigma_{\rm calib}^2 + \cdots
$$

- $\sigma_{\rm fit}^2 = 0.28 (d/r_0)^{5/3}$ (actuator spacing $d$ 한계)
- $\sigma_{\rm temp}^2 = (f_G / f_{3\text{dB}})^{5/3}$
- $\sigma_{\rm aniso}^2 = (\theta/\theta_0)^{5/3}$

Maréchal: $S \approx \exp(-\sigma_{\rm tot}^2)$ 로 Strehl을 추정.

The **residual variance** sums fitting, temporal, noise, anisoplanatic, and calibration errors; $S \approx \exp(-\sigma_{\rm tot}^2)$. Each term has an analytic scaling that shapes system design: $\sigma_{\rm fit}^2 \propto (d/r_0)^{5/3}$ sets actuator count; $\sigma_{\rm temp}^2 \propto (f_G/f_{3\rm dB})^{5/3}$ sets loop rate; $\sigma_{\rm aniso}^2 \propto (\theta/\theta_0)^{5/3}$ sets FOV.

---

### Part IV: Wavefront Sensing for the Sun (§4) — **Heart of the Paper** / 태양 파면 감지

**문제의식 / The problem** — 별은 점광원이라 SH subaperture 이미지가 한 픽셀 Airy 패턴에 집중되어, 센트로이드로 기울기를 구한다. 태양은 과립 패턴(대비 ~8%, 각 과립 ~1″)으로 이루어진 **2D 확장 장면**이라 센트로이드가 정의되지 않는다.

Stellar SH subaperture images are point-source Airy patches; slopes are estimated by centroids. Solar subaperture images are **extended granulation patterns** (contrast ~8%, granule size ~1″), so centroids are ill-defined.

**해법: 상관 추적(Correlation Tracking) / Solution: correlation tracking** — Livingston(1976), von der Lühe(1983)가 제안, **Rimmele(1999)** 가 실시간 SH로 완성. 각 subaperture 이미지 $I_i(\mathbf{x})$를 참조 패치 $R(\mathbf{x})$에 대해 교차상관하여 최대값 위치 $\hat{\mathbf{s}}_i$를 국부 기울기로 채택:

$$
C_i(\mathbf{s}) = \sum_{\mathbf{x}} I_i(\mathbf{x})\, R(\mathbf{x} - \mathbf{s})
$$

서브 픽셀 정확도는 2D 파라볼라 피팅 또는 Fourier 시프트 정리로 달성. 전형적인 서브이미지는 16×16 ~ 32×32 pix, 참조 패치는 현재 프레임 중앙을 기준으로 실시간 업데이트.

**Correlation tracking** — proposed by Livingston (1976) and von der Lühe (1983), realized in real-time SH by **Rimmele (1999)**. Each subimage is cross-correlated against a reference patch, and the peak location $\hat{\mathbf{s}}_i$ replaces the centroid. Sub-pixel accuracy comes from 2D parabolic fit or Fourier shift theorem. Typical subimage size 16–32 pix; the reference patch is updated in real time from the live frame center.

**변형: Absolute Difference Squared (ADS)** — 곱셈 대신 차이 제곱합이 빠르고 저대비에서 견고:

$$
D_i(\mathbf{s}) = \sum_{\mathbf{x}} [I_i(\mathbf{x}) - R(\mathbf{x} - \mathbf{s})]^2
$$

최소화 위치가 기울기. GPU/FPGA에서 효율적으로 구현.

**ADS** — sum of squared differences; robust and fast on GPUs/FPGAs, often preferred in hardware implementations.

**잡음 해석 / Noise analysis** — SH 상관 추적의 기울기 잡음은 광자 잡음, 검출기 잡음, 과립 대비 $c$, 서브이미지 크기 $N$ 에 의존:

$$
\sigma_s^2 \approx \frac{\sigma_{\rm ph}^2 + \sigma_{\rm det}^2}{c^2\, N^2}
$$

과립 대비가 높을수록(계절·고도 의존) 잡음이 줄어든다. 일반적 프레임 속도 1–2 kHz에서 서브애퍼처당 ~0.01 arcsec rms 달성 가능.

**Noise** — slope variance scales as $\sigma_s^2 \propto (\sigma_{\rm ph}^2 + \sigma_{\rm det}^2)/(c^2 N^2)$ with granule contrast $c$ and subimage size $N$. Achievable ~0.01″ rms per subaperture at 1–2 kHz.

**차선 WFS / Alternative sensors** — Curvature 센서는 저차 모드에 강하나 저대비 태양에 취약. Pyramid WFS는 별 AO에서 성능이 좋으나 태양용 상관 기반 확장이 필요(미성숙). **실제 태양 AO는 거의 모두 상관 추적 SH.**

Alternative sensors (curvature, pyramid) see limited use; **correlation-SH is the de facto standard**.

---

### Part V: Deformable Mirrors & Real-Time Control (§5–6) / 변형 거울과 실시간 제어

**DM 종류 비교 / DM technologies**

| Type | Stroke | Actuators | Bandwidth | Solar use |
|---|---|---|---|---|
| PZT continuous facesheet | 5–10 μm | 97–~360 | kHz | DST, SST, GREGOR |
| Bimorph | 20+ μm | 수십 | kHz | 초기 시스템 |
| MEMS | 2–4 μm | 수백 ~ 수천 | kHz | 차세대, 실험 |

태양 AO는 **스트로크보다 액추에이터 수**가 핵심. $N_{\rm act} \sim (D/d)^2$, $d \lesssim r_0/2$ 요구. 1-m / $r_0 = 10$ cm → $N_{\rm act} \gtrsim 100$.

DMs: **actuator count** is more critical than stroke. With $d \lesssim r_0/2$, a 1-m aperture at $r_0 = 10$ cm needs $\gtrsim 100$ actuators. PZT continuous facesheets dominate solar AO.

**Tip-Tilt 거울** — 대진폭·저주파 전체 기울기는 별도 TT 거울이 담당하여 DM 스트로크를 보존.

**Tip-tilt** is handled by a dedicated TT mirror to preserve DM stroke.

**실시간 제어 하드웨어 / RTC hardware** — 2011년 기준 DSP/FPGA 기반 시스템이 주류. DST RTC: 1.2 kHz 루프, 지연 200 μs, 76-subap / 97-act. SST: 85-subap / 37-act, ~1 kHz. **GPU 기반은 2010년대 중반부터 확산.**

RTC hardware circa 2011 is DSP/FPGA-based; DST runs at 1.2 kHz with 200 μs latency and 76-subap/97-actuator order; SST runs 85/37 at ~1 kHz. GPU-based RTCs became common after this review.

---

### Part VI: System Performance & Science Results (§7) / 시스템 성능과 과학 성과

**Representative systems (2011)**

| 시설 | 구경 | 서브애퍼처 | 액추에이터 | 루프 / Loop | 대표 성능 |
|---|---|---|---|---|---|
| **DST (NSO/Sac Peak)** | 0.76 m | 76 | 97 | 1.2 kHz | $S \simeq 0.3$ @ 500 nm, $r_0 = 10$ cm |
| **SST (La Palma)** | 1.0 m | 85 | 37 (초기)/85+ | 1 kHz | 회절 한계 근접 @ 600 nm |
| **VTT (Tenerife)** | 0.7 m | 36 | 35 | ~1 kHz | KAOS 시스템 |
| **GREGOR (Tenerife)** | 1.5 m | ~155 | 256 | 2 kHz | 2011 first light |

DST는 $r_0=10$ cm 조건에서 가시광 Strehl ~0.3–0.5를 일상적으로 달성. 빨간색(656 nm Hα)과 적외선(1.56 μm Fe I)에서는 더 높은 Strehl.

At $r_0=10$ cm, DST routinely delivers visible Strehl 0.3–0.5 and near-diffraction-limited Hα and 1.56 μm infrared imaging.

**과학적 성과 예시 / Science highlights**
- **Penumbral filaments**: Scharmer et al.(2002), SST — 0.1″ 해상도로 penumbra의 dark cores 발견.
- **G-band bright points**: 자기 플럭스 관(flux-tube) 모델 검증, 자기 다이나모 이해 진전.
- **Chromospheric oscillations**: IBIS, CRISP narrow-band imager와 AO 결합으로 H$\alpha$·Ca II 미세 파동 관측.
- **Sunspot umbral dots**: DST/IBIS로 umbra 내부 대류 구조 시각화.

**Science highlights** enabled by SAO include Scharmer et al. (2002) discovery of penumbral dark cores (SST), G-band bright points confirming flux-tube models, chromospheric oscillation imaging with IBIS/CRISP, and resolved umbral-dot convection inside sunspots.

**AO + Speckle reconstruction / AO와 스펙클 재구축의 결합** — AO는 잔차를 남기므로 **Multi-Object Multi-Frame Blind Deconvolution(MOMFBD)** 또는 phase-diversity 후처리로 완성. SST의 대표 작품들은 AO + MOMFBD 조합.

**AO + post-processing**: residual aberrations are cleaned by **MOMFBD** or phase diversity. SST's flagship images combine AO with MOMFBD.

---

### Part VII: Multi-Conjugate AO (§8) / 다중 공액 AO

**동기 / Motivation** — $\theta_0 \sim 5″$ 는 활동 영역 전체(~60″) 보다 훨씬 작아 단일 DM으로는 **isoplanatic patch** 밖에서 Strehl이 급격히 떨어진다. **MCAO**는 여러 고도의 난류층을 별도의 DM이 보정하여 시야 전체에 걸쳐 거의 일정한 Strehl을 제공.

$\theta_0 \sim 5″$ is far smaller than a typical AR FOV (~60″). **MCAO** uses multiple DMs conjugated to different turbulent layers, providing near-uniform Strehl across the full FOV.

**구성 / Configuration**

1. **여러 WFS** — 시야 내 다수 지점에서 상관 추적 SH를 병렬 운용 (solar metapupil 개념).
2. **토모그래피 재구축(Tomographic reconstruction)** — 다수 시선의 측정으로 3D $C_n^2(h)$ 추정.
3. **여러 DM** — 각각 특정 고도($h_j$)의 pupil에 공액(conjugate). 예: 지표면(0 km) + 고층(6 km) 두 DM.

Core MCAO components: **multiple WFSs** (parallel correlation-SH across the FOV), **tomographic reconstruction** to estimate $C_n^2(h)$, and **multiple DMs** conjugated to different altitudes (e.g., ground + 6 km).

**Ground-Layer AO (GLAO)** — 단일 DM을 지표 공액으로 두어 지표층만 제거. 낮은 Strehl이지만 **매우 넓은 FOV(>1 arcmin)** 제공. 장시간 statistics 관측에 유용.

**GLAO** uses a single ground-conjugated DM, delivering modest Strehl over a very wide FOV (>1′) — useful for long-duration statistics.

**태양 MCAO 초기 결과 / Early MCAO results**
- **DST MCAO 실증(Rimmele et al. 2010)**: 2-DM 배치로 60″ FOV에 걸쳐 균일한 AO 보정 시연.
- **KAOS/VTT**: 2-DM GLAO 프로토타입.
- **ATST/DKIST 설계**: 6 DM을 목표, 원뿔 효과(cone effect) 최소화를 위한 topology 연구.

Early demos: DST 2-DM MCAO (Rimmele et al. 2010) showed uniform correction across 60″; KAOS/VTT fielded a 2-DM GLAO prototype; ATST/DKIST targets a 6-DM MCAO.

---

### Part VIII: Conclusions & Outlook (§10) / 결론과 전망

- **SAO는 태양 물리학의 표준 기술**이 되었다. 상관 추적 SH의 개발로 가능.
- 4 m급 DKIST/EST에서는 MCAO가 필수가 될 것. $N_{\rm act} \sim 1000+$ 규모.
- **실시간 계산량** 급증 — GPU, FPGA, predictive controller(LQG, Kalman)가 필수.
- **편광 AO / 적외선 AO** 확장, **coronagraphic AO**(예: CoMP, DKIST Cryo-NIRSP) 결합 연구.

SAO has become standard; next-gen 4-m telescopes demand MCAO with $N_{\rm act}$ in the thousands. Real-time computing explodes; GPUs, FPGAs, and predictive controllers (LQG/Kalman) are essential. Polarimetric AO, IR AO, and coronagraphic AO are the frontiers.

---

## 3. Key Takeaways / 핵심 시사점

1. **AO is prerequisite, not enhancement / AO는 태양 관측의 전제조건** — 구경 $D \gg r_0$ 체제에서 AO 없이는 회절 한계 해상도가 원천적으로 불가능하다. 1 m 이상 태양망원경은 전부 AO에 의존한다. / Without AO, any solar telescope with $D > r_0$ cannot exploit its diffraction limit — SAO is foundational, not optional.

2. **Correlation tracking is the enabling breakthrough / 상관 추적이 결정적 돌파구** — Livingston의 아이디어를 Rimmele(1999)가 실시간 SH로 실현. 센트로이드 한계를 피함으로써 별 AO와 태양 AO의 격차를 해소. / Rimmele's 1999 real-time correlation SH closed a decades-long gap between stellar and solar AO by abandoning centroid estimation in favor of image cross-correlation.

3. **Error budget is additive in variance / 오차는 분산의 합** — Fitting, temporal, noise, anisoplanatic errors가 독립적으로 더해져 $\sigma_{\rm tot}^2$를 구성. Strehl $S \approx e^{-\sigma_{\rm tot}^2}$. 각 항의 스케일링이 시스템 설계(액추에이터 간격, 루프 속도, FOV)를 직접 결정. / Residual variance sums independently across error sources, directly prescribing actuator spacing ($d \lesssim r_0/2$), loop bandwidth ($\gg f_G$), and FOV ($\theta \lesssim \theta_0$).

4. **High-altitude turbulence constrains FOV / 고고도 난류가 FOV를 제한** — $\theta_0$는 $h^{5/3}$ 가중 적분으로 상층 난류가 지배적. 단일 DM AO는 isoplanatic patch 바깥에서 급격히 성능이 저하된다. / Because $\theta_0$'s integrand carries $h^{5/3}$, high-layer turbulence dominates anisoplanatism, making MCAO necessary for wide-field correction.

5. **Infrared AO is far easier / 적외선 AO는 훨씬 쉽다** — $r_0 \propto \lambda^{6/5}$이므로 파장이 길수록 seeing이 완화된다. 500 nm에서 $r_0=10$ cm라면 1.6 μm에서 $r_0 \approx 42$ cm. 태양 IR 분광(1.56 μm Fe I)이 AO 도입 초기부터 큰 성과를 낸 이유. / $r_0 \propto \lambda^{6/5}$ makes IR observations far more forgiving; a 10 cm $r_0$ at 500 nm becomes 42 cm at 1.56 μm — explaining the early success of IR solar AO.

6. **Ground-layer turbulence dominates daytime seeing / 지표층 난류가 주간 seeing의 주범** — 돔/경면 열 설계, 사이트 선정(고산·해안 풍상)이 난류 프로파일을 결정. 이는 단순히 공학 이슈가 아니라 과학 성과의 상한선을 정한다. / Daytime seeing is ground-layer-dominated; dome/mirror thermal design and site selection are not engineering details but hard limits on achievable science.

7. **AO and post-processing are complementary / AO와 후처리는 상호보완** — AO는 Strehl을 0.5 근처까지 끌어올리고, MOMFBD나 phase diversity가 잔차를 제거한다. SST의 대표 이미지들은 모두 AO + 재구축 조합. / AO pushes Strehl to ~0.5; post-facto reconstruction (MOMFBD, phase diversity) cleans the residuals. The SST's signature imagery depends on both.

8. **MCAO is the gateway to 4-m class SAO / MCAO가 4 m급 태양 AO의 문** — DKIST/EST 규모에서는 전통 AO의 $\theta_0$ 한계가 치명적. MCAO와 GLAO가 활동영역 전체 FOV에서 AO를 유지하는 유일한 경로이며, 2010년대 태양 관측의 핵심 기술 축으로 자리잡았다. / At 4-m aperture, the classical $\theta_0$ ceiling is fatal; MCAO and GLAO are the only paths to AR-scale corrected FOV and became the defining technology thrust of the 2010s for solar observing.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Turbulence parameters

$$
\boxed{r_0 = \left[ 0.423\, k^2 \sec\zeta \int_0^\infty C_n^2(h)\,dh \right]^{-3/5}}, \quad k = 2\pi/\lambda
$$

$$
\sigma_\phi^2\big|_{D} = 1.03 \left(\frac{D}{r_0}\right)^{5/3}\ \text{rad}^2
$$

$$
\theta_0 = \left[ 2.914\, k^2 \sec^{8/3}\zeta \int_0^\infty C_n^2(h)\, h^{5/3}\,dh \right]^{-3/5}
$$

$$
f_G = \left[ 0.102\, k^2 \sec\zeta \int_0^\infty C_n^2(h)\, v^{5/3}(h)\,dh \right]^{3/5}
$$

$$
\tau_0 \approx 0.314\, r_0 / \bar{v} \quad (\bar{v}\ \text{turbulence-weighted wind speed})
$$

### 4.2 Shack–Hartmann correlation tracking

$$
\boxed{C_i(\mathbf{s}) = \sum_{\mathbf{x}} I_i(\mathbf{x})\, R(\mathbf{x} - \mathbf{s})}, \qquad \hat{\mathbf{s}}_i = \arg\max_{\mathbf{s}} C_i(\mathbf{s})
$$

Alternate (ADS) metric:

$$
D_i(\mathbf{s}) = \sum_{\mathbf{x}} \big[ I_i(\mathbf{x}) - R(\mathbf{x}-\mathbf{s}) \big]^2, \qquad \hat{\mathbf{s}}_i = \arg\min_{\mathbf{s}} D_i(\mathbf{s})
$$

Subpixel refinement via 2D parabolic fit around the peak:

$$
\delta_x = \frac{C(-1,0) - C(+1,0)}{2\,[C(-1,0) - 2C(0,0) + C(+1,0)]}
$$

### 4.3 Reconstruction and control

Measurement model:

$$
\mathbf{s} = H\, \boldsymbol{\phi} + \mathbf{n}
$$

Minimum-variance reconstruction:

$$
R = (H^T C_n^{-1} H + \lambda\, I)^{-1} H^T C_n^{-1}
$$

Integrator closed loop with gain $g$:

$$
\mathbf{a}_{k+1} = \mathbf{a}_k + g\, R\, \mathbf{s}_k, \qquad 0 < g \lesssim 0.5
$$

Temporal error power spectrum and residual variance:

$$
\sigma_{\rm temp}^2 \approx (f_G / f_{3\text{dB}})^{5/3}
$$

### 4.4 Error budget & Strehl

$$
\sigma_{\rm fit}^2 = 0.28 \left(\frac{d}{r_0}\right)^{5/3}, \quad \sigma_{\rm aniso}^2 = \left(\frac{\theta}{\theta_0}\right)^{5/3}
$$

$$
\sigma_{\rm noise}^2 \propto \frac{N_{\rm modes}\,(\sigma_{\rm ph}^2 + \sigma_{\rm det}^2)}{(c\, N_{\rm pix})^2}
$$

$$
\boxed{S \approx \exp\!\big(-(\sigma_{\rm fit}^2 + \sigma_{\rm temp}^2 + \sigma_{\rm noise}^2 + \sigma_{\rm aniso}^2 + \sigma_{\rm calib}^2)\big)}
$$

### 4.5 Worked example (DST, 500 nm) / 수치 예제

- $D=0.76$ m, $r_0=10$ cm → $\sigma_\phi^2 = 1.03\,(7.6)^{5/3} \approx 30$ rad².
- Fit with $d \simeq 8.7$ cm (76 subap over 0.76 m) → $\sigma_{\rm fit}^2 = 0.28\,(0.87)^{5/3} \approx 0.22$ rad².
- Temporal at $f_{3\rm dB}/f_G = 5$ → $\sigma_{\rm temp}^2 \simeq 5^{-5/3} \approx 0.068$ rad².
- Noise + calib ~ 0.1 rad².
- Total $\sigma_{\rm tot}^2 \approx 0.4$ rad² → $S \approx e^{-0.4} \approx 0.67$.

실제 DST의 on-sky Strehl (0.3–0.5)보다 낙관적인 추정 — 실전에는 aniso, dome seeing, calibration drift가 추가됨. / Predicts $S\approx 0.67$ vs on-sky 0.3–0.5 — real systems suffer additional anisoplanatic, dome, and calibration losses.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1941  Kolmogorov — turbulence cascade theory
1953  Babcock — proposes adaptive optics concept
1957  Tatarskii — wave propagation through turbulence
1966  Fried — defines r₀, isoplanatic angle
1977  Greenwood — AO control bandwidth theorem
1983  von der Lühe — correlation wavefront sensing (non-real-time)
1984  Dunn — first low-order solar AO experiments (Sac Peak)
1990s Keck, VLT — nighttime AO matures
1999  ★ Rimmele — real-time correlation-tracking SH (NSO)
2002  Scharmer — SST AO; penumbral dark-core discovery
2003  DST high-order AO (76-subap)
2008  GREGOR AO first light
2010  Rimmele et al. — DST 2-DM MCAO demonstration
2011  ★ Rimmele & Marino — THIS REVIEW (LRSP)
2013  DKIST / ATST construction begins
2016  EST preliminary design
2020  DKIST first light (4-m, MCAO-ready)
2024  DKIST Cryo-NIRSP science operations
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Babcock (1953) "The Possibility of Compensating Astronomical Seeing" | Foundational AO concept / AO 개념의 기원 | **Direct antecedent** — this review realizes the vision for solar applications. / 본 리뷰가 구현한 바로 그 비전. |
| Fried (1966) "Optical Resolution Through a Randomly Inhomogeneous Medium" | Defines $r_0$ and isoplanatic angle / $r_0$와 등각편차 각도 정의 | **Core formalism** — every error term in §4 uses Fried's definitions. / §4 모든 오차항의 기반. |
| Greenwood (1977) "Bandwidth specification for AO systems" | Temporal bandwidth theorem / 시간 대역폭 정리 | **Loop design law** — $f_G$ is used directly in §3 temporal error. / §3 temporal error 분석의 근거. |
| Rimmele (1999) "Solar Adaptive Optics" (SPIE) | First real-time correlation SH / 최초 실시간 상관 SH | **Enabling breakthrough** — makes solar AO practical; foundational for this entire review. / 본 리뷰의 가능 기반. |
| Roddier (1988) "Curvature sensing and compensation" | Alternative WFS concept / 대안 WFS | **Comparison baseline** — §4 discusses why curvature is weaker for extended sources. / §4 확장 광원에 대한 상대적 한계 비교. |
| Scharmer et al. (2002) "Dark cores in sunspot penumbral filaments" | SST science enabled by AO / AO가 가능케 한 SST 과학 | **Scientific justification** — exemplar of what SAO unlocks. / SAO가 가능케 한 대표 과학. |
| van Noort et al. (2005) MOMFBD | Post-processing complement / 후처리 보완 | **Partner technique** — §7 notes AO+MOMFBD synergy. / AO+재구축의 상보성. |
| Schmidt et al. (2012) GREGOR first-light | 1.5-m European SAO system / 1.5 m 유럽 SAO | **Contemporary system** — validates predictions of this review. / 본 리뷰 예측을 검증. |
| Rimmele et al. (2020) DKIST / ATST | 4-m MCAO-ready telescope / 4 m MCAO 대비 망원경 | **Direct descendant** — implements the roadmap laid out in §8. / §8 로드맵을 실현. |

---

## 7. References / 참고문헌

- Rimmele, T. R., & Marino, J., "Solar Adaptive Optics", *Living Reviews in Solar Physics*, 8, 2 (2011). [DOI: 10.12942/lrsp-2011-2]
- Babcock, H. W., "The Possibility of Compensating Astronomical Seeing", *PASP*, 65, 229 (1953).
- Fried, D. L., "Optical Resolution Through a Randomly Inhomogeneous Medium", *JOSA*, 56, 1372 (1966).
- Greenwood, D. P., "Bandwidth specification for adaptive optics systems", *JOSA*, 67, 390 (1977).
- Rimmele, T. R., "Solar adaptive optics", *Proc. SPIE*, 4007, 218 (2000).
- Roddier, F., "Curvature sensing and compensation: a new concept in adaptive optics", *Appl. Opt.*, 27, 1223 (1988).
- Scharmer, G. B. et al., "Dark cores in sunspot penumbral filaments", *Nature*, 420, 151 (2002).
- von der Lühe, O., "A study of a correlation tracking method to improve imaging quality of ground-based solar telescopes", *A&A*, 119, 85 (1983).
- van Noort, M., Rouppe van der Voort, L., & Löfdahl, M. G., "Solar Image Restoration by use of MOMFBD", *Solar Phys.*, 228, 191 (2005).
- Rimmele, T. R. et al., "Solar Multi-Conjugate Adaptive Optics at the Dunn Solar Telescope", *Proc. SPIE*, 7736 (2010).
- Rimmele, T. R. et al., "The Daniel K. Inouye Solar Telescope — Observatory Overview", *Solar Phys.*, 295, 172 (2020).
