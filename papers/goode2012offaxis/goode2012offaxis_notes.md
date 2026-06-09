---
title: "The 1.6 m Off-Axis New Solar Telescope (NST) in Big Bear"
authors: Philip R. Goode, Wenda Cao, and BBSO Team
year: 2012
journal: "Proc. SPIE, Vol. 8444, 844403"
doi: "10.1117/12.925494"
topic: Solar Observation / Ground-Based Telescopes
tags: [NST, GST, Big Bear, off-axis Gregorian, adaptive optics, AO-76, AO-308, MCAO, IRIM, VIM, FISS, CYRA, diffraction-limited, polarimetry, seeing, Fried parameter, isoplanatic angle]
status: completed
date_started: 2026-04-13
date_completed: 2026-04-13
---

# The 1.6 m Off-Axis New Solar Telescope (NST) in Big Bear

## 핵심 기여 / Core Contribution

이 논문은 Big Bear Solar Observatory(BBSO)의 **1.6m New Solar Telescope(NST)**의 설계, 커미셔닝, 초기 관측 결과를 기술한다. NST는 미국에서 한 세대 만에 건설된 최초의 "facility class" 태양 망원경이며, 2009년 첫 빛 달성 후 당시 세계 최대 운용 태양 망원경이었다. 핵심 혁신은 세 가지이다: (1) **off-axis Gregorian 반사 설계**로 중앙 차폐를 완전히 제거하여 높은 대비와 깨끗한 MTF(Modulation Transfer Function) 확보; (2) **두 세대의 적응광학**(초기 AO-76 → 고차 AO-308)을 단계적으로 통합하여 가시광~근적외선에서 회절 한계 달성; (3) Big Bear Lake 호수 위 독특한 입지를 최대한 활용한 **열 관리와 MCAO(Multi-Conjugate AO) 개발**. NST는 ATST(현 DKIST, 4m)의 사실상 pathfinder로서 off-axis 설계의 실증과 AO 기술 개발 플랫폼 역할을 수행했다.

This paper describes the design, commissioning, and early observational results of the **1.6 m New Solar Telescope (NST)** at Big Bear Solar Observatory (BBSO). NST was the first "facility class" solar telescope built in the U.S. in a generation. After achieving first light in 2009, it was the world's largest operational solar telescope. Its three key innovations are: (1) an **off-axis Gregorian reflective design** that completely eliminates central obscuration, providing high contrast and a clean MTF; (2) **two generations of adaptive optics** (initial AO-76 → high-order AO-308) integrated progressively, achieving diffraction-limited imaging from visible to NIR; (3) maximizing Big Bear Lake's unique site conditions through **thermal management and MCAO development**. NST served as the de facto pathfinder for ATST (now DKIST, 4 m), demonstrating off-axis design feasibility and serving as an AO technology development platform.

---

## 읽기 노트 / Reading Notes

### 1. 서론 — NST의 목적과 위상 / Introduction — NST's Purpose and Position

논문은 NST의 첫 빛(2009년 1월)과 이후 3년간의 커미셔닝 과정에서 점진적으로 달성한 성능을 보고한다. Table 1에 요약된 마일스톤이 이 과정을 명확히 보여준다:

The paper reports NST's first light (January 2009) and the performance progressively achieved during three years of commissioning. Table 1 milestones illustrate this process:

| 날짜 / Date | 마일스톤 / Milestone | 분해능 / Resolution |
|---|---|---|
| 2009.01 | 첫 빛 (First Light) | 0".50 |
| 2009 여름 | 첫 회절 한계 관측 (First Diffraction Limited) | 0".12 |
| 2010 여름 | AO 보정 관측 (AO Corrected) | 0".10–0".20 |
| 2011 여름 | AO 보정 자기장 관측 (AO Corrected Magnetograms) | 0".40 |
| 2012 여름 | **AO-308 관측** (AO-308 Observations) | **0".05** |
| 2013 여름 | **첫 MCAO 관측** (First MCAO Observations) | 0".05 |
| 2013 여름 | CYRA 첫 빛 (CYRA First Light) | 0".15–0".80 |

분해능이 0".50 → 0".05로 **10배** 개선된 것은 망원경 자체가 아니라 **AO 시스템의 진화**에 의한 것이다. 이는 현대 태양 망원경에서 AO가 얼마나 결정적인지를 극명하게 보여준다.

The 10× improvement from 0".50 to 0".05 was driven not by the telescope itself but by **the evolution of the AO system**. This starkly demonstrates how critical AO is for modern solar telescopes.

NST의 두 가지 주요 역할:

NST's two major roles:

1. **ATST(DKIST)의 pathfinder**: 둘 다 off-axis 설계를 채택. NST는 "ATST 설계와 구현의 모든 측면에 대한 이상적인 시험대(ideal testbed in all manner of ATST design and implementation issues)"
   **Pathfinder for ATST (DKIST)**: both adopt off-axis design. NST is "the ideal testbed in all manner of ATST design and implementation issues"

2. **ATST 이후에도 보완적 역할**: ATST가 초과 구독(oversubscribed)되므로 NST가 대형 구경이 필요 없는 관측에 기여하고, 소규모 대학/연구자에게 접근성 제공
   **Complementary role even after ATST**: NST serves observations not requiring ATST-size aperture and provides accessibility for smaller institutions

논문에서 인용할 만한 Fig. 1이 인상적이다 — TiO 705.7nm 흑점 이미지가 National Geographic "2010년 최고 우주 사진 10선"에 선정되었다.

The paper's Fig. 1 is striking — the TiO 705.7 nm sunspot image was selected as one of National Geographic's "top ten space images of 2010."

### 2. 망원경 설계 — Off-Axis Gregorian / Telescope Design — Off-Axis Gregorian

**광학 구성 / Optical Configuration:**
- **Off-axis Gregorian**: 포물면 주경(parabolic primary) + 타원면 부경(elliptical secondary) + heat reflector + diagonal flats
  Off-axis Gregorian: parabolic primary + elliptical secondary + heat reflector + diagonal flats
- 주경(PM) 직경 1.7 m, **유효 구경(clear aperture) 1.6 m**
  Primary mirror diameter 1.7 m, **clear aperture 1.6 m**
- 초점비: PM은 $f/2.4$, 최종 $f/52$
  Focal ratio: PM is $f/2.4$, final $f/52$
- 시야각(FOV): field stop의 100" 원형 개구로 정의, **최대 70"×70" 정사각형 시야**
  FOV: defined by 100" circular opening in field stop, **maximum 70"×70" square FOV**
- 파장 범위: 0.4–1.7 μm (가시광 ~ 근적외선)
  Wavelength range: 0.4–1.7 μm (visible to NIR)

**왜 Off-Axis인가? / Why Off-Axis?**

이것은 이 논문의 핵심 설계 결정이며, SST(Paper #3)와의 근본적 차이점이다:

This is the paper's key design decision and the fundamental difference from SST (Paper #3):

- 중앙 차폐가 없으므로 **산란광(stray light)이 대폭 감소**
  No central obscuration means **vastly reduced stray light**
- MTF(Modulation Transfer Function)가 **고공간주파수에서 저하되지 않음** → 대비(contrast) 극대화
  MTF is **not degraded at high spatial frequencies** → contrast maximized
- 태양 관측에서 대비는 분해능만큼 중요하다 — 밝은 태양면 위의 미세한 구조를 보려면 높은 대비가 필수
  In solar observation, contrast is as important as resolution — high contrast is essential to see fine structures on the bright solar surface

**SST와의 설계 철학 비교 / Design Philosophy Comparison with SST:**

| 특성 / Feature | SST (Paper #3) | NST (본 논문) |
|---|---|---|
| 광학 형식 / Type | 굴절식 (Singlet + Schupmann) / Refractive | **반사식 (Off-axis Gregorian)** / Reflective |
| 구경 / Aperture | 1.0 m | **1.6 m** |
| 중앙 차폐 / Obscuration | 없음 (singlet이므로) / None | **없음 (off-axis이므로)** / None |
| 색수차 / Chromatic aberr. | 있음 → Schupmann으로 보정 / Yes → corrected | **없음 (반사식)** / None (reflective) |
| 열 관리 / Thermal | 진공 시스템 / Vacuum | **Field stop + heat reflector** |
| AO | 저차 (~37) / Low-order | **고차 (76→308)** / High-order |
| 편광 / Polarimetry | Singlet이 유리 (편광 왜곡 적음) | Off-axis 거울 → **편광 교정 필요** |

1.6m 이상의 구경에서는 **굴절식 설계가 현실적으로 불가능**하다 — 대형 광학 유리의 균질성 확보와 자중에 의한 변형 문제 때문이다. 따라서 NST가 반사식을 채택한 것은 구경 확대의 필연적 결과이다.

At apertures of 1.6 m and above, **refractive designs become impractical** — due to difficulty in achieving homogeneity of large optical glass and self-weight deformation. NST's choice of reflective design is an inevitable consequence of aperture scaling.

### 3. 주경 제작 — GMT의 시험대 / Primary Mirror Fabrication — GMT Test Bed

주경 제작 과정은 이 논문에서 가장 구체적인 공학적 내용 중 하나이다:

The primary mirror fabrication is one of the most specific engineering topics in this paper:

- Steward Observatory Mirror Lab(Tucson)에서 연마 — **Giant Magellan Telescope(GMT)의 시험대**로 사용
  Figured at Steward Observatory Mirror Lab, Tucson — used as a **test bed for the Giant Magellan Telescope (GMT)**
- GMT의 8.4m 주경은 off-axis 8.4m 세그먼트 6개로 구성. NST의 1.7m PM은 GMT의 **1/5 스케일** 단일 거울
  GMT's primary consists of six off-axis 8.4 m segments. NST's 1.7 m PM is a **1/5 scale** single mirror
- 원래 계획: CGH(Computer Generated Hologram)로 시험 간섭계의 wavefront를 변경하여 off-axis 비구면 검증
  Original plan: use CGH to alter test interferometer wavefront for off-axis asphere verification

**제작 중 겪은 문제와 해결 / Problems Encountered and Solutions:**
- 일 단위로 ~150 nm의 저차 광학 수차 변동이 관찰됨
  Daily variations of ~150 nm in low-order optical aberrations observed
- 원인: 간섭계와 CGH의 정렬이 매일 변동
  Cause: interferometer and CGH alignment drifted daily
- 해결: 레이저 트래커로 CGH와 간섭계를 고정 위치에 "볼" 수 있도록 함
  Solution: laser tracker to fix CGH and interferometer in position
- 최종적으로 **두 가지 독립적 시험** 방법 사용:
  Ultimately used **two independent test** methods:
  1. CGH + 간섭계 (저차 수차에 민감) / CGH + interferometer (sensitive to low-order)
  2. **Pentaprism scan** (저차 수차에 민감) / Pentaprism scan (sensitive to low-order aberrations)
- 두 방법의 합의 결과: **잔류 RMS figure error 16 nm**
  Agreement of both methods: **residual RMS figure error of 16 nm**

이 16 nm RMS는 파장 500nm 기준 $\lambda/31$로서, 주경 자체만으로 회절 한계(Maréchal 기준 $\lambda/14$ RMS)를 충분히 만족한다.

This 16 nm RMS is $\lambda/31$ at 500 nm, well within the diffraction limit (Maréchal criterion: $\lambda/14$ RMS) from the primary alone.

### 4. AO-76 — 첫 번째 세대 적응광학 / AO-76 — First Generation Adaptive Optics

NST의 AO 이야기는 두 세대에 걸쳐 전개된다. 먼저 AO-76:

NST's AO story unfolds across two generations. First, AO-76:

**시스템 구성 / System Configuration:**
- 97 actuator deformable mirror (DM)
- 76 서브-애퍼처 Shack-Hartmann wavefront sensor
- 디지털 신호 처리(DSP) 시스템
- 구조적으로 DST(Dunn Solar Telescope, Paper #2 관련)의 "AO-76"과 동일
  Structurally the same as DST's "AO-76"

**AO-76의 성능과 한계 / AO-76 Performance and Limitations:**

논문에서 가장 솔직한 부분 중 하나가 AO-76의 한계에 대한 논의이다:

One of the paper's most candid sections discusses AO-76 limitations:

- AO-76은 **sub-meter급 망원경을 위해 설계**된 시스템을 1.6m에 적용한 것
  AO-76 was designed for **sub-meter class telescopes** adapted to 1.6 m
- Granulation 추적(locking) 가능, 그러나 **pore(작은 암점)에서만** — 즉 고대비 특징이 필요
  Can track (lock on) granulation, but **only on pores** — needs high-contrast features
- 정상 BBSO 시상(seeing)에서의 Fried parameter $r_0 \sim 6$ cm (가시광, 정상 상태 평균)
  Fried parameter under steady BBSO seeing: $r_0 \sim 6$ cm (visible, steady-state average)
- AO-76으로 보정 가능한 모드 수: 약 $0.97 \times (160/6)^2 \approx 690$ 모드가 필요한데, 76 서브-애퍼처로는 **턱없이 부족**
  Required correction modes: ~$0.97 \times (160/6)^2 \approx 690$, far exceeding 76 sub-apertures

**파장별 성능 차이 / Wavelength-Dependent Performance:**

Rimmele(2008)의 오차 분석에 기반한 AO-76의 Strehl ratio 예측:

Predicted Strehl ratios from AO-76 based on Rimmele (2008) error budget:

| 파장 / Wavelength | Strehl (중간 시상) / Strehl (median seeing) | 평가 / Assessment |
|---|---|---|
| 1 μm (NIR) | ~0.7 | **높음** — 근적외선에서 우수 / High — good in NIR |
| 0.5 μm (가시광) | ~0.3 | **낮음** — 회절 한계(0.8) 미달 / Low — below diffraction limit |

이것이 핵심이다: AO-76으로는 **근적외선에서는 괜찮지만 가시광에서는 회절 한계에 도달하지 못한다**. 특히 편광 측정(polarimetry)에서 이것이 문제가 된다 — 반대 편광의 차이 측정에서 낮은 Strehl은 해석 불가능한 결과를 만든다.

This is the key point: AO-76 is **adequate in NIR but cannot reach diffraction limit in visible**. This is especially problematic for polarimetry — low Strehl makes opposite-polarity difference measurements difficult to impossible to interpret.

### 5. AO-308 — 차세대 고차 적응광학 / AO-308 — Next-Generation High-Order AO

AO-76의 한계를 극복하기 위해 개발된 AO-308:

AO-308, developed to overcome AO-76 limitations:

**시스템 구성 / System Configuration:**
- **357 actuator DM** (Xinetics), 실리콘 faceplate — ULE 대비 열 전도율 100배
  357-actuator DM from Xinetics with silicon faceplate — 100× thermal conductivity of ULE
- **308 서브-애퍼처** Shack-Hartmann wavefront sensor
  308 sub-aperture Shack-Hartmann wavefront sensor
- Bittware DSP (AO-76 대비 **10배 빠름**)
  Bittware DSP (**10× faster** than AO-76)
- Phantom V7.3 wavefront sensing camera (Vision Research) — Quasar 인터페이스
  Phantom V7.3 wavefront sensing camera with Quasar interface

**왜 실리콘 faceplate인가? / Why Silicon Faceplate?**

DM의 faceplate 재질 선택은 태양 관측 AO의 고유한 문제를 해결한다:

DM faceplate material addresses a problem unique to solar AO:

- 태양광이 DM에 직접 집중 → DM 표면이 가열됨
  Sunlight concentrates on the DM → DM surface heats up
- ULE(Ultra-Low Expansion glass)는 열 전도율이 낮아 열이 국소적으로 축적
  ULE has low thermal conductivity, causing local heat accumulation
- 실리콘은 열 전도율이 ULE의 **100배** → 열을 빠르게 확산/방출
  Silicon has **100×** the thermal conductivity of ULE → rapidly dissipates heat

**AO-308의 예상 성능 / Expected Performance of AO-308:**
- 중간 BBSO 시상(0.5 μm 기준)에서 Strehl ~0.3 → 현실적으로 **가시광 회절 한계 달성은 예외적 시상에서만 가능**
  Under median BBSO seeing (at 0.5 μm): Strehl ~0.3 → diffraction-limited in visible **only under exceptional seeing**
- 그러나 NIR에서는 회절 한계를 안정적으로 달성하며, G-band 등 적색~NIR 관측에서 큰 개선 예상
  However, stably diffraction-limited in NIR, with major improvement expected for G-band and red/NIR observations

논문 시점(2012)에서 AO-308은 벤치 테스트 완료 후 여름에 AO-76과 병렬 운용 예정이었다.

At the time of the paper (2012), AO-308 had completed bench testing and was scheduled for parallel operation with AO-76 in summer.

### 6. Big Bear 사이트와 대기 프로파일 / Big Bear Site and Atmospheric Profiles

이 섹션은 MCAO 개발의 근거를 제시하며, Fig. 2의 대기 프로파일이 핵심 데이터이다:

This section provides the basis for MCAO development, with Fig. 2 atmospheric profiles as key data:

**Big Bear의 사이트 특성 / Big Bear Site Characteristics:**
- 해발 2,060 m의 호수 위에 위치 — 낮 시간 대기 시상(seeing)이 우수
  Located on a lake at 2,060 m elevation — excellent daytime seeing
- ATST 사이트 조사 결과: $r_0$ 중앙값이 가시광에서 약 6 cm
  ATST site survey results: median $r_0$ ~6 cm in visible
- Kellerer et al. (2012): **여름이 겨울보다 시상이 훨씬 좋음** — $r_0 = 9.1 \pm 3.3$ cm (여름) vs $r_0 = 5.5 \pm 0.8$ cm (겨울)
  **Seeing much better in summer than winter** — $r_0 = 9.1 \pm 3.3$ cm (summer) vs $r_0 = 5.5 \pm 0.8$ cm (winter)

**Fig. 2 대기 프로파일 분석 / Fig. 2 Atmospheric Profile Analysis:**

2012년 여름 이틀간의 프로파일 비교가 매우 교훈적이다:

Comparison of two days' profiles in summer 2012 is very instructive:

- **6월 22일**: jet stream이 BBSO 북쪽 → 시상 양호, 평균 isoplanatic angle 1".6 (5".5)
  June 22: jet stream well north of BBSO → good seeing, average isoplanatic angle 1".6 (5".5)
- **6월 28일**: jet stream이 BBSO 위 → 시상 악화, 평균 isoplanatic angle 2".0 (2".5)
  June 28: jet stream over BBSO → worse seeing, average isoplanatic angle 2".0 (2".5)
- jet stream의 위치(Vandenberg AFB에서 ~300 km 서쪽으로 추적)로 관측일 선정 가능
  Jet stream location (tracked ~300 km west from Vandenberg AFB) can guide observing day selection

**Isoplanatic angle의 의미 / Meaning of Isoplanatic Angle:**

$\theta_0$는 AO 보정이 유효한 각도 범위이다. 태양면에서 수 arcsec에 불과하므로, 단일 DM으로는 **70"×70" 시야 전체를 보정할 수 없다**. 이것이 MCAO가 필요한 근본적 이유이다.

$\theta_0$ is the angular range over which AO correction is valid. At only a few arcseconds on the Sun, a single DM **cannot correct the entire 70"×70" FOV**. This is the fundamental reason MCAO is needed.

### 7. MCAO — 미래 기술의 시작 / MCAO — Beginning of Future Technology

Fig. 3은 MCAO의 효과를 시뮬레이션한 결과로, 논문의 가장 야심찬 내용이다:

Fig. 3 shows simulated MCAO effectiveness, the paper's most ambitious content:

**MCAO 컨셉 / MCAO Concept:**
- 대기 난류는 여러 고도(지면, 3km, 6km 등)에 분포
  Atmospheric turbulence is distributed across multiple altitudes (ground, 3 km, 6 km, etc.)
- 각 고도의 난류를 **별도의 DM으로 보정** → 보정 시야 확대
  Correct turbulence at each altitude with **separate DMs** → expand corrected FOV
- "Conjugation(공역)"이란 DM을 특정 고도의 난류층과 광학적으로 대응시키는 것
  "Conjugation" means optically matching a DM to a turbulence layer at a specific altitude

**Fig. 3의 시뮬레이션 결과 / Fig. 3 Simulation Results:**

550개의 여름+겨울 대기 프로파일을 기반으로 $C_N^2$ 모델링 후 MCAO 효과를 계산:

MCAO effect calculated from $C_N^2$ modeling based on 550 summer + 311 winter profiles:

| 구성 / Configuration | DM 공역 고도 / Conjugation | 효과 / Effect |
|---|---|---|
| No AO | — | 기준선 / Baseline |
| Ground-layer AO | 지면(ground)만 | 가장 큰 개선의 첫 단계 / First big improvement |
| 2-mirror MCAO | 지면 + 3 km | 상당한 추가 개선 / Substantial additional gain |
| **3-mirror MCAO** | **지면 + 3 km + 6 km** | **최대 효과, 하지만 3번째 DM의 추가 이득은 제한적** |

**핵심 발견 / Key Finding:**
- **최대 이득은 지면층(ground layer) 보정**에서 온다
  **Maximum gain comes from ground-layer correction**
- 2번째 DM은 약 3 km에 공역시키는 것이 최적
  Second DM optimally conjugated at ~3 km
- 3번째 DM: 트로포포즈(6 km) 근처가 적합, 이론적으로 높은(약 10 km) jet stream 고도보다 낮은 곳이 더 효과적
  Third DM: near tropopause (~6 km) is suitable; lower altitude is more effective than the theoretical ~10 km jet stream altitude
- 겨울보다 여름에 MCAO 효과가 큼 — 여름에 고층 대기가 더 안정적이기 때문
  MCAO more effective in summer — upper atmosphere more stable in summer

### 8. 초점면 기기 — IRIM / Focal Plane Instrumentation — IRIM

NST의 "workhorse(주력)" 기기는 **IRIM(InfraRed Imaging vector Magnetograph)**이다:

NST's workhorse instrument is **IRIM**:

**IRIM 설계와 성능 / IRIM Design and Performance:**
- 근적외선(NIR) 1565 nm에서의 **이중 빔(dual-beam) 편광 측정**
  **Dual-beam polarimetry** at NIR 1565 nm
- Stokes I, Q, U, V 모두 보정된 벡터 자기장 측정(Fig. 4)
  Calibrated vector magnetograms of all Stokes I, Q, U, V (Fig. 4)
- 시야각: 약 50"×25"
  FOV: ~50"×25"
- 편광 정확도: 약 $10^{-3} I_c$
  Polarization accuracy: ~$10^{-3} I_c$
- 공간 분해능: **0".2 (직접 영상), 0".4 (자기장 측정)**
  Spatial resolution: **0".2 (direct imaging), 0".4 (magnetograms)**
- 벡터 자기장 측정 cadence: 최소 45 s (스펙트럼 위치 수에 따라)
  Vector magnetogram cadence: as short as 45 s (depending on spectral positions)

**NST의 편광 측정 우위 / NST's Polarimetric Advantage:**

논문에서 강조하는 중요한 점:

A key point emphasized in the paper:

- 이전 0.6m BBSO 망원경에서는 심한 편광 문제로 벡터 자기장 측정이 불가능했음
  Previous 0.6 m BBSO telescope had severe polarization problems, making vector magnetograms impossible
- NST의 **off-axis 설계**와 우수한 편광 교정으로 **Stokes Q, U에서 line-of-sight 성분의 cross-talk이 없음** (Fig. 4에서 확인)
  NST's **off-axis design** and excellent polarization calibration yield **no apparent cross-talk from line-of-sight component into Stokes Q, U** (confirmed in Fig. 4)

**NIR의 추가 이점 / Additional Advantages of NIR:**
- NIR에서의 대기 $r_0$는 가시광의 약 **4배** (Kolmogorov 모델 가정 시 $r_0 \propto \lambda^{6/5}$)
  Atmospheric $r_0$ in NIR is ~**4×** that in visible (assuming Kolmogorov model: $r_0 \propto \lambda^{6/5}$)
- 따라서 AO-76만으로도 NIR에서 회절 한계 달성 가능 → IRIM이 주력 기기가 된 이유
  Thus AO-76 alone can achieve diffraction limit in NIR → reason IRIM became the workhorse
- Zeeman splitting이 파장에 **2차적으로 비례** ($\Delta\lambda \propto \lambda^2 g B$) → NIR에서 약한 자기장 검출에 유리
  Zeeman splitting scales **quadratically** with wavelength → NIR advantageous for detecting weak fields

**IRIM 업그레이드 → NIRIS / IRIM Upgrade → NIRIS:**
- IRIM을 dual FPI(Fabry-Pérot Interferometer) 시스템으로 업그레이드 → **NIRIS**
  IRIM upgraded to dual FPI system → **NIRIS**
- 파장 범위: 1.0–1.7 μm, 분해능 0".2–0".3
  Wavelength: 1.0–1.7 μm, resolution 0".2–0".3
- 10830 Å He I 라인 관측 가능 — 색층(chromosphere) 자기장 측정
  Can observe 10830 Å He I line — chromospheric magnetic field measurement

### 9. 추가 기기 — VIM, FISS, CYRA / Additional Instruments — VIM, FISS, CYRA

**VIM (Visible Imaging Magnetograph):**
- 가시광 분광 편광 측정 + 시선 방향 자기장(line-of-sight magnetograms)
  Visible spectropolarimetry + line-of-sight magnetograms
- IRIM/VIM이 같은 설계 철학과 운용 모드 → NST에 동시 통합 가능
  IRIM/VIM share design philosophy and operation modes → simultaneous integration on NST
- AO-308 필요: 가시광에서 최고 분해능(0".1) 벡터 자기장 달성 목표
  Requires AO-308: targets 0".1 resolution vector magnetograms in visible

**FISS (Fast Imaging Solar Spectrograph):**
- 고분산 슬릿 분광기, 분해능 $1.4 \times 10^5$, 빠른 스캔(10 s)
  High-dispersion slit spectrograph, resolving power $1.4 \times 10^5$, fast scan (10 s)
- 시야: 40"×60"
  FOV: 40"×60"
- 주요 스펙트럼 라인: Hα, Ca II H, K, Ca II 8542 Å — 색층 역학(chromospheric dynamics) 연구
  Key spectral lines: Hα, Ca II H, K, Ca II 8542 Å — chromospheric dynamics research
- **2대의 동일 CCD**로 2개 파장 대역 동시 관측
  Two identical CCDs for simultaneous observation at two wavelength bands

**CYRA (CrYogenic infraRed spectrograph):**
- 1.0–5.0 μm 적외선 분광기 — **완전 극저온 냉각**
  1.0–5.0 μm infrared spectrograph — **fully cryogenic**
- 기존 IR 분광기(McMath-Pierce, Mees)는 검출기와 필터만 냉각, 광학계는 상온 → 열 배경 높음
  Existing IR spectrographs cool only detectors/filters, optics at ambient → high thermal background
- CYRA는 광학계 전체를 냉각 → 고감도, 고cadence 가능
  CYRA cools entire optics → high sensitivity, high cadence possible
- 광구(photosphere) → 색층(chromosphere) → 코로나(corona)를 연속적으로 탐사 가능
  Can probe photosphere → chromosphere → corona continuously

### 10. Post-facto 이미지 재구성 — KISIP과 Speckle / Post-facto Image Reconstruction — KISIP and Speckle

논문이 간략히 언급하지만 중요한 기술:

A technique the paper mentions briefly but is important:

- NST 이미지는 일상적으로 **speckle interferometry** 후처리를 거침
  NST images routinely undergo **speckle interferometry** post-processing
- KISIP(Kiepenheur-Institut für Sonnenphysik 소프트웨어): AO 보정 후 잔여 수차를 추가 제거
  KISIP software: removes residual aberrations after AO correction
- Burst 이미지(짧은 시간에 다수의 프레임)를 결합하여 **단일 회절 한계 이미지** 생성
  Combines burst images (many frames in short time) into a **single diffraction-limited image**
- AO 보정 + speckle = **isoplanatic patch보다 넓은 시야에서 회절 한계 달성** 가능
  AO correction + speckle = diffraction limit achievable **over a wider FOV than the isoplanatic patch**
- 단점: 약 5초의 시간 cadence가 필요 → sub-second 현상은 놓칠 수 있음
  Drawback: ~5 s cadence needed → sub-second phenomena may be missed
- 이 한계를 극복하려면 **MCAO**가 필요함
  Overcoming this limitation requires **MCAO**

---

## 핵심 시사점 / Key Takeaways

1. **Off-axis 설계는 대형 태양 망원경의 필수** — 1.6m 이상에서는 굴절식이 불가능하고, 반사식에서 중앙 차폐 제거가 대비와 MTF에 결정적. 이 설계가 DKIST(4m)에 직접 채택되었다.
   Off-axis design is essential for large solar telescopes — refractive designs impossible at ≥1.6 m, and obscuration removal is decisive for contrast/MTF. This design was directly adopted by DKIST (4 m).

2. **AO 차수(order)가 곧 과학적 성능** — AO-76(76 서브-애퍼처)은 NIR에서만 회절 한계를 달성하고 가시광에서는 실패. AO-308(308 서브-애퍼처)로 업그레이드해야 가시광 회절 한계 접근 가능. 분해능의 10배 개선(0".5→0".05)은 거울이 아닌 AO가 달성한 것이다.
   AO order directly determines scientific performance — AO-76 achieves diffraction limit only in NIR, fails in visible. AO-308 needed for visible diffraction limit. The 10× resolution improvement (0".5→0".05) was achieved by AO, not the mirror.

3. **Isoplanatic angle이 MCAO를 요구** — 단일 DM AO는 수 arcsec의 isoplanatic patch만 보정. 70"×70" 시야에서 균일한 회절 한계를 얻으려면 다중 고도의 난류를 각각 보정하는 MCAO가 필수적이다.
   Isoplanatic angle demands MCAO — single-DM AO corrects only a few arcsec isoplanatic patch. MCAO correcting turbulence at multiple altitudes is essential for uniform diffraction limit over 70"×70" FOV.

4. **NIR 편광 측정이 약한 자기장 검출의 핵심** — Zeeman splitting의 $\lambda^2$ 의존성과 대기 $r_0$의 $\lambda^{6/5}$ 의존성이 모두 NIR에 유리. IRIM/NIRIS가 NST의 주력 기기가 된 것은 물리적 필연이다.
   NIR polarimetry is key to weak-field detection — both Zeeman splitting's $\lambda^2$ dependence and atmospheric $r_0$'s $\lambda^{6/5}$ dependence favor NIR. IRIM/NIRIS becoming NST's workhorse was a physical inevitability.

5. **사이트가 기기만큼 중요** — Big Bear Lake의 호수 위 입지는 지면층 난류를 억제하고, jet stream 위치로 관측일을 선정할 수 있음. 여름 $r_0$가 겨울의 1.7배로, 계절적 관측 전략이 가능하다.
   Site matters as much as instrumentation — Big Bear Lake's over-lake location suppresses ground-layer turbulence, and jet stream position guides observing day selection. Summer $r_0$ is 1.7× winter, enabling seasonal observing strategies.

6. **NST는 DKIST의 pathfinder 이상** — off-axis 주경 제작(GMT 시험대), 고차 AO 개발, MCAO 시연, 다양한 초점면 기기 개발까지, NST에서 검증된 기술이 DKIST에 직접 이전되었다.
   NST was more than a DKIST pathfinder — from off-axis mirror fabrication (GMT test bed) to high-order AO, MCAO demonstration, and diverse focal plane instruments, technologies validated at NST were directly transferred to DKIST.

7. **Post-facto 재구성과 AO는 보완적** — KISIP/speckle이 AO 잔차를 추가 제거하여 isoplanatic patch 너머까지 확장하지만, sub-second cadence와 real-time 보정은 MCAO만이 제공할 수 있다.
   Post-facto reconstruction and AO are complementary — KISIP/speckle extends beyond the isoplanatic patch by removing AO residuals, but only MCAO can provide sub-second cadence and real-time correction.

---

## 수학적 요약 / Mathematical Summary

### 회절 한계 분해능 / Diffraction-Limited Resolution

$$\theta_{\text{diff}} = 1.22 \frac{\lambda}{D}$$

NST ($D = 1.6$ m): $\theta = 0.079''$ at $\lambda = 500$ nm, $\theta = 0.25''$ at $\lambda = 1.6$ μm

### Fried Parameter와 AO 모드 수 / Fried Parameter and AO Mode Count

$$N_{\text{modes}} \approx 0.97 \left(\frac{D}{r_0}\right)^2$$

| $r_0$ (cm) | $N_{\text{modes}}$ (D=1.6m) | AO-76 충분? | AO-308 충분? |
|---|---|---|---|
| 6 (중간 시상) | 690 | **아니오** | 부분적 |
| 10 (좋은 시상) | 248 | 아니오 | **예** |
| 30 (NIR 등가) | 28 | **예** | 예 |

### Fried Parameter의 파장 의존성 / Wavelength Dependence of Fried Parameter

$$r_0(\lambda) \propto \lambda^{6/5}$$

NIR(1.6 μm)에서의 $r_0$는 가시광(0.5 μm)의 약 $\left(\frac{1.6}{0.5}\right)^{6/5} \approx 4.0$배.

### Strehl Ratio (Maréchal 근사) / Strehl Ratio (Maréchal Approximation)

$$S \approx \exp\left(-\sigma_\phi^2\right)$$

회절 한계: $S \geq 0.8$ → $\sigma_\phi \leq 0.47$ rad → $\sigma_{\text{WFE}} \leq \lambda/13.4$ RMS

### Zeeman Splitting / 제만 분리

$$\Delta\lambda = \frac{e \lambda^2 g B}{4\pi m_e c}$$

$\lambda^2$ 의존성 → NIR(1565 nm)에서의 Zeeman splitting은 가시광(500 nm)의 약 $(1565/500)^2 \approx 9.8$배.

---

## 역사적 맥락의 타임라인 / Paper in the Arc of History

```
1964     1969     2002     2004     2009     2012     2013     2022
 │        │        │        │        │        │        │        │
 ▼        ▼        ▼        ▼        ▼        ▼        ▼        ▼
Pierce   BBSO     SST     NST     NST     ★ 본    MCAO    DKIST
(#1)     설립    1m 첫빛  프로젝트  첫빛    논문    시연    4m 운용
McMath   Big     Scharmer  시작   1.6m   Goode  NST→GST  개시
-Pierce  Bear    (#3)     Off-    AO-76   & Cao  개명
Tower    Lake            axis    통합
                        설계

────────────────────────────────────────────────────────────────►
  굴절 → 반사식 전환          적응광학 세대 교체        대형화
  Refractive → Reflective    AO generational shift    Scaling up
```

---

## 다른 논문과의 연결 / Connections to Other Papers

| 논문 / Paper | 연결 / Connection | 관련성 / Relevance |
|---|---|---|
| Pierce (1964) — McMath-Pierce (#1) | NST가 위치한 BBSO는 McMath-Pierce와 같은 미국 태양 관측 전통의 계승자. NST는 "미국에서 한 세대 만의 facility class 태양 망원경" | NST is the successor to the American solar observation tradition of McMath-Pierce — "first facility class solar telescope built in the U.S. in a generation" |
| Dunn (1964) — DST (#2) | AO-76이 DST의 AO 시스템과 구조적으로 동일. NST는 DST의 기술적 유산을 직접 계승 | AO-76 is structurally the same as DST's AO system. NST directly inherits DST's technical legacy |
| Scharmer (2003) — SST (#3) | 1m 굴절식 vs 1.6m 반사식의 설계 철학 대비. 둘 다 중앙 차폐 없음을 공유하지만 접근 방식이 근본적으로 다름 | 1 m refractive vs 1.6 m reflective design philosophy contrast. Both share no central obscuration but fundamentally different approaches |
| DKIST (Rimmele et al., 2020) | NST의 off-axis Gregorian 설계가 DKIST에 직접 채택. NST가 DKIST의 "이상적 시험대" 역할 수행 | NST's off-axis Gregorian design directly adopted by DKIST. NST served as DKIST's "ideal testbed" |
| GMT (Johns et al., 2012) | NST 주경이 Steward Mirror Lab에서 GMT의 1/5 스케일 시험품으로 제작됨 | NST primary fabricated at Steward Mirror Lab as a 1/5 scale test for GMT |
| Kellerer et al. (2012) | BBSO 사이트의 대기 프로파일 특성화, MCAO 개발의 근거 데이터 제공 | Characterized BBSO site atmospheric profiles, providing data basis for MCAO development |

---

## 참고문헌 / References

- Goode, P. R., Coulter, R., Gorceix, N., Yurchyshyn, V., and Cao, W., "The NST: First results and some lessons for ATST and EST," *Astron. Nachr.* **331**, 620–623 (2010).
- Cao, W., Gorceix, N., Coulter, R., Ahn, K., Rimmele, T. R., and Goode, P. R., "Scientific Instruments of 1.6 m New Solar Telescope in Big Bear," *Astron. Nachr.* **331**, 636–639 (2010).
- Wöger, F. and von der Lühe, O., "KISIP: a software package for speckle interferometry of adaptive optics corrected solar data," *Proc. SPIE* **7019**, 46–54 (2008).
- Kellerer, A., Gorceix, N., Marino, J., Cao, W., and Goode, P. R., "Profiles of the Daytime Atmospheric Turbulence above Big Bear Solar Observatory," *A&A* **542**, A2 (2012).
- Cao, W., Ahn, K., Goode, P. R., Shumko, S., Gorceix, N., and Coulter, R., "The New Solar Telescope in Big Bear: Polarimetry II," *ASP Conference Series* **437**, 345–349 (2011).
- Scharmer, G. B., et al., "The 1-m Swedish Solar Telescope," *Proc. SPIE* **4853**, 341–350 (2003).
- Rimmele, T. R., et al., "The Daniel K. Inouye Solar Telescope," *Solar Phys.* **295**, 172 (2020).
