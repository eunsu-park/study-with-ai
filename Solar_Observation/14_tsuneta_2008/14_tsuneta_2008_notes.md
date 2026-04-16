---
title: "Reading Notes: The Solar Optical Telescope for the Hinode Mission: An Overview"
paper_id: "14_tsuneta_2008"
topic: Solar_Observation
date: 2026-04-16
tags: [Hinode, SOT, space telescope, spectropolarimetry, solar magnetic field, diffraction-limited, image stabilization]
---

# The Solar Optical Telescope for the Hinode Mission: An Overview — Reading Notes / 읽기 노트

**Paper**: Tsuneta, S. et al., *Solar Physics*, Vol. 249, pp. 167–196, 2008
**DOI**: 10.1007/s11207-008-9174-z

---

## 1. 핵심 기여 / Core Contribution

Hinode/SOT는 우주에 발사된 최대 구경(50 cm)의 태양 광학 망원경으로, 회절 한계 분해능(0.2–0.3 arcsec)에서 광구와 채층의 고해상도 광도측정 및 벡터 자기장 관측을 처음으로 실현했습니다. SOT는 Optical Telescope Assembly(OTA)와 Focal Plane Package(FPP)로 구성되며, FPP에는 광대역 필터(BFI), 협대역 필터(NFI), Stokes 분광편광측정기(SP)가 포함됩니다. 핵심 기술 혁신은 (1) ULE 주경+부경을 사용한 비열팽창 Gregorian 설계, (2) 연속 회전 파장판(PMU)에 의한 편광 변조, (3) correlation tracker와 tip-tilt mirror를 결합한 0.007" rms 영상 안정화 시스템입니다. 대기 왜곡 없이 연속적이고 안정적인 PSF를 제공함으로써, 지상 관측에서는 불가능한 장기간(수일) 고해상도 관측을 가능하게 했습니다.

Hinode/SOT is the largest aperture (50 cm) solar optical telescope launched into space, achieving for the first time diffraction-limited (0.2–0.3 arcsec) high-resolution photometric and vector magnetic field observations of the photosphere and chromosphere. SOT consists of the Optical Telescope Assembly (OTA) and the Focal Plane Package (FPP), which includes broadband (BFI) and narrowband (NFI) filtergraphs plus a Stokes spectro-polarimeter (SP). Key technological innovations include: (1) a thermally stable Gregorian design using ULE primary and secondary mirrors, (2) polarization modulation via a continuously rotating waveplate (PMU), and (3) an image stabilization system combining a correlation tracker with a tip-tilt mirror achieving 0.007" rms stability. By providing continuous, stable PSF free from atmospheric distortion, SOT enables long-duration (multi-day) high-resolution observations impossible from the ground.

---

## 2. 읽기 노트 / Reading Notes

### Section 1: Introduction (pp. 167–169)

**과학적 동기 / Scientific Motivation**

Yohkoh의 X선 관측은 자기 재결합이 코로나 가열에 필수적임을 밝혔지만, 구체적인 코로나/채층 가열 메커니즘은 미해결이었습니다. 지상 관측에서 태양 자기장이 0.1–0.2 arcsec 규모의 미세 구조를 포함한다는 사실이 밝혀졌지만, 대기 seeing 때문에 정밀 측정이 제한되었습니다.

Yohkoh's X-ray observations revealed magnetic reconnection as essential for coronal heating, but the specific coronal/chromospheric heating mechanisms remained unsolved. Ground-based observations showed solar magnetic fields contain fine structures at 0.1–0.2 arcsec scales, but atmospheric seeing limited precise measurements.

**Hinode의 시스템 접근법 / Hinode's Systems Approach**

Hinode는 3개 망원경의 협동 관측으로 자기장의 생성-수송-소산 과정을 추적합니다:
Hinode uses coordinated observations with three telescopes to trace magnetic field generation-transport-dissipation:

| 망원경 / Telescope | 관측 영역 / Domain | 역할 / Role |
|---|---|---|
| SOT | 광구 + 채층 / Photosphere + Chromosphere | 자기 플럭스 출현·진화 / Magnetic flux emergence & evolution |
| EIS | 전이영역 + 코로나 / Transition region + Corona | 플라즈마 진단 / Plasma diagnostics |
| XRT | 코로나 / Corona | 에너지 방출·소산 / Energy release & dissipation |

**설계 기준선 (1995–96) / Design Baseline (1995–96)**

초기 개념 설계 단계에서 SOT는 50 cm 구경, 0.2–0.3 arcsec 회절 한계의 망원경에 filtergraph와 spectro-polarimeter를 모두 탑재하기로 결정되었습니다. 이는 지상 관측과의 과학적 이점과 기술적 제약 사이의 균형을 고려한 것입니다.

In the early concept design phase, SOT was defined as a 50 cm aperture, 0.2–0.3 arcsec diffraction-limited telescope carrying both a filtergraph and a spectro-polarimeter, balancing scientific advantage over ground-based observations against technical constraints.

**지상 망원경과의 비교 / Comparison with Ground-Based Telescopes**

| 특성 / Property | SOT | SST (1 m) | DST/VTT |
|---|---|---|---|
| 공간 분해능 / Spatial resolution | 0.2–0.3" (회절 한계) | ≈0.1" (적응광학) | ≈0.4–0.6" |
| PSF 안정성 / PSF stability | 매우 안정 / Very stable | 변동 / Variable | 변동 / Variable |
| 연속 관측 / Continuous obs. | 8개월/년 / 8 months/year | 주간만 / Daytime only | 주간만 / Daytime only |
| 편광 정밀도 / Polarimetric accuracy | 일정 / Constant | 대기 의존 / Atmosphere-dependent | 대기 의존 |

핵심 차별점: SOT는 분해능 자체는 SST보다 낮지만, **안정적 PSF + 연속 관측**이라는 결정적 이점을 제공합니다.
Key differentiator: SOT has lower resolution than SST, but provides the decisive advantage of **stable PSF + continuous observations**.

**태양 동기 궤도 / Sun-Synchronous Orbit**

Hinode의 태양 동기 궤도는 연간 약 8개월의 중단 없는 관측을 가능하게 하며, 망원경에 일정한 열 입력을 제공하여 광열 안정성을 확보합니다. ESA Svalbard 지상국을 통한 거의 매 궤도 데이터 다운링크가 가능하여 높은 시간 케이던스와 넓은 시야의 관측을 지원합니다.

Hinode's sun-synchronous orbit enables uninterrupted observations for ~8 months per year and provides constant thermal input to the telescope for opto-thermal stability. Data downlink through the ESA Svalbard station in nearly every orbit supports high cadence and wide field-of-view observations.

---

### Section 2: Science Overview (pp. 169–172)

논문은 SOT로 연구할 6개 핵심 과학 주제를 제시합니다:
The paper presents 6 key science topics to be studied with SOT:

**2.1 코로나 가열, 재결합, 파동 / Coronal Heating, Reconnection, and Waves**

- 코로나는 자기 재결합 또는 MHD 파동의 소산으로 가열된다고 믿어집니다 / Corona is believed heated by magnetic reconnection or MHD wave dissipation
- Parker (1988)의 nanoflare 모델: 광구 자기 원소의 운동이 코로나 자기장을 꼬이게 하여 재결합 유발 / Parker's (1988) nanoflare model: photospheric magnetic element motions braid coronal fields causing reconnection
- SOT는 개별 자기 원소의 Lagrangian 추적으로 이를 검증 가능 / SOT can verify this through Lagrangian tracking of individual magnetic elements
- SP로 MHD 파동의 직접 검출 가능 (Ulrich, 1996) / Direct detection of MHD waves possible with SP

**2.2 활동 영역과 흑점 / Active Regions and Sunspots**

- 흑점의 반암부(umbral)/편암부(penumbral) 구조 형성·유지 메커니즘 / Sunspot umbral/penumbral structure formation and maintenance
- Evershed flow와 역 Evershed flow의 구동 원리 / Driving mechanism of Evershed and inverse Evershed flows
- Moving magnetic features(MMF)를 통한 자기 플럭스 분산 / Magnetic flux dispersal via MMFs
- SOT의 Dopplergram 기능으로 **local helioseismology** 적용 가능 — 표면 아래 3D 유동·자기장 지도 작성 / SOT's Dopplergram capability enables local helioseismology — mapping 3D sub-surface flows and magnetic fields

**2.3 Flux Tube와 quiet Sun 자기장 / Flux Tubes and Quiet Sun Magnetic Fields**

- 광구의 보편적 자기 형태: 소규모, 단극, 수직 kG 자기장 (G-band bright point로 관측) / Ubiquitous magnetic form at photosphere: small-scale, unipolar, vertical kG fields (observed as G-band bright points)
- Convective collapse (Parker, 1978)로 kG 강도 tube 형성 가능 / Convective collapse (Parker, 1978) can form kG-strength tubes
- Quiet Sun의 자기장은 수시간 수명의 bipolar ephemeral region과 소규모 수평 자기장이 끊임없이 교체됨 (Title, 2007) / Quiet Sun magnetic fields are constantly replaced by bipolar ephemeral regions with lifetimes of hours and small-scale horizontal fields

**2.4 데이터 기반 코로나 역학 시뮬레이션 / Data-Driven Simulation of Coronal Dynamics**

- SOT/SP의 벡터 자기장 데이터로 코로나 자기장의 시간 의존 경계 조건 제공 / SOT/SP vector magnetic field data provide time-dependent boundary conditions for coronal magnetic fields
- 3D 자기장 외삽 → 전류 시트 구조 → 플레어/CME 예측 가능성 / 3D field extrapolation → current sheet structure → flare/CME forecasting potential
- **주의**: 편광 정밀도와 SP 스캔 시간 사이의 trade-off — 소규모 flux element의 경우 스캔 시간이 변화 시간보다 길 수 있음 / **Caveat**: trade-off between polarimetric accuracy and SP scan time — scan duration may exceed change timescale for small flux elements

**2.5 채층 가열과 역학 / Chromospheric Heating and Dynamics**

- 채층의 에너지 플럭스는 코로나 유지에 필요한 양의 ≈10배 / Chromospheric energy flux is ≈10× that required to maintain corona
- spicule 등의 제트로 코로나와 태양풍에 질량 공급 / Spicules and jets supply mass to corona and solar wind
- 채층 자기장은 force-free 코로나에 더 가까운 경계 조건 제공 — 비 force-free 광구보다 코로나 외삽에 유리 / Chromospheric magnetic fields are closer to force-free corona — better boundary conditions for coronal extrapolation than non-force-free photosphere

**2.6 자기 재결합 / Magnetic Reconnection**

(본문에서 간략히 언급) 재결합 과정의 직접적 관측 증거를 SOT와 XRT/EIS의 협동 관측으로 확보 가능 / Direct observational evidence of reconnection obtainable through coordinated SOT + XRT/EIS observations.

---

### Section 3: System Overview (pp. 172–174)

**SOT 시스템 구성 / SOT System Architecture**

```
                    ┌─────────────────────────┐
                    │   OBU (Optical Bench)    │
                    │                          │
  Solar light ──►   │  ┌─────┐    ┌─────┐     │
                    │  │ OTA │───►│ FPP │     │
                    │  └─────┘    └──┬──┘     │
                    │                │         │
                    │  ┌─────┐   ┌──┴──┐      │
                    │  │ XRT │   │ MDP │      │
                    │  └─────┘   └─────┘      │
                    │  ┌─────┐                 │
                    │  │ EIS │                 │
                    │  └─────┘                 │
                    └─────────────────────────┘
```

- OTA와 FPP는 OBU 위에 장착 (XRT, EIS도 동일) / OTA and FPP mounted on OBU (along with XRT, EIS)
- 3개 망원경의 정밀 정렬이 필수 — OBU의 열변형 특성화를 위한 광범위한 시험 수행 / Precise alignment of three telescopes essential — extensive testing for OBU thermal deformation characterization
- SOT ↔ XRT ↔ EIS 영상 정렬: 인접 파장 영상을 통한 "ladder" 정렬 기법 사용 / Image alignment between telescopes: "ladder" alignment through nearby wavelength images

**전자 시스템 / Electronics**

| 컴퓨터 / Computer | 역할 / Role |
|---|---|
| FPP-E | FPP 제어, Stokes 복조 등 온보드 처리 / FPP control, Stokes demodulation, onboard processing |
| FPP-PWR | FPP 전원 공급 / FPP power supply |
| MDP | 관측 테이블 실행, 데이터 압축, SOT/FPP 명령 / Observation table execution, data compression, SOT/FPP commands |
| CTM-E | Tip-tilt mirror 서보 제어 / Tip-tilt mirror servo control |

- FPP와 CTM-E는 MDP를 거치지 않고 직접 통신 → 영상 안정화 제어 루프 폐쇄 / FPP and CTM-E communicate directly without MDP → close image stabilization control loop
- MDP는 궤도 요소, Doppler shift 보정 정보를 SOT에 전달 / MDP sends orbital elements and Doppler shift correction to SOT
- 온도 센서 데이터 → 서보 제어기에 피드백 / Temperature sensor data → feedback to servo controller
- FPP 구역 히터로 전체 FPP 온도를 20±1°C로 유지 / FPP zone heaters maintain entire FPP at 20±1°C

---

### Section 4: Optical Telescope Assembly and Focal Plane Package (pp. 174–180)

#### 4.1 OTA 광학 / OTA Optics

**Gregorian 망원경 설계 / Gregorian Telescope Design**

OTA는 50 cm 구경의 회절 한계 aplanatic Gregorian 망원경입니다.

OTA is a diffraction-limited aplanatic Gregorian telescope with a 50 cm aperture.

```
  Solar light
      │
      ▼
┌─────────────┐  Entrance aperture (50 cm)
│             │
│  ┌───────┐  │  Primary mirror (ULE, 14 kg)
│  │  HDM  │  │  Heat dump mirror (aluminum)
│  └───┬───┘  │
│      │      │  Primary focus
│  ┌───┴───┐  │
│  │  2FS  │  │  Secondary field stop (conical, 361"×197")
│  └───┬───┘  │
│  ┌───┴───┐  │
│  │ Sec.M │  │  Secondary mirror (ULE, invar/Ti mount)
│  └───┬───┘  │
│      │      │  1.5 m separation
│  ┌───┴───┐  │
│  │  CLU  │  │  Collimator lens unit (6 lenses, f=37 cm)
│  └───┬───┘  │
│  ┌───┴───┐  │
│  │  PMU  │  │  Polarization modulator (rotating waveplate)
│  └───┬───┘  │
│  ┌───┴───┐  │
│  │CTM-TM │  │  Tip-tilt fold mirror
│  └───┬───┘  │
│      │      │  → Parallel beam to FPP
└─────┴───────┘
```

**핵심 성능 사양 / Key Performance Specifications (Table 1)**

| 매개변수 / Parameter | 사양 / Specification |
|---|---|
| 구경 / Aperture | 50 cm |
| 유효 초점거리 / Effective focal length | 1550 cm (f/31, combined with FPP) |
| Strehl ratio (OTA) | > 0.9 at 500 nm |
| Strehl ratio (FPP) | ≈ 0.9 (FOV 평균) |
| Combined Strehl | > 0.8 |
| 관측 파장 범위 / Wavelength range | 380–700 nm |
| 시야 / Max FOV (NFI) | 328" × 164" |
| 초점면 깊이 / Depth of focus | ~400 μm |

**주경과 부경 / Primary and Secondary Mirrors**

- 재질: ULE (Ultra-Low Expansion glass) — 열팽창 계수 극소 / Material: ULE — extremely low thermal expansion coefficient
- 주경 무게: 14 kg (경량화) / Primary mirror weight: 14 kg (lightweight)
- 코팅: protected silver / Coating: protected silver
- 주경은 정교한 kinematic mount로 지지 — 넓은 온도 범위에서 표면 변형 최소화 / Primary supported by elaborate kinematic mount — minimizes surface deformation over wide temperature range
- 부경은 고정 invar/titanium mount로 지지 / Secondary supported by fixed invar/titanium mount
- 주경-부경 간격: 1.5 m (위성 공간 제약, 광기계 공차, 저 f-number 주경 제작성 고려) / Primary-secondary separation: 1.5 m (considering spacecraft space, opto-mechanical tolerance, low f-number manufacturability)

**Heat Dump Mirror (HDM)**

- 알루미늄 제작, 400" 직경 FOV 밖의 태양광을 측면 창으로 반사·방출 / Aluminum-made, reflects sunlight outside 400" dia. FOV through side window to space
- 약 1500 solar의 조사를 받음 — enhanced silver 코팅의 특수 개발·시험 필요 / Illuminated by ~1500 solar — required special development/testing of enhanced silver coating
- 온도 20–40°C로 유지 — 높은 반사율과 혁신적 방사 냉각 설계 / Maintained at 20–40°C through high reflectivity and innovative radiation-cooling design

**Secondary Field Stop (2FS)**

- 원추형, FOV를 361" × 197" arcsec으로 제한 / Conical, limits FOV to 361" × 197" arcsec
- 차단된 빛은 부경과 주경을 통해 우주로 반사 / Rejected light reflected back through secondary and primary mirrors to space

**Collimator Lens Unit (CLU)**

- 6매 렌즈, 초점거리 37 cm / 6 lenses, focal length 37 cm
- IR 차단 필터 내장 / Built-in IR-rejection filter
- 색수차 보정(achromatic), 기기 편광 무시 가능 / Aberration-free (achromatic), practically instrument-polarization free
- 파장 범위: 380–700 nm / Wavelength range: 380–700 nm
- 처음 2매: 방사선에 강한 fused silica (내부 4매 보호) / First 2 lenses: radiation-robust fused silica (protecting inner 4)
- 출사동(exit pupil)이 PMU/CTM-TM 근처에 위치 / Exit pupil located near PMU/CTM-TM

**시스템 수준 비점수차 문제 / System-Level Astigmatism Issue**

개발 후반에 시스템 광학 시험에서 허용 불가한 비점수차를 발견 (주경 원인 추정). 출사동에 단일 원통 렌즈를 추가하여 해결 — 하드웨어 변경을 최소화한 후기 수정 사례.

Late in the program, system-level optical testing revealed unacceptable astigmatism (probably from primary mirror). Fixed by adding a single cylindrical lens at the exit pupil — a late correction minimizing hardware changes.

#### 4.2 Polarization Modulation (pp. 175–176)

**PMU 구조와 작동 / PMU Structure and Operation**

- 위치: 출사동 근처 / Location: near exit pupil
- 연속 회전 파장판 (revolution period $T = 1.6$ s) / Continuously rotating waveplate (period $T = 1.6$ s)
- 재질: 석영(quartz) + 사파이어(sapphire) 2결정 — 열계수 보상으로 온도 의존성 최소화 / Material: quartz + sapphire dual crystals — compensating thermal coefficients minimize temperature dependence
- 최적화 파장: 630.2 nm (retardation 1.35 waves)와 517.2 nm (1.85 waves) / Optimized for 630.2 nm (1.35 waves) and 517.2 nm (1.85 waves)
- 이 두 파장에서 Stokes $Q$, $U$, $V$의 변조 효율이 균등하게 약 0.5 / At these two wavelengths, modulation efficiency of Stokes $Q$, $U$, $V$ is equally ~0.5

**Stokes 변조 원리 / Stokes Modulation Principle**

편광 상태 ($I, Q, U, V$)는 FPP의 편광 빔스플리터로 강도의 정현파 변화로 변환됩니다:
Polarization states ($I, Q, U, V$) are converted to sinusoidal intensity variations by FPP polarizing beam splitters:

| Stokes | 변조 주기 / Modulation period | 위상 / Phase |
|---|---|---|
| $Q$ | $T/4 = 0.4$ s | 기준 / Reference |
| $U$ | $T/4 = 0.4$ s | $Q$에서 22.5° 지연 / 22.5° lag from $Q$ |
| $V$ | $T/2 = 0.8$ s | — |

- PMU 1회전당 16회 샘플링으로 복조 / Demodulated by 16 samples per PMU revolution
- 각 샘플을 FPP의 4개 메모리 슬롯에 가감산하여 $I$, $Q$, $U$, $V$ 스펙트라 생성 / Each sample added/subtracted into 4 memory slots in FPP to generate $I$, $Q$, $U$, $V$ spectra

**PMU의 투명성 / PMU Transparency**

- 회전하는 PMU는 비자기 광도측정 관측에서 완전히 투명 / Rotating PMU is completely invisible for non-magnetic photometric observations
- 최소 잔류 wedge에 의한 영상 이동은 영상 안정화 시스템이 제거 / Minimum residual wedge causing image motion is removed by image stabilization system
- PMU 이전의 모든 광학 요소가 광축 대칭 → 기기 편광 최소화 / All optical elements before PMU are rotationally symmetric about optical axis → minimizes instrumental polarization

**OTA-FPP 인터페이스 / OTA-FPP Interface**

- FPP는 re-imaging lens로 평행광을 수신 → pupil reducer 역할 / FPP receives parallel light via re-imaging lens → acts as pupil reducer
- OTA-FPP 인터페이스는 afocal (무초점) → FPP의 위치 공차 크게 완화 / OTA-FPP interface is afocal → greatly relaxes positional tolerance of FPP
- OTA와 FPP를 OBU에 독립적으로 장착 가능 / OTA and FPP can be independently mounted on OBU

#### 4.3 Optical Testing (pp. 176–178)

**Wavefront Error (WFE) 측정 / WFE Measurement**

- 대구경 회전형 정밀 평면경을 OTA 입구 앞에 설치 (auto-collimation, double pass) / Large rotatable precision flat placed before OTA entrance (auto-collimation, double pass)
- Fizeau 간섭계를 FPP 위치에 설치하여 OTA WFE 측정 / Interferometer placed at FPP position to measure OTA WFE
- 중력 효과 제거: OTA를 뒤집어서 두 번째 WFE 맵 취득 후 두 맵을 합산 → 무중력 상태의 WFE 추정 / Gravity removal: OTA flipped upside-down for second WFE map, sum of two maps estimates zero-gravity WFE
- 결과: 주경에 의한 삼각형 비점수차 발견 → 원통 렌즈로 보정 / Result: triangle-astigmatism from primary mirror discovered → corrected with cylindrical lens

**Sun Test**

- 헬리오스탯을 통해 실제 태양광을 클린룸에 도입, OTA+FPP 비행 전자장치로 종합 관측 시험 / Real sunlight introduced to clean room via heliostat, end-to-end observation test with OTA+FPP flight electronics
- Muller matrix 결정을 위한 편광 교정 수행 (Ichimoto et al., 2008) / Polarization calibration for Muller matrix determination (Ichimoto et al., 2008)

**Opto-Thermal Test**

- 열진공 챔버에서 궤도상 온도 기울기를 시뮬레이션하여 OTA WFE 측정 / OTA WFE measured in thermal vacuum chamber simulating in-orbit temperature gradient
- 특수 슈라우드로 광축 방향 고온도 기울기 재현 / Special shroud reproduces high temperature gradient along optical axis

**광학 유지보수 포트 / Optical Maintenance Port**

- OBU의 OTA-FPP 광학 인터페이스 부근 소형 구멍 / Small hole on OBU near OTA-FPP optical interface
- 위성 통합 후에도 OTA WFE 측정 가능 — 진동·충격 시험 후 광학 건전성 확인에 활용 / Enables OTA WFE measurement even after satellite integration — used for optical health checks after vibration/shock tests
- 발사장 인도 시까지 반복 활용 → 궤도 성능에 대한 강한 확신 제공 / Used repeatedly until delivery to launch site → strong confidence in in-orbit performance

**교훈**: 우주 광학 계기 미션의 성공에는 광범위하고 완전한 시험이 필수적입니다.
**Lesson learned**: Extensive and complete testing is essential to the success of advanced space-optics instrumentation missions.

#### 4.4 Structural and Thermal Properties (pp. 178–180)

**OTA 구조 / OTA Structure**

- 주 구조: graphite-cyanate composite 파이프 + 허니콤 패널의 정밀 트러스 / Main structure: precision truss of graphite-cyanate composite pipes + honeycomb panels
- 열팽창 계수: 0.05 ppm/°C (극도로 낮음) / Thermal expansion: 0.05 ppm/°C (extremely low)
- 모든 building block을 접착제로 연결 (볼트/핀 미사용) → 온도 변화와 기계적 환경에서 치수 안정성 확보 / All building blocks connected with adhesive (no bolts/pins) → dimensional stability under temperature changes and mechanical environments
- 총 무게: OTA 103 kg, FPP 46 kg / Total weight: OTA 103 kg, FPP 46 kg
- ISAS/JAXA M-V 고체 로켓의 심한 진동/음향/충격 하중 견딤 / Withstands severe vibration/acoustic/shock loads of ISAS/JAXA M-V solid rocket

**열 설계 / Thermal Design**

- 주경이 태양 에너지의 약 6.5% 흡수 → 주요 내부 열원 / Primary mirror absorbs ~6.5% of solar energy → main internal heat source
- 모든 광학 요소는 망원경 구조에 복사적으로 결합 / All optical elements radiatively coupled to telescope structure
- HDM의 특수 냉각 핀으로 흡수 열 방출 / HDM has special fins to dump absorbed heat
- 열 방출 경로: 입구 구경 측 OSR (태양 대면 방사기) → 후면은 위성 버스가 차지 / Heat dump path: sun-facing OSR radiator at entrance aperture → backside occupied by spacecraft bus
- Heat pipe 미사용 — 복사 결합이 주 열전달 경로 / No heat pipe — radiation coupling is primary heat transfer path
- Z축 온도 기울기: 주경 30°C → 부경 0°C 이하 / Z-axis temperature gradient: primary mirror 30°C → secondary mirror below 0°C
- 3종 히터 시스템: 운용 히터, 오염제거 히터, 생존 히터 / Three heater types: operational, decontamination, survival

**FPP 구조 / FPP Structure**

- 알루미늄 허니콤 광학 벤치 + 측면 패널 + 커버 플레이트 / Aluminum honeycomb optical bench + side panels + cover plate
- OBU에 kinematic mount로 장착, OBU와 열적 절연 / Mounted on OBU by kinematic mount, thermally isolated from OBU
- 각 CCD 검출기는 전용 방열기로 냉각 / Each CCD detector cooled by dedicated radiator

#### 4.5 Contamination Control (pp. 180)

- 초기 설계 단계부터 엄격한 오염 제어 프로그램 시행 / Stringent contamination control from early design phase
- 오염으로 인한 광학면 어두워짐 → 열 흡수 증가 → 주경 온도 상승 → 거울 변형 (ULE-super invar 패드 간 CTE 차이) / Contamination darkens optical surfaces → increased heat absorption → primary mirror temperature rise → mirror deformation (CTE difference between ULE and super-invar pads)
- 모든 비행 부품의 탈가스율을 Thermoelectric Quartz Crystal Microbalance (TQCM)으로 정량 모니터링 / All flight component outgas rates quantitatively monitored with TQCM
- 수학적 오염 모델로 궤도 수명 예측 / Mathematical contamination model predicts orbital lifetime
- 주경/부경/HDM에 전용 오염제거 히터 — 주변보다 10°C 이상 고온 유지 / Dedicated decontamination heaters on primary/secondary/HDM — maintain ≥10°C above surroundings
- 발사 후 측면 문만 먼저 개방 (탈가스 배출), 탈가스 기간 후 주문 개방 → 관측 시작 / After launch, only side door opened first (outgas venting), main door opened after outgassing period → observations begin
- FPP는 밀폐 구조이고 유해 UV는 OTA가 흡수하므로 FPP에는 엄격한 오염 제어 불필요 / FPP is closed structure and hazardous UV absorbed by OTA, so stringent contamination control not needed for FPP

---

### Section 5: SOT Observing Modes (pp. 180–184)

#### 광학 경로 / Optical Path in FPP

```
OTA (평행광) ──► Re-imaging lens ──► Beam splitter
                                        │
                              ┌─────────┼──────────┐
                              │         │          │
                              ▼         ▼          ▼
                          SP channel  Filter channel  CT
                              │         │
                              │    Polarizing BS
                              │    ┌────┴────┐
                              │    ▼         ▼
                              │   NFI       BFI
                              │  (p-pol)   (s-pol)
                              ▼
                          SP CCD (2개)
```

- Non-polarizing beam splitter로 SP와 filtergraph 채널로 분리 / Non-polarizing beam splitter divides light between SP and filtergraph channels
- Polarizing beam splitter로 NFI (p-편광)와 BFI (s-편광)로 분리 / Polarizing beam splitter separates NFI (p-polarized) and BFI (s-polarized)
- BFI와 NFI는 4K×2K CCD를 공유 / BFI and NFI share a 4K×2K CCD
- FG와 SP는 MDP의 매크로 명령에 따라 독립적으로 동시 관측 가능 / FG and SP can observe simultaneously and independently per MDP macro-commands

#### 5.1 Filter Observations (BFI + NFI)

**BFI (Broadband Filter Imager)**

| 파장대 / Band | 파장 / λ (nm) | 용도 / Purpose |
|---|---|---|
| CN band | 388.3 | 광구 영상 / Photosphere imaging |
| Ca II H | 396.9 | 저 채층 / Low chromosphere |
| G band | 430.5 | 자기 bright point 검출 / Magnetic bright point detection |
| Blue continuum | 450.5 | 연속 측정 / Continuum measurement |
| Green continuum | 555.1 | 연속 측정 / Continuum measurement |
| Red continuum | 668.4 | 복사 조도 연구 / Irradiance studies |

- 픽셀 크기: 0.0541 arcsec/pixel / Pixel size: 0.0541 arcsec/pixel
- FOV: 218" × 109" arcsec
- 케이던스: <10 s / Cadence: <10 s
- 노출 시간: 0.03–0.8 s (더 긴 노출 가능) / Exposure time: 0.03–0.8 s (longer possible)
- BFI 필터 FWHM: 0.3–0.7 nm → Doppler 운동에 무감 / BFI filter FWHM: 0.3–0.7 nm → insensitive to Doppler motion

**NFI (Narrowband Filter Imager)**

| 스펙트럼선 / Line | 파장 / λ (nm) | 관측 / Observation |
|---|---|---|
| Mg I b | 517.3 | 채층 Dopplergram + Magnetogram |
| Fe I | 525.0, 524.7, 525.0 | 광구 자기장 |
| Fe I | 557.6 | 광구 Dopplergram (Landé $g = 0$) |
| Na I | 589.6 | 채층 Dopplergram |
| Fe I | 630.3, 630.2 | 광구 자기장 (SP와 동일선) |
| H I (Hα) | 656.3 | 채층 구조 |

- 픽셀 크기: 0.08 arcsec/pixel / Pixel size: 0.08 arcsec/pixel
- FOV: 328" × 164" arcsec
- Lyot 필터 대역폭: ≈95 mÅ at 630 nm, 파장 가변 / Lyot filter bandwidth: ≈95 mÅ at 630 nm, tunable
- 텔레센트릭 빔 → FOV 전체에서 파장 이동 없음 / Telecentric beam → no wavelength shift across FOV
- 비네팅 없는 영역: 직경 264 arcsec / Un-vignetted area: 264 arcsec diameter

**Filter 관측의 4가지 데이터 산물 / Four Data Products from Filter Observations**

1. **Filtergram**: 단일 노출 스냅샷, 태양 특징의 강도 분포 / Single exposure snapshot, intensity distribution of solar features
2. **Dopplergram**: NFI에서 스펙트럼선의 여러 위치에서 2 또는 4개 영상 → Doppler shift 추정. 최적선: Fe I 557.6 (Landé $g = 0$). rms 노이즈 ≈ 30 m/s (4-image) / NFI images at 2 or 4 positions in a spectral line → Doppler shift estimate. Best line: Fe I 557.6 (Landé $g = 0$). rms noise ≈ 30 m/s (4-image)
3. **Longitudinal Magnetogram**: 시선 방향 자기장 성분의 위치, 극성, 플럭스 추정. 주요선: Fe I 630.25 (광구), Mg I 517.27 (저 채층). 케이던스 ≈20 s (8-image), rms 노이즈 ≈ $10^{15}$ Mx/pixel / LOS magnetic field component. Primary lines: Fe I 630.25 (photosphere), Mg I 517.27 (low chromosphere). Cadence ≈20 s, rms noise ≈ $10^{15}$ Mx/pixel
4. **Stokes $IQUV$ 영상**: PMU의 8 위상에서 NFI filtergram → 온보드 복조 → 벡터 자기장 정보. 셔터리스 모드로 1.6–4.8 s 케이던스 가능 (focal plane mask 필요) / Stokes $IQUV$ from 8 PMU phases of NFI filtergrams → onboard demodulation → vector magnetic field info. Shutterless mode for 1.6–4.8 s cadence (focal plane mask required)

**CCD 판독 시간 / CCD Readout Times**

| Summing | 판독 시간 / Readout time |
|---|---|
| 1×1 | 3.4 s |
| 2×2 | 1.7 s |
| 4×4 | 0.9 s |

- 중앙 2K×2K 윈도우 판독으로 더 빠른 케이던스 가능 / Faster cadence with central 2K×2K window readout
- 메커니즘 재구성 시간 (필터휠 교환 포함): ≈2.5 s 이하 / Mechanism reconfiguration time (including filter wheel change): ≤ ≈2.5 s

#### 5.2 Spectral Observations (SP)

**SP 기본 사양 / SP Basic Specifications**

- Off-axis Littrow-Echelle 분광기 / Off-axis Littrow-Echelle spectrograph
- 관측선: Fe I 630.15 nm + Fe I 630.25 nm (이중선) + 인접 연속 / Lines: Fe I 630.15 nm + Fe I 630.25 nm (dual line) + nearby continuum
- 슬릿 크기: 0.16" × 151" arcsec / Slit size: 0.16" × 151" arcsec
- 분광 분해능: 2.15 pm / Spectral resolution: 2.15 pm
- CCD 2개로 직교 선형 편광을 동시 측정 → 다운링크 후 합산으로 잔류 jitter/태양 진화에 의한 허위 편광 감소 / Two CCDs simultaneously record orthogonal linear polarizations → combined after downlink to reduce spurious polarization from residual jitter/solar evolution
- PMU 1회전당 16회 연속 노출·판독 → 온보드 실시간 복조 → Stokes $IQUV$ 스펙트럼 생성 / 16 continuous exposures per PMU revolution → onboard real-time demodulation → Stokes $IQUV$ spectra

**SP 관측 모드 / SP Observing Modes**

| 모드 / Mode | 슬릿 스캔 영역 / Scan width | 픽셀 크기 / Pixel | 케이던스 / Cadence | 편광 정밀도 / Pol. accuracy |
|---|---|---|---|---|
| Normal Map | 160" | 0.15"×0.16" | 83 min | 0.1% |
| Fast Map | 160" | 0.30"×0.32" | 30 min | ~0.1% (1.15× better) |
| Dynamics | 1.6" | 0.16" | 18 s | 낮음 / Lower |
| Deep Magnetogram | 가변 / Variable | 0.16" | 가변 / Variable | 매우 높음 / Very high |

- Normal Map: 중간 크기 활동영역 커버 가능 / Covers moderate-sized active region
- Fast Map: 1.6 s 적분, 슬릿 방향 2 pixel 합산 → 빠른 스캔 / 1.6 s integration, 2 pixels summed along slit
- Dynamics: 슬릿 고정, 고시간 분해능 / Fixed slit, high temporal resolution
- Deep Magnetogram: 다수 PMU 회전에 걸쳐 광자 축적 → quiet Sun에서 극고 편광 정밀도 (시간 분해능 희생) / Photon accumulation over many PMU rotations → very high polarimetric accuracy in quiet Sun (sacrificing time resolution)

**셔터리스 모드 / Shutterless Mode**

- CCD의 frame transfer를 사용 / Uses CCD frame transfer
- FOV 제한 (focal plane mask 사용): 0.08" pixel에서 5.1"×164" ~ 0.32" pixel에서 25.6"×164" / FOV limited by focal plane mask: 5.1"×164" for 0.08" pixels to 25.6"×164" for 0.32" pixels
- 기계식 셔터 사용 시 더 넓은 FOV 가능: 최대 82"×164" (0.08") 또는 328"×164" (0.16"/0.32") / Wider FOV with mechanical shutter: up to 82"×164" (0.08") or 328"×164" (0.16"/0.32")

---

### Section 6: Image Stabilization System and Micro-vibration (pp. 184–186)

#### 6.1 Image Stabilization System

**필요성 / Necessity**

- 영상 jitter는 분해능 저하와 편광 cross-talk 유발 / Image jitter causes resolution degradation and polarization cross-talk
- 원인: 위성 자세 요동/표류, PMU 회전 관련 흔들림, 광열 변형에 의한 느린 표류 / Causes: spacecraft attitude jitter/drift, wobbling from PMU rotation, slow drift from opto-thermal deformation
- 요구 안정도: 0.03 arcsec rms 이하 (Shimizu et al., 2008) / Required stability: less than 0.03 arcsec rms

**시스템 구성 / System Components (Table 5)**

| 구성 요소 / Component | 사양 / Specification |
|---|---|
| Correlation Tracker (CT) | 580 Hz CCD, 11"×11" FOV, 태양립 패턴 추적 / 580 Hz CCD, 11"×11" FOV, tracks granulation pattern |
| CTM-TM (Tip-Tilt Mirror) | Piezo 구동, 3개 piezo (2축 + 1 여분) / Piezo-driven, 3 piezos (2-axis + 1 redundant) |
| CTM-E (Controller) | 디지털 서보, 폐루프 대역폭 < 14 Hz / Digital servo, closed-loop bandwidth < 14 Hz |
| 달성 안정도 / Achieved stability | **0.007" (1σ) in orbit** |

**작동 원리 / Operating Principle**

1. CT(580 Hz CCD)가 11"×11" FOV에서 태양립 패턴을 기준점으로 사용하여 영상 변위 검출 / CT (580 Hz CCD) detects image displacement using granulation pattern as fiducial in 11"×11" FOV
2. 기준 영상은 40초 간격으로 갱신 / Reference image updated at 40 s intervals
3. 변위 신호 → FPP 컴퓨터 → CTM-E로 고속 전달 / Displacement signal → FPP computer → high-speed transfer to CTM-E
4. CTM-E가 piezo 구동으로 tip-tilt mirror 조정 → 폐루프 서보로 jitter 보정 / CTM-E adjusts tip-tilt mirror via piezo → closed-loop servo corrects jitter
5. FPP ↔ CTM-E 직접 통신 (MDP 비개입) → 제어 루프 지연 최소화 / FPP ↔ CTM-E direct communication (no MDP involvement) → minimizes control loop delay

**대역폭 제한 / Bandwidth Limitation**

- 14 Hz 미만의 비교적 낮은 대역폭 — CCD 판독 지연 시간 때문 / Relatively low bandwidth below 14 Hz due to CCD readout delay time
- 그럼에도 0.007" (1σ) 달성 — 이는 요구 사양(0.03")의 4배 이상 우수 / Despite this, 0.007" (1σ) achieved — 4× better than the 0.03" requirement

**안정성의 추가 기여 요소 / Additional Factors Contributing to Stability**

- 우수한 위성 자세 안정성 / Excellent spacecraft attitude stability
- 기기의 구조-열 설계 / Structural-thermal design of the instrument
- 태양 동기 궤도의 안정적 열 입력 / Stable thermal input from sun-synchronous orbit
- Tip-tilt mirror 위치에서의 동공 크기 축소(~16배) → 각도 증폭 / Pupil size reduction (~16×) at tip-tilt mirror location → angle amplification

**Piezo 우주 검증 / Piezo Space Qualification**

- Queensgate Instruments 제조 상용 piezo 사용 / Commercial piezos by Queensgate Instruments
- NAOJ에서 고온·진공 장기 수명 시험 수행 / Long-term high-temperature vacuum life test at NAOJ
- 3개 piezo 중 1개 고장 시에도 작동 가능 (감소된 stroke) — CTM-E 소프트웨어가 contingency 모드 지원 / Operable even if 1 of 3 piezos fails (reduced stroke) — CTM-E software supports contingency mode

#### 6.2 Micro-vibration

- 위성의 자이로스코프, 모멘텀 휠, 기기 메커니즘 등에서 발생 / Generated by satellite gyroscopes, momentum wheels, instrument mechanisms
- 주파수 범위가 영상 안정화 대역폭(14 Hz)보다 훨씬 높음 → 안정화 시스템으로 보정 불가 / Frequency range much higher than stabilization bandwidth (14 Hz) → cannot be corrected by stabilization system
- 일부 micro-vibration이 망원경 구조와 공진 / Some micro-vibrations resonated with telescope structure
- 대책: (1) 노이즈원을 전달 함수가 낮은 위치로 이전, (2) 회전 주파수 조정으로 공진 회피, (3) 구조 개선, (4) 운용상 우회 / Countermeasures: (1) relocate noise sources to lower transfer function locations, (2) adjust rotation frequency to avoid resonance, (3) structural improvement, (4) operational workaround

---

### Section 7: SOT Observation Control and Data Flows (pp. 186–188)

**관측 제어 구조 / Observation Control Architecture**

- SOT 관측 시퀀스는 MDP가 전적으로 제어 / SOT observation sequence entirely controlled by MDP
- 2개 동시 관측 테이블: FG용 1개 + SP용 1개 / Two concurrent observation tables: one for FG + one for SP
- 테이블 내용은 지상에서 과학 관측 계획으로 작성 → 거의 매일 업로드 / Table contents prepared as science observing plans on ground → uploaded almost daily

**관측 프로그램 구조 (Sequence Tables) / Observation Program Structure**

```
Observation Program (최대 20개 / max 20)
├── Main Routine (loop count: 0 = infinite)
│   ├── Subroutine 1 (repeat count + interval)
│   │   ├── Sequence Table 1 (max 8 command lines)
│   │   ├── Sequence Table 2
│   │   └── ... (max 8 sequences)
│   ├── Subroutine 2
│   ├── Subroutine 3
│   └── Subroutine 4 (max 8 subroutines)
└── (max 100 sequence tables total)
```

- 각 Sequence Table은 최대 8개의 매크로 명령줄, 다음 명령까지의 타이밍 포함 / Each sequence table has max 8 macro-command lines with timing to next command
- 공학/유지보수 명령도 sequence table에 포함 가능 / Engineering/maintenance commands can also be included

**데이터 흐름 / Data Flow**

1. CCD 영상 취득 (FG 또는 SP) / CCD image acquisition (FG or SP)
2. SOT/FPP에서 실시간 처리: Dopplergram, Magnetogram, Stokes 복조 등 / Real-time processing in SOT/FPP: Dopplergrams, Magnetograms, Stokes demodulation
3. 고속 병렬 인터페이스로 MDP에 전송 / Transfer to MDP via high-speed parallel interface
4. MDP에서 데이터 압축:
   - 16 → 12 bit 심도 압축 (8개 look-up table) / 16→12 bit depth compression (8 look-up tables)
   - 2D 영상 압축: JPEG DCT lossy 또는 DPCM lossless / 2D image compression: JPEG DCT lossy or DPCM lossless
   - Filtergram: ≈3 bits/pixel (JPEG), Stokes: ≈1.5 bits/pixel / Filtergram: ≈3 bits/pixel, Stokes: ≈1.5 bits/pixel
5. CCSDS 패킷화 → DHU → 데이터 레코더(DR) 기록 / CCSDS packetization → DHU → data recorder
6. 지상국 패스에서 다운링크 / Downlink during ground station pass

**텔레메트리 예산 / Telemetry Budget**

| 항목 / Item | 수치 / Value |
|---|---|
| DR 용량 / Capacity | ≈8 Gbit 총 (SOT ≈5.6 Gbit, ~70%) |
| 1 패스 다운링크 / Per-pass downlink | 1.7 Gbit (4 Mbps × 10 min) |
| 15 패스/일 최대 / 15 passes/day max | 25.5 Gbit/day |
| 평균 데이터율 / Avg data rate | ≈300 kbps (post-compression) |
| 최대 (SOT 단독) / Max (SOT-dominant) | ≈1.8 Mbps |
| Burst 모드 / Burst mode | DR을 1시간에 채움 → 3 패스 필요 / Fills DR in 1 hour → needs 3 passes |

---

### Section 8: Conclusions (pp. 188–189)

**궤도 성능 요약 / In-Orbit Performance Summary**

- SOT는 우주에서 가장 큰 구경의 첨단 태양 광학 망원경 / SOT is the largest aperture advanced solar optical telescope in space
- BFI, SP, 영상 안정화 시스템: 발사 전 기대를 충족하거나 초과 / BFI, SP, image stabilization: met or exceeded pre-launch expectations
- 안정적인 관측 케이던스 — 위성 밤/식(eclipse)이나 나쁜 seeing에 무영향 → 고품질 동영상 획득에 특히 유효 / Stable observation cadence — unaffected by spacecraft night/eclipses or bad seeing → particularly effective for high-quality movies

**NFI Bubble 문제 / NFI Bubble Problem**

궤도상에서 발견된 주요 문제:
Major issue discovered in orbit:

- NFI의 tunable Lyot 필터 내부에 **기포(air bubble)**가 존재 / **Air bubbles** exist inside NFI's tunable Lyot filter
- 필터 조율 시 기포가 변형되고 이동 → FOV 일부의 영상 왜곡·가림 / Bubbles distort and move when filter is tuned → image degradation/obscuration over part of FOV
- 시간이 지나면 기포가 FOV 가장자리로 이동하는 경향 / Bubbles tend to drift toward FOV edges over time

**대응책 / Countermeasures:**

1. 소프트웨어 변경으로 기포 위치에 대한 상당한 제어 확보 → 관측 대상을 blemish-free 영역에 배치 / Software changes gave considerable control over bubble location → place targets in blemish-free areas
2. 기포를 교란하지 않는 조율 방식 개발 → 라인 프로파일 내 여러 위치 조율 가능 / Tuning schemes developed that don't disturb bubbles → can tune to different line profile positions
3. NFI 관측은 한 번에 하나의 스펙트럼선에서 수행 (빠른 파장 전환 금지) / NFI observing done in one spectral line at a time (rapid line switching not allowed)
4. 예상된 NFI 관측의 대부분 수집 가능 / Most expected NFI observations could be collected
5. Flat field 보정은 여전히 과제 — 그러나 magnetogram/Dopplergram은 강도 비율 차이로 만들므로 self-correcting / Flat field correction remains challenging — but magnetograms/Dopplergrams are self-correcting since made from intensity difference ratios

---

## 3. 핵심 시사점 / Key Takeaways

1. **우주 최대의 태양 광학 망원경**: SOT는 50 cm 구경으로 우주에서 가장 큰 태양 광학 망원경이며, 0.2–0.3 arcsec 회절 한계 분해능을 달성했습니다. 이는 지상 적응광학 망원경과 달리 시간적으로 안정적이고 연속적인 관측을 보장합니다.
**Largest solar optical telescope in space**: SOT's 50 cm aperture is the largest solar optical telescope in space, achieving 0.2–0.3 arcsec diffraction-limited resolution. Unlike ground-based adaptive optics telescopes, this guarantees temporally stable and continuous observations.

2. **3중 관측 시너지**: Hinode의 핵심 전략은 SOT(광구/채층) + EIS(전이영역/코로나) + XRT(코로나)의 동시 관측으로 자기 에너지의 생성-수송-소산 전 과정을 추적하는 것입니다. SOT는 이 연쇄에서 "근원"을 관측합니다.
**Triple observation synergy**: Hinode's key strategy is simultaneous SOT (photosphere/chromosphere) + EIS (transition region/corona) + XRT (corona) observations to trace the full generation-transport-dissipation cycle of magnetic energy. SOT observes the "source" in this chain.

3. **영상 안정화의 탁월성**: 0.007" (1σ)의 궤도상 안정도는 0.03" 요구 사양의 4배 이상을 달성했습니다. Correlation tracker(580 Hz) + tip-tilt mirror의 조합과 동공 축소에 의한 각도 증폭이 핵심입니다.
**Exceptional image stabilization**: In-orbit stability of 0.007" (1σ) exceeded the 0.03" requirement by 4×. The combination of correlation tracker (580 Hz) + tip-tilt mirror with angle amplification from pupil reduction is key.

4. **연속 회전 PMU의 우아함**: 석영+사파이어 이중결정 파장판의 연속 회전(T=1.6 s)은 비자기 관측에 완전 투명하면서도 Stokes $IQUV$ 전체를 균등하게 변조합니다. 16-sample 복조로 높은 편광 정밀도를 달성합니다.
**Elegance of continuously rotating PMU**: The quartz+sapphire dual-crystal waveplate's continuous rotation (T=1.6 s) is completely transparent to non-magnetic observations while uniformly modulating all Stokes $IQUV$. 16-sample demodulation achieves high polarimetric accuracy.

5. **SP의 다중 모드 유연성**: Normal Map(83 min, 0.1% 정밀도)부터 Dynamics(18 s, 고시간 분해능), Deep Magnetogram(극고 편광 정밀도)까지 — 과학 목표에 따라 시간 분해능과 편광 정밀도를 교환할 수 있습니다.
**SP multi-mode flexibility**: From Normal Map (83 min, 0.1% accuracy) to Dynamics (18 s, high temporal resolution) to Deep Magnetogram (very high polarimetric accuracy) — time resolution and polarimetric accuracy can be traded depending on science objectives.

6. **열 설계의 혁신성**: ULE 거울 + graphite-cyanate composite 트러스(0.05 ppm/°C) + 복사 결합 열전달 + 30°C→0°C Z축 기울기라는 독특한 열 설계가 광학 안정성을 보장합니다. Heat pipe 미사용이라는 점이 특이합니다.
**Innovative thermal design**: The unique thermal design — ULE mirrors + graphite-cyanate composite truss (0.05 ppm/°C) + radiation coupling heat transfer + 30°C→0°C Z-axis gradient — ensures optical stability. The absence of heat pipes is notable.

7. **NFI bubble 문제의 교훈**: Lyot 필터 내부 기포라는 예기치 않은 문제가 발생했지만, 소프트웨어 대응과 운용 전략으로 과학적 영향을 최소화했습니다. 이는 우주 기기 개발에서 "완벽한 기기보다 유연한 운용"의 중요성을 보여줍니다.
**Lesson from NFI bubble problem**: An unexpected air bubble issue in the Lyot filter was mitigated through software countermeasures and operational strategies to minimize scientific impact. This demonstrates the importance of "flexible operations over perfect hardware" in space instrument development.

8. **광학 유지보수 포트의 선견지명**: 위성 통합 후에도 OTA WFE를 측정할 수 있는 유지보수 포트의 설계는 발사까지의 반복 검증을 가능하게 하여 궤도 성능에 대한 강한 확신을 제공했습니다.
**Foresight of optical maintenance port**: The maintenance port design enabling OTA WFE measurement even after satellite integration allowed repeated verification until launch, providing strong confidence in in-orbit performance.

---

## 4. 수학적 요약 / Mathematical Summary

### 회절 한계 분해능 / Diffraction-Limited Resolution

$$\theta = 1.22 \frac{\lambda}{D}$$

- $\theta$: 각분해능 (Airy disk 첫 번째 영 위치) / Angular resolution (first zero of Airy disk)
- $\lambda$: 관측 파장 / Observing wavelength
- $D$: 구경 (= 0.5 m for SOT) / Aperture diameter

**구체적 계산 / Worked Example:**

500 nm에서 / At 500 nm:

$$\theta = 1.22 \times \frac{500 \times 10^{-9}}{0.5} = 1.22 \times 10^{-6} \text{ rad} = 0.252 \text{ arcsec}$$

630 nm에서 / At 630 nm:

$$\theta = 1.22 \times \frac{630 \times 10^{-9}}{0.5} = 1.537 \times 10^{-6} \text{ rad} = 0.317 \text{ arcsec}$$

### Strehl Ratio와 Maréchal Criterion / Strehl Ratio and Maréchal Criterion

$$S = \frac{I_{\text{peak, actual}}}{I_{\text{peak, ideal}}} \approx \exp\left[-\left(\frac{2\pi \sigma}{\lambda}\right)^2\right]$$

- $S$: Strehl ratio
- $\sigma$: wavefront error rms
- $\lambda$: 파장 / wavelength

**Maréchal criterion**: $S \geq 0.8$ ↔ $\sigma \leq \lambda / 14$

SOT에서 / For SOT:

$$\sigma_{\text{max}} = \frac{500 \text{ nm}}{14} \approx 35.7 \text{ nm rms}$$

실제 달성: OTA Strehl > 0.9, FPP Strehl ≈ 0.9, combined > 0.8

### Stokes 편광 변조 / Stokes Polarization Modulation

PMU 회전각 $\phi = 2\pi t / T$ ($T = 1.6$ s)에서 Stokes 매개변수의 강도 변조:

With PMU rotation angle $\phi = 2\pi t / T$ ($T = 1.6$ s), intensity modulation of Stokes parameters:

$$I_{\text{measured}} = \frac{1}{2}\left[I + Q\cos(4\phi) + U\sin(4\phi) + V\sin(2\phi)\right]$$

여기서 (간략화된 형태) / where (simplified form):
- $Q$: $4\phi$ (= $T/4$ 주기)에서 cosine 변조 / cosine modulation at $4\phi$ (period $T/4$)
- $U$: $4\phi$에서 sine 변조 (Q 대비 22.5° 지연) / sine modulation at $4\phi$ (22.5° lag from Q)
- $V$: $2\phi$ (= $T/2$ 주기)에서 sine 변조 / sine modulation at $2\phi$ (period $T/2$)

복조: PMU 1회전당 16 등간격 샘플($\phi_k = k \cdot 2\pi/16$, $k = 0, ..., 15$)을 4개 메모리에 가감산

Demodulation: 16 equally-spaced samples per revolution ($\phi_k = k \cdot 2\pi/16$, $k = 0, ..., 15$) added/subtracted into 4 memories

### Dopplergram 유도 / Dopplergram Derivation

NFI에서 스펙트럼선의 $n$개 위치($\lambda_1, ..., \lambda_n$)에서 filtergram 취득 시:

From NFI filtergrams at $n$ positions ($\lambda_1, ..., \lambda_n$) in a spectral line:

4-image 방법 ($\lambda_1 < \lambda_2 < \lambda_3 < \lambda_4$, 등간격) / 4-image method (equally spaced):

$$v_{\text{Doppler}} \propto \frac{(I_1 + I_2) - (I_3 + I_4)}{(I_1 + I_2) + (I_3 + I_4)}$$

- 최적선: Fe I 557.6 nm (Landé $g = 0$ → 자기장에 무감) / Best line: Fe I 557.6 nm (Landé $g = 0$ → insensitive to magnetic field)
- rms 노이즈: ≈30 m/s (4-image) / rms noise: ≈30 m/s (4-image)

### Longitudinal Magnetogram 유도 / Longitudinal Magnetogram Derivation

$$M_{\text{long}} \propto \frac{I_R - I_L}{I_R + I_L} \propto V / I$$

- $I_R$, $I_L$: 우원편광, 좌원편광 강도 / Right and left circular polarization intensities
- $V / I$: Stokes $V$ 신호의 $I$ 정규화 → 시선 방향 자기장에 비례 / Stokes $V$ normalized by $I$ → proportional to LOS magnetic field
- rms 노이즈: ≈$10^{15}$ Mx/pixel (8-image, ≈20 s) / rms noise: ≈$10^{15}$ Mx/pixel (8-image, ≈20 s)

---

## 5. 역사 속의 논문 / Paper in the Arc of History

```
1868  Lockyer/Janssen — 분광기로 채층 발견 / Spectroscopic chromosphere discovery
  │
1908  Hale — 흑점 자기장 발견 (Zeeman 효과) / Sunspot magnetic field (Zeeman)
  │
1933  Lyot — 코로나그래프 발명 / Coronagraph invention
  │
1944  Lyot — 복굴절 필터 / Birefringent filter (Paper #3)
  │
1962  OSO-1 — 최초 우주 태양 관측 / First space solar observation
  │
1973  Skylab/ATM — 우주 태양 UV/X선 관측 / Space solar UV/X-ray
  │
1991  Yohkoh — 우주 X선 태양 관측 / Space X-ray solar observation
  │
1995  SOHO — L1점 태양 관측소 / L1 solar observatory
  │
1998  TRACE — 고해상도 EUV 영상 / High-res EUV imaging
  │
2000  SST — 지상 1 m, 적응광학 0.1" / Ground 1 m, AO 0.1"
  │
2006  ★ Hinode/SOT — 우주 50 cm, 회절 한계 0.2" + 분광편광
  │    ★ First diffraction-limited optical telescope in space
  │    ★ + spectropolarimetry
  │
2008  ★★ 본 논문 출판 / ★★ This paper published
  │
2010  SDO/HMI — 전일면 자기장 / Full-disk magnetograms (Paper #13)
  │
2013  IRIS — 채층/전이영역 UV 분광 / Chromosphere/TR UV spectroscopy
  │
2020  DKIST — 지상 4 m, 적응광학 / Ground 4 m, AO
  │
2020  Solar Orbiter/PHI — 근일점 편광측정 / Perihelion polarimetry
```

---

## 6. 다른 논문과의 연결 / Connections to Other Papers

| 논문 / Paper | 관계 / Connection |
|---|---|
| #3 Lyot (1944) | SOT/NFI의 tunable Lyot 필터의 원리적 기초. Lyot가 발명한 복굴절 필터가 NFI의 핵심 분광 소자 / Foundational principle for SOT/NFI's tunable Lyot filter |
| #12 Kosugi et al. (2007) | Hinode 미션 전체 개요. SOT는 Hinode의 3개 망원경 중 하나 / Hinode mission overview. SOT is one of Hinode's three telescopes |
| #13 Scherrer et al. (2012) | SDO/HMI — SOT/SP와 상보적. HMI는 전일면 but 낮은 분해능, SOT/SP는 고분해능 but 좁은 FOV / SDO/HMI — complementary. HMI: full-disk but lower resolution; SOT/SP: high-res but narrow FOV |
| Suematsu et al. (2008) | OTA 상세 기술 — 본 논문의 동반 논문 / OTA detailed description — companion paper |
| Tarbell et al. (2008) | FPP 상세 기술 — 본 논문의 동반 논문 / FPP detailed description — companion paper |
| Shimizu et al. (2008) | 영상 안정화 시스템 상세 — 본 논문의 동반 논문 / Image stabilization system detailed — companion paper |
| Ichimoto et al. (2008) | 편광 교정 상세 — 본 논문의 동반 논문 / Polarization calibration detailed — companion paper |
| Lites et al. (2008) | SP를 사용한 quiet Sun 수평 자기장 발견 — SOT의 대표적 과학 성과 / Quiet Sun horizontal field discovery using SP — SOT's landmark science result |

---

## 7. 참고문헌 / References

- Tsuneta, S. et al., "The Solar Optical Telescope for the Hinode Mission: An Overview", *Solar Physics*, Vol. 249, pp. 167–196, 2008. [DOI: 10.1007/s11207-008-9174-z](https://doi.org/10.1007/s11207-008-9174-z)
- Kosugi, T. et al., "The Hinode (Solar-B) Mission: An Overview", *Solar Physics*, Vol. 243, pp. 3–17, 2007.
- Suematsu, Y. et al., "The Solar Optical Telescope of Solar-B (Hinode): The Optical Telescope Assembly", *Solar Physics*, Vol. 249, pp. 197–220, 2008.
- Tarbell, T. D. et al., "The Solar Optical Telescope for the Hinode Mission: The Focal Plane Package", *Solar Physics*, Vol. 249, pp. 167–196, 2008.
- Shimizu, T. et al., "Image Stabilization System for Hinode (Solar-B) Solar Optical Telescope", *Solar Physics*, Vol. 249, pp. 221–232, 2008.
- Ichimoto, K. et al., "Polarization Calibration of the Solar Optical Telescope onboard Hinode", *Solar Physics*, Vol. 249, pp. 233–261, 2008.
- Parker, E. N., "Nanoflares and the Solar X-Ray Corona", *Astrophysical Journal*, Vol. 330, pp. 474–479, 1988.
- Lites, B. W. et al., "The Horizontal Magnetic Flux of the Quiet-Sun Internetwork as Observed with the Hinode Spectro-Polarimeter", *Astrophysical Journal*, Vol. 672, pp. 1237–1253, 2008.
- Scherrer, P. H. et al., "The Helioseismic and Magnetic Imager (HMI) Investigation for the Solar Dynamics Observatory (SDO)", *Solar Physics*, Vol. 275, pp. 207–227, 2012.
