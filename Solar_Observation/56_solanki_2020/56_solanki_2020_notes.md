---
title: "The Polarimetric and Helioseismic Imager on Solar Orbiter (SO/PHI)"
authors: [Solanki, S. K., del Toro Iniesta, J. C., Woch, J., Gandorfer, A., Hirzberger, J., Alvarez-Herrero, A., et al.]
year: 2020
journal: "Astronomy & Astrophysics"
doi: "10.1051/0004-6361/201935325"
topic: Solar_Observation
tags: [solar_orbiter, magnetograph, helioseismology, fabry_perot, LCVR, milne_eddington, RTE_inversion, fe_6173]
status: completed
date_started: 2026-04-25
date_completed: 2026-04-25
---

# 56. The Polarimetric and Helioseismic Imager on Solar Orbiter (SO/PHI) / 솔라 오비터의 편광·헬리오사이즈몰로지 영상기

---

## 1. Core Contribution / 핵심 기여

SO/PHI is the first space-based magnetograph and helioseismic imager that observes the Sun from outside the Sun-Earth line. Mounted on ESA's Solar Orbiter (launched 10 February 2020), it samples the Fe I 617.3 nm photospheric absorption line through a tunable LiNbO3 Fabry-Pérot etalon (free spectral range FSR=0.301 nm, FWHM=106 mÅ, finesse 30) and modulates polarisation through liquid crystal variable retarders (LCVRs) — the first time LCVRs have been space-qualified for science use. The instrument provides simultaneous maps of continuum intensity ($I_c$), the full magnetic field vector $(B,\gamma,\phi)$, and line-of-sight velocity ($v_{LOS}$) using two co-aligned telescopes that share a common filtergraph: the **Full Disc Telescope** (FDT, 17.5 mm aperture, 2° FOV) for full-disc context at all orbital phases, and the **High Resolution Telescope** (HRT, 140 mm aperture, 0°.28 FOV) which reaches **~200 km spatial resolution at the 0.28 AU perihelion**, comparable to the best ground-based diffraction-limited observations.

SO/PHI는 ESA의 Solar Orbiter (2020년 2월 10일 발사)에 탑재되어 **지구-태양선 바깥에서 태양을 관측하는 사상 최초의 우주 자기장계 및 헬리오사이즈몰로지 영상기** 이다. 광구의 자기장 진단선인 Fe I 617.3 nm를, 조정 가능한 LiNbO3 Fabry-Pérot 에탈론 (FSR=0.301 nm, FWHM=106 mÅ, finesse=30)으로 sampling 하고, 액정 가변 위상지연자 (LCVR; 우주 과학 미션에서 처음으로 사용)로 편광 변조한다. 두 망원경 — **FDT** (17.5 mm, 2° FOV)와 **HRT** (140 mm, 0°.28 FOV) — 가 공통 filtergraph 경로를 공유하며, HRT는 0.28 AU 근일점에서 **태양 표면 ~200 km 분해능** 을 달성한다. 그 결과는 연속체 강도 $I_c$, 자기장 벡터 $(B,\gamma,\phi)$, 시선 속도 $v_{LOS}$의 동시 맵이다.

The most distinctive engineering contribution is that SO/PHI is the **first space spectropolarimeter to invert the radiative transfer equation (RTE) on-board** — using a Milne-Eddington atmosphere and dedicated FPGA-based hardware (Cobos Carrascosa et al. 2014-2016) — in order to fit within Solar Orbiter's tight 20 kbits/s telemetry allocation. The on-board RTE inverter processes 3500 Stokes profiles per second from the 2048×2048-pixel APS detector, returning 5 physical parameter maps per pixel (and yielding ~5× compression vs raw Stokes data). This in-flight pipeline includes dark/flat-field correction, Fourier filtering for thermo-mechanical defocus, polarimetric demodulation with FOV- and temperature-dependent demodulation matrices, residual crosstalk correction, ME inversion, and CCSDS 122.0-B-1 image compression. The instrument was developed by an international consortium (Germany, Spain, France) and successfully integrated into the Solar Orbiter spacecraft at Airbus Defence and Space, Stevenage.

가장 차별화된 엔지니어링 기여는, SO/PHI가 **방사 전달 방정식 (RTE) 인버전을 우주에서 (in-flight) 직접 수행한 사상 최초의 spectropolarimeter** 라는 점이다. Milne-Eddington 가정 하에서 전용 FPGA (Cobos Carrascosa et al. 2014-2016)가 초당 3500개 Stokes profile을 처리하며, 픽셀당 5개 물리량 맵을 산출 (raw Stokes 대비 ~5배 텔레메트리 절감). 이는 Solar Orbiter의 빠듯한 20 kbits/s 텔레메트리 한계를 극복하기 위한 결정적 설계 선택이며, 다크/플랫 보정, 푸리에 필터링 (열-기계 디포커스 보정), FOV·온도 의존 demodulation matrix를 사용한 편광 demodulation, 잔여 crosstalk 보정, ME 인버전, CCSDS 122.0-B-1 영상 압축을 모두 비행 중 수행한다. 이 기기는 독일·스페인·프랑스 국제 컨소시엄에 의해 개발되었고, Airbus Defence and Space (Stevenage, UK)에서 Solar Orbiter 우주선에 성공적으로 통합되었다.

---

## 2. Reading Notes / 읽기 노트

### Part I: Science Objectives (Sect. 2) / 과학 목표

The paper organises SO/PHI's science around the four top-level Solar Orbiter questions plus four extra questions enabled by viewing geometry beyond the Sun-Earth line. Each is mapped to a specific SO/PHI capability.

논문은 SO/PHI의 과학 목표를 솔라 오비터의 4대 핵심 질문 + SO/PHI의 황도면 이탈 관측이 가능하게 하는 4개의 추가 질문으로 정리한다. 각 질문은 SO/PHI의 구체적 측정 능력에 매핑된다.

**Q1: How does the solar dynamo work? / 태양 다이나모는 어떻게 작동하는가?**

SO/PHI uniquely contributes here through (a) **stereoscopic helioseismology**: combining SO/PHI Doppler with SDO/HMI from Earth view yields skip distances up to half a circumference, reaching the tachocline at $0.7 R_\odot$ where the dynamo is thought to operate (Fig. 1). (b) **High-latitude polar field measurements**: at viewing angles 35° from the limb (vs 7° from Earth), granulation contrast and Stokes V signal-to-noise improve dramatically (Fig. 2 — Hinode/SOT-SP comparison). The paper notes the polar magnetic flux at activity minimum is the best predictor of the next solar cycle (Schatten et al. 1978; Petrovay 2010).

SO/PHI는 다음 두 가지 측면에서 다이나모 연구에 독특하게 기여한다. (a) **스테레오 헬리오사이즈몰로지**: SO/PHI 도플러 + SDO/HMI를 결합하면 음파의 "skip distance"가 한 둘레의 절반까지 늘어나, 다이나모가 작동한다고 추정되는 $0.7\,R_\odot$ 의 타코클라인을 직접 탐사할 수 있다 (Fig. 1). (b) **고위도 극자기장 관측**: 림에서 35° 시야각 (지구에서는 7°)에서 granulation 대비와 Stokes V SNR이 극적으로 개선됨 (Fig. 2 — Hinode/SOT-SP 비교). 활동 극소기의 극자기 플럭스는 다음 사이클 강도의 가장 좋은 예측 변수이다.

**Q2: Solar wind origins / 태양풍 기원**

SO/PHI provides photospheric vector magnetograms that serve as boundary conditions for force-free coronal field extrapolations (Wiegelmann & Sakurai 2012). Co-rotation phases close to perihelion allow the same active region to be tracked for far longer than from Earth (5-10 days vs 1 day before solar rotation carries it away). Polar plume footpoint observations during high-latitude phases are unique to SO/PHI.

SO/PHI는 광구 벡터 magnetogram을 제공하여 코로나 자기장 force-free extrapolation의 경계 조건을 만들고, 근일점 근처의 co-rotation 단계에서 동일 활동 영역을 5-10일간 추적할 수 있다 (지구 관측 1일 vs). 고위도 단계의 극 플룸 footpoint 관측은 SO/PHI의 독점 영역이다.

**Q3: Solar transients & heliospheric variability / 태양 trahsients와 헬리오스피어 변동성**

CMEs and flares are studied through high-cadence vector magnetograms (helicity, current sheets) combined with EUI imaging and the heliospheric in-situ instruments. SO/PHI provides the photospheric input for ICME flux-rope reconstructions sampled later by the in-situ suite (MAG, SWA, RPW, EPD).

CME와 플레어를 고해상도 벡터 magnetogram으로 연구 (helicity, current sheet) 하며, EUI 영상 및 in-situ 관측기들과 결합한다. SO/PHI는 ICME flux rope 재구성을 위한 광구 입력을 제공하며, 이는 후속 SO 통과 시 in-situ 기기 (MAG, SWA, RPW, EPD) 들이 직접 sampling한다.

**Q4: Solar energetic particles (SEPs) / 태양 고에너지 입자**

Photospheric magnetograms allow tracing field lines from in-situ SEP detection back to acceleration sites on the Sun. SO/PHI Doppler maps additional help locate downward-streaming particle effects.

광구 magnetogram은 in-situ SEP 감지로부터 태양 표면 가속 영역까지의 자기력선 추적을 가능하게 한다. SO/PHI 도플러 맵은 하향 전류 입자 효과의 위치 결정에도 기여한다.

**Q5-8: SO-only science / SO 추가 과학**

Beyond the four core questions, SO/PHI uniquely enables (Q5) solar irradiance measurements from off-ecliptic viewpoints (using the SATIRE-3D model, Yeo et al. 2017), (Q6) magnetoconvection studies via co-temporal proper motion + Doppler from two viewpoints (resolves the 180° azimuth ambiguity intrinsic to Zeeman observations), (Q7) Wilson depression height differences from stereoscopic photometry, and (Q8) far-side magnetogram support for space weather forecasting (a first-ever capability).

추가로 SO/PHI는 (Q5) 황도면 외 시점에서의 태양 irradiance 측정 (SATIRE-3D 모델 활용), (Q6) co-temporal proper motion + Doppler를 두 시점에서 결합하여 Zeeman 관측에 내재된 180° azimuth ambiguity 해소, (Q7) Wilson depression 높이 차이 측정, (Q8) 우주 기상 예보를 위한 사상 최초의 태양 뒷면 magnetogram을 가능하게 한다.

### Part II: Functional Principle (Sect. 3) / 동작 원리

The instrument combines four operations on every photon of the Fe I 617.3 nm passband (Fig. 3):

(1) **Imaging** at 2048×2048 pixels through one of two telescopes (HRT or FDT, selected by the Feed-Select Mechanism FSM).
(2) **Spectroscopy** by sequentially tuning the LiNbO3 etalon to 6 wavelength positions: 5 inside the line at $\lambda - \lambda_0 \in [-140,-70,0,+70,+140]$ mÅ plus 1 continuum point at $\pm 300$ mÅ (sign chosen based on orbital Doppler shift, range $\pm 23.6$ km/s = $\pm 486.9$ mÅ).
(3) **Polarimetry** by recording 4 polarisation modulation states per wavelength (LCVR retardances cycle through values listed in Table 5: e.g., (315°, 234.74°), (315°, 125.26°), (225°, 54.74°), (225°, 305.26°)).
(4) **On-board demodulation, calibration, and ME inversion** to deliver $I_c, B, \gamma, \phi, v_{LOS}$ maps.

기기는 Fe I 617.3 nm 협대역 빛에 대해 4가지 기능을 결합한다 (Fig. 3):

(1) **영상**: 2048×2048 픽셀 APS sensor에 두 망원경 중 하나 (HRT 또는 FDT, Feed-Select Mechanism FSM이 선택)
(2) **분광**: LiNbO3 에탈론을 6개 파장 위치 $[-140,-70,0,+70,+140]$ mÅ + 연속체 1점 ($\pm 300$ mÅ; 궤도 도플러 천이 $\pm 23.6$ km/s = $\pm 486.9$ mÅ에 따라 결정)에 sequentially 튜닝
(3) **편광계측**: 파장당 4개 LCVR 변조 상태 기록 (Table 5의 위상지연자 패턴)
(4) **On-board demodulation, calibration, ME 인버전**으로 $I_c, B, \gamma, \phi, v_{LOS}$ 맵 산출

The total cycle time depends on the number of polarisation cycles $N_P$ accumulated for SNR. From Table 6: $N_P=1$ gives 45.5 s/cycle, but the optimum $N_P=16$ takes 79.76 s for HRT (fitting the 60 s minimum cadence constraint). Single-exposure SNR is $S/N_{single}=255$; required map SNR is $10^3$ (noise floor $10^{-3} I_c$ in Q,U,V), giving $N_{acc}=16$ (Eq. 1).

전체 사이클 시간은 SNR 누적용 편광 사이클 수 $N_P$에 의존한다 (Table 6). $N_P=1$ 이면 45.5초/사이클이지만, 최적 $N_P=16$ 은 HRT에서 79.76초 (60초 최소 cadence 조건 부합). 단일 노출 SNR $S/N_{single}=255$, 목표 맵 SNR $10^3$ (Q,U,V 노이즈 $10^{-3} I_c$) 이면 식 (1)에서 $N_{acc}=16$.

### Part III: Optical Unit (Sect. 4) / 광학 유닛

**4.1.1 Why a Fabry-Pérot etalon (not Michelson)?** MDI and HMI use Michelson interferometers, but the Solar Orbiter Doppler swing ($\pm$487 mÅ) requires a tunable filter spanning multiple bandpass widths — too constraining for resonance absorption cells. Air-spaced classical Fabry-Pérot etalons (multi-pass between two mirrors) are too heavy and shock-sensitive for space. Solid-state LiNbO3 etalons solve this: single piece of crystal, intrinsically aligned, voltage-tunable through the electro-optic effect (combined refractive index change and piezo-electric strain). The 351.1 mÅ/kV tuning constant means a $\pm 1.3$ to $+2.0$ kV swing covers the full required range.

**왜 Fabry-Pérot? Michelson이 아닌가?** MDI/HMI는 Michelson 간섭계를 사용하나, Solar Orbiter의 도플러 스윙 ($\pm$ 487 mÅ) 은 multiple bandpass 폭의 가변 필터를 요구한다 — resonance absorption cell로는 불가능. 공기-갭 Fabry-Pérot은 우주에서 너무 무겁고 진동에 약하다. 솔리드 LiNbO3 에탈론은 이 모두를 해결: 단일 결정 조각으로 본질적으로 정렬되고, 전기광학 효과 (굴절률 변화 + 피에조 변형) 로 전압 튜닝 가능. 튜닝 상수 351.1 mÅ/kV로 $-1.3$ 에서 $+2.0$ kV 스윙이 필요 범위를 모두 커버.

**4.1.2 Telecentric vs collimated mounting**: SO/PHI uses a **telecentric** etalon configuration. In telecentric, each image point sees only a small etalon area but with infinite pupil distance — uniform spectral characteristics across the FOV but spectral purity limited by F-ratio. The high refractive index of LiNbO3 (~2.3) tolerates much steeper light cones than air-spaced etalons; SO/PHI runs at F/56.6 (HRT) or F/63.5 (FDT) at the etalon focus, vs ~F/150 for classical Fabry-Pérot. The collimated alternative would induce unacceptable etalon blueshift across the wide observed FOV.

**Telecentric vs collimated mounting**: SO/PHI는 **telecentric** 구성. 각 영상점이 에탈론의 작은 영역만 보지만 동공 거리는 무한 — FOV 전반에 균일한 spectral 특성을 가지나 spectral purity는 F-ratio에 의해 제한. LiNbO3의 고굴절률 (~2.3)은 공기-갭보다 훨씬 가파른 광원뿔을 허용; SO/PHI는 etalon focus에서 F/56.6 (HRT) 또는 F/63.5 (FDT) 사용 (전통 Fabry-Pérot ~F/150 대비). collimated 대안은 넓은 FOV에서 etalon blueshift가 너무 커져 부적합.

**4.2.1 HRT optical layout (Fig. 6, 7)**: The HRT is a decentred Ritchey-Chrétien with an off-axis aperture (no central obstruction), 140 mm aperture, primary focal length 2475 mm. A Barlow magnifier (4 lenses, doubles as refocus mechanism) brings the effective focal length to 4125 mm at the science focal plane (and 7920 mm at the etalon focus). The decentred design avoids secondary mirror obstruction and prevents direct sunlight on the secondary. The "parent" symmetric system would have a 480 mm aperture; only the 170 mm decentred portion is realised. F# at the science focus is 29.5, giving 0".5/pixel sampling (matched to diffraction limit at 617 nm for a 14 cm telescope; angular resolution 0".15/pixel at 0.28 AU).

**4.2.4 FDT optical layout (Fig. 9, 10)**: The FDT is an on-axis refractive system with a 17.5 mm external entrance pupil, 579 mm effective focal length, 2° round FOV (full disc at perihelion). Plate scale is 3".75/pixel = 761 km/pixel at 0.28 AU. The optical paths join at the Feed-Select Mechanism (FSM, Fig. 13), a fold mirror that selects HRT or FDT and blocks the unused channel.

**HRT 광학** (Fig. 6, 7): Off-axis 140 mm 구경 Ritchey-Chrétien (중앙 obstruction 없음), 주거울 focal length 2475 mm. Barlow magnifier (4 렌즈, refocus 메커니즘 겸용) 가 effective focal length를 과학 초점면에서 4125 mm, etalon focus에서 7920 mm로 만든다. Decentred 설계는 secondary mirror 차단을 회피하고 직사광이 secondary에 닿지 않게 한다. 모-시스템은 480 mm 구경 대칭, 170 mm decentred 부분만 실현.

**FDT 광학** (Fig. 9, 10): 17.5 mm 외부 entrance pupil, focal length 579 mm, 2° round FOV. Plate scale 3".75/pixel = 0.28 AU에서 761 km/pixel.

**4.2.6 Heat Rejecting Entrance Windows (HREWs)**: Solar Orbiter sees up to 13 solar constants (17.5 kW/m²) at 0.28 AU. HREWs are multi-coated SUPRASIL plates: UV blocker + high-pass + low-pass + IR blocker. The combined filter passes only a 30 nm passband around 617 nm — only 3.2% of incoming solar energy enters the instrument. The HRT HREW is 9.5 mm thick × 262 mm⌀, the FDT HREW is 9 mm thick × 95 mm⌀. Mounted in titanium flanges with steel spiral springs to compensate thermal stress. Operating temperatures −29°C to +204°C (HRT HREW), −55°C to +243°C (FDT HREW).

**HREW**: Solar Orbiter는 0.28 AU에서 13 solar constants (17.5 kW/m²) 까지 받는다. HREW는 다층 코팅 SUPRASIL: UV blocker + high-pass + low-pass + IR blocker. 결합 필터는 617 nm 근방 30 nm 밴드만 통과시켜 입사 태양 에너지의 3.2%만 기기 내로 진입.

**4.2.7 Polarisation Modulation Package (PMP) (Fig. 16, 17)**: Each of HRT and FDT has its own PMP. Each PMP consists of two anti-parallel nematic LCVRs at 45° to each other, followed by a linear polariser at 0° relative to the first LCVR fast axis. The 4 modulation states are listed in Table 5. LCVRs are temperature-stabilised to ±0.5°C via PID control with PT100 sensor + 4 W heater foil. **This is the first time LCVR technology is used on a space science mission** (validated by the IMaX magnetograph on the Sunrise balloon flights, Martínez Pillet et al. 2011).

**PMP** (Fig. 16, 17): HRT/FDT 각각 자체 PMP를 가진다. 두 anti-parallel nematic LCVR (45° 상호 회전) + 선형 편광자 (첫 LCVR fast axis와 0°). 4개 변조 상태는 Table 5에 나열. LCVR은 PT100 센서 + 4 W 히터 + PID 제어로 ±0.5°C에 안정화. **LCVR 기술이 우주 과학 미션에서 사용된 사상 최초의 사례** (Sunrise 풍선 비행 IMaX magnetograph로 사전 검증).

**4.2.8 Filtergraph (etalon, Fig. 18, 19)**: The filtergraph contains the LiNbO3 etalon (40×40 mm useful area, 50 mm⌀, produced by CSIRO Australia using ion deposition, Gensemer & Farrant 2014), two prefilters (FWHM 0.27 and 10 nm respectively, by Materion USA), and field lenses FL1/FL2. The etalon is held at 66°C with **±0.3 mK rms** stability over 4 hours (corresponds to wavelength stability of $1.03×10^{-5}$ Å rms = 0.5 m/s Doppler error). The high voltage power supply (HVPS, $-2.6$ kV to $+3.9$ kV range) has 1.3 V rms time stability (0.45 mÅ shift, 22 m/s rms Doppler). Combined photon-noise-equivalent velocity error <100 m/s rms.

**Filtergraph** (Fig. 18, 19): LiNbO3 에탈론 (40×40 mm 유효, 50 mm⌀, 호주 CSIRO 제작) + 2개 prefilter (FWHM 0.27 nm, 10 nm) + field lenses. 에탈론은 66°C에서 4시간 동안 **±0.3 mK rms** 안정성 (파장 안정성 $1.03×10^{-5}$ Å rms = 0.5 m/s 도플러 오차). HVPS ($-2.6$ kV ~ $+3.9$ kV) 시간 안정성 1.3 V rms.

**4.2.9 Focal Plane Array (FPA)**: 2048×2048 APS sensor (CMOSIS, Belgium), 10 μm pixel pitch, 11 fps, 14-bit ADCs, 65% well-fill at $10^5 e^-$ full well. Detector cooled via cold finger to ≤ −25°C operationally.

**FPA**: 2048×2048 APS 센서 (CMOSIS, 벨기에), 10 μm 픽셀 피치, 11 fps, 14-bit ADC, $10^5 e^-$ full well의 65% 충전. cold finger로 운영 시 ≤ −25°C 냉각.

**4.2.10 Image Stabilisation System (ISS)**: A correlation-tracker camera (Star1000 sensor, 600 fps, 64×64-128×128 pixel) sees 2.8% of HRT light via a beam-splitter behind the active mirror. It compares to a reference image (updated every ~60 s) and feeds a tip-tilt mirror at 30 Hz bandwidth. Required residual jitter: <1/20 pixel between exposures (essential for differential polarimetry to avoid spurious polarisation).

**ISS**: Correlation tracker 카메라 (Star1000, 600 fps, 64×64-128×128 픽셀) 가 active mirror 뒤 빔 분리기에서 HRT 빛의 2.8%를 받는다. 참조 영상 (~60초 업데이트) 과 비교해 30 Hz 대역폭의 tip-tilt 거울에 피드백. 필요 잔여 jitter: 노출 간 1/20 픽셀 이하.

### Part IV: Electronics Unit (Sect. 5) / 전자 유닛

The E-Unit comprises 6 modules in an Aluminium 7075 housing, totaling 6 kg (Fig. 30): PCM (Power Converter, main+redundant), DPU (Data Processing Unit), AMHD (Analogue Motor & Heater Driver), TTC (Tip/Tilt Controller), HVPS (High-Voltage Power Supply for etalon, $-5$ to $+5$ kV), and the Electric Distribution System (EDS).

**5.2 Data Processing Unit (DPU)**: A SoC (System on Chip) using a Cobham Gaisler GR712RC LEON-3FT processor + a Microsemi RTAX FPGA (system supervisor) + **two Xilinx Virtex-4 FPGAs reconfigurable in flight** (FPGA#1, FPGA#2). 1 GiB volatile SDRAM + 512 GiB non-volatile NAND flash. The DPU is the hardware foundation for SO/PHI's most ambitious feature: in-flight reconfiguration of FPGAs to switch between **(a) acquisition + ISS control, (b) preprocessing + demodulation, (c) RTE inversion, (d) image compression**.

**DPU**: System-on-Chip 구조 — Cobham Gaisler GR712RC LEON-3FT 프로세서 + Microsemi RTAX FPGA + 비행 중 재구성 가능한 **Xilinx Virtex-4 FPGA 2개** (FPGA#1, FPGA#2). 1 GiB 휘발성 SDRAM + 512 GiB 비휘발성 NAND. SO/PHI의 가장 야심찬 기능 — 비행 중 FPGA 재구성으로 (a) 획득 + ISS 제어, (b) 전처리 + demodulation, (c) RTE 인버전, (d) 영상 압축의 4단계를 차례로 수행 — 의 하드웨어 기반.

**5.4 HVPS**: Provides −5 kV to +5 kV differential to the etalon, with $\pm 5$-kV reversible polarity (avoiding etalon polarisation drift), 300 V/s slew rate (capped to protect LiNbO3 from electric stress), 1.3 V rms stability.

**HVPS**: 에탈론에 $-5$ kV ~ $+5$ kV 차동 전압 공급, 극성 반전 가능 (에탈론 편극 drift 방지), 300 V/s slew rate (LiNbO3 전기 스트레스 보호 위해 제한), 1.3 V rms 안정성.

### Part V: Calibration and Characterisation (Sect. 6) / 보정 및 특성화

**6.1 On-ground polarimetric calibration**: Done at MPS using a Polarimetric Calibration Unit (PCU) with a linear polariser + quarter-wave retarder (the same unit used for HMI). 4×36 input states fitted to a model. Resulting **FOV-averaged polarimetric efficiencies $\boldsymbol{\epsilon = [0.9917, 0.5697, 0.5666, 0.5745]}$** for $(I, Q, U, V)$, very close to the theoretical optimum of $(1, 0.5774, 0.5774, 0.5774)$ for a balanced modulator.

**On-ground 편광 보정**: MPS에서 Polarimetric Calibration Unit (PCU; 선형 편광자 + quarter-wave 위상지연자, HMI와 같은 장비)으로 수행. 4×36 입력 상태를 모델에 적합. **FOV 평균 편광 효율 $\epsilon = [0.9917, 0.5697, 0.5666, 0.5745]$** for $(I,Q,U,V)$ — 이상적 최적값 $(1, 0.5774, 0.5774, 0.5774)$ 에 매우 근접.

**Spectral characterisation**: Etalon FWHM 106 ± 5 mÅ, FSR 0.301 nm, finesse 30. Wavelength sensitivity to voltage: $351.1 ± 1.0$ mÅ/kV. Sensitivity to temperature: $37.9 ± 4.9$ mÅ/K. Fig. 37 shows the line scan over Fe I 6173 Å made during HRT ground calibration in vacuum (lit by 53 cm coelostat). Fig. 38 shows the etalon cavity thickness map across the detector — large-scale trend reflects the 1 AU solar rotation pattern (HRT FOV covers nearly the full 1 AU disc).

**Spectral 특성**: 에탈론 FWHM 106 ± 5 mÅ, FSR 0.301 nm, finesse 30. 전압 감도 $351.1 ± 1.0$ mÅ/kV, 온도 감도 $37.9 ± 4.9$ mÅ/K. Fig. 37은 진공 (코일로스타트 53 cm) 에서 HRT 지상 보정 시 측정한 Fe I 6173 Å 선 스캔. Fig. 38은 검출기 전체에 걸친 에탈론 cavity thickness map — 대규모 trend는 1 AU 태양 자전 패턴 (HRT FOV가 1 AU 디스크 거의 전체를 커버).

### Part VI: Scientific Operations and On-board Pipeline (Sect. 7) / 과학 운영 및 비행 중 파이프라인

**7.2 Twelve system states**: off, boot, safe, idle, observational_idle, observation, process_sci, process_heater, process_cal, process_anneal, annealing, debug. Allows known-state operation despite limited 150 telecommand/day budget. Auto-entry to safe state on anomaly.

**12개 시스템 상태**: off, boot, safe, idle, observational_idle, observation, process_sci, process_heater, process_cal, process_anneal, annealing, debug. 일일 150 telecommand 제한 하에서도 알려진 상태로 운영 보장. 이상 시 자동 safe 상태 진입.

**Table 4 — 8 science operating modes**:

| Mode | Description | Telescope | Cadence (min⁻¹) |
|---|---|---|---|
| 0 | nominal ($I_c, v_{LOS}, B, \gamma, \phi$) | FDT/HRT | 1 to 1/60 |
| 1 | vector ($B, \gamma, \phi$ only) | FDT/HRT | 1 to 1/60 |
| 2 | magnetograph ($I_c, v_{LOS}, B_{LOS}$) | FDT/HRT | 1 to 1/60 |
| 3 | global helioseismology ($I_c, v_{LOS}$) | FDT/HRT | 1 |
| 4 | synoptic ($I_c, v_{LOS}, B, \gamma, \phi$) | FDT | 1/240 to 1/1440 |
| 5 | burst ($I_c$) | HRT | 60 |
| 6 | raw_data (24 raw images) | FDT/HRT | 1 to 1/60 |
| LL | low latency ($I_c, B_{LOS}$) | FDT | 1/day |

**7.4 On-board pipeline (Fig. 40, 41)**: The most novel SO/PHI feature. Pipeline blocks (Fig. 40):
1. Load raw science data (24 frames per dataset)
2. Optional cropping
3. Load + apply dark field
4. Load + apply flat field (Kuhn et al. 1991 method via tip-tilt motion for HRT; off-pointing for FDT)
5. Load Fourier filter; FFT, optional binning, Fourier filtering, FFT⁻¹ (per single image, repeats 24 times) — to remove thermo-mechanical defocus, etalon cavity drift, etc.
6. Load demodulation matrices (4th-order polynomial fits with 15 parameters per matrix element, T-dependent) and apply polarimetric demodulation
7. Residual crosstalk correction (Eq. 2):

$$Q_{measured} = Q_{corr} + a V_{measured},\quad U_{measured} = U_{corr} + b V_{measured}$$

8. Re-sorting and conversion to floating point → stream to **RTE inversion** core

The RTE inverter has 5 modes: full ME inversion, classical estimations (centre-of-gravity for $B_{LOS}, v_{LOS}$ + weak-field for B,γ,φ), RTE+classical-as-initial-guess (the **default mode**), longitudinal-only, no-polarisation-modulation. ME inversion uses Singular Value Decomposition of a correlation matrix and analytical response functions (Orozco Suárez & Del Toro Iniesta 2007), iterating up to 128 times (default 15). Throughput: **3500 Stokes profiles/second** for the 2048² detector.

**비행 중 파이프라인** (Fig. 40, 41): SO/PHI 가장 혁신적 기능. 파이프라인 블록 (Fig. 40):
1. 24장 원시 영상 로드
2. (선택) 영역 자르기
3. 다크 필드 적용
4. 플랫 필드 적용 (HRT는 Kuhn et al. 1991 tip-tilt 방법; FDT는 off-pointing)
5. Fourier 필터 로드; 영상별 FFT → optional binning → Fourier filtering → FFT⁻¹ (24회 반복) — 열-기계 디포커스, 에탈론 cavity drift 제거
6. Demodulation matrices (4차 다항식 적합, 행렬 원소당 15개 파라미터, 온도 의존) 적용
7. 잔여 crosstalk 보정 (Eq. 2)
8. Re-sorting, floating point 변환 → **RTE 인버전** 코어

RTE 인버터 모드 5가지: full ME inversion, classical estimations, RTE+classical-as-initial-guess (**기본 모드**), longitudinal-only, no-polarisation-modulation. ME 인버전은 correlation matrix의 SVD + 분석적 response function 사용, 최대 128 반복 (기본 15회). 처리량: **초당 3500개 Stokes profile**.

**Compression**: CCSDS 122.0-B-1 standard, applied to 16-bit integer parameter maps, achieves 4-5 bits/pixel (factor 3-4 compression). For raw data (24 images), can compress by factor 5 without violating SNR. Compression performed on FPGA#1, ~30× faster than software on a space-qualified LEON3.

**압축**: CCSDS 122.0-B-1 표준, 16-bit 정수 파라미터 맵에 적용해 4-5 bits/pixel (3-4배 압축). 원시 데이터 (24장) 도 SNR 유지하며 5배 압축 가능. FPGA#1에서 수행, 우주급 LEON3 소프트웨어 대비 ~30배 빠름.

### Part VII: Conclusions / 결론

The paper closes (Sect. 8) by identifying five **novel concepts that SO/PHI verified through technology development for first-time application in space**:
1. LiNbO3 Fabry-Pérot etalons as electrically tunable narrow-band filters
2. LCVR-based polarisation analysers
3. APS sensor for space solar magnetography
4. On-board RTE inversion
5. HREW multi-coating design qualified for high solar flux exposure

논문 결론 (Sect. 8) 은 SO/PHI가 **우주 응용을 위해 처음 검증한 5가지 혁신 개념** 을 정리한다:
1. 전기 조정 협대역 필터로서의 LiNbO3 Fabry-Pérot
2. LCVR 기반 편광 분석기
3. 우주 태양 자기장계용 APS 센서
4. On-board RTE 인버전
5. 고태양 플럭스에 대한 HREW 다층 코팅 설계

---

## 3. Key Takeaways / 핵심 시사점

1. **First out-of-ecliptic magnetograph / 황도면 외 첫 자기장계** — SO/PHI is the first ever space magnetograph designed to observe the Sun from outside the Sun-Earth line, with heliographic latitudes reaching ~33° in the late mission phase. This breaks the 60-year ecliptic-plane monopoly of solar magnetography (post-Babcock 1953). 솔라 오비터의 황도면 이탈 궤도 (최대 ~33° 헬리오그래픽 위도) 와 SO/PHI를 결합하여, Babcock 1953 이래 60년간 지속된 황도면 독점에서 태양 자기장 관측을 해방시킨다.

2. **Two telescopes, one detector / 두 망원경 + 하나의 검출기** — HRT (140 mm aperture, 0°.28 FOV, 200 km res. at 0.28 AU) and FDT (17.5 mm aperture, 2° FOV, full disc) share a single 2048² APS detector via the Feed-Select Mechanism. This dual-telescope design is unique among solar magnetographs and gives both context and detail in the same instrument. HRT (140 mm 구경, 0.28 AU에서 200 km 분해)와 FDT (17.5 mm 구경, 2° FOV, 풀 디스크) 가 Feed-Select Mechanism을 통해 단일 2048² APS 검출기를 공유. 이 이중 망원경 설계는 태양 자기장계 중 유일.

3. **First in-flight RTE inversion / 비행 중 RTE 인버전 첫 적용** — SO/PHI is the first space spectropolarimeter to perform full Milne-Eddington RTE inversion on-board, using FPGAs that process 3500 Stokes profiles/second. This is essential because Solar Orbiter's tight 20 kbits/s telemetry budget (raw data would need ~300 kbits/s) makes raw-Stokes downlink impossible at science cadence. SO/PHI는 우주에서 전 ME RTE 인버전을 최초로 비행 중 수행 (FPGA에서 초당 3500 Stokes profile). Solar Orbiter의 20 kbits/s 텔레메트리 한계 (raw 데이터는 ~300 kbits/s 필요)에서 raw-Stokes downlink가 불가능하므로 필수.

4. **LiNbO3 etalon enables wide tuning / LiNbO3 에탈론으로 넓은 튜닝 범위** — The electro-optically tunable LiNbO3 Fabry-Pérot (FSR 0.301 nm, FWHM 106 mÅ, finesse 30) achieves $\pm 627$ mÅ tuning over $-1.3$ to $+2.0$ kV. This is needed to compensate the spacecraft's $\pm 23.6$ km/s ($\pm 487$ mÅ) Doppler swing along the elliptical orbit, which Michelson interferometers (used in MDI/HMI) cannot match. 전기광학 LiNbO3 Fabry-Pérot (FSR 0.301 nm, FWHM 106 mÅ)이 $-1.3$ ~ $+2.0$ kV 스윙으로 $\pm 627$ mÅ 튜닝 — 솔라 오비터의 $\pm 23.6$ km/s 도플러 보정에 필수, MDI/HMI의 Michelson은 불가능.

5. **First space-qualified LCVR polarisation modulator / 우주 검증 LCVR 첫 사용** — The PMP uses two anti-parallel nematic LCVRs at 45° + linear polariser, achieving FOV-averaged polarimetric efficiencies $\epsilon = [0.99, 0.57, 0.57, 0.57]$ — within 1% of the theoretical optimum. LCVRs eliminate moving parts (vs rotating waveplates), and SO/PHI is their first qualification for a space science mission. 두 anti-parallel nematic LCVR (45° 회전) + 선형 편광자로 구성된 PMP가 FOV 평균 편광 효율 $[0.99, 0.57, 0.57, 0.57]$ 달성 — 이상적 최적값과 1% 이내 차이. 회전 waveplate 대비 가동부 없음. 우주 과학 미션 첫 LCVR 검증.

6. **Extreme thermal management / 극한 열 관리** — SO/PHI faces 13 solar constants (17.5 kW/m²) at 0.28 AU. HREWs reject 96.8% of incoming flux yet pass the science 30 nm passband. The etalon must be held at 66°C with **±0.3 mK rms** stability over 4 hours (corresponding to 0.5 m/s rms Doppler error) — comparable to the best ground laboratory standards but in deep space. SO/PHI는 0.28 AU에서 13 solar constants (17.5 kW/m²) 노출. HREW는 입사 플럭스의 96.8%를 차단하면서도 과학 30 nm 밴드는 통과. 에탈론은 66°C에 $\pm 0.3$ mK rms (4시간) 안정성 유지 — 도플러 0.5 m/s rms 에 해당, 지상 최고 표준급을 deep space에서 달성.

7. **Stereoscopic helioseismology unlocks the tachocline / 스테레오 헬리오사이즈몰로지로 타코클라인 탐사** — Combining SO/PHI Doppler with SDO/HMI yields skip distances up to half a circumference, enabling the first direct probing of the tachocline at $0.7 R_\odot$ where the dynamo is thought to live. Single-vantage helioseismology achieves only ~38 Mm spatial resolution at latitude 75°; stereoscopic data approaches the diffraction limit. SO/PHI + SDO/HMI 결합으로 음파 skip distance가 한 둘레의 절반에 도달, 다이나모가 작동한다고 추정되는 $0.7 R_\odot$ 타코클라인 첫 직접 탐사 가능. 단일 시점은 위도 75°에서 ~38 Mm 분해능, 스테레오는 회절 한계 근처.

8. **Far-side imaging for space weather / 태양 뒷면 영상으로 우주 기상 예보** — During phases when Solar Orbiter views the Sun's far side from Earth, SO/PHI provides the **first-ever direct magnetograms of the solar far side**, allowing emerging active regions to be identified before they rotate into Earth view. This significantly enhances CME propagation predictions and SEP source identification. 솔라 오비터가 지구로부터 태양 뒷면을 보는 단계에서, SO/PHI가 **사상 최초의 태양 뒷면 직접 magnetogram** 을 제공 — 신규 활동 영역이 지구 시야로 자전해 들어오기 전에 식별 가능. CME 전파 예측과 SEP 발생원 식별에 결정적.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Required SNR and accumulation count / 요구 SNR과 누적 수

The fundamental detector relation for polarimetry is

$$
N_{acc} = \frac{1}{4}\left(\frac{S/N}{\bar\epsilon\, S/N_{single}}\right)^2
$$

with parameters / 다음 파라미터를 가진다:
- $S/N$: required final SNR per pixel per Stokes parameter ($10^3$ for $10^{-3}I_c$ noise floor)
- $S/N_{single}$: single-exposure SNR = $\sqrt{Q\cdot f \cdot N_{electrons}} = 255$ where $Q\cdot f = 0.59$ is detector quantum-efficiency × fill-factor product and $N_{electrons}=0.65\cdot10^5$ (65% of full well)
- $\bar\epsilon$: mean polarimetric efficiency of (Q,U,V) = 0.5703

Substituting: $N_{acc} = \frac{1}{4}(1000/(0.57\cdot 255))^2 \approx 16$ frames per polarisation/wavelength state.

The factor 1/4 comes from the four-state modulation cycle (each Stokes parameter is recovered as a linear combination of 4 measurements, averaging the noise by $\sqrt{4}=2$ in each direction → $1/4$ in variance).

### 4.2 Cycle time / 사이클 시간

For HRT a complete dataset takes (Table 6, $N_P=16$):
- 24 (4 polarisation × 6 wavelengths) × 11 fps frames = 24×16 = 384 raw exposures
- Plus LCVR switching ($t_P$ ~ 19-95 ms, see Table 5) and etalon tuning time
- Total: $t_{cycle}(N_P=16) = 79.76$ s

Required minimum cadence is 60 s, so $N_P=8$ ($t_{cycle}=61.48$ s) is preferred for nominal mode. Higher $N_P$ improves SNR but at the cost of cadence.

### 4.3 Etalon physics / 에탈론 물리

The Fabry-Pérot transmission profile (Airy function) is

$$
T(\lambda) = \frac{1}{1 + F\sin^2(\delta/2)},\quad \delta = \frac{4\pi n d \cos\theta}{\lambda}
$$

with finesse $\mathcal{F}^{*} = \pi\sqrt{F}/2$. For SO/PHI:
- $n d \approx$ optical thickness (LiNbO3 refractive index ~2.3, physical thickness ~mm)
- FSR $\Delta\lambda_{FSR} = \lambda^2/(2nd) = 0.301$ nm at 617 nm
- FWHM $\Delta\lambda_{FWHM} = \Delta\lambda_{FSR}/\mathcal{F}^* = 106$ mÅ → finesse $\mathcal{F}^* = 30$

Wavelength tuning via electro-optic effect:

$$
\frac{\partial\lambda}{\partial V} = 351.1 \pm 1.0 \text{ mÅ/kV},\quad \frac{\partial\lambda}{\partial T} = 37.9 \pm 4.9 \text{ mÅ/K}
$$

Required temperature stability for 0.5 m/s Doppler error ($1.03\times 10^{-5}$ Å):

$$
\Delta T_{req} = \frac{1.03\times 10^{-5}}{37.9\times 10^{-3}} \approx 2.7\times 10^{-4}\text{ K} = 0.27 \text{ mK}
$$

— matching the achieved ±0.3 mK rms.

### 4.4 Milne-Eddington RTE / ME RTE

The polarised RTE in vector form is

$$
\frac{d\mathbf{I}}{d\tau} = \mathbf{K}(\mathbf{I} - \mathbf{S})
$$

where $\mathbf{I}=(I,Q,U,V)^T$, $\mathbf{S}=(S,0,0,0)^T$ is the unpolarised source, and $\mathbf{K}$ is the 4×4 absorption matrix:

$$
\mathbf{K} = \begin{pmatrix} \eta_I & \eta_Q & \eta_U & \eta_V \\ \eta_Q & \eta_I & \rho_V & -\rho_U \\ \eta_U & -\rho_V & \eta_I & \rho_Q \\ \eta_V & \rho_U & -\rho_Q & \eta_I \end{pmatrix}
$$

Under the **Milne-Eddington approximation** the source function is linear, $S(\tau) = S_0 + S_1 \tau$, and the absorption coefficients $\eta_i, \rho_i$ are independent of optical depth. Then the RTE has the analytic Unno-Rachkovsky solution:

$$
\mathbf{I}(0) = S_0\,\hat{e}_I + S_1 \mathbf{K}^{-1}\,\hat{e}_I
$$

(at the surface $\tau=0$). The 9 free parameters fit by SO/PHI's RTE inverter are:
1. $B$ — magnetic field strength (typically Gauss)
2. $\gamma$ — inclination from LOS (degrees)
3. $\phi$ — azimuth (degrees, with 180° ambiguity)
4. $v_{LOS}$ — line-of-sight velocity (km/s)
5. $\eta_0$ — line-to-continuum opacity ratio
6. $\Delta\lambda_D$ — Doppler width
7. $a$ — damping parameter
8. $S_0$ — source function at $\tau=0$
9. $S_1$ — source function gradient

Plus continuum intensity $I_c$ from the dedicated continuum sample at $\pm 300$ mÅ.

### 4.5 V→Q,U crosstalk correction / V→Q,U 교차오염 보정

Fitting linear relations in continuum regions (where Q,U,V should be zero):

$$
Q_{measured}(x,y) = Q_{corr}(x,y) + a\, V_{measured}(x,y)
$$

$$
U_{measured}(x,y) = U_{corr}(x,y) + b\, V_{measured}(x,y)
$$

with $a, b$ determined by least-squares over the full FOV. The coefficients capture residual V leakage into Q,U from imperfect demodulation matrix calibration.

### 4.6 FDT focus metric / FDT 초점 평가량

For automated re-focusing of FDT (where solar limb dominates the image):

$$
\delta I = \frac{1}{\langle I\rangle \sum_{i,j} M_{i,j}}\sum_{i,j}\left[\left(\frac{\partial I}{\partial x}\right)^2 + \left(\frac{\partial I}{\partial y}\right)^2\right] M_{i,j}
$$

where $M_{i,j}$ is an annular binary mask containing the defocused solar limb and $\langle I\rangle$ is the spatial mean. The focus position maximising $\delta I$ is selected. For HRT (no limb in the FOV), focus uses rms contrast of granulation.

### 4.7 Worked numerical example / 수치 예시

**Required RMS jitter for differential polarimetry**: SO/PHI takes 4 polarisation states sequentially, each $\sim 32$ ms (FDT) or $\sim 24$ ms (HRT). Spurious polarisation arises if image shifts between exposures cause structures to fall on different pixels. To keep spurious polarisation below the $10^{-3}$ noise floor, image shift between consecutive exposures must be:

$$
\Delta x_{rms} < \frac{1}{20} \text{ pixel} = \frac{10\,\mu m}{20} = 0.5\,\mu m \text{ at detector}
$$

This translates to ~0.025 arcsec for HRT (plate scale 0".5/pixel × 0.05). At Solar Orbiter pointing accuracy ~10 arcsec, the ISS must reduce jitter by a factor ~400 — achieved by the 30 Hz tip-tilt loop locking on cross-correlation with a reference image.

**Required photon flux**: 24 ms exposure × 11 fps duty + readout, with $0.65\times 10^5 e^-/$pixel. For an HRT pixel of 0".5×0".5 = 0.0625 arcsec², solar surface brightness at 617 nm gives ~$10^9$ photons/cm²/s/arcsec². Combined with the HRT 140 mm aperture and 7.5% efficiency: $\sim 10^9 \times \pi (7)^2 \times 0.0625 \times 0.075 \times 0.024 \approx 1.5\times 10^7$ electrons/pixel/exposure — far in excess of the 65k required. Hence the HREW + prefilter combination must attenuate by ~$10^{2.5}$ on top of the 30 nm spectral cut.

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1908 ─── Hale: First sunspot Zeeman magnetic field detection
1953 ─── Babcock: First photographic magnetograph (ground)
1995 ─── SOHO/MDI launched (Michelson interferometer, Ni I 6768 Å, full disc)
2006 ─── Hinode/SP launched (Fe I 6301/6302 Å, slit spectrograph, 0".3 res.)
2009 ─── Sunrise-1 balloon flight (IMaX: prototype LiNbO3 + LCVR imager)
2010 ─── SDO/HMI launched (Michelson, Fe I 6173 Å, 1" resolution, full disc)
2013 ─── Sunrise-2 balloon flight (IMaX validates LCVR space-readiness)
2017 ─── DKIST first light planning (4 m, ground-based)
2018 ─── Parker Solar Probe launched (no remote-sensing magnetograph)
2020 ─── ★ Solar Orbiter launched (10 Feb 2020) — SO/PHI is the magnetograph
2022 ─── First SO/PHI close perihelion (~0.32 AU)
2024 ─── Solar Orbiter resonant orbit raises inclination to 17°
2025 ─── First high-latitude phase
2029+ ── Mission extended phases reach >30° heliographic latitude
2030+ ── Solar-C/EUVST (no magnetograph but inherits SO/PHI on-board concepts)
```

SO/PHI sits at the intersection of three technological lineages: (1) the **MDI→HMI line** of full-disc Doppler/magnetograph instruments, (2) the **Hinode-SP line** of high-resolution Stokes vector polarimeters, and (3) the **Sunrise/IMaX line** of LiNbO3+LCVR balloon-borne spectropolarimeters. It is the first space mission to fuse all three into a single instrument that operates from the harsh thermal/orbital environment of a deep-space platform.

SO/PHI는 세 기술 계보의 교차점에 있다: (1) 풀-디스크 도플러/자기장계의 **MDI→HMI 계보**, (2) 고해상도 Stokes 벡터 편광계의 **Hinode-SP 계보**, (3) LiNbO3+LCVR 풍선 분광편광계의 **Sunrise/IMaX 계보**. SO/PHI는 이 세 계보를 deep-space 플랫폼의 가혹한 열·궤도 환경에서 작동하는 단일 기기로 융합한 사상 최초의 우주 미션이다.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Scherrer et al. 1995 (SOHO/MDI) | 직계 선조 / Direct ancestor — 첫 우주 자기장계, Michelson 기반. First space magnetograph, Michelson-based. | High — SO/PHI는 MDI의 한계 (좁은 튜닝 범위, 황도면 한정) 극복을 명시. SO/PHI explicitly addresses MDI's limitations. |
| Schou et al. 2012 (SDO/HMI) | 동시대 비교 대상 / Contemporary comparison — Fe I 6173, 1" 분해, 풀 디스크. Same Fe I 6173 line, 1" resolution, full disc. | High — HMI는 SO/PHI의 stereoscopic 동반자. Stereoscopic counterpart from Earth view. |
| Tsuneta et al. 2008 (Hinode/SP) | 분광편광 정밀 비교 / Spectropolarimetric precision benchmark — 슬릿 스펙트로미터, $10^{-3}$ noise. Slit spectrograph polarimeter, $10^{-3}$ noise. | High — SO/PHI도 동일 노이즈 목표. Same noise floor target. |
| Martínez Pillet et al. 2011 (IMaX/Sunrise) | 직접 기술적 선조 / Direct technological ancestor — LiNbO3 + LCVR 첫 비행 검증. First flight validation of LiNbO3 + LCVR. | Critical — IMaX 없이는 SO/PHI 불가능. SO/PHI inherits IMaX hardware concepts directly. |
| Müller et al. 2020 (Solar Orbiter mission) | 호스트 미션 / Host mission paper — 궤도, 시점, 미션 단계. Orbit, viewpoint, mission phases. | Critical — SO/PHI 과학 목표는 SO 미션 능력에서 파생. SO/PHI science derives from SO orbit. |
| Müller & Marsden 2013 (SO Red Book) | 미션 정의 / Mission definition — 4대 핵심 과학 질문. Four top-level science questions. | High — SO/PHI 8개 과학 목표의 모태. Source of SO/PHI's 8 science objectives. |
| Cobos Carrascosa et al. 2014-2016 | RTE 인버전 알고리즘 / RTE inverter algorithm papers — FPGA 구현 상세. FPGA implementation details. | Medium — on-board ME inverter의 알고리즘 기반. Algorithmic basis for on-board ME inverter. |
| Orozco Suárez & Del Toro Iniesta 2007 | RTE 인버전 수학 기초 / Mathematical foundation — analytical response functions, SVD. | Medium — SO/PHI ME 인버터 수학 기초. Mathematical basis used by SO/PHI's inverter. |
| del Toro Iniesta 2003 (textbook) | 분광편광 표준 교재 / Spectropolarimetry reference textbook. | Medium — Stokes 형식주의, ME 인버전 이론 표준 참조. Standard reference for Stokes formalism and ME inversion. |
| Gizon & Birch 2005 (LRSP review) | 헬리오사이즈몰로지 기초 / Local helioseismology review. | Medium — SO/PHI 과학 목표 Q1-Q3 의 이론적 기반. Theoretical basis for SO/PHI's helioseismic goals. |
| Gandorfer et al. 2018 | HRT 광학 metrology 상세 / HRT optical metrology details. | Medium — Section 4.2.1-4.2.3 의 보충 상세. Supplementary details for Sect. 4.2. |
| García-Marirrodriga et al. 2020 | Solar Orbiter 우주선 시스템 paper / Solar Orbiter S/C system paper. | Medium — 호스트 플랫폼의 환경 제약. Host platform environmental constraints. |

---

## 7. References / 참고문헌

- **Primary**: Solanki, S. K., del Toro Iniesta, J. C., Woch, J., Gandorfer, A., Hirzberger, J., Alvarez-Herrero, A., et al., "The Polarimetric and Helioseismic Imager on Solar Orbiter", A&A 642, A11 (2020). DOI: 10.1051/0004-6361/201935325.
- Babcock, H. W., "The Solar Magnetograph", ApJ 118, 387 (1953).
- Cobos Carrascosa, J. P., Aparicio del Moral, B., Ramos Más, J. L., et al., "RTE inversion FPGA designs", various IEEE/SPIE conferences (2014-2016).
- del Toro Iniesta, J. C., *Introduction to Spectropolarimetry*, Cambridge University Press (2003).
- del Toro Iniesta, J. C. & Collados, M., "Optimum Modulation and Demodulation Matrices for Solar Polarimetry", Appl. Opt. 39, 1637 (2000).
- García-Marirrodriga, C., Pacros, A., Strandmoe, S., et al., "Solar Orbiter: Mission and spacecraft design", A&A 642 (2020).
- Gandorfer, A., Grauf, B., Staub, J., et al., "The high-resolution telescope of SO/PHI", SPIE 10698 (2018).
- Gensemer, S. D. & Farrant, D., "Lithium niobate Fabry-Pérot filter for narrow-band imaging", Adv. Opt. Technol. 3, 309 (2014).
- Gizon, L. & Birch, A. C., "Local Helioseismology", Living Rev. Solar Phys. 2, 6 (2005).
- Kuhn, J. R., Lin, H., & Loranz, D., "Gain-correction algorithm for solar magnetograph data", PASP 103, 1097 (1991).
- Martínez Pillet, V., del Toro Iniesta, J. C., Álvarez-Herrero, A., et al., "The Imaging Magnetograph eXperiment (IMaX) for the Sunrise balloon-borne solar observatory", Sol. Phys. 268, 57 (2011).
- Müller, D., St. Cyr, O. C., Zouganelis, I., et al., "The Solar Orbiter mission. Science overview", A&A 642, A1 (2020).
- Müller, D. & Marsden, R. G., "Solar Orbiter — Exploring the Sun-heliosphere connection", Tech. rep. ESA (2013).
- Orozco Suárez, D. & Del Toro Iniesta, J. C., "The usefulness of analytic response functions", A&A 462, 1137 (2007).
- Scherrer, P. H., Bogart, R. S., Bush, R. I., et al., "The Solar Oscillations Investigation – Michelson Doppler Imager", Sol. Phys. 162, 129 (1995).
- Schou, J., Scherrer, P. H., Bush, R. I., et al., "Design and ground calibration of the Helioseismic and Magnetic Imager (HMI) instrument on the Solar Dynamics Observatory (SDO)", Sol. Phys. 275, 229 (2012).
- Tsuneta, S., Ichimoto, K., Katsukawa, Y., et al., "The Solar Optical Telescope for the Hinode mission", Sol. Phys. 249, 167 (2008).
