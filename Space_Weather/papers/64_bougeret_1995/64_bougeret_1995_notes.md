---
title: "WAVES: The Radio and Plasma Wave Investigation on the WIND Spacecraft"
authors: J.-L. Bougeret, M. L. Kaiser, P. J. Kellogg, R. Manning, K. Goetz, S. J. Monson, N. Monge, L. Friel, C. A. Meetre, C. Perche, L. Sitruk, S. Hoang
year: 1995
journal: "Space Science Reviews"
doi: "10.1007/BF00751331"
topic: Space_Weather
tags: [WIND, WAVES, radio-instrument, plasma-waves, solar-wind, type-III-bursts, Langmuir-waves, thermal-noise, goniopolarimetry, neural-networks]
status: completed
date_started: 2026-04-25
date_completed: 2026-04-25
---

# 64. WAVES: The Radio and Plasma Wave Investigation on the WIND Spacecraft / WIND 위성의 전파·플라즈마 파동 관측기

---

## 1. Core Contribution / 핵심 기여

**English.** Bougeret et al. (1995) is the *instrument paper* for **WIND/WAVES**, a comprehensive radio and plasma-wave package covering electric fields from a fraction of a Hz to 13.825 MHz and magnetic fields up to ~3 kHz. The paper documents the design philosophy, hardware architecture, signal processing chain, calibration, and operational modes of five distinct receivers that share six antennas (three orthogonal electric dipoles of 100 m, 15 m, 12 m tip-to-tip, and three orthogonal magnetic search coils). The five receivers are: (i) the **FFT receiver** (DC–10 kHz), implementing a 1024-point floating-point FFT with 144 dB theoretical dynamic range across three sub-bands; (ii) the **Thermal Noise Receiver** (4–256 kHz), using FIR digital filters in five logarithmically spaced bands with 32 quasi-log compressed channels each, equivalent to a discretized wavelet transform; (iii) **RAD1** (20–1040 kHz) and (iv) **RAD2** (1.075–13.825 MHz), both dual super-heterodyne receivers with 256 programmable hop frequencies and SUM/SEP antenna combination modes for goniopolarimetry; and (v) the **Time Domain Sampler** (waveform capture at up to 120 ksps with 1 Mbit memory). Two innovations distinguish WAVES from prior heritage (ISEE-3, Voyager, Ulysses URAP): (a) the *first use of onboard neural networks* on a scientific spacecraft, embedded as feed-forward back-propagation networks transformed into integer-domain lookup tables, used both for plasma-line band selection in the TNR and for TDS event triage; and (b) wavelet-transform-like real-time spectral analysis through the constant-Q FIR filter banks.

**Korean.** Bougeret 등(1995)은 NASA WIND 위성의 종합 전파·플라즈마 파동 관측기 **WAVES**의 *기기 설계 논문*이다. 1 Hz 미만에서 13.825 MHz까지의 전기장과 3 kHz까지의 자기장을 측정하며, 본 논문은 다섯 개의 수신기가 여섯 개의 안테나(100 m, 15 m, 12 m tip-to-tip의 직교 전기 쌍극자 셋, 직교 자기 서치코일 셋)를 공유하는 설계 철학, 하드웨어 구조, 신호 처리 사슬, 보정, 운용 모드를 전부 기술한다. 다섯 수신기는: (i) **FFT 수신기**(DC–10 kHz) — 1024점 부동소수점 FFT, 3개 서브밴드, 이론 동적범위 144 dB; (ii) **열잡음 수신기**(TNR, 4–256 kHz) — 로그 등간격 5밴드, 각 32채널, FIR 디지털 필터, 의사-로그 압축; 이는 이산 웨이블릿 변환과 등가; (iii) **RAD1**(20–1040 kHz) + (iv) **RAD2**(1.075–13.825 MHz) — 이중 슈퍼헤테로다인, 256개 프로그램 주파수, 회전 위성 goniopolarimetry용 SUM/SEP 안테나 결합 모드; (v) **TDS**(시간 영역 샘플러) — 최대 120 ksps 파형, 1 Mbit 메모리. 선행 기기(ISEE-3, Voyager, Ulysses URAP)와 차별화되는 두 혁신은: (a) *과학 위성 최초의 온보드 신경망* — 정수 영역 룩업 테이블로 구현된 순방향 역전파 망, TNR 플라즈마 라인 밴드 선택 및 TDS 이벤트 선별에 사용; (b) 상수 Q FIR 필터 뱅크를 통한 *실시간 웨이블릿 유사* 스펙트럼 분석.

---

## 2. Reading Notes / 읽기 노트

### Part I: Investigation Objectives (§1, p. 231–233) / 관측 목적

**English.** The opening section frames WAVES within four science thrusts:

- **§1.1 Plasma physics.** *In situ* measurements of magneto-acoustic, ion-cyclotron, whistler, electron-plasma, and electron-noise modes near plasma boundaries (e.g., the bow shock). The TDS continues the dissipation-process program begun by URAP/Ulysses (Stone et al. 1992), where strong/weak turbulence theories of Langmuir-wave dissipation through parametric decay, soliton formation, collapse, and electron tails remain conflicting. WIND TDS time resolution is sufficient to *waveform* most Langmuir waves and study them statistically.
- **§1.2 In situ electron diagnosis from thermal noise analysis.** Following ICE/Ulysses heritage (Meyer-Vernet & Perche 1989, used at comet Giacobini-Zinner and Jupiter), the TNR will recover n_e from <1 cm⁻³ to 500 cm⁻³ and T_e from 10³ to 10⁶ K, with cadence of every 1.5 s (half spin).
- **§1.3 Remote sensing of Earth's magnetosphere.** AKR (Gurnett 1974), NTC (Gurnett 1975), ITKR (Steinberg et al. 1990), foreshock plasma-frequency emission (Burgess et al. 1987) and 2f_p (Lacombe et al. 1988) are all in the WAVES frequency window.
- **§1.4 Remote sensing of the interplanetary medium.** Tracking shocks and electron streams from ~3–4 R⊙ to 1 AU. The 1–14 MHz range (RAD2) was previously *poorly covered* and is precisely where most large interplanetary shocks form. Coordinate with ground-based ARTEMIS (30 MHz–500 MHz) at Thermopiles, Greece.

**Korean.** 도입부는 WAVES를 네 과학 축으로 정의한다:

- **§1.1 플라즈마 물리.** 활꼴 충격면 등 플라즈마 경계에서 자기음향파, 이온 사이클로트론파, 휘슬러파, 전자 플라즈마 진동, 전자 잡음 등 *in situ* 관측. TDS는 URAP/Ulysses(Stone et al. 1992)에서 시작된 소산 과정 연구 — 파라메트릭 붕괴, 솔리톤 형성, 붕괴, 전자 꼬리에 의한 Langmuir 파 소산의 강·약 난류 이론 — 을 이어받으며, WIND TDS의 시간 분해능은 대부분 Langmuir 파를 *파형* 수준으로 통계 연구할 만큼 충분하다.
- **§1.2 열잡음 분석에 의한 in situ 전자 진단.** ICE/Ulysses 유산(Meyer-Vernet & Perche 1989, 자코비니-지너 혜성, 목성 응용)을 따라, TNR은 n_e를 1 cm⁻³ 미만에서 500 cm⁻³까지, T_e를 10³–10⁶ K까지, 1.5 s(반 자전) 간격으로 결정한다.
- **§1.3 자기권 원격 관측.** AKR, NTC, ITKR, 전조 충격 플라즈마 주파수 방출 및 2f_p 모두 WAVES 주파수 창 안에 든다.
- **§1.4 행성간 매질 원격 관측.** 충격파와 전자 흐름을 ~3–4 R⊙에서 1 AU까지 추적. RAD2의 1–14 MHz 대역은 *기존 관측이 부족했던* 영역이며, 대부분의 거대 행성간 충격파가 형성되는 영역이다. 그리스 Thermopiles의 지상 ARTEMIS 전파망(30–500 MHz)과 협력 관측.

### Part II: Instrument Overview (§2, p. 233–235) / 기기 개요

**English.** WAVES is a tri-national effort: Paris-Meudon (DESPA), University of Minnesota, and NASA/Goddard. The block diagram (Fig. 1) shows: three electric-dipole antenna systems → preamplifiers (×2 outputs each: unity gain for high-frequency, shared gain for low-frequency, unity gain for TDS+DC) → three magnetic search coils → five receiver subsystems → DPU → spacecraft. Total power 28 W with 68% converter efficiency. Frequency-coverage map (Fig. 2) shows the receivers logarithmically tile the spectrum from ~0.01 Hz (TDS lowest) through ~14 MHz (RAD2 highest), with deliberate overlap in the TNR–RAD1 transition (256 kHz–1040 kHz overlap region). Sub-band assignments: FFT-Low (0.3–170 Hz), FFT-Mid (5.5 Hz–2.275 kHz), FFT-High (22 Hz–11 kHz), TNR (4–256 kHz, 5 sub-bands), RAD1 (20–1040 kHz), RAD2 (1.075–13.825 MHz), TDS (waveforms covering all of the above).

**Korean.** WAVES는 3국 협력 프로젝트(파리-뫼동 DESPA, 미네소타 대학, NASA 고다드)이다. 블록 다이어그램(그림 1): 세 개의 전기 쌍극자 안테나 → 전치증폭기(각 2 출력: 고주파용 1배 이득, 저주파용 공유 이득, TDS+DC용 1배 이득) → 세 개의 자기 서치코일 → 다섯 수신기 → DPU → 위성. 총 전력 28 W, 컨버터 효율 68%. 주파수 커버리지 지도(그림 2)는 ~0.01 Hz(TDS 최저)부터 ~14 MHz(RAD2 최고)까지 로그 등간격으로 타일링하며, TNR–RAD1 전환부(256 kHz–1040 kHz)에 의도적 중첩이 있다. 서브밴드 배정: FFT-Low (0.3–170 Hz), FFT-Mid (5.5 Hz–2.275 kHz), FFT-High (22 Hz–11 kHz), TNR (4–256 kHz, 5 서브밴드), RAD1 (20–1040 kHz), RAD2 (1.075–13.825 MHz), TDS (위 모두 파형 캡처).

### Part III: Instrumentation (§3, p. 235–255) / 기기 상세

#### §3.1 Sensors / 센서

**English.** The two spin-plane electric dipoles are motor-driven wire type, deployed by centrifugal force. **Ex = 100 m** tip-to-tip (longest, used by FFT and TNR for low-noise diagnostics) and **Ey = 15 m** tip-to-tip (shorter, used by RAD2 because Ex is unusable above its full-wave resonance ≈ 3.3 MHz). The **Ez = 12 m** spin-axis dipole is rigid (motor-driven flexible extended tube) for spin-stability. The triaxial magnetic search coil is similar to that on POLAR. The 6 antenna units total 27.5 kg of sensor mass (Table II).

**Korean.** 자전면 전기 쌍극자 두 개는 모터 구동 와이어형으로, 원심력으로 전개된다. **Ex = 100 m** tip-to-tip(가장 긺, 저잡음 진단을 위해 FFT와 TNR이 사용); **Ey = 15 m** tip-to-tip(짧음, Ex가 전파장 공명 약 3.3 MHz 위로는 쓸 수 없으므로 RAD2가 사용); **Ez = 12 m** 자전축 쌍극자는 자전 안정성 때문에 강성(모터 구동 신축 튜브). 삼축 자기 서치코일은 POLAR와 유사. 안테나 6개 합계 27.5 kg(표 II).

#### §3.2 Preamplifiers and photoemission compensation / 전치증폭기와 광방출 보상

**English.** Preamps are at the antenna base (minimizing base capacity). Each preamp has three outputs: (1) near-unity gain for RAD2; (2) shared gain for TNR/RAD1/RAD2/FFT/TDS; (3) unity gain for TDS DC channel. **Critical design issue**: in sunlight the wire emits photoelectrons; in shadow it does not. Antenna potential swings ≥10 V per spin, antenna-plasma resistance varies ~10×. Solution: bias each antenna through a programmable resistor (Ex: 10 MΩ to 1 GΩ; Ey: 50 MΩ to 10 GΩ; Ez: 50 MΩ) and programmable voltage source (−10 V to +10 V in 256 steps). When antenna points to/away from Sun (high resistance), the bias supplies replacement photoelectrons; when perpendicular (low resistance), bias is minimal. The same resistors are used for in-flight impedance measurement.

**Korean.** 전치증폭기는 안테나 베이스에 위치(베이스 용량 최소화). 각 전치증폭기는 세 출력: (1) RAD2용 거의 1배 이득, (2) TNR·RAD1·RAD2·FFT·TDS 공유 이득, (3) TDS DC 채널용 1배 이득. **핵심 설계 문제**: 태양광 하에서는 와이어가 광전자를 방출하지만 그림자에서는 방출 안 함 ⇒ 안테나 전위가 자전당 10 V 이상 흔들리고, 안테나-플라즈마 저항이 약 10배 변동. 해법: 각 안테나를 프로그램 가능한 저항(Ex: 10 MΩ–1 GΩ; Ey: 50 MΩ–10 GΩ; Ez: 50 MΩ)과 −10 V에서 +10 V(256단계)의 프로그램 전압원으로 바이어싱. 안테나가 태양 방향(저항 높음)일 때 바이어스가 대체 광전자를 공급하고, 수직일 때(저항 낮음) 바이어스가 최소. 같은 저항으로 비행 중 임피던스 측정.

#### §3.3 FFT receiver / FFT 수신기

**English.** Microprocessor-controlled FFT processor (Fig. 3a–f). Three frequency bands: high (22 Hz–11 kHz), medium (5 Hz–2.7 kHz), low (0.3–170 Hz; synchronized to spin rate). Each FFT channel has anti-aliasing low-pass filter → "floating-point" ADC → 1024-pt FFT.

The **floating-point ADC** is the design jewel. Three cascaded ×16 amplifiers feed an analog switch with selectable gains (×1, ×16, ×256, ×4096) to a 12-bit linear ADC. An analog comparator picks the highest-gain stage that does not saturate. The 12-bit mantissa + 2-bit exponent (4 gain steps) gives:

$$\text{DR}_{\text{theory}} = 20\log_{10}(2^{12}\cdot 4^3) \approx 144\,\text{dB}.$$

Realised: 110 dB (mid), 128 dB (high), 72 dB (low using 12-bit straight). DSP is a TMS320C30 — 1024-point FFT in 3 ms. After FFT, the spectrum is converted to logarithmic frequency spacing (3–4 freq/octave) by averaging in frequency. Telemetered values are average and peak power in 1/2 dB steps; phases in 22.5° steps.

**Korean.** 마이크로프로세서 제어 FFT 프로세서(그림 3a–f). 세 주파수 밴드: 고(22 Hz–11 kHz), 중(5 Hz–2.7 kHz), 저(0.3–170 Hz; 자전 속도와 동기화). 각 FFT 채널은 안티앨리어싱 LPF → "부동소수점" ADC → 1024점 FFT.

**부동소수점 ADC**는 설계의 정수다. 세 ×16 증폭기 직렬 → 가변 이득(×1, ×16, ×256, ×4096) 아날로그 스위치 → 12bit 선형 ADC. 아날로그 비교기가 포화되지 않는 최고 이득 단계를 선택. 12bit 가수 + 2bit 지수(4 이득 단계):

$$\text{DR}_{\text{이론}} = 20\log_{10}(2^{12}\cdot 4^3) \approx 144\,\text{dB}.$$

실현: 110 dB(중), 128 dB(고), 72 dB(저, 12bit 직접 사용). DSP는 TMS320C30 — 1024점 FFT 3 ms. FFT 후 스펙트럼은 로그 주파수 간격(옥타브당 3–4 주파수)으로 평균화. 텔레메트리는 평균·피크 전력 1/2 dB 단계, 위상 22.5° 단계.

#### §3.4 Thermal Noise Receiver (TNR) / 열잡음 수신기

**English.** Two multi-channel receivers covering 4 kHz–256 kHz in **5 logarithmically spaced bands** (each spans 2 octaves with 1-octave overlap):

| Band | Range (kHz) | Sampling rate (kHz) | Measurement time (ms) |
|---|---|---|---|
| A | 4–16 | 64.1 | 320 |
| B | 8–32 | 126.5 | 160 |
| C | 16–64 | 255.7 | 80 |
| D | 32–128 | 528.5 | 40 |
| E | 64–256 | 1000 | 20 |

Each band is divided into 32 (or 16) logarithmically-spaced channels.

**Digital filters (FIR)** were first used in the RETE/TSS-1 experiment. They operate at sampling frequencies above 1 MHz, with passbands of 4.4% or 9% of channel center frequency. Out-of-band rejection >45 dB (8-bit digitization). 32 channels are calculated *simultaneously*. Outputs compressed to 8-bit quasi-log words (avg 0.375 dB resolution). **This processing is equivalent to a wavelet transform analysis** (paper's words, p. 245).

The basic measurement cycle is 20 ms at 1 MHz sampling. The **plasma-frequency tracking** problem: only one of the 5 bands can be telemetered at full resolution at a given time. Solution: three methods determine which band — (1) real-time density from SWE plasma experiment, (2) real-time density from 3D-PLASMA, (3) onboard *neural network* recognizing the plasma line (see §4.3).

**Korean.** 4 kHz–256 kHz를 **5개 로그 등간격 밴드**(각 2 옥타브 폭, 1 옥타브 중첩)로 덮는 다채널 수신기 두 대:

| 밴드 | 범위 (kHz) | 샘플링 (kHz) | 측정 시간 (ms) |
|---|---|---|---|
| A | 4–16 | 64.1 | 320 |
| B | 8–32 | 126.5 | 160 |
| C | 16–64 | 255.7 | 80 |
| D | 32–128 | 528.5 | 40 |
| E | 64–256 | 1000 | 20 |

각 밴드는 32(또는 16) 로그 등간격 채널로 분할.

**디지털 FIR 필터**는 RETE/TSS-1 실험에서 처음 사용됨. 1 MHz 이상 샘플링에서 동작, 통과대역은 채널 중심 주파수의 4.4% 또는 9%. 대역 외 억제 >45 dB(8bit 디지털화). 32채널 *동시* 계산. 출력은 8bit 의사로그 워드로 압축(평균 해상도 0.375 dB). **이 처리는 웨이블릿 변환 분석과 등가**(논문 본문 p. 245).

기본 측정 주기는 1 MHz 샘플링에서 20 ms. **플라즈마 주파수 추적** 문제: 동시에 한 밴드만 전체 해상도로 텔레메트리. 해법은 세 가지: (1) SWE 플라즈마 실험의 실시간 밀도, (2) 3D-PLASMA의 실시간 밀도, (3) 플라즈마 라인을 인식하는 *온보드 신경망*(§4.3 참조).

#### §3.5–3.6 RAD1 and RAD2 receivers / 라디오 수신기

**English.** Both are dual super-heterodyne receivers (Fig. 4) with Ulysses heritage.

| Property | RAD1 | RAD2 |
|---|---|---|
| Frequency span | 20 kHz–1040 kHz | 1.075 MHz–13.825 MHz |
| Number of frequencies | 256 | 256 |
| Frequency step | 4 kHz | 50 kHz |
| Nominal 3-dB BW | 3 kHz | 20 kHz |
| Acquisition time per (S, S′, Z) set | 358 ms | 63 ms |
| Sets per spin | 8S+8S′+8Z | 12S+12S′+12Z |
| Frequencies per spin | 1 | 4 |
| Steps per cycle | 64 | 48 |
| Cycle duration | 192 s | 36 s |

The **SUM mode** (Fig. 5) is the goniopolarimetric trick: signal from Ex (or Ey) is phase-shifted by +45°, signal from Ez by +45°, and they are summed before mixing — synthesizing an *inclined dipole*. A second channel uses Ex with −45° shift, summed with Ez at +45° (total 90° relative shift) for the Z channel. With three channels (S, S′, Z), the system measures source direction, source diameter, and 4 Stokes parameters (Manning & Fairberg 1980; Fainberg et al. 1985). **SEP mode** measures Ex and Ez separately for diaphony calibration.

RAD1 frequency tables (16 freq × 4 = 64 step cycles) are reprogrammable in flight; one default program covers the same frequencies as Ulysses RAR for baseline cross-calibration. RAD2 uses Ey only (longer Ex would alias above its full-wave resonance ≈ 3.3 MHz).

**Korean.** 둘 다 Ulysses 유산의 이중 슈퍼헤테로다인 수신기(그림 4).

| 항목 | RAD1 | RAD2 |
|---|---|---|
| 주파수 범위 | 20–1040 kHz | 1.075–13.825 MHz |
| 주파수 수 | 256 | 256 |
| 주파수 스텝 | 4 kHz | 50 kHz |
| 공칭 3-dB 대역폭 | 3 kHz | 20 kHz |
| (S, S′, Z) 한 세트 획득 시간 | 358 ms | 63 ms |
| 자전당 세트 수 | 8S+8S′+8Z | 12S+12S′+12Z |
| 자전당 주파수 수 | 1 | 4 |
| 한 주기 스텝 수 | 64 | 48 |
| 주기 지속 시간 | 192 s | 36 s |

**SUM 모드**(그림 5)가 goniopolarimetry의 핵심 트릭이다: Ex(또는 Ey) 신호에 +45° 위상 시프트, Ez 신호에 +45° 위상 시프트 후 혼합 전 합산 — *경사 쌍극자*를 합성. 두 번째 채널은 Ex에 −45° 시프트, Ez에 +45° 시프트(총 90° 상대 시프트)로 Z 채널 형성. 세 채널(S, S′, Z)로 원천 방향, 원천 지름, Stokes 4파라미터 측정(Manning & Fainberg 1980; Fainberg et al. 1985). **SEP 모드**는 Ex와 Ez를 분리 측정, diaphony 보정용.

RAD1의 주파수 테이블(16주파수 × 4 = 64 스텝 주기)은 비행 중 재프로그램 가능; 기본 프로그램 중 하나는 Ulysses RAR과 동일 주파수로 기준선 교차보정. RAD2는 Ey만 사용(긴 Ex는 전파장 공명 ≈ 3.3 MHz 위에서 앨리어싱).

#### §3.7 Time Domain Sampler (TDS) / 시간 영역 샘플러

**English.** Waveform capture instrument extending coverage down to 0.03 Hz. Two parts:

- **Fast sampler**: 2 channels selected from Ex, Ey, Ez. Sampling rates 120, 30, 7.5, 2 ksps. 1 Mbit memory. Sensitivity 80 μV (rms). Dynamic range 90 dB.
- **Slow sampler**: 4 channels selected from Ex, Ey, Ez, Bx, By, Bz. Sampling rates 7.5, 2, 0.5, 0.12 ksps. 1 Mbit memory. Sensitivity 300 μV (rms).

TDS samples at >2 Mbit/s but telemetry budget is only ~15 bit/s — a **gap of ~5 orders of magnitude**. Solution: hardware comparators detect peak absolute values, store the data with peak in the *center* of the recorded event (1024 samples around peak), and a microprocessor invokes a ground-loadable **neural network** that ranks event quality. Highest-quality events are packetized and transmitted. The neural net is trained to recognize specific waveform "signatures"; new training tables can be uploaded as mission goals evolve. To prevent the network from being misled, a programmable "honesty factor" sometimes selects events without ML quality consideration.

Logarithmic ADC (Fig. 6a): a 90 dB log A/D running at 120 ksps. The peak detection-and-centering is done in a 1/2-bank shift register. Maximum-detector format provides per-major-frame transient overview.

**Korean.** 0.03 Hz까지 커버리지를 확장하는 파형 캡처 기기. 두 부분:

- **빠른 샘플러**: Ex, Ey, Ez 중 2채널. 샘플링 120, 30, 7.5, 2 ksps. 1 Mbit 메모리. 감도 80 μV (rms). 동적범위 90 dB.
- **느린 샘플러**: Ex, Ey, Ez, Bx, By, Bz 중 4채널. 샘플링 7.5, 2, 0.5, 0.12 ksps. 1 Mbit 메모리. 감도 300 μV (rms).

TDS는 >2 Mbit/s 샘플링하지만 텔레메트리 예산은 ~15 bit/s — **약 5자릿수 격차**. 해법: 하드웨어 비교기가 절대값 피크 검출, 피크가 *중앙*에 오도록 데이터 저장(1024 샘플), 마이크로프로세서가 지상 로드 가능한 **신경망**으로 이벤트 품질 순위 매김. 최상위 이벤트만 패킷화하여 전송. 신경망은 특정 파형 "서명"을 인식하도록 훈련; 임무 목표 변화에 따라 새 훈련 테이블 업로드 가능. 신경망의 오도 방지를 위해 프로그램 가능한 "정직도 인자"가 가끔 ML 품질과 무관하게 이벤트 선택.

로그 ADC(그림 6a): 120 ksps, 90 dB 로그 A/D. 피크 검출·중앙 정렬은 1/2-뱅크 시프트 레지스터에서 수행. 최대값 검출 포맷은 메이저 프레임당 트랜지언트 개요 제공.

#### §3.8–3.9 DC/DC converter and EMC/ESC / 전원 변환기와 전자기·정전기 청결도

**English.** DC/DC converter is URAP-heritage (Ulysses) with separate output stages for preamps, TNR/RAD1/RAD2, FFT/TDS/search coils, and DPU — *no single failure* loses the whole instrument. The DPU output stage and oscillators are redundant.

EMC: a frequency control plan limits power-converter frequencies to narrow bands; system-level test in an EMC chamber on a wooden mount; the spacecraft was determined to be exceptionally "clean". ESC: spacecraft surface designed near-equipotential for low-frequency E-field measurements; cooperated with low-energy particle experiments (3D-PLASMA, SWE).

**Korean.** DC/DC 변환기는 URAP(Ulysses) 유산이며, 전치증폭기, TNR/RAD1/RAD2, FFT/TDS/서치코일, DPU에 대해 *별도의 출력 단계*를 두어 *단일 고장*이 전체 기기를 잃게 하지 않는다. DPU 출력 단계와 발진기는 이중화.

EMC: 주파수 제어 계획으로 전력 변환기 주파수를 좁은 대역으로 제한; 목재 받침대 위 EMC 챔버에서 시스템 시험; 위성은 *매우 청결*로 판정됨. ESC: 저주파 E-필드 측정을 위해 위성 표면을 거의 등전위로 설계; 저에너지 입자 실험(3D-PLASMA, SWE)과 협력.

### Part IV: Onboard Data Processing (§4, p. 256–260) / 온보드 데이터 처리

#### §4.1 Data Processing Unit (DPU) / 데이터 처리부

**English.** The DPU is the master processor (Sandia SA3300 / National Semiconductor NS32016). Functions:
1. Receive telecommands; 2. Send telemetry; 3–4. Control TNR + determine optimum frequency range; 5–6. Control RAD1, RAD2; 7–8. Pass commands/telemetry to TDS, FFT; 9. Send housekeeping. Specialized DSPs (TMS320C30, ADSP2100) are used in FFT and TNR. **All flight software is replaceable in-flight** by memory-load telecommand.

The DPU implements *packetized telemetry* — flexibility unavailable in fixed-format systems, allowing dynamic reallocation among RAD1/RAD2/TNR/FFT/TDS. Default packet allocation (Table VI):

| Subsystem | Low-bit rate | High-bit rate |
|---|---|---|
| RAD1 | 8% | 4% |
| RAD2 | 40% | 20% |
| TNR | 25% | 25% |
| FFT | 25% | 45% |
| TDS | 2% | 6% |

**Korean.** DPU는 마스터 프로세서(Sandia SA3300 / NS32016). 기능:
1. 텔레커맨드 수신; 2. 텔레메트리 송신; 3–4. TNR 제어 + 최적 주파수 범위 결정; 5–6. RAD1, RAD2 제어; 7–8. TDS, FFT 명령·텔레메트리 전달; 9. 하우스키핑. FFT와 TNR에는 전용 DSP(TMS320C30, ADSP2100). **모든 비행 소프트웨어는 메모리 로드 텔레커맨드로 비행 중 교체 가능**.

DPU는 *패킷화 텔레메트리*를 구현 — 고정 포맷이 못 주는 유연성, RAD1/RAD2/TNR/FFT/TDS 사이 동적 재할당 허용. 기본 패킷 할당(표 VI):

| 서브시스템 | 저비트율 | 고비트율 |
|---|---|---|
| RAD1 | 8% | 4% |
| RAD2 | 40% | 20% |
| TNR | 25% | 25% |
| FFT | 25% | 45% |
| TDS | 2% | 6% |

#### §4.2 Operational Modes / 운용 모드

**English.** Standard low-bit rate is 936 bps. The instrument and DPU flexibility allows trivial increase if telemetry becomes available. During standard survey mode the format is variable; the DPU "packetizes" data from different scientific regions.

**Korean.** 표준 저비트율은 936 bps. 텔레메트리 가용 시 손쉽게 증가 가능. 표준 측량 모드에서 포맷은 가변; DPU가 다른 과학 영역의 데이터를 *패킷화*.

#### §4.3 Use of neural networks in real-time / 실시간 신경망 사용

**English.** This is the WAVES *signature innovation*. Networks are feed-forward back-propagation, **transformed into the integer domain for speed of processing — using lookup tables rather than continuous functions**. Network coefficients are generated on the ground in real-valued domain, transformed to integers, then uploaded to the instrument.

**For TNR**: ROM coefficients were determined from Ulysses Radio Receiver (RAR) data, processed to resemble WIND TNR response. Simulations: the channel containing the thermal plasma line is determined exactly **~85% of the time** and within ±1 TNR channel **>99% of the time**. The neural net output is used alone or with SWE/3D-PLASMA plasma-line indications. The network estimates peak frequency at real-time rates so peak can be followed during shock activity. In TNR fast mode, the NN peak-finding allows intelligent point selection so plasma characteristics resolve accurately even with fractional spectral coverage. ~4 measurements of n_e per half spin (1.5 s). To avoid being misled, a programmable "honesty factor" causes periodic sweeps through all 5 bands.

**For TDS**: Simulations with rocket data show NN can select the time series containing a given signature. NN ranks events; the "best" candidate uses telemetry. Tables uploaded to RAM after launch once flight data are available. The mission can re-train and upload new coefficient tables as desired signatures evolve. To maintain integrity, TDS sometimes selects events without regard for NN quality.

**Korean.** 이것이 WAVES의 *대표적 혁신*이다. 망은 순방향 역전파 구조이며, **속도를 위해 정수 영역으로 변환되어 연속 함수 대신 룩업 테이블로 구현**된다. 망 계수는 지상에서 실수 영역으로 생성, 정수로 변환, 기기에 업로드.

**TNR**: ROM 계수는 Ulysses 전파 수신기(RAR) 데이터로 결정 후 WIND TNR 응답에 맞게 처리. 시뮬레이션 결과: 열 플라즈마 라인이 있는 채널을 *정확히* 결정하는 비율 **~85%**, ±1 채널 이내가 **>99%**. 망 출력은 단독 또는 SWE·3D-PLASMA 라인 지시와 함께 사용. 망은 실시간 속도로 피크 주파수를 추정하여 충격파 활동 중에도 피크 추적 가능. TNR 빠른 모드에서 NN의 피크 찾기로 지능적 점 선택 — 분광 커버리지가 일부분이어도 플라즈마 특성을 정확히 해석. 자전 반주기(1.5 s)당 약 4 회의 n_e 측정. 망의 오도 방지를 위해 프로그램 가능한 "정직도 인자"가 주기적으로 5 밴드를 모두 스캔.

**TDS**: 로켓 데이터로 한 시뮬레이션은 NN이 주어진 서명을 포함한 시계열을 선택할 수 있음을 보임. NN이 이벤트 순위를 매기고, "최고" 후보가 텔레메트리 사용. 발사 후 비행 데이터 가용 시 RAM에 테이블 업로드. 원하는 서명이 진화하면 새 계수 테이블 재훈련·업로드 가능. 무결성 유지를 위해 TDS는 때때로 NN 품질과 무관하게 이벤트 선택.

#### §4.4 Inter-experiment interfaces / 실험간 인터페이스

**English.** Three:
- N_e signal *from* SWE (single serial line) — used to set TNR frequency band.
- N_e signal *from* 3D-PLASMA — similar.
- E_x and E_y signals *to* 3D-PLASMA — analog mid-frequency preamp outputs for wave-particle correlation studies. The 3D-PLASMA correlator performs one-bit cross-correlation between field signals and particle counts.

**Korean.** 세 가지:
- SWE에서 *오는* N_e 신호(단일 직렬선) — TNR 주파수 밴드 설정에 사용.
- 3D-PLASMA에서 *오는* N_e 신호 — 유사.
- 3D-PLASMA로 *가는* Ex, Ey 신호 — 파장-입자 상관 연구용 아날로그 중주파 전치증폭기 출력. 3D-PLASMA 상관기가 필드 신호와 입자 카운트의 1bit 교차 상관 수행.

### Part V: Calibrations (§5, p. 260–261) / 보정

**English.** Ground calibrations (Oct–Dec 1993, U. Minnesota) measured antenna mechanism stray capacitance to ±1 pF for wire antennas, ±3 pF for axial booms (88 ± 3 pF). Receiver transfer characteristics known to 2–10%. The internal noise generator (pseudo-random, 7 frequency bands) is used for in-flight calibration.

For RAD1/RAD2: (1) gain curve "log law," (2) frequency response (BW), (3) phase. The log-law:

$$y = A_2\log_{10}\!\Big[10^{(A_1-x)/10} + 10^{-A_4(1/4-1)}\Big] + A_3.$$

A_1, A_2, A_3, A_4 are the four AGC parameters fit numerically. They have direct physical meaning (saturation, slope, offset, noise floor) so any drift is diagnostic.

For TNR: gain curve + frequency response + 0 dB level of noise generator.

In-flight: (a) effective resistance to spacecraft and to plasma (switch in known resistors and source voltages, measure the loading effect); (b) magnetic search coil calibration windings provide low-frequency cal signals computer-controlled; (c) RAD1, RAD2, TNR have an internal noise gen. with 8-step (8 dB) attenuators after preamps and at IF stages; cycle every 24 hours or by command.

**Korean.** 지상 보정(1993년 10–12월, 미네소타대): 안테나 기구의 부유 용량을 와이어 안테나에 대해 ±1 pF, 축방향 붐에 대해 ±3 pF(88 ± 3 pF)로 측정. 수신기 전달 특성은 2–10% 정확도. 내부 잡음 생성기(의사난수, 7 주파수 밴드)가 비행 중 보정에 사용.

RAD1/RAD2: (1) 이득 곡선 "log law", (2) 주파수 응답(BW), (3) 위상. 로그 법칙:

$$y = A_2\log_{10}\!\Big[10^{(A_1-x)/10} + 10^{-A_4(1/4-1)}\Big] + A_3.$$

A_1, A_2, A_3, A_4는 수치적합되는 4개 AGC 파라미터로, 직접적 물리 의미(포화, 기울기, 오프셋, 잡음 바닥)를 가져 드리프트가 진단된다.

TNR: 이득 곡선 + 주파수 응답 + 잡음 생성기 0 dB 레벨.

비행 중: (a) 위성과 플라즈마에 대한 유효 저항(알려진 저항·전압원 스위칭, 부하 효과 측정); (b) 자기 서치코일 보정 권선이 컴퓨터 제어 저주파 보정 신호 제공; (c) RAD1·RAD2·TNR은 내부 잡음 생성기 + 8단계(8 dB) 감쇠기를 전치증폭기 뒤·IF 단계에서 사용; 24시간 주기 또는 명령으로.

### Part VI: Conclusions (§6, p. 261–262) / 결론

**English.** Summary by science thrust: Geospace (solar wind, heat flux, anisotropies, bow shock incl. upstream waves), Solar/Interplanetary (source region, large-scale structures, stream interactions, impulsive phenomena), Plasma physics (instabilities, waves, turbulence, basic radiation mechanisms). Coordinated programs with Ulysses (ecliptic-plane reference) and SOHO/LASCO/EIT (corona/heliosphere overlap). The investigation explicitly aims at the Global Geospace Science (GGS) Initiative and ISTP.

**Korean.** 과학 축별 요약: Geospace(태양풍, 열속, 이방성, 활꼴 충격면 및 상류파), 태양/행성간(원천 지역, 대규모 구조, 흐름 상호작용, 충격적 현상), 플라즈마 물리(불안정성, 파동, 난류, 기본 복사 메커니즘). Ulysses(황도면 기준)와 SOHO/LASCO/EIT(코로나/태양권 중첩)와의 협력 프로그램. GGS 이니셔티브와 ISTP를 명시적으로 겨냥.

---

## 3. Key Takeaways / 핵심 시사점

1. **Multi-receiver coverage spans 7 orders of magnitude in frequency through deliberate band partitioning** — *7자릿수의 주파수 범위는 의도적 밴드 분할로 다중 수신기로 덮인다.* Each subsystem (FFT, TNR, RAD1, RAD2, TDS) is optimized for a specific phenomenon class (low-f turbulence, plasma line, kilometric radio, MHz radio, waveform), and the deliberate overlap (e.g., TNR–RAD1 at 256 kHz–1040 kHz) provides cross-calibration. The *modular* receiver philosophy is essential — no single receiver could span both DC and 14 MHz with adequate SNR. / 각 서브시스템은 특정 현상에 최적화되며, 중첩 영역으로 교차보정한다. 단일 수신기로는 DC에서 14 MHz까지 충분한 SNR을 얻을 수 없으므로 모듈식 설계가 필수다.

2. **The floating-point ADC architecture is the unsung hero of plasma-wave instrumentation** — *플라즈마 파동 관측의 숨은 영웅은 부동소수점 ADC 아키텍처이다.* Plasma-wave amplitudes vary 10⁵× between thermal noise and a Langmuir wave packet, requiring ~140 dB dynamic range — far beyond a 16-bit linear ADC's 96 dB. The cascaded ×16 amplifiers + analog comparator + small linear ADC gives 144 dB at low gate count and minimal power, a design later widely copied. / 플라즈마 파의 진폭은 열잡음과 Langmuir 파 패킷 사이에 10⁵배 차이가 있어 약 140 dB 동적범위가 필요하며 — 16bit 선형 ADC의 96 dB로는 부족 — ×16 증폭기 직렬 + 아날로그 비교기 + 소형 선형 ADC가 최소 게이트·전력으로 144 dB를 제공.

3. **A spinning spacecraft is *not* a limitation — it is the goniopolarimetric advantage** — *회전 위성은 제약이 아니라 goniopolarimetric 이점이다.* By using SUM mode (Ex+Ez phase-shifted sum) on a spinning platform, WAVES synthesizes a full 3D dipole pattern over one spin period (3 s), recovering source direction, source angular size, and four Stokes parameters from spin modulation alone — without needing physically multiple antennas. This is the WAVES instrument's most elegant trick. / SUM 모드(Ex+Ez 위상 시프트 합)를 회전 플랫폼에 적용하여 한 자전 주기(3 s) 동안 3D 쌍극자 패턴을 합성, 자전 변조만으로 원천 방향·각크기·Stokes 4파라미터를 회수. 물리적으로 다수의 안테나가 필요 없는 가장 우아한 트릭.

4. **Onboard intelligence is dictated by telemetry economics** — *온보드 지능은 텔레메트리 경제학이 결정한다.* TNR has 5 bands × 32 channels = 160 channels but only one band fits in telemetry. TDS samples at 2 Mbit/s but downlinks at 15 bit/s. The neural network is a *compression-by-selection* algorithm: pick the most scientifically valuable subset. The fact that this was implemented as integer lookup tables on a 1990s embedded CPU presaged today's edge-AI paradigm by 25 years. / TNR은 5 밴드 × 32 채널 = 160 채널이지만 한 밴드만 텔레메트리. TDS는 2 Mbit/s 샘플링하지만 15 bit/s 다운링크. 신경망은 *선택에 의한 압축* 알고리즘이며, 1990년대 임베디드 CPU에서 정수 룩업 테이블로 구현된 사실은 오늘날 엣지 AI를 25년 앞선다.

5. **Photoemission compensation is mandatory for low-frequency E-field on long wire dipoles** — *긴 와이어 쌍극자의 저주파 E-필드 측정에는 광방출 보상이 필수이다.* In sunlight a 100 m dipole emits photoelectrons; in shadow it does not, swinging antenna potential by ≥10 V per spin and changing antenna-plasma resistance by 10×. WAVES' programmable bias system (10 MΩ–10 GΩ resistors, ±10 V in 256 steps) maintains preamp linearity. Without this, FFT-Low (0.3–170 Hz) measurements would be unusable. / 100 m 쌍극자는 햇빛에서 광전자를 방출하나 그림자에서는 방출 안 하므로, 자전당 10 V 이상의 전위 흔들림과 10배의 저항 변동이 발생. WAVES의 프로그램 가능한 바이어스(10 MΩ–10 GΩ 저항, ±10 V 256단계)가 전치증폭기 선형성을 유지하며 — 없으면 FFT-Low가 무용지물.

6. **Wavelet-transform-like analysis is a natural fit for plasma-wave spectra** — *플라즈마 파동 스펙트럼에는 웨이블릿 유사 분석이 자연스럽게 들어맞는다.* The TNR's constant-Q FIR filter bank (4.4% or 9% relative bandwidth on 32 logarithmically-spaced channels per band) achieves what a wavelet transform does mathematically — time-frequency tiling with shorter windows at higher frequency. This matches plasma-wave physics: high-frequency Langmuir waves are short-lived, low-frequency MHD turbulence is slow. / TNR의 상수 Q FIR 필터 뱅크(밴드당 32 로그 등간격 채널, 상대 대역폭 4.4% 또는 9%)는 수학적으로 웨이블릿 변환과 동등 — 고주파에서 짧은 창, 저주파에서 긴 창의 시간-주파수 타일링. 플라즈마 파동 물리(고주파 Langmuir는 짧음, 저주파 MHD는 느림)와 잘 부합.

7. **Calibration with physically-meaningful AGC parameters enables long-term data quality** — *물리적 의미를 가진 AGC 파라미터로의 보정이 장기 데이터 품질을 가능케 한다.* The "log law" with parameters (A_1: saturation, A_2: log slope, A_3: offset, A_4: noise floor) maps directly to AGC circuitry components. Drift in any A_i is diagnostic of a specific component aging — saturation drift = output stage, slope drift = AGC reference, etc. This makes WIND/WAVES the rare instrument with *self-diagnosing* calibration, contributing to its >30-year operational life. / "log law"의 파라미터(A_1: 포화, A_2: 로그 기울기, A_3: 오프셋, A_4: 잡음 바닥)는 AGC 회로에 직접 대응. 어떤 A_i의 드리프트도 특정 부품 노화의 진단(포화 드리프트 = 출력 단, 기울기 드리프트 = AGC 기준 등). WIND/WAVES가 30년 넘는 운영 수명을 갖는 데 기여.

8. **Coordinated multi-spacecraft science was designed in from the start** — *다중 위성 협력 과학은 처음부터 설계에 들어가 있었다.* The paper explicitly references coordination with Ulysses (out-of-ecliptic), SOHO/LASCO/EIT (corona), ground-based ARTEMIS (30–500 MHz radio at Thermopiles, Greece), and ICE/Voyager heritage. WIND/WAVES was thought of as one node in a *network of observatories*, not a stand-alone mission. This design philosophy underpins today's heliospheric system science with PSP + Solar Orbiter + STEREO + WIND. / 논문은 Ulysses(황도면 외), SOHO/LASCO/EIT(코로나), 그리스 Thermopiles의 ARTEMIS(30–500 MHz 지상 전파), ICE/Voyager 유산과의 협력을 명시적으로 언급. WIND/WAVES는 *관측소 네트워크*의 한 노드로 구상되었으며 — 오늘날 PSP + Solar Orbiter + STEREO + WIND 태양권 시스템 과학의 토대.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 The plasma-frequency map / 플라즈마 주파수 매핑

The single most important relation in WAVES analysis is the conversion of received frequency to local electron density:

$$\boxed{\;f_p [\text{Hz}] = \frac{1}{2\pi}\sqrt{\frac{n_e e^2}{\varepsilon_0 m_e}} = 8980\,\sqrt{n_e[\text{cm}^{-3}]}\;}$$

Numerically:

| n_e (cm⁻³) | f_p (kHz) | TNR band | Region |
|---|---|---|---|
| 0.5 | 6.4 | A | distant solar wind |
| 5 | 20 | A/B | 1 AU solar wind (typical) |
| 10 | 28 | B | dense streamer belt |
| 100 | 90 | D | magnetosheath |
| 500 | 200 | E | plasmasphere edge |

The TNR's 5-band coverage (4–256 kHz) maps to n_e from 0.2 cm⁻³ to ~800 cm⁻³.

### 4.2 Floating-point ADC dynamic range / 부동소수점 ADC 동적범위

Three cascaded ×16 amplifiers + 12-bit linear ADC + 4-position analog switch (gains 1, 16, 256, 4096):

$$N_{\text{eff}} = 12\,(\text{mantissa}) + 2\,(\text{exponent})\,\Rightarrow\,\text{DR} = 20\log_{10}(2^{12}\cdot 4^3) \approx 144\,\text{dB}.$$

The realised dynamic ranges: **128 dB** (FFT high-band), **110 dB** (FFT mid-band), **72 dB** (FFT low-band, 12-bit linear), **90 dB** (TDS log A/D).

### 4.3 RAD1/RAD2 receiver log-law calibration / 로그 법칙 보정

$$y = A_2\,\log_{10}\!\left[10^{(A_1-x)/10} + 10^{-A_4(1/4-1)}\right] + A_3,$$

where x is input level (dB), y is digital output (counts), and {A_1, A_2, A_3, A_4} are physically interpretable:

- A_1: input level at saturation (knee position)
- A_2: log-domain slope (counts per dB above noise floor)
- A_3: digital offset (zero-input output)
- A_4: noise-floor curvature term

In the log-linear regime (signal well above noise floor and below saturation): y ≈ A_2 (A_1 − x)/10 + A_3 — i.e., a slope of A_2/10 counts per dB.

### 4.4 Spin modulation for goniopolarimetry / goniopolarimetry의 자전 변조

For an arriving wave with direction cosine (θ, ϕ_s) onto a synthetic inclined dipole at instantaneous spin angle ϕ:

$$P(\phi) \propto |\hat{\mathbf e}\cdot\hat{\mathbf E}_{\text{wave}}|^2 = (1-\cos^2\theta\cos^2(\phi-\phi_s))\,(I/2) + \cos^2(\phi-\phi_s)\,Q + \cdots$$

After Fourier-decomposing P(ϕ) over one spin (3 s):

- DC term: source intensity I and a function of θ
- 2nd-harmonic amplitude & phase: source longitude ϕ_s and Stokes Q
- 1st-harmonic: U / V coupling via the inclined-dipole synthesis

With three channels (S, S′, Z) sampled at multiple ϕ values per spin, the system has 3 × N samples and 4 unknowns (θ, ϕ_s, plus 4 Stokes — 5 dof per source) — overdetermined → least-squares fit.

### 4.5 TNR digital-filter sampling / TNR 디지털 필터 샘플링

For each TNR band of bandwidth B_band (octaves 2 wide), the FIR filter passband is δf/f_c = 4.4% or 9%. For 32 channels per 2 octaves:

$$\frac{f_{c,k+1}}{f_{c,k}} = 2^{2/32} = 2^{1/16} \approx 1.044,\quad k=0,\ldots,31.$$

Hence δf/f_c ≈ 4.4% gives slight overlap between adjacent channels. Sampling rate per band is set so Nyquist ≥ 2 × upper band edge:

| Band | f_low (kHz) | f_high (kHz) | Sampling (kHz) | Nyquist (kHz) |
|---|---|---|---|---|
| A | 4 | 16 | 64.1 | 32.05 |
| B | 8 | 32 | 126.5 | 63.25 |
| C | 16 | 64 | 255.7 | 127.85 |
| D | 32 | 128 | 528.5 | 264.25 |
| E | 64 | 256 | 1000 | 500 |

The Nyquist comfortably covers each band's f_high, with margin for filter roll-off.

### 4.6 TDS event triage by neural network / 신경망에 의한 TDS 이벤트 선별

Conceptually, the trained NN computes a quality score Q(w) on each 1024-sample waveform w:

$$Q(\mathbf w) = \sigma\!\Big(\sum_j W_j^{(2)}\,\sigma\!\big(\sum_i W_{ji}^{(1)} f_i(\mathbf w) + b_j^{(1)}\big) + b^{(2)}\Big),$$

where f_i are hand-engineered features (peak amplitude, kurtosis, dominant frequency, etc.), σ is a sigmoid (here approximated by a piecewise-linear lookup table for integer arithmetic), W^(1,2) and b^(1,2) are trainable weights. Events are sorted by Q; the top-k fit the telemetry budget. This is *exactly* the structure of a small modern ML classifier — implemented in 1994 with NS32016 instructions and ROM lookup.

### 4.7 Worked example: a type III burst in WAVES / 작업 예시: WAVES에서의 III형 폭발

Consider an electron beam at v_b = 0.3 c emitted from r_0 = 1.5 R⊙ at t = 0. Using a Saito-Newkirk-style coronal density model:

$$n_e(r) = A\,r^{-2} + B\,r^{-4},\quad A = 6.5\times10^{4}\,\text{cm}^{-3}\,R_\odot^2,\ B = 4\times10^{4}\,\text{cm}^{-3}\,R_\odot^4.$$

At r = 1.5 R⊙: n_e ≈ 5.7 × 10⁴ cm⁻³ ⇒ f_p = 2.15 MHz, f_2p = 4.3 MHz → captured by RAD2 (1.075–13.825 MHz).
At r = 10 R⊙: n_e ≈ 650 cm⁻³ ⇒ f_p = 230 kHz → RAD1 (20–1040 kHz).
At r = 50 R⊙: n_e ≈ 26 cm⁻³ ⇒ f_p = 46 kHz → still RAD1.
At r = 215 R⊙ (= 1 AU): n_e ≈ 5 cm⁻³ ⇒ f_p = 20 kHz → TNR band A.

Time to reach r:

$$t(r) = (r - r_0)\,R_\odot / v_b.$$

For v_b = 0.3 c: r_0 = 1.5 R⊙ (t = 0); 10 R⊙ at t = 19.7 s; 50 R⊙ at t = 112 s; 215 R⊙ at t = 495 s ≈ 8 minutes.

So a single type III burst sweeps the WAVES band from RAD2 → RAD1 → TNR over ~8 minutes — a perfect dynamic-spectrum signature observable end-to-end with WAVES, *and* with the *associated Langmuir wave* observable in TDS at f_p ≈ 20 kHz when the beam reaches WIND (3D-PLASMA confirms the beam in particles).

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1968 ── OGO-5 plasma-wave receiver (Scarf)
   │
1974 ── Gurnett: Earth as radio source — AKR identified
   │       paper #54 in this collection
1975 ── Gurnett: Non-Thermal Continuum (NTC)
   │
1977 ── Voyager 1/2 PWS launched
   │
1978 ── ISEE-1, ISEE-2, ISEE-3 launched (Helios/ICE)
   │
1980 ── Manning & Fairberg: goniopolarimetry method
   │
1985 ── Lacombe et al.: foreshock electron plasma waves
   │       Fainberg et al.: distributed source goniopolarimetry
   │
1989 ── Meyer-Vernet & Perche: thermal-noise tool kit
   │
1990 ── Ulysses launched (URAP experiment)
   │       Steinberg et al.: ITKR
   │
1992 ── Stone et al.: Unified Radio and Plasma Wave Investigation (URAP/Ulysses)
   │       paper to compare WAVES with
   │
1993 ── WIND construction & calibration; this paper submitted
   │
1994 ── 1 Nov: WIND launched
   │
1995 ── ★ BOUGERET et al.: WIND/WAVES (THIS PAPER)
   │       paper #64
   │
1995 ── SOHO launched (LASCO + EIT context for WAVES)
   │
1996 ── Bale et al.: Langmuir wave statistics from WIND TDS
   │       Reiner et al.: type II radio emission from CME shocks
   │
2000 ── Cluster/STAFF launched — wavelet analysis becomes standard
   │
2006 ── STEREO/WAVES launched — stereoscopic radio imaging
   │
2018 ── Parker Solar Probe / FIELDS — WAVES heritage
   │
2020 ── Solar Orbiter / RPW — direct WAVES descendant
   │
2024 ── ★ WIND continues operating at L1 — 30+ year mission
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **Stone et al. 1992** ("Unified Radio and Plasma Wave Investigation", URAP/Ulysses) | Direct technological predecessor: RAD1 design, DC/DC converter, FES experiment (TDS heritage) | High — WAVES is "URAP next-gen" with FFT, neural nets, longer Ex |
| **Meyer-Vernet & Perche 1989** ("Tool kit for antennae and thermal noise") | Thermal-noise spectroscopy theory: how to invert (n_e, T_e) from the open-circuit voltage spectrum | High — entire TNR scientific case rests on this method |
| **Gurnett 1974** ("Earth as a radio source") | Discovered AKR; one of the four phenomena WAVES targets in §1.3 | High — defines a key science case for RAD1 |
| **Manning & Fainberg 1980** + **Fainberg et al. 1985** | Goniopolarimetry method: direction-of-arrival from spinning-spacecraft modulation; ISEE-3 application | High — SUM-mode design (Fig. 5) is a direct implementation |
| **Lacombe et al. 1985, 1988** | Foreshock electron plasma waves and 2f_p emission upstream of bow shock | Medium-high — TNR + TDS science rationale for Langmuir wave studies |
| **Bougeret, Fainberg & Stone 1983, 1984** | Determining solar-wind speed from remote radio observations of source regions | Medium — WAVES will reproduce/extend these techniques with better cadence |
| **Burgess et al. 1987** | Radio waves at the plasma frequency from the terrestrial foreshock | Medium — defines a TNR/RAD1 case study target |
| **Hoang et al. 1994** ("IP type III bursts approaching f_p", Ulysses) | Direct precedent for WAVES type III statistics | Medium — WAVES extends this to ecliptic plane |
| **Reiner et al. 1993** ("Source characteristics of Jovian narrow-band kilometric radio emission") | Goniopolarimetry applied to Jupiter; same technique WAVES will apply to type II/III | Medium — methodological link |
| **Steinberg et al. 1990** ("Isotropic Terrestrial Kilometric Radiation") | One more terrestrial radio component WAVES can study | Low-medium — adds breadth |
| **(Future) Bale et al. 1996+** | Langmuir wave statistics from WIND TDS — first scientific harvest of TDS data | High — confirms TDS design goals were achieved |
| **(Future) Maksimovic et al. 2005+, Salem et al.** | Solar wind n_e, T_e climatology from WIND TNR thermal-noise inversion | High — proves the TNR method on >10-year baseline |

---

## 7. References / 참고문헌

### Primary paper / 본 논문

- Bougeret, J.-L., Kaiser, M. L., Kellogg, P. J., Manning, R., Goetz, K., Monson, S. J., Monge, N., Friel, L., Meetre, C. A., Perche, C., Sitruk, L., and Hoang, S., "WAVES: The Radio and Plasma Wave Investigation on the WIND Spacecraft", *Space Science Reviews* **71**, 231–263, 1995. DOI: 10.1007/BF00751331

### Cited in the paper / 논문에서 인용

- Bougeret, J.-L., Fainberg, J., and Stone, R. G., "Determining the Solar Wind Speed Above Active Regions Using Remote Radio Wave Observations", *Science* **222**, 506, 1983.
- Bougeret, J.-L., Fainberg, J., and Stone, R. G., "Interplanetary Radio Storms: 1. Extension of Solar Active Regions Through the Interplanetary Medium", *Astron. Astrophys.* **136**, 255, 1984.
- Burgess, D., Harvey, C. C., Steinberg, J.-L., and Lacombe, C., "Simultaneous Observation of Fundamental and Second Harmonic Radio Emission from the Terrestrial Foreshock", *Nature* **330**, 732, 1987.
- Canu, P., "Oblique Broadband Electron Plasma Waves Above the Plasma Frequency in the Electron Foreshock: 1990", *J. Geophys. Res.* **95**, 11,983, 1990.
- Fainberg, J., Hoang, S., and Manning, R., "Measurements of Distributed Polarized Radio Sources from Spinning Spacecraft; Effect of a Tilted Axial Antenna. ISEE-3 Application and Results", *Astron. Astrophys.* **153**, 145, 1985.
- Gurnett, D. A., "The Earth as a Radio Source: Terrestrial Kilometric Radiation", *J. Geophys. Res.* **79**, 4227, 1974.
- Gurnett, D. A., "The Earth as a Radio Source: the Non-Thermal Continuum", *J. Geophys. Res.* **80**, 2751, 1975.
- Hoang, S., Dulk, G. A., and Leblanc, Y., "Interplanetary Type III Radio Bursts that Approach the Plasma Frequency: Ulysses Observations", *Astron. Astrophys.* in press, 1994.
- Lacombe, C., Mangeney, A., Harvey, C. C., and Scudder, J. D., "Electron Plasma Waves Upstream of the Earth's Bow Shock", *J. Geophys. Res.* **90**, 73, 1985.
- Lacombe, C., Harvey, C. C., Hoang, S., Mangeney, A., Steinberg, J.-L., and Burgess, D., "ISEE Observations of Radiation at Twice the Solar Wind Plasma Frequency", *Ann. Geophysicae* **6**, 113, 1988.
- Manning, R., and Fairberg, J., "A New Method of Measuring Radio Source Parameters of a Partially Polarized Distributed Source from Spacecraft Observations", *Space Sci. Inst.* **5**, 161, 1980.
- Maroulis, D., Dumas, G., Bougeret, J.-L., Caroubalos, C., and Poquerusse, M., "The Digital System ARTEMIS for Real-Time Processing of Radio Transient Emissions in the Solar Corona", *Solar Phys.* **147**, 359, 1993.
- Meyer-Vernet, N., and Perche, C., "Tool Kit for Antennae and Thermal Noise Near the Plasma Frequency", *J. Geophys. Res.* **94**, 2405, 1989.
- Reiner, M. J., Fainberg, J., Stone, R. G., Kaiser, M. L., Desch, M. D., Manning, R., Zarka, P., and Pedersen, B. M., "Source Characteristics of Jovian Narrow-Band Kilometric Radio Emission", *J. Geophys. Res.* **98**, 13,163, 1993.
- Steinberg, J.-L., Hoang, S., and Bosqued, J.-M., "Isotropic Terrestrial Kilometric Radiation: a New Component of the Earth's Radio Emission", *Ann. Geophysicae* **8**, 671, 1990.
- Stone, R. G., and 31 co-authors, "The Unified Radio and Plasma Wave Investigation", *Astron. Astrophys. Suppl. Ser.* **92**, 291, 1992.

### Useful follow-up references / 후속 참고자료

- Bale, S. D., Burgess, D., Kellogg, P. J., Goetz, K., Howard, R. L., and Monson, S. J., "Phase Coupling in Langmuir Wave Packets: Possible Evidence of Three-Wave Interactions in the Upstream Solar Wind", *Geophys. Res. Lett.* **23**, 109, 1996.
- Maksimovic, M., Pierrard, V., and Lemaire, J. F., "A kinetic model of the solar wind with Kappa distribution functions in the corona", *Astron. Astrophys.* **324**, 725, 1997.
- Salem, C., Bosqued, J.-M., Larson, D., Mangeney, A., Maksimovic, M., Perche, C., Lin, R. P., and Bougeret, J.-L., "Determination of accurate solar wind electron parameters using particle detectors and radio wave receivers", *J. Geophys. Res.* **106**, 21701, 2001.
- Reiner, M. J., Kaiser, M. L., Fainberg, J., and Stone, R. G., "A new method for studying remote type II radio emissions from coronal mass ejection-driven shocks", *J. Geophys. Res.* **103**, 29651, 1998.
