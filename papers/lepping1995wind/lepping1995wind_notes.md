---
title: "The WIND Magnetic Field Investigation"
authors: "Lepping, R. P., Acuña, M. H., Burlaga, L. F., Farrell, W. M., Slavin, J. A., Schatten, K. H., Mariani, F., Ness, N. F., Neubauer, F. M., Wang, Y. C., Byrnes, J. B., Kennon, R. S., Panetta, P. V., Scheifele, J., Worley, E. M."
year: 1995
journal: "Space Science Reviews"
doi: "10.1007/BF00751330"
topic: Space_Weather
tags: [WIND, MFI, fluxgate, magnetometer, ISTP, IMF, magnetic_cloud, instrumentation]
status: completed
date_started: 2026-04-25
date_completed: 2026-04-25
---

# 61. The WIND Magnetic Field Investigation / WIND 자기장 탐사 장비

---

## 1. Core Contribution / 핵심 기여

본 논문은 1994년 11월 발사 예정이었던 NASA WIND 위성에 탑재된 자기장 탐사 장비 (Magnetic Field Investigation, MFI)의 전체 설계 사양, 잡음 성능, 데이터 처리 흐름, 그리고 ISTP (International Solar-Terrestrial Physics) 프로그램과의 연계 과학 목표를 종합적으로 기술한 *instrument paper*이다. 핵심 하드웨어는 (i) 12 m astromast 붐의 끝 (outboard, OB)과 약 2/3 지점 (inboard, IB)에 장착된 *듀얼 삼축 플럭스게이트 자력계*, (ii) 12-bit ADC와 8단계 자동 다이나믹 레인지 (±4 nT ~ ±65,536 nT), (iii) 80C86 기반 이중화 DPU (Digital Processing Unit), (iv) 256 kbit 트리거 스냅샷 메모리, (v) TI320C10 DSP 기반 256-점 FFT 처리기로 구성된다. 측정 성능은 ±0.001 nT 디지털 분해능, <0.006 nT r.m.s. (0–10 Hz) 잡음, ±0.1 nT 우주선 자기 잔여 기여, ±0.08 nT 절대 정확도, 0.025 % 정밀도, 0.0227 s ~ 1 hr의 다중 시간 평균에 해당한다.

This instrument paper comprehensively documents the design specification, noise performance, data-processing flow, and ISTP-program scientific objectives of the Magnetic Field Investigation (MFI) aboard NASA's WIND spacecraft (launched 1 November 1994). The instrument hardware comprises (i) a *dual triaxial fluxgate magnetometer* with one sensor at the end of a 12-m astromast boom (outboard, OB) and a second at ∼2/3 of the boom (inboard, IB), (ii) a 12-bit A/D converter with eight automatic dynamic ranges (±4 nT to ±65,536 nT), (iii) an 80C86-based fully redundant Digital Processing Unit (DPU), (iv) a 256 kbit triggered snapshot memory, and (v) a TI320C10 DSP-based 256-point FFT processor. The measurement performance is ±0.001 nT digital resolution, <0.006 nT r.m.s. (0–10 Hz) noise, ±0.1 nT residual spacecraft contamination, ±0.08 nT absolute accuracy, 0.025 % precision, with multiple averaging time-scales from 0.0227 s detail data to 1 hr.

과학적으로 MFI는 IMF의 (1) 대규모 (sector boundary, magnetic cloud, ∼50–100 hr), (2) 중규모 (interplanetary ejecta, plasmoid, ∼16–50 hr), (3) 미세 규모 (Alfvén wave, MHD discontinuity, 2.5 s–3.5 hr), (4) 운동 규모 (perpendicular fast shock ramp, <0.06 s) 현상을 동시에 관측하여 (a) 비충돌 플라즈마 실험실로서의 태양풍, (b) 태양 에너지·질량 출력의 발현, (c) 자기권을 구동하는 외부원으로서의 IMF 작용을 통합 연구하도록 설계되었다. 또한 GEOTAIL/GIM (Geotail Inboard Magnetometer)이 본 팀이 함께 제작한 자매 장비로서, 본 논문은 GIM 관련 과학 목표 (자기꼬리 plasmoid, 자기뇌우 trigger 등) 또한 함께 다룬다.

Scientifically, MFI is designed to simultaneously sample IMF phenomena across (1) large scale (sector boundaries, magnetic clouds, ∼50–100 hr), (2) meso scale (interplanetary ejecta, plasmoids, ∼16–50 hr), (3) micro scale (Alfvén waves, MHD discontinuities, 2.5 s–3.5 hr), and (4) kinetic scale (perpendicular fast shock ramps, <0.06 s), supporting the unified study of (a) the solar wind as a collisionless plasma laboratory, (b) the manifestation of solar energy/mass output, and (c) the IMF as the external driver of the terrestrial magnetosphere. Because GEOTAIL's GIM (Inboard Magnetometer) is the same team's sister instrument, the paper also discusses GIM-related goals (magnetotail plasmoids, geomagnetic-storm triggers).

---

## 2. Reading Notes / 읽기 노트

### Part I — Abstract & Introduction (pp. 207–209) / 초록 및 서론

**원문 핵심**: WIND/MFI는 IMF 구조와 변동의 정밀·정확·초고감도 측정을 목표로 한다. 측정 정밀도 0.025 %, 정확도 <0.08 nT, 0.008 nT/step의 양자화. *Key Parameter*는 92 s마다 1 vector, *standard*는 10.9 vec/s, *snapshot/FFT*는 44 vec/s. 6개월 동안 zero-drift는 <0.1 nT. 표준 평균: 0.0227 s detail (SS), 0.092 s standard, 3 s, 1 min, 1 hr, GSE/GSM 좌표.

**Abstract & §1 highlights**: WIND/MFI's design targets precise (0.025 %), accurate (<0.08 nT), ultra-sensitive (0.008 nT/step) IMF measurements. Three main data streams: KP at 1 vec / 92 s, standard at 10.9 vec/s, and snapshot/FFT at 44 vec/s. Intrinsic zero drift below 0.1 nT over 6 months. Standard averages are 0.0227 s detail, 0.092 s standard, 3 s, 1 min, and 1 hr in GSE/GSM coordinates.

**핵심 질문**: 왜 *듀얼* 자력계인가?
- (i) 완전 이중화 (full redundancy) — 단일 점 고장 회피.
- (ii) 우주선 *쌍극자 (dipolar)* 자기장 기여를 거리^3 의존성으로 분리. OB는 12 m, IB는 ∼8 m → IB가 OB의 약 (12/8)³ ≃ 3.4배 강한 우주선 잔류 자기장을 본다. 두 측정값의 선형 결합으로 우주선 기여를 *해석적으로* 제거 가능.
- (iii) 12 m 위치에서 IF 우주선 잔류장이 ±0.1 nT 이하가 되도록 boom 길이가 결정됨.

**Why dual?** (i) Full hardware redundancy. (ii) Algebraic separation of the spacecraft's *dipolar* field, which falls as 1/r³ — IB at ∼8 m sees roughly (12/8)³ ≃ 3.4× the OB residual, allowing analytic removal. (iii) The 12 m boom length is chosen such that the spacecraft residual at OB is ≤±0.1 nT.

### Part II — Scientific Objectives (§2, pp. 209–214) / 과학 목표

**스케일 분류표 / Scale-classification table** (paper's own taxonomy):

| Scale | Length | Time | Examples |
|---|---|---|---|
| Large | ½ ↔ 1 AU (longitudinal) | 50 ↔ 100 hr | sector boundaries, magnetic clouds |
| Meso | 1/6 ↔ ½ AU | 16 ↔ 50 hr | interplanetary ejecta, plasmoids |
| Micro | 10³ ↔ 5×10⁶ km | 2.5 s ↔ 3.5 hr | Alfvén, D-sheet, magnetic holes, tangential/rotational/contact discontinuities |
| Kinetic | <25 km | <0.06 s | shock ramps, ⊥ fast shock ramps |

**SSC와 행성간 충격파 (pp. 210–211)**: Smith et al. (1986)의 ISEE-3 통계 (1978–1980, ∼50 shocks)로부터 *interplanetary shock ↔ SSC*의 인과 관계는 80–90 % 확률, 역방향 ∼80 %. SSC는 1958–1959에 Dessler (1958), Dessler & Parker (1959)에서 처음 논의. Wilson & Sugiura (1961)의 MHD 묘사 — 저위도는 *longitudinal (압축)* mode, 고위도는 *transverse* mode 우세.

**§2 SSC and IP shocks**: From Smith et al. (1986) ISEE-3 statistics (∼50 shocks 1978–1980), an interplanetary shock causes an SSC with 80–90 % probability, and conversely an SSC implies a shock with ∼80 %. Wilson & Sugiura (1961): the MHD signal is longitudinal (compressional) at low latitudes, transverse at high latitudes.

**Magnetic clouds (pp. 211–212)**: Burlaga et al. (1981), Goldstein (1983), Burlaga (1991)이 정의 — 부드럽게 회전하는 강한 자기장, 절반 정도의 기간 동안 (∼12 hr) Bz가 일관되게 음 (southward)을 유지 → 자기뇌우 주요 main phase 구동자. Farrugia et al. (1993a–d)는 (a) 정상 남향 Bz cloud의 *비주기적 substorm trigger*, (b) cusp 일측 dayside reconnection 구동, (c) cloud 통과 동안 자기권의 SEP 가이드, (d) magnetosheath의 *높은 Te/Tp 비율* 등을 보고. **Gonzalez & Tsurutani (1987) 의 결정적 발견**: 자기뇌우 진폭은 IP shock 강도가 아니라 *cloud (CME)의 Bz와 지속 시간*에 의해 결정된다.

**Magnetic clouds**: Defined by Burlaga et al. (1981), Goldstein (1983), Burlaga (1991) — smooth-rotation, strong-B, low-β structures of ∼12 hr duration with Bz consistently southward for roughly half the cloud, driving the main phase of geomagnetic storms. Farrugia et al. (1993a–d) report (a) quasi-periodic substorm trigger before Bz turns southward, (b) cusp-region dayside reconnection driven by southward Bz, (c) SEP guidance through clouds, (d) anomalously high Te/Tp ratios in the cloud sheath. **Gonzalez & Tsurutani (1987)** is decisive: storm amplitude is set not by shock strength but by cloud (CME) Bz and duration.

**Lepping et al. (1992) Earth-side timing (p. 212)**: The MHD signal triggered by an IP shock at the magnetopause reaches the Earth's surface in 81 ± 18 s with average speed 580 km/s, with latitude dependence (10 well-determined IP shocks via the Vinas–Scudder 1986 shock-fitting technique using single-S/C IMP-8 plasma + B data). Wilken et al. (1982)는 동일 1977 Jul 29 SSC를 6개 위성으로 분석하여 600 km/s (frontside, equatorial)와 910 km/s (high-latitude outer/geosynchronous magnetosphere, tangential)를 보고.

**Other team interests (p. 213)**: (1) slow-mode shocks in solar wind / sheath / plasma sheet boundary (Whang 1988 1991, Lee et al. 1991, Feldman et al. 1987); (2) interstellar pickup hydrogen via foreshock-free intervals — Gloeckler et al. (1993), Smith (1993); (3) lunar-pickup ions downstream of Moon (Hilchenbach et al. 1992); (4) Ulysses–WIND combined 3D solar-wind studies; (5) CLUSTER+WIND two-station array; (6) bow-shock/MP/boundary-layer studies during WIND's brief magnetosheath passages.

### Part III — GIM-related GEOTAIL studies (§3, pp. 214–215) / GIM 관련 GEOTAIL 연구

**GEOTAIL 두 단계 궤도**: (i) 2.3 yr deep tail double-lunar swing-by (apogee ∼220 R_E), (ii) 60 R_E elliptical near-tail orbit. Plasmoid: (X,Z)_GSM 평면 타원형 단면, ∼60 R_E 길이 (Richardson et al. 1987), 일부 8 R_E 소형도 가능 (Slavin et al. 1990, Moldwin & Hughes 1991, 1992). 본 팀의 GIM은 14-bit 분해능, 4 vec/s. 자기꼬리 boundary, MP-surface waves, flux transfer events, ELF 파동도 연구 대상.

**GEOTAIL two-phase orbit**: (i) ∼2.3 yr deep-tail double-lunar swing-by (apogee ∼220 R_E), (ii) 60 R_E elliptical near-tail orbit. Plasmoids: oval cross-section in (X,Z)_GSM, ∼60 R_E length (Richardson et al. 1987); small 8 R_E variants possible (Slavin et al. 1990, Moldwin & Hughes 1991, 1992). GIM differs from MFI in 14-bit resolution and 4 vec/s sampling. Studies include tail boundary, MP surface waves, flux-transfer events, and ELF waves.

### Part IV — MFI Instrumentation (§4, pp. 215–221) / MFI 장비

**Heritage**: Voyager (1977), ISPM/Ulysses (1990), Giotto (1985), Mars Observer (1992) — 모두 GSFC Acuña 그룹의 ring-core 플럭스게이트 시리즈. MFI는 그 기술의 정점.

**Heritage**: Voyager, ISPM/Ulysses, Giotto, Mars Observer — all GSFC Acuña-group ring-core fluxgate magnetometers. MFI is the apex of this lineage.

**Table I (instrument summary)**:

| Parameter | Value |
|---|---|
| Type | Dual triaxial fluxgate (boom-mounted) |
| Dynamic ranges (8) | ±4, ±16, ±64, ±256, ±1024, ±4096, ±16,384, ±65,536 nT |
| 12-bit digital resolution | ±0.001, ±0.004, ±0.016, ±0.0625, ±0.25, ±1.0, ±4.0, ±16.0 nT |
| Sensor noise | <0.006 nT r.m.s. 0–10 Hz |
| Sampling rate | 44 vec/s (snapshot), 10.87 vec/s (standard) |
| Signal processing | FFT 32 log-channels, 0–22 Hz, 23 s (high-rate) / 46 s (low-rate) cadence on B_x B_y B_z |B| |
| FFT windows | full de-spin, 10 % cosine taper, Hanning, first-difference filter |
| FFT dynamic range | 72 dB μ-law compressed 13-bit → 7-bit + sign |
| Sensitivity threshold | ∼0.5 × 10⁻³ nT/√Hz in range 0 |
| Snapshot memory | 256 kbit |
| Trigger modes | magnitude jump, peak-to-peak directional, spectral-band r.m.s. |
| Telemetry modes | 3 (ground-commandable) |
| Mass | sensors 450 g + electronics 2100 g (redundant) |
| Power | 2.4 W |

**Fluxgate operation (Fig. 2, p. 218) / 플럭스게이트 동작**:
- Ring-core ferromagnetic 센서가 60 KHz / 4 = 15 KHz drive로 *cyclic saturation* 구동 (15 KHz는 DPU master clock에서 분주).
- 외부장이 0이면 sensor coil 출력에 짝수 고조파 (even harmonics)는 *균형* 상태로 0.
- 외부장이 인가되면 짝수 고조파(특히 2f)에 신호가 발생 → synchronous detector + integrator → feedback 전류로 sensor 자기장을 *null*. Feedback 전류 ∝ 외부장.
- Driving 강도가 ring core 보자력 (coercive) 의 100배 이상 → *perming* (자성 잔류) 문제 제거.
- 단축 자력계 3개를 직교 배치 → 삼축. OB 어셈블리와 IB 어셈블리 각각 동일 구조.

**Fluxgate operation (Fig. 2)**:
- A ring-core ferromagnetic sensor is driven into cyclic saturation by a 15 KHz signal (60 KHz ÷ 4) sourced from the DPU master clock.
- With zero external field, the sensor is *balanced* and even harmonics vanish.
- With an external field, the second harmonic (2f) carries the signal; a synchronous detector + integrator produces a *feedback current* that nulls the field at the sensor; this current ∝ ambient field.
- Drive amplitude exceeds 100× ring-core coercivity, eliminating *perming* problems.
- Three orthogonal single-axis units form a triaxial assembly; OB and IB assemblies are identical.

**Noise (Fig. 3a/3b, pp. 219–220)**:
- 프로토타입 시험: 0.05 nT p-p sinewave가 BW=8.3 Hz와 BW=0.1 Hz 모두에서 분명히 식별됨.
- 0.1 nT 계단 응답이 BW=8.3 Hz와 BW=1.0 Hz 모두에서 깨끗하게 추적됨.
- Power Spectral Density: f > 10 Hz에서 *flat* PSD ≈ 2 × 10⁻⁶ nT²/Hz, total r.m.s. = 12.1 × 10⁻³ nT (0–50 Hz). 0–10 Hz 대역에서는 <0.006 nT r.m.s.

**Noise (Figs. 3a, 3b)**:
- Prototype test: a 0.05 nT p-p sinewave is clearly resolved at both 8.3 Hz and 0.1 Hz bandwidths.
- A 0.1 nT step is cleanly tracked at both 8.3 Hz and 1.0 Hz BW.
- PSD flattens at ∼2 × 10⁻⁶ nT²/Hz for f > 10 Hz; total r.m.s. = 12.1 × 10⁻³ nT (0–50 Hz); 0–10 Hz r.m.s. < 0.006 nT.

**Range switching (Fig. 4, p. 221)**:
- 출력이 full-scale의 *7/8* 초과 → step *up* (덜 민감한 다음 레인지로).
- 모든 축 출력이 full-scale의 *1/8* 미만 → step *down* (더 민감한 레인지로).
- *Guard band* 1/8 of scale 폭으로 hysteresis 제공 → 포화 손실 방지.
- 결정은 44 vec/s 내부 sample rate에서 수행 (이때 OB 데이터가 다른 WIND 장비로도 분배됨).

**Range switching (Fig. 4)**:
- Output exceeds 7/8 of full-scale → step up (less sensitive range).
- All axes drop below 1/8 → step down (more sensitive range).
- A 1/8-wide guard band provides hysteresis to prevent saturation loss.
- Decisions made at the 44 vec/s internal rate (and OB data are simultaneously delivered to other WIND instruments at this rate).

### Part V — Digital Processing Unit (§5, pp. 222–224) / 디지털 처리부

**80C86 microprocessor**: ISTP project office에서 제공한 *radiation-hardened* version. Smart system, 모든 동작은 ROM에 저장된 default executive로 무명령 동작. 모든 default parameter는 ROM에서 RAM으로 부팅 시 매핑되어 ground command로 수정 가능 (calibrations, alignments, sample rates, zero levels).

**80C86**: A radiation-hardened version provided by the ISTP Project Office. Implements a smart-system architecture; all default behaviour is stored in ROM and the instrument starts up valid without any commanding. All defaults are mapped from ROM to RAM at boot for ground modification.

**Watchdogs**: 외부 hardware + 내부 software watchdog timer로 freeze 시 자동 ROM reload + restart.

**Watchdog**: External hardware + internal software watchdog timers automatically reset the DPU and reload defaults if the executive freezes.

**Snapshot memory (256 kbit)**:
- 44 vec/s OB 데이터를 cyclical하게 overwrite. 메모리 가득 차면 trigger까지 계속 덮어쓰기.
- *7282 vectors* ≈ 165 s 데이터 = 약 2 min 45 s.
- DPU의 memory pointer로 trigger 발생 *82 s 이전* 데이터까지 보존 (약 절반의 buffer).
- Trigger 조건 3가지: (1) magnitude jump (shock ramp 검출), (2) peak-to-peak 방향 변화 (TD/RD 검출), (3) spectral-band r.m.s. 변화 (kinetic-scale 파동 검출).

**Snapshot memory (256 kbit)**:
- Cyclical overwrite at 44 vec/s OB data. On trigger the memory is *frozen*.
- 7282 vectors ≈ 165 s ≈ 2 min 45 s.
- DPU memory pointers preserve up to 82 s of *pre-trigger* data — half the buffer — enabling precursor studies.
- Three trigger classes: (1) magnitude jump (for shock ramps), (2) peak-to-peak directional change (for TD/RD discontinuities), (3) spectral r.m.s. change in a band (for kinetic-scale waves).

**FFT processor (TI320C10 DSP)**:
- 256-spectral-band raw output from 512-sample (11.6 s) ambient B time series, plus magnitude FFT.
- De-spinning 적용 (spin plane 성분만, X-축은 spin axis라 영향 없음).
- 전처리: pre-whitening, 10 % cosine taper window, Hanning, first-difference filter.
- 256 → *32 log-spaced channels* (constant fractional BW, "Q-filter" 등가) 압축.
- Amplitude 압축: 12-bit → 7-bit + sign via (a) MSB truncation 또는 (b) μ-law (통신 시스템 표준).
- 결과: 4 axes × 32 channels = 128 8-bit estimate, 23 s (high-rate) 또는 46 s (low-rate) cadence로 송신.

**FFT processor (TI320C10 DSP)**:
- 256-band raw spectral estimate from 512 samples (11.6 s) of ambient B, plus a |B| magnitude FFT.
- De-spinning applied to spin-plane components prior to FFT (X is along the spin axis and unaffected).
- Preprocessing: pre-whitening, 10 % cosine taper, Hanning window, first-difference filter.
- 256 raw bands compressed to 32 log-spaced channels (constant-Q "fractional-BW" filter equivalent).
- Amplitude domain compressed from 12-bit to 7-bit + sign via either (a) variable MSB truncation or (b) μ-law (telecom standard).
- Final telemetry: 4 axes × 32 channels = 128 8-bit estimates per FFT block at 23 s (high-rate) or 46 s (low-rate) cadence.

**CAP (Command and Attitude Processor)** distributes high-time-resolution MFI data to other WIND instruments in two formats: (a) raw 44 vec/s serial stream, or (b) two pulses per 3-s spin (spin-plane azimuth + spin-axis elevation), enabling other instruments to phase-lock to the field.

### Part VI — Telemetry modes (Table II, p. 224) / 텔레메트리 모드

| Stream | Mode 0 'normal' | Mode 1 'FFT' | Mode 2 'SSM' |
|---|---|---|---|
| Outboard | 195.8 BPS (5.43 vec/s) | 391.3 BPS (10.87 vec/s) | 391.3 BPS (10.87 vec/s) |
| Inboard | 195.8 BPS (5.43 vec/s) | 19.6 BPS (0.54 vec/s) | 19.6 BPS (0.54 vec/s) |
| FFT | 55.6 | 55.6 | 0 |
| SSM | 31.0 | 13.9 | 69.5 |
| H.K. status | 2.2 | 0.0 | 0.0 |
| **Total** | **480.4** | **480.4** | **480.4** |
| Inst. status | 4.3 | 4.3 | 4.3 |
| Combined | 484.7 | 484.7 | 484.7 |

Mode 0 = OB·IB 동등 sampling (default, S/C field 특성화). Mode 1 = OB 우선, FFT 강조. Mode 2 = OB 우선, snapshot 강조. WIND가 *달의 궤도 안쪽*에 있을 때는 telemetry 할당이 두 배 → 모든 rate가 두 배. Mode 0 = equal OB/IB sampling (default, used to characterize S/C field). Mode 1 prioritises OB and FFT; Mode 2 prioritises OB and snapshot. Inside the lunar orbit, telemetry doubles, doubling all rates.

### Part VII — Power, Thermal, Ground Processing (§6–7, pp. 224–227) / 전원, 열, 지상 처리

**Power**: 28 V S/C bus → 2개 redundant 50 KHz converter → DPU master clock에 동기되어 EMI 최소화. 특정 시점에 한 subsystem만 power-on.

**Power**: 28 V bus → 2 redundant 50 KHz converters synchronized to the DPU master clock to minimize EMI; only one subsystem is energized at a time.

**Thermal**: Boom 센서 어셈블리는 일식 기간 (Sun occultation) 동안 heater 필요. DC heater의 stray field를 피하기 위해 50 KHz AC를 magnetic-amplifier로 비례 제어 → MFI 측정에 자기 잡음 영향 최소화. 정상 가열 전력 0.3–0.5 W.

**Thermal**: During solar occultations, boom sensors need heating. DC heaters would generate stray fields, so a 50 KHz AC supply is proportionally controlled by a magnetic amplifier, eliminating heater-induced field contamination. Nominal heater power 0.3–0.5 W.

**Ground processing flow (Fig. 5 KP / Fig. 6 RDAF, pp. 225–226)**:

```
MAG Level-1 → DEBLOCK → CALIBRATION (zeros, offsets) → COUNTS→nT
            → SUBTRACT S/C FIELD → AMBIENT in S/C SPIN COORDS
            → DESPIN (with ORBIT/ATTITUDE FILE) → GSE→GSM TRANSFORM
            → MAJOR FRAME AVERAGES → KP OUTPUT FILE
```

Production은 SUN 4/380 워크스테이션 (RDAF), CD-ROM 출력. KP는 92 s 프레임에 1 vector. 표준 평균: 0.0227 s, 0.092 s, 3 s, 1 min, 1 hr, GSE+GSM 양쪽. NSSDC 표준 포맷 파일로 저장. Production runs on a SUN 4/380 (RDAF) producing CD-ROM deliveries. KP at 1 vec / 92 s. Standard averages: 0.0227 s, 0.092 s, 3 s, 1 min, 1 hr in both GSE and GSM. NSSDC Standard-Format Files produced.

**드리프트 모니터링**: 180° electronic flipper로 spin-plane sensor의 *zero level*을 정기 점검. Z-축 (spin axis) drift는 큰 통계 분석을 통해 GSFC + Tor Vergata (Mariani 그룹) 공동 평가.

**Drift monitoring**: A 180° electronic flipper periodically reverses spin-plane sensor zero readings; Z-axis (spin) drift requires a large statistical study, jointly performed by GSFC and the Tor Vergata team (Mariani group).

---

## 3. Key Takeaways / 핵심 시사점

1. **Dual-sensor 1/r³ trick gives ±0.1 nT cleanliness / 듀얼 센서 1/r³ 기법으로 ±0.1 nT 청결도 확보** — 우주선 자기장은 주로 dipolar이고 거리^3로 떨어지므로, 두 센서 (12 m, ∼8 m)의 측정값을 선형 결합하여 우주선 기여를 *해석적으로* 제거할 수 있다. 이는 단순 boom 길이 증가만으로는 불가능한 정밀도이다. Because the spacecraft's residual field is dominantly dipolar (1/r³), the linear combination of OB (12 m) and IB (∼8 m) measurements algebraically removes the contamination — a precision that no single-sensor boom length alone can achieve.

2. **8-range × 12-bit ADC achieves 156 dB end-to-end / 8 레인지 × 12-bit ADC = 실효 156 dB** — 단일 12-bit ADC의 72 dB native dynamic range를 8개의 레인지 (factor 4 between consecutive ranges) 자동 전환으로 ±0.001 nT부터 ±65,536 nT까지 8자리 (8 orders of magnitude) 측정 가능하게 확장. *Guard band*가 hysteresis로 작동하여 포화·과민 사이의 chatter를 방지. By stacking eight ranges (factor 4 each) on a single 12-bit ADC, the instrument extends a 72 dB native ADC into an 8-orders-of-magnitude (∼156 dB) end-to-end range with guard-band hysteresis preventing chatter.

3. **Snapshot memory with 82 s pre-trigger enables shock precursor studies / 82 s 사전-트리거 스냅샷 메모리로 충격파 전조 연구 가능** — 165 s 버퍼의 절반은 *trigger 발생 이전*을 향하므로, shock ramp 또는 directional discontinuity가 검출된 후에도 그 *upstream* 미세 구조 (foreshock, precursor wave)가 44 vec/s 고분해능으로 남는다. 이는 단순 후행 기록 (post-trigger only)으로는 영원히 잃어버리는 정보다. The 165 s circular buffer dedicates ∼82 s to *pre-trigger* data, preserving high-rate (44 vec/s) precursor structure (foreshock, upstream waves) that simple post-trigger logging would lose.

4. **TI320C10 DSP-based on-board FFT collapses telemetry burden / TI320C10 DSP 기반 on-board FFT가 telemetry 부담 해소** — 256-bin × 4-axis × 12-bit FFT를 32 log-bin × 8-bit (μ-law)로 압축하면 데이터 양이 약 23배 감소. 그러면서도 0–22 Hz의 4축 power+phase 정보가 23 s 마다 전송 가능 → IMF wave 환경의 *연속 모니터링* 실현. The 256-bin × 4-axis × 12-bit FFT compressed to 32 log-bin × 8-bit (μ-law) gives a ∼23× reduction, enabling continuous 0–22 Hz monitoring of IMF waves at 23 s cadence — impossible with raw waveform telemetry.

5. **Triple-redundancy philosophy sets the GSFC standard / 삼중 이중화 철학이 GSFC 표준** — (i) sensor-level (dual OB/IB), (ii) DPU-level (A-side / B-side, primary/secondary), (iii) self-resetting electronic 'fuses'가 공통 subsystem을 격리. 이 설계 철학은 STEREO/MAG, Parker Solar Probe FIELDS, Solar Orbiter MAG 모든 후속 임무에 직계 계승. The triple redundancy — (i) dual OB/IB sensors, (ii) A/B DPU sides, (iii) self-resetting electronic fuses — sets the design template directly inherited by STEREO/MAG, PSP/FIELDS, and Solar Orbiter MAG.

6. **Magnetic clouds, not shocks, drive major storms / 주요 자기뇌우는 충격파가 아닌 자기 구름이 구동** — Gonzalez & Tsurutani (1987), Tsurutani et al. (1988)은 storm 진폭이 IP shock 강도와는 *무관*하고, southward Bz의 지속 시간 + 크기에 의해 결정됨을 보였다. MFI의 핵심 운영 우선순위 (Bz를 92 s마다 KP로 dissemination)는 이 발견에서 직접 도출된다. The 92-s KP cadence priority for IMF Bz dissemination follows directly from Gonzalez & Tsurutani (1987)'s finding that storm amplitude depends not on shock strength but on the duration × magnitude of southward Bz in the cloud (CME).

7. **Cross-instrument timing distribution via CAP / CAP을 통한 장비 간 타이밍 분배** — MFI는 자기장 측정만이 아니라 *spin-phase reference*를 다른 WIND 장비에 공급한다 (CAP의 두 pulse per 3 s spin 방식). 이는 다중-장비 phase-coherent 측정을 가능케 하여, 자기장-플라즈마 *cross-spectral* 분석 (Alfvén ratio, polarization)이 위성 차원에서 실현된다. MFI not only measures the field but supplies a *spin-phase reference* to all other WIND instruments via CAP (two pulses per 3 s spin), enabling instrument-level phase-coherent cross-spectral analyses (Alfvén ratio, polarization).

8. **GIM-MFI commonality enables coordinated tail/IMF studies / GIM-MFI 공통성으로 자기꼬리·IMF 동시 연구 가능** — 같은 팀이 만든 GEOTAIL/GIM (4 vec/s, 14-bit)과 WIND/MFI (44 vec/s, 12-bit)는 calibration 철학이 같아 *cross-comparison*에 직접 활용 가능. WIND가 IMF upstream을 측정하는 동안 GEOTAIL이 자기꼬리에서 결과를 측정하는 *cause-and-effect* 짝 관측이 정량적 기반 위에 가능해진다. Because GIM (4 vec/s, 14-bit) and MFI (44 vec/s, 12-bit) share calibration philosophy, the WIND-upstream / GEOTAIL-tail pair becomes quantitatively comparable, enabling cause-and-effect studies on a calibrated common scale.

---

## 4. Mathematical Summary / 수학적 요약

### (M1) Fluxgate sensor governing equations / 플럭스게이트 지배 방정식

Drive: $H_{\text{drive}}(t) = H_0\cos(2\pi f_d t)$, $f_d = 15$ kHz, $H_0 \gg H_c$ (coercive).

Pickup-coil EMF (with external field $B_{\text{ext}}$):
$$V_{\text{pickup}}(t) \;=\; -N\,\frac{d\Phi}{dt}\;=\;\sum_{n=1}^{\infty} a_n\cos(2n\pi f_d t + \phi_n)$$

By symmetry, even-harmonic amplitudes scale linearly with $B_{\text{ext}}$:
$$a_{2n} \;\propto\; B_{\text{ext}} \quad(\text{small-field linear regime})$$

Synchronous detection at $2f_d$ yields:
$$E_o \;=\; G\,a_2 \;\propto\; G\,B_{\text{ext}}$$

The integrator drives feedback current $I_{fb}$ such that $B_{\text{feedback}} = -B_{\text{ext}}$ at the sensor:
$$I_{fb} \;=\; \frac{B_{\text{ext}}}{k_{\text{coil}}}\quad \Longrightarrow\quad V_{\text{out}}\;=\;R_{\text{shunt}}\,I_{fb}\;\propto\;B_{\text{ext}}$$

### (M2) Dual-magnetometer S/C-field separation / 듀얼 자력계 우주선장 분리

For a dipolar S/C field $\mathbf{B}_{S/C}(\mathbf{r}) = \mathbf{m}/(4\pi r^3) \cdot (\hat{m},\hat{r}\text{ geometric factors})$, treating as a scalar coefficient $\alpha$ at each sensor:

OB sensor at $r_O = 12$ m measures: $\mathbf{B}_{OB} = \mathbf{B}_{\text{ambient}} + \alpha/r_O^3$
IB sensor at $r_I \approx 8$ m measures: $\mathbf{B}_{IB} = \mathbf{B}_{\text{ambient}} + \alpha/r_I^3$

Solving the 2×2 linear system:
$$\boxed{\;\mathbf{B}_{\text{ambient}} \;=\; \frac{r_I^3\,\mathbf{B}_{IB} - r_O^3\,\mathbf{B}_{OB}}{r_I^3 - r_O^3}\;}$$

With $r_O=12$ m, $r_I=8$ m: $r_O^3=1728$ m³, $r_I^3=512$ m³, difference $-1216$. So
$\mathbf{B}_{\text{ambient}} = (512\,\mathbf{B}_{IB} - 1728\,\mathbf{B}_{OB})/(-1216) = 1.421\,\mathbf{B}_{OB} - 0.421\,\mathbf{B}_{IB}$.

Worked example: if true $B=5$ nT and S/C contributes $0.05$ nT at OB ($\alpha/r_O^3=0.05$), then at IB it is $0.05 \cdot (12/8)^3 = 0.169$ nT. OB reads 5.05 nT, IB reads 5.169 nT, and the formula gives $1.421(5.05) - 0.421(5.169) = 7.176 - 2.176 = 5.000$ nT — exactly the true ambient.

### (M3) Dynamic range arithmetic / 동적 범위

Native single-range ADC: 12-bit signed → 4096 counts → $20\log_{10}(2^{11}) = 66.2$ dB plus sign extension; effectively cited as 72 dB.

End-to-end across 8 ranges:
$$\text{DR}_{\text{total}} = 20\log_{10}\!\left(\frac{65{,}536}{0.001}\right) = 20\log_{10}(6.55\times 10^7) \approx 156\text{ dB}$$

Range step factor: $4096/16 = 256$ per 4 ranges, or $4$× per range (geometric).

Range-switch thresholds (Fig. 4):
- Up-switch: $|N_{\text{counts}}| > (7/8)\cdot 2048 = 1792$
- Down-switch: $|N_{\text{counts}}| < (1/8)\cdot 2048 = 256$
- Guard band: $1792 \leq |N| \leq 2047$ (saturation buffer)

### (M4) FFT data reduction / FFT 데이터 압축

Raw FFT block: $N=512$ samples, $\Delta t = 1/44.0\;\text{s} = 22.7$ ms → block duration $= 512/44 = 11.64$ s.

Frequency resolution: $\Delta f = 1/(N\Delta t) = 1/11.64 = 0.0859$ Hz; Nyquist $f_N = 1/(2\Delta t) = 22$ Hz; useful 256 bins.

Log-binning: 256 → 32 log-spaced bins, "constant-Q":
$$f_k \;\propto\; \exp(k/k_0),\qquad \frac{\Delta f_k}{f_k} = \text{constant}$$

μ-law amplitude compression (telecom standard):
$$y(x) \;=\; \text{sgn}(x)\,\frac{\ln(1+\mu|x|)}{\ln(1+\mu)},\qquad \mu \approx 255$$

Compression factor: 12-bit → 7-bit + sign = 8-bit ⇒ 1.5× per estimate; combined with 256→32 bin reduction the total FFT-block compression is $(256/32)\times(12/8) = 12$ for amplitude; including 4 axes the *raw-to-telemetry* ratio for the spectral block is roughly:

$$\frac{256\text{ bins} \times 4\text{ axes}\times 12\text{ bit}}{32\text{ bins}\times 4\text{ axes}\times 8\text{ bit}} \;=\; 12\times$$

### (M5) Coordinate frames GSE → GSM / 좌표 변환

GSE: X = Earth–Sun, Z = ecliptic-north, Y = Z×X.
GSM: X = Earth–Sun (shared with GSE), Z = projection of the Earth's magnetic-dipole axis onto the GSE Y-Z plane.

The transformation is a rotation about X by the dipole-tilt angle $\mu$:
$$\mathbf{B}_{\text{GSM}}\;=\;R_x(\mu)\,\mathbf{B}_{\text{GSE}},\qquad R_x(\mu) = \begin{pmatrix} 1 & 0 & 0 \\ 0 & \cos\mu & \sin\mu \\ 0 & -\sin\mu & \cos\mu \end{pmatrix}$$

with
$$\tan\mu \;=\; \frac{Y_{\text{GSE}}^{\text{dipole}}}{Z_{\text{GSE}}^{\text{dipole}}}$$

For typical conditions $\mu$ varies between $\pm 35°$ over a year and $\pm 11°$ over a day.

### (M6) Magnetic-cloud Bz model (Burlaga 1991) / 자기 구름 Bz 모델

Force-free flux-rope solution (Lundquist):
$$B_z(r) = B_0\,J_0(\alpha r),\qquad B_\phi(r) = B_0\,J_1(\alpha r),\qquad B_r = 0$$

with the cylindrically-symmetric current $\mathbf{J} = \alpha \mathbf{B}/\mu_0$. Along a radial transit through the cloud center, the on-axis $B_z$ traces approximately a half-cosine:
$$B_z(t) \;\approx\; B_0\,\cos\!\left(\pi\,\frac{t-t_c}{\Delta T}\right),\qquad \Delta T\sim 12\text{ hr}$$

Storm Dst response (empirical, Burton 1975-style):
$$\frac{dD_{st}^*}{dt} \;=\; Q(VB_s) - \frac{D_{st}^*}{\tau},\qquad B_s = \max(-B_z,0)$$

Storm intensity therefore scales with the *integral* of southward $B_z$ over the cloud passage — confirming Gonzalez & Tsurutani (1987).

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1958  Dessler              : MHD propagation of SSC discussed
1961  Wilson & Sugiura     : Longitudinal/transverse SSC modes
1965  Sugiura              : Two-component cold-plasma MHD review
1970  Ness                 : Space magnetometers — review
1971  Ness, Behannon,      : Dual-magnetometer concept (IMP/Explorer-43)
       Lepping, Schatten
1974  Acuña                : Ring-core fluxgate principle (IEEE Trans.)
1976  Mish & Lepping       : IMP magnetometer ground processing
1977  Voyager 1/2 launch   : Triaxial fluxgate to outer planets
1981  Burlaga et al.       : Magnetic clouds defined
1985  Giotto/MAG           : Comet Halley fluxgate
1986  Smith et al.         : ISEE-3 SSC ↔ IP shock 80–90% statistic
1986  Vinas & Scudder      : Single-S/C shock-fitting technique
1987  Gonzalez & Tsurutani : Storm amplitude tied to cloud Bz
1990  Ulysses (ISPM)       : Polar heliosphere fluxgate
1992  Mars Observer        : Latest dual-fluxgate heritage
1992  Lepping et al.       : SSC propagation 81 ± 18 s, 580 km/s
1993  WIND MFI submitted   : (16 March 1993)
1994  WIND launch          : (1 November 1994)
1995  THIS PAPER           : MFI instrument paper published
1995  GEOTAIL/GIM          : Sister fluxgate, near-tail orbit
2000  CLUSTER FGM          : 4-S/C constellation (Balogh et al.)
2007  STEREO/MAG           : Dual-S/C heliospheric (Acuña et al.)
2015  DSCOVR (LASCO at L1) : Operational successor (cross-cal w/ MFI)
2018  Parker Solar Probe   : FIELDS — direct MFI heritage
2020  Solar Orbiter MAG    : ESA/IAS — dual-sensor MFI heritage
2026  TODAY                : WIND/MFI still operating, 32+ yr
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Acuña (1974) IEEE Trans. Magnetics MAG-10 | Defines ring-core fluxgate principles used in MFI sensors | Direct hardware foundation / 직접적 하드웨어 기반 |
| Ness et al. (1971) JGR | Original dual-magnetometer concept (IMP/Explorer-43) | MFI is the matured version 30 yr later / MFI는 30년 후 성숙판 |
| Burlaga et al. (1981) JGR 86, 6673 | Defines magnetic clouds — primary MFI science target | MFI's KP cadence chosen for cloud detection / MFI KP cadence가 cloud 검출을 위해 설계됨 |
| Gonzalez & Tsurutani (1987) Planet. Space Sci. 35 | Storm amplitude ↔ cloud Bz, not shock strength | Primary motivation for Bz priority in KP / KP의 Bz 우선순위 동기 |
| Smith et al. (1986) Solar Wind-Magnetosphere Coupling | ISEE-3 SSC ↔ IP shock 80–90% statistic | Quantitative basis of shock studies that MFI extends / MFI가 확장하는 충격파 연구의 정량 기반 |
| Tsurutani et al. (1992) GRL 19, 73 | SSC and storm causes & classifications | MFI's inflow-classification science target / MFI 유입 분류 과학 대상 |
| Lepping et al. (1992) EOS | SSC MHD propagation 81 ± 18 s, 580 km/s | Same first author; methodology that WIND data refines / 같은 first author, WIND data로 정밀화하는 방법론 |
| Vinas & Scudder (1986) JGR 91 | Single-S/C shock-fitting | Used with MFI+SWE data on every shock crossing / 모든 shock 통과에서 MFI+SWE와 함께 사용 |
| Kokubun et al. (1990) GEOTAIL Interim Report | GEOTAIL/MGF instrument including GIM | MFI's sister; same team designed both / MFI 자매기, 같은 팀 설계 |
| Farrugia et al. (1993a–d) | Magnetic-cloud substorm/SEP/cusp studies | MFI-era observational follow-ups / MFI-시대 관측 후속 연구 |

---

## 7. References / 참고문헌

- Acuña, M. H., "Fluxgate Magnetometers for Outer Planets Exploration," *IEEE Trans. Magnetics* **MAG-10**, 519, 1974.
- Acuña, M. H. and Ness, N. F., in T. Gehrels (ed.), *Jupiter*, U. Arizona Press, p. 830, 1976a; *J. Geophys. Res.* **81**, 2917, 1976b.
- Burlaga, L. F., Sittler, E. C., Mariani, F., and Schwenn, R., "Magnetic Loop Behind an Interplanetary Shock," *J. Geophys. Res.* **86**, 6673, 1981.
- Burlaga, L. F., "Magnetic Clouds," in L. Lanzerotti, R. Schwenn, and E. Marsch (eds.), *Physics of the Inner Heliosphere*, Springer, 1991.
- Dessler, A. J., "Effect of Magnetic Anomaly on Particle Radiation Trapped in Geomagnetic Field," *J. Geophys. Res.* **63**, 405, 1958.
- Farrugia, C. J. et al., 1993a–d (multiple papers on magnetic-cloud / substorm / SEP / cusp coupling).
- Gonzalez, W. D. and Tsurutani, B. T., "Criteria of Interplanetary Parameters Causing Intense Magnetic Storms (Dst < −100 nT)," *Planet. Space Sci.* **35**, 1101, 1987.
- Lepping, R. P., Vinas, A. F., Lazarus, A. J., Sugiura, M., Iyemori, T., Kokubun, S. S., and Spreiter, J. R., "MHD Signal Propagation Times from Magnetopause to Earth's Surface," EOS, 1992.
- Lepping, R. P. et al., "The WIND Magnetic Field Investigation," *Space Sci. Rev.* **71**, 207–229, 1995. DOI: 10.1007/BF00751330. (this paper)
- Ness, N. F., "Magnetometers for Space Research," *Space Sci. Rev.* **11**, 111, 1970.
- Ness, N. F., Behannon, K. W., Lepping, R. P., and Schatten, K. H., "Use of Two Magnetometers for Magnetic Field Measurements on a Spacecraft," *J. Geophys. Res.* **76**, 3564, 1971.
- Smith, E. J., Slavin, J. A., Zwickl, R. D., and Bame, S. J., in *Solar Wind–Magnetosphere Coupling*, p. 345, 1986.
- Sugiura, M., "Propagation of Hydromagnetic Waves in the Magnetosphere," *Radio Science* **69D**, 1133, 1965.
- Tsurutani, B. T., Gonzalez, W. D., Tang, F., and Lee, Y. T., "Great Magnetic Storms," *Geophys. Res. Lett.* **19**, 73, 1992.
- Vinas, A. F. and Scudder, J. D., "Fast and Optimal Solution to the Rankine–Hugoniot Problem," *J. Geophys. Res.* **91**, 39, 1986.
- Wilson, C. R. and Sugiura, M., "Hydromagnetic Interpretation of Sudden Commencements of Magnetic Storms," *J. Geophys. Res.* **66**, 4097, 1961.

---

## Appendix A: Numerical worked examples / 부록 A — 수치 예제

### A.1 1/r³ separation with realistic geometry / 현실적 기하 1/r³ 분리

Suppose the spacecraft body has a residual magnetic moment $|\mathbf{m}|=0.05\;\text{A m}^2$ aligned roughly along the spin axis. The on-axis dipole field magnitude at distance $r$ is
$$|\mathbf{B}_d(r)| \;=\; \frac{\mu_0}{4\pi}\,\frac{2|\mathbf{m}|}{r^3}.$$
At $r_O=12$ m: $|\mathbf{B}_d|= 10^{-7}\cdot 2\cdot 0.05 / 1728 = 5.78\times 10^{-12}$ T $= 5.8\times 10^{-3}$ nT, well below the ±0.1 nT specification.
At $r_I=8$ m: $|\mathbf{B}_d|= 10^{-7}\cdot 0.1/ 512 = 1.95\times 10^{-11}$ T $= 0.0195$ nT. The 1/r³ ratio is exactly $(12/8)^3 = 3.375$. The OB+IB linear combination (M2) recovers the ambient field to better than 0.001 nT, comfortably within MFI's range-0 digital resolution.

스핀 축에 정렬된 잔여 쌍극자 모멘트 $|\mathbf{m}|=0.05\;\text{A m}^2$를 가정. OB(12 m)에서 dipole 자기장 ≃ 0.0058 nT, IB(8 m)에서 ≃ 0.0195 nT, 비율 정확히 (12/8)³ = 3.375. M2의 선형 결합으로 ambient 자기장이 0.001 nT 수준 (range-0 분해능) 이내로 복원된다.

### A.2 FFT bin telemetry budget / FFT 빈 텔레메트리 예산

Per FFT block (11.64 s of data):
- Raw 4 axes × 256 bins × 12-bit signed = 12,288 bits → 1056 bps if transmitted continuously.
- After 256 → 32 bin log-binning + 12-bit → 8-bit μ-law: 4 × 32 × 8 = 1024 bits per block.
- Block cadence 23 s (high-rate) → 1024/23 ≈ 44.5 bps total; the 55.6 BPS allocation in Mode 0/1 covers this with ∼25% header/parity margin.

블록 당 (11.64 s 데이터): 원본 12,288 bit. 32 log-bin × 8-bit μ-law = 1024 bit. 23 s cadence → 44.5 bps — Mode 0/1의 55.6 BPS 할당이 약 25% 헤더 마진을 두고 충분함.

### A.3 Snapshot pre-trigger arithmetic / 스냅샷 사전-트리거 산술

Snapshot 256 kbit ÷ (3 axes × 12 bit + housekeeping) ≈ 256,000 bit / 36 bit/vector ≈ 7111 vectors (paper cites 7282; difference due to actual word packing). At 44 vec/s this gives 7282/44 ≈ 165.5 s, of which a memory-pointer offset reserves 82 s pre-trigger, leaving 83 s post-trigger. For a typical fast-mode shock with up-stream wave train of 30–60 s upstream, the buffer comfortably contains the precursor *and* the ramp.

256 kbit ÷ 36 bit/vector ≈ 7111 vector (논문 표기 7282 - word packing 차이). 44 vec/s → 165.5 s, 그 중 memory-pointer로 82 s 사전-트리거 확보, 83 s 후속 trigger. 일반 fast-mode shock의 30–60 s upstream wave 영역과 ramp 모두 안전하게 버퍼링.

### A.4 Range-switching example / 레인지 절환 예제

Assume an Alfvén wave excursion drives $|B|$ from 8 nT (range 1, ±16 nT, 1792 counts at full = 16 nT → 8 nT = 896 counts) up to 18 nT in 5 s. At ∼900 counts the OB sensor remains in range 1; at $|B|>14$ nT (i.e. 7/8 of 16) the count exceeds 1792 → step up to range 2 (±64 nT, where 18 nT is now 576 counts). Subsequent decay below 1/8 of 64 = 8 nT triggers step-down back to range 1 — but the *guard band* at 1/8 of full-scale prevents repeated chatter at the boundary.

8 nT → 18 nT의 Alfvén wave 변동에서 14 nT (=7/8·16) 초과 시 range 1 → 2로 절환, 8 nT (=1/8·64) 미만으로 감쇠 시 다시 range 2 → 1. Guard band가 경계에서 반복 절환 (chatter)을 방지.

---

## Appendix B: Reading-cycle reflections / 부록 B — 학습 회고

이 논문은 *instrument paper*이지만 단순한 spec 나열이 아니라 1990년대 초의 ISTP 과학 비전, 30년 누적 fluxgate 기술, 그리고 multi-spacecraft 연합 관측 철학을 응축한 종합 문서다. 특히 (a) 듀얼 자력계의 *수학적* 우주선 자기장 분리, (b) snapshot의 사전-트리거 메모리 디자인, (c) on-board FFT의 μ-law 압축은 서로 독립적으로 보이지만 모두 *limited telemetry under non-negotiable science requirements*라는 동일한 제약 하의 해법이다. 이 관점에서 보면 MFI는 'precision fluxgate에 telemetry-aware DSP를 결합한 최초의 IMF 모니터'로 자리매김할 수 있다.

This is an instrument paper, but it is far more than a spec list — it condenses the early-1990s ISTP science vision, three decades of fluxgate heritage, and the multi-spacecraft coordinated-observation philosophy. The (a) algebraic dual-magnetometer S/C-field separation, (b) pre-trigger snapshot design, and (c) on-board μ-law-compressed FFT may look independent, but all three are solutions to the same constraint: *limited telemetry under non-negotiable science requirements*. From this perspective MFI can be characterised as the first IMF monitor to combine precision fluxgate sensors with telemetry-aware DSP.
