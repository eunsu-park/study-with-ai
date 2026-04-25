---
title: "The FIELDS Instrument Suite for Solar Probe Plus"
authors: "S.D. Bale et al."
year: 2016
journal: "Space Science Reviews"
doi: "10.1007/s11214-016-0244-5"
topic: Solar_Observation
tags: [parker_solar_probe, fields, instrumentation, plasma_waves, magnetometer, electric_field, in_situ, mission]
status: completed
date_started: 2026-04-25
date_completed: 2026-04-25
---

# 52. The FIELDS Instrument Suite for Solar Probe Plus / Solar Probe Plus를 위한 FIELDS 종합 관측 장비

---

## 1. Core Contribution / 핵심 기여

**English**:
Bale et al. (2016) is the reference instrument paper for the FIELDS suite aboard NASA's Solar Probe Plus (SPP, later renamed Parker Solar Probe). FIELDS is designed to make the first in situ measurements of electric and magnetic fields, plasma waves, electron density and temperature, dust impacts, and solar radio emissions inside 0.16 AU, reaching a closest perihelion of 9.86 solar radii (R_s). The hardware comprises five voltage probes (V1-V4 mounted at the base of the carbon-composite Thermal Protection System, V5 on the magnetometer boom), two GSFC-built triaxial fluxgate magnetometers (MAGi inboard at 1.9 m, MAGo outboard at 2.72 m), a CNES-built triaxial search-coil magnetometer (SCM, end-of-boom at 3.5 m), and a stack of digital boards: Antenna Electronics Boards (AEB1/AEB2), Digital Fields Board (DFB), Time Domain Sampler (TDS), Radio Frequency Spectrometer (RFS), Data Control Board (DCB), and dual Low Noise Power Supplies (LNPS1/LNPS2). The suite measures from DC to >20 MHz with 140 dB dynamic range and is split into two redundant halves (FIELDS1, FIELDS2) so no single failure compromises the mission.

The paper documents (i) Level-1 measurement requirements derived from re-analysis of 11 years of Helios 1 data, (ii) the spacecraft-charging and plasma-wake environment that drives sensor placement and biasing, (iii) every sensor and electronics board with its heritage, (iv) the EMC "picket-fence" power-supply scheme synchronized to a 150 kHz master clock and the 0.873813 s "NY second" time unit, (v) the burst-memory and Coordinated Burst Signal (CBS) data-prioritization scheme, and (vi) the 90-day orbit operations cycle from 8-day pre-perihelion calibration through aphelion playback. This paper is the canonical citation for every Parker Solar Probe fields/waves publication and the design template for future near-Sun missions.

**한국어**:
Bale 외 (2016)는 NASA Solar Probe Plus(SPP, 후에 Parker Solar Probe로 개명)에 탑재된 FIELDS 장비 모음에 대한 표준 기기 논문이다. FIELDS는 0.16 AU(최저 근일점 9.86 R_s) 이내에서 전기장·자기장, 플라즈마파, 전자 밀도·온도, 먼지 충돌, 태양 전파 방출을 인-시추(in situ)로 측정하는 최초의 기기다. 하드웨어 구성은 다음과 같다: 5개의 전압 탐침(V1-V4는 carbon-composite Thermal Protection System 기저부에, V5는 자력계 boom 위에 위치), 2개의 GSFC제 삼축 fluxgate 자력계(MAGi 1.9 m 내측, MAGo 2.72 m 외측), 1개의 CNES제 삼축 search-coil 자력계(SCM, 3.5 m boom 끝부), 그리고 디지털 보드 스택인 Antenna Electronics Board(AEB1/AEB2), Digital Fields Board(DFB), Time Domain Sampler(TDS), Radio Frequency Spectrometer(RFS), Data Control Board(DCB), 이중화된 Low Noise Power Supply(LNPS1/LNPS2). 측정 대역은 DC ~ 20 MHz 이상, 동적 범위 140 dB이며, 단일 고장이 미션 전체를 무력화하지 않도록 FIELDS1, FIELDS2 두 개의 독립적 절반으로 분할되었다.

이 논문은 (i) Helios 1의 11년 데이터 재분석에서 도출된 Level-1 측정 요구사항, (ii) 센서 배치와 바이어싱을 좌우하는 우주선 charging 및 플라즈마 wake 환경, (iii) 각 센서·전자보드의 설계와 헤리티지, (iv) 150 kHz 마스터 클럭과 0.873813 s NY second 시간단위에 동기화된 EMC "picket-fence" 전원 정책, (v) burst memory와 Coordinated Burst Signal(CBS) 기반 데이터 우선순위 결정 체계, (vi) 90일 주기 궤도(perihelion 전 8일 calibration → 6일 high-rate science → SSR playback → aphelion event selection)의 운영 흐름을 모두 기술한다. 이 논문은 Parker Solar Probe의 모든 fields/waves 출판물의 표준 인용이 되었고 차세대 근태양 미션의 설계 템플릿이 된다.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction & Mission Context (Sect. 1) / 서론과 미션 맥락

**English**:
FIELDS is one of four SPP instrument suites (along with SWEAP for thermal plasma, IS⊙IS for energetic particles, WISPR for white-light imaging). Its primary measurement targets are (a) the 3-component magnetic field from DC beyond electron cyclotron frequency, (b) two components of the DC and fluctuating electric field in the plane of the heat shield from V1-V4, (c) the third (sunward) component from V5, (d) quasi-thermal noise (QTN) for n_e and T_e diagnostics, and (e) solar radio emissions to 20 MHz. Heritage: Wind, Polar, STEREO, THEMIS, Van Allen Probes (RBSP). The launch was originally mid-2018 (actual launch: August 12, 2018).

**한국어**:
FIELDS는 SPP의 네 개 기기 모음 중 하나(나머지는 열적 플라즈마 SWEAP, 에너지입자 IS⊙IS, 백색광 영상 WISPR)이다. 주요 측정 항목은 (a) DC ~ 전자 자이로주파수 이상 대역의 3성분 자기장, (b) heat shield 평면 안의 DC·요동 전기장 2성분(V1-V4), (c) sunward 방향 3성분(V5), (d) n_e·T_e 진단을 위한 quasi-thermal noise(QTN), (e) 20 MHz까지의 태양 전파 방출이다. 헤리티지는 Wind, Polar, STEREO, THEMIS, Van Allen Probes(RBSP). 원래 2018년 중반 발사 계획(실제 발사: 2018년 8월 12일).

### Part II: Level-1 Measurement Requirements (Sect. 1.1, Table 1) / Level-1 측정 요구사항

| Measurement | Dynamic range | Cadence | Bandwidth |
|---|---|---|---|
| Magnetic Field | 140 dB | 100k vectors/s | DC–50 kHz |
| Electric Field | 140 dB | 2M vectors/s | DC–1 MHz |
| Plasma Waves | 140 dB | 1 spectrum/s | 5 Hz–1 MHz |
| QTN/Radio | 100 dB QTN, 80 dB radio | 1/(4 s) QTN, 1/(16 s) radio | 10–2500 kHz QTN, 1–16 MHz radio |

**English**:
The 140 dB dynamic range is required because plasma fluctuations span seven decades from quiet-solar-wind turbulence (~10^-9 nT/√Hz at 1 Hz) up to large-amplitude shocks and Langmuir waves (V/m in E). The 100k vectors/s magnetic and 2M vectors/s electric cadences reach beyond the proton cyclotron frequency (32 Hz at 10 R_s) and electron cyclotron frequency (60 kHz at 10 R_s).

**한국어**:
140 dB 동적 범위는 플라즈마 요동이 조용한 태양풍 난류(1 Hz에서 ~10^-9 nT/√Hz) 부터 대진폭 충격파·Langmuir 파(전기장 V/m 단위)까지 7차원 이상 펼쳐지기 때문에 요구된다. 100k vectors/s 자기장과 2M vectors/s 전기장 cadence는 10 R_s에서의 proton cyclotron(32 Hz)와 electron cyclotron(60 kHz) 주파수를 모두 넘어선다.

### Part III: Plasma Environment of Inner Heliosphere (Sect. 1.2, Table 2, Figs. 1-2) / 내부 태양권 플라즈마 환경

**English**:
Re-analysis of Helios 1 (1974-1986, 0.3-1 AU) gives radial power-laws extrapolated inward. Best fits (Fig. 1): |B| follows a Parker spiral with B_0 ≈ 4 nT at 1 AU; v_sw uses Cranmer (2007) equatorial speed and an empirical "Sheeley-like" model fit to Helios; n_p follows a Sittler-Guhathakurta or Helios power-law (~r^-2.13); T_p ~ T_0 r^-α with α ≈ 0.6, T_p ≈ 9 eV at 1 AU. Table 2 expected values at 10 R_s perihelion: |B_0| = 2000 nT, |E_c| = v_sw·δB_A ≈ 100 mV/m, n_e = 7000 cm^-3, T_e = 85 eV, v_sw = 210 km/s, v_A = 500 km/s, f_pe = 750 kHz, f_ce = 60 kHz, f_cp = 32 Hz; convected Debye length 4 μs, electron inertial 0.3 ms, proton inertial 13 ms, proton gyroscale 9 ms.

The turbulence break-point between energy injection (k^-1 or f^-1) and inertial (f^-5/3 Kolmogorov) ranges scales radially as f_i (Hz) ≈ 4.9 r^-1.66 (r in R_s), giving f_i ≈ 100 Hz at 10 R_s, with associated turbulence amplitude δB^2 (nT^2/Hz) ≈ 10^8.1 r^-2 ≈ 10^6.1 nT^2/Hz. The kinetic dissipation break is at f_d ≈ f_ci (v_A + v_sw)/v_thi.

**한국어**:
Helios 1 (1974-1986, 0.3-1 AU) 데이터 재분석으로 안쪽으로 외삽한 반경 거듭제곱(Fig. 1): |B|는 1 AU에서 B_0 ≈ 4 nT인 Parker 나선식; v_sw는 Cranmer(2007) 적도 속도와 Helios에 맞춘 실험적 "Sheeley-like" 모형; n_p는 Sittler-Guhathakurta 혹은 Helios 거듭제곱(~r^-2.13); T_p ~ T_0 r^-α with α ≈ 0.6, 1 AU에서 T_p ≈ 9 eV. Table 2의 10 R_s 근일점 예상치: |B_0| = 2000 nT, |E_c| = v_sw·δB_A ≈ 100 mV/m, n_e = 7000 cm^-3, T_e = 85 eV, v_sw = 210 km/s, v_A = 500 km/s, f_pe = 750 kHz, f_ce = 60 kHz, f_cp = 32 Hz; convected 길이 척도는 Debye 4 μs, electron inertial 0.3 ms, proton inertial 13 ms, proton gyroscale 9 ms.

난류 에너지 주입 영역(k^-1 또는 f^-1)과 inertial(Kolmogorov f^-5/3) 영역의 break-point 주파수는 f_i (Hz) ≈ 4.9 r^-1.66(r은 R_s 단위)로 스케일하며 10 R_s에서 ≈100 Hz, 진폭은 δB^2 (nT^2/Hz) ≈ 10^8.1 r^-2 ≈ 10^6.1 nT^2/Hz. kinetic 소산 break은 f_d ≈ f_ci (v_A + v_sw)/v_thi.

### Part IV: Spacecraft Charging and Wake (Sect. 1.3, Fig. 3) / 우주선 charging과 wake

**English**:
Photoemission currents from sunlit surfaces near perihelion exceed plasma electron currents by 10-100×. Sunlit surfaces typically equilibrate at +5 to +10 V; shadowed surfaces would charge negatively (potentially hundreds of volts) without electrostatic-cleanliness controls. SPP enforces conductive coupling so the entire spacecraft body floats together. However, the heat-shield photoelectron Debye length is small compared to TPS dimensions, creating an "electrostatic barrier" (Fig. 3, the +X side ~ thin negative layer) that blocks ~90% of photoelectron escape, paradoxically driving the spacecraft to ~−25 V even with cleanliness compliance (Ergun et al. 2010, Guillemant et al. 2012).

The plasma wake forms downstream because v_sw exceeds ion thermal speed but electron thermal speed exceeds v_sw — an ion cavity fills with electrons at potential V_wake ~ −60 V at 10 R_s (~70% of k_B T_e). The wake direction at perihelion is ~45° from the Sun (the spacecraft moves at 180 km/s vs solar wind 200 km/s in the spacecraft frame). To minimize wake contamination, V1-V4 are placed forward (sunward) at the base of the TPS.

**한국어**:
근일점에서 햇빛 받는 표면의 광전자 전류가 플라즈마 전자 전류를 10-100배 초과한다. 햇빛 받는 면은 +5 ~ +10 V로 평형을 이루고, 그늘진 면은 정전기 청결성 제어가 없으면 수백 V 음전위로 charging될 수 있다. SPP는 전체 우주선 표면이 한 몸으로 부유하도록 도전체로 결합한다. 그러나 heat shield의 광전자 Debye 길이가 TPS 치수에 비해 작아 비단조 "electrostatic barrier"(Fig. 3, +X 측 얇은 음전위 층)가 형성되어 광전자 탈출의 ~90%를 차단하고, 청결성 준수에도 우주선 전체가 ~−25 V로 charging된다(Ergun et al. 2010, Guillemant et al. 2012).

플라즈마 wake은 v_sw가 이온 열속도를 초과하지만 전자 열속도가 v_sw를 초과하기 때문에 후방에 형성된다 — 이온 cavity가 전자로 채워져 전위 V_wake ~ −60 V (10 R_s에서 k_B T_e의 70%). 근일점에서 우주선은 180 km/s, 태양풍은 200 km/s로 우주선 프레임에서 wake 방향은 태양으로부터 ~45°. wake 오염을 최소화하기 위해 V1-V4는 TPS 기저부에 전방(sunward)으로 배치된다.

### Part V: Sensors (Sect. 2.1) / 센서

**Sect. 2.1.1 V1-V4 Electric Antennas** (Fig. 6 CAD):
**English**: Each unit consists of a 2 m long 1/8" diameter Niobium C-103 thin-walled "whip" sensor + 30 cm Molybdenum "stub" thermal/electrical isolator + chevron-shaped C-103 heat shield + preamp at base. The whip reaches >1300 °C at perihelion; the stub reaches <230 °C; sapphire isolators thermally separate stages. Pure Niobium signal wire runs through the stub to the preamp. The V1-V4 deploy by spring force after launch with rate-limiting flyweight brakes; each sensor deploys individually for in-flight characterization.

**한국어**: 각 유닛은 2 m 길이, 1/8" 직경의 Niobium C-103 박벽관 "whip" 센서 + 30 cm Molybdenum "stub" 열·전기 절연체 + chevron 모양의 C-103 heat shield + 기저부 preamp로 구성. whip은 근일점에서 >1300 °C, stub은 <230 °C; sapphire 절연체로 단을 열적으로 분리. 순수 Niobium 신호선이 stub을 통과해 preamp까지 연결. V1-V4는 발사 후 스프링 힘으로 전개되고 flyweight brake로 속도 제한; 보정을 위해 각 센서를 개별 전개.

**Sect. 2.1.2 V5 Voltage Sensor** (Fig. 7):
**English**: V5 is a simple voltage probe on the magnetometer boom in the spacecraft umbra (in shadow), coupled to plasma through thermal electrons rather than photoelectrons. Two short tubes act as a single tied sensor (not differential). It senses the radial component E_∥ of plasma waves and helps locate the electrostatic center of the spacecraft.

**한국어**: V5는 자력계 boom 위 그림자(umbra) 영역의 단순 전압 탐침으로, 광전자가 아닌 열적 전자를 통해 플라즈마와 결합. 짧은 두 튜브가 하나로 묶인 단일 센서(차동 아님). 플라즈마 파의 반경 성분 E_∥을 감지하고 우주선의 electrostatic 중심 위치 파악을 돕는다.

**Sect. 2.1.3 Electric Preamps** (Fig. 8):
**English**: V1-V4 preamps provide three outputs: HF (20 MHz BW) → RFS, MF (1 MHz BW) → TDS, LF (64 kHz BW) → AEB/DFB. The HF chain uses a FET buffer + wide-bandwidth op amp + 50 Ω terminator. LF/MF use a unity-gain op amp on a "floating ground driver" supply with ±70 V range from DC to 300 Hz, ±10 V from 300 Hz to 1 MHz. V5 has only LF/MF (no HF). The capacitive gain of ~−0.4 with R_sheath up to 329 kΩ at 0.25 AU first-science distance.

**한국어**: V1-V4 preamp는 세 출력을 제공: HF(20 MHz BW) → RFS, MF(1 MHz BW) → TDS, LF(64 kHz BW) → AEB/DFB. HF 사슬은 FET 버퍼 + 광대역 op amp + 50 Ω 종단. LF/MF는 "floating ground driver" 공급의 unity gain op amp로 DC~300 Hz에서 ±70 V, 300 Hz~1 MHz에서 ±10 V 범위. V5는 LF/MF만 (HF 없음). 0.25 AU 첫 과학 거리에서 R_sheath 최대 329 kΩ에서 capacitive gain ~−0.4.

**Sect. 2.1.4 Fluxgate Magnetometers** (Fig. 9-10):
**English**: GSFC-built triaxial fluxgates similar to those for MAVEN, Van Allen Probes, STEREO. Bandwidth ~140 Hz, sampling 292.97 Sa/s, range ±65,536 nT, 16-bit resolution, 4 auto-ranges (±1024, ±4096, ±16,384, ±65,536 nT). MAGi (inboard) at 1.9 m, MAGo (outboard) at 2.72 m. Composite kinematic mounts and proportional AC heater minimize sensor temperature variation. MAGi is read by the TDS, MAGo by the DCB. Fluxgate cadence: 1 message/0.874 s. Total flight heritage: 79 instruments since IMP-4 (1966).

**한국어**: MAVEN, Van Allen Probes, STEREO 헤리티지의 GSFC제 삼축 fluxgate. 대역폭 ~140 Hz, 샘플링 292.97 Sa/s, 범위 ±65,536 nT, 16-bit 분해능, 4개 자동 ranging(±1024, ±4096, ±16,384, ±65,536 nT). MAGi(내측) 1.9 m, MAGo(외측) 2.72 m. 복합재 kinematic mount와 비례 AC 히터로 센서 온도 변화 최소화. MAGi는 TDS가, MAGo는 DCB가 읽음. cadence는 1 message/0.874 s. IMP-4(1966) 이후 총 79기 비행 헤리티지.

**Sect. 2.1.5 Search-Coil Magnetometer** (Fig. 11-12):
**English**: CNES (LPC2E)/Berkeley triaxial search-coil. Two ELF/VLF sensors (10 Hz–50 kHz) plus one dual-band sensor covering ELF/VLF and LF/MF (1 kHz–1 MHz). Each sensor 104 mm long, mounted orthogonally on non-magnetic support. Located at end of boom (3.5 m). Dynamic range: 160 dB ELF/VLF, 130 dB LF/MF — peak signals can reach 3000 nT at 0.29 AU. The instrument has a heater to stay above deep-space temperatures, MLI insulation, 3D-printed miniaturized preamp at the base.

**한국어**: CNES (LPC2E)/Berkeley 삼축 search-coil. 두 ELF/VLF 센서(10 Hz–50 kHz)와 ELF/VLF·LF/MF 양 대역(1 kHz–1 MHz)을 모두 다루는 하나의 dual-band 센서. 각 104 mm 길이, 비자성 지지대 위에 직교 배치. boom 끝(3.5 m)에 위치. 동적범위: ELF/VLF 160 dB, LF/MF 130 dB — 0.29 AU에서 최대 3000 nT까지. 깊은 우주 온도 이상으로 유지하기 위한 heater, MLI 단열, 기저부의 3D 프린트 소형화 preamp.

### Part VI: Main Electronics Package (Sect. 2.2) / 주 전자장비 패키지

**Sect. 2.2.1 AEB**:
**English**: AEB1 (FIELDS1 side) handles V1, V2, V5; AEB2 (FIELDS2 side) handles V3, V4. DC signal gain ≈ unity, dynamic range ±115 V. AEB generates current biases and voltage biases for the whip and stub/shield. Three current-bias ranges (±802 nA, ±14.1 μA, ±414 μA, 12-bit accuracy = 0.025% of full range). Current bias bandwidth 450 Hz. Voltage biases on V1-V4 stub/shield up to ±40 V offset relative to sensor (±60 V DC for sensor-to-S/C differences). The bias circuit puts the operational point on the current-voltage curve at the plasma potential, where dV/dI is minimum. AEB also houses the floating ground driver (±100 V dynamic range, 450 Hz BW) and floating power supplies for the preamps.

**한국어**: AEB1(FIELDS1 측)이 V1, V2, V5 담당; AEB2(FIELDS2 측)가 V3, V4 담당. DC 신호 이득 ≈ unity, 동적범위 ±115 V. AEB는 whip과 stub/shield용 전류 바이어스와 전압 바이어스 생성. 전류 바이어스 3 범위(±802 nA, ±14.1 μA, ±414 μA, 12-bit 정확도 = 풀 레인지의 0.025%). 전류 바이어스 대역폭 450 Hz. V1-V4 stub/shield 전압 바이어스는 센서 대비 최대 ±40 V offset(센서-S/C 차이 ±60 V DC). 바이어스 회로는 dV/dI가 최소인 plasma potential에 동작점을 둔다. AEB는 floating ground driver(±100 V 동적범위, 450 Hz BW)와 preamp용 floating 전원도 내장.

**Sect. 2.2.2 DFB** (Fig. 14, Table 3):
**English**: The DFB processes 26 input signals at 150 kSa/s, performs digital filtering, and produces 25 digital data streams. Inputs: V1-V5 (DC-coupled, AC-coupled), differential combinations E_12=V_1-V_2, E_34=V_3-V_4, E_z=V_5-(V_1+V_2+V_3+V_4)/4 in low-gain and high-gain DC, plus AC; SCM B_x, B_y, B_z (LF, low/high gain), and B_x (MF). Anti-alias filters: 4-pole low-pass Bessel −3 dB at 7.5 kHz (DC channels) and 60 kHz (AC channels) on E/V; 6-pole low-pass Bessel −3 dB at 60 kHz on SCM. After digitization on Teledyne SIDECAR ASIC (16-bit ADC × 32), cascading digital filter banks (5th-order FIR Bessel) decimate by factors of 2 to produce waveforms at 18.75/2^N kSa/s (low-speed, N=0..14) and 150/2^N kSa/s (high-speed, N=0..6). Burst memory (DBM) holds ~3.5 s × 6 channels of high-rate snapshots evaluated by Coordinated Burst Signal (CBS) at 4 times/NYsec. Spectral products: bandpass filter (BP) bank with 15 (low-speed) and 7 (high-speed) bins; FFT-based power spectra and cross spectra on 1024-point Hanning-windowed segments at 9.375 kSa/s low-speed or 150 kSa/s high-speed, averaged into 56 or 96 pseudo-logarithmic frequency bins (df/f = 6-12% with 56 bins, 3-6% with 96 bins).

**한국어**: DFB는 26개 입력 신호를 150 kSa/s로 처리하여 25개 디지털 데이터 스트림을 만든다. 입력: V1-V5(DC, AC), 차동 조합 E_12=V_1-V_2, E_34=V_3-V_4, E_z=V_5-(V_1+V_2+V_3+V_4)/4의 low-gain/high-gain DC와 AC; SCM B_x, B_y, B_z(LF, low/high gain), B_x(MF). Anti-alias filter: 4-pole 저역통과 Bessel −3 dB at 7.5 kHz(DC 채널)와 60 kHz(AC 채널) on E/V; 6-pole Bessel −3 dB at 60 kHz on SCM. Teledyne SIDECAR ASIC(16-bit ADC × 32)로 디지타이즈한 후, cascading 디지털 필터뱅크(5차 FIR Bessel)가 2배씩 데시메이션하여 low-speed 18.75/2^N kSa/s(N=0..14), high-speed 150/2^N kSa/s(N=0..6) 파형을 만든다. Burst memory(DBM)에 ~3.5 s × 6 채널 고속 snapshot을 보관, Coordinated Burst Signal(CBS)로 NYsec당 4번 품질 평가. 스펙트럼: bandpass filter bank(BP) 15(low) /7(high) 빈; 1024 포인트 Hanning windowed 9.375 kSa/s(저속) 또는 150 kSa/s(고속) FFT 기반 전력/교차 스펙트럼, 56 또는 96 pseudo-log 주파수 빈 평균(df/f = 56빈에서 6-12%, 96빈에서 3-6%).

**Sect. 2.2.3 TDS** (Fig. 15, Table 4):
**English**: TDS samples 6 channels at 1.92 MSa/s (effective ~1 MHz Nyquist), with commandable lower rates (480, 120 kSa/s, ...). Inputs: V1, V2, V3, V4, V5, B_MF (one search-coil channel). 16-bit ADCs, ~30 μV RMS noise at 100 kHz. Top throughput is 160 Mbps, while telemetry is only ~hundreds of bps — TDS uses on-board "quality" evaluation to select interesting events. Captures 65,536 samples (~33 ms) per event. Provides 1-message-per-minute monitor with peak/mean/RMS/zero-crossing-count/dust impacts. Heritage: STEREO/WAVES TDS (Bougeret et al. 2008).

**한국어**: TDS는 6 채널을 1.92 MSa/s(유효 ~1 MHz Nyquist)로 샘플링, 명령 가능한 저속(480, 120 kSa/s …). 입력: V1-V5와 B_MF(서치코일 한 채널). 16-bit ADC, 100 kHz에서 ~30 μV RMS 잡음. 최고 처리량 160 Mbps인 반면 텔레메트리는 수백 bps — 따라서 온보드 "quality" 평가로 흥미 이벤트만 선택. 이벤트당 65,536 샘플(~33 ms) 포착. 분당 1 메시지 모니터(피크/평균/RMS/zero-crossing 카운트/먼지 충돌). 헤리티지: STEREO/WAVES TDS(Bougeret et al. 2008).

**Sect. 2.2.4 RFS** (Fig. 16-17):
**English**: Dual-channel digital spectrometer using HF preamp output. Input choices via multiplexer: dipole (any two antennas) or monopole (antenna − S/C ground) plus B_MF from SCM. Sample at 38.4 MSa/s, Nyquist 19.2 MHz. EMC plan locks DC-DC chopping to 150 kHz multiples (picket fence), creating noise-free spectral gaps. RFS subdivides into LFR (10 kHz–2.4 MHz; QTN focus) and HFR (~1.6 MHz–19.2 MHz; remote sensing focus). LFR uses Cascade Integrator Comb (CIC) filter to downsample by 8 → 4.8 MSa/s for finer frequency resolution. The DSP chain uses 8-tap polyphase filter bank (PFB, Vaidyanathan 1990) with 32,768-sample input → 4096-point FFT → 2048 positive frequencies, of which selected bins are saved. Δf/f ≈ 4.5% in both LFR and HFR. Sensitivity reaches a few nV/√Hz at ~1 MHz, sufficient to observe galactic synchrotron background as absolute calibration source.

**한국어**: HF preamp 출력 사용 이중 채널 디지털 분광기. 멀티플렉서 입력 선택: dipole(임의 두 안테나) 또는 monopole(안테나 − S/C 접지) 및 SCM의 B_MF. 38.4 MSa/s 샘플링, Nyquist 19.2 MHz. EMC 정책으로 DC-DC chopping을 150 kHz 정수배에 고정(picket fence) → 노이즈 없는 스펙트럼 틈 확보. RFS는 LFR(10 kHz–2.4 MHz; QTN 중심)과 HFR(~1.6 MHz–19.2 MHz; 원격 감지 중심)로 분할. LFR는 Cascade Integrator Comb(CIC) 필터로 8배 다운샘플링 → 4.8 MSa/s로 미세 주파수 분해능. DSP 사슬: 8-탭 polyphase filter bank(PFB, Vaidyanathan 1990)로 32,768 샘플 입력 → 4096점 FFT → 2048개 양 주파수, 선택된 빈만 저장. LFR/HFR 모두 Δf/f ≈ 4.5%. 감도는 ~1 MHz에서 수 nV/√Hz 수준 — 은하 동기복사 배경(절대 보정원)을 관측하기에 충분.

**Sect. 2.2.5 DCB** (Fig. 17, 19):
**English**: DCB hosts a Coldfire 32-bit IP-Core in a radiation-hard RTAX-4000 FPGA, with 32 kB PROM, 512 kB EEPROM, 2 MB SRAM, and a 32 GB flash bulk store. Provides 38.4 MHz master clock (= 256 × 150 kHz) for all FIELDS subsystems and the picket-fence supplies. Runs flight software for AEB/RFS/LNPS/DFB/TDS/MAG control, CCSDS packet generation, median filtering, peak tracking of the LF spectrum to determine plasma frequency, and CBS computation. Flash-bad-block management with Virtual-to-Physical mapping; hardware-rad-hard scrubbing for single-event upsets. 7-year minimum mission lifetime.

**한국어**: DCB는 방사선 강화 RTAX-4000 FPGA 위의 Coldfire 32-bit IP-Core, PROM 32 kB, EEPROM 512 kB, SRAM 2 MB, 32 GB 플래시 대용량 저장. 38.4 MHz 마스터 클럭(= 256 × 150 kHz) 제공으로 모든 FIELDS 서브시스템과 picket-fence 전원을 동기화. 비행 소프트웨어가 AEB/RFS/LNPS/DFB/TDS/MAG 제어, CCSDS 패킷 생성, median 필터, LF 스펙트럼의 peak tracking으로 plasma 주파수 결정, CBS 계산을 담당. 플래시 bad block은 Virtual-to-Physical 매핑으로 관리; 하드웨어 rad-hard scrubbing으로 single-event upset 정정. 최소 7년 미션 수명.

**Sect. 2.3 LNPS**:
**English**: Two independent supplies (LNPS1 powering RFS/DFB/MAGo/SCM/AEB1/DCB; LNPS2 powering TDS/MAGi/AEB2). Pre-regulator buck-coil to 12 V, then PWM at 150 kHz drives 6 transformers; second-stage PWMs slaved to first; soft-start; common-mode chokes on every output. Total power: 7-11 W (LNPS1) and 4-7 W (LNPS2).

**한국어**: 두 독립 전원(LNPS1: RFS/DFB/MAGo/SCM/AEB1/DCB 공급; LNPS2: TDS/MAGi/AEB2 공급). buck-coil prereg 12 V → 150 kHz PWM 6개 transformer; 2단 PWM은 1단에 slave; soft-start; 모든 출력에 common-mode choke. 총 전력: LNPS1 7-11 W, LNPS2 4-7 W.

### Part VII: Operations (Sect. 3, Fig. 18) / 운영

**English**: 90-day orbit. (1) ~8 days pre-perihelion: Calibration mode + antenna bias sweeps; z-axis spacecraft slew calibration. (2) ~6 days at perihelion: High-rate science, ~20 kbps real-time Survey + ~100 kbps to 32 GB SSR. (3) Post-encounter: Calibration and bias sweeps, return to low-rate Survey. (4) Off period: heaters keep sensors warm; previous-encounter Survey downlink. (5) Burst selection: SOC reviews quick-look, identifies high-interest periods. (6) Aphelion: command sequence for next perihelion. Data products: L0 raw → L1 engineering units (24 hours, automated) → L2 physical units (24 hours, automated) → L3 (3 months, with SWEAP V×B removal, MAG+SCM merge, validated) → L4 event lists (3 months). Output: ISTP-compliant CDF via FIELDS webpage and SPDF/CDAWeb.

**한국어**: 90일 궤도. (1) 근일점 전 ~8일: Calibration 모드와 안테나 바이어스 sweep; z-축 우주선 slew 보정. (2) 근일점 ±6일: High-rate science, ~20 kbps 실시간 Survey + ~100 kbps 32 GB SSR 저장. (3) Post-encounter: Calibration·bias sweep 후 low-rate Survey 복귀. (4) 비활성 기간: 히터로 센서 온도 유지; 이전 인카운터 Survey 다운링크. (5) Burst 선택: SOC가 quick-look 검토하여 고관심 구간 결정. (6) Aphelion: 다음 근일점 명령 시퀀스 작성. 데이터 산출: L0 raw → L1 엔지니어링 단위(24시간, 자동) → L2 물리단위(24시간, 자동) → L3(3개월, SWEAP V×B 제거, MAG+SCM 병합, 검증) → L4 이벤트 목록(3개월). 산출물: ISTP-compliant CDF는 FIELDS 웹페이지와 SPDF/CDAWeb로 배포.

---

## 3. Key Takeaways / 핵심 시사점

1. **FIELDS는 코로나 인-시추 전자기 측정의 단일 진입점이다 / FIELDS is the single in situ EM gateway to the corona** — One coordinated suite (5 V-probes + 3 magnetometers + 6 digital boards + 2 power supplies) covers DC to 20 MHz with 140 dB dynamic range. Inside 9.86 R_s, it samples regions previously seen only by remote sensing. / 5개 V-probe + 3 자력계 + 6 디지털 보드 + 2 전원의 통합 모음이 DC ~ 20 MHz와 140 dB 동적범위를 모두 커버한다. 9.86 R_s 안쪽은 원격 관측만 가능했던 영역이다.

2. **하드웨어 분할 이중화(FIELDS1/FIELDS2)는 미션 보장을 위한 설계 원칙이다 / Two-sided redundancy (FIELDS1/FIELDS2) protects mission-critical measurements** — Failure mode analysis showed a single failure in FIELDS could lose unacceptable science. The split AEB/MAG/LNPS/processor architecture means no single point of failure compromises the suite. / 단일 고장 분석에서 단일 실패가 허용 불가한 과학 손실을 일으킬 수 있음이 드러나 AEB/MAG/LNPS/프로세서를 모두 두 측으로 분할.

3. **Picket-fence EMC + 150 kHz 마스터 클럭이 노이즈를 좁은 스펙트럼 줄로 가둔다 / Picket-fence EMC + 150 kHz master clock confine noise to narrow spectral lines** — All DC-DC converters chop at integer multiples of 150 kHz on a crystal-controlled clock. Picket-fence harmonics fill known frequencies; PFB-FFT preserves clean inter-line bins for nV/√Hz radio measurements. / 모든 DC-DC 변환기가 결정자 제어된 150 kHz의 정수배에서만 chopping. picket-fence 고조파가 알려진 주파수만 차지하므로 PFB-FFT가 그 사이의 깨끗한 빈을 보존하여 nV/√Hz 전파 측정 가능.

4. **광전자 wake와 electrostatic barrier는 sensor 배치를 결정한다 / Photoelectron wake and electrostatic barrier dictate sensor placement** — V1-V4 sit forward at the TPS base in full sunlight (current bias on photoelectron curve), V5 sits in shadow on the boom (thermal-electron coupling). The forward placement minimizes ion-wake contamination of the in-plane E-field. / V1-V4는 햇빛 전방 TPS 기저(광전자 곡선 위 전류 바이어스), V5는 boom의 그림자(열전자 결합). 전방 배치로 이온 wake 오염 최소화.

5. **Coordinated Burst Signal(CBS)이 데이터 비율 100,000:1 격차를 해결한다 / The Coordinated Burst Signal solves the 100,000:1 data-rate gap** — TDS produces 160 Mbps but telemetry is only ~hundreds of bps. CBS is a weighted linear combination of DFB+TDS+RFS+SWEAP quality metrics calculated at 4×/NYsec to identify the most science-rich burst windows for ground retrieval. / TDS는 160 Mbps 생산하지만 텔레메트리는 수백 bps. CBS는 DFB+TDS+RFS+SWEAP 품질지표의 가중합으로 NYsec당 4번 계산되어 지상 회수 시 최고 가치 burst 구간 선정.

6. **NY second는 모든 FIELDS 데이터의 시간 격자를 정의한다 / The NY second defines the FIELDS time grid** — 1 NY sec = 2^17/150,000 ≈ 0.873813 s. All cadences are power-of-two divisions: DBM at 150 kSa/s = 2^17/NYsec; MAG at 256 Sa/NYsec ≈ 293 Sa/s. This power-of-two structure preserves FFT efficiency and exact decimation. / 1 NY sec = 2^17/150,000 ≈ 0.873813 s. 모든 cadence는 2의 거듭제곱 분수: DBM 150 kSa/s = 2^17/NYsec, MAG 256 Sa/NYsec ≈ 293 Sa/s. 2의 거듭제곱 구조가 FFT 효율과 정확 데시메이션 보장.

7. **Helios 1 데이터 재분석이 SPP 사양을 정량적으로 결정했다 / Re-analysis of Helios 1 quantitatively set SPP requirements** — Power-law extrapolations of |B|, n_e, T_e plus Sittler-Guhathakurta and Sheeley models give Table 2 perihelion estimates. Turbulence break-points scaling f_i ~ r^-1.66, δB^2 ~ r^-2 produce the noise-floor specs (Fig. 2 SCM/MAG). / |B|, n_e, T_e의 거듭제곱 외삽과 Sittler-Guhathakurta, Sheeley 모형으로 Table 2의 근일점 추정치 획득. 난류 break f_i ~ r^-1.66, δB^2 ~ r^-2 스케일링이 SCM/MAG 노이즈 한계 사양을 결정(Fig. 2).

8. **헤리티지 기기들의 종합이 새로운 임무를 가능케 했다 / Synthesis of heritage instruments enabled the new mission** — V1-V4 inherits THEMIS/RBSP double-probe; V5 from inner-magnetosphere monopole sensors; MAG from MAVEN/RBSP/STEREO/Juno fluxgates; SCM from Cluster/Cassini/Solar Orbiter LPC2E search-coils; TDS from STEREO/WAVES TDS; RFS from STEREO/WAVES + RBSP/EFW spectral processing. PSP would have been infeasible without these flight-proven foundations. / V1-V4는 THEMIS/RBSP double-probe, V5는 내부 자기권 monopole, MAG는 MAVEN/RBSP/STEREO/Juno 자력계, SCM은 Cluster/Cassini/Solar Orbiter LPC2E, TDS는 STEREO/WAVES TDS, RFS는 STEREO/WAVES + RBSP/EFW 스펙트럼 처리에서 계승. 비행 검증 토대 없이는 PSP 자체가 불가능.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Double-probe E-field reconstruction / 이중 탐침 전기장 재구성

**Three orthogonal components from five sensors / 5 센서로 3 성분**:
$$
E_{12} = \frac{V_1 - V_2}{L_{12}}, \quad
E_{34} = \frac{V_3 - V_4}{L_{34}}, \quad
E_z = \frac{V_5 - \bar V}{L_z}, \quad
\bar V = \frac{V_1+V_2+V_3+V_4}{4}
$$
- L_{12}, L_{34}: effective baselines between V1-V2 and V3-V4 antennas (≈ 6-7 m total tip-to-tip after deployment).
- L_z: effective baseline from V5 to the centroid of V1-V4.
- Subtracting the mean V̄ removes common-mode S/C floating potential.
- 한국어: V1-V2와 V3-V4 사이 유효 거리, V̄ 빼기는 공통모드(S/C 부유전위) 제거.

### 4.2 Search-coil voltage response / 서치코일 전압 응답 (Faraday)

$$
V_{out}(\omega) = j\omega N A \mu_{eff} B(\omega) - R\,I(\omega) \quad\Rightarrow\quad |H(\omega)| = \frac{|V_{out}|}{|B|} = \omega N A \mu_{eff}\,\frac{1}{\sqrt{1 + (\omega/\omega_0)^2}}
$$
- N: turns of pickup winding; A: cross-sectional area; μ_eff: effective relative permeability of the saturable core
- ω_0: cutoff/resonance set by LR or LC of the coil + preamp
- Below ω_0 response rises ~ω; above ω_0 rolls off (Fig. 12 ELF/VLF and LF/MF curves; floor: ~10^-5 nT/√Hz at 1 kHz LF/MF; ~10^-3 nT/√Hz at 10 Hz ELF/VLF)
- 한국어: 권선수 N, 단면적 A, 유효 투자율 μ_eff. ω < ω_0에서 ω에 비례 상승, ω > ω_0에서 하강.

### 4.3 Fluxgate quantization / Fluxgate 양자화

For a 16-bit ADC over symmetric range ±B_R / ±B_R 대칭 범위의 16-bit ADC:
$$
\Delta B_{LSB} = \frac{2 B_R}{2^{16}}, \quad
\Delta B_{LSB}\Big|_{\pm 65{,}536\,{\rm nT}} = 2\,{\rm nT}, \quad
\Delta B_{LSB}\Big|_{\pm 1024\,{\rm nT}} \approx 0.031\,{\rm nT}
$$
Auto-ranging algorithm picks the smallest range that does not saturate, maximizing resolution. / Auto-ranging이 saturate되지 않는 최소 범위를 선택하여 분해능 최대화.

### 4.4 NY second and cadence ladder / NY 초와 cadence 사다리

$$
1\,{\rm NYsec} = \frac{2^{17}}{f_{master}} = \frac{2^{17}}{150{,}000\,{\rm Hz}} \approx 0.873813\,{\rm s}
$$
$$
{\rm DBM\ rate} = 150{,}000\,{\rm Sa/s} = \frac{2^{17}}{1\,{\rm NYsec}}
$$
$$
{\rm MAG\ rate} = \frac{256\,{\rm Sa}}{1\,{\rm NYsec}} = \frac{150{,}000}{2^9} = 292.969\,{\rm Sa/s}
$$
- All cadences are 2^k divisions of the master clock — preserves exact FFT efficiency and decimation
- 한국어: 모든 cadence는 마스터 클럭의 2^k 분수 — FFT와 데시메이션의 정확성 보장

### 4.5 Turbulence and dissipation breakpoints / 난류 break 주파수

Helios re-analysis (Sect. 1.2):
$$
f_i({\rm Hz}) \approx 4.9\, r^{-1.66}, \quad \delta B^2({\rm nT^2/Hz}) \approx 10^{8.1}\, r^{-2}, \quad r\ {\rm in}\ R_s
$$
$$
f_d \approx \frac{f_{ci}\,(v_A + v_{sw})}{v_{thi}}\quad\text{(dissipation breakpoint by k}\rho_i \sim 1)
$$
At r = 10 R_s: f_i ≈ 100 Hz, f_d ≈ 1-10 kHz / 10 R_s에서 f_i ≈ 100 Hz, f_d ≈ 1-10 kHz.

### 4.6 Quasi-thermal noise / 준-열적 노이즈 (Meyer-Vernet & Perche 1989)

$$
V^2(\omega) \propto \frac{n_e}{f^2_{pe}}\, F(\omega/\omega_{pe},\,L_{ant}/\lambda_D,\,T_e)
$$
- Spectrum peaks at the local plasma frequency f_pe = (1/2π)√(n_e e²/ε_0 m_e); shape determines T_e and the ion sound shoulder
- Used by RFS LFR to give independent absolute n_e (provides L3 calibration for E×B and SWEAP)
- 한국어: 국소 plasma 주파수에서 정점, 모양에서 T_e와 ion sound shoulder 추출. RFS LFR이 독립적 절대 n_e 제공 → L3 보정.

### 4.7 RFS spectral resolution / RFS 주파수 분해능

8-tap PFB on N=32,768 input samples → 4096-point FFT → 2048 positive bins, with frequency spacing Δf and relative resolution / 32,768 입력 샘플의 8-tap PFB → 4096-점 FFT → 2048 양주파수 빈, 주파수 간격 Δf:
$$
\Delta f_{HFR} = \frac{f_s}{2 \cdot 2048} = \frac{38.4\ {\rm MHz}}{4096} \approx 9.375\ {\rm kHz}, \quad
\Delta f_{LFR} = \frac{4.8\ {\rm MHz}}{4096} \approx 1.17\ {\rm kHz}
$$
- HFR: 9.375 kHz at f = 19.2 MHz → Δf/f ≈ 0.05%; selected log-bins give Δf/f ≈ 4.5%
- 한국어: HFR은 19.2 MHz에서 9.375 kHz 분해능 → Δf/f ≈ 0.05%; 선택 log 빈으로 Δf/f ≈ 4.5%.

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1958 ─ Parker: solar wind theory predicts supersonic outflow / 태양풍 이론 (paper #11/30 series)
1959 ─ Lunik 2: first interplanetary B measurement / 첫 행성간 자기장
1962 ─ Mariner 2: first solar-wind in situ confirmation / 첫 인-시추 확인
1966 ─ IMP-4 fluxgate magnetometer / fluxgate 자력계 시작 (heritage start, 79 instruments to 2016)
1974 ─ Helios 1 (0.29 AU perihelion, 1974-1986) / Helios 1
1981 ─ Whipple: Potentials of surfaces in space / 우주선 표면 전위 이론
1989 ─ Meyer-Vernet & Perche: QTN antenna theory / QTN 안테나 이론
1990 ─ Vaidyanathan: Polyphase filter banks / PFB 이론
1995 ─ Wind/WAVES launch, Polar/EFI launch / Wind WAVES & Polar EFI
1997 ─ Sheeley: solar-wind speed model / 태양풍 속도 모형
1999 ─ Sittler-Guhathakurta corona model / SG 코로나 모형
2006 ─ STEREO/WAVES (heritage TDS, FFT spectrometer) / STEREO WAVES
2007 ─ THEMIS launch (digital fields board heritage) / THEMIS DFB
2009 ─ Bonnell: THEMIS EFI paper / THEMIS EFI
2012 ─ Van Allen Probes launch (RBSP/EFW heritage) / RBSP EFW
2014 ─ Ergun: MMS axial double probe paper / MMS ADP
2016 ─ ★ Bale et al.: FIELDS instrument paper (this paper) / 본 논문
2018 ─ Parker Solar Probe launch (Aug 12) / PSP 발사
2019 ─ Bale et al.: switchback discovery (Nature) / switchback 발견
2021 ─ PSP enters Alfven critical surface / 알펜 임계면 진입
2024 ─ PSP final perihelion 9.86 R_s / 최종 근일점
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **Parker (1958)** "Dynamics of the interplanetary gas" | Theoretical prediction of supersonic solar wind that FIELDS will measure in situ for the first time inside the Alfvén surface / Alfvén 면 안쪽에서 FIELDS가 처음 인-시추로 검증할 초음속 태양풍 이론 | High — origin of the science problem / 과학 문제의 출발점 |
| **Bougeret et al. (1995)** "Wind/WAVES: radio and plasma wave investigation" | Heritage for FIELDS HF preamp + RFS spectral processing + PFB-FFT chain / FIELDS HF preamp + RFS 스펙트럼 처리 + PFB-FFT 사슬의 헤리티지 | High — direct hardware lineage / 직접 하드웨어 계보 |
| **Bonnell et al. (2009)** "THEMIS EFI" | Double-probe E-field architecture + spin-axis vs spin-plane sensor differentiation; current bias circuit ancestor / 이중탐침 E-field 구조와 전류 바이어스 회로 조상 | High — V1-V4 design template / V1-V4 설계 템플릿 |
| **Wygant et al. (2013)** "Van Allen Probes EFW" | Auto current-bias algorithm directly carried over to FIELDS for use in dynamic perihelion plasma / 동적 근일점 플라즈마용 자동 전류 바이어스 알고리즘을 FIELDS가 그대로 차용 | High — flight software heritage / 비행 SW 헤리티지 |
| **Cully et al. (2008)** "THEMIS Digital Fields Board" | Direct ancestor of the FIELDS DFB filter bank, BP processing, spectral matrix / FIELDS DFB 필터뱅크, BP 처리, 스펙트럼 행렬의 직계 조상 | High — DFB software/hardware lineage / DFB SW/HW 계보 |
| **Bougeret et al. (2008)** "STEREO/WAVES" | Heritage TDS architecture (16-bit, ~1 MHz Nyquist, quality-ranked downselection) / TDS 아키텍처(16-bit, ~1 MHz Nyquist, 품질 평가) 헤리티지 | High — TDS direct lineage / TDS 직계 |
| **Meyer-Vernet & Perche (1989)** "QTN antenna toolkit" | Theoretical basis for using LFR spectra to derive absolute n_e and T_e at SPP / SPP에서 LFR 스펙트럼으로 절대 n_e, T_e 도출하는 이론적 기반 | High — QTN methodology / QTN 방법론 |
| **Whipple (1981)** "Potentials of surfaces in space" | Foundation for spacecraft charging analysis driving sensor placement and bias strategy / 센서 배치와 바이어스 전략을 결정짓는 우주선 charging 해석 기반 | Medium — engineering basis / 공학 기반 |
| **Ergun et al. (2010)** "Spacecraft charging in near-Sun environment" | Predicted electrostatic barrier on the SPP heat shield → drove forward antenna placement / SPP heat shield의 전기장 장벽 예측 → 안테나 전방 배치 결정 | High — placement justification / 배치 근거 |
| **Kasper et al. (2016)** "SWEAP" (sister paper) | Companion suite providing v_sw and density that FIELDS uses for V×B subtraction in L3 / FIELDS가 L3에서 V×B 제거에 쓰는 v_sw와 밀도를 제공하는 자매 기기 | High — joint analysis chain / 합동 분석 사슬 |
| **Fox et al. (2016)** "Solar Probe Plus mission" | Mission-level overview defining science objectives that drove FIELDS Level-1 requirements / FIELDS Level-1 요구사항을 결정한 미션 단위 과학 목표 | High — top-level driver / 최상위 추진력 |
| **Bale et al. (2019)** "Switchbacks in solar wind" (Nature) | First major science result from FIELDS data — magnetic switchbacks observed at 35 R_s / FIELDS의 첫 주요 결과 — 35 R_s에서 자기장 switchback 발견 | High — flagship result / 대표 결과 |

---

## 7. References / 참고문헌

- S.D. Bale et al., "The FIELDS Instrument Suite for Solar Probe Plus", *Space Sci. Rev.* **204**, 49–82 (2016). DOI: 10.1007/s11214-016-0244-5
- B. Bavassano et al., "Radial evolution of power spectra of interplanetary Alfvénic turbulence", *J. Geophys. Res.* **87**, 3617–3622 (1982).
- R. Bruno, V. Carbone, "The solar wind as a turbulence laboratory", *Living Rev. Sol. Phys.* **2**, 4 (2005).
- J.W. Bonnell et al., "The electric field instrument (EFI) for THEMIS", *Space Sci. Rev.* **141**, 303–341 (2009).
- J.-L. Bougeret et al., "WAVES: the radio and plasma wave investigation on the Wind spacecraft", *Space Sci. Rev.* **71**, 231–263 (1995).
- J.-L. Bougeret et al., "S/WAVES: STEREO mission", *Space Sci. Rev.* **136**, 487–528 (2008).
- S.R. Cranmer, A.A. van Ballegooijen, "On the generation, propagation, and reflection of Alfvén waves from the solar photosphere to the distant heliosphere", *Astrophys. J. Suppl.* **156**, 265–290 (2005).
- S.R. Cranmer et al., "Self-consistent coronal heating and solar wind acceleration", *Astrophys. J. Suppl.* **171**, 520–551 (2007).
- C.M. Cully et al., "The THEMIS digital fields board", *Space Sci. Rev.* **141**, 343–355 (2008).
- M.M. Donegan et al., "Surface charging predictions onboard cross-scale", in *Proc. 13th Spacecraft Charging Techn. Conf.* (2014).
- T. Dudok de Wit et al., "AC magnetic field measurements onboard cross-scale", *Planet. Space Sci.* **59**, 580–584 (2011).
- R.E. Ergun et al., "Spacecraft charging and ion wake formation in the near-Sun environment", *Phys. Plasmas* **17**(7), 072903 (2010).
- R.E. Ergun et al., "Axial double probe and fields signal processing for the MMS mission", *Space Sci. Rev.* (2014). DOI: 10.1007/s11214-014-0115-x.
- N. Fox et al., "The Solar Probe Plus mission: humanity's first visit to our star", *Space Sci. Rev.* (2016). DOI: 10.1007/s11214-015-0211-6.
- S. Guillemant et al., "Solar wind plasma interaction with solar probe plus spacecraft", *Ann. Geophys.* **30**(7), 1075–1092 (2012).
- P. Harvey et al., "The electric field instrument on the Polar satellite", *Space Sci. Rev.* **71**, 583–596 (1995).
- J.C. Kasper et al., "Solar wind electrons alphas and protons (SWEAP) for SPP", *Space Sci. Rev.* (2016). DOI: 10.1007/s11214-015-0206-3.
- Y. Leblanc, "Tracing the electron density from the corona to 1 AU", *Sol. Phys.* **183**, 165–180 (1998).
- M. Loose et al., "The SIDECAR ASIC", in *Proc. SPIE* **5904** (2005). DOI: 10.1117/12.619638.
- D.M. Malaspina et al., "Interplanetary and interstellar dust observed by the Wind/WAVES electric field instrument", *Geophys. Res. Lett.* **41**, 266 (2014).
- D.M. Malaspina et al., "The digital fields board for FIELDS on Solar Probe Plus", (2016, submitted).
- R.M. Manning, G.A. Dulk, "The Galactic background radiation from 0.2 to 13.8 MHz", *Astron. Astrophys.* **372** (2001).
- D.J. McComas et al., "Integrated Science Investigation of the Sun (IS⊙IS)", *Space Sci. Rev.* (2016, this issue).
- N. Meyer-Vernet, C. Perche, "Tool kit for antennae and thermal noise near the plasma frequency", *J. Geophys. Res.* **94**, 2405–2415 (1989).
- J.C. Novaco, L.W. Brown, "Nonthermal galactic emission below 10 MHz", *Astrophys. J.* **5**, 221 (1978).
- H.C. Seran, P. Fergeau, "An optimized low-frequency three-axis search coil magnetometer for space research", *Rev. Sci. Instrum.* **76**, 044502 (2005).
- N.R. Sheeley et al., "Measurements of flow speeds in the corona between 2 and 30 R_s", *Astrophys. J.* **484**, 472 (1997).
- E.C. Sittler Jr., M. Guhathakurta, "Semiempirical two-dimensional MHD model of the solar corona", *Astrophys. J.* **523**, 812–826 (1999).
- P.P. Vaidyanathan, "Multirate digital filters, filter banks, polyphase networks, and applications", *Proc. IEEE* **78**, 56–93 (1990).
- A. Verdini, M. Velli, "Alfvén waves and turbulence in the solar atmosphere and solar wind", *Astrophys. J.* **662**, 669–676 (2007).
- A. Verdini et al., "A turbulence-driven model for heating and acceleration of the fast wind in coronal holes", *Astrophys. J. Lett.* **708**, L116–L120 (2010).
- A. Vourlidas et al., "The wide-field imager for Solar Probe Plus (WISPR)", *Space Sci. Rev.* (2016). DOI: 10.1007/s11214-014-0114-y.
- E. Whipple, "Potentials of surfaces in space", *Rep. Prog. Phys.* **44**, 1197–1250 (1981).
- J.R. Wygant et al., "The electric field and waves instruments on the radiation belt storm probes mission", *Space Sci. Rev.* **179**, 183–220 (2013).
- A. Zaslavsky et al., "Interplanetary dust detection by radio antennas: mass calibration and fluxes measured by STEREO/WAVES", *J. Geophys. Res.* **117**, 5102 (2012).
