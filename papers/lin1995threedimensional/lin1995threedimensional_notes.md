---
title: "A Three-Dimensional Plasma and Energetic Particle Investigation for the Wind Spacecraft"
authors: [R. P. Lin, K. A. Anderson, S. Ashford, C. Carlson, D. Curtis, R. Ergun, D. Larson, J. McFadden, M. McCarthy, G. K. Parks, H. Rème, J. M. Bosqued, J. Coutelier, F. Cotin, C. d'Uston, K.-P. Wenzel, T. R. Sanderson, J. Henrion, J. C. Ronnet, G. Paschmann]
year: 1995
journal: "Space Science Reviews"
doi: "10.1007/BF00751328"
topic: Space_Weather
tags: [wind_spacecraft, instrumentation, suprathermal_electrons, top_hat_esa, sst, fast_particle_correlator, isep, gss, pad]
status: completed
date_started: 2026-04-25
date_completed: 2026-04-25
---

# 63. A Three-Dimensional Plasma and Energetic Particle Investigation for the Wind Spacecraft / Wind 우주선의 3차원 플라스마 및 고에너지 입자 관측 실험

---

## 1. Core Contribution / 핵심 기여

이 논문은 NASA-ESA의 GGS (Global Geospace Science) 프로그램의 Wind 우주선에 탑재된 3DP (3D Plasma and Energetic Particles) 관측기의 설계 / 측정 원리 / 운용 방식을 종합적으로 기술한 instrument paper이다. 3DP는 단일 패키지 안에 (i) **EESA-L/H** (3 eV–30 keV 전자, top-hat 대칭 quadrispherical ESA, 두 가지 sensitivity), (ii) **PESA-L/H** (3 eV–30 keV 이온, 동일 광학), (iii) **SST** (semiconductor 망원경 어레이, 25 keV–11 MeV의 전자/이온, foil-side는 전자 magnet-side는 이온), (iv) **FPC** (Fast Particle Correlator: WAVE 실험 전기장과 EESA-H 전자 카운트의 직접 상관) 4개 서브시스템을 통합하여, 태양풍 코어부터 저에너지 우주선까지 ~9 decade의 동적 범위와 4π sr × ~3 s의 시간/각 분해능으로 분포함수와 그 모멘트(밀도, 속도, 압력 텐서, 열속), 피치각 분포(PAD)를 한꺼번에 측정한다. 1990년대 ISEE-3 세대가 부족했던 통계, 시간/각 분해능, 동적 범위를 모두 끌어올린 본격적 suprathermal in-situ 측정 표준 도구로서의 청사진이다.

This is the comprehensive instrument paper for the 3D Plasma and Energetic Particles (3DP) experiment on the NASA/ESA GGS Wind spacecraft. 3DP unifies four detector subsystems in one package: (i) **EESA-L/H** electron electrostatic analyzers (3 eV–30 keV) using Carlson-type symmetric quadrispherical (top-hat) optics, with low- and high-sensitivity pairs to cover the enormous dynamic range; (ii) **PESA-L/H** ion analyzers with the same optics; (iii) the **SST** triplet silicon-detector telescopes (25 keV–~11 MeV) using a Lexan foil at one end (clean electrons) and a broom magnet at the other (clean ions); and (iv) the **Fast Particle Correlator (FPC)**, which directly correlates EESA-H electron counts with the WAVE-experiment electric field in quadrature. Together they deliver full 4π sr × 3-s 3-D distribution functions, onboard moments (density, velocity, pressure tensor, heat flux), and pitch-angle distributions across ~9 decades of differential flux — a quantum leap beyond ISEE-3 in sensitivity, angular resolution, dynamic range, and time resolution, and the de-facto blueprint for suprathermal-particle instrumentation that followed (Cluster/PEACE, THEMIS/ESA, MMS/FPI, PSP/SWEAP, Solar Orbiter/SWA).

---

## 2. Reading Notes / 읽기 노트

### Part I. Introduction (§1, p. 126) / 도입

**한국어**: 저자들은 "suprathermal" 영역을 수 eV에서 ~수백 keV 사이로 정의하고, 이 영역이 태양풍, 행성간 충격파, type III 라디오 버스트, foreshock, 자기권 입자 입출력 같은 대부분의 우주 플라스마 현상의 주역임을 강조한다. ISEE-3가 ~2 keV 전자에서 새로운 현상들을 잇따라 발견했지만 dynamic range / 통계 / 분해능이 부족해 정량 분석에 한계가 있었다. 이 실험의 5대 과학 목표는 (a) 행성간 suprathermal 입자의 첫 정밀 탐사, (b) 태양/IPM/upstream 가속 연구, (c) suprathermal transport, (d) 파-입자 상호작용·충격파·heat flux 같은 plasma processes, (e) 자기권 입출력 측정이다.

**English**: The authors define the suprathermal regime as ~few eV–several hundred keV, and argue this is the regime where most of the interesting physics — solar wind, IP shocks, type III bursts, foreshock, magnetospheric I/O — actually lives. ISEE-3 had opened the regime but with poor statistics, time resolution, and angular resolution. The five science goals are: (a) the first detailed exploration of the IP suprathermal regime, (b) acceleration at the Sun, IPM, and upstream, (c) suprathermal transport, (d) basic plasma processes (wave–particle, type III, shocks, soliton collapse, heat flux), and (e) magnetospheric input/output.

### Part II. Scientific Objectives (§2.1–§2.8, pp. 126–132) / 과학 목표

저자들은 8개의 미해결 과학 의문을 카탈로그처럼 나열한다. 이는 단순한 동기 부여가 아니라 instrument design driver이다.

| § | Topic | 1995년의 핵심 질문 / Open question | 어떤 측정이 필요한가 / Required measurement |
|---|---|---|---|
| 2.1 | Suprathermal electrons | quiet-time 2–20 keV 전자의 기원? scatter-free→diffusive 전이의 원인? | 고감도 PAD, ~수 eV–수백 keV 연속 스펙트럼, 빠른 시간분해 |
| 2.2 | Solar acceleration/storage | flash phase 가속 메커니즘? 한 메커니즘인가 둘인가? | 100 eV–10 MeV 시간프로파일, 스펙트럼, 각분포 |
| 2.3 | Langmuir waves & type III | bump-on-tail이 어떻게 wave를 만드나? soliton collapse가 일어나나? | $f(v_\parallel)$ 의 수십 ms 시간분해, wave와 직접 상관 |
| 2.4 | Suprathermal ions <40 keV | quiet-time 이온 스펙트럼? CIR/storm/shock 이온의 기원? | 첫 quiet-time 측정; 0.1–1 cm² sr의 큰 G |
| 2.5 | IP shocks & bow shock | 충격파 가속의 미시 메커니즘? 입자가 충격파를 들락거리는 트레이싱? | 빠른 3D 분포로 remote-sense |
| 2.6 | Upstream ions/electrons | foreshock beam의 원인? wave 생성 vs. heat flux 전자와의 관계? | 이중 lunar swingby 동안 수십 회 foreshock 통과 |
| 2.7 | Magnetospheric entry | IMF-자기권 연결층의 물리? | Wind/Geotail/Polar 다중 위치 비교 |
| 2.8 | Magnetospheric bursts in IPM | ≳0.3 MeV 자기권 기원 입자가 어떻게 IPM으로 새나오는가? | 정밀 3D 분포 |

이 8개 의문은 곧 instrument의 측정 요구로 직결된다: 9 decade dynamic range (그래서 -L/-H sensitivity 분리), ~ms 시간분해 (FPC의 burst memory), full 4π PAD (3 s에 한 spin), 깨끗한 전자/이온 분리 (SST의 foil/magnet).

These eight open questions translate directly into instrument requirements: the ~9-decade dynamic range from solar wind to "interplanetary quiet" (justifying separate -L and -H sensitivities), ms-cadence sampling (justifying the FPC and 2-MB burst memory), full 4π PADs once per 3-s spin, and clean electron/ion separation (foil/magnet pair in SST).

### Part III. Instrument Overview (§3, p. 133) / 기기 개요

전체 시스템은 세 detector system + 주 처리장치(DPU)로 구성된다.

```
                +-------------------+
                |  S/C C&DH bus     |
                +--------+----------+
                         |
                  +------v------+    +-------+    +----------+
                  |  MAIN DPU   |<-->| EESA  |<-->|  PESA-   |
                  | (3 micro-   |    |  CPU  |    |  /EESA-  |
                  |  processors)|    +-------+    |  L,H ESA |
                  | + SST proc  |                 +----------+
                  +------+------+
                         |
                    +----v----+
                    |  SST    |  Foil-end (electrons 25-400 keV)
                    | analog  |  Magnet-end (ions 20 keV-6 MeV)
                    +---------+
                         |
                    +----v----+
                    |   FPC   |  EESA-H + WAVE E-field
                    +---------+
```

**한국어**: 두 boomlet (각 0.5 m)에 EESA-L/H가 한쪽, PESA-L/H + SST가 반대쪽에 위치해 우주선 표면 효과를 최소화한다. EESA-H의 360° FOV는 ±45° 정전 디플렉터로 자기장 방향을 따라가도록 조향 가능 — 이는 type III 동안 field-aligned 전자 측정을 늘리기 위한 설계이다. Main DPU는 모멘트(밀도, 3성분 속도, 6성분 압력 텐서)와 PAD를 onboard에서 계산해 telemetry 부담을 줄인다.

**English**: EESA-L/H sit on one 0.5-m boomlet; PESA-L/H + SST sit on an opposing 0.5-m boomlet — to minimize spacecraft-potential effects on the very low-energy plasma. EESA-H's 360° in-plane FOV can be electrostatically deflected ±45° to track the magnetic field, increasing field-aligned electron statistics during type III bursts. Three onboard microprocessors (main, EESA, PESA) compute moments (density, 3-comp velocity, 6 unique components of pressure tensor, heat flux) and pitch-angle distributions in real time, reducing telemetry load.

### Part IV. Electrostatic Analyzers (§4, pp. 133–142) / 정전 분석기

#### 4.1 Top-hat symmetric quadrispherical optics / Top-hat 대칭 광학

**원리 / Principle**: 두 동심 반구(반지름 $R_1, R_2$, $\Delta R = R_2 - R_1$) 사이를 통과하는 입자는 안쪽 반구의 음의 전위가 만드는 $E$-field에 의해 곡률 반경에 가까운 궤도로 휘어 분석기를 통과한다. 통과 조건은
$$ \frac{E}{q} \;\approx\; \frac{R_1 R_2}{R_2^2 - R_1^2}\, V \;\approx\; \frac{R_0}{2\Delta R}\, V, $$
여기서 $V$는 안쪽 반구에 인가된 전위. EESA-H/PESA-H: $R_1 = 8.0$ cm, plate separation $0.6$ cm, top-cap separation $1.2$ cm, entrance opening half-angle $19°$. PESA-L/EESA-L: $R_1 = 3.75$ cm, separation $0.28$ cm, top-cap $0.56$ cm.

이 모든 분석기는 $\Delta E/E = 0.20$ FWHM, 각 수용범위 $\pm 7°$ FWHM를 갖는다. 이는 Carlson et al. (1983)의 "symmetric spherical-section analyzer" 설계로 360°(또는 180°) 디스크형 시야와 ~1° 본질 분해능(평행광선의 detector plane focusing)을 동시에 제공한다.

**English**: The hemisphere voltage selects $E/q$. Both -H analyzers ($R_1 = 8.0$ cm, $\Delta R = 0.6$ cm, $19°$ entrance half-angle) and -L analyzers ($R_1 = 3.75$ cm, $\Delta R = 0.28$ cm) achieve $\Delta E/E = 0.20$ FWHM and $\pm 7°$ angular acceptance. The Carlson 1983 quadrispherical design focuses parallel rays onto a position-sensitive MCP detector plane, giving ~1° intrinsic angular resolution while preserving a 360°/180° disk-shaped FOV.

#### 4.2 Geometric factors and dynamic range / Geometric factor와 동적 범위

| 분석기 | $G$ (cm² sr) | FOV (°) | Dynamic range (eV/(cm²·s·sr·eV)⁻¹) |
|---|---|---|---|
| EESA-H/FPC | $0.1\, E$ | 360 × 90 | ~1–10⁸ |
| EESA-L | $1.3\times 10^{-2}\, E$ | 180 × 14 | ~10²–10⁹ |
| PESA-H | $1.5\times 10^{-2}\, E$ | 360 × 14 | ~10–10⁹ |
| PESA-L | $1.6\times 10^{-4}\, E$ | 180 × 14 | ~10⁴–10¹¹ |

**해석 / Interpretation**: 작은 $G$의 -L 분석기는 강력한 태양풍 코어/halo와 강한 솔라 플레어 이벤트의 saturation을 막고, 큰 $G$의 -H 분석기는 quiet-time suprathermal에서 의미 있는 통계를 모은다. PESA-L의 $G$가 매우 작은 이유는 직경 0.24 mm, 1×2.25 mm² 간격의 collimator/attenuator로 50배 감쇠시켜 태양풍 본 분포의 바로 측정을 가능케 했기 때문이다.

**English**: The small-$G$ -L analyzers prevent saturation from the dense solar wind and large flares; the large-$G$ -H units provide statistics for the rarefied quiet-time suprathermal flux. PESA-L's tiny $G$ comes from a 50×-attenuating collimator (0.24-mm holes, 1×2.25 mm² spacing) that lets it sample the solar-wind core itself directly.

#### 4.3 Detector implementation / 검출기 구현

MCPs는 chevron pair (8° bias angle, 1 mm thick, gain ~$2\times 10^6$). EESA-L/PESA-L: 단일 180° half-ring; EESA-H: 6개 60° sector; PESA-H: 두 set의 half-ring으로 360° ring. **anode segmentation**:
- EESA-H/PESA-H: 24 anodes — 황도 ±22.5° 안에서 5.6° 분해능, 11.25° (±45°), 22.5° (그 너머).
- EESA-L/PESA-L: 16 anodes — 5.6° 분해능 (적도면).

전자는 +500 V (EESA-L) 또는 더 높은 전위로 MCP에 post-accelerate하여 ~70% 검출 효율; 이온은 -2500 V로 PESA-H에서 post-accelerate. PESA-H/EESA-H에는 L자형 plastic anti-coincidence scintillator가 MCP를 둘러싸 penetrating particle background를 거부.

**English**: All ESAs use chevron MCP pairs (gain ~$2\times 10^6$). Anode segmentation: 24 anodes for -H (5.6° within ±22.5° of ecliptic, 11.25° to ±45°, 22.5° beyond); 16 anodes for -L (5.6°). Electrons are post-accelerated to ~70% MCP efficiency; ions in PESA-H to ~50%. -H units use an L-shaped plastic anti-coincidence scintillator around the MCP to reject penetrating background.

#### 4.4 Sweep and sampling cadence / 스윕과 샘플링

분석기 전압은 로그 스윕, 카운터는 스핀당 1024회(약 3 ms 간격) 샘플. 대표적 solar-wind 운영 모드:
1. EESA-L: 3–300 eV, 32 step, 32 sweep/spin → 11.25° spin phase 분해능, 황도면에서 gap 없음.
2. EESA-H: 300 eV–30 keV, 32 step, 16 sweep/spin → 11.25° spin phase 분해 (type III burst 측정용).
3. PESA-L: 100 eV–10 keV, 16 step, 64 sweep/spin → 5.6° spin phase 분해 (반사 이온 빔용).
4. PESA-H: 3 eV–30 keV, 32 step, 32 sweep/spin → 11.25° spin phase 분해 (suprathermal ion).

**English**: Voltages are swept logarithmically; counters sample 1024 times per spin (~3 ms each). The representative solar-wind mode uses 32 energy samples × 16–64 sweeps/spin, giving 5.6°–11.25° spin-phase resolution.

### Part V. Solid State Telescopes (§5, pp. 142–145) / 반도체 검출기 망원경

#### 5.1 Geometry / 기하

3개 어레이, 각 어레이는 한 쌍의 double-ended telescope. 따라서 6 foil-side (F) detectors + 6 magnet-side (O) detectors. 각 망원경은 (front, center, back) triplet (1.5 cm² 면적; F/O = 300 µm, T = 500 µm). 양 끝의 한쪽은 Lexan **foil**로 ~400 keV 이하 양성자를 정지 (전자는 거의 그대로 통과), 반대편은 broom **magnet**으로 ~400 keV 이하 전자를 휘게 하여 검출기를 비껴가게 함 (이온은 영향 없음).

**Detector logic / 검출기 논리**:
- F (front foil) only: 전자 25–400 keV (anti-coincidence with center).
- O (front magnet) only: 이온 20 keV–6 MeV.
- FT (front+center foil, coincidence): 전자 ~400 keV–~1 MeV.
- OT (front+center magnet, coincidence): 양성자 6–11 MeV.
- 가까이 sandwich (≤100 µm 간격) → penetrating 입자는 거의 모두 anti-coincidence로 거부.

#### 5.2 4π coverage / 4π 커버리지

5개 telescope이 180° × 20° (foil-side는 180° × 20°, magnet-side는 magnet으로 동일 슬라이스) FOV를 만든다. spin 360°가 완성되면 4π sr 전체가 sweep된다. Telescope 6 (F₆T₆)은 spin axis 방향과 같은 각도; telescope 2 (F₂T₂, O₂T₂)는 가장 강한 flux를 위해 drilled tantalum cover로 $G$를 1/10로 감쇠.

```
   Spin axis (Z)
        |
        | 18°
        v          54°
   +----+-----------+
   |    O3,F3       |
   |    O4,F6T6,F2T2|
   |    O2T2,F4,O6T6|
   |        F5      |
   |        O5      |
   +----------------+
   (FWHM 36° wide, FOV centerlines per Fig. 10)
```

각 36° × 20° FWHM, $G \approx 0.33$ cm² sr per detector (single-end), telescope 2 reduced to 0.03 cm² sr.

#### 5.3 Calibration / 캘리브레이션

비행 중 캘리브레이션은 (i) F, O, T detector에 ramp pulser로 500 Hz 신호 주입, (ii) penetrating particle minimum-ionizing energy를 두께만에 의존하는 절대 기준으로 사용, (iii) 양성자가 검출기에서 정지하는 highest energy를 모니터링.

**English**: Each detector is fed a ramp pulser at 500 Hz for in-flight gain calibration; absolute energy is anchored by minimum-ionizing-energy (depends only on detector thickness) and by the maximum-stopping-proton energy.

### Part VI. Digital Electronics (§6, pp. 145–147) / 디지털 전자 시스템

3개 마이크로프로세서: **Main CPU** (S/C C&DH, SST, FPC supervision, 2 MB burst memory), **EESA CPU** (EESA-L/H 제어 + FPC 데이터), **PESA CPU**. 각각 별도 인터페이스로 격리되어 있어 한 시스템 고장이 다른 시스템을 마비시키지 않는다. 모든 프로세서는 in-flight 재프로그래밍 가능. 데이터는 Main CPU에서 telemetry rate에 맞게 spin 평균 횟수를 자동 조정.

**English**: Three microprocessors (Main, EESA CPU, PESA CPU) supervise their own subsystem. The Main CPU also operates the SST and the FPC, and houses a 2-MB burst memory for high-cadence event capture. Onboard moments and PAD computation reduce telemetry; the system auto-adapts to changing telemetry rates by averaging more spins per data product.

### Part VII. Experiment Modes (§7, pp. 147–148) / 운용 모드

기본 'Solar Wind' 모드 외에 텔레메트리 테이블과 program up-link로 변경. SST는 spin당 16회 (16-channel F, 24-channel O/OT/FT) 데이터 수집. 텔레메트리 product는 (a) 전 detector full energy spectrum, (b) 40-direction 3-D distribution (F는 7 energies, O/OT는 9 energies), (c) 자기장 방향 기준 PAD (magnetometer 입력), (d) ESA의 모멘트(밀도/속도/압력 텐서/heat flux), (e) high-resolution snapshot. **Burst memory**는 트리거 이벤트(예: type III burst, shock) 발생 시 모든 분석기의 high-cadence 3D 분포를 2 MB ring buffer에 저장 → 천천히 telemetry로 재생.

**English**: Default mode auto-adjusts spin averaging to telemetry. SST collects 16 samples per spin × 16/24 channels. The 2-MB burst memory captures triggered events (type III, shocks) at maximum cadence, then plays them back as telemetry permits, enabling event-driven high-resolution science despite limited average bandwidth.

### Part VIII. Fast Particle Correlator (§8, pp. 148–151) / 고속 입자 상관기

#### 8.1 Direct wave-particle correlation / 직접 파-입자 상관

상관 함수
$$ C(v, \Theta) = \frac{\displaystyle\int E_0 \sin(kx - \omega t + \Theta)\, F(v,t)\, dt}{\langle E^2\rangle_t^{1/2}\, \langle F^2\rangle_t^{1/2}} $$

WAVE 실험의 $E_x, E_y$ 안테나 중 자기장과 가장 평행한 것을 선택, 4가지 high-pass + 4가지 low-pass filter 조합으로 12–50 kHz Langmuir 대역 분리. Phase splitter는 5–125 kHz 범위에서 90° ± 3°의 SIN/COS 신호를 만들고, 각각 비교기로 1-bit 디지털화. 카운터 게이트가 SIN 또는 COS의 양극일 때만 EESA-H 카운트를 누적, ~3 ms 적분 후 phase shift는 baseline restorer로 보정. SIN/COS 두 채널의 결합으로
- 진폭 $A = \sqrt{C_S^2 + C_C^2}$
- 위상 $\phi = -\arctan(C_S / C_C)$.

이는 Langmuir wave가 전자 분포 $f(v_\parallel)$에 만드는 bunching의 진폭과 위상을 in situ로 측정.

#### 8.2 Auto- and burst correlation / 자기·버스트 상관

**Auto-correlation**: Gough (1985) 방식 — EESA-H 입자 이벤트의 inter-arrival time histogram을 ~1–2분 누적, 1.6 µs 비닝. Poisson에서 벗어남이 곧 bunching 주기. Phase 정보는 없지만 long coherence 신호에 더 민감.

**Burst correlation**: 625 kHz 1.6 µs sample stream을 1024-bit 연속 burst로 기록 (3 s 이내, 2-MB 메모리), 후속 Fourier 변환으로 스펙트럼화. 평균 sensitivity는 auto보다 좋다.

#### 8.3 Statistical requirements / 통계적 요구

분포의 1% 변화를 $3\sigma$로 검출하려면 $N \gtrsim 9 \times 10^4$ counts/bin. Langmuir wave가 ~수백 ms 지속이므로 ~1 MHz count rate 필요 → EESA-H의 $G_0 = 0.1\, E$ 와 16 cm 직경(velocity acceptance width)이 요구되었다.

### Part IX. Summary (§9, p. 152) / 요약

WIND/3DP는 태양풍부터 저에너지 우주선까지의 plasma environment를 광범위한 에너지/각/시간 분해능으로 측정하기 위해 설계되었다. 다운링크 한계 때문에 substantial한 데이터 선택/압축이 필요하고, burst memory + FPC가 이를 보완한다. 정상 운용에서 (1) 태양풍 이온 속도/온도/밀도, (2) 태양풍 전자 속도/온도/밀도, (3) 4개 에너지 대역의 suprathermal 전자/이온 flux를 GGS 커뮤니티에 직접 제공한다.

### Part X. Worked Example: Identifying a type III burst electron beam / 풀이 예제: type III 버스트 전자 빔 식별

**시나리오 / Scenario**: Wind/3DP가 1996년에 어떤 type III burst를 관측했다고 가정. 다음과 같은 시간 순서가 일어난다.

1. **t = 0**: WAVE 실험이 ~80 kHz에서 라디오 방출 시작을 기록. 주파수가 시간에 따라 빠르게 감소 ($df/dt \sim -10$ kHz/s).
2. **t ≈ 30 s**: 라디오 방출이 ~30 kHz로 떨어졌을 때 (지역 plasma frequency $f_{pe} \approx 30$ kHz at 1 AU), Langmuir wave가 in situ로 검출됨. 전자 빔이 우주선에 도달했다는 의미.
3. **t = 30 s 직후**: EESA-H가 anti-sunward (자기장 평행) 방향으로 좁은 PAD ($\Delta \alpha < 30°$)의 ~5–15 keV 전자 enhancement를 측정. PESA-H의 ion flux는 변화 없음.
4. **t = 30 s 동시간**: FPC가 $\delta f / f_0 \sim 0.05$ 의 전자 bunching을 ~30 kHz wave 위상에 잠긴 채로 측정.

**3DP 데이터로 본 동일 사건 / Same event in 3DP data**:
- EESA-H 32-step sweep에서 $E$ = 5 keV 채널의 anti-sunward 90° 부근 anode (자기장 방향) count rate이 ~10⁵ Hz로 상승 ($G = 0.1\cdot 5000 \cdot 10^7 \cdot 0.20 \approx 10^9$/s가 saturation 한계임을 확인).
- PAD CPU가 $\hat{B}$ 방향 PAD를 onboard에서 계산해 telemetry로 보냄 → ground에서 즉시 빔 검출.
- 트리거 조건이 충족되면 burst memory가 모든 분석기의 ms-cadence 3D 분포를 2 MB ring buffer에 저장 → 이후 ~수십 분에 걸쳐 천천히 telemetry로 재생.
- FPC의 SIN/COS 카운트로 $C_S, C_C$ 측정 → 진폭 $A = \sqrt{C_S^2 + C_C^2}$, 위상 $\phi = -\arctan(C_S/C_C)$. 이 위상이 wave의 $E$-field 위상과 어떤 관계인지로 quasilinear vs. strong-turbulence 식별.

**핵심 메시지 / Key message**: 단일 instrument가 (a) 빔의 에너지 스펙트럼(EESA-H), (b) PAD (onboard rebinning), (c) wave-particle bunching (FPC)을 동시에 제공한다. ISEE-3 시대에는 이 셋을 동시에 정량화할 방법이 없었다 / No previous mission could simultaneously deliver beam energy spectrum, pitch-angle distribution, and wave-particle bunching with this resolution; 3DP makes a single-event quasilinear test feasible.

### Part XI. Quantitative Performance Numbers / 정량적 성능 수치

논문 본문과 Table I에서 추출한 핵심 숫자들을 한 표로 정리. 이는 instrument 비교 paper나 reanalysis에서 가장 자주 인용되는 값들이다 / The following numbers, scattered through the paper and Table I, are the most cited specs in subsequent reanalyses.

| 항목 / Item | 수치 / Value | 출처 / Source |
|---|---|---|
| 우주선 spin period | ~3 s | §1, Table I |
| 분석기 sample rate | 1024 samples/spin (~3 ms 간격) | §4, p. 141 |
| EESA-L energy range | 3 eV – 30 keV | Table I |
| EESA-H energy range | 100 eV – 30 keV (FPC 모드는 더 좁게) | Table I |
| PESA-L energy range | 3 eV – 30 keV | Table I |
| PESA-H energy range | 3 eV – 30 keV | Table I |
| SST F (foil) electron range | 25–400 keV | §5, p. 142 |
| SST FT coincidence electron range | ~400 keV – ~1 MeV | §5, p. 143 |
| SST O (magnet) ion range | 20 keV – 6 MeV | Table I, §5 |
| SST OT coincidence proton range | 6–11 MeV | Table I |
| EESA-H geometric factor | $0.1\, E$ cm² sr | Table I |
| EESA-L geometric factor | $1.3\times 10^{-2}\, E$ cm² sr | Table I |
| PESA-H geometric factor | $1.5\times 10^{-2}\, E$ cm² sr | Table I |
| PESA-L geometric factor | $1.6\times 10^{-4}\, E$ cm² sr | Table I |
| PESA-L attenuator factor | 50× | §4, p. 140 |
| SST geometric factor / detector | 0.33 cm² sr (telescope 2: 0.03) | Table I, §5 |
| ESA energy resolution $\Delta E/E$ | 0.20 FWHM | §4 |
| ESA angular acceptance | $\pm 7°$ FWHM | §4 |
| SST angular FWHM | 36° × 20° | §5 |
| Spin-phase angular resolution | 5.6° (within ±22.5° of ecliptic), 11.25°, 22.5° | Table I, §4 |
| EESA-H deflection range | $\pm 45°$ out of plane | §4 |
| Inner hemisphere radius (EESA-H/PESA-H) | 8.0 cm | §4 |
| Plate separation (EESA-H/PESA-H) | 0.6 cm | §4 |
| Inner hemisphere radius (-L) | 3.75 cm | §4 |
| Plate separation (-L) | 0.28 cm | §4 |
| MCP gain | $\sim 2 \times 10^6$ | §4 |
| Number of anodes (EESA-H/PESA-H) | 24 | §4 |
| Number of anodes (-L) | 16 | §4 |
| Ion post-acceleration (PESA-H) | $-2500$ V | §4 |
| Electron post-acceleration (EESA-L) | $+500$ V | §4 |
| Total instrument mass | 18.2 kg | Table I |
| Total power | 15.6 W | Table I |
| Bit rate (nominal / inside 60 R_E) | 1035 / 2070 bps | Table I |
| Burst memory size | 2 MB | §6 |
| FPC frequency band | 12–50 kHz (Langmuir) | §8 |
| FPC integration period | ~3 ms | §8 |
| FPC duty-cycle accuracy | $50.0 \pm 1.0\%$ | §8 |
| FPC required count rate (1% to $3\sigma$) | ~1 MHz | §8.3 |
| FPC required N | $\geq 9 \times 10^4$ counts | §8.3 |

이 수치들이 후속 Wind/3DP 논문(Lin 1996, Larson 1996, Krucker 1999 등)의 데이터 reduction 단계에서 그대로 사용된다 / These numbers are used directly in the data-reduction pipelines of all subsequent Wind/3DP science papers.

---

## 3. Key Takeaways / 핵심 시사점

1. **Top-hat 대칭 quadrispherical 광학은 고감도 + 360° FOV의 동시 달성을 가능케 한다 / The top-hat symmetric quadrispherical optic is the key enabler of simultaneous high sensitivity and 360° in-plane FOV** — Carlson 1983의 spherical-section 설계를 채택함으로써, EESA-H는 $G = 0.1\, E$ cm² sr (PESA-L의 600배)와 함께 ±22.5° 안에서 5.6° 분해능을 동시에 갖는다. 이전 세대의 곡면 거울 분석기로는 둘 중 하나만 가능했다 / Achieving both at once was infeasible with prior curved-mirror designs, but the spherical-section optic focuses parallel rays onto an MCP plane, decoupling sensitivity from angular resolution.

2. **두 sensitivity (-L vs. -H) 분리가 9 decade 동적 범위를 만든다 / Two-sensitivity (-L vs. -H) separation provides the ~9-decade dynamic range** — PESA-L($G \sim 1.6\times 10^{-4}E$)이 ~10¹¹의 태양풍 코어를 saturation 없이 측정하고, EESA-H($G \sim 0.1E$)가 quiet-time suprathermal의 ~1 sccount까지 통계를 잡는다. 단일 분석기로는 MCP가 saturate되거나 통계가 부족해 둘 다 잃는다 / A single analyzer either saturates on the solar wind or starves on the quiet-time tail; splitting into two sensitivities (with a 50× collimator-attenuator on PESA-L) covers both extremes.

3. **SST의 foil/magnet 이중 콜리메이터는 species 분리의 우아한 해법이다 / SST's foil-magnet double-end is an elegant species discriminator** — Lexan foil이 ~400 keV 양성자를 정지시키지만 동일 에너지의 전자는 거의 그대로 통과시키고, broom magnet은 <400 keV 전자를 휘게 하지만 이온에는 영향이 없다. 단일 텔레스코프 안에서 두 종을 동시에, 깨끗이 측정 / The Lexan foil stops ions ≤400 keV with negligible electron loss; the broom magnet sweeps electrons <400 keV with no ion deflection. Same telescope body, two species, clean separation.

4. **Anti-coincidence triplet 구조가 penetrating background를 본질적으로 제거한다 / The triplet anti-coincidence structure essentially eliminates penetrating-particle background** — front/center/back 검출기의 ≤100 µm 간격 sandwich로, 모든 검출기를 통과하는 입자는 모두 거부된다. 우주선 표면에서 발생하는 secondary와 cosmic-ray background를 ~10⁻⁶ counts level까지 줄여 quiet-time 측정을 가능케 함 / The triplet is a 3-fold veto: any particle reaching the back detector is rejected. This drives backgrounds below the quiet-time signal level — without it, suprathermal-tail measurements are simply impossible.

5. **Onboard 모멘트와 PAD 계산은 텔레메트리 한계의 핵심 우회로 / Onboard moments and PAD computation are the key bypass to telemetry limits** — 풀 3D 분포(80 anode × 32 energy × 16 spin = ~40000 numbers)를 매 spin 보내는 것은 1035 bps의 bit budget으로 불가능. EESA/PESA CPU가 magnetometer 자료를 받아 onboard에서 모멘트(10개)와 1D PAD를 계산해 보냄 / Full 3D distributions per spin (80 anodes × 32 energies × 16 sweeps) far exceed the 1035 bps bandwidth. The two analyzer CPUs ingest magnetometer data and compute moments + PADs on board, dramatically reducing downlink load.

6. **Burst memory + FPC는 "burst-driven 과학"의 청사진 / Burst memory + FPC define burst-driven science** — type III, shock crossing 같은 transient에서 ms-cadence가 필요한데 평균 다운링크는 sub-Hz. 2 MB ring buffer가 트리거 시 high-cadence 데이터를 capture하고 천천히 재생하는 패러다임은 이후 MMS, PSP, Solar Orbiter의 모든 burst-mode 관측의 표준이 된다 / Average downlink is sub-Hz, but transient phenomena need ms cadence. The 2-MB ring-buffer + trigger paradigm becomes the universal template for all later burst-mode space-plasma instruments.

7. **FPC의 직접 quadrature 상관은 in-situ wave-particle 진단의 첫 본격 구현 / FPC's direct quadrature correlation is the first serious in-situ wave-particle diagnostic** — wave 전기장과 전자 카운트의 SIN/COS 상관으로 진폭과 위상을 동시에 측정. 이는 이후 MMS FPI+EDP의 디지털 후처리 방식으로 재현되며, Langmuir wave의 quasilinear 이론 검증에 결정적 데이터를 제공 / The FPC quadrature correlation provides simultaneous amplitude and phase of electron bunching at the wave frequency. Reincarnated as digital post-processing on MMS, this technique is the empirical foundation for testing quasilinear / strong-turbulence theories.

8. **8개 미해결 과학 의문이 곧 instrument design driver / The eight open science questions of §2 are themselves the design drivers** — quiet-time 2–20 keV 전자 (→ 큰 G), scatter-free vs. diffusive 전이 (→ 빠른 PAD), bump-on-tail (→ FPC), suprathermal ion <40 keV (→ 큰 PESA-H G), shock 가속 추적 (→ ms 시간분해), foreshock (→ lunar swingby 궤도), 자기권 입출력 (→ 광범위 에너지) — 각 요구사항이 직접 specific 부품에 매핑된다 / Each open question maps directly to a specific component requirement: quiet-time electrons → high $G$, scatter-free→diffusive transition → fast PADs, bump-on-tail → FPC, sub-40-keV ions → high PESA-H $G$, shocks → ms cadence, foreshock → lunar-swingby orbit. Science requirements drive hardware in a transparent traceable way.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 ESA optical equation / 정전 분석기 광학 방정식

두 동심 반구 사이 $E$-field에서의 등에너지 궤도:
$$\boxed{\;\frac{E}{q} \;=\; k\, V, \qquad k \;\approx\; \frac{R_1 R_2}{R_2^2 - R_1^2} \;\approx\; \frac{R_0}{2\,\Delta R}\;}$$

EESA-H/PESA-H 수치: $R_1 = 8.0$ cm, $R_2 = 8.6$ cm, $\Delta R = 0.6$ cm, $R_0 \approx 8.3$ cm → $k \approx 6.9$. EESA-L/PESA-L: $R_1 = 3.75$ cm, $\Delta R = 0.28$ cm → $k \approx 6.7$. 즉 $V$가 ~150 V일 때 ~1 keV 전자/이온이 통과. 이 비례 계수가 모든 분석기에서 거의 같게 설계되어 동일 $\Delta E/E = 0.20$ FWHM과 ±7° 각수용을 보장 / The fact that all four ESAs share nearly the same $k$, $\Delta E/E$, and angular acceptance means a single calibration framework applies to the whole suite.

### 4.2 Counting rate formula / 카운팅 레이트 공식

차등 directional flux $j(E, \hat{\Omega})$ [particles cm⁻² s⁻¹ sr⁻¹ keV⁻¹]:
$$\boxed{\;R(E) \;=\; G(E)\, \cdot\, j(E)\, \cdot\, \Delta E \;=\; G_0\, E\, j(E)\, (\Delta E/E)\;}$$

여기서 Table I의 $G$가 $G_0 \cdot E$ 형식인 이유는 ESA의 $\Delta E \propto E$ 특성 때문. 예: EESA-H, $E = 1$ keV, $j = 10^7$ /(cm²·s·sr·keV) → $R = 0.1\cdot 10^3\cdot 10^7\cdot 0.20 = 2\times 10^8$ Hz는 너무 큼 → 이 영역은 EESA-L이 담당 / This explains why $G$ scales as $E$ in Table I and the natural division of labor between -L and -H.

### 4.3 SST species separation / SST 종 분리

Foil 두께 $t_{\text{foil}}$는 양성자 range가 ~400 keV가 되도록 선택:
$$ R_p(E_p^{\text{cut}}) = t_{\text{foil}} \quad\Rightarrow\quad E_p^{\text{cut}} \approx 400 \text{ keV}. $$
한편 동일 두께에서 ~25 keV 이상 전자의 에너지 손실 $\Delta E_e \ll E_e$ (전자 mass scattering power is ~2000× smaller than for protons of same energy). Magnet side: gyroradius $r_L = m v / (qB)$. Broom magnet의 $B \sim 0.2$ T에서 $E_e = 400$ keV ($v \approx 0.83c$)인 전자의 $r_L$은 collimator 단면 크기와 비슷 → deflection으로 검출기를 비껴감. 동일 $E$의 양성자는 $r_L$이 ~43배 커서 거의 직진 / The asymmetry in $r_L$ between electrons and ions of the same energy provides the magnet-side filter; the much larger ion range vs. electron range (per unit foil thickness) provides the foil-side filter.

### 4.4 Geometric factor calculation / Geometric factor 계산

순수 기하학적 단면적 $A$와 입체각 수용범위 $\Omega$의 곱에 검출 효율 $\eta$를 곱한 값:
$$ G = A \cdot \Omega \cdot \eta_{\text{grid}} \cdot \eta_{\text{MCP}}. $$

PESA-H의 경우: $A_{\text{aperture}} \cdot \Omega_{\text{accept}} = 0.04\, E$ cm² sr (computer simulation), $\eta_{\text{MCP}} \approx 0.50$, $\eta_{\text{grid}} \approx 0.75$ → $G_{\text{tot}} = 0.04\cdot 0.50\cdot 0.75\, E \approx 0.015\, E$ cm² sr. EESA-H 동일 계산: $0.20\cdot 0.70\cdot 0.73\, E \approx 0.10\, E$ cm² sr (3 grids).

### 4.5 Direct wave-particle correlation / 직접 파-입자 상관

$$\boxed{\;C(v, \Theta) \;=\; \frac{\displaystyle\int E_0 \sin(kx - \omega t + \Theta)\, F(v,t)\, dt}{\langle E^2\rangle_t^{1/2}\, \langle F^2\rangle_t^{1/2}}\;}$$

quadrature 두 측정 $C_S = C(v, 0)$, $C_C = C(v, \pi/2)$의 결합:
- 진폭 $A(v) = \sqrt{C_S^2 + C_C^2}$
- 위상 $\phi(v) = -\arctan(C_S / C_C)$.

$F(v,t) = F_0(v) + \delta F(v) \sin(\omega t + \phi_F(v))$ 로 가정하면 $A(v) \propto \delta F / F_0$이며 이는 wave에 의한 전자 bunching 비율을 직접 측정.

### 4.6 Statistical sensitivity / 통계적 감도

분포 1% 변화 $3\sigma$ 검출 조건:
$$ \frac{3}{\sqrt{N}} = 0.01 \quad\Rightarrow\quad N = 9\times 10^4. $$

Langmuir burst $\Delta t \sim 100$ ms → required count rate $\geq 10^6$ Hz. 이는 EESA-H의 16-cm 직경 + $G_0 = 0.1E$를 정당화 / Justifies the EESA-H size and high $G$ design choice on purely statistical grounds.

### 4.7 Pitch-angle binning / 피치각 비닝

3D 분포 $f(v, \theta_{\text{spin}}, \phi_{\text{anode}})$를 magnetometer가 제공한 $\hat{B}$ 기준 1D PAD로 재 binning:
$$ f_{\text{PAD}}(v, \alpha) = \frac{1}{\Delta\Omega(\alpha)} \int_{\hat\Omega \in [\alpha, \alpha+\Delta\alpha]} f(v, \hat\Omega)\, d\Omega, \qquad \cos\alpha = \hat\Omega\cdot\hat{B}. $$

이는 onboard에서 EESA/PESA CPU가 미리 계산해 telemetry 부담을 줄이고, scatter-free 빔 (좁은 PAD)과 diffusive 산란 (넓은 PAD)의 구별을 가능케 한다.

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1968 ─ Asbridge+: upstream solar-wind ions (foreshock seed)
1971 ─ Lin & Hudson: solar-flare flash-phase electrons
1973 ─ Feldman+: solar-wind heat flux & halo electrons
1978 ─ ISEE-3 launch — first ~2 keV electron analyzer
1978 ─ Sarris+: magnetospheric burst particles in IPM
1979 ─ Filbert & Kellogg: foreshock electron beams
1980 ─ Gosling+, Paschmann+, Potter+: shock-reflected ions
1981 ─ Lin+: type III electron f(v_||) measurement
1982 ─ Anderson+: structured PAD at IP shocks
1983 ─ Carlson+: top-hat symmetric ESA design ★
1985 ─ Lin: quiet-time 2-20 keV electrons
1985 ─ Gough: particle auto-correlator technique
1992 ─ Lin & Kahler: long-range PAD probes
1993 ─ Manuscript received (Jan 28)
        ─────────────────────────
1994 ★ Wind launch (Nov 1) — 3DP begins operation
1995 ★ THIS PAPER — Lin et al., SSR 71, 125
1995 ─ Bougeret+: WAVE experiment paper (companion)
1995 ─ Acuna+: MFI magnetometer paper (companion)
1996 ─ Lin+: type III electron 3D distributions
1996 ─ Larson+: Wind/3DP halo, strahl observations
1998 ─ Larson+: scatter-free electron events
2000 ─ Cluster launch (PEACE inherits top-hat)
2007 ─ THEMIS launch (ESA inherits top-hat)
2015 ─ MMS launch (FPI inherits top-hat + burst mode + WPC)
2018 ─ Parker Solar Probe (SPAN-A/E inherits top-hat optics)
2020 ─ Solar Orbiter (SWA, EPD-STEP inherit foil/magnet idea)
2024+ ─ Wind/3DP still operating ─ 30-year reference dataset
```

이 paper는 Carlson 1983의 광학 설계와 Gough 1985의 상관기 기법을 결합하여, 이후 30년간 우주 플라스마 직접 측정의 표준이 되는 instrument architecture를 정의했다. ISEE-3가 발견을 했다면, Wind/3DP는 정량화하고 검증했다 / Wind/3DP was where ISEE-3-era discoveries became quantitative measurements.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **Carlson et al. (1983)** *Adv. Space Res.* 2(7), 67 | Top-hat 대칭 quadrispherical analyzer 광학 원조 / Origin of the symmetric quadrispherical optic | EESA/PESA 광학의 직접 heritage. Wind이 첫 본격 적용. / Direct heritage; Wind is the first major flight application. |
| **Lin (1985)** *Solar Phys.* 100, 537 | ISEE-3에서 발견한 quiet-time 2-20 keV 전자 / Quiet-time 2-20 keV electrons on ISEE-3 | §2.1의 핵심 motivating discovery. 3DP는 이 population을 통계적으로 정량화하기 위해 설계됨. / Motivating discovery for §2.1; 3DP is purpose-built to quantify it. |
| **Lin et al. (1981)** *Astrophys. J.* 251, 364 | ISEE-3에서 처음 측정한 type III f(v_||) / First type III $f(v_\parallel)$ from ISEE-3 | §2.3의 motivation. FPC가 이 측정을 ms-cadence + bunching phase로 확장. / Motivation for §2.3 and FPC. |
| **Anderson et al. (1982)** *Space Sci. Rev.* 32, 169 | IP shocks에서 ~2-10 keV 전자의 PAD 구조 / Pitch-angle structure at IP shocks | §2.5 motivating data; Wind/3DP의 빠른 PAD가 직접 후속 측정. / Wind/3DP fast PADs follow up directly. |
| **Gough (1985)** *IEEE TGRS* GE-23, 305 | Particle auto-correlator 기법 / Particle auto-correlator technique | FPC §8.2의 직접 heritage. Wind에서 in-flight 적용. / Direct heritage of FPC auto-correlation. |
| **Bougeret et al. (1995)** *Space Sci. Rev.* 71 (this volume) | WAVE 실험 instrument paper / WAVE experiment paper | FPC가 WAVE의 $E_x, E_y$ 신호를 입력으로 받음 — 두 paper는 companion. / WAVE provides FPC's input E-field; companion paper. |
| **Acuna et al. (1995)** *Space Sci. Rev.* 71 (this volume) | MFI magnetometer paper | onboard PAD 계산이 MFI의 $\vec{B}$ vector에 의존. / Onboard PAD relies on MFI $\vec{B}$. |
| **Ergun et al. (1991)** *J. Geophys. Res.* 96, 225 | Auroral sounding-rocket FPC prototype | FPC §8.1 직접 heritage 기기. / Direct prototype for FPC architecture. |
| **Paschmann et al. (1980)** *J. Geophys. Res.* 85, 4689 | Bow shock에서 반사된 이온 빔 / Bow-shock reflected ion beams | §2.6 motivating discovery; PESA-H 빠른 sweep으로 추적. / Motivating discovery for §2.6. |
| **Lin & Kahler (1992)** *J. Geophys. Res.* 97, 8203 | 전자 PAD를 long-range probe로 사용 / Electrons as long-range heliospheric probes | §2.1의 과학적 동기를 제공 / Provides the scientific rationale of §2.1. |

---

## 7. References / 참고문헌

- Lin, R. P., Anderson, K. A., Ashford, S., Carlson, C., Curtis, D., Ergun, R., Larson, D., McFadden, J., McCarthy, M., Parks, G. K., Rème, H., Bosqued, J. M., Coutelier, J., Cotin, F., d'Uston, C., Wenzel, K.-P., Sanderson, T. R., Henrion, J., Ronnet, J. C., and Paschmann, G., "A Three-Dimensional Plasma and Energetic Particle Investigation for the Wind Spacecraft", *Space Science Reviews* **71**, 125–153, 1995. DOI: 10.1007/BF00751328
- Carlson, C. W., Curtis, D. W., Paschmann, G., and Michael, W., "An Instrument for Rapidly Measuring Plasma Distribution Functions With High Resolution", *Adv. Space Res.* **2** (7), 67, 1983.
- Acuna, M. et al., *Space Sci. Rev.* **71**, this volume, 1995. (MFI magnetometer)
- Bougeret, J.-L. et al., *Space Sci. Rev.* **71**, this volume, 1995. (WAVE experiment)
- Anderson, K. A., Lin, R. P., and Potter, D. W., *Space Sci. Rev.* **32**, 169, 1982.
- Anderson, K. A., Lin, R. P., Martel, F., Lin, C. S., Parks, G. K., and Rème, H., *Geophys. Res. Lett.* **6**, 401, 1979.
- Asbridge, J. R., Bame, S. J., and Strong, I. B., *J. Geophys. Res.* **73**, 5777, 1968.
- Ergun, R. E., Carlson, C. W., McFadden, J. P., Clemmons, J. H., and Boehm, M. H., *J. Geophys. Res.* **96**, 225, 1991.
- Feldman, W. C., Asbridge, J. R., Bame, S. J., and Montgomery, M. D., *J. Geophys. Res.* **78**, 3697, 1973.
- Filbert, P. C. and Kellogg, P. J., *J. Geophys. Res.* **84**, 1369, 1979.
- Gloeckler, G., Hovestadt, D., and Fisk, L. A., *Astrophys. J.* **230**, L191, 1979.
- Gosling, J. T., Asbridge, J. R., Bame, S. J., Feldman, W. C., Paschmann, G., and Sckopke, N., *J. Geophys. Res.* **85**, 744, 1980.
- Gough, M. P., *IEEE Trans. Geosci. Remote Sens.* **GE-23**, 305, 1985.
- Lin, R. P., *Space Sci. Rev.* **16**, 189, 1974.
- Lin, R. P., *Solar Phys.* **100**, 537, 1985.
- Lin, R. P. and Hudson, H. S., *Solar Phys.* **17**, 412, 1971; **50**, 153, 1976.
- Lin, R. P. and Kahler, S. W., *J. Geophys. Res.* **97**, 8203, 1992.
- Lin, R. P., Levendahl, W. K., Lotko, W., Gurnett, D. A., and Scarf, F. L., *Astrophys. J.* **308**, 954, 1986.
- Lin, R. P., Potter, D. W., Gurnett, D. A., and Scarf, F. L., *Astrophys. J.* **251**, 364, 1981.
- Paschmann, G., Sckopke, N., Asbridge, J. R., Bame, S. J., and Gosling, J. T., *J. Geophys. Res.* **85**, 4689, 1980.
- Potter, D. W., Lin, R. P., and Anderson, K. A., *Astrophys. J.* **236**, L97, 1980.
- Sarris, E. T., Krimigis, S. M., Bostrom, C. O., and Armstrong, T. P., *J. Geophys. Res.* **83**, 4289, 1978.
