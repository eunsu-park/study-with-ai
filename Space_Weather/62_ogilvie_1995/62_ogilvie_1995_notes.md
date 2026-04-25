---
title: "SWE, A Comprehensive Plasma Instrument for the Wind Spacecraft"
authors: K. W. Ogilvie, D. J. Chornay, R. J. Fritzenreiter, F. Hunsaker, J. Keller, J. Lobell, G. Miller, J. D. Scudder, E. C. Sittler Jr., R. B. Torbert, D. Bodet, G. Needell, A. J. Lazarus, J. T. Steinberg, J. H. Tappan, A. Mavretic, E. Gergin
year: 1995
journal: "Space Science Reviews"
doi: "10.1007/BF00751326"
topic: Space_Weather
tags: [WIND, SWE, faraday-cup, electrostatic-analyzer, strahl, solar-wind, ISTP, plasma-instrument]
status: completed
date_started: 2026-04-25
date_completed: 2026-04-25
---

# 62. SWE, A Comprehensive Plasma Instrument for the Wind Spacecraft / WIND 위성용 종합 플라즈마 측정기 SWE

---

## 1. Core Contribution / 핵심 기여

This paper is the **definitive instrument paper** for the Solar Wind Experiment (SWE) — the dedicated solar-wind plasma analyzer of NASA's WIND spacecraft and a cornerstone of the International Solar-Terrestrial Physics (ISTP) program. SWE solves a long-standing measurement compromise: no single detector class can simultaneously characterize **supersonic, narrow-cone solar-wind ions** (where Faraday cups excel), **subsonic or warm plasmas with broad angular distributions** (where electrostatic analyzers excel), AND the **narrow field-aligned strahl beam** (where high-angular-resolution toroidal analyzers excel). Ogilvie et al. address this by integrating all three sensor classes — two MIT-style Faraday cups (FC) covering 150 V to 8 kV; two triads of 127° cylindrical electrostatic analyzers (VEIS) covering 7 V to 24.8 kV; and a 131° toroidal strahl analyzer covering 5 V to 5 kV — under one Sandia 3300-based Data Processing Unit (DPU). The instrument delivers proton velocity (3 components, 200-1250 km/s, ±3%), density (0.1-200 cc⁻¹, ±10%), thermal speed (0-200 km/s, ±10%), and α/p ratio (0-100%, ±10%) "key parameters" at 1-second cadence in single-spin mode, and full 3-D ion + electron VDFs over a few seconds in burst mode.

이 논문은 NASA WIND 위성에 탑재된 **태양풍 실험(SWE) 측정기의 결정판 설명 논문**으로, ISTP 프로그램의 핵심 자료원이다. SWE는 측정 분야의 오랜 절충 문제를 해결한다. 즉 어떤 단일 검출기도 **초음속·좁은 콘 태양풍 이온**(FC가 최적), **마하수 ≤ 1의 따뜻한 광각 분포**(ESA가 최적), **자기력선 정렬 좁은 스트랄**(고분해능 토로이달 분석기가 최적)을 동시에 측정할 수 없다. Ogilvie 외는 세 종류의 센서를 단일 Sandia 3300 기반 DPU 아래 통합해 이를 해결한다 — 150 V부터 8 kV의 MIT 형 FC 두 대, 7 V부터 24.8 kV의 127° 원통형 VEIS 분석기 두 트라이어드, 5 V부터 5 kV의 131° 토로이달 스트랄 분석기. 이 측정기는 양성자 속도(3성분, 200-1250 km/s, ±3%), 밀도(0.1-200 cc⁻¹, ±10%), 열 속도(0-200 km/s, ±10%), 알파/양성자 비(0-100%, ±10%) 등 "키 매개변수"를 단일 회전 모드에서 1초 시간 분해능으로, 버스트 모드에서는 수 초 내 완전한 3차원 이온·전자 분포함수를 얻는다. SWE의 통합형 설계는 ACE/SWEPAM, Parker Solar Probe/SWEAP/SPC, Solar Orbiter/SWA-PAS의 모범이 되었다.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction and Scientific Aims (Sections 1-2) / 서론과 과학 목표 (1-2장)

**Page 55 (Abstract + opening).** SWE is described as "a comprehensive, integrated set of sensors which is designed to investigate outstanding problems in solar wind physics." Three sensor families are listed: FC, VEIS, strahl. Energy ranges: FC 150 V – 8 kV; VEIS 7 V – 24.8 kV; strahl ≈ 5 V – 5 kV (Table II). Time resolution depends on operational mode but can reach a few seconds for 3-D measurements. Heritage: FC inherits from Voyager (Bridge et al. 1977) and IMP-7/8 (Bellomo & Mavretic 1978); VEIS inherits from ISEE-1 (Ogilvie et al. 1978).

**Page 55 발췌 + 서론.** SWE는 "태양풍 물리의 미해결 문제를 다루기 위한 종합·통합 센서군"으로 소개된다. 세 센서: FC, VEIS, 스트랄. 에너지 범위는 FC 150 V–8 kV, VEIS 7 V–24.8 kV, 스트랄 ≈ 5 V–5 kV(표 II). 시간 분해능은 모드에 따라 다르며 3차원 측정에서도 수 초 가능. 계보: FC는 Voyager(Bridge 외 1977)와 IMP-7/8(Bellomo & Mavretic 1978), VEIS는 ISEE-1(Ogilvie 외 1978)에서 계승.

**Table I (page 56) — Key parameters specification.**

| Parameter | Range | Precision |
|---|---|---|
| Proton velocity (3 components) | 200-1250 km/s | ±3% |
| Proton number density | 0.1-200 cc⁻¹ | ±10% |
| Thermal speed | 0-200 km/s | ±10% |
| α/p number density ratio × 100 | 0-100 % | ±10% |

표 I (56쪽)은 키 매개변수 사양표이다. 양성자 속도 3성분 200-1250 km/s ±3%, 밀도 0.1-200 cc⁻¹ ±10%, 열 속도 0-200 km/s ±10%, 알파/양성자 비 0-100% ±10%.

**Section 2 — Scientific objectives (pages 56-60).** Seven sub-aims:
1. Energy/momentum input to the magnetosphere (GSE-related).
2. Magnetospheric effects of upstream waves and pressure changes.
3. L1 — bow shock region (foreshock).
4. Bow shock and magnetosheath studies (Mach 0 - 20).
5. VDFs of protons, alphas, and electrons.
6. High time resolution (1 s) ion VDFs.
7. Interplanetary shocks and CIR / CME interaction regions.

The paper distinguishes ISTP-related goals (1-2) from heliospheric goals (5-7); bow-shock studies (4) bridge both. Figure 1 (page 59) shows ISEE-1-derived electron moments — n_e, u_e, elevation/azimuth angles, T_e, anisotropy, heat flux, and h direction — as the spacecraft crosses a bow shock at 22:52 UT. The heat flux $h$ jumps and reverses direction multiple times across the foreshock, highlighting why high-cadence strahl observations matter.

2장의 일곱 가지 과학 목표 중 (1-2)는 ISTP 자기권 관련, (5-7)은 헬리오스피어 물리, (4)는 양쪽을 잇는다. 그림 1(59쪽)은 ISEE-1에서 도출한 전자 모멘트(밀도, 흐름, T_e, 비등방성, 열속, 방향)가 22:52 UT 보우 쇼크 통과 시 어떻게 변하는지를 보여주며, 전방 충격파역에서 열속 $h$가 수 차례 방향을 뒤집는 모습이 스트랄 관측의 중요성을 부각한다.

### Part II: Instrument General Configuration (Section 3.1-3.2) / 측정기 전반 구성 (3.1-3.2장)

**Page 60 — General layout.** Five sensor boxes + DPU + calibrator. Figure 2 (page 61) shows the spatial arrangement on the WIND spacecraft: FC1 + VEIS#1 (looking down) on the bottom shelf, FC2 + VEIS#2 (looking up) on the top shelf — diametrically opposite. Strahl detector mounts at -X_SC near VEIS#1. DPU sits at +X_SC on top of the bottom shelf. Each sensor group communicates with the DPU through an interface board housed in VEIS. Cup normals are tilted ±15° relative to spin plane (one cup at +15°, other at -15°), permitting elevation-angle measurement from cup-current ratio.

5개 센서 박스 + DPU + 교정기. 그림 2(61쪽)는 WIND 위성에서의 배치: FC1+VEIS#1(아래쪽 보기)이 아래 데크, FC2+VEIS#2(위쪽 보기)가 위 데크 — 직경 양 끝. 스트랄 검출기는 -X_SC 쪽 VEIS#1 근처. DPU는 +X_SC. 각 센서 그룹은 VEIS 하우징에 든 인터페이스 보드로 DPU와 통신. 컵 법선이 회전면 ±15° 기울어져 있어 두 컵 전류비로 고도각 산출 가능.

**Page 60 — DPU.** Sandia 3300 CPU. Functions: command decoding, sensor control through pre-defined modes, data formatting, on-board key-parameter computation, science-mode storage in EEPROM (re-programmable in flight). Mode switching by time-tagged pointers — no per-event uplink. Figure 3 (page 63) is the block diagram showing Strahl, FC1, VEIS1 → Interface Board 1 → DPU → Interface Board 2 → FC2, VEIS2; spacecraft provides power, commands, magnetic-field data input, and accepts data + a density signal to the WAVES experiment. Two fiber-optic cables carry signals to the diametrically-opposite second sensor group.

DPU는 Sandia 3300 CPU. 기능: 명령 해석, 사전 정의 모드로 센서 제어, 데이터 포맷팅, 키 매개변수 온보드 연산, EEPROM 모드 저장(재프로그래밍 가능). 모드 전환은 시각태그 포인터로 — 사건마다 명령 업링크 불필요. 그림 3은 신호 흐름도. 위성은 전원·명령·자기장 자료를 제공하고, 데이터와 WAVES용 밀도 신호를 받는다. 광섬유 두 가닥이 직경 반대쪽 센서 그룹과 통신.

**Table II (page 62) — Instrument characteristics.**

| Component | Parameter | Value |
|---|---|---|
| Faraday cup | E/q range | 150 V – 8 kV |
| | Operating frequency | ≈ 200 Hz |
| | Effective area/cup | 35 cm² |
| | ΔE/E narrow / double windows | 0.065 / 0.130 |
| | Maximum window width | 1 kV |
| | Equivalent geometric factor | 1.1×10² cm² sr |
| VEIS | E/q range | 7 V – 24.8 kV |
| | Analyzer FOV | 7.5° × 6.5° |
| | ΔE/E | 0.06 |
| | GF per analyzer | 4.6×10⁻⁴ cm² sr |
| | Min step dwell time | 5 ms |
| | Analyzer constant | 7:1 |
| | Plate radii (in/out) | 4.717 / 5.443 cm |
| Strahl | E/q range | 5 V – 5 kV |
| | Analyzer FOV | ≈ 3° × ±30° |
| | ΔE/E | 0.03 |
| | GF per anode | 7×10⁻⁴ cm² sr |
| | Min step time | 30 ms |
| | Plate radii (in/out) | 5.40 & 14.4 / 6.60 & 15.6 cm |

표 II는 측정기 사양표이다. FC는 150 V–8 kV, 35 cm² 유효 면적, ΔE/E 0.065/0.130 두 가지 창. VEIS는 7 V–24.8 kV, FOV 7.5°×6.5°, ΔE/E 0.06, GF 4.6×10⁻⁴ cm² sr. 스트랄은 5 V–5 kV, FOV ≈ 3°×±30°, ΔE/E 0.03, GF 7×10⁻⁴ cm² sr.

### Part III: Vector Electron and Ion Spectrometer (Section 3.3) / 벡터 전자·이온 분광계 (3.3장)

**Pages 62-66 — VEIS.** Two triads (one looking up, one looking down) of three 127° cylindrically-symmetric electrostatic analyzers each. Plate radii 4.717 cm / 5.443 cm; analyzer constant 7:1 means $V_{plate}/V_{particle} \approx 1/7$. Balanced bipolar deflection through 127°. Same plates measure ions and electrons sequentially by reversing the polarity of the deflection field — a shunt-regulator with an LED-controlled high-voltage diode (Loidl 1984) handles the polarity switch. Each analyzer has TWO channeltron detectors — one for each polarity slot at the exit — equipped with cones to match the analyzer slit. AMPTEK A-111 charge-sensitive preamplifiers provide both digital and log pulse-height outputs; the latter are periodically pulse-height-analyzed to verify gain saturation.

VEIS는 직경 반대쪽 두 트라이어드(각각 위·아래 보기), 트라이어드 당 127° 원통대칭 정전 분석기 3개. 플레이트 반지름 4.717/5.443 cm, 분석기 상수 7:1로 $V_{plate}/V_{particle} \approx 1/7$. 127° 평형 양극 편향. 편향 전기장 극성을 뒤집어 같은 플레이트로 이온·전자를 순차 측정 — Loidl(1984)의 LED 제어 고전압 다이오드 션트가 극성 전환. 분석기 당 두 채널트론(이온용·전자용), 콘으로 분석기 슬릿 형상에 맞춤. AMPTEK A-111 프리앰프에서 디지털·로그 펄스 출력, 후자는 주기적으로 PHA로 이득 포화 검증.

**Solid-angle coverage strategy (Figure 4, 5).** Each analyzer FOV is 7.5°×6.5°, GF 4.6×10⁻⁴ cm² sr, ΔE/E ≈ 0.06. To cover 4π sr with only six analyzers, each must represent ~4π/6 ≈ 2 sr — valid only if the underlying VDF varies smoothly over solid angles much larger than the analyzer FOV. This holds for Mach ≤ 1 plasmas (electrons in solar wind, ions in magnetosheath, diffuse foreshock ions). Higher-Mach ion flows are NOT captured by VEIS — that's the FC's job. With 6 energy scans of 16 points per 60° spin sector, one full 3-D snapshot contains 6×16×6 = **576 points**. Figure 5 (page 66) shows a typical ISEE-1 reduced electron VDF: contour plot in (v_∥, v_⊥) plane shows clear core (Maxwellian-like) plus halo, but a gap along v_∥ — exactly the gap the strahl detector is designed to fill on WIND.

각 분석기 FOV 7.5°×6.5°, 4π를 6개로 분할하면 한 분석기당 ~2 sr 담당 → 분포가 분석기 FOV보다 훨씬 넓은 입체각에서 매끄럽게 변할 때만 유효. 이는 마하수 ≤ 1 플라즈마(태양풍 전자, 자기권 외피 이온, 확산성 전방 충격파역 이온)에 적용된다. 마하수 큰 이온은 FC 담당. 60° 섹터당 16점×6 에너지 스캔 = 6×16×6 = 576점이 3차원 스냅샷. 그림 5는 ISEE-1 전자 분포 등고선 — 코어+할로 구조 + $v_\parallel$ 방향 갭. 이 갭이 WIND 스트랄 검출기의 표적.

**Channeltron/MCP cleanliness (page 66-67).** Channeltrons, like MCPs, are gain-degraded by contamination. SWE design: pre-launch nitrogen purging, mechanical isolation of detector chamber from electronics outgassing, and channeltron installation AFTER thermal-vacuum testing. Reference: ISEE instrument lasted 12 years and accumulated 2×10¹¹ counts using less rigorous protection. The UV calibrator (page 67) is the periodic in-flight check: a single 2 W RF UV lamp coupled by optical fibers to all six analyzers; ~1% relative gain stability is the goal.

채널트론·MCP는 오염 시 이득 감소. SWE 대책: 발사 전 질소 퍼지, 검출기 챔버를 전자장비 가스 방출에서 격리, 열진공 후 채널트론 장착. ISEE는 약한 보호로도 12년·2×10¹¹ 카운트 달성. 비행 중 점검은 UV 교정기 — 단일 2 W RF UV 램프가 광섬유로 6개 분석기에 분배, 월 1회 ~1% 상대 이득 안정.

### Part IV: Faraday Cup Subsystem (Section 3.4) / 패러데이 컵 부속계통 (3.4장)

**Pages 67-70 — FC mechanical layout.** Figure 6a shows a cross section: planar grids stacked above two semicircular collector plates (Collector A, Collector B). Outermost grid grounded; modulator grid carries 200 Hz square wave at 0 to V+ΔV; suppressor grid biased at -130 V to prevent secondary-electron escape. Cup OD 6.000″ (15.24 cm); modulator-to-collector distance ~3.5 cm; collector plates split in halves to give elevation-angle information from a single cup.

FC 기계 구조(그림 6a): 평면 그리드들이 두 반원 콜렉터 위에 적층. 최외곽 그리드 접지; 변조 그리드는 200 Hz 사각파(0~V+ΔV); 억제 그리드는 -130 V로 2차 전자 탈출 차단. 컵 외경 6″(15.24 cm); 변조-콜렉터 거리 ~3.5 cm; 콜렉터를 둘로 분할해 단일 컵에서도 고도각 정보 획득.

**FC operating principle (Figure 6b).** Modulator potential alternates between V and V+ΔV. Only ions with $\frac{1}{2} m v_n^2$ in [qV, q(V+ΔV)] pass during the V state and get blocked during V+ΔV state — producing a 200 Hz chopped current at the collector. Synchronous (lock-in) detection at 200 Hz extracts the chopped current while rejecting DC photo-electron leakage from sunlit interior surfaces. Light-pipe path through a hole in the modulator + outer plate prevents sunlight from striking the collector chamber directly. ΔV up to 1 kV → energy window up to 1 kV wide; "double-window" mode uses two consecutive levels to give ΔE/E = 0.130. Voltage steps are logarithmically spaced from 64 levels covering 150 to 8000 V, $\Delta v/v ≈ 0.033$ per single step.

작동 원리(그림 6b): 변조 전위가 V와 V+ΔV를 교번. $\frac{1}{2} m v_n^2$가 [qV, q(V+ΔV)] 안인 이온만 V 상태에서 통과, V+ΔV 상태에서 차단 → 200 Hz 변조 전류. 200 Hz 락인 동기 검파로 광전자 DC 누설을 제거. 모듈레이터·외측 플레이트 구멍을 통한 광 경로 설계로 햇빛이 콜렉터를 직접 비추지 않게 차단. ΔV는 1 kV까지 → 최대 1 kV 폭의 에너지 창; 2단 창(double-window)은 ΔE/E = 0.130. 64개 로그 등간격 전압으로 150-8000 V 범위 커버, 단일 창 $\Delta v/v ≈ 0.033$.

**Effective area vs angle (Figure 6c).** Plot of total effective collecting area vs incidence angle to cup normal: ~35 cm² at 0°-30°, dropping smoothly past 40°, falling to 0 at ~60° (the geometric half-angle limit set by aperture-to-collector ratio). This $A_{\text{eff}}(\theta)$ enters Equation 1 below.

유효 면적 vs 각(그림 6c): 0°-30°에서 ~35 cm², 40° 이후 부드럽게 감소, 약 60°에서 0(개구-콜렉터 비율로 결정되는 기하학적 반각 한계). $A_{\text{eff}}(\theta)$가 식 (1)에 들어간다.

**Four advantages of FC over ESA (page 69).**
1. **Variable bandwidth**: ΔV is software-tunable (vs ESA's geometry-fixed bandwidth) — useful for reflected-ion studies and automated peak-finding.
2. **Flow direction <1°**: Two cups give precise direction reconstruction.
3. **Wide acceptance ≈ ±60°**: With two cups at opposite ends of a 3-s spinner, the SW is being sampled 2/3 of the time; full VDF in ~1 s in single-spin mode.
4. **Reduced VDF measurement**: FC measures $F = \iint f \, dv_\perp dv_\perp$ along its normal — ideal for compact telemetry and absolute-density determinations because no energy-dependent efficiency correction is needed.

ESA 대비 FC의 네 이점:
1. **가변 대역폭**: ΔV는 소프트웨어 튜닝 가능(ESA는 기하학적 고정).
2. **흐름 방향 정밀도 < 1°**: 두 컵으로 정확히 재구성.
3. **넓은 수용 ≈ ±60°**: 직경 양 끝의 두 컵이 3 s 스핀에서 SW를 2/3 시간 표본화; 단일 회전 모드로 약 1초에 전체 VDF.
4. **축소 분포 측정**: 컵 법선 방향으로 $F = \iint f \, dv_\perp dv_\perp$ 적분; 콤팩트 텔레메트리, 에너지 의존 효율 보정 불필요로 절대 밀도 측정에 적합.

**FC current signals — Figure 7 (page 70).** Simulated currents for V_sw = 400 km/s, thermal speed 40 km/s, n_p = 10 cc⁻¹, n_α/n_p = 0.05. Five panels show current vs azimuth angle for five voltage windows scanning from below to above the wind speed:
- Top panel (309-329 km/s, below peak): twin peaks at ±~30° because only off-axis ions have $v_n = v_{sw}\cos\theta$ in the slow window. Peak amplitude ~480 pA.
- 329-350 km/s: twin peaks closer (±~25°), higher (~580 pA).
- 350-373 km/s: twin peaks barely separated, plateau-like.
- 373-397 km/s (at peak): nearly flat top across ±30°, reaching ~800 pA.
- 397-423 km/s (above peak): single peak at 0° (only fastest ions, normally incident); amplitude ~620 pA.

The angular separation of the twin peaks encodes thermal speed; the centroid encodes flow direction; the plateau height encodes density; the energy at which the plateau is widest encodes bulk speed. **In one 3-s spin, all four parameters are extractable** — basis of the single-spin mode.

그림 7은 V_sw = 400 km/s, 열속 40 km/s, n_p = 10 cc⁻¹, α/p = 0.05 모의에서의 Azimuth-전류 곡선 5개(다른 변조 창). 창이 벌크보다 느릴수록 큰 각에서 쌍봉이 나타나고, 벌크에 일치하면 평탄, 빠르면 0° 단봉. 쌍봉 분리 폭 → 열속, 중심 → 흐름 방향, 평탄 높이 → 밀도, 평탄 폭 최대 에너지 창 → 벌크 속도. 한 회전(3 s) 안에 네 매개변수 모두 추출 — 단일 회전 모드의 원리.

**FC measurement chain — Figure 8 (page 72).** Block diagram. Modulator high-voltage supplies 64 logarithmically-spaced levels 150-8000 V; control logic synchronizes timing. Each collector half feeds a preamp (bandwidth ≈ 200 Hz, centered at 200 Hz). Three series range amplifiers: gains 7, 46.5, 46.5 (cumulative ~1.5×10⁴). Synchronous detector / integrator on each range output, 30 ms integration normally (120 ms when cup looks away from Sun). Multiplexer selects highest-gain unsaturated output → log A/D → 10-bit + 2-bit range = effective 10⁵ dynamic range. Current range 3×10⁻¹³ A (thermal noise floor at 30 ms) to 3×10⁻⁸ A. Combined data rate from both FC sensors = **320 bits s⁻¹**.

측정 체인(그림 8): 변조 고압 64개 로그 등간격 150-8000 V; 콜렉터 반쪽마다 프리앰프(대역 200 Hz)→3단 직렬 레인지 앰프(이득 7, 46.5, 46.5)→동기 검파/적분기(30 ms 또는 120 ms)→멀티플렉서가 최고이득 비포화 출력 선택→로그 A/D 10비트+2비트 레인지 = 동적 범위 10⁵. 전류 3×10⁻¹³–3×10⁻⁸ A. 두 FC 데이터율 합 320 bits s⁻¹.

**Modes of FC operation (page 71-73).**
- **Full-scan mode**: 31 velocity windows over 93 s at 3 s spin → complete energy-azimuth map every ~30 spins.
- **Tracking mode**: 14 windows centered on previous spectrum's peak; 42 s cadence; full-scan re-initiated every 30 minutes or when peak current drops below threshold.
- **Single-spin mode**: One double-window just below the VDF peak; produces twin-peak trace in 3 s; fits give $n, \vec{u}, T$.

운영 모드: 풀스캔(31창/93초), 트래킹(14창/42초, 30분마다 재초기화 또는 피크 임계값 미만시), 단일 회전(피크 직하 한 창, 3초).

### Part V: The Strahl Detector (Section 3.5) / 스트랄 검출기 (3.5장)

**Pages 73-74 — Strahl scientific motivation.** Solar-wind electrons above ~40 eV traveling along $\vec{B}$ have very low Coulomb collision rates and travel from the corona to 1 AU with little scattering — forming a narrow field-aligned beam, the "strahl." This is well-studied between 0.3 and 1 AU by Helios (Marsch 1991), but extended 1-AU monitoring is needed to clarify how strahl pitch-angle width varies with interplanetary structures. Required spec: angular resolution ≈ few degrees, total angular coverage ≈ 30°×30° centered on $\vec{B}$.

> 40 eV 태양풍 전자는 자기력선을 따라 충돌 거의 없이 코로나에서 1 AU까지 도달 — 자기력선 정렬 좁은 빔(strahl). Helios(0.3-1 AU; Marsch 1991)가 잘 연구했으나 1 AU 장기 관측이 필요. 요구 사양: 각 분해능 수 도, 총 커버리지 ~30°×30°.

**Strahl detector design (Figure 9, page 74).** Truncated toroidal electrostatic analyzer with 131° included angle (Young et al. 1987 design). Plate radii: 5.40 cm + 14.4 cm radius of curvature on inner / 6.60 cm + 15.6 cm on outer. ΔE/E = 0.03 (twice the energy resolution of VEIS). Field of view ≈ ±28° in a plane containing the spin axis. Two channel plates (MCPs) with **6 anodes each** of ~5°. Read out 16° after the solar direction every 31 ms until 72° after solar direction → 15 observations per spin half. Sequence repeats when WIND has rotated half a revolution and looks anti-Sunward along $\vec{B}$. With magnetic-field fluctuations, the strahl is within the sensor's FOV approximately **60% of total observing time**. The center anode and channel plates are absent so that no detector receives direct sunlight; the outer plate is serrated to minimize reflected light.

스트랄 검출기 설계(그림 9): 131° 잘린 토로이달 정전 분석기(Young 외 1987 설계). ΔE/E = 0.03(VEIS의 두 배 정밀). FOV는 회전축 포함 평면에서 ±28°. 두 MCP, 각 6 anode×~5°. 태양 방향 16° 후부터 72° 후까지 31 ms마다 읽기 → 회전 반쪽당 15회 관측. 반대 방향 회전(반사 일주)에도 반복. 자기장 변동을 포함해 스트랄이 센서 FOV에 들어오는 시간은 전체 관측 시간의 약 60%. 중앙 anode와 MCP는 직접 햇빛을 받지 않도록 제거; 외측 플레이트는 톱니 형태로 반사광 최소화.

### Part VI: Modes of Operation (Section 4) / 운영 모드 (4장)

**Mode 0 (page 75) — Survival/ROM mode.**
- FC: One double window per spin; 28 spins to scan 150 to 5478 V. Plus one calibration spin per 28 (12 input currents).
- VEIS: 60° spin sector → 16 sequential energy steps, ~3.53 ms per step (1/17 of sector). Full energy range covered in many spins.
- Strahl: Fourteen 60°/17 samples per spin centered on nominal 45° spiral angle of B, twice per rotation (both field directions). 16 spins for full energy range.
- Integration time 30 ms when cup within ±60° of S/C-Sun line; 120 ms otherwise (low flux).
- Sun-relative trigger angle is 4° before Sun direction to compensate for orbital-aberration angle.

모드 0(생존/ROM): FC는 28회 스핀으로 150-5478 V 스캔(이중 창); VEIS 60° 섹터당 16 에너지 단계 ~3.53 ms/단계; 스트랄 60°/17 샘플 14개씩 양 방향. ±60° 이내 30 ms, 외 120 ms. Sun 트리거는 광행차 보정을 위해 -4°.

**Mode 1 (page 75) — General-purpose EEPROM mode.**
- Full FC E/q range covered in contiguous spins.
- Calibration currents injected.
- VEIS energies switched between electrons and ions every other 60° sector → 6 s time resolution for both species in foreshock.

모드 1(범용): FC 전 범위 연속 스핀, 교정 전류 주입, VEIS 60° 섹터마다 전자·이온 전환 — 6 s 시간 분해능.

**Other modes (page 76).**
- **Burst (event) mode**: Triggered by pre-defined criteria (FC current threshold, magnetic field change, command from MFI/WAVES); fills buffer with high-rate data before and after the trigger. Re-played slowly afterward.
- **Tracking mode**: 14 FC windows centered on previous-scan peak.
- **Single-spin mode**: Window just below peak; twin-peak trace in one spin.

기타 모드: 버스트(트리거 시 사전·사후 고속 자료), 트래킹(14창), 단일 회전(피크 직하 한 창 + 한 회전).

---

## 3. Key Takeaways / 핵심 시사점

1. **Three sensor classes united under one DPU** — The fundamental design choice of SWE is to give up "one detector does everything" in favor of three optimized sub-instruments (FC for supersonic ions, VEIS for M ≤ 1 plasmas, strahl for field-aligned beams) sharing a single Sandia 3300 DPU. This breaks the energy/angular-resolution/dynamic-range trade-offs that plagued earlier instruments.
   **세 센서군의 단일 DPU 통합** — SWE의 핵심 설계 결정은 "단일 검출기로 모든 영역 측정"을 포기하고, 초음속 이온용 FC, 마하수 ≤ 1용 VEIS, 자기력선 정렬 빔용 스트랄의 세 최적화 서브측정기를 단일 Sandia 3300 DPU 아래 통합한 것이다. 이로써 이전 측정기를 괴롭혔던 에너지·각·동적 범위 절충을 깬다.

2. **200 Hz lock-in detection eliminates photo-electron noise** — A modulator grid driven at 200 Hz square wave gates the energy window; synchronous detection at 200 Hz preserves only the AC component, throwing away DC photo-electrons that would otherwise dominate (~nA vs pA signal). Multiple grounded grids prevent capacitive coupling. This technique sets the standard for FC instruments and is reused on ACE/SWEPAM and PSP/SPC.
   **200 Hz 락인 검파로 광전자 잡음 제거** — 200 Hz 사각파 변조 그리드 + 200 Hz 동기 검파로 AC 성분만 보존, DC 광전자(~nA, 신호 pA) 완전 제거. 접지 그리드 다수로 용량 결합 차단. ACE/SWEPAM, PSP/SPC가 그대로 채택한 표준 기법이다.

3. **Twin-peak FC trace encodes all key parameters in one spin** — Figure 7 shows that an FC sweep at one energy window across a 3-s spin produces twin peaks whose **angular separation = thermal speed**, **centroid = flow azimuth**, **plateau height = density**, and the window where the plateau widens = bulk speed. This single-spin mode delivers $\{n, \vec{u}, T\}$ at 1-second cadence — an order of magnitude faster than ESA-based instruments of the era.
   **단일 회전 쌍봉 곡선이 모든 키 매개변수를 인코딩** — 한 에너지 창의 3 s 회전 곡선에서 쌍봉 각 분리 = 열속, 중심 = 흐름 방향, 평탄 높이 = 밀도, 평탄이 가장 넓은 창 = 벌크 속도. 단일 회전 모드로 $\{n, \vec{u}, T\}$를 1초마다 — 동시대 ESA 기반보다 한 자릿수 빠르다.

4. **Single 2 W UV lamp + fiber-optic distribution gives 1% relative gain stability** — The UV calibrator solves a chronic ESA problem: independent channeltron drift would corrupt the 6×16×6 = 576-point 3-D distribution reconstruction. A common photon source (one lamp, six fibers) keeps inter-detector ratios known to ~1% even if the lamp itself drifts, because all six detectors see the same drift simultaneously.
   **단일 2 W UV 램프 + 광섬유 분배로 1% 상대 이득 안정** — UV 교정기는 ESA의 고질적 문제(개별 채널트론 표류로 576점 3차원 분포 재구성 손상)를 해결한다. 공통 광원(한 램프, 6 광섬유)을 사용하면 램프 자체가 표류해도 모든 검출기가 동시에 같은 표류를 보므로 검출기 간 비율은 ~1% 유지.

5. **Strahl detector closes the angular gap of VEIS** — Figure 5 (ISEE-1 reduced VDF) shows a clear gap along $v_\parallel$ that omnidirectional ESAs cannot fill because of their ~7° FOV. The 131° toroidal Young et al. analyzer with 6 anodes × 5° per MCP gives ~5° angular resolution and ±28° FOV in the spin-axis plane — enough to resolve the few-degree-wide field-aligned electron strahl.
   **스트랄 검출기가 VEIS의 각 갭을 메운다** — 그림 5의 ISEE-1 분포는 7° FOV의 ESA로는 자기력선 정렬 방향 갭을 채울 수 없음을 보여준다. Young 외의 131° 토로이달 분석기 + 6 anode×5° MCP로 약 5° 각 분해능과 회전축면 ±28° FOV — 수 도 폭 strahl 분해 가능.

6. **Logarithmic A/D + range-amplifier multiplexing achieves 10⁵ dynamic range** — Three series range amplifiers (gains 7, 46.5, 46.5; total ~1.5×10⁴) followed by a multiplexer that picks the highest unsaturated output, then a logarithmic 10-bit A/D + 2-bit range tag = effective 12-bit equivalent dynamic range from 3×10⁻¹³ A (thermal noise floor) to 3×10⁻⁸ A. This covers everything from quiet solar-wind to the densest magnetosheath without gain switching at 30 ms cadence.
   **로그 A/D + 레인지 앰프 다중화로 10⁵ 동적 범위 달성** — 직렬 3단 레인지 앰프(이득 7, 46.5, 46.5; 총 ~1.5×10⁴) → 비포화 최고 이득 선택 멀티플렉서 → 로그 10비트 A/D + 2비트 레인지 = 12비트 등가 동적 범위, 3×10⁻¹³ A부터 3×10⁻⁸ A. 정온 SW부터 자기권 외피까지 30 ms로 이득 전환 없이 측정.

7. **Re-programmable EEPROM modes enable post-launch optimization** — Mode 0 lives in ROM as a survival baseline; Mode 1 + custom modes live in EEPROM and can be uploaded after launch via time-tagged pointers, requiring no per-event commands. This flexibility allowed SWE to be reprogrammed multiple times during 30 years of operation, adapting to L1 halo orbit, lunar wake, and switchback science.
   **재프로그래밍 가능한 EEPROM 모드** — 모드 0은 ROM의 생존 기본값, 모드 1 외 사용자 모드는 EEPROM에 저장되어 발사 후 시각태그 포인터로 업로드 가능, 매 사건마다 명령 불필요. 이 유연성으로 SWE는 30년 운영 중 L1 헤일로 궤도·달 후류·스위치백 과학에 맞춰 여러 번 재프로그래밍되었다.

8. **The single-DPU architecture cuts spacecraft interface to one harness** — Figure 3 shows that the spacecraft sees only the DPU; all five sensor sub-units interface internally. Power, command, telemetry, and even WAVES density signal pass through this single boundary. This dramatically reduced spacecraft integration complexity and is now standard practice (Solar Orbiter/SWA, MMS/FPI).
   **단일 DPU 구조로 위성 인터페이스 일원화** — 그림 3은 위성이 DPU만 본다는 것을 보여준다. 5개 센서 서브유닛은 내부적으로 인터페이스되며 전원·명령·텔레메트리·WAVES 밀도 신호 모두 이 단일 경계를 거친다. 위성 통합 복잡도를 크게 줄였으며 Solar Orbiter/SWA, MMS/FPI 등 후속 미션의 표준 관행이다.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Faraday cup current integral / 패러데이 컵 전류 적분

The chopped current at the collector when the modulator gates a window $[v_1, v_2]$ along the cup normal $\hat{n}$:

$$
\boxed{\,I(\hat{n}, v_1, v_2) = q \, A_{\text{eff}}(\theta) \int_{v_1}^{v_2} v_n \, F(v_n; \hat{n}) \, dv_n\,}
$$

where $v_1 = \sqrt{2qV/m}$, $v_2 = \sqrt{2q(V+\Delta V)/m}$, $\theta = \arccos(\hat{n} \cdot \hat{u}/u)$ is the incidence angle, and the **reduced distribution function** along $\hat{n}$ is

$$
F(v_n; \hat{n}) = \iint_{v_\perp} f(v_n \hat{n} + \vec{v}_\perp) \, d^2 v_\perp.
$$

Variables: $q$ ion charge, $m$ ion mass, $A_{\text{eff}}(\theta)$ the effective collecting area (Figure 6c, ~35 cm² near $\theta = 0$, dropping to 0 at $\theta \approx 60°$), $f$ the full 3-D VDF.

### 4.2 Convected (drifting) Maxwellian for proton fitting / 양성자 적합용 컨벡티드 맥스웰

$$
f_p(\vec{v}) = n_p \left(\frac{m_p}{2\pi k_B T_p}\right)^{3/2} \exp\!\left[-\frac{m_p (\vec{v} - \vec{u})^2}{2 k_B T_p}\right]
$$

The reduced distribution along an arbitrary $\hat{n}$ becomes a 1-D Maxwellian centered at $\hat{n}\cdot\vec{u}$:

$$
F_p(v_n; \hat{n}) = n_p \left(\frac{m_p}{2\pi k_B T_p}\right)^{1/2} \exp\!\left[-\frac{m_p (v_n - \hat{n}\cdot\vec{u})^2}{2 k_B T_p}\right].
$$

The four parameters $\{n_p, \vec{u}, T_p\}$ are extracted by **non-linear least-squares fit** to the chopped currents measured at multiple $(\hat{n}, V, \Delta V)$ — explicitly stated by the authors in §3.4.1.

매개변수 $\{n_p, \vec{u}, T_p\}$는 여러 $(\hat{n}, V, \Delta V)$에서의 변조 전류에 대한 비선형 최소제곱 적합으로 추출.

### 4.3 Adding alphas (two-species fit) / 알파 입자 포함

$$
F_{\text{tot}}(v_n; \hat{n}) = F_p(v_n; \hat{n}) + F_\alpha(v_n; \hat{n})
$$

With $q_\alpha = 2q_p$ and $m_\alpha = 4 m_p$, the **same** modulator voltage $V$ corresponds to a different normal-component speed for $\alpha$:

$$
v_\alpha = \sqrt{\frac{2 q_\alpha V}{m_\alpha}} = \sqrt{\frac{V}{(\text{V/(km/s)}^2)}\cdot\frac{q_p}{m_p}} \cdot \sqrt{2/4} = v_p / \sqrt{2}.
$$

Equivalently, $\alpha$ peaks at twice the energy/charge $V_p$ that $p$ does for the same speed: $V_\alpha = (m_\alpha/q_\alpha) v_\alpha^2/2 = 2 V_p$ at the same $v$. Figure 7 caption notes the alpha currents peak in an energy window higher than shown.

알파 입자는 같은 변조 전압 $V$에서 양성자의 $1/\sqrt{2}$ 배 속도에 해당하므로 같은 흐름 속도에 대해 알파의 최대치는 $V_\alpha = 2 V_p$에서 나타난다.

### 4.4 ESA energy band / ESA 에너지 대역

For a 127° cylindrical analyzer with plate radii $r_1, r_2$ and analyzer constant $K = (r_1+r_2)/(2(r_2-r_1))$:

$$
E/q = K \cdot V_{\text{plate}}, \qquad K = 7 \text{ for VEIS},
$$

so $V_{\text{plate}} = 1$ V → $E/q = 7$ eV/q. The analyzer transmission has FWHM $\Delta E / E = 0.06$ (Figure 4 lower panel).

VEIS 분석기 상수 7로 $V_{\text{plate}} = 1$ V 시 $E/q = 7$ eV/q; ΔE/E = 0.06.

### 4.5 Geometric factor → count rate / 기하 인자→카운트율

Differential count rate from a VDF:

$$
\frac{dC}{dt} = \text{GF} \cdot v^4 \cdot f(\vec{v}) \quad \text{[counts/s per energy bin]}
$$

with GF = $A \cdot \Omega \cdot \Delta E/E$ in cm² sr (units of GF as quoted in Table II). For VEIS: GF = 4.6×10⁻⁴ cm² sr; for strahl: 7×10⁻⁴ cm² sr per anode.

VDF에서 미분 카운트율 $dC/dt = \text{GF} \cdot v^4 \cdot f$.

### 4.6 Electron heat flux / 전자 열속

$$
\vec{q}_e = \tfrac{1}{2} m_e \int (\vec{v} - \vec{u})^2 (\vec{v} - \vec{u}) \, f_e(\vec{v}) \, d^3 v
$$

In the solar-wind frame this points along $\hat{B}$ outward from the Sun. Decomposing into core + halo + strahl:

$$
\vec{q}_e \approx \vec{q}_{e,\text{strahl}} \quad \text{(dominant for } |v_\parallel| > 3 v_{th,e}\text{)}.
$$

Figure 1 shows $h$ varying by ~10× across foreshock crossings — the strahl detector is what makes such measurements possible.

태양풍 좌표계에서 열속은 $\hat{B}$ 따라 태양 반대 방향. $|v_\parallel| > 3 v_{th,e}$ 영역에서 strahl가 지배. 그림 1: 전방 충격파역 통과 시 ~10× 변화.

### 4.7 Single-spin twin-peak inversion / 단일 회전 쌍봉 역산

For a Maxwellian solar wind at angle $\theta$ from cup normal, the FC current with window $[V, V+\Delta V]$ (where modulator velocity $v_M = \sqrt{2qV/m}$ is just below the peak):

$$
I(\theta) \propto v_n F_p(v_n)\bigg|_{v_n=v_M} \cdot \Delta v_M \approx I_0 \exp\!\left[-\frac{(v_M - u\cos\theta)^2}{w_p^2}\right]
$$

with $w_p = \sqrt{2 k_B T_p/m_p}$. The current $I(\theta)$ peaks where $v_M = u\cos\theta$, giving a peak azimuth

$$
\theta_{\text{peak}} = \pm \arccos(v_M/u).
$$

The angular separation $2\theta_{\text{peak}}$ depends only on $v_M$ and $u$. Once $u$ is known, the **width** of each peak (in $\theta$) gives $w_p$ via:

$$
\text{HWHM}_\theta \approx \frac{w_p \sqrt{\ln 2}}{u \sin \theta_{\text{peak}}}.
$$

This is the algebra behind the single-spin mode.

쌍봉 azimuth는 $\theta_{\text{peak}} = \pm \arccos(v_M/u)$, 분리 = $2\theta_{\text{peak}}$로 $u$ 결정; 봉 폭으로 $w_p$ 결정.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1959 ── Lunik 1 (Gringauz: first ion trap, hint of solar wind)
1962 ── Mariner 2 (Neugebauer & Snyder; FC; first proven solar wind)
1965 ── Vela series (FCs + ESAs; bow shock identified)
1971 ── Vasyliunas review (deep-space technique compendium; cited)
1973 ── IMP-7 launch (Bellomo & Mavretic FC heritage)
1974 ── Helios 1 launch (in-situ to 0.3 AU; Marsch reviews cited)
1976 ── IMP-8 launch (FC continued)
1977 ── Voyager 1/2 launch (Bridge et al. PLS — direct ancestor of SWE FC)
1977 ── ISEE-1 launch (Ogilvie et al. e-spectrometer — VEIS ancestor)
1979 ── Scudder & Olbert (theory of strahl — cited)
1984 ── Loidl (HV diodes for polarity switching — cited)
1987 ── Young et al. toroidal analyzer test (basis for strahl detector)
1991 ── Marsch in Schwenn & Marsch (Helios review of strahl — cited)
1993 ── Manuscript received by Space Sci. Rev. (May 27)
1994 ── WIND launch (1 Nov)
1995 ── THIS PAPER (Space Sci. Rev. 71, 55-77)         ◄────────────
1995 ── Lepping et al. MFI paper (companion in same volume)
1995 ── Lin et al. 3DP paper (companion)
1995 ── Bougeret et al. WAVES paper (companion)
1997 ── ACE launch (SWEPAM = simplified SWE-FC)
2003 ── Marsch (Living Reviews) review of strahl after WIND
2015 ── DSCOVR launch (FC again)
2018 ── Parker Solar Probe / SWEAP / SPC (FC heritage in ~1 R_☉ regime)
2020 ── Solar Orbiter / SWA-PAS (top-hat ESA + heritage)
2024+ ── WIND still operating in extended-mission L1 halo
```

This 30-year heritage line traces directly from Mariner 2's pioneering FC through Voyager → IMP → SWE → SWEPAM → SPC. The strahl detector in SWE became the prototype for all subsequent dedicated suprathermal-electron analyzers.

이 30년 계보는 Mariner 2의 선구적 FC에서 Voyager → IMP → SWE → SWEPAM → SPC로 이어진다. SWE의 스트랄 검출기는 이후 모든 전용 초열전자 분석기의 원형이 되었다.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| #43 Ogilvie & Desch 1997 (WIND mission overview) | This SWE paper provides the instrument detail that #43 references when listing WIND's payload; #43 reports early SWE results (Russell-McPherron, foreshock ions, etc.) | Direct: same PI Ogilvie; #43 cannot be interpreted without the FC/VEIS/strahl architecture described here |
| #61 Lepping et al. 1995 (WIND/MFI) | Companion paper in same Space Sci. Rev. volume; MFI provides the magnetic field that defines the strahl direction and the GSE/GSM coordinate transforms used by SWE moments | Companion: SWE strahl detector is meaningful only with co-located 1-Hz B from MFI |
| #63 Lin et al. 1995 (WIND/3DP) | Companion paper; 3DP electron analyzers cover 3 eV - 30 keV with ~22.5° resolution and complement VEIS for transient electron events; 3DP also provides ions to MeV | Companion: 3DP and SWE are the two ion/electron instruments on WIND; cross-calibration is routine |
| #64 Bougeret et al. 1995 (WIND/WAVES) | WAVES receives the SWE density signal (Figure 3) for plasma-frequency comparison and trigger generation | Direct: SWE outputs density to WAVES; WAVES identifies Type III bursts in regions where SWE characterizes the local plasma |
| #42 Stone et al. 1998 (ACE mission) | ACE/SWEPAM is a simplified, mass-saving descendant of SWE-FC (single cup, no strahl) | Heritage: SWEPAM directly inherits the modulated-cup principle described here |
| #45 Goodman 2019 (DSCOVR review) | DSCOVR Faraday cup uses the same chopped-modulator, lock-in detection principle | Heritage: 200 Hz lock-in technique is a direct descendant |
| #4 Parker 1958 (solar wind) | SWE measures the supersonic flow Parker predicted; thermal speed × angular separation diagnostic confirms Parker's adiabatic-cooling expectation | Foundational: provides the theoretical context for what SWE measures |
| #15 Gonzalez 1994 (geomagnetic storms) | SWE supplies the upstream $V_{sw}$, $n_p$, and dynamic pressure that drive Burton-McPherron-RG-style empirical Dst models | Application: SWE feeds storm-prediction models |

---

## 7. References / 참고문헌

### Primary
- Ogilvie, K. W., Chornay, D. J., Fritzenreiter, R. J., Hunsaker, F., Keller, J., Lobell, J., Miller, G., Scudder, J. D., Sittler Jr., E. C., Torbert, R. B., Bodet, D., Needell, G., Lazarus, A. J., Steinberg, J. T., Tappan, J. H., Mavretic, A., and Gergin, E., "SWE, A Comprehensive Plasma Instrument for the Wind Spacecraft", *Space Science Reviews* **71**, 55-77, 1995. DOI: [10.1007/BF00751326](https://doi.org/10.1007/BF00751326)

### Cited heritage instruments
- Bellomo, A. and Mavretic, A., "MIT Plasma Experiment on IMP H and J Earth Orbited Satellites", MIT Center for Space Research Technical Report CSR-TR-78-2, 1978.
- Bridge, H. S., Belcher, J. W., Butler, R. J., Lazarus, A. J., Mavretic, A. M., Sullivan, J. D., Siscoe, G. L., and Vasyliunas, V. M., "The Plasma Experiment on the 1977 Voyager Mission", *Space Sci. Rev.* **21**, 259, 1977.
- Ogilvie, K. W., Scudder, J. D., and Doong, H., "The Electron Spectrometer on ISEE-1", *ISEE Trans. on Geosci. Electronics* **GE-16**, 261, 1978.
- Young, D. T., Ghielmetti, A. G., Shelley, E. G., Marshall, J. A., Burch, J. L., and Booker, T. L., "Experimental Tests of a Toroidal Electrostatic Analyzer", *Rev. Sci. Instrum.* **58**, 501, 1987.

### Cited theory and reviews
- Vasyliunas, V. M., "Deep Space Plasma Measurements", in *Methods of Experimental Physics* **9B**, 49, Academic Press, 1971.
- Scudder, J. D. and Olbert, S., "A Theory of Local and Global Processes which Affect Solar Wind Electrons. 1. The Origin of Typical 1 AU Velocity Distribution Functions — Steady State Theory", *J. Geophys. Res.* **84**, 2755, 1979a.
- Scudder, J. D. and Olbert, S., "A Theory of Local and Global Processes which Affect Solar Wind Electrons. 2. Experimental Support", *J. Geophys. Res.* **84**, 6603, 1979b.
- Marsch, E., "Kinetic Physics of the Solar Wind Plasma", in R. Schwenn and E. Marsch (eds.), *Physics of the Inner Heliosphere*, Vol. II, p. 45, Springer-Verlag, Heidelberg, 1991.
- Belcher, J. W., Lazarus, A. J., McNutt, R. L. Jr., and Gordon, G. S., "Solar Wind Conditions in the Outer Heliosphere and the Distance to the Termination Shock", *J. Geophys. Res.*, 1993.
- Loidl, A., "HV Diodes Used as Variable Resistors and Switches", *J. Phys. E: Sci. Instruments* **17**, 357, 1984.

### Companion ISTP instrument papers (same Space Sci. Rev. 71 volume)
- Lepping, R. P. et al., "The Wind Magnetic Field Investigation", *Space Sci. Rev.* **71**, 207, 1995.
- Lin, R. P. et al., "A Three-Dimensional Plasma and Energetic Particle Investigation for the WIND Spacecraft", *Space Sci. Rev.* **71**, 125, 1995.
- Bougeret, J.-L. et al., "WAVES: The Radio and Plasma Wave Investigation on the WIND Spacecraft", *Space Sci. Rev.* **71**, 231, 1995.

### Foreshock / upstream waves background
- "Upstream Waves and Particles" compilation, *J. Geophys. Res.* **86**, 4319, 1981.
