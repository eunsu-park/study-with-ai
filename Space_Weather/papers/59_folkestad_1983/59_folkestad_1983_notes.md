---
title: "EISCAT: An Updated Description of Technical Characteristics and Operational Capabilities"
authors: [Kristen Folkestad, Tor Hagfors, Svante Westerlund]
year: 1983
journal: "Radio Science, 18(6), 867–879"
doi: "10.1029/RS018i006p00867"
topic: Space_Weather
tags: [EISCAT, incoherent_scatter_radar, tristatic, ionosphere, auroral, radar_engineering, ACF, multipulse]
status: completed
date_started: 2026-04-25
date_completed: 2026-04-25
---

# 59. EISCAT: An Updated Description of Technical Characteristics and Operational Capabilities / EISCAT: 기술적 특성과 운용 역량의 갱신된 기술

---

## 1. Core Contribution / 핵심 기여

**EN**: This paper is the definitive engineering description of the **European Incoherent SCATter (EISCAT)** facility as commissioned in 1981–1982. EISCAT is the world's first auroral-zone tristatic incoherent-scatter radar: a single transmitter at Ramfjordmoen, Norway illuminates a scattering volume in the high-latitude ionosphere while three antennas — at Ramfjordmoen, Kiruna (Sweden), and Sodankylä (Finland) — receive the scattered signal simultaneously. Two independent transmitting systems are fielded: a fully steerable 32 m parabolic dish at **UHF 933.5 MHz** (2 MW peak, 250 kW average, 12.5% duty) optimised for ion-line measurements, and a 4×30 m × 40 m **VHF 224 MHz** parabolic-cylinder phased array (5 MW peak, 625 kW average) optimised for $D$-region and plasma-line work. The receiver chain uses helium-cooled parametric amplifiers (system $T_{\rm sys} \sim 40$ K), 8-bit complex I/Q digitisers at up to 500 kHz, a 13-bit Barker decoder, and a custom 16-bit programmable hardware correlator clocked at 5 MHz with pipeline architecture, 4096-word ping-pong buffer, and 2048-word result memory. Operational software stack: **EROS** real-time OS (SINTRAN III based) running on NORD-10 minicomputers, **TARLAN** radar-controller assembly language, **CORLAN** correlator high-level language, **PLAN** experiment-design tool, with permanent 9600-baud lines linking the three sites and a NORD-500 for offline analysis at Kiruna.

**KR**: 본 논문은 1981–1982년 인수시험을 마친 **유럽 비간섭산란(EISCAT)** 시설에 대한 결정판 공학 기술서이다. EISCAT은 세계 최초의 오로라 영역 삼정점(tristatic) ISR이다. 노르웨이 Ramfjordmoen의 송신기 1대가 고위도 전리권 산란체적을 조사하고, 같은 Ramfjordmoen·스웨덴 Kiruna·핀란드 Sodankylä의 안테나 3대가 동시 수신한다. 송신 시스템은 두 가지가 있다: 이온선 측정에 최적화된 **UHF 933.5 MHz** 완전조향식 32 m 파라볼라(첨두 2 MW, 평균 250 kW, 듀티 12.5%)와 $D$ 영역·플라스마선 측정용 **VHF 224 MHz** 4×30×40 m 원통형 위상배열(첨두 5 MW, 평균 625 kW). 수신 체인은 헬륨 냉각 파라메트릭 전치증폭기(시스템 잡음 $T_{\rm sys} \sim 40$ K), 최대 500 kHz의 8-bit 복소 I/Q 디지타이저, 13-bit Barker 복호기, 5 MHz 클럭으로 동작하는 16-bit 자체개발 하드웨어 상관기(파이프라인 구조, 4096-word ping-pong 버퍼, 2048-word 결과 메모리)로 구성된다. 운영 소프트웨어는 NORD-10 미니컴퓨터에서 동작하는 SINTRAN III 기반 **EROS** 실시간 OS, **TARLAN** 레이더 제어 어셈블리어, **CORLAN** 상관기 고수준 언어, **PLAN** 실험 설계 도구로 이루어지며, 세 사이트는 9600-baud 영구회선으로 연결되고 Kiruna에는 오프라인 분석용 NORD-500이 배치된다.

---

## 2. Reading Notes / 읽기 노트

### Part I: §1 Introduction & §2 Design Characteristics / 서론과 설계 특성

#### §1 Introduction (p. 867)

**EN**: The paper positions itself explicitly as an **update**, not a first description: prior treatments include du Castel & Testud (1974), Evans (1975), Rishbeth (1978), and Hagfors (1982 Nobel Symposium). The contribution here is twofold — (i) finalised technical numbers reflecting acceptance tests, and (ii) the operational procedures for new external users. The observatory comprises **two independent systems**: a tristatic UHF radar (the scientific star, optimised for ion drifts) and a monostatic VHF radar (extending range to mesosphere and plasma-line). Both transmitters live at Ramfjordmoen near Tromsø; remote receiving sites are Kiruna and Sodankylä.

**KR**: 본 논문은 자신을 **갱신본**으로 명시 — du Castel & Testud(1974), Evans(1975), Rishbeth(1978), Hagfors(1982)를 선행 자료로 인용. 기여는 두 가지: (i) 인수시험을 반영한 최종 공학 수치, (ii) 외부 사용자를 위한 운용 절차. 관측소는 **두 독립 시스템**으로 구성 — 이온 표류 측정에 최적화된 삼정점 UHF 레이더(과학적 주역)와 중간권·플라스마선까지 범위를 확장하는 단정점 VHF 레이더. 두 송신기는 모두 Tromsø 근처 Ramfjordmoen에 위치하고, 원격 수신 사이트는 Kiruna와 Sodankylä.

#### §2.1 Transmitters (p. 867–868, Table 1)

**EN/KR table**:

| Parameter | UHF | VHF |
|---|---|---|
| Frequency / 주파수 | 933.5 ± 3.5 MHz | 224.0 ± 1.75 MHz |
| Frequency step / 주파수 간격 | 0.5 MHz | 0.25 MHz |
| Peak power / 첨두출력 | 2 MW | 5 MW |
| Average power / 평균출력 | 250 kW | 625 kW |
| Pulse length / 펄스 길이 | 10 µs – 10 ms | 10 µs – 1 ms |
| PRF / 펄스 반복률 | 0–1000 Hz | 0–1000 Hz |
| Duty cycle / 듀티 | 12.5% | 12.5% |

**EN**: Both finals are klystrons (UHF: 1 tube; VHF: 2 tubes — provision exists for adding a 2nd UHF klystron). Phase coding is supported in single- and multipulse sequences, with frequency switchable pulse-to-pulse or even within a pulse. The VHF dual-tube architecture allows fast polarization switching by adjusting the phase between low-power RF drives feeding the two klystrons; the UHF system has only one klystron and so cannot polarisation-flip. Since installation in 1980 the UHF transmitter was subjected to lengthy modifications and at the time of writing operates at full rated power in all required modes.

**KR**: 두 송신기 모두 클라이스트론을 종단으로 사용(UHF 1관, VHF 2관 — UHF에는 2번째 클라이스트론 추가 여지 존재). 위상 코딩은 단일·다중펄스 시퀀스에서 지원되며 펄스 간뿐 아니라 펄스 내에서도 주파수 전환이 가능. VHF 이중관 구조는 두 클라이스트론을 구동하는 저전력 RF 사이의 위상을 조절해 빠른 편파 전환을 가능하게 하지만, UHF는 클라이스트론이 1관뿐이라 같은 빠른 편파 플리핑이 불가. 1980년 설치 이후 UHF 송신기는 장기 보정·개조를 거쳐 본 논문 작성 시점에는 정격 출력으로 모든 요구 모드에서 동작.

#### §2.2 Antennas (p. 868–869, Tables 2–4)

**EN**: The **UHF antenna** is a fully steerable 32 m Cassegrain parabolic dish (subreflector 4.6 m, efficiency 0.65, gain 48 dB, half-power beamwidth 0.6°, slew 80°/min). The Kiruna and Sodankylä antennas are identical except for the feed: the Tromsø antenna has a **polarizer** (mounted near the feed horn) that can synthesise any polarization — RHCP, LHCP, elliptical, or linear at any orientation — controlled by two motor-driven phase changers. This is critical because Faraday rotation along the slant path differs at the three sites; without per-site polarization optimisation the link budget would suffer.

**KR**: **UHF 안테나**는 완전조향식 32 m Cassegrain 파라볼라(부반사경 4.6 m, 효율 0.65, 이득 48 dB, 반전력빔폭 0.6°, 회전속도 80°/min). Kiruna와 Sodankylä의 안테나는 급전부를 제외하면 동일. Tromsø 안테나는 급전혼 근처에 장착된 **편파기**로 RHCP, LHCP, 임의 방향 타원·직선 편파 합성 가능 — 두 개의 모터 구동 위상 변환기로 제어. 세 사이트의 사선 경로별 패러데이 회전이 서로 다르므로, 사이트별 편파 최적화 없이는 링크 예산이 손해를 봄. 따라서 매우 중요.

**EN**: The **VHF antenna** is a parabolic cylinder of four identical 30 m × 40 m elements at Ramfjordmoen. Mode I uses all four elements as a single antenna; **Mode II ("split beam")** splits elements 1+2 into one beam and 3+4 into another, giving two independent radiators (LHCP only — no polarisation flipping in Mode II). The primary feed is an array of 128 crossed dipoles on a feeder bridge along the focal line; beam steering is by phasing modulo 2π in 1.2° steps over ±21.3° from broadside. Because cabling lengths apply phasing modulo 2π (rather than absolute time delay), the antenna is **frequency-dispersive**: bandwidth shrinks rapidly with off-broadside steering (Figure 2). At broadside the half-antenna 64-element bandwidth is ~16 MHz, full antenna ~14 MHz; at 20° off-broadside both fall below 4 MHz. Table 3 lists effective area: **3330 m²** measured horizontal (vs 3250 m² calculated circular), beam 0.6° E/W × 1.7° N/S in Mode I, slew 5°/min, transit-plane steering 30° south to 60° north of zenith. Antenna locations (Table 4) are surveyed under the Nordic Geodetic Commission datum ED-50: Ramfjordmoen 69°35'11" N 19°13'23"–38" E (VHF and UHF separated); Kiruna 67°51'38" N 20°26'06" E; Sodankylä 67°21'49" N 26°37'37" E.

**KR**: **VHF 안테나**는 Ramfjordmoen에 있는 30 m × 40 m 동일 4개 요소 원통형 파라볼라. Mode I은 4개 요소 전체를 하나의 안테나로 사용; **Mode II("split beam")**는 요소 1+2를 한 빔, 3+4를 다른 빔으로 분리해 독립 방사체 2개 운용(LHCP만 가능 — Mode II에서는 편파 전환 불가). 주 급전부는 초점선을 따라 놓인 급전 브리지의 128개 십자 다이폴 배열; 빔 조향은 broadside로부터 ±21.3°까지 1.2° 단위로 modulo 2π 위상 조정. 케이블 위상이 절대 지연이 아닌 modulo 2π 형태이므로 안테나는 **주파수 분산성**을 가지며, broadside에서 멀어질수록 대역폭이 급감(Figure 2). Broadside에서 절반 안테나(64 요소) 약 16 MHz, 전체 약 14 MHz; 20° 이탈 시 양쪽 모두 4 MHz 미만으로 떨어짐. 표 3 유효 면적: 수평 측정 **3330 m²**(원편파 계산값 3250 m² 대비), 빔 Mode I에서 0.6° E/W × 1.7° N/S, 회전속도 5°/min, 통과면 조향 천정 남쪽 30°–북쪽 60°. 사이트 좌표(표 4, ED-50): Ramfjordmoen 69°35'11" N 19°13'23"–38" E (VHF·UHF 분리), Kiruna 67°51'38" N 20°26'06" E, Sodankylä 67°21'49" N 26°37'37" E.

#### §2.3 Receivers, A/D, Decoders (p. 869–870, Figure 4)

**EN**: At UHF, the cooled parametric amplifier yields $T_{\rm sys} \sim 40$ K (paramp itself ~20 K). At Kiruna and Sodankylä the first stage is a helium-cooled paramp; at the transmitter site an uncooled GaAs-FET amp gives $T_{\rm sys} = 120{-}150$ K. Two-stage downconversion: RF → 120 MHz IF (transferred to main building) → 30 MHz IF after second LO. The back-end provides **eight channels** for simultaneous reception of eight frequencies. After synchronous detection to baseband, I and Q channels are filtered by Butterworth filters (12.5/25/50/100 kHz) or new linear-phase Bessel filters (25/50/100/250 kHz) for Barker/multipulse, plus mesospheric filters (718 Hz, 3.34 kHz). Six channels sample at up to 500 kHz with **8-bit A/D**; two new channels run at 10 MHz (for plasma-line). Samples flow to the Barker decoder (matched to 13-bit code) when phase-coded pulses are used, otherwise direct to correlator buffer. At VHF the sky noise (100–200 K) makes a paramp unnecessary; a high-pass filter suppresses TV interference in 210–220 MHz.

**KR**: UHF 측은 헬륨 냉각 파라메트릭 증폭기로 $T_{\rm sys} \sim 40$ K(파라메트릭 자체 약 20 K). Kiruna와 Sodankylä 1단은 헬륨 냉각 파라메트릭; 송신지에서는 비냉각 GaAs-FET 증폭기로 $T_{\rm sys} = 120{-}150$ K. 2단 하향 변환: RF → 120 MHz IF(중앙 건물 전송) → 두 번째 LO 후 30 MHz IF. 백엔드는 **8 채널** 동시 수신. 베이스밴드 동기 검파 후 I·Q는 Butterworth(12.5/25/50/100 kHz) 또는 새 선형 위상 Bessel(25/50/100/250 kHz, Barker·multipulse용), 그리고 중간권용(718 Hz, 3.34 kHz)으로 필터링. 6 채널은 최대 500 kHz, **8-bit A/D**로 표본화; 신규 2 채널은 10 MHz(플라스마선용). 위상 부호 펄스 사용 시 13-bit Barker 복호기를 거쳐, 아니면 곧장 상관기 버퍼로. VHF 측은 하늘 잡음(100–200 K)이 커 파라메트릭 불필요; 210–220 MHz TV 간섭 억제용 고역 통과 필터 설치.

#### §2.4 Correlators (p. 870–871, Table 5)

**EN**: The principal correlator is a **purpose-built 16-bit programmable digital correlator** designed by Alker and colleagues at Tromsø (Alker 1976; Alker, Brattli & Roaldsen 1981). Architecture: pipeline structure with separated buffer memory and main correlator unit (control logic + arithmetic + result memory). Buffer is a **ping-pong 2 × 4096 16-bit-word** memory — one half collects new samples while the other is processed. Each buffer word packs 8 bits inphase + 8 bits quadrature. Result memory holds **2048 64-bit words** (32 bits real + 32 bits imaginary per lag product). The control microprogram has up to **64 × 128-bit instructions** in internal program memory; each experiment (single pulse, multipulse, power profile) ships its own microprogram. Master clock: **5 MHz**. Programming is hand-coded at machine level for each waveform, but a higher-level language **CORLAN** (Tørrustad 1982) was developed to ease this. A second French multibit correlator (Chabert et al. 1974) is co-located for plasma-line work; at Sodankylä an online spectral analyser based on a CCD (Aijänen 1981) computes 512-point spectra at 2 MHz sample rate, potentially yielding 10–100× plasma-line detection improvement when used with chirped transmissions.

**KR**: 주 상관기는 Tromsø의 Alker 그룹이 설계한 **목적 제작 16-bit 프로그래머블 디지털 상관기**(Alker 1976; Alker, Brattli & Roaldsen 1981). 구조: 버퍼 메모리와 주 상관기(제어로직 + 산술 + 결과 메모리)를 분리한 파이프라인. 버퍼는 **ping-pong 2 × 4096개 16-bit 워드** — 한 절반이 새 표본을 수집할 동안 다른 절반이 처리. 워드당 I 8-bit + Q 8-bit. 결과 메모리는 **2048개 64-bit 워드**(래그곱 1개당 실수 32-bit + 허수 32-bit). 제어 마이크로프로그램은 내부 프로그램 메모리에 **64 × 128-bit 명령어**까지 보유, 실험(단일 펄스·다중 펄스·전력 프로파일)별 자체 마이크로프로그램. 마스터 클럭 **5 MHz**. 본래 기계어 직접 작성이지만 고수준 언어 **CORLAN**(Tørrustad 1982)으로 간소화. 플라스마선용 프랑스 multibit 상관기(Chabert et al. 1974) 병설; Sodankylä에는 CCD 기반 온라인 스펙트럼 분석기(Aijänen 1981, 2 MHz, 512점), chirp 송신과 결합 시 플라스마선 검출이 10–100배 향상 가능.

### Part II: §2.5 – §2.9 Control, Timing & Software / 제어·타이밍·소프트웨어

#### §2.5 Radar controllers (p. 872)

**EN**: All high-precision timing signals (1 µs accuracy) are generated by a radar controller — one for VHF, one for UHF. Heart of it is a **4-kword matrix of 2 × 16-bit words**, half forming the "instruction time table" (ITT) and half the "instruction table" (IT). Bits 0–13 (a few unassigned) route to signal lines: e.g. bit 14 selects transmit mode, bit 11 controls RF modulation (klystron switching), bit 12 the dwell-time array. Programs are written in **TARLAN** (TRansmitter And Receiver LANguage) and compiled into bit patterns that drive the signal lines.

**KR**: 모든 고정밀 타이밍 신호(1 µs 정확도)는 레이더 제어기에서 생성 — VHF용·UHF용 각 1대. 핵심은 **4-kword의 2 × 16-bit 워드 행렬**: 절반은 "명령 시간표"(ITT), 절반은 "명령표"(IT). 비트 0–13(일부 미할당)이 신호선에 매핑 — 예) 비트 14 송신 모드 선택, 비트 11 RF 변조(클라이스트론 스위칭), 비트 12 dwell-time 배열. 프로그램은 **TARLAN**(TRansmitter And Receiver LANguage)으로 작성되어 컴파일러를 통해 신호선 구동 비트 패턴으로 변환.

#### §2.6 Frequency and timing standards (p. 872)

**EN**: Each station has a **Cesium standard** generating 1, 5, 10 MHz at $10^{-11}$ accuracy, plus a "traveling clock" for occasional cross-comparison. Long-term drift is corrected against Loran-C signals at 100 kHz received from a transmitter in Vesterålen. An **RT (real-time) clock** takes input from the Cs standard and feeds the radar controller with shaped second pulses; it can apply a programmable delay to compensate the propagation-time difference from the scattering volume to the three sites.

**KR**: 각 사이트는 **세슘 표준**($10^{-11}$ 정확도로 1·5·10 MHz 생성)을 보유하고, 비교·조정용 "traveling clock" 추가. 장기 드리프트는 Vesterålen에서 송신되는 100 kHz Loran-C로 보정. **RT(실시간) 시계**가 Cs 표준 입력을 받아 정형된 second pulse로 레이더 제어기에 공급; 산란체적에서 세 사이트로의 전파 시간 차이를 보상하는 프로그래머블 지연 가능.

#### §2.7 Computers and intersite communication (p. 872)

**EN**: At the transmitter station: a **NORD-10S** with two 75 MB discs. Each station also has a **NORD-10** with 128 kword memory, two 10 MB discs. CAMAC interfacing standardises peripheral connections. The three site computers are linked by permanent **9600-baud telephone lines** for mutual data/control transfer. A larger **NORD-500** is shared with Kiruna Geophysical Institute for offline scientific analysis.

**KR**: 송신 사이트에는 75 MB 디스크 2개를 가진 **NORD-10S**. 각 사이트에는 128 kword 메모리·10 MB 디스크 2개의 **NORD-10**. CAMAC으로 주변기기 연결 표준화. 세 사이트 컴퓨터는 영구 **9600-baud 전화회선**으로 상호 데이터·제어 전송. 오프라인 과학 분석은 Kiruna 지구물리연구소와 공유되는 **NORD-500**.

#### §2.8 Online graphical supervision (p. 872–873)

**EN**: **RT-graphs** (real-time graphs) are display systems on graphical terminals showing quick-look quality of received signals during integration. **MT-graph** is the tape-replay version. Options: ACF real/imag/spectrum for any range gate; mapping correlator-result-memory data fields. Hardcopies are producible on demand.

**KR**: **RT-graphs**(실시간 그래프)는 그래픽 단말의 표시 시스템으로 적분 중 수신 신호 품질을 즉석 진단. **MT-graph**는 테이프 재생 버전. 옵션: 임의 거리 게이트의 ACF 실수/허수/스펙트럼, 상관기 결과 메모리의 데이터 필드 매핑. 하드카피 인쇄 가능.

#### §2.9 EROS real-time operating system (p. 873)

**EN**: **EROS** (acronymic notation, syntax modeled on SINTRAN III; Armstrong 1980) is the real-time OS organising hardware control and data acquisition. To the user EROS is an assembly of ~75 high-level commands (point antennas, set receivers, load and run correlators, handle tapes, activate/terminate RT programs; Turunen 1982). In **REMOTE mode** one site can control parameters at the other two sites and transfer their RT-graph data in real time (Johansson 1982) — operationally proven in tristatic conditions.

**KR**: **EROS**(SINTRAN III 문법 기반의 EISCAT 자체 실시간 OS; Armstrong 1980)는 하드웨어 제어와 데이터 수집을 통합 관리. 사용자에게는 약 **75개 고수준 명령어** 묶음(안테나 지향, 수신기 설정, 상관기 적재·운용, 테이프 처리, RT 프로그램 활성/종료; Turunen 1982). **REMOTE 모드**에서는 한 사이트가 다른 두 사이트의 시스템 매개변수를 제어하고 RT-graph 데이터를 실시간 전송(Johansson 1982); 삼정점 운용에서 실증.

### Part III: §3 Preparation and Operational Philosophy / 실험 준비와 운용 철학

#### §3.1 Experiment files and parameter block (p. 873)

**EN**: A program **PLAN** (Williams 1982) helps observers define experiments: it computes SNR for various ionospheric models given pulse lengths, defines antenna pointing geometry for given scatter-volume locations, and constructs the **3 × 3 matrix** that converts the three line-of-sight velocities into velocity components along and transverse to $\vec{B}$. After the observer specifies pulse pattern, interpulse period, frequencies, sampling rates, antenna positions, scanning sequences and dwell times, two computer files are generated: (1) a file specifying all EROS commands, and (2) a TARLAN file defining the radar-controller bit patterns. Observers also prepare a "description file" for downstream analysts.

**KR**: 프로그램 **PLAN**(Williams 1982)은 실험 정의를 보조: 펄스 길이별로 다양한 전리권 모델 SNR 계산, 산란체적별 안테나 지향 기하 계산, 그리고 세 시선 속도를 $\vec{B}$ 평행·수직 성분으로 변환하는 **3 × 3 행렬** 도출. 관측자가 펄스 패턴, 펄스간 주기, 주파수, 표본화율, 안테나 위치, 스캐닝 시퀀스, dwell time을 지정하면 두 파일이 생성: (1) 모든 EROS 명령을 담은 파일, (2) 레이더 제어기 비트 패턴을 정의하는 TARLAN 파일. 관측자는 후속 분석자를 위한 "description file"도 준비.

#### §3.2 Library of correlator programs (p. 873–874, Figure 5)

**EN**: The library has grown rapidly in the three years since the first routines appeared. Most existing microprograms are by Ho (1981) and Kofman/CEPHAG-Grenoble (1982). Components: power profiles (zero-lag ACFs), single- and multipulse ACFs, cross-correlation functions (for tristatic mode). A special mesospheric ACF program (Kofman 1982; tested by Röttger) uses pulse-to-pulse integration to handle long correlation times in the $D$ region. A general-purpose waveform combining single+multipulse ACFs and two power profiles at different range resolutions has been designed by LaHoz and Hansen (1982).

**EN — gating philosophy**: In single-pulse experiments so far, signals are **range-gated before** correlation; lag estimates are then weighted by a triangular function (matched-filter result), and only lags up to the pulse length contribute. The contribution increases linearly with lag number from zero up to the maximum lag. Ho et al. (1983) added an alternative: form lag products from samples within an interval **3× the pulse length**, using expectation-value cross products. This widens the effective height range, but only contributions actually within the pulse correlate. Advantages: weighting compensates triangular distortion to some extent, and **all lags share the same range resolution** — important for spectral fitting since the range cell otherwise depends on lag number.

**KR**: 라이브러리는 첫 루틴 등장 이후 3년 만에 빠르게 성장. 대부분의 마이크로프로그램은 Ho(1981)와 Kofman/CEPHAG-Grenoble(1982) 작; 구성요소: 전력 프로파일(zero-lag ACF), 단일·다중펄스 ACF, 교차상관함수(삼정점 모드용). 특수 중간권 ACF 프로그램(Kofman 1982; Röttger 검증)은 $D$ 영역의 긴 상관시간을 위해 펄스간 적분 사용. LaHoz와 Hansen(1982)은 단일+다중펄스 ACF와 서로 다른 거리 분해능의 전력 프로파일 2개를 결합한 범용 파형 설계.

**KR — 게이팅 철학**: 지금까지의 단일펄스 실험에서는 신호를 상관 **전에** 거리 게이팅; 그 결과 래그 추정값은 삼각 함수로 가중되며(매치드 필터 결과), 펄스 길이까지의 래그만 기여. 기여는 0에서 최대 래그까지 선형 증가. Ho 등(1983)이 새 방법 추가: 펄스 길이의 **3배 구간** 표본에서 기댓값 교차곱으로 래그곱 형성. 유효 고도 범위가 넓어지지만, 펄스 안 표본만이 실제 상관에 기여. 장점: 가중치가 삼각 왜곡을 일부 보상하고 **모든 래그가 같은 거리 분해능을 공유** — 스펙트럼 적합 시 매우 중요(통상 거리 셀이 래그 번호에 의존).

#### §3.3 Modes of operation: CP and SP (p. 874–875)

**EN**: Per the EISCAT Association statutes, time is allocated in two modes:
1. **CP (Common Programme)** — fixed monthly suite, data shared with all associates after a reduced-CP latency. Specific common programmes:
   - **CP 0**: transmitter pointed along $\vec{B}$ at 300 km. Two pulses: short (10 km range resolution) for monostatic $N_e$, longer (50 km) for $T_e/T_i$ and tristatic ACFs.
   - **CP 1**: remote antennas scanned in double cycle to give tristatic at 6 $E$-region points and 2 $F$-region points. Composite waveform with single-pulse, multipulse, and power profile.
   - **CP 2**: monitors $F$- and $E$-layer space-time variations (irregularities, AGWs). Transmitter cycles through 3 directions in 6-min cycle (in Tromsø/Kiruna plane, in Tromsø/Sodankylä plane, near zenith). Remote-antenna 12-min cycle (6 min at 110 km × 3 positions, 6 min at 300 km × 3 positions). 3-pulse sequence: long $L$ for tristatic remote ACFs, short for monostatic ACF, multipulse for tristatic $E$-layer.
   - **CP 3**: 16-position scan transverse to $L$-shells from 64°–74° latitude along Tromsø/Kiruna line, 30-min cycle, remote at 325 km. Designed for high-latitude convection-pattern mapping.
2. **SP (Special Programme)** — bespoke proposals; data reserved 1 year for the proposer.

**EN**: The Association aims for **2500 hours/year**, evenly shared among associates; SP slots are sized proportional to financial contribution. Third-party (non-associate) scientists may collaborate via national associates.

**KR**: EISCAT 협약상 시간은 두 모드로 배정:
1. **CP(공통 프로그램)** — 매월 정기 운용; 축소판 데이터는 모든 회원국에 즉시 공유. 구체적인 공통 프로그램:
   - **CP 0**: 송신기를 300 km에서 $\vec{B}$ 방향으로 지향. 두 펄스: 단(10 km 거리 분해능)으로 monostatic $N_e$, 장(50 km)으로 $T_e/T_i$ 및 tristatic ACF.
   - **CP 1**: 원격 안테나를 이중 사이클로 스캔하여 $E$ 영역 6점, $F$ 영역 2점에서 tristatic 관측. 단일펄스·다중펄스·전력 프로파일 복합 파형.
   - **CP 2**: $F$·$E$ 층의 시공간 변동(이상현상, 음향중력파) 모니터. 송신기는 6분 주기로 3 방향 순환(Tromsø/Kiruna 평면, Tromsø/Sodankylä 평면, 천정 근처). 원격 안테나는 12분 주기(110 km 3점 6분, 300 km 3점 6분). 3펄스 시퀀스: 장 $L$로 tristatic 원격 ACF, 단으로 monostatic ACF, multipulse로 tristatic $E$ 층.
   - **CP 3**: Tromsø/Kiruna 선을 따라 64°–74° 위도의 $L$ shell 횡단 16점 스캔, 30분 주기, 원격은 325 km. 고위도 대류 패턴 매핑 목적.
2. **SP(특별 프로그램)** — 개별 제안; 데이터는 제안자에 1년간 독점.

**KR**: 협회는 **연 2500시간**을 목표로 회원국 간 균등 분배; SP 시간은 재정 분담 비율에 따라 크기 조정. 비회원국 과학자는 국가 위원회를 통해 협업 가능.

#### §3.4 Data analysis (p. 875)

**EN**: CP data are processed at EISCAT HQ in Kiruna, with subsequent tape copying, presentation, archiving, and distribution. EISCAT has a contract with **CNES** (Toulouse) for SP data processing. Suites by Lejeune (1979, 1982) and Silén (1981) compare theoretical ACFs — modified for finite pulse length and receiver characteristics — with measured ACFs and iterate to fit physical parameters. The routinely derived parameters are: **electron density $N_e$, electron and ion temperatures $T_e, T_i$ (assuming an ion mass), drift velocity $\vec{v}_d$**. Ion-neutral collision frequencies derivable in lower $E$ region (assuming temperature equilibrium). With good SNR the programs handle ion mixtures and the molecular-to-atomic transition (NO$^+$, O$_2^+$ → O$^+$) in lower $F$, and possibly the transition to lighter ions (He$^+$, H$^+$) at higher altitudes.

**KR**: CP 데이터는 Kiruna의 EISCAT 본부에서 처리 후 테이프 복사·발표·보관·배포. SP 데이터 처리에는 툴루즈의 **CNES**와 계약 체결. Lejeune(1979, 1982), Silén(1981) 분석 패키지는 유한 펄스·수신기 특성으로 보정한 이론 ACF와 측정 ACF를 비교해 반복 적합으로 물리량 도출. 일상 도출 변수: **$N_e$, $T_e$, $T_i$(이온질량 가정), 표류 속도 $\vec{v}_d$**. 하부 $E$ 영역에서 (온도 평형 가정 시) 이온-중성 충돌 진동수 가능. SNR 충분 시 이온 혼합비, 하부 $F$ 영역의 분자-원자 천이(NO$^+$, O$_2^+$ → O$^+$), 고도 상층의 경이온 천이(He$^+$, H$^+$) 처리 가능.

### Part IV: §4 Observational Results / 관측 결과

#### §4.1 Summary of effort and findings (p. 875–876)

**EN**: First spectra: **June 1981**, single-pulse, fixed directions, peak 500 kW. Activity escalated through autumn 1981, peaking November–December 1981 with concurrent observations from GEOS spacecraft, MITHRAS, the Heating facility at Ramfjordmoen, optical recording at Spitzbergen, Dynamic Explorer satellite passes, and rocket launches from ESRANGE. Toward end of 1981: 3-pulse experiments (1 ms long pulse for remote sites, moderate-length for monostatic ACF, short for power profile). **February–April 1982**: UHF transmitter overhaul by manufacturer to reach rated power and fix rapid frequency-switching problem. From May 1982: routine 24-hour CP runs and tests of new programs (mesospheric, Barker, multipulse). Single-pulse ACF and power profile options proven reliable; multipulse data being verified. Naturally occurring plasma lines observed with the French correlator (Kofman et al. 1982). Recent runs proved EISCAT correlator can detect natural plasma lines too. Strong heater-induced plasma lines observed Dec 1981 (Hagfors et al. 1983). Mesospheric echoes recorded during disturbed conditions; clear-air turbulence echoes seen at all three stations, 25–30 km altitude. Barker code worked; Barker multipulse for extreme height resolution hampered by low SNR.

**KR**: 최초 스펙트럼: **1981년 6월**, 단일펄스, 고정 방향, 첨두 500 kW. 1981년 가을 활동 증가, 11–12월에 GEOS 위성, MITHRAS, Ramfjordmoen Heating, Spitzbergen 광학, Dynamic Explorer, ESRANGE 로켓과 동시 관측으로 정점. 1981년 말에는 3펄스 실험(1 ms 장펄스를 원격, 중간 길이로 monostatic ACF, 단펄스로 전력 프로파일). **1982년 2–4월**: 제조사 엔지니어팀이 UHF 송신기를 정밀 점검해 정격 출력 도달 및 빠른 주파수 전환 문제 해결. 1982년 5월부터 24시간 CP 정기 운용과 신규 프로그램(중간권·Barker·multipulse) 시험. 단일펄스 ACF와 전력 프로파일은 신뢰성 입증; 다중펄스는 검증 진행 중. 프랑스 상관기로 자연 플라스마선 관측(Kofman et al. 1982). 최근 EISCAT 상관기도 자연 플라스마선 검출 가능 입증. 1981년 12월 가열기 유도 강한 플라스마선 관측(Hagfors et al. 1983). 교란기에는 중간권 에코 기록; 25–30 km의 청천난류 에코는 세 사이트 모두에서 관측. Barker 부호 정상 동작; 극한 고도 분해능을 위한 Barker multipulse는 SNR 부족으로 어려움.

#### §4.2 Examples (p. 876–878, Figures 6–10)

**EN**:
- **Figure 6**: Combined-mode test transmission cycle (~1 ms total). Five frequencies: $f_8$ for remote sites, $f_7$ ACF monostatic, $f_9$ power profile monostatic, $f_8$ multipulse monostatic, $f_{10}$ for plasma-line. Demonstrates flexibility of mode mixing.
- **Figure 7**: Single-pulse ACFs and power spectra for **eight 30 km range gates** (Ramfjordmoen, 12 July 1982, 1854 UT, 1 MW pulse power, AZ 180° EL 77.8°). Real and imaginary parts of ACFs from 200 µs single pulses; nonzero imaginary indicates ionosphere drifting along $\vec{B}$, with reversal of longitudinal drift in 250–300 km range. ACFs shown without correction for falloff with distance or finite pulse length.
- **Figure 8**: Sequence of power profiles measured with **80 µs pulse**, corrected for inverse-square falloff. 3-min integration each. **$E$ layer dynamic**: maximum density increased and decayed by factor of 2 over 12-min display.
- **Figure 9**: Multipulse ACFs with power spectra at **3 km range resolution** (1554 UT). Real-part time series and corresponding spectra demonstrate the multipulse advantage. Caveat: contiguous elemental pulses create lag-1 from unwanted heights — a filter-dependent effect under assessment.
- **Figure 10**: **Naturally occurring plasma lines** detected with EISCAT correlator using 330 µs pulse for ACF and 40 µs for power profile, second LO offset by 4.4 MHz to bring plasma lines into the 100 kHz low-pass filter band. Plasma line clearly seen in first range gate; signal **an order of magnitude smaller** than 130 K noise injection. Power profile at 6 km range resolution overlaid below.

**KR**:
- **그림 6**: 복합 모드 시험 송신 주기(약 1 ms). 5 주파수: $f_8$ 원격용, $f_7$ ACF monostatic, $f_9$ 전력 프로파일 monostatic, $f_8$ multipulse monostatic, $f_{10}$ 플라스마선용. 모드 혼합의 유연성 시연.
- **그림 7**: **8개 30 km 거리 게이트**에서의 단일펄스 ACF·전력 스펙트럼(Ramfjordmoen, 1982-07-12, 1854 UT, 1 MW, AZ 180° EL 77.8°). 200 µs 단일펄스의 ACF 실수·허수; 0 아닌 허수는 $\vec{B}$ 방향 이온 표류를 의미하며 250–300 km에서는 종방향 표류가 반전. 거리 감쇠·유한 펄스 보정은 미적용.
- **그림 8**: **80 µs 펄스**로 측정된 전력 프로파일 시퀀스, 거리 제곱 감쇠 보정 후. 각 적분 3분. **$E$ 층 동역학**: 최대 밀도가 12분 동안 2배 증가·감소.
- **그림 9**: **3 km 거리 분해능**의 multipulse ACF·전력 스펙트럼(1554 UT). 실수부 시계열과 스펙트럼이 multipulse 이점 시연. 단점: 연속 원소 펄스가 원치 않은 고도에서 lag-1 신호를 생성 — 수신기 필터 의존적 효과로 검토 중.
- **그림 10**: **자연 발생 플라스마선** EISCAT 상관기 검출. ACF 330 µs 펄스 + 전력 프로파일 40 µs 펄스, 두 번째 LO를 4.4 MHz 오프셋해 플라스마선을 100 kHz 저역 필터 대역으로 이동. 첫 거리 게이트에서 플라스마선 명확; 신호 강도는 130 K 잡음 주입 대비 **약 10배 작음**. 6 km 거리 분해능 전력 프로파일도 하단에 함께.

#### §4.3 Additional geophysical installations (p. 878)

**EN**: Co-located instruments: ionosondes, riometers, magnetometers, optical instruments at and around the EISCAT sites. **Heating facility** at Ramfjordmoen (Stubbe & Kopka 1979) — high-power 2.5–8.0 MHz pulsed/CW for ionospheric modification. **PRE** (partial reflection experiment, Holt et al. 1980) at Ramfjordmoen for $D$-region. **STARE** auroral electric field radar (Greenwald et al. 1978). **SABRE** (Nielsen 1980). Two rocket ranges (Andøya 69.2°N 16.0°E; ESRANGE 67.9°N 21.1°E) within EISCAT field of view enable comparative ISR-in-situ studies of ion composition, particle fluxes, electric fields, irregularities. EISCAT can provide real-time ionospheric measurements for rocket launch decisions.

**KR**: 동지점 계측기: 이온소드, 리오미터, 자력계, 광학기기 다수가 EISCAT 사이트 주변에 분포. **Heating 시설**(Ramfjordmoen, Stubbe & Kopka 1979) — 2.5–8.0 MHz 고출력 펄스/CW로 전리권 변조. **PRE**($D$ 영역 부분반사 실험, Holt et al. 1980, Ramfjordmoen). **STARE** 오로라 전기장 레이더(Greenwald et al. 1978). **SABRE**(Nielsen 1980). EISCAT 시야 내 두 로켓 발사장(Andøya 69.2°N 16.0°E; ESRANGE 67.9°N 21.1°E) — ISR과 in-situ 비교 연구(이온 조성, 입자 유속, 전기장, 이상현상)를 가능하게 하며, EISCAT은 로켓 발사 결정에 실시간 전리권 정보 제공.

---

## 3. Key Takeaways / 핵심 시사점

1. **Tristatic geometry is the headline scientific advantage / 삼정점 기하가 핵심 과학적 장점** — EN: One transmitter at Ramfjordmoen + three receivers (Ramfjordmoen, Kiruna, Sodankylä) observing a common scattering volume yields three independent line-of-sight Doppler velocities, which invert to the full 3-D ion-drift vector $\vec{v}_i$ — and hence the convection electric field $\vec{E} = -\vec{v}_i \times \vec{B}$ that drives ionosphere–magnetosphere coupling. KR: 송신기 1대(Ramfjordmoen) + 수신기 3대(Ramfjordmoen·Kiruna·Sodankylä)가 공통 산란체적을 관측하여 3개의 시선 도플러를 측정 → 3차원 이온 표류 $\vec{v}_i$ → 자기권–전리권 결합을 구동하는 대류 전기장 $\vec{E} = -\vec{v}_i \times \vec{B}$ 직접 도출.

2. **UHF and VHF are complementary, not redundant / UHF와 VHF는 상호보완** — EN: UHF (933.5 MHz, 32 m dish, 2 MW) is optimised for ion-line measurements and tristatic operation; VHF (224 MHz, parabolic cylinder, 5 MW) reaches further into mesosphere ($D$-region) and resolves plasma lines better thanks to longer pulse capability and larger collecting area. KR: UHF(933.5 MHz, 32 m, 2 MW)는 이온선·tristatic용, VHF(224 MHz, 원통형, 5 MW)는 더 큰 유효 면적과 긴 펄스로 중간권($D$ 영역)과 플라스마선에 유리.

3. **Custom hardware correlator is the engineering crown jewel / 맞춤형 하드웨어 상관기가 공학적 백미** — EN: A purpose-built 16-bit programmable digital correlator with pipeline architecture, 5 MHz clock, 4096-word ping-pong buffer, 2048-word result memory, and microprogrammable instruction store made real-time ACF computation possible at 1980s technology — the ancestor of every modern ISR digital backend. KR: 5 MHz 클럭, 4096 ping-pong 버퍼, 2048-word 결과 메모리, 마이크로프로그램 가능한 명령어 저장소를 가진 16-bit 파이프라인 디지털 상관기는 1980년대 기술로 실시간 ACF 연산을 가능하게 한 모든 현대 ISR 디지털 백엔드의 조상.

4. **CP/SP scheduling is a durable institutional model / CP/SP 시간 배정 모델의 지속성** — EN: Dividing observing time into Common Programmes (4 fixed scientific suites with shared data) and Special Programmes (proposal-based, 1-year proprietary) balances community access with PI-driven science. This model has become the de-facto standard for every major ISR consortium since. KR: 관측 시간을 공통 프로그램(4개 과학 묶음, 데이터 공유)과 특별 프로그램(제안 기반, 1년 독점)으로 이분하는 방식은 커뮤니티 접근성과 PI 주도 과학의 균형을 이루며, 이후 모든 주요 ISR 컨소시엄의 사실상 표준이 됨.

5. **Lag-weighting reveals the cost of finite pulses / 래그 가중치는 유한 펄스의 대가** — EN: When the receiver gate equals the pulse length, the matched filter introduces a triangular weighting in lag, so longer lags receive less weight and the effective range cell depends on lag number. The Ho et al. (1983) alternative gating (3× pulse-length window with cross-product expectation) keeps a uniform range cell across lags at the cost of widened height range — a trade well worth making for spectral fitting. KR: 거리 게이트가 펄스 길이와 같으면 매치드 필터가 래그에 삼각 가중을 도입해, 큰 래그는 가중치가 작고 유효 거리 셀이 래그 번호에 의존. Ho 등(1983)의 대안 게이팅(펄스 길이의 3배 창에서 교차곱 기댓값)은 고도 범위가 넓어지는 대가로 모든 래그가 같은 거리 셀을 공유 — 스펙트럼 적합에 매우 유리한 거래.

6. **Polarization control matches Faraday rotation per site / 사이트별 편파 제어로 패러데이 회전 보상** — EN: The motorised polarizer at the UHF feed allows arbitrary RHCP/LHCP/elliptical/linear synthesis, vital because Faraday rotation along the slant path differs at the three remote receivers. Without per-site polarization optimisation, link budgets at the bistatic legs would suffer. The VHF system can flip polarization in Mode I (dual klystron) but not Mode II. KR: UHF 급전부의 모터 구동 편파기는 임의 RHCP/LHCP/타원/직선 편파 합성 가능 — 세 원격 수신기의 사선 경로별 패러데이 회전이 달라 사이트별 편파 최적화가 필수. VHF는 Mode I(이중 클라이스트론)에서 편파 전환 가능, Mode II에서는 불가.

7. **Quantitative engineering numbers fix the design space / 정량적 공학 수치가 설계 공간을 고정** — EN: 12.5% duty cycle (avg/peak = 250 kW/2 MW UHF, 625 kW/5 MW VHF), pulse range 10 µs–10 ms (UHF) / 10 µs–1 ms (VHF), 0–1000 Hz PRF, $T_{\rm sys}$ 40 K cooled / 120–150 K uncooled, 8-bit A/D at 500 kHz, 5 MHz correlator clock, 1 µs timing accuracy, $10^{-11}$ Cs-standard frequency stability. These numbers define what experiments are physically possible. KR: 듀티 12.5%(UHF 250 kW/2 MW, VHF 625 kW/5 MW), 펄스 10 µs–10 ms(UHF) / 10 µs–1 ms(VHF), PRF 0–1000 Hz, $T_{\rm sys}$ 40 K(냉각) / 120–150 K(비냉각), 500 kHz 8-bit A/D, 5 MHz 상관기 클럭, 1 µs 타이밍, Cs 표준 $10^{-11}$ — 이 수치들이 어떤 실험이 물리적으로 가능한지 정의.

8. **Multi-instrument ecosystem multiplies scientific return / 다중계측 생태계가 과학적 산물 배가** — EN: Co-location with the Heating facility, PRE, STARE, SABRE, two rocket ranges (Andøya, ESRANGE), and satellite passes (GEOS, Dynamic Explorer) makes EISCAT not just an isolated radar but the centre of a high-latitude observational ecosystem. The 1981 November–December campaign concurrently with all of these is the prototypical multi-instrument auroral campaign. KR: Heating 시설, PRE, STARE, SABRE, 두 로켓 발사장(Andøya, ESRANGE), 위성 패스(GEOS, Dynamic Explorer)와의 동지점 운용은 EISCAT을 고립된 레이더가 아닌 고위도 관측 생태계의 중심으로 만든다. 1981년 11–12월 동시 캠페인이 다중계측 오로라 캠페인의 원형.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Radar equation / 레이더 방정식

EN/KR: For a backscattering volume $V$ at range $R$ illuminated by a transmitter of power $P_t$ with antenna effective area $A_e$ and gain $G_t$:

$$P_r = \frac{P_t G_t A_e}{(4\pi)^2 R^4} \, \sigma_{\rm IS} N_e V$$

For incoherent scatter the volumetric cross-section per electron is

$$\sigma_{\rm IS} \approx \frac{\sigma_T}{1 + T_e/T_i}, \qquad \sigma_T = \frac{8\pi}{3}\left(\frac{e^2}{4\pi\varepsilon_0 m_e c^2}\right)^2 \approx 6.65\times10^{-29}\text{ m}^2$$

EN: $\sigma_T$ is the Thomson cross-section; the $(1+T_e/T_i)$ suppression arises from the ion-acoustic dressing of the Bragg-scale density spectrum (Salpeter 1960; Hagfors 1961). KR: $\sigma_T$는 톰슨 단면적; $(1+T_e/T_i)$ 억제는 Bragg 파장 밀도 스펙트럼의 이온 음파 dressing에서 비롯(Salpeter 1960; Hagfors 1961).

### 4.2 Wiener–Khinchin pair / 위너-킨친 쌍

$$\boxed{\, S(\omega) \;=\; \int_{-\infty}^{\infty} R(\tau)\, e^{-i\omega\tau}\, d\tau, \qquad R(\tau) \;=\; \frac{1}{2\pi}\int_{-\infty}^{\infty} S(\omega)\, e^{i\omega\tau}\, d\omega \,}$$

EN: EISCAT measures $R(\tau)$ in hardware (correlator) at discrete lags $\tau_n = n \Delta t$; $S(\omega)$ used for $T_e/T_i$ fitting comes from FFT.
KR: EISCAT 하드웨어 상관기는 $\tau_n = n\Delta t$ 이산 래그에서 $R(\tau)$ 측정; $T_e/T_i$ 적합용 $S(\omega)$는 FFT로 도출.

### 4.3 Lag-weighted ACF estimate (conventional gating) / 종래 게이팅의 래그 가중 ACF

For a rectangular transmitted pulse of length $\tau_p$ matched to a range gate of equal length, the expected correlation estimate at lag $\tau$ is

$$\hat R(\tau) = w(\tau)\, R_{\rm true}(\tau), \qquad w(\tau) = \tau_p - |\tau| \quad (|\tau| \le \tau_p)$$

EN: The triangular weight is the matched-filter result: contributions to lag $\tau$ come only from sample pairs within the pulse, which decreases linearly with $|\tau|$. Range resolution is $\Delta R = c\tau_p/2$.
KR: 삼각 가중은 매치드 필터 결과 — 래그 $\tau$에 기여하는 표본 쌍은 펄스 안에 동시에 들어 있어야 하므로 $|\tau|$에 따라 선형 감소. 거리 분해능 $\Delta R = c\tau_p/2$.

### 4.4 Alternative gating (Ho et al. 1983) / 대안 게이팅

Form lag products from samples within an interval $3\tau_p$:

$$\hat R(\tau) = \mathbb{E}\!\left[V(t_k)\, V^*(t_k+\tau)\right], \qquad t_k \in [t_0, t_0+3\tau_p]$$

EN: Although this widens the effective height window, only contributions actually within the transmitted pulse correlate; the weighting then varies less drastically with lag and **all lags share the same range resolution** — important for spectral fitting since otherwise the range cell is lag-dependent.
KR: 유효 고도 창은 넓어지지만 송신 펄스 안에 들어온 표본만이 실제 상관에 기여; 가중치가 래그에 덜 의존하고 **모든 래그가 같은 거리 분해능**을 공유 — 그렇지 않으면 거리 셀이 래그별로 달라져 스펙트럼 적합에 불리.

### 4.5 Tristatic ion-velocity inversion / 삼정점 이온 속도 역변환

For a bistatic geometry with transmitter direction $\hat n_{\rm tx}$ and receiver direction $\hat n_{{\rm rx},i}$ (both pointing **from** the scatter volume), the Bragg vector is

$$\vec k_i = k_0 \,(\hat n_{\rm tx} + \hat n_{{\rm rx},i}), \qquad \hat k_i = \frac{\vec k_i}{|\vec k_i|}$$

The measured Doppler velocity at receiver $i$ is the projection of the ion drift onto $\hat k_i$:

$$v_{\rm los}^{(i)} = \hat k_i \cdot \vec v_i$$

Stacking three measurements gives the linear system

$$\underbrace{\begin{pmatrix} \hat k_1^{\,T} \\ \hat k_2^{\,T} \\ \hat k_3^{\,T} \end{pmatrix}}_{\mathbf K \,\in\, \mathbb R^{3\times 3}} \vec v_i = \begin{pmatrix} v_{\rm los}^{(1)} \\ v_{\rm los}^{(2)} \\ v_{\rm los}^{(3)} \end{pmatrix} \quad\Rightarrow\quad \vec v_i = \mathbf K^{-1} \vec v_{\rm los}$$

EN: For monostatic ($i=1$, Ramfjordmoen) $\hat n_{\rm tx}=\hat n_{\rm rx,1}$ so $\hat k_1 = \hat n_{\rm tx}$; for bistatic $i=2,3$ the Bragg direction bisects $\hat n_{\rm tx}$ and $\hat n_{{\rm rx},i}$. PLAN (Williams 1982) builds $\mathbf K$ from antenna pointings and inverts on the fly.
KR: monostatic($i=1$, Ramfjordmoen)에서 $\hat n_{\rm tx}=\hat n_{\rm rx,1}$이므로 $\hat k_1 = \hat n_{\rm tx}$; bistatic($i=2,3$)에서는 $\hat n_{\rm tx}$와 $\hat n_{{\rm rx},i}$의 이등분선 방향. PLAN(Williams 1982)이 안테나 지향에서 $\mathbf K$를 구축하고 실시간 역변환.

### 4.6 Convection electric field / 대류 전기장

$$\boxed{\,\vec E_\perp = -\vec v_i \times \vec B \,}$$

EN: With $|\vec B| \approx 5\times 10^{-5}$ T at high latitudes, an ion drift of 1 km/s gives $|\vec E_\perp| = 50$ mV/m — typical auroral magnitudes.
KR: 고위도 $|\vec B| \approx 5\times 10^{-5}$ T에서 1 km/s 표류는 $|\vec E_\perp| = 50$ mV/m — 전형적 오로라 크기.

### 4.7 Multipulse ACF / 다중펄스 ACF

EN: A multipulse waveform is a sequence of $M$ short subpulses with carefully chosen interpulse delays $\{T_1, T_2, \ldots, T_{M-1}\}$ such that the **lag set** $\{T_k - T_j : k>j\}$ covers all desired lags **uniquely**. Each lag is then estimated from the cross-correlation of the corresponding pair, preserving the short-pulse range resolution while sampling many lags simultaneously. EISCAT's general waveform combines this with single-pulse ACFs and power profiles.
KR: 다중펄스 파형은 $M$ 개의 짧은 부펄스 시퀀스로, 펄스 간 지연 $\{T_1, T_2, \ldots, T_{M-1}\}$이 **래그 집합** $\{T_k - T_j : k>j\}$가 원하는 모든 래그를 **유일하게** 덮도록 선택됨. 각 래그는 해당 부펄스 쌍의 교차상관으로 추정 — 짧은 펄스의 거리 분해능을 유지하면서 동시에 많은 래그를 표본화. EISCAT의 범용 파형은 이를 단일펄스 ACF·전력 프로파일과 결합.

### 4.8 Antenna gain and beamwidth / 안테나 이득과 빔폭

$$G = \frac{4\pi A_e}{\lambda^2}, \qquad \theta_{\rm HPBW} \approx \frac{70\lambda}{D} \text{ degrees}$$

EN: At UHF $\lambda = 32$ cm, $D=32$ m gives $\theta_{\rm HPBW} \approx 0.7°$ (matches the quoted 0.6°), $A_e = 0.65 \times \pi (16)^2 \approx 522$ m² and $G \approx 6.4\times 10^4 \approx 48$ dB. At VHF $\lambda = 1.34$ m, the parabolic-cylinder effective area 3330 m² gives $G \approx 4\pi A_e/\lambda^2 \approx 2.3\times 10^4 \approx 44$ dB.
KR: UHF에서 $\lambda = 32$ cm, $D = 32$ m $\Rightarrow$ $\theta_{\rm HPBW} \approx 0.7°$(보고치 0.6°와 일치), $A_e = 0.65 \times \pi (16)^2 \approx 522$ m², $G \approx 6.4\times 10^4 \approx 48$ dB. VHF에서 $\lambda = 1.34$ m, 유효면적 3330 m² $\Rightarrow$ $G \approx 4\pi A_e/\lambda^2 \approx 2.3\times 10^4 \approx 44$ dB.

### 4.9 Numerical worked example: SNR for CP 0 / 수치 예: CP 0의 SNR

EN: CP 0 short pulse: $\tau_p = 67$ µs ($\Delta R = 10$ km), $P_t = 2$ MW, $G_t = 6.4\times 10^4$, $A_e = 522$ m². At $R = 300$ km in $F$-region with $N_e = 10^{11}$ m$^{-3}$, $T_e = T_i = 1500$ K so $\sigma_{\rm IS}/\sigma_T = 0.5$:

$$P_r \approx \frac{(2\times 10^6)(6.4\times 10^4)(522)}{(4\pi)^2 (3\times 10^5)^4} \cdot 0.5 \cdot (6.65\times 10^{-29}) \cdot (10^{11}) \cdot V$$

with $V = A_{\rm beam} \cdot \Delta R$, beam cross-section $A_{\rm beam} \approx \pi (R \theta_{\rm HPBW}/2)^2 = \pi(300\text{ km}\cdot 0.005)^2 \approx 7\text{ km}^2 = 7\times 10^6$ m². So $V \approx 7\times 10^{10}$ m³.
Plugging in: $P_r \approx 2.3 \times 10^{-15}$ W. With $T_{\rm sys} = 40$ K, $B = 25$ kHz: $kT_{\rm sys}B \approx 1.4\times 10^{-17}$ W. Single-pulse SNR $\approx 165$. Integrating 1 s at 1 kHz PRF (1000 pulses, $\sqrt{N}$ improvement): SNR$_{\rm int}\sim 5200$. This explains why EISCAT can fit $T_e/T_i$ from a few seconds of data.
KR: CP 0 단펄스: $\tau_p = 67$ µs ($\Delta R = 10$ km), $P_t = 2$ MW, $G_t = 6.4\times 10^4$, $A_e = 522$ m². $F$ 영역 $R = 300$ km, $N_e = 10^{11}$ m$^{-3}$, $T_e = T_i = 1500$ K로 $\sigma_{\rm IS}/\sigma_T = 0.5$:

$P_r \approx 2.3\times 10^{-15}$ W. $T_{\rm sys} = 40$ K, $B = 25$ kHz에서 잡음 전력 $kT_{\rm sys}B \approx 1.4\times 10^{-17}$ W. 단펄스 SNR $\approx 165$. PRF 1 kHz로 1초 적분(1000 펄스, $\sqrt{N}$): SNR$_{\rm int} \sim 5200$. 그래서 EISCAT은 수 초 자료로 $T_e/T_i$ 적합 가능.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1958 ─── Gordon proposes IS theory; Bowles (Long Branch, NJ) demonstrates at 41 MHz
1960 ─── Salpeter / Dougherty-Farley: spectrum theory with ion-acoustic feature
1961 ─── Hagfors: backscatter spectrum closed form (foundational EISCAT-co-author paper)
1963 ─── Jicamarca 50 MHz (equatorial) operational
1964 ─── Arecibo 430 MHz operational (largest ISR in the world)
1969 ─── Millstone Hill 440 MHz operational (mid-latitude)
1971 ─── Chatanika 1290 MHz operational (auroral, Alaska — predecessor concept for EISCAT)
1974 ─── du Castel & Testud: original EISCAT design concept paper
1975 ─── St. Santin tristatic operational (mid-latitude, France)
1978 ─── Rishbeth EISCAT review (Esrange Symposium)
1980 ─── EISCAT UHF system installed at Ramfjordmoen
1981 ─── First EISCAT spectra (June); November-December multi-instrument campaign
1982 ─── UHF achieves rated power; Hagfors Nobel Symposium overview
1983 ─── ★ THIS PAPER ★ definitive engineering description (Folkestad et al.)
1985 ─── VHF system operational
1990 ─── Sondrestrom (Greenland) ISR operational
1996 ─── EISCAT Svalbard Radar (ESR) commissioned at 78°N
1996 ─── GUISDAP analysis package (Lehtinen & Huuskonen) — descendant of §3.4 here
2007 ─── PFISR (Poker Flat) — first phased-array ISR
2013 ─── RISR-N/-C (Resolute Bay) — Arctic phased-array twins
2027+ ── EISCAT_3D (under construction) — multistatic, MIMO, fully digital
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Gordon (1958) "Incoherent scattering of radio waves by free electrons" | EN: First proposed IS as a remote-sensing technique. KR: IS 원격 감지 기법을 처음 제안. | EN: The theoretical seed of every paper in this lineage. KR: 이 계보의 모든 논문의 이론적 씨앗. |
| Bowles (1958) "Observation of vertical incidence scatter from the ionosphere at 41 Mc/sec" | EN: First IS detection at Long Branch, NJ. KR: Long Branch에서의 IS 첫 검출. | EN: Empirical proof that the cross-section is large enough to be useful. KR: 단면적이 실용적으로 충분함을 입증. |
| Salpeter (1960); Dougherty & Farley (1960) | EN: Worked out the IS spectrum with ion-acoustic dressing. KR: 이온 음파 dressing이 포함된 IS 스펙트럼 도출. | EN: Theory underlying the $(1+T_e/T_i)^{-1}$ suppression and the double-humped ion line that EISCAT fits. KR: EISCAT이 적합하는 $(1+T_e/T_i)^{-1}$ 억제와 이중봉 이온선의 이론. |
| Hagfors (1961) "Density fluctuations in a plasma in a magnetic field" | EN: EISCAT co-author's foundational paper on magnetised IS spectrum. KR: EISCAT 공동저자의 자화 IS 스펙트럼 기초 논문. | EN: Provides the magnetic-field correction needed at high latitudes where $\vec B$ is far from horizontal. KR: $\vec B$가 거의 수직인 고위도에서 필요한 자기장 보정. |
| du Castel & Testud (1974) | EN: Original EISCAT design concept paper. KR: EISCAT 설계 개념 원안. | EN: The blueprint that this paper updates with as-built numbers. KR: 본 논문이 인수시험 후 수치로 갱신하는 청사진. |
| Evans (1975) "High-power radar studies of the ionosphere" (Proc. IEEE) | EN: Comprehensive ISR review of pre-EISCAT facilities. KR: EISCAT 이전 ISR 시설들의 종합 리뷰. | EN: Sets the comparison baseline; EISCAT's tristatic and high-latitude location are the new contributions. KR: 비교 기준을 설정 — EISCAT의 삼정점·고위도가 새 기여. |
| Rishbeth (1978) "EISCAT" Esrange Symposium | EN: Mid-construction EISCAT review. KR: 건설 중간 단계의 EISCAT 리뷰. | EN: Earlier snapshot to which §1 explicitly refers as the predecessor description. KR: §1이 선행 기술로 명시 인용하는 이전 스냅샷. |
| Hagfors (1982) Nobel Symposium 54 | EN: Co-author's own update prior to this paper. KR: 본 논문 직전의 공동저자 자체 업데이트. | EN: Companion overview from EISCAT scientific perspective. KR: EISCAT 과학적 관점의 동반 개요. |
| Hagfors et al. (1982) "EISCAT VHF antenna for incoherent scatter radar research" Radio Sci 17 | EN: Detailed design of the VHF parabolic cylinder. KR: VHF 원통형 파라볼라의 상세 설계. | EN: Companion engineering paper; §2.2 here cites it for full antenna details. KR: 동반 공학 논문; 본 논문 §2.2가 상세는 그 논문 인용. |
| Lehtinen & Huuskonen (1996) GUISDAP | EN: Modern EISCAT/ESR analysis package. KR: 현대 EISCAT/ESR 분석 패키지. | EN: Direct descendant of the Lejeune (1979, 1982) and Silén (1981) routines described in §3.4. KR: §3.4의 Lejeune·Silén 루틴을 계승. |
| Wannberg et al. (1997) ESR design | EN: EISCAT Svalbard Radar engineering paper. KR: EISCAT Svalbard Radar 공학 논문. | EN: Direct sibling — same consortium, polar-cap extension, applies all lessons of this paper. KR: 직계 형제 — 동일 컨소시엄의 극관 확장, 본 논문의 모든 교훈 적용. |
| McCrea et al. (2015) EISCAT_3D science case | EN: Multistatic phased-array EISCAT successor. KR: EISCAT_3D 다중정점 위상배열 후속. | EN: Inherits tristatic philosophy in fully digital MIMO form — direct descendant. KR: 본 논문의 삼정점 철학을 완전 디지털 MIMO 형태로 계승하는 직계 후손. |

---

## 7. References / 참고문헌

- Folkestad, K., Hagfors, T., and Westerlund, S. (1983). EISCAT: An updated description of technical characteristics and operational capabilities. *Radio Science*, 18(6), 867–879. DOI: 10.1029/RS018i006p00867
- Gordon, W. E. (1958). Incoherent scattering of radio waves by free electrons. *Proc. IRE*, 46, 1824–1829.
- Bowles, K. L. (1958). Observation of vertical incidence scatter from the ionosphere at 41 Mc/sec. *Phys. Rev. Lett.*, 1, 454–455.
- Salpeter, E. E. (1960). Electron density fluctuations in a plasma. *Phys. Rev.*, 120, 1528–1535.
- Hagfors, T. (1961). Density fluctuations in a plasma in a magnetic field. *J. Geophys. Res.*, 66, 1699–1712.
- du Castel, F., and Testud, J. (1974). Some aspects of the design concept of a European incoherent scatter facility (EISCAT project). *Acta Geophys. Pol.*, 22, 113–119.
- Evans, J. V. (1975). High-power radar studies of the ionosphere. *Proc. IEEE*, 63, 1636–1650.
- Rishbeth, H. (1978). EISCAT, in *Proc. Esrange Symposium*, ESA/ESTEC, 85–94.
- Alker, H. J. (1976). A design study of a multibit digital correlator for the EISCAT radar system. Thesis, Univ. Trondheim.
- Alker, H. J., Brattli, T., and Roaldsen, T. (1981). A programmable, high-speed correlator for the EISCAT radar system, Auroral Observatory, Tromsø.
- Hagfors, T., Kildal, P.-S., Kärcher, H. J., Liesenkötter, B., and Schroer, G. (1982). VHF parabolic cylinder antenna for incoherent scatter radar research. *Radio Sci.*, 17, 1607–1621.
- Hagfors, T. (1982). The EISCAT facility, in *Proc. Nobel Symposium 54*, Plenum.
- Hagfors, T., Aijänen, T., Kildal, P., and Kofman, W. (1983). Observations of heater-induced plasma lines with EISCAT. *Radio Sci.*, 18.
- Lehtinen, M., and Turunen, T. (1981). EISCAT UHF antenna direction calibration, Tech. Note 81/30, EISCAT.
- Williams, P. (1982). The use of the programme PLAN for EISCAT observations.
- Tørrustad, B. (1982). CORLAN (CORrelator LANguage), Tech. Note 82/36, EISCAT.
- Armstrong, J. (1980). EISCAT experiment preparation manual, Tech. Note 80/22, EISCAT.
- Turunen, T. (1982). RT-system, an overview, *Proc. EISCAT Annual Review Meeting*.
- Lejeune, G. (1982). EISCAT data analysis package, *Proc. EISCAT Annual Review Meeting*.
- Silén, J. (1981). Incoherent-scatter analysis package, *Proc. EISCAT Annual Review Meeting*.
- Stubbe, P., and Kopka, H. (1979). Ionospheric modification experiments in Northern Scandinavia: HEATING project, MPAE-W-02-79-04.
- Greenwald, R. A., Weiss, W., Nielsen, E., and Thomson, N. R. (1978). STARE: A new radar auroral backscatter experiment in northern Scandinavia. *Radio Sci.*, 13, 1021–1039.
- Lehtinen, M. S., and Huuskonen, A. (1996). General incoherent scatter analysis and GUISDAP. *J. Atmos. Terr. Phys.*, 58, 435–452.
- Wannberg, G. et al. (1997). The EISCAT Svalbard radar: A case study in modern incoherent scatter radar system design. *Radio Sci.*, 32, 2283–2307.
- McCrea, I. et al. (2015). The science case for the EISCAT_3D radar. *Prog. Earth Planet. Sci.*, 2, 21.
