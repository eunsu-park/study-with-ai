---
title: "Electron, Proton, and Alpha Monitor on the Advanced Composition Explorer Spacecraft"
authors: ["R. E. Gold", "S. M. Krimigis", "S. E. Hawkins III", "D. K. Haggerty", "D. A. Lohr", "E. Fiore", "T. P. Armstrong", "G. Holland", "L. J. Lanzerotti"]
year: 1998
journal: "Space Science Reviews 86, 541-562"
doi: "10.1023/A:1005088115759"
topic: Space_Weather
tags: [ACE, EPAM, instrument-paper, SEP, anisotropy, RTSW, LEMS, LEFS, composition-aperture, dE-E-telescope]
status: completed
date_started: 2026-04-25
date_completed: 2026-04-25
---

# 70. Electron, Proton, and Alpha Monitor on the Advanced Composition Explorer Spacecraft / ACE 위성의 전자, 양성자, 알파 입자 모니터

---

## 1. Core Contribution / 핵심 기여

**English.** Gold et al. (1998) is the **definitive instrument paper for EPAM**, the energetic particle monitor on NASA's Advanced Composition Explorer (ACE), launched 25 August 1997 to L1. EPAM was reconfigured from the flight-spare unit of the Ulysses HI-SCALE instrument and provides nearly full unit-sphere coverage of energetic ions (46 keV - 4.8 MeV) and electrons (40 keV - $\sim$350 keV) using **five solid-state detector telescopes** distributed between two stub-arm assemblies (2A: LEMS120 + LEFS60; 2B: LEMS30 + LEFS150 + CA60). Two LEMS (Low-Energy Magnetic Spectrometer) telescopes use a rare-earth magnet to deflect electrons so they measure ions only; two LEFS (Low-Energy Foil Spectrometer) telescopes use an aluminized Parylene foil to absorb low-energy ions so they measure electrons (plus high-energy "foil-protons"); the CA60 (Composition Aperture) is a $\Delta E \times E$ stack with a 4.8 µm thin detector $D$ and a 200 µm thick detector $C$ that uniquely identifies ion species (H, He, CNO, Fe groups) above $\sim$0.5 MeV/nuc. The instrument also includes a **32-channel logarithmic MFSA** spectrum accumulator and a **PHA system** with an adaptive priority scheme that boosts rare-species sampling. EPAM masses 11.8 kg (with bracket), uses 4.0 W, and has 1.5-6 s time resolution exploiting ACE's 12 s spin period.

**한국어.** Gold et al. (1998)은 1997년 8월 25일 L1으로 발사된 NASA Advanced Composition Explorer (ACE)의 에너지 입자 모니터인 **EPAM의 정식 instrument paper**이다. EPAM은 Ulysses의 HI-SCALE 비행 예비품(flight spare)을 재구성한 것으로, 두 개의 stub-arm 조립체(2A: LEMS120 + LEFS60; 2B: LEMS30 + LEFS150 + CA60)에 분산된 **5개의 solid-state detector 망원경**을 통해 에너지 이온(46 keV - 4.8 MeV)과 전자(40 keV - $\sim$350 keV)에 대한 거의 전 unit-sphere 커버리지를 제공한다. 두 개의 LEMS(Low-Energy Magnetic Spectrometer) 망원경은 희토류 자석으로 전자를 휘어내고 이온만 측정하며, 두 개의 LEFS(Low-Energy Foil Spectrometer) 망원경은 aluminized Parylene foil로 저에너지 이온을 흡수하고 전자(및 고에너지 "foil-protons")를 측정하고, CA60(Composition Aperture)은 4.8 µm 얇은 검출기 $D$와 200 µm 두꺼운 검출기 $C$로 구성된 $\Delta E \times E$ 스택으로 약 0.5 MeV/nuc 이상의 이온 종(H, He, CNO, Fe 그룹)을 명확히 식별한다. **32 채널 logarithmic MFSA** 스펙트럼 축적기와 희귀 종 sampling을 강화하는 적응형 우선순위 방식을 가진 **PHA 시스템**도 포함한다. 기기 질량은 브라켓 포함 11.8 kg, 소비전력 4.0 W, 시간 해상도는 ACE의 12초 spin 주기를 활용하여 1.5-6초이다.

The paper's significance extends well beyond instrument heritage: EPAM is the **only ACE instrument that measures electrons** and the only one with full unit-sphere angular coverage, making it indispensable for SEP onset detection, anisotropy studies, and shock-acceleration physics. Selected channels feed the NOAA Real-Time Solar Wind (RTSW) system 24/7, providing the de-facto operational L1 baseline for space weather alerts that has supported civilian and military operations for over a quarter century. / 본 논문의 의의는 단순한 기기 계보를 넘어선다: EPAM은 **ACE에서 전자를 측정하는 유일한 기기**이자 전 unit-sphere 각도 커버리지를 가진 유일한 기기로서, SEP onset 감지, anisotropy 연구, 충격파 가속 물리에 없어서는 안 될 도구이다. 선택된 채널들은 NOAA Real-Time Solar Wind (RTSW) 시스템에 24/7 공급되어 25년 이상 민·군 운영을 지원해온 사실상 표준의 L1 우주환경 baseline이 되고 있다.

---

## 2. Reading Notes / 읽기 노트

### Part I: Science Objectives (Sec. 1, pp. 541-543) / 과학적 목표

**English.** EPAM's primary mission is to support the high-sensitivity ACE composition spectrometers (CRIS, SIS, SEPICA, SWIMS, SWICS, ULEIS) by providing the **interplanetary context** — fast detection of energetic events, anisotropy information, and broad spectral coverage. Energy range: 50 keV to 5 MeV ions, 40 keV to 350 keV electrons. The Composition Aperture (CA) is a $\Delta E \times E$ telescope that resolves ion species groups (H, He, CNO, Fe) above 0.5 MeV/nuc. Spatial coverage uses five telescopes oriented at various angles to the spacecraft spin axis, sampling nearly the full unit sphere.

EPAM is also part of the NASA/NOAA **Real-Time Solar Wind (RTSW)** system providing 24-hr space weather coverage. It is the only ACE instrument that:
1. Observes the full unit-sphere distribution of particle fluxes (3-D angular coverage),
2. Measures **electrons** (40 keV - 350 keV),
3. Has the highest time resolution (1.5-6 s) of the energetic particle suite,
4. Has dynamic range to handle small upstream events through the most intense flares and shock spikes.

Specific science targets discussed: (i) acceleration and propagation in solar energetic particle (SEP) events spanning the prompt ($\geq$ 5 MeV/nuc) to slow ($\sim$50 keV) regimes — a 32-channel MFSA can track velocity dispersion; (ii) interplanetary shock acceleration (50 keV - few MeV) where 3-D anisotropy reveals true particle flow patterns; (iii) Co-rotating Interaction Regions (CIRs) accelerating ions primarily beyond 1 AU but flowing back inside; (iv) upstream magnetospheric particle bursts observed both en route to L1 and in steady operation; (v) full radial/latitudinal context with the identical-design HI-SCALE on Ulysses.

**한국어.** EPAM의 주요 임무는 ACE의 고감도 조성 분광기(CRIS, SIS, SEPICA, SWIMS, SWICS, ULEIS)를 지원하기 위해 **행성간 맥락(interplanetary context)** — 에너지 이벤트의 빠른 감지, anisotropy 정보, 넓은 스펙트럼 커버리지 — 을 제공하는 것이다. 에너지 범위: 이온 50 keV ~ 5 MeV, 전자 40 keV ~ 350 keV. CA는 $\Delta E \times E$ 망원경으로 0.5 MeV/nuc 이상의 이온 종 그룹(H, He, CNO, Fe)을 분해한다. 공간적 커버리지는 위성 spin 축에 대해 다양한 각도로 배치된 5개 망원경이 거의 전 unit-sphere를 sampling함으로써 달성된다.

EPAM은 또한 NASA/NOAA **Real-Time Solar Wind (RTSW)** 시스템의 일부로서 24시간 우주환경 감시를 제공한다. ACE 기기 중 EPAM만이 갖는 특성:
1. 전 unit-sphere 입자 플럭스 분포 관측 (3-D 각도 커버리지),
2. **전자** (40 keV - 350 keV) 측정,
3. 에너지 입자 기기 중 최고 시간 해상도 (1.5-6 s),
4. 작은 상류 이벤트부터 가장 강한 플레어 및 충격파 spike까지 처리하는 dynamic range.

논의된 구체적 과학 목표: (i) prompt ($\geq$ 5 MeV/nuc)에서 slow ($\sim$50 keV)까지의 SEP 가속·전파 — 32 채널 MFSA로 velocity dispersion 추적; (ii) 행성간 충격파 가속(50 keV - 수 MeV) — 3-D anisotropy로 실제 입자 흐름 패턴 규명; (iii) 1 AU 너머에서 주로 가속되어 안쪽으로 흘러들어오는 CIR(Co-rotating Interaction Region) 이온; (iv) L1 도달 경로 및 정상 운용 중 관측되는 자기권 상류 입자 burst; (v) 동일 설계의 Ulysses HI-SCALE과 함께 radial/latitudinal 전체 맥락 제공.

### Part II: Instrument Configuration & Detector Geometry (Sec. 2, 2.1, pp. 543-547) / 기기 구성 및 검출기 기하

**English.** Mechanically, EPAM consists of two stub-arm telescope assemblies labeled **2A** and **2B**, mounted on a support bracket that also encloses the electronics box. The assembly was elevated above the ACE top deck and placed between two solar arrays for unobstructed field of view (necessitated because the original HI-SCALE was designed for Ulysses' boxy structure). EPAM mass: 5.64 kg instrument + 6.13 kg bracket = **11.8 kg total**. Power: **4.0 W**. Data rate: **168 bits/s continuous**.

The five telescopes (Fig. 2 cylindrical projection):

| Name | Polar angle from spin axis | Sectors per spin | Function |
|---|---|---|---|
| **LEMS30** | 30° | 4 (A,B,C,D, each 90°) | Ions (magnet sweeps electrons) |
| **LEMS120** | 120° | 8 (1-8, each 45°) | Ions (magnet sweeps electrons) |
| **LEFS60** | 60° | 8 (a-h, each 45°) | Electrons (foil absorbs ions) |
| **LEFS150** | 150° | 4 (1,2,3,4, each 90°) | Electrons (foil absorbs ions) |
| **CA60** | 60° | 8, each 45° | Ion composition ($\Delta E \times E$) |

Geometric factors: LEMS 0.428 cm$^2$ sr, LEFS 0.397 cm$^2$ sr, CA 0.103 cm$^2$ sr (CA is reduced because the $\Delta E \times E$ stack requires tighter collimation to ensure path-length uniformity). Look-angles (full-cone): LEFS 53°, LEMS 51°, CA 45°. ACE rotates at $\sim$5 rpm with spin period $\sim$12 s, with the spin vector within 20° of the Sun (away from Earth). The 8-sector telescopes get angular resolution 45° at $\sim$1.5 s/sector; 4-sector ones get 90° at $\sim$3 s/sector.

**Detector technology:** every detector except $D$ in the CA is a totally depleted, silicon surface-barrier SSD of approximately 200 µm thickness; $D$ is a thin 4.8 µm epitaxial Si layer optimized for the $\Delta E$ measurement. The naming convention: LEMS30 uses detector $M$; LEMS120 uses $M'$; LEFS150 uses $F$ with $M$ as anti-coincidence; LEFS60 uses $F'$ with $M'$ as anti-coincidence; CA60 has $D \to C \to B$ depth order, with $B$ also serving as anti-coincidence for both CA and the deflected-electron channels of LEMS30 (where electrons swept by the magnet end up).

**한국어.** 기계적으로 EPAM은 **2A**와 **2B**로 명명된 두 stub-arm 망원경 조립체로 구성되며, 전자장비 박스를 함께 둘러싸는 support bracket에 장착된다. 이 조립체는 ACE 상부 데크 위로 들어올려져 두 태양전지판 사이에 배치되어 시야가 가려지지 않도록 했다(원래 HI-SCALE이 Ulysses의 직육면체 구조용으로 설계되었기에 필요한 변경). EPAM 질량: 기기 5.64 kg + 브라켓 6.13 kg = **총 11.8 kg**. 소비전력: **4.0 W**. 데이터율: **168 bps 연속**.

다섯 망원경(그림 2 원통 투영):

| 명칭 | spin 축 대비 polar angle | spin당 sector 수 | 기능 |
|---|---|---|---|
| **LEMS30** | 30° | 4 (A,B,C,D, 각 90°) | 이온 (자석이 전자 sweep) |
| **LEMS120** | 120° | 8 (1-8, 각 45°) | 이온 (자석이 전자 sweep) |
| **LEFS60** | 60° | 8 (a-h, 각 45°) | 전자 (foil이 이온 흡수) |
| **LEFS150** | 150° | 4 (1,2,3,4, 각 90°) | 전자 (foil이 이온 흡수) |
| **CA60** | 60° | 8, 각 45° | 이온 조성 ($\Delta E \times E$) |

기하인자: LEMS 0.428 cm$^2$ sr, LEFS 0.397 cm$^2$ sr, CA 0.103 cm$^2$ sr (CA는 $\Delta E \times E$ 스택의 경로길이 균일성을 위해 더 좁은 collimation 필요로 작음). Look-angle(full-cone): LEFS 53°, LEMS 51°, CA 45°. ACE는 약 5 rpm, spin 주기 약 12초로 회전하며, spin 벡터는 태양 방향에서 20° 이내(지구 반대편). 8 sector 망원경은 1.5 s/sector로 45° 각도 해상도, 4 sector 망원경은 3 s/sector로 90° 해상도를 갖는다.

**검출기 기술:** CA의 $D$를 제외한 모든 검출기는 약 200 µm 두께의 totally depleted Si surface-barrier SSD이고, $D$는 $\Delta E$ 측정에 최적화된 얇은 4.8 µm epitaxial Si 층이다. 명명 규칙: LEMS30은 $M$, LEMS120은 $M'$, LEFS150은 $F$ + $M$(anti-coincidence), LEFS60은 $F'$ + $M'$(anti-coincidence), CA60은 깊이 순으로 $D \to C \to B$이고 $B$는 CA의 anti-coincidence이자 LEMS30 자기장에 휘어진 전자 채널의 anti-coincidence 역할도 한다.

### Part III: LEMS/LEFS Detector Systems (Sec. 2.1.1, pp. 547-549) / LEMS/LEFS 검출기 시스템

**English.** LEFS works by foil discrimination: an aluminized Parylene foil, nominally **0.35 mg cm$^{-2}$**, is placed in front of detector $F$ (or $F'$). Range-energy physics dictates that ions below $\sim$350 keV cannot punch through (they have too high $dE/dx$ and short range), but electrons above $\sim$35 keV pass through (much smaller $dE/dx$). Thus $F$ measures electrons in the range $\sim$35-350 keV in three rate channels $E1, E2, E3$ (Table III). Above $\sim$400 keV, ions can also penetrate the foil; these are recorded as separate "foil proton" channels $FP4-FP7$, with energy inferred by adding the foil energy loss back. Pure electrons above $\sim$350 keV are recovered from the $B$ detector behind the magnet in the LEMS30 telescope (those that punched through $F$ AND triggered $M$ in anti-coincidence).

LEMS uses a rare-earth magnet (a "magnetic broom"): electrons below about **500 keV** are deflected by the magnetic field and never reach detector $M$. Therefore $M$ (and $M'$) responds purely to ions, in eight rate channels $P1$-$P8$ (LEMS30) or $P'1$-$P'8$ (LEMS120). The energy passbands span **0.046 to 4.700 MeV** for LEMS30 and **0.047 to 4.800 MeV** for LEMS120 (Table II).

The deflected electrons from LEMS30 are collected at the back of the CA60 assembly by detector $B$, which doubles as the CA's anti-coincidence detector. Because $B$ is so deep in the stack, it is not generally susceptible to ion contamination — making it a clean electron detector. The $B$ detector provides the LEMS30 electron channels DE1-DE4 covering 38-315 keV (Table V).

Logic equations (Tables II, III) are written with overbars denoting anti-coincidence. Examples:
- LEMS30 P3 channel: $M3\overline{4F}$ — particle deposits in $M$ above threshold 3, below threshold 4, AND $F$ does not fire. Passband **0.115-0.193 MeV**.
- LEFS150 E1 channel: $F1\overline{F2M}$ — i.e., $F$ deposits between threshold 1 and 2 with $M$ veto. Passband **0.044-0.058 MeV**.
- DE channels use $B$ thresholds with $C$ as veto, e.g. DE2 = $B2\overline{B3C}$ (53-103 keV).

**한국어.** LEFS는 foil 판별을 활용한다: 약 **0.35 mg cm$^{-2}$**의 aluminized Parylene foil이 검출기 $F$ (또는 $F'$) 앞에 놓인다. Range-energy 물리에 따라 약 350 keV 이하 이온은 foil을 통과하지 못하고($dE/dx$가 크고 사정거리 짧음), 약 35 keV 이상 전자는 통과한다($dE/dx$가 훨씬 작음). 따라서 $F$는 약 35-350 keV의 전자를 세 rate channel $E1, E2, E3$(표 III)에서 측정. 약 400 keV 이상에서는 이온도 foil을 관통하므로, 이들은 별도의 "foil proton" 채널 $FP4-FP7$에 기록되며 foil 에너지 손실을 더해 에너지를 추정. 약 350 keV 이상의 순수 전자는 LEMS30 망원경의 자석 뒤편 $B$ 검출기에서 회복($F$를 관통 + $M$ anti-coincidence).

LEMS는 희토류 자석("magnetic broom")을 사용: 약 **500 keV** 이하 전자는 자기장에 휘어 검출기 $M$에 도달하지 못한다. 따라서 $M$($M'$)은 순수 이온에 대해 8개 rate channel $P1$-$P8$ (LEMS30) 또는 $P'1$-$P'8$ (LEMS120)에 응답한다. 에너지 통과대역은 LEMS30이 **0.046 ~ 4.700 MeV**, LEMS120이 **0.047 ~ 4.800 MeV** (표 II).

LEMS30에서 휘어진 전자는 CA60 조립체 후방의 $B$ 검출기에 모이고, $B$는 CA의 anti-coincidence 역할도 한다. $B$가 스택 깊숙이 위치하므로 이온 오염에 일반적으로 노출되지 않아 깨끗한 전자 검출기가 된다. $B$는 38-315 keV의 LEMS30 전자 채널 DE1-DE4(표 V)를 제공한다.

논리식(표 II, III)은 상선(overbar)으로 anti-coincidence를 표시한다. 예:
- LEMS30 P3 채널: $M3\overline{4F}$ — 입자가 $M$에서 임계값 3 초과 4 미초과로 손실 AND $F$ 미발화. 통과대역 **0.115-0.193 MeV**.
- LEFS150 E1 채널: $F1\overline{F2M}$ — $F$가 임계값 1과 2 사이 손실, $M$ veto. 통과대역 **0.044-0.058 MeV**.
- DE 채널은 $B$ 임계값 + $C$ veto, 예: DE2 = $B2\overline{B3C}$ (53-103 keV).

### Part IV: MF Spectrum Accumulator (Sec. 2.1.2, p. 550) / MF 스펙트럼 축적기

**English.** The MFSA produces a **32-point logarithmically-spaced energy spectrum** for each LEMS/LEFS detector, providing finer energy resolution than the rate channels. It samples the same analog signal but routes it through a logarithmic amplifier and an 8-bit ADC. At the time of design, 8-bit ADCs needed $\sim$30 µs per conversion — too slow to process every event from the rate channels — so MFSA processes a subset.

Accumulation is scheduled (Table IV): four memory banks (I-IV) hold 32-bin spectra per sector. The 4-sector telescopes (LEMS30 = $M$, LEFS150 = $F$) need a single 128 s schedule to accumulate four sector spectra. The 8-sector telescopes (LEMS120 = $M'$, LEFS60 = $F'$) require two schedules. A complete cycle through all detector-sector combinations is **1024 s** (8 schedules × 128 s).

Practically, the MFSA gives clean energy spectra for fitting power laws ($j \propto E^{-\gamma}$), locating spectral breaks, and tracking velocity dispersion in SEP events. Figure 9 shows four MFSA spectra following the November 1997 solar event, illustrating the spectrum hardening at shock arrival (panel d).

**한국어.** MFSA는 각 LEMS/LEFS 검출기에 대해 **32 점 logarithmically 간격의 에너지 스펙트럼**을 생성하여 rate channel보다 더 정밀한 에너지 해상도를 제공한다. 동일한 아날로그 신호를 sampling하되 logarithmic 앰프와 8 bit ADC를 거친다. 설계 당시 8 bit ADC는 변환당 약 30 µs가 필요해 rate channel의 모든 이벤트를 처리할 수 없었고, 따라서 MFSA는 일부만 처리.

축적은 일정에 따라 진행(표 IV): 네 개의 메모리 뱅크(I-IV)가 sector별 32 bin 스펙트럼을 저장한다. 4 sector 망원경(LEMS30 = $M$, LEFS150 = $F$)은 단일 128 s 일정으로 4 sector 스펙트럼을 모두 축적. 8 sector 망원경(LEMS120 = $M'$, LEFS60 = $F'$)은 두 일정 필요. 모든 검출기-sector 조합 완성 주기는 **1024 s** (8 일정 × 128 s).

실용적으로 MFSA는 멱법칙 적합($j \propto E^{-\gamma}$), 스펙트럼 break 위치 결정, SEP 이벤트의 velocity dispersion 추적에 깨끗한 스펙트럼을 제공한다. 그림 9는 1997년 11월 태양 이벤트 후의 4개 MFSA 스펙트럼으로, 충격파 도착 시(패널 d) 스펙트럼이 hardening되는 것을 보여준다.

### Part V: Composition Aperture (Sec. 2.1.3, pp. 550-553) / 조성 입자 입구

**English.** The CA60 is the **species-identification telescope**, capable of unambiguous determination of ion composition via $\Delta E \times E$. Three detectors in series:
1. $D$ — thin 4.8 µm epitaxial Si: measures specific energy loss $\Delta E$.
2. $C$ — 200 µm totally depleted SSD: measures residual energy $E_{res}$.
3. $B$ — 200 µm SSD acting as anti-coincidence (and pickup of LEMS30 deflected electrons).

The CA's eight rate channels W1-W8 (Table V) are defined by slanted discriminators in the $D$-vs-$C$ plane that bound species-group regions:

| Channel | Logic | Passband (MeV/nuc) | Species | Z |
|---|---|---|---|---|
| W1 | $C1D1\overline{C2D2B}$ | 0.521-1.048 | H | 1 |
| W2 | $C2D1\overline{D4D2S1B}$ | 1.048-1.734 | H | 1 |
| W3 | $C1(D2+S1)\overline{C3D3B}$ | 0.389-1.278 | He | 2 |
| W4 | $C3D1S1\overline{C4D3B}$ | 1.278-6.984 | He | 2 |
| W5 | $C2D3\overline{C4D4B}$ | 0.546-1.831 | O (CNO) | 6-9 |
| W6 | $C4D2S2\overline{D4S3B}$ | 1.831-19.107 | O (CNO) | 6-9 |
| W7 | $C2D4\overline{C4B}$ | 0.298-0.955 | Fe | 10-28 |
| W8 | $C4D3S3\overline{B}$ | 0.955-92.663 | Fe | 10-28 |

Note: the species labels denote the **dominant** species in each group; for instance, the "O group" includes C, N, and O, while the "Fe group" covers $10 \leq Z \leq 28$ with Fe dominant.

The **$D$ detector also has DE channels** that count electrons that were magnetically deflected from LEMS30. DE1-DE4 cover 38-315 keV electrons (Table V).

**Adaptive priority for PHA (Table VI).** The $D$ and $C$ pulse-heights are digitized to 8 bits each (a 256×256 PHA matrix). With $\sim$30 µs per ADC conversion, only a subset of events can be telemetered. Two PHA events are downlinked per $\sim$1.5 s sector ($\sim$1.3 events/s average). Without intervention, abundant H and He would dominate; rare Fe and O would be undersampled. The adaptive priority scheme works as a **per-sector finite-state machine**: based on the **species group of the previous event** in that sector, a priority list is used for the current event:

| Previous → | H | He | O | Fe |
|---|---|---|---|---|
| Highest | Fe | H | He | O |
| 2nd | O | Fe | Fe | Fe |
| 3rd | He | O | O | He |
| Lowest | H | He | H | H |

So if the last event was H, the next priority is Fe > O > He > H — i.e. boost the rare species. The scheme is updated independently per sector. Of the two events downlinked per sector, only the **lower-priority** one drives the priority update for the next spin (so high-priority finds are not "rewarded" with reduced priority).

**한국어.** CA60은 **종 식별 망원경**으로 $\Delta E \times E$를 통해 이온 조성을 모호함 없이 결정한다. 직렬 3개 검출기:
1. $D$ — 얇은 4.8 µm epitaxial Si: 비저항 에너지 손실 $\Delta E$ 측정.
2. $C$ — 200 µm totally depleted SSD: 잔여 에너지 $E_{res}$ 측정.
3. $B$ — 200 µm SSD, anti-coincidence(및 LEMS30 휘어진 전자 픽업) 역할.

CA의 8개 rate channel W1-W8(표 V)은 $D$-$C$ 평면의 비스듬한 discriminator로 종 그룹 영역을 정의:

| 채널 | 논리 | 통과대역 (MeV/nuc) | 종 | Z |
|---|---|---|---|---|
| W1 | $C1D1\overline{C2D2B}$ | 0.521-1.048 | H | 1 |
| W2 | $C2D1\overline{D4D2S1B}$ | 1.048-1.734 | H | 1 |
| W3 | $C1(D2+S1)\overline{C3D3B}$ | 0.389-1.278 | He | 2 |
| W4 | $C3D1S1\overline{C4D3B}$ | 1.278-6.984 | He | 2 |
| W5 | $C2D3\overline{C4D4B}$ | 0.546-1.831 | O (CNO) | 6-9 |
| W6 | $C4D2S2\overline{D4S3B}$ | 1.831-19.107 | O (CNO) | 6-9 |
| W7 | $C2D4\overline{C4B}$ | 0.298-0.955 | Fe | 10-28 |
| W8 | $C4D3S3\overline{B}$ | 0.955-92.663 | Fe | 10-28 |

종 라벨은 각 그룹의 **dominant** 종을 의미한다; 예를 들어 "O group"은 C, N, O를 포함, "Fe group"은 $10 \leq Z \leq 28$에서 Fe가 dominant.

**$D$ 검출기는 LEMS30에서 자기적으로 휘어진 전자를 세는 DE 채널**도 가진다. DE1-DE4는 38-315 keV 전자를 커버(표 V).

**PHA 적응형 우선순위(표 VI).** $D$와 $C$의 pulse-height는 각각 8 bit로 디지털화(256×256 PHA matrix). ADC당 약 30 µs이므로 일부 이벤트만 전송 가능. sector당 약 1.5 s에 2개 PHA 이벤트 송출(평균 약 1.3/s). 개입이 없으면 풍부한 H, He가 dominant; 희귀한 Fe, O는 undersampling. 적응형 우선순위 방식은 **sector별 유한 상태 기계**로 작동: 같은 sector의 **이전 이벤트 종 그룹**에 따라 현재 이벤트의 우선순위 목록이 결정.

이전이 H였다면 다음 우선순위는 Fe > O > He > H — 즉 희귀 종 강화. 방식은 sector별로 독립 갱신. sector당 송출되는 두 이벤트 중 **낮은 우선순위** 이벤트만 다음 spin의 우선순위 갱신을 구동(고우선순위 발견이 우선순위 감소로 "보상"되지 않게 함).

### Part VI: Calibration (Sec. 3, pp. 553-554) / 보정

**English.** **Preflight (Sec. 3.1):** ground tests of EPAM exercised the basic instrument functions and electronic thresholds. The LEMS and LEFS telescopes were characterized using NASA Goddard's two particle accelerators: an electrostatic accelerator delivered protons (32-114 keV) and electrons (40-117 keV) to verify lower-energy thresholds; a Van de Graaff generator delivered protons (0.391-1.492 MeV) and helium ions (1.500-1.608 MeV) to stimulate higher-energy LEMS channels.

**Inflight (Sec. 3.2, Table VII):** EPAM has reclosable telescope covers containing radioactive sources. Bi-metallic springs attached to each cover slowly close it when commanded heaters fire, allowing inflight calibration to be performed:

| Telescope | Source | Strength (µCi) | Particle | Half-life |
|---|---|---|---|---|
| CA60 | $^{244}$Cm | 1.0 | 5.81 MeV α | 18.11 yr |
| CA60 | $^{148}$Gd | 0.16 | 3.18 MeV α | 75 yr |
| LEMS120 | $^{241}$Am | 1.0 | 5.49 MeV α + 59.5 keV X-ray | 432.2 yr |
| LEMS30 | $^{241}$Am | 1.0 | 5.49 MeV α + 59.5 keV X-ray | 432.2 yr |
| LEMS30 | $^{133}$Ba | 1.0 | 45 keV β + X-rays (31, 80, 302, 356 keV) | 10.5 yr |

The 5.486 MeV α from $^{241}$Am is observed in the highest LEMS channel (P8); the 60 keV X-rays in P1. The $^{133}$Ba in LEMS30 emits energetic electrons that enter the LEMS aperture and are deflected by the magnet into $B$ — this calibrates both the magnet deflection geometry and the deflected-electron channels DE1-DE4. The CA60 cover's $^{244}$Cm + $^{148}$Gd dual source produces He α at 5.8 and 3.18 MeV, observed in W3 and W4.

**한국어.** **발사 전(Sec. 3.1):** 지상 테스트로 기본 기능과 전자장비 임계값을 검증. LEMS/LEFS 망원경은 NASA Goddard의 두 입자가속기로 특성화: 정전 가속기는 양성자(32-114 keV)와 전자(40-117 keV)로 저에너지 임계값을 검증; Van de Graaff는 양성자(0.391-1.492 MeV)와 He(1.500-1.608 MeV)로 고에너지 LEMS 채널을 자극.

**비행 중(Sec. 3.2, 표 VII):** EPAM은 방사성 선원을 포함하는 reclosable 망원경 cover를 가진다. 각 cover에 부착된 bi-metallic 스프링이 명령된 히터가 작동하면 천천히 cover를 닫아 비행 중 보정을 수행할 수 있게 한다. (위 표 참조.)

$^{241}$Am의 5.486 MeV α는 LEMS의 최고 에너지 채널 P8에서, 60 keV X-선은 P1에서 관측. LEMS30의 $^{133}$Ba는 에너지 전자를 방출하여 LEMS 입구로 들어와 자석에 휘어 $B$로 들어감 — 이로써 자석 deflection 기하와 DE1-DE4 deflected-electron 채널을 동시 보정. CA60 cover의 $^{244}$Cm + $^{148}$Gd 이중 선원은 5.8 및 3.18 MeV He α를 생성하여 W3, W4에서 관측.

### Part VII: Inflight Performance — Ions and Electrons (Sec. 4.1, pp. 554-560) / 비행 중 성능

**English.** ACE was launched **25 August 1997** and EPAM turned on two days later. After a 75-min calibration test (cover release pyrotechnics fired), EPAM has performed continuously, immediately observing upstream magnetospheric events, solar events, and other interplanetary phenomena.

**12-day overview (Fig. 5, days 258-270 / 15-27 September 1997).** Six channels plotted: LEMS120 47-68 keV ions, LEMS120 0.59-1.06 MeV ions, CA60 0.39-1.28 MeV/nuc He, CA60 0.55-1.83 MeV/nuc CNO, CA60 0.30-0.96 MeV/nuc Fe, and CA60 38-53 keV electrons (DE1). The 47-68 keV LEMS120 channel shows numerous **upstream magnetospheric (MS)** spikes (LEMS120's 120° look angle gives an oblique view back at Earth's bow-shock region from L1). Roughly **100 upstream events** were observed in the first 30 mission days. The MS events have soft spectra and do not reach 500 keV (no signal in the 0.59-1.06 MeV channel). Beginning day 261, a **steep rise in 0.39-1.28 MeV/nuc helium** lasted until day 266, with simultaneous CNO and Fe enhancements — signature of a small solar event. The 38-53 keV electron channel triple-onsets coincide with helium and 0.587-1.060 MeV ion onsets, consistent with prompt SEP injection.

**Anisotropy event (Fig. 6, day 243 / 1 September 1997).** Eight LEMS120 sectors plotted for the 47-68 keV ion channel over $\sim$4 hours (10:00-14:30 UT). The event has four labeled sub-events A, B, C, D:
- **A** (just after 11:00 UT): rates rise to $\sim$100 c/s in **all sectors** uniformly for $\sim$10 min, then jump to $\sim$1000 c/s.
- **B** (around 11:30): rates in sectors **2, 3, 4** are very intense ($\sim$1000 c/s) while **opposite sectors 7, 8** show no enhancement — strong **anisotropy**.
- **C** ($\sim$12:00): small but clear timing difference between sectors. Sector 2 onset is just AFTER the event marker, sector 6 onset just BEFORE — time difference $\sim$**7 min**. This is consistent with particles arriving along magnetic field lines from a localized source.
- **D** ($\sim$13:25): nearly **isotropic** — rates similar in all 8 sectors.

**Pie plots (Fig. 7).** Polar plots in the spin plane at 12:00 UT (event B; strong anisotropy: sectors 3, 4 show $\sim$1000 c/s, sectors 7, 8 show no signal) and 13:25 UT (event D; nearly isotropic). This is a beautiful demonstration of EPAM's angular resolution and the **physical reality of magnetic-field-aligned (or shock-aligned) low-energy particle streaming**.

**Large solar event (Fig. 8, days 308-314 / 4-10 November 1997).** The 4 November X9 flare and associated CME produced a major SEP event at L1. Three groups of curves:
1. **Top — LEMS30 deflected electrons** in 4 channels (38-53, 53-103, 103-175, 175-315 keV), 1-hr-averaged: clean impulsive electron onset on day 308, decaying through day 311.
2. **Middle — LEMS120 ions** at 47-68, 115-195, 311-587, 1060-1900 keV: gradual rise extending to a strong **shock spike** later in the day on day 311 (the interplanetary CME-driven shock arrives).
3. **Bottom — CA species groups** at 521-1048 keV/nuc protons, 389-1278 keV/nuc He, 546-1831 keV/nuc CNO, 298-955 keV/nuc Fe: progressive intensity rises with delays consistent with velocity dispersion.

Onset is **clearly seen first in the energetic electrons** — they outrun the ions, exactly the relativistic electron / ion velocity-dispersion physics that gives the early-warning lead time of EPAM's RTSW data product.

**MFSA spectra (Fig. 9).** Four 32-channel spectra from the LEMS30 $M$ detector following the November 1997 event: (a) 308:09:41 (pre-event), (b) 308:14:14 (high-energy ions arriving while low-energy still pre-event = velocity dispersion), (c) 309:19:32 (extension to low energies), (d) 310:22:16 (shock spike: dramatic increase in low-energy particles, characteristic flat-then-falling shock spectrum).

**PHA matrix (Fig. 10).** A 12-hr accumulation (1997 day 311 06:00-18:00) of pulse-height matrix in the $C$ vs $D$ energy plane during the November 1997 flare peak. Distinct **hyperbolic tracks** corresponding to H, He, C, N, O, Ne, Mg, Si, Fe are visible. The boundaries of W1-W8 species-group rate channels (Table V) are overlaid as boxes. Fe is clearly resolved in W7-W8; Si, Mg, Ne, O, N, C are visible as separate tracks within W5-W6; He is in W3-W4 as a clean band; H is in W1-W2.

**한국어.** ACE는 **1997년 8월 25일** 발사되고 EPAM은 이틀 후 켜졌다. 75분 보정 테스트(cover 해제 pyrotechnic 작동) 이후 EPAM은 연속 가동되며 즉시 자기권 상류 이벤트, 태양 이벤트, 행성간 현상을 관측해왔다.

**12일 개요(그림 5, day 258-270 / 1997년 9월 15-27일).** 6개 채널: LEMS120 47-68 keV 이온, LEMS120 0.59-1.06 MeV 이온, CA60 0.39-1.28 MeV/nuc He, CA60 0.55-1.83 MeV/nuc CNO, CA60 0.30-0.96 MeV/nuc Fe, CA60 38-53 keV 전자(DE1). LEMS120 47-68 keV 채널은 다수의 **자기권 상류(MS)** spike를 보임(LEMS120의 120° look-angle은 L1에서 지구 bow-shock 영역을 비스듬히 본다). 임무 첫 30일간 약 **100개 상류 이벤트** 관측. MS 이벤트는 soft spectrum이며 500 keV에 미치지 못함(0.59-1.06 MeV 채널에 신호 없음). day 261부터 **0.39-1.28 MeV/nuc 헬륨의 가파른 상승**이 day 266까지 이어지고, CNO, Fe 동시 증가 — 작은 태양 이벤트의 특징. 38-53 keV 전자의 triple onset이 헬륨 및 0.587-1.060 MeV 이온 onset과 일치 — prompt SEP injection과 부합.

**Anisotropy 이벤트(그림 6, day 243 / 1997년 9월 1일).** LEMS120의 8 sector를 47-68 keV 이온 채널에 대해 약 4시간(10:00-14:30 UT) 플롯. 네 부 이벤트 A, B, C, D:
- **A** (11:00 직후): 약 10분간 **모든 sector에서 균일하게** ~100 c/s로 상승 후 ~1000 c/s로 점프.
- **B** (~11:30): sector **2, 3, 4**의 rate는 매우 강한 ~1000 c/s, **반대 sector 7, 8**은 enhancement 없음 — 강한 **anisotropy**.
- **C** (~12:00): sector 간 작지만 분명한 timing 차이. sector 2 onset이 이벤트 marker 직후, sector 6 onset이 직전 — 시간차 ~**7 분**. 국소 source로부터 자기력선을 따라 입자가 도착하는 것과 부합.
- **D** (~13:25): 거의 **isotropic** — 8 sector 모두 유사 rate.

**Pie plot(그림 7).** spin 평면의 polar plot, 12:00 UT(이벤트 B; 강한 anisotropy: sector 3, 4에서 ~1000 c/s, sector 7, 8은 신호 없음)와 13:25 UT(이벤트 D; 거의 isotropic). EPAM의 각도 해상도와 **자기장 정렬(또는 shock 정렬) 저에너지 입자 streaming의 물리적 실재**를 멋지게 보여줌.

**대형 태양 이벤트(그림 8, day 308-314 / 1997년 11월 4-10일).** 11월 4일 X9 플레어와 동반 CME가 L1에서 주요 SEP 이벤트를 생성. 세 그룹:
1. **상단 — LEMS30 deflected electron** 4 채널(38-53, 53-103, 103-175, 175-315 keV), 1시간 평균: day 308에 깨끗한 impulsive 전자 onset, day 311까지 감쇠.
2. **중단 — LEMS120 이온** 47-68, 115-195, 311-587, 1060-1900 keV: 점진적 상승, day 311 후반의 강한 **shock spike** (행성간 CME 충격파 도착).
3. **하단 — CA 종 그룹** 521-1048 keV/nuc 양성자, 389-1278 keV/nuc He, 546-1831 keV/nuc CNO, 298-955 keV/nuc Fe: velocity dispersion과 부합하는 지연을 가진 점진적 강도 상승.

onset은 **에너지 전자에서 가장 먼저 명확하게** 관측됨 — 전자가 이온보다 빠르게 이동하는 상대론적 전자/이온 velocity-dispersion 물리, 이것이 EPAM RTSW 데이터의 조기경보 lead time을 만든다.

**MFSA 스펙트럼(그림 9).** LEMS30 $M$ 검출기의 32 채널 스펙트럼 4개, 1997년 11월 이벤트 직후: (a) 308:09:41 (이벤트 전), (b) 308:14:14 (고에너지 이온 도착, 저에너지는 아직 이벤트 전 = velocity dispersion), (c) 309:19:32 (저에너지로의 확장), (d) 310:22:16 (shock spike: 저에너지 입자의 극적 증가, 충격파 spectrum의 특징적 평탄-후-감소 형태).

**PHA matrix(그림 10).** 1997년 day 311 06:00-18:00의 12시간 누적 pulse-height matrix, $C$ vs $D$ 평면, 11월 플레어 peak 동안. H, He, C, N, O, Ne, Mg, Si, Fe에 대응하는 뚜렷한 **쌍곡선 궤적** 가시. W1-W8 종 그룹 rate channel 경계(표 V)가 박스로 overlay됨. Fe는 W7-W8에서 명확히 분해; Si, Mg, Ne, O, N, C는 W5-W6 내 별도 궤적; He는 W3-W4의 깨끗한 띠; H는 W1-W2.

### Part VIII: Summary and Modern Operations Context (Sec. 5, pp. 558-559) / 요약 및 현대 운영 맥락

**English.** EPAM provides comprehensive energy, angular, and species coverage with good resolution over a parameter space critical for energetic-particle studies in the near-Earth interplanetary medium. It serves three roles: (1) the **monitor/context** instrument for the high-sensitivity ACE composition spectrometers, (2) one of the cornerstones of the NOAA RTSW system providing real-time space weather information broadcast worldwide (Zwickl et al. 1998), and (3) the **1 AU baseline** for radial/latitudinal gradient studies in conjunction with the identical-design HI-SCALE on Ulysses. The authors explicitly anticipate "a long-term, continuous data stream documenting the onset of the new solar cycle" — a prediction that has held: EPAM is still operating in 2026 with a $\sim$29-year continuous record covering Cycle 23, 24, and most of 25.

**한국어.** EPAM은 근지구 행성간 매질의 에너지 입자 연구에 핵심적인 매개변수 공간에서 종합적인 에너지·각도·종 커버리지와 좋은 해상도를 제공한다. 세 가지 역할: (1) ACE의 고감도 조성 분광기를 위한 **monitor/context** 기기, (2) 전 세계로 송출되는 실시간 우주환경 정보를 제공하는 NOAA RTSW 시스템의 핵심 (Zwickl et al. 1998), (3) Ulysses의 동일 설계 HI-SCALE과 함께 radial/latitudinal gradient 연구의 **1 AU baseline**. 저자들은 "새로운 태양 주기의 시작을 기록하는 장기 연속 데이터 스트림"을 명시적으로 기대 — 이 예측은 그대로 실현되어 EPAM은 2026년 현재까지 약 29년 연속 가동 중이며 Cycle 23, 24, 25 대부분을 포괄하는 기록을 보유한다.

---

## 3. Key Takeaways / 핵심 시사점

1. **Heritage flight: EPAM = HI-SCALE flight-spare reconfigured for ACE.** / **Heritage flight: EPAM은 HI-SCALE 비행 예비품의 ACE용 재구성.** EPAM did not start from a clean sheet — it is the Ulysses HI-SCALE flight-spare physically rebuilt for the ACE mounting interface. This significantly reduced cost and risk while inheriting two decades of LECP/HI-SCALE design heritage. The primary mechanical change was relocating the two telescope assemblies onto a bracket that elevates them above ACE's deck and between solar arrays. / EPAM은 백지에서 시작하지 않았다 — Ulysses HI-SCALE의 비행 예비품을 ACE 마운팅 인터페이스용으로 물리적으로 재구성한 것이다. 비용과 위험이 크게 감소하면서 LECP/HI-SCALE의 20년 설계 계보를 그대로 계승했다. 주요 기계적 변경은 두 망원경 조립체를 ACE 데크 위로 들어올린 브라켓에 재배치하고 태양전지판 사이에 위치시킨 것이다.

2. **Five-telescope geometry achieves nearly full unit-sphere coverage on a spinning platform.** / **5 망원경 기하로 spin 위성에서 거의 전 unit-sphere 커버리지 달성.** Telescopes at polar angles 30°, 60°, 60°, 120°, 150° from the spin axis sweep great-circle bands as the spacecraft rotates. With 4-or-8 angular sectors per spin, EPAM resolves 3-D ion anisotropy with $\geq 4 \pi$ steradian coverage — uniquely capable on ACE. The price is overlap (good for cross-calibration) and that no telescope points exactly along $\pm$z. / spin 축 대비 polar angle 30°, 60°, 60°, 120°, 150°의 망원경들이 위성 회전 시 대원(great circle) 띠를 sweep. spin당 4 또는 8 angular sector를 통해 EPAM은 $\geq 4\pi$ sr 커버리지로 3-D 이온 anisotropy를 분해 — ACE 내에서 유일하게 가능한 기능. 대가는 sector 간 overlap(상호보정에 유리)과 정확히 $\pm z$를 향하는 망원경이 없다는 점.

3. **Magnet vs. foil: dual technique cleanly separates ions and electrons.** / **자석 vs foil: 이중 기법이 이온과 전자를 깨끗이 분리.** LEMS uses a rare-earth magnet to deflect electrons below $\sim$500 keV out of the aperture (pure ion telescope); LEFS uses a 0.35 mg cm$^{-2}$ aluminized Parylene foil to absorb ions below $\sim$350 keV (electron telescope, with high-energy "foil-protons" as bonus). Together they yield the cleanest possible low-energy charged-particle measurements at L1. The deflected electrons in LEMS30 are not wasted — they are caught in the $B$ detector deep in the CA60 stack, providing pure-electron channels DE1-DE4 (38-315 keV). / LEMS는 희토류 자석으로 약 500 keV 이하 전자를 입구 밖으로 휘어내고(순수 이온 망원경), LEFS는 0.35 mg cm$^{-2}$ aluminized Parylene foil로 약 350 keV 이하 이온을 흡수한다(전자 망원경, 고에너지 "foil-protons"는 보너스). 둘이 함께 L1에서 가장 깨끗한 저에너지 하전입자 측정을 가능케 한다. LEMS30의 휘어진 전자도 버려지지 않는다 — CA60 스택 깊숙한 $B$ 검출기에 잡혀 38-315 keV의 순수 전자 채널 DE1-DE4를 만든다.

4. **$\Delta E \times E$ + adaptive PHA priority enables ion species ID despite limited bandwidth.** / **$\Delta E \times E$ + 적응형 PHA 우선순위가 제한된 대역폭에서도 이온 종 식별 가능케 함.** The CA60 with thin $D$ (4.8 µm) + thick $C$ (200 µm) + anti-coincidence $B$ produces a $\Delta E$-vs-$E_{res}$ matrix where H, He, C, N, O, Ne, Mg, Si, Fe form distinct hyperbolic tracks. Because the 8-bit ADC takes 30 µs per event, only ~1.3 events/s can be telemetered as PHA. Table VI's adaptive priority is a per-sector finite-state machine that boosts rare species (Fe, O) over abundant species (H, He) — a clever bandwidth-conservation technique that increased rare-species statistics by orders of magnitude over a naive scheme. / CA60의 얇은 $D$ (4.8 µm) + 두꺼운 $C$ (200 µm) + anti-coincidence $B$ 구성은 H, He, C, N, O, Ne, Mg, Si, Fe가 별개의 쌍곡선 궤적을 형성하는 $\Delta E$ vs $E_{res}$ matrix를 생성. 8 bit ADC가 이벤트당 30 µs 걸리므로 PHA로 송출 가능한 이벤트는 약 1.3/s. 표 VI의 적응형 우선순위는 sector별 유한 상태 기계로 풍부한 종(H, He) 위에 희귀 종(Fe, O)을 우선시 — naive 방식 대비 희귀 종 통계를 수십 배 증가시키는 영리한 대역폭 절약 기법.

5. **Time/spectral hierarchy: rate channels (high stats, coarse) + MFSA (low stats, fine).** / **시간/스펙트럼 계층: rate channel(통계 높음, 거침) + MFSA(통계 낮음, 정밀).** Rate channels (8 ion + 3 electron + 4 foil-proton + 4 deflected-electron) accumulate continuously for high statistics with octave-or-wider passbands. The MFSA logarithmic 32-channel spectrum runs on a 1024 s rotating schedule giving fine $\Delta\log E \approx 0.16$ resolution but with subset-of-events processing. Together they support both transient detection (rate channels) and detailed spectral fitting (MFSA). / rate channel(이온 8 + 전자 3 + foil-proton 4 + deflected-electron 4)은 통계 높은 연속 축적, octave 이상의 통과대역. MFSA logarithmic 32 채널 스펙트럼은 1024 s 회전 일정으로 작동하여 정밀 $\Delta\log E \approx 0.16$ 해상도를 가지나 일부 이벤트만 처리. 둘이 함께 transient 감지(rate channel)와 정밀 스펙트럼 적합(MFSA)을 모두 지원.

6. **Anisotropy revealed in "minutes-of-arc" of timing — Fig. 6 sector-2 vs sector-6 lag is $\sim$7 min.** / **수 분 단위의 timing에 anisotropy 드러남 — 그림 6의 sector-2 vs sector-6 lag은 약 7분.** Day 243 (1 Sept 1997) shows four upstream events with markedly different angular signatures: A is isotropic, B is sector-2,3,4-dominated, C has a 7-min onset lag between sectors, and D is again isotropic (1.5 hr after B). This demonstrates that EPAM resolves both **angular anisotropy** AND **fine timing differences** within a single event — both essential for distinguishing magnetically channeled flows from cross-field gradients and for tracking source location. / 1997년 9월 1일 (day 243)은 매우 다른 각도 특성을 가진 4개 상류 이벤트를 보여준다: A는 isotropic, B는 sector 2,3,4 우세, C는 sector 간 7분 onset lag, D는 다시 isotropic(B 후 1.5시간). 이는 EPAM이 단일 이벤트 내에서 **angular anisotropy**와 **정밀한 timing 차이**를 모두 분해함을 시연 — 자기적으로 channeled된 흐름과 cross-field gradient를 구별하고 source 위치를 추적하는 데 모두 필수.

7. **Electrons lead ions in SEP onsets — physical basis of EPAM's RTSW alert value.** / **SEP onset에서 전자가 이온보다 먼저 도착 — EPAM의 RTSW 경보 가치의 물리적 토대.** In Fig. 8 (Nov 1997 SEP), the energetic electron onset (38-53 keV through 175-315 keV channels in LEMS30 deflected-electrons) precedes the ion arrival, and a strong shock spike appears later in the day on day 311. Because near-relativistic electrons travel faster than mostly non-relativistic ions, EPAM's electron channels at L1 provide 10-60 min of advance warning before ions hit Earth's magnetosphere. This is the operational physics behind NOAA's SEP alerts. / 그림 8 (1997년 11월 SEP)에서 에너지 전자 onset(LEMS30 deflected-electron의 38-53 keV ~ 175-315 keV 채널)이 이온 도착에 앞서 일어나고, day 311 후반에 강한 shock spike가 출현. 거의 상대론적인 전자가 대부분 비상대론적인 이온보다 빠르게 이동하므로, EPAM의 L1 전자 채널은 이온이 지구 자기권에 도달하기 10-60분 전 사전 경보 제공. 이것이 NOAA SEP 경보의 운영 물리.

8. **Inflight calibration via reclosable covers + radioactive sources is standard but elegant.** / **Reclosable cover + 방사성 선원을 통한 비행 중 보정은 표준이지만 우아하다.** $^{241}$Am for LEMS gives 5.49 MeV α (P8) plus 60 keV X-ray (P1) — calibrating both ends of the rate-channel passband with a single source. $^{133}$Ba in LEMS30 emits energetic electrons that are magnetically deflected into $B$ — calibrating both magnet geometry and deflected-electron channels in one stroke. $^{244}$Cm + $^{148}$Gd in CA60 give a He α doublet (5.81 + 3.18 MeV) for W3, W4 calibration. The bi-metallic-spring closure mechanism is fail-safe (no active actuator needed; just heater). / LEMS의 $^{241}$Am은 5.49 MeV α (P8) + 60 keV X-선 (P1) — 단일 선원으로 rate channel 통과대역 양 끝을 보정. LEMS30의 $^{133}$Ba는 자기적으로 $B$로 휘어지는 에너지 전자를 방출 — 자석 기하와 deflected-electron 채널을 한 번에 보정. CA60의 $^{244}$Cm + $^{148}$Gd는 He α 이중선(5.81 + 3.18 MeV)으로 W3, W4 보정. Bi-metallic 스프링 closure는 fail-safe(능동 actuator 불필요, 히터로 충분).

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Bethe-Bloch and the basis for $\Delta E \times E$ ion identification / Bethe-Bloch와 $\Delta E\times E$ 이온 식별의 토대

For a non-relativistic ion of charge $z e$, mass number $A$, kinetic energy $E$, traversing material of atomic number $Z_{mat}$, mean excitation $I$:

$$-\frac{dE}{dx} = \frac{4\pi N_A r_e^2 m_e c^2}{A_{mat}}\,Z_{mat}\,z^2\,\frac{1}{\beta^2}\left[\ln\frac{2 m_e c^2 \beta^2 \gamma^2}{I} - \beta^2\right]$$

For low energies $\beta^2 \ll 1$ and slowly varying log, $-dE/dx \approx K z^2/\beta^2 \approx K' z^2 A/E$ (constants absorb material properties). Therefore: at fixed total energy $E$, $\Delta E$ in the thin $D$ detector scales as $z^2 A$. This separates H ($z^2 A = 1$), He ($z^2 A = 16$), C ($z^2 A = 432$), O ($z^2 A = 1024$), Fe ($z^2 A \approx 39000$) — wide separation $\Rightarrow$ clean tracks in the $D$-vs-$C$ matrix (Fig. 10).

**한국어.** 비상대론적 이온의 에너지 손실은 $-dE/dx \propto z^2/\beta^2$이며, 고정 총에너지 $E$에서 얇은 $D$ 검출기의 $\Delta E$는 $z^2 A$에 비례. H, He, C, O, Fe의 $z^2 A$ 값이 1 → 16 → 432 → 1024 → ~39000으로 크게 벌어져 PHA matrix의 궤적이 깨끗이 분리.

### 4.2 Differential flux from rate channels / Rate channel로부터 미분 플럭스

$$j(E_k) = \frac{N_k}{G\,\Delta E_k\,\Delta t}\quad \text{[particles/(cm$^2$ s sr MeV)]}$$

For LEMS P3 (passband $\Delta E_3 = E_4 - E_3 = 0.193 - 0.115 = 0.078$ MeV), $G_{\text{LEMS}} = 0.428$ cm$^2$ sr, $\Delta t$ = sector dwell time (1.5 s for 8-sector mode, 3 s for 4-sector). Per-spin sector counts:

$$j_{P3} = \frac{N_{P3}}{0.428 \times 0.078 \times 1.5} \approx \frac{N_{P3}}{0.050}\;\text{counts}^{-1} \cdot \text{(cm}^2\text{ s sr MeV)}^{-1}$$

So $N_{P3} = 50$ counts/sector $\Rightarrow$ $j \approx 1000$ /(cm$^2$ s sr MeV), typical for a moderate solar event.

**한국어.** LEMS P3 (통과대역 0.078 MeV), $G=0.428$ cm$^2$ sr, $\Delta t=1.5$ s ($8$ sector 모드)에서 sector당 50 카운트는 약 $j \approx 1000$ /(cm$^2$ s sr MeV) — 중간 강도 태양 이벤트의 전형적 값.

### 4.3 Angular sectoring and clock-angle binning / 각도 sector 분할과 clock-angle binning

Spin period $T_s = 12$ s, angular speed $\omega = 30^\circ/\text{s}$. For an 8-sector telescope with sector dwell $\tau = T_s/8 = 1.5$ s:

$$\phi_{\text{sec},k}(t) = \omega\,t\bmod 360^\circ,\quad \text{sector index } k = \left\lfloor\frac{\phi_{\text{sec}}}{45^\circ}\right\rfloor$$

For a 4-sector telescope (LEMS30, LEFS150), $\tau = 3$ s and sectors are 90° wide. Pitch angle of a magnetic-field-aligned particle is recovered from $\phi_{\text{sec}}$ + telescope polar angle $\theta_{\text{tel}}$ + the Parker spiral direction in spacecraft coordinates.

**한국어.** spin 주기 12 s, 각속도 30°/s. 8 sector 망원경의 sector dwell 1.5 s, 폭 45°; 4 sector 망원경(LEMS30, LEFS150)의 dwell 3 s, 폭 90°. 자기장 정렬 입자의 pitch angle은 sector clock-angle + 망원경 polar angle + 위성 좌표계에서의 Parker spiral 방향으로 복원.

### 4.4 MFSA log-binning and power-law fit / MFSA log binning과 멱법칙 적합

MFSA bins are equally spaced in $\log E$ over 32 channels covering ~3 decades:
$$E_k = E_{\min}\cdot 10^{(k-1)\Delta_{\log E}},\quad \Delta_{\log E} = \frac{\log_{10}(E_{\max}/E_{\min})}{31}\approx \frac{3}{31}\approx 0.097$$

For an SEP power-law $j(E) = j_0 (E/E_0)^{-\gamma}$, log-log linear fit:
$$\log j = \log j_0 - \gamma\,\log(E/E_0)$$

Slope = $-\gamma$. Typical impulsive-flare ion spectra: $\gamma \approx 2\text{-}4$. EPAM's MFSA fine binning enables identification of spectral breaks where $\gamma$ changes (e.g. softer below break, harder above), often signatures of acceleration mechanism transitions.

**한국어.** MFSA bin은 약 3 decade를 32 채널에 logarithmically 등간격 배치, $\Delta_{\log E}\approx 0.097$. SEP 멱법칙 $j = j_0(E/E_0)^{-\gamma}$의 log-log 직선 적합으로 기울기 $-\gamma$ 추출. 전형적 impulsive 플레어 이온 스펙트럼은 $\gamma \approx 2$-4. MFSA 정밀 binning은 가속 메커니즘 전이에 동반되는 spectral break 식별 가능.

### 4.5 Anti-coincidence rate-channel logic in Boolean form / 불 대수 형태의 anti-coincidence 논리

For LEMS P3 with thresholds $\theta_3, \theta_4$ on $M$ and threshold $\theta_F$ on $F$:

$$\text{P3} = \mathbf{1}[E_M \geq \theta_3] \wedge \mathbf{1}[E_M < \theta_4] \wedge \mathbf{1}[E_F < \theta_F]$$

In paper notation: $M3\overline{4F}$. The first two factors require the particle to deposit between thresholds 3 and 4 in $M$; the third (overbar over $F$) demands $F$ does not fire $\Rightarrow$ particle stops in $M$, not punching through.

For LEMS DE2 (electrons deflected to $B$, anti-coincidence $C$): $\text{DE2} = \mathbf{1}[E_B \geq \theta_2] \wedge \mathbf{1}[E_B < \theta_3] \wedge \mathbf{1}[E_C < \theta_C]$, i.e. $B2\overline{B3C}$, passband 53-103 keV.

**한국어.** LEMS P3는 $M$에서 임계값 3 초과 4 미초과 + $F$ 미발화 = $M3\overline{4F}$. DE2는 $B$에서 임계값 2 초과 3 미초과 + $C$ 미발화 = $B2\overline{B3C}$, 53-103 keV.

### 4.6 Adaptive priority transition rule / 적응형 우선순위 전이 규칙

State variable: $s_n \in \{H, He, O, Fe\}$ = species group of $n$-th processed event in a sector. Transition reads from Table VI columns:

$$P(\text{next} = X \mid s_n = Y) = \delta(X, \text{prio}_1(Y))$$

where $\text{prio}_1$ is the highest priority for previous-state $Y$:

$$\text{prio}_1(H)=Fe,\;\text{prio}_1(He)=H,\;\text{prio}_1(O)=He,\;\text{prio}_1(Fe)=O$$

If the highest-priority species is unavailable in the current accumulator queue, the scheme falls back to $\text{prio}_2(Y)$, then $\text{prio}_3$, then $\text{prio}_4$.

**한국어.** 상태 변수 $s_n$은 sector에서 $n$번째 처리 이벤트의 종 그룹. 표 VI 열로부터 최우선 종을 결정 ($H\to Fe$, $He\to H$, $O\to He$, $Fe\to O$); 해당 종이 큐에 없으면 2순위 → 3순위 → 4순위로 fallback.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1958 ─── Van Allen radiation belts (early SSD telescopes)
1961 ─── Anderson et al. solar electron events (Explorer 12)
1971 ─── Pioneer 10 launch (Krimigis CRT instrument)
1977 ─── Voyager LECP (Krimigis et al.) ── direct ancestor of HI-SCALE
1978 ─── ISEE-3 launched to L1 ── first dedicated solar-monitor at L1
1990 Oct ── Ulysses launch with HI-SCALE (Lanzerotti et al. 1992)
1992 ─── Lanzerotti et al., A&A 92, 349 ── HI-SCALE design paper
1993 ─── HI-SCALE Jupiter encounter (Lanzerotti et al. 1993)
1997 Aug 25 ── ACE launch ── Gold et al. EPAM (★ this paper, 1998)
1998 ─── Zwickl et al. NOAA RTSW system using ACE/EPAM data
2006 ─── STEREO A/B launch (IMPACT-LET, SEPT inherit SSD-telescope tradition)
2018 Aug ── Parker Solar Probe launch with ISOIS-EPI-Lo/Hi
2020 Feb ── Solar Orbiter launch with EPD-EPT/HET/STEP
2026 ─── EPAM still operating; ~29-year continuous L1 record (Cycles 23, 24, 25)
```

**English.** EPAM sits at the convergence of three traditions: (i) the LECP/HI-SCALE design lineage (Krimigis-Lanzerotti-Armstrong) — the longest-running family of low-energy SSD telescopes in heliophysics; (ii) the L1 monitor concept pioneered by ISEE-3 in 1978 — establishing that a dedicated upstream-of-Earth monitor is operationally invaluable; (iii) the dedicated composition mission (ACE) — providing comprehensive isotopic resolution from H to Ni. The paper is therefore both a heritage instrument paper (bridging Ulysses to ACE) and a foundational document for an operational space weather era that began in 1998 and continues to 2026 and beyond.

**한국어.** EPAM은 세 전통의 교차점에 위치: (i) LECP/HI-SCALE 설계 계보(Krimigis-Lanzerotti-Armstrong) — 태양권 물리 분야 최장 가동 저에너지 SSD 망원경 가족; (ii) 1978년 ISEE-3가 개척한 L1 monitor 개념 — 지구 상류 전용 monitor의 운영적 가치 확립; (iii) 종합 동위원소 분해를 제공하는 전용 조성 임무(ACE) — H부터 Ni까지. 따라서 본 논문은 heritage 기기 논문(Ulysses → ACE)이자 1998년 시작되어 2026년 이후까지 이어지는 운영 우주환경 시대의 기초 문서이다.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **Lanzerotti et al. 1992, A&A 92, 349 (HI-SCALE)** | Direct progenitor: EPAM is the flight-spare of HI-SCALE physically rebuilt for ACE / 직접 모태: EPAM은 HI-SCALE의 비행 예비품을 ACE용으로 재구성 | High / 매우 높음 |
| **Stone et al. 1998 (ACE mission paper)** | Defines the ACE mission and instrument complement that EPAM monitors / EPAM이 monitor하는 ACE 임무·기기 구성 정의 | High / 매우 높음 |
| **Zwickl et al. 1998, Space Sci. Rev. 86, 633** | NOAA RTSW system that uses EPAM rate channels for real-time space weather alerts / EPAM rate channel을 실시간 우주환경 경보에 활용 | High / 매우 높음 |
| **Reames 1999** | SEP physics: solar energetic particle classes (impulsive vs gradual) that EPAM detects and characterizes / EPAM이 감지·특성화하는 SEP 분류(impulsive vs gradual) | High / 매우 높음 |
| **Parker 1965 (Cosmic-ray transport)** | Theoretical framework for diffusion-convection of energetic particles that EPAM measures in situ / EPAM이 in-situ로 측정하는 에너지 입자의 확산-대류 이론 | Medium / 중간 |
| **Mason et al. 1998, Space Sci. Rev. 86, 409 (ULEIS)** | Companion ACE composition spectrometer; EPAM provides the angular/temporal context for ULEIS measurements / EPAM이 ULEIS 측정에 각도·시간 맥락 제공 | High / 매우 높음 |
| **Stone et al. 1998 (CRIS)** | High-energy GCR composition spectrometer; EPAM provides the low-energy SEP context that contaminates CRIS background / 고에너지 GCR 분광기; EPAM이 CRIS 배경을 오염시키는 저에너지 SEP 맥락 제공 | Medium / 중간 |
| **Müller-Mellin et al. 1995 (SOHO/COSTEP)** | Contemporary L1 SEP detector with which EPAM cross-calibrates / EPAM과 상호보정되는 동시대 L1 SEP 검출기 | Medium / 중간 |
| **von Rosenvinge et al. 2008 (STEREO IMPACT-LET)** | Modern descendant; same SSD-telescope $\Delta E \times E$ design philosophy / 현대 후손; 동일한 SSD 망원경 $\Delta E \times E$ 설계 철학 | Medium / 중간 |
| **McComas et al. 2016 (Parker Solar Probe ISOIS)** | Modern descendant for inner heliosphere; inherits HI-SCALE/EPAM lineage / 내태양권 현대 후손; HI-SCALE/EPAM 계보 계승 | Medium / 중간 |

---

## 7. References / 참고문헌

- Gold, R. E., Krimigis, S. M., Hawkins, S. E. III, Haggerty, D. K., Lohr, D. A., Fiore, E., Armstrong, T. P., Holland, G., and Lanzerotti, L. J., "Electron, Proton, and Alpha Monitor on the Advanced Composition Explorer Spacecraft", Space Science Reviews 86, 541-562, 1998. DOI: 10.1023/A:1005088115759
- Chiu, M. C., et al., "ACE Spacecraft", Space Science Reviews 86, 257, 1998.
- Lanzerotti, L. J., Gold, R. E., Anderson, K. A., Armstrong, T. P., Lin, R. P., Krimigis, S. M., Pick, M., Roelof, E. C., Sarris, E. T., Simnett, G., and Frain, W. E., 1992, "Heliosphere Instrument for Spectra, Composition, and Anisotropy at Low Energies", Astron. Astrophys. 92, 349.
- Lanzerotti, L. J., et al., 1993, "Measurements of Hot Plasma in the Magnetosphere of Jupiter", Planet. Space Sci. 41, 893.
- Lanzerotti, L. J., et al., 1983, "The ISPM Experiment for Spectra, Composition, and Anisotropy Measurements of Charged Particles at Low Energies", in K.-P. Wenzel, R. G. Marsden, B. Battrick (eds.), The International Solar Polar Mission - Its Scientific Investigations, Noordwijk, Netherlands, 141-154.
- Zwickl, R. D., et al., 1998, "The NOAA Real-Time Solar-Wind (RTSW) System Using ACE Data", Space Sci. Rev. 86, 633.
- Stone, E. C., et al., "The Advanced Composition Explorer", Space Science Reviews 86, 1, 1998.
- Mason, G. M., et al., "The Ultra-Low-Energy Isotope Spectrometer (ULEIS) for the ACE Spacecraft", Space Science Reviews 86, 409, 1998.
- Reames, D. V., "Particle Acceleration at the Sun and in the Heliosphere", Space Science Reviews 90, 413, 1999.
