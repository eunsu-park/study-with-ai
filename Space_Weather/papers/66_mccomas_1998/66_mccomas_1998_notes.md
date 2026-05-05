---
title: "Solar Wind Electron Proton Alpha Monitor (SWEPAM) for the Advanced Composition Explorer"
authors: D. J. McComas, S. J. Bame, P. Barker, W. C. Feldman, J. L. Phillips, P. Riley, J. W. Griffee
year: 1998
journal: "Space Science Reviews"
doi: "10.1023/A:1005040232597"
topic: Space_Weather
tags: [ACE, SWEPAM, solar_wind, electrostatic_analyzer, CEM, plasma_instrumentation, L1, real_time_solar_wind]
status: completed
date_started: 2026-04-25
date_completed: 2026-04-25
---

# 66. Solar Wind Electron Proton Alpha Monitor (SWEPAM) for the Advanced Composition Explorer / ACE를 위한 태양풍 전자·양성자·알파 모니터

---

## 1. Core Contribution / 핵심 기여

McComas et al. (1998)은 ACE 우주선의 태양풍 모니터 SWEPAM의 설계, 보정, 운영 모드를 종합 기술하는 1차 문서이다. SWEPAM은 두 개의 독립 기기 — 이온 기기 SWEPAM-I (양성자·알파; 260 eV/q–36 keV/q; 16 CEM)와 전자 기기 SWEPAM-E (1.6 eV–1.35 keV; 7 CEM) — 로 구성된다. 두 기기 모두 곡판형 정전기 분석기(spherical-section ESA)와 채널 전자 증배기(CEM) 어레이를 사용하며, 우주선 자전과 결합된 부채꼴 시야로 4π sr 거의 전체를 64초마다 스캔한다. SWEPAM 기기는 NASA/ESA Ulysses 임무의 SWOOPS(Bame et al. 1992) *비행 예비품(flight spare)*을 재활용하여 ACE 프로그램 비용을 크게 절감했지만, 세 가지 결정적 개선 — (1) 헤일로 전자(>100 eV) 누적시간 16배, (2) 이온 극각 분해능 5°→2.5°, (3) 태양풍 빔 외부의 초열적 이온 측정용 20° 원뿔 — 을 통해 ACE 과학 목표에 부합하게 향상되었다. 보정된 기하 인자(SWEPAM-I: 1-20×10⁻⁶ cm² sr eV/eV per CEM; SWEPAM-E: ~250-735 ×10⁻⁶ cm² sr eV/eV per CEM)와 풀 64초 3차원 분포 함수에서 64초마다 NOAA로 송출되는 RTSW 부분 데이터셋(33픽셀)을 통해 우주 기상 운영도 지원한다.

McComas et al. (1998) is the primary documentation source for the SWEPAM solar-wind monitor on the ACE spacecraft. SWEPAM consists of two independent instruments — an ion sensor SWEPAM-I (protons + alphas; 260 eV/q–36 keV/q; 16 CEMs) and an electron sensor SWEPAM-E (1.6 eV–1.35 keV; 7 CEMs). Both employ spherical-section electrostatic analyzers (ESAs) followed by arrays of channel electron multipliers (CEMs); their fan-shaped fields of view, coupled to spacecraft spin, sweep nearly the full 4π sr every 64 seconds. The hardware was built from refurbished flight spares of the Ulysses SWOOPS experiment (Bame et al. 1992), saving major cost, but with three decisive enhancements: (1) factor-of-16 increase in halo-electron accumulation time, (2) halving of the effective ion-detecting CEM polar spacing from ~5° to ~2.5°, and (3) addition of a 20° conical swath of enhanced sensitivity for suprathermal ions outside the bulk solar-wind beam. Calibrated geometric factors (SWEPAM-I: 1-20×10⁻⁶ cm² sr eV/eV per CEM; SWEPAM-E: ~250-735 ×10⁻⁶ cm² sr eV/eV per CEM) plus the 64-s, full 3-D distribution functions and a 64-s real-time-solar-wind (RTSW) subset of 33 telemetered pixels together support both fundamental research and operational space-weather forecasting at L1.

---

## 2. Reading Notes / 읽기 노트

### Part I: ACE Mission Context and Scientific Objectives / ACE 임무 컨텍스트와 과학 목표

**§1 Introduction (pp. 563-567)**

ACE 임무는 "물질의 4 저장소(four reservoirs of matter)" — 태양 코로나, 태양 에너지 입자(SEP), 국부 성간 매질 (변칙 우주선 ACR을 통해), 은하 우주선(GCR) — 의 동위원소 및 원소 조성을 직접 측정하기 위해 설계되었다. 6개의 조성 실험(CRIS, SIS, ULEIS, SEPICA, SWIMS, SWICS)이 ~100 eV에서 수백 MeV/nuc 영역을 덮으며, 추가로 태양풍 모니터 SWEPAM, 에너지 입자 모니터 EPAM, 그리고 두 개의 자기계 MAG가 *해석을 위한 컨텍스트*를 제공한다.

The ACE mission was designed to measure the isotopic and elemental composition of the "four reservoirs of matter" — the solar corona, solar energetic particles, local interstellar medium (via ACR), and galactic cosmic rays. Six composition experiments span ~100 eV to several hundred MeV/nuc; additional context is provided by the solar-wind monitor SWEPAM, the energetic-particle monitor EPAM, and two magnetometers (MAG).

표 I (Table I)은 ACE 레벨 1 과학 목표 매트릭스를 보여준다. SWEPAM은 6개 목표에 대해 *primary measurement (P)* 로 분류되고, 추가 5개 목표에 *contributing measurement (C)* 로 분류된다. 즉 SWEPAM은 ACE 임무의 핵심 기둥(critical cornerstone)이다.

Table I gives the ACE Level 1 science objectives matrix. SWEPAM is classified as a primary measurement (P) for six objectives and a contributing measurement (C) for five more — a critical cornerstone of the ACE mission.

SWOOPS heritage: Ulysses 1990년 발사, 1992년 Jupiter 중력 보조 후 극궤도. SWOOPS는 두 개 독립 기기(전자/이온)이며 둘 다 곡판형 ESA + CEM 디자인. ACE 측 수정 항목은: (1) 회전축 방향 — Ulysses는 Earth-pointing이지만 ACE는 Sun-pointing 회전축이므로 ESA 입사구의 새 마운팅 브래킷 설계, (2) ESA 전압 sweeping 새 lookup table, (3) CEM 카운팅 logic 5V CMOS로 현대화. Ulysses launched 1990, Jupiter swingby 1992, polar orbit; SWOOPS comprised two independent ESA+CEM instruments. ACE modifications: new mounting brackets (Sun-pointing spin axis instead of Earth-pointing), new ESA voltage sweeping lookup tables, modernized 5V CMOS counting logic.

**§2 Scientific Objectives (pp. 567-580)**

**§2.1 Solar Wind Formation and Acceleration / 태양풍 형성과 가속**: Ulysses는 1994-95 빠른 위도 스캔에서 *bimodal* 태양풍을 발견했다 — 위도 ±22° 외측은 거의 구조 없는 고속 풍(>700 km/s, coronal hole 기원), 22° 내측은 변동성 큰 저속 풍(streamer belt 기원). 그림 1(폴라플롯)은 이 패턴을 시각화한다. SWEPAM은 ACE/Ulysses 동시 다점 관측 (그림 2: 1998-2002 두 우주선 위치)을 통해 헬리오스피어 글로벌 구조와 태양 활동주기 (solar minimum → maximum) 의존성을 추적한다.

**§2.1 Solar Wind Formation and Acceleration**: Ulysses' 1994-95 fast latitude scan discovered a bimodal solar wind — nearly structure-free high-speed flow (>700 km/s, from coronal holes) outside ±22° latitude, and variable slow flow (streamer belt) within ±22°. Fig. 1 (polar plot) visualizes this pattern. SWEPAM enables ACE/Ulysses dual-point heliospheric structure studies (Fig. 2 shows 1998-2002 spacecraft locations) and tracks solar-cycle dependence (solar minimum → maximum).

**§2.1.1 Streamer Belt**: 헬리오스피어 전류 시트(HCS) 인근의 더블 양성자/알파 빔(그림 4) — Hammond et al. (1995)이 자기 재결합 증거로 해석한 신호 — 그리고 열속 dropouts, NCDEs, 헬륨 함량 변동 등 streamer belt의 미세 구조 진단 도구들을 SWEPAM이 제공한다.

**§2.1.1 Streamer Belt**: Double proton/alpha beams near the heliospheric current sheet (HCS, Fig. 4), interpreted by Hammond et al. (1995) as evidence of magnetic reconnection. SWEPAM provides diagnostics for HCS fine structure, heat-flux dropouts, NCDEs, and helium-abundance variations.

**§2.1.2 Coronal Mass Ejections (CMEs)**: CME의 가장 강력한 단일 식별자는 *counter-streaming suprathermal electrons* (Gosling 1996). 그림 5(11/29/1990 CME 사례)는 두 영역(C, D)에서 양방향 전자 열속을, 영역(A, E)에서는 단방향 전자 열속을 보여 CME 구조의 자기적 연결성을 직접 진단한다. 그림 6의 CME schematic, 그림 7의 4 가지 자기 위상학(open / disconnected plasmoid / bottle / flux rope)도 핵심.

**§2.1.2 CMEs**: The single best identifier of a CME is counter-streaming suprathermal electrons. Fig. 5 (a CME on 11/29/1990) shows bidirectional heat flux in regions C-D and unidirectional flux in A, E — directly diagnosing CME magnetic topology. Fig. 6 shows a CME schematic; Fig. 7 lists four magnetic topologies (open, disconnected plasmoid, bottle, flux rope).

**§2.2 Particle Acceleration and Transport**: 강한 행성간 충격 — CIR forward/reverse shock, fast-CME 전방 충격 — 은 입자 가속 사이트이다. 그림 8(1993년 1월 Ulysses ~5 AU CIR 관측)은 forward/reverse shock 양쪽으로 광범위한 수페어쓸 전자 분포를 보여준다. SWEPAM은 enhancement 모양·강도로 충격 토폴로지를 진단한다. **§2.2 Particle Acceleration and Transport**: Strong interplanetary shocks — CIR forward/reverse pairs and fast-CME-driven forward shocks — are particle-acceleration sites. Fig. 8 (Ulysses CIR at ~5 AU, January 1993) shows extensive suprathermal electron enhancements upstream of both shocks. SWEPAM diagnoses shock topology via enhancement shape and intensity.

**§2.2.3 Pickup Ions of Interstellar Origin**: 충격 부근 전자 충돌 이온화는 성간 H 픽업 이온화를 ~10% 증가시킨다(Isenberg & Feldman 1995; Feldman et al. 1996). 그림 9는 그 ionization rate 곡선을 보여준다. SWEPAM의 양성자 질량 플럭스와 전자 분포는 이러한 픽업 양·수송을 정량화한다. **§2.2.3 Pickup Ions of Interstellar Origin**: Electron-impact ionization near shocks enhances interstellar-H pickup by ~10%. SWEPAM's proton mass flux + electron distributions quantify pickup amount and transport.

### Part II: SWEPAM-I (Ion Instrument) / SWEPAM-I (이온 기기)

**§3 SWEPAM-I overview (pp. 580-589)**

이온 기기는 다음 단계로 작동한다 / The ion instrument operates as follows:
1. 이온이 단일 oversized 입사구를 통해 들어오고 / Ions enter a single oversized entrance aperture;
2. ESA 두 곡판 사이의 105° gap에서 음성 고전압이 좁은 $E/q$ 영역과 azimuthal 각 영역(3°-4.5°, 극각에 따라)을 통과하도록 선택; / a negative HV between two 105°-bending plates selects a narrow $E/q$ band (~5%) and azimuthal angle (3-4.5° depending on polar angle);
3. 통과한 이온은 16개 CEM 중 하나에 의해 검출(어떤 CEM이 트리거되었는지가 극각 $\theta$를 식별); / a transmitted ion is detected by one of 16 CEMs (the CEM index identifies polar angle $\theta$);
4. 우주선 회전 위상이 방위각 $\phi$를 식별; / spacecraft spin phase identifies azimuth $\phi$;
5. ESA 전압 step이 $E/q$를 식별. / the ESA voltage step identifies $E/q$.

따라서 (CEM index, 회전 위상, ESA step) → ($\theta$, $\phi$, $E$) 의 1대1 대응이 측정 행렬을 구성한다. The triple (CEM index, spin phase, ESA step) maps one-to-one to ($\theta$, $\phi$, E$), forming the basic measurement matrix.

**기하학 / Geometry (Fig. 10)**: 입사구 normal은 우주선 자전축에서 18.75° polar angle 떨어져 있다. CEM 1-12는 태양풍 빔(보통 polar <25° 도착)을 sample. CEM 13-16은 더 큰 polar angle (>25°)에서 초열적 이온을 측정. 자전축이 Sun-facing deck에 수직 (계획 ±0.5° 이내)이라면, CEM 1-5와 CEM 6-10이 두 반회전(half rotation)에서 데이터를 *interleave*하여 효과적 폴라 분해능을 5° → 2.5°로 단축한다. **Geometry**: aperture normal tilted 18.75° from the spin axis. CEMs 1-12 sample the solar-wind beam; CEMs 13-16 sample suprathermal ions at >25° polar. CEMs 1-5 and 6-10 interleave across two half-rotations to halve effective polar resolution from 5° to 2.5°.

**Tables II-III (능력/하드웨어)**: SWEPAM-I energy range 260 eV/q–36 keV/q; ΔE/E = 5% (or 2.5% with interleaving); polar FOV (-)25° to 65° in 5° (2.5° interleaved); azimuth FWHM 3-4.5°; G-factor per pixel 1-20×10⁻⁶; time resolution 64 s. Mass 3.7 kg, average power 3.1 W, telemetry 540 b/s. Box: 36×24×30 cm.

**SWEPAM-I 센서 (§3.1)**: 곡판은 알루미늄 합금, 105° 굴절각, gap 2.84 mm, 평균 반경 100 mm. 곡판은 Ebanol-C 처리한 구리로 코팅되어 UV 산란을 줄인다. 16개 CEM은 7 mm funnel 직경, 5° 폴라 간격으로 배열. CEM 1-12는 0.4 mm wide aperture(태양풍 이온 플럭스 ~1 AU에 맞춤; gap 중앙만), CEM 13-16은 큰 원형 aperture(suprathermal에 모든 ESA gap 통과). Mesh: 70 lines/inch, 92% 투과 nickel, ESA로부터의 전기장 누설 차단 + funnel과 동일 전위. **SWEPAM-I sensor**: aluminum-alloy spherical-section plates, 105° bend, gap 2.84 mm, mean radius 100 mm; plates coated with Ebanol-C blackened copper to suppress UV scatter. The 16 CEMs (7 mm funnel) are at 5° polar spacing. CEMs 1-12 have 0.4 mm narrow apertures matched to 1-AU solar-wind flux (only the gap center contributes); CEMs 13-16 have circular apertures sampling the full ESA gap for suprathermal ions. Nickel mesh (70 lines/inch, 92% transmission) at the funnel potential blocks ESA fields from corrupting CEM gain.

**Pump-out baffle**: 7 surface-reflection minimum, blocking photon/charged-particle background through the pump-out channel. **CEM amplification**: each CEM has a dedicated amplifier-discriminator with two threshold levels (1×10⁶ and 2×10⁷ electrons per pulse).

**SWEPAM-I 전자장치 (§3.2)**: 4개 모듈 — LVPS (low voltage power supply), HVPS (high voltage), motherboard, main electronics. Main contains five boards: PRO-11 processor (80C51 microcontroller, 7.4 MHz, 28K EEPROM, 48K RAM), SPM-14 counters (16 × 16-bit), BUF-19 level shifter (8V→5V), SIM-11 spacecraft interface, BAM-16 HV controller. ESA HVPS: 256 logarithmic levels, only 200 used (-15 V to -2 kV). CEM HVPS: 16 linear levels, -2.4 to -3.9 kV. **SWEPAM-I electronics**: four modules; main electronics has five boards: PRO-11 processor (80C51 + 7.4 MHz + 28 K EEPROM + 48 K RAM), SPM-14 counters, BUF-19 level shifter, SIM-11 spacecraft interface, BAM-16 HV controller. ESA HVPS: 256 log levels (only 200 used, -15 V to -2 kV). CEM HVPS: 16 linear levels, -2.4 to -3.9 kV.

**§3.3 SWEPAM-I 보정 (Figs. 13-15)**: Los Alamos plasma analyzer 보정 시설에서 5 kV 단방향 양성자 빔, 컴퓨터 제어 3축 (2 회전 + 1 병진) 테이블 사용. 모든 16개 CEM에 대해 (azimuth, polar, ESA voltage) 3차원 응답 함수를 측정.

**§3.3 SWEPAM-I Calibration (Figs. 13-15)**: At the Los Alamos plasma calibration facility, a 5 kV unidirectional proton beam was scanned over a 3-axis (two rotations + one translation) computer-controlled table. The 3D response function was measured for each of the 16 CEMs as a function of (azimuth, polar, ESA voltage).

- **Fig. 13** (CEM 4): azimuth–ESA voltage 응답에서 두 축은 강하게 결합 — 구형 ESA의 예상 동작(Gosling et al. 1978). CEM 4에 대한 azimuth–ESA voltage cut shows the two axes are strongly coupled, as expected for a spherical-section ESA.
- **Fig. 14a** (energy cut): 분포는 가우시안에 잘 맞춰지며 ΔE/E ≈ 5% FWHM. The energy distribution fits a Gaussian with ΔE/E ≈ 5% FWHM.
- **Fig. 14b** (azimuth): 가우시안, FWHM ~3.6°. 
- **Fig. 14c** (polar): 사다리꼴(trapezoid)이 가우시안보다 더 좋은 fit; 폭 ~5°. Polar response fit by a trapezoid (~5° wide), not a Gaussian.
- **Analyzer constant K**: 입사 5 keV 양성자 / 293 V plate voltage = **K ≈ 17.1**.
- **Fig. 15** (16 CEM polar 컷): 12 태양풍 채널 + 4 초열 채널의 응답이 명확히 분리. CEM 13-16의 큰 aperture 덕에 큰 폴라각에서 transmission이 명확히 더 높다. 16-CEM polar cut clearly separates 12 solar-wind channels from 4 suprathermal channels; CEMs 13-16 have visibly higher transmission at large polar angles.

**Table IV** — Geometric factors (SWEPAM-I, ×10⁻⁶ cm² sr eV/eV):
| CEM | G | CEM | G | CEM | G | CEM | G* |
|-----|---|-----|---|-----|---|-----|----|
| 1 | 7.62 | 5 | 8.50 | 9 | 0.92 | 13* | 16.67 |
| 2 | 7.42 | 6 | 6.01 | 10 | 1.59 | 14* | 15.20 |
| 3 | 9.23 | 7 | 5.07 | 11 | 2.86 | 15* | 20.77 |
| 4 | 8.67 | 8 | 1.63 | 12 | 4.69 | 16* | 14.88 |

(*Suprathermal channels — 더 큰 G로 높은 감도). 솔라윈드 채널의 G에는 mask aperture에 의한 systematic 변화 (CEMs 9-11에서 가장 작음) 가 있고, 이는 실제 1 AU 태양풍 플럭스 분포에 fitted 디자인이다.

### Part III: SWEPAM-E (Electron Instrument) / SWEPAM-E (전자 기기)

**§4 SWEPAM-E (pp. 592-601)**

전자 기기 디자인 차이 / Electron-instrument design differences from SWEPAM-I:
- ESA 굴절각 **120°** (vs 105°), 평균 반경 41.9 mm, gap 3.5 mm.
- 7개 CEM, 11 mm 직경 funnel, 21° 폴라 간격 (vs 16 CEM × 5° in SWEPAM-I).
- ESA inner plate에 **양**(+) 고전압 (전자 인력); +8.6 V to +300 V, 32 logarithmic levels.
- 입사구 normal이 자전축에 *수직* — fan FOV가 자전과 함께 >95% 4π sr 휩쓴다 (small ~10° half-angle conical holes along ±spin-axis only).
- Polar FOV: 10°-170° (즉 양 반구 모두 cover). Polar resolution 21° per CEM. Azimuthal resolution 9° (normal incidence) to 28° (extreme polar).
- Energy: 1.6 eV - 1.35 keV in 20 contiguous bins, ΔE/E = 12%.
- **Analyzer constant K = 4.3**.
- ITO(인듐 주석 산화물) coated thermal blanket — 전자 측정에 결정적 (표면 전하 build-up 방지, 자기장 방출 없음).

**§4.3 SWEPAM-E Calibration (Figs. 19-22)**: 전자 빔 대신 1.05 kV **양성자** 빔으로 보정 (Los Alamos 시설은 ions만 생성). 음성 HV를 inner plate에 인가해 양성자가 전자처럼 휘게 했고, 양성자가 CEM funnel post-acceleration bias를 잘 통과하는 에너지 사용. 이 기법은 지구 자기장에 의한 전자 deflection을 회피하는 부수 이점. **Calibration**: with 1.05 kV proton beam (the Los Alamos facility produces ions, not electrons); negative HV on inner plate makes protons curve like electrons. Bonus: protons unaffected by Earth's field.

- **Fig. 19** (CEM 5, polar -21°): azimuth-ESA voltage cut, 결합된 응답.
- **Fig. 20a-c** (energy/azim/polar cuts): energy and azimuth fit Gaussians; polar response fit by a *skewed Gaussian* (asymmetric).
- **Fig. 21** (7 CEM polar cut at 0° azimuth): clean 21° separated channels covering ±90° polar.
- **Fig. 22** (SWOOPS vs SWEPAM-E geometric factors): excellent agreement across CEMs except for ~20% flattening at CEMs 3, 4 — likely subtle hardware/CEM differences. Confirms calibration accuracy across decades.

**Table V** — SWEPAM-E geometric factors (×10⁻⁶ cm² sr eV/eV):
| CEM | G | CEM | G |
|-----|---|-----|---|
| 1 | 255.6 | 5 | 733.5 |
| 2 | 511.0 | 6 | 540.3 |
| 3 | 633.4 | 7 | 272.9 |
| 4 | 659.5 |  |  |

(SWEPAM-E G가 SWEPAM-I G보다 ~100배 더 큼 — 전자 플럭스가 훨씬 낮은 데 대한 보상.) SWEPAM-E G is ~100× larger than SWEPAM-I G — compensating for the much lower electron differential flux.

### Part IV: Operations and Data / 운영과 데이터

**§5 Operations (pp. 601-609)**

"Cycle → Modes → Segments" 계층. Nominal ACE spin: 5 rpm = 12 s/spin. SWEPAM은 텔레메트리에 동기화: 데이터 누적 segment 길이 12.8 s (가장 긴 예상 spin 12.2 s보다 약간 김 → 누락 없음). 각 segment의 sampling window는 spin to spin precess.

The hierarchy is "cycle → modes → segments". ACE nominal spin: 5 rpm = 12 s/spin. SWEPAM is synchronized to telemetry, with 12.8 s segment length (slightly longer than worst-case 12.2 s spin → no gap).

**Table VI** — SWEPAM 과학 데이터 산출물 / Science data products:
| Acronym | Mode | Species | Description | Matrix | Time res / cadence |
|---|---|---|---|---|---|
| DSWI | SWI | Ion | Plasma ion VDF | 40(E)×96(θ-φ) | 64/64 s |
| DSTI | SWI | Ion | Suprathermal ion VDF | 20(E)×4(θ)×6(φ) | 64/64 s |
| DSSTI | SSTI | Ion | Plasma ion VDF (search) | 40(E)×96(θ-φ) | 64/1984 s |
| DSTI2 | SSTI | Ion | Suprathermal ion VDF (search) | 20(E)×4(θ)×6(φ) | 64/1984 s |
| DRTSW | SWI | Ion | Real time solar wind | 40(E)×33(θ-φ) | 64/64 s |
| DNSWE | NSWE | Electron | Plasma electron VDF | 20(E)×5(θ)×30(φ); 2(θ)×15(φ)* | 64/128 s |
| DSTEA | STEA | Electron | Suprathermal electron | 10(E)×5(θ)×60(φ); 2(θ)×30(φ)* | 64/128 s |
| DPHE | PHE | Electron | Photoelectron | 20(E)×5(θ)×30(φ); 2(θ)×15(φ)* | 64/1984 s |

(*outer CEMs sum 2 adjacent φ-sectors)

**§5.2 Ion modes**:
- **SWI (Solar Wind Ion / "Track Mode")**: 이전 ($L_{\text{mx}}$) 피크 $E/q$ 레벨에서 시작해 40 step ×5 segments × 8 levels per step. $E/q$ spacing 2.5%, 인접 데이터셋을 odd/even으로 *interleave*하면 2.5% 분해능. 61 φ-sectors at 5 rpm ⇒ 6.19° spacing × 0.59° wide. The track mode samples 40 levels in 5 segments of 8 levels each, with 5% Δ(E/q) between adjacent levels (2.5% with interleaving). At 5 rpm → 61 φ-sectors per spin (6.19° spacing × 0.59° wide).
- **DSWI** 텔레메트리: 12 θ × 61 φ × 40 E 픽셀(=29,280) 전송 불가 → 12 *masks* (one per θ) 가 angular offset >25° 인 픽셀을 버리고, 96 best 픽셀을 보낸다. 13% of pixels but tests show even broad/extreme winds yield essentially no nonzero pixel discarded.
- **DSTI** (suprathermal): outer CEMs 13-16, 8 E levels paired → 4 levels per segment, 61 φ paired → 6 per segment. 최종 29 E × 4 θ × 6 φ.
- **SSTI (Search/Suprathermal Ion)**: 고정 voltage stepping, 500 eV/q–35.2 keV/q. 12.5% (low) - 10% (high) Δ(E/q) spacing. 항상 1984 s 마다 한 번 (32 minutes) 실행되어 SWI peak-tracking이 잃었을 때 robust 복구 보장.
- **CALI (Calibration)**: $L_{\text{mx}}$ 고정, CEM bias level varied around set point at $L_{\text{mx}} - 2, -1, 0, +1, +2$, 두 threshold (A/B). CEM gain saturation 추적용.

**§5.3 Electron modes**:
- **NSWE (Normal Solar Wind Electron)**: 12-31 voltage levels 사용 → 3.2 eV-1377 eV. 각 12.8 s segment에서 4 $E/q$ × 30 φ × 7 CEM. DNSWE: 20 E × 30 φ for inner 5 CEMs (2-6); 15 φ for outer 2 CEMs (1, 7).
- **STEA (Suprathermal Electron Angle scan)**: top 10 voltage levels (84-1377 eV); 2 $E/q$ × 60 φ per segment. DSTEA: 10 E × 60 φ for inner CEMs, 30 φ for outer CEMs. φ resolution 2× higher than NSWE.
- **PHE (Photoelectron)**: voltages 8-27 → 0.86-429 eV. PHE is identical in format to DNSWE.
- **CALE**: same as CALI but for electrons.

**§5.4 Real Time Solar Wind (RTSW)**: SWEPAM RTSW 데이터셋은 DSWI의 33 픽셀 부분 (96 telemetered 중 34%, total 측정의 4.4%)이며 모든 offset angle <18°이다. NOAA Space Environment Center가 24/7 다운링크하여 우주 기상 모니터링/예보에 사용. 저자들은 이 RTSW에서 도출된 plasma parameters를 *pseudo-* 또는 *real-time-* prefix로 표기할 것을 권고하며, 과학적 publication용 full processing과 구분한다. SWEPAM RTSW is a 33-pixel subset of DSWI (34% of telemetered, 4.4% of measured pixels) with offset angles <18°. NOAA SEC downlinks 24/7 for space-weather monitoring; authors recommend prefixing derived parameters as "pseudo-" or "real-time-" to distinguish from full science processing.

---

## 3. Key Takeaways / 핵심 시사점

1. **재활용 + 표적화된 업그레이드 = 비용 효과적 우주 기상 자산** / **Heritage reuse + targeted upgrade = cost-effective space-weather asset** — Ulysses/SWOOPS 비행 예비품을 재활용하면서도 (1) 헤일로 전자 16배 누적, (2) 폴라 분해능 절반, (3) 20° 초열 원뿔 추가의 세 핵심 개선만 집중 적용. 이는 *추가하지 말고 다듬어라*는 헤리티지 기기 철학의 모범 사례. The team reused Ulysses/SWOOPS flight spares while focusing only on three critical upgrades (16× halo accumulation, halved polar resolution, 20° suprathermal cone) — a model "polish, don't add" heritage philosophy.

2. **3D VDF에서 모멘트로의 직접 경로** / **Direct path from 3D VDF to moments** — (CEM index, spin phase, ESA step) → (θ, φ, E)의 1대1 측정 매트릭스가 64초마다 풀 3D 분포 함수를 생성한다. 모멘트(밀도, 속도, 온도, 열속)는 calibrated geometric factor 와 적절한 모멘트 적분으로 직접 도출된다. The (CEM, spin, ESA-step) triple maps one-to-one to (θ, φ, E), producing full 3D distributions every 64 s; moments follow directly from the calibrated geometric factors.

3. **이온은 좁은 빔, 전자는 거의 전 4π — 두 ESA 디자인의 차이를 결정** / **Narrow ion beam vs near-isotropic electrons drives two ESA designs** — 태양풍 이온 빔은 매우 좁기 때문에(보통 ±25° 이내) SWEPAM-I aperture는 자전축에서 18.75° 기울어져 12 빔 채널 + 4 suprathermal 채널의 비대칭 디자인을 가진다. 반대로 전자는 거의 등방성이므로 SWEPAM-E aperture는 자전축에 수직이어서 7 CEM이 ±73° 폴라를 cover. The narrow solar-wind ion beam (~±25°) drives SWEPAM-I's tilted aperture and asymmetric 12+4 channel layout, while quasi-isotropic electrons drive SWEPAM-E's spin-axis-perpendicular aperture covering ±73° with 7 CEMs.

4. **마스크가 동적 범위 문제를 해결** / **Exit-aperture mask resolves dynamic range** — 동일 ESA에서 빔 코어와 초열적 꼬리를 동시에 측정하려면 ~10⁴-10⁶ 동적 범위가 필요. SWEPAM-I 출구 mask는 CEM 1-12에 0.4 mm 좁은 슬릿, CEM 13-16에 큰 원형 aperture를 두어 한 분석기로 두 mode를 통합. Measuring the bulk-wind core and the suprathermal tail in one analyzer requires a 10⁴-10⁶ dynamic range; the exit-aperture mask uses 0.4 mm narrow slits for CEMs 1-12 and large circular apertures for CEMs 13-16 — bridging both regimes in a single analyzer.

5. **CEM 식별 + 회전 위상 = 두 각도 좌표의 분리** / **CEM index + spin phase = two angular coordinates** — 회전 우주선의 fan FOV 디자인은 폴라(CEM index)와 방위각(spin phase)을 자연스럽게 분리한다. 이는 Cluster, STEREO, Solar Orbiter 등 수많은 후속 임무에서 표준이 되었다. The fan-FOV-on-spinner design naturally separates polar (via CEM index) and azimuth (via spin phase) — a paradigm copied by Cluster, STEREO, Solar Orbiter, and PSP.

6. **양성자 빔으로 전자 기기를 보정한다는 영리한 트릭** / **The clever trick of calibrating an electron instrument with a proton beam** — SWEPAM-E를 ESA 안쪽 plate에 negative HV를 주고 1.05 keV 양성자 빔으로 보정. Los Alamos 시설이 ion만 만들 수 있다는 제약을 극복할 뿐 아니라 지구 자기장의 전자 deflection도 우회. Calibrating SWEPAM-E with a 1.05 keV proton beam (negative HV on the inner plate) sidesteps the fact that the Los Alamos facility produces only ions and avoids Earth-field deflection of electrons.

7. **운영 모드는 텔레메트리 예산이 강제하는 trade-off** / **Operational modes encode the telemetry trade-off** — 96 픽셀을 송출하는 SWI/DSWI의 angular mask는 최대 카운트 픽셀로부터 25° 이상 떨어진 픽셀을 버리고 13% pixel만 다운링크. 그래도 시뮬레이션은 거의 모든 nonzero pixel이 살아있음을 확인. 텔레메트리 예산과 과학 정보 보존의 균형을 정량화한 사례. Telemetry constraints force masking that keeps only ~13% of pixels but simulations confirm essentially all nonzero pixels survive — a quantitative balance between bandwidth and science.

8. **L1 + RTSW = 경고 시간 (warning time) 패러다임의 정착** / **L1 + RTSW = entrenchment of the geomagnetic warning paradigm** — ACE/SWEPAM RTSW는 NOAA SEC가 24/7 활용한 최초의 운영 우주 기상 데이터 스트림이며, ~30-60분의 사전 경고 시간을 제공한다. 이 패러다임은 DSCOVR (2015), IMAP (2025) 등 후속 운영 임무로 직접 이어져 우주 기상 운영 인프라의 토대가 되었다. ACE/SWEPAM RTSW was the first operational space-weather data stream used 24/7 by NOAA SEC, giving ~30-60 min lead time before solar wind reaches Earth — a paradigm carried directly into DSCOVR, IMAP, and Space Weather Follow-On.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 정전기 분석기의 이상적 응답 / Ideal ESA response

곡판형 ESA에서 균일한 두 곡판 사이의 전기장 $E_r$ 은 곡률 반경 $r_0$의 원호를 따라 입자 운동의 *구심력*과 균형을 이룬다 / The electric field $E_r$ between two curved plates balances the *centripetal force* on a particle moving along an arc of curvature radius $r_0$:

$$q E_r = \frac{m v^2}{r_0} = \frac{2 (E_{\text{kin}}/q) \cdot q}{r_0}$$

전압 차 $V = E_r \cdot d$ (gap $d$) 와 결합하면 / Combining with the plate-voltage relation:

$$\boxed{ \frac{E_{\text{kin}}}{q} = K \cdot V \qquad \text{where} \quad K = \frac{r_0}{2 d}}$$

SWEPAM-I: $r_0 = 100$ mm, $d = 2.84$ mm → $K_{\text{ideal}} = 17.6$. 측정 $K$ ≈ 17.1 → 1차 이상적 모델과 ~3% 일치. SWEPAM-I: ideal $K_{\text{ideal}}$ = 17.6 vs. measured 17.1 — agreement to ~3%.

SWEPAM-E: $r_0 = 41.9$ mm, $d = 3.5$ mm → $K_{\text{ideal}} = 5.99$. 측정 $K$ ≈ 4.3. 차이는 120° 굴절각의 fringing field와 mesh 효과 때문. SWEPAM-E ideal 5.99 vs. measured 4.3, with deviation due to 120° fringing fields and mesh effects.

### 4.2 응답 함수의 분해 / Response-function decomposition

Gosling et al. (1978)에 따라 SWEPAM CEM의 3D 응답 함수는 거의 분리 가능하다 / Per Gosling et al. (1978), the 3D response approximately separates as:

$$T(E, \theta, \phi) \approx T_E(E - E_0) \cdot T_\theta(\theta - \theta_0) \cdot T_\phi(\phi - \phi_0; E)$$

- $T_E$: 가우시안, $\sigma_E = 0.05 \, E_0 / 2.355$ (FWHM 5%) — Gaussian, 5% FWHM
- $T_\theta$: 사다리꼴(trapezoidal), 평탄 폭 ~5° — Trapezoidal with flat-top width ~5°
- $T_\phi$: 가우시안, 폭 3-4.5° (energy 의존; coupled to $T_E$ in spherical-section)

전자에서 $T_\theta$는 *왜곡된 가우시안*(skewed Gaussian, Fig. 20c). For electrons, $T_\theta$ is a skewed Gaussian.

### 4.3 기하 인자 / Geometric factor

기하 인자 $G_i$ 는 응답 함수의 부피 적분이다 / Geometric factor is the volume integral of the response:

$$G_i = \int_E \int_\theta \int_\phi A_{\text{eff}} \, T_i(E, \theta, \phi) \, \frac{dE}{E} \, \sin\theta \, d\theta \, d\phi$$

여기서 $A_{\text{eff}}$ 는 실효 입사구 면적 (cm²). SWEPAM-I의 0.4 mm 좁은 mask에서 $A_{\text{eff}}$ 는 ~ (7.8 mm gap height) × (0.4 mm slit) = 3.12 mm² = 0.0312 cm². CEM 4 G = 8.67 ×10⁻⁶ cm² sr eV/eV ≈ 0.0312 × ΔΩ × 0.05; ΔΩ = (~5°)(~3.6°) ≈ 5.4×10⁻³ sr → consistency check ≈ 8.4×10⁻⁶, 측정 8.67 ×10⁻⁶ 와 ~3% 일치. Order-of-magnitude consistency with measured $G$.

### 4.4 카운트 → 위상 공간 밀도 / Counts to phase-space density

기기에서 측정한 카운트 $C_i$ 와 위상 공간 밀도 $f_i$ 의 관계 / The relation between measured counts and phase-space density:

$$f(\vec{v}) = \frac{C_i}{G_i \, \tau \, v_i^4 \cdot (\Delta E / E)}$$

핵심 인자 $v^4$는 두 효과의 곱이다 / The crucial $v^4$ factor combines two effects:
- $v^2$: 미분 플럭스 → 위상 밀도 ($f \propto J / E$, $J \propto v^2 f$)
- $v^2$: 에너지-속도 자코비안 ($d^3 v = v^2 \, dv \, d\Omega$, $dE = m v \, dv$)

결합하면: $dE / d^3v = m v / v^2 = m/v$ → $f = J/(2 E^2/m^2) = J \cdot m^2 / (2 E^2)$ → $J \cdot m / (m^2 v^4 / 2)$ → $f \propto 1/v^4$.

### 4.5 VDF 모멘트 → 플라즈마 매개변수 / VDF moments → plasma parameters

밀도, 벌크 속도, 온도 텐서, 압력, 열속 / Density, bulk velocity, temperature tensor, pressure, heat flux:

$$n = \int f \, d^3 v$$

$$\vec{u} = \frac{1}{n} \int \vec{v} f \, d^3 v$$

$$P_{jk} = m \int (v_j - u_j)(v_k - u_k) f \, d^3 v$$

$$T = \frac{1}{3 n k_B} \, \mathrm{Tr}(P)$$

$$\vec{q} = \frac{1}{2} m \int |\vec{v} - \vec{u}|^2 (\vec{v} - \vec{u}) f \, d^3 v$$

이산 sample (SWEPAM의 96 픽셀) 에 대한 합 / Discrete sums over the 96 SWEPAM pixels:

$$n \approx \sum_i \frac{C_i}{G_i \, \tau \, v_i^2 \cdot (\Delta E_i / E_i)} \cdot (\text{angular weight})$$

$$\vec{u} \approx \frac{1}{n} \sum_i \vec{v}_i \cdot \frac{C_i}{G_i \tau v_i^2 (\Delta E_i / E_i)}$$

수치 예 / Numerical example (slow wind):
- $n_p = 5 \, \mathrm{cm}^{-3}$, $|\vec{u}_p| = 400$ km/s, $T_p = 5 \times 10^4$ K (≈ 4.3 eV)
- 양성자 thermal speed $v_{\text{th}} = \sqrt{2 k_B T / m_p}$ ≈ 29 km/s
- Mach number $M = u/v_{\text{th}}$ ≈ 14 (very supersonic, narrow beam)
- 빔 차원 ΔE/E (FWHM, full beam) ≈ 2 v_th / u ≈ 14% → SWEPAM의 ΔE/E = 5%로 ~3 levels에 걸쳐 분해됨. With $u/v_{\text{th}}$ ≈ 14, the beam is highly supersonic; ΔE/E (FWHM, beam) ≈ 14% spans ~3 SWEPAM voltage levels.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1958 ── Parker: Theoretical solar wind prediction
        │
1962 ── Mariner 2 (Neugebauer & Snyder): First solar wind in situ
        │
1972 ── HEOS-2 / Helios: ESA + CEM solar wind plasma instruments
        │
1977 ── ISEE-3 launch (L1 monitor heritage; halo electrons from L1)
        │
1978 ── Gosling et al.: Spherical-section ESA azimuthal response
        │  (used directly for SWEPAM calibration math)
        │
1990 ── Ulysses launch (NASA/ESA), SWOOPS instruments
        │  (Bame et al. 1992: documentation paper)
        │
1992 ── Ulysses Jupiter swingby into polar orbit
        │
1994-95 Ulysses fast latitude scan; bimodal solar wind discovered
        │
1997 ── ACE launch (August 25, 1997); SWEPAM commissioning at L1
        ★
1998 ── ★ McComas et al.: SWEPAM documentation paper (this paper)
        │
1998 ── Zwickl et al.: NOAA Real-Time Solar Wind system (companion)
        │
2003 ── Halloween storms — RTSW operationally vital
        │
2006 ── STEREO launch with PLASTIC (SWEPAM-style heritage)
        │
2015 ── DSCOVR replaces ACE primary RTSW role
        │
2018 ── Parker Solar Probe SWEAP-SPC/SPAN (top-hat ESA evolution)
        │
2020 ── Solar Orbiter SWA (PAS, EAS, HIS — direct lineage)
        │
2025 ── IMAP launch (RTSW continuity)
        ▼
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Parker (1958, ApJ 128, 664) — *Dynamics of Interplanetary Gas* | 태양풍 이론적 예측의 원조; SWEPAM이 측정하는 모든 매개변수의 원천 동기 | 매우 높음 / Very high — theoretical foundation of what SWEPAM measures |
| Bame et al. (1992, A&AS 92, 237) — *Ulysses Solar Wind Plasma Experiment* | SWOOPS 1차 문서; SWEPAM의 직접적 헤리티지(비행 예비품) | 매우 높음 / Very high — direct hardware heritage |
| Gosling, Asbridge, Bame (1978, RSI 49, 1260) — *Effects of a Long Entrance Aperture upon the Azimuthal Response of Spherical Section ESAs* | 응답 함수 분리·기하 인자 계산 수학의 토대; SWEPAM 보정 분석에 직접 사용 | 매우 높음 / Very high — calibration math foundation |
| Stone et al. (1998, SSR 86, 1) — *The Advanced Composition Explorer* | ACE 임무 overview 논문; SWEPAM의 컨텍스트 페이로드 정의 | 높음 / High — mission context |
| Smith et al. (1998, SSR 86, 613) — *MAG: Magnetic Field Investigation* | ACE의 MAG; SWEPAM 플라즈마 측정과 결합되어 IMF 위상학 진단 | 높음 / High — combined plasma+B for CME diagnostics |
| Gloeckler et al. (1998, SSR 86, 497) — *SWICS/SWIMS: Composition Spectrometers* | SWEPAM이 *컨텍스트*를 제공하는 두 조성 실험; charge state vs. bulk wind | 높음 / High — companion composition |
| Zwickl et al. (1998, SSR 86, 633) — *NOAA Real-Time Solar Wind System* | RTSW 운영 시스템 — SWEPAM의 24/7 우주 기상 데이터 스트림 사용 | 매우 높음 / Very high — companion RTSW paper |
| Gosling (1996, AIP Proc. 382, 438) — *Magnetic Topologies of CMEs* | counter-streaming suprathermal electrons (CME 식별자); SWEPAM-E 핵심 동기 | 높음 / High — CME diagnostic motivation |
| Hammond et al. (1995, JGR 100, 7881) — *Solar Wind Double Ion Beams and the HCS* | HCS 인근 더블 빔 자기 재결합 증거; SWEPAM-I 고분해능 동기 | 중간 / Medium — streamer-belt motivation |

---

## 7. References / 참고문헌

**Primary paper / 본 논문**
- McComas, D. J., Bame, S. J., Barker, P., Feldman, W. C., Phillips, J. L., Riley, P., and Griffee, J. W., "Solar Wind Electron Proton Alpha Monitor (SWEPAM) for the Advanced Composition Explorer", *Space Sci. Rev.* **86**, 563-612, 1998. DOI: 10.1023/A:1005040232597

**Direct ACE/Ulysses heritage / 직접적 헤리티지**
- Bame, S. J. et al., "The Ulysses Solar Wind Plasma Experiment", *Astron. Astrophys. Suppl. Ser.* **92**, 237, 1992.
- Stone, E. C. et al., "The Advanced Composition Explorer", *Space Sci. Rev.* **86**, 1, 1998.
- Zwickl, R. et al., "The NOAA Real-Time Solar-Wind (RTSW) System Using ACE Data", *Space Sci. Rev.* **86**, 633, 1998.

**Calibration / instrument theory**
- Gosling, J. T., Asbridge, J. R., Bame, S. J., and Feldman, W. C., "Effects of a Long Entrance Aperture upon the Azimuthal Response of Spherical Section Electrostatic Analyzers", *Rev. Sci. Inst.* **49**, 1260, 1978.
- McComas, D. J. and Bame, S. J., "Channel Multiplier Compatible Materials and Lifetime Tests", *Rev. Sci. Inst.* **55**, 463, 1984.

**Solar wind science motivation / 태양풍 과학 동기**
- Parker, E. N., "Dynamics of the Interplanetary Gas and Magnetic Fields", *Astrophys. J.* **128**, 664, 1958.
- McComas, D. J. et al., "Ulysses' Return to the Slow Solar Wind", *Geophys. Res. Lett.* **25**, 1, 1998.
- Gosling, J. T., "Magnetic Topologies of CME Events: Effects of 3-D Reconnection", *Solar Wind Eight*, AIP Proc. 382, 438, 1996.
- Hammond, C. M. et al., "Solar Wind Double Ion Beams and the HCS", *J. Geophys. Res.* **100**, 7881, 1995.
- Hundhausen, A. J., 1977, "An Interplanetary View of Coronal Holes", in *Coronal Holes and High Speed Wind Streams*, Colorado Associated Univ. Press, p. 225.

**CME / shock / particle acceleration**
- Gosling, J. T. et al., "Bidirectional Solar Wind Electron Heat Flux Events", *J. Geophys. Res.* **92**, 8519, 1987.
- McComas, D. J. et al., "Electron Heat Flux Dropouts in the Solar Wind", *J. Geophys. Res.* **94**, 6907, 1989.
- McComas, D. J. et al., "Magnetic Reconnection Ahead of a CME", *Geophys. Res. Lett.* **21**, 1751, 1994.
- Gosling, J. T. et al., "Counterstreaming Suprathermal Electron Events Upstream of Corotating Shocks at Ulysses Beyond 2 AU", *Geophys. Res. Lett.* **20**, 2335, 1993.
- Isenberg, P. A. and Feldman, W. C., "Electron-Impact Ionization of Interstellar Hydrogen and Helium at Interplanetary Shocks", *Geophys. Res. Lett.* **22**, 873, 1995.
- Feldman, W. C. et al., "Electron Impact Ionization Rates for Interstellar H and He Atoms Near Interplanetary Shocks: Ulysses Observations", *Solar Wind Eight*, AIP Proc. 382, 622, 1996.
