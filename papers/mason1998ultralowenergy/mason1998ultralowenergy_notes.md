---
title: "The Ultra-Low-Energy Isotope Spectrometer (ULEIS) for the ACE Spacecraft"
authors: [G. M. Mason, R. E. Gold, S. M. Krimigis, J. E. Mazur, G. B. Andrews, K. A. Daley, J. R. Dwyer, K. F. Heuerman, T. L. James, M. J. Kennedy, T. Lefevere, H. Malcolm, B. Tossman, P. H. Walpole]
year: 1998
journal: "Space Science Reviews"
doi: "10.1023/A:1005079930780"
topic: Space_Weather
tags: [ULEIS, ACE, time-of-flight, isotope, mass-spectrometer, SEP, instrument-paper, MCP]
status: completed
date_started: 2026-04-25
date_completed: 2026-04-25
---

# 69. The Ultra-Low-Energy Isotope Spectrometer (ULEIS) for the ACE Spacecraft / ACE 우주선용 초저에너지 동위원소 분광기

---

## 1. Core Contribution / 핵심 기여

이 논문은 ACE 우주선의 4종 분광기 가운데 하나인 **ULEIS**(Ultra-Low-Energy Isotope Spectrometer)의 설계, 보정, 비행 전 성능을 종합적으로 보고하는 기기 논문이다. ULEIS는 헬륨(He, Z=2)부터 니켈(Ni, Z=28)까지의 원소를 ~45 keV/nucleon에서 수 MeV/nucleon 영역에서 동위원소 수준의 질량 분해능으로 측정한다. 핵심 기술은 (1) 두 박막(START-1, START-2)에서의 이차전자 방출과 정전 거울에 의한 등시(等時, isochronous) 결상, (2) Z-stack MCP의 ~5×10⁶ 게인과 wedge-and-strip 양극의 (x, y) 위치 결정, (3) ~50 cm의 긴 비행거리에서 <300 ps FWHM의 시간 분해능, (4) 7개 SSD로 구성된 ~73 cm² 잔류에너지 측정 어레이의 결합이다. 이 결합으로 ULEIS는 σ_m ≈ 0.04 amu (⁴He), 0.13 amu (¹⁵N), 0.17 amu (¹⁶O), 0.20 amu (²⁸Si), 0.33 amu (⁵⁶Fe)의 분해능을 1–2 MeV/nuc에서 입증했고, 이는 C–Si 인접 동위원소를 분리하는 데 충분한 수준이다.

This is the instrument paper for ULEIS, one of four mass spectrometers on the ACE mission. ULEIS measures elements from helium (Z=2) through nickel (Z=28) at ~45 keV/nucleon to several MeV/nucleon with isotopic mass resolution. The core technique combines (1) secondary-electron emission from two thin foils (START-1, START-2) imaged isochronously onto MCPs by electrostatic mirrors, (2) triple-MCP "Z-stacks" with ~5×10⁶ gain feeding wedge-and-strip anodes for (x, y) position, (3) a ~50 cm flight path with <300 ps FWHM timing precision, and (4) a seven-element silicon SSD array (~73 cm² active area) measuring residual kinetic energy. Pre-flight calibration with ⁴He, ¹⁵N, ¹⁶O, ²⁸Si and ⁵⁶Fe accelerator beams demonstrated mass resolution σ_m of 0.04, 0.13, 0.17, 0.20, and 0.33 amu, respectively, at 1–2 MeV/nuc — sufficient to resolve adjacent isotopes from C through Si and even-mass isotopes through Fe. The geometric factor of ~1.27 cm² sr at 100% duty cycle, the dynamic range from 1 event/week to 10⁵ events/s (via four-position iris), and the 100% duty cycle make ULEIS uniquely suited to span small ³He-rich flares to the largest CME-driven shock events.

---

## 2. Reading Notes / 읽기 노트

### Part I: Scientific Goals (Section 1) / 과학적 목표

논문 첫 절은 ULEIS가 다룰 4가지 모집단의 과학적 동기를 밝힌다. (i) **태양 동위원소 풍부도** — 분광선이 동위원소를 구분하지 못하므로 SEP의 동위원소 측정이 필수이며, 이전 측정들은 ²²Ne/²⁰Ne의 사건 간 변동을 시사했다(SAMPEX). (ii) **코로나 조성** — Galvin 등의 Ulysses 측정이 FIP(First Ionization Potential) 분획이 황도면에서 강하게 나타나고 극풍에서는 약함을 보였다. (iii) **회전 상호작용 영역(CIR)** — 고속·저속 태양풍의 충돌로 형성되는 forward-reverse 충격파에서 가속된 이온이 CIR 조성을 보여주며 픽업 이온 기여도 가능. (iv) **이상우주선(ACR)** — 성간 중성 원자가 종단 충격파에서 가속된 입자, ~10 MeV/nuc 부근의 동위원소 비가 국부 성간 매질의 조성을 알려준다.

The opening section frames the science around four particle populations: (i) **Solar isotopic abundances** — since spectroscopy cannot resolve isotopes, SEPs are the only route; SAMPEX hinted at event-to-event ²²Ne/²⁰Ne variability. (ii) **Coronal composition** — Ulysses revealed strong FIP fractionation in the slow ecliptic wind but weak fractionation over the poles, supporting magnetic-field-mediated ion-neutral separation models (von Steiger & Geiss 1989; Fisk, Schwadron & Zurbuchen 1998). (iii) **Corotating interaction regions** — forward/reverse shock pairs in CIRs accelerate solar-wind seed ions; CIR abundances at 1 AU lie between fast and slow solar-wind values (Mason et al. 1997), with possible pickup-ion contributions (Gloeckler et al. 1994). (iv) **Anomalous Cosmic Rays** — interstellar neutrals ionized and accelerated at the termination shock to ~10 MeV/nuc; SAMPEX showed the singly→multiply charged transition near 20 MeV/nuc (Mewaldt et al. 1996), constraining the acceleration cutoff to ~240 MV/Q and the timescale to ~1 year (Jokipii 1996).

The figures of this section establish the energy spectra ULEIS must capture. **Figure 3** (Mazur et al. 1992) shows the 28 April 1978 SEP event H/He/O/Fe spectra spanning ~0.1–100 MeV/nuc, with stochastic-shock fits, and abundance ratios H/O, Fe/O that depend on energy — implying the acceleration efficiency depends on Q/m. **Figure 4** (Reames et al. 1997, EPACT/WIND) shows gradual SEP spectra extending down to ~20 keV/nuc — the lower energy boundary that ULEIS must capture. **Figure 5** shows five impulsive flare ³He and Fe spectra continuing to rise at the lowest energies (~10 keV/nuc), suggesting the bulk of the flare-released energy lies below traditional measurement bands — motivating ULEIS's 45 keV/nuc threshold (page 414–417, Figs 3–5).

논문의 Figure 1은 SEP/태양풍 비를 Z의 함수로 보이며, 0.3 MeV/nuc 이상의 평균에서는 태양풍과 거의 일치하지만 4.9–22 MeV/nuc 구간에서는 큰 분획이 나타남을 보여준다(Mazur et al. 1993). 이 사실은 에너지에 따른 가속 효율 차이가 동위원소 측정에서도 나타날 가능성을 시사하며, ULEIS의 광범위 에너지 커버리지의 정당성이 된다.

Figure 1 (page 413) plots SEP/solar-wind ratios versus Z. Above 0.3 MeV/nuc the ratios cluster around unity — averaged over many events the SEP composition matches the solar wind. But for 4.9–22.5 MeV/nuc particles in the same events, deviations of factors of 2–3 appear, motivating ULEIS's broad-energy mapping of isotopes and elements together.

### Part II: Design Requirements (Section 2) / 설계 요구사항

핵심은 **Table I** (page 419)이다.

| Goal / 목표 | Value / 값 | Science driver / 과학 동기 |
|---|---|---|
| Geometric factor / 기하 인자 | 1 cm² sr | 작은 사건 통계 + ACR / counting statistics for small events + ACR |
| Z range / 원소 범위 | 2 ≤ Z ≤ 28 | He–Ni, 핵합성 주계열 / He–Ni main nucleosynthetic sequence |
| Energy range / 에너지 범위 | 0.3–2.0 MeV/n | 대형 SEP, 임펄시브 플레어, IP 충격파, CIR / large SEPs, impulsive flares, IP shocks, CIRs |
| Mass resolution σ_m | <0.15 amu (Z=6); <0.5 amu (Z=26) | C-Si 인접 동위원소; Fe짝수질량 / adjacent C–Si isotopes; even-mass Fe |
| Event rate range R | 1/week < R < 10⁵ s⁻¹ | ACR(저)~대형 SEP(고) / ACR (low) to large SEP (high) |

이 표에서 **σ_m ~ 0.2 amu**는 결정적이다. 주 동위원소 피크는 인접 희소 동위원소보다 50–100배 풍부하므로, σ_m가 0.2 amu보다 크면 인접 피크가 합쳐져 식별 불가능. 1 cm² sr × 100% 듀티 사이클은 ACR 등 매우 약한 신호를 잡기 위한 필수 조건이다. 데이터 처리 단(DPU)은 ~1 kbit/s 텔레메트리 한계 때문에 상세 PHA 이벤트는 초당 4개만 전송하고, 나머지는 매트릭스 박스에 species/energy로 분류해 카운트율로 텔레메트리한다 — 즉 "matrix rates"는 통계량을 살리되 대역폭을 절약하는 핵심 발명이다.

The design table is the heart of Section 2. **σ_m ~ 0.2 amu** is the gate: main isotope peaks are 50–100× more abundant than adjacent rare isotopes, so any wider σ_m would prevent ground analysis from separating them. The 1 cm² sr × 100% duty-cycle target is needed for ACR sensitivity. The DPU's ~1 kbit/s downlink cannot transmit full pulse-height data for every event (the telescope sees up to 10⁵ s⁻¹), so only 4 PHA events are telemetered per second; the rest are binned on board into "matrix rates" — pre-defined boxes in TOF-vs-E space tagged with species and energy. This bandwidth-saving design is itself an instrument-level innovation.

### Part III: Telescope Schematic (Section 3.1–3.2) / 망원경 광학

Figure 6 (page 418)은 ULEIS의 단면도이다. 위에서 아래로 광선 경로를 따라가면:

Reading Figure 6 from top to bottom along the typical ion path:

1. **Sliding iris (셔터)**: 네 위치(100%, 25%, 6%, 1%)로 입사면적을 줄여 강한 사건 시 START-1 카운트율이 ~5×10⁶ s⁻¹에 도달해도 시간 시스템을 보호한다. Clifton 11형 스테퍼 모터로 구동되며 광스위치로 위치 확인. 6%/1%는 셔터 자체의 구멍으로 구현하므로 정밀 위치 의존성이 없음.
   Four-position aperture stop; reduces entrance area when START-1 rate exceeds ~5×10⁶/s; stepper motor driven, optical-switch sensed; 6%/1% via fixed holes in the cover (page 422–423).

2. **Sunshade (선쉐이드)**: ACE 회전축이 태양을 향하므로, ULEIS 망원경 축은 태양에서 60° 떨어진 방향. 선쉐이드는 직접 태양광이 START-1 박막에 닿는 것을 차단해 UV 광전자 노이즈를 줄임.
   Sunshade prevents direct sunlight on START-1; instrument boresight at 60° to ACE spin axis; the FOV scans a band perpendicular to the ecliptic.

3. **Entrance harp (진입 하프)**: 0.001″ 스테인리스 선, 1 mm 간격, 투명도 97.5%. 음전위 START-1 박막을 정전적으로 차폐.
   Entrance harp at ground potential electrostatically shields the negatively-biased START-1 foil; 97.5% transparency.

4. **START-1 foil (시작-1 박막)**: 2000 Å 폴리이미드 + 양면 300 Å Al 코팅; -4000 V로 바이어스. 이온 통과 시 박막 내면(이온 진행 방향)에서 이차전자 방출. **표 IV**가 박막의 정확한 두께(폴리이미드 22.8 ± 1.1 µg/cm², Al 8.1 µg/cm² × 2)를 제공.
   START-1 foil: 2000 Å polyimide with 300 Å Al on both sides, biased −4000 V; secondary electrons emitted from the inner surface as ions traverse. Total foil mass listed in Table IV (~50 µg/cm² polyimide+Al combined).

5. **Wedge 어셈블리** — 박막, 정전 거울(45°), MCP Z-stack을 삼각형 모듈로 묶음. 이차전자는 가속 하프(-3000 V)를 통해 균일 전위 영역으로 들어가 정전 거울에서 90° 반사된 뒤 MCP 전면(+50 V)으로 향함. 거울 외측 하프는 접지, 내측 하프는 가속 전위와 같음. 등시 광학으로 박막 임의 위치에서 출발한 전자가 동일한 시간에 MCP에 도달.
   Wedge assembly: foil + electrostatic mirror + MCP triple-stack as a single triangular module. Secondary electrons accelerated by the −3000 V harp into a uniform-potential drift region, reflected 90° by the electrostatic mirror, and detected by the MCP biased +50 V positive of the drift region; the mirror geometry is engineered to be **isochronous** so that electrons from any (x, y) on the foil arrive simultaneously at the MCP.

6. **START-2 + STOP foils**: START-2는 진입에서 17.4 cm, STOP은 50 cm 뒤. START-1과 STOP 사이 비행거리 50.0 ± 0.1 cm, START-2와 STOP 사이 32.6 ± 0.1 cm. 경로 비율 ~1.53이 이중 TOF 일관성 검사의 기반.
   START-2 and STOP foils: START-2 at 17.4 cm into the path, STOP at 50.0 cm; the START-1→STOP and START-2→STOP path lengths are 50.0±0.1 cm and 32.6±0.1 cm, respectively; ratio fixed at ~1.53 used for TOF consistency check (Sec. 4.4).

7. **Solid-state detector array (SSD)**: 7개 검출기가 8×10 cm 활성 영역의 90% 이상을 덮음. 4개의 작은 D1–D4 (9.4×48 mm 각각)는 저에너지 시스템(검출기 알파 피크 25 keV FWHM, 임계값 55 keV); 3개의 큰 D5–D7 (38×48 mm)은 고에너지 시스템(45 keV FWHM, 임계값 500 keV, 0.5–160 MeV 범위). 모두 500 ± 30 µm 두께 Si.
   Seven Si SSDs cover >90% of the 8×10 cm active area: four small D1–D4 (9.4×48 mm; 25 keV FWHM α-peak; 55 keV threshold) form the low-noise / low-threshold system, and three large D5–D7 (38×48 mm; 45 keV FWHM; 500 keV threshold) the high-energy system. All Si is 500 ± 30 µm thick with 2000 ± 1000 Å Al front contact.

**Foil transparency budget (Table III, page 424)**: 6 harps + 1 entrance harp + 3 foil meshes give cumulative transparency 0.596. 즉 입사면을 통과한 이온의 ~60%만이 모든 메시를 거쳐 SSD에 도달.
The cumulative geometric transparency through the harp/mesh stack is 0.596 — only ~60% of ions making it through all wires reach the SSD.

### Part IV: MCP Z-stack and WSA (Section 3.2.4–3.2.5) / MCP 적층과 WSA

각 MCP Z-stack은 80 × 100 mm 크기의 세 장의 마이크로채널 플레이트(1 mm 두께, 25 µm 채널, 19° bias angle)를 직접 접촉으로 쌓아 ~5×10⁶ 게인 (3000 V/set)을 얻는다. 단일 플레이트 연수 전류는 10–25 µA. 적층 구성에서 인접 플레이트 간격이 큰 경우 가운데 영역(전자 구름이 옆으로 퍼짐)의 게인이 ~5배 증가하여 응답이 비균일해지므로, ULEIS 비행 모델은 UV 보정으로 가장 균일한 응답을 보이는 방향으로 플레이트를 조립했다.

Each MCP Z-stack stacks three 80×100 mm × 1-mm-thick MCPs with 25 µm pores at 19° bias, giving ~5×10⁶ combined gain at 3000 V/set. Where plates are not perfectly flat, gaps between plates allow the electron cloud to spread, producing ~5× higher gain in those (typically central) regions. For the flight units, plate orientations were UV-tested and selected for the most uniform response (page 426–427).

**Wedge-and-strip anode (WSA)** — Anger 1966에서 유래한 charge-division 위치 검출 양극. ULEIS의 WSA는 1 mm 알루미나 기판에 W/S/Z 세 전극을 0.002″ 간극으로 두꺼운 막 인쇄. 30 cell × 8.5×10.4 cm 활성 영역. 전자 구름의 무게중심을 (x', y')로 디코딩하는 핵심 식이 Eq. (2)–(3):

The wedge-and-strip anode (Anger 1966; Lapington & Schwarz 1986) is a charge-division position-sensitive readout. ULEIS uses 30-cell pattern over 8.5×10.4 cm on alumina, with three electrodes (W=wedge, S=strip, Z=zigzag) separated by 0.002″ gaps. The centroid of the electron cloud is recovered through

$$
x' = \frac{Q_S - X_{\text{talk}}\,Q_Z}{Q_W + Q_S + Q_Z}, \qquad y' = \frac{Q_W - X_{\text{talk}}\,Q_Z}{Q_W + Q_S + Q_Z}
$$

X_talk는 양극 간 용량성 결합(수백 pF)에 의한 누화 보정. 보정 후 위치 정확도 수 mm — 1.85 MeV/nuc ⁴⁰Ar 빔에서 σ ≈ 3 mm로 측정됨(Figure 14). 이 정확도는 동위원소 분리에 필요한 ~1 cm 경로 길이 보정에 충분.

X_talk corrects for capacitive coupling (few-hundred pF) between anode regions. The position resolution achieved (Figure 14, page 439) is σ ≈ 3 mm for 1.85 MeV/nuc ⁴⁰Ar at all three foils, well below the 1-cm accuracy required for path-length correction in isotope analysis.

### Part V: Electronics and Data Processing (Section 3.3) / 전자공학과 데이터 처리

망원경 박스 안의 전자공학(Figure 11, page 429)은 SSD 부근의 저잡음 요건과 검출기 ~0°C 운용을 위한 저전력 요건의 절충이다. TOF 회로 등 고전력 회로는 외부 아날로그 박스에 분리.

**Event format (Table V, page 430)**: 각 PHA 이벤트는 14 워드 × 12 비트 = 168 비트. 워드 0–8: WSA 펄스 높이 (W/S/Z × START-1/START-2/STOP = 9), 워드 9: SSD 에너지, 워드 10/11: TOF-1, TOF-2, 워드 12/13: 트리거 검출기, 매트릭스 박스 번호, 캘리브레이션 플래그 등. 정상 모드에서 4 이벤트/s 텔레메트리.

Each PHA event = 14 12-bit words = 168 bits. Words 0–8 are W/S/Z pulse heights for the three wedges; word 9 is SSD energy; words 10–11 are TOF-1, TOF-2; words 12–13 carry detector ID, matrix-box number, and mode flags. Four PHA events per second downlinked.

**Singles rates (Table VII)**: 16개의 24-bit 누산기에서 D1–D7, START-1/2, STOP, Valid Stop 1/2, Event(=VS·E·/BUSY), wedge sums을 spin (12 s)당 8 sector로 누적, 두 spin마다 (24 s) 텔레메트리.

Singles-rate accumulators: 16 24-bit counters logging D1–D7, START-1/2, STOP, Valid Stops, valid-events, and wedge sums; sectored 8/spin and read every 2 spins (24 s).

**Matrix rates (Section 3.4.2, Figure 12, Table VIII)**: 시간-비행 vs 에너지 평면을 species×energy 박스로 분할. 34 매트릭스율 (12 s 주기) + 42 매트릭스율 (24 s 주기) = 76 박스. 박스 정의는 lookup table로 비행 중 갱신 가능. 박스에는 우선순위 매겨 PHA 이벤트 텔레메트리 분배도 제어. H/³He/⁴He은 매 spin, 무거운 종은 두 spin마다.

Matrix rates: the TOF×E plane is divided into species/energy boxes (Figure 12, p. 434; Table VIII, p. 433); H, ³He, ⁴He read every 12 s, heavier species every 24 s. Boxes are uploadable lookup tables, supporting in-flight reconfiguration. Priority assignments rotate so each species sees high PHA telemetry priority.

### Part VI: Performance — Mass Resolution Theory (Section 4.1–4.3) / 성능: 질량 분해능

**Geometric factor (Table X, page 437)**: iris 100%에서 D1–D4 합 0.314, D5–D7 합 0.954, 총 1.268 cm² sr (harp/foil 투명도 포함). iris 1%에서 0.0161 cm² sr. 이 단계적 감소가 강한 사건에서 통계적 정확도를 유지하는 핵심.

ULEIS geometric factor (Table X): 1.268 cm² sr at 100% iris (D5–D7 dominate at 0.954), dropping to 0.0161 cm² sr at 1% iris — keeps trigger rate and pile-up under control in the largest events while preserving useful statistics.

**Mass resolution equation (Eq. 4)** is the core diagnostic:

$$
\left(\frac{\sigma_m}{m}\right)^{2} = \left(\frac{\sigma_E}{E}\right)^{2} + \left(\frac{2\sigma_\tau}{\tau}\right)^{2} + \left(\frac{2\sigma_L}{L}\right)^{2}
$$

세 기여 항:

The three contributions are:

- **σ_E/E (energy term)**: SSD 노이즈가 약 25 keV (소형) / 45 keV (대형) FWHM. 저에너지에서 입사 에너지의 큰 분율을 차지. 박막 통과 후 잔류 에너지 측정.
  σ_E/E (energy term): set by SSD noise ~25 keV / 45 keV FWHM; large fraction of deposited E at low energies; dominant below ~0.6 MeV/nuc.
- **2σ_τ/τ (TOF term)**: 시간 산포 σ_τ는 기본적으로 ~115 ps (위치/거울 기여)에 ~50 ps (전자공학 walk) 등을 합산해 <130 ps FWHM 목표. τ는 에너지 증가에 따라 감소하므로 이 항은 고에너지에서 지배.
  2σ_τ/τ (timing term): dispersion ~115 ps from mirror plus electronics; total <130 ps target FWHM. τ shrinks with energy, so this term dominates above ~1 MeV/nuc.
- **2σ_L/L (path length term)**: 비행거리 50 cm에서 σ_L ~ 3 mm (위치 분해능에서) → σ_L/L ~ 0.0006. 항상 다른 두 항보다 작음.
  2σ_L/L (path term): with σ_L ≈ 3 mm at L = 500 mm, σ_L/L ≈ 6×10⁻⁴ — always sub-dominant.

**Figure 13 (page 438)**은 Eq. (4)의 세 항을 ¹⁶O에 대해 입사 에너지의 함수로 보여준다. σ_m/m 곡선은 ~1 MeV/nuc 부근에서 최소를 가진다. 이는 저에너지에서 σ_E/E 폭발(에너지가 노이즈에 비해 작아지므로)과 고에너지에서 2σ_τ/τ 증가(τ → 작은 값) 사이의 절충에서 나오는 자연스러운 우물 형태.

Figure 13 plots all three terms for ¹⁶O versus incident energy. The composite σ_m/m curve has a minimum near 1 MeV/nuc — the natural valley between the energy-noise blowup at low energy and the timing blowup at high energy. The mass resolution at this minimum is σ_m/m ~ 6×10⁻³, equivalent to σ_m ≈ 0.1 amu at mass 16.

### Part VII: Performance — Calibration Measurements (Section 4.3.1) / 보정 결과

가속기 보정은 Brookhaven Tandem Van de Graaff와 Lawrence Berkeley 88인치 사이클로트론에서 수행.

**Position (Figure 14, page 439)**: 1.85 MeV/nuc ⁴⁰Ar 빔, 2 mm 직경 콜리메이터, 정상 입사. START-1, START-2, STOP 박막의 (x') 분포 모두 σ = 3 mm. 박막 간 빔 직경 변화 미미 → ULEIS 박막에서의 쿨롱 산란이 무시할 수준.

The position calibration: 2 mm diameter ⁴⁰Ar beam at 1.85 MeV/nuc, normal incidence; all three foils show σ = 3 mm in x; the beam diameter is nearly unchanged through the three foils, indicating negligible Coulomb scattering in the ULEIS foils.

**TOF (Figure 15, page 440)**: 1.84 MeV/nuc ¹⁵N, START-1↔STOP TOF = 26.6 ns. 측정된 σ_τ = 114 ps, 즉 σ_τ/τ = 0.0045. 설계 목표 130 ps보다 12% 양호. 이 값은 거울/wedge 분산과 잔류 walk만 포함하며 박막 안의 에너지 손실/스트래글링은 작음.

TOF calibration: 1.84 MeV/nuc ¹⁵N gives mean τ = 26.6 ns (START-1→STOP); the measured σ_τ = 114 ps is 12% better than the 130 ps design goal; σ_τ/τ = 0.0045. Energy loss/straggling in foils is small at this energy; the 114 ps reflects mirror and electronics dispersion.

**Energy (Section 4.3.1.3)**: 2 MeV/nuc ¹⁶O는 SSD에서 32.0 MeV을 잔류 에너지로 남기고 σ_E = 0.160 MeV. σ_E/E = 0.005. 모든 불확정성이 에너지 시스템에서만 온다 가정 시 σ_m/m = 0.005 → σ_m = 0.08 amu (mass 16). 이는 0.16 amu 설계 목표보다 2배 우수.

Energy calibration: 2 MeV/nuc ¹⁶O → 32.0 MeV deposited, σ_E = 0.160 MeV → σ_E/E = 0.005, contributing σ_m = 0.08 amu at mass 16 if it were the sole contributor — a factor of 2 below the design goal.

**Mass resolution (Figure 16, page 441)**: 종합 1–2 MeV/nuc 측정 결과:

The aggregate measurements (Figure 16):

| Species / 종 | σ_m (amu) | Energy / 에너지 |
|---|---|---|
| ⁴He | 0.04 | ~1 MeV/n |
| ¹⁵N | 0.13 | 1.84 MeV/n |
| ¹⁶O | 0.17 | ~1 MeV/n |
| ²⁸Si | 0.20 | 1–2 MeV/n |
| ⁵⁶Fe | 0.33 | 1–2 MeV/n |

**Figure 17**에서 측정값과 Eq. (4)의 예측 곡선은 ~20% 이내로 일치. 특히 측정 지점들은 각 종의 분해능 곡선의 최소 부근에 위치하므로 timing-dominated 영역에서 작동.

Figure 17 (page 442) compares measurements with the Eq. (4) prediction curves; agreement within ~20%. Each measurement sits near each species' minimum, where timing dominates.

### Part VIII: Background Rejection and Efficiency (Section 4.4–4.5) / 배경 제거와 효율

**Dual TOF consistency (Figure 18a, page 443)**: TOF-1/TOF-2 비율은 두 비행거리의 비 50.0/32.6 ≈ 1.534로 고정. 1.84 MeV/nuc ¹⁵N에서 산점도가 이 값 주변에 군집. ±5% 윈도우 밖의 점들은 'Y' 패턴을 형성: (i) 우측 가지 = TOF-1이 ~4 ns 길게 측정 (전자공학 ringing), (ii) 수직 가지 = TOF-2 신호 문제, (iii) 좌상 가지 = 두 TOF 모두 짧지만 비율은 어긋남 (메커니즘 불명). 그러나 모든 배경은 메인 피크의 1% 미만이므로 이중 TOF 일관성 검사가 매우 효과적인 배경 제거 도구.

The two-TOF consistency check exploits the fixed ratio 50.0/32.6 ≈ 1.534. Most events cluster within ±5% of this ratio (Figure 18a, page 443). Outlier "Y"-pattern branches correspond to (i) electronics ringing in TOF-1 (~4 ns offset), (ii) ringing affecting TOF-2 only, (iii) both TOFs short but in wrong ratio (mechanism unclear). Background contamination is <1% of the main peak (Figure 18b), and ground analysis applies the ±5% TOF-1/TOF-2 window for clean isotope studies — essential for measuring rare ¹⁸O at ~0.2% the level of ¹⁶O.

**Triggering efficiency (Section 4.5, Table XI)**: 일차전자 수율은 dE/dx에 따르므로 종/에너지 의존. EPACT/STEP의 0.5 MeV/nuc 측정값(가이드용): H 0.03, He 0.22, C 0.97, O 1.00, Fe 1.00. ULEIS도 무거운 이온은 ~1, He는 분율적, H는 매우 작음 (이는 양자화된 구별을 가능하게 하므로 텔레메트리 보존에 유리).

Trigger efficiency depends on dE/dx (~3–10 secondary electrons emitted per ion at hundreds of keV/nuc). Forward emission (relevant to STARTs) is ~2× backward (relevant to STOP). MCP front dead area ~50%; bias minimizes ion feedback so single-electron trigger probability < 1. Net telescope efficiency is the SQUARE of the per-MCP single-electron trigger probability since both START and STOP must fire. The Table XI estimate from EPACT/STEP at 0.5 MeV/nuc gives H 0.03, He 0.22, C 0.97, O 1.00, Fe 1.00 — heavy ions detected with near-unit efficiency, while H/He are suppressed (helpful for telemetry budget since heavy ions are the science target).

### Part IX: Flight Operations (Section 5) / 비행 운용

ULEIS는 정상 모드 연속 운용. 한 달에 ~1시간 보정 모드를 지상 명령으로 활성화. 슬라이딩 iris 내부에 1 µCi ²⁴⁴Cm (5.80, 5.76 MeV α) 및 0.3 µCi ¹⁴⁸Gd (3.2 MeV α) α-방사선원이 있어 닫힌 상태에서 in-flight 입자 응답 검증 가능. 다중 매개변수 측정으로 검출기 응답 변동을 추적할 수 있고, 매트릭스 박스 정의를 lookup table 갱신으로 보정 가능.

ULEIS runs continuously in normal mode with ~1 hour/month in calibrate mode triggered by ground command. Two α-sources mounted inside the iris cover (1 µCi ²⁴⁴Cm with 5.80/5.76 MeV α, half-life 18 yr; 0.3 µCi ¹⁴⁸Gd with 3.2 MeV α, half-life ~35 yr) enable closed-cover in-flight calibration. Multi-parameter (TOF, E, position) data permit self-monitoring; matrix-box definitions are uploadable to absorb detector drifts.

---

## 3. Key Takeaways / 핵심 시사점

1. **TOF × E mass equation as instrument core / TOF×E 질량 방정식이 기기의 핵심** — Eq. (1) m = 2E(τ/L)²은 비상대론적 운동학에서 직접 나오며, ULEIS 모든 설계 결정(50 cm 비행거리, 7-element SSD, MCP Z-stack)이 이 식의 세 측정량(E, τ, L)을 정확히 결정하는 방향으로 이루어진다.
   The non-relativistic mass equation drives every design choice: a long L (~50 cm) reduces 2σ_τ/τ; a low-noise SSD reduces σ_E/E; precision (x, y) keeps σ_L small.

2. **Resolution physics is a competing-error well / 분해능 물리는 경쟁 오차의 우물** — Eq. (4)의 합 형태는 σ_m/m이 ~1 MeV/nuc 부근에서 최소를 가지는 우물 형태를 만든다. 저에너지에서는 σ_E/E (SSD 노이즈), 고에너지에서는 2σ_τ/τ (작은 τ)가 지배하며 σ_L/L 항은 항상 작다. 이 우물의 폭이 ULEIS의 사용 가능 에너지 범위를 정의한다.
   The summed-error structure of Eq. (4) creates a U-shape in σ_m/m. The energy-noise term blows up at low E; the timing term blows up at high E (small τ). The U-bottom near 1 MeV/nuc defines the sweet spot, with usable resolution extending across decades in either direction.

3. **Isochronous secondary-electron optics enable the long flight path / 등시 이차전자 광학이 긴 비행거리를 가능케 함** — wedge 어셈블리(박막 + 정전 거울 + MCP)는 박막의 임의 (x, y)에서 출발한 이차전자를 같은 시간에 MCP에 결상한다. 이 isochronous 설계가 없다면 큰 비행거리에서 위치 의존 타이밍 오차가 ~ns 수준에 이르러 σ_m을 망친다.
   The isochronous mirror geometry is the unsung hero: by ensuring that secondary electrons from any foil position arrive simultaneously at the MCP, ULEIS decouples timing from impact location and enables the long L-arm.

4. **Dual TOF for background rejection / 이중 TOF로 배경 제거** — START-1과 START-2의 두 비행 시간을 측정하여 비율 1.534 (=50.0/32.6)와 ±5% 일치하는지 검사함으로써 ringing, accidental coincidence 등을 1% 미만으로 거부. ¹⁸O/¹⁶O ~0.002 같은 희귀 동위원소를 보려면 필수.
   The dual-TOF system provides redundant timing whose ratio is geometrically fixed; events outside the ±5% window are rejected as background, reducing noise to <1% — essential for rare isotopes like ¹⁸O at ~0.2% of ¹⁶O.

5. **Matrix rates conserve telemetry while preserving statistics / 매트릭스 율이 통계를 보존하며 텔레메트리 절약** — 텔레메트리 한계로 4 PHA 이벤트/s만 전송하나 ~10⁵ 이벤트/s를 처리해야 한다. DPU의 species-energy 박스 분류와 우선순위 회전은 모든 종이 충분한 통계를 갖도록 PHA 이벤트를 분배한다. 이는 1990년대 다른 미션에서 채택될 표준이 됨.
   Matrix-rate binning solves the bandwidth problem: only 4 PHA events/s downlinked, but on-board species/energy boxes count every event (~76 boxes total). Priority rotation guarantees each species gets fresh PHA samples — a paradigm later adopted across multiple missions.

6. **Sliding iris extends dynamic range to 5 decades / 슬라이딩 iris로 5단위 동적범위 확장** — 4 위치(100/25/6/1%)로 입사면적을 점진 축소하여 정상 시 1.27 cm² sr를 강한 사건 시 0.016 cm² sr까지 줄임 — 80배 감쇠. 광스위치/홀 위치 감지로 모든 설정이 자가 일관적. 이로써 ACR(주당 수개) ~ 대형 SEP(초당 10⁵)까지 한 기기로 커버 가능.
   The four-position iris compresses 80× dynamic range entirely upstream of the foils, allowing the same instrument to operate from ACR-level (events/week) to peak SEP rates (~10⁵/s) without saturating the timing system. Optical-switch and hole-based position sensing make all four states unambiguous.

7. **In-flight calibration via matrix and α-sources / 매트릭스와 α-방사선원에 의한 비행 중 보정** — 다중매개변수 PHA 이벤트는 TOF-vs-E 평면에서 식별된 종의 트랙으로 검출기 응답 변화를 자가 모니터링. 닫힌 iris의 ²⁴⁴Cm/¹⁴⁸Gd α-방사선원은 한 달에 한 번 절대 보정. 매트릭스 박스 lookup table은 비행 중 갱신 가능.
   The combination of (a) intrinsic redundancy of TOF/E/position multi-parameter data, (b) closed-cover α-sources, and (c) uploadable matrix box definitions provides self-calibration over the multi-decade ACE mission lifetime — explaining why ULEIS is still scientifically useful nearly three decades after launch.

8. **Verification: pre-flight σ_m beats design goals / 검증: 비행 전 σ_m이 설계 목표 능가** — ⁴He 0.04, ¹⁵N 0.13, ¹⁶O 0.17, ²⁸Si 0.20, ⁵⁶Fe 0.33 amu (1–2 MeV/nuc, Figure 16) 모두 σ_m < 0.5 amu (Z=26) 목표를 통과; C-Si 인접 동위원소(σ_m < 0.15 amu) 목표는 ¹⁵N에서 약간 초과하나 가속기 측정의 단일점으로는 충분. 측정-Eq. (4) 예측이 20% 이내 일치하여 이론적 이해도 확립.
   Demonstrated σ_m values for ⁴He through ⁵⁶Fe at 1–2 MeV/nuc all meet or beat the Table I design goals, with measured values within 20% of Eq. (4) predictions — a clean closure between theory, calibration data, and the as-built instrument.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Mass Equation / 질량 방정식

이온의 비상대론적 운동에너지와 속도:

The non-relativistic kinematic identities give:

$$E = \tfrac{1}{2} m v^2, \qquad v = \frac{L}{\tau}$$

이를 결합하면

Combining,

$$
\boxed{\;m \;=\; 2 E \left( \frac{\tau}{L} \right)^{2}\;} \quad \text{(Eq. 1)}
$$

각 변수: m [amu] = 이온 질량, E [MeV] = SSD 잔류 에너지, τ [ns] = 비행시간, L [cm] = 비행거리. 단위 처리에 있어 m_amu·v² (cm²/ns²) → E_MeV의 변환 인자가 들어간다. ULEIS의 표준 단위 환산: 1 MeV/u 의 양성자 속도 ≈ 1.389 cm/ns.

Variables: m in amu, E in MeV, τ in ns, L in cm. With consistent unit conversions (1 MeV/u corresponds to v ≈ 1.389 cm/ns for a proton), the equation gives the ion mass directly.

### 4.2 WSA Position Decoding / WSA 위치 디코딩

세 양극 W, S, Z에 수집된 전하 Q_W, Q_S, Q_Z로부터:

From the three anode charges Q_W, Q_S, Q_Z:

$$
x' \;=\; \frac{Q_S - X_{\text{talk}}\,Q_Z}{Q_W + Q_S + Q_Z}, \qquad y' \;=\; \frac{Q_W - X_{\text{talk}}\,Q_Z}{Q_W + Q_S + Q_Z} \quad \text{(Eqs. 2–3)}
$$

물리적 의미: 분자는 분획적 전하 비, 분모는 총 전하 정규화. X_talk는 전극 간 용량성 결합(수백 pF)을 보정. 최종 (x, y)는 오프셋과 스케일링 후 잔여 왜곡 매핑을 거쳐 박막의 실제 통과점에 1:1 대응 (정전 거울 반사성).

Physical meaning: the numerator gives the fractional charge on each electrode minus a cross-talk correction; the denominator normalizes by the total. The corrected (x, y) maps one-to-one to the foil traversal point because the electrostatic mirror is geometrically reflective.

### 4.3 Mass Resolution Error Propagation / 질량 분해능 오차 전파

Eq. (1)의 로그 미분:

Taking the logarithmic differential of Eq. (1):

$$
\frac{dm}{m} \;=\; \frac{dE}{E} + 2\,\frac{d\tau}{\tau} - 2\,\frac{dL}{L}
$$

독립 표준편차로 가정하면 분산 합:

Treating E, τ, L as independent random variables and adding variances:

$$
\boxed{\;\left(\frac{\sigma_m}{m}\right)^{2} \;=\; \left(\frac{\sigma_E}{E}\right)^{2} + \left(\frac{2\sigma_\tau}{\tau}\right)^{2} + \left(\frac{2\sigma_L}{L}\right)^{2}\;} \quad \text{(Eq. 4)}
$$

세 항의 ULEIS 대표 값 (m=16, ¹⁶O):

Representative ULEIS values (m=16, ¹⁶O):

| Term / 항 | Value at 0.05 MeV/nuc | Value at 1 MeV/nuc | Value at 10 MeV/nuc |
|---|---|---|---|
| σ_E/E | ~0.05 (지배) | ~0.005 | ~0.001 |
| 2σ_τ/τ | ~0.001 | ~0.005 (지배 시작) | ~0.01 (지배) |
| 2σ_L/L | ~0.0006 | ~0.0006 | ~0.0006 |
| **σ_m/m** | ~0.05 | ~0.007 | ~0.01 |
| **σ_m (amu)** | ~0.8 | ~0.11 | ~0.16 |

이 표는 Figure 13의 곡선을 정량적으로 재현한 것이다.

This table reproduces Figure 13 quantitatively.

### 4.4 Geometric Factor / 기하 인자

각 SSD의 기하 인자 G_i는 입체각 Ω_i와 면적 A_i의 곱에 누적 투명도 T_total를 곱한 값. ULEIS 기준치 (iris 100%):

The geometric factor for each SSD i is G_i = A_i Ω_i T_total, with cumulative harp+mesh transparency T_total = 0.596:

$$
G_{\text{total}} = \sum_{i=1}^{7} A_i \Omega_i T_{\text{total}} \;\approx\; 1.27\ \text{cm}^2\,\text{sr}
$$

iris 1% 설정에서는 이 값이 0.0161 cm² sr까지 떨어진다 (~80배 감쇠).

At 1% iris, G_total drops to 0.0161 cm² sr, an ~80× attenuation that protects the timing system in the largest events.

### 4.5 TOF Consistency Constraint / TOF 일관성 조건

이중 TOF 시스템의 두 비행거리 비:

The fixed ratio of the two flight paths,

$$
\frac{\tau_1}{\tau_2} = \frac{L_1}{L_2} = \frac{50.0}{32.6} = 1.534 \pm 0.003
$$

배경 제거 조건: |τ_1/τ_2 − 1.534| / 1.534 ≤ 0.05.

Background-rejection criterion: keep events with |τ_1/τ_2 − 1.534|/1.534 ≤ 0.05 (±5% window). Reduces background to <1% of main peak.

### 4.6 Empirical Mass Resolution Summary / 실측 질량 분해능 요약

가속기 보정에서 측정된 1–2 MeV/nuc 분해능 (Figure 16):

Calibration-measured σ_m at 1–2 MeV/nuc (Figure 16):

| Mass / 질량 | σ_m (amu) | σ_m / m | Goal / 목표 (Table I) | Pass? / 통과? |
|---|---|---|---|---|
| 4 (⁴He) | 0.04 | 0.010 | (n/a) | ✓ |
| 15 (¹⁵N) | 0.13 | 0.0087 | <0.15 (Z=6) similar / 유사 | ✓ |
| 16 (¹⁶O) | 0.17 | 0.011 | <0.15 (Z=6) | ≈ marginal |
| 28 (²⁸Si) | 0.20 | 0.0071 | between / 사이 | ✓ |
| 56 (⁵⁶Fe) | 0.33 | 0.0059 | <0.5 (Z=26) | ✓ |

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1966  Anger                  - Position-sensing anode concept (광 → 적용 가능한 충전분할 양극)
1976  Mewaldt, Stone, Vidor  - Isotopic composition of ACR low-energy fluxes (첫 ACR 동위원소)
1978  Fisk                   - ³He-rich flares: plasma resonance preheating model
1978  Ipavich et al.         - Pulse-height defect in Au-Si detectors (계측 baseline)
1982  Anders & Ebihara       - Solar-system abundances compendium (조성 baseline)
1985  Reames, Lin, von Rosenvinge - ³He-rich + non-relativistic electron events
1986  Lapington & Schwarz    - WSA design refined for fast timing
1986  Siegmund et al.        - Z-stack MCP timing characterization
1989  Stone et al.           - ACE Phase A study (ACE 미션 컨셉)
1992  Mazur et al.           - Energy spectra of solar flare H/He/O/Fe (논문 Fig. 3)
1992  SAMPEX 발사            - ACR multi-charge state transition
1993  Mazur et al.           - Abundances of H/He/O/Fe in large SEP events (논문 Fig. 1)
1994  Mason et al.           - Heavy-ion isotopic anomalies in ³He-rich events
1995  Klecker (review)       - ACR composition state of art
1996  Mewaldt et al.         - Multiply charged ACRs from SAMPEX
1997  Reames et al. (EPACT)  - ³He-rich flares & gradual SEP spectra (논문 Fig. 4, 5)
1997  Mason, Mazur et al.    - CIR heavy ion spectral and abundance features
1997 Aug  ACE launch         - ULEIS commissioning
================================================================================
1998  THIS PAPER             - Mason et al. ULEIS instrument paper (Space Sci. Rev. 86)
================================================================================
1998  Companion ACE papers   - Stone et al. (CRIS, SIS); Gloeckler et al. (SWICS, SWIMS);
                              McComas et al. (SWEPAM); Möbius et al. (SEPICA)
2000s ULEIS science          - Reames CME-driven shock studies, ³He systematics,
                              Fe/O spectral hardening, ⁵⁹Ni decay flag
2010s STEREO/LET, SIT        - ULEIS-style TOF×E instruments on STEREO-A/B
2020+ Solar Orbiter/SIS,     - Modern descendants: same design lineage from this paper
      Parker Solar Probe
      IS⊙IS / EPI-Lo
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Stone et al. 1998a (CRIS) | ACE 4-spectrometer suite의 자매 기기, GeV/nuc cosmic-ray composition | High — 같은 미션, 상보적 에너지 범위 / sister instrument, complementary energy band |
| Stone et al. 1998b (SIS) | SIS는 10–100 MeV/nuc; ULEIS와 SIS 사이 부드러운 매칭 | High — 직접 인접 기기 / immediate neighbor in energy |
| Gloeckler et al. 1998 (SWICS, SWIMS) | 태양풍/픽업 이온; <100 keV/nuc; ULEIS 저에너지 끝과 매칭 | High — 헬리오스피어 종합 조성 그림의 전반부 / lower-energy half of composition picture |
| Möbius et al. 1998 (SEPICA) | SEP 전하상태 측정; ULEIS 동위원소 + SEPICA 전하 = Q/M 결정 | High — 전하상태 보완 / charge-state complement |
| McComas et al. 1998 (SWEPAM) | 태양풍 플라즈마 컨텍스트; CIR 식별 도구 | Medium — 태양풍 환경 데이터 / solar-wind context |
| Mazur et al. 1992 | 28 April 1978 SEP 사건 스펙트럼 (논문 Fig. 3) | High — ULEIS 저에너지 영역의 과학 동기 / motivates low-E coverage |
| Mason et al. 1994 | ³He-rich 사건의 중원소 동위원소 이상 | High — ULEIS의 목표 과학 산출물 / target science output |
| Reames et al. 1997 (EPACT) | 임펄시브/점진적 SEP 스펙트럼 (논문 Fig. 4, 5) | High — ULEIS 사양의 직접 동기 / direct motivation for ULEIS specs |
| Anger 1966 | 위치 감지 양극 원래 개념 | Medium — WSA 가계도 / WSA ancestry |
| Lapington & Schwarz 1986 | WSA 설계 | High — Eqs. (2)–(3)의 기원 / origin of Eqs. (2)–(3) |
| Siegmund et al. 1986a, b | Z-stack MCP 타이밍 | High — MCP 어셈블리의 기술 baseline |
| Ipavich et al. 1978 | Au-Si 검출기 펄스 높이 결손 | Medium — SSD 응답 보정 baseline |
| von Rosenvinge et al. 1995 | EPACT/STEP / WIND 종합 / 효율 측정 | High — Table XI의 효율 추정 / efficiency estimates in Table XI |
| Anders & Ebihara 1982 | 태양계 원소/동위원소 조성 baseline | Medium — 비교 기준 / comparison standard |

---

## 7. References / 참고문헌

- Anders, E. and Ebihara, M., "Solar-System Abundances of the Elements", Geochim. Cosmochim. Acta 46, 2363, 1982.
- Anger, H. O., Trans. Instr. Soc. Am. 5, 311, 1966.
- Fisk, L. A., "³He-Rich Flares: a Possible Explanation", Astrophys. J. 224, 1048, 1978.
- Gloeckler, G., Bedini, P., Bochsler, P., et al., "Investigation of the Composition of Solar and Interstellar Matter Using Solar Wind and Pickup Ion Measurements with SWICS and SWIMS on the ACE Spacecraft", Space Sci. Rev. 86, 497, 1998.
- Ipavich, F. M., Lundgren, R. A., Lambird, B. A., and Gloeckler, G., "Measurement of Pulse-Height Defect for H, He, C, N, O, Ne, Ar, Kr from ~2 to ~400 keV/nucl", Nucl. Instr. Methods 154, 291, 1978.
- Lapington, J. S. and Schwarz, H. E., "The Design and Manufacture of Wedge and Strip Anodes", IEEE Trans. Nucl. Sci. NS-33, 288, 1986.
- Mason, G. M., Mazur, J. E., and Hamilton, D. C., "Heavy Ion Isotopic Anomalies in ³He-Rich Solar Particle Events", Astrophys. J. 425, 843, 1994.
- Mason, G. M., Mazur, J. E., Dwyer, J. R., Reames, D. V., and von Rosenvinge, T. T., "New Spectral and Abundance Features of Interplanetary Heavy Ions in Corotating Interaction Regions", Astrophys. J. 486, L149, 1997.
- Mason, G. M., Gold, R. E., Krimigis, S. M., Mazur, J. E., Andrews, G. B., Daley, K. A., Dwyer, J. R., Heuerman, K. F., James, T. L., Kennedy, M. J., Lefevere, T., Malcolm, H., Tossman, B., and Walpole, P. H., "The Ultra-Low-Energy Isotope Spectrometer (ULEIS) for the ACE Spacecraft", Space Sci. Rev. 86, 409–448, 1998. DOI: 10.1023/A:1005079930780.
- Mazur, J. E., Mason, G. M., Klecker, B., and McGuire, R. E., "The Energy Spectra of Solar Flare Hydrogen, Helium, Oxygen, and Iron: Evidence for Stochastic Acceleration", Astrophys. J. 401, 398, 1992.
- Mazur, J. E., Mason, G. M., Klecker, B., and McGuire, R. E., "The Abundances of Hydrogen, Helium, Oxygen, and Iron Accelerated in Large Solar Particle Events", Astrophys. J. 404, 810, 1993.
- Meckbach, R., "Secondary Electron Emission from Foils Traversed by Ion Beams", in Beam-Foil Spectroscopy, Plenum Press, p. 577, 1976.
- Mewaldt, R. A. and Stone, E. C., "Isotope Abundances of Solar Coronal Material", Astrophys. J. 337, 959, 1989.
- Mewaldt, R. A., Selesnick, R. S., Cummings, J. R., Stone, E. C., and von Rosenvinge, T. T., "Evidence for Multiply Charged Anomalous Cosmic Rays", Astrophys. J. 466, L43, 1996.
- Möbius, E. et al., "The Solar Energetic Particle Ionic Charge Analyzer (SEPICA) and the Data Processing Unit for SWICS, SWIMS, and SEPICA", Space Sci. Rev. 86, 449, 1998.
- Reames, D. V., Barbier, L. M., von Rosenvinge, T. T., Mason, G. M., Mazur, J. E., and Dwyer, J. R., "Energy Spectra of Ions Accelerated in Impulsive and Gradual Solar Events", Astrophys. J. 483, 515, 1997.
- Siegmund, O. H. W., Lampton, M., Bixler, J., Bowyer, S., and Malina, R. F., "Operation Characteristics of Wedge and Strip Readout Systems", IEEE Trans. Nucl. Sci. 33(1), 724, 1986a.
- Stone, E. C. et al., "Phase A Study of an Advanced Composition Explorer", California Institute of Technology, 1989.
- Stone, E. C. et al., "The Cosmic Ray Isotope Spectrometer for the Advanced Composition Explorer", Space Sci. Rev. 86, 285, 1998a.
- Stone, E. C. et al., "The Solar Isotope Spectrometer for the Advanced Composition Explorer", Space Sci. Rev. 86, 355, 1998b.
- von Rosenvinge, T. T. et al., "The Energetic Particles: Acceleration, Composition, and Transport (EPACT) Investigation on the Wind Spacecraft", Space Sci. Rev. 71, 155, 1995.
