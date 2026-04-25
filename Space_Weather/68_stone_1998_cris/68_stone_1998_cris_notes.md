---
title: "The Cosmic-Ray Isotope Spectrometer for the Advanced Composition Explorer"
authors: "E. C. Stone, C. M. S. Cohen, W. R. Cook, A. C. Cummings, B. Gauld, B. Kecman, R. A. Leske, R. A. Mewaldt, M. R. Thayer, B. L. Dougherty, R. L. Grumm, B. D. Milliken, R. G. Radocinski, M. E. Wiedenbeck, E. R. Christian, S. Shuman, H. Trexel, T. T. von Rosenvinge, W. R. Binns, D. J. Crary, P. Dowkontt, J. Epstein, P. L. Hink, J. Klarmann, M. Lijowski, M. A. Olevitch"
year: 1998
journal: "Space Science Reviews"
doi: "10.1023/A:1005075813033"
topic: Space_Weather
tags: [galactic-cosmic-rays, isotope-spectrometer, ACE, CRIS, SOFT-hodoscope, Si-Li-detector, dE-dx-technique, instrument-design]
status: completed
date_started: 2026-04-25
date_completed: 2026-04-25
---

# 68. The Cosmic-Ray Isotope Spectrometer for the Advanced Composition Explorer / ACE 우주선용 우주선 동위원소 분광기

---

## 1. Core Contribution / 핵심 기여

CRIS(Cosmic-Ray Isotope Spectrometer)는 1997년 8월 25일 발사된 ACE 위성에 탑재된 6개 고분해능 분광기 중 하나로, 가장 높은 에너지 영역(~50–500 MeV/nucleon)에서 갈락틱 우주선의 원소 및 동위원소 조성(2 ≤ Z ≤ 30)을 ≲0.25 amu의 질량 분해능으로 측정하기 위해 설계되었다. CRIS는 두 가지 검출기 시스템을 결합한다 — Scintillating Optical Fiber Trajectory(SOFT) 호도스코프(궤적 측정, < 100 µm r.m.s. 위치 분해능)와 4개의 Si(Li) 검출기 스택(에너지 손실 측정, 각 스택당 15개 검출기, 두께 3 mm × 직경 10 cm). 이 조합은 ~250 cm² sr의 기하학적 인자를 달성하여 이전 우주선 동위원소 분광기들보다 한 차원 더 큰 통계 능력을 제공하며, 2년의 태양 극소기 운용 동안 약 5×10⁶개의 정지 중원자핵(Z > 2)을 수집할 것으로 예상된다.

The Cosmic-Ray Isotope Spectrometer (CRIS), one of six high-resolution spectrometers on the ACE spacecraft (launched 25 August 1997), is designed to measure the elemental and isotopic composition of galactic cosmic rays (2 ≤ Z ≤ 30) over the highest-energy decade of ACE's interval, ~50–500 MeV/nucleon, with mass resolution ≲0.25 amu. CRIS combines two detector systems: a Scintillating Optical Fiber Trajectory (SOFT) hodoscope providing trajectory measurements with <100 µm r.m.s. position resolution and a 7.2-cm lever arm, plus four stacks of fifteen lithium-drifted silicon Si(Li) detectors (3 mm thick × 10 cm diameter) for multiple ΔE and total-energy measurements. This combination yields a geometrical factor of ~250 cm² sr — an order of magnitude larger than previous space-borne isotope spectrometers — and is expected to collect ~5×10⁶ stopping heavy nuclei (Z > 2) during two years of solar-minimum operation. Charge and mass identification follow the multiple-dE/dx-versus-residual-energy technique, with explicit error budgeting for Bohr/Landau fluctuations, multiple Coulomb scattering, mapping errors, energy-measurement errors, and trajectory uncertainties (Appendix A).

---

## 2. Reading Notes / 읽기 노트

### Part I: Scientific Objectives / 과학적 목표 (Section 2, pp. 286–296)

**갈락틱 우주선의 천체물리학적 의의 / Astrophysical role of GCRs.** 우주선의 에너지 밀도는 성간가스, 자기장, 별빛의 에너지 밀도와 비슷하며, 우리 은하의 중요한 구성요소이다. 그러나 광자와 달리 전하를 가지므로 자기장에 따라 나선 운동하여 도착 방향이 출처를 알려주지 않는다. 따라서 원소 및 동위원소 조성이 우주선의 기원과 역사에 대한 핵심 단서를 제공한다.

The energy density of cosmic rays is comparable to that of interstellar gas, magnetic fields, and starlight, making them an essential component of the Galaxy. Because charged particles spiral around galactic field lines, their arrival directions reveal nothing about their sources; instead, the elemental and isotopic composition encodes the history of cosmic-ray nucleosynthesis, fractionation, acceleration, and propagation.

**4가지 동위원소 그룹 / Four isotope groups (Figure 1, p. 290)**:

1. **Primary isotopes (1차 동위원소)**: 가속 시점부터 모재의 핵합성 정보를 보존하는 동위원소(예: ²⁰Ne, ²⁴Mg, ²⁸Si, ⁵⁶Fe). FIP(First Ionization Potential)에 의한 조성 분별이 관측되며 (Figure 2) 태양계 조성과 ~5배 차이. / Source-dominated isotopes preserving nucleosynthesis signatures; FIP fractionation observed (~factor 5 enhancement for low-FIP elements).
2. **Acceleration-delay clocks (가속 지연 시계)**: 전자포획(EC)으로 붕괴하는 1차 동위원소(⁵⁹Ni T₁/₂ ~ 10⁵ yr, ⁵⁷Co ~1 yr). 가속 전 저밀도 환경에서 시간이 흐르면 EC로 붕괴하나, 일단 가속되어 전자가 모두 분리되면 안정. ⁵⁹Ni 결손은 합성-가속 시간차 >10⁵ yr를 시사. / EC isotopes that decay only when slow; their depletion measures the elapsed time between nucleosynthesis and acceleration.
3. **Propagation clocks (전파 시계)**: β⁻로 붕괴하는 2차 동위원소(¹⁰Be 1.5 Myr, ²⁶Al 0.87 Myr, ³⁶Cl 0.30 Myr, ¹⁴C 5730 yr). Leaky-box 모델에서 ¹⁰Be로부터 갇힘시간 1–2×10⁷ yr, 평균 가스밀도 ~0.3 atoms cm⁻³ 도출. / β⁻-decay secondaries giving mean confinement time; ¹⁰Be yields τ_esc ≈ 1.5×10⁷ yr.
4. **Reacceleration clocks (재가속 시계)**: 오직 EC로만 붕괴하는 2차 동위원소(⁴⁹V, ⁵¹Cr, ⁵⁵Fe). 저에너지에서 전자 부착, 고에너지에서 박리되므로 에너지 변화 과정 추적. / Pure-EC secondaries probe re-acceleration during propagation.

**Table II (p. 289) — 19개 방사성 동위원소의 붕괴 모드와 반감기.** ⁷Be (53 days), ¹⁰Be (1.5 Myr), ¹⁴C (5730 yr), ²⁶Al (0.87 Myr) 등이 측정 가능한 시계 후보. ⁵⁴Mn은 EC, β⁺, β⁻ 세 가지 모드 동시 가능. / Table II lists 19 radioactive nuclides with Z ≤ 30, including the key clock isotopes ⁷Be, ¹⁰Be, ¹⁴C, ²⁶Al, ³⁶Cl, ⁴¹Ca, ⁵⁴Mn, ⁵⁹Ni.

**FIP 효과 / FIP effect (Figure 2, p. 291).** GCR 출처/태양계 조성 비율 대 첫 이온화 전위. FIP < 10 eV 원소(Na, Mg, Al, Si, Ca, Fe, Co, Ni)는 ~5배 풍부; FIP > 10 eV (H, He, N, O, Ne, Ar)는 약화. 이온이 광구→코로나로 더 효율적으로 운반되는 메커니즘과 일치. 최근 Meyer/Drury/Ellison(1998)은 휘발성(volatility)이 진정한 원인이라 제안 — 비휘발성(refractory) 원소가 grain으로 가속되고 휘발성은 개별 핵으로 가속됨.

**Section 2.3 — Confinement time / 갇힘 시간 (p. 293).** ¹⁰Be의 surviving fraction에서 1.5×10⁶ yr 반감기와 1–2×10⁷ yr 갇힘시간 도출. 이로부터 평균 가스밀도 ~0.3 atoms cm⁻³ — 은하 디스크 평균보다 작음 → 우주선이 디스크 외부 헤일로에서도 상당시간 보냄을 의미. CRIS는 2년 동안 ¹⁰Be 3000개, ²⁶Al 3000개, ³⁶Cl 1000개를 관측할 것으로 예상.

### Part II: Design Requirements / 설계 요구사항 (Section 3, pp. 296–299)

**ΔE-E' 기법의 기본 원리 / The fundamental ΔE-E' technique.** 전하 Z, 질량 M, 운동에너지 E의 입자가 두께 L을 통과한 후 잔여 에너지 E'으로 빠져나오면, 사정거리(range)의 변화가 통과한 두께와 같다.

$$\mathcal{R}_{Z,M}(E/M) - \mathcal{R}_{Z,M}(E'/M) = L \tag{Eq. 1}$$

ℛ_{Z,M}은 Hubert et al. (1990) 표를 통해 알려진 함수이므로, ΔE = E − E', E', L 측정에서 Z와 M을 동시 결정 가능.

**파워-로 근사 / Power-law approximation.** ℛ_{Z,M}(E/M) ≃ k(M/Z²)(E/M)^a, a ≃ 1.7 로 근사하면:

$$M \simeq \left(\frac{k}{Z^2 L}\right)^{1/(a-1)} \left(E^a - E'^a\right)^{1/(a-1)} \tag{Eq. 2}$$

$$Z \simeq \left(\frac{k}{L(2+\epsilon)^{(a-1)}}\right)^{1/(a+1)} \left(E^a - E'^a\right)^{1/(a+1)} \tag{Eq. 3}$$

여기서 M/Z ≈ 2+ε. ΔE-E' 도면 위에서 인접한 원소들 간 트랙 간격은 인접 동위원소 간 간격보다 (2+ε)(a+1)/(a−1) ≃ 8배 더 크다 → Z를 먼저 명료하게 분리, 그 후 동위원소 분리 가능.

**오차 전파 / Error propagation.** σ_M ∝ L^{1/(a-1)} 이므로 (σ_M/M)_L = (σ_L/L)/(a−1). Fe의 경우 σ_M < 0.1 amu 달성을 위해서는 두께 L을 ~0.12% 정확도로 결정해야 함 — 이것이 SOFT 호도스코프의 100 µm r.m.s. 분해능 요구의 출처.

**수직선이 아닌 사선 입사 / Oblique incidence.** L = L₀ sec θ, dead layer 두께 l = l₀ sec θ. CRIS는 ±70° 시야 내에서 입사각도(zenith angle)를 0.1° 정확도로 결정.

**Bohr/Landau 변동 + 다중 Coulomb 산란 / Bohr/Landau and multiple scattering.** 검출기에서 입자가 슬로다운 할 때 발생하는 본질적 통계적 변동이 질량 분해능의 궁극적 한계를 정한다. CRIS는 이러한 fundamental contributions가 dominant하도록 설계됨(즉, 측정 오차가 본질 한계 이하).

**Table III (p. 300) — 설계 요구 사양**:

| 항목 / Parameter | 요구값 / Requirement |
|---|---|
| 천정각 분해능 / Zenith resolution | < 0.1° |
| 위치 분해능(상대) / Relative position res | < 0.13 mm r.m.s. |
| 검출 효율 (Be) | > 50% |
| 검출 효율 (Z ≥ 8) | > 90% |
| Si 검출기 두께 / Si thickness | 3.0 ± 0.1 mm |
| Si 두께 변화 / Thickness variation | < 60 µm |
| Dead layer | ≲ 50 µm |
| Depletion voltage | < 250 V |
| PHA dynamic range | > 700:1 |
| PHA quantization | 12 bits |
| PHA nonlinearity | < 0.05% |

### Part III: Sensor System / 센서 시스템 (Section 4, pp. 299–313)

**4.1 SOFT 호도스코프 / SOFT hodoscope.**
- **구조 / Structure**: 3개의 xy 플레인(H1, H2, H3) + 1개의 trigger 플레인(T). 각 플레인은 직교하는 두 층으로 구성 (총 6개 호도스코프 섬유 층 + 2개 트리거 층).
- **섬유 / Fibers**: 폴리스티렌 코어 + BPBD/DPOPOP scintillation dyes (방출 피크 430 nm), 200 µm 정사각형 단면(10 µm acrylic cladding 포함), Washington University 제작. 호도스코프 섬유는 EMA(extramural absorber, 검은 잉크)로 코팅하여 광학적 결합 방지; 트리거 섬유는 EMA 없이 최대 감도.
- **활성 영역 / Active area**: 26 cm × 26 cm.
- **플레인 간격 / Plane spacing**: H1–H2 = 3.9 cm, H2–H3 = 3.3 cm (서로 다른 간격은 photoelectron 'hopping' 모호성 해결용).
- **이미지 인텐시파이어 / Image intensifiers**: Photek Model MCP-340S, 40 mm 직경, dual MCP, gateable; S-20 photocathode; ~2×10⁶ photon gain. 두 카메라(A, B) 중복.
- **CCD readout**: Thompson TH-7866, 244×550 픽셀, 16 µm × 27 µm 픽셀, fiber-optic reducer 통해 결합. 한 섬유 → 4×2.4 픽셀 영역.

**4.2 Si(Li) 검출기 스택 / Si(Li) detector stack (Figure 11, p. 308).**
- 4개 telescope (A, B, C, D), 각 telescope에 15개 Si(Li) 검출기 (E1, E2, E3-1, E3-2, ..., E8-1, E8-2, E9).
- E1, E9 — single-groove (활성 영역 68 cm²); E2–E8 — double-groove with active guard ring (중심 영역 57 cm² + 외부 가드 링).
- 두 개씩 짝지어진 검출기(예: E3-1+E3-2)는 단일 PHA로 합산 — 평행 PHA 입력으로 전자장치 단순화 (총 32 PHA).
- 가드 링은 telescope 측면을 통과하는 입자를 anti-coincidence로 제거.
- LBNL 제조: p-type silicon → Li 진공 증착 → drifting → groove 식각 → Au/Al 금속화. ~2990 µm 평균 두께, ±55 µm 변동.

**4.2.3 두께 매핑 / Thickness mapping.** MSU/NSCL cyclotron에서 100 MeV/nuc ³⁶Ar 빔을 1 cm 빔스폿으로 검출기를 스캔, 잔여 에너지 E'에서 두께 변화 추출. 모든 비행 검출기에 대한 두께 맵 제작 (Figure 12). 두께 분해능 < 1%, 절대 정확도 ~0.1%.

**4.2.4 PHA 펄스-하이트 분석 / PHA analysis.** 60개 중심 + 44개 가드 신호를 32 PHA로 압축. Table VI는 PHA-to-detector 매핑.

### Part IV: Electronics and Onboard Processing / 전자장치 및 탑재 처리 (Section 5, pp. 313–321)

**5.1 RTX2010 마이크로프로세서 / Logic and microprocessor.** Forth로 프로그램된 RTX2010이 시스템 코어. 모든 제어 + 온보드 데이터 압축 + SOFT 픽셀 클러스터 추출을 담당. < 1 W 소비, 단일 PCB ~450 cm². 별도의 'event processor' state machine이 시간임계적 코인시던스를 처리.

- 1단계 phase (4 µs): 스택 검출기 hybrid의 low-level discriminator 결과 latch.
- 2단계 phase (12 µs): 더 높은 레벨 discriminator + SOFT trigger plane discriminator 결과 latch.
- 결과에 따라 처리 진행 또는 중단.

**Z 분류 / Charge classification by discriminator levels**: Low/Medium/High threshold가 stopping Z ≥ 1, Z ≥ 2, Z > 2를 트리거. Z=1, Z=2 이벤트는 마이크로프로세서가 throttle하여 라이브타임을 무거운 핵에 양보.

**5.2 PHA Hybrid (custom VLSI).** Cook et al. (1993) custom bipolar ASIC at Harris Semiconductor. preamp + post amp + AOG + peak detector + Wilkinson ADC, 12-bit 변환, 256 µs max 변환시간 (16 MHz clock), 2 µs shaping (3 µs time-to-peak). 동적 범위 2000:1, 게인 안정성 20 ppm/°C, 비선형성 < 0.01% full scale. Power 40 mW.

**5.4 데이터 압축 / Data compression.** 464 bits/s 텔레메트리에 맞추기 위해 SOFT pixel data를 'cluster' (centroid + intensity, 절반-픽셀 해상도)로 압축. 보통 6개 fiber layer × 2 cluster + 6 추가 cluster = 최대 18 cluster/event. Variable-length 이벤트 형식 (31–162 bytes). 일반 이벤트 길이 ~52 bytes.

**5.4.2 Priority system.** 이벤트를 61개 buffer로 sort: range, pulse-height, trajectory quality, single/multiple telescope, hazard 조건 기준. 우선순위 큰 buffer (긴 range, 높은 Z, good trajectory)부터 readout. 1 event/s만 텔레메트리 가능 → 정지하는 무거운 핵 0.1–0.2 events/s 정도 예상.

### Part V: Mechanical and Resource / 기계 설계 및 자원 (Sections 6–7, pp. 321–326)

- 두 박스 구조: 상자 1 (위) — SOFT, 53.3 × 43.8 × 10.1 cm; 상자 2 (아래) — Si(Li) 스택 + 전자장치, 53.3 × 40.6 × 13.4 cm.
- 총 질량 29.2 kg, 전력 ~12 W (max ~16 W 고온 시), bit rate 464 bits/s.
- 1차 FOV 45° half-angle, full FOV 70° half-angle.
- 기계 정렬 0.1° 면-대-면, 1° 검출기-대-검출기.

### Part VI: Expected Performance / 기대 성능 (Section 8, pp. 326–330)

**8.1 에너지 범위 / Energy coverage (Figure 18).** 동위원소 식별: O 60–280, Si 85–400, Fe 115–570 MeV/nuc. 'Elements only' (관통하는 입자) 영역에서는 Bragg 곡선 패턴으로 Z+E 도출 가능, 질량 미결정.

**8.2 기하학적 인자 / Geometrical factor (Figure 19).** Monte Carlo 계산 결과 16O는 ~350 cm² sr peak, 56Fe ~250 cm² sr peak. SOFT가 큰 활성 영역을 제공하여 60° 이상의 큰 입사각도까지 응답.

**기대 이벤트 수 (2년, solar minimum, θ < 45°) / Expected yields (Figure 20)**:
- ⁷Be: 1.5×10⁴
- ¹⁰Be: 3000
- ¹⁵N: ~5×10⁴
- ¹⁶O: 8.9×10⁵
- ²²Ne: ~10⁴
- ²⁶Al: 3000
- ²⁸Si: 1.7×10⁵
- ³⁶Cl: 1000
- ⁵⁴Mn: ~10²
- ⁵⁶Fe: 1.4×10⁵
- ⁵⁹Ni: ~10²
- 총 Z>2: ~5×10⁶

**8.3 질량 분해능 (Figure 21)**: ¹⁶O σ_M ≈ 0.10–0.12 amu, ²⁸Si ≈ 0.15 amu, ⁵⁶Fe ≈ 0.20–0.25 amu (θ = 20° 가정). Bohr/Landau 변동 + 다중 산란이 dominant.

### Part VII: Accelerator Calibrations / 가속기 보정 (Section 9, pp. 330–337)

**9.1 GSI 보정 / GSI calibrations.** 500 MeV/nuc ⁵⁶Fe 빔을 polyethylene target에 분쇄, 다양한 fragment 동위원소로 CRIS를 보정. Figure 22의 ΔE-E' 분산도 — Fe, Mn, Cr, V, Ti, Sc, Ca까지 명확히 분리. Figure 23 — Iron 질량 히스토그램, Gaussian fit 결과 σ_M = 0.205 amu (E4 stopping).

**9.2 SOFT 좌표 보정 / SOFT coordinate calibration.** MSU/NSCL oxygen 빔을 SOFT 전체 영역에 산포, fiber centroid가 individual fiber 위치에 cluster (Figure 24). 그 후 1.27 cm × 1.27 cm 그리드의 lead mask를 통해 absolute spatial calibration (Figure 25).

**9.3 위치/각도 분해능 / Position and angular resolution (Figure 26, 27).** 155 MeV/nuc oxygen으로 측정, σ_Y2 = 83 µm, σ_ΔY2 = 102 µm — 단일좌표 분해능 ~83 µm, 천정각 분해능 ≲ 0.1°. Fe → ~60 µm, Be → ~100 µm로 dE/dx에 따라 향상.

**9.4 검출 효율 / Detection efficiency (Figure 28).** Camera B가 더 높은 효율 (default), Be까지 측정 가능. trigger efficiency Be 이상에서 ~100%.

### Part VIII: Mass Resolution Error Budget (Appendix A, pp. 339–348)

CRIS의 핵심 분석 — 질량 분해능에 기여하는 모든 오차 항목을 quadrature로 합산. 두 검출기 (ΔE, E') + dead layer (두께 l) 경우 (Figure 29):

$$\mathcal{R}_{Z,M_0}\!\left(\frac{\Delta E + \delta E + E'}{M}\right) - \mathcal{R}_{Z,M_0}\!\left(\frac{E'}{M}\right) = \frac{M_0}{M}(L+l) \tag{Eq. 4}$$

$$\mathcal{R}_{Z,M_0}\!\left(\frac{\delta E + E'}{M}\right) - \mathcal{R}_{Z,M_0}\!\left(\frac{E'}{M}\right) = \frac{M_0}{M}\, l \tag{Eq. 5}$$

dead layer의 알 수 없는 에너지 손실 δE를 소거 →

$$\frac{\Delta E}{M} = \mathcal{E}_{Z,M_0}\!\left(\mathcal{R}_{Z,M_0}\!\left(\frac{E'}{M}\right) + \frac{M_0}{M}(L+l)\right) - \mathcal{E}_{Z,M_0}\!\left(\mathcal{R}_{Z,M_0}\!\left(\frac{E'}{M}\right) + \frac{M_0}{M}l\right) \tag{Eq. 6}$$

이로부터 측정량 ΔE, E', L, l 변화에 대한 ∂M 편미분 (Table VIII, p. 341).

**A.1 Bohr/Landau fluctuations**: σ_{ΔE,Landau} 적분 (Eq. 8). 두 가지 기여 (ΔE 검출기 내 + dead layer 내).

**A.2 Multiple Coulomb scattering** (Eq. 11):

$$\frac{d\sigma_{\delta\theta}^2}{dx} \simeq \left(\frac{Z}{M}\frac{0.0146}{\beta^2 \gamma}\right)^2 \frac{1}{X_0}$$

silicon X₀ = 21.82 g cm⁻². Path-length error σ_L (Eq. 12):

$$\sigma_{L,\text{scat}} = L \tan\theta \sqrt{M} \times \sqrt{\int_{E'/M}^{E/M}\left(\frac{\mathcal{R}(E/M) - \mathcal{R}(\epsilon)}{L}\right)^2\!\left(\frac{d\sigma_{\delta\theta}^2}{dx}\right)_\epsilon \frac{d\epsilon}{\delta_Z(\epsilon)}}$$

**A.3 Trajectory measurement errors**: n개 측정에서 slope, intercept 오차. 단순화 — 모든 σ_x_i 동일하면 σ_{dx/dz} = √2 σ_x / Δz, σ_{x_0(z)} = (σ_x/√n) × (geometry factor).

$$\frac{\sigma_{\sec\theta}}{\sec\theta} = \frac{1}{\sqrt{2}} \sin 2\theta\, \frac{\sigma_x}{\Delta z}$$

**A.4 Mapping errors**: 두께 맵 σ_L(map) (3개 무관 항: L₀, x, y).
**A.5 Energy measurement errors**: PHA 노이즈 + 양자화 (Eq. 25):

$$\sigma_{E_k}^2 = \sigma_{E_k(\text{noise})}^2 + \frac{w_k^2}{12}$$

**A.6 Additional effects**: charge-state fluctuations (Figure 30) — Fe 20 MeV/nuc에서 q_eff = +24.9 (vs Z = 26).

### Part IX: Conclusion / 결론 (Section 10, p. 338)

CRIS는 1997년 8월 25일 발사 후 이틀 만에 turn-on. 제출 시점(1998년) 약 두 달의 데이터 분석 결과 모든 설계 목표 달성 확인. 이후 수년간 데이터 수집을 통해 갈락틱 우주선 천체물리학에 중요한 진전 기대.

CRIS turned on 27 August 1997, two days after launch. By submission, two months of in-flight data confirmed achievement of essentially all design goals. Continued operation over years was expected to produce data sets of unprecedented mass resolution and statistical accuracy.

---

## 3. Key Takeaways / 핵심 시사점

1. **ΔE-E' 기법이 1차 측정 원리이다 / The ΔE-E' technique is the primary measurement principle** — 식 (1)이 모든 분석의 출발점. Si(Li) 스택에서 침투 두께 L = L₀ sec θ가 사정거리 차와 같다는 점이 Z, M을 동시에 결정 가능하게 한다. 이는 1970년대부터 Stone, Althouse, Cook 등이 발전시킨 검출 기법의 정점이며, ISEE-3 (Althouse 1978), MAST/SAMPEX (Cook 1993)으로 이어진 헤리티지를 계승한다. / Equation 1 is the foundation; everything else (including the entire instrument architecture) is engineering to satisfy it. This is the culmination of detection techniques developed by Stone, Althouse, Cook and others since the 1970s.

2. **SOFT 호도스코프는 큰 기하학적 인자를 가능케 한다 / SOFT enables the large geometrical factor** — 4 stack × 단일 호도스코프 구조는 ~250 cm² sr 달성. 이전 기기들(MAST, HEIST)은 multi-wire proportional counter나 두꺼운 산화물 검출기로 같은 기능을 했지만 작은 면적과 낮은 효율 때문에 통계가 부족했다. SOFT는 26 × 26 cm 활성 면적과 < 100 µm 위치 분해능을 동시 달성. / The SOFT design — unique among 1990s cosmic-ray spectrometers — is what makes the CRIS geometrical factor (~250 cm² sr) feasible. The active area of 26×26 cm and position resolution <100 µm are simultaneously achieved, providing trajectory measurements for multiple stacks via a single detector system.

3. **에러 버짓이 모든 설계를 지배한다 / Error budgeting drives every design choice** — Appendix A의 σ_M^2 = Σσ_i² 합산이 검출기 두께 정확도(0.12%), 위치 분해능(100 µm), PHA dynamic range(700:1) 같은 사양을 직접 도출한다. 이러한 'top-down' 사양 도출은 우주 분광기 설계의 모범. / Appendix A turns σ_M ≲ 0.25 amu into Si thickness 0.12% accuracy, 100 µm position resolution, 700:1 dynamic range, and so on. This top-down specification flow is a textbook example for space instrument design.

4. **Bohr/Landau 변동과 다중 산란이 본질 한계 / Bohr/Landau fluctuations and multiple scattering set the fundamental limit** — 식 (7)과 (11)이 분해능의 궁극적 한계. CRIS는 다른 측정 오차 (위치, 에너지, 매핑)를 이 본질 한계 이하로 줄이는 방향으로 설계되었다. / Equations 7 and 11 set the limits; CRIS is designed so that all measurement errors stay below these intrinsic statistical limits — the design "succeeds" only if you can't tell the engineering errors from physics.

5. **여러 ΔE 측정이 redundancy와 background rejection을 제공 / Multiple ΔE measurements provide redundancy and background rejection** — E2~E8까지 8개 detector position에서 stopping이 가능하며 각각이 독립적인 (ΔE, E') 측정을 제공한다. 일치 검사로 fragmentation events, hazard events를 제거. / Eight stop positions (E2 through E8) provide independent mass determinations; consistency checks reject fragmentation events and edge effects.

6. **온보드 우선순위 시스템이 telemetry bottleneck을 해결 / Onboard prioritization solves the telemetry bottleneck** — 464 bits/s, 즉 ~1 event/s만 송신 가능하나 trigger rate은 수 events/s. 61개 buffer로 분류 후 priority 기반 readout이 희귀한 무거운 핵을 우선. 이는 모든 우주선 분광기의 전형 문제이며 CRIS의 솔루션은 명확한 청사진. / At 464 bits/s, only ~1 event/s can be telemetered, but trigger rates can exceed several/s. Sorting events into 61 buffers by Z, range, and trajectory quality, then reading high-priority buffers first, ensures rare heavy events are preserved.

7. **대형 Si(Li) 검출기 기술의 임무화 / Maturation of large-area Si(Li) detector technology** — 10 cm 직경, 3 mm 두께의 lithium-drifted silicon은 LBNL과 협업으로 새로 개발 (Allbritton et al. 1996). Fully depleted, 60 cm² 중심 + active guard ring을 갖춘 이 검출기 기술은 향후 우주선 분광기의 표준이 됨. / The 10-cm diameter, 3-mm thick Si(Li) wafer technology developed with LBNL (Allbritton et al. 1996) — fully depleted, with 60 cm² central area plus active guard rings — became the standard for subsequent cosmic-ray spectrometers.

8. **모듈식 광학 readout 아키텍처 / Modular optical readout architecture** — SOFT의 6 layer × 2 carriage × 1 image intensifier × 1 CCD 구조는 완전 redundant 카메라 시스템을 단일 호도스코프 검출기와 결합. 이미지 인텐시파이어가 photon → electron → MCP gain → phosphor → CCD pixel의 chain을 통해 ~2×10⁶ gain 제공. / The image-intensified CCD readout (S-20 photocathode → MCP gain → P-20 phosphor → fiber-optic reducer → CCD) provides 2×10⁶ photon gain, with full redundancy via two independent cameras.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Range-Energy Relation Power Law / 사정거리-에너지 멱법칙

$$\mathcal{R}_{Z,M}(E/M) \simeq k\, \frac{M}{Z^2}\, \left(\frac{E}{M}\right)^a, \quad a \simeq 1.7$$

- ℛ : range in g cm⁻², k: 물질에 따른 상수, a ≈ 1.7 in CRIS energy range / Range in g cm⁻²; k depends on the absorber; a ≈ 1.7 in the CRIS energy range.

### 4.2 Mass Determination / 질량 결정

$$\boxed{M \simeq \left(\frac{k}{Z^2 L}\right)^{1/(a-1)} \left(E^a - E'^a\right)^{1/(a-1)}}$$

- L = L₀ sec θ : 사선 입사 시 경로 길이 / Slanted path length
- (E^a − E'^a)는 ΔE-E' 측정에서 직접 계산 / Computed from ΔE-E' measurements
- a = 1.7이면 1/(a−1) = 1/0.7 ≈ 1.43 → σ_M/M = (σ_L/L)·1.43 (즉 두께 0.1% 오차 → 질량 0.14% 오차) / At a = 1.7, σ_M/M = 1.43 σ_L/L, so 0.1% thickness error → 0.14% mass error.

### 4.3 Charge Determination / 전하 결정

$$Z \simeq \left(\frac{k}{L(2+\epsilon)^{(a-1)}}\right)^{1/(a+1)} \left(E^a - E'^a\right)^{1/(a+1)}$$

- 인접 원소 트랙 분리 / 인접 동위원소 분리 = (2+ε)(a+1)/(a−1) ≃ 8 / Ratio of element-to-element separation versus isotope-to-isotope separation in the ΔE-E' plane is ≈ 8.

### 4.4 Bohr/Landau Variance / 보어/란다우 분산

$$\left(\frac{d\sigma_{\Delta E}^2}{dx}\right)_{\text{Landau}} = Z^2 \frac{(0.396 \text{ MeV})^2}{\text{g cm}^{-2}} \frac{Z_m}{A_m} \frac{\gamma^2 + 1}{2}$$

- Z: incident particle charge, Z_m, A_m: medium properties (Si: Z_m=14, A_m=28)
- γ: Lorentz factor, large γ → larger fluctuations (relativistic rise) / γ: Lorentz factor; high energy increases fluctuations

For a thick ΔE detector (energy varies along path):
$$\sigma_{\Delta E,\text{Landau}} = \sqrt{M \int_{e/M}^{E/M} \frac{\delta_Z^2(\epsilon')}{\delta_Z^3(\epsilon)} \left(\frac{d\sigma_{\Delta E}^2}{dx}\right)_{\text{Landau}} d\epsilon}$$

### 4.5 Multiple Coulomb Scattering / 다중 쿨롱 산란

$$\frac{d\sigma_{\delta\theta}^2}{dx} \simeq \left(\frac{Z}{M}\frac{0.0146}{\beta^2\gamma}\right)^2 \frac{1}{X_0}$$

- 0.0146 GeV: Highland 공식의 상수 / Highland formula constant
- X₀: radiation length in the medium (silicon X₀ = 21.82 g cm⁻²)

Path-length error from scattering (Eq. 12):
$$\sigma_{L,\text{scat}} = L \tan\theta\, \sqrt{M} \, \sqrt{\int_{E'/M}^{E/M} \left(\frac{\mathcal{R}(E/M) - \mathcal{R}(\epsilon)}{L}\right)^2 \left(\frac{d\sigma_{\delta\theta}^2}{dx}\right)_\epsilon \frac{d\epsilon}{\delta_Z(\epsilon)}}$$

### 4.6 Trajectory Reconstruction / 궤적 재구성

n개 좌표 측정에서:
$$\sigma_{dx/dz} = \left[\sum_{i=1}^n \frac{(z_i - \bar{Z}_x)^2}{\sigma_{x_i}^2}\right]^{-1/2}$$

CRIS-like geometry (3 planes, σ_x equal):
$$\sigma_{dx/dz} = \frac{\sqrt{2}\, \sigma_x}{\Delta z}$$

For Δz = 7.2 cm, σ_x = 100 µm: σ_{dx/dz} ≈ 2×10⁻³ rad ≈ 0.11° → meets requirement.

### 4.7 Geometric Factor / 기하학적 인자

For a parallel plate stack with active area A and stack height h, half-angle θ_max = arctan(D/h):
$$A\Omega = 2\pi A \int_0^{\theta_\text{max}} \sin\theta\, \cos\theta\, d\theta = \pi A \sin^2\theta_\text{max}$$

CRIS: A_stack ≈ 57 cm² (×4 = 228 cm²) × π × sin²(45°) ≈ 358 cm² sr theoretical, with actual MC value ~250 cm² sr after Si stack thickness limits.

### 4.8 Total Mass Resolution / 전체 질량 분해능

$$\sigma_M^2 = \underbrace{\sigma_{M,\text{Landau in }L}^2 + \sigma_{M,\text{Landau in }l}^2}_{\text{intrinsic}} + \underbrace{\sigma_{M,\text{scat}}^2}_{\text{intrinsic}} + \underbrace{\sigma_{M(\text{map})}^2}_{\text{calibration}} + \underbrace{\sigma_{M(\Delta E)}^2 + \sigma_{M(E')}^2}_{\text{electronics}} + \underbrace{\sigma_{M,\sec\theta}^2}_{\text{trajectory}}$$

CRIS의 ⁵⁶Fe at θ = 20° → σ_M ≈ 0.25 amu, dominated by intrinsic (Landau + scattering) terms.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1952  Rossi  ──────── 'High-Energy Particles' (energy loss theory)
1968  Halpern & Marshall ──── space PHA hybrids
1969  Payne  ─────── energy straggling in thick absorbers
1976  Fisher et al. ── Z=5–26 isotopes (early measurements)
1977  Garcia-Muñoz ─── ¹⁰Be confinement time
1978  Soutoul et al. ── EC clocks (acceleration delay concept)
1978  Althouse et al. ── ISEE-3 cosmic-ray isotope spectrometer
1979  Stone & Wiedenbeck ─ secondary tracer GCRS abundances
1980  Wiedenbeck & Greiner ── ¹⁰Be cosmic-ray age
1985  Meyer / Breneman & Stone ── FIP fractionation
1989  Davis et al. ─── Scintillating Optical Fiber Trajectory detectors
1990  Hubert et al. ── range/stopping-power tables 2.5–500 MeV/nuc
1990  Engelmann et al. ── HEAO-3-C2 elemental composition
1992  Crary et al. ── Bevalac SOFT calibration
1993  Cook et al.  ── custom analog VLSI for ACE
1993  Cook et al. ── MAST/SAMPEX precursor
1995  Connell & Simpson ── Ulysses Fe, Ni isotopes
1996  Allbritton et al. ── large-diameter Si(Li) for ACE
1996  Hubert et al. ── ACE-CRIS SOFT NSCL calibration
1997  ★ ACE launch (25 August), CRIS turn-on (27 August)
1998  ★ THIS PAPER ★ — Stone et al., CRIS instrument paper
1998  Stone et al. ── SIS instrument paper (companion)
1999  Yanasak et al. ── ²⁶Al/²⁷Al confinement time
2003  Wiedenbeck et al. ── ⁵⁹Co excess (acceleration delay confirmed)
2008  Binns et al. ── ²²Ne/²⁰Ne Wolf-Rayet origin
2016  Binns et al. ── first GCR ⁶⁰Fe detection
2025+ ── IMAP / IS⊙IS deployment (CRIS-style architecture continues)
2026  ★ CRIS still operating (~28 years, 5×10⁸+ events archived)
```

이 논문은 1990년대 후반 우주선 분광기 공학의 정점에 위치하며, 이후 ACE/CRIS 데이터를 활용한 모든 갈락틱 우주선 동위원소 연구의 instrument-level reference로 인용된다.

This paper sits at the apex of late-1990s cosmic-ray spectrometer engineering and is the canonical instrument-level reference cited by all subsequent ACE/CRIS galactic-cosmic-ray isotope studies.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Stone et al. (1998), "The Solar Isotope Spectrometer for ACE" (companion paper) | ACE 자매기기 (낮은 에너지 영역, 0.1–100 MeV/nuc) / Sister instrument on ACE for the lower energy decade | Same architecture (ΔE-E' + trajectory hodoscope), different energy band — together they cover 0.1 to 500 MeV/nuc / 같은 아키텍처 다른 에너지 밴드 |
| Althouse et al. (1978), ISEE-3 cosmic ray isotope spectrometer | CRIS 직계 조상 / Direct ancestor | First space mission to measure GCR isotopes systematically; CRIS extends with 10× better statistics / 첫 우주 GCR 동위원소 임무, CRIS는 통계 10배 개선 |
| Cook et al. (1993), MAST mass spectrometer | SAMPEX 동시기 분광기 / Contemporary spectrometer on SAMPEX | Established the multi-detector silicon stack approach refined in CRIS / CRIS에서 정제된 다층 Si 스택 방식의 기원 |
| Hubert et al. (1990), Range and Stopping Power Tables | Range-energy ℛ_{Z,M}(ε) 함수 / The function ℛ_{Z,M} | Tabulated R-E function used directly in Eq. 1 — without it CRIS analysis is impossible / 식 (1)에 직접 입력되는 표 |
| Davis et al. (1989), Scintillating Optical Fiber Trajectory Detectors | SOFT 기술의 기반 / Foundation of SOFT technology | The original SOFT proof-of-concept; CRIS is the first space-flight implementation / 우주 비행 첫 SOFT 구현 |
| Allbritton et al. (1996), Large-Diameter Si(Li) Detectors for ACE | CRIS Si(Li) 검출기의 제작 / Fabrication of CRIS Si(Li) detectors | LBNL technology paper describing how the 10 cm × 3 mm Si(Li) wafers were made / Si(Li) 제작 기술 |
| Engelmann et al. (1990), HEAO-3-C2 | 비교 데이터 / Comparison dataset | Primary source of pre-CRIS GCR elemental composition; CRIS validates and extends to isotopes / CRIS가 검증 및 동위원소로 확장 |
| Soutoul et al. (1978), Time Delay Between Nucleosynthesis and Acceleration | EC 시계 컨셉 / EC clock concept | The theoretical motivation for measuring ⁵⁹Ni — exactly what CRIS targets / ⁵⁹Ni 측정의 이론적 동기 |
| Connell & Simpson (1995, 1997), Ulysses HET | 직접 비교/검증 / Direct comparison and validation | Earlier statistical-limit isotope measurements; CRIS aims to surpass / CRIS가 능가하는 통계 한계 |
| Wiedenbeck & Greiner (1980), Cosmic-Ray Age | ¹⁰Be 연구 / ¹⁰Be confinement-time work | Established the ¹⁰Be propagation clock CRIS measures with 100x better statistics / CRIS가 100배 향상시킨 ¹⁰Be 시계 |

---

## 7. References / 참고문헌

- Stone, E. C. et al., "The Cosmic-Ray Isotope Spectrometer for the Advanced Composition Explorer", Space Science Reviews 86, 285-356, 1998. DOI: 10.1023/A:1005075813033
- Stone, E. C. et al., "The Solar Isotope Spectrometer for the Advanced Composition Explorer", Space Science Reviews 86, 357, 1998.
- Althouse, W. E. et al., "A Cosmic Ray Isotope Spectrometer", IEEE Trans. Geosci. Electronics GE-16, 204-207, 1978.
- Cook, W. R. et al., "MAST: A Mass Spectrometer Telescope for Studies of the Isotopic Composition of Solar, Anomalous, and Galactic Cosmic Ray Nuclei", IEEE Trans. Geosci. Remote Sensing 31, 557-564, 1993.
- Hubert, F., Bimbot, R., and Gauvin, H., "Range and Stopping-Power Tables for 2.5–500 MeV/Nucleon Heavy Ions In Solids", Atom. Dat. Nucl. Dat. Tables 46, 1-213, 1990.
- Davis, A. J. et al., "Scintillating Optical Fiber Trajectory Detectors", Nucl. Inst. Meth. A276, 347-358, 1989.
- Allbritton, G. A. et al., "Large-Diameter Lithium Compensated Silicon Detectors for the NASA Advanced Composition Explorer (ACE) Mission", IEEE Trans. Nucl. Sci. 43, 1505-1509, 1996.
- Engelmann, J. J. et al., "Charge Composition and Energy Spectra of Cosmic-Ray Nuclei for Elements from Be to Ni. Results from HEAO-3-C2", Astron. Astrophys. 233, 96-111, 1990.
- Soutoul, A., Cassé, M., and Juliusson, E., "Time Delay Between the Nucleosynthesis of Cosmic Rays and Their Acceleration to Relativistic Energies", Astrophys. J. 219, 753-755, 1978.
- Stone, E. C. and Wiedenbeck, M. E., "A Secondary Tracer Approach to the Derivation of Galactic Cosmic Ray Source Isotopic Abundances", Astrophys. J. 231, 606-623, 1979.
- Connell, J. J. and Simpson, J. A., "The Ulysses Cosmic Ray Isotope Experiment: Isotopic Abundances of Fe and Ni from High Resolution Measurements", Proc. 24th Int. Cosmic Ray Conf., Rome 2, 602-605, 1995.
- Rossi, B., "High-Energy Particles", Prentice-Hall, 1952.
- Payne, M. G., "Energy Straggling of Heavy Charged Particles in Thick Absorbers", Phys. Rev. 185, 611-623, 1969.
- Wiedenbeck, M. E. and Greiner, D. E., "A Cosmic-Ray Age Based on the Abundance of ¹⁰Be", Astrophys. J. 239, L139-L142, 1980.
- Crary, D. J. et al., "A Bevalac Calibration of a Scintillating Optical Fiber Hodoscope", Nucl. Inst. Meth. A316, 311-317, 1992.
- Cook, W. R. et al., "Custom Analog VLSI for the Advanced Composition Explorer", Small Instruments Workshop Proc., JPL, 1993.
- Hink, P. L. et al., "The ACE-CRIS Scintillating Optical Fiber Trajectory (SOFT) Detector: A Calibration at the NSCL", Proc. SPIE 2806, 199-208, 1996.
- Dougherty, B. L. et al., "Characterization of Large-Area Silicon Ionization Detectors for the ACE Mission", Proc. SPIE 2806, 188-198, 1996.
- Meyer, J.-P., "Solar-Stellar Outer Atmospheres and Energetic Particles, and Galactic Cosmic Rays", Astrophys. J. Suppl. 57, 173-204, 1985.
- Ahlen, S. P., "Theoretical and Experimental Aspects of the Energy Loss of Relativistic Heavily Ionizing Particles", Rev. Mod. Phys. 52, 121-173, 1980.
