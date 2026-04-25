---
title: "Investigation of the Composition of Solar and Interstellar Matter Using Solar Wind and Pickup Ion Measurements with SWICS and SWIMS on the ACE Spacecraft"
authors: [Gloeckler, Cain, Ipavich, Tums, Bedini, Fisk, Zurbuchen, Bochsler, Fischer, Wimmer-Schweingruber, Geiss, Kallenbach]
year: 1998
journal: "Space Science Reviews"
doi: "10.1023/A:1005036131689"
topic: Space_Weather
tags: [ACE, SWICS, SWIMS, mass-spectrometer, time-of-flight, solar-wind-composition, pickup-ions, freeze-in, instrument-paper]
status: completed
date_started: 2026-04-25
date_completed: 2026-04-25
---

# 71. SWICS and SWIMS on ACE: Solar Wind and Pickup Ion Composition / ACE의 SWICS와 SWIMS — 태양풍과 픽업 이온 조성

---

## 1. Core Contribution / 핵심 기여

이 논문(Gloeckler et al. 1998, Space Sci. Rev. 86, 497–539)은 1997년 8월 25일 발사된 NASA의 **ACE(Advanced Composition Explorer)** 위성에 탑재된 두 개의 정밀 질량 분광계 — **SWICS(Solar Wind Ion Composition Spectrometer)** 와 **SWIMS(Solar Wind Ions Mass Spectrometer)** — 의 설계, 측정 원리, 성능, 그리고 과학적 목표를 종합적으로 기술하는 기기 논문(instrument paper)이다. 두 기기는 모두 **정전기 편향 분석기(ESA)에 의한 E/Q 선택**을 1차 단계로 사용하지만, 그 다음 단계가 다르다. SWICS는 **ESA → 후가속(≤30 kV) → 비행시간(TOF, 10 cm) → 잔여 에너지 (SSD)** 의 5단계 조합으로 H부터 Fe까지의 모든 주요 태양풍 이온의 **질량 M, 전하 상태 Q, 입사 에너지 E, 속도 V** 를 동시에 결정하며, 픽업 이온까지 100 keV e⁻¹ 영역까지 측정한다. SWIMS는 **ESA → 박막 통과로 전하 평형(주로 Q*=1) → 하모닉 정전기 포텐셜에서 TOF** 만으로 M/ΔM > 100의 이례적 고분해능 질량 분광을 달성하며, He에서 Ni까지의 모든 원소와 대부분의 동위원소(예: ²⁰Ne, ²²Ne)를 분리한다. 이 논문은 이후 25년 이상 태양풍 조성 연구의 표준이 된 데이터의 측정 근거를 제공한다.

This paper (Gloeckler et al. 1998, Space Sci. Rev. 86, 497–539) is the comprehensive instrument paper describing two high-precision mass spectrometers — **SWICS (Solar Wind Ion Composition Spectrometer)** and **SWIMS (Solar Wind Ions Mass Spectrometer)** — aboard NASA's **ACE (Advanced Composition Explorer)** spacecraft launched 25 August 1997. Both instruments use **electrostatic deflection (ESA) for E/Q selection** as the first stage, but diverge afterwards. SWICS combines **ESA → post-acceleration (≤30 kV) → time-of-flight (TOF, 10 cm) → residual energy (SSD)** to simultaneously determine **mass M, charge state Q, incident energy E, and velocity V** for every major solar wind ion from H through Fe, extending up to 100 keV e⁻¹ for pickup ions. SWIMS uses **ESA → carbon-foil charge-equilibration (mostly Q*=1) → TOF in a harmonic electrostatic potential** alone to achieve an exceptional mass resolution M/ΔM > 100, resolving all elements from He through Ni and most isotopes (e.g. ²⁰Ne vs ²²Ne). The paper provides the measurement foundation for data that have become the standard reference for over 25 years of solar wind composition research.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction and Scientific Objectives / 도입과 과학적 목표 (Sections 1–2, pp. 498–506)

논문은 태양풍의 기원과 가속이 여전히 미해결 과제이며, 정밀 조성 측정과 원격 관측의 결합이 가장 유망한 접근임을 명시한다. ACE 궤도(L1 라그랑주점, ~0.99 AU)의 ecliptic 위치는 Ulysses(out-of-ecliptic, 동일한 SWICS 탑재)와의 동시 관측을 통해 위도/경도 차이를 분리할 수 있게 한다. SWICS가 H~Fe의 원소·전하·동력학을 측정하고 SWIMS는 더 긴 시간 평균에서 원소·동위원소 조성을 정확하게 측정하는 **상보적(complementary)** 역할을 한다.

The paper establishes that solar wind origin and acceleration remain unsolved, and that combining precision composition measurements with remote sensing is the most promising approach. ACE's L1 ecliptic location enables simultaneous observations with Ulysses (out-of-ecliptic, same SWICS heritage), separating latitudinal versus temporal effects. SWICS measures elemental, charge, and kinetic properties of H through Fe; SWIMS provides accurate elemental and isotopic composition over longer averages — the two instruments are **complementary**.

**Ten scientific objectives (paper §2):**
1. Solar abundances (elemental + isotopic) across solar cycle.
2. Solar wind acceleration via heavy-ion composition + charge states.
3. Atom-ion separation processes (FIP effect).
4. Acceleration-region physical properties via charge-state distributions of C, O, Mg, Si, Fe.
5. Plasma processes — thermal speeds and distribution functions of H to ~100 keV e⁻¹.
6. Interplanetary acceleration (CIRs, ESPs, upstream ions).
7. Galactic chemical evolution: ³He/⁴He ratio in interstellar gas vs solar wind.
8. Local interstellar cloud characterization (Ne, O abundance; ²²Ne/²⁰Ne; T_He, T_O).
9. Inner-source pickup ions (C⁺, N⁺, O⁺ velocity distributions for V < V_sw).
10. Magnetospheric input characterization (mass, charge, energy of incoming ions).

**Freeze-in physics (§2.1.1, p. 501).** 태양풍 팽창 시간 τ_exp가 이온화/재결합 시간보다 짧아지면 전하 상태가 동결된다(Hundhausen 1968). 산소는 ~1.5 R_⊙에서, 철은 ~3 R_⊙에서 동결되어 코로나 전자 온도의 **반경 프로파일**을 직접 진단할 수 있다. SWICS는 10⁶ K부터 2×10⁶ K 이상까지의 온도 영역을 다룬다. / When the solar wind expansion time becomes long compared to ionization/recombination times, the charge state is "frozen in" (Hundhausen 1968). Oxygen freezes near 1.5 R_⊙ and iron near 3 R_⊙, giving a direct probe of the **radial profile** of coronal T_e.

### Part II: SWICS Instrument / SWICS 기기 (Section 3.1, pp. 506–518)

**Five basic elements (Figure 1):**
1. **Multi-slit collimator** — 큰 영역의 입사 트래직토리를 선별. 메인 채널 주공간: 87 cm² 슬릿 영역, 18 plates, 2960 채널, 등방적 기하인자(geometrical factor) 2×10⁻³ cm² sr.
2. **Electrostatic deflection (ESA)** — UV 트랩 + E/Q 필터. 메인 채널 0.49–100.0 keV/q, 분해능 6.4%; H/He 채널 0.16–15.05 keV/q, 분해능 5.2%. 60 단계 로그 스텝, 1.0744 비율로 증가.
3. **Post-acceleration (PAPS, ≤30 kV)** — 후가속을 통해 SSD 임계(25–35 keV)를 넘기 위함.
4. **TOF (10 cm 카본 포일–SSD 사이)** — 시작·정지 검출기는 secondary electrons로 트리거. 시작: 1.5 μg cm⁻² 카본 포일 + 86% 투과 니켈 그리드, 정지: SSD의 Au 표면. MCP 게인 2×10⁶ at 3 kV.
5. **Solid-State Detector (SSD)** — 3개 직사각형 SSD (11.9 mm × 13.9 cm × 309 μm), 35–600 keV 범위, 8 keV FWHM 잡음.

**Fundamental measurement equations (Eq. 1, p. 508):**

$$M = 2\left(\frac{\tau}{d}\right)^{2}\frac{E_\text{meas}}{\alpha}$$

$$Q = \frac{E_\text{meas}/\alpha}{(U_a + E/Q)\beta} \approx \frac{E_\text{meas}/\alpha}{U_a}$$

$$\frac{M}{Q} = 2\left(\frac{\tau}{d}\right)^{2}(U_a + E/Q)\beta \approx 2\left(\frac{\tau}{d}\right)^{2}U_a$$

$$E_\text{ion} = Q \cdot \left(\frac{E}{Q}\right)$$

$$V_\text{ion} = 438 \sqrt{(E/Q)/(M/Q)} \quad [\text{km s}^{-1};\ E/Q\text{ in keV e}^{-1},\ M/Q\text{ in amu e}^{-1}]$$

여기서 d=10 cm, β는 시작 박막에서의 작은 에너지 손실 보정, α는 SSD에서의 핵 결손(nuclear defect; Ipavich et al. 1978). 후가속 U_a이 E/Q보다 훨씬 크므로 근사식이 성립한다. / Here d=10 cm; β corrects for small energy loss in the start foil; α accounts for nuclear defect in SSD. Approximations hold because U_a ≫ E/Q.

**Key insight: triple coincidence.** start-stop-energy 3중 일치 조건은 SSD 단독의 ~10⁻²~1 count s⁻¹ 배경을 지수적으로 감소시킨다. **TCR(Triple Coincidence Rate)** 가 telemetry에 보고된다. / Triple coincidence reduces SSD background by orders of magnitude.

**TOF telescope (Figure 5).** Carbon foil에서 방출된 secondary electrons는 ~1 kV로 가속되어 chevron MCP에 도달한다. 비행 경로 분산 Δd/d < 0.005, 이차 전자 시간 분산 0.3 ns, 전체 분해능 < 0.5 ns FWHM. T-ADC: 1024 채널, 0.176 ns/channel; E-ADC: 256 채널, 2.34 keV/channel.

**Telemetry (Table III, p. 517).**
- Monitor rates: FSR (front start), DCR (double coincidence), TCR (triple coincidence), MSS (combined SSD), PROT, ALFA.
- Matrix elements (MEi): 30 m vs m/q 박스로 정의된 15 항목.
- Matrix rates (MRi): 8개 선택된 종.
- PHA (Pulse-Height Analysis): 194 24-bit E/T events per spin. Total: 505.33 bit s⁻¹.

### Part III: SWIMS Instrument / SWIMS 기기 (Section 3.2, pp. 518–525)

**SWIMS principle.** ESA로 E/Q 선택 후 박막에서 전하 평형: 대부분 Q*=0(중성) 또는 +1, 소수만 ≥+2 또는 −1. **하모닉 정전기 포텐셜** V(z) ∝ z²을 사용함으로써 단순조화진동(SHO) 관계로 환원된다. 단순조화 진동의 주기는 **속도와 입사각에 무관**하므로:

$$\boxed{\tau \propto \sqrt{M/Q^{*}}}\qquad(\text{SWIMS Eq. 2, p. 520})$$

Q*=1인 대부분의 경우 τ ∝ √M이 되고, M/ΔM > 100을 달성한다. TOF 범위 60–460 ns; 1024 byte 스펙트럼.

**WAVE (Wide-Angle Variable Energy/charge) entrance.** Wide-acceptance ESA(다중 챔버 반사 시스템). 60 step의 E/Q를 0.5–9.5 keV e⁻¹ 범위에서 로그 스텝으로 스캔. 분해능 ~5%.

**Acceleration/deceleration system V_F.** −5.0 ~ +5.0 kV 가변 전위로 SWIMS HMRS 전체를 floating. 저속 이온(~200 km s⁻¹)을 가속해 박막 통과 효율을 높이고, 고속 Fe(~600 km s⁻¹)는 감속해 ESA passband 안에 가둔다. 256 단계 (∼40 V step).

**HMRS detail (Figure 10).** 카본 포일(<2 μg cm⁻², 4×15 mm) 통과 → 시작 신호. 양이온은 finned hyperbola(전위 V_H, 최대 +30 kV)에서 반사되어 stop MCP(100×15 mm) 도달. **Finned hyperbola**의 핀 구조는 hyperbola 표면에 부딪힌 중성/음이온이 sputter 산란해 stop MCP에 도달하는 것을 차단해 백그라운드를 매우 낮춘다.

**Telemetry (Table IV, p. 526).** PHA 99 events × 48 bits = 396 bit s⁻¹가 주된 부담. TOFi spectrum 35 byte/spin, BRi 4 sectored basic rates, FSR variants. Total 510 bit s⁻¹.

### Part IV: Data Processing / 데이터 처리 (Section 3.3, pp. 525–528)

**S³DPU (SEPICA/SWICS/SWIMS DPU)** — 텔레메트리 이전에 spin마다 fast classification:
- SWICS: (T, E) 룩업 테이블 → M, M/Q 결정 → matrix box로 카운트.
- SWIMS: T와 hyperbola voltage → M 결정.
- 4 Basic Rates(coarse), 8 Matrix Rates(medium), 15 Matrix Elements(fine) per SWICS.

**Figure 12 (matrix boxes).** Mass vs mass-per-charge 평면에서 H⁺(1,1), He²⁺(4,2), He⁺(4,4), O(16,~2.4), Si, Iron 박스가 정의된다.

### Part V: Performance / 성능 (Sections 4–5, pp. 528–534)

**Energy range.** SWICS: 110 eV/q (145 km s⁻¹ for H) – 100 keV/q (4380 km s⁻¹ for H, 1660 km s⁻¹ for Fe⁸⁺). 다이내믹 레인지 1000.

**Intensity dynamic range.** ~10⁹ — H/He 채널이 ~10⁹ cm⁻²s⁻¹ proton flux 수용; H/He 검출기 최대 5×10⁵ count s⁻¹.

**Mass and M/Q resolution (Table VI).** 440 km s⁻¹ 태양풍 + 30 kV 후가속 가정.

| Element | Mass (amu) | Charge (e) | E (keV) | τ (ns) | Δm/m | Δ(m/q)/(m/q) |
|---|---|---|---|---|---|---|
| H | 1 | 1 | 26 | 38.5 | 0.472 | 0.052 |
| He | 4 | 2 | 52 | 53.1 | 0.253 | 0.043 |
| C | 12 | 6 | 143 | 52.6 | 0.163 | 0.035 |
| N | 14 | 7 | 164 | 52.6 | 0.167 | 0.035 |
| O | 16 | 6 | 131 | 60.2 | 0.221 | 0.035 |
| Ne | 20 | 8 | 166 | 58.1 | 0.220 | 0.028 |
| Si | 28 | 9 | 174 | 64.1 | 0.246 | 0.028 |
| S | 32 | 10 | 189 | 64.9 | 0.254 | 0.028 |
| Fe | 56 | 11 | 157 | 79.3 | 0.302 | 0.026 |

질량 분해능은 Δm/m ~ 0.16–0.47 (가벼운 종일수록 나쁨), M/Q 분해능은 모든 종에서 ≤ 5%. **SWIMS는 Δm/m < 0.02**로 약 10배 우수. / Mass resolution Δm/m ~ 0.16–0.47 (worse for light); M/Q resolution ≤ 5% across species. **SWIMS reaches Δm/m < 0.02**, ~10× better.

**Stop efficiencies (Figure 15).** 0.5–6 keV/nuc 영역에서 Kr ~0.5, Ar/O/C ~0.3, He ~0.1. 카본 포일에서의 산란/이차전자 생성에 좌우된다.

**SWIMS calibrations (Figure 16, 17).** Bern 가속기에서 1995–1997년 3차 calibration; Giessen에서 ECR 소스로 Si, S, Na, Fe, O⁵⁺, Ti, Ni, Co, Zn, Cr 빔. ²⁰Ne와 ²²Ne가 명확히 분리됨.

**Mass and power (Table V).**
- SWICS: 5970 g, 4950 mW(평균), 6.11 W(피크).
- SWIMS: 8050 g, 6800 mW(평균), 7.2 W(피크).

### Part VI: Browse Parameters and Summary / 브라우즈 파라미터와 요약 (Sections 6–7, pp. 534–536)

12분 평균(또는 그 배수) 브라우즈 파라미터:
1. He²⁺와 O⁶⁺의 bulk speed와 kinetic temperature.
2. O⁶⁺/O⁷⁺로부터 freeze-in 코로나 전자 온도.
3. Fe/O (low-FIP/high-FIP) abundance ratio.
4. He/O abundance ratio.

요약은 SWICS + SWIMS의 상보성을 강조한다: SWICS는 모든 시간/속도에서의 charge & dynamics, SWIMS는 정밀 elemental/isotopic.

---

## 3. Key Takeaways / 핵심 시사점

1. **Two complementary instruments exhaust the composition phase space / 두 기기로 조성 매개변수 공간을 모두 커버.** SWICS는 빠른(12 min) 시간 분해능과 charge-state 측정에 최적화; SWIMS는 더 긴 평균(>1 hr)에서 isotope 분해능에 최적화. 함께 운용되어 동일한 ACE 위성에서 24/7 종합 조성 모니터링을 수행한다. / SWICS = fast charge states; SWIMS = high-resolution isotopes; together they provide 24/7 composition monitoring.

2. **Triple coincidence and harmonic isochronism are the two key tricks / 3중 일치와 하모닉 등시성이라는 두 핵심 트릭.** SWICS는 start–stop–energy 일치로 ~10⁻²~1 cps 배경을 무시할 수준까지 낮춘다. SWIMS는 V ∝ z² 포텐셜의 등시성으로 입사 속도 의존성을 제거해 M/ΔM>100을 달성한다. 두 트릭은 모두 1970년대부터 발전한 입자 물리 기법을 우주 환경에 적용한 사례이다. / Triple coincidence (SWICS) and harmonic isochronism (SWIMS) — both are particle-physics techniques adapted to space.

3. **Equation (1) is a complete identification recipe / 식 (1)은 완전한 식별 레시피.** τ와 E_meas 두 측정만으로 (E/Q는 ESA가 알려주므로) M, Q, V를 동시에 결정한다. 이는 자기 분광계(magnetic mass spectrometer)와 비교해 magnet의 무게/소비전력 없이 같은 정보를 얻는 방법이다. / Two measurements (τ, E_meas) suffice to determine M, Q, V — equivalent power to a magnetic spectrometer without the weight and power of magnets.

4. **Freeze-in temperature is the first-class scientific output / 동결 온도가 1급 과학 산출물.** O⁶⁺/O⁷⁺ 비는 ~1.5 R_⊙ 코로나 전자 온도(~1.5–2.5 MK)를 직접 측정한다. fast wind는 낮은 freeze-in T (~1.2 MK), slow wind는 높은 freeze-in T (~1.6 MK)로 구분되어 코로나 기원 진단의 결정타가 된다. / The O⁶⁺/O⁷⁺ ratio provides freeze-in T_e (~1.5 R_⊙) — fast wind ~1.2 MK, slow wind ~1.6 MK, distinguishing coronal sources.

5. **Pickup ions become a window into the inner heliosphere neutrals / 픽업 이온은 내부 헬리오스피어 중성 입자의 창.** SWICS의 0.5–100 keV e⁻¹ 영역과 12분 시간 분해능으로 He⁺, Ne⁺, O⁺의 분포 함수를 측정 가능. 단열 냉각 V<V_sw 영역의 pickup-ion phase-space density가 R(중성 발생 위치) 별 적분이므로, 이를 inverse하면 inner-source(0.1–0.5 AU 먼지 증발)의 spatial distribution을 매핑할 수 있다. / SWICS measures pickup ion distribution functions (0.5–100 keV e⁻¹, 12 min cadence), enabling mapping of inner-source neutral distributions via Vasyliunas–Siscoe inversion.

6. **Carbon foils are the unsung hero / 카본 포일은 숨은 주인공.** SWICS의 1.5 μg cm⁻² 시작 포일은 secondary-electron 시작 신호를, SWIMS의 <2 μg cm⁻² 입구 포일은 charge-equilibration(Q*=±1)과 secondary-electron 시작 신호를 동시에 제공한다. 포일에서의 에너지 손실(β factor), 산란(0.005 path-length dispersion), nuclear defect(α factor)는 모든 분해능 식의 핵심 보정항이다. / Carbon foils provide the start signal AND charge equilibration; their energy loss, scattering and defect corrections are the heart of resolution calculations.

7. **On-board classification beats raw telemetry / 보드 위 분류로 텔레메트리 한계 돌파.** 24-bit/event PHA를 spin마다 ~194 (SWICS) + 99 (SWIMS) = 293개만 다운링크. 그 외에는 S³DPU가 룩업 테이블로 사전 분류해 Matrix Element/Rate로 압축 — 데이터의 "관심 영역"만 보존. 1990년대 후반 ISTP 임무 통신 대역폭(~510 bit s⁻¹) 한계 안에서 무손실 과학을 가능케 한 설계 패턴. / On-board classification trims raw 293 PHA events/spin via lookup tables to matrix-element rates, preserving science within ~510 bit s⁻¹ telemetry budgets.

8. **The instrument paper IS the calibration baseline / 기기 논문이 곧 보정 기준선.** Tables I–VI, Figures 13/14/15/16/17은 25년 동안 ACE 데이터 분석 코드(SWICS Level 2 처리)에서 직접 인용되는 수치이다. β, α, MCP 효율, geometrical factor 등 모든 변환 상수가 이 논문에서 제공된다. / Tables I–VI and Figures 13–17 are the calibration baseline directly used in SWICS Level-2 processing for 25 years.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 SWICS ion identification equations / SWICS 이온 식별 방정식

ESA가 E/Q를 선택하고, 후가속 U_a로 ion 에너지를 증가시키며, TOF d=10 cm, SSD가 잔여 에너지 E_meas를 측정. 카본 포일을 통과한 후의 ion 운동에너지는

$$E_\text{after foil} = Q\cdot(E/Q) + Q U_a - \Delta E_\text{foil}$$

이고, 시작 직후 속도는

$$v = \sqrt{\frac{2 E_\text{after foil}}{M}}\implies \tau = \frac{d}{v}=d\sqrt{\frac{M}{2 E_\text{after foil}}}$$

따라서

$$M = 2\left(\frac{\tau}{d}\right)^{2}E_\text{after foil}=2\left(\frac{\tau}{d}\right)^{2}\frac{E_\text{meas}}{\alpha}$$

여기서 α는 SSD nuclear defect (Ipavich et al. 1978) 보정. / α corrects nuclear defect.

E_meas와 E_after_foil의 관계 (β: 박막 에너지 손실 인수):

$$E_\text{meas} = \alpha\cdot E_\text{after foil},\quad E_\text{after foil}=\beta\cdot Q (E/Q + U_a)$$

전하 상태:

$$Q=\frac{E_\text{meas}/\alpha}{(U_a + E/Q)\beta}\approx \frac{E_\text{meas}/\alpha}{U_a}\quad(U_a\gg E/Q)$$

질량/전하:

$$M/Q = 2(\tau/d)^{2}(U_a + E/Q)\beta\approx 2(\tau/d)^{2}U_a$$

속도:

$$V_\text{ion}=438\sqrt{\frac{E/Q}{M/Q}}\ \text{km s}^{-1}\quad(E/Q \text{ in keV/e},\ M/Q\text{ in amu/e})$$

이 단순한 4개 식이 SWICS 데이터 분석의 전부이다.

### 4.2 SWIMS harmonic isochronism / SWIMS 하모닉 등시성

V(z) = V_H(z/L)² 형태(harmonic potential). 양이온의 운동방정식 (1D):

$$M\ddot z = -Q^{*}\frac{dV}{dz}=-2 Q^{*}\frac{V_H}{L^{2}}z\implies \omega = \sqrt{\frac{2 Q^{*}V_H}{M L^{2}}}$$

**SHO의 주기는 진폭(즉, 입사 속도)과 무관**하므로, 이온이 카본 포일에서 출발해 hyperbola에서 반사되어 stop MCP로 돌아오는 시간(반주기 + 직선 구간 보정)은:

$$\tau = \pi/\omega + \tau_\text{linear} \propto \sqrt{M/(Q^{*}V_H)}$$

V_H를 25–30 kV로 고정하면 τ ∝ √(M/Q*). 대부분의 ion이 박막 통과 후 Q*=1이므로:

$$\boxed{\tau \propto \sqrt{M}}$$

질량 분해능은 dτ/τ = (1/2)(dM/M)이므로 dM/M = 2 dτ/τ. τ 측정 정밀도가 0.5 ns이고 τ ~ 100 ns이면 dτ/τ ~ 0.005, dM/M ~ 0.01 — M/ΔM ≥ 100. / Mass resolution dM/M ≈ 2 dτ/τ; with 0.5 ns/100 ns timing, M/ΔM ≥ 100.

### 4.3 ESA E/Q analyzer / ESA E/Q 분석기

이상적 평행 deflection plate에서 E/Q는 deflection voltage V_D에 비례:

$$\frac{E}{Q} = K_\text{ana}\,V_D$$

SWICS main channel: K_ana = 12.46 (E/Q in keV/q, V_D in V·100 등 단위 변환 포함). 60 단계 로그 스캔, step ratio 1.0744 → 60 step에서 1.0744⁵⁹ ≈ 71배. 0.49 keV/q × 71 ≈ 35 keV/q (실제로는 0.49–100 keV/q 도달).

### 4.4 Triple-coincidence background suppression / 3중 일치 배경 억제

각 검출기 단독 잡음률을 r_S (start MCP), r_T (stop MCP), r_E (SSD)라 하면, 우연 일치율은:

$$r_\text{accidental, triple}\sim r_S\cdot r_T\cdot \Delta\tau_\text{ST}\cdot r_E\cdot\Delta\tau_\text{TE}$$

Δτ_ST ~ 200 ns, Δτ_TE ~ 200 ns이고 r_S, r_T, r_E ~ 10³–10⁴ s⁻¹라면:

$$r_\text{acc}\sim 10^{4}\cdot 10^{4}\cdot 2\times 10^{-7}\cdot 10^{3}\cdot 2\times 10^{-7}\sim 4\times 10^{-3}\ \text{s}^{-1}$$

→ 단일 SSD 잡음률 1 cps 대비 ~250배 감소. 실제로는 더 큰 감소.

### 4.5 Freeze-in temperature from O⁷⁺/O⁶⁺ / 동결 온도

코로나 전자 온도 T_e에서 ionization (C_n) 및 recombination (R_n+1) 계수가 균형을 이루고, 팽창 시간이 길어질 때:

$$\frac{n(O^{q+1})}{n(O^{q})}=\frac{C_q(T_e)}{R_{q+1}(T_e)}$$

O⁶⁺ → O⁷⁺ 천이 ionization energy ΔE₆₇ ≈ 138.1 eV. 단순 Boltzmann 근사:

$$\ln\frac{n_{O^{7+}}}{n_{O^{6+}}}\approx -\frac{\Delta E_{67}}{k_B T_e}+\text{const}$$

→ 비율을 측정해 T_e ~ 1.5–2.5 MK 영역에서 ~10% 정밀도로 freeze-in 온도 추정.

### 4.6 Pickup-ion velocity distribution / 픽업 이온 속도 분포

중성 원자가 광이온화 후 vsw×B에서 pickup된 후 단열 냉각으로 형성되는 분포 (Vasyliunas & Siscoe 1976):

$$f(w,r)=\frac{3}{8\pi}\frac{n_n(r) \nu_\text{ion}}{V_{sw}^{3}}w^{-3/2}\Theta(1-w)$$

여기서 w = V/V_sw, r = R/R_⊙. SWICS는 1 AU에서 측정한 f(w)를 통해 R(=heliocentric distance)에서의 중성 밀도 n_n(R) 매핑 가능.

### 4.7 Mass resolution budget / 질량 분해능 예산

SWICS 질량 분해능은 quadrature sum:

$$\left(\frac{\Delta M}{M}\right)^{2}=\left(\frac{\Delta\tau}{\tau}\right)^{2}_\text{TOF}+\left(\frac{\Delta E}{E}\right)^{2}_\text{SSD}+\left(\frac{\Delta d}{d}\right)^{2}+\left(\frac{\Delta(E/Q)}{E/Q}\right)^{2}_\text{ESA}+\delta_\text{nuclear}^{2}$$

기여 항목:
- ESA dispersion: < 0.5%
- TOF electronic: 0.5 ns FWHM, 0.3 ns secondary electron → ~0.6 ns total / τ ~ 0.01–0.02
- SSD electronic: 5–8 keV FWHM / 50–600 keV → 0.01–0.16
- Path length: < 0.005
- Nuclear defect spread: dominant for low-energy heavy ions

→ Table VI의 Δm/m 값을 정량적으로 재현.

### 4.8 Worked numerical example: O⁶⁺ at 440 km s⁻¹ / 작업 예제: 440 km s⁻¹의 O⁶⁺

태양풍 O⁶⁺ (M=16, Q=6, V=440 km s⁻¹), 후가속 U_a=30 kV.

- 동력학적 에너지: KE = ½ × 16 × 1.66×10⁻²⁷ × (440×10³)² = 2.57×10⁻¹⁶ J ≈ 1.61 keV.
- E/Q = 1.61/6 ≈ 0.268 keV/e (이는 ESA 메인 채널 0.49–100 keV/e 하한 근처).
- ESA 통과 후 후가속: total ion energy = Q(E/Q + U_a) = 6 × (0.268 + 30) = 181.6 keV.
- β ~ 0.7 (carbon foil energy loss ~30%): E_after_foil ≈ 127 keV.
- α ~ 0.6 (nuclear defect for heavy O on Si): E_meas ≈ 76 keV.

논문 Table VI (440 km s⁻¹ + 30 kV)의 O 값 E=131 keV는 위 모델보다 약간 더 큰 값을 보고하는데, 이는 V=440 km s⁻¹가 이미 ESA의 두 번째 step 이상에서 측정됨(즉, 더 큰 E/Q step)을 시사한다. — TOF τ_predicted = d/v_after = 0.10 m / sqrt(2×127×10³×1.6×10⁻¹⁹/(16×1.66×10⁻²⁷)) ≈ 0.10/1.24×10⁶ m s⁻¹ ≈ 81 ns.

Table VI 보고값: τ(O at 440 km/s, 30 kV) = 60.2 ns. 차이는 v_after 계산 시 ion이 +6에서 +1로 박막 통과 후 charge 변화하지 않는다는 가정(SWICS의 경우 charge state 보존됨, SWIMS 와 다름)과 β 추정의 차이에 기인. — 실제 측정 시 룩업 테이블이 모든 보정을 흡수.

### 4.9 Geometric factor budget / 기하 인자 예산

**SWICS main channel:**

$$G_\text{eff} = G_0 \cdot \epsilon_\text{ESA}\cdot\epsilon_\text{start}\cdot\epsilon_\text{stop}\cdot\epsilon_\text{SSD}$$

- G_0 (geometrical factor / channel) = 7×10⁻⁷ cm² sr (Table I).
- N_channels = 2960 → 등방 G_iso = 2×10⁻³ cm² sr.
- ESA passband 효율 ε_ESA ≈ 1 (passband 안에서).
- Start MCP 효율 ε_start: 카본 포일에서의 secondary electron yield × MCP detection (typical 0.5–0.8 for heavy ions).
- Stop MCP 효율 ε_stop: SSD에서의 backscattered secondary electron × MCP (Figure 15: 0.1–0.5).
- SSD 효율 ε_SSD ≈ 1 (after threshold ~25 keV).

총 효율 ε ≈ 0.05–0.4 종 의존. / Total efficiency 0.05–0.4 species-dependent.

### 4.10 Stepping cycle and time resolution / 스텝 사이클과 시간 분해능

ACE spacecraft spin rate = 5.0 rpm, spin period T_spin = 12 s.

- Per spin: deflection voltage steps from V_max → V_min in 60 logarithmic steps; one E/Q step per spin.
- Full cycle: 60 spins × 12 s = 12 min = 720 s.
- Browse parameters reported every 12 min ≈ one full E/Q sweep.
- For pickup ions one needs ≥1 full cycle to map V/V_sw from ~0.5 to ~2.

8 azimuthal sectors (45° each) per spin enable directional sampling of the velocity distribution and identification of the solar-wind look direction.

---

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
 1968 ── Hundhausen, Gilbert & Bame: freeze-in concept
              │
 1972 ── Geiss et al.: Apollo 16 SWC foil / 솔라풍 동위원소
              │
 1977 ── Gloeckler: TOF-vs-E principle (UMd TR 77-043)
              │
 1979 ── Gloeckler & Hsieh: secondary-electron TOF (NIM 165)
              │
 1985 ── Möbius et al.: He+ pickup ions discovered (AMPTE/SULEICA)
              │
 1990 ── Hamilton et al.: HMRS technique (RSI 61, 3104)
        Ulysses launch with first SWICS
              │
 1992 ── Gloeckler et al.: Ulysses SWICS instrument paper
              │
 1995 ── SOHO launch: CELIAS-MTOF (Hovestadt et al.)
        Geiss et al.: inner-source pickup ions at Ulysses
              │
 1995b ── Gloeckler et al.: WIND/SMS instrument paper
              │
 1997 Aug ── ACE launch (this paper's hardware)
              │
 1998 ── ★ THIS PAPER (Gloeckler et al., Space Sci. Rev. 86)
              │
 2000 ── Schwadron, Geiss: inner-source physics
              │
 2002 ── Zurbuchen et al.: ACE/SWICS slow vs fast composition
              │
 2007 ── von Steiger et al.: SWICS Ulysses+ACE longitudinal/latitudinal
              │
 2018 ── Parker Solar Probe launch: SWEAP/SPAN-Ion
 2020 ── Solar Orbiter: SWA/HIS (direct SWICS heir)
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Hundhausen, Gilbert & Bame (1968) | Original "freeze-in" concept; basis for charge-state thermometry / 동결 개념의 원형 | High — SWICS의 핵심 과학 원리. SWICS measures freeze-in T directly. |
| Gloeckler et al. (1992, A&AS 92) — Ulysses SWICS | Heritage instrument; ACE SWICS is improved flight spare / Ulysses용 동일 설계 | Critical — same equations, slightly improved electronics. |
| Hamilton et al. (1990, RSI 61, 3104) | HMRS technique paper for SWIMS predecessor (WIND/SMS MASS) / SWIMS 선조 기기 | Critical — HMRS principle directly inherited. |
| Hovestadt et al. (1995, Solar Phys. 162) — SOHO/CELIAS | MTOF technique combined with MASS to form SWIMS / MTOF + MASS = SWIMS | High — direct ancestor of SWIMS. |
| Möbius et al. (1985, Nature 318) | Discovery of interstellar He+ pickup ions; motivates SWICS pickup-ion science / 성간 픽업 이온 발견 | High — underpins objectives 8–9. |
| Möbius et al. (1998, this issue) | S³DPU and SEPICA paper; same DPU shared with SWICS/SWIMS / 동일 DPU 공유 | Direct — shared electronics. |
| Geiss et al. (1995, Science 268) | Inner-source pickup ions discovered at Ulysses with SWICS / 내부 source 발견 | High — provides context for Section 2.3. |
| Ipavich et al. (1978, NIM 154) | Nuclear defect α factor measurements / 핵 결손 α 측정 | Medium — calibration constants used in Eq. (1). |
| Gloeckler & Hsieh (1979, NIM 165) | Secondary-electron TOF technique; foundational / 이차 전자 TOF 기법 | Medium — design heritage. |
| Möbius et al. (1995, A&A 304) — pickup-ion modelling | Theoretical f(w) model used for SWICS pickup analysis | Medium — interpretive framework. |
| Stone et al. (1998, Space Sci. Rev. 86, 1) | ACE mission overview paper | Direct — companion paper in same SSR volume. |
| Zurbuchen & von Steiger (2007, Living Rev. Sol. Phys.) | Modern review built on SWICS/SWIMS data | Future — shows scientific impact. |

---

## 7. References / 참고문헌

**Primary**
- Gloeckler, G., Cain, J., Ipavich, F. M., Tums, E. O., Bedini, P., Fisk, L. A., Zurbuchen, T. H., Bochsler, P., Fischer, J., Wimmer-Schweingruber, R. F., Geiss, J., and Kallenbach, R., "Investigation of the Composition of Solar and Interstellar Matter Using Solar Wind and Pickup Ion Measurements with SWICS and SWIMS on the ACE Spacecraft", *Space Science Reviews* **86**, 497–539, 1998. DOI: 10.1023/A:1005036131689.

**Cited in the paper / 논문이 인용한 주요 문헌**
- Hundhausen, A., Gilbert, H., and Bame, S., *J. Geophys. Res.* **73**, 5485, 1968.
- Geiss, J., Bühler, F., Cerutti, H., Eberhardt, P., and Filleux, C., *Apollo 16 Preliminary Science Report*, NASA SP-315, 1972.
- Gloeckler, G., UMd Technical Report TR 77-043, 1977.
- Gloeckler, G. and Hsieh, K. C., *Nucl. Inst. Meth.* **165**, 537, 1979.
- Ipavich, F. M., Lundgren, R. A., Lambird, B. A., and Gloeckler, G., *Nucl. Inst. Meth.* **154**, 291, 1978.
- Möbius, E., Hovestadt, D., Klecker, B., Scholer, M., Gloeckler, G., and Ipavich, F. M., *Nature* **318**, 426, 1985.
- Hamilton, D. C., Gloeckler, G., Ipavich, F. M., Lundgren, R. A., Sheldon, R. B., and Hovestadt, D., *Rev. Sci. Instrum.* **61**, 3104, 1990.
- Gloeckler, G. et al., *Astron. Astrophys. Suppl. Ser.* **92**, 267, 1992.
- Geiss, J. et al., *Science* **268**, 1033, 1995.
- Hovestadt, D. et al., *Solar Phys.* **162**, 441, 1995.
- Möbius, E. et al., *Space Sci. Rev.* **86**, 449, 1998.
- Vasyliunas, V. M. and Siscoe, G. L., *J. Geophys. Res.* **81**, 1247, 1976.

**Companion / Modern follow-ups / 동반 및 후속**
- Stone, E. C. et al., "The Advanced Composition Explorer", *Space Sci. Rev.* **86**, 1, 1998.
- Zurbuchen, T. H., Hefti, S., Fisk, L. A., Gloeckler, G., and von Steiger, R., *Space Sci. Rev.* **87**, 353, 1999.
- Zurbuchen, T. H. and von Steiger, R., *Living Rev. Sol. Phys.* **3**, 1, 2007.
- von Steiger, R. et al., *J. Geophys. Res.* **105**, 27217, 2000.
- Schwadron, N. A., Geiss, J., Fisk, L. A., Gloeckler, G., Zurbuchen, T. H., and von Steiger, R., *J. Geophys. Res.* **105**, 7465, 2000.
