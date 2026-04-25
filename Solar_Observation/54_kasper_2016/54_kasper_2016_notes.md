---
title: "Solar Wind Electrons Alphas and Protons (SWEAP) Investigation: Design of the Solar Wind and Coronal Plasma Instrument Suite for Solar Probe Plus"
authors: ["Justin C. Kasper", "Robert Abiad", "Gerry Austin", "Marianne Balat-Pichelin", "Stuart D. Bale", "John W. Belcher", "et al."]
year: 2016
journal: "Space Science Reviews"
doi: "10.1007/s11214-015-0206-3"
topic: Solar_Observation
tags: [parker_solar_probe, sweap, faraday_cup, electrostatic_analyzer, solar_wind, instrumentation, in_situ]
status: completed
date_started: 2026-04-25
date_completed: 2026-04-25
---

# 54. Solar Wind Electrons Alphas and Protons (SWEAP) Investigation / 태양풍 전자·알파·양성자 측정기

---

## 1. Core Contribution / 핵심 기여

Kasper et al. (2016) provide the Preliminary Design Review (PDR) snapshot of SWEAP, the four-sensor in-situ plasma suite on NASA's Solar Probe Plus (later Parker Solar Probe). The suite combines (i) the **Solar Probe Cup (SPC)** — a sun-staring Faraday Cup mounted on the edge of the heat shield with ±28° half-angle FOV, 50 eV–8 keV/q ion and 50 eV–2 keV electron range, four-quadrant collectors and an AC-coupled synchronous detection scheme; (ii) **SPAN-A**, a ram-side electrostatic analyzer pair (top-hat ion ESA with time-of-flight mass section + electron ESA, 240°×120° FOV); (iii) **SPAN-B**, the anti-ram-side electron ESA that completes near-4π sr electron coverage; and (iv) the **SWEAP Electronics Module (SWEM)**, the central FPGA-based controller, telemetry interface and 80 GB SPAN flash buffer. The paper traces driving science requirements from three overarching SPP objectives (Sources, Heating, Energetic Particles) through Level-1 measurement requirements (Table 3) to instrument-level performance (Tables 4–9), and shows via Monte Carlo simulation that the SPC + SPAN-A combination recovers the proton core 100% of the time and the electron strahl 98% over the final 9.86 R☉ orbit.

Kasper 등(2016)은 NASA Solar Probe Plus(PSP의 전신) 미션에 탑재될 SWEAP 측정기 묶음의 PDR 시점 종합 보고서를 제공한다. 4 개 센서 — (i) 열차폐 모서리에 장착되어 태양을 직시하는 Faraday Cup인 **SPC**(반각 ±28° FOV, 50 eV–8 keV/q 이온, 50 eV–2 keV 전자, 4 분면 collector + AC 동기 검파), (ii) 램 면 정전 분석기 쌍 **SPAN-A**(top-hat ion ESA + TOF 질량 분광 섹션 + 전자 ESA, 240°×120° FOV), (iii) 반(反)-램 면 전자 ESA **SPAN-B**(SPAN-A와 합쳐 거의 4π sr 전자 커버리지), (iv) FPGA 기반 중앙 제어기·텔레메트리·80 GB SPAN flash buffer를 담당하는 **SWEM** — 으로 구성된다. 논문은 SPP 미션의 3 대 과학 목표(태양풍의 기원 / 코로나·태양풍 가열 / 에너지 입자 가속)에서 출발하여, Level-1 측정 요구(Table 3) 와 측정기 사양(Tables 4–9) 을 단계적으로 도출하고, Monte Carlo 시뮬레이션으로 SPC + SPAN-A 결합이 양성자 core 100%, 전자 strahl 98% 의 시간 동안 검출됨을 입증한다(Fig. 9).

---

## 2. Reading Notes / 읽기 노트

### Part I: Science Background and Objectives (Sec. 1) / 과학 배경과 목표

#### 1.1 Three overarching objectives / 세 가지 과학 목표

The SWEAP science objectives mirror the SPP mission's three top-level goals, each broken into operational sub-goals:
1. **Sources of the solar wind** — connect large-scale solar wind structure to solar sources, understand acceleration, characterize the variable corona–solar-wind connection, discover small coronal structures embedded in the wind.
2. **Heating the corona and solar wind** — measure the energy budget of the solar wind, identify which heating mechanisms dominate vs distance, determine the limits imposed by instabilities and Coulomb relaxation.
3. **Acceleration and transport of energetic particles** — characterize particle acceleration by CMEs and IP shocks, study acceleration/transport from solar flares into the wind, determine the role of stochastic in-situ acceleration.

SWEAP의 과학 목표는 SPP 미션의 세 최상위 목표와 일치하며, 각 목표마다 측정기 설계를 결정짓는 sub-goal 들을 갖는다.
1. **태양풍의 기원** — 큰 규모 태양풍 구조와 태양 소스 영역 연결, 가속 메커니즘 이해, 코로나-태양풍 자기 연결의 변동성 측정, 태양풍에 내포된 작은 코로나 구조 발견.
2. **코로나·태양풍 가열** — 태양풍 에너지 수지 측정, 거리 함수로 가열 메커니즘 식별, 불안정성·Coulomb 완화에 의한 한계 규명.
3. **에너지 입자 가속·전달** — CME / IP 충격에 의한 입자 가속, 태양 플레어 입자의 태양풍 내 전송, in-situ 통계적 가속의 역할.

**Table 1** classifies in-situ solar wind into three regimes: slow (<400 km/s, T_α<T_p, near-Maxwellian, He/H ~1–5%), fast (700–900 km/s, T_α>T_p, highly non-Maxwellian, He/H ~5%), and transients (CMEs, 400–2000 km/s, highly variable). Figure 1 shows that *collisional age* A_c (Coulomb collision frequency × transit time) discriminates non-Maxwellian features better than bulk speed. SPP's young plasma will allow us to see if non-Maxwellian features are universal near the Sun.

**Table 1** 은 in-situ 태양풍을 세 영역으로 분류한다: 저속(<400 km/s, T_α<T_p, 거의 맥스웰), 고속(700–900 km/s, T_α>T_p, 강한 non-Maxwellian, He/H ~5%), 일시적 사건(CME, 400–2000 km/s). Fig. 1 은 *collisional age* A_c (Coulomb 충돌 빈도 × 전송 시간) 가 bulk 속도보다 non-Maxwellian 특성을 더 잘 가른다는 것을 보여준다. SPP가 sampling 할 어린 plasma 는 non-Maxwellian 특성이 보편적인지를 처음으로 검증할 기회다.

**Table 2** lists five candidate heating mechanisms, each with its observable signature and the SWEAP measurement requirement:
1. *Ion cyclotron resonant absorption* — high-frequency E/B + i+ anisotropy T_⊥>T_∥; require p+ flow angles, 2D i+ and e-VDF at 64 Hz cross-correlated with E and B.
2. *Turbulent cascade* — broken power law in PSD of B and V at the convected gyro-frequency; requires 30 Hz 2D VDFs.
3. *Shock-steepened acoustic modes* — n_p fluctuations at 16 Hz, weak shocks with thickness ~R_L.
4. *Reconnection / nanoflares* — bidirectional beams, energized e-strahl, 20% energy resolution to detect p+ and e- beams.
5. *Velocity filtration (kappa distributions)* — peaked VDFs from preferential escape of high-energy tail; needs <5% energy resolution and 2D i+ + e-halo measurements.

**Table 2** 는 다섯 가지 후보 가열 메커니즘과 그 관측적 특성, 그리고 SWEAP의 측정 요구를 정리한다. 이 표는 SWEAP 의 시간 분해능(64–128 Hz burst), 각 분해능, 에너지 분해능 사양을 직접 결정한다.

#### 1.2 Driving performance: FOV and cadence / 핵심 성능: 시야각과 측정 주기

Two technical drivers dominate SWEAP design:
- **Field of view**: The heat shield (TPS) blocks the anti-sun half-sky for instruments behind it. Far from the Sun (>0.3 AU) the wind is nearly radial; SPAN-A on the ram side cannot see the wind because the orbital aberration is small. **Closer to the Sun the orbital velocity at perihelion rises to ~200 km/s and tilts the apparent flow toward the ram side**, so SPC alone is insufficient and SPAN-A must look out from the ram. Strong waves can also cause flow-angle fluctuations of ±25° that require both SPC and SPAN-A. Figure 8 illustrates the proton VDF blocked by the heat shield at 9.5 R☉ and 38.4 R☉.
- **Cadence**: To resolve a single proton gyro-radius R_L (a few km at 10 R☉) at orbital speeds requires SPC's 128 Hz burst flow-angle measurement. Figure 2 shows SWEAP's 16 Hz survey + 128 Hz burst cadence relative to the time to convect/cross R_L.

A Monte Carlo simulation (Fig. 9) integrates probability of detecting the proton core and electron strahl over the final perihelion. Results: SPC alone — proton core 87%, e-strahl 71%; SPAN alone — proton core 24%, strahl 62%; **combined — proton core 100%, strahl 98%**. Table 4 then folds in shock/CME/HCS occurrence rates, showing >90% probability of meeting Level-1 science targets (≥5 CME-driven shocks, ≥20 HCS crossings within 0.25 AU).

두 기술적 추진력이 SWEAP 설계를 지배한다.
- **FOV**: TPS 가 차폐 뒤 측정기의 반(反)태양 반구를 가린다. 0.3 AU 바깥에서는 태양풍이 거의 방사형이라 ram 쪽 SPAN-A 가 흐름을 볼 수 없다. **태양에 다가갈수록 궤도 속도(근일점 ~200 km/s)가 흐름을 ram 쪽으로 기울이므로** SPC만으로는 불충분하고 SPAN-A 가 필수가 된다. 강한 파동은 ±25° 의 flow-angle 변동을 일으켜 두 측정기 결합 시야가 필수다.
- **Cadence**: 양성자 gyro-radius R_L(10 R☉ 에서 수 km) 를 해상하려면 SPC 128 Hz burst flow-angle 측정이 필요하다.

Monte Carlo 시뮬레이션(Fig. 9): SPC 단독 — proton core 87%, e-strahl 71%; SPAN 단독 — proton core 24%, strahl 62%; **결합 — proton core 100%, strahl 98%**. Table 4 는 shock/CME/HCS 발생률을 곱해 Level-1 과학 목표 달성 확률 >90% 를 보여준다.

### Part II: Suite Architecture (Sec. 2) / 묶음 구조

The SWEAP suite consists of the four sensors mentioned plus the SWEM. The **SWEM** has two boards: a Data Controller Board (DCB, MAVEN heritage) with a Coldfire processor in an FPGA, on-board moment computation, 10×8 GB flash memory, and a Low Voltage Power Supply Board (LVPS, ±5%-regulated PWM). All particle counts can also be relayed at high speed to FIELDS (Bale et al. 2016) for wave-particle correlations.

SWEAP 묶음은 4 개 센서 + SWEM 으로 구성된다. **SWEM** 은 두 보드를 갖는다: MAVEN 헤리티지의 Coldfire 프로세서를 FPGA에 IP-core로 구현한 DCB(온보드 모멘트 계산, 10×8 GB flash memory)와 ±5% 정류 PWM LVPS. 입자 카운트는 FIELDS (Bale et al. 2016) 로도 고속 중계되어 wave–particle 상관 분석을 가능케 한다.

The **electron FOV map** (Fig. 13) shows that SPAN-A and SPAN-B together cover all sky except the directly sunward direction (blocked by TPS, but covered by SPC up to 2 keV) and a small ~50° cone around (110°, -50°). The two electron analyzers are "rotated like the seams of a baseball" so their broad fields combine for full coverage.

**전자 FOV 지도**(Fig. 13)에서 SPAN-A와 SPAN-B는 태양 직시 방향(TPS 차폐, 다만 SPC가 2 keV 까지 커버)과 (110°, −50°) 부근 ~50° cone을 제외한 모든 하늘을 본다. 두 전자 분석기는 "야구공의 솔기처럼" 회전 배치되어 결합 시야가 완성된다.

### Part III: Solar Probe Cup — SPC (Sec. 3) / 솔라 프로브 컵

#### 3.1 Faraday Cup principle / Faraday Cup 원리

A Faraday cup measures the current produced on a metal plate by charged particles whose energy/charge exceeds the modulator-grid potential. SPC's modulator grid oscillates between two voltages V_low and V_high at f_mod = 1280 Hz with V_AC = 50–800 V. Only particles with E_∥/q ∈ [V_low, V_high] reach the collector plates, producing an AC current at f_mod whose amplitude is the integrated flux in that energy window. Synchronous (lock-in) detection at f_mod isolates the AC signal from DC noise sources: thermionic emission from hot surfaces, photoelectrons from sunlight, and SEP penetrating radiation. **Figure 15** demonstrates this AC immunity using Wind/SWE FC data: noise current is constant ~10⁻¹² A across SEP events with 5 orders of magnitude variation in >2 MeV electron flux.

Faraday Cup은 변조 격자 전압을 초과하는 에너지의 하전 입자가 금속판에 만드는 전류를 측정한다. SPC 변조 격자는 두 전압 V_low, V_high 사이를 f_mod = 1280 Hz, V_AC = 50–800 V 로 진동한다. E_∥/q ∈ [V_low, V_high] 인 입자만 collector 에 도달해 f_mod 진동수의 AC 전류를 만들고, 그 진폭이 해당 에너지 창의 적분 flux이다. f_mod 동기 검파는 DC 잡음(고온 표면의 thermionic emission, 햇빛에 의한 광전자, SEP 관통 방사)을 자연스럽게 거른다. **Fig. 15** 는 Wind/SWE FC 데이터로 이 AC 면역성을 보여준다 — SEP flux가 5 차수 변동해도 noise current는 ~10⁻¹² A 로 일정하다.

The **reduced distribution function** is what an FC reports:
$$ F(v_\|) = \int\!\!\int f(\mathbf{v})\, dv_x\, dv_y, \qquad v_\| = \sqrt{2qE/m} $$
where the integral is over velocity components transverse to the cup line-of-sight. Sweeping the modulator window through 8–16 voltage steps (Proton Tracking, PT mode) or 128 steps (Full Scan, FS mode) reconstructs F(v_∥). Detailed solar wind parameters (n_p, V_p, T_p, anisotropy, secondary beam) emerge from convolving a model VDF with the instrument response and fitting (Kasper et al. 2006 procedure).

**축소 분포 함수**가 FC가 측정하는 양이다:
$$ F(v_\|) = \int\!\!\int f(\mathbf{v})\, dv_x\, dv_y $$
LOS 횡축 속도 성분에 대한 적분이다. 변조 창을 8–16 전압 단계(Proton Tracking, PT 모드) 또는 128 단계(Full Scan, FS 모드)로 sweep 하여 F(v_∥) 를 재구성한다. 상세 태양풍 매개변수(n_p, V_p, T_p, 비등방성, 2차 빔) 는 모델 VDF 와 측정기 응답의 convolution 을 fitting 하여 얻는다(Kasper et al. 2006).

#### 3.2 Mechanical and electrical layout / 기계·전기 배치

Cross section (Fig. 17): two thin annular niobium "shield plates" at the front limit FSU exposure to sunlight; below them are a ground grid → HV modulator grid → ground grid → limiting aperture → ground grids → suppressor grid (-55 V) → four wedge-shaped collector plates. The HV modulator can swing -2 kV to +8 kV. The modulator and collector subassemblies are clamped in a sapphire cup with niobium rings; insulators are sapphire (high resistivity at high T) and grids are monolithic etched single-crystal high-purity tungsten (replacing failure-prone woven mesh). Other materials: TZM (Mo-Ti-Zr alloy) for the body; pyrolytic boron nitride avoided due to mechanical issues at high T.

Cross-section(Fig. 17): 전면의 두 얇은 환형 니오븀 shield plate 가 FSU의 햇빛 노출을 제한; 그 아래는 ground grid → HV modulator grid → ground grid → limiting aperture → ground grids → suppressor grid(−55 V) → 네 개의 쐐기형 collector plate. HV 변조기는 −2 kV 에서 +8 kV 까지 swing. modulator·collector 조립체는 sapphire cup 안에 niobium ring으로 고정; 절연재는 sapphire(고온 고저항), 격자는 etched monolithic single-crystal high-purity tungsten(과거 wire mesh 의 단선 문제 해소).

Four collectors are arranged in a 2×2 cross. The currents from the four plates yield the flow angle:
$$ \tan\theta_y \approx \frac{(I_1+I_2) - (I_3+I_4)}{(I_1+I_2)+(I_3+I_4)},\qquad \tan\theta_z \approx \frac{(I_1+I_4)-(I_2+I_3)}{\sum I} $$
Three plates suffice geometrically; a fourth provides redundancy. Expected currents: 5×10⁻¹³ to 10⁻⁷ A, system noise ~5×10⁻¹³ A → SNR up to 10⁵.

네 개 collector 가 2×2 십자형 배치이고, 4 plate 전류 비율로 흐름 각을 추정한다. 세 개로도 기하학적 충분; 네 번째는 redundancy. 예상 전류 5×10⁻¹³ ~ 10⁻⁷ A, 시스템 잡음 ~5×10⁻¹³ A → SNR 최대 10⁵.

**Thermal**: at closest approach, grid 1 reaches >1600°C, modulator housing ~1000°C, collector plates ~700°C. Thermal model + photon-input Solar Environment Simulator (SES, Cheimets et al. 2013) tests confirm signal stability vs temperature (Fig. 23).

**열적 거동**: 최근접에서 grid 1은 >1600°C, modulator housing ~1000°C, collector plate ~700°C에 도달. 열 모델과 SES(Cheimets et al. 2013) 광학 입력 테스트로 온도 대비 신호 안정성을 입증(Fig. 23).

#### 3.3 Performance: energy and angle resolution / 성능: 에너지·각 분해능

Phase-B prototype (Freeman et al. 2013, MSFC SWF) demonstrated:
- **Energy windows**: 7% wide, overlapping by 2% (Fig. 18). Each window's peak is well separated, indicating <5% energy resolution.
- **Angular response**: cos²(θ) dependence (modulator filters parallel energy only); FOV ≥30° beyond the 28° requirement (Fig. 19).
- **Calibration drift**: <0.01% per °C — temperature-induced flux error <few percent over operational range, correctable in pipeline.

Phase-B prototype(Freeman et al. 2013, MSFC SWF) 측정:
- **에너지 창**: 폭 7%, 2% 중첩(Fig. 18), 각 창 peak 분리 → <5% 에너지 분해능 입증.
- **각 응답**: cos²(θ) 의존성(modulator 가 LOS-parallel 에너지만 거름); 요구치 28° 를 넘어 ≥30° (Fig. 19).
- **드리프트**: <0.01%/°C — 작동 범위 온도 변동에 의한 flux 오차 수 % 이내, 파이프라인에서 보정.

### Part IV: Solar Probe Analyzers — SPAN (Sec. 4) / 솔라 프로브 분석기

#### 4.1 Top-hat ESA design / Top-hat 정전 분석기

SPAN inherits the **top-hat hemispherical ESA** design developed by UCB (Carlson et al. 1983; Carlson and McFadden 1998), with successful flights on FAST, STEREO STE, THEMIS, MAVEN STATIC. Particles entering the analyzer pass between two concentric hemispheres held at potentials such that only those with E/q in a narrow band reach the exit. The energy/charge selection follows
$$ \frac{E}{q} = k_\text{ESA}\, V_\text{inner} $$
with an inter-hemisphere gap **ΔR/R = 0.03** giving ~7% energy resolution. A pair of electrostatic deflectors above the main aperture redirects the cylindrically symmetric 360° planar FOV to ±60° in elevation, producing a 240°×120° FOV per sensor (after spacecraft blockage). Fig. 24 shows the optics ray-tracing; Fig. 25 shows that angular resolution remains <10° across the deflection range.

SPAN은 UCB(Carlson et al. 1983; Carlson and McFadden 1998)가 개발하고 FAST, STEREO STE, THEMIS, MAVEN STATIC 에서 비행 검증한 **top-hat 반구형 ESA** 헤리티지를 계승한다. 입자는 두 동심 반구 사이를 통과하며 좁은 E/q 대역만 출구에 도달한다.
$$ \frac{E}{q} = k_\text{ESA}\, V_\text{inner} $$
반구 간격 **ΔR/R = 0.03** → ~7% 에너지 분해능. 주 입구 위 전기 deflector 쌍이 360° 평면 FOV를 ±60° elevation 으로 방향전환 → 센서당 240°×120° FOV. Fig. 24 는 광학 추적, Fig. 25 는 deflection 범위 전체에서 각 분해능 <10° 유지를 보여준다.

#### 4.2 SPAN-A ions / SPAN-A 이온 분석기

SPAN-A's ion sensor adds a **time-of-flight section** below the analyzer: a pre-acceleration voltage (15 kV supply) accelerates ions through a thin carbon foil, generating start-pulse secondary electrons; the ion continues to a stop foil generating a stop pulse. Time-of-flight gives m/q with resolution sufficient to separate H⁺, He²⁺, He⁺, and heavier species. Stop electrons that penetrate the thick foil eliminate the dominant background source for ion mass composition sensors.

SPAN-A 이온 센서는 분석기 아래에 **time-of-flight 섹션** 을 추가한다. 사전 가속 전압(15 kV) 이 이온을 얇은 carbon foil 을 통해 가속해 start-pulse 2 차 전자를 발생시키고, 이온은 stop foil 까지 진행해 stop-pulse 를 만든다. Time-of-flight 로 m/q를 결정하여 H⁺, He²⁺, He⁺, 무거운 이온 종을 분리한다. Stop 전자가 두꺼운 foil을 관통하므로 이온 질량 조성 측정기의 주요 배경 잡음원을 제거한다.

The HV supply controlling inner hemisphere + deflector voltages is limited to 3–4 kV → headroom for 30 keV particles, but full ±60° deflection only up to ~4.5–6 keV. Higher energies measured within reduced angle range.

내반구 + deflector 의 HV 공급은 3–4 kV → 30 keV 까지 측정 가능하나 ±60° 완전 deflection은 ~4.5–6 keV 까지. 고에너지는 좁은 각도 범위 내에서 측정.

**Dynamic range**: Two-stage attenuation extends sensitivity over 0.046–0.7 AU.
1. *Mechanical attenuator*: SMA-driven "visor" with small slit — ~×10 reduction; one-time deployable cover acts as pressure relief valve.
2. *Electrostatic attenuator ("spoiler")*: voltage on lower section of outer hemisphere, ~25% of inner-hemisphere voltage, also ~×10 reduction; switchable in <2 ms within an energy sweep, allowing minor-species (alpha, heavies) measurements while spoiling proton flux.

**동적 범위**: 0.046–0.7 AU 에서 sensitivity 변동을 두 단계 감쇠로 흡수.
1. *기계 감쇠기*: SMA 구동 "visor" + 작은 슬릿 — ~×10 감쇠; 발사 후 1 회 전개되며 압력 방출 밸브 역할.
2. *정전 감쇠기 ("spoiler")*: 외반구 하부에 내반구 전압의 ~25% 인가 — ~×10 감쇠; <2 ms 내 전환 가능 → 양성자 flux를 spoil 하면서 minor species(알파, 중이온) 측정.

#### 4.3 Operating modes — Coarse / Targeted / Alternating Sweep / 동작 모드

All SPAN sensors operate in **table-driven** mode: programmable lookup tables set inner-hemisphere voltage, deflector voltages, and attenuator state at each 0.874 ms accumulation interval. The standard mode is **Alternating Sweep**:
- *Coarse sweep* — covers full phase space at low resolution (e.g., 32E×8d×16A).
- *Targeted sweep* — selects a smaller region around the peak (located using coarse-sweep counts) at full intrinsic resolution.

Total cadence: each energy/angle sweep takes 256 accumulations (0.224 s); coarse + targeted product every 0.447 s. Fig. 26 demonstrates the coarse vs targeted result for a two-component proton VDF at closest approach.

모든 SPAN 센서는 **table-driven** 모드: 0.874 ms 누적 간격마다 inner-hemisphere 전압, deflector 전압, attenuator 상태를 lookup table 이 결정한다. 표준 모드는 **Alternating Sweep**:
- *Coarse sweep* — 전체 phase space 를 저해상도로 sampling(예: 32E×8d×16A).
- *Targeted sweep* — coarse 에서 찾은 peak 주변을 본래 해상도로 sampling.

총 cadence: 각 sweep 256 누적(0.224 s); coarse+targeted 한 세트가 0.447 s 마다. Fig. 26 은 최근접 시 2-성분 양성자 VDF 의 coarse vs targeted 결과를 보여준다.

#### 4.4 SPAN-A vs SPAN-B differences / SPAN-A 대 SPAN-B 차이

SPAN-B is essentially a duplicate of the SPAN-A electron sensor mounted on the anti-ram side, with anode pattern rotated 90° to optimize coverage of the strahl direction. The two electron analyzers' FOVs combine like the seams on a baseball to cover all sky except the directly sunward portion (covered by SPC up to 2 keV) and a small ~50° spot near (110°, -50°). Key differences in SPAN-B: (1) operational heater added because at aphelion the spacecraft rotates so the high-gain antenna faces Earth, prolonging anti-ram illumination but not enough to keep SPAN-B above its minimum operating temperature when close to the Sun; (2) shares the same ESA optics design but with an orthogonal anode pattern; (3) no TOF section (electron-only sensor). SPAN-A combines an electron sensor + ion sensor (with TOF) sharing a common pedestal. The ion sensor's pre-acceleration HV is 15 kV — much higher than the 3–4 kV inner-hemisphere supply — which is what permits ion mass measurements via TOF.

SPAN-B는 SPAN-A 의 전자 센서를 거의 그대로 복사하여 반-램 면에 장착하고 anode 패턴을 90° 회전시켜 strahl 방향 커버리지를 최적화한 것이다. 두 전자 분석기의 FOV는 야구공의 솔기처럼 결합해 태양 직시 부분(SPC 가 2 keV까지 커버) 과 (110°, −50°) 부근 ~50° 스팟을 제외한 모든 하늘을 본다. SPAN-B 의 차이: (1) 원일점에서 우주선이 고이득 안테나를 지구 쪽으로 돌리는 동안 반-램 면이 길게 조명을 받지만 태양 근처에서는 최저 동작 온도 미만으로 떨어지므로 운영용 히터 추가; (2) 동일한 ESA 광학 + 직교 anode 패턴; (3) TOF 섹션 없음(전자 전용). SPAN-A 는 전자 + 이온(TOF 포함) 센서를 공통 pedestal 에 결합. 이온 센서의 사전 가속 HV는 15 kV로, 내반구 공급(3–4 kV) 보다 훨씬 높으며 TOF 질량 측정을 가능케 한다.

#### 4.5 Front-end electronics / 프런트엔드 전자공학

The electron sensors use **chevron MCP pairs** (2 plates angled to suppress ion feedback) feeding **segmented anodes** read out by a 16-channel preamplifier ASIC developed by LPP for Solar Orbiter. The ion sensor uses **Z-stack MCPs** (3 plates) for larger pulse heights to allow constant-fraction discrimination timing for TOF. Each anode's pulses are accumulated in counters and read out every 0.5 ms. SPAN-A's ion-side electronics adds an additional ASIC for time-to-digital conversion of TOF start/stop intervals. Pulse-counting electronics for the electron sensors and the new SO-heritage ASIC keep cross-talk and noise minimal across the 240°×120° FOV.

전자 센서는 **chevron MCP 쌍**(이온 피드백 억제를 위해 각도를 준 2 장) 과 **분할 anode** 를 LPP가 Solar Orbiter용으로 개발한 16채널 전치증폭기 ASIC으로 읽는다. 이온 센서는 **Z-stack MCP**(3 장) 로 큰 펄스 높이를 만들어 TOF용 constant-fraction discrimination 타이밍을 가능케 한다. 각 anode 펄스는 0.5 ms 마다 카운터에 누적·판독. SPAN-A의 이온 측 전자공학은 TOF start/stop 간격을 디지털화하는 추가 ASIC을 갖는다. 240°×120° 전 FOV에서 cross-talk와 잡음을 최소화한 디자인.

### Part V: Operations and Data Products (Sec. 5) / 운영과 데이터 제품

**SPC modes**:
- Proton Tracking (PT): 8E_∥×4I × 16 measurements / 0.874 s — main product n_p, V_p, T_p of proton core.
- Full Scan (FS): 128 E in 0.874 s — alpha + proton beam properties + e-strahl spectrum.
- Flux Angle (FA): single energy window, 128 measurements/0.874 s @ 128 Hz, 0.1° flow-angle precision — burst mode for wave-particle physics.

**SPAN data products** (Tables 5, 6): Full-3D and Targeted-3D ion and electron VDFs at survey cadence 56 s with archive cadence 0.437 s; heavy ion 4D products for mass-selected studies. Total survey volume ~6 Gbits/encounter with ~6 Gbits archive (2% return).

**SPC modes**:
- Proton Tracking (PT): 8E_∥×4I × 16 measurements / 0.874 s — n_p, V_p, T_p of proton core.
- Full Scan (FS): 128 E in 0.874 s — alpha + proton beam + e-strahl spectrum.
- Flux Angle (FA): 단일 에너지 창에서 128 Hz, 0.1° flow-angle 정밀도 — wave-particle 분석용 burst 모드.

**SPAN 데이터 제품** (Tables 5, 6): Full-3D 와 Targeted-3D 이온·전자 VDF, survey cadence 56 s + archive cadence 0.437 s; heavy ion 4D 제품. 총 survey 양 ~6 Gbits/encounter, archive ~6 Gbits (2% 반환).

**Data levels** (Table 9):
- L0: raw CCSDS packets.
- L1: instrument count rates / currents.
- L2: calibrated 3D distributions in physical units.
- L3: ground-derived solar wind moments incorporating FIELDS magnetometer; joint SPC–SPAN parameters.
- L4: derived shock list, event list, power spectra.

**데이터 레벨** (Table 9):
- L0: 원시 CCSDS 패킷.
- L1: 카운트율/전류.
- L2: 보정된 3D 분포.
- L3: FIELDS 자기장과 결합된 모멘트, 합동 SPC–SPAN 매개변수.
- L4: 충격 목록, 이벤트 목록, 파워 스펙트럼.

---

## 3. Key Takeaways / 핵심 시사점

1. **Heat shield FOV constraint dictates dual-architecture** — Because the SPP TPS blocks the anti-sun half-sky for instruments behind it, no single sensor can capture the proton VDF over the full mission. SWEAP's response is to combine a sun-staring Faraday Cup (SPC) on the heat-shield edge with a ram-side ESA (SPAN-A) and an anti-ram ESA (SPAN-B). Monte Carlo simulation shows this combination recovers proton core 100% and electron strahl 98% across the final 9.86 R☉ orbit, vs 87% / 71% for SPC alone or 24% / 62% for SPAN alone. / **열차폐 FOV 제약이 이중 구조를 강제한다** — TPS가 차폐 뒤 반(反)태양 반구를 가리므로 단일 센서로는 전체 미션 동안 양성자 VDF를 얻을 수 없다. SWEAP 의 해법은 차폐 모서리의 태양 직시 Faraday Cup(SPC), 램 면 ESA(SPAN-A), 반-램 면 ESA(SPAN-B)의 결합이다. Monte Carlo 결과 결합 시 proton core 100%, e-strahl 98% — 단독 SPC(87/71%) 또는 SPAN(24/62%) 대비 압도적이다.

2. **Faraday Cup AC synchronous detection beats SEP noise** — SPC's 1280 Hz modulator + lock-in detection rejects DC noise from photoemission, thermionic emission and SEP penetration. Wind/SWE flight data (Fig. 15) show FC noise current remains ~10⁻¹² A across SEP events with 5 orders of magnitude variation in >2 MeV electron flux. This makes SPC the only plasma sensor robust to the intense SEP environment within 0.25 AU. / **Faraday Cup AC 동기 검파는 SEP 잡음을 압도한다** — SPC 의 1280 Hz 변조 + lock-in 검파는 photoemission, thermionic emission, SEP 관통의 DC 잡음을 거른다. Wind/SWE 비행 데이터(Fig. 15)에서 SEP flux 가 5 차수 변동해도 FC noise 는 ~10⁻¹² A 로 일정. 0.25 AU 내 강한 SEP 환경에서 안정적인 유일한 plasma 측정기.

3. **Top-hat ESA + electrostatic+mechanical attenuation handle 4-orders-of-magnitude flux range** — SPAN's Carlson-1983 hemispherical optics with ΔR/R = 0.03 give ~7% E/q resolution and uniform 360° planar response. Two attenuation stages (one-shot SMA mechanical visor ×10 + spoiler-voltage electrostatic ×10) extend dynamic range from 0.046 AU to 0.7 AU without saturating the MCPs and let SPAN switch within an energy sweep to measure minor species (alphas, heavies) by spoiling the proton flux. / **Top-hat ESA + 정전+기계 감쇠기는 4 차수 flux 범위를 흡수한다** — Carlson-1983 반구 광학 + ΔR/R = 0.03 → ~7% E/q 분해능, 균일 360° 평면 응답. 일회성 SMA 기계 visor ×10 + spoiler 전압 정전 감쇠 ×10 의 두 단계로 0.046–0.7 AU 의 동적 범위를 MCP 포화 없이 처리. 한 sweep 내 전환으로 양성자 flux를 spoil 하면서 minor species(알파, 중이온) 측정.

4. **5-mechanism heating taxonomy drives 64–128 Hz cadence** — Table 2 enumerates ion cyclotron, turbulent cascade, shock-steepened, reconnection/nanoflare, and velocity-filtration heating mechanisms. Each leaves a distinct VDF signature; collectively they require 64 Hz 2D cross-correlation with E/B fields, 30 Hz turbulence-cascade detection, 16 Hz density fluctuations, 20% energy resolution for beams, and <5% resolution for kappa-distribution peaks. SWEAP's 16 Hz survey + 128 Hz burst architecture is sized to discriminate among these. / **5 가지 가열 메커니즘 분류가 64–128 Hz cadence 를 결정한다** — Table 2 는 ion cyclotron, 난류 cascade, 충격 steepening, reconnection/nanoflare, velocity filtration 의 5 메커니즘을 열거하고 각각의 관측 signature 를 정의한다. 종합 요구는 64 Hz 2D + E/B 교차상관, 30 Hz 난류 검출, 16 Hz 밀도 요동, 20% 빔 에너지 분해능, <5% kappa peak 분해능 — SWEAP의 16 Hz survey + 128 Hz burst 구조가 이를 정확히 충족한다.

5. **Reduced Distribution Function inversion enables flow-angle determination** — SPC measures the 1-D RDF F(v_∥) by sweeping the modulator window. Convolving a model VDF with the instrument response and fitting (Kasper et al. 2006 procedure) yields n_p, V_p, T_p, anisotropy and secondary beam properties even without 3D coverage. The four-quadrant collector geometry adds two transverse flow-angle components from current ratios. / **RDF 역해석으로 흐름 각 결정** — SPC는 변조 창 sweep으로 1-D RDF F(v_∥)를 측정하고, 모델 VDF와 측정기 응답의 convolution을 fitting(Kasper et al. 2006)하여 n_p, V_p, T_p, 비등방성, 2차 빔까지 3D 커버 없이 도출한다. 4-quadrant collector는 전류 비율로 2 개 횡축 흐름 각 성분을 추가 제공한다.

6. **Aberration management is essential at perihelion** — At 9.86 R☉ the orbital velocity reaches ~200 km/s, comparable to the slow solar wind, tilting the apparent flow toward ram by ~25°. SPC's ±28° half-cone FOV and SPAN-A's ram orientation are sized exactly for this aberration; without both, the proton core would drift outside any single FOV during the encounter. / **근일점에서 aberration 관리가 핵심** — 9.86 R☉ 에서 궤도 속도 ~200 km/s 는 저속 태양풍과 동급이며 보이는 흐름을 ram 쪽으로 ~25° 기울인다. SPC ±28° 반각 FOV와 SPAN-A ram 배치가 정확히 이 aberration 을 위해 설계 되었다.

7. **Onboard moments + flash buffer + selective downlink solve telemetry crisis** — Telemetry from SPP is severely constrained, but high-cadence wave-particle physics requires 128 Hz data. SWEAP's solution: SWEM computes onboard moments and stores all full-resolution data in 80 GB flash for the entire encounter; survey + moments downlink quickly; scientists then select interesting periods (shocks, switchbacks, reconnection) for full-resolution archive return. ~2% archive return suffices for science closure. / **온보드 모멘트 + flash buffer + 선택 다운로드로 텔레메트리 위기 해결** — SPP 텔레메트리는 매우 제한적이나 128 Hz wave-particle physics 가 필요하다. SWEM 이 온보드 모멘트를 계산하고 80 GB flash에 encounter 전체 풀 해상도 데이터를 저장; survey+moments 는 빠르게 다운; 과학자가 흥미로운 구간(충격, switchback, reconnection)을 선택해 archive return → ~2% 반환으로 과학 목표 달성.

8. **Materials engineering sets the closest approach** — Beyond optics and electronics, the binding constraint is materials surviving >1600°C with sun-induced radiation and ion sputtering. SPC's choices: monolithic single-crystal high-purity tungsten grids (replacing wire mesh), TZM body, sapphire insulators, niobium spacers, laser-welded HV joints, ground-tested in MSFC SWF and SAO SES. These thermal+materials decisions are what make 9.86 R☉ feasible. / **소재 공학이 최근접 거리를 결정** — 광학·전자공학을 넘어, >1600°C + 햇빛·이온 스퍼터링 환경을 견디는 소재 선택이 결정적이다. SPC: monolithic single-crystal high-purity tungsten 격자(wire mesh 대체), TZM 본체, sapphire 절연재, niobium 스페이서, laser-welded HV 결합 — MSFC SWF, SAO SES 에서 지상 검증. 이 결정이 9.86 R☉ 를 가능케 했다.

---

## 4. Mathematical Summary / 수학적 요약

### A. Reduced distribution function and Faraday cup current / RDF와 FC 전류

For a 3-D plasma VDF f(**v**), the line-of-sight reduced distribution function is
$$ F(v_\|) = \int\!\!\int f(v_\|, v_x, v_y)\, dv_x\, dv_y $$
where v_∥ is the velocity component along the cup axis. The instantaneous current onto a collector area A from particles with velocity in [v_∥, v_∥ + dv_∥] is
$$ dI = q n A v_\| f(\mathbf{v})\, d^3v $$
With the modulator filtering only the parallel component,
$$ I_\text{plate} = q A \int_{v_\text{low}}^{v_\text{high}} v_\| F(v_\|)\, dv_\| $$
Sweeping the window [v_low, v_high] reconstructs F(v_∥). When the modulator oscillates V(t) = V_DC + V_AC sin(ω_mod t), the AC component of I is proportional to ∂F/∂E and a synchronous detector at ω_mod isolates it from DC noise.

3-D plasma VDF f(**v**) 에 대해 line-of-sight RDF는 위 식. modulator가 LOS-parallel 에너지만 거르므로 plate 전류는 modulator 창 [v_low, v_high] 내 적분이다. V(t) = V_DC + V_AC sin(ω_mod t) 의 AC 성분이 ∂F/∂E 에 비례하고, ω_mod 동기 검파가 이를 추출한다.

### B. Plasma moments and SPC fitting / 모멘트 적분

Standard moments of f(v):
$$ n = \int f\, d^3v,\quad \mathbf{V} = \frac{1}{n}\int \mathbf{v} f\, d^3v $$
$$ T_{ij} = \frac{m}{n k_B}\int (v_i - V_i)(v_j - V_j) f\, d^3v $$
For an anisotropic Maxwellian aligned with **B**:
$$ f(v_\|, v_\perp) = \frac{n m^{3/2}}{(2\pi k_B)^{3/2} T_\| ^{1/2} T_\perp}\, \exp\!\left[-\frac{m(v_\| - V_\|)^2}{2k_B T_\|} - \frac{m v_\perp^2}{2k_B T_\perp}\right] $$
SPC fitting convolves this model (or a two-component sum core+beam) with the modulator-window response and four-collector angular response, and fits n, **V**, T_∥, T_⊥, secondary-beam parameters by χ² minimization (Kasper et al. 2006).

표준 모멘트와 anisotropic Maxwellian 모델식. SPC fitting은 modulator 창 응답과 4-collector 각 응답을 모델 VDF (또는 core+beam 2 성분) 와 convolve 하고 χ² 최소화로 매개변수를 결정한다(Kasper et al. 2006).

### C. Top-hat ESA energy selection / Top-hat 에너지 선택

For two concentric hemispheres at radii R₁ (inner) < R₂ (outer) with potential difference V, particles with E/q satisfying
$$ \frac{E}{q} = \frac{V}{2}\,\frac{R_1 R_2}{R_2 - R_1}\,\frac{1}{R_\text{mean}} \approx \frac{V}{\Delta R / R} $$
pass through. SPAN uses ΔR/R = 0.03, yielding ~7% E/q resolution.

내반구 R₁ < 외반구 R₂, 전위차 V 인 동심 반구에서 E/q ≈ V / (ΔR/R) 의 입자가 통과. SPAN ΔR/R = 0.03 → ~7% 분해능.

### D. SPAN VDF accumulation / SPAN VDF 누적

SPAN bins counts in a 4D (E × deflection × anode × time) array. Background-subtracted counts C_ijkl convert to phase-space density via
$$ f(\mathbf{v}) = \frac{C_{ijkl}}{G_{ijkl}\, \Delta t\, v^4 / (2\, E/m)} $$
where G_ijkl is the geometric factor (constant ~0.0015 cm² sr eV/eV from Fig. 25 black curve), Δt the accumulation time (0.874 ms), and v² = 2E/m (with the factor v⁴ from the Jacobian of (E, Ω) → v-space integration).

SPAN 은 4D (E×deflection×anode×t) 어레이에 카운트를 누적. 배경 차감 후 위 식으로 phase-space density 환산. G는 ~0.0015 cm² sr eV/eV (Fig. 25 흑색), Δt = 0.874 ms, Jacobian v⁴ 인자 포함.

### E. Aberration / Aberration

In the spacecraft frame, the observed solar wind velocity is
$$ \mathbf{V}_\text{obs} = \mathbf{V}_\text{SW} - \mathbf{V}_\text{SC} $$
At 9.86 R☉, V_SW ≈ 200–500 km/s (radial) and V_SC ≈ 200 km/s (tangential), so the apparent flow tilts by an angle θ_aberr = atan(V_SC / V_SW) ~ 22–45°. SPC's ±28° half-cone FOV plus SPAN-A's 240°×120° ram FOV are sized to maintain proton-core coverage despite this tilt.

우주선 좌표계에서 V_obs = V_SW − V_SC. 9.86 R☉ 에서 V_SC ≈ 200 km/s 가 보이는 흐름을 ~22–45° 기울인다. SPC ±28° + SPAN-A 240°×120° 가 이 기울기를 흡수하도록 설계되었다.

### F. Worked example: SPC at 9.86 R☉ / 9.86 R☉ SPC 계산 예

Assume the proton core: n_p = 1800 cm⁻³, V_p = 285 km/s (radial), T_p = 41 eV → v_th = √(2kT/m) ≈ 88 km/s. The proton kinetic energy E = ½ m_p V_p² ≈ 425 eV. Spacecraft velocity at 9.86 R☉ ≈ 200 km/s tangential.

In the SC frame, the flow vector is **V**_obs = (-285, 0, 0) - (0, -200, 0) = (-285, +200, 0) km/s, with magnitude 348 km/s and angle 35° from radial. Energy in SC frame: E_obs = ½ m_p (348 km/s)² ≈ 632 eV. SPC's modulator window must include this.

The flux F = n_p V_obs ≈ 1800·(348×10⁵) cm⁻²s⁻¹ ≈ 6.3×10¹⁰. With SPC limiting aperture A_lim ~ few cm² (sized for 5×10⁻¹³ A minimum at 0.25 AU), proton current onto a collector is order
$$ I_\text{plate} \sim q F A \cdot \text{frac in window} \sim 1.6\times10^{-19} \cdot 6.3\times10^{10} \cdot 1 \cdot 0.07 \sim 7\times10^{-10}~\text{A} $$
well within SPC's 10⁻⁷ A saturation and far above the 5×10⁻¹³ A noise — SNR ~ 10³–10⁴.

가정: 양성자 core n_p = 1800 cm⁻³, V_p = 285 km/s (radial), T_p = 41 eV → v_th ≈ 88 km/s, E ≈ 425 eV. 9.86 R☉ 우주선 속도 ~200 km/s tangential. SC 좌표계에서 V_obs = (−285, +200, 0) km/s, |V| = 348 km/s, 방사 방향에서 35° 기울어짐, E_obs ≈ 632 eV. SPC 변조 창이 이 에너지를 포함해야 한다. flux F ≈ 6.3×10¹⁰ cm⁻²s⁻¹, plate 전류 ~7×10⁻¹⁰ A — 포화 한계(10⁻⁷ A) 내, 잡음 5×10⁻¹³ A 대비 SNR ~10³–10⁴.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1958 ─ Parker: solar wind theory
1960 ─ Gringauz: first FC particle detection
1962 ─ Mariner 2 FC: first solar wind in-situ
1971 ─ Vasyliunas: kappa distributions / VDF moment formalism
1974/76 ─ Helios 1, 2: 0.29 AU FC measurements (Pilipp 1987 e-VDF)
1983 ─ Carlson et al.: top-hat hemispherical ESA design
1995 ─ Wind launch (SWE Faraday Cup, Ogilvie et al. 1995)
1996–2008 ─ SWE FC papers (Kasper 2006 fit method, Maruca, Kasper 2008 cyclotron)
2008 ─ MAVEN STATIC heritage (McFadden 2015): TOF + ESA
2013 Oct ─ SWEAP PDR (snapshot of this paper)
2015 Oct ─ Kasper et al. 2016 (this paper) published online
2018 Aug ─ Parker Solar Probe launch, first encounter Nov 2018
2019 ─ Bale et al., Kasper et al.: switchbacks (Nature)
2021 Apr ─ PSP first sub-Alfvénic crossing (Kasper PRL 2021)
2024 Dec ─ PSP final perihelion 9.86 R☉
```

This paper is the linchpin between the heritage (Carlson 1983 ESA, Wind 1995 FC) and the discoveries (switchbacks 2019, sub-Alfvénic plasma 2021, ion heating). Every published PSP plasma result cites Kasper et al. (2016) as the calibration baseline.

이 논문은 헤리티지(Carlson 1983 ESA, Wind 1995 FC)와 발견(2019 switchback, 2021 sub-Alfvénic plasma, ion heating) 사이의 연결고리이다. 모든 PSP plasma 논문이 Kasper et al. (2016) 을 보정 기준으로 인용한다.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Parker (1958, *ApJ* 128, 664), "Dynamics of the interplanetary gas and magnetic fields" | Theoretical prediction of supersonic solar wind that SWEAP measures at its source | Foundational: SPP's mission to test Parker's theory inside the Alfvén surface / Parker 이론을 Alfvén surface 안쪽에서 직접 검증 |
| Carlson et al. (1983) — Top-hat ESA design / Carlson and McFadden (1998) — ESA optics simulations | SPAN's hemispherical analyzer optics directly inherits from this design family (FAST, STEREO STE, THEMIS, MAVEN STATIC) | Direct heritage: SPAN ΔR/R = 0.03, 240°×120° FOV are 30-year-old optimized solutions / 30년 헤리티지의 직계 계승 |
| Ogilvie et al. (1995, *SSR* 71) — Wind/SWE | Wind FC is the immediate ancestor of SPC; AC modulation, sapphire insulators, niobium grids all proven on Wind | Operational heritage: SPC noise demonstration (Fig. 15) uses Wind FC data / SPC 잡음 demonstration이 Wind FC 데이터로 입증됨 |
| Kasper et al. (2006, *JGR* 111, A03105) — FC physics-based test method | Defines the SPC fitting procedure (model VDF × instrument response → χ² fit for n, V, T, anisotropy) | Measurement framework: every L2/L3 SPC product runs this algorithm / 모든 L2/L3 SPC 제품의 핵심 알고리즘 |
| Pilipp et al. (1987, *JGR* 92, 1075) — Helios e-VDF | Empirical core/halo/strahl decomposition that SPAN reproduces but at coronal distances | Comparison baseline: Fig. 5 e-strahl evolution comes from Helios / e-strahl 진화의 비교 기준 |
| McComas et al. (2014) — ISIS instrument suite | ISIS measures energetic particles starting where SWEAP ends (≥30 keV); together they cover thermal–suprathermal–energetic | Cross-instrument: SWEAP + ISIS coordinated for shock acceleration studies / 충격 가속 연구의 공동 측정기 |
| Bale et al. (2016, *SSR* this issue) — FIELDS | Magnetometer + EM waves; SWEAP relays particle counts at high speed for wave-particle correlations | Direct hardware link: SWEAP-FIELDS data interface for kinetic physics / Kinetic physics를 위한 직접 데이터 링크 |
| Kasper et al. (2008, *PRL* 101) — Alfvén-cyclotron resonance evidence | Wind/SWE ion data showing T_α/T_p tracks cyclotron resonance — predicted to be much stronger near Sun | Scientific motivation: Table 2 ion-cyclotron measurement requirement comes directly from this work / Table 2 의 ion cyclotron 측정 요구가 이 연구에서 유도 |
| Bale et al. (2019, *Nature*); Kasper et al. (2019, *Nature*) — switchbacks | First PSP results enabled by SWEAP at 35 R☉ | Forward citation: this paper enables those discoveries / 본 논문이 이 발견들을 가능케 함 |

---

## 7. References / 참고문헌

- **Primary**: Kasper, J.C., Abiad, R., Austin, G., et al. (2016). "Solar Wind Electrons Alphas and Protons (SWEAP) Investigation: Design of the Solar Wind and Coronal Plasma Instrument Suite for Solar Probe Plus". *Space Science Reviews*, 204, 131–186. DOI: 10.1007/s11214-015-0206-3
- Bale, S.D., et al. (2016). "The FIELDS Instrument Suite for Solar Probe Plus". *Space Science Reviews*, this issue.
- Bale, S.D., Kasper, J.C., Howes, G.G., Quataert, E., Salem, C., Sundkvist, D. (2009). "Magnetic fluctuation power near proton temperature anisotropy instability thresholds in the solar wind". *Phys. Rev. Lett.* 103, 211101.
- Carlson, C.W., Curtis, D.W., Paschmann, G., Michael, W. (1983). "An instrument for rapidly measuring plasma distribution functions with high resolution". *Adv. Space Res.* 2, 67–70.
- Carlson, C.W., McFadden, J.P. (1998). "Computer simulation in designing electrostatic optics for space plasma experiments". *Geophys. Monograph* 102.
- Case, A.W., et al. (2013). "Designing a sun-pointing Faraday cup for Solar Probe Plus". *AIP Conf. Proc.* 1539, 458–461.
- Cheimets, P., et al. (2013). "The design, development, and implementation of a solar environmental simulator (SES) for the SAO Faraday Cup". *Proc. SPIE* 8862, 88620N.
- Fox, N.J., et al. (2015). "The Solar Probe Plus Mission: Humanity's First Visit to Our Star". *Space Sci. Rev.* this issue.
- Freeman, M., et al. (2013). "Technology development for the Solar Probe Plus Faraday Cup". *Proc. SPIE* 8862.
- Gringauz, K.I. (1960). "Results of observations of charged particles observed out to R = 100,000 km, with the aid of charged-particle traps on Soviet space rockets". *Sov. Astron.* 4, 680.
- Halekas, J.S., et al. (2013). "The Solar Wind Ion Analyzer for MAVEN". *Space Sci. Rev.* DOI:10.1007/s11214-013-0029-z
- Kasper, J.C., et al. (2006). "Physics-based tests to identify the accuracy of solar wind ion measurements: a case study with the Wind Faraday cups". *J. Geophys. Res.* 111, A03105.
- Kasper, J.C., et al. (2008). "Hot solar-wind helium: direct evidence for local heating by Alfvén-cyclotron dissipation". *Phys. Rev. Lett.* 101, 261103.
- Kasper, J.C., et al. (2013). "Sensitive test for ion-cyclotron resonant heating in the solar wind". *Phys. Rev. Lett.* 110, 091102.
- Korreck, K.E., et al. (2014). "Solar Wind Electrons Alphas and Protons (SWEAP) science operations center initial design and implementation". *Proc. SPIE* 9149.
- Maruca, B.A., Bale, S.D., Sorriso-Valvo, L., Kasper, J.C., Stevens, M.L. (2013). "Collisional thermalization of hydrogen and helium in solar-wind plasma". *Phys. Rev. Lett.* 111, 241101.
- McComas, D.J., et al. (2014). "Integrated Science Investigation of the Sun (ISIS): design of the energetic particle investigation". *Space Sci. Rev.* DOI:10.1007/s11214-014-0059-1
- McFadden, J.P., et al. (2008). "The THEMIS ESA plasma instrument and in-flight calibration". *Space Sci. Rev.* 141, 302.
- McFadden, J.P., et al. (2015). "MAVEN SupraThermal And Thermal Ion Composition (STATIC) instrument". *Space Sci. Rev.*
- Ogilvie, K.W., et al. (1993). "High-velocity tails on the velocity distribution of solar wind ions". *J. Geophys. Res.* 98, 3611.
- Pilipp, W.G., et al. (1987). "Characteristics of electron velocity distribution functions in the solar wind derived from the Helios plasma experiment". *J. Geophys. Res.* 92, 1075.
- Vourlidas, A., et al. (2015). "The Wide-Field Imager for Solar Probe Plus (WISPR)". *Space Sci. Rev.* DOI:10.1007/s11214-014-0114-y
