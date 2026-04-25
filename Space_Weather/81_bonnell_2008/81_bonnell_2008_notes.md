---
title: "The Electric Field Instrument (EFI) for THEMIS"
authors: Bonnell, Mozer, Delory, Hull, Ergun, Cully, Angelopoulos, Harvey
year: 2008
journal: "Space Science Reviews"
doi: "10.1007/s11214-008-9469-2"
topic: Space_Weather
tags: [THEMIS, EFI, electric_field, double_probe, magnetosphere, instrumentation]
status: completed
date_started: 2026-04-25
date_completed: 2026-04-25
---

# 81. The Electric Field Instrument (EFI) for THEMIS / THEMIS 전기장 측정 기기 (EFI)

---

## 1. Core Contribution / 핵심 기여

이 논문은 NASA THEMIS 5기 동일 위성에 탑재된 3축 전기장 기기 (Electric Field Instrument, EFI)의 측정 요구사항, 기계·전기 설계, 데이터 산출물, 그리고 첫 1년 궤도 운용 성능을 종합 정리한다. EFI는 spin plane에 4개의 wire boom (sphere sensor 1-4, 49.6 m × 40.4 m tip-to-tip), spin axis에 2개의 axial stacer boom (whip sensor 5-6, 6.93 m tip-to-tip)으로 6개의 sensor를 배치하고, 각 sensor의 floating potential ($V_n$, $n=1\ldots6$)과 차동 전위 ($E_{12}, E_{34}, E_{56}$)를 동시에 측정하여 DC부터 8 kHz까지의 파형/스펙트럼 E-field와 sub-spin-period plasma density (spacecraft floating potential 통해) 그리고 100-400 kHz 단일 통합 채널 (HF, AKR 전용)을 제공한다. 모든 5 위성에서 2007년 deploy 이래 어떤 mechanical/electrical failure 없이 운용 중이며, substorm onset (8-10 RE plasma sheet에서 1 mV/m / 10 % 정확도), dawn-dusk convection (18-30 RE), substorm onset 시 1-60 Hz 3D wave field, radiation belt에서 local electron cyclotron freq까지의 wave field 등 THEMIS의 mission-level 과학 요구사항을 충족하도록 설계되었다.

This paper is the comprehensive instrument-description article for the three-axis Electric Field Instrument (EFI) flown on the five identical NASA THEMIS spacecraft. EFI deploys six sensors: four spheres on radial spin-plane wire booms (sensors 1-4, baselines 49.6 m and 40.4 m tip-to-tip) and two whips on axial stacer booms along the spin axis (sensors 5-6, 6.93 m tip-to-tip). Both the individual floating potentials $V_n$ ($n=1\ldots6$) and the differential potentials $E_{mn}$ ($mn = 12, 34, 56$) are measured simultaneously, yielding waveform and spectral E-field from DC to 8 kHz, sub-spin-period plasma density via spacecraft floating potential, and a single integrated 100-400 kHz HF channel for AKR monitoring. All five EFIs have operated without any mechanical or electrical failure since deployment in early 2007. The design satisfies THEMIS's quantitative mission requirements: 2D spin-plane E to 1 mV/m or 10% at substorm onset (8-10 RE), dawn-dusk E to 1 mV/m at 18-30 RE, 3D wave E from 1-60 Hz at substorm onset, and 3D wave E up to local electron cyclotron frequency in the radiation belts. The paper additionally documents the on-orbit bias optimization, Sensor Diagnostic Tests (SDT), boom-shorting cross-calibration against $-\mathbf{V}_i\times\mathbf{B}$, and the THEMIS Electrostatic Cleanliness specification — all establishing the baseline for a generation of follow-on missions (MMS, Parker, Solar Orbiter).

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction and Measurement Requirements (Section 1, pp. 304-305) / 도입 및 측정 요구사항

EFI는 THEMIS 위성의 ambient vector E-field 3성분을 측정한다. Waveform: DC-4 kHz (AC-coupled differential은 10 Hz-8 kHz까지). Spectral: 동일 범위 + 100-400 kHz integrated band. On-board spin-fit E-field가 spin plane과 axial 모두 산출되며, on-board spacecraft floating potential 추정값은 particle moment 보정과 burst trigger 평가에 사용된다.

EFI measures the three components of the ambient vector E-field. Waveform from DC up to 4 kHz (AC-coupled differential up to 8 kHz), spectral over the same range plus an integrated 100-400 kHz channel. On-board spin-fit E-field is produced for both spin plane and axial measurements; on-board $V_{sc}$ estimates feed the particle moment computation and burst trigger logic.

**6 sensors**: 차동 측정은 $E_{mn} = V_m - V_n$, $mn \in \{(1,2),(3,4),(5,6)\}$. spin plane sensor는 우주선 spin coordinate system에 align되어 있고 (spin은 z축), 이중 한 쌍은 다른 쌍보다 약간 더 길게 (49.6 vs 40.4 m) deploy되어 DC와 AC E-field 측정의 systematic error 결정에 활용된다.

The differential sensors are $E_{mn} = V_m - V_n$ with $mn \in \{(1,2),(3,4),(5,6)\}$. Spin-plane sensors are aligned in the spacecraft spin coordinate system (z is spin axis); the long-pair (49.6 m) versus short-pair (40.4 m) deployment asymmetry on a single spacecraft enables determination of certain systematic errors and electrostatic-wake diagnostics.

**Top-level science requirements** (page 304):
- 2D spin-plane E at 8-10 RE in plasma sheet/PSBL: 1 mV/m or 10% — substorm onset E-field associated with flows, flow diversions, interchange instabilities at inner edge of plasma sheet.
- Dawn-dusk E at 18-30 RE: 1 mV/m or 10% — up-tail and down-tail flows during substorm onset.
- 3D wave E at 1-60 Hz at 8-10 RE — current disruption and interchange-like instabilities.
- 3D wave E up to local $f_{ce}$ in radiation belts — whistler-mode hiss, chorus, electron cyclotron fundamentals/harmonics for energization, scattering, loss of relativistic electrons.

또한 일반적 환경 (radiation, thermal, shock, vibration, acoustic), 자원 (mass, power), 호환성 (EMI/EMC, DC magnetics, ESC) 사양도 충족. EFI 팀은 ESC 사양 자체를 mission에 부과했다. The instrument also complies with environmental, resource, and compatibility specifications; notably the EFI team imposed the ESC specification on the mission itself.

### Part II: Mechanical Design (Section 2.1, pp. 306-308) / 기계 설계

#### 2.1.1 Spin Plane Booms (SPB) — 4개 / four units

각 SPB는 sensor + preamp 어셈블리를 ~22 m custom cable 끝에 부착한 형태. **Sphere**: 8 cm 직경 graphite-coated (Acheson Colloid DAG-213, baked-on epoxy-resin matrix) Al sphere, 내부 spring-loaded keyreel mechanism, 3 m × 0.2 mm (0.009″) diameter 7×7 braided 302/304 stainless-steel **fine wire**가 sphere에서 cable로 연결.

Each SPB: sensor + preamp affixed to the end of a ~22-m custom cable. Sphere is an 8-cm-diameter graphite-coated (DAG-213 over epoxy resin) aluminum sphere; inside is a spring-loaded keyreel mechanism and a 3-m × 0.009-inch 7×7 braided 302/304 stainless-steel **fine wire** between sphere and cable. Graphite coating ensures uniform work function (uniform photoemission across the sunlit/shadowed hemispheres).

**Preamp enclosure**: 2.3 cm 직경 × 3.7 cm 길이 dual-tapered cylinder; 안에 preamp electronic board, USHER/GUARD photoelectron control surfaces.

**Preamp enclosure**: 2.3 cm × 3.7 cm dual-tapered cylinder containing the preamp board and acting as USHER and GUARD photoelectron-control surfaces.

**Custom cable** (W.L. Gore): 안쪽 coax (AWG-36 single conductor with Al-mylar overwrap shield) + 8개 insulated single conductor (AWG-36) + kevlar load-bearing braid + helical aluminized mylar shield + 외측 silver-plated copper braid. ~2.5 mm 직경 (~0.100″). 외측 braid의 distal 3 m는 proximal과 전기적으로 isolated되어 있어 BEB로부터 driven될 수 있다 → DBraid (Distal Braid). 한 쌍이 다른 쌍보다 길게 deploy됨 (49.6 vs 40.4 m). Effective sensor separation은 electrostatic effect 때문에 physical separation보다 작다.

The cable's outer braid is split: the **distal 3 m** is electrically isolated from the proximal portion and can be driven (controlled voltage bias up to ±40 V relative to AGND or to a low-pass-filtered $V_1$/$V_3$). This provides leverage to control wire-boom shorting and electrostatic wake effects (similar to "bootstrapped braid" of FAST and Cluster-II EFW). Total deployed length up to 25 m (s/c centerline to sphere center). One SPB pair is deployed slightly longer than the other to allow determination of systematic error.

Deploy mechanism: spool, motor and meter wheel pulling cable at calibrated rate, click counter (resolution < 4.8 cm), wire-tension switch and shear-pin end-of-deploy stop, snout-and-doors caging spheres for launch with low-impedance (few-hundred-Ω) contact for self-test. Snout doors held by plunger/spring released by TiNi wire actuator (~2 A at 36 V, 0.1-0.2 A at 36 V to operate motor). SPB motor magnets shielded with three-layer mu-metal → dipole moment < 10 mA·m² (worst case 20), corresponding to ~0.1-0.2 nT at FGM.

#### 2.1.2 Axial Booms (AXB) — 2개 / two units

각 AXB는 ~2.5 m stacer boom + Deploy Assist Device (DAD). **Sensor**: 0.75 m 길이 tapered (4.8-7.0 mm) graphite-coated (DAG-213) Elgiloy whip stacer with 1.6 cm 직경 4.6 cm 길이 can. Preamp enclosure는 SPB와 유사하며 USHER/GUARD 역할. ~2 cm 직경 graphite-coated (DAG-214) Elgiloy main stacer.

Each AXB: ~2.5 m main stacer + Deploy Assist Device (DAD) + 0.75 m tapered Elgiloy **whip stacer** (4.8-7.0 mm taper, 1.6 cm × 4.6 cm can outboard end), preamp enclosure (USHER/GUARD), and ~2 cm graphite-coated DAG-214 Elgiloy main stacer. Two AXB assemblies mounted back-to-back inside a 102-mm diameter graphite-composite **central tube** that is also the spacecraft's primary structural member, S-band antenna mount, and reaction-control system mount.

**Final deployed lengths** (Table 1):
- X (V1, V2, E12): 49.6 m tip-to-tip
- Y (V3, V4, E34): 40.4 m tip-to-tip
- Z (V5, V6, E56): 6.93 m tip-to-tip (0.76 m whip; ~6.2 m whip center-to-center)

### Part III: Electrical Design (Section 2.2, pp. 309-313) / 전기 설계

EFI는 3축 각각에 대해 standard instrumentation amplifier 디자인이다. 각 sensor electrode는 high-input-impedance (~$10^{12}$ Ω) low-noise unity-gain preamp 입력에 연결. Preamp는 buffered sensor potential을 cable을 통해 BEB와 DFB (둘 다 IDPU 내) 로 driven. 각 preamp는 separate floating power supply에서 구동되며, 그 floating ground potential (FGND)는 preamp 출력의 low-pass-filtered version. FGND 동적 범위는 ±100 V (vs s/c ground), preamp 자체는 sensor potential vs FGND ±10 V.

EFI is a standard 3-axis instrumentation-amplifier design. Each sensor connects to a ~$10^{12}$ Ω high-input-impedance, low-noise unity-gain preamp (OP-15, 100-kRad(Si) rad-hard, few-pA input bias). The preamp drives the buffered sensor potential into the BEB and DFB. **Each preamp runs on a separate floating supply** whose FGND is derived from a low-pass-filtered version of the preamp output, so FGND tracks the (highly variable) sensor potential. FGND has ±100 V dynamic range vs spacecraft ground, while preamp supplies provide ±10 V vs floating ground. This topology lets low-bias-current op-amps see only ±10 V locally while the sensor floats over ±100 V vs the spacecraft.

#### 2.2.1 Sensors and Preamps

**Frequency response** (Fig. 7-8): SPB and AXB total response is the product of upstream (sheath impedance + effective input impedance) and downstream (output Z, cable, analog electronics) factors. Black dashed total response lines show roll-offs and a mid-band plateau ≈ 0.4-0.7 set by the divider between sheath impedance (~tens of MΩ when biased) and the input impedance of the follower (with 100 kΩ ESD-protect resistor in series, 10-pF capacitor). Variation with operating point is ~20% gain and 5-10° phase, sufficient for accurate polarization, Poynting flux, and mode determination. **No on-board compensation** for frequency response; corrections applied either in frequency domain (spectral) or time domain (waveform) on the ground.

Predicted gain vs frequency (Fig. 7) shows: low-frequency divider plateau dependent on sheath resistance ($R_s$ = 20, 50, 100 MΩ), DC-end roll-off due to FGND filter, and high-frequency roll-off from cable capacitance. Phase: < 1 degree absolute knowledge at SCM/EFI signals (better than 1°), enabling Poynting-flux measurements in concert with the search-coil magnetometer (SCM, Roux et al. 2008).

**Thermal qualification**: preamp survives -120°C (3-h eclipse) to +70°C (full sunlight). Six-cycle qualification (24 hot-cold-hot thermal-vacuum cycles on 8 ETU; five flight + spare). No units failed; <5% change in all parameters. **44 kRad(Si)** total dose to OP-15 (RDM = 2; OP-15 qualified to 100 kRad).

#### 2.2.2 Boom Electronics Board (BEB)

Single centralized BEB in IDPU (mass/volume budget driven). Drives FGND, BIAS, USHER, GUARD, DBRAID for all 6 sensors. **AD5544 16-bit DACs**: BIAS dynamic range ±528 nA/sensor; USHER, GUARD, DBraid ±40 V relative to sensor potential. Bias signals are referenced to a passive single-pole ~400-Hz low-pass-filtered sensor potential — allowing stable DC biasing in presence of large low-frequency excursions of sensor-to-spacecraft potential. Only 11-12 of 16 DAC bits needed for 0.1% precision; remaining 4-5 bits are margin against degradation.

**Typical bias settings (Table 2)**:
| Surface | SPB | AXB |
|---|---|---|
| Sensor | ≈ 180 nA (120 nA initially) | ≈ 180 nA (120 nA initially) |
| USHER | ≈ +4 V | ≈ +4 V |
| GUARD | ≈ +4 V | ≈ +4 V |
| DBraid | ≈ 0 V, V1 reference, or grounded | n/a (no DBraid on AXB) |

ACTEST self-test capability: 5 V_pp, 128-Hz square wave injected on stowed sensors via 10 MΩ / 7 MΩ grounding resistors, providing ±5 V DC functional test and AC test by toggling ACTEST lines.

### Part IV: Data Quantities (Section 2.3, pp. 314-318) / 데이터 산출물

**Table 3 data products**:
| Product | Range | Bits | Resolution | Sampling rates |
|---|---|---|---|---|
| V1...V6 | ±105 V (±100 V supply) | 16 | 3.2 mV/ADC | 2-8192 samp/s |
| EDC12 (49.6 m) | ±300 mV/m | 16 | 9.2 µV/m/ADC | 2-8192 samp/s |
| EAC12 (49.6 m) | ±51 mV/m | 16 | 1.6 µV/m/ADC | up to 16384 |
| EDC34 (40.4 m) | ±370 mV/m | 16 | 11 µV/m/ADC | 2-8192 |
| EAC34 (40.4 m) | ±63 mV/m | 16 | 1.9 µV/m/ADC | up to 16384 |
| EDC56 (6.2 m) | ±2.7 V/m | 16 | 81 µV/m/ADC | 2-8192 |
| EAC56 (6.2 m) | ±450 mV/m | 16 | 14 µV/m/ADC | up to 16384 |
| HF | 4 µV/m to 12 mV/m | 8 | 0.01 decade | 2-8192 |
| Spin fit $E_{xy}$ | as EDC12 or EDC34 | 16-bit float | 1° angle | 1 vec/spin |
| Spin avg $E_z$ | as EDC56 | 16-bit float | as EDC56 | 1/spin |
| Spacecraft potential | — | 16 | — | 1/spin |

DFB common-mode rejection: DC CMRR > 80 dB (better than 0.1 mV/V common-mode input), AC CMRR > 40 dB (10 mV/V common-mode input). Note that displacement of the **electrical center** of the spacecraft from the geometric center is an additional source of common-mode error — insignificant for spin-plane sensors but **significant for axial measurements** (see Sec 4.6).

**HF channel**: 100-400 kHz auroral kilometric radiation band; logarithmic amplitude over ~3.5 decades; integrated power (no spectral resolution). Used for AKR substorm monitoring and as a burst trigger.

**Spin fit $E_{xy}$**: 128-sample/s data from EDC12 (or EDC34) fit to $A + B\sin\psi + C\cos\psi$ (where $\psi$ is spin phase relative to Sun pulse) using iterative outlier subtraction. Spin avg $E_z$: 128-sample/s data from EDC56 fit to constant with iterative outlier subtraction.

**Spacecraft floating potential**: average of several samples of one pair of spin-plane sensors (V1+V2 or V3+V4) or spot sample of same; scaled and offset by two adjustable parameters; provided to ground for particle moment correction.

**On-orbit configurations** (Table 4): Slow Survey (full orbit, 0/2/4 samp/s for $V_n$, $E_{nm}$), Fast Survey (12 hr, 2/4 samp/s), Particle Burst (16/128 samp/s), Wave Burst (8192 samp/s).

**The "_dot0" data type** at Level-2: replaces axial E with $E_{axial} = -((B_x/B_z) E_x + (B_y/B_z) E_y)$ from $\mathbf{E}\cdot\mathbf{B}=0$. Useful when perpendicular E dominates and B is not too close to spin plane (otherwise error grows ~10:1 for B angles below ~6°).

**Housekeeping (Table 5)**: IMON_EFI_BOARD (0-150 mA, typ 100-130 mA), IMON_EFI_X/Y/Z (0-130 mA, typ 55-70 mA sunlit, up to 80 mA in cold eclipse-saturated state), IEFI_IBIAS (-528 to 528 nA), USHER/GUARD/BRAID readbacks (±40 V), preamp temperatures (−258 to 357°C, typ 20-30°C SPB, 30-40°C AXB, −135°C eclipse), BEB FPGA temperature.

### Part V: First Results (Section 3, pp. 318-322) / 초기 관측 결과

#### 3.1 Large-Amplitude Whistler Wave Observations (Fig 9-10)

THEMIS-E 2007-11-14 0335:07 UT, full bandwidth 8192 samp/s burst: whistler-mode fluctuations up to **400 mV/m peak-to-peak** (corrected for AC gain) during a ~1 V change in spacecraft floating potential (≈10-20% ambient density variation). The enhancement extends at least to the 4096-Hz Nyquist of burst mode. Such waves are **sparse, high-amplitude bursts** of duty cycle ~1% rather than uniform spectral hiss as previous purely spectral observations had assumed → fundamentally different model for relativistic electron acceleration (Cattell et al. 2008; Cully et al. 2008a).

#### 3.2 Hall Electric Field at Magnetopause (Fig 11-12)

THEMIS-C, D, E inbound-outbound magnetopause crossings 1736-1743 UT 20 July 2007. EFI E-field from spin fit + perpendicular ($\mathbf{E}\cdot\mathbf{B}=0$) estimate, vs $-\mathbf{V}_i\times\mathbf{B}$ (ideal MHD). EFI agrees with $-\mathbf{V}_i\times\mathbf{B}$ to a few tenths of mV/m in magnetosheath, but **deviates at the magnetopause crossings** (consistent with Hall term predictions of generalized Ohm's law):

$$\mathbf{E} + \mathbf{V}_i\times\mathbf{B} = \frac{\mathbf{j}\times\mathbf{B}}{en} - \frac{\nabla\cdot P_e}{en} + \frac{m_e}{ne^2}\frac{\partial\mathbf{j}}{\partial t} + \eta\mathbf{j}$$

Fig 12 shows $\mathbf{E} + \mathbf{V}_i\times\mathbf{B}$ matches estimated Hall term $\mathbf{j}\times\mathbf{B}/en$ (with $\mathbf{j}$ from $\partial\mathbf{B}/\partial t$ and multi-spacecraft timing) to better than 1 mV/m over bulk of crossing, demonstrating Hall-dominance. The Hall E-field appears on the **magnetospheric** (low-density, high-B) side, in contrast to the symmetric quadrupolar Hall pattern of magnetotail reconnection (Oieroset et al. 2001) — an asymmetric magnetopause pattern consistent with generalized Ohm's law predictions (Mozer et al. 2008).

#### 3.3 Electron Diffusion Region E-Field (Fig 13)

THEMIS-D 2007-08-24 0041:39 UT, 11.8 RE, 1500 MLT. $E_{perp}$ from spin-plane wire boom + $\mathbf{E}\cdot\mathbf{B}=0$. **~100 mV/m spike** lasting **30 ms** in spacecraft frame. Magnetopause velocity 10-20 km/s (multi-s/c timing) → spatial size **< 1 km**, fraction of $c/\omega_{pe}$ (~4 km at $n=2$ cm⁻³) → candidate **electron diffusion region** event.

### Part VI: On-Orbit Performance (Section 4, pp. 323-339) / 궤도 성능

#### 4.1 Bias Optimization (Fig 14-16)

In sunlit tenuous magnetospheric plasma the natural sensor floating potential makes $R_s \sim 10^9$ Ω. To accurately measure mV/m fields one needs $R_s \ll R_{external}$; the standard solution is **negative bias current** (electron injection into sphere) drawing the operating point near the photoelectron-saturation region of the I-V curve.

**Fig 14** (modeled CPS conditions, $n=0.3$ cm⁻³, $T_e=600$ eV, $T_i=4.2$ keV, sunlit):
- Top panel: collected current vs probe potential — log scale, photoelectron, ambient electron, ambient proton contributions.
- Second panel: total current.
- Third panel: small-signal sheath resistance $R_s = (dI/dV)^{-1}$.
- Fourth panel: $R_s$ vs IBIAS — broad minimum near **half the saturation photoemission current** (~120 nA for assumed 4 nA/cm² density on 8-cm sphere).

Without bias: $R_s \sim 10^9$ Ω → 10-100 mV error voltages, very poor S/N. With ~180 nA bias: $R_s \sim 10^7$ Ω, factor 100-1000 reduction → fraction-of-mV error.

**Fig 15-16** (CPS vs PSBL): floating potentials and $R_s$ vs IBIAS. PSBL has lower-density plasma → required bias range is 20-30 nA/sensor (much narrower than CPS's optimum 120 nA). On-orbit, photoelectron return from the spacecraft body broadens the workable bias range up to ~180 nA in CPS, but the more stringent low-bias-current requirement (for accurate accounting of photoelectron recollection) dominated final design.

USHER and GUARD biased ±40 V relative to sensor allow leverage over photoelectron transport in their local potential structures. Exact biasing determined on-orbit via Sensor Diagnostic Tests (SDT, Sec 4.2).

**Spacecraft saturation**: when bias drives the floating ground supply outside its ±80 V dynamic range, sensors saturate and EFI cannot measure external field. Practical operation keeps $|V_{sc}| \lesssim$ tens of V.

#### 4.2 Sensor Diagnostic Tests (SDT) (Fig 17-18)

THEMIS-A 2008-01-15 0230-0313 UT X-axis SDT. **5 USHER/GUARD steps × 16 IBIAS steps**, each held constant for 3 s (one spin). Y and Z sensors held at nominal to monitor ambient field while X is swept.

Findings:
- X-axis sensors saturate at -80 V at IBIAS = -250 nA initially (driving spacecraft positive).
- Sensors **come out of saturation** at IBIAS = 190-150 nA → saturation photocurrent ≈ 170-200 nA (consistent with prediction). Achieved after ~1 month on orbit.
- Y- and Z-axis floating potentials respond to spacecraft potential changes: Y shows much larger swings than Z because Y-sensors are closer to s/c (~20 m vs 2.5 m to AXB), seeing radial falloff in s/c potential structure.
- Optimal flight configuration (June-July 2007 dayside campaign): **+4 V USHER/GUARD, V1-driven DBraid**. Negative USHER/GUARD biasing produced intense low-energy (few eV) electron returns into ESA detector — significant ESA error source not seen on previous missions due to THEMIS's compact geometry.

#### 4.3 Spacecraft Potential Variations (Fig 19-20)

Once bias is optimal, residual **spin-periodic variations** of tens of mV remain in $V_{sc}$ due to spin-modulated photoemission. **Four-per-spin variation** of ≈1 V seen (Fig 19, THEMIS-C 2008-02-22 0655 UT). Origin: relatively large photoemission/collection area of PBraid (proximal braid) of SPB cable compared to ~25-30% of central spacecraft body's exposed area.

**Fig 20** model: total photoemission area vs spin phase and Sun angle. Four-lobe pattern (period 90°) reproduces observation. Long boom (V1, V2) vs short boom (V3, V4) sensor potentials differ by few tenths of V, consistent with this hypothesis.

Axial sensors (V5, V6) show **several volts** of difference from spin-plane potentials — because axial spheres are 10× closer to potential center (~2.5 m vs 20 m), they see less floating-potential drop. Four-per-spin spikes in axial measurements are momentary shadowing of axial sensor by SPB.

#### 4.4 Cross-Calibration & Boom Shorting (Fig 21)

Methodology: in regions where ideal MHD holds ($\mathbf{E} = -\mathbf{V}_i\times\mathbf{B}$) and conditions slowly vary, correlate $\mathbf{E}_{EFI}$ with $-\mathbf{V}_{i,ESA}\times\mathbf{B}_{FGM}$. Best-fit linear model gives:
- **Sunward offset**: typical 2.5 mV/m (well-known photoelectron asymmetry).
- **Dawn-dusk offset**: ~0 mV/m.
- **Boom shorting factor**: 1.3-1.6 for THEMIS spin-plane (from THEMIS-C July-Aug 2007). Equivalent slope of 0.69 means EFI underestimates by factor 1/0.69 ≈ 1.45.

For comparison: Polar boom shorting 1.2-1.4 (Mozer, priv. comm.); Fahleson 1974 analytical theory predicts smaller boom shorting for THEMIS than Polar (different spacecraft sizes, boom geometries).

Shorting factor approaches 1 as boom length increases — vital reason to use long booms.

Spread $\mathbf{E}_{EFI}$ vs $-\mathbf{V}_{i,ESA}\times\mathbf{B}_{FGM}$ is ≈ 2 mV/m, representing both calibration accuracy and intrinsic differences (EFI/FGM are spin-fit; ion velocity is from spin-resolution 3D particle data with counting statistics + aliasing). Over shorter intervals, accuracy can be a few tenths of mV/m (Mozer et al. 2008).

#### 4.5 Electrostatic Wake Effects (Fig 22-23)

Cold (< tens of eV) plasma flowing past spacecraft creates **electrostatic wakes** behind the s/c, causing spurious E-field signatures. THEMIS magnetopause/dayside has higher cold-plasma occurrence than expected.

**Fig 22**: THEMIS-C 21 July 2007 0825-1320 UT magnetopause crossings. EFI vs $-\mathbf{V}_i\times\mathbf{B}$ agrees in magnetosheath, **disagrees** by tens of mV/m in plasma sheet (after 1100 UT) when cold ion population (10-100 eV below main peak) is present.

Wake-field amplitude scales as $1/L^2$ or $1/L^3$ (boom length), so the **long-vs-short SPB length asymmetry** on THEMIS is essential for **diagnosing** wakes — long boom (E12, 49.6 m) signatures should be smaller than short boom (E34, 40.4 m) when wake is present, and both deviate from sinusoidal spin signature. Fig 23 shows clear diagnostic: long-boom E ~few mV/m, short-boom E ~tens of mV/m, both non-sinusoidal — wake event confirmed and flagged.

#### 4.6 Axial Boom Performance (Fig 24-26)

Short axial boom (6.93 m tip-to-tip) is strongly affected by spacecraft potential variations. **Charge-center displacement effect** scales as:

$$E_{err} \approx V_{sc}\cdot\frac{2ad}{L^3}$$

with $V_{sc}$ spacecraft potential, $a$ effective spacecraft radius, $d$ charge-center displacement along boom axis, $L$ boom length. Factor ~1000 larger on axial booms than spin-plane.

Fig 24 (THEMIS-C 25 May 2007): **4 (mV/m)/V** correlation between $E_z$ and sensor potentials → ~6 cm displacement of charge center along axial direction. Predicted < 0.5 m from antenna mast geometry; the axial booms were intentionally deployed asymmetrically (+Z sensor 4 cm further from geometric center) in attempt to compensate. Observed 6 cm residual indicates the antenna mast was not the only source; magnetometer booms and photoelectron asymmetries between top/bottom decks contribute.

**Fig 25-26** (THEMIS-C 2007-08-08 magnetopause crossing): axial $E_{56}$ shows clean spin-modulation correlated with V1 (s/c potential proxy) at proportionality ~2.7 (mV/m)/V. Spin-dependent variation grows from 6 mV/m peak-to-peak (high-density magnetosheath, smaller $V_{sc}$) to 20 mV/m peak-to-peak (low-density magnetosphere, larger $V_{sc}$). After common-mode subtraction and filtering, residual axial $E_z$ matches both $-\mathbf{V}_i\times\mathbf{B}$ and $\mathbf{E}\cdot\mathbf{B}=0$ derived $E_z$ within mV/m.

Bottom line: short axial booms work for AC measurements (above few hundred Hz) but DC accuracy is event-dependent, requiring careful event-by-event calibration. **AC field measurements (above ~few hundred Hz)** do not suffer this and have been used to make clean 3D E-field estimates (Cully et al. 2008a).

### Part VII: Summary and ESC Appendix (Section 5 + Appendix, pp. 339-340) / 요약 및 ESC 부록

THEMIS-EFI provides high-quality estimates of the near-ecliptic DC E-field (better than 1 mV/m accuracy) for THEMIS magnetotail observations. DC measurement is susceptible to cold-plasma wake contamination, but the long-vs-short SPB length asymmetry allows **routine detection and monitoring** of these effects. 3D E up to 4 kHz, plus integrated 100-400 kHz for AKR, supports both large-scale electrodynamics and small-scale wave phenomena.

**Appendix: ESC Specification.** THEMIS science required tighter specifications than typical:
- Few V to few tens of V absolute charging (sunlit).
- **Differential charging < 1 V** between exposed surfaces (goal < 0.1 V).
- Maximum allowed surface resistance to ground at any point: **125 MΩ·cm²/A** where A is exposed area in cm². Derivation: 8 nA/cm² photoelectron current density (twice expected) → max V drop limited to 1 V.

Implementation: round/square patches grounded at corners or edges; surfaces embedded in apertures; insulating epoxy bond lines; modest surface resistivity max from materials selection. Verification by point-resistance ohmmeter measurement (rather than surface-resistance) was simpler and more achievable.

---

## 3. Key Takeaways / 핵심 시사점

1. **Double-probe E = ΔV / ΔL is conceptually simple but practically a 30-year battle against systematic errors / 더블 프로브 E-field 측정은 개념적으로 단순하나 30년에 걸친 systematic error와의 싸움이다** — 이 논문이 정리한 sheath impedance, 광전자, boom shorting, ES wake, charge-center displacement 모두가 1 mV/m DC accuracy를 위해 신중히 다뤄져야 한다. The simple formula $V_n - V_m = -\mathbf{E}\cdot(\mathbf{X}_n - \mathbf{X}_m)$ hides 30 years of accumulated systematic-error mitigation: sheath impedance, photoelectron emission/collection asymmetries, boom shorting, electrostatic wakes, and charge-center displacement all must be controlled to reach 1 mV/m DC accuracy.

2. **Current biasing transforms sheath impedance by 100-1000× / 전류 바이어스는 sheath impedance를 100-1000배 낮춘다** — sphere에 negative bias current (~180 nA, ≈half-saturation photocurrent)를 주입하여 $R_s$를 $10^9$ Ω에서 $10^7$ Ω로 감소시킴. 이것이 mV/m DC 측정의 핵심. Injecting ~180 nA (about half the photoelectron saturation current) into each sphere drops sheath resistance from $\sim10^9$ Ω to $\sim10^7$ Ω, the single most important on-orbit operational parameter for mV/m DC accuracy.

3. **Asymmetric boom lengths are a feature, not a bug / 비대칭 boom 길이는 결함이 아닌 의도된 설계** — THEMIS는 한 spin-plane pair (49.6 m)를 다른 pair (40.4 m)보다 길게 배포하여 boom shorting과 ES wake를 진단할 수 있다 (wake field ∝ $1/L^{2-3}$). Cluster-II EFW의 동일 길이 88 m booms는 이 진단을 갖지 못한다. THEMIS deliberately deployed one spin-plane pair longer (49.6 m) than the other (40.4 m); since wake fields scale as $1/L^{2-3}$, the asymmetry diagnoses wakes from differential agreement — Cluster-II EFW's equal-length booms cannot do this.

4. **$\mathbf{E}\cdot\mathbf{B}=0$ rescues the short axial boom for perpendicular E / E·B=0 가정이 짧은 axial boom의 perpendicular E를 구원한다** — axial boom의 charge-center displacement error는 $L^{-3}$로 scaling하므로 SPB보다 ~1000배 더 큼. Macroscopic scales에서 perpendicular E가 dominant이므로 spin-plane E와 FGM B에서 axial을 재구성: $E_{axial} = -((B_x/B_z)E_x + (B_y/B_z)E_y)$. The $L^{-3}$ scaling of charge-center displacement makes the 6.93-m AXB ~1000× more sensitive to $V_{sc}$ variations than the 50-m SPB; replacing measured $E_z$ with the perpendicular-component reconstruction recovers usable DC perpendicular-field estimates.

5. **$\mathbf{E} + \mathbf{V}_i\times\mathbf{B}$ ≠ 0 directly reveals Hall reconnection / E + V_i × B ≠ 0 만으로 Hall reconnection이 직접 가시화** — magnetosheath에서는 ideal MHD가 거의 perfect ($-\mathbf{V}_i\times\mathbf{B} \approx \mathbf{E}_{EFI}$), 자기권계면 통과 시에만 mV/m 수준 deviation이 나타나며 이는 $\mathbf{j}\times\mathbf{B}/en$과 일치. 비대칭 reconnection의 Hall E-field가 magnetospheric (low-density) side에 위치하는 첫 검증. In magnetosheath ideal MHD holds to better than a fraction of mV/m; the residual at magnetopause crossings matches the Hall term $\mathbf{j}\times\mathbf{B}/en$, providing the first direct multi-probe verification that Hall E sits on the magnetospheric (low-n, high-B) side in asymmetric reconnection.

6. **400 mV/m whistler bursts redefined relativistic electron acceleration / 400 mV/m whistler 버스트는 상대론적 전자 가속 모델을 재정의했다** — 8192 samp/s burst capture가 이전 spectral observation에서 보지 못했던 1% duty cycle 큰 amplitude bursts를 발견. 이는 균일한 hiss bath에 의한 quasi-linear diffusion 모델이 아닌 nonlinear coherent acceleration 모델 필요. Wave-burst 8192-samp/s capture revealed that inner-magnetosphere whistler-mode hiss is mostly **sparse 1%-duty-cycle 400-mV/m bursts**, not uniform low-amplitude noise — fundamentally changing models of relativistic electron acceleration from quasi-linear diffusion to nonlinear coherent processes.

7. **Spacecraft electrostatic cleanliness is mandatory, not optional / 우주선 정전기 청결도는 옵션이 아닌 필수** — THEMIS-EFI 팀은 ESC 사양 (125 MΩ·cm²/A 표면 저항, < 1 V 차등 충전)을 mission level에 부과했다. 이 사양이 충족되지 않으면 mV/m 정확도는 처음부터 불가능. The EFI team imposed the 125 MΩ·cm²/A surface-resistance and < 1-V differential-charging specification on the entire mission; without it, mV/m DC accuracy would have been unattainable regardless of sensor design.

8. **Multi-spacecraft DC E-field changes electrodynamics from inferred to measured / 다중 위성 DC E 측정은 전기역학을 추론에서 측정으로 바꾼다** — 단일 위성에서는 current sheet, reconnection, wave-particle interaction의 underlying electromagnetic processes를 inference (e.g., $-\mathbf{V}_i\times\mathbf{B}$)로만 알 수 있었으나, THEMIS는 직접 measurement로 hypothesis 간 discrimination을 가능케 했다. With single-spacecraft inference one could only "infer rather than measure" the electromagnetic processes in current sheets, reconnection, and wave-particle interactions; multi-probe direct E-field measurement makes hypothesis-discrimination quantitative.

---

## 4. Mathematical Summary / 수학적 요약

### Double-probe baseline / 더블 프로브 기준식

For two sphere sensors at positions $\mathbf{X}_n$ and $\mathbf{X}_m$ in a uniform external field $\mathbf{E}$:

$$V_n - V_m = -\mathbf{E}\cdot(\mathbf{X}_n - \mathbf{X}_m)$$

Dividing by the baseline length $|\mathbf{X}_n - \mathbf{X}_m| = L$ gives one component of E along the baseline:

$$E_\parallel = -\frac{V_n - V_m}{L}$$

For THEMIS:
- $L_{12} = 49.6$ m (X axis, EDC12)
- $L_{34} = 40.4$ m (Y axis, EDC34)
- $L_{56} = 6.93$ m (Z axis, EDC56)

Effective $L$ on-orbit is reduced by **boom shorting factor** 1.3-1.6 (sensor separation effectively shorter than physical separation due to grounded conductors between spheres):

$$E_{true} = (\text{shorting factor})\cdot E_{measured}$$

### Sheath impedance and bias optimization / Sheath 저항 및 바이어스 최적화

Small-signal sheath resistance:

$$R_s = \left(\frac{dI}{dV}\right)^{-1}$$

For unbiased sphere in tenuous sunlit plasma: $R_s \sim 10^9$ Ω. Optimal bias current $I_{BIAS}$ near half the photoelectron saturation current $I_{ph,sat}$ minimizes $R_s$:

$$I_{BIAS,opt} \approx \frac{1}{2} I_{ph,sat} = \frac{1}{2} J_{ph} A_{sphere}$$

For 8-cm sphere with $J_{ph} = 4$ nA/cm²: $A_{sphere} = 4\pi(4)^2 / 4 = 50.3$ cm² (cross-section), $I_{ph,sat} \approx 200$ nA, $I_{BIAS} \approx 100$ nA. THEMIS uses ~180 nA (above optimum, but extends operational dynamic range). Resulting $R_s$: 10-100 MΩ.

The **error voltage** from external impedance loading is:

$$V_{err} = E_{ext}\cdot L\cdot \frac{R_{ext}}{R_{ext} + R_s}$$

In tenuous plasma $R_{ext}$ between long booms can be $\sim 10^{10}$ Ω from "partial shorting" through the spacecraft body. Reducing $R_s$ from $10^9$ to $10^7$ Ω reduces error voltage by 100-1000×.

### Generalized Ohm's law / 일반화 옴 법칙

$$\mathbf{E} + \mathbf{V}_i\times\mathbf{B} = \frac{\mathbf{j}\times\mathbf{B}}{en} - \frac{\nabla\cdot P_e}{en} + \frac{m_e}{ne^2}\frac{\partial\mathbf{j}}{\partial t} + \eta\mathbf{j}$$

Term-by-term:
- $\mathbf{V}_i\times\mathbf{B}$: ideal MHD convection (motional E-field)
- $\mathbf{j}\times\mathbf{B}/(en)$: **Hall term** — non-zero where ions and electrons decouple (ion scale)
- $\nabla\cdot P_e / (en)$: electron pressure gradient — important in electron diffusion region
- $(m_e / ne^2) \partial\mathbf{j}/\partial t$: electron inertia — electron-scale dynamics
- $\eta\mathbf{j}$: resistive diffusion — usually negligible in collisionless plasma

THEMIS-EFI use: compute LHS from EFI ($\mathbf{E}$) + ESA ($\mathbf{V}_i$) + FGM ($\mathbf{B}$); compute Hall term from FGM time series + ESA + multi-spacecraft separation. Match to mV/m level demonstrates Hall dominance.

### E·B = 0 reconstruction / E·B = 0 재구성

For collisionless magnetized plasma at scales above the inertial length, $\mathbf{E}_\parallel \to 0$ → $\mathbf{E}\cdot\mathbf{B} = 0$:

$$E_x B_x + E_y B_y + E_z B_z = 0$$

Solving for $E_z$:

$$E_z = -\frac{B_x E_x + B_y E_y}{B_z}$$

THEMIS Level-2 "_dot0" replaces measured (noisy) axial E with this reconstruction. **Error amplification**: when $|B_z| / |\mathbf{B}|$ is small (e.g., B in spin plane), small errors in $E_x, E_y$ get multiplied by $|\mathbf{B}_\perp|/|B_z|$ → 10:1 error for B angles below ~6° from spin plane.

### Charge-center displacement error / 전하 중심 변위 오차

$$E_{err} \approx V_{sc} \cdot \frac{2 a d}{L^3}$$

with $V_{sc}$ s/c potential, $a$ effective s/c radius, $d$ charge-center displacement, $L$ boom length. Cubic scaling: 50-m vs 6.93-m booms → ratio $(50/6.93)^3 \approx 376$ → axial boom is hundreds of times more sensitive. Observed THEMIS $d \approx 6$ cm produces 4 (mV/m)/V proportionality on axial sensors.

### Spin fit model / 스핀 핏 모델

$$E(\psi) = A + B\sin\psi + C\cos\psi$$

with $\psi$ spin phase relative to Sun pulse. Recovers despun horizontal field $\mathbf{E}_{DSL,xy}$:

$$E_{DSL,x} = -B,\qquad E_{DSL,y} = C\quad\text{(sign convention dependent)}$$

$A$ is a per-spin offset (e.g., sunward photoelectron asymmetry). Iterative outlier subtraction handles spin-modulated wake/photoelectron features that are not pure sinusoids.

### Wake field scaling / Wake 필드 스케일링

Theoretical prediction (Engwall 2004 internal report; Puhl-Quinn et al. 2008): electrostatic wake contribution to differential E scales as $1/L^2$ to $1/L^3$ depending on spacecraft geometry and boom grounding scheme. THEMIS leverages 49.6 vs 40.4 m → factor $(49.6/40.4)^3 \approx 1.85$ — long boom should see ~half the wake contamination of short boom; comparison detects wakes.

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1960s -- ATS / OGO geosynchronous: first attempts at magnetospheric DC E
1974 ---- Mozer et al., balloon E-fields; Mozer et al. NASA proposal for E on
          low-altitude electrodynamics explorer
1977 ---- ISEE-1/2 launched: 73-m double-probe, first practical
          magnetospheric DC E-field
1981 ---- Whipple, "Potentials of surfaces in space", Rep. Prog. Phys.
          (foundational s/c charging reference)
1990s -- S3-3, CRRES, FAST: progressively refined double-probe technique
1996 ---- POLAR EFI launched (Harvey et al. 1995): 130 m & 100 m wire booms,
          14 m axial. Demonstrated quasi-DC vector E with bootstrapped braid.
1998 ---- Pedersen et al., "Electric field measurements in tenuous plasma..."
          AGU Monograph 103 — canonical double-probe technique reference
2000 ---- Cluster-II EFW (Gustafsson et al.): 88 m booms × 4 spacecraft, equal
          lengths. First multi-spacecraft E-field measurement
2007 Feb - THEMIS launched: 5 probes, EFI on each. ARTEMIS subsequently
        |  reorbits two probes to lunar orbit (~2009-2010)
2008 -->* THIS PAPER (Bonnell et al.): first comprehensive THEMIS-EFI report
          along with companion papers (Cully et al., Mozer et al., 
          Angelopoulos et al., Auster et al., McFadden et al., Roux et al.)
          and discoveries (large-amplitude whistlers; asymmetric Hall 
          reconnection; EDR candidates).
2015 ---- MMS launched: SDP (60 m wire booms × 4) + ADP (14.6 m axial),
          inheriting Berkeley double-probe lineage
2018 ---- Parker Solar Probe FIELDS (Bale et al.): 4 × 2 m radial wire 
          booms in solar wind
2020 ---- Solar Orbiter RPW (Maksimovic et al.): triple monopole antennas
2025+--- THEMIS still operating (>17 years) — longest E-field dataset 
          in magnetospheric science
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Pedersen et al. 1998 (AGU Mon. 103) | Foundational double-probe technique reference cited throughout THEMIS-EFI design | Establishes mathematical framework $V_n - V_m = -\mathbf{E}\cdot(\mathbf{X}_n - \mathbf{X}_m)$ that EFI implements |
| Harvey et al. 1995 (Polar EFI) | Direct mechanical/electrical predecessor at UCB SSL | Polar's bootstrapped-braid, current-biasing approach directly inherited and refined for THEMIS |
| Angelopoulos 2008 (#77, THEMIS mission) | Defines the science requirements that drove EFI design (substorm onset, dayside reconnection) | Sets the 1 mV/m / 10% accuracy at 8-10 RE plasma sheet, dawn-dusk at 18-30 RE, etc. |
| Auster et al. 2008 (#78, THEMIS FGM) | Provides B-field for $\mathbf{E}\cdot\mathbf{B}=0$ reconstruction and $-\mathbf{V}_i\times\mathbf{B}$ cross-cal | EFI relies on FGM B for axial-component reconstruction and boom-shorting calibration |
| Roux et al. 2008 (#79, THEMIS SCM) | Search-coil magnetometer for AC magnetic; combined with EFI for Poynting flux | EFI/SCM phase agreement < 1° enables Poynting flux and wave mode identification |
| McFadden et al. 2008 (#80, THEMIS ESA) | ESA ion velocity for $-\mathbf{V}_i\times\mathbf{B}$ MHD reference; spacecraft potential correction needs EFI | Cross-calibration of EFI uses ESA $\mathbf{V}_i$; ESA particle moments need EFI's $V_{sc}$ correction |
| Mozer et al. 2008 (Hall reconnection) | Companion paper using THEMIS-EFI to detect asymmetric Hall E-fields at magnetopause | Section 3.2 results — Hall E on magnetospheric side, mV/m agreement with $\mathbf{j}\times\mathbf{B}/en$ |
| Cattell et al. 2008 (large-amp whistlers) | Companion paper announcing 400 mV/m whistler bursts in radiation belt | Section 3.1 result — redefined relativistic electron acceleration physics |
| Cully et al. 2008b (THEMIS DFB) | Sister-instrument paper describing Digital Fields Board signal processing | Defines spectral products, filter banks, FFT cadences referenced in EFI Table 3-4 |
| Burton-McPherron-Russell 1975 (#11) | Vsw·B → Dst empirical formula presaged the in-situ E-measurement need | Showed E-field controls magnetic storm response; THEMIS-EFI provides direct in-situ E |
| Oieroset et al. 2001 (Nature) | Magnetotail Hall reconnection (symmetric quadrupole) | Contrasted in this paper with THEMIS-EFI's asymmetric magnetopause Hall pattern |

---

## 7. References / 참고문헌

- Bonnell, J.W., Mozer, F.S., Delory, G.T., Hull, A.J., Ergun, R.E., Cully, C.M., Angelopoulos, V., Harvey, P.R., "The Electric Field Instrument (EFI) for THEMIS", Space Sci. Rev. 141, 303-341 (2008). DOI: 10.1007/s11214-008-9469-2
- Pedersen, A., Mozer, F., Gustafsson, G., "Electric field measurements in a tenuous plasma with spherical double probes", in Measurement Techniques in Space Plasmas: Fields, AGU Geophysical Monograph 103 (1998).
- Harvey, P.R. et al., "The electric field instrument on the Polar satellite", Space Sci. Rev. 71, 583-573 (1995).
- Angelopoulos, V., "The THEMIS Mission", Space Sci. Rev. 141, 5-34 (2008).
- Auster, H.U. et al., "The THEMIS fluxgate magnetometer", Space Sci. Rev. 141, 235-264 (2008).
- McFadden, J.P. et al., "The THEMIS ESA plasma instrument and in-flight calibration", Space Sci. Rev. 141, 277-302 (2008).
- Roux, A. et al., "The Search Coil Magnetometer for THEMIS", Space Sci. Rev. 141, 265-275 (2008).
- Cully, C.M., Bonnell, J.W., Ergun, R.E., "THEMIS observations of long-lived regions of large-amplitude whistler waves in the inner magnetosphere", Geophys. Res. Lett. 35, L17S16 (2008a). DOI: 10.1029/2008GL033643
- Cully, C.M. et al., "The THEMIS Digital Fields Board", Space Sci. Rev. (2008b).
- Cattell, C. et al., "Discovery of very large amplitude whistler-mode waves in Earth's radiation belts", Geophys. Res. Lett. 35, L01105 (2008). DOI: 10.1029/2007GL032009
- Mozer, F.S., Angelopoulos, V., Bonnell, J., Glassmeier, K.H., McFadden, J.P., "THEMIS observations of modified Hall fields in asymmetric magnetic field reconnection", Geophys. Res. Lett. 35, L17S04 (2008).
- Oieroset, M. et al., "In situ detection of collisionless reconnection in the Earth's magnetotail", Nature 412, 414 (2001).
- Whipple, E.C., "Potentials of surfaces in space", Rep. Prog. Phys. 44, 1197-1250 (1981).
- Puhl-Quinn, P.A. et al., "An effort to derive an empirically based, inner-magnetospheric electric field model: Merging Cluster EDI and EFW data", J. Atmos. Sol.-Terr. Phys. 70, 564-573 (2008).
- Laasko, H., Aggson, T.L., Pfaff, R.F., "Plasma gradient effects on double-probe measurements in the magnetosphere", Ann. Geophys. 13, 130-146 (1995).
- Sahu, K., Kniffen, S., Radiation report on OP15 (Analog Devices) (LDC9722A), Unisys Corp.—Federal Systems Div., Tech. Memo PPM-98-008 (1998).
- Garrett, H.B., Whittlesey, A.C., "Spacecraft charging, an update", IEEE Trans. Plasma Sci. 28, 2017-2028 (2000).
- Engwald, E., "Numerical studies of spacecraft-plasma interaction: simulation of wake effects on the Cluster electric field instrument EFW", IRF Scientific Report 284 (2004).
- Mozer, F.S., Berthelier, J.-J., Fahleson, U.V., Falthammar, C.-G., "A proposal to measure the quasi-static vector electric field on the low altitude and the elliptic orbiting electrodynamics explorer satellites", Research Proposal to NASA, UCBSSL No. 552/75, 1974.
- Ludlam, M. et al., "THEMIS magnetic cleanliness specification", Space Sci. Rev. 141, 171-184 (2008).
