---
title: "The Coronal Diagnostic Spectrometer for the Solar and Heliospheric Observatory (SOHO/CDS)"
authors: ["R. A. Harrison", "E. C. Sawyer", "M. K. Carter", "et al. (30+ co-authors)"]
year: 1995
journal: "Solar Physics"
doi: "10.1007/BF00733431"
topic: Solar_Observation
tags: [SOHO, CDS, EUV, spectrometer, instrument-paper, coronal-diagnostics, Wolter-Schwarzschild, Rowland-circle, MCP, line-ratio]
status: completed
date_started: 2026-04-25
date_completed: 2026-04-25
---

# 44. The Coronal Diagnostic Spectrometer for SOHO (CDS) / SOHO 코로나 진단 분광기

---

## 1. Core Contribution / 핵심 기여

Harrison et al. (1995)은 SOHO 위성의 핵심 EUV 분광 장비인 CDS (Coronal Diagnostic Spectrometer)의 설계, 제작, 보정 및 과학 운용을 종합적으로 기술한 instrument paper다. CDS의 과학적 목표는 SOHO 미션의 두 가지 큰 질문 — "코로나는 왜 가열되는가?", "태양풍은 어떻게 가속되는가?" — 에 답하는데 필요한 플라즈마 진단(밀도 $n_e$, 온도 $T$, 도플러 속도 $v$, 원소 풍부도)을 transition region부터 hottest corona까지 ($\log T = 4.0$-$6.5$, 즉 $10^4$-$\text{few}\times 10^6$ K) 광범위하게 수행하는 것이다. 핵심 설계 혁신은 단일 Wolter-Schwarzschild II 망원경이 두 개의 분광기 (Grazing Incidence Spectrometer, GIS와 Normal Incidence Spectrometer, NIS)를 동시에 공급한다는 점이다. GIS는 4개의 MCP 검출기로 151-785 Å의 광폭 파장을 다루며 astigmatic하지만 핀홀 슬릿+래스터 스캔으로 영상을 구축한다. NIS는 2개의 toroidal 그레이팅으로 308-381 / 513-633 Å 좁은 두 밴드만 다루지만 stigmatic하여 단일 노출로 1차원 슬릿 영상을 직접 얻는다. 둘을 결합하면 광범위한 spectral coverage와 고품질 imaging을 모두 달성한다.

Harrison et al. (1995) is the comprehensive instrument paper for the Coronal Diagnostic Spectrometer (CDS), one of twelve instruments on the ESA/NASA SOHO mission stationed at L1. CDS addresses SOHO's two grand questions — "How is the corona heated?" and "How is the solar wind accelerated?" — by providing simultaneous diagnostics of electron density, temperature, Doppler flow velocity, and elemental abundance for plasmas spanning $\log T = 4.0$-$6.5$. The principal design choice is to feed a single Wolter-Schwarzschild type 2 grazing-incidence telescope simultaneously into two spectrometers: a Grazing Incidence Spectrometer (GIS) covering 151-785 Å in four discontinuous bands using four MCP/SPAN detectors on a 750-mm Rowland circle, and a Normal Incidence Spectrometer (NIS) covering 308-381 / 513-633 Å using two toroidal gratings imaging onto a single intensified-CCD detector. GIS is astigmatic but has wide coverage; NIS is stigmatic and provides direct slit-imaging spectra. The pair therefore offers wide diagnostic coverage *and* high-quality imaging — a combination previously impossible. The paper documents optics design, telescope/grating performance, detector characterisation (MTF, DQE, deadtime, noise), straylight control, mechanical/thermal/electrical implementation, pre-flight BESSY-traceable calibration (6-8% absolute), and operating modes including "interesting event" cross-instrument triggers.

---

## 2. Reading Notes / 읽기 노트

### Part I: Science Objectives and Diagnostic Line List / §1 과학 목표와 진단 라인 리스트

논문 §1은 CDS의 성능 요구사항을 SOHO의 거대한 두 질문(코로나 가열, 태양풍 가속)으로부터 도출한다. 요구사항은 다음과 같이 정리된다:

The paper opens by deriving CDS performance requirements from SOHO's two grand-challenge questions on coronal heating and solar wind acceleration:

| 요구 / Requirement | 값 / Value |
|---|---|
| 파장 / Wavelength | 150-800 Å (포함 hottest coronal lines $\log T = 6.0$-$6.5$) |
| 분광 분해능 / Spectral resolution | $\delta\lambda \sim 0.3$ Å, $\lambda/\delta\lambda \approx 500$ at 150 Å |
| 시야 / FOV | 4 arcmin (telescope), repointing to ±0.75° (full disc) |
| 공간 분해능 / Spatial resolution | A few arcsec |
| 시간 분해능 / Time resolution | Down to ~1 s |

라인 선택은 Tables I-IV에 정리된다. **Table I (밀도 진단)** 은 metastable level이 collisional de-excitation에 의해 인구가 결정되는 forbidden-allowed 라인쌍 — Mg VI/VII/VIII, Si VIII/IX/X, S X/XI/XII, Fe X/XI/XII/XIII/XIV (총 22+ 라인쌍). **Table II (온도 진단)** 은 같은 이온의 큰 에너지 차 두 라인 — O III, O V, O VI, Ne V, Mg IX, Si XI. **Table III (이온화 시리즈 by 원소)** 은 같은 원소의 인접 이온화 단계로 광범위 $T$ 커버 (Ne I-VIII: $\log T = 4.4$-$5.8$; Mg VI-IX: $5.6$-$6.0$; Si VII-X: $5.8$-$6.0$; Fe VIII-XVI: $5.6$-$6.4$). **Table IV (밝고 잘 분리된 dynamics 라인)** 은 도플러 속도 측정용 라인 후보. 추가로 high-T 라인 Fe XXI 270.52 Å, Fe XXIII 263.76 Å (flare용)도 언급.

Line selection drives the wavelength range. **Table I (density-sensitive ratios)** lists ~22 line pairs in Mg VI/VII/VIII, Si VIII/IX/X, S X/XI/XII, Fe X-XIV. **Table II (temperature ratios)** uses widely-separated levels in O III/V/VI, Ne V, Mg IX, Si XI. **Table III** organizes lines as ionization series in Ne, Mg, Si, Fe. **Table IV** lists bright, well-separated lines for dynamics. The hottest lines (Fe XXI 270.52, Fe XXIII 263.76) extend into flare regimes.

### Part II: Instrument Overview / §2 장비 개요

CDS는 단일 망원경 + 이중 분광기 구조로 (Fig. 1, Table V 요약):

CDS combines one telescope with two spectrometers:

- **Telescope**: Wolter-Schwarzschild type 2 (paraboloid + confocal hyperboloid), zerodur, gold coated, 10 Å rms surface, made by Zeiss. Outer f/9.38, focal length 257.831 cm, area 289.28 cm², plate scale 12.5 μm/arcsec. PSF FWHM 1.2-1.7" (Fig. 2). FOV 4 arcmin, repoint ±0.75°.
- **Light split**: An aperture stop after the telescope defines two beam sections — 47 cm² for GIS, 34.3 cm² (per grating) for NIS.
- **Common slit assembly**: 6 slits in front of both spectrometers; sizes 2"×2", 4"×4", 8"×50" for GIS; 2"×240", 4"×240", 90"×240" for NIS.
- **GIS**: 1500-mm-radius spherical grating at $\theta = 84.75°$ grazing, 1000 lines/mm, gold-coated zerodur replica from Hyperfine. 4 MCP detectors on a 750-mm Rowland circle covering 151-221, 256-338, 393-493, 656-785 Å.
- **NIS**: 2 toroidal gratings (radii 743.6 / 736.4 mm) at 7-9° incidence, 2400 and 4000 lines/mm respectively, blazed at 4°. Imaging onto one VDS detector (Tektronix 1024×1024 CCD with MCP image intensifier).

전체 사양: 100 kg, 1.7 m, 58 W, 11.3 kbit/s telemetry. 첫 자연주파수 52 Hz.

Mass 100 kg, length 1.7 m, average power 58 W, telemetry rate 11.3 kbit/s, first natural frequency 52 Hz.

CDS는 이전 EUV 장비들 — OSO VII (1972, ~20" 분해능), Skylab S082 (1973-74, 사진건판 → 시간분해 한계), CHASE (Spacelab II, 1985, 단기 비행), SERTS (rocket, 단시간) — 에 비해 모든 측면에서 우월하다고 §2 후반부가 주장한다.

§2 argues CDS is the first long-duration EUV experiment satisfying all of (i) wavelength coverage to 150 Å (hottest plasmas), (ii) some spatial resolution (not slitless integrated-Sun), (iii) some spectral resolution. Skylab (slitless), CHASE/SERTS (short flights), OSO VII (~20" pixels) all fail at least one criterion.

### Part III: Optical Design / §3 광학 설계

**§3.1 Telescope.** Wolter-Schwarzschild II는 grazing 입사 두 번 반사: paraboloid 12.6-17.3°, hyperboloid 11.1-18.4°. Abbé sine 조건이 spherical aberration과 coma를 최소화하고, Schwarzschild 변형 (true conic에서의 deviation, 10±2 nm)이 off-axis 성능을 개선해 4 arcmin FOV를 이룬다. Zerodur 글래스 세라믹 선택 이유: 매우 낮은 thermal expansion 계수. PSF는 헬륨 hollow-cathode 광원 (304, 584 Å)을 125 m 거리의 핀홀로 사용해 측정. NIS aperture: $X = 1.5"$, $Y = 1.2"$ FWHM. GIS aperture: $X = 1.7"$, $Y = 1.2"$ FWHM. half-energy width < 5". Large-angle scatter (Fig. 3): $10^{-4.3}$ at 20", $10^{-6.6}$ at 150", $10^{-8.5}$ at 1000".

The telescope is a Wolter-Schwarzschild II — paraboloid + confocal hyperboloid — both at grazing incidence (12-18°). Built from zerodur (very low CTE), surfaces polished to 10 Å rms and gold coated. Schwarzschild deviation 10±2 nm. PSF measured with helium hollow-cathode at 125 m, pinhole 5 μm: FWHM 1.2-1.7" depending on direction and aperture. Half-energy width <5". Large-angle scatter (Fig. 3) drops $\sim 10^{-4.3}$ at 20" to $10^{-8.5}$ at 1000".

**§3.2 Scan mirror.** Flat zerodur (160×35×20 mm), gold coated, $\sim$10 Å rms, $\lambda/10$ flat unconstrained ($\lambda = 6328$ Å). Rotates ±0.3° → covers ±2 arcmin on Sun. Stepper motor: 1 step = 2" view direction change. Kinematically mounted, pre-loaded 17 kg/axis to survive 3 kN launch loads. Lubrication: solid lubricants (no organics) for cleanliness.

A flat zerodur mirror just upstream of the slits scans ±2 arcmin in the dispersion direction; one motor step = 2" on the Sun.

**§3.3 Slits.** 6 slits in a common assembly. Sizes:
- NIS (long slits, dispersion-perpendicular): 2"×240", 4"×240", 90"×240" (viewfinder).
- GIS (square pinholes — astigmatic): 2"×2", 4"×4", 8"×50".
- Fabrication: 25 μm Ni foil electroformed onto 100 μm Cu support, epoxy-bonded to Al frame.
- Linear translator selects slits in 1" increments.

Six slits in a single assembly. NIS uses long slits (dispersion-perpendicular spatial info preserved by 2-D detector), GIS uses pinholes (rasters to build images).

**§3.4.1 GIS.** Rowland-circle Spectrometer. Equation (1): $n\lambda = d(\sin\theta + \sin\alpha)$. With $\theta = 84.75°$, $d = 1\,\mu\mathrm{m}$, $n=1$, the diffracted angle $\alpha$ varies from ~78° (at 151 Å) to ~67° (at 785 Å). Resolving power Eq. (2): $\lambda/\delta\lambda = 2\lambda R / (d\,\epsilon\cos\alpha)$. Table VI gives values at 11 wavelengths for $R = 750$ mm, 25 μm slit:

| $\lambda$ (Å) | $\alpha$ (°) | Width (μm) | $\lambda/\delta\lambda$ | $\delta\lambda$ (Å) | Disp (Å/mm) |
|---|---|---|---|---|---|
| 151 | 78.7 | 163 | 709 | 0.21 | 1.29 |
| 256 | 76.0 | 132 | 1201 | 0.21 | 1.59 |
| 393 | 73.0 | 110 | 1837 | 0.21 | 1.91 |
| 656 | 68.5 | 87 | 3082 | 0.21 | 2.41 |
| 785 | 66.5 | 80 | 3697 | 0.21 | 2.63 |

GIS의 4개 검출기는 Rowland 원의 chord로 배치되어 다음 4 밴드를 잡는다: 151-221, 256-338, 393-493, 656-785 Å. 추가로 zero-order 검출기(50 μm slit + 광트랜지스터, 5000-9500 Å 감도)는 태양 limb 통과로 in-flight pointing 보정.

GIS uses a 750-mm Rowland circle; four flat MCP detectors cover the four bands. Resolution $\delta\lambda \approx 0.21$ Å across the entire range. A zero-order photo-transistor detector (sensitive 5000-9500 Å) helps in-flight pointing calibration via solar limb crossings.

**§3.4.2 GIS detectors (Fig. 4).** "Open-face" Z-stack of 3 microchannel plates (50 mm × 25 mm sensitive area, 12.5 μm pores, 15 μm web, angled 13°). Each EUV photon → ~$4\times 10^7$ electrons. Behind: a SPAN (Spiral Anode) with 11-bit positional resolution along dispersion direction. SPAN consists of damped-sine-wave pitched electrodes (390 μm pitch, 40 μm minimum width) on chrome+aluminium-coated fused silica, gaps cut by Nd:YAG laser. The ratio $X/(X+Y+Z)$ of charges on three electrodes uniquely encodes position. 4.3 kV max across MCPs, -12 V front-face to repel photoelectrons. Two ADCs (8-bit) divide ratios to give 13-bit serial event stream → 2048 pixels per detector at 25 μm. MTF 10% at 18 lp/mm = 28 μm = 2.24". Dead time: $T_w = 1.5\,\mu$s, $T_p = 0.6\,\mu$s; extending model $R_o = R_i \exp[-R_i(T_w+T_p)]$ peaks at $R_i = 4.75\times 10^5$ c/s, where $R_o/R_i = 1/e$.

Z-stack of three MCPs with a SPAN anode behind. Each photon produces ~$4\times 10^7$ electrons; charge ratios on three sine-wave-pitched electrodes encode 1-D position to 11 bits. MTF 10% at 18 lp/mm. Dead-time peaks at $R_i = 4.75\times 10^5$ c/s with $1/e$ throughput.

**§3.4.3-4 NIS and VDS detector.** Sirks-style Rowland design with exit beam along the diameter — almost-stigmatic. Two toroidal gratings (mounted side-by-side as if one grating with two ruling densities) disperse two bands and a small out-of-plane tilt vertically separates them on the detector. Bands: 308-381 (top) / 513-633 Å (bottom). Resolving powers (Table VII): $\lambda/\delta\lambda = 3635-4500$, $\delta\lambda = 0.08$ Å (short band) / $0.14$ Å (long band), dispersion 3.17 / 5.56 Å/mm. VDS detector: MCP image intensifier (12 μm pores, 600-950 V) + P-20 phosphor + lens + Tektronix 1024×1024 CCD (21 μm pixels, cooled to -70°C). DQE ~20% in 300-650 Å (Fig. 8); intentionally low at H I Lyα 1216 Å to suppress straylight. Resolution 23 lp/mm = 22 μm pixels = 1.7" effective. Dynamic range 2000:1 single exposure (12-bit ADC; CCD full well 150,000 e⁻).

Two toroidal gratings near-normal incidence (7-9°), 2400 and 4000 lines/mm, blazed at 4°. Two-band stacking on one detector via small out-of-plane grating tilt. NIS resolution $\delta\lambda = 0.08$/$0.14$ Å. The VDS uses MCP intensifier + lens + Tektronix CCD cooled to -70°C; DQE ~20%, dynamic range 2000:1.

### Part IV: Straylight Control / §3.5 산란광 제어

두 종류의 contamination: (1) 공간 — 시야 밖 EUV가 시야 안으로 산란, (2) 분광 — 다른 파장의 EUV가 검출기에 산란. 가장 큰 적은 H I Lyα 1216 Å — CDS 어떤 line보다 ~20배 강함. 9-610 Å에서 Lyα가 모든 코로나 라인 합보다 8배 강함. 다행히 CDS 검출기는 1216 Å에 거의 둔감하지만, 산란된 1216 Å이 photoelectron-emitting 표면을 때리면 detector가 응답할 수 있다. 대응책: (a) 1216 Å이 직접 보지 못하는 baffle 설계, (b) Alochrome-treated Al alloy로 1216 Å scatter를 최소화. NIS는 single-detector + long tube + baffle 구조로 잘 차폐 가능; GIS는 4-detector 구조로 더 어려워 inter-detector baffles 추가. 일반 규칙: "어떤 표면도 illuminated non-optical 표면을 보지 않게 한다."

Two types of contamination — spatial (out-of-FOV EUV) and spectral (other wavelengths). The dominant spectral contaminant is H I Lyα 1216 Å, ~20× brighter than any single CDS line and ~8× the sum of all coronal lines in 9-610 Å. CDS detectors are largely blind to 1216 Å but Lyα hitting low-work-function surfaces ejects photoelectrons that detectors *do* see. Mitigation: (a) baffles forbidding direct view of zero-order or first-order 1216 Å reflective surfaces, (b) Alochrome-treated aluminium for low EUV scatter, (c) inter-detector baffles in GIS, (d) baffled long tube before NIS detector.

### Part V: Mechanical / Thermal / Electrical / §4-6

**§4 Mechanical**: Open truss main structure (Al alloy BS 1470 L115 T651), 6-leg isostatic mounting; front 4 pivoted, rear 2 are linear actuators for ±0.38° pointing. CAD/CNC drawings on RAL Medusa system. First natural frequency 52 Hz (after iteration to overcome non-linear behaviour of mounting). Telescope in own sealed volume connected to optical bench via flexible bellows. Slit drive: 200-step ball-screw on cross-roller slideways, accuracy 10% of smallest slit width.

**§5 Electrical**: CDHS (Command and Data Handling System) — IMS T805 transputers, 32-bit. Dual operating + standby + redundant CPUs. EPS (Experiment Power Supply): 27 V DC from spacecraft, MBB-supplied switch-mode DC-DC converter producing +5 V (isolated) and ±12 V rails. MCU (Mechanism Control Unit) drives stepper motors with trapezoidal waveforms (smoother than square wave). 32 temperature sensors @ 12-bit precision (0.125 K).

**§6 Thermal**: Optical bench at 20°C, gradients <2°C. 16 operational + 8 backup heaters (12 W budget). VDS detector requires <-80°C achieved by 1534 cm² space-facing radiator. Solar input 19 W, half absorbed by gold telescope and re-radiated through invar support tube. MLI: 18 core layers double-aluminised mylar + Dacron net + 4 Kapton; effective emissivity 0.02. Spacecraft thermal exchange <1.5 W.

§4-6 cover engineering implementation: Al-alloy open-truss main structure (52 Hz first mode), ±0.38° pointing via two linear actuators, transputer-based CDHS, switch-mode EPS (+5, ±12 V rails), trapezoidal stepper drive, 32 temperature sensors, and a 1534 cm² radiator for the VDS at -80°C.

### Part VI: Calibration and Performance / §8

**§8.1 Wavelength calibration**: Hollow-cathode discharge lamp emits known ionic lines from buffer gas + cathode (99.5% Al). Gaussian-profile fits to detector positions → straight-line position-vs-wavelength relation. Wavelength ranges all within a few Å of design (Table V).

**§8.2 Photometric sensitivity**: Pre-flight via BESSY synchrotron primary (calculable spectral flux from storage ring parameters) + hollow-cathode lamp transfer source. End-to-end: feed CDS through 5 mm collimated beam at vacuum tank (2×10⁻⁶ torr), translate the source across the apertures. February-March 1994, 4 weeks at RAL.

**Count rate equation**:
$$
\text{count/s} = \frac{I\lambda \, a\,b\, A\, \epsilon_t \epsilon_m \epsilon_g \epsilon_d}{h\,c\,f^2}
$$
with $I$ in erg cm⁻² s⁻¹ sr⁻¹, $a\times b$ slit dimensions, $A$ telescope area, $f$ focal length, $\epsilon$ efficiencies. Pre-build estimate of efficiency product for GIS-2: $\epsilon_t \epsilon_m \epsilon_g \epsilon_d = 0.25 \times 0.80 \times 0.03 \times 0.10 = 6\times 10^{-4}$. Measured: $1$-$2\times 10^{-4}$ (Fig. 17), so 20-30% of optimistic prediction — acceptable.

**Absolute uncertainty 6-8% (1σ)**.

**Table XI** gives expected count rates per 2"×2" pixel for quiet Sun:

| Ion | $\lambda$ (Å) | Counts/s | Detector |
|---|---|---|---|
| Fe XV | 284.16 | 90.5 | GIS2 |
| He II | 303.78 | 89.6 | GIS2 |
| He I | 584.33 | 30.0 | NIS2 |
| O IV | 554.52 | 24.9 | NIS2 |
| O V | 629.73 | 14.9 | NIS2 |
| Mg IX | 368.06 | 8.9 | NIS1 |
| Fe XII | 195.12 | 6.2 | GIS1 |
| Fe XI | 180.40 | 5.9 | GIS1 |
| Fe XI | 188.22 | 4.9 | GIS1 |

**In-flight cal**: (i) regular quiet-Sun monitoring, (ii) GIS-NIS cross-cal in overlap regions, (iii) cross-cal with other SOHO instruments and SERTS underflights.

§8 documents wavelength calibration via hollow-cathode lines, photometric calibration traceable to BESSY synchrotron via a hollow-cathode transfer source, end-to-end illumination at RAL, and the efficiency-product chain. Pre-flight uncertainty 6-8% (1σ). Table XI gives quiet-Sun count rates: brightest are Fe XV 284 (90.5), He II 304 (89.6), He I 584 (30.0).

### Part VII: Operations / §10

CDS is "extremely versatile" — slit + raster + wavelength selection + accumulation freely chosen. But 11.3 kbit/s telemetry vs. 1024×300 12-bit pixels means ~5 minutes to dump one full NIS frame, hence on-board compression and windowing. ~50 pre-defined "studies" in CDS Blue Book (Harrison 1993). Five examples:

1. **SYNOP**: Daily N-S NIS scan along central meridian, 16 prime lines → EUV Carrington maps.
2. **SPECT**: GIS spectral atlas, 15×15 raster over 30"×30" field (2"×2" slit) → quiet-Sun monitoring + atlas.
3. **HIVEL**: Rapid NIS raster 30"×30" (2"×240" slit) → Doppler at multiple T's, sensitivity to tens of km/s.
4. **BROAD**: GIS raster 2"×200" over limb → line broadening as a function of altitude (wave activity).
5. **EJECT**: NIS scan 4'×4' on prominence/CME source.

CDS receives "interesting event" coordinates from other SOHO instruments and slews mirror/slit/instrument pointing (foundation of Joint Observing Programmes / JOPs).

§10 covers operations: ~50 pre-defined studies in the Blue Book, on-board windowing/compression to fit telemetry, five representative studies (SYNOP, SPECT, HIVEL, BROAD, EJECT), and the inter-instrument event coordinate sharing that anchors SOHO Joint Observing Programmes.

### Part VIII: Worked Numerical Example — count rate for Fe XV 284.16 Å / 작업 예시: Fe XV 284.16 Å 계수율 산출

CDS의 throughput 식이 어떻게 실측 카운트율을 예측하는지 Fe XV 284.16 Å (GIS2)에 대해 점검한다.
Let us trace how CDS's throughput equation predicts the measured count rate for Fe XV 284.16 Å on GIS2.

**Inputs (Table V, §3.4, §8.2)**:
- Solar quiet-Sun intensity at 284.16 Å (Vernazza & Reeves 1978): $I \approx 1.5\times 10^3$ erg cm⁻² s⁻¹ sr⁻¹.
- Slit (2"×2" pinhole on GIS): $a = b = 25$ μm $= 2.5\times 10^{-3}$ cm.
- Telescope area for GIS aperture: $A = 47$ cm².
- Telescope focal length: $f = 257.831$ cm.
- Pre-build efficiency product: $\epsilon_t \epsilon_m \epsilon_g \epsilon_d \approx 6\times 10^{-4}$ (cited in §8.2 for GIS-2 ~30 nm); measured $\approx 1.5\times 10^{-4}$.

**Photon energy at 284.16 Å**:

$$
E_\gamma \;=\; \dfrac{hc}{\lambda} \;=\; \dfrac{(6.626\times 10^{-27})(3\times 10^{10})}{2.84\times 10^{-6}} \approx 7.0\times 10^{-11}\ \text{erg/photon}
$$

Therefore $\lambda/(hc) = 1/E_\gamma \approx 1.43\times 10^{10}$ photon/erg.

**Geometric factor**:

$$
\dfrac{a\,b\,A}{f^2} \;=\; \dfrac{(2.5\times 10^{-3})^2 \times 47}{(257.831)^2} \;=\; \dfrac{2.94\times 10^{-4}}{6.66\times 10^4} \;\approx\; 4.41\times 10^{-9}\ \text{sr}\cdot\text{cm}^2
$$

(Note: this combined factor has units of slit-solid-angle × area; alternative derivation via $a/f \times b/f \times A$.)

**Predicted count rate**:

$$
\text{count/s} \;=\; I\;\cdot\;\dfrac{\lambda}{hc}\;\cdot\;\dfrac{a\,b\,A}{f^2}\;\cdot\;\epsilon_t\epsilon_m\epsilon_g\epsilon_d
$$

$$
\approx (1.5\times 10^3)(1.43\times 10^{10})(4.41\times 10^{-9})(1.5\times 10^{-4}) \approx 14\ \text{c/s}
$$

Table XI에 기록된 측정값은 **90.5 c/s**. 우리 추정값보다 높은 이유: (1) Fe XV 284 라인이 quiet Sun에서도 active region에 강하게 contaminated, (2) Vernazza-Reeves가 약간 낮게 추정, (3) 효율 product가 30 nm 부근에서 $1.5\times 10^{-4}$에서 $\sim 5\times 10^{-4}$ (282 Å 부근 GIS2 peak). 어느 정도 수준의 ~factor-of-few 일치가 EUV instrument에서 통상이며, 정확한 absolute calibration은 in-flight cross-cal과 ADAS 원자 데이터로 보완된다.
The measured value in Table XI is 90.5 c/s. Our estimate of 14 c/s is low by ~6×, reasonable for an order-of-magnitude verification given (i) Fe XV 284 contamination from active regions even in "quiet" Sun, (ii) Vernazza-Reeves being slightly conservative, (iii) GIS2 efficiency peaking nearer $5\times 10^{-4}$ around 282 Å. Order-of-magnitude agreement is typical for EUV throughput predictions; precise absolute calibration relies on BESSY-traceable lab calibration plus in-flight cross-checks.

### Part IX: Trade-Off Table — Why these specific numbers / 선택의 근거

CDS의 주요 설계 수치들이 왜 그렇게 선택되었는지 정리:

Why CDS's principal design parameters take the values they do:

| Parameter / 설계 수치 | Value / 값 | Rationale / 근거 |
|---|---|---|
| Wavelength range / 파장 범위 | 150-800 Å | Hottest coronal lines (Fe XII 195, Fe XV 284) require ≤200 Å; brightest TR lines (He I 584, O V 630) up to 800 Å |
| Spectral resolution / 분광 분해능 | $\delta\lambda \sim 0.3$ Å (GIS) / 0.08-0.14 Å (NIS) | 150-220 Å crowded blends require $\lambda/\delta\lambda \sim 500$ at shortest |
| Spatial resolution / 공간 분해능 | ~3-5" effective | "fine-scale" coronal structures (loops have widths down to 1") |
| Time resolution / 시간 분해능 | down to 1 s | flare bursts and wave dynamics |
| FOV / 시야 | 4 arcmin telescope; ±0.75° repoint | covers active region; reach any disc location |
| Telescope area / 망원경 면적 | 289.28 cm² (47 GIS, 34.3×2 NIS) | balances throughput vs. PSF (Wolter-Schwarzschild 4 arcmin FOV constraint) |
| GIS Rowland $R$ / GIS Rowland 반경 | 750 mm | trades resolution ($\propto R$) vs. instrument size |
| NIS grating pitch / NIS 그레이팅 피치 | 2400 / 4000 lines/mm | sets two bands at 308-381 / 513-633 Å for stigmatic Sirks geometry |
| MCP pore size / MCP 구멍 | 12.5 μm | maximizes EUV QE while keeping spatial resolution |
| SPAN bits / SPAN 비트 | 11-bit | matches MCP pore size at 25 μm/pixel = 47 μm/pos res |
| Total mass / 총 질량 | 100 kg | SOHO instrument allocation |
| Telemetry / 데이터율 | 11.3 kbit/s | SOHO equitable allocation; on-board compression handles overflow |
| First natural freq / 첫 자연주파수 | 52 Hz | Ariane V launcher requirement (>40 Hz) |
| VDS temperature / VDS 온도 | -70°C (ops), -80°C limit | dark current control; achieved by 1534 cm² radiator |

이 표는 "instrument paper의 모든 수치는 우주 시스템 공학 절충에서 나온다"는 사실을 보여준다 — 한 수치의 변경은 다른 수치 여러 개의 변경을 강제한다.
The table illustrates that every single number in an instrument paper emerges from system-engineering trade-offs — change one and several others are forced.

### Part X: Limitations and Subsequent Lessons / 한계와 후속 교훈

CDS는 미션 수명 동안 다음 한계를 노출했다:

CDS's operational lifetime exposed several limitations:

- **NIS-1 detector damage**: 누적된 솔라 EUV 노출이 phosphor와 MCP를 손상, 1998년 SOHO 사고 후 파장 보정과 spatial response 변화. Lesson for EIS/SPICE: 더 견고한 detector design + regular flat-fielding.
- **Lyα contamination**: 이미 §3.5에서 다뤘지만 실제로는 NIS 양 끝 (308 Å, 633 Å)에서 측정된 background level이 예상보다 높았다. EIS는 multilayer + entrance filter로 해결.
- **Calibration drift**: pre-flight 6-8% absolute uncertainty가 in-flight 시간에 따라 누적된 contamination + detector aging으로 ~factor-of-2 까지 변동. SERTS underflights와 cross-cal로 추적했으나 완전 해결은 어려웠음.
- **Telemetry vs. coverage trade**: NIS 1024×512 12-bit full frame을 5+ 분에 한 번 down한다는 것은 fast event capture가 어렵다는 의미. SUMER/EIT와 사전 협조가 필수였다.
- **Astigmatic GIS의 영상**: GIS pinhole + raster 영상은 시간이 길고 평균화. 실제 dynamic 연구는 거의 NIS에 의존, GIS는 spectroscopy만.

These shortcomings informed the next-generation designs: EIS chose pure normal-incidence multilayers (no astigmatism penalty), SPICE chose NIS-only (drop GIS-style overhead), and both adopted active-pixel sensors (no MCP intensifier, less straylight, more dynamic range).

이러한 제약은 후속 설계에 반영되어, EIS는 multilayer로 normal incidence를 EUV에 적용해 astigmatism을 제거했고, SPICE는 NIS-style만 채택해 단순화했다.

---

## 3. Key Takeaways / 핵심 시사점

1. **이중 분광기 = wide coverage + imaging의 절충 / Dual-spectrometer architecture as a coverage-vs-imaging compromise** — EUV에서 normal-incidence는 ~300 Å 이하에서 무용지물(반사율 ~0)이므로 hottest coronal lines (Fe IX-XIV at 169-220 Å)는 grazing-incidence 필수. 그러나 grazing은 toroidal grating으로 stigmatic 만들기 어려워 영상이 흐림. CDS는 (1) 2-band stigmatic NIS와 (2) 4-band astigmatic GIS를 결합해 둘을 모두 얻음. 이는 EUV 분광기 설계에서 항상 부닥치는 trade-off의 모범 답안. — Below 300 Å, normal-incidence reflectivity vanishes, forcing grazing geometry for the hottest coronal lines (Fe IX-XIV, 169-220 Å); but grazing-incidence Rowland-circle designs are astigmatic. CDS solves the trade-off by combining stigmatic NIS (308-381 / 513-633 Å) with astigmatic GIS (151-785 Å, 4 bands). This is a classic resolution-coverage compromise solved by parallelization rather than choice.

2. **선비율 진단을 instrument 설계에 직접 반영 / Line-ratio diagnostics drive line list, which drives wavelength bands** — Tables I-IV는 단순한 catalog가 아니라 GIS 4 channels의 정확한 위치를 결정한다. Mg VII 280/319 (밀도, both in GIS1-2), Fe XII 186/196 (밀도, GIS1), Si XI 580/604 (온도, GIS3), Mg IX 706/750 (온도, GIS4) — 각 채널이 의도된 진단 페어를 동시에 잡도록 grating 분산이 설계됨. NIS 308-381 / 513-633 Å은 Mg IX 368, Fe XV 284 (GIS overlap), He II 304/584, O V 630, Mg X 625 등을 stigmatic하게 잡도록 선택. — Tables I-IV are not a catalogue but a constraint on the four GIS bands and two NIS bands. Each band is positioned so that critical density-sensitive (Mg VII 280/319, Fe XII 186/196) or temperature-sensitive (Si XI 580/604, Mg IX 706/750) pairs fall within a single detector — eliminating cross-calibration uncertainty between detectors for ratio measurements.

3. **MCP+SPAN과 MCP+CCD의 상보적 검출기 선택 / Complementary detector technologies for the two spectrometers** — GIS는 1-D photon-counting (high QE in EUV, 11-bit position via SPAN, 4.75×10⁵ c/s 한계). NIS는 2-D integrating (MCP intensifier amplifies → P-20 phosphor → CCD reads 1024² @ 21 μm, dynamic range 2000:1). 광자수가 적은 hottest 라인은 photon-counting GIS에 유리, 밝은 transition region 라인은 integrating NIS가 적절 — 검출기 선택 자체가 과학 우선순위 반영. — GIS uses photon-counting MCP+SPAN (high EUV QE, fast response, but Poisson-limited dead-time at 4.75×10⁵ c/s); NIS uses MCP-intensified Tektronix CCD (integrating, 2000:1 dynamic range per exposure). Photon-counting suits faint hot lines; integrating suits bright transition-region lines — detector choice mirrors the scientific priorities of each spectrometer.

4. **Lyα 1216 Å이 EUV instrument의 적인 이유와 대응 / Lyman-α as the dominant straylight nuisance** — Lyα가 어떤 CDS line보다 ~20배 강하고, 9-610 Å 모든 코로나 라인 합보다 8배 강함. Detector는 1216 Å에 직접 둔감하지만 산란된 Lyα가 photoelectron-emitting 표면을 때리면 응답함. 대응: zero-order baffling, Alochrome-treated Al, inter-detector baffles, NIS의 long-tube 차폐. 이는 향후 모든 EUV/UV instrument의 표준 설계 원칙이 됨. — H I Lyα at 1216 Å is the dominant straylight contaminant — ~20× brighter than any single CDS line. Detectors are blind to 1216 Å directly but Lyα hitting low-work-function surfaces ejects photoelectrons. Mitigations (zero-order baffles, Alochrome-treated Al, inter-detector baffles, long-tube NIS) became standard EUV instrument practice.

5. **BESSY-traceable 절대 보정과 그 한계 / BESSY-traceable absolute calibration and its limits** — Pre-flight: BESSY synchrotron (calculable photon flux from storage-ring parameters) → hollow-cathode transfer source → end-to-end CDS illumination → 6-8% absolute (1σ). 그러나 CDS aperture 전체를 균일하게 비추지 못해 5 mm collimated beam을 raster해야 했고, 이는 systematic을 만든다. In-flight: quiet-Sun 안정 비율 모니터, GIS-NIS overlap cross-cal, SERTS rocket underflight. — Pre-flight chain: BESSY synchrotron → hollow-cathode transfer source → end-to-end CDS illumination at RAL achieves 6-8% absolute (1σ). The 5-mm collimated beam couldn't fill the full aperture, requiring raster-mapping that introduced systematics. In-flight monitoring relies on quiet-Sun stability, GIS-NIS overlap, and rocket underflights — establishing the standard EUV calibration paradigm.

6. **검출기 dead-time이 광자속 한계를 정한다 / Dead-time sets the photon-rate limit** — GIS extending dead-time $T_w + T_p = 2.1$ μs → max input rate $4.75\times 10^5$ c/s에서 $1/e$ throughput. flare나 활동영역에서 brightest 라인은 이 한계에 접근. Solution: 작은 슬릿 + 짧은 노출 + 하드웨어 upper-level discriminator $8\times 10^7$ e⁻로 over-energetic 이벤트 reject. — The GIS extending dead-time ($T_w = 1.5\,\mu$s, $T_p = 0.6\,\mu$s, total $2.1\,\mu$s) caps useful input rate at $4.75\times 10^5$ c/s where $R_o/R_i = 1/e$. In flares/AR brightest lines saturate; mitigated with small slits, short exposures, and an upper-level discriminator at $8\times 10^7$ e⁻.

7. **JOP과 cross-instrument event triggering / JOPs and cross-instrument event triggering** — CDHS는 다른 SOHO instrument로부터 흥미로운 이벤트 좌표를 받아 즉시 mirror/slit/pointing을 그쪽으로 향함, 또 자기 이벤트도 broadcast 가능. 이 기능이 Joint Observing Programmes (JOPs)의 토대가 되어 SOHO를 단순한 12개 instrument 묶음이 아니라 진정한 통합 관측소로 만들었다. 후속 미션 (Hinode, IRIS, Solar Orbiter)도 모두 이 모델 계승. — CDHS receives event coordinates from other SOHO instruments and immediately points its mirror/slit/instrument; it can also broadcast its own events. This capability underpins SOHO Joint Observing Programmes (JOPs), turning SOHO from a bag of 12 instruments into an integrated observatory — a model inherited by Hinode, IRIS, and Solar Orbiter.

8. **CDS는 EIS, SPICE의 직접적 조상 / CDS as direct ancestor of EIS and SPICE** — Hinode/EIS (2006-)는 CDS의 single-band normal-incidence multilayer 버전 (170-290 Å, 영상 분광기). Solar Orbiter/SPICE (2020-)는 CDS의 NIS-only 버전 (700-790 / 970-1050 Å). 즉 CDS의 dual-architecture는 후속 미션에서 한 쪽씩 채택되는 형태로 분기. CHIANTI/ADAS 원자 데이터 + DEM 분석 파이프라인도 CDS가 표준화. — Hinode/EIS (2006-) is the normal-incidence-multilayer descendant (170-290 Å, single imaging spectrometer); Solar Orbiter/SPICE (2020-) is the NIS-only descendant (700-790 / 970-1050 Å). The CHIANTI/ADAS atomic-data + DEM analysis pipeline standardised by CDS remains in routine use today.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Grating equation / 그레이팅 방정식

$$
n\,\lambda \;=\; d\,(\sin\theta + \sin\alpha)
$$

- $n$: diffraction order (CDS uses $n=1$).
- $\lambda$: wavelength.
- $d$: grating spacing — $1\,\mu$m (1000 lines/mm) for GIS, $1/2400$ mm and $1/4000$ mm for NIS.
- $\theta$: angle of incidence (84.75° for GIS grazing; 7-9° for NIS).
- $\alpha$: angle of diffraction.

GIS의 경우 $\sin\theta \approx 0.996$ 으로 거의 일정. 따라서 $\lambda$가 커질수록 $\sin\alpha$가 작아져 $\alpha$가 감소 (78.7°@151Å → 66.5°@785Å, Table VI).
For GIS $\sin\theta \approx 0.996$ is nearly constant; longer $\lambda$ requires smaller $\sin\alpha$, hence $\alpha$ decreases from 78.7° at 151 Å to 66.5° at 785 Å.

### 4.2 Spectral resolving power / 분광 분해능

$$
\frac{\lambda}{\delta\lambda} \;=\; \frac{2\,\lambda\,R}{d\,\epsilon\,\cos\alpha}
$$

- $R$: Rowland circle radius — 750 mm for GIS, 743.6 / 736.4 mm for NIS.
- $\epsilon$: line-width on detector (sum of slit-projection $a/\cos\alpha$ and geometric defocus from circular-Rowland mismatch).

$\cos\alpha$가 분모에 있어, $\alpha$가 클수록 분해능이 떨어진다. 또한 $\epsilon$이 작은 슬릿일수록 분해능이 높지만 throughput이 감소.
$\cos\alpha$ in the denominator means resolution falls as $\alpha$ rises; small slits give better resolution but worse throughput.

### 4.3 Throughput / count-rate equation / 처리량·계수율 방정식

$$
\boxed{\;\text{count/s} \;=\; \frac{I\,\lambda \, a\,b\, A\,\epsilon_t\,\epsilon_m\,\epsilon_g\,\epsilon_d}{h\,c\,f^2}\;}
$$

- $I$: solar surface intensity (erg cm⁻² s⁻¹ sr⁻¹).
- $a\times b$: slit linear dimensions (cm).
- $A$: telescope geometric area (cm²) — 47 cm² (GIS), 34.3 cm² per grating (NIS).
- $\epsilon_t,\epsilon_m,\epsilon_g,\epsilon_d$: telescope reflectance, scan-mirror reflectance, grating efficiency, detector DQE.
- $f$: telescope focal length (cm) — 257.831 cm.
- $h c / \lambda$ converts erg → photon.
- $a b R^2/f^2$ converts slit FOV to solid angle on Sun (with $R$ = Earth-Sun distance — implicit in original derivation; final form simplifies to $abA/f^2$ for pixel-on-sun).

Pre-build estimate for GIS-2 at $\lambda \sim 30$ nm:

$$
\epsilon_t \cdot \epsilon_m \cdot \epsilon_g \cdot \epsilon_d \;=\; 0.25 \times 0.80 \times 0.03 \times 0.10 \;=\; 6\times 10^{-4}
$$

측정값은 $1$-$2\times 10^{-4}$ (Fig. 17), 즉 예측의 ~25% 수준 — 통상 EUV instrument에서 acceptable.
Measured value $1$-$2\times 10^{-4}$ is ~25% of the optimistic prediction — typical for EUV experiments.

### 4.4 Detector dead-time / 검출기 사장시간

For "extending" deadtime:

$$
R_o \;=\; R_i\,\exp\!\bigl[-R_i\,(T_w + T_p)\bigr]
$$

Maximum output: $\dfrac{dR_o}{dR_i} = 0 \Rightarrow R_i = \dfrac{1}{T_w+T_p}$. At this rate $R_o/R_i = 1/e \approx 0.368$.

For GIS: $T_w = 1.5$ μs, $T_p = 0.6$ μs → $R_i^{\max} = 4.75\times 10^5$ c/s.

### 4.5 Doppler shift sensitivity / 도플러 감도

$$
v \;=\; c\;\frac{\Delta\lambda}{\lambda}
$$

NIS at $\lambda = 360$ Å: single-pixel $\delta\lambda = 0.08$ Å → $v_{\min} \approx 67$ km/s. With centroiding (sub-pixel, ~1/10 pixel typical), reaches ~6-7 km/s — ample for HIVEL study targets at "tens of km/s".

### 4.6 Optically-thin EUV emissivity / 광학적으로 얇은 EUV emissivity (background)

$$
\varepsilon \;=\; n_e\,n_H\,G(T,\,n_e)
$$

Line intensity along LOS:

$$
I \;=\; \frac{1}{4\pi}\int n_e\,n_H\,G(T,\,n_e)\,dh \;=\; \frac{1}{4\pi}\int G(T,\,n_e)\,\mathrm{DEM}(T)\,\frac{dh}{dT}\,dT
$$

For two lines $A$, $B$ from the same ion, $n_H/n_e$ is constant and the spatial integral cancels — ratio $I_A/I_B$ depends only on atomic physics ($G_A/G_B$). For density-sensitive pairs $G$ depends on $n_e$ via collisional de-excitation; for temperature-sensitive pairs $G$ has different $T$-peaks.

This "ratio cancels DEM" property is what makes line-ratio diagnostics robust — and is why CDS's whole line-list strategy is built around chosen ratio pairs.
이 "ratio가 DEM을 cancel" 성질이 line-ratio 진단을 강력하게 만들고, CDS의 라인 리스트 전략의 토대.

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1962  OSO I — 첫 EUV 관측 / first EUV solar observations
1972  OSO VII — 초기 EUV 분광 / early EUV spectroscopy, ~20"
1973  Skylab S082A/B — slitless spectroheliograph (사진건판)
1985  Spacelab II / CHASE — 단기간 코로나 He abundance experiment
1986  CDS Science Team meetings start (Patchett PI)
1989  Patchett, Harrison, Sawyer et al. — early CDS concept (ESA SP-1104)
1992  CDS design papers (Harrison & Sawyer — 1st SOHO Workshop)
1993  Harrison "Blue Book" — CDS Science Report (RAL SC-CDS-RAL-SN-93-0007)
1994  CDS pre-flight calibration at BESSY/RAL (Feb-Mar)
►1995 THIS PAPER — Harrison et al., Solar Physics 162, 233-290
1995-12 SOHO launched (December 2)
1996  SOHO L1 insertion; CDS NIS first light April 1996
1998  SOHO accident — recovered after months of dark
1998+ NIS-1 detector progressively damaged by accumulated solar UV
2006  Hinode/EIS — direct successor (170-290 Å, single imaging spec)
2014  CDS turned off (CDS retired with SOHO)
2020  Solar Orbiter/SPICE — NIS-only descendant (700-790 / 970-1050 Å)
2027  Solar-C/EUVST — next-generation EUV spectrometer
```

CDS는 instrument paper로서 SOHO 이전의 "단발성 EUV experiment" 시대와 SOHO 이후의 "장기 운용 코로나 분광기" 시대 사이의 분기점이다. 그 후 Hinode/EIS (2006-)와 Solar Orbiter/SPICE (2020-)는 CDS의 dual-architecture를 한 쪽 (각각 NIS 또는 GIS-style)으로 specializing해 발전시킨 형태다.
CDS marks the watershed between sporadic short-duration EUV experiments (OSO, Skylab S082, CHASE, SERTS) and continuous long-duration coronal spectroscopy. Hinode/EIS and SPICE are direct architectural descendants, each specializing one half of CDS's dual layout.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Domingo, Fleck & Poland (1995) "The SOHO Mission: An Overview" | SOHO 전체 미션 개요 — CDS의 형제 12개 instrument 맥락 / SOHO mission overview | CDS는 SOHO 12 instrument 중 하나, JOPs로 SUMER/EIT/UVCS와 통합 운용 |
| Wilhelm et al. (1995) "SUMER: Solar UV Measurements of Emitted Radiation" | SUMER 자매 instrument — 800-1610 Å 보완 / sister UV spectrometer 800-1610 Å | CDS-SUMER overlap region (~800 Å)에서 cross-cal; 함께 transition region 다이내믹스 연구 |
| Delaboudinière et al. (1995) "EIT" | EIT imager — 4 narrow-band telescopes / EUV imager on SOHO | CDS JOP partner; EIT가 광역 영상으로 target 식별, CDS가 그 위치 분광 진단 |
| Culhane et al. (2007) "EIS on Hinode" | Direct successor — normal incidence multilayer / EIS는 CDS의 직접 후계자 | EIS는 170-290 Å normal-incidence multilayer; CDS dual-arch의 GIS half를 발전시킴 |
| SPICE Consortium / Anderson et al. (2020) "SPICE on Solar Orbiter" | NIS-only descendant / NIS-only 후계자 | SPICE는 Solar Orbiter의 NIS-style EUV 분광기 (700-790, 970-1050 Å); CDS의 NIS 절반 계승 |
| Mason & Monsignori-Fossi (1994) "EUV emission from coronal plasmas" | Atomic physics foundation for line ratios / 라인비율의 원자물리 토대 | Tables I-IV의 line-ratio 후보 선정의 학문적 근거; CDS analysis pipeline의 ADAS/CHIANTI 사용 |
| Vernazza & Reeves (1978) "Extreme UV composite spectra" | Solar EUV reference spectra used to predict count rates / 예상 계수율 산출의 reference | CDS Table XI의 expected count rates 산출 기반 |

---

## 7. References / 참고문헌

- Harrison, R. A., Sawyer, E. C., Carter, M. K., et al. (1995). "The Coronal Diagnostic Spectrometer for the Solar and Heliospheric Observatory", *Solar Physics* **162**, 233-290. DOI: 10.1007/BF00733431. *(this paper)*
- Harrison, R. A. (1993). "The CDS Science Report — Blue Book", RAL Publication SC-CDS-RAL-SN-93-0007.
- Harrison, R. A. & Sawyer, E. C. (1992). "The Coronal Diagnostic Spectrometer for SOHO", *Proc. 1st SOHO Workshop*, ESA SP-348, 17.
- Patchett, B. E., Harrison, R. A., Sawyer, E. C., et al. (1989). "CDS — The Coronal Diagnostic Spectrometer for SOHO", ESA SP-1104, 39.
- Domingo, V., Fleck, B., Poland, A. I. (1995). "The SOHO Mission: An Overview", *Solar Physics* **162**, 1-37.
- Wilhelm, K., Curdt, W., Marsch, E., et al. (1995). "SUMER — Solar Ultraviolet Measurements of Emitted Radiation", *Solar Physics* **162**, 189-231.
- Delaboudinière, J. P., et al. (1995). "EIT: Extreme-Ultraviolet Imaging Telescope for the SOHO Mission", *Solar Physics* **162**, 291-312.
- Hollandt, J., Kühne, M., Huber, M. C. E. (1993). "Hollow cathode transfer standards for the radiometric calibration of VUV telescopes of SOHO", *Metrologia* **30**, 381.
- Breeveld, A. A., Edgar, M. L., Smith, A., Lappington, J. S., Thomas, P. D. (1992). "A SPAN MCP detector for the Coronal Diagnostic Spectrometer", *Rev. Sci. Inst.* **63**, 1.
- Mason, H. E. & Monsignori-Fossi, B. C. (1994). "EUV emission from coronal plasmas", *Proc. 3rd SOHO Workshop*, ESA SP.
- Vernazza, J. E. & Reeves, E. M. (1978). "Extreme ultraviolet composite spectra of representative solar features", *Astrophys. J. Suppl.* **37**, 485.
- Schmidt, M., Dinger, U., Petasch, T., Trebstein, F. (1994). "Wolter-Schwarzschild solar telescope. CDS flight model manufacturing and assembly", *SPIE Proc.* **2210**, 383.
- Thompson, W. T., Poland, A. I., Sigmund, O. W., Swartz, M., Leviton, D. B., Payne, L. J. (1992). "Measurements of an intensified CCD detector for SOHO", *SPIE Proc.* **1743**, 464.
- Boucarut, R. A., Leviton, D. B., Thomas, R. J., Madison, T. (1993). "Evaluation of the EUV toroidal diffraction gratings for SOHO/CDS", *SPIE Proc.* **2011**, 565.
- Summers, H. P. (1994). "Atomic Data and Analysis Structure: User Manual", JET Publication IR(94)06.
- Riehle, F. & Wende, B. (1985). "Electron storage ring BESSY as a radiometric source", *Opt. Lett.* **10**, 365.
- Culhane, J. L., et al. (2007). "The EUV Imaging Spectrometer for Hinode", *Solar Physics* **243**, 19.
- SPICE Consortium / Anderson, M., et al. (2020). "The Solar Orbiter SPICE instrument", *Astronomy & Astrophysics* **642**, A14.
