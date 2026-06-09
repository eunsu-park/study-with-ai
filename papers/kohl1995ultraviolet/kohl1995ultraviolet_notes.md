---
title: "The Ultraviolet Coronagraph Spectrometer for SOHO"
authors: ["J. L. Kohl", "R. Esser", "L. D. Gardner", "S. Habbal", "P. S. Daigneau", "E. F. Dennis", "G. U. Nystrom", "A. Panasyuk", "J. C. Raymond", "P. L. Smith", "L. Strachan", "A. A. Van Ballegooijen", "G. Noci", "S. Fineschi", "M. Romoli", "A. Ciaravella", "A. Modigliani", "M. C. E. Huber", "E. Antonucci", "C. Benna", "S. Giordano", "G. Tondello", "P. Nicolosi", "G. Naletto", "C. Pernechele", "D. Spadaro", "G. Poletto", "S. Livi", "O. von der Lühe", "J. Geiss", "J. G. Timothy", "G. Gloeckler", "A. Allegra", "G. Basile", "R. Brusa", "B. Wood", "O. H. W. Siegmund", "W. Fowler", "R. Fisher", "M. Jhabvala"]
year: 1995
journal: "Solar Physics, Vol. 162, pp. 313-356"
doi: "10.1007/BF00733433"
topic: Solar_Observation
tags: [UVCS, SOHO, coronagraph, UV-spectroscopy, solar-wind, Doppler-dimming, toric-grating, occulter, instrument-paper]
status: completed
date_started: 2026-04-25
date_completed: 2026-04-25
---

# 45. The Ultraviolet Coronagraph Spectrometer for SOHO / SOHO 자외선 코로나그래프 분광기

---

## 1. Core Contribution / 핵심 기여

The Kohl et al. (1995) paper is the definitive instrument paper for the **Ultraviolet Coronagraph Spectrometer (UVCS)** flown aboard the *Solar and Heliospheric Observatory* (SOHO). UVCS is engineered to provide the critical empirical data needed to identify and quantify the dominant physical mechanisms that **accelerate the solar wind** and **heat the extended corona** between 1.2 and 12 solar radii (R⊙). The instrument is composed of three reflecting telescopes with combined external + internal occultation, two near-twin toric-grating UV spectrometer channels (centered on **HI Lyα 1216 Å** and **O VI 1031.93/1037.61 Å**), and a visible-light polarimeter (4500–6000 Å). Together they deliver simultaneous and cospatial measurements of UV line profiles, total-line radiance for several minor ions (Mg X 610/625, Si XII 499/521, Fe XII 1242), and the polarized K-corona radiance — the three diagnostics required to derive electron density, electron temperature, ion kinetic temperatures (parallel and perpendicular to the magnetic field) and bulk outflow velocities from the **Doppler-dimming** technique.

본 논문은 SOHO 위성에 탑재된 자외선 코로나그래프 분광기(UVCS)에 대한 결정적인 장비 논문이다. UVCS는 1.2–12 R⊙ 영역에서 태양풍을 가속하고 확장 코로나를 가열하는 주요 물리 메커니즘을 식별·정량화하는 데 필요한 결정적 실증 데이터를 제공하도록 설계되었다. 외부+내부 차폐를 갖춘 세 개의 반사 망원경, HI Lyα 1216 Å과 OVI 1032/1037 Å 라인을 위한 두 개의 거의 동일한 토릭 격자 자외선 분광기 채널, 그리고 4500–6000 Å 가시광 편광계로 구성된다. 이 세 채널은 자외선 라인 프로파일, 여러 미소 이온(Mg X, Si XII, Fe XII)에 대한 총 강도, 그리고 K-코로나 편광 복사를 동시·동공간적으로 측정하여 Doppler-dimming 진단 기법으로 전자 밀도·온도, 이온 운동 온도(자기장 평행/수직 성분), 그리고 대규모 흐름 속도를 도출할 수 있게 해준다.

---

## 2. Reading Notes / 읽기 노트

### Part I: Primary Scientific Objectives (Sect. 1, pp. 314–316) / 주요 과학 목표

The paper organizes UVCS goals around four pillars / 본 논문은 UVCS의 목표를 다음 네 기둥으로 정리한다:

1. **태양풍 가속 메커니즘 / Solar-wind acceleration mechanisms** — distinguish thermal pressure (Parker-type), wave–particle interactions, and suprathermal-electron driving; test heavy-ion acceleration models that produce the variable composition of the wind. / Parker 열압력, 파동–입자 상호작용, 초열 전자 구동의 상대적 기여 식별, 태양풍 화학 조성 변동의 근원 검증.
2. **코로나 가열 / Coronal heating** — identify whether MHD-wave dissipation (especially in open-field regions), and which dissipation lengths and channels (ions vs electrons), explain the observed radial heating profile. / MHD 파 산란이 주요한지, 이온/전자 가열의 분리 여부, 그리고 반경 방향 가열 프로파일.
3. **태양풍 근원 영역 매핑 / Locating solar-wind sources** — global tomographic maps of T, n, V, particle flux from coronal hole and streamer, including polar plumes; signatures distinguishing fast and slow streams. / 토모그래피로 코로나홀·스트리머·극지 plume 등의 흐름 근원 식별.
4. **자유 매개변수 결정 / Plasma properties at the freezing-in radius** — temperatures, densities, flow velocities and abundances at 1.5–5 R⊙ where the heavy-ion ionization states freeze in, complementing in-situ observations far from the Sun. / 1.5–5 R⊙ 동결(freezing) 영역에서의 온도·밀도·흐름·풍부도 측정.

### Part II: Spectroscopic Diagnostic Techniques (Sect. 2, pp. 316–320) / 분광 진단 기법

The diagnostic suite uses **three coupled observables** from the same line of sight: / 세 가지 결합된 관측량을 동일 시선에서 사용:

- **Line profiles** of HI Lyα, Fe XII 1242, O VI 1032/1037, Mg X 610/625, Si XII 499/521. The shape encodes thermal motions, non-thermal motions (waves, turbulence), and bulk outflow projected onto the line of sight. Coronal HI Lyα is dominated by thermal protons (~130 km s⁻¹ at 10⁶ K), while heavier ions like O⁵⁺ are most sensitive to non-thermal motions. / 라인 프로파일은 열운동·비열운동·시선 속도를 모두 인코딩.
- **Total radiance ratios** to derive Doppler-dimming corrections and abundances. / Doppler dimming 보정과 풍부도 계산용 총 강도 비.
- **Visible polarized radiance** (K-corona) → electron density via Thomson-scattering inversion (Billings 1966). / Thomson 산란을 통한 전자 밀도.

**Hughes (1965) electron-scattered Lyα formula**:

$$ I_e(\lambda) = \mathrm{const}\;\int_{-\infty}^{\infty} N_e \exp\!\left[-\frac{(\lambda-\lambda_0)^2}{\Delta\lambda_e^2}\right] dx \quad (1) $$

The electron-scattered component has Δλ ~ 50 Å (FWHM), while resonantly scattered Lyα is only ~1 Å wide. By fitting the broad pedestal one obtains T_e independently of Doppler-dimming radiometric uncertainties.

전자 산란 Lyα 성분은 약 50 Å의 광폭을 가지며, 공명 산란 라인(~1 Å)을 빼면 코로나 전자 온도를 직접 측정할 수 있다(Hughes 1965).

**Doppler dimming** (Hyder & Lites 1970; Withbroe et al. 1982; Noci, Kohl & Withbroe 1987): The resonantly scattered intensity from a coronal ion depends on the convolution between the chromospheric incoming line and the local absorption profile. As the absorbing ions stream away from the Sun with bulk speed V_W, the chromospheric line is Doppler-shifted out of the absorber's profile, suppressing the intensity by a factor D_i(V_W). Equation (2):

$$ \frac{I_r}{I_{WL}} = \mathrm{const} \times A_\mathrm{el}\,\langle R_i(T_e)\rangle\,\langle D_i(V_W)\rangle \quad (2) $$

where I_r is the resonant UV line intensity, I_WL is the cospatial visible (electron-scattered) intensity that gives N_e, A_el is the elemental abundance, ⟨R_i⟩ is the ionization fraction at electron temperature T_e and ⟨D_i⟩ is the dimming factor.

Doppler dimming은 코로나 이온이 흐름 속도 V_W로 이동할 때 색구권으로부터 입사되는 공명 라인의 도플러 이동으로 인해 흡수가 줄어드는 효과이다. 식 (2)는 자외선 공명 강도와 가시광(전자 산란) 강도의 비를 풍부도, 이온 분율, dimming 인자의 곱으로 표현한다.

**OVI 1037 Å C II pumping** (Noci et al. 1987): At outflow velocities V_W ≈ 90–250 km/s, the chromospheric C II 1037.018 Å line is Doppler-shifted onto the O VI 1037.613 Å absorption profile in the rest frame of the streaming O⁵⁺ ions, *re-pumping* the line. This produces a non-monotonic dimming curve (Fig. 2 of paper) in which the OVI 1037 Å intensity drops then partially recovers. The OVI 1032/1037 Å doublet ratio thus becomes a sensitive bulk-velocity diagnostic over the range that covers the fast-wind acceleration region.

OVI 1037 Å 라인은 90 km/s 이상의 흐름에서 색구권 C II 1037.018 Å이 적색 이동되어 O VI 1037.613 Å 흡수 프로파일에 정확히 들어맞기 때문에 다시 펌핑되어 강도가 비단조적으로 변한다. OVI 1032/1037 라인 비는 이 영역에서 흐름 속도에 매우 민감하여 빠른 태양풍 가속 영역의 결정적 진단이 된다.

### Part III: Instrument Overview & Telescope (Sect. 3–4, pp. 320–330) / 장비 및 망원경 구조

UVCS hardware splits into the **Telescope-Spectrometer Unit (TSU)** and the **Remote Electronics Unit (REU)** (Fig. 3 of paper). TSU is a triple-channel occulted telescope with: aperture door (Teflon-window), serrated external occulter (knife-edge), three telescope mirrors (one per channel), independent internal occulters, sunlight trap, and the spectrometer assembly with a high-resolution spectrometer for each UV channel + the WLC polarimeter. / 장비는 TSU(망원경+분광기)와 REU(전자유닛)로 분리됨.

| Channel | Spectral Range / Function |
|---|---|
| Lyα channel | Toric grating spectrometer with neutral-density filter, optimized for HI Lyα 1216 Å profile (1145–1287 Å, extended to 1100–1361 Å). Detector: KBr+MCP+XDL with optically flat MgF₂ window. |
| OVI channel | Toric grating, optimized for OVI 1032/1037 Å, also covers 984–1080 Å (1st order) and 492–540 Å (2nd order). Convex internal mirror folds Lyα + Mg X (as 2nd order) onto the same OVI XDL — the "redundant Lyα path". |
| White Light Channel (WLC) | Visible polarimeter, broadband 4500–6000 Å, photomultiplier (EMI 9130B). Achromatic Pancharatnam λ/2 retarder + Polaroid HN38S linear polarizer + bandpass filter. |

The instantaneous FOV (Fig. 4) is a 40-arcmin-long slit segment, with the WLC FOV (14"×14") centered. The internal mirror motion steps across the 141-arcmin primary FOV; the roll/pointing mechanism scans the radial direction (1.2–12 R⊙). Total roll range ±179.75°.

순간 시야는 40분 길이의 슬릿이며, WLC는 14×14 arcsec를 본다. 내부 거울의 회전으로 141 arcmin 방향을 스캔하고, 롤·포인팅 메커니즘으로 1.2–12 R⊙의 반경 방향을 스캔한다.

**Table I (typical observation specs)** — HI 1216 Å profile: ΔλFWHM = 0.23 Å, spatial 12"×15" to 24"×24"; OVI 1032/1037 profile: 0.15 Å, 12"×5'.

**Unvignetted area (Eq. 3)**:

$$ A = h\,D\,\tan[16/60\,(r-1.2)] - b $$

The mirror has h = 72 mm × 50 mm (UV) or 72 × 30 mm (WLC), focal length 750 mm, spherical figure with R = 1502 mm. Surface roughness specs: 8 Å rms (UV), 5 Å goal 3 Å (visible); SiC-clad SiC for UV (~45% reflectance at 1216 Å), Cr-coated Si-clad SiC for visible (~65%).

**Stray-light triple defense** = External occulter (geometric block of disk) → Internal occulter (blocks edge-diffracted light reflected by the mirror) → Sunlight trap (3-cavity black multilayer Be-Ni cavity at 20° internal angles, achieving ≥7 specular bounces, with reflectance < 3% per UV bounce / < 10% per visible bounce; net irradiance suppression 2×10⁻⁵ UV, 5×10⁻⁷ visible).

세 단계의 산란광 억제: 외부 차폐 + 내부 차폐 + sunlight trap. 7회 이상의 specular 반사를 거쳐 자외선에서는 2×10⁻⁵, 가시광에서는 5×10⁻⁷ 수준의 일사도 비로 억제.

A four-photodiode **Sun sensor** integrated into the trap allows ±15 arcmin orientation sensing and ±4″ fine pointing.

### Part IV: Spectrometer Assembly (Sect. 5, pp. 331–343) / 분광기 조립체

**Why toric gratings?** A toric grating has different curvature radii in the dispersion (R_h) and spatial (R_v) directions. The horizontal (spectral) focus lies on the Rowland circle of diameter R_h, while the vertical focus lies elsewhere unless

$$ R_v / R_h = \cos\alpha \cdot \cos|\beta_o| \quad (4) $$

is satisfied — at this point both foci coincide at the **stigmatic points ±β_o**. Effective stigmatic imaging is possible across a band 2 R_h β_o around the central wavelength. Astigmatism correction is essential because UVCS records 2D images: dispersion (X) × spatial along-slit (Y).

토릭 격자는 분산과 공간 방향의 곡률 반경이 다른 격자로, R_v/R_h = cos α cos|β_o| 조건을 만족할 때 ±β_o 회절각에서 분광 초점과 공간 초점이 동시에 일치하는 stigmatic 점을 형성한다.

**Table II — Optical parameters**:

| Parameter | Lyα channel | OVI channel |
|---|---|---|
| Ruling frequency | 2400 l/mm | 3600 l/mm |
| Angle of incidence α | 12.85° | 18.85° |
| Angle of diffraction β | 3.98° | 2.78° |
| R_h | 750 mm | 750 mm |
| R_v | 729.5 mm | 708.9 mm |
| Reciprocal dispersion (1st order) | 5.54 Å/mm | 3.70 Å/mm |
| Spectral bandwidth/pixel | 0.14 Å | 0.0925 Å |
| Spatial size/pixel | 7" (25 µm) | 7" (25 µm) |

**Grating implementation** — Lyα: holographic on flexible substrate, deformed into toric, then replicated. OVI: mechanically ruled. Coatings: Lyα = Al+MgF₂; OVI = Iridium (Ir). Lab efficiencies: 23% (Lyα), 10% (OVI). The 2400/3600 l/mm choice in OVI also enables 2nd-order use to access Mg X (610/625 Å) and Si XII (499/521 Å). Fig. 10 of the paper shows ray-tracing blurs vs wavelength: spatial blur < pixel for y ≤ 1 mm from slit center; spectral blur < 50 µm except at ±extreme wavelengths.

**Slit widths**: 355, 213, 53, 25, 10 µm (height 8.73 mm), with mechanism repeatability ±1 µm.

**Filter inserter** for solar-disk on-orbit calibrations: 10⁻³ attenuation at 1216 Å (UV) and 4500–6000 Å (WLC). Fail-safe "out" rest position.

**Grating mechanism** (Table III): rotation range ±0.5° = ±4 Å spectral shift on any pixel; placement precision 0.014 Å, stability 0.005 Å during an observation, step size 0.005 Å. Voice-coil actuator with two pairs of differential impedance transducers.

**XDL detector (Sect. 5.6)** — Two virtually identical KBr-coated Z-stack MCPs with cross-delay-line anode, 26 mm × 9 mm active area digitized to **1024 × 360 pixels** (6.5 spectral × 4 spatial pixels per millimeter, ~25 µm pixels). Charge gain ~2×10⁷. Position is decoded from arrival-time differences along delay lines. The Lyα detector has an MgF₂ window (cuts < 1100 Å); the OVI detector is open-faced with a 90% transmission +15 V mesh in front. Background < 1 event cm⁻² s⁻¹. Single-pixel rates up to 100 events/s; global rates up to 5×10⁵ s⁻¹ with < 40% dead time. Stability < 1 pixel over expected on-orbit T range. Pulse-height < 30% FWHM (OVI), enabling background discrimination.

**Table IV — Quantum detection efficiencies (KBr)**: ~17–18% at 1216 Å, peak ~33% near 490 Å. Cuts off at 1600 Å (solar-blind).

**Image processors (Sect. 5.7)** — Each detector has an Image Processor with two RAM tables (X and Y) acting as a programmable detector mask, plus an accumulator array. Mask = (1024 AX values, 360 AY values); ⟨nonzero⟩ pixels are binned into superpixels. Up to 6 masks per detector stored in REU; up to 5 panels of arbitrary location per mask. Photon arrival rate up to 10⁶ s⁻¹.

**WLC (Sect. 5.8)** — 50 µm × 50 µm entrance pinhole; achromatic Pancharatnam λ/2 retarder rotated to three positions separated by 30°; HN38S Polaroid linear polarizer; lens (f = 124.8 mm); bandpass filter 4500–6000 Å; EMI 9130B PMT (S-20 cathode, QE > 15%). Linearly polarized radiance:

$$ pI = \frac{4}{3}\sqrt{I_o^2 + I_+^2 + I_-^2 - I_o I_+ - I_o I_- - I_+ I_-} \quad (5) $$

Subtraction of common terms cancels the unpolarized F-corona contribution.

### Part V: Stray-Light Suppression (Sect. 6, pp. 343–347) / 산란광 억제

Stray-light suppression is the **defining challenge** of any coronagraph. UVCS targets coronal radiances 6–9 orders of magnitude below the disk irradiance (Fig. 13). Sources (Table VI):

- Sunlight trap scatter (knife edge + surfaces);
- External-occulter diffraction reaching the entrance slit via mirror surface, mirror edge, internal occulter;
- Multiple non-specular reflections off structural elements, aperture secondary edges, off-band stray light, light leaks.

Key result (Fig. 14): At Lyα the dominant stray-light contribution is non-specular reflection from the mirror surface (mirror surface roughness 8 Å rms is the binding constraint). For visible, internal-occulter dominates up to 2.5 R⊙ and mirror-surface above. The white-light polarizer further suppresses unpolarized stray light.

**Lyα off-band suppression**: Holographic grating with > 10⁴ rejection in any 5 Å band 4–100 Å away from primary diffraction angle. An opaque blocker strip at the grating-launch-lock position blocks the resonant Lyα line during e-scattered Lyα profile measurements (~50 Å pedestal) up to 4 R⊙. Net off-band reduction > 1×10⁻⁶ in the near UV, > 1×10⁻⁹ in the visible.

산란광 억제는 코로나그래프의 가장 중요한 기술적 요건이다. UVCS는 디스크 대비 약 10⁻⁸ 수준의 산란광을 목표로 한다. Fig. 14의 산란광 budget에서 Lyα는 거울 표면 비-경면 반사가, 가시광은 내부 occulter가 주요 기여원이다. Holographic Lyα 격자와 blocker strip으로 off-band 신호를 자외선에서 10⁶, 가시광에서 10⁹ 이하로 억제한다.

### Part V-bis: Detector and Image-Processor Implementation Details / 검출기와 이미지 프로세서 세부

The XDL design choice deserves careful note because it shapes every UVCS observation. The two-dimensional cross delay-line readout decodes the (X,Y) photon centroid from the *time difference* of pulses propagating in opposite directions along orthogonal serpentine delay lines. Centroid resolution is set by the delay-line discriminator timing jitter, not by the underlying pixel pitch — which is why the active 26 mm × 9 mm detector area can be sub-divided into 1024 × 360 logical pixels (~25 µm × 25 µm each). The Z-stack of MCPs gives total charge gain ~2×10⁷, large enough that single photoelectron events are easily discriminated from background. The KBr photocathode is solar-blind beyond ~1600 Å, so background from out-of-band light leaks is suppressed at the photocathode itself. Background rate < 1 event cm⁻² s⁻¹ is dominated by ⁴⁰K beta-decay in the Philips MCP glass — already very low (Siegmund et al. 1988; Fraser et al. 1987).

XDL는 평행 지연선의 펄스 도착 시간 차이로 광자 위치를 계산하는 방식이라 픽셀 피치보다 분해능이 더 좋다. KBr 광음극은 1600 Å 이상에서 거의 반응하지 않아 자체 솔라 블라인드 특성을 갖고 있으며, 배경은 약 1 event/(cm²·s) 수준이다.

The **image processor** layer is conceptually a programmable detector mask: each X coordinate (1–1024) maps via the X RAM to an address `AX` (0 = reject), each Y coordinate (1–360) maps to `AY`. If both are nonzero, the concatenated address `A` increments the accumulator. This permits the operator to define up to five rectangular "panels" per mask, with arbitrary spatial/spectral binning, so the science telemetry budget (~133 bps for SOHO) can be focused on the spectral lines of interest while skipping detector regions devoid of useful signal. Up to 6 masks per detector can be cached in REU memory.

이미지 프로세서는 사실상 프로그램 가능한 마스크로, X·Y 좌표를 RAM 주소로 변환하여 0이면 무시, 비-0이면 누적기에 카운트한다. 이를 통해 한 번의 노출에서 자외선 라인 영역만 미세하게 비닝하고 나머지는 거칠게 비닝하여 133 bps 텔레메트리 예산을 효율적으로 분배한다.

### Part V-ter: Calibration and Stability Strategy / 보정·안정성 전략

UVCS supplements ground calibration (vacuum chamber, monochromator-fed beam, photodiode irradiance scan) with on-orbit cross-calibration: (i) the **filter inserter** (10⁻³ neutral density) lets UVCS occasionally point at the solar disk for end-to-end response checks; (ii) the **stimulation pulser** in the XDL electronics injects a synthetic spot at the centre of each delay line at ~40 Hz to monitor electronics health; (iii) the **redundant Lyα path** in the OVI channel cross-checks the primary Lyα channel at 1216 Å; (iv) the WLC photomultiplier dark-count rate (~180 cps) is monitored to track gain/temperature drifts. Long-term stability is enabled by the closed-loop thermal control (89.7 W non-operational + 41.2 W operational heaters) with PI control architecture on three PCBs.

UVCS의 보정 전략은 진공 챔버 지상 보정 + 궤도상 디스크 직접 관측(filter inserter) + XDL 전기적 자체 시험 + redundant Lyα 채널 + WLC 다크 카운트 모니터링의 다중 안전망으로 구성된다.

### Part VI: Performance, Operations, Data (Sect. 7–11, pp. 347–354) / 성능, 운영, 데이터

**Table VII — Measured performance**:

| Channel | λ (Å) | Efficiency | Stray light | Spectral res. (Å) | Spatial res. ('') |
|---|---|---|---|---|---|
| Lyα | 1216 | 0.002 | < 1×10⁻⁸ | 0.23 | 15 |
| Lyα off-band | 2537 | < 3×10⁻⁹ | — | — | — |
| Redundant Lyα | 1216 | 0.001 | < 5×10⁻⁸ | 0.31 | 15 (inferred) |
| OVI | 1032 | 0.0035 | < 5×10⁻⁸ (from redundant Lyα) | < 0.24 | 15 (inferred) |
| Visible (WLC) | 5460 | 0.004 | < 1.5×10⁻⁸ | 4500–6000 | 15 |

Calibration was done in a vacuum chamber (Fig. 15) with light trap simulating low-radiance corona, monochromator at the collimator focus, and translation-stage photodiodes for absolute irradiance. Aperture limits set the calibration to 2.7 R⊙.

성능은 진공 챔버에서 측정되었으며 Lyα 채널 효율 0.002, OVI 0.0035, WLC 0.004를 달성. Stray-light는 모두 1×10⁻⁸ 수준 이하.

**Electronics (Sect. 8)** — Two redundant CPUs (80C86, 5 MHz, 4K/128K/32K of PROM/EEPROM/RAM each), 133 bps SOHO telemetry. **Thermal (Sect. 9)** — passive (MLI, ITO-coated film, silvered Teflon) + active heaters (89.7 W non-operational, 41.2 W operational). **Flight software (Sect. 10)** — observation sequences pre-uplinked to memory; can run interactively only during contact. **Operations (Sect. 11)** — three command paths (SMOCC, IWS, on-board sequence); data products: spectral FITS files, visible-light files, calibration parameters, image files, auxiliary disk-irradiance logs, IDL data-analysis software at SAO and Italian centers.

The data-handling pipeline produces seven product classes: (1) spectral FITS files containing uncalibrated detector counts plus configuration/timing; (2) visible-light FITS files from the WLC; (3) calibration parameter files (pointing, spatial, wavelength, radiometric); (4) image data files including Lyα synoptic maps; (5) auxiliary files (disk irradiance history, instrument profile, stray light, pulse-height distributions); (6) data catalogs and the Mission Log File for operational tracking; (7) the IDL Data Analysis Software which converts raw counts to physical units (photons cm⁻² s⁻¹ Å⁻¹ ster⁻¹) using the calibration database.

데이터 산출물은 7종으로 분류되며, IDL 분석 소프트웨어가 raw count를 물리 단위(photons cm⁻² s⁻¹ Å⁻¹ ster⁻¹)로 변환한다.

### Part VII: Synthesis — Why the design works / 종합: 설계가 작동하는 이유

The UVCS design is best appreciated as the *intersection* of three independent constraints: (i) the science goals require simultaneous spectroscopy of HI Lyα, OVI doublet, and visible polarization at the same pointing — hence three coupled telescopes pointed by a single shared mechanism; (ii) UV reflectance is severely limited by available coatings (~45% for SiC at 1216 Å), so the optical chain must be minimized: a *single* aspheric (toric) grating performs reflection, dispersion, focusing and astigmatism correction in one bounce; (iii) coronal-to-disk radiance ratios of 10⁻⁶ to 10⁻⁹ demand stray-light suppression that only a multi-stage occulter chain can provide, with each stage attacking a different scatter mechanism (geometric blocking → diffraction blocking → trap absorption → grating off-band rejection → polarizer F-corona rejection). The toric grating, MCP+XDL detector, and triple-occulter geometry are choices that simultaneously satisfy *all three* constraints — which is why the same architecture has propagated, with refinements, into Metis on Solar Orbiter and into ASPIICS on PROBA-3.

UVCS의 설계는 (i) 과학 목표가 요구하는 동시 다채널 분광·편광 관측, (ii) 자외선에서의 낮은 반사율로 인한 광학 체인 최소화, (iii) 10⁻⁸ 수준의 산란광 억제 요구라는 세 제약의 교차점에 위치한다. 토릭 격자, MCP+XDL 검출기, 다단계 차폐 기하 모두가 이 세 제약을 동시에 만족시키는 선택이며, 그래서 Solar Orbiter Metis나 PROBA-3 ASPIICS에 그대로 계승되었다.

---

## 3. Key Takeaways / 핵심 시사점

1. **Triple-channel design enables the 3 simultaneous diagnostics needed for Doppler dimming.** UVCS uniquely combines HI Lyα profile, O VI doublet, and visible K-corona polarimetry in a single instrument so that ⟨D_i(V_W)⟩, ⟨R_i(T_e)⟩, and N_e are all measured cospatially. / 세 채널이 동시에 같은 시선을 관측하기 때문에 Doppler dimming에 필요한 세 측정량(흐름 인자, 이온 분율, 전자 밀도)을 하나의 장비에서 일관되게 얻을 수 있다.
2. **Toric Rowland-Onaka geometry is the optical key** to imaging-quality UV spectra at low reflectance wavelengths: a single aspheric surface achieves diffraction + reflection + focusing + astigmatism correction, maximizing efficiency. / 토릭 격자가 저반사율 자외선에서 단일 비구면 표면으로 회절·반사·집속·비점수차 보정을 동시에 달성하는 핵심 광학 기술이다.
3. **Stray-light suppression to ≲ 10⁻⁸** of disk irradiance is achieved through a chain of *external occulter → internal occulter → sunlight trap → entrance-slit baffle → holographic grating → blocker strip → polarizer*. Each stage attacks a different scatter or diffraction pathway. / 외부·내부 차폐, sunlight trap, 슬릿 baffle, holographic 격자, blocker strip, 편광기로 이어지는 여섯 단계의 산란광 억제로 디스크 대비 10⁻⁸ 수준을 달성한다.
4. **C II pumping turns the OVI 1037 Å line into a non-monotonic velocity diagnostic** (Fig. 2). The 1032/1037 Å ratio breaks the degeneracy between line-of-sight kinetic broadening and bulk outflow that single resonant lines suffer from. / OVI 1037 Å의 C II 펌핑으로 라인 강도가 비단조적으로 변하기 때문에 1032/1037 비율이 시선 운동 폭과 대규모 흐름의 축퇴(degeneracy)를 깨는 결정적 진단이 된다.
5. **HI Lyα profile width is dominated by proton thermal motions** (~130 km/s at 10⁶ K), while heavier ions (OVI, MgX) are dominated by non-thermal motions. Combining both lets UVCS separate kinetic temperature anisotropy (T_⊥ vs T_∥) from non-thermal wave amplitudes. / HI Lyα 폭은 양성자 열운동, 무거운 이온은 비열 운동이 지배하므로 두 라인을 같이 관측하면 운동 온도의 비등방성과 파동 진폭을 분리해낼 수 있다.
6. **Electron temperature is measured from the Thomson-scattered Lyα pedestal** (Hughes 1965; Eq. 1), independent of UV-radiometric uncertainties, by isolating the broad ~50 Å component beneath the resonant 1 Å core via a blocker strip. / Thomson 산란된 Lyα 광폭 성분(약 50 Å)을 blocker strip으로 분리 측정하면 전자 온도를 자외선 라디오메트리 불확실성과 독립적으로 얻을 수 있다.
7. **2D photon-counting KBr+MCP+XDL detectors (1024×360 pixels) with programmable masks** allow on-board binning into super-pixels and on-board sequence execution, accommodating the 133 bps SOHO telemetry. / 1024×360 픽셀의 KBr+MCP+XDL 검출기와 프로그래머블 마스크/이미지 프로세서가 SOHO의 133 bps 텔레메트리에 적합하도록 데이터 부피를 조절한다.
8. **UVCS is the first long-duration instrument designed to probe the 1.5–5 R⊙ "freeze-in" zone**, providing unique cross-validation of in-situ heavy-ion charge-state observations at 1 AU and serving as the empirical bedrock for the modern ion-cyclotron-resonance heating paradigm. / UVCS는 1.5–5 R⊙ "freeze-in" 영역의 장기 진단을 가능케 하는 최초의 장비로, 1 AU에서의 현장 측정과 교차 검증을 제공하여 현대 ion-cyclotron 가열 패러다임의 실증적 근거가 되었다.

### Selected Numerical Specs (Memorize) / 기억할 핵심 수치

| Quantity | UVCS Lyα | UVCS OVI | WLC |
|---|---|---|---|
| Wavelength (Å) | 1216 | 1032/1037 | 4500–6000 |
| Spectral resolution (Å, Table I/VII) | 0.23 (profile) | 0.15/<0.24 | 1500 |
| Reciprocal dispersion (Å/mm) | 5.54 | 3.70 | — |
| Detector format | XDL 1024×360 | XDL 1024×360 | EMI 9130B PMT |
| Spatial sampling (per pixel) | 7'' = 25 µm | 7'' = 25 µm | 14×14 arcsec aperture |
| Efficiency (Table VII) | 0.002 | 0.0035 | 0.004 |
| Stray-light (disk-relative) | < 1×10⁻⁸ | < 5×10⁻⁸ | < 1.5×10⁻⁸ |
| Effective field of view | 40' instantaneous | 40' instantaneous | 14×14'' |
| Range covered | 1.2–10 R⊙ (12 R⊙ extended) | 1.2–10 R⊙ | 1.5–5 R⊙ |
| Roll mechanism | ±179.75° | (shared) | (shared) |
| Pointing precision | ±4'' fine | (shared) | (shared) |

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Electron-scattered Lyα (Eq. 1) / 전자 산란 Lyα

$$ I_e(\lambda) = \mathrm{const}\,\int_{-\infty}^{\infty} N_e(x)\,\exp\!\left[-\frac{(\lambda-\lambda_0)^2}{\Delta\lambda_e^2}\right] dx $$

- N_e(x) = electron density along line of sight
- Δλ_e = thermal Doppler width of the electron-scattered profile, with electron speed v_e ~ √(2kT_e/m_e); at T_e = 1.5 MK, v_e ≈ 7000 km/s, giving Δλ ≈ 50 Å
- The integral form arises because Thomson scattering preserves the broad Doppler profile of the electrons, convolved with the chromospheric source.

전자 산란 프로파일의 폭은 전자 열속도(약 7000 km/s)에 비례하여 약 50 Å에 이르고, 적분은 시선 방향의 N_e 누적을 나타낸다.

### 4.2 Doppler-dimming intensity ratio (Eq. 2) / Doppler-dimming 비

$$ \frac{I_r}{I_{WL}} = \mathrm{const}\,\times\,A_\mathrm{el}\,\langle R_i(T_e)\rangle\,\langle D_i(V_W)\rangle $$

The dimming factor is a 3D integral

$$ D_i(V_W) = \frac{1}{\Phi_0}\int d\Omega\,\Phi(\hat n)\int d\lambda\,\phi_\mathrm{chromo}(\lambda - \lambda_0\,\hat n\cdot \vec V_W/c)\,\phi_\mathrm{abs}(\lambda) $$

with φ_chromo = chromospheric incoming line shape and φ_abs = absorber line profile in the ion's rest frame. For pure radial outflow and isothermal ions:

$$ D_i(V_W) \approx \int d\lambda\,\phi_\mathrm{chromo}(\lambda - \lambda_0 V_W/c)\,\phi_\mathrm{abs}(\lambda) $$

Doppler dimming 인자는 색구권 입사 라인 모양과 코로나 흡수 프로파일의 도플러 시프트 컨볼루션이며, 흐름 속도 V_W의 단조 감소 함수이다.

### 4.3 OVI 1037 Å with C II pumping / C II 펌핑

The local incoming radiance at the absorber 1037.613 Å frame includes both OVI 1037.613 Å and a Doppler-shifted C II 1037.018 Å contribution:

$$ \phi_\mathrm{chromo}^{tot}(\lambda') = \phi_\mathrm{OVI}(\lambda' - \lambda_{OVI}V_W/c) + \alpha\,\phi_\mathrm{CII}(\lambda' - \lambda_{CII}V_W/c) $$

where α is the C II/O VI radiance ratio at the chromospheric source. As V_W increases, the C II line slides into the O VI absorber (Δλ ≈ 0.595 Å), creating a secondary maximum near V_W ≈ 175 km/s (Fig. 2 of paper).

OVI 1037 라인은 C II 1037.018 Å이 약 0.595 Å 적색 이동하여 흡수 프로파일에 들어맞는 V_W ≈ 175 km/s 부근에서 강도 최대(2차 봉우리)가 나타난다.

### 4.4 Geometry — unvignetted area (Eq. 3) / 비비네팅 면적

$$ A = h\,D\,\tan\!\left[\frac{16}{60}(r-1.2)\right] - b $$

- h = mirror height (parallel to limb tangent), 72 mm
- D = external occulter to mirror distance
- r = heliocentric line-of-sight height in R⊙
- b = over-occulting width
- 16/60 ≈ ¼° = (R⊙/1 AU) angular size; the tangent factor expresses the geometric shadow boundary

식 (3)은 시선 높이 r이 1.2 R⊙에서 멀어질수록 거울 노출 폭이 선형적으로 증가함을 보인다.

### 4.5 Toric grating astigmatism correction (Eq. 4) / 토릭 격자 비점수차 조건

$$ \frac{R_v}{R_h} = \cos\alpha\,\cos|\beta_o| $$

For the **Lyα channel**: cos(12.85°) × cos(3.98°) = 0.9750 × 0.9976 = 0.9727. Then R_v = 0.9727 × 750 = 729.5 mm — exactly Table II. ✓

For the **OVI channel**: cos(18.85°) × cos(2.78°) = 0.9462 × 0.9988 = 0.9450 → R_v = 0.9450 × 750 = 708.7 mm vs Table II 708.9 mm. ✓ (matches within ray-tracing rounding).

Lyα 채널에서 cos(α)cos(β_o) = 0.9727이고, R_h = 750 mm이므로 R_v = 729.5 mm로 표 II와 정확히 일치한다.

### 4.6 Polarized radiance (Eq. 5) / 편광 복사

$$ pI = \frac{4}{3}\sqrt{I_o^2 + I_+^2 + I_-^2 - I_o I_+ - I_o I_- - I_+ I_-} $$

This is algebraically equivalent to the magnitude of the Stokes vector (Q,U) recovered from three retarder positions separated by 30°:

$$ Q = \frac{2}{3}(2I_o - I_+ - I_-),\quad U = \frac{2}{\sqrt{3}}(I_+ - I_-),\quad pI = \sqrt{Q^2 + U^2} $$

The unpolarized common-mode (F-corona, stray light) cancels in the differences.

식 (5)는 30° 간격의 세 retarder 위치에서 측정된 강도로부터 Stokes Q, U의 크기를 직접 계산하는 공식이며, F-corona나 산란광의 비편광 성분은 차분에서 상쇄된다.

### 4.7 Pixel quantitative budget — Lyα line profile / Lyα 프로파일 픽셀 예산

Spectral pixel = 0.14 Å (Table II); HI Lyα coronal FWHM in slow wind at 2 R⊙ ≈ 1.0 Å (Fig. 1) → ~7 pixels FWHM, well over-Nyquist. In fast wind / coronal hole at 4 R⊙ the FWHM grows to ~1.4–1.6 Å (>10 pixels) due to non-thermal motions and proton heating. Spatial pixel = 7" (25 µm) → at 1 R⊙ ≈ 950" the spatial sampling is r/Δr ≈ 135 — sufficient to resolve polar plumes. With slit 53 µm width × 8.73 mm height, the projected sampling element is ~15"×15" (matches Table I). Photon throughput at 1216 Å with η = 0.002 (Table VII) and a typical disk-equivalent coronal radiance of 10⁻⁶ × disk gives ~10² counts pixel⁻¹ s⁻¹ — adequate for 100-s exposures.

Lyα 채널의 분광 픽셀(0.14 Å)은 코로나 라인 폭 ~1 Å에 비해 충분히 미세하여 (Nyquist 만족) 라인 프로파일을 정확히 샘플링할 수 있고, 공간 픽셀(7")은 1 R⊙ 직경의 약 1/135로 극지 플룸 분해에 충분하다.

### 4.8 Worked example — line profile of streaming O⁵⁺ ions / 흐름 OVI 라인 프로파일 예제

Assume O⁵⁺ ions with bulk radial speed V_W = 200 km/s at 2 R⊙, parallel kinetic temperature T_∥ = 5×10⁶ K and perpendicular T_⊥ = 5×10⁷ K. The most-probable speed parallel to the LOS (which equals magnetic field B for radial geometry above a polar coronal hole) is

$$ v_{1/e,\|} = \sqrt{2 k_B T_\|/m_O} \approx 80\;\mathrm{km/s} $$

while perpendicular

$$ v_{1/e,\perp} = \sqrt{2 k_B T_\perp/m_O} \approx 250\;\mathrm{km/s} $$

For an off-limb LOS perpendicular to B, the observed Doppler width comes from v_{1/e,⊥}, giving FWHM = 2√(ln 2) λ₀ v_{1/e,⊥}/c ≈ 1.43 Å at 1032 Å. Combined with C II pumping contribution shifting at 1037 Å by V_W/c × 1037 = 0.69 Å, the simulated line ratio I_{1032}/I_{1037} drops well below the optically-thin static value of 2:1 (often → 0.7 or lower in fast wind) — the famous UVCS ratio inversion that signaled extreme T_⊥ anisotropy.

200 km/s 흐름의 O⁵⁺ 이온에서 시선이 자기장에 수직이라 가정하면 관측되는 도플러 폭은 v_{1/e,⊥}에서 오며, 1032 Å에서 약 1.43 Å이다. C II 펌핑으로 인해 1032/1037 비율이 정적인 2:1에서 크게 벗어나 ~0.7까지 내려가는 것이 빠른 태양풍에서 관측된 UVCS의 핵심 결과이다.

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1869 ── Harkness/Young: green coronal line / 녹색 코로나선 (Fe XIV, before UVCS)
1934 ── Lyot: visible coronagraph / Lyot 코로나그래프
1958 ── Parker: thermal-pressure solar wind / 열압력 태양풍
1965 ── Hughes: e-scattered Lyα electron T_e diagnostic
1970 ── Hyder & Lites: Doppler dimming concept
1973 ── Skylab ATM: white-light + EUV imaging coronagraph
1980 ── Kohl et al.: rocket UV coronagraph; first coronal Lyα profile
1982 ── Kohl & Withbroe; Withbroe et al.: hole vs streamer Lyα profiles
1985 ── Withbroe et al.: Spartan 201 — first non-rocket UV coronagraph
1987 ── Noci, Kohl & Withbroe: O VI doublet diagnostic; C II pumping
1995 ── ★ THIS PAPER ★ — UVCS instrument paper / 본 논문
1995 Dec ─ SOHO launch
1996+ ─ UVCS coronal-hole observations: T_⊥(O⁵⁺) ≫ T_∥, ion-cyclotron heating
2003+ ─ Cranmer reviews; LASCO + UVCS streamer/CME synergy
2018 ── Parker Solar Probe launch — in-situ comparison
2020 ── Solar Orbiter launch; Metis = visible+UV coronagraph (UVCS heir)
2024 ── PROBA-3/ASPIICS — formation-flying coronagraph
```

UVCS는 로켓·Spartan 시대의 자외선 코로나그래프를 SOHO 시대 장기 관측 표준으로 격상시킨 결정적 단계이다. / UVCS marks the transition from rocket/Spartan-era UV coronagraphs to the SOHO long-duration era and underpins all subsequent UV coronagraph designs.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Lyot (1939) — coronagraph | 외부 차폐 코로나그래프의 원리 / Original external-occulter idea | UVCS의 외부 occulter 설계 직접 계승 / Direct ancestor of UVCS external occulter |
| Hughes (1965) — e-scattered Lyα | 코로나 T_e 측정 기법 제안 / Proposed Thomson-scattered Lyα as T_e diagnostic | UVCS Lyα 채널 핵심 진단 / UVCS Lyα channel implements this diagnostic |
| Hyder & Lites (1970) — Doppler dimming | 흐름 속도 진단 원리 / First Doppler-dimming proposal | UVCS의 핵심 V_W 진단 / Core UVCS outflow diagnostic |
| Kohl & Withbroe (1982) — rocket UV coronagraph | 첫 코로나홀 Lyα 프로파일 / First rocket UV Lyα profile in coronal hole | UVCS의 직접 기술 전구체 / Direct technological precursor |
| Noci, Kohl & Withbroe (1987) — OVI doublet | C II 펌핑·OVI 비율 진단 정립 / OVI doublet + C II pumping diagnostics | UVCS의 OVI 채널 직접 응용 / OVI channel implements this |
| Brueckner et al. (1995) — LASCO instrument paper | SOHO 가시광 코로나그래프 / Visible coronagraph on SOHO | UVCS와 동시 운용·CME 분석 시너지 / Synergy with UVCS for CME analysis |
| Parker (1958) — solar wind | 열압력 구동 태양풍 모델 / Thermal-pressure solar wind | UVCS가 검증·반증할 핵심 가설 / The hypothesis UVCS tests |
| van de Hulst (1950) — K/F corona theory | Thomson 산란 이론 / Thomson scattering theory | WLC 채널의 이론적 기반 / Theoretical basis of WLC |

---

## 7. References / 참고문헌

- Kohl, J. L., et al., "The Ultraviolet Coronagraph Spectrometer for the Solar and Heliospheric Observatory", *Solar Physics* **162**, 313–356 (1995). DOI: 10.1007/BF00733433
- Hughes, M. P., 1965 (e-scattered Lyα profile theory).
- Hyder, C. L., and Lites, B. W., "Coronal Doppler dimming", *Solar Physics* **14**, 147 (1970).
- van de Hulst, H. C., "K and F corona", *Bull. Astron. Inst. Netherlands* **11**, 135 (1950).
- Billings, D. E., *A Guide to the Solar Corona*, Academic Press, New York (1966).
- Allen, C. W., *Astrophysical Quantities*, The Athlone Press, London (1964).
- Pancharatnam, S., *Proc. Indian Acad. Sci.* **A41**, 137 (1955).
- Noci, G., Kohl, J. L., and Withbroe, G. L., "OVI doublet diagnostic with C II pumping", *Astrophys. J.* **315**, 706 (1987).
- Kohl, J. L., and Withbroe, G. L., "Coronal hole Doppler-dimming", *Astrophys. J.* **256**, 263 (1982).
- Withbroe, G. L., Kohl, J. L., Weiser, H., Noci, G., Munro, R. H., *Astrophys. J.* **254**, 361 (1982b).
- Withbroe, G. L., Kohl, J. L., Weiser, H., and Munro, R. H., *Astrophys. J.* **297**, 324 (1985).
- Strachan, L., *Measurement of Outflow Velocities in the Solar Corona*, PhD thesis, Harvard University (1990).
- Siegmund, O. H. W., et al., "MCP+XDL detectors for UVCS", *Proc. SPIE* **2280**, 89–100 (1994).
- Romoli, M., et al., *Appl. Opt.* **32**, 3559 (1993). [WLC stray-light]
- Pernechele, C., Naletto, G., Nicolosi, P., Poletto, L., Tondello, G., *Proc. SPIE* **2517** (1995). [Spectrometer characterization]
- Fineschi, S., et al. (1994). [UVCS grating efficiency]
- Huber, M. C. E., Timothy, J. G., Lemaître, G., Tondello, G., Jannitti, E., Scarin, P., "Toric-grating EUV spectrometer", *Appl. Opt.* **27**, 3503 (1988).
- Parker, E. N., "Dynamics of the interplanetary gas and magnetic fields", *Astrophys. J.* **128**, 664 (1958).
- Brueckner, G. E., et al., "LASCO", *Solar Physics* **162**, 357 (1995).

---

## Appendix A: Symbol Glossary / 기호 사전

| Symbol | Meaning |
|---|---|
| R⊙ | Solar radius (1 R⊙ ≈ 6.957×10⁸ m) |
| α, β | Grating angle of incidence / diffraction |
| ±β_o | Stigmatic diffraction angles |
| R_h, R_v | Toric grating major / minor curvature radii |
| N_e | Electron number density |
| T_e | Electron temperature |
| T_∥, T_⊥ | Ion kinetic temperatures parallel / perpendicular to magnetic field |
| V_W | Solar-wind bulk outflow velocity |
| A_el | Elemental abundance |
| ⟨R_i(T_e)⟩ | LOS-averaged ionization fraction |
| ⟨D_i(V_W)⟩ | LOS-averaged Doppler-dimming factor |
| φ_chromo | Chromospheric incoming line profile |
| φ_abs | Coronal absorption line profile |
| pI | Linearly polarized radiance |
| QDE | Quantum detection efficiency |
| FWHM | Full width at half maximum |

## Appendix B: Acronyms / 약어

- **UVCS** — Ultraviolet Coronagraph Spectrometer
- **SOHO** — Solar and Heliospheric Observatory
- **TSU / REU** — Telescope-Spectrometer Unit / Remote Electronics Unit
- **WLC** — White Light Channel
- **MCP** — Microchannel Plate
- **XDL** — Cross Delay Line (anode)
- **MOM** — Mirror/Occulter Mechanism
- **VCA / DIT** — Voice Coil Actuator / Differential Impedance Transducer
- **EOF / IWS / SMOCC** — Experiment Operations Facility / Instrument Workstation / SOHO Mission Operations Control Centre
- **DRAF / EAF** — Data Reduction and Analysis Facility / Experiment Analysis Facility
- **MLI** — Multi-Layer Insulation
- **ITO** — Indium Tin Oxide
