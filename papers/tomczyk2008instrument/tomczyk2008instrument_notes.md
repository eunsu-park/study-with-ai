---
paper_id: 34
topic: Solar_Observation
date: 2026-04-23
type: notes
title: "An Instrument to Measure Coronal Emission Line Polarization"
authors: "Tomczyk, S.; Card, G.L.; Darnell, T.; Elmore, D.F.; Lull, R.; Nelson, P.G.; Streander, K.V.; Burkepile, J.; Casini, R.; Judge, P.G."
year: 2008
journal: "Solar Physics 247, 411-428"
doi: "10.1007/s11207-007-9103-6"
tags: [solar_observation, coronal_magnetometry, polarimetry, CoMP, Fe_XIII, Hanle, Zeeman, Lyot_filter, Alfven_waves]
---

# Notes: CoMP — An Instrument to Measure Coronal Emission Line Polarization

## Core Contribution / 핵심 기여

**EN:** Tomczyk et al. (2008) describe the Coronal Multichannel Polarimeter (CoMP), a ground-based instrument installed on the 20-cm Coronal One Shot (COS) coronagraph at Sacramento Peak (National Solar Observatory). CoMP combines a four-stage calcite birefringent (Lyot) filter tunable via six liquid-crystal variable retarders (LCVRs) with a calcite Wollaston polarizing beamsplitter to measure the complete Stokes vector (I, Q, U, V) of the Fe XIII 1074.7 and 1079.8 nm coronal emission lines and the He I 1083.0 nm chromospheric line, simultaneously imaging line and continuum over a 2.8 R_sun field of view with 4.5″ pixel sampling. The paper establishes the design rationale (S/N optimization following Babcock 1953), describes the optical, mechanical, thermal, and detector subsystems, derives the 4×4 polarimetric response matrix and the allowable calibration errors, and demonstrates with 2.4 hours of October 2005 data that the instrument achieves essentially photon-noise-limited coronal magnetic-field measurements with σ_B ≈ 3.5 G per 4.5″ pixel. CoMP is the enabling instrument behind the first detection of Alfvén waves in the corona (Tomczyk et al., 2007, Science).

**KR:** Tomczyk et al. (2008)은 미국 국립태양관측소 Sacramento Peak 시설의 20 cm Coronal One Shot (COS) 코로나그래프에 탑재된 지상 코로나 다채널 편광계 CoMP를 기술한다. CoMP는 6개의 액정 가변 위상지연자(LCVR)로 전기 조정되는 4단 calcite 복굴절(Lyot) 필터에 calcite Wollaston 편광 빔스플리터를 결합하여, Fe XIII 1074.7·1079.8 nm 코로나 금지 방출선과 He I 1083.0 nm 채층선의 완전 Stokes 벡터(I, Q, U, V)를 측정하고, 2.8 R_sun의 시야와 4.5″ 픽셀로 선과 연속광을 동시에 영상화한다. 본 논문은 Babcock (1953)을 따른 S/N 최적화 설계 논리, 광학·기계·열·검출기 서브시스템, 4×4 편광 응답행렬과 허용 보정오차, 2005년 10월 2.4시간 관측에서 픽셀당 σ_B ≈ 3.5 G의 사실상 광자잡음 한계 성능을 입증한 것을 보고한다. CoMP는 코로나 Alfvén 파 최초 검출(Tomczyk et al., 2007, Science)의 기반 장비이다.

---

## Reading Notes / 읽기 노트

### Section 1. Introduction (p. 411-412) / 1. 서론

**EN:** The introduction places CoMP in the broader context of coronal magnetic-field measurements. All major energetic solar phenomena — flares, CMEs, SEPs, the solar wind — derive their energy from the coronal magnetic field, yet routine B measurements in the corona have remained elusive. Existing techniques are reviewed:

- **Gyroresonance radio emission** (Gary & Hurford 1994; Brosius & White 2006): limited to B > 200 G, only in active regions.
- **Faraday rotation** from natural radio sources (Sofue et al. 1976; Mancuso & Spangler 2000) or spacecraft carriers (Stelzried et al. 1970): sparse pointings, limited sources.
- **Linear polarization of visible/IR forbidden lines** (Mickey 1973; Querfeld & Smartt 1984; Arnaud & Newkirk 1987): maps POS field direction.
- **Circular polarization / Zeeman effect**: first detected in the coronal green line by Harvey (1969); became practical only with near-IR detector arrays (Lin, Penn & Tomczyk 2000; Lin, Kuhn & Coulter 2004).
- **Theory**: House (1972, 1977), Sahal-Bréchot (1974a,b, 1977), Judge (1998), Casini & Judge (1999), Judge, Low & Casini (2006) established that IR forbidden lines are optimal due to the λ² scaling of Zeeman shift.

Motivated by these advances, the authors built CoMP as a tunable filter/polarimeter targeting Fe XIII 1074.7/1079.8 nm and He I 1083.0 nm.

**KR:** 서론에서는 코로나 자기장 측정의 맥락을 정리한다. 플레어·CME·SEP·태양풍 등 모든 대형 태양 에너지 현상은 코로나 자기장에서 에너지를 얻지만, 코로나 B의 상시 측정은 난제로 남아 있었다. 기존 기법은 다음과 같다.

- **Gyroresonance 라디오 방출**: B > 200 G인 활동영역에만 한정.
- **Faraday 회전**: 드문 라디오원 또는 우주선 반송파 의존.
- **가시·근적외 금지선 선형편광**: POS 자기장 방향 지도화.
- **원편광 / Zeeman 효과**: Harvey (1969)의 녹색선 최초 검출 이후, 근적외 어레이 검출기(Lin, Penn & Tomczyk 2000)로 실용화.
- **이론**: House, Sahal-Bréchot, Judge 등이 Zeeman 시프트의 λ² 의존성으로 근적외 금지선이 최적임을 확립.

이러한 발전에 힘입어 저자들은 Fe XIII 1074.7/1079.8 nm와 He I 1083.0 nm를 대상으로 CoMP를 구축하였다.

### Section 2. Instrument Design Drivers (p. 412-413) / 2. 설계 구동 요인

**EN:** Two polarimetric signals are targeted:

1. **Stokes V (circular polarization, Zeeman)**: in the weak-field limit,
   $$V(\lambda) \propto \frac{dI}{d\lambda} g_{\rm eff} \lambda^2 B_{\rm LOS}$$
   The amplitude for Fe XIII 1074.7 nm is V/I ≈ 10⁻⁴ per gauss of B_LOS. This is **extremely small** and drives nearly all design choices.

2. **Stokes Q, U (linear polarization, resonance scattering + Hanle)**: dominated by resonance scattering, not second-order Zeeman. Q/I, U/I ≈ 1-10%. Because Fe XIII is in the **saturated Hanle regime** for typical coronal B, Q and U can only constrain the POS **direction** (azimuth) of B, not its magnitude.

**Three-passband strategy**: sampling the line at three wavelengths permits simultaneous determination of line intensity, Doppler shift (LOS velocity), and (via Fe XIII 1074.7/1079.8 nm ratio) coronal density. Filter multiplexing beats a slit spectrograph for rapid imaging of few samples.

**S/N optimization** (Babcock 1953): for Gaussian line with e-folding half-width w and constant background B, a four-stage birefringent filter of FWHM Δλ displaced by d from line center gives
$$\frac{S}{N} \propto \frac{\int_\lambda V(\lambda, w) F(\lambda, \Delta\lambda, d)\,d\lambda}{\sqrt{\int_\lambda [I(\lambda, w) + B]\,F(\lambda, \Delta\lambda, d)\,d\lambda}}.$$
With w = 0.107 nm (30 km/s thermal width for Fe XIII), optimal values are:
- **No background**: Δλ = 0.161 nm, d = 0.134 nm.
- **Background = 10×I_peak**: Δλ = 0.117 nm, d = 0.088 nm.

CoMP selected **FWHM = 0.13 nm** as a compromise for ground operation. The S/N maxima are broad (Figure 1), so this is robust.

**Selected design**: 4-stage calcite birefringent filter with LCVR tuning at 1074.7, 1079.8, 1083.0 nm; 0.13 nm FWHM; complete Stokes I/Q/U/V via LCVR polarization analysis; polarizing beamsplitter for simultaneous line+continuum; IR HgCdTe imager; large coronal FOV (2.8 R_sun).

**KR:** 두 편광 신호를 목표로 한다.

1. **Stokes V (원편광, Zeeman)**: 약자기장 극한에서 V(λ) ∝ (dI/dλ) g_eff λ² B_LOS. Fe XIII 1074.7 nm의 경우 B_LOS 1 G당 V/I ≈ 10⁻⁴로 **극도로 작다** — 거의 모든 설계 선택을 좌우한다.

2. **Stokes Q, U (선형편광, 공명산란 + Hanle)**: 공명산란이 지배적이며 Q/I, U/I ≈ 1-10%. 전형적 코로나 B에서 Fe XIII는 **포화 Hanle 영역**에 있어 Q·U는 B의 **POS 방위각**만 제약한다.

**3 통과대역 전략**: 선을 세 파장에서 샘플링하면 선 강도, 도플러 시프트(LOS 속도), Fe XIII 1074.7/1079.8 nm 비를 통한 코로나 밀도를 동시에 구할 수 있다. 소수 샘플 고속 영상에는 slit 분광기보다 필터 다중화가 유리하다.

**S/N 최적화** (Babcock 1953): e-folding 반폭 w인 Gaussian 선, 일정 배경 B, FWHM Δλ이 선 중심에서 d만큼 이동한 4단 복굴절 필터 F에 대해 식 (1). w = 0.107 nm (Fe XIII 30 km/s 열폭)에서
- 배경 0: Δλ = 0.161 nm, d = 0.134 nm.
- 배경 = 10×I_peak: Δλ = 0.117 nm, d = 0.088 nm.

CoMP는 지상 관측 타협안으로 **FWHM 0.13 nm**를 선택; 최댓값이 완만하여 견고하다.

**채택 설계**: 4단 calcite 복굴절 필터 + LCVR 조정(1074.7/1079.8/1083.0 nm), 0.13 nm FWHM, LCVR 편광 분석으로 I/Q/U/V 완전 측정, 편광 빔스플리터로 선+연속광 동시, IR HgCdTe 이미저, 2.8 R_sun FOV.

### Section 3. Instrument (p. 414-419) / 3. 장비

#### 3.1 Coronagraph and Optical System / 코로나그래프와 광학계

**EN:** CoMP replaces the back end of the COS coronagraph (Smartt, Dunn & Fisher 1981). COS: 20 cm aperture, uncoated BK7 biconvex f/11 singlet objective with an aspheric front surface polished to coronagraphic quality. Scattered light: ~3 μB_sun at 0.28° off axis (Smartt 1979). A lens cover is located 46 cm ahead of the objective.

Optical path (Figure 2): 20-cm objective → occulting disk at prime focus → collimating lens (two 600 mm f/l lenses combined to give 300 mm, f/5) → filter wheel with pre-filters → birefringent filter/polarimeter → Mamiya R22 110 mm f/2.8 camera lens as reimager → HgCdTe detector. Result: **2.8 R_sun FOV with 4.5″/pixel sampling**.

Chromatism of the singlet: focal length varies 0.49 mm over 1074.7-1083.0 nm, within the 1.06 mm depth of focus — **no refocusing needed**.

Occulting disk: original COS diamond-polished aluminum disk with an angled reflector dumping solar-disk light into a trap. Field stop limits FOV to 2.8 R_sun. Collimating lens forms an image of the objective on a **Lyot stop** (12.7 mm radius, 0.93× projected objective) inside the birefringent filter: diffraction scattered light reduced by ~5×10⁻⁵ (Noll 1973; Johnson 1987). A Lyot spot to block multiple-reflection ghosts was omitted (ghost brightness ~7×10⁻⁷ B_sun, below sky background).

**KR:** CoMP는 COS 코로나그래프의 후단을 교체한다. COS: 20 cm 구경, 비구면 전면이 연마된 BK7 양볼록 f/11 단일렌즈. 축외 0.28° 산란광 ~3 μB_sun. 렌즈 커버는 대물 앞 46 cm.

광학 경로(그림 2): 20 cm 대물 → prime focus의 occulting disk → 시준렌즈(600 mm f/l 2매 결합, 합성 300 mm, f/5) → pre-filter 휠 → 복굴절 필터/편광계 → Mamiya R22 110 mm f/2.8 재결상렌즈 → HgCdTe 검출기. **2.8 R_sun FOV, 4.5″/pixel**.

단일렌즈 색수차: 1074.7-1083.0 nm 구간 초점거리 변동 0.49 mm, 초점심도 1.06 mm 이내 — **재초점 불필요**.

Occulting disk: 각도 반사판이 태양디스크 빛을 trap으로 보낸다. Field stop가 FOV를 2.8 R_sun으로 제한. 시준렌즈가 대물상을 필터 내부 **Lyot stop**(반경 12.7 mm, 투영 대물의 0.93배)에 맺어 회절 산란광을 ~5×10⁻⁵로 감쇠. Lyot spot은 생략(고스트 ~7×10⁻⁷ B_sun, 하늘 배경보다 작음).

#### 3.2 Filter/Polarimeter (Figure 3) / 필터/편광계

**EN:** Four-stage calcite Lyot-Evans birefringent filter (Lyot 1944; Evans 1949) with exit polarizer **replaced by a Wollaston polarizing beamsplitter** (Öhman 1956) for simultaneous line+continuum imaging. Each calcite stage is tuned with a Nematic LCVR. Two additional LCVRs **before the first polarizer** analyze the input Stokes state. Prefilters (FWHM ~1.7 nm) block unwanted filter orders.

Design rigor:
- **Beckers & Dunn (1965)** Jones-matrix formulation; bandpass 0.13 nm; free spectral range 2.34 nm.
- **Monte Carlo tolerance**: calcite elements require angular alignment better than 0.5° rms (others much less sensitive). Achieved: <0.1° by rotating in an alignment jig for minimum transmission between crossed reference polarizers, then fixing with Silicone RTV.
- Wollaston: 15° cut angle, calcite.
- Polarizers: Corning Polarcor, transmission 0.97 (polarized), contrast >10⁴.
- Half-waveplates: polymer, sandwiched in glass.
- Elements oil-coupled (Dow Corning 200 silicone, η=1.4, 10000 cSt); ~5 lb spring force; transmission 0.29 (maximum 0.5); primary loss is LCVRs (0.96 each) × 6 and polarizers (0.98) × 2.
- Total filter/polarimeter weight: 5.5 kg.
- Figure 5: five measured filter tunings in the 1074.7 nm vicinity, each shifted by 0.2 nm — filter works as designed.

**KR:** 4단 calcite Lyot-Evans 복굴절 필터(Lyot 1944; Evans 1949)의 exit 편광자를 **Wollaston 편광 빔스플리터**(Öhman 1956)로 교체하여 선+연속광 동시 결상. 각 calcite 단은 Nematic LCVR로 조정. 첫 편광자 **앞**에 2개의 LCVR을 추가 배치하여 입력 Stokes 상태를 분석. 사전필터(FWHM ~1.7 nm)로 원치 않는 차수 차단.

설계 엄밀성:
- **Beckers & Dunn (1965)** Jones 행렬 공식; 대역 0.13 nm; FSR 2.34 nm.
- **Monte Carlo 허용도**: calcite 요소는 각도 정렬 0.5° rms 이내 필요. 실제 달성 <0.1° (교차편광자 사이 투과 최소화 지그로 회전 후 Silicone RTV 고정).
- Wollaston: 절단각 15°, calcite.
- 편광자: Corning Polarcor, 투과 0.97, 대비 >10⁴.
- 반파장판: 폴리머, 유리 샌드위치.
- 요소 간 실리콘 오일(η=1.4, 10000 cSt) 접합, ~5 lb 스프링 가압; 투과 0.29(최대 0.5); 주 손실은 LCVR 6매(각 0.96)와 편광자 2매(0.98).
- 필터/편광계 총중량 5.5 kg.
- 그림 5: 1074.7 nm 근방 5 튜닝, 각 0.2 nm 시프트 — 설계대로 작동.

#### 3.3 Detector / 검출기

**EN:** Rockwell Scientific **TCM8600 HgCdTe 1024×1024** with 18 μm pixels. Readout noise ~70 e⁻; full well ~150000 e⁻; 14-bit digitization via 8 output channels; conversion ~25 e⁻/LSB; 30 Hz readout. LN₂-cooled, 12 h hold time. Excellent linearity (Cao et al. 2005).

**KR:** Rockwell Scientific **TCM8600 HgCdTe 1024×1024** (18 μm 픽셀). 읽기잡음 ~70 e⁻, full well ~150000 e⁻, 14비트/8 채널 디지털화, 변환비 ~25 e⁻/LSB, 읽기속도 30 Hz. 액체질소 냉각, 12시간 유지.

#### 3.4 Intensity Calibration / 강도 보정

**EN:** A 40° FWHM holographic Light Shaping Diffuser (LSD) can be inserted in front of the objective on command. Uniform across FOV to a few parts in 10⁴. The diffuser provides: (1) flat-fielding, (2) correction of filter transmission vs wavelength, (3) relative normalization of the two Wollaston beams, (4) normalization to disk-intensity units. Cross-calibration against disk observation through a neutral density filter (measured transmission 3.0×10⁻⁵ at 1064 nm) gives a diffuser radiance of **84 μB_sun**.

**KR:** 대물 앞에 삽입 가능한 40° FWHM 홀로그래픽 LSD 확산판. FOV 균일도 ~10⁻⁴. 기능: (1) flat-field, (2) 필터 투과율 보정, (3) 두 Wollaston 빔 정규화, (4) 디스크 단위로 정규화. 중성필터(1064 nm에서 투과 3.0×10⁻⁵)를 통한 디스크 관측과 교차검정 → 확산판 복사휘도 **84 μB_sun**.

#### 3.5 Temperature Control / 온도 제어

**EN:** Calcite birefringence: filter bandpass shifts **−0.056 nm/°C** (−15.6 km/s/°C in Doppler). Nested PID loops:
- **Instrument enclosure**: 35°C ±1°C, Kapton Thermofoil heaters (110 W).
- **Birefringent filter housing**: 30°C, 12 Inconel cartridge heaters (45 W each) under precision PID; stability **<5 mK over 24 h**.
RTD sensors throughout; separate temperature-logger unit.

**KR:** Calcite 복굴절: 대역 시프트 **−0.056 nm/°C**(−15.6 km/s/°C). 이중 PID 루프:
- **장비 캐비닛**: 35°C ±1°C, Kapton 발열체 110 W.
- **필터 하우징**: 30°C, Inconel 카트리지 히터 12개(각 45 W); 24시간 **<5 mK** 안정도.
장비 전역에 RTD 센서, 별도 온도 로거.

#### 3.6 Data Acquisition / 데이터 수집

**EN:** Pentium 4 Windows 2000 PC, LabView control. Camera via Camera Link. Block diagram (Figure 6): camera + 6 LCVRs + 16 temperature sensors + calibration polarizer + diffuser + occulting disk X/Y + reimaging lens Z + filter wheel + lens cover, all orchestrated from one PC. Automated mode: observing scripts of wavelengths × polarization states.

**KR:** Pentium 4 Windows 2000 PC, LabView 제어. 카메라 Camera Link. 그림 6 블록도: 카메라, LCVR 6개, 온도센서 16개, 보정편광자, 확산판, occulting disk X/Y, 재결상렌즈 Z, 필터휠, 렌즈 커버 — 모두 단일 PC 조율. 자동모드: 파장×편광 상태 스크립트.

### Section 4. Polarimetric Calibration (p. 419-422) / 4. 편광 보정

**EN:** The polarimetric response is
$$\mathbf{S}_{\rm meas} = \mathbf{R}\,\mathbf{S}_{\rm input},$$
where **R** is the 4×4 response matrix. An ideal polarimeter has R = I.

**Calibration procedure**: the diffuser is placed in front of the objective; a rotating linear polarizer and a fixed quarter-wave plate are inserted just behind the occulting disk assembly, producing known input states:
1. No polarizer or retarder → unpolarized.
2. Linear polarizer at 0°, 45°, 90°, 135° → I±Q, I±U.
3. Retarder fixed at 0° and polarizer at ±45° → I±V.

A nonlinear least-squares fit solves for **23 unknowns** per pixel: polarizer transmission, retarder transmission, 4 Stokes components of stationary stray light, retarder retardance, retarder orientation error, and 15 response-matrix elements (R_11 is normalized to 1). Low-order 2-D polynomial smooths the per-pixel fit.

**Measured R** (FOV average):
$$\mathbf{R} = \begin{bmatrix}
1.000 & -0.026 & -0.014 & -0.005 \\
-0.005 & 0.952 & -0.002 & 0.046 \\
0.002 & -0.004 & 0.977 & -0.056 \\
-0.001 & -0.018 & -0.048 & 0.876
\end{bmatrix}$$

**Calibration accuracy requirement**: Given
$$\mathbf{S}_{\rm corona} \approx \begin{bmatrix} 1 \\ 0.1 \\ 0.1 \\ 10^{-3} \end{bmatrix} I$$
(10% linear polarization, 10 G field → V/I ~10⁻³), and desired noise
$$\boldsymbol{\sigma}_{\rm S} \leq \begin{bmatrix} 10^{-3} \\ 10^{-3} \\ 10^{-3} \\ 10^{-4} \end{bmatrix} I$$
(1 G LOS-field precision), the acceptable R errors are:
$$\boldsymbol{\sigma}_{\rm R} \leq \begin{bmatrix} - & 10^{-2} & 10^{-2} & 10^{0} \\ 10^{-3} & 10^{-2} & 10^{-2} & 10^{0} \\ 10^{-3} & 10^{-2} & 10^{-2} & 10^{0} \\ 10^{-4} & 10^{-3} & 10^{-3} & 10^{-1} \end{bmatrix}.$$
The **bottom row** (I,Q,U → V crosstalk) needs 10⁻⁴ accuracy.

**Monte Carlo (1000 realizations)** gives measured σ_R elements ~10⁻³ to 10⁻⁴. Most satisfy requirements; the V row is 5-8×10⁻⁴ — **close to but not below 10⁻⁴**. Remedy: exploit symmetry (V antisymmetric around line center vs. I, Q, U symmetric) to empirically subtract residual crosstalk. This has been demonstrated to reduce Q/U → V crosstalk to ~10% of field strength "to a fraction of a gauss" (Lin, Kuhn & Coulter 2004).

**KR:** 편광 응답은 S_meas = R·S_input (식 2), R은 4×4 응답행렬, 이상적 편광계는 R = I.

**보정 절차**: 대물 앞 확산판, occulting disk 직후 회전 선형편광자 + 고정 사분파장판 삽입으로 기지 입력 상태 생성:
1. 편광자·지연자 제거 → 비편광.
2. 선형편광자 0°, 45°, 90°, 135° → I±Q, I±U.
3. 지연자 0° 고정, 편광자 ±45° → I±V.

비선형 최소제곱으로 픽셀당 **23 미지수**(편광자 투과, 지연자 투과, 정적 미광의 4 Stokes, 지연자 retardance, 지연자 방위 오차, R의 15원소 (R_11=1 정규화))를 풀고 2D 저차 다항식으로 평활.

**측정 R** (FOV 평균): 위 행렬. 대각은 0.88-1.00 범위, 비대각 crosstalk ~1-6%.

**보정 정확도 요구**: S_corona ≈ [1, 0.1, 0.1, 10⁻³]·I (선형편광 10%, 10 G에서 V/I ~10⁻³), σ_S ≤ [10⁻³, 10⁻³, 10⁻³, 10⁻⁴]·I (1 G LOS 정밀도). 허용 σ_R은 위 행렬과 같고, **하단 V 행**은 10⁻⁴ 필요.

**1000회 몬테카를로** 결과: 대부분 요구치 충족, V 행은 5-8×10⁻⁴로 경계. 해결: V는 선중심 반대칭, I·Q·U는 대칭이라는 성질을 이용해 잔여 crosstalk을 경험적으로 제거. Lin, Kuhn & Coulter (2004)는 이 방식으로 Q·U→V crosstalk을 자기장의 10%까지, 측정 불확도를 1 G 이하로 낮추었다.

### Section 5. Sample Data (p. 422-425) / 5. 샘플 데이터

**EN:** CoMP was deployed on COS **29 January 2004**. Alfvén-wave time series (Tomczyk et al. 2007) is presented elsewhere; this paper shows linear and circular polarization data from **31 October 2005, 15:04-17:46 UT** in Fe XIII 1074.7 nm.

**Observing cadence**:
- **Linear-polarization image group**: 5 images each in I+Q, I−Q, I+U, I−U at 3 wavelengths (1074.52, 1074.65, 1074.78 nm).
- **Circular-polarization image group**: 10 images each in I+V, I−V at the same 3 wavelengths.
- Exposure 250 ms, 100 ms LCVR settling delay.
- **One group: ~29 s (52% duty cycle, 15 s photon collection)**.

**Total collected**: 146 circular + 37 linear polarization groups over 2.4 h. Median sky background 16.4 μB_sun (good sky).

**Data reduction**:
1. Subtract mean dark.
2. Normalize by mean diffuser image.
3. Register, translate, rotate to solar-N.
4. Subtract continuum image from line image.
5. Compute mean Stokes images per wavelength per group.
6. Apply inverse response matrix R⁻¹.

**Line fitting**: 7 parameters per pixel — center intensity, line width, center wavelength, linear polarization degree p = √(Q²+U²)/I, magnetic azimuth φ = 0.5 arctan(U/Q), and a parameter to remove the residual symmetric Gaussian (I,Q,U → V crosstalk). I, Q, U profiles: Gaussian; V profile: first derivative of Gaussian.

**Results** (Figures 7-11):
- **Figure 7**: FOV-wide FeXIII 1074.7 nm intensity image with overlaid Hα disk (HAO PICS, Mauna Loa). Subarray 0.4×0.7 R_sun on east limb selected; lower limit 1.05 R_sun.
- **Figure 8a**: intensity 0-35 μB_sun, bright streamer structures.
- **Figure 8b**: **LOS velocity** ±6 km/s, significant relative flows.
- **Figure 8c**: intensity image overlaid with **POS B-field azimuth vectors** (180° ambiguous) tracing coronal loops.
- **Figure 8d**: **LOS B_LOS** bipolar, ±40 G, with upper region negative and lower region positive.
- **Figure 9**: latitudinally-averaged linear polarization fraction rises from **2% at 1.05 R_sun to 7% at 1.3 R_sun** — normal Hanle-regime behavior (Arnaud & Newkirk 1987).

**Error analysis**: The formal fit errors underestimate reality. Using Penn et al. (2004),
$$\sigma_{\lambda_0} = \frac{w}{\sqrt{2}} \frac{\sigma_I}{I}.$$
Converting to B_LOS via the Zeeman shift 4.67×10⁻¹² g λ² nm/G (Landi Degl'Innocenti 1992) with g=1.5, λ=1074.7 nm, w=0.107 nm:
$$\sigma_B = 9396\,\frac{\sigma_V}{I}\;{\rm (G)}.$$
σ_V measured as scatter of V among 146 groups divided by √146. Normalized by local I → σ_B map.

**Photon noise**: σ_N = N^(1/2) = (Ik)^(1/2), where k = 875 photons/μB_sun/(each 20 V exposures × 146 groups)^(1/2). Derived σ_B map from photon noise closely matches the empirical map (**Figure 10a vs 10b**). Figure 11 histograms: median observed σ_B = **3.5 G** vs. photon-noise σ_B = **3.2 G** — essentially photon-noise-limited.

**KR:** CoMP는 **2004년 1월 29일** COS에 장착. Alfvén 파 시계열(Tomczyk+ 2007 Science)은 별도 논문. 본 논문은 **2005년 10월 31일 15:04-17:46 UT** Fe XIII 1074.7 nm 선형·원편광 데이터를 제시.

**관측 cadence**:
- **선형 편광 그룹**: 3 파장(1074.52/1074.65/1074.78 nm) × (I±Q, I±U) 각 5장.
- **원편광 그룹**: 3 파장 × (I±V) 각 10장.
- 노출 250 ms, LCVR 안정 대기 100 ms.
- **그룹당 ~29 s (듀티 52%, 광자수집 15 s)**.

**총 수집**: 원편광 146, 선형 37 그룹 / 2.4시간. 하늘 배경 중앙값 16.4 μB_sun.

**환원 절차**: (1) 평균 dark 차감, (2) 평균 diffuser로 정규화, (3) 정합·이동·태양N 회전, (4) 연속광 차감, (5) 파장별 그룹 평균 Stokes 계산, (6) R⁻¹ 적용.

**선 피팅**: 픽셀당 7 파라미터 — 중심 세기, 선폭, 중심파장, 선편광도 p=√(Q²+U²)/I, 자기장 방위각 φ=0.5 arctan(U/Q), I·Q·U→V 잔여 대칭성분 제거 파라미터. I·Q·U: Gaussian; V: Gaussian 1차 도함수.

**결과** (그림 7-11):
- 그림 7: FOV Fe XIII 1074.7 nm intensity + Hα 디스크(HAO PICS). 동쪽 림 0.4×0.7 R_sun; 하한 1.05 R_sun.
- 그림 8a: 세기 0-35 μB_sun.
- 그림 8b: **LOS 속도** ±6 km/s.
- 그림 8c: 세기 + **POS B 방위 벡터**(180° 불확정) — 코로나 루프 추적.
- 그림 8d: **B_LOS** ±40 G 쌍극 구조.
- 그림 9: 위도 평균 선편광 비율 1.05 R_sun 2% → 1.3 R_sun 7%. 정상 Hanle 행태(Arnaud & Newkirk 1987).

**오차분석**: Penn et al. (2004)로 σ_λ₀ = (w/√2)(σ_I/I). Zeeman 시프트 4.67×10⁻¹² g λ² nm/G (Landi Degl'Innocenti 1992)로 변환, g=1.5, λ=1074.7 nm, w=0.107 nm → **σ_B = 9396 σ_V/I (G)** (식 10). σ_V는 146 그룹 V 산포 /√146.

**광자잡음**: σ_N = √N, k=875 photons/μB_sun. 그림 10a(측정) vs 10b(광자잡음 예측) 밀접 일치. 그림 11 히스토그램: 측정 중앙값 **3.5 G** vs 광자잡음 **3.2 G** — 사실상 광자잡음 한계.

### Section 6. Conclusion and Future Prospects (p. 426) / 6. 결론과 전망

**EN:** CoMP demonstrates near-photon-noise-limited coronal magnetic measurements at 4.5″ pixels in 2.4 h with 20-cm aperture and 16.4 μB_sun sky. Future improvement requires **more photons** — meaning **larger coronagraphs**. CoMP continues to be used for outstanding coronal questions.

**KR:** CoMP는 20 cm 구경, 16.4 μB_sun 하늘에서 4.5″ 픽셀, 2.4시간 적분으로 광자잡음 한계 코로나 자기장 측정을 입증. 향후 개선은 **더 많은 광자** → **더 큰 코로나그래프**. CoMP는 현재도 코로나 연구에 활용.

---

## Key Takeaways / 핵심 시사점

1. **Forbidden lines + near-IR = coronal magnetography** / 금지선+근적외 = 코로나 자기 지도화
   **EN:** The Fe XIII 1074.7/1079.8 nm M1 forbidden lines, discovered by Harvey (1969) and theoretically developed by House, Sahal-Bréchot, and Judge, are the optimal probes of coronal B: (a) Zeeman shift ∝ g λ² is maximized; (b) the Hanle effect saturates for typical coronal B so Q/U cleanly give POS direction; (c) HgCdTe detector maturity (2000s) made them observationally accessible.
   **KR:** Fe XIII 1074.7/1079.8 nm M1 금지선은 코로나 B의 최적 탐침: (a) Zeeman 시프트 ∝ g λ² 최대, (b) 전형 코로나 B에서 Hanle 포화로 Q/U가 POS 방위각을 깔끔히 제공, (c) HgCdTe 성숙으로 실측 가능.

2. **Tunable Lyot filter + LCVRs = imaging polarimetry without a spectrograph** / 조정가능 Lyot 필터 + LCVR = 분광기 없는 영상 편광측정
   **EN:** Four calcite stages tuned by 6 LCVRs give 0.13 nm bandpass, 2.34 nm free spectral range, and full Stokes I/Q/U/V with three-point line sampling. This multiplex advantage beats slit spectrographs for mapping polarization over a wide FOV in the seconds needed for dynamic coronal studies.
   **KR:** 6개 LCVR로 조정된 4단 calcite가 0.13 nm 대역, 2.34 nm FSR, 세 파장 샘플링으로 I/Q/U/V 완전 관측을 제공. 초단위 역동 관측을 요하는 넓은 FOV 편광 지도화에서 slit 분광기 대비 다중화 이점.

3. **Wollaston beamsplitter eliminates flat-field systematics** / Wollaston 빔스플리터로 flat-field 계통오차 제거
   **EN:** Replacing the Lyot exit polarizer with a 15°-cut calcite Wollaston gives two simultaneous orthogonally-polarized images. Differencing (I+V)−(I−V), etc., cancels gain variations and atmospheric transparency fluctuations that would otherwise swamp the 10⁻⁴ Stokes V signal.
   **KR:** Lyot exit 편광자를 15° 절단 calcite Wollaston으로 교체하여 직교 편광 두 이미지를 동시 획득. (I+V)−(I−V) 차분으로 게인 변동과 대기 투과 변화를 상쇄; 없으면 10⁻⁴ Stokes V 신호를 묻어버림.

4. **Babcock S/N optimization ≠ just narrow filter** / Babcock S/N 최적화 ≠ 단순 좁은 필터
   **EN:** The Stokes V signal peaks in the line wings (it ∝ dI/dλ), so the optimum filter sits **displaced** from line center (d ≈ 0.09-0.13 nm) with FWHM 0.13 nm. Too narrow: lose V signal; too wide: pick up photon noise without V. Ground operation with ~10× sky background pulls optimum to smaller Δλ and d than space operation.
   **KR:** Stokes V는 dI/dλ ∝ 선 날개에서 최대 → 최적 필터는 선 중심에서 **변위 d ≈ 0.09-0.13 nm**, FWHM 0.13 nm. 너무 좁으면 V 손실, 너무 넓으면 V 없이 광자잡음. 지상 10× 하늘 배경은 공간 대비 더 좁은 Δλ, 작은 d 선호.

5. **Calibration R matrix needs 10⁻⁴ on V row — achievable with symmetry tricks** / R 행렬 V 행 10⁻⁴ — 대칭 이용으로 달성
   **EN:** Because V/I is 10⁻⁴ per gauss, the I,Q,U → V entries of R must be known to ~10⁻⁴. Direct calibration reaches ~10⁻³ to 5-8×10⁻⁴. The remaining factor is recovered by exploiting V's antisymmetry around line center versus I/Q/U symmetry — a key polarimetric trick that Lin, Kuhn & Coulter (2004) demonstrated reaches sub-gauss precision.
   **KR:** V/I는 1 G당 10⁻⁴이므로 R의 I·Q·U → V 성분은 10⁻⁴ 정확도가 필요. 직접 보정은 10⁻³~5-8×10⁻⁴ 수준. 나머지는 V(선중심 반대칭) vs I·Q·U(대칭) 성질을 이용한 잔여 crosstalk 제거로 확보; Lin, Kuhn & Coulter (2004)이 서브가우스 정밀도 입증.

6. **Thermal stability is existential** / 열 안정성은 필수
   **EN:** Filter bandpass drifts −0.056 nm/°C ≡ −15.6 km/s/°C. For km/s Doppler and stable polarization calibration, the filter must be thermally controlled to better than a few mK; CoMP achieves <5 mK / 24 h via nested PID loops (35°C instrument, 30°C filter).
   **KR:** 필터 대역이 −0.056 nm/°C = −15.6 km/s/°C로 드리프트. km/s 도플러와 안정 편광 보정에는 수 mK 수준 안정도가 필수; CoMP는 이중 PID(35°C 캐비닛, 30°C 필터)로 24시간 <5 mK 달성.

7. **CoMP is photon-noise-limited — the path forward is bigger coronagraphs** / CoMP는 광자잡음 한계 — 해법은 더 큰 코로나그래프
   **EN:** 2005-10-31 data: observed σ_B = 3.5 G matches photon-noise prediction 3.2 G. No hidden systematic floor is limiting performance — additional photons directly translate to lower σ_B. Hence the community's subsequent push toward COSMO (Coronal Solar Magnetism Observatory) 1.5-m coronagraph and DKIST's Cryo-NIRSP.
   **KR:** 2005-10-31: 측정 σ_B=3.5 G ≈ 광자잡음 3.2 G — 숨은 계통한계 없음. 광자 증가가 곧바로 σ_B 감소. 이 결과가 COSMO 1.5 m 코로나그래프와 DKIST Cryo-NIRSP 추진의 기반.

8. **CoMP enabled the first detection of Alfvén waves in the corona** / CoMP가 코로나 Alfvén 파 최초 검출을 가능케 함
   **EN:** The unique combination of 2.8 R_sun FOV, ~30 s cadence, and ~km/s Doppler precision enabled Tomczyk et al. (2007, Science 317, 1192) to detect ubiquitous transverse MHD waves propagating at ~2000 km/s with 5-min periods across the corona — the first direct confirmation of Alfvén waves as a candidate coronal-heating mechanism. This transformed observational MHD and motivated DKIST.
   **KR:** 2.8 R_sun FOV + ~30 s 주기 + km/s 도플러 정밀도의 조합으로 Tomczyk et al. (2007, Science 317, 1192)가 5분 주기·2000 km/s 전파의 코로나 전역 횡MHD 파를 최초 검출 — 코로나 가열 후보로서 Alfvén 파의 직접 확인. 관측 MHD를 혁신하고 DKIST 추진 동력.

---

## Mathematical Summary / 수학적 요약

### Zeeman shift and Stokes V / Zeeman 시프트와 Stokes V

In the weak-field limit the Zeeman shift of a split component is
$$\Delta\lambda_B = 4.67\times10^{-12}\,g_{\rm eff}\,\lambda^2\,B_{\rm LOS}\quad [\text{nm, with } \lambda \text{ in nm}, B \text{ in G}],$$
and the Stokes V profile is the first derivative of I:
$$V(\lambda) = -\Delta\lambda_B\,\frac{dI}{d\lambda}.$$
- g_eff = effective Landé factor (1.5 for Fe XIII 1074.7 nm).
- λ²-scaling: going from 500 nm to 1074 nm boosts V by (1074/500)² ≈ 4.6.
- For Fe XIII 1074.7 nm, 1 G LOS field gives V/I ≈ 10⁻⁴.

### Signal-to-noise for V measurement (Babcock 1953) / V 측정 S/N

$$\frac{S}{N} \propto \frac{\displaystyle\int_\lambda V(\lambda, w)\, F(\lambda, \Delta\lambda, d)\,d\lambda}{\sqrt{\displaystyle\int_\lambda [I(\lambda, w) + B]\,F(\lambda, \Delta\lambda, d)\,d\lambda}}.$$
- V: first derivative of Gaussian of e-folding half-width w.
- I: Gaussian line profile.
- B: constant background (sky + continuum).
- F: birefringent filter transmission, FWHM Δλ, displacement d from line center.
- Optimum at (Δλ, d) = (0.161, 0.134) nm for B=0; (0.117, 0.088) nm for B=10 I_peak.

### Polarimetric response / 편광 응답

$$\mathbf{S}_{\rm meas} = \mathbf{R}\,\mathbf{S}_{\rm input},\qquad \mathbf{S}_{\rm input} = \mathbf{R}^{-1}\,\mathbf{S}_{\rm meas}.$$
- R: 4×4 real matrix, ideally identity.
- Off-diagonal elements = crosstalk between I, Q, U, V.
- Measured (FOV average): R_11=1, R_22=0.952, R_33=0.977, R_44=0.876, max |crosstalk| ≈ 0.056.

### Calibration error budget / 보정 오차 예산

Coronal signal:
$$\mathbf{S}_{\rm corona} \approx \begin{bmatrix} 1 \\ 0.1 \\ 0.1 \\ 10^{-3} \end{bmatrix}\,I.$$
Desired noise:
$$\boldsymbol{\sigma}_{\rm S} \le \begin{bmatrix} 10^{-3} \\ 10^{-3} \\ 10^{-3} \\ 10^{-4} \end{bmatrix}\,I.$$
Acceptable R errors (propagating S_corona through σ_R to get σ_S):
$$\boldsymbol{\sigma}_{\rm R} \le \begin{bmatrix}
- & 10^{-2} & 10^{-2} & 10^{0} \\
10^{-3} & 10^{-2} & 10^{-2} & 10^{0} \\
10^{-3} & 10^{-2} & 10^{-2} & 10^{0} \\
10^{-4} & 10^{-3} & 10^{-3} & 10^{-1}
\end{bmatrix}.$$
Monte Carlo σ_R (measured) ≈ 10⁻³ to 10⁻⁴; V row 5-8×10⁻⁴, borderline.

### Line-center uncertainty and B_LOS / 선중심 불확도와 B_LOS

From Penn et al. (2004),
$$\sigma_{\lambda_0} = \frac{w}{\sqrt{2}}\,\frac{\sigma_I}{I}.$$
Combining with the Zeeman calibration (g=1.5, λ=1074.7 nm, w=0.107 nm):
$$\sigma_B = 9396\,\frac{\sigma_V}{I}\;[\text{G}].$$
Photon noise:
$$\sigma_V = \frac{\sqrt{I}}{\sqrt{k}}\quad [\mu B_\odot],$$
with k = 875 photons/μB_sun (CoMP 2005 setup).

### Linear polarization and magnetic azimuth / 선형편광과 자기장 방위각

$$p = \frac{\sqrt{Q^2 + U^2}}{I},\qquad \phi = \frac{1}{2}\arctan\!\left(\frac{U}{Q}\right).$$
The factor 1/2 reflects the π-ambiguity of linear polarization. In the saturated Hanle regime the azimuth φ traces the POS projection of B up to a 90° ambiguity (the van Vleck effect).

### Worked example: photon noise for CoMP / 계산 예: CoMP 광자잡음

Consider a coronal pixel at 1.1 R_sun with I = 20 μB_sun at the line peak.
- Photons per pixel per V sub-exposure: N = I × k = 20 × 875 = 17500.
- Single-exposure σ_I/I ≈ 1/√N ≈ 0.0076.
- For 20 V-images × 146 groups = 2920 samples: σ_V / I ≈ 0.0076 / √2920 ≈ 1.4×10⁻⁴.
- σ_B = 9396 × 1.4×10⁻⁴ ≈ 1.3 G.

For a fainter pixel I = 5 μB_sun the scaling gives σ_B ≈ 2.6 G, matching the Figure 11 histogram median ~3 G.

---

## Paper in the Arc of History / 역사 속의 논문

**EN:**
```
1869 ─── Harkness / Young: coronal green line 530.3 nm "coronium"
             |
1939 ─── Edlén: green line = Fe XIV M1 forbidden line (proves hot corona)
             |
1944 ─── Lyot: birefringent filter
1947 ─── Billings: 4-stage birefringent filter analysis
1949 ─── Evans: Evans filter refinement
1953 ─── Babcock: photon-noise-optimal magnetograph
             |
1956 ─── Öhman: polarizing Wollaston exit for filters
1965 ─── Beckers & Dunn: Jones-matrix Lyot filter design
1969 ─── Harvey: first Zeeman detection in green line
1972-77 ─── House, Sahal-Bréchot: coronal emission-line polarization theory
             |
1973 ─── Mickey; 1984 Querfeld & Smartt; 1987 Arnaud & Newkirk: IR linear polarization maps
             |
1997 ─── Kopp et al.: HAO He I 1083 nm imaging
1998-99 ─── Judge; Casini & Judge: full Fe XIII Hanle+Zeeman theory
             |
2000 ─── Lin, Penn, Tomczyk: IR HgCdTe coronal polarimetry demonstration
2004 ─── Lin, Kuhn & Coulter: Fe XIII Zeeman magnetometry; crosstalk subtraction
             |
             |
★ 2007 ─── Tomczyk et al.: Alfvén waves in corona (Science) — CoMP's first triumph
★ 2008 ─── THIS PAPER — CoMP instrument description (Solar Physics)
             |
2010s ─── CoMP continues daily observations at MLSO (Hawaii)
2015+ ─── COSMO (1.5 m coronagraph) proposal
2020 ─── DKIST first light with Cryo-NIRSP
2024 ─── UCoMP upgrade (Upgraded CoMP) observing in green+red lines
```

**KR:**
```
1869 ─── Harkness / Young: 코로나 녹색선 530.3 nm "coronium"
1939 ─── Edlén: 녹색선 = Fe XIV M1 금지선 (고온 코로나 증명)
1944 ─── Lyot: 복굴절 필터
1953 ─── Babcock: 광자잡음 최적 자기망원경
1965 ─── Beckers & Dunn: Jones 행렬 Lyot 설계
1969 ─── Harvey: 녹색선 Zeeman 최초 검출
1972-77 ─── House, Sahal-Bréchot: 코로나 방출선 편광 이론
2000 ─── Lin, Penn, Tomczyk: IR HgCdTe 코로나 편광측정
★ 2007 ─── Tomczyk 외: 코로나 Alfvén 파 (Science) — CoMP 최초 성과
★ 2008 ─── 본 논문 — CoMP 장비 기술 (Solar Physics)
2020 ─── DKIST Cryo-NIRSP 첫 관측
2024 ─── UCoMP 업그레이드
```

---

## Connections to Other Papers / 다른 논문과의 연결

| Paper | Relevance / 관련성 |
|-------|---------------------|
| **#7 Coronal forbidden lines** (if present) / 코로나 금지선 | Provides the physics of the Fe XIII 1074.7/1079.8 nm lines that CoMP exploits — level diagram, Einstein A coefficients, density/temperature diagnostics. / CoMP가 이용하는 Fe XIII 1074.7/1079.8 nm 물리 (준위도, A계수, 밀도·온도 진단). |
| **#25 Polarimetric sensitivity** / 편광 감도 | Establishes the theoretical framework for Stokes V detectability versus photon noise, which Babcock 1953 summarizes and CoMP applies. / Stokes V 검출한계의 이론적 틀; Babcock 1953 및 CoMP가 적용. |
| Babcock (1953) | Source of the S/N formula Eq. (1); foundational magnetograph paper. / 식 (1)의 S/N 출처; 자기망원경의 기초. |
| Lyot (1944); Evans (1949); Beckers & Dunn (1965) | Birefringent filter theory used to design CoMP's 4-stage calcite filter. / CoMP 4단 calcite 필터 설계의 복굴절 필터 이론. |
| House (1972, 1977); Sahal-Bréchot (1974a,b, 1977) | Coronal emission-line polarization theory; explains why Hanle saturates for Fe XIII. / 코로나 방출선 편광 이론; Fe XIII Hanle 포화 설명. |
| Casini & Judge (1999); Judge, Low & Casini (2006) | Theoretical advances that motivated CoMP's construction. / CoMP 제작을 촉발한 이론 발전. |
| Lin, Penn & Tomczyk (2000); Lin, Kuhn & Coulter (2004) | Demonstrated IR HgCdTe coronal polarimetry; established symmetry-based crosstalk subtraction that CoMP uses. / IR HgCdTe 코로나 편광측정을 입증; CoMP가 쓰는 대칭성 기반 crosstalk 차감 확립. |
| **Tomczyk et al. (2007, Science 317, 1192)** | THE scientific payoff of CoMP — first detection of Alfvén waves. / CoMP의 과학적 성과 — Alfvén 파 최초 검출. |
| Penn et al. (2004) | Source of the line-center uncertainty formula used for σ_B. / σ_B 계산의 선중심 불확도 공식. |
| Landi Degl'Innocenti (1992) | Canonical Zeeman-shift constant 4.67×10⁻¹² used in Eq. (10). / 식 (10)의 Zeeman 상수 출처. |

---

## References / 참고문헌

**Primary paper**:
- Tomczyk, S., Card, G.L., Darnell, T., Elmore, D.F., Lull, R., Nelson, P.G., Streander, K.V., Burkepile, J., Casini, R., Judge, P.G. "An Instrument to Measure Coronal Emission Line Polarization." *Solar Physics* **247**, 411-428 (2008). [DOI: 10.1007/s11207-007-9103-6]

**Key cited works**:
- Arnaud, J., Newkirk, G. Jr. *Astron. Astrophys.* **178**, 263 (1987). — IR linear polarization observations.
- Babcock, H.W. *Astrophys. J.* **118**, 387 (1953). — Photon-noise-optimal magnetograph.
- Beckers, J.M., Dunn, R.B. AFCRL Tech. Rep. 65-605 (1965). — Jones-matrix filter theory.
- Billings, B.H. *J. Opt. Soc. Am.* **37**, 738 (1947). — 4-stage birefringent filter.
- Casini, R., Judge, P.G. *Astrophys. J.* **522**, 524 (1999). — Fe XIII full Hanle+Zeeman theory.
- Evans, J.W. *J. Opt. Soc. Am.* **39**, 229 (1949). — Evans filter.
- Harvey, J.W. Ph.D. Thesis, U. Colorado (1969). — First green-line Zeeman detection.
- House, L.L. *Solar Phys.* **23**, 103 (1972); *Astrophys. J.* **214**, 632 (1977). — Coronal polarization theory.
- Judge, P.G. *Astrophys. J.* **500**, 1009 (1998). — Coronal line polarization review.
- Judge, P.G., Low, B.C., Casini, R. *Astrophys. J.* **651**, 1229 (2006). — IR line magnetography potential.
- Landi Degl'Innocenti, E. in *Solar Observations* (1992). — Zeeman-shift constant.
- Lin, H., Penn, M.J., Tomczyk, S. *Astrophys. J.* **541**, L83 (2000). — IR coronal polarimetry demo.
- Lin, H., Kuhn, J.R., Coulter, R. *Astrophys. J.* **613**, L177 (2004). — Sub-gauss coronal Zeeman; crosstalk removal.
- Lyot, B. *Ann. d'Astrophys.* **7**, 31 (1944). — Birefringent filter.
- Öhman, Y. *Stockh. Obs. Ann.* **19**, 3 (1956). — Polarizing beamsplitter filter exit.
- Penn, M.J., Lin, H., Tomczyk, S., Elmore, D., Judge, P.G. *Solar Phys.* **222**, 61 (2004). — Line-center uncertainty formula.
- Querfeld, C.W., Smartt, R.N. *Solar Phys.* **91**, 299 (1984). — IR linear polarization maps.
- Sahal-Bréchot, S. *Astron. Astrophys.* **32**, 147; **36**, 355 (1974). — Polarization theory.
- Smartt, R.N., Dunn, R.B., Fisher, R.R. *Proc. SPIE* **288**, 395 (1981). — COS coronagraph description.
- **Tomczyk, S., McIntosh, S.W., Keil, S.L., Judge, P.G., Schad, T., Seeley, D.H., Edmondson, J. "Alfvén Waves in the Solar Corona." *Science* **317**, 1192 (2007).** — CoMP's first major discovery.

---

## Notes End / 노트 끝

This document is self-contained: a reader unfamiliar with the original CoMP paper should now understand why IR forbidden lines were chosen, how the instrument was designed and calibrated, what its photon-noise-limited performance demonstrates, and why this 2008 engineering paper is the foundation of modern ground-based coronal magnetometry.

본 문서는 독립적이다: 원 논문을 읽지 않은 독자도 왜 근적외 금지선이 선택되었는지, 장비가 어떻게 설계·보정되었는지, 광자잡음 한계 성능이 무엇을 의미하는지, 그리고 이 2008년 장비 논문이 현대 지상 코로나 자기장 관측의 기반인 이유를 이해할 수 있다.
