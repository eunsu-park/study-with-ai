---
title: "The Solar Orbiter Heliospheric Imager (SoloHI)"
authors: "R. A. Howard, A. Vourlidas, R. C. Colaninno, C. M. Korendyke, S. P. Plunkett, et al."
year: 2020
journal: "Astronomy & Astrophysics 642, A13"
doi: "10.1051/0004-6361/201935202"
topic: Solar_Observation
tags: [heliospheric_imager, Solar_Orbiter, SoloHI, white_light_corona, APS_CMOS, stray_light_baffle, Thomson_scattering, F_corona]
status: completed
date_started: 2026-04-25
date_completed: 2026-04-25
---

# 58. The Solar Orbiter Heliospheric Imager (SoloHI) / Solar Orbiter 헬리오스피어 영상기

---

## 1. Core Contribution / 핵심 기여

**한국어**: 본 논문은 ESA/NASA Solar Orbiter 임무에 탑재된 단일 광시야(40°×40°, detector corner까지 48°) 백색광 망원경 SoloHI의 설계와 발사 전 성능을 종합 정리한 instrument paper다. SoloHI는 STEREO/SECCHI HI-1을 직계 계승하지만 세 가지를 처음 시도한다. 첫째, HI-1 + HI-2 두 망원경을 하나의 망원경으로 대체하면서 inner FOV를 HI-1의 두 배(5.4° vs 약 13°)로 확장했다. 둘째, 전통적 CCD 대신 4-die 모자이크 형태의 custom CMOS APS(3968×3968 픽셀, 5T pinned-photodiode, 32% QE, 5.8 e⁻ read noise) 검출기를 채택하여 mass·전력·방사선 내성을 동시에 개선했다. 셋째, Solar Orbiter의 0.28 AU 근일점 + 30°+ 황도면 경사 궤도 덕분에 5°–45° elongation 영역의 corona–heliosphere를 Sun에 매우 가깝게, 그리고 처음으로 out-of-ecliptic 시점에서 영상화한다. 이를 가능하게 하는 핵심 엔지니어링은 다단 baffle 시스템(F1–F4 + I0 forward baffle, 9개 interior baffle, AE1·AE2 light-trap baffle, peripheral baffle)으로 outer FOV에서 $10^{-13}\,B_\odot$까지 stray-light를 억제한다.

**English**: This paper is the comprehensive instrument description and pre-flight performance summary for SoloHI, the wide-field (40°×40°, 48° at detector corners) white-light telescope on ESA/NASA Solar Orbiter. SoloHI directly inherits from STEREO/SECCHI HI-1 but introduces three firsts. (i) It collapses the HI-1 + HI-2 telescope pair into a single telescope while doubling the inner-FOV reach (5.4° vs ~13° AUeq). (ii) It replaces the heritage CCD with a custom CMOS APS in a 4-die pinwheel mosaic (3968×3968 pixels, 5-transistor pinned-photodiode, 32% QE, 5.8 e⁻ read noise), simultaneously improving mass, power, and radiation tolerance. (iii) By exploiting Solar Orbiter's 0.28 AU perihelion and >30° ecliptic-inclination orbit, SoloHI is the first heliospheric imager flown both close to the Sun and out of the ecliptic, providing a fundamentally new vantage on the inner heliosphere over elongations 5°–45°. The enabling engineering is the multi-stage baffle system — five forward baffles (F1–F4 plus I0), nine interior baffles, two aperture-light-trap baffles (AE1, AE2), and a peripheral baffle — that suppresses stray light to $10^{-13}\,B_\odot$ at the outer FOV.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction and Mission Context / 도입 및 임무 맥락 (§1)

**한국어**: Solar Orbiter는 2020년 2월 발사 예정인 ESA/NASA 협력 임무로, 6개의 remote-sensing(RS) + 4개의 in-situ(IS) 총 10개 계기를 탑재한다. Venus·Earth flyby로 중력 보조를 받아 최소 근일점 0.28 AU(Helios와 유사)에 도달하며, 또한 황도면에서 최소 30° 이상 경사된 궤도에 진입한다. SoloHI는 NRL이 제작한 단일 백색광 imager로 5.4°–44.9° elongation 범위를 영상화하고, boresight는 Sun 중심에서 anti-ram 동쪽 25° 방향에 위치한다. SECCHI HI-1(2006~)에 비해 인너 시야가 두 배 확장되어 LASCO C3와 PSP/WISPR을 잇는 핵심 연결고리가 된다. Solar Orbiter의 또 다른 RS·IS 계기 — RPW(Radio and Plasma Waves), Metis(coronagraph), EUI(EUV imager), SPICE(spectrograph) — 와 함께 SoloHI는 corona–solar wind의 물리적 연결성(physical connectivity)을 추적하는 기둥 역할을 한다.

**English**: Solar Orbiter, an ESA/NASA mission planned for February 2020 launch, carries six remote-sensing (RS) and four in-situ (IS) instruments. Venus and Earth gravity assists reduce the minimum perihelion to 0.28 AU (Helios-class) and tilt the orbit > 30° out of the ecliptic. SoloHI, built by NRL, is a single white-light imager covering 5.4°–44.9° elongation with boresight 25° east of Sun centre (anti-ram). Compared with SECCHI HI-1 (2006–), the inner FOV is doubled, making SoloHI the bridge between LASCO C3 (≤30 R☉) and PSP/WISPR. Together with Solar Orbiter's other RS instruments (RPW, Metis coronagraph, EUI, SPICE) and IS instruments, SoloHI is one of the pillars used to trace the physical connectivity between the corona and the solar wind.

### Part II: Science Objectives / 과학 목표 (§2)

**한국어**: §2는 SoloHI의 과학을 Solar Orbiter의 네 가지 미션 질문으로 정렬한다. (1) **태양풍과 코로나 자기장의 기원** (§2.1): SoloHI는 streamer 위쪽에서 형성되는 plasma blob의 속도·가속도 프로파일을 추적한다. Fig. 1은 LASCO blob fit으로 $V_0=298.3$ km/s, $r_0=8.1\,R_\odot$, $r_1=2.8\,R_\odot$를 보고했고, SoloHI는 42 R☉까지 같은 fit을 검증할 수 있다. 또한 heliospheric plasma sheet(HPS)와 heliospheric current sheet(HCS) 관계, MHD 모델 검증, 밀도 turbulence power spectrum(Marsch et al. 2000) 측정에 활용된다. (2) **태양 transient의 heliosphere 변동성 driving** (§2.2): SECCHI HI는 50–60 R☉ 이상에서 ICME 추적이 어려웠는데(긴 노출, 작은 신호) SoloHI는 inner FOV를 2배 늘리고 더 가까이 비행함으로써 0.28 AU 이내에서 CME, SIR, flux rope를 고분해능으로 추적한다. WISPR·SECCHI·SOHO와 합치면 같은 CME를 3개 다른 시점에서 동시 촬영하는 것이 가능해진다. (3) **SEP 생성 충격파** (§2.3): CME-driven shock는 SEP의 주요 가속원이지만 발생 높이·3D 범위가 불확실하다. 0.28 AU 근일점에서 5.2–42 R☉ FOV (52″ AUeq 분해능)로 LASCO C3의 두 배 분해능 + 더 가까운 거리로 충격파를 직접 영상화 가능. (4) **3D heliosphere 구조** (§2.4): 황도면 밖 시점은 streamer·CME의 longitudinal extent를 직접 측정 가능하게 한다. Solar Orbiter는 2주 안에 남·북 위도 극단을 sweep하여 LASCO 단일 시점으로는 불가능한 tomographic 재구성을 가능하게 한다.

**English**: Section 2 aligns SoloHI's science with the four Solar Orbiter mission questions. **(1) Solar wind & coronal magnetic field origin** (§2.1): SoloHI tracks streamer blob acceleration. Fig. 1 shows the LASCO fit $V^2 = V_0^2[1 - e^{-(R-R_1)/R_0}]$ with $V_0=298.3$ km/s, $r_0=8.1\,R_\odot$, $r_1=2.8\,R_\odot$; SoloHI extends this to 42 R☉ at higher S/N. It also probes the HPS–HCS relationship (Fig. 2 with MHD model overlay), validates MHD models, and supports density-turbulence power-spectrum measurements (Marsch et al. 2000) via dedicated ROI programs. **(2) Solar transients driving heliospheric variability** (§2.2): HI-1 lost CMEs above ~50–60 R☉ due to long exposure (20–60 min) and small fine structure; SoloHI doubles the inner FOV and flies closer, tracking CMEs, SIRs, and flux ropes inside 0.5 AU. With WISPR, SECCHI, and SOHO, the same CME can be observed from 3+ vantage points simultaneously. **(3) SEP-driving shocks** (§2.3): CME-driven shocks accelerate SEPs, but their formation height and 3D extent are poorly constrained. At 0.28 AU, SoloHI's FOV (5.2–42 R☉, 52″ AUeq half-resolution) provides twice LASCO C3's resolution from a closer vantage, enabling direct shock imaging. **(4) 3D heliosphere structure** (§2.4): Out-of-ecliptic vantage gives the longitudinal extent of streamers/CMEs directly. Solar Orbiter sweeps between southern and northern latitudinal extremes within ~2 weeks, enabling tomographic reconstructions inaccessible from LASCO alone.

#### §2.5 SoloHI unique science / 고유 과학

**한국어**: 5 R☉ 이상에서 가시광 신호는 F-corona(zodiacal dust, IPD)에 의해 dominate된다. 정확한 F-corona 제거가 필수이며, Stenborg & Howard 2017a와 Stauffer et al. 2018은 LASCO/SECCHI 데이터에서 F-corona가 ecliptic longitude·heliocentric distance에 따라 상수가 아니고 비축대칭이라는 점을 발견했다. SoloHI는 이를 위해 single-image background 모델(Stenborg & Howard 2017b)을 사용한다. 30° 이상 황도경사에서 F-corona를 처음으로 out-of-ecliptic 시점에서 정량화할 수 있어, IPD의 3D 분포·시간 변동(혜성·CME 상호작용)을 처음으로 단층촬영할 수 있다. 금성 궤도의 dust ring(Leinert & Moster 2007; Jones et al. 2013, 2017)과 수성 궤도 dust 증가(Stenborg et al. 2018b)도 검증 대상이다.

**English**: Above 5 R☉ the visible signal is dominated by F-corona (interplanetary dust). Accurate F-corona removal is critical (Hayes et al. 2001) but recent papers (Stenborg & Howard 2017a; Stauffer et al. 2018) showed it is neither constant in time nor axisymmetric. SoloHI uses a new single-image background technique (Stenborg & Howard 2017b). With > 30° ecliptic inclination it can quantify F-corona out of the ecliptic for the first time, performing 3D dust tomography. Targets include the Venus dust ring (Leinert & Moster 2007; Jones et al. 2013, 2017) and the Mercury-orbit ~3–5 % dust enhancement (Stenborg et al. 2018b).

##### §2.5.1 Signal-to-noise ratio / 신호대잡음비

**한국어**: 신호 = K-corona(전자) + F-corona(먼지) + integrated star light. F-corona가 SoloHI 시야 전체에서 dominant 성분이다. 1σ photon noise detection limit은 0.88 AU에서 30 min, 0.28 AU에서 30 s 노출 기준 (Fig. 3). S/N 기준은 단순/알려진 타겟 ≥ 5, 복잡/미지 타겟 ≥ 30. 외곽 FOV에서는 픽셀 binning과 long integration으로 충족, 내부 FOV에서는 subframe과 짧은 노출로 cadence 향상.

**English**: The signal is the sum of K-corona (electron-scattered), F-corona (dust-scattered), and integrated unresolved-star light, with F-corona dominant. The 1σ photon-noise detection limit is shown in Fig. 3 for 30 min @ 0.88 AU and 30 s @ 0.28 AU. S/N criteria are ≥ 5 for simple known targets and ≥ 30 for complex unknown targets (Rose 1948; Barrett 1990). The criteria are met by binning + long integration in the outer FOV and by subframes + faster cadence in the inner FOV.

##### §2.5.2 Thomson surface considerations / Thomson 표면 고려

**한국어**: Thomson surface는 우주선–Sun 선분을 지름으로 하는 구면이고, 이 표면에서 전자가 가장 효율적으로 scattering한다. Fig. 4는 0.28 AU 근일점("2"), 0.34 AU("1","3")에서 LOS의 5%, 50%, 95% brightness integral 위치를 보여준다. 90%의 scene brightness는 가장 바깥 두 실선 사이에 있다. 0.28 AU 근일점에서 SoloHI는 사실상 Sun 중심 40 R☉ 이내의 local imager가 되며 1 AU에서의 scattering 양상과 매우 다르다.

**English**: The Thomson surface is the sphere with the spacecraft–Sun line as diameter — electrons there scatter most efficiently. Fig. 4 plots loci of 5 %, 50 %, 95 % LOS-integrated brightness at perihelion (0.28 AU) and at 0.34 AU. Ninety percent of scene brightness lies between the two outermost solid lines. At 0.28 AU SoloHI is effectively a local imager within ~40 R☉ of Sun centre — qualitatively different from 1 AU scattering geometries.

### Part III: Instrument Overview / 기기 개요 (§3)

**한국어**: SoloHI는 25° boresight, 40°×40° 광시야(corner 48°) 단일 망원경. 입사구는 16 mm 사각(라운드 19 mm 직경). 5-element 굴절 렌즈, 5.4° 완전 vignetting / 9.3° 완전 unvignetting. 4-die APS 모자이크(3968×3968 px)로 36.7″ full-resolution / 73.5″ 2×2 binned. 0.28 AU에서 1 AU 환산 분해능 10.3″/20.6″. spectral bandpass 500–850 nm (HI-1의 630–730 nm보다 넓어 photon 수집량 증가). 노출 0.1–65 s (nominal 30 s). Telemetry 53.2 Gbits/orbit. SIM(Instrument Module) 15.18 kg + SPS(Power System) 1.38 kg, 평균 13.5 W. 핵심 design 변경 두 가지: (1) HI-1+HI-2를 단일 망원경으로 (mass·volume·power 절감), (2) CCD 대신 custom CMOS APS — on-chip 신호처리로 driving 요구 감소·radiation tolerance↑, 다만 read noise↑·QE↓·column pattern noise 등 새로운 도전 존재.

**English**: SoloHI is a single refractive telescope with 25° boresight, 40°×40° FOV (48° at corners), 16 mm square (19 mm diameter rounded) entrance aperture, 5-element lens, fully vignetted at 5.4° and fully unvignetted at 9.3°. The detector is a 4-die APS mosaic (3968×3968 px) giving 36.7″ full / 73.5″ 2×2 binned (10.3″/20.6″ AUeq at 0.28 AU). Spectral bandpass is 500–850 nm (broader than HI-1's 630–730 nm to gather more photons). Exposure 0.1–65 s (30 s nominal); telemetry 53.2 Gbits/orbit; SIM mass 15.18 kg, SPS 1.38 kg, mean power 13.5 W. Two key design changes from heritage: (1) merging HI-1+HI-2 into one telescope (mass/volume/power savings); (2) replacing the CCD with a custom CMOS APS — fewer drive signals, better radiation tolerance, at the cost of higher read noise, lower QE, and column-pattern noise that required a dedicated calibration program.

#### §3.2 Accommodation challenges / 탑재 난제

**한국어**: (i) **Stray light**: SoloHI는 anti-ram side에 위치하지만 8.2 m 태양 어레이가 SIM 후방 0.6 sr를 차지하여 BRDF 따라 산란광이 baffle 뒷면을 비춘다. ray-tracing 모델로 정량화하고 in-flight test로 검증. (ii) **Dust impacts**: 0.28 AU 먼지 밀도는 1 AU보다 낮고 anti-ram 쪽이라 영향 미미(SECCHI HI-1 12+년 동안 손상 없음). (iii) **Radiation**: TID 60 krad 마진 (Al 2.54 mm shield 후), SEU/transient용 LET 25 MeV cm²/mg 임계 부품 회피. APS는 NIEL과 CTE 손상에 대해 CCD보다 우월. (iv) **EMI/EMC**: shutter 등 모터 없음(door는 일회성). (v) **Contamination**: propulsion jet droplet 차단용 baffle 추가.

**English**: (i) **Stray light**: SoloHI sits on the anti-ram side, but the 8.2 m solar array occupies ~0.6 sr behind the SIM, so reflected light off the array hits the back of the baffles. Quantified via ray-tracing of the SIM CAD model and validated in-flight. (ii) **Dust**: 0.28 AU dust density is lower than at 1 AU, and the anti-ram orientation makes impacts insignificant (HI-1 saw none in 12+ years). (iii) **Radiation**: TID margin 60 krad behind 2.54 mm Al; LET threshold ≤ 25 MeV cm²/mg parts excluded. APS chosen partly because radiation-induced charge-transfer-efficiency loss is greatly reduced vs CCDs. (iv) **EMI/EMC**: only mechanism is the one-shot door, eliminating shutter/wheel issues. (v) **Contamination**: baffles added in front of propulsion jets to block droplet contamination.

### Part IV: Instrument Design / 기기 설계 (§4)

#### §4.1–4.2 Optical and Lens Design / 광학·렌즈 설계

**한국어**: 5-element 광각 렌즈(Jenoptik Optical Systems). 직접 햇빛은 spacecraft heat shield가 차단. 광 설계 파라미터는 Table 2: 1.082 mm/deg plate scale @ 20° 시야각, F/3.48, focal length 55.9 mm, 측정 분해능 1.3 arcmin, 첫 element는 LASCO에서 22년 검증된 radiation-tolerant glass(약 0.5%/year 성능 저하). 렌즈 cell은 Ti6Al4V로 -45°C에서도 안정. anti-reflective + absorbing coating으로 ghost 억제. 7번/9번 면에 long/short pass coating으로 500–850 nm bandpass 정의. forward baffle 때문에 5.4°→8.8° 인너 시야는 cos³θ vignetting을 따라 추가 vignetting됨.

**English**: A 5-element wide-angle lens (Jenoptik). Direct sunlight is blocked by the spacecraft heat shield. Table 2 lists 1.082 mm/deg plate scale at 20° field, F/3.48, 55.9 mm focal length, 1.3 arcmin measured resolution. The first element is the same radiation-tolerant glass used in LASCO (22 yrs heritage, ~0.5 %/yr degradation). The titanium-alloy lens cell stays stable to −45 °C. Anti-reflective and absorbing coatings suppress ghosts. Long/short-pass coatings on surfaces 7/9 set the 500–850 nm bandpass. Forward-baffle vignetting from 5.4° to 8.8° follows roughly a cos³θ law, plus the natural cos⁴ falloff.

#### §4.3 Stray-light rejection / 잡광 억제 — 핵심 섹션

**한국어**: SoloHI에서 stray-light 차폐는 사실상 이 instrument의 존재 이유다. K-corona 신호는 disk brightness보다 10⁹–10¹³ 배 어둡기 때문이다. 네 종류의 baffle 시스템:

1. **Forward baffles (F1, F2, F3, F4 + I0)**: 5단으로 직접 sunlight·반사광·회절광 차단. F1은 spacecraft heat shield(HS) 가장자리로 정의되어 실질적 첫 baffle, F2/F3는 그 회절광을 추가 감쇠, F4는 정렬 오차 마진용, I0는 light-trapping용. 각 baffle은 약 3 orders of magnitude의 회절광 감쇠를 제공하여 5단으로 $10^{-13}\,B_\odot$ 수준 도달. F1과 HS edge 거리 68.27 cm, F1–shadow line 각도 2.05°(0.95° spacecraft offpoint @ 0.28 AU + 1° 임의 + 0.1° 정밀도).

2. **Interior baffles**: 9개 baffle을 light trap 방향으로 배열. SIM 위로 들어온 stray light가 입사구에 도달하기 전에 두 번 bounce하게 강제하여 reflectance 차감 효과를 제곱으로.

3. **Aperture light-trap baffles (AE1, AE2)**: F4 통과 잔여 회절광 + interior baffle 윗면에서 반사된 stray light를 흡수하는 2단.

4. **Peripheral baffle**: interior baffle 위 평면 baffle. 망원경의 FOV에서 spacecraft heat shield의 직접 시야를 차단.

baffle 재질은 알루미늄에 black anodisation, Laser Black, A382 또는 Z307 paint. interior baffle은 측면 wall + 전·후·하 cover로 통합 unit, no-touch Laser Black coating 사용. forward·light-trap baffle은 ±0.5 mm 정밀도로 정렬, Invar braces로 forward baffle 정렬 안정성 확보. **Diffracted stray light** 요구치는 0.28 AU에서 인너 FOV $1\times10^{-10}\,B_\odot$ → 외곽 FOV $1\times10^{-13}\,B_\odot$. NRL의 SCOTCH(Solar Coronagraph Optical Test Chamber, Korendyke et al. 1993) 챔버에서 비행 모델 시험. **Reflected stray light**: 8.2 m × 1.2 m 전체 조명되는 solar array가 SIM 시야의 0.6 sr 차지. BRDF 측정과 ray-tracing 모델로 sufficient suppression 확인 (Fig. 12 비교).

**English**: Stray-light suppression is the raison-d'être of SoloHI's mechanical design — the K-corona signal is 10⁹–10¹³ times fainter than disk brightness. Four baffle systems:

1. **Forward baffles (F1–F4 + I0)**: five stages blocking direct sunlight, reflected sunlight, and diffracted light. F1 is effectively the spacecraft heat-shield edge; F2/F3 attenuate diffraction further; F4 absorbs alignment errors; I0 traps residual light. Each baffle provides ~3 orders of attenuation, yielding $10^{-13}\,B_\odot$ at the outer FOV after five stages. F1-to-HS distance is 68.27 cm; F1-to-shadow line angle is 2.05° (= 0.95° offpoint @ 0.28 AU + 1° margin + 0.1° pointing accuracy).

2. **Interior baffles**: a set of nine baffles all pointing to the light trap. Any stray-light source above the SIM must reflect twice before reaching the entrance aperture, squaring the reflectance attenuation.

3. **Aperture light-trap baffles (AE1, AE2)**: trap residual diffracted light coming over F4 and reflected stray light off the interior baffle tops.

4. **Peripheral baffle**: a planar baffle on top of the interior baffles, blocking the telescope's direct view of the spacecraft heat shield.

All baffles are aluminium with one of: black anodisation, Laser Black, A382, or Z307 paint. The interior baffle box is a unified unit using no-touch Laser Black. Alignment tolerances are ±0.5 mm; the forward-baffle ledge braces were upgraded from Ti6Al4V to Invar. **Diffracted stray-light requirement** at 0.28 AU: $1\times10^{-10}\,B_\odot$ at the inner FOV to $1\times10^{-13}\,B_\odot$ at the outer FOV. Verified in NRL's SCOTCH test chamber (Korendyke et al. 1993). **Reflected stray light**: the 8.2 × 1.2 m solar array, fully illuminated, occupies ~0.6 sr from the SIM. BRDF measurements + ray-tracing of the SIM/array system showed compliance (Fig. 12 compares model and laboratory measurement).

#### §4.4 Mechanical Design / 기계 설계

**한국어**: 모듈 구성: instrument enclosure(M55J carbon-fibre composite + Ti6Al4V clips, EA9394+EA9396 접착), 외부 baffle, 내부 baffle box, FPA. 4단계 baffle assembly + door + 광학계가 SIM 안에 일체. PFM(Protoflight) 모델 + FPA QM. **Door**: T-300 satin weave 직물, post-machined lip로 labyrinth seal, 두 cup-cone joint preload(strain-sense bolt + Belleville washer), one-shot ERM(Ejection Release Mechanism), 두 torsion spring + kickoff spring으로 225°까지 열림. 두 ascent vent + sintered metal filter. **Instrument mount**: SIM을 220 mm 들어올리는 M55J/Ti6Al4V isolation 시스템, 1차 mode > 140 Hz. Ti spacer + shim stack로 정렬.

**English**: Modules: instrument enclosure (M55J carbon-fibre composite with Ti6Al4V clips, bonded with EA9394 + EA9396), four sets of external baffles, the interior baffle box, and the FPA. Protoflight (PFM) approach, with a separate FPA QM. **Door**: T-300 satin-weave fabric, post-machined lip forming a labyrinth seal, preloaded against two cup-cone joints (strain-sensed bolt + Belleville washer), released by a one-shot ERM, then driven open by two torsion springs + a kickoff spring to ~225°. Two ascent vents with sintered-metal filters. **Instrument mounts**: M55J + Ti6Al4V brackets raise the SIM ~220 mm off the deck; primary mode > 140 Hz; Ti spacer + shim stack for alignment.

#### §4.5 Electrical Design / 전기 설계

**한국어**: 4개 sub-system: SoloHI Power System (SPS) — Relay Electronics Card (REC) + Power Electronics Card (PEC); SoloHI Camera Electronics (SCE) — Camera Card (CC) + Processor Card (PC); Detector Readout Board (DRB); Detector Interface Board (DIB). PC는 Aeroflex Gaisler LEON3FT CPU, 256 MB SDRAM + 64 kB PROM + 4 MB MRAM, RTAX2000 FPGA @ 25 MHz로 20 MIPS. RTEMS 4.10 RTOS + C++. CC는 image acquisition·signal processing 수행하는 "smart camera"(256 MB memory). camera FPGA가 SEP/cosmic-ray scrub, bias subtraction, pixel binning, image summing(16-bit packing), truncation 처리. CMOS APS는 row/column addressing이라 한 die 안 다른 row 블록을 다른 timing으로 읽기 가능 → 영역별 다른 cadence 가능.

**English**: Four subsystems: SPS (REC + PEC); SCE (CC + PC); DRB; DIB. The PC uses an Aeroflex Gaisler LEON3FT (256 MB SDRAM, 64 kB PROM, 4 MB MRAM, RTAX2000 FPGA @ 25 MHz, 20 MIPS) with RTEMS 4.10 RTOS and C++. The CC is a "smart camera" with 256 MB of memory whose FPGA does on-board SEP/cosmic-ray scrub, bias subtraction, pixel binning, image summing (16-bit packing), and truncation. Because CMOS APS pixels are row/column-addressed, different blocks of rows on the same die can be read at different rates — enabling per-region cadence.

#### §4.6 Active Pixel Sensor / 능동 픽셀 센서 — 핵심 섹션

**한국어**: SoloHI 검출기는 4개의 APS die (각 2048 × 1920 px, 10 µm pitch) 모자이크. 두 변(top·right) buttable 구조 때문에 pinwheel 구성을 채택 — die 사이 gap < 1 mm (전형 0.88 mm), 외각 10 row + 10 column이 opaque(bias·dark 신호 측정용). 전체 imaging area 38.84 mm × 40.12 mm. 패키지는 molybdenum (열팽창계수가 APS와 잘 매칭, machinable, non-ferrous). solar disk 방향은 die mosaic의 inner corner이고 increasing heliocentric distance는 outer corner 방향. **5T pixel 구조**: 5개 트랜지스터(reset, gain control, transfer gate, source follower, row select). photon은 pinned photodiode(픽셀 면적의 ~63% 차지 = fill factor)에서 모임. transfer gate로 sense node 분리, MIM(metal-insulator-metal) capacitor로 low/high gain 선택. CDS(Correlated Double Sampling): 픽셀 reset voltage를 capacitor에 저장 후 photo-electron voltage와 차감하여 1/f noise 억제. **progressive scan rolling-curtain shutter**로 device readout 중에도 photon 수집 가능 → 별도 mechanical shutter 불필요. **성능**: 32% QE (480–750 nm 평균), >86 400 e⁻ low-gain full well / >19 200 e⁻ high-gain, read noise 35.1 e⁻ (low gain) / 5.8 e⁻ (high gain) — 이는 과학 등급 CCD에 비견. dark current < 0.3 e⁻/px/s @ BOL, < 2 e⁻/px/s @ EOL. MTF > ideal 10 µm pitch detector의 80%. Pixel readout rate 4 MHz/port (비행 시 2 MHz). die는 Jazz Semiconductor Inc. wafer 11–15에서 채택, 5T architecture 상세는 Janesick et al. 2010, 2013과 Korendyke et al. 2013 참고.

**English**: The detector is a 4-die APS mosaic (each die 2048 × 1920 px, 10 µm pitch). The two-side buttable design forces a "pinwheel" arrangement — gaps < 1 mm (typically 0.88 mm) — with the outer 10 rows and 10 columns of each die opaque (used for bias/dark). Imaging area 38.84 × 40.12 mm. Package is molybdenum (CTE-matched to silicon, machinable, non-ferrous). The solar disk direction is the inner corner of the mosaic; increasing heliocentric distance is the outer corner. **5T pixel architecture**: five transistors (reset, gain control, transfer gate, source follower, row select). Photons collect in a pinned photodiode (~63 % fill factor). A transfer gate separates the sense node, a MIM (metal-insulator-metal) capacitor sets low/high gain. **CDS** (Correlated Double Sampling) stores the reset voltage on a capacitor and subtracts the photo-electron voltage, suppressing 1/f noise. A **progressive-scan rolling-curtain shutter** allows photon collection during readout, eliminating any mechanical shutter. **Performance**: 32 % QE (480–750 nm average), > 86 400 e⁻ low-gain full well, > 19 200 e⁻ high gain, read noise 35.1 e⁻ low gain / 5.8 e⁻ high gain (CCD-class). Dark current < 0.3 e⁻/px/s @ BOL, < 2 e⁻/px/s @ EOL. MTF > 80 % of an ideal 10 µm-pitch detector. Pixel readout 4 MHz/port (flight rate 2 MHz). Die from wafers 11–15 of the Jazz Semiconductor lot run; full architecture details in Janesick et al. 2010, 2013 and Korendyke et al. 2013.

#### §4.7 Thermal Design / 열 설계

**한국어**: SIM-spacecraft 열전도 인터페이스 ±1 W. APS는 -55°C 미만으로 passive radiator(-YSIM 면)로 냉각, Z93 white paint coated. detector goal -65 ± 10°C. 렌즈는 detector에 가까워 -45°C 이상 유지(렌즈 자체 heater + thermal insensitivity). off-pointing 시 직접 햇빛이 50 s 비춰도 APS 100°C 미만으로 유지.

**English**: SIM–spacecraft conductive interface ±1 W. APS cooled below −55 °C via a passive radiator on the −YSIM panel coated with Z93 white paint; detector goal −65 ± 10 °C. The lens is heated/insulated to stay above −45 °C. Even with worst-case 50 s direct-Sun off-pointing, APS stays below 100 °C.

#### §4.8 Flight Software / 비행 소프트웨어

**한국어**: SECCHI flight software 기반(>80% 재사용). NVRAM에 task 단위로 저장. camera FPGA가 1×1, 2×2, 4×4, 8×8 binning, masking, sub-region, lossless(Rice 1978) 또는 lossy(H-compress, Wang et al. 1998) 압축, header 생성. 이미지당 최대 16개의 다른 처리 파라미터 적용 가능. 36 h centred-on-perihelion 고케이던스 관측 + 이외 lower cadence synoptic. 53.1 Gbits/orbit 할당량 자동 modify 기능.

**English**: Heritage SECCHI flight software with > 80 % reuse. Tasks loaded individually into NVRAM. Camera FPGA performs 1×1, 2×2, 4×4, 8×8 binning, masking, sub-regions, lossless (Rice 1978) or lossy (H-compress, Wang et al. 1998) compression, and header generation. Up to 16 different processing parameters per detector. A 36 h high-cadence window is centred on perihelion; lower-cadence synoptic observations otherwise. Auto-modify if 53.1 Gbits/orbit quota will be exceeded.

### Part V: Operations, Data Processing, Products / 운영·데이터 처리·산출물 (§5)

**한국어**: nominal 관측은 orbit 당 3 RS window × ~10 days. 30 day 표준 image 시퀀스. Table 4는 5개 region(near/far perihelion, 남·북 out-of-ecliptic)에 대해 image type(full frame, inner FOV subframe, radial swath), 시야 범위 5°–25°, 25°–35° 등, bin size, downlink pixel count, image size(압축 전·후), cadence(24 min ~ 72 min) 정리. **Calibration**: 매 perihelion 사이 (1) degradation 평가, (2) APS annealing, (3) pre-perihelion cal sequence 수행. star tracking으로 photometric cal 3% 정확도 목표(SOHO/STEREO 경험 기반). **Data products** (Table 5): Level 0 (compressed packets) → Level 1 (raw count FITS) → Level 2 (calibrated, solar brightness) → Level 3 (movies, mosaics, Carrington maps, derived densities). Level 3는 background subtraction technique (Stenborg & Howard 2017b)을 적용해 100–1000× 약한 wind features 가시화. SoloHI Science Operations Center (ISOC at NRL) 운영, ESA SOAR가 primary archive, NASA SDAC 사본.

**English**: Nominal observations: 3 RS windows × ~10 days per orbit, 30 days of standard image sequences. Table 4 details five regions (near/far perihelion, southern/northern out-of-ecliptic) with image types (full frame, inner-FOV subframe, radial swath), FOV ranges (5°–25°, 25°–35°, etc.), bin sizes, downlink pixel counts, image sizes (with/without compression), and cadences (24–72 min). **Calibration**: between perihelion passes a three-phase sequence runs — (1) degradation assessment, (2) APS annealing, (3) pre-perihelion cal — followed by in-flight star tracking for ~3 % photometric accuracy (per SOHO/STEREO experience). **Data products** (Table 5): L0 (compressed packets) → L1 (raw-count FITS) → L2 (calibrated, solar-brightness units) → L3 (movies, mosaics, Carrington maps, derived densities). L3 uses the background-subtraction technique of Stenborg & Howard 2017b to reveal features 100–1000× weaker than the corona. ISOC (NRL) operates SoloHI; ESA SOAR is primary archive, NASA SDAC mirrors.

### Part VI: Summary / 요약 (§6)

**한국어**: SoloHI는 첫 out-of-ecliptic + 0.28 AU 근일점 heliospheric imager. 5°–45° FOV로 streamer·CME·CIR을 ecliptic 위·아래에서 영상화. coronal neutral line, CME 종방향 extent, F-corona 분포, 혜성 dust 방출 등 새로운 현상 관측 가능. SoloHI + WISPR + LASCO + Metis + EUI + SPICE의 결합은 corona–heliosphere 연결을 처음으로 다중 시점에서 동시에 추적한다.

**English**: SoloHI is the first out-of-ecliptic, 0.28 AU heliospheric imager. With 5°–45° FOV it images streamers, CMEs, CIRs both in and above/below the ecliptic, opening views of the coronal neutral line, CME longitudinal extent, F-corona distribution, and cometary dust release. Together with WISPR, LASCO, Metis, EUI, and SPICE, SoloHI provides the first simultaneous multi-vantage tracing of corona-to-heliosphere connectivity.

---

## 3. Key Takeaways / 핵심 시사점

1. **SoloHI = HI-1 + HI-2 → 단일 망원경 / single-telescope replacement** — Heritage SECCHI는 두 별도 망원경(HI-1: 4°–24°, HI-2: 19°–89°)을 썼지만 SoloHI는 5.4°–44.9° 단일 시야로 통합, mass·power·volume·complexity 모두 절감하면서도 inner FOV는 두 배 더 가까이. The key trade is to lose extreme outer angles (>45°) and replace them with a closer-to-Sun perihelion (0.28 AU).

2. **CMOS APS 채택은 instrumentation 패러다임 전환 / CMOS APS adoption is a paradigm shift** — 우주 광학계에서 30년 가까이 CCD가 표준이었는데, SoloHI는 4-die pinwheel APS mosaic을 처음 fully qualify. 5T pinned-photodiode + MIM cap + on-chip CDS + rolling-curtain shutter라는 조합이 기계적 shutter 제거·radiation tolerance↑·driving 신호 단순화로 mass/EMI/power를 동시에 줄인다. Future heliospheric missions will likely follow this CMOS path.

3. **Stray-light 억제는 instrument의 핵심 / Stray-light is the central engineering challenge** — F1–F4+I0의 5단 forward baffle, 9개 interior baffle, AE1/AE2 trap, peripheral baffle의 연속이 K-corona 신호 (10⁻¹³ B☉)를 가능하게 한다. 각 baffle 약 3 orders of magnitude 회절 감쇠 → 5단으로 10⁻¹⁵ 누적. 이는 LASCO/SECCHI에서 발전된 기법을 Solar Orbiter heat-shield edge가 첫 baffle인 새로운 기하에 맞춰 재설계한 것이다. The lesson generalises to all close-Sun coronagraphs: heat-shield edges become first baffles.

4. **0.28 AU 근일점이 imaging 의미를 바꾼다 / 0.28 AU perihelion changes what "imaging" means** — Thomson surface가 spacecraft–Sun 거리에 따라 dramatically 좁아진다. Fig. 4: 0.28 AU에서 90% scene brightness가 Sun 중심 ~40 R☉ 안에 들어와 SoloHI는 사실상 local imager로 변모한다. 이것이 LASCO(1 AU) 데이터와 같은 처리법을 단순 적용할 수 없는 이유이고, 새로운 background 모델(Stenborg & Howard 2017b)이 필요한 이유다.

5. **Out-of-ecliptic 시점은 heliosphere의 3D tomography 도구 / Out-of-ecliptic vantage = 3D tomography tool** — 30°+ 황도면 경사에서 streamer·HCS·F-corona·CME longitudinal extent를 직접 측정 가능. 2주에 남·북 sweep하여 LASCO 단일 시점 또는 STEREO 두 시점으로는 불가능했던 tomographic reconstruction을 시도. F-corona의 ecliptic-방향 비대칭을 처음 정량화하여 IPD 분포 모델 검증 가능.

6. **Multi-vantage 동시관측 시대의 시작 / Dawn of multi-vantage simultaneity** — SoloHI + WISPR + LASCO + Metis + EUI + IS suite = 6 RS + 4 IS. 같은 CME를 4–5개 시점에서 동시 영상화하여 mass·속도·3D 형상·shock·driver 관계의 inversion 불확실성을 historically 줄임. 또한 PSP가 SoloHI FOV에 들어올 작지만 finite한 확률이 있어 직접 cross-calibration 가능성도 존재.

7. **APS 4-die mosaic의 운영 유연성 / APS 4-die mosaic operational flexibility** — die 단위·row 블록 단위로 다른 cadence·binning·exposure 적용 가능. Table 4: full frame 24 min cadence + inner subframe 0.6 min cadence + radial swath 6 min cadence를 동시 실행. 이는 고전적 CCD에서는 불가능했던 새로운 관측 자유도이고, 동일 perihelion pass 안에서 turbulence·CME·SEP shock의 서로 다른 시간 척도 현상을 함께 관측하게 해준다.

8. **Pre-flight 성능 시험은 실제 비행 운영의 reference / Pre-flight performance is the reference for in-flight ops** — 본 논문은 측정(M)/계산(C) 라벨로 모든 핵심 수치를 명시(Table 1). NRL SCOTCH 챔버에서의 stray-light 시험, BRDF + ray-tracing 비교, APS read noise/linearity 분석(Fig. 21) 등이 in-flight 발견할 수 있는 미세 변동을 진단하기 위한 baseline. 이는 LASCO 22년·SECCHI 12년 운영 경험에서 나온 "초기 calibration이 mission lifetime calibration의 anchor"라는 교훈의 적용이다.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Streamer blob acceleration profile / 스트리머 블롭 가속 프로파일

LASCO blob fit (Fig. 1):
$$V^2(r) = V_0^2\left[1 - e^{-(R-R_1)/R_0}\right]$$

| Symbol | Value | Meaning |
|---|---|---|
| $V_0$ | 298.3 km/s | Asymptotic velocity / 점근 속도 |
| $R_0$ | 8.1 R☉ | Acceleration scale length / 가속 스케일 |
| $R_1$ | 2.8 R☉ | Origin offset / 원점 오프셋 |

SoloHI extends this fit to 42 R☉ with much higher S/N than LASCO.

### 4.2 Thomson scattering scene brightness / Thomson 산란 장면 밝기

For a sight line at elongation $\varepsilon$ from Sun centre:
$$B(\varepsilon) = \int_{\rm LOS} G(\theta_{\rm sc}, \chi)\, n_e(s)\, ds$$

where $G$ is the Thomson-scattering geometry factor depending on the scattering angle $\theta_{\rm sc}$ and the angle $\chi$ between Sun–scatterer and Sun–observer; $n_e(s)$ is the electron density along the line of sight. The Thomson surface (sphere with spacecraft–Sun line as diameter) is the locus where $G$ peaks. For SoloHI, scene contributions are quantified in Fig. 4 by 5 %, 50 %, 95 % integrated brightness loci.

### 4.3 Multi-stage diffraction attenuation / 다단 회절 감쇠

For $N$ baffles each providing attenuation $\alpha_k \sim 10^{-3}$:
$$\frac{I_{\rm exit}}{I_{\rm entrance}} \approx \prod_{k=1}^{N}\alpha_k$$

For SoloHI, $N=5$ (HS, F1, F2, F3, F4 with I0 trap) gives a cumulative diffraction reduction of order $10^{-15}$, which combined with the lens transmission and bandpass yields the requirement $1\times10^{-13}\,B_\odot$ at outer FOV (Fig. 11).

### 4.4 Vignetting / 비네팅

Total vignetting:
$$T_{\rm total}(\theta) = T_{\rm natural}(\theta) \cdot V_{\rm baffle}(\theta), \quad T_{\rm natural} \propto \cos^3\theta$$

with $V_{\rm baffle}$ being the forward-baffle vignetting profile that varies from 0 at 5.4° to 1 at ~9.3° (Fig. 9 lower panel).

### 4.5 APS pixel response and CDS / APS 픽셀 응답과 CDS

CDS-corrected pixel signal:
$$S_{\rm CDS} = V_{\rm reset} - V_{\rm photo}$$

with $V_{\rm reset}$ stored on a capacitor at the start of integration and $V_{\rm photo}$ measured at end. This subtracts kTC noise and 1/f drift. The conversion to photo-electrons:
$$N_e = \frac{S_{\rm CDS}}{g_{\rm conv}}, \quad g_{\rm conv} = \text{conversion gain}$$

For SoloHI: high-gain mode 5.8 e⁻ read noise (median); low-gain mode 35.1 e⁻; full well > 86 400 e⁻ (low) / > 19 200 e⁻ (high); QE 32 % @ 480–750 nm; MIM capacitor sets gain.

### 4.6 Photon-noise S/N requirement / 광자 잡음 S/N 요구

Detection criteria (Rose 1948; Barrett 1990):
$$\frac{S}{N} = \frac{N_e}{\sqrt{N_e + N_{\rm BG} + N_{\rm read}^2}}, \quad \begin{cases} \geq 5, & \text{simple known target}\\ \geq 30, & \text{complex unknown target}\end{cases}$$

where $N_e$ is target photo-electrons, $N_{\rm BG}$ background photo-electrons (F-corona dominant), $N_{\rm read}$ read noise. SoloHI exposures are tuned per region (Table 4) to meet these.

### 4.7 Plate scale and angular resolution / 플레이트 스케일과 각분해능

Theoretical plate scale at field angle $\theta$:
$$p(\theta) = \frac{f}{1\;\rm rad} \approx p(0) \cdot (1 + \tan^2\theta \cdot \text{distortion correction})$$

For SoloHI: $p(0) = 0.971$ mm/deg, $p(20°) = 1.082$ mm/deg, $f = 55.9$ mm, F/# = 3.48. Per-pixel angular resolution:
$$\Delta \alpha = \frac{a_{\rm pix}}{f} = \frac{10\;\mu\rm m}{55.9\;\rm mm} \approx 36.7\;\rm arcsec$$

(2×2 binned: 73.5 arcsec.) AUeq at 0.28 AU: 10.3″ full / 20.6″ binned.

### 4.8 Worked numerical example: photon budget at 0.28 AU / 0.28 AU 광자 예산 계산

Take a faint coronal feature at outer FOV with brightness $B_{\rm feat} = 1\times10^{-12}\,B_\odot$.

Disk brightness $B_\odot \approx 2.0\times10^{10}$ photons s⁻¹ sr⁻¹ cm⁻² nm⁻¹ (visible band, scaled).

Aperture area $A = 16 \times 16$ mm² $= 2.56$ cm². Bandpass $\Delta\lambda \approx 350$ nm. Pixel solid angle $\Omega_{\rm px} = (10\;\mu m / 55.9\;mm)^2 \approx 3.20\times10^{-8}$ sr. QE = 0.32. Lens transmission ≈ 0.6.

Photo-electron rate per pixel for the feature:
$$\dot{N}_e \approx B_\odot \cdot 10^{-12} \cdot A \cdot \Delta\lambda \cdot \Omega_{\rm px} \cdot \text{QE} \cdot T_{\rm lens}$$
$$\approx 2.0\times10^{10} \cdot 10^{-12} \cdot 2.56 \cdot 350 \cdot 3.20\times10^{-8} \cdot 0.32 \cdot 0.6$$
$$\approx 1.1\times10^{-7}\;\text{e}^-\,\text{px}^{-1}\,\text{s}^{-1}\;\text{(unphysically low — illustrative only)}$$

The point of the calculation is: the F-corona (~10⁻⁹–10⁻¹⁰ B☉) is many orders of magnitude brighter than any K-corona transient, so the dominant photon-noise contributor in a single 30 s exposure at perihelion is F-corona photons, not the feature itself. Therefore **summing many short exposures** (image summing on-board CC) is mandatory: $N_{\rm sum}$ short exposures improve S/N as $\sqrt{N_{\rm sum}}$ on shot noise but average out cosmic-ray spikes.

This is exactly why SoloHI's Table 4 specifies, e.g., 30 min cadence at near-perihelion = many short exposures co-added, and 24 min full-frame cadence at perihelion with 2×2 binning.

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1948 ───── Rose: photometric S/N criteria for detection [foundational]
1974 ───── Helios launch — first in-situ + photometric inner heliosphere (ZL)
1981 ───── Solwind / SMM C/P: early space coronagraphs
1985 ───── Sheeley et al.: linking remote-sensing to in-situ
1995 ───── SOHO/LASCO C1/C2/C3 — single-vantage 30 R☉ coronagraphy [reference]
2006 ───── STEREO/SECCHI HI-1, HI-2 [Howard et al. 2008; Eyles et al. 2009]
                — first heliospheric imagers, dual-vantage in ecliptic
2010 ───── Janesick et al.: APS 5T pixel architecture
2013 ───── Korendyke et al.: APS for solar imaging (SPIE)
2016 ───── PSP/WISPR design [Vourlidas et al. 2016] — sister of SoloHI
2017 ───── Stenborg & Howard 2017b: single-image background subtraction
2018 ───── PSP launch (perihelion 9.86 R☉)
2018 ───── DeForest et al.: CME structures from PROBA-2 forward modelling
2020 ────► SoloHI launch on Solar Orbiter ◄── this paper [reference]
                — first close-Sun + out-of-ecliptic heliospheric imager
2020+ ──── Joint SoloHI + WISPR + LASCO 3D inversions
20??  ──── Future missions inheriting SoloHI's CMOS APS + multi-baffle design
```

**한국어**: 이 논문은 LASCO(1995)와 SECCHI(2006)가 만든 white-light heliospheric imaging 패러다임의 세 번째 세대를 정의하는 instrument paper다. 첫 세대(LASCO)는 단일 시점·1 AU·황도면, 두 번째 세대(SECCHI)는 이중 시점·1 AU·황도면, SoloHI는 변동 시점·0.28 AU·30°+ 황도경사로 세 자유도를 모두 깬다. 동시에 검출기 측면에서도 CCD에서 CMOS APS로의 전환을 본격화한 첫 fully-qualified 우주 광학계 사례다.

**English**: This paper defines the third generation of the white-light heliospheric-imaging paradigm established by LASCO (1995) and SECCHI (2006). Gen-1 was single-vantage at 1 AU in the ecliptic; Gen-2 was dual-vantage at 1 AU in the ecliptic; SoloHI breaks all three degrees of freedom — variable vantage, 0.28 AU perihelion, and >30° ecliptic inclination. It is also the first fully qualified space-borne optical instrument to formally transition the heliospheric-imager detector lineage from CCD to custom CMOS APS.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Howard et al. 2008 (SECCHI) | Direct heritage telescope suite — HI-1 and HI-2 are SoloHI's parents | Defines the HI imaging concept SoloHI extends; design philosophy reused |
| Eyles et al. 2009 (STEREO HI) | Predecessor instrument paper — stray-light baffle architecture inherited | Forward-baffle technique, image-summing approach, S/N strategy |
| Vourlidas et al. 2016 (PSP/WISPR) | Sister wide-field imager on PSP — same era, complementary vantage | Joint SoloHI + WISPR observations are central to mission goals |
| Müller et al. 2013, 2020 (Solar Orbiter) | Mission-level paper defining four science questions | SoloHI §2 maps directly to these mission objectives |
| Vourlidas & Howard 2006 (Thomson surface) | Theoretical foundation for scene coverage at varying Sun-spacecraft distance | Justifies why 0.28 AU SoloHI is a "local imager" |
| Brueckner et al. 1995 (SOHO/LASCO) | First-generation reference coronagraph — 22 yrs heritage | Lens material reused; calibration approach inherited |
| Janesick et al. 2010, 2013 | APS 5T pixel architecture papers | Defines the detector technology SoloHI fully qualifies |
| Korendyke et al. 2013 | NRL APS development for solar imaging | Direct lineage to SoloHI 4-die pinwheel design |
| Stenborg & Howard 2017a,b | F-corona variability + single-image background technique | Required for SoloHI L3 data products |
| Thernisien et al. 2006, 2018 | Baffle design and stray-light test methodology | SCOTCH chamber + ray-tracing approach used for SoloHI |

---

## 6.1 Additional Cross-Mission Comparison / 추가 다중 미션 비교

| Property / 특성 | LASCO C3 | SECCHI HI-1 | SECCHI HI-2 | SoloHI | WISPR-I | WISPR-O |
|---|---|---|---|---|---|---|
| Mission / 임무 | SOHO | STEREO | STEREO | Solar Orbiter | PSP | PSP |
| Launch year / 발사년 | 1995 | 2006 | 2006 | 2020 | 2018 | 2018 |
| Heliocentric distance / 일심 거리 | 1 AU (L1) | 1 AU | 1 AU | 0.28–0.88 AU | 0.046–0.7 AU | 0.046–0.7 AU |
| Inclination / 경사 | 0° | 0° | 0° | up to 33° | <4° | <4° |
| Inner FOV / 내부 시야 | 4 R☉ | ~4° | ~19° | 5.4° (5.25 R☉ @ 0.28 AU) | 13.5° | 53° |
| Outer FOV / 외부 시야 | 30 R☉ | ~24° | ~89° | 44.9° | 53° | 108° |
| Detector / 검출기 | 1024×1024 CCD | 2048×2048 CCD | 2048×2048 CCD | 4-die APS 3968×3968 | 2048×1920 APS | 1920×2048 APS |
| Spectral band / 스펙트럼 대역 | 400–850 nm | 630–730 nm | 400–1000 nm | 500–850 nm | 480–770 nm | 480–770 nm |
| Notes / 비고 | Coronagraph w/ occulter | First HI | First HI | First out-of-ecliptic | Inner WISPR | Outer WISPR |

**한국어**: 표는 본 논문이 자리하는 instrument 계보를 한눈에 보여준다. SoloHI는 시야 측면에서 HI-1 + HI-2를 합친 것에 약간 못 미치지만(45° vs 89°), 0.28 AU 근일점에서 본 5.4° 내부 시야는 1 AU에서 본 HI-1의 4°보다 절대 거리상 훨씬 더 Sun에 가깝다. 또한 황도경사 33°까지 sweep하는 능력은 어떤 기존 imager도 갖지 못한 자유도다. WISPR과는 시야 일부가 겹치고 동시 관측 시점이 다르므로 보완적이다.

**English**: The table situates this paper in its instrument lineage at a glance. In raw FOV, SoloHI's 45° outer extent is less than HI-1+HI-2's combined 89°, but its 5.4° inner cutoff at 0.28 AU is much closer to the Sun in absolute terms than HI-1's 4° at 1 AU. The ability to sweep to 33° ecliptic inclination is a degree of freedom no prior imager possessed. SoloHI and WISPR overlap in FOV but observe from distinct, simultaneous vantage points and are therefore complementary.

---

## 6.2 Pre-Flight Verification Highlights / 발사 전 검증 하이라이트

**한국어**: 본 논문은 측정(M)·계산(C) 라벨로 모든 핵심 파라미터를 명시한다. 주요 사전 검증 결과:

- **APS read noise (Fig. 21 left)**: histogram 측정으로 high-gain 5.7 e⁻ (median), low-gain 35.0 e⁻ — 요구치 충족.
- **APS linearity (Fig. 21 right)**: 16개 영역(64×64 px) 측정에서 ±1% 이내(대부분 범위), curved 부분만 ±5% — pre-launch cal 보정으로 처리 가능.
- **MTF**: 단일 die @ −65°C에서 측정한 test pattern (Fig. 20)은 ideal 10 µm pitch detector의 80% 이상 — 요구치 충족.
- **Stray-light test (Fig. 12)**: NRL SCOTCH 챔버에서 측정한 single-bounce pattern과 ray-tracing 모델의 일치 → forward-baffle + interior-baffle 시스템이 의도한 대로 동작함을 검증.
- **Diffraction profile (Fig. 11)**: HS edge → F1 → F2 → F3 → F4 → I0 단계별 attenuation을 모델링; aperture A1 평면에서 normalised flux ~10⁻⁵ 이하로 떨어짐.
- **Lens transmission (Fig. 9 top)**: 500–850 nm bandpass 측정 (Jenoptik) — 평균 transmission ~0.7 + 두 흡수 dip (coating).

**English**: The paper labels every key parameter (M)easured or (C)alculated. Highlights of pre-flight verification:

- **APS read noise (Fig. 21 left)**: histograms give 5.7 e⁻ high-gain (median), 35.0 e⁻ low-gain — meets requirement.
- **APS linearity (Fig. 21 right)**: 16 regions (64×64 px), ±1 % over most of the range, curved tails only ±5 % — handled by pre-launch calibration.
- **MTF**: test pattern on a single die at −65 °C (Fig. 20) gives MTF > 80 % of an ideal 10 µm-pitch detector.
- **Stray-light verification (Fig. 12)**: SCOTCH chamber single-bounce measurement vs ray-tracing model show good agreement — confirms the forward + interior baffle systems work as designed.
- **Diffraction profile (Fig. 11)**: HS-edge → F1 → F2 → F3 → F4 → I0 stage-by-stage attenuation modelled; normalised flux at the A1 aperture plane drops below 10⁻⁵.
- **Lens transmission (Fig. 9 top)**: bandpass 500–850 nm measured by Jenoptik — average transmission ~0.7 with two coating absorption dips.

---

## 6.3 Failure Modes & Risk Mitigations / 고장 모드와 리스크 완화

**한국어**: SoloHI는 protoflight 모델 철학(no QM of full instrument, FPA QM only)을 채택하므로 lifetime 리스크 관리가 중요하다. 주요 모드:

| Risk / 리스크 | Mitigation / 완화 |
|---|---|
| TID damage to APS over 7 yrs | 60 krad margin behind 2.54 mm Al; periodic annealing between perihelions |
| Charge transfer efficiency loss (CTE) | APS pixels are read directly without shifting → CTE risk inherently lower than CCD |
| SEU/transient on FPGA | LET threshold ≤ 25 MeV cm²/mg parts excluded; on-board cosmic-ray scrub |
| Lens degradation | Same radiation-tolerant glass as LASCO (22 yrs heritage, ~0.5 %/yr) |
| Off-pointing direct-Sun exposure | Thermal study shows APS stays < 100 °C during 50 s direct-Sun off-point |
| Door fail-to-open | One-shot ERM with strain-sensed bolt + Belleville washer; redundant kick-off spring |
| Contamination from propulsion droplets | Dedicated baffles in front of jets |
| Stray-light degradation | Multi-stage baffle redundancy (5 forward, 9 interior, 2 trap, 1 peripheral) |

**English**: SoloHI uses a protoflight model approach (no full-instrument QM, only an FPA QM), so lifetime risk management is critical. Major modes:

The table above lists the main risks and mitigations. The combination of CMOS architecture (no charge transfer + on-pixel CDS), heritage radiation-tolerant glass, multi-stage baffle redundancy, and conservative thermal design gives a 7-year mission margin even under conservative assumptions. The annealing-between-perihelions strategy is a key innovation for managing radiation damage in close-Sun missions and will likely be inherited by future heliospheric imagers.

---

## 6.4 Lessons for Future Heliospheric Imagers / 미래 헬리오스피어 영상기를 위한 교훈

**한국어**: 본 논문은 향후 close-Sun, multi-vantage 영상기 설계에 다음과 같은 직접적 교훈을 남긴다:

1. **CMOS APS는 우주 광학계에서 검증되었다** — radiation tolerance, mass, EMI 측면에서 CCD 대비 명확한 이득. 4-die mosaic은 단점(2-side buttable)을 pinwheel로 우회하는 검증된 패턴.
2. **Heat-shield edge는 새로운 첫 baffle** — close-Sun mission에서 spacecraft heat shield 자체가 첫 diffraction edge가 되므로 baffle 시스템의 시작점이 망원경 외부로 옮겨진다.
3. **Per-row programmable cadence** — APS의 row-addressable 특성을 활용하면 같은 die에서 다른 영역에 다른 cadence·binning·exposure를 적용할 수 있어 동일 perihelion pass 안에서 multi-scale science를 가능하게 한다.
4. **F-corona는 더 이상 axisymmetric이 아니다** — Stenborg & Howard 2017b 같은 single-image background 모델이 close-Sun, out-of-ecliptic imager에 필수적.
5. **Multi-vantage simultaneity는 inversion 패러다임을 바꾼다** — 4–5개 시점 동시 관측은 단일 시점에서의 forward-modelling에 의존하던 기존 방식 대신 직접 inversion/tomography로의 전환을 가능하게 한다.

**English**: The paper leaves direct lessons for future close-Sun, multi-vantage imagers:

1. **CMOS APS is now flight-qualified** — clear advantages over CCDs in radiation tolerance, mass, and EMI. The 4-die mosaic with pinwheel arrangement is a proven pattern around the 2-side buttable limitation.
2. **The heat-shield edge becomes the new first baffle** — for close-Sun missions, the spacecraft heat shield itself is the first diffraction edge, shifting the start of the baffle system outside the telescope.
3. **Per-row programmable cadence** — exploiting the row-addressable nature of the APS allows different regions of the same die to use different cadences, binnings, and exposures, enabling multi-scale science within a single perihelion pass.
4. **F-corona is no longer axisymmetric** — single-image background models (Stenborg & Howard 2017b) are essential for close-Sun, out-of-ecliptic imagers.
5. **Multi-vantage simultaneity changes the inversion paradigm** — 4–5 simultaneous viewpoints replace single-vantage forward-modelling with direct inversion/tomography.

---

## 7. References / 참고문헌

- Howard, R. A. et al., "The Solar Orbiter Heliospheric Imager (SoloHI)", A&A 642, A13 (2020). https://doi.org/10.1051/0004-6361/201935202
- Müller, D. et al., "The Solar Orbiter mission. Science overview", A&A 642, A1 (2020).
- Howard, R. A. et al., "Sun Earth Connection Coronal and Heliospheric Investigation (SECCHI)", Space Sci. Rev. 136, 67 (2008).
- Eyles, C. J. et al., "The Heliospheric Imagers Onboard the STEREO Mission", Sol. Phys. 254, 387 (2009).
- Vourlidas, A. et al., "The Wide-Field Imager for Solar Probe Plus (WISPR)", Space Sci. Rev. 204, 83 (2016).
- Brueckner, G. E. et al., "The Large Angle Spectroscopic Coronagraph (LASCO)", Sol. Phys. 162, 357 (1995).
- Vourlidas, A. & Howard, R. A., "The Proper Treatment of Coronal Mass Ejection Brightness", ApJ 642, 1216 (2006).
- Stenborg, G. & Howard, R. A., "A Heuristic Approach to Remove the Background Intensity on White-Light Solar Images", ApJ 839, 68 (2017b).
- Janesick, J. et al., "Fundamental performance differences between CMOS and CCD imagers", Proc. SPIE 7742, 77420B (2010).
- Korendyke, C. M. et al., "Development and test of an active pixel sensor detector for heliospheric imager on Solar Orbiter and Solar Probe Plus", Proc. SPIE 8862, 88620J (2013).
- Müller, D. et al., "Solar Orbiter — Exploring the Sun-heliosphere connection", Sol. Phys. 285, 25 (2013).
- Marsch, E., "The Outer Heliosphere: Beyond the Planets" (2000).
- Rose, A., J. Opt. Soc. Am. 38, 196 (1948).
- Stenborg, G. et al., Sol. Phys. 868, 74 (2018a).
- Stauffer, J. R. et al., ApJ 864, 29 (2018).
- Rice, R. F., NASA STI/Recon Technical Report No. 78 (1978).
- Wang, D. et al., Proc. SPIE 3442, 150 (1998).
