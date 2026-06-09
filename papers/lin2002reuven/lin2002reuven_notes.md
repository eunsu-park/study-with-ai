---
title: "The Reuven Ramaty High-Energy Solar Spectroscopic Imager (RHESSI)"
authors: [R. P. Lin, B. R. Dennis, G. J. Hurford, D. M. Smith, A. Zehnder, P. R. Harvey, et al.]
year: 2002
journal: "Solar Physics 210, 3-32"
doi: "10.1023/A:1022428818870"
topic: Solar_Observation
tags: [RHESSI, hard-X-ray, gamma-ray, imaging-spectroscopy, RMC, germanium-detector, solar-flare, particle-acceleration]
status: completed
date_started: 2026-04-25
date_completed: 2026-04-25
---

# 39. The Reuven Ramaty High-Energy Solar Spectroscopic Imager (RHESSI) / 류번 라마티 고에너지 태양 분광 이미저

---

## 1. Core Contribution / 핵심 기여

RHESSI (originally HESSI) is the sixth NASA Small Explorer (SMEX) mission and the first managed in PI mode. Launched on 5 February 2002, the spacecraft carries a single rotation-modulated instrument that delivers, for the first time, simultaneous high-resolution **imaging** and **spectroscopy** of solar hard X-rays and gamma-rays from 3 keV to 17 MeV. The imager is a stack of nine Rotating Modulation Collimators (RMCs); the spectrometer is nine cryogenically cooled segmented germanium detectors (GeDs), one per RMC. The combination achieves 2.3 arcsec angular resolution (full-Sun, ~1° FOV), ~1 keV FWHM energy resolution at 3 keV, ~5 keV FWHM at 5 MeV, and a dynamic range of more than 10^7 in flare intensity using automatic shutters. Every photon is time-tagged to 1 µs and stored on board, allowing scientists to trade off time, energy and spatial resolution on the ground.

RHESSI는 NASA Small Explorer (SMEX) 시리즈의 6번째이자 PI(Principal Investigator) 모드로 운영된 최초의 임무로, 2002년 2월 5일 발사되었다. 이 위성은 단일 회전 변조 기기를 통해 3 keV-17 MeV의 hard X-ray와 gamma-ray를 동시에 **고해상도 영상**과 **분광**으로 관측한 최초의 망원경이다. Imager는 9개의 회전 변조 콜리메이터(RMC)로 구성되며, spectrometer는 RMC 한 개당 한 개씩 배치된 9개의 저온 냉각 segmented germanium detector(GeDs)이다. 이 조합으로 2.3 arcsec의 공간 분해능(~1° FOV), 3 keV에서 ~1 keV FWHM, 5 MeV에서 ~5 keV FWHM의 에너지 분해능, 그리고 자동 셔터를 통한 10^7 이상의 동적 범위를 달성한다. 모든 광자는 1 µs 시간 태깅되어 탑재되며, 지상에서 시간·에너지·공간 분해능을 사용자가 자유롭게 trade-off할 수 있다.

RHESSI's primary scientific goal is to clarify *particle acceleration and explosive energy release* in solar flares — the most powerful particle accelerators in the solar system, releasing 10^32-10^33 erg in 10^2-10^3 s and accelerating ions to tens of GeV and electrons to hundreds of MeV. By probing thermal-nonthermal transitions, microflare populations, footpoint geometry, and (for the first time) the spatial location of gamma-ray lines, RHESSI provides quantitative tests of acceleration mechanisms in magnetised plasmas — physics that mirrors processes throughout cosmic plasmas from planetary magnetospheres to active galactic nuclei.

RHESSI의 주된 과학 목표는 태양 플레어에서의 *입자 가속과 폭발적 에너지 해방*의 정량적 이해이다. 태양 플레어는 태양계에서 가장 강력한 입자 가속기로, 10^2-10^3 초 동안 10^32-10^33 erg를 방출하며, 이온은 수십 GeV, 전자는 수백 MeV까지 가속된다. RHESSI는 thermal-nonthermal 전이, microflare 통계, footpoint 기하 구조, 그리고 (최초로) gamma-ray 라인의 공간 위치를 측정함으로써 자기화된 플라즈마에서의 가속 메커니즘을 정량적으로 검증한다. 이는 행성 자기권부터 활동성 은하핵까지 우주 플라즈마 전반에서 일어나는 과정과 동일한 물리이다.

---

## 2. Reading Notes / 읽기 노트

### Part I: Abstract & Introduction (§1) / 초록과 서론

**Mission identity (p. 4).** RHESSI is the sixth SMEX, the first PI-mode mission. The instrument is a single payload: 9 RMC imager + 9 GeD spectrometer. Spatial: ~2.3 arcsec, FOV ≥ 1° (full-Sun). Spectral: ~1-10 keV FWHM over 3 keV-17 MeV. Dynamic range: > 10^7 via auto-shutters. Spin-stabilised at ~15 rpm, pointing to ~0.2°. 4-Gbyte solid-state memory holds all photons. Data and software are released publicly and immediately.

**임무 정체성 (4쪽).** RHESSI는 6번째 SMEX이며 PI 모드 운영의 첫 사례이다. 단일 탑재체로 9 RMC imager와 9 GeD spectrometer를 사용한다. 공간 ~2.3 arcsec, FOV ≥ 1° (전 태양). 분광 ~1-10 keV FWHM, 3 keV-17 MeV. 동적 범위 > 10^7. ~15 rpm으로 회전 안정화되며 ~0.2°의 정밀도로 태양을 지향한다. 4 GB 솔리드 스테이트 메모리에 모든 광자 사건이 저장된다. 데이터와 소프트웨어는 즉시 공개된다.

**Why high-energy emissions? (p. 5-6).** Bremsstrahlung from accelerated electrons (10-100 keV) and gamma-ray lines from accelerated ions (10-100 MeV/nucleon) provide the most *direct* signatures of acceleration. Thermal flare plasmas at ~10^7 K (and "superhot" 3×10^7 K) dominate up to ~10-30 keV, then bremsstrahlung continuum dominates up to a few tens of MeV, where pion-decay photons take over (Fig. 1). RHESSI spans nearly four decades in photon energy.

**고에너지 방출이 중요한 이유 (5-6쪽).** 가속 전자(10-100 keV)에 의한 bremsstrahlung과 가속 이온(10-100 MeV/nucleon)에 의한 gamma-ray 라인은 가속 과정의 가장 직접적인 신호이다. ~10^7 K의 열적 플라즈마(및 3×10^7 K의 "superhot")는 ~10-30 keV까지를, 그 이후 수십 MeV까지는 bremsstrahlung continuum이, 더 위에서는 π-decay 광자가 지배한다(그림 1). RHESSI는 광자 에너지 4-decade를 거의 모두 덮는다.

**PI mode and mission history (p. 5-6).** Selected October 1997, target launch July 2000. JPL vibration accident (March 2000): shake-table malfunction subjected the spacecraft to >25 G instead of 2 G; major repair followed. After Mars failures, NASA imposed Red-Team reviews. Pegasus-XL solid-rocket problem and X-43 prototype launch failure further delayed launch. Finally launched 5 Feb 2002 from L-1011 over the Atlantic into a 38°-inclination, 600-km circular orbit. Renamed *RHESSI* in honour of Reuven Ramaty, who passed away in March 2001 — the first NASA scientist to have a mission named after him.

**PI 모드와 임무 역사 (5-6쪽).** 1997년 10월 선정, 2000년 7월 발사 예정. 2000년 3월 JPL 진동시험에서 셰이크 테이블 오작동으로 25 G가 가해져 우주선과 기기가 손상. Mars 임무 실패 이후 Red-Team 리뷰가 추가됨. Pegasus-XL 1단 문제와 X-43 시제기 발사 실패로 또 지연. 결국 2002년 2월 5일 L-1011 항공기로부터 대서양 상공에서 발사되어 경사 38°, 600 km 원궤도 진입. Reuven Ramaty(2001년 3월 별세)를 기려 *RHESSI*로 개명; NASA 과학자 이름이 임무에 붙은 첫 사례.

### Part II: Scientific Objectives (§2) / 과학 목표

**§2.1 Acceleration of electrons.** Electron Coulomb energy loss dominates over bremsstrahlung by a factor ~10^5, so observed hard X-ray fluxes imply that the energy in >20 keV electrons is comparable to the entire flare radiative + mechanical output (Lin & Hudson 1976). High-resolution **imaging spectroscopy** lets us invert the spatially resolved photon spectrum into the source electron distribution $N(E,\vec r,t)$ (Johns & Lin 1992). Combined with a spatially dependent continuity equation including loss processes, the accelerated-electron source $F(E,\vec r,t)$ can be inferred. RHESSI design drivers: spatial scale matching Coulomb-loss length in lower corona/upper chromosphere ($n \lesssim 10^{12}\,$cm$^{-3}$), ~1 keV FWHM to resolve thermal-nonthermal transition, energy range from soft X-rays to relativistic, very high sensitivity for microflares, very wide dynamic range for X-class flares.

**§2.1 전자 가속.** 전자의 Coulomb 손실이 bremsstrahlung 손실의 ~10^5 배이기 때문에, 관측된 hard X-ray flux는 >20 keV 전자에 들어 있는 에너지가 플레어 전체 복사·역학 출력에 필적함을 시사한다(Lin & Hudson 1976). 공간 분해 imaging spectroscopy를 통해 광자 스펙트럼을 source 전자 분포 $N(E,\vec r,t)$로 inversion할 수 있다(Johns & Lin 1992). 손실 과정을 포함한 공간 의존 연속 방정식과 결합하면 가속 전자 source $F(E,\vec r,t)$를 추론할 수 있다. RHESSI 설계 요구사항: 하부 코로나·상부 채층 ($n\lesssim10^{12}$ cm$^{-3}$)에서의 Coulomb 손실 길이에 해당하는 공간 분해능, thermal-nonthermal 전이를 분해하는 ~1 keV FWHM, 연성 X선부터 상대론적 영역까지의 에너지 범위, microflare 감지를 위한 고감도, X-class 플레어에서 포화 없는 광대역 동적 범위.

**§2.2 Acceleration of ions.** Nuclear interactions of accelerated ions with the ambient atmosphere produce a rich gamma-ray line spectrum (Ramaty & Murphy 1987). Narrow lines (widths ~ keV-100 keV) come from energetic protons/α exciting C and heavier nuclei; broad lines (widths ~hundreds of keV-MeV) come from energetic heavy ions hitting H and He. Neutron capture on H produces the **2.223 MeV** delayed line; positron annihilation produces the **0.511 MeV** line. Most line emission is from 10-100 MeV/nuc ions whose total energy may rival that of the electrons (Ramaty et al. 1995; Emslie et al. 1997). The shape of the 0.511 MeV line probes density and temperature of the ambient medium because positrons slow before annihilating. RHESSI is the first to image gamma-ray lines, comparing ion locations to electron locations.

**§2.2 이온 가속.** 가속 이온과 ambient 대기의 핵충돌은 다양한 gamma-ray 라인 스펙트럼을 만든다(Ramaty & Murphy 1987). 좁은 라인(폭 ~keV-100 keV)은 양성자/알파가 C와 중원소를 들뜨게 해 발생; 넓은 라인(수백 keV-MeV)은 가속 중원소가 H, He에 충돌해 발생. 중성자 포획은 **2.223 MeV** 지연 라인을, 양전자 소멸은 **0.511 MeV** 라인을 만든다. 라인의 대부분은 10-100 MeV/nuc 이온에서 오며, 총에너지는 전자에 필적할 수 있다(Ramaty et al. 1995; Emslie et al. 1997). 0.511 MeV 라인의 형상은 양전자가 소멸 전 감속하기 때문에 ambient 매질의 밀도·온도를 진단한다. RHESSI는 gamma-ray 라인의 영상을 처음 얻어 이온의 위치를 전자와 비교한다.

**§2.3 Non-solar science.** RHESSI lacks heavy shielding, so it doubles as an all-sky hard X-ray/gamma-ray monitor with ~150 cm² collecting area. Spacecraft rotation produces detector occultations that allow source localisation. Targets: black-hole/neutron-star transients (e.g. A0535+26), the Crab (once per year), Galactic 511 keV and ^26Al lines, gamma-ray bursts, and terrestrial gamma-ray flashes from lightning (Fishman et al. 1994).

**§2.3 비태양 과학.** RHESSI는 무거운 차폐가 없어 ~150 cm² 유효면적의 전 천 hard X-ray/gamma-ray 모니터로도 작동한다. 회전에 의한 검출기 occultation으로 source 위치를 결정할 수 있다. 대상: 블랙홀·중성자별 천이 (예: A0535+26), Crab(연 1회 1.6° 이내 통과), 은하 511 keV 및 ^26Al 라인, gamma-ray burst, 번개에 의한 지구 gamma-ray flash 등.

### Part III: Instrument (§3) / 기기

**Architecture.** Imaging System (9 RMCs) + Spectrometer (9 GeDs, one per RMC) + Instrument Data Processing Unit (IDPU). Pointing from Solar Aspect System (SAS) and redundant Roll Angle Systems (RAS). GeDs cooled to ≤ 75 K by Stirling-cycle cryocooler. The spacecraft rotates and the RMCs convert source angular structure to temporal modulation of GeD count rates. Energy and arrival time of *every* photon are stored in 4-Gbyte memory and telemetered within 48 hours. The 1° FOV is much wider than the 0.5° solar diameter so all flares are detected and pointing is automatic.

**구조.** Imaging System (9 RMCs) + Spectrometer (9 GeDs) + IDPU 전자보드. 지향 정보는 Solar Aspect System(SAS)과 이중 Roll Angle System(RAS)이 제공. GeD는 Stirling 냉각기로 ≤ 75 K 냉각. 우주선 회전에 의해 RMC가 source 각도 구조를 GeD 계수율의 시간 변조로 변환한다. 모든 광자의 에너지와 도착 시각은 4 GB 메모리에 저장되어 48시간 내 텔레메트리 된다. 1° FOV는 태양면(0.5°)보다 넓어 모든 플레어가 자동 검출된다.

**§3.1 Imaging System.** Each RMC: pair of widely separated planar grids (X-ray opaque slats with transparent slits). For matched pitches $p$ and grid separation $L$, transmission modulates 0-50 % over a source-angle change of $p/L$ orthogonal to slits — angular resolution $p/(2L)$. RHESSI uses $L = 1.55$ m and grid pitches in steps of $\sqrt 3$ from $p = 34\,\mu$m to 2.75 mm, producing logarithmically spaced angular resolutions from 2.3 arcsec to ≥ 3 arcmin. Diffuse sources > 3 arcmin are not imaged but full spectroscopy is preserved. In one half rotation (2 s) the 9 RMCs sample ~1100 Fourier components of the source (vs. 32 for Yohkoh HXT). Critical alignment requirement: relative *twist* of $p/D$ ($D$ = 9 cm grid diameter) reduces modulation to ~0; finest grids need 1-arcmin twist control. The Twist Monitoring System (TMS) — photodiodes/pinholes plus CCD — verifies alignment to launch.

**§3.1 영상 시스템.** RMC = 두 grid (X-ray 불투과 슬랫 + 투명 슬릿)의 한 쌍. 같은 pitch $p$와 grid 간격 $L$에서 슬릿 직교 방향으로 $p/L$의 source angle 변화에 대해 투과율이 0-50 %로 변조 → 각 분해능 $p/(2L)$. RHESSI는 $L = 1.55$ m, pitch는 $\sqrt 3$ 비율로 34 µm에서 2.75 mm까지 증가하여 2.3 arcsec ~ ≥ 3 arcmin의 로그 분해능을 제공. 3 arcmin 이상의 확산 source는 영상화되지 않지만 분광은 가능. 반회전(2 s) 동안 9 RMC가 ~1100개의 Fourier 성분을 측정 (Yohkoh HXT는 32개). Twist 허용치는 $p/D$ ($D$=9 cm); 가장 미세한 grid는 1-arcmin 정렬 필요. Twist Monitoring System(TMS)이 발사 직전까지 정렬 검증.

**§3.1.1 Grids.** Finest grid: 20 µm slit, 14 µm slat at 34 µm pitch — 50:1 aspect ratio for ~1° absorption. Tecomet (USA) made the four finest pairs by foil stacking (etched + epoxied tungsten foils); pair No. 1 is molybdenum (max modulation ~100 keV instead of ~200 keV) because thin tungsten was unavailable. Van Beek (NL) made the five coarsest pairs with packed tungsten blades. Thickest grids (pairs 6 and 9, 1.85 and 3 cm) modulate up to 17 MeV. Each grid optically and X-ray characterised at GSFC; aligned at PSI (Switzerland). End-to-end check used a ^109Cd 22 keV source behind a spare grid to verify modulation.

**§3.1.1 그리드.** 가장 미세한 grid: 34 µm pitch에서 20 µm 슬릿, 14 µm 슬랫, 50:1 aspect ratio로 ~1°의 흡수율 확보. 가장 미세한 4쌍은 Tecomet (미국)이 텅스텐 박판 stacking으로 제작; 1쌍만 몰리브덴(얇은 텅스텐 판이 부족하여 최대 변조 100 keV로 제한). 가장 거친 5쌍은 Van Beek (네덜란드)이 텅스텐 블레이드 패킹으로 제작. 6, 9번 grid는 1.85 cm, 3 cm 두께로 17 MeV까지 변조. GSFC에서 광학·X선 특성화 후 PSI(스위스)에서 정렬. ^109Cd 22 keV source로 종단 검사.

**§3.1.2 Aspect Systems.** SAS provides pitch-yaw to ~1.5 arcsec (3σ) on 10-ms timescales using three lens-filter assemblies imaging six chords of the solar limb on 2048×13-µm linear diode arrays. CCD RAS uses an *f*/1.0, 50-mm lens looking at +2 mag stars on a CCD for roll determination to 2.7 arcmin (3σ). PMT RAS is a redundant photomultiplier-based scanner.

**§3.1.2 자세 시스템.** SAS는 lens-filter 3개로 태양 limb의 6 chord를 2048×13 µm 선형 다이오드 어레이에 결상해 10 ms 시간 척도에서 ~1.5 arcsec(3σ) pitch/yaw 정밀도 제공. CCD RAS는 *f*/1.0, 50 mm 렌즈로 +2등성 별을 CCD로 관측해 2.7 arcmin(3σ) roll. PMT RAS는 photomultiplier 기반 redundant 스캐너.

**§3.2 Spectrometer.** Nine 7.1 cm × 8.5 cm n-type hyperpure coaxial GeDs from ORTEC, each segmented (electrically) into a ~1.5-cm planar **front segment** in front of a ~7 cm coaxial **rear segment**. Top/curved outer surfaces have a 0.3-µm boron implant for transparency down to 3 keV. Front segment: 3-keV threshold; stops photons up to ~250 keV (photoelectric absorption dominates). Rear segment: stops 250 keV-17 MeV. Compton-scatter or rear-incidence backgrounds are vetoed by anticoincidence; passive graded-Z (Pb, Cu, Sn) ring around the front segment shields side incidence. **F/R coincidence** (front + rear simultaneous deposition) provides additional photopeak efficiency above ~250 keV.

**§3.2 분광계.** 9개의 7.1 cm × 8.5 cm n-type 동축 hyperpure Ge 검출기 (ORTEC). 각 검출기는 두 개의 segment로 분할되며, 전방 ~1.5 cm planar **front segment** + 후방 ~7 cm 동축 **rear segment**. 상부/외면에 0.3 µm boron implant로 3 keV까지 투과. Front segment: 3 keV 임계값, ~250 keV까지 흡수 (광전 효과 지배). Rear segment: 250 keV-17 MeV 흡수. Compton 산란 background는 anticoincidence로 제거; passive graded-Z (Pb, Cu, Sn) 링이 측면 입사를 차폐. ~250 keV 이상에서 **F/R coincidence** 모드가 유효 면적에 추가 기여.

**Attenuators.** Two aluminium disks ("thin" and "thick") move in front of GeDs by Shape Memory Alloy (SMA) actuators when count rate exceeds set thresholds, attenuating low-energy photons; held in for ~5 min then removed. This is what gives the 10^7 dynamic range from microflares to X-class.

**감쇠기 (셔터).** "thin"과 "thick" 두 알루미늄 디스크가 SMA 액추에이터로 GeDs 앞에 자동 삽입; 계수율 임계값을 넘으면 저에너지 광자를 흡수해 saturation 방지. ~5 분 유지 후 제거. 마이크로플레어부터 X-class까지 10^7 동적 범위의 핵심.

**Cooling.** Sunpower M77B single-stage Stirling cryocooler delivers up to 4 W at 77 K with 100 W input. Cryocooler isolated by gas-bearing/flexure system to minimise microphonics; moving-magnet motor avoids flex leads. 76-cm anti-Sun radiator. Equilibrium radiator −15 to −30 °C.

**냉각.** Sunpower M77B 단단 Stirling 냉각기는 100 W 입력에서 77 K에 4 W 냉각. 가스 베어링/flexure로 microphonic 노이즈 격리; moving-magnet 모터로 lead wire 제거. 76 cm anti-Sun 방열판; 평형 온도 −15 ~ −30 °C.

**Radiation damage / annealing.** SAA passages (~5/day) trap defects in the Ge that degrade resolution. The spectrometer can anneal in-flight by heating to ~100 °C, expected unnecessary within nominal 2-year mission.

**방사선 손상·annealing.** SAA 통과 (하루 ~5회)에서 고에너지 양성자가 Ge 내 trap을 형성해 분해능을 저하. 스펙트로미터는 100 °C 가열로 in-flight annealing이 가능. 명목 2년 임무 동안은 불필요할 것으로 예상.

**§3.3 Instrument Electronics.** GeDs biased at 4-5 kV. Charge-Sensitive Amplifier (CSA) with advanced 4-terminal FET. IDPU (Curtis et al. 2002): 9 Detector Interface Boards (DIBs) per GeD; quasi-trapezoidal shaping for ballistic-deficit; dual fast/slow chains for pile-up rejection; ultrahigh rate counting in broad bands with 0.5-ms live-time sampling to preserve imaging. Front segments: ~3 keV - 2.7 MeV in 8192 channels (0.33 keV/ch). Rear segments: ~20 keV - 17 MeV (2.7 keV/ch above 2.7 MeV via low-gain amp). Each photon → 24-bit event word (14 bits energy + 1 µs time + detector ID + live time).

**§3.3 기기 전자.** GeD는 4-5 kV 바이어스. CSA는 4-terminal FET 사용. IDPU는 9개 DIB, quasi-trapezoidal shaping (ballistic-deficit 보정), fast/slow 이중 chain (pile-up 제거), 0.5 ms 단위의 ultrahigh-rate counting. Front segment: 3 keV-2.7 MeV, 8192 채널 (0.33 keV/ch). Rear segment: 20 keV-17 MeV (2.7 MeV 이상은 저게인 증폭기, 2.7 keV/ch). 광자당 24-bit event word (에너지 14 bit + 1 µs 시간 + 검출기 ID + live time).

### Part IV: Spacecraft (§4) / 우주선

**Bus.** Built by Spectrum Astro. Octagonal Al honeycomb deck with imager forward and spectrometer aft. Four solar array wings (4 × 133.5 W = 534 W EOL); fine-tuning by two motorised Inertia Adjustment Devices (IADs).

**버스.** Spectrum Astro 제작. 팔각형 Al 허니콤 deck. Imager 앞, spectrometer 뒤. 태양전지 4 wing (4×133.5 W = 534 W EOL). 두 개의 IAD (Inertia Adjustment Device)로 회전 균형 미세 조정.

**ACS.** Three Ithaco 60 A·m² torque rods, magnetometer, Adcole fine Sun sensor (FSS, ±32°, 0.005° resolution). Spin-stabilised at ~15 rpm with 3σ pointing 0.14° (8.4 arcmin). Modes: Acquisition, Precession, Spin Control, Normal, Idle (safe-mode for spinner). ACS auto-coded from MatrixX.

**자세제어.** 세 개의 Ithaco 60 A·m² 토크 rod, 자력계, Adcole FSS (±32°, 0.005°). ~15 rpm 회전 안정화, 3σ 0.14° (8.4 arcmin) 정밀도. 모드: Acquisition, Precession, Spin Control, Normal, Idle (회전체 safe-mode). MatrixX로 자동 코드 생성된 ACS.

**C&DH.** RAD6000 CPU (BAE Systems), 128 MB DRAM + 3 MB EEPROM. SEM, 4 GB SSR (SEAKR), VxWorks RTOS. SEM oven-controlled crystal at 2^22 Hz divided to 1 Hz/1 MHz for time-stamping.

**명령·데이터 처리.** RAD6000 (BAE), 128 MB DRAM + 3 MB EEPROM. SEM과 4 GB SSR (SEAKR), VxWorks RTOS. SEM의 OCXO는 2^22 Hz, 1 Hz / 1 MHz 클록 분배.

**EPS.** GaAs triple-junction solar cells; 15 A·h NiH battery (50 % DOD, 280 W during 35-min eclipse). PWM-FET regulation, > 95 % efficient.

**전력.** GaAs 트리플정션, 15 A·h NiH 배터리(50 % DOD, 35 min eclipse 시 280 W). PWM-FET 효율 95 % 이상.

**Telecom.** S-band transponder (Cincinnati Electronics), 4 Mbps downlink at 2215 MHz, 2 kbps uplink BPSK at 2040 MHz, NRZM baseband. Four patch antennas → near-4π steradian uplink coverage.

**통신.** Cincinnati Electronics S-band 트랜스폰더, 다운링크 4 Mbps @ 2215 MHz, 업링크 2 kbps @ 2040 MHz BPSK, NRZM 기저대역. 4-patch 안테나로 거의 4π 스테라디안 업링크.

### Part V: Ground System & Operations (§5) / 지상 시스템과 운영

**Mission Operations.** Run from MOC at SSL/UC Berkeley, co-located with FAST. Store-and-dump mode; ATS commands generated by MPS uploaded every two days, covering 4-5 days. ITOS for command/control. SatTrack v4.4 for flight dynamics. SERS (Spacecraft Emergency Response System) auto-pages on-call staff for anomalies.

**임무 운영.** UC Berkeley의 SSL MOC에서 FAST와 공동 운영. Store-and-dump 모드; ATS 명령은 MPS가 2일마다 업로드(4-5일분). ITOS로 명령 통제. SatTrack v4.4 비행 역학. SERS가 자동 경보·페이저 호출.

**Berkeley Ground Station.** 11-m parabolic dish, 3-axis drive (no zenith keyhole), full-duplex S-band, dual receivers with diversity, conical scan autotrack 0.1°, EIRP 63 dBW.

**Berkeley 지상국.** 11 m 파라볼릭 안테나, 3축 구동(천정 keyhole 없음), full-duplex S-band, 다이버시티 이중 수신기, 0.1° conical scan autotrack, EIRP 63 dBW.

**Backup stations.** Wallops (NASA), Weilheim (DLR), Santiago (Universidad de Santiago) provide 51 + 16 min/day extra. Berkeley + Wallops average 55 min/day.

**백업 지상국.** Wallops(NASA), Weilheim(DLR), Santiago(우니베르시다드 데 산티아고) 추가 51 + 16 min/day. Berkeley+Wallops 평균 55 min/day.

### Part VI: Science Operations & Data Analysis (§6) / 과학 운영과 데이터 분석

**§6.1 Science Operations.** Autonomous; main task is SSR management. SSR kept ≤ ~20 % full at end of Berkeley pass to leave room for X-class flares. A team scientist serves a *Tohban* role (modelled on Yohkoh) for daily monitoring. Data are available 1-3 days post-observation; FITS-formatted; archived at Berkeley, GSFC, ETH Zürich (HEDC).

**§6.1 과학 운영.** 자율 운영, 주요 작업은 SSR 관리. Berkeley pass 종료 시 SSR 사용량을 ≤ ~20 %로 유지. Yohkoh를 본받은 *Tohban* 과학자가 일일 모니터링. 데이터는 관측 1-3일 후 공개, FITS 포맷. Berkeley, GSFC, ETH Zürich(HEDC)에 아카이브.

**§6.2 Data Analysis.** Photon-by-photon telemetry → on-ground trade-offs of time/energy/spatial resolution. RHESSI software (Schwartz et al. 2002) is in IDL/SSW, freely downloadable, runs on UNIX/Windows. Common interface to SOHO, TRACE, GOES, BBSO products. Three workshops (~30 scientists each) trained users in image reconstruction and spectral analysis. Documentation at <http://hesperia.gsfc.nasa.gov/rhessidatacenter/>.

**§6.2 데이터 분석.** Photon 단위 텔레메트리 → 지상에서 분해능 trade-off. RHESSI 소프트웨어(Schwartz et al. 2002)는 IDL/SSW, 무료 배포, UNIX/Windows 호환. SOHO, TRACE, GOES, BBSO 데이터와 공통 인터페이스. 3회 워크숍(~30명/회)으로 영상 재구성과 분광 분석 교육. 문서: <http://hesperia.gsfc.nasa.gov/rhessidatacenter/>.

### Part VII: Summary (§7) / 요약

First flare detected 12 February 2002, a GOES C2 at 02:14 UT. By end August 2002, RHESSI had detected > 1900 flares above 12 keV, > 600 above 25 keV, and the **first** imaging spectroscopy of solar flares; first 3-10 keV hard X-ray microflares; the Sun continually emits hard X-rays above ~3 keV. On 23 July 2002 RHESSI obtained the **first** high-resolution gamma-ray line spectrum and the **first** images of a gamma-ray line, from a GOES X4.8 flare.

2002년 2월 12일 첫 플레어 (GOES C2, 02:14 UT) 검출. 같은 해 8월 말까지 12 keV 이상 1900회, 25 keV 이상 600회 검출, 태양 플레어 imaging spectroscopy의 **최초** 사례. 3-10 keV hard X-ray microflare 최초 검출, 태양은 ~3 keV 이상에서 지속적으로 X선을 방출함을 확인. 7월 23일 GOES X4.8 플레어로부터 gamma-ray 라인의 **최초** 고분해 스펙트럼과 영상 획득.

---

## 3. Key Takeaways / 핵심 시사점

1. **High-resolution imaging spectroscopy unifies two separate measurement traditions / 고분해 영상-분광 통합** — Before RHESSI, hard X-ray imaging (Yohkoh HXT, four bands) and gamma-ray spectroscopy (CGRO BATSE) lived in different instruments. RHESSI's RMC + GeD architecture delivers both at once: ~2.3 arcsec spatial *and* ~1 keV spectral resolution, enabling the first direct inversion of N(E,r,t).
   RHESSI 이전에는 hard X-ray imaging (Yohkoh HXT, 4 bands)과 gamma-ray spectroscopy (CGRO BATSE)가 별개 기기에 분리되어 있었다. RMC + GeD 구조는 ~2.3 arcsec 공간과 ~1 keV 분광을 동시에 달성해 N(E,r,t)의 직접 inversion을 가능하게 했다.

2. **Fourier-imaging is the only viable hard X-ray approach / Fourier 영상화가 hard X-ray의 유일한 길** — Focusing optics fail above ~10 keV. RMCs convert source angle into temporal modulation of count rate, which is precisely a Fourier component of the source. RHESSI samples ~1100 components per half rotation (Yohkoh HXT: 32). Angular resolution = $p/(2L)$.
   ~10 keV 이상에서는 focusing optics가 작동하지 않는다. RMC는 source angle을 시간 변조 신호로 변환하며, 이는 Fourier 성분과 일치한다. RHESSI는 반회전당 ~1100 성분 측정 (Yohkoh HXT는 32). 각 분해능은 $p/(2L)$.

3. **Cryogenically cooled segmented Ge detectors set a new standard / 저온 segmented Ge 검출기가 새 표준을 만든다** — Front+rear segmentation lets the *same* crystal handle 3 keV-17 MeV with sub-keV to 5-keV FWHM resolution. F/R coincidence recovers >250 keV photopeaks without losing imaging. SMA-actuated attenuators give 10^7 dynamic range.
   Front+rear segment 분할 덕분에 동일한 결정이 3 keV-17 MeV에서 sub-keV~5 keV FWHM 분해능을 제공한다. F/R coincidence가 250 keV 이상의 photopeak을 회복하며 imaging을 유지. SMA 셔터는 10^7 동적 범위 제공.

4. **Photon-by-photon telemetry is a paradigm shift in instrument data philosophy / Photon-by-photon 텔레메트리는 패러다임 변화** — RHESSI tells the ground every 24-bit event (energy, time, detector). This means analysts trade off time/energy/space resolution *after the fact* per science goal — no preselected images. Combined with free IDL/SSW software, RHESSI established the open-data culture later embraced by SDO and Solar Orbiter.
   RHESSI는 모든 24-bit event를 지상에 전송한다. 따라서 분석자는 분해능 trade-off를 *사후에* 과학 목표에 맞춰 자유롭게 결정. 무료 IDL/SSW와 결합되어, SDO, Solar Orbiter로 이어지는 open-data 문화의 시작점이 되었다.

5. **PI-mode SMEX is feasible — and risky / PI 모드 SMEX의 가능성과 위험** — RHESSI is the first SMEX with PI-led mission management (instrument + spacecraft + ops + data). The JPL shake-table accident, Pegasus delays, X-43 failure, and Reuven Ramaty's death made the mission a stress test of the PI model — but it succeeded and shaped subsequent SMEX programmes.
   RHESSI는 PI가 기기·우주선·운영·데이터를 모두 책임지는 첫 SMEX. JPL 진동시험 사고, Pegasus 지연, X-43 실패, Ramaty의 별세가 PI 모델을 시험했지만 성공해 후속 SMEX의 모범이 되었다.

6. **Twist-tolerance is the dominant alignment requirement / Twist 허용치가 핵심 정렬 요구사항** — A relative twist of $p/D$ between two grids (D = 9 cm) reduces modulation to zero. For the 34-µm pitch, this is ~1 arcmin; achieved with graphite-epoxy support tube and continuously monitored by the Twist Monitoring System (TMS) until launch.
   두 grid 간의 상대 twist가 $p/D$ ($D$=9 cm)를 넘으면 변조가 0이 된다. 34 µm pitch에서는 ~1 arcmin. 흑연-에폭시 지지 튜브와 발사 직전까지 TMS로 지속 감시.

7. **Imaging gamma-ray lines is the new science / Gamma-ray 라인 영상이 신과학** — RHESSI is the *first* mission to image gamma-ray lines (e.g., 2.223 MeV neutron-capture line, 0.511 MeV annihilation line). On 23 July 2002 it imaged the X4.8 flare and revealed that *ion* footpoints can differ in location from *electron* footpoints — a long-standing puzzle in flare acceleration.
   RHESSI는 gamma-ray 라인 영상화의 *최초* 사례. 2.223 MeV (중성자 포획), 0.511 MeV (양전자 소멸) 라인을 영상화. 2002년 7월 23일 X4.8 플레어에서 *이온* footpoint가 *전자* footpoint와 다른 위치에 있음을 보이며 가속 메커니즘 연구에 새 단서를 제공했다.

8. **Dual-use platform: a serendipitous all-sky monitor / 이중 용도 플랫폼: 우연한 전 천 모니터** — Lacking heavy shielding (an SMEX mass constraint) turned RHESSI into an all-sky hard X-ray/gamma-ray monitor: black-hole transients, the Crab, Galactic 511 keV, GRBs, and Terrestrial Gamma-ray Flashes (TGFs). The same rotation that drives RMC modulation also gives detector occultations for source localisation.
   SMEX 질량 제약으로 차폐가 부족했던 점이 오히려 RHESSI를 전 천 hard X-ray/gamma-ray 모니터로 만들었다: 블랙홀 천이, Crab, 은하 511 keV, GRB, 지구 감마선 섬광(TGF). RMC 변조를 만드는 회전이 detector occultation으로 source 위치 결정에도 사용된다.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 RMC Imaging Geometry / RMC 영상 기하

**Angular resolution of one RMC.** For a grid pair of pitch $p$ and separation $L$:
$$\boxed{\Delta\theta = \frac{p}{2L}}$$
- $p$: 슬릿 주기 / slit period (RHESSI: 34 µm-2.75 mm)
- $L$: 두 grid 사이 거리 / grid separation (RHESSI: 1.55 m)
- The factor 2 arises because transmission goes 0 → 50 % over $p/L$, and the FWHM of the response is half that.
- 인자 2는 투과율이 0 → 50 %로 변하는 angular 범위가 $p/L$이고, response의 FWHM이 그 절반이기 때문이다.

**Numerical.** $p = 34$ µm, $L = 1.55$ m → $\Delta\theta = 1.10\times10^{-5}$ rad = 2.27 arcsec. Coarsest grid: $p = 2.75$ mm → $\Delta\theta \approx 3.05$ arcmin.

수치 예: $p = 34$ µm, $L = 1.55$ m → $\Delta\theta = 1.10\times10^{-5}$ rad = 2.27 arcsec. 가장 거친 grid: $p = 2.75$ mm → $\Delta\theta \approx 3.05$ arcmin.

**Twist tolerance.**
$$\delta\phi \lesssim \frac{p}{D},\qquad D = 9~\text{cm grid diameter}$$
For finest grid ($p = 34$ µm): $\delta\phi \lesssim 3.78\times10^{-4}$ rad ≈ 1.3 arcmin. Beyond this the modulation collapses.

가장 미세한 grid에서 $p = 34$ µm → $\delta\phi \lesssim 3.78\times10^{-4}$ rad ≈ 1.3 arcmin. 이를 초과하면 변조가 사라진다.

### 4.2 RMC Modulation Profile / RMC 변조 함수

For an off-axis point source at angle $\theta$ from the optical axis (orthogonal to slits), and rotation phase $\phi$, the transmitted count rate (idealised, equal slits/slats) is approximately
$$M(\phi; \theta) = \frac{C_0}{2}\Bigl[1 + T(\phi)\cos\bigl(\tfrac{2\pi L}{p}\sin\theta \cdot \cos(\phi-\phi_0)\bigr)\Bigr]$$
where $T(\phi)$ is a smooth triangular envelope of the grid response and $\phi_0$ is the source position angle. Over short rotation arcs the waveform is quasi-triangular; its amplitude is proportional to source intensity and its phase encodes the position angle. Different RMCs (different $p$) sample different spatial frequencies, providing a logarithmic ladder of Fourier components.

축 외 점원이 광축과 각도 $\theta$, 회전 위상 $\phi$일 때, 이상적으로 (슬릿/슬랫 동일 폭) 통과되는 계수율은 위 식과 같이 표현된다. $T(\phi)$는 grid 응답의 매끈한 삼각형 envelope. 진폭 ∝ source 세기, 위상 ∝ source 위치각. 9개의 RMC가 9개의 서로 다른 $p$로 서로 다른 공간주파수를 샘플하여 로그 사다리 형태의 Fourier 성분 집합을 제공.

In one half rotation (2 s), the 9 RMCs collect ~1100 Fourier components — far richer than the 32 components measured by Yohkoh HXT. These components are the visibilities $V(u,v)$ in the radio-interferometry analogue, and image reconstruction algorithms (back-projection, CLEAN, MEM, Pixon) are adapted from interferometry.

반회전(2 s) 동안 9 RMC가 ~1100개의 Fourier 성분을 수집 (Yohkoh HXT는 32개). 이는 전파 간섭계의 visibility $V(u,v)$에 해당하며, back-projection, CLEAN, MEM, Pixon 등 간섭계 알고리즘을 그대로 활용해 영상 재구성.

### 4.3 Bremsstrahlung Photon Spectrum / 제동복사 광자 스펙트럼

The thin-target bremsstrahlung photon flux from a population of accelerated electrons with flux density $F(E)$ [electrons cm$^{-2}$ s$^{-1}$ keV$^{-1}$] colliding with ambient ions of density $n$ is
$$I(\epsilon) = \frac{n}{4\pi R^2}\int_{\epsilon}^{\infty} F(E)\,\sigma_{\rm B}(\epsilon, E)\,v(E)\,dE$$
- $\epsilon$: photon energy / 광자 에너지
- $E$: electron kinetic energy / 전자 운동에너지
- $\sigma_{\rm B}(\epsilon, E)$: bremsstrahlung cross section (Bethe-Heitler) / Bethe-Heitler 단면적
- $v(E)$: electron velocity / 전자 속도
- $R$: heliocentric distance / 1 AU 거리

For thick-target (Brown 1971), $F(E)$ is replaced by an integral that includes Coulomb energy loss; the inverted electron spectrum has an additional energy index of $+2$ due to the loss kernel. For a power-law electron flux $F(E) \propto E^{-\delta}$, the photon spectrum is approximately $I(\epsilon) \propto \epsilon^{-(\delta+1)}$ (thin-target) or $\epsilon^{-(\delta-1)}$ (thick-target). RHESSI's ~1 keV resolution lets us *invert* $I(\epsilon)$ to get $F(E)$ directly (Johns & Lin 1992).

박막 표적(thin-target)에서 가속 전자 flux $F(E)$가 밀도 $n$의 ambient ion에 충돌해 만드는 광자 flux는 위 식과 같다. Thick-target (Brown 1971)에서는 Coulomb loss 적분이 추가되어 전자 spectral index가 $+2$ 만큼 가팔라진다. Power-law $F(E)\propto E^{-\delta}$일 때, 광자 spectrum은 thin-target에서 $\epsilon^{-(\delta+1)}$, thick-target에서 $\epsilon^{-(\delta-1)}$. RHESSI의 ~1 keV 분해능 덕분에 $I(\epsilon)$로부터 $F(E)$를 직접 inversion 가능 (Johns & Lin 1992).

### 4.4 Spatially Dependent Continuity Equation / 공간 의존 연속 방정식

To go from the inferred $N(E,\vec r,t)$ to the *acceleration* source $S$, one solves
$$\frac{\partial N}{\partial t} + \vec v\cdot\nabla N + \frac{\partial}{\partial E}(\dot E\,N) = S(E,\vec r, t)$$
- $N(E,\vec r,t)$: bremsstrahlung-producing electrons / 제동복사 전자 수밀도
- $\dot E = -\dot E_{\rm Coul} \propto -n/\sqrt{E}$: Coulomb loss rate / Coulomb 손실율
- $\vec v$: streaming velocity / 흐름 속도
- $S$: acceleration source term / 가속 source 항

Given context measurements of $n$, $T$, magnetic topology (e.g., from SOHO/MDI, TRACE, GOES, ground-based H$\alpha$), one can untangle thermal vs. nonthermal contributions and test specific acceleration models (stochastic, electric-field, shock).

추론된 $N(E,\vec r,t)$로부터 가속 source $S$를 얻기 위해 위 방정식을 푼다. $n$, $T$, 자기장 구조 측정(SOHO/MDI, TRACE, GOES, 지상 Hα 등)과 결합하면 thermal/nonthermal 기여를 분리하고 stochastic·전기장·shock 등 특정 가속 모델을 검증할 수 있다.

### 4.5 Gamma-ray Lines / 감마선 라인

**2.223 MeV neutron-capture line.** Energetic ions produce neutrons via spallation; neutrons thermalise and capture on protons:
$$n + p \rightarrow d + \gamma\,(2.223~\text{MeV})$$
The line is delayed (~100 s) because thermalisation/capture takes time. Since line counts dominate background, RHESSI can image this line uniquely.

**2.223 MeV 중성자 포획 라인.** 가속 이온이 spallation으로 중성자 생성 → 중성자 열화 → 양성자 포획. 라인은 ~100 초 지연 (열화 시간). 배경에 비해 라인 신호가 우세해 RHESSI가 단독 영상화 가능.

**0.511 MeV positron annihilation.** Energetic ions produce positrons (via β^+ decay of unstable nuclei or π^+ decay); positrons slow down then annihilate:
$$e^+ + e^- \rightarrow 2\gamma\,(\text{each } 511~\text{keV})$$
Line shape encodes ambient density and temperature. Width $\sim$ 1-10 keV depending on whether annihilation is direct or via positronium.

**0.511 MeV 양전자 소멸.** 가속 이온이 β^+ 붕괴 또는 π^+ 붕괴로 양전자 생성 → 감속 후 소멸. 라인 폭 ~1-10 keV로 ambient 밀도·온도와 positronium 형성 여부를 진단.

**Other narrow lines.** ^{20}Ne (1.634 MeV), ^{12}C (4.438 MeV), ^{16}O (6.129 MeV) de-excitation. Cross section threshold for ^{20}Ne (~2.5 MeV) is unusually low, so ^{20}Ne enhancement signals large fluxes of low-energy ions.

**기타 좁은 라인.** ^{20}Ne (1.634 MeV), ^{12}C (4.438 MeV), ^{16}O (6.129 MeV). ^{20}Ne의 cross section threshold가 ~2.5 MeV로 낮아, 이 라인의 강화는 저에너지 이온 다량 가속의 신호.

### 4.6 Detection Numerics / 검출 수치

| Quantity / 항목 | Value / 값 |
|---|---|
| Energy range / 에너지 범위 | 3 keV-17 MeV |
| FWHM resolution at 3 keV / 3 keV FWHM | $\lesssim$ 1 keV |
| FWHM at 5 MeV / 5 MeV FWHM | ~5 keV |
| Angular resolution / 각 분해능 | 2.3 arcsec to 100 keV; 7 arcsec at 400 keV; 36 arcsec at 15 MeV |
| Temporal / 시간 분해능 | 2 s detailed image, tens of ms basic |
| FOV | full Sun (~1°) |
| Effective area / 유효 면적 | 10^{-3} cm² @ 3 keV (atten in); 32 cm² @ 10 keV; 60 cm² @ 100 keV; 15 cm² @ 5 MeV |
| Detectors / 검출기 | 9 GeDs, 7.1 cm dia × 8.5 cm long, < 75 K |
| Imager grids / 영상 grid | 9 pairs, 34 µm-2.75 mm pitch, 1.55 m separation |
| Spacecraft mass / 우주선 질량 | 291.1 kg total, 130.8 kg instrument |
| Power / 전력 | 220.4 W total, 142.3 W instrument |
| Telemetry / 텔레메트리 | 4 Mbps down, 2 kbps up |
| Storage / 저장 | 4 GB SSR |
| Spin / 회전 | 15 rpm, pointing 0.2° |
| Orbit / 궤도 | 38° inclination, 600 km circular |
| Launch / 발사 | 5 Feb 2002, Pegasus-XL |
| Expected flares / 예상 플레어 | ~1000 imaged > 100 keV; tens with spectroscopy to 10 MeV |

### 4.7 Effective Area Trade-off / 유효 면적 트레이드오프

The total photopeak effective area $A_{\rm eff}(\epsilon)$ is the sum of contributions from front-only, rear-only, and front+rear (F/R) coincidence modes:
$$A_{\rm eff}(\epsilon) = A_{\rm F}(\epsilon) + A_{\rm R}(\epsilon) + A_{\rm FR}(\epsilon)\cdot\mathbb{1}_{\rm coinc}(\epsilon)$$
- 3-30 keV: $A_{\rm F}$ dominates (light photons stop in front segment).
- 30-250 keV: $A_{\rm F}$ peaks around ~60 cm² at 100 keV.
- 250 keV-2 MeV: $A_{\rm R}$ dominates as photons penetrate to rear.
- 250 keV-17 MeV: $A_{\rm FR}$ contributes when photons Compton-scatter and deposit energy in both segments.

Inserting attenuators reduces the low-energy effective area (Fig. 4(c) of paper) by ~3-4 orders of magnitude at 3 keV, preserving spectroscopy of large flares without saturation.

전체 photopeak 유효 면적 $A_{\rm eff}(\epsilon)$은 front 단독, rear 단독, F/R coincidence의 합이다. 3-30 keV에서는 $A_{\rm F}$ 우세, 30-250 keV는 $A_{\rm F}$가 100 keV에서 ~60 cm² 정점, 250 keV-2 MeV는 $A_{\rm R}$ 우세, 250 keV-17 MeV는 $A_{\rm FR}$ 기여. 셔터 삽입 시 3 keV에서 유효 면적이 ~3-4 자릿수 감소(Fig. 4(c)) → 대형 플레어에서도 saturation 없이 분광 가능.

### 4.8 Worked Example / 작업 예제

**Question:** A flare emits a power-law photon spectrum $I(\epsilon) = 0.1\,(\epsilon/50~\text{keV})^{-3}$ photons s$^{-1}$ cm$^{-2}$ keV$^{-1}$ at 1 AU. Estimate counts in RHESSI's 50-100 keV band over a 4-s integration. Use 60 cm² effective area.

**문제:** 플레어가 1 AU에서 $I(\epsilon) = 0.1\,(\epsilon/50~\text{keV})^{-3}$ photons s$^{-1}$ cm$^{-2}$ keV$^{-1}$의 power-law 광자 spectrum을 방출한다. 50-100 keV에서 4 초 동안 RHESSI 계수는? 유효 면적 60 cm² 사용.

**Solution / 풀이:**
$$\int_{50}^{100} I(\epsilon)\,d\epsilon = 0.1\int_{50}^{100} (\epsilon/50)^{-3} d\epsilon = 0.1\cdot 50^3 \int_{50}^{100} \epsilon^{-3} d\epsilon$$
$$= 12500\,[-\tfrac12\epsilon^{-2}]_{50}^{100} = 12500\cdot\frac12(50^{-2}-100^{-2})$$
$$= 6250\cdot(4.0\times10^{-4} - 1.0\times10^{-4}) = 6250\cdot 3.0\times10^{-4}$$
$$= 1.875~\text{ph s}^{-1}\text{cm}^{-2}$$
Counts = $1.875 \times 60 \times 4 \approx 450$ counts. Photon-noise σ ≈ 21 counts (~5 % statistical).

계수 = $1.875\times60\times4\approx 450$. 광자 잡음 σ ≈ 21 (~5 %).

### 4.9 Case Study: 23 July 2002 X4.8 Flare / 사례 연구: 2002년 7월 23일 X4.8 플레어

The first gamma-ray line image obtained by RHESSI revealed an instructive surprise. Below ~30 keV the source is a single elongated coronal blob; above ~40 keV the image splits into three distinct sources (Fig. 3 of paper). Northern and southern sources have similar power-law spectra (slopes −2.2 / −3.2 and −2.1 / −2.8 with break around 100 keV), interpreted as conjugate footpoints of accelerated electrons. The middle source shows a much steeper single power-law spectrum (slope −3.2), consistent with a *coronal* loop-top source (thermal + nonthermal mix). The 2.223 MeV neutron-capture line image, when compared with the >40 keV electron footpoints, revealed a *spatial offset* between ion and electron footpoints — a finding that constrains acceleration models requiring co-spatial ions and electrons (e.g., simple stochastic).

RHESSI가 처음 얻은 gamma-ray 라인 영상은 흥미로운 결과를 보여주었다. ~30 keV 이하에서는 source가 하나의 길쭉한 코로나 덩어리이지만, ~40 keV 이상에서는 세 개의 분리된 source로 갈라진다(논문 Fig. 3). 북쪽·남쪽 source는 비슷한 power-law spectrum (기울기 −2.2/−3.2 및 −2.1/−2.8, ~100 keV 부근 break)을 보여 가속 전자의 conjugate footpoint로 해석된다. 가운데 source는 −3.2의 더 가파른 단일 power-law로, *코로나* loop-top source(thermal + nonthermal 혼합)에 부합한다. 2.223 MeV 중성자 포획 라인 영상을 >40 keV 전자 footpoint와 비교한 결과, 이온과 전자 footpoint 사이에 *공간적 어긋남*이 나타나, 동공간 가속을 가정하는 단순 모델(예: 단일 stochastic)에 제약을 주었다.

This case study illustrates the unique science enabled by simultaneous imaging spectroscopy: with photon-by-photon data, the same flare can be re-binned in 20 logarithmic energy channels (1 keV bins at 4 keV, 28 keV bins at 138 keV) to follow how the source morphology evolves with energy, then compared directly with gamma-ray line images.

이 사례는 동시 imaging-spectroscopy가 제공하는 독특한 과학을 보여준다: photon-by-photon 데이터 덕분에 동일 플레어를 20개의 로그 에너지 채널(4 keV에서 1 keV bin, 138 keV에서 28 keV bin)로 다시 binning하여 source 모폴로지의 에너지 의존성을 추적하고, gamma-ray 라인 영상과 직접 비교할 수 있다.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1895 ── Röntgen discovers X-rays
1923 ── Bremsstrahlung theory (Bethe-Heitler)
1958 ── Peterson & Winckler — first solar hard X-ray detection (balloon)
1973 ── OSO-7 — first solar gamma-ray flare
1977 ── Hinotori RMC concept (Makishima)
1980 ── SMM (HXRBS, GRS)  — keV spectroscopy, no imaging
1991 ── Yohkoh HXT — 4-band hard X-ray imaging
1991 ── CGRO/BATSE — gamma-ray spectroscopy, no imaging
1992 ── Johns & Lin — inversion of electron spectrum from photons
1995 ── Lin et al. — WIND 3DP first observed escaping electrons
1997 ── HESSI selected for SMEX
2002 ── *RHESSI launched* — Lin et al. 2002 (THIS PAPER)
2002 ── First gamma-ray line image (X4.8 flare, 23 July)
2010 ── SDO — context observations for RHESSI flares
2018 ── RHESSI decommissioned (~120,000 flares catalogued)
2020 ── STIX on Solar Orbiter — RHESSI's heir, with focusing optics
```

This paper sits at the watershed between "imaging without spectroscopy" (Yohkoh HXT) and "spectroscopy without imaging" (BATSE) eras and the modern *imaging-spectroscopy* era. It is the design reference of the field for ~20 years.

이 논문은 "분광 없는 영상화"(Yohkoh HXT)와 "영상화 없는 분광"(BATSE)이라는 두 시대를 연결하고, 현대의 *imaging-spectroscopy* 시대를 여는 분기점에 있다. ~20년간 이 분야의 설계 표준이 되었다.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Makishima et al. 1977 (Hinotori RMC) | Original solar RMC concept that RHESSI inherited and refined / RHESSI가 계승·정교화한 최초의 태양 RMC 개념 | Foundational design heritage / 설계 기반 |
| Kosugi et al. 1991 (Yohkoh HXT, *Solar Phys.* 136, 17) | Predecessor RMC instrument; 4-band imager that RHESSI surpasses in resolution and Fourier-component count / RHESSI가 분해능과 Fourier 성분 수에서 능가한 직전 세대 RMC | Direct lineage / 직접 계보 |
| Hurford & Curtis 2002, this volume | RHESSI imaging concept and roll-angle aspect details / RHESSI imaging 개념 및 RAS 상세 | Companion paper / 동반 논문 |
| Smith et al. 2002, this volume | Detailed GeD spectrometer description / GeD 분광계 상세 | Companion paper / 동반 논문 |
| Schwartz et al. 2002, this volume | Data analysis software (IDL/SSW) / IDL/SSW 데이터 분석 소프트웨어 | Companion paper / 동반 논문 |
| Curtis et al. 2002, this volume | IDPU electronics / 기기 전자 | Companion paper / 동반 논문 |
| Zehnder et al. 2002, this volume | SAS and CCD RAS / SAS와 CCD RAS | Companion paper / 동반 논문 |
| Lin & Hudson 1976 (*Solar Phys.* 50, 153) | Showed > 20 keV electrons carry energy comparable to total flare output — the scientific motivation / >20 keV 전자가 플레어 전체 출력에 필적하는 에너지를 운반함을 보임 — 과학적 동기 | Sets science requirements / 과학 요구사항 정의 |
| Johns & Lin 1992 (*Solar Phys.* 137, 121) | Inversion of photon → electron spectrum that RHESSI's energy resolution makes possible / RHESSI 분해능으로 가능해진 광자→전자 spectrum inversion | Enables key science / 핵심 과학 가능 |
| Ramaty & Murphy 1987 (*Space Sci. Rev.* 45, 213) | Theoretical framework for gamma-ray line spectrum from accelerated ions / 가속 이온의 gamma-ray 라인 이론 틀 | Theoretical basis for ion science / 이온 과학 이론 |
| Brown 1971 (*Solar Phys.*) | Thick-target bremsstrahlung — basis for spectrum interpretation / Thick-target bremsstrahlung 이론 | Spectrum modelling / 스펙트럼 해석 |
| Krucker, Hudson et al. various | RHESSI early-results papers in the same Solar Physics 210 volume / 같은 Solar Physics 210 호의 RHESSI 초기 결과 논문 | First science output / 첫 과학 결과 |
| Krucker et al. 2020 (STIX paper) | Solar Orbiter's STIX is RHESSI's spiritual successor with focusing optics / STIX는 focusing optics를 갖춘 RHESSI의 후계 | Modern descendant / 현대 후속 |

### 6.1 RHESSI vs. Predecessors and Successor / 선행·후속 기기와 RHESSI 비교

| Property / 성질 | Yohkoh HXT (1991) | CGRO BATSE (1991) | **RHESSI (2002)** | STIX (2020) |
|---|---|---|---|---|
| Energy range / 에너지 범위 | 15-100 keV | 25 keV-10 MeV | **3 keV-17 MeV** | 4-150 keV |
| Spectral resolution / 분광 분해능 | 4 broad bands | 30 % FWHM | **~1 keV FWHM** | ~1 keV @ 6 keV |
| Imaging / 영상화 | Yes (5 arcsec) | No | **Yes (2.3 arcsec)** | Yes (~7 arcsec) |
| Imaging method / 방법 | RMC, 64 grids | None | **RMC, 9 grids** | RMC + tungsten focusing |
| Fourier components / 성분 수 | 32 | 0 | **~1100** | ~30 |
| Gamma-ray line imaging / 감마선 라인 영상 | No | No | **First** | No |
| Detector / 검출기 | NaI(Tl) | NaI(Tl) | **GeD (cryo)** | Cd(Zn)Te |
| Telemetry / 텔레메트리 | Image-based | Spectrum-based | **Photon-by-photon** | Photon-by-photon |
| Data philosophy / 데이터 철학 | Closed | Closed | **Open / IDL/SSW** | Open |

이 비교는 RHESSI의 차별성을 명확히 보여준다: 동시에 영상·분광이 가능한 첫 기기이며, gamma-ray 라인 영상화의 최초 사례이고, photon-by-photon 텔레메트리와 무료 SSW 소프트웨어로 데이터 개방의 새 표준을 세웠다. STIX는 이 유산을 잇되 focusing optics와 작은 mission profile로 진화시킨 형태이다.

This comparison highlights what made RHESSI distinctive: the first instrument to combine imaging and spectroscopy, the first to image gamma-ray lines, and the first to push photon-by-photon telemetry plus free SSW software as the new open-data norm. STIX inherits this legacy while evolving toward focusing optics and a smaller mission envelope.

---

## 7. References / 참고문헌

- Lin, R. P., Dennis, B. R., Hurford, G. J., et al., "The Reuven Ramaty High-Energy Solar Spectroscopic Imager (RHESSI)", *Solar Physics* **210**, 3-32, 2002. DOI: 10.1023/A:1022428818870 — *primary source / 본 논문*.
- Hurford, G. J., Schmahl, E. J., Schwartz, R. A., et al., "The RHESSI Imaging Concept", *Solar Physics* **210**, 61-86, 2002.
- Smith, D. M., Lin, R. P., Turin, P., et al., "The RHESSI Spectrometer", *Solar Physics* **210**, 33-60, 2002.
- Schwartz, R. A., Csillaghy, A., Tolbert, A. K., et al., "RHESSI Data Analysis Software", *Solar Physics* **210**, 165-191, 2002.
- Curtis, D. W., Berg, P., Gordon, D., et al., "The HESSI Instrument Data Processing Unit", *Solar Physics* **210**, 115-141, 2002.
- Zehnder, A., Bialkowski, J., Burri, F., et al., "The Solar Aspect System and CCD Roll Angle System on RHESSI", *Solar Physics* **210**, 143-164, 2002.
- Hurford, G. J. and Curtis, D. W., "The PMT-Based RHESSI Roll Angle System", *Solar Physics* **210**, 101-114, 2002.
- Makishima, K., Miyamoto, S., Murakami, T., et al., "Modulation Collimator Imaging for Hard X-rays", in van der Hucht & Vaiana (eds.), *New Instrumentation for Space Astronomy*, Pergamon, 1977.
- Kosugi, T., Makishima, K., Murakami, T., et al., "The Hard X-ray Telescope (HXT) for the Solar-A Mission", *Solar Physics* **136**, 17-36, 1991.
- Lin, R. P. and Hudson, H. S., "Non-Thermal Processes in Large Solar Flares", *Solar Physics* **50**, 153-178, 1976.
- Johns, C. and Lin, R. P., "The Derivation of Parent Electron Spectra from Bremsstrahlung Hard X-Ray Spectra", *Solar Physics* **137**, 121-140, 1992.
- Brown, J. C., "The Deduction of Energy Spectra of Non-Thermal Electrons in Flares from the Observed Dynamic Spectra of Hard X-Ray Bursts", *Solar Physics* **18**, 489-502, 1971.
- Ramaty, R. and Murphy, R. J., "Nuclear Processes and Accelerated Particles in Solar Flares", *Space Sci. Rev.* **45**, 213-268, 1987.
- Ramaty, R., Mandzhavidze, N., Kozlovsky, B., and Murphy, R. J., "Solar Atmospheric Abundances and Energy Content in Flare-Accelerated Ions from Gamma-Ray Spectroscopy", *Astrophys. J.* **455**, L193, 1995.
- Emslie, A. G., Brown, J. C., and Mackinnon, A. L., "On the Numbers and Energy Content of Mildly Relativistic Solar-Flare Electrons", *Astrophys. J.* **485**, 430-440, 1997.
- Prince, T. A., Hurford, G. J., Hudson, H. S., and Crannell, C. J., "Gamma-Ray and Hard X-Ray Imaging of Solar Flares", *Solar Physics* **118**, 269-290, 1988.
- Bougeret, J.-L., Kaiser, M. L., Kellogg, P. J., et al., "Waves: The Radio and Plasma Wave Investigation on the Wind Spacecraft", *Space Sci. Rev.* **71**, 231-263, 1995.
- Lin, R. P., Anderson, K. A., Ashford, S., et al., "A Three-Dimensional Plasma and Energetic Particle Investigation for the Wind Spacecraft", *Space Sci. Rev.* **71**, 125-153, 1995.
- Fishman, G. J., Bhat, P. N., Mallozzi, R., et al., "Discovery of Intense Gamma-Ray Flashes of Atmospheric Origin", *Science* **264**, 1313-1316, 1994.
- Krucker, S., Hurford, G. J., Grimm, O., et al., "The Spectrometer/Telescope for Imaging X-rays (STIX)", *Astron. & Astrophys.* **642**, A15, 2020 — *RHESSI's heir / RHESSI 후계*.
