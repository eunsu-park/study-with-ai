---
title: "The SOHO Mission: An Overview"
authors: Vicente Domingo, Bernhard Fleck, Arthur I. Poland
year: 1995
journal: "Solar Physics, Vol. 162, pp. 1–37"
doi: "10.1007/BF00733425"
topic: Solar Observation
tags: [SOHO, L1, helioseismology, corona, solar wind, MDI, LASCO, EIT, GOLF, VIRGO, SUMER, CDS, UVCS, SWAN, CELIAS, COSTEP, ERNE, ESA, NASA, space mission]
status: completed
date_started: 2026-04-16
date_completed: 2026-04-16
---

# 8. The SOHO Mission: An Overview / SOHO 미션: 개관

---

## 1. Core Contribution / 핵심 기여

이 논문은 **SOHO(Solar and Heliospheric Observatory)**의 과학적 목표, 12개 탑재체 사양, 우주선 설계, L1 헤일로 궤도, 운영 계획, 지상 세그먼트를 종합적으로 기술한 미션 개요 논문이다. SOHO는 ESA의 Horizon 2000 프로그램(STSP 코너스톤)과 NASA의 협력으로 탄생한 최초의 포괄적 우주 태양 관측소로, 태양 내부(helioseismology)에서 코로나(원격 탐사), 그리고 태양풍(현장 측정)까지 하나의 플랫폼에서 동시에 관측하는 것을 목표로 한다. 1995년 12월 발사되어 L1 라그랑주점의 헤일로 궤도에 투입되었으며, 30년이 지난 현재까지도 운영 중인 태양 물리학 역사상 가장 성공적인 미션 중 하나이다. 12개 기기는 세 그룹—helioseismology(GOLF, VIRGO, MDI), solar atmosphere remote sensing(SUMER, CDS, EIT, UVCS, LASCO, SWAN), solar wind in-situ(CELIAS, COSTEP, ERNE)—으로 나뉘며, 이들이 결합하여 태양의 심부에서 태양권까지를 하나의 연결된 시스템으로 연구할 수 있게 한다.

This paper is the comprehensive mission overview of **SOHO (Solar and Heliospheric Observatory)**, describing its scientific objectives, 12-instrument payload, spacecraft design, L1 halo orbit, operations plan, and ground segment. SOHO is the first comprehensive space-based solar observatory, born from the ESA Horizon 2000 program (STSP cornerstone) in cooperation with NASA, designed to simultaneously observe from the solar interior (helioseismology) through the corona (remote sensing) to the solar wind (in-situ measurements) on a single platform. Launched in December 1995 into a halo orbit around the L1 Lagrangian point, it remains one of the most successful missions in solar physics history, still operational after 30 years. The 12 instruments are organized into three groups—helioseismology (GOLF, VIRGO, MDI), solar atmosphere remote sensing (SUMER, CDS, EIT, UVCS, LASCO, SWAN), and solar wind in-situ (CELIAS, COSTEP, ERNE)—which together enable the study of the Sun from its deep core to the heliosphere as a coupled system.

---

## 2. Reading Notes / 읽기 노트

### §1 Introduction (pp. 1–8)

논문은 태양 물리학의 핵심 미해결 문제 세 가지를 제시하며 시작한다: (1) 태양 내부의 구조와 동역학, (2) 가열 메커니즘과 코로나 및 태양풍의 동역학, (3) 태양풍의 생성과 가속. 이 세 문제는 1960년대부터 축적된 관측적 발견에 뿌리를 두고 있다.

The paper opens by posing three key unsolved problems in solar physics: (1) the structure and dynamics of the solar interior, (2) heating mechanisms and the dynamics of the corona and solar wind, and (3) the origin and acceleration of the solar wind. These three problems are rooted in observational discoveries accumulated since the 1960s.

**Helioseismology의 역사적 배경**: Leighton, Noyes, and Simon (1962)이 태양 표면의 5분 진동(five-minute oscillations)을 발견한 이래, Ulrich (1970)와 Leibacher and Stein (1971)이 이를 태양 내부에 갇힌 정상 음파(standing acoustic waves)로 해석하였고, Deubner (1975)가 이론적 예측과 일치하는 진단 다이어그램(diagnostic diagram)을 관측적으로 확인하였다. 이후 helioseismology는 태양 내부의 온도, 밀도, 화학 조성, 회전 프로파일, 대류대 저부의 구조를 밝히는 강력한 도구로 성장하였다.

**Historical background of helioseismology**: Since Leighton, Noyes, and Simon (1962) discovered five-minute oscillations on the solar surface, Ulrich (1970) and Leibacher and Stein (1971) interpreted these as standing acoustic waves trapped within the solar interior, and Deubner (1975) observationally confirmed the diagnostic diagram matching theoretical predictions. Helioseismology subsequently grew into a powerful tool for probing the temperature, density, chemical composition, rotation profile, and base-of-convection-zone structure of the solar interior.

**우주 태양 관측의 역사**: Skylab (1973)의 ATM(Apollo Telescope Mount)은 최초의 포괄적 우주 태양 관측을 수행하여 코로나 홀, 코로나 질량 방출(CME), 전이 영역의 미세 구조를 발견하였다. Solar Maximum Mission (SMM, 1980)은 플레어와 총 태양 복사(total solar irradiance)의 변동을 처음으로 정밀 측정하였다. 그러나 이들 미션은 저지구 궤도(LEO)에서 운영되어 지구 차폐에 의한 관측 공백(45분 궤도 주기 중 약 36분만 관측 가능)이 불가피했다.

**History of space solar observation**: Skylab's ATM (Apollo Telescope Mount, 1973) performed the first comprehensive space-based solar observations, discovering coronal holes, coronal mass ejections (CMEs), and fine-scale structure in the transition region. The Solar Maximum Mission (SMM, 1980) made the first precise measurements of flare emissions and total solar irradiance variations. However, these missions operated in low-Earth orbit (LEO), inevitably suffering observational gaps from Earth occultation (~36 minutes of observation out of a 45-minute orbital period).

**SOHO의 탄생**: SOHO는 1982년에 처음 제안되었고, ESA의 Horizon 2000 장기 계획에서 STSP(Solar-Terrestrial Science Programme) 코너스톤의 한 요소로 1986년 2월 승인되었다(다른 요소는 Cluster). ESA-NASA 양해각서(MOU)가 1989년에 체결되어, ESA가 우주선과 12개 탑재체 중 9개의 유럽 분담분을 제공하고, NASA가 발사체(Atlas II-AS), 운영, 지상 세그먼트, 그리고 3개의 미국 탑재체(MDI, LASCO, UVCS)를 제공하기로 하였다. Phase B는 1989년 12월, Phase C/D는 1991년에 시작되었고, 탑재체는 1993–94년에 납품되어, 1995년 8월에 Cape Canaveral로 발송되었다.

**Birth of SOHO**: SOHO was first proposed in 1982 and approved in February 1986 as part of the STSP (Solar-Terrestrial Science Programme) cornerstone within ESA's Horizon 2000 long-term plan (the other element being Cluster). An ESA-NASA Memorandum of Understanding (MOU) was signed in 1989, with ESA providing the spacecraft and the European share of 9 of the 12 instruments, and NASA providing the launch vehicle (Atlas II-AS), operations, ground segment, and 3 US instruments (MDI, LASCO, UVCS). Phase B began in December 1989, Phase C/D in 1991, instruments were delivered in 1993–94, and the spacecraft was shipped to Cape Canaveral in August 1995.

### §2 Payload (pp. 9–16)

12개 기기는 세 그룹으로 나뉘며, 각 그룹은 태양의 서로 다른 영역을 연구한다. Fig. 4는 탑재체의 전체 배치를 보여준다.

The 12 instruments are divided into three groups, each studying different domains of the Sun. Fig. 4 shows the overall payload accommodation.

#### 2.1 Helioseismology 기기 / Helioseismology Instruments

**GOLF (Global Oscillations at Low Frequencies)**: Na 증기 공명 산란(resonant scattering)을 이용하여 전 태양 디스크(full-disk, Sun-as-a-star) 속도 진동을 측정한다. 감도 <1 mm/s, 주파수 범위 0.1 μHz~6 mHz, 구면 조화 차수 $l \le 3$. 특히 가장 깊은 태양 내부를 탐사하는 저차 g-mode 검출을 목표로 한다. PI: A. Gabriel (IAS, Orsay). 비트율: 0.16 kb/s.

**GOLF (Global Oscillations at Low Frequencies)**: Uses Na-vapor resonant scattering to measure full-disk (Sun-as-a-star) velocity oscillations. Sensitivity <1 mm/s, frequency range 0.1 μHz to 6 mHz, spherical harmonic degree $l \le 3$. A key objective is detecting low-order g-modes that probe the deepest solar interior. PI: A. Gabriel (IAS, Orsay). Bit rate: 0.16 kb/s.

**VIRGO (Variability of Solar Irradiance and Gravity Oscillations)**: 세 가지 하위 기기로 구성된다: (1) 3채널 SPM(Sun-PhotoMeter, 402/500/862 nm 밴드), (2) 12픽셀 LOI(Luminosity Oscillations Imager, 분해능 $l \le 7$), (3) 절대 복사계(PMO6-V 및 DIARAD, 총 태양 복사량 TSI 정밀도 0.15%). 강도 진동(intensity oscillations)을 통한 helioseismology와 태양 상수(solar constant) 장기 모니터링을 동시에 수행한다. PI: C. Fröhlich (PMOD/WRC). 비트율: 0.1 kb/s.

**VIRGO (Variability of Solar Irradiance and Gravity Oscillations)**: Comprises three sub-instruments: (1) 3-channel SPM (Sun-PhotoMeter, 402/500/862 nm bands), (2) 12-pixel LOI (Luminosity Oscillations Imager, resolving $l \le 7$), (3) absolute radiometers (PMO6-V and DIARAD, TSI precision 0.15%). Simultaneously performs helioseismology through intensity oscillations and long-term solar constant (TSI) monitoring. PI: C. Fröhlich (PMOD/WRC). Bit rate: 0.1 kb/s.

**MDI/SOI (Michelson Doppler Imager / Solar Oscillations Investigation)**: Ni I 676.8 nm 흡수선을 이용한 Michelson 간섭계로 전 태양 도플러그램(Dopplergram)을 촬영한다. 1024×1024 CCD로 전 디스크 모드(2"/pixel, $l \le 1500$)와 고분해능 모드(0.65"/pixel, $l \le 4500$)를 지원한다. 속도(velocity), 강도(intensity), 자기장(line-of-sight magnetic field)을 동시에 측정한다. 연속 텔레메트리 5 kb/s, 고속 모드에서 160 kb/s 추가. 매분 전 디스크 도플러그램을 생산하는 "workhorse" 기기이다. PI: P. Scherrer (Stanford). 

**MDI/SOI (Michelson Doppler Imager / Solar Oscillations Investigation)**: A Michelson interferometer using the Ni I 676.8 nm absorption line to produce full-disk Dopplergrams. The 1024×1024 CCD supports full-disk mode (2"/pixel, $l \le 1500$) and high-resolution mode (0.65"/pixel, $l \le 4500$). Simultaneously measures velocity, intensity, and line-of-sight magnetic field. Continuous telemetry 5 kb/s, plus 160 kb/s in high-rate mode. The "workhorse" instrument producing full-disk Dopplergrams every minute. PI: P. Scherrer (Stanford).

Fig. 5는 세 helioseismology 기기의 공간 분해능 비교를 보여준다: GOLF은 $l = 0$–$3$, VIRGO는 $l \le 7$, MDI는 $l \le 4500$까지 커버하여, 세 기기가 결합하면 태양 내부 전 영역을 진단할 수 있다.

Fig. 5 compares the spatial coverage of the three helioseismology instruments: GOLF covers $l = 0$–$3$, VIRGO extends to $l \le 7$, and MDI reaches $l \le 4500$, so combined they diagnose the entire solar interior.

#### 2.2 Solar Atmosphere Remote Sensing 기기 / Solar Atmosphere Remote Sensing Instruments

**SUMER (Solar Ultraviolet Measurements of Emitted Radiation)**: 수직 입사(normal incidence) UV 분광기. 파장 범위 500–1600 Å, 분광 분해능 $\lambda/\Delta\lambda = 18{,}800$–$40{,}000$, 공간 분해능 1.5", 시간 분해능 ~10 s. 전이 영역과 코로나 플라즈마의 온도, 밀도, 속도, 화학 조성을 진단한다. PI: K. Wilhelm (MPAe). 비트율: 10.5 kb/s.

**SUMER (Solar Ultraviolet Measurements of Emitted Radiation)**: Normal-incidence UV spectrometer. Wavelength range 500–1600 Å, spectral resolution $\lambda/\Delta\lambda = 18{,}800$–$40{,}000$, spatial resolution 1.5", time resolution ~10 s. Diagnoses temperature, density, velocity, and chemical composition of transition region and coronal plasma. PI: K. Wilhelm (MPAe). Bit rate: 10.5 kb/s.

**CDS (Coronal Diagnostic Spectrometer)**: Wolter II형 사입사(grazing incidence)와 수직 입사 EUV 분광기의 조합. 파장 범위 150–800 Å, 공간 분해능 ~3", 분광 분해능 $\lambda/\Delta\lambda = 2{,}000$–$10{,}000$. 코로나와 전이 영역의 다온도 플라즈마 진단에 특화. PI: R. Harrison (RAL). 비트율: 12 kb/s.

**CDS (Coronal Diagnostic Spectrometer)**: Combination of Wolter II grazing-incidence and normal-incidence EUV spectrometers. Wavelength range 150–800 Å, spatial resolution ~3", spectral resolution $\lambda/\Delta\lambda = 2{,}000$–$10{,}000$. Specialized in multi-temperature plasma diagnostics of the corona and transition region. PI: R. Harrison (RAL). Bit rate: 12 kb/s.

**EIT (Extreme Ultraviolet Imaging Telescope)**: 전 디스크 EUV 이미저. 4개 밴드: Fe IX 171 Å ($T \approx 1.0$ MK), Fe XII 195 Å ($T \approx 1.6$ MK), Fe XV 284 Å ($T \approx 2.0$ MK), He II 304 Å ($T \approx 0.08$ MK). 다층 코팅(multilayer coating) 기술로 특정 파장만 선택적으로 반사한다. 1024×1024 CCD, 2.6"/pixel, 전 디스크 시야각 45'. 이후의 모든 EUV 이미저(STEREO/EUVI, SDO/AIA)의 원형이 된 기기이다. PI: J.-P. Delaboudinière (IAS). 비트율: 1 kb/s (26.2 kb/s 고속).

**EIT (Extreme Ultraviolet Imaging Telescope)**: Full-disk EUV imager with 4 bands: Fe IX 171 Å ($T \approx 1.0$ MK), Fe XII 195 Å ($T \approx 1.6$ MK), Fe XV 284 Å ($T \approx 2.0$ MK), He II 304 Å ($T \approx 0.08$ MK). Uses multilayer coating technology to selectively reflect specific wavelengths. 1024×1024 CCD, 2.6"/pixel, full-disk FOV of 45'. Became the template for all subsequent EUV imagers (STEREO/EUVI, SDO/AIA). PI: J.-P. Delaboudinière (IAS). Bit rate: 1 kb/s (26.2 kb/s high rate).

**UVCS (Ultraviolet Coronagraph Spectrometer)**: 차폐식 UV 코로나그래프 분광기. 관측 범위 1.3–10 $R_\odot$. Ly-$\alpha$ (1216 Å)와 O VI (1032, 1037 Å) 선 프로파일을 측정하여 코로나의 속도 분포, 온도, 밀도를 진단한다. 특히 코로나 가열과 태양풍 가속 영역을 직접 탐사한다. PI: J. Kohl (SAO). 비트율: 5 kb/s.

**UVCS (Ultraviolet Coronagraph Spectrometer)**: Occulted UV coronagraph spectrometer. Observing range 1.3–10 $R_\odot$. Measures Ly-$\alpha$ (1216 Å) and O VI (1032, 1037 Å) line profiles to diagnose the velocity distribution, temperature, and density of the corona. Directly probes the coronal heating and solar wind acceleration regions. PI: J. Kohl (SAO). Bit rate: 5 kb/s.

**LASCO (Large Angle Spectroscopic Coronagraph)**: 삼중 코로나그래프 시스템:
- **C1** (1.1–3 $R_\odot$): 내부 차폐 코로나그래프, Fabry-Perot 간섭계를 이용한 분광 관측 가능 (~700 mÅ 분해능). Fe XIV 5303 Å 등의 코로나 방출선 관측.
- **C2** (1.5–6 $R_\odot$): 외부 차폐 코로나그래프, 백색광.
- **C3** (3–30 $R_\odot$): 외부 차폐 코로나그래프, 백색광. 최대 시야각.
- 모두 1024×1024 CCD. 이 세 기기를 결합하면 태양 표면에서 30 태양 반경까지의 코로나를 연속적으로 관측할 수 있다.
- PI: G. Brueckner (NRL). 비트율: 4.2 kb/s (26.2 kb/s 고속).

**LASCO (Large Angle Spectroscopic Coronagraph)**: Triple coronagraph system:
- **C1** (1.1–3 $R_\odot$): Internally occulted coronagraph with Fabry-Perot spectroscopic capability (~700 mÅ resolution). Observes coronal emission lines such as Fe XIV 5303 Å.
- **C2** (1.5–6 $R_\odot$): Externally occulted white-light coronagraph.
- **C3** (3–30 $R_\odot$): Externally occulted white-light coronagraph with maximum FOV.
- All use 1024×1024 CCDs. Combined, the three instruments provide continuous coronal coverage from the solar surface out to 30 solar radii.
- PI: G. Brueckner (NRL). Bit rate: 4.2 kb/s (26.2 kb/s high rate).

**SWAN (Solar Wind ANisotropies)**: 행성간 매질의 Ly-$\alpha$ (1216 Å) 후방 산란 복사를 전 천구 1° 분해능으로 매핑한다. 태양풍의 대규모 비등방성과 질량 플럭스를 진단하며, 코로나 홀의 위치 및 크기를 추적한다. PI: J.-L. Bertaux (SA, Verrières). 비트율: 0.2 kb/s.

**SWAN (Solar Wind ANisotropies)**: Maps the Ly-$\alpha$ (1216 Å) backscattered radiation of the interplanetary medium over the full sky at 1° resolution. Diagnoses large-scale solar wind anisotropies and mass flux, tracking the location and extent of coronal holes. PI: J.-L. Bertaux (SA, Verrières). Bit rate: 0.2 kb/s.

Fig. 6은 SUMER, CDS, EIT, UVCS의 파장-온도 커버리지를 비교한다. 네 기기가 결합하면 $10^4$–$10^7$ K 범위의 태양 대기 전 온도 영역을 진단할 수 있다.

Fig. 6 compares the wavelength-temperature coverage of SUMER, CDS, EIT, and UVCS. Combined, the four instruments diagnose the entire temperature range of $10^4$–$10^7$ K in the solar atmosphere.

#### 2.3 Solar Wind In-Situ 기기 / Solar Wind In-Situ Instruments

**CELIAS (Charge, ELement, and Isotope Analysis System)**: 세 가지 time-of-flight 센서로 구성: CTOF(질량/전하 비), MTOF(질량/전하 고분해능), STOF(초열 이온). 에너지 범위 0.1–1000 keV/e. SEM(Solar EUV Monitor)이 He II 304 Å 플럭스를 모니터한다. PI: D. Hovestadt/P. Bochsler. 비트율: 1.5 kb/s.

**CELIAS (Charge, ELement, and Isotope Analysis System)**: Comprises three time-of-flight sensors: CTOF (mass/charge ratio), MTOF (high-resolution mass/charge), STOF (suprathermal ions). Energy range 0.1–1000 keV/e. The SEM (Solar EUV Monitor) monitors the He II 304 Å flux. PI: D. Hovestadt/P. Bochsler. Bit rate: 1.5 kb/s.

**COSTEP (Comprehensive Suprathermal and Energetic Particle Analyzer)**: 두 센서로 구성: LION(저에너지 이온/전자)과 EPHIN(고에너지 양성자/헬륨 핵/전자). 전자 0.04–5 MeV, 양성자/헬륨 0.04–53 MeV/n. 태양 에너지 입자(SEP) 이벤트를 실시간으로 감시한다. PI: H. Kunow (Kiel). 비트율: 0.3 kb/s.

**COSTEP (Comprehensive Suprathermal and Energetic Particle Analyzer)**: Two sensors: LION (low-energy ions/electrons) and EPHIN (high-energy protons/helium nuclei/electrons). Electrons 0.04–5 MeV, protons/helium 0.04–53 MeV/n. Monitors solar energetic particle (SEP) events in real time. PI: H. Kunow (Kiel). Bit rate: 0.3 kb/s.

**ERNE (Energetic and Relativistic Nuclei and Electron Experiment)**: LED(Low Energy Detector)와 HED(High Energy Detector)로 구성. 원소 $Z = 1$–$30$, 에너지 1.4–540 MeV/n, 전자 5–60 MeV. COSTEP보다 높은 에너지 범위를 커버하며, 중원소(heavy elements)의 조성 분석이 가능하다. PI: J. Torsti (Turku). 비트율: 0.7 kb/s.

**ERNE (Energetic and Relativistic Nuclei and Electron Experiment)**: Comprises LED (Low Energy Detector) and HED (High Energy Detector). Elements $Z = 1$–$30$, energy 1.4–540 MeV/n, electrons 5–60 MeV. Covers a higher energy range than COSTEP and enables compositional analysis of heavy elements. PI: J. Torsti (Turku). Bit rate: 0.7 kb/s.

Fig. 7은 CELIAS, COSTEP, ERNE 세 기기의 전하-에너지 범위를 비교하며, 세 기기가 결합하여 수 eV에서 수백 MeV까지의 입자 에너지 스펙트럼을 완전히 커버함을 보여준다.

Fig. 7 compares the charge-energy ranges of CELIAS, COSTEP, and ERNE, showing that the three instruments combined fully cover the particle energy spectrum from a few eV to hundreds of MeV.

### §3 Spacecraft (pp. 16–26)

**구조 및 질량**: SOHO 우주선은 모듈식 설계로, Service Module (SVM)과 Payload Module (PLM)으로 구성된다. 전체 크기 $4.3 \times 2.7 \times 3.7$ m³, 총 질량 1861 kg (탑재체 655 kg, 추진제 250 kg). SVM은 버스 시스템(전력, 자세 제어, 통신)을, PLM은 12개 과학 기기를 수용한다.

**Structure and mass**: The SOHO spacecraft has a modular design comprising a Service Module (SVM) and Payload Module (PLM). Overall dimensions $4.3 \times 2.7 \times 3.7$ m³, total mass 1861 kg (payload 655 kg, propellants 250 kg). The SVM houses the bus systems (power, attitude control, communications), while the PLM accommodates the 12 science instruments.

**전력 시스템**: 태양 전지판에서 1400 W 발생 (28 V 버스 전압). 2×20 Ah NiCd 배터리로 비상 시 백업 전력 제공. L1 궤도에서는 지구 차폐가 없으므로 배터리 사용은 최소화된다.

**Power system**: Solar panels generate 1400 W (28 V bus voltage). 2×20 Ah NiCd batteries provide backup power for contingencies. Since there is no Earth occultation at L1, battery usage is minimized.

**데이터 처리 및 저장**: CDMU(Command and Data Management Unit)는 MAS281 16비트 프로세서를 사용한다. 데이터 저장은 2 Gbit SSR(Solid-State Recorder) + 1 Gbit 테이프 레코더로 구성되어, DSN 패스 사이에 데이터를 버퍼링한다.

**Data processing and storage**: The CDMU (Command and Data Management Unit) uses an MAS281 16-bit processor. Data storage comprises a 2 Gbit SSR (Solid-State Recorder) + 1 Gbit tape recorder, buffering data between DSN passes.

**텔레메트리**: S-band (하향 2245 MHz / 상향 2067 MHz) 사용. 세 가지 비트율: 245.8 kb/s (고속), 54.6 kb/s (중속), 1.4 kb/s (저속). DSN 26 m 안테나가 주 수신, 34 m 안테나가 고속 모드용으로 사용된다. 모든 과학 데이터의 연속 텔레메트리 합계는 약 40 kb/s이다. 이는 1990년대 기준으로 혁신적이었지만, 오늘날의 기준으로는 극히 제한적이어서 창의적인 데이터 압축이 필수적이었다.

**Telemetry**: Uses S-band (downlink 2245 MHz / uplink 2067 MHz). Three bit rates: 245.8 kb/s (high), 54.6 kb/s (medium), 1.4 kb/s (low). DSN 26 m antennas serve as primary receivers; 34 m antennas are used for high-rate mode. The total continuous science telemetry is approximately 40 kb/s. While revolutionary by 1990s standards, this is extremely limited by today's measures, making creative data compression essential.

**자세 제어(Pointing)**: 3축 안정화(3-axis stabilized). 절대 포인팅 정밀도 <5' (태양 방향 $X_{pi}$ 축), 중기 안정도 <10" (6개월), 단기 안정도 <1" (15분), 롤 안정도 <1.5' (15분). 이 수준의 포인팅 안정도는 고분해능 이미징과 분광에 필수적이다.

**Attitude control (Pointing)**: 3-axis stabilized. Absolute pointing accuracy <5' (Sun-pointing $X_{pi}$ axis), medium-term stability <10" over 6 months, short-term stability <1" over 15 minutes, roll stability <1.5' over 15 minutes. This level of pointing stability is essential for high-resolution imaging and spectroscopy.

**온보드 시간**: OBT(On-Board Time) 정밀도 ±20 ms 대비 TAI(International Atomic Time). Helioseismology 기기들의 시계열 분석에 이 정밀도가 필수적이다.

**On-board time**: OBT precision ±20 ms relative to TAI (International Atomic Time). This precision is essential for time-series analysis of helioseismology instruments.

Table III는 우주선의 주요 기술 사양을 종합한 참조 테이블이다.

Table III is the reference table summarizing the spacecraft's key technical specifications.

### §4 Orbit, Operations, and Ground Segment (pp. 26–34)

**발사 및 궤도 투입**: Atlas II-AS 발사체를 사용하여 LEO 주차 궤도에 진입한 후, 전이 궤도를 거쳐 약 4개월 만에 L1 헤일로 궤도에 투입된다.

**Launch and orbit insertion**: Using an Atlas II-AS launch vehicle, the spacecraft enters a LEO parking orbit, then transfers to the L1 halo orbit over approximately 4 months.

**L1 헤일로 궤도**: 태양-지구 L1 라그랑주점은 지구에서 약 $1.5 \times 10^6$ km 떨어져 있다. SOHO의 헤일로 궤도 반지름은 $x$ (지구-태양 방향) ~200,000 km, $y$ (황도면 수직) ~650,000 km, $z$ (황도면 밖) ~200,000 km이며, 궤도 주기는 약 180일이다. 이 궤도는 불안정하여 주기적인 궤도 유지 기동(station-keeping maneuvers)이 필요하다.

**L1 halo orbit**: The Sun-Earth L1 Lagrangian point is approximately $1.5 \times 10^6$ km from Earth. SOHO's halo orbit semi-diameters are ~200,000 km in $x$ (Earth-Sun direction), ~650,000 km in $y$ (perpendicular in ecliptic plane), ~200,000 km in $z$ (out of ecliptic), with an orbital period of approximately 180 days. This orbit is unstable and requires periodic station-keeping maneuvers.

**L1 궤도의 이점**:
1. **연속 관측**: 지구 차폐 없이 태양을 24시간 관측 가능 (~100% duty cycle). 지상 관측소의 ~80%와 대비.
2. **매끄러운 속도 변화**: L1에서의 시선 속도 변화(Fig. 13)는 지구 궤도보다 훨씬 부드러워, helioseismology 데이터의 시계열 분석에 유리.
3. **지구 자기권 밖**: 태양풍 in-situ 측정이 교란 없이 가능.
4. **상류 위치**: 태양풍이 지구에 도달하기 약 1시간 전에 측정하여, 우주 날씨 조기 경보 역할.

**Advantages of L1 orbit**:
1. **Continuous observation**: 24-hour solar viewing with no Earth occultation (~100% duty cycle), compared to ~80% for ground-based observatories.
2. **Smooth velocity changes**: The line-of-sight velocity variation at L1 (Fig. 13) is much smoother than in Earth orbit, favorable for helioseismology time-series analysis.
3. **Outside magnetosphere**: Solar wind in-situ measurements can be made without magnetospheric disturbance.
4. **Upstream position**: Measures solar wind approximately 1 hour before it reaches Earth, providing early warning for space weather.

**지상 세그먼트**: DSN(Deep Space Network)은 하루에 3회 짧은 패스(1.6시간)와 1회 긴 패스(8시간)를 제공하여, 데이터 연속성 >96%를 달성한다. EOF(Experiment Operations Facility)는 GSFC Building 3에 3200 sq ft로 설치되며, EAF(Experiment Analysis Facility)는 Building 26에 위치한다.

**Ground segment**: The DSN (Deep Space Network) provides 3 short passes (1.6 hours) and 1 long pass (8 hours) per day, achieving data continuity >96%. The EOF (Experiment Operations Facility) is located in GSFC Building 3 (3200 sq ft), and the EAF (Experiment Analysis Facility) in Building 26.

**기기 간 플래그 시스템(Inter-Instrument Flag System)**: 과도 이벤트(플레어, CME 등) 발생 시 기기 간 자율적 조율 메커니즘이다. 한 기기가 이벤트를 감지하면 플래그를 올리고, 다른 기기가 이에 반응하여 관측 모드를 전환한다. 이는 제한된 텔레메트리 대역폭 내에서 과도 현상을 효율적으로 포착하기 위한 혁신적 설계이다.

**Inter-Instrument Flag System**: An autonomous coordination mechanism between instruments for transient events (flares, CMEs, etc.). When one instrument detects an event, it raises a flag, and other instruments respond by switching observation modes. This was an innovative design for efficiently capturing transient phenomena within limited telemetry bandwidth.

**SOHO Data System**: 모든 과학 데이터는 FITS 형식으로 아카이브되며, 표준화된 데이터 시스템을 통해 커뮤니티에 공개된다.

**SOHO Data System**: All science data are archived in FITS format and made available to the community through a standardized data system.

### §5 SOHO and the Community (pp. 33–35)

**개방형 데이터 정책**: SOHO는 발사 초기부터 개방형 데이터 정책을 채택하여, 모든 과학 데이터를 커뮤니티에 공개한다. 이는 당시 우주 미션에서는 선구적인 정책이었다.

**Open data policy**: SOHO adopted an open data policy from the outset, making all science data available to the community. This was a pioneering policy for space missions at the time.

**Guest Investigator (GI) 프로그램**: 운영 첫해부터 GI 프로그램을 통해 외부 연구자들이 SOHO 기기의 관측 시간을 신청하고 데이터를 이용할 수 있게 하였다.

**Guest Investigator (GI) programme**: From the first year of operations, the GI programme allowed external researchers to request observation time on SOHO instruments and use the data.

**ISTP 프로그램과의 연계**: SOHO는 ISTP(International Solar-Terrestrial Physics) 프로그램의 일부로, Cluster(ESA), Geotail(일본/NASA), Wind(NASA), Polar(NASA) 등의 미션과 조율된 태양-지구 환경 연구를 수행한다. SOHO가 태양 측 입력을 제공하고, 나머지 미션들이 태양풍의 전파와 지구 자기권 반응을 측정하는 상보적 구조이다.

**Coordination with the ISTP programme**: SOHO is part of the ISTP (International Solar-Terrestrial Physics) programme, conducting coordinated Sun-Earth environment research with missions including Cluster (ESA), Geotail (Japan/NASA), Wind (NASA), and Polar (NASA). SOHO provides the solar-side input while the other missions measure solar wind propagation and magnetospheric response.

**지상 관측소와의 연계**: GONG(Global Oscillation Network Group)과 같은 지상 helioseismology 네트워크와 상보적 관측을 수행하며, 특히 MDI와 GONG의 교차 보정(cross-calibration)이 중요하다.

**Coordination with ground observatories**: Complementary observations are conducted with ground-based helioseismology networks such as GONG, with MDI-GONG cross-calibration being particularly important.

### §6 Summary (p. 35)

논문은 SOHO가 태양의 심부 핵에서부터 코로나와 태양풍을 거쳐 태양권까지를 전례 없는 분광·공간·시간 분해능으로 연속 관측하는 최초의 포괄적 우주 태양 관측소가 될 것임을 요약한다. 세 기기 그룹의 시너지와 L1 궤도의 이점이 결합하여, 지상 관측만으로는 불가능했던 태양 연구의 새로운 시대를 열 것이라고 전망한다.

The paper concludes that SOHO will be the first comprehensive space-based solar observatory providing continuous coverage from the deep solar core through the corona and solar wind to the heliosphere, with unprecedented spectral, spatial, and temporal resolution. The synergy of the three instrument groups combined with the advantages of the L1 orbit will open a new era in solar research that was impossible with ground-based observations alone.

---

## 3. Key Takeaways / 핵심 시사점

1. **최초의 포괄적 우주 태양 관측소 / First comprehensive space-based solar observatory**: SOHO는 태양 내부(helioseismology)–대기(원격 탐사)–태양풍(현장 측정)을 단일 플랫폼에서 동시에 관측하는 최초의 미션이다. 이전의 Skylab이나 SMM은 특정 영역에 집중했지만, SOHO는 태양을 하나의 연결된 시스템으로 연구하는 패러다임을 확립했다. / SOHO is the first mission to simultaneously observe the solar interior (helioseismology), atmosphere (remote sensing), and solar wind (in-situ measurements) from a single platform. While previous missions like Skylab and SMM focused on specific domains, SOHO established the paradigm of studying the Sun as a coupled system.

2. **L1 헤일로 궤도의 혁신성 / Revolutionary L1 halo orbit**: L1 궤도는 ~100% 관측 의무 주기(duty cycle)를 제공하여, 지상 관측소의 ~80%(낮/밤, 날씨)나 LEO 관측소의 ~60%(지구 차폐)를 크게 능가한다. 특히 helioseismology에서 연속 시계열이 필수적이므로, 이 선택은 과학적으로 결정적이었다. / The L1 orbit provides ~100% observational duty cycle, far exceeding the ~80% of ground observatories (day/night, weather) and ~60% of LEO observatories (Earth occultation). Continuous time series are essential for helioseismology, making this orbit choice scientifically decisive.

3. **세 기기 그룹의 시너지 / Synergy of three instrument groups**: Helioseismology(GOLF+VIRGO+MDI), remote sensing(SUMER+CDS+EIT+UVCS+LASCO+SWAN), in-situ(CELIAS+COSTEP+ERNE) 세 그룹은 각각 독립적으로도 가치 있지만, 결합할 때 태양의 에너지 생성(핵)→전달(대류대)→방출(코로나)→전파(태양풍) 전 과정을 추적할 수 있다. / The three groups—helioseismology, remote sensing, and in-situ—are each valuable independently, but when combined they can trace the entire chain from energy generation (core) through transport (convection zone) to emission (corona) and propagation (solar wind).

4. **MDI: helioseismology의 워크호스 / MDI: the helioseismology workhorse**: MDI는 매분 전 디스크 도플러그램을 생산하여 $l \le 4500$까지의 p-mode를 분해하며, 동시에 자기장을 측정한다. 이 기기 하나가 SOHO 총 텔레메트리의 상당 부분(5 kb/s 연속, 160 kb/s 고속)을 점유할 만큼 핵심적이었다. / MDI produces full-disk Dopplergrams every minute resolving p-modes up to $l \le 4500$ while simultaneously measuring magnetic fields. This single instrument was so central that it consumed a major fraction of SOHO's total telemetry (5 kb/s continuous, 160 kb/s high rate).

5. **LASCO의 전례 없는 코로나 커버리지 / LASCO's unprecedented coronal coverage**: C1+C2+C3 삼중 코로나그래프는 1.1–30 $R_\odot$를 연속적으로 관측하여, CME의 발생과 전파를 처음으로 일상적으로 추적할 수 있게 했다. LASCO C2/C3 영상은 30년간 가장 널리 사용된 코로나 데이터가 되었다. / The C1+C2+C3 triple coronagraph continuously covers 1.1–30 $R_\odot$, enabling routine tracking of CME initiation and propagation for the first time. LASCO C2/C3 images became the most widely used coronal data over 30 years.

6. **EIT: 이후 모든 EUV 이미저의 원형 / EIT: template for all subsequent EUV imagers**: EIT의 4밴드 다층 코팅 EUV 이미징 설계는 STEREO/EUVI, SDO/AIA 등 모든 후속 태양 EUV 이미저의 직접적 원형이 되었다. 특히 171 Å, 195 Å, 284 Å, 304 Å의 4개 온도 채널 조합은 코로나 다온도 진단의 표준이 되었다. / EIT's 4-band multilayer-coated EUV imaging design became the direct template for all subsequent solar EUV imagers (STEREO/EUVI, SDO/AIA). The combination of four temperature channels (171, 195, 284, 304 Å) became the standard for coronal multi-thermal diagnostics.

7. **40 kb/s 텔레메트리의 혁신과 제약 / 40 kb/s telemetry: innovation and constraint**: 1995년 기준 40 kb/s 연속 과학 텔레메트리는 혁신적이었지만, 12개 기기가 공유하기에는 매우 제한적이었다. 이 제약이 창의적인 데이터 압축, 온보드 처리, 기기 간 플래그 시스템 등의 혁신을 촉진했다. SDO(2010)의 ~150 Mb/s와 비교하면 약 4000배 차이이다. / By 1995 standards, 40 kb/s continuous science telemetry was revolutionary, but it was very limited for 12 instruments to share. This constraint drove innovations in data compression, on-board processing, and the inter-instrument flag system. Compared to SDO's (2010) ~150 Mb/s, this is a ~4000× difference.

8. **기기 간 자율 조율 메커니즘 / Autonomous inter-instrument coordination**: 플래그 시스템(inter-instrument flag system)은 한 기기가 과도 이벤트를 감지하면 다른 기기에 알려 관측 모드를 전환하는 자율적 조율 메커니즘이다. 제한된 텔레메트리 내에서 과도 현상을 효율적으로 포착하기 위한 혁신적 설계로, 이후 SDO, Solar Orbiter 등의 미션에서도 유사한 개념이 채택되었다. / The inter-instrument flag system is an autonomous mechanism where one instrument detecting a transient event notifies others to switch observation modes. This innovative design for efficiently capturing transients within limited telemetry was later adopted in similar forms by SDO, Solar Orbiter, and other missions.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 L1 라그랑주점 위치 / L1 Lagrangian Point Location

L1 점의 지구로부터의 거리는 다음 근사식으로 주어진다:

The distance of the L1 point from Earth is given by the approximation:

$$r_{L1} \approx R \left( \frac{M_\oplus}{3 M_\odot} \right)^{1/3}$$

여기서 / where:
- $R \approx 1 \text{ AU} = 1.496 \times 10^8$ km: 지구-태양 거리 / Earth-Sun distance
- $M_\oplus = 5.97 \times 10^{24}$ kg: 지구 질량 / Earth mass
- $M_\odot = 1.99 \times 10^{30}$ kg: 태양 질량 / Solar mass

대입하면 / Substituting:

$$r_{L1} \approx 1.496 \times 10^8 \left( \frac{5.97 \times 10^{24}}{3 \times 1.99 \times 10^{30}} \right)^{1/3} \approx 1.496 \times 10^8 \times (10^{-6})^{1/3} \approx 1.496 \times 10^8 \times 10^{-2} \approx 1.5 \times 10^6 \text{ km}$$

이 거리는 지구-태양 거리의 약 1%에 해당한다. / This distance corresponds to approximately 1% of the Earth-Sun distance.

### 4.2 p-mode 점근 관계 / p-mode Asymptotic Relation

태양 음향 진동(p-mode)의 주파수는 다음 점근 관계를 따른다:

The frequencies of solar acoustic oscillations (p-modes) follow the asymptotic relation:

$$\nu_{n,l} \approx \Delta\nu \left( n + \frac{l}{2} + \varepsilon \right) - D_0 \, l(l+1)$$

여기서 / where:
- $\nu_{n,l}$: 방사 차수(radial order) $n$, 구면 조화 차수(spherical harmonic degree) $l$의 모드 주파수 / mode frequency of radial order $n$ and degree $l$
- $\Delta\nu \approx 135 \, \mu\text{Hz}$: 대간격(large frequency separation), 태양 평균 밀도에 의존 / large frequency separation, depends on solar mean density
- $\varepsilon$: 위상 상수(phase constant), 표면 경계 조건에 의존 / phase constant, depends on surface boundary conditions
- $D_0$: 소간격(small separation) 계수, 핵 영역의 음속 프로파일에 민감 / small separation coefficient, sensitive to the sound speed profile in the core

대간격 $\Delta\nu$는 태양의 평균 밀도와 직접 관련된다:

The large separation $\Delta\nu$ is directly related to the solar mean density:

$$\Delta\nu = \left( 2 \int_0^R \frac{dr}{c(r)} \right)^{-1}$$

여기서 $c(r)$은 반경 $r$에서의 음속이다. / where $c(r)$ is the sound speed at radius $r$.

GOLF ($l \le 3$)은 핵을 관통하는 저차 모드를, VIRGO ($l \le 7$)은 중간 차수를, MDI ($l \le 4500$)은 대류대까지의 고차 모드를 각각 측정한다. 특히 소간격 $D_0$는 핵의 상태(나이, 수소 함량)에 민감하여, GOLF의 $l = 0$–$3$ 측정이 결정적이다.

GOLF ($l \le 3$) measures low-degree modes penetrating the core, VIRGO ($l \le 7$) intermediate degrees, and MDI ($l \le 4500$) high-degree modes probing down to the convection zone. The small separation $D_0$ is particularly sensitive to core conditions (age, hydrogen content), making GOLF's $l = 0$–$3$ measurements decisive.

### 4.3 헤일로 궤도 파라미터 / Halo Orbit Parameters

SOHO 헤일로 궤도의 세 축 반지름:

SOHO halo orbit semi-diameters along three axes:

$$A_x \approx 200{,}000 \text{ km}, \quad A_y \approx 650{,}000 \text{ km}, \quad A_z \approx 200{,}000 \text{ km}$$

궤도 주기 / Orbital period:

$$T_{\text{halo}} \approx 180 \text{ days}$$

이 궤도는 동적으로 불안정하여, 궤도 유지에 약 $\Delta v \approx 2.4$ m/s/yr의 추진력이 필요하다. SOHO에 탑재된 250 kg의 hydrazine 추진제는 수십 년의 수명을 보장한다.

This orbit is dynamically unstable, requiring approximately $\Delta v \approx 2.4$ m/s/yr for station-keeping. The 250 kg of hydrazine propellant aboard SOHO ensures a lifetime of several decades.

### 4.4 Thomson 산란 편광 밝기 / Thomson Scattering Polarized Brightness

LASCO의 백색광 코로나그래프가 측정하는 편광 밝기(pB)는 코로나 전자 밀도에 의한 Thomson 산란으로 발생한다:

The polarized brightness (pB) measured by LASCO's white-light coronagraph arises from Thomson scattering by coronal electrons:

$$pB = \frac{\pi \sigma_T}{2} \int n_e(r) \left[ (1 - u) A(r) + u B(r) \right] dr$$

여기서 / where:
- $\sigma_T = 6.65 \times 10^{-25}$ cm²: Thomson 산란 단면적 / Thomson scattering cross-section
- $n_e(r)$: 시선 방향의 전자 밀도 분포 / electron density along the line of sight
- $u$: 주변 감광 계수(limb-darkening coefficient) / limb-darkening coefficient
- $A(r), B(r)$: van de Hulst (1950)의 기하학적 인자, 관측점에서의 산란각에 의존 / geometric factors from van de Hulst (1950), depending on scattering angle at the observation point

pB 측정은 코로나 전자 밀도의 3D 재구성의 기초가 되며, LASCO C2/C3 데이터로부터 CME의 질량을 추정하는 데 사용된다.

pB measurements form the basis for 3D reconstruction of coronal electron density and are used to estimate CME mass from LASCO C2/C3 data.

### 4.5 텔레메트리 링크 예산 / Telemetry Link Budget

L1에서의 데이터 전송률은 거리와 안테나 이득에 의해 결정된다. 수신 신호 대 잡음비(SNR):

The data transmission rate from L1 is determined by distance and antenna gain. The received signal-to-noise ratio (SNR):

$$\frac{E_b}{N_0} = \frac{P_t G_t G_r \lambda^2}{(4\pi)^2 d^2 k T_s R_b}$$

여기서 / where:
- $P_t$: 송신 전력 / transmitter power
- $G_t, G_r$: 송신/수신 안테나 이득 / transmitter/receiver antenna gains
- $\lambda$: 파장 (S-band, ~13.3 cm) / wavelength
- $d \approx 1.5 \times 10^6$ km: L1 거리 / L1 distance
- $k$: Boltzmann 상수 / Boltzmann constant
- $T_s$: 시스템 잡음 온도 / system noise temperature
- $R_b$: 비트율 / bit rate

DSN 26 m 안테나(중속 54.6 kb/s)와 34 m 안테나(고속 245.8 kb/s)의 차이는 안테나 이득 $G_r \propto D^2$에 의한 것으로, $(34/26)^2 \approx 1.71$배의 이득 향상이 약 4.5배의 비트율 증가를 가능하게 한다(추가 코딩 이득 포함).

The difference between the DSN 26 m antenna (medium rate 54.6 kb/s) and 34 m antenna (high rate 245.8 kb/s) is due to antenna gain $G_r \propto D^2$, where $(34/26)^2 \approx 1.71$ gain improvement enables approximately 4.5× bit rate increase (including additional coding gains).

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1960  Leighton et al. — 5분 진동 발견 / 5-min oscillations discovered
  │
1970  Ulrich; Leibacher & Stein — 정상 음파 이론 / Standing wave interpretation
  │
1973  Skylab/ATM — 최초 우주 태양 관측 / First space solar observation
  │
1975  Deubner — p-mode 확인 / p-mode confirmation
  │
1980  SMM — 플레어·TSI 정밀 관측 / Flare & TSI precision measurement
  │
1982  SOHO 제안 / SOHO proposed
  │
1986  ESA Horizon 2000 승인 / ESA Horizon 2000 approved (STSP cornerstone)
  │
1989  ESA-NASA MOU 체결 / ESA-NASA MOU signed
  │
1991  Phase C/D 시작 / Phase C/D began
  │
1995  ★ Domingo et al. — SOHO 미션 개관 (본 논문) ★ [#8]
  │   ★ SOHO 발사 (12월 2일) / SOHO launched (Dec 2)
  │
1996  Harvey et al. — GONG 네트워크 완성 [#5]
  │   Chaplin et al. — BiSON 네트워크 [#6]
  │   Delaboudinière et al. — EIT 기기 논문 [#9]
  │   Brueckner et al. — LASCO 기기 논문 [#10]
  │   Scherrer et al. — MDI 기기 논문 [#11]
  │
1998  SOHO 교신 두절·복구 / SOHO contact loss & recovery
  │
2003  Scharmer et al. — SST [#3]
  │
2006  STEREO 발사 — 최초 쌍둥이 태양 관측 / First twin solar observatory
  │
2010  SDO 발사 — AIA [#12], HMI [#13]
  │   AIA: EIT의 후계자 (7→10 채널, 0.6" 분해능)
  │   HMI: MDI의 후계자 (4096×4096, 연속 자기장)
  │
2012  Goode & Cao — BBSO/GST [#4]
  │
2016  Tomczyk et al. — COSMO 제안 [#7]
  │
2020  Solar Orbiter 발사 — 근접 태양 관측 / Close-up solar observation
  │
2026  SOHO 운영 31년차 — 여전히 활동 중 / Still operational after 31 years
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| # | 논문 / Paper | 연결 / Connection |
|---|-------------|-------------------|
| 1 | Pierce (1964) — FTS at Kitt Peak | 지상 분광 기기의 전통. SUMER/CDS는 이 분광 기법을 우주로 확장 / Ground spectroscopic tradition. SUMER/CDS extend this to space |
| 3 | Scharmer et al. (2003) — SST | 지상 고분해능 이미징. EIT/AIA와 상보적 / Ground high-resolution imaging, complementary to EIT/AIA |
| 4 | Goode & Cao (2012) — BBSO/GST | 지상 적응광학 망원경. SOHO 우주 관측과 상보적 / Ground AO telescope, complementary to SOHO space observations |
| 5 | Harvey et al. (1996) — GONG | 지상 helioseismology 네트워크. MDI의 지상 대응 기기, 교차 보정 파트너 / Ground helioseismology network. MDI's ground counterpart and cross-calibration partner |
| 6 | Chaplin et al. (1996) — BiSON | 저차 모드($l = 0$–$3$) 전문 네트워크. GOLF의 지상 대응 기기 / Low-degree ($l = 0$–$3$) specialist network. GOLF's ground counterpart |
| 7 | Tomczyk et al. (2016) — COSMO | K-Cor 코로나그래프. LASCO와 상보적 (지상 vs 우주, 편광 vs 백색광) / K-Cor coronagraph. Complementary to LASCO (ground vs space, polarimetric vs white-light) |
| 9 | Delaboudinière et al. (1995) — EIT | SOHO 개별 기기 논문. EIT의 상세 설계, 보정, 초기 성능 / Individual SOHO instrument paper. EIT detailed design, calibration, and initial performance |
| 10 | Brueckner et al. (1995) — LASCO | SOHO 개별 기기 논문. LASCO C1/C2/C3의 상세 설계와 성능 / Individual instrument paper. LASCO C1/C2/C3 detailed design and performance |
| 11 | Scherrer et al. (1995) — MDI | SOHO 개별 기기 논문. MDI의 Michelson 간섭계 설계, 관측 모드, 데이터 파이프라인 / Individual instrument paper. MDI Michelson interferometer design, modes, data pipeline |
| 12 | Lemen et al. (2012) — AIA | SDO의 EUV 이미저. EIT의 직접적 후계자 (4→10 채널, 1.5"→0.6") / SDO EUV imager. Direct successor to EIT (4→10 channels, 1.5"→0.6") |
| 13 | Scherrer et al. (2012) — HMI | SDO의 도플러/자기장 기기. MDI의 직접적 후계자 (1024²→4096², 연속 전 디스크 벡터 자기장) / SDO Doppler/magnetograph. Direct successor to MDI (1024²→4096², continuous full-disk vector magnetic field) |

---

## 7. References / 참고문헌

- Domingo, V., Fleck, B., and Poland, A.I., "The SOHO Mission: An Overview," *Solar Physics*, Vol. 162, pp. 1–37, 1995. [DOI: 10.1007/BF00733425](https://doi.org/10.1007/BF00733425)
- Leighton, R.B., Noyes, R.W., and Simon, G.W., "Velocity Fields in the Solar Atmosphere. I. Preliminary Report," *Astrophysical Journal*, Vol. 135, pp. 474–499, 1962.
- Ulrich, R.K., "The Five-Minute Oscillations on the Solar Surface," *Astrophysical Journal*, Vol. 162, pp. 993–1002, 1970.
- Leibacher, J.W. and Stein, R.F., "A New Description of the Solar Five-Minute Oscillation," *Astrophysical Letters*, Vol. 7, pp. 191–192, 1971.
- Deubner, F.-L., "Observations of Low Wavenumber Nonradial Eigenmodes of the Sun," *Astronomy & Astrophysics*, Vol. 44, pp. 371–375, 1975.
- Delaboudinière, J.-P. et al., "EIT: Extreme-Ultraviolet Imaging Telescope for the SOHO Mission," *Solar Physics*, Vol. 162, pp. 291–312, 1995.
- Brueckner, G.E. et al., "The Large Angle Spectroscopic Coronagraph (LASCO)," *Solar Physics*, Vol. 162, pp. 357–402, 1995.
- Scherrer, P.H. et al., "The Solar Oscillations Investigation — Michelson Doppler Imager," *Solar Physics*, Vol. 162, pp. 129–188, 1995.
- Harvey, J.W. et al., "The Global Oscillation Network Group (GONG) Project," *Science*, Vol. 272, pp. 1284–1286, 1996.
- Chaplin, W.J. et al., "BiSON Performance," *Solar Physics*, Vol. 168, pp. 1–18, 1996.
- Lemen, J.R. et al., "The Atmospheric Imaging Assembly (AIA) on the Solar Dynamics Observatory (SDO)," *Solar Physics*, Vol. 275, pp. 17–40, 2012.
- Scherrer, P.H. et al., "The Helioseismic and Magnetic Imager (HMI) Investigation for the Solar Dynamics Observatory (SDO)," *Solar Physics*, Vol. 275, pp. 207–227, 2012.
- Tomczyk, S. et al., "Scientific Objectives and Capabilities of the Coronal Solar Magnetism Observatory," *Journal of Geophysical Research: Space Physics*, Vol. 121, pp. 7470–7487, 2016.
- van de Hulst, H.C., "The Electron Density of the Solar Corona," *Bulletin of the Astronomical Institutes of the Netherlands*, Vol. 11, pp. 135–150, 1950.
