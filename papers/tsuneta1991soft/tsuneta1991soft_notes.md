---
title: "The Soft X-ray Telescope for the Solar-A Mission"
authors: ["Tsuneta, S.", "Acton, L.", "Bruner, M.", "Lemen, J.", "Brown, W.", "Caravalho, R.", "Catura, R.", "Freeland, S.", "Jurcevich, B.", "Morrison, M.", "Ogawara, Y.", "Hirayama, T.", "Owens, J."]
year: 1991
journal: "Solar Physics"
doi: "10.1007/BF00151694"
topic: Solar_Observation
tags: [SXT, Yohkoh, SOLAR-A, X-ray-telescope, grazing-incidence, CCD, instrument-paper]
status: completed
date_started: 2026-04-25
date_completed: 2026-04-25
---

# 38. The Soft X-ray Telescope for the Solar-A Mission / SOLAR-A 미션의 연 X-선 망원경

---

## 1. Core Contribution / 핵심 기여

이 논문은 일본 SOLAR-A(발사 후 Yohkoh) 위성에 탑재된 Soft X-ray Telescope (SXT) 기기의 광학·검출기·필터·자료처리·운영 모드를 종합적으로 기술하는 instrument paper이다. SXT는 grazing-incidence Wolter-I 변형 광학(Nariai 1987, 1988의 hyperboloid–hyperboloid 형식)을 사용하며, 직경 230.65 mm, 유효 초점 거리 1535.6 mm, 거울 길이 4.5 cm로 짧고 광시야(±21 arcmin) 결상을 달성한다. 검출기는 텍사스 인스트루먼츠가 SXT 전용으로 제작한 1024×1024 virtual-phase CCD(VPCCD, 18.281 µm 픽셀, 2.4528 arcsec/pixel)이며, −18 °C로 냉각된다. 5종의 분석 필터(Al 1265 Å, Al/Mg/Mn 복합, Mg 2.52 µm, Al 11.6 µm, Be 119 µm)와 8.05% 투과 메시를 조합하여 동적 범위 >5×10⁹, 등온 플라즈마 온도 측정 정밀도 $\Delta\log T \approx 0.1$을 제공한다. 자동 노출 제어(AEC), 자동 관측 영역 선택/추적(ARS/ART), 패트롤 영상, square-root LUT 기반 12→8 bit 압축 등 온보드 자율 운영 시스템을 통해 32 kbps 텔레메트리 안에서 2 s 카덴스 부분 프레임 영상과 다중 필터 시퀀스를 동시에 운영한다.
This paper is the formal instrument description of the Soft X-ray Telescope (SXT) on the Japanese SOLAR-A satellite (renamed Yohkoh after launch). SXT employs a grazing-incidence Wolter-I variant optic — twin hyperboloids of revolution following Nariai (1987, 1988) — with diameter 230.65 mm, effective focal length 1535.6 mm and an unusually short 4.5 cm mirror length to achieve wide-field (±21 arcmin) flat-focal-plane imaging. The focal plane uses a 1024×1024 virtual-phase CCD (VPCCD, 18.281 µm pixels = 2.4528 arcsec/pixel) custom-built by Texas Instruments and cooled to −18 °C. Five X-ray analysis filters (Al 1265 Å, Al/Mg/Mn composite, Mg 2.52 µm, Al 11.6 µm, Be 119 µm) plus an 8.05% transmission mesh provide >5×10⁹ dynamic range and ~0.1 dex temperature precision via filter-ratio diagnostics. On-board autonomy — Automatic Exposure Control (AEC), Automatic OR Selection/Tracking (ARS/ART), patrol imaging, and a square-root 12→8 bit lookup-table compression — allows 2 s cadence partial-frame movies and multi-filter sequences within the 32 kbps high-rate telemetry budget. SXT is the first major solar X-ray imaging telescope after Skylab and defined the technical template for every subsequent solar X-ray imager (notably Hinode/XRT in 2006).

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction and Scientific Objectives / 서론 및 과학 목표 (pp. 37–39)

서두에서 저자들은 SXT가 "for the first time" 고시간·고공간 분해능의 X-선 영상을 장기간 제공하는 기기임을 강조한다. 1.54 m 초점 거리의 grazing-incidence 망원경이 0.25–4.0 keV 영역(파장 3–45 Å) X-선 영상을 1024×1024 VPCCD에 결상하며, 초점면 부근의 박막 금속 필터들로 X-선 에너지 분리(=온도 진단)를 수행한다. 동축 가시광 망원경은 X-선/가시광 정합용 영상을 제공한다 (p. 37–38).
The paper opens by stressing that SXT will, "for the first time", produce sustained high-cadence high-resolution X-ray solar imaging over a long campaign. A 1.54 m focal length grazing-incidence telescope forms images in the 0.25–4.0 keV (3–45 Å) range on a 1024×1024 VPCCD, with thin metallic filters near the focal plane providing energy/temperature discrimination. A coaxially mounted visible-light telescope provides aspect images for X-ray/optical co-alignment.

§1.1의 과학 목표 7개 항목은 (i) 고온 코로나 가스의 분포 및 자기장 위상학, (ii) X-선 방출 플라즈마의 온도·밀도, (iii) flare 에너지 deposition의 시공간 특성, (iv) 에너지 입자 및 conduction front 수송, (v) spray·filament eruption·coronal transient 동반 파동/자기 disturbance, (vi) 에너지 해방·입자 가속의 위치, 그리고 (vii) flare 외 영역에서 코로나 자기 morphology 진화·X-선 bright point·coronal hole 연구이다. 또한 aspect telescope의 부수 기능으로 헬리오사이즈모로지 ($dI/I \approx 10^{-7}$ 검출 가능) 가능성을 언급한다 (p. 38–39).
Section 1.1 lists seven scientific objectives: (i) X-ray emitting structures and inferred coronal magnetic topology, (ii) plasma temperature and density, (iii) spatial-temporal characteristics of flare energy deposition, (iv) transport of energetic particles and conduction fronts, (v) waves/disturbances associated with sprays, filament eruptions and coronal transients, (vi) locations of energy release and particle acceleration, and (vii) non-flare studies — coronal magnetic morphology evolution, X-ray bright points, coronal holes. The aspect telescope is also flagged as a potential helioseismology channel sensitive to $dI/I \approx 10^{-7}$ given dedicated time.

§1.2는 정량적 instrument requirements를 열거한다: 동적 범위 >10⁷, 시간분해능 ≤2 s, 각분해능 ≤3 arcsec, full-disk FOV, 온도 진단 spectral capability, 가시광 동축 영상, 발사 환경 내성(20 g rms 진동), 1차 기계 공진 >100 Hz, 동작 온도 0–25 °C (p. 39).
Section 1.2 enumerates the hard requirements: dynamic range >10⁷, ≤2 s cadence, ≤3 arcsec angular resolution, full-disk FOV, spectral diagnostic capability for plasma temperature, co-aligned visible imaging, vibration tolerance 20 g rms, fundamental mechanical resonance >100 Hz, operating range 0–25 °C.

### Part II: Optics — Overall Concept (Section 2 intro, §2.1) / 광학 — 전체 개념 및 응답 (pp. 40–43)

§2의 도입부는 SXT가 개념적으로 매우 단순한 기기임을 강조한다. 센서, 셔터, 이중 필터 휠, 동축 X-선 거울 + 가시광 렌즈, 차광 도어로 구성된다(Fig. 1a, b의 분해도 p. 40). 기계·광학 설계는 Bruner et al. (1989)에 상세히 기술되어 있고, 본 논문은 과학 성능 이해에 필요한 수준만을 다룬다 (p. 41).
Section 2 emphasizes that SXT is conceptually simple: sensor, shutter, dual filter wheel, coaxial X-ray mirror plus visible lens, and a commandable door. The mechanical/optical design proper is published separately (Bruner et al. 1989); this paper treats only what is needed for scientific use.

§2.1은 SXT 종합 응답을 정리한다. Table I (p. 42)의 핵심 파라미터: 망원경 무게 14.7 kg + 전자부 9.0 kg, 평균 주광 시 전력 7–12 W, 외형 30×30×170 cm. CCD는 −18 °C 냉각, 픽셀 18.281 µm = 2.4528 arcsec, on-chip summation 2×2 및 4×4, 12 bit → 8 bit 압축, 시간분해능 2 s(통상) / 0.5 s(특수 64×64 모드). 거울은 직경 230.65 mm, 기하 면적 261.75 mm², 8 Å에서 peak effective area 78 mm², 유효 초점 거리 1535.6 mm, spectral 1% 응답 범위 3–45 Å, 8 Å에서 FWHM ≤3 arcsec, plate scale 134.2 arcsec mm⁻¹, 동적 범위 >5×10⁹. Aspect telescope는 50 mm 구경, 1538.4 mm 초점, 0.013% (3500–4500 Å) 투과율 (p. 42).
Section 2.1 summarizes SXT's overall response. Table I (p. 42) lists: telescope mass 14.7 kg + electronics 9.0 kg, daytime power 7–12 W, envelope 30×30×170 cm. The CCD is cooled to −18 °C; pixel size 18.281 µm = 2.4528 arcsec; on-chip 2×2 and 4×4 summation; 12-bit ADC compressed to 8-bit; nominal 2 s cadence (0.5 s in a special 64×64 mode). The mirror has 230.65 mm diameter, 261.75 mm² geometric area, peak effective area 78 mm² at 8 Å, effective focal length 1535.6 mm, 1%-of-peak spectral range 3–45 Å, ≤3 arcsec FWHM at 8 Å, 134.2 arcsec mm⁻¹ plate scale, and dynamic range >5×10⁹. The aspect telescope is f/31 with 50 mm aperture and 1538.4 mm focal length.

CCD는 42×42 arcmin 영역을 한 영상에 담아 단일 노출로 full disk를 커버한다. on-chip summation 모드는 1×1 (2.45″), 2×2 (4.9″), 4×4 (9.8″) 분해능을 제공한다. 이미지 데이터는 12 bit → square-root LUT으로 8 bit 압축, 지상에서 다시 12 bit로 복원된다 (p. 41). 셔터는 회전식 정속 모터로 3°와 60° 두 sector opening을 가지며, 0.077 ms 최단 노출에서 242 s 최장 노출까지 37 단계의 effective exposure를 제공한다(Table II, p. 45). 8.05% 투과 메시와 결합하여 dynamic range 확장.
The CCD subtends 42×42 arcmin so a single exposure covers the full disk. Three on-chip summation modes (1×1, 2×2, 4×4) yield 2.45″/4.9″/9.8″ pixels. Image data go through a 12→8 bit square-root LUT for downlink and are decompressed on the ground. The shutter is a rotating constant-velocity disc with 3° and 60° sector openings, giving 37 commandable effective exposure steps from 0.077 ms to 242 s (Table II), some in tandem with the 8.05% mesh to extend dynamic range.

### Part III: X-Ray Telescope Optics (§2.2) / X-선 망원경 광학 (pp. 41–43)

핵심 광학적 혁신은 두 가지이다. 첫째, **paraboloid–hyperboloid 대신 hyperboloid–hyperboloid** 광학(Nariai 1987, 1988)을 채택하여 평탄 초점면에서의 광시야 각분해능을 향상시켰다. 둘째, **거울 길이를 단지 4.5 cm**로 단축(Watanabe 1987)하여 광시야 성능을 추가로 개선하였다. 두 광학면은 단일 실린더의 저팽창 Zerodur glass-ceramic으로 가공되어 lightweight 티타늄 stress-free 마운트에 본딩된다. 반사면 코팅은 80 Å Cr 위에 420 Å Au를 증착한 형태이며, 6 Å 이상 파장에서 예측 vs. 보정된 거울 effective area는 90% 이상 일치한다 (p. 41–42).
Two optical innovations stand out. First, instead of a classical paraboloid–hyperboloid Wolter-I, SXT uses **twin hyperboloids of revolution** (Nariai 1987, 1988) for better wide-field angular resolution on a flat focal plane. Second, the **mirror is unusually short — only 4.5 cm total along the optical axis** (Watanabe 1987) — further enhancing wide-field performance. Both reflecting surfaces are formed in a single low-expansion Zerodur glass-ceramic cylinder bonded into a stress-free titanium mount. Coating is 420 Å Au on top of 80 Å Cr; predicted vs. measured effective area agree to better than 90% for wavelengths longer than 6 Å, where small-angle scattering is moderate.

산란 후광(scattering halo)은 grazing-incidence 광학의 고질적 문제이지만 SXT는 surface roughness 3.8 Å rms, mid-frequency error 51 Å rms를 달성하여 Skylab S-054 대비 산란 wing이 크게 감소했다(Fig. 4 비교, p. 46). PSF는 파장과 off-axis 각의 함수이며, 50% encircled energy 지름은 경험식
$D_{50} = 7.0 - 2.4\,\log_{10}\lambda$ (arcsec, λ in Å) (Eq. 1, p. 44)
로 표현된다. PSF의 비-Gaussian 윙은 modified Moffat (Bendinelli 1991)
$N(r) = C / [1+(r/a)^2]^b$ (Eq. 2, p. 44)
으로 잘 맞는다. CCD는 best on-axis focus보다 0.1 mm 앞으로 의도적으로 배치하여 광시야 균일도를 개선했다(Fig. 5, p. 46). 5종의 X-선 분석 필터(rear wheel)는 attenuation, dynamic range 확장, flare 온도 진단 정보를 제공하며, 4종은 스테인리스 메시에 지지된다(p. 47).
Scattering halos plague grazing-incidence optics, but SXT achieved surface roughness 3.8 Å rms and mid-frequency figure error 51 Å rms, dramatically reducing scattering wings compared with Skylab S-054 (Fig. 4 comparison). The PSF depends on wavelength and off-axis angle. The 50% encircled-energy diameter follows the empirical fit $D_{50} = 7.0 - 2.4\log_{10}\lambda$ arcsec (Eq. 1). Its non-Gaussian wings are well represented by a modified Moffat profile $N(r) = C/[1+(r/a)^2]^b$ (Eq. 2). The CCD is deliberately positioned 0.1 mm in front of best on-axis focus to give a more uniform PSF over the disk (Fig. 5). The five X-ray analysis filters (rear wheel) attenuate, extend dynamic range and discriminate flare temperatures; four are supported on stainless mesh.

### Part IV: Aspect Telescope (§2.3) / 가시광 정합 망원경 (pp. 47–48)

Aspect 채널은 자체로 50 mm 구경 / 1538 mm 초점의 소형 가시광 망원경이며 6가지 기능을 수행한다: (1) sunspot/limb 영상으로 SXT pointing을 ≤1 arcsec 정밀도로 결정, (2) magnetic plage·sunspot·pore 운동 기록, (3) white-light flare 관측, (4) 헬리오사이즈모로지, (5) CCD 이득 보정용 flat-field 조명, (6) soft X-ray 손상 회복을 위한 photon flood 광원 (p. 47).
The aspect channel is itself a 50 mm aperture, 1538 mm focal length visible telescope serving six purposes: (1) sunspot/limb images for ≤1″ pointing determination, (2) recording motion/development of magnetic plage, sunspots, and pores, (3) white-light flare imaging, (4) helioseismology, (5) flat-field illumination for CCD gain calibration, (6) blue-light photon flood that anneals soft X-ray-induced damage in the CCD oxide.

입사 필터는 흰빛 감쇠기 + bandpass 필터이며 결합 out-of-band 거부율은 10⁻⁸ 수준이다(Fig. 6, 7). 두 광학 bandpass 필터는 4308 Å (CN band, magnetic plage 강조)과 4580 Å (continuum, drift에 robust); 50 µm Airy disk 직경, ±0.5 mm depth of focus를 가진다.
The entrance combines a white-light attenuator and bandpass filter with combined out-of-band rejection ~10⁻⁸ (Figs. 6, 7). The two filter-wheel bandpass filters are 4308 Å (CN band, emphasizing magnetic plage) and 4580 Å (continuum, drift-robust); the doublet lens gives a 50 µm Airy disk and ±0.5 mm depth of focus.

### Part V: CCD Image Sensor (§2.4) / CCD 영상 센서 (pp. 48–50)

Texas Instruments Miho 공장에서 SXT 전용으로 제작된 1024×1024 virtual-phase CCD(VPCCD, 18.3 µm 픽셀, Hynecek 1979; Janesick et al. 1981). VPCCD 표면의 얇은 산화막은 thinned/back-illuminated CCD 없이도 우수한 soft-X 응답을 제공한다(Fig. 8a의 픽셀 단면, p. 50). SXT는 광자 계수가 아닌 charge-collection 모드로 운영되며, 1 s 미만 노출에서도 거의 만재 신호를 얻는다. 이로 인해 read noise (~85 e⁻ rms)와 dark current (−18 °C에서 ~9 e⁻/pixel/s)에 대한 요구사항이 비교적 느슨하다. CCD는 closed-loop 3-stage 열전 냉각기로 −18 °C로 냉각된다 (p. 49).
The 1024×1024 virtual-phase CCD (VPCCD, 18.3 µm pixels, Hynecek 1979; Janesick et al. 1981) was custom-built by Texas Instruments at their Miho, Japan facility. The VPCCD's thin oxide layer over the virtual well gives excellent soft X-ray QE without the difficulties of thinned, back-illuminated devices (pixel cross-section in Fig. 8a). SXT operates in charge-collection (not photon-counting) mode; solar features routinely produce near-full-well exposures in <1 s, so requirements on read noise (~85 e⁻ rms) and dark current (~9 e⁻ pix⁻¹ s⁻¹ at −18 °C) are modest. A closed-loop 3-stage thermoelectric cooler holds −18 °C.

CCD full well capacity is ~250 000 e⁻ ≈ 10³ × 1 keV(12.4 Å) photons. Charge transfer efficiency 0.999989 for signals >10⁴ e⁻. 양자 효율은 5–60 Å에서 ~30%, blue light (aspect)에 대해서도 ~30%; flat-field calibration에 활용된다(Fig. 8b, p. 50). 우주선·복사대 양성자에 의한 dark current 증가, dark spike 발생, oxide 이온화로 인한 charge transfer 효율 저하는 aspect 채널의 청색 photon flood (3300–4700 Å, daytime pass의 처음 4분간)로 anneal하여 보정한다 (p. 50).
Full well ≈ 250 000 e⁻, equivalent to ~10³ × 1 keV photons. Charge transfer efficiency is 0.999989 above 10⁴ e⁻. QE is ~30% over 5–60 Å and similarly ~30% for the aspect blue light, exploited for flat-field calibration (Fig. 8b). Cosmic-ray and trapped-proton induced dark spikes plus oxide ionization (which slowly destroys CTE) are mitigated by an aspect-channel blue-light photon flood (3300–4700 Å) for the first ~4 min of each daytime pass — this regenerates free electrons at the Si–SiO₂ interface and anneals damage.

### Part VI: SXT Response to the Sun (§2.5) / 태양에 대한 SXT 응답 (pp. 51–53)

§2.5는 instrument response를 atomic emission spectra와 결합하여 과학적 응답으로 변환한다. SXT는 <1 MK에서 >50 MK 범위의 플라즈마를 다양한 강도로 영상화 가능하다. Fig. 9 (p. 51)는 Mewe, Gronenschild, van den Oord (1985)의 X-선 emission line spectra와 Mewe, Lemen, van den Oord (1986)의 continuum과 Fig. 2의 instrument response를 convolution하여 얻은 6개 곡선(no filter, Al 1265, Al/Mg/Mn, Mg 2.5, Al 11.6, Be 119)이며, $EM = 10^{44}$ cm⁻³ 가정 하에 DN/sec 단위로 표시된다. Filter 비율(Fig. 10, p. 52)은 등온 플라즈마의 온도 진단을 가능하게 하며, 가장 두꺼운 두 필터(Al 11.6, Be 119)는 비-flare 온도에서는 감도가 부족하고, 얇은 필터들은 flare 시 saturate될 수 있다. 등온 가정 시 온도 결정 정밀도는 $\Delta\log T \approx 0.1$ (p. 51).
Section 2.5 turns the instrument response into a scientific response by convolving it with atomic emission. SXT is sensitive to <1 MK to >50 MK plasma over a wide range of intensities. Fig. 9 (p. 51) shows the six response curves (no filter, Al 1265, Al/Mg/Mn, Mg 2.5, Al 11.6, Be 119) obtained by convolving Mewe, Gronenschild & van den Oord (1985) line spectra and Mewe, Lemen & van den Oord (1986) continuum with the Fig. 2 instrument response, assuming $EM = 10^{44}$ cm⁻³, in DN s⁻¹. Filter ratios (Fig. 10, p. 52) allow isothermal-plasma temperature diagnostics; the two thickest filters lose sensitivity at non-flare temperatures, while the thinnest filters may saturate during flares. The isothermal temperature uncertainty is $\Delta\log T \approx 0.1$.

DEM 분석은 atomic line emissivity 분석과 비슷한 형식으로 수행 가능하지만 multi-thermal plasma의 경우 더 ill-conditioned (Craig & Brown 1986)이다. Strong et al. (1991)의 시뮬레이션(Fig. 11, p. 53)은 SXT가 high-T 플라즈마 존재와 total emission measure를 신뢰성 있게 복원하지만, DEM 분포의 세부 구조 또는 peak는 Poisson statistics 한계로 잘 복원되지 않으며 저온 끝(end)도 부정확함을 보여준다. Photon statistics가 SXT 광도측정 오차의 지배적 요소이다. CCD에서 100 e⁻ ↔ 1 DN이고 3.65 eV/e⁻이므로 1 DN ≈ 34 Å 광자 1개의 검출에 해당한다. 노출당 read+detector 노이즈는 ~1 DN. McTiernan (1991)은 ISEE-3/ICE에서 부분 occult된 두 nonthermal flare를 분석하여 SXT가 disk 위 16만 km까지 보이는 flare에 대해 500–1500 DN/pixel/s 신호를 발생시킬 것임을 보였다 (p. 52–53).
DEM analysis follows the same formalism as atomic-line emissivity analysis, but the multi-thermal case is more ill-conditioned (Craig & Brown 1986). Strong et al. (1991) simulations (Fig. 11) show SXT reliably recovers the presence of hot plasma and total emission measure, but fine DEM structure and the low-temperature end are degraded by Poisson counting statistics. Photon statistics dominate SXT photometric error: 100 e⁻ ↔ 1 DN with 3.65 eV per e⁻ means 1 DN corresponds to detecting one 34 Å photon; per-image read+detector noise ~1 DN. McTiernan (1991) analyzed two partially occulted ISEE-3/ICE non-thermal flares and predicted SXT would see them at 500–1500 DN pix⁻¹ s⁻¹ even ~160 000 km above the limb, demonstrating sensitivity to high-corona events (Table III, p. 54).

### Part VII: SXT Image Data Pipeline (§3 intro, §3.1) / SXT 영상 자료 파이프라인 (pp. 53–57)

§3은 SOLAR-A Data Processor (DP)와 SXT 마이크로프로세서가 어떻게 영상 획득·압축·텔레메트리·자율 운영을 처리하는지 다룬다. 텔레메트리 모드는 high (32 kbps), medium (4 kbps), low (1 kbps, night). 12 bit ADC는 데이터 압축을 위해 8 bit로 변환되며, 보존이 필요한 high/low order 8 bit를 선택 가능하다 (p. 54–55).
Section 3 covers how the SOLAR-A Data Processor (DP) and SXT microprocessor handle acquisition, compression, telemetry and autonomy. Telemetry rates are 32 kbps (high), 4 kbps (medium), 1 kbps (night, low). The 12-bit ADC is converted to 8 bits for telemetry, with selectable high/low order 8-bit transfers for full-precision applications.

압축은 square-root LUT으로 수행된다(Eq. 3, 4). N (원본), X (압축), M (복원):
- $N \le 64$: $X(N)=N$, $M=X=N$ (Eq. 3).
- $N > 64$: $X(N) = \mathrm{round}(59.249 + \sqrt{3510.39 - 9.50(431.14 - N)})$, $M = \mathrm{round}(0.10526 X^2 - 12.473 X + 431.14)$ for $X<255$, $M = 4085$ for $X=255$ (Eq. 4).

광자 통계와 DN의 관계 (Eq. 5):
$N = \frac{n h\nu}{3.65 c} + 11.5$ — n=검출 광자 수, $h\nu$ in eV, c=100 e⁻/DN gain, 11.5는 디지털 오프셋. 역으로 (Eq. 6, 7): $n = (N-11.5)\lambda/34$ (단위: λ in Å). 압축 오차 (Eq. 8): $\varepsilon = \sqrt{\lambda/34}\,(N-M)/\sqrt{N-11.5}$. 이 LUT 설계의 핵심은 압축 오차를 photon shot noise보다 작게 유지한다는 점이다.
Compression uses a square-root lookup table (Eqs. 3, 4) with N (raw), X (compressed), M (decompressed). Equation 5 ties DN to detected photons via gain, photon energy and offset; Eqs. 6, 7 invert it (n = (N − 11.5)λ/34, with λ in Å); Eq. 8 gives the compression error $\varepsilon = \sqrt{\lambda/34}(N-M)/\sqrt{N-11.5}$. The LUT is designed so compression error stays below photon shot noise.

§3.1은 영상 포맷을 정의한다. Full Frame Image (FFI) — 64/128/256/512 라인의 CCD 데이터, Partial Frame Image (PFI) — 64×64 sub-image (1×1 modes에서 64×64 = 2.6×2.6 arcmin), patrol image — 항상 128 라인 4×4 summed, ARS용. PFI들은 인접 16개까지 합쳐 임의 형태의 Observing Region (OR)을 구성한다(Fig. 12, p. 56). 태양 영상은 frame transfer area를 위해 의도적으로 위쪽으로 de-center된다.
Section 3.1 defines image formats: Full Frame Image (FFI) — 64/128/256/512 lines of CCD data; Partial Frame Image (PFI) — a 64×64 sub-image (2.6×2.6 arcmin in 1×1 mode); patrol image — always 128 lines, 4×4 summed, used for ARS. Up to 16 PFIs can be combined into an arbitrary-shape Observing Region (OR; Fig. 12). The solar image is deliberately de-centered upward to leave room for an unilluminated frame-transfer register at the bottom of the CCD.

### Part VIII: Experiment Control & Autonomy (§3.2) / 실험 제어 및 자율 운영 (pp. 57–60)

SXT 운영은 DP와 SXT 마이크로프로세서 양 컴퓨터의 협업으로 이루어지며, 메모리 'mailbox'를 통해 명령·상태가 교환된다. **Sequence Tables**(§3.2.1, Table IV, p. 63)는 각 노출에 대해 ROI 위치/크기, on-chip summation (1×1, 2×2, 4×4), 노출 시간, 필터를 지정하며 nested do-loop 구조이다. DP에는 8개 sequence table (FFI 4 + PFI 4)이 저장되어 있고, 4가지 SOLAR-A 과학 모드(QT/HIGH, QT/MED, FL/HIGH, FL/MED)가 entry table로 이 중 6개를 선택한다 (p. 58–59).
SXT operations are split between the DP and the SXT microprocessor, communicating through a memory 'mailbox'. **Sequence tables** (§3.2.1, Table IV) specify, for every numbered exposure, the ROI, on-chip summation, exposure duration and filter, organized as nested do-loops. The DP holds 8 sequence tables (4 FFI + 4 PFI); an 'entry table' picks 6 of these for the four science modes (QT/HIGH, QT/MED, FL/HIGH, FL/MED).

**Automatic OR Selection (ARS)** (§3.2.2)는 patrol image (42×21 arcmin, 10 arcsec 해상도)를 사용하여 4개 가장 밝은 X-선 source 위치를 결정한다. QT 모드는 search/tracking 두 알고리즘, FL 모드는 별도 알고리즘 1개를 가진다. 9개 OR target 레지스터 중 5개는 ARS로, 4개는 ground 명령으로 채워진다. 9번째 레지스터는 가장 밝은 영역으로 redundant하게 채워져 flare flag 응답 시 시작점으로 활용되며, 4 s 안에 flare OR 관측을 시작할 수 있다 (p. 59).
**Automatic OR Selection (ARS)** (§3.2.2) uses patrol images (42×21 arcmin, 10″ resolution) to find the four brightest X-ray sources. QT mode has two algorithms (search/tracking), FL mode has one. Nine OR target registers — five filled by ARS, four by ground command. The ninth register is redundantly filled with the brightest QT feature so it can serve as starting point for flare-mode searches; an SXT flare response can begin within 4 s.

**Automatic OR Tracking (ART)** (§3.2.3)는 fine Sun sensor 데이터로 spacecraft attitude drift를 보정하여 OR을 같은 태양 위치에 고정한다. 망원경을 회전시키지 않고 다른 CCD 영역을 선택하는 방식이며, high-rate에서 32 s마다 보정. **Automatic Exposure Control (AEC)** (§3.2.4)는 모든 sequence table 항목에 대해 over/under-bright 픽셀 수를 카운트하여 (~10 over / ~100 under threshold) 노출 시간 또는 필터를 자동 조절한다. 6 s 내에 다음 노출에 반영되며, sequence table에 충분한 중간 영상이 있어야 안정성이 보장된다 (p. 60).
**Automatic OR Tracking (ART)** uses fine Sun sensor data to compensate spacecraft attitude drift; the OR location on the CCD is shifted (not the telescope) every 32 s in high rate. **Automatic Exposure Control (AEC)** counts pixels above an upper threshold (~10) and below a lower threshold (~100), then adjusts exposure or selects a thicker/thinner filter; the new value is applied within 6 s. Stability requires enough intermediate images between repeats of the same sequence-table entry.

### Part IX: Image Cadence & High Time Resolution (§3.3) / 영상 카덴스 및 고시간 분해능 모드 (pp. 61–65)

§3.3은 카덴스를 결정하는 요소들을 정리한다. SOLAR-A는 QT 모드에서 SXT에 텔레메트리의 62.5%, FL 모드에서 50%를 할당한다. PFI dominant vs FFI dominant 모드에서는 PFI 또는 FFI 데이터가 4배 빠른 텔레메트리 비율로 전송된다. 자료 transfer 시간이 노출 시간을 능가하는 경우가 많아 image interval $dt$는 보통 transfer time이 결정한다 (p. 61).
Section 3.3 summarizes cadence drivers. SOLAR-A allocates 62.5% of telemetry to SXT in QT mode and 50% in FL mode. In PFI-dominant or FFI-dominant modes, the chosen image type goes out at 4× the rate of the other. Image intervals $dt$ are usually limited by transfer time, not exposure time.

Table V (p. 64)는 모든 모드 조합의 시간분해능과 최대 노출을 정리한다. 예: HIGH PFI dominant 1×1: 2.6×2.6 arcmin PFI를 2 s마다, max 0.5 s 노출. HIGH FFI dominant 1×1 FFI: 41.9×20.9 arcmin (반(半)디스크) full frame을 4.27 min마다 254.5 s max 노출. 가장 빠른 PFI cadence 2 s에서는 셔터·필터 휠 동작 시간 때문에 max 노출 0.5 s 제한. **High Time-Resolution Mode** (§3.3.3)에서는 PFI를 동서로 2 또는 4 sub-image로 분할, 각 16 또는 32 라인 ROI로 0.5 또는 1 s 카덴스의 16×64 또는 32×64 영상 동영상을 만든다. 짧은 노출과 단일 OR, 필터 alternation 불가, AEC 가능 등의 제약이 있다 (p. 64–65).
Table V tabulates time resolution and maximum exposure for all mode combinations. Examples: HIGH PFI dominant 1×1 → 2.6×2.6 arcmin PFI every 2 s with max 0.5 s exposure; HIGH FFI dominant 1×1 → 41.9×20.9 arcmin half-disk FFI every 4.27 min with max 254.5 s exposure. Even at the 2 s PFI cadence, shutter and filter-wheel motion limits exposure to ≤0.5 s. In **High Time-Resolution Mode** (§3.3.3) a PFI is split E–W into 2 or 4 sub-images of 16 or 32 lines each, yielding a 0.5 or 1 s cadence movie of 16×64 or 32×64 pixel frames. Constraints: only one OR, no filter alternation, but AEC still allowed.

### Part X: Conclusion (§4) / 결론 (p. 65)

저자들은 Skylab 망원경이 1970년대에 코로나 X-선 영상 시대를 연 의의를 회고하면서, SOLAR-A SXT가 더 단순(simpler)하고 작고 저렴한(smaller and less costly) 기기로 그 한계를 뛰어넘는 새로운 단계의 X-선 영상을 제공한다고 평한다. 자료처리 기술의 phenomenal advance 덕분에 SXT 영상은 이전보다 훨씬 빠르고 쉽게 분석된다. 저자들은 미션이 명목 수명(3년 + 5년 연장)보다 길게 운영되어 코로나 파라미터의 태양 주기 의존성을 연구할 수 있기를 희망한다. flare 관측 시 SXT, HXT, BCS, WBS의 결합이 'sum of the parts'를 능가하는 시너지를 만든다 (p. 65).
The authors retrospect on Skylab opening the era of solar X-ray imaging in the 1970s, then frame SOLAR-A SXT as a simpler, smaller, cheaper instrument that nonetheless advances beyond Skylab. Thanks to phenomenal advances in data processing technology, SXT images should be much more accessible. They hope the mission survives well beyond its nominal lifetime so that solar-cycle dependence of coronal parameters can be studied with unprecedented reliability. The combination of SXT with HXT, BCS, WBS makes for flare science that exceeds the sum of the parts.

### Part XI: Quantitative summary of Table III (Examples of SXT response, p. 54) / Table III의 정량 요약

Table III은 SXT의 dynamic range와 필터 선택 전략을 가장 효과적으로 보여주는 데이터이며, 노트북의 검증용으로도 활용된다. 모든 신호는 단일 1×1 픽셀(2.45 arcsec) 기준이다.
Table III is the most informative data table in the paper for understanding SXT's dynamic range and filter strategy; it is also the validation set for the implementation notebook. All signals are quoted per single 1×1 pixel (2.45″).

| Feature / 현상 | T (MK) | EM (cm⁻³) | Exp (s) | Open | Al 1265 | Al/Mg/Mn | Mg 2.5 | Al 11.6 | Be 119 |
|---|---|---|---|---|---|---|---|---|---|
| Coronal hole / 코로나 hole | 1.3 | 2×10⁴² | 60.4 | 18 | 10 | 4 | 1 | 0 | 0 |
| X-ray bright point / X-선 bright point | 1.8 | 7×10⁴³ | 0.948 | 34 | 23 | 11 | 6 | 0 | 0 |
| Large-scale loops / 대규모 loop | 2.1 | 6×10⁴² | 5.34 | 30 | 22 | 11 | 8 | 0 | 0 |
| Active region / 활성 영역 | 2.5 | 3×10⁴⁵ | 0.468 | sat | 2034 | 1064 | 878 | 37 | 2 |
| LDE flare / 장지속 플레어 | 7.5 | 1×10⁴⁸ | 9.6×10⁻⁴ | sat | sat | 1958 | 1515 | 201 | 74 |
| Impulsive C flare / 충격 C-급 flare | 12 | 5×10⁴⁸ | 2.3×10⁻⁴ | sat | sat | 2047 | 1449 | 286 | 180 |
| M flare / M-급 | 17 | 2×10⁴⁸ | 2.3×10⁻⁴ | sat | 938 | 690 | 464 | 101 | 92 |
| X flare / X-급 | 20 | 1×10⁴⁹ | 7.7×10⁻⁵ | 1191 | 1501 | 1101 | 723 | 162 | 163 |
| Post-flare loop / 플레어 후 loop | 7 | 3×10⁴⁶ | 0.0172 | 1886 | 1671 | 1070 | 856 | 104 | 344 |

해석 / Interpretation:
- 가장 약한 source (coronal hole, ~10⁴² EM at 1.3 MK)는 60 s 노출에서도 thinnest 필터 (Open) 18 DN, 가장 두꺼운 필터 (Be 119)는 0 DN. → Be 119는 비-flare 관측에서 사용 불가 / Coronal hole at 60 s gives only 18 DN open; Be 119 cannot be used for non-flare science.
- 가장 강한 source (X flare, ~10⁴⁹ EM at 20 MK)는 7.7×10⁻⁵ s = 77 µs 노출이 필요(즉 셔터의 가장 빠른 1 ms 모드와 8.05% mesh의 결합). Be 119에서 163 DN으로 잘 응답 / X flare needs 77 µs effective exposure (shutter's fastest mode plus 8.05% mesh) — Be 119 still gives a healthy 163 DN.
- 동일한 1 픽셀 기기로 신호가 18 DN ~ 2034 DN ~ 1191 DN(개별 모드)이며, 노출시간 차이 60.4 s ~ 7.7×10⁻⁵ s = 7.8×10⁵ 배 + 필터/메시 attenuation 결합이 5×10⁹ dynamic range의 실제 구현 / The same single pixel handles signals from 18 to 2034 DN; combined with the 7.8×10⁵ exposure-time range and filter/mesh attenuations, this realizes the >5×10⁹ dynamic range claim.

### Part XII: SXT image cadence/Telemetry tradeoffs (Table V quantified, p. 64) / 카덴스/텔레메트리 트레이드오프 정량 정리

| Mode (TLM / dom) | Pixel sum | PFI 시간분해능 (s) | PFI 최대 노출 (s) | FFI 시간분해능 (min) | FFI 최대 노출 (s) |
|---|---|---|---|---|---|
| HIGH / PFI dom. | 1×1 | 2.0 | 0.5 | 17.07 | 1022.5 |
| HIGH / PFI dom. | 2×2 | 2.0 | 0.5 | 8.53 | 510.5 |
| HIGH / PFI dom. | 4×4 | 2.0 | 0.5 | 2.13 | 126.5 |
| HIGH / FFI dom. | 1×1 | 8.0 | 6.5 | 4.27 | 254.5 |
| MED / PFI dom. | 1×1 | 16.0 | 14.5 | 136.53 | 8190.5 |
| MED / FFI dom. | 1×1 | 64.0 | 62.5 | 34.13 | 2046.5 |

핵심 통찰 / Key insight: telemetry 비율과 dominant mode 선택이 imaging cadence를 4–32배 변경한다. flare 관측에서는 HIGH/PFI dom. 1×1을 선택하여 2 s/0.5 s 카덴스를 얻고, full-disk synoptic 영상은 HIGH/FFI dom. 4×4의 2.13 min cadence로 매핑한다. 이러한 trade-off가 sequence table 설계의 본질이다.
Telemetry rate and dominant-mode choice change the imaging cadence by 4×–32×. For flares one chooses HIGH/PFI dom. 1×1 to get 2 s cadence with 0.5 s exposure; for full-disk synoptic imaging one switches to HIGH/FFI dom. 4×4 with 2.13 min cadence. Optimizing this trade-off is precisely what sequence-table programming does.

---

## 3. Key Takeaways / 핵심 시사점

1. **Twin-hyperboloid wide-field grazing-incidence design / 이중 hyperboloid 광시야 grazing-incidence 광학.** SXT는 고전 paraboloid–hyperboloid Wolter-I 대신 Nariai (1987, 1988)의 hyperboloid–hyperboloid 광학과 4.5 cm의 매우 짧은 거울을 결합하여, 2 arcsec 수준의 PSF를 광시야(±20 arcmin)에 걸쳐 평탄 초점면에서 유지한다. 이는 후속 Hinode/XRT 광학 설계의 직접적 선조이다.
SXT replaces the classic paraboloid–hyperboloid Wolter-I with the Nariai (1987, 1988) twin-hyperboloid prescription and an unusually short (4.5 cm) mirror, keeping the PSF near 2 arcsec across a wide ±20 arcmin field on a flat focal plane. This is the direct optical ancestor of Hinode/XRT.

2. **Virtual-phase CCD as the right detector for soft-X imaging / soft-X 영상에 적합한 VPCCD.** 텍사스 인스트루먼츠의 VPCCD(18.3 µm 픽셀, 1024², −18 °C)는 thinned/back-illuminated CCD의 어려움 없이 5–60 Å에서 ~30% QE를 달성한다. SXT는 photon counting이 아닌 charge collection 모드에서 1 s 미만 노출로 거의 만재 신호를 얻으므로 read noise (85 e⁻) 요구가 매우 느슨하다.
The TI virtual-phase CCD (18.3 µm pixel, 1024², −18 °C) achieves ~30% QE over 5–60 Å without thinning or back-illumination. SXT operates in charge-collection (not photon-counting) mode — solar features fill the well in <1 s — so the 85 e⁻ read noise is non-critical.

3. **Filter-ratio temperature diagnostics give 0.1 dex precision / 필터비율 온도 진단의 0.1 dex 정밀도.** 5종의 X-선 필터(Al 1265 Å, Al/Mg/Mn 복합, Mg 2.52, Al 11.6, Be 119 µm)의 응답 곡선을 신호비로 결합하여 isothermal plasma의 온도를 $\Delta\log T \approx 0.1$ 정밀도로 결정한다. 두꺼운 필터는 비-flare 온도에서 둔감하고, 얇은 필터는 flare에서 saturate된다는 trade-off가 명시된다.
Five X-ray filters (Al 1265 Å, Al/Mg/Mn composite, Mg 2.52, Al 11.6, Be 119 µm) combined as ratios give isothermal temperature to $\Delta\log T \approx 0.1$. The thickest filters are too insensitive for non-flare regions; thin filters saturate during flares — an explicit observational trade-off.

4. **Square-root LUT preserves photon statistics / square-root LUT으로 광자 통계 보존.** 12-bit → 8-bit 압축의 최대값 4085의 square-root LUT은 압축 오차 ε(Eq. 8)를 photon shot noise보다 작게 유지하도록 설계되었다. 정보 이론적으로는 Poisson 분포에서 분산-안정화(square-root) 변환을 디지털 도메인에 그대로 적용한 것이다.
The 12→8 bit square-root LUT (max 4085) is designed so the compression error ε (Eq. 8) stays below photon shot noise — the same variance-stabilizing transform that statisticians apply to Poisson data, implemented in the digital domain.

5. **Massive on-board autonomy enables 32 kbps science / 32 kbps 텔레메트리에서 과학을 가능케 하는 온보드 자율성.** ARS(자동 OR 선택), ART(자동 OR tracking), AEC(자동 노출 제어), patrol image, sequence table, flare flag 응답 4 s까지의 자율성이 결합되어 좁은 텔레메트리 안에서 다중 필터, 부분 프레임, 전(全) 프레임을 동시 운영할 수 있다.
The combination of ARS (automatic OR selection), ART (OR tracking), AEC (exposure control), patrol images, sequence tables, and a 4 s flare response is what makes simultaneous multi-filter PFI/FFI movies fit inside a 32 kbps high-rate downlink.

6. **Dynamic range >5×10⁹ via shutter + mesh / 셔터 + 메시로 동적 범위 5×10⁹ 초과.** 회전 셔터 0.077 ms–242 s (37 단계) + 8.05% 메시 + 5종 필터의 직렬 조합으로 코로나 hole(2 MK, EM 10⁴² cm⁻³)부터 X-class flare(20 MK, EM 10⁴⁹ cm⁻³)까지를 같은 기기로 영상화한다. Table III가 이 구체적 trade-off를 보여준다.
The combination of a 37-step shutter (0.077 ms to 242 s), an 8.05% transmission mesh, and five filters in series gives dynamic range >5×10⁹ — sufficient to image both coronal holes (2 MK, EM 10⁴²) and X-class flares (20 MK, EM 10⁴⁹) with the same instrument. Table III lays this out concretely.

7. **DEM analysis is feasible but ill-conditioned / DEM 분석은 가능하지만 ill-conditioned.** Strong et al. (1991) 시뮬레이션은 SXT가 high-T 플라즈마와 total emission measure를 신뢰성 있게 복원하지만, DEM 분포의 세부 구조와 저온 끝(low-T tail)은 Poisson statistics 때문에 잘 복원되지 않음을 보였다. 이는 후속 multi-filter EUV/X-ray 미션(EIS/AIA/XRT)의 DEM 분석 한계의 출발점이다.
Strong et al. (1991) simulations show SXT recovers the existence of hot plasma and total EM but loses fine DEM structure and the low-T tail to Poisson noise. This is the canonical reference point for the DEM-inversion limitations of all later multi-filter EUV/X-ray missions.

8. **Skylab → SXT → XRT lineage / Skylab → SXT → XRT 계보.** Vaiana et al. (1977)의 Skylab S-054에 비해 SXT는 PSF wing이 훨씬 작고(Fig. 4), 디지털 자료, 자율 운영이 가능하며, 결과적으로 17년 만에 X-ray solar imaging의 패러다임을 완전히 바꾸었다. 2006년 Hinode/XRT는 SXT 광학과 운영 개념을 그대로 계승·확장했다.
Compared with Skylab S-054 (Vaiana et al. 1977), SXT has dramatically reduced PSF wings (Fig. 4), digital readout, and on-board autonomy — it changed the paradigm of solar X-ray imaging in the 17 years between Skylab and Yohkoh. Hinode/XRT (2006) is its direct successor in optics and operations philosophy.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Effective area / 유효 면적

$$A_{\text{eff}}(\lambda, \theta) = A_{\text{geom}} \cdot R^2(\lambda, \theta_{\text{graze}}) \cdot T_{\text{filt}}(\lambda) \cdot Q_{\text{CCD}}(\lambda)$$

- $A_{\text{geom}} = 261.75$ mm² (Table I): 거울 입구의 기하학적 환형 면적 / Geometric annular collecting area at mirror entrance.
- $R^2$: grazing-incidence 두 번 반사이므로 단일 반사율의 제곱. λ ≳ 6 Å에서 예측 vs. 보정 90% 일치 (p. 42) / Two-bounce reflectivity squared; predicted/measured agree to 90% beyond 6 Å.
- $T_{\text{filt}}$: 두 입사 필터(Lexan + Al + Ti) + 분석 필터의 곱 / Product of dual entrance filters and analysis filter transmissions.
- $Q_{\text{CCD}}$: VPCCD QE, 5–60 Å에서 ~30%, Fig. 8b / VPCCD quantum efficiency, ~30% in 5–60 Å (Fig. 8b).

피크값: 8 Å에서 $A_{\text{eff}} \approx 78$ mm² (Table I).
Peak ~78 mm² at 8 Å.

### 4.2 PSF parameters / PSF 파라미터

#### $D_{50}$ — 50% 에너지 인사이클 직경 (Eq. 1, p. 44)

$$D_{50} = 7.0 - 2.4 \log_{10} \lambda \quad [\text{arcsec}, \, \lambda \text{ in Å}]$$

- 4 Å에서 $D_{50} \approx 7.0 - 2.4 \times 0.602 = 5.55$ arcsec.
- 10 Å에서 $D_{50} \approx 7.0 - 2.4 \times 1.0 = 4.60$ arcsec.
- 45 Å에서 $D_{50} \approx 7.0 - 2.4 \times 1.653 = 3.03$ arcsec.

장파장에서 산란 wing이 약해 $D_{50}$이 작아진다 / Longer wavelengths give tighter cores because scattering halos drop.

#### Modified Moffat (Bendinelli 1991) (Eq. 2, p. 44)

$$N(r) = \frac{C}{[1 + (r/a)^2]^b}$$

- C: 중심 첨두 진폭 (DN) / central amplitude.
- a: core scale (pixels) — 중심 첨두 폭 / core width.
- b: power-law wing 지수 / wing exponent.

Gaussian + 비-Gaussian 첨두를 동시에 표현 / Captures both quasi-Gaussian core and non-Gaussian wings.

### 4.3 Photon-to-DN conversion / 광자-DN 변환 (Eqs. 5–8, p. 55)

데이터 수 N을 검출 광자 수와 연결한다:
$$N = \frac{n \, h\nu_{[\text{eV}]}}{3.65 \, c} + 11.5 \quad (5)$$

- $n$ = 검출 광자 수 (per pixel per exposure) / detected photons per pixel.
- $h\nu_{[\text{eV}]}$ = 평균 광자 에너지 in eV (예: 8 Å = 1550 eV) / mean photon energy in eV.
- 3.65 eV/e⁻: Si의 평균 e–h 생성 에너지 / mean energy per e–h pair in Si.
- $c = 100$ e⁻/DN: CCD camera gain / camera gain.
- 11.5: 디지털 오프셋 / digital offset.

역변환:
$$n = \frac{(N - 11.5)\,\lambda_{[\text{Å}]}}{34} \quad (6), \qquad m = \frac{(M - 11.5)\,\lambda_{[\text{Å}]}}{34} \quad (7)$$

여기서 ratio 34는 다음에서 나온다: $h\nu_{[\text{eV}]} \cdot \lambda_{[\text{Å}]} = 12398.4 \approx 12400$, $12400/(3.65 \times 100) \approx 34$. 즉 1 DN ≈ 1 photon at 34 Å.
The factor 34 comes from $h\nu_{[\text{eV}]} \lambda_{[\text{Å}]} = 12398$ and $12398/(3.65 \times 100) \approx 34$, so 1 DN ≈ 1 photon at 34 Å.

압축 오차 (Eq. 8):
$$\varepsilon = \sqrt{\frac{\lambda}{34}} \cdot \frac{N - M}{\sqrt{N - 11.5}}$$

Poisson shot noise는 $\sqrt{n} = \sqrt{(N-11.5)\lambda/34}$ DN. 따라서 $\varepsilon$/shot-noise = $|N-M|/(N-11.5)$이며, square-root LUT 설계로 이 비율이 ≤1로 유지됨 / Poisson shot noise is $\sqrt{(N-11.5)\lambda/34}$ DN, so $\varepsilon$ stays below shot noise by LUT design.

### 4.4 12→8 bit square-root LUT (Eqs. 3, 4)

For $N \le 64$: $X = M = N$ (선형) / linear region.

For $N > 64$:
$$X(N) = \mathrm{round}\!\left[59.249 + \sqrt{3510.39 - 9.50(431.14 - N)}\right]$$
$$M(X) = \begin{cases} \mathrm{round}\!\left[0.10526 X^2 - 12.473 X + 431.14\right] & X < 255 \\ 4085 & X = 255 \end{cases}$$

X(N)은 단조 증가 sqrt 인코딩으로 N 범위 64–4085 (12 bit max 4096)를 X 범위 64–255에 맵핑 / The encoding is a monotone square-root mapping of N=64–4085 (≈12-bit dynamic range) into X=64–255 (8 bits).

### 4.5 Plasma response model / 플라즈마 응답 모델

SXT 신호 (DN s⁻¹ pixel⁻¹) for isothermal plasma at temperature T, emission measure EM:

$$S_f(T) = EM \cdot \int A_{\text{eff},f}(\lambda) \cdot \mathcal{P}(\lambda, T) \, d\lambda$$

- $\mathcal{P}(\lambda, T)$: Mewe et al. (1985, 1986) line + continuum emissivity per emission measure (photons s⁻¹ Å⁻¹ per cm⁻³) / per-EM emissivity.
- $A_{\text{eff},f}$: 필터 f의 effective area (cm²) / filter f's effective area.
- 단위 변환은 $1/(h\nu) \cdot (e^- \to \text{DN})$을 통해 흡수 / Unit conversion via photon→e→DN absorbed in the integral.

Filter ratio for two filters f, g:
$$R_{f/g}(T) = S_f(T) / S_g(T)$$

EM-independent (등온 가정) → T 진단. 비-등온의 경우:
$$S_f = \int_{T} A_{\text{eff},f}(\lambda) \mathcal{P}(\lambda, T) \, \xi(T) \, d\log T \, d\lambda$$

with $\xi(T) = n_e^2 \, dV/d\log T$ — DEM. 이는 Strong et al. (1991)이 Fludra & Sylwester (1986) 알고리즘으로 풀었다 / DEM inversion via Fludra & Sylwester (1986) as performed by Strong et al. (1991).

### 4.6 Worked example: Active region count rate / 작업 예: 활성 영역 카운트율

Table III, "Active region", T = 2.5 MK, EM = 3×10⁴⁵ cm⁻³, exposure 0.468 s. 단일 픽셀 신호 in DN (1×1 mode):

| Filter | Open | Al 1265 | Al/Mg/Mn | Mg 2.5 | Al 11.6 | Be 119 |
|---|---|---|---|---|---|---|
| DN | sat. | 2034 | 1064 | 878 | 37 | 2 |

해석 / Interpretation: thin (Al 1265, Al/Mg/Mn, Mg 2.5) 필터들은 활성 영역에 잘 응답하지만 (~10³ DN), thick filter (Be 119)에서는 신호 ~2 DN로 거의 검출 한계. 따라서 활성 영역 routine 관측에는 중간 두께 필터들이 적합하다 / Thin/medium filters are sensitive to active regions (~10³ DN) but Be 119 only ~2 DN — at the detection floor. Hence Be 119 is reserved for flares (X-class M flare 92–180 DN per pixel in Be 119).

### 4.7 Photon counting limit / 광자 계수 한계

1 DN = 1 photon at 34 Å (=0.36 keV). 따라서 read+detector noise ~1 DN = ~1 photon at 34 Å (or ~6 photons at 6 Å, etc.). 두꺼운 필터에서는 SXT 분광 acceptance가 좁아 photon shot noise를 더 정확히 추정 가능 (p. 52).
Read+detector noise ~1 DN equals one 34 Å photon (~6 photons at 6 Å). Thicker filters narrow the spectral acceptance, allowing tighter Poisson noise estimates.

### 4.8 Filter ratio interpretation / 필터 비율 해석

Fig. 10 (p. 52)는 4가지 ratio (b/a, c/b, 5e/d, f/2e)를 $\log T$의 함수로 보여준다 (필터 a–f는 Fig. 9의 instrument response 곡선을 가리킨다). 두 필터를 선택할 때의 운영적 고려사항:
Fig. 10 plots four signal ratios (b/a, c/b, 5e/d, f/2e) versus $\log T$ with filters a–f referring to Fig. 9. Operational considerations when choosing a pair:

- **단조성(monotonicity)**: 비율이 $\log T$에 대해 단조 증가/감소해야 isothermal T를 유일하게 결정 가능 / Ratio must be monotone in $\log T$ for unique isothermal T determination.
- **민감도(sensitivity)**: $|d \log R / d \log T|$가 클수록 photon noise 대비 T 분해능 우수 / Larger $|d \log R / d \log T|$ gives better T resolution per photon.
- **온도 적용 영역(temperature range)**: thicker filter pair는 더 높은 T 영역에서만 유효 / Thicker pairs only work at higher T.
- **노출 시간 균형(exposure balance)**: '5e/d', 'f/2e'와 같이 어느 한 필터 노출을 2배 또는 5배 늘려야 비율이 의미를 가지는 경우는 운영 부담 증가 / Pairs requiring 2× or 5× exposure (5e/d, f/2e) impose operational cost.

이러한 trade-off가 SXT 운영 sequence table의 필터 선택 알고리즘을 결정한다. 실용적 우선순위는 (i) flare 영역에서는 Be 119/Al 11.6 비율을 사용하여 hot plasma만 측정, (ii) 활성 영역과 loop은 Mg 2.5/Al 1265 또는 Al/Mg/Mn/Al 1265을 사용, (iii) coronal hole 등 저온 비-flare 영역은 thinnest filter pair에 의존하지만 photon statistics 한계가 크다는 점이다.
Practical priorities: (i) flare regions use the Be 119/Al 11.6 pair for hot plasma; (ii) active regions and loops use Mg 2.5/Al 1265 or Al/Mg/Mn/Al 1265; (iii) coronal holes and other cool non-flare features depend on the thinnest pair but suffer from limited photon statistics.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1958 ──── Wolter (1952) X-ray optics theory ──┐
1965 ──── Giacconi & Rossi: paraboloid+hyperboloid concept │
1973–74 ── Skylab S-054 (Vaiana et al. 1977) ─┐ ▼
              First sustained solar X-ray imaging (film) │
1979 ──── Hynecek: virtual-phase CCD ────────┐ │
1980 ──── SMM HXIS / BCS — strong hard X-ray, weak imaging │ │
1981 ──── Janesick, Hynecek & Blouke: VPCCD for astronomy │ │
1986 ──── Mewe, Lemen & van den Oord: continuum spectra │ │ │
1987 ──── Nariai: hyperboloid–hyperboloid wide field ──┐ │ │ │
1987 ──── Watanabe: short-mirror geometry ─────────────┤ │ │ │
1989 ──── Bruner et al.: SXT engineering description ──┤ │ │ │
1991 ★── Tsuneta et al.: SXT instrument paper ◀───────┴─┴─┴─┴───┘
1991 Aug 30 SOLAR-A launched → renamed Yohkoh
1991–2001 SXT operates, ~10 years of solar-cycle-spanning data
1998 ──── TRACE (EUV imaging successor of cadence philosophy)
2006 ──── Hinode/XRT (Golub et al.) — direct SXT optical descendant
2010 ──── SDO/AIA — EUV imaging at 12 s cadence (SXT-like autonomy)
2013 ──── IRIS — same operational philosophy in UV/EUV
```

- 별표 ★는 본 논문이며, X-선 광학 이론(Wolter), 검출기(VPCCD), 플라즈마 emission code(Mewe), 광시야 광학(Nariai/Watanabe) 4축이 1991년 SOLAR-A에서 통합되는 결정점이다 / The star marks this paper — the convergence point where Wolter X-ray optics, VPCCD detector technology, Mewe plasma codes and Nariai/Watanabe wide-field optics all came together in SOLAR-A.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Vaiana et al. 1977 (Skylab S-054, *Space Sci. Instr.* 3, 19) | Direct predecessor; SXT compares its PSF and scattering wings against S-054 in Fig. 4 / SXT가 PSF를 직접 비교하는 직계 조상 | High — defines the baseline SXT improves upon |
| Bruner et al. 1989 (1988 Yosemite Conf., AGU Monogr. 54, 187) | SXT mechanical/optical engineering paper, complementary to this scientific description / 본 논문이 과학용으로 짧게 다룬 기계·광학 설계의 상세 | High — engineering counterpart |
| Nariai 1987, 1988 (*Appl. Optics* 26, 4428; 27, 345) | Defines the hyperboloid–hyperboloid wide-field optical prescription used by SXT / SXT가 채택한 광학식의 원전 | High — optical theory ancestry |
| Watanabe 1987 (*Bull. Tokyo Astron. Obs.* 277, 3213) | Short-mirror geometry adopted to enhance SXT wide-field performance / SXT의 4.5 cm 짧은 거울 설계 기원 | High — design choice ancestry |
| Mewe, Gronenschild & van den Oord 1985; Mewe, Lemen & van den Oord 1986 | Atomic line + continuum emissivity codes used to compute SXT response in Fig. 9 / SXT 응답 곡선 계산의 emission code | High — converts instrument response into temperature response |
| Hynecek 1979; Janesick, Hynecek & Blouke 1981 | Virtual-phase CCD theory and astronomical use; basis for the SXT TI-Miho VPCCD / VPCCD 이론 및 천문 이용 | Medium — detector technology basis |
| Craig & Brown 1986 (*Inverse Problems in Astronomy*) | Establishes ill-posedness of DEM inversion that limits SXT's multi-thermal analysis (Fig. 11) / SXT DEM 분석의 ill-posed 한계 근거 | Medium — analysis limitation reference |
| Fludra & Sylwester 1986 (*Solar Phys.* 105, 323) | DEM inversion algorithm Strong et al. (1991) ran on simulated SXT data / SXT DEM 시뮬레이션에 사용된 알고리즘 | Medium — DEM analysis tool |
| Ogawara et al. 1991 (*Solar Phys.* 136, 1, this issue) | SOLAR-A mission overview accompanying this paper / 같은 호의 SOLAR-A 미션 개관 | High — companion paper |
| Morrison et al. 1991 (*Solar Phys.* 136, 105, this issue) | SOLAR-A joint analysis methodology / SOLAR-A 공동 자료 분석 방법론 | Medium — companion paper |
| Golub et al. 2007 (Hinode/XRT, *Solar Phys.* 243, 63) | Direct optical and operational successor 15 years later / 15년 뒤 SXT의 직계 후속 기기 | High — defines SXT's lineage forward in time |

---

## 7. References / 참고문헌

### Primary / 본 논문

- Tsuneta, S., Acton, L., Bruner, M., Lemen, J., Brown, W., Caravalho, R., Catura, R., Freeland, S., Jurcevich, B., Morrison, M., Ogawara, Y., Hirayama, T., and Owens, J., "The Soft X-ray Telescope for the Solar-A Mission", *Solar Physics* **136**, 37–67, 1991. DOI: [10.1007/BF00151694](https://doi.org/10.1007/BF00151694)

### Companion mission papers / 동반 미션 논문

- Ogawara, Y., Takano, T., Kato, T., Kosugi, T., Tsuneta, S., Watanabe, T., Kondo, I., and Uchida, Y., "The SOLAR-A Mission: An Overview", *Solar Physics* **136**, 1–16, 1991.
- Morrison, M. D., Lemen, J. R., Acton, L. W., Bentley, R. D., Kosugi, T., Tsuneta, S., Ogawara, Y., and Watanabe, T., *Solar Physics* **136**, 105, 1991.

### Optics & engineering / 광학 및 엔지니어링

- Bruner, M. E., Acton, L. W., Brown, W. A., Stern, R. A., Hirayama, Y., Tsuneta, S., Watanabe, T., Ogawara, Y., 1989, in *Proc. 1988 Yosemite Conf. on Outstanding Problems in Solar System Plasma Physics*, AGU Monograph 54, p. 187.
- Nariai, K., "Geometrical aberrations of a generalized Wolter type 1. 2. Analytical study", *Appl. Optics* **26**, 4428–4432, 1987.
- Nariai, K., *Appl. Optics* **27**, 345, 1988.
- Watanabe, T., *Bull. Tokyo Astron. Obs.* **277**, 3213, 1987.
- Lemen, J. R., Claflin, E. S., Brown, W. A., Bruner, M. E., Catura, R. C., Morrison, M. D., 1989, *Proc. SPIE X-Ray/EUV Optics for Astronomy and Microscopy* **1160**, 316.
- Lemen, J. R., Acton, L. W., Brown, W. A., Bruner, M. E., Catura, R. C., Strong, K. T., Watanabe, T., 1991, *Adv. Space Res.* (to be published).

### Detector / 검출기

- Hynecek, J., 1979, *IEEE IEDM Tech. Dig.*, 611.
- Janesick, J., Hynecek, J., and Blouke, M., 1981, *Proc. SPIE Solid-State Imagery for Astronomy* **290**, 165.
- Janesick, J., Klaasen, K., and Elliott, T., *Optical Engineering* **26**, 972, 1987.
- Acton, L., Morrison, M., Janesick, J., and Elliott, T., 1991, *Proc. SPIE Charge-Coupled Devices and Solid State Optical Sensors* **1447**, 123.

### Plasma emission and analysis / 플라즈마 방출 및 분석

- Mewe, R., Gronenschild, E. H. B. M., and van den Oord, G. H. J., *Astron. Astrophys. Suppl.* **62**, 197, 1985.
- Mewe, R., Lemen, J. R., and van den Oord, G. H. J., *Astron. Astrophys. Suppl.* **63**, 511, 1986.
- Craig, I. J. D. and Brown, J. C., *Inverse Problems in Astronomy*, Adam Hilger, 1986.
- Fludra, A. and Sylwester, J., *Solar Phys.* **105**, 323, 1986.
- Strong, K. T., Acton, L. W., Brown, W. A., Claflin, S. L., Lemen, J. R., and Tsuneta, S., 1991, *Adv. Space Res.* (to be published).
- Strong, K. T. and Lemen, J. R., 1987, unpublished work.

### Comparison and predecessors / 비교 및 선행

- Vaiana, G. S., Van Speybroek, L., Zombeck, M. V., Krieger, A. S., Silk, J. K., and Timothy, A., "The S-054 X-ray Telescope Experiment on Skylab", *Space Sci. Instr.* **3**, 19–76, 1977.
- Sakurai, T., 1990, in Y. Osaki and H. Shibahashi (eds.), *Progress of Seismology of the Sun and Stars*, Springer Lecture Notes in Physics **367**, p. 253.
- Kane, S. R., Anderson, K. A., Fenimore, E. E., Klebesadel, R. W., and Laros, J. G., *Astrophys. J.* **233**, L151, 1979.
- McTiernan, J. M., 1991, in R. Canfield and Y. Uchida (eds.), *Proc. K. Tanaka Memorial Symposium*, Springer-Verlag.
- Bendinelli, O., *Astrophys. J.* **366**, 599, 1991.

### Successors / 후속 미션

- Golub, L., Deluca, E. E., Austin, G., Bookbinder, J., Caldwell, D., Cheimets, P., et al., "The X-Ray Telescope (XRT) for the Hinode Mission", *Solar Physics* **243**, 63–86, 2007. — Direct optical and operational successor to SXT / SXT의 직계 광학·운영 후속 기기.
- Handy, B. N. et al., "The transition region and coronal explorer (TRACE)", *Solar Physics* **187**, 229–260, 1999. — Inherited SXT cadence/autonomy philosophy in the EUV / EUV에서 SXT의 카덴스/자율 철학을 계승.
- Lemen, J. R. et al., "The Atmospheric Imaging Assembly (AIA) on the Solar Dynamics Observatory (SDO)", *Solar Physics* **275**, 17–40, 2012. — 12 s cadence multi-channel imager extending SXT philosophy / SXT 철학을 EUV 다채널 12 s 카덴스로 확장.

### Note on archive / 자료 보관

SXT raw and Level-0 data products are archived at the Solar Data Analysis Center (NASA/GSFC) and at the ISAS Yohkoh archive; the SolarSoft IDL distribution still ships the SXT response and PSF routines (`sxt_eff_area.pro`, `sxt_temp.pro`) that operationally implement Equations 1–8 and Fig. 9 of this paper.
SXT raw and Level-0 data are archived at NASA/GSFC SDAC and at ISAS; SolarSoft IDL ships `sxt_eff_area.pro` and `sxt_temp.pro` that operationally implement Equations 1–8 and Fig. 9 of this paper.
