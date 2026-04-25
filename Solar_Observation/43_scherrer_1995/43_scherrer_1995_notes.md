---
title: "The Solar Oscillations Investigation - Michelson Doppler Imager"
authors: [Scherrer, P. H., Bogart, R. S., Bush, R. I., Hoeksema, J. T., Kosovichev, A. G., Schou, J., et al.]
year: 1995
journal: "Solar Physics 162, 129-188"
doi: "10.1007/BF00733429"
topic: Solar_Observation
tags: [helioseismology, MDI, SOHO, Michelson_interferometer, Doppler, magnetogram, p_modes, instrumentation]
status: completed
date_started: 2026-04-25
date_completed: 2026-04-25
---

# 43. The Solar Oscillations Investigation - Michelson Doppler Imager / 태양 진동 탐사 - 마이켈슨 도플러 이미저

---

## 1. Core Contribution / 핵심 기여

This paper is the comprehensive design and overview document for the Solar Oscillations Investigation (SOI), one of the three helioseismology experiments aboard ESA/NASA's SOHO mission (the others being GOLF and VIRGO). The instrument that implements SOI is the Michelson Doppler Imager (MDI), built jointly by Stanford's Hansen Experimental Physics Laboratory and Lockheed Palo Alto Research Laboratory under principal investigator P. H. Scherrer. MDI is a Fourier-tachometer-class imaging spectrometer that records narrow-band (94 mÅ FWHM) filtergrams at five wavelengths spanning the Ni I 6768 Å mid-photospheric absorption line on a 1024×1024 CCD every minute, computing on-board the line-of-sight Doppler velocity (20 m/s 1-σ noise per pixel per minute), continuum intensity, line depth, and longitudinal magnetic field (20 G noise per pixel) at 4″ resolution over the full disk or 1.25″ over an 11′×11′ field of view. The optical heart of MDI is a cascade consisting of a 50 Å front window, an 8 Å blocker, a 465 mÅ Lyot filter, and two solid Michelson interferometers (free spectral ranges 377 mÅ and 189 mÅ) tuned by rotating waveplates. The paper covers eleven scientific objectives (interior structure, internal rotation, solar core, convection-zone dynamics, magnetic structures, excitation/damping, large-scale flows, magnetic diffusion, limb figure, radiative flux), the full instrument subsystem design (imaging optics, image stabilization, filters, CCD camera, on-board image processor, mechanisms), the calibration program, the four observing programs (Dynamics, Structure, Campaigns, Magnetic), the SSSC data reduction infrastructure, and projected performance.

본 논문은 ESA/NASA SOHO 임무에 탑재된 세 개의 일진동학 (helioseismology) 실험 중 하나인 SOI (Solar Oscillations Investigation)의 종합 설계 및 개요 문서이다 (다른 둘은 GOLF와 VIRGO). SOI를 구현하는 기기는 MDI (Michelson Doppler Imager)로, P. H. Scherrer를 PI로 하여 Stanford의 Hansen Experimental Physics Laboratory와 Lockheed Palo Alto Research Laboratory가 공동 제작했다. MDI는 푸리에 타코미터 (Fourier tachometer) 계열의 영상 분광기로, Ni I 6768 Å 중광구 흡수선을 가로지르는 5개 파장에서 94 mÅ FWHM의 협대역 필터그램을 1024×1024 CCD에 매분 촬영한다. 산출물은 시선 도플러 속도 (1분/픽셀 당 20 m/s 1-σ 노이즈), 연속 강도, 선 깊이, 종방향 자기장 (픽셀당 20 G 노이즈)이며, 공간 분해능은 전체 원반 4″ 또는 11′×11′ 시야의 1.25″ (고분해능)이다. MDI 광학의 핵심은 50 Å 전면창, 8 Å 차단 필터, 465 mÅ Lyot 필터, 그리고 자유 스펙트럼 범위 377 mÅ와 189 mÅ인 두 개의 고체 마이켈슨 간섭계 (회전 파장판으로 조정)의 캐스케이드이다. 본 논문은 11개 과학 목표, 전 부속계 설계 (영상 광학, 영상 안정화, 필터, CCD 카메라, 온보드 이미지 프로세서, 메커니즘), 보정 프로그램, 4개 관측 프로그램 (Dynamics, Structure, Campaigns, Magnetic), SSSC 데이터 처리 인프라, 예측 성능을 다룬다.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction and Mission Context (Sec. 1) / 도입과 임무 맥락

The SOHO mission, launched in December 1995 (after this paper went to press) into a halo orbit around the Sun-Earth L1 Lagrangian point, addresses three areas: (1) probing the solar interior via helioseismology, (2) heating mechanism of the corona, (3) origin of the solar wind. Three helioseismology instruments work in concert: GOLF (Global Oscillations at Low Frequencies, full-disk integrated Na D Doppler, sensitive to $\ell \le 3$), VIRGO (Variability of solar IRradiance and Gravity Oscillations, Sun-as-a-star irradiance plus 16-pixel image), and SOI/MDI (full-disk imaging Doppler/magnetic up to $\ell \approx 1500$).

SOHO 임무는 1995년 12월 (본 논문 출간 이후) 태양-지구 L1 라그랑주 점 헤일로 궤도로 발사되었다. 세 가지 과학 영역은 (1) 일진동학으로 태양 내부 조사, (2) 코로나 가열 기제, (3) 태양풍의 기원이다. 세 개의 일진동학 기기가 상보적으로 작동한다: GOLF (저주파 적분 도플러, $\ell \le 3$), VIRGO (태양 통합 복사휘도 + 16-픽셀 영상), SOI/MDI ($\ell \approx 1500$까지의 전체 원반 영상 도플러/자기). 헤일로 궤도의 연속 햇빛은 세 실험 모두에 필수이며, MDI의 고속 텔레메트리 요구가 SOHO 사양을 결정한 주요 인자였다.

The paper traces a 17-year history from Bohlin's 1978 NASA Science Working Group through the Noyes & Rhodes (1984) report (recommending "a space-qualified two-dimensional imaging Doppler instrument" at L1), the 1987 ESA/NASA SOHO Announcement of Opportunity, March 1988 selection of SOI-MDI, October 1990 start of full-scale development, January 1994 interface testing, and April 28 1994 final delivery of the flight instrument.

논문은 1978년 Bohlin의 NASA SWG → 1984년 Noyes & Rhodes 보고서 ("L1에서의 우주급 2차원 영상 도플러 기기" 권고) → 1987년 ESA/NASA SOHO AO → 1988년 3월 SOI-MDI 선정 → 1990년 10월 본격 개발 시작 → 1994년 1월 인터페이스 시험 → 1994년 4월 28일 최종 비행 모델 인도까지의 17년 역사를 추적한다.

### Part II: SOI Science Program (Sec. 2) / SOI 과학 프로그램

**Mode types (Sec. 2.1)**: MDI is sensitive to all three principal classes of solar oscillations:
- $p$-modes: acoustic waves trapped between the high density gradient just below the photosphere (upper turning point $r_2$) and a deeper sphere of avoidance ($r_1$) determined by the sound-speed gradient. $r_1$ depends only on frequency $\nu$ and degree $\ell$; thus combining modes with the same $r_2$ but different $r_1$ gives interior sound speed via inversion.
- $g$-modes: internal gravity waves trapped beneath the convection zone; sensitivity diagnostic of solar core ($g$-modes have highest amplitude near center). MDI predicts SNR ≈ 5 for 1 mm/s amplitude modes if energy is constant in frequency.
- $f$-modes: surface gravity waves (Lamb-like, $\omega^2 \approx g_\odot k$).

**The $\ell$-$\nu$ diagram (Fig. 2)**: At fixed $\ell$, modes form a discrete set indexed by radial order $n$. MDI accesses the entire diagram up to $\ell \approx 4000$ and 17 mHz (vs ground-based limit $\ell \le 200$). Modes $a, b, c, d$ in Fig. 2 have lower turning points respectively at the base of the convection zone, the middle of the He II ionization zone, the H ionization zone, and the peak of the superadiabatic gradient.

**모드 종류 (2.1절)**: MDI는 세 가지 주요 진동 모드에 모두 민감하다. $p$-모드는 광구 바로 아래 고밀도 기울기 (상부 회귀점 $r_2$)와 음속 기울기에 의해 결정되는 더 깊은 회피 구면 ($r_1$) 사이에 갇힌 음향파다. $r_1$은 주파수 $\nu$와 차수 $\ell$에만 의존하므로, 동일 $r_2$이지만 다른 $r_1$의 모드를 결합하면 역산을 통해 내부 음속을 얻을 수 있다. $g$-모드는 대류대 아래 내부 중력파로, 진폭이 중심부에서 최대이므로 태양 코어 진단에 중요하다. MDI는 1 mm/s 진폭 모드에 대해 SNR ≈ 5를 예측한다. $f$-모드는 표면 중력파 ($\omega^2 \approx g_\odot k$)이다.

**$\ell$-$\nu$ 도표 (Fig. 2)**: 고정 $\ell$에서 모드는 방사 차수 $n$으로 인덱스된 이산 집합을 이룬다. MDI는 $\ell \approx 4000$, 17 mHz까지 전 도표를 다룬다 (지상 관측은 $\ell \le 200$).

**Eleven SOI science objectives (Sec. 2.3)**: (1) Convection-zone dynamics — giant cells, supergranulation, mid-latitude jets via local helioseismology and surface correlation tracking. (2) Mean radial structure — invert $m$-averaged frequencies for $\rho(r)$, $p(r)$, $c(r)$; helium abundance $Y$; equation of state. (3) Internal rotation — antisymmetric splitting of $(n,\ell,m)$ multiplets gives $\Omega(r,\theta)$. (4) Solar core — $g$-modes if detectable. (5) Magnetic structures and activity cycle — frequency shifts at 0.1% level scale as $f^{-1}$ where $f$ is filling factor. (6) Excitation and damping — line widths and asymmetries diagnose mode-flow coupling. (7) Convection and large-scale flows — combine LOS Doppler with horizontal correlation tracking. (8) Magnetic field measurements — full-disk magnetograms every 96 minutes (later changed to 96 s during operations). (9) Magnetic diffusion and field advection. (10) Limb figure — 0.0007″ accuracy on solar limb shape per minute. (11) Radiative flux budget — proxy continuum and magnetic mapping for VIRGO calibration.

**11개 과학 목표 (2.3절)**: 대류대 동력학, 평균 방사 구조, 내부 회전, 태양 코어, 자기 구조 및 활동 주기, 여기/감쇠, 대규모 흐름, 자기장 측정, 자기 확산/수송, 림 형상, 방사 플럭스 예산. 각 목표는 특정 $\ell$, $\nu$ 영역의 모드와 분석 기법 (구면조화 분해, 링 다이어그램, 시간-거리, 상관 추적)에 대응된다.

### Part III: The MDI Instrument (Sec. 3) / MDI 기기

**Overview (Sec. 3 intro)**: MDI is based on the Fourier Tachometer technique (Brown 1980; Evans 1980): record filtergrams at multiple wavelengths around a spectral line and infer Doppler shift from intensity ratios. Two physical enclosures: Optics Package (OP, 23.8 kg, mounted isostatically on six fiberglass legs) and Electronics Package (EP, 31.0 kg, with 42 MB image-processor memory). Total mass 56.5 kg, power 38 W. Telemetry: 5 kbps continuous (LRT) plus 160 kbps (HRT) when available.

**Imaging optics (Sec. 3.1)**: 12.5 cm primary objective + secondary form a refracting telescope with effective focal length 1867 mm. Light is folded by an Image Stabilization System (ISS) tilt mirror, then split by a polarizing beamsplitter. The s-component enters the instrument through a quarter-wave plate (which circularly polarizes back-reflections to a light trap) and through the linear entrance polarizer of the Lyot filter. The Calibration/Focus wheels provide nine focus depths and a CALMODE lens pair (images the pupil onto the focal plane for integrated-sunlight calibration). The beam distribution system after the oven separates Full Disk (4″ resolution, 34×34′ FOV) and High Resolution (1.25″, 11×11′ FOV, magnified ×3.2, centered 160″ north of equator). The FD path is intentionally defocused by two focus steps to suppress aliasing of unresolved 2″ structures; this drops MTF at 80% Nyquist from 47% to 17% but reduces aliased signal from 22% to 3%.

**광학 개요 (3.1절)**: 12.5 cm 일차 대물 + 이차 렌즈가 1867 mm 유효 초점 거리 굴절 망원경을 형성한다. ISS 기울기 거울이 빛을 접고 편광 빔스플리터가 분리한다. s-성분은 1/4 파장판 (역반사를 원편광으로 변환해 광 트랩으로 보냄)과 Lyot 필터의 직선 진입 편광기를 거쳐 기기 내부로 들어간다. 보정/초점 휠은 9가지 초점 깊이와 CALMODE 렌즈 쌍을 제공한다. 빔 분배 시스템은 FD (4″, 34×34′)와 HR (1.25″, 11×11′, ×3.2 확대, 적도 북쪽 160″) 경로로 분리한다. FD 경로는 의도적으로 2 step 디포커스되어 알리아싱을 억제한다.

**Image Stabilization System (Sec. 3.2)**: A 3-point PZT-actuated tilt mirror locks the solar limb position via four orthogonal photodiode pairs. Total tilt range ±19″, operational jitter range ±7″. Specification: jitter < 0.03″ p-p (1/20 of an HR pixel) keeps velocity error below 7 m/s. Servo response (Fig. 6) shows >100× jitter reduction below 10 Hz. Error signals sampled at 16 Hz (averaged over 7 s for housekeeping) or 512 Hz for special calibrations.

**ISS (3.2절)**: 3점 PZT 구동 기울기 거울이 태양 림 위치를 폐루프 잠금. 진동 < 0.03″ p-p가 속도 오차 < 7 m/s를 보장. 10 Hz 이하에서 100배 이상의 진동 감쇠.

**Filter system (Sec. 3.3)**: The cascade is the heart of MDI:
1. **50 Å front window**: bonded RG630 + GG475 glass with multilayer dielectric coating; only blocker of IR.
2. **8 Å blocker**: three-period dielectric interference filter inside the oven.
3. **Lyot filter (465 mÅ FWHM)**: six-element 2:2:4:6:8 birefringent design, 70+ components, oil-coupled (Q2-3067), 157.2 mm long. Temperature sensitivity ≤ 8 mÅ/°C.
4. **Michelson 1 (188 mÅ FSR)**: solid BK7 polarizing-beamsplitter design with vacuum leg (Stonehenge copper standoffs for thermal compensation) and solid glass leg. 43 mm clear aperture.
5. **Michelson 2 (94 mÅ FSR)**: same design, narrower bandpass.

The entire filter set sits in a temperature-stable oven (33–40 °C selectable, ±0.01°C/hr stability via ±0.5°C/hr OP stability and fiberglass thermal isolation). Tuning by rotating half-wave retarders (MTM1 and MTM2) in 2° steps; image registration during tuning < 0.15″.

**필터 시스템 (3.3절)**: MDI의 심장. 50 Å 전면창 → 8 Å 블로커 → 465 mÅ Lyot → 188 mÅ M1 → 94 mÅ M2 캐스케이드. Lyot은 6원소 2:2:4:6:8 복굴절 설계, 70+ 부품, 오일 결합. Michelson은 BK7 편광 빔스플리터에 진공 다리와 고체 유리 다리; "스톤헨지" 구리 스페이서가 열보상. 회전 1/2 파장판으로 조정. 오븐은 0.01°C/hr 안정도. Fig. 9는 마이켈슨 표면에 걸쳐 중심 파장이 ±5 mÅ 이상 변하는 비균일성을 보여주며, 이는 보정의 주된 도전 과제이다.

The Michelson central wavelength varies significantly across the face of the interferometer (Fig. 9), causing a calibration challenge: peak transmission wavelength maps must be measured. Filtergrams at five tunings 75 mÅ apart sample the line: $F_0$ near continuum, $F_1$ and $F_4$ on the wings, $F_2$ and $F_3$ on the line core.

5개 필터그램은 75 mÅ 간격으로 선을 샘플링한다: $F_0$ 연속체 부근, $F_1$/$F_4$ 날개, $F_2$/$F_3$ 코어 근처.

**Camera system (Sec. 3.4)**: 1024×1024 front-illuminated 3-phase CCD with 21 µm pixels (Loral Aeronutronic), MPP technology, partially inverted for higher full well. Operated at -80 °C via 960 cm² radiator + flexible heater. Performance at -70 °C: max signal 450 ke⁻; read noise 50 e⁻; dark current 0.3 e⁻/s; gain 110 e⁻/DN; CTE 0.999995 (1/2 full well), 0.99998 (Fe⁵⁵); linearity within 0.5%. Read-out 500 kpixel/s (frame in 2.2 s); rapid flush 87 ms. To maintain 3-second cadence the instrument has 0.8 s minus exposure (~100 ms) to reconfigure.

**카메라 (3.4절)**: 1024² 전조명 3상 CCD, 21 µm 픽셀, MPP 기술, -80 °C에서 운용. 노이즈 50 e⁻, 풀 웰 450 ke⁻, 게인 110 e⁻/DN. 픽셀당 12-bit A/D, 500 kpix/s 판독.

**On-board computer (Sec. 3.5)**: Two units: Dedicated Experiment Processor (DEP, Intel 80C86, 64 KB EEPROM, 128 KB RAM) for sequencing and Image Processor (IP, custom ASICs based on 2900 bitslice) for real-time image arithmetic with 2.125 MB memory in 20 pages. Time resolution 1/16 s, accuracy 1/2048 s. Frame list synchronized to TAI minute (5 s offset accounts for SOHO at L1 light travel).

**온보드 컴퓨터 (3.5절)**: DEP (Intel 80C86)가 시퀀싱, IP (커스텀 ASIC)가 실시간 영상 산술 (20 페이지 × 2.125 MB). 1/16 초 시간 해상도, TAI 분 동기화 (L1 광행 시간을 위해 5초 오프셋).

**Mechanisms (Sec. 3.6)**: Eight brushless DC stepper motors (Akin et al. 1993): front door (redundant pair), alignment (2 motors driving rear leg spindles, ±13′ range), two cal/focus wheels (144 steps/rev), polarization analyzer wheel (RCP/LCP/s/p positions), Michelson tuning motors MTM1/MTM2 (2° steps, 1″ image-stability requirement), shutter (80° sector blade, 40 ms–16.4 s exposure, 250 µs resolution).

### Part IV: Observables and Performance (Sec. 4) / 관측량과 성능

**Observable computation (Sec. 4.1)**: From the five filtergrams $F_0,\dots,F_4$ at $\lambda_0,\lambda_0\pm 75,\lambda_0\pm 150$ mÅ, the on-board IP computes:

$$\alpha = \begin{cases} (F_1+F_2-F_3-F_4)/(F_1-F_3), & \text{if } F_1+F_2-F_3-F_4 > 0 \\ (F_1+F_2-F_3-F_4)/(F_4-F_2), & \text{otherwise} \end{cases}$$

The IP looks up Doppler velocity from $\alpha$ via a precomputed table built from simulations using parameterized solar line profiles and measured filter transmission profiles (Fig. 12). Properties:
1. Numerator is essentially "blue-wing − red-wing" intensity; denominator is "continuum − line-center", so $\alpha$ is the ratio of the antisymmetric to symmetric line-profile components.
2. Insensitive to linear gain and offset variations (raw CCD images usable without flat-fielding).
3. Insensitive to line depth but has modest systematic errors when line width differs from the table's assumption (Fig. 12: ~150 m/s error over ±3000 m/s for ±25% width variation).
4. Errors from Michelson mistuning and Lyot miscentering are calibrated and removed.
5. Width-dependent errors rise steeply for $|v| > 4000$ m/s.

Line depth: $I_{\text{depth}} = \sqrt{2[(F_1-F_3)^2+(F_2-F_4)^2]}$ — from Fourier interpretation $I = I_c - I_d \cos(2\pi(\lambda-\lambda_0)/P)$. Continuum: $I_c = 2 F_0 + I_{\text{depth}}/2 + I_{\text{ave}}$ with cancellation of Doppler crosstalk to 0.2%. Magnetogram: difference of velocities measured in RCP and LCP, scaled by Zeeman calibration (Fig. 13 confirms y=x relation between modeled flux density and signal up to 3000 G).

**관측량 계산 (4.1절)**: 5개 필터그램으로부터 $\alpha$를 계산하고 lookup table로 도플러 속도 변환. $\alpha$는 비대칭/대칭 선 성분 비율로, 게인/오프셋에 둔감. 선 깊이는 푸리에 해석에서 유도. 자기장은 RCP–LCP 도플러 차로 측정.

**Noise levels (Table III)**: 1-σ per pixel per minute — Doppler 20 m/s, continuum 0.3%, line depth 0.7%, magnetogram 20 G. Horizontal velocity 30 m/s over 8 hours via correlation tracking. Limb position 0.02″ over 5 minutes.

**노이즈 (Table III)**: 도플러 20 m/s, 연속 강도 0.3%, 선 깊이 0.7%, 자기장 20 G (모두 1-분 1-픽셀 1-σ). 수평 속도 30 m/s (8시간), 림 위치 0.02″ (5분).

**On-board compression (Sec. 4.2)**: Lossless Rice (1979) algorithm — first pixel raw, then differences encoded as $k$ low bits unchanged + $n-k$ bits in efficient table. ~6-7 bits/pixel for velocity, 5-6 for magnetic. Intensity images use a scaled square root before differencing → 4-5 bits/pixel (lossy but white-noise-limited).

**압축 (4.2절)**: 무손실 Rice 알고리즘. 속도 6-7 bits/pixel, 자기장 5-6, 강도 (스케일된 제곱근 후) 4-5 bits/pixel.

**Calibration (Sec. 4.3)**: Most challenging task. Four critical instrument parameters (Lyot center, two Michelson centers, intensity at CCD) must be measured in 4-D ray space (pixel x,y, ray angle $\theta_x,\theta_y$). Laboratory tests over thousands of CCD images during 11-hour time series. Mathematical model of the full measurement process predicts velocity for known solar inputs and yields per-pixel calibration curves. CALMODE provides integrated-sunlight reference (each pixel sees the whole Sun via pupil imaging). Fig. 14 compares observed vs simulated CALMODE velocity (gradient ~1200 m/s); Fig. 15 shows raw and corrected dopplergrams (saturating at -800/+2500 raw vs -500/+1000 m/s corrected).

**보정 (4.3절)**: 4개 핵심 매개변수 (Lyot 중심, Michelson 두 중심, CCD 강도)를 4차원 광선 공간 (픽셀 x,y, 광선각 $\theta_x,\theta_y$)에서 측정. 수학적 모델이 픽셀별 보정 곡선 생성. CALMODE는 적분 햇빛 기준 제공.

**Performance (Sec. 4.4)**: Pre-flight test data show photon-noise-limited performance. Fig. 16 shows an $\ell$-$\nu$ diagram from July 5, 1993 ground test data (modes visible from 1.5 to 5 mHz, $\ell$ up to 250). Fig. 17: large-scale 5% intensity gradient corrected by Kuhn et al. (1991) flat-field method. Fig. 18: high-resolution magnetogram compared to KPNO (2.3″) - MDI 4″ FD and 1.3″ HR.

**성능 (4.4절)**: 사전 비행 시험에서 광자 노이즈 한계 성능 입증. Fig. 16의 $\ell$-$\nu$ 도표는 1993년 7월 지상 시험 데이터에서 1.5–5 mHz, $\ell \le 250$ 모드를 보여준다.

### Part V: Observing Programs (Sec. 5) / 관측 프로그램

**Dynamics Program (Sec. 5.1)**: 60+ continuous days of 160 kbps HRT each year. Two complete images per minute: full-disk velocity + (FD intensity or HR velocity). 86,400 consecutive 1-min frames → mode frequencies to 0.2 µHz. Full-disk modes accessible up to $\ell = 750$ as resolved peaks; up to $\ell = 1500$ as ridges. Mix of 10 days FD intensity at start/middle/end + two 15-day HR velocity intervals.

**Dynamics 프로그램 (5.1절)**: 매년 60일 이상 연속 160 kbps HRT. 분당 두 영상 (전체 원반 속도 + FD 강도 또는 HR 속도). 86,400 프레임으로 0.2 µHz 모드 주파수 정밀도. $\ell \le 750$ 분리, $\ell \le 1500$ 능선.

**Structure Program (Sec. 5.2)**: Always-on 5 kbps channel. 86% allocated to ~20,000 spatial-averaged velocity bins (Table IV): initially 12″-spaced 24″ FWHM Gaussian on inner 90% of disk ($\ell \le 250$); later switches to non-uniform binning (Fig. 20) sensitive to zonal ($m=0$) and sectoral ($m=\ell$) modes up to $\ell_{\max} \approx 350$. Plus 64 low-$\ell$ velocity super-pixels (matched to LOI/VIRGO), 64 low-$\ell$ continuum, 15,000 limb-figure pixels (6-pixel annulus 24-min Gaussian averaged), 128² flux-budget continuum and magnetic proxy line-depth.

**Structure 프로그램 (5.2절)**: 5 kbps 채널 상시. 86%가 20,000개 공간 평균 속도 빈 (초기 12″ 간격 24″ FWHM 가우시안, 이후 zonal/sectoral 비균일 binning). 림 figure 6-픽셀 annulus, 저-$\ell$ 슈퍼픽셀, 자기 프록시 등.

**Campaigns (Sec. 5.3)**: Daily 8-hour HRT intervals. Up to 10 filtergrams/min with full flexibility (FD/HR, OBSMODE/CALMODE, custom polarization). Examples: continuous high-res magnetograms, simultaneous velocity+intensity, line-scan diagnostics, 20-second-cadence high-frequency mode hunting.

**Campaigns (5.3절)**: 일일 8시간 HRT 동안 분당 최대 10 필터그램, 사용자 정의 시퀀스. 고분해능 자기, 빠른 카덴스 등.

**Magnetic Program (Sec. 5.4)**: Full-disk magnetograms every 96 minutes during all observing programs (stored in MDI memory if HRT not available, downlinked during HRT bursts).

**Magnetic 프로그램 (5.4절)**: 모든 프로그램 중 96분마다 전체 원반 자기장 (MDI 메모리에 저장 후 HRT 버스트로 다운링크).

### Part VI: Data Analysis Infrastructure (Sec. 6–7) / 데이터 분석 인프라

**SOI Science Support Center (SSSC)**: At Stanford. Production system: 6× R8000 300-MFLOPS CPUs, 2 GB memory, 400 GB online disk. Analysis system: 4× R4400 75-MFLOPS, 768 MB, 100 GB. Off-line storage: pair of Ampex 410 DD-2 19-mm helical-scan tape libraries, 15 MB/s sustained, 2 TB random access. Plus Lago wheel (2× Exabyte 5 GB tapes × 54 slots).

**SSSC**: Stanford 소재. R8000 6 코어 + R4400 4 코어 SGI 시스템. 2 TB 자기 테이프 라이브러리.

**Processing levels**: Level 0 = raw telemetry decoded; Level 1 = calibrated physical units (dopplergrams, magnetograms, photograms); Level 2 = derived (spherical harmonic mode amplitudes, ring-diagram parameters, flow fields); Level 3 = inversions and modeling (interactive, not pipelined).

**처리 레벨**: 0 (원시), 1 (보정 물리 단위), 2 (구면조화 모드 진폭, 링 다이어그램, 흐름장), 3 (역산, 대화형).

**Annual data volume**: 274 GB raw, 540 GB Level 0, 620 GB Level 1, ~1500 GB Level 2 → total ~2.9 TB/year.

**연간 데이터량**: 274 GB 원시 → 540 GB Level 0 → 620 GB Level 1 → ~1500 GB Level 2 → 약 2.9 TB/년.

**Analysis modules (Sec. 7.3)**: (1) Apodize + remap → spherical harmonic projection → time series of $a_\ell^m(t)$ → mode parameters via Schou (1992) method including frequencies, amplitudes, linewidths, asymmetries, $a$-coefficients. (2) Ring-diagram analysis (Hill 1988): 3-D Fourier transform on small co-moving patch; ring distortion gives subsurface flows. (3) Local correlation tracking on high-resolution images for surface flows. Plus inversions: Optimally Localized Averages (OLA), Regularized Least Squares (RLS) for radial structure; 1.5-D and 2-D RLS for rotation rate (Schou et al. 1994).

**분석 모듈 (7.3절)**: 구면조화 분해 → Schou 방법으로 모드 파라미터 추출; 링 다이어그램 분석 (Hill 1988)으로 표면 아래 흐름; 입자 상관 추적; 역산 (OLA, RLS).

---

## 3. Key Takeaways / 핵심 시사점

1. **Imaging Doppler from space transforms helioseismology** — MDI's $\ell$-$\nu$ access (up to 4000) and uninterrupted L1 viewing exceed any ground-based instrument by orders of magnitude in coverage and duty cycle. / **우주 영상 도플러는 일진동학을 변환한다** — MDI는 $\ell \le 4000$ 접근과 L1 헤일로 궤도의 무중단 시야로 모든 지상 기기를 압도한다.

2. **The Fourier-tachometer concept** — Sample a spectral line at five evenly-spaced wavelengths; ratio $\alpha = (F_1+F_2-F_3-F_4)/(F_{\max\text{wing}})$ gives Doppler velocity insensitive to gain/offset. The denominator switching ensures monotonic single-valued response over ±4000 m/s. / **푸리에 타코미터 개념** — 5개 등간격 파장 샘플링; $\alpha$ 비율이 게인/오프셋에 둔감한 도플러 속도 제공. 분모 전환으로 ±4000 m/s 범위에서 단일값 응답 보장.

3. **Cascade filtering is essential** — Resolving 94 mÅ in a 6768 Å line ($R \approx 7\times 10^4$) requires layered filtering: 50 Å front window → 8 Å blocker → 465 mÅ Lyot → 188 mÅ Michelson → 94 mÅ Michelson. Each filter narrows by factor ~2-50. / **캐스케이드 필터가 필수** — 6768 Å에서 94 mÅ 분해 ($R \approx 7\times 10^4$)에는 50 Å 전면창 → 8 Å 블로커 → 465 mÅ Lyot → 188 mÅ M1 → 94 mÅ M2의 다층 필터링이 필요. 각 단계가 2–50배 좁힌다.

4. **Calibration is the dominant challenge** — Wavelength gradients across the Michelson faces (±5–8 mÅ over 50 mm) and angular sensitivities require per-pixel velocity calibration curves derived from a 4-D (x, y, $\theta_x$, $\theta_y$) instrumental model. CALMODE (pupil imaging onto focal plane) provides integrated-sunlight reference. / **보정이 주된 도전** — Michelson 면을 가로지르는 파장 기울기 (±5–8 mÅ)와 각 민감도 때문에 4차원 (x, y, $\theta_x$, $\theta_y$) 기기 모델 기반 픽셀별 속도 보정 곡선이 필요. CALMODE이 적분 햇빛 기준 제공.

5. **Stability over speed** — MDI prioritizes long-term stability: ±0.01°C/hr filter oven, < 0.03″ jitter via ISS, telecentric optics, temperature-compensated fiberglass mounts, "Stonehenge" copper standoffs. Photon shot noise (20 m/s/pix/min) is the design floor, not the systematic error. / **속도보다 안정성** — MDI는 장기 안정성을 최우선화: ±0.01°C/hr 오븐, ISS로 < 0.03″ 진동, 텔레센트릭 광학, 열보상 유리섬유 마운트, 스톤헨지 구리 스페이서. 광자 산탄 노이즈 (20 m/s)가 설계 한계이고 계통오차가 아님.

6. **Multi-tier observing strategy matches telemetry budget** — 5 kbps Structure (always on, ~20,000 spatial averages giving long-duration $\ell \le 250$ coverage), 160 kbps Dynamics (60-day annual block, full $\ell \le 1500$), 8-hour daily Campaigns, plus 96-min synoptic magnetograms. Each tier optimizes data return for its niche. / **다층 관측 전략** — 5 kbps Structure (상시, $\ell \le 250$ 장기), 160 kbps Dynamics (연 60일, $\ell \le 1500$), 일일 8시간 캠페인, 96분 자기장. 각 계층이 자신의 영역에 최적화.

7. **On-board image processing is mission-enabling** — Raw filtergram rate (4.2 Mbit/s) far exceeds telemetry (5 + 160 kbps). The Image Processor performs the velocity, intensity, magnetogram, line-depth computation, off-limb pixel deletion, lossless Rice compression, and spatial averaging in real time using custom 2900-bitslice ASICs. / **온보드 영상 처리가 임무를 가능케 함** — 원시 필터그램 4.2 Mbit/s ≫ 텔레메트리 (5 + 160 kbps). IP는 속도/강도/자기장/선 깊이/림 외 픽셀 삭제/Rice 무손실 압축/공간 평균화를 실시간 수행 (커스텀 2900-bitslice ASIC).

8. **MDI is a blueprint** — The Stanford SSSC infrastructure, four-tier observing program, on-board observable computation, and Michelson+Lyot cascade are inherited (with improvements) by SDO/HMI (Schou et al. 2012) operating since 2010 and slated to continue through the 2030s. / **MDI는 청사진** — Stanford SSSC 인프라, 4계층 관측 프로그램, 온보드 관측량 계산, Michelson+Lyot 캐스케이드 모두 SDO/HMI (2010~)가 (개선과 함께) 계승.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Doppler shift / 도플러 천이

For nonrelativistic line-of-sight velocity $v$ in a spectral line at rest wavelength $\lambda$:

$$\boxed{\frac{\Delta\lambda}{\lambda} = \frac{v}{c}}$$

At Ni I 6767.78 Å: $\Delta\lambda/v = 22.58$ µÅ per m/s. MDI's 94 mÅ FWHM corresponds to ±4160 m/s line-of-sight velocity span, comfortably enclosing solar p-mode amplitudes (≤ 1 m/s coherent) and rotation (±2 km/s at limb).

### 4.2 Michelson channel spectrum / 마이켈슨 채널 스펙트럼

A polarizing Michelson interferometer with optical-path difference $\Delta = 2 n d$ between its two arms produces a sinusoidal spectral transmission:

$$\boxed{T(\lambda) = \frac{1}{2}\Big[1 + \cos\!\big(\tfrac{2\pi \Delta}{\lambda}\big)\Big] = \frac{1}{2}\Big[1 + \cos\!\big(\tfrac{2\pi(\lambda-\lambda_0)}{\Delta\lambda_{\text{FSR}}}\big)\Big]}$$

with free spectral range:

$$\Delta\lambda_{\text{FSR}} = \frac{\lambda^2}{2 n d}$$

For MDI: M1 has $\Delta\lambda_{\text{FSR}} = 377$ mÅ at 6768 Å (so $2 n d \approx 12.15$ mm); M2 has 189 mÅ (so $2 n d \approx 24.30$ mm — exactly twice M1). FWHM of one period is $\Delta\lambda_{\text{FSR}}/2$ = 188 mÅ (M1) and 94 mÅ (M2). Tuning is by rotating a half-wave plate between fixed polarizers, which shifts $\lambda_0$ by twice the rotation angle in retardance units.

### 4.3 Combined Lyot + Michelsons transmission / Lyot + Michelson 결합 투과

Full instrument transmission profile is the product:

$$T_{\text{tot}}(\lambda) = T_{\text{front}}(\lambda) \cdot T_{\text{block}}(\lambda) \cdot T_{\text{Lyot}}(\lambda) \cdot T_{M1}(\lambda) \cdot T_{M2}(\lambda)$$

The Lyot Gaussian-like 465 mÅ FWHM passband selects one period of $T_{M1}$ (188 mÅ); $T_{M1}$ in turn selects one period of $T_{M2}$ (94 mÅ). Sidelobes (apparent in upper panels of Fig. 7) yield few-percent peak-to-peak intensity modulation when integrated against a white-light source.

### 4.4 MDI Doppler proxy / MDI 도플러 프록시

Five filtergrams at $\lambda_0 + k \cdot 75$ mÅ for $k = -2,-1,0,+1,+2$, labeled $F_4, F_3, F_0, F_2, F_1$ (note ordering: $F_0$ central, $F_{1,2,3,4}$ at $\pm 75, \pm 150$ mÅ):

$$\boxed{\alpha = \begin{cases} \dfrac{F_1+F_2-F_3-F_4}{F_1-F_3}, & F_1+F_2-F_3-F_4 > 0 \\[6pt] \dfrac{F_1+F_2-F_3-F_4}{F_4-F_2}, & F_1+F_2-F_3-F_4 \le 0 \end{cases}}$$

**Interpretation**: $F_1+F_2-F_3-F_4$ is "blue-wing total" minus "red-wing total" — the antisymmetric component of the line about $\lambda_0$. Denominators $F_1-F_3$ (or $F_4-F_2$) measure "wing minus core" — a symmetric estimate of line depth. The ratio is therefore a Doppler-shift indicator normalized to line depth, hence insensitive to line-strength variations and to multiplicative gain. Switching denominators keeps the function monotonic and well-defined for both blue and red shifts up to ±4000 m/s.

The Image Processor stores a precomputed lookup table $v(\alpha)$ derived from convolving parameterized solar line profiles with the measured filter transmission profiles in 4-D ray space.

### 4.5 Line depth and continuum / 선 깊이와 연속 강도

Modeling the line near $\lambda_0$ as $I(\lambda) = I_c - I_d \cos(2\pi(\lambda-\lambda_0)/P)$ with period $P$ matched to filtergram spacing, the discrete Fourier interpretation gives:

$$\boxed{I_{\text{depth}} = \sqrt{2\,\big[(F_1-F_3)^2 + (F_2-F_4)^2\big]}}$$

(equivalent to magnitude of cosine + sine Fourier components from the four wing samples)

$$\boxed{I_{\text{cont}} = 2 F_0 + I_{\text{depth}}/2 + I_{\text{ave}}, \quad I_{\text{ave}} \equiv (F_1+F_2+F_3+F_4)/4}$$

Doppler crosstalk in $I_c$ cancels to 0.2%.

### 4.6 Magnetogram / 자기장 영상

Alternating polarization analyzer wheel between RCP and LCP yields two velocity measurements:

$$\boxed{B_\| = G \cdot (v_{\text{RCP}} - v_{\text{LCP}})}$$

where $G$ is a calibration factor including the Landé $g$ factor for Ni I 6768 (effective $g \approx 1.43$) and the unsplit equivalent width times continuum (Rees & Semel 1979). The relation is approximately linear (Fig. 13: y=x to within scatter for sunspot, penumbra, and plage profiles up to 3000 G).

### 4.7 Photon noise / 광자 노이즈

For a perfectly Doppler-sensitive instrument, the velocity error from photon counting is:

$$\sigma_v \sim \frac{c}{\lambda} \cdot \frac{1}{(\partial \ln I/\partial\lambda)\cdot \sqrt{N_\gamma}}$$

For MDI's 110 e⁻/DN, ~0.1 s exposures filling 1/2 well (~80,000 e⁻/pixel/filtergram), 5 filtergrams, $\partial\ln I/\partial\lambda \sim 1/(50\,\text{mÅ})$, this gives $\sigma_v \sim 20$ m/s — matching the measured noise floor in Table III.

### 4.8 $\ell$-$\nu$ power spectrum / 모드 능선

The full-disk Doppler image $v(\theta,\phi,t)$ is decomposed onto spherical harmonics:

$$a_\ell^m(t) = \int v(\theta,\phi,t) Y_\ell^{m*}(\theta,\phi) \sin\theta\, d\theta\, d\phi$$

then Fourier-transformed in time:

$$\hat a_\ell^m(\nu) = \int a_\ell^m(t) e^{-2\pi i \nu t} dt$$

The power spectrum $|\hat a_\ell^m(\nu)|^2$ averaged over $m$ at fixed $\ell$ produces ridges in $(\ell,\nu)$ space (Fig. 16) corresponding to radial orders $n = 1, 2, 3, \dots$. Mode frequencies $\nu_{n\ell m}$, line widths $\Gamma_{n\ell m}$, asymmetries, and rotational splitting coefficients $a_i$ are extracted by simultaneous fitting (Schou 1992).

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1908 ── Hale: solar magnetic field via Zeeman / 제만으로 태양 자기 측정
   │
1928 ── Lyot: birefringent filter / Lyot 복굴절 필터 발명
   │
1953 ── Babcock: solar magnetograph / 태양 자기영상기
   │
1962 ── Leighton, Noyes & Simon: 5-min oscillations / 5분 진동 발견
   │
1970 ── Ulrich: trapped acoustic mode hypothesis / 갇힌 음향 모드 가설
   │
1971 ── Leibacher & Stein: independent same hypothesis
   │
1975 ── Deubner: k-ω diagram observed / k-ω 도표 관측
   │
1979 ── Claverie et al.: low-degree global modes / 저차 전역 모드
   │
1980 ── Brown: Fourier Tachometer technique / 푸리에 타코미터 기법
   │
1983 ── "Helioseismology" coined / 일진동학 용어 정착
   │
1984 ── Noyes & Rhodes report: space mission recommended
   │
1988 ── GONG approved (Harvey et al.); SOI-MDI selected (Mar)
   │
1991 ── Duvall et al.: South Pole helioseismology
   │
1993 ── Duvall et al.: time-distance helioseismology
   │
1995 ── ◆ This paper: SOI-MDI design ◆ / 본 논문
   │     SOHO launched (Dec 2)
   │
1996 ── MDI begins routine science (May)
   │     Kosovichev: tachocline first imaged
   │
1998 ── Christensen-Dalsgaard et al.: solar interior reference model
   │
2010 ── SDO/HMI launches (Schou et al. 2012) / HMI 발사
   │     MDI's direct heir, Fe I 6173 Å, 4096² CCD
   │
2011 ── MDI retires after 15 years of operation
   │
2025 ── HMI continues; MDI-style helioseismology mature science
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Leighton, Noyes & Simon (1962) | Discovery of 5-min oscillations / 5분 진동 발견 | The phenomenon MDI was built to study at unprecedented precision. / MDI가 정밀 연구하려는 현상의 발견. |
| Deubner (1975) | $k$-$\omega$ diagram observed from ground / 지상에서 $k$-$\omega$ 관측 | First confirmation of trapped-wave dispersion; MDI extends to $\ell\le 4000$. / 갇힌 파의 분산 관계 첫 관측 확인; MDI는 $\ell\le 4000$ 까지 확장. |
| Brown (1980) / Evans (1980) | Fourier Tachometer technique | The measurement principle MDI implements: filtergram ratios give velocity. / MDI가 구현하는 측정 원리. |
| Title & Rosenberg (1981) | Lyot filter design | The 465 mÅ Lyot at the heart of MDI's filter cascade. / MDI 필터 캐스케이드의 핵심 Lyot. |
| Title & Ramsey (1980) | Tunable Michelson interferometer | The two solid Michelsons (188, 94 mÅ FSR) tunable by rotating waveplates. / 회전 파장판으로 조정되는 두 고체 Michelson의 원형. |
| Harvey et al. (1988) | GONG ground-based network / GONG 지상 네트워크 | Complementary; six sites, modes with $\ell \le 200$; cross-calibrated with MDI. / 상보적 6-사이트 지상 네트워크; MDI와 상호 보정. |
| Noyes & Rhodes (1984) | NASA SWG report / NASA SWG 보고서 | Recommended the L1 imaging Doppler mission that became SOI-MDI. / SOI-MDI의 임무 사양을 권고. |
| Duvall et al. (1993, 1997) | Time-distance helioseismology | Local helioseismology technique pioneered with MDI data. / MDI 데이터로 개척된 국부 일진동학 기법. |
| Schou et al. (2012) | SDO/HMI design paper / SDO/HMI 설계 논문 | MDI's direct successor: Fe I 6173 Å, 4096² CCD, 1″ resolution, 45-s cadence. / MDI 직계 후속 기기. |
| Domingo et al. (1995) | SOHO mission overview / SOHO 임무 개요 | Parent paper for the SOHO mission; companion to this paper. / 이 논문의 모 임무 논문. |
| Fröhlich et al. (1995) | VIRGO instrument paper / VIRGO 기기 논문 | Sister SOHO helioseismology instrument: Sun-as-a-star irradiance + 16-pixel image. / SOHO 자매 일진동학 기기. |
| Gabriel et al. (1995) | GOLF instrument paper / GOLF 기기 논문 | Sister SOHO helioseismology instrument: integrated Na D Doppler for low-$\ell$ modes. / SOHO 자매 저차 모드 도플러. |

---

## 7. References / 참고문헌

### Primary paper / 주 논문
- Scherrer, P. H., Bogart, R. S., Bush, R. I., Hoeksema, J. T., Kosovichev, A. G., Schou, J., Rosenberg, W., Springer, L., Tarbell, T. D., Title, A., Wolfson, C. J., Zayer, I., and the MDI Engineering Team (1995). "The Solar Oscillations Investigation - Michelson Doppler Imager." *Solar Physics* 162, 129-188. DOI: 10.1007/BF00733429

### Background and foundational / 배경 및 기초
- Brown, T. M. (1980). "The Fourier Tachometer." In *Solar Instrumentation: What's Next?*, ed. R. B. Dunn, NSO Sunspot, p. 150.
- Deubner, F.-L. (1975). "Observations of low wavenumber nonradial eigenmodes of the Sun." *Astron. Astrophys.* 44, 371.
- Evans, J. W. (1980). In *Solar Instrumentation: What's Next?*, ed. R. B. Dunn, NSO Sunspot, p. 155.
- Leibacher, J. W. and Stein, R. F. (1971). "A new description of the solar five-minute oscillation." *Astrophys. Lett.* 7, 191.
- Leighton, R. B., Noyes, R. W., and Simon, G. W. (1962). "Velocity fields in the solar atmosphere I. Preliminary report." *Astrophys. J.* 135, 474.
- Title, A. M. and Ramsey, H. E. (1980). "Improvements in birefringent filters." *Applied Optics* 19, 2046.
- Title, A. M. and Rosenberg, W. J. (1981). "Tunable birefringent filters." *Opt. Eng.* 20:6, 815.
- Ulrich, R. K. (1970). "The five-minute oscillations on the solar surface." *Astrophys. J.* 162, 933.

### Companion SOHO papers / 동반 SOHO 논문
- Domingo, V., et al. (1995). "The SOHO Mission: An Overview." *Solar Physics* (this issue).
- Fröhlich, C., et al. (1995). "VIRGO: Experiment for helioseismology and solar irradiance monitoring." *Solar Physics* (this issue).
- Gabriel, A., et al. (1995). "Global Oscillations at Low Frequencies from the SOHO mission (GOLF)." *Solar Physics* (this issue).

### Historical context / 역사적 맥락
- Noyes, R. W., and Rhodes, E. J. (eds.) (1984). *Probing the Depths of a Star: The Study of Solar Oscillations from Space*. NASA JPL, Pasadena.
- Gough, D. O., and Toomre, J. (1991). "Seismic observations of the solar interior." *Annu. Rev. Astron. Astrophys.* 29, 627.
- Harvey, J. W., and the GONG Instrument Development Team (1988). In *Seismology of the Sun and Sun-Like Stars*, ed. E. J. Rolfe, ESA Netherlands, p. 203.

### Successor instrument / 후속 기기
- Schou, J., et al. (2012). "Design and Ground Calibration of the Helioseismic and Magnetic Imager (HMI) Instrument on the Solar Dynamics Observatory (SDO)." *Solar Physics* 275, 229-259.

### Algorithms and analysis / 알고리즘 및 분석
- Hill, F. (1988). "Rings and trumpets — three-dimensional power spectra of solar oscillations." *Astrophys. J.* 333, 996.
- Kuhn, J. R., Lin, H., and Loranz, D. (1991). "Gain calibrating nonuniform solid-state arrays." *PASP* 103, 1097.
- Rees, D. E., and Semel, M. D. (1979). "Line formation in an unresolved magnetic element: A test of the centre of gravity method." *Astron. Astrophys.* 74, 1.
- Rice, R. F. (1979). "Some practical universal noiseless coding techniques." *SPIE Symp. Proc.* 207.
- Schou, J., Christensen-Dalsgaard, J., and Thompson, M. J. (1994). "On comparing helioseismic two-dimensional inversion methods." *Astrophys. J.* 433, 389.
