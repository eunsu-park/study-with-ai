---
paper_id: 31
topic: Solar_Observation
date: 2026-04-23
type: notes
title: "Observables Processing for HMI on SDO"
authors: "S. Couvidat, J. Schou, J.T. Hoeksema, R.S. Bogart, R.I. Bush, T.L. Duvall Jr., Y. Liu, A.A. Norton, P.H. Scherrer"
year: 2016
doi: "10.1007/s11207-016-0957-3"
journal: "Solar Physics, 291, 1887-1938"
tags: [HMI, SDO, observables, MDI-like, VFISV, polarization, calibration, Doppler, magnetogram]
---

# Observables Processing for HMI on SDO / HMI 관측량 처리

## Core Contribution / 핵심 기여

**한국어:**
본 논문은 NASA SDO 위성의 HMI(Helioseismic and Magnetic Imager) 장비로부터 도플러 속도, 시선방향(LoS) 자기장, 연속복사 강도, 선폭, 선깊이, 그리고 Stokes [I, Q, U, V] 편광 파라미터를 산출하는 두 개의 독립 파이프라인의 전체 설계·구현·교정을 가장 포괄적으로 기술한 참조 문헌이다. 45초 간격으로 전 태양면 4096 x 4096 픽셀 필터그램을 연속 수집하는 HMI는 Fe I 6173 Å 선의 6개 파장, 2개(LoS) 또는 4개(벡터) 편광 상태를 측정하며, LoS 파이프라인은 SOHO/MDI에서 승계한 Fourier 기반 MDI-like 알고리즘으로 45초마다 LoS 관측량을, 벡터 파이프라인은 720초 평균 Stokes 벡터(4 x 6 = 24개 이미지)를 산출한다. 저자들은 Level-1 준비, CCD 비선형성, 왜곡·롤·정렬, 편광 보정(편광 PSF 및 I-ripple 포함), 위상 맵, MDI-like Fourier 알고리즘, 다항식 속도 보정의 단계별 처리를 기술하고, 롤 각도 민감도, 24시간 궤도 속도 잔차, 강자기장 영역에서의 LoS 자기장 과소추정 등 주요 계통 오차를 정량화한다. 결론적으로 HMI는 원 설계 사양(LoS 속도 잡음 17 m/s, LoS 자기장 잡음 7 G/45s)을 모두 만족하지만, 저궤도 기인 잔차(궤도 주기 잡음)와 강자기장 포화가 주된 미해결 문제로 남아 있다.

**English:**
This paper is the most comprehensive reference for the end-to-end design, implementation, and calibration of the two independent pipelines that compute the HMI Level-1.5 observables: Doppler velocity, line-of-sight (LoS) magnetic field, continuum intensity, line width, line depth, and the Stokes polarization parameters [I, Q, U, V]. Operating on NASA's SDO spacecraft, HMI continuously observes the full solar disk with two 4096 x 4096 CCD cameras, sampling six wavelengths around the Fe I 6173 Å line with two circular polarizations (LoS) or four full-Stokes states (vector). The LoS pipeline runs every 45 seconds, applying a Fourier-based "MDI-like" algorithm inherited from SOHO/MDI, while the vector pipeline produces 720-s averaged Stokes vectors (4 x 6 = 24 images). The authors describe Level-1 preparation, CCD nonlinearity, distortion/roll/alignment, polarization calibration (including polarization PSF and I-ripple), phase maps, the MDI-like Fourier algorithm, and the polynomial velocity correction, then quantify systematic errors: roll-angle sensitivity, 24-hour orbital-velocity residuals, and LoS magnetic-field underestimation in strong-field regions. HMI meets all original performance specifications (LoS velocity noise 17 m/s; LoS magnetic-field noise 7 G at 45 s), with orbital-period systematics and strong-field saturation remaining the chief unresolved issues.

## Reading Notes / 읽기 노트

### §1 Introduction / 서론 (pp. 1888-1889)

**한국어:**
HMI 프라임 미션은 2010년 5월 1일부터 2015년 4월 30일까지 5년간 진행되었으며, 두 대의 4096 x 4096 CCD 카메라가 45초 또는 135초 주기로 Fe I 6173 Å 선 주변 6개 파장의 좁은 대역 필터그램을 수집했다. 총 8400만 장 이상이 촬영되어 설계 기대치의 99.86 %에 달한다. 2015년 봄까지 HMI 데이터를 인용한 논문이 1000편을 넘었다.

HMI는 SOHO 임무의 MDI(Michelson Doppler Imager; Scherrer+ 1995)를 계승한다. MDI는 Earth-Sun L1 지점에서 비교적 온화한 궤도에 있었으나, HMI는 지구동기궤도에 있어 SDO의 태양 시선속도가 하루 단위로 ±3500 m/s 수준으로 크게 변하며, 이로 인해 일단위 교정 잔차가 발생한다(이 논문에서 반복적으로 다루는 핵심 문제).

**English:**
The HMI prime mission ran from 1 May 2010 to 30 April 2015. Two 4096 x 4096 CCD cameras recorded narrow-band filtergrams at six wavelengths around Fe I 6173 Å every 45 s or 135 s, accumulating over 84 million exposures (99.86 % of the design goal). By spring 2015, more than 1000 published papers had used HMI data.

HMI succeeds SOHO's Michelson Doppler Imager (MDI; Scherrer+ 1995). While MDI's Earth-Sun L1 orbit was benign, SDO's geosynchronous orbit produces daily Sun-spacecraft radial velocity variations up to ±3500 m/s, causing daily calibration residuals - a recurring issue throughout the paper.

### §2 Observables Computation / 관측량 산출 (pp. 1889-1917)

#### §2.1 Production of Level-1 Filtergrams / 레벨-1 필터그램 생성

**한국어:**
Bush+ 2016에서 상세히 다뤄지는 Level-1 처리는 CCD 과주사행·열 제거, 다크 및 페데스탈 감산, 플랫필드 곱, 노출 시간(EXPTIME)으로 정규화(DN/s 단위)를 포함한다. 플랫필드는 매주 갱신되며 이상 픽셀은 Level-1 필터그램에 동반되는 bad-pixel list에 등록된다. Level-1 데이터는 definitive(3-4일 지연)과 NRT(near-real-time, 수 분 이내) 두 가지 모드로 생산된다. 링퓸(림) 위치와 반경은 target wavelength에 따라 공식상 변동하며(근태양 림에서 형성 고도 차이), CRPIX1/CRPIX2/R_SUN keyword에 기록된다.

**English:**
Level-1 processing (detailed by Bush+ 2016) removes CCD overscan rows and columns, subtracts dark and pedestal offsets, multiplies by a flat-field image, flags pixels with anomalous values, and normalizes by EXPTIME to give DN/s. Flat fields are updated weekly, and bad-pixel lists are attached to Level-1 records. Level-1 products come in definitive (3-4-day latency) and near-real-time (NRT, minutes) modes. The limb-fitter determines the effective solar radius as a function of wavelength because the formation height varies near the limb; the resulting corrected values are stored in CRPIX1, CRPIX2, and R_SUN keywords.

#### §2.2 The Observables Pipelines / 관측량 파이프라인

**한국어:**
두 개의 파이프라인이 있다. **LoS 파이프라인**(HMI_observables 모듈)은 전방 카메라(front camera)에서 45초 간격으로 6개 파장 x 2개 원형편광 = 12장의 Level-1 필터그램을 모아 LoS 속도, 자기장, 연속 강도, 선폭, 선깊이를 산출한다. **벡터 파이프라인**(HMI_IQUV_averaging 모듈)은 측면 카메라(side camera)에서 135초마다 6파장 x 6편광 = 36장을 촬영하며, 10개 framelist를 1350초 동안 임시 Wiener 보간으로 평균해 720초 Stokes [I, Q, U, V]를 생성한다. 그 결과 6파장 각각에 대해 I, Q, U, V의 24개 이미지가 vector-field 관측량의 기본 입력이 된다.

**English:**
Two pipelines coexist. The **LoS pipeline** (HMI_observables module) consumes 12 Level-1 filtergrams (six wavelengths x two circular polarizations) every 45 s from the front camera to compute LoS velocity, magnetic field, continuum intensity, line width, and line depth. The **vector pipeline** (HMI_IQUV_averaging module) operates on the side camera, capturing 36 filtergrams (six wavelengths x six polarizations) every 135 s. Ten framelists are averaged via a temporal Wiener interpolation over 1350 s (apodized with a cos^2 envelope, FWHM 720 s, nominally 23 nonzero weights) to produce the 720-s Stokes vector. The result is 24 images (I, Q, U, V at each of six wavelengths), feeding the vector-field observables.

#### §2.3 Filtergram Selection, Mapping, and Image Processing / 필터그램 선택과 영상 처리

**한국어:**
목표 시간 T_OBS에 가장 가까운 Level-1 필터그램을 target filtergram으로 선택한 뒤, 동일 파장·편광 조합의 다른 필터그램으로부터 OBS_VR, OBS_VW, OBS_VN(위성 속도 3성분), DSUN_OBS, CRLT_OBS, CRLN_OBS(Carrington 좌표), CROTA2(위치 각) 등의 키워드를 선형 보간한다. 각 Level-1 이미지는 관측 시각이 조금씩 다르므로, Wiener 방식의 아공간 보간으로 태양 자전에 의한 픽셀 변위를 보정한다. T_OBS는 SDO에서의 관측 중앙 시각(TAI), T_REC = T_OBS - (DSUN_OBS - 1 AU)/c는 1 AU 등가 시각이다.

**English:**
A target filtergram is chosen as the one closest in wavelength and time to T_OBS. Keywords - spacecraft velocity components OBS_VR, OBS_VW, OBS_VN, solar distance DSUN_OBS, Carrington coordinates CRLT_OBS, CRLN_OBS, and position angle CROTA2 - are linearly interpolated from adjacent same-setting filtergrams. Because each Level-1 image samples a slightly different time, a Wiener-based subpixel interpolation corrects for solar rotation. Two time keywords exist: T_OBS is the midpoint TAI at the spacecraft, and T_REC = T_OBS - (DSUN_OBS - 1 AU)/c is the 1-AU-equivalent time.

#### §2.4 CCD Nonlinearity / CCD 비선형성

**한국어:**
HMI CCD는 12,000 DN 이하에서 약 1 % 수준의 비선형성을 가진다(Wachter+ 2012). 전형적 140 ms 노출은 약 4200 DN이며, Level-1 중앙값은 30,000-50,000 DN/s이다. 비선형 교정은 잉여 강도(실제 - 선형 외삽)를 3차 다항식으로 피팅해 보정한다. 2010년 5월 이후 최초 교정 계수(전방 카메라 -11.08, 0.0174, -2.716e-6, 6.923e-11; 측면 카메라 -8.28, 0.0177, -3.716e-6, 9.014e-11)를 사용했으나, 일부 픽셀이 음의 절편 때문에 비물리적인 음수 값을 가지는 문제로 2014년 1월 15일부터 새 계수(전방 0, 0.0207, -3.187e-6, 8.754e-11; 측면 0, 0.0254, -4.009e-6, 1.061e-10)로 교체되었다. 버전 번호는 CALVER64 키워드로 기록된다.

**English:**
HMI's CCDs exhibit ~1 % nonlinearity below 12,000 DN (Wachter+ 2012). A typical 140 ms exposure gives ~4200 DN, while Level-1 medians are 30,000-50,000 DN/s. The correction subtracts a third-order polynomial fit of (actual - linear) vs. intensity. Initial coefficients applied from May 2010 - front (-11.08, 0.0174, -2.716e-6, 6.923e-11); side (-8.28, 0.0177, -3.716e-6, 9.014e-11) - were replaced on 15 January 2014 with front (0, 0.0207, -3.187e-6, 8.754e-11) and side (0, 0.0254, -4.009e-6, 1.061e-10) to avoid unphysical negative intercepts. The calibration version is stored in the CALVER64 keyword.

#### §2.5 Distortion Correction and Image Alignment / 왜곡 보정과 영상 정렬

**한국어:**
Zernike 다항식 계수(Wachter+ 2012 발사 전 측정)로 광학 왜곡을 보정한다. 평균 잔차 0.043 ± 0.005 pixel, 최대 약 2 pixel(CCD 상·하단 근처)이다. 두 카메라 간 차이는 0.2 pixel 정도. LoS 관측량 한 개에는 약 72장의 Level-1 필터그램이, 벡터 관측량 한 개에는 360장이 사용된다.

정렬 순서: 각 필터그램을 공통 p-각과 B0 각으로 회전 → 태양 차등회전 보정(디스크 중심 부근에서 약 3분에 1 pixel) → 공통 디스크 중심 기준으로 재중심 맞춤. 공간 보간은 **10차 Wiener 보간** (가측 covariance는 이상적 diffraction-limited MTF 기반). HMI 픽셀은 PSF를 ~10 %만큼 언더샘플링하기 때문에 Nyquist 주파수의 약 0.9배까지는 매우 정확하지만 그 이상에서는 부정확하다.

**English:**
Optical distortion is removed using Zernike-polynomial coefficients measured pre-launch (Wachter+ 2012). The mean residual is 0.043 ± 0.005 pixel, with a maximum of ~2 pixels near the top/bottom of the CCD. Inter-camera differences are ~0.2 pixel. One LoS observable requires ~72 Level-1 filtergrams; one vector observable requires 360.

Alignment sequence: each filtergram is rotated to a common p-angle and B0-angle, solar differential rotation is removed (at disk center, rotation shifts features by 1 pixel in ~3 minutes), and all images are re-centered around a common solar disk center. Spatial interpolation uses a separable order-10 Wiener scheme whose covariance is the ideal diffraction-limited MTF. HMI pixels undersample the PSF by ~10 %, so the interpolation is excellent up to ~0.9 times the Nyquist frequency and imperfect above.

#### §2.6 Roll, Absolute Roll Calibration, Distortion, and Venus Transit / 롤각과 금성 통과 교정

**한국어:**
두 카메라 간 상대적 롤각 차이는 일일 캘리브레이션 데이터로 측정되며, 약 0.0837°이다(2010년 5월 - 2015년 9월). 시간에 따라 -0.00020 ± 0.00006°/yr 로 매우 느리게 감소한다. 2012년 6월 5-6일 금성 일면통과(Venus transit)는 절대 롤각과 왜곡 지도를 검증하는 희귀한 기회였다. 해석 결과: 카메라 1은 -0.0142°, 카메라 2는 +0.0709°, 차이 0.0851°로 직접 비교법(0.0837°)과 0.0014° 이내 일치. 왜곡 모델의 잔차는 약 0.1 pixel로 확인됐다.

**English:**
Inter-camera roll differences are determined daily: ~0.0837° on average (May 2010-September 2015), drifting at -0.00020 ± 0.00006°/yr. The 5-6 June 2012 Venus transit provided a rare opportunity to verify the absolute roll and distortion map. Camera 1 had -0.0142°, Camera 2 had +0.0709°, giving a difference of 0.0851° - within 0.0014° of the direct-comparison result (0.0837°). Residuals from the distortion model are ~0.1 pixel.

#### §2.7 Polarization / 편광

**한국어:**
**편광 교정은 여러 단계로 이뤄진다** (polcal 모듈):
1. **변조 행렬**(Schou+ 2012b 모형, 편광 선택 위치·온도 의존, 32 x 32 그리드)을 결정. I → QUV와 QUV → I 항은 0으로, I → I 항은 1로 가정.
2. 각 픽셀에서 최소제곱법으로 **복조 행렬**을 결정. 4096 x 4096 이미지의 복조 행렬은 32 x 32 그리드에서 선형 보간.
3. **망원경 편광 보정**: I에서 Q, U, V로의 소량 누설을 빼는 4차 다항식 피팅(픽셀 중심 거리의 함수). Q, U는 약 14, 18 ppm, V는 28 ppm 잔차. HMI 사양(1000 ppm) 대비 매우 양호.
4. **편광 PSF 보정**: I를 5 x 5 커널로 컨볼루션해 Q, U, V에 더함으로써 granulation-like 패턴 제거. 커널 합 = 0(망원경 편광 보정 교란 방지).

**LoS 편광 오염**(§2.7.3): 전방 카메라는 LCP, RCP만 측정하므로 Q, U 누설을 계산만 하고 빼진 않는다(Q 누설 ≤ 0.003, U 누설 ≤ 0.014). 원래 사양(5 %)보다 훨씬 양호.

**English:**
The polarization calibration (module polcal) proceeds in several stages:
1. A **modulation matrix** (Schou+ 2012b model, function of polarization-selector position and temperature, on a 32 x 32 grid) is determined. The I → QUV and QUV → I terms are set to zero; the I → I term to unity.
2. At each pixel, a least-squares fit determines the **demodulation matrix**. For the 4096 x 4096 image, the demodulation matrix is bilinearly interpolated from the 32 x 32 grid.
3. **Telescope-polarization correction**: a small I leakage into Q, U, V is removed via a fourth-order polynomial fit (function of distance from image center). Residuals: 14, 18 ppm for Q, U; 28 ppm for V - well below the 1000 ppm specification.
4. **Polarization-PSF correction**: I is convolved with a 5 x 5 kernel and added to Q, U, V to remove a granulation-like pattern. The kernel sum is zero to avoid contaminating the telescope-polarization correction.

**LoS polarization contamination** (§2.7.3): the front camera measures only LCP and RCP, so Q, U leaks into I ± V are calculated but not corrected (Q leak <= 0.003, U leak <= 0.014) - well below the original 5 % specification.

#### §2.8 Filters / 필터 시스템

**한국어:**
HMI 필터는 입력창, broad-band blocking filter, 다단 Lyot 필터(최 narrowest 요소 E1은 조정 가능), 두 조정형 Michelson 간섭계(narrowband NB, wideband WB)로 구성된다. 투과는
$$T(\lambda) = \frac{1 + B \cos(2\pi\lambda/\mathrm{FSR} + \Phi + 4\phi)}{2}$$
로 모형화되며, FSR은 free spectral range, φ는 tuning motor 위상, Φ는 필터 고유 위상이다. **위상 맵**은 격주 detune sequence를 통해 128 x 128 그리드로 결정한다.

**CalMode fringe 제거**(§2.8.2): front window가 다층 유리 구조로 약한 Fabry-Pérot 간섭계 역할을 해 관측 시에는 보이지 않는 CalMode 줄무늬가 phase map에 섞인다. A_C cos(φ) + A_S sin(φ) 항으로 모형화해 SVD로 φ를 초기 추정, 이후 alternating fit으로 제거.

**I-ripple**(§2.8.3): 조정 필터의 waveplate/편광소자 결함으로 투과 강도가 tuning에 따라 변한다. 수식:
$$I(\lambda)/\bar{I}(\lambda) = K_0 + [K_1 \cos(\phi/2) + K_2 \sin(\phi/2)]^2$$
피크-투-피크 진폭은 NB ~0.006 → 0.008, WB ~0.013 → 0.011, E1 ~0.003 → 0.008(약 1500일 동안)로 변한다. 현재 I-ripple은 교정되지 않으며, 시뮬레이션상 수십 m/s 수준의 속도 오차가 SDO 속도와 선형으로 변한다.

필터 드리프트: NB Michelson ~6 mÅ/yr, WB Michelson ~30 mÅ/yr, E1 Lyot <7 mÅ/yr. 연간 속도 영점이 약 100-200 m/s 이동한다. 따라서 HMI는 주기적으로 재조정되며(Table 5: 2010.04.30, 2010.12.13, 2011.07.13, 2012.01.18, 2013.03.14, 2014.01.15, 2015.04.08, 2016.04.27), 이는 CALVER64 버전에 반영된다.

**English:**
The HMI filter system comprises an entrance window, broad-band blocking filter, multi-stage Lyot filter (with tunable E1 element), and two tunable Michelson interferometers (narrowband NB and wideband WB). The transmittance is
$$T(\lambda) = \frac{1 + B \cos(2\pi\lambda/\mathrm{FSR} + \Phi + 4\phi)}{2}$$
where FSR is the free spectral range, phi the tuning-motor phase, and Phi the filter-specific phase. **Phase maps** are computed every two weeks from detune sequences on a 128 x 128 grid.

**CalMode fringe removal** (§2.8.2): HMI's front window is a multi-layer glass-glue sandwich acting as a weak Fabry-Perot, producing fringes imaged onto the CCD in calibration mode (CalMode) that contaminate the phase maps. The fringes are modelled with cos(phi) and sin(phi) terms, fit by SVD followed by alternating least-squares.

**I-ripple** (§2.8.3): tunable-filter imperfections (waveplate misalignments etc.) cause a tuning-dependent intensity variation
$$I(\lambda)/\bar{I}(\lambda) = K_0 + [K_1 \cos(\phi/2) + K_2 \sin(\phi/2)]^2.$$
Over ~1500 days the peak-to-peak amplitudes grew/decayed: NB ~0.006 → 0.008, WB ~0.013 → 0.011, E1 ~0.003 → 0.008. I-ripple is not corrected; simulations indicate a zero-point velocity error of a few tens of m/s that varies linearly with Sun-SDO velocity.

Filter drift: NB Michelson ~6 mÅ/yr, WB Michelson ~30 mÅ/yr, E1 Lyot <7 mÅ/yr. The annual velocity zero-point shifts by ~100-200 m/s. HMI is retuned whenever the offset exceeds a few hundred m/s (Table 5: 2010.04.30, 2010.12.13, 2011.07.13, 2012.01.18, 2013.03.14, 2014.01.15, 2015.04.08, 2016.04.27). The CALVER64 keyword records the calibration version.

#### §2.9 MDI-like Algorithm / MDI 유사 알고리즘

**한국어:**
**핵심 아이디어**: MDI(Scherrer+ 1995)는 4개의 등간격 파장에서 Ni I 선을 샘플링했고, 투과 프로파일 FWHM = 선 FWHM로 설계되어 첫 번째 Fourier 계수만으로 속도를 추정할 수 있었다. HMI는 6개 샘플을 사용하지만 SDO 궤도로 인한 큰 속도 변화를 수용하기 위해 선 FWHM의 2배가 아닌 6x68.8 = 412.8 mÅ = T의 간격을 사용한다. 따라서 첫 번째와 두 번째 Fourier 계수 모두 필요하다.

**식 (4), (5)**: 이상 연속 프로파일 I(λ)의 Fourier 계수(식 번호는 원문)
$$a_n = \frac{2}{T}\int_{-T/2}^{+T/2} I(\lambda) \cos\!\left(2\pi n \frac{\lambda}{T}\right) d\lambda, \quad b_n = \frac{2}{T}\int_{-T/2}^{+T/2} I(\lambda) \sin\!\left(2\pi n \frac{\lambda}{T}\right) d\lambda.$$

**식 (6)**: Fe I 선을 Gaussian으로 가정
$$I(\lambda) = I_c - I_d \exp\!\left[-\frac{(\lambda-\lambda_0)^2}{\sigma^2}\right].$$

**식 (7), (8)**: 도플러 속도
$$v = \frac{dv}{d\lambda}\frac{T}{2\pi}\arctan\!\left(\frac{b_1}{a_1}\right), \qquad v_2 = \frac{dv}{d\lambda}\frac{T}{4\pi}\arctan\!\left(\frac{b_2}{a_2}\right),$$
여기서 dv/dλ = c/λ_0 = 299792458.0 m/s / 6173.3433 Å = 48562.4 m/s/Å.

**식 (9), (10)**: 선깊이, 선폭
$$I_d = \frac{T}{2\sigma\sqrt{\pi}}\sqrt{a_1^2+b_1^2}\exp\!\left(\frac{\pi^2\sigma^2}{T^2}\right), \qquad \sigma = \frac{T}{\pi\sqrt{6}}\sqrt{\log\!\left(\frac{a_1^2+b_1^2}{a_2^2+b_2^2}\right)}.$$

**식 (11)**: HMI는 연속 적분 대신 6 샘플의 이산합
$$a_1 \approx \frac{2}{6}\sum_{j=0}^{5} I_j \cos\!\left(2\pi\frac{2.5-j}{6}\right), \quad b_1 \text{ analogous}.$$

이 합은 LCP(I+V)와 RCP(I-V) 각각에 대해 별도 계산되어 v_LCP, v_RCP를 얻는다.

**식 (12), (13)**: 결합 속도와 LoS 자기장 추정
$$V = \frac{V_{\mathrm{LCP}} + V_{\mathrm{RCP}}}{2}, \qquad B = (V_{\mathrm{LCP}} - V_{\mathrm{RCP}}) K_m,$$
여기서 K_m = 1.0 / (2.0 x 4.67 x 10^{-5} λ_0 g_L c) = 0.231 G m^{-1} s, g_L = 2.5.

**식 (14)**: 연속 강도 재구성
$$I_c \approx \frac{1}{6}\sum_{j=0}^{5}\left[I_j + I_d \exp\!\left(-\frac{(\lambda_j-\lambda_0)^2}{\sigma^2}\right)\right].$$

**룩업 테이블(LUT)** (§2.9.2): 이상 Gaussian 가정이 실제 Fe I 선 및 유한한 필터 투과 프로파일과 다르므로, 사전 시뮬레이션으로 input velocity → output (Fourier) velocity 매핑을 계산. 24 m/s 단위로 ±9840 m/s 범위(1642 값)를 128 x 128 그리드에 저장. 가능한 속도 기여도: SDO 궤도 3500 m/s + 태양 자전 2000 m/s + 그래뉼/p-mode 1000 m/s + Zeeman 3400 m/s ≈ 9900 m/s.

**보정 계수**: Gaussian 테스트로부터 I_d는 K_1 = 5/6로, σ는 K_2 = 6/5로 스케일(둘의 곱은 I_d x σ로 Gaussian 적분이 불변). σ가 강자기장 근방에서 불안정하면, 저활동기 선폭 지도의 5차 방사 다항식 fit으로 대체한 후 K_1로 보정.

**라인 프로파일 모델**(§2.9.3): Voigt + 두 Gaussian(라인 비대칭)
$$I = I_g - d_g\exp(-l^2)\left(1 - \frac{a}{\sqrt{\pi}l^2}\left[(4l^2+3)(l^2+1)\exp(-l^2) - \frac{2l^2+3}{l^2}\sinh(l^2)\right]\right) - A\exp(-(\lambda+B)^2/C^2) + D\exp(-(\lambda-E)^2/F^2),$$
여기서 l = λ/w_g, |l| ≤ 26.5. Table 4에 Calibration 11, 12, 13 계수. FSR: NB 168.9 mÅ, WB 336.85 mÅ, E1 1417, E2 2779, E3 5682, E4 11354 mÅ.

**English:**
**Key idea**: MDI (Scherrer+ 1995) sampled the Ni I line at 4 equally spaced wavelengths with filter FWHM matched to the line FWHM, so that the first Fourier coefficient alone sufficed to estimate velocity. HMI uses 6 samples spanning 412.8 mÅ = 6 x 68.8 mÅ (T) - six times the line FWHM - to accommodate SDO's large velocity variations, requiring both first and second Fourier coefficients.

**Eqs. (4), (5)**: Fourier coefficients of the continuous profile
$$a_n = \frac{2}{T}\int_{-T/2}^{+T/2} I(\lambda) \cos\!\left(2\pi n \frac{\lambda}{T}\right) d\lambda, \quad b_n = \frac{2}{T}\int_{-T/2}^{+T/2} I(\lambda) \sin\!\left(2\pi n \frac{\lambda}{T}\right) d\lambda.$$

**Eq. (6)**: Gaussian Fe I profile
$$I(\lambda) = I_c - I_d \exp\!\left[-\frac{(\lambda-\lambda_0)^2}{\sigma^2}\right].$$

**Eqs. (7), (8)**: Doppler velocities from first/second coefficients
$$v = \frac{dv}{d\lambda}\frac{T}{2\pi}\arctan\!\left(\frac{b_1}{a_1}\right), \qquad v_2 = \frac{dv}{d\lambda}\frac{T}{4\pi}\arctan\!\left(\frac{b_2}{a_2}\right),$$
with dv/dλ = c/λ_0 = 299792458.0 m/s / 6173.3433 Å = 48562.4 m/s/Å.

**Eqs. (9), (10)**: line depth, line width
$$I_d = \frac{T}{2\sigma\sqrt{\pi}}\sqrt{a_1^2+b_1^2}\exp\!\left(\frac{\pi^2\sigma^2}{T^2}\right), \qquad \sigma = \frac{T}{\pi\sqrt{6}}\sqrt{\log\!\left(\frac{a_1^2+b_1^2}{a_2^2+b_2^2}\right)}.$$

**Eq. (11)**: HMI's discrete 6-sample estimate
$$a_1 \approx \frac{2}{6}\sum_{j=0}^{5} I_j \cos\!\left(2\pi\frac{2.5-j}{6}\right),$$
with b_1 analogous. Separate sums are computed for LCP (I+V) and RCP (I-V), producing v_LCP and v_RCP.

**Eqs. (12), (13)**: combined velocity and LoS field
$$V = \frac{V_{\mathrm{LCP}} + V_{\mathrm{RCP}}}{2}, \qquad B = (V_{\mathrm{LCP}} - V_{\mathrm{RCP}}) K_m,$$
where K_m = 1.0 / (2.0 x 4.67 x 10^{-5} lambda_0 g_L c) = 0.231 G m^{-1} s, g_L = 2.5.

**Eq. (14)**: reconstructed continuum intensity
$$I_c \approx \frac{1}{6}\sum_{j=0}^{5}\left[I_j + I_d \exp\!\left(-\frac{(\lambda_j-\lambda_0)^2}{\sigma^2}\right)\right].$$

**Look-up tables (LUTs)** (§2.9.2): because the Gaussian assumption and finite filter transmittance distort the ideal mapping, LUTs pre-simulate input -> output velocity on a 128 x 128 grid with ±9840 m/s range in 24 m/s steps (1642 entries). Velocity budget: ~3500 m/s SDO orbit + ~2000 m/s solar rotation + ~1000 m/s granulation/p-modes + ~3400 m/s Zeeman splitting.

**Scale factors**: K_1 = 5/6 for I_d, K_2 = 6/5 for sigma (their product preserves the Gaussian integral). When sigma is noisy in strong-field regions, it is replaced by a fifth-order radial polynomial fit from a low-activity period.

**Line profile model** (§2.9.3): Voigt + two Gaussians for asymmetry
$$I = I_g - d_g\exp(-l^2)\left(1 - \frac{a}{\sqrt{\pi}l^2}\left[(4l^2+3)(l^2+1)\exp(-l^2) - \frac{2l^2+3}{l^2}\sinh(l^2)\right]\right) - A\exp(-(\lambda+B)^2/C^2) + D\exp(-(\lambda-E)^2/F^2),$$
with l = lambda/w_g, |l| <= 26.5. Table 4 lists Calibration 11, 12, 13 coefficients. FSRs: NB 168.9 mÅ, WB 336.85 mÅ, E1 1417, E2 2779, E3 5682, E4 11354 mÅ.

#### §2.10 Polynomial Velocity Correction / 다항식 속도 보정

**한국어:**
HMI 도플러 교정 후에도 SDO 궤도 속도에 비례하는 잔차가 남는다. 24시간 구간에 대해 전 디스크 중앙값 RAWMEDN과 알려진 OBS_VR의 차이를 3차 다항식으로 피팅:
$$\mathrm{RAWMEDN} - \mathrm{OBS\_VR} = C_0 + C_1 \mathrm{RAWMEDN} + C_2 \mathrm{RAWMEDN}^2 + C_3 \mathrm{RAWMEDN}^3$$
(식 16). 잔차 변동성은 5 m/s 이하이지만 일변화는 15-60 m/s이다. C_0는 NB Michelson FSR 드리프트로 인해 시간에 따라 증가하며, 재조정 시 점프한다. 이 보정은 v_LCP, v_RCP에 적용된 후 V와 B가 재계산된다(식 17).

**English:**
After HMI Doppler calibration a residual proportional to SDO orbital velocity remains. A 24-hour segment fit of the difference between full-disk median velocity RAWMEDN and the known OBS_VR gives
$$\mathrm{RAWMEDN} - \mathrm{OBS\_VR} = C_0 + C_1 \mathrm{RAWMEDN} + C_2 \mathrm{RAWMEDN}^2 + C_3 \mathrm{RAWMEDN}^3 \quad \text{(Eq. 16)}.$$
The residual variability is < 5 m/s, but the daily variation is 15-60 m/s. C_0 grows with time due to NB Michelson FSR drift and jumps at each retuning. The correction is applied to v_LCP and v_RCP before re-deriving V and B (Eq. 17).

### §3 Performance, Error Estimates, and Impact on Observables / 성능·오차 (pp. 1919-1932)

#### §3.1 Sensitivity of Doppler Velocity to Roll / 롤 민감도

**한국어:**
매년 4월, 10월에 HMI를 360° 회전하는 롤 캘리브레이션이 수행된다. 이상적 교정 환경이라면 중앙값 속도가 롤각 무관해야 하지만, 실제로는 5-10 m/s, 180° 주기의 체계적 변동이 있다. 이는 필터의 파장 특성이 롤에 따라 미세하게 변하는 효과를 반영한다.

**English:**
In April and October each year HMI performs a 360° roll calibration. Ideally the median velocity should be independent of roll, but systematic variations of 5-10 m/s with 180° periodicity persist, reflecting a small wavelength sensitivity of the filter to roll.

#### §3.2 24 h Variations in Observables / 24시간 변동

**한국어:**
HMI의 모든 관측량에서 24시간 주기 변동이 관찰된다. 주원인은 SDO 궤도 속도(~±3500 m/s 일변화)이다. Hoeksema+ 2014에 따르면 2500 G 장에서 ±3 m/s의 OBS_VR 변화가 ±48 G의 자기장 변화를 일으킨다. 암부(2500 G): 일변화 진폭 5 % 미만, 반그림자(1300 G): 중간, 정온 태양(100 G): 무시할 만큼. ME-B_LoS와 MDI-like B_LoS 모두에서 속도 의존 변동이 뚜렷하다. 횡단 자기장 ME-B_trans은 암부에서 속도 민감도 없지만 정온 영역에서는 영 속도 부근에서 복잡한 구조.

**English:**
All HMI observables exhibit 24-hour variations, driven primarily by SDO's ±3500 m/s daily orbital velocity. Hoeksema+ 2014 found that at 2500 G, the field changes by ±48 G for ±3 m/s of OBS_VR. Umbra (2500 G): <5 % daily amplitude; penumbra (1300 G): intermediate; quiet Sun (100 G): negligible. Both ME-B_LoS (inversion) and B_LoS (MDI-like) show velocity-dependent daily variation. Transverse ME-B_trans is insensitive in umbra but shows complex behavior near zero velocity in quiet Sun.

#### §3.3 Errors with the LoS Algorithm / LoS 알고리즘 오차

**한국어:**
강한 자기장 영역에서 Fe I 선은 Zeeman splitting으로 LCP와 RCP 프로파일이 파장 축에서 크게 분리된다. 6 파장 샘플링으로는 한쪽 원편광 성분이 HMI 동적 영역(±412.8 mÅ/2) 밖으로 벗어날 수 있으며, MDI-like 알고리즘은 이를 포화시키거나 과소평가한다. 2014년 10월 24일 NOAA AR 12192에 대한 10파장 특별 관측이 이를 정량적으로 보여준다: 6파장 선폭은 암부로 갈수록 감소(정상과 반대), LCP 속도는 과포화, RCP는 과소평가. **결론**: 강자기장에서 MDI-like LoS 자기장은 체계적으로 과소추정된다.

**English:**
In strong fields, Zeeman splitting separates the LCP and RCP profiles significantly in wavelength. HMI's 6-wavelength sampling (±412.8 mÅ/2 dynamic range) can push one circular-polarization component outside the sampled range, causing the MDI-like algorithm to saturate or underestimate. A special 10-wavelength sequence on 24 October 2014 (NOAA AR 12192) showed: the 6-wavelength line width decreases toward the umbra (opposite of expected), LCP velocity saturates, RCP is underestimated. **Conclusion**: strong-field LoS magnetograms from the MDI-like algorithm are systematically under-estimated.

#### §3.4 Magnetic-Field Error with Stokes-Vector Inversion / Stokes 벡터 인버전의 오차

**한국어:**
VFISV(Borrero+ 2011; Centeno+ 2014)는 Milne-Eddington 대기에서 Stokes 프로파일을 fitting해 벡터 자기장을 반전한다. ME 가정(물리량이 광학 깊이 무관)과 6 파장의 제한된 샘플링이 핵심 한계다. 10파장 시퀀스로 얻은 "진실값"과 비교하면, 6파장 VFISV는 암부 중심에서 LoS 자기장을 더 음의 방향으로 과대 산출한다(LoS field 절대값 과대평가). 이 부분은 ±100 G 수준의 체계적 오차를 시사한다.

**English:**
VFISV (Borrero+ 2011; Centeno+ 2014) inverts the vector field by fitting the Stokes profiles in a Milne-Eddington atmosphere. Its main limitations are the ME assumption (no depth dependence except the source function) and the 6-wavelength sampling. Comparing 6- and 10-wavelength inversions for AR 12192, the 6-wavelength VFISV gives a more-negative LoS-field in umbra, i.e. it overestimates |B_LoS| by ~100 G in the strongest regions.

#### §3.5 Temperature Dependence of CCD Gain / CCD 이득 온도 의존성

**한국어:**
HMI CCD 이득은 온도에 의존해 DATAMEDN이 1 K 상승 시 0.25 % 감소한다(기울기 ≈ -0.0025 K^-1). CCD 온도 일변화는 3 K 이하, 연변화는 1-2 K. 속도·자기장 관측량은 강도 차에 기반하므로 영향은 작지만, 강도 관측량에는 직접적. 온도 보정이 곧 구현될 예정이다.

**English:**
HMI CCD gain varies with temperature: DATAMEDN falls 0.25 % per 1 K rise (slope ~-0.0025 K^-1). Daily CCD-temperature excursions are <3 K; annual variations are 1-2 K. Velocity and magnetic observables (based on intensity differences) are little affected, but intensity observables are directly impacted. A correction is planned.

#### §3.6 Correcting the HMI Point Spread Function / PSF 보정

**한국어:**
이상 diffraction-limited OTF: OTF(ρ') = (2/π)[acos(ρ') - ρ' sqrt(1-ρ'^2)], ρ' = fλ/(PD)ρ ≈ 1.82ρ (D = 140 mm, f = 4953 mm, λ = 6173 Å, P = 12 μm). 실제 OTF는 이상값 x exp(-πρ'/γ), γ = 4.5(금성 통과에서 측정). PSF는 Airy 함수와 Lorentzian의 컨볼루션으로 모형화되며, 장거리 산란은 2010년 10월 7일 달 식으로 특성화(달 중심에서 200 pixel 떨어진 곳의 산란광이 디스크 중심 연속 강도의 0.34 %). 실제 PSF는 다음 형태:
$$\mathrm{PSF}(r) = \mathcal{F}(\mathrm{OTF}) + c \exp\!\left(-\frac{\pi r}{\xi r_{\max}}\right)$$
c = 2 x 10^{-9}, ξ = 0.7, r_max = 2048. Richardson-Lucy 반복으로 GPU 상에서 디컨볼루션. 흑점 주변 granulation 대비는 3.7 % → 7.2 %로 증가, 암부 최소 강도는 5.5 % → 3.3 %로 감소.

**English:**
The ideal diffraction-limited OTF is OTF(rho') = (2/pi)[acos(rho') - rho' sqrt(1-rho'^2)], with rho' = f lambda/(PD) rho ~= 1.82 rho (D = 140 mm, f = 4953 mm, lambda = 6173 Å, P = 12 micron). The actual OTF is approximated as the ideal multiplied by exp(-pi rho'/gamma) with gamma = 4.5 (fit from Venus transit). The PSF is the convolution of an Airy pattern with a Lorentzian; long-distance scattering is characterized from the 7 October 2010 lunar eclipse (scattered light 0.34 % of disk-center continuum 200 pixels inside the Moon). The composite PSF is
$$\mathrm{PSF}(r) = \mathcal{F}(\mathrm{OTF}) + c \exp\!\left(-\frac{\pi r}{\xi r_{\max}}\right),$$
c = 2 x 10^-9, xi = 0.7, r_max = 2048. Richardson-Lucy deconvolution on a GPU recovers contrast: granulation RMS around a sunspot grows from 3.7 % to 7.2 %, and umbra minimum intensity drops from 5.5 % to 3.3 %.

### §4 Summary and Conclusion / 요약 및 결론 (pp. 1932-1935)

**한국어:**
HMI는 3.75초마다 2장의 4096 x 4096 필터그램을 5년간 연속 수집했으며, 설계 사양을 모두 충족한다. 잔여 문제는 다음과 같다:
1. **SDO 궤도 속도의 일변동 효과**(가장 큰 미해결 교정 이슈): LoS 속도는 다항식 보정, 자기장은 보정 없음.
2. **강자기장 영역 포화/과소추정**(MDI-like) 및 VFISV 과대추정(100 G).
3. **편광 원천** 미확인(망원경 편광의 근본 원인).
4. **CCD 온도 의존성** 보정 미구현.
5. **스트레이 라이트/PSF 보정** 정제 필요.

2016년 4월 13일 벡터 카메라 framelist를 90초로 단축하는 개선이 이뤄졌다.

**English:**
HMI has acquired two 4096 x 4096 filtergrams every 3.75 s for five years, meeting all specifications. Remaining issues:
1. **SDO orbital velocity daily effects** (largest unresolved calibration issue): LoS velocity gets the polynomial correction; magnetic field has none.
2. **Strong-field saturation/underestimation** (MDI-like) and VFISV over-estimation (~100 G).
3. **Source of telescope polarization** still unidentified.
4. **CCD temperature gain correction** not yet implemented.
5. **Stray-light/PSF correction** needs refinement.

On 13 April 2016 the vector-camera framelist was shortened from 135 s to 90 s.

## Key Takeaways / 핵심 시사점

### 1. HMI = MDI의 고해상도·고속 후계자 / HMI as MDI's high-resolution successor

**한국어:** HMI는 SOHO/MDI를 구조적으로 계승하지만, 4096 x 4096 CCD, 45초 cadence, 전체 Stokes 벡터 측정, 두 카메라 병행 운용으로 1000배 이상의 데이터 규모를 달성했다. 동시에 MDI(4 파장, 선 FWHM의 2배 span)와 달리 HMI(6 파장, 선 FWHM의 6배 span)는 SDO 궤도 속도를 수용하기 위한 더 큰 동적 영역이 필요했다.

**English:** HMI structurally inherits SOHO/MDI but scales up three orders of magnitude: 4096 x 4096 CCD, 45-s cadence, full-Stokes measurement, and two cameras in parallel. Unlike MDI (4 wavelengths spanning 2 x FWHM), HMI uses 6 wavelengths spanning 6 x FWHM = 412.8 mÅ to accommodate SDO's large orbital-velocity range.

### 2. MDI-like 알고리즘은 이산 Fourier 근사에 의존 / MDI-like algorithm relies on discrete Fourier approximation

**한국어:** 식 (4)-(14)의 모든 도플러/자기장/선깊이/선폭은 6개 이산 파장 샘플에서 계산한 a_1, b_1, a_2, b_2 Fourier 계수로부터 유도된다. Fe I 선이 실제로는 Gaussian이 아니고, 필터 투과 프로파일이 δ 함수가 아니며, 샘플이 6개뿐이라는 세 가지 이유로 원시 Fourier 속도는 실제 속도와 다르다. 이 차이를 룩업 테이블이 해소한다.

**English:** All Doppler/magnetic-field/line-depth/line-width quantities in Eqs. (4)-(14) flow from the discrete Fourier coefficients a_1, b_1, a_2, b_2 of the 6 wavelength samples. The raw Fourier velocity differs from the true velocity because (1) the Fe I line is not Gaussian, (2) the filter transmittance is not delta-functions, and (3) only 6 samples are available. Look-up tables bridge this gap.

### 3. Zeeman splitting → LoS B 환산 계수 / Zeeman splitting -> LoS B conversion factor

**한국어:** 식 (13)의 K_m = 0.231 G m^{-1} s는 Landé g_L = 2.5와 파장 6173 Å에 대한 상수이다. 1000 m/s의 LCP-RCP 속도 차이는 약 231 G의 LoS 자기장에 해당한다. HMI 광자 잡음 한계는 45초에 7 G, 720초 평균에 3 G로, 약 30-100 G의 자기장을 신뢰 있게 측정할 수 있다.

**English:** The constant K_m = 0.231 G m^{-1} s in Eq. (13) follows from the Landé factor g_L = 2.5 and wavelength 6173 Å. A 1000 m/s LCP-RCP velocity difference corresponds to ~231 G of LoS field. Photon-noise floors are 7 G at 45 s and 3 G at 720 s, enabling reliable measurements of ~30-100 G fields.

### 4. 편광 교정은 다단계 디컨볼루션 과정 / Polarization calibration is a multi-stage deconvolution

**한국어:** 변조 행렬 → 복조 행렬 보간 → 망원경 편광 다항식 제거 → 편광 PSF 5 x 5 커널 콘볼루션으로 4096 x 4096 이미지 한 장당 네 단계의 편광 처리가 필요하다. 망원경 편광 잔차는 원래 사양(1000 ppm)보다 두 자릿수 이상 개선된 14-28 ppm이다.

**English:** The four-step polarization pipeline - modulation matrix → interpolated demodulation → polynomial telescope-polarization removal → 5 x 5 polarization-PSF kernel - reduces the telescope-polarization residual to 14-28 ppm, two orders of magnitude better than the 1000 ppm specification.

### 5. 강자기장에서 MDI-like 알고리즘은 포화 / Strong-field saturation of the MDI-like algorithm

**한국어:** 6 파장 샘플링의 동적 영역 한계 때문에 암부(B > 2000 G)에서 LoS 자기장이 체계적으로 과소추정된다. 10 파장 시퀀스는 이를 해결하지만 cadence를 75초로 늦추고 S/N을 저하시킨다. 이는 MDI-like 알고리즘의 본질적 한계로, 향후 개선 방향이다.

**English:** The 6-wavelength sampling's limited dynamic range systematically under-estimates the LoS field in strong-field umbrae (B > 2000 G). A 10-wavelength sequence resolves this at the cost of 75-s cadence and reduced S/N - an intrinsic limitation of the MDI-like method and a target for future upgrades.

### 6. 궤도 속도가 모든 관측량에 24시간 변동 유발 / Orbital velocity drives 24-hour variations in all observables

**한국어:** SDO의 지구동기궤도는 ±3500 m/s 일변화를 일으키며, 파장 이동으로 인해 LoS 속도(15-60 m/s 잔차), 자기장(암부 ±50 G), 강도 모두에서 24시간 시그널이 나타난다. 다항식 보정이 일부 완화하지만 여전히 가장 큰 미해결 교정 이슈다.

**English:** SDO's geosynchronous orbit imposes ±3500 m/s daily velocity excursions, producing 24-hour signatures in every observable: LoS velocity (15-60 m/s residuals), magnetic field (±50 G in umbrae), and intensity. The polynomial correction helps but remains the largest unresolved calibration issue.

### 7. 필터 요소의 장기 드리프트 → 주기적 재조정 / Filter-element drift → periodic retuning

**한국어:** NB Michelson은 6 mÅ/yr, WB는 30 mÅ/yr, E1은 <7 mÅ/yr 드리프트한다. 이는 매년 100-200 m/s의 속도 영점 이동에 해당해 약 1-2년 주기로 기기 재조정이 이루어지며(Table 5의 7-8회), 이에 맞춰 calibration version(11, 12, 13)이 변경된다.

**English:** The NB Michelson drifts at 6 mÅ/yr, the WB at 30 mÅ/yr, and the E1 Lyot at <7 mÅ/yr. These shifts translate to 100-200 m/s annual zero-point drifts, so HMI is retuned approximately every 1-2 years (7-8 events in Table 5) and the calibration version (11, 12, 13) is updated accordingly.

### 8. 두 카메라 파이프라인 = 과학적 분업 / Two-camera parallel pipelines = scientific specialization

**한국어:** 전방 카메라(front)는 LCP/RCP만 측정해 45초 cadence의 helioseismology·실시간 자기장 감시를, 측면 카메라(side)는 전체 Stokes 측정으로 720초 벡터 자기장 반전을 지원한다. 한 대씩 특화시킴으로써 LoS 고속 관측과 벡터 고정밀 관측을 동시에 달성한다.

**English:** The front camera measures only LCP/RCP for 45-s helioseismology and real-time magnetic monitoring; the side camera performs full-Stokes polarimetry for 720-s vector field inversion. Specializing one camera for each mode enables simultaneous fast-LoS and high-precision vector observation.

## Mathematical Summary / 수학적 요약

### Observation Equations / 관측 방정식

**Stokes-based filter signal (LCP/RCP decomposition):**
$$I_\mathrm{LCP} = I + V, \qquad I_\mathrm{RCP} = I - V.$$

**Filter transmittance of a tunable element:**
$$T(\lambda) = \frac{1 + B\cos(2\pi\lambda/\mathrm{FSR} + \Phi + 4\phi)}{2}.$$

### MDI-like Fourier Reconstruction / MDI 유사 Fourier 복원

**Discrete first Fourier coefficients over 6 samples (Eq. 11):**
$$a_1 \approx \tfrac{1}{3}\sum_{j=0}^{5} I_j \cos\!\left(\frac{\pi(2.5-j)}{3}\right), \quad b_1 \approx \tfrac{1}{3}\sum_{j=0}^{5} I_j \sin\!\left(\frac{\pi(2.5-j)}{3}\right),$$
$$a_2 \approx \tfrac{1}{3}\sum_{j=0}^{5} I_j \cos\!\left(\frac{2\pi(2.5-j)}{3}\right), \quad b_2 \approx \tfrac{1}{3}\sum_{j=0}^{5} I_j \sin\!\left(\frac{2\pi(2.5-j)}{3}\right).$$

**Doppler velocity (Eq. 7):**
$$v = \frac{c}{\lambda_0}\frac{T}{2\pi}\arctan\!\left(\frac{b_1}{a_1}\right),$$
with T = 412.8 mÅ, c/λ_0 = 48562.4 m/s/Å.

**Line width and depth (Eqs. 9, 10):**
$$\sigma = \frac{T}{\pi\sqrt{6}}\sqrt{\log\!\left(\frac{a_1^2+b_1^2}{a_2^2+b_2^2}\right)}, \qquad I_d = \frac{T}{2\sigma\sqrt{\pi}}\sqrt{a_1^2+b_1^2}\exp\!\left(\frac{\pi^2\sigma^2}{T^2}\right).$$

**Combined velocity and LoS field (Eqs. 12, 13):**
$$V = \tfrac{1}{2}(V_\mathrm{LCP} + V_\mathrm{RCP}), \qquad B = K_m (V_\mathrm{LCP} - V_\mathrm{RCP}),$$
$$K_m = \frac{1}{2 \times 4.67\times10^{-5}\lambda_0 g_L c} = 0.231 \text{ G m}^{-1}\text{s}, \quad g_L = 2.5.$$

**Reconstructed continuum (Eq. 14):**
$$I_c \approx \tfrac{1}{6}\sum_{j=0}^{5}\!\left[I_j + I_d \exp\!\left(-\frac{(\lambda_j-\lambda_0)^2}{\sigma^2}\right)\right].$$

### Polynomial Velocity Correction (Eqs. 16, 17) / 다항식 속도 보정

$$\mathrm{RAWMEDN} - \mathrm{OBS\_VR} = C_0 + C_1 \mathrm{RAWMEDN} + C_2 \mathrm{RAWMEDN}^2 + C_3 \mathrm{RAWMEDN}^3,$$
$$v_\mathrm{LCP}' = V_\mathrm{LCP} - (C_0' + C_1' V_\mathrm{LCP} + C_2' V_\mathrm{LCP}^2 + C_3' V_\mathrm{LCP}^3).$$

### Worked Numerical Example / 수치 예제

**Goal**: compute LoS B from a simulated 6-wavelength measurement with v_LCP = +100 m/s, v_RCP = -100 m/s.

1. Δv = v_LCP - v_RCP = 200 m/s.
2. B = K_m Δv = 0.231 G m^-1 s x 200 m/s = **46.2 G**.
3. V = (v_LCP + v_RCP)/2 = 0 m/s (no bulk Doppler shift).

**Scaling check**: the 45-s photon-noise floor is 7 G. A measured field of 46.2 G is therefore a ~6.6 sigma detection, consistent with measurable weak-field quiet-Sun magnetic network. For 720-s averages (noise 3 G), the same field is a ~15.4 sigma detection.

### OTF & PSF / OTF와 PSF

$$\mathrm{OTF}_\mathrm{ideal}(\rho') = \tfrac{2}{\pi}\left[\arccos\rho' - \rho'\sqrt{1-\rho'^2}\right], \quad \rho' = \tfrac{f\lambda}{PD}\rho \approx 1.82\rho,$$
with D = 140 mm, f = 4953 mm, lambda = 6173 Å, P = 12 micron.

**Empirical tail fit:** OTF(rho) = OTF_ideal(rho) exp(-pi rho'/gamma), gamma = 4.5.

## Paper in the Arc of History / 역사 속의 논문

```
         MDI (Scherrer+ 1995) SOHO
                    |
                    | Ni I 6768 Å, 4 wavelengths, 1024 x 1024, 96 min full-disk cadence
                    v
  +--------------------------------------+
  | HMI Design Phase (2000s)             |
  |  Norton+ 2006: Fe I 6173 vs Ni I     |
  |  Schou+ 2012a: ground calibration    |
  |  Wachter+ 2012: image quality        |
  |  Schou+ 2012b: polarization cal      |
  |  Couvidat+ 2012a,b: phase maps       |
  +------------------|-------------------+
                     | 2010 May: SDO launch, HMI prime mission
                     v
  +--------------------------------------+
  | HMI Prime Mission (2010-2015)        |
  |  Martinez Oliveros+ 2011: interp.    |
  |  Liu+ 2012: MDI vs HMI comparison    |
  |  Borrero+ 2011, Centeno+ 2014: VFISV |
  |  Bobra+ 2014: SHARPs                 |
  |  Hoeksema+ 2014: vector pipeline     |
  +------------------|-------------------+
                     | 2016: this paper consolidates
                     v
  ** Couvidat+ 2016 (THIS PAPER) **
     Observables processing pipeline reference
                     |
                     v
  Later: Schuck+ 2016 (Doppler consistency),
         Bush+ 2016 (Level-1 performance),
         GONG/SOLIS/EST/Aditya-L1 calibration
         heritage, DKIST polarimetric pipelines
```

**한국어:** 이 논문은 HMI 프라임 미션의 끝에 발표되어 5년간 축적된 교정 경험을 정리하며, 이후 관측(확장 미션 및 후속 태양 편광계)의 파이프라인 설계 참고자료가 된다.

**English:** Published at the end of the HMI prime mission, this paper consolidates five years of calibration experience and becomes a reference for subsequent observatory pipelines (HMI extended mission and next-generation solar polarimeters).

## Connections to Other Papers / 다른 논문과의 연결

| Paper | 관계 / Relation | 핵심 요점 / Key point |
|-------|-----------------|----------------------|
| **Scherrer+ 1995 (MDI)** | 직접 선행 / direct predecessor | MDI-like 알고리즘의 Fourier 기반 속도 추정 원형 |
| **Scherrer+ 2012 (HMI investigation)** | 동반 논문 / companion | HMI 미션 개관과 과학 목표 |
| **Schou+ 2012a (HMI design)** | 동반 논문 / companion | 기기 설계와 지상 교정 |
| **Schou+ 2012b (polarization)** | 세부 / detailed | 편광 교정 방법론의 모델 |
| **Couvidat+ 2012a (LoS testing)** | 세부 / detailed | IBIS 고분해능 데이터로 LoS 알고리즘 검증 |
| **Couvidat+ 2012b (filter calibration)** | 세부 / detailed | 필터 요소의 파장 의존 교정 |
| **Wachter+ 2012 (image quality)** | 세부 / detailed | 왜곡·PSF·비선형성 지상 측정 |
| **Borrero+ 2011 (VFISV)** | 방법론 / methodology | Milne-Eddington Stokes 벡터 반전 코드 |
| **Centeno+ 2014 (vector pipeline)** | 후속 / follow-up | HMI 벡터 자기장 파이프라인 업데이트 |
| **Hoeksema+ 2014 (vector performance)** | 보완 / complementary | 벡터 자기장의 24시간 변동 상세 분석 |
| **Bobra+ 2014 (SHARPs)** | 후속 / follow-up | 활동영역 패치 생성 |
| **Liu+ 2012 (MDI vs HMI)** | 검증 / validation | MDI와 HMI 자기장 값의 비교 |
| **Bush+ 2016 (on-orbit performance)** | 동반 논문 / companion | Level-1 처리의 상세 및 기기 성능 |
| **Norton+ 2006 (line selection)** | 배경 / background | HMI 선택이 Fe I 6173인 이유 |

## References / 참고문헌

- Couvidat, S. et al., "Observables Processing for the Helioseismic and Magnetic Imager Instrument on the Solar Dynamics Observatory", *Solar Physics* 291, 1887-1938, 2016. [DOI:10.1007/s11207-016-0957-3]
- Scherrer, P.H. et al., "The Helioseismic and Magnetic Imager (HMI) Investigation for the Solar Dynamics Observatory (SDO)", *Solar Phys.* 275, 207, 2012.
- Scherrer, P.H. et al., "The Solar Oscillations Investigation - Michelson Doppler Imager", *Solar Phys.* 162, 129, 1995.
- Schou, J. et al., "Design and Ground Calibration of the Helioseismic and Magnetic Imager (HMI)", *Solar Phys.* 275, 229, 2012a.
- Schou, J. et al., "Polarization Calibration of the Helioseismic and Magnetic Imager (HMI)", *Solar Phys.* 275, 327, 2012b.
- Borrero, J.M. et al., "VFISV: Very Fast Inversion of the Stokes Vector for the HMI", *Solar Phys.* 273, 267, 2011.
- Centeno, R. et al., "The HMI Vector Magnetic Field Pipeline: Optimization of the Spectral Line Inversion Code", *Solar Phys.* 289, 3531, 2014.
- Hoeksema, J.T. et al., "The HMI Vector Magnetic Field Pipeline: Overview and Performance", *Solar Phys.* 289, 3483, 2014.
- Wachter, R. et al., "Image Quality of the Helioseismic and Magnetic Imager (HMI) onboard the Solar Dynamics Observatory (SDO)", *Solar Phys.* 275, 261, 2012.
- Liu, Y. et al., "Comparison of Line-of-sight Magnetograms Taken by SDO/HMI and SOHO/MDI", *Solar Phys.* 279, 295, 2012.
- Bobra, M.G. et al., "The HMI Vector Magnetic Field Pipeline: SHARPs", *Solar Phys.* 289, 3549, 2014.
- Norton, A.A. et al., "Spectral Line Selection for HMI: Fe I 6173 vs Ni I 6768", *Solar Phys.* 239, 69, 2006.
- Martinez Oliveros, J.C. et al., "Imaging Spectroscopy of a White-light Solar Flare", *Solar Phys.* 269, 269, 2011.
- Couvidat, S. et al., "Line-of-sight Observables Algorithms for the HMI Instrument Tested with IBIS", *Solar Phys.* 278, 217, 2012a.
- Couvidat, S. et al., "Wavelength Dependence of the HMI Instrument onboard SDO", *Solar Phys.* 275, 285, 2012b.
- Pesnell, W.D., Thompson, B.J., Chamberlin, P.C., "The Solar Dynamics Observatory (SDO)", *Solar Phys.* 275, 3, 2012.
- Domingo, V., Fleck, B., Poland, A.I., "The SOHO Mission: an Overview", *Solar Phys.* 162, 1, 1995.
- Bush, R.I. et al., "On-Orbit Performance of the HMI on the SDO", *Solar Phys.*, 2016.
- Schuck, P.W. et al., "Achieving Consistent Doppler Measurements from SDO/HMI Vector Field Inversions", *Astrophys. J.* 823, 101, 2016.
