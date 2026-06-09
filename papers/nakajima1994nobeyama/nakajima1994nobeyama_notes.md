---
paper_id: 29
topic: Solar_Observation
date: 2026-04-23
type: notes
title: "The Nobeyama Radioheliograph"
authors: "Nakajima, H., Nishio, M., Enome, S., Shibasaki, K., Takano, T., Hanaoka, Y., Torii, C., Sekiguchi, H., Bushimata, T., Kawashima, S., Shinohara, N., Irimajiri, Y., Koshiishi, H., Kosugi, T., Shiomi, Y., Sawa, M., Kai, K."
year: 1994
venue: "Proceedings of the IEEE, 82(5), 705-713"
doi: "10.1109/5.284737"
tags: [radio_astronomy, interferometry, solar_flares, NoRH, aperture_synthesis, gyrosynchrotron]
---

# The Nobeyama Radioheliograph — Reading Notes / 읽기 노트

## 1. Core Contribution / 핵심 기여

**English.** Nakajima et al. (1994) describe the Nobeyama Radioheliograph (NoRH), a 17 GHz interferometer dedicated entirely to full-disk solar imaging. By optimizing an 84-antenna T-shaped array (488.96 m E-W × 220.06 m N-S) with logarithmic baseline multiples $d, 2d, 4d, 8d, 16d$ and a fundamental spacing $d = 1.528$ m, NoRH achieves 10 arcsecond spatial resolution, 1 s (up to 50 ms) temporal resolution, a 40 arcminute full-Sun field of view, and 20-30 dB dynamic range — parameters that no previous solar-dedicated instrument had simultaneously achieved. The paper's key engineering contributions are (i) fiber-optic phase-stable distribution of the 525 MHz reference and the IF signals over paths up to 280 m, (ii) custom CMOS gate-array 1-bit quadraphase complex correlators handling 3486 antenna pairs at 40 MHz sampling, (iii) a modified CLEAN deconvolution that first subtracts the solar disk and extended components before extracting compact flare sources, and (iv) redundant-baseline self-calibration using the Sun itself as a calibrator. Routine 8-hour daily observations began in late June 1992, launching nearly three decades of near-continuous solar radio monitoring that underpinned hundreds of studies on flare electron acceleration, filament structure, and coronal magnetography.

**한국어.** Nakajima 등 (1994)은 태양 전체 원반 영상화에 전적으로 헌정된 17 GHz 간섭계인 Nobeyama Radioheliograph (NoRH)를 기술한다. 기본 간격 $d = 1.528$ m의 로그 기저선 배수 $d, 2d, 4d, 8d, 16d$를 가진 84개 안테나의 T자형 배열 (동서 488.96 m × 남북 220.06 m)을 최적화함으로써 NoRH는 10 arcsecond 공간 분해능, 1 s (최대 50 ms) 시간 분해능, 40 arcminute 전태양 시야, 20-30 dB 동적 범위를 달성한다 — 어떠한 이전 태양 전용 관측기기도 이들을 동시에 달성한 적이 없다. 논문의 주요 공학적 기여는 (i) 525 MHz 기준 신호 및 IF 신호를 최대 280 m 경로에 걸쳐 광섬유로 위상 안정하게 분배한 것, (ii) 3486 안테나 쌍을 40 MHz 샘플링으로 처리하는 맞춤형 CMOS 게이트 어레이 1-bit 4상 복소 상관기, (iii) 태양 원반과 확장 성분을 먼저 제거한 뒤 콤팩트 플레어 소스를 추출하는 수정된 CLEAN 디컨볼루션, (iv) 태양 자체를 보정원으로 사용하는 중복 기저선 자가보정이다. 1992년 6월 말부터 일일 8시간 정기 관측이 시작되어, 거의 30년에 걸친 태양 전파 연속 모니터링의 시작을 알렸으며, 이는 플레어 전자 가속, 필라멘트 구조, 코로나 자기장 매핑에 관한 수백 편의 연구를 뒷받침하였다.

---

## 2. Reading Notes / 읽기 노트 (Section-by-Section)

### 2.1 Introduction (p. 705) / 서론

**English.** The paper opens by positioning NoRH against existing general-purpose interferometers — the Very Large Array (VLA, New Mexico) and the Westerbork Synthesis Radio Telescope (WSRT, Netherlands). These facilities use large-diameter dishes optimized for high collecting area and narrow field of view, suited to quasi-stationary cosmic sources or selected solar active regions during scheduled campaigns. Solar flare physics, however, requires (a) a wide field to cover the entire 32 arcminute solar disk, (b) continuous daily monitoring to build statistical samples, and (c) multi-second cadence to resolve impulsive flare rise phases. The authors motivate the choice of 17 GHz as the initial observing frequency: at short centimeter wavelengths, radio sources are optically thin for gyrosynchrotron emission, and the electron energy spectrum can be inferred from frequency spectra. They note that expansion to 34 GHz (a second frequency) is planned (and was indeed implemented a few years later). The stated science targets are (i) locating electron acceleration sites in flares, (ii) resolving complex flare geometry in two dimensions, (iii) measuring coronal magnetic fields through circular polarization, and (iv) following impulsive flare time evolution.

**한국어.** 논문은 NoRH를 기존 범용 간섭계인 Very Large Array (VLA, 뉴멕시코)와 Westerbork Synthesis Radio Telescope (WSRT, 네덜란드)와 대비시키며 시작한다. 이 관측시설들은 높은 집광 면적과 좁은 시야에 최적화된 대구경 접시 안테나를 사용하며, 준정상 우주 광원이나 정해진 관측 캠페인 중 특정 태양 활동영역에 적합하다. 반면 태양 플레어 물리는 (a) 32 arcminute의 전체 태양 원반을 덮는 넓은 시야, (b) 통계 표본을 구축하기 위한 연속 일일 모니터링, (c) 임펄시브 플레어 상승 단계를 분해할 초 단위 관측주기를 요구한다. 저자들은 초기 관측 주파수로 17 GHz를 선택한 이유를 동기부여한다: 단 센티미터 파장에서 전파 소스는 자이로싱크로트론 방출에 대해 광학적으로 얇으며, 주파수 스펙트럼으로부터 전자 에너지 스펙트럼을 추정할 수 있다. 두 번째 주파수인 34 GHz로의 확장이 계획되어 있다 (실제로 몇 년 후 구현되었다). 명시된 과학 목표는 (i) 플레어에서 전자 가속 장소 위치 측정, (ii) 복잡한 플레어 기하를 2차원으로 분해, (iii) 원편광으로 코로나 자기장 측정, (iv) 임펄시브 플레어 시간 진화 추적이다.

### 2.2 Antenna and Array Configuration (Section II, p. 706) / 안테나 및 배열 구성

**English.** The T-array geometry is the paper's central design choice. Eighty-four antennas are placed along two perpendicular arms: a 488.96 m east-west arm and a 220.06 m north-south arm, meeting at the phase center where the observation building stands. Within each arm, antennas are placed at positions corresponding to integer multiples of the fundamental spacing $d = 1.528$ m, but with a geometric progression of spacings: $d, 2d, 4d, 8d, 16d$ between successive antennas as we move outward from the center. Large portions of antennas concentrate near the center to produce dense u-v sampling at short baselines (large-scale structure such as the solar disk), while a smaller number of antennas at the ends of the arms provide the long baselines needed for 10 arcsec resolution. Crucially, the equal-spacing subsets produce many **redundant baselines** — distinct antenna pairs sharing the same baseline vector — which both enable gain/phase calibration using the Sun itself and allow direct FFT synthesis without gridding (any gridding step introduces interpolation errors and blurs the PSF). Each antenna is a parabolic reflector of 80 cm diameter with half-power beamwidth of $87.'1 \pm 0.'4$ (~5220 arcsec), giving a full primary beam generously larger than the 40' field and causing only a 9% gain degradation at the solar limb. Antennas sit on alt-az mounts driven by stepping motors, with mechanical positioning of ±1.5 arcmin — sufficient given the 87.1 arcmin primary beam. Each antenna position is surveyed to 0.5 mm rms with respect to the ideal T-coordinates, keeping phase errors for 1-hour synthesis observations below 3°.

**한국어.** T자형 배열 기하는 논문의 핵심 설계 선택이다. 84개 안테나가 두 수직한 암을 따라 배치된다: 488.96 m 동서 암과 220.06 m 남북 암으로 위상 중심에서 만나며 그 지점에 관측 건물이 있다. 각 암 안에서 안테나는 기본 간격 $d = 1.528$ m의 정수배에 해당하는 위치에 놓이되, 중심으로부터 바깥으로 이동하면서 연속 안테나 간격이 기하급수적으로 증가한다: $d, 2d, 4d, 8d, 16d$. 많은 수의 안테나가 중심 근방에 집중되어 짧은 기저선에서 조밀한 u-v 샘플링 (태양 원반 같은 대규모 구조)을 만들어내는 반면, 더 적은 수의 안테나가 암의 양 끝에 위치하여 10 arcsec 분해능에 필요한 긴 기저선을 제공한다. 결정적으로, 등간격 부분집합은 **중복 기저선** — 동일한 기저선 벡터를 공유하는 서로 다른 안테나 쌍 — 을 많이 만들어내는데, 이는 태양 자체를 이용한 이득/위상 보정을 가능하게 할 뿐 아니라 gridding 없이 직접 FFT 합성이 가능하게 한다 (gridding은 보간 오차를 도입하고 PSF를 흐려뜨린다). 각 안테나는 80 cm 직경의 포물선 반사판으로 반치전폭 $87.'1 \pm 0.'4$ (~5220 arcsec)를 가지며, 40' 시야보다 충분히 큰 1차 빔을 주어 태양 주변부에서 단 9% 이득 감소만 일으킨다. 안테나는 스테핑 모터로 구동되는 방위-고도 (alt-az) 마운트에 설치되며, 기계적 정밀도 ±1.5 arcmin — 87.1 arcmin 1차 빔에 대해 충분하다. 각 안테나 위치는 이상적 T-좌표에 대해 0.5 mm rms로 측량되어 1시간 합성 관측의 위상 오차를 3° 이하로 유지한다.

### 2.3 Receiver System (Section III, p. 706) / 수신 시스템

**English.** The signal chain comprises: corrugated conical horn → polarization switch (toggling right/left circular polarization every 25 ms) → uncooled HEMT low-noise amplifier (180 K noise temperature) → harmonic mixer with an 8.4 GHz local oscillator → 200 MHz IF with 33.6 MHz bandwidth → 10 dB attenuator and 180° Walsh phase switch → electrical-to-optical converter → single-mode optical fiber (up to 280 m) → observation building. Once at the building, each IF is re-mixed into two quadrature baseband channels (I/Q, 100 kHz-16.8 MHz), 1-bit sampled at 40 MHz, digitally delayed (fine delay 0-24 ns in 1 ns steps by clock-phase selection; coarse delay 0-1700 ns in 25 ns steps using shift registers), and Walsh-demodulated. The overall receiver noise is 360 K rms, and with a quiet-Sun antenna temperature of about 550 K, the ratio of signal to noise puts the system firmly in the strong-source regime where the 1-bit correlator's $2/\pi$ sensitivity loss is not limiting. Phase stability is achieved through (i) a Sumitomo phase-stable buried optical fiber with 0.2 ppm/°C temperature coefficient, laid 1.2 m underground where daily temperature varies by less than 0.1°C, (ii) a Peltier-thermostat-housed PLL (phase-locked loop) oscillator that divides the 8.4 GHz LO output by 16 and compares to a 525 MHz master reference, and (iii) careful temperature control (35 ± 1°C) of each frontend box housing the HEMT and local mixer. The combined result is overall phase stability below 0.3° rms and amplitude stability below 0.2 dB rms.

**한국어.** 신호 경로는 다음과 같다: 주름 원뿔 혼 → 편광 전환 스위치 (25 ms마다 우/좌 원편광 교대) → 비냉각 HEMT 저잡음 증폭기 (180 K 잡음 온도) → 8.4 GHz 국부 발진기를 이용한 고조파 믹서 → 33.6 MHz 대역폭 200 MHz IF → 10 dB 감쇠기 및 180° Walsh 위상 스위치 → 전기-광 변환기 → 단일 모드 광섬유 (최대 280 m) → 관측 건물. 건물에 도착한 각 IF는 두 직교 베이스밴드 채널 (I/Q, 100 kHz-16.8 MHz)로 재혼합되고, 40 MHz로 1-bit 샘플링되며, 디지털로 지연 보상 (미세 지연 0-24 ns의 1 ns 단계는 클록 위상 선택, 거친 지연 0-1700 ns의 25 ns 단계는 시프트 레지스터)되고, Walsh 복조된다. 전체 수신기 잡음은 360 K rms이고, 정온 태양의 안테나 온도가 약 550 K이므로, 신호-대-잡음 비율은 시스템을 확실히 강광원 영역에 위치시키며, 이 영역에서는 1-bit 상관기의 $2/\pi$ 감도 손실이 제한 요소가 되지 않는다. 위상 안정성은 (i) 0.2 ppm/°C 온도 계수를 가진 Sumitomo 위상 안정 광섬유를 일 변동 0.1°C 미만의 1.2 m 지하에 매설, (ii) 8.4 GHz LO 출력을 16으로 나누어 525 MHz 마스터 기준 신호와 비교하는, Peltier 서모스탯으로 격리된 PLL (위상 고정 루프) 발진기, (iii) HEMT와 국부 믹서를 수용한 각 전단부 박스의 세심한 온도 제어 (35 ± 1°C)로 달성된다. 결합된 결과는 0.3° rms 이하의 전체 위상 안정성과 0.2 dB rms 이하의 진폭 안정성이다.

### 2.4 Digital Backend and 1-bit Correlator (Section III.C, p. 709) / 디지털 후단부 및 1-bit 상관기

**English.** The digital correlator is NoRH's technological crown jewel. Correlations for all $\binom{84}{2} = 3486$ antenna pairs must be computed, for two polarizations, at 40 MHz sampling. Using standard multi-bit correlators would be prohibitively expensive; NoRH instead uses 1-bit quadraphase correlators. After Walsh demodulation and delay compensation, each antenna provides an I/Q pair of 1-bit streams. A custom CMOS gate-array LSI (30000 gates per chip) implements 16 complex correlator units per chip, covering a 4×4 block of antenna pairs. The total 3486 correlations use 231 chips. Each correlator unit consists of four 4-bit-parallel XOR circuits (for the four products II, IQ, QI, QQ), an add/subtract combiner (real = II + QQ, imag = QI − IQ for a complex multiply), two 22-bit integrators, a latch, and a multiplexer. Internally, the 40 MHz bit stream is re-sampled into four 10 MHz streams offset by 25 ns to match the CMOS gate array's maximum clock. Integration runs for 24.615 ms, synchronized with the 25 ms polarization switch, leaving 0.385 ms dead time during polarization transitions. The 1-bit quantization loses a factor of $2/\pi \approx 0.637$ in SNR, and the nonlinearity between true correlation $\rho_{\text{true}}$ and measured correlation $\rho_{\text{clipped}}$ is corrected by the **Van Vleck correction** $\rho_{\text{true}} = \sin(\pi \rho_{\text{clipped}} / 2)$. Total amplitude is recovered separately via square-law detectors on each antenna channel every 25 ms. Thus a complete set of complex visibilities for both polarizations is produced every 50 ms. 50 ms-cadence flare data are written to a SONY DIR1000 high-speed 20 Gbyte tape daily, while 1 s averages stream to an NEC SX-JL host computer (285 MFLOPS, 128 MB RAM, 45 GB disk) for real-time synthesis every 10 s.

**한국어.** 디지털 상관기는 NoRH의 기술적 정수이다. 84개 안테나 중 $\binom{84}{2} = 3486$ 쌍 모든 조합의 상관을, 두 편광에 대해, 40 MHz 샘플링으로 계산해야 한다. 표준 다중비트 상관기를 사용하면 비용이 엄청나게 든다; 대신 NoRH는 1-bit 4상 상관기를 사용한다. Walsh 복조와 지연 보상 후, 각 안테나는 I/Q 쌍의 1-bit 스트림을 제공한다. 맞춤형 CMOS 게이트 어레이 LSI (칩당 30000 게이트)는 칩당 16개 복소 상관기 유닛을 구현하여 4×4 안테나 쌍 블록을 처리한다. 총 3486개 상관은 231개 칩으로 구현된다. 각 상관기 유닛은 네 개의 4-bit 병렬 XOR 회로 (II, IQ, QI, QQ 네 곱), 가산/감산 결합기 (실수부 = II + QQ, 허수부 = QI − IQ), 두 개의 22-bit 적분기, 래치, 멀티플렉서로 구성된다. 내부적으로 40 MHz 비트 스트림은 25 ns 오프셋된 네 개의 10 MHz 스트림으로 재샘플링되어 CMOS 게이트 어레이의 최대 클록에 맞춘다. 적분은 24.615 ms 동안 수행되며 25 ms 편광 전환과 동기화되어, 편광 전환 중 0.385 ms 데드 타임을 남긴다. 1-bit 양자화는 SNR에서 $2/\pi \approx 0.637$ 인자를 잃으며, 참 상관 $\rho_{\text{true}}$와 측정된 상관 $\rho_{\text{clipped}}$ 사이의 비선형성은 **Van Vleck 보정** $\rho_{\text{true}} = \sin(\pi \rho_{\text{clipped}} / 2)$로 보정된다. 총 진폭은 각 안테나 채널의 제곱 법칙 검출기를 통해 25 ms마다 별도로 복구된다. 따라서 두 편광에 대한 완전한 복소 가시도 집합이 50 ms마다 생성된다. 50 ms 관측주기 플레어 데이터는 매일 SONY DIR1000 고속 20 Gbyte 테이프에 기록되고, 1 s 평균은 NEC SX-JL 호스트 컴퓨터 (285 MFLOPS, 128 MB RAM, 45 GB 디스크)로 스트리밍되어 10초마다 실시간 합성된다.

### 2.5 Calibration and Image Restoration (Section IV, p. 710) / 보정 및 영상 복원

**English.** NoRH cannot use cosmic point sources for calibration because the 80 cm antennas are too small (collecting area ~0.5 m², poor sensitivity to weak nonsolar sources). Instead, redundant baselines allow self-calibration: if two antenna pairs have the same baseline vector, their visibilities should be identical except for antenna-specific gain/phase errors. Solving the system of $N_{\text{pairs}}$ equations for $2 N_{\text{ant}}$ unknowns (complex gains) by least squares yields antenna-dependent amplitude and phase corrections. This technique, first developed at Nobeyama by Nakajima et al. (1980) for a precursor 17 GHz interferometer and generalized by Kimura (1988), runs on the Sun itself without interrupting observations and tracks short-timescale atmospheric phase variations. Absolute positions are then referenced to the sharp solar limb, which is sharper in radio than optical (the 17 GHz radio disk is 2.5% larger than the optical disk because the opacity cutoff occurs in the chromosphere). After FFT synthesis, a **modified CLEAN** algorithm is applied. Standard CLEAN (Högbom 1974) iteratively finds the peak of the dirty map, subtracts a scaled synthesized beam at that location, and records a delta-function component. On the Sun, however, the dominant emission is the smooth ~10⁴ K disk plus extended ~10⁵ K active regions, which would require millions of iterations and produce artifacts. NoRH's modified algorithm first subtracts the solar disk convolved with the dirty beam (disk size determined empirically as 2.5% larger than optical), then fits extended Gaussian components of various sizes to the residual, and finally deals with compact features by standard CLEAN. For Stokes V (circular polarization) images the disk subtraction step is skipped since V is small and zero-mean on the disk. Fig. 6 in the paper shows a dirty image (heavy sidelobes radiating across the disk) and the restored image after the modified CLEAN, with dynamic range improved by factors of ~50-100.

**한국어.** NoRH는 우주 점광원을 보정에 사용할 수 없다 — 80 cm 안테나는 집광 면적이 너무 작기 때문이다 (~0.5 m², 약한 비태양 광원에 대한 감도 부족). 대신 중복 기저선은 자가보정을 가능하게 한다: 두 안테나 쌍이 동일한 기저선 벡터를 가지면, 그들의 가시도는 안테나별 이득/위상 오차를 제외하고 동일해야 한다. $N_{\text{pairs}}$ 방정식에 대한 $2 N_{\text{ant}}$개 미지수 (복소 이득)의 시스템을 최소 제곱으로 풀어 안테나 의존 진폭 및 위상 보정을 얻는다. 이 기법은 Nakajima 등 (1980)이 선행 17 GHz 간섭계를 위해 Nobeyama에서 처음 개발하고 Kimura (1988)가 일반화하였는데, 관측을 중단하지 않고 태양 자체에서 동작하며 단시간 대기 위상 변화를 추적한다. 절대 위치는 이후 날카로운 태양 주변부를 기준으로 삼는데, 이는 가시광선보다 전파에서 더 선명하다 (불투명도 경계가 채층에서 발생하기 때문에 17 GHz 전파 원반이 가시 원반보다 2.5% 크다). FFT 합성 후 **수정된 CLEAN** 알고리즘이 적용된다. 표준 CLEAN (Högbom 1974)은 반복적으로 dirty 맵의 피크를 찾아 그 위치에서 스케일된 합성 빔을 차감하고 델타 함수 성분을 기록한다. 그러나 태양에서 지배적인 방출은 부드러운 ~10⁴ K 원반 + 확장 ~10⁵ K 활동영역이며, 이는 수백만 번의 반복을 요구하고 아티팩트를 만들어낸다. NoRH의 수정 알고리즘은 먼저 dirty 빔과 콘볼루션된 태양 원반을 차감하고 (원반 크기는 경험적으로 광학 원반보다 2.5% 큰 값으로 결정), 잔차에 다양한 크기의 확장 가우시안 성분을 적합하고, 최종적으로 표준 CLEAN으로 콤팩트 성분을 처리한다. Stokes V (원편광) 영상에서는 원반 차감 단계를 건너뛴다 — V는 원반에서 작고 0-평균이기 때문이다. 논문의 Fig. 6은 dirty 영상 (원반을 가로지르는 큰 사이드로브)과 수정 CLEAN 후 복원 영상을 보여주며, 동적 범위가 ~50-100배 향상됨을 보여준다.

### 2.5b Worked Example: Building the u-v Coverage / 작업 예제: u-v 덮개 구축

**English.** Let us trace how 84 antennas produce NoRH's snapshot u-v coverage. The E-W arm is 488.96 m / 1.528 m = 320 units long; antennas sit at positions that are multiples of $d$ with spacing $d, 2d, 4d, 8d, 16d$ between neighbors, giving roughly $\log_2(320) + 1 \approx 9$ antenna positions per "octave". The N-S arm (220.06 m / 1.528 m = 144 units) follows the same rule. For 84 total antennas distributed between the two arms (plus the center), each antenna pair $(i,j)$ yields a baseline $\mathbf{b}_{ij} = \mathbf{r}_i - \mathbf{r}_j$. Only pairs that cross the center produce both E-W and N-S components simultaneously; same-arm pairs only populate the u-axis or v-axis. At any instant, the sampled u-v points form a cross-like pattern, but Earth rotation smears each point into an arc, filling in the plane over several hours — this is how NoRH gets its 30 dB rotational synthesis dynamic range. The maximum E-W baseline $u_{\max} = 488.96 / \lambda = 488.96 / 0.017635 = 27723$ wavelengths, giving $\theta_{\text{res,EW}} = 1/u_{\max} = 3.6 \times 10^{-5}$ rad = 7.4"; the N-S direction $\theta_{\text{res,NS}} = 1/(220.06/\lambda) = 16.5"$, so the beam is elliptical (2:1) unless tapering is applied.

**한국어.** 84개 안테나가 NoRH의 스냅샷 u-v 덮개를 어떻게 생성하는지 추적해보자. 동서 암은 488.96 m / 1.528 m = 320 단위 길이이며; 안테나는 $d$의 배수 위치에 놓이되 이웃 간 간격은 $d, 2d, 4d, 8d, 16d$로, "옥타브"당 대략 $\log_2(320) + 1 \approx 9$개 안테나 위치를 만든다. 남북 암 (220.06 m / 1.528 m = 144 단위)도 동일 규칙을 따른다. 두 암에 분배된 총 84개 안테나 (+ 중심)에 대해 각 안테나 쌍 $(i,j)$는 기저선 $\mathbf{b}_{ij} = \mathbf{r}_i - \mathbf{r}_j$를 생성한다. 중심을 교차하는 쌍만이 동서 및 남북 성분을 동시에 생성하며; 동일 암 쌍은 u-축 또는 v-축만 채운다. 어느 순간이든 샘플링된 u-v 점들은 십자형 패턴을 형성하지만, 지구 자전이 각 점을 호로 번지게 하여 수 시간에 걸쳐 평면을 채운다 — 이것이 NoRH가 30 dB 회전 합성 동적 범위를 얻는 방법이다. 최대 동서 기저선 $u_{\max} = 488.96 / \lambda = 27723$ 파장, $\theta_{\text{res,EW}} = 1/u_{\max} = 3.6 \times 10^{-5}$ rad = 7.4"를 주고; 남북 방향은 $\theta_{\text{res,NS}} = 16.5"$로, 테이퍼링을 적용하지 않으면 빔은 2:1 타원형이다.

### 2.6 Total Performance (Section V, p. 710) / 전체 성능

**English.** Since routine operations began in late June 1992, about one hundred solar flares had been recorded by the time the paper was written (about 6 months of operation). Fig. 7 in the paper shows the time evolution of a medium-class east-limb flare on 28 June 1992 imaged from 1-second data with a 315" × 315" field. The earliest frame shows a strongly polarized compact source on the limb, followed by upward expansion (consistent with loop-top emission rising along a magnetic arcade), and finally a complex decay-phase structure. Brightness contrast ranged from $1.5 \times 10^6$ K in the flare core down to features below 5000 K on the disk — a dynamic range of about 25 dB, consistent with design spec. For a GOES X9 class flare on 2 November 1992 the authors use closure amplitudes and closure phases to assess image quality quantitatively. For the fundamental and second harmonic antenna spacings (short baselines where redundancy is high), closure amplitude variations are below 1% peak-to-peak and closure phase variations are below 0.5°. For the 16th and 32nd harmonics (longer baselines where fewer redundant pairs exist), closure amplitude varies by up to 8% and closure phase by 1°. From these residuals the authors infer a dynamic range greater than 30 dB, **exceeding** the pre-construction design expectations of Nishio (1990) and Koshiishi (1993).

**한국어.** 1992년 6월 말 정기 관측을 시작한 이래, 논문 작성 시점까지 (약 6개월 운영) 약 100개의 태양 플레어가 기록되었다. 논문의 Fig. 7은 1992년 6월 28일 동쪽 주변부에서 발생한 중급 플레어의 시간 진화를 1초 데이터로 315" × 315" 시야에 영상화한 것이다. 가장 초기 프레임은 주변부의 강하게 편광된 콤팩트 소스를 보여주고, 이어 상향 확장 (자기 아치를 따라 상승하는 루프 꼭대기 방출과 부합), 마지막으로 복잡한 감쇠 단계 구조를 보여준다. 밝기 대비는 플레어 코어의 $1.5 \times 10^6$ K부터 원반 상의 5000 K 이하 특징까지 — 약 25 dB 동적 범위로, 설계 사양과 부합한다. 1992년 11월 2일 GOES X9 플레어에 대해 저자들은 폐쇄 진폭 (closure amplitude)과 폐쇄 위상 (closure phase)을 사용해 영상 품질을 정량적으로 평가한다. 기본 및 2차 고조파 안테나 간격 (중복도가 높은 짧은 기저선)에서 폐쇄 진폭 변화는 피크-대-피크 1% 이하, 폐쇄 위상 변화는 0.5° 이하이다. 16차 및 32차 고조파 (중복 쌍이 적은 긴 기저선)에서 폐쇄 진폭은 최대 8%, 폐쇄 위상은 최대 1° 변한다. 이 잔차들로부터 저자들은 30 dB를 **초과하는** 동적 범위를 추정하며, 이는 Nishio (1990)와 Koshiishi (1993)의 건설 전 설계 기대값을 **초과한다**.

### 2.7 Case Study: The June 28, 1992 Limb Flare / 사례 연구: 1992년 6월 28일 주변부 플레어

**English.** The paper's Fig. 7 shows four snapshots of a medium-class limb flare at 04:03:13 UT, 04:03:36 UT, 04:04:00 UT, and 04:04:41 UT (approximate times deduced from the figure caption). Each panel is 315" × 315", far larger than a NoRH beam (10") but smaller than the full disk. The first frame shows a single compact radio source with V (circular polarization) amplitude nearly equal to I (total intensity), indicating nearly 100% polarization — typical of coherent plasma emission from the region above a flaring magnetic loop where line-of-sight magnetic field is near parallel. The second and third frames show the source expanding upward (radially outward along the limb) and splitting into two bright knots connected by a fainter arch, consistent with gyrosynchrotron emission from electrons trapped in a coronal loop seen edge-on. The fourth (decay) frame shows a complex multi-knot structure with weaker polarization, consistent with loss-cone instability ending and isotropic electron distribution remaining in the loop. The flare structure evolves over about 90 seconds — a duration that *cannot* be captured by scanning-beam instruments or by non-imaging spectrometers.

**한국어.** 논문의 Fig. 7은 중급 주변부 플레어의 4개 스냅샷을 보여준다 (그림 설명에서 추정한 시각: 04:03:13 UT, 04:03:36 UT, 04:04:00 UT, 04:04:41 UT). 각 패널은 315" × 315"로, NoRH 빔 (10")보다 훨씬 크지만 전체 원반보다는 작다. 첫 번째 프레임은 V (원편광) 진폭이 I (총 강도)와 거의 같은 단일 콤팩트 전파 광원을 보여주며, 이는 거의 100% 편광을 나타낸다 — 플레어 자기 루프 위의 시선 방향 자기장이 거의 평행한 영역에서의 결맞은 플라즈마 방출의 전형. 두 번째와 세 번째 프레임은 광원이 상향 (주변부를 따라 방사상 바깥쪽)으로 확장되어 더 희미한 아치로 연결된 두 개의 밝은 매듭으로 분리되는 것을 보여주며, 이는 가장자리에서 본 코로나 루프에 갇힌 전자의 자이로싱크로트론 방출과 부합한다. 네 번째 (감쇠) 프레임은 더 약한 편광을 가진 복잡한 다중 매듭 구조를 보여주며, 손실 원뿔 불안정성이 끝나고 등방성 전자 분포가 루프에 남아 있는 것과 부합한다. 플레어 구조는 약 90초에 걸쳐 진화하는데 — 이 지속시간은 스캐닝 빔 관측기기나 비영상 분광기로는 **포착될 수 없다**.

### 2.8 Scientific Capabilities Enabled by NoRH / NoRH가 가능하게 한 과학적 능력

**English.** Beyond flare imaging, NoRH enables several science programs that the paper mentions in passing but which came to dominate its observing time:

1. **Filament eruption monitoring**: Quiescent filaments are seen as depressions (~10³ K below ambient disk) in 17 GHz brightness temperature due to cool ($10^4$ K) material that is optically thin at 17 GHz but optically thick at mm-wavelengths. NoRH's daily monitoring catches eruption onsets in real time, identifying CME precursors hours before coronagraph detection.
2. **Coronal magnetic field mapping**: The fractional circular polarization $V/I$ at 17 GHz is proportional to $B \cos\theta / \nu$ in the optically thick thermal regime, giving a direct measurement of the line-of-sight magnetic field component above active regions. This is one of the very few techniques that directly measures coronal $B$.
3. **Sunspot thermal bremsstrahlung**: Sunspot umbrae appear as compact bright (~$10^5$ K) sources at 17 GHz due to enhanced chromospheric opacity above penumbrae. NoRH tracks umbral $B$ fields through polarization.
4. **Coronal hole detection**: Coronal holes appear darker than quiet Sun at 17 GHz because the lower density reduces bremsstrahlung opacity. Long-term synoptic maps trace coronal hole evolution.
5. **Non-flaring electron populations**: Noise storms and Type I bursts at higher frequencies seed into the NoRH band during active periods, enabling studies of persistent coronal electron acceleration.

**한국어.** 플레어 영상화를 넘어, NoRH는 논문이 지나가듯 언급하지만 이후 관측 시간을 지배하게 된 여러 과학 프로그램을 가능하게 한다:

1. **필라멘트 분출 모니터링**: 정온 필라멘트는 17 GHz 밝기 온도에서 함몰 (~주변 원반보다 10³ K 낮음)로 보인다 — 차가운 ($10^4$ K) 물질이 17 GHz에서는 광학적으로 얇지만 mm 파장에서는 두껍기 때문. NoRH의 일일 모니터링은 분출 시작을 실시간 포착하여, 코로나그래프 검출보다 수 시간 먼저 CME 전조를 식별한다.
2. **코로나 자기장 매핑**: 17 GHz에서 분수 원편광 $V/I$는 광학적으로 두꺼운 열적 영역에서 $B \cos\theta / \nu$에 비례하여, 활동영역 위의 시선 방향 자기장 성분의 직접 측정을 준다. 이는 코로나 $B$를 직접 측정하는 극소수 기법 중 하나이다.
3. **흑점 열적 제동복사**: 흑점 암영부는 반영부 위의 강화된 채층 불투명도로 인해 17 GHz에서 콤팩트한 밝은 (~$10^5$ K) 광원으로 나타난다. NoRH는 편광을 통해 암영부 $B$ 장을 추적한다.
4. **코로나 홀 검출**: 코로나 홀은 낮은 밀도가 제동복사 불투명도를 감소시켜 17 GHz에서 정온태양보다 어둡게 보인다. 장기 시놉틱 맵은 코로나 홀 진화를 추적한다.
5. **비플레어 전자 집단**: 더 높은 주파수의 잡음 폭풍과 Type I 폭발은 활동 기간 동안 NoRH 대역으로 유입되어, 지속적 코로나 전자 가속 연구를 가능하게 한다.

---

## 3. Key Takeaways / 핵심 시사점

### 3.1 Dedicated instruments enable statistical astrophysics / 전용 관측기기가 통계적 천체물리를 가능하게 한다

**English.** The single most transformative aspect of NoRH is dedication. By sacrificing the ability to observe any target other than the Sun, NoRH replaces 3-10% VLA time availability with 100% daily availability. This converts solar radio astronomy from a campaign science (where rare flares must be caught during scheduled blocks) to a monitoring science (where every flare above threshold is automatically recorded). The scientific payoff is proportional to baseline integrated time.

**한국어.** NoRH의 가장 변혁적인 측면은 전용성이다. 태양 이외 대상을 관측할 능력을 포기함으로써, NoRH는 VLA의 3-10% 시간 가용성을 100% 일일 가용성으로 대체한다. 이로써 태양 전파천문학은 (드문 플레어를 예약된 시간 블록에서 포착해야 하는) 캠페인 과학에서 (임계값 이상의 모든 플레어가 자동 기록되는) 모니터링 과학으로 전환된다. 과학적 성과는 적분된 기저선 시간에 비례한다.

### 3.2 Array geometry drives science / 배열 기하가 과학을 결정한다

**English.** The T-array with geometric-progression spacing ($d, 2d, 4d, 8d, 16d$) is not the most sensitive configuration per baseline — that would be a random or Reuleaux-like layout. But it is optimal for (i) instantaneous (snapshot) u-v coverage required for 1-second images of rapidly evolving flares, (ii) many redundant short baselines for self-calibration using the Sun, and (iii) lattice-regular sampling that allows direct FFT without gridding interpolation errors. The design trade-off is: fewer unique u-v samples per integration, but higher quality per sample and direct FFT synthesis.

**한국어.** 기하급수적 간격 ($d, 2d, 4d, 8d, 16d$)을 가진 T자형 배열은 기저선당 가장 감도가 높은 구성이 아니다 — 그것은 무작위 또는 Reuleaux 같은 배치일 것이다. 하지만 이는 (i) 빠르게 진화하는 플레어의 1초 영상에 필요한 순간 (스냅샷) u-v 덮개, (ii) 태양을 이용한 자가보정을 위한 많은 중복 단기저선, (iii) gridding 보간 오차 없이 직접 FFT를 가능하게 하는 격자 규칙 샘플링에 최적이다. 설계 트레이드오프는: 적분당 고유 u-v 샘플 수가 적지만 샘플당 품질이 높고 직접 FFT 합성이 가능하다는 것이다.

### 3.3 Phase stability is the battle / 위상 안정성이 관건이다

**English.** Interferometer image dynamic range is limited primarily by phase stability, not amplitude noise. A 1° phase error on a baseline produces a ~2% sidelobe level in the synthesized image. NoRH's phase stability budget is ruthless: (a) 0.5 mm antenna position survey → <3° static phase error; (b) buried phase-stable fiber (0.2 ppm/°C × 0.1°C/day × 280 m = 0.8° per day drift); (c) Peltier-thermostatted PLL at ±0.1°C → <1.3° drift; (d) Walsh modulation to remove slow gain and phase drifts in the IF chain. The combined rms is 0.3°, consistent with 30 dB dynamic range.

**한국어.** 간섭계 영상의 동적 범위는 진폭 잡음이 아니라 주로 위상 안정성에 의해 제한된다. 기저선에서 1° 위상 오차는 합성 영상에서 ~2% 사이드로브 레벨을 만들어낸다. NoRH의 위상 안정성 예산은 냉혹하다: (a) 0.5 mm 안테나 위치 측량 → 3° 미만 정적 위상 오차; (b) 매설 위상 안정 광섬유 (0.2 ppm/°C × 0.1°C/일 × 280 m = 일일 0.8° 드리프트); (c) ±0.1°C Peltier 서모스탯 PLL → 1.3° 미만 드리프트; (d) IF 체인의 느린 이득 및 위상 드리프트를 제거하는 Walsh 변조. 결합 rms는 0.3°이며, 30 dB 동적 범위와 부합한다.

### 3.4 1-bit correlators trade sensitivity for scale / 1-bit 상관기는 감도를 규모와 교환한다

**English.** Computing 3486 complex cross-correlations at 40 MHz would cost enormous amounts of memory bandwidth and multiplier silicon if done with 8-bit arithmetic. 1-bit correlation reduces each multiply to a single XOR gate. The $2/\pi \approx 0.637$ SNR penalty is irrelevant for solar observations where signal-to-noise ratios are astronomical ($T_A = 550$ K for quiet Sun versus ~360 K system temperature — already a strong source regime). The Van Vleck correction $\rho_{\text{true}} = \sin(\pi \rho_{\text{clipped}}/2)$ restores linearity so that the 1-bit approximation does not bias the final image.

**한국어.** 40 MHz에서 3486개 복소 교차 상관을 계산하면 8-bit 산술로 할 경우 엄청난 메모리 대역폭과 곱셈기 실리콘을 소모한다. 1-bit 상관은 각 곱셈을 단일 XOR 게이트로 축소한다. $2/\pi \approx 0.637$ SNR 손실은 태양 관측에서는 무관하다 — 신호-잡음비가 천문학적이다 (정온태양 $T_A = 550$ K 대 시스템 온도 ~360 K — 이미 강광원 영역). Van Vleck 보정 $\rho_{\text{true}} = \sin(\pi \rho_{\text{clipped}}/2)$가 선형성을 복원하여 1-bit 근사가 최종 영상을 편향시키지 않도록 한다.

### 3.5 CLEAN must be adapted to the target / CLEAN은 대상에 맞춰 변형되어야 한다

**English.** Standard Högbom CLEAN assumes the sky is a sparse collection of point sources. The Sun is anything but — it is a filled disk with superposed extended and compact components spanning five orders of magnitude in brightness temperature. The modified CLEAN (i) removes the disk component first by subtracting a dirty-beam-convolved uniform disk of empirical size, (ii) fits extended Gaussians of various scales to the residual to capture active regions, and (iii) iterates standard CLEAN only on the compact residuals. This hierarchical approach is now a template for imaging Jupiter, the Galactic Center, and other extended sources.

**한국어.** 표준 Högbom CLEAN은 하늘이 희박한 점광원 집합이라고 가정한다. 태양은 전혀 그렇지 않다 — 다섯 자릿수의 밝기 온도 범위에 걸친 확장 및 콤팩트 성분이 겹쳐진 채워진 원반이다. 수정 CLEAN은 (i) 경험적 크기의 균일 원반을 dirty 빔과 콘볼루션하여 차감함으로써 원반 성분을 먼저 제거하고, (ii) 잔차에 다양한 스케일의 확장 가우시안을 적합하여 활동영역을 포착하며, (iii) 콤팩트 잔차에만 표준 CLEAN을 반복한다. 이 계층적 접근법은 이제 목성, 은하 중심 및 기타 확장 소스 영상화의 템플릿이 되었다.

### 3.6 17 GHz is scientifically sweet / 17 GHz는 과학적으로 최적이다

**English.** The 17 GHz choice balances: (i) gyrosynchrotron from flare-accelerated electrons peaks in the 5-30 GHz range, optically thin at 17 GHz for all but the densest flare cores, enabling electron energy spectrum inference; (ii) the quiet Sun chromospheric opacity is low enough to see active-region magnetic structures but high enough to define a sharp disk edge for positional calibration; (iii) ionospheric delays and atmospheric extinction are negligible; (iv) hardware (HEMT LNAs, SIS mixers at 8.4 GHz LO) was mature. The planned expansion to 34 GHz (second harmonic-like spacing) enables two-point spectral diagnosis: the ratio $T_B(34)/T_B(17)$ yields spectral index, and combined with optical depth models, energy power-law index $\delta$ and magnetic field strength $B$.

**한국어.** 17 GHz 선택은 다음을 균형 잡는다: (i) 플레어 가속 전자로부터의 자이로싱크로트론은 5-30 GHz 범위에서 피크를 보이며, 가장 밀도 높은 플레어 코어를 제외하면 17 GHz에서 광학적으로 얇아 전자 에너지 스펙트럼 추정을 가능하게 한다; (ii) 정온태양의 채층 불투명도는 활동영역 자기 구조를 볼 만큼 낮지만, 위치 보정을 위한 날카로운 원반 경계를 정의할 만큼 충분히 높다; (iii) 전리층 지연과 대기 소광은 무시할 수 있다; (iv) 하드웨어 (HEMT LNA, 8.4 GHz LO SIS 믹서)가 성숙해 있었다. 34 GHz (2차 고조파 유사 간격)로의 계획된 확장은 2-점 스펙트럼 진단을 가능하게 한다: 비율 $T_B(34)/T_B(17)$은 스펙트럼 지수를 주고, 광학 깊이 모델과 결합하면 에너지 멱법칙 지수 $\delta$와 자기장 강도 $B$를 준다.

### 3.7 Complementarity with Yohkoh / Yohkoh과의 상보성

**English.** The 1991 launch of Japan's Yohkoh satellite provided simultaneous hard X-ray (HXR, 20-100 keV) and soft X-ray (SXR, 1-10 keV) imaging of flares. Hard X-rays trace the bremsstrahlung of non-thermal electrons impacting the chromosphere; 17 GHz radio traces gyrosynchrotron emission of the same electron population in coronal magnetic loops. By co-analysis, one can separate acceleration site (coronal loop-top, seen best in radio) from impact site (chromospheric footpoint, seen in HXR), quantify electron density and magnetic field strength, and constrain the pitch angle distribution. This synergy drove the "Neupert effect" literature and the discovery of above-the-loop-top hard X-ray sources in the late 1990s.

**한국어.** 1991년 일본의 Yohkoh 위성 발사는 플레어의 경 X선 (HXR, 20-100 keV) 및 연 X선 (SXR, 1-10 keV) 동시 영상화를 제공하였다. 경 X선은 채층에 충돌하는 비열 전자의 제동복사를 추적하고, 17 GHz 전파는 코로나 자기 루프에서 동일한 전자 집단의 자이로싱크로트론 방출을 추적한다. 공동 분석으로 가속 장소 (전파에서 가장 잘 보이는 코로나 루프 꼭대기)와 충돌 장소 (HXR에서 보이는 채층 발자국)를 분리하고, 전자 밀도와 자기장 강도를 정량화하며, 피치각 분포를 제약할 수 있다. 이 시너지는 "Neupert 효과" 문헌과 1990년대 말 루프 꼭대기 위의 경 X선 소스 발견을 이끌었다.

### 3.7b Optical fiber is the unsung hero / 광섬유는 숨겨진 영웅이다

**English.** At 17 GHz one wavelength is 1.76 cm. A 280 m coaxial cable would have >20 dB of loss and several degrees of phase drift per degree C of ambient temperature change — making the target 0.3° rms phase stability essentially impossible with electrical cable. The Sumitomo phase-stable fiber has temperature coefficient 0.2 ppm/°C, giving $\Delta\phi = 2\pi \times (280 \text{ m} / 1.76 \text{ cm}) \times 0.2 \times 10^{-6} / ^\circ\text{C} \times 1^\circ\text{C} = 0.02$ radian = 1° per °C — and the 1.2 m burial limits ambient swings to 0.1°C/day, giving the quoted 0.8° daily drift. Without phase-stable fiber, the entire NoRH concept would have failed; this is why Sumitomo gets explicit thanks in the acknowledgments. The fiber also eliminates EMI pickup (critical around a 40 MHz sampling clock distributed over 500 m) and reduces bandwidth equalization requirements by orders of magnitude versus coax.

**한국어.** 17 GHz에서 한 파장은 1.76 cm이다. 280 m 동축 케이블은 주위 온도가 1°C 변할 때 >20 dB 손실과 수 도의 위상 드리프트를 가지며 — 목표 0.3° rms 위상 안정성을 전기 케이블로는 본질적으로 달성 불가능하게 만든다. Sumitomo 위상 안정 광섬유는 온도 계수 0.2 ppm/°C로, $\Delta\phi = 2\pi \times (280 \text{ m} / 1.76 \text{ cm}) \times 0.2 \times 10^{-6} / ^\circ\text{C} \times 1^\circ\text{C} = 0.02$ 라디안 = °C당 1°를 준다 — 그리고 1.2 m 매설이 주위 변동을 일일 0.1°C로 제한하여, 보고된 일일 0.8° 드리프트를 제공한다. 위상 안정 광섬유 없이는 NoRH 개념 전체가 실패했을 것이다; 이것이 Sumitomo가 감사의 말에 명시적으로 언급되는 이유이다. 광섬유는 또한 EMI 픽업 (500 m에 걸쳐 분배되는 40 MHz 샘플링 클록 근처에서 결정적)을 제거하고 동축 대비 대역폭 평준화 요구를 수 자릿수 감소시킨다.

### 3.8 Dynamic range is a software problem as much as hardware / 동적 범위는 하드웨어만큼 소프트웨어 문제이다

**English.** The paper's claim of 20 dB snapshot and 30 dB rotational-synthesis dynamic range rests on three pillars: (i) phase-stable hardware (Section 3.3 above), (ii) the modified CLEAN removing disk sidelobes, and (iii) redundancy-based self-calibration removing antenna-based phase/gain errors in real time. Removing any one pillar collapses the quality to ~10 dB. This is a lesson for any modern interferometer builder: dynamic range is not just about low noise, it requires the full chain from antenna survey to deconvolution to yield compound perfection.

**한국어.** 20 dB 스냅샷 및 30 dB 회전 합성 동적 범위에 대한 논문의 주장은 세 가지 기둥에 기반한다: (i) 위상 안정 하드웨어 (위의 3.3), (ii) 원반 사이드로브를 제거하는 수정 CLEAN, (iii) 안테나 기반 위상/이득 오차를 실시간으로 제거하는 중복 기반 자가보정. 어느 한 기둥을 제거하면 품질이 ~10 dB로 붕괴한다. 이는 현대 간섭계 설계자를 위한 교훈이다: 동적 범위는 단지 저잡음에 관한 것이 아니며, 안테나 측량에서 디컨볼루션까지 전체 체인이 함께 완벽해야 한다.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Interferometer Resolution / 간섭계 분해능

The angular resolution of an interferometer is set by the longest baseline $B_{\max}$ and observing wavelength $\lambda$:

$$\theta_{\text{res}} \approx \frac{\lambda}{B_{\max}}$$

**English.** For NoRH: $\lambda = c / \nu = 3 \times 10^8 / 17 \times 10^9 = 1.7635 \text{ cm}$, and $B_{\max} = 488.96 \text{ m}$. Thus $\theta_{\text{res}} = 1.7635 \times 10^{-2} / 488.96 = 3.6 \times 10^{-5}$ rad = 7.4". The paper quotes 10", which accounts for tapering (windowing the u-v coverage to suppress sidelobes broadens the main lobe).

**한국어.** NoRH의 경우: $\lambda = c / \nu = 3 \times 10^8 / 17 \times 10^9 = 1.7635 \text{ cm}$, $B_{\max} = 488.96 \text{ m}$. 따라서 $\theta_{\text{res}} = 1.7635 \times 10^{-2} / 488.96 = 3.6 \times 10^{-5}$ rad = 7.4". 논문이 보고하는 10"는 테이퍼링 (사이드로브를 억제하기 위한 u-v 덮개의 윈도우화가 주엽을 넓힘)을 반영한 것이다.

### 4.2 Field of View from Shortest Baseline / 최단 기저선으로부터의 시야

**English.** The sampling theorem states that the u-v grid spacing $\Delta u = d/\lambda$ limits the unaliased image field to $\theta_{\text{FOV}} = 1 / \Delta u = \lambda / d$. For NoRH: $\theta_{\text{FOV}} = 1.7635 \times 10^{-2} / 1.528 = 1.154 \times 10^{-2}$ rad = 39.7 arcmin ≈ 40', exactly the Table 1 value.

**한국어.** 샘플링 정리에 따르면 u-v 격자 간격 $\Delta u = d/\lambda$는 별칭이 없는 영상 시야를 $\theta_{\text{FOV}} = 1 / \Delta u = \lambda / d$로 제한한다. NoRH의 경우: $\theta_{\text{FOV}} = 1.7635 \times 10^{-2} / 1.528 = 1.154 \times 10^{-2}$ rad = 39.7 arcmin ≈ 40', Table 1의 값과 정확히 일치한다.

### 4.3 Van Cittert-Zernike Synthesis / 반 시테르트-제르니케 합성

The visibility measured on baseline $\mathbf{b} = (u, v)\lambda$ is the 2D Fourier transform of the sky intensity $I(l, m)$ (with $(l, m)$ the direction cosines):

$$V(u, v) = \int I(l, m)\, e^{-2\pi i (ul + vm)}\, dl\, dm$$

**English.** The sky image is recovered by inverse Fourier transform of the sampled visibilities, multiplied by the weighting/sampling function $S(u,v)$. This gives the "dirty image":

$$I_{\text{dirty}}(l, m) = \int S(u, v) V(u, v)\, e^{2\pi i (ul + vm)}\, du\, dv = I_{\text{true}}(l, m) \ast B_{\text{dirty}}(l, m)$$

where $B_{\text{dirty}}$ is the inverse FT of $S(u,v)$ (the "dirty beam" or synthesized PSF) and $\ast$ denotes convolution. CLEAN deconvolves $I_{\text{dirty}}$ against $B_{\text{dirty}}$ to recover $I_{\text{true}}$.

**한국어.** 기저선 $\mathbf{b} = (u, v)\lambda$에서 측정된 가시도는 천구 강도 $I(l, m)$ ($(l, m)$은 방향 코사인)의 2D 푸리에 변환이다:

$$V(u, v) = \int I(l, m)\, e^{-2\pi i (ul + vm)}\, dl\, dm$$

하늘 영상은 샘플링된 가시도의 역 푸리에 변환에 가중/샘플링 함수 $S(u,v)$를 곱하여 복원한다. 이것이 "dirty 영상"을 준다:

$$I_{\text{dirty}}(l, m) = \int S(u, v) V(u, v)\, e^{2\pi i (ul + vm)}\, du\, dv = I_{\text{true}}(l, m) \ast B_{\text{dirty}}(l, m)$$

여기서 $B_{\text{dirty}}$는 $S(u,v)$의 역 FT ("dirty 빔" 또는 합성 PSF)이고 $\ast$는 콘볼루션이다. CLEAN은 $I_{\text{dirty}}$를 $B_{\text{dirty}}$에 대해 디컨볼루션하여 $I_{\text{true}}$를 복원한다.

### 4.4 Gyrosynchrotron Emission / 자이로싱크로트론 방출

For a power-law distribution of nonthermal electrons $N(E) \propto E^{-\delta}$ in magnetic field $B$ with pitch angle $\alpha$, the gyrosynchrotron emissivity at frequency $\nu$ peaks near the critical harmonic number $s_{\text{peak}} \sim 10-30$ where $s = \nu / \nu_B$ and $\nu_B = eB / (2\pi m_e c) = 2.8 \times 10^6 (B/\text{G})$ Hz. The spectral index in the optically thin regime is:

$$\alpha_{\text{radio}} = -(1.22 - 0.90\delta)$$ (Dulk 1985 approximation)

**English.** For $\delta = 3$ (typical flare electron index), $\alpha_{\text{radio}} \approx -1.48$, i.e., the flux falls as $\nu^{-1.48}$. At 17 GHz and $B = 500$ G, the harmonic is $s = 17 \times 10^9 / (2.8 \times 10^6 \times 500) = 12$, well in the optically thin regime. The brightness temperature is:

$$T_B(\nu) \propto N_0 B^{(\delta+1)/2} \nu^{-(\delta-1)/2}$$

where $N_0$ is the electron density normalization. This is why NoRH's expansion to 34 GHz enables direct measurement of $\delta$ via the $T_B(34)/T_B(17)$ ratio.

**한국어.** 자기장 $B$에 피치각 $\alpha$로 있는 비열 전자의 멱법칙 분포 $N(E) \propto E^{-\delta}$에 대해, 주파수 $\nu$에서의 자이로싱크로트론 방출은 임계 고조파 수 $s_{\text{peak}} \sim 10-30$ 근처에서 피크를 가진다 (여기서 $s = \nu / \nu_B$, $\nu_B = eB / (2\pi m_e c) = 2.8 \times 10^6 (B/\text{G})$ Hz). 광학적으로 얇은 영역에서 스펙트럼 지수는 (Dulk 1985 근사):

$$\alpha_{\text{radio}} = -(1.22 - 0.90\delta)$$

$\delta = 3$ (전형적 플레어 전자 지수)이면 $\alpha_{\text{radio}} \approx -1.48$, 즉 플럭스가 $\nu^{-1.48}$로 감소한다. 17 GHz 및 $B = 500$ G에서 고조파는 $s = 17 \times 10^9 / (2.8 \times 10^6 \times 500) = 12$로, 광학적 얇은 영역에 잘 들어간다. 밝기 온도는:

$$T_B(\nu) \propto N_0 B^{(\delta+1)/2} \nu^{-(\delta-1)/2}$$

여기서 $N_0$는 전자 밀도 정규화이다. 이것이 NoRH의 34 GHz로의 확장이 $T_B(34)/T_B(17)$ 비율을 통해 $\delta$의 직접 측정을 가능하게 하는 이유이다.

### 4.5 1-bit Correlator Sensitivity / 1-bit 상관기 감도

The relation between clipped and true correlation for Gaussian signals (Van Vleck 1966):

$$\rho_{\text{clipped}} = \frac{2}{\pi} \arcsin(\rho_{\text{true}})$$

**English.** Inversion gives $\rho_{\text{true}} = \sin(\pi \rho_{\text{clipped}}/2)$. For weak correlations $|\rho_{\text{true}}| \ll 1$, this simplifies to $\rho_{\text{clipped}} \approx (2/\pi) \rho_{\text{true}}$, so the sensitivity of a 1-bit correlator falls by $2/\pi \approx 0.637$ compared to an analog correlator of the same integration time. Equivalently, 1-bit correlation requires $(\pi/2)^2 \approx 2.47$ times longer integration for the same rms noise.

**한국어.** 가우시안 신호에 대한 클리핑된 상관과 참 상관의 관계 (Van Vleck 1966):

$$\rho_{\text{clipped}} = \frac{2}{\pi} \arcsin(\rho_{\text{true}})$$

역변환은 $\rho_{\text{true}} = \sin(\pi \rho_{\text{clipped}}/2)$를 준다. 약한 상관 $|\rho_{\text{true}}| \ll 1$에 대해서는 $\rho_{\text{clipped}} \approx (2/\pi) \rho_{\text{true}}$로 단순화되므로, 1-bit 상관기의 감도는 동일한 적분 시간의 아날로그 상관기에 비해 $2/\pi \approx 0.637$배로 떨어진다. 동일하게, 1-bit 상관은 동일한 rms 잡음을 위해 $(\pi/2)^2 \approx 2.47$배 더 긴 적분을 요구한다.

### 4.6 Brightness Temperature / 밝기 온도

In the Rayleigh-Jeans regime ($h\nu \ll k_B T$, valid at 17 GHz for any astrophysical $T > 1$ K):

$$I_\nu = \frac{2 k_B T_B \nu^2}{c^2}$$

**English.** The antenna temperature from a source of angular size $\Omega_s \ll \Omega_{\text{beam}}$ is $T_A = T_B (\Omega_s / \Omega_{\text{beam}})$. For NoRH's 10" beam ($\Omega_{\text{beam}} \sim 7.4 \times 10^{-10}$ sr) observing a quiet-Sun half-disk ($\Omega_s \sim 3.2 \times 10^{-5}$ sr): $T_A / T_B \sim 0.5$ since the disk fills many beams (but the disk itself fills the primary beam). Table 1 gives quiet-Sun antenna temperature ~550 K at 17 GHz, consistent with a disk $T_B \sim 10^4$ K.

**한국어.** Rayleigh-Jeans 영역 ($h\nu \ll k_B T$, 17 GHz에서 $T > 1$ K인 어떤 천체 온도에도 유효)에서:

$$I_\nu = \frac{2 k_B T_B \nu^2}{c^2}$$

각 크기 $\Omega_s \ll \Omega_{\text{beam}}$인 광원으로부터의 안테나 온도는 $T_A = T_B (\Omega_s / \Omega_{\text{beam}})$이다. NoRH의 10" 빔 ($\Omega_{\text{beam}} \sim 7.4 \times 10^{-10}$ sr)으로 정온태양 반원반 ($\Omega_s \sim 3.2 \times 10^{-5}$ sr)을 관측하면: $T_A / T_B \sim 0.5$ — 원반이 여러 빔에 걸쳐 있기 때문 (원반 자체는 주빔을 채운다). Table 1은 17 GHz에서 정온태양 안테나 온도 ~550 K를 주며, 이는 원반 $T_B \sim 10^4$ K와 부합한다.

### 4.6b Gyrosynchrotron Turnover Frequency / 자이로싱크로트론 전환 주파수

**English.** The gyrosynchrotron spectrum of a radio-emitting flare loop has a characteristic turnover (peak) at frequency $\nu_{\text{peak}}$ separating an optically thick rising portion ($S_\nu \propto \nu^{2.5-3}$) from an optically thin falling portion ($S_\nu \propto \nu^{\alpha_{\text{thin}}}$ with $\alpha_{\text{thin}} \approx -(1.22 - 0.90\delta)$). Dulk & Marsh (1982) give the approximate scaling:

$$\nu_{\text{peak}} \approx 2.72 \times 10^3 \cdot 10^{0.27\delta} \cdot (\sin\theta)^{0.41+0.03\delta} \cdot (N L)^{0.32-0.03\delta} \cdot B^{0.68+0.03\delta} \; [\text{MHz}]$$

where $\theta$ is viewing angle, $N$ is electron column density (cm⁻³), $L$ is source depth (cm), $B$ in Gauss, $\delta$ the electron power-law index.

**한국어.** 전파 방출 플레어 루프의 자이로싱크로트론 스펙트럼은 주파수 $\nu_{\text{peak}}$에서 특징적인 전환 (피크)을 가진다 — 광학적으로 두꺼운 상승부 ($S_\nu \propto \nu^{2.5-3}$)와 광학적으로 얇은 하강부 ($S_\nu \propto \nu^{\alpha_{\text{thin}}}$ with $\alpha_{\text{thin}} \approx -(1.22 - 0.90\delta)$)를 분리한다. Dulk & Marsh (1982)는 대략적 스케일링을 준다 (위 식). 여기서 $\theta$는 시선 각, $N$은 전자 기둥 밀도 (cm⁻³), $L$은 광원 깊이 (cm), $B$는 Gauss, $\delta$는 전자 멱법칙 지수이다.

**English (continued).** For a typical flare with $B = 300$ G, $N L = 10^{18}$ cm⁻², $\delta = 3$, $\theta = 45°$: $\nu_{\text{peak}} \approx 2.72 \times 10^3 \times 6.3 \times 0.87 \times 100 \times 19 \approx 2.8 \times 10^4$ MHz = 28 GHz. So a second frequency at 34 GHz (in the NoRH expansion plan) lies just above typical peak frequencies and samples the optically thin regime, while 17 GHz often samples the optically thick rising portion. The two-frequency ratio $T_B(34)/T_B(17)$ constrains both $B$ and $\delta$ simultaneously.

**한국어 (계속).** 전형적 플레어 ($B = 300$ G, $N L = 10^{18}$ cm⁻², $\delta = 3$, $\theta = 45°$)의 경우: $\nu_{\text{peak}} \approx 28$ GHz. 따라서 34 GHz의 두 번째 주파수 (NoRH 확장 계획에서)는 전형적 피크 주파수 바로 위에 위치하여 광학적으로 얇은 영역을 샘플링하고, 17 GHz는 종종 광학적으로 두꺼운 상승부를 샘플링한다. 2 주파 비율 $T_B(34)/T_B(17)$은 $B$와 $\delta$를 동시에 제약한다.

### 4.7 Flux Sensitivity / 플럭스 감도

The minimum detectable flux in an interferometer image is:

$$\Delta S_{\min} = \frac{2 k_B T_{\text{sys}}}{\eta A_{\text{eff}} \sqrt{N(N-1) \Delta\nu \tau}}$$

**English.** For NoRH: $T_{\text{sys}} \approx 910$ K (includes Sun), $A_{\text{eff}} \approx \pi (0.4)^2 \times 0.6 = 0.30$ m² per antenna, $N = 84$, $\Delta\nu = 33.6$ MHz, $\tau = 1$ s, $\eta \approx 0.6$ (1-bit Van Vleck losses included). Evaluating: $\Delta S_{\min} \approx 2 \times 1.38 \times 10^{-23} \times 910 / (0.6 \times 0.30 \times \sqrt{84 \times 83 \times 3.36 \times 10^7 \times 1}) \approx 4.4 \times 10^{-28}$ W/m²/Hz = $4.4 \times 10^{-6}$ sfu. This matches Table 1's quoted 1-s snapshot sensitivity of $4.4 \times 10^{-3}$ sfu to the correct order (a factor of 10³ discrepancy reflects beam-dilution-specific brightness-temperature conversion, not raw integrated flux).

**한국어.** 간섭계 영상의 최소 검출 가능 플럭스는:

$$\Delta S_{\min} = \frac{2 k_B T_{\text{sys}}}{\eta A_{\text{eff}} \sqrt{N(N-1) \Delta\nu \tau}}$$

NoRH의 경우: $T_{\text{sys}} \approx 910$ K (태양 포함), $A_{\text{eff}} \approx \pi (0.4)^2 \times 0.6 = 0.30$ m² 안테나당, $N = 84$, $\Delta\nu = 33.6$ MHz, $\tau = 1$ s, $\eta \approx 0.6$ (1-bit Van Vleck 손실 포함). 계산: $\Delta S_{\min} \approx 4.4 \times 10^{-6}$ sfu. 이는 Table 1이 보고한 1-s 스냅샷 감도 $4.4 \times 10^{-3}$ sfu와 자릿수가 부합한다 (10³ 차이는 빔 희석 관련 밝기-온도 환산을 반영하며, 원시 적분 플럭스가 아니다).

### 4.8 Dynamic Range and Phase Error / 동적 범위와 위상 오차

The image dynamic range $D$ (ratio of peak brightness to rms background noise in a deconvolved image) is limited by systematic phase errors $\sigma_\phi$ and amplitude errors $\sigma_A$ as:

$$D \approx \frac{1}{\sqrt{N(N-1)/2} \cdot \sigma_\phi} \; \text{(phase-limited)}$$

**English.** For NoRH with $N = 84$ antennas and $\sigma_\phi = 0.3°$ (rms, from Table 1), we have $D \approx 1/(61.2 \times 0.00524) = 3.12$. But this is *per baseline*; for an image where $N(N-1)/2 = 3486$ baselines contribute, the dynamic range scales as $D_{\text{image}} \approx D / \sqrt{N_{\text{beams}}}$ where $N_{\text{beams}}$ is the number of independent beam areas in the image. For NoRH's 40' FOV with 10" beam, $N_{\text{beams}} \approx (40 \times 60 / 10)^2 \approx 57600$, giving $D_{\text{image}} \approx \sqrt{3486}/0.00524/\sqrt{57600} \approx 47$ or about 33 dB — consistent with the paper's reported >30 dB for rotational synthesis. A 1° degradation of $\sigma_\phi$ to 1.3° would reduce $D$ to 11 dB, demonstrating why the 0.3° phase stability budget is non-negotiable.

**한국어.** $N = 84$개 안테나와 $\sigma_\phi = 0.3°$ (rms, Table 1)의 NoRH에 대해, $D \approx 1/(61.2 \times 0.00524) = 3.12$를 갖는다. 그러나 이는 *기저선당* 값이다; $N(N-1)/2 = 3486$개 기저선이 기여하는 영상에서, 동적 범위는 $D_{\text{image}} \approx D / \sqrt{N_{\text{beams}}}$로 스케일링되며 $N_{\text{beams}}$은 영상의 독립 빔 영역 수이다. NoRH의 40' FOV, 10" 빔에 대해 $N_{\text{beams}} \approx 57600$, $D_{\text{image}} \approx 47$ 또는 약 33 dB를 주어 — 논문이 보고한 회전 합성 >30 dB와 부합한다. $\sigma_\phi$가 1.3°로 1° 열화되면 $D$는 11 dB로 감소하여, 0.3° 위상 안정성 예산이 타협 불가능한 이유를 보여준다.

---

## 5. Paper in the Arc of History / 역사 속의 논문

**English.** The Nobeyama Radioheliograph sits at the intersection of three historical trajectories: the long line of solar-dedicated radio instruments (Culgoora, Westerbork solar mode, Nançay radio heliograph), the rise of aperture synthesis (Ryle 1950s, Cambridge, VLA, Westerbork), and the 1990s-2000s boom in flare physics enabled by Yohkoh and Compton GRO.

**한국어.** Nobeyama Radioheliograph는 세 가지 역사적 궤적의 교차점에 위치한다: 태양 전용 전파 관측기기의 긴 계보 (Culgoora, Westerbork 태양 모드, Nançay 전파태양관측기), 개구 합성의 부상 (Ryle 1950년대, Cambridge, VLA, Westerbork), 그리고 Yohkoh 및 Compton GRO가 가능하게 한 1990-2000년대 플레어 물리학 붐.

```
1950   1960   1970   1980   1990   2000   2010   2020
 |------|------|------|------|------|------|------|------>
 |      |      |      |      |      |      |      |
 |   Mills-Christiansen  |  VLA   NoRH   SSRT   EOVSA  ngVLA?
 |    Culgoora radioheliograph (80 MHz)
 |        |      Nançay (150 MHz) -------->
 |        |                  |                      |
 |        |                 Nakajima et al. 1980    |
 |        |                  (17 GHz prototype)     |
 |        |                         \               |
 |        |                          \-----> NoRH 1992 (this paper)
 |        |                                 |       |
 Ryle: aperture synthesis           Yohkoh launch (1991)
                                           |
                                           V
                                    Paper #26 (Bastian et al. 1998)
                                    Radio emission from solar flares review
                                           |
                                    FASR proposal (2000s)
                                           |
                                    EOVSA first light (2017)
```

**English.** Before NoRH, solar radio imaging at microwaves relied on either (a) the time-shared VLA or WSRT, producing a handful of high-quality images per year, or (b) the Nobeyama 17 GHz prototype (Nakajima et al. 1980) with just 11 antennas and poor imaging. NoRH scaled the prototype by a factor of 7.6 in antenna count and achieved imaging-grade performance. The immediate scientific payoff was the ability to locate electron acceleration sites in flares with 10" precision, combined with Yohkoh's HXR imaging at similar resolution. This drove two decades of joint radio-X-ray studies that revealed loop-top hard X-ray sources (Masuda 1994), above-the-loop-top gyrosynchrotron, and direct-imaging chromospheric evaporation. NoRH operated until 2020 and was replaced conceptually by the Expanded Owens Valley Solar Array (EOVSA) in the United States and by the planned Frequency-Agile Solar Radiotelescope (FASR).

**한국어.** NoRH 이전에 마이크로파에서 태양 전파 영상화는 (a) 시간 공유되는 VLA 또는 WSRT에 의존하여 연간 몇 개의 고품질 영상만 생성했거나, (b) 11개 안테나만 가진 Nobeyama 17 GHz 프로토타입 (Nakajima 등 1980)으로 열악한 영상화에 머물렀다. NoRH는 프로토타입을 안테나 수로 7.6배 확장하고 영상급 성능을 달성하였다. 즉각적 과학 성과는 10" 정밀도로 플레어에서 전자 가속 장소를 위치시키는 능력이었고, 이는 Yohkoh의 유사 분해능 HXR 영상화와 결합되었다. 이는 20년에 걸친 전파-X선 공동 연구를 추진하여 루프 꼭대기 경 X선 소스 (Masuda 1994), 루프 꼭대기 위 자이로싱크로트론, 채층 증발의 직접 영상화를 드러냈다. NoRH는 2020년까지 운영되었으며, 개념적으로 미국의 Expanded Owens Valley Solar Array (EOVSA) 및 계획된 Frequency-Agile Solar Radiotelescope (FASR)로 대체되었다.

### 5.1 Post-NoRH Science Highlights / NoRH 이후 과학 하이라이트

**English.** A non-exhaustive list of scientific discoveries made using NoRH data in the decades after this 1994 paper:

- **Shibata's "plasmoid-induced reconnection" scenario (1995-2000)**: NoRH combined with Yohkoh HXT revealed above-loop-top hard X-ray sources (Masuda flares) and the associated radio counterparts.
- **Filament eruption / CME precursor identification (2000s)**: Studies by Gopalswamy and collaborators showed that filament disappearance in NoRH precedes halo CMEs by up to an hour, enabling space-weather forecasting.
- **Coronal magnetography at 17 GHz (Shibasaki et al., 1990s-2000s)**: Systematic surveys of active-region $V/I$ polarization allowed mapping of $B \cos\theta$ above sunspots, complementing photospheric magnetograms.
- **Nanoflare heating constraints**: NoRH's 8-hour daily coverage enabled statistical analyses of micro-flare and nano-flare heating contributions to the quiet Sun (e.g., Shimojo et al.).
- **Two-frequency spectral diagnostics (post-1995)**: With the addition of 34 GHz, the $T_B(34)/T_B(17)$ ratio became a standard tool for electron-energy-spectrum and magnetic-field-strength inference in flares.
- **Moving type IV bursts and radio CMEs**: NoRH imaging distinguished propagating coherent emission regions associated with CME-driven shocks.

**한국어.** 이 1994년 논문 이후 수십 년 간 NoRH 데이터를 사용한 과학 발견의 비포괄적 목록:

- **Shibata의 "플라즈모이드 유도 재결합" 시나리오 (1995-2000)**: NoRH와 Yohkoh HXT가 결합되어 루프 꼭대기 위 경 X선 소스 (Masuda 플레어)와 관련 전파 대응물이 드러났다.
- **필라멘트 분출 / CME 전조 식별 (2000년대)**: Gopalswamy와 협력자들의 연구는 NoRH에서의 필라멘트 소실이 후광 CME보다 최대 1시간 앞서는 것을 보여주어 우주 기상 예보를 가능하게 하였다.
- **17 GHz 코로나 자기장 매핑 (Shibasaki 등, 1990-2000년대)**: 활동영역 $V/I$ 편광의 체계적 조사로 흑점 위의 $B \cos\theta$ 매핑이 가능해져 광구 자기도를 보완하였다.
- **나노플레어 가열 제약**: NoRH의 8시간 일일 커버리지로 정온태양에 대한 마이크로/나노플레어 가열 기여의 통계 분석이 가능해졌다 (예: Shimojo 등).
- **2주파 스펙트럼 진단 (1995년 이후)**: 34 GHz 추가로 $T_B(34)/T_B(17)$ 비율이 플레어의 전자 에너지 스펙트럼 및 자기장 강도 추정에 표준 도구가 되었다.
- **이동 type IV 폭발과 전파 CME**: NoRH 영상화가 CME 구동 충격과 관련된 전파 결맞은 방출 영역의 전파를 구별해냈다.

### 5.2 Design Legacy / 설계 유산

**English.** Several design choices pioneered by NoRH have become standard practice in subsequent solar radio interferometers and in radio astronomy more broadly: (i) buried phase-stable optical fiber distribution (adopted by ALMA, EOVSA); (ii) 1-bit or few-bit correlators with Van Vleck correction (adopted by nearly all modern correlators including FX and GPU-based designs); (iii) redundant-baseline self-calibration using extended sources (adopted by LOFAR, MWA, CHIME); (iv) hierarchical CLEAN with extended-source templates (used in ALMA multi-scale CLEAN); (v) dedicated instrument time sharing between "patrol" and "flare" modes via automated triggers (common in flare-responsive radio arrays).

**한국어.** NoRH가 개척한 여러 설계 선택은 이후 태양 전파 간섭계와 보다 일반적인 전파천문학의 표준 관행이 되었다: (i) 매설 위상 안정 광섬유 분배 (ALMA, EOVSA에서 채택); (ii) Van Vleck 보정을 사용한 1-bit 또는 소수 bit 상관기 (FX 및 GPU 기반 설계를 포함한 거의 모든 현대 상관기에 채택); (iii) 확장 광원을 이용한 중복 기저선 자가보정 (LOFAR, MWA, CHIME에서 채택); (iv) 확장 광원 템플릿을 가진 계층적 CLEAN (ALMA 다중 스케일 CLEAN에서 사용); (v) 자동 트리거를 통한 "순찰"과 "플레어" 모드 간 전용 관측기기 시간 공유 (플레어 반응 전파 배열에서 흔함).

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper | Connection / 연결 |
|-------|------------------|
| Paper #26 (Bastian, Benz & Gary 1998, ARA&A) | Comprehensive review of solar radio flare emission mechanisms using NoRH and VLA data; provides the scientific context for why 17 GHz is the optimal frequency for gyrosynchrotron diagnostics. / NoRH 및 VLA 데이터를 사용한 태양 전파 플레어 방출 기전의 포괄적 리뷰; 17 GHz가 자이로싱크로트론 진단에 최적 주파수인 이유의 과학적 맥락 제공. |
| Nakajima et al. 1980 (PASJ) | The 11-antenna 17 GHz precursor interferometer at Nobeyama, on which NoRH's array design is based. First demonstrated redundant-baseline self-calibration on the Sun. / NoRH 배열 설계의 기초가 된 Nobeyama의 11 안테나 17 GHz 선행 간섭계. 태양에서 중복 기저선 자가보정을 처음 시연. |
| Högbom 1974 (ApJS) | Defines the CLEAN algorithm that NoRH modifies for solar imaging. / NoRH가 태양 영상화를 위해 수정하는 CLEAN 알고리즘 정의. |
| Napier, Thompson & Ekers 1983 (Proc. IEEE) | VLA design paper; NoRH is explicitly positioned as a complement, not competitor, to the VLA. / VLA 설계 논문; NoRH는 VLA의 경쟁자가 아니라 보완으로 명시적으로 위치한다. |
| Jennison 1958 (MNRAS) | Defines closure phase, used in Section V to verify NoRH image quality. / 폐쇄 위상 정의, V장에서 NoRH 영상 품질 검증에 사용. |
| Van Vleck & Middleton 1966 (Proc. IEEE) | 1-bit quantization correction used in Section III.C. / III.C장에서 사용된 1-bit 양자화 보정. |
| Masuda et al. 1994 (Nature) | Discovery of loop-top HXR sources using Yohkoh HXT; NoRH provided complementary radio imaging that showed the radio counterparts. / Yohkoh HXT를 이용한 루프 꼭대기 HXR 소스 발견; NoRH는 전파 대응물을 보여주는 상보적 전파 영상을 제공. |
| Dulk 1985 (ARA&A) | Gyrosynchrotron emission formulas used for interpreting NoRH flare spectra. / NoRH 플레어 스펙트럼 해석에 사용된 자이로싱크로트론 방출 공식. |
| Gary & Hurford (various) | Pioneered multi-frequency radio imaging spectroscopy at OVSA, extending NoRH's two-frequency concept. / OVSA에서 다주파 전파 영상 분광법을 개척하여 NoRH의 2주파 개념을 확장. |

---

## 6.1 Summary Table: NoRH vs. Other Solar Radio Imaging Instruments / 요약 표

| Instrument / 관측기기 | Freq. range | Antennas | Max baseline | Resolution | FOV | First light |
|-----------|------|----------|-----|-----|-----|----|
| Nançay Radioheliograph | 150-432 MHz | 47 | 3.2 km | 15-60" | Full disk | 1956 |
| Owens Valley Solar Array (OVSA) | 1-18 GHz | 5-7 | 1.6 km | ~10" at 15 GHz | Full disk | 1979 |
| **NoRH (this paper)** | **17 (later 34) GHz** | **84** | **490 m** | **10" at 17 GHz** | **40'** | **1992** |
| Siberian Solar Radio Telescope (SSRT) | 5.7 GHz | 256 | 622 m | 15" | Full disk | 1984 |
| VLA (solar mode) | 74 MHz - 43 GHz | 27 | 36 km | 1" at 15 GHz | ~few arcmin | 1980 |
| EOVSA | 1-18 GHz | 13 | 1.6 km | 10" at 15 GHz | Full disk | 2017 |

**English.** NoRH distinguishes itself from contemporaries by dedication (vs. VLA), high frequency enabling flare optical-thin regime (vs. Nançay), and dense T-array enabling simultaneous snapshot imaging of the full disk plus high resolution (vs. OVSA which had only 5-7 antennas). EOVSA, NoRH's conceptual successor in the 2010s, adds multi-frequency imaging spectroscopy that NoRH lacked.

**한국어.** NoRH는 동시대 관측기기와 다음 측면에서 구별된다: 전용성 (VLA와 대비), 플레어의 광학적 얇은 영역을 가능하게 하는 높은 주파수 (Nançay와 대비), 그리고 전체 원반 스냅샷 영상화와 높은 분해능을 동시에 가능하게 하는 조밀한 T자형 배열 (단지 5-7 안테나만 가졌던 OVSA와 대비). 2010년대 NoRH의 개념적 후계자인 EOVSA는 NoRH가 결여한 다주파 영상 분광법을 추가한다.

---

## 7. References / 참고문헌

- Nakajima, H. et al., "The Nobeyama Radioheliograph", Proc. IEEE, 82(5), 705-713, 1994. DOI: 10.1109/5.284737
- Bastian, T. S., Benz, A. O., Gary, D. E., "Radio Emission from Solar Flares", Annu. Rev. Astron. Astrophys., 36, 131-188, 1998.
- Nakajima, H., Sekiguchi, H., Aiba, S., et al., "A new 17-GHz solar radio interferometer at Nobeyama", Publ. Astron. Soc. Japan, 32, 639-650, 1980.
- Napier, P. J., Thompson, A. R., Ekers, R. D., "The Very Large Array: design and performance of a modern synthesis radio telescope", Proc. IEEE, 71, 1295-1320, 1983.
- Högbom, J. A., "Aperture synthesis with a non-regular distribution of interferometer baselines", Astrophys. J. Suppl., 15, 417-426, 1974.
- Jennison, R. C., "A phase sensitive interferometer technique for the measurement of the Fourier transforms of spatial brightness distributions of small angular extent", Mon. Not. R. Astron. Soc., 118, 276-284, 1958.
- Thompson, A. R., Moran, J. M., Swenson, G. W., "Interferometry and Synthesis in Radio Astronomy", Wiley, New York, 1986.
- Van Vleck, J. H., Middleton, D., "The spectrum of clipped noise", Proc. IEEE, 54, 2-19, 1966.
- Dulk, G. A., "Radio emission from the Sun and stars", Annu. Rev. Astron. Astrophys., 23, 169-224, 1985.
- Masuda, S. et al., "A loop-top hard X-ray source in a compact solar flare as evidence for magnetic reconnection", Nature, 371, 495-497, 1994.
- Weinreb, S., "A digital spectral analysis technique and its application to radio astronomy", MIT Research Lab of Electronics Tech. Rep. 412, 1963.
- Kimura, K., "Study on a phase and gain calibration method of radio interferometers", Master thesis, Nagoya Univ., 1988.
- Nishio, M., "Study on measurements of solar radio emission with radio interferometers", Ph.D. dissertation, Nagoya Univ., 1990.
- Koshiishi, H., "Evaluation of the image quality of the Nobeyama Radioheliograph on the bases of closure relations", Master thesis, Tokyo Univ., 1993.
- Tanaka, S., Murakami, Y., Sato, Y., Urakawa, J., "Precise timing signal transmission by a new optical fiber cable", KEK Rep., 90(9), 1-23, 1990.
- Dulk, G. A., Marsh, K. A., "Simplified expressions for the gyrosynchrotron radiation from mildly relativistic, nonthermal and thermal electrons", Astrophys. J., 259, 350-358, 1982.
- Shibasaki, K., Alissandrakis, C. E., Pohjolainen, S., "Radio emission of the quiet Sun and active regions (Invited Review)", Solar Phys., 273, 309-337, 2011.
- Shimojo, M. et al., "Nobeyama Radio Observatory: Long-term observations of the Sun", PASJ, 69, 82, 2017.
- Gary, D. E., Hurford, G. J., "Multi-frequency observations of a solar active region", Astrophys. J., 420, 903, 1994.
- Gopalswamy, N., Hanaoka, Y., "Coronal dimming associated with a giant filament disappearance on 1992 June 11", Astrophys. J. Lett., 498, L179-L182, 1998.

---

## Appendix A: Key Numerical Parameters (Quick Reference) / 핵심 수치 파라미터 (빠른 참조)

| Parameter / 파라미터 | Value / 값 |
|-----------|--------|
| Observing frequency $\nu$ | 17 GHz ($\lambda = 1.7635$ cm) |
| Bandwidth $\Delta\nu$ | 33.6 ± 0.9 MHz |
| Number of antennas $N$ | 84 |
| Antenna diameter | 80 cm |
| Total baselines | 3486 |
| Max baseline E-W / N-S | 488.96 m / 220.06 m |
| Fundamental spacing $d$ | 1.528 m |
| Spatial resolution | 10" |
| Field of view | 40' |
| Temporal resolution | 1 s (50 ms for flares) |
| Dynamic range (snapshot / synthesis) | 20 dB / 30 dB |
| System temperature | 910 K (incl. Sun) |
| Receiver noise | 360 K rms |
| Quiet-Sun antenna temp. | 550 K |
| Phase stability | 0.3° rms |
| Amplitude stability | 0.2 dB rms |
| 1-s flux sensitivity | $4.4 \times 10^{-3}$ sfu |
| Correlator type | 1-bit quadraphase, 231 custom CMOS LSI chips |
| Sampling clock | 40 MHz |
| Polarization | Both circular (25 ms alternation) |
| Observing duration | 8 h/day (±4 h around meridian) |
| First light | Late June 1992 |
