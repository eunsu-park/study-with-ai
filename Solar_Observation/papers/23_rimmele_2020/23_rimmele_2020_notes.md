---
title: "The Daniel K. Inouye Solar Telescope — Observatory Overview"
authors: Thomas R. Rimmele, Mark Warner, Stephen L. Keil et al.
year: 2020
journal: "Solar Physics, Vol. 295, Article 172"
doi: "10.1007/s11207-020-01736-7"
topic: Solar_Observation
tags: [DKIST, solar telescope, off-axis Gregorian, adaptive optics, polarimetry, coronal magnetic field, ground-based observing, Haleakala]
status: completed
date_started: 2026-04-19
date_completed: 2026-04-19
---

# 23. The Daniel K. Inouye Solar Telescope — Observatory Overview / 다니엘 K. 이노우에 태양 망원경 — 관측소 개요

---

## 1. Core Contribution / 핵심 기여

### 한국어
이 논문은 **세계 최대의 태양 망원경인 DKIST (Daniel K. Inouye Solar Telescope)**의 설계, 능력, 과학적 동기, 운영 개념을 종합한 **플래그십 overview**이다. 하와이 Haleakalā(해발 3,067 m)에 건설된 **4 m 구경 비축(off-axis) Gregorian f/2 망원경**은 태양 표면에서 **약 20 km의 공간 분해능**(500 nm에서 0.026″)을 달성하며, 이는 태양 대기의 **광자 평균자유행로(photon mean-free path)**와 **압력 스케일 높이(pressure scale height)**가 작용하는 물리적 스케일이다. DKIST의 세 가지 혁신은 **(i)** 비축 설계가 중앙 차폐와 거미 회절을 제거해 **산란광을 획기적으로 낮춰 코로나 관측이 가능**한 점, **(ii)** 4 m 구경이 기존 46 cm Solar-C 대비 **75배의 집광 면적**을 제공해 $\sim 10^{-9}$ 수준의 **코로나 자기장 편광 신호**를 측정 가능하게 한 점, **(iii)** 1,600 액추에이터 고차 AO + 4-단계 캘리브레이션으로 **$5\times10^{-4}$ 편광 정확도**를 달성한 점이다. 5대의 first-light 기기(VBI, ViSP, DL-NIRSP, CRYO-NIRSP, VTF)는 380 nm–5000 nm를 동시 커버하며 FIDO(Facility Instrument Distribution Optics)를 통해 **다기기 동시 관측**이 가능하다. 2019년 12월 first light를 달성했고, 44년 운영(태양 자기 주기 2회)을 계획 중이며, 연간 3 PB의 데이터를 Boulder 소재 Data Center에서 배포한다.

### English
This paper is the **flagship overview** that comprehensively describes the design, capabilities, science drivers, and operational concepts of **DKIST — the world's largest solar telescope**. Built atop Haleakalā in Hawaiʻi (3,067 m elevation), the **4 m off-axis Gregorian f/2 telescope** achieves **≈ 20 km resolution on the Sun** (0.026″ at 500 nm), matching the **photon mean-free path** and **pressure scale height** of the solar atmosphere. Three innovations define DKIST: **(i)** the off-axis design eliminates central obstruction and spider diffraction, **drastically reducing scattered light to enable coronal observations**; **(ii)** the 4 m aperture provides **75× the collecting area of the 46 cm Solar-C coronagraph**, making measurable the **$\sim 10^{-9}$ coronal polarimetric signals** that encode the magnetic field; **(iii)** a 1,600-actuator high-order AO system combined with a four-stage calibration strategy delivers $5\times 10^{-4}$ **polarimetric accuracy**. Five first-light instruments (VBI, ViSP, DL-NIRSP, CRYO-NIRSP, VTF) cover 380–5000 nm, and the FIDO (Facility Instrument Distribution Optics) system enables **simultaneous multi-instrument observing**. First light was achieved in December 2019; DKIST is designed for a 44-year lifetime (two Hale cycles) and will deliver 3 PB of calibrated data per year through its Boulder-based Data Center.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction & Science Drivers (§1–§2) / 도입 및 과학적 동기

**한국어**
§1은 DKIST가 해결하려는 태양 물리학의 근본적 질문을 정의한다. 기존 1 m급 망원경(SST, DST, VTT)은 수십 년간 workhorse였지만, **현대 MHD 시뮬레이션이 예측하는 수십 km 스케일의 자기 구조**와 **코로나 자기장**을 관측할 능력이 없었다. Stenflo(2008)는 태양 대기의 "critical scales"가 **광자 평균자유행로와 압력 스케일 높이** 수준이라 주장했다 — 광구에서 이는 **약 70 km (0.1″)** 또는 그 이하이며, 이를 분해하려면 **1.6 μm 파장에서 최소 4 m 구경**이 필요하다. IR을 택한 이유는 (a) 태양 대기의 불투명도(opacity) 최저점이 1.6 μm에 있어 **가장 깊은 광구층**을 볼 수 있고, (b) Zeeman 분열 $\Delta\lambda_B \propto \lambda^2 B$에 따라 **IR에서 자기장 감도가 10배 증가**하기 때문이다.

§2는 과학 동기를 상세화한다. **Coronal magnetic field** 측정은 DKIST의 최우선 과학 목표이며, 코로나 밝기는 디스크의 $\sim 10^{-6}$, 편광 신호는 코로나의 $10^{-3}$–$10^{-4}$ 수준이라 **$\sim 10^{-9}$ 상대 신호**를 검출해야 한다. 이는 공간 분해능(1–2 arcsec면 충분)보다 **집광 면적**이 결정적임을 의미 — 4 m 망원경은 "light bucket"으로 기능한다. 기타 과학 동기:
- **Photospheric dynamo**: 1.56 μm Fe 라인에서의 약한 자기장 측정으로 surface dynamo 검증
- **Chromospheric 3D magnetic structure**: 플레어, CME, 코로나 가열의 열쇠
- **다중 메신저 협업**: Parker Solar Probe, Solar Orbiter와의 연계 관측

**English**
§1 frames DKIST's scientific motivation. While 1 m-class telescopes were workhorses for decades, they could not resolve the **tens-of-km scales predicted by modern MHD simulations** nor measure the **coronal magnetic field**. Stenflo (2008) argued the "critical scales" of the solar atmosphere are the **photon mean-free path and pressure scale height** — ~70 km (0.1″) in the photosphere. Resolving these at 1.6 μm (where solar opacity reaches a minimum) requires a 4 m aperture. IR is preferred because (a) opacity minimum at 1.6 μm gives access to the deepest photospheric layers, and (b) Zeeman splitting $\Delta\lambda_B \propto \lambda^2 B$ makes IR **10× more sensitive** to weak fields than visible.

§2 details the science drivers. **Coronal magnetometry** is DKIST's top priority — coronal brightness is $\sim 10^{-6}$ of the disk, polarization signals $10^{-3}$–$10^{-4}$ of corona, so we must detect $\sim 10^{-9}$ relative signals. Spatial resolution matters less (1–2″ suffices); collecting area dominates. A 4 m telescope acts as a "light bucket." Other drivers: photospheric dynamo verification via 1.56 μm Fe lines, chromospheric 3D magnetic structure, and multi-messenger coordination with Parker Solar Probe and Solar Orbiter.

### Part II: DKIST Design Overview (§3) / DKIST 설계 개요

**한국어**
§3은 시설 전체 구조를 설명한다. 전체 높이 **41.4 m**, 회전 기기 플랫폼 직경 **16.5 m**. 주요 서브시스템:
- **Alt-az 망원경 마운트**: Warner et al. (2006) 설계
- **열 제어 인클로저(co-rotating enclosure)**: 직경 22 m, 높이 26.6 m, 약 750톤
- **Coudé lab**: $20 \pm 0.5\,°$C로 온도 안정화, 회전 플랫폼 위에 5대 기기 탑재
- **Utility building**: 냉각수, HVAC, backup 전력
- **Support building**: 사무실, 기기 조립실, 원격 운영실
- **Base facility (Pukalani)**: 과학 지원 센터 (DSSC)

4가지 광학 그룹: (i) main telescope(M1, M2 + heat stop + Lyot stop + GOS), (ii) transfer optics(M3–M6), (iii) coudé optics(M7–M10 with DM at M10), (iv) FIDO(기기 빛 분배).

**English**
§3 introduces facility architecture: overall height **41.4 m**, rotating instrument platform diameter **16.5 m**. Subsystems: alt-az mount, co-rotating thermal enclosure (22 m diameter, 26.6 m tall, ~750 tons), coudé lab stabilized at $20 \pm 0.5$ °C on the rotating platform with five instruments, utility/support buildings, and the Pukalani base facility (DSSC) for remote operations. Four optical groups: main telescope (M1–M2 + stops + GOS), transfer optics (M3–M6), coudé optics (M7–M10 with deformable mirror at M10), and FIDO for beam distribution.

### Part III: Optical System (§4) / 광학 시스템

**한국어**
§4는 DKIST의 광학 설계를 심도있게 다룬다.

**4.1 Main Telescope**
- **M1 (primary)**: 4.24 m thin meniscus (clear aperture 4.0 m), **두께 75 mm** (열 제어용), **Zerodur** (Schott AG 제작), **12 m parent paraboloid의 off-axis segment** (off-axis offset 4 m, conic constant $-1$), 표면 figure error $< 25$ nm rms (1–100 mm 공간 주기), roughness 1.2 nm 달성 — "smoothest large mirror ever made" (Oh et al. 2016)
- **코팅**: unprotected aluminum (AMOS 벨기에에서 초기 코팅, 이후 Haleakalā AFRL에서 재코팅), 흡수율 120 W/m²
- **능동 지지**: 118개 축 액추에이터 + 24개 측면 지지 (gravity 및 thermal deformation 보정)
- **M2 (secondary)**: 0.65 m concave ellipsoid (conic $-0.54$), **SiC 경량 재질** (light-weight), **air-jet 냉각**으로 표면 온도 $\pm 1\,°$C 이내 유지, **6-DOF hexapod 마운트**로 active alignment, tip-tilt로 near-limb coronal 관측의 image-motion 보상

**4.2 Heat Stop & Lyot Stop**
- **Heat stop**: f/2 prime focus에 위치, 5 arcmin 과학 FOV 통과, **13 kW 중 98% 제거**, 열유속 밀도 **2.5 MW/m²** at prime focus. First-light에는 reflecting heat stop 임시 설치, 이후 reflector-absorber 조합으로 교체 예정
- **Lyot stop**: M2에 의해 형성된 pupil image에 위치, 주경 edge diffraction 제거 (코로나 관측 필수)

**4.3 Gregorian Optical Station (GOS)**
- f/13 Gregorian focus 근처에 위치 → polarimetry calibration optics, calibration targets, pinholes, calibration light sources 탑재
- Upper GOS: **3355 × 1042 × 660 mm**, 편광 calibration 광학계
- Lower GOS: **1105 × 1042 × 406.4 mm**, field stops (2.8 arcmin AO-corrected, 5 arcmin), limb occulter

**4.4 Transfer Optics (M3–M6)**
- M3 + M4: 빛을 고도축 위로 올림 → f/13 빔을 coudé lab 바닥 위 4 m에 이미지 형성
- M5 + M6: azimuth 축을 따라 coudé 회전축으로 정렬
- **22.5° 2회 반사** 선택 이유: 45° 1회 반사보다 **instrumental polarization 감소**
- **M5**: 275 mm 직경, AO 시스템의 **fast tip-tilt corrector**, 능동 냉각 SiC
- **M3**: 능동 pupil alignment (액체 냉각, 정사각형 SiC)

**4.5 Coudé Optics (M7–M10)**
- **M7**: 큰 flat mirror, 빛을 coudé rotator에 평행하게 재지향
- **M8**: off-axis parabola, collimated beam 생성, M1 entrance aperture를 DM(M10)에 이미지
- **M9**: 작은 coma 보정용 fold mirror — **M9a** (선택 가능 deployed 위치)는 **모든 빛을 CRYO-NIRSP로** 전송 (photon-starved coronal observing), 기타 configuration은 M9a를 빔에서 제거해 DM으로 보냄
- **M10 (DM)**: 1,600 액추에이터 Deformable Mirror, pupil plane에 위치, 고도로 전도성 있는 silicon face sheet + silicon heat sink, 열전(TEC) 냉각

**4.6 FIDO (Facility Instrument Distribution Optics)**
- Dichroic beamsplitters + mirrors로 빛을 파장별 기기에 분배
- 설정 변경은 **하루 소요** (수작업 재정렬) → proposal cycle 내에서 최소화
- **Wavefront Correction (WFC) beamsplitter**: 영구 설치, **4% 반사광을 AO wavefront sensor + context viewer**로 분기

**4.7 Thermal Control of Optics**
- M1: unprotected Al, **air-jet 냉각 (555 air jets)** on 뒤쪽, 표면 온도 0 ~ $-2$ °C ambient 이내 제어
- M2: 60 W/m² 흡수, air-jet 냉각
- M5: 390 W/m² 흡수 (5 arcmin FOV), air-jet 냉각 — liquid 냉각은 frequency 응답 못 미쳐 불가
- M3: 2 kW 흡수 (액체 냉각), M6: 350 W 흡수 (액체 냉각)
- M10 (DM): 90 W/m² 흡수, **Thermoelectric** 냉각

**4.8 Alignment Strategies**
- **Wavefront sensor at point sources (stars)**: 초기 정렬
- **Laser trackers, CMMs, interferometers, theodolites**: 정밀 계측
- **Active alignment**: LUT(Look-Up Table)과 wavefront sensor 조합으로 M3/M6 pupil/boresight drift 보정 (coudé rotator 회전 중)

**English**
§4 details DKIST's optical system.

The **primary mirror M1** is a 4.24 m Zerodur thin meniscus (75 mm thick), an off-axis segment of a 12 m parent paraboloid (4 m offset, conic $= -1$), polished to $< 25$ nm rms figure error over 1–100 mm spatial periods and 1.2 nm roughness. It is coated with unprotected aluminum (120 W/m² absorbed), actively supported on 118 axial actuators, and cooled from the back by 555 air jets to keep the surface within 0 to $-2$ °C of ambient. **M2** (0.65 m SiC ellipsoid, conic $-0.54$) is temperature-controlled by air jets to $\pm 1$ °C and mounted on a 6-DOF hexapod, enabling tip-tilt image-motion compensation for near-limb observations.

The **heat stop** at f/2 prime focus handles a density of **2.5 MW/m²**, rejecting ~98% of the 13 kW to preserve the 5 arcmin science FOV. A **Lyot stop** near the pupil image blocks primary-edge diffraction — essential for coronal work. The **GOS** at the f/13 Gregorian focus houses polarization-calibration optics, field stops, and a limb occulter.

**Transfer optics** M3–M6 relay the beam from the Gregorian focus to the coudé lab. Two 22.5° reflections (M5/M6) instead of one 45° reflection were chosen to **reduce instrumental polarization**. **M5** provides fast tip-tilt correction.

**Coudé optics** M7–M10: M7 redirects light parallel to the coudé rotator axis; M8 collimates and images the entrance aperture onto M10 (the DM); M9 provides coma correction. A deployable mirror **M9a** routes all light to the CRYO-NIRSP for photon-starved coronal observing. The **1,600-actuator DM** at pupil plane M10 is on a highly conductive silicon face sheet with a TEC cooler.

**FIDO** uses dichroic beamsplitters for multi-instrument distribution; a permanent **4% WFC beamsplitter** feeds the wavefront sensors. Configuration changes take ~1 day of manual realignment. **Thermal control**: M3/M6 are liquid cooled; M1, M2, M5 use air-jet cooling (liquid cooling insufficient for fast tip-tilt response).

### Part IV: Wavefront Correction System (§5) / 파면 보정 시스템

**한국어**
DKIST의 wavefront-correction(WFC) 시스템은 세 서브시스템:
1. **AO (Adaptive Optics)**: 1,600-actuator DM + **1,457 sub-aperture correlating Shack-Hartmann WFS** (Rimmele & Marino 2011), 2 kHz 업데이트 rate, 약 **1400 Karhunen-Loève (KL) modes** 보정
2. **aO (active Optics)**: M1 surface figure + M2 focus/astigmatism의 **slow correction** (수십 초–분 timescale)
3. **Active alignment**: M3/M6 pupil/boresight drift 보정

**Strehl 요구사항**:
- $r_0(500\,\text{nm}) > 7$ cm → $S(500) > 0.3$
- $r_0(630\,\text{nm}) > 20$ cm → $S(630) > 0.6$
- 이는 MHD 시뮬레이션 기반 simulated observations로부터 도출: Rempel 2006의 mixed-polarity field에서 Stokes-V를 인식 가능한 PSF Strehl 요건을 역산 (Rimmele & ATST SWG 2005)

**FPGA 기반 real-time control**: 1,600 액추에이터 × 2 kHz 주기 → **고속 FPGA reconstructor** (Richards et al. 2010). **Fitting error** (DM 재현 한계), **bandwidth error** (2 kHz 유한 업데이트) 두 가지가 median-seeing 성능을 제한.

**MCAO 업그레이드**: M7 (11 km conjugate), M9 (4 km conjugate) 위치는 **대기 터뷸런스 층에 near-optimal conjugate** → 향후 M7과 M9를 DM으로 교체해 3-DM MCAO 시스템 구축 예정 (Schmidt et al. 2017, 2018).

**First light 성능**: 2019년 12월 first light 당시 WFS telemetry로 $r_0 = 6$–$7$ cm 측정, Strehl ratio는 specification 근접. 열 제어 시스템 완전 가동 전이라 local seeing 영향 존재.

**English**
The WFC system has three subsystems:
1. **AO**: 1,600-actuator DM + **1,457 sub-aperture correlating Shack-Hartmann WFS**, 2 kHz update rate, correcting ~1,400 KL modes.
2. **aO**: slow correction of M1 figure and M2 focus/astigmatism (seconds to minutes).
3. **Active alignment**: M3/M6 corrections for pupil/boresight drift during coudé rotation.

**Strehl requirements** — $S(500) > 0.3$ for $r_0(500) > 7$ cm, $S(630) > 0.6$ for $r_0(630) > 20$ cm — derived from MHD-based simulated observations by folding AO-corrected PSFs of varying Strehl into Stokes-V simulations (Rimmele & ATST SWG 2005). The FPGA-based reconstructor handles $1,600 \times 2000$ /s updates. Fitting error (DM reproduction limit) and bandwidth error (finite 2 kHz) dominate median-seeing performance.

**MCAO upgrade path**: M7 (11 km conjugate) and M9 (4 km conjugate) sit at near-optimal turbulence-layer conjugate heights — replacing them with DMs yields a 3-DM MCAO system, already in design/prototyping (Schmidt et al. 2017, 2018).

### Part V: Polarimetry (§6) / 편광 측정

**한국어**
DKIST의 $5\times 10^{-4}$ polarimetric accuracy는 극한 사양이다. 전략:

1. **System Mueller matrix 분해**: 망원경 + relay optics + FIDO dichroics + instrument feed + instrument optics 각각의 sub-matrix $M_{\text{sys}} = M_{\text{inst}} M_{\text{feed}} M_{\text{FIDO}} M_{\text{relay}} M_{\text{tel}}$로 분해
2. **오차 구조**:
   - **Retardance sub-matrix** (3×3): 측정 Stokes vector의 방향 결정
   - **Depolarization row/column**: Stokes vector의 magnitude에 영향
   - **Demodulation matrix**: 위 파라미터 전반에 영향

3. **Calibration 목표**:
   - Continuum polarization stability: **0.05% for polarizer column**
   - Sensitivity: **0.001%** (photon statistics limit, 장시간 averaging 후)
4. **새로운 calibration 방법들**:
   - **Mueller-matrix spatial-spectral mapping systems** (UV–IR, Harrington et al. 2020a)
   - **Retarder mapping**: elliptical retardance의 공간적 mapping → ellipticity를 calibration에 포함
   - **Mirror-coating polarization**: 모든 코팅 표면에 대해 측정 (Harrington & Sueoka 2017; Harrington, Sueoka & White 2019)
   - **Dual fiber-fed spectrograph**: 390 nm – 1650 nm 커버, 3000+ fit variables
5. **검증 방법**:
   - **Line-correlation techniques** (Elmore et al. 2010; Derks, Beck & Martínez Pillet 2018)
   - **Daytime sky observations** (Harrington et al. 2015, 2017; Harrington, Kuhn & Hall 2011)
   - **Polarized star observations** (Bailey et al. 2008, 2010; Fossati et al. 2007)

**English**
DKIST's $5\times 10^{-4}$ polarimetric accuracy is cutting-edge. The approach decomposes the system Mueller matrix as
$$M_{\text{sys}} = M_{\text{inst}}\, M_{\text{feed}}\, M_{\text{FIDO}}\, M_{\text{relay}}\, M_{\text{tel}}$$
Errors are structured: the **3×3 retardance sub-matrix** sets measured Stokes orientation; **depolarization row/column** affects magnitude; the **demodulation matrix** couples everything. Calibration goals: 0.05% continuum-polarization stability; 0.001% sensitivity after averaging. Methods include Mueller-matrix spatial-spectral mapping (UV–IR), retarder-ellipticity mapping, mirror-coating-polarization measurement for every surface, a dual fiber-fed spectrograph (390–1650 nm) with 3,000+ fit variables, and validation via line-correlation, daytime sky, and polarized star observations.

### Part VI: Mechanical, Enclosure, and Facility Systems (§7–§10) / 기계·인클로저·시설 시스템

**한국어**
- **Telescope mount**: alt-az, off-pointing accuracy 0.5″, tracking stability < 0.5″ / hour, off-pointing range 1.5 $R_\odot$ (모든 방향). First-light pointing model (100 야간 object 관측)으로 **2.65″ 잔차** 측정, 2배 개선 가능
- **Coudé rotator**: 16.5 m 직경, **115 톤**, lateral bearing run-out **~50 μm**, MT Mechatronics 설계 + Ingersoll Machine Tools 제작, IDOM 공장에서 pre-assembly
- **Enclosure**: 높이 22 m × 직경 26.6 m, 약 750 톤, co-rotating ventilated 설계 (Murga et al. 2012), **large motorized vent gates**로 air-flow 제어 — 고풍속 시 과도한 jitter/M1 변형 방지
- **Thermal systems**: 야간에 얼음 제조 (12 km 파이프로 분배), **Coudé lab $20 \pm 0.5$ °C**, air-knife로 coudé lab과 enclosure 분리 (glass window는 IR 흡수로 불가)
- **Control room**: Summit + Pukalani mirror control room (DSSC), 초기에는 service-mode 중심

**English**
The alt-az mount supports M1–M6 assemblies with < 5″ absolute pointing, < 0.5″ off-pointing accuracy, and < 0.5″/hr tracking. The 16.5 m coudé rotator weighs 115 tons with ~50 μm lateral bearing run-out. The 750-ton co-rotating enclosure (22 m tall, 26.6 m diameter) uses large motorized vent gates to manage airflow — in high winds, throttled flow prevents excessive M1 deformation. Ice is produced nightly and distributed through 12 km of cooling pipes; the coudé lab is stabilized at $20 \pm 0.5$ °C, separated from the enclosure by an **air knife** (IR absorption precludes a glass window). Remote operation is planned via the Pukalani DSSC once service-mode matures.

### Part VII: Software & Instrumentation (§10–§11) / 소프트웨어 및 기기

**한국어**
**Software (§10)**:
- **OCS (Observatory Control System)**: top-level user interface, 실험·관측 실행
- **TCS (Telescope Control System)**: mount, pointing, ephemeris, thermal, wavefront 관리 (Observatory Sciences 제작)
- **ICS (Instrument Control System)**: 기기 카메라, polarizer, modulation, synchronization 제어
- **CSF (Common Services Framework)**: connections, events, logging, databases (Hubbard et al. 2010)

**Instrumentation (§11)** — 5대 first-light 기기, Table 2에 요약:

| 기기 / Instrument | 타입 / Type | 파장 / Spectral Range | 분해능 / Spectral Res. | FOV | Polarimeter? |
|---|---|---|---|---|---|
| **VBI** (blue) | High-cadence imager | 390–550 nm | $985$–$10,500$ | $45'' \times 45''$ | No |
| **VBI** (red) | High-cadence imager | 600–860 nm | $1,200$–$14,000$ | $69'' \times 69''$ | No |
| **ViSP** | Scanning slit spectro-polarimeter | 380–900 nm (3 spectral windows) | $>180,000$ | 5 slits | **Yes** |
| **VTF** | Fabry-Pérot imaging spectro-polarimeter | 520–870 nm | FWHM 6–8 pm | $60''$ (round) | **Yes** |
| **DL-NIRSP** | Integral Field Unit | 500–1800 nm (3 arms) | $125,000$ | $2.4' \times 1.8'$ max | **Yes** |
| **CRYO-NIRSP** | Scanning slit + context imager | 1000–5000 nm | $100,000$ | $4' \times 3''$ (on-disk) to $5'$ (limb) | **Yes** |

- **VBI**: 4k×4k CMOS, 30 Hz frame rate, speckle reconstruction으로 3-second cadence
- **ViSP**: HAO 제작, photosphere/chromosphere vector magnetic field
- **VTF**: KIS (독일) 제작, dual-etalon Fabry-Pérot, $5 \times 10^{-3}$ polarimetric sensitivity in 13 s, 0.028″/pixel (20 km on Sun)
- **DL-NIRSP**: Univ. Hawaiʻi IfA 제작, integral-field, 코로나 자기장 측정용 coronagraphic mode
- **CRYO-NIRSP**: cryogenic spectrograph, 1–5 μm, coronal Fe XIII (1075 nm), Si IX (3935 nm), CO bands (2333, 4666 nm)
- **Common Instrument Systems**: polarimetry calibration, cameras, software, motion control, **TRADS (Time-based Recording And Distribution System)** — Ferayorni et al. 2014

**Detectors**:
- Infrared: Teledyne 2k×2k Hawaii2-RG (DL-NIRSP, CRYO-NIRSP)
- Visible: 4k×4k ANDOR sensors (VBI, VTF, DL-NIRSP); 2k×2k ANDOR ZYLA (ViSP)

**Data Handling System (DHS)**: summit에서 **gigabyte/s per camera line** 전송, Boulder Data Center로 전달

**English**
**Software** is layered: OCS (top user interface) → TCS (telescope) + ICS (instruments), with CSF providing services. **Instruments** (Table 2): VBI (high-cadence imaging, 4k×4k CMOS, 30 Hz), ViSP (slit spectro-polarimeter, 380–900 nm, $R > 180,000$), VTF (Fabry-Pérot imaging spectro-polarimeter, 520–870 nm, 6–8 pm FWHM, $5\times 10^{-3}$ polarimetric sensitivity in 13 s), DL-NIRSP (integral-field, 500–1800 nm, coronagraphic mode for coronal magnetometry), and CRYO-NIRSP (cryogenic slit spectro-polarimeter, 1–5 μm, Fe XIII 1075 nm and Si IX 3935 nm coronal lines). Detectors are Teledyne Hawaii2-RG (IR) and ANDOR 4k×4k / ZYLA 2k×2k (visible). A summit DHS transfers gigabyte/s per line to the Boulder Data Center.

### Part VIII: Science Operations & Data Center (§12–§13) / 과학 운영 및 데이터 센터

**한국어**
- **Service mode 중심**: 전통적 PI mode (독점 access)가 아닌 **service mode**가 기본. TAC(Time Allocation Committee)가 proposal 평가 → resident scientist가 일일 단위로 조건에 맞는 실험 선택 → 관측소 staff가 실행
- **3단계 계획**:
  - **Long-term (6 months)**: solar cycle에 따라 coupled
  - **Medium-term (1–3 months)**: co-observing, FIDO·DHS 재설정
  - **Short-term (days-weeks)**: 태양 상태·날씨 기반
- **Coordinated observations**: Parker Solar Probe, Solar Orbiter, 지상 시설과 정기적 협업 예상

- **Data Center**: Boulder, NSO HQ, Davey et al. 2021 상세
  - 연간 **3 PB raw data**, calibrated도 유사 규모
  - (i) quality-controlled calibrated 데이터 제공, (ii) long-term curation, (iii) open/searchable access, (iv) 44년 수명 지원 소프트웨어 유지
  - 자동 calibration pipeline이 가장 큰 기술적 도전 (기기 + seeing 영향 제거)

**English**
DKIST defaults to **service mode**, a paradigm shift from traditional PI mode. Proposals are ranked by a TAC; resident scientists execute matching experiments daily. Planning operates on three timescales: long-term (6 months, coupled to solar cycle), medium-term (1–3 months, co-observing and FIDO/DHS configurations), short-term (days–weeks, weather and solar conditions). Coordinated observations with Parker Solar Probe and Solar Orbiter are routine. The **Data Center** in Boulder delivers ~3 PB/year of calibrated data with automatic pipelines, a mission-critical departure from previous NSO facilities where the PI had sole access.

### Part IX: First Light and First Results (§14) / 첫 빛 및 첫 결과

**한국어**
**2019년 12월 first solar light** 달성. 초기 관측 조건:
- VBI red (789 nm, 705 nm filters)로 **granulation** 근 Sun-center 촬영
- AO가 granulation에 locked (태양 구조 기반 correlating WFS의 핵심)
- Fried parameter $r_0 = 6$–$7$ cm (열 시스템 미작동, local seeing 영향)
- 80–100 images bursts, 3 s cadence → **speckle reconstruction**

**핵심 결과**:

**Figure 15** — **First-light granulation image (VBI red, 789 nm)**: 55″×55″ FOV, 80 short-exposure images speckle-reconstructed, **0.04″ 분해능 (회절 한계 근접)**, granulation의 **bright points/sheets** 구조가 선명히 보임

**Figure 16** — **MHD simulation (MURaM) vs. DKIST 비교** (Vögler et al. 2005, Rempel 2014의 8 km grid 재실행): **처음으로 observation과 simulation이 동일 스케일에서 비교 가능** — DKIST 20 km 분해능이 MHD sim의 8 km grid와 직접 맵핑 가능

**Figure 17** — **H-α wing (red wing, 0.05 nm 패스밴드)**과 **continuum (789 nm) 동시 관측**: H-α wing에 **field-aligned chromospheric fibril** 구조 명확, bright points와의 correlation 확인, **초 단위 진화**

**Figure 18** — **DKIST first sunspot image** (commissioning phase, WFC context viewer @ 530 nm): 25″×25″ FOV, center-to-limb 거리 0.45, penumbra/umbra dark cores 분해 (Langhans et al. 2007), **0.027″ filament width 분해** (530 nm 회절 한계), umbral dot & penumbral grain 내 **< 0.1″ dark lane** 관측 (Schüssler & Vögler 2006 magnetoconvection 시뮬레이션과 일치), 100초 단위로 overturning convection 진화 촬영

**English**
**First solar light was achieved in December 2019**. VBI-red (789 nm and 705 nm) observed **granulation** near Sun-center with AO locked on granulation. $r_0 = 6$–$7$ cm (local seeing was still present since thermal systems were not fully operational); 80-image bursts with 3-s cadence were speckle-reconstructed.

**Key Figures**:
- **Fig. 15** — First-light granulation at 789 nm, 55″×55″, 0.04″ resolution, bright points/sheets clearly resolved.
- **Fig. 16** — Side-by-side with Rempel's MURaM simulation at 8 km grid: for the first time, DKIST data and MHD simulations are comparable at the same spatial scale.
- **Fig. 17** — Simultaneous continuum (789 nm) and H-α wing: chromospheric fibrils aligned with magnetic field show bright-point correlation and second-scale evolution.
- **Fig. 18** — First DKIST sunspot image (530 nm, WFC context viewer): resolves penumbral/umbral dark cores, 0.027″ filament width (≈ diffraction limit at 530 nm), and < 0.1″ dark lanes in umbral dots — consistent with Schüssler & Vögler (2006) magnetoconvection simulations.

### Part X: Summary & Outlook (§15) / 요약 및 전망

**한국어**
- 20년 이상 걸린 DKIST 프로젝트가 **SWG가 정의한 science requirement를 충족**함을 first results가 증명
- Operations Commissioning Phase (OCP) proposal call 2020년 5월 발표
- 향후 higher-level data products (spectro-polarimetric inversions) 추가 예정
- **44년 수명** 동안 2 Hale cycle 관측 — 많은 새로운 과학 프로젝트가 데이터 기반에서 파생될 것

**English**
First results demonstrate that DKIST, after two decades of development, achieves the science requirements defined by the SWG. The Operations Commissioning Phase proposal call was released in May 2020. Higher-level data products (e.g., inversions) will follow as resources allow. Over its 44-year lifetime (two Hale cycles), DKIST will anchor many new science projects driven by its data archive.

---

## 3. Key Takeaways / 핵심 시사점

1. **Off-axis 4 m 설계가 scattered light와 photon-starved corona 문제를 동시에 해결한다 / Off-axis 4 m design simultaneously solves scattered-light and photon-starved corona problems** — 중앙 차폐·거미 회절 제거로 산란광 감소 + 기존 46 cm Solar-C 대비 75배 집광 면적으로 $10^{-9}$ 상대 신호의 코로나 자기장 측정이 가능. 두 요구사항이 **같은 설계 선택으로 해결**되었음이 DKIST의 핵심 우아함. Off-axis 제거하고 central obstruction으로 갔다면 둘 다 불가능했을 것.

2. **4 m는 "critical scale" 요구에서 수학적으로 도출된 최소값이다 / 4 m is the minimum aperture derived from "critical scale" requirements** — Stenflo (2008)의 광자 평균자유행로·압력 스케일 높이 논증 + Zeeman 감도 $\propto \lambda^2$ 고려 → 1.6 μm IR에서 0.1″ 분해능 = **4 m diffraction limit**이 최소 구경. 더 큰 구경(SWG는 24 m까지 검토)은 far-IR(12 μm 온도 최소층 magnetometry)에 필요하지만 현재 기술로는 solar interferometer가 필요. 4 m는 "achievable maximum" 선택.

3. **열 관리가 전체 시스템 설계의 1차 제약 / Thermal management is the primary design constraint** — 13 kW 태양 부하, 75 mm M1 두께, $20 \pm 0.5$ °C coudé lab, 555 air jets, 12 km 냉각 파이프, 야간 얼음 생산 — **열이 곧 seeing**이라는 원칙이 설계 모든 층에 침투. "Glass window가 IR 흡수로 불가능해 air knife 사용"처럼 **가장 단순한 해결책이 IR 요건과 충돌하는 패턴**이 반복된다.

4. **Polarimetric accuracy $5\times 10^{-4}$는 모델링 중심의 달성 / $5\times 10^{-4}$ polarimetric accuracy is achieved through modeling, not just hardware** — full system Mueller matrix를 **모든 광학 표면**에 대해 측정·모델링 (Harrington et al. 2017, 2018a/b, 2019, 2020a/b). 3000+ fit variables의 fiber-fed calibration spectrograph는 **measurement이 아니라 characterization**의 도구. 이 패러다임은 "calibrate each element in isolation" → "model the whole system"으로의 전환이다.

5. **Service mode로의 운영 패러다임 전환 / Operational paradigm shift to service mode** — 지상 태양 관측소 역사상 **가장 큰 운영 혁신**. 전통 PI-mode (배정된 주간에 PI가 배타적 access)는 seeing이 나쁜 날 낭비가 되고 flexible coordination이 불가능. Service mode는 TAC ranking + daily condition matching + staff execution — **night-time 대형 시설(Gemini, VLT)의 모델을 solar에 이식**.

6. **Data Center 모델이 "PI가 calibration 책임진다"를 깨뜨린다 / Data Center model breaks the "PI owns calibration" tradition** — 3 PB/year 규모에서 PI 개별 calibration은 불가능. DKIST는 **calibrated 데이터를 직접 배포**하며, 이는 solar community의 데이터 접근 방식을 근본적으로 재편. Higher-level products (inversions)는 점진적으로 추가. 44년 수명 동안 소프트웨어·아카이브 유지 필요.

7. **MHD simulation과 observation의 첫 "same-scale" 비교 / First "same-scale" comparison of MHD simulations and observations** — Fig. 16의 MURaM 8 km grid vs. DKIST 20 km resolution 비교는 **지난 30년 이론-관측 갭의 상징적 종결**. 이후 simulation-observation iterative verification이 본격 시작. Cheung et al. 2019 flare simulation, Bjørgen et al. 2019 chromosphere sim 등이 직접 검증 대상.

8. **향후 4 m 태양 망원경 네트워크의 선도 시설 / Vanguard of the future 4 m solar telescope network** — EST(La Palma, 4 m), Russian Large Solar Telescope(Sayan, 3 m), NLST(India, 2.5 m), CGST(China, 8 m 계획). DKIST와 EST는 **11시간 시간대 차이로 연속 관측 가능** → active region evolution tracking. 단일 망원경이 아닌 **network of sites**가 차세대 패러다임.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 회절 한계 분해능 / Diffraction-limited angular resolution

Airy disk의 첫 minimum까지의 각도:
$$
\theta_{\text{diff}} = 1.22 \, \frac{\lambda}{D}
$$

- $\lambda$: 관측 파장 / observing wavelength
- $D$: 구경 / aperture diameter

DKIST ($D = 4$ m):
$$
\theta_{\text{diff}}(500\,\text{nm}) = 1.22 \cdot \frac{500 \times 10^{-9}}{4} \approx 0.026''
$$

지구–태양 거리 1 AU에서 linear scale:
$$
\ell = D_\odot \cdot \theta_{\text{diff}} = (1.496 \times 10^{11}\,\text{m}) \cdot (0.026 \cdot 4.85 \times 10^{-6}\,\text{rad}) \approx 18.9 \,\text{km} \approx 20 \,\text{km}
$$

### 4.2 Zeeman splitting / 제이만 분열

정상 Zeeman 효과 (normal Zeeman effect)에서 $\sigma$ 성분의 파장 shift:
$$
\Delta\lambda_B = \frac{e}{4\pi m_e c} \, g_{\text{eff}} \, \lambda^2 \, B
$$

Landé factor $g_{\text{eff}}$, 전자 전하 $e$, 전자 질량 $m_e$, 광속 $c$.

**한국어**: $\lambda^2$ 의존성이 IR의 magnetic sensitivity 우위의 수학적 근거. 예:
- Fe I 630.25 nm ($g = 2.5$): $\Delta\lambda_B \approx 2.8$ mÅ at $B = 100$ G
- Fe I 1564.85 nm ($g = 3$): $\Delta\lambda_B \approx 21$ mÅ at $B = 100$ G → **~8배 더 큰 분열**

**English**: The $\lambda^2$ scaling is the mathematical basis for IR's magnetic-sensitivity advantage. At $B = 100$ G, Fe I 1564.85 nm exhibits ~8× larger splitting than Fe I 630.25 nm.

### 4.3 Strehl ratio와 seeing / Strehl ratio and seeing

AO가 없는 경우, long-exposure Strehl은
$$
S_{\text{no-AO}} \approx \left( \frac{r_0}{D} \right)^2 \quad (D \gg r_0)
$$

AO 보정 후 잔여 파면 오차 $\sigma_\phi^2$ (rad²)로:
$$
S_{\text{AO}} \approx \exp(-\sigma_\phi^2)
$$

총 파면 오차 = fitting + bandwidth + sensor noise + anisoplanatism:
$$
\sigma_\phi^2 = \sigma_{\text{fit}}^2 + \sigma_{\text{BW}}^2 + \sigma_{\text{noise}}^2 + \sigma_{\text{aniso}}^2
$$

**Fitting error** (DM 유한 액추에이터 수):
$$
\sigma_{\text{fit}}^2 = \kappa \left( \frac{d}{r_0} \right)^{5/3}
$$
$d$: sub-aperture size, $\kappa \approx 0.28$ for square geometry.

**Bandwidth error** (AO 업데이트 주기 $\tau$):
$$
\sigma_{\text{BW}}^2 = \left( \frac{\tau}{\tau_0} \right)^{5/3}
$$
$\tau_0 = 0.314 r_0 / \bar{v}$, $\bar{v}$: atmospheric wind-weighted mean speed.

**한국어**: DKIST는 $\tau = 0.5$ ms (2 kHz), 1,600 액추에이터로 $d \approx 10$ cm pitch → median seeing ($r_0 \approx 10$ cm)에서 fitting + bandwidth가 주요 기여.

**English**: DKIST operates at $\tau = 0.5$ ms (2 kHz) with $d \approx 10$ cm; under median seeing ($r_0 \approx 10$ cm), fitting and bandwidth dominate.

### 4.4 Strehl의 파장 의존성 / Wavelength dependence of Strehl

동일 seeing ($r_0 \propto \lambda^{6/5}$)에서:
$$
S(\lambda_2) = S(\lambda_1)^{(\lambda_1/\lambda_2)^2}
$$

예: $S(630) = 0.6$일 때 $S(1600) = 0.6^{(630/1600)^2} = 0.6^{0.155} \approx 0.92$

즉 **IR에서 훨씬 높은 Strehl** 달성.

### 4.5 편광 측정 불확도 / Polarimetric uncertainty

광자 통계 한계 (sensitivity):
$$
\sigma_{\text{photon}} = \frac{1}{\sqrt{N_{\text{photon}}}}
$$

$5\times 10^{-4}$ 달성을 위해 $N \gtrsim 4 \times 10^6$ 광자/resolution element/modulation state 필요.

Full Stokes 측정:
$$
\vec{S}_{\text{measured}} = \mathbf{O} \cdot \mathbf{M}_{\text{sys}} \cdot \vec{S}_{\text{sun}}
$$

- $\vec{S} = (I, Q, U, V)^T$: Stokes vector
- $\mathbf{M}_{\text{sys}}$: 4×4 system Mueller matrix
- $\mathbf{O}$: demodulation operator

**Accuracy requirement**:
$$
\|\mathbf{M}_{\text{sys, true}} - \mathbf{M}_{\text{sys, model}}\|_{\text{sup}} \lesssim 5 \times 10^{-4}
$$

Mueller matrix 각 element의 절대 오차가 specified 수준.

### 4.6 Coronal signal & collecting area / 코로나 신호와 집광 면적

코로나 밝기 / 디스크 밝기:
$$
\frac{I_{\text{corona}}}{I_{\text{disk}}} \sim 10^{-6} \quad (\text{near active region})
$$

Polarimetric signal / coronal intensity:
$$
\frac{\delta I_{\text{pol}}}{I_{\text{corona}}} \sim 10^{-3} \text{ to } 10^{-4}
$$

Required sensitivity per unit observation time scales with **collecting area**:
$$
t_{\text{int}} \propto \frac{1}{A \cdot \eta} \cdot \left( \frac{S/N_{\text{target}}}{I_{\text{signal}}} \right)^2
$$

DKIST vs. Solar-C:
$$
\frac{A_{\text{DKIST}}}{A_{\text{Solar-C}}} = \left( \frac{4.0}{0.46} \right)^2 = 75.6
$$

→ 동일 S/N 달성에 **75배 적은 integration time** (또는 75배 높은 S/N).

### 4.7 Heat load / 열 부하

주경이 받는 전체 태양 플럭스:
$$
P_{\odot, \text{M1}} = F_\odot \cdot A_{\text{M1}} = 1361 \,\text{W/m}^2 \cdot \pi (2\,\text{m})^2 \approx 17.1 \,\text{kW}
$$

대기 투과율 ($\tau \approx 0.75$) 고려:
$$
P_{\text{on-axis}} \approx 13 \,\text{kW} \quad (\text{at M1 entrance})
$$

Heat stop at prime focus (5 arcmin FOV image radius $r$):
- Prime focus image size $\approx 75$ mm diameter for full Sun
- 5 arcmin FOV $\approx 2.18$ mm diameter
- Passed fraction $= (2.18/75)^2 \approx 0.0008$ → **99.92% 차단**

Heat flux at prime focus:
$$
q_{\text{prime}} = \frac{P_{\odot, \text{M1}}}{A_{\text{prime image}}} = \frac{13\,\text{kW}}{\pi (0.0375)^2} \approx 2.9\,\text{MW/m}^2 \;(\approx 2.5\,\text{MW/m}^2 \text{ quoted})
$$

**M1 absorption** (unprotected Al 반사율 $\sim 91\%$):
$$
P_{\text{M1, abs}} \approx 0.09 \cdot 13\,\text{kW} \approx 1.2\,\text{kW} \approx 120\,\text{W/m}^2 \text{ (avg)}
$$

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
태양 망원경 발전사 / History of solar telescope development
────────────────────────────────────────────────────────────────────────────

1960s ─ McMath-Pierce (1.5 m, Kitt Peak) ─ largest solar telescope of its era
1970s ─ DST (Dunn Solar Telescope, 0.76 m, Sacramento Peak) ─ vacuum tower
1980s ─ VTT (0.7 m, Tenerife) ─ German workhorse
1990s ─ SST (1 m, La Palma) ─ Swedish Solar Telescope
       ═══════════════════════════════════════════════════════
       CLEAR concept (NSO) ─ DKIST의 시작점
       ═══════════════════════════════════════════════════════
2000  ─ NRC Decadal Review: AST(ATST) 높은 순위
2002  ─ SST 1 m 업그레이드 + AO 시스템
2003  ─ ATST 설계 단계 시작
2004  ─ Haleakalā 사이트 선정 (72 후보 중)
2006  ─ 해상도 > 0.5″ 지상 solar 영상 한계
2008  ─ ATST 최종 설계 검토 통과
2010  ─ NSF 건설 승인 / GREGOR (1.5 m, Tenerife) 첫 빛
2012  ─ Haleakalā 부지 공사 시작
2013  ─ GST (1.6 m, BBSO) 운영 / ATST → DKIST로 개명
2014  ─ DKIST M1 최초 Al 코팅 (AMOS 벨기에)
       ▼
═════════════════════════════════════════════════════════════════════════════
  2020: DKIST overview paper (이 논문) / First light Dec 2019
═════════════════════════════════════════════════════════════════════════════
       ▲
2021  ─ DKIST Topical Collection 기기별 논문 출판
2022  ─ DKIST OCP (Operations Commissioning Phase) 공식 시작
2022  ─ ViSP (de Wijn et al.), DL-NIRSP (Jaeggli et al.) 기기 논문
2024  ─ DKIST 본격 과학 운영
────  ─ 미래 / Future:
2029  ─ EST (European Solar Telescope, 4 m) 완공 예정
2030+ ─ DKIST MCAO 업그레이드 / CGST (8 m, China) 계획
2064  ─ DKIST 설계 수명 만료 (44년 = 2 Hale cycles)

현대 MHD 시뮬레이션 발전 / Modern MHD simulation milestones
────────────────────────────────────────────────────────────
2005 ─ MURaM (Vögler et al.) ─ photospheric magneto-convection
2006 ─ Stein & Nordlund ─ realistic granulation
2014 ─ Rempel ─ 8 km grid, sunspot fine structure
2019 ─ Cheung et al. ─ 3D flare simulation
       ═══ 2020: DKIST first light 으로 same-scale 비교 가능 ═══
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **#4** (Engvold 1991, LEST 논의) | Early off-axis 개념이 DKIST의 선구자 — LEST 작업이 ATST 개념 설계에 정보 제공 | DKIST의 off-axis 설계 철학의 역사적 뿌리 / Historical root of DKIST's off-axis philosophy |
| **#20** (Rimmele & Marino 2011, Solar AO review) | DKIST AO 시스템 설계의 기술적 기반 — 1,600 actuator + 1,457 sub-aperture Shack-Hartmann은 이 리뷰의 principle 적용 | DKIST AO 구현의 직접적 이론·알고리즘 기반 / Direct theoretical/algorithmic basis of DKIST AO |
| **#21** (Wöger et al. 2008, Speckle interferometry) | VBI의 first-light 이미지는 speckle reconstruction을 적용 — 80 short-exposure images + AO telemetry → 0.04″ 분해능 복원 | DKIST의 post-processing pipeline 핵심 / Key post-processing pipeline for DKIST imaging |
| **#22** (Scharmer et al. 2008, CRISP on SST) | Fabry-Pérot dual-etalon 기술이 DKIST **VTF**로 계승 — KIS가 CRISP 경험을 DKIST에 적용 | VTF 설계의 직접적 전신 / Direct predecessor of VTF design |
| **#24** (de Wijn et al. 2022, ViSP) | DKIST 기기 시리즈의 첫 번째 — photosphere/chromosphere vector B-field | DKIST의 과학 역량을 실현하는 기기 세부 / Instrument-level detail realizing DKIST's capability |
| **#25** (Jaeggli et al. 2022, DL-NIRSP) | 근적외선 integral-field 기기, 코로나 자기장 측정의 핵심 | DKIST의 "coronal magnetometry" 과학 목표 직접 구현 / Directly implements coronal magnetometry goal |
| **Future (EST Collados et al. 2013)** | 유럽의 4 m 대응 망원경 — DKIST + EST로 **11시간 coverage** 확보 | 지상 4 m 네트워크의 파트너 / Partner in 4 m ground network |

---

## 7. References / 참고문헌

### Primary paper / 본 논문
- Rimmele, T. R., Warner, M., Keil, S. L., Goode, P. R., Knölker, M., Kuhn, J. R., et al., "The Daniel K. Inouye Solar Telescope – Observatory Overview", *Solar Physics*, Vol. 295, Article 172 (2020). [DOI: 10.1007/s11207-020-01736-7]

### DKIST Topical Collection companion papers
- Wöger, F., et al., "Visible Broadband Imager (VBI)", *Solar Physics*, in press (2021).
- de Wijn, A. G., et al., "Visible Spectro-Polarimeter (ViSP)", *Solar Physics*, 297, 22 (2022). [DOI: 10.1007/s11207-022-01954-1]
- Jaeggli, S. A., et al., "Diffraction-Limited Near-Infrared Spectro-Polarimeter (DL-NIRSP)", *Solar Physics*, 297, 137 (2022). [DOI: 10.1007/s11207-022-02062-w]
- Fehlmann, A., et al., "Cryogenic Near-Infrared Spectro-Polarimeter (CRYO-NIRSP)", *Solar Physics*, in preparation.
- von der Lühe, O., et al., "Visible Tunable Filter (VTF)", *Solar Physics*, in preparation.
- Harrington, D. M., et al., "Polarimetry with DKIST", *Solar Physics*, in preparation.
- Rast, M. P., Cauzzi, G., Martínez Pillet, V., "Critical Science Plan for DKIST", *Solar Physics*, 2021.
- Tritschler, A., et al., "DKIST Operations", *Solar Physics*, 2021.
- Davey, A., et al., "DKIST Data Center", *Solar Physics*, 296 (2021).

### Supporting design/technical references
- Rimmele, T. R. & Marino, J. M., "Solar Adaptive Optics", *Living Reviews in Solar Physics*, 8, 2 (2011). [DOI: 10.12942/lrsp-2011-2]
- Harrington, D. M. & Sueoka, S. R., "Polarization modeling and predictions for Daniel K. Inouye Solar Telescope part 1", *J. Astron. Telesc. Instrum. Syst.*, 3, 018002 (2017).
- Harrington, D. M., Sueoka, S. R., & White, A. J., "Polarization modeling and predictions for Daniel K. Inouye Solar Telescope part 5", *J. Astron. Telesc. Instrum. Syst.*, 5, 038001 (2019).
- Johnson, L. C., et al., "The active and adaptive optics systems at DKIST", *Proc. SPIE*, 11450-108 (2020).
- Richards, K., Rimmele, T. R., et al., "Active and adaptive optics for ATST", *Proc. SPIE* (2010).

### MHD simulation & theoretical references
- Stenflo, J. O., "Hanle-Zeeman synergies for magnetic field measurements", *Physica Scripta*, 2008.
- Rempel, M., "Numerical simulations of quiet Sun magnetism", *ApJ*, 789, 132 (2014).
- Vögler, A., Shelyag, S., Schüssler, M., et al., "MURaM code", *A&A*, 429, 335 (2005).
- Schüssler, M. & Vögler, A., "Magnetoconvection in sunspots", *ApJL*, 641, L73 (2006).
- Cheung, M. C. M., et al., "A comprehensive 3D radiative MHD simulation of a solar flare", *Nature Astronomy*, 3, 160 (2019).

### Site survey & enclosure
- Hill, F., et al., "Site testing for ATST", *Proc. SPIE* (2004, 2006).
- Murga, G., et al., "DKIST enclosure design", *Proc. SPIE* (2012, 2014).
- Phelps, L. & Warner, M., "DKIST lower enclosure design", *Proc. SPIE* (2008).
