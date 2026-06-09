---
title: "The Atmospheric Imaging Assembly (AIA) on the Solar Dynamics Observatory (SDO)"
authors: "J.R. Lemen, A.M. Title, D.J. Akin, P.F. Boerner, C. Chou, J.F. Drake, D.W. Duncan, C.G. Edwards, F.M. Friedlaender, G.F. Heyman, N.E. Hurlburt, N.L. Katz, G.D. Kushner, M. Levay, R.W. Lindgren, D.P. Mathur, E.L. McFeaters, S. Mitchell, R.A. Rehse, C.J. Schrijver, L.A. Springer, R.A. Stern, T.D. Tarbell, J.-P. Wuelser, C.J. Wolfson, C. Yanari, J.A. Bookbinder, P.N. Cheimets, D. Caldwell, E.E. DeLuca, R. Gates, L. Golub, S. Park, W.A. Podgorski, R.I. Bush, P.H. Scherrer, M.A. Gummin, P. Smith, G. Auker, P. Jerram, P. Pool, R. Soufli, D.L. Windt, S. Beardsley, M. Clapp, J. Lang, N. Waltham"
year: 2012
journal: "Solar Physics, Vol. 275, pp. 17–40"
doi: "10.1007/s11207-011-9776-8"
topic: Solar Observation
tags: [AIA, SDO, EUV, UV, full disk, high cadence, multilayer, Cassegrain, temperature response, DEM, JSOC, back-illuminated CCD, guide telescope, image stabilization, Lockheed Martin, SAO]
status: completed
date_started: 2026-04-16
date_completed: 2026-04-16
---

# 12. The Atmospheric Imaging Assembly (AIA) on the Solar Dynamics Observatory (SDO) / 태양 동역학 관측위성(SDO)의 대기 영상 장치(AIA)

---

## 1. Core Contribution / 핵심 기여

AIA(Atmospheric Imaging Assembly)는 NASA의 **SDO(Solar Dynamics Observatory)** 미션에 탑재된 태양 대기 영상 기기로, 2010년 2월 11일 Atlas V-401 로켓에 의해 발사되어 지구정지 경사궤도(GEO-synchronous inclined orbit)에 투입되었다. AIA는 **4대의 독립적인 20 cm 구경 f/20 Cassegrain 망원경**으로 구성되며, 총 **10개 파장 채널** — EUV 7개(94, 131, 171, 193, 211, 304, 335 Å), UV 2개(1600, 1700 Å), 가시광 1개(4500 Å) — 에서 태양 전면(full-disk)을 동시에 관측한다. 4096×4096 후면 박화(back-thinned) CCD(12 μm 픽셀)로 **0.6"/pixel** 공간 분해능과 **41'×41' FOV**를 달성하며, 8개 EUV/UV 채널의 **12초 케이던스**(cadence)는 선행 기기 SOHO/EIT 대비 약 60배 빠른 것이다. 전용 White Sands 지상국을 통해 약 67 Mbit/s의 연속 다운링크를 수행하며, 하루 약 1.5 TB(Rice 무손실 압축 후)의 데이터를 전송한다. AIA는 역사상 가장 많이 사용된 태양 관측 기기로, 2010년 이후 태양 물리학 연구의 표준 데이터 소스가 되었다.

AIA (Atmospheric Imaging Assembly) is the primary imaging instrument on NASA's **Solar Dynamics Observatory (SDO)**, launched on 11 February 2010 aboard an Atlas V-401 rocket into a geosynchronous inclined orbit. AIA consists of **four independent 20 cm aperture f/20 Cassegrain telescopes** observing the full solar disk simultaneously across **10 wavelength channels** — 7 EUV (94, 131, 171, 193, 211, 304, 335 Å), 2 UV (1600, 1700 Å), and 1 visible (4500 Å). Using 4096×4096 back-thinned CCDs (12 μm pixels), AIA achieves **0.6"/pixel** spatial resolution with a **41'×41' FOV**, and an unprecedented **12-second cadence** for all 8 EUV/UV channels — approximately 60× faster than SOHO/EIT. A dedicated ground station at White Sands enables continuous downlink at ~67 Mbit/s, delivering approximately 1.5 TB/day (after Rice lossless compression). AIA has become the most-used solar instrument in history and the standard data source for solar physics research since 2010.

---

## 2. Reading Notes / 읽기 노트

### §1 Introduction / 서론 (pp. 17–19)

AIA는 NASA의 **Living With a Star (LWS)** 프로그램의 핵심 미션인 SDO에 탑재된 세 기기 중 하나이다(나머지는 HMI와 EVE). SDO의 목표는 태양의 자기 환경(magnetic environment)의 진화를 이해하는 것이며, AIA는 코로나와 천이 영역의 **다온도(multi-thermal) 구조**를 연속적으로 영상화하는 역할을 담당한다.

AIA is one of three instruments aboard SDO, NASA's flagship mission in the **Living With a Star (LWS)** program (the others being HMI and EVE). SDO's goal is to understand the evolution of the Sun's magnetic environment, and AIA's role is to continuously image the **multi-thermal structure** of the corona and transition region.

AIA는 이전 세대 기기들의 기술적 유산을 계승·통합한다: NIXT(1990), EIT(1995), TRACE(1998), STEREO/EUVI(2006). 그러나 AIA는 이들 모두를 대폭 능가하는데, **EIT 대비 정보율(information rate)이 400배에서 최대 22,000배** 향상되었다(Table 1과의 비교). 이는 16배 더 많은 픽셀(4096² vs 1024²), 2배 더 많은 채널(8 vs 4 EUV), 그리고 60배 빠른 케이던스(12초 vs 12분)의 조합이다.

AIA inherits and integrates the technical heritage of previous-generation instruments: NIXT (1990), EIT (1995), TRACE (1998), and STEREO/EUVI (2006). However, AIA vastly surpasses all of them — the **information rate improvement over EIT ranges from 400× to 22,000×**. This results from the combination of 16× more pixels (4096² vs 1024²), 2× more channels (8 vs 4 EUV), and 60× faster cadence (12 s vs 12 min).

4대의 망원경은 SDO 우주선의 기기 모듈(instrument module)에 장착되어 있다(Fig. 1, 2). 각 망원경은 독립적인 가이드 망원경(GT)과 영상 안정화 시스템(ISS)을 갖추고 있어, 상호 독립적으로 작동한다.

The four telescopes are mounted on the SDO spacecraft instrument module (Fig. 1, 2). Each telescope has its own independent guide telescope (GT) and image stabilization system (ISS), operating autonomously.

### §2 Science Overview / 과학 개요 (pp. 19–22)

AIA의 과학 목표는 다섯 가지 핵심 주제로 구성된다:

AIA's science objectives are organized into five core themes:

1. **에너지 입력·저장·방출 / Energy input, storage, and release**: 자기 재결합(reconnection), 전류 시트(current sheets), 플레어 에너지론(flare energetics). 고케이던스 다파장 영상으로 재결합 과정의 시간 진화를 추적.
2. **코로나 가열 및 복사도 / Coronal heating and irradiance**: DEM(Differential Emission Measure) 분석을 통한 열적 구조 규명. 10개 온도 채널의 동시 관측이 DEM 역산(inversion)을 가능케 함.
3. **과도 현상 / Transients**: 플레어, CME(Coronal Mass Ejection), EIT 파(EIT wave), 제트(jet). 12초 케이던스로 초기 진화 단계를 포착.
4. **지구 근방 우주 환경 연결 / Connections to geospace**: CME 발생부터 지구 영향까지의 연결고리. SDO/AIA + STEREO의 다시점 조합이 핵심.
5. **코로나 지진학 / Coronal seismology**: 코로나 루프의 진동(oscillation)과 파동(wave) 관측을 통한 플라즈마 물성 진단. 12초 케이던스가 MHD 파동의 주기(수분~수십분)를 충분히 분해.

1. **Energy input, storage, and release**: Magnetic reconnection, current sheets, flare energetics. High-cadence multi-wavelength imaging tracks the temporal evolution of reconnection processes.
2. **Coronal heating and irradiance**: Characterizing thermal structure through DEM (Differential Emission Measure) analysis. Simultaneous observation in 10 temperature channels enables DEM inversion.
3. **Transients**: Flares, CMEs (Coronal Mass Ejections), EIT waves, jets. The 12-second cadence captures early evolutionary phases.
4. **Connections to geospace**: Linking CME origins to terrestrial impacts. Multi-viewpoint combination of SDO/AIA + STEREO is key.
5. **Coronal seismology**: Diagnosing plasma properties through coronal loop oscillations and wave observations. The 12-second cadence adequately resolves MHD wave periods (minutes to tens of minutes).

### §3 Instrument Overview / 기기 개요 (pp. 22–35)

AIA의 핵심 사양은 다음과 같다:

AIA's key specifications are as follows:

| 항목 / Parameter | 값 / Value |
|---|---|
| 주경 직경 / Primary diameter | 20 cm |
| 유효 초점 거리 / Effective focal length | 4.125 m (f/20) |
| FOV | 41' × 41' (46' diagonal) |
| 공간 분해능 / Spatial resolution | 0.6"/pixel (12 μm) |
| CCD | 4096 × 4096, back-thinned (e2v CCD203-82) |
| Full well capacity | 150,000 electrons |
| 읽기 잡음 / Read noise | < 25 electrons |
| 지향 안정도 / Pointing stability | 0.12" (1σ, with ISS) |
| 케이던스 / Cadence | 10–12 s (8 EUV/UV channels) |
| 노출 시간 / Exposure time | 0.5–3 s |
| 비행 컴퓨터 / Flight computer | BAe RAD6000 |
| 망원경 질량 / Telescope mass | 4 × 28 kg = 112 kg |
| AEB 질량 / AEB mass | 26 kg |
| 소비 전력 / Power | 160 W |
| 다운링크 / Downlink | ~67 Mbit/s to spacecraft |
| 데이터량 / Data volume | ~1.5 TB/day (compressed), ~2 TB/day (uncompressed) |

#### 채널 구성 / Channel Configuration (Table 1)

10개 관측 채널의 핵심 이온과 온도 감응 범위:

Core ions and temperature sensitivity ranges for the 10 observation channels:

| 채널 / Channel (Å) | 주요 이온 / Primary Ion | log T (K) | 망원경 / Telescope |
|---|---|---|---|
| 94 | Fe XVIII | 6.8 | T4 |
| 131 | Fe VIII / Fe XXI | 5.6 / 7.0 | T1 |
| 171 | Fe IX | 5.8 | T3 |
| 193 | Fe XII / Fe XXIV | 6.2 / 7.3 | T2 |
| 211 | Fe XIV | 6.3 | T2 |
| 304 | He II | 4.7 | T4 |
| 335 | Fe XVI | 6.4 | T1 |
| 1600 | C IV + continuum | 5.0 | T3 |
| 1700 | Continuum | 3.7 | T3 |
| 4500 | Continuum | 3.7 | T3 |

4개의 망원경에 10개 채널이 배분되는 방식이 AIA 설계의 핵심이다. 각 망원경의 주경·부경에는 **반씩(half-mirror) 서로 다른 다층막 코팅**이 적용되어 있어, 필터 휠 전환만으로 두 파장 대역을 교대 관측한다. 이를 통해 4대의 망원경으로 최대 8개 EUV/UV 채널을 동시에(10–12초 이내) 관측할 수 있다.

The distribution of 10 channels across 4 telescopes is central to AIA's design. Each telescope's primary and secondary mirrors carry **different multilayer coatings on each half**, so switching between two wavelength bands requires only a filter wheel change. This allows 4 telescopes to observe up to 8 EUV/UV channels simultaneously (within 10–12 seconds).

#### §3.1 Mirrors / 거울 (pp. 24–26)

주경과 부경의 기판은 **Zerodur**(Schott 제조, 극저열팽창 유리)로 제작되었으며, 표면 거칠기(micro-roughness)는 **5 Å 미만**으로 초정밀 연마되었다. 이는 TRACE의 ULE 유리 기판(7 Å RMS)보다 더 매끄러운 수준이다.

The primary and secondary mirror substrates are made of **Zerodur** (manufactured by Schott, ultra-low thermal expansion glass) with surface micro-roughness of **< 5 Å** — even smoother than TRACE's ULE glass substrates (7 Å RMS).

각 망원경의 다층막 코팅 구성:

Multilayer coating configuration for each telescope:

| 망원경 / Telescope | 코팅 A / Coating A | 코팅 B / Coating B | 제작 / Deposited by |
|---|---|---|---|
| T1 | 131 Å — Mo/Si | 335 Å — SiC/Si | LLNL / RXO |
| T2 | 193 Å — Mo/Si | 211 Å — Mo/Si | LLNL |
| T3 | 171 Å — Mo/Si | UV — Al/MgF₂ | LLNL |
| T4 | 94 Å — Mo/Y | 304 Å — SiC/Si | LLNL / RXO |

다층막 코팅은 LLNL(Lawrence Livermore National Laboratory)과 RXO(Reflective X-ray Optics)에서 제작되었다. 94 Å 채널에는 Mo/Y 다층막이 사용되는데, 이는 짧은 파장에서 Mo/Si보다 높은 반사율을 제공하기 때문이다. UV 채널에는 넓은 대역 반사를 위한 Al/MgF₂ 코팅이 적용된다.

The multilayer coatings were fabricated at LLNL (Lawrence Livermore National Laboratory) and RXO (Reflective X-ray Optics). The 94 Å channel uses Mo/Y multilayers because they provide higher reflectivity than Mo/Si at shorter wavelengths. The UV channel uses Al/MgF₂ coatings for broadband reflectivity.

#### §3.2 Filters / 필터 (pp. 26–27)

각 망원경에는 **입구 필터(entrance filter)**와 **초점면 필터(focal plane filter)**가 장착되어 있으며, 모두 **Ni 메쉬(nickel mesh)**로 지지된다. 입구 필터의 주 역할은 가시광과 열을 차단하는 것이다.

Each telescope is equipped with an **entrance filter** and a **focal plane filter**, both supported on **nickel mesh**. The primary function of the entrance filter is to reject visible light and heat.

- **T1 (131/335)**: Zr(zirconium) 필터 사용 — 94, 131, 335 Å 채널에 최적화
- **T2 (193/211)**: Al(aluminum) 필터 사용
- **T3 (171/UV)**: Al 필터 (EUV 쪽) + 별도 UV 윈도우
- **T4 (94/304)**: Zr 필터 사용

- **T1 (131/335)**: Zr (zirconium) filters — optimized for 94, 131, 335 Å channels
- **T2 (193/211)**: Al (aluminum) filters
- **T3 (171/UV)**: Al filters (EUV side) + separate UV window
- **T4 (94/304)**: Zr filters

Table 3에는 각 채널의 필터·코팅·파장에 대한 완전한 사양이 제시되어 있으며, 유효 면적(effective area)과 함께 온도 감응 함수(temperature response function) 계산의 기초가 된다.

Table 3 provides complete specifications for filters, coatings, and wavelengths for each channel, forming the basis for calculating effective areas and temperature response functions.

#### §3.3 CCD / CCD 검출기 (pp. 27–29)

AIA의 검출기는 **e2v CCD203-82**, 4096×4096 픽셀, 12 μm 픽셀 크기의 **후면 박화(back-thinned, back-illuminated)** CCD이다. 이는 TRACE가 사용한 전면 조사(front-illuminated) + lumogen 코팅 방식과 근본적으로 다른 접근이다. 후면 박화 CCD는 EUV에 대해 직접 감응하므로 형광 코팅이 필요 없으며, 양자 효율(QE)이 현저히 높다.

AIA's detector is the **e2v CCD203-82**, a 4096×4096 pixel, 12 μm pixel pitch **back-thinned (back-illuminated)** CCD. This is a fundamentally different approach from TRACE's front-illuminated + lumogen coating method. Back-thinned CCDs are directly sensitive to EUV without the need for phosphor coatings, achieving significantly higher quantum efficiency (QE).

핵심 성능 지표:

Key performance metrics:

| 항목 / Parameter | 값 / Value |
|---|---|
| 풀웰 용량 / Full well | 150,000 e⁻ |
| 읽기 잡음 / Read noise | < 25 e⁻ |
| 읽기 속도 / Readout speed | 2 Mpixels/s per quadrant |
| 읽기 방식 / Readout | 4 quadrants simultaneously |
| 픽셀 크기 / Pixel size | 12 μm |
| 어레이 크기 / Array size | 4096 × 4096 |

CCD는 4개의 사분면(quadrant)을 동시에 읽어내므로, 전체 4096×4096 영상의 읽기 시간은 약 2초이다. 이 CCD 설계는 HMI(Helioseismic and Magnetic Imager)와 공유되지만, HMI는 전면 조사(front-illuminated) 버전을 사용한다는 점에서 차이가 있다.

The CCD reads out all four quadrants simultaneously, resulting in a total readout time of approximately 2 seconds for the full 4096×4096 image. This CCD design is shared with HMI (Helioseismic and Magnetic Imager), though HMI uses a front-illuminated version.

#### §3.4 Guide Telescope and ISS / 가이드 망원경 및 영상 안정화 시스템 (pp. 29–30)

각 망원경에는 전용 **가이드 망원경(GT)**과 **영상 안정화 시스템(ISS)**이 장착되어 있다. 이 설계는 TRACE와 STEREO/EUVI에서 계승한 것이다.

Each telescope is equipped with a dedicated **guide telescope (GT)** and **image stabilization system (ISS)**. This design is inherited from TRACE and STEREO/EUVI.

- **가이드 망원경**: 차폐판(occulter) 뒤에 배치된 **4개의 포토다이오드(photodiode)**가 태양 림(limb) 위치를 측정. 태양 가장자리의 밝기 구배(brightness gradient)를 이용하여 X, Y 방향 오프셋을 산출.
- **ISS 작동기**: 부경(secondary mirror) 뒤의 **PZT(piezoelectric) 작동기**가 부경을 기울여 영상 지터를 보정.
  - 범위: ±46" (충분히 넓어 SDO 우주선의 지향 오차를 커버)
  - 대역폭: 30 Hz
  - 잔여 지터: < 0.24" (1σ), 목표 0.12"

- **Guide telescope**: **Four photodiodes** positioned behind an occulter measure the solar limb position. The brightness gradient at the solar limb is used to calculate X and Y offsets.
- **ISS actuators**: **PZT (piezoelectric) actuators** behind the secondary mirror tilt the secondary to correct image jitter.
  - Range: ±46" (wide enough to cover SDO spacecraft pointing errors)
  - Bandwidth: 30 Hz
  - Residual jitter: < 0.24" (1σ), goal 0.12"

ISS는 AIA의 0.6"/pixel 공간 분해능을 지향 안정도로 뒷받침하는 핵심 서브시스템이다.

The ISS is a critical subsystem that underpins AIA's 0.6"/pixel spatial resolution with commensurate pointing stability.

#### §3.5 Mechanisms / 메커니즘 (pp. 30–31)

4대의 망원경에 총 **17개의 메커니즘**이 분포되어 있다:

A total of **17 mechanisms** are distributed across the four telescopes:

1. **전면 도어(Front door)**: 원샷(one-shot) 파라핀 작동기(paraffin actuator)로 개방. 발사 시 망원경 내부를 보호. 일단 개방되면 재폐쇄 불가.
2. **초점 조절(Focus)**: ±800 μm 범위, 2.2 μm 스텝. 궤도 내 열변형 보정에 사용.
3. **필터 휠(Filter wheel)**: 5개 포지션, 전환 시간 약 1초. 파장 선택(두 다층막 코팅 중 선택)과 어두운 프레임(dark frame) 취득에 사용.
4. **구경 선택기(Aperture selector)**: T2에만 장착. 193 Å과 211 Å 사이 전환 시 빛 경로를 선택.
5. **셔터(Shutter)**: 두 개의 개구부 — 작은 것(5 ms) 대형(80 ms). 노출 시간 0.5–3초 범위.

1. **Front door**: Opened by a one-shot paraffin actuator. Protects the telescope interior during launch. Cannot be re-closed once opened.
2. **Focus**: ±800 μm range, 2.2 μm steps. Used to compensate for on-orbit thermal deformation.
3. **Filter wheel**: 5 positions, ~1 second transition time. Used for wavelength selection (choosing between two multilayer coatings) and dark frame acquisition.
4. **Aperture selector**: Installed only on T2. Selects the light path when switching between 193 Å and 211 Å.
5. **Shutter**: Two apertures — small (5 ms) and large (80 ms). Exposure time range 0.5–3 seconds.

#### §3.6 Electronics / 전자 시스템 (pp. 31–33)

AIA 전자 장치 박스(AEB, AIA Electronics Box)는 기기의 두뇌 역할을 한다:

The AIA Electronics Box (AEB) serves as the instrument's brain:

- **프로세서**: BAe RAD6000 (방사선 경화 PowerPC 기반). **VxWorks** RTOS(실시간 운영체제) 탑재.
- **이중화(Redundancy)**: 완전한 이중화 시스템으로, 주계통 고장 시 예비 시스템으로 전환 가능.
- **통신**: SpaceWire 100 Mbit/s 링크를 통해 SDO 우주선에 데이터 전송.
- **영상 시퀀서(Image sequencer)**: 지상에서 재구성 가능(reconfigurable). 관측 순서, 노출 시간, 압축 설정 등을 업로드 가능.
- **자동 노출 제어(AEC, Automatic Exposure Control)**: TRACE 유산. 태양 활동 수준에 따라 노출 시간을 자동 조절하여 동적 범위를 확보.

- **Processor**: BAe RAD6000 (radiation-hardened PowerPC). Runs **VxWorks** RTOS (real-time operating system).
- **Redundancy**: Fully redundant system, capable of switching to backup upon primary failure.
- **Communication**: Data transmitted to SDO spacecraft via SpaceWire 100 Mbit/s link.
- **Image sequencer**: Ground-reconfigurable. Observation sequence, exposure times, and compression settings can be uploaded.
- **AEC (Automatic Exposure Control)**: TRACE heritage. Automatically adjusts exposure time based on solar activity level to maintain dynamic range.

#### §3.7 Calibration / 보정 (pp. 33–35)

AIA의 **온도 감응 함수(temperature response function)**는 CHIANTI 원자 데이터베이스를 기반으로 계산된다. 이 함수는 각 채널이 특정 온도의 플라즈마에 얼마나 민감한지를 정량적으로 나타내며, **Fig. 13**은 태양 물리학 기기 논문에서 가장 많이 인용되는 그림 중 하나이다.

AIA's **temperature response functions** are calculated based on the CHIANTI atomic database. These functions quantitatively describe how sensitive each channel is to plasma at specific temperatures, and **Fig. 13** is one of the most-cited figures in solar physics instrumentation literature.

상세한 보정 절차와 유효 면적 측정은 Boerner et al. (2012)에서 별도로 다룬다. 지상 보정(ground calibration)은 발사 전 Lockheed Martin의 실험실에서 수행되었으며, 궤도 내 보정(in-orbit calibration)은 EVE(Extreme Ultraviolet Variability Experiment) 로켓 비행과의 교차 보정으로 지속된다.

Detailed calibration procedures and effective area measurements are covered separately in Boerner et al. (2012). Ground calibration was performed at Lockheed Martin's laboratory before launch, and in-orbit calibration continues through cross-calibration with EVE (Extreme Ultraviolet Variability Experiment) sounding rocket flights.

### §4 Operations / 운용 (pp. 35–37)

기본 운용 모드는 **12초 케이던스**로 8개 EUV/UV 채널을 순차 관측하는 것이다. 특수 모드:

The baseline operating mode observes 8 EUV/UV channels sequentially at **12-second cadence**. Special modes include:

- **고속 모드**: 10초 또는 2초 케이던스. 플레어 캠페인이나 특수 관측 시 사용.
- **Rice 무손실 압축**: 전형적으로 2배 압축률. 온보드 저장 장치 없이 연속 다운링크하므로, 압축률이 직접 관측 효율에 영향.
- **연속 다운링크**: GEO-sync 궤도의 최대 장점. 전용 White Sands 지상국을 통해 ~67 Mbit/s 연속 전송. LEO 궤도(예: TRACE, Hinode)와 달리 지상국 접촉 시간 제약이 없다.

- **High-speed mode**: 10 s or 2 s cadence. Used during flare campaigns or special observations.
- **Rice lossless compression**: Typical compression ratio of 2×. Since data is continuously downlinked without on-board storage, the compression ratio directly affects observing efficiency.
- **Continuous downlink**: The primary advantage of GEO-sync orbit. Continuous transmission at ~67 Mbit/s through the dedicated White Sands ground station. Unlike LEO orbits (e.g., TRACE, Hinode), there are no ground station contact time constraints.

데이터 전송량은 하루 약 **1.5 TB**(압축 후)로, 이는 당시 기준으로 전례 없는 규모이다. 비압축 데이터는 약 2 TB/day에 달한다.

Data transmission volume is approximately **1.5 TB/day** (after compression), unprecedented at the time. Uncompressed data reaches approximately 2 TB/day.

### §5 Data Processing / 데이터 처리 (pp. 37–39)

AIA 데이터는 단계별로 처리된다:

AIA data is processed in stages:

| 수준 / Level | 처리 내용 / Processing | 설명 / Description |
|---|---|---|
| **Level 0** | 원시 데이터 / Raw data | 텔레메트리에서 추출한 그대로의 FITS 파일 |
| **Level 1** | 기본 보정 / Basic calibration | 다크(dark) 보정, 플랫필드(flat-field), 스파이크 제거(despiking), 불량 픽셀 보정(bad pixel correction) |
| **Level 1.5** | 정밀 보정 / Refined calibration | 회전(rotation) 보정, 판 스케일(plate scale) 통일, 채널 간 정렬(co-alignment) |

| **Level 0** | Raw data | FITS files as extracted from telemetry |
| **Level 1** | Basic calibration | Dark subtraction, flat-field, despiking, bad pixel correction |
| **Level 1.5** | Refined calibration | Rotation correction, plate scale normalization, inter-channel co-alignment |

데이터 관리는 Stanford 대학교의 **JSOC(Joint Science Operations Center)**에서 수행한다. **DRMS(Data Record Management System)**가 데이터 아카이빙과 검색을 담당하며, 관측 후 약 **48시간 이내**에 공개된다. 모든 AIA 데이터는 자유롭게(freely) 접근 가능하며, SunPy와 같은 Python 라이브러리를 통해서도 다운로드할 수 있다.

Data management is handled by **JSOC (Joint Science Operations Center)** at Stanford University. The **DRMS (Data Record Management System)** manages data archiving and retrieval, with data becoming publicly available within approximately **48 hours** of observation. All AIA data is freely accessible and can also be downloaded through Python libraries such as SunPy.

### §6 First Results and Conclusion / 첫 관측 결과와 결론 (pp. 39–40)

AIA의 첫 과학 영상은 **2010년 3월 27일**에 취득되었다. SDO/AIA는 첫 영상 공개 직후부터 태양 물리학 커뮤니티의 주요 데이터 소스로 자리 잡았다. 운용 듀티 사이클은 **95% 이상**이며, 이는 GEO-sync 궤도의 연속 관측 능력과 높은 시스템 신뢰성을 반영한다.

AIA's first science images were obtained on **27 March 2010**. SDO/AIA established itself as the primary data source for the solar physics community immediately after the first image release. The operational duty cycle exceeds **95%**, reflecting both the continuous observation capability of the GEO-sync orbit and high system reliability.

---

## 3. Key Takeaways / 핵심 시사점

1. **4대 독립 망원경으로 동시 관측 / Simultaneous observation with 4 independent telescopes**: AIA는 4대의 망원경을 사용하여 모든 10개 채널을 사실상 동시에(10–12초 이내) 관측한다. 이는 EIT나 TRACE가 한 번에 하나의 채널만 관측할 수 있었던 것과 근본적으로 다르다. 동시 다파장 관측은 DEM 분석의 정확도를 극적으로 향상시켰다.
   AIA uses four telescopes to observe all 10 channels virtually simultaneously (within 10–12 seconds). This is fundamentally different from EIT or TRACE, which could observe only one channel at a time. Simultaneous multi-wavelength observation dramatically improved the accuracy of DEM analysis.

2. **16배 더 많은 픽셀 / 16× more pixels**: 4096×4096 = 16,777,216 픽셀은 EIT/TRACE의 1024×1024 = 1,048,576 픽셀의 정확히 16배이다. 이를 통해 전체 태양 디스크를 0.6" 분해능으로 커버하면서도 충분한 시야를 확보한다.
   4096×4096 = 16,777,216 pixels is exactly 16× the 1024×1024 = 1,048,576 pixels of EIT/TRACE. This covers the full solar disk at 0.6" resolution while maintaining adequate field of view.

3. **60배 빠른 케이던스 / 60× faster cadence**: 12초 케이던스는 EIT의 12분(720초) 대비 60배, TRACE의 약 2분 대비 약 10배 빠르다. 이는 플레어 초기 진화, CME 발생, 코로나 루프 진동 등 빠른 현상의 연구를 가능케 했다.
   The 12-second cadence is 60× faster than EIT's 12 minutes (720 seconds) and ~10× faster than TRACE's ~2 minutes. This enabled studies of fast phenomena such as early flare evolution, CME onset, and coronal loop oscillations.

4. **GEO-sync 궤도 + 전용 지상국 / GEO-sync orbit + dedicated ground station**: 지구정지 경사궤도는 연속 다운링크를 가능케 하며, 전용 White Sands 지상국은 ~67 Mbit/s의 데이터 전송률을 보장한다. 이로 인해 온보드 저장 장치 없이도 ~1.5 TB/day의 데이터를 전송할 수 있다. LEO 위성(TRACE, Hinode)의 제한된 지상국 접촉 시간 문제를 완전히 해결하였다.
   The geosynchronous inclined orbit enables continuous downlink, and the dedicated White Sands ground station guarantees ~67 Mbit/s data transfer rate. This allows transmission of ~1.5 TB/day without on-board storage. This completely solved the limited ground station contact time problem of LEO satellites (TRACE, Hinode).

5. **온도 감응 함수(Fig. 13)의 역사적 중요성 / Historical importance of temperature response functions (Fig. 13)**: AIA의 온도 감응 함수(CHIANTI 기반)를 보여주는 Fig. 13은 태양 물리학 기기 논문에서 가장 많이 인용되는 그림 중 하나이다. 이 함수는 DEM 역산(inversion), 코로나 온도 진단, 그리고 거의 모든 AIA 관련 연구의 기초 도구가 된다.
   Fig. 13, showing AIA's temperature response functions (CHIANTI-based), is one of the most-cited figures in solar physics instrumentation literature. These functions serve as fundamental tools for DEM inversion, coronal temperature diagnostics, and virtually all AIA-related research.

6. **4개의 새로운 전면 EUV 채널 / Four new full-disk EUV channels**: 94, 131, 211, 335 Å 채널은 AIA 이전에는 전면 분해능(full-disk resolution)으로 영상화된 적이 없었다. 특히 94 Å(Fe XVIII, log T = 6.8)과 131 Å(Fe XXI, log T = 7.0)은 플레어 플라즈마 온도를 직접 진단하는 데 핵심적이다.
   The 94, 131, 211, and 335 Å channels had never been imaged at full-disk resolution before AIA. In particular, 94 Å (Fe XVIII, log T = 6.8) and 131 Å (Fe XXI, log T = 7.0) are crucial for directly diagnosing flare plasma temperatures.

7. **데이터 파이프라인과 공개 접근성 / Data pipeline and open access**: Level 0 → 1 → 1.5 처리 파이프라인과 JSOC/DRMS 시스템은 관측 후 48시간 이내에 전 세계에 데이터를 공개한다. 이러한 개방적 데이터 정책은 AIA가 태양 물리학 연구의 표준 도구가 된 핵심 요인이다.
   The Level 0 → 1 → 1.5 processing pipeline and JSOC/DRMS system make data publicly available worldwide within 48 hours of observation. This open data policy is a key factor in AIA becoming the standard tool for solar physics research.

8. **기술 유산의 계보 / Heritage lineage**: NIXT(1990) → EIT(1995) → TRACE(1998) → STEREO/EUVI(2006) → AIA(2010)로 이어지는 계보에서, 각 단계는 이전 기기의 교훈을 반영하여 진화하였다. AIA는 이 계보의 정점(culmination)으로, 다층막 기술, 후면 박화 CCD, 영상 안정화, 자동 노출 제어 등 모든 핵심 기술의 최종 통합체이다.
   In the lineage from NIXT (1990) → EIT (1995) → TRACE (1998) → STEREO/EUVI (2006) → AIA (2010), each step evolved by incorporating lessons from its predecessors. AIA represents the culmination of this lineage — the final integration of all key technologies including multilayer coatings, back-thinned CCDs, image stabilization, and automatic exposure control.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Pixel Scale and FOV / 픽셀 스케일 및 시야

$$\text{pixel scale} = \frac{d_{\text{pixel}}}{f} = \frac{12\;\mu\text{m}}{4125\;\text{mm}} = 2.91 \times 10^{-6}\;\text{rad} \approx 0.6''$$

$$\text{FOV} = 4096 \times 0.6'' = 2457.6'' \approx 41'$$

여기서 $d_{\text{pixel}}$은 픽셀 크기(12 μm), $f$는 유효 초점 거리(4125 mm = 4.125 m)이다.

Where $d_{\text{pixel}}$ is the pixel size (12 μm) and $f$ is the effective focal length (4125 mm = 4.125 m).

### 4.2 Diffraction Limit / 회절 한계

$$\theta_{\text{diff}} = 1.22 \times \frac{\lambda}{D}$$

171 Å 채널의 경우:

For the 171 Å channel:

$$\theta_{\text{diff}} = 1.22 \times \frac{171 \times 10^{-10}\;\text{m}}{0.20\;\text{m}} = 1.04 \times 10^{-7}\;\text{rad} \approx 0.021''$$

이는 0.6"/pixel보다 훨씬 작으므로, AIA는 **픽셀 제한(pixel-limited)** 기기이다 — 광학 성능이 검출기 분해능을 크게 초과한다.

This is far smaller than 0.6"/pixel, so AIA is a **pixel-limited** instrument — optical performance far exceeds detector resolution.

### 4.3 Data Rate / 데이터 전송률

$$R_{\text{raw}} = \frac{N_{\text{pix}} \times b \times N_{\text{ch}}}{\Delta t} = \frac{4096^2 \times 16\;\text{bit} \times 8}{12\;\text{s}} \approx 178\;\text{Mbit/s}$$

Rice 무손실 압축(약 2배) 적용 후:

After Rice lossless compression (~2×):

$$R_{\text{compressed}} \approx \frac{178}{2} \approx 67\;\text{Mbit/s}$$

여기서 $N_{\text{pix}} = 4096^2 = 16,777,216$, $b = 16$ bit (ADC 비트 수), $N_{\text{ch}} = 8$ (EUV/UV 채널 수), $\Delta t = 12$ s (케이던스).

Where $N_{\text{pix}} = 4096^2 = 16,777,216$, $b = 16$ bit (ADC bit depth), $N_{\text{ch}} = 8$ (number of EUV/UV channels), $\Delta t = 12$ s (cadence).

### 4.4 Temperature Response Function / 온도 감응 함수

관측된 DN(Data Number) 신호 $g_i$는 채널 $i$에 대해:

The observed DN (Data Number) signal $g_i$ for channel $i$ is:

$$g_i = \int_0^{\infty} K_i(T) \times \text{DEM}(T)\;dT$$

여기서:
- $K_i(T)$: 채널 $i$의 온도 감응 함수 [DN cm⁵ s⁻¹ pixel⁻¹]
- $\text{DEM}(T) = n_e^2 \frac{dh}{dT}$: 미분 방출 측정(Differential Emission Measure) [cm⁻⁵ K⁻¹]
- $T$: 전자 온도 [K]

Where:
- $K_i(T)$: temperature response function of channel $i$ [DN cm⁵ s⁻¹ pixel⁻¹]
- $\text{DEM}(T) = n_e^2 \frac{dh}{dT}$: Differential Emission Measure [cm⁻⁵ K⁻¹]
- $T$: electron temperature [K]

$K_i(T)$는 유효 면적(effective area), 양자 효율, CHIANTI 원자 데이터를 종합하여 계산된다. AIA의 10개 채널(특히 7개 EUV 채널)은 넓은 온도 범위(log T = 4.7–7.3)를 커버하므로, DEM 역산(inversion)을 통해 코로나 플라즈마의 온도 분포를 재구성할 수 있다.

$K_i(T)$ is calculated by combining effective area, quantum efficiency, and CHIANTI atomic data. AIA's 10 channels (especially the 7 EUV channels) cover a broad temperature range (log T = 4.7–7.3), enabling reconstruction of coronal plasma temperature distributions through DEM inversion.

### 4.5 Information Rate Improvement / 정보율 향상

EIT 대비 AIA의 정보율 향상 배수:

AIA's information rate improvement factor over EIT:

$$\text{Factor} = \left(\frac{4096}{1024}\right)^2 \times \frac{8}{4} \times \frac{720\;\text{s}}{12\;\text{s}} = 16 \times 2 \times 60 = 1920$$

최대 향상 배수(케이던스와 노출 시간 최적화 포함)는 약 22,000배에 달한다.

The maximum improvement factor (including cadence and exposure time optimization) reaches approximately 22,000×.

### 4.6 Worked Example: 171 Å Channel SNR / 수치 예제: 171 Å 채널 SNR

조용한 태양(quiet Sun)에서 171 Å 채널의 전형적 신호:

Typical signal from the quiet Sun in the 171 Å channel:

- 노출 시간 / Exposure: ~2.9 s
- 전형적 DN / Typical DN: ~3000 DN/pixel
- CCD 이득 / CCD gain: ~17 e⁻/DN
- 총 전자수 / Total electrons: 3000 × 17 ≈ 51,000 e⁻
- 광자 잡음 / Photon noise: $\sqrt{51000} \approx 226$ e⁻
- 읽기 잡음 / Read noise: ~25 e⁻
- SNR ≈ $\frac{51000}{\sqrt{51000 + 25^2}} \approx \frac{51000}{231} \approx 221$

이는 단일 픽셀 기준으로도 매우 높은 SNR이며, AIA 영상의 뛰어난 품질을 반영한다.

This is a very high SNR even on a per-pixel basis, reflecting the excellent quality of AIA images.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1990       1995       1998       2006       2010       2020
  |          |          |          |          |          |
NIXT       EIT       TRACE    EUVI/       AIA      EUI/
(sounding  (SOHO)    (SMEX)   STEREO     (SDO)    Solar
 rocket)                                          Orbiter
  |          |          |          |          |          |
  ·----------·----------·----------·----------·----------·
  
  Normal-incidence     First high-res    Full-disk      Next-gen
  multilayer EUV       EUV imaging      + high-res     close-up
  demonstrated         from space       + high-cadence  EUV
  
Key Milestones:
  
1990  NIXT — first normal-incidence multilayer EUV telescope (sounding rocket)
1991  MSSTA — Multi-Spectral Solar Telescope Array (sounding rocket)
1995  SOHO/EIT — first space-based full-disk EUV imager (1024², 4 channels, 12 min)
1998  TRACE — first sub-arcsecond EUV imager (1024², 3 EUV + UV, 8.5' FOV)
2006  STEREO/EUVI — first stereoscopic EUV imaging (2048², 4 channels)
2010  SDO/AIA — full-disk, high-res, high-cadence (4096², 10 channels, 12 s) ← THIS PAPER
2020  Solar Orbiter/EUI — close-up EUV at 0.28 AU (HRI: 0.5" pixel)
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| 논문 / Paper | 연결 / Connection |
|---|---|
| **#8 Domingo et al. (1995) — SOHO** | AIA의 모미션인 SDO는 SOHO의 후계 미션. SOHO가 L1 궤도에서 시연한 연속 태양 관측 개념을 GEO-sync 궤도에서 대폭 강화. / SDO is the successor to SOHO. The continuous solar observation concept demonstrated by SOHO at L1 was greatly enhanced in GEO-sync orbit. |
| **#9 Delaboudinière et al. (1995) — EIT** | AIA의 직접적 선행 기기. 4-quadrant 다층막 설계, 풀디스크 EUV 영상의 원형(prototype). AIA는 EIT 대비 정보율 400–22,000배 향상. / Direct predecessor instrument. Prototype of 4-quadrant multilayer design and full-disk EUV imaging. AIA achieves 400–22,000× information rate improvement over EIT. |
| **#10 Brueckner et al. (1995) — LASCO** | 동일 시대 SOHO 기기. LASCO의 코로나그래프 데이터와 AIA의 코로나 영상은 CME 연구에서 상호 보완적. / Contemporary SOHO instrument. LASCO coronagraph data and AIA coronal images are complementary for CME research. |
| **#11 Handy et al. (1999) — TRACE** | AIA의 가장 직접적인 기술적 선행자. TRACE의 Cassegrain 설계, 자동 노출 제어(AEC), 가이드 망원경/ISS를 AIA가 계승. 그러나 TRACE의 8.5' FOV → AIA의 41' FOV, 1024² → 4096², lumogen CCD → back-thinned CCD로 대폭 업그레이드. / Most direct technical predecessor. AIA inherited TRACE's Cassegrain design, AEC, and GT/ISS. However, major upgrades: TRACE's 8.5' FOV → AIA's 41', 1024² → 4096², lumogen CCD → back-thinned CCD. |
| **#13 Scherrer et al. (2012) — HMI** | SDO의 동반 기기. AIA와 CCD/카메라 설계를 공유(e2v CCD203-82, 단 HMI는 전면 조사 버전). AIA의 코로나 구조와 HMI의 광구 자기장 데이터를 결합하면 자기장–코로나 연결(magnetic field–corona coupling)을 연구할 수 있다. / SDO companion instrument. Shares CCD/camera design with AIA (e2v CCD203-82, though HMI uses front-illuminated version). Combining AIA coronal structure with HMI photospheric magnetic field data enables study of magnetic field–corona coupling. |
| **#7 Tomczyk et al. (2016) — CoMP/COSMO** | 지상 코로나 관측 기기. AIA가 EUV 코로나를 영상화하는 반면, CoMP는 근적외선 코로나 분광편광 관측을 수행. 코로나 자기장 진단에서 상호 보완적. / Ground-based coronal instrument. While AIA images the EUV corona, CoMP performs near-IR coronal spectropolarimetry. Complementary in coronal magnetic field diagnostics. |

---

## 7. References / 참고문헌

- Lemen, J.R., et al., "The Atmospheric Imaging Assembly (AIA) on the Solar Dynamics Observatory (SDO)," Solar Physics, 275, 17–40, 2012. [DOI: 10.1007/s11207-011-9776-8](https://doi.org/10.1007/s11207-011-9776-8)
- Boerner, P., et al., "Initial Calibration of the Atmospheric Imaging Assembly (AIA) on the Solar Dynamics Observatory (SDO)," Solar Physics, 275, 41–66, 2012. [DOI: 10.1007/s11207-011-9804-8](https://doi.org/10.1007/s11207-011-9804-8)
- Pesnell, W.D., Thompson, B.J., Chamberlin, P.C., "The Solar Dynamics Observatory (SDO)," Solar Physics, 275, 3–15, 2012. [DOI: 10.1007/s11207-011-9841-3](https://doi.org/10.1007/s11207-011-9841-3)
- Delaboudinière, J.-P., et al., "EIT: Extreme-Ultraviolet Imaging Telescope for the SOHO Mission," Solar Physics, 162, 291–312, 1995. [DOI: 10.1007/BF00733432](https://doi.org/10.1007/BF00733432)
- Handy, B.N., et al., "The Transition Region and Coronal Explorer," Solar Physics, 187, 229–260, 1999. [DOI: 10.1023/A:1005166902804](https://doi.org/10.1023/A:1005166902804)
- Scherrer, P.H., et al., "The Helioseismic and Magnetic Imager (HMI) Investigation for the Solar Dynamics Observatory (SDO)," Solar Physics, 275, 207–227, 2012. [DOI: 10.1007/s11207-011-9834-2](https://doi.org/10.1007/s11207-011-9834-2)
- Wuelser, J.-P., et al., "EUVI: the STEREO-SECCHI Extreme Ultraviolet Imager," Proc. SPIE, 5171, 111–122, 2004. [DOI: 10.1117/12.506877](https://doi.org/10.1117/12.506877)
- Dere, K.P., et al., "CHIANTI — An Atomic Database for Emission Lines," A&AS, 125, 149–173, 1997. [DOI: 10.1051/aas:1997368](https://doi.org/10.1051/aas:1997368)
- Domingo, V., Fleck, B., Poland, A.I., "The SOHO Mission: An Overview," Solar Physics, 162, 1–37, 1995. [DOI: 10.1007/BF00733425](https://doi.org/10.1007/BF00733425)
