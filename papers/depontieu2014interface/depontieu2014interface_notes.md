---
title: "The Interface Region Imaging Spectrograph (IRIS)"
authors: De Pontieu, B., Title, A.M., Lemen, J.R. et al.
year: 2014
journal: "Solar Physics, Vol. 289, No. 7, pp. 2733–2779"
doi: "10.1007/s11207-014-0485-y"
topic: Solar_Observation
tags: [IRIS, UV spectroscopy, chromosphere, transition region, interface region, slit-jaw imager, space instrumentation]
status: completed
date_started: 2026-04-16
date_completed: 2026-04-16
---

# 16. The Interface Region Imaging Spectrograph (IRIS) / 인터페이스 영역 영상 분광기 (IRIS)

---

## 1. Core Contribution / 핵심 기여

IRIS(Interface Region Imaging Spectrograph)는 태양 대기에서 가장 복잡하고 이해가 부족한 영역인 **채층(chromosphere)과 전이 영역(transition region)** — 합쳐서 "인터페이스 영역(interface region)"이라 부르는 — 을 전례 없는 공간·시간·분광 해상도로 관측하기 위해 설계된 NASA Small Explorer 우주 관측소입니다. 19 cm Cassegrain 망원경에 슬릿 기반 이중 대역 UV 분광기(FUV: 1332–1407 Å, NUV: 2783–2835 Å)와 4개 필터의 슬릿 턱 영상기(SJI)를 결합하여, **0.33 arcsec 공간 분해능, 2초 분광 주기, 1 km s⁻¹ 속도 분해능**을 달성했습니다. 이는 이전 UV 분광기(SOHO/SUMER, Hinode/EIS)보다 처리량이 한 자릿수 이상 높고, 공간 분해능이 5–10배 향상된 것입니다. IRIS는 2013년 6월 27일 Pegasus-XL 로켓으로 태양 동기 궤도에 발사되었으며, 하루 약 8 GB의 보정 데이터를 관측 후 수일 내에 공개합니다.

IRIS (Interface Region Imaging Spectrograph) is a NASA Small Explorer spacecraft designed to observe the **chromosphere and transition region** — collectively called the "interface region" — the most complex and poorly understood layers of the solar atmosphere. Combining a 19-cm Cassegrain telescope with a slit-based dual-bandpass UV spectrograph (FUV: 1332–1407 Å, NUV: 2783–2835 Å) and a four-filter slit-jaw imager (SJI), it achieves **0.33 arcsec spatial resolution, 2-second spectral cadence, and 1 km s⁻¹ velocity resolution**. This represents more than an order of magnitude improvement in throughput and 5–10 times better spatial resolution than previous UV spectrographs (SOHO/SUMER, Hinode/EIS). Launched on 27 June 2013 via a Pegasus-XL rocket into a Sun-synchronous orbit, IRIS produces approximately 8 GB of calibrated data per day, made publicly available within a few days of observation.

본 논문은 47페이지에 걸쳐 IRIS의 과학 목표(Section 3), 기기 설계(Section 4), 관측 시퀀서(Section 5), 운용(Section 6), 보정(Section 7), 데이터 처리(Section 8), 수치 모델링(Section 9)을 포괄적으로 기술한 기기 참고 논문(instrument reference paper)입니다.

This 47-page paper is the comprehensive instrument reference, covering IRIS's science goals (Section 3), instrument design (Section 4), observing sequencer (Section 5), operations (Section 6), calibration (Section 7), data processing (Section 8), and numerical modeling (Section 9).

---

## 2. Reading Notes / 읽기 노트

### Section 1: Introduction / 서론 (pp. 2735–2736)

채층과 전이 영역(TR)은 태양 표면과 코로나 사이의 복잡한 **인터페이스 영역**을 형성합니다. 태양 활동을 구동하는 기계적 에너지의 거의 전부가 이 영역에서 열과 복사로 변환되며, 코로나 가열과 태양풍 구동을 위해 빠져나가는 것은 극히 일부입니다.

The chromosphere and transition region (TR) form a complex **interface region** between the solar surface and corona. Almost all mechanical energy driving solar activity is converted to heat and radiation within this region, with only a small fraction leaking through to power coronal heating and the solar wind.

이 영역의 관측이 어려운 이유:
- 플라즈마 $\beta$ 전이: 자기장과 가스 압력이 경쟁하는 영역
- 밀도가 6자릿수 급변, 온도가 5,000 K → 1 MK로 급상승
- non-LTE 복사 전달로 스펙트럼 해석이 비직관적
- 20초 이하의 고시간 해상도와 0.5 arcsec 이하의 고공간 해상도가 동시에 필요

Key observational challenges:
- Plasma $\beta$ transition: magnetic field and gas pressure compete for dominance
- Density drops by six orders of magnitude; temperature surges from 5,000 K to 1 MK
- Non-LTE radiative transfer makes spectral interpretation non-intuitive
- Simultaneous high cadence (<20 s) and high spatial resolution (<0.5 arcsec) required

IRIS가 관측하는 파장 대역은 이전에 로켓(Bates et al., 1969), 벌룬(Lemaire, 1969), 위성(SUMER, EIS) 등으로 관측된 적이 있지만, IRIS는 이전 기기보다 훨씬 높은 해상도와 처리량을 제공합니다.

The spectral ranges IRIS observes have been previously studied with rockets, balloons, and satellites (SUMER, EIS), but IRIS provides far higher resolution and throughput.

### Section 2: IRIS Observatory / IRIS 관측소 (pp. 2739–2741)

#### 2.1 핵심 요구 사양과 달성 / Key Requirements and Achievements

| 요구 사항 / Requirement | IRIS 달성 / Achievement |
|---|---|
| 공간 해상도 <0.5 arcsec | 0.33 arcsec (FUV), 0.4 arcsec (NUV) |
| 시야 ≥120 arcsec | 175×175 arcsec² (SJI), 130×175 arcsec² (래스터) |
| 분광 해상도 5–10 km s⁻¹ | 26 mÅ FUV (~6 km s⁻¹), 53 mÅ NUV (~6 km s⁻¹) |
| 속도 정밀도 ~1 km s⁻¹ | ~1 km s⁻¹ (sub-pixel centroiding) |
| 시간 해상도 <20 s | 2 s (분광), 5 s (SJI) |
| 연속 관측 ≥6개월/년 | 7–8개월/년 (eclipse-free) |

IRIS의 유효 공간 분해능은 0.4 arcsec로, SDO/AIA의 10개 분해 요소, SOHO/SUMER 또는 Hinode/EIS의 25개 분해 요소에 해당합니다. 속도 분해능은 SUMER보다 3배, EIS보다 10배 우수합니다.

IRIS effective spatial resolution of 0.4 arcsec corresponds to 10 resolution elements for each SDO/AIA element and 25 for each SOHO/SUMER or Hinode/EIS element. Velocity resolution is 3× better than SUMER and 10× better than EIS.

#### 2.2 궤도 / Orbit

- Pegasus-XL 로켓으로 2013년 6월 27일 태양 동기 궤도에 발사
- 경사각 97.9°, 근지점 620 km, 원지점 670 km
- 2월~10월: eclipse-free 관측 (연 7–8개월)
- 11월~1월: 지구가 태양을 차단하는 기간

Launched 27 June 2013 into Sun-synchronous orbit; inclination 97.9°, perigee 620 km, apogee 670 km. Eclipse-free viewing from February to October (~7–8 months/year).

#### 2.3 관측소 사양 / Observatory Specifications

- 총 질량: 183 kg (기기 87 kg + 우주선 버스 96 kg)
- 총 전력: 302 W (기기 55 W + 우주선 247 W)
- 태양 전지판: 2개, 총 면적 1.7 m², 340 W 생산
- 자세 제어: 자이로 없음(gyroless), 별 추적기 2개, 반작용 휠 4개, 자력계
- 다운링크: X-band 15 Mbit s⁻¹, 하루 ~15회 패스 (Svalbard KSAT + NEN)
- 유효 다운링크율: 13 Mbit s⁻¹ (LDPC 7/8 오버헤드 제외)
- 일일 데이터량: ~20 GB (비압축), ~60 Gbit (압축 후)

Total mass: 183 kg; total power: 302 W. Gyroless three-axis stabilized. X-band downlink at 15 Mbit s⁻¹, ~15 passes/day. Daily data volume: ~20 GB (uncompressed).

### Section 3: Science Overview / 과학 개요 (pp. 2741–2744)

IRIS의 과학 목표는 세 가지 핵심 질문으로 구성됩니다:

IRIS's science goals center on three key questions:

#### Q1: 채층과 그 너머에서 어떤 유형의 비열적 에너지가 지배하는가?
#### Q1: Which Types of Non-thermal Energy Dominate in the Chromosphere and Beyond?

세 가지 에너지 전달 메커니즘이 경쟁합니다:

Three energy transport mechanisms compete:

1. **파동(Waves)**: 음향파, 대기 중력파, Alfvén파. IRIS는 광구에서 코로나까지 파동의 전파를 분광적으로 추적하여 에너지 플럭스 예산을 확립할 수 있습니다.
   Acoustic waves, atmospheric gravity waves, Alfvén waves. IRIS traces wave propagation spectroscopically from photosphere to corona.

2. **전류와 재결합(Currents and Reconnection)**: 채층 내 콤팩트 피브릴에서 강한 전류가 예상되며, 저항성 소산이 급격한 온도 상승을 유발합니다. IRIS는 열적 진화를 추적하여 전자기 에너지의 열 변환을 정량화합니다.
   Strong currents expected in compact fibrils; resistive dissipation causes rapid temperature rises. IRIS traces thermal evolution to quantify electromagnetic-to-thermal conversion.

3. **네트워크/플라지 자기장 재결합**: 약한 자기 플럭스가 매일 과립 규모에서 부상하여, 강한 네트워크 자기장과 재결합하면 상당한 에너지를 방출할 수 있습니다.
   Weak magnetic flux emerging daily at granular scales reconnects with strong network fields, potentially releasing significant energy.

#### Q2: 채층은 코로나와 태양권으로의 질량·에너지 공급을 어떻게 조절하는가?
#### Q2: How Does the Chromosphere Regulate Mass and Energy Supply to the Corona and Heliosphere?

IRIS는 다음을 관측합니다:
- 복잡한 선 프로파일: 폭발적 상승류, 증발류, 하강류의 공존
- 개방 자기장 영역에서의 파동 에너지와 가속
- 에너지의 하향 전달: 열전도 vs. 입자 침전 (후자는 더 간헐적)

IRIS observes complex line profiles revealing coexisting explosive upflows, evaporative flows, and downflows; wave energy in open-field regions; and downward energy transport via thermal conduction vs. particle precipitation.

#### Q3: 자기 플럭스와 물질은 하층 대기를 어떻게 통과하며, 플레어와 분출에서 플럭스 부상은 어떤 역할을 하는가?
#### Q3: How Does Magnetic Flux Rise Through the Lower Atmosphere, and What Role Does Flux Emergence Play in Flares and Mass Ejections?

자기 플럭스가 태양 표면을 뚫고 나올 때, 기존 자기장과 상호작용합니다. IRIS는 다음을 관측할 수 있습니다:
- 저층 채층에서 수 km s⁻¹의 상승류
- 드레이닝 아크-필라멘트 시스템의 하강류
- 채층과 코로나에서의 자기장 팽창 (Alfvén 속도)
- 플레어 리본의 채층 분광: 재결합과 입자 침전의 추적자

When magnetic flux breaches the solar surface, it interacts with the existing field. IRIS observes upflows of a few km s⁻¹ in the low chromosphere, downdrafts of draining arch-filament systems, field expansion at Alfvén speeds, and chromospheric spectroscopy of flare ribbons as tracers of reconnection and particle precipitation.

### Section 4: Instrument Overview / 기기 개요 (pp. 2744–2752)

#### 4.1 망원경 / Telescope

- 19 cm Cassegrain 망원경 + 활성 보조 거울 (초점 조절)
- AIA 망원경 설계 기반이나, 더 긴 초점 거리 (6.895 m)와 다른 열적 접근법 사용
- 주경 전면의 유전체 코팅이 UV만 반사, 가시광/IR은 ULE 기판을 투과하여 열 흡수판으로
- 시야: ~3 arcmin × 3 arcmin

19-cm Cassegrain telescope with active secondary mirror (focus adjustment). Based on AIA telescope design but with longer focal length (6.895 m) and different thermal approach. Dielectric coating on primary reflects UV only; visible/IR passes through ULE substrate to heat sink. FOV: ~3 arcmin × 3 arcmin.

#### 4.2 분광기 / Spectrograph

빛의 경로 (Figure 9 참조):
1. 망원경에서 분광기 박스로 빛이 입사
2. **슬릿/프리-디스퍼서 프리즘**: 슬릿을 통과한 빛은 FUV(1332–1407 Å)와 NUV(2783–2835 Å)로 분산, 슬릿 반사면은 SJI 경로로 반사
3. **콜리메이터**: FUV와 NUV 빔을 평행화
4. **회절격자**: Horiba Jobin-Yvon 제작, 3600 lines mm⁻¹
5. **카메라 미러 → CCD**: FUV 2개 + NUV 1개 = 3개 CCD

Light path (see Figure 9):
1. Light enters spectrograph box from telescope
2. **Slit/pre-disperser prism**: slit-transmitted light dispersed into FUV and NUV; slit reflective surface directs light to SJI path
3. **Collimator**: collimates FUV and NUV beams
4. **Gratings**: Horiba Jobin-Yvon, 3600 lines mm⁻¹
5. **Camera mirrors → CCDs**: 2 FUV + 1 NUV = 3 CCDs

분광기 채널 상세 (Table 2):

| Band | 파장 범위 / Wavelength [Å] | 분산 / Disp. [mÅ pix⁻¹] | 유효 면적 / EA [cm²] | 온도 범위 / log T |
|---|---|---|---|---|
| FUV 1 | 1331.7–1358.4 | 12.98 | 1.6 | 3.7–7.0 |
| FUV 2 | 1389.0–1407.0 | 12.72 | 2.2 | 3.7–5.2 |
| NUV | 2782.7–2835.1 | 25.46 | 0.2 | 3.7–4.2 |

주요 분광선과 형성 온도 (Table 4):

| 이온 / Ion | 파장 / Wavelength [Å] | log T [K] | 대역 / Band |
|---|---|---|---|
| C II | 1334.5, 1335.7 | 4.3 | FUV 1 |
| Si IV | 1393.8, 1402.8 | 4.8 | FUV 2 |
| O IV | 1399.8, 1401.2 | 5.2 | FUV 2 |
| Mg II k | 2796.4 | 4.0 | NUV |
| Mg II h | 2803.5 | 4.0 | NUV |
| Fe XII | 1349.4 | 6.2 | FUV 1 |
| Fe XXI | 1354.1 | 7.0 | FUV 1 |

핵심 포인트: IRIS는 **5,000 K (광구)에서 10 MK (코로나)**까지의 온도 범위를 커버합니다. 특히 Fe XII와 Fe XXI 선은 활동 영역과 플레어 관측에서 코로나까지의 온도 커버리지를 확장합니다.

Key point: IRIS covers temperatures from **5,000 K (photosphere) to 10 MK (corona)**. Fe XII and Fe XXI lines extend temperature coverage to the corona for active regions and flares.

#### 4.3 슬릿 턱 영상기 (SJI) / Slit-Jaw Imager

SJI는 슬릿 주변의 반사면에서 반사된 빛으로 맥락 영상을 제공합니다. 필터 휠에 6개 필터 (태양 관측용 4개 + 지상 시험용 2개):

SJI provides context images from light reflected off the slit's reflective surface. Six filters in filter wheel (four for solar, two for ground testing):

| 필터 / Filter | 중심 [Å] / Center | 폭 [Å] / Width | 온도 / Temp. [log T] | 주요 방출 / Key emission |
|---|---|---|---|---|
| C II | 1330 | 55 | 3.7–7.0 | C II 1335 Å (채층 상부/TR) |
| Si IV | 1400 | 55 | 3.7–5.2 | Si IV 1394/1403 Å (TR) |
| Mg II h/k | 2796 | 4 | 3.7–4.2 | Mg II k 2796 Å (채층 상부) |
| Mg II wing | 2832 | 4 | 3.7–3.8 | Mg II wing (광구 상부/채층 하부) |

각 SJI의 시야는 **175 × 175 arcsec²**이며, 슬릿 위치의 스펙트럼과 동시에 촬영됩니다.

Each SJI has a FOV of **175 × 175 arcsec²**, acquired simultaneously with spectra at the slit position.

#### 4.4 CCD 검출기 시스템 / CCD Detector System

- 4개의 e2v CCD267 (2061 × 1056 pixels, 13 μm pixel)
- 후면 조사(back-illuminated), 후면 박막화(thinned)
- 양자 효율: ~31% at 1400 Å
- Full well: 150,000 electrons
- 읽기 잡음: <20 electrons
- 2개의 CEB(Camera Electronics Box)가 4개 CCD를 제어
- 온칩 합산: 공간 1×, 2×, 4× / 분광 1×, 2×, 4×, 8×

Four e2v CCD267 devices (2061 × 1056 pixels, 13 μm pixel). Back-illuminated, thinned. QE ~31% at 1400 Å. Full well 150,000 e⁻, read noise <20 e⁻. Two CEBs control four CCDs. On-chip summing: spatial 1×–4×, spectral 1×–8×.

#### 4.5 가이드 망원경과 영상 안정화 시스템 (GT & ISS) / Guide Telescope and Image Stabilization System

- TRACE와 AIA에서 사용된 시스템 기반
- GT: 색수차 보정 굴절 망원경, 5700 Å 중심 대역통과 필터, FWHM 500 Å
- 4개의 포토다이오드로 태양 림 위치 측정 → 변위 오차 신호 생성
- ISS: 32 Hz로 GT 오차 신호를 샘플링, 3개 PZT로 보조 거울 틸트
- 지터 제거: 0.05 arcsec RMS까지 안정화

Based on TRACE and AIA systems. GT: achromatic refractor with bandpass filter centered at 5700 Å (FWHM 500 Å). Four photodiodes measure solar limb position → displacement error signal. ISS samples GT error at 32 Hz, tilts secondary mirror via three PZTs. Stabilizes jitter to 0.05 arcsec RMS.

### Section 5: Instrument Sequencer / 기기 시퀀서 (pp. 2752–2755)

IRIS 관측은 계층적 구조의 시퀀서 테이블로 제어됩니다:

IRIS observations are controlled by a hierarchical structure of sequencer tables:

```
OBS (Observing List)     ← 과학 프로그램 / Science program
 └── FRM (Frame List)    ← 타이밍, 반복, 래스터 스캔 / Timing, repeats, raster scan
      └── FDB (Frame Definition Block)  ← 노출, 압축 / Exposure, compression
           └── CRS (Camera Readout Structure)  ← CCD 영역, 합산 / CCD regions, summing
```

래스터 스캔은 PZT로 보조 거울의 방향을 변경하여 수행됩니다. 기본 래스터 모드 ~50가지 (Table 12):

Raster scanning is performed by changing secondary mirror orientation via PZTs. ~50 basic raster modes (Table 12):

- **Dense rasters**: 스텝 크기 = 슬릿 폭 (0.33 arcsec). 완전한 공간 커버리지.
  Step size = slit width. Complete spatial coverage.
- **Sparse/coarse rasters**: 스텝 크기 > 슬릿 폭 (1 or 2 arcsec). 넓은 영역의 빠른 스캔.
  Step size > slit width. Rapid scans of larger areas.
- **Sit-and-stare**: 래스터 없음. 한 위치의 시간 진화 관측.
  No rastering. Temporal evolution at one position.
- **Multi-point dense/sparse**: 소수의 정지점에서의 래스터. 파동 전파 연구용.
  Rasters with a small number of dwelling locations. For wave propagation studies.

최소 주기: 분광 3초, 영상 5초 (기준선). 최소 가능 스텝 크기: 0.054 arcsec.

Minimum cadence: 3 s spectral, 5 s imaging (baseline). Smallest possible step size: 0.054 arcsec.

### Section 6: Operations / 운용 (pp. 2755–2758)

#### 6.1 일일 운용 / Daily Operations

- 연 7–8개월 연속 관측 (11월–1월은 대기 흡수로 제한)
- X-band 다운링크: Svalbard (~8–9 패스/일), Alaska (~3–4), Wallops (~1–2)
- 평균 데이터율: 0.7 Mbit s⁻¹, 일일 ~60 Gbit (압축 후)
- 온보드 메모리: 48 Gbit SSMB — 여러 궤도에 걸쳐 데이터 저장 가능
- 관측 프로그램: 주 5회 업로드, 데이터 수일 내 공개

7–8 months continuous observation per year. X-band downlink via Svalbard, Alaska, Wallops. Average 0.7 Mbit s⁻¹, ~60 Gbit/day. 48-Gbit onboard memory. Observing programs uploaded 5×/week, data public within days.

#### 6.2 타임라인 / Timeline

TRACE 타임라인 도구 기반으로 매 평일 업로드됩니다. 과학 플래너가 제어하는 주요 기능:
- 태양면 임의 위치 지향 (디스크 중심에서 21 arcmin 이내)
- 태양 자전 추적 on/off
- 궤도 열변형(wobble) 보정
- 우주선 롤 각도 ±90° 조정 (림 관측 시)
- AEC(자동 노출 제어) 파라미터 설정

Based on TRACE timeline tool, uploaded every weekday. Key planner-controlled functions: pointing anywhere on the solar disk (within 21 arcmin of disk center), solar rotation tracking, orbital wobble correction, spacecraft roll ±90°, AEC parameter setting.

#### 6.4 자동 노출 제어 (AEC) / Automatic Exposure Control

플레어 관측 시 포화를 방지하기 위한 핵심 기능입니다. SJI 밝기에 기반하여 모든 검출기의 노출 시간을 자동 조정합니다. 스펙트럼 기반 AEC는 래스터 스캔 시 입력 데이터 지연 때문에 효율적이지 않으므로, IRIS는 SJI 기반 AEC 방식을 채택했습니다.

Critical feature to prevent saturation during flares. Automatically adjusts exposure times for all detectors based on SJI brightness. Spectrum-based AEC is inefficient during raster scans due to input data lag, so IRIS adopted SJI-based AEC.

### Section 7: Calibration / 보정 (pp. 2758–2767)

#### 7.1 CCD 특성화 / CCD Characterization

다크 레벨 모델:

$$D_j = P_j[T_{\text{CEB}_j}(t - \delta t_j)] + e^{(a_j + b_j T_{\text{CCD}_j})} n_x n_y t_{\text{int}} + \Delta D_j(x, n_x, n_y, t_{\text{int}})$$

- $P_j$: CEB 온도에 의존하는 페데스탈 레벨
- 지수항: CCD 온도에 의존하는 다크 전류 (합산과 적분 시간에 비례)
- $\Delta D_j$: 파장 방향 다크 형상 변화

읽기 간섭 패턴: 두 CEB가 동시 읽기 시 ~340 pixel 주기의 패턴 발생 (FUV SG에서 ±3 DN). 비동시 읽기로 제거 가능하나 시간 해상도 저하.

Read-interference pattern: ~340-pixel-period pattern (±3 DN in FUV SG) when both CEBs read simultaneously. Eliminated by non-simultaneous reads at cost of cadence.

#### 7.2 플랫필드 / Flat Field

두 가지 전략:

**분광기**: 태양 관측에서 스펙트럼 프로파일을 추출한 후 푸리에 변환으로 업샘플링. Mg II 선 코어의 높은 대비도 때문에 충분한 평균화가 필요 (최소 200 영상).

**SJI**: Chae (2004) 기법 — 20 arcsec Reuleaux 삼각형 디더 패턴으로 상대적으로 이동된 태양 영상에서 정지 플랫필드 패턴을 추출. 15 × 3 플랫으로 충분.

Two strategies: **Spectrograph** — extract spectral profile from solar observations, up-sample via Fourier transform (minimum 200 images). **SJI** — Chae (2004) technique with 20-arcsec Reuleaux triangle dither pattern (15 × 3 flats sufficient).

#### 7.3 광학 성능 / Optical Performance

- **초점**: 보조 거울 2.28 μm 단위 조정 가능. 최적 초점 위치 ≈ −115 steps. 회절 한계에 근접.
  Focus adjustable in 2.28 μm increments. Best focus ≈ −115 steps. Near diffraction-limited.

- **공간 분해능**: NUV SJI의 MTF 딥이 ~1 cycle arcsec⁻¹ (중앙 차폐 회절 한계). MTF 10%가 2.6 cycles arcsec⁻¹에서 달성 — 분해능 요구 사항 초과. FUV 영상은 거의 pixel-limited: PSF 코어가 단일 밝은 픽셀에 34%, 3×3에 70% 집중.
  NUV SJI MTF dip at ~1 cycle arcsec⁻¹ (central obscuration). MTF falls to 10% at 2.6 cycles arcsec⁻¹ — exceeds resolution requirement. FUV images nearly pixel-limited: PSF core 34% in single bright pixel, 70% in 3×3.

- **분광 해상도**: Nyquist 기준으로 제한됨. FUV: FWHM 25.85 mÅ, NUV: 50.54 mÅ. 설계 사양에 도달.
  Limited by Nyquist criterion. FUV: FWHM 25.85 mÅ, NUV: 50.54 mÅ. Meets design specifications.

#### 7.4 데이터 압축 / Compression

두 단계 압축:
1. **Rice 압축** (무손실): 연속 픽셀의 차이 인코딩. ~2:1 압축비.
2. **LUT 압축** (손실): AIA/HMI LUT 사용. 제곱근 함수 기반으로 일정 S/N 유지. ~2:1 추가 압축.

Two-stage compression: (1) Lossless Rice compression (~2:1) encoding running differences. (2) Lossy LUT compression using AIA/HMI lookup tables based on square-root function maintaining constant S/N (~2:1 additional).

#### 7.5–7.6 지향 안정성과 파장 보정 / Pointing Stability and Wavelength Calibration

**궤도 열변형(Wobble)**: GT 마운트의 열 변형으로 궤도 주기(~97분)의 지향 진동 발생. x 방향 ~3 arcsec, y 방향 ~1 arcsec peak-to-peak. 궤도 워블 테이블(OWT)로 보정. 잔차: ~2 IRIS pixel.

**파장 보정**: 기하 보정(회전, 곡률, 비선형성 제거) + 상대 보정(열적 드리프트). Ni I 2799.474 Å 중성선의 열적 이동을 기준으로 NUV 파장 보정을 수행하고, FUV 1/2와의 상관관계를 이용하여 전체 보정. 절대 정밀도: ~1 km s⁻¹.

Orbital wobble: thermal flexing causes ~3 arcsec (x) and ~1 arcsec (y) peak-to-peak oscillation per orbit (~97 min). Corrected by orbital wobble tables (OWT); residual ~2 IRIS pixels. Wavelength calibration: geometric correction + relative calibration using Ni I 2799.474 Å neutral line thermal drift. Absolute precision: ~1 km s⁻¹.

#### 7.7 처리량 / Throughput

UV-밝은 별(HD 86360, HD 91316) 관측으로 유효 면적과 방사 보정 인자를 측정:

Effective areas and radiometric calibration measured using UV-bright stars:

| 채널 / Channel | 유효 면적 (발사 전) / EA Pre-launch [cm²] | 유효 면적 (비행) / EA Flight [cm²] | 방사 보정 / Radiometric [erg s⁻¹ sr⁻¹ cm⁻² Å⁻¹ / DN s⁻¹] |
|---|---|---|---|
| SJI 1330 | 0.5 | 0.41 | N/A |
| SJI 1400 | 0.6 | 0.81 | N/A |
| FUV 1 SG | 1.6 | 1.2 | 2960 |
| FUV 2 SG | 2.2 | 2.2 | 1600 |
| NUV SG | 0.2 | N/A | N/A |

### Section 8: Data Processing / 데이터 처리 (pp. 2768–2770)

IRIS 데이터는 SDO와 Hinode의 인프라를 활용하여 처리됩니다.

IRIS data processing leverages SDO and Hinode heritage infrastructure.

| 레벨 / Level | 설명 / Description |
|---|---|
| 0 | 원시 디패킷화 영상 / Depacketized raw images with housekeeping |
| 1 | 축 미러링, North "up" (0° 롤), λ/x 증가 방향 정렬 / Axes reoriented, North up, increasing λ/x |
| 1.5 | 다크·스파이크 제거, 플랫필드, 기하·파장 보정 / Dark/spike removal, flat-field, geometric/wavelength calibration |
| 1.6 | 물리 단위 변환 (노출·광자 변환) / Physical units (exposure and photon conversion) |
| **2** | **래스터 + SJI 시계열로 재구성. 표준 과학 데이터.** / **Recast as rasters and SJI time series. Standard science product.** |
| 3 | CRISPEX용 4D 큐브 (NUV/FUV) + 3D 큐브 (SJI) / 4D cubes for CRISPEX (NUV/FUV) + 3D cubes (SJI) |
| HCR | 관측 시퀀스 기술. Heliophysics Coverage Registry. / Observing sequence descriptions for HCR at LMSAL |

Level 2가 가장 널리 사용되는 과학 데이터 제품입니다. 두 가지 유형:
- **스펙트럼 래스터**: OBS 리스트 한 실행 내의 스펙트럼 프레임 모음
- **SJI 시계열**: 한 필터/CRS의 모든 SJI 영상을 하나의 파일에 저장

Level 2 is the most widely used science product. Two types: spectral rasters (collection of spectral frames within one OBS execution) and SJI time series (all SJI images for one filter/CRS in one file).

데이터 접근: IRIS 웹사이트(iris.lmsal.com), Hinode archive (University of Oslo), Heliophysics Coverage Registry, Virtual Solar Observatory를 통해 제공.

Data access via IRIS website (iris.lmsal.com), Hinode archive (University of Oslo), Heliophysics Coverage Registry, and Virtual Solar Observatory.

### Section 9: Numerical Modeling / 수치 모델링 (pp. 2770–2771)

IRIS 과학 조사에는 광범위한 **복사-MHD 수치 시뮬레이션** 구성 요소가 포함됩니다. University of Oslo의 **Bifrost** 코드가 핵심입니다.

The IRIS science investigation includes an extensive **radiative-MHD numerical simulation** component. The **Bifrost** code from the University of Oslo is central.

모델링 노력의 초점:
- 공간 분해능 향상, 소규모 자기 플럭스 부상 포함
- 수소와 헬륨의 비평형 이온화
- 부분 이온화 채층에서의 이온-중성자 상호작용
- 하이브리드 PIC/MHD 접근법에 의한 운동론적 효과
- C II 1335 Å, Mg II h/k 2796/2803 Å 선 형성의 해석
- 다양한 수치 모델의 물리적·진단적 결과 비교

Modeling efforts focus on: increased spatial resolution, non-equilibrium ionization of H and He, ion-neutral interactions in partially ionized chromosphere, kinetic effects via hybrid PIC/MHD, C II and Mg II line formation interpretation, and comparison of physical/diagnostic results from various numerical models.

IRIS 시뮬레이터: 관측과 동일한 관측 테이블을 사용하여 모델을 "관측"할 수 있는 도구. SolarSoft 트리에서 제공.

IRIS simulator: tool to "observe" models using the same observing tables as actual observations. Available in the IRIS SolarSoft tree.

### Section 10: Conclusion / 결론 (p. 2772)

IRIS는 2013년 7월 17일 첫 관측을 수행한 이후, 60일 초기 관측 계획을 성공적으로 완료했습니다. sit-and-stare, 밀집/성긴 래스터, 다양한 시야 크기의 관측 데이터를 수집했으며, 고도로 역동적이고 미세하게 구조화된 채층과 전이 영역의 모습을 보여주었습니다. 모든 채널(FUV, NUV)이 우수하게 작동하며, 보정 데이터는 수일 내에 공개됩니다.

IRIS acquired first light on 17 July 2013 and successfully completed its initial 60-day observing plan. All channels perform well, calibrated data are made public within days.

---

## 3. Key Takeaways / 핵심 시사점

1. **인터페이스 영역의 핵심성 / Centrality of the interface region** — 채층과 전이 영역은 태양의 모든 질량·에너지 흐름이 통과하는 영역이며, 코로나 가열에 필요한 에너지보다 1–2자릿수 더 많은 가열을 필요로 합니다. IRIS는 이 영역에 최초로 전용된 관측소입니다.
   The chromosphere and TR are the conduit for all solar mass and energy flow, requiring 1–2 orders of magnitude more heating than the corona. IRIS is the first dedicated observatory for this region.

2. **10배 이상의 성능 도약 / Order-of-magnitude performance leap** — IRIS는 SUMER/EIS 대비 처리량 10배 이상, 공간 분해능 5–10배, 속도 분해능 3–10배 향상을 달성했습니다. 이 성능은 이전에 간접적으로만 추론되던 다중 스펙트럼 성분을 직접 분리할 수 있게 합니다.
   IRIS achieves >10× throughput, 5–10× spatial resolution, and 3–10× velocity resolution improvement over SUMER/EIS, enabling direct separation of previously indirectly inferred multi-component spectra.

3. **동시적 분광+영상 / Simultaneous spectroscopy + imaging** — 분광기(SG)와 슬릿 턱 영상기(SJI)의 동시 작동으로, 스펙트럼의 공간적 맥락을 실시간으로 제공합니다. 이는 SUMER에는 없었고, EIS에서도 제한적이었던 기능입니다.
   Simultaneous SG and SJI operation provides real-time spatial context for spectra — absent in SUMER and limited in EIS.

4. **5,000 K에서 10 MK까지의 온도 커버리지 / Temperature coverage from 5,000 K to 10 MK** — Mg II wing(광구), Mg II h/k(채층), C II/Si IV(TR), Fe XII(코로나), Fe XXI(플레어)까지. 단일 기기로 이렇게 넓은 온도 범위를 분광 관측하는 것은 전례가 없습니다.
   From Mg II wing (photosphere) through Mg II h/k (chromosphere), C II/Si IV (TR), Fe XII (corona), to Fe XXI (flares) — unprecedented temperature coverage in a single spectrograph.

5. **유산 설계의 영리한 재활용 / Smart reuse of heritage design** — AIA 망원경, HMI 메커니즘, SDO 비행 여유 카메라, TRACE 타임라인 도구 등 기존 미션의 부품과 소프트웨어를 광범위하게 재활용하여 Phase B 시작부터 발사까지 45개월이라는 빠른 일정과 Small Explorer 예산을 달성했습니다.
   Extensive reuse of AIA telescope, HMI mechanisms, SDO flight spare cameras, and TRACE timeline tools enabled 45-month development schedule within Small Explorer budget.

6. **수치 모델링과의 긴밀한 통합 / Tight integration with numerical modeling** — Bifrost 복사-MHD 코드와의 합성 관측 비교가 미션 설계 단계부터 포함되었습니다. IRIS 시뮬레이터를 통해 모델을 실제 관측과 동일한 조건으로 "관측"할 수 있습니다.
   Bifrost radiative-MHD synthetic observable comparison was built into the mission from design phase. The IRIS simulator allows "observing" models under identical conditions.

7. **개방적 데이터 정책 / Open data policy** — 보정된 Level 2 데이터가 관측 후 수일 내에 제한 없이 공개됩니다. SolarSoft 도구(iris_prep.pro, iris_make_fits_level3.pro, CRISPEX)가 분석을 지원합니다.
   Calibrated Level 2 data released without restriction within days. SolarSoft tools support analysis.

8. **SDO/Hinode와의 시너지 극대화 / Maximized synergy with SDO/Hinode** — IRIS의 태양 동기 궤도는 SDO, Hinode와의 동시 관측을 가능하게 하며, AIA/HMI와의 교차 보정(SJI-AIA 정렬)이 내장되어 있습니다.
   Sun-synchronous orbit enables coordinated observations with SDO and Hinode; cross-calibration with AIA/HMI built in.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 CCD 다크 전류 모델 / CCD Dark Current Model (Eq. 1)

$$D_j = P_j[T_{\text{CEB}_j}(t - \delta t_j)] + e^{(a_j + b_j T_{\text{CCD}_j})} n_x n_y t_{\text{int}} + \Delta D_j(x, n_x, n_y, t_{\text{int}})$$

| 기호 / Symbol | 의미 / Meaning |
|---|---|
| $D_j$ | 읽기 포트 $j$의 총 다크 레벨 [DN] / Total dark level in read port $j$ |
| $P_j$ | CEB 온도 $T_{\text{CEB}_j}$에 의존하는 페데스탈 레벨 (시간 지연 $\delta t_j$) / Pedestal level depending on CEB temperature (time-lagged by $\delta t_j$) |
| $e^{(a_j + b_j T_{\text{CCD}_j})}$ | CCD 온도에 지수적으로 의존하는 다크 전류율 / Dark current rate, exponential in CCD temperature |
| $n_x, n_y$ | 온칩 합산 계수 (공간, 분광) / On-chip summing factors (spatial, spectral) |
| $t_{\text{int}}$ | 적분 시간 (= 다크 전류 축적 시간) / Integration time (dark current integration time) |
| $\Delta D_j$ | 파장 방향 다크 형상 변화 보정항. $t_{\text{int}} \approx 0$일 때 평탄, 증가 시 약 bilinear / Correction for dark shape variation along wavelength; flat at $t_{\text{int}} \approx 0$, roughly bilinear as $t_{\text{int}}$ increases |

실용적 보정: 19개의 $t_{\text{int}} \approx 0$ 전프레임 다크 영상에서 기저 다크를 구성. 평균 오차: FUV < 0.12 DN, NUV/SJI < 0.08 DN.

Practical calibration: basal dark constructed from 19 full-frame darks at $t_{\text{int}} \approx 0$. Average errors: FUV < 0.12 DN, NUV/SJI < 0.08 DN.

### 4.2 Doppler 속도 / Doppler Velocity

$$v_{\text{LOS}} = c \cdot \frac{\lambda_{\text{obs}} - \lambda_0}{\lambda_0} = c \cdot \frac{\Delta\lambda}{\lambda_0}$$

IRIS 측정 정밀도: sub-pixel line centroiding으로 ~1 km s⁻¹. 3 km s⁻¹ spectral pixel에서 달성.

IRIS measurement precision: ~1 km s⁻¹ via sub-pixel line centroiding at 3 km s⁻¹ spectral pixels.

### 4.3 분광 분해능과 Nyquist 기준 / Spectral Resolution and Nyquist Criterion

$$\Delta\lambda_{\text{Nyquist}} = 2 \times \text{dispersion per pixel}$$

- FUV: $\Delta\lambda = 2 \times 12.98 = 25.96$ mÅ (측정: 25.85 mÅ FWHM) ✓
- NUV: $\Delta\lambda = 2 \times 25.46 = 50.92$ mÅ (측정: 50.54 mÅ FWHM) ✓

분광 분해능이 본질적으로 Nyquist 기준에 의해 제한됨을 확인.

Spectral resolution confirmed to be essentially Nyquist-limited.

### 4.4 LUT 압축의 신호대잡음비 / LUT Compression Signal-to-Noise

LUT 압축은 제곱근 함수 기반:

$$\text{compressed} \approx \sqrt{\text{counts}}$$

이렇게 하면 낮은 카운트는 압축하지 않고, 높은 카운트를 더 강하게 압축합니다. 제곱근 함수의 양자화 오차는 광자 카운팅 잡음에 비례하여, **일정한 S/N 비율**을 유지합니다.

Low counts are not compressed; higher counts are compressed more strongly. Quantization error from the square-root function is proportional to photon-counting noise, maintaining **constant S/N ratio**.

### 4.5 궤도 워블 위상 이동 / Orbital Wobble Phase Shift

롤 각도 $\alpha$에서의 워블은 0° 워블 곡선을 $\alpha/360°$만큼 위상 이동하여 근사:

$$\text{wobble}(\alpha, \phi) \approx \text{wobble}(0°, \phi + \alpha/360°)$$

여기서 $\phi$는 궤도 위상 (상승 노드 통과 시 0). 이 근사로 모든 롤 각도의 워블을 0°, ±90° 보정 데이터에서 유도할 수 있습니다.

Where $\phi$ is orbital phase (0 at ascending node). This approximation allows deriving wobble corrections for all roll angles from 0° and ±90° calibration data.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1962 ──── 최초의 태양 UV 로켓 관측 (Mg II, C II 선 발견)
           First solar UV rocket observations (Mg II, C II lines discovered)
  |
1973 ──── Skylab/ATM — 최초의 우주 기반 태양 UV 분광
           Skylab/ATM — first space-based solar UV spectroscopy
  |
1980 ──── SMM/UVSP — 최초의 자동화된 UV 분광 관측
           SMM/UVSP — first automated UV spectroscopic observations
  |
1995 ──── SOHO/SUMER 발사 — UV 분광의 새 시대 (2 arcsec, ~20s 주기)
           SOHO/SUMER launch — new era of UV spectroscopy
  |
1996 ──── SOHO/CDS — EUV 분광 (코로나 진단)
           SOHO/CDS — EUV spectroscopy (coronal diagnostics)
  |
1998 ──── TRACE 발사 — 고해상도 EUV 영상 (1 arcsec, 맥락 영상의 중요성)
           TRACE launch — high-res EUV imaging (context imaging importance)
  |
2006 ──── Hinode 발사 — SOT (광구 자기장) + EIS (EUV 분광, 2 arcsec)
           Hinode launch — SOT (photospheric field) + EIS (EUV spectroscopy)
  |
2006 ──── IBIS (DST) — 지상 2D 분광 (채층 Fabry-Perot)
           IBIS at DST — ground-based 2D spectroscopy (chromospheric F-P)
  |
2008 ──── CRISP (SST) — 지상 고해상도 채층 분광편광 측정
           CRISP at SST — ground-based high-res chromospheric spectropolarimetry
  |
2010 ──── SDO 발사 — AIA (코로나 전 태양면 영상) + HMI (자기장)
           SDO launch — AIA (full-disk coronal imaging) + HMI (magnetograms)
  |
2013 Jun ─ ★ IRIS 발사 — 채층/전이 영역 전용, 0.33 arcsec, 2s 주기
           ★ IRIS launch — dedicated chromosphere/TR, 0.33 arcsec, 2s cadence
  |
2014 ──── ★ 본 논문 출판 (De Pontieu et al., Solar Physics)
           ★ This paper published
  |
2020s ──── DKIST 가동 (4m 지상 망원경) — IRIS와 상보적 지상 관측
            DKIST operational (4m ground telescope) — complementary to IRIS
  |
2030s? ─── MUSE 계획 — 다중 슬릿 태양 탐사기 (IRIS 후속)
            MUSE planned — Multi-slit Solar Explorer (IRIS successor)
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Lemen et al. (2012) — AIA on SDO [#15 in SO series] | IRIS 망원경이 AIA 설계 기반; 카메라는 SDO 비행 여유분 사용; SJI-AIA 교차 보정 / IRIS telescope based on AIA design; cameras are SDO flight spares; SJI-AIA cross-calibration | ★★★ 직접 계승 / Direct heritage |
| Scherrer et al. (2012) — HMI on SDO | IRIS 메커니즘(필터 휠, 셔터, 초점)이 HMI 설계 기반 / IRIS mechanisms based on HMI designs | ★★★ 하드웨어 계승 / Hardware heritage |
| Handy et al. (1999) — TRACE [#12 in SO series] | IRIS 타임라인 도구, GT/ISS 시스템, 운용 개념이 TRACE에서 계승 / IRIS timeline tool, GT/ISS, and operations concept inherited from TRACE | ★★★ 운용 계승 / Operations heritage |
| Kosugi et al. (2007) — Hinode [#13 in SO series] | EIS 분광 유산; Hinode와의 동시 관측 조정 / EIS spectroscopy heritage; coordinated Hinode observations | ★★★ 분광 유산 / Spectroscopy heritage |
| Culhane et al. (2007) — EIS on Hinode | IRIS가 대체하고 보완하는 이전 세대 UV/EUV 분광기 / Previous-generation UV/EUV spectrograph that IRIS replements and complements | ★★☆ 선행 기기 / Predecessor |
| Wilhelm et al. (1995) — SUMER on SOHO | IRIS가 한 자릿수 이상 성능 개선한 선행 UV 분광기 / Predecessor UV spectrograph that IRIS improves by >10× | ★★☆ 선행 기기 / Predecessor |
| Pesnell, Thompson, Chamberlin (2012) — SDO [#14 in SO series] | IRIS와의 동시 관측 파트너; AIA/HMI 데이터와의 교차 보정 / Co-observation partner; cross-calibration with AIA/HMI | ★★★ 시너지 / Synergy |
| De Pontieu et al. (2007a) — Chromospheric Alfvén waves | IRIS 과학 동기의 핵심 — Alfvén 파동의 채층/코로나 전파 관측 필요성 / Core science motivation for IRIS — need to observe Alfvén wave propagation | ★★☆ 과학 동기 / Science motivation |

---

## 7. References / 참고문헌

- De Pontieu, B. et al., "The Interface Region Imaging Spectrograph (IRIS)", *Solar Physics*, 289, 2733–2779, 2014. [DOI: 10.1007/s11207-014-0485-y](https://doi.org/10.1007/s11207-014-0485-y)
- Lemen, J.R. et al., "The Atmospheric Imaging Assembly (AIA) on the Solar Dynamics Observatory (SDO)", *Solar Physics*, 275, 17, 2012. [DOI: 10.1007/s11207-011-9776-8](https://doi.org/10.1007/s11207-011-9776-8)
- Scherrer, P.H. et al., "The Helioseismic and Magnetic Imager (HMI) investigation for the Solar Dynamics Observatory (SDO)", *Solar Physics*, 275, 207, 2012. [DOI: 10.1007/s11207-011-9834-2](https://doi.org/10.1007/s11207-011-9834-2)
- Handy, B.N. et al., "The Transition Region and Coronal Explorer", *Solar Physics*, 187, 229, 1999. [DOI: 10.1023/A:1005194832212](https://doi.org/10.1023/A:1005194832212)
- Kosugi, T. et al., "The Hinode (Solar-B) Mission: An Overview", *Solar Physics*, 243, 3, 2007. [DOI: 10.1007/s11207-007-9014-6](https://doi.org/10.1007/s11207-007-9014-6)
- Culhane, J.L. et al., "The EUV Imaging Spectrometer for Hinode", *Solar Physics*, 243, 19, 2007. [DOI: 10.1007/s11207-007-0293-1](https://doi.org/10.1007/s11207-007-0293-1)
- Wilhelm, K. et al., "SUMER – solar ultraviolet measurements of emitted radiation", *Solar Physics*, 162, 189, 1995. [DOI: 10.1007/BF00733430](https://doi.org/10.1007/BF00733430)
- Pesnell, W.D., Thompson, B.J., Chamberlin, P.C., "The Solar Dynamics Observatory (SDO)", *Solar Physics*, 275, 3, 2012. [DOI: 10.1007/s11207-011-9841-3](https://doi.org/10.1007/s11207-011-9841-3)
- De Pontieu, B. et al., "Chromospheric Alfvénic waves strong enough to power the solar wind", *Science*, 318, 1574, 2007a. [DOI: 10.1126/science.1151747](https://doi.org/10.1126/science.1151747)
- Wülser, J.-P. et al., "The interface region imaging spectrograph for the IRIS Small Explorer mission", *Proc. SPIE*, 8443, 2012. [DOI: 10.1117/12.927038](https://doi.org/10.1117/12.927038)
- Podgorski, W.A. et al., "Design, performance prediction, and measurements of the interface region imaging spectrograph (IRIS) telescope", *Proc. SPIE*, 8443, 2012. [DOI: 10.1117/12.926344](https://doi.org/10.1117/12.926344)
- Leenaarts, J. et al., "The formation of IRIS diagnostics. I. A quintessential model atom of Mg II", *Astrophys. J.*, 772, 89, 2013a. [DOI: 10.1088/0004-637X/772/2/89](https://doi.org/10.1088/0004-637X/772/2/89)
- Pereira, T.M.D. et al., "The formation of IRIS diagnostics. III. Near-ultraviolet spectra and images", *Astrophys. J.*, 778, 143, 2013. [DOI: 10.1088/0004-637X/778/2/143](https://doi.org/10.1088/0004-637X/778/2/143)
- Chae, J., "Flat-fielding of solar Hα observations using relatively shifted images", *Solar Physics*, 221, 1, 2004. [DOI: 10.1023/B:SOLA.0000033352.00928.66](https://doi.org/10.1023/B:SOLA.0000033352.00928.66)
