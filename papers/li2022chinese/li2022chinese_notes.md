---
title: "The Chinese Hα Solar Explorer (CHASE) mission: An overview"
authors: [Chuan Li, Cheng Fang, Zhen Li, MingDe Ding, PengFei Chen, et al.]
year: 2022
journal: "Science China Physics, Mechanics & Astronomy"
doi: "10.1007/s11433-022-1893-3"
topic: Solar_Observation
tags: [CHASE, Xihe, Halpha, space-mission, spectroscopy, instrument-paper, China, photosphere, chromosphere]
status: completed
date_started: 2026-04-27
date_completed: 2026-04-27
---

# 60. The Chinese Hα Solar Explorer (CHASE) mission: An overview / 중국 Hα 태양 탐사선(CHASE) 미션 개요

---

## 1. Core Contribution / 핵심 기여

**한국어**
CHASE("Xihe"/羲和, 태양의 여신)는 2021년 10월 14일 발사된 **중국 최초의 태양 우주 관측 미션**이다. 본 논문은 (1) 미션의 4가지 과학 목표 — 필라멘트 형성·동역학·카이랄리티, 광구·채층 동역학, 태양–항성 비교 활동, Sun-as-a-star — (2) 과학 탑재체 **Hα Imaging Spectrograph (HIS)** 의 광학·검출기·운영 모드 사양, (3) 첫 궤도 데이터를 Level 0(원시) → Level 1(과학)으로 변환하는 **6단계 보정 파이프라인**(다크 → 슬릿 곡률 → 플랫필드 → 파장 → 강도 → 좌표), (4) 2021년 10월 24일 첫 빛(first light) 및 12월 22일 풀-Sun 도플러그램(Dopplergram) 등 첫 결과를 종합 보고한다.

핵심 기여는 **풀-Sun Hα 분광 영상의 우주 시대 개막**이다. 이전까지 풀-Sun Hα 분광 관측은 일본 SDDI(지상 0.25 Å 통과대역)와 러시아 Kislovodsk(0.16 Å 분해능) 등 지상 기반에 한정되었다. CHASE/HIS는 0.072 Å FWHM의 분광 분해능, 0.52″ 화소 공간 분해능, **풀-Sun 1분 cadence**로 동시 풀-Sun Hα(6559.7–6565.9 Å) + Fe I(6567.8–6570.6 Å) 분광 영상을 제공하며, 별도 채널로 6689 Å 광구 연속체 1 fps 영상을 동반한다. Level 1 RSM 데이터는 슬릿 4608 화소 × Hα 260 + Fe I 116 = **총 376 파장 채널**의 풀-Sun 영상 큐브 형태로 SSDC-NJU에서 공개된다. 이는 **광구–채층 결합을 우주에서 처음으로 풀-Sun 시계열로 진단**할 수 있게 만든 데이터 차원의 도약이다.

**English**
CHASE ("Xihe"/羲和, Goddess of the Sun) is **China's first solar space mission**, launched on October 14, 2021. This paper provides a comprehensive overview of (1) the mission's four scientific objectives — solar filament formation/dynamics/chirality, photosphere–chromosphere dynamics, solar–stellar comparative activity, and Sun-as-a-star observations — (2) the scientific payload **Hα Imaging Spectrograph (HIS)** including its optics, detectors, and operating modes, (3) the **six-stage calibration pipeline** (dark → slit-curvature → flat-field → wavelength → intensity → coordinate) that converts Level 0 (raw) data to Level 1 (science) products, and (4) the first on-orbit results — first-light spectra on October 24, 2021 and a full-Sun chromospheric Dopplergram on December 22, 2021.

The core contribution is **opening the space era of full-Sun Hα spectroscopic imaging**. Before CHASE, full-Sun Hα spectroscopy was confined to ground-based platforms — Japan's SDDI (0.25 Å passband filter) and Russia's Kislovodsk spectroheliograph (0.16 Å resolution). CHASE/HIS delivers 0.072 Å FWHM spectral resolution, 0.52″ per-pixel spatial sampling, and **1-min full-Sun cadence** with simultaneous Hα (6559.7–6565.9 Å) and Fe I (6567.8–6570.6 Å) windows, plus a separate 6689 Å continuum channel at 1 fps. Level 1 RSM data are published at SSDC-NJU as full-Sun image cubes of 4608 slit pixels × (260 Hα + 116 Fe I) = **376 wavelength channels**. This represents a leap in data dimensionality, enabling for the first time **full-Sun, time-resolved space-based diagnosis of photosphere–chromosphere coupling**.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction (§1) / 서론

**한국어**
Hα 6562.8 Å 선은 태양 하부 대기 진단의 가장 유력한 광학선이다. 선 중심은 **채층(chromosphere)** 정보를, 양 날개(far wings)는 **광구(photosphere)** 정보를 동시에 담아 단일 분광 프로파일로 연직 단면을 진단한다. Hale의 1900년대 Mount Wilson 분광태양사진(spectroheliograph) 이래 100년 이상 **지상 기반** Hα 관측만 존재했고, 풀-Sun Hα 분광은 일본 Hida의 **SDDI**(0.25 Å 통과대역 가변필터)와 러시아 Kislovodsk(0.16 Å 분해능) 정도였다.

지상 관측의 본질적 한계 — (1) 대기 시상(seeing), (2) 야간/악천후 관측 불가 — 를 극복하려면 우주에서 풀-Sun 분광을 수행해야 한다. CHASE는 이 빈자리를 메우는 미션이며, 위성 무게 508 kg, 크기 1210 mm × 1210 mm × 1350 mm, 평균 고도 ~517 km의 태양동기궤도(주기 ~95분), 설계 수명 3년(2021–2024, **태양 사이클 25 상승기 커버**)이다. CHASE는 (1) FY-3E의 X/EUV 영상기(2021년 7월 발사)와 (2) ASO-S(2022년 발사 예정)와 함께 중국 우주 태양 미션 군의 핵심 구성 요소이다.

**English**
The Hα 6562.8 Å line is the strongest optical diagnostic of the lower solar atmosphere. Its line center carries **chromospheric** information while the far wings carry **photospheric** information, providing a vertical cross-section in a single spectral profile. Since Hale's Mount Wilson spectroheliograph in the early 1900s, Hα observation has remained predominantly **ground-based** for over a century — only Japan's Hida **SDDI** (0.25 Å passband tunable filter) and Russia's Kislovodsk spectroheliograph (0.16 Å resolution) provided full-Sun Hα coverage.

Ground-based observation suffers two fundamental limits: (1) atmospheric seeing, (2) inability to observe day-and-night or in any weather. Performing full-Sun Hα spectroscopy from space is the only way to overcome these. CHASE addresses this gap. The satellite weighs 508 kg, measures 1210 × 1210 × 1350 mm, flies in a Sun-synchronous orbit at ~517 km altitude with a ~95-min period, and has a 3-year design lifetime (2021–2024, **covering the rise of Solar Cycle 25**). CHASE forms the core of China's solar fleet alongside (1) FY-3E's X-EUVI (launched July 2021) and (2) ASO-S (launched 2022).

### Part II: Scientific Objectives (§2) / 과학 목표

The paper organizes science into 4 sub-objectives. 본 논문은 4가지 하위 목표로 정리.

#### §2.1 Filament formation, dynamics, and chirality / 필라멘트 형성·동역학·카이랄리티

**한국어**
필라멘트(filament) — 활성 영역 자기 극성반전선(polarity inversion line) 위에 떠 있는 차가운(약 8000 K) 플라스마 — 의 형성 메커니즘, 카운터스트리밍(counterstreaming, 양방향 흐름) 시선속도, 진동(longitudinal/transverse oscillations) 진단은 코로나 자기장 추정의 핵심 단서이다. 필라멘트는 **전단된 자기 아치(sheared arcade)** 또는 **자기 다발(magnetic flux rope)** 위에 받쳐지며, 이 두 구조 모두 CME 직전 상태로 추정된다. CHASE의 풀-Sun Hα 분광은 필라멘트 토폴로지·카이랄리티(왼손/오른손 비틀림)를 직접 진단하여 자기장 구조를 역추정 가능하게 한다.

**English**
Filaments — cool (~8000 K) plasma volumes suspended above polarity inversion lines in active regions — provide critical clues to coronal magnetic structure via their formation, counterstreaming line-of-sight velocities, and longitudinal/transverse oscillations. They are supported either by **sheared magnetic arcades** or **magnetic flux ropes**, both candidates for pre-eruption CME structure. CHASE's full-Sun Hα spectroscopy directly diagnoses filament topology and chirality (left/right twist), enabling inverse inference of the magnetic configuration.

#### §2.2 Dynamics of solar activity in photosphere and chromosphere / 광구·채층 활동 동역학

**한국어**
HIS는 Hα(6562.8 Å, 채층) + Fe I(6569.2 Å, 광구) + Si I(6560.6 Å, 광구) **3개 선을 동시 관측**한다. 이를 통해:
- **백색광 플레어(white-light flare)** 의 자기 에너지 방출/수송 메커니즘 — 열적 vs. 비열적 에너지 소산
- **Ellerman bomb** — 광구–채층 경계의 미세 자기 재결합 폭발
- 태양 폭발의 **전조(precursor)와 트리거 메커니즘**
- 광구·채층의 **차등 자전(differential rotation)** 비교
- **저층 → 코로나 물질 수송(material transport)**

문제들을 통합 진단할 수 있다. 단일 Hα 관측이 **3개 다른 고도**의 정보를 동시에 제공하는 점이 핵심.

**English**
HIS observes **three lines simultaneously** — Hα (6562.8 Å, chromosphere) + Fe I (6569.2 Å, photosphere) + Si I (6560.6 Å, photosphere). This enables integrated diagnosis of:
- **White-light flares** — magnetic energy release/transport (thermal vs. non-thermal dissipation)
- **Ellerman bombs** — fine-scale magnetic reconnection at the photosphere–chromosphere boundary
- **Precursors and triggers** of solar eruptions
- **Differential rotation** comparison between photosphere and chromosphere
- **Material transport** from the lower atmosphere to the corona

The key advantage is that a single Hα observation provides simultaneous information at **three different heights**.

#### §2.3 Comparative studies of solar and stellar magnetic activities / 태양–항성 자기 활동 비교

**한국어**
별의 슈퍼플레어(superflare), 항성 CME 연구는 외계 항성을 점원(point source)으로 분광 관측하기 때문에, 태양을 같은 방식으로 점원 적분하면 직접 비교 데이터셋이 된다. 이를 **Sun-as-a-star** 관측이라 부른다. 항성 CME는 외계행성 거주 가능성, 항성 질량/각운동량 손실에 직결되지만 관측이 매우 제한적이라, CHASE의 풀-Sun 1분 cadence는 항성 시뮬레이션의 정량적 기준이 된다. Balmer 선 비대칭성(asymmetry)이 핵심 진단량.

**English**
Stellar flares — including superflares — and stellar CMEs are observed only as point-source spectra. Integrating CHASE's full-Sun Hα data over the disk produces a comparable point-source spectrum (**Sun-as-a-star**), serving as a quantitative benchmark for stellar simulations. Stellar CMEs are critical for exoplanet habitability and stellar angular-momentum loss, yet are observationally rare. Balmer line asymmetry is the key diagnostic quantity.

### Part III: Instrument Overview (§3, Tables 1–2) / 기기 개요

#### Optical design / 광학 설계

**한국어**
HIS는 무게 54.9 kg, 크기 635 × 556 × 582 mm, 평균 소비전력 58 W (피크 98 W)이다. 광학계는 3개의 하위 시스템으로 구성:
1. **Preposition optics** — 필터 어셈블리(6430–6830 Å 통과, 적외선 차단) + **비축 3거울 무수차(off-axis TMA)**. 초점거리 **1820 mm**, 유효 구경 **180 mm**, F/10.1, FOV **40′ × 40′**.
2. **Raster scanning system** — 슬릿(23 mm × 9 μm) + 접이 거울 + 평면 격자(1900 lp/mm) + 시준 거울 + 영상 거울 + CMOS 검출기. 슬릿 폭 9 μm, 격자 1900 lp/mm로 화소 분해능 **0.024 Å**, FWHM **0.072 Å** 도달.
3. **Continuum imaging system** — 빔분리기 + 중성밀도 필터(투과율 1/5000) + 협대역 필터(6689 Å, FWHM 13.4 Å) + 5120² CMOS.

연속체 통과대역 6689 Å는 **순수 연속체** 영역(흡수선이 거의 없음)이라 광구 영상에 이상적. ND 필터는 채층보다 밝은 광구 신호를 검출기 다이내믹 레인지 안에 들어오게 감쇠.

**English**
HIS weighs 54.9 kg, measures 635 × 556 × 582 mm, with average power 58 W (peak 98 W). Three optical sub-systems:
1. **Preposition optics** — filter assembly (6430–6830 Å passband, IR cut) + **off-axis TMA**. f = **1820 mm**, effective aperture **180 mm**, F/10.1, FOV **40′ × 40′**.
2. **Raster scanning system** — slit (23 mm × 9 μm) + folding mirror + plane grating (1900 lp/mm) + collimating mirror + imaging mirror + CMOS detector. Spectral resolution: **0.024 Å pixel sampling**, **0.072 Å FWHM**.
3. **Continuum imaging system** — beam splitter + neutral density filter (transmittance 1/5000) + narrow bandpass filter (6689 Å, FWHM 13.4 Å) + 5120² CMOS.

The 6689 Å continuum is **a clean window with virtually no spectral lines**, ideal for photospheric imaging. The ND filter attenuates the bright photosphere into the detector dynamic range.

#### RSM (Raster Scanning Mode) parameters / 매개변수

| Item / 항목 | Value / 값 |
|---|---|
| Detector array | 4608 (slit) × 376 (spectral) |
| Pixel size | 4.6 μm, 12-bit ADC, full-well 14.5 k e⁻ |
| Spatial sampling | 0.52″ / pixel |
| Spectral FWHM / pixel | 0.072 Å / 0.024 Å |
| Passbands | Hα 6559.7–6565.9 Å, Fe I 6567.8–6570.6 Å |
| Full-Sun scan time | ~46 s (designed 60 s with redundancy) |
| ROI scan time | 30–60 s |
| Sit-stare exposure | < 10 ms |

**RSM has 3 sub-modes**: full-Sun scanning, region-of-interest scanning, sit-stare spectroscopy. RSM은 3개 하위 모드를 가진다: 풀-Sun 스캔, 관심 영역 스캔, 슬릿 고정(sit-stare) 분광.

스캔 속도 / Scan speed: 검출기 평면에서 4.6 ± 0.3 mm/s. 풀-Sun ~46 s 도달 → 1분 cadence 설계.

#### CIM (Continuum Imaging Mode) parameters / 매개변수

| Item / 항목 | Value / 값 |
|---|---|
| Detector array | 5120 × 5120 |
| Pixel size | 4.5 μm, 10-bit ADC, full-well 12 k e⁻ |
| Spatial sampling | 0.52″ / pixel |
| Center wavelength | 6689 Å (FWHM 13.4 Å) |
| Exposure | < 5 ms |
| Frame rate | 1 fps |

CIM의 주 목적은 (1) 광구 연속체 영상 자체와 (2) **위성 자세/지향 안정성 검증** (1초 cadence로 RSM 스캔 동안 위성 운동 추적).

The CIM serves two purposes: (1) photospheric continuum imaging and (2) **monitoring satellite pointing stability** (1-s cadence tracks platform jitter during RSM scanning).

#### Telemetry and ground processing / 텔레메트리 및 지상 처리

| Item / 항목 | Value / 값 |
|---|---|
| Transmission rate | 300 Mbps |
| Daily ground capture | ~1.2 Tb/day (compressed) |
| Compression | Level 0: JPEG2000 6:1 (lossy); Level 1: Rice (lossless) |
| Ground stations | Miyun, Kashi, Sanya (China) |
| Processing center | SSDC-NJU (Solar Science Data Center, Nanjing University) |
| Compute / Storage | 102.4 Tflops, 6 PB |

### Part IV: Data Processing — RSM Calibration Flow (§4.1, Figure 4) / RSM 보정 흐름

**한국어**
2021년 10월 24일 first-light 분광이 처음 지상에 내려왔고, 이를 기준으로 보정 파이프라인이 확립되었다. Figure 4의 **5-패널 시각화**가 핵심 — Level 0 raw → Level 1 calibrated의 전 과정이 보인다:

(a) **Raw Level 0** — 슬릿 곡률(slit curvature)이 명확히 보이며, 다크 오프셋·플랫필드 패턴이 섞여 있다.
(b) **Dark field** — 위성이 차가운 우주 공간(dark cold space)을 향했을 때 측정. 검출기 다크 전류 + 디지털 오프셋.
(c) **After curvature correction** — 비축 거울과 회절 격자 때문에 슬릿이 휘어 보이는 현상을 **실험적 곡률 계수**(파장 가변 레이저로 측정)로 보정. 이 단계 후 흡수선이 수평이 됨.
(d) **Flat field** — 비네팅, 슬릿/검출기 결함, 슬릿 폭 불균일 등 시스템 강도 패턴. CHASE는 **태양 디스크 중심을 슬릿을 따라 움직이며 분광을 동시 기록**하여 플랫을 추정 (이 분광은 태양 가장자리 어두워짐(limb darkening)이나 차등 자전 정보를 포함하지 **않음**).
(e) **Calibrated Level 1** — Hα + Fe I + Si I(6560.6 Å) 흡수선이 깔끔히 나타남. Si I는 Hα 청색 날개 쪽에 위치한 추가 광구 진단선.

**6단계 보정 / Six-step calibration**:
1. Dark correction (다크/오프셋)
2. Slit-curvature correction (슬릿 곡률)
3. Flat-field correction (플랫필드)
4. Wavelength calibration (파장 가변 레이저 기반)
5. Intensity calibration (강도 정규화)
6. Coordinate transformation (검출기 좌표 → helioprojective 태양 좌표)

상세 절차는 동반 논문 Qiu et al. (2022) [ref 41]에 정의됨.

**English**
First-light spectra on October 24, 2021 anchored the calibration pipeline. Figure 4's **5-panel visualization** shows the entire Level 0 → Level 1 flow:

(a) **Raw Level 0** — slit curvature is visible; dark offsets and flat-field patterns are mixed in.
(b) **Dark field** — measured when CHASE/HIS pointed at cold space. Dark current + digital offset.
(c) **After curvature correction** — slit curvature (caused by off-axis mirrors and grating) corrected via **experimental curvature coefficients** measured with a wavelength-tunable laser. Absorption lines become horizontal.
(d) **Flat field** — captures vignetting, slit/detector artifacts, slit-width nonuniformity. CHASE estimates flat by **moving the disk center along the slit while recording spectra simultaneously** — these spectra do **not** contain limb-darkening or differential rotation information.
(e) **Calibrated Level 1** — clean Hα + Fe I + Si I (6560.6 Å) absorption lines. Si I is an additional photospheric diagnostic in the Hα blue wing.

Six-step calibration: dark → slit-curvature → flat → wavelength → intensity → coordinate. Details in companion paper Qiu et al. (2022) [ref 41].

### Part V: Level 1 Data Products & Higher-Level Products (§4.1–4.2) / Level 1 산출물

**한국어**
**Level 1 RSM 데이터의 실제 차원**:
- Hα 윈도우: **4608 × 260** 배열 (slit pixels × wavelength steps for 6559.7–6565.9 Å)
- Fe I 윈도우: **4608 × 116** 배열 (6567.8–6570.6 Å)
- 한 스캔 시퀀스 = **376개 솔라 영상** (260 Hα + 116 Fe I, 각 파장에 대한 풀-Sun 영상)
- 사용자에게는 일반적으로 **3차원 배열**(slit × scan-step × wavelength) 형태로 제공
- Rice 무손실 압축으로 아카이브
- 배포: **SSDC-NJU** (https://ssdc.nju.edu.cn), IDL + Python 읽기 routine 제공

**Higher-level products** (사용자 후처리 권장):
- **풀-Sun 도플러그램(Dopplergram)**: Hα/Fe I/Si I 프로파일을 디스크 중심 평균 프로파일과 **상호상관(cross-correlation)** 비교 → 시선속도 정확도 **~0.06 km/s**, 1분 미만에 산출 가능
- 강도 보정(emission measure) 기반 발광 강도
- 채층 차등 자전 (극→적도)

CIM Level 1: dark + flat-field 보정. Flat-field는 **KLL 방법(Kuhn, Lin, Loranz 1991, [42])** 사용 — 검출기에서 디스크 중심을 여러 위치로 옮겨가며 영상을 기록한 후 반복 알고리즘으로 플랫 도출.

**English**
**Actual Level 1 RSM data dimensions**:
- Hα window: **4608 × 260** array (slit pixels × wavelength steps over 6559.7–6565.9 Å)
- Fe I window: **4608 × 116** array (6567.8–6570.6 Å)
- One scan sequence = **376 solar images** at distinct wavelengths
- Users typically receive **3-D arrays** (slit × scan-step × wavelength)
- Archived with Rice lossless compression
- Distribution: **SSDC-NJU** (https://ssdc.nju.edu.cn) with IDL + Python reading routines

**Higher-level products** (user-derived):
- **Full-Sun Dopplergram** — cross-correlate each pixel's Hα/Fe I/Si I profile against a disk-center average reference; velocity accuracy **~0.06 km/s**, computable in < 1 min
- Emission measure / intensity-calibrated radiance
- Chromospheric differential rotation (poles to equator)

CIM Level 1: dark + flat-field correction. Flat field uses **KLL method (Kuhn, Lin, Loranz 1991, [42])** — multiple disk-centered images at different detector positions are iteratively combined.

### Part VI: First Results (§4.1, Figures 5–6) / 첫 결과

**한국어**
- **Figure 5(a)**: 2021년 12월 22일 06:01:05–06:01:52 UT 풀-Sun Hα 6562.8 Å 분광 영상. 활성 영역, 필라멘트, 플라쥬(plage)가 깔끔하게 보인다.
- **Figure 5(b)**: 디스크 중심 평균 Hα 프로파일. 선 깊이가 매우 깊고(~0.2× continuum), Si I 6560.6 Å 흡수선이 청색 날개에 명확히 위치. 강도 단위는 erg s⁻¹ cm⁻² Å⁻¹ sr⁻¹.
- **Figure 6**: 동일 시각의 풀-Sun 채층 도플러그램. **태양 차등 자전**이 동서 방향(red shift / blue shift)으로 명확히 가시화 — 적도에서 ~3 km/s 방향성, 극지에서 0에 가까움. 활성 영역의 복잡한 속도장(국소 흐름, 필라멘트 진동 등)도 함께 드러남.

이 결과들은 **위성 플랫폼의 지향 정확도 5×10⁻⁴°, 안정도 5×10⁻⁵°/s**(동반 논문 Zhang et al. [14])라는 사양이 실제로 분광 관측에 충분함을 입증.

**English**
- **Figure 5(a)**: full-Sun Hα 6562.8 Å spectroscopic image, December 22, 2021, 06:01:05–06:01:52 UT. Active regions, filaments, and plage are clearly visible.
- **Figure 5(b)**: disk-center averaged Hα profile. Deep absorption (~0.2× continuum), with the Si I 6560.6 Å line in the blue wing. Intensity in erg s⁻¹ cm⁻² Å⁻¹ sr⁻¹.
- **Figure 6**: full-Sun chromospheric Dopplergram at the same epoch. **Solar differential rotation** appears clearly as a global red/blue shift gradient (~3 km/s magnitude at the equator, near zero at poles). Complex velocity fields in active regions (local flows, filament oscillations) are also resolved.

These results validate the satellite platform specs — pointing accuracy 5×10⁻⁴°, stability 5×10⁻⁵°/s (companion paper Zhang et al. [14]) — as sufficient for spectroscopic observation.

---

## 3. Key Takeaways / 핵심 시사점

1. **CHASE는 풀-Sun Hα 분광의 우주 시대 개막 / CHASE opens the space era of full-Sun Hα spectroscopy** —
   *한국어*: 100년 넘게 지상에만 의존하던 풀-Sun Hα 분광을 우주에서 전천 1분 cadence로 수행. SDDI(지상)·Kislovodsk(지상) 외엔 사실상 첫 번째 풀-Sun Hα 우주 미션.
   *English*: Performs full-Sun Hα spectroscopy from space at 1-min cadence, ending century-long dependence on ground-based observation. The first dedicated space platform for full-Sun Hα aside from ground-based SDDI and Kislovodsk.

2. **단일 관측이 3개 고도(광구·하부 채층·채층) 동시 진단 / Single observation diagnoses three heights simultaneously** —
   *한국어*: Hα 6562.8 Å(채층) + Fe I 6569.2 Å(광구) + Si I 6560.6 Å(광구) 흡수선이 한 분광 프로파일에 함께 들어와 광구–채층 결합을 직접 추적.
   *English*: Hα (chromosphere) + Fe I (photosphere) + Si I (photosphere) appear in a single spectral profile, enabling direct tracking of photosphere–chromosphere coupling.

3. **두 모드 동시 운영으로 시너지 / Dual-mode synergy** —
   *한국어*: RSM(분광 영상, 1분)과 CIM(연속체 영상, 1초)이 동시에 운영. CIM은 RSM 스캔 중 위성 자세 안정성을 추적하여 RSM의 좌표 보정에 직접 기여.
   *English*: RSM (spectroscopic, 1-min) and CIM (continuum, 1-s) operate simultaneously. CIM monitors platform stability during RSM scans and feeds back into RSM coordinate correction.

4. **Level 1 데이터 차원: 4608 × 376 × N_scan + 5120² 광구 / Level 1 data dimensionality** —
   *한국어*: RSM 한 스캔이 **376 wavelength channels × 4608 slit pixels × 스캔 단계** 큐브를 생성. 우주에서 처음으로 등장한 데이터 타입이며, ML/패턴 인식 기반 새로운 분석 패러다임의 출발점.
   *English*: One RSM scan yields a cube of **376 wavelength channels × 4608 slit pixels × scan steps** — a data type new to space astronomy, opening ML/pattern-recognition analysis paradigms.

5. **6단계 표준화된 보정 파이프라인 / Standardized 6-step calibration pipeline** —
   *한국어*: dark → slit-curvature → flat → wavelength → intensity → coordinate. 슬릿 곡률 보정에 파장 가변 레이저, 플랫필드에 디스크 이동 기법(중심 위치 이동) 채용. 시선속도 정확도 ~0.06 km/s 도달.
   *English*: dark → slit-curvature → flat → wavelength → intensity → coordinate. Slit curvature corrected via tunable-laser-derived coefficients; flat-field via disk-center motion technique. Achieves ~0.06 km/s LOS velocity accuracy.

6. **Sun-as-a-star 기준 데이터셋 제공 / Provides Sun-as-a-star benchmark** —
   *한국어*: 풀-Sun Hα를 디스크 적분하여 항성 관측과 직접 비교 가능한 점원 분광을 1분마다 생성. Kepler/TESS 시대 슈퍼플레어·항성 CME 연구의 정량적 기준.
   *English*: Disk-integrated Hα profiles, produced every minute, are directly comparable to stellar spectra — a quantitative benchmark for superflare and stellar-CME research in the Kepler/TESS era.

7. **Solar Cycle 25 상승기·최대기 풍부한 사건 커버 / Rich coverage of Cycle 25 rise and maximum** —
   *한국어*: 2021–2024 임무 기간이 사이클 25 최대(2025 추정)와 정확히 겹쳐 플레어·CME·필라멘트 이벤트가 풍부. 데이터 가치 극대화.
   *English*: 2021–2024 mission overlaps Cycle 25 maximum (~2025), maximizing flare/CME/filament event yield.

8. **다국적 다중 미션 동기 관측의 핵심 노드 / Key node in multi-mission synergy** —
   *한국어*: SDO(HMI/AIA) + IRIS + Hinode + Solar Orbiter와 협업해 광구 자기장 + EUV 코로나 + Hα 채층의 3차원 진단 체인 완성. 향후 ASO-S와 더 강한 연계.
   *English*: With SDO (HMI/AIA), IRIS, Hinode, and Solar Orbiter, completes a 3-D diagnostic chain — photospheric magnetic field + EUV corona + Hα chromosphere. Stronger synergy expected with ASO-S.

---

## 4. Mathematical Summary / 수학적 요약

CHASE 논문은 기기 paper이므로 새로운 이론 수식은 적지만, 기기 사양과 보정에 사용된 핵심 관계식을 정리한다.
The CHASE paper is an instrument paper with few new theoretical equations; below are the key relations behind its specs and calibration.

### (1) Spectral resolution from slit width / 슬릿 폭에서 분광 분해능

$$
\Delta\lambda_{\text{FWHM}} \approx \frac{\lambda \, d_{\text{slit}}}{f \, N \, m}
$$

- $\lambda$: observed wavelength (~6562.8 Å)
- $d_{\text{slit}}$: slit width (9 μm)
- $f$: focal length (1820 mm)
- $N$: grating groove density (1900 lp/mm = 1.9 × 10⁶ lp/m)
- $m$: diffraction order

**Worked example**: $\Delta\lambda \approx \frac{6562.8 \times 10^{-10} \times 9 \times 10^{-6}}{1.820 \times 1.9 \times 10^6 \times m}$. For $m=1$, this yields ~1.7 × 10⁻¹¹ m ≈ 0.17 Å (instrument resolution dominated by other factors); CHASE achieves measured FWHM **0.072 Å** with 0.024 Å pixel sampling, indicating ~3× Nyquist oversampling.

### (2) Pixel spatial sampling / 화소 공간 샘플링

$$
\theta_{\text{pix}} = \frac{p_{\text{pix}}}{f} \cdot \frac{180 \cdot 3600}{\pi} \quad [\text{arcsec}]
$$

- RSM: $p_{\text{pix}} = 4.6\,\mu\text{m}$, $f = 1820$ mm → $\theta_{\text{pix}} \approx 0.521''$ ✓
- CIM: $p_{\text{pix}} = 4.5\,\mu\text{m}$, $f = 1820$ mm → $\theta_{\text{pix}} \approx 0.510''$ ≈ 0.52″ ✓

Both modes match the 0.52″ value quoted in Table 2. 두 모드 모두 표 2의 0.52″ 값과 일치.

### (3) Full-Sun raster timing / 풀-Sun 래스터 시간

$$
T_{\text{full-Sun}} = \frac{D_{\odot}}{v_{\text{scan, plane}}} = \frac{D_{\odot}^{\text{ang}}}{\dot\theta_{\text{scan}}}
$$

Apparent solar diameter $D_{\odot}^{\text{ang}} \approx 32' = 1920''$. With per-pixel scan rate that yields total ~46 s for 1920″, design cadence is 60 s with redundancy. ROI scan: 30–60 s.

### (4) Doppler velocity from line shift / 도플러 시선속도

$$
v_{\text{LOS}} = c \cdot \frac{\Delta\lambda}{\lambda_0}, \qquad c = 2.998 \times 10^5 \text{ km/s}
$$

- One-pixel shift sensitivity (Hα): $v_{\text{LOS}}^{\text{1px}} = c \cdot \frac{0.024}{6562.8} \approx 1.10$ km/s
- CHASE-reported velocity accuracy via cross-correlation: **~0.06 km/s**
- Implies sub-pixel centroid precision of ~0.06/1.10 ≈ **0.054 pixel** (consistent with typical line-fit precision of ~σ/SNR with high-SNR profiles)

### (5) Cross-correlation Doppler retrieval / 상호상관 도플러 산출

For a pixel profile $I_p(\lambda)$ and reference disk-center average $I_{\text{ref}}(\lambda)$, the lag $\Delta\lambda^*$ that maximizes the cross-correlation:

$$
\Delta\lambda^* = \arg\max_{\Delta\lambda} \int I_p(\lambda) \, I_{\text{ref}}(\lambda - \Delta\lambda) \, d\lambda
$$

is converted to $v_{\text{LOS}}$ via Eq. (4). This is the production algorithm CHASE uses for full-Sun Dopplergrams (Figure 6).

### (6) Calibration model / 보정 모델

A simplified pipeline at the pixel level:

$$
I_{\text{L1}}(x, \lambda) = \frac{I_{\text{raw}}(x, \lambda) - D(x, \lambda)}{F(x, \lambda)} \cdot \mathcal{G}(\lambda) \cdot \mathcal{C}_{\text{curv}}\bigl[\, \cdot \,\bigr] \cdot \mathcal{T}_{\text{coord}}\bigl[\, \cdot \,\bigr]
$$

- $D$: dark frame (Step 1)
- $F$: flat field via disk-motion technique (Step 3)
- $\mathcal{C}_{\text{curv}}$: slit-curvature warp from tunable-laser coefficients (Step 2)
- $\mathcal{G}$: wavelength + intensity calibration (Steps 4–5)
- $\mathcal{T}_{\text{coord}}$: detector → helioprojective coordinate transform (Step 6)

### (7) Data volume / 데이터 부피

$$
V_{\text{raster}} = N_{\text{slit}} \cdot N_{\text{scan}} \cdot N_{\text{spec}} \cdot \frac{b}{8}
$$

- RSM full-Sun: $N_{\text{slit}} = 4608$, $N_{\text{spec}} = 376$, $N_{\text{scan}} \approx N_{\text{slit}}$ (~3700 scan steps for 1920″ at 0.52″ stride), $b = 12$ bit
- $V \approx 4608 \times 3700 \times 376 \times 1.5\,\text{B} \approx 9.6 \times 10^9$ B ≈ **9.6 GB**; paper quotes ~14.9 GB, suggesting Nyquist-oversampled scanning or larger $N_{\text{scan}}$ count
- After 6:1 JPEG2000 compression (Level 0): ~2.5 GB; daily ground capture ~1.2 Tb/day

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1908  ─── Hale: Mount Wilson spectroheliograph (first Hα spectroheliograms)
   │
1980s─── Hida observatory full-disk Hα monitoring (ground)
   │
1985  ─── Schroeter: vacuum tower telescopes (ground Hα tradition)
   │
1995  ─── SOHO launched (EUV/UV; no Hα spectroscopy)
   │
1998  ─── Kislovodsk spectroheliograph (0.16 Å, ground)
   │
2006  ─── Hinode launched (SOT/EIS — narrow Hα filtergrams, not full-Sun spectra)
   │
2010  ─── SDO launched (HMI continuum + AIA EUV; no Hα)
   │
2013  ─── IRIS launched (UV 1300–2800 Å; no Hα)
   │
2016  ─── SDDI installed at Hida (Japan, 0.25 Å passband, ground full-Sun Hα Doppler)
   │
2020  ─── Solar Orbiter launched (in-situ + remote, no Hα)
   │
2021  ─── ★ CHASE launched 14 Oct (China's first solar space mission, full-Sun Hα)
   │       FY-3E launched 5 Jul (X-EUVI, Chinese fleet sibling)
   │       CHASE first-light spectra: 24 Oct 2021
   │
2022  ─── ★ Li et al. — CHASE mission overview (THIS PAPER)
   │       Qiu et al. — CHASE/HIS calibration (companion paper)
   │       Liu et al. — HIS detailed design (companion paper)
   │       ASO-S launched (Chinese Coordinated Solar fleet expanded)
   │
2024  ─── CHASE design lifetime ends, mission operating in extended phase
   │
2025  ─── Solar Cycle 25 maximum (CHASE captures rich activity)
   │
202?  ─── Future: SUNDIAL and follow-on Chinese solar missions
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Qiu et al. 2022, *Sci. China P.M.A.* **65**, 289603 (ref [41]) | Companion paper detailing CHASE/HIS calibration procedures (dark, flat, slit-curvature, wavelength, intensity, coordinate) | **Critical**: must be read for any quantitative CHASE data analysis. Defines Level 1 data product. |
| Liu et al. 2022, *Sci. China P.M.A.* **65**, 289604 (ref [39]) | Companion paper on HIS instrument detailed design (optics, mechanics, electronics) | **High**: hardware reference; complements §3 of this paper. |
| Zhang et al. 2022, *Sci. China P.M.A.* **65**, 289602 (ref [14]) | Companion paper on satellite platform pointing/stability (5×10⁻⁴° / 5×10⁻⁵° s⁻¹) | **High**: explains why HIS achieves spectroscopic-grade observations without a dedicated guide telescope. |
| Fang et al. 2019, *Sci. Sin. P.M.A.* **49**, 059603 (ref [5]) | Earlier conceptual review of CHASE — pre-launch mission design | **Medium**: historical context; the pre-launch vision now realized. |
| Pesnell et al. 2012, *Sol. Phys.* **275**, 3 (ref [25]) | SDO mission overview | **High**: SDO/HMI/AIA = primary co-observation partner; CHASE Hα + SDO HMI magnetograms = standard joint diagnostic. |
| De Pontieu et al. 2014, *Sol. Phys.* **289**, 2733 (ref [32]) | IRIS mission overview | **High**: complementary UV spectroscopy. CHASE Hα + IRIS Mg II + Si IV = full chromosphere + transition region. |
| Müller et al. 2020, *A&A* **642**, A1 (ref [33]) | Solar Orbiter mission overview | **Medium**: in-situ + remote synergy; CHASE provides Sun-side context for Solar Orbiter's plasma and magnetic measurements. |
| Kuhn, Lin, & Loranz 1991, *PASP* **103**, 1097 (ref [42]) | KLL flat-field algorithm used by CHASE CIM | **Specific**: the standard solar flat-field method; required reading for CIM calibration understanding. |
| Sakai & Ichimoto et al. 2017, *Sol. Phys.* **292**, 63 (ref [8]) | SDDI at Hida — closest analog instrument | **High**: ground-based full-Sun Hα Doppler comparator; CHASE's primary scientific competitor/complement. |
| Gan et al. 2022, *Res. Astron. Astrophys.* (ref [12]) | ASO-S mission overview | **Medium**: sister Chinese mission; coordinated science campaigns. |

---

## 7. References / 참고문헌

### Primary paper / 본 논문

- C. Li, C. Fang, Z. Li, M. D. Ding, P. F. Chen, et al., "The Chinese Hα Solar Explorer (CHASE) mission: An overview," *Sci. China Phys. Mech. Astron.* **65**, 289602 (2022). [DOI: 10.1007/s11433-022-1893-3](https://doi.org/10.1007/s11433-022-1893-3) | [arXiv: 2205.05962](https://arxiv.org/abs/2205.05962)

### Companion CHASE papers / CHASE 동반 논문

- Y. Qiu, S. H. Rao, C. Li, et al., "Calibration procedures for the CHASE/HIS science data," *Sci. China Phys. Mech. Astron.* **65**, 289603 (2022). [arXiv: 2205.06075](https://arxiv.org/abs/2205.06075)
- Q. Liu, et al., HIS instrument detailed design, *Sci. China Phys. Mech. Astron.* **65**, 289604 (2022).
- W. Zhang, et al., CHASE platform pointing/stability, *Sci. China Phys. Mech. Astron.* **65**, 289605 (2022).

### Predecessor / Comparator instruments / 선행/비교 기기

- G. E. Hale, *Astrophys. J.* **27**, 219 (1908) — first spectroheliograph.
- C. Fang, B. Gu, Y. Yuan, et al., "CHASE — a complementary space mission to ASO-S," *Sci. Sin. P.M.A.* **49**, 059603 (2019) — pre-launch concept.
- K. Ichimoto, T. T. Ishii, K. Otsuji, et al., "SDDI installation at Hida," *Sol. Phys.* **292**, 63 (2017). [arXiv: 1612.01054](https://arxiv.org/abs/1612.01054)
- I. A. Berezin, G. Tlatov, N. N. Skorbezh, *Geomag. Aeron.* **61**, 1075 (2021) — Kislovodsk spectroheliograph.

### Co-observing missions / 협업 미션

- W. D. Pesnell, B. J. Thompson, P. C. Chamberlin, "The Solar Dynamics Observatory (SDO)," *Sol. Phys.* **275**, 3 (2012).
- B. De Pontieu, A. M. Title, J. R. Lemen, et al., "The Interface Region Imaging Spectrograph (IRIS)," *Sol. Phys.* **289**, 2733 (2014). [arXiv: 1401.2491](https://arxiv.org/abs/1401.2491)
- D. Müller, O. C. St. Cyr, I. Zouganelis, et al., "The Solar Orbiter mission," *A&A* **642**, A1 (2020). [arXiv: 2009.00861](https://arxiv.org/abs/2009.00861)
- W. Q. Gan, et al., "ASO-S overview," *Res. Astron. Astrophys.* (2022).

### Calibration / Methods / 보정 방법

- J. R. Kuhn, H. Lin, D. Loranz, "Gain calibrating nonuniform area-array detectors with the solar limb (KLL method)," *PASP* **103**, 1097 (1991).

### Data access / 데이터 접근

- SSDC-NJU CHASE Data Portal: https://ssdc.nju.edu.cn (FITS Level 1, IDL + Python read routines)
- CNSA Service Rules for the CHASE Satellite Data: https://www.cnsa.gov.cn/english/n6465645/n6465648/c10373923/content.html
