---
title: "EUVI: The STEREO-SECCHI Extreme Ultraviolet Imager"
authors: "Jean-Pierre Wuelser, James R. Lemen et al."
year: 2004
journal: "Proc. SPIE 5171, Telescopes and Instrumentation for Solar Astrophysics, pp. 111–122"
doi: "10.1117/12.506877"
topic: Solar_Observation
tags: [EUVI, STEREO, SECCHI, EUV, stereoscopy, normal-incidence, multilayer, Ritchey-Chrétien, CCD, corona]
status: completed
date_started: 2026-04-17
date_completed: 2026-04-17
---

# 17. EUVI: The STEREO-SECCHI Extreme Ultraviolet Imager / STEREO-SECCHI 극자외선 촬상기

---

## 1. Core Contribution / 핵심 기여

EUVI(Extreme Ultraviolet Imager)는 NASA STEREO 미션의 SECCHI 장비 패키지에 탑재된 EUV 망원경으로, SOHO/EIT의 설계를 계승하면서 공간 분해능, 촬영 주기, 감도를 대폭 향상시켰다. EUVI는 Ritchey-Chrétien 광학 설계를 채택하여 4개 EUV 파장 밴드(17.1, 19.5, 28.4, 30.4 nm)에서 1.6 arcsec 픽셀 스케일로 태양 전면(±1.7 태양 반경)을 촬영한다. 두 대의 동일한 STEREO 우주선에 각각 탑재되어 **사상 최초의 태양 코로나 입체 관측(stereoscopic observation)**을 실현했으며, 이를 통해 CME의 3차원 구조와 활동 영역의 입체적 형태를 직접 관측할 수 있게 되었다. 본 논문은 EUVI의 광학 설계, 다층 코팅, 필터, 검출기, 기계 구조, 보정 결과, 그리고 관측 프로그램을 상세히 기술하는 pre-launch instrument paper이다.

EUVI (Extreme Ultraviolet Imager) is an EUV telescope aboard the SECCHI instrument suite of NASA's STEREO mission. Inheriting the proven EIT design while substantially improving spatial resolution, cadence, and sensitivity, EUVI adopts a Ritchey-Chrétien optical design to image the full solar disk (±1.7 solar radii) in four EUV bandpasses (17.1, 19.5, 28.4, 30.4 nm) at 1.6 arcsec pixel scale. Mounted on each of two identical STEREO spacecraft, EUVI enabled **the first-ever stereoscopic observations of the solar corona**, allowing direct observation of the 3D structure of CMEs and active regions. This pre-launch instrument paper provides a detailed description of the optical design, multilayer coatings, filters, detector, mechanical structure, calibration results, and observing programs.

---

## 2. Reading Notes / 읽기 노트

### Section 1: Introduction / 서론 (pp. 111)

EUVI는 SECCHI(Sun Earth Connection Coronal and Heliospheric Investigation) 장비의 핵심 구성 요소로, SECCHI의 과학 목표를 달성하는 데 중요한 역할을 한다. 주요 과학 목표는 세 가지이다:

EUVI is a core component of the SECCHI instrument suite, playing a critical role in achieving SECCHI's science objectives. The three main science objectives are:

1. **CME 발생 메커니즘 연구**: 플럭스 시스템의 상호작용, 재결합(reconnection)의 역할, coronal dimming 현상
   Investigation of CME initiation: flux system interactions, role of reconnection, coronal dimming
2. **CME의 물리적 진화**: 3차원 구조, 가속 메커니즘, 저층 코로나의 반응
   Physical evolution of CMEs: 3D structure, acceleration mechanisms, low corona response
3. **활동 영역의 3차원 구조**: 입체 관측을 통한 직접적인 3D 형태 파악
   3D structure of Active Regions: direct 3D morphology via stereoscopic observation

STEREO 미션은 두 대의 동일한 우주선으로 구성되며, 태양 중심 궤도(heliocentric orbit)에서 지구로부터 매년 약 22도씩 반대 방향으로 벌어진다. SECCHI 장비는 NRL(Naval Research Lab)이 총괄하고, EUVI 망원경은 LMSAL(Lockheed Martin Solar and Astrophysics Lab)에서 개발했다. 거울은 프랑스 IOTA(Institut d'Optique)에서 제작·코팅하고, IAS(Institut d'Astrophysique Spatiale)에서 보정했다. 초점면 어셈블리는 NRL, 카메라 전자장치는 Rutherford Appleton Lab, 조리개 도어는 MPAe(Max-Planck Institut für Aeronomie)가 각각 담당했다.

The STEREO mission consists of two identical spacecraft in heliocentric orbit, drifting away from Earth at ~22 degrees per year in opposite directions. The SECCHI investigation is led by NRL; the EUVI telescope is developed at LMSAL. Mirrors are figured and coated at IOTA (France) and calibrated at IAS. The focal plane assembly is from NRL, camera electronics from Rutherford Appleton Lab, and the aperture door from MPAe.

SECCHI는 5개의 망원경으로 구성되어 태양 표면에서 지구까지의 넓은 시야를 커버한다. EUVI는 이 중 가장 안쪽 영역(태양 색구에서 내부 코로나 ±1.7 태양 반경)을 담당한다.

SECCHI comprises five telescopes covering a broad range of fields of view from the solar surface to Earth. EUVI covers the innermost portion: from the solar chromosphere to the inner corona at ±1.7 solar radii.

### Section 2: EUVI Telescope Overview / EUVI 망원경 개요 (pp. 112–113)

EUVI는 소형 normal-incidence 망원경으로, 다음과 같은 광학 경로를 따른다:

EUVI is a small normal-incidence telescope with the following optical path:

1. **입사 필터(Entrance Filter)**: 150 nm 두께의 알루미늄 박막 필터가 UV, 가시광, IR을 차단하고 태양열을 제거
   150 nm thick aluminum thin-film filter suppresses UV, visible, and IR radiation and rejects solar heat
2. **조리개 선택기(Quadrant Selector)**: 4개 광학 사분면 중 하나를 선택
   Selects one of four optical quadrants
3. **주경 및 부경(Primary & Secondary Mirror)**: 각 사분면이 서로 다른 파장에 최적화된 multilayer 코팅
   Each quadrant coated with multilayer optimized for a different wavelength
4. **필터 휠(Filter Wheel)**: 초점면 근처에 위치; 잔여 가시광/IR을 제거하는 이중 알루미늄 필터
   Located near focal plane; redundant aluminum filters remove remaining visible/IR
5. **셔터(Shutter)**: 회전 블레이드 방식, 노출 시간 60 ms ~ 60 s 이상
   Rotating blade type, exposure times from 60 ms to over 60 s
6. **CCD 검출기**: e2v CCD42-40, 2048 × 2048 픽셀, back-thinned
   e2v CCD42-40, 2048 × 2048 pixels, back-thinned

**Table 1 — EUVI 주요 사양 요약 / Key Specifications Summary:**

| 파라미터 / Parameter | 값 / Value |
|---|---|
| 망원경 형식 / Instrument type | Normal incidence EUV, Ritchey-Chrétien |
| 파장 밴드 / Bandpasses | He II 30.4 nm, Fe IX 17.1 nm, Fe XII 19.5 nm, Fe XV 28.4 nm |
| 조리개(구경) / Aperture | 98 mm (at primary mirror) |
| 유효 초점거리 / Effective focal length | 1750 mm |
| 시야(FOV) / Field of View | ±1.7 solar radii (circular full sun) |
| 픽셀 스케일 / Spatial scale | 1.6 arcsec/pixel |
| 검출기 / Detector | e2v CCD42-40, 2048 × 2048 pixels, back-illuminated |
| 영상 안정화 / Image stabilization | Active secondary mirror (tip/tilt) |

### Section 3.1: Optical Design / 광학 설계 (pp. 113–114)

EUVI는 **Ritchey-Chrétien 광학계**를 채택했다. 이는 쌍곡면(hyperboloidal) 주경과 부경으로 구성되어 넓은 시야에서 coma와 spherical aberration을 동시에 보정하는 설계이다. 부경 배율(secondary mirror magnification)은 2.42로, 이 낮은 배율은 두 가지 장점을 제공한다:

EUVI adopts a **Ritchey-Chrétien optical design**, consisting of hyperboloidal primary and secondary mirrors that simultaneously correct coma and spherical aberration over a wide field. The secondary mirror magnification is 2.42, and this low magnification provides two advantages:

- 전체 시야에 걸쳐 pixel-limited resolution 달성 (광학 수차가 픽셀보다 작음)
  Pixel-limited resolution across the entire field of view (optical aberrations smaller than pixel)
- 거울 간격 변화에 대한 민감도 감소 → 초점 조절 메커니즘 불필요
  Reduced sensitivity to mirror separation changes → no focus mechanism needed

**Table 2 — 광학 파라미터 / Optical Parameters:**

| 파라미터 / Parameter | System | Primary mirror | Secondary mirror |
|---|---|---|---|
| 유효 초점거리 / Effective focal length | 1750 mm | — | — |
| 거울 간격 / Mirror separation | 460 mm | — | — |
| 부경-초점 거리 / Secondary–focus distance | 635 mm | — | — |
| 조리개 마스크 외경 / Aperture mask outer diameter | 98 mm | — | — |
| 중심 차폐 내경 / Central obscuration diameter | 65 mm | — | — |
| 곡률 반경 / Radius of curvature | — | 1444 mm (concave) | 892 mm (convex) |
| 원추 상수 / Conic constant | — | −1.194 | −8.42 |
| 거울 직경 / Mirror diameter | — | 105 mm | 48 mm |

전체 망원경 길이는 800.93 mm이다. 바플(baffle)이 전방 조리개에서 CCD까지 대전 입자의 침입을 방지한다. 망원경 동공(pupil)은 주경 바로 앞의 조리개 마스크에 의해 정의되며, 비네팅 없는(unvignetted) 시야는 ±1.7 태양 반경이다.

The total telescope length is 800.93 mm. Baffles prevent charged particles from entering from the front aperture to the CCD. The telescope pupil is defined by an aperture mask just in front of the primary mirror, and the unvignetted field of view extends to ±1.7 solar radii.

Ray trace 결과(Figure 3)에 따르면, 축상(on-axis)과 시야 가장자리(27 arcmin) 모두에서 광학 수차가 매우 작다. 초점 위치를 ±150 μm 이동해도 성능 저하가 최소화되며, 공칭 초점 위치는 시야 전체에 걸쳐 수차를 최소화하도록 선택된다.

Ray trace results (Figure 3) show very small optical aberrations both on-axis and at the edge of field (27 arcmin). Even with ±150 μm focus shifts, performance degradation is minimal; the nominal focus location is chosen to minimize aberrations across the field.

### Section 3.2: Mirrors / 거울 (pp. 114)

EUVI 거울은 EIT 거울을 제작한 IOTA에서 동일한 방식으로 제작된다:

EUVI mirrors are fabricated at IOTA using the same process as the EIT mirrors:

1. **Zerodur 기판**을 구면으로 연마 후 초정밀 연마(superpolish)
   Zerodur substrates are polished to a sphere, then superpolished
2. **이온빔 에칭(ion beam etching)**으로 비구면화(aspherizing) — 초정밀 연마 표면 품질을 유지
   Aspherized using ion beam etching, preserving the superpolished surface quality
3. 각 사분면에 해당 파장에 최적화된 **MoSi multilayer 코팅**을 증착
   Each quadrant coated with MoSi multilayer optimized for the corresponding wavelength

**Table 3 — 다층 코팅 특성 / Multilayer Coating Properties:**

| 속성 / Property | 17.1 nm | 19.5 nm | 28.4 nm | 30.4 nm |
|---|---|---|---|---|
| 주요 방출선 / Emission lines | Fe IX, Fe X | Fe XII, Fe XXIV | Fe XV | He II |
| 중심 파장 / Center wavelength | 17.2 nm | 19.4 nm | 28.4 nm | 30.4 nm |
| 대역폭(FWHM) / Bandwidth | 1.4 nm | 1.6 nm | 1.9 nm | 3.0 nm |
| 최대 반사율(단일 반사) / Peak reflectivity (single) | 39% | 35% | 15% | 23% |
| 코팅 재료 / Coating material | MoSi | MoSi | MoSi, var. spacing | MoSi |

28.4 nm 채널의 코팅은 **가변 간격(variable layer spacing)**을 사용한다. 이는 인접한 강한 He II 30.4 nm 방출선을 최적으로 억제하기 위한 설계이다. 이로 인해 28.4 nm의 최대 반사율은 15%로 다른 채널보다 낮다.

The 28.4 nm channel coating uses **variable layer spacing** to optimally suppress the nearby strong He II 30.4 nm emission line. This results in a lower peak reflectivity of 15% compared to other channels.

거울 보정은 IAS의 싱크로트론에서 거울 쌍(pair)으로 수행되었다. 실제 EUVI 망원경과 동일한 기하학적 배치에서 거의 평행한 단색광 빔으로 조명하여, 각 사분면의 절대 반사율을 측정했다(Figure 9). 모든 코팅이 높은 반사율과 올바른 최대 반사 파장을 보여주었다.

Mirror calibration was performed at the IAS synchrotron with mirror pairs arranged in the same geometry as in the EUVI telescope, illuminated with a nearly collimated monochromatic beam. The absolute reflectivity of each quadrant was measured (Figure 9). All coatings showed high reflectivity and proper peak wavelength.

### Section 3.3: Filters / 필터 (pp. 114–115)

EUVI는 **입구 필터**와 **초점면 필터** 두 단계로 가시광/UV/IR을 억제한다:

EUVI suppresses visible/UV/IR light in two stages with **entrance filters** and **focal-plane filters**:

**입구 필터(Entrance Filters):**
- 단파장 사분면(17.1, 19.5 nm): **알루미늄-온-폴리이미드(aluminum-on-polyimide)** 호일을 거친 니켈 그리드에 장착
  Short-wavelength quadrants (17.1, 19.5 nm): aluminum-on-polyimide foil on coarse nickel grid
- 장파장 사분면(28.4, 30.4 nm): **단층 알루미늄** 호일을 미세 니켈 메시에 장착
  Long-wavelength quadrants (28.4, 30.4 nm): single-layer aluminum foil on fine nickel mesh
- 모두 150 nm 두께의 알루미늄으로 가시광 차단; 폴리이미드 지지층은 70 nm 두께
  Both use 150 nm thick aluminum for visible-light rejection; polyimide backing layer is 70 nm thick
- 그리드 지지 필터는 5 mm 간격의 거친 그리드로 회절이 최소화됨
  Grid-supported filters use 5 mm line spacing for minimal diffraction
- 폴리이미드는 EUV 파장에서 ~50% 투과율이지만, 17.1/19.5 nm의 강한 방출선에서는 큰 문제가 되지 않음
  Polyimide transmits only ~50% of EUV, but not a major concern for the strong 17.1/19.5 nm lines
- 메시 지지 필터는 흡수성 폴리이미드 층을 피하므로, 상대적으로 약한 28.4 nm 방출선에 유리
  Mesh-supported filters avoid the absorbing polyimide layer, advantageous for the weaker 28.4 nm line
- 미세 메시(0.36 mm 간격)는 밝은 태양 구조물 근처에서 눈에 띄는 회절을 유발
  Fine mesh (0.36 mm spacing) causes noticeable diffraction near very bright solar features

**초점면 필터(Focal-Plane Filters):**
- 150 nm 알루미늄을 미세 메시에 장착, 필터 휠에 내장
  150 nm aluminum on fine mesh, housed in a filter wheel
- 입구 필터에 핀홀이 발생할 경우를 대비한 이중 필터(redundant filters) 제공
  Redundant filters provided in case pinholes develop in entrance filters on orbit
- 3번째 필터 휠 위치: 입구 필터의 치명적 손상에 대비한 직렬 2매 필터
  3rd filter wheel position: two filters in series for catastrophic entrance filter damage
- 4번째 위치: 개방(지상 시험용)
  4th position: open (for ground testing)

### Section 3.4: Detector / 검출기 (pp. 115)

검출기는 e2v Technologies의 **CCD42-40**으로, 핵심 사양은 다음과 같다:

The detector is an e2v Technologies **CCD42-40** with the following key specifications:

- **Back-thinned, backside-illuminated** full-frame CCD: 뒷면을 얇게 깎아 EUV 광자가 직접 감광층에 도달
  Back-thinned so EUV photons reach the active layer directly
- **2048 × 2048 픽셀**, 정사각형 13.5 μm 픽셀
  2048 × 2048 pixels, square 13.5 μm pixels
- e2v는 XUV에서 높은 양자 효율(QE)과 안정성을 가진 CCD 제조에 오랜 실적 보유
  e2v has a long track record for CCDs with high QE and stability in the XUV
- **수동 냉각**: −60°C 이하로 유지하여 암전류 최소화 및 방사선 손상 경감
  Passively cooled below −60°C to minimize dark current and mitigate radiation damage

CCD QE 보정은 Brookhaven 싱크로트론과 LMSAL XUV 보정 시설에서 수행되었다. LMSAL 시설은 Manson X-ray 소스(1–17 nm)와 hollow cathode 소스(20–122 nm)를 사용한다. Figure 10의 측정 결과에 따르면, EUV 파장에서 QE가 매우 높으며(약 60–90%), CCD 응답 모델이 데이터와 잘 일치한다.

CCD QE calibration was performed at the Brookhaven synchrotron and the LMSAL XUV calibration facility. The LMSAL facility uses a Manson X-ray source (1–17 nm) and a hollow cathode source (20–122 nm). Figure 10 measurements show very high QE at EUV wavelengths (~60–90%), with the CCD response model fitting the data well.

### Section 3.5: Aliveness Source / 작동 확인 광원 (pp. 115)

EUVI에는 시험 및 보정용 **LED(발광 다이오드)**가 내장되어 있다:

EUVI contains built-in **LEDs** for testing and calibration:

- **청색 LED (470 nm)**: 스파이더에 장착, 두 거울에 반사되어 검출기를 조명; Si에서 EUV 광자와 유사한 침투 깊이를 가져 EUV 반응 대리 측정에 활용
  Blue LEDs (470 nm) mounted in spider, illuminate detector via both mirrors; similar penetration depth in Si as EUV photons, useful as proxy for EUV response
- **보라색 LED (400 nm)**: CCD 근처에 장착; CCD 표면 효과에 민감한 진단 도구
  Violet LEDs (400 nm) mounted near CCD; diagnostic sensitive to CCD surface effects

### Section 4: Mechanical Design / 기계 설계 (pp. 115–118)

#### 4.1 Metering Structure / 미터링 구조 (pp. 115)

- 주 구조체: **Graphite/Cyanate Ester 미터링 튜브** — 낮은 열팽창 계수(CTE)로 온도 변화 시 거울 간격 유지
  Main structure: Graphite/Cyanate Ester metering tube — low CTE maintains mirror separation across temperature range
- 초점 조절 메커니즘이 불필요한 이유: 낮은 CTE가 운용 온도 범위 전체에서 광학계를 초점 상태로 유지
  No focus mechanism needed: low CTE keeps optics in focus throughout operational temperature range
- 내부에 알루미늄 호일 라이닝: 증기 및 오염 차단 배리어
  Aluminum foil lining inside: vapor and contamination barrier
- 별도의 벤트 경로(vent path)를 통해 메커니즘의 오염물이 광학 공동(optical cavity)으로 유입되는 것을 방지
  Separate vent paths prevent contaminants from mechanisms entering the optical cavity

#### 4.2 Mirror Mounts / 거울 마운트 (pp. 116)

**주경 마운트:**
- **6각형 티타늄 링**에 3개의 **bi-pod flexure**로 거울 기판을 지지
  Hexagonal titanium ring with three bi-pod flexures supporting the mirror substrate
- Semi-kinematic 설계: 각 bi-pod가 2자유도를 강하게 구속하고 나머지 4자유도에서는 유연하게 작동
  Semi-kinematic design: each bi-pod strongly constrains 2 DOF, flexible in the other 4
- 열응력으로부터 거울을 격리: 22°C 온도 변화에서도 거울 형상에 측정 가능한 변형 없음 (간섭계 시험으로 확인)
  Isolates mirror from thermal stress: interferometric tests showed no measurable deformation with 22°C temperature change
- Bi-pod 재질: **Invar** (낮은 CTE)
  Bi-pod material: Invar (low CTE)

**부경 마운트:**
- **단일 Invar 부품**에 3개의 가공된 핑거(finger)로 Zerodur 기판의 원통형 베이스에 접착
  Single Invar piece with three machined fingers bonded to the cylindrical base of the Zerodur substrate
- **Tip-tilt 메커니즘**: 3개의 **압전 액추에이터(PZT)**가 Invar 마운트를 밀어 영상 안정화 수행
  Tip-tilt mechanism: three PZT actuators push against the Invar mount for image stabilization
- TRACE 망원경과 매우 유사한 설계
  Very similar design to the TRACE telescope
- SECCHI 가이드 망원경(GT)의 미세 포인팅 신호를 처리하여 PZT를 개루프(open loop)로 구동
  Fine pointing signals from the SECCHI guide telescope (GT) drive PZTs in open loop
- Tip-tilt 범위: EUVI 영상 공간에서 ±7 arcsec — 최악의 우주선 지터를 수용하기에 충분
  Tip-tilt range: ±7 arcsec in EUVI image space — sufficient for worst-case spacecraft jitter

#### 4.3 Mechanisms / 메커니즘 (pp. 117–118)

모든 EUVI 메커니즘은 이전 비행 프로그램에서 검증된 유산(heritage) 설계를 기반으로 한다:

All EUVI mechanisms are based on heritage designs from previous flight programs:

**도어(Door):**
- SOHO-LASCO 기반 설계, MPAe 제공
  SOHO-LASCO based design, provided by MPAe
- 발사 시 취약한 입구 필터 보호가 주 기능
  Primary function: protect fragile entrance filters during launch
- 스텝 모터 구동, 재폐쇄 가능(recloseable)
  Stepper motor driven, recloseable
- 모터 고장 대비 **단일 사용 왁스 액추에이터** 이중화
  Redundant single-use wax actuator in case of motor failure
- 이전 EUV 망원경들은 진공 챔버를 사용했으나, EUVI는 소형 필터이므로 밀폐된 도어 뚜껑만으로 발사 환경 생존 가능 (시험 확인)
  Previous EUV telescopes used vacuum chambers, but EUVI's small filters survive launch with just a firmly closed door lid (test-verified)

**사분면 선택기(Quadrant Selector):**
- 브러시리스 DC 모터 + 일체형 광학 인코더로 구동
  Brushless DC motor with integral optical encoder
- TRACE의 사분면 메커니즘 대비 **대폭 향상**: 사분면 전환 빈도에 제한 없음
  Major advance over TRACE's quadrant mechanism: no restrictions on switching frequency
- 매우 높은 신뢰성
  Highly reliable

**필터 휠(Filter Wheel) 및 셔터(Shutter):**
- GOES SXI-N에서 사용된 것과 거의 동일한 브러시리스 DC 모터 사용
  Nearly identical brushless DC motors as used on GOES SXI-N
- 셔터: 60 ms ~ 60 s 이상의 노출 시간 허용
  Shutter: allows exposure times from 60 ms to over 60 s
- 사분면 선택기, 필터 휠, 셔터 모두 LMSAL 제공
  Quadrant selector, filter wheel, and shutter all provided by LMSAL

#### 4.4 Focal Plane Assembly / 초점면 어셈블리 (pp. 118)

- NRL 제공
  Provided by NRL
- e2v CCD42-40 탑재
  Houses e2v CCD42-40
- **수동 냉각**: 알루미늄 콜드 핑거와 STEREO 반태양면 복사판을 통해 냉각
  Passively cooled via aluminum cold finger and radiator surfaces at the anti-sun deck
- 히터 장착: 발사 직후 높은 아웃가싱 시기와 오염 제거(decontamination) 시 CCD를 가온
  Heater equipped: keeps CCD warm during high outgassing period after launch and for decontamination
- 자체 벤트 경로: 콜드 핑거와 복사판을 통해 가스 배출
  Own vent path along cold finger and through radiator

### Section 5: Instrument Response and Calibration / 기기 응답 및 보정 (pp. 118–121)

#### 5.1 Calibration Results / 보정 결과 (pp. 118–119)

거울 쌍의 EUV 보정은 IAS 싱크로트론에서 수행되었다. 거울을 EUVI와 동일한 기하학적 배치로 배열하고, 부착된 단색화기(monochromator)로부터 거의 평행한 빔을 조사했다. 각 사분면을 개별 측정하여 **2-bounce 절대 반사율**을 얻었다(Figure 9).

Mirror pair EUV calibration was performed at the IAS synchrotron. Mirrors were arranged in the same geometry as in EUVI, illuminated with a nearly collimated beam from an attached monochromator. Each quadrant was measured individually to obtain **2-bounce absolute reflectivity** (Figure 9).

Figure 9의 결과:
- 17.1 nm 채널: 가장 높은 2-bounce 반사율 (~15%), 좁은 대역폭
  17.1 nm channel: highest 2-bounce reflectivity (~15%), narrow bandwidth
- 19.5 nm 채널: 비슷한 수준의 반사율 (~12%)
  19.5 nm channel: similar level reflectivity (~12%)
- 28.4 nm 채널: 가변 간격 코팅으로 인해 상대적으로 낮은 반사율 (~2%), 30.4 nm He II 억제 최적화
  28.4 nm channel: relatively low reflectivity (~2%) due to variable spacing coating, optimized for 30.4 nm He II suppression
- 30.4 nm 채널: 중간 수준의 반사율 (~5%)
  30.4 nm channel: moderate reflectivity (~5%)

CCD 보정은 Brookhaven 싱크로트론과 LMSAL XUV 보정 시설에서 수행되었다. Figure 10은 CCD의 양자 효율(QE)을 보여주며, 주요 결과는:

CCD calibration was performed at the Brookhaven synchrotron and LMSAL XUV facility. Figure 10 shows the CCD quantum efficiency, with key results:

- EUV 영역(10–30 nm)에서 QE ~60–90%로 매우 높음
  Very high QE of ~60–90% in the EUV range (10–30 nm)
- CCD 응답 모델이 측정 데이터와 우수하게 일치
  CCD response model fits measurement data excellently
- 표준(non-enhanced) e2v 후면 처리를 사용하여 GOES SXI-N CCD와는 약간 다른 응답 곡선
  Standard (non-enhanced) e2v backside treatment, resulting in slightly different response curve from GOES SXI-N CCDs

#### 5.2 Predicted Response to Solar Phenomena / 태양 현상에 대한 예측 응답 (pp. 120–121)

보정 결과를 바탕으로, CHIANTI 원자 데이터베이스와 문헌의 differential emission measure (DEM) 분포를 사용하여 다양한 태양 현상에 대한 EUVI의 응답을 예측했다:

Based on calibration results, EUVI response to various solar phenomena was predicted using the CHIANTI atomic database and differential emission measure (DEM) distributions from literature:

**Table 4 — 태양 현상별 예측 광자 카운트율 / Predicted Photon Count Rates (photons/s/pixel):**

| 채널 / Channel | Quiet Sun | Active Region | M-Class Flare |
|---|---|---|---|
| 17.1 nm | 95 | 986 | 25,800 |
| 19.5 nm | 43 | 852 | 92,100 |
| 28.4 nm | 3 | 110 | 5,150 |
| 30.4 nm | 30* | 416* | 18,200 |

(*30.4 nm 값은 CHIANTI가 He II 플럭스를 약 3배 과소평가하므로 3배 보정 / *30.4 nm values tripled because CHIANTI underestimates He II flux by a factor of ~3)

주요 시사점:
- 17.1 nm 채널이 전반적으로 가장 높은 카운트율 → quiet Sun에서도 높은 S/N 확보 가능
  17.1 nm channel has the highest overall count rate → high S/N even for quiet Sun
- 19.5 nm 채널은 flare 시 극도로 높은 카운트율(92,100 photons/s/pixel) → Fe XXIV 기여(높은 온도 감도)
  19.5 nm channel has extremely high flare count rate (92,100 photons/s/pixel) → Fe XXIV contribution (high temperature sensitivity)
- 28.4 nm 채널은 quiet Sun에서 가장 약함(3 photons/s/pixel) → 긴 노출 시간 필요
  28.4 nm channel is weakest for quiet Sun (3 photons/s/pixel) → longer exposures needed

Figure 11은 등온 플라즈마에 대한 온도별 응답을 보여준다:
- 17.1 nm: log T ≈ 5.9 (약 0.8 MK)에서 피크
- 19.5 nm: log T ≈ 6.2 (약 1.6 MK)에서 피크, log T > 7에서 Fe XXIV로 인한 2차 피크
- 28.4 nm: log T ≈ 6.3 (약 2 MK)에서 피크
- 30.4 nm: log T ≈ 4.9 (약 0.08 MK)에서 피크

Figure 11 shows the temperature response for isothermal plasmas:
- 17.1 nm: peaks at log T ≈ 5.9 (~0.8 MK)
- 19.5 nm: peaks at log T ≈ 6.2 (~1.6 MK), secondary peak at log T > 7 due to Fe XXIV
- 28.4 nm: peaks at log T ≈ 6.3 (~2 MK)
- 30.4 nm: peaks at log T ≈ 4.9 (~0.08 MK)

### Section 6: Observational Programs and Constraints / 관측 프로그램 및 제약 (pp. 121–122)

SECCHI의 관측 프로그램은 두 가지로 구성된다:

SECCHI's observing program consists of two components:

**1. Synoptic 프로그램 (시놉틱):**
- 시간 기반 관측 일정, **양쪽 관측소에서 동일하게** 수행
  Time-scheduled observations, **identical on both observatories**
- 가용 텔레메트리의 약 80% 사용
  Uses about 80% of available telemetry
- 안정적이고 지속적인 데이터 스트림 보장
  Ensures stable and continuous data stream

**2. Campaign 프로그램 (캠페인):**
- 더 유연하며, 제한된 기간 동안 **높은 촬영 주기** 허용
  More flexible, allows **higher cadence** for limited periods
- 별도의 덮어쓰기 가능한(overwritable) 텔레메트리 버퍼 사용 — 가장 최근의 고주기 관측만 보관
  Uses separate, overwritable telemetry buffer — retains only the most recent high-cadence observations
- SECCHI가 **CME 또는 플레어 이벤트 트리거**에 기반하여 자율적으로 고주기 버퍼를 동결(freeze) 가능
  SECCHI can autonomously freeze the high-cadence buffer based on CME or flare event triggers
- 양쪽 관측소의 엄밀한 동시 관측이 바람직하지만 필수는 아님
  Strictly simultaneous observations from both observatories desirable but not required

**운용 제약:**
- 두 우주선이 서로 통신할 수 없으므로, 관측 프로그램의 온보드 자율성이 제한됨
  The two spacecraft cannot communicate with each other, limiting on-board autonomy
- 제한된 다운링크 텔레메트리
  Limited downlink telemetry
- 미션의 과학적 강조점이 두 관측소의 이격 각도에 따라 지속적으로 변화:
  Science emphasis changes constantly as the two observatories drift apart:
  - **초기(이격 각도 소)**: 고전적 입체 영상에 적합 → CME 발생 연구에 주력, 고주기 관측
    Early mission (small separation): classical stereoscopy → CME initiation studies with high cadence
  - **이격 각도 ~90°**: 한쪽 EUVI가 다른 쪽 코로나그래프에서 관측된 CME의 저층 대기 효과를 관측
    ~90° separation: one EUVI observes low-atmospheric effects of CMEs seen by the other's coronagraphs
- 특정 이격 각도에서의 관측은 반복 불가 → 사전 계획 필수
  Observations at specific separation angles cannot be repeated → advance planning essential

---

## 3. Key Takeaways / 핵심 시사점

1. **Ritchey-Chrétien 설계의 채택이 핵심 혁신** — EIT의 단순한 설계에서 벗어나 쌍곡면 거울 쌍으로 업그레이드하여, 전체 시야(±1.7 R☉)에서 pixel-limited resolution을 달성하고 초점 조절 메커니즘을 불필요하게 만들었다.
   The adoption of Ritchey-Chrétien design was a key innovation — upgrading from EIT's simpler design to hyperboloidal mirror pairs achieved pixel-limited resolution across the full FOV (±1.7 R☉) and eliminated the need for a focus mechanism.

2. **EIT와 동일한 4개 파장 밴드를 유지한 전략적 선택** — 17.1, 19.5, 28.4, 30.4 nm의 4밴드 조합은 EIT에서 입증된 온도 진단 능력을 계승하면서, 분해능(1.6" vs 5.2")과 감도를 대폭 향상시켰다. 이 조합은 이후 AIA에서 10채널로 확장되었다.
   Maintaining the same four wavelength bands as EIT was a strategic choice — the 4-band combination inherits EIT's proven temperature diagnostic capability while dramatically improving resolution (1.6" vs 5.2") and sensitivity. This combination was later expanded to 10 channels in AIA.

3. **사분면 설계(quadrant design)로 자원 효율 극대화** — 하나의 망원경에서 4개 파장을 관측하기 위해 거울을 4사분면으로 나누어 각각 다른 multilayer 코팅을 적용했다. 이는 4개의 개별 망원경 대신 단일 망원경으로 질량·체적·전력을 절약하는 설계이다.
   The quadrant design maximizes resource efficiency — dividing mirrors into four quadrants with different multilayer coatings enables observation at four wavelengths from a single telescope, saving mass, volume, and power compared to four separate telescopes.

4. **28.4 nm 채널의 가변 간격 코팅은 교차 오염 억제의 좋은 사례** — 인접한 강한 He II 30.4 nm 선을 억제하기 위해 일정하지 않은 다층 주기를 사용한 것은, 밴드패스 설계에서 인접 파장 간 간섭 문제에 대한 실용적 해결책이다.
   The variable-spacing coating for the 28.4 nm channel is a good example of cross-contamination suppression — using non-uniform multilayer periods to reject the nearby strong He II 30.4 nm line is a practical solution to inter-band interference in bandpass design.

5. **PZT 기반 tip-tilt 영상 안정화는 TRACE 유산의 성공적 활용** — 부경에 3개의 압전 액추에이터를 장착하여 ±7 arcsec 범위의 개루프 보정을 수행한다. SECCHI 가이드 망원경의 신호를 사용하므로, EUVI 자체에는 별도의 가이드 시스템이 필요 없다.
   PZT-based tip-tilt image stabilization is a successful adaptation of TRACE heritage — three piezoelectric actuators on the secondary mirror provide ±7 arcsec open-loop correction using signals from the SECCHI guide telescope, so EUVI itself needs no separate guide system.

6. **Synoptic + Campaign 이중 관측 모드는 텔레메트리 제약 하의 유연한 설계** — 80%의 텔레메트리로 안정적 시놉틱 데이터를 보장하면서, 나머지 20%로 이벤트 기반 고주기 관측을 가능하게 한다. 자율적 이벤트 트리거로 고주기 버퍼를 동결하는 기능은 우주 기상 모니터링에 특히 유용하다.
   The dual Synoptic + Campaign observing mode is a flexible design under telemetry constraints — 80% of telemetry ensures stable synoptic data while the remaining 20% enables event-based high-cadence observations. The autonomous event trigger to freeze the high-cadence buffer is particularly useful for space weather monitoring.

7. **Graphite/Cyanate Ester 미터링 튜브의 낮은 CTE가 초점 안정성의 핵심** — 낮은 열팽창 계수로 운용 온도 범위 전체에서 거울 간격을 일정하게 유지하여, 능동 초점 조절 메커니즘 없이도 안정적인 영상을 생성한다. 이는 기계적 복잡성과 고장 위험을 줄이는 설계 철학이다.
   The low CTE of the Graphite/Cyanate Ester metering tube is key to focus stability — maintaining constant mirror separation across the operational temperature range produces stable images without an active focus mechanism, reducing mechanical complexity and failure risk.

8. **EUVI는 EIT → AIA 진화 계보의 결정적 중간 단계** — EIT의 입증된 개념(normal incidence, 4-band multilayer, quadrant)을 계승하면서, 해상도·감도·안정화 등을 현대화하여 AIA로 이어지는 기술 발전의 다리 역할을 했다.
   EUVI is a decisive intermediate step in the EIT → AIA evolutionary lineage — inheriting EIT's proven concepts (normal incidence, 4-band multilayer, quadrant) while modernizing resolution, sensitivity, and stabilization, serving as a bridge to the technological advances realized in AIA.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Ritchey-Chrétien 광학 파라미터 / R-C Optical Parameters

EUVI의 Ritchey-Chrétien 계에서, 부경 배율 $m$은:

In the EUVI Ritchey-Chrétien system, the secondary mirror magnification $m$ is:

$$m = \frac{f}{f_1} = \frac{1750}{1750/2.42} \approx 2.42$$

여기서 $f$는 시스템 유효 초점거리(1750 mm), $f_1$은 주경 단독 초점거리이다.

Where $f$ is the system effective focal length (1750 mm) and $f_1$ is the primary mirror alone focal length.

### 4.2 판 스케일 및 픽셀 스케일 / Plate Scale and Pixel Scale

$$\text{Plate Scale} = \frac{206265''}{f} = \frac{206265''}{1750 \text{ mm}} = 117.9 \text{ arcsec/mm}$$

$$\text{Pixel Scale} = \text{Plate Scale} \times p = 117.9 \times 0.0135 \text{ mm} = 1.59 \approx 1.6 \text{ arcsec/pixel}$$

여기서 $p = 13.5$ μm은 CCD 픽셀 크기이다.

Where $p = 13.5$ μm is the CCD pixel size.

### 4.3 시야 계산 / Field of View Calculation

$$\text{FOV} = \text{Pixel Scale} \times N_{\text{pixels}} = 1.6'' \times 2048 = 3276.8'' \approx 54.6'$$

태양의 각지름이 ~32 arcmin이므로, FOV ≈ 54.6 arcmin은 ±1.7 태양 반경에 해당한다:

Since the Sun's angular diameter is ~32 arcmin, FOV ≈ 54.6 arcmin corresponds to ±1.7 solar radii:

$$\frac{54.6'}{2 \times 16'} \approx 1.71 \, R_\odot$$

### 4.4 Multilayer Bragg 조건 / Multilayer Bragg Condition

Normal incidence에서의 constructive interference 조건:

Constructive interference condition at normal incidence:

$$m\lambda = 2d \cos\theta$$

Normal incidence ($\theta \approx 0°$, 표면 법선으로부터 측정)에서:

At normal incidence ($\theta \approx 0°$, measured from surface normal):

$$\lambda \approx 2d \quad (m = 1)$$

예를 들어, 17.1 nm 채널의 경우 $d \approx 8.6$ nm의 MoSi 이중층 주기가 필요하다.

For example, the 17.1 nm channel requires a MoSi bilayer period of $d \approx 8.6$ nm.

### 4.5 2-bounce 반사율 / Two-Bounce Reflectivity

시스템의 유효 반사율은 두 거울의 반사율 곱:

The system effective reflectivity is the product of both mirror reflectivities:

$$R_{\text{system}}(\lambda) = R_1(\lambda) \times R_2(\lambda)$$

Table 3의 단일 반사 데이터로부터:

From the single-reflection data in Table 3:

| 채널 | $R_1$ (single) | $R_{\text{system}}$ (estimated) |
|---|---|---|
| 17.1 nm | 39% | ~15.2% |
| 19.5 nm | 35% | ~12.3% |
| 28.4 nm | 15% | ~2.3% |
| 30.4 nm | 23% | ~5.3% |

### 4.6 예측 카운트율 계산 개요 / Predicted Count Rate Calculation Overview

EUVI의 예측 광자 카운트율은 다음 요소의 합성곱으로 계산된다:

The predicted photon count rate is calculated as a convolution of:

$$C(\lambda) = \int A_{\text{eff}}(\lambda) \cdot \Phi(\lambda) \, d\lambda$$

여기서:
- $A_{\text{eff}}(\lambda) = A_{\text{geo}} \cdot T_{\text{filter}}(\lambda) \cdot R_{\text{system}}(\lambda) \cdot QE(\lambda)$: 유효 면적
  Effective area combining geometric aperture, filter transmission, mirror reflectivity, and CCD QE
- $\Phi(\lambda)$: DEM과 CHIANTI로부터 예측한 태양 스펙트럼 복사 플럭스
  Solar spectral radiance flux predicted from DEM and CHIANTI
- $A_{\text{geo}}$: 기하학적 조리개 면적 (유효 구경 98 mm, 중심 차폐 65 mm)
  Geometric aperture area (effective aperture 98 mm, central obscuration 65 mm)

$$A_{\text{geo}} = \frac{\pi}{4}(D^2 - d^2) = \frac{\pi}{4}(98^2 - 65^2) \approx 4221 \text{ mm}^2$$

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1991 ─── Yohkoh/SXT 발사: 연 X선 태양 영상의 시대 개막
          Yohkoh/SXT launch: era of soft X-ray solar imaging begins

1995 ─── SOHO/EIT 발사 (Paper #9): 4밴드 EUV full-disk 영상의 표준 확립
          SOHO/EIT launch (Paper #9): establishes 4-band EUV full-disk imaging standard
          - Normal incidence, Mo/Si multilayer, quadrant design 개념 정립

1998 ─── TRACE 발사: 고분해능(0.5") EUV 영상, 좁은 시야
          TRACE launch: high-resolution (0.5") EUV imaging, narrow FOV
          - EUVI의 tip-tilt 메커니즘, 메시 필터 등의 직접적 유산

2003 ─── CHIANTI v4 출시: 태양 EUV 스펙트럼 예측을 위한 핵심 원자 데이터베이스
          CHIANTI v4 released: key atomic database for predicting solar EUV spectra

2004 ─── ★ 본 논문: EUVI 설계 및 예측 성능 기술
          ★ This paper: EUVI design and predicted performance

2006 ─── STEREO 발사 (10월 25일): EUVI 운용 시작, 사상 최초의 태양 입체 관측
          STEREO launch (Oct 25): EUVI operations begin, first-ever solar stereoscopy

2010 ─── SDO/AIA 발사 (Paper #12): 10채널, 12초 주기, 0.6" 해상도
          SDO/AIA launch (Paper #12): 10 channels, 12-sec cadence, 0.6" resolution
          - EUVI의 기술적 혁신(R-C 설계, 향상된 CCD)이 AIA에 반영

2011 ─── STEREO-B 지구 대면 180° 도달: 태양 전면 360° 관측 실현
          STEREO-B reaches 180° from Earth: 360° solar coverage achieved

2014 ─── STEREO-B 통신 두절
          STEREO-B communication lost

2018 ─── Solar Orbiter EUI 발사 준비: EUVI 기술 계보의 최신 후속
          Solar Orbiter EUI preparation: latest successor in EUVI technology lineage
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| #9 — Delaboudinière et al. (1995), EIT | EUVI의 직접적 전신; 4밴드 EUV, quadrant design, normal incidence 개념의 원조 / Direct predecessor; originated 4-band EUV, quadrant design, normal incidence concepts | EUVI가 EIT의 모든 핵심 개념을 계승하면서 분해능(5.2"→1.6")과 광학 설계(R-C)를 업그레이드 / EUVI inherits all core concepts while upgrading resolution and optics |
| #12 — Lemen et al. (2012), AIA | EUVI의 기술적 후속; 4밴드→10밴드, 1.6"→0.6", quadrant→개별 망원경 설계로 발전 / Technological successor; evolved from 4-band to 10-band, 1.6" to 0.6", quadrant to individual telescopes | EUVI에서 검증된 R-C 설계, back-thinned CCD, 영상 안정화 기술이 AIA에 반영 / R-C design, back-thinned CCD, and image stabilization proven in EUVI carried forward to AIA |
| #11 — Title et al. (TRACE) | EUVI의 tip-tilt 메커니즘, 메시 지지 필터, focal plane 설계의 직접적 유산 / Direct heritage for tip-tilt mechanism, mesh-supported filters, and focal plane design | PZT 기반 부경 tip-tilt이 TRACE에서 검증 후 EUVI에 적용 / PZT-based secondary tip-tilt validated on TRACE before EUVI adoption |
| GOES SXI-N (Stern et al. 2003) | EUVI의 필터 휠 및 셔터 메커니즘(브러시리스 DC 모터)의 비행 유산 / Flight heritage for filter wheel and shutter mechanisms (brushless DC motors) | 검증된 메커니즘 설계를 재사용하여 개발 위험 감소 / Reuse of proven mechanism designs to reduce development risk |
| SOHO-LASCO | EUVI 도어 메커니즘의 설계 기반 / Design basis for EUVI door mechanism | 발사 시 필터 보호를 위한 도어 설계의 비행 유산 / Flight heritage for door design protecting filters during launch |
| CHIANTI (Young et al. 2003) | EUVI의 예측 응답 계산에 사용된 원자 데이터베이스 / Atomic database used for EUVI predicted response calculations | DEM과 결합하여 각 채널의 광자 카운트율과 온도 응답 함수를 예측 / Combined with DEM to predict photon count rates and temperature response functions |

---

## 7. References / 참고문헌

- Wuelser, J.-P., Lemen, J.R., et al., "EUVI: the STEREO-SECCHI extreme ultraviolet imager," *Proc. SPIE 5171*, Telescopes and Instrumentation for Solar Astrophysics, pp. 111–122, 2004. [DOI: 10.1117/12.506877]
- Delaboudinière, J.-P., et al., "EIT: Extreme-Ultraviolet Imaging Telescope for the SOHO Mission," *Solar Physics*, Vol. 162, pp. 291–312, 1995. [DOI: 10.1007/BF00733432]
- Ravet, M.F., Bridou, F., et al., "Ion beam deposited Mo/Si multilayers for EUV imaging applications in astrophysics," *Proc. SPIE 5250*, 2003.
- Handy, B.N., et al., "The Transition Region and Coronal Explorer," *Solar Physics*, Vol. 187, pp. 229–260, 1999.
- Akin, D., Horber, R., Wolfson, C.J., "Three High Duty Cycle, Space-Qualified Mechanisms," *NASA Conf. Pub. 3205*, 1993.
- Windt, D.L. and Catura, R.C., "Multilayer Characterization at LPARL," *Proc. SPIE 984*, pp. 132–139, 1988.
- Stern, R.A., Shing, L., Blouke, M.M., "Quantum efficiency measurements of ion-implanted, laser-annealed charge coupled devices: x-ray, extreme-ultraviolet, ultraviolet, and optical data," *Appl. Opt.*, Vol. 33, pp. 2521–2533, 1994.
- Stern, R.A. and the SXI Team, "Solar X-ray imager for GOES," *Proc. SPIE 5171*, 2003.
- Young, P.R., Del Zanna, G., Landi, E., et al., "CHIANTI – An Atomic Database for Emission Lines," *The Astrophysical Journal Suppl. Series*, Vol. 144, pp. 135–152, 2003.
- Lemen, J.R., et al., "The Atmospheric Imaging Assembly (AIA) on the Solar Dynamics Observatory (SDO)," *Solar Physics*, Vol. 275, pp. 17–40, 2012. [DOI: 10.1007/s11207-011-9776-8]
