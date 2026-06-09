---
title: "The McMath Solar Telescope of Kitt Peak National Observatory"
authors: A. Keith Pierce
year: 1964
journal: "Applied Optics, Vol. 3, No. 12, pp. 1337–1346"
doi: "10.1364/AO.3.001337"
topic: Solar Observation / Ground-Based Telescopes
tags: [solar telescope, heliostat, spectrograph, vacuum spectrometer, McMath-Pierce, Kitt Peak, infrared spectroscopy, Fraunhofer lines]
status: completed
date_started: 2026-04-10
date_completed: 2026-04-10
---

# The McMath Solar Telescope of Kitt Peak National Observatory

## 핵심 기여 / Core Contribution

이 논문은 세계 최대의 태양 망원경인 McMath Solar Telescope의 기계적·광학적 설계를 포괄적으로 기술한 최초의 정식 instrument paper이다. 160cm 구경, 90m 초점 거리를 가진 이 망원경은 헬리오스탯 방식을 채택하여 태양광을 지하 관측실로 유도하며, 광학 경로 전체의 열 제어를 통해 internal seeing을 최소화했다. 논문은 fused quartz 거울과 metal 거울의 비교, vacuum double-pass spectrometer의 설계와 성능(resolution 600,000, scattered light 3%), 그리고 기계적 구동 시스템의 정밀도를 상세히 다룬다. 이 시설은 이후 수십 년간 적외선 태양 분광학의 세계적 중심지가 되었다.

This paper is the first formal instrument paper comprehensively describing the mechanical and optical design of the McMath Solar Telescope — the world's largest solar telescope. With a 160-cm aperture and 90-m focal length, it employs a heliostat configuration to direct sunlight into an underground observing room, minimizing internal seeing through thermal control of the entire optical path. The paper details the comparison between fused quartz and metal mirrors, the design and performance of the vacuum double-pass spectrometer (resolution 600,000, 3% scattered light), and the precision of the mechanical drive systems. This facility became the world's premier center for infrared solar spectroscopy for decades to come.

---

## 읽기 노트 / Reading Notes

### 1. 서론과 과학적 동기 / Introduction and Scientific Motivation

Pierce는 태양이 약 5750K의 유효 온도를 가진 거의 Planckian 스펙트럼을 방출하며, 그 위에 약 30,000개의 Fraunhofer 흡수선이 중첩되어 있다고 설명한다. 각 흡수선은 단순히 화학 조성만이 아니라, 태양 대기에 있는 원자의 **물리적 상태**를 반영한다:

Pierce explains that the Sun emits a nearly Planckian spectrum at ~5750 K, superimposed with ~30,000 Fraunhofer absorption lines. Each line encodes not just chemical composition, but the **physical state** of atoms in the solar atmosphere:

- **파장 위치의 변위 (wavelength displacement)**: 태양-지구 간 중력 적색편이, 압력 이동(pressure shift), 질량 운동(mass motion)에 의한 Doppler shift
  Wavelength displacements from gravitational redshift, pressure shifts, and Doppler shifts from mass motions
- **등가 폭과 $f$-값 (equivalent width and $f$-values)**: 화학 조성 결정
  Chemical composition determination
- **선 폭과 비대칭 (line width and asymmetry)**: 운동 온도, 난류 속도, 자기장(Zeeman splitting)
  Kinetic temperature, turbulent velocity, magnetic fields (Zeeman splitting)
- **선 형성 높이 (height of line formation)**: Stark broadening 등을 통한 대기 구조 정보
  Atmospheric structure from Stark broadening, etc.

1960년대에 접어들면서 태양 물리학의 관심사가 **거시적 특성**(평균 Fraunhofer 선 파장, 광구/채층 모델)에서 **미시적 비균질성**(active region, flare, 대류, 자기유체역학적 구조)으로 전환되고 있었다. 이 미세구조를 연구하려면 Fraunhofer 선의 세밀한 프로파일을 슬릿 길이 방향으로 점별(point by point)로 측정해야 하며, 이를 위해서는 **stigmatic spectrograph**(상이 왜곡되지 않는 분광기)가 필요했다.

By the 1960s, solar physics was shifting from **macroscopic properties** (mean Fraunhofer wavelengths, photosphere/chromosphere models) to **microscopic inhomogeneities** (active regions, flares, convection, magnetohydrodynamic structures). Studying this fine structure requires detailed profiles of Fraunhofer lines measured point by point along the slit length, demanding a **stigmatic spectrograph**.

이것이 대형 태양 망원경이 필요한 근본적 이유이다: **더 많은 빛을 모아** 고분산(high-dispersion) 스펙트럼에서도 충분한 신호대잡음비(S/N ratio)를 확보하고, 동시에 **큰 이미지 스케일**로 태양 표면의 미세구조를 공간적으로 분해해야 했기 때문이다.

This is the fundamental reason for a large solar telescope: to collect **enough light** for adequate S/N ratio in high-dispersion spectra, while simultaneously providing a **large image scale** to spatially resolve fine structure on the solar surface.

논문은 또한 당시의 주요 관측 성과들을 언급한다:
The paper also mentions key contemporary observational achievements:

- Leighton 등의 5분 진동(5-min oscillation) 발견
  Discovery of 5-minute oscillations by Leighton et al.
- Moreton의 플레어 충격파(blast waves) 관측 — 1100 km/sec
  Moreton's observation of flare blast waves at 1100 km/sec
- 채층의 "wand waving" 필라멘트 운동
  "Wand waving" motion of filamentary structures in the chromosphere
- Babcock magnetograph로 0.2 G 민감도의 전면 자기장 측정
  Full-disk magnetic field mapping with 0.2 G sensitivity using the Babcock magnetograph

### 2. 일반 설계 철학 / General Design Philosophy

McMath 망원경의 최종 설계를 결정한 두 가지 핵심 기준:

Two principal design criteria determined the final configuration:

**기준 1: 관측 가능 시간 / Criterion 1: Observing Time**
- 연간 **30시간** 이상의 0.5 arcsec 이하 seeing 조건 확보 목표
  Goal: at least **30 hours per year** of sub-0.5 arcsec seeing
- 빛이 대기를 통과한 후, 지형이나 망원경 자체에 의해 이미지 품질이 파괴되어서는 안 된다는 원칙
  Guiding principle: after light passes through the atmosphere, image quality should not be destroyed by local terrain or the telescope itself

**기준 2: 0.33 arcsec 분해능 / Criterion 2: 0.33 arcsec Resolution**
- 이를 위해 30cm 구경이 필요 (회절 한계 $\theta = 1.22\lambda/D$에 의해)
  Requires 30-cm aperture (from diffraction limit $\theta = 1.22\lambda/D$)
- 그러나 주간 seeing은 이보다 훨씬 열악하므로, 더 큰 구경이 필요
  But daytime seeing is far worse, requiring larger aperture

**이미지 크기와 초점 거리의 관계 / Image Size and Focal Length:**
Pierce는 최적 이미지 크기를 결정하는 여러 요인을 분석한다:
Pierce analyzes several factors governing optimal image size:

- 태양상 지름 ~1m일 때, 매우 좋은 seeing에서 1 arcsec (0.5 mm) 크기의 granule을 분광측광 가능
  With ~1m solar image, spectrophotometry of 1-arcsec (0.5 mm) granules is possible in very good seeing
- 흑점, 자기장의 물리적 미세구조 연구에는 상당한 이미지 크기가 필요
  Physical fine structure of sunspots and magnetic fields requires considerable image size
- Bowen image slicer를 사용하면 S/N ratio가 구경 $A$에 직접 비례 ($\propto A$), 미사용 시 $\propto \sqrt{A}$
  With a Bowen image slicer, S/N ∝ aperture $A$ directly; without it, S/N ∝ $\sqrt{A}$

**왜 헬리오스탯인가? / Why a Heliostat?**
여러 광학 배치(Cassegrain, coelostat, heliostat, siderostat) 중 헬리오스탯을 선택한 이유:
Reasons for choosing the heliostat over other optical arrangements:

| 장점 / Advantage | 설명 / Explanation |
|---|---|
| 단순한 마운팅 | 단일 평면 거울만 필요 → 비용 절감 / Single flat mirror → lower cost |
| 반사 1회 | 편광과 타원율(ellipticity)이 일정 / Constant polarization and ellipticity |
| "noon shadow" 없음 | Coelostat에서는 두 거울의 그림자가 겹칠 수 있음 / No shadow overlap as in coelostat |
| 높은 위치 배치 가능 | 지면에서 높이 올려 열 난류를 회피 / Can be placed high above ground to avoid thermal turbulence |

| 단점 / Disadvantage | 설명 / Explanation |
|---|---|
| 시야 회전 | 24시간에 1회전 → 이미지가 회전함 / Field of view rotates once per 24h |
| 비최적 입사각 | 입사각과 반사각 사이의 각도가 coelostat보다 큼 → 거울 왜곡에 더 민감 / Angle between incident and reflected light is generally greater → more sensitive to mirror figure |

### 3. 망원경 구조와 마운팅 / Telescope Structure and Mountings

McMath Solar Telescope의 물리적 구조는 매우 독특하다 — 거대한 경사면(incline)을 따라 빛이 이동한다:

The physical structure is highly distinctive — light travels along a massive incline:

**광학 경로 / Optical Path:**
- 광학 축은 수평면에서 31° 57.5분 기울어져 있음 (Kitt Peak의 위도에 대응)
  Optical axis inclined at 31° 57.5' to horizontal (matching Kitt Peak latitude)
- 지면 위 60m, 지하 90m — 총 경로 약 155m
  60 m above ground, 90 m below ground — total path ~155 m
- 헬리오스탯에서 반사된 빛은 경사면을 따라 내려가 60인치 주경(No. 2 mirror)에 도달
  Light reflected from the heliostat travels down the incline to the 60-inch primary (No. 2 mirror)
- 주경에서 반사된 빛은 되돌아 올라가 No. 3 평면 거울에서 관측실로 반사
  Light reflects back up from primary to No. 3 flat mirror, which directs it to the observing room

**열 제어 시스템 / Thermal Control System:**
이것이 이 논문의 가장 핵심적인 엔지니어링 기여 중 하나이다:
This is one of the paper's most critical engineering contributions:

- 지면 위 경사면(60m)은 10m × 정사각 단면의 수냉식 외피(enclosure)로 차폐
  Above-ground incline (60 m) is shielded by a water-cooled enclosure (10 m × square cross-section)
- 외피 내부에 Airtex 패널을 통해 냉각수를 순환시켜 **비대류 공기층(nonconvective air column)** 유지
  Coolant circulated through Airtex panels inside the enclosure to maintain a **nonconvective air column**
- 155m 길이에서 약 5°C의 온도 구배를 유지 (바닥에서 위로)
  ~5°C temperature differential maintained over the 155-m length (bottom to top)
- 외부는 30톤의 구리판(10m × 2.5m 패널)으로 덮여 있으며, 각 패널 내부에 3개의 1cm 직경 구리 튜브
  Exterior covered with 30 tons of copper sheeting (10 m × 2.5 m panels), each containing three 1-cm diameter copper tubes
- **70,000 리터**의 부동액(antifreeze) 혼합 냉각수를 4,000 liters/min으로 순환
  **70,000 liters** of refrigerant water with antifreeze circulated at 4,000 liters/min
- 외부를 흰색 TiO₂ 안료로 도장 → 가시광 반사, 적외선 방출로 냉각 부하 대폭 감소
  Exterior painted white with TiO₂ pigment → reflects visible, emits infrared, greatly reducing cooling load
- 하늘을 향한 패널의 유효 하늘 온도: −15°C ~ −30°C
  Effective sky temperature for upward-facing panels: −15°C to −30°C

**헬리오스탯 타워 / Heliostat Tower:**
- 높이 30m, 직경 9m의 콘크리트 원주
  30-m high, 9-m diameter concrete column
- 1.2m 두께의 수냉식 콘크리트 벽(windshield)으로 둘러싸여 헬리오스탯만 노출
  Surrounded by 1.2-m thick water-cooled concrete walls, exposing only the heliostat
- 설계 기준: 18 m/sec 바람에서 태양상 떨림이 0.33 arcsec 미만
  Design criterion: image deflection < 0.33 arcsec in 18 m/sec wind
- Kitt Peak에서의 미세열구조(microthermal structure) 실험 결과, 지면에서 멀어질수록 대기 요동이 지수적으로 감소 → 15m 고도에서 0.4°C (지면 부근 3°C에 비해)
  Microthermal experiments at Kitt Peak showed fluctuations decrease exponentially with height → 0.4°C at 15-m elevation (vs. 3°C near ground)

### 4. 거울 시스템 / Mirror System

#### No. 1 — 헬리오스탯 거울 (80인치/203cm 평면) / Heliostat Mirror (80-inch/203-cm flat)

- Westinghouse Electric Corporation에서 제작
  Fabricated by Westinghouse Electric Corporation
- 적도의 요크(equatorial yoke) 마운팅, 12,000 kg
  Equatorial yoke mounting, 12,000 kg
- 남단의 304cm 직경 oil-pressure-pad 베어링으로 지지
  Supported by a 304-cm diameter oil-pressure-pad bearing at the south end
- 적경 구동: 720개 이(teeth)의 worm wheel, 주파수 제어 동기 모터로 평균 태양시 추적
  Right ascension drive: 720-tooth worm wheel, frequency-controlled synchronous motor tracking mean solar time
- 0.05 arcsec 단위의 미세 보정 가능 (differential stepping motor)
  Fine corrections in 0.05 arcsec steps via differential stepping motor
- 적위 구동: 2.7m 긴 탄젠트 암(tangent arm)으로 느린 보정
  Declination drive: 2.7-m long tangent arm for slow corrections
- **공기압 부양(pneumatic flotation) 시스템**: 거울의 고도각에 따라 공기압/진공이 비례하여 변하는 피스톤으로 거울 무게를 지지 — 기존의 복잡한 counterbalance 대신 사용
  **Pneumatic flotation system**: a piston driven by air pressure/vacuum proportional to the sine of altitude angle supports the mirror weight — replacing the elaborate counterbalance systems
- 악천후 시 55,000 kg의 전체 마운팅이 경사면 위 15m를 하강하여 보관, 금속 문이 닫힘 → 복원에 ~20분
  In inclement weather, the entire 55,000-kg mounting is lowered 15 m along the incline; restoration takes ~20 min

#### No. 2 — 주경 (60인치/160cm 오목) / Primary Mirror (60-inch/160-cm concave)

- 초점 거리 90m (약 300 feet), 88m로 기술된 곳도 있음 (aluminizing 후)
  Focal length 90 m (~300 ft), also described as 88 m (after aluminizing)
- 30cm heavy-wall 알루미늄 파이프로 용접된 프레임에 6개의 패드로 고정, 고정 방향
  Welded frame of 30-cm heavy-wall aluminum pipe, supported on six pads, fixed orientation
- 2m 범위에서 5,000 kg 캐리지를 모터 구동 볼 스크류로 이동하여 초점 조절
  Focus adjusted by moving the 5,000-kg carriage via motor-driven ball screw over a 2-m range

#### No. 3 — 평면 거울 (48인치/122cm) / Flat Mirror (48-inch/122-cm)

- No. 2보다 작은 마운팅, 고정 위치
  Smaller mounting than No. 2, fixed location
- No. 2와 No. 3 캐리지를 트랙 위 3개 위치 중 하나로 이동 가능 → 3개의 다른 기기로 빛을 보낼 수 있음
  No. 2 and No. 3 carriages can move to any of three positions on the track → light directed to three different instruments
- 전체 태양 디스크를 보아야 하므로 크기가 결정됨 (림, 적도, 극 좌표 확인용)
  Size determined by the need to see the full solar disk (for limb, equator, pole reference)

모든 거울은 경사면 전체 길이를 따르는 3.66m 게이지 트랙 위의 캐리지에 장착되어, 적절한 호이스트로 알루미늄 코팅실로 이동 가능하다.

All mirrors are mounted on carriages riding a 3.66-m gauge track extending the full length of the incline, allowing any mirror to be transported to the aluminizing room.

### 5. 태양 망원경 거울: Quartz vs. Metal / Solar Telescope Mirrors: Quartz vs. Metal

이 섹션은 단순한 사양 기술을 넘어, 태양 망원경 거울 재료에 대한 **심층 비교 분석**을 담고 있다. 태양 관측에서 거울 재료 선택이 왜 중요한지의 물리적 이유를 명확히 한다.

This section goes beyond simple specification — it contains a **comparative analysis** of mirror materials for solar telescopes, clarifying the physical reasons why material choice matters.

**근본 문제 / The Fundamental Problem:**
대부분의 태양 망원경에서 돔을 열고 거울을 태양광에 노출시킨 지 약 2분 후부터 seeing/이미지 품질이 현저히 악화된다. 원인은 거울 표면 위의 대류(convection currents)나 거울 자체의 열변형(distortion)이다.

In most solar telescopes, seeing/image quality markedly deteriorates within about 2 minutes of opening the dome and exposing mirrors to sunlight. The cause is either convection currents over the mirror surface or thermal distortion of the mirror itself.

알루미늄 코팅은 에너지의 약 90%를 반사하고 10%를 흡수한다. 흡수된 열은 전면에서 대류와 복사로, 거울 내부로 전도되며, 후면과 측면에서도 복사와 전도로 방출된다. 이 **비대칭 가열**이 거울 형상을 파괴한다.

An aluminum coating reflects ~90% and absorbs ~10% of incident energy. The absorbed heat dissipates by convection and radiation from the front, conduction into the mirror body, and radiation/conduction from the rear and edges. This **asymmetric heating** destroys the mirror figure.

**열 변형 계수 $K$ / Thermal Distortion Factor $K$:**
열적으로 응력을 받은 두께 $t$의 원형 판에서, 한 면이 온도 $T$이고 다른 면이 $T + \Delta T$일 때, 균일한 온도 구배에 대해 판은 다음의 곡률을 갖는다:

For a thermally stressed circular plate of thickness $t$, with one face at temperature $T$ and the other at $T + \Delta T$, for a uniform temperature gradient, the plate assumes a spherical curvature of:

$$\text{curvature} = \frac{t}{\alpha \times \Delta T}$$

여기서 $\alpha$는 열팽창 계수이다. 예를 들어, 150cm 직경 × 20cm 두께의 quartz 거울이 1$\lambda$ 이내로 평탄하려면 전후면 온도차가 0.06°C 미만이어야 한다.

where $\alpha$ is the coefficient of thermal expansion. For example, a 150-cm diameter × 20-cm thick quartz mirror must have a front-to-back temperature differential of less than 0.06°C to remain flat to within 1λ.

Couder의 비교 계수 $K = \alpha \delta c / m$:
Couder's comparison factor $K = \alpha \delta c / m$:

| 재료 / Material | $K$ (× 10⁻⁴ cgs) |
|---|---|
| Glass | 152 |
| Pyrex | 50 |
| Metals | 6 |
| **Fused Quartz** | **5** |

여기서 $\alpha$는 열팽창 계수, $\delta$는 밀도, $c$는 비열, $m$은 열전도도이다. $\alpha$와 $m$ 사이의 **역 상관관계** 때문에, 열팽창이 큰 금속이 열전도도도 높아서 결국 quartz와 비슷한 $K$ 값을 갖게 된다. 이것은 직관에 반하는 결과이다.

where $\alpha$ is thermal expansion, $\delta$ density, $c$ specific heat, $m$ thermal conductivity. The **inverse correlation** between $\alpha$ and $m$ means metals (high expansion but high conductivity) end up with comparable $K$ to quartz. This is a counterintuitive result.

**Fused Quartz 거울 / Fused Quartz Mirrors:**
- 1953년 Ira S. Bowen(Mt. Wilson/Palomar 소장)을 통해, Palomar 200인치 망원경 제작 시 만들어진 quartz blank들이 McMath에게 제공됨
  In 1953, quartz blanks from the Palomar 200-inch project were made available to McMath through Ira S. Bowen
- 3개의 디스크: (a) 160cm × 24cm — 헬리오스탯 평면 거울 및 No. 3 평면 거울로 사용, (b) 165cm × 21cm — 대각선으로 깨져 수리 후 122cm로 절단 → No. 3 거울, (c) quartz 코팅 없는 디스크
  Three disks: (a) 160 cm × 24 cm → heliostat flat and No. 3 flat, (b) 165 cm × 21 cm → cracked, cut to 122 cm → No. 3 mirror, (c) no quartz coating disk
- 200년 관측에서도 비점수차(astigmatism) 없음, 우수한 태양 거울 후보
  No astigmatism found in observations; excellent solar mirror candidates
- 단, 표면 결함이 산란광(scattered light)을 증가시키고 대비(contrast)를 저하시킴
  However, surface defects scatter light and degrade contrast

**Metal 거울 / Metal Mirrors:**
- 높은 열전도도로 주위 온도까지 빠르게 냉각 가능 → 대류 제거 가능성
  High thermal conductivity allows rapid cooling to ambient → potential to eliminate convection
- 내부 덕트를 통한 냉각도 용이, 온도 변화에 대한 시정수(time constant)가 짧음
  Internal duct cooling feasible, shorter time constant for temperature changes
- 26cm 직경의 Fe, Cu, Be, Al 거울을 시험. 130μ Kanigen(amorphous nickel-phosphorous alloy) 코팅
  Tested 26-cm diameter mirrors of Fe, Cu, Be, and Al, coated with 130μ Kanigen
- Fe, Be, Al이 적합. Cu는 너무 연함(soft)
  Fe, Be, Al suitable; Cu too soft
- 대부분 **356-T6 알루미늄 주조 합금** 사용, 40–160 cm 블랭크 제작
  Most work with **356-T6 aluminum casting alloy**, blanks of 40–160 cm produced
- No. 2 주경: 알루미늄 주조, 160cm 직경, 25.4cm 두께, triangular-hexagonal 리브 패턴(삼각형 높이 18cm, 리브 두께 2.5cm), 무게 710 kg, 표면 두께 3.8cm + 130μ Kanigen 코팅
  No. 2 primary: aluminum casting, 160-cm diameter, 25.4-cm overall thickness, triangular-hexagonal rib pattern (18-cm high, 2.5-cm thick ribs), 710 kg, 3.8-cm face thickness + 130μ Kanigen coating
- 문제: Kanigen 표면 연마 시 5–20cm 간격의 리플(ripple) 발생, 진폭 $\lambda$의 일부. Fused silica만큼 매끄러운 표면은 아직 달성하지 못함
  Problem: figuring Kanigen-coated surfaces produces ripple at 5–20 cm spacing; not yet as smooth as fused silica

**결론 / Conclusion:**
quartz와 metal 중 최종 결정이 어려움. quartz의 장기 안정성에는 합리적 의심이 없으나, metal 거울의 상대적 장점을 더 조사하기 위해 turntable에 양쪽을 장착하여 직접 비교할 계획.

Difficult to decide between quartz and metal. No reasonable doubt about quartz's long-term stability, but plan to mount both on a turntable for direct comparison.

### 6. 망원경 성능 / Telescope Performance

**외부 seeing / External Seeing:**
- 알루미늄 거울 + No. 3 평면 거울 조합에서 별(star) 관측 시: **2 arcsec 원 안에 100%, 1 arcsec 안에 50%**의 빛 집중
  With aluminum mirror + No. 3 flat, stellar observations: **100% of light within 2 arcsec, 50% within 1 arcsec**
- Seeing과 거울 형상이 합쳐져 smearing 발생
  Combined seeing + mirror figure produces smearing
- W. C. Livingston의 광전 측정: 이미지 smear는 거의 Gaussian, 폭(FWHM) 약 1.5 arcsec
  W. C. Livingston's photoelectric measurement: image smear is nearly Gaussian, width ~1.5 arcsec
- 일출 후 2시간이 가장 좋은 시간대, 때때로 1 arcsec granulation이 보임
  Best seeing 1–2 hours after sunrise; 1-arcsec granulation occasionally visible
- 최고 조건에서 초점은 ±2 cm 이내로 결정 가능
  In best conditions, focus determinable to ±2 cm

**내부 seeing / Internal Seeing:**
- 헬리오스탯에 초점을 맞춘 소형 망원경으로 광학 열차(optical train)를 통해 관찰
  Examined through a small telescope focused on the heliostat through the optical train
- 조용한 날에는 먼지 입자가 거의 정지 → 빛줄기에서 거의 움직임 없음 → 우수한 internal seeing (<0.5 arcsec)
  On quiet days, dust particles nearly motionless → excellent internal seeing (<0.5 arcsec)
- 큰 주야간 온도 변화 + 바람이 있는 날에는 internal seeing 악화
  Large diurnal temperature changes + wind degrade internal seeing
- 야간 별 관측에서 0.2 arcsec까지 안정된 기록 있음 (이미지 크기는 주간보다 훨씬 작음)
  Nighttime stellar images stable to 0.2 arcsec (though image size is much larger daytime)

### 7. Vacuum Spectrograph / 진공 분광기

이 섹션은 논문의 핵심 기술 기여이다. 현대 태양 분광학의 목표인 **Fraunhofer 선의 정밀 프로파일 측정**을 위한 기기 설계를 상세히 다룬다.

This section is the paper's core technical contribution, detailing instrument design for **precision Fraunhofer line profile measurement**.

**설계 요구사항 / Design Requirements:**
- 등가 폭(equivalent width), 반치폭(half-width), 비대칭성(asymmetry), 중심 세기(central intensity)를 0.01% 정밀도로 측정 목표
  Goal: measure equivalent width, half-width, asymmetry, and central intensity to 0.01% precision
- 이를 위해 최대 분해능(highest attainable resolution), 기기 프로파일(instrumental profile)과 고스트(ghosts) 결정, 산란광(scattered light) 제거/측정이 필수
  Requires highest resolution, characterization of instrumental profile and ghosts, and elimination/measurement of scattered light

**$f$-ratio와 분광기 설계의 연쇄 / The $f$-ratio Chain:**
Pierce는 분광기 설계의 핵심 논리를 전개한다:

Pierce develops the key logic of spectrograph design:

1. 고분해능 판 분해능(plate resolution) 10μ를 사용하면 정밀 측광이 불가능 (판 grain 때문)
   High plate resolution (10μ) makes precision photometry impossible (plate grain)
2. 더 넓은 슬릿 → $f$-ratio 60 선택 → 분해능 30μ
   Wider slit → select $f$-ratio 60 → resolution 30μ
3. $f/60$에서 150 mm × 250 mm 격자 → 분광기 초점 거리 10m 필요
   At $f/60$, a 150 mm × 250 mm grating requires 10-m spectrograph focal length
4. 격자를 완전히 채우고 싶으므로(overfill) → 13.7m 초점 거리의 collimator 선택
   Want to overfill the grating → select 13.7-m focal length collimator
5. 이 collimator의 $f$-ratio가 망원경의 최적 $f$-ratio를 결정
   This collimator's $f$-ratio determines the optimum telescope $f$-ratio
6. 30cm 구경이 0.33 arcsec 분해능을 줌 → but 더 큰 구경 필요 (주간 seeing 고려)
   30-cm aperture gives 0.33 arcsec resolution → but larger aperture needed (daytime seeing)
7. 최종적으로 160cm 구경의 quartz blank 2개를 확보, $f/60$으로 초점 거리 100m (실제 약 90m)
   Finally secured two 160-cm quartz blanks; at $f/60$, focal length ~100 m (actual ~90 m)

**광학 시스템 / The Optical System:**
- 짧은 파장 구간만 관측 시: 전반사(all-reflecting) 시스템 선호 — 색수차 없음, 다이어프램으로 산란광 차단 가능
  For short wavelength intervals: all-reflecting system preferred — no chromatic aberration, diaphragms catch scattered light
- McMath-Hulbert 진공 분광기의 변형 Czerny 광학계를 복제
  Copied the modified Czerny optical system from the McMath-Hulbert vacuum spectrograph
- 사진 촬영은 거의 모두 **single pass**로 수행
  Nearly all photographic work done in **single pass**
- 비점수차 보정 없음(non-compensating for astigmatism) — but Czerny-Turner 배치에서 $f/60$의 큰 focal ratio 덕분에 이미지 품질 우수
  No astigmatism compensation — but excellent image quality due to large $f/60$ focal ratio in Czerny-Turner arrangement

**Double-Pass 배치 / Double-Pass Arrangement:**
이것이 가장 혁신적인 부분이다:
This is the most innovative part:

- 첫 번째 시도: 입사 슬릿 S₁ → single pass → 중간 슬릿 S₂ → 시스템을 통해 되돌아감 → 입사 슬릿의 반대편으로 출사. 하지만 격자에서 발산한 빛이 시야 전체를 범람시켜 **실패**
  First attempt: entrance slit S₁ → single pass → intermediate slit S₂ → return through system → exit at other half of entrance slit. Failed because divergent light from grating flooded the entire field of view
- 성공한 방식: 거울 M₂를 기울여(tilting) 빛을 시스템 평면 밖의 거울 쌍(M₃, M₄)으로 보내고, 두 번째 패스를 거쳐 출사 슬릿으로 돌아옴 — **가상 이미지(virtual images)가 없어** 완전한 배플링 가능
  Successful approach: tilting M₂ to send light to a mirror pair (M₃, M₄) off the system plane, returning through a second pass to the exit slit — **no virtual images**, enabling complete baffling

**격자 (Grating) / The Grating:**
- Horace W. Babcock 제작, 610 grooves/mm
  Ruled by Horace W. Babcock, 610 grooves/mm
- 알루미늄 코팅 블랭크 31cm 직경 × 5cm 두께
  Aluminum-coated blank 31-cm diameter × 5-cm thick
- 유효 ruled 영역: 25cm × 15cm ($N$ = 155,000)
  Ruled area: 25 cm × 15 cm ($N$ = 155,000)

**분해능 측정 / Resolution Measurements:**
- Single pass, 5차: 약 500,000 (직접 사진 촬영)
  Single pass, 5th order: ~500,000 (direct photography)
- Double pass: **600,000** (슬릿 폭이나 선 폭 보정 전)
  Double pass: **600,000** (uncorrected for slit width or line width)
- 광전자 주사(photoelectric scanning): Rayleigh 기준으로 single pass 775,000, double pass **1,550,000**에 매우 근접
  Photoelectric scanning: approaches Rayleigh values of 775,000 (single) and **1,550,000** (double pass)
- I₂ 흡수선(λ5328.904 Å)의 측정 분리: $a = 0.0090$ Å, $b = 0.0104$ Å, $c = 0.0100$ Å (5th order double pass, 30μ 슬릿, 310°K 아이오딘 셀)
  I₂ absorption line (λ5328.904 Å) measured separations: $a = 0.0090$ Å, $b = 0.0104$ Å, $c = 0.0100$ Å

**산란광 / Scattered Light:**
- 모든 고스트 구조의 적분 세기: 중심 피크의 약 5%
  Integrated intensity of all ghost structures: ~5% of central peak
- 일반 산란광: 8% (single pass에서)
  General scattered light: 8% (in single pass)
- **Double pass에서 총 산란광: 3%** ("shutter closed" 절대 영점 기준)
  **Total scattered light in double pass: 3%** (referred to absolute zero "shutter closed")
- 이것이 double-pass의 핵심 장점: 산란광이 제곱으로 줄어듦
  This is the key advantage of double-pass: scattered light reduces quadratically

### 8. 기계 시스템 / The Mechanical System

**분광기 회전 / Spectrograph Rotation:**
- 태양 표면의 흑점이나 plage에 대해 슬릿을 임의 방향으로 정렬해야 함
  Need to orient the slit at any angle relative to sunspots or plages on the solar surface
- 이미지 회전기(image rotator) 대신 **분광기 전체를 회전**시키는 방식 채택
  Instead of an image rotator, chose to **rotate the entire spectrograph**
- 이유: 대형 이미지 스케일에서의 추가 반사, 정렬 유지의 어려움, 24시간 시야 회전
  Reasons: additional reflections at large image scale, difficulty maintaining alignment, 24-h field rotation

**분광기 탱크 / Spectrograph Tank:**
- 직경 2m, 길이 21m의 강철 탱크, 무게 17 metric tons
  2-m diameter, 21-m long steel tank, 17 metric tons
- 바닥에 56cm 직경의 구면 캡(spherical cap)이 컵(cup)에 안착 → 베어링 역할
  Bottom carries a 56-cm diameter spherical cap resting in a cup → acts as bearing
- 동적 오일 필름(4 liters/min)이 75–125μ 간격을 유지하며 탱크의 무게를 부양
  Dynamic oil film (4 liters/min) maintains 75–125μ separation, floating the tank weight
- 탱크 상부의 링이 간단한 롤러로 지지 → 마찰 드라이브 제공
  Ring near top supported by simple rollers → provides friction drive

**격자 구동 / Grating Drive:**
- 격자 셀은 push-pull 나사로 홈(grooves)이 회전축에 정확히 평행하도록 조정 가능
  Grating cell adjustable by push-pull screws to align grooves exactly parallel to rotation axis
- 76.2cm 직경의 대형 알루미늄 스풀(spool)에 장착
  Mounted in a large 76.2-cm diameter aluminum spool
- 내부 기어가 격자 스풀을 보조 휠에 연결
  Internal gear couples grating spool to auxiliary wheels
- 정밀 스캔: 2.5cm 직경의 나사(20 threads/cm), 랩핑(lapping) 처리, 사파이어 thrust bearing
  Precision scan: 2.5-cm diameter screw (20 threads/cm), lapped, sapphire thrust bearing
- 두 개의 리본 테이프가 equalizer arm을 통해 일정한 장력 유지
  Two ribbon tapes maintain constant tension through an equalizer arm
- 격자 회전축에 미세 분할 원(finely divided circle) → 1 arcmin 단위로 시각적 설정 가능
  Finely divided circle on grating rotation axis → visual setting to 1 arcmin of arc

---

## 핵심 시사점 / Key Takeaways

1. **대형 태양 망원경의 핵심은 열 제어이다**: 70,000 리터의 냉각수, 30톤의 구리 외피, TiO₂ 도장 등 — 광학 설계만큼이나 열공학이 성능을 결정한다. 2분 만에 이미지 품질이 악화되는 문제를 해결하는 것이 가장 큰 도전이었다.
   **Thermal control is the key challenge of large solar telescopes**: 70,000 liters of coolant, 30 tons of copper sheeting, TiO₂ paint — thermal engineering determines performance as much as optical design.

2. **헬리오스탯 방식은 trade-off의 산물이다**: 시야 회전과 비최적 입사각이라는 단점에도 불구하고, 단일 반사면의 단순성, 높은 위치 배치, 일정한 편광 특성 때문에 선택되었다. 이 선택이 전체 기기의 구조(경사면 배치)를 결정했다.
   **The heliostat is a product of trade-offs**: despite field rotation and non-optimal incidence angles, it was chosen for simplicity, high placement, and constant polarization. This choice determined the entire instrument's structure.

3. **Quartz vs. Metal 거울 논쟁은 직관에 반한다**: 열팽창 계수가 극히 작은 quartz와 열전도도가 높은 metal이 Couder의 $K$ 계수에서 거의 동등하다. 이는 $K = \alpha\delta c/m$에서 $\alpha$와 $m$의 역상관 때문이다.
   **The quartz vs. metal mirror debate is counterintuitive**: extremely low thermal expansion (quartz) and high thermal conductivity (metal) yield nearly equal Couder $K$ factors.

4. **Double-pass 분광은 산란광을 제곱으로 줄인다**: single pass에서 8%였던 산란광이 double pass에서 3%로 감소. 이것은 Fraunhofer 선의 중심 세기(core intensity) 정밀 측정에 결정적이다.
   **Double-pass spectroscopy reduces scattered light quadratically**: from 8% (single pass) to 3% (double pass) — decisive for precision measurement of Fraunhofer line core intensities.

5. **분광기 설계는 역방향 논리를 따른다**: 원하는 분해능 → 필요한 격자 크기 → collimator 초점 거리 → 망원경의 $f$-ratio → 구경과 초점 거리. 최종 기기의 모든 사양이 과학 목표에서 역산된다.
   **Spectrograph design follows backward logic**: desired resolution → required grating size → collimator focal length → telescope $f$-ratio → aperture and focal length. Every specification is derived backward from the science goal.

6. **공기압 부양(pneumatic flotation)은 혁신적 해결책이다**: 55,000 kg의 헬리오스탯 마운팅에서 고도각에 따른 무게 보상을 기존의 복잡한 counterbalance 대신 공기압/진공 피스톤으로 해결한 것은 우아한 엔지니어링이다.
   **Pneumatic flotation is an elegant engineering solution**: using air pressure/vacuum proportional to altitude angle to support the 55,000-kg heliostat mounting, replacing complex counterbalance systems.

7. **분광기 전체를 회전시키는 결정은 대담하다**: 17톤의 강철 탱크를 오일 필름 위에 떠서 회전시키는 방식은 이미지 회전기의 추가 광학적 손실을 피하면서도 슬릿을 태양 표면의 임의 특징에 정렬할 수 있게 했다.
   **Rotating the entire spectrograph is a bold decision**: floating a 17-ton steel tank on an oil film avoids optical losses of an image rotator while allowing slit alignment to any solar feature.

---

## 수학적 요약 / Mathematical Summary

### 이미지 스케일 / Image Scale

$$s = \frac{f}{206265} \quad (\text{mm/arcsec})$$

McMath: $f \approx 90\text{m}$ → $s \approx 0.436$ mm/arcsec → 태양상 직경 ~84 cm ("약 1 yard")

### 회절 한계 분해능 / Diffraction-Limited Resolution

$$\theta = 1.22 \frac{\lambda}{D}$$

$D = 160$ cm, $\lambda = 550$ nm: $\theta \approx 0.084''$ (실제로는 seeing에 의해 ~1.5 arcsec)

### 격자 분해능 / Grating Resolving Power

$$R = mN$$

$m = 5$ (5th order), $N = 155{,}000$: $R_{\text{theoretical}} = 775{,}000$ (single), $1{,}550{,}000$ (double pass)
실측: $R \approx 500{,}000$ (single photo), $600{,}000$ (double photo)

### 격자 방정식 / Grating Equation

$$m\lambda = d(\sin\alpha + \sin\beta)$$

$d = 1/610$ mm (groove spacing), 선형 분산 약 7.5 mm/Å (5th order)

### 열 변형 계수 / Thermal Distortion Factor

$$K = \frac{\alpha \delta c}{m}$$

Fused quartz: $K = 5 \times 10^{-4}$ cgs, Metals: $K = 6 \times 10^{-4}$ cgs → 놀랍도록 유사

### 집광력 / Light-Gathering Power

분광기의 S/N ratio:
- Image slicer 미사용: $\text{S/N} \propto \sqrt{A}$ (구경의 제곱근)
- Image slicer 사용: $\text{S/N} \propto A$ (구경에 직접 비례)

---

## 역사적 맥락의 타임라인 / Paper in the Arc of History

```
1868 ── Janssen & Lockyer: spectroscopic observation of prominences
  │
1890s ─ Hale: spectroheliograph → Mt. Wilson solar towers
  │
1930s ─ Lyot: coronagraph (artificial eclipse)
  │
1940s ─ McMath-Hulbert Observatory (Michigan) — 선행 분광기 설계
  │         McMath-Hulbert vacuum spectrograph prototype
  │
1953 ── Palomar 200-inch quartz blanks donated to McMath
  │
1954 ── NSF ad hoc committee (Bowen, Goldberg, Stromgren, Struve,
  │       Whitford, McMath) → national solar observatory plan
  │
1958 ── Kitt Peak National Observatory established
  │
1962 Jan ── Robert R. McMath dies
  │
1962 Nov ── Telescope dedicated as McMath Solar Telescope
  │
★ 1964 ── Pierce publishes instrument paper ← 이 논문 / THIS PAPER
  │          Resolution 600,000 in double pass, 3% scattered light
  │
1969 ── Dunn: Vacuum Tower Telescope (Sacramento Peak)  ← Paper #2
  │       → evacuated path 방식으로 internal seeing 문제 해결
  │
1985 ── Renamed McMath-Pierce Solar Telescope
  │
2000s ─ FTS (Fourier Transform Spectrometer) 설치 → IR atlas 생산
  │
2013 ── McMath-Pierce decommissioned for DKIST construction
  │
2020 ── DKIST (4-m aperture) — 현대 최대 태양 망원경
```

---

## 다른 논문과의 연결 / Connections to Other Papers

| 논문 / Paper | 연결 / Connection |
|---|---|
| **#2 Dunn (1969) — Vacuum Tower Telescope** | Pierce가 해결하지 못한 internal seeing 문제를 진공 광로(evacuated path)로 근본적으로 해결. McMath의 수냉 외피 방식 vs. Dunn의 진공 방식은 두 가지 대조적 접근법 / Fundamentally solves the internal seeing problem with an evacuated path — contrasting approach to McMath's water-cooled enclosure |
| **#3 Swedish Solar Telescope** | McMath의 대형 구경 철학을 이어받되, adaptive optics로 seeing 한계를 극복. 회절 한계에 근접하는 성능 달성 / Inherits McMath's large-aperture philosophy but overcomes seeing limits with adaptive optics |
| **Babcock (1953) — Magnetograph** | Pierce가 McMath에서 Babcock magnetograph 사용(0.2 G sensitivity)을 언급. 자기장 측정이 대형 태양 분광기의 주요 활용처 / Pierce mentions Babcock magnetograph use at McMath; magnetic field measurement as a key application of large solar spectrographs |
| **Leighton (1960s) — 5-min oscillation** | Pierce가 도입부에서 언급. McMath의 높은 분광 분해능이 이러한 속도장 연구를 가능케 함 / Mentioned in the introduction; McMath's high spectral resolution enables such velocity field studies |
| **Solar Physics #10 Leighton (1969)** | McMath 망원경으로 관측된 태양 대기의 동적 현상(대류, 진동, 플레어)이 Leighton의 태양 자기장 모델의 관측적 기반 / Dynamic phenomena observed at McMath provide observational basis for Leighton's solar magnetic field models |

---

## 참고문헌 / References

- Pierce, A. K., "The McMath Solar Telescope of Kitt Peak National Observatory," *Applied Optics*, Vol. 3, No. 12, pp. 1337–1346, 1964. [DOI: 10.1364/AO.3.001337]
- McMath, R. R. & Mohler, O. C., "Telescope Driving Mechanisms," in *Stars and Stellar Systems, Vol. I: Telescopes*, eds. G. P. Kuiper & B. M. Middlehurst, Univ. of Chicago Press, 1960.
- Couder, A., cited in Pierce (1964) for mirror thermal distortion comparison factor $K$.
- Bowen, I. S., image slicer analysis for solar spectrographs, cited in Pierce (1964).
- Livingston, W. C., photoelectric seeing measurements at McMath, cited in Pierce (1964).
- Plymate, C., "A History of the McMath-Pierce Solar Telescope," in *Advanced Maui Optical and Space Surveillance Technologies Conference*, 2002.
