---
title: "An Evacuated Tower Telescope"
authors: Richard B. Dunn
year: 1964
journal: "Applied Optics, Vol. 3, No. 12, pp. 1353–1357"
topic: Solar Observation / Ground-Based Telescopes
tags: [vacuum telescope, tower telescope, internal seeing, turret, Sacramento Peak, Dunn Solar Telescope, solar spectroscopy, mercury float]
status: completed
date_started: 2026-04-10
date_completed: 2026-04-10
---

# An Evacuated Tower Telescope

## 핵심 기여 / Core Contribution

이 논문은 태양 망원경의 핵심 난제인 **internal seeing**(내부 공기 대류에 의한 이미지 열화)을 **광학 경로 전체를 진공으로 만드는** 방식으로 근본적으로 해결하는 새로운 설계를 제시한다. Sacramento Peak Observatory에 건설될 이 76cm 구경, 55m 초점 거리의 진공 타워 망원경은 헬리오스탯이나 코엘로스탯 대신 **altazimuth 회전 터렛(turret)**으로 태양을 추적하고, 전체 광학 계를 250μ(약 0.2 Torr)의 진공으로 유지한다. 230 metric ton의 내부 구조물을 11톤의 수은(mercury) 위에 부양시키는 독창적인 진동 격리 시스템, 그리고 브러시리스 토크 모터에 의한 정밀 회전 구동 등 혁신적 엔지니어링이 특징이다. 이 설계는 이후 모든 현대 고해상도 태양 타워 망원경의 표준 템플릿이 되었다.

This paper presents a fundamentally new design that solves **internal seeing** — image degradation from air convection inside the telescope — by **evacuating the entire optical path**. This 76-cm aperture, 55-m focal length vacuum tower telescope for Sacramento Peak Observatory uses an **altazimuth rotating turret** instead of a heliostat or coelostat, and maintains the entire optical system at 250μ (~0.2 Torr) vacuum. It features innovative engineering: a 230 metric-ton inner structure floated on 11 tons of mercury for vibration isolation, and precision rotation driven by brushless torque motors. This design became the standard template for all modern high-resolution solar tower telescopes.

---

## 읽기 노트 / Reading Notes

### 1. 설계 목표 / Design Objectives

Dunn은 새로운 태양 타워 망원경의 네 가지 설계 목표를 명확히 한다:

Dunn states four explicit design objectives:

1. **태양 가열에 의한 이미지 품질 저하를 줄일 것** — 망원경 내외부 표면의 태양 가열 효과를 최소화
   Reduce deleterious effects on image quality from solar heating of interior and exterior surfaces

2. **단순한 광학계, 최소 반사면, 조리개 변화 없음** — 하루 중 조리개가 변하지 않는 것이 중요
   Simple optical system with fewest possible reflecting surfaces, with no change in illumination of the aperture during the day

3. **유연한 가이딩** — 태양의 중심이나 시야의 중심을 기준으로 스캔 및 회전 가능
   Flexible guiding arrangement — scan and rotation about the center of the sun or field of view

4. **다수 기기 간 신속한 전환** — 빠르게 변하는 태양 현상의 다양한 측면을 관측
   Rapid and accurate switching between multiple instruments for different types of rapidly changing solar phenomena

이 목표들은 Pierce의 McMath 설계와 대조적이다. McMath는 **분광 분해능**과 **집광력**을 최적화한 반면, Dunn은 **이미지 품질**과 **운용 유연성**을 우선시했다.

These objectives contrast with Pierce's McMath design. McMath optimized for **spectral resolution** and **light-gathering power**, while Dunn prioritized **image quality** and **operational flexibility**.

### 2. 진공 시스템 / The Vacuum System

이 논문의 가장 핵심적인 기여는 진공 광로의 도입이다:

The paper's most critical contribution is the introduction of the evacuated optical path:

**진공 수준 / Vacuum Level:**
- 내부를 250μ의 수은주(mercury)까지 배기 — 이는 약 55 km 고도의 대기압에 해당
  Interior evacuated to 250μ of mercury — equivalent to atmospheric pressure at ~55 km altitude
- 비교: 25 km (80,000 ft)에서의 기압은 약 20 mm Hg, 풍선 망원경이 도달하는 고도
  For comparison: pressure at 25 km (balloon telescope altitude) is ~20 mm Hg
- 이 진공 수준에서 내부 난류의 광학적 효과가 **무시할 수 있는 수준(negligible)**으로 감소
  At this vacuum level, optical effects of internal turbulence are reduced to **negligible**

**배기 시스템 / Pumping System:**
- 2대의 blower와 diffusion pump + 짧은 배관 연결
  Two blowers and diffusion pumps with short pipe connections
- 8.5 m³/min (300 ft³/min)의 forepump 용량
  8.5 m³/min (300 ft³/min) forepump capacity
- $4.5 \times 10^2$ m³ ($1.6 \times 10^4$ ft³)의 망원경 체적을 약 3시간에 250μ까지 배기
  Evacuate the $4.5 \times 10^2$ m³ telescope volume to 250μ in approximately 3 hours
- 더 높은 진공도 가능하지만, blower와 diffusion pump 추가 및 배관 단축이 필요
  Higher vacuum possible but requires additional pumps and shorter pipe connections

**왜 진공인가? / Why Vacuum?**

Pierce의 McMath 논문에서 보았듯이, 대부분의 태양 망원경에서 돔을 열고 2분 후면 이미지가 악화된다. McMath는 이를 70,000리터 냉각수와 30톤 구리 외피로 완화하려 했지만, 공기 자체가 존재하는 한 대류는 완전히 제거할 수 없다. Dunn의 해결책은 근본적이다: **공기를 아예 없앤다**.

As seen in Pierce's McMath paper, in most solar telescopes image quality deteriorates within 2 minutes of opening the dome. McMath tried to mitigate this with 70,000 liters of coolant and 30 tons of copper sheeting, but convection cannot be fully eliminated while air exists. Dunn's solution is radical: **remove the air entirely**.

### 3. 광학 시스템 / Optical System

**거울 3개 + 창 2개의 단순한 구성 / Simple 3-mirror + 2-window configuration:**

- 모든 광학 소자는 투명 fused quartz로 제작
  All optics made from transparent fused quartz

- **입사창 (Entrance window)**: 두께 10 cm, quartz. 250 psi의 진공 하중과 $7.4 \times 10^6$ dyn cm⁻² (10.8 psi)의 진공 부하를 지탱. Sacramento Peak이 해발 2.8 km (9200 ft)에 있어 해수면보다 압력이 낮음
  Entrance window: 10-cm thick quartz, bearing 250 psi stress and $7.4 \times 10^6$ dyn cm⁻² vacuum load. Sacramento Peak at 2.8 km elevation reduces atmospheric pressure

- **구경 76 cm**: 최소 별 이미지 0.2 arcsec를 기록한 최소 구경. 이보다 크면 입사창이 지나치게 두꺼워짐
  76-cm aperture: smallest aperture recording a 0.2-arcsec star image. Larger would make entrance window excessively thick

- **주경 (Primary mirror)**: 단일 구면경(spherical), 초점 거리 55 m, 축(shaft) 바닥의 지면 관측실에서 태양상을 형성. 태양상 지름 약 51 cm, $f/72$
  Primary mirror: single spherical, focal length 55 m, forms solar image at ground-level instruments. Solar image diameter ~51 cm, $f/72$

- **초점 깊이(depth of focus)**: 0.76 cm. 주경은 공칭 위치에서 위로 4.9 m, 아래로 2.4 m 초점 이동 가능 — 출사창에 장착된 소구경 렌즈 교환으로 $f$-ratio를 변경할 수 있음
  Depth of focus: 0.76 cm. Primary can be focused 4.9 m up and 2.4 m down from nominal — small-aperture lenses on exit windows allow changing $f$-ratio

- **비네팅 없는 전체 태양 디스크**: 비네팅 없이 전체 태양 디스크를 촬상하려면 거울 지름이 163 cm이어야 함 (실제 주경은 이보다 작으므로 가장자리에서 약간의 비네팅 있음)
  Unvignetted full disk: requires 163-cm mirror. The actual primary is smaller, so slight vignetting at edges

- **수차 / Aberrations**: 주경을 0.0111 rad 기울여 보조 기기로 빛을 보내면 0.69 cm의 비점수차(astigmatism) 도입. 출사창의 $4.8 \times 10^4$ cm 초점 거리의 cylindrical lens (180 cm from focus)가 이를 보정. 빔 폭이 각 점에서 2.5 cm에 불과하므로 실린더 렌즈 제작이 용이. 구면수차(spherical aberration)는 무시 가능, 코마(coma)의 이미지 길이는 0.1 arcsec 미만
  Tilting primary 0.0111 rad to send light to auxiliary instruments introduces 0.69 cm astigmatism. Corrected by a $4.8 \times 10^4$ cm focal-length cylindrical lens on exit windows. Spherical aberration negligible, comatic image length < 0.1 arcsec

### 4. 타워 구조 / Tower Structure

**콘크리트 타워 / Concrete Tower:**
- 팔각형(octagon) 기초, 바닥면 17 m
  Octagonal base, 17 m across the flats
- 꼭대기(40 m 높이)에서 직경 1.5 m까지 균일하게 테이퍼(taper)
  Uniform taper to 1.5-m diameter at 40-m elevation
- 북면은 엘리베이터를 수용하기 위해 변형. 36 m 높이에 터렛 정비용 문
  North side distorted for elevator. Doors at 36 m for turret service
- 높이 41.5 m — 내부 walkway와 기기 튜브를 수용할 만큼 높음
  Height 41.5 m — just high enough for internal walkway and instrument tubes

**왜 콘크리트인가? / Why Concrete?**
- 높은 질량 → 좋은 감쇠(damping)
  High mass → good damping
- 저비용
  Low cost
- 벽 두께: 위에서 아래까지 일정한 91 cm
  Wall thickness: constant 91 cm top to bottom
- 동등한 강성을 위한 강철 벽 두께: 7.6 cm — 콘크리트가 훨씬 효과적
  Steel wall for equivalent rigidity: 7.6 cm — concrete far more effective
- 바람막이(windscreen) 없음! 단일 두꺼운 벽 타워가 충분히 안정적. 두 벽(타워 + windscreen) 사이의 진동 결합이 오히려 더 불안정
  No windscreen! Single thick-walled tower sufficiently stable. Vibrational coupling between two walls (tower + windscreen) would be worse

**지반 / Foundation:**
- 파쇄 석회암(fractured limestone) 기반
  Fractured limestone foundation
- 기반 압축 탄성률: $2.7 \times 10^{10}$ dyn cm⁻² (390,000 psi)
  Compression modulus: $2.7 \times 10^{10}$ dyn cm⁻² (390,000 psi)
- 비교란 지반의 고유 주파수: 120 cps, 감쇠비 0.06
  Undisturbed material frequency: 120 cps, damping ratio 0.06
- 이론적 타워 상부 진동: 7 cps 주파수, 13.5 m/sec (30 mph) 바람에서 진폭 0.2 arcsec 미만
  Theoretical tower-top vibration: 7 cps, amplitude < 0.2 arcsec in 13.5 m/sec wind

**열 제어 / Thermal Control:**
- 외표면: TiO₂ 흰색 도장 (McMath와 동일한 기법)
  Exterior: TiO₂ white paint (same technique as McMath)
- 콘크리트 내부에 수직 냉각 파이프를 3단(tier)으로 매설
  Vertical cooling pipes embedded in concrete in three tiers
- 파이프 위치: 외표면에서 7.6 cm 안쪽, 간격 10–25 cm (높이에 따라)
  Pipe location: 7.6 cm below outer surface, spaced 10–25 cm depending on elevation
- 표면을 주위 온도보다 **5°C 이상 낮게** 냉각 가능
  Surface can be cooled to **≥5°C below** ambient temperature
- 엘리베이터 문, 상부 금속 부품도 모두 냉각
  Elevator doors and all metal parts near top also cooled

### 5. 터렛 (Turret) — 가장 혁신적인 요소 / The Turret — Most Innovative Element

이것이 이 논문에서 가장 독창적인 부분이다. Dunn은 헬리오스탯도 코엘로스탯도 아닌 전혀 새로운 빔 유도 방식을 고안했다:

This is the most original part of the paper. Dunn devised a beam-steering method that is neither a heliostat nor a coelostat:

**구성 / Configuration (Fig. 2):**
- 두 개의 평면 거울이 터렛 내부에 장착
  Two flat mirrors mounted inside the turret
- 거울은 수은 링(mercury rings)에 의해 **횡방향으로 구속**, 9점 flexural 서스펜션으로 **축방향 지지**
  Mirrors **laterally confined** by mercury rings, **axially supported** by 9-point flexural suspension
- 태양 가열에 의한 convex 변형 방지를 위해 거울 뒷면과 앞면의 비조사 초승달(crescent) 부분에 히터 부착 — 태양 에너지를 상쇄하여 **열 보상(thermal compensation)**
  Heaters on mirror backs and unilluminated front crescents to compensate solar energy — **thermal compensation** to prevent convex distortion
- Altazimuth 좌표로 추적 — 극축(polar axis)과는 무관
  Tracks in altazimuth coordinates — no relationship to polar angle
- "전함의 포탑(battleship turret)"에 비유됨
  Likened to a "battleship turret"

**베어링 및 구동 / Bearings and Drive:**
- 각 좌표(방위각, 고도각)에 3개의 롤러가 4,536 kg (10,000 lb)의 추력을 지지 — 진공 하중에서 발생
  Three rollers per coordinate support 4,536 kg (10,000 lb) thrust from vacuum load
- 각 롤러에 150 cm-kg (11 ft-lb) DC 서보모터가 직결
  150 cm-kg (11 ft-lb) DC servomotor direct-coupled to each roller
- 롤러 감속비 약 11:1 → 실제 토크 $4.5 \times 10^3$ cm-kg (330 ft-lb) — 얼음 깨기와 가속에 충분
  Roller reduction ~11:1 → effective torque $4.5 \times 10^3$ cm-kg — sufficient for shearing ice and acceleration
- 서보 강성: $1.4 \times 10^4$ cm-kg (1000 ft-lb) per arcsec → 터렛을 0.1 arcsec까지 가이딩
  Servo rigidity: $1.4 \times 10^4$ cm-kg per arcsec → guides turret to 0.1 arcsec
- 서보 속도 상수: 1000, 교차 주파수(crossover frequency): 10 cps
  Servo velocity constant: 1000, crossover frequency: 10 cps

**밀봉 / Sealing:**
- 방위각 밀봉: 수은(mercury seal)
  Azimuth seal: mercury
- 고도각 밀봉: Teflon + 고무 O-링
  Elevation seal: machined Teflon + rubber O-rings
- Grease-filled 래비린스 밀봉이 구동부를 방수
  Grease-filled labyrinth seals weatherproof the drives
- 닫힘 상태: 창 뒤의 커버가 터렛 방위각 부분에 연결되어 회전
  Closed: window cover rotates attached to turret azimuth part

**Sunseeker:**
- 양 좌표 모두에 sunseeker 장착 — 하늘 어디서든 태양 포착 가능
  Sunseekers in both coordinates — acquire sun in any part of sky
- 180° 감시 TV 카메라 + 보호 필터로 하늘 상태 관찰
  180° surveillance TV camera with protective filters for sky monitoring

### 6. 진동 격리와 수은 부양 / Vibration Isolation and Mercury Float

이 논문의 두 번째로 독창적인 엔지니어링:

The second most innovative engineering in this paper:

**문제 / Problem:**
타워 진동이 분광기에 전달되면 안 됨. 또한 사람이 걸어 다니면 테이블이 흔들림.

Tower vibrations must not reach spectrographs. Also, personnel walking shakes the table.

**해결책: 수은 부양 (Mercury Float):**
- 전체 내부 강철 구조물(230 metric ton)이 3.7 m 직경의 수은 부양체(float) 위에 지지
  Entire inner steel structure (230 metric tons) supported on a 3.7-m diameter mercury float
- 수은 깊이 1.8 m, 약 **11 metric ton**의 수은 사용
  Mercury depth 1.8 m, approximately **11 metric tons** of mercury
- 부양체와 타워 사이 간극: 약 1 cm — 수은이 이 간극을 채움
  Gap between float and tower: ~1 cm, filled with mercury
- 부양체는 타워에서 3개의 봉(rod)으로 매달려 있고, 스프링-질량-댐퍼 시스템을 형성
  Float suspended from tower by three rods, forming spring-mass-damper system
- 고유 주파수: 0.5 cps부터 더 높은 주파수까지 조정 가능
  Resonant frequency adjustable from 0.5 cps to higher frequencies
- 결과: 타워 진동이 분광기에 전달되지 않음
  Result: tower vibrations do not reach spectrographs

**테이블 처짐 / Table Deflection:**
- 테이블 가장자리에서 사람이 걸으면 이론적 처짐: 1 arcsec
  Theoretical deflection from person walking on table edge: 1 arcsec
- 하부 베어링과 상부 베어링이 이 모멘트를 상쇄
  Upper and lower bearings counteract this moment

**수은의 다른 역할들 / Other Roles of Mercury:**
- 터렛의 무게도 내부 수은 위에 부양 → 방위각 롤러의 마찰 감소
  Turret weight also floated on mercury inside → reduces friction on azimuth rollers
- 방위각 밀봉에도 수은 사용
  Mercury used for azimuth sealing as well
- 터렛 질량 부양체(turret mass float)가 방위각 튜브 상단에서 별도로 부양
  Turret mass float decoupled from azimuth turret at top of tube

### 7. 기기 회전 시스템 / Instrument Rotation System

**중앙 진공 튜브 / Central Vacuum Tube:**
- 직경 1.2 m의 배기된 강철 튜브
  1.2-m diameter evacuated steel tube
- 테이블 위: 직경 1.2 m → 테이블 아래(지면 이하): 직경 3 m로 확장
  Above table: 1.2 m → below table (below ground): expands to 3 m diameter
- 보조 기기는 직경 1.5 m, 길이 21 m의 수직 진공 튜브 3개에 수용 + 21 m 수직 광학 벤치 2개
  Auxiliary instruments in three vertical vacuum tubes (1.5 m diam, 21 m long) + two 21-m vertical optical benches
- 12 m 직경 작업 테이블에 수평 벤치 4개 추가
  Four horizontal benches on 12-m diameter work table

**튜브 회전 / Tube Rotation:**
- $1.4 \times 10^6$ cm-kg (10,000 ft-lb)의 브러시리스, 직접 구동, DC 토크 모터로 회전
  Rotated by $1.4 \times 10^6$ cm-kg brushless, direct-drive, DC torque motor
- 최고 속도 1 rpm
  Top speed: 1 rpm
- 모터 설계: 자속이 극에서 극으로 정현파적으로 변화. 토크 출력이 회전각의 sin²에 비례하는 권선 + cos²에 비례하는 권선 → 합이 항상 일정 → **리플(ripple) 제거**
  Motor design: sinusoidal flux variation pole-to-pole. sin² + cos² windings → constant total torque → **ripple eliminated**
- 모터를 분절(segment)로 조립 가능 → 튜브 주위에 현장 조립
  Motor assembled in segments around the tube
- 정류자(commutator) 없음 → 브러시리스
  No machined commutator → brushless

**420개 슬립 링 / 420 Slip Rings:**
- 부양체 위의 가스 충전 외피에 전력 및 제어 슬립 링 420개
  420 power and control slip rings in gas-filled enclosure above the float
- 수은을 채운 RTV(상온 가황 고무) 홈에 담긴 저소음, 저마찰 슬립 링
  Low-noise, low-friction slip rings in mercury-filled RTV troughs

### 8. 가이딩 시스템 / Guiding System

**주 가이더 / Main Guider:**
- 주 망원경 빔의 중앙 3인치(7.62 cm)를 분리하여 21 m 레벨의 광학 검출기에 결상
  Splits central 3-inch (7.62 cm) of main beam onto optical detector at 21-m level
- 태양 대향 양측에서 빛을 420 cps로 초퍼(chopping) → 오차 신호 생성
  Light from opposite sides of sun chopped at 420 cps → error signal
- 피라미드형 빔 분할기, 광전자 증배관(photomultiplier), 자동 dynode 제어
  Pyramid beam splitter, photomultiplier, automatic dynode control
- 밝은 별과 행성/달의 limb crescent에서도 가이딩 가능
  Can guide on bright stars and planetary/lunar crescents

**오차 검출기 회전 / Error Detector Rotation:**
- 오차 검출기의 좌표가 터렛과 함께 회전하여 항상 정렬 유지
  Error detector coordinates rotate with turret to remain aligned
- 직교 좌표로 이동하는 서브테이블 위에 장착 → 이미지 회전 제어 가능
  Mounted on a subtable moving in rectilinear coordinates → image rotation control

**H-α 가이더:**
- 빔 분할기가 두 번째 광학 경로로 빛을 보내 H-α 복굴절 필터(birefringent filter)에 결상
  Beam splitter sends light to a second optical path for H-α birefringent filter
- 이미지 크기 1.3 cm로 전체 태양 디스크 2배 이상 시야
  Image size 1.3 cm — covers more than two solar diameters
- 십자선(crosswires)으로 주 망원경을 태양의 특정 세부 구조에 정밀 정렬
  Crosswires permit precise alignment of main telescope on particular solar detail

**synchro 시스템:**
- Synchro + 기계적 좌표 변환기(mechanical coordinate converter)가 테이블 회전 서보와 터렛의 오차 정보를 동시 도출
  Synchro + mechanical coordinate converter derives error information for both table rotation servo and turret
- 주 테이블은 20 arcsec, 태양 위 위치는 0.1 arcsec 이내로 가이딩
  Main table guided to 20 arcsec, position on sun to within 0.1 arcsec

### 9. 보조 기기 / Auxiliary Instruments

**사진 분광기 / Photographic Spectrograph:**
- 12 m 초점 거리의 전반사(all-mirror) echelle 분광기
  12-m focal-length all-mirror echelle spectrograph
- 5.3 m 초점 거리의 quartz-prism monochromator로 사전 분산
  Predispersed by 5.3-m focal-length quartz-prism monochromator
- 8인치(12.32 cm) 긴 슬릿으로 개별 파장을 분리하여 여러 회절 차수의 스펙트럼선을 동시 촬영
  Eight-inch long slits isolate individual wavelengths for simultaneous photography in different echelle orders
- 분광태양사진기(spectroheliograph)로도 변환 가능
  Also converts to spectroheliograph

**광전 기기 / Photoelectric Instrument:**
- 18 m 초점 거리의 double- 또는 single-pass 기기
  18-m focal-length double- or single-pass instrument
- 일정 편차(constant-deviation) quartz-prism predisperser
  Constant-deviation quartz-prism predisperser
- 격자 회전 서보가 monochromator prism을 자동으로 대형 기기의 파장에 맞춤
  Grating rotation servo automatically keeps monochromator prism adjusted to wavelength of larger instrument

**플레어 순찰 망원경 / Flare Patrol Telescope:**
- 20 cm 구경, 부양체 위 21 m 높이
  20-cm aperture, located 21 m above ground on the float
- 헬리오스탯으로 구동, 완전 자동
  Fed by heliostat, completely automatic
- 더 높은 곳의 seeing을 활용
  Takes advantage of higher-quality seeing at elevation

### 10. 요약과 Trade-offs / Summary and Trade-offs

Dunn은 논문 말미에서 설계의 장단점을 솔직하게 정리한다:

Dunn honestly lists the design's advantages and disadvantages at the end:

**장점 / Advantages:**
- 단순한 광학계 + 환경 제어(진공) → 내부 난류 제거
  Simple optics + environmental control (vacuum) → internal turbulence eliminated
- 이미지 회전 완전 제어
  Complete control of image motion
- 다수 보조 기기 수용
  Multiplicity of auxiliary instruments

**단점 / Disadvantages:**
- Altazimuth → 서보 제어 필수 (극축식처럼 단순 시계 구동 불가)
  Altazimuth → servo control required (not simple clock drive like equatorial)
- 터렛 거울이 45°에 배치 → **일상 관측 중 편광(polarization) 변화** 도입 (자기장 관측 시 보정 필요)
  Turret mirrors at 45° → **daily polarization variation** (compensation needed for magnetic field observations)
- 거울 가열이나 구부러짐 → 비점수차(astigmatism)
  Mirror heating or bending → astigmatism
- 고품질 입사창 필요
  High-quality entrance windows required
- 대형 질량체 회전 → 이미지 회전 제거에 필요
  Rotation of large mass required to eliminate image rotation
- 긴 기기는 반드시 수직이어야 함
  Longer instruments must be vertical

"대부분의 이 문제들은 설계 과정에서 해결되었다. 어떤 것도 기기의 효용을 심각하게 제한하지 않는다."

"Most of these problems have been solved in the design. None of them seriously limits the effectiveness of the instrument."

---

## 핵심 시사점 / Key Takeaways

1. **진공은 internal seeing의 근본적 해결책이다**: McMath의 수냉 외피가 완화(mitigation)였다면, Dunn의 진공은 제거(elimination)이다. 250μ의 진공은 55 km 고도에 해당하며, 이 수준에서 내부 난류의 광학적 효과는 무시 가능하다.
   **Vacuum is the fundamental solution to internal seeing**: McMath's water-cooling was mitigation; Dunn's vacuum is elimination. 250μ vacuum is equivalent to 55-km altitude.

2. **구경과 이미지 품질은 별개의 문제이다**: McMath(160 cm)의 이론적 분해능이 Dunn(76 cm)보다 2배 좋지만, seeing에 의해 둘 다 ~1.5 arcsec로 제한된다. Dunn은 구경을 줄이는 대신 **실제로 달성 가능한 분해능**을 최적화했다.
   **Aperture and image quality are separate problems**: McMath's theoretical resolution is 2× better, but both are seeing-limited to ~1.5 arcsec. Dunn optimized for **actually achievable resolution**.

3. **터렛은 헬리오스탯의 모든 단점을 해결한다**: 시야 회전 보상, 일정한 빔 방향, 하늘 어디든 추적 가능. 대가는 altazimuth 서보 제어의 복잡성과 편광 변화이다.
   **The turret solves all heliostat disadvantages**: field rotation compensation, constant beam direction, tracking anywhere in sky. The cost is altazimuth servo complexity and polarization variation.

4. **11톤의 수은 위에 230톤을 띄우는 것은 우아한 엔지니어링이다**: 이 진동 격리 시스템은 타워 진동과 관측 기기를 완전히 분리한다. 스프링-질량-댐퍼의 고유 주파수를 0.5 cps까지 낮출 수 있어 거의 모든 진동을 차단한다.
   **Floating 230 tons on 11 tons of mercury is elegant engineering**: this vibration isolation completely decouples tower vibrations from instruments, with resonant frequency adjustable down to 0.5 cps.

5. **입사창은 진공 망원경의 아킬레스건이다**: 76 cm 구경에서 이미 10 cm 두께가 필요하며, 구경이 커지면 창 두께가 비현실적으로 증가한다. 이것이 진공 망원경의 구경 상한을 결정하는 핵심 제약이다 (DKIST가 4m 구경에서 진공 대신 active cooling을 선택한 이유).
   **The entrance window is the Achilles' heel of vacuum telescopes**: at 76 cm, a 10-cm thick window is already needed. Larger apertures require impractically thick windows — this is the fundamental constraint limiting vacuum telescope size (why DKIST chose active cooling over vacuum at 4 m).

6. **브러시리스 토크 모터의 리플 제거 설계가 인상적이다**: sin² + cos² 권선의 조합으로 총 토크가 회전각에 무관하게 일정한 모터를 설계한 것은 정밀 천문 기기에 필수적인 혁신이다.
   **The brushless torque motor with ripple elimination is impressive**: sin² + cos² windings producing rotation-angle-independent torque is an essential innovation for precision astronomical instruments.

7. **Dunn은 Pierce와 같은 호에 실리면서 두 가지 철학을 대조시켰다**: McMath = "더 크게, 더 많은 빛" vs. Dunn = "더 깨끗하게, 더 높은 이미지 품질". 역사는 Dunn의 접근법이 고해상도 태양 관측에 더 효과적이었음을 증명했다.
   **Dunn contrasted two philosophies in the same journal issue as Pierce**: McMath = "bigger, more light" vs. Dunn = "cleaner, better image quality." History proved Dunn's approach more effective for high-resolution solar observation.

---

## 수학적 요약 / Mathematical Summary

### 이미지 스케일 / Image Scale

$$s = \frac{f}{206265} = \frac{55000}{206265} \approx 0.267 \text{ mm/arcsec}$$

태양상 지름: $32' \times 60 \times 0.267 \approx 512$ mm ≈ 51 cm

### 회절 한계 분해능 / Diffraction-Limited Resolution

$$\theta = 1.22\frac{\lambda}{D} = 1.22 \times \frac{550 \times 10^{-7}}{76} \times 206265 \approx 0.18''$$

실제 최소 별 이미지: 0.2 arcsec → 회절 한계에 근접!

### 진공 하중 / Vacuum Load

$$F = P_{\text{atm}} \times A$$

입사창 (76 cm 직경): $F = 7.4 \times 10^6 \text{ dyn/cm}^2 \times \pi \times 38^2 \approx 3.4 \times 10^{10}$ dyn ≈ 34,000 N ≈ 3.5 톤

### 타워 진동 / Tower Vibration

고유 주파수 7 cps, 13.5 m/sec 바람에서 진폭 < 0.2 arcsec
서보에 의해 부분 보정 → 잔차 허용 가능 (30 mph 바람에서는 seeing도 나쁨)

### 수은 부양체 부력 / Mercury Float Buoyancy

$$F_{\text{buoyancy}} = \rho_{\text{Hg}} \times g \times V_{\text{submerged}}$$

$\rho_{\text{Hg}} = 13,600$ kg/m³. 230 ton을 부양하려면 약 17 m³의 수은 — 실제로는 부분 부양 + 봉(rod) 서스펜션 조합

---

## 역사적 맥락의 타임라인 / Paper in the Arc of History

```
1890s ── Hale: open-air solar tower at Mt. Wilson
  │
1940s ── McMath-Hulbert Observatory: precursor spectrograph designs
  │
1954 ── NSF committee → national solar observatory plan
  │
1958 ── Sacramento Peak — Air Force Cambridge Research Labs
  │       Dunn joins, begins small-aperture flare patrol observations
  │
1960 ── Pierce: McMath telescope design → approach A (water-cooled)
  │
★ 1964 ── Dunn: "An Evacuated Tower Telescope" ← 이 논문 / THIS PAPER
  │       → approach B (vacuum), 76 cm, turret, mercury float
  │       (same issue as Pierce's McMath paper!)
  │
1964 ── Pierce: McMath instrument paper (Applied Optics, same issue)
  │
1969 ── Dunn: "Sacramento Peak's New Solar Telescope" (Sky & Tel)
  │       → construction completed, first observations
  │
1985 ── Renamed "Dunn Solar Telescope" (DST)
  │
1990s ─ DST + adaptive optics experiments → proves AO for solar
  │
2002 ── Swedish Solar Telescope (SST, 1m) — vacuum + AO   ← Paper #3
  │       → Dunn's vacuum concept + modern AO = 0.1 arcsec resolution
  │
2013 ── McMath-Pierce decommissioned
  │
2020 ── DKIST (4m) — NOT vacuum (too large for window)
  │       but inherits: turret concept, active thermal control, AO
  │       → shows Dunn's window-size limit was correct
```

---

## 다른 논문과의 연결 / Connections to Other Papers

| 논문 / Paper | 연결 / Connection |
|---|---|
| **#1 Pierce (1964) — McMath Solar Telescope** | 같은 호에 실린 대조적 접근법. McMath: 수냉 외피, 160 cm, 분광 최적화. Dunn: 진공, 76 cm, 이미지 품질 최적화. Pierce가 해결하지 못한 internal seeing을 Dunn이 근본적으로 해결 / Contrasting approach in the same issue. Pierce's unsolved internal seeing problem is fundamentally solved by Dunn |
| **#3 Swedish Solar Telescope (2002)** | Dunn의 진공 설계를 계승하면서 adaptive optics를 추가. 1m 구경에서 회절 한계(0.1 arcsec)에 근접하는 성능 달성 — Dunn의 설계 철학의 궁극적 검증 / Inherits Dunn's vacuum design + adds adaptive optics, achieving near-diffraction-limited performance at 1 m aperture |
| **DKIST (2020)** | 4m 구경에서 Dunn의 입사창 제약을 직면. 진공 대신 active cooling 선택. 하지만 turret 개념, 서보 가이딩, 열 제어의 기본 원리는 Dunn에서 계승 / Faces Dunn's entrance window constraint at 4 m. Chooses active cooling over vacuum, but turret concept and thermal control principles inherited from Dunn |
| **Dunn (1959) — Astrophys. J. 130, 972** | Dunn의 이전 논문. Sacramento Peak에서의 고해상도 사진 관측 경험이 이 망원경 설계의 동기가 됨 / Dunn's earlier paper; high-resolution photography experience at Sacramento Peak motivated this telescope design |

---

## 참고문헌 / References

- Dunn, R. B., "An Evacuated Tower Telescope," *Applied Optics*, Vol. 3, No. 12, pp. 1353–1357, 1964. [DOI: 10.1364/AO.3.001353]
- Pierce, A. K., "The McMath Solar Telescope of Kitt Peak National Observatory," *Applied Optics*, Vol. 3, No. 12, pp. 1337–1346, 1964.
- McMath, R. R. & Pierce, A. K., *Sky and Telescope* **20**, 64, 1960.
- McMath, R. R. & Pierce, A. K., *Sky and Telescope* **27**, 132, 1964.
- Dunn, R. B., *Astrophys. J.* **130**, 972, 1959.
- McIntosh, P. S., *Sky and Telescope* **27**, 280, 1964.
- Couder, A., *Vistas in Astronomy*, Vol. 1, A. Beer, ed. (Pergamon, 1955), pp. 372–377.
- Evans, J. W. & Waddell, J., *Appl. Opt.* **1**, 111, 1962.
