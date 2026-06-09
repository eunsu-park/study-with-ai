# Space Weather Paper Reading List

A curated list of landmark papers in Space Weather, organized by conceptual flow.
Each paper builds on the concepts introduced by earlier ones.

우주기상 분야의 주요 논문을 개념 흐름에 따라 정리한 목록입니다.
각 논문은 이전 논문에서 소개된 개념을 바탕으로 합니다.

---

## Phase 1: Early Theory & Geomagnetic Observations / 초기 이론 및 지자기 관측 (1908–1940)

### 1. The Norwegian Aurora Polaris Expedition 1902–1903
- **Authors**: Kristian Birkeland
- **Year**: 1908
- **DOI**: NO_DOI
- **Why it matters**: First systematic expedition to study the aurora borealis and its connection to electric currents in the polar atmosphere. Birkeland proposed that auroras are caused by charged particles from the Sun guided along Earth's magnetic field lines — a revolutionary idea confirmed decades later. 오로라와 극지 대기 전류의 관계를 최초로 체계적으로 연구한 탐험. Birkeland은 오로라가 태양에서 방출된 하전 입자가 지구 자기장 선을 따라 유도되어 발생한다고 제안했으며, 이 혁명적 아이디어는 수십 년 후 확인되었다.
- **Prerequisites**: Basic electromagnetism (전자기학 기초), Earth's magnetic field structure (지구 자기장 구조)
- **Status**: [x]

### 2. A New Theory of Magnetic Storms, Part I: The Initial Phase
- **Authors**: Sydney Chapman, Vincenzo C.A. Ferraro
- **Year**: 1931–1933
- **DOI**: 10.1029/TE036i002p00077
- **Why it matters**: Proposed that magnetic storms are caused by streams of charged particles (plasma) from the Sun compressing Earth's magnetic field, forming a temporary cavity (now called the magnetosphere). This was the first quantitative theory of solar-terrestrial interaction. 자기폭풍이 태양에서 온 하전 입자(플라즈마) 흐름이 지구 자기장을 압축하여 임시 공동(현재 자기권이라 불림)을 형성함으로써 발생한다고 제안. 태양-지구 상호작용의 최초 정량적 이론.
- **Prerequisites**: Paper #1; classical electrodynamics (고전 전기역학), plasma physics basics (플라즈마 물리학 기초)
- **Status**: [x]

### 3. Geomagnetism (monograph)
- **Authors**: Sydney Chapman, Julius Bartels
- **Year**: 1940
- **DOI**: NO_DOI
- **Why it matters**: The definitive reference on geomagnetism for decades. Systematized geomagnetic indices (Dst, Kp), established statistical methods for analyzing geomagnetic activity, and laid the mathematical framework for understanding Earth's internal and external magnetic fields. 수십 년간 지자기학의 결정적 참고 문헌. 지자기 지수(Dst, Kp)를 체계화하고, 지자기 활동 분석을 위한 통계적 방법을 확립했으며, 지구의 내부 및 외부 자기장 이해를 위한 수학적 틀을 마련.
- **Prerequisites**: Papers #1–2; spherical harmonics (구면 조화 함수), time-series analysis basics (시계열 분석 기초)
- **Status**: [x]

---

## Phase 2: Magnetosphere Discovery & Radiation Belts / 자기권 발견 및 방사선대 (1957–1966)

### 4. Dynamics of the Interplanetary Gas and Magnetic Fields
- **Authors**: Eugene Parker
- **Year**: 1958
- **DOI**: 10.1086/146579
- **Why it matters**: Predicted the existence of the solar wind — a continuous supersonic flow of plasma from the Sun. This was initially controversial but soon confirmed by spacecraft observations. The solar wind is the medium through which all space weather disturbances propagate. 태양풍의 존재를 예측 — 태양에서 나오는 초음속 플라즈마의 연속적 흐름. 초기에는 논란이 있었으나 곧 우주선 관측으로 확인. 태양풍은 모든 우주기상 교란이 전파되는 매질.
- **Prerequisites**: Papers #1–2; fluid dynamics (유체역학), MHD basics (자기유체역학 기초)
- **Status**: [x]

### 5. Observation of High Intensity Radiation by Satellites 1958 Alpha and Gamma
- **Authors**: James Van Allen, Carl McIlwain, George Ludwig
- **Year**: 1958
- **DOI**: 10.2514/8.7396
- **Why it matters**: Discovered the Van Allen radiation belts — zones of energetic charged particles trapped by Earth's magnetic field. This was one of the first major discoveries of the Space Age and demonstrated that near-Earth space is a hazardous radiation environment. Van Allen 방사선대 발견 — 지구 자기장에 갇힌 고에너지 하전 입자 영역. 우주 시대의 첫 번째 주요 발견 중 하나이며, 지구 근처 우주가 위험한 방사선 환경임을 입증.
- **Prerequisites**: Papers #2–3; charged particle motion in magnetic fields (자기장 내 하전 입자 운동)
- **Status**: [x]

### 6. Interplanetary Magnetic Field and the Auroral Zones
- **Authors**: James Dungey
- **Year**: 1961
- **DOI**: 10.1103/PhysRevLett.6.47
- **Why it matters**: Proposed magnetic reconnection as the mechanism coupling the solar wind to Earth's magnetosphere. When the interplanetary magnetic field (IMF) has a southward component, it reconnects with Earth's field, opening the magnetosphere to solar wind energy. This is the single most important concept in magnetospheric physics. 자기 재결합(magnetic reconnection)을 태양풍과 지구 자기권을 결합하는 메커니즘으로 제안. 행성간 자기장(IMF)이 남향 성분을 가질 때 지구 자기장과 재결합하여 자기권에 태양풍 에너지가 유입. 자기권 물리학에서 가장 중요한 단일 개념.
- **Prerequisites**: Papers #2, #4; magnetic reconnection basics (자기 재결합 기초), MHD (자기유체역학)
- **Status**: [x]

### 7. A Unifying Theory of High-Latitude Geophysical Phenomena and Geomagnetic Storms
- **Authors**: W. Ian Axford, Colin O. Hines
- **Year**: 1961
- **DOI**: 10.1139/p61-172
- **Why it matters**: Proposed viscous interaction between the solar wind and magnetosphere as a driver of magnetospheric convection. Together with Dungey's reconnection model, this paper established the two fundamental paradigms for how energy enters the magnetosphere. 태양풍과 자기권 사이의 점성 상호작용을 자기권 대류의 구동력으로 제안. Dungey의 재결합 모델과 함께 에너지가 자기권에 유입되는 두 가지 근본적 패러다임을 확립.
- **Prerequisites**: Papers #4, #6; fluid dynamics (유체역학), convection concepts (대류 개념)
- **Status**: [x]

### 8. The Development of the Auroral Substorm
- **Authors**: Syun-Ichi Akasofu
- **Year**: 1964
- **DOI**: 10.1016/0032-0633(64)90151-5
- **Why it matters**: Defined the auroral substorm as a distinct, repeatable sequence of auroral morphological changes. Introduced the concept of substorm phases (growth, expansion, recovery) based on all-sky camera data. This phenomenological framework remains foundational for substorm research. 오로라 substorm을 뚜렷하고 반복 가능한 오로라 형태 변화 시퀀스로 정의. 전천 카메라 데이터를 기반으로 substorm 단계(성장, 팽창, 회복)의 개념을 도입. 이 현상학적 틀은 substorm 연구의 기초로 남아 있음.
- **Prerequisites**: Papers #5–7; auroral physics basics (오로라 물리학 기초)
- **Status**: [x]

### 9. The Earth's Magnetic Tail
- **Authors**: Norman Ness
- **Year**: 1965
- **DOI**: 10.1029/JZ070i013p02989
- **Why it matters**: Used IMP-1 satellite data to confirm the existence of the magnetotail — the elongated nightside extension of Earth's magnetosphere. Showed that the magnetotail stores magnetic energy that is released during substorms. IMP-1 위성 데이터를 사용하여 자기꼬리(magnetotail)의 존재를 확인 — 지구 자기권의 야간측 확장 구조. 자기꼬리가 substorm 동안 방출되는 자기 에너지를 저장함을 보여줌.
- **Prerequisites**: Papers #4, #6; satellite data interpretation (위성 데이터 해석), magnetic field measurements (자기장 측정)
- **Status**: [x]

---

## Phase 3: Substorms, Convection & M-I Coupling / Substorm, 대류 및 자기권-전리층 결합 (1970s–1990s)

### 10. Satellite Studies of Magnetospheric Substorms on August 15, 1968
- **Authors**: Robert McPherron, Christopher Russell, Michel Aubry
- **Year**: 1973
- **DOI**: 10.1029/JA078i016p03131
- **Why it matters**: Provided the first comprehensive multi-satellite analysis of a substorm, establishing the near-Earth neutral line (NENL) model. Defined the growth-expansion-recovery substorm sequence using in-situ spacecraft data, complementing Akasofu's ground-based morphology. 최초의 포괄적 다중 위성 substorm 분석을 제공하여 near-Earth neutral line (NENL) 모델을 확립. 현장 우주선 데이터를 사용하여 성장-팽창-회복 substorm 시퀀스를 정의.
- **Prerequisites**: Papers #6, #8, #9; magnetotail dynamics (자기꼬리 역학), neutral line concept (중성선 개념)
- **Status**: [x]

### 11. An Empirical Relationship between Interplanetary Conditions and Dst
- **Authors**: Rande Burton, Robert McPherron, Christopher Russell
- **Year**: 1975
- **DOI**: 10.1029/JA080i031p04204
- **Why it matters**: Derived the first empirical formula relating solar wind parameters to the Dst index (a measure of geomagnetic storm intensity). This Burton equation became the foundation for all subsequent empirical storm forecasting models. 태양풍 매개변수와 Dst 지수(지자기 폭풍 강도의 척도)를 연결하는 최초의 경험적 공식을 도출. Burton 방정식은 이후 모든 경험적 폭풍 예보 모델의 기초가 됨.
- **Prerequisites**: Papers #3, #6; Dst index (Dst 지수), ring current physics (환전류 물리학)
- **Status**: [x]

### 12. The Causes of Convection in the Earth's Magnetosphere: A Review of Developments During the IMS
- **Authors**: Steven Cowley
- **Year**: 1982
- **DOI**: 10.1029/RG020i003p00531
- **Why it matters**: Synthesized Dungey-cycle reconnection and viscous interaction into a unified framework for magnetospheric convection driven by the solar wind. Clarified how the IMF orientation controls the global convection pattern and energy transfer. Dungey 순환 재결합과 점성 상호작용을 태양풍에 의해 구동되는 자기권 대류의 통합 틀로 종합. IMF 방향이 전역 대류 패턴과 에너지 전달을 어떻게 제어하는지 명확히 함.
- **Prerequisites**: Papers #6, #7, #10; magnetospheric convection (자기권 대류), IMF coupling (IMF 결합)
- **Status**: [x]

### 13. Mapping Electrodynamic Features of the High-Latitude Ionosphere from Localized Observations (AMIE)
- **Authors**: Arthur D. Richmond, Yoshuke Kamide
- **Year**: 1988
- **DOI**: 10.1029/JA093iA06p05741
- **Why it matters**: Introduced the Assimilative Mapping of Ionospheric Electrodynamics (AMIE) technique, which combines diverse ground and satellite observations to produce global maps of ionospheric electric fields and currents. This data assimilation approach became a standard tool for magnetosphere-ionosphere coupling studies. 다양한 지상 및 위성 관측을 결합하여 전리층 전기장과 전류의 전역 지도를 생성하는 AMIE 기법을 도입. 이 데이터 동화 접근법은 자기권-전리층 결합 연구의 표준 도구가 됨.
- **Prerequisites**: Papers #8, #11; ionospheric electrodynamics (전리층 전기역학), data assimilation concepts (데이터 동화 개념)
- **Status**: [x]

### 14. A Magnetospheric Magnetic Field Model with a Warped Tail Current Sheet
- **Authors**: Nikolai Tsyganenko
- **Year**: 1989
- **DOI**: 10.1016/0032-0633(89)90066-4
- **Why it matters**: Developed the first widely-used empirical model of Earth's magnetospheric magnetic field (T89), parameterized by the Kp index. The Tsyganenko models became the standard tool for mapping magnetic field lines and tracing particle trajectories in space weather research. Kp 지수로 매개변수화된 지구 자기권 자기장의 최초 널리 사용된 경험적 모델(T89)을 개발. Tsyganenko 모델은 우주기상 연구에서 자기장 선 매핑과 입자 궤적 추적의 표준 도구가 됨.
- **Prerequisites**: Papers #3, #9; magnetic field modeling (자기장 모델링), spherical harmonics (구면 조화 함수)
- **Status**: [x]

### 15. What Is a Geomagnetic Storm?
- **Authors**: Walter Gonzalez, Joselyn Joselyn, Yohsuke Kamide, Bruce Tsurutani, Risto Pirjola et al.
- **Year**: 1994
- **DOI**: 10.1029/93JA02867
- **Why it matters**: Provided the definitive review and classification of geomagnetic storms — distinguishing storms from substorms, defining storm phases, and establishing quantitative criteria (Dst thresholds) for storm intensity. Essential for standardizing space weather terminology. 지자기 폭풍의 결정적 검토와 분류를 제공 — 폭풍과 substorm을 구별하고, 폭풍 단계를 정의하며, 폭풍 강도에 대한 정량적 기준(Dst 임계값)을 확립. 우주기상 용어 표준화에 필수적.
- **Prerequisites**: Papers #8, #10, #11; ring current (환전류), magnetic indices (자기 지수)
- **Status**: [x]

### 16. Response of the Thermosphere and Ionosphere to Geomagnetic Storms
- **Authors**: Timothy J. Fuller-Rowell, Mihail V. Codrescu, R. J. Moffett, S. Quegan
- **Year**: 1994
- **DOI**: 10.1029/93JA02015
- **Why it matters**: Explains how geomagnetic storms drive neutral composition changes (O/N2 ratio), heating, and winds in the thermosphere, which in turn modify the ionosphere. This thermosphere-ionosphere coupling is critical for understanding satellite drag and ionospheric storm effects. 지자기 폭풍이 열권에서 중성 조성 변화(O/N2 비), 가열, 바람을 구동하고, 이것이 전리층을 변경하는 방법을 설명합니다. 이 열권-전리층 결합은 위성 항력과 전리층 폭풍 효과를 이해하는 데 중요합니다.
- **Prerequisites**: Papers #13, #15; thermospheric physics (열권 물리학), neutral-ion coupling (중성-이온 결합)
- **Status**: [x]

---

## Phase 4: Space Weather Effects & Forecasting / 우주기상 영향 및 예보 (1989–2010)

### 17. Effects of the March 1989 Solar Activity
- **Authors**: Joe Allen, Herb Sauer, Loren Frank, Patricia Reiff
- **Year**: 1989
- **DOI**: 10.1029/89EO00409
- **Why it matters**: Documented the effects of the great March 1989 geomagnetic storm, including the collapse of the Hydro-Quebec power grid (the most famous space weather event in modern history). Demonstrated that space weather can cause catastrophic infrastructure failures and catalyzed the space weather forecasting effort. 1989년 3월 대자기폭풍의 영향을 기록, Hydro-Quebec 전력망 붕괴 포함 (현대 역사상 가장 유명한 우주기상 사건). 우주기상이 치명적인 인프라 장애를 초래할 수 있음을 입증하고 우주기상 예보 노력을 촉진.
- **Prerequisites**: Papers #11, #15; geomagnetically induced currents (지자기 유도 전류, GIC), power grid basics (전력망 기초)
- **Status**: [x]

### 18. Improvement in the Prediction of Solar Wind Conditions Using Near-Real Time Solar Magnetic Field Updates (WSA)
- **Authors**: C. Nick Arge, Victor J. Pizzo
- **Year**: 2000
- **DOI**: 10.1029/1999JA000262
- **Why it matters**: Introduced the Wang-Sheeley-Arge (WSA) model that maps photospheric magnetic fields to solar wind speed at the source surface using flux tube expansion factors. WSA is the input driver for ENLIL and most operational solar wind forecasting systems. Wang-Sheeley-Arge(WSA) 모델을 도입하여 플럭스 관 팽창 계수를 사용하여 광구 자기장을 소스면의 태양풍 속도로 매핑합니다. WSA는 ENLIL 및 대부분의 운용 태양풍 예보 시스템의 입력 구동기입니다.
- **Prerequisites**: Papers #4, #11; potential field source surface (PFSS) model (포텐셜 필드 소스 서피스 모델), flux tube expansion (플럭스 관 팽창)
- **Status**: [x]

### 19. Effects of the Sun on the Earth's Environment and the Consequences for Mankind
- **Authors**: Daniel Baker
- **Year**: 2000
- **DOI**: 10.1016/S1364-6826(00)00119-X
- **Why it matters**: Comprehensive review of how solar variability affects Earth's technological systems — from satellite operations and communications to power grids and GPS. Established the modern framing of "space weather" as a discipline focused on societal impacts. 태양 변동성이 지구의 기술 시스템에 미치는 영향에 대한 포괄적 검토 — 위성 운영 및 통신에서 전력망 및 GPS까지. '우주기상'을 사회적 영향에 초점을 맞춘 학문으로 현대적 틀을 확립.
- **Prerequisites**: Papers #5, #15, #17; solar cycle basics (태양 주기 기초), technological vulnerability concepts (기술적 취약성 개념)
- **Status**: [x]

### 20. GPS and Ionospheric Scintillations
- **Authors**: Paul M. Kintner, Brent M. Ledvina, Eurico R. de Paula
- **Year**: 2007
- **DOI**: 10.1029/2006SW000260
- **Why it matters**: The canonical review of how ionospheric scintillation disrupts GPS/GNSS signals — linking the physics of equatorial plasma bubbles and high-latitude irregularities to receiver-level effects (amplitude fades, cycle slips, loss of lock). Cornerstone reference for space-weather impacts on navigation. 전리층 신틸레이션이 GPS/GNSS 신호를 교란하는 메커니즘을 정리한 정본(canonical) 리뷰 — 적도 플라즈마 버블과 고위도 불규칙성의 물리를 수신기 레벨 영향(진폭 fade, cycle slip, loss of lock)과 연결. 우주기상의 항법 영향 연구의 주춧돌 논문.
- **Prerequisites**: Papers #13, #16; ionospheric physics (전리층 물리학), radio wave propagation (전파 전파), Fresnel diffraction (Fresnel 회절), GPS carrier/code tracking loops (GPS 반송파/코드 추적 루프)
- **Status**: [x]

### 21. Modeling 3-D Solar Wind Structure
- **Authors**: Dusan Odstrcil
- **Year**: 2003
- **DOI**: 10.1016/S0273-1177(03)00332-6
- **Why it matters**: ENLIL is the primary operational heliospheric MHD model used by NOAA/SWPC for CME arrival time forecasting. Understanding its physics, inputs (from WSA), and limitations is essential for anyone working in space weather operations. ENLIL은 NOAA/SWPC가 CME 도착 시간 예보에 사용하는 주요 운용 태양권 MHD 모델입니다. 물리학, 입력(WSA에서), 한계를 이해하는 것은 우주기상 운용에 종사하는 모든 사람에게 필수적입니다.
- **Prerequisites**: Paper #18; MHD simulation methods (MHD 시뮬레이션 방법), cone model for CME input (CME 입력을 위한 원뿔 모델)
- **Status**: [x]

### 22. The Global Dayside Ionospheric Uplift and Enhancement Associated with Interplanetary Electric Fields
- **Authors**: Bruce Tsurutani, Anthony Mannucci, Olga Verkhoglyadova et al.
- **Year**: 2004
- **DOI**: 10.1029/2003JA010342
- **Why it matters**: Discovered the prompt penetration of interplanetary electric fields to low-latitude ionosphere during great storms, causing rapid global ionospheric uplift. Revealed a previously unknown mechanism of solar wind-ionosphere coupling that impacts GPS accuracy worldwide. 대폭풍 동안 행성간 전기장의 저위도 전리층 즉시 침투를 발견, 급속한 전구 전리층 상승을 유발. 전 세계 GPS 정확도에 영향을 미치는 이전에 알려지지 않은 태양풍-전리층 결합 메커니즘을 밝힘.
- **Prerequisites**: Papers #12, #15; ionospheric physics (전리층 물리학), electric field penetration (전기장 침투)
- **Status**: [x]

### 23. Introduction to Violent Sun–Earth Connection Events of October–November 2003
- **Authors**: Nat Gopalswamy, Seiji Yashiro, Sachiko Akiyama et al.
- **Year**: 2005
- **DOI**: 10.1029/2005JA011268
- **Why it matters**: Comprehensive case study of the October–November 2003 Halloween storms, one of the most well-observed extreme space weather events. Provides end-to-end analysis from solar eruption through interplanetary propagation to terrestrial impact. 2003년 10-11월 Halloween 폭풍의 포괄적 사례 연구로, 가장 잘 관측된 극한 우주기상 사건 중 하나입니다. 태양 분출에서 행성간 전파를 거쳐 지상 영향까지 종단 간 분석을 제공합니다.
- **Prerequisites**: Papers #15, #17; CME propagation (CME 전파), storm sudden commencement (폭풍 급시)
- **Status**: [x]

### 24. Space Weather: Terrestrial Perspective
- **Authors**: Tuija Pulkkinen
- **Year**: 2007
- **DOI**: 10.12942/lrsp-2007-1
- **Why it matters**: Authoritative review of ground-level space weather effects, focusing on geomagnetically induced currents (GICs) in technological systems. Provided the theoretical framework for quantifying GIC risk to power grids, pipelines, and communication cables. 지상 수준 우주기상 영향에 대한 권위 있는 검토, 기술 시스템의 지자기 유도 전류(GIC)에 초점. 전력망, 파이프라인, 통신 케이블에 대한 GIC 위험을 정량화하는 이론적 틀을 제공.
- **Prerequisites**: Papers #11, #17; electromagnetic induction (전자기 유도), conductivity models (전도도 모델)
- **Status**: [x]

### 25. Geomagnetic Storms and Their Impacts on the U.S. Power Grid
- **Authors**: John Kappenman
- **Year**: 2010
- **DOI**: NO_DOI
- **Why it matters**: Quantified the vulnerability of the U.S. power grid to extreme geomagnetic storms using detailed GIC modeling. Showed that a Carrington-class event could cause widespread transformer damage and prolonged blackouts. This analysis drove policy changes in critical infrastructure protection. 상세한 GIC 모델링을 사용하여 극한 지자기 폭풍에 대한 미국 전력망의 취약성을 정량화. Carrington급 사건이 광범위한 변압기 손상과 장기 정전을 초래할 수 있음을 보여줌. 이 분석은 핵심 인프라 보호 정책 변화를 촉진.
- **Prerequisites**: Papers #17, #24; GIC modeling (GIC 모델링), power system engineering basics (전력 시스템 공학 기초)
- **Status**: [x]

### 26. Radiation Belt Dynamics: The Importance of Wave-Particle Interactions
- **Authors**: Richard M. Thorne
- **Year**: 2010
- **DOI**: 10.1029/2010GL044990
- **Why it matters**: Comprehensive review of wave-particle interactions controlling radiation belt dynamics — chorus waves, EMIC waves, plasmaspheric hiss. Explains both acceleration and loss of relativistic electrons, critical for satellite operations. 방사선대 역학을 제어하는 파동-입자 상호작용의 포괄적 리뷰 — 코러스 파, EMIC 파, 플라즈마스피어 히스. 상대론적 전자의 가속과 손실 모두를 설명하며, 위성 운용에 중요합니다.
- **Prerequisites**: Papers #5, #9; plasma wave theory (플라즈마 파동 이론), quasi-linear diffusion (준선형 확산)
- **Status**: [x]

---

## Phase 5: Modern Space Weather & Machine Learning / 현대 우주기상 및 머신러닝 (2008–Present)

### 27. The THEMIS Mission
- **Authors**: Vassilis Angelopoulos
- **Year**: 2008
- **DOI**: 10.1007/s11214-008-9336-1
- **Why it matters**: Described the Time History of Events and Macroscale Interactions during Substorms (THEMIS) mission — a constellation of five spacecraft designed to resolve the onset and evolution of magnetospheric substorms. THEMIS settled key debates about substorm triggering mechanisms. Substorm의 시작과 진화를 규명하기 위해 설계된 5개 우주선 성좌인 THEMIS 미션을 기술. THEMIS는 substorm 촉발 메커니즘에 대한 핵심 논쟁을 해결.
- **Prerequisites**: Papers #8, #10; multi-spacecraft analysis techniques (다중 우주선 분석 기법)
- **Status**: [x]

### 28. Science Objectives and Rationale for the Radiation Belt Storm Probes Mission (Van Allen Probes)
- **Authors**: Barry Mauk, Nicola Fox, Shrikanth Kanekal, Robin Kessel, Daniel Sibeck, Aleksandr Ukhorskiy
- **Year**: 2013
- **DOI**: 10.1007/s11214-012-9908-y
- **Why it matters**: Outlined the science objectives of the Van Allen Probes — twin spacecraft designed to study Earth's radiation belts with unprecedented resolution. The mission revolutionized our understanding of radiation belt dynamics, including particle acceleration and loss processes critical for satellite safety. Van Allen Probes의 과학 목표를 개요 — 전례 없는 해상도로 지구의 방사선대를 연구하기 위해 설계된 쌍둥이 우주선. 이 미션은 위성 안전에 중요한 입자 가속 및 손실 과정을 포함하여 방사선대 역학에 대한 이해를 혁신.
- **Prerequisites**: Papers #5, #14; radiation belt physics (방사선대 물리학), wave-particle interactions (파동-입자 상호작용)
- **Status**: [x]

### 29. A Major Solar Eruptive Event in July 2012: Defining Extreme Space Weather Scenarios
- **Authors**: Daniel N. Baker et al.
- **Year**: 2013
- **DOI**: 10.1002/swe.20097
- **Why it matters**: Analyzed the extreme July 2012 CME that narrowly missed Earth, estimating it would have been comparable to the 1859 Carrington event. Provided quantitative analysis of what an extreme event would do to modern infrastructure, driving policy discussions on space weather preparedness. 지구를 간신히 빗겨간 2012년 7월 극한 CME를 분석하여, 1859년 Carrington 사건에 필적할 것으로 추정했습니다. 극한 사건이 현대 인프라에 미칠 영향의 정량적 분석을 제공하여 우주기상 대비에 대한 정책 논의를 촉진했습니다.
- **Prerequisites**: Papers #15, #17, #25; CME speeds (CME 속도), extreme value statistics (극한값 통계)
- **Status**: [x]

### 30. An Impenetrable Barrier to Ultrarelativistic Electrons in the Van Allen Radiation Belts
- **Authors**: Daniel Baker, A.N. Jaynes, V.C. Hoxie, R.M. Thorne et al.
- **Year**: 2014
- **DOI**: 10.1038/nature13956
- **Why it matters**: Discovered a sharp, persistent inner boundary to ultrarelativistic electrons at about 2.8 Earth radii — an "impenetrable barrier" maintained by wave-particle interactions. This unexpected result from the Van Allen Probes changed our understanding of radiation belt structure. 약 2.8 지구 반경에서 초상대론적 전자에 대한 날카롭고 지속적인 내부 경계 — 파동-입자 상호작용에 의해 유지되는 '뚫을 수 없는 장벽'을 발견. Van Allen Probes의 이 예상치 못한 결과는 방사선대 구조에 대한 이해를 변화시킴.
- **Prerequisites**: Papers #5, #26, #28; relativistic particle dynamics (상대론적 입자 역학), plasma waves (플라즈마 파동)
- **Status**: [x]

### 31. Space Weather Operations: The NOAA Space Weather Prediction Center
- **Authors**: Douglas A. Biesecker et al.
- **Year**: 2015
- **DOI**: NO_DOI
- **Why it matters**: Describes the operational forecasting framework at SWPC including data inputs, model chains (WSA-ENLIL, OVATION), product dissemination, and forecast verification. Essential for understanding how research transitions to operations in space weather. SWPC의 운용 예보 프레임워크를 기술하며, 데이터 입력, 모델 체인(WSA-ENLIL, OVATION), 제품 배포, 예보 검증을 포함합니다. 우주기상에서 연구가 운용으로 전환되는 방법을 이해하는 데 필수적입니다.
- **Prerequisites**: Papers #18, #21; forecast verification (예보 검증), operational workflow (운용 워크플로우)
- **Status**: [x]

### 32. Geomagnetically Induced Currents: Science, Engineering, and Applications Readiness
- **Authors**: Antti Pulkkinen et al.
- **Year**: 2017
- **DOI**: 10.1002/2016SW001501
- **Why it matters**: Updated GIC review incorporating lessons from the Van Allen Probes era, including improved 3D Earth conductivity models and real-time GIC monitoring capabilities. Bridges the gap between research and modern operational GIC mitigation. Van Allen Probes 시대의 교훈을 포함한 업데이트된 GIC 리뷰로, 개선된 3D 지구 전도도 모델과 실시간 GIC 모니터링 능력을 포함합니다. 연구와 현대 운용 GIC 완화 간의 간극을 메웁니다.
- **Prerequisites**: Papers #24, #25; 3D Earth conductivity models (3D 지구 전도도 모델), GIC network modeling (GIC 네트워크 모델링)
- **Status**: [x]

### 33. The Challenge of Machine Learning in Space Weather: Nowcasting and Forecasting
- **Authors**: Enrico Camporeale
- **Year**: 2019
- **DOI**: 10.1029/2018SW002061
- **Why it matters**: Comprehensive review of machine learning applications in space weather — from solar flare prediction to radiation belt modeling and geomagnetic index forecasting. Established best practices for applying ML to space physics data and identified key challenges (class imbalance, interpretability, limited training data). 우주기상에서의 머신러닝 응용에 대한 포괄적 검토 — 태양 플레어 예측에서 방사선대 모델링 및 지자기 지수 예보까지. 우주물리 데이터에 ML을 적용하기 위한 모범 사례를 확립하고 주요 과제(클래스 불균형, 해석 가능성, 제한된 훈련 데이터)를 식별.
- **Prerequisites**: Papers #11, #15; machine learning fundamentals (머신러닝 기초), classification and regression (분류 및 회귀)
- **Status**: [x]

### 34. A Comparison of Flare Forecasting Methods
- **Authors**: KD Leka, Sung-Hong Park, Kanya Kusano, Jesse Andries et al.
- **Year**: 2019
- **DOI**: 10.3847/1538-4365/ab2e12
- **Why it matters**: Systematic comparison of dozens of solar flare prediction methods (both physics-based and ML-based) using standardized metrics. Established benchmarks and revealed that no single method dominates across all metrics — highlighting the need for ensemble approaches and better evaluation standards in space weather forecasting. 표준화된 메트릭을 사용하여 수십 가지 태양 플레어 예측 방법(물리 기반 및 ML 기반 모두)을 체계적으로 비교. 벤치마크를 확립하고 단일 방법이 모든 메트릭에서 우위를 차지하지 않음을 밝힘 — 우주기상 예보에서 앙상블 접근법과 더 나은 평가 표준의 필요성을 강조.
- **Prerequisites**: Paper #33; solar flare physics (태양 플레어 물리학), forecast verification metrics (예보 검증 메트릭, TSS, HSS, BSS)
- **Status**: [x]

### 35. Solar Wind Prediction Using Deep Learning
- **Authors**: Vishal Upendran et al.
- **Year**: 2020
- **DOI**: 10.1029/2020SW002478
- **Why it matters**: Applies modern deep learning to predict solar wind parameters (especially Bz) from solar disk observations. Represents the cutting edge of ML-based space weather forecasting, complementing traditional physics-based models. 현대 딥러닝을 적용하여 태양 디스크 관측으로부터 태양풍 매개변수(특히 Bz)를 예측합니다. 전통적 물리 기반 모델을 보완하는 ML 기반 우주기상 예보의 최첨단을 대표합니다.
- **Prerequisites**: Papers #33, #34; deep learning architectures (딥러닝 아키텍처), time-series forecasting (시계열 예보)
- **Status**: [x]

---

## Legend
- `[ ]` Not started / 시작 전
- `[~]` In progress / 진행 중
- `[x]` Completed / 완료

---

## User-Added Papers / 사용자 추가 논문

### 36. Extreme Space-Weather Events and the Solar Cycle
- **Authors**: Mathew J. Owens, Mike Lockwood, Luke A. Barnard, Chris J. Scott, Carl Haines, Allan Macneil
- **Year**: 2021
- **DOI**: 10.1007/s11207-021-01831-3
- **Why it matters**: 150년 aa_H 지자기 기록을 사용하여 극한 우주기상 사건의 발생이 태양 주기에 의해 조절됨을 통계적으로 입증. 홀수/짝수 태양 주기 간의 극한 사건 발생 시기 차이를 보고하고, Solar Cycle 25의 극한 사건 확률 추정 가능성을 제시. / Uses the 150-year aa_H geomagnetic record to statistically demonstrate that extreme space-weather event occurrence is modulated by the solar cycle. Reports differences in extreme-event timing between odd and even solar cycles, and shows how the probability of extreme events for Solar Cycle 25 can be estimated.
- **Prerequisites**: Papers #3, #11, #15, #29; 지자기 지수 (aa, Dst), 태양 주기 기초, 극한값 통계 (extreme-value statistics), 확률론적 모델링 / Geomagnetic indices (aa, Dst), solar cycle basics, extreme-value statistics, probabilistic modeling
- **Status**: [x]

### 37. Extended Lead-Time Geomagnetic Storm Forecasting With Solar Wind Ensembles and Machine Learning
- **Authors**: M. Billcliff, A. W. Smith, M. Owens, W. L. Woo, L. Barnard, N. Edward-Inatimi, I. J. Rae
- **Year**: 2026
- **DOI**: 10.1029/2025SW004823
- **Why it matters**: 태양풍 앙상블(MAS+HUXt)과 로지스틱 회귀 분류기를 결합하여 Hp30 지자기 폭풍 예보의 리드 타임을 24시간으로 확장. 기존 L1 관측 기반 예보(30-90분)를 크게 초과하는 확률적 폭풍 예보 프레임워크를 제시. / Extends geomagnetic storm lead time to 24 hr by combining solar wind ensembles (MAS+HUXt) with logistic regression classifiers for Hp30 forecasting. Provides a probabilistic storm forecasting framework far exceeding current L1-based forecasts (30-90 min).
- **Prerequisites**: Papers #18, #21, #33; 태양풍 모델링 (HUXt, MAS), Hp30/Kp 지자기 지수, 로지스틱 회귀, 앙상블 예측 / Solar wind modeling (HUXt, MAS), Hp30/Kp geomagnetic indices, logistic regression, ensemble prediction
- **Status**: [x]

### 38. Prediction of the SYM-H Index Using a Bayesian Deep Learning Method With Uncertainty Quantification
- **Authors**: Yasser Abduallah, Khalid A. Alobaid, Jason T. L. Wang, Haimin Wang, Vania K. Jordanova, Vasyl Yurchyshyn, Husein Cavus, Ju Jing
- **Year**: 2024
- **DOI**: 10.1029/2023SW003824
- **Why it matters**: GNN과 BiLSTM을 결합한 SYMHnet으로 SYM-H 지수를 1-2시간 전에 예측하며, Bayesian inference를 통해 모델 및 데이터 불확실성을 정량화. 1분 해상도의 SYM-H 예측에 딥러닝을 최초로 적용. / SYMHnet combines GNN and BiLSTM with Bayesian inference to predict SYM-H index 1-2 hr in advance while quantifying model and data uncertainty. First deep learning application to 1-min resolution SYM-H prediction.
- **Prerequisites**: Papers #11, #15, #33; SYM-H/Dst 지자기 지수, 그래프 신경망(GNN), BiLSTM, Bayesian deep learning, 태양풍/IMF 파라미터 / SYM-H/Dst geomagnetic indices, graph neural networks (GNN), BiLSTM, Bayesian deep learning, solar wind/IMF parameters
- **Status**: [x]

### 39. Daily Predictions of F10.7 and F30 Solar Indices With Deep Learning
- **Authors**: Zhenduo Wang, Yasser Abduallah, Jason T. L. Wang, Haimin Wang, Yan Xu, Vasyl Yurchyshyn, Vincent Oria, Khalid A. Alobaid, Xiaoli Bai
- **Year**: 2026
- **DOI**: 10.1029/2025JA034868
- **Why it matters**: SINet(Solar Index Network)으로 F10.7과 F30 태양 활동 지수를 1-60일 전에 일별 예측. TimesNet 기반 아키텍처로 FFT를 활용한 주기성 포착. F30 예측에 딥러닝을 최초 적용하며, 기존 5개 방법 대비 우수한 성능. / SINet predicts daily F10.7 and F30 solar indices 1-60 days in advance. TimesNet-based architecture with FFT for periodicity capture. First deep learning method for F30 prediction, outperforming 5 related methods.
- **Prerequisites**: Papers #33, #35; F10.7/F30 태양 활동 지수, 시계열 예측, TimesNet 아키텍처, CNN, FFT / F10.7/F30 solar activity indices, time-series forecasting, TimesNet architecture, CNN, FFT
- **Status**: [x]

### 40. Automatic 3D Reconstruction of Coronal Mass Ejections Based on Dual-viewpoint Observations and Machine Learning
- **Authors**: Rongpei Lin, Yi Yang, Fang Shen, Gilbert Pi, Yucong Li
- **Year**: 2025
- **Why it matters**: CME는 우주기상 폭풍의 주요 구동자이며, 정확한 도착 시각·속도·폭 예측을 위해서는 단일 시점 LASCO 관측의 투영 효과를 보정한 3D 파라미터가 필수적이다. 본 논문은 SOHO/LASCO C2와 STEREO-A/COR2 이중 시점 관측에 CNN 기반 영역 검출, PCA 기반 colocalization, Otsu 이진화, GCS(Graduated Cylindrical Shell) 모델 피팅, 그리고 differential evolution 최적화를 결합하여 CME 3D 구조를 자동 재구성하는 최초의 통합 프레임워크를 제시한다. 97개 CME(2007-2018)에 대한 통계 분석으로 2D 측정이 속도를 8% 과소, 폭을 47% 과대 추정함을 정량화하여, 향후 ML 기반 우주기상 예보의 표준 입력을 제공한다. / CMEs are the major drivers of severe space-weather storms, and accurate prediction of their arrival, speed, and width requires 3D parameters that correct for the projection effect of single-viewpoint LASCO observations. This paper presents the first integrated framework that automatically reconstructs the 3D structure of CMEs by combining dual-viewpoint observations (SOHO/LASCO C2 and STEREO-A/COR2) with CNN-based region detection, PCA-based colocalization, Otsu binarization, GCS (Graduated Cylindrical Shell) model fitting, and differential-evolution optimization. A statistical analysis of 97 CMEs (2007-2018) quantifies that 2D measurements underestimate velocity by 8% and overestimate width by 47%, providing standard inputs for future ML-based space-weather forecasting.
- **Prerequisites**: CME 기본 물리(자속관·전방 충격파·구조), 코로나그래프 영상 처리(running difference, Thomson scattering), GCS 모델(6 파라미터: longitude·latitude·tilt·half-angle·aspect ratio·height), CNN(LeNet-5 변형), PCA, Otsu 이진화, differential evolution 최적화. SOHO/LASCO와 STEREO 미션의 기본 이해. / Basics of CME physics (flux ropes, shocks, morphology), coronagraph image processing (running difference, Thomson scattering), the GCS model (6 parameters: longitude, latitude, tilt, half-angle, aspect ratio, height), CNNs (LeNet-5 variants), PCA, Otsu binarization, differential evolution. Familiarity with SOHO/LASCO and STEREO missions.
- **Status**: [x]

### 41. CMEGNets: A self-supervised framework for coronal mass ejection detection & region segmentation
- **Authors**: Besma Guesmi, Jinen Daghrir, David Moloney, Jose Luis Espinosa-Aranda, Elena Hervas-Martin
- **Year**: 2026
- **Why it matters**: 수동 어노테이션 없이 LASCO C2/C3에서 SSL(SimCLR)로 CME를 탐지·분할하는 현대 ML 파이프라인. 99% 분류·95% Dice를 달성하며 레이블링 비용을 80% 이상 절감 — Phase 5(현대 우주기상 ML) 대표 사례. / A modern ML pipeline for CME detection/segmentation on LASCO C2/C3 using SSL (SimCLR) without manual annotation. Achieves 99% classification & 95% Dice while cutting annotation effort by >80% — a flagship Phase 5 (modern space-weather ML) exemplar.
- **Prerequisites**: CNN·U-Net 기초, 대조학습(SimCLR)과 NT-Xent 손실, Mahalanobis 거리, UMAP, CME·LASCO 코로나그래프 관측 기본. / Basics of CNN/U-Net, contrastive learning (SimCLR) with NT-Xent loss, Mahalanobis distance, UMAP, and CME/LASCO coronagraph fundamentals.
- **Status**: [x]

---

## Instruments & Missions / 관측 기기 및 미션 (L1 모니터, GOES, 오로라 관측)

### 42. The Advanced Composition Explorer (ACE)
- **Authors**: E. C. Stone, A. M. Frandsen, R. A. Mewaldt, E. R. Christian, D. Margolies, J. F. Ormes, F. Snow
- **Year**: 1998
- **Journal**: Space Science Reviews, Vol. 86, pp. 1–22
- **DOI**: 10.1023/A:1005082526237
- **Why it matters**: L1에 위치한 ACE는 ~1시간 전 태양풍·IMF 실시간 경보의 사실상 표준으로, 27년간 NOAA/SWPC 운용 예보의 주 입력이 되어왔다. 9개 기기(SWEPAM, MAG, SWICS, SWIMS, ULEIS, SEPICA, SIS, CRIS, EPAM)가 태양풍·입자·조성을 동시 측정한다. / ACE at L1 is the de-facto standard for real-time solar wind and IMF monitoring (~1-hour lead time), serving as the primary input to NOAA/SWPC operational forecasts for 27 years. Its nine instruments (SWEPAM, MAG, SWICS, SWIMS, ULEIS, SEPICA, SIS, CRIS, EPAM) simultaneously measure solar wind, energetic particles, and composition.
- **Prerequisites**: Papers #4, #11, #15; L1 Lagrangian orbit (L1 라그랑주 궤도), solar wind plasma parameters (태양풍 플라즈마 매개변수), IMF measurement (IMF 측정)
- **Status**: [x]

### 43. The WIND Spacecraft and Its Early Scientific Results
- **Authors**: K. W. Ogilvie, M. D. Desch
- **Year**: 1997
- **Journal**: Advances in Space Research, Vol. 20(4–5), pp. 559–568
- **DOI**: 10.1016/S0273-1177(97)00439-0
- **Why it matters**: Wind는 ACE 이전부터 운용된 L1 태양풍 모니터이자 현재도 지속되는 최장 연속 태양풍 데이터셋(1994–현재)을 제공한다. SWE·MFI·3DP·WAVES 기기 구성과 초기 과학 성과를 소개한다. / Wind predates ACE as an L1 solar wind monitor and provides the longest continuous solar wind dataset (1994–present). The paper describes the SWE, MFI, 3DP, and WAVES instruments and early scientific findings.
- **Prerequisites**: Papers #4, #6; solar wind plasma and magnetic-field measurement techniques (태양풍 플라즈마·자기장 측정 기법)
- **Status**: [x]

### 44. Deep Space Climate Observatory: The DSCOVR Mission
- **Authors**: J. Burt, B. Smith
- **Year**: 2012
- **Journal**: 2012 IEEE Aerospace Conference, pp. 1–13
- **DOI**: 10.1109/AERO.2012.6187025
- **Why it matters**: DSCOVR는 ACE의 후계 L1 태양풍 실시간 모니터로 NOAA/SWPC의 현재 주 경보원이다. NASA Triana(2000) 미션을 재활용해 2015년 발사되었으며, PlasMag과 Faraday Cup을 통해 태양풍 속도·밀도·IMF Bz를 실시간 전송한다. / DSCOVR is the successor L1 solar wind real-time monitor to ACE and the current primary alert source for NOAA/SWPC. Originally developed as NASA's Triana (2000) and launched in 2015, it transmits solar wind speed, density, and IMF Bz in real time via PlasMag and the Faraday Cup.
- **Prerequisites**: Paper #42; L1 orbit (L1 궤도), real-time space-weather alerting (실시간 우주기상 경보)
- **Status**: [x]

### 45. The GOES-R Series: A New Generation of Geostationary Environmental Satellites (Chapter 1)
- **Authors**: S. J. Goodman, T. J. Schmit, J. Daniels, R. J. Redmon (eds.)
- **Year**: 2019
- **Journal**: Elsevier (ISBN 978-0-12-814327-8), Chapter 1, pp. 1–3
- **DOI**: 10.1016/B978-0-12-814327-8.00001-9
- **Why it matters**: 현재 운용 중인 GOES-16/17/18/19(GOES-R 계열) 위성의 종합 개요. 우주기상 관측 탑재체인 SUVI(Solar Ultraviolet Imager), EXIS(X-ray/EUV), MAG(마그네토미터), SEISS(입자 분광계), GLM(Geostationary Lightning Mapper)을 포괄한다. / A comprehensive overview of the currently operational GOES-16/17/18/19 (GOES-R series). Covers the space-weather payload — SUVI (Solar Ultraviolet Imager), EXIS (X-ray/EUV sensors), MAG (magnetometer), SEISS (particle spectrometers), and GLM (Geostationary Lightning Mapper).
- **Prerequisites**: Satellite instrumentation basics (위성 탑재체 기초), geostationary orbit (지구정지궤도)
- **Status**: [x]

### 46. Operational Uses of the GOES Energetic Particle Detectors
- **Authors**: T. G. Onsager, R. Grubb, J. Kunches, L. Matheson, D. Speich, R. Zwickl, H. Sauer
- **Year**: 1996
- **Journal**: Proc. SPIE 2812 (GOES-8 and Beyond), pp. 281–290
- **DOI**: 10.1117/12.254075
- **Why it matters**: NOAA/SWPC가 SEP(고에너지 태양 입자) 경보와 위성 운용자 대상 방사선 경고를 발령할 때 사용하는 GOES EPS/HEPAD 채널의 운용 기준과 임계값을 정의한 논문. "S1–S5" Solar Radiation Storm 등급 체계의 기반이 된다. / Defines the operational criteria and thresholds for the GOES EPS/HEPAD channels that NOAA/SWPC uses to issue SEP (Solar Energetic Particle) alerts and radiation warnings to satellite operators. Forms the basis of the "S1–S5" Solar Radiation Storm scale.
- **Prerequisites**: Papers #17, #24; SEP events (SEP 사건), solid-state particle detectors (고체 입자 검출기)
- **Status**: [x]

### 47. Imaging Results from Dynamics Explorer 1
- **Authors**: L. A. Frank, J. D. Craven
- **Year**: 1988
- **Journal**: Reviews of Geophysics, Vol. 26, pp. 249–283
- **DOI**: 10.1029/RG026i002p00249
- **Why it matters**: **최초의 전역 오로라 영상** — 지구 궤도 고고도에서 오로라 타원 전체를 한 번에 포착한 혁명적 데이터를 제시. DE-1의 SAI(Spin-Scan Auroral Imager)는 이후 모든 우주 오로라 영상기의 원형이 되었다. / **First global auroral imaging from space** — revolutionary data capturing the entire auroral oval at once from high-altitude Earth orbit. DE-1's SAI (Spin-Scan Auroral Imager) became the prototype for all subsequent space-borne auroral imagers.
- **Prerequisites**: Paper #8; auroral morphology (오로라 형태), all-sky vs global imaging (전천 vs 전역 영상)
- **Status**: [x]

### 48. An Ultraviolet Auroral Imager for the Viking Spacecraft
- **Authors**: C. D. Anger, S. K. Babey, A. L. Broadfoot, R. G. Brown, L. L. Cogger, R. Gattinger, J. W. Haslett, R. A. King, D. J. McEwen, J. S. Murphree, E. H. Richardson, B. R. Sandel, K. Smith, A. Vallance Jones
- **Year**: 1987
- **Journal**: Geophysical Research Letters, Vol. 14(4), pp. 387–390
- **DOI**: 10.1029/GL014i004p00387
- **Why it matters**: 최초의 UV 전역 오로라 영상기 — 주간 반구에서도 오로라 관측을 가능하게 한 돌파구(UV는 태양광 산란이 낮음). Polar/UVI, IMAGE-FUV 등 후속 UV 영상기의 기술적 기반. / The first UV global auroral imager — a breakthrough enabling auroral observation on the dayside hemisphere (UV wavelengths have low solar scattering background). Technical foundation for subsequent UV imagers including Polar/UVI and IMAGE-FUV.
- **Prerequisites**: Paper #47; UV photometry (UV 광도측정), dayglow suppression (낮측 산란광 억제)
- **Status**: [x]

### 49. A Far Ultraviolet Imager for the International Solar-Terrestrial Physics Mission (Polar/UVI)
- **Authors**: M. R. Torr, D. G. Torr, M. Zukic, R. B. Johnson, J. Ajello, P. Banks, K. Clark, K. Cole, C. Keffer, G. Parks, B. Tsurutani, J. Spann
- **Year**: 1995
- **Journal**: Space Science Reviews, Vol. 71, pp. 329–383
- **DOI**: 10.1007/BF00751335
- **Why it matters**: ISTP 미션의 Polar/UVI 기기를 상세히 기술. Lyman-Birge-Hopfield(LBH) 대역과 130.4·135.6 nm 산소선을 이용해 오로라 에너지 유입·강수 전자 특성을 정량화하는 도구를 제공. / Detailed description of the Polar/UVI instrument, part of the ISTP mission. Provides tools for quantifying auroral energy deposition and precipitating electron characteristics via the Lyman-Birge-Hopfield (LBH) band and the 130.4/135.6 nm oxygen lines.
- **Prerequisites**: Paper #48; auroral emissions (오로라 방출선), electron precipitation spectroscopy (강수 전자 분광)
- **Status**: [x]

### 50. IMAGE Mission Overview
- **Authors**: J. L. Burch
- **Year**: 2000
- **Journal**: Space Science Reviews, Vol. 91, pp. 1–14
- **DOI**: 10.1023/A:1005245323115
- **Why it matters**: IMAGE(Imager for Magnetopause-to-Aurora Global Exploration)는 자기권 전역을 동시에 영상화한 최초의 전용 미션이다. ENA·FUV·RPI·HENA·MENA 다파장 원격감지로 링커런트·플라즈마권·자기권계면·오로라를 동시 관측하는 새 패러다임을 확립. / IMAGE was the first dedicated mission to simultaneously image the entire magnetosphere. Established a new paradigm of multi-wavelength remote sensing (ENA, FUV, RPI, HENA, MENA) that simultaneously observes the ring current, plasmasphere, magnetopause, and aurora.
- **Prerequisites**: Papers #5, #8, #47; ENA (Energetic Neutral Atom) imaging, magnetospheric imaging concept (자기권 영상화 개념)
- **Status**: [x]

### 51. Far Ultraviolet Imaging from the IMAGE Spacecraft. 1. System Design
- **Authors**: S. B. Mende, H. Heetderks, H. U. Frey, M. Lampton, S. P. Geller, B. Habraken, E. Renotte, C. Jamar, P. Rochus, J. Spann, S. A. Fuselier, J.-C. Gerard, R. Gladstone, S. Murphree, L. Cogger
- **Year**: 2000
- **Journal**: Space Science Reviews, Vol. 91, pp. 243–270
- **DOI**: 10.1023/A:1005271728567
- **Why it matters**: IMAGE-FUV는 세 채널(SI: OI 135.6 nm / WIC: LBH 연속체 / GEO: OI 130.4 nm)로 오로라·지구 코로나를 영상화한다. Frey et al. 2004 substorm onset 카탈로그(#52)의 관측적 기반. / IMAGE-FUV images the aurora and geocorona in three channels (SI: OI 135.6 nm / WIC: LBH continuum / GEO: OI 130.4 nm). The observational foundation for the Frey et al. 2004 substorm onset catalog (#52).
- **Prerequisites**: Papers #48, #49, #50; FUV spectroscopy (FUV 분광), auroral brightness modeling (오로라 밝기 모델링)
- **Status**: [x]

### 52. Substorm Onset Observations by IMAGE-FUV
- **Authors**: H. U. Frey, S. B. Mende, V. Angelopoulos, E. F. Donovan
- **Year**: 2004
- **Journal**: Journal of Geophysical Research: Space Physics, Vol. 109, A10304
- **DOI**: 10.1029/2004JA010607
- **Why it matters**: IMAGE-FUV WIC 데이터에서 **2,437개 substorm onset을 자동 식별·카탈로그화**. onset 위치 통계(MLT 분포, 자기위도), 계절/태양풍 의존성 등 후속 연구의 벤치마크 데이터셋. onset과 IMF/태양풍 조건의 상관이 약함을 보여 substorm의 내부 자기권 구동 우세성을 시사. / Automatically identified and catalogued **2,437 substorm onsets** from IMAGE-FUV WIC data. The statistical distributions of onset location (MLT, magnetic latitude) and seasonal/solar-wind dependencies established a benchmark dataset. Weak correlation of onsets with IMF/solar-wind conditions suggests that substorms are internally driven.
- **Prerequisites**: Papers #8, #10, #51; substorm phenomenology (substorm 현상학), automated feature detection (자동 특징 탐지)
- **Status**: [x]

### 53. GUVI: A Hyperspectral Imager for Geospace (TIMED/GUVI)
- **Authors**: L. J. Paxton, A. B. Christensen, D. Morrison, B. Wolven, H. Kil, Y. Zhang, B. S. Ogorzalek, D. C. Humm, J. O. Goldsten, R. DeMajistre, C.-I. Meng
- **Year**: 2004
- **Journal**: Proc. SPIE 5660 (Instruments, Science, and Methods for Geospace and Planetary Remote Sensing), pp. 228–240
- **DOI**: 10.1117/12.579171
- **Why it matters**: TIMED 위성 탑재 GUVI는 FUV 5색 분광 영상으로 **열권 조성(O/N₂), 오로라, 주간 대기광**을 동시 측정한다. 전리층 총전자수(TEC)와 상관된 O/N₂ 비 변동은 현대 우주기상 전리층 모델링의 핵심 입력. / GUVI on TIMED uses FUV 5-color spectral imaging to simultaneously measure **thermospheric composition (O/N₂), aurora, and dayglow**. The O/N₂ ratio variations (correlated with ionospheric TEC) are key inputs to modern space-weather ionospheric modeling.
- **Prerequisites**: Papers #16, #49; thermospheric chemistry (열권 화학), hyperspectral imaging (초분광 영상)
- **Status**: [x]

### 54. Special Sensor Ultraviolet Spectrographic Imager (SSUSI): An Instrument Description
- **Authors**: L. J. Paxton, C.-I. Meng, G. H. Fountain, B. S. Ogorzalek, E. H. Darlington, S. A. Gary, J. O. Goldsten, D. Y. Kusnierkiewicz, S. C. Lee, L. A. Linstrom, J. J. Maynard, K. Peacock, D. F. Persons, B. E. Smith
- **Year**: 1992
- **Journal**: Proc. SPIE 1745 (Instrumentation for Planetary and Terrestrial Atmospheric Remote Sensing), pp. 2–15
- **DOI**: 10.1117/12.60595
- **Why it matters**: 미 공군 DMSP 계열 위성에 탑재된 SSUSI는 **운용 오로라·전리층 모니터**로, NOAA OVATION 오로라 경보 모델의 입력 데이터원이다. 5개 FUV 색 + 크로스트랙 스캔으로 오로라 타원 전역 영상 생성. / SSUSI aboard the USAF DMSP satellites is an **operational aurora/ionosphere monitor** and a data input to the NOAA OVATION aurora forecast model. Uses 5 FUV colors plus cross-track scanning to generate full auroral-oval imagery.
- **Prerequisites**: Paper #53; DMSP mission context (DMSP 미션 맥락), operational vs science instruments (운용 vs 과학 기기)
- **Status**: [x]

### 55. The Fast Auroral SnapshoT (FAST) Mission
- **Authors**: C. W. Carlson, R. F. Pfaff, J. G. Watzin
- **Year**: 1998
- **Journal**: Geophysical Research Letters, Vol. 25, pp. 2013–2016
- **DOI**: 10.1029/98GL01592
- **Why it matters**: FAST는 **오로라 가속 영역(저고도 ~4000 km) in-situ 관측의 결정판**이다. 영상 아닌 입자·장 고속 측정(ms 해상도)으로 오로라 전자의 inverted-V 분포, 평행 전기장, Alfvén 파 등 미시 가속 물리를 규명. / FAST is the **definitive in-situ platform for the auroral acceleration region (low altitude ~4000 km)**. High-cadence particle/field measurements (millisecond resolution, not imaging) characterize inverted-V electron distributions, parallel electric fields, and Alfvén waves — the microphysics of auroral acceleration.
- **Prerequisites**: Papers #8, #47; auroral acceleration physics (오로라 가속 물리), inverted-V electrons (역 V 전자), parallel electric fields (평행 전기장)
- **Status**: [x]

### 56. The THEMIS Array of Ground-Based Observatories for the Study of Auroral Substorms
- **Authors**: S. B. Mende, S. E. Harris, H. U. Frey, V. Angelopoulos, C. T. Russell, E. Donovan, B. Jackel, M. Greffen, L. M. Peticolas
- **Year**: 2008
- **Journal**: Space Science Reviews, Vol. 141, pp. 357–387
- **DOI**: 10.1007/s11214-008-9380-x
- **Why it matters**: THEMIS 위성 5기의 지상 보조 자산 — 북미 20+ 관측소의 전천 카메라(ASI)와 자력계(GMAG) 네트워크. Akasofu 1964(#8)의 지상 관측 방법론을 현대 디지털·네트워크 시대로 확장해 substorm 오로라-자기권 결합 관측의 표준 인프라를 구축. / The ground-based support asset for the five-satellite THEMIS constellation — a network of 20+ all-sky imagers (ASI) and magnetometers (GMAG) across North America. Extends Akasofu's 1964 (#8) approach into the modern digital/networked era, establishing standard infrastructure for coupled aurora-magnetosphere substorm observations.
- **Prerequisites**: Papers #8, #27; all-sky imaging (전천 영상), magnetometer network analysis (자력계 네트워크 분석)
- **Status**: [x]

### 57. The THEMIS All-Sky Imaging Array — System Design and Initial Results from the Prototype Imager
- **Authors**: E. Donovan, S. Mende, B. Jackel, H. Frey, M. Syrjäsuo, I. Voronkov, T. Trondsen, L. Peticolas, V. Angelopoulos, S. Harris, M. Greffen, M. Connors
- **Year**: 2006
- **Journal**: Journal of Atmospheric and Solar-Terrestrial Physics, Vol. 68, pp. 1472–1487
- **DOI**: 10.1016/j.jastp.2005.03.027
- **Why it matters**: THEMIS ASI 네트워크의 기기 설계 상세 — fisheye 렌즈, CCD, 3초 케이던스, 백색광·557.7/630.0 nm 선택, NTP 시각 동기, 데이터 배포 프로토콜. 후속 TREx·MIRACLE 네트워크의 설계 템플릿이 됨. / Detailed instrument design of the THEMIS ASI network — fisheye lens, CCD, 3-second cadence, white-light vs 557.7/630.0 nm line filtering, NTP time sync, and data distribution protocol. The design template for subsequent TREx and MIRACLE networks.
- **Prerequisites**: Paper #56; CCD photometry (CCD 광도측정), NTP time synchronization (NTP 시각 동기)
- **Status**: [x]

### 58. DARN/SuperDARN: A Global View of the Dynamics of High-Latitude Convection
- **Authors**: R. A. Greenwald, K. B. Baker, J. R. Dudeney, M. Pinnock, T. B. Jones, E. C. Thomas, J.-P. Villain, J.-C. Cerisier, C. Senior, C. Hanuise, R. D. Hunsucker, G. Sofko, J. Koehler, E. Nielsen, R. Pellinen, A. D. M. Walker, N. Sato, H. Yamagishi
- **Year**: 1995
- **Journal**: Space Science Reviews, Vol. 71, pp. 761–796
- **DOI**: 10.1007/BF00751350
- **Why it matters**: SuperDARN은 HF 레이더로 **오로라 영역 전리층 불규칙성의 도플러 속도**를 측정해 자기권-전리층 대류 패턴을 글로벌 실시간 관측한다. 남북반구 수십 개 레이더의 국제 네트워크로 Tsyganenko 모델과 함께 표준 대류 맵(APL Potential Maps)의 입력이 된다. / SuperDARN uses HF radars to measure **Doppler velocities of ionospheric irregularities in the auroral region**, providing global real-time observation of magnetosphere-ionosphere convection patterns. An international network of dozens of radars in both hemispheres — standard input for convection maps (APL Potential Maps) alongside Tsyganenko models.
- **Prerequisites**: Papers #12, #13; HF radar and ionospheric backscatter (HF 레이더와 전리층 후방산란), coherent radar Doppler (간섭 레이더 도플러)
- **Status**: [x]

### 59. EISCAT: An Updated Description of Technical Characteristics and Operational Capabilities
- **Authors**: K. Folkestad, T. Hagfors, S. Westerlund
- **Year**: 1983
- **Journal**: Radio Science, Vol. 18(6), pp. 867–879
- **DOI**: 10.1029/RS018i006p00867
- **Why it matters**: EISCAT(European Incoherent Scatter Radar)는 오로라대 전리층의 **전자 밀도(Nₑ), 전자·이온 온도(Tₑ·Tᵢ), 이온 속도(vᵢ)를 직접 측정**하는 VHF/UHF 레이더 시설. 북극권 Tromsø·Kiruna·Sodankylä + Svalbard 배치로 전리층 3D 재구성을 가능케 하는 유일한 비간섭 산란(IS) 인프라. / EISCAT is a VHF/UHF radar facility that **directly measures electron density (Nₑ), electron/ion temperatures (Tₑ/Tᵢ), and ion velocity (vᵢ) in the auroral ionosphere**. The three-site arctic configuration (Tromsø, Kiruna, Sodankylä) plus Svalbard provides the only Incoherent Scatter (IS) infrastructure enabling 3D ionospheric reconstruction.
- **Prerequisites**: Paper #58; Incoherent scatter radar theory (비간섭 산란 레이더 이론), ionospheric plasma parameters (전리층 플라즈마 매개변수)
- **Status**: [x]

### 60. First Observations From the TREx Spectrograph: The Optical Spectrum of STEVE and the Picket Fence Phenomena
- **Authors**: D. M. Gillies, E. Donovan, D. Hampton, J. Liang, M. Connors, Y. Nishimura, B. Gallardo-Lacourt, E. Spanswick
- **Year**: 2019
- **Journal**: Geophysical Research Letters, Vol. 46(13), pp. 7207–7213
- **DOI**: 10.1029/2019GL083272
- **Why it matters**: TREx(Transition Region Explorer)는 THEMIS ASI 후계 지상 오로라 네트워크로 RGB·NIR·분광기 다채널 관측을 제공한다. 본 논문은 TREx 분광기의 first-light 결과로서 **STEVE와 "picket fence" 현상의 광학 스펙트럼**을 최초로 측정 — STEVE가 전통적 오로라가 아닌 서브오로라 이온 드리프트(SAID) 기반 대기광 발광임을 분광학적으로 확증. / TREx is the successor ground-based aurora network to the THEMIS ASI array, providing RGB, NIR, and spectrographic multi-channel observations. This paper reports the TREx spectrograph's first light — **first optical spectral measurements of STEVE and the "picket fence" phenomena** — spectroscopically confirming that STEVE is not a traditional aurora but a SAID (subauroral ion drift)-driven airglow emission.
- **Prerequisites**: Paper #57; subauroral ion drift (SAID) (서브오로라 이온 드리프트), STEVE phenomenon (STEVE 현상), spectrographic imaging (분광 영상)
- **Status**: [x]

---

## Mission Instrument Papers / 미션별 기기 논문 (Wind · ACE · IMAGE · THEMIS)

### 61. The WIND Magnetic Field Investigation (Wind/MFI)
- **Authors**: R. P. Lepping, M. H. Acuña, L. F. Burlaga, W. M. Farrell, J. A. Slavin, K. H. Schatten, F. Mariani, N. F. Ness, F. M. Neubauer, Y. C. Whang, J. B. Byrnes, R. S. Kennon, P. V. Panetta, J. Scheifele, E. M. Worley
- **Year**: 1995
- **Journal**: *Space Science Reviews*, Vol. 71, pp. 207–229
- **DOI**: 10.1007/BF00751330
- **Why it matters**: Wind/MFI는 듀얼 트라이엑시얼 fluxgate magnetometer로 1994년 발사 이후 30년+ L1 지점에서 행성간 자기장(IMF) 표준 데이터를 제공. ICME·자기 구름·충격파 연구의 기준선. / Wind's dual triaxial fluxgate magnetometer has provided the standard interplanetary magnetic field (IMF) data at L1 since 1994 launch, the baseline for ICMEs, magnetic clouds, and shock studies.
- **Prerequisites**: Papers #4, #43; fluxgate magnetometer principles, IMF and Parker spiral, GSE/GSM coordinates / 플럭스게이트 자력계 원리, IMF·Parker 나선, GSE/GSM 좌표계
- **Status**: [x]

### 62. SWE: A Comprehensive Plasma Instrument for the Wind Spacecraft
- **Authors**: K. W. Ogilvie, D. J. Chornay, R. J. Fritzenreiter, F. Hunsaker, J. Keller, J. Lobell, G. Miller, J. D. Scudder, E. C. Sittler, R. B. Torbert, D. Bodet, G. Needell, A. J. Lazarus, J. T. Steinberg, J. H. Tappan, A. Mavretic, E. Gergin
- **Year**: 1995
- **Journal**: *Space Science Reviews*, Vol. 71, pp. 55–77
- **DOI**: 10.1007/BF00751326
- **Why it matters**: Wind/SWE는 두 Faraday cup, VEIS, strahl 센서를 결합해 양성자·알파·전자 모멘트(밀도·속도·온도)를 정밀 측정. 우주환경 연구에서 가장 오래 운용 중인 태양풍 플라즈마 표준 데이터셋의 기반. / SWE combines two Faraday cups, a VEIS, and a strahl sensor to precisely measure proton/alpha/electron moments — underpinning one of the longest-running solar-wind plasma datasets, foundational to virtually every L1-based space-weather analysis.
- **Prerequisites**: Paper #61; solar wind plasma physics, Faraday cup and ESA principles, VDF moments, electron strahl/halo / 태양풍 플라즈마 물리, Faraday cup·ESA 원리, VDF 모멘트, 전자 strahl/halo
- **Status**: [x]

### 63. A Three-Dimensional Plasma and Energetic Particle Investigation for the Wind Spacecraft (Wind/3DP)
- **Authors**: R. P. Lin, K. A. Anderson, S. Ashford, C. Carlson, D. Curtis, R. Ergun, D. Larson, J. McFadden, M. McCarthy, G. K. Parks, H. Rème, J. M. Bosqued, J. Coutelier, F. Cotin, C. d'Uston, K.-P. Wenzel, T. R. Sanderson, J. Henrion, J. C. Ronnet, G. Paschmann
- **Year**: 1995
- **Journal**: *Space Science Reviews*, Vol. 71, pp. 125–153
- **DOI**: 10.1007/BF00751328
- **Why it matters**: Wind/3DP는 solid-state telescope과 top-hat ESA를 결합해 태양풍부터 저에너지 우주선까지 suprathermal 전자/이온 3D 분포를 ~3초 분해능으로 측정. 입자 가속·파동-입자 상호작용·충격파 가속·SEP 연구의 표준 도구. / Wind/3DP measures the full 3D distribution of suprathermal electrons and ions from solar wind to low-energy cosmic rays at ~3 s resolution. The reference for particle acceleration, wave–particle interactions, shock acceleration, and SEP events.
- **Prerequisites**: Paper #62; top-hat ESA and SST principles, suprathermal/SEP physics, shock acceleration / top-hat ESA·SST 원리, suprathermal/SEP 물리, 충격파 가속
- **Status**: [x]

### 64. WAVES: The Radio and Plasma Wave Investigation on the WIND Spacecraft
- **Authors**: J.-L. Bougeret, M. L. Kaiser, P. J. Kellogg, R. Manning, K. Goetz, S. J. Monson, N. Monge, L. Friel, C. A. Meetre, C. Perche, L. Sitruk, S. Hoang
- **Year**: 1995
- **Journal**: *Space Science Reviews*, Vol. 71, pp. 231–263
- **DOI**: 10.1007/BF00751331
- **Why it matters**: Wind/WAVES는 DC–14 MHz 광대역 전파/플라즈마 파동을 측정하여 태양 type II/III 폭발, Langmuir 파, 행성간 충격파 라디오 방출, 지구권 라디오 현상을 동시에 관측. STEREO/WAVES와 PSP/FIELDS의 직접적 선행자. / Wind/WAVES measures broadband radio and plasma waves DC to 14 MHz, simultaneously observing solar type II/III bursts, Langmuir waves, IP-shock-driven emissions, and geospace radio phenomena — direct predecessor of STEREO/WAVES and PSP/FIELDS.
- **Prerequisites**: Paper #61; plasma wave theory (Langmuir/whistler/ion-acoustic), type II/III/IV classification, goniopolarimetry / 플라즈마 파동 이론, type II/III/IV 분류, goniopolarimetry
- **Status**: [x]

### 65. The ACE Magnetic Fields Experiment (ACE/MAG)
- **Authors**: C. W. Smith, J. L'Heureux, N. F. Ness, M. H. Acuña, L. F. Burlaga, J. Scheifele
- **Year**: 1998
- **Journal**: *Space Science Reviews*, Vol. 86, pp. 613–632
- **DOI**: 10.1023/A:1005092216668
- **Why it matters**: ACE/MAG은 L1 지점 IMF 벡터 성분을 연속 측정하여 우주기상 예보의 핵심 입력값(Bz)을 제공하며, 다른 ACE 입자 관측의 자기장 컨텍스트로 필수. 듀얼 트라이엑시얼 fluxgate 설계로 우주선 자기 잡음 최소화. / ACE/MAG provides continuous vector IMF measurements at L1, supplying the critical Bz input for space-weather forecasting and the magnetic context essential to all other ACE particle observations.
- **Prerequisites**: Papers #4, #42, #61; fluxgate magnetometer, IMF and Parker spiral, solar-wind MHD basics / 플럭스게이트 자력계, IMF·Parker 나선, 태양풍 MHD 기초
- **Status**: [x]

### 66. Solar Wind Electron Proton Alpha Monitor (SWEPAM) for ACE
- **Authors**: D. J. McComas, S. J. Bame, P. Barker et al.
- **Year**: 1998
- **Journal**: *Space Science Reviews*, Vol. 86, pp. 563–612
- **DOI**: 10.1023/A:1005040232597
- **Why it matters**: SWEPAM은 ACE의 핵심 태양풍 벌크 플라즈마 관측 장비로, 양성자·알파·전자 3D 분포 측정으로 밀도·속도·온도 산출. Ulysses 비행 예비품을 재활용한 비용 효율적 설계로 L1 실시간 우주기상 모니터링 가능. / SWEPAM is ACE's primary bulk solar-wind plasma instrument, measuring 3D velocity distributions of protons, alphas, electrons. Cost-effectively built from Ulysses flight spares, it enables real-time L1 space-weather monitoring.
- **Prerequisites**: Paper #42; ESA principles, solar-wind plasma parameters, VDF, MCP detectors / ESA 원리, 태양풍 플라즈마 매개변수, VDF, MCP 검출기
- **Status**: [x]

### 67. The Solar Isotope Spectrometer (ACE/SIS)
- **Authors**: E. C. Stone, C. M. S. Cohen, W. R. Cook, A. C. Cummings et al.
- **Year**: 1998
- **Journal**: *Space Science Reviews*, Vol. 86, pp. 357–408
- **DOI**: 10.1023/A:1005027929871
- **Why it matters**: SIS는 He–Zn(Z=2–30) 영역의 10–100 MeV/nuc 입자에 대해 고분해능 동위원소 측정을 수행하여 대규모 SEP 사건에서 태양 코로나 조성을 직접 결정하고 입자 가속을 연구. 정온기에는 갤럭시·비정상 우주선 동위원소 조성 측정. / SIS performs high-resolution isotopic measurements of energetic nuclei from He to Zn over 10–100 MeV/nuc, directly determining coronal composition during large SEP events and probing acceleration. During quiet times measures isotopes of galactic and anomalous cosmic rays.
- **Prerequisites**: Papers #42, #46; dE/dx vs. total energy, silicon SSDs, isotope separation, SEP events / dE/dx vs 총에너지, 실리콘 SSD, 동위원소 분리, SEP 사건
- **Status**: [x]

### 68. The Cosmic-Ray Isotope Spectrometer (ACE/CRIS)
- **Authors**: E. C. Stone, C. M. S. Cohen, W. R. Cook, A. C. Cummings et al.
- **Year**: 1998
- **Journal**: *Space Science Reviews*, Vol. 86, pp. 285–356
- **DOI**: 10.1023/A:1005075813033
- **Why it matters**: CRIS는 ACE 에너지 범위 최상단(50–500 MeV/nuc)을 담당하며 ~250 cm² sr의 큰 기하학적 인자로 정온기 갤럭시 우주선 동위원소 조성을 전례 없는 정밀도로 측정. SOFT 호도스코프 + 실리콘 스택으로 수십 종 동위원소 분리 — 핵합성과 우주선 기원 제약. / CRIS covers ACE's highest decade (50–500 MeV/nuc) with large ~250 cm² sr geometry, measuring galactic cosmic-ray isotopic composition with unprecedented statistics — SOFT hodoscope + Si stacks resolve dozens of isotopes constraining nucleosynthesis.
- **Prerequisites**: Paper #67; galactic cosmic-ray propagation, nucleosynthesis, multiple dE/dx, scintillating optical fiber tracking / 갤럭시 우주선 전파, 핵합성, 다중 dE/dx, 섬광 광섬유 추적기
- **Status**: [x]

### 69. The Ultra-Low-Energy Isotope Spectrometer (ACE/ULEIS)
- **Authors**: G. M. Mason, R. E. Gold, S. M. Krimigis, J. E. Mazur, G. B. Andrews, K. A. Daley, J. R. Dwyer, K. F. Heuerman, T. L. James, M. J. Kennedy, T. LeFevere, H. Malcolm, B. Tossman, P. H. Walpole
- **Year**: 1998
- **Journal**: *Space Science Reviews*, Vol. 86, pp. 409–448
- **DOI**: 10.1023/A:1005079930780
- **Why it matters**: ULEIS는 He–Ni 원소를 ~45 keV/nuc부터 수 MeV/nuc까지 초고분해능 질량 분석으로 측정하여 SEP·행성간 충격파·CIR에서 가속된 저에너지 입자 조성을 연구. SIS·CRIS와 결합해 ACE의 에너지 범위를 5–6 데케이드로 확장. / ULEIS provides ultra-high-resolution mass analysis of He–Ni from ~45 keV/nuc to a few MeV/nuc, studying particles accelerated in SEPs, IP shocks, and CIRs. Combined with SIS and CRIS extends ACE coverage across 5–6 decades.
- **Prerequisites**: Papers #46, #67; TOF spectroscopy, SEP acceleration mechanisms, IP shocks / TOF 분광, SEP 가속 메커니즘, 행성간 충격파
- **Status**: [x]

### 70. Electron, Proton, and Alpha Monitor on ACE (ACE/EPAM)
- **Authors**: R. E. Gold, S. M. Krimigis, S. E. Hawkins III, D. K. Haggerty, D. A. Lohr, E. Fiore, T. P. Armstrong, G. Holland, L. J. Lanzerotti
- **Year**: 1998
- **Journal**: *Space Science Reviews*, Vol. 86, pp. 541–562
- **DOI**: 10.1023/A:1005088115759
- **Why it matters**: EPAM은 5개 독립 솔리드스테이트 검출기 망원경으로 이온(≳50 keV)과 전자(≳40 keV)를 광범위 에너지·강도 영역에서 측정. Ulysses HI-SCALE 유산을 활용하여 SEP·자기권 폭발·충격 가속 입자에 대한 빠른 시간 분해능 데이터 제공. / EPAM uses five independent SSD telescopes to measure ions (≳50 keV) and electrons (≳40 keV). Building on Ulysses HI-SCALE heritage, delivers high-time-resolution data on SEPs, magnetospheric bursts, and shock-accelerated particles.
- **Prerequisites**: Papers #5, #42; SSD telescopes, particle telescope geometry, SEP and electron events / SSD 망원경, 입자 망원경 기하, SEP·전자 사건
- **Status**: [x]

### 71. SWICS and SWIMS on ACE: Solar and Interstellar Composition via Solar Wind and Pickup Ions
- **Authors**: G. Gloeckler, J. Cain, F. M. Ipavich, E. O. Tums, P. Bedini, L. A. Fisk, T. H. Zurbuchen, P. Bochsler, J. Fischer, R. F. Wimmer-Schweingruber, J. Geiss, R. Kallenbach
- **Year**: 1998
- **Journal**: *Space Science Reviews*, Vol. 86, pp. 497–539
- **DOI**: 10.1023/A:1005036131689
- **Why it matters**: SWICS는 H–Fe까지 태양풍 이온의 화학 조성·전하 상태·열속도를 결정하고 픽업 이온을 통해 성간 중성 물질을 탐사. SWIMS는 He–Ni 모든 원소에 대한 화학·동위원소·전하 상태 조성을 정밀 측정하여 코로나 온도와 기원을 진단. / SWICS determines chemical composition, charge states, and thermal speeds of solar-wind ions H–Fe, and probes interstellar neutrals via pickup ions. SWIMS precisely measures chemistry, isotopes, and charge states for every element He–Ni — diagnosing coronal temperatures and source regions.
- **Prerequisites**: Paper #42; ESA + TOF + energy spectroscopy, freeze-in charge states, pickup ion physics / ESA + TOF + 에너지 분광법, 동결 전하 상태, 픽업 이온 물리
- **Status**: [x]

### 72. High Energy Neutral Atom Imager (IMAGE/HENA)
- **Authors**: D. G. Mitchell, S. Jaskulek, C. E. Schlemm, E. P. Keath, R. E. Thompson, B. E. Tossman, J. D. Boldt, J. R. Hayes, G. B. Andrews, N. Paschalidis, D. C. Hamilton, R. A. Lundgren, E. O. Tims, P. Williams, S. M. Krimigis, E. C. Roelof
- **Year**: 2000
- **Journal**: *Space Science Reviews*, Vol. 91, pp. 67–112
- **DOI**: 10.1023/A:1005207308094
- **Why it matters**: HENA는 ~50 keV/nuc 이상 고에너지 ENA 영상기로, 지자기 폭풍 동안 자기권 내부 환전류 이온의 전역 이미지를 2분 시간분해능·8° 각분해능으로 처음 제공. 내자기권 동역학을 'in-situ'가 아닌 원격 영상으로 재구성하는 패러다임 전환. / HENA is the first high-energy ENA imager (≥50 keV/nuc) delivering 2-min, 8°-resolution global images of inner-magnetospheric ring-current ions — enabling remote-sensing reconstruction of dynamics rather than in-situ point measurements.
- **Prerequisites**: Paper #50; ENA imaging, ring-current physics, charge-exchange cross sections, MCP/SSD + TOF / ENA 영상화, 환전류 물리, 전하 교환 단면적, MCP/SSD + TOF
- **Status**: [x]

### 73. Medium Energy Neutral Atom Imager (IMAGE/MENA)
- **Authors**: C. J. Pollock, K. Asamura, J. Baldonado, M. M. Balkey, P. Barker, J. L. Burch, E. J. Korpela, T. Cravens et al.
- **Year**: 2000
- **Journal**: *Space Science Reviews*, Vol. 91, pp. 113–154
- **DOI**: 10.1023/A:1005259324933
- **Why it matters**: MENA는 1–30 keV ENA를 슬릿형 이방향 동시 영상화로 포착하여 환전류 형성·소멸과 substorm 주입을 추적. HENA(고에너지)와 LENA(저에너지) 사이 결정적 중간 에너지 윈도우를 채움. / MENA captures 1–30 keV ENAs through slit-based 2D simultaneous imaging, allowing tracking of ring-current build-up/decay and substorm injections — filling the critical middle-energy window between HENA and LENA.
- **Prerequisites**: Paper #72; slit/pinhole optics, carbon-foil TOF, ring-current dynamics, substorm injection physics / 슬릿/핀홀 광학, carbon-foil TOF, 환전류 동역학, substorm 주입 물리
- **Status**: [x]

### 74. The Low-Energy Neutral Atom Imager for IMAGE (LENA)
- **Authors**: T. E. Moore, D. J. Chornay, M. R. Collier, F. A. Herrero, J. Johnson, M. A. Johnson, J. W. Keller, J. F. Laudadio, J. F. Lobell, K. W. Ogilvie, P. Rozmarynowski, S. A. Fuselier, A. G. Ghielmetti, R. Hertzberg, D. C. Holland, P.-K. Tian, W. K. Peterson, K. C. Hsieh, D. C. Curtis
- **Year**: 2000
- **Journal**: *Space Science Reviews*, Vol. 91, pp. 155–195
- **DOI**: 10.1023/A:1005211509003
- **Why it matters**: LENA는 10–750 eV ENA를 표면 변환(atom-to-negative-ion) 기술로 영상화한 최초의 우주기반 LENA 이미저로, 극위 이온 유출과 태양풍-자기권 경계영역의 저에너지 중성입자 분포(H, O 조성)를 처음 전역 추적. / LENA is the first space-borne LENA imager using atom-to-negative-ion surface conversion to image 10–750 eV neutrals, providing the first global view of polar ionospheric outflow and low-energy neutrals at the solar-wind/magnetosphere boundary.
- **Prerequisites**: Paper #72; surface ionization/conversion physics, polar wind/cleft outflow, low-energy particle detection / 표면 이온화/변환 물리, 극풍·cleft 유출, 저에너지 입자 검출
- **Status**: [x]

### 75. The Extreme Ultraviolet Imager Investigation for IMAGE (EUV)
- **Authors**: B. R. Sandel, A. L. Broadfoot, C. C. Curtis, R. A. King, T. C. Stone, R. H. Hill, J. Chen, O. H. W. Siegmund, R. Raffanti, D. D. Allred, R. S. Turley, D. L. Gallagher
- **Year**: 2000
- **Journal**: *Space Science Reviews*, Vol. 91, pp. 197–242
- **DOI**: 10.1023/A:1005263510820
- **Why it matters**: IMAGE/EUV는 He+ 30.4 nm 공명 산란선을 통해 차가운 플라즈마권 분포·동역학을 처음 전역 영상화. 3개의 30° 시야 센서 헤드를 결합해 플라즈마권 플룸·노치 등 폭풍 응답 직접 시각화. / IMAGE/EUV provides the first global imaging of the cold plasmasphere via resonantly-scattered He+ 30.4 nm emission. Three 30° FOV sensor heads visualize plasmaspheric plumes, notches, and storm-time responses.
- **Prerequisites**: Paper #50; resonant scattering, He+ optical depth, plasmasphere formation/erosion, EUV multilayer optics / 공명 산란, He+ 광학적 깊이, 플라즈마권 형성/침식, EUV 다층막 광학
- **Status**: [x]

### 76. The Radio Plasma Imager Investigation on IMAGE (RPI)
- **Authors**: B. W. Reinisch, D. M. Haines, K. Bibl, G. Cheney, I. A. Galkin, X. Huang, S. H. Myers, G. S. Sales, R. F. Benson, S. F. Fung, J. L. Green, S. Boardsen, W. W. L. Taylor, J.-L. Bougeret, R. Manning, N. Meyer-Vernet, M. Moncuquet, D. L. Carpenter, D. L. Gallagher, P. Reiff
- **Year**: 2000
- **Journal**: *Space Science Reviews*, Vol. 91, pp. 319–359
- **DOI**: 10.1023/A:1005252602159
- **Why it matters**: RPI는 능동 레이더 사운딩(3 kHz–3 MHz)으로 자기권 내 원격 플라즈마 밀도 분포를 측정한 최초의 우주 임무. 플라즈마권계·자기경계·polar cap 경계의 거리·구조 직접 영상화 — topside 사운딩의 자기권 확장. / RPI is the first spaceborne active radio sounder (3 kHz–3 MHz) to remotely measure magnetospheric plasma density distributions, directly imaging the plasmapause, magnetopause, and polar-cap boundary distances — extending topside sounding into the magnetosphere.
- **Prerequisites**: Paper #50; EM wave propagation in plasma (cutoffs/resonances), ionosonde principles, FFT signal processing, long dipole antenna / 플라즈마 내 EM 파 전파, ionosonde 원리, FFT 신호처리, 긴 다이폴 안테나
- **Status**: [x]

### 77. The THEMIS Mission (Satellite Suite Overview)
- **Authors**: V. Angelopoulos
- **Year**: 2008
- **Journal**: *Space Science Reviews*, Vol. 141, pp. 5–34
- **DOI**: 10.1007/s11214-008-9336-1
- **Why it matters**: THEMIS 미션의 정식 개요 논문으로, 5기 위성 궤도 설계(상승점 정렬 대접합), 과학 목표(substorm 트리거 위치 결정), GBO(#56) 지상 자력계·광학 네트워크와의 통합 운용을 종합 기술. 모든 THEMIS 도구 논문이 인용하는 미션 차원 최상위 참조. / The official mission overview paper for THEMIS — comprehensively describing the 5-spacecraft orbital design (apogee-aligned major conjunctions), substorm-trigger science objectives, and integrated operation with the GBO (#56). The top-level mission reference cited by all THEMIS instrument papers.
- **Prerequisites**: Papers #8, #10, #56; magnetospheric substorm phenomenology, multi-spacecraft mission concepts / 자기권 substorm 현상학, 다중 위성 미션 개념
- **Status**: [x]

### 78. The THEMIS Fluxgate Magnetometer (THEMIS/FGM)
- **Authors**: H. U. Auster, K. H. Glassmeier, W. Magnes, O. Aydogar, W. Baumjohann, D. Constantinescu, D. Fischer, K. H. Fornacon, E. Georgescu, P. Harvey, O. Hillenmaier, R. Kroth, M. Ludlam, Y. Narita, R. Nakamura, K. Okrafka, F. Plaschke, I. Richter, H. Schwarzl, B. Stoll, A. Valavanoglou, M. Wiedemann
- **Year**: 2008
- **Journal**: *Space Science Reviews*, Vol. 141, pp. 235–264
- **DOI**: 10.1007/s11214-008-9365-9
- **Why it matters**: FGM은 THEMIS 5기 위성 클러스터의 핵심 자기장 측정기로, 0.01 nT 분해능과 64 Hz까지의 주파수 응답으로 substorm 개시 단계의 자기권 급격한 재구성을 다중 지점에서 동시 관측. 자기 재결합·전류시트 운동·자기경계 식별의 기준 측정. / FGM is the core magnetometer for the THEMIS 5-spacecraft constellation (0.01 nT resolution, up to 64 Hz), enabling simultaneous multi-point observation of magnetospheric reconfigurations during substorm onset.
- **Prerequisites**: Papers #61, #65, #77; fluxgate magnetometer principles, substorm/current systems, spacecraft magnetic cleanliness / 플럭스게이트 자력계 원리, substorm·전류계, 우주선 자기 청결도
- **Status**: [x]

### 79. The Search Coil Magnetometer for THEMIS (THEMIS/SCM)
- **Authors**: A. Roux, O. Le Contel, C. Coillot, A. Bouabdellah, B. de la Porte, D. Alison, S. Ruocco, M. C. Vassal
- **Year**: 2008
- **Journal**: *Space Science Reviews*, Vol. 141, pp. 265–275
- **DOI**: 10.1007/s11214-008-9455-8
- **Why it matters**: SCM은 0.1 Hz–4 kHz 자기장 변동을 측정하여 FGM의 DC/저주파 측정을 보완. substorm 개시 영역의 휘슬러파·이온사이클로트론파·ULF/ELF 난류 진단으로 파동-입자 상호작용·무충돌 산일 메커니즘 연구의 핵심 도구. / SCM measures magnetic fluctuations 0.1 Hz–4 kHz, complementing FGM's DC/low-frequency data — diagnoses electromagnetic waves (whistler, ion-cyclotron, ULF/ELF) in substorm onset regions, essential for wave–particle interactions and collisionless dissipation.
- **Prerequisites**: Paper #78; induction coil magnetometer theory, plasma wave modes (whistler, ion-cyclotron, lower-hybrid), Fourier/spectral analysis / 유도 코일 자력계 이론, 플라즈마 파동 모드, Fourier/스펙트럼 분석
- **Status**: [x]

### 80. The THEMIS ESA Plasma Instrument and In-flight Calibration (THEMIS/ESA)
- **Authors**: J. P. McFadden, C. W. Carlson, D. Larson, M. Ludlam, R. Abiad, B. Elliott, P. Turin, M. Marckwordt, V. Angelopoulos
- **Year**: 2008
- **Journal**: *Space Science Reviews*, Vol. 141, pp. 277–302
- **DOI**: 10.1007/s11214-008-9440-2
- **Why it matters**: THEMIS/ESA는 top-hat ESA로 수 eV–30 keV(전자) 및 수 eV–25 keV(이온) 플라즈마 분포를 측정하여 substorm 시 플라즈마 시트 동력학·경계층 흐름·단열 가열을 정량화. 5기 위성 동일 사양으로 다중 지점 모멘트 비교 가능. / THEMIS/ESA uses top-hat ESAs to measure plasma distribution functions over a few eV–30 keV (electrons) and eV–25 keV (ions), quantifying plasma-sheet dynamics, boundary-layer flows, and adiabatic heating during substorms — identical units on all five spacecraft enable multi-point moment comparison.
- **Prerequisites**: Paper #77; top-hat ESA principles, particle distribution function moments, plasma sheet/magnetosheath populations / top-hat ESA 원리, 입자 분포 모멘트, 플라즈마 시트·자기초 모집단
- **Status**: [x]

### 81. The Electric Field Instrument (EFI) for THEMIS
- **Authors**: J. W. Bonnell, F. S. Mozer, G. T. Delory, A. J. Hull, R. E. Ergun, C. M. Cully, V. Angelopoulos, P. R. Harvey
- **Year**: 2008
- **Journal**: *Space Science Reviews*, Vol. 141, pp. 303–341
- **DOI**: 10.1007/s11214-008-9469-2
- **Why it matters**: EFI는 20 m 반경 4쌍 와이어 붐과 두 개의 축방향 stacer 붐으로 3축 전기장(DC~수 kHz)과 위성 전위를 측정. 대류 전기장·정상상태 전기장·substorm 개시 시 파동/난류를 직접 관측 — FGM과 결합해 E×B 흐름과 Poynting 플럭스 계산의 핵심. / EFI uses four pairs of 20 m wire booms plus two axial stacer booms to measure 3-axis E-fields (DC to several kHz) and spacecraft potential — combined with FGM provides the key inputs for E×B drift and Poynting flux calculations.
- **Prerequisites**: Paper #78; double-probe technique, spacecraft sheath/charging physics, MHD electric fields, wave–particle interactions / 이중 탐침법, 우주선 차폐/대전 물리, MHD 전기장, 파동-입자 상호작용
- **Status**: [x]
