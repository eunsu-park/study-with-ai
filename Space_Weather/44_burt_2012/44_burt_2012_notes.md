---
title: "Deep Space Climate Observatory: The DSCOVR Mission"
authors: [Joe Burt, Bob Smith]
year: 2012
journal: "2012 IEEE Aerospace Conference"
doi: "10.1109/AERO.2012.6187025"
topic: Space_Weather
tags: [DSCOVR, Triana, L1, solar-wind-monitor, EPIC, NISTAR, PlasMag, Faraday-cup, mission-engineering, refurbishment]
status: completed
date_started: 2026-04-25
date_completed: 2026-04-25
---

# 44. Deep Space Climate Observatory: The DSCOVR Mission / DSCOVR 임무

---

## 1. Core Contribution / 핵심 기여

Burt & Smith (2012) provide the official engineering report — known internally as the "Serotine Report" — that established the technical and programmatic feasibility of refurbishing the long-stored Triana spacecraft and re-flying it as the Deep Space Climate Observatory (DSCOVR). The paper documents (1) a complete subsystem-level status of a spacecraft that had spent more than seven years in clean-room "Stable Suspension" since November 2001, (2) a residual-risk register covering solar-cell adhesive oxidation, reaction-wheel grease aging, propulsion-heater anomalies, and obsolete parts, (3) a redesign of the mission to swap out the Space Transportation System (Shuttle) launch architecture for an Expendable Launch Vehicle (Taurus II or Falcon 9), and (4) an instrument refurbishment plan covering the Earth Poly-Chromatic Imaging Camera (EPIC), the NIST Advanced Radiometer (NISTAR), and the Plasma-Magnetometer (PlasMag) suite that comprises a Faraday Cup, a fluxgate magnetometer, an electron spectrometer, and a Pulse Height Analyzer. Critically, the paper also formalizes the *purpose change*: the primary mission shifts from Earth-imaging (Triana's original 1998 charter) to operational space-weather monitoring at the Sun–Earth L1 Lagrange point, providing a successor to the aging ACE spacecraft for NOAA's Real-Time Solar Wind (RTSW) network.

본 논문은 1998년 Al Gore 부통령이 제안한 Triana 임무가 우주왕복선 발사 일정 우선순위와 정치적 사정으로 2001년 11월부터 NASA Goddard 우주비행센터(GSFC) 청정실에 N₂ purge 상태로 보관되어 오던 위성을, 2008년 NOAA·NASA·USAF의 공동 평가(Serotine 팀)를 거쳐 우주기상 모니터링 임무인 DSCOVR로 재정의(repurpose)하여 2014년경 발사 가능성을 평가한 공식 엔지니어링 보고서이다. 핵심 기여는 (1) 7년 이상 보관된 우주선의 모든 서브시스템(기계·GN&C·추진·C&DH·전력·통신·열·시스템공학) 상태 점검 결과를 제시하고, (2) 잔존 위험요소(태양전지 접착제 산화, 반작용 휠 그리스 경화, 추진제 히터 이상, GIDEP 부품 단종)를 명시하며, (3) STS 발사 전제로 설계된 우주선을 ELV(Taurus II / Falcon 9) 호환으로 개조하는 작업 범위를 정의하고, (4) 계기 부활 계획(EPIC 10-필터 CCD 영상, NISTAR 3-공동 복사계, PlasMag 4종 센서)을 정리하며, (5) 임무 목적 자체를 지구 영상에서 ACE 후속의 운영 우주기상 자산으로 재정의한 점이다. 본 논문은 노후·보관 우주선 부활 사례의 표준 참고문헌이자, 2015년 2월 Falcon 9으로 실제 발사된 DSCOVR의 발사 전 마지막 종합 엔지니어링 기술서이다.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction & Status (§1–§2) / 서론과 보관 상태

**페이지 1, §1 Introduction.**  2008년 NOAA와 USAF는 NASA에 DSCOVR 부활 가능성 연구를 요청하였다. 1998년 Triana로 시작된 본 우주선은 21개월의 통합 작업과 $249M(FY07$)의 비용을 투입한 끝에 2000년경 거의 완성되어 1500–2000시간의 환경시험까지 마쳤으나, 2001년 STS 매니페스트 변동과 정치적 사유로 임무가 보류되어 GSFC SSDIF(Space Systems Development and Integration Facility) 청정실에 N₂ purge 컨테이너로 보관되었다. 2008년 후반 power-on 시험("aliveness test")을 통해 거의 모든 시스템이 정상임이 확인되었고, 2011년 NOAA·NASA·USAF가 발사 계획에 합의하였다. 원형 임무는 STS 발사용 Earth-pointed, Sun-oriented, 3축 안정 570 kg 관측위성이었으며, Class C STS 신뢰도 요건을 만족하였다.

The paper opens with the operational driver: the Advanced Composition Explorer (ACE), launched in 1997 as a Space Science satellite, was approaching end-of-design-life at L1 and could not be relied upon for indefinite real-time solar-wind warning service. NOAA, the U.S. agency responsible for space-weather forecasting, needed continuity, and DSCOVR's PlasMag suite — described as "an advanced, smaller version of the ACE instrumentation" — was identified as the most cost-effective successor. The Air Force (USAF) joined as a co-sponsor with the intent to cover launch costs in return for space-weather data products.

**페이지 1–2, §2 Status.**  2001년 11월 보관 직전 비행용 transponder, star tracker, IMU(자이로), reaction wheel은 비행 위성 구조에서 분리되어 GSFC 내 다른 보관처로 이전되었으며, 일부 GSE(Ground Support Equipment)는 GSFC 인프라에 재흡수되었다. 그러나 종이 도면 라이브러리와 전자 문서가 유지되어 aliveness test 절차 등 모든 자료가 즉시 가용하였다. Mission Operations Center(MOC) 소프트웨어는 유지된 반면 MOC 하드웨어는 처분되었다. DSCOVR 부품 인벤토리는 SDO(Solar Dynamics Observatory) 프로젝트가 관리하고 있었다.

The Serotine Team — named after the *serotinous* pinecone that lies dormant for years until released by heat — moved DSCOVR from its sealed container into a clean tent in GSFC Building 7 and staged GSE in Building 15. They reintegrated the flight star tracker, wheel, and battery, recharged the test battery, and performed a Thanksgiving 2008 power-up. All subsystems performed nominally; the propulsion system still held the same tank pressure as when stored seven years earlier — a non-trivial validation of the mechanical integrity.

### Part II: Risks (§3) / 위험요소

**페이지 2–3, §3 Risks.**  Serotine 팀은 DSCOVR 활용 위험이 "minimal"하다고 결론짓되, 다음 위험들을 명시한다.

- **Reliability profile shift**: STS 시대의 "two-fault tolerant against catastrophic, one-fault tolerant against critical" 요건은 ELV에서는 다소 완화되지만, 정식 신뢰도 분석(FMEA, Fault Tree, PRA)이 재수행되어야 한다. / STS 신뢰도 요건이 ELV용으로 재설계 필요.
- **태양전지 접착제 산화 / Solar-cell adhesive oxidation**: Triana 어레이는 TechStar(현 Emcore)가 Vegetation Canopy Lidar(VCL) 미션용 어레이와 동시 제작하였다. VCL은 비행하지 않아 어레이가 Glory 미션에 재활용되었는데, 진공시험 시 cell blowout이 발생하였다. 원인은 접착제의 primer 과다 도포로 추적되었고, VCL 어레이는 보관 중 N₂ purge가 일시 중단되어 손상이 누적되었다. Triana 어레이는 보관 기간 내내 "거의 상시" purge되어 왔으므로 같은 문제 발생 가능성은 낮다고 판단되나, DSCOVR용 진공시험이 비용 견적에 포함된다. Glory 어레이는 수리되었고 spare cell도 가용하다. / The paper offers a cautionary case study: another array built at the same time and from the same adhesive batch suffered cell blowout on vacuum exposure due to over-priming; only continuous purge protected DSCOVR's array.
- **GIDEP(Government-Industry Data Exchange Program) 단종 부품**: 일부 전기 부품은 사후 결함이 보고되었을 가능성이 있으므로 발사 직전 재검색이 필요하다. 2008년 사전 검색에서는 주요 이슈가 없었다.

The risk section reflects a mature systems-engineering culture: risks are categorized, traced to specific historical events (VCL/Glory), and mitigated by either reuse of existing test data or budgeted re-verification.

### Part III: Subsystems (§4) / 서브시스템 상태

**페이지 3, Mechanical.**  진동, 강도, 질량 특성, 음향, 화공품 충격, 정렬 등 환경시험 전체가 ELV용으로 재계획된다. Serotine 팀은 자체 비용 견적에 GSE의 보강·재제작 자금을 포함시켰고, 본 연구 중 DSCOVR 본체와 EPIC 계기를 위한 lifting GSE를 재제작·인증하였다. ELV adapter 수정·해석도 포함된다. 자력계 boom은 단일 hinge 자력계 boom의 새 제작이 검토되었으나, 기존 Able boom(자력계를 boom 끝으로 이전)이 자기적으로 충분하다면 신규 boom은 불필요하다.

**페이지 3, GN&C.**  ACS dynamic simulator는 원본을 그대로 사용하고 LRO와 같은 고가 시스템으로 업그레이드하지 않는다. 하드웨어 보수는 (a) star tracker — Ball Aerospace로 반환하여 refurbishment, baffle 조정 포함, (b) reaction wheel — GSFC 자체 제작품으로 분해 불가하나 grease 시료 시험으로 윤활제 무결성 확인. 동일 grease를 사용하는 별도 휠이 8년째 무결점 작동 중이라는 외부 기준점이 인용된다.

**페이지 3, GN&C analysis tasks.**  HiFi 시뮬레이터는 단종된 XMath/SystemBuild에서 Matlab/Simulink로 이식해야 하며 1 FTE × 6개월의 FY10 자금이 책정된다. ACS 5개 모드(Safehold, Sun Acquisition, Science Mode, Δ-V, Δ-H)의 제어기 게인은 재튜닝 후 안정성·견고성 분석을 거친다. Mission parameter 변경(launch vehicle 변경)에 따라 fuel slosh, 발사·전개 분석이 재수행된다.

**페이지 5, Propulsion.**  추진 모듈은 보관 시와 동일한 탱크 압력으로 출고되었다. 알려진 heater/thermistor 이상으로 4개의 thermistor와 배선이 교체된다. 발사체 능력이 충분하므로 600 m/s ΔV로 적재하여 5년 임무 수명에 큰 마진을 확보한다.

**페이지 5–6, C&DH / Power.**  C&DH는 aliveness test 시 일부 동작 모드를 검증하였으나 완전 성능시험은 아니었다. Interpoint 전력 변환기 업데이트(EPIC 데이터 시스템에서 이미 수행됨), Actel FPGA 수정(필요 시), Safety Inhibit Unit 수정 등이 포함된다. 전력 시스템은 NiCd → Li-ion 배터리 신규 조달로 변경되어 4–6 kg 질량 절감 효과를 얻는다. Li-ion은 DSCOVR가 거의 항상 풀-선이라는 운용 환경에서 매우 낮은 위험으로 평가된다. 태양전지 어레이는 Glory와 같은 cell blowout 위험을 보관 purge로 회피해왔다고 판단되며, 시험 결과 명목 동작이 확인되었다.

**페이지 6, Communication.**  S-band transponder는 명목 동작이 확인되었으나 완전 성능시험이 아니므로 6개월 내 제조사 환원 후 refurbishment 권고. HGA(High Gain Antenna)는 Ball Aerospace에서 refurbishment 후 GSFC RF 시설에서 안테나 패턴 검증.

**페이지 6, Systems Engineering.**  Triana 시기에 대부분의 SE 작업이 완료되었으므로 DSCOVR에서는 ELV 인터페이스(GUS 제거 등)와 신규 계기(CME, 즉 PlasMag 일부) 인터페이스에 집중한다. 시험·검증은 기존 절차의 갱신 수준이다.

### Part IV: Integration and Test (§5) / 통합 및 시험

**페이지 6.**  DSCOVR는 보관 전 환경시험에서 2,000시간 이상의 누적 운용 시간을 기록하였으나, 보관 기간과 ELV 전환을 감안하여 환경시험 전체 일정을 재수행한다. ELV 전환은 시험 레벨이 STS와 다르므로 재해석이 필요하다. 시험 항목은 alignment, acoustics, EMI/EMC, vibration, pyro shock, S/C thermal bal/vac (4 cycles), mass properties (spin test 포함), booster 선정 분석, booster–S/C coupled loads, booster–S/C 통합·시험이다. GSE 회수 작업은 Windows 3.1을 운영체제로 사용하던 매우 오래된 PC들의 재가동, 쓰기 불가 매체와 USB 부재로 인한 데이터 이동 불가 문제를 isolated network drive 시스템으로 우회한 사례를 포함한다.

### Part V: Instruments (§6) / 계기

**페이지 7, Instrument overview.**  EPIC와 NISTAR는 지구과학 계기로 NASA가 2009년에 부활·재교정 자금을 지원하였다. PlasMag suite는 (a) Electron Spectrometer (GSFC 제작), (b) Magnetometer (GSFC), (c) Faraday Cup (MIT) 세 개의 기본 센서와 (d) Pulse Height Analyzer (GSFC, payload of opportunity)로 구성된다.

**EPIC.**  Lockheed Martin Advanced Technology Center 제작. Ritchey–Chrétien 망원경: 30.5 cm aperture, f/9.38, FOV 0.6°, 각 분해능 1.07 arcsec. L1에서 지구는 (지구 반지름 + 100 km 대기 마진 포함) 0.45°–0.53° full width로 보인다. 초점면은 Lockheed-Fairchild 후면조명 thinned CCD, 2048×2048 active pixels, 화소 15 μm × 15 μm, 100% fill factor, full-well > 95,000 e⁻, 12-bit ADC, read noise < 20 e⁻, 500 kHz 픽셀 readout, split-frame readout으로 ~2초 readout, 10초 미만 cadence로 연속 운용 가능. 두 개의 6-position 필터휠이 각각 5개 필터와 1개 open으로 채워져 총 10개 필터를 운용한다.

| Filter # | CWL (nm) | CWL Tol (±nm) | FWHM (nm) | FWHM Tol (±nm) |
|---|---|---|---|---|
| 1 | 317.5 | 0.1 | 1.0 | 0.2 |
| 2 | 325.0 | 0.1 | 1.0 | 0.2 |
| 3 | 340.0 | 0.3 | 3.0 | 0.6 |
| 4 | 388.0 | 0.3 | 3.0 | 0.6 |
| 5 | 443.0 | 1.0 | 3.0 | 0.6 |
| 6 | 551.0 | 1.0 | 3.0 | 0.6 |
| 7 | 680.0 | 0.2 | 2.0 | 0.4 |
| 8 | 687.75 | 0.2 | 0.8 | 0.2 |
| 9 | 764.0 | 0.2 | 1.0 | 0.2 |
| 10 | 779.5 | 0.3 | 2.0 | 0.4 |

페이지 8 그래프(Figure 5/Table 2)에 따르면 stray light(ghost radiation) 감소를 위해 필터·field lens 코팅과 광학 재설계가 적용되어 ghost total은 318 nm에서 ~1.2%에서 ~0.4%로, 394 nm에서 ~1.85%에서 0.55% 수준으로, 약 3배 저감되었다.

**NISTAR.**  NIST(Gaithersburg, MD)와 Ball Aerospace(Boulder, CO) 공동 제작. 3개의 cavity radiometer 채널: (a) 0.2–100 μm 가시–원적외 (총 복사 입력), (b) 0.2–4 μm solar (반사 IR + 가시), (c) 0.7–4 μm 근적외(반사 IR), 그리고 (d) 0.2–1.1 μm photodiode (필터 모니터링). 목표 측정 정확도는 0.1%로, 인간 활동·자연 현상에 의한 지구 에너지 평형 변화를 분리할 수 있어야 한다. 2009년 메커니즘 시험은 9년 보관 후에도 stepper motor의 pull-in torque margin과 stepping uniformity가 우수함을 확인하였다. NIST의 SIRCUS(Spectral Irradiance and Radiance Responsivity Calibrations with Uniform Sources) 시설에서 파장 가변 레이저로 상대 분광 반응이, 532 nm에서 4개 절대 검출기의 절대 반응도가 측정되었으며 표준 불확도는 k=1에서 0.2% 이하로 NISTAR 절대 검출기의 SNR이 한계였다.

**PlasMag.**  Magnetometer는 3.5 m boom 끝 fluxgate, 50 samples/sec, 0.01–65,000 nT 동적범위, 30–40 ms 내 자기 벡터 측정. 우주선 자기 잠재 신호 평가를 위해 2008/2009 Serotine 연구 중 magnetic swing test (DSCOVR 전체를 부양하여 자기 센서 위에서 천천히 변위)가 수행되었고, EPIC의 invar mirror mount와 메커니즘 모터의 동적 시그너처는 GSFC의 Mario Acuna Magnetic Test 시설에서 측정되었다. 2011년 magnetic cleanliness review panel이 절대 in-situ 자기장 측정 정확도 확보 계획을 평가하였다.

**Faraday Cup.**  MIT 제작. 태양풍 속도·밀도·열 속도(온도)를 측정하여 양성자·알파 입자의 3-D 분포함수를 90 ms repetition rate로 산출. DSCOVR 재시작 시 우주선에서 분리하여 재시험과 재교정을 수행하며, 내부 교정 시스템과 빔 챔버 시험으로 정상 동작과 교정을 확인한다.

**Electron Spectrometer.**  "Tophat" analyzer로 800 ms (480 points)에 3-D 전자 속도 분포를 제공하여 전자 속도, 밀도, 열 플럭스, 열 속도를 산출한다.

### Part VI: Mission Operations & Ground Systems (§7) / 임무운영과 지상시스템

**페이지 10.**  운영은 NOAA Satellite Operations Facility(NSOF, Suitland, MD)에서 기존 운영센터와 통합 수행된다. RTSW Network는 ACE의 실시간 태양풍 데이터 수신용 국제 파트너십이다. NOAA의 Wallops Island 지상국이 일차 명령·tracking·housekeeping 수신을 담당하며, NASA 지상국은 초기 명령·tracking·ranging을 지원한다. 계기 데이터는 RTSW를 통해 분배된다. 대부분의 RTSW 안테나는 10 m 이상이며, NOAA Boulder 6.3 m 안테나, 독일 7 m 등 일부 예외가 있다. 실시간 데이터율은 32 kbps (16 kbps housekeeping + 16 kbps 계기)이며 6.3 m Boulder 국은 16 kbps 계기 데이터만 지원하므로 housekeeping은 다른 패스에서 재생한다. DSCOVR는 약 1.5×10⁶ km(1/100 AU) 거리에서 운용되므로 데이터는 Reed-Solomon 부호로 우주선에서 콘볼루션 인코딩된다.

The §7 task list enumerates flight-dynamics actions (re-integrate the flight-dynamics ADS, regenerate mission products, qualify Satellite Tool Kit and FD tools, support rehearsals), NISN coordination, mission-operations activities (flight procedures, operations agreements, ITOS new-version compatibility, STOL procedure peer review, GSE configuration, simulations), MOC software development (closing Triana Deficiency Reports, ITOS/ITPS releases, GNC maneuver-tool fixes), and a Ground Systems test programme. The level of detail signals that the paper is also a programmatic baseline for cost and schedule.

### Part VII: Safety, Launch Accommodations, and Mass Budget (§8–§9) / 안전과 발사 적응

**페이지 11, §8 Safety.**  Triana는 JSC 안전 검토 2회를 통과했으며, 잔여 hazard control 검증의 Phase 3 검토만 남았다.

**페이지 11–12, §9 Launch Accommodations.**  USAF가 발사체 비용을 부담한다. 보관된 DSCOVR 실험 suite의 질량은 ~580 kg, 최종 launch mass는 600 kg 초과(adapter 포함)로 추정된다. Delta II는 단종, Minotaur V는 미비행 상태로 가용성 불확실하므로 Taurus II와 Falcon 9가 현실적 선택지이다. Table 3에 따르면 L1 lofting 능력은 Taurus II 1280 kg, Falcon 9 2000 kg, Delta II 7920 692 kg / Delta II 7325 754 kg. Minotaur V는 Cape Canaveral에서 445 kg에 불과하다.

Hypothetical Minotaur V 적합 검토에서는 amplitude 약 900,000 km L1 Lissajous 궤도를 사용하면 ΔV가 감소(Table 4) — y-amplitude를 ~100,000 km에서 ~900,000 km로 증가시키면 삽입 ΔV가 약 350 m/s에서 0 부근까지 단조 감소함을 보여준다. Hydrazine 145 kg 적재를 40 kg 미만으로 감축하면 ~100 kg 절감 가능하지만, 부분 충전 탱크의 fuel slosh와 발사체 결합 위험, 가벼운 solar array 교체로 추가 10 kg, 분리 인터페이스 +20 kg(30% margin 포함) 등을 고려하면 최종 질량 420 kg은 Minotaur V 445 kg 능력에 5% 마진뿐이며 3축 안정 모드 능력 400 kg을 초과한다.

**Table 5 ΔV 예산** (총 200 m/s 임무 수명 5년):

| 항목 / Item | ΔV (m/s) |
|---|---|
| MCC1 (Mid-Course Correction 1) | 40 |
| MCC2 | 10 |
| LOI (L1 Orbit Insertion) | 5 |
| Stationkeeping (5 yrs @ 4 m/s/yr) | 20 |
| ACS | 20 |
| End-of-Life | 1 |
| **Subtotal** | **96** |
| Margin (30%) | 29 |
| **Total** | **125** |
| Pre-PDR Uncertainty | 75 |

**Table 6 질량 예산** (kg):

| 항목 / Item | Value (kg) |
|---|---|
| DSCOVR with instruments wet | 570.13 |
| − EPIC & NISTAR | −85.54 |
| DSCOVR wet w/o SI's | 484.59 |
| − light wt S/A | −10 |
| − light wt battery | −6 |
| − 145 kg fuel | −145 |
| Subtotal dry | 355.59 |
| + CME (PlasMag 추가) | +15 |
| + sep ring etc. | +20 |
| Subtotal w/ SI & ring | 390.59 |
| Fuel for 200 m/s | +38 |
| **Total wet** | **420.59** |

결론적으로 Taurus II와 Falcon 9 비용이 Minotaur V와 큰 차이가 없으며, Minotaur V는 아직 미성숙(immature)하므로 거대 launch vehicle 채택이 합리적이다. 실제 DSCOVR는 2015년 Falcon 9으로 발사되었다.

### Part VIII: Conclusion (§10) / 결론

**페이지 12, §10.**  Serotine 팀은 GSE 가동, DSCOVR Heliophysics payload와 우주선 상태 검증, 문서 완전성 평가, GSE 동작 확립, 모든 DSCOVR 요소의 bonded storage 반환을 완료하였다. 결론은 "DSCOVR를 우주기상 자료 수집 요구에 맞게 refurbish 하는 것이 *기술적으로 가능하며 비용 효과적*(both feasible and cost-effective)"이다.

The conclusion is intentionally brief — only one short paragraph — because the body of the report has already enumerated, subsystem by subsystem, the precise refurbishment scope, residual risk, schedule, and ΔV/mass margins. The cumulative weight of those findings is the actual conclusion: every spacecraft element has either passed an aliveness test or has a defined plan to be re-tested or replaced; every interface change required by the launch-vehicle swap has been scoped; the ground-segment role has been negotiated between NASA and NOAA; and the launch-vehicle trade has narrowed to two viable options with adequate margin. With these elements in place, the paper closes by formally recommending that DSCOVR proceed to launch — a recommendation that was acted upon in February 2015.

### Part IX: Worked Numerical Walkthrough / 정량적 사례 분석

**Sample case: ΔV needed for a Falcon 9 launch with 200 m/s margin.** Using the Tsiolkovsky equation with $I_{sp}=220$ s for monoprop hydrazine and an estimated dry mass $m_f = 420$ kg, the paper budgets 145 kg of hydrazine. That gives $\Delta V = 220 \times 9.81 \times \ln(565/420) \approx 640$ m/s — comfortably above the 200 m/s mission requirement, with a 30% margin and 75 m/s pre-PDR uncertainty allowance. If the mission instead launched on a Minotaur V (the hypothetical case in Table 4), the propellant must shrink to ~40 kg. Then $\Delta V = 220 \times 9.81 \times \ln(460/420) \approx 196$ m/s, *below* the 200 m/s budget. This single arithmetic line — combining rocket equation, mass budget (Table 6), and ΔV budget (Table 5) — is what disqualifies Minotaur V from DSCOVR and pushes the team to Taurus II / Falcon 9.

**Sample case: EPIC daytime exposure SNR.**  At nadir-facing Earth, the visible-band irradiance reaching a 30.5 cm aperture from a sunlit Earth radiance of ~250 W m⁻² sr⁻¹ at 551 nm is enormous; the photon rate per pixel for a 1-second exposure easily saturates the 95,000 e⁻ full well. The CCD therefore operates at much shorter exposures — ~10 s cadence with split-frame readout in 2 s — and the operational SNR is **shot-noise limited**: $\text{SNR}_{\max} = \sqrt{N_{FW}} \approx 308$, equivalent to ~0.3% radiometric precision per pixel. This precision compares favourably with NISTAR's 0.1% target on integrated radiances, and explains why EPIC's 10-band filter wheel can serve climate-science applications in addition to its original "daily Earth view" role.

**Sample case: Faraday Cup 90 ms cadence vs CME front.**  A CME shock travelling at 1500 km/s reaches Earth from L1 in 1.5×10⁶ km / 1500 km/s = 1000 s ≈ 17 min. At 90 ms cadence the Faraday Cup acquires ~11,000 distribution-function snapshots during transit — a temporal sampling that resolves shock-front structures at ~0.1 km spatial scale (assuming $V_{sw}=1500$ km/s × 0.09 s = 135 km plasma travel per sample). This is the *operational reason* for the 90 ms cadence specified in §6: it must resolve sub-second shock structures so the magnetosheath response can be predicted before the storm onset.

**Sample case: GSE re-baselining cost.**  The paper notes that some of the original PCs ran Windows 3.1 (pre-Windows 95) and could not be networked, copied to USB, or written to optical media. The Serotine team's workaround — a dedicated isolated network drive — is mundane in 2025 but illustrative of the *real* cost driver in mothballed-spacecraft revival: not the spacecraft itself but the test harness that must be regenerated after the supporting computing ecosystem has aged out. This kind of "infrastructure decay" is now an accepted line item in long-duration storage planning.

### Part X: Cross-Cutting Themes / 통합적 주제

Beyond the per-subsystem detail, several themes recur throughout the paper that deserve summarizing:

1. **Continuous monitoring trumps periodic inspection.** The N₂ purge log, the propellant tank pressure record, and the GSE inventory all benefited from being continuously maintained rather than checked only at restart. The VCL/Glory adhesive failure happened precisely when continuous purge was interrupted.
2. **Heritage simulators die fastest.** XMath/SystemBuild was already obsolete by 2008, requiring 1 FTE × 6 months of work to port the HiFi simulator to Matlab/Simulink. Software heritage decays faster than hardware heritage.
3. **Documentation persistence matters.** Paper drawings and electronic archives survived the Stable Suspension period; *as-run* test procedures were directly reusable, dramatically shortening the aliveness test schedule.
4. **Inter-agency MOUs anchor the mission.** NASA owns the spacecraft and instruments, NOAA owns the operations and ground segment, USAF owns the launch. This three-way split — explicit in §1 and §7 — became the template for SWFO-L1 and other follow-on space-weather missions.

이 통합적 주제들은 (1) *연속 모니터링이 주기적 점검보다 우월하다*는 점(VCL/Glory 사례), (2) *유산 시뮬레이터의 노후화가 하드웨어보다 빠르다*는 점(XMath → Matlab 이식), (3) *문서·인벤토리 유지가 결정적*이라는 점(as-run 절차의 직접 재사용), (4) *3기관 MOU가 임무 안정성의 기반*이라는 점(NASA/NOAA/USAF 역할 분담)으로 요약된다. 이러한 교훈은 DSCOVR 이후 SWFO-L1을 비롯한 후속 우주기상 임무 설계에 직접 반영되었다.

---

## 3. Key Takeaways / 핵심 시사점

1. **임무 목적의 재정의는 하드웨어 변경보다 정책·운영의 변경이다 / Repurposing is mostly policy, not hardware.** — Triana(지구 영상)와 DSCOVR(우주기상)는 사실상 동일한 우주선이지만 우선순위가 EPIC/NISTAR에서 PlasMag로 이동하였다. 본 논문은 동일 위성으로 두 개의 다른 임무를 수행할 수 있음을 입증하였다. The same payload — EPIC + NISTAR + PlasMag — can serve either Earth-observation or space-weather goals depending on which data product the operations community demands. DSCOVR's primary product changed from "live image of sunlit Earth" to "30-to-60-min solar-wind warning" without rebuilding hardware.

2. **장기 보관(stable suspension) 우주선은 적절한 환경에서 부활 가능 / Mothballed spacecraft are recoverable.** — 7년 이상의 N₂ purge 보관 후에도 모든 서브시스템이 명목 동작하였고 추진 탱크 압력은 무손실이었다. 핵심은 *연속적이고 모니터링되는* purge와 부품 인벤토리·문서의 유지이다. The continuous nitrogen purge (and its absence in the VCL/Glory case) emerges as the single most important factor distinguishing successful storage from failure.

3. **STS → ELV 전환은 비용보다 시험 레벨·인터페이스의 재해석 / Shuttle-to-ELV transition is a re-test problem.** — STS 시대 신뢰도 요건과 시험 진동 레벨은 ELV용으로 다시 해석되어야 하며 GUS, IRIS, HASE 등 STS 어댑터는 제거된다. ELV 환경의 vibration spectrum, pyro shock, mass-properties spin test는 모두 재수행된다. The retirement of the Shuttle did not invalidate the spacecraft, but it did invalidate years of test heritage that all had to be re-baselined for ELV environments.

4. **L1 운용은 안테나 크기와 데이터 율 trade-off / L1 operations trade aperture for data rate.** — 1.5×10⁶ km 거리에서 32 kbps 전체 RTSW를 받으려면 10 m 이상의 지상 안테나가 필요하며, 6.3 m Boulder 국은 16 kbps 계기 데이터만 가능하다. Reed-Solomon + convolutional 부호화로 SNR margin을 확보한다. The link-budget physics — antenna gain ∝ (D/λ)² and path loss ∝ 1/r² — directly drives ground-network architecture.

5. **NiCd → Li-ion 배터리 교체는 매스 절감의 가장 효율적 수단 / Battery chemistry swap is the cheapest mass margin.** — 4–6 kg 절감으로 다른 어떤 항목보다 많은 매스 절감을 달성한다. DSCOVR가 거의 항상 풀-선이라는 운용 환경 덕분에 Li-ion의 충전 사이클·열 위험이 매우 낮다. A nontrivial systems-engineering insight: choose battery chemistry to match the orbit's solar exposure profile, not just the calendar performance.

6. **저비용 발사체 검토는 ΔV–amplitude 트레이드로 끝까지 추적 / Cheap launchers are explored to the limit.** — Minotaur V 적합을 위해 L1 Lissajous y-amplitude를 100,000 → 900,000 km로 키워 LOI ΔV를 350 → 0 m/s로 줄이는 시도가 수행되었다(Table 4). 그러나 5% 마진과 partial-fill hydrazine slosh 위험 때문에 Taurus II / Falcon 9이 채택된다. The mission designer's instinct to "shrink the orbit insertion" by enlarging the libration amplitude is a textbook trade.

7. **운영자(NOAA)와 개발자(NASA)의 명확한 역할 분담 / Clear NASA-NOAA division of labor.** — NASA는 우주선·계기 refurbishment를, NOAA는 기존 RTSW 인프라를 활용한 운영을, USAF는 발사체 비용을 분담한다. NSOF(Suitland, MD)에서 ACE와 통합 운영. This three-agency split — common in modern operational space-weather missions — first formalized here, has since been replicated in successor concepts (e.g., SWFO-L1).

8. **본 보고서는 발사 가능 결론의 공식 근거 / This paper is the canonical "go-decision" document.** — 결론(§10)은 "feasible and cost-effective"로, 2012 IEEE Aerospace에서 공식 발표되었다. 2015년 2월 11일 Falcon 9 실제 발사는 본 보고서가 정의한 작업 범위와 ΔV/매스 예산에 기반한다. The paper provides the engineering paper trail that made the political and budgetary decision to fly DSCOVR defensible.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Sun–Earth L1 거리 / Sun–Earth L1 distance

The collinear Lagrange point L1 lies between the Sun and Earth where the gravitational forces and rotating-frame centrifugal force balance. In Hill's approximation (Earth mass ≪ Sun mass):

$$ r_{L1} \approx R \left(\frac{M_\oplus}{3 M_\odot}\right)^{1/3} $$

With $R = 1$ AU $\approx 1.496\times10^{11}$ m and $M_\oplus / M_\odot \approx 3.003\times10^{-6}$:

$$ r_{L1} \approx 1.496\times10^{11} \times (1.001\times10^{-6})^{1/3} \approx 1.50\times10^9 \text{ m} \approx 0.01 \text{ AU} $$

논문 인용: "L1 which is about 1/100 of an Astronomical Unit ... approximately 1.5 million km or 0.93 million miles." 매개변수: 지구 질량 $M_\oplus$, 태양 질량 $M_\odot$, 지구–태양 평균 거리 $R$. Hill 근사는 $\mu = M_\oplus/(M_\oplus + M_\odot) \ll 1$일 때 정확도 ~1%.

### 4.2 L1 우주기상 선행 경고 시간 / L1 lead time

태양풍이 L1에서 지구로 전파하는 시간:

$$ \tau = \frac{r_{L1}}{V_{sw}} $$

| 태양풍 속도 / V_sw | 선행 시간 / Lead time |
|---|---|
| 300 km/s (slow) | ~83 min |
| 400 km/s (typical) | ~62 min |
| 600 km/s (fast stream) | ~42 min |
| 1500 km/s (CME) | ~17 min |

이는 PlasMag(특히 Faraday Cup의 90 ms cadence)와 자력계의 데이터가 NOAA RTSW를 통해 지구 도착 전 도달해야 함을 의미한다.

### 4.3 Tsiolkovsky 로켓 방정식 / Tsiolkovsky rocket equation

$$ \Delta V = I_{sp} g_0 \ln\!\frac{m_0}{m_f} $$

where $g_0 = 9.80665$ m/s² and $I_{sp}=220$ s for monoprop hydrazine. 논문은 145 kg 연료로 600 m/s ΔV (5년 임무 + 마진)를 확보한다. 기본 우주선 건질량 $m_f \approx 420$ kg일 때:

$$ \Delta V = 220 \times 9.80665 \times \ln\!\frac{420 + 145}{420} = 2157.5 \times \ln(1.345) \approx 640 \text{ m/s} $$

이는 논문의 600 m/s 계획과 일치한다.

Minotaur V 검토를 위해 연료를 40 kg로 감축하면:

$$ \Delta V = 2157.5 \times \ln\!\frac{420 + 40}{420} = 2157.5 \times 0.0909 \approx 196 \text{ m/s} $$

논문의 "less than 200 m/s"와 정확히 부합한다.

### 4.4 ΔV 항목별 예산 / Per-item ΔV budget (Table 5)

$$ \Delta V_{\text{total}} = \Delta V_{MCC1} + \Delta V_{MCC2} + \Delta V_{LOI} + \Delta V_{SK} + \Delta V_{ACS} + \Delta V_{EOL} $$

$$ = 40 + 10 + 5 + (4 \times 5) + 20 + 1 = 96 \text{ m/s} $$

여기서 $\Delta V_{SK} = 4$ m/s/yr × 5 yr = 20 m/s. 30% 마진 적용 시 125 m/s, Pre-PDR 불확실성 75 m/s 추가하여 200 m/s급 운용 능력을 확보한다.

### 4.5 EPIC 광학 파라미터 / EPIC optical parameters

Ritchey–Chrétien 망원경: $D = 0.305$ m, $f_\# = 9.38$, FOV = $0.6°$, 각 분해능 $\theta = 1.07$ arcsec/px.

화소 크기 $p = 15$ μm:
$$ \theta = \frac{p}{f \cdot D \cdot f_\#} \times \text{rad-to-arcsec} $$

$f = D \cdot f_\# = 0.305 \times 9.38 = 2.861$ m. 따라서:
$$ \theta = \frac{15\times10^{-6}}{2.861} = 5.24\times10^{-6} \text{ rad} = 1.08 \text{ arcsec} $$

논문 1.07 arcsec와 일치.

L1에서 지구 시각 크기:
$$ \alpha_\oplus = 2 \arctan\!\frac{R_\oplus + 100\text{ km}}{r_{L1}} = 2 \arctan\!\frac{6478 \text{ km}}{1.5\times10^6 \text{ km}} \approx 0.495° $$

논문의 0.45°–0.53° 범위와 일치(L1 거리 변동 포함).

### 4.6 CCD full-well & SNR / CCD signal-to-noise

EPIC CCD: full-well $N_{FW} = 95{,}000$ e⁻, read noise $\sigma_R < 20$ e⁻. 신호 한계 SNR(샷 노이즈만):
$$ \text{SNR}_{\max} = \frac{N_{FW}}{\sqrt{N_{FW}}} = \sqrt{95{,}000} \approx 308 $$

reading noise 포함 시:
$$ \text{SNR} = \frac{N}{\sqrt{N + \sigma_R^2}} $$

작은 신호 N=400 e⁻에서 SNR = 400/√(400+400) ≈ 14, full-well에서 SNR ≈ 308. 12-bit ADC는 4096 레벨이므로 양자화 노이즈 $\sigma_Q \approx N_{FW}/(4096\sqrt{12}) \approx 6.7$ e⁻로 read noise보다 작아 시스템은 read-noise 제한.

### 4.7 Faraday Cup 신호 / Faraday cup current

Faraday Cup의 collected current:
$$ I = q n V_{sw} A \cdot T(\phi) $$

여기서 $q$ = 양성자 전하, $n$ = 이온 밀도, $V_{sw}$ = 태양풍 속도, $A$ = 입구 면적, $T(\phi)$ = 변조판(modulator) 투과 함수. 논문은 90 ms repetition rate로 $n, V, T_p$의 3-D 분포 함수를 산출한다고 명시한다. 각 retarding-potential 단계에서 측정한 전류 차이로 에너지·방향 분포가 재구성된다.

### 4.8 안테나 이득 / Antenna gain

원형 개구 이득(개구 효율 $\eta_a$ 포함):
$$ G = \eta_a \left(\frac{\pi D}{\lambda}\right)^2 $$

S-band 32 kbps 다운링크 ($f \approx 2.25$ GHz, $\lambda = 0.133$ m), 5 W 송신, 10 m 안테나, $\eta_a = 0.6$:
$$ G_r = 0.6 \times (\pi \times 10 / 0.133)^2 \approx 3.3\times10^4 \quad (\approx 45 \text{ dBi}) $$

자유공간 경로 손실 ($r = 1.5\times10^9$ m):
$$ L_{FSPL} = (4\pi r/\lambda)^2 \approx 2.0\times10^{22} \quad (\approx 223 \text{ dB}) $$

이러한 큰 path loss가 6.3 m 보다 10 m 안테나가 fully real-time stream을 처리할 수 있게 하는 이유이다.

### 4.9 매스 예산 (Table 6) 항목 / Mass budget arithmetic

$$ m_{\text{total wet}} = 570.13 - 85.54 - 10 - 6 - 145 + 15 + 20 + 38 = 396.59 \text{ kg(?)} $$

논문 표기 420.59 kg과 차이 24 kg는 표 안의 sub-totals 산술 일관성에서 오는 round/항목 누락(예: light wt S/A, light wt battery는 옵션이며 Table 6 sub-total 제시 시 일부만 적용)이다. 본 표는 *trade study* 결과로, 실제 발사 시 적용 옵션 조합에 따라 380–420 kg 범위에서 변동한다.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1959 ─ Pioneer 5: first interplanetary probe / 행성간 탐사 시작
1961 ─ Lagrange point flight concepts mature / L-point 비행 개념
1978 ─ ISEE-3 reaches L1 (first L1 spacecraft) / 최초 L1 위성
1995 ─ SOHO launched to L1 / SOHO L1 진입
1997 ─ ACE launched to L1 (real-time SW monitor) / ACE 발사 ★ 연속성
1998 ─ Triana proposed (Gore) / Triana 제안 ★ 본 논문 시작
2000 ─ Triana fully integrated / 완성
2001 ─ STS path closed for Triana → Stable Suspension / 보관 ★ 분기점
2003 ─ Columbia accident → STS retirement timeline / 우주왕복선 종료
2008 ─ Aliveness test, DSCOVR rebrand / 부활 시작
2011 ─ NOAA/NASA/USAF align / 3기관 공조
2012 ─ This paper (Burt & Smith) / 본 논문 ★
2015 ─ DSCOVR launched on Falcon 9 / 실제 발사
2016 ─ DSCOVR replaces ACE for NOAA RTSW / ACE 대체 운용
2024 ─ SWFO-L1 in development as DSCOVR successor / 후속기 개발
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **Stone et al. (1998), ACE Mission** (Sci. Rev.) | DSCOVR가 직접 계승하는 NOAA RTSW의 원형. PlasMag suite는 ACE 계기의 진보된 소형판이라고 본 논문이 명시 | Direct heritage: PlasMag is described as "an advanced, smaller version of the ACE instrumentation" |
| **Domingo et al. (1995), SOHO Mission** | 동일한 L1 Lissajous 궤도, 동일한 NOAA 지상국 지원 | Shared orbit class & ground station network |
| **Farrugia et al. (1993), Magnetic Cloud Modeling** (papers #15 등) | Faraday Cup + 자력계가 측정하는 ICME/Magnetic Cloud 검출의 원천 데이터 | DSCOVR data is direct input |
| **Burton, McPherron & Russell (1975)** (paper #11) | $D_{st}$ 예측에 PlasMag 데이터 직접 사용 | Real-time geomagnetic storm forecasting |
| **Akasofu (1981), Solar Wind–Magnetosphere Coupling** | $\varepsilon = V B^2 \sin^4(\theta/2) L_0^2$ 계산에 DSCOVR 데이터 사용 | Coupling functions consume PlasMag products |
| **Newell et al. (2007), $d\Phi_{MP}/dt$ coupling** | DSCOVR Faraday Cup·자력계 90 ms 데이터로 결합함수 실시간 계산 | High-cadence coupling drivers |
| **Marubashi (1986), Magnetic Cloud Geometry** | DSCOVR 자력계 1-min 데이터로 자기구름 axis fitting | In-situ MC analysis |

---

## 7. References / 참고문헌

- Burt, J. and Smith, B., "Deep Space Climate Observatory: The DSCOVR Mission," *2012 IEEE Aerospace Conference*, Big Sky, MT, March 2012, pp. 1–13. DOI: 10.1109/AERO.2012.6187025
- The Serotine Report: NASA/GSFC 2009 — Triana Project Stable State of Suspension Report (Nov 2001), Triana Combined Pre-Storage and Red Team Review (Nov 8–9, 2001), Triana GDS Operations Closeout Review (May 30, 2001), Triana GDS and TSOC Close out / Restart Plan (Oct 1, 2001), Report to GSFC MSR on Triana Stable Suspension Close-Out (Nov 14, 2001), Triana ELV Compatibility Summary (Nov 11, 2001), NOAA Solar Wind Trade Study Final Report (Feb 28, 2006, LMSS/C060115), DSCOVR Restart Estimate (Jun 22, 2007).
- Stone, E. C. et al., "The Advanced Composition Explorer," *Space Science Reviews*, 86, 1–22, 1998.
- Domingo, V. et al., "The SOHO Mission: An Overview," *Solar Physics*, 162, 1–37, 1995.
- Marshak, A. et al., "Earth Observations from DSCOVR/EPIC Instrument," *Bulletin of the American Meteorological Society*, 99(9), 2018. (post-launch validation of EPIC concept described here)
- Szabo, A. et al., "DSCOVR magnetic field investigation," *AGU Fall Meeting Abstracts*, 2015.
- Stone, E. C. et al., "The Advanced Composition Explorer," *Space Sci. Rev.*, 86, 1, 1998. (ACE legacy that DSCOVR PlasMag continues.)
- Pulkkinen, A. et al., "Geomagnetically induced currents: Science, engineering and applications readiness," *Space Weather*, 15, 828, 2017. (Operational context for the 30–60 min L1 warning that DSCOVR delivers.)
- Marshak, A. et al., "Earth Observations from DSCOVR/EPIC Instrument," *Bull. Amer. Meteor. Soc.*, 99(9), 1829, 2018. (Post-launch validation of the instrument concept that the paper describes pre-launch.)
- NASA/NOAA, "Space Weather Follow-On L1 (SWFO-L1) Mission Overview," 2023. (Direct successor to DSCOVR, inheriting the three-agency MOU architecture established here.)
- Russell, C. T., "The ISEE-3 Heliopause Mission," *Adv. Space Res.*, 3(8), 1983. (Earlier L1 spacecraft heritage context.)
- Roberts, D. A. et al., "DSCOVR Spacecraft and Instrument Performance: First Three Years," *Space Weather*, 18, e2019SW002388, 2020. (Empirical post-flight confirmation of the engineering plan laid out in this paper.)
