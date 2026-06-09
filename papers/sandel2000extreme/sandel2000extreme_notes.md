---
title: "The Extreme Ultraviolet Imager Investigation for the IMAGE Mission"
authors: ["Sandel B. R.", "Broadfoot A. L.", "Curtis C. C.", "King R. A.", "Stone T. C.", "Hill R. H.", "Chen J.", "Siegmund O. H. W.", "Raffanti R.", "Allred D. D.", "Turley R. S.", "Gallagher D. L."]
year: 2000
journal: "Space Science Reviews"
doi: "10.1023/A:1005263510820"
topic: Space_Weather
tags: [IMAGE_mission, plasmasphere, He_plus, EUV_imaging, multilayer_mirror, MCP_detector, magnetospheric_imaging, 30.4_nm]
status: completed
date_started: 2026-04-25
date_completed: 2026-04-25
---

# 75. The Extreme Ultraviolet Imager Investigation for the IMAGE Mission / IMAGE 임무 극자외선 영상기 연구

---

## 1. Core Contribution / 핵심 기여

Sandel et al. (2000) describe the Extreme Ultraviolet Imager (EUV), one of seven instruments on NASA's IMAGE Mission, designed to obtain the first global "snapshots" of Earth's plasmasphere by imaging the resonantly scattered He+ 30.4 nm emission line. EUV consists of three identical fast (f/0.8) sensor heads, each with a 30°-diameter conical field of view, tilted by 27° relative to one another to produce an instantaneous 84° × 30° fan that becomes an 84° × 360° annulus through spacecraft spin. The instrument's headline parameters — 0.6° angular resolution (≈0.1 R_E in the equatorial plane from apogee), 10-min time resolution, and 1.9 count s⁻¹ Rayleigh⁻¹ sensitivity — derive from the multiplication of five well-characterized factors: 21.8 cm² annular aperture, 1.1×10⁻⁴ sr resolution-element solid angle, 22% multilayer reflectivity, 33% Al-filter transmission, and 14% bare-MCP quantum detection efficiency. The paper is a comprehensive end-to-end design and calibration report covering optics, novel U/Si multilayer mirrors that must reflect well at 30.4 nm but poorly at 58.4 nm, spherically curved triple-MCP wedge-and-strip detectors, on-board time-delayed-integration sky-mapping, and ground/in-flight calibration.

Sandel 외 (2000) 논문은 NASA IMAGE 임무의 일곱 기기 중 하나인 극자외선 영상기(EUV)의 설계와 보정을 종합적으로 기술한다. EUV는 He+ 이온이 태양 30.4 nm 광자를 공명 산란하여 빛나는 플라스마권을 외부에서 한 번에 영상화하기 위해 만들어졌다. 동일한 f/0.8 빠른 센서 헤드 3개가 각 30° 원뿔 시야를 가지며 서로 27°씩 기울여져 순간 84°×30° 부채꼴을, 위성 자전을 통해 84°×360° 환형 영역을 덮는다. 각해상도 0.6°(원지점 적도면에서 ≈0.1 R_E), 시간 분해능 10분, 감도 1.9 count s⁻¹ R⁻¹의 핵심 사양은 21.8 cm² 환형 입사창, 1.1×10⁻⁴ sr 해상 셀 입체각, 22% 다층 반사율, 33% Al 필터 투과율, 14% 곡면 MCP 양자 효율의 곱으로 결정된다. 논문은 광학·다층 반사경(30.4 nm는 잘 반사하고 58.4 nm는 < 0.2%로 억제), 곡면 트리플 MCP wedge-and-strip 검출기, 자전 동기 TDI 스카이맵 생성, 지상·궤도 보정 등 기기 전 과정을 다룬다.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction & Scientific Goals (§1–§2) / 도입과 과학 목표

The Earth's plasmasphere — the corotating, cold (~1 eV), dense (peak ~1000 cm⁻³) population trapped on closed dipole field lines — is the hidden third party in the magnetosphere–ionosphere–ring-current system. He+ constitutes ~20% of plasmaspheric ions and is, after H+ (which has no optical emission), the most abundant species. The He+ 30.4 nm line is the He II analog of hydrogen Lyα: ions in sunlight resonantly scatter solar 30.4 nm photons, and because the plasmasphere is optically thin at this line, the integrated brightness is directly proportional to the He+ column density along the line of sight. This makes inversion straightforward — no heroic radiative transfer modeling is needed.

지구 플라스마권은 차갑고(~1 eV) 조밀한(피크 ~1000 cm⁻³) 자전 동행 플라스마로, 자기권–전리권–링 커런트 결합에서 가장 늦게 영상화된 영역이다. He+는 플라스마권 이온의 약 20%를 차지하고, 광방출이 없는 H+를 제외하면 가장 풍부하다. He+ 30.4 nm 라인은 H Lyα에 대응하는 He II 라인으로, 태양빛 속의 He+ 이온이 30.4 nm 광자를 공명 산란하여 빛난다. 광학적으로 얇기 때문에 시선 방향 강도는 He+ 칼럼 밀도에 정비례한다.

Open scientific questions in 1999 (paraphrased from §2): (i) how do ring-current ion injections, wave-particle processes, and storm-time electric fields erode the plasmapause? (ii) what is the fate of eroded thermal plasma convected toward the dayside magnetopause? (iii) does the two-stage refilling paradigm (Thomsen et al. 1998) hold? EUV is intended to address these by providing repeated global He+ images cadenced at 10 minutes, complementing the IMAGE neutral-atom imagers (MENA, HENA) that observe the ring current and the Radio Plasma Imager (RPI) that probes thermal-plasma densities by sounding.

1999년 시점의 미해결 질문(요약): (i) 링 커런트 이온 주입과 파동–입자 상호작용, 폭풍기 전기장이 어떻게 플라스마포즈를 침식시키는가? (ii) 침식된 열적 플라스마는 낮 쪽 자기경계로 어떻게 운반되며 그 운명은 무엇인가? (iii) Thomsen 외 (1998)의 2단계 리필링 모델이 트로프 외에 자속관에서도 성립하는가? EUV는 10분 간격 전 지구 He+ 영상을 통해 이러한 질문들에 답하며, IMAGE의 MENA·HENA(중성 원자 영상기)와 RPI(전파 플라스마 영상기)와 데이터를 결합한다.

The four measurement requirements explicitly listed (§2):
1. accommodate maximum plasmaspheric brightness of 10 R, plus the much brighter localized ionospheric source;
2. measure 0.1–0.3 R features in 10-min integration time;
3. wide field of view encompassing the whole plasmasphere in one snapshot;
4. reject H Lyα (geocoronal/interplanetary) and He I 58.4 nm (ionospheric).

명시된 네 가지 측정 요건(§2):
1. 플라스마권 최대 밝기 10 R과 훨씬 밝은 전리권 국소 광원 동시 수용;
2. 10분 적분 시간 내 0.1–0.3 R 미세구조 측정;
3. 한 번의 스냅샷에서 전체 플라스마권을 담는 광시야;
4. H Lyα(지오코로나/행성간)와 He I 58.4 nm(전리권) 차단.

### Part II: Instrument Architecture (§3.0–§3.1) / 기기 구조와 센서 헤드

EUV is a 15.5 kg, 9.0 W package of size 49.7×23.3×49.5 cm housing three identical sensor heads serviced by common Controller electronics, three high-voltage power supplies, and shared mechanical structure (Table I). Each sensor head is its own f/0.8 prime-focus telescope: light enters an annular aperture (outer 8.2 cm, inner 6.0 cm — open area 21.8 cm²), passes a 150 nm aluminum filter on a nickel mesh, reflects off a 12.3 cm-diameter spherical mirror with R = 13.5 cm radius of curvature and U/Si multilayer coating, and converges onto a spherical 4 cm-diameter MCP detector with R = 7.0 cm focal-surface radius (Table II). The optical design follows the ALEXIS X-ray heritage (Bloch et al. 1990).

EUV는 질량 15.5 kg, 전력 9.0 W, 크기 49.7×23.3×49.5 cm 패키지로, 동일한 센서 헤드 3개와 공용 컨트롤러 전자회로, 고전압 전원 3채널, 공유 기계 구조로 구성된다(Table I). 각 센서 헤드는 자체 f/0.8 prime-focus 망원경이다. 빛은 환형 입사창(외경 8.2 cm, 내경 6.0 cm — 개방면적 21.8 cm²)을 통과해 니켈 메시에 지지된 150 nm 알루미늄 필터를 거치고, 직경 12.3 cm 곡률반경 13.5 cm의 구면 거울(U/Si 다층 코팅)에 반사되어 직경 4 cm, 곡률반경 7.0 cm의 곡면 MCP 검출기에 수렴한다. 광학 설계는 ALEXIS X-선 망원경(Bloch 외 1990) 유산을 잇는다.

Three sensor heads are tilted by 27° relative to one another so their 30°-FOV cones overlap by 3° at the seams; together they cover an instantaneous 84°×30° fan. As IMAGE spins at one revolution per 120 s, the fan sweeps an 84°×360° annulus across the sky — at apogee (~7 R_E) this annulus images the entire plasmasphere from outside.

세 센서 헤드는 서로 27° 기울여져 30° 시야 원뿔이 이음매에서 3°씩 중첩하도록 배치되어 순간 84°×30° 부채꼴을 형성한다. IMAGE가 120 s 주기로 자전하면 부채꼴이 84°×360° 환형 영역을 휩쓸며, 원지점(~7 R_E)에서 플라스마권 전체를 외부에서 한 번에 영상화한다.

Spot diagrams (Figure 5) show that ray-traced angular resolution is 0.6° on-axis, degrading mildly at the edge due to detector vignetting (which begins at 7° off-axis and is severe at 14°). Vignetting (Figure 6) reduces the effective throughput at 15° off-axis to ~53% of the on-axis value. Because the spin scan exposes top/bottom edges of the field for less time, the team overlaps adjacent heads' fields by 3° to compensate.

스팟 다이어그램(Figure 5)에서 광축 상 각해상도는 0.6°이며 시야 가장자리(7°에서 비네팅 시작, 14°에서 두드러짐)에서 약간 저하된다. 비네팅(Figure 6)에 의해 15° 가장자리 처리량은 광축 대비 약 53%로 감소한다. 자전 스캔이 시야 위·아래 가장자리를 짧게 노출하므로, 인접 헤드의 시야를 3°씩 겹쳐 보상한다.

The annular entrance aperture serves a dual purpose: it limits incidence-angle range on the curved multilayer mirror (rays span only 12°–18° from normal — Figure 7), within which the multilayer maintains satisfactory reflectivity. This is critical because multilayer-mirror response is strongly angle-dependent.

환형 입사창은 두 가지 목적을 동시에 수행한다. (i) 광량 입사창 역할; (ii) 다층막 거울에 도달하는 광선의 입사각을 12°–18°로 제한(Figure 7)해 다층막의 좁은 각도 수용 범위 내에서 동작시키는 것. 후자는 다층막 반사율이 입사각에 매우 민감하기 때문에 결정적이다.

### Part III: Multilayer Mirrors (§3.2) / 다층 반사경

This is the technical centerpiece. The mirror specification was challenging: > 20% reflectivity at 30.4 nm and < 0.2% at 58.4 nm, both at 14.5°±3.5° from surface normal. Because most candidate materials reflect 58.4 nm more strongly than 30.4 nm (e.g., bare Mo gives 24% at 58.4 nm), the top of the multilayer stack must function as an antireflection layer at 58.4 nm.

논문의 기술적 핵심이다. 다층막 사양은 14.5°±3.5°에서 30.4 nm 반사율 > 20%, 58.4 nm 반사율 < 0.2%로 설정되었는데, 대부분의 재료가 58.4 nm를 30.4 nm보다 잘 반사하므로(예: bare Mo의 58.4 nm 반사율 24%) 다층 적층의 상단이 58.4 nm에 대해 반사 방지층 역할을 동시에 해야 했다.

**Design approach**: BYU group (Lunt & Turley 1998–1999) used a Genetic Algorithm (GA) to optimize material selection and individual layer thicknesses with discrete variable handling. The GA identified Y₂O₃/Al as the best fully aperiodic design, but Y₂O₃ requires RF rather than DC magnetron sputtering and aperiodic stacks were hard to characterize via X-ray diffraction (XRD). The team therefore moved to a U/Si periodic stack with a UOₓ cap.

**설계 접근**: BYU 팀(Lunt & Turley 1998–1999)은 유전 알고리즘(GA)으로 재료와 층 두께를 동시 최적화했다. GA는 Y₂O₃/Al 비주기 설계를 최선으로 제시했으나, Y₂O₃는 RF 스퍼터링이 필요하고 비주기 적층은 X-선 회절(XRD)로 특성화하기 어려워, 결국 UOₓ 캡을 가진 U/Si 주기 다층막으로 결정되었다.

**Final design (top→bottom)**:
1. ~1.5 nm uranium oxidized in air to UOₓ (likely UO₃, swelling > 3×) — top high-index layer at 30.4 nm and primary contributor to the 58.4 nm anti-reflection effect; **first XUV multilayer ever to use this trick**.
2. Six and one-half periods of bilayers: 12.8±0.01 nm Si and 5.3±0.1 nm U.
3. A 10.6 nm uranium release layer (twice the normal U thickness) at the bottom, to enable later mirror release/recoating.

**최종 설계(상→하)**:
1. ~1.5 nm 우라늄을 공기 산화시킨 UOₓ 캡(UO₃로 산화 시 두께 3× 이상 팽윤) — 30.4 nm에서 고굴절률 상층이며 58.4 nm 반사 방지의 주역. **XUV 다층막에서 이 기법을 처음 사용**.
2. 6.5 주기의 12.8±0.01 nm Si / 5.3±0.1 nm U 이중층.
3. 바닥에 10.6 nm U 분리(release) 층(다른 U층의 2배 두께) — 추후 거울 재코팅을 위해 화학적으로 분리 가능하도록 함.

**Departures from ideal models**: layer-boundary roughness (~0.5 nm rms) was acceptable, but Si diffused 5–10 nm into U layers (deeper than expected), optical constants for sputtered U at 30 nm were uncertain (Fennimore et al. 1999, Squires 1999), and oxidation during/after growth changed densities and thicknesses. These factors made empirical iteration with XRD necessary.

**이상 모델과의 편차**: 층 경계 거칠기는 ~0.5 nm rms로 허용 가능했으나, Si가 U 층 내부로 5–10 nm까지 확산되었고(예상보다 깊음), 30 nm에서 스퍼터링 U의 광학상수가 불확실하며, 성장 중·후 산화로 밀도와 두께가 변했다. 이로 인해 XRD를 사용한 경험적 반복 최적화가 필수였다.

**Fabrication**: DC magnetron sputtering in argon at 2.8×10⁻³ torr (chamber base 3×10⁻⁶ torr). U gun at 80 W, Si gun at 78 W. Mirror spun on its axis above sputter targets, with masks ensuring < 1% thickness variation across the highly curved 12.3 cm mirror. Sputtering times: Si 360 s, U 70 s per layer. Stack characterized by XRD (Cu K-α 0.15406 nm), atomic force microscopy, Auger electron spectroscopy, and TEM. Particle emission from U was measured against ambient backgrounds — depleted U is naturally radioactive but its particle-induced MCP signals are comparable to intrinsic dark-count rates (Section 3.4.2.3).

**제작**: Ar 분위기(2.8×10⁻³ torr, 챔버 base 3×10⁻⁶ torr) DC 마그네트론 스퍼터링. U 건 80 W, Si 건 78 W. 거울이 자체 축으로 회전하면서 스퍼터 타깃 위를 지나가도록 회전대를 사용해 곡면 12.3 cm 거울 전체에서 < 1% 두께 균일성 달성. 층당 시간: Si 360 s, U 70 s. XRD(Cu K-α), AFM, Auger 분광, TEM으로 특성화. 우라늄(열화 우라늄)의 자연 방사 입자가 MCP에 미치는 영향은 본질적 dark count와 비슷한 수준임을 확인.

**Test results**: At the BYU test stand using a McPherson 629 He hollow-cathode source and Model 225 monochromator (line widths ~0.025 nm), and at the LPL EUV calibration facility, flight mirror 9 produced ~22% reflectivity at 30.4 nm and ~0.7% at 58.4 nm across a useful 6° angular range (Figures 11–12). The four production batches of 12 flight filters showed 33±2% transmission at 30.4 nm and 12–18% at 58.4 nm; H Lyα (121.6 nm) transmission was below the 10⁻⁴ measurement floor (model: ~10⁻⁷).

**시험 결과**: BYU(McPherson 629 He hollow-cathode + Model 225 분광기, 라인 폭 ~0.025 nm)와 LPL EUV 보정 시설에서 비행 거울 9번은 6° 각도 범위에서 30.4 nm 반사율 ~22%, 58.4 nm 반사율 ~0.7% 달성(Figures 11–12). 12개 비행 필터의 30.4 nm 투과율 33±2%, 58.4 nm 투과율 12–18%, H Lyα(121.6 nm) 투과율 < 10⁻⁴(모델 예측 ~10⁻⁷).

### Part IV: Detector System (§3.4) / 검출기 시스템

The detector is a triple stack of 4.6 cm-diameter, 80:1 length-to-diameter MCPs with 12 μm pores, thermally slumped to a 7-cm radius of curvature. Top plate has 0° pore bias; middle and bottom plates have 13° bias to break ion feedback paths. A wedge-and-strip readout anode (4.2 cm diameter, three-element Siegmund 1986b array, 0.1 cm period) sits ~1 cm below the MCP exit face under ~300 V accelerating bias. Photoevents — ~10⁷ electron clouds — divide their charge between Wedge (W), Strip (S) and Zigzag (Z) electrodes, decoded via Eqs. 1–3 to (X,Y) positions on the focal plane.

검출기는 직경 4.6 cm, 길이/직경 비 80:1, 기공 12 μm의 MCP 3장을 7 cm 곡률반경으로 열 slump한 트리플 스택이다. 상단 플레이트는 0° pore bias, 중간/하단은 13° bias로 이온 feedback 경로를 차단한다. 4.2 cm 직경 wedge-and-strip 양극(Siegmund 1986b의 3-element 배열, 0.1 cm 주기)이 MCP 출구면 아래 ~1 cm에 ~300 V 가속 전압으로 배치된다. 광 이벤트의 ~10⁷ 전자운이 W/S/Z 전극에 전하를 분배하며 식 (1)–(3)을 통해 초점면의 (X,Y) 좌표로 디코딩된다.

**Performance**: Pulse-height distribution (PHD) FWHM of fully illuminated curved MCPs is ~120% (vs ~60% for spot illumination, Figure 17), broader than flat MCPs (~60%) due to gain non-uniformity from imperfect spherical fit. Peak gain at 3600 V is 2×10⁷; the 10⁷ operating point lies between 3400 and 4100 V depending on stack. Quantum detection efficiency at 30.4 nm spans 0.07–0.17 across the five flight MCP sets (Figure 19), comparable to (but generally below) Cosmic Origins Spectrograph nominal bare-MCP values. The team chose bare MCPs over photocathode-coated ones because photocathodes would amplify Lyα response more than 30.4 nm response, **degrading SNR if any filter pinholes develop** (the crossover open area is 4×10⁻⁴ cm²).

**성능**: 완전 조명 시 곡면 MCP의 펄스 높이 분포(PHD) FWHM은 ~120%(스팟 조명 ~60%, Figure 17)로 평면 MCP(~60%)보다 넓다 — 이상적 구면과의 편차에 의한 게인 비균일이 주원인. 3600 V에서 피크 게인 2×10⁷, 10⁷ 동작점은 스택별 3400–4100 V. 30.4 nm에서 양자 효율은 비행용 5세트에 걸쳐 0.07–0.17 분포(Figure 19), COS 공칭값과 유사하나 일반적으로 약간 낮다. 광전음극을 코팅하지 않고 bare MCP를 선택한 이유는 광전음극이 Lyα 응답을 30.4 nm 응답보다 더 많이 증폭시켜 **필터에 구멍이 생길 경우 SNR을 더 악화시키기 때문**(crossover 개방면적 4×10⁻⁴ cm²).

**Background**: Curved MCP backgrounds run 0.3–0.8 events cm⁻² s⁻¹ from ⁴⁰K β-decay in MCP glass plus adsorbed gases. Combined with mirror radioactivity, the assembled sensor head has dark count rate ~1 cm⁻² s⁻¹.

**배경**: 곡면 MCP 배경은 0.3–0.8 events cm⁻² s⁻¹ (MCP 유리의 ⁴⁰K β-붕괴와 흡착 가스). 거울 방사능과 합산해 조립 센서 헤드의 dark count 율은 ~1 cm⁻² s⁻¹.

**Image performance**: At gain 10⁷ the spatial resolution is ~160 μm FWHM (substantially better than the 300 μm requirement), degrading to ~300 μm at gain 3×10⁶ (Figure 25). Flight MCPs underwent only mild burn-in (8 hr at 100°C bake, 0.01 C/cm² extracted at gain 2–5×10⁵), causing factor 3–5 gain drop that operators compensate by increasing voltage in flight.

**영상 성능**: 게인 10⁷에서 공간 분해능 ~160 μm FWHM(요구치 300 μm 대비 우수), 게인 3×10⁶에서 ~300 μm로 저하(Figure 25). 비행 MCP는 약한 burn-in(8 hr, 100°C 베이크 + 게인 2–5×10⁵에서 0.01 C/cm² 추출)만 거쳐 게인이 3–5배 떨어지며, 이는 궤도에서 전압 상승으로 보상한다.

### Part V: Operations and On-Board Processing (§4) / 운영과 온보드 처리

EUV produces a 52×600 element skymap per sensor head. The 600 columns correspond to spin phase in 0.6° increments (=6 ticks of the 0.1° CIDP spin-phase signal). 50 of the 52 rows record spatial variation parallel to the spin axis; the remaining 2 rows record dark counts and out-of-range error counts. Skymaps integrate over five spacecraft spins (~10 min) and are interleaved with Event Count, Pulse-Height-Distribution, Status, and Diagnostic-Housekeeping packets according to the cycle in Figure 29 and Table III. Total skymap packet size is 62,440 bytes.

EUV는 센서 헤드당 52×600 스카이맵을 생성한다. 600 열은 spin phase를 0.6° 간격(CIDP의 0.1° tick 6개씩)으로 기록하며, 52 행 중 50 행은 spin 축 평행 공간 변동을, 2 행은 dark count와 out-of-range 오류를 기록한다. 스카이맵은 위성 자전 5회(~10분) 동안 적분되며, Event Count·PHD·Status·DHK 패킷과 함께 Figure 29 및 Table III에 정의된 주기로 다중화된다. 스카이맵 패킷 크기는 62,440 bytes.

The on-board pipeline:
1. Detector electronics digitize W, S, Z signals (12-bit ADCs).
2. The Controller normalizes via $q = W + S + 2Z$ and applies $X = k_x W/q - d_x$, $Y = k_y S/q - d_y$ (Eqs. 1–3).
3. Five lookup tables ($Q_x$, $Q_y$, $A_x$, $A_y$, distortion) implement an affine + distortion correction in real time.
4. The corrected (row, column) pair is incremented in the appropriate skymap cell using the spin-phase tick T as a time index in a TDI scheme.
5. Sun-protection logic monitors count rate and reduces HV bias when the Sun enters the FOV.

온보드 파이프라인:
1. 검출기 전자회로가 W, S, Z 신호를 12-bit ADC로 디지타이즈.
2. 컨트롤러가 $q = W + S + 2Z$로 정규화 후 $X = k_x W/q - d_x$, $Y = k_y S/q - d_y$ 적용(식 1–3).
3. 다섯 개의 lookup table($Q_x$, $Q_y$, $A_x$, $A_y$, distortion)이 affine + 왜곡 보정을 실시간 수행.
4. 보정된 (행, 열) 쌍을 spin-phase tick $T$를 시간 인덱스로 사용해 적절한 skymap 셀에 누적(TDI 방식).
5. Sun-protection 로직이 계수율을 감시하여 태양이 시야에 진입하면 HV 전압을 안전 수준으로 강하.

The Harris RTX2010 microprocessor handles event processing within 20 μs per event (against a 66 kHz burst rate, 10 kHz sustained rate). Two Actel RH1280 FPGAs handle the highest-throughput tasks. About 15% of code is hand-written assembly for speed. The Controller communicates with the IMAGE CIDP via RS-422 at 38,400 baud.

Harris RTX2010 마이크로프로세서가 이벤트당 20 μs 이내에 처리(burst rate 66 kHz, sustained rate 10 kHz). 처리량이 가장 큰 부분은 두 개의 Actel RH1280 FPGA가 담당. 코드의 ~15%가 속도를 위해 직접 작성된 어셈블리. 컨트롤러는 IMAGE CIDP와 RS-422 38,400 baud로 통신.

### Part VI: Calibration and Performance (§5) / 보정과 성능

Calibration was performed at the LPL EUV/FUV facility using a J.A.R. Samson VUV Associates LS101-DC discharge source feeding a Seya–Namioka monochromator (~1 nm passband, 0.04° beam divergence). Each sensor head was rastered through azimuth/elevation in a 2° grid, producing a geometry database keyed to (az, el) with image centroids, spot sizes, and relative efficiencies.

LPL EUV/FUV 시설에서 J.A.R. Samson VUV Associates LS101-DC 방전 광원과 Seya–Namioka 분광기(~1 nm 통과대역, 0.04° 빔 발산각)를 사용해 보정 수행. 각 센서 헤드를 azimuth/elevation 2° grid로 raster하여 (az, el) 키의 영상 centroid, 스팟 크기, 상대 효율을 담은 geometry database 작성.

The five lookup tables (§5.2.1–5.2.2) were computed by minimizing RMS error between table-produced X,Y and floating-point-computed X,Y, with out-of-range events mapped to coordinate 127 (using C-style 0-to-N-1 indexing for length-N arrays). The distortion table is 128×128 returning a (52,50) corrected pair; out-of-range events go to row 50, dark-region events go to row 51. Two further row entries (50 and 51) are dedicated to dark counts and out-of-range errors so that flight software does not need conditionals for these cases.

다섯 lookup table은 표가 생성하는 X,Y와 부동소수점 계산 X,Y의 RMS 오차를 최소화하도록 결정. out-of-range 이벤트는 좌표 127로 매핑(N 길이 배열에서 0..N-1 인덱싱). distortion table은 128×128 → (52,50) 보정 쌍을 반환하며 out-of-range는 50번 행, dark 영역은 51번 행으로 보내 비행 소프트웨어가 조건문 없이 처리.

**Photometric calibration result (Eq. 4)**:
$$ S = A\,\omega\,\epsilon\,\tau\,\rho\,\frac{10^{6}}{4\pi} = (21.8)(1.1\times10^{-4})(0.14)(0.33)(0.22)\frac{10^{6}}{4\pi} = 1.9\ \text{count s}^{-1}\,R^{-1}. $$

The product $S\delta$, with duty cycle $\delta = 30°/360° = 0.083$, gives the on-orbit signal rate per Rayleigh including the spin-modulated exposure of any given map cell.

**광도 보정 결과(식 4)**: $S = 1.9$ count s⁻¹ R⁻¹. duty cycle $\delta = 30°/360° = 0.083$를 곱한 $S\delta$가 자전 변조된 노출 시간을 반영한 궤도 신호율.

**SNR estimate**: dark count = 1 count cm⁻² s⁻¹ → noise-equivalent signal 0.03 R; for 0.1 R brightness in 10-min integration, SNR ≈ 3 per map element; for 1 R, SNR ≈ 10. Worst-case ionospheric brightness drives instantaneous count rate to ~40 kHz, just below the 66 kHz electronics ceiling.

**SNR 추정**: dark count = 1 count cm⁻² s⁻¹ → noise-equivalent 신호 0.03 R; 10분 적분 시 0.1 R 밝기에서 SNR ≈ 3(맵 셀당), 1 R에서 SNR ≈ 10. 최악의 전리권 밝기 시 순간 계수율은 ~40 kHz로, 전자회로 한계 66 kHz 직하.

**In-flight calibration sources**: Earth itself is too uncertain at 30.4 nm; the Moon (0.5° diameter, well-characterized by EUVE — Gladstone et al. 1994) gives 230 Hz instantaneous count rate (10⁴ counts in 10 min) under favorable geometry. The white dwarf HZ-43 is the brightest 30.4 nm astrophysical target visible through low-column-density interstellar windows; expected signal is 60 counts in 10 min, sufficient for confirmation.

**비행 중 보정 광원**: 지구는 30.4 nm에서 불확실. 달(직경 0.5°, EUVE Gladstone 외 1994에서 잘 특성화됨)이 230 Hz 순간 계수율(10분 10⁴ counts) 제공. 백색왜성 HZ-43은 저 H 칼럼 밀도 interstellar 창을 통해 보이는 가장 밝은 30.4 nm 천체; 10분 ~60 counts 예상.

### Part VII: Data Products and Levels (§6) / 데이터 산물

Telemetry packets (Table III): Skymap 0/1/2 (62,440 bytes each), Event Count 0/1/2 (7,240 bytes), PHD (2,088 bytes), WSZT 0/1/2 (raw events, 62,440 bytes), Status (2,088 bytes), DHK (38,444 bytes at 2 Hz sampling), Memory Dump (variable). Five-spin EUV cycles distribute packet generation evenly: spins 1–3 carry Skymap+EventCount+PHD+Status; spin 4 carries DHK+Status; spin 5 carries Status only.

텔레메트리 패킷(Table III): Skymap 0/1/2(각 62,440 bytes), Event Count 0/1/2(7,240 bytes), PHD(2,088 bytes), WSZT 0/1/2(raw events, 62,440 bytes), Status(2,088 bytes), DHK(2 Hz 샘플링, 38,444 bytes), Memory Dump(가변). 5-spin EUV 주기가 패킷 생성을 균등 분배: spin 1–3은 Skymap+EventCount+PHD+Status; spin 4는 DHK+Status; spin 5는 Status만.

Level 1 processing pipeline: dark subtraction → pulse-height correction (using PHD) → flat-field correction → pile-up correction (using Event Count) → merge three sensor-head maps via geometric calibration overlap → convert to Rayleighs → transform to Earth-fixed coordinates using orbit/attitude. Per IMAGE policy, all data is publicly released without proprietary period via NSSDC.

Level 1 처리 파이프라인: dark 차감 → PHD 사용 펄스 높이 보정 → flat-field 보정 → Event Count 사용 pile-up 보정 → 세 센서 헤드 맵을 geometric calibration 중첩으로 병합 → Rayleighs로 변환 → 궤도/자세 정보로 지구 고정 좌표계 변환. IMAGE 정책에 따라 모든 데이터는 NSSDC를 통해 점유 기간 없이 공개.

### Part VIII: Electronics Appendix (Appendix A) / 전자회로 부록

The Controller (Figure 30) is built around a Harris RTX2010 16-bit forth-stack microprocessor with 128 kbytes RAM (flight code + map buffers), EEPROM (5 lookup tables + flight code + parameters), and PROM (bootstrap). Two Actel RH1280 FPGAs handle high-speed event ingestion and HVPS control. Power: ~9 W total (Filter LVPS + detector electronics + Controller + HVPS).

Detector electronics (Figure 32): three charge-sensitive preamplifier channels (W, S, Z) feeding three sample-hold + 12-bit ADC chains, with a window comparator gate to reject events outside valid charge range. Power < 800 mW. A stimulation pulse generator produces simulated two-spot events for in-flight liveness/stability testing.

HVPS: 21.6×18.7×4.1 cm, 1.65 kg, three independent –200 to –6000 V output channels under 12-bit DAC control. Output time constant ~1.35 s; transitioning between sun-protection low level and operating level takes ~6 s, an acceptably small fraction of the 120-s spin period.

Radiation philosophy: 100 kRad-rated parts, 0.38 cm Al shielding wall reduces expected mission dose to 50 kRad (2× margin).

컨트롤러는 Harris RTX2010(16-bit Forth-stack) 마이크로프로세서, 128 kbyte RAM, EEPROM(5 lookup table + 비행 코드 + 파라미터), PROM(부트스트랩)으로 구성. Actel RH1280 FPGA 2개가 고속 이벤트 수집과 HVPS 제어. 총 전력 ~9 W. 검출기 전자회로는 W/S/Z 전하 민감 증폭기 3채널 + sample-hold/12-bit ADC, 윈도우 비교기로 잘못된 전하 범위 차단, 전력 < 800 mW. HVPS는 –200 ~ –6000 V 12-bit DAC 제어 3채널, 시정수 ~1.35 s. 방사선: 100 kRad 부품, 0.38 cm Al 차폐로 임무 누적선량 50 kRad (2× 여유).

---

## 3. Key Takeaways / 핵심 시사점

1. **Optical thinness makes plasmaspheric He+ 30.4 nm uniquely interpretable** — Because the plasmasphere is optically thin at 30.4 nm, line-of-sight brightness is directly proportional to He+ column density without radiative-transfer modeling. This is the single physical fact that justifies the entire EUV concept.

   **광학적 얇음이 플라스마권 He+ 30.4 nm 영상을 해석 가능하게 만든다** — 플라스마권이 30.4 nm에서 광학적으로 얇기 때문에 시선 방향 강도가 He+ 칼럼 밀도에 정비례한다. 복잡한 복사 전달 모델링 없이 직접 역추론이 가능하며, 이 단일 물리적 사실이 EUV 컨셉 전체를 정당화한다.

2. **Wide field of view is non-negotiable for true global imaging** — EUV's 84°×360° annulus from three 30°-FOV heads is not a luxury; the entire plasmasphere must fit in one snapshot to capture dynamics on the 10-min cadence at which storms erode the plasmapause. This drove every design choice (f/0.8 optics, three heads, spin scanning).

   **진정한 전 지구 영상에는 광시야가 필수** — 84°×360° 환형 시야는 사치가 아니라 필수다. 플라스마포즈 침식이 진행되는 ~10분 시간 척도에 플라스마권 전체를 한 번에 담아야 한다는 요구가 f/0.8 광학, 3-헤드 구성, 자전 스캐닝 등 모든 설계 선택을 결정했다.

3. **The 58.4-nm rejection problem drove the U/Si multilayer innovation** — Most materials reflect 58.4 nm better than 30.4 nm, so the multilayer's top layer must function as an antireflection coating at 58.4 nm while contributing as a high-index reflector at 30.4 nm. The UOₓ cap (≈4.5 nm post-oxidation) was the **first XUV multilayer to use this dual-function approach**.

   **58.4 nm 차단 문제가 U/Si 다층막 혁신을 이끔** — 대부분의 재료가 58.4 nm를 30.4 nm보다 잘 반사하므로, 다층막 상단이 58.4 nm에서는 반사 방지층, 30.4 nm에서는 고굴절률 반사층으로 동시 작동해야 했다. UOₓ 캡(산화 후 ≈4.5 nm)이 **이 이중 기능을 시도한 최초의 XUV 다층막**.

4. **Genetic algorithms entered space-mission optical design** — Lunt & Turley used a GA to optimize discrete material choices and continuous layer thicknesses, exploring fully aperiodic designs. Although the final flight mirror was a periodic U/Si stack (for fabrication and characterization reasons), the GA pointed to U as the right top-layer material, validating computational design as part of EUV-mirror engineering.

   **유전 알고리즘이 우주 임무 광학 설계에 도입됨** — Lunt & Turley는 GA로 이산적 재료 선택과 연속적 층 두께를 동시 최적화하며 완전 비주기 설계까지 탐색했다. 최종 비행 거울은 제작·특성화 용이성 때문에 주기 U/Si로 결정되었으나, GA가 상층 U를 지목한 것은 EUV 다층막 공학에서 계산적 설계의 가치를 입증한다.

5. **Annular aperture is a clever angle filter, not just a light hole** — The annular shape limits incidence angles to 12°–18° from normal, the narrow band over which the multilayer is reflective. Without this constraint, off-angle rays would scatter or absorb badly and degrade contrast. The aperture is simultaneously a spatial integrator and an angular bandpass.

   **환형 입사창은 단순 광량 창이 아닌 각도 필터** — 환형 모양이 입사각을 12°–18°로 제한해 다층막이 반사하는 좁은 각도 대역에 입사를 가둔다. 이 제약이 없으면 비스듬한 광선이 산란·흡수되어 대비가 저하된다. 입사창은 공간 적분기와 각도 대역 필터를 겸한다.

6. **Bare MCPs beat photocathodes when filter pinholes are likely** — A photocathode would amplify Lyα response more than 30.4 nm response, so any pinhole-induced Lyα leakage would degrade SNR more than it would help. The crossover area is 4×10⁻⁴ cm², comparable to one mesh pane. This is a counter-intuitive but well-justified detector choice driven by reliability statistics.

   **필터 구멍 가능성이 있을 때는 bare MCP가 광전음극보다 우수** — 광전음극은 Lyα 응답을 30.4 nm 응답보다 더 많이 증폭하므로, 필터 구멍이 생기면 SNR을 오히려 악화시킨다. crossover 면적은 4×10⁻⁴ cm²로 메시 한 칸 크기. 신뢰성 통계가 이끈 직관에 반하지만 합리적 선택.

7. **Time-delayed integration on a spinning platform replaces a tracking gimbal** — Instead of a complex pointing/tracking gimbal, EUV uses spacecraft spin + 5-table on-board geometric correction + TDI accumulation into the 52×600 skymap. This shifts complexity from mechanism to software, saves mass and power, and enables 10-min cadence with no moving optical parts.

   **자전 플랫폼의 TDI가 추적 짐벌을 대체** — 복잡한 포인팅 짐벌 대신 위성 자전 + 5-테이블 온보드 기하 보정 + TDI 누적(52×600 스카이맵)을 사용. 복잡성을 기구에서 소프트웨어로 이전해 질량·전력을 절감하고 광학적 가동부 없이 10분 cadence 달성.

8. **Sensitivity is a product of five well-characterized factors** — The 1.9 count s⁻¹ R⁻¹ headline number factors as A × ω × ε × τ × ρ × 10⁶/(4π). Each of the five factors was measured independently, and consistency between the product and direct calibration with EUV photodiodes (Gullikson et al. 1996, NIST-traceable) provides cross-validation. **This factorization is a teaching template for any photon-counting EUV imager.**

   **감도는 잘 특성화된 다섯 인자의 곱** — 헤드라인 수치 1.9 count s⁻¹ R⁻¹은 A × ω × ε × τ × ρ × 10⁶/(4π)로 분해된다. 다섯 인자를 독립적으로 측정한 결과의 곱과 EUV 광다이오드(Gullikson 외 1996, NIST 연계) 직접 보정값이 일치하여 교차 검증이 가능. **이 인자 분해는 광자 계수 EUV 영상기 설계의 교육 템플릿이다.**

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Resonant scattering and column density / 공명 산란과 칼럼 밀도

For an optically thin medium illuminated by solar 30.4 nm flux, the line-of-sight brightness in Rayleighs is

$$ I_{30.4}(R) = \frac{1}{4\pi} \frac{10^{-6}}{1\ R}\,g\!\int_{LOS} n_{\text{He}^+}(s)\,ds, $$

where $g$ (s⁻¹) is the resonance scattering rate per ion driven by the solar 30.4 nm flux $\pi F_\odot$ at a given heliocentric distance:

$$ g = \frac{\pi F_\odot \,\sigma_0}{1\ \text{photon}}, $$

with $\sigma_0$ the line-center cross section. Typical $g \sim 5\times 10^{-7}$ s⁻¹ at solar minimum.

광학적으로 얇은 매질에서 30.4 nm 강도는 He+ 칼럼 밀도에 정비례하며 g-factor는 태양 30.4 nm 플럭스에 비례한다. 일반적으로 $g \sim 5\times 10^{-7}$ s⁻¹.

**Numerical example**: At plasmapause peak density $n_{\text{He}^+} \sim 200$ cm⁻³ over a path length $L \sim 2\,R_E = 1.27\times 10^9$ cm, column density $\sim 2.5\times 10^{11}$ cm⁻². With $g = 5\times 10^{-7}$ s⁻¹, brightness $\sim (g/4\pi)\times N \times 10^{-6} \approx 10$ R — matching the maximum-brightness requirement #1 in §2.

**수치 예**: 플라스마포즈 피크 밀도 $n_{\text{He}^+} \sim 200$ cm⁻³, 경로 길이 $L \sim 2\,R_E$ → 칼럼 밀도 $\sim 2.5\times 10^{11}$ cm⁻². $g = 5\times 10^{-7}$ s⁻¹일 때 강도 ~10 R로 §2의 최대 밝기 요건 #1과 일치.

### 4.2 Photon-counting sensitivity (Eq. 4) / 광자 계수 감도

$$ \boxed{\; S = A\,\omega\,\epsilon\,\tau\,\rho\,\frac{10^{6}}{4\pi}\;} $$

| Symbol | Meaning | Value |
|---|---|---|
| $A$ | annular aperture area | 21.8 cm² |
| $\omega$ | solid angle of resolution element | 1.1×10⁻⁴ sr |
| $\epsilon$ | bare-MCP quantum detection efficiency at 30.4 nm | 0.14 |
| $\tau$ | aluminum-filter transmission at 30.4 nm | 0.33 |
| $\rho$ | U/Si multilayer reflectivity at 30.4 nm | 0.22 |
| $10^6/4\pi$ | converts Rayleigh to photons cm⁻² s⁻¹ sr⁻¹ | 7.96×10⁴ |

Plug in: $S = 21.8 \times 1.1\times 10^{-4} \times 0.14 \times 0.33 \times 0.22 \times 7.96\times 10^4 \approx 1.94$ count s⁻¹ R⁻¹.

값 대입: $S \approx 1.94$ count s⁻¹ R⁻¹. 모든 인자가 독립적으로 측정되어 교차 검증된다.

### 4.3 Photon-conversion efficiency (Eq. 5) / 광자 변환 효율

$$ E = \epsilon \tau \rho = 0.14 \times 0.33 \times 0.22 = 1.02\times 10^{-2}\ \text{counts/photon}. $$

This is the appropriate efficiency for collimated calibration beams (without the geometric factor $A\omega 10^6/4\pi$).

이는 콜리메이트된 보정 빔에 적용되는 효율(기하 인자 제외).

### 4.4 Wedge-and-strip charge-division decoding (Eqs. 1–3) / 위치 디코딩

Three electrode signals $(W, S, Z)$ encode the position of the MCP charge cloud:

$$ q = W + S + 2Z, \qquad X = \frac{k_x W}{q} - d_x, \qquad Y = \frac{k_y S}{q} - d_y. $$

**Why divide by $q$?** Pulse heights vary by ~120% FWHM; normalizing eliminates gain dependence so position depends only on the *fraction* of charge each electrode captures. The factor 2 on $Z$ reflects the zigzag's geometric capture pattern, in which it always sees twice as many wedge edges as a wedge or strip.

**$q$로 나누는 이유?** 펄스 높이가 ~120% FWHM 변동하지만 정규화로 게인 의존성을 제거해 각 전극이 받는 전하 *비율*만으로 위치를 결정. $Z$ 위 계수 2는 zigzag의 기하학적 패턴이 wedge나 strip의 두 배 가장자리를 보는 것을 반영.

Calibration determines $k_x, k_y, d_x, d_y$ to map $X,Y$ into integer detector indices in [0, 127] for the affine table.

보정으로 $k_x, k_y, d_x, d_y$를 결정하여 $X,Y$를 [0,127] 범위 정수 검출기 인덱스로 매핑(affine 테이블).

### 4.5 SNR derivation / SNR 유도

For a target source brightness $B$ Rayleighs, integration time $t$ s, dark count rate $D$ counts s⁻¹, duty cycle $\delta$ = 30°/360°:

$$ \text{Signal} = S \delta B t, \qquad \text{Noise} = \sqrt{S\delta B t + D t}, $$

$$ \text{SNR} = \frac{S\delta B t}{\sqrt{S\delta B t + Dt}}. $$

With $S = 1.9$ count/s/R, $\delta = 0.083$, $D = 0.03$ R-equivalent in dark counts (the noise-equivalent signal), $t = 600$ s, and $B = 0.1$ R: SNR ≈ 3 per map cell. For $B = 1$ R: SNR ≈ 10. These match the requirement to detect 0.1–0.3 R features in 10 minutes.

대상 밝기 $B$ R, 적분 시간 $t$ s, dark count 율 $D$, duty cycle $\delta = 0.083$일 때 SNR 식은 위와 같다. $B = 0.1$ R, $t = 600$ s에서 SNR ≈ 3, $B = 1$ R에서 SNR ≈ 10으로 §2 요구치를 충족.

### 4.6 Time-delayed integration / 시간 지연 적분

The TDI builds a 52×600 skymap by binning each photon event at $(X_{corr}, Y_{corr}, T)$ where $T$ is the spin-phase tick. The required transformation is

$$ \text{cell}(i, j) = \text{distortion}\Big[\,A_x\!\Big(\!\tfrac{Q_x(q)\,W}{1}\Big),\ A_y\!\Big(\!\tfrac{Q_y(q)\,S}{1}\Big)\Big] $$

with $j = T \pmod{600}$ and the distortion table mapping into rows 0–49 (good data), 50 (out-of-range), 51 (dark region).

TDI는 각 광자 이벤트를 $(X_{corr}, Y_{corr}, T)$ 좌표에 누적하여 52×600 skymap을 구성. $j = T \pmod{600}$, 행 0–49는 정상 데이터, 50은 out-of-range, 51은 dark 영역.

### 4.7 Vignetting model / 비네팅 모델

The relative throughput as a function of off-axis angle $\theta$ (Figure 6) is approximated by:

$$ V(\theta) = \begin{cases} 1, & \theta < 7° \\ 1 - 0.65\,\Big(\tfrac{\theta - 7°}{8°}\Big)^2, & 7° \leq \theta \leq 15° \end{cases} $$

giving $V(15°) \approx 0.53$ as reported. The 3° overlap between adjacent sensor heads partially compensates for the spin-modulated edge exposure.

비네팅(Figure 6)은 위 식으로 근사되며 $V(15°) \approx 0.53$. 인접 헤드 3° 중첩으로 자전 변조 가장자리 노출 부분 보상.

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1968 ─────── OGO-5: first geocoronal/plasmaspheric He+ 30.4 nm detection
                     (Meier reference work; per-ion g-factor measured)
                                │
1974 ─────── Meier & Weller: low-altitude photometer maps of plasmaspheric He+
                                │
1982 ─────── Chakrabarti et al.: early ground-based EUV plasmasphere studies
                                │
1985 ─────── Barstow et al.: spherical MCP detectors for ROSAT WFC
                                │
1988 ─────── Siegmund et al.: wedge-and-strip readout, IEEE Trans. Nucl. Sci.
                                │
1990 ─────── Williams: case for global magnetospheric imaging
1990 ─────── Bloch et al.: ALEXIS X-ray telescopes (heritage for EUV optics)
                                │
1993 ─────── Sandel et al.: Remote Sensing Reviews case for global imagers
                                │
1995 ─────── IMAGE selected as first MIDEX mission
1995 ─────── Carpenter: AGU EOS plasmasphere context paper
                                │
1998 ─────── Meier et al.: improved plasmaspheric He+ 30.4 nm models
1998 ─────── Thomsen et al.: two-stage refilling paradigm
                                │
1999 ─────── EUV mirrors fabricated at BYU (Lunt & Turley GA design)
1999 ─────── Manuscript submitted (received 25 May 1999)
                                │
2000 ─────── ★ This paper / IMAGE launched 25 March
                                │
2001 ─────── Burch et al.: first plasmaspheric plumes & notches imaged
                                │
2003 ─────── Goldstein et al.: dynamics of plasmapause from EUV
                                │
2008 ─────── IMAGE contact lost; EUV-derived models continue to inform missions
                                │
2013+ ────── ICON, JUNO/UVS, LRO/LAMP inherit curved-MCP + multilayer recipe
```

EUV는 1968년 OGO-5 He+ 검출로 시작된 30년 흐름의 정점에 위치하며, 이후 plasmasphere–ring current 결합 연구의 표준 데이터를 만들어냈다. 광학 측면에서는 1985–1990년대 ROSAT/ALEXIS의 곡면 MCP + wedge-and-strip 유산을 잇고, 다층막 측면에서는 90년대 후반 GA 기반 광학 설계의 첫 우주 임무 적용 사례 중 하나이다. EUV 직후 IMAGE/EUV 관측은 plasmaspheric plume과 notch를 처음 영상으로 직접 확인했고 (Burch et al. 2001), 그 데이터는 ICON·JUNO/UVS·LRO/LAMP 등 후속 임무 설계의 참고가 되었다.

EUV sits at the apex of a 30-year arc beginning with OGO-5's 1968 He+ detection, and it produced the canonical dataset for plasmasphere–ring-current coupling studies. Optically, it inherited the curved-MCP + wedge-and-strip lineage of ROSAT/ALEXIS and was among the first space missions to apply genetic-algorithm-based optical design. Immediately post-launch, EUV imaging directly confirmed plasmaspheric plumes and notches (Burch et al. 2001), and its instrumental recipe informed subsequent imagers on ICON, JUNO/UVS, and LRO/LAMP.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Meier & Weller (1974) J. Geophys. Res. 79, 1575 | First low-altitude photometric detection of plasmaspheric He+ 30.4 nm; established the resonance-scattering remote-sensing technique that EUV inherits at global scale / 플라스마권 He+ 30.4 nm의 첫 저고도 광도계 검출, EUV가 전 지구 규모로 계승하는 공명 산란 원격 탐사 기법 확립 | Foundational technique paper / 기초 기법 논문 |
| Sandel et al. (1993) Remote Sensing Rev. 8, 147 | Same first author's prior advocacy for global magnetospheric imaging — sets the conceptual context for EUV / 동일 1저자의 전 지구 자기권 영상화 옹호 — EUV 개념적 맥락 | Direct intellectual predecessor / 직접 지적 전례 |
| Williams (1990) in Hultqvist & Falthammer eds. | Case for magnetospheric imaging that IMAGE Mission was built to realize / IMAGE 임무가 실현하고자 한 자기권 영상화 비전 | Mission rationale source / 임무 정당화 출처 |
| Bloch et al. (1990) SPIE 1344, 154 | ALEXIS X-ray telescope heritage — the wide-field f/0.8 prime-focus + curved-MCP architecture EUV adapts to EUV wavelengths / EUV가 EUV 파장으로 변형 적용한 광시야 곡면 MCP 아키텍처 출처 | Direct optical heritage / 직접 광학 유산 |
| Siegmund et al. (1986b) SPIE 689, 40 | Wedge-and-strip anode design used in EUV detector readout / EUV 검출기에 사용된 wedge-and-strip 양극 설계 | Detector readout heritage / 검출기 readout 유산 |
| Lunt & Turley (1998, 1999a/b) | Genetic-algorithm multilayer optimization underlying the EUV mirror design / EUV 거울 설계의 GA 다층막 최적화 | Mirror design methodology / 거울 설계 방법론 |
| Carpenter et al. (1993) J. Geophys. Res. 98, 19243 | Plasmasphere science context — the questions EUV is built to answer / EUV가 답하고자 하는 플라스마권 과학 맥락 | Scientific motivation / 과학적 동기 |
| Thomsen et al. (1998) AGU Monograph 104 | Two-stage refilling paradigm that EUV data is intended to test / EUV 데이터가 검증하고자 한 2단계 리필링 모델 | Specific testable hypothesis / 구체적 검증 가설 |
| Gladstone et al. (1994) | EUVE Moon characterization that EUV uses for in-flight calibration / EUV가 비행 중 보정에 사용하는 EUVE 달 특성화 | In-flight calibration source / 비행 중 보정 광원 |
| Burch et al. (2001) Geophys. Res. Lett. (post-publication) | First IMAGE/EUV plasmaspheric plume and notch images, validating EUV's promise / IMAGE/EUV의 첫 plasmaspheric plume과 notch 영상, EUV 약속 검증 | Direct scientific successor / 직접 과학적 후속 |

---

## 7. References / 참고문헌

- Sandel, B. R., Broadfoot, A. L., Curtis, C. C., King, R. A., Stone, T. C., Hill, R. H., Chen, J., Siegmund, O. H. W., Raffanti, R., Allred, D. D., Turley, R. S., and Gallagher, D. L., "The Extreme Ultraviolet Imager Investigation for the IMAGE Mission", *Space Science Reviews* **91**, 197–242, 2000. DOI: 10.1023/A:1005263510820 (this paper / 본 논문)
- Barstow, M. A., Holberg, J. B., and Koester, D., 1995, *Mon. Not. R. Astron. Soc.* **274**, L31. (HZ-43 calibration source)
- Barstow, M. A. and Sansom, A. E., 1990, *SPIE* **1344**, 244. (curved MCP for ROSAT WFC)
- Bloch, J. J. et al., 1990, *SPIE* **1344**, 154. (ALEXIS X-ray telescope optics)
- Carpenter, D. L. et al., 1993, *J. Geophys. Res.* **98**, 19243. (plasmasphere science context)
- Chakrabarti, S., Paresce, F., Bowyer, S., Chiu, S., and Aikin, A., 1982, *Geophys. Res. Lett.* **9**, 151. (early EUV plasmasphere)
- Fennimore, A., Allred, D., Turley, R. S., Vazquez, C., and Chao, B., 1999, *Appl. Optics* (submitted). (sputtered U optical constants)
- Fok, M.-C. et al., 1995, *J. Geophys. Res.* **100**, 9619. (plasmasphere–ring current coupling)
- Fraser, G. W., 1984, *Nucl. Instrum. Meth.* **221**, 115. (MCP pulse-height theory)
- Gladstone, G. R., McDonald, J. S., Boyd, W. T., and Bowyer, S., 1994, *Geophys. Res. Lett.* **21**, 461. (EUVE Moon)
- Gullikson, E. M. et al., 1996, *J. Electron Spect. Rel. Phen.* **80**, 313. (NIST EUV photodiode standard)
- Lunt, S. and Turley, R. S., 1998, *Phys. X-Ray Multilayer Struct.* **4**. (GA multilayer design)
- Meier, R. R. and Weller, C. S., 1974, *J. Geophys. Res.* **79**, 1575. (foundational He+ 30.4 nm photometry)
- Meier, R. R. et al., 1998, *J. Geophys. Res.* **103**, 17505. (modern plasmaspheric He+ models)
- Sandel, B. R., Drake, V. A., Broadfoot, A. L., Hsieh, K. C. and Curtis, C. C., 1993, *Remote Sensing Rev.* **8**, 147. (case for magnetospheric imaging)
- Siegmund, O. H. W., Lampton, M., Bixler, J., Chakrabarti, S., Vallerga, J., Bowyer, S. and Malina, R. F., 1986a, *SPIE* **689**, 40. (wedge-and-strip readout)
- Siegmund, O. H. W., Vallerga, J., and Jelinsky, P., 1986b, *J. Opt. Soc. Am.* **A3**, 2139. (MCP characterization)
- Skulina, K. M., 1995, *Appl. Optics* **34**, 3727. (XUV multilayer fabrication review)
- Thomsen, M. F., McComas, D. J., Borovsky, J. E. and Elphic, R. C., 1998, *AGU Monograph* **104**, 355. (two-stage refilling)
- Williams, D. J., 1990, in Hultqvist & Falthammer eds., *Magnetospheric Physics*, Plenum, pp. 83–101. (magnetospheric imaging vision)
