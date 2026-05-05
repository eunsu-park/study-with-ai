---
paper_id: 34
topic: Solar_Observation
date: 2026-04-23
type: briefing
title: "An Instrument to Measure Coronal Emission Line Polarization"
authors: "Tomczyk, S.; Card, G.L.; Darnell, T.; Elmore, D.F.; Lull, R.; Nelson, P.G.; Streander, K.V.; Burkepile, J.; Casini, R.; Judge, P.G."
year: 2008
journal: "Solar Physics 247, 411-428"
doi: "10.1007/s11207-007-9103-6"
---

# CoMP: An Instrument to Measure Coronal Emission Line Polarization / 코로나 방출선 편광 측정 장비

## 1. Paper Metadata / 논문 메타데이터

- **Title / 제목**: An Instrument to Measure Coronal Emission Line Polarization
- **Authors / 저자**: S. Tomczyk, G.L. Card, T. Darnell, D.F. Elmore, R. Lull, P.G. Nelson, K.V. Streander, J. Burkepile, R. Casini, P.G. Judge (HAO/NCAR)
- **Year / 연도**: 2008 (received 2007-06-28, accepted 2007-12-03)
- **Journal / 저널**: Solar Physics, Vol. 247, pp. 411-428
- **DOI**: 10.1007/s11207-007-9103-6
- **Pages / 분량**: 18 pages

## 2. Context and Motivation / 배경 및 동기

**EN:** Measuring coronal magnetic fields is one of the most stubborn outstanding problems in solar physics. Essentially every dynamic coronal phenomenon — flares, CMEs, SEPs, the solar wind — is driven by magnetic energy, but routine, direct measurements of **B** in the corona have been essentially unavailable. Existing techniques (gyroresonance radio emission, Faraday rotation, thermal bremsstrahlung) are either limited to active-region strengths >200 G or to sparse pointings. CoMP was designed to fill this gap by exploiting the **Hanle** and **Zeeman** effects in near-infrared forbidden lines of Fe XIII at 1074.7 and 1079.8 nm, and in the He I 1083.0 nm chromospheric line.

**KR:** 태양 코로나 자기장 측정은 태양 물리학의 오래된 난제이다. 플레어, CME, SEP, 태양풍 등 거의 모든 코로나 역동 현상은 자기 에너지에 의해 구동되지만, 코로나의 **B**를 직접 상시적으로 측정하는 기법은 사실상 부재했다. 기존 방법(gyroresonance, Faraday 회전, thermal bremsstrahlung)은 활동영역 200 G 이상 혹은 매우 드문 시선 방향에 한정된다. CoMP는 Fe XIII 1074.7/1079.8 nm 근적외선 금지선과 He I 1083.0 nm 채층선의 **Hanle 효과**와 **Zeeman 효과**를 이용해 이 공백을 메우기 위해 설계되었다.

## 3. Prerequisites / 선행 지식

**EN:**
1. **Stokes vector formalism** (I, Q, U, V) and Mueller matrix description of polarizing optics.
2. **Zeeman effect** in the weak-field limit: Stokes V ∝ g_eff λ² (dI/dλ) B_LOS.
3. **Hanle effect**: modification of resonance-scattering linear polarization by a magnetic field; saturated regime for B ≳ 10 G in Fe XIII.
4. **Birefringent Lyot filter theory** (Lyot 1944; Evans 1949; Beckers & Dunn 1965): multi-stage calcite/polarizer stack producing narrow bandpass.
5. **Liquid-crystal variable retarders (LCVRs)** for electrically tunable birefringence.
6. **Coronagraph principles** (Lyot): internal occulting, aspheric singlet objective, Lyot stop.
7. **Photon noise statistics** and signal-to-noise optimization for polarimetry.

**KR:**
1. **Stokes 벡터** 형식(I, Q, U, V)과 편광 광학의 Mueller 행렬 기술.
2. 약자기장 극한의 **Zeeman 효과**: Stokes V ∝ g_eff λ² (dI/dλ) B_LOS.
3. **Hanle 효과**: 공명산란 선형편광의 자기장에 의한 변형; Fe XIII에서 B ≳ 10 G 포화.
4. **Birefringent Lyot 필터** 이론(Lyot 1944 등): calcite/편광자 다단 적층으로 좁은 대역폭 구현.
5. **액정 가변 위상지연자(LCVR)**로 전기적 조정 가능.
6. **Lyot 코로나그래프** 원리: 내부 occulting, 비구면 단렌즈 대물, Lyot stop.
7. **광자 잡음** 통계와 편광 측정의 S/N 최적화.

## 4. Key Vocabulary / 핵심 용어

| Term | 한글 | Meaning |
|------|------|---------|
| CoMP | 코로나 다채널 편광계 | Coronal Multichannel Polarimeter |
| COS | 코로나 원샷 코로나그래프 | Coronal One Shot coronagraph (20 cm, f/11, Sac Peak) |
| Fe XIII forbidden line | 철 XIII 금지선 | 1074.7, 1079.8 nm magnetic-dipole M1 transitions |
| Hanle effect | 한레 효과 | Scattering polarization modification by B |
| Zeeman effect | 제만 효과 | Line splitting/polarization by B |
| LCVR | 액정 가변 지연자 | Liquid Crystal Variable Retarder |
| Lyot filter | 라이오트 필터 | Birefringent tunable narrow-band filter |
| Lyot stop | 라이오트 스톱 | Diffraction-suppression aperture |
| FOV | 시야각 | Field of View (CoMP: 2.8 R_sun) |
| FWHM | 반치전폭 | Full Width at Half Maximum (0.13 nm) |
| Wollaston prism | 월라스톤 프리즘 | Polarizing beamsplitter |
| POS / LOS | 천구면/시선 | Plane of Sky / Line of Sight |
| Stokes I,Q,U,V | 스토크스 벡터 | Polarization state basis |
| g_eff | 유효 란데 인자 | Effective Landé factor (1.5 for Fe XIII 1074.7) |
| μB_sun | 밀리오닛 태양 디스크 휘도 | 10⁻⁶ of disk center intensity |
| Response matrix R | 응답 행렬 | 4×4 Mueller-like calibration matrix |

## 5. Key Questions / 주요 질문

**EN:**
1. What drives the choice of **1074.7/1079.8 nm Fe XIII** forbidden lines over shorter-wavelength lines?
2. Why is the optimum filter FWHM near **0.13 nm** and displaced ≈ 0.1 nm into the line wing?
3. How does the **four-stage calcite Lyot filter + 6 LCVRs + Wollaston beamsplitter** achieve complete I, Q, U, V measurement?
4. What is the **polarimetric accuracy requirement** (Q/I, U/I ≈ 10⁻³, V/I ≈ 10⁻⁴ for 1 G)?
5. How is the **4×4 response matrix R** determined and how small must its errors be?
6. Can CoMP reach **photon-noise-limited** performance for coronal B_LOS?
7. What was the scientific payoff — first detection of **Alfvén waves** (Tomczyk+ 2007 Science)?

**KR:**
1. 왜 **Fe XIII 1074.7/1079.8 nm** 근적외 금지선이 선택되었는가?
2. 최적 필터 FWHM이 **0.13 nm**이고 선 중심에서 약 0.1 nm 벗어난 이유는?
3. **4단 calcite Lyot 필터 + 6 LCVR + Wollaston 빔스플리터**가 어떻게 I, Q, U, V 완전측정을 달성하는가?
4. 편광 정확도 요구사항(Q/I, U/I ≈ 10⁻³, V/I ≈ 10⁻⁴, 1 G 대응)은?
5. **4×4 응답 행렬 R**을 어떻게 결정하고 허용 오차는 얼마인가?
6. CoMP가 **광자잡음 한계** 성능에 도달할 수 있는가?
7. 과학적 성과: **Alfvén 파 최초 검출**(Tomczyk+ 2007 Science)?

## 6. Reading Strategy / 읽기 전략

**EN:**
1. **Section 2 (Design drivers)**: derive Equation (1) for the S/N and understand Figure 1 — the optimum filter width/displacement.
2. **Section 3 (Instrument)**: trace the optical path in Figure 2; understand the role of each element in Figure 3 (LCVRs → polarizer → calcite stages → Wollaston).
3. **Section 4 (Calibration)**: understand Equation (2) S_meas = R·S_input and the 23-parameter nonlinear fit. Equations (5)-(8) quantify tolerable errors.
4. **Section 5 (Sample data)**: study Figures 7-10 — intensity, LOS velocity, POS B azimuth, LOS B strength, and the photon-noise confirmation (~3 G in 2.4 h).
5. Cross-reference Tomczyk+ 2007 Science for the Alfvén wave discovery.

**KR:**
1. **2절 설계 구동 요인**: 식 (1) S/N 유도, 그림 1로 최적 폭·변위 이해.
2. **3절 장비**: 그림 2의 광학 경로 추적; 그림 3의 각 요소(LCVR → 편광자 → calcite → Wollaston) 역할.
3. **4절 보정**: 식 (2) S_meas = R·S_input과 23 파라미터 비선형 피팅; 식 (5)-(8)의 허용오차.
4. **5절 샘플 데이터**: 그림 7-10 — intensity, LOS 속도, POS B 방위각, LOS B 세기, 2.4시간 3 G 광자잡음 확인.
5. Alfvén 파 발견은 Tomczyk+ 2007 Science 참조.

## 7. Q&A / 질문과 답변

### Q1. Why near-IR forbidden lines rather than visible? / 왜 근적외 금지선?

**EN:** The Zeeman shift scales as g·λ², so moving from 500 nm to 1074 nm boosts the V/I signal by (1074/500)² ≈ 4.6× for the same field. Additionally, the Fe XIII 1074.7 line forms in the million-degree corona (1.6 MK), is a magnetic dipole (M1) forbidden line with g_eff = 1.5, and the Hanle effect becomes saturated in this line for B ≳ 10 G, so Q/U measurements cleanly constrain the POS field direction independent of field strength. Near-IR detectors (HgCdTe) were reaching maturity (Lin, Penn & Tomczyk 2000).

**KR:** Zeeman 시프트는 g·λ²에 비례하므로 500 nm에서 1074 nm로 이동하면 같은 B에 대해 V/I 신호가 (1074/500)² ≈ 4.6배 커진다. 또한 Fe XIII 1074.7 nm는 100만 도 코로나(1.6 MK)에서 형성되는 자기 쌍극자(M1) 금지선으로 g_eff = 1.5이며, B ≳ 10 G에서 Hanle 효과가 포화되어 Q/U가 자기장 세기와 무관하게 POS 방향만 제약한다. 근적외 HgCdTe 검출기 기술 성숙도 상승도 결정적이었다.

### Q2. What's behind the S/N optimization in Eq. (1)? / 식 (1) S/N 최적화의 의미?

**EN:** Babcock (1953)'s formulation: for a constant-intensity spectral line, the Stokes V signal is proportional to the first derivative of the line profile, peaking in the wings. Integrating V(λ)·F(λ,Δλ,d)dλ over the filter bandpass F, and dividing by √[(I + B)·F dλ] (photon noise), gives Eq. (1). The maximum S/N lies at Δλ ≈ 0.13-0.16 nm FWHM and d ≈ 0.09-0.13 nm displacement from line center. CoMP chose 0.13 nm as a compromise for ground observations where sky background is ~10× line intensity.

**KR:** Babcock (1953) 식: 일정 강도의 선에 대해 Stokes V는 선 프로파일의 1차 도함수에 비례하며 선 날개(wing)에서 최대이다. 필터 통과대역 F에 대해 V(λ)F dλ를 적분하고 광자잡음 √((I+B)F dλ)로 나누면 식 (1)이 된다. 최대 S/N은 Δλ ≈ 0.13–0.16 nm FWHM, 선 중심 변위 d ≈ 0.09–0.13 nm에 있다. CoMP는 지상 관측 배경(~10×)을 고려해 0.13 nm를 선택했다.

### Q3. What is the Wollaston beamsplitter doing? / Wollaston 빔스플리터의 역할?

**EN:** Instead of the exit polarizer of a classical Lyot filter, CoMP uses a **calcite Wollaston prism with 15° cut angle** that simultaneously passes orthogonal polarization components into two spatially-separated images on the HgCdTe detector. This gives simultaneous (I+V, I−V) or (I+Q, I−Q) or (I+U, I−U) images eliminating flat-field and transparency artifacts through differencing. Combined with the 6 LCVRs (2 for input analysis, 4 interleaved between calcite stages), arbitrary polarization states can be selected.

**KR:** 전통적 Lyot 필터의 exit 편광자 대신 CoMP는 **15° 절단각 calcite Wollaston 프리즘**을 써서 직교 편광 성분을 공간적으로 분리하여 HgCdTe 검출기 위에 동시에 결상한다. 따라서 (I+V, I−V), (I+Q, I−Q), (I+U, I−U) 쌍을 동시에 얻어 차분으로 flat-field·투명도 오차를 제거한다. 6개 LCVR(입력 분석용 2개, calcite 단 사이 4개)과 조합해 임의 편광 상태를 선택한다.

### Q4. Why 35°C instrument and 30°C filter? / 35°C 챔버, 30°C 필터?

**EN:** Calcite birefringence has a strong temperature dependence: the filter bandpass shifts by **−0.056 nm/°C** (equivalent to −15.6 km/s/°C in Doppler). A nested PID loop stabilizes the whole instrument at 35°C (±1°C) and the filter housing at 30°C (<5 mK over 24 h). Since CoMP aims to measure velocities to ~km/s, the filter temperature must be controlled to better than ~mK.

**KR:** Calcite 복굴절은 강한 온도 의존성을 보여 필터 통과대역이 **−0.056 nm/°C**(도플러 환산 −15.6 km/s/°C) 시프트한다. 이중 PID 제어 루프로 장비 전체를 35°C (±1°C), 필터 하우징을 30°C(24시간 <5 mK)로 유지한다. km/s 수준 속도 측정을 위해 필터 온도는 mK 수준 안정도가 필요하다.

### Q5. How stringent is the response matrix R requirement? / R 행렬 요구 정확도?

**EN:** Given expected coronal S_corona = [1, 0.1, 0.1, 10⁻³]·I and desired noise σ_S = [10⁻³, 10⁻³, 10⁻³, 10⁻⁴]·I, the acceptable R errors are σ_R ≈ 10⁻² off-diagonal for I,Q,U rows but **10⁻⁴ for the bottom row** (I→V crosstalk). Monte Carlo with 1000 trials gives measured σ_R ≈ 10⁻³ for most elements but ~5-8×10⁻⁴ for the V row — close to but not below the 10⁻⁴ requirement. The remedy: exploit V antisymmetry vs. Q,U symmetry around line center to empirically subtract residual crosstalk (Lin, Kuhn & Coulter 2004).

**KR:** 예상 코로나 신호 S_corona = [1, 0.1, 0.1, 10⁻³]·I, 원하는 잡음 σ_S = [10⁻³, 10⁻³, 10⁻³, 10⁻⁴]·I에 대해, R 오차 허용치는 I·Q·U 행은 10⁻² 비대각까지, 그러나 **하단 V 행은 10⁻⁴**가 필요하다. 1000회 몬테카를로 결과 대부분 10⁻³이지만 V 행은 5-8×10⁻⁴로 요구치 근처. 해결책: V는 선 중심 기준 반대칭, Q·U는 대칭이라는 성질을 이용해 잔여 crosstalk을 경험적으로 빼준다(Lin, Kuhn & Coulter 2004).

### Q6. What does the 2005 Oct 31 data demonstrate? / 2005년 10월 31일 샘플의 의미?

**EN:** 2.4 hours of integration (146 circular-polarization image groups, 37 linear; 15 s photon collection per group; 29 s cadence; 52% duty cycle) over Fe XIII 1074.7 nm on the east limb. Figure 8: intensity, LOS velocity (±6 km/s), POS B azimuth vectors tracing coronal loops, and LOS B_LOS with bipolar structure (±40 G). Figure 10: measured scatter gives σ_B = 3.5 G vs. photon-noise predicted 3.2 G — **CoMP is essentially photon-noise-limited**. Figure 9: linear polarization fraction climbs from 2% at 1.05 R_sun to 7% at 1.3 R_sun (normal Hanle behavior).

**KR:** 2.4시간 적분(원편광 146 그룹, 선형 37 그룹; 그룹당 광자 수집 15 s, 주기 29 s, 52% 듀티) Fe XIII 1074.7 nm 동쪽 림. 그림 8: intensity, LOS 속도(±6 km/s), 코로나 루프를 따르는 POS B 방위 벡터, 그리고 ±40 G 범위 쌍극성 B_LOS. 그림 10: 측정 산포 3.5 G vs 광자잡음 예측 3.2 G — **CoMP는 사실상 광자잡음 한계에 도달**. 그림 9: 선형편광 비율 1.05 R_sun에서 2%, 1.3 R_sun에서 7%로 정상 Hanle 증가.

### Q7. Connection to the Alfvén wave discovery? / Alfvén 파 발견과의 연결?

**EN:** Tomczyk et al. (2007, Science 317, 1192) used this very instrument to detect ubiquitous transverse (Alfvénic) oscillations in the corona with amplitudes ~0.3 km/s, periods ~5 min, propagating at ~2000 km/s — the first direct detection of coronal Alfvén waves. This was enabled by CoMP's unique combination of wide FOV (2.8 R_sun), fast cadence (~30 s), and ~km/s Doppler precision in Fe XIII 1074.7 nm. The 2008 paper formalizes the instrument that made that discovery possible.

**KR:** Tomczyk et al. (2007, Science 317, 1192)는 바로 이 장비로 코로나 전역에 걸친 횡(Alfvén형) 진동을 최초로 검출했다: 진폭 ~0.3 km/s, 주기 ~5 분, 전파속도 ~2000 km/s. 이는 CoMP의 넓은 FOV(2.8 R_sun), 빠른 주기(~30 s), Fe XIII 1074.7 nm에서 km/s급 도플러 정밀도의 조합으로 가능했다. 2008년 논문은 그 발견을 가능케 한 장비를 정식 발표한 것이다.

## Ready to Read / 읽기 준비 완료

**EN:** With the Stokes formalism, Babcock S/N, Lyot filter theory, and calibration matrix concepts reviewed, we are ready to walk through the CoMP design and appreciate how near-IR polarimetry opened routine coronal magnetography.

**KR:** Stokes 형식, Babcock S/N, Lyot 필터 이론, 보정 행렬 개념을 확인했으므로, CoMP 설계를 따라가며 근적외 편광측정이 어떻게 상시적 코로나 자기 관측을 가능케 했는지 감상할 준비가 완료되었다.
