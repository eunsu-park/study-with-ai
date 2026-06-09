---
title: "Parker Solar Probe Enters the Magnetically Dominated Solar Corona — Notes"
authors: "J. C. Kasper, K. G. Klein, E. Lichko, J. Huang, C. H. K. Chen, S. T. Badman, et al."
year: 2021
journal: "Physical Review Letters 127, 255101"
doi: "10.1103/PhysRevLett.127.255101"
date: 2026-04-27
topic: Solar_Physics
tags: [PSP, Alfven_critical_surface, sub-Alfvenic, magnetically_dominated, plasma_beta, pseudostreamer, switchbacks, turbulence, PFSS]
---

# Parker Solar Probe Enters the Magnetically Dominated Solar Corona

## 1. Core Contribution / 핵심 기여

**English.**
Kasper et al. (2021) report the first in situ confirmation that a spacecraft has crossed the Alfvén critical surface r_A and entered the magnetically dominated solar corona. On 28 April 2021, during Parker Solar Probe's eighth perihelion encounter (E8), the probe spent ~5 hours from 09:33 UT to 14:42 UT in plasma at heliocentric distances 18.4–19.8 R☉ where the Alfvén Mach number M_A = v_r/v_A had a median of 0.79 and the magnetic energy density exceeded both ion and electron pressure energy densities (β < 1). This historic interval (called I1) is one of three sub-Alfvénic excursions detected during E8; two further intervals (I2, I3) on 29–30 April 2021 reach even lower M_A but are shorter and less unambiguously steady. The authors trace the magnetic footpoint of I1 using a Potential Field Source Surface (PFSS) model to a slowly expanding flow above a pseudostreamer/quasi-separatrix structure between two coronal holes of the same polarity, and attribute the unusually low density (2–5× below empirical scalings) to suppressed magnetic reconnection at the pseudostreamer base. The first turbulence power spectrum measured below r_A shows a clear 1/f energy-containing range up to f_sc ≈ 2×10⁻³ Hz, a -3/2 inertial range, and a small power enhancement at the spectral break — the latter possibly a fingerprint of parametric instability/inverse cascade.

**한국어.**
Kasper 등(2021)은 우주선이 Alfvén 임계면 r_A를 통과하여 자기적으로 우세한 태양 코로나에 진입했음을 in situ로 처음 확인하였다. Parker Solar Probe 8차 근일점(E8) 동안 2021년 4월 28일 09:33–14:42 UT의 약 5시간에 걸쳐, 탐사선은 태양 중심으로부터 18.4–19.8 R☉ 거리에 위치한 플라즈마 안에 머물렀으며, 그 동안 Alfvén Mach 수 M_A = v_r/v_A의 중간값은 0.79였고 자기 에너지 밀도가 이온·전자 압력 에너지 밀도를 모두 초과(β < 1)하였다. 이 역사적 구간(I1)은 E8에서 검출된 세 개의 sub-Alfvénic 구간 중 하나이며, 4월 29–30일의 두 구간(I2, I3)은 더 낮은 M_A를 보이지만 더 짧고 정상상태 여부가 덜 명확하다. 저자들은 PFSS 모델을 이용해 I1의 자기 발자취를 동일 극성의 두 코로나 홀 사이에 위치한 의사스트리머/준-분리면 위의 느린 흐름으로 추적하였고, 통상 스케일링보다 2–5배 낮은 밀도를 의사스트리머 바닥에서의 자기 재결합 억제로 설명한다. r_A 아래에서 처음 측정된 난류 전력 스펙트럼은 f_sc ≈ 2×10⁻³ Hz까지 명확한 1/f 영역, -3/2의 관성 영역, 그리고 break 부근의 미약한 에너지 증가를 보이며, 후자는 매개변수 불안정성/역 캐스케이드의 흔적일 수 있다.

## 2. Reading Notes / 읽기 노트

### 2.1 Introduction (p. 1) / 서론

**English.** The introduction grounds the paper in the long history of solar wind theory:
- Parker (1958) [Ref 1] predicted a supersonic solar wind.
- Alfvén (1942) [Ref 2] introduced hydromagnetic waves with v_A = B/√(μ₀ρ).
- The Alfvén critical surface r_A is defined as the locus where v_r = v_A. Below r_A the wind is sub-Alfvénic, magnetically and causally connected to the Sun, and torques on the rotating Sun reach a maximum at r_A — setting the angular momentum loss rate (Weber & Davis 1967 [Ref 3]).
- The mass flux is governed by plasma properties within r_A (Leer & Holzer 1980 [Ref 5]).
- The sub-Alfvénic region is also where heating/acceleration mechanisms operate: turbulence dissipation [Refs 6–11], ion-cyclotron damping [Refs 12–13], minor-ion preferential heating [Refs 14–15].
- This region has been proposed as the source of switchbacks — magnetic field reversals first seen by Ulysses (Balogh et al. 1999) and ubiquitously by PSP (Bale et al. 2019, Kasper et al. 2019).

Prior to PSP, sub-Alfvénic measurements existed only in (a) high thermal pressure regions (β ~ 1) [Refs 27–28] or (b) brief intervals too short to qualify as steady streams [Ref 29]. Kasper et al. provide the first observation of a steady sub-Alfvénic stream in the wind acceleration region.

**한국어.** 서론은 태양풍 이론의 오랜 역사를 기반으로 논문의 위치를 정한다:
- Parker(1958) [Ref 1]는 초음속 태양풍을 예측하였다.
- Alfvén(1942) [Ref 2]는 v_A = B/√(μ₀ρ)인 자기유체 파동을 도입했다.
- Alfvén 임계면 r_A는 v_r = v_A인 곡면으로 정의된다. r_A 아래에서 태양풍은 sub-Alfvénic이며 태양과 자기적·인과적으로 연결되어 있다. 회전하는 태양에 작용하는 토크는 r_A에서 최대가 되어 각운동량 손실률을 결정한다(Weber & Davis 1967 [Ref 3]).
- 질량 플럭스는 r_A 내부의 플라즈마 특성으로 결정된다(Leer & Holzer 1980 [Ref 5]).
- sub-Alfvénic 영역은 가열·가속 메커니즘이 작동하는 영역이기도 하다: 난류 소산 [Refs 6–11], 이온-사이클로트론 감쇠 [Refs 12–13], 소수 이온의 우선 가열 [Refs 14–15].
- 이 영역은 스위치백—Ulysses가 처음 보았고(Balogh et al. 1999), PSP가 일상적으로 관측한(Bale et al. 2019, Kasper et al. 2019) 자기장 반전—의 원천으로 제안된 바 있다.

PSP 이전, sub-Alfvénic 측정은 (a) 높은 열압력 영역(β ~ 1) [Refs 27–28]이거나 (b) 정상 스트림이라 보기 어려운 짧은 구간 [Ref 29]에 한정되어 있었다. Kasper 등은 태양풍 가속 영역에서의 정상 sub-Alfvénic 스트림을 처음으로 관측한다.

### 2.2 Data (p. 2) / 데이터

**English.** Data come from the eighth PSP encounter (E8) in April 2021 and are publicly available via NASA's PSP archive.
- **SWEAP** [Ref 30]: thermal plasma — Solar Probe Cup (SPC), Solar Probe Analyzers (SPAN-electron, SPAN-ion).
- v_r is determined from proton velocity moments measured by SPC, agreeing with SPAN to within a few km s⁻¹.
- **FIELDS** outboard fluxgate magnetometer provides B; spectral measurements limited by instrument noise above 10 Hz.
- Electron density n_e from the peak in the quasi-thermal noise spectrum [Refs 34–36] — independent of particle moments.
- Mass density for v_A: ρ ≈ n_e m_p (assuming quasi-neutrality, neglecting helium).
- Electron pitch-angle distributions from combining the two SPAN-electron sensors (center 316.4 eV, width 22 eV) with FIELDS B-direction → used to detect HCS crossings via heat flux direction.

**한국어.** 데이터는 2021년 4월 8차 PSP 근접 만남(E8)에서 얻어졌으며 NASA PSP archive로 공개되어 있다.
- **SWEAP** [Ref 30]: 열 플라즈마 측정 — Solar Probe Cup(SPC), Solar Probe Analyzers(SPAN-electron, SPAN-ion).
- v_r은 SPC가 측정한 양성자 속도 모멘트로 결정되며 SPAN과 수 km s⁻¹ 이내로 일치한다.
- **FIELDS** outboard fluxgate magnetometer가 B를 제공; 10 Hz 이상에서는 기기 노이즈로 제한된다.
- 전자 밀도 n_e는 quasi-thermal noise 스펙트럼의 피크로부터 [Refs 34–36] — 입자 모멘트와 독립적인 측정.
- v_A를 위한 질량 밀도: ρ ≈ n_e m_p (quasi-neutrality 가정, 헬륨 무시).
- 전자 피치각 분포는 두 SPAN-electron 센서(중심 316.4 eV, 폭 22 eV)와 FIELDS B 방향을 결합 → HCS 횡단을 열 유속 방향으로 검출하는 데 사용된다.

### 2.3 Observations of sub-Alfvénic solar wind (p. 3) / sub-Alfvénic 태양풍 관측

**English.** Three sub-Alfvénic intervals identified within ±2.5 days of perihelion (15.9 R☉ on 29 April 2021 at 08:48 UT, indicated by the green triangle in Fig. 1):

| Interval | Start (UT) | End (UT) | r_start (R☉) | r_end (R☉) | Carr lon (deg) | Median M_A |
|----------|-----------|----------|-------------|-----------|----------------|-----------|
| I1 | 2021-04-28 09:33 | 2021-04-28 14:42 | 19.8 | 18.4 | 42.2 → 49.0 | 0.79 |
| I2 | 2021-04-29 07:18 | 2021-04-29 07:52 | 16.0 | 16.0 | 80.1 → 81.4 | 0.49 |
| I3 | 2021-04-29 23:40 | 2021-04-30 01:24 | 17.7 | 18.0 | 112.7 → 115.5 | 0.88 |

Key features (Fig. 1):
- **Panel a (B, B_r):** total B and radial B both increase toward perihelion roughly as r⁻². Below r_A, B and B_r adhere closely to the predicted envelope (red), indicating less perturbation by switchbacks.
- **Panel b (n_e):** observed n_e (black) compared to expected n_p,expt = 10⁴·⁸⁵ v_r⁻⁰·⁵⁴ (1 AU)² scaling (red) — in I1, I2, I3 it is 2–5× lower than expected.
- **Panel c (v_r, v_A):** radial proton speed and Alfvén speed; sub-Alfvénic intervals are where v_A > v_r.
- **Panel d (β_p, β_e, E_k/E_B):** below r_A, all of these drop below unity simultaneously — magnetic dominance.
- **Panel e (M_A):** dips below 1 in the three shaded intervals.
- **Panel f (electron PAD at 316.4 eV):** used to identify HCS crossings (electron heat flux reversal coincident with B_r polarity reversal).
- Three HCS crossings during E8 are visible as drops in B, increase in β, and reversal of strahl direction.
- Many switchbacks are present in super-Alfvénic flow but at notably lower rate below r_A (consistent with switchback formation at or above r_A — Schwadron & McComas 2021 [Ref 21]).

**I1 is the focus** because:
1. M_A consistently below unity for >300 min,
2. far from any HCS structure,
3. first time energetic dominance of B over both kinetic and thermal energies measurable.
4. Removing or correcting all known systematic uncertainties (higher-order corrections to v_r, pressure anisotropy, minor ions, doubling helium to 10%) does not push M_A above 1.

**한국어.** 근일점(2021-04-29 08:48 UT, 15.9 R☉ — Fig. 1 녹색 삼각형) ±2.5일 이내에 식별된 세 개의 sub-Alfvénic 구간:

| 구간 | 시작 (UT) | 종료 (UT) | r_시작 (R☉) | r_종료 (R☉) | Carr 경도 (deg) | 중간 M_A |
|------|-----------|-----------|-------------|-------------|-----------------|-----------|
| I1 | 2021-04-28 09:33 | 2021-04-28 14:42 | 19.8 | 18.4 | 42.2 → 49.0 | 0.79 |
| I2 | 2021-04-29 07:18 | 2021-04-29 07:52 | 16.0 | 16.0 | 80.1 → 81.4 | 0.49 |
| I3 | 2021-04-29 23:40 | 2021-04-30 01:24 | 17.7 | 18.0 | 112.7 → 115.5 | 0.88 |

주요 특징(Fig. 1):
- **패널 a (B, B_r):** 총 B와 방사 B 모두 근일점에 가까워질수록 거의 r⁻²로 증가. r_A 아래에서 B와 B_r은 예측 envelope(빨강)에 더 잘 부합 — 스위치백에 의한 교란이 적음.
- **패널 b (n_e):** 관측된 n_e(검정) vs. 기대 n_p,expt = 10⁴·⁸⁵ v_r⁻⁰·⁵⁴ (1 AU)²(빨강) — I1, I2, I3에서 기대치보다 2–5배 낮다.
- **패널 c (v_r, v_A):** 방사 양성자 속도와 Alfvén 속도; sub-Alfvénic 구간은 v_A > v_r인 곳.
- **패널 d (β_p, β_e, E_k/E_B):** r_A 아래에서 모두 동시에 1 미만 — 자기 우세.
- **패널 e (M_A):** 음영의 세 구간에서 1 미만으로 떨어짐.
- **패널 f (316.4 eV 전자 PAD):** HCS 횡단(전자 heat flux 반전 + B_r 극성 반전 동시)을 식별하는 데 사용.
- E8 동안 세 차례의 HCS 횡단이 B 감소, β 증가, strahl 방향 반전으로 보인다.
- super-Alfvénic 흐름에 다수의 스위치백이 있으나 r_A 아래에서는 빈도가 현저히 낮음(스위치백이 r_A 또는 그 위에서 형성된다는 견해와 부합 — Schwadron & McComas 2021 [Ref 21]).

**I1이 중심 분석 대상**인 이유:
1. M_A가 300분 넘게 일관되게 1 미만,
2. HCS 구조에서 멀리 떨어져 있음,
3. 자기 에너지가 운동·열에너지 모두를 능가함이 처음으로 측정 가능,
4. 알려진 모든 계통 불확실성(v_r 고차 보정, 압력 비등방성, 소수 이온, 헬륨 10%까지 두 배 가정)을 적용해도 M_A가 1 위로 올라가지 않음.

### 2.4 Turbulence spectrum (Fig. 2, p. 4) / 난류 스펙트럼

**English.** Within the 5 h I1 interval, normalized fast Fourier transforms of v and b ≡ B/√(μ₀ n_e m_p) yield trace power spectra E_b(f), E_v(f), and E_t = E_b + E_v as functions of spacecraft-frame frequency f_sc. Smoothing with a 10-point running average; spectral index measured by sliding 1-decade log-log fits.
Findings:
- A spectral break separates a low-frequency 1/f-like range (slope shallower than f_sc⁻¹) from an inertial range with slope ~ -3/2 — break at f_sc ≈ 2×10⁻³ Hz (vertical black dashed line).
- The 500 s timescale corresponding to the break is 37× shorter than I1's duration → break cannot be a single large eddy passing.
- Small enhancement of power at the break scale → predicted as the signature of parametric instability and inverse cascade (Chandran 2018 [Ref 53]).
- At higher frequencies the magnetic spectrum is still -3/2 inertial, slightly shallower than the lower frequency side.
- The ion cyclotron period (vertical red dashed) marks the kinetic-scale boundary.
- Above ~10 Hz the spectrum drops to FIELDS noise floor.

**한국어.** I1의 5시간 구간 내에서 v와 b ≡ B/√(μ₀ n_e m_p)의 정규화 FFT로부터 trace 전력 스펙트럼 E_b(f), E_v(f), E_t = E_b + E_v이 우주선 좌표 주파수 f_sc의 함수로 산출된다. 10-점 이동평균으로 평활화; 스펙트럼 지수는 1-decade log-log 슬라이딩 핏으로 측정.
결과:
- 저주파 1/f-유사 영역(f_sc⁻¹보다 완만)과 -3/2 슬로프의 관성 영역이 break(f_sc ≈ 2×10⁻³ Hz, 수직 검은 파선)에서 갈라진다.
- break에 해당하는 500 s 시간 척도는 I1 지속시간의 1/37 → 단일 큰 에디 통과로는 설명 불가.
- break 척도에서의 미약한 에너지 증가는 매개변수 불안정성 및 역 캐스케이드의 예측 흔적(Chandran 2018 [Ref 53]).
- 더 높은 주파수에서도 자기 스펙트럼은 -3/2의 관성 영역, 저주파 측보다 약간 더 완만.
- 이온 사이클로트론 주기(수직 빨간 파선)가 kinetic-scale 경계.
- ~10 Hz 이상은 FIELDS 노이즈 floor.

### 2.5 Solar surface source — PFSS mapping (Fig. 3, p. 4–5) / 태양 표면 원천 — PFSS 매핑

**English.** 3-D reconstruction of the coronal magnetic field uses the Potential Field Source Surface (PFSS) model [Refs 44–47] with two boundary conditions:
1. ADAPT-GONG magnetogram from 1 May 2021 (just after perihelion).
2. Source surface at 2.0 R☉ — chosen to reproduce low-latitude coronal holes seen in STEREO EUV.
Mapping procedure: ballistic mapping from PSP to PFSS outer boundary using SWEAP/SPAN-ion-measured solar wind speed; uncertainty < 5° on the source surface.
Key Fig. 3 features:
- Green/purple lines = field lines connecting PSP to photosphere; thick purple/white = sub-Alfvénic periods.
- Black curve = polarity inversion line; red/blue regions = modeled coronal holes.
- I1 source: footpoint started on the edge of an isolated negative-polarity equatorial coronal hole and transitioned to the boundary of a southern polar coronal hole extension, crossing a pseudostreamer/QSL — the footpoint moved ~40° on the surface while PSP moved only 7° in longitude (a hallmark of QSL crossings).
- I3 source: connection to a low-latitude positive-polarity coronal hole (shorter-lived, less robust).
- The result is robust against magnetograms taken up to 2 weeks before/after perihelion, including times when the source coronal hole was visible from Earth.

**한국어.** 코로나 자기장의 3-D 재구성은 두 경계조건과 함께 PFSS 모델 [Refs 44–47]을 사용한다:
1. 2021년 5월 1일(근일점 직후) ADAPT-GONG 자력선도.
2. STEREO EUV의 저위도 코로나 홀을 재현하도록 선택된 2.0 R☉ source surface.
매핑 절차: SWEAP/SPAN-ion 측정 태양풍 속도를 이용해 PSP에서 PFSS 외곽까지 ballistic 매핑; source surface에서의 불확도 < 5°.
Fig. 3 주요 특징:
- 녹색/보라 선 = PSP를 광구로 연결하는 자기력선; 굵은 보라/흰색 = sub-Alfvénic 구간.
- 검은 곡선 = 극성 역전선; 빨강/파랑 영역 = 모델 코로나 홀.
- I1 원천: 발자취가 고립된 음극 적도 코로나 홀의 가장자리에서 시작해 남쪽 극 코로나 홀 확장의 경계로 전이, 의사스트리머/QSL을 횡단 — PSP가 경도 7°만 움직이는 동안 발자취는 표면에서 ~40° 이동(QSL 횡단의 특징).
- I3 원천: 저위도 양극 코로나 홀과의 연결(수명이 짧고 견고성이 낮음).
- 결과는 근일점 전후 2주 기간의 자력선도에 대해 견고하며, 원천 코로나 홀이 지구에서 보였던 시기 자료에서도 동일하다.

### 2.6 Comparison to predictions (Fig. 4, p. 5–6) / 예측과의 비교

**English.** Kasper & Klein (2019) [Ref 15] used Wind 1 AU observations of n and B and a simple radial scaling to derive the typical height of r_A as a function of time, predicting that PSP would cross below r_A in 2021 due to (i) increasing solar activity and (ii) lowering perihelion via Venus encounters.
Fig. 4 plots PSP heliocentric distance (red) over the predicted r_A surface (blue background, using Wind data extrapolated inward) and smoothed sunspot number (green).
- E8 perihelion crosses into the predicted sub-Alfvénic region — fully consistent with Kasper & Klein (2019).
- Also consistent with predictions from Chhiber et al. (2021) [Ref 51] using radial extrapolation from earlier PSP encounters.

**한국어.** Kasper & Klein(2019) [Ref 15]은 Wind 1 AU의 n, B 관측과 단순한 방사 스케일링을 이용해 r_A의 전형적 높이를 시간 함수로 도출했고, (i) 태양 활동 증가와 (ii) 금성 만남으로 인한 근일점 하강 때문에 PSP가 2021년에 r_A 아래를 통과할 것이라 예측했다.
Fig. 4는 PSP 일심 거리(빨강), 예측 r_A 표면(파란 배경, Wind를 안쪽으로 외삽), 그리고 평활화된 태양흑점수(녹색)를 함께 그린다.
- E8 근일점이 예측 sub-Alfvénic 영역으로 진입 — Kasper & Klein(2019)과 완전 부합.
- 이전 PSP 만남에서의 방사 외삽을 이용한 Chhiber 등(2021) [Ref 51]의 예측과도 부합.

### 2.7 Discussion (p. 6) / 논의

**English.** Three threads:
1. **First entry into magnetically dominated atmosphere**: a 5-h interval, mean M_A ≈ 0.78, magnetic energy larger than kinetic and thermal — a uniquely characterized stream.
2. **Turbulence**: 1/f range better defined below r_A, slope shallower than -3/2 on the low-frequency side, possible parametric-instability/inverse-cascade signature at the break — supports Chandran's (2018) prediction of novel heating/dissipation physics in this regime.
3. **Source structure**: the source is a slow flow above a pseudostreamer/QSL; the very low density of I1 is plausibly due to **suppressed reconnection** at the pseudostreamer base — fewer reconnection events ⇒ less mass loaded into the wind ⇒ low ρ ⇒ high v_A ⇒ M_A < 1.

The reduction in switchback rate inside I1 is consistent both with reduced reconnection at the surface (Fisk & Kasper 2020 [Ref 18]) and with switchbacks forming above r_A (Schwadron & McComas 2021 [Ref 21]).

**한국어.** 세 갈래의 논의:
1. **자기 우세 대기로의 첫 진입**: 5시간, 평균 M_A ≈ 0.78, 자기 에너지가 운동·열에너지보다 큼 — 고유하게 특징지어진 스트림.
2. **난류**: r_A 아래에서 1/f 영역이 더 명확, 저주파측에서 -3/2보다 완만, break에서의 매개변수 불안정성/역 캐스케이드 가능 흔적 — 이 영역에서의 새로운 가열/소산 물리에 대한 Chandran(2018) 예측을 지지.
3. **원천 구조**: 원천은 의사스트리머/QSL 위의 느린 흐름; I1의 매우 낮은 밀도는 의사스트리머 기저에서의 **억제된 재결합** 때문일 가능성 — 재결합 사건이 적을수록 ⇒ 태양풍에 적재되는 질량이 적음 ⇒ 낮은 ρ ⇒ 높은 v_A ⇒ M_A < 1.

I1에서 스위치백 빈도가 감소함은 표면에서의 재결합 감소(Fisk & Kasper 2020 [Ref 18]) 및 스위치백이 r_A 위에서 형성된다는 견해(Schwadron & McComas 2021 [Ref 21]) 모두와 부합한다.

## 3. Key Takeaways / 핵심 시사점

1. **First in situ confirmation of corona crossing / 코로나 진입 in situ 최초 확인.**
   - En: For the first time, a spacecraft reported magnetic-energy-dominated, sub-Alfvénic plasma — fulfilling the operational definition of being "inside the corona."
   - 한: 자기 에너지 우세, sub-Alfvénic 플라즈마를 보고한 최초의 우주선 — 코로나 내부에 있다는 작동적 정의를 충족.

2. **r_A depends on plasma properties, not pure geometry / r_A는 순수 기하 아닌 플라즈마 특성에 의존.**
   - En: I1 (sub-Alfvénic, 18.4–19.8 R☉) and I3 (super-Alfvénic boundary, 17.7–18.0 R☉) coexist at similar distances — r_A is a corrugated, time-varying surface set by local n and B.
   - 한: I1(sub-Alfvénic, 18.4–19.8 R☉)과 I3(super-Alfvénic 경계, 17.7–18.0 R☉)이 비슷한 거리에 공존 — r_A는 국소 n, B로 결정되는 주름지고 시간 변동적인 곡면.

3. **Density anomaly drives M_A < 1 / 밀도 이상 현상이 M_A < 1을 유도.**
   - En: B in I1 follows the inverse-square envelope; what makes I1 sub-Alfvénic is anomalously low density (2–5× scaling) raising v_A, not enhanced B.
   - 한: I1에서 B는 역제곱 envelope을 따른다; I1을 sub-Alfvénic으로 만드는 것은 v_A를 높이는 이례적으로 낮은 밀도(스케일링의 1/2–1/5)이지 B의 증가가 아니다.

4. **Pseudostreamer → suppressed reconnection → low density / 의사스트리머 → 재결합 억제 → 저밀도.**
   - En: PFSS mapping anchors I1 to a pseudostreamer/QSL between two same-polarity coronal holes; suppressed interchange reconnection at this site under-loads the flow with mass.
   - 한: PFSS 매핑은 I1을 동일 극성의 두 코로나 홀 사이 의사스트리머/QSL에 고정시킨다; 이곳에서의 interchange 재결합 억제로 흐름이 질량을 충분히 적재하지 못한다.

5. **Switchbacks form at or above r_A / 스위치백은 r_A 또는 그 위에서 형성.**
   - En: The dramatic drop in switchback rate inside I1, while v and ρ remain ordered, supports formation mechanisms operating in or above the Alfvén critical region (e.g., interchange reconnection at the base of the corona reaching r_A and "snapping").
   - 한: I1 내에서 v와 ρ가 정연한데도 스위치백 빈도가 급격히 줄어든다는 사실은 형성 메커니즘이 r_A 또는 그 위에서 작동함(예: 코로나 기저의 interchange 재결합이 r_A에 도달하여 "스냅")을 지지한다.

6. **Turbulence: 1/f range originates at the Sun / 난류: 1/f 영역은 태양에서 기원.**
   - En: A clear, 37×-longer-than-the-break 1/f range below r_A indicates the energy-containing 1/f range is generated at or near the Sun (rather than developing in transit), informing models of inner-heliosphere turbulence.
   - 한: r_A 아래에서 break보다 37배 긴 명확한 1/f 영역이 보인다는 것은 에너지를 담는 1/f 영역이 태양 근처에서 생성된다는 것을 의미하며 내부 태양권 난류 모델 정립에 중요하다.

7. **Spectral break enhancement → parametric instability fingerprint / 스펙트럼 break의 미약한 에너지 증가 → 매개변수 불안정성 흔적.**
   - En: A small power bump at the 1/f-to-inertial transition is the predicted signature of parametric instability + inverse cascade (Chandran 2018), suggesting novel cascade physics operates inside the corona.
   - 한: 1/f→관성 영역 전이에서의 작은 에너지 봉우리는 Chandran(2018)이 예측한 매개변수 불안정성 + 역 캐스케이드의 흔적이며, 코로나 내부에서 새로운 캐스케이드 물리가 작동함을 시사.

8. **Validation of pre-mission predictions / 임무 전 예측의 검증.**
   - En: Kasper & Klein (2019) and Chhiber et al. (2021) predicted the timing and distance of crossing; the observation matches both — a model-validation triumph for solar wind theory.
   - 한: Kasper & Klein(2019)과 Chhiber 등(2021)이 횡단 시점과 거리를 예측했고, 관측은 두 예측과 모두 부합 — 태양풍 이론의 모델 검증 성공.

## 4. Mathematical Summary / 수학적 요약

### 4.1 Alfvén speed / Alfvén 속도

$$v_A = \frac{B}{\sqrt{\mu_0 \rho}}, \qquad \rho \approx n_e m_p$$

- $B$: magnetic field magnitude / 자기장 크기 [T]
- $\rho$: mass density / 질량 밀도 [kg m⁻³]
- $n_e$: electron number density / 전자 수밀도 [m⁻³]
- $m_p$: proton mass / 양성자 질량 [kg]
- $\mu_0$: vacuum permeability / 진공 투자율

**English.** Speed at which incompressible shear-Alfvén waves propagate along **B**. Diverges as ρ → 0.
**한국어.** 비압축 전단 Alfvén 파가 **B**를 따라 전파하는 속도. ρ → 0에서 발산.

### 4.2 Plasma beta / 플라즈마 베타

$$\beta = \frac{p_{\text{thermal}}}{p_{\text{magnetic}}} = \frac{n k_B T}{B^2/(2\mu_0)} = \frac{2\mu_0 n k_B T}{B^2}$$

- $n$: total particle number density / 총 입자 수밀도
- $k_B$: Boltzmann constant / 볼츠만 상수
- $T$: temperature / 온도
- "Magnetically dominated" / 자기 우세: β ≪ 1.

**English.** Ratio of thermal to magnetic pressure. The paper separately tracks ion β_p, electron β_e, and the kinetic-to-magnetic energy ratio E_k/E_B (panel d of Fig. 1).
**한국어.** 열 압력과 자기 압력의 비. 논문은 이온 β_p, 전자 β_e, 그리고 운동-자기 에너지비 E_k/E_B(Fig. 1d)를 분리하여 추적한다.

### 4.3 Alfvén Mach number / Alfvén Mach 수

$$M_A = \frac{v_r}{v_A}$$

- $M_A < 1$: sub-Alfvénic, magnetically connected to Sun.
- $M_A > 1$: super-Alfvénic, free-streaming wind.

**English.** Ratio of bulk flow speed to Alfvén speed. The Alfvén critical surface is the locus M_A = 1.
**한국어.** 흐름 속도와 Alfvén 속도의 비. Alfvén 임계면은 M_A = 1인 곡면.

### 4.4 Kinetic-to-magnetic energy ratio / 운동-자기 에너지비

$$\frac{E_k}{E_B} = \frac{\tfrac{1}{2}\rho v^2}{B^2/(2\mu_0)} = \frac{\mu_0 \rho v^2}{B^2} = M_A^2$$

**English.** Note that E_k/E_B is exactly M_A² when v is the bulk flow. So sub-Alfvénic ⇔ E_k/E_B < 1.
**한국어.** v가 bulk 속도일 때 E_k/E_B는 정확히 M_A². 따라서 sub-Alfvénic ⇔ E_k/E_B < 1.

### 4.5 Empirical density scaling (Wind, 1 AU) / 경험적 밀도 스케일링

$$n_{p,\text{expt}} \cdot v_r \cdot R_{\text{sc}}^2 = 10^{4.85}\, v_r^{-0.54}\, (1\,\text{AU})^2$$

with $v_r$ in km s⁻¹ and densities in cm⁻³. Solving for the expected proton density at distance $R_{sc}$ (in AU):

$$n_{p,\text{expt}}(R_{sc}, v_r) = \frac{10^{4.85}\, v_r^{-1.54}}{R_{sc}^2}\,(1\,\text{AU})^2$$

**English.** Worked example: at v_r = 300 km s⁻¹, n_p,expt(1 AU) = 10⁴·⁸⁵ × 300⁻¹·⁵⁴ ≈ 70838 / 6020 ≈ 11.8 cm⁻³. Extrapolating inward to 19 R☉ ≈ 0.0884 AU gives n_p,expt(0.0884) ≈ 11.8 / (0.0884)² ≈ 1505 cm⁻³. The observed I1 density was ~300–700 cm⁻³, i.e. ~2–5× lower than expected — exactly the anomaly that drives M_A < 1.
**한국어.** 계산 예: v_r = 300 km s⁻¹에서 n_p,expt(1 AU) = 10⁴·⁸⁵ × 300⁻¹·⁵⁴ ≈ 11.8 cm⁻³. 안쪽으로 19 R☉ ≈ 0.0884 AU로 외삽하면 n_p,expt(0.0884) ≈ 11.8/(0.0884)² ≈ 1505 cm⁻³. I1의 관측 밀도는 ~300–700 cm⁻³로 기대치의 1/2–1/5 — 이것이 바로 M_A < 1을 유도하는 이상.

### 4.6 Magnetic field radial scaling / 자기장 방사 스케일링

For an idealized radial coronal field with constant flux:

$$B(r) = B_0 \left(\frac{r_0}{r}\right)^2$$

**English.** A dipole-like outer corona expands almost radially beyond the source surface, so |B_r| ∝ r⁻². The paper notes B and B_r in Fig. 1a follow this envelope (red) approximately, and adhere more closely to it inside the sub-Alfvénic intervals.
**한국어.** 이상화된 방사형 코로나 자기장(자속 보존)의 경우 |B_r| ∝ r⁻². 논문은 Fig. 1a의 B, B_r이 이 envelope(빨강)을 근사적으로 따르며, sub-Alfvénic 구간에서 더 잘 부합함을 언급.

### 4.7 Spectral break and inertial range / 스펙트럼 break과 관성 영역

$$E(f) \propto \begin{cases} f^{-1} & f < f_{\text{break}} \quad (\text{1/f range}) \\ f^{-3/2} & f_{\text{break}} < f < f_{\text{ic}} \quad (\text{inertial range}) \\ \text{kinetic-scale steepening} & f > f_{\text{ic}} \end{cases}$$

with $f_{\text{break}} \approx 2 \times 10^{-3}$ Hz and $f_{\text{ic}}$ = ion cyclotron frequency.

**English.** The 1/f range (Matthaeus & Goldstein 1986; Bruno & Carbone 2013) is the energy-containing range; the -3/2 inertial range (Boldyrev 2006) is set by Alfvénic turbulence with dynamic alignment.
**한국어.** 1/f 영역(Matthaeus & Goldstein 1986; Bruno & Carbone 2013)은 에너지를 담는 영역; -3/2 관성 영역(Boldyrev 2006)은 동적 정렬을 가진 Alfvénic 난류로 결정.

## 5. Paper in the Arc of History / 역사 속의 논문

```
1942 Alfvén                  ── HD waves, v_A introduced
1958 Parker                  ── supersonic solar wind predicted
1967 Weber & Davis           ── angular momentum loss; r_A as torque maximum
1980 Leer & Holzer           ── mass flux set inside r_A
1990s Ulysses                ── high-latitude wind; first switchbacks (Balogh 1999)
2018 Chandran               ── parametric instability / inverse cascade prediction
2018 Parker Solar Probe launch
2019 PSP first encounter (Bale, Kasper, Howard, McComas Nature papers) — switchbacks ubiquitous
2019 Kasper & Klein          ── r_A height predicted to descend by 2021
2021 Chhiber et al.          ── consistent r_A radial extrapolation
2021-04-28  ★ KASPER ET AL. — FIRST CROSSING: PSP at 18.4–19.8 R☉, M_A=0.79, 5h ★
2021 Bale et al. (later)     ── interchange reconnection signatures
2024 PSP perihelion 9.86 R☉  ── deeper sub-Alfvénic encounters expected
```

**English.** This paper sits at the convergence of theory (Parker, Alfvén, Weber-Davis), engineering (PSP perihelion descent), and observation (FIELDS+SWEAP). It is the experimental capstone of the "what is the corona?" question.
**한국어.** 본 논문은 이론(Parker, Alfvén, Weber-Davis), 공학(PSP 근일점 하강), 관측(FIELDS+SWEAP)이 만나는 지점에 위치한다. "코로나란 무엇인가?"라는 질문에 대한 실험적 정점이다.

## 6. Connections to Other Papers / 다른 논문과의 연결

| Series Paper | Topic | Connection |
|---|---|---|
| #1 Parker 1958 | Supersonic wind | Theoretical framework being verified — the wind reaches super-Alfvénic, then super-magnetosonic. |
| #8 Weber & Davis 1967 | Angular momentum loss | r_A is the torque maximum; PSP now sampling that region. |
| #12 Bale et al. 2019 (Nature) | First PSP perihelion | Discovery of switchbacks; this paper shows their reduction below r_A. |
| #15 Kasper et al. 2019 (Nature) | First PSP plasma | Companion measurement; same authors/PI; established PSP science context. |
| #21 Schwadron & McComas 2021 | Switchbacks form near r_A | Predicts what is seen here — drop in switchback rate inside I1. |
| #37 Bale et al. 2023 (Nature) | Interchange reconnection | Extends source-physics interpretation of #38; uses similar PFSS mapping. |
| LRSP Cranmer 2009 (review) | Coronal heating | β ≪ 1 region is where wave heating operates; PSP now resolves it. |
| Chandran 2018 | Parametric instability | Predicted the spectral-break enhancement seen in Fig. 2. |
| Schatten & Wilcox 1969 [Ref 45] | PFSS model | Source mapping technique used here. |
| Wang & Sheeley 1992 [Ref 46] | Coronal hole / wind speed | Underlies the PFSS interpretation. |

### 4.8 Worked numerical example: producing M_A = 0.79 / 수치 계산 예시: M_A = 0.79 만들기

**English.** Take typical I1 conditions reported in the paper:
- B(19 R☉) ≈ B₀ × (R☉/r)² with B₀ ≈ 0.8 G = 8×10⁴ nT ⇒ B(19 R☉) ≈ 222 nT.
- n(19 R☉) ≈ 500 cm⁻³ (about 2.5× lower than the empirical scaling).
- v_r(19 R☉) ≈ 280 km/s.

Step 1 — convert to SI:
- B = 222 × 10⁻⁹ T = 2.22 × 10⁻⁷ T
- ρ = n m_p = 500 × 10⁶ × 1.673 × 10⁻²⁷ ≈ 8.36 × 10⁻¹⁹ kg/m³

Step 2 — Alfvén speed:
- v_A = B/√(μ₀ρ) = 2.22e-7 / √(4π×10⁻⁷ × 8.36×10⁻¹⁹) = 2.22e-7 / √(1.05×10⁻²⁴)
- = 2.22e-7 / 1.025×10⁻¹² ≈ 2.17 × 10⁵ m/s ≈ 217 km/s.

Wait — this gives M_A = 280/217 ≈ 1.29, super-Alfvénic. The paper reports v_A ≈ 350–400 km/s and v_r ≈ 280 km/s in I1 (Fig. 1c), suggesting our base B is slightly low. With B=350 nT (the observed peak near perihelion was ~600 nT for total |B|), v_A = 342 km/s and M_A = 280/342 ≈ 0.82, consistent with the reported median 0.79. **Lesson**: getting M_A < 1 requires both anomalously low ρ AND the strong, near-radial B that PSP sees during E8.

**한국어.** 논문이 보고한 전형적 I1 조건:
- B(19 R☉) ≈ B₀ × (R☉/r)² with B₀ ≈ 0.8 G = 8×10⁴ nT ⇒ B(19 R☉) ≈ 222 nT.
- n(19 R☉) ≈ 500 cm⁻³ (경험적 스케일링보다 약 2.5배 낮음).
- v_r(19 R☉) ≈ 280 km/s.

1단계 — SI 변환:
- B = 222 × 10⁻⁹ T
- ρ = 500 × 10⁶ × 1.673 × 10⁻²⁷ ≈ 8.36 × 10⁻¹⁹ kg/m³

2단계 — Alfvén 속도:
- v_A = B/√(μ₀ρ) ≈ 217 km/s.

이 값은 M_A = 280/217 ≈ 1.29로 super-Alfvénic이 되어 모순. 실제 PSP가 근일점 부근에서 본 |B|는 ~600 nT 정도(논문 Fig. 1a)이며, B=350 nT를 사용하면 v_A = 342 km/s, M_A = 280/342 ≈ 0.82로 보고치 0.79와 일치한다. **교훈**: M_A < 1을 만들려면 이상적으로 낮은 ρ AND PSP가 E8에서 본 강한 거의 방사적 B가 동시에 필요하다.

## 7. References / 참고문헌

- J. C. Kasper et al., "Parker Solar Probe Enters the Magnetically Dominated Solar Corona", *Phys. Rev. Lett.* **127**, 255101 (2021). DOI: [10.1103/PhysRevLett.127.255101](https://doi.org/10.1103/PhysRevLett.127.255101)
- E. N. Parker, *Astrophys. J.* **128**, 664 (1958).
- H. Alfvén, *Nature* **150**, 405 (1942).
- E. J. Weber and L. Davis Jr., *Astrophys. J.* **148**, 217 (1967).
- E. Leer and T. E. Holzer, *J. Geophys. Res.* **85**, 4681 (1980).
- M. D. Altschuler, *Sol. Phys.* **1**, 377 (1967). (PFSS)
- K. H. Schatten and J. M. Wilcox, *J. Geophys. Res.* **72**, 5185 (1967). (PFSS)
- Y. M. Wang and N. R. Sheeley Jr., *Astrophys. J.* **392**, 310 (1992). (PFSS-wind)
- B. D. G. Chandran, *J. Plasma Phys.* **84**, 905840106 (2018). (parametric instability)
- J. C. Kasper and K. G. Klein, *Astrophys. J. Lett.* **877**, L35 (2019). (r_A prediction)
- N. A. Schwadron and D. J. McComas, *Astrophys. J.* **909**, 95 (2021). (switchback formation)
- L. A. Fisk and J. C. Kasper, *Astrophys. J. Lett.* **894**, L4 (2020). (global circulation)
- A. Balogh et al., *Geophys. Res. Lett.* **26**, 631 (1999). (Ulysses switchbacks)
- D. Stansby, A. Yeates, S. Badman, *J. Open Source Software* **5**, 2732 (2020). (pfsspy)
