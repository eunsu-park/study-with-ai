---
title: "The Fast Auroral SnapshoT (FAST) Mission"
authors: "Carlson, C. W., Pfaff, R. F., Watzin, J. G."
year: 1998
journal: "Geophysical Research Letters"
doi: "10.1029/98GL01592"
topic: Space_Weather
tags: [aurora, FAST_mission, parallel_electric_field, inverted_V, AKR, electron_holes, ion_beams, magnetosphere_ionosphere_coupling]
status: completed
date_started: 2026-04-25
date_completed: 2026-04-25
---

# 55. The Fast Auroral SnapshoT (FAST) Mission / FAST 오로라 스냅샷 임무

---

## 1. Core Contribution / 핵심 기여

**English**: This 1998 Geophysical Research Letters mission paper introduces NASA's Fast Auroral SnapshoT (FAST) satellite — the second Small Explorer (SMEX) satellite, launched on August 21, 1996 into a 350 × 4175 km, 83° inclination elliptical orbit. FAST was conceived as a single integrated scientific instrument: every sensor was synchronized to the spin-stabilized spacecraft and to a common Instrument Data Processing Unit (IDPU) that buffered up to 1 Gbit of "snapshot" data triggered when the spacecraft crossed an auroral arc. By aligning the spin plane with the geomagnetic field and reaching cadences of 1.7 ms for electrons and 70 ms for full distribution functions, FAST resolved auroral microphysics at horizontal scales of tens of meters — shorter than the electron inertial length and ion gyroradius. This paper presents the mission concept, instrument package, and an executive summary of the special-section results: (1) confirmation that upward and downward current regions are mirror-image electrodynamic systems with quasi-static parallel potentials accelerating electrons (downward) and ion beams (upward) in the upward-current region, and accelerating electron beams (upward) in the downward-current region; (2) discovery of Debye-scale, three-dimensional electron-hole solitary waves carrying tens to hundreds of volts in upgoing electron beams; (3) identification of ion cyclotron waves as the modulator producing flickering aurora; and (4) a self-consistent, quantitative explanation of auroral kilometric radiation (AKR) from cold-plasma-depleted density cavities in which a "horseshoe" hot electron distribution provides the cyclotron-maser free energy.

**한국어**: 본 1998년 Geophysical Research Letters 임무 논문은 1996년 8월 21일에 350 × 4175 km, 경사 83°의 타원 궤도로 발사된 NASA의 두 번째 Small Explorer(SMEX) 위성인 FAST(Fast Auroral SnapshoT)를 소개한다. FAST는 단일 통합 과학 측정기로 설계되었다: 모든 센서가 스핀 안정 우주선과 동기화되고, 우주선이 오로라 호를 가로지를 때 트리거되는 최대 1 Gbit의 "스냅샷" 데이터를 버퍼링하는 공통 측정기 자료 처리부(IDPU)에 연결된다. 스핀 평면을 지자기장에 정렬하고 전자 1.7 ms, 전체 분포 함수 70 ms의 시간 분해능을 달성함으로써, FAST는 전자 관성 길이 및 이온 자이로 반경보다 짧은 수십 m의 수평 스케일에서 오로라 미시 물리를 분해했다. 본 논문은 임무 개념, 측정기 패키지, 그리고 특별호 결과의 행정적 요약을 제시한다: (1) 상향과 하향 전류 영역이 거울상 전기역학 시스템이라는 확인 — 상향 전류 영역에서는 준정적 평행 전위가 전자(하향)와 이온 빔(상향)을 가속시키고, 하향 전류 영역에서는 전자 빔(상향)을 가속시킴; (2) 상향 전자 빔에서 수십~수백 볼트를 운반하는 Debye 스케일의 3차원 전자 hole 고립파 발견; (3) 점멸 오로라를 만드는 변조원으로서의 이온 사이클로트론파 식별; (4) 차가운 플라즈마가 결핍된 밀도 공동에서의 오로라 킬로미터 복사(AKR)에 대한 자기일관적·정량적 설명 — "말발굽" 뜨거운 전자 분포가 사이클로트론 메이저 자유 에너지를 제공.

---

## 2. Reading Notes / 읽기 노트

### Part I: Auroral Acceleration Region — Background Physics / 오로라 가속 영역의 배경 물리

**English**: The auroral acceleration region (AAR) is a band of altitudes — typically 1000-12000 km along auroral magnetic field lines — where parallel ($\mathbf{E}_\parallel \neq 0$) electric fields exist for substantial fractions of an hour. Standard MHD predicts $\mathbf{E}\cdot\mathbf{B}=0$, but the AAR is one of a few places where MHD breaks down. The breakdown happens because the magnetosphere demands current continuity: large-scale magnetospheric Birkeland currents (Region-1, Region-2) must close through the high-latitude ionosphere. In the upward-current region (current carried by downward electrons or upward ions), the cold ionosphere cannot supply enough upgoing current carriers, and a parallel potential drop $\Phi_\parallel$ develops to "pump" magnetospheric electrons downward, accelerating them through several keV by the time they hit the upper atmosphere where they ionize $N_2$/$O$ and produce visible aurora. The energy-vs-latitude pattern looks like an inverted "V" because the satellite cuts across an arc whose potential peaks in the middle.

**한국어**: 오로라 가속 영역(AAR)은 일반적으로 오로라 자기장선을 따라 1000-12000 km의 고도대로, 평행 전기장($\mathbf{E}_\parallel \neq 0$)이 한 시간의 상당 분율 동안 존재하는 곳이다. 표준 MHD는 $\mathbf{E}\cdot\mathbf{B}=0$을 예측하지만, AAR은 MHD가 깨지는 몇 안 되는 장소 중 하나다. 그 붕괴는 자기권이 전류 연속성을 요구하기 때문에 일어난다: 대규모 자기권 Birkeland 전류(Region-1, Region-2)는 고위도 전리권을 통해 닫혀야 한다. 상향 전류 영역(하향 전자 또는 상향 이온이 전류를 운반)에서, 차가운 전리권은 충분한 상향 전류 운반자를 공급할 수 없어, 평행 전위 강하 $\Phi_\parallel$이 발생하여 자기권 전자를 아래로 "펌프"하고, 그것들이 상층 대기에 도달할 때까지 수 keV로 가속시켜 $N_2$/$O$를 이온화하고 가시 오로라를 생성한다. 에너지-위도 패턴은 위성이 호를 가로지르며 호의 중앙에서 전위가 최고가 되기 때문에 거꾸로 된 "V" 모양으로 보인다.

### Part II: Mission Concept and Orbit (p. 2013) / 임무 개념과 궤도

**English**: FAST is the second SMEX satellite. It was launched from Vandenberg AFB on August 21, 1996 by a Pegasus-XL air-launched rocket. The orbit is highly elliptical: perigee 350 km, apogee 4175 km, inclination 83°. This places apogee well within the auroral acceleration region, with periodic passes through visible-aurora altitudes. The orbit precesses (wraps in local time) over the year; launch was timed so that initial apogee was over the northern auroral zone near the noon-midnight meridian during the 1996-97 northern winter, coinciding with the International Auroral Study campaign (Jan 1 - Mar 15, 1997) at Poker Flat, Alaska, and complementary measurements from the NASA Polar satellite (~9 $R_E$ above the polar regions).

The observing strategy exploits a spatial fact: auroral processes occur in narrow latitudinal bands (10-15° wide) circling the magnetic poles. Therefore, high-rate measurements are not needed throughout the orbit. Instead, the on-board computer triggers "snapshot" recordings (up to 8 Mbps) when the satellite enters an auroral region. Data is buffered in a 1 Gbit (128 MB) solid-state recorder until a ground station passes overhead. Three downlink rates (900 kbps, 1.5 Mbps, 2.25 Mbps) are available via S-band.

**한국어**: FAST는 두 번째 SMEX 위성이다. 1996년 8월 21일 Vandenberg 공군기지에서 Pegasus-XL 공중 발사 로켓에 의해 발사되었다. 궤도는 매우 타원형이다: 근지점 350 km, 원지점 4175 km, 경사 83°. 이는 원지점을 오로라 가속 영역 내에 놓이게 하며, 가시 오로라 고도를 주기적으로 통과한다. 궤도는 일 년에 걸쳐 (지방시상으로) 세차운동한다; 발사 시기는 초기 원지점이 1996-97년 북반구 겨울에 정오-자정 자오선 근처에서 북쪽 오로라 지역 위에 있도록 조정되었으며, 이는 알래스카 Poker Flat에서의 International Auroral Study 캠페인(1997년 1월 1일~3월 15일) 및 NASA Polar 위성(극 영역 위 ~9 $R_E$)으로부터의 보완 측정과 일치했다.

관측 전략은 공간적 사실을 활용한다: 오로라 과정은 자극을 둘러싼 좁은 위도대(10-15° 폭) 내에서 일어난다. 따라서 궤도 전체에 걸친 고속 측정은 필요하지 않다. 대신, 온보드 컴퓨터는 위성이 오로라 영역에 진입할 때 "스냅샷" 기록(최대 8 Mbps)을 트리거한다. 자료는 지상국이 머리 위를 지날 때까지 1 Gbit(128 MB) 고체 상태 기록기에 버퍼링된다. S-밴드를 통해 세 가지 다운링크 속도(900 kbps, 1.5 Mbps, 2.25 Mbps)가 사용 가능하다.

### Part III: Instrument Package (pp. 2013-2014) / 측정기 패키지

**English**:
- **Spacecraft**: 191 kg total, 51 kg of instruments. ~1 m diameter × 1 m height. Spin-stabilized "orbit-normal spinner". Body-mounted solar arrays (5.6 m² cells, 52 W orbit-average). Two magnetic torque coils for attitude. No on-board propulsion.

- **Electrostatic Analyzers (ESAs)**: 16 "top-hat" toroidal ESAs in 4 stacks of 4, 90° apart. Each instrument has a 180° fan-shaped field of view in the spin plane (typically aligned within ~6° of B). Energy range 4 eV-30 keV (electrons), 3 eV-25 keV (ions). Three "Stepped ESA" (SESA) per stack act as spectrographs: 1.7 ms cadence in 16 pitch-angle bins. The fourth analyzer per stack is an IESA or EESA (Ion / Electron Energy Spectrometer): full distribution every 70 ms, 32 pitch-angle bins, with magnetic-tracking deflection plates.

- **TEAMS (Time-of-flight Energy Angle Mass Spectrograph)**: instantaneous 360°×8° FOV; 1.2-12000 eV/q; full 3D distribution per half-spin (2.5 s); resolves H⁺, He⁺, He²⁺, O⁺, O₂⁺, NO⁺ via TOF. 16 azimuthal bins of 22.5°.

- **Electric Field Instrument**: was designed with 10 spherical sensors — 8 on four 28-m radial wire booms (two per boom, at 28 m and 23 m from spacecraft), 2 on axial stacers (8 m tip-to-tip). Each sphere has its own preamp. One wire boom failed to deploy properly, but three booms suffice for 3-component vector E. Spheres can be operated as Langmuir probes for plasma density. Frequency: DC to 2 MHz; dynamic range 100 dB. Data products: continuous waveform (2 ksps), burst waveform (2 Msps), spectra (16 Hz - 2 MHz). Onboard: tracking spectrum analyzer, wave-particle correlator, DSP for FFT/cross-spectrum.

- **Magnetic Field Sensors**: DC fluxgate (3-axis, ring-core, on a 2-m boom) + AC search-coil (3-axis: 10 Hz - 2.5 kHz on two axes; third axis to 500 kHz).

- **IDPU (Instrument Data Processor Unit)**: single point of contact between instruments and spacecraft. Houses 1 Gbit mass memory, microprocessor, formatter, on-board burst-trigger logic.

**한국어**:
- **우주선**: 총 191 kg, 측정기 51 kg. 직경 ~1 m × 높이 1 m. 스핀 안정 "궤도 법선 스피너". 본체 장착 태양 전지판(5.6 m² 셀, 궤도 평균 52 W). 자세 제어용 자기 토크 코일 2개. 온보드 추진제 없음.

- **정전기 분석기 (ESA)**: 16개의 "top-hat" 환상 ESA가 90° 간격으로 4개 스택에 4개씩 배치. 각 측정기는 스핀 평면 내 180° 부채꼴 시야(보통 B에서 ~6° 이내 정렬). 에너지 범위 전자 4 eV-30 keV, 이온 3 eV-25 keV. 스택당 3개의 "Stepped ESA"(SESA)는 분광기로 작동: 16개 피치각 빈에서 1.7 ms 시간분해능. 스택당 네 번째 분석기는 IESA 또는 EESA(이온/전자 에너지 분광기): 70 ms마다 전체 분포, 32 피치각 빈, 자기장 추적 편향판 포함.

- **TEAMS (비행시간 에너지·각도·질량 분광기)**: 순간 360°×8° 시야; 1.2-12000 eV/q; 반스핀(2.5 s)당 전체 3D 분포; TOF로 H⁺, He⁺, He²⁺, O⁺, O₂⁺, NO⁺ 분해. 22.5°의 16개 방위각 빈.

- **전기장 측정기**: 10개의 구형 센서로 설계 — 4개의 28 m 방사형 와이어 붐에 8개(붐당 2개, 우주선으로부터 28 m 및 23 m), 축 스테이서에 2개(끝점 간 8 m). 각 구체는 자체 프리앰프 보유. 한 와이어 붐이 제대로 펼쳐지지 않았으나 세 붐으로 3성분 벡터 E 측정 가능. 구체는 플라즈마 밀도 측정을 위한 Langmuir 프로브로 작동 가능. 주파수: DC~2 MHz; 동적 범위 100 dB. 자료 산출물: 연속 파형(2 ksps), 버스트 파형(2 Msps), 스펙트럼(16 Hz~2 MHz). 온보드: 추적 스펙트럼 분석기, 파동-입자 상관기, FFT/교차 스펙트럼용 DSP.

- **자기장 센서**: DC 플럭스게이트(3축, 링 코어, 2 m 붐) + AC 서치코일(3축: 두 축 10 Hz-2.5 kHz, 세 번째 축 최대 500 kHz).

- **IDPU (측정기 자료 처리부)**: 측정기와 우주선 간 단일 접점. 1 Gbit 대용량 메모리, 마이크로프로세서, 포매터, 온보드 버스트 트리거 논리 포함.

### Part IV: Symmetric Auroral Current Regions (Figure 2, p. 2015) / 대칭 오로라 전류 영역

**English**: Figure 2 of the paper is the conceptual centerpiece. It tabulates eight pairs of features distinguishing the **upward** (visible aurora, right) and **downward** ("inverse aurora", left) current regions:

| # | Upward current region (visible aurora) | Downward current region (inverse aurora) |
|---|---|---|
| 1 | Upward J | Downward J |
| 2 | Converging electrostatic shocks (E points outward at edges) | Diverging electrostatic shocks (E points inward at edges) |
| 3 | Large-scale density cavity ($n_e$ depletion) | Small-scale density cavities |
| 4 | Down-going "inverted-V" electrons (precipitating) | Up-going field-aligned electron beams + counter-streaming electrons |
| 5 | Up-going ion beam + ion conics | Ion heating transverse to B; energetic ion conics |
| 6 | Large-amplitude ion cyclotron waves + E-field turbulence | ELF E-field turbulence + ion cyclotron waves |
| 7 | Nonlinear time-domain structures (NL) associated with ion cyclotron | Fast solitary waves: 3-D, rapidly moving electron holes |
| 8 | AKR (Auroral Kilometric Radiation) source region | VLF saucer source region |

The symmetry is profound. In the upward-current region, the parallel potential drop sits **above** the satellite; downgoing electrons are accelerated through it. In the downward-current region, the parallel potential drop sits **below** the satellite; upgoing electrons are accelerated through it. Ion-beam energy on the upward side and electron-beam energy on the downward side both directly measure $e\Phi_\parallel$, providing the single most decisive test of the quasi-static potential model.

**한국어**: 논문의 Figure 2는 개념적 핵심이다. **상향 전류**(가시 오로라, 오른쪽)와 **하향 전류**("역 오로라", 왼쪽) 영역을 구분하는 여덟 쌍의 특성을 표로 제시한다:

| # | 상향 전류 영역 (가시 오로라) | 하향 전류 영역 (역 오로라) |
|---|---|---|
| 1 | 상향 J | 하향 J |
| 2 | 수렴 정전기 충격파 (가장자리에서 E가 바깥쪽) | 발산 정전기 충격파 (가장자리에서 E가 안쪽) |
| 3 | 대규모 밀도 공동 ($n_e$ 결핍) | 소규모 밀도 공동들 |
| 4 | 하향 "인버티드-V" 전자 (침강) | 상향 자기장선 전자 빔 + 역방향 전자 |
| 5 | 상향 이온 빔 + 이온 콘 | B에 수직한 이온 가열; 에너지 이온 콘 |
| 6 | 대진폭 이온 사이클로트론파 + E장 난류 | ELF E장 난류 + 이온 사이클로트론파 |
| 7 | 이온 사이클로트론과 동반된 비선형 시간 영역 구조(NL) | 빠른 고립파: 3D, 빠르게 이동하는 전자 hole |
| 8 | AKR (오로라 킬로미터 복사) 원천 영역 | VLF saucer 원천 영역 |

이 대칭은 심오하다. 상향 전류 영역에서, 평행 전위 강하는 위성 **위에** 있다; 하향 전자가 그것을 통해 가속된다. 하향 전류 영역에서, 평행 전위 강하는 위성 **아래에** 있다; 상향 전자가 그것을 통해 가속된다. 상향 측의 이온 빔 에너지와 하향 측의 전자 빔 에너지는 모두 $e\Phi_\parallel$을 직접 측정하며, 이는 준정적 전위 모델의 가장 결정적인 단일 검증을 제공한다.

### Part V: Scientific Highlights — Upward Current Regions (p. 2015) / 과학 하이라이트 — 상향 전류 영역

**English**: Quantitative measurements show that hot auroral inverted-V electrons account for essentially all of the field-aligned current in arcs (Elphic et al., 1998; McFadden et al., 1998a,b). The current density derived from $\nabla \times \mathbf{B}$ matches the moment $\int e v_\parallel f \, d^3v$ taken from the electron distribution. This eliminates the need for invoking other current carriers in the upward-current region.

The most decisive test of the quasi-static parallel-potential model: **upgoing ion beam energy = parallel potential drop**. When an electrostatic-shock pair extends below the spacecraft, ions in those flux tubes were accelerated upward through $e\Phi_\parallel(z_{\text{sc}})$, and the ion beam energy must equal that potential. McFadden et al. (1998a) and Ergun et al. (1998c) show this works to within measurement uncertainty: the kinetic energy of ion beams agrees quantitatively with $\int E_\parallel \, dz$ inferred from the convergent E-field structure. Mass-resolved TEAMS measurements (Moebius et al., 1998) reveal that different ion species emerge with slightly different energies — implying that an additional, mass-dependent acceleration process operates alongside the parallel potential.

Intense electromagnetic ion cyclotron (EMIC) waves are generated within inverted-V electron regions (Cattell et al., 1998). Chaston et al. (1998) measure the wave Poynting flux and find it can reach 10% of the precipitating electron energy flux — a substantial dissipation channel. McFadden et al. (1998a) directly observe electron flux modulation at the ion cyclotron frequency, confirming Temerin et al. (1986)'s prediction that EMIC waves modulate inverted-V electrons to produce flickering aurora at 5-15 Hz. Lund et al. (1998) report preferential heating of He⁺ ions in association with EMIC waves, providing detailed verification of resonant ion-cyclotron heating theory.

For AKR: Strangeway et al. (1998) confirm very low plasma density in the AKR source region, consistent with Benson & Calvert (1979) and Persoon et al. (1988). New finding: the hot auroral electron density alone matches the total plasma density (combined magnetospheric ions + upgoing beam ions), proving that essentially **no cold plasma exists in the cavity** (McFadden et al., 1998b). The AKR low-frequency cutoff lies below the cold electron cyclotron frequency $\Omega_{ce}$ and only matches when one applies the relativistic correction $\omega = \Omega_{ce}/\gamma$ for the observed hot (few keV) electrons (Ergun et al., 1998c). With cold plasma absent, Delory et al. (1998) compute that the positive slope $\partial f / \partial v_\perp > 0$ on the observed "horseshoe" electron distribution provides the cyclotron-maser free energy that drives AKR growth.

**한국어**: 정량적 측정은 뜨거운 오로라 인버티드-V 전자가 호 안의 자기장선 전류의 본질적으로 전부를 설명함을 보인다(Elphic et al., 1998; McFadden et al., 1998a,b). $\nabla \times \mathbf{B}$로부터 유도된 전류 밀도는 전자 분포로부터 얻은 모멘트 $\int e v_\parallel f \, d^3v$와 일치한다. 이는 상향 전류 영역에서 다른 전류 운반자를 도입할 필요를 제거한다.

준정적 평행 전위 모델의 가장 결정적인 검증: **상향 이온 빔 에너지 = 평행 전위 강하**. 정전기 충격파 쌍이 우주선 아래까지 확장될 때, 그 자속관 안의 이온은 $e\Phi_\parallel(z_{\text{sc}})$을 통해 위로 가속되었으며, 이온 빔 에너지는 그 전위와 같아야 한다. McFadden et al. (1998a)과 Ergun et al. (1998c)은 이것이 측정 불확도 안에서 작동함을 보인다: 이온 빔의 운동 에너지는 수렴 E장 구조로부터 추론된 $\int E_\parallel \, dz$와 정량적으로 일치한다. 질량 분해 TEAMS 측정(Moebius et al., 1998)은 서로 다른 이온 종이 약간 다른 에너지로 나타남을 밝혀, 평행 전위와 함께 추가의 질량 의존적 가속 과정이 작동함을 시사한다.

강한 전자기 이온 사이클로트론(EMIC) 파동이 인버티드-V 전자 영역 내에서 생성된다(Cattell et al., 1998). Chaston et al. (1998)은 파동 포인팅 유속을 측정해 침강 전자 에너지 유속의 10%까지 도달할 수 있음을 발견 — 상당한 소산 채널. McFadden et al. (1998a)은 이온 사이클로트론 주파수에서 전자 유속 변조를 직접 관측해, EMIC 파동이 인버티드-V 전자를 변조하여 5-15 Hz의 점멸 오로라를 만든다는 Temerin et al. (1986)의 예측을 확인한다. Lund et al. (1998)은 EMIC 파동과 동반된 He⁺ 이온의 우선적 가열을 보고하며, 공명 이온 사이클로트론 가열 이론의 상세한 검증을 제공한다.

AKR에 대해: Strangeway et al. (1998)은 AKR 원천 영역의 매우 낮은 플라즈마 밀도를 확인하며, 이는 Benson & Calvert (1979)와 Persoon et al. (1988)과 일치한다. 새로운 발견: 뜨거운 오로라 전자 밀도만으로 총 플라즈마 밀도(자기권 이온 + 상향 빔 이온의 결합)와 일치하며, 이는 공동 안에 본질적으로 **차가운 플라즈마가 없음**을 증명한다(McFadden et al., 1998b). AKR 저주파 절단은 차가운 전자 사이클로트론 주파수 $\Omega_{ce}$ 아래에 있으며, 관측된 뜨거운(수 keV) 전자에 대한 상대론적 보정 $\omega = \Omega_{ce}/\gamma$를 적용해야만 일치한다(Ergun et al., 1998c). 차가운 플라즈마가 부재한 상태에서, Delory et al. (1998)은 관측된 "말발굽" 전자 분포 위의 양의 기울기 $\partial f / \partial v_\perp > 0$가 AKR 성장을 추동하는 사이클로트론 메이저 자유 에너지를 제공함을 계산한다.

### Part VI: Scientific Highlights — Downward Current Regions (p. 2016) / 과학 하이라이트 — 하향 전류 영역

**English**: FAST observations frequently show intense upgoing field-aligned electron beams with energies up to several keV — the most intense electron fluxes found in the auroral region (Carlson et al., 1998). These are accelerated upward by **diverging** electric field structures (parallel potential drop located **below** the satellite). The agreement between measured potentials and observed beam energies confirms parallel potential structures as the acceleration mechanism — exactly the symmetric counterpart to the ion-beam test in the upward-current region.

These electron beams are clearly associated with VLF saucers (Gurnett & Frank 1972 had postulated this), and accompanied by deep density cavities and the most energetic ion conics in the auroral region. Ergun et al. (1998c) discovered large-amplitude, three-dimensional electric solitary structures in the upgoing electron beams. These are Debye-scale, positively charged electron "holes" — phase-space vortices in which trapped electrons are missing — that move with the beam velocity and contain potential wells of 10-100 V. They may help support the parallel potentials that accelerate the beams (an anomalous resistivity contribution), and they appear to be very effective at heating ions transversely.

The parallel electric fields that accelerate upgoing electrons also **inhibit plasmasheet electron precipitation**, producing dark regions in the diffuse aurora. Marklund et al. (1994), using Freja, proposed that these structures are the source of "black aurora". FAST extended these observations to higher altitudes: upgoing beams are seldom found below 2000 km, but the diverging-field signature persists across that boundary.

**한국어**: FAST 관측은 종종 수 keV까지의 에너지를 가진 강력한 상향 자기장선 전자 빔을 보여주는데, 이는 오로라 영역에서 발견되는 가장 강력한 전자 유속이다(Carlson et al., 1998). 이들은 **발산** 전기장 구조(위성 **아래에** 위치한 평행 전위 강하)에 의해 위로 가속된다. 측정된 전위와 관측된 빔 에너지의 일치는 가속 메커니즘으로서의 평행 전위 구조를 확인한다 — 이는 상향 전류 영역의 이온 빔 검증의 정확한 대칭 짝이다.

이 전자 빔들은 VLF saucer와 명확히 동반되며(Gurnett & Frank 1972가 이를 가정함), 깊은 밀도 공동 및 오로라 영역의 가장 에너지 높은 이온 콘과 함께 나타난다. Ergun et al. (1998c)은 상향 전자 빔에서 대진폭 3차원 전기 고립 구조를 발견했다. 이들은 Debye 스케일의 양전하 전자 "hole" — 갇힌 전자가 결핍된 위상 공간 소용돌이 — 으로, 빔 속도로 이동하고 10-100 V의 전위 우물을 포함한다. 이들은 빔을 가속시키는 평행 전위를 지지하는데 도움을 줄 수 있고(이상 저항 기여), 이온의 수직 가열에 매우 효과적인 것으로 보인다.

상향 전자를 가속하는 평행 전기장은 또한 **플라즈마시트 전자 침강을 억제**하여 확산 오로라 안에 어두운 영역을 만든다. Marklund et al. (1994)은 Freja를 이용해 이 구조들이 "검은 오로라"의 원천이라고 제안했다. FAST는 이 관측을 더 높은 고도까지 확장했다: 상향 빔은 2000 km 아래에서는 거의 발견되지 않지만, 발산장 서명은 그 경계를 가로질러 지속된다.

### Part VII: Other Regions and Correlative Studies / 다른 영역 및 상관 연구

**English**: FAST data also support studies outside the nightside auroral zone:
- **Cusp particle acceleration** (Pfaff et al., 1998) — FAST observed magnetosheath plasma entry features at high latitudes
- **Plasmasheet inner edge** (Kistler et al., 1998) — drifting ions and charge-exchange signatures from low-altitude mirroring
- **Ground/airborne conjugate studies** (Stenbaek-Nielsen et al., 1998) — auroral arc thicknesses compared with FAST-measured electron energy fluxes
- **FAST-Geotail** (Sigsbee et al., 1998) — magnetosphere-ionosphere coupling
- **FAST-Polar** (Peterson et al., 1998) — solar wind plasma entry at the cusp

**한국어**: FAST 자료는 또한 야간 오로라 영역 외의 연구도 지원한다:
- **커스프 입자 가속**(Pfaff et al., 1998) — FAST는 고위도에서 자기권계면 플라즈마 진입 특성을 관측
- **플라즈마시트 안쪽 경계**(Kistler et al., 1998) — 저고도 거울 반사로부터의 드리프트 이온과 전하 교환 서명
- **지상/항공 결합 연구**(Stenbaek-Nielsen et al., 1998) — 오로라 호 두께를 FAST 측정 전자 에너지 유속과 비교
- **FAST-Geotail**(Sigsbee et al., 1998) — 자기권-전리권 결합
- **FAST-Polar**(Peterson et al., 1998) — 커스프에서의 태양풍 플라즈마 진입

---

## 3. Key Takeaways / 핵심 시사점

1. **The auroral acceleration region is the textbook breakdown of ideal MHD / 오로라 가속 영역은 이상 MHD가 깨지는 교과서적 사례** — Parallel electric fields exist over thousands of km. FAST quantitatively confirmed this through ion-beam energy = parallel potential drop. / 평행 전기장은 수천 km에 걸쳐 존재한다. FAST는 이를 이온 빔 에너지 = 평행 전위 강하의 관계로 정량적으로 확인했다.

2. **Upward and downward current regions are mirror-image systems / 상향과 하향 전류 영역은 거울상 시스템** — Figure 2 codifies eight symmetric features. The downward-current region ("inverse aurora") is not just a current return path; it is its own rich electrodynamic environment with its own characteristic plasma signatures. / Figure 2는 여덟 가지 대칭 특성을 코드화한다. 하향 전류 영역("역 오로라")은 단순한 전류 귀환 경로가 아니라, 고유한 풍부한 전기역학적 환경이며 고유한 특성적 플라즈마 서명을 갖는다.

3. **Hot inverted-V electrons account for essentially all the field-aligned current / 뜨거운 인버티드-V 전자가 자기장선 전류의 본질적으로 전부를 설명** — This eliminates the need for cold-plasma current carriers in upward-current arcs and validates the "Knight relation" framework where current is set by the source-region distribution and parallel potential. / 이것은 상향 전류 호에서 차가운 플라즈마 전류 운반자의 필요성을 제거하고, 전류가 원천 영역 분포와 평행 전위에 의해 결정되는 "Knight 관계" 틀을 검증한다.

4. **Debye-scale electron holes are real and ubiquitous in upgoing beams / Debye 스케일 전자 hole이 상향 빔에서 실재하며 흔하다** — Ergun et al. (1998c)'s discovery showed that auroral parallel-current physics has a strongly nonlinear, kinetic component. These holes are now recognized as a generic feature of beam-plasma interactions throughout the heliosphere. / Ergun et al. (1998c)의 발견은 오로라 평행 전류 물리에 강한 비선형·운동학적 성분이 있음을 보였다. 이 hole들은 이제 태양권 전반의 빔-플라즈마 상호작용의 일반적 특성으로 인식된다.

5. **AKR is generated by the cyclotron maser instability in cold-plasma-depleted cavities / AKR은 차가운 플라즈마가 결핍된 공동에서 사이클로트론 메이저 불안정에 의해 생성** — FAST showed that (i) cavities truly contain no cold plasma; (ii) the relativistic $\gamma$ correction is necessary; (iii) "horseshoe" hot electron distributions provide the free energy. This template now applies to planetary radio emissions and even ultracool dwarfs. / FAST는 (i) 공동이 실제로 차가운 플라즈마를 포함하지 않음, (ii) 상대론적 $\gamma$ 보정이 필요함, (iii) "말발굽" 뜨거운 전자 분포가 자유 에너지를 제공함을 보였다. 이 템플릿은 이제 행성 전파 방출과 초저온 왜성에도 적용된다.

6. **Flickering aurora is electron modulation by ion cyclotron waves / 점멸 오로라는 이온 사이클로트론파에 의한 전자 변조** — Direct in-situ measurement (McFadden et al., 1998a) confirmed Temerin et al. (1986)'s decade-old prediction. This established a clean kinetic mechanism connecting visible auroral dynamics to wave physics in the AAR. / 직접 in-situ 측정(McFadden et al., 1998a)이 10년 된 Temerin et al. (1986)의 예측을 확인했다. 이는 가시 오로라 역학을 AAR의 파동 물리에 연결하는 깔끔한 운동학적 메커니즘을 확립했다.

7. **Mission design philosophy: "snapshot" data + on-board buffering / 임무 설계 철학: "스냅샷" 자료 + 온보드 버퍼링** — FAST proved that you can do high-rate science in a Small Explorer budget by exploiting the spatially localized nature of the target phenomena and using mass memory. This blueprint has since been adopted by THEMIS, MMS, TRACERS, and many lunar/planetary missions. / FAST는 표적 현상의 공간적 국소성을 활용하고 대용량 메모리를 사용함으로써 Small Explorer 예산 내에서 고속 과학을 할 수 있음을 증명했다. 이 청사진은 이후 THEMIS, MMS, TRACERS 및 많은 달/행성 임무에 채택되었다.

8. **Black aurora is the absence-of-precipitation signature of downward-current parallel potentials / 검은 오로라는 하향 전류 평행 전위의 비침강 서명** — The same fields that accelerate upgoing electrons inhibit plasmasheet precipitation. Black auroral structures are thus dynamically active, not "blank canvas" — they map directly to the inverse-aurora electrodynamics. / 상향 전자를 가속하는 동일한 전기장이 플라즈마시트 침강을 억제한다. 따라서 검은 오로라 구조는 "빈 캔버스"가 아니라 동적으로 활성이며, 역 오로라 전기역학에 직접 사상된다.

---

## 4. Mathematical Summary / 수학적 요약

### (A) Parallel-potential acceleration and inverted-V electrons / 평행 전위 가속과 인버티드-V 전자

For a magnetospheric source electron of energy $W_0 = \frac{1}{2} m_e v_{\parallel 0}^2$ entering an upward parallel potential drop $\Phi_\parallel > 0$ from above, energy conservation gives the kinetic energy at the bottom of the potential:

$$
W_{\text{precip}} = W_0 + e \, \Phi_\parallel
$$

For a Maxwellian source $f_0 \propto \exp(-W_0/k_B T_e)$, the precipitating distribution becomes a shifted, accelerated Maxwellian and the differential energy flux (number flux × energy per energy bin) shows a peak at $E_{\text{peak}} \approx e\Phi_\parallel + (1\!-\!2) k_B T_e$. This produces the characteristic "monoenergetic" inverted-V signature.

뜨거운 자기권 전자가 위로부터 상향 평행 전위 강하 $\Phi_\parallel > 0$에 진입할 때, 에너지 보존 법칙은 위 식과 같다. Maxwellian 원천 $f_0 \propto \exp(-W_0/k_B T_e)$에 대해, 침강 분포는 이동된 가속 Maxwellian이 되며, 차등 에너지 유속은 $E_{\text{peak}} \approx e\Phi_\parallel + (1\!-\!2) k_B T_e$에서 최대값을 보인다. 이것이 특성적 "단색" 인버티드-V 서명을 만든다.

### (B) Knight relation: current-voltage relation for parallel potentials / Knight 관계: 평행 전위에 대한 전류-전압 관계

For a Maxwellian source ($n_e$, $T_e$) and a non-zero parallel potential $\Phi_\parallel$ (with magnetic-mirror ratio $R = B_{\text{iono}}/B_{\text{source}} \gg 1$), the precipitating field-aligned current density is:

$$
j_\parallel = e n_e \sqrt{\frac{k_B T_e}{2 \pi m_e}} \left[ 1 + \left(R - 1\right) \exp\!\left(-\frac{e\Phi_\parallel}{k_B T_e (R-1)}\right) \right]^{-1} \cdot \left[1 - \exp\!\left(-\frac{e\Phi_\parallel}{k_B T_e}\right)\right]
$$

In the limit $e\Phi_\parallel \gg k_B T_e$ this becomes approximately linear: $j_\parallel \approx e n_e \sqrt{k_B T_e / 2\pi m_e}$ (saturation), and at small $\Phi_\parallel$ it is linear in $\Phi_\parallel$ (Ohmic). FAST verified this relation directly.

Maxwellian 원천($n_e$, $T_e$)과 0이 아닌 평행 전위 $\Phi_\parallel$(자기 거울비 $R = B_{\text{iono}}/B_{\text{source}} \gg 1$ 포함)에 대해, 침강 자기장선 전류 밀도는 위와 같다. $e\Phi_\parallel \gg k_B T_e$ 극한에서는 근사적으로 포화하며($j_\parallel \approx e n_e \sqrt{k_B T_e / 2\pi m_e}$), 작은 $\Phi_\parallel$에서는 $\Phi_\parallel$에 대해 선형(Ohmic)이다. FAST는 이 관계를 직접 검증했다.

### (C) Ion beam energy (decisive test) / 이온 빔 에너지 (결정적 검증)

For the upward-current region, ion beams measured at the spacecraft were accelerated up through any parallel potential drop located below it:

$$
\frac{1}{2} m_i v_{i,\text{beam}}^2 = q_i \int_{z_{\text{below}}}^{z_{\text{sc}}} E_\parallel(z) \, dz \equiv q_i \, \Phi_\parallel^{\text{below}}
$$

Quantitative agreement (within ~10-20%) between $E_{\text{ion-beam}}$ and $\int E_\parallel dz$ inferred from the perpendicular electric-field structure was the cleanest verification of the quasi-static potential picture.

상향 전류 영역에서, 우주선에서 측정된 이온 빔은 우주선 아래의 어떤 평행 전위 강하를 통해 위로 가속되었다. $E_{\text{ion-beam}}$과 수직 전기장 구조로부터 추론된 $\int E_\parallel dz$ 사이의 정량적 일치(~10-20% 이내)는 준정적 전위 그림의 가장 깨끗한 검증이었다.

### (D) Cyclotron maser instability and AKR / 사이클로트론 메이저 불안정과 AKR

The relativistic resonance condition for waves at frequency $\omega$ near the electron cyclotron frequency is:

$$
\omega - k_\parallel v_\parallel - \frac{\Omega_{ce}}{\gamma} = 0, \qquad \gamma = \left(1 - \frac{v^2}{c^2}\right)^{-1/2} \approx 1 + \frac{W}{m_e c^2}
$$

For typical AKR conditions ($W \sim 5\,\text{keV}$, $\gamma - 1 \approx 10^{-2}$), $\omega/\Omega_{ce} \approx 1 - 10^{-2}$, i.e., AKR sits about 1% below the cold cyclotron frequency. The growth rate requires a positive perpendicular slope of the distribution function:

$$
\gamma_g \propto \int \frac{1}{\gamma}\frac{\partial f}{\partial v_\perp}\Big|_{\text{resonance}} v_\perp^2 \, dv_\perp \, dv_\parallel > 0
$$

The "horseshoe" distribution observed by FAST has $\partial f / \partial v_\perp > 0$ over a wide range of $v_\perp$, providing the free energy.

주파수 $\omega$가 전자 사이클로트론 주파수 근처일 때 상대론적 공명 조건은 위와 같다. 일반적인 AKR 조건($W \sim 5\,\text{keV}$, $\gamma - 1 \approx 10^{-2}$)에서, $\omega/\Omega_{ce} \approx 1 - 10^{-2}$, 즉 AKR은 차가운 사이클로트론 주파수보다 약 1% 아래에 위치한다. 성장률은 분포 함수의 양의 수직 기울기를 요구한다. FAST가 관측한 "말발굽" 분포는 넓은 $v_\perp$ 범위에서 $\partial f / \partial v_\perp > 0$을 가져 자유 에너지를 제공한다.

### (E) Debye scaling for electron holes / 전자 hole의 Debye 척도

The natural width of an electron hole is a few Debye lengths:

$$
\lambda_D = \sqrt{\frac{\varepsilon_0 k_B T_e}{n_e e^2}} = 7.4\,\text{m} \cdot \sqrt{\frac{T_e/\text{1 keV}}{n_e/\text{1 cm}^{-3}}}
$$

For FAST AAR conditions ($T_e \sim 1$ keV, $n_e \sim 1$ cm⁻³) this gives $\lambda_D \sim 7$ m. Electron-hole widths of 10-100 m and potential depths of 10-100 V (small fraction of beam energy) are observed.

전자 hole의 자연 폭은 수 Debye 길이이다. FAST AAR 조건($T_e \sim 1$ keV, $n_e \sim 1$ cm⁻³)에서 $\lambda_D \sim 7$ m. 10-100 m의 hole 폭과 10-100 V의 전위 깊이(빔 에너지의 작은 분율)가 관측된다.

### (F) Poynting flux comparison: EMIC waves vs precipitating electrons / 포인팅 유속 비교: EMIC 파동 vs 침강 전자

$$
S_{\text{wave}} = \langle \mathbf{E} \times \mathbf{B} \rangle / \mu_0, \qquad S_{\text{electron}} = \int W \, v_\parallel \, f \, d^3 v
$$

Chaston et al. (1998): $S_{\text{EMIC}} \lesssim 0.1 \, S_{\text{electron}}$, demonstrating that wave generation is a substantial energy sink for the precipitating electron beam — but not the dominant one.

Chaston et al. (1998)은 $S_{\text{EMIC}} \lesssim 0.1 \, S_{\text{electron}}$을 보여, 파동 생성이 침강 전자 빔에 대한 상당한 에너지 흡수원이지만 지배적이지는 않음을 입증했다.

### Worked numerical example: parallel potential drop from inverted-V peak energy / 인버티드-V 최고 에너지로부터 평행 전위 강하 계산

Suppose FAST observes inverted-V electron precipitation peaked at $E_{\text{peak}} = 6$ keV, with a magnetospheric source temperature $k_B T_e = 1.5$ keV. Then the parallel potential drop sitting above the spacecraft is approximately:

$$
e\Phi_\parallel \approx E_{\text{peak}} - 1.5 \, k_B T_e = 6\,\text{keV} - 2.25\,\text{keV} \approx 3.75\,\text{keV}
$$

If at the same time the spacecraft sees an upgoing $H^+$ ion beam at $E_{\text{H}^+} = 4$ keV, this would imply $\Phi_\parallel^{\text{below}}$ contributes $\sim 4$ kV of additional drop located **below** the satellite, so the total altitude-integrated parallel potential along the field line is roughly $\Phi_\parallel^{\text{above}} + \Phi_\parallel^{\text{below}} \approx 3.75 + 4 = 7.75$ kV. Inverted-V arcs typically sit between 1 and 15 kV total.

가정: FAST가 자기권 원천 온도 $k_B T_e = 1.5$ keV에서 $E_{\text{peak}} = 6$ keV에 최대를 갖는 인버티드-V 전자 침강을 관측. 그러면 우주선 위의 평행 전위 강하는 위와 같이 약 3.75 kV. 동시에 우주선이 $E_{\text{H}^+} = 4$ keV의 상향 $H^+$ 이온 빔을 본다면, 이는 위성 **아래에** 추가 4 kV의 전위 강하가 있음을 의미하며, 자기장선을 따른 총 적분 평행 전위는 약 7.75 kV가 된다. 인버티드-V 호는 일반적으로 1~15 kV 사이에 있다.

### Worked example: AKR cutoff frequency and relativistic correction / AKR 절단 주파수와 상대론적 보정

At AKR source altitude (say $L \sim 6$ corresponding to $B \approx 0.18$ G $= 1.8 \times 10^{-5}$ T), the cold electron cyclotron frequency is

$$
f_{ce}^{\text{cold}} = \frac{eB}{2\pi m_e} = \frac{(1.6 \times 10^{-19})(1.8 \times 10^{-5})}{2\pi (9.11 \times 10^{-31})} \approx 504\,\text{kHz}
$$

For hot electrons with $W = 5$ keV, $\gamma - 1 = W/(m_e c^2) = 5/511 \approx 9.8 \times 10^{-3}$, giving $\gamma \approx 1.0098$. The AKR emission therefore appears at:

$$
f_{\text{AKR}} \approx f_{ce}/\gamma \approx 504/1.0098 \approx 499\,\text{kHz}
$$

That is, about 5 kHz (~1%) below the cold cyclotron frequency. FAST's combined wave-and-particle measurements demonstrated this match for the first time on the same flux tube.

AKR 원천 고도(예: $L \sim 6$, $B \approx 0.18$ G $= 1.8 \times 10^{-5}$ T)에서 차가운 전자 사이클로트론 주파수는 위와 같이 약 504 kHz. $W = 5$ keV 뜨거운 전자에서 $\gamma - 1 = 5/511 \approx 9.8 \times 10^{-3}$, $\gamma \approx 1.0098$. 따라서 AKR 방출은 약 499 kHz, 즉 차가운 사이클로트론 주파수 아래 약 5 kHz(~1%). FAST의 결합된 파동·입자 측정은 동일 자속관에서 처음으로 이 일치를 입증했다.

### Plasma parameter table for the auroral acceleration region / 오로라 가속 영역의 플라즈마 매개변수 표

| Parameter / 매개변수 | Typical value / 전형값 | Comment / 비고 |
|---|---|---|
| Altitude / 고도 | 1000-12000 km | FAST apogee 4175 km |
| Background $n_e$ / 배경 전자 밀도 | $0.1 - 10$ cm⁻³ | Density cavity: $< 0.1$ cm⁻³ |
| $T_e$ (hot) / 뜨거운 전자 온도 | $500$ eV $- 5$ keV | Magnetospheric source |
| $T_e$ (cold) / 차가운 전자 온도 | $0.1 - 1$ eV | Ionospheric origin |
| $|B|$ | $0.1 - 0.5$ G | $1-5 \times 10^{-5}$ T |
| Electron gyrofrequency $f_{ce}$ | $300 - 1500$ kHz | AKR band |
| Plasma frequency $f_{pe}$ | $\sim 30$ kHz at $n_e = 10$ cm⁻³ | Low in cavity |
| Debye length $\lambda_D$ | $5 - 50$ m | Sets electron-hole scale |
| Electron inertial length $c/\omega_{pe}$ | $\sim 1.6$ km at $n_e = 10$ cm⁻³ | Auroral arc width |
| Ion gyroradius (1 keV $H^+$) | $\sim 250$ m | Sub-arc-width |

Background plasma in the AAR transitions from cold and dense (ionospheric) at the bottom to hot and tenuous (magnetospheric) at the top. The crossover and especially density-cavity formation are the necessary conditions for parallel-E to appear.

AAR의 배경 플라즈마는 바닥에서 차갑고 조밀한(전리권) 상태로부터 위에서 뜨겁고 희박한(자기권) 상태로 전이한다. 그 교차 영역과 특히 밀도 공동 형성이 평행 E의 출현 조건이다.

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1958 ─ Van Allen 벨트 발견 / Van Allen belts discovered (Explorer 1)
1969 ─ McIlwain: ATS-5 첫 인버티드-V 관측 / ATS-5 first inverted-V
1972 ─ Gurnett & Frank: VLF saucer, 자기장선 전류 / VLF saucers
1976 ─ S3-3 발사: 첫 정전기 충격파 관측 / first electrostatic shocks
1978 ─ Iijima & Potemra: TRIAD Birkeland 전류 분류 / R-1, R-2 currents
1979 ─ Benson & Calvert: ISIS-1, AKR 밀도 공동 / AKR density cavities
1981 ─ Dynamics Explorer DE-1, DE-2 / DE
1983 ─ Mozer: S3-3 평행 전위 정량화 / parallel potentials quantified
1986 ─ Viking (스웨덴): 정전기 충격파 / electrostatic shocks
1986 ─ Temerin et al.: 점멸 오로라 EMIC 모델 / flickering aurora model
1989 ─ Akebono (일본): 평행 전기장 / parallel E
1992 ─ Freja (스웨덴): 1750 km 고분해능 / high resolution at 1750 km
1992 ─ Vago et al.: 사운딩 로켓 LH wave 이온 가속 / sounding rocket
1996 ─ ★ FAST 발사 (1996년 8월 21일) / FAST launch
1997 ─ International Auroral Study 캠페인 / January-March campaign
1998 ─ ★ 본 논문 + GRL 특별호 (전자 hole, 말발굽 분포)
2007 ─ THEMIS 임무 (5개 위성) / THEMIS 5 spacecraft
2015 ─ MMS (Magnetospheric Multiscale): 4개 위성 / MMS
2024+ ─ TRACERS (계획): 커스프 자기 재결합 / cusp reconnection
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Iijima & Potemra (1978) | Birkeland 전류 R-1/R-2 분류 — FAST의 상향/하향 전류 영역 구분의 토대 / R-1/R-2 framework underlying FAST's up/down classification | High — FAST's central organizing principle / FAST의 중심 조직 원리 |
| Benson & Calvert (1979), Persoon et al. (1988) | AKR 원천 영역의 밀도 공동 / AKR source-region density cavities | High — FAST quantitatively confirmed and refined / FAST가 정량적으로 확인·정교화 |
| Temerin, McFadden, Boehm, Carlson, Lotko (1986) | 점멸 오로라의 EMIC 변조 모델 / EMIC modulation model of flickering aurora | High — FAST 직접 in-situ로 확인 / directly confirmed in situ by FAST |
| Gurnett & Frank (1972) | VLF saucer와 자기장선 전류 / VLF saucers + field-aligned currents | Medium — downward-current region 연결 / linked to downward-current region |
| Marklund et al. (1994, Freja) | 검은 오로라의 발산 전기장 구조 / diverging E-fields & black aurora | Medium — FAST가 더 높은 고도까지 확장 / extended by FAST to higher altitudes |
| McFadden et al. (1987) | 사운딩 로켓 점멸 오로라 / sounding rocket flickering aurora | Medium — FAST's predecessor experimental work |
| Vago et al. (1992) | 사운딩 로켓 LH 파동 이온 가속 / lower-hybrid wave ion acceleration | Low-Medium — context for ion heating / 이온 가열 맥락 |
| McFadden et al. (1998a,b), Ergun et al. (1998a,b,c), Carlson et al. (1998) | 같은 GRL 특별호의 동반 논문 / companion papers in same special issue | High — primary follow-on results (이 논문의 후속 결과) |

---

## 7. References / 참고문헌

- Carlson, C. W., Pfaff, R. F., and Watzin, J. G., "The Fast Auroral SnapshoT (FAST) mission", *Geophys. Res. Lett.*, **25**(12), 2013-2016, 1998. DOI: 10.1029/98GL01592
- Benson, R. F., and W. Calvert, "ISIS 1 observations of ion energization and outflow in the high latitude magnetosphere", *Space Sci. Rev.*, **80**, 27-48, 1979.
- Carlson, C. W., et al., "FAST observations in the downward auroral current region: Energetic upgoing electron beams, parallel potential drops, and ion heating", *Geophys. Res. Lett.*, this issue, 1998.
- Cattell, C., et al., "The association of electrostatic ion cyclotron waves, ion and electron beams and field-aligned current: FAST observations", *Geophys. Res. Lett.*, this issue, 1998.
- Chaston, C. C., et al., "Characteristics of electromagnetic proton cyclotron waves along auroral field lines observed by FAST in regions of upward current", *Geophys. Res. Lett.*, this issue, 1998.
- Delory, G. T., et al., "FAST observations of electron distributions within AKR source regions", *Geophys. Res. Lett.*, this issue, 1998.
- Elphic, R. C., et al., "The auroral current circuit and field-aligned currents observed by FAST", *Geophys. Res. Lett.*, this issue, 1998.
- Ergun, R. E., et al., "FAST satellite observations of electric field structures in the auroral zone", *Geophys. Res. Lett.*, this issue, 1998a.
- Ergun, R. E., et al., "FAST satellite wave observations in the AKR source region", *Geophys. Res. Lett.*, this issue, 1998b.
- Ergun, R. E., et al., "FAST satellite observations of large-amplitude solitary structures", *Geophys. Res. Lett.*, this issue, 1998c.
- Gurnett, D. A., and L. A. Frank, "VLF hiss and related plasma observations in the polar magnetosphere", *J. Geophys. Res.*, **77**, 172-190, 1972.
- Iijima, T., and T. A. Potemra, "Large-scale characteristics of field-aligned currents associated with substorms", *J. Geophys. Res.*, **83**, 599, 1978.
- Kistler, L. M., et al., "FAST/TEAMS observations of charge exchange signatures in ions mirroring at low altitudes", *Geophys. Res. Lett.*, this issue, 1998.
- Lund, E. J., et al., "FAST observations of preferentially accelerated He+ in association with auroral electromagnetic ion cyclotron waves", *Geophys. Res. Lett.*, this issue, 1998.
- Lundin, R., et al., references to Freja results, 1998.
- Marklund, G., L. Blomberg, C. G. Falthammar, P. A. Lindqvist, "On intense diverging electric fields associated with black aurora", *Geophys. Res. Lett.*, **21**, 1859, 1994.
- McFadden, J. P., C. W. Carlson, M. H. Boehm, T. J. Halliman, "Field-aligned electron flux oscillations that produce flickering aurora", *J. Geophys. Res.*, **92**, 11133, 1987.
- McFadden, J. P., et al., "Electron modulation and ion cyclotron waves observed by FAST", *Geophys. Res. Lett.*, this issue, 1998a.
- McFadden, J. P., et al., "Spatial structure and gradients of ion beams observed by FAST", *Geophys. Res. Lett.*, this issue, 1998b.
- Möbius, E., et al., "Species dependent energies in upward directed ion beams over auroral arcs as observed with FAST TEAMS", *Geophys. Res. Lett.*, this issue, 1998.
- Persoon, A. M., et al., "Electron density depletions in the nightside auroral zone", *J. Geophys. Res.*, **93**, 1871-1895, 1988.
- Peterson, W. K., et al., "Simultaneous observations of solar wind plasma entry from FAST and POLAR", *Geophys. Res. Lett.*, this issue, 1998.
- Pfaff, R., et al., "Initial FAST satellite observations of acceleration processes in the cusp", *Geophys. Res. Lett.*, this issue, 1998.
- Sigsbee, K., et al., "FAST-Geotail correlative studies of magnetosphere-ionosphere coupling in the nightside magnetosphere", *Geophys. Res. Lett.*, this issue, 1998.
- Stenbaek-Nielsen, H. C., et al., "Aircraft observations conjugate to FAST auroral arc thicknesses", *Geophys. Res. Lett.*, this issue, 1998.
- Strangeway, R. J., et al., "FAST observations of VLF waves in the auroral zone: evidence of very low plasma densities", *Geophys. Res. Lett.*, this issue, 1998.
- Temerin, M., J. P. McFadden, M. Boehm, C. W. Carlson, W. Lotko, "Production of flickering aurora and field-aligned electron fluxes by electromagnetic ion cyclotron waves", *J. Geophys. Res.*, **91**, 5769, 1986.
- Vago, J. L., et al., "Transverse ion acceleration by lower hybrid waves in the topside auroral ionosphere", *J. Geophys. Res.*, **97**, 16935-16957, 1992.
