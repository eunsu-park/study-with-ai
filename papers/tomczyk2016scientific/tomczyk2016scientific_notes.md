---
title: "Scientific Objectives and Capabilities of the Coronal Solar Magnetism Observatory"
authors: Steven Tomczyk, Enrico Landi, Joan T. Burkepile, Roberto Casini, Edward E. DeLuca, Yuhong Fan, Sarah E. Gibson, Haosheng Lin, Scott W. McIntosh, Stanley C. Solomon, Giuliana de Toma, Alfred G. de Wijn, Jie Zhang
year: 2016
journal: "Journal of Geophysical Research: Space Physics"
doi: "10.1002/2016JA022871"
topic: Solar Observation
tags: [COSMO, coronagraph, coronal magnetic field, K-Cor, ChroMag, Large Coronagraph, spectropolarimetry, CME, space weather, synoptic observation]
status: completed
date_started: 2026-04-16
date_completed: 2026-04-16
---

# 7. Scientific Objectives and Capabilities of the Coronal Solar Magnetism Observatory / 코로나 태양 자기장 관측소의 과학적 목표와 역량

---

## 1. Core Contribution / 핵심 기여

이 논문은 HAO/NCAR에서 개발 중인 **COSMO(Coronal Solar Magnetism Observatory)**의 과학적 목표, 측정 요구사항, 기기 사양을 종합적으로 기술한다. COSMO는 세 가지 시놉틱(synoptic) 기기—K-Coronagraph(K-Cor), Chromosphere and Prominence Magnetometer(ChroMag), Large Coronagraph(LC)—로 구성되며, 코로나와 채층의 자기장, 밀도, 온도, 속도를 일상적으로 측정하는 것을 목표로 한다. 논문은 네 가지 핵심 과학 질문—(1) 프로미넌스 분출과 CME 개시의 자기적 진화, (2) 코로나의 자기/열역학 구조와 태양 주기 변화, (3) 코로나 가열과 태양풍 가속, (4) 우주 날씨 예보 개선—을 제시하고, 각 질문에 대응하는 측정 요구사항과 기기 사양을 Table 1의 과학 추적성 매트릭스(Science Traceability Matrix)로 체계화한다. COSMO는 DKIST의 고분해능·소시야 관측을 넓은 시야각의 시놉틱 관측으로 보완하며, 우주 기반 관측소(SOHO, SDO, STEREO)와도 상보적 역할을 수행하도록 설계되었다.

This paper comprehensively describes the scientific objectives, measurement requirements, and instrument specifications of **COSMO (Coronal Solar Magnetism Observatory)**, under development at HAO/NCAR. COSMO comprises three synoptic instruments—K-Coronagraph (K-Cor), Chromosphere and Prominence Magnetometer (ChroMag), and Large Coronagraph (LC)—aimed at routine measurements of the magnetic field, density, temperature, and velocity of the corona and chromosphere. The paper poses four key science questions—(1) the magnetic evolution of prominence eruption and CME initiation, (2) the magnetic and thermodynamic structure of the corona and its solar-cycle evolution, (3) coronal heating and solar wind acceleration, and (4) improving space weather prediction—and maps each to measurement requirements and instrument specifications through the Science Traceability Matrix (Table 1). COSMO is designed to complement DKIST's high-resolution, small-FOV observations with wide-FOV synoptic coverage, and to play a complementary role to space-based observatories (SOHO, SDO, STEREO).

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction (§1, pp. 7470–7471)

논문은 태양 자기장이 광구에서 코로나로 갈수록 측정이 어려워지는 역설적 상황을 제시한다. 광구 자기장은 Zeeman 효과를 이용하여 일상적으로 측정되지만(GONG, HMI 등), 코로나에서는 자기장이 약하고(~1–10 G) 열적 선폭이 넓어 Zeeman 분리가 사실상 감지 불가능하다. 대부분의 코로나 자기장 정보는 광구 측정에서의 외삽(extrapolation)에 의존하지만, 코로나의 복잡한 비선형 동역학 때문에 외삽의 신뢰도는 본질적으로 제한적이다.

The paper opens with the paradox that magnetic field measurements become increasingly difficult from the photosphere to the corona. Photospheric fields are routinely measured via the Zeeman effect (GONG, HMI, etc.), but coronal fields are weak (~1–10 G) with broad thermal line widths, making Zeeman splitting virtually undetectable. Most coronal magnetic field information relies on extrapolation from photospheric measurements, but the complex nonlinear dynamics of the corona fundamentally limit extrapolation reliability.

COSMO는 이 간극을 메우기 위해 제안된 시설로, 태양 대기를 "결합 시스템(coupled system)"으로 연구하기 위한 기기 모음(suite)을 제공한다.

COSMO is proposed to fill this gap, providing a suite of instruments to study the solar atmosphere as a "coupled system."

### Part II: Scientific Objectives (§2, pp. 7471–7474)

논문은 네 가지 핵심 과학 질문을 제시한다:

The paper poses four key science questions:

**§2.1 프로미넌스 분출과 CME 개시의 자기적/플라즈마 진화 (Prominence Eruption & CME Initiation)**

이론 모델과 수치 시뮬레이션은 활동 영역의 이상화된 자기장이 불안정해져 분출할 수 있음을 보여주지만(Hood and Priest, 1981; Antiochos et al., 1999; Török et al., 2004; Fan, 2011), 트리거 메커니즘을 관측으로 제약할 수 없다. 광구 자기장 관측에 기반한 예측 능력은 통계적 수준에 머물며, 특히 미래 사건의 시기 예측에는 신뢰성이 부족하다. CME는 3D 전류 채널(magnetic flux rope)과 관련이 있으며, 축 자기장을 감싸는 나선형 자기장 구조를 가지지만, 코로나 자기장 직접 관측 없이는 이 구조의 형성·진화·분출 과정을 규명할 수 없다.

Theoretical models and numerical simulations show that idealized active region magnetic fields can become unstable and erupt, but the trigger mechanism cannot be constrained observationally. Predictive capability from photospheric field observations remains statistical, especially for timing future events. CMEs are associated with 3D current channels (magnetic flux ropes) with helical field lines, but without direct coronal magnetic field observations, the formation, evolution, and eruption processes of these structures cannot be determined.

프로미넌스 자기장 관측은 희귀하며(Bommier et al., 1994; Casini et al., 2003; Orozco Suarez et al., 2014), 최근 코로나 공동(coronal cavity)의 선편광·시선 속도 측정이 flux rope 모델과 일치하는 자기 배열을 보여주었지만(Bak-Steslicka et al., 2013), 시선 방향 자기장 측정이 필요하다(Rachmeler et al., 2013).

Prominence magnetic field observations are rare. Recent coronal cavity linear polarization and LOS velocity measurements have indicated magnetic configurations consistent with flux rope models, but LOS magnetic field measurements are needed.

**§2.2 코로나의 자기/열역학 구조와 태양 주기 변화 (Coronal Structure & Solar Cycle Evolution)**

태양 코로나는 모든 규모에서 자기장에 의해 구조화되며, 태양 주기에 따라 극소기의 쌍극자 형태에서 극대기의 복잡한 자기 구조로 변화한다. 현재 코로나 구조 예측은 광구 측정의 외삽에 의존하지만, 채층의 높은 동역학성과 코로나의 복잡한 전류 시스템이 외삽의 유용성을 제한한다(DeRosa et al., 2009). 코로나 자기장 시선 방향 성분의 일별 태양 전면 지도(daily solar coronal maps of LOS magnetic field)를 백색광 코로나 편광 밝기와 결합하면, 태양 쌍극자 진화, 남-북 반구 비대칭, 코로나-태양풍 경계 등의 장기 변화를 추적할 수 있다.

The solar corona is structured by magnetic fields at all scales and varies from dipole-like at solar minimum to complex configurations at maximum. Current predictions rely on extrapolation, but the highly dynamic chromosphere and complex current systems limit its utility. Combining daily coronal LOS magnetic field maps with white-light pB would enable tracking of long-term evolution including solar dipole evolution, hemispheric asymmetries, and the corona-solar wind interface.

자기 헬리시티(magnetic helicity) 문제도 제기된다: CME가 헬리시티의 중요한 sink라면 태양 주기적 다이나모 메커니즘에서 핵심 역할을 하므로, 코로나 자기장과 백색광 코로나를 결합한 측정이 전 지구 자기장의 태양 주기 변화를 추적하는 유일한 방법이 될 수 있다.

The question of magnetic helicity evolution is raised: if CMEs are important helicity sinks, combined coronal magnetic field and white-light measurements may provide unique tracking of solar-cycle variation of the global magnetic field.

**§2.3 코로나 가열과 태양풍 가속 (Coronal Heating & Solar Wind Acceleration)**

코로나 가열은 태양 물리학의 주요 미해결 문제 중 하나이다. 나노플레어 가열(nanoflare heating)과 파동 구동 가열(wave-driven heating) 두 이론이 관측적 지지를 받고 있다(Klimchuk, 2006; Aschwanden et al., 2007). 최근 관측은 파동이 채층과 코로나에 편재함을 보여주었다(De Pontieu et al., 2007; Tomczyk et al., 2007; Morton and McLaughlin, 2013; Morton et al., 2015). 코로나 가열과 태양풍 가속이 태양 표면 근처에서 일어나므로, 채층과 저 코로나 관측이 핵심이다.

Coronal heating remains one of the leading open questions. Both nanoflare and wave-driven heating theories have observational support. Recent observations have shown ubiquitous waves in the chromosphere and corona. Since heating and acceleration occur close to the solar surface, chromospheric and low-corona observations are critical.

백색광 코로나 영상으로 전자 밀도의 3D 분포를 제약할 수 있고, 다중 온도 방출선 관측을 통해 코로나 가열의 진화를 추적할 수 있다. COSMO LC 관측은 EUV 관측과 매우 상보적이다: EUV 방출선은 충돌 과정(밀도 제곱에 비례)에 의해 결정되어 고밀도 영역에 편향되는 반면, 가시광/근적외선 코로나 금지선은 복사 과정(밀도에 선형 비례)에 의해 림 위에서도 더 높은 고도까지 관측 가능하다.

White-light coronal images can constrain the 3D electron density distribution, and multi-temperature emission line observations can track the evolution of coronal heating. COSMO LC observations are highly complementary to EUV: EUV emission lines are determined by collisional processes (proportional to density squared), biased toward high-density regions, while visible/near-IR forbidden lines are dominated by radiative processes (linear in density) and observable to greater heights above the limb.

**§2.4 우주 날씨 예보 개선 (Space Weather Prediction)**

우주 날씨 예보에서 코로나 물리학의 가장 중요한 응용 중 하나이다. 플레어와 CME는 광구 자기장에 뚜렷한 전조가 없는 경우가 많아, 코로나 자기장 직접 관측이 필요하다. 현재 우주 날씨 모델은 광구 자기장 외삽과 코로나 자유 에너지·3D 토폴로지의 추정에 의존하며, 관측으로 충분히 제약되지 않는다.

Space weather prediction is one of the most important applications. Flares and CMEs often show no obvious precursors in photospheric magnetic fields, necessitating direct coronal field observations. Current models rely on photospheric extrapolation and estimates of coronal free energy and 3D topology that are insufficiently constrained by observations.

COSMO 데이터는 FORWARD 코드(Gibson, 2015b)—CHIANTI 데이터베이스를 이용하여 모든 COSMO LC 선과 K-Cor 백색광 데이터를 합성할 수 있는 코드—와 SWMF(Space Weather Modeling Framework) 모델을 구동·검증하는 데 사용될 것이다.

COSMO data will be used to drive and validate the FORWARD code (which synthesizes all COSMO LC lines and K-Cor white-light data using the CHIANTI database) and the SWMF model.

### Part III: Required Measurements (§3, pp. 7474–7475)

**Table 1: Science Traceability Matrix**는 논문의 핵심 구조를 보여준다. 네 가지 과학 목표를 측정 요구사항과 기기 사양으로 체계적으로 연결한다:

**Table 1: Science Traceability Matrix** is the structural backbone of the paper, systematically linking four science objectives to measurement requirements and instrument specifications:

핵심 측정 요구사항:
- 코로나 자기장: 1 G B 감도, 15분 내, 2″ 공간 분해능, 넓은 FOV, 시놉틱
- 코로나 전자 밀도: pB 측정, 15초 케이던스, 넓은 FOV, 시놉틱
- 코로나/채층 온도·밀도: 다중 온도 방출선, 밀도 민감 선 쌍
- 채층/프로미넌스 자기장: 자기 민감 채층선, 1 G B 감도, 넓은 FOV

Key measurement requirements:
- Coronal magnetic field: 1 G B sensitivity, within 15 min, 2″ spatial resolution, large FOV, synoptic
- Coronal electron density: pB measurement, 15 s cadence, large FOV, synoptic
- Coronal/chromospheric temperature and density: multi-temperature emission lines, density-sensitive line pairs
- Chromospheric/prominence magnetic field: magnetically sensitive chromospheric lines, 1 G B sensitivity, large FOV

시야각은 대형 코로나 루프, 프로미넌스, 공동, 스트리머 시스템, CME 추적이 가능해야 하므로 코로나에 1°, 채층에는 다소 작은 FOV가 필요하다. 공간 분해능(2″)과 FOV 사이의 트레이드오프가 존재하며, 4 m DKIST가 최고 분해능을 담당하는 반면 COSMO는 넓은 FOV 시놉틱 관측에 최적화된다.

The FOV must encompass large coronal loops, prominences, cavities, streamer systems, and CME tracking: 1° for corona, somewhat smaller for chromosphere. A tradeoff exists between spatial resolution (2″) and FOV, with 4 m DKIST handling highest resolution while COSMO is optimized for wide-FOV synoptic observations.

### Part IV: The COSMO Suite of Instruments (§4, pp. 7475–7479)

**§4.1 Large Coronagraph (LC)**

COSMO LC는 논문의 핵심 기기이다. 설계 사양:

The COSMO LC is the flagship instrument. Design specifications:

| 사양 / Specification | 값 / Value |
|---|---|
| 형식 / Type | 내부 차폐 Lyot 코로나그래프 / Internally occulted Lyot coronagraph |
| 구경 / Aperture | 1.5 m (렌즈 / lens) |
| 시야각 / FOV | 1° (1.05–2 $R_\odot$) |
| 공간 분해능 / Spatial resolution | 2″ (3″ 사양 / 3″ specified) |
| 분광 분해능 / Spectral resolution | $\lambda/\Delta\lambda > 8000$ |
| 파장 범위 / Wavelength range | 500–1100 nm |
| 코로나 자기장 감도 / Coronal B sensitivity | 1 G, 15분 내 / within 15 min |
| 강도/Doppler 케이던스 / Intensity/Doppler cadence | 1초 / 1 s |
| 편광 케이던스 / Polarimetric cadence | 30초 / 30 s |

자기장 감도 오차 추정 (Eq. 1):

Magnetic field sensitivity error estimate (Eq. 1):

$$\sigma_B = \frac{16.5}{\sqrt{N}} \left(1 + 2\frac{B}{N}\right)^{1/2} \quad \text{(kG)}$$

여기서 $N$은 방출선에 걸쳐 적분한 광자 수, $B$는 같은 파장 구간의 배경 광자 수이다. Fe XIII 1074.7 nm 선을 가정하면 이 선이 가장 높은 신호 대 잡음비를 가진다. 배경이 없는 이상적 경우($B = 0$)에도 1 G 정밀도를 달성하려면 $2.7 \times 10^8$개의 광자가 필요하다. 1.5 m 구경, 10 ppm 코로나 밝기, 5 ppm 배경, 이중 빔(dual-beam) 편광계(효율 0.57)를 가정하면 15분 내 1 G 감도를 달성할 수 있다.

Where $N$ is the number of photons integrated over the emission line, and $B$ is the number of background photons over the same wavelength interval. Assuming Fe XIII 1074.7 nm (highest signal-to-noise ratio), even with no background ($B = 0$), $2.7 \times 10^8$ photons are needed for 1 G precision. With 1.5 m aperture, 10 ppm coronal brightness, 5 ppm background, and a dual-beam polarimeter (efficiency 0.57), 1 G sensitivity is achievable within 15 minutes.

렌즈를 선택한 이유: 산란광 분석에 따르면 렌즈가 거울보다 표면 거칠기와 먼지 오염에 의한 산란광이 현저히 적다(Nelson et al., 2008). 먼지가 주요 산란원이 될 것이므로, COSMO LC 돔은 최소 구경으로 여압(pressurized)하고 여과된 공기를 사용한다.

Reason for choosing a lens: scattered-light analysis shows lenses scatter significantly less light than mirrors from surface roughness and dust contamination. Since dust will be the dominant scatter source, the LC dome will have a minimally sized aperture and be pressurized with filtered air.

핵심 기술은 **후초점 가변 필터(postfocus tunable filter)**이다. 1° FOV에 걸쳐 500–1100 nm에서 분광 분해능 > 8000을 제공해야 한다. Lithium Niobate 결정을 이용한 광시야 복굴절 필터(wide-field birefringent filter)가 étendue 요구사항을 충족한다(Tomczyk et al., 2016 별도 논문).

The key enabling technology is the **postfocus tunable filter**, which must provide spectral resolution > 8000 over 500–1100 nm across a 1° FOV. A wide-field birefringent filter using Lithium Niobate crystals meets the étendue requirement.

편광 분석은 회전 파장판 + 편광 빔 분리기(rotating waveplate + polarizing beamsplitter)로 수행된다. 이중 빔 기법으로 직교 편광 상태를 동시 측정하여 시상(seeing) 유도 잡음을 제거한다.

Polarization analysis uses a rotating waveplate followed by a polarizing beamsplitter. The dual-beam technique simultaneously measures orthogonal polarization states to eliminate seeing-induced noise.

**Table 2: 관측 가능한 가시광/근적외선 선 목록**

COSMO LC로 관측할 후보 방출선은 CME 핵심, CME 고온 성분, 정적 코로나의 세 범주로 분류된다:

Candidate emission lines are categorized into CME core, CME hot component, and quiescent corona:

- CME 핵심: Hα 656.3, HeI 587.6, HeI 1083.0, CaII 854.2, OII 732.1, OII 733.2, OIII 500.8, FeVI 520.0
- CME 고온 성분: FeXIV 530.3, FeXV 706.2, SXI 761.1, ArXIII 810.0, ArXII 1014.3, CaXV 544.5, CaXV 569.4
- 정적 코로나: FeX 637.5, FeXI 789.2, **FeXIII 1074.7** ($N_e$), **FeXIII 1079.8** ($N_e$), FeXIV 530.3, ArX 552.2, ArXI 691.8

밀도 민감 선 쌍: 같은 이온의 두 선 비율이 전자 밀도에 민감한 경우 ($N_e$)로 표시. Fe XIII 1074.7/1079.8 nm 쌍이 가장 중요하다.

Density-sensitive line pairs: marked with ($N_e$) when the ratio of two lines from the same ion is sensitive to electron density. The Fe XIII 1074.7/1079.8 nm pair is the most important.

**§4.2 Chromosphere and Prominence Magnetometer (ChroMag)**

ChroMag 사양:

| 사양 / Specification | 값 / Value |
|---|---|
| 구경 / Aperture | 15 cm (doublet objective lens) |
| 시야각 / FOV | 2.5 $R_\odot$ (전일면 + 림 위 / full disk + above limb) |
| 공간 분해능 / Spatial resolution | 2″ |
| 관측 파장 / Spectral lines | HeI 587.6 & 1083.0 nm (프로미넌스), Hα 656.3 nm, CaII 854.2 nm (채층), FeI 617.3 nm (광구) |
| 필터 대역폭 / Filter bandwidth | 0.025 nm (가시광) ~ 0.046 nm (적외선) |
| 편광 감도 / Polarimetric sensitivity | $10^{-3}$, 1분 이내/선 / within 1 min per line |
| 강도/Doppler 케이던스 / Intensity/Doppler cadence | 10초/선 / 10 s per line |

가변 필터(tunable filter)는 광시야 Lyot 복굴절 필터로, 6단(stage)으로 구성된다. 각 단은 입사 편광기, 방해석 파장판, 또 다른 방해석 파장판, 네마틱 액정 가변 지연판, 출사 편광기로 이루어진다. 방해석 결정의 두께를 단별로 2배씩 늘려가며, 가장 두꺼운 단은 44 mm이다. 파장 튜닝은 액정에 인가하는 전압 변경으로 수행한다.

The tunable filter is a wide-field Lyot birefringent filter with 6 stages. Each stage consists of an entrance polarizer, calcite waveplate, another calcite waveplate, nematic liquid crystal variable retarder, and exit polarizer. The calcite crystal thickness doubles for each adjacent stage, with the thickest at 44 mm. Wavelength tuning is accomplished by changing the voltage applied to the liquid crystals.

ChroMag 편광계는 가변 필터 전단에 위치하며, 두 개의 강유전 액정과 선형 편광기로 구성된다. 가변 필터의 첫 번째 편광기가 분석기(analyzer) 역할을 한다. 587–1083 nm에서 full Stokes 분석이 가능하다.

The ChroMag polarimeter precedes the tunable filter, consisting of two ferroelectric liquid crystals and a linear polarizer. The first polarizer of the tunable filter acts as the analyzer. Full Stokes analysis is possible over 587–1083 nm.

채층 음속이 광구보다 ~10배 빠르므로 광구보다 높은 케이던스가 필요하다. 10초 케이던스는 채층과 코로나의 MHD 파동 관측에 충분하다.

Since the chromospheric sound speed is ~10× faster than in the photosphere, higher cadence than photospheric measurements is needed. A 10 s cadence is sufficient for observing MHD waves in the chromosphere and corona.

**§4.3 K-Coronagraph (K-Cor)**

K-Cor는 COSMO 기기 중 유일하게 **이미 운용 중**인 기기이다(2013년 9월 Mauna Loa 배치). 사양:

K-Cor is the only COSMO instrument **already operational** (deployed to Mauna Loa September 2013). Specifications:

| 사양 / Specification | 값 / Value |
|---|---|
| 구경 / Aperture | 20 cm |
| 형식 / Type | 내부 차폐 코로나그래프 / Internally occulted coronagraph |
| 시야각 / FOV | 1.05–3 $R_\odot$ |
| 공간 분해능 / Spatial resolution | 6″/pixel |
| 관측 파장 / Wavelength | 735 nm (35 nm 대역폭 / bandwidth) |
| 케이던스 / Cadence | 15초 / 15 s |
| 감도 / Sensitivity | $10^{-9}$ $B_\odot$ |

K-Cor는 Thomson 산란 편광(pB)을 관측하여 코로나 전자 기둥 밀도(column density)를 측정한다. 이중 빔 광학계로 직교 선편광 상태를 동시 촬영하여 시상과 강도 변화에 의한 편광 측정 잡음을 상쇄한다. 초연마 대물렌즈(superpolished objective lens)와 HEPA 여과 먼지 제어로 기기 산란광을 5 ppm 이하로 억제한다.

K-Cor observes Thomson-scattered polarization brightness (pB) to measure coronal electron column density. The dual-beam optical system simultaneously images orthogonal linear polarization states to cancel polarization measurement noise from seeing and intensity variations. A superpolished objective lens and HEPA-filtered dust control suppress instrumental scattering to below 5 ppm.

K-Cor의 시야각 하한은 1.05 $R_\odot$으로, 대부분의 CME가 발생·가속되는 **최저 코로나 스케일 하이트**의 최초 일상 백색광 관측을 제공한다. 15초 케이던스는 CME 개시, 프로미넌스 분출/회전, 자기 재결합, 충격파 전파 등의 동역학적 과정을 추적하기에 충분하다.

The lower limit of 1.05 $R_\odot$ provides the first routine white-light measurements of the **lowest coronal scale height** where most CMEs originate and are accelerated. The 15 s cadence is sufficient to follow dynamical processes such as CME initiation, prominence eruption/rotation, magnetic reconnection, and shock propagation.

Figure 5는 K-Cor의 핵심 역할을 보여준다: 2014년 10월 14일 CME에서 K-Cor(파란색)는 AIA(금색) 영상과 LASCO C2(빨간색) 영상 사이의 공백을 메우며, LASCO보다 ~20분 먼저 CME를 감지했다.

Figure 5 demonstrates K-Cor's critical role: for the October 14, 2014 CME, K-Cor (blue) fills the gap between AIA (gold) and LASCO C2 (red), detecting the CME ~20 minutes before LASCO.

### Part V: COSMO Uniqueness and Complementarity (§5, pp. 7479–7481)

**§5.1 Large Coronagraph의 고유성**

EUV 영상기(AIA 등)는 좁은 대역폭 필터를 사용하므로 Doppler 이동·선폭·편광 측정이 불가능하고, Zeeman 분리도 EUV 파장에서는 감지 불가능하다. EUV 방출선은 충돌 과정($n_e^2$)에 의해 결정되어 높은 고도에서 급격히 감소하는 반면, 가시광/근적외선 금지선은 복사 과정(photospheric radiation field에 의해 여기)에 의해 더 높은 고도까지 관측 가능하다. COSMO LC는 EUV 관측에 대해 방출선 강도, 분광 정보, 편광 분석을 1° FOV에서 고케이던스로 제공하여 매우 상보적이다.

EUV imagers (AIA, etc.) use narrowband filters, precluding Doppler shift, line width, and polarization measurements. Zeeman splitting is also undetectable at EUV wavelengths. EUV emission is determined by collisional processes ($n_e^2$) and decreases steeply with height, while visible/near-IR forbidden lines are excited by the photospheric radiation field and observable to greater heights. COSMO LC is highly complementary to EUV observations.

**DKIST와의 비교**: DKIST는 4 m 구경으로 집광 면적이 7배 크지만, FOV는 최대 5′(COSMO LC의 1°에 비해 매우 작음). COSMO LC의 집광력(aperture × solid angle of FOV)은 DKIST의 20배이다. DKIST는 사용자 주도(user-driven) 고분해능 관측에, COSMO LC는 시놉틱·넓은 FOV 관측에 최적화되어 상보적이다.

**Comparison with DKIST**: DKIST has 4 m aperture with 7× greater collecting area, but FOV is only 5′ (vs. COSMO LC's 1°). The LC's light-gathering power (aperture × solid angle of FOV) exceeds DKIST by a factor of 20. DKIST is optimized for user-driven high-resolution observations; COSMO LC for synoptic wide-FOV observations.

전파(radio) 관측도 코로나 자기장 정보를 제공할 수 있다: 자이로 공명(gyroresonance, >100 G의 강한 자기장에 민감)과 제동 복사(bremsstrahlung)의 편광, Faraday 회전 등. 그러나 온도, 밀도, 채움 인자, 자기장 기하에 대한 서로 다른 의존성 때문에 가시광/적외선 관측과 상보적이다.

Radio observations can also provide coronal magnetic field information: gyroresonance (sensitive to strong fields >100 G), bremsstrahlung polarization, and Faraday rotation. However, different dependences on temperature, density, filling factor, and field geometry make them complementary to visible/IR observations.

**§5.2 ChroMag의 고유성**

SOLIS/VSM은 CaII 854.2 nm와 HeI 1083.0 nm에서 채층 관측을 제공하지만, 슬릿 스캐닝 방식이라 케이던스가 느리고 플레어 같은 급변 현상 관측에 부적합하다. IRIS는 2초 시간 분해능으로 채층선을 관측하지만 FOV가 175″×175″로 작고 편광 측정이 불가하다. Dunn Solar Telescope의 IBIS와 SST의 CRIsp는 ChroMag과 유사한 역량을 가지지만 시놉틱 기기가 아니다. ChroMag은 넓은 FOV, Doppler·편광 영상, 높은 케이던스를 결합한 유일한 시놉틱 기기이다.

SOLIS/VSM provides chromospheric observations in CaII 854.2 nm and HeI 1083.0 nm but uses slit scanning with slow cadence, unsuitable for transient events. IRIS has 2 s temporal resolution but a small 175″×175″ FOV with no polarimetry. IBIS and CRIsp have similar capabilities but are not synoptic instruments. ChroMag is unique as a synoptic instrument combining wide FOV, Doppler and polarization imaging, and high cadence.

**§5.3 K-Cor의 고유성**

K-Cor는 유일한 지상 백색광 코로나그래프이자 1.5 $R_\odot$ 이하 코로나의 유일한 백색광 관측 소스이다. 우주 기반 LASCO(1.5 $R_\odot$ 이상)와 STEREO COR1(1.5 $R_\odot$ 이상)이 관측하지 못하는 저 코로나 영역을 채우며, EUV/X선 관측과 외부 코로나 백색광 관측 사이의 공백을 메운다.

K-Cor is the only ground-based white-light coronagraph and the sole source of white-light observations below 1.5 $R_\odot$. It fills the gap between space-based EUV/X-ray observations and outer-corona white-light observations from LASCO and STEREO.

### Part VI: COSMO Enabling Science (§6, pp. 7481–7483)

COSMO 기기 모음은 시너지 효과를 통해 과학 목표를 달성한다:

The COSMO suite works in synergy to address science objectives:

1. **CME 연구**: 코로나와 프로미넌스의 자기장 측정으로 CME 전구체(filament channels, prominence-cavity systems)의 자기 구조와 토폴로지를 제약하고, 분출 중 동적 특성을 측정. 코로나 방출선으로 0.01–5 MK 범위의 CME 열 에너지 수지(thermal energy budget)를 완전히 결정.

   CME studies: constrain magnetic structure and topology of precursors, measure dynamic properties during onset. Fully resolve CME thermal energy budget from 0.01–5 MK using coronal emission lines.

2. **Alfvén 파와 코로나 가열**: CoMP의 코로나 MHD 파동 측정을 더 나은 시간·공간 분해능과 넓은 온도 범위의 방출선으로 개선. ChroMag의 채층 파동 관측과 결합하여 에너지 결합(energy coupling)을 연구. 파동의 횡파 성분(선편광)이 Zeeman 측정(원편광)과 상보적으로 코로나 자기장 벡터를 완성.

   Alfvén waves and coronal heating: improve CoMP's MHD wave measurements with better temporal/spatial resolution and wider temperature range. Combine with ChroMag chromospheric wave observations to study energy coupling.

3. **우주 날씨 모델 구동**: FORWARD 코드와 SWMF/AWSoM 모델에 관측 데이터를 직접 제공. AWSoM은 현재 관측적 제약이 없는 경계 조건(상부 채층 Alfvén 파 에너지, 태양 표면 밀도 등)을 사용하는데, COSMO가 이 경계 조건의 현실적 제약을 제공.

   Space weather model driving: provide observational data directly to FORWARD code and SWMF/AWSoM model. COSMO will provide realistic constraints on boundary conditions currently lacking observational constraints.

### Part VII: COSMO Implementation and Status (§7, p. 7483)

COSMO는 NCAR, University of Michigan, University of Hawaii, George Mason University, Harvard-Smithsonian CfA 등의 파트너 기관과 산업체가 공동으로 설계·제작한다. 관측 부지는 Haleakala와 Mauna Loa가 평가되었으며, Mauna Loa가 약간 선호된다.

COSMO is designed and built by partner institutions including NCAR, U. Michigan, U. Hawaii, George Mason U., and Harvard-Smithsonian CfA. Mauna Loa is slightly preferred over Haleakala for the observing site.

기기 배치 계획:
- **LC**: 별도의 태양 추적 스파(solar pointed spar)의 돔 내에 설치. 스파에 8면이 있어 커뮤니티 개발 기기도 배치 가능.
- **K-Cor & ChroMag**: 인접한 작은 돔의 태양 추적 플랫폼에 설치.

Instrument deployment plan:
- **LC**: housed in a dome on a separate solar pointed spar, with 8 sides available for community instruments.
- **K-Cor & ChroMag**: on a solar pointed platform in a nearby smaller dome.

2016년 기준 진행 상황:
- K-Cor: 2013년부터 운용 중
- LC: 예비 설계 완료, 대형 굴절 코로나그래프 제작 가능성 확인 (공학 연구, 설계, 벤더 견적 완료)
- CoMP → UCoMP 업그레이드 진행 중 (1°로 FOV 확대, 5단 Lithium Niobate 가변 필터 — LC 기술 시연)
- ChroMag: 2016년 프로토타입 예정

Status as of 2016:
- K-Cor: operational since 2013
- LC: preliminary design phase complete, feasibility of large refractive coronagraph documented
- CoMP → UCoMP upgrade underway (FOV expanded to 1°, 5-stage LiNbO₃ tunable filter — LC technology demonstration)
- ChroMag: prototype expected in 2016

### Part VIII: Summary (§8, pp. 7483–7484)

COSMO는 코로나와 채층의 자기장, 밀도, 온도, 속도 관측의 고유한 조합을 제공하여 태양 활동과 우주 날씨 발생 과정에 대한 이해를 변혁할 것이다. COSMO의 넓은 FOV 시놉틱 관측은 DKIST의 고분해능·소시야 관측과, 그리고 전파 관측 기법과 상보적이다. COSMO는 2013년 최신 Heliophysics Decadal Survey에서 높은 우선순위로 지지받았다(Baker et al., 2013).

COSMO will provide a unique combination of coronal and chromospheric magnetic field, density, temperature, and velocity observations that will transform understanding of solar activity and space weather generation. Its wide-FOV synoptic observations complement DKIST's high-resolution observations and radio techniques. COSMO was endorsed as a high priority in the latest Heliophysics Decadal Survey.

---

## 3. Key Takeaways / 핵심 시사점

1. **코로나 자기장은 태양 물리학의 "missing observable"이다** — 광구 자기장은 일상적으로 측정되지만, 코로나 자기장은 약한 세기(1–10 G)와 넓은 열적 선폭으로 인해 직접 측정이 거의 불가능했다. COSMO는 코로나 금지선(특히 Fe XIII 1074.7/1079.8 nm)의 편광 관측을 통해 이 간극을 메우고자 한다.
   The coronal magnetic field is the "missing observable" in solar physics — routinely measured in the photosphere but nearly impossible to directly measure in the corona due to weak field strength and broad thermal line widths. COSMO aims to fill this gap through forbidden-line polarization.

2. **세 기기의 상보성이 COSMO의 핵심 설계 철학이다** — K-Cor(전자 밀도/동역학, 15초), ChroMag(채층 자기장/플라즈마, 10초/선), LC(코로나 자기장/온도/밀도/속도, 1초–15분)가 광구–채층–코로나를 "결합 시스템"으로 관측한다.
   The complementarity of three instruments is COSMO's core design philosophy — K-Cor (electron density/dynamics), ChroMag (chromospheric magnetism/plasma), and LC (coronal magnetism/temperature/density/velocity) observe the photosphere–chromosphere–corona as a "coupled system."

3. **1.5 m 렌즈 코로나그래프는 전례 없는 기기이다** — 산란광 억제를 위해 거울 대신 렌즈를 선택하고, Lithium Niobate 복굴절 필터로 1° FOV에서 λ/Δλ > 8000 분광 분해능을 달성하며, 이중 빔 편광법으로 시상 유도 잡음을 제거한다.
   The 1.5 m lens coronagraph is unprecedented — choosing a lens over mirror for stray light suppression, achieving λ/Δλ > 8000 spectral resolution over 1° FOV with LiNbO₃ birefringent filters, and eliminating seeing-induced noise with dual-beam polarimetry.

4. **K-Cor는 저 코로나의 유일한 백색광 관측 소스이다** — 1.05 $R_\odot$까지 관측 가능하여 LASCO C2(1.5 $R_\odot$)보다 ~20분 먼저 CME를 감지하며, EUV 관측과 외부 코로나 관측 사이의 공백을 메운다.
   K-Cor is the sole white-light observation source of the low corona — observing down to 1.05 $R_\odot$, detecting CMEs ~20 minutes before LASCO C2, and bridging the gap between EUV and outer-corona observations.

5. **COSMO와 DKIST는 상보적이다** — DKIST(4 m)는 집광 면적이 7배 크지만 FOV가 5′로 작고 사용자 주도인 반면, COSMO LC의 집광력(구경 × 시야 입체각)은 DKIST의 20배이며 시놉틱 운용에 최적화된다.
   COSMO and DKIST are complementary — DKIST (4 m) has 7× greater collecting area but only 5′ FOV and is user-driven, while COSMO LC's light-gathering power is 20× that of DKIST, optimized for synoptic operation.

6. **가시광/근적외선 금지선은 EUV 관측과 본질적으로 다른 물리를 탐사한다** — EUV 방출은 충돌 과정($n_e^2$)에 의존하여 고밀도 영역에 편향되고 고도에 따라 급감하는 반면, 가시광 금지선은 복사 여기(photospheric radiation field)에 의존하여 더 높은 고도까지 관측 가능하다.
   Visible/near-IR forbidden lines probe fundamentally different physics from EUV — EUV emission depends on collisional processes ($n_e^2$), biased toward high density and rapidly declining with height, while visible forbidden lines depend on radiative excitation and are observable to greater heights.

7. **Science Traceability Matrix가 기기 설계의 체계적 근거를 제공한다** — Table 1은 과학 질문 → 측정 목표 → 측정 요구사항 → 기기 사양의 논리적 흐름을 명확히 보여주며, 기기 설계의 모든 결정이 과학적 필요에서 도출됨을 입증한다.
   The Science Traceability Matrix provides systematic justification for instrument design — Table 1 clearly shows the logical flow from science questions to measurement objectives to requirements to instrument specifications.

8. **COSMO 데이터는 우주 날씨 모델의 핵심 누락 입력이다** — FORWARD 코드와 SWMF/AWSoM 모델의 경계 조건과 검증 데이터를 직접 제공하여, 현재 관측적 제약이 없는 파라미터(코로나 자기장, Alfvén 파 에너지 등)를 처음으로 공급한다.
   COSMO data is the critical missing input for space weather models — directly providing boundary conditions and validation data for FORWARD and SWMF/AWSoM, supplying first-ever observational constraints on currently unconstrained parameters.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 코로나 자기장 측정 오차 / Coronal Magnetic Field Measurement Error

코로나 방출선의 원편광(Stokes V)에서 시선 방향 자기장을 측정할 때의 오차 추정:

Error estimate for LOS magnetic field measurement from circular polarization (Stokes V) of coronal emission lines:

$$\sigma_B = \frac{16.5}{\sqrt{N}} \left(1 + 2\frac{B}{N}\right)^{1/2} \quad \text{(kG)} \tag{1}$$

- $N$: 방출선에 걸쳐 적분한 광자 수 / Number of photons integrated over emission line
- $B$: 같은 파장 구간의 배경 광자 수 / Number of background photons over same wavelength interval
- Fe XIII 1074.7 nm 가정 (최고 신호 대 잡음비) / Assuming Fe XIII 1074.7 nm (highest SNR)
- 단일 빔 편광계, 효율 0.57 가정 / Single-beam polarimeter, efficiency 0.57

**수치 예시 / Numerical example**:
- $B = 0$ (배경 없음): $\sigma_B = 1$ G를 달성하려면 $N = 2.7 \times 10^8$ 광자 필요
- $B = 0$, $N = 2.7 \times 10^8$: $\sigma_B = 16.5 / \sqrt{2.7 \times 10^8} = 16.5 / 16432 \approx 0.001$ kG $= 1$ G ✓
- 실제 조건 (코로나 10 ppm, 배경 5 ppm): 1.5 m 구경, 2″ 분해능, 15분 적분으로 ~1 G 달성 가능

### 4.2 Thomson 산란 편광 밝기 / Thomson Scattering Polarization Brightness

$$pB(r) = \frac{\pi \sigma_T \bar{B}_\odot}{2} \int_{-\infty}^{\infty} n_e(l) \cdot \mathcal{G}(r, l) \, dl$$

- $\sigma_T = 6.65 \times 10^{-29}$ m²: Thomson 산란 단면적
- $\bar{B}_\odot$: 태양 표면 평균 밝기 / Mean solar surface brightness
- $n_e(l)$: 시선 경로 따른 전자 밀도 / Electron density along LOS
- $\mathcal{G}(r, l)$: 기하학적 가중 함수 (산란각, 림 어두움 포함) / Geometric weighting function (scattering angle, limb darkening)

pB는 $n_e$에 선형 비례 → 전자 밀도 직접 진단 가능.

pB is linearly proportional to $n_e$ → direct electron density diagnostic.

### 4.3 Alfvén 파 위상 속도와 코로나 자기장 / Alfvén Wave Phase Speed and Coronal Magnetic Field

$$v_A = \frac{B}{\sqrt{\mu_0 \rho}} \quad \Rightarrow \quad B = v_A \sqrt{\mu_0 \rho}$$

- $v_A$: Alfvén 파 위상 속도 (CoMP/LC Doppler 시계열에서 측정) / Alfvén wave phase speed (measured from Doppler time series)
- $\rho = n_e m_p \mu$ ($\mu \approx 0.6$, 평균 분자량): 플라즈마 질량 밀도 / Plasma mass density
- $n_e$: Fe XIII 1074.7/1079.8 nm 선 비율에서 추정 / Estimated from line ratio

코로나 seismology: 파동 관측 + 밀도 추정 → 자기장 간접 결정. 이것은 Zeeman/Hanle 효과에 의한 직접 측정과 상보적이다.

Coronal seismology: wave observations + density estimation → indirect magnetic field determination. Complementary to direct Zeeman/Hanle measurements.

### 4.4 금지선 비율과 전자 밀도 / Forbidden Line Ratio and Electron Density

$$R = \frac{I(\text{Fe XIII } 1079.8)}{I(\text{Fe XIII } 1074.7)} = f(n_e)$$

같은 이온의 두 선은 같은 온도 의존성을 가지므로, 비율은 전자 밀도에만 민감하다. 이것이 밀도 민감 선 쌍(density-sensitive line pair)의 원리이다.

Two lines from the same ion have the same temperature dependence, so their ratio is sensitive only to electron density. This is the principle of density-sensitive line pairs.

### 4.5 Zeeman 분리의 한계 / Limitation of Zeeman Splitting

$$\Delta \lambda_Z = \frac{e \lambda^2 B}{4\pi m_e c} = 4.67 \times 10^{-13} \lambda^2 g B \quad \text{(nm)}$$

Fe XIII 1074.7 nm에서 $B = 10$ G, $g = 1.5$일 때: $\Delta \lambda_Z \approx 0.008$ nm

코로나 온도 $T \sim 10^6$ K에서의 열적 선폭: $\Delta \lambda_{th} \sim 0.04$ nm

$\Delta \lambda_Z \ll \Delta \lambda_{th}$ → 직접 Zeeman 분리 감지 불가 → 편광(Stokes V) 측정 필요.

At coronal temperature $T \sim 10^6$ K, thermal line width $\Delta \lambda_{th} \sim 0.04$ nm. Since $\Delta \lambda_Z \ll \Delta \lambda_{th}$, direct Zeeman splitting is undetectable → Stokes V polarization measurement is needed.

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1930  ── Lyot: 최초의 코로나그래프 발명
         First coronagraph invented
         │
1940s ── 코로나 금지선 발견, 코로나 ~1 MK 확인
         Coronal forbidden lines, corona ~1 MK confirmed
         │
1964  ── Pierce: McMath Solar Telescope [이 시리즈 #1]
         │  Dunn: Evacuated Tower Telescope [이 시리즈 #2]
         │
1978  ── Dulk & McLean: 전파로 코로나 자기장 추정
         Radio coronal magnetic field estimation
         │
1980  ── HAO Mk-III K-coronameter, Mauna Loa
         │
1995  ── SOHO/LASCO 발사 — 우주 코로나그래프 시대 개막
         SOHO/LASCO launch [이 시리즈 #8, #10]
         │
1996  ── GONG [#5], BiSON [#6] — 시놉틱 네트워크
         │
2001  ── Judge & Casini: Zeeman/공명 산란 편광 → 코로나 자기장 최유망 기법
         Most promising methods for coronal B measurement
         │
2003  ── SST 1m [#3] — AO로 지상 분해능 혁신
         │  DKIST 개념 설계 (Keil et al.)
         │
2004  ── Lin et al.: Hawaii Optical Fiberbundle Imaging Spectropolarimeter
         │  Tomczyk: CoMP 원형 개발
         │
2006  ── STEREO 발사 — 다시점 코로나 관측
         │
2007  ── Tomczyk et al., Science: CoMP로 코로나 Alfvén 파 최초 감지
         First detection of coronal Alfvén waves
         │
2008  ── Tomczyk et al.: CoMP 기기 논문 [이 시리즈 #34]
         │
2012  ── SDO/AIA [#12], HMI [#13] — 고해상도 태양 전면 관측
         │  Goode: NST 1.6m [#4]
         │
2013  ── K-Cor Mauna Loa 배치 (Mk4 대체) — COSMO 첫 기기 운용 시작
         K-Cor deployed, first COSMO instrument operational
         │
2014  ── IRIS [#16] — 채층/천이 영역 고분해능 분광
         │
2016  ── ★ Tomczyk et al.: COSMO 과학 목표와 기기 역량 종합 기술 ★
         │  This paper
         │
2020  ── DKIST 첫 관측 [#23] — 4m 태양 망원경
         │
2021  ── UCoMP (Upgraded CoMP) 가동 — LC 기술 시연
         UCoMP operational — LC technology demonstration
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| #5 Harvey et al. (1996) — GONG | 전 지구 시놉틱 네트워크의 모범 / Model for global synoptic networks | COSMO도 시놉틱 연속 관측을 목표. GONG이 일진학에서 한 것을 COSMO가 코로나 자기장에서 수행 / COSMO aims for synoptic continuous observation, doing for coronal magnetism what GONG did for helioseismology |
| #6 Chaplin et al. (1996) — BiSON | 공명 산란 분광 기법 / Resonant scattering spectroscopy | ChroMag의 분광 편광 기법과 개념적 유사성. 전체 태양 vs 분해 관측의 상보성 / Conceptual similarity with ChroMag's spectropolarimetric technique |
| #8 Domingo et al. (1995) — SOHO | LASCO 코로나그래프 모선(母船) / LASCO coronagraph host mission | K-Cor가 LASCO C2 아래 공백(1.05–1.5 $R_\odot$)을 메움. Fig. 5에서 직접 비교 / K-Cor fills the gap below LASCO C2. Direct comparison in Fig. 5 |
| #10 Brueckner et al. (1995) — LASCO | 우주 백색광 코로나그래프의 기준 / Space white-light coronagraph benchmark | K-Cor가 LASCO보다 ~20분 먼저 CME 감지. 지상/우주 상보성의 사례 / K-Cor detects CMEs ~20 min before LASCO. Example of ground/space complementarity |
| #12 Lemen et al. (2012) — SDO/AIA | EUV 코로나 영상의 기준 / EUV coronal imaging benchmark | COSMO LC가 AIA와 상보적: AIA(좁은 대역, $n_e^2$ 의존) vs LC(분광+편광, 복사 여기) / COSMO LC complements AIA: narrowband $n_e^2$-dependent vs spectropolarimetric radiatively-excited |
| #13 Scherrer et al. (2012) — SDO/HMI | 광구 자기장 측정의 기준 / Photospheric magnetometry benchmark | COSMO가 측정하려는 코로나 자기장은 HMI 광구 측정의 "위쪽 확장". 외삽 검증에 핵심 / Coronal B measured by COSMO is the "upward extension" of HMI photospheric measurements. Critical for validating extrapolations |
| #16 De Pontieu et al. (2014) — IRIS | 채층/천이 영역 고분해능 분광 / High-res chromospheric spectroscopy | ChroMag과 상보적: IRIS(고분해능, 소시야, 비편광) vs ChroMag(시놉틱, 전일면, 편광) / Complementary: IRIS (high-res, small FOV, no polarimetry) vs ChroMag (synoptic, full disk, polarimetry) |
| #23 Rimmele et al. (2020) — DKIST | 4 m 태양 망원경 / 4 m solar telescope | 가장 직접적인 상보 관계: DKIST(고분해능, 5′ FOV) vs COSMO LC(시놉틱, 1° FOV, 집광력 20배) / Most direct complementarity: DKIST (high-res, 5′) vs COSMO LC (synoptic, 1°, 20× light-gathering power) |
| #34 Tomczyk et al. (2008) — CoMP | 코로나 편광계 원형 / Coronal polarimeter prototype | COSMO LC의 직접적 전신. CoMP의 과학적 성과(Alfvén 파 감지)가 COSMO 동기. CoMP의 한계(작은 구경)를 LC가 극복 / Direct predecessor of COSMO LC. CoMP's science (Alfvén waves) motivated COSMO. LC overcomes CoMP's limitations (small aperture) |

---

## 7. References / 참고문헌

- Tomczyk, S., E. Landi, J. T. Burkepile, R. Casini, E. E. DeLuca, Y. Fan, S. E. Gibson, H. Lin, S. W. McIntosh, S. C. Solomon, G. de Toma, A. G. de Wijn, and J. Zhang (2016), "Scientific objectives and capabilities of the Coronal Solar Magnetism Observatory," *J. Geophys. Res. Space Physics*, 121, 7470–7487. [DOI: 10.1002/2016JA022871]
- Tomczyk, S., et al. (2008), "An Instrument to Measure Coronal Emission Line Polarization," *Sol. Phys.*, 247, 411. [DOI: 10.1007/s11207-007-9103-6]
- Tomczyk, S., and S. W. McIntosh (2009), "Time-distance seismology of the solar corona with CoMP," *Astrophys. J.*, 697, 1384–1391.
- Tomczyk, S., et al. (2007), "Alfvén waves in the solar corona," *Science*, 317, 1192.
- Judge, P. G., R. Casini, S. Tomczyk, D. P. Edwards, and E. Francis (2001), "Coronal magnetometry: A feasibility study," *NCAR Tech. Note*.
- Gibson, S. (2015b), "Data-model comparison using FORWARD and CoMP," *Proc. Int. Astron. Union*, 305, 245–250.
- Penn, M. J., H. Lin, S. Tomczyk, D. F. Elmore, and P. G. Judge (2004), "Background induced measurement errors of the coronal intensity, density, velocity and magnetic field," *Sol. Phys.*, 222, 61–78.
- Nelson, P. G., S. Tomczyk, D. F. Elmore, and D. J. Kolinski (2008), "The feasibility of large refracting telescopes for solar coronal research," *Proc. SPIE*, 7012.
- de Wijn, A. G., J. T. Burkepile, S. Tomczyk, P. Nelson, P. Huang, and D. Gallagher (2012), "Stray light and polarimetry considerations for the COSMO K-coronagraph," *Proc. SPIE*, 8444.
- Landi, E., S. R. Habbal, and S. Tomczyk (2016), "Coronal plasma diagnostics from ground-based observations," *J. Geophys. Res. Space Physics*, 122. [DOI: 10.1002/2016JA022598]
- Klimchuk, J. A. (2006), "On solving the coronal heating problem," *Sol. Phys.*, 234, 41.
- Aschwanden, M. J., et al. (2007), "The EUV imaging spectrometer for Hinode," *Sol. Phys.*, 243, 19–61.
- De Pontieu, B., et al. (2007), "Chromospheric Alfvénic waves strong enough to power the solar wind," *Science*, 318, 1574.
- Morton, R. J., and J. A. McLaughlin (2013), "Hi-C and AIA observations of transverse magnetohydrodynamic waves in active regions," *Astron. Astrophys.*, 553, L10.
- DeRosa, M. L., et al. (2009), "A critical assessment of nonlinear force-free field modeling of the solar corona for active region 10953," *Astrophys. J.*, 696, 1780.
- Baker, D. N., et al. (2013), "Solar and Space Physics: A Science for a Technological Society," *National Academies Press* (Heliophysics Decadal Survey).
