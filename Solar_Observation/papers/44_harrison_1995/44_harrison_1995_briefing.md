---
title: "Pre-Reading Briefing: The Coronal Diagnostic Spectrometer for SOHO (CDS)"
paper_id: "44_harrison_1995"
topic: Solar_Observation
date: 2026-04-25
type: briefing
---

# The Coronal Diagnostic Spectrometer (CDS) for SOHO: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Harrison, R. A., Sawyer, E. C., Carter, M. K., Cruise, A. M., et al., "The Coronal Diagnostic Spectrometer for the Solar and Heliospheric Observatory (SOHO/CDS)", *Solar Physics* **162**, 233-290, 1995. DOI: 10.1007/BF00733431.
**Author(s)**: R. A. Harrison and 30+ co-authors (RAL, MSSL, GSFC, Oslo, MPE, PTB, UCB, ESA, Orsay, Cambridge, UCLan)
**Year**: 1995

---

## 1. 핵심 기여 / Core Contribution

이 논문은 SOHO 위성에 탑재된 코로나 진단 분광기(CDS, Coronal Diagnostic Spectrometer)의 설계, 제작, 보정, 그리고 과학 운용 계획을 종합적으로 기술하는 instrument paper다. CDS는 150-800 Å 파장 범위의 EUV 영역에서 방출선을 분광 관측하여 태양 대기(transition region 및 corona)의 온도, 전자밀도, 도플러 속도, 원소존재비를 동시에 진단하는 것을 목표로 한다. 가장 중요한 설계 혁신은 하나의 Wolter-Schwarzschild II 망원경이 두 개의 분광기를 동시에 공급한다는 점이다 — Grazing Incidence Spectrometer (GIS)는 넓은 파장 범위(150-785 Å, 4채널)를 다루지만 비점수차(astigmatic) 분광기로서 핀홀 슬릿 스캐닝으로 영상을 만들고, Normal Incidence Spectrometer (NIS)는 308-381 / 513-633 Å의 좁은 두 밴드만 다루지만 stigmatic 영상을 직접 제공한다. 이 이중 분광기 구성을 통해 광범위한 spectral 커버리지와 고품질 imaging을 모두 달성한다.

This paper is the comprehensive instrument paper describing the design, build, calibration, and scientific operations plan for the Coronal Diagnostic Spectrometer (CDS) on the ESA/NASA SOHO mission. CDS observes EUV emission lines in 150-800 Å to derive temperature, electron density, Doppler flow velocity, and elemental abundance information for plasmas spanning $10^4$-$\text{few}\times 10^6$ K — i.e., the entire transition region through the hot corona. The principal design innovation is that a single Wolter-Schwarzschild type 2 telescope feeds two spectrometers simultaneously: a Grazing Incidence Spectrometer (GIS) covering 150-785 Å in four bands but astigmatic, and a Normal Incidence Spectrometer (NIS) covering 308-381 / 513-633 Å but stigmatic and directly imaging. Together they offer broad coverage plus high-quality imaging diagnostic capability — a combination not previously available on a long-duration solar mission.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

1990년대 초반 태양 물리학의 두 가지 거대 미해결 문제는 (1) 코로나 가열 메커니즘과 (2) 태양풍 가속 메커니즘이었다. 이를 해결하려면 transition region (TR)부터 corona까지 광범위한 온도($10^4$-$10^7$ K)에서 플라즈마의 밀도, 온도, 속도, 풍부도를 정량적으로 측정해야 했다. 그러나 이전의 EUV 분광 관측 — OSO 시리즈, Skylab S082A/B (1973-74), Spacelab II CHASE (1985), SERTS rocket — 은 모두 단점이 있었다: Skylab은 사진건판이라 시간 분해능이 매우 낮고, CHASE/SERTS는 비행시간이 짧으며, OSO VII는 공간 분해능이 ~20"로 거칠었다. SOHO는 L1 궤도에서 24시간 연속 태양 관측이 가능한 최초의 ESA/NASA 합동 미션으로 기획되었으며, CDS는 그 핵심 코로나 진단 장비로 자리잡았다.

In the early 1990s solar physics had two grand-challenge questions: (1) why is the corona heated?, and (2) how is the solar wind accelerated? Answering them requires diagnosing densities, temperatures, flows, and abundances of plasmas across the full $10^4$-$10^7$ K range with adequate spatial, temporal, and spectral resolution. Earlier EUV missions — OSO VII (1972), Skylab S082 slitless spectrograph (1973/74), CHASE on Spacelab II (1985), the SERTS sounding rocket — each had limitations: Skylab used film (poor time resolution), CHASE/SERTS were short-duration flights, OSO VII had ~20" spatial resolution. SOHO, the first joint ESA/NASA mission stationed at L1 to give continuous solar viewing, was conceived to break this impasse. CDS — together with SUMER, EIT, UVCS, LASCO, MDI, etc. — was the key instrument for hot-plasma EUV diagnostics on SOHO.

### 타임라인 / Timeline

```
1962  OSO I — first EUV observations of the Sun
1972  OSO VII — early EUV spectroscopy, ~20" spatial resolution
1973  Skylab S082A/B — slitless EUV spectroheliographs (film)
1985  Spacelab II / CHASE — Coronal Helium Abundance Experiment
1989  Patchett, Harrison, Sawyer et al. — early CDS concept (ESA SP-1104)
1992  CDS design papers; first SOHO Workshop
1993  Harrison "Blue Book" — CDS Science Report
1994  CDS pre-flight calibration at BESSY (Berlin) / RAL
1995  THIS PAPER — CDS instrument paper
1995-12  SOHO launched (Dec 2, 1995)
1996+ CDS operations begin (NIS first light Apr 1996)
1998  SOHO accident (recovered); NIS-1 detector damaged later
2014-09 SOHO operational status; CDS turned off
```

---

## 3. 필요한 배경 지식 / Prerequisites

- **EUV 방출선의 형성 이론 / EUV line formation theory**: optically-thin coronal-approximation (collisional excitation balanced by spontaneous radiative decay), $\varepsilon = n_e n_H G(T, n_e)$ — 광학적으로 얇은 플라즈마에서의 emissivity. / Optically thin emissivity formula in the coronal regime.
- **Differential Emission Measure (DEM)**: $\mathrm{DEM}(T) = n_e^2 \, dh/dT$, line intensity는 contribution function과 DEM의 적분. / Definition of DEM and how line intensity integrals depend on it.
- **Line ratio diagnostics / 선비율 진단**:
  - 밀도 진단 / density-sensitive ratios: forbidden vs. allowed transitions에서 metastable level 인구가 collisional de-excitation으로 결정되어 $n_e$에 민감해진다 (e.g., Mg VII, Si IX, Fe XII, Fe XIII, S X). 임계밀도 $n_c \sim A_{ji}/q_{ji}$.
  - Density diagnostics: ratio of forbidden to allowed lines becomes sensitive to $n_e$ near the critical density where collisional de-excitation balances spontaneous decay.
  - 온도 진단 / temperature-sensitive ratios: 같은 이온의 에너지 차가 큰 두 선, 또는 인접 이온화 단계에서의 두 선 — Boltzmann factor가 $T$에 강하게 의존. / Lines of widely separated upper energy or two adjacent ionization stages of the same element — strong Boltzmann dependence on $T$.
- **Doppler 속도 측정 / Doppler shifts**: $\Delta\lambda/\lambda = v/c$. CDS는 NIS의 $\delta\lambda \sim 0.08$ Å로 ~수십 km/s 분해 가능. / CDS NIS spectral resolution allows tens-of-km/s velocity discrimination.
- **그레이팅 분광기 광학 / Grating spectrometer optics**: grating equation $n\lambda = d(\sin\theta + \sin\alpha)$, Rowland circle, $\lambda/\delta\lambda = 2\lambda R / (d\,\epsilon\cos\alpha)$.
- **Wolter-Schwarzschild II telescope**: paraboloid + confocal hyperboloid, both at grazing incidence — used because EUV reflectivity at normal incidence is essentially zero below ~300 Å. / Necessary because EUV reflectivity at normal incidence is negligible below ~300 Å.
- **MCP and CCD detectors / MCP·CCD 검출기**: microchannel plate 광전증배, SPAN (Spiral Anode) 1-D position-sensing, CCD with image intensifier (VDS).
- **이전 관측들에 대한 친숙함 / Familiarity with prior missions**: OSO, Skylab, SMM, SERTS.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| CDS | Coronal Diagnostic Spectrometer — SOHO 탑재 EUV 이중 분광기 / dual-spectrometer EUV instrument on SOHO |
| GIS | Grazing Incidence Spectrometer — Rowland-circle 설계, 4 채널 (151-221, 256-338, 393-493, 656-785 Å), astigmatic / astigmatic Rowland-circle spectrometer with four MCP detectors |
| NIS | Normal Incidence Spectrometer — toroidal 그레이팅 2개, 308-381 & 513-633 Å, stigmatic, single 2-D detector / two toroidal gratings imaging onto one VDS detector |
| Wolter-Schwarzschild type 2 | 포물면(primary) + 쌍곡면(secondary) grazing-incidence 망원경 / paraboloid plus confocal hyperboloid grazing telescope |
| VDS | Viewfinder Detector Subsystem — NIS의 MCP-intensified CCD 검출기 / NIS detector: MCP intensifier + 1024×1024 Tektronix CCD |
| SPAN | Spiral Anode — GIS MCP 뒤의 1-D position-sensing 양극 / 1-D position-sensing anode behind the GIS MCPs |
| Rowland circle | 직경 $R$인 원, 그 위에 grating·slit·detector를 두면 자동 초점 / circle of diameter $R$ on which grating, slit, and detector lie for self-focusing |
| Stigmatic / Astigmatic | stigmatic = 점이 점으로 결상; astigmatic = 한 방향의 결상이 무너짐 / stigmatic preserves point images; astigmatic loses one spatial dimension |
| DEM | Differential Emission Measure $= n_e^2 \, dh/dT$ — 온도별 방출 물질 분포 / amount of plasma per unit temperature |
| Line ratio diagnostic | 두 선의 강도비를 사용해 $T$ 또는 $n_e$를 결정 / use of two-line intensity ratio to derive $T$ or $n_e$ |
| Coronal approximation | 충돌여기 = 자발방출, 광학적으로 얇음 / collisional excitation balances spontaneous decay in optically thin plasma |
| MCP | Microchannel plate — 단일 광자에서 ~$4\times 10^7$ 전자 cascade 생성 / electron-multiplier producing $\sim 4\times 10^7$ electrons per EUV photon |
| CDHS | Command and Data Handling System — transputer-기반 CDS 두뇌 / transputer-driven on-board processor |
| BESSY | Berlin synchrotron, 절대 광도 보정 표준 / synchrotron used as absolute radiometric standard for calibration |

---

## 5. 수식 미리보기 / Equations Preview

**(1) Grating equation**: $n\lambda = d(\sin\theta + \sin\alpha)$
- $n$: order, $d$: grating spacing, $\theta$: incidence angle, $\alpha$: diffraction angle
- GIS의 경우 $\theta = 84.75°$ (grazing), $d = 1\,\mu\mathrm{m}$ (1000 lines/mm). / For GIS, $\theta = 84.75°$ at near-grazing.

**(2) Spectral resolving power**: $\dfrac{\lambda}{\delta\lambda} = \dfrac{2\lambda R}{d\,\epsilon\,\cos\alpha}$
- $R$: Rowland circle radius, $\epsilon$: line-width on detector. / Resolution depends on detector linewidth and diffraction geometry.

**(3) Count rate for solar emission line**:
$$
\text{count/s} = \dfrac{I\lambda \, a\,b\, A\, \epsilon_t \epsilon_m \epsilon_g \epsilon_d}{h\,c\,f^2}
$$
- $I$: line intensity (erg cm$^{-2}$ s$^{-1}$ sr$^{-1}$), $a\times b$: slit dimensions, $A$: telescope geometric area, $f$: focal length, $\epsilon_t,\epsilon_m,\epsilon_g,\epsilon_d$: telescope, scan-mirror, grating, detector efficiencies.
- 이 식은 instrument throughput을 광원 강도와 검출 카운트율에 직접 연결한다. / Connects source intensity to count rate via the throughput product.

**(4) Detector dead-time (extending model)**: $R_o = R_i \exp\!\left[-R_i (T_w + T_p)\right]$
- $R_i$: input rate, $R_o$: output rate, $T_w + T_p = 2.1\,\mu\mathrm{s}$ for GIS.
- Maximum throughput is $R_i = 1/(T_w+T_p) = 4.75\times 10^5$ c/s; $R_o/R_i = 1/e$ at that point. / Maximum throughput condition gives $1/e$ efficiency.

**(5) Doppler shift (background)**: $\Delta\lambda/\lambda = v/c$
- For NIS: $\delta\lambda = 0.08$ Å at $\lambda = 360$ Å gives $v_{\text{min}}\sim 67$ km/s in single-pixel, sub-pixel centroiding pushes this to a few km/s. / Centroiding extends sensitivity to a few km/s.

---

## 6. 읽기 가이드 / Reading Guide

이 논문은 58페이지의 instrument paper로, 빠르게 읽을 부분과 천천히 읽을 부분을 구분하라:

This is a 58-page instrument paper. Distinguish between the parts to read carefully and parts to skim:

- **천천히, 핵심 / Read slowly (key sections)**:
  - §1 Science Objectives + Tables I-IV (어떤 진단 라인을 왜 선택했는가) / why these diagnostic lines
  - §2 Instrument Overview + Fig. 1 + Table V (전체 구조 한눈에) / overall layout
  - §3.1 Telescope, §3.4 Spectrometers + Tables VI, VII (광학 핵심) / optics
  - §3.4.2 GIS detectors + §3.4.4 NIS detector (검출기 원리) / detector principles
  - §3.5 Straylight + §8 Calibration & Performance (보정과 한계) / calibration philosophy
  - §10 Scientific Operations (관측 모드) / observing modes

- **건너뛰어도 좋은 / Can skim**:
  - §4 Mechanical design (engineering details)
  - §5 Electrical design (power/CDHS implementation)
  - §6 Thermal design (radiator design)
  - §9 Ground segment (data pipeline)

- **핵심 그림 / Key figures**: Fig. 1 (CDS optics layout), Fig. 2 (PSF), Fig. 8 (DQE), Fig. 13 (electrical block), Fig. 17 (sensitivity).
- **핵심 표 / Key tables**: Table I-IV (diagnostic lines), Table V (specs), Table VI (GIS resolution), Table VII (NIS resolution), Table XI (expected count rates).

읽기 순서 추천: §1 → §2 → §3.1 → §3.4 → §8 → §10. 먼저 "무엇을 측정하는가"와 "어떻게 측정 가능한가"를 잡고, 그 다음 보정/운용을 따라가라.
Recommended order: §1 → §2 → §3.1 → §3.4 → §8 → §10. First grasp what is measured and how, then follow calibration and operations.

---

## 7. 현대적 의의 / Modern Significance

CDS는 SOHO의 핵심 코로나 진단 장비로서 ~18년간(1995-2014) 운용되며 transition region과 corona에 대한 방대한 분광 데이터를 생산했다. 그 직접적인 후계자는 Hinode/EIS (2006-, 170-290 Å, normal-incidence multilayer), Solar Orbiter/SPICE (2020-, 700-790 / 970-1050 Å, NIS-style), 그리고 향후 MUSE, Solar-C/EUVST이다. CDS가 확립한 "이중 분광기 = wide coverage + imaging" 설계 철학은 EIS의 단일 분광기 + 좁은 밴드 / SPICE의 NIS-only 설계로 갈리는 분기점이 되었다. 또한 CDS는 ADAS 원자 데이터 시스템을 적극 활용하여 "관측 → CHIANTI/ADAS → DEM/T/n_e" 파이프라인의 표준을 구축하였다. 오늘날에도 Mg/Si/Fe 선비율을 사용한 밀도 진단, Fe XV 284.16 / Fe XVI 335.40 등의 온도 진단은 CDS가 닦아놓은 토대 위에서 계속된다.

CDS served as SOHO's hot-plasma EUV diagnostic workhorse for ~18 years (1995-2014), producing a vast spectroscopic database on the transition region and corona. Its direct successors are Hinode/EIS (2006-, 170-290 Å, normal-incidence multilayers), Solar Orbiter/SPICE (2020-, NIS-style), and the upcoming MUSE and Solar-C/EUVST. CDS's design philosophy of pairing a wide-coverage astigmatic spectrometer with a stigmatic imaging spectrometer split into two paths — EIS chose pure normal-incidence with multilayers, SPICE chose NIS-only — but both descend conceptually from CDS. CDS also helped establish the "observation → ADAS/CHIANTI atomic data → DEM/T/n_e" analysis pipeline that remains standard today. Many density-sensitive line ratios in Mg VII/VIII, Si IX/X, Fe XII/XIII routinely used in the EIS/SPICE era trace their diagnostic catalog back to the work that motivated CDS's line list.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)

### Q1. Why use TWO spectrometers (GIS + NIS) instead of one? / 왜 분광기를 두 개 썼는가?

**A.** EUV 영역에서 normal-incidence 광학은 ~300 Å 이하에서 반사율이 사라진다. 따라서 가장 뜨거운 코로나 라인(Fe XV 284.16 Å, Fe XVI 335.40 Å는 NIS로 가능하지만 그 아래 Fe IX-XIV 같은 hottest 코로나 라인 169-220 Å은 grazing-incidence가 필수). 반대로 grazing-incidence는 광폭 파장 커버리지에 좋지만 toroidal grating으로 stigmatic을 만들 수 없어 영상이 흐려진다. 그래서 (1) 좁은 두 밴드의 stigmatic NIS와 (2) 넓은 4-채널 astigmatic GIS를 결합했다.
EUV reflectivity at normal incidence becomes negligible below ~300 Å, so the hottest coronal lines (Fe IX-XIV at 169-220 Å) require grazing incidence. But grazing-incidence Rowland-circle designs are astigmatic, so direct imaging is poor. CDS combines the two: NIS for stigmatic imaging in 308-381 / 513-633 Å, GIS for broad coverage in 151-785 Å (4 bands).

### Q2. How does CDS measure electron density without measuring it directly? / 직접 측정 없이 어떻게 전자밀도를 결정하는가?

**A.** Density-sensitive line-ratio 사용. 같은 이온의 forbidden 라인과 allowed 라인은 임계밀도 $n_c \sim A_{ji}/q_{ji}$ 부근에서 인구분포가 충돌탈여기에 의해 변한다. 비율 $I_1/I_2$가 $n_e$의 단조함수이므로 ratio measurement → $n_e$. Table I이 후보 라인쌍 목록 (Mg VI/VII/VIII, Si VIII/IX/X, S X/XI/XII, Fe X-XIV).
Density-sensitive line ratios. Two transitions of the same ion — typically a forbidden + allowed pair from a metastable upper level — have populations that shift near the critical density $n_c \sim A_{ji}/q_{ji}$ where collisional de-excitation matches spontaneous decay. The ratio $I_1/I_2$ is monotonic in $n_e$, so a ratio measurement yields $n_e$.

### Q3. What about temperature? / 온도는?

**A.** Two methods: (1) 같은 원소의 인접 이온화 단계 두 선의 비 — ionization equilibrium curve가 sharply peaked이므로 비율이 $T$의 강한 함수, (2) 한 이온 내에서 에너지 차이가 큰 두 선의 Boltzmann factor $\exp(-\Delta E/kT)$. Table II + III이 추천 라인 목록. Ne, Mg, Si, Fe series가 광범위한 $T = 10^{4.4}$-$10^{6.5}$ K 커버.
Two approaches: (i) the ratio of two lines from adjacent ionization stages of the same element — sharply $T$-dependent because each stage has a narrow ionization-equilibrium peak; (ii) the ratio of two lines within one ion at very different upper energies, exploiting $\exp(-\Delta E/kT)$. Tables II and III list recommended pairs spanning $\log T = 4.4$-$6.5$.

### Q4. What sets CDS's spatial resolution? / CDS의 공간 분해능을 결정하는 요소는?

**A.** 망원경 PSF FWHM (1.2-1.7" — Fig. 2), 검출기 픽셀 (NIS: 21 μm = 1.7"; GIS: 25 μm), MCP 모듈전달함수 (10% MTF at 16-18 lp/mm = 2.5-3"), 그리고 raster step (slit scan은 1" 단위). 한계는 ~3-5". Slit choices: NIS 2"×240" (stigmatic, 240" 1차원 즉시 영상) or 4"×240"; GIS 2"×2" or 4"×4" (point pinholes — 영상은 raster로 구축).
PSF FWHM is 1.2-1.7" (Fig. 2). Detector pixels are 21 μm ≈ 1.7" on NIS, 25 μm on GIS. MCP MTF reaches 10% at 16-18 lp/mm (~3"). Raster step is 1-2". Effective spatial resolution settles at 3-5".

### Q5. How is CDS calibrated absolutely? / 절대 보정은 어떻게 했나?

**A.** Pre-flight: 베를린 BESSY synchrotron을 1차 표준 광원으로 두고, hollow-cathode discharge lamp를 transfer source로 만들어 RAL로 옮긴 뒤 CDS 전체 조립체에 빛을 쏘는 end-to-end 보정 (1994년 2-3월, 4주). 절대 강도 불확도 6-8% (1σ). In-flight: (1) quiet Sun 강도 정기 모니터, (2) GIS-NIS 공통 파장 cross-cal, (3) 다른 SOHO 장비 cross-cal, (4) SERTS rocket cross-cal flights.
Pre-flight: BESSY synchrotron as primary, a hollow-cathode lamp as portable transfer source, end-to-end illumination of CDS at RAL (Feb-Mar 1994, 4 weeks). Absolute uncertainty 6-8% (1σ). In-flight: regular quiet-Sun monitoring, GIS-NIS overlap-region cross-calibration, cross-cal with other SOHO instruments, periodic SERTS rocket underflights.

### Q6. What if a SUMER-style "interesting event" alarm fires? / 흥미로운 이벤트 발생 시 CDS는?

**A.** CDS의 CDHS는 다른 SOHO 장비로부터 솔라 좌표를 받을 수 있다. Enabled되면 mirror, slit (그리고 필요하면 instrument pointing)을 즉시 그 좌표로 이동. 반대로 CDS도 흥미로운 이벤트를 다른 장비에 broadcast 가능. 이 협업 모드가 SOHO Joint Observing Programmes (JOPs)의 토대.
The CDHS can receive solar coordinates from other SOHO instruments and command its mirror, slit (and pointing if needed) to that target — and broadcast its own event coordinates to others. This collaborative capability underpins SOHO Joint Observing Programmes (JOPs).
