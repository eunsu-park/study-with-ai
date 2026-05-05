---
title: "Pre-Reading Briefing: Extreme Ultraviolet Variability Experiment (EVE) on SDO"
paper_id: "51_woods_2012"
topic: Solar_Observation
date: 2026-04-25
type: briefing
---

# Extreme Ultraviolet Variability Experiment (EVE) on SDO: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Woods, T.N., Eparvier, F.G., Hock, R., Jones, A.R., Woodraska, D., Judge, D., Didkovsky, L., Lean, J., Mariska, J., Warren, H., McMullin, D., Chamberlin, P., Berthiaume, G., Bailey, S., Fuller-Rowell, T., Sojka, J., Tobiska, W.K., Viereck, R., "Extreme Ultraviolet Variability Experiment (EVE) on the Solar Dynamics Observatory (SDO): Overview of Science Objectives, Instrument Design, Data Products, and Model Developments", *Solar Physics* **275**, 115-143, 2012. DOI: 10.1007/s11207-009-9487-6
**Author(s)**: T.N. Woods et al. (18 co-authors)
**Year**: 2012 (received 2009, published online January 2010)

---

## 1. 핵심 기여 / Core Contribution

EVE는 NASA의 Solar Dynamics Observatory (SDO)에 탑재된 세 가지 주요 관측기 중 하나로, 0.1–105 nm 범위의 태양 극자외선(EUV) 분광 복사 조도(spectral irradiance)를 0.1 nm의 분광 분해능과 10초의 시간 분해능, 그리고 20% 이내의 절대 정확도로 측정하기 위해 설계된 분광기 모음(instrument suite)이다. EVE는 MEGS-A (Multiple EUV Grating Spectrograph A, 5–37 nm), MEGS-B (35–105 nm), MEGS-SAM (Solar Aspect Monitor, 0.1–7 nm), MEGS-P (Lyman-α, 121.6 nm), 그리고 ESP (EUV SpectroPhotometer, 0.1–39 nm 광대역) 등 다섯 가지 검출기 채널을 통합하여 광범위한 파장 영역을 동시에 관측한다. 이 논문은 EVE의 네 가지 과학 목표(EUV 복사 조도 명세화, 변동 원인 이해, 예보 능력 개선, 지구 환경 응답 이해), 기기 설계, 데이터 제품 위계(Level 0C/0CS/1/2/3), 그리고 NRLEUV·FISM·SIP 같은 태양 복사 모델과의 연계를 종합 정리하여 우주 기상 연구와 작업 운용을 위한 기준 문서(reference document) 역할을 한다.

EVE is one of three primary instruments aboard NASA's Solar Dynamics Observatory (SDO), designed to measure the solar extreme ultraviolet (EUV) spectral irradiance from 0.1 to 105 nm with unprecedented combination of 0.1 nm spectral resolution, 10 s temporal cadence, and 20 % absolute accuracy. EVE integrates five detector channels — MEGS-A (5–37 nm grazing-incidence spectrograph), MEGS-B (35–105 nm normal-incidence dual-pass spectrograph), MEGS-SAM (Solar Aspect Monitor pinhole camera, 0.1–7 nm), MEGS-P (Lyman-α photometer at 121.6 nm), and ESP (broadband EUV SpectroPhotometer covering 0.1–39 nm) — to span the full spectral range that drives Earth's upper atmosphere. The paper articulates EVE's four science objectives (specify EUV irradiance, understand variability, forecast variations, understand geospace response), describes the instrument hardware, lays out the Level 0C/0CS/1/2/3 data-product hierarchy, and frames the modeling ecosystem (NRLEUV, FISM, SIP, CTIPe, NRLMSIS, JB2006/2008) that EVE is intended to feed. It is the canonical reference document for the EVE mission and its space-weather operational role.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

2010년대 초반은 NASA의 Living With a Star (LWS) 프로그램이 본격적으로 가동되기 시작한 시기로, 태양 변동성과 그 사회적·기술적 영향을 정량적으로 이해하려는 노력이 정점에 달했다. 이전 세대의 EUV 측정 기기들 — UARS/SOLSTICE, SOHO/SEM (Solar EUV Monitor), TIMED/SEE (Solar EUV Experiment), GOES/XRS, SORCE/XPS — 은 일별 또는 광대역 측정에 머물러, 태양 플레어처럼 수 분 단위로 진행되는 사건의 분광 변화를 관측할 수 없었다. EVE는 이러한 한계를 극복하고 SOHO/SEM(1996년~)과 TIMED/SEE(2003년~) 측정의 연속 기록을 확장하면서, 동시에 10초 시간 분해능과 0.1 nm 분광 분해능이라는 새로운 표준을 도입했다.

The early 2010s saw NASA's Living With a Star (LWS) program reach maturity, with strong emphasis on quantifying solar variability and its societal/technological impacts. Earlier-generation EUV measurement instruments — UARS/SOLSTICE, SOHO/SEM, TIMED/SEE, GOES/XRS, SORCE/XPS — provided either daily-averaged or broadband measurements, leaving unresolved the minutes-to-tens-of-seconds spectral evolution of solar flares. EVE was designed to fill this gap, extend the continuous EUV irradiance record begun by SOHO/SEM (1996–) and TIMED/SEE (2003–), and simultaneously introduce a new standard of 10 s cadence and 0.1 nm spectral resolution.

### 타임라인 / Timeline

```
1979─────1991─────1996─────2002─────2010─────2014
  │        │        │        │        │        │
AE-E     UARS/    SOHO/    TIMED/    SDO/     IRIS
SEUM     SOLSTICE  SEM      SEE       EVE
                            (broad-   (0.1 nm
                            band)     0.1-105 nm
                                      10 s)
```

- **1979** — AE-E satellite, early daily EUV irradiance / 초기 일별 EUV 측정
- **1991** — UARS launch, SOLSTICE measures FUV / FUV 측정 시작
- **1996** — SOHO launch, SEM provides broadband EUV / SEM 광대역 EUV
- **2002** — TIMED launch, SEE provides daily EUV spectra / 일별 EUV 분광
- **2010** — SDO launch, EVE provides 10 s EUV spectra at 0.1 nm / 10초 0.1 nm 분광
- **2014** — IRIS, complementary chromospheric/TR spectroscopy / 보완 채층/전이층

---

## 3. 필요한 배경 지식 / Prerequisites

### 분광학 및 광학 / Spectroscopy and Optics

- **Grazing-incidence vs. normal-incidence optics**: EUV는 모든 물질에서 흡수가 크기 때문에 73°–80° 같은 grazing 입사각이 효율적이다. MEGS-A는 grazing-incidence이고 MEGS-B는 normal-incidence다. / EUV is strongly absorbed by all materials, so grazing angles of 73°–80° are efficient. MEGS-A uses grazing incidence; MEGS-B uses normal incidence.
- **Rowland circle geometry**: 구면 회절격자에서 자기 집속(self-focusing)을 보장하는 원으로, 슬릿·격자·검출기가 같은 원 위에 놓인다. / Self-focusing geometry where slit, grating, and detector lie on the same circle.
- **Diffraction grating equation**: $d (\sin\alpha + \sin\beta) = m\lambda$ — 격자 분산 / Grating dispersion.
- **Filter transmission**: Zr, Al, Ti 등 박막 필터로 차수(order) 분리 / Thin-film filters (Zr, Al, Ti) for order sorting.

### 검출기 / Detectors

- **Back-illuminated CCD**: MIT-LL이 제작한 1024 × 2048 픽셀, 15 μm 픽셀, –70°C 운용 / MIT-LL fabricated, operated at –70 °C.
- **Silicon photodiode (AXUV/IRD)**: ESP와 MEGS-P에서 사용, 변환 효율은 ~3.66 eV/e⁻ / Used in ESP and MEGS-P; charge generation ~3.66 eV/e⁻.

### 태양 물리 / Solar Physics

- **Chromosphere, transition region, corona**: $T \approx 10^4$–$10^7$ K, EUV 방출선의 출처 / Origin of EUV emission lines.
- **Solar flares (Hα, GOES classes)**: 분광 변화 시간 척도가 수 분~수 시간 / Spectral variation timescale minutes to hours.
- **F10.7 cm radio flux**: 전통적인 태양 활동 대용 지표 / Traditional proxy for solar activity.

### 지구 대기 / Earth's Upper Atmosphere

- **Thermosphere/ionosphere coupling**: EUV가 가열·이온화 주도 / EUV drives heating and ionization.
- **Atmospheric drag**: NRLMSIS, JB2006/2008 모델이 EUV를 입력으로 사용 / Models use EUV as input.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **EUV / 극자외선** | Extreme UltraViolet, ~10–121 nm 영역. 지구 대기를 가열·이온화. / Region ~10–121 nm; heats and ionizes Earth's upper atmosphere. |
| **XUV / 연 X선** | 0.1–10 nm 영역. 플레어 시 강도가 매우 가변. / 0.1–10 nm; highly variable during flares. |
| **MEGS-A** | 5–37 nm grazing-incidence 분광기 (off-Rowland circle). / Grazing-incidence spectrograph for 5–37 nm. |
| **MEGS-B** | 35–105 nm normal-incidence dual-pass cross-dispersing 분광기. / Normal-incidence cross-dispersing spectrograph. |
| **MEGS-SAM** | 0.1–7 nm pinhole camera, MEGS-A CCD의 한 모서리 사용, X-ray 광자 계수 모드. / Pinhole camera using corner of MEGS-A CCD; XUV photon counting. |
| **MEGS-P** | 121.6 nm Lyman-α 광도계 (silicon photodiode + Acton 필터). / Lyman-α photometer. |
| **ESP** | EUV SpectroPhotometer, 5채널 광대역 (18.2, 25.7, 30.4, 36.6 nm + 0.1–7 nm QD). / Broadband 5-channel transmission grating photometer. |
| **Spectral irradiance / 분광 복사 조도** | $E_\lambda$ in mW m⁻² nm⁻¹, 1 AU에서 단위 면적당 단위 파장당 태양 복사. / Solar power per unit area per unit wavelength at 1 AU. |
| **Level 0C / 0CS** | 우주 기상 제품, 약 15분 지연 (0CS는 약 1분 지연). / Space-weather products with ~15 min (0CS ~1 min) latency. |
| **Level 1/2/3** | Full-science 보정·병합·일평균 제품. / Fully calibrated, merged, daily-averaged science products. |
| **NRLEUV** | Naval Research Laboratory EUV 모델, DEM 기반 물리 모델. / NRL EUV model based on differential emission measure. |
| **FISM** | Flare Irradiance Spectral Model, 0.1–190 nm 1 nm 1분 경험 모델. / Empirical EUV/UV flare model. |
| **DEM** | Differential Emission Measure, $\xi(T) = n_e^2 \, dV/dT$. / DEM distribution. |
| **F10.7** | 10.7 cm (2800 MHz) 전파 플럭스, 태양 활동 대용 지표. / 10.7 cm radio flux proxy. |

---

## 5. 수식 미리보기 / Equations Preview

### (1) 회절 격자 방정식 / Diffraction grating equation

$$d \, (\sin\alpha + \sin\beta) = m \, \lambda$$

- $d$: 격자선 간격 (groove spacing) — 예: MEGS-A는 $1/767 \approx 1.304$ μm. / Groove spacing.
- $\alpha$: 입사각 (grazing incidence MEGS-A: 80°). / Incidence angle.
- $\beta$: 회절각 (73°–79°). / Diffraction angle.
- $m$: 차수 (1차 사용). / Diffraction order.
- $\lambda$: 파장. / Wavelength.

### (2) Spectral irradiance 정의 / Spectral irradiance

$$E_\lambda(\lambda, t) = \frac{C(\lambda, t) - C_{\rm dark}(t)}{A_{\rm eff}(\lambda) \, \Delta t \, \Delta \lambda} \cdot \frac{h c}{\lambda} \cdot \left(\frac{1\,\text{AU}}{r(t)}\right)^2$$

- $C$: 검출기 카운트 (DN 또는 photoelectrons). / Detector counts.
- $A_{\rm eff}$: 유효 면적 (cm²). / Effective area.
- $\Delta t$: 적분 시간 (10 s). / Integration time.
- $\Delta\lambda$: 파장 빈 폭 (0.02 nm 또는 0.1 nm). / Wavelength bin.
- $hc/\lambda$: 광자 에너지. / Photon energy.
- $(1\,\text{AU}/r)^2$: 1 AU 정규화. / 1 AU normalization.

### (3) DEM 기반 라인 강도 / DEM-based line intensity

$$I_{\rm line} = \frac{1}{4\pi} \int G(T, n_e, \lambda) \, \xi(T) \, dT, \quad \xi(T) = n_e^2 \frac{dV}{dT}$$

- $G(T)$: contribution function (CHIANTI 데이터베이스). / Contribution function from CHIANTI.
- $\xi(T)$: differential emission measure. / DEM.
- NRLEUV 모델의 핵심 / Core of NRLEUV model.

### (4) FISM 플레어 성분 / FISM flare component

$$E_{\rm FISM}(\lambda, t) = E_{\rm QS}(\lambda) + C_{\rm SR}(\lambda) \cdot P_{\rm SR}(t) + C_{\rm SC}(\lambda) \cdot P_{\rm SC}(t) + C_{\rm flare}(\lambda) \cdot P_{\rm flare}(t)$$

- $E_{\rm QS}$: quiet-Sun reference. / 정온 태양 기준.
- $P_{\rm SR}$: solar-rotation proxy (Mg II index). / 자전 대용.
- $P_{\rm SC}$: solar-cycle proxy (F10.7). / 주기 대용.
- $P_{\rm flare}$: GOES XRS-derived. / 플레어 대용.

### (5) Photon-counting energy mapping (MEGS-SAM) / SAM 광자 에너지 매핑

$$E_{\rm photon} \approx 3.66\,{\rm eV} \times N_{e^-}, \qquad \lambda = \frac{hc}{E_{\rm photon}}$$

- 단일 광자가 CCD 픽셀에 생성하는 광전자 수로부터 X선 에너지(따라서 파장)를 결정. / SAM determines X-ray wavelength from charge generated by single photons; ~3.66 eV per electron-hole pair in Si.

---

## 6. 읽기 가이드 / Reading Guide

### 권장 읽기 순서 / Suggested order

1. **Abstract + Section 1 (Introduction)** — 미션 개요와 SDO 컨텍스트 (HMI, AIA와의 관계). / Overview and SDO context.
2. **Section 2 (Science Plan)** — 네 가지 과학 목표를 머릿속에 정리. / Four science objectives.
3. **Section 3 (Instrumentation) + Table 1** — MEGS-A/B/SAM/P 및 ESP의 각 광학·검출기 사양. **이 표를 노트에 옮겨 적을 것.** / Optical/detector specs; copy Table 1 into notes.
4. **Section 4 (Data Products) + Table 2, 3** — Level 0C → Level 3 위계 및 EUV 라인 목록. / Data product hierarchy and EUV line list.
5. **Sections 5–6 (Models)** — NRLEUV, FISM, SIP, CTIPe, NRLMSIS, JB2006/2008. 각 모델의 입력/출력 관계를 다이어그램으로. / Sketch input/output relations.

### 주의해서 볼 부분 / Watch carefully

- **Figure 3**: EVE 다섯 기기의 파장 커버리지 그림 — 노트에 ASCII로 재현. / Replicate in ASCII.
- **Table 1**: 각 채널의 분광 분해능, 검출기, 격자, 필터 — 핵심 사양. / Core specs.
- **Section 3.3 (MEGS-SAM)**: pinhole + 광자 계수 = 기발한 설계. / Clever design.
- **Section 4.1 (Level 0C)**: 15분 지연 우주 기상 산출 — operational 측면. / Operational latency.

### 함께 읽으면 좋은 논문 / Companion papers

- Hock et al. (2010, in press): MEGS instrument calibration details. / MEGS 보정.
- Didkovsky et al. (2010): ESP calibration. / ESP 보정.
- Chamberlin, Woods, Eparvier (2007, 2008): FISM model. / FISM 모델.
- Lemen et al. (2012): SDO/AIA 동반 논문. / Companion AIA paper.
- Pesnell et al. (2012): SDO mission overview. / SDO 미션 개요.

---

## 7. 현대적 의의 / Modern Significance

### 우주 기상 운영 / Space-weather operations

EVE Level 0C 제품은 약 15분 이내 지연으로 NOAA SWPC, USAF, 그리고 ESA의 우주 기상 모델에 입력되어 위성 궤도 예측, GPS 보정, HF 통신 예보에 사용된다. EVE가 도입한 10초 시간 분해능은 IFM, GAIM, CTIPe 등의 이온층/열권 모델이 플레어와 같은 빠른 사건의 응답을 모델링할 수 있게 했다.

EVE Level 0C products feed NOAA SWPC, USAF, and ESA space-weather models within ~15 minutes of measurement, supporting orbit prediction, GPS corrections, and HF communications forecasts. The 10 s cadence introduced by EVE enabled ionospheric/thermospheric models (IFM, GAIM, CTIPe) to model responses to fast events such as flares.

### 후속 미션 / Successor missions

EVE는 2014년 5월 MEGS-A의 전원 이상으로 5–37 nm 채널이 손실되었으나, MEGS-B/SAM/P/ESP는 계속 운용 중이다. 후속으로는 GOES-R/EXIS (2016년~), DSCOVR, 그리고 향후 NOAA Space Weather Follow-On (SWFO) 미션이 EVE의 데이터 연속성을 확장하고 있다. FISM은 FISM2로 진화하여 EVE 데이터를 표준 보정에 사용한다.

EVE lost its MEGS-A channel (5–37 nm) due to a power anomaly in May 2014, but MEGS-B/SAM/P/ESP continue to operate. Successor instruments include GOES-R/EXIS (2016–), and the upcoming NOAA Space Weather Follow-On (SWFO). FISM has evolved into FISM2, calibrated against EVE.

### 과학적 유산 / Scientific legacy

EVE의 10초 분광 데이터는 플레어의 EUV late-phase 발견 (Woods et al. 2011), CME 단열 냉각 (Mason et al. 2014), 비EUV 활동 영역 변동 등 새로운 태양 물리 현상의 발견으로 이어졌다. EVE 시대를 거치며 EUV 분광 복사 조도는 단순한 태양-지구 결합 입력에서 코로나 진단 도구로 진화했다.

EVE's 10 s spectral cadence enabled discoveries such as the EUV late phase of solar flares (Woods et al. 2011), CME adiabatic cooling signatures (Mason et al. 2014), and active-region variability studies. EUV spectral irradiance has evolved from being merely an input to Sun–Earth coupling models to serving as a coronal diagnostic tool in its own right.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)

### Q1. MEGS-A의 grazing-incidence와 MEGS-B의 normal-incidence는 왜 서로 다른가? / Why does MEGS-A use grazing incidence while MEGS-B uses normal incidence?

**A.** EUV 광자는 파장이 짧을수록 모든 물질에서 흡수가 강해지므로, 5–37 nm 영역(MEGS-A)에서는 80°의 grazing 입사각이 효율적이다. 반면 35–105 nm 영역(MEGS-B)에서는 Pt 코팅 격자로도 normal incidence가 충분히 효율을 낼 수 있고, 두 격자(900 + 2140 grooves/mm)를 cross-dispersing 구성으로 사용하여 0.1 nm 분해능을 달성한다.

EUV photons are increasingly absorbed by all materials at shorter wavelengths, so for 5–37 nm MEGS-A uses 80° grazing incidence. For 35–105 nm MEGS-B, normal-incidence Pt-coated gratings give adequate efficiency, and a two-grating (900 + 2140 grooves mm⁻¹) cross-dispersing configuration achieves 0.1 nm resolution.

### Q2. MEGS-SAM은 어떻게 분광기 없이 0.1–7 nm 스펙트럼을 만드는가? / How does MEGS-SAM produce a 0.1–7 nm spectrum without a spectrograph?

**A.** SAM은 핀홀(26 μm 직경) 카메라로 태양 이미지를 MEGS-A CCD 한 모서리에 투영한다. XUV 광자 계수 모드에서 단일 광자가 만드는 전하량(~3.66 eV/e⁻)이 광자 에너지에 비례하므로, 각 광자 사건의 픽셀 강도로부터 파장을 ~1 nm 분해능으로 결정한다. 이는 X선 검출기 표준 기법을 EUV에 응용한 것이다.

SAM is a pinhole (26 μm) camera projecting a Sun image onto a corner of the MEGS-A CCD. In XUV photon-counting mode, the charge generated by each single photon (~3.66 eV per e⁻ in Si) is proportional to its energy, allowing wavelength determination at ~1 nm resolution. This applies standard X-ray detection technique to EUV.

### Q3. Level 0C, 1, 2, 3 의 차이는? / What distinguishes Level 0C, 1, 2, 3?

**A.** Level 0C는 우주 기상용으로 ~15분 지연·1분 평균·간단 보정만 적용된다. Level 1은 10초 고급 보정 데이터(MEGS-A/B/SAM/P/ESP 별도 파일). Level 2는 MEGS-A+B를 0.02 nm 균일 격자에 병합하여 1시간 단위 파일. Level 3는 1년 단위 일평균 통합 스펙트럼 (0.1–105 nm)을 제공한다.

Level 0C is the space-weather product with ~15 min latency, 1 min averaging, and minimal calibration. Level 1 is the 10 s fully calibrated science product, separated by instrument. Level 2 merges MEGS-A+B onto a uniform 0.02 nm grid in one-hour files. Level 3 provides daily-averaged combined spectra (0.1–105 nm) in one-year files.

### Q4. EVE 측정의 왜 0.1 nm 분광 분해능과 10초 시간 분해능이 동시에 중요한가? / Why are 0.1 nm spectral and 10 s temporal resolution both critical?

**A.** 이전 측정은 일별 분광(TIMED/SEE) 또는 광대역 1초(SEM/XRS) 중 하나만 가능했다. 그러나 플레어는 수 분 단위로 진화하며 EUV 라인마다 다른 시간 프로파일을 보인다(예: Fe XX는 임펄시브, Fe IX는 gradual). 0.1 nm 분해능으로 개별 라인을 분리하고 10초로 그 진화를 추적해야만 플레어의 코로나 가열·냉각 메커니즘을 진단할 수 있다.

Previous measurements offered either daily spectral (TIMED/SEE) or sub-second broadband (SEM/XRS) data, not both. Flares evolve on minute timescales, with each EUV line showing a different time profile (e.g. Fe XX impulsive, Fe IX gradual). Only 0.1 nm spectral isolation combined with 10 s cadence resolves both the spectral identity and temporal evolution needed to diagnose flare heating/cooling mechanisms.

### Q5. EVE의 5년 임무 중 가장 중요한 발견은? / What was EVE's most significant discovery during its mission?

**A.** EVE 시대의 핵심 발견 중 하나는 "EUV late phase" 플레어로, GOES soft X-ray가 감쇠한 후에도 ~1 시간 후 warmer EUV 라인(Fe XV, Fe XVI 등)이 두 번째 피크를 보이는 현상이다 (Woods et al. 2011). 이는 플레어 후 거대 루프의 단열 냉각과 추가 가열의 증거로 해석되며, 이러한 발견은 EVE의 동시 분광·시간 분해능이 없으면 불가능했다.

A signature EVE-era discovery is the "EUV late phase" of flares (Woods et al. 2011): warmer EUV lines (Fe XV, Fe XVI) show a secondary peak ~1 hour after GOES soft X-ray decay, interpreted as evidence of post-flare loop adiabatic cooling and continued reheating. This required EVE's simultaneous spectral and temporal coverage.
