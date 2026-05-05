---
title: "Pre-Reading Briefing: The Low-Energy Neutral Atom Imager for IMAGE"
paper_id: "74_moore_2000"
topic: Space_Weather
date: 2026-04-25
type: briefing
---

# The Low-Energy Neutral Atom Imager for IMAGE: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Moore, T. E., Chornay, D. J., Collier, M. R., Herrero, F. A., Johnson, J., et al., "The Low-Energy Neutral Atom Imager for IMAGE", *Space Science Reviews* **91**, 155-195, 2000. DOI: 10.1023/A:1005211509003
**Author(s)**: T. E. Moore et al. (NASA GSFC, Lockheed Martin, U. Maryland, U. Denver, U. Bern, U. New Hampshire, Mission Research Corp.)
**Year**: 2000

---

## 1. 핵심 기여 / Core Contribution

LENA(Low-Energy Neutral Atom) 이미저는 IMAGE 위성에 탑재되어 10–750 eV 범위의 가장 낮은 에너지 중성원자(ENA)를 촬영하기 위한 세계 최초의 우주 비행 장비이다. 이 에너지 영역은 전리권 superthermal 이온이 thermosphere 중성 대기와 charge exchange할 때 생성되는 중성원자, 그리고 가속되는 태양풍 열이온의 charge exchange 영역, 성간 중성 가스 침투 영역에 해당한다. LENA는 새로운 atom-to-negative-ion 표면 변환 기술(polished tungsten conversion surface, –20 kV 가속, 75° 입사각)과 ITOF(imaging time-of-flight) 시스템을 결합하여 H와 O를 질량별로 구분하면서 90°×360° 시야와 ~수 분의 시간 분해능으로 전리권 outflow 가열을 원격 측정한다.

The LENA imager is the first space-flight instrument designed to image the lowest-energy neutral atoms (10–750 eV), produced when superthermal ionospheric ions undergo charge exchange with thermospheric neutrals (and to a lesser degree by solar-wind charge exchange and interstellar penetration). To overcome the impossibility of detecting such low-energy neutrals with conventional charge-exchange cells or carbon foils, LENA introduces a novel atom-to-negative-ion conversion-surface technology (polished polycrystalline tungsten at –20 kV, 75° grazing incidence) followed by electrostatic optics, a broom magnet, an ESA, and an imaging TOF analyzer. With a 90°×8° instantaneous field of view swept by spacecraft spin into a 90°×360° image (12×45 pixels), LENA enables global remote sensing of ionospheric plasma heating with time resolution as short as 2 minutes — a quantum leap over previous in-situ techniques whose temporal resolution was limited by spacecraft precession periods (months to years).

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

1980년대 후반부터 ENA(Energetic Neutral Atom) imaging 개념이 본격화되었으며, Roelof (1987)의 storm-time ring current ENA 이미지는 자기권 원격탐사의 가능성을 보여주었다. 그러나 이 초기 작업은 keV 이상의 에너지 영역에 집중되었고, 이는 carbon foil이나 thin-film 기술로 처리 가능했다. 한편 Yau et al. (1988), Pollock et al. (1990), Giles et al. (1994) 등의 in-situ DE-1, DMSP 관측은 cleft ion fountain, 극관(polar cap) outflow, auroral ion beam과 같은 저에너지(<수백 eV) 전리권 outflow가 자기권에 막대한 질량과 에너지를 공급함을 입증했다. 그러나 in-situ 위성의 spacecraft precession 시간 척도(수개월~수년) 때문에 outflow의 시간적 변동을 sub-hour scale에서 분리할 수 없었다. IMAGE 미션(2000년 발사)은 자기권 글로벌 이미징 전용 미션으로 LENA, MENA, HENA, FUV/SI/WIC, RPI 등 다양한 imager를 통합 탑재했고, LENA는 그 중 가장 어려운 저에너지 ENA 영역을 담당했다.

ENA imaging emerged as a magnetospheric remote-sensing technique in the late 1980s, with Roelof (1987) producing the first ring-current ENA image. Early efforts focused on energies above ~1 keV, accessible via carbon foils and thin-film stripping. In parallel, in-situ measurements from DE-1, DMSP, and Akebono (Yau et al. 1988; Pollock et al. 1990; Giles et al. 1994) revealed that low-energy (10–500 eV) ionospheric outflows from the cleft ion fountain, auroral zone, and polar cap supply substantial mass and energy to the magnetosphere — but the time resolution was bounded by spacecraft orbital precession (months to years). The IMAGE mission (launched March 25, 2000), the first spacecraft dedicated entirely to magnetospheric imaging, integrated LENA along with MENA (1–30 keV), HENA (>20 keV), FUV (SI/WIC), EUV, and RPI to produce simultaneous, multi-wavelength views of the magnetosphere. LENA tackled the most technically demanding regime — the lowest-energy neutrals — by pioneering surface-conversion ionization in space.

### 타임라인 / Timeline

```
1977 ─ Fasola: H- ion source via charge-exchange cell (lab)
1982 ─ Pargellis & Seidl: H- formation from cesiated surfaces
1986 ─ Herrero & Smith: Original LENA conceptual design
1987 ─ Roelof: First storm-time ring current ENA image
1988 ─ Yau et al.: Quantitative parametrization of ionospheric outflow
1990 ─ Pollock et al.: Survey of upwelling ion events
1992 ─ Gruntman: ENA imaging review; Herrero & Smith ILENA SPIE
1994 ─ Ghielmetti et al.: Mass spectrograph for low-E atoms (Bern)
1995 ─ Wurz et al.: Neutral atom mass spectrograph
1995 ─ Moore & Delcourt: "The Geopause" review
1996 ─ Stephen et al.: Fast-O beam from O- via cavity radiation
1998 ─ Aellig et al.: Cesiated converter surfaces for space
1999 ─ Moore et al.: Ionospheric mass ejection response to CME
2000 ─ ★ LENA paper / IMAGE launch (this work)
2002+ ─ Operational LENA observations of ionospheric outflow
```

---

## 3. 필요한 배경 지식 / Prerequisites

- **전리권 outflow 물리** — Polar wind, cleft ion fountain, auroral acceleration; bi-Maxwellian heated distributions; H+ vs O+ flux scaling; Yau et al. (1988) outflow parametrization
- **Charge exchange 기본** — H+ + H, H+ + O, O+ + H, O+ + O 반응의 cross section과 그것이 ENA를 어떻게 생성하는지
- **표면 산란/이온화 물리** — Auger neutralization, resonant electron transfer; electron affinity (H: 0.75 eV, O: 1.46 eV); 일함수(work function)와 전자 친화도의 관계
- **Time-of-flight 분광법** — TOF principle: $t = L\sqrt{m/(2qV)}$; carbon foil energy loss 및 angular straggling; start/stop coincidence
- **Electrostatic ion optics** — Immersion lens, spherical ESA, broom magnet for electron rejection; energy-per-charge dispersion
- **Microchannel plate detectors** — Chevron stack, gain, dead time, wedge-and-strip anode position sensing
- **IMAGE 미션 구조** — Spin-stabilized 위성, 1.25×8 R_E orbit, 120 s spin period, 동시 작동 imaging payload

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| LENA | Low-Energy Neutral Atom (이미저 본체) |
| ENA | Energetic Neutral Atom — 충전 교환으로 생성된 빠른 중성원자 / Fast neutral produced by charge exchange |
| Conversion surface (CS) | Polished W (텅스텐) 표면, –20 kV 인가; incident neutrals의 일부를 반사하며 negative ion으로 전환 / Polished tungsten plate at –20 kV that reflects incident neutrals while converting a fraction to negative ions |
| Conversion efficiency η | $\eta = A^-/A_{inc}$, 입사 중성속에 대한 negative ion 비율 / Ratio of negative-ion to incident-neutral flux |
| CPR (Charged Particle Rejector) | Collimator + bias plates, ≤100 keV ions/electrons rejection / Vanes biased to ±8 kV reject charged particles below ring-current energies |
| IXL (Ion Extraction Lens) | Immersion lens; CS 후방에서 negative ion을 가속·focus·energy disperse / Accelerates and focuses negative ions, dispersing energy across slit S2 |
| ESA | Electrostatic Analyzer (truncated hemispherical) — UV 차단 + energy/charge 매핑 / Maintains energy dispersion while removing residual UV photons |
| ITOF | Imaging Time-Of-Flight — carbon foil + start/stop MCP + position-sensing anode / Carbon foil + chevron MCPs measuring m/q via flight time |
| Specular reflection | 입사각=반사각 거울 반사 (CS는 ⟨E_t⟩ ~ 0.6–0.8 E_i) / Mirror-like reflection at the conversion surface, with ⟨E_t⟩ ≈ 0.6–0.8 E_i |
| Sputtered hydrogen | O 원자가 W 표면 흡착물(주로 H₂O)을 두드려 떨어뜨리는 저에너지 H- / Low-energy H- knocked off adsorbates (mostly water) by incident O atoms |
| Cleft ion fountain | 자극 cusp 영역에서 가열·상승하는 O+ 분수 / Heated O+ outflow from the magnetic cusp region |
| Geocorona | 지구 외곽의 중성 H 가스 (Lα 1216 Å 광원이기도 함) / Earth's neutral H exosphere, also a Lyman-α photon source |

---

## 5. 수식 미리보기 / Equations Preview

**(1) Conversion efficiency**
$$\eta = \frac{A^-}{A_{\text{inc}}}$$
입사 중성 원자속에 대한 specularly-reflected negative ion 속의 비율. LENA 디자인 목표는 H에 대해 ~1×10⁻³ 정도. / Ratio of specularly-reflected negative-ion flux to incident neutral flux. LENA achieves ~10⁻³ for H at 1 keV.

**(2) Random TOF coincidence rate**
$$R_{12} = R_1 R_2 t$$
With $R_1 \sim R_2 \sim 100~\text{s}^{-1}$ and TOF window $t = 300$ ns: $R_{12} = 3 \times 10^{-3}$ s⁻¹. 우주 환경에서 noise floor를 결정. / Determines the random-coincidence noise floor in space.

**(3) Energy retention on CS reflection**
$$\langle E_t \rangle \approx (0.6 \text{–} 0.8) \cdot E_i$$
H의 경우 ~80%, O의 경우 ~60% 에너지가 보존됨. / Hydrogen retains ~80% and oxygen ~60% of its incident energy after reflection.

**(4) Charge-exchange reactions in the model (Eq. 2–6 in paper)**
$$\text{H}^+ + \text{H} \rightarrow \text{H} + \text{H}^+$$
$$\text{H}^+ + \text{O} \rightarrow \text{H} + \text{O}^+$$
$$\text{O}^+ + \text{H} \rightarrow \text{O} + \text{H}^+$$
$$\text{O}^+ + \text{O} \rightarrow \text{O} + \text{O}^+$$
이 반응들이 outflow 이온을 ENA로 전환하여 LENA가 그것을 본다. / These charge-exchange reactions convert outflowing ions into observable ENAs.

**(5) TOF time relation (implicit)**
$$t_{\text{TOF}} = L\sqrt{\frac{m}{2qV_{\text{post}}}}$$
With $L \sim$ flight path through ITOF, $V_{\text{post}} = 20$ kV. m/q discrimination of H vs O. / Mass identification via post-acceleration TOF.

---

## 6. 읽기 가이드 / Reading Guide

1. **§1 Introduction & §2 Science Objectives (155–157쪽)**: LENA가 왜 등장했는지, 어떤 과학적 질문에 답하는지에 집중. 특히 plasma heating 시간 척도가 in-situ로 풀리지 않는다는 점.
2. **§3 Instrument Description (157–166쪽)**: 광학 path를 따라가며 read — collimator/CPR → CS → IXL → ESA → ITOF. Figure 4와 Figure 8의 ray tracing이 핵심.
3. **§4 Operations (166–183쪽)**: §4.1 LENA Response가 가장 중요. Effective area vs energy (Figure 12), 절대 효율 ~10⁻⁴ for O/H. §4.2.3 Steering Controller에서 sputtered H 분리 방법.
4. **§4.4 Science Operations (180–183쪽)**: Figure 15 simulated auroral oval ENA image — 모델 가정과 결과 해석.
5. **Appendix A (183–187쪽)**: Conversion surface 물리. Cesiated W 시도와 폐기, polished W + 흡착물 결정 과정.
6. **Appendix B (187–193쪽)**: 전자장비 — 1차 통독에서는 skim 가능.

읽으면서 다음 질문을 염두에 두시오 / Keep these questions in mind:
- 왜 charge-exchange cell 대신 surface conversion인가?
- 왜 cesium 표면이 아닌 polished W (적셔진 흡착물 위)인가?
- LENA의 효율 ~10⁻³–10⁻⁴는 충분한가? (Figure 15 시뮬레이션 결과 확인)
- Sputtered H가 진짜 incident H signal과 어떻게 구분되는가?

---

## 7. 현대적 의의 / Modern Significance

LENA는 IMAGE 미션 동안(2000–2005) 자극 cleft 영역과 substorm 시간 척도의 plasma heating 변동을 처음으로 글로벌 원격 감지하는 데 성공했다. 후속 미션인 TWINS (2008–) 의 LENA-class imager, ENA imaging on Cassini/INCA, IBEX (2009–)의 heliospheric ENA imaging, 그리고 IMAP (2025) 미션의 IMAP-Lo/Hi/Ultra 모두 LENA의 surface-conversion 또는 carbon-foil ITOF 유산을 직접 이어받았다. 또한 polished tungsten 변환 표면 기술은 BepiColombo SERENA, Solar Orbiter HIS의 일부 design 결정에 영향을 주었다. 더 넓게는 LENA는 자기권 plasma escape (Yau, Andre, Strangeway 등) 연구에서 "globally heated, dynamically variable" 패러다임을 입증하는 도구가 되었다.

LENA's legacy directly enabled the TWINS mission's LENA-class imagers (2008–), influenced IBEX's heliospheric-ENA conversion-surface and TOF design (2009–), and established the technology lineage that continues into IMAP-Lo (NASA, 2025) and Solar Orbiter / BepiColombo neutral-particle instruments. Beyond hardware, LENA observations during 2000–2005 confirmed that ionospheric outflow varies on substorm timescales (<1 hour) and traced cleft, auroral, and polar-cap heating during CME-driven storms — providing a critical piece of evidence for the "geopause" picture in which terrestrial plasma populates a substantial fraction of the magnetosphere. This shifted how we model magnetospheric dynamics and is foundational for current geospace forecasting efforts.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
