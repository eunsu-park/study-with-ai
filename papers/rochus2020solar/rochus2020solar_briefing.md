---
title: "Pre-Reading Briefing: The Solar Orbiter EUI Instrument"
paper_id: "19_rochus_2020"
topic: Solar_Observation
date: 2026-04-17
type: briefing
---

# The Solar Orbiter EUI Instrument: The Extreme Ultraviolet Imager — Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Rochus, P., Auchère, F., Berghmans, D., et al., "The Solar Orbiter EUI instrument: The Extreme Ultraviolet Imager", *Astronomy & Astrophysics*, Vol. 642, A8 (2020). DOI: 10.1051/0004-6361/201936663
**Author(s)**: P. Rochus, F. Auchère, D. Berghmans, L. Harra, W. Schmutz, U. Schühle and the EUI consortium (CSL Liège, ROB, MPS, MSSL, PMOD, IAS, and partners)
**Year**: 2020

---

## 1. 핵심 기여 / Core Contribution

**한국어**
EUI(Extreme Ultraviolet Imager)는 ESA-NASA 공동 미션 Solar Orbiter(2020년 발사)에 탑재된 세 대의 극자외선(EUV) 영상기 패키지이다. 근일점 0.28 AU까지 접근하는 궤도를 활용하여, 기존 우주망원경(SOHO/EIT, TRACE, SDO/AIA)으로는 도달 불가능했던 **200 km 이하의 유효 공간 분해능**과 **1초 이하의 고케이던스(cadence)**, 그리고 **태양 황도면 이탈 이후의 고위도 관측**을 가능케 하도록 설계되었다. 본 논문은 (1) 전태양 영상기 FSI(Full-Sun Imager, 174/304 Å 이중 밴드, 3.8° 시야), (2) 고해상도 EUV 영상기 HRI_EUV(174 Å, 100 arcsec 내외 시야), (3) 고해상도 Lyman-α 영상기 HRI_Lya(1216 Å)의 세 채널에 대한 **광학 설계, 검출기(APS/CMOS), 필터, 전자부, 열설계, 지상 교정, 예상 성능**을 종합적으로 기술한다. 이 논문이 제시하는 설계 철학 — 소형·경량·저전력 + 방사선·열·먼지 환경에 대한 강건성 + APS 검출기의 고속 랜덤 액세스 — 은 이후 EUI가 발견한 "campfires"(편재하는 소규모 EUV 밝기)와 코로나 가열 문제 재조명의 기술적 토대가 되었다.

**English**
EUI is the three-channel extreme ultraviolet imager suite aboard ESA-NASA's Solar Orbiter mission (launched 2020). Exploiting the orbit's close perihelion (0.28 AU) and progressively higher heliographic inclinations, EUI is designed to deliver an **effective spatial sampling below 200 km on the Sun**, **sub-second to few-second cadence**, and **the first sustained high-latitude EUV imaging**, capabilities unavailable to predecessors (SOHO/EIT, TRACE, SDO/AIA). This instrument paper comprehensively describes the three telescopes — (1) the **Full-Sun Imager (FSI)** with a dual 174 Å / 304 Å passband and 3.8° field of view, (2) the **High Resolution Imager in EUV (HRI_EUV)** at 174 Å, and (3) the **High Resolution Imager in Lyman-α (HRI_Lya)** at 1216 Å — detailing optical concepts, back-thinned APS (CMOS) detectors, multilayer and thin-film filters, electronics, thermo-mechanical design, calibration, and expected on-orbit performance. EUI's design choices (compactness, low power, radiation/thermal hardening, APS random-access readout) later enabled the discovery of "campfires" — small, ubiquitous EUV brightenings relevant to the coronal heating problem.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

**한국어**
EUV 태양 영상은 1973년 Skylab을 시작으로, SOHO/EIT(1996), TRACE(1998), Hinode/XRT-EIS(2006), STEREO/EUVI(2006), SDO/AIA(2010)로 이어지며 분해능과 케이던스가 지속적으로 향상되어 왔다. 그러나 이들 모두 지구 궤도(약 1 AU)에서 관측되었고, 이는 **플라즈마 구조를 구성하는 기본 길이 스케일(수십~수백 km)을 직접 분해하기 어렵다는 한계**를 의미했다. 동시에 황도면 관측에 국한되어 **극지역의 공간 구조·자기장 진화**를 선명하게 볼 수 없었다. Ulysses(1990년대, 극궤도)는 인시투 관측만 가능했고, 원격 영상은 상실된 상태였다. Solar Orbiter는 이 두 간극을 동시에 메우는 미션으로 기획되었으며, EUI는 그 원격 영상 패키지 중 핵심이다.

**English**
EUV solar imaging began with Skylab (1973) and advanced through SOHO/EIT (1996), TRACE (1998), Hinode/XRT-EIS (2006), STEREO/EUVI (2006), and SDO/AIA (2010), each improving resolution and cadence. Yet all observed from ~1 AU and from the ecliptic plane — too far to resolve the fundamental length scales of coronal plasma (tens to hundreds of km) and too flat to image the poles. Ulysses (1990s, polar orbit) carried only in-situ instruments, leaving a gap in high-latitude remote sensing. Solar Orbiter was conceived to close both gaps; EUI is the mission's EUV remote-sensing cornerstone.

### 타임라인 / Timeline

```
1973 ─ Skylab ATM (first extended EUV imaging)
1995 ─ SOHO launch; EIT begins routine 4-band EUV imaging (1996)
1998 ─ TRACE (1 arcsec, single-passband EUV)
2006 ─ Hinode (XRT, EIS); STEREO (EUVI on two ecliptic spacecraft)
2010 ─ SDO/AIA (7 EUV channels, 0.6 arcsec, 12 s cadence, geosynchronous)
2011 ─ Solar Orbiter EUI PDR (Preliminary Design Review) era
2017–19 ─ EUI Flight Model integration, vacuum calibration (PTB/BESSY, RAL)
2020 Feb ─ Solar Orbiter launch (Cape Canaveral, Atlas V 411)
2020 Jun ─ First Light; "campfires" reported from HRI_EUV
2020 ─ THIS PAPER (A&A 642, A8) published as instrument description
2022 ─ First perihelion ≤0.3 AU; full science phase begins
```

---

## 3. 필요한 배경 지식 / Prerequisites

**한국어**
- **논문 #12, #18 (이전 EUV/자외선 영상기 계통 논문)**: SOHO/EIT와 SDO/AIA의 다층막 EUV 영상 원리를 이해하면 EUI 설계 선택의 근거가 명확해진다.
- **APS (CMOS) 검출기**: CCD와 달리 픽셀마다 증폭기가 있어 **랜덤 액세스**, **고속 읽기**, **낮은 소비전력**, **방사선 내성**이 특징. EUI가 Solar Orbiter의 열·전력·대역폭 제약 하에 선택한 핵심 기술.
- **EUV 광학**: EUV(100–1000 Å)는 모든 물질에 강하게 흡수되므로 굴절 광학 불가능. **다층막(Multilayer) 반사경**(예: Mo/Si for 174 Å, SiC/Mg for 304 Å)과 **박막 필터**(Al, Zr)를 사용.
- **소형 망원경 설계**: 고전적 Ritchey-Chrétien 대신 오프-축 Gregorian, 단일 반사경, 슈바르츠실트(Schwarzschild) 구성 등이 Solar Orbiter의 질량·부피 제약에 맞게 채택됨.
- **Lyman-α (1216 Å) 이미징**: 상부 채층(chromosphere) 방출의 주요 선. 수소 공명선으로 광학 두께 효과가 크다.
- **열설계**: 0.28 AU에서 태양상수는 ~13배 증가. 입사구(entrance aperture)의 **열거절 필터(heat rejection window)**와 **방열판** 설계가 핵심.
- **Solar Orbiter 궤도**: 근일점 0.28 AU, 금성 중력 보조로 경사각 ~33° 확보, 임무 수명 7–10년.

**English**
- **Papers #12, #18 (prior EUV/UV imager lineage)**: SOHO/EIT and SDO/AIA establish multilayer-based EUV imaging; EUI's design choices are readable against their trade-offs.
- **APS (CMOS) detectors**: Per-pixel amplifiers enable **random-access windowing**, **high frame rates**, **low power**, and **radiation hardness** — essential under Solar Orbiter's mass/power/downlink budget.
- **EUV optics**: At 100–1000 Å, every material absorbs strongly → no refractive optics. **Multilayer mirrors** (Mo/Si for 174 Å, SiC/Mg for 304 Å) and **thin-film filters** (Al, Zr) are the building blocks.
- **Compact telescope design**: Off-axis Gregorian, single-mirror, and Schwarzschild layouts replace classical Ritchey-Chrétien to meet Solar Orbiter's envelope.
- **Lyman-α (1216 Å) imaging**: Dominant chromospheric line; optically thick resonance transition of hydrogen.
- **Thermal design**: Solar flux at 0.28 AU is ~13× Earth-orbit. Entrance **heat-rejection windows** and radiator design are critical.
- **Solar Orbiter orbit**: Perihelion 0.28 AU; Venus gravity assists raise inclination to ~33°; 7–10 yr mission.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **FSI** (Full-Sun Imager) | 전태양을 한 시야(3.8°)에 담는 듀얼 밴드(174/304 Å) EUV 영상기. Single-mirror off-axis Herschel 구성. / Dual-band (174/304 Å) full-disk EUV imager with 3.8° FOV; off-axis Herschel single-mirror. |
| **HRI_EUV** | 174 Å 단일 밴드 고해상도 EUV 영상기. 약 1000×1000″ 시야, 0.28 AU에서 픽셀 풋프린트 ~100 km. / High-resolution 174 Å EUV imager; pixel footprint ≈100 km at 0.28 AU. |
| **HRI_Lya** | 1216 Å Lyman-α 고해상도 영상기. 상부 채층 구조 관측. / High-resolution Lyman-α imager for upper chromosphere. |
| **Multilayer mirror** | 수십 쌍의 Mo/Si (또는 SiC/Mg) 박막으로 구성된 EUV 반사경. 간섭에 의해 특정 파장만 반사. / Stack of tens of Mo/Si (or SiC/Mg) bilayers reflecting a narrow EUV band by Bragg interference. |
| **APS / CMOS detector** | 픽셀당 증폭기를 내장한 능동형 화소 센서. EUI는 back-thinned 10 μm 픽셀, 3k×3k 어레이 사용. / Active-pixel CMOS sensor with per-pixel amplifiers; EUI uses back-thinned 10 μm, 3k×3k arrays. |
| **Heat rejection window** | 입사구의 투명한 열차단창. 가시광을 반사/흡수하여 EUV만 통과시키는 엔트런스 필터. / Entrance-aperture heat-rejection filter: transmits EUV while rejecting visible/IR heat load. |
| **Al / Zr thin-film filter** | EUV만 통과시키고 가시광·UV를 차단하는 ~150 nm 금속 박막. / ~150 nm metallic films passing EUV and blocking visible/UV. |
| **Doorplate mechanism** | 발사·초기운용·안전 모드에서 광경로를 보호하는 1회성 도어. / Protective door for launch/safe-mode, opened in-flight. |
| **Occulter / entrance baffle** | 직접 태양광을 최소화하는 배플·차폐 구조. / Baffles to suppress direct/scattered sunlight. |
| **Campfires** | EUI HRI_EUV가 발견한 편재하는 소형 EUV 밝기(1–4 Mm, 10–200 s). 본 논문은 발견 이전 설계 문서이나 이를 가능케 한 기술 기반을 기술. / Ubiquitous small EUV brightenings (1–4 Mm, 10–200 s) later discovered with HRI_EUV; this paper predates the discovery but describes the enabling technology. |
| **RPW-EUI co-observation** | Solar Orbiter의 원격·인시투 기기 동시 운용을 통한 연결 관측. / Coordinated remote+in-situ observing campaigns. |
| **Flat-field / on-board calibration** | LED·필터휠을 통한 궤도 상 교정. / In-flight calibration via LEDs and filter-wheel positions. |

---

## 5. 수식 미리보기 / Equations Preview

### (1) 픽셀 풋프린트 (Plate scale at distance $d$)

$$
\text{Pixel footprint} \;=\; d \cdot \tan(\theta_{\text{pix}}) \;\approx\; d \cdot \theta_{\text{pix}}
$$

**한국어** $\theta_{\text{pix}}$는 픽셀이 덮는 각도(rad). 1 AU = 1.496×10⁸ km이므로 0.28 AU에서 1″ 픽셀은 지상 203 km에 해당한다. HRI_EUV의 1 pixel ≈ 0.492″ → 0.28 AU에서 **~100 km/pixel**. Nyquist 샘플링을 고려하면 유효 분해능 ~200 km.

**English** $\theta_{\text{pix}}$ is the angular size of one pixel (rad). At 0.28 AU, 1″ subtends 203 km; HRI_EUV's ~0.492″ pixel yields **~100 km/pixel footprint**, with Nyquist-limited resolution ~200 km.

---

### (2) 다층막 반사율 — Bragg 조건 (Multilayer peak wavelength)

$$
m\lambda \;=\; 2d\,\cos\theta
$$

**한국어** $d$는 박막 주기(period), $\theta$는 입사각, $m$은 차수. Mo/Si 다층막의 경우 $d \approx 9.9$ nm, $\theta \approx 0°$에서 $\lambda \approx 19.8$ nm = 198 Å로 조정되며, EUI 174 Å에 맞게 설계된다. 반사율 피크 폭(FWHM)은 약 ±5 Å 수준.

**English** With $d \approx 9.9$ nm and near-normal incidence, Mo/Si multilayers peak around 198 Å; EUI's coating is tuned for 174 Å, with a reflectivity FWHM of roughly ±5 Å.

---

### (3) 광자 플럭스 → DN 변환 (Radiometric model)

$$
S_{\text{DN}} \;=\; B_\lambda \,\Omega_{\text{pix}} \,A_{\text{eff}}(\lambda) \,\tau_{\text{filter}}(\lambda) \,\eta_{\text{QE}}(\lambda) \,\Delta t / g
$$

**한국어** $B_\lambda$ 스펙트럼 라디언스, $\Omega_{\text{pix}}$ 픽셀 고체각, $A_{\text{eff}}$ 유효 광학 면적, $\tau_{\text{filter}}$ 필터 투과율, $\eta_{\text{QE}}$ 검출기 양자효율, $\Delta t$ 노출, $g$ DN당 전자수(gain). 논문은 각 채널의 $A_{\text{eff}}$와 예상 신호 대 잡음을 표로 제시.

**English** Standard radiometric chain: source radiance × pixel solid angle × effective area × filter transmission × QE × exposure / gain, yielding counts per pixel. The paper tabulates $A_{\text{eff}}(\lambda)$ and expected SNR for each channel.

---

### (4) 근일점 열유속 (Perihelion heat flux)

$$
F(r) \;=\; F_\oplus \left(\frac{1\,\text{AU}}{r}\right)^2
$$

**한국어** 0.28 AU 근일점에서 $F = F_\oplus/0.28^2 \approx 12.8\,F_\oplus \approx 17{,}400\,\text{W/m}^2$. EUI 입사구의 열거절 창은 이 부하를 감당하도록 설계된다.

**English** At 0.28 AU, the solar flux is $F_\oplus/0.28^2 \approx 12.8\,F_\oplus \approx 17{,}400\,$W/m². Entrance heat-rejection windows must survive this thermal load.

---

### (5) 회절한계 분해능 (Diffraction-limited angular resolution)

$$
\theta_{\text{diff}} \;=\; 1.22 \,\frac{\lambda}{D}
$$

**한국어** HRI_EUV ($\lambda=174$ Å, $D \approx 4$ cm 유효 구경)에서 $\theta_{\text{diff}} \approx 0.11″$. 픽셀 샘플링(0.49″)은 회절한계보다 굵으므로 **검출기 샘플링이 분해능의 병목**이다. HRI_Lya는 $\lambda=1216$ Å로 회절한계가 더 크다.

**English** For HRI_EUV ($\lambda=174$ Å, $D\approx 4$ cm), $\theta_{\text{diff}} \approx 0.11″$; pixel sampling (0.49″) dominates. For HRI_Lya at 1216 Å, diffraction is the limiting factor.

---

## 6. 읽기 가이드 / Reading Guide

**한국어**
이 논문은 전형적인 **instrument paper**로, 순수 과학 논문과 달리 "이 기기가 무엇을, 어떻게, 얼마나 정확히 관측하는가"의 순서로 읽으면 효율적이다. 다음 순서를 추천한다.

1. **Sec. 1 (Introduction) + Sec. 2 (Scientific objectives)**: EUI가 Solar Orbiter 과학 목표 중 어떤 질문(코로나 가열, 태양풍 기원, 자기 활동, 극지 이해)에 대응하는지 파악. 전체 설계를 이해하는 열쇠.
2. **Sec. 3 (Instrument overview)**: 세 채널의 개념도와 파라미터 표. **핵심 표(FOV, 픽셀, 케이던스, 중량, 전력)를 반드시 메모**.
3. **Sec. 4 (Optical design)**: 각 채널의 광학 구조. Herschel/Schwarzschild/Ritchey-Chrétien 중 무엇을 썼는지, 왜인지.
4. **Sec. 5 (Detectors) + Sec. 6 (Filters, multilayers)**: APS 상세, 다층막 설계. 이후 campfire 연구가 왜 가능한지 기술적 근거.
5. **Sec. 7 (Mechanisms, thermal, electronics)**: 도어, 필터휠, 열 제어, 데이터 처리. 엔지니어링 관심이 있다면 깊이 읽고, 과학 중심이면 요점만.
6. **Sec. 8 (Calibration)**: PTB/BESSY 방사 교정 결과. 이후 데이터의 절대 단위 신뢰성을 판단할 때 참조.
7. **Sec. 9 (Expected performance, in-flight commissioning)**: 예상 SNR, 첫 빛(first light) 결과. 실제 관측 논문과 연결.
8. **부록(Appendix)**: 파라미터 요약표. 구현/분석 시 가장 많이 참조.

읽는 동안 **"이 설계 선택이 왜 SDO/AIA 등 기존 기기와 다른가?"**를 지속적으로 질문하라. 예: 왜 FSI는 단일 반사경인가? 왜 HRI_EUV는 Ritchey-Chrétien이 아니라 Gregorian인가? 왜 Lyman-α 채널을 넣었나?

**English**
This is an **instrument paper**; read it asking what the instrument measures, how, and with what fidelity.

1. **§1–2 Introduction + Science objectives** — frames EUI's role in addressing coronal heating, solar-wind origin, and polar dynamics.
2. **§3 Instrument overview** — note the summary table (FOV, pixel, cadence, mass, power).
3. **§4 Optical design** — which topology (Herschel/Schwarzschild/RC) per channel and why.
4. **§5–6 Detectors & filters/multilayers** — the technical enablers of campfire-scale imaging.
5. **§7 Mechanisms, thermal, electronics** — skim unless engineering-focused.
6. **§8 Calibration** — determines credibility of later absolute photometry.
7. **§9 Expected performance & first light** — transitions to science papers.
8. **Appendix / summary tables** — your go-to reference for implementation.

Throughout, keep asking: *why does this design differ from SDO/AIA / SOHO/EIT?* (Single-mirror FSI, Gregorian HRI_EUV, Lyman-α channel, APS detectors, heat-rejection windows.)

---

## 7. 현대적 의의 / Modern Significance

**한국어**
EUI는 2020년 6월 **첫 빛 직후 "campfires"라 이름 붙은 소규모 EUV 밝기 현상**(Berghmans et al. 2021)을 발견했고, 이는 1988년 Parker가 제안한 **나노플레어(nanoflare) 코로나 가열 가설**의 관측적 재점화를 가져왔다. Solar Orbiter의 고위도 통과(2025–)와 맞물려 EUI는 **최초의 태양 극지역 고해상 EUV 영상**을 제공한다. 또한 Parker Solar Probe와의 공조 관측(Solar Orbiter가 원격, PSP가 인시투)은 **태양풍의 원천 영역을 동시 촬영/측정**하는 전례 없는 데이터셋을 만든다. 2026년 현재 EUI 데이터는 **기계학습 기반 superflare 감지, 자기 재결합 사이트 탐색, 코로나 파동 연구**의 표준 자료원이 되었고, 차세대 MUSE, Solar-C(EUVST)의 설계 벤치마크이기도 하다.

**English**
Immediately after first light (Jun 2020), EUI discovered **"campfires"** (Berghmans et al. 2021) — ubiquitous small-scale EUV brightenings that re-ignited the debate on **Parker's (1988) nanoflare coronal-heating hypothesis**. With Solar Orbiter's increasing orbital inclination, EUI now returns **the first high-resolution EUV imagery of the solar poles** and forms, together with Parker Solar Probe's in-situ suite, an unprecedented joint remote-sensing + in-situ dataset probing solar-wind source regions. By 2026, EUI data underpin ML-based flare detection, reconnection-site surveys, and coronal-wave studies, and set the design benchmark for future missions such as MUSE and Solar-C (EUVST).

---

## Q&A

### Q1. EUI가 174 Å (FSI는 174 + 304) 만 탑재한 설계 의도는? / Why only 174 Å (and 174 + 304 for FSI)?

**한국어**
핵심은 **"최소 채널로 태양의 수직 온도층 커버 + 페이로드 내 역할 분담 + SO 고유의 자원 제약"** 이다.

- **174 Å (Fe IX/X, T ≈ 1 MK)**: 조용한 코로나의 정규 밴드. SNR이 높아 고케이던스(≤1 s) 가능 → **campfire(10–200 s 수명)** 관측 가능. Mo/Si 다층막 성숙 기술(반사율 ~40%). TRACE 171·AIA 171·EIT 171과 직접 비교 가능.
- **304 Å (He II, T ≈ 5×10⁴ K)**: 전이영역/상부 채층. FSI에서 필터휠로 174/304 번갈아 촬영 → **단일 광학계로 듀얼 밴드** 실현(질량/전력 절감). 필라멘트·프로미넌스·CME 개시영역의 채층 뿌리 추적.
- **AIA 스타일 다채널(193/211/335/94/131) 미채택 이유**:
  1. *질량·전력·부피 제약*: SO 기기 엔벨로프가 SDO의 1/5 이하. 4-망원경 구성 불가.
  2. *텔레메트리 제약*: SO는 근일점에서도 지구 다운링크가 수십 kbps–Mbps 수준. 채널 수 × 데이터량은 치명적.
  3. *SPICE와 분업*: 동일 탑재체 SPICE가 C III/O VI/Ne VIII/Mg IX/Fe XVIII 등 다온도 라인을 분광으로 제공. EUI가 다온도 진단을 중복할 필요 없음 → EUI는 **이미징 특화 (공간·시간 분해능)**.
  4. *0.28 AU의 이점*: 태양 플럭스 ~13× → 저광자 채널을 굳이 안 넣어도 주요 현상 포착 가능.
  5. *Lyman-α가 cool 구조 커버*: HRI_Lya(1216 Å)가 채층을 담당하여 174/304만으로 못 보는 영역을 보완.

결론: **EUI = imaging specialist (174 코로나 + 304 전이영역 + Lyα 채층), SPICE = spectroscopy specialist (다온도 진단)** — 역할 분담으로 채널을 최소화했다.

**English**
EUI's narrow channel selection reflects three overlapping constraints and a conscious division of labor within the Solar Orbiter remote-sensing payload.

- **174 Å (Fe IX/X, T ≈ 1 MK)** is the canonical quiet-corona band: highest SNR of any EUV passband, enabling sub-second cadence and detection of short-lived (10–200 s) small brightenings — the "campfires" discovered at first light. Mo/Si multilayer technology is mature (~40% reflectivity), and TRACE/AIA/EIT 171 provide direct comparison baselines.
- **304 Å (He II, T ≈ 5×10⁴ K)** covers the transition region / upper chromosphere. FSI alternates 174/304 via a filter wheel, realizing dual-band imaging in a **single telescope** — a major saving in mass and power. 304 Å reveals filament/prominence structure and the chromospheric roots of CMEs.
- **Why no AIA-style multi-temperature channels (193/211/335/94/131)?**
  1. *Mass/power/volume*: SO's instrument envelope is ~1/5 of SDO's; AIA-style 4-telescope arrays are not feasible.
  2. *Telemetry*: SO's downlink is only tens of kbps–few Mbps; each added channel multiplies data volume.
  3. *Role sharing with SPICE*: The onboard EUV spectrometer SPICE delivers multi-temperature diagnostics (C III, O VI, Ne VIII, Mg IX, Fe XVIII). EUI specializes in imaging — spatial/temporal resolution — rather than thermal diagnostics.
  4. *0.28 AU advantage*: Solar flux is ~13× higher, relaxing SNR pressure and removing the need for faint-line channels.
  5. *Lyman-α complement*: HRI_Lya (1216 Å) independently covers the cool chromosphere.

Net design: **EUI = imaging specialist (174 corona + 304 TR + Lyα chromosphere); SPICE = spectroscopy specialist (multi-T diagnostics)** — channels are deliberately minimized by splitting roles.

---

### Q2. Solar Orbiter 미션 개요 논문은 읽기 목록에 있는가? / Is the Solar Orbiter mission overview paper on the reading list?

**한국어**
**현재는 없음.** Solar_Observation 목록에는 Solar Orbiter 관련 기기 논문 2편만 있다:
- #18 Anderson et al. 2020 — SPICE (EUV spectrometer)
- #19 Rochus et al. 2020 — EUI (본 논문)

다른 미션과 비교하면 불균형하다:
| 미션 / Mission | Overview 논문 / Overview paper |
|---|---|
| SOHO | #8 Domingo et al. 1995 ✓ |
| SDO | #35 Pesnell et al. 2012 ✓ |
| Solar Orbiter | **(없음 / missing)** ✗ |

**추천**: **Müller, D., St. Cyr, O. C., Zouganelis, I., et al. 2020, "The Solar Orbiter mission — Science overview", A&A 642, A1** 을 추가하는 것을 권장. 기존 #18 앞(새로운 #18)에 삽입하여 "미션 개요 → 기기 논문" 순서로 읽는 흐름이 자연스럽다. 끝에 붙이는 방식도 가능.

**English**
**Not currently.** Only two Solar Orbiter instrument papers are on the list (#18 SPICE, #19 EUI); the mission overview is missing, unlike SOHO (#8) and SDO (#35) which both have overview entries. Recommended addition: **Müller et al. 2020, A&A 642, A1** — ideally inserted before the current #18 so the reading order is mission overview → instruments.
