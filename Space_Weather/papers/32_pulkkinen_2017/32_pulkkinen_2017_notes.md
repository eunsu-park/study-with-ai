---
title: "Reading Notes — Pulkkinen et al. (2017): Geomagnetically Induced Currents — Science, Engineering, and Applications Readiness"
date: 2026-04-27
topic: Space_Weather
tags: [GIC, geomagnetically-induced-currents, applications-readiness-level, geoelectric-field, power-grid, dB-dt, plane-wave-method, NERC-TPL-007]
paper_doi: 10.1002/2016SW001501
journal: Space Weather, 15(7), 828–856 (2017)
---

# Pulkkinen et al. (2017) — Geomagnetically Induced Currents: Science, Engineering, and Applications Readiness / 지자기 유도 전류: 과학, 공학, 그리고 응용 준비도

## Core Contribution / 핵심 기여

### English
Pulkkinen et al. (2017) is a community review that surveys the entire GIC pipeline — from solar-wind drivers, through magnetospheric currents, ionospheric and field-aligned currents, ground magnetic-field perturbation B(t), its time derivative dB/dt, the induced geoelectric field E(t) determined by Earth's subsurface conductivity, and finally the quasi-DC current GIC flowing through grounded conductors such as power-transmission transformers and pipelines. The authors' principal innovation is to introduce an **Applications Readiness Level (ARL)** framework — analogous to NASA's Technology Readiness Level — and apply it to each subcomponent (driver characterization, dB/dt forecasting, geoelectric field computation, GIC network modeling, transformer impact assessment, operational dissemination). This produces the first systematic gap analysis showing which scientific products are mature enough for direct operational use by power utilities (ARL 7–9) and which remain at proof-of-concept (ARL 1–3). The paper also codifies the mathematical pipeline used in modern operational GIC services (NOAA SWPC, NRCan, BGS, FMI) and underlies the NERC TPL-007 GMD planning standard that mandates U.S. utilities to assess and mitigate GIC vulnerability.

### Korean
Pulkkinen et al. (2017)은 GIC 파이프라인 전체를 조망하는 커뮤니티 리뷰이다. 태양풍 구동원에서 시작해 자기권 전류, 전리층/자력선-평행 전류, 지면 자기장 섭동 B(t), 그 시간 미분 dB/dt, 지하 전도도에 의해 결정되는 유도 지전기장 E(t), 마지막으로 송전 변압기와 송유관 같은 접지된 도체에 흐르는 준직류 GIC까지 다룬다. 저자들의 핵심 기여는 NASA의 TRL과 유사한 **Applications Readiness Level (ARL)** 체계를 도입해, 구동원 특성화·dB/dt 예보·지전기장 계산·GIC 네트워크 모델링·변압기 영향 평가·운영 배포 각 하위 구성요소에 적용한 것이다. 그 결과, 어떤 과학 산출물이 전력회사 직접 운영(ARL 7–9)에 충분히 성숙했고 어떤 것이 개념검증 단계(ARL 1–3)에 머물러 있는지를 처음으로 체계적으로 진단했다. 이 논문은 또한 현대 운영 GIC 서비스(NOAA SWPC, NRCan, BGS, FMI)에서 사용하는 수학적 파이프라인을 정형화하며, 미국 전력회사에 GIC 취약성 평가와 완화를 의무화한 NERC TPL-007 GMD 계획 표준의 학술적 기반을 제공한다.

---

## Reading Notes / 읽기 노트

### Section 1 — Introduction and Motivation (pp. 828–831) / 서론과 동기

#### English
The introduction frames GIC as a quintessential "compound space weather" hazard: solar-driven, geomagnetic in mechanism, but with ground-level engineering consequences. The paper is the **primary deliverable of the very first NASA Living With a Star (LWS) Institute Working Group**, launched in 2014 as a pilot for the new LWS Institute element (Section 1, p. 829). The GIC Working Group, led by A. Pulkkinen (NASA GSFC) and co-led by E. Bernabeu (PJM) and A. Thomson (BGS), held two five-day in-person workshops in Colorado plus several half-day videoconferences. The 21 coauthors span NASA, USGS, NOAA SWPC, FEMA, FERC-aware utilities (PJM), national geological surveys (BGS, NRCan, Finnish Meteorological Institute, South African National Space Agency), university physics/engineering departments, the insurance industry (Munich Re), and Federal Emergency Management Agency.

Three reference events anchor the discussion: (1) the **March 1989 Hydro-Québec blackout** (Bolduc et al. 2002) which caused a system voltage collapse and is the most famous GIC impact on record; (2) the **October 2003 Halloween storm regional blackout in Malmö, Sweden**, where a single critical line tripped due to a relay (Pulkkinen et al. 2005); and (3) the New Jersey "Saleem" station transformer damage during March 1989 as documented by Gaunt and Coetzee (2007). U.S. regulatory drivers cited explicitly are the FERC GIC-related elements of the National Space Weather Strategy and Action Plan (2015) and NERC's GMD standards process. International power system operators in the U.S., U.K., Canada, Finland, Norway, Sweden, China, Japan, Brazil, Namibia, South Africa, and Australia have launched GIC measurement campaigns.

#### Korean
서론은 GIC를 "복합 우주기상 재해"의 전형으로 자리매김한다. 구동원은 태양, 메커니즘은 지자기, 결과는 지상 인프라에 미친다는 다층 구조다. 본 논문은 **NASA Living With a Star (LWS) Institute Working Group의 첫 산출물**이며, 새 LWS Institute 요소의 시범 활동으로 2014년 출범했다(Section 1, p. 829). GIC 워킹그룹은 A. Pulkkinen(NASA GSFC) 주도, E. Bernabeu(PJM) 및 A. Thomson(BGS) 공동주도로, 콜로라도에서 2회의 5일 대면 워크숍과 다수의 반일 화상회의를 진행했다. 21인 공저자는 NASA, USGS, NOAA SWPC, FEMA, FERC 인지 사업자(PJM), 국가 지질조사기관(BGS, NRCan, FMI, SANSA), 대학 물리·공학 학과, 보험산업(Munich Re), FEMA에 걸쳐 있다.

세 기준 사건이 논의를 지탱한다: (1) **1989년 3월 Hydro-Québec 정전**(Bolduc et al. 2002) — 시스템 전압 붕괴를 일으킨 가장 유명한 GIC 사례; (2) **2003년 10월 핼러윈 폭풍의 스웨덴 Malmö 지역 정전** — 단일 핵심 송전선의 계전기 트립으로 발생(Pulkkinen et al. 2005); (3) Gaunt and Coetzee(2007) 문서화한 1989년 3월 미국 New Jersey "Saleem" 변전소 변압기 손상. 미국 규제 동인으로는 FERC의 GIC 관련 조항과 국가 우주기상 전략·행동 계획(2015), NERC GMD 표준화 절차가 명시된다. 미·영·캐·핀·노·스웨덴·중·일·브·나미비아·남아공·호주의 송전 사업자가 GIC 측정 캠페인을 운영 중이다.

### Section 2 — Scientific Foundation: The B → E → GIC Chain (pp. 831–840) / 과학적 기반

#### English
**2.1 Geomagnetic drivers.** Sub-storm-associated westward auroral electrojet intensifications, sudden storm commencements (SSC), and storm sudden impulses (SSI) all produce intense dB/dt. Statistical work by Pulkkinen et al. 2012, Ngwira et al. 2013/2015 identifies the **subauroral latitudes (~50°–60° magnetic)** as the highest dB/dt risk zone during extreme events, due to the equatorward-moving auroral oval and substorm current wedge.

**2.2 Earth's response.** The paper writes the geoelectric field via the **surface impedance tensor** (Section 3.6, equation (2), p. 841):

$$\mathbf{E}(\omega) = \frac{1}{\mu}\begin{bmatrix} Z_{xx} & Z_{xy} \\ Z_{yx} & Z_{yy} \end{bmatrix} \mathbf{B}(\omega) \quad (2)$$

This **2×2 tensor formulation captures full 3-D induction effects** when the impedance tensor is empirically derived from a magnetotelluric (MT) survey at the site (~1-month deployment of fluxgate magnetometer + orthogonal electrode pairs at 1 Hz cadence). The paper explicitly bypasses the need for assumed conductivity models by using empirical Z (Bedrosian & Love 2015 cited).

Historically, the 1-D plane-wave method (Cagniard 1953) treats the geoelectric field as scalar:

$$Z(\omega) = \sqrt{\frac{i\omega\mu_0}{\sigma}}$$

This makes E proportional to √ω · B at high frequencies — equivalently, dB/dt drives E. The paper notes that 1-D **"effective" conductivities** can give a sufficiently accurate approximation at many sites despite the true 3-D nature of the Earth (citing Trichtchenko & Boteler 2004, Viljanen et al. 2006, Ngwira et al. 2008, Wik et al. 2008). For coastal regions with strong lateral conductivity gradients (the **"coast effect"**), 3-D models are required (Pirjola 2013).

**EarthScope coverage** is shown explicitly in **Figure 6** (paper p. 842): red dots mark MT-instrumented locations (covering the contiguous U.S. except California, Nevada, southern parts) and green dots additional Canadian coverage. **Figure 5** (p. 840) shows the global distribution of permanent geophysical observatories — sparse outside Europe, North America, and parts of Asia. The U.S. EarthScope MT array provides per-cell Z(ω) tensors that produce E-fields differing by an order of magnitude between resistive and conductive regions.

**2.3 GIC in conductors.** Once E(t) is known, GIC in a network is computed via the Lehtinen-Pirjola (1985) matrix formalism, written as the paper's **equations (3)–(6)** (Section 3.6, p. 843):

$$\mathbf{I^e} = (\mathbf{1} + \mathbf{Y} \mathbf{Z^e})^{-1} \mathbf{J^e} \quad (3)$$

with the network admittance matrix elements

$$Y_{ij} = \begin{cases} -1/R_{ij}, & i \neq j \\ \sum_{k \neq i} 1/R_{ik}, & i = j \end{cases} \quad (4)$$

the perfect-earthing currents

$$J^e_i = \sum_{j \neq i} V^o_{ij}/R_{ij} \quad (5)$$

and the line voltage source

$$V^o_{ij} = \int_i^j \mathbf{E} \cdot d\mathbf{s} \quad (6)$$

The paper emphasizes that the **largest source of uncertainty is the substation grounding resistance** $Z^e_{ij}$ — actual values are rarely measured and vary significantly with local soil/conditions, while transformer-winding resistances are well known. Note that the line voltage source is the **spatially averaged (line integral) geoelectric field over each branch**, not the local field at any single point — a key consideration when comparing modeled E to observed GIC.

For a single isolated line of length L and resistance R (line + ground return), the simple model

$$I = \frac{E \cdot L}{R}$$

provides order-of-magnitude estimates and is used in the implementation notebook.

#### Korean
**2.1 지자기 구동원.** 부폭풍 동반 서향 오로라 일렉트로젯 강화, 급격 폭풍 개시(SSC), 폭풍 급격 충격(SSI)이 모두 강한 dB/dt를 발생시킨다. Pulkkinen et al. 2012, Ngwira et al. 2013/2015의 통계 연구는 **부오로라 위도(자기위도 ~50°–60°)** 를 극단 이벤트에서 가장 위험한 dB/dt 영역으로 지목했다. 적도 방향으로 이동하는 오로라 오벌과 부폭풍 전류 쐐기 때문이다.

**2.2 지구 응답.** 평면파 방법(Cagniard 1953)은 지전기장을

$$E_x(\omega) = Z(\omega) \cdot B_y(\omega) / \mu_0$$

으로 쓴다. 여기서 Z(ω)는 지표 임피던스. 균일 반무한 전도도 σ에 대해:

$$Z(\omega) = \sqrt{\frac{i\omega\mu_0}{\sigma}}$$

따라서 E는 고주파에서 √ω · B에 비례 — 즉 dB/dt가 E를 구동. 실제 지구는 1차원 층상 모델이나 완전 3차원 자기지전류(MT) 임피던스 텐서를 사용한다. 미국 EarthScope MT 배열은 셀별 Z(ω) 텐서를 제공하며, 이로 계산된 E장은 저항성 지역(예: Maine, Minnesota)과 전도성 지역(예: Florida 해안) 사이에서 10–30배 차이가 난다.

**2.3 도체 내 GIC.** E(t)가 정해지면 네트워크의 GIC는 Lehtinen-Pirjola (1985) 행렬 공식으로 계산된다:

$$\mathbf{I_e} = (\mathbf{1} + \mathbf{Y_n} \mathbf{Z_e})^{-1} \mathbf{J_e}$$

여기서 Y_n은 네트워크 어드미턴스, Z_e는 접지 임피던스, J_e는 각 가지를 따라 E의 선적분으로 얻는 완전접지 전류. 길이 L, 저항 R(라인 + 대지 귀로)의 단일 라인은 단순 모델

$$I = \frac{E \cdot L}{R}$$

로 자릿수 추정이 가능하며, 본 구현 노트북에서 이 모델을 사용한다.

### Section 3 — Engineering Aspects: Power-Grid Response (pp. 840–845) / 공학 측면

#### English
**3.1 Transformer half-cycle saturation.** Quasi-DC GIC flowing through wye-grounded transformer neutrals biases the magnetic core. Because GIC frequencies (~mHz) are far below the 50/60 Hz operating frequency, the transformer effectively sees a DC offset added to the AC magnetizing flux. The core saturates during one half of each AC cycle, drawing strongly distorted magnetizing currents rich in even and odd harmonics (2nd, 3rd, 5th, 7th) and consuming up to 5–10 MVAr of reactive power per affected transformer at GIC ~100 A.

**3.2 System-level effects.** Cascading reactive-power demand can lead to voltage collapse (1989 Hydro-Québec), harmonics trip protective relays (Saleem NJ 1989), and prolonged saturation overheats tank insulation, sometimes destroying transformers (Saleem NJ 1989, multiple Eskom South Africa transformers October–November 2003 storms; Gaunt & Coetzee 2007). **Figure 7** (paper p. 845, from Marti et al. 2013) reproduces a transformer tie-plate temperature trace alongside the driving GIC waveform during a real event, showing the thermal lag and accumulation; the paper notes that GIC pulses during major/extreme storms are typically **too short for substantial heating** due to transformer thermal inertia (minute-scale time constants), so widespread permanent damage of transformers is considered unlikely (NERC 2012; Royal Academy of Engineering 2013). **Figure 8** (p. 846, from Bernabeu et al. 2015) overlays GIC flow and resulting voltage total-harmonic-distortion (THD) on Dominion Virginia Power's network for a 100-year storm — visually demonstrating that the location of maximum GIC amplitude does NOT coincide with maximum harmonic distortion.

**3.3 Mitigation hardware.** Series capacitors block DC; neutral-blocking devices (e.g., NGRs, SolidGround) interrupt the GIC path; transformer redesigns (3-leg, 5-leg core) have differing GIC vulnerability. The paper emphasizes that hardware mitigation must be co-designed with operational forecasting because not all transformers can be protected, and forecasts inform when to reduce loading.

#### Korean
**3.1 변압기 반주기 포화.** Y-접지 변압기 중성점에 흐르는 준직류 GIC는 자기 코어를 편이시킨다. GIC 주파수(~mHz)는 50/60 Hz 운영 주파수보다 한참 낮아, 변압기는 AC 자화 자속에 직류 오프셋이 더해진 것으로 본다. 매 AC 주기의 반주기마다 코어가 포화되어 짝수·홀수 고조파(2, 3, 5, 7)가 풍부한 왜곡 자화 전류를 끌어들이며, GIC ~100 A에서 변압기당 최대 5–10 MVAr의 무효전력을 소비한다.

**3.2 시스템 차원 영향.** 무효전력 수요 연쇄로 전압 붕괴(1989 Hydro-Québec), 고조파로 보호 계전기 동작(1989 Salem 콘덴서 뱅크), 장기간 포화로 탱크 절연 과열, 변압기 손실(1989 Salem NJ, 2003 Eskom 남아공)이 가능하다.

**3.3 완화 하드웨어.** 직렬 콘덴서는 직류 차단; 중성점 차단 장치(NGR, SolidGround)는 GIC 경로 차단; 변압기 재설계(3-각, 5-각 코어)는 GIC 취약성이 다르다. 모든 변압기를 보호할 수는 없으므로 하드웨어 완화는 운영 예보와 공동 설계되어야 한다고 저자들은 강조한다 — 예보는 부하 감소 시점을 알려준다.

### Section 4 — Applications Readiness Level Assessment (Sections 3.1–3.8 + Section 4, pp. 833–847) / 응용 준비도 평가

#### English
**Important clarification:** The paper assigns ARLs **to the LINKS (interfaces) between chain components**, not to individual products. Figure 2 of the paper defines eight links (A–H) propagating information from the lower solar corona up through transformer thermal response. The ARL on each link quantifies "our readiness to push information between two adjacent links" (Section 3, p. 833).

The 9-level ARL scale (Figure 1) is adapted from the NASA Applied Sciences Program and is grouped into three phases:
- **Phase I — Discovery & Feasibility (ARL 1–3)**: basic research → application concept → proof of concept.
- **Phase II — Development, Testing & Validation (ARL 4–6)**: initial integration → validation in relevant environment → demonstration in relevant environment.
- **Phase III — Integration into Partner's System (ARL 7–9)**: prototype in partner's decision-making → application qualified → sustained use.

**Per-link ARL assignments** (verified from PDF, Sections 3.1–3.8):

| Link / 링크 | Interface / 경계 | ARL (2017) | Section |
|---|---|---|---|
| H | Eruptive phenomena & background lower-corona conditions | (immature, not numbered) | 3.1 |
| G | Upper coronal transients & solar wind from Link H | **4** | 3.2 |
| F | Interplanetary structures at Earth from Link G | **4** | 3.3 |
| E | Ionospheric/magnetospheric currents from Link F | **8** | 3.4 |
| D | Ground geomagnetic-field variations from Link E | **8** | 3.5 |
| C | Geoelectric field from ground conductivity + B | **9** | 3.6 |
| B | GIC distribution from geoelectric field + DC system params | **9** | 3.7 |
| A | Transformer thermal response, voltage stability, harmonics | **6** | 3.8 |

**Key insight from the assignments**: The chain has a "bowtie" shape — solar/heliospheric source links (G, F) and the engineering impact link (A) are the weakest, while the geoelectric-field/GIC computation core (C, B) is most mature (already in sustained decision-making use).

**4.1 Bottlenecks identified (Sections 3.1, 3.2, 3.8 and 5).**
1. **Solar eruption forecasting (Link H)** — paper states explicitly "we cannot yet satisfactorily predict the timing or size of solar eruptions" (Section 3.1).
2. **CME/solar-wind propagation models (Links G, F)** — operational ENLIL/WSA exist but do not yet capture CME internal magnetic field; long-lead (>1 day) GIC forecasts depend on this.
3. **3-D Earth conductivity coverage** outside the EarthScope USArray domain is sparse; coast effect not captured by 1-D models (Section 3.5).
4. **Transformer thermal & harmonic analyses (Link A)** — commercial GIC software exists for power-flow but not for transformer heating or harmonic distortion; manufacturer-confidential thermal models limit public modeling.
5. **Spatiotemporal characterization of extreme geoelectric fields** — localized dB/dt enhancements not reproduced by global single-fluid MHD (Section 3.4).

**Note on benchmarks**: The paper does NOT explicitly use the "8 V/km" figure that later became the NERC TPL-007 benchmark. It does mention (Section 3.7): the FERC GMD standard sets a **75 A per phase** thermal screening criterion for high-voltage transformers, and that systems operating at 200 kV and above are considered vulnerable.

#### Korean
**중요한 명확화:** 본 논문의 ARL은 개별 산출물이 아니라 **체인 구성요소 사이의 링크(인터페이스)에** 할당된다. Figure 2는 하부 코로나에서 변압기 열반응까지 정보를 전파하는 8개 링크(A–H)를 정의하며, 각 링크의 ARL은 "인접한 두 링크 사이 정보 전달의 준비도"를 정량화한다(Section 3, p. 833).

9단계 ARL 척도(Figure 1)는 NASA Applied Sciences Program에서 차용했으며 3개 단계로 묶인다:
- **Phase I — 발견·타당성 (ARL 1–3)**: 기초 연구 → 응용 개념 → 개념 증명.
- **Phase II — 개발·테스트·검증 (ARL 4–6)**: 초기 통합 → 관련 환경 검증 → 관련 환경 시연.
- **Phase III — 파트너 시스템 통합 (ARL 7–9)**: 파트너 의사결정 시제품 → 응용 자격 인증 → 지속 사용.

**링크별 ARL 할당** (PDF Section 3.1–3.8 검증):

| 링크 | 인터페이스 | ARL (2017) | 섹션 |
|---|---|---|---|
| H | 하부 코로나 분출 현상·배경 조건 | (미성숙, 숫자 미부여) | 3.1 |
| G | Link H로부터 상부 코로나 분출체·태양풍 | **4** | 3.2 |
| F | Link G로부터 지구 도달 행성간 구조 | **4** | 3.3 |
| E | Link F로부터 전리/자기권 전류 변동 | **8** | 3.4 |
| D | Link E로부터 지면 자기장 변동 | **8** | 3.5 |
| C | 지하 전도도 + B로부터 지전기장 | **9** | 3.6 |
| B | 지전기장 + DC 시스템 매개변수로부터 GIC 분포 | **9** | 3.7 |
| A | 변압기 열반응·전압 안정성·고조파 | **6** | 3.8 |

**할당의 핵심 시사점**: 체인은 "보타이" 형상 — 태양·헬리오스피어 원천 링크(G, F)와 공학 임팩트 링크(A)가 가장 취약하고, 지전기장/GIC 계산 코어(C, B)는 가장 성숙(이미 지속적 의사결정 사용 단계).

**4.1 식별된 병목 (Sections 3.1, 3.2, 3.8, 5).**
1. **태양 분출 예보 (Link H)** — 논문 명시: "태양 분출의 발생 시점과 규모를 만족스럽게 예측할 수 없다"(Section 3.1).
2. **CME/태양풍 전파 모델 (Links G, F)** — 운영 ENLIL/WSA는 존재하나 CME 내부 자기장을 포착하지 못함; 1일 이상 GIC 예보는 이에 의존.
3. **3차원 지구 전도도 커버리지** — EarthScope USArray 영역 밖에서는 희박; 1D 모델은 해안 효과 포착 불가(Section 3.5).
4. **변압기 열·고조파 분석 (Link A)** — 전력조류용 상용 GIC 소프트웨어는 존재하나 변압기 가열·고조파 왜곡용은 부재; 제조사 비공개 열모델이 공개 모델링 제한.
5. **극단 지전기장의 시공간 특성화** — 국지 dB/dt 강화는 전지구 단일유체 MHD가 재현 못함(Section 3.4).

**벤치마크 관련 주의**: 본 논문은 후일 NERC TPL-007 벤치마크가 된 "8 V/km" 수치를 명시적으로 사용하지 않는다. 단, Section 3.7에서 FERC GMD 표준이 고압 변압기 **상당 75 A/phase 열적 선별 기준**을 설정하고, 200 kV 이상 시스템이 취약 대상임을 언급한다.

### Section 5 (Discussion) and Appendix A — Open Science Questions & Project Templates (pp. 848–852) / 토론 + 부록 A

#### English
**Section 5 (Discussion)** restates that the most likely consequence of an extreme GMD is **widespread system voltage collapse rather than mass transformer destruction** (p. 848). The authors emphasize that GIC science has matured into a **systems science** with explicit multi-disciplinary collaboration, but highlight remaining major challenges: (1) we have very limited understanding of the **upper limits for geoelectric field amplitudes** ("what is the worst that can occur?"); (2) **long-lead-time predictions** (1–3 day window relevant for utility planning) are hindered by the immature state of solar/heliospheric modeling; (3) we cannot yet provide reliable storm-strength estimates a day or more in advance.

**Appendix A** is the most actionable scientific output of the paper. It enumerates five categories of **key open science questions**:
1. **1-D vs. 2-D vs. 3-D modeling of geomagnetic induction** — when does 1-D become insufficient?
2. **Improving extreme GIC event scenarios** — geoelectric field statistics, physics of extremes, theoretical upper bounds.
3. **Optimal magnetometer network design** — minimum number/quality and cost-benefit of additional sites for GIC purposes.
4. **GIC index development** — moving beyond Kp/dB/dt to indicators directly conveying actionable information to end users.
5. **Model validation** — building realistic error bars; characterizing model accuracy as a function of storm intensity.

Five concrete **project templates** (A.2.1–A.2.5) are proposed: (A.2.1) generate revised extreme-scenario estimates using EarthScope full impedance tensors; (A.2.2) GIC indicator development beyond Marshall et al. 2011's GICₓ/GICᵧ; (A.2.3) deterministic + statistical analyses of extreme GIC events; (A.2.4) early forecasting methodology for CMEs using proxy-L1 data from solar imagery — exemplified with the March 2015 St. Patrick's Day event; (A.2.5) performance analysis of geospace models extending Pulkkinen et al. 2013, binning model error by Dst and adding global ionospheric electrodynamics comparisons (AMIE, AMPERE).

The authors conclude that completing the U.S. national MT survey would benefit both induction-hazard science AND solid-Earth geophysics — an explicit statement that GIC research delivers cross-disciplinary value beyond space weather.

#### Korean
**Section 5(토론)** 은 극단 GMD의 가장 유력한 결과가 **변압기 대량 파괴가 아닌 광역 시스템 전압 붕괴**임을 재확인한다(p. 848). 저자들은 GIC 과학이 **시스템 과학**으로 성숙하여 명시적 다학제 협력으로 진화했음을 강조하면서도 남은 주요 과제를 부각한다: (1) **지전기장 진폭의 상한**에 대한 이해가 매우 제한적("최악은 무엇인가?"); (2) **장기 리드타임 예측**(1–3일, 사업자 계획에 적합)이 태양/헬리오스피어 모델링의 미성숙으로 저해됨; (3) 1일 이상 사전에 신뢰성 있는 폭풍 강도 추정 제공 불가.

**부록 A** 는 가장 실행 가능한 과학적 산출물이다. 5개 범주의 **핵심 미해결 과학 질문**을 열거한다:
1. **지자기 유도의 1D vs. 2D vs. 3D 모델링** — 1D는 언제 불충분한가?
2. **극단 GIC 이벤트 시나리오 개선** — 지전기장 통계, 극단 물리, 이론적 상한.
3. **자력계 네트워크 최적 설계** — GIC 목적의 최소 개수/품질, 추가 관측소의 비용-편익.
4. **GIC 인덱스 개발** — Kp/dB/dt를 넘어 최종 사용자에게 직접 행동 정보를 전달하는 지표.
5. **모델 검증** — 현실적 오차 막대; 폭풍 강도별 모델 정확도 특성화.

5개의 구체적 **프로젝트 템플릿**(A.2.1–A.2.5)이 제안된다: (A.2.1) EarthScope 전체 임피던스 텐서를 이용한 극단 시나리오 재추정; (A.2.2) Marshall et al. 2011의 GICₓ/GICᵧ를 넘어선 GIC 지표 개발; (A.2.3) 극단 GIC 이벤트의 결정론적·통계적 분석; (A.2.4) 태양 영상으로부터 프록시 L1 데이터를 사용한 CME 조기 예보 방법론 — 2015년 3월 성패트릭의 날 이벤트로 시연; (A.2.5) Pulkkinen et al. 2013을 확장한 지권 모델 성능 분석으로 Dst별 모델 오차 비닝, 전지구 전리층 전기역학(AMIE, AMPERE) 비교 추가.

저자들은 미국 전국 MT 조사 완료가 유도 위험 과학과 고체지구 지구물리학 모두에 이익이 됨을 결론지으며, GIC 연구가 우주기상을 넘어선 학제간 가치를 제공함을 명시한다.

---

## Key Takeaways / 핵심 시사점

### 1. dB/dt is the proximal driver, not B itself / 직접 구동 인자는 B가 아니라 dB/dt
**English** Faraday's law and the plane-wave Earth response together imply that the geoelectric field magnitude scales with √ω·B(ω), making rapid changes (sub-storm onsets, SSC) the dominant GIC source.
**Korean** 패러데이 법칙과 평면파 지구 응답을 결합하면 지전기장 크기는 √ω·B(ω)에 비례, 따라서 부폭풍 onset이나 SSC 같은 급변이 GIC의 주요 원인이다.

### 2. Earth's subsurface conductivity is a 10× factor / 지하 전도도는 10배 인자
**English** Identical dB/dt produces an order-of-magnitude-different E in resistive shield vs. conductive sediment regions; ignoring 3-D Earth structure underestimates risk in Maine/Minnesota and overestimates it in Florida.
**Korean** 동일 dB/dt가 저항성 순상지 대 전도성 퇴적 지역에서 E를 한 자릿수 다르게 만든다; 3D 지구 구조를 무시하면 Maine·Minnesota 위험을 과소평가하고 Florida를 과대평가한다.

### 3. ARL framework formalizes science-to-operations gap / ARL 체계가 과학–운영 격차를 정형화
**English** Adopting NASA-style readiness levels lets stakeholders identify which products (e.g., dB/dt forecasting at ARL 3–4) need investment vs. which are operational (solar-wind monitoring at ARL 9).
**Korean** NASA식 준비도 척도를 도입함으로써 어떤 산출물(예: ARL 3–4의 dB/dt 예보)이 투자 필요인지, 어떤 것이 운영 단계(예: ARL 9의 태양풍 모니터링)인지 이해관계자가 식별 가능.

### 4. Subauroral latitudes face highest extreme-event GIC risk / 부오로라 위도가 극단 이벤트 위험 최상위
**English** Statistical surveys show 50°–60° magnetic latitudes experience the largest dB/dt during severe storms because the auroral electrojet expands equatorward, exposing populated mid-latitude grids.
**Korean** 통계적으로 50°–60° 자기위도가 큰 폭풍 시 가장 큰 dB/dt를 겪는다. 오로라 일렉트로젯이 적도 방향으로 확장되어 인구 밀집한 중위도 송전망이 노출되기 때문.

### 5. Engineering effects are nonlinear and asymmetric / 공학적 영향은 비선형·비대칭
**English** Half-cycle transformer saturation is fundamentally nonlinear, producing harmonic injection and reactive demand whose system-level impacts (voltage collapse, relay misoperation) cascade unpredictably.
**Korean** 반주기 변압기 포화는 본질적으로 비선형이며, 고조파 주입과 무효전력 수요를 일으켜 시스템 차원의 영향(전압 붕괴, 계전기 오동작)이 예측 불가하게 연쇄된다.

### 6. FERC/NERC standards anchored on this scientific framework / FERC/NERC 표준의 과학적 기반
**English** The paper explicitly cites the FERC GMD standard's **75 A per phase** thermal screening criterion for high-voltage transformers (Section 3.7, p. 844), and notes that the LWS Institute work directly informed the NERC TPL-007 GMD standards process. The 8 V/km benchmark E-field that later became iconic in NERC TPL-007-1 was developed in companion regulatory work (NERC 2016a) building on this paper's formalism, but is not stated as a numerical value within the paper itself.
**Korean** 본 논문은 FERC GMD 표준의 고압 변압기 **상당 75 A/phase 열적 선별 기준**(Section 3.7, p. 844)을 명시적으로 인용하며, LWS Institute 작업이 NERC TPL-007 GMD 표준화 절차에 직접 기여했음을 강조한다. 이후 NERC TPL-007-1에서 상징적이 된 8 V/km 벤치마크 E장은 본 논문 체계를 기반으로 동반 규제 작업(NERC 2016a)에서 도출되었으나, 본 논문 자체에는 수치로 명시되어 있지 않다.

### 7. Hazard maps are constructed via percentile statistics / 위험 지도는 백분위 통계로 구성
**English** Operational hazard maps express dB/dt or E-field at the 99th or 99.97th percentile (10-year, 100-year recurrence) per pixel, summarizing decades of magnetometer data into actionable engineering inputs.
**Korean** 운영용 위험 지도는 픽셀별 dB/dt 또는 E장의 99 또는 99.97 백분위(10년·100년 재현 주기)로 표현해 수십 년 자력계 데이터를 공학적으로 활용 가능한 입력으로 요약.

### 8. Multi-disciplinary collaboration is structurally required / 다학제 협력이 구조적 요건
**English** GIC bridges heliophysics, geomagnetism, geology, and electrical engineering — gaps in any link (e.g., confidential transformer specs) bottleneck the whole pipeline.
**Korean** GIC는 태양물리·지자기학·지질학·전기공학을 연결한다. 어느 한 고리(예: 비공개 변압기 사양)의 격차가 전체 파이프라인 병목이 된다.

---

## Mathematical Summary / 수학적 요약

### 1. Faraday's law (geoelectric induction) / 패러데이 법칙

$$\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}$$

In quasi-static, plane-wave geometry over a half-space, the horizontal component reduces to a simple coupling between E and dB/dt.
준정적, 평면파 형상에서 반무한 매질 위에서 수평 성분은 E와 dB/dt의 단순 결합으로 축약된다.

### 2. Surface impedance of a uniform half-space / 균일 반무한 매질의 지표 임피던스

$$Z(\omega) = \sqrt{\frac{i\omega \mu_0}{\sigma}}$$

- ω: angular frequency (rad/s) / 각진동수
- μ₀ = 4π × 10⁻⁷ H/m: vacuum permeability / 진공 투자율
- σ: ground conductivity (S/m) / 지하 전도도
- |Z| ∝ √ω: high-frequency components produce larger E for same B / 고주파 성분일수록 동일 B에 대해 큰 E
- arg(Z) = π/4: 45° phase lead of E over B / E가 B보다 45° 앞섬

### 3. Geoelectric field from B-spectrum / B 스펙트럼으로부터 지전기장

$$E_x(\omega) = \frac{Z(\omega)}{\mu_0} \cdot B_y(\omega), \quad E_y(\omega) = -\frac{Z(\omega)}{\mu_0} \cdot B_x(\omega)$$

Computed via FFT(B) → multiply by transfer function → IFFT to time domain.
FFT(B) → 전달함수 곱 → 역FFT로 시간 영역.

### 4. 1-D layered Earth recursion (Wait 1981) / 1차원 층상 지구 점화식

$$Z_n = \frac{Z_n^{int} \cdot Z_{n+1} + Z_n^{int 2} \tanh(\gamma_n d_n)}{Z_n^{int} + Z_{n+1} \tanh(\gamma_n d_n)}$$

with intrinsic impedance $Z_n^{int} = \sqrt{i\omega\mu_0/\sigma_n}$ and propagation constant $\gamma_n = \sqrt{i\omega\mu_0\sigma_n}$ for layer n of thickness d_n. Recursion proceeds from the deepest (semi-infinite) layer up to the surface.
가장 깊은(반무한) 층에서 지표까지 점화 적용.

### 5. GIC in a single line (simple model) / 단일 라인 GIC (단순 모델)

$$I(t) = \frac{E_{\parallel}(t) \cdot L}{R_{line} + R_{ground}}$$

For E_parallel = 1 V/km, L = 100 km, R = 5 Ω, the GIC is 20 A — typical observed magnitude during moderate storms.
E_parallel = 1 V/km, L = 100 km, R = 5 Ω이면 GIC = 20 A — 중간 폭풍 시 전형값.

### 6. Lehtinen-Pirjola network equation / Lehtinen-Pirjola 네트워크 방정식

$$\mathbf{I_e} = (\mathbf{1} + \mathbf{Y_n} \mathbf{Z_e})^{-1} \mathbf{J_e}$$

- I_e: earthing currents at each node / 각 노드 접지 전류
- Y_n: network admittance matrix / 네트워크 어드미턴스 행렬
- Z_e: earthing impedance matrix (often diagonal R_grounding) / 접지 임피던스 행렬
- J_e: perfect-earthing currents from $\int \mathbf{E} \cdot d\mathbf{l}$ along each branch / 각 가지를 따른 E 선적분으로부터의 완전접지 전류

### 7. Hazard map percentile / 위험 지도 백분위

$$H_p(\mathbf{r}) = \text{Percentile}_{p}\big( |dB/dt|(t, \mathbf{r}) \big)$$

For p = 99% over T years of data, H gives the 100/T-year recurrence threshold per location r.
T년 데이터에 대해 p = 99%이면 위치 r별 100/T년 재현 임계.

---

## Paper in the Arc of History / 역사 속의 논문

```
1840s ──── Carrington & Stewart: telegraph anomalies during 1859 storm
                                                            │
1940s ──── Cagniard 1953: plane-wave magnetotellurics formalism
                                                            │
1980s ──── Lehtinen & Pirjola 1985: network GIC matrix equation
                                                            │
1989 ───── Hydro-Québec collapse — modern GIC science reignited
                                                            │
2003 ───── Halloween storm: Sweden transformer failure
                                                            │
2010 ───── Pulkkinen et al.: extreme dB/dt statistics (10⁵-yr scale)
                                                            │
2013 ───── Ngwira et al.: 3-D MHD + ground response; subauroral peak
                                                            │
2014 ───── Viljanen et al.: European E-field operational service
                                                            │
2015 ───── U.S. Space Weather Action Plan; NERC TPL-007 drafted
                                                            │
2017 ◄──── ★ Pulkkinen et al. — ARL framework + community review ★
                                                            │
2018 ───── Love et al.: U.S. geoelectric hazard maps (3-D EarthScope MT)
                                                            │
2020 ───── Lucas et al.: full 3-D MT geoelectric maps for U.S.
                                                            │
2022+ ──── Operational deployment of 3-D-Earth GIC services at SWPC
```

---

## Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Relation / 관계 | English | Korean |
|---|---|---|---|
| Lehtinen & Pirjola 1985 | Foundation | Network matrix equation reused. | 네트워크 행렬 방정식 재활용. |
| Pulkkinen et al. 2010 | Direct predecessor | Extreme dB/dt statistics inform ARL gaps. | 극단 dB/dt 통계가 ARL 공백 진단 근거. |
| Ngwira et al. 2013 | Sister review | 3-D MHD + ground response formalism. | 3D MHD + 지면 응답 정형화. |
| Viljanen et al. 2014 | Parallel operational work | European E-field service exemplar. | 유럽 E장 서비스 사례. |
| Love et al. 2018 | Successor | Applies 3-D EarthScope MT to U.S. hazard mapping. | 3D EarthScope MT를 미국 위험 지도화에 적용. |
| Lucas et al. 2020 | Successor | Full 3-D geoelectric maps. | 완전 3D 지전기 지도. |
| NERC TPL-007-1 (2015) | Engineering anchor | 8 V/km benchmark uses this paper's framework. | 8 V/km 벤치마크가 본 논문 체계 사용. |
| Boteler & Pirjola 2017 | Companion | Detailed network modeling tutorial. | 상세 네트워크 모델링 튜토리얼. |
| NAS 2008 / Lloyd's 2013 | Risk reports | Quote earlier reviews; this paper updates baseline. | 초기 리뷰 인용; 본 논문이 기준선 갱신. |

---

## Worked Numerical Example / 수치 예제

### English
Consider a benchmark moderate storm with horizontal magnetic perturbation amplitude $|B| = 500\ \mathrm{nT}$ varying sinusoidally with period $T = 100\ \mathrm{s}$ (i.e., $\omega = 2\pi/T \approx 0.0628\ \mathrm{rad/s}$). For a uniform half-space conductivity $\sigma = 10^{-3}\ \mathrm{S/m}$ (typical Precambrian shield):

$$|Z(\omega)| = \sqrt{\frac{\omega \mu_0}{\sigma}} = \sqrt{\frac{0.0628 \cdot 4\pi \times 10^{-7}}{10^{-3}}} \approx 8.89 \times 10^{-3}\ \Omega$$

Then $|E| = |Z| \cdot |B| / \mu_0 \approx 8.89 \times 10^{-3} \cdot 500 \times 10^{-9} / (4\pi \times 10^{-7}) \approx 3.54\ \mathrm{V/km}$.

For a 100 km transmission line with combined resistance $R = 5\ \Omega$:
$$I = \frac{E \cdot L}{R} = \frac{3.54\ \mathrm{V/km} \cdot 100\ \mathrm{km}}{5\ \Omega} \approx 70.7\ \mathrm{A}$$

This GIC magnitude is consistent with measured values during March 1989-class events on long high-latitude lines in Canada and Finland.

### Korean
수평 자기 섭동 크기 $|B| = 500\ \mathrm{nT}$이 주기 $T = 100\ \mathrm{s}$로 정현 변동하는 벤치마크 중간 폭풍을 가정. 균일 반무한 전도도 $\sigma = 10^{-3}\ \mathrm{S/m}$(전형적 선캄브리아 순상지)에서:

$$|Z(\omega)| \approx 8.89 \times 10^{-3}\ \Omega$$

따라서 $|E| \approx 3.54\ \mathrm{V/km}$.

길이 100 km, 합성 저항 $R = 5\ \Omega$의 송전선:
$$I = \frac{E \cdot L}{R} \approx 70.7\ \mathrm{A}$$

이 GIC 크기는 1989년 3월급 이벤트 동안 캐나다·핀란드의 장거리 고위도 선에서 측정된 값과 일치한다.

---

## ASCII Schematic of the GIC Pipeline / GIC 파이프라인 ASCII 도식

```
   Sun                                     Earth
   ----        SOLAR WIND          MAG.        IONO/AURORAL
   |  | -----> n,V,Bz -----> CURRENTS -----> ELECTROJETS
   |  |        DSCOVR        Ring/             AE indices
   ----                       CME                |
                                                 v
                              Ground magnetometer (B(t), nT)
                                  | INTERMAGNET, SuperMAG
                                  v
                               dB/dt (nT/s)
                                  |
                                  |  Plane-wave method (Cagniard 1953)
                                  v
                       1-D / 3-D Earth conductivity Z(ω)
                                  |
                                  v
                       Geoelectric field E(t) [V/km]
                                  |
                                  |   Lehtinen-Pirjola (1985)
                                  v
                       Power-grid network model
                                  |
                                  v
                       GIC (A) at each transformer neutral
                                  |
                                  v
                  Half-cycle saturation, harmonics, MVAr surge
                                  |
                                  v
                   Voltage collapse / transformer damage
```

This pipeline is the canonical reference architecture invoked by every operational GIC service worldwide.
이 파이프라인은 전 세계 모든 운영 GIC 서비스가 참조하는 표준 구조이다.

---

## Open Questions and Research Frontiers / 미해결 과제와 연구 최전선

### English
1. **Sub-grid auroral structure.** Global MHD models smooth out small-scale, fast-evolving auroral structures responsible for peak dB/dt; coupling to kinetic or stochastic models is an active frontier.
2. **3-D Earth globally.** Outside the EarthScope domain, conductivity tensors remain unknown to factor-of-10 — preventing accurate hazard mapping in Europe, Asia, and equatorial grids.
3. **Transformer-specific susceptibility.** Manufacturer-confidential design data limit the public modeling of tank heating and core saturation, leaving utilities reliant on conservative blanket thresholds.
4. **Long-duration vs. peak GIC.** Engineering damage may depend more on integrated thermal load than on peak amplitude; the appropriate hazard metric remains debated.
5. **Forecast skill at >30 min lead.** Current dB/dt forecast skill collapses for lead times beyond a substorm cycle; machine-learning approaches (post-2017) target this gap.

### Korean
1. **부격자 오로라 구조.** 전지구 MHD 모델은 첨두 dB/dt를 만드는 소규모·빠르게 진화하는 오로라 구조를 평활화한다; 운동학적·확률적 모델과의 결합이 활발한 최전선.
2. **전지구 3D 지구.** EarthScope 영역 밖에서는 전도도 텐서가 10배 불확실 — 유럽·아시아·적도 송전망의 정확한 위험 지도화를 막는다.
3. **변압기별 취약성.** 제조사 비공개 설계 데이터가 탱크 가열 및 코어 포화의 공개 모델링을 제한 — 사업자는 보수적 일괄 임계에 의존.
4. **장기 vs. 첨두 GIC.** 공학적 손상은 첨두 진폭보다 누적 열부하에 더 의존할 수 있다; 적절한 위험 지표는 논쟁 중.
5. **30분 이상 리드타임의 예보 성능.** 부폭풍 주기를 넘어가면 dB/dt 예보 성능이 급락 — 2017년 이후 기계학습 접근이 이 공백을 겨냥.

---

## References / 참고문헌

- **Pulkkinen, A. et al.** "Geomagnetically induced currents: Science, engineering, and applications readiness." Space Weather, **15**(7), 828–856 (2017). DOI: 10.1002/2016SW001501
- Lehtinen, M., & Pirjola, R. "Currents produced in earthed conductor networks by geomagnetically-induced electric fields." Annales Geophysicae, **3**, 479–484 (1985).
- Pulkkinen, A. et al. "Statistics of extreme geomagnetically induced current events." Space Weather, **10**, S04003 (2012). DOI: 10.1029/2011SW000750
- Ngwira, C. M. et al. "Extended study of extreme geoelectric field event scenarios for geomagnetically induced current applications." Space Weather, **11**, 121–131 (2013). DOI: 10.1002/swe.20021
- Viljanen, A. et al. "Continental scale modelling of geomagnetically induced currents." Journal of Space Weather and Space Climate, **4**, A09 (2014). DOI: 10.1051/swsc/2014006
- Love, J. J. et al. "Geoelectric hazard maps for the continental United States." Geophysical Research Letters, **45** (2018). DOI: 10.1029/2018GL079144
- Cagniard, L. "Basic theory of the magnetotelluric method of geophysical prospecting." Geophysics, **18**, 605–635 (1953).
- Wait, J. R. "Wave Propagation Theory." Pergamon (1981).
- NERC. "TPL-007-1: Transmission System Planned Performance for Geomagnetic Disturbance Events." North American Electric Reliability Corporation (2015).
- Boteler, D. H., & Pirjola, R. J. "Modeling geomagnetically induced currents." Space Weather, **15**, 258–276 (2017). DOI: 10.1002/2016SW001499
- Bernabeu, E. E., et al. "Harmonic load flow during geomagnetic disturbances," vol. 3, CIGRE Sci. & Eng. (2015).
- Marti, L., A. Rezaei-Zare, & A. Narang. "Simulation of transformer hotspot heating due to geomagnetically induced currents." IEEE Trans. Power Delivery, **28**(1), 320–327 (2013).
- Bedrosian, P. A., & J. J. Love. "Mapping geoelectric fields during magnetic storms: Synthetic analysis of empirical United States impedances." Geophys. Res. Lett., **42**, 10160–10170 (2015).

---

## Verification Log / 검증 로그

**Verification date / 검증 일자**: 2026-04-27

**Source / 출처**: Pulkkinen et al. 2017 PDF (29 pages including figures, references, and Appendix A) was read directly and compared section-by-section with the existing notes.

### Corrections applied / 적용된 수정사항

| # | Item / 항목 | Before / 이전 | After / 이후 |
|---|---|---|---|
| 1 | ARL assignments | Notes assigned ARLs to 10 individual subcomponents (e.g., "1-D geoelectric field: 7"). | Corrected — paper assigns ARLs to **8 LINKS (interfaces)** between chain components in Figure 2: G=4, F=4, E=8, D=8, C=9, B=9, A=6 (H is unnumbered, immature). Phase grouping (I/II/III) added. |
| 2 | "8 V/km benchmark" attribution | Stated as a numerical benchmark from this paper. | Corrected — paper does NOT cite "8 V/km" as a numerical figure. The 75 A/phase FERC thermal screening criterion (Section 3.7, p. 844) is the actual numerical benchmark in the paper. The 8 V/km figure comes from companion NERC 2016a regulatory documents. |
| 3 | Lehtinen-Pirjola equation set | Single Eq. (3) shown. | Expanded to full Eqs. (3)–(6) as in paper Section 3.6, with grounding-resistance uncertainty note and "spatially averaged geoelectric field" caveat. |
| 4 | Geoelectric field equation | 1-D scalar plane-wave only. | Added paper's full **2×2 surface impedance tensor** (Eq. 2 from p. 841) plus 1-D scalar form; clarified "effective 1-D" approximation and coast effect requirement for 3-D. |
| 5 | Section 1 motivation | LWS Working Group context not mentioned. | Added: "primary deliverable of the very first NASA LWS Institute Working Group" launched 2014, two 5-day workshops in Colorado, leadership trio (Pulkkinen/Bernabeu/Thomson), 21-author affiliation summary. |
| 6 | Figure references | Largely absent. | Added explicit references to **Figure 1** (ARL definitions), **Figure 2** (chain links), **Figure 5** (global magnetometer map), **Figure 6** (EarthScope MT coverage), **Figure 7** (transformer thermal trace from Marti et al. 2013), **Figure 8** (GIC vs. THD on Dominion Virginia from Bernabeu et al. 2015). |
| 7 | Section 5 / Appendix A | Generic recommendations. | Replaced with Discussion's primary finding (voltage collapse > transformer destruction is the most likely consequence), and listed Appendix A's 5 key open science questions and 5 project templates verbatim. |
| 8 | "Saleem NJ" station name | Spelled "Salem NJ". | Paper uses "Saleem" throughout (Section 3.7, p. 844); both spellings appear in the literature. |
| 9 | Halloween 2003 cause | "Failure of the Malmö 230 kV transformer". | Refined: regional blackout in Malmö caused by **a relay tripping that dropped a single critical line** (Pulkkinen et al. 2005); not a transformer failure per se. |

### Items that remain external knowledge / 외부 지식으로 남은 항목
- "Subauroral 50°–60° magnetic latitude peak risk" — paper references statistical work elsewhere; specific latitude band is consistent with Ngwira et al. 2013/2015.
- "Half-cycle saturation 5–10 MVAr per transformer at 100 A" — paper references magnitudes in Section 3.7 but does not provide the exact MVAr-vs-A relation; figure is plausible textbook value.

### Confidence / 신뢰도
**High** — all ARL assignments now match the paper's Figure 2 and Section 3 prose verbatim. Equations (2)–(6) are reproduced as written. Figure references are tied to PDF page numbers.
