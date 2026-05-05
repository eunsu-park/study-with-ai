---
title: "The THEMIS Mission — Pre-Reading Briefing"
paper: "Angelopoulos, V. (2008), The THEMIS Mission, Space Sci. Rev., 141, 5–34"
date: 2026-04-27
topic: Space_Weather
tags: [THEMIS, substorm, magnetosphere, multi-spacecraft, mission_overview]
---

# Pre-Reading Briefing / 사전 학습 브리핑

## 1. Why This Paper Matters / 이 논문이 중요한 이유

**English.** The THEMIS (Time History of Events and Macroscale Interactions during Substorms) mission, launched in February 2007, was the first dedicated multi-probe mission specifically designed to settle the long-standing debate over the trigger mechanism of magnetospheric substorms. By placing five identically instrumented spacecraft along the magnetotail at conjunctions of $\sim 10$, $\sim 12$, $\sim 20$, and $\sim 30\,R_E$, THEMIS provides the first true causal-chain measurement: which signature appears first — tail reconnection at $\sim 20$–$30\,R_E$, or current disruption at $\sim 8$–$12\,R_E$? The mission's two-year baseline science phase was orchestrated to obtain ~30 perfectly-aligned tail conjunctions per year, complemented by a dense ground-based network of all-sky imagers (ASIs) and magnetometers (GMAGs) across North America. Angelopoulos (2008) is the master overview paper of the special issue, defining the science objectives, orbit design, instrument suite, and mission phases.

**한국어.** 2007년 2월에 발사된 THEMIS(Time History of Events and Macroscale Interactions during Substorms) 미션은 자기권 substorm의 촉발 메커니즘 논쟁을 해결하기 위해 특별히 설계된 최초의 다중탐사선 미션입니다. 동일하게 계측된 5개의 탐사선을 자기권 꼬리 영역의 $\sim 10$, $\sim 12$, $\sim 20$, $\sim 30\,R_E$ 합류점에 배치함으로써 THEMIS는 최초로 진정한 인과 사슬 측정을 제공합니다: 어느 신호가 먼저 나타나는가 — $\sim 20$–$30\,R_E$의 꼬리 재연결인가, 아니면 $\sim 8$–$12\,R_E$의 전류차단인가? 이 미션의 2년 기준 과학 단계는 연간 약 30회의 완벽하게 정렬된 꼬리 합류 이벤트를 확보하도록 조직되었으며, 북미 전역의 전천 영상장치(ASIs)와 지상 자력계(GMAGs) 밀집 네트워크로 보완됩니다. Angelopoulos (2008)는 특집호의 마스터 개요 논문으로 과학 목표, 궤도 설계, 계측기 모음, 미션 단계를 정의합니다.

## 2. Prerequisites / 선수 지식

### Physics / 물리학
- **MHD and reconnection**: Sweet–Parker, Petschek models; $X$-line structure / MHD와 재연결: Sweet–Parker, Petschek 모델; $X$-line 구조
- **Substorm phases**: growth, expansion, recovery / Substorm 위상: 성장기, 팽창기, 회복기
- **Magnetotail structure**: plasma sheet, neutral sheet, lobes / 자기권 꼬리 구조: 플라즈마 시트, 중성 시트, 로브
- **Field-aligned currents (Region 1/2, substorm current wedge)** / 자기력선 정렬 전류

### Observation / 관측
- **All-sky imager (ASI) photometry** of auroral arcs / 오로라 호의 전천 영상장치 측광
- **Geomagnetic indices**: AE, AL, AU, Pi2 pulsations / 지자기 지수: AE, AL, AU, Pi2 파동
- **Multi-spacecraft timing**: $\delta t = \mathbf{n}\cdot\Delta\mathbf{r}/v$ / 다중 우주선 타이밍 분석

### Prior Reading / 선행 논문
- Paper #8 (substorm phenomenology) / Paper #8 (substorm 현상학)
- Paper #10 (multi-spacecraft analysis) / Paper #10 (다중탐사선 분석)

## 3. The Substorm Trigger Debate / Substorm 촉발 논쟁

**English.** Two competing models had dominated for decades:

1. **Near-Earth Neutral Line (NENL) model**: Reconnection at $20$–$30\,R_E$ launches earthward flow bursts that brake near $\sim 10\,R_E$, depositing energy and triggering current disruption and the auroral breakup. Causality: tail $\rightarrow$ ionosphere.
2. **Current Disruption (CD) model**: Cross-tail current near $\sim 8$–$12\,R_E$ becomes unstable (ballooning, kinetic cross-field), disrupting locally; a rarefaction wave then propagates tailward, *triggering* reconnection downstream. Causality: ionosphere/inner tail $\rightarrow$ outer tail.

Distinguishing these requires simultaneous measurements at both regions with timing precision $\lesssim 30\,\mathrm{s}$ — the duration of the substorm onset preamble. THEMIS's five-probe design solves this exactly.

**한국어.** 두 가지 경쟁 모델이 수십 년 동안 지배해 왔습니다:

1. **근지구 중성선(NENL) 모델**: $20$–$30\,R_E$에서의 재연결이 지구 방향 흐름 버스트를 발생시키고, 이것이 $\sim 10\,R_E$ 근방에서 제동되며 에너지를 침적하고 전류 차단과 오로라 폭발을 촉발합니다. 인과: 꼬리 $\rightarrow$ 전리권.
2. **전류 차단(CD) 모델**: $\sim 8$–$12\,R_E$ 근방의 횡꼬리 전류가 불안정해지고 (벌루닝, 운동학적 교차장 불안정성) 국소적으로 차단되며, 희박파가 꼬리 방향으로 전파되어 하류에서 재연결을 *촉발*합니다. 인과: 전리권/내부 꼬리 $\rightarrow$ 외부 꼬리.

이 둘을 구별하려면 두 영역에서 동시에 $\lesssim 30\,\mathrm{s}$의 타이밍 정밀도로 측정해야 합니다 — substorm 시작 전조의 지속 시간입니다. THEMIS의 5탐사선 설계가 이를 정확히 해결합니다.

## 4. The Five-Probe Conjunction Geometry / 5탐사선 합류 기하

**English.** During tail science (Phase 2, ~Dec 2007 – Mar 2009), the five probes occupy nested orbits with apogees aligned along the magnetotail near 0 MLT during winter:

| Probe | Apogee ($R_E$) | Period | Role |
|-------|----------------|--------|------|
| TH-A (P5) | 10 (initially $<24$h, ~19.2 h) | sub-synchronous | Inner edge / 내부 가장자리 |
| TH-D (P3) | 12 | 24 h | Inner CD region |
| TH-E (P4) | 12 | 24 h | Inner CD region |
| TH-B (P1) | ~30 | 96 h (4 d) | Outer NENL region |
| TH-C (P2) | ~19 (paper p. 7) | 48 h (2 d) | Mid-tail bridge |

Conjunction recurs every 4 days — the **96 h** synodic period of the outermost probe. Aligned with ground ASIs over Canada/Alaska.

**한국어.** 꼬리 과학 단계(Phase 2, 약 2007년 12월–2009년 3월) 동안, 5개 탐사선은 겨울철 0 MLT 근방의 자기권 꼬리를 따라 정렬된 원지점을 갖는 중첩된 궤도를 점유합니다. 합류는 4일마다 반복됩니다 — 가장 외곽 탐사선의 **96시간** 회합주기입니다. 캐나다/알래스카 상공의 지상 ASI와 정렬됩니다.

## 5. Key Vocabulary / 핵심 용어

| Term | Definition (EN) | 정의 (KO) |
|------|----------------|----------|
| Substorm | Energy storage-release cycle in the magnetotail | 자기권 꼬리에서의 에너지 저장-방출 순환 |
| Onset | Earliest auroral brightening of expansion phase | 팽창기의 최초 오로라 밝아짐 |
| NENL | Near-Earth Neutral Line ($\sim 20$–$30\,R_E$) | 근지구 중성선 |
| CD | Current Disruption ($\sim 8$–$12\,R_E$) | 전류 차단 |
| BBF | Bursty Bulk Flow ($v_x > 400$ km/s) | 폭발적 대량 흐름 |
| Dipolarization | Sudden $B_z$ increase signaling collapse | 쌍극자화: 갑작스러운 $B_z$ 증가 |
| AE index | Auroral Electrojet (AU − AL) | 오로라 전기제트 지수 |
| Pi2 | Pulsations 40–150 s, onset signature | 시작 신호 파동 |
| ESA | Electrostatic Analyzer (ions 5 eV–25 keV; electrons 5 eV–30 keV) | 정전 분석기 |
| SST | Solid State Telescope (ions 25 keV–6 MeV; electrons 25 keV–1 MeV) | 고체 망원경 |
| FGM | Fluxgate Magnetometer (DC–64 Hz; 3 pT resolution) | 플럭스게이트 자력계 |
| SCM | Search Coil Magnetometer (1 Hz–4 kHz) | 서치 코일 자력계 |
| EFI | Electric Field Instrument (DC–8 kHz; 50 m/40 m SpB, 7 m AxB tip-to-tip) | 전기장 계측기 |

## 6. Pre-Reading Q&A / 사전 Q&A

**Q1. Why five probes and not three (like Cluster)? / 왜 클러스터처럼 3대가 아닌 5대인가?**
EN: Substorm science requires simultaneous radial sampling at *four* distinct distances ($10, 12, 20, 30\,R_E$) plus an inner-edge probe — three is insufficient to bracket both the CD and NENL regions while spanning the gap between them.
KO: Substorm 과학은 *4개*의 서로 다른 거리($10, 12, 20, 30\,R_E$)에서의 동시 방사 표본화와 내부 가장자리 탐사선이 필요합니다 — 3대로는 CD와 NENL 두 영역을 모두 포괄하면서 그 사이의 간격을 잇기에 부족합니다.

**Q2. What is the role of ground-based ASIs? / 지상 ASI의 역할은?**
EN: ASIs (auroral imagers) provide the *ionospheric anchor* — the precise UT of auroral breakup defines $t=0$. Tail signatures are then timed relative to optical onset.
KO: ASI는 *전리권 앵커*를 제공합니다 — 오로라 폭발의 정확한 UT가 $t=0$을 정의합니다. 꼬리 신호는 광학적 시작에 대해 상대적으로 타이밍됩니다.

**Q3. Why a 4-day conjunction cadence? / 왜 4일 합류 주기인가?**
EN: TH-B's 96-h orbit sets the recurrence; its apogee at $30\,R_E$ aligns with the inner-probe apogees once per orbit, giving ~30 high-quality conjunctions over the winter tail season.
KO: TH-B의 96시간 궤도가 반복을 결정합니다; $30\,R_E$ 원지점이 궤도당 한 번 내부 탐사선 원지점과 정렬되어 겨울 꼬리 시즌 동안 약 30회의 고품질 합류를 제공합니다.

**Q4. How does THEMIS handle the radiation hazard? / THEMIS는 방사선 위험을 어떻게 처리하는가?**
EN: First-orbit phase used "coast" orbits avoiding radiation belts; subsequent orbit raises (placement maneuvers in summer 2007) lifted apogees to tail-science values.
KO: 초기 궤도 단계는 방사선대를 피하는 "코스트" 궤도를 사용했습니다; 이후 궤도 상승(2007년 여름의 배치 기동)이 원지점을 꼬리 과학 값으로 들어 올렸습니다.

**Q5. What is the role of Pi2 pulsations? / Pi2 파동의 역할은?**
EN: Pi2 (40–150 s) is the classical ground signature of substorm onset, marking the formation of the substorm current wedge. THEMIS GMAGs detect Pi2 simultaneously with ASI breakup.
KO: Pi2(40–150 s)는 substorm 시작의 고전적 지상 신호로, substorm 전류 쐐기의 형성을 표시합니다. THEMIS GMAG는 ASI 폭발과 동시에 Pi2를 감지합니다.

## 7. Reading Strategy / 읽기 전략

**English.** Read in this order:
1. Sections 1–2 (Introduction, science objectives) — establish the substorm question.
2. Section 3 (Mission design) — orbits, conjunction cadence, tail/dayside seasons.
3. Section 4 (Instruments) — ESA, SST, FGM, SCM, EFI specs.
4. Section 5 (Ground-based observatories) — ASIs, GMAGs, EPO sites.
5. Sections 6–7 (Operations, expected science) — data flow, science closure.

**한국어.** 다음 순서로 읽으세요:
1. 1–2절 (서론, 과학 목표) — substorm 질문 설정.
2. 3절 (미션 설계) — 궤도, 합류 주기, 꼬리/태양면 시즌.
3. 4절 (계측기) — ESA, SST, FGM, SCM, EFI 사양.
4. 5절 (지상 관측소) — ASI, GMAG, EPO 사이트.
5. 6–7절 (운영, 예상 과학) — 데이터 흐름, 과학 마무리.

## References / 참고문헌
- Angelopoulos, V. (2008). The THEMIS Mission. *Space Sci. Rev.*, 141, 5–34. [DOI:10.1007/s11214-008-9336-1]
- Sibeck, D. G., & Angelopoulos, V. (2008). THEMIS science objectives and mission phases. *Space Sci. Rev.*, 141, 35–59.
- Baker, D. N., et al. (1996). Neutral line model of substorms. *J. Geophys. Res.*, 101, 12975.
- Lui, A. T. Y. (1996). Current disruption in the Earth's magnetosphere. *J. Geophys. Res.*, 101, 13067.
