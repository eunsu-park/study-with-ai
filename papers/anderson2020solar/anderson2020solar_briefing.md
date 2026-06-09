---
title: "Pre-Reading Briefing: The Solar Orbiter SPICE Instrument"
paper_id: "18_anderson_2020"
topic: Solar_Observation
date: 2026-04-17
type: briefing
---

# The Solar Orbiter SPICE Instrument: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: SPICE Consortium (M. Anderson et al.), "The Solar Orbiter SPICE instrument — An extreme UV imaging spectrometer", *Astronomy & Astrophysics*, Vol. 642, A14 (2020). DOI: 10.1051/0004-6361/201935574
**Author(s)**: SPICE Consortium (M. Anderson 외 수백 명의 공동저자; corresponding author: D. Müller, ESA)
**Year**: 2020

---

## 1. 핵심 기여 / Core Contribution

**한국어**
SPICE(Spectral Imaging of the Coronal Environment)는 ESA/NASA의 Solar Orbiter 미션에 탑재된 **극자외선(EUV) 영상 분광기(imaging spectrometer)** 로, 파장 범위 **70.4–79.0 nm(SW) 및 97.3–104.9 nm(LW)** 에서 태양 대기를 관측한다. 이 논문은 SPICE의 **과학 목표 · 광학·기계·열 설계 · 전자부 · 발사 전 성능·보정 · 운영 개념과 데이터 처리** 를 종합적으로 기술한 공식 기기 논문(instrument paper)이다. 핵심 기여는 (1) 태양에 **0.28 AU까지 근접**하는 전례 없는 환경에서 작동하는 **단일 거울 + Toroidal Variable Line Space(TVLS) 그레이팅** 광학 설계를 제시하고, (2) 채층(~10 000 K)부터 플레어 코로나(~10 MK)까지 폭넓은 온도대를 커버하는 **선택된 EUV 스펙트럼 라인들**(H I Lyβ, C II/III, O IV/V/VI, Ne VI–VIII, Mg VIII–XI, Fe X/XVIII/XX, Si XII 등)을 통해 **transition region과 coronal plasma의 소스 영역** 을 in-situ 입자 관측과 원격 관측을 연결하는 "tracer"로서 진단할 수 있음을 보인다는 점이다. 나아가 Solar Orbiter의 **out-of-ecliptic(황도면 외) 시야** 덕분에 **극지 태양풍의 분광 진단을 최초로 수행**할 수 있다는 점에서 과학적 독창성이 크다.

**English**
SPICE (Spectral Imaging of the Coronal Environment) is an **extreme ultraviolet imaging spectrometer** aboard ESA/NASA's Solar Orbiter mission, observing in two EUV bands **70.4–79.0 nm (SW) and 97.3–104.9 nm (LW)**. This paper is the official instrument paper describing SPICE's **science objectives, optical/mechanical/thermal/electronics design, pre-launch performance and calibration, operations concept, and data processing**. The key contributions are (1) presenting an optical design — a **single off-axis parabola telescope** feeding a **Toroidal Variable Line Space (TVLS) grating spectrometer** — that operates at **perihelion distances down to 0.28 AU** under a solar flux ~13× the solar constant, and (2) demonstrating that a carefully chosen set of EUV emission lines (H I Lyβ; C II/III; O IV/V/VI; Ne VI–VIII; Mg VIII–XI; Fe X/XVIII/XX; Si XII) enables SPICE to diagnose plasma from the chromosphere (~10 000 K) through the transition region and corona up to flaring plasma at ~10 MK, thereby serving as a **tracer that connects remote-sensing observations of the solar surface/corona with in-situ measurements of the solar wind** by Solar Orbiter's other instruments (SWA, MAG, RPW, EPD). SPICE's unique **out-of-ecliptic vantage point** will also enable the **first-ever spectral observations of the solar polar regions**.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

**한국어**
SPICE는 태양 EUV 분광기의 50여 년 계보에서 가장 최신 세대에 속한다. 선행 임무들은 각자의 한계를 가지고 있었다:
- **SOHO/SUMER** (1995–, Wilhelm et al.): 넓은 파장 범위를 커버했으나 라인별로 **파장을 stepping** 해야 해서 동적 현상 관측이 느렸다.
- **SOHO/CDS**: EUV 분광이 가능했으나 공간분해능(~4″)과 민감도가 제한적이었다.
- **Hinode/EIS** (2006–, Culhane et al.): 공간분해능 약 2″, 주로 **1 MK 이상의 뜨거운 코로나** 에 민감했으나 transition region 라인은 약했다.
- **IRIS** (2014–, De Pontieu et al.): 공간분해능 0.4″로 최고지만 **0.3 MK–8 MK** 로 온도 범위가 제한적이었다.

Solar Orbiter는 Helios(1974)와 유사한 0.28 AU 근접에 더해 **황도면에서 30° 이상 궤도 경사** 를 실현한 최초의 미션이며, SPICE는 **넓은 온도대 + 라인별 stepping 없이 동시 관측 + 극지 분광** 이라는 독자 영역을 개척한다. 2020년 2월 발사되어 현재(2026년 시점) 정상 운용 중이다.

**English**
SPICE sits at the cutting edge of a ~50-year lineage of solar EUV spectrographs. Its predecessors each had distinct limitations:
- **SOHO/SUMER** (1995–, Wilhelm et al.): broad wavelength coverage but had to **step through wavelengths**, slowing dynamic observations.
- **SOHO/CDS**: EUV spectroscopy but limited in spatial resolution (~4″) and sensitivity.
- **Hinode/EIS** (2006–, Culhane et al.): spatial resolution ~2″; strong on **hot corona (>1 MK)** but weak in transition-region lines.
- **IRIS** (2014–, De Pontieu et al.): best spatial resolution (~0.4″) but limited to **0.3 MK–8 MK**.

Solar Orbiter is the first mission to combine **Helios-class perihelion (0.28 AU)** with a **>30° orbital inclination** out of the ecliptic. SPICE opens a unique niche: **broad temperature coverage, simultaneous line coverage without stepping, and first-ever spectroscopy of the solar poles**. Launched Feb 2020; as of 2026 it is in its nominal mission phase.

### 타임라인 / Timeline

```
1974 ───── Helios 1/2 launch (perihelion 0.3 AU; no EUV spectrometer)
1995 ───── SOHO/SUMER, CDS (first sustained EUV spectroscopy from L1)
2006 ───── Hinode/EIS (hot corona EUV spectroscopy)
2013 ───── Fludra et al. — first published SPICE optical design
2014 ───── IRIS (high-resolution UV spectroscopy of chromosphere/TR)
2019 ───── SPICE flight unit accepted (March 2019 — this paper received)
2020.02 ── Solar Orbiter launch (SPICE onboard)
2020.08 ── This paper published in A&A Solar Orbiter Special Issue
2022 ───── First Solar Orbiter perihelion (~0.29 AU)
~2025 ──── Out-of-ecliptic phase begins (inclination >17°)
2026 ───── (Today) Extended mission; SPICE observing polar regions
```

---

## 3. 필요한 배경 지식 / Prerequisites

**한국어**
이 논문은 engineering-heavy한 instrument paper이므로, 다음 배경지식이 있으면 훨씬 효율적으로 읽을 수 있다:

1. **EUV 분광의 기초**
   - 왜 태양 대기의 채층·transition region·코로나는 EUV에서 빛나는가(광학적으로 얇고, 형성온도가 각 이온의 전리평형에 의해 결정됨).
   - 발광선의 형성온도($\log T$)와 이온의 관계(e.g., Fe XII → 1.5 MK).

2. **분광기 광학**
   - 회절격자 방정식 $m\lambda = d(\sin\theta_i + \sin\theta_m)$.
   - Slit · grating · detector의 기하학, plate scale, dispersion.
   - TVLS(Toroidal Variable Line Space) 그레이팅: 단일 광학 소자로 focusing + dispersion + astigmatism 보정을 동시에 수행.

3. **이전 논문들(필수 배경)**
   - **논문 #15 (Culhane et al. 2007, Hinode/EIS)**: EUV 영상 분광기의 설계 원리와 관측 전략.
   - **논문 #16 (De Pontieu et al. 2014, IRIS)**: UV 영상 분광으로 채층/TR 동역학 연구의 선례.

4. **plasma 진단**
   - **FIP effect(First Ionisation Potential bias)**: FIP < 10 eV인 원소(Si, Mg, Fe)가 코로나/태양풍에서 ~4× 강화되는 현상 — 소스 영역 식별의 열쇠.
   - Doppler shift에서 유출 속도 얻기($\Delta\lambda/\lambda = v/c$).
   - Non-thermal broadening과 turbulence/wave 진단.

5. **Solar Orbiter 미션 설계**
   - 0.28 AU 근일점 → **태양 플럭스 ~13배** → 열부하 ~17 kW/m², 입자 피해.
   - Heat-shield aperture와 dichroic mirror가 왜 필요한지.
   - Remote-sensing + in-situ 결합 전략(SPICE가 SWA, MAG, EPD와 어떻게 연동되는가).

**English**
Because this is an engineering-heavy instrument paper, the following prepares you to read efficiently:

1. **EUV spectroscopy fundamentals** — why the chromosphere/transition region/corona emit EUV (optically thin; formation temperature set by ionisation balance). Ion → $\log T$ relation.
2. **Spectrograph optics** — grating equation $m\lambda = d(\sin\theta_i + \sin\theta_m)$; slit/grating/detector geometry, plate scale, dispersion; TVLS gratings that provide focusing + dispersion + astigmatism correction in one element.
3. **Prior papers (essential)** — Paper #15 (Culhane et al. 2007, Hinode/EIS) and Paper #16 (De Pontieu et al. 2014, IRIS).
4. **Plasma diagnostics** — **FIP effect** (low-FIP elements enhanced ~4× in corona/wind — key source-region tracer); Doppler shifts for outflow velocity ($\Delta\lambda/\lambda = v/c$); non-thermal broadening → turbulence/waves.
5. **Solar Orbiter mission design** — 0.28 AU perihelion → **~13× solar constant** → 17 kW/m² heat load, particle damage; why a heat-shield aperture and dichroic mirror are needed; how SPICE pairs with SWA/MAG/EPD.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **SPICE** | Spectral Imaging of the Coronal Environment — Solar Orbiter의 EUV 영상 분광기 / Solar Orbiter's EUV imaging spectrometer |
| **EUV (Extreme UV)** | 극자외선, 대략 10–120 nm. SPICE는 **70.4–79.0 nm (SW)** 와 **97.3–104.9 nm (LW)** 를 관측 / Extreme ultraviolet, ~10–120 nm; SPICE covers two bands |
| **TVLS grating** | Toroidal Variable Line Space grating — 격자선 간격이 표면에 따라 변하는 toroidal(2곡률) 격자. 추가 광학 없이 수차 보정 + 분산 / grating with varying line spacing on a toroidal surface; corrects aberrations without extra optics |
| **Transition region (TR)** | 채층과 코로나 사이 얇은 층, ~10⁴ → 10⁶ K로 급격히 상승. O IV–VI, Ne VII이 형성되는 영역 / thin layer between chromosphere and corona; temperature jumps from ~10⁴ to 10⁶ K; O IV–VI, Ne VII form here |
| **FIP effect / FIP bias** | 첫 이온화 전위가 낮은(<10 eV) 원소(Si, Mg, Fe)가 코로나·태양풍에서 강화되는 조성 편향 — 태양풍 소스 영역 tracer / enrichment of low-FIP elements (Si, Mg, Fe) in corona/solar wind; a source-region tracer |
| **Slit scan (rastering)** | 망원경 거울을 회전시켜 슬릿을 태양면에 걸쳐 쓸어 2D 영상 구축 / rotating the telescope mirror to scan the slit across the Sun to build a 2D image |
| **Dichroic coating** | 특정 파장만 반사하고 나머지는 투과시키는 코팅. SPICE 거울은 boron carbide 10 nm 층으로 EUV만 반사 / coating that reflects only EUV (B₄C, 10 nm) — dumps ~70% of solar spectrum for heat management |
| **MCP (Micro-Channel Plate)** | 광전면과 APS 사이에 놓인 전자 증폭기 — EUV 광자 → 가시광 변환 경로에서 이득 제공 / electron multiplier between photocathode and APS; amplifies signal |
| **APS (Active Pixel Sensor)** | SPICE 검출기의 CMOS 기반 디지털 이미지 센서 / CMOS-based digital image sensor in SPICE's detector |
| **SOU (SPICE Optics Unit)** | SPICE의 광학 구조체; CFRP + Al honeycomb 재질, 질량 ~13 kg / SPICE's optics bench; CFRP+Al honeycomb, ~13 kg |
| **SDM / SCM / SFM** | Slit Door/Change/Scan-Focus Mechanism — 슬릿 도어·슬릿 교체·거울 스캔 포커스 기구 / instrument mechanisms for door seal, slit selection, and mirror scan/focus |
| **Dumbbell aperture** | 슬릿 양끝의 정사각 구멍(0.5′×0.5′). 슬릿 이외의 소규모 영상 획득 → pointing 정보 / square alignment apertures (0.5′×0.5′) at slit ends; give pointing info |
| **Solar constant** | 1 AU에서 태양 방사 플럭스 ~1361 W/m². 0.28 AU에서 13배 / solar flux at 1 AU (~1361 W/m²); ~13× at 0.28 AU |
| **Doppler velocity accuracy** | SPICE는 긴 파장 기준 ~5 km/s 정확도 목표 / SPICE aims for ~5 km/s line-centroid accuracy (at longer wavelengths) |

---

## 5. 수식 미리보기 / Equations Preview

### (1) 회절격자 방정식 / Grating equation

$$
\sin(\theta_m) = m \cdot \frac{\lambda}{d} + \sin(\theta_i)
$$

- $m$: 회절 차수 (diffraction order) — SPICE는 주로 1차, LW 단파장은 2차
- $\lambda$: 파장 (wavelength)
- $d$: 격자 간격 (ruling period) — SPICE는 $d = 1/2400$ mm
- $\theta_i$: 입사각 (angle of incidence) — $-1.7584°$
- $\theta_m$: 회절각 (angle of diffraction) — $+8.55°$(SW), $+12.24°$(LW)

**한국어** 분광기의 색분산이 이 식에서 시작된다. TVLS 격자는 $d$가 표면 위치에 따라 ~1% chirp되도록 설계되어, 파장에 따른 수차를 소자 내부에서 보정한다.
**English** Wavelength dispersion follows directly. TVLS adds a **~1% chirp** in $d$ across the aperture so aberrations are corrected inside the grating itself.

### (2) 도플러 속도 / Doppler velocity (line-of-sight flow)

$$
\frac{\Delta \lambda}{\lambda_0} = \frac{v_\mathrm{LOS}}{c}
$$

**한국어** 라인 중심의 이동으로 시선방향 플라즈마 속도 측정. SPICE 목표 정확도는 긴 파장에서 ~5 km/s. 이것이 태양풍의 소스를 찾아내는 1차 관측량.
**English** Line-centroid shift yields line-of-sight plasma velocity; SPICE targets ~5 km/s accuracy at longer wavelengths — the primary observable for identifying solar-wind source regions.

### (3) 비열적 선폭 / Non-thermal broadening

$$
\sigma_\mathrm{obs}^2 = \sigma_\mathrm{inst}^2 + \underbrace{\frac{\lambda_0^2}{c^2}\frac{2 k_B T_i}{M_i}}_{\text{thermal}} + \underbrace{\frac{\lambda_0^2}{c^2}\xi^2}_{\text{non-thermal}}
$$

- $\sigma_\mathrm{inst}$: 기기 선확산함수(LSF) 폭 — SPICE LSF FWHM ≈ 4 픽셀
- $T_i$: 이온 온도 (ion temperature)
- $M_i$: 이온 질량 (ion mass)
- $\xi$: 비열적 속도 성분 (non-thermal velocity) — 파동·난류 진단의 핵심

**한국어** 관측 선폭을 기기·열·비열 성분으로 분리. 이 식으로 SPICE가 **wave/turbulence 진단**을 수행한다.
**English** Decomposes the observed linewidth into instrumental, thermal, and non-thermal components; enables SPICE's wave/turbulence diagnostics.

### (4) 배출측정량(Emission measure) / Emission measure and line intensity

$$
I_\mathrm{line} \propto \mathrm{Ab}(X) \cdot G(T, n_e) \cdot \mathrm{EM}, \qquad \mathrm{EM} = \int n_e n_H \, dl
$$

- $\mathrm{Ab}(X)$: 원소 존재량 (elemental abundance) — FIP 편향으로 변화
- $G(T, n_e)$: contribution function — 이온의 여기·전리 통계(CHIANTI 등에서 계산)
- EM: 방출측정량 (emission measure) — 플라즈마의 양·밀도 정보

**한국어** 여러 라인의 강도로부터 EM과 원소 존재비를 역산 — **FIP bias 지도**를 만든다.
**English** Multi-line intensities are inverted to yield EM and abundances — producing **FIP-bias maps**.

### (5) 태양 플럭스의 거리의존성 / Solar flux vs. heliocentric distance

$$
F(r) = F_\oplus \left(\frac{1 \, \mathrm{AU}}{r}\right)^2
$$

**한국어** $r=0.28$ AU에서 $F \approx 12.8 \, F_\oplus$. 이 13배 플럭스가 **SPICE 열·기계 설계의 주된 제약**.
**English** At $r=0.28$ AU, $F \approx 12.8\,F_\oplus$. This ~13× enhancement is the dominant constraint on SPICE's thermal/mechanical design.

---

## 6. 읽기 가이드 / Reading Guide

**한국어**
이 논문은 25페이지의 긴 instrument paper입니다. 모든 세부를 읽을 필요는 없습니다. 다음 순서로 읽으시길 권장합니다:

1. **우선 읽기 (1차 통독)** — Sections 1–3, 4.1, 9, 10
   - §1 Introduction: 임무의 큰 그림
   - §2 Scientific objectives: SPICE가 **무엇을** 관측하는지 — Table 1(spectral lines)이 매우 중요
   - §3 Instrument overview: 기기의 구성요소 개요
   - §4.1 Imaging resolution: 핵심 성능 지표
   - §9 Performance vs. science requirements: 기기가 설계대로 작동하는가
   - §10 Observation examples: 실제 관측 예시 — 여기서 "SPICE가 할 수 있는 것"의 감이 잡힙니다
2. **2차 읽기 (선택적 심화)** — 관심 있는 서브시스템만
   - §4 Optical design, §5 Mechanical/thermal design, §6 Electronics, §7 Operations, §8 Data processing
3. **수식 이해 체크포인트**
   - Table 1의 각 라인이 왜 선택되었는지 설명할 수 있는가?
   - 0.28 AU의 ~17 kW/m² 열부하를 어떻게 관리하는가(dichroic mirror + 열제거 거울 + heat shield)?
   - TVLS grating이 왜 단일 소자로 수차를 잡을 수 있는가(toroidal + chirp)?
4. **중요한 그림**
   - Fig. 1: Spacecraft 내 SPICE 위치
   - Fig. 2, 3: 광학 구성 및 광로도(가장 중요한 그림)
   - Fig. 4: Detector layout — SW/LW 분리
   - Table 1: 관측 라인 목록(여러 번 돌아올 것)
   - Table 2: 광학 시스템 파라미터

**English**
This is a 25-page instrument paper — you do **not** need every detail on the first pass.
1. **First pass**: Sections 1–3, 4.1, 9, 10 (science goals → design overview → performance → observation examples).
2. **Second pass (selective)**: §4 optics, §5 mechanical/thermal, §6 electronics, §7 operations, §8 data processing — read whichever subsystem interests you.
3. **Checkpoints**: Can you justify each line choice in Table 1? How is the ~17 kW/m² heat load at 0.28 AU managed? Why does a TVLS grating correct aberrations alone?
4. **Key figures**: Fig. 1 (spacecraft), Figs. 2–3 (optical layout — the most important), Fig. 4 (detector layout), Table 1 (line list — revisit repeatedly), Table 2 (optical parameters).

---

## 7. 현대적 의의 / Modern Significance

**한국어**
SPICE는 2020년 발사 이후 **태양물리학의 대표적인 현역 분광기**로 자리잡았습니다. 본 논문이 출판된 2020년 이후의 과학적 성과와 의의는 다음과 같습니다:

1. **극지 태양풍 기원의 분광 진단 (최초)** — Solar Orbiter의 황도면 경사가 17°를 넘어서는 2025년부터 SPICE는 **인류 최초로 태양 극지의 EUV 분광 관측**을 시작했습니다. 이는 "빠른 태양풍이 어디서 어떻게 시작되는가"라는 60년 된 미해결 문제에 직접적인 답을 제공합니다.
2. **Remote-sensing ↔ in-situ 연결의 완성** — SPICE의 FIP-bias 지도와 Doppler 지도를 in-situ 기기(SWA 조성, MAG 자기장, EPD 입자)와 결합하면 **"이 태양풍 덩어리가 태양의 어디서 왔는가"를 실시간으로 추적**할 수 있습니다.
3. **소규모 가열 이벤트와 파동** — IRIS(채층/낮은 TR)와 EIS(코로나)의 공백을 메우는 SPICE의 **넓은 온도 범위**는 나노플레어·파동에 의한 코로나 가열 메커니즘을 규명하는 데 핵심적 역할을 합니다.
4. **Parker Solar Probe와의 공조** — PSP는 태양에 더 가깝지만 원격관측은 없습니다. Solar Orbiter + PSP의 동시 관측은 **"가장 가까이서 바라보는 + 직접 통과하는"** 쌍을 완성합니다.
5. **공학적 유산** — 0.28 AU에서의 dichroic B₄C mirror, CFRP 광학 벤치, TVLS 격자 등은 이후 **차세대 태양 관측 미션(e.g., MUSE, Solar-C EUVST)** 에 직접 영향을 주고 있습니다.

**English**
Since its 2020 launch, SPICE has become one of the flagship active solar spectrometers:

1. **First-ever spectroscopy of the solar poles** — with orbital inclination passing 17° in 2025, SPICE is now delivering the first polar EUV spectra, directly addressing the 60-year-old question of the fast-wind origin.
2. **Closing the remote/in-situ loop** — combining SPICE FIP-bias and Doppler maps with SWA/MAG/EPD makes it possible to **trace a given parcel of solar wind back to its source region on the Sun in near real time**.
3. **Small-scale heating and waves** — SPICE's temperature coverage fills the gap between IRIS (chromosphere/lower TR) and EIS (hot corona), enabling tests of nanoflare and wave-based coronal heating.
4. **Parker Solar Probe synergy** — PSP goes closer (0.05 AU) but has no imaging. Solar Orbiter + PSP together form the **"closest imager + in-situ flyby"** pair that solar physics always wanted.
5. **Engineering legacy** — the dichroic B₄C mirror, CFRP optical bench, and TVLS grating design are influencing next-generation missions (MUSE, Solar-C EUVST).

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
