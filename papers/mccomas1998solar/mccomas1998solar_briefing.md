---
title: "Pre-Reading Briefing: Solar Wind Electron Proton Alpha Monitor (SWEPAM) for the Advanced Composition Explorer"
paper_id: "66_mccomas_1998"
topic: Space_Weather
date: 2026-04-25
type: briefing
---

# SWEPAM for ACE: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: McComas, D. J., Bame, S. J., Barker, P., Feldman, W. C., Phillips, J. L., Riley, P., and Griffee, J. W., "Solar Wind Electron Proton Alpha Monitor (SWEPAM) for the Advanced Composition Explorer", Space Science Reviews 86, 563-612, 1998. DOI: 10.1023/A:1005040232597
**Author(s)**: D. J. McComas, S. J. Bame, P. Barker, W. C. Feldman, J. L. Phillips, P. Riley (Los Alamos National Laboratory); J. W. Griffee (Sandia National Laboratory)
**Year**: 1998

---

## 1. 핵심 기여 / Core Contribution

SWEPAM은 ACE 우주선의 태양풍 플라즈마 모니터로, 두 개의 독립 기기 — 이온 기기(SWEPAM-I)와 전자 기기(SWEPAM-E) — 를 사용해 양성자, 알파 입자, 전자의 3차원 속도 분포 함수(VDF)를 측정한다. SWEPAM은 NASA/ESA Ulysses 임무의 SWOOPS 비행 예비품(flight spare)을 재활용하여 비용을 절감했지만, ACE 임무 요구에 맞추어 (1) 헤일로 전자(>100 eV) 감도 16배 향상, (2) 이온 측정 극각(polar angle) 분해능을 5°에서 2.5°로 절반 단축, (3) 태양풍 빔 외부의 초열적(suprathermal) 이온 측정용 20° 원뿔 영역 추가 — 세 가지 주요 개선이 이루어졌다. 본 논문은 ACE/SWEPAM의 1차 문서로 기능하며, 과학 목표, 정전기 분석기(ESA) 전자광학 설계, 채널 전자 증배기(CEM) 검출 시스템, 보정 결과, 운영 모드, 그리고 우주 기상 모니터링용 실시간 텔레메트리 능력을 포괄적으로 기술한다.

SWEPAM is the bulk solar wind plasma monitor for the Advanced Composition Explorer (ACE), comprising two independent instruments — an ion sensor (SWEPAM-I) and an electron sensor (SWEPAM-E) — that measure the three-dimensional velocity distribution functions (VDFs) of protons, alpha particles, and electrons. SWEPAM was built from refurbished flight spare hardware of the joint NASA/ESA Ulysses SWOOPS experiment to save cost, but underwent three major upgrades for ACE: (1) a factor-of-16 increase in halo-electron (>100 eV) accumulation interval; (2) halving of the effective ion-detecting CEM polar spacing from ~5° to ~2.5°; and (3) addition of a 20° conical swath of enhanced sensitivity for measuring suprathermal ions outside the bulk solar-wind beam. This paper serves as the primary documentation source for the ACE/SWEPAM experiment, covering scientific objectives, electrostatic-analyzer (ESA) electro-optics, channel electron multiplier (CEM) detection, calibration results, operational modes, and real-time telemetry capability for space-weather purposes.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

1990년대 후반, 태양풍 *in situ* 측정은 IMP, ISEE-3, Helios, Ulysses 등 30년의 유산을 축적하고 있었다. 그러나 코로나 가열 메커니즘, 느린 태양풍의 기원, CME의 자기 위상학 등은 여전히 미해결이었다. ACE 임무(1997년 8월 발사, L1 도착 1998년 초)는 이 질문들을 동위원소·원소 조성 측정의 새로운 정밀도로 공격하기 위해 설계되었으며, SWEPAM은 모든 6개 조성 실험(CRIS, SIS, ULEIS, SEPICA, SWIMS, SWICS)이 해석되는 *플라즈마 컨텍스트*를 제공한다. 비용 절감을 위해 SWEPAM 팀은 1990년에 발사된 Ulysses SWOOPS 비행 예비품을 재정비·현대화했다.

In the late 1990s, *in situ* solar wind measurements had ~30 years of heritage from IMP, ISEE-3, Helios, and Ulysses. Yet coronal heating, the origin of the slow wind, and the magnetic topology of CMEs remained unsolved. The ACE mission (launched August 1997; arrived at L1 in early 1998) was designed to attack these questions with unprecedented isotopic and elemental composition precision, and SWEPAM provides the *plasma context* in which all six composition experiments (CRIS, SIS, ULEIS, SEPICA, SWIMS, SWICS) are interpreted. To save cost, the SWEPAM team refurbished and modernized the flight spares of the Ulysses SWOOPS instruments, which had launched in 1990.

### 타임라인 / Timeline

```
1958 ──── Parker: Solar wind theoretical prediction
1962 ──── Mariner 2: First in situ solar wind measurements (Neugebauer & Snyder)
1977 ──── ISEE-3 launch (L1 plasma + fields heritage)
1990 ──── Ulysses launch with SWOOPS (Bame et al. 1992)
1992 ──── Ulysses Jupiter swingby into polar orbit
1994-95 ─ Ulysses fast latitude scan: bimodal solar wind discovered
1997 ──── ACE launch (August 25); SWEPAM commissioning at L1
1998 ──── This paper published (Space Sci. Rev. 86)
```

---

## 3. 필요한 배경 지식 / Prerequisites

**물리 / Physics**
- 정전기 에너지 분석(electrostatic energy analysis): $qV = \tfrac{1}{2}m v^2$ 와 곡판 분석기의 $E/q$ 선택 원리
- Maxwell-Boltzmann 분포와 VDF의 모멘트(밀도, 속도, 온도 텐서, 열속)
- 태양풍 양성자/알파 빔 운동학(coronal hole vs streamer belt 풍속, 4-5% 헬륨 함량)

**Mathematics**
- VDF 모멘트 적분 $n = \int f \, d^3v$, $n\vec{u} = \int \vec{v} f \, d^3v$, $n k_B T = \tfrac{m}{3}\int |\vec{v}-\vec{u}|^2 f \, d^3v$
- 기하 인자 $G = \int\int A(\vec{r})\,T(E,\theta,\phi)\,d\Omega\,dE/E$ 의 의미

**기기 / Instrumentation**
- 곡판형 정전기 분석기(spherical-section ESA), 채널 전자 증배기(CEM)의 단일 입자 계수
- 회전 우주선의 *fan-shaped* FOV가 4π sr을 어떻게 휩쓰는지

**선행 논문 / Prior papers**
- Bame et al. (1992) "Ulysses Solar Wind Plasma Experiment" (SWOOPS heritage)
- Gosling et al. (1978) "Effects of a Long Entrance Aperture upon the Azimuthal Response of Spherical Section ESAs"

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **ESA (Electrostatic Analyzer)** | 두 곡판 사이의 전압이 입사 입자의 $E/q$ 비를 선택; SWEPAM-I는 105° 굴절각, SWEPAM-E는 120° 굴절각. The voltage between two curved plates selects the energy-per-charge ($E/q$) of the transmitted particle; bending angles 105° (ion) and 120° (electron). |
| **CEM (Channel Electron Multiplier)** | 분석기 출구 단일 입자 펄스를 증폭하는 검출기; SWEPAM-I 16개, SWEPAM-E 7개. Single-particle pulse-amplifying detectors at analyzer exits — 16 in SWEPAM-I and 7 in SWEPAM-E. |
| **Analyzer constant K** | $K = E_{\text{ion}} / V_{\text{plate}}$; SWEPAM-I K=17.1, SWEPAM-E K=4.3. The ratio of ion energy to plate voltage. |
| **Fan-shaped FOV** | 우주선 자전축 둘레로 회전하며 4π sr을 휩쓰는 부채꼴 시야. Fan-shaped field-of-view that rotates around the spacecraft spin axis to cover ~4π sr. |
| **Geometric factor G** | 입자 플럭스를 계수율로 변환하는 기기 응답 부피적분; 단위 cm² sr eV/eV. Volumetric instrument response converting flux to count rate; units cm² sr eV/eV. |
| **VDF (Velocity Distribution Function)** | 위상 공간 밀도 $f(\vec{v})$; 모든 플라즈마 모멘트의 원천. Phase-space density $f(\vec{v})$, the source of all plasma moments. |
| **Suprathermal** | 태양풍 빔 코어보다 훨씬 빠른 꼬리 입자(ion >5 keV/q, electron >100 eV). Tail particles much faster than the bulk-wind core. |
| **Halo electrons** | 전자 VDF의 비-Maxwellian 고속 꼬리(>~70 eV); CME 카운터 스트리밍의 표지자. Non-Maxwellian high-speed tail of electron VDF (>~70 eV); marker of CME counter-streaming. |
| **Polar / azimuthal angle** | $\theta$ = 자전축에서의 각 (CEM 식별); $\phi$ = 자전 위상으로부터의 각 (회전 시간으로). $\theta$ identified by CEM number; $\phi$ identified by spin phase. |
| **SWI / SSTI / NSWE / STEA / PHE** | SWEPAM 데이터 모드 약어 (Solar Wind Ion / Search Suprathermal Ion / Normal Solar Wind Electron / Suprathermal Electron Angle scan / Photoelectron). SWEPAM data-mode acronyms. |
| **RTSW (Real Time Solar Wind)** | NOAA SEC에 24/7 다운링크되는 64초 케이던스 부분 데이터셋; 우주 기상 경보용. Subset data set downlinked 24/7 to NOAA SEC at 64 s cadence; used for space weather alerts. |
| **L1 (Lagrangian point 1)** | ACE의 정점 위치, 태양-지구 사이 ~0.99 AU; 지구 도달 ~1시간 전 태양풍 경고 가능. ACE's vantage; provides ~1 hr advance warning of solar wind reaching Earth. |

---

## 5. 수식 미리보기 / Equations Preview

### (1) 정전기 분석기 / Electrostatic analyzer
$$\frac{E}{q} = K \cdot V_{\text{plate}}$$
입사 이온의 에너지(eV)와 인가 전압의 비례 관계; SWEPAM-I에서 K≈17.1. The ion energy in eV is K times the analyzer plate voltage; K≈17.1 for SWEPAM-I.

### (2) VDF 0차 모멘트 (밀도) / Zeroth moment (density)
$$n = \int f(\vec{v}) \, d^3 v$$
관측된 계수 $C_i$ 로부터: $f_i = C_i / (G_i \cdot \tau \cdot v_i^4 \cdot \Delta E/E)$. 여기서 $\tau$는 적분시간, $G_i$는 픽셀 기하인자, $v_i$는 픽셀 속도. From observed counts: $f_i = C_i / (G_i \tau v_i^4 \Delta E / E)$, with $\tau$ accumulation time, $G_i$ pixel geometric factor, $v_i$ pixel speed.

### (3) 1차 모멘트 (벌크 속도) / First moment (bulk velocity)
$$\vec{u} = \frac{1}{n} \int \vec{v} \, f(\vec{v}) \, d^3 v$$
실험적으로는 $\theta$-$\phi$ 평면의 카운트 분포 모드를 추적해 얻는다. Experimentally obtained by tracking the peak of the count distribution over $\theta$-$\phi$.

### (4) 2차 모멘트 (온도 텐서) / Second moment (temperature tensor)
$$P_{jk} = m \int (v_j - u_j)(v_k - u_k) \, f \, d^3 v, \qquad k_B T = \frac{1}{3 n} \, \mathrm{Tr}(P)$$
Maxwellian 가정 시 온도는 분포의 분산에 비례. Under a Maxwellian assumption, temperature is proportional to the distribution variance.

### (5) 기하 인자 / Geometric factor
$$G_{i} = \int_{E} \int_{\theta} \int_{\phi} A(\vec{r}) \, T_i(E,\theta,\phi) \, d\Omega \, \frac{dE}{E}$$
세 응답 함수(에너지·극각·방위각)의 부피적분; SWEPAM-I CEM당 1-20 × 10⁻⁶ cm² sr eV/eV. Volume integral of three response functions; per-CEM 1-20 × 10⁻⁶ cm² sr eV/eV for SWEPAM-I.

---

## 6. 읽기 가이드 / Reading Guide

**Section 1 (Introduction)** — ACE 임무 컨텍스트, 6개 조성 실험과 SWEPAM의 보완 관계. 표 I 검토. ACE mission context, six composition experiments, SWEPAM's complementary role; review Table I.

**Section 2 (Scientific Objectives)** — 태양풍 형성/가속, 입자 가속/수송, 픽업 이온. 그림 1(Ulysses 양극 풍속), 그림 5-7(CME 자기 위상학)은 핵심 동기. Solar-wind formation/acceleration, particle acceleration/transport, pickup ions.

**Section 3 (SWEPAM-I)** — 이온 기기 전자광학과 보정. 표 II/III(능력/하드웨어), 그림 10(전자광학), 그림 13-15(CEM 4 보정)에 집중. Ion electro-optics & calibration; focus on Tables II/III, Fig. 10, Figs. 13-15.

**Section 4 (SWEPAM-E)** — 전자 기기 (120° ESA, 7 CEMs). 그림 16, 19-22; 양성자 빔으로의 보정 트릭. Electron 120° ESA with 7 CEMs; Figs. 16, 19-22.

**Section 5 (Operations & Data)** — 데이터 모드 (SWI, SSTI, NSWE, STEA, PHE), 표 VI 데이터 산출물, RTSW 우주 기상 채널. Data modes, Table VI products, RTSW.

**시간 분배 / Suggested time allocation**: 4-5 시간 / 4-5 hours: 1h on §1-2 (목표/과학 motivation), 1.5h on §3 (이온 기기 + 보정), 1h on §4 (전자), 1h on §5 (모드/RTSW), 30 min on Tables and Figures.

---

## 7. 현대적 의의 / Modern Significance

**현재까지의 영향 / Lasting impact** (1997 → 2026)

ACE/SWEPAM은 27년 이상 연속 가동되어 (2010년 기준 NASA가 mission 연장) 우주 기상 운영의 *진정한 작동마(workhorse)*가 되었다. NOAA Space Weather Prediction Center는 ACE RTSW 스트림을 사용해 지자기 폭풍 경보를 발령하며, ACE→DSCOVR(2015)→IMAP(2025)→Space Weather Follow-On으로 이어지는 NOAA 운영 라인의 표준이 되었다. SWEPAM의 측정 원리(ESA + CEM, 회전 fan-FOV)는 STEREO-PLASTIC, Solar Orbiter-SWA, Parker Solar Probe-SWEAP-SPC/SPAN 모두에 직접 계승되었다.

ACE/SWEPAM has operated continuously for over 27 years (NASA extended the mission past 2010), becoming the de-facto workhorse of operational space weather. The NOAA Space Weather Prediction Center uses ACE RTSW to issue geomagnetic-storm alerts, and SWEPAM set the standard for the NOAA operational line continuing through DSCOVR (2015), IMAP (2025), and Space Weather Follow-On. SWEPAM's measurement principles (top-hat/spherical ESA + CEMs, rotating fan FOV) are directly inherited by STEREO-PLASTIC, Solar Orbiter-SWA, and Parker Solar Probe SWEAP-SPC/SPAN.

**주요 과학 결실 / Key scientific results enabled**: (1) ACE composition measurements interpreted with full plasma context; (2) >2000 ICME catalogues at L1; (3) precise solar-wind alpha-to-proton ratio variations; (4) electron heat-flux as CME marker; (5) lead time for L1-to-Earth geomagnetic storm warnings (~30-60 min).

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
