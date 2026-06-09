---
title: "Pre-Reading Briefing: Large-scale Globally Propagating Coronal Waves"
paper_id: "42"
topic: Living_Reviews_in_Solar_Physics
date: 2026-04-23
type: briefing
---

# Large-scale Globally Propagating Coronal Waves: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Warmuth, A., "Large-scale Globally Propagating Coronal Waves", *Living Reviews in Solar Physics*, **12**, 3 (2015). DOI: 10.1007/lrsp-2015-3
**Author(s)**: Alexander Warmuth (Leibniz-Institut für Astrophysik Potsdam, AIP)
**Year**: 2015

---

## 1. 핵심 기여 / Core Contribution

이 리뷰 논문은 1960년대 Hα Moreton wave 발견 이후 1997년 SOHO/EIT가 극자외선(EUV)으로 코로나 전체를 휩쓸고 지나가는 거대한 wave-like 교란을 포착하기 시작하면서 본격적으로 연구된 "large-scale globally propagating coronal waves" 현상의 관측·이론·모델을 종합 정리한다. Warmuth는 관측 증거(EUV, SXR, white light, Hα, He I, radio 등)가 "fast-mode MHD wave/shock" 해석과 "magnetic reconfiguration (pseudo-wave)" 해석 사이의 15년 이상 이어진 논쟁에서 어떻게 통합된 hybrid scenario — 빠른 외곽 fast-mode wavefront + 느린 내곽 CME 팽창에 의한 non-wave 증광 — 로 수렴하고 있는지를 보여준다.

This review synthesizes more than a decade and a half of observational, theoretical, and numerical work on large-scale coronal wave-like disturbances, from the first Hα Moreton wave observations in the 1960s through the revolutionary SOHO/EIT detections in 1997 and the high-cadence era of SDO/AIA and STEREO/EUVI. Warmuth demonstrates how the long-standing controversy between the "fast-mode MHD wave/shock" interpretation and the "magnetic reconfiguration / pseudo-wave" interpretation is being resolved by a unified hybrid scenario: an outer fast-mode MHD wave or shock accompanied by a slower inner brightening caused by the lateral expansion of an erupting coronal mass ejection (CME). The review shows how each observational constraint — kinematics, perturbation profile, Mach number, thermal response, interaction with coronal structures — can be used to discriminate between competing models.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

Moreton waves는 1960년 R. Moreton이 Lockheed Solar Observatory의 고 cadence Hα 필터그램에서 발견했고, 약 1000 km/s로 플레어 활동 영역에서 바깥으로 전파하는 호(arc) 모양의 교란으로 보였다. 1968년 T. Uchida는 이 Hα 신호가 실제로는 코로나에서 팽창하는 fast-mode MHD wavefront가 채층을 "sweeping-skirt"처럼 아래로 눌러서 생기는 지상 자취라고 제안했다(Uchida 1968, "sweeping-skirt hypothesis"). 이 예측된 코로나 상대물은 1997년 SOHO/EIT(195Å)가 Thompson et al. (1998)이 문서화한 "EIT waves"로 직접 영상화하면서 비로소 확인되었다.

Moreton waves were discovered by R. Moreton in 1960 using high-cadence Hα filtergrams at Lockheed Solar Observatory: arc-shaped disturbances propagating at ~1000 km/s away from flaring active regions. In 1968 T. Uchida proposed the "sweeping-skirt hypothesis": the chromospheric Hα signature is the ground track of a dome-shaped fast-mode MHD wavefront expanding through the corona, whose flanks press down on the denser chromosphere. This predicted coronal counterpart was finally imaged in 1997 by SOHO/EIT (195 Å), famously documented by Thompson et al. (1998), initiating the modern era of coronal wave research.

### 타임라인 / Timeline

```
1947 --- Payne-Scott: metric type II radio bursts (later interpreted as coronal shocks)
1960 --- Moreton: Hα Moreton waves
1964 --- Moreton (1964): Doppler interpretation — chromospheric depression
1968 --- Uchida: "sweeping-skirt" fast-mode MHD wave hypothesis
1973 --- Uchida et al.: refined coronal fast-mode wave simulations
1989 --- Neupert: first tentative fast EUV disturbance report
1995 --- SOHO launched (EIT instrument)
1997 --- Thompson, Moses et al.: first "EIT waves" imaged
1999 --- Delannée & Aulanier: stationary bright fronts → pseudo-wave challenge
2002 --- Chen et al.: field-line stretching / hybrid model (2D MHD)
2004 --- Zhukov & Auchère: bimodality of coronal waves
2005 --- Warmuth et al.: combined Hα + EIT + SXI deceleration curves
2007 --- Attrill et al.: reconnection-front pseudo-wave model
2008 --- Delannée et al.: 3D current-shell model
2008 --- STEREO era: stereoscopic EUV waves (EUVI)
2009 --- Gopalswamy et al.: first clear reflection at coronal hole
2010 --- SDO/AIA launched: 12 s cadence, 7 EUV channels
2010 --- Veronig et al.: dome-like coronal wave (2010 Jan 17)
2011 --- Warmuth & Mann: three kinematical classes (wave / weak wave / pseudo-wave)
2011 --- Downs et al.: 3D thermodynamic MHD synthesis vs AIA
2013 --- Kwon et al.: WL STEREO/COR1 counterpart in upper corona
2015 --- Warmuth: this review — unified hybrid scenario
```

---

## 3. 필요한 배경 지식 / Prerequisites

- **Ideal MHD (ideal magnetohydrodynamics)** 방정식과 세 가지 선형 파동 모드: Alfvén, fast-mode, slow-mode 파의 분산 관계.
  Ideal MHD equations and the three linear MHD wave modes — Alfvén, fast-mode, slow-mode — and their dispersion relations.

- **Rankine–Hugoniot 충격파 관계식**과 fast-mode perpendicular Mach number 개념.
  Rankine–Hugoniot shock jump conditions and the concept of fast magnetosonic Mach number $M_\mathrm{ms}$.

- **태양 코로나의 구조**: quiet corona ($n_e \approx 5 \times 10^8$ cm$^{-3}$, $T \approx 1.5$ MK, $B \approx 3$ G), active regions, coronal holes, magnetic topology (separatrices), differential emission measure (DEM).
  Structure of the solar corona: quiet corona parameters, active regions, coronal holes, separatrices, and DEM.

- **CME와 eruptive event 현상학**: flare / CME / eruption 관계, type II radio bursts, coronal dimmings.
  CME phenomenology: flares, CMEs, type II radio bursts, and coronal dimmings.

- **관측 기기**: SOHO/EIT, STEREO/EUVI, SDO/AIA (EUV), Yohkoh/SXT, GOES/SXI, Hinode/XRT (SXR), Kanzelhöhe GBOs (Hα), LASCO/COR1 (white light), ground radiospectrographs (type II bursts).
  Instrumentation: major EUV, SXR, Hα, white-light, and radio instruments listed in Table 1 of the paper.

- **Geometric acoustics / ray tracing in stratified media**: Snell의 법칙과 feast-mode speed의 공간적 변화에 의한 wave refraction.
  Geometric acoustics / ray tracing: Snell's-law-like refraction of MHD wave rays in a medium with spatially varying fast-mode speed.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| Moreton wave | Hα에서 관측되는 호 모양의 빠른(≈1000 km/s) 채층 전파 교란; 코로나 fast-mode wavefront의 채층 자취로 해석 / Arc-shaped Hα disturbance at ~1000 km/s interpreted as the chromospheric ground track of a coronal fast-mode wavefront |
| EIT wave | SOHO/EIT가 처음 영상화한 코로나 EUV 파동 전면(195 Å); 광범위 속도 분포(수 10 ~ 700 km/s) / EUV coronal wavefronts first imaged by SOHO/EIT at 195 Å with speeds spanning tens to ~700 km/s |
| Fast-mode MHD wave | 자기 및 기체 압력이 함께 복원력 역할; 자기장에 거의 무관하게 전파; $c_f = \sqrt{v_A^2 + c_s^2}$ (수직 전파 한계) / Compressive MHD mode with combined magnetic + gas pressure restoring force |
| Alfvén speed $v_A$ | $v_A = B / \sqrt{4\pi \rho}$; 자기장 강도와 밀도로 결정되는 특성 속도 / Characteristic speed set by magnetic tension |
| Magnetosonic speed $v_\mathrm{ms}$ | $v_\mathrm{ms} = \sqrt{v_A^2 + c_s^2}$; 수직 fast-mode wave의 속도 / Speed of perpendicular fast-mode wave |
| Mach number $M_\mathrm{ms}$ | $M_\mathrm{ms} = v_\mathrm{shock} / v_\mathrm{ms}$; fast magnetosonic 충격파의 Mach number / Fast magnetosonic Mach number of a shock |
| Pseudo-wave | 진짜 MHD 파동이 아닌, 팽창하는 CME에 의한 magnetic reconfiguration으로 밝기 증가가 겉보기 전파처럼 보이는 현상 / Apparent propagating brightening caused by CME-driven magnetic reconfiguration, not a real MHD wave |
| Dome-like wavefront | 지표에 붙는 flank(측면)와 코로나 상부로 상승하는 dome(돔)을 동시에 보여주는 3D wavefront / 3D wavefront showing both lateral (on-disk) flanks and upward-propagating dome |
| Stationary brightening | 파동이 지나간 후 특정 경계(주로 separatrix)에서 지속되는 비전파 밝기 / Persistent non-propagating brightening at a topological boundary after a wave passes |
| Coronal dimming | 파동 사건 중 CME 체적이 상승해 밀도가 줄어 밝기가 감소한 영역 / Region of reduced emission caused by plasma evacuation from an erupting CME |
| Type II radio burst | 코로나 충격파 앞쪽의 전자 가속에 의한 느린 주파수 drift radio 방출; 충격파의 원격 진단자 / Slow-drifting metric radio emission from shock-accelerated electrons; remote signature of a coronal shock |
| Coronal seismology | 관측된 MHD 파동 속도와 성질을 이용해 코로나 자기장·밀도·온도를 역으로 추정하는 진단법 / Diagnostic technique using MHD wave observations to infer coronal plasma parameters |

---

## 5. 수식 미리보기 / Equations Preview

### (a) Sound speed / 음속

$$c_s = \sqrt{\frac{\gamma_\mathrm{ad} k T}{\bar{\mu} m_p}}$$

$\gamma_\mathrm{ad} = 5/3$, $\bar{\mu} = 0.6$ for fully ionized plasma. For $T = 1.5$ MK this gives $c_s \approx 185$ km/s.

### (b) Alfvén speed / 알벤 속도

$$v_A = \frac{B}{\sqrt{4 \pi \rho}} = \frac{B}{\sqrt{4 \pi \bar{\mu} m_p n}}$$

자기장 강도와 밀도에 의해 결정되는 특성 속도. Quiet corona에서 $B \approx 3$ G, $n_e = 5 \times 10^8$ cm$^{-3}$이면 $v_A \approx 273$ km/s.

### (c) Fast/slow-mode speed / 빠른·느린 모드 속도

$$v_{f/s} = \left\{ \frac{1}{2} \left[ v_A^2 + c_s^2 \pm \sqrt{(v_A^2 + c_s^2)^2 - 4 v_A^2 c_s^2 \cos^2\theta_B} \right] \right\}^{1/2}$$

+ 부호는 fast-mode, − 부호는 slow-mode. $\theta_B$는 파동 벡터와 자기장 사이의 각.

### (d) Magnetosonic (perpendicular fast-mode) speed / 수직 자기음속

$$v_\mathrm{ms} = \sqrt{v_A^2 + c_s^2}$$

수직 전파($\theta_B = 90^\circ$)에서 fast-mode 속도. Quiet corona 기본값 $v_\mathrm{ms} \approx 330$ km/s (관측된 EIT wave 속도 범위와 정합).

### (e) Fast magnetosonic Mach number / 자기음속 Mach 수

$$M_\mathrm{ms} = \sqrt{\frac{(5\beta_p + 5 + X) X}{(2 + \gamma \beta_p)(4 - X)}}$$

$X = n_d / n_u$는 압축비, $\beta_p$는 플라즈마 베타, $\gamma = 5/3$. Rankine–Hugoniot 관계에서 유도된 수직 fast-mode 충격 Mach 수.

---

## 6. 읽기 가이드 / Reading Guide

1. **Sections 1 ~ 2 (Introduction + Physical Concepts)** — MHD 파동 모드와 비선형 steepening, shock 형성, pseudo-wave 개념을 확실히 숙지한다. 뒤에서 계속 인용된다.
   Make sure you internalize the MHD wave modes, nonlinear steepening, shock formation, and pseudo-wave definitions — everything else depends on them.

2. **Section 3 (Observational Signatures)** — 파장별(EUV, SXR, WL, Hα, He I, radio) 관측 특성을 표처럼 정리하며 읽자. 각 채널이 어떤 파라미터(온도, 밀도, 충격파 강도)를 제약하는지 생각.
   Read Section 3 as a catalog: each wavelength channel constrains a different plasma parameter.

3. **Section 4 (Physical Characteristics)** — 가장 정보량이 큰 장. 특히 4.2 Kinematics(세 class), 4.3 Perturbation profile, 4.4 Mach numbers, 4.7 Interaction(refraction, reflection, transmission)은 모델 구분 핵심 근거다.
   Section 4 is the richest — focus on kinematical classification (three classes), perturbation profile evolution, Mach numbers, and wave–structure interactions.

4. **Section 5 (Relationship with eruptive events)** — flare vs CME 중 어느 쪽이 "driver"인지, Moreton wave와 CME lateral expansion의 시공간적 연관.
   Section 5: what actually drives coronal waves — flares or CMEs?

5. **Section 6 (Models)** — 세 가지 "진짜 파동" 모델(fast-mode wave/shock, slow-mode soliton, surface gravity)과 세 가지 "pseudo-wave" 모델(field-line stretching, current shell, reconnection front), 그리고 이들을 통합하는 hybrid scenario.
   Section 6 is the theoretical core. Focus on how each model predicts specific observables and where each one fails.

6. **Section 7 ~ 8 (Wider significance + Conclusions)** — coronal seismology와 SEP 가속, 향후 과제.
   Section 7–8: broader implications and open questions.

---

## 7. 현대적 의의 / Modern Significance

태양 코로나 파동은 여전히 활발한 연구 주제로, 이 2015년 리뷰는 그 분야의 표준 참고 문헌이다. 현대적 의의는 (1) SDO/AIA·STEREO·Hinode/EIS의 고 cadence·다파장 데이터를 종합하는 틀을 제시했고, (2) hybrid scenario가 coronal mass ejection, solar energetic particle, space weather forecasting에 직접 응용 가능함을 보이고, (3) global coronal seismology를 통해 측정하기 어려운 quiet corona의 $B$ field를 수 Gauss 수준으로 추정할 수 있음을 정립한 데 있다. 특히 SEP의 광범위 경도 분포를 설명하는 유력 메커니즘으로 coronal wave-driven shock이 부상했으며, Parker Solar Probe, Solar Orbiter 시대의 coronal shock 연구에도 핵심 배경이 된다.

The 2015 Warmuth review has become the standard reference in the field. Its modern significance lies in (1) establishing a coherent framework that integrates SDO/AIA, STEREO, and Hinode/EIS multi-wavelength high-cadence data; (2) providing the hybrid "fast-mode wave + CME-driven pseudo-wave" scenario now widely adopted in CME, SEP, and space-weather research; and (3) cementing global coronal seismology as a practical diagnostic of the otherwise inaccessible quiet-Sun magnetic field (few Gauss level). Coronal wave-driven shocks have emerged as a leading candidate for explaining the wide longitudinal spread of solar energetic particle (SEP) events, making this review foundational background for the Parker Solar Probe and Solar Orbiter eras.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
