---
title: "Pre-Reading Briefing: Prominence Oscillations"
paper_id: "57"
topic: Living_Reviews_in_Solar_Physics
date: 2026-04-23
type: briefing
---

# Prominence Oscillations: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Arregui, I., Oliver, R., & Ballester, J.L. "Prominence oscillations", *Living Reviews in Solar Physics*, 15:3 (2018). DOI: 10.1007/s41116-018-0012-6
**Author(s)**: Iñigo Arregui, Ramón Oliver, José Luis Ballester
**Year**: 2018

---

## 1. 핵심 기여 / Core Contribution

**한국어**: 이 논문은 태양 프로미넌스(filament)에서 관측되는 진동 현상의 **관측·이론·지진학(seismology) 체계를 망라한 종합 리뷰**이다. 이전 Oliver & Ballester (2002) 및 Arregui et al. (2012) 리뷰를 전면 개정·확장하였으며, 진폭에 따라 **대진폭 진동(Large Amplitude Oscillations, LAO; v > 20 km/s)**과 **소진폭 진동(Small Amplitude Oscillations, SAO; v < 3 km/s)**으로 분류된 진동 현상의 관측(주기 P, 감쇠시간 τ, 파장 λ, 위상속도 c_ph, 군속도 v_g, 편광), 이론 모델(MHD 파동: 고속·알펜·저속 모드, slab/thread/flux tube 구성), 감쇠 메커니즘(열적 비단열, 공명흡수, 이온-중성자 충돌, 파동 누출), 그리고 이들 관측·이론을 결합하여 자기장 세기(B ~ 5–30 G), 알펜속도(v_A ~ 100 km/s), 가로 밀도 불균일 길이 스케일 등 직접 측정이 어려운 물리량을 추정하는 **프로미넌스 지진학(prominence seismology)**의 최신 성과(특히 Bayesian 추론 기법)를 정리한다.

**English**: This paper provides a **comprehensive review of solar prominence oscillations covering observations, theoretical MHD models, and seismology applications**. It is a major revision and expansion of Oliver & Ballester (2002) and Arregui et al. (2012), classifying oscillations by velocity amplitude into **Large Amplitude Oscillations (LAO; v > 20 km/s)** typically triggered by flares, EUV waves, or jets, and **Small Amplitude Oscillations (SAO; v < 3 km/s)** that appear local and flare-unrelated. The review systematically covers observational aspects (periods P, damping times τ, wavelengths, phase/group velocities, polarisations), theoretical models based on linear ideal MHD (string analogues, slab and thread/flux-tube configurations supporting fast, Alfvén, and slow modes), damping mechanisms (non-adiabatic thermal processes, ion-neutral collisions, resonant absorption in the Alfvén/slow continua, wave leakage), and **prominence seismology**, where comparison of models and observations yields physical parameters that resist direct measurement — magnetic field strengths B ~ 5–30 G, thread Alfvén speeds ~ 100–150 km/s, and transverse density inhomogeneity scales l/a ~ 0.2. The review emphasises the recent introduction of **Bayesian inference** methods for seismology inversion and model comparison.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

**한국어**: 프로미넌스 진동 연구의 현대적 출발은 플레어에 의해 흔들리는 "winking filament" 현상(Moreton & Ramsey 1960; Ramsey & Smith 1965, 1966; Hyder 1966)과 저진폭 Doppler 진동(Harvey 1969)의 발견이다. 1960–1970년대는 현상학적 관측에, 1980–1990년대는 분광학적 기법(MSDP, SUMER, CDS)과 **slab 이론 모델(Joarder & Roberts 1992a,b; Oliver et al. 1992, 1993)**에 집중되었다. 2000년대 이후 Hinode/SOT, SDO/AIA, SST, IRIS 등 고해상도 기기가 **개별 thread의 진동과 eruption, 전파하는 kink 파동**을 직접 관측하게 했고, 이는 thread를 flux tube로 모델링하는 **fine-structure 이론**과 **공명흡수(resonant absorption) 감쇠**의 발전으로 이어졌다. 2010년대에는 Bayesian seismology라는 통계적 역문제 프레임이 등장했다.

**English**: Modern study of prominence oscillations began with observations of "winking filaments" (Moreton & Ramsey 1960; Ramsey & Smith 1965, 1966; Hyder 1966), flare-driven vertical oscillations that vanish from H-alpha when the Doppler shift exceeds the filter bandpass, and with Harvey's (1969) detection of persistent low-amplitude Doppler oscillations in quiescent prominences. The 1960s–1970s were phenomenological; the 1980s–1990s combined dedicated spectroscopy (MSDP, SUMER, CDS) with **linear MHD slab models** (Joarder & Roberts 1992a,b; Oliver et al. 1992, 1993). From the mid-2000s, space- and ground-based high-resolution imaging (Hinode/SOT, SDO/AIA, SST, DOT, IRIS) revealed **fine-structure thread oscillations, propagating kink waves, and detailed behaviour during eruptions**, prompting flux-tube thread models and the theory of **resonant absorption damping**. In the 2010s **Bayesian inference** was introduced as a rigorous framework for the inversion problem.

### 타임라인 / Timeline

```
1960 ── Moreton & Ramsey: Moreton wave → winking filaments
1965-66 ── Ramsey & Smith; Hyder: 11 winking events, P = 6-40 min
1969 ── Harvey: quiescent Doppler oscillations; v < 2 km/s
1984 ── Roberts et al.: seismology idea (coronal loops)
1991 ── Yi et al.; Yi & Engvold: He I thread oscillations
1992 ── Joarder & Roberts: slab model; fast/Alfvén/slow modes
1995 ── Tandberg-Hanssen: first proposed "prominence seismology"
2002 ── Oliver & Ballester: first Living Reviews article
2002 ── Ruderman & Roberts: resonant absorption for kink damping
2003-06 ── Jing et al.: Large Amplitude Longitudinal Oscillations
2005-07 ── Lin, Okamoto et al.: Hinode/SOT propagating thread waves
2008 ── Soler et al.: non-adiabatic + flow thread damping
2009-10 ── Díaz et al.; Soler et al.: thread seismology inversions
2012 ── Luna & Karpen: pendulum model (P = 2π√(R/g))
2014 ── Arregui et al.: Bayesian prominence seismology
2018 ── ★ Arregui, Oliver, Ballester (this review)
```

---

## 3. 필요한 배경 지식 / Prerequisites

**한국어**:
- **이상 MHD(ideal MHD)**: 연속 방정식, 운동량 방정식, 유도 방정식, 압력·밀도 섭동의 선형화
- **MHD 파동**: 알펜 속도 $v_A = B/\sqrt{\mu_0 \rho}$, 음속 $c_s = \sqrt{\gamma p/\rho}$, 고속·저속 마그네토음향 모드, cusp 속도 $c_T = v_A c_s/\sqrt{v_A^2+c_s^2}$
- **가이드 파동(guided waves)**: 원통/slab flux tube의 kink, sausage, 내부/외부 모드, thin tube 근사
- **분산 관계(dispersion relation)** 해석 및 경계 조건
- **공명흡수(resonant absorption)**: Alfvén/cusp continuum, 가로 비균일층에서의 kink 모드 감쇠
- **통계 추론(Bayesian inference)**: Bayes' 정리, likelihood, prior, posterior, marginalisation
- **태양 플라즈마 파라미터**: quiescent prominence 기준 T ~ 8000 K, n_H ~ 10¹⁰–10¹¹ cm⁻³, B ~ 5–20 G, 길이 L ~ 10⁵ km

**English**:
- **Ideal MHD equations**: continuity, momentum, induction, energy; linearisation of pressure, density, velocity, magnetic-field perturbations
- **MHD wave modes**: Alfvén speed $v_A = B/\sqrt{\mu_0 \rho}$, sound speed $c_s = \sqrt{\gamma p / \rho}$, fast and slow magnetoacoustic modes, tube/cusp speed $c_T = v_A c_s / \sqrt{v_A^2 + c_s^2}$
- **Guided MHD waves**: kink and sausage modes in cylindrical/slab flux tubes, internal vs. external modes, thin-tube (thin-flux-tube) approximation
- **Dispersion relations**: solving transcendental eigenvalue problems with tied/leaky boundary conditions
- **Resonant absorption**: mode conversion at Alfvén/cusp continuum resonances in an inhomogeneous transitional layer
- **Bayesian inference**: Bayes' theorem, likelihoods, priors, posteriors, marginalisation
- **Quiescent prominence parameters**: T ~ 8000 K, n_H ~ 10¹⁰–10¹¹ cm⁻³, B ~ 5–20 G (nearly horizontal), thread width ~ 0.3″ (~210 km), length ~ 5–40″ (3500–28,000 km)

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| Prominence / Filament | 차갑고 밀도가 큰 플라즈마가 자기장 구조에 매달려 있는 코로나 천체. 림에서 밝게 보이면 prominence, 디스크에서 어둡게 보이면 filament. / Cool, dense plasma suspended in the corona by magnetic structures; bright above the limb (prominence), dark against the disk (filament). |
| Large Amplitude Oscillation (LAO) | v > 20 km/s, 플레어·EUV 파·제트에 의해 트리거, 프로미넌스 전체가 흔들림. / Global oscillation with v > 20 km/s triggered by flares/EUV waves/jets. |
| Small Amplitude Oscillation (SAO) | v < 3 km/s (일반적으로 < 10), 국소적·반복적·플레어 무관. / Local, flare-unrelated oscillations with v < 3 km/s. |
| Kink mode | 플럭스 튜브의 비대칭 변위 모드, 가로 방향 운동(transverse). / Transverse m=1 mode that displaces the flux-tube axis. |
| Sausage mode | 대칭 팽창·수축 모드(axisymmetric, m=0). / Axisymmetric compressional m=0 mode. |
| Fast / Slow / Alfvén mode | 이상 MHD 세 모드. 고속(횡방향 압축), 저속(자기장선 따라 종방향), 알펜(횡방향 비압축). / The three ideal MHD modes: fast (transverse compressive), slow (longitudinal along B), Alfvén (transverse incompressible). |
| Resonant absorption | Alfvén/cusp continuum에서 global 모드 에너지가 국소 Alfvén 진동으로 변환되어 감쇠. / Mode conversion at the Alfvén/cusp resonance layer, producing damping of global kink modes. |
| Pendulum model | Luna & Karpen 2012: 자기장선 dip에서 중력을 복원력으로 하는 longitudinal 진동, $P = 2\pi\sqrt{R/g}$. / Longitudinal oscillation with gravity in magnetic dip as restoring force. |
| Thread | 프로미넌스 fine structure (폭 ~ 210 km, 길이 ~ 3500–28000 km), flux tube로 모델링됨. / Fine filamentary sub-structure of a prominence modelled as a flux tube. |
| Winking filament | Doppler 쉬프트로 H-alpha filter 대역을 벗어났다 들어왔다 하는 대진폭 vertical 진동. / Prominence whose H-alpha signal periodically vanishes due to large Doppler oscillation. |
| Prominence seismology | 관측 진동 특성(P, τ, λ, c_ph) + MHD 이론 → B, v_A, l/a 등 추론. / Inversion of observed oscillation properties through MHD models to infer physical parameters. |
| Bayesian seismology | Bayes' 정리 기반 확률적 파라미터 추론·모델 비교. / Probabilistic parameter inference and model comparison via Bayes' theorem. |

---

## 5. 수식 미리보기 / Equations Preview

### (i) Kink mode period for an infinitely long cylindrical thread / 무한 원통 thread의 kink 주기

$$
P_k = \frac{2L}{c_k}, \qquad c_k = \sqrt{\frac{2 B^2}{\mu_0 (\rho_i + \rho_e)}} = v_{A,i}\sqrt{\frac{2\zeta}{1+\zeta}}
$$

**한국어**: $L$은 flux tube 길이(반파장 기본 kink 모드), $\rho_i, \rho_e$는 내부(프로미넌스)·외부(코로나) 밀도, $\zeta = \rho_i/\rho_e$. 고밀도 대비에서는 $c_k \approx \sqrt{2}\, v_{A,i}$.
**English**: $L$ is flux-tube length (fundamental kink is half-wavelength); $\rho_i, \rho_e$ are internal/external densities; in the high-density-contrast limit $c_k \approx \sqrt{2}\,v_{A,i}$.

### (ii) Pendulum period for longitudinal oscillations / 종방향 진동 진자 주기

$$
P_{\text{long}} = 2\pi \sqrt{\frac{R}{g_\odot}}
$$

**한국어**: $R$은 자기장선 dip의 곡률반경. 중력이 복원력(Luna & Karpen 2012). $g_\odot = 274$ m/s².
**English**: $R$ is the curvature radius of the magnetic dip, $g_\odot = 274$ m/s² is solar gravity (Luna & Karpen 2012).

### (iii) Seismology inversion for magnetic field / 자기장 세기 seismology 역산

$$
B = \frac{L}{P}\sqrt{2 \mu_0 \rho_0 \left(1 + \frac{\rho_e}{\rho_0}\right)}
$$

**한국어**: 관측된 $P$, $L$, 추정 $\rho_0, \rho_e$에서 $B$를 역산 (Nakariakov & Verwichte 2005 공식 적용, Liu et al. 2012; Xue et al. 2014).
**English**: Inversion formula from the standing kink-mode period (Nakariakov & Verwichte 2005), applied e.g. by Liu et al. (2012) and Xue et al. (2014).

### (iv) Bayes' theorem for seismology / 지진학의 Bayes' 정리

$$
p(\boldsymbol{\theta}|D,M) = \frac{p(D|\boldsymbol{\theta},M)\, p(\boldsymbol{\theta},M)}{\int p(D|\boldsymbol{\theta},M)\, p(\boldsymbol{\theta},M)\, d\boldsymbol{\theta}}
$$

**한국어**: $\boldsymbol{\theta}$는 파라미터 벡터($v_A, l/a, \zeta$ 등), $D$는 관측 데이터($P, \tau_d$).
**English**: $\boldsymbol{\theta}$ is the parameter vector ($v_A$, $l/a$, $\zeta$ …), $D$ the observed data ($P$, $\tau_d$).

### (v) Resonantly-damped kink period and damping ratio / 공명 감쇠 kink 모드

$$
P \approx \frac{\sqrt{2}}{2}\frac{\lambda}{v_{A,i}}, \qquad \frac{\tau_d}{P} \approx \frac{2}{\pi}\frac{R}{l}\frac{\rho_i + \rho_e}{\rho_i - \rho_e}
$$

**한국어**: $l$은 가로 비균일층 두께, $R$은 thread 반경. Goossens et al. (2008) TT-TB 근사.
**English**: $l$ is the transverse inhomogeneity layer width, $R$ the thread radius; thin-tube/thin-boundary (TT-TB) approximation of Goossens et al. (2008).

---

## 6. 읽기 가이드 / Reading Guide

**한국어**:
1. **Sect. 1–2 (Introduction, Classification)**: 빠르게 읽고 LAO/SAO 구분을 명확히 한다.
2. **Sect. 3, 4 (LAO 관측·이론)**: vertical/transverse/longitudinal 분류, Hyder·Kleczek–Kuperus·pendulum model을 식 (1)–(7)과 함께 정독.
3. **Sect. 5 (SAO 관측)**: 주기 분포(단·중·장), 진폭, 편광, 감쇠 τ/P ~ 1–4의 정량 결과에 주목.
4. **Sect. 6 (SAO 이론)**: **6.1 loaded string → 6.2 slab → 6.3 thread (propagating) → 6.4 thread (standing)** 순으로 복잡도 상승. 식 (12) $P = 2\pi\sqrt{Lx_p}/c_{\text{pro}}$, 식 (30)–(31) kink 식을 반드시 이해.
5. **Sect. 7 (감쇠 메커니즘)**: 비단열 열적 감쇠(slow 모드만 효과), 이온-중성자 충돌, 공명흡수(fast/kink 감쇠), 파동 누출. τ_d/P ~ 1–4 관측과 비교.
6. **Sect. 8 (Seismology)**: 8.1 LAO → 8.3 propagating thread → 8.4 damped thread → 8.5 period ratio → 8.6 flowing → **8.7 Bayesian** 순. 식 (44)–(55) 역산 공식 중점.
7. **Sect. 9 (Open issues)**: 미래 연구 방향 점검.

**English**:
1. **Sect. 1–2**: read quickly; anchor LAO vs. SAO classification.
2. **Sect. 3–4**: vertical/transverse/longitudinal LAOs and their harmonic-oscillator models, Eqs. (1)–(7), including pendulum $P = 2\pi\sqrt{R/g}$.
3. **Sect. 5**: observed periods (short < 10 min, intermediate 10–40, long 40–90 min, ultra-long hours), damping τ/P ~ 1–4, polarisation, spatial distribution.
4. **Sect. 6**: build complexity **loaded string → slab → cylindrical thread**; master kink eqs. (30)–(31), dispersion relations (17)–(18).
5. **Sect. 7**: which mechanism damps which mode — thermal (slow), ion-neutral (fast/Alfvén possibly), resonant absorption (kink, most plausible for fast damping), wave leakage.
6. **Sect. 8**: inversion formulas — Eq. (44) for $B$ from kink $P$; Eq. (48) $P = P(k_z, \zeta, l/a, v_{A,i})$; Bayesian approach in 8.7.
7. **Sect. 9**: open issues — driver of SAOs, 3D modelling, partial ionisation, coupled LAO/SAO.

---

## 7. 현대적 의의 / Modern Significance

**한국어**: 프로미넌스 진동 연구는 (i) **우주기상** 측면에서 코로나질량방출(CME)로 발전할 수 있는 필라멘트의 불안정성 진단 도구이며, (ii) **코로나 자기장 측정**의 대표적 간접 기법으로 Zeeman/Hanle 분광과 상보적이고, (iii) **플라즈마 다이아그노스틱스** 관점에서 밀도 구조·비균일 스케일·부분 이온화 정도를 제한한다. 최근 **DKIST (Daniel K. Inouye Solar Telescope)**, **Solar Orbiter**, **ASO-S/CHASE** 등 고해상도 미션이 thread 스케일 진동과 감쇠를 더 정밀히 관측함으로써 **Bayesian seismology**의 정확도를 크게 향상시킬 것이다. 이 리뷰는 특히 resonant absorption 기반 damping 분석과 Bayesian inversion을 coronal loop seismology와 연결시켜, **태양 대기 전역의 seismology 프레임**을 구축하는 데 기초가 된다.

**English**: Prominence-oscillation research matters today for (i) **space weather**, as oscillations can diagnose filament (in)stability before eruption and CME onset; (ii) **coronal magnetometry**, providing an indirect probe complementary to Hanle/Zeeman spectropolarimetry of cool prominence plasma; (iii) **plasma diagnostics**, constraining transverse density structuring, inhomogeneity scales, mass flows, and the degree of partial ionisation. Facilities such as **DKIST**, **Solar Orbiter**, **ASO-S/CHASE**, and next-generation ALMA observations will resolve thread-scale oscillations and damping with unprecedented accuracy, sharpening **Bayesian seismology** inversions. The review's unified treatment of resonant absorption damping and Bayesian inversion has bridged prominence seismology with the older, more mature coronal-loop seismology, building toward a **solar-atmospheric seismology framework** that spans chromosphere through corona.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
