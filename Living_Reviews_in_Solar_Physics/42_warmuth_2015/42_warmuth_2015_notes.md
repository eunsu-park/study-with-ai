---
title: "Large-scale Globally Propagating Coronal Waves"
authors: Alexander Warmuth
year: 2015
journal: "Living Reviews in Solar Physics"
doi: "10.1007/lrsp-2015-3"
topic: Living_Reviews_in_Solar_Physics
tags: [coronal_waves, MHD, fast_mode, Moreton_wave, EIT_wave, shock, CME, coronal_seismology]
status: completed
date_started: 2026-04-23
date_completed: 2026-04-23
---

# 42. Large-scale Globally Propagating Coronal Waves / 대규모 전구적 전파 코로나 파동

---

## 1. Core Contribution / 핵심 기여

Warmuth (2015)는 1960년대 Hα Moreton wave 발견부터 2015년 SDO/AIA 시대까지 약 반세기 동안 축적된 대규모 코로나 파동 관측, 이론, 수치 모델링을 종합 리뷰한다. 핵심 기여는 두 가지이다. 첫째, 관측적 증거(다파장 영상, 운동학, 동요 프로파일, Mach 수, 열적 응답, 코로나 구조와의 상호작용)가 어떻게 경쟁하는 물리 해석 — "fast-mode MHD wave/shock" 대 "magnetic reconfiguration (pseudo-wave)" — 을 제약하는지 체계적으로 정리한다. 둘째, 15년 이상 이어진 이 논쟁이 단일 모델이 아니라 **통합 하이브리드 시나리오**로 수렴하고 있음을 보인다: **빠르고 외곽에 있는 fast-mode MHD wave 또는 shock**와 **느리고 내부에 위치한 CME 팽창에 의한 자기 재구성 밝기**가 동시에 관측되며, 사건의 특성(eruption 속도, 주변 corona 구조)에 따라 어느 쪽이 더 두드러지는지가 결정된다. Warmuth–Mann (2011)의 세 가지 kinematical class (Class 1 nonlinear wave/shock, Class 2 linear/weakly nonlinear wave, Class 3 pseudo-wave)는 이 시나리오를 운동학적으로 뒷받침한다.

Warmuth (2015) is a comprehensive Living Reviews article synthesizing five decades of observational, theoretical, and numerical work on large-scale globally propagating coronal waves — from the 1960 Hα Moreton-wave discovery through the revolutionary 1997 SOHO/EIT detections and into the high-cadence SDO/AIA era. Two central contributions stand out. First, the paper systematically catalogs how each observational constraint — multi-wavelength morphology, kinematics, perturbation-profile evolution, fast-mode Mach numbers, thermal response, and interactions with coronal structures (refraction, reflection, transmission, stationary brightenings) — discriminates between the competing "fast-mode MHD wave/shock" and "magnetic reconfiguration / pseudo-wave" interpretations. Second, Warmuth shows that the decade-and-a-half controversy has converged onto a **unified hybrid scenario**: a fast outer fast-mode MHD wave (or shock, when the driver is sufficiently impulsive) accompanied by a slower inner brightening caused by the lateral expansion of an erupting CME. Which component dominates depends on the eruption's impulsiveness and the ambient corona, providing a natural explanation for the three kinematical event classes of Warmuth & Mann (2011): Class 1 nonlinear waves/shocks, Class 2 linear or weakly nonlinear waves, and Class 3 pseudo-waves.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction & Physical Concepts / 서론과 물리 개념 (Sec. 1–2, pp. 5–10)

**개요 / Overview.** 코로나는 자화된 플라즈마이며 전형적 온도는 1–2 MK, 자기장은 quiet corona에서 수 Gauss, AR에서 수 kG. Eruptive event(flare, CME, filament eruption)는 최대 $10^{33}$ erg, 1000 km/s 이상의 물질 분출, 상대론적 입자 가속을 수반한다. 이런 강력한 에너지 방출이 주변 corona에 파동·충격·재구성 신호를 남기는 것은 자연스러운 기대였다.

**Three linear MHD modes.** 선형 MHD는 세 모드를 허용한다:

$$v_A = \frac{B}{\sqrt{4\pi\rho}}, \quad c_s = \sqrt{\frac{\gamma_\mathrm{ad} k T}{\bar{\mu} m_p}}$$

Alfvén mode(횡파, 비압축, 자기 tension만 복원)은 자기장선 방향으로만 전파. Fast / slow-mode는 자기 + 기체 압력 모두 복원력:

$$v_{f/s} = \left\{\frac{1}{2}\left[v_A^2 + c_s^2 \pm \sqrt{(v_A^2 + c_s^2)^2 - 4 v_A^2 c_s^2 \cos^2\theta_B}\right]\right\}^{1/2}$$

관측된 코로나 파동은 압축성이며 자기장선을 가로질러 큰 거리까지 전파하므로 Alfvén은 제외된다. 일반적으로 **fast-mode**로 해석한다. 수직 전파에서 $v_{ms} = \sqrt{v_A^2 + c_s^2}$ (magnetosonic speed), 슬로우 모드는 $\theta_B = 90^\circ$에서 소멸한다.

**Nonlinear fast-mode waves and shocks.** 대진폭 교란은 비선형 MHD로 다뤄야 한다. 파동 crest가 leading edge보다 빨리 전파해 profile이 가파르게 steepening 되고, 결국 shock가 형성된다(Fig. 1). Shock는 엔트로피 도약이 있는 discontinuity. Fast-mode 충격의 Rankine–Hugoniot 관계에서 수직 fast-mode Mach 수:

$$M_\mathrm{ms} = \sqrt{\frac{(5\beta_p + 5 + X) X}{(2 + \gamma \beta_p)(4 - X)}}$$

$X = n_d/n_u$는 압축비. 두 가지 shock 형성 경로: **piston-driven shock** ($v_\mathrm{piston} \le v_\mathrm{shock}$, 예: 폭발적 CME 팽창)과 **bow shock** ($v_\mathrm{piston} = v_\mathrm{shock}$, supermagnetosonic driver 필요).

**Pseudo-wave 개념.** MHD 파동이 아닌 자기 재구성(field-line stretching, current shell Joule heating, reconnection front)이 전파성 밝기처럼 보이게 할 수 있다. 세 가지 pseudo-wave 모델은 Sec. 6에서 자세히 다뤄진다.

### Part II: Observational Signatures / 관측 신호 (Sec. 3, pp. 11–29)

**Table 1 관측 기기**. EUV: SOHO/EIT (720 s cadence), TRACE (30 s), STEREO/EUVI (150 s), PROBA2/SWAP (60 s), SDO/AIA (12 s, 7 channels). SXR: Yohkoh/SXT, GOES/SXI, Hinode/XRT. Hα: Kanzelhöhe 등 GBOs. He I 10830 Å, radio (NRH, NoRH). Coronagraph overlap(C1, COR1, MK3/MK4)은 low corona 관측에 필수.

**EUV waves (3.1.1).** EIT 195 Å(Fe XII, 1.4 MK)에서 처음 관측된 "EIT waves": 전형적 수 100 km/s, emission 증가 < 25%, 100 Mm 스케일 angular extent는 전체 반구. 다양한 morphology (Fig. 5): (a) 1997 May 12 — diffuse global; (b) 1997 Sep 24 — "S-wave" (sharp); (c) 1998 Jun 13 — small irregular; (d) 2010 Jan 17 — dome-shaped. S-waves (7%)은 sharp, 처음엔 좁고(≈ 20 Mm) 높은 진폭(>100%)인 뒤 diffuse로 decay. AIA, EUVI, SWAP의 고 cadence 덕분에 bimodality (fast outer + slow inner)가 확립되었다.

**SXR (3.1.2)** 신호는 ≥ 2 MK에 민감해 뜨거운 wave 구성요소만 본다. Yohkoh/SXT Narukage 2002 "SXT waves"는 강한 Moreton-associated 이벤트에서만 관측, 진폭 > 200%, $M_\mathrm{ms} \approx 1.1$–1.3. GOES/SXI는 2–4 min cadence로 6 event에서 EIT, Moreton, He I wavefront와 공간적으로 일치함을 확인(Warmuth et al. 2005).

**White light (3.1.3).** LASCO coronagraph에서 CME-driven fast-mode shock가 smooth leading edge로 관찰됨. Kwon et al. (2013b)이 STEREO/COR1에서 upper corona wavefront를 EUV wave의 상대물로 처음 확인.

**Moreton waves (3.2.1).** Hα line center에서 bright front, blue wing에서 bright + red wing에서 dark → Doppler 해석: chromosphere가 하향 compression. 전형 속도 1000 km/s (sound speed 10 km/s, Alfvén speed 100 km/s를 훨씬 초과). 1968 Uchida sweeping-skirt hypothesis 확립.

**He I 10830 Å (3.2.2).** EUV irradiation + collisional process에 의한 복잡한 선; wave signature가 dark front (증가 흡수)로 나타남. Moreton 신호보다 앞서거나 약간 다른 고도 signature 제공.

**Radio (3.3) & Type II bursts (3.4.1).** 대부분의 Moreton-associated wave는 metric type II burst와 연관: 느린 주파수 drift는 코로나 shock의 외곽 전파를 시사. Type II source의 height와 EIT wavefront 위치 비교로 shock 진단 가능. Coronal dimmings(3.4.2)은 CME 질량 이탈을 반영, wave와 함께 관찰.

### Part III: Physical Characteristics / 물리적 특성 (Sec. 4, pp. 30–55)

**4.1 Spatial characteristics.** Radiant point (extrapolated source) 기반 distance-time. 평균 angular extent 대개 반구 이상, 일부 360° 도달. Dome 구조를 가진 limb event에서 radial speed 1.3–2.3 × lateral speed (Sec. 4.2.4).

**4.2 Kinematics — 가장 중요한 부분.**

**4.2.1 Mean velocities (Fig. 22).**
- **Moreton waves** (Hα): 400 – 1500 km/s, 평균 ~650 km/s (Smith & Harvey 1971; Warmuth et al. 2004b; Zhang et al. 2011).
- **EIT waves** (SOHO/EIT 195 Å): 10 – 700 km/s, 평균 ~191 km/s (Thompson & Myers 2009, $N = 123$).
- **EUVI waves**: 평균 ~300 km/s (Nitta et al. 2014, $N = 34$); 250 km/s (Muhr et al. 2014, $N = 60$).
- **AIA waves** (193 Å): 200 – 1500 km/s, 평균 ~644 km/s (Nitta et al. 2013, $N = 138$) — SOHO/EIT보다 고 cadence가 빠른 이벤트를 잡아냄.
- **Moreton-associated EIT waves**: ~300 km/s (더 빠른 Moreton 해당 코로나 파동은 deceleration 후 측정).

**4.2.2 Deceleration (Fig. 23).** 모든 기록된 Moreton waves는 감속: 초기 600–1200 km/s → ≈500 km/s로 Hα 소실 전까지 감속. 평균 deceleration ~1 km/s². 거리-시간 curve는 2차 다항식보다 **power-law $d \sim t^\delta$** 로 더 잘 적합, 평균 $\delta \approx 0.6$ (Sedov blast-wave solution과 일치). 결합된 Hα+EIT+SXI wavefront의 deceleration은 $-0.2$ km/s² (약해지나 같은 감속 흐름).

**4.2.3 Kinematical classification (Warmuth & Mann 2011; Fig. 25).**
- **Class 1**: 초기 속도 ≥ 320 km/s, 강한 감속(최대 $-2$ km/s²). → nonlinear fast-mode wave/shock.
- **Class 2**: 170 – 320 km/s, 거의 constant. → linear 또는 약한 nonlinear fast-mode wave.
- **Class 3**: 낮은 초기 속도(수 10 km/s)와 erratic kinematics. → pseudo-wave (magnetic reconfiguration).

Class 1과 2 사이는 연속적이고, Class 3는 뚜렷이 분리. Nitta et al. (2013) AIA 데이터는 더 부드러운 분포를 보고(선택 기준 차이) 했으나 Warmuth–Mann 시나리오와는 여전히 정합.

**4.2.4 Lateral vs radial kinematics.** Dome 구조에서 radial (top) / lateral (flank) = 1.3 – 2.3. (i) fast-mode speed의 고도 증가 + (ii) upper part가 CME로 지속 driven되는 효과.

**4.3 Perturbation profile.** $I(r)$을 Gaussian fit, FWHM = pulse width. 전형 EIT width 100 Mm, 진폭 < 25%. S-wave: sharp 초기 (20 Mm, > 100%) → decay. Moreton-associated wave: width 40 → 150 Mm, 진폭 100% → < 10%. Pulse **broadening과 amplitude decrease**가 freely propagating nonlinear wave의 전형적 특성. Amplitude × width 적분은 대체로 보존(에너지 보존). 진폭을 compression ratio로 변환: $X \sim (I/I_0)^{1/2}$. EIT wave 대부분 $X \le 1.1$ (linear 혹은 매우 약한 nonlinear); S-waves 초기 $X \approx 1.5$ (강한 nonlinear).

**4.4 Mach numbers.** $M_\mathrm{ms}$은 $X$와 $\beta_p, \gamma$로 계산. EUV wave 대부분 $M_\mathrm{ms} \le 1.04$ (linear wave와 일치). 초기의 빠르고 Moreton-associated는 $M_\mathrm{ms} \ge 1.15$ (nonlinear wave/shock). S-wave $M_\mathrm{ms} \ge 1.3$. SXR amplitude > 200%는 $M_\mathrm{ms} \approx 1.3$–1.4. Moreton wave는 독립적인 type II band-split에서도 $M_\mathrm{ms} = 1.9$–2.2 → 강한 fast-mode shock 확인.

**4.5 Thermal characteristics.** EIT wave는 multithermal (Long et al. 2008 — 171, 195, 284 Å 모두). AIA는 7 EUV 채널: 304(0.05), 131(0.4), 171(0.6), 193(1.3 · 1.6), 211(2.0), 335(2.5), 94(6.3) MK. Wave는 193, 211 Å에서 가장 밝고 171 Å에서 약화/감쇠(⇒ 0.8–1.6 MK에서 plasma heating; DEM shift). Schrijver et al. (2011): $X = 1.1$, $\Delta T / T \approx 7$%. Ma et al. (2011): $X = 1.56$에서 $T$ 최대 2.8 MK.

**4.6 Flows.** Hinode/EIS: wave 앞쪽 downward flow(redshift 20 km/s), 뒤쪽 outflow(blueshift). Veronig et al. (2011): "sweeping skirt"와 정합. 약한 wave는 compression 작아 flow signature 없음.

**4.7 Interaction with coronal structures.**

- **Refraction (4.7.1).** Wave는 AR과 coronal hole (높은 $v_\mathrm{ms}$ ~1000 km/s)을 피한다. $c_f$ gradient에 따라 low speed valley로 휨. Mann, Uchida, Afanasyev & Uralov, Wang 등 다수 시뮬레이션이 재현.

- **Reflection (4.7.2).** Coronal hole 경계는 가파른 $v_\mathrm{ms}$ gradient ⇒ 반사. Long et al. (2008), Veronig et al. (2008), Gopalswamy et al. (2009) 첫 명확 관측. Huygens–Fresnel 원리(입사각 = 반사각) 따름 (Kienreich et al. 2013). 반사파 속도(100–500 km/s)는 입사파(300–800 km/s)보다 느림 ⇒ nonlinear wave 해석 또는 flow field frame 효과. **반사는 진짜 wave 해석의 강력한 증거.**

- **Transmission (4.7.3).** AR, coronal cavity, coronal hole 경계를 일부 에너지가 통과. 투과파 속도가 더 빠르고(10–60% 높음) 진폭이 작아짐(에너지 보존 + 높은 $c_f$). Topological separatrix 교차는 non-wave 모델을 반증함.

- **Stationary brightenings (4.7.4).** Separatrix에서 파동 정지가 아니라 wave 통과로 trigger되는 국소적 에너지 방출로 해석.

- **Oscillations / sympathetic eruption (4.7.5–6).** Filament "winking"과 loop oscillation 유발, 코로나 구조 재밸런싱, 원격 CME trigger 가능.

**4.8 Energetics.** Fast-mode wave에너지 $\sim 10^{30}$ erg order — total flare/CME energy의 작은 일부.

**ASCII schematic of a dome-like coronal wave.**

```
                    CME flank / dome apex
                      (upward radial)
                           ^
                           |  v_r ~ 700 km/s
                          /|\
                         / | \
                        /  |  \
               ________/___|___\________ ← SXR/EUV wavefront (dome surface)
              /        |   |   |        \
             /         |   |   |         \
            /   AR     |   |   |   quiet corona
           /___________|___|___|___________\
          |  dimming   |   |   |   refracted wave bends away
          |____________| Ch |_____|   from AR (high v_ms)
                       |    |
                       |    |___ chromospheric Moreton wave
                       |_______ (ground track of flanks)
                           ←  lateral v ~ 300 km/s →

     Key:
       AR   = Active Region (high v_A, high v_ms)
       Ch   = Chromosphere
       Flanks of dome impact chromosphere → Moreton signature
       Top of dome rises radially 1.3–2.3× faster than lateral flanks
```

This cartoon (adapted from Warmuth's Sec. 4.1.3 and Fig. 42) captures three key observations in one diagram: (i) the 3D dome structure with faster radial apex and slower lateral flanks, (ii) the Moreton wave as the chromospheric footprint of the descending dome flank, and (iii) refraction of the wave away from the high-$v_\mathrm{ms}$ active region.

### Part IV: Relationship with Eruptive Events / 분출 이벤트와의 관계 (Sec. 5, pp. 55–64)

Flare vs CME 어느 쪽이 driver? 통계적으로 **CME 발생이 더 밀접한 상관**: CME의 lateral expansion이 3D piston으로 작용. Cospatial-cotemporal 분석에서 EUV wave front와 CME flank가 일치하는 사건 다수. Flare pressure pulse만으로는 일부 Moreton 이벤트의 impulsive 특성을 설명하기 어렵다는 반례도 있음. Small-scale ejecta (jets, surges)도 약한 wave 생성 가능.

### Part V: Physical Interpretation and Models / 물리적 해석과 모델 (Sec. 6, pp. 65–75)

**6.1.1 Fast-mode wave/shock model (Uchida 1968).**
Quiet corona 정전 값: $n_e = 5 \times 10^8$ cm$^{-3}$, $T = 1.5$ MK, $B = 3$ G $\Rightarrow c_s = 185$, $v_A = 273$, $v_\mathrm{ms} = 330$ km/s. 관측된 EIT wave 속도 범위와 잘 맞음. Chromosphere에서 $c_s \approx 10$, $v_A \approx 100$ km/s — Moreton 속도 1000 km/s이면 $M \ge 10$로 강하게 damped, 따라서 Uchida의 sweeping-skirt는 필수. 모델 가정:
1. Initial 교란이 AR에서 CME bubble 팽창 혹은 flare pressure pulse로 생성.
2. 작은 진폭이면 linear/weakly nonlinear wave, $v_\mathrm{ms}$로 isotropic 전파 — refraction, reflection, transmission 관측.
3. 큰 진폭이면 nonlinear steepening → shock, $M_\mathrm{ms} > 1$, amplitude decrease와 deceleration → 결국 linear wave로 degenerate.

**6.1.2 Slow-mode soliton model (Wills-Davey 2007).** Soliton의 dispersive balance로 coherence 유지. 그러나 slow-mode는 $\theta_B = 90^\circ$에서 소멸해 수직 전파 필요; Hall-MHD 스케일은 10 m 수준(비현실적); profile broadening이 소리톤 가정과 상충 → 대부분의 관측과 불일치.

**6.1.3 Magnetoacoustic surface gravity waves (Ballai 2011a).** Transition region/corona 경계의 spherical interface 파동. Full wave dome 관측과 불일치.

**6.2.1 Field line stretching model (Chen et al. 2002, 2005b).** 상승하는 flux rope가 overlying field line을 순차적으로 stretch ⇒ footpoint에서 compression 밝기. 속도 ~250 km/s (EIT wave와 일치), separatrix에서 정지. 문제: 2D, 관측된 quasi-circular wavefront 설명 곤란; density enhancement 몇 % → 진폭 부족; large-arcade 이벤트에서만.

**6.2.2 Current shell model (Delannée et al. 2008).** 3D MHD: erupting flux rope 주변에 current shell 형성, Joule heating으로 밝기. 그러나 current 강도는 300 Mm 고도에서 최대 ⇒ 관측된 낮은 고도(50–100 Mm)와 상충. Wavefront가 driver acceleration을 따라야 하나 관측은 deceleration.

**6.2.3 Reconnection front model (Attrill et al. 2007).** Lateral 팽창하는 CME flux rope가 quiet-sun loop와 순차 reconnection ⇒ plasma heating brightening. Delannée (2009) 지적: magnetic field extrapolation에서 small-scale reconnection은 일어나기 어려움. Transition region 신호 결여.

**6.3 Hybrid models (Chen et al. 2002; Downs et al. 2011, 2012; Cohen et al. 2009; Schrijver et al. 2011; Wang et al. 2009; Pomoell et al. 2008).** Fast outer front + slow inner front. Downs et al. (2012)의 3D thermodynamic MHD는 AIA synthesis로 두 front 모두 재현. Inner front의 기원은 CME 팽창 compression이 지배적이며 current shell이나 slow-mode는 부수적.

**6.4 Unified scenario.** Fast-mode wave(또는 shock, impulsive driver)가 **관측되는 주된 wavefront**; 자기 재구성이 slower inner bright를 만든다. Class 1/2/3 구분은 driver impulsiveness의 스펙트럼을 반영. 이로써 "wave vs non-wave" 양립 가능.

### Part VI: Wider Significance / 확장된 의의 (Sec. 7, pp. 76–79)

**7.1 Global coronal seismology.** $v_\mathrm{ms}$ 측정 ⇒ (corona density model) ⇒ $B$ 추정. Quiet low corona에서 $B_\mathrm{cor} \approx 0.4$–6 G, 평균 수 Gauss — photospheric mean field와 정합. Yang & Chen (2010)이 음의 correlation으로 pseudo-wave 주장했으나 Zhao et al. (2011)은 large-distance AIA에서 양의 correlation(fast-mode와 일치)을 확인.

**7.2 SEPs.** Coronal wave-driven shock가 wide longitudinal SEP spread 설명 후보. Miteva et al. (2014): 179 SEP 중 87%가 EIT wave 동반. Refracting shock의 downstream에 관찰자가 연결되면 diffusive shock acceleration로 power-law spectrum 생성. Near-relativistic electron 일부는 이 시나리오와 불일치 ⇒ cross-field diffusion 필요.

### Part VII: Conclusions and Open Issues / 결론과 미해결 과제 (Sec. 8, p. 80)

Fast-mode wave/shock + pseudo-wave hybrid가 **standard model**. 미해결: (1) multiwavelength integration (X-ray, radio, optical); (2) 3D MHD를 다양한 event class에 적용; (3) flare vs CME driver의 명확한 구분; (4) large-amplitude wave/shock 이론 추가 개발; (5) SEP wave association의 명확한 증명; (6) 공개된 이벤트 catalog.

The hybrid fast-mode-wave + pseudo-wave model is now the accepted standard. Open issues enumerated in Section 8 include: (1) deeper integration of EUV with X-ray, radio, and optical data for Moreton-associated events; (2) applying 3D MHD simulations to the full spectrum of event classes; (3) clearly discriminating flare pressure-pulse from CME lateral-expansion drivers; (4) further analytic and numerical development of large-amplitude wave/shock theory, particularly wave–chromosphere coupling to produce Moreton signatures; (5) unambiguously establishing SEP association; (6) producing publicly available coronal-wave event catalogs comparable to existing CME catalogs.

### Numerical example: a full decelerating Moreton-associated event / 수치 예시: 감속하는 Moreton 연관 사건

Consider a representative Moreton + EIT event following Warmuth et al. (2004a,b). Initial coronal fast-mode speed $v_0 = 1000$ km/s at $t = 0$. Assume power-law deceleration $v(t) = v_0 (1 + t/\tau)^{\delta - 1}$ with $\delta = 0.6$ and $\tau = 300$ s.

| $t$ (s) | $v$ (km/s) | $d$ (Mm) | Observational trace |
|---------|-----------|----------|---------------------|
| 0 | 1000 | 0 | Launch at flare onset; Hα Moreton visible |
| 60 | 893 | 56.5 | Sharp Moreton front; SXR amplitude 200% ($M_\mathrm{ms} \approx 1.3$) |
| 300 | 660 | 231 | Moreton front fades (distance > 200 Mm) |
| 600 | 527 | 410 | Only EUV wavefront remains; $M_\mathrm{ms} \to 1.1$ |
| 1200 | 405 | 690 | Low-cadence EIT first captures wave here, reports "191 km/s" |
| 2400 | 310 | 1120 | Wave reaches coronal hole, reflects (Huygens–Fresnel) |

The observer at 12-minute EIT cadence therefore samples only the late, slow phase and naturally recovers a low mean speed ~300 km/s, while AIA at 12-s cadence would trace the full decelerating curve and extract an initial speed ≥ 1000 km/s. This simple kinematic exercise resolves much of the historical Moreton–EIT speed controversy.

**따로 떼어낸 예시로서**, 이 감속 곡선은 (i) 세 channel (Hα, SXR, EUV)의 cospatial-but-offset 관측, (ii) Moreton wave가 200–300 Mm 이후 소실되는 경험적 사실, (iii) EIT wave의 평균 속도가 Moreton 평균 속도의 약 1/3인 이유를 통합적으로 설명한다.

### On the "standalone" utility of these notes / 독립 이해 가능성에 관한 메모

이 notes.md를 원 논문 없이 읽는 독자는 다음을 습득할 수 있도록 설계되었다: (1) MHD 선형 파동 세 모드와 nonlinear steepening → shock 경로; (2) EUV/SXR/Hα/He I/radio 채널이 각기 다른 온도와 고도 진단을 제공한다는 사실; (3) Moreton–EIT 속도 불일치가 deceleration으로 해결되는 논증; (4) 세 kinematical class(Warmuth & Mann 2011)의 의미; (5) refraction/reflection/transmission이 wave interpretation을 확증하는 이유; (6) CME 주도 hybrid 시나리오의 수식적·관측적 근거; (7) coronal seismology로 quiet-Sun $B$ field ~3 G를 추정하는 원리; (8) wave-driven shock가 SEP 광범위 spreading을 설명하는 역할. 각 주제에 대해 수치 예시와 LaTeX 수식이 동반된다.

These notes have been written to pass the "standalone test": a reader who has never opened Warmuth (2015) should nevertheless be able to (1) sketch the three linear MHD modes and the nonlinear steepening → shock pathway, (2) explain what each wavelength channel (EUV, SXR, Hα, He I, radio) diagnoses, (3) reconstruct the Moreton–EIT velocity-discrepancy argument through deceleration, (4) name and justify the three kinematical classes of Warmuth & Mann (2011), (5) articulate why refraction, reflection, and transmission support the wave interpretation, (6) state the hybrid CME-driven scenario and its mathematical/observational basis, (7) describe global coronal seismology as a magnetic-field diagnostic (~ 3 G quiet corona), and (8) explain the wave-driven-shock pathway for wide-longitude SEP events. Each major topic is supported by worked examples and LaTeX-rendered equations.

---

## 3. Key Takeaways / 핵심 시사점

1. **코로나 파동의 관측 채널은 서로 다른 물리적 깊이를 진단한다.** — Hα Moreton은 chromosphere에서의 downward 압축을 보이고, EUV/SXR은 corona 내부의 compressive heating을, white-light은 upper corona의 density enhancement를 진단한다. 따라서 하나의 파동이 동시다발적으로 여러 채널에서 "cospatial but offset" 신호를 남긴다는 사실 자체가 단일 underlying disturbance를 지지한다. / Different wavelengths probe different atmospheric depths of a single 3D disturbance, so cospatial multi-band signatures support a single underlying wave rather than independent phenomena.

2. **Moreton wave와 EIT wave의 속도 불일치는 deceleration으로 해결된다.** — Moreton 평균 ~650 km/s, EIT 평균 ~191 km/s는 "다른 현상"이라기보다 **감속하는 하나의 파동**을 각기 다른 시점에서 sampling한 결과다(EIT는 느린 cadence로 late phase만 포착). AIA 고 cadence 관측은 초기 > 1000 km/s를 실제로 기록한다. / The Moreton–EIT velocity mismatch is resolved by recognizing a single decelerating disturbance, with low-cadence EIT under-sampling the fast early phase.

3. **세 운동학적 class는 driver impulsiveness 스펙트럼을 반영한다.** — Class 1 (≥ 320 km/s, 강한 감속) = nonlinear wave/shock; Class 2 (170–320 km/s, 거의 constant) = linear/weakly nonlinear wave; Class 3 (수 10 km/s, erratic) = pseudo-wave. 이것은 Warmuth & Mann (2011)의 핵심 정리이다. / The three kinematical classes span a continuum from nonlinear wave/shock (Class 1) to linear wave (Class 2) to pseudo-wave (Class 3), reflecting driver impulsiveness.

4. **Refraction, reflection, transmission은 "진짜 wave" 해석의 결정적 증거.** — Non-wave pseudo-wave 모델은 topological separatrix에서 파동이 멈춰야 하지만, 관측은 coronal hole에서의 명확한 반사(Gopalswamy 2009, Kienreich 2013)와 AR을 통과하는 transmission을 보여준다. / Refraction, reflection at coronal hole boundaries (obeying Huygens–Fresnel), and transmission across separatrices are defining wave-like behaviors ruling out pure pseudo-wave interpretations.

5. **Mach 수와 perturbation profile의 정합은 fast-mode wave/shock 시나리오를 뒷받침한다.** — EIT wave 대부분 $X \le 1.1$, $M_\mathrm{ms} \le 1.04$(linear); S-wave와 Moreton-associated의 $M_\mathrm{ms} \ge 1.3$–2.2는 nonlinear shock와 정합. Type II radio burst의 독립적 Mach number 측정도 이를 확인. / Mach numbers derived independently from perturbation profiles and from type II radio bursts agree and bracket coronal waves into linear (Class 2) and shocked (Class 1) regimes.

6. **통합 하이브리드 시나리오가 "wave vs non-wave" 논쟁을 해소한다.** — Fast-mode wave/shock는 주된 외곽 bright front, CME 팽창에 의한 magnetic reconfiguration은 느린 내부 front. 3D thermodynamic MHD (Downs et al. 2012)가 AIA synthesis로 이를 직접 재현. / A unified hybrid scenario — outer fast-mode wave/shock + inner CME-driven pseudo-wave — accounts for the bimodal morphology and resolves the wave-vs-non-wave controversy.

7. **Global coronal seismology로 quiet corona의 $B$를 수 Gauss로 측정한다.** — 관측 $v_\mathrm{ms}$와 density model로부터 $B_\mathrm{cor} \approx 0.4$–6 G, 평균 약 3 G (Warmuth & Mann 2005a; Ballai 2007; Long et al. 2013). 이는 photospheric mean field와 정합하며 다른 방법으로는 측정이 매우 어려운 값이다. / Global coronal seismology based on wave kinematics yields quiet-coronal magnetic field strengths of a few Gauss, consistent with photospheric mean fields and otherwise hard to measure.

8. **Coronal wave-driven shock는 SEP의 광범위 경도 분포를 설명하는 유력 메커니즘.** — Miteva et al. (2014): 179 SEP 중 87%가 EIT wave 동반; refracting shock downstream에서의 diffusive shock acceleration이 power-law 스펙트럼 생성 가능. / Wave-driven refracting shocks provide a natural mechanism for producing SEPs over a wide range of heliolongitudes, a known challenge for purely flare-driven scenarios.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Sound speed

$$c_s = \sqrt{\frac{\gamma_\mathrm{ad} k T}{\bar{\mu} m_p}}$$

$\gamma_\mathrm{ad} = 5/3$ (monatomic fully ionized plasma), $\bar{\mu} = 0.6$ (mean molecular weight, Priest 1982), $k$ = Boltzmann constant, $T$ = temperature, $m_p$ = proton mass. **Worked example (quiet corona, $T = 1.5$ MK):** $c_s = \sqrt{(5/3)(1.38 \times 10^{-16})(1.5 \times 10^6) / (0.6 \times 1.67 \times 10^{-24})} \approx 1.85 \times 10^7$ cm/s ≈ **185 km/s**.

### 4.2 Alfvén speed

$$v_A = \frac{B}{\sqrt{4\pi \rho}} = \frac{B}{\sqrt{4\pi \bar{\mu} m_p n}}$$

$n = 1.92 n_e$ (Priest 1982, 전 입자 밀도). **Worked example (quiet corona, $B = 3$ G, $n_e = 5 \times 10^8$ cm$^{-3}$):** $\rho = \bar{\mu} m_p n = 0.6 \times 1.67\times10^{-24} \times 1.92 \times 5\times10^8 \approx 9.6\times10^{-16}$ g/cm³; $v_A = 3/\sqrt{4\pi \cdot 9.6\times10^{-16}} \approx 2.7 \times 10^7$ cm/s ≈ **273 km/s**.

### 4.3 Fast-mode speed (general propagation angle)

$$v_{f} = \left\{\tfrac{1}{2}\left[v_A^2 + c_s^2 + \sqrt{(v_A^2 + c_s^2)^2 - 4 v_A^2 c_s^2 \cos^2\theta_B}\right]\right\}^{1/2}$$

The plus sign gives $v_f$ (fast-mode), the minus gives $v_s$ (slow-mode). For $\theta_B = 90^\circ$ (perpendicular propagation):

$$v_\mathrm{ms} = \sqrt{v_A^2 + c_s^2}$$

**Worked example (quiet corona):** $v_\mathrm{ms} = \sqrt{273^2 + 185^2} \approx 330$ km/s. This matches the mean EIT-wave speed range for Class 2 events.

### 4.4 Fast-mode Mach number from Rankine–Hugoniot

$$M_\mathrm{ms} = \sqrt{\frac{(5\beta_p + 5 + X) X}{(2 + \gamma \beta_p)(4 - X)}}$$

$X = n_d / n_u$ (density compression ratio), $\beta_p = 8\pi n k T / B^2$, $\gamma = 5/3$.

**Worked example:** For $X = 1.1$ (typical EIT wave amplitude) and $\beta_p = 0.1$ (quiet corona), $M_\mathrm{ms} \approx \sqrt{(0.5 + 5 + 1.1)(1.1) / ((2 + 5/3 \cdot 0.1)(4 - 1.1))} \approx \sqrt{7.26/6.29} \approx 1.07$ — linear wave. For $X = 1.5$ (S-wave), $M_\mathrm{ms} \approx 1.3$ — moderate shock.

### 4.5 Wave refraction in stratified corona (geometric acoustics)

Ray equation (2D, slowly varying medium):

$$\frac{d}{dt}\left(\frac{\vec{k}}{\omega}\right) = -\nabla\left(\frac{1}{v_\mathrm{ms}(\vec{r})}\right)$$

Rays bend **away** from regions of high $v_\mathrm{ms}$ (ARs, coronal holes — $v_\mathrm{ms} \approx 1000$ km/s) and **toward** valleys of low $v_\mathrm{ms}$. Since $v_\mathrm{ms}$ increases with height in the low corona ($\nabla v_\mathrm{ms}$ points upward through the first 1–2 scale heights), wave fronts tilt **forward (toward the surface)** as they propagate, explaining the Moreton-wave lagging ~25 Mm behind the coronal front (Warmuth 2010).

### 4.6 Moreton wave as chromospheric signature

The chromosphere has $c_s \approx 10$ km/s and $v_A \approx 100$ km/s, giving $v_\mathrm{ms}^\mathrm{chrom} \approx 100$ km/s. A chromospheric disturbance at 1000 km/s would require $M \ge 10$, which would dissipate rapidly. Therefore Moreton signatures **cannot be a true chromospheric wave**; they are the **ground track of a coronal fast-mode wavefront** pressing the chromosphere downward (Uchida 1968 sweeping-skirt). The downward compression produces Hα emission in line center, blue-wing brightening (approaching plasma), and red-wing darkening (relaxation).

### 4.7 EIT-wave speed range and three kinematical classes

| Class | Initial speed | Kinematics | Interpretation |
|-------|---------------|------------|----------------|
| 1 | $\ge 320$ km/s | Strong deceleration, $\dot v$ up to $-2$ km/s² | Nonlinear fast-mode wave/shock |
| 2 | 170–320 km/s | Nearly constant | Linear / weakly nonlinear fast-mode |
| 3 | few tens km/s | Erratic, may accelerate | Pseudo-wave (magnetic reconfiguration) |

Deceleration follows a power-law $d \sim t^{\delta}$ with $\delta \approx 0.6$ — consistent with Sedov (1959) blast-wave self-similar solution.

### 4.8 Moreton vs EIT speed ratio

Average Moreton speed $\bar{v}_M \approx 650$ km/s. Average EIT-wave speed (low-cadence SOHO/EIT) $\bar{v}_E \approx 191$ km/s.

$$\frac{\bar{v}_M}{\bar{v}_E} \approx 3.4$$

This ratio led to the initial controversy. Resolved via deceleration: initial speeds $\sim$1000 km/s decay to $\sim$300 km/s over 100–500 s, sampled preferentially by low-cadence EIT in its late slow phase. High-cadence AIA (12 s) recovers initial speeds $> 1000$ km/s, with mean $\approx 644$ km/s — closer to the Moreton mean.

### 4.9 Shock formation: characteristic (simple wave) steepening

For a 1D nonlinear MHD simple wave, the characteristic propagation speed at each point on the profile is $v = v_\mathrm{ms}(\rho) + u(\rho)$, which increases with the local density perturbation (crest travels faster than leading edge). The time to form a shock by steepening is approximately:

$$t_\mathrm{shock} \approx \frac{L_0}{\Delta v_\mathrm{ms}} \approx \frac{L_0}{v_\mathrm{ms} \cdot (X - 1)}$$

**Worked example:** Pulse width $L_0 = 50$ Mm, $v_\mathrm{ms} = 330$ km/s, initial compression $X - 1 = 0.3$ (strong S-wave) ⇒ $t_\mathrm{shock} \approx 5 \times 10^7 / (330 \times 0.3) \approx 505$ s ≈ 8 min — consistent with the rapid appearance of Moreton fronts and early-phase sharp EUV waves. For a weak wave with $X - 1 = 0.05$, $t_\mathrm{shock} \approx 51$ min ≳ observation window, hence no shock forms (Class 2).

### 4.10 Deflection at coronal hole boundary

Using Huygens–Fresnel with angle of incidence = angle of reflection (Kienreich 2013):

$$\theta_\mathrm{inc} = \theta_\mathrm{refl}$$

Reflected wave typically deflected to higher corona; reflected speed 100–500 km/s vs primary 300–800 km/s. Lower reflected speed can be due to: (i) nonlinear primary + linear reflection; (ii) reflected wave propagating in flow field of primary; (iii) deflection into 3D with projected foreshortening.

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1947 --- Payne-Scott: metric type II radio bursts (solar shock signature)
1960 --- Moreton: Hα Moreton waves (1000 km/s chromospheric disturbance)
1968 --- Uchida: sweeping-skirt hypothesis — coronal fast-mode MHD wavefront
1973 --- Uchida et al.: refined fast-mode simulations predict dome structure
1991 --- Yohkoh launch: SXT instrument
1995 --- SOHO launch: EIT, LASCO, CDS, UVCS
1997 --- Moses / Thompson et al.: first EIT-wave observations (May 12)
1999 --- Delannée & Aulanier: stationary bright fronts → pseudo-wave challenge
2001 --- Warmuth et al.: Moreton-wave deceleration, unified interpretation
2002 --- Chen et al.: field-line stretching / hybrid model (2D MHD)
2004 --- Zhukov & Auchère: bimodality hypothesis for coronal waves
   |    Narukage et al.: SXT dome limb event
2005 --- Warmuth et al.: combined Hα + EIT + SXI deceleration (power-law $\delta = 0.6$)
2006 --- STEREO launch (two-spacecraft EUVI)
2007 --- Attrill et al.: reconnection-front pseudo-wave model
2008 --- Delannée et al.: 3D current-shell model
   |    Long et al.: first EUV-wave reflection candidate (EUVI)
2009 --- Gopalswamy et al.: first clear coronal-hole reflection (multi-reflection)
   |    Patsourakos & Vourlidas: stereoscopic dome evidence
2010 --- SDO launch: AIA 12-s cadence, 7 EUV channels
   |    Veronig et al.: dome-like CME-driven wave (2010 Jan 17)
2011 --- Warmuth & Mann: three kinematical classes
   |    Downs et al.: 3D thermodynamic MHD synthesis vs AIA
   |    Schrijver et al. / Kozarev et al. / Ma et al.: DEM / heating analyses
2013 --- Kwon et al.: white-light STEREO/COR1 upper-corona counterpart
   |    Nitta et al.: AIA 138-event statistical study
   |    Kienreich et al.: homologous reflections (Huygens–Fresnel)
2014 --- Miteva et al.: 179 SEP statistics, 87% EIT-wave associated
   |    Nitta et al.: 34-event EUVI, mean 300 km/s
   |    Muhr et al.: 60-event EUVI, cadence test
2015 --- *** Warmuth: this Living Reviews article — unified hybrid scenario ***
```

이 2015 리뷰는 반세기의 관측과 이론을 통합해 "fast-mode wave + pseudo-wave" 하이브리드 시나리오를 표준 모델로 확립한 분수령이며, 이후 Parker Solar Probe (2018)와 Solar Orbiter (2020) 시대의 coronal-shock/SEP 연구의 핵심 배경이 된다.

This 2015 review consolidates half a century of observations and theory into the hybrid "fast-mode wave + pseudo-wave" unified scenario that became the standard model for the Parker Solar Probe (2018) and Solar Orbiter (2020) eras of coronal-shock and SEP research.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Moreton (1960) | 최초 Hα wave 발견 / First Hα wave discovery | Warmuth의 모든 chromospheric signature 논의의 출발점 / Starting point for all chromospheric wave discussion |
| Uchida (1968) | Sweeping-skirt hypothesis; coronal fast-mode wavefront | Warmuth Sec. 6.1.1의 표준 모델의 기반 / Foundation of the fast-mode wave model adopted in Sec. 6.1.1 |
| Thompson et al. (1998) | 최초 EIT wave 관측 / First EIT wave observation | Warmuth Sec. 3.1.1의 "textbook" 예 / Textbook example in Sec. 3.1.1 |
| Delannée & Aulanier (1999) | Stationary bright fronts as pseudo-wave evidence | Warmuth Sec. 4.7.4, 6.2의 pseudo-wave 논쟁 촉발 / Ignited the pseudo-wave debate covered in Sec. 4.7.4, 6.2 |
| Chen et al. (2002, 2005b) | 2D hybrid model: fast-mode shock + field-line stretching | Warmuth Sec. 6.2.1, 6.3의 hybrid 시나리오 원형 / Prototype of the hybrid scenario in Sec. 6.2.1, 6.3 |
| Zhukov & Auchère (2004) | Bimodality of coronal waves | Warmuth Sec. 4.1.4, 6.3 다중 wavefront 논의의 씨앗 / Seed of the multiple-wavefront discussion |
| Warmuth et al. (2004b, 2005) | Combined deceleration of Hα + EIT + SXI | Warmuth Sec. 4.2.2 단일 감속 disturbance의 핵심 증거 / Key evidence for a single decelerating disturbance |
| Gopalswamy et al. (2009) | First clear coronal-hole reflection | Warmuth Sec. 4.7.2 wave 해석의 결정적 근거 / Decisive evidence for wave interpretation |
| Warmuth & Mann (2011) | Three kinematical event classes | Warmuth Sec. 4.2.3의 class 시스템 / Source of the three-class kinematical system |
| Downs et al. (2011, 2012) | 3D thermodynamic MHD synthesis for AIA | Warmuth Sec. 6.3 관측-모델 직접 비교 / Direct observation–model comparison in Sec. 6.3 |
| Liu & Ofman (2014) | Companion Living Reviews on small-scale waves | Complementary review on small-scale and QFP waves / Warmuth와 상보적인 리뷰 |

---

## 7. References / 참고문헌

1. Warmuth, A., "Large-scale Globally Propagating Coronal Waves", *Living Rev. Solar Phys.*, **12**, 3 (2015). DOI: 10.1007/lrsp-2015-3.
2. Moreton, G. E., "Hα Observations of Flare-Initiated Disturbances with Velocities ~1000 km/sec", *Astron. J.*, **65**, 494 (1960).
3. Uchida, Y., "Propagation of Hydromagnetic Disturbances in the Solar Corona and Moreton's Wave Phenomenon", *Sol. Phys.*, **4**, 30 (1968).
4. Thompson, B. J., Plunkett, S. P., Gurman, J. B. et al., "SOHO/EIT Observations of an Earth-directed Coronal Mass Ejection on 1997 May 12", *Geophys. Res. Lett.*, **25**, 2465 (1998).
5. Delannée, C., Aulanier, G., "CME Associated with Transequatorial Loops and a Bald Patch Flare", *Sol. Phys.*, **190**, 107 (1999).
6. Chen, P. F., Wu, S. T., Shibata, K., Fang, C., "Evidence of EIT and Moreton Waves in Numerical Simulations", *ApJ*, **572**, L99 (2002).
7. Chen, P. F., Fang, C., Shibata, K., "A Full View of EIT Waves", *ApJ*, **622**, 1202 (2005b).
8. Warmuth, A., Vršnak, B., Magdalenić, J. et al., "A Multiwavelength Study of Solar Flare Waves", *Astron. Astrophys.*, **418**, 1101 & 1117 (2004a,b).
9. Warmuth, A., Mann, G., "Kinematical Evidence for Physically Different Classes of Large-scale Coronal EUV Waves", *Astron. Astrophys.*, **532**, A151 (2011).
10. Warmuth, A., Mann, G., "A Model of the Alfvén Speed in the Solar Corona", *Astron. Astrophys.*, **435**, 1123 (2005a).
11. Gopalswamy, N., Yashiro, S., Temmer, M. et al., "EUV Wave Reflection from a Coronal Hole", *ApJ*, **691**, L123 (2009).
12. Veronig, A. M., Muhr, N., Kienreich, I. W., Temmer, M., Vršnak, B., "First Observations of a Dome-shaped Large-scale Coronal EUV Wave", *ApJ*, **716**, L57 (2010).
13. Downs, C., Roussev, I. I., van der Holst, B. et al., "Studying Extreme Ultraviolet Wave Transients with a Digital Laboratory", *ApJ*, **728**, 2 (2011); **750**, 134 (2012).
14. Kienreich, I. W., Muhr, N., Veronig, A. M. et al., "Solar TErrestrial RElations Observatory-A (STEREO-A) and PRoject for OnBoard Autonomy 2 (PROBA2) Quadrature Observations of Reflections of Three EUV Waves from a Coronal Hole", *Sol. Phys.*, **286**, 201 (2013).
15. Liu, W., Ofman, L., "Advances in Observing Various Coronal EUV Waves in the SDO Era and Their Seismological Applications (Invited Review)", *Sol. Phys.*, **289**, 3233 (2014).
16. Miteva, R., Klein, K.-L., Kienreich, I. et al., "Solar Energetic Particles and Associated EIT Disturbances in Solar Cycle 23", *Sol. Phys.*, **289**, 2601 (2014).
17. Nitta, N. V., Schrijver, C. J., Title, A. M., Liu, W., "Large-scale Coronal Propagating Fronts in Solar Eruptions as Observed by the Atmospheric Imaging Assembly on SDO — A Systematic Study", *ApJ*, **776**, 58 (2013).
18. Attrill, G. D. R., Harra, L. K., van Driel-Gesztelyi, L., Démoulin, P., "Coronal 'Wave': Magnetic Footprint of a Coronal Mass Ejection?", *ApJ*, **656**, L101 (2007).
19. Delannée, C., Török, T., Aulanier, G., Hochedez, J.-F., "A New Model for Propagating Parts of EIT Waves: A Current Shell in a CME", *Sol. Phys.*, **247**, 123 (2008).
20. Priest, E. R., *Solar Magnetohydrodynamics*, Reidel (1982).
21. Sedov, L. I., *Similarity and Dimensional Methods in Mechanics*, Academic Press (1959).
22. Moses, D., Clette, F., Delaboudinière, J.-P. et al., "EIT Observations of the Extreme Ultraviolet Sun", *Sol. Phys.*, **175**, 571 (1997).
23. Delaboudinière, J.-P., Artzner, G. E., Brunaud, J. et al., "EIT: Extreme-Ultraviolet Imaging Telescope for the SOHO Mission", *Sol. Phys.*, **162**, 291 (1995).
24. Klassen, A., Aurass, H., Mann, G., Thompson, B. J., "Catalogue of the 1997 SOHO-EIT coronal transient waves and associated type II radio burst spectra", *Astron. Astrophys. Suppl. Ser.*, **141**, 357 (2000).
25. Vršnak, B., Cliver, E. W., "Origin of Coronal Shock Waves", *Sol. Phys.*, **253**, 215 (2008).
26. Veronig, A. M., Gömöry, P., Kienreich, I. W. et al., "Plasma Diagnostics of Coronal Dimming Events", *ApJ*, **743**, L10 (2011).
27. Long, D. M., DeLuca, E. E., Gallagher, P. T., "The Wave Properties of Coronal Bright Fronts Observed Using SDO/AIA", *ApJ*, **741**, L21 (2011a).
28. Kozarev, K. A., Korreck, K. E., Lobzin, V. V., Weber, M. A., Schwadron, N. A., "Off-limb Solar Coronal Wavefronts from SDO/AIA Running-difference Data", *ApJ*, **733**, L25 (2011).

---

## 8. Summary One-liner / 한 줄 요약

**한국어:** Warmuth (2015)는 대규모 코로나 파동이 "fast-mode MHD wave/shock (외곽) + CME 팽창으로 인한 자기 재구성 (내부)"의 통합 하이브리드 시나리오로 설명됨을 약 반세기의 관측·이론·수치를 종합해 보여주며, 이는 현대 태양 활동 및 우주기상 연구의 표준 해석이 되었다.

**English:** Warmuth (2015) synthesizes half a century of observations, theory, and simulations to show that large-scale globally propagating coronal waves are best understood as a hybrid of an outer fast-mode MHD wave/shock and an inner CME-driven magnetic-reconfiguration brightening — the standard interpretation now used across solar-eruption and space-weather research.
