---
title: "Pre-Reading Briefing: Wave Modeling of the Solar Wind"
paper_id: "21_ofman_2010"
topic: Living_Reviews_in_Solar_Physics
date: 2026-04-19
type: briefing
---

# Wave Modeling of the Solar Wind: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Ofman, L. (2010). *Wave Modeling of the Solar Wind*. Living Reviews in Solar Physics, 7, 4.
**Author(s)**: Leon Ofman (Catholic University of America / NASA GSFC)
**Year**: 2010
**DOI**: 10.12942/lrsp-2010-4

---

## 1. 핵심 기여 / Core Contribution

**한국어:**
이 논문은 **파동이 주도하는 태양풍 모델링**에 대한 결정판 리뷰다. Parker (1958)의 전통적 열압력 구동 모델로는 **빠른 태양풍**(극관 coronal hole에서 나오는 ~700–800 km/s)과 **양성자·이온의 강한 비등방 가열**을 설명할 수 없다는 관측적 증거가 1990년대 이후 축적되자, Alfvén파를 비롯한 MHD 파동이 **운동량과 에너지의 핵심 원천**으로 떠올랐다. Ofman은 (a) 1D WKB 파동 구동 모델, (b) 2.5D 다유체 MHD 시뮬레이션, (c) 이온 사이클로트론 공명 흡수, (d) 난류 캐스케이드(Kolmogorov vs. Kraichnan) 기반 가열, (e) PIC/혼합 운동론(kinetic) 접근을 체계적으로 정리하고, 각 모델이 Helios, Ulysses, SOHO/UVCS 관측과 얼마나 일치하는지를 평가한다. 이 리뷰는 **태양풍 가속·가열 문제의 "현재 수준"**을 정의하며, 이후 Parker Solar Probe와 Solar Orbiter 시대의 해석 틀을 제공했다.

**English:**
This is the definitive review of **wave-driven modeling of the solar wind**. By the late 1990s observations had accumulated showing that Parker's (1958) classical thermal-pressure model cannot explain either the **fast solar wind** (~700–800 km/s from polar coronal holes) or the **strong anisotropic heating** of protons and minor ions, making MHD waves — especially Alfvén waves — the leading candidate source of momentum and energy. Ofman systematically reviews (a) 1D WKB wave-driven models, (b) 2.5D multifluid MHD simulations, (c) ion-cyclotron resonant absorption, (d) turbulent-cascade heating (Kolmogorov vs. Kraichnan), and (e) PIC/hybrid kinetic approaches. He evaluates each against Helios, Ulysses, SOHO/UVCS, and in-situ data. The review defines the **state of the art of solar wind acceleration/heating theory** and supplied the interpretive framework for the Parker Solar Probe and Solar Orbiter era that followed.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

**한국어:**
Biermann(1951)의 혜성 꼬리 관측과 Parker(1958)의 이론적 예측 이후 태양풍은 "입증된 현상"이었다. Mariner 2(1962)가 직접 관측했고, 이후 Helios 1·2(1970년대)로 내부 태양권 플라스마 특성이 밝혀졌다. 그러나 문제는 **에너지 수지**였다: Parker의 Bondi-Parker 유형 열압력 구동 해는 코로나 온도(~$10^6$ K)에서 ~400 km/s까지만 만들어낼 수 있을 뿐, 빠른 바람의 ~750 km/s는 설명하지 못한다. Alfvén(1942)의 파동 이론과 Belcher-Davis(1971)의 in-situ Alfvén 변동 관측이 "파동이 남은 에너지를 공급한다"는 가설로 이어졌다. 1980–90년대 Hollweg, Isenberg, Leer, Holzer가 1D 파동 구동 유체 모델을 개발했다. **결정적 관측**은 SOHO/UVCS(1998)의 coronal hole 이온 비등방성 발견 — O⁵⁺의 수직 온도가 $10^8$ K 이상, 평행 온도보다 훨씬 높음. 이것은 이온-사이클로트론 공명에 의한 선택적 가열을 강력히 시사했다. 2010년 Ofman의 리뷰는 이 10년간의 폭발적 발전을 종합.

**English:**
Since Biermann's (1951) comet-tail observations and Parker's (1958) theoretical prediction, the solar wind has been an established phenomenon. Mariner 2 (1962) gave the first direct measurement, and Helios 1–2 (1970s) mapped the inner heliosphere. The open problem was **energy balance**: Parker's thermal-pressure solution at coronal temperature (~$10^6$ K) reaches only ~400 km/s, not the ~750 km/s fast wind. Alfvén's (1942) wave theory and Belcher-Davis (1971) *in situ* Alfvén-wave observations gave rise to the hypothesis that waves supply the missing energy. Hollweg, Isenberg, Leer, and Holzer developed 1D wave-driven fluid models through the 1980s–90s. The **pivotal observation** was SOHO/UVCS (1998), which found highly anisotropic minor-ion temperatures in coronal holes: O⁵⁺ perpendicular temperature > $10^8$ K, far above parallel, strongly implying selective ion-cyclotron heating. Ofman's 2010 review synthesizes a decade of such explosive progress.

### 타임라인 / Timeline

```
1942  Alfvén — 자기유체 파동 / hydromagnetic waves
1951  Biermann — 혜성 꼬리의 태양풍 증거 / comet-tail evidence
1958  Parker — 이론적 태양풍 예측 / theoretical prediction
1962  Mariner 2 — 태양풍 직접 관측 / direct measurement
1971  Belcher & Davis — in situ Alfvén 파 검출 / Alfvén waves detected
1975+ Hollweg — WKB 파동 구동 풍 모델 / WKB wave-driven wind models
1990s Ulysses — 고위도 빠른 바람 관측 / fast wind over poles
1998  SOHO/UVCS — O⁵⁺ 비등방 가열 발견 / anisotropic ion heating
2000s Cranmer, Isenberg, Ofman — multifluid & cyclotron 모델 / multifluid + cyclotron
▶ 2010  ★ Ofman — 이 논문 / THIS PAPER
2018  Parker Solar Probe launch — in situ 코로나 관측 / in situ at the corona
2020  Solar Orbiter launch — 고위도 + 원격 조합 / high-latitude + remote combo
```

---

## 3. 필요한 배경 지식 / Prerequisites

**한국어:**
- **MHD 기초**: Maxwell + Navier-Stokes + 유도 방정식. Alfvén 속도 $v_A = B/\sqrt{\mu_0\rho}$
- **파동 이론**: Alfvén(비압축 횡파), slow/fast magnetosonic wave, dispersion relation
- **이온 사이클로트론 공명**: 자이로 주파수 $\Omega_i = q_iB/m_i$, 공명 조건
- **난류 캐스케이드**: Kolmogorov $E(k)\propto k^{-5/3}$ vs. Iroshnikov-Kraichnan $k^{-3/2}$
- **이전 논문 연결**: LRSP #3 (Marsch — kinetic properties), #8 (Cranmer — coronal heating/wind)
- **Parker 태양풍 솔루션**: 임계점(critical point) 통과 개념, Bondi-Parker 수식

**English:**
- **MHD basics**: Maxwell + Navier-Stokes + induction equation; Alfvén speed $v_A = B/\sqrt{\mu_0\rho}$
- **Wave theory**: Alfvén (incompressible transverse), slow/fast magnetosonic modes, dispersion relations
- **Ion cyclotron resonance**: gyrofrequency $\Omega_i = q_iB/m_i$, resonance condition
- **Turbulent cascade**: Kolmogorov $E(k)\propto k^{-5/3}$ vs. Iroshnikov-Kraichnan $k^{-3/2}$
- **Prior papers**: LRSP #3 (Marsch — kinetic properties), #8 (Cranmer — coronal heating/wind)
- **Parker solution**: critical point, Bondi-Parker integral

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **Fast / slow solar wind** | 극관 coronal hole에서 나오는 700–800 km/s 바람과 helmet streamer에서 나오는 300–500 km/s 바람 / 700–800 km/s from polar coronal holes vs. 300–500 km/s from streamer belt |
| **Alfvén wave** | 자기장 방향으로 전파되는 비압축 횡파. $v_A = B/\sqrt{\mu_0\rho}$로 전파 / incompressible transverse wave along $\mathbf{B}$, speed $v_A$ |
| **Wave pressure / ponderomotive force** | 진폭 프로파일의 기울기가 만드는 평균 힘 $-\nabla\langle\delta B^2\rangle/(8\pi)$ — 바람 가속 기여 / mean force from gradient of wave amplitude; accelerates flow |
| **Alfvén surface** | 바람 속도 = Alfvén 속도인 경계면 (~10–20 $R_\odot$). 이 안은 "연결됨", 바깥은 "분리됨" / surface where $u = v_A$; inside Sun-connected, outside not |
| **Ion cyclotron resonance** | $\omega - k_\parallel v_\parallel = n\Omega_i$에서 파동과 이온이 공명하여 에너지 이전 / wave-particle resonance where wave and ion frequencies match |
| **WKB approximation** | 파동 진폭이 배경보다 천천히 변한다고 가정한 준정상 전파 근사 / quasi-stationary wave propagation when amplitude varies slowly |
| **Multifluid MHD** | 양성자, 전자, 알파 입자 등 각 종을 별도 유체로 취급하는 모델 / each species (proton, electron, α) treated as a separate fluid |
| **Anisotropic temperature** | $T_\perp \ne T_\parallel$ — 자기장 평행/수직 방향의 운동 에너지가 다름 / kinetic energies differ parallel vs. perpendicular to B |
| **Turbulent cascade** | 큰 스케일 파동이 비선형 상호작용으로 작은 스케일로 에너지 전달, 이온 스케일에서 소산 / wave energy transported to small scales, dissipated at ion scales |
| **Elsässer variables** $\mathbf{z}^\pm = \mathbf{u}\pm\mathbf{b}/\sqrt{\mu_0\rho}$ | Alfvén 파의 상/하류 전파 성분 분리. 난류 상호작용 해석의 핵심 변수 / upstream/downstream Alfvén modes; key for turbulence analysis |
| **Kolmogorov vs. Iroshnikov-Kraichnan** | 전자의 $k^{-5/3}$와 후자의 $k^{-3/2}$ 에너지 스펙트럼 예측 / $k^{-5/3}$ vs. $k^{-3/2}$ spectral predictions |
| **Bondi-Parker critical point** | 열압력 구동 바람의 음속 임계점 통과 조건 / sonic critical point of thermally driven wind |

---

## 5. 수식 미리보기 / Equations Preview

### (1) Parker의 이중성 태양풍 방정식 / Parker's isothermal wind equation

$$
\frac{1}{u}\frac{du}{dr}\left(u^2 - c_s^2\right) = \frac{2 c_s^2}{r} - \frac{GM_\odot}{r^2}
$$

**한국어:** $u$ = 바람 속도, $c_s$ = 이온음속. 임계점 $r_c = GM_\odot/(2c_s^2)$에서 분자·분모 동시 0 조건을 만족하는 해가 물리적 "바람" 해. 파동 없는 고전 이론.
**English:** $u$ = wind speed, $c_s$ = ion sound speed. The physical "wind" solution passes the critical point $r_c = GM_\odot/(2c_s^2)$ where numerator and denominator both vanish. Classical wave-free theory.

### (2) 파동 압력을 포함한 운동량 방정식 / Momentum equation with wave pressure

$$
\rho u\frac{du}{dr} = -\frac{dp}{dr} - \frac{\rho GM_\odot}{r^2} + F_{\rm wave}, \quad F_{\rm wave} = -\frac{1}{2}\frac{d\langle\delta B^2\rangle/\mu_0}{dr}
$$

**한국어:** 마지막 항이 Alfvén 파의 운동량 기여. 바람을 추가로 가속하는 핵심 메커니즘.
**English:** The last term is the momentum contribution of Alfvén waves — the key extra accelerator.

### (3) WKB 파동 작용 보존 / WKB wave-action conservation

$$
\frac{d}{dr}\left[\langle\delta B^2\rangle\frac{(u+v_A)^2}{v_A}A\right] = 0
$$

**한국어:** 단면적 $A(r)$ 흐름관을 따라 파동 작용이 보존됨. 감쇠 없이 전파되는 WKB 근사의 핵심.
**English:** Wave action is conserved along a flux tube with cross-section $A(r)$ — the core WKB statement, absent damping.

### (4) 이온 사이클로트론 공명 조건 / Ion-cyclotron resonance condition

$$
\omega - k_\parallel v_\parallel = n\,\Omega_i, \quad \Omega_i = \frac{q_iB}{m_ic}
$$

**한국어:** 파동 주파수(도플러 보정)와 이온 자이로 주파수의 $n$배가 같을 때 공명. $T_\perp > T_\parallel$을 만드는 주 메커니즘.
**English:** Resonance when Doppler-shifted wave frequency matches $n\Omega_i$. Primary mechanism behind $T_\perp > T_\parallel$.

### (5) Elsässer 변수를 이용한 난류 스펙트럼 진화 / Turbulence in Elsässer form

$$
\partial_t z^\pm + (\mathbf{u}\mp\mathbf{v}_A)\cdot\nabla z^\pm = -\rho^{-1}\nabla p_T + \nu\nabla^2 z^\pm
$$

**한국어:** 상/하류 성분 $\mathbf{z}^\pm$의 비선형 상호작용이 난류 캐스케이드를 만든다. 이온 소산 스케일에서 가열로 종결.
**English:** Nonlinear coupling of co-/counter-propagating modes $\mathbf{z}^\pm$ drives the cascade, terminating in ion-scale heating.

---

## 6. 읽기 가이드 / Reading Guide

**한국어:**
이 리뷰는 약 60페이지로, 다음 **3-pass 전략**을 권장:

1. **1-pass (1–2시간)**: §1(서론), §2(관측 요약), §9(결론). 관측 제약 먼저 파악.
2. **2-pass (2–3시간)**: §3(Parker 이론 복습), §4(WKB 1D 파동 구동 모델), §5(다유체 MHD). 수식 (1)–(3)을 손으로.
3. **3-pass (심화)**: §6(이온 사이클로트론), §7(난류 캐스케이드), §8(kinetic/PIC). 2010년 당시의 전선.

**특히 주의할 점:**
- Figure 2(Helios/Ulysses 바람 속도 대 위도), Figure 8(SOHO/UVCS O⁵⁺ 비등방 온도)은 이 리뷰의 "상징 그림".
- 저자는 모델을 소개할 때 **관측과의 비교**를 철저히 한다. "모델이 어떤 관측을 설명하고 어떤 것을 놓치는가"를 메모할 것.
- 파동 감쇠와 가열 사이의 에너지 수지를 계속 추적하라.

**English:**
~60 pages. Recommended **3-pass strategy**:

1. **Pass 1 (1–2h)**: §1 intro, §2 observations, §9 conclusion. Fix the observational constraints first.
2. **Pass 2 (2–3h)**: §3 (Parker review), §4 (WKB 1D wave-driven), §5 (multifluid MHD). Work Eqs. (1)–(3) by hand.
3. **Pass 3 (deep dive)**: §6 (cyclotron), §7 (turbulence cascade), §8 (kinetic/PIC). The 2010 frontier.

**Watch for:** Fig. 2 (Helios/Ulysses speed vs. latitude) and Fig. 8 (UVCS O⁵⁺ anisotropic temperatures) are iconic. Ofman compares each model to observation — tabulate what it explains vs. what it misses. Track the wave-energy budget throughout.

---

## 7. 현대적 의의 / Modern Significance

**한국어:**
이 리뷰는 **Parker Solar Probe(2018)**와 **Solar Orbiter(2020)** 임무의 이론적 전주곡이 되었다. 세 가지 영향:

1. **PSP 이후 switchback 관측의 해석틀 제공**: Alfvén파가 유한 진폭에서 S자 반전(switchback)으로 변형. Ofman이 미리 정리한 파동 전파 이론이 기초 언어가 됨.
2. **코로나 가열 문제(coronal heating problem)와의 통합**: 태양풍 가열과 코로나 가열이 같은 파동 구동 메커니즘으로 연결될 수 있음을 체계화.
3. **수치 시뮬레이션의 현대적 설계**: BATS-R-US, AWSoM, PENCIL-MHD 등 현대 태양풍 코드가 이 리뷰의 모델 분류에서 파생. 특히 AWSoM(Alfvén Wave Solar Model)은 리뷰의 multi-wave-mode 접근을 구현.

실용적으로, 이 논문은 **우주 기상 예보**(태양풍 속도·밀도 예측), **항성풍 이론**(K, M-dwarf 별), **외부 태양권 경계** 연구와 공통 틀을 공유한다.

**English:**
This review became the theoretical prelude to **Parker Solar Probe (2018)** and **Solar Orbiter (2020)**. Three impacts:

1. **Framework for interpreting switchbacks**: finite-amplitude Alfvén waves that deform into S-shaped reversals observed by PSP — Ofman's wave-propagation review provides the baseline vocabulary.
2. **Unification with the coronal-heating problem**: organizes how the same wave-driven mechanisms may solve both coronal heating and solar wind heating/acceleration.
3. **Blueprint for modern simulations**: codes like BATS-R-US, AWSoM, PENCIL-MHD descend from the review's model taxonomy; AWSoM (Alfvén Wave Solar Model) directly implements the multi-wave-mode approach.

Practically, the paper shares its framework with **space-weather forecasting** (speed/density prediction), **stellar-wind theory** (K, M dwarfs), and **outer-heliosphere boundary** research.

---

## Q&A

### Q1. Nanoflare는 현재 받아들여지고 있는 정설인가? 관측된 사례가 있는가? / Are nanoflares an established consensus? Have they been observed?

**한국어:**
**결론: "유력한 후보지만 완전히 확립된 정설은 아니다."** 코로나 가열 문제에서 nanoflare는 **파동 가열(Alfvén wave dissipation)과 함께 양대 후보**이며, 둘 중 어느 한 쪽이 단독으로 승리했다고 보는 연구자는 소수다. 현재의 합의는 "두 메커니즘이 공존하며, 영역에 따라 지배 비율이 다르다"는 것에 가깝다.

**이론의 기원:**
- **Parker (1988)**: 광구 대류가 코로나 자기장 다발을 꼬아(braiding) 무수히 많은 작은 current sheet를 형성 → 작은 reconnection event (~10²⁴ erg, 일반적 flare의 10⁻⁹) → 연속적으로 발생하여 코로나를 가열.

**관측적 증거:**

1. **간접적 증거(강함)**: flare 에너지 분포 $dN/dE \propto E^{-\alpha}$. $\alpha > 2$이면 작은 이벤트가 총 에너지를 지배 → nanoflare 가설 지지. 관측된 값은 $\alpha = 1.5–2.7$로 **논쟁 중** (경계선 근처).
2. **준직접 관측**: RHESSI/NuSTAR의 **hard X-ray 비열 방출** — active region에서 수 keV의 비열 성분이 검출됨(Hannah et al. 2008, Ishikawa et al. 2017). nanoflare 가열의 **부산물**로 해석.
3. **EUV impulsive brightening**: Hi-C(2012), IRIS, SDO/AIA가 작은 spatial scale의 짧은 밝아짐 이벤트를 관측 — "campfires"라고도 불림.
4. **Solar Orbiter/EUI의 결정적 발견(2021)**: Berghmans et al.은 quiet Sun에서 **~400 km 크기, 10–200초 지속의 "campfires"**를 발견. 개별 에너지는 nanoflare 수준(~10²³–10²⁴ erg). *이것이 현재까지 가장 직접에 가까운 관측.*
5. **Parker Solar Probe의 switchback**: nanoflare-구동 제트가 switchback의 기원 중 하나일 가능성 제시(Bale et al. 2019, Fisk & Kasper 2020).

**여전히 남은 논쟁:**
- Berghmans의 campfires가 진짜 Parker-type nanoflare(reconnection)인지, 아니면 다른 현상(파동 가열 위치, small-scale emergence)인지 **아직 결론 없음**.
- nanoflare가 총 코로나 가열 에너지의 몇 %를 담당하는지 정량화가 어려움.
- 난류/파동 가열과의 **에너지 분할 비율**이 active region vs. coronal hole에서 어떻게 다른지 여전히 미해결.

**Ofman (2010) 리뷰의 관점:**
이 리뷰는 **파동 쪽에 집중**하므로 nanoflare는 §1 서론에서 "alternative mechanism"으로 언급만 하고 깊이 다루지 않습니다. 코로나 가열 문제의 전체 그림은 **LRSP #8 Cranmer (2009)** 리뷰나 **Klimchuk (2006) "On Solving the Coronal Heating Problem"**에서 균형 있게 다룹니다.

**English:**
**Bottom line: "a leading candidate, but not a fully established consensus."** In the coronal heating problem, nanoflares stand **alongside wave dissipation as one of two main candidates**; few researchers believe either has won outright. Current consensus is closer to: "both operate, with the mix varying by region."

**Origin (Parker 1988):** photospheric convection braids coronal field lines → myriad small current sheets → small reconnection events (~10²⁴ erg, ~$10^{-9}$ of a normal flare) → collectively heat the corona.

**Observational evidence:**

1. **Indirect (strong)**: flare energy distribution $dN/dE \propto E^{-\alpha}$. If $\alpha > 2$, small events dominate the total energy — supporting nanoflares. Observed values $\alpha = 1.5–2.7$ straddle the critical boundary — **still contested**.
2. **Quasi-direct**: RHESSI/NuSTAR detect weak **non-thermal hard X-rays at a few keV** in active regions (Hannah et al. 2008, Ishikawa et al. 2017) — interpreted as byproducts of nanoflare heating.
3. **EUV impulsive brightenings**: Hi-C (2012), IRIS, SDO/AIA see short brightenings at small spatial scales — sometimes called "campfires."
4. **The Solar Orbiter/EUI breakthrough (2021)**: Berghmans et al. discovered **"campfires" ~400 km in size, lasting 10–200 s** in the quiet Sun, with individual energies at nanoflare level (~10²³–10²⁴ erg). *The closest-to-direct observation to date.*
5. **Parker Solar Probe switchbacks**: nanoflare-driven jets suggested as a possible origin (Bale et al. 2019, Fisk & Kasper 2020).

**Remaining controversies:**
- Whether campfires are true Parker-type reconnection events or something else (wave-heating sites, small-scale emergence) is still **unresolved**.
- Quantifying the fraction of coronal heating attributable to nanoflares remains hard.
- The **energy partition between nanoflares and wave/turbulent heating** in active regions vs. coronal holes is still an open question.

**Context in Ofman (2010):** this review focuses on wave-driven mechanisms, so nanoflares are only briefly mentioned in §1 as an "alternative." For a balanced view of the full coronal-heating problem, see **LRSP #8 Cranmer (2009)** or **Klimchuk (2006) "On Solving the Coronal Heating Problem."**

