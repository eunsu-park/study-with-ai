---
title: "Pre-Reading Briefing: The Evolution of the Solar Wind"
paper_id: "74"
topic: Living_Reviews_in_Solar_Physics
date: 2026-04-23
type: briefing
---

# The Evolution of the Solar Wind: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Vidotto, A. A., "The Evolution of the Solar Wind", *Living Reviews in Solar Physics*, 18:3, 2021. DOI: 10.1007/s41116-021-00029-w
**Author(s)**: Aline A. Vidotto (Trinity College Dublin)
**Year**: 2021

---

## 1. 핵심 기여 / Core Contribution

**English.** This Living Review attacks one of the deepest open problems in solar physics: *how did the solar wind evolve to become what we see today, and what was it like 4 billion years ago?* Because we cannot time-travel to measure the young solar wind directly, Vidotto's strategy is to place the Sun within a population of "solar-like" stars at different evolutionary ages and use those stars as snapshots of the Sun's past and future. The review (i) surveys all available observational techniques for detecting the extremely tenuous winds of Sun-like stars (Ly-alpha astrospheric absorption, free-free radio, exoplanet transits, slingshot prominences, X-ray CME signatures, etc.), (ii) derives an empirical evolutionary sequence for the mass-loss rate $\dot M(t)$, (iii) connects this to the observed evolution of the three wind "ingredients" — magnetism, rotation, and coronal activity — via Skumanich-type braking laws, and (iv) examines implications for Earth's magnetosphere, atmospheric erosion, habitability of exoplanets, and galactic cosmic-ray modulation across the Sun's 4.6 Gyr history.

**한국어.** 이 Living Review는 태양물리학의 가장 근본적인 미해결 문제 중 하나를 다룬다: *오늘날의 태양풍은 어떻게 진화해 왔으며, 40억 년 전의 태양풍은 어떤 모습이었을까?* 과거의 태양풍을 직접 측정할 수 없으므로 Vidotto는 태양을 "solar-like stars" 집단 안에 배치하고, 서로 다른 진화 단계의 별들을 태양의 과거·미래를 보여주는 스냅샷으로 사용한다. 본 리뷰는 (i) 태양형 항성의 매우 희박한 항성풍을 탐지하기 위한 모든 관측 기법(Ly-alpha astrosphere 흡수, free-free 전파, 외계행성 통과, slingshot prominence, X-ray CME 신호 등)을 정리하고, (ii) 질량 손실률 $\dot M(t)$의 경험적 진화 시퀀스를 도출하며, (iii) 이를 태양풍의 세 가지 핵심 재료인 **자기장·자전·코로나 활동도**의 관측된 진화와 Skumanich 형 braking 법칙으로 연결하고, (iv) 지구 자기권, 행성 대기 침식, 외계행성 거주가능성, 은하우주선 변조 등 46억 년에 걸친 영향을 검토한다.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

**English.** By 2021 the solar physics community had accumulated five decades of in-situ solar wind data (Mariner 2 in 1962 through Parker Solar Probe's first perihelia in 2018-2020) and three decades of stellar wind detections, most notably Brian Wood's Ly-alpha astrosphere campaign (Wood 2004; Wood et al. 2005a) that provided ~20 mass-loss rate measurements of solar-like stars. Zeeman Doppler Imaging (ZDI, Donati & Brown 1997) had matured enough by ~2014 to yield large-scale magnetic maps for a hundred cool stars. In parallel, transiting exoplanets (HD 209458b, HD 189733b, GJ 436b) revealed evaporating atmospheres whose Ly-alpha line profiles encoded the host-star wind conditions. MHD wind simulations informed by ZDI maps (Vidotto et al. 2014a,b) could for the first time tie stellar-wind theory to observed surface magnetograms, enabling a proper evolutionary synthesis.

**한국어.** 2021년 시점에 태양물리 커뮤니티는 1962년 Mariner 2부터 2018-2020년 Parker Solar Probe 근일점 통과까지 50년에 걸친 in-situ 태양풍 자료와 30년치 항성풍 탐지 데이터를 축적하고 있었다. 특히 Brian Wood의 Ly-alpha astrosphere 관측(Wood 2004; Wood et al. 2005a)은 태양형 항성 ~20개의 질량 손실률을 제공했다. Zeeman Doppler Imaging(ZDI, Donati & Brown 1997)은 2014년경 성숙하여 100여 개 차가운 별의 대규모 자기장 지도를 산출하였다. 동시에 통과 외계행성(HD 209458b, HD 189733b, GJ 436b)은 증발하는 대기에서 Ly-alpha 선 프로파일을 통해 모성 항성풍 조건을 드러냈다. ZDI 기반 MHD 항성풍 시뮬레이션(Vidotto et al. 2014a,b)이 처음으로 이론과 관측 자기도를 연결하여 본격적인 진화 종합이 가능해졌다.

### 타임라인 / Timeline

```
1958  Parker — predicts supersonic solar wind (Paper #1)
1962  Mariner 2 — first in-situ solar wind detection
1967  Weber & Davis — magneto-rotator wind, Alfvén radius
1972  Skumanich — v sin i ∝ t^(-1/2) for cool cluster stars
1988  Kawaler — parametric braking law J-dot(Ω, M, R, B)
1997  Donati & Brown — Zeeman Doppler Imaging method
2004  Wood — astrosphere Ly-alpha method for M-dot
2005  Wood et al. — first evolutionary M-dot vs age power-law
2012  Matt et al. — 2D MHD-based braking law (Eq. 71)
2014  Vidotto et al. — first ZDI-based MHD wind grid
2019  ó Fionnagáin et al. — radio emission predictions for stellar winds
2021  Vidotto — THIS REVIEW: synthesis of the evolutionary picture
```

---

## 3. 필요한 배경 지식 / Prerequisites

**English.** The reader should be comfortable with: (i) ideal MHD equations (continuity, momentum with Lorentz force, induction, energy); (ii) Parker's isothermal wind solution including the critical/sonic point $r_c = GM_\star/(2 c_s^2)$ and the transonic topology of the momentum equation; (iii) the Weber-Davis (1967) rotating magnetic wind — the Alfvén point and its role as the lever arm; (iv) power-law gyrochronology ($\Omega_\star \propto t^{-b}$, with $b \simeq 0.5$) and the Rossby number $Ro = P_{rot}/\tau_c$; (v) stellar spectral classification (FGK main-sequence stars, M-dwarfs), X-ray / EUV activity proxies ($L_X/L_{bol}$), and basic exoplanet transit spectroscopy; (vi) order-of-magnitude estimates in cgs for solar quantities ($M_\odot = 2\times10^{33}$ g, $R_\odot = 7\times10^{10}$ cm, solar rotation $\Omega_\odot \approx 2.7\times10^{-6}$ rad s$^{-1}$). Readers should also recall that the current solar wind mass-loss rate is $\dot M_\odot \simeq 2\times10^{-14}\,M_\odot$ yr$^{-1}$, corresponding to ~10$^{12}$ g s$^{-1}$ of plasma leaving the Sun.

**한국어.** 독자는 다음을 숙지해야 한다: (i) 이상 MHD 방정식(연속, 운동량(Lorentz 포함), 유도, 에너지); (ii) Parker 등온 항성풍 해와 임계/음속점 $r_c = GM_\star/(2 c_s^2)$ 및 운동량 방정식의 transonic 위상; (iii) Weber-Davis(1967) 회전 자기 항성풍과 Alfvén 점의 "지레" 역할; (iv) 멱법칙 gyrochronology($\Omega_\star \propto t^{-b}$, $b \simeq 0.5$)와 Rossby 수 $Ro = P_{rot}/\tau_c$; (v) 항성 분광형(FGK 주계열성, M 왜성), X-선/EUV 활동 지표 $L_X/L_{bol}$, 외계행성 통과 분광; (vi) cgs 단위의 태양 기본량 추정치($M_\odot = 2\times10^{33}$ g, $R_\odot = 7\times10^{10}$ cm, $\Omega_\odot \approx 2.7\times10^{-6}$ rad s$^{-1}$). 현재 태양 질량 손실률은 $\dot M_\odot \simeq 2\times10^{-14}\,M_\odot$ yr$^{-1}$ 또는 플라즈마 ~$10^{12}$ g s$^{-1}$ 수준임을 기억하자.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| Astrosphere / 외부구(Astrosphere) | Analogue of the heliosphere around any star: a bubble blown by stellar wind into the ISM, bounded by bow shock - hydrogen wall - astropause - termination shock. Probed via Ly-alpha absorption / 항성풍이 성간 매질에 형성하는 거대 구조; bow shock / hydrogen wall / astropause / termination shock로 경계. Ly-alpha 흡수로 탐지 |
| Hydrogen wall / 수소 장벽 | Charge-exchange-generated neutral-H enhancement between bow shock and astropause whose absorption in Ly-alpha quantifies the wind ram pressure / bow shock과 astropause 사이의 charge exchange로 생성된 중성 수소 밀도 증가 영역; Ly-alpha 흡수로 ram pressure 측정 |
| Alfvén radius $r_A$ / 알펜 반경 | The surface where $u_r = v_A = B/\sqrt{4\pi\rho}$; inside it magnetic tension forces co-rotation, outside it the wind flows freely / 항성풍 방사속도가 알펜 속도와 같아지는 위치; 이 안쪽은 자기 장력이 공동회전을 강제, 바깥은 자유 유출 |
| Mass-loss rate $\dot M$ / 질량 손실률 | $\dot M = 4\pi r^2 \rho u_r$ — conserved along a steady spherically symmetric wind; for the Sun $\dot M_\odot \simeq 2\times10^{-14}\,M_\odot$ yr$^{-1}$ / 정상·구형 대칭 항성풍에서 보존되는 양; 태양은 $\dot M_\odot \simeq 2\times10^{-14}\,M_\odot$ yr$^{-1}$ |
| Wind dividing line / Wind 분할선 | Apparent drop of $\dot M$ at $F_X \gtrsim 10^6$ erg s$^{-1}$ cm$^{-2}$ (age ~600 Myr); controversial — may reflect magnetic topology change / $F_X \gtrsim 10^6$ erg s$^{-1}$ cm$^{-2}$ (~600 Myr)에서 $\dot M$이 급감하는 가설적 경계; 자기장 형상 변화 반영 가능성 |
| Skumanich law / Skumanich 법칙 | Empirical $\Omega_\star \propto t^{-1/2}$ spin-down relation for solar-type stars older than ~Hyades age (625 Myr); basis of gyrochronology / 태양형 별의 자전속도 감쇠($\Omega_\star \propto t^{-1/2}$); Hyades 이후 유효, gyrochronology의 기반 |
| Kraft break / Kraft 경계 | Transition near spectral type F5-F6 ($T_\textrm{eff}\simeq 6200$ K) above which stars lack thick convective envelopes and thus do not spin down; cooler stars brake / F5-F6(6200 K) 경계; 그 위 별들은 두꺼운 대류층이 없어 감속하지 않음 |
| Rossby number $Ro$ / Rossby 수 | $Ro = P_{rot}/\tau_c$; controls dynamo efficiency; activity saturates for $Ro \lesssim 0.1$ / 자전주기 / 대류 회전시간; $Ro \lesssim 0.1$에서 활동도 포화 |
| ZDI / Zeeman Doppler 영상 | Tomographic reconstruction of stellar surface magnetic field vector from polarized line profiles; sees only large-scale ($\ell \lesssim 10$) field / 편광 선 프로파일로부터 항성 표면 자기장 벡터를 토모그래피로 복원; 대규모($\ell \lesssim 10$) 성분만 포착 |
| Polytropic wind / 폴리트로픽 항성풍 | Wind with equation of state $P \propto \rho^\Gamma$; isothermal ($\Gamma=1$) is a special case used by Parker / $P \propto \rho^\Gamma$ 상태방정식을 가진 항성풍; 등온($\Gamma=1$)은 Parker의 특수해 |
| Saturated regime / 포화 영역 | Young stars with $Ro \lesssim 0.1$ whose $L_X/L_{bol}$ and magnetic flux saturate at ~10$^{-3}$; essential for avoiding runaway spin-down / $Ro \lesssim 0.1$ 젊은 별들에서 $L_X/L_{bol}$과 자기 flux가 ~10$^{-3}$에서 포화; runaway spin-down 방지에 필수 |
| Magnetic braking index $b$ / 자기 제동 지수 | Exponent in $\Omega_\star \propto t^{-b}$; modern values $b=0.56-0.62$ (Mamajek & Hillenbrand 2008; Delorme et al. 2011) / $\Omega_\star \propto t^{-b}$의 지수; 최근 값 $b=0.56$-$0.62$ |

---

## 5. 수식 미리보기 / Equations Preview

**1. Parker (polytropic) wind momentum equation / Parker 폴리트로픽 운동량 방정식.**
$$\frac{1}{u_r}\frac{du_r}{dr} = \frac{-GM_\star/r^2 + 2c_s^2/r}{u_r^2 - c_s^2}$$
**English.** This is Eq. (42) of the review. The numerator vanishing defines the sonic/critical point $r_c = GM_\star/(2c_s^2)$; the transonic solution is the physical wind.
**한국어.** 리뷰 식 (42). 분자가 0이 되는 곳이 음속/임계점 $r_c = GM_\star/(2c_s^2)$이며, 아음속→초음속으로 지나는 해가 물리적 항성풍이다.

**2. Mass-loss rate / 질량 손실률.**
$$\dot M = 4\pi r^2 \rho u_r \quad \text{(Eq. 34)}$$
**English.** Conserved for a steady 1D spherical wind; order $10^{12}$ g s$^{-1}$ for the Sun.
**한국어.** 정상 1D 구형 항성풍에서 보존; 태양의 경우 ~$10^{12}$ g s$^{-1}$.

**3. Weber-Davis angular momentum loss rate / 각운동량 손실률.**
$$\dot J = \tfrac{2}{3}\,\dot M\,\Omega_\star\, r_A^2 \quad \text{(Eq. 62)}$$
**English.** Even a modest magnetic field makes $r_A \gg R_\star$, so magnetic braking carries away far more angular momentum than an unmagnetised wind. This is what spins stars down.
**한국어.** 약한 자기장에도 $r_A \gg R_\star$이므로, 비자기 항성풍보다 훨씬 많은 각운동량을 빼앗는다. 이것이 별의 감속 메커니즘이다.

**4. Skumanich spin-down / Skumanich 감쇠.**
$$\Omega_\star \propto t^{-1/2} \quad \text{(Eq. 22 with }b=1/2\text{)}$$
**English.** Derived in Sect. 5.3.1 by combining $\dot J = \mathcal{I}\dot\Omega$, $B_{r,\star}\propto \Omega_\star$, and $\dot J \propto \Omega_\star^3$. Agrees with Skumanich's 1972 observation.
**한국어.** 5.3.1절에서 $\dot J = \mathcal{I}\dot\Omega$, $B_{r,\star}\propto \Omega_\star$, $\dot J \propto \Omega_\star^3$를 조합하여 유도. Skumanich(1972) 관측과 일치.

**5. Astrospheric/heliospheric radius / Astrosphere 반경.**
$$r_{\rm astro} = \left[\frac{\dot M\,u_\infty}{4\pi\rho_{\rm ISM}u_{\rm ISM}^2}\right]^{1/2} \quad \text{(Eq. 74)}$$
**English.** Pressure balance between stellar-wind ram pressure and ISM ram pressure; today $r_\odot^{astro}\simeq 122$ au; at $t\simeq 600$ Myr could have been 1300-1700 au (Rodgers-Lee et al. 2020).
**한국어.** 항성풍 ram pressure와 ISM ram pressure 균형; 현재 태양의 경우 $r_\odot^{astro}\simeq 122$ au; $t\simeq 600$ Myr에는 1300-1700 au까지 확장 가능.

---

## 6. 읽기 가이드 / Reading Guide

**English.** Read section-by-section: start with Sect. 1 to see the "big picture" diagram (Fig. 1) of wind-rotation-magnetism feedback. Sections 2.1-2.9 are best read as a catalogue — focus on Sect. 2.1 (astrosphere method) and Sect. 2.3 (exoplanet method), which are responsible for most $\dot M$ detections. Sect. 3 delivers the single most important empirical result (Eq. 17 and the $\dot M\propto t^{-0.99}$ or $t^{-2.33}$ evolutionary tracks, Fig. 11). Sect. 4 can be read selectively for the evolution of magnetism (4.1), rotation (4.2, Fig. 15), activity (4.3). Sect. 5 is the theory core: Sect. 5.2 derives Parker's and Weber-Davis's equations, Sect. 5.3.1 gives the elegant Skumanich derivation (Eqs. 64-68), and Sect. 5.3.2 compares braking laws (Kawaler 1988 vs Matt et al. 2012). Sect. 6 is practical impact: magnetosphere (6.1), exoplanet atmospheres (6.2), heliosphere/cosmic rays (6.3). Keep paper #1 (Parker 1958) accessible for reference.

**한국어.** 섹션별로 읽되, 먼저 1절의 "big picture"(Fig. 1 - 항성풍-자전-자기 피드백)을 확인한다. 2.1-2.9절은 카탈로그로 읽고, 2.1절(astrosphere 방법)과 2.3절(외계행성 방법)에 집중하자 - 대부분의 $\dot M$ 측정이 여기서 나온다. 3절에서 가장 핵심 경험 결과(식 17과 $\dot M\propto t^{-0.99}$ 또는 $t^{-2.33}$ 진화 궤적, Fig. 11)가 제시된다. 4절은 자기장(4.1)·자전(4.2, Fig. 15)·활동도(4.3) 진화를 선택적으로 읽는다. 5절이 이론의 핵심이며, 5.2절에서 Parker와 Weber-Davis 식을 유도, 5.3.1절에서 Skumanich 관계의 우아한 유도(식 64-68), 5.3.2절에서 제동 법칙 비교(Kawaler 1988 vs Matt et al. 2012)를 다룬다. 6절은 자기권(6.1), 외계행성 대기(6.2), 태양권/우주선(6.3) 등 실용적 파급을 담는다. Parker(1958, 논문 #1)를 참고자료로 함께 준비하자.

---

## 7. 현대적 의의 / Modern Significance

**English.** Vidotto's review sits at the crossroads of three major modern programs. **(1) Solar-terrestrial past:** understanding the young Sun's wind is essential for the "Faint Young Sun" paradox, for explaining Mars' atmospheric loss, and for calibrating paleomagnetosphere/paleoclimate models. **(2) Exoplanet habitability:** the wind of the host star sets the radiation pressure, magnetospheric stand-off distance, and atmospheric-escape regime of close-in planets; Kepler-11, TRAPPIST-1, and Proxima b studies all hinge on the host's wind evolution. **(3) Stellar astrophysics:** angular-momentum-loss laws are the foundation of gyrochronology, which dates stars without isochrones and feeds back into Galactic archaeology. As Parker Solar Probe and Solar Orbiter probe the corona in situ (2018+) and SKA promises radio detections of individual stellar winds (late 2020s), the empirical-theoretical synthesis laid out by Vidotto will become testable and refinable in ways previously impossible.

**한국어.** Vidotto의 리뷰는 세 가지 주요 현대 연구 프로그램의 교차점에 있다. **(1) 태양-지구 과거:** 젊은 태양풍 이해는 "Faint Young Sun" 역설, 화성 대기 소실 설명, paleomagnetosphere/고기후 모델링 보정에 필수적이다. **(2) 외계행성 거주가능성:** 모성 항성풍이 복사압·자기권 stand-off 거리·대기 탈출 체제를 결정하며, Kepler-11, TRAPPIST-1, Proxima b 연구가 모두 이에 의존한다. **(3) 항성 천문학:** 각운동량 손실 법칙은 gyrochronology의 기반이며, isochrone 없이 항성 연령을 측정함으로써 은하 고고학(Galactic archaeology)에 기여한다. 2018년 이후 Parker Solar Probe와 Solar Orbiter가 in situ로 코로나를 탐사하고, 2020년대 말 SKA가 개별 항성풍의 전파 탐지를 약속하는 지금, Vidotto의 경험-이론 종합은 이전에 불가능했던 방식으로 검증·정교화될 것이다.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
