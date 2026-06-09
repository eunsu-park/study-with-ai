---
title: "Pre-Reading Briefing: Oscillations and Waves in Sunspots"
paper_id: "45"
topic: Living_Reviews_in_Solar_Physics
date: 2026-04-23
type: briefing
---

# Oscillations and Waves in Sunspots: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Khomenko, E., & Collados, M., "Oscillations and Waves in Sunspots", *Living Reviews in Solar Physics*, **12**, 6 (2015). DOI: 10.1007/lrsp-2015-6
**Author(s)**: Elena Khomenko, Manuel Collados (Instituto de Astrofísica de Canarias, ULL)
**Year**: 2015

---

## 1. 핵심 기여 / Core Contribution

**한국어**: 이 리뷰 논문은 흑점(sunspot) 내 진동과 파동 현상에 대한 관측 및 이론 연구의 현 상태를 종합적으로 정리한다. 흑점의 강한 자기장은 광구(photosphere)부터 코로나(corona)까지 다양한 층에서 파동의 성질을 근본적으로 변형시킨다. 저자들은 (1) 암부(umbra)의 5분 광구 진동과 3분 채층(chromosphere) 진동, (2) 반암부(penumbra)의 running penumbral waves, (3) 자기-음향-중력 파동 모드(slow, fast, Alfvén)의 전파와 변환, (4) β=1 층에서의 모드 변환(mode conversion), (5) 공명 공동(resonant cavity) 모델을 통합적으로 논의한다. 이는 흑점 파동 물리에 대한 2015년 시점의 가장 일관된 그림을 제공한다.

**English**: This review paper provides a comprehensive synthesis of observational and theoretical work on oscillations and waves in sunspots. The strong sunspot magnetic field fundamentally modifies wave properties across atmospheric layers from the photosphere to the corona. The authors discuss in integrated fashion: (1) 5-min photospheric and 3-min chromospheric oscillations in the umbra, (2) running penumbral waves, (3) propagation and conversion of magneto-acoustic-gravity modes (slow, fast, Alfvén), (4) mode conversion at the β=1 layer, and (5) resonant cavity models. It delivers the most coherent picture of sunspot wave physics available as of 2015.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

**한국어**: 흑점의 진동 현상은 1960–1970년대 Beckers, Tallant, Howard, Bhatnagar 등의 분광 관측으로 처음 확인되었다. 광구에서의 5분 주기와 채층에서의 3분 umbral flashes가 일찍이 발견되었고, 1980–1990년대에는 Zhugzhda & Dzhalilov, Thomas, Cally 등이 자기 성층 대기에서의 MHD 파동 이론을 본격적으로 전개했다. 2000년대 SOHO, TRACE, Hinode, SDO, IRIS 같은 우주 망원경과 spectropolarimetric 기기 GRIS, CRISP의 등장으로 광구–채층–전이영역–코로나를 동시에 관측할 수 있게 되었다. 2015년 이 리뷰는 관측과 3D MHD 수치 시뮬레이션이 처음으로 통합적 그림을 그릴 수 있게 된 시점에 작성되었다.

**English**: Sunspot oscillations were first identified in the 1960s–70s via spectroscopic observations by Beckers, Tallant, Howard, Bhatnagar and others. The 5-min photospheric period and 3-min chromospheric umbral flashes were discovered early. Theoretical MHD wave theory in stratified magnetized atmospheres was then developed vigorously in the 1980s–90s by Zhugzhda & Dzhalilov, Thomas, Cally and others. The 2000s brought space missions (SOHO, TRACE, Hinode, SDO, IRIS) and spectropolarimeters (GRIS, CRISP) that enabled simultaneous multi-layer observations from photosphere to corona. This 2015 review was written at the moment observations and 3D MHD simulations could finally be unified into a single picture.

### 타임라인 / Timeline

```
1969 ─── Beckers & Tallant: umbral flashes in Ca II K
  │
1972 ─── Beckers & Schultz / Bhatnagar: 5-min umbral velocity oscillations
  │
1974 ─── Nye & Thomas: running penumbral wave theory
  │
1982 ─── Zhugzhda & Dzhalilov: MHD wave theory in stratified atmosphere
  │
1988 ─── Braun et al.: p-mode absorption by sunspots
  │
2001 ─── Cally: exact solution in vertical field, mode conversion theory
  │
2006 ─── Centeno et al.: slow waves propagating photosphere→chromosphere
  │
2008 ─── Cally & Goossens: 3D fast-to-Alfvén conversion
  │
2013 ─── Jess et al. / Reznikova et al.: cutoff-inclination connection for penumbral waves
  │
2015 ─── KHOMENKO & COLLADOS REVIEW (this paper)
```

---

## 3. 필요한 배경 지식 / Prerequisites

**한국어**:
- **유체역학 / MHD 기초**: 연속, 운동량, 에너지 방정식; 이상 유도 방정식; 자기압력 $B^2/(2\mu_0)$, 자기 장력
- **플라즈마 $\beta$**: $\beta = P_\text{gas}/P_\text{mag}$; 광구 아래 $\beta \gg 1$, 채층 이상 $\beta \ll 1$
- **음속/알펜 속도**: $c_s = \sqrt{\gamma P/\rho}$, $v_A = B/\sqrt{\mu_0 \rho}$
- **음향 cutoff 주파수**: $\omega_c = c_s/(2H)$ (등온 대기), 전파 가능 조건 $\omega > \omega_c$
- **p-mode 기초**: Sun의 global acoustic modes, $k$–$\omega$ 진단도
- **흑점 구조**: umbra(강한 수직 자기장 $\sim$2–3 kG), penumbra(경사진 자기장), Wilson depression
- **선 형성 / spectropolarimetry**: Stokes $I, Q, U, V$로부터 자기장 복원

**English**:
- **Fluid / MHD basics**: continuity, momentum, energy equations; ideal induction; magnetic pressure $B^2/(2\mu_0)$ and tension
- **Plasma $\beta$**: $\beta = P_\text{gas}/P_\text{mag}$; $\beta \gg 1$ below photosphere, $\beta \ll 1$ in chromosphere and above
- **Sound/Alfvén speeds**: $c_s = \sqrt{\gamma P/\rho}$, $v_A = B/\sqrt{\mu_0 \rho}$
- **Acoustic cutoff**: $\omega_c = c_s/(2H)$ (isothermal); propagation requires $\omega > \omega_c$
- **p-mode basics**: solar global acoustic modes, $k$–$\omega$ diagnostic diagram
- **Sunspot structure**: umbra (strong vertical $\sim$2–3 kG field), penumbra (inclined field), Wilson depression
- **Line formation / spectropolarimetry**: recovering magnetic field from Stokes $I, Q, U, V$

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| Umbra / 암부 | 흑점의 어두운 중심부, 수직 자기장 $\sim$2–3 kG / Dark central region with near-vertical $\sim$2–3 kG field |
| Penumbra / 반암부 | 섬유(filament) 구조를 가진 외곽부, 경사 자기장 60°–80° / Filamentary outer region with inclined field |
| Slow magneto-acoustic wave / 느린 자기음향파 | 저-$\beta$에서 자기장선 따라 전파하는 음향류 파 / Acoustic-like, propagates along $\vec{B}$ in low-$\beta$ |
| Fast magneto-acoustic wave / 빠른 자기음향파 | 저-$\beta$에서 등방적 자기류 파, 알펜 속도로 전파 / Isotropic magnetic-like, travels at $v_A$ in low-$\beta$ |
| Alfvén wave / 알펜파 | 비압축성 횡파, $\vec{v}_1 \perp \vec{B}_0$ / Incompressible transverse wave, $\vec{v}_1 \perp \vec{B}_0$ |
| Acoustic cutoff / 음향 차단 주파수 | $\omega_c = c_s/(2H)$; 이하에서는 evanescent / Below this, waves are evanescent |
| Mode conversion / 모드 변환 | $\beta \approx 1$ 층에서 fast↔slow 에너지 교환 / Energy exchange between fast/slow at $\beta \approx 1$ |
| Umbral flash / 암부 섬광 | Ca II K 선 코어의 주기적 3분 밝기 증가 (충격파) / Periodic 3-min brightening in Ca II K core (shocks) |
| Running penumbral wave / 이동 반암부파 | 내부→외부로 방사 전파하는 채층 파 (~10–15 km/s) / Chromospheric waves propagating radially outward (~10–15 km/s) |
| Plasma $\beta$ / 플라즈마 베타 | $\beta = P_\text{gas}/P_\text{mag}$; 광구 대기에서 height에 따라 1을 넘나듦 / Crosses unity somewhere in the atmosphere |
| Resonant cavity / 공명 공동 | 파동이 갇히는 영역 (광구/채층 등) / Region where waves become trapped (photospheric/chromospheric) |
| Eikonal approximation / 에이코날 근사 | 국소 평면파 가정으로 ray 경로 계산 / Local plane-wave assumption for ray-path calculation |

---

## 5. 수식 미리보기 / Equations Preview

**한국어 / English 공통**:

**(a) Acoustic cutoff frequency / 음향 차단 주파수**
$$\omega_c = \frac{c_s}{2H}$$
등온 대기에서 음향파 전파의 하한. $H = k_B T/(\mu m_H g)$ pressure scale height. / Lower bound for acoustic propagation in isothermal atmosphere.

**(b) Dispersion relation for magneto-acoustic-Alfvén modes / 자기음향-알펜 모드 분산 관계**
$$\omega^2 \vec{v}_1 = c_s^2 \vec{k}(\vec{k}\cdot\vec{v}_1) + [\vec{k}\times(\vec{k}\times(\vec{v}_1\times\vec{B}_0))] \times \frac{\vec{B}_0}{\mu\rho_0}$$
이 시스템은 세 가지 모드(fast, slow, Alfvén)를 지원한다. / This system supports fast, slow, and Alfvén modes.

**(c) Slow/fast limits / 느린/빠른 모드 극한**
$$\omega \approx c_s k, \quad (c_s \gg v_A);\quad\quad \omega \approx v_A k,\quad (c_s \ll v_A)$$
fast mode: 강/약자기장 극한에서 근사. / fast mode approximations in strong/weak field limits.

**(d) Fast-to-slow transmission coefficient / 빠른→느린 모드 투과 계수 (Cally 2005)**
$$T = \exp\left(-\frac{\pi k_\perp^2}{k_z |d/dz| (c_s^2/v_A^2)}\right)$$
$\alpha \approx 0$(파-자기장 정렬)에서 $T\to 1$, 완전 변환. / $T \to 1$ for zero attack angle (perfect alignment).

**(e) Evanescent vs propagating amplitude growth / Evanescent vs propagating 진폭 성장**
$$v \sim \exp\left(z\left[\frac{1}{2H} - \sqrt{\omega_c^2-\omega^2}/c_s\right]\right),\quad v \sim \exp(z/2H)$$
3분 진동이 5분보다 높은 곳에서 우세해지는 이유 설명. / Explains why 3-min oscillations dominate at height over 5-min.

---

## 6. 읽기 가이드 / Reading Guide

**한국어**:
- **Section 1–2**: 관측적 사실부터 시작 — 암부 5분(광구) vs 3분(채층) 진동, velocity/intensity power spectra
- **Section 3**: 반암부 파동 — running penumbral waves의 10–15 km/s 속도, outer 방향 전파
- **Section 4–5**: 자기장 fluctuations (작고 불확실) + sunspot surroundings (acoustic halo 5.5–7.5 mHz에서 40–60% enhancement)
- **Section 7**: 핵심 이론부 — homogeneous dispersion, eikonal approx, mode conversion (β=1), Alfvén 변환, resonant absorption & cavity
- **Section 8**: 논의 — "무엇이 파동을 구동하는가?" "5→3분 전환의 원리는?" "power suppression/halo의 메커니즘은?"
- **읽기 팁**: Section 7 수식 집중 — 실제로 이 리뷰의 진수. 관측 섹션은 서사적으로 읽어도 됨.

**English**:
- **Sections 1–2**: Start with observational facts — 5-min (photosphere) vs 3-min (chromosphere) umbral oscillations, velocity/intensity power spectra
- **Section 3**: Penumbral waves — running penumbral waves at 10–15 km/s propagating outward
- **Sections 4–5**: Magnetic field fluctuations (small, uncertain) + sunspot surroundings (acoustic halos, 40–60% enhancement at 5.5–7.5 mHz)
- **Section 7**: Theory core — homogeneous dispersion, eikonal approximation, mode conversion at β=1, Alfvén conversion, resonant absorption & cavity
- **Section 8**: Discussion — "What drives sunspot waves?" "Origin of 5→3 min transition?" "Mechanisms of power suppression/halo?"
- **Reading tip**: Focus on Section 7 equations — the true heart of this review. Observational sections can be read narratively.

---

## 7. 현대적 의의 / Modern Significance

**한국어**: 흑점 파동 연구는 태양 코로나 가열 문제, 태양 풍 가속, 공간 기상 예측의 선행 지식으로서 직접적 의의를 가진다. β=1 층에서의 fast→slow/Alfvén 변환은 채층 이상으로 에너지를 수송하는 주요 경로로 지목되며, 최근의 DKIST 같은 고분해능 관측기와 Bifrost 같은 radiative MHD 시뮬레이션이 이 리뷰에 제시된 이론 틀을 검증/확장하고 있다. 또한 일식 학(helioseismology)에서 흑점 아래 구조 탐사의 기반이며, 관측된 "absorption"을 mode conversion으로 해석하는 것은 이 논문이 정립한 표준 관점이다. 2020년대의 운영 우주 기상 모델(NASA/CCMC, NOAA/SWPC)도 흑점 파동 물리에 간접적으로 의존한다.

**English**: Sunspot wave research has direct relevance to the coronal heating problem, solar wind acceleration, and space weather prediction. The fast→slow/Alfvén conversion at the β=1 layer is identified as a key channel for energy transport above the chromosphere, and modern high-resolution instruments (DKIST) and radiative MHD simulations (Bifrost) are now validating and extending the theoretical framework established here. The review also underpins local helioseismology probing sub-sunspot structure; interpreting observed "absorption" as mode conversion is the standard view this paper consolidated. Operational space weather models in the 2020s (NASA/CCMC, NOAA/SWPC) depend indirectly on sunspot wave physics.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
