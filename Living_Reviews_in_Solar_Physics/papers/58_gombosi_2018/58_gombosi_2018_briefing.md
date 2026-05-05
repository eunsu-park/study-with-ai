---
title: "Pre-Reading Briefing: Extended MHD Modeling of the Steady Solar Corona and the Solar Wind"
paper_id: "58"
topic: Living_Reviews_in_Solar_Physics
date: 2026-04-23
type: briefing
---

# Extended MHD Modeling of the Steady Solar Corona and the Solar Wind: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Gombosi, T. I., van der Holst, B., Manchester, W. B., & Sokolov, I. V. (2018). "Extended MHD modeling of the steady solar corona and the solar wind." *Living Reviews in Solar Physics*, 15:4. DOI: 10.1007/s41116-018-0014-4
**Author(s)**: Tamas I. Gombosi, Bart van der Holst, Ward B. Manchester IV, Igor V. Sokolov (University of Michigan, Center for Space Environment Modeling)
**Year**: 2018

---

## 1. 핵심 기여 / Core Contribution

**English**: This *Living Review* is a comprehensive historical and technical survey of large-scale magnetohydrodynamic (MHD) modeling of the steady solar corona and solar wind. The authors trace the intellectual arc from Biermann's 1951 comet-tail inference of a continuous solar corpuscular radiation, through Parker's 1958 hydrodynamic solar-wind solution, to today's global, data-driven, three-temperature, Alfvén-wave-turbulence-driven simulations that connect the chromosphere to 1 AU. The central modern contribution is the **Alfvén Wave Solar Model (AWSoM)** developed within the **Space Weather Modeling Framework (SWMF)**, which self-consistently heats and accelerates the solar wind via dissipation of counter-propagating Alfvén-wave turbulence, removes ad-hoc heating functions, reproduces fast (≈700–800 km/s) and slow (≈300–450 km/s) wind bimodality, and provides realistic synthetic EUV images comparable to SDO/AIA and STEREO/EUVI.

**Korean / 한국어**: 이 *Living Review* 논문은 정상 상태 태양 코로나와 태양풍의 대규모 자기유체역학(MHD) 모델링에 관한 포괄적 역사적·기술적 총설이다. 저자들은 Biermann(1951)이 혜성 꼬리로부터 태양 입자 방사선 존재를 유추한 것부터 시작하여, Parker(1958)의 유체역학적 태양풍 해, 그리고 오늘날 색구~1 AU를 연결하는 글로벌·관측자료 기반·3-온도·알펜파 난류 구동 시뮬레이션까지 이어지는 지적 흐름을 추적한다. 핵심적 현대 기여는 **Space Weather Modeling Framework(SWMF)** 내에서 개발된 **Alfvén Wave Solar Model(AWSoM)** 로, 역방향 전파 알펜파 난류의 소산으로 태양풍을 자기정합적으로 가열·가속하여 임시 가열 함수를 제거하고, 고속(≈700–800 km/s)·저속(≈300–450 km/s) 바이모달 구조를 재현하며, SDO/AIA·STEREO/EUVI와 비교 가능한 합성 EUV 이미지를 제공한다.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

**English**: By 2018, solar-wind modeling had reached its third generation. The first generation (1960s–70s) consisted of 1-D spherically symmetric hydrodynamic/Navier–Stokes solutions (Scarf–Noble, Sturrock–Hartle, Whang–Chang). The second generation (late 1970s–90s) introduced 2-D axisymmetric and early 3-D MHD models driven by Potential Field Source Surface (PFSS) boundary conditions (Pneuman–Kopp, Steinolfson, Pizzo, Usmanov, Linker–Mikić). The third generation (post-2000) adds thermodynamic physics — radiative losses, Spitzer heat conduction, transition-region resolution — and Alfvén-wave-turbulence heating physics rather than ad-hoc polytropic equations of state. This review synthesizes all three eras and identifies **AWSoM** as the state of the art.

**Korean / 한국어**: 2018년까지 태양풍 모델링은 3세대에 도달했다. 1세대(1960–70년대)는 구대칭 1차원 유체역학/Navier–Stokes 해(Scarf–Noble, Sturrock–Hartle, Whang–Chang)로 구성되었다. 2세대(1970년대 후반~90년대)는 Potential Field Source Surface(PFSS) 경계조건으로 구동되는 2차원 축대칭 및 초기 3차원 MHD 모델(Pneuman–Kopp, Steinolfson, Pizzo, Usmanov, Linker–Mikić)이 도입되었다. 3세대(2000년 이후)는 복사 냉각, Spitzer 열전도, 천이영역 해상도와 같은 열역학 물리, 그리고 임시 폴리트로픽 상태방정식 대신 알펜파 난류 가열 물리를 추가하였다. 본 총설은 이 세 시기를 종합하며 **AWSoM**을 최첨단으로 제시한다.

### 타임라인 / Timeline

```
1859 ─ Carrington flare + geomagnetic storm
1892 ─ FitzGerald: corpuscular emission hypothesis
1931 ─ Chapman & Ferraro: plasma beam + magnetic dipole
1951 ─ Biermann: solar corpuscular radiation from comet tails
1957 ─ Alfvén: frozen-in field in solar outflow
1958 ─ Parker: "solar wind" — hydrodynamic expansion of hot corona
1960 ─ Chamberlain: "solar breeze" counter-hypothesis
1962 ─ Mariner 2: direct confirmation of solar wind
1966 ─ Sturrock & Hartle: first 2-fluid (Tp, Te) model
1969 ─ Altschuler & Newkirk: PFSS model (Rs = 2.5 R⊙)
1971 ─ Pneuman & Kopp: first 2-D MHD helmet streamer
1990s ─ First 3-D MHD heliosphere (Usmanov, Linker, Mikić)
2000 ─ Usmanov+: first 2-D Alfvén-wave-driven corona
2010 ─ van der Holst+: 2-T AWSoM
2014 ─ van der Holst+: 3-T AWSoM with anisotropic protons
2016 ─ Sokolov+: Threaded Field Line Model (TFLM) for TR
2018 ─ THIS REVIEW
```

---

## 3. 필요한 배경 지식 / Prerequisites

**English**:
- **Ideal MHD equations**: mass, momentum, energy, induction; frozen-in flux theorem
- **Plasma physics**: Alfvén waves, Alfvén speed $V_A = B/\sqrt{\mu_0 \rho}$, plasma beta $\beta = p/(B^2/2\mu_0)$
- **Parker wind solution**: hydrodynamic critical-point analysis, sonic/Alfvén/fast-mode critical points
- **Potential field theory**: Laplace's equation $\nabla^2\psi=0$, spherical-harmonic expansion, source surface at 2.5 R⊙
- **Radiative transfer**: optically thin cooling, $\Lambda(T)$ curve from CHIANTI
- **Spitzer heat conduction**: $\mathbf{q} = -\kappa_0 T_e^{5/2}\,\hat{\mathbf{b}}(\hat{\mathbf{b}}\cdot\nabla T_e)$
- **Turbulence theory**: Elsässer variables $\mathbf{z}_\pm$, WKB approximation, counter-propagating wave interaction, inertial range and cascade
- **Numerics**: finite volume, adaptive mesh refinement (AMR), time-stepping (explicit/implicit), block-structured grids

**Korean / 한국어**:
- **이상 MHD 방정식**: 질량, 운동량, 에너지, 유도 방정식; 자기장 동결 정리
- **플라즈마 물리**: 알펜파, 알펜 속도 $V_A = B/\sqrt{\mu_0 \rho}$, 플라즈마 베타 $\beta = p/(B^2/2\mu_0)$
- **Parker 태양풍 해**: 유체역학적 임계점 분석, 음속/알펜/빠른 모드 임계점
- **퍼텐셜 자기장 이론**: Laplace 방정식 $\nabla^2\psi=0$, 구면 조화 함수 전개, 2.5 R⊙ source surface
- **복사 전달**: 광학적 얇은 냉각, CHIANTI의 $\Lambda(T)$ 곡선
- **Spitzer 열전도**: $\mathbf{q} = -\kappa_0 T_e^{5/2}\,\hat{\mathbf{b}}(\hat{\mathbf{b}}\cdot\nabla T_e)$
- **난류 이론**: Elsässer 변수 $\mathbf{z}_\pm$, WKB 근사, 역방향 전파파 상호작용, 관성영역과 캐스케이드
- **수치해석**: 유한 부피, 적응 격자(AMR), 시간 적분(explicit/implicit), 블록 격자

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **AWSoM** (Alfvén Wave Solar Model) | SWMF 내 MHD 코로나-태양풍 모듈; 알펜파 난류를 가열/가속 기제로 사용 / MHD corona+wind module in SWMF driven self-consistently by Alfvén-wave turbulence |
| **SWMF** (Space Weather Modeling Framework) | 미시간 대학 CSEM이 개발한 다층 통합 프레임워크(코로나·내부 해류권·자기권 결합) / Multi-component framework from U. Michigan coupling corona, inner heliosphere, magnetosphere |
| **PFSS** (Potential Field Source Surface) | $\nabla^2\psi=0$ 해로 광구~Rs의 자기장을 계산; Rs=2.5 R⊙에서 장이 방사형이 되도록 설정 / Current-free solution with outer boundary at Rs=2.5 R⊙ forcing B radial |
| **Elsässer variables** $\mathbf{z}_\pm = \delta\mathbf{u} \mp \delta\mathbf{B}/\sqrt{\mu_0\rho}$ | 평행/역평행 전파 알펜파 진폭; 난류 계단 표현의 자연스러운 기저 / Amplitudes of parallel/antiparallel Alfvén waves; natural basis for turbulence |
| **Alfvén wave pressure** $p_A=(w_+ + w_-)/2$ | 등방 근사한 파동 압력; 태양풍 가속의 핵심 추진력 / Isotropic wave pressure; key driver of wind acceleration |
| **Two-temperature (2-T) model** | 양성자와 전자가 별개 에너지 방정식을 가짐($T_p\neq T_e$); Coulomb 충돌 약한 영역에서 필수 / Separate energy equations for protons and electrons; essential where Coulomb collisions are weak |
| **Three-temperature (3-T) model** | 양성자 $T_{\parallel}$, $T_{\perp}$ 및 전자 $T_e$의 세 온도 방정식; 이방성을 포착 / Captures proton parallel/perp anisotropy + electron |
| **TFLM** (Threaded Field Line Model) | 1-D "실"(thread) 집단으로 천이영역을 분석적으로 다루는 방법 / 1-D thread-based analytical treatment of TR |
| **Transition region (TR)** | 색구(≈2×10⁴ K)와 코로나(≈10⁶ K) 사이 급격한 온도 점프; 두께 ≈10 Mm / Sharp temperature jump layer, ~10 Mm thick |
| **Radiative loss function** $\Lambda(T)$ | 광학적 얇은 복사 냉각률; $Q_{\rm rad}=N_eN_i\Lambda(T_e)$ / CHIANTI-based cooling curve |
| **Spitzer–Härm conductivity** $\kappa_0 T^{5/2}$ | 전자 열 유속의 기본 식; 자기장을 따라서만 유효 / Field-aligned electron heat flux |
| **Poynting flux boundary** $\Pi_A/B\approx 1.1\times 10^6$ W m⁻² T⁻¹ | 광구에서 주입하는 알펜파 에너지 경계조건 / Imposed Alfvén wave energy input at photosphere |
| **Reflection coefficient** $\mathcal{R}$ | 알펜 속도 gradient에서의 파동 반사율; counter-propagating 파 생성 / Controls generation of counter-waves |
| **CME flux rope** | Titov–Démoulin, spheromak, Gibson–Low 모델로 삽입되는 휘인 자속관 / Twisted flux rope inserted as CME seed |

---

## 5. 수식 미리보기 / Equations Preview

**English**: Five equation families dominate this review:

**1. Ideal MHD (continuity, momentum, induction)**:

$$\frac{\partial \rho}{\partial t}+\nabla\cdot(\rho \mathbf{u})=0$$

$$\frac{\partial(\rho\mathbf{u})}{\partial t}+\nabla\cdot\!\left(\rho\mathbf{uu}-\frac{\mathbf{BB}}{\mu_0}+\mathbb{I}\left(p_i+p_e+\frac{B^2}{2\mu_0}+p_A\right)\right)=-\frac{GM_\odot\rho\,\mathbf{r}}{r^3}$$

$$\frac{\partial \mathbf{B}}{\partial t}+\nabla\cdot(\mathbf{uB}-\mathbf{Bu})=0$$

**2. Two-temperature energy equations** (AWSoM):

$$\frac{\partial}{\partial t}\!\left(\frac{p_e}{\gamma-1}\right)+\nabla\cdot\!\left(\frac{p_e}{\gamma-1}\mathbf{u}\right)+p_e\nabla\cdot\mathbf{u}=-\nabla\cdot\mathbf{q}_e+\frac{N_eN_ik_B}{\gamma-1}\frac{\nu_{ei}}{N_i}(T_i-T_e)-Q_{\rm rad}+Q_e$$

**3. Alfvén wave transport (Elsässer representation)**:

$$\frac{\partial w_\pm}{\partial t}+\nabla\cdot[(\mathbf{u}\pm\mathbf{V}_A)w_\pm]+\frac{w_\pm}{2}(\nabla\cdot\mathbf{u})=-\Gamma_\pm w_\pm \mp \mathcal{R}\sqrt{w_-w_+}$$

with dissipation $\Gamma_\pm=\frac{2}{L_\perp}\sqrt{w_\mp/\rho}$ where $L_\perp\propto B^{-1/2}$.

**4. Parker isothermal wind**:

$$\left[\frac{v^2}{v_m^2}-\ln\!\left(\frac{v^2}{v_m^2}\right)\right]=4\ln\!\left(\frac{r}{a}\right)+\left(\frac{v_{\rm esc}^2}{v_m^2}\right)\!\left(\frac{a}{r}\right)-4\left(\frac{v_{\rm esc}^2}{v_m^2}\right)-3+\ln 256$$

with $v_m^2=2kT_0/m_p$.

**5. Spitzer–Härm parallel heat flux**:

$$\mathbf{q}_e=-\kappa_\parallel\,\hat{\mathbf{b}}\,(\hat{\mathbf{b}}\cdot\nabla T_e),\qquad \kappa_\parallel\propto T_e^{5/2}$$

**Korean / 한국어**: 본 총설을 지배하는 다섯 가지 식 체계는 위와 같다: (1) 이상 MHD의 연속·운동량·유도 방정식에 알펜파 압력 $p_A$가 운동량 방정식에 추가된다. (2) 2-온도 에너지 방정식은 전자/양성자를 분리하며, Coulomb 충돌 항, 복사 냉각 $Q_{\rm rad}$, 알펜 가열 분배 $Q_e$/$Q_i$를 포함한다. (3) Elsässer 변수로 표현된 알펜파 전달 방정식은 감쇠율 $\Gamma_\pm$와 반사 계수 $\mathcal{R}$를 통해 난류 계단을 기술한다. (4) Parker 등온 해는 임계점 구조로 태양풍의 본질을 보인다. (5) Spitzer–Härm 열유속은 자기장 정렬 방향으로만 유효하다.

---

## 6. 읽기 가이드 / Reading Guide

**English**:
1. **Skim § 1–2** (Early ideas) for historical charm (Carrington, FitzGerald, Chapman, Biermann, Parker, Chamberlain).
2. **§ 3 (first numerical models)** is important: Scarf–Noble, Sturrock–Hartle 2-fluid, PFSS, Pneuman–Kopp. Read Eqs. (4)–(16) carefully.
3. **§ 4 (steady-state solar wind)** describes 2-D/3-D MHD models — note bimodal wind pattern and WSA-type boundaries.
4. **§ 5 (Alfvén wave turbulence)** is the technical heart. Master Eqs. (23)–(36), particularly the wave transport Eq. (31), dissipation $\Gamma_\pm$, and reflection $\mathcal{R}$.
5. **§ 6 (AWSoM)** puts it all together. Note the 3-T anisotropic proton version.
6. **§ 7 (TFLM)** is the latest innovation — 1-D threads that bypass expensive TR resolution.
7. **§ 8 (summary)** for conceptual wrap-up.

**Korean / 한국어**:
1. **§ 1–2(초기 아이디어)** 는 역사적 즐거움을 위해 훑어본다 (Carrington, FitzGerald, Chapman, Biermann, Parker, Chamberlain).
2. **§ 3(최초 수치 모델)** 은 중요: Scarf–Noble, Sturrock–Hartle 2-fluid, PFSS, Pneuman–Kopp. Eq. (4)–(16)을 주의 깊게 읽는다.
3. **§ 4(정상 태양풍)** 는 2차원/3차원 MHD 모델을 다룬다 — 이중 태양풍 및 WSA 경계에 주목.
4. **§ 5(알펜파 난류)** 는 기술적 핵심. Eq. (23)–(36), 특히 전달 방정식 (31), 감쇠 $\Gamma_\pm$, 반사 $\mathcal{R}$를 숙달한다.
5. **§ 6(AWSoM)** 은 모든 것을 통합한다. 3-T 이방성 양성자 버전에 주목.
6. **§ 7(TFLM)** 은 최신 혁신 — 비싼 천이영역 해상도를 우회하는 1-D thread.
7. **§ 8(요약)** 으로 개념을 마무리한다.

---

## 7. 현대적 의의 / Modern Significance

**English**: AWSoM and its successors are now the operational backbone for space-weather forecasting at CCMC and at ESA's VSWMC. The same machinery is used to predict CME arrival at Earth (e.g., for the September 2017 events and ongoing Parker Solar Probe/Solar Orbiter campaigns). Extended-MHD corona/wind models are the high-resolution backbone that feeds inner-heliosphere codes (ENLIL, EUHFORIA, SWMF-IH). They also serve as baseline models for interpreting Parker Solar Probe in-situ data in the previously inaccessible inner heliosphere (≤ 20 R⊙), and are being adapted to model other stars' winds (e.g., TRAPPIST-1 host-star winds for exoplanet habitability studies).

**Korean / 한국어**: AWSoM과 그 후속 모델은 현재 NASA CCMC와 ESA VSWMC의 우주 날씨 예보 운영 뼈대이다. 동일한 엔진이 지구 도달 CME 예측(예: 2017년 9월 사건, 진행 중인 Parker Solar Probe/Solar Orbiter 캠페인)에 사용된다. 확장 MHD 코로나/태양풍 모델은 내부 해류권 코드(ENLIL, EUHFORIA, SWMF-IH)에 공급되는 고해상도 상위 모델이며, 이전에 접근 불가능했던 내부 해류권(≤ 20 R⊙)의 Parker Solar Probe 관측 해석을 위한 기준 모델이고, 외계 행성 거주가능성 연구(예: TRAPPIST-1 모항성풍)를 위한 다른 항성의 항성풍 모델링에도 응용되고 있다.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
