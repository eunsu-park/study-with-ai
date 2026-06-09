---
title: "The Solar Wind as a Turbulence Laboratory"
authors: [Roberto Bruno, Vincenzo Carbone]
year: 2013
journal: "Living Reviews in Solar Physics"
doi: "10.12942/lrsp-2013-2"
topic: Living_Reviews_in_Solar_Physics
tags: [turbulence, MHD, solar_wind, Kolmogorov, Iroshnikov-Kraichnan, Elsasser, intermittency, Yaglom, inertial_range, spectral_break]
status: completed
date_started: 2026-04-23
date_completed: 2026-04-23
---

# 32. The Solar Wind as a Turbulence Laboratory / 태양풍: 거대한 난류 실험실

---

## 1. Core Contribution / 핵심 기여

Bruno & Carbone (2013)은 태양풍을 "천연 난류 실험실 (natural wind tunnel)"로 간주하여, Mariner (1962)부터 Helios (1974-81), Voyager (1977-), Ulysses (1990-2008), ACE/Wind (1997-), Cluster (2000-)에 이르는 40여 년간 in-situ 탐사선 자료와 MHD 난류 이론, 수치 시뮬레이션, 동역학 시스템 이론의 주요 결과를 통합한 종합 리뷰이다. 이 업데이트판은 2005년 원판 (lrsp-2005-4)에 비해 Yaglom 법칙 관측, 소규모 소산/분산 영역, KAW/whistler 시나리오, 스펙트럼 분절에 대한 절이 대대적으로 보강되었으며 참고문헌이 296편에서 439편으로 확장되었다. 저자들은 Kolmogorov (1941) K41 현상학과 Iroshnikov (1963)-Kraichnan (1965)의 MHD 현상학을 토대로, 관성영역(inertial range) 스펙트럼 지수의 측정값 (자기장 $\sim -5/3$, 속도 $\sim -3/2$), Elsässer 변수 $\mathbf{z}^{\pm} = \mathbf{v} \pm \mathbf{b}/\sqrt{4\pi\rho}$로 본 Alfvén파 상관의 반경 진화, 구조 함수의 비정상 스케일링과 다중분산(multifractal) 간헐성, Politano-Pouquet의 MHD Yaglom 법칙, 이온 사이클로트론 근처 ($\sim 0.3$ Hz at 1 AU)의 스펙트럼 분절과 소규모 플라즈마 물리를 체계적으로 제시한다.

Bruno & Carbone (2013) treat the solar wind as a giant natural wind tunnel and synthesize four decades of in-situ spacecraft data (Mariner, Helios, Voyager, Ulysses, ACE, Wind, Cluster) with MHD turbulence theory, numerical simulations, and dynamical-systems ideas. This 2013 update of their 2005 *Living Reviews* article adds substantial new material on Yaglom's law observations, the dissipative/dispersive small-scale range, KAW and whistler scenarios, and the spectral break, growing the reference count from 296 to 439. Anchored in Kolmogorov (K41) and Iroshnikov-Kraichnan (IK) phenomenology, the review covers: measured inertial-range slopes (magnetic spectra $\sim -5/3$, velocity $\sim -3/2$ at 1 AU), the radial evolution of Alfvénic correlations via Elsässer variables $\mathbf{z}^{\pm} = \mathbf{v} \pm \mathbf{b}/\sqrt{4\pi\rho}$, anomalous scaling of structure functions and multifractal intermittency, Politano-Pouquet's MHD Yaglom law, and the ion-cyclotron spectral break around $\sim 0.3$ Hz at 1 AU beyond which small-scale kinetic physics (KAW, whistlers, Landau damping) dominates.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction & Equations / 서론과 방정식 (Chs 1-2)

**What does turbulence stand for? / 난류란 무엇인가?**
저자들은 turbulent (라틴어 *turba* = 혼란)라는 단어의 어원에서 출발하여, Reynolds 수 $Re = UL\rho/\eta$가 임계값 $Re \sim 4000$을 초과할 때 층류-난류 천이가 일어난다는 Reynolds (1883)의 고전적 결과를 소개한다. 태양풍은 Reynolds 수가 매우 높은 극한 상태 ($Re_{\rm eff} \sim 10^3 - 10^5$로 추정)에서 존재하므로 "완전히 발달한 난류 (fully developed turbulence)"의 이상적 실험장이다.

The authors begin with etymology (*turba* = disorder) and recall Reynolds' experimental demonstration that flow in a pipe transitions from laminar to turbulent when $Re = UL\rho/\eta$ exceeds ~4000. The solar wind, with effective Reynolds numbers estimated at $10^3$-$10^5$, is an ideal laboratory for fully developed turbulence since all dynamically relevant scales are excited.

**Dynamics vs. statistics / 동역학 대 통계학**
개별 난류 실현(realization)의 세부사항은 초기 조건에 극도로 민감하지만(Lorenz 1963), 앙상블 혹은 시간 평균은 재현 가능한 통계적 성질을 가진다. 우주에서는 탐사선이 고정점에서 시간 평균을 수행하며, Taylor의 "frozen-in" 가설 ($V_{\rm sw} \gg \delta v, c_A$)에 의해 주파수 축 $f$와 파수 축 $k$가 $k = 2\pi f / V_{\rm sw}$로 연결된다. Helios 2의 0.9 AU 고속풍 샘플 (Figure 9)은 대규모 구조(큰 스케일)와 자기 강도 변동의 자기 유사성(self-similarity)을 한 달/하루/한 시간 배율에서 시각적으로 보여준다 (Figure 11).

Details of individual realizations are initial-condition-sensitive, but ensemble/time averages exhibit reproducible statistics. Spacecraft rely on Taylor's frozen-in hypothesis ($V_{\rm sw} \gg \delta v, c_A$) to convert frequency to wavenumber via $k = 2\pi f/V_{\rm sw}$. Helios 2 samples at 0.9 AU (Figure 9) demonstrate self-similarity across month/day/hour zooms (Figure 11).

**Navier-Stokes and MHD / NS 및 MHD**
비압축 NS 방정식은 (Eq. 4): $\partial_t \mathbf{u} + (\mathbf{u}\cdot\nabla)\mathbf{u} = -\nabla p/\rho + \nu \nabla^2 \mathbf{u}$. 무차원화하면 $Re^{-1}$이 유일한 파라미터가 된다. 자화된 플라즈마에서 Lorentz 힘 $\mathbf{j}\times\mathbf{B}$와 유도 방정식 $\partial_t \mathbf{B} = \nabla\times(\mathbf{u}\times\mathbf{B}) + (c^2/4\pi\sigma)\nabla^2 \mathbf{B}$가 추가된다. 비압축 MHD는 대칭적 Elsässer 형식으로 쓰인다 (Eq. 15):
$$\partial_t \mathbf{z}^{\pm} \mp (\mathbf{c}_A\cdot\nabla)\mathbf{z}^{\pm} + (\mathbf{z}^{\mp}\cdot\nabla)\mathbf{z}^{\pm} = -\nabla P_{\rm tot} + \nu^+ \nabla^2 \mathbf{z}^{\pm} + \nu^- \nabla^2 \mathbf{z}^{\mp}$$
여기서 비선형항 $(\mathbf{z}^{\mp}\cdot\nabla)\mathbf{z}^{\pm}$은 반대 방향으로 전파하는 Alfvén파만 결합시킨다 — 이것이 MHD 난류의 핵심 비대칭성이다.

The incompressible MHD equations cast in Elsässer form (Eq. 15) show that nonlinear coupling happens *only* between counter-propagating fluctuations $\mathbf{z}^+$ and $\mathbf{z}^-$. This is a fundamental departure from fluid NS, with deep consequences for the cascade.

**Cascade phenomenology / 캐스케이드 현상학**
K41 차원해석: 관성영역에서 에너지 주입률 $\varepsilon \sim (\Delta u_\ell)^2 / t_\ell$이 일정해야 하므로 $t_\ell \sim \ell/\Delta u_\ell$ (eddy-turnover time)을 대입하면 $\Delta u_\ell \sim \varepsilon^{1/3} \ell^{1/3}$ (Eq. 26)과 $E(k) \sim \varepsilon^{2/3} k^{-5/3}$ (Eq. 27)이 유도된다. Iroshnikov-Kraichnan 현상학: Alfvén파가 서로를 쓸고 지나가므로 유효 전달 시간이 $T_\ell^{\pm} \sim (t_\ell^{\pm})^2/t_A$로 길어지고 $\Delta z_\ell^{\pm} \sim (\varepsilon c_A)^{1/4}\ell^{1/4}$ (Eq. 28), $E^{\pm}(k) \sim (\varepsilon c_A)^{1/2} k^{-3/2}$ (Eq. 29)가 된다.

K41 derivation: constancy of $\varepsilon$ plus eddy-turnover time gives $\Delta u_\ell \sim \varepsilon^{1/3}\ell^{1/3}$ and $E(k)\sim k^{-5/3}$. IK derivation: Alfvén sweeping stretches transfer time, yielding $\Delta z_\ell^{\pm}\sim\ell^{1/4}$ and $E(k)\sim k^{-3/2}$. Goldreich-Sridhar's (1995) critical-balance refinement predicts $k_\perp^{-5/3}$ perpendicular and $k_\parallel^{-2}$ parallel anisotropic spectra.

**Kolmogorov 4/5 law and MHD Yaglom law / Kolmogorov 4/5 법칙과 MHD Yaglom 법칙**
NS 방정식에서 직접 유도되는 유일한 엄밀한 결과 (Kolmogorov 1941):
$$\langle (\Delta v_\parallel(\ell))^3 \rangle = -\frac{4}{5}\varepsilon\, \ell \quad \text{(Eq. 30)}$$
저자들은 Politano-Pouquet (1998)의 MHD 일반화도 유도한다 (Eqs 31-40):
$$Y_\ell^{\pm} \equiv \langle \Delta z_\ell^{\mp} |\Delta \mathbf{z}^{\pm}|^2 \rangle = -\frac{4}{3}\varepsilon^{\pm}\, \ell \quad \text{(Eq. 40)}$$
음의 부호는 에너지가 작은 스케일로 흐르는 직접 캐스케이드(direct cascade)를 의미한다. 이는 Gaussian이 아닌 위상 상관을 반드시 요구하며, 관성영역의 엄밀한 조작적 정의가 된다.

The only exact non-trivial result for NS turbulence. Politano-Pouquet's MHD analog (Eq. 40) provides an operative definition of the inertial range: the negative sign encodes direct cascade toward small scales and requires non-Gaussian phase correlations.

### Part II: Ecliptic Observations / 황도면 관측 (Ch. 3)

**Spectral properties / 스펙트럼 특성**
Coleman (1968)이 Mariner 2 (1962, 0.87-1.0 AU) 데이터로 최초의 저주파 스펙트럼을 측정하여 $f^{-1.2}$ 거듭제곱을 보고했다 — Kraichnan의 $f^{-3/2}$ 기대와의 차이는 이온 garden-hose 불안정성에서 생성된 고주파 횡파 때문으로 추정되었다. Russell (1972)은 Mariner/OGO 조합 스펙트럼을 세 영역으로 구분했다: (i) $f \lesssim 10^{-4}$ Hz에서 $1/f$, (ii) $10^{-4} \lesssim f \lesssim 10^{-1}$ Hz의 "중간 영역"에서 $f^{-3/2}$, (iii) 고주파에서 $f^{-2}$. 이후 Podesta et al. (2007)이 WIND 데이터를 이용해 놀랍게도 **속도 스펙트럼은 $-3/2$, 자기장 스펙트럼은 $-5/3$**임을 입증하여, 두 이론의 예상과 반대되는 비대칭이 발견됐다. Salem et al. (2009)는 wavelet 기법으로 대규모 구조를 제거하면 speed와 B 스펙트럼 지수가 각각 $p/4$(속도)와 $p/3$(자기장)의 선형 스케일링으로 수렴함을 보였다.

Coleman (1968) first reported $f^{-1.2}$; Russell (1972) refined to three ranges with intermediate $f^{-3/2}$. The surprising modern result (Podesta et al. 2007; Figure 22): **velocity spectra ~$-3/2$, magnetic spectra ~$-5/3$** — *opposite* to what K41 and IK individually predict. Salem et al. (2009) showed via wavelets that removing rare large structures recovers linear scaling exponents $p/4$ and $p/3$ for kinetic and magnetic fields.

**Alfvénic correlations / Alfvén 상관**
Belcher & Davis (1971)가 최초로 빠른 태양풍에서 $\mathbf{v}$와 $\mathbf{b}$가 상관되어 있음을 발견: $\delta\mathbf{v} \approx \mp \delta\mathbf{b}/\sqrt{4\pi\rho}$ (부호는 $\mathbf{B}_0$의 극성에 따름), 즉 주로 바깥쪽으로 전파하는 Alfvén파. Elsässer 변수로 본 radial evolution (Helios 관측, 0.3-1 AU):
- 빠른 바람: $\sigma_c$가 0.3 AU에서 $+0.8-0.9$로 매우 높으며 1 AU에서 $+0.5$까지 완만히 감소.
- 느린 바람: $\sigma_c \approx 0$이 공통 특징 — 내향 및 외향 모드가 비슷한 비율로 존재, 이는 비선형 상호작용이 활발함을 의미.
- 잔차 에너지 $\sigma_r$: 1 AU에서 $\sim -0.4$ (자기 에너지 과잉), 반경에 따라 점점 음으로 커짐.

Belcher & Davis (1971) first established Alfvénic correlation in fast wind. Radial evolution via Helios: fast wind $\sigma_c$ drops from $+0.8$-$0.9$ at 0.3 AU to $+0.5$ at 1 AU; slow wind has $\sigma_c \approx 0$; residual energy $\sigma_r \approx -0.4$ with magnetic energy in excess.

**Spectral anisotropy / 스펙트럼 이방성**
Matthaeus et al. (1990)의 "Maltese cross" 분석은 태양풍 난류가 순수 2D도 순수 slab도 아닌 혼합임을 보였다: 관성영역에서 약 $\sim 80\%$ 2D + $\sim 20\%$ slab. 이는 Goldreich-Sridhar critical balance 이론의 핵심 예측 $k_\perp \gg k_\parallel$과 정성적으로 일치한다.

Matthaeus' "Maltese cross" showed ~80% 2D + ~20% slab partition, qualitatively consistent with Goldreich-Sridhar critical balance favoring $k_\perp \gg k_\parallel$.

### Part III: Polar Wind & Compressive Turbulence / 극풍과 압축성 난류 (Chs 4-6)

Ulysses 관측 (1990-2008)은 태양 활동 최소기에 $|b|>30°$ 고위도에서 안정적인 빠른 태양풍(~750 km/s)을 발견. 극풍은 황도면 빠른 바람보다 더 강한 Alfvén성 ($\sigma_c \sim 0.8$ at 2 AU)을 지속적으로 유지. 다만 반경 방향 진화는 매우 느려 ~5 AU에도 $\sigma_c > 0.5$. 압축 난류(density/pressure fluctuations)는 대개 $\delta\rho/\rho \lesssim 10\%$로 작지만, 공회전 상호작용 영역(Co-rotating Interaction Region, CIR)에서는 충격파와 관련하여 증폭된다. 밀도로 가중된 Elsässer 변수 $\mathbf{w}^{\pm} \equiv \rho^{1/3}\mathbf{z}^{\pm}$는 Yaglom 관계를 압축 MHD로 일반화한다 (Eq. 47, Carbone et al. 2009).

Ulysses polar observations: sustained Alfvénicity ($\sigma_c \sim 0.8$ at 2 AU), slow radial evolution, persistent cross-helicity out to ~5 AU. Compressive fluctuations are typically small ($\delta\rho/\rho \lesssim 10\%$) but amplified within CIRs. Density-weighted Elsässer variables $\mathbf{w}^{\pm} \equiv \rho^{1/3}\mathbf{z}^{\pm}$ (Carbone et al. 2009) extend Yaglom to compressible cases.

### Part IV: Intermittency & Natural Wind Tunnel / 간헐성과 천연 풍동 (Ch. 7, 9-10)

**Structure-function scaling / 구조 함수 스케일링**
$p$차 구조 함수 $S_p(\tau) = \langle |\delta u_\tau|^p \rangle$는 관성영역에서 거듭제곱 법칙 $S_p(\tau) \sim \tau^{\zeta_p}$을 따른다. K41 예측은 $\zeta_p = p/3$ (선형), IK는 $\zeta_p = p/4$. **Helios 2 데이터에서 측정된 실제 값 (Table 1)**:

| p | 속도 $\zeta_p$ (slow wind 0.9 AU) | 자기장 $\xi_p$ | 유체 실험 비교 |
|---|---|---|---|
| 1 | 0.37 ± 0.06 | 0.56 ± 0.06 | 0.37 |
| 2 | 0.70 ± 0.05 | 0.83 ± 0.05 | 0.70 |
| 3 | 1.00 | 1.00 | 1.00 |
| 4 | 1.28 ± 0.02 | 1.14 ± 0.02 | 1.28 |
| 5 | 1.54 ± 0.03 | 1.25 ± 0.03 | 1.54 |
| 6 | 1.79 ± 0.05 | 1.35 ± 0.05 | 1.78 |

주요 관찰: (i) $p < 3$에서는 $\zeta_p/\zeta_3 > p/3$, $p > 3$에서는 $\zeta_p/\zeta_3 < p/3$인 **비정상 스케일링(anomalous scaling)** — 간헐성의 지문. (ii) 속도 지수는 지상 풍동 값(Ruíz-Chavarría et al. 1995)과 거의 동일 → 일종의 보편성(universality). (iii) **자기장은 속도보다 간헐성이 더 강하다** ($\xi_p/\xi_3$가 $p/3$에서 더 크게 벗어남).

Key findings: (i) anomalous non-linear scaling $\zeta_p/\zeta_3 \neq p/3$ — the signature of intermittency; (ii) velocity exponents closely match wind-tunnel results, suggesting universality between solar wind and laboratory fluids; (iii) the magnetic field is *more* intermittent than velocity.

**Extended Self-Similarity (ESS)**
Benzi et al. (1993)의 ESS 기법: $S_p(r)$를 $\tau$가 아닌 $S_3(r)$의 함수로 plot하면 선형 영역이 관성영역을 크게 넘어 확장되어 상대 스케일링 지수 $\zeta_p/\zeta_3$를 저 Reynolds 수에서도 정확히 뽑아낼 수 있다. 이는 $\zeta_3 = 1$ (Yaglom 법칙)이 어떤 extension이든 성립하기 때문이다. ESS는 이후 태양풍 간헐성 분석의 표준 도구가 되었다.

Benzi's ESS: plotting $S_p$ vs. $S_3$ extends linear scaling beyond the nominal inertial range since $\zeta_3 = 1$ is exact. ESS became the standard diagnostic.

**PDFs and multifractal models / PDF와 다중분산 모델**
증분 PDF $P(\delta u_\tau)$는 큰 스케일에서 거의 Gaussian이지만 $\tau$가 작아질수록 stretched exponential/heavy tails로 변모한다 (Figure 82). 이는 "글로벌 자기 유사성의 붕괴"이자, 다중분산(multifractal) 서술로 복원된다: 국소 Hölder 지수 $h$에 의존하는 singularity set $S_h$의 프랙탈 차원 $D(h)$로 $\zeta_p = \min_h [ph + 3 - D(h)]$.

대표적 다중분산 모델 (저자들이 논의하는 주요 후보):
- **β-모델** (Frisch et al. 1978): 선형 수정, 단일 공간 채움 지수.
- **p-모델** (Meneveau 1991; Carbone 1993): $\zeta_p = 1 - \log_2[\mu^{p/m} + (1-\mu)^{p/m}]$ — 이중 스케일 Cantor 집합 모델.
- **She-Leveque** (1994): 로그-포아송 계층, $\zeta_p = (p/m)(1-x) + C[1-(1-x/C)^{p/m}]$, MHD에서 $C=1$ (가장 특이한 구조가 평면 sheet), fluid에서 $C=2$ (filaments).

세 모델 모두 관측과 동등하게 잘 맞아 현재 데이터로 구별 불가 — "통계는 증명이 아닌 반증(disprove)만 가능."

PDFs evolve from near-Gaussian at large scales to stretched-exponential at small scales, signaling broken global self-similarity (multifractal). Three main models (β, p, She-Leveque) all fit the data; current statistics can disprove but not discriminate between them.

### Part V: Yaglom Law Observations / Yaglom 법칙 관측 (Ch. 8)

Sorriso-Valvo et al. (2007)은 Ulysses 극풍 데이터 (1996, 3-4 AU, 55°→30° 위도)에서 **선형 스케일링** $Y_\ell^{\pm} \propto \ell$을 20배율 이상에서 확인. 주목할 점: (a) 고 Alfvén성 극풍에서도 비선형 캐스케이드가 활성 (대부분의 기존 현상학은 순수 $\sigma_c = \pm 1$에서 캐스케이드 부재 예측). (b) 에너지 소산률 $\varepsilon \sim$ 수백 J/(kg·s)이며 지상 유체 ($1$-$50$ J/(kg·s))보다 크다. (c) 부호 변경(+ ↔ -)이 ~1 day 스케일에서 발생, Alfvén 상관이 무너지는 척도에 해당. MacBride et al. (2008, 2010)이 ACE 데이터로 황도면 1 AU에서 $\varepsilon \simeq 1.22 \times 10^4$ J/(kg·s)로 평가, 이는 **Kolmogorov 법칙으로 추정한 태양풍 가열률($\sim 10^3$-$10^4$)과 일치** — 태양풍의 이상적-가스 이탈 온도 감쇠 ($T \sim r^{-0.7}$)를 난류 캐스케이드로 설명 가능함을 실증.

Sorriso-Valvo et al. (2007) verified linear Yaglom scaling over 20 decades in Ulysses polar wind; MacBride et al. (2008, 2010) at 1 AU found $\varepsilon \simeq 1.22 \times 10^4$ J/(kg·s), matching heating rates needed to explain the slower-than-adiabatic $T \sim r^{-0.7}$ profile.

### Part VI: Solar Wind Heating & Small Scales / 태양풍 가열과 소규모 (Chs 11-13)

**Heating puzzle / 가열 문제**
Parker (1964) 이론은 단열 팽창 $T \sim r^{-4/3}$를 예측하지만 실제 Helios/Voyager 관측은 $T \sim r^{-\xi}$, $\xi \in [0.7, 1]$으로 훨씬 느린 감쇠. Verma et al. (1995)의 열 방정식 (Eq. 74):
$$\frac{dT(r)}{dr} + \frac{4}{3}\frac{T(r)}{r} = \frac{m_p \varepsilon}{(3/2)V_{\rm sw}(r) k_B}$$
는 국소 가열률 $\varepsilon(r) = (3/2)(4/3 - \xi)V_{\rm sw}(r)k_B T(r)/(r m_p)$ (Eq. 75)를 산출. 1 AU 기준 $\sim 10^2$-$10^4$ J/(kg·s)로 MHD 캐스케이드가 공급하는 에너지와 일치.

**Spectral break near ion-cyclotron / 이온 사이클로트론 근처 스펙트럼 분절**
Leamon et al. (1998)이 Wind 자기장 데이터에서 **1 AU에서 $f_{\rm br} \simeq 0.44$ Hz** ($f_{ci} \simeq 0.1$ Hz 근처)의 분절을 발견. 분절 이하는 $f^{-5/3}$, 이상은 $f^{-\alpha}$, $\alpha \in [2, 4]$ (전형적 $\alpha \simeq 7/3$). Alexandrova et al. (2008)은 Cluster로 $f_{\rm br} \simeq 0.3$ Hz 확인, 공간 스케일로 $\sim 1900$ km $\sim 15 \lambda_i$ (이온 관성 길이의 15배).

**소규모 해석 시나리오 (미해결)** / Unresolved scenarios:
- **Whistler 모드**: 우측 편광 magnetosonic 모드의 캐스케이드. 약 감쇠.
- **Kinetic Alfvén Waves (KAW)**: 준수직 전파로 변환된 Alfvén 파가 양자 Landau 감쇠로 소산 ($k_\perp \rho_i \sim 1$ 스케일).
- **Hall-MHD**: $E(k) \sim k^{-7/3+r}$, $r$은 압축 정도 (Eq. 78). 관측된 $\alpha \in [2,4]$는 $r \in [-5/6, 1/6]$로 설명.

탐사선 관측만으로는 KAW와 whistler 시나리오를 구별 불가. Sahraoui et al. (2009)은 전자 스케일 근처($k_\perp \rho_e \sim 1$, ~수십 Hz)에서 두 번째 분절과 지수 감쇠 $\exp[-\sqrt{k\rho_e}]$를 보고, 이는 완전한 소산의 시작으로 해석.

Spectral break at ~0.3-0.5 Hz (at 1 AU) near $f_{ci}$; steeper slope beyond (typically $-7/3$). Mechanisms debated: Whistler cascade, KAW, Hall-MHD. Sahraoui et al. (2009): second break at electron scale with $\exp[-\sqrt{k\rho_e}]$ cutoff.

**Intermittency at small scales / 소규모 간헐성**
Alexandrova et al. (2008)은 자기 증분의 4차 모멘트(flatness) $K(f)$를 조사, $f < f_{ci}$에서 $K \sim 1-10$ (온화한 간헐성), $f > f_{ci}$에서 $K$가 급격히 증가 (1000 이상) 하여 **소규모 간헐성이 관성영역보다 훨씬 강하다**. Kiyani et al. (2009)의 고차 통계는 소규모에서 오히려 스케일링 지수가 단일 프랙탈로 회귀한다고 보고 — 보편성 부재의 단서.

Small-scale flatness $K(f)$ rises sharply beyond $f_{ci}$ to >1000 (Alexandrova et al. 2008); Kiyani et al. (2009) report scaling becomes mono-fractal at small scales — a hint of non-universality.

---

## 3. Key Takeaways / 핵심 시사점

1. **태양풍 = 자연 MHD 난류 실험실 / Solar wind as a natural MHD turbulence laboratory** — 지상 풍동 대비 $\sim 10^9$배 더 큰 스케일 분리를 제공하며 관성영역이 수 decade에 걸쳐 명확히 관측된다. 지상 실험에서 불가능한 Alfvén 상관, cross-helicity의 반경 진화, 양극 풍 관측 등 독특한 현상을 제공.
   The solar wind offers scale separation ~$10^9$× greater than laboratory wind tunnels, with an inertial range spanning several decades, and provides access to unique phenomena like Alfvénic correlations, $\sigma_c$ radial evolution, and polar turbulence that ground-based experiments cannot reach.

2. **관측 스펙트럼 지수의 이중성 / Dual spectral indices** — 1 AU에서 자기장 스펙트럼은 Kolmogorov $-5/3$, 속도 스펙트럼은 Iroshnikov-Kraichnan $-3/2$를 따른다 (Podesta et al. 2007). 이는 어느 단일 현상학도 완벽하지 않음을 의미하며 현재도 이론적 도전 과제이다.
   At 1 AU, magnetic-field spectra follow Kolmogorov's $-5/3$ while velocity spectra follow IK's $-3/2$ (Podesta et al. 2007) — neither theory alone captures the behavior, and reconciling this remains an open theoretical challenge.

3. **Elsässer 변수가 Alfvén 반경 진화를 해부 / Elsässer variables reveal Alfvénic radial evolution** — $\mathbf{z}^{\pm} = \mathbf{v}\pm\mathbf{b}/\sqrt{4\pi\rho}$로 분해하면 빠른 바람은 ($\sigma_c$: 0.3 AU에서 0.9 → 1 AU에서 0.5)로 외향 Alfvén파가 지배적, 느린 바람은 $\sigma_c \approx 0$으로 이미 "발달된" 난류 상태. 잔차 에너지 $\sigma_r \approx -0.4$는 자기 에너지 과잉 (원인은 미해결).
   Elsässer decomposition reveals fast wind as outward Alfvén-dominated ($\sigma_c$: 0.9 at 0.3 AU → 0.5 at 1 AU), slow wind as evolved turbulence ($\sigma_c \approx 0$); the $\sigma_r \approx -0.4$ magnetic-energy excess remains unexplained.

4. **간헐성은 보편적이고 자기장이 속도보다 강하다 / Intermittency is universal and stronger for B** — 구조 함수 스케일링 지수 $\zeta_p$의 비선형성은 태양풍, 지상 유체, 실험실 플라즈마에서 동일한 형태로 관측됨. 그러나 태양풍 자기장은 속도 (그리고 passive scalar)보다 더 강한 간헐성을 보임 — 자기장이 passive field가 아닌 능동적 역할을 시사.
   Anomalous scaling $\zeta_p/\zeta_3$ deviates universally from $p/3$ across solar wind, wind tunnels, and laboratory plasmas, but solar-wind magnetic fields are *more* intermittent than velocity, suggesting B is not passive.

5. **Yaglom 법칙이 관성영역을 엄밀히 정의하고 가열률을 제공 / Yaglom law provides rigorous inertial range and heating rates** — Politano-Pouquet (1998)의 $Y_\ell^{\pm} = -\frac{4}{3}\varepsilon^{\pm}\ell$은 Sorriso-Valvo et al. (2007)이 Ulysses 극풍에서, MacBride et al. (2008, 2010)이 ACE 황도면 1 AU에서 검증. $\varepsilon \sim 10^3-10^4$ J/(kg·s)는 관측된 이상 온도 감쇠 $T \sim r^{-0.7}$를 설명하는 난류 가열률과 일치.
   Politano-Pouquet's $Y_\ell^{\pm}=-\frac{4}{3}\varepsilon^{\pm}\ell$ has been verified in polar (Ulysses) and ecliptic (ACE) wind, yielding $\varepsilon \sim 10^3$-$10^4$ J/(kg·s) — consistent with the heating needed to explain the anomalous $T\sim r^{-0.7}$ profile.

6. **이온 사이클로트론 스펙트럼 분절의 미스터리 / The ion-cyclotron spectral break mystery** — 1 AU에서 $f_{\rm br} \simeq 0.3$-0.5 Hz (공간 스케일 $\sim 15\lambda_i$) 이상에서 스펙트럼이 $-5/3$에서 $-7/3$으로 급격히 기울어진다. Whistler 캐스케이드, KAW, Hall-MHD 세 시나리오가 경쟁하나 탐사선 데이터로 구별 불가. Perri et al. (2011)은 분절 위치가 반경에 거의 무관함을 보고, 통념의 이온 자이로 척도 해석에 의문을 제기.
   At 1 AU, spectra steepen sharply from $-5/3$ to $\sim-7/3$ around $f_{\rm br}\simeq0.3$-$0.5$ Hz (~$15\lambda_i$); the whistler, KAW, and Hall-MHD scenarios remain indistinguishable, and Perri et al. (2011) find the break position radially independent — challenging ion-gyroscale interpretations.

7. **다중분산(multifractal) 모델들의 구별 불가능성 / Indistinguishable multifractal models** — β-모델, p-모델, She-Leveque 모델이 모두 관측 구조 함수와 동등하게 잘 맞으며 현재 통계 정밀도로는 식별 불가. 이는 "통계는 이론을 증명할 수 없고, 반증만 할 수 있다"는 원칙의 구체 사례이며 추후 더 긴 시계열이 필요.
   β-model, p-model, and She-Leveque all fit observations equally well; current statistics cannot disprove any. Illustrates the adage that statistics can only disprove, not confirm.

8. **소규모에서의 보편성 상실 / Loss of universality at small scales** — Kiyani et al. (2009)는 $f > f_{ci}$ 영역에서 스케일링 지수가 단일 프랙탈로 회귀함을 발견 (관성영역의 강한 간헐성과 대조). Alexandrova et al. (2008)의 $K(f)$는 오히려 소규모에서 급증. 해석은 여전히 논쟁적.
   Kiyani et al. (2009) find scaling becomes mono-fractal at $f > f_{ci}$ while Alexandrova et al. (2008) observe drastic kurtosis rise — a yet-unresolved tension hinting at non-universal small-scale physics.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Kolmogorov K41 phenomenology / K41 현상학

에너지 주입률 $\varepsilon$가 일정하다는 가정과 eddy-turnover time $t_\ell \sim \ell / \Delta u_\ell$로부터:
$$\varepsilon \sim \frac{(\Delta u_\ell)^2}{t_\ell} \sim \frac{(\Delta u_\ell)^3}{\ell} \implies \Delta u_\ell \sim \varepsilon^{1/3} \ell^{1/3}$$
스펙트럼 형태: $E(k) \cdot dk$의 차원이 $(\Delta u)^2$, $k \sim 1/\ell$을 이용해
$$\boxed{E(k) = C_K \varepsilon^{2/3} k^{-5/3}}$$
여기서 $C_K \approx 1.6$ (Kolmogorov constant). 이것이 유체 난류의 상징적 결과.

Assuming constant energy flux $\varepsilon$ and $t_\ell \sim \ell/\Delta u_\ell$ gives $\Delta u_\ell \sim \varepsilon^{1/3}\ell^{1/3}$ and $E(k) = C_K \varepsilon^{2/3} k^{-5/3}$, $C_K \approx 1.6$.

### 4.2 Iroshnikov-Kraichnan MHD phenomenology / IK MHD 현상학

강한 평균장 $\mathbf{B}_0$ 하에서 Alfvén 파 쓸기(sweeping)의 감속 효과:
$$T_\ell^{\pm} \sim \frac{(t_\ell^{\pm})^2}{t_A}, \quad t_A = \ell/c_A, \quad t_\ell^{\pm} = \ell/\Delta z_\ell^{\pm}$$
$\varepsilon^{\pm} \sim (\Delta z_\ell^{\pm})^2 / T_\ell^{\pm}$에서:
$$\Delta z_\ell^{\pm} \sim (\varepsilon c_A)^{1/4} \ell^{1/4}$$
스펙트럼:
$$\boxed{E^{\pm}(k) \sim (\varepsilon c_A)^{1/2} k^{-3/2}}$$

Under strong $\mathbf{B}_0$, Alfvén sweeping lengthens the transfer time by $t_\ell/t_A$, giving $\Delta z_\ell^{\pm}\sim\ell^{1/4}$ and $E(k)\sim k^{-3/2}$.

### 4.3 Elsässer variables / Elsässer 변수

Elsässer (1950) 변수 변환:
$$\boxed{\mathbf{z}^{\pm} = \mathbf{v} \pm \frac{\mathbf{b}}{\sqrt{4\pi\rho}}}$$
비압축 MHD 방정식의 대칭 형태 (Eq. 15):
$$\frac{\partial \mathbf{z}^{\pm}}{\partial t} \mp (\mathbf{c}_A \cdot \nabla)\mathbf{z}^{\pm} + (\mathbf{z}^{\mp} \cdot \nabla)\mathbf{z}^{\pm} = -\nabla P_{\rm tot} + \nu^+ \nabla^2 \mathbf{z}^{\pm} + \nu^- \nabla^2 \mathbf{z}^{\mp}$$
**핵심 용어 해석 / Term-by-term**:
- $\mp (\mathbf{c}_A\cdot\nabla)\mathbf{z}^{\pm}$: Alfvén파 전파 ($\mathbf{z}^{\pm}$는 $\mp \mathbf{B}_0$ 방향으로 전파).
- $(\mathbf{z}^{\mp}\cdot\nabla)\mathbf{z}^{\pm}$: 비선형 항 — **반대 방향 펄스끼리만** 상호작용 (MHD 난류의 핵심!).
- $\nu^{\pm} = (\nu \pm \eta)/2$: 합/차 확산 계수.

### 4.4 Cross-helicity and residual energy / 교차 나선도와 잔차 에너지

Ideal MHD에서 보존되는 두 이차 불변량:
$$H_c(t) = \int_V \mathbf{v}\cdot\mathbf{b}\, d^3\mathbf{r}$$
정규화:
$$\boxed{\sigma_c = \frac{\langle |\mathbf{z}^+|^2 \rangle - \langle |\mathbf{z}^-|^2 \rangle}{\langle |\mathbf{z}^+|^2 \rangle + \langle |\mathbf{z}^-|^2 \rangle}}, \qquad \boxed{\sigma_r = \frac{E_v - E_b}{E_v + E_b}}$$
여기서 $E_v = \langle |\mathbf{v}|^2 \rangle /2$, $E_b = \langle |\mathbf{b}|^2/(4\pi\rho) \rangle/2$. 한계 조건:
- $\sigma_c = +1, \sigma_r = 0$: 순수 외향 Alfvén파 (빠른 풍 내층 근접).
- $\sigma_c = -1, \sigma_r = 0$: 순수 내향 Alfvén파.
- $\sigma_c = 0$: 등방 혼합 (느린 풍).
- $\sigma_r < 0$: 자기 에너지 초과 (실제 태양풍).

Normalized cross-helicity $\sigma_c$ and residual energy $\sigma_r$ characterize the Alfvén-balance state; pure outward wave has $(\sigma_c,\sigma_r)=(+1,0)$, observed slow wind has $\sigma_c\approx0$, and $\sigma_r\approx-0.4$ indicates persistent magnetic-energy excess.

### 4.5 Structure functions and Kolmogorov 4/5 law / 구조 함수와 4/5 법칙

$p$차 구조 함수:
$$\boxed{S_p(r) = \langle |\delta v(r)|^p \rangle \sim r^{\zeta_p}}$$
K41 예측: $\zeta_p = p/3$ (선형). **엄밀한 Kolmogorov 4/5 법칙** (Eq. 30):
$$\boxed{\langle (\Delta v_\parallel(\ell))^3 \rangle = -\frac{4}{5}\, \varepsilon\, \ell}$$
유도: NS 방정식에서 직접, 비압축성, 국소 등방성, 무한 Re 극한 가정. 항별 해석:
- $\langle(\cdot)^3\rangle$: 3차 적률로 방향성(cascade direction)을 담는다.
- $-4/5$: 유체의 정확한 수치 계수 (단위 소산률 당).
- 음부호: **직접 캐스케이드 (에너지가 작은 스케일로)**.
- 좌변이 0이 아니라는 사실 ⇒ 비 Gaussian 위상 상관이 필수.

The 4/5 law is the *only* exact non-trivial result for NS turbulence, with the minus sign encoding direct energy cascade toward small scales.

### 4.6 MHD Yaglom law / MHD Yaglom 법칙

Politano-Pouquet (1998):
$$\boxed{Y_\ell^{\pm} \equiv \langle \Delta z_\ell^{\mp}\, |\Delta \mathbf{z}^{\pm}|^2 \rangle = -\frac{4}{3}\, \varepsilon^{\pm}\, \ell}$$
여기서 $\Delta z_\ell^{\mp}$는 $\mathbf{z}^{\mp}$의 증분의 종방향 성분. 항별:
- $\varepsilon^{\pm}$: $\mathbf{z}^{\pm}$ 의사 에너지 (pseudo-energy) 소산률.
- $-4/3$ (vs. 유체의 $-4/5$): MHD 기하학 반영.
- **엄밀한 관성영역 정의**: $Y_\ell^{\pm} \propto \ell$가 성립하는 스케일 구간.

### 4.7 Partial Variance of Increments (PVI) / 부분 분산 증분

Greco et al. (2008)이 도입한 국소 진단:
$$\boxed{\text{PVI}(t, \tau) = \frac{|\Delta \mathbf{B}(t, \tau)|}{\sqrt{\langle |\Delta \mathbf{B}|^2 \rangle}}}$$
여기서 $\Delta \mathbf{B}(t,\tau) = \mathbf{B}(t+\tau) - \mathbf{B}(t)$. PVI > 3-4 문턱 이상의 사건은 강한 국소 불연속성 (current sheets, discontinuities)과 1:1 대응 — 간헐성의 물리적 해석을 제공.

Partial Variance of Increments identifies coherent intermittent structures (current sheets, discontinuities) via threshold-crossings of the normalized increment magnitude.

### 4.8 Spectral break & Hall-MHD / 스펙트럼 분절과 Hall-MHD

Hall 효과 포함 유도 방정식 (Eq. 76):
$$\frac{\partial \mathbf{B}}{\partial t} = \nabla \times \left[\mathbf{V}\times\mathbf{B} - \frac{m_i}{\rho e}(\nabla\times\mathbf{B})\times\mathbf{B} + \eta\nabla\times\mathbf{B}\right]$$
세 특성 시간: eddy-turnover $T_{NL}\sim \ell/u_\ell$, Hall $T_H \sim \rho_\ell \ell^2/B_\ell$, 확산 $T_D \sim \ell^2/\eta$. 압축 가중 $\rho_\ell \sim \ell^{-3r}$과 일정 에너지 플럭스 조건:
$$\boxed{E(k) \sim k^{-7/3 + r}}$$
$r \in [-5/6, 1/6]$ 범위는 관측되는 기울기 $\alpha \in [2, 4]$에 해당.

Hall-MHD with compressibility exponent $r \in [-5/6, 1/6]$ reproduces observed high-frequency slopes in $[-4, -2]$ around $k^{-7/3+r}$.

### 4.9 Heating rate estimation / 가열률 평가

Kolmogorov 스펙트럼의 파워에서 역산 (Eq. 72):
$$\varepsilon_P = \left[\frac{5}{3} P(k) C_K^{-1}\right]^{3/2} k^{5/2}$$
또는 정상 상태 온도 방정식(Eq. 75):
$$\varepsilon(r) = \frac{3}{2}\left(\frac{4}{3} - \xi\right)\frac{V_{\rm sw}(r) k_B T(r)}{r\, m_p}$$
1 AU 빠른 바람에서 $\varepsilon \sim 10^4$ J/(kg·s), 느린 바람 $\sim 10^2$ J/(kg·s).

Heating rate from Kolmogorov spectrum (Eq. 72) or steady-state temperature equation (Eq. 75) yields $\varepsilon \sim 10^2$-$10^4$ J/(kg·s) at 1 AU.

### 4.10 Worked numerical example / 수치 예제

1 AU 태양풍 전형 파라미터 (Table 6-8 in Appendix A):
- $n_p \simeq 4$ cm$^{-3}$, $|B| \simeq 6$ nT, $V_{\rm sw} \simeq 600$ km/s (fast wind).
- Alfvén speed $c_A = B/\sqrt{4\pi n_p m_p} \simeq 60$ km/s.
- Proton cyclotron frequency $f_{ci} = eB/(2\pi m_p) \simeq 0.09$ Hz.
- Ion inertial length $\lambda_i = c/\omega_{pi} \simeq 130$ km.

Kolmogorov 상수 $C_K \simeq 1.6$, 관측 magnetic spectrum 파워 at $f = 10^{-3}$ Hz: $P(f) \simeq 10^3$ nT$^2$/Hz. Taylor 변환 $k = 2\pi f/V_{\rm sw} \simeq 10^{-8}$ m$^{-1}$. Eq. (72)로 $\varepsilon$ 산정:
$$\varepsilon \simeq [5/3 \cdot 10^3 \cdot (10^{-9})^2 / 1.6]^{3/2} \cdot (10^{-8})^{5/2}$$
이 수치 조합은 $\varepsilon \sim 10^3-10^4$ J/(kg·s)의 order를 되살리며 MacBride et al. (2010)의 값과 일치.

Worked example: with $|B|=6$ nT, $n_p=4$ cm$^{-3}$, $V_{\rm sw}=600$ km/s at 1 AU, we obtain $c_A\simeq60$ km/s, $f_{ci}\simeq0.09$ Hz, $\lambda_i\simeq130$ km. Kolmogorov relation yields $\varepsilon\sim10^3$-$10^4$ J/(kg·s), matching MacBride et al. (2010).

**Spectral break numerical scale / 분절 스케일 추정**: Cluster measured $f_{\rm br} \simeq 0.3$ Hz $\implies$ scale $\ell = V_{\rm sw}/(2\pi f_{\rm br}) \simeq 600 / 2 \simeq 300$ km $\sim 2-3 \lambda_i$ (Alexandrova et al. 2008 reported $\sim 15\lambda_i$ depending on $V_{\rm sw}$).

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1883  Reynolds: laminar-turbulent transition,
        Re = UL/ν ~ 4000 for pipe flow
          │
          ▼
1941  Kolmogorov K41: E(k) = C_K ε^{2/3} k^{-5/3}
        4/5 law: <(Δv_∥)^3> = -(4/5) ε ℓ
          │
1950  Elsässer: z± = v ± b/√(4πρ),
        symmetric MHD equations
          │
1958  Parker: solar wind theory (supersonic expansion)
          │
1963-65 Iroshnikov/Kraichnan: MHD E(k) ~ k^{-3/2}
        (Alfvén effect slows cascade)
          │
1963  Lorenz: deterministic chaos → butterfly
          │
1968  Coleman (Mariner 2): first solar wind
        spectra, f^{-1.2} slope
          │
1971  Belcher & Davis: Alfvénic fluctuations
        in fast wind (δv = ∓ δb/√(4πρ))
          │
1974-81 Helios 1/2: inner heliosphere (0.3-1 AU)
        radial evolution of σ_c, σ_r
          │
1977- Voyager 1/2: outer heliosphere turbulence
          │
1978  Frisch-Sulem-Nelkin β-model (intermittency)
          │
1990- Ulysses: polar heliosphere survey,
        sustained Alfvénicity at high latitude
          │
1990  Matthaeus et al.: "Maltese cross"
        ~80% 2D + 20% slab
          │
1993  Benzi et al.: Extended Self-Similarity (ESS)
          │
1994  She & Leveque: log-Poisson ζ_p model
          │
1995  Goldreich-Sridhar: critical balance,
        k_⊥^{-5/3}, k_∥^{-2} anisotropy
          │
1998  Politano & Pouquet: MHD Yaglom law
        Y_ℓ± = -(4/3) ε± ℓ
          │
1998  Leamon et al. (Wind): spectral break
        at f_br ≃ 0.44 Hz
          │
2000- Cluster: 4-spacecraft, small-scale physics
          │
2007  Podesta et al.: v ~ -3/2, B ~ -5/3
        (dual spectra puzzle)
          │
2007  Sorriso-Valvo et al.: Yaglom law observed
        in Ulysses polar wind
          │
2008- MacBride, Alexandrova, Sahraoui, Kiyani:
        heating rates, small-scale intermittency,
        KAW/whistler scenarios
          │
2013  ★ Bruno & Carbone LRSP update (this paper)
          │
          ▼
2018-  Parker Solar Probe (< 0.3 AU)
2020-  Solar Orbiter: multi-instrument inner
        heliosphere; future Yaglom/σ_c mapping
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Kolmogorov (1941), "Local structure of turbulence" | K41 phenomenology와 4/5 법칙. Bruno-Carbone의 Ch. 2.4-2.10의 이론적 출발점. / K41 phenomenology and 4/5 law; starting point for all Bruno-Carbone theory chapters. | **Foundational** — 모든 난류 이론의 기반 / Foundation for all turbulence theory |
| Kraichnan (1965), "Inertial-range spectrum of hydromagnetic turbulence" | IK MHD 현상학으로 $k^{-3/2}$ 스펙트럼 도출. 태양풍 속도 스펙트럼과 일치. / IK MHD phenomenology yielding $k^{-3/2}$; matches solar-wind velocity. | **Critical** — MHD 난류의 한 기둥 / One pillar of MHD turbulence |
| Belcher & Davis (1971), "Large-amplitude Alfvén waves in the interplanetary medium" | 태양풍 Alfvén파 상관의 최초 관측. Ch. 3.1.8-3.2에서 핵심 관측. / First observation of Alfvénic correlation in solar wind. | **Historical** — 태양풍 난류 관측의 시작 / Origin of solar-wind turbulence observations |
| Goldreich & Sridhar (1995), "Toward a theory of interstellar turbulence" | Critical balance와 MHD 이방성. Bruno-Carbone이 spectral anisotropy 논의에서 참조. / Critical balance and MHD anisotropy; referenced in anisotropy discussions. | **Modern theory** — 이방성 MHD 난류의 표준 / Standard anisotropic MHD theory |
| Politano & Pouquet (1998), "Dynamical length scales for turbulent magnetized flows" | MHD Yaglom 법칙의 수학적 유도. Ch. 2.11-8의 핵심. / Mathematical derivation of MHD Yaglom law; central to Chs 2.11, 8. | **Critical derivation** — 관성영역의 엄밀한 정의 / Exact inertial-range law |
| Sorriso-Valvo et al. (2007), "Observation of inertial energy cascade in interplanetary space plasma" | Ulysses polar wind에서 Yaglom 법칙 최초 관측. Ch. 8의 주된 근거. / First Yaglom-law observation (Ulysses polar wind); main evidence for Ch. 8. | **Experimental verification** — Yaglom 법칙의 실증 / Verifies Yaglom's law |
| Leamon et al. (1998), "Observational constraints on the dynamics of the interplanetary magnetic field dissipation range" | 스펙트럼 분절 최초 탐색. Ch. 11.1의 주요 근거. / First identification of spectral break; basis of Ch. 11.1. | **Key finding** — 스펙트럼 분절의 발견 / Discovery of spectral break |
| She & Leveque (1994), "Universal scaling laws in fully developed turbulence" | 간헐성 모델. Bruno-Carbone Ch. 7에서 다중분산 모델 중 하나로 논의. / Log-Poisson intermittency model; discussed in Ch. 7. | **Strong connection** — 간헐성 모델의 한 예 / Sample intermittency model |

---

## 7. References / 참고문헌

- Bruno, R. and Carbone, V., "The Solar Wind as a Turbulence Laboratory", *Living Rev. Solar Phys.*, **10**, (2013), 2. doi:[10.12942/lrsp-2013-2](https://doi.org/10.12942/lrsp-2013-2)
- Kolmogorov, A. N., "The local structure of turbulence in incompressible viscous fluid for very large Reynolds numbers", *Dokl. Akad. Nauk SSSR*, **30**, 301 (1941).
- Iroshnikov, P. S., "Turbulence of a conducting fluid in a strong magnetic field", *Sov. Astron.*, **7**, 566 (1963).
- Kraichnan, R. H., "Inertial-range spectrum of hydromagnetic turbulence", *Phys. Fluids*, **8**, 1385 (1965).
- Elsässer, W. M., "The hydromagnetic equations", *Phys. Rev.*, **79**, 183 (1950).
- Belcher, J. W., and Davis Jr, L., "Large-amplitude Alfvén waves in the interplanetary medium, 2", *J. Geophys. Res.*, **76**, 3534 (1971).
- Coleman Jr, P. J., "Turbulence, viscosity, and dissipation in the solar-wind plasma", *Astrophys. J.*, **153**, 371 (1968).
- Goldreich, P. and Sridhar, S., "Toward a theory of interstellar turbulence. II. Strong Alfvénic turbulence", *Astrophys. J.*, **438**, 763 (1995).
- Politano, H. and Pouquet, A., "Dynamical length scales for turbulent magnetized flows", *Geophys. Res. Lett.*, **25**, 273 (1998).
- Sorriso-Valvo, L., Marino, R., Carbone, V., et al., "Observation of inertial energy cascade in interplanetary space plasma", *Phys. Rev. Lett.*, **99**, 115001 (2007).
- Leamon, R. J., Smith, C. W., Ness, N. F., Matthaeus, W. H., and Wong, H. K., "Observational constraints on the dynamics of the interplanetary magnetic field dissipation range", *J. Geophys. Res.*, **103**, 4775 (1998).
- Alexandrova, O., Lacombe, C., and Mangeney, A., "Spectra and anisotropy of magnetic fluctuations in the Earth's magnetosheath: Cluster observations", *Ann. Geophys.*, **26**, 3585 (2008).
- Podesta, J. J., Roberts, D. A., and Goldstein, M. L., "Spectral exponents of kinetic and magnetic energy spectra in solar wind turbulence", *Astrophys. J.*, **664**, 543 (2007).
- MacBride, B. T., Smith, C. W., and Forman, M. A., "The turbulent cascade at 1 AU: Energy transfer and the third-order scaling for MHD", *Astrophys. J.*, **679**, 1644 (2008).
- She, Z.-S. and Leveque, E., "Universal scaling laws in fully developed turbulence", *Phys. Rev. Lett.*, **72**, 336 (1994).
- Benzi, R., Ciliberto, S., Tripiccione, R., Baudet, C., Massaioli, F., and Succi, S., "Extended self-similarity in turbulent flows", *Phys. Rev. E*, **48**, R29 (1993).
- Matthaeus, W. H., Goldstein, M. L., and Roberts, D. A., "Evidence for the presence of quasi-two-dimensional nearly incompressible fluctuations in the solar wind", *J. Geophys. Res.*, **95**, 20673 (1990).
- Parker, E. N., "Dynamics of the interplanetary gas and magnetic fields", *Astrophys. J.*, **128**, 664 (1958).
- Tu, C.-Y. and Marsch, E., "MHD structures, waves and turbulence in the solar wind: Observations and theories", *Space Sci. Rev.*, **73**, 1 (1995).
- Frisch, U., *Turbulence: The Legacy of A. N. Kolmogorov*, Cambridge Univ. Press (1995).
