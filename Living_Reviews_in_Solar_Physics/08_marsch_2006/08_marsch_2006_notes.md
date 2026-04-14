---
title: "Kinetic Physics of the Solar Corona and Solar Wind"
authors: Eckart Marsch
year: 2006
journal: "Living Reviews in Solar Physics, 3, 1"
topic: "Living Reviews in Solar Physics / Kinetic Plasma Physics"
tags: [kinetic physics, solar wind, solar corona, VDF, velocity distribution function, Vlasov-Boltzmann, kappa distribution, wave-particle interaction, quasilinear theory, cyclotron resonance, Landau damping, exospheric model, coronal heating, proton beam, strahl, temperature anisotropy, KAW, multi-moment fluid, transport theory]
status: completed
date_started: 2026-04-09
date_completed: 2026-04-09
---

# Kinetic Physics of the Solar Corona and Solar Wind — Reading Notes
# 태양 코로나와 태양풍의 운동론적 물리학 — 읽기 노트

---

## 핵심 기여 / Core Contribution

이 리뷰는 태양 코로나와 태양풍의 **운동론적(kinetic) 물리학**을 포괄적으로 종합한 핵심 참고문헌이다. MHD(자기유체역학) 수준의 유체 기술만으로는 포착할 수 없는 입자 수준의 물리 현상 -- 속도 분포 함수(VDF)의 비Maxwell 특성, 온도 비등방성, 양성자 beam, 전자 strahl, 파동-입자 상호작용 -- 을 Helios, SOHO, Ulysses의 *in situ* 관측 데이터를 바탕으로 체계적으로 정리한다. Vlasov-Boltzmann 방정식에서 출발하여 고전적 수송 이론의 한계를 규명하고, 준선형 이론(QLT)에 기반한 파동-입자 에너지 교환, cyclotron/Landau 공명, kinetic Alfven wave(KAW)의 역할을 상세히 논의하며, Vlasov 방정식의 수치 풀이를 통한 코로나 이온/전자 모델링 결과를 제시한다. 코로나 가열과 태양풍 가속이라는 태양물리학의 핵심 미해결 문제에 대해 유체 관점을 넘어선 운동론적 해답의 필요성을 설득력 있게 논증한다.

This review is a key reference that comprehensively synthesizes the **kinetic (particle-level) physics** of the solar corona and solar wind. It systematically organizes phenomena that cannot be captured by MHD-level fluid descriptions alone -- non-Maxwellian velocity distribution functions (VDFs), temperature anisotropies, proton beams, electron strahl, and wave-particle interactions -- grounded in *in situ* observational data from Helios, SOHO, and Ulysses. Starting from the Vlasov-Boltzmann equation, it identifies the breakdown of classical transport theory, discusses in detail the wave-particle energy exchange based on quasilinear theory (QLT), the roles of cyclotron/Landau resonance, and kinetic Alfven waves (KAWs), and presents results from numerical solutions of the Vlasov equation for coronal ion/electron modelling. The paper convincingly argues for the necessity of kinetic-level answers -- beyond fluid perspectives -- to the central unsolved problems of solar physics: coronal heating and solar wind acceleration.

---

## 목차 / Table of Contents

1. [서론 / Introduction](#1-서론--introduction)
2. [입자 속도 분포 / Particle Velocity Distributions](#2-입자-속도-분포--particle-velocity-distributions)
3. [운동론적 기술 / Kinetic Description](#3-운동론적-기술--kinetic-description)
4. [수송 / Transport](#4-수송--transport)
5. [플라즈마 파동과 미시불안정성 / Plasma Waves and Microinstabilities](#5-플라즈마-파동과-미시불안정성--plasma-waves-and-microinstabilities)
6. [파동-입자 상호작용 / Wave-Particle Interactions](#6-파동-입자-상호작용--wave-particle-interactions)
7. [운동론적 모델링 / Kinetic Modelling](#7-운동론적-모델링--kinetic-modelling)
8. [요약 및 결론 / Summary and Conclusions](#8-요약-및-결론--summary-and-conclusions)

---

## 1. 서론 / Introduction

### 1.1 논문 범위와 운동론의 중요성 / Scope and Importance of Kinetic Physics

이 리뷰는 태양풍의 열적(thermal) 입자와 초열적(suprathermal) 입자의 운동론적 성질에 초점을 맞춘다. MHD 수준의 파동과 난류는 이미 다른 리뷰(예: Bruno & Carbone 2005, Nakariakov & Verwichte 2005 = LRSP #3)에서 다루어졌으므로, 여기서는 MHD를 **넘어서는** 운동론적 영역 -- 입자의 자이로주파수 근처에서 작동하는 kinetic plasma wave, 입자 VDF의 비열적 특성, 파동-입자 상호작용 -- 을 집중적으로 다룬다.

This review focuses on the kinetic properties of thermal and suprathermal particles in the solar wind. Since MHD-level waves and turbulence have already been covered in other reviews (e.g., Bruno & Carbone 2005, Nakariakov & Verwichte 2005 = LRSP #3), it concentrates on the kinetic domain **beyond MHD** -- kinetic plasma waves operating near particle gyrofrequencies, non-thermal features of particle VDFs, and wave-particle interactions.

태양 코로나 플라즈마는 희박하고(tenuous), 다성분(multi-component)이며, 비균일(non-uniform)하고, 대부분 LTE(국소 열역학적 평형) 또는 충돌 평형 상태에 있지 않다. 따라서 다중 유체(multi-fluid) 이론이나 완전한 운동론적 물리가 필수적이다. VDF를 측정할 수 있는 우주 탑재 입자 분광계가 이론적 기술의 적절한 수단인 Vlasov-Boltzmann 운동 플라즈마 이론과 직접 연결된다.

The solar coronal plasma is tenuous, multi-component, non-uniform, and mostly not at LTE (Local Thermodynamic Equilibrium) or collisional equilibrium conditions. Therefore multi-fluid theories or full kinetic physics are required. Space-borne particle spectrometers that can measure VDFs connect directly to Vlasov-Boltzmann kinetic plasma theory, the appropriate theoretical framework.

### 1.2 태양풍의 주요 유형 / Main Types of Solar Wind

Marsch는 태양풍을 세 가지 기본 유형으로 분류한다:

Marsch classifies the solar wind into three basic types:

| 유형 / Type | 기원 / Origin | 특성 / Characteristics |
|---|---|---|
| **고속풍 (Fast wind)** | 코로나 홀의 열린 자기장선 / Open field lines in coronal holes | 정상적, 균일, $V \sim 700$-800 km/s, 강한 비Maxwell VDF / Steady, uniform, strong non-Maxwellian VDFs |
| **저속풍 (Slow wind)** | Streamer 경계, 일시적 열림 / Streamer boundaries, transiently open fields | 비정상적, 가변적, $V \sim 300$-400 km/s, 보다 Maxwell에 가까움 / Unsteady, variable, more nearly Maxwellian |
| **과도풍 (Transient wind)** | CME (코로나 질량 방출) / Coronal mass ejections | 태양 극대기에 우세, 매우 가변적 / Prevalent at solar maximum, highly variable |

코로나 깔때기(coronal funnel) 구조가 빠른 태양풍의 기원과 직접 연결된다. Tu et al. (2005)은 Doppler shift와 코로나 자기장의 상관관계를 통해 태양풍이 이 깔때기 구조에서 기원함을 확인했다. 빠른 태양풍은 chromospheric network boundary에서 직접 발원하며, 초기 유출 속도는 최대 10 km/s에 달한다. 열린 코로나 자기장(~10 G)은 supergranular network에 고정되어 있으며, 이는 코로나 밑면 면적의 약 10%만 차지한다.

The coronal funnel structure is directly connected to the origin of the fast solar wind. Tu et al. (2005) confirmed that the solar wind originates from these funnel structures through correlations between Doppler shifts and coronal magnetic fields. The fast wind emanates directly from chromospheric network boundaries with initial outflow speeds up to 10 km/s. The open coronal magnetic field (~10 G) is anchored in the supergranular network, occupying only about 10% of the coronal base area.

---

## 2. 입자 속도 분포 / Particle Velocity Distributions

이 장은 논문의 관측적 기반을 제공하며, 이후 모든 운동론적 이론의 동기를 부여한다.

This section provides the observational foundation and motivates all subsequent kinetic theory.

### 2.1 일반적 고려사항 / General Considerations

태양풍은 **비충돌 플라즈마(collisionless plasma)**의 고전적 패러다임이다. ~40년에 걸친 *in situ* 측정이 수행되었으며, 태양풍의 상태는 코로나에서 heliosphere 경계까지 잘 알려져 있다. 세 가지 유형의 태양풍은 운동론적 성질에서 근본적으로 다르며, 이는 서로 다른 코로나 경계 조건과 행성간 플라즈마 역학에 기인한다.

The solar wind is the classical paradigm of a **collisionless plasma**. ~40 years of *in situ* measurements have been carried out, and the solar wind state is well known from the corona to the heliopause. The three types of solar wind differ fundamentally in their kinetic properties, due to different coronal boundary conditions and interplanetary plasma dynamics.

핵심 통찰: 태양풍의 반경 방향 진화는 복잡한 이완(relaxation) 과정을 닮으며, 입자의 자유 에너지(운동론적/MHD 평형에 비해)가 열에너지와 파동 에너지로 전환된다.

Key insight: The radial evolution of the solar wind's internal state resembles a complicated relaxation process, in which the particles' free energy is converted to thermal and wave energy distributed over a range of scales.

### 2.2 태양풍 전자 / Solar Wind Electrons

전자 VDF는 **세 가지 성분**으로 구성된다:

The electron VDF consists of **three components**:

1. **Core (핵심부)**: 차가운 열적 벌크 집단. 거의 등방적이며 충돌에 의해 형성. 총 전자 밀도의 ~96%. / Cold thermal bulk population. Nearly isotropic, shaped by collisions. ~96% of total electron density.

2. **Halo (후광)**: 뜨거운 초열적 집단. Core보다 ~7배 높은 온도. 거의 등방적. 총 밀도의 ~4%. / Hot suprathermal population. ~7 times core temperature. Nearly isotropic. ~4% of total density.

3. **Strahl (줄기)**: 자기장 방향을 따라 좁은 pitch-angle 분포를 가진 초열적 전자 빔. 주로 빠른 태양풍에서 뚜렷. **열 흐름(heat flux)의 주요 운반체**. / Suprathermal electron beam with narrow pitch-angle distribution along the magnetic field. Most prominent in fast solar wind. **Primary carrier of heat flux**.

Figure 1은 Helios가 측정한 전형적 전자 VDF를 보여준다. 자기장 방향을 따른 뚜렷한 bulge(strahl)가 관찰되며, core보다 변위된 위치에 있는 halo와 함께 열 흐름을 운반한다. Maksimovic et al. (2005)에 따르면 (Figure 1 하단), strahl의 상대적 밀도는 태양에서 멀어질수록 감소하고 halo는 증가하는데, 이는 strahl 전자가 산란되어 halo로 변환됨을 시사한다.

Figure 1 shows a typical electron VDF measured by Helios. A distinct bulge (strahl) is observed along the magnetic field direction, carrying heat flux together with the halo displaced relative to the core. According to Maksimovic et al. (2005) (lower panel of Figure 1), the relative density of the strahl decreases with radial distance while the halo increases, suggesting that strahl electrons are scattered into the halo.

Figure 2는 빠른/중간/느린 태양풍에서의 전자 VDF를 에너지 스펙트럼과 속도 공간 등고선으로 보여준다. Core-halo 구조는 종종 두 개의 convecting bi-Maxwellian으로 모델링된다.

Figure 2 shows electron VDFs in fast/intermediate/slow solar wind as energy spectra and velocity space contours. The core-halo structure is often modeled as two convecting bi-Maxwellians.

전자 열 흐름은 멱법칙 스케일링 $q_e \sim R^{-2.9}$을 따르며, 태양 주기나 태양풍 속도에 대한 유의미한 의존성은 없다. 전자-양성자 온도비 $T_e/T_p$는 풍속에 따라 체계적으로 변하며, 300 km/s에서 ~4, 700 km/s에서 ~0.5이다.

The electron heat flux follows a power-law scaling $q_e \sim R^{-2.9}$, with no significant dependence on the solar cycle or wind speed. The electron-to-proton temperature ratio $T_e/T_p$ varies systematically with wind speed: ~4 at 300 km/s and ~0.5 at 700 km/s.

Break-point 에너지(core와 halo 사이)는 평균적으로 core 온도의 약 7배에 해당하며, 이는 Coulomb 충돌만으로 전자가 매개될 때의 운동론적 예측(Scudder & Olbert 1979)과 일치한다. 이 에너지는 또한 행성간 정전 포텐셜 $\Phi_e \sim 50$-100 eV (1 AU)와 일치한다.

The break-point energy (between core and halo) scales on average as about 7 times the core temperature, consistent with kinetic predictions for electrons mediated by Coulomb collisions alone (Scudder & Olbert 1979). This energy is also consistent with the interplanetary electrostatic potential $\Phi_e \sim 50$-100 eV at 1 AU.

### 2.3 태양풍 양성자와 알파 입자 / Solar Wind Protons and Alpha Particles

양성자 VDF는 전자보다 **훨씬 비열적(non-thermal)**이다. 이는 약한 충돌성(weak collisionality) 때문에 위상 공간에서 큰 왜곡이 생기고, 난류 파동과의 상호작용에 의해 강하게 형성되기 때문이다.

Proton VDFs are **far more non-thermal** than electrons. This is because weak collisionality allows large distortions in phase space, and they are strongly shaped by interactions with turbulent waves.

Figure 3은 Helios가 측정한 빠른 태양풍의 양성자 VDF 네 가지 예를 보여준다. 핵심 특징:

Figure 3 shows four examples of proton VDFs in fast solar wind measured by Helios. Key features:

- **Core 온도 비등방성**: $T_{p\perp} > T_{p\parallel}$ (빠른 풍), $T_{p\perp} < T_{p\parallel}$ (느린 풍). 비등방성 비율 $T_{\perp c}/T_{\parallel c} \sim 2$-3 (0.3 AU 근처). / Core temperature anisotropy: $T_{p\perp} > T_{p\parallel}$ (fast wind), $T_{p\perp} < T_{p\parallel}$ (slow wind). Anisotropy ratio $T_{\perp c}/T_{\parallel c} \sim 2$-3 near 0.3 AU.

- **양성자 beam**: 자기장 방향으로 core에서 분리된 2차 성분. 드리프트 속도 $v_d \sim 1.5 V_A$ (국소 Alfven 속도). / Proton beam: secondary component separated from core along the magnetic field. Drift speed $v_d \sim 1.5 V_A$ (local Alfven speed).

- **자기 모멘트 비보존**: Figure 4 상단은 양성자 자기 모멘트 $\mu_p = T_{p\perp}/B$가 태양중심 거리에 따라 증가함을 보여준다. 이는 자기장에 수직한 방향으로의 **지속적 가열(continuous perpendicular heating)**이 발생함을 의미한다. 단열 팽창만으로는 $\mu_p$가 보존되어야 하므로, 파동에 의한 가열이 필수적이다. / Non-conservation of magnetic moment: The upper panel of Figure 4 shows that proton magnetic moment $\mu_p = T_{p\perp}/B$ increases with heliocentric distance. This implies **continuous perpendicular heating** of ions. Under adiabatic expansion alone, $\mu_p$ should be conserved, so wave heating is essential.

빠른 태양풍에서 전자는 양성자보다 차갑다($T_e = 0.1$-0.2 MK, $T_p = 0.5$-0.8 MK at 0.3 AU). Alfven 파동이 광대역 주파수로 빠른 풍에 침투하며, 이온 온도를 단열 냉각 수준 이상으로 유지하는 핵심 역할을 한다.

In fast solar wind, electrons are cooler than protons ($T_e = 0.1$-0.2 MK, $T_p = 0.5$-0.8 MK at 0.3 AU). Alfven waves permeate the fast wind at broad-band frequencies and play a key role in maintaining ion temperatures above the level expected from adiabatic cooling.

### 2.4 중이온 / Heavy Ions

중이온의 3차원 VDF는 양성자/알파와 달리 직접 측정되지 않았지만, 에너지 스펙트럼은 측정되었다.

Unlike protons and alphas, 3D VDFs of heavy ions have not been directly measured, but energy spectra have been obtained.

중이온 가열의 에너지 요구조건은 까다롭다. Coulomb 마찰이 희박한 코로나에서 중이온을 태양 중력에 대항하여 끌어내기 어렵다. 양성자와 중이온의 코로나 속도 분포가 겹치려면(즉, 같은 벌크 속도를 달성하려면), 온도 관계식 (Eq 1)이 성립해야 한다:

The energy requirements for heavy ion heating are demanding. Coulomb friction in the dilute corona is too weak to drag heavy ions against solar gravity. For coronal velocity distributions of protons and heavy ions to overlap (i.e., achieve equal bulk speeds), the temperature relation (Eq 1) must hold:

$$T_i + Z_i T_e = A_i(T_p + T_e) \tag{1}$$

여기서 $Z_i$는 전하, $A_i$는 원자 질량수. 이는 $T_i > A_i T_p$를 의미한다. 파동 가열이 중이온을 코로나 밖으로 끌어내는 결정적 역할을 하며, 미지의 소수 이온 가열률 $Q_i > A_i Q_p$가 필요하다.

Here $Z_i$ is the charge and $A_i$ the atomic mass number. This implies $T_i > A_i T_p$. Wave heating plays a decisive role in dragging heavy species out of the corona, requiring an unknown minor ion heating rate $Q_i > A_i Q_p$.

Figure 5는 WIND의 이온 질량 분광계가 측정한 He, O, Ne의 VDF를 보여준다. 확장된 멱법칙 꼬리(power-law tail)가 뚜렷하며, kappa 함수($\kappa \sim 2.5$-4)로 잘 피팅된다. 이 초열적 꼬리는 열적 keV 영역과 MeV 에너지 입자를 연결하는 다리 역할을 한다.

Figure 5 shows VDFs of He, O, and Ne measured by WIND's ion mass spectrometer. Extended power-law tails are pronounced and well-fitted by kappa functions ($\kappa \sim 2.5$-4). These suprathermal tails serve as a bridge linking the thermal keV range with MeV energetic particles.

---

## 3. 운동론적 기술 / Kinetic Description

### 3.1 코로나 팽창의 기본 에너지론 / Basic Energetics of Coronal Expansion

코로나 팽창의 polytropic 모델에서 Bernoulli 에너지 보존(Eq 2):

In the polytropic model of coronal expansion, Bernoulli energy conservation (Eq 2):

$$\frac{1}{2}V^2 = \frac{\gamma}{\gamma-1}\frac{2k_B T_C}{m_p} - \frac{GM_\odot}{R_\odot} \tag{2}$$

여기서 $V$는 종단 태양풍 속도, $T_C$는 코로나 온도, $\gamma$는 polytropic 지수. 태양 표면 탈출 속도 $V_\infty = (2GM_\odot/R_\odot)^{1/2} = 618$ km/s이다.

Here $V$ is the terminal wind speed, $T_C$ the coronal temperature, $\gamma$ the polytropic index. Solar surface escape speed $V_\infty = (2GM_\odot/R_\odot)^{1/2} = 618$ km/s.

물리적 해석: 이 관계는 코로나 양성자-전자 쌍당 약 5 keV의 에너지가 필요함을 보여준다. $\gamma = 5/3$이면 $T_C = 1$ MK일 때 임계점이 없어 코로나가 중력적으로 속박된다. $V = 700$ km/s의 빠른 흐름을 얻으려면 $T_C \sim 10$ MK이 필요하다. 그러나 등온 모델($\gamma \to 1$)은 무한한 내부 에너지를 요구하므로 비물리적이다. 핵심 문제는 코로나 가열인데, 이는 polytropic 모델에서 전혀 다루어지지 않는다.

Physical interpretation: This relation shows that about 5 keV per proton-electron pair of coronal energy is needed. If $\gamma = 5/3$ and $T_C = 1$ MK, there is no critical point and the corona is gravitationally bound. To obtain fast flow with $V = 700$ km/s requires $T_C \sim 10$ MK. However, isothermal models ($\gamma \to 1$) require infinite internal energy and are unphysical. The key issue is coronal heating, which is not addressed at all in the polytropic model.

### 3.2 코로나와 태양풍의 충돌 조건 / Collisional Conditions (Table 1)

Table 1은 태양 대기의 세 영역에서 충돌 파라미터가 극적으로 변화함을 보여준다:

Table 1 shows the dramatic variation of collision parameters across three regions of the solar atmosphere:

| 파라미터 / Parameter | 채층 / Chromosphere ($1.01 R_\odot$) | 코로나 / Corona ($1.3 R_\odot$) | 태양풍 / Solar wind (1 AU) |
|---|---|---|---|
| $n$ (cm$^{-3}$) | $10^{10}$ | $10^7$ | 10 |
| $T$ (K) | $10^3$ | $1$-$2 \times 10^6$ | $10^5$ |
| $\lambda_c$ (km) | 1 | $10^3$ | $10^7$ |

물리적 의미: 코로나에서 이미 충돌 자유 경로 $\lambda_c$가 온도 기울기 스케일 높이 $L$보다 훨씬 크며, 이는 고전적 수송 이론의 기본 가정($L \gg \lambda_c$)이 위반됨을 의미한다. 코로나 홀의 빠른 태양풍은 본질적으로 비충돌적(collisionless)이며, streamer/current sheet 근처의 느린 풍만이 약간의 충돌성을 갖는다.

Physical significance: In the corona, the collisional free path $\lambda_c$ already far exceeds the temperature gradient scale height $L$, meaning the basic assumption of classical transport theory ($L \gg \lambda_c$) is violated. Fast solar wind from coronal holes is essentially collisionless, and only slow wind near streamers/current sheets has marginal collisionality.

### 3.3 Exospheric 패러다임 / The Exospheric Paradigm

Exospheric 모델은 코로나를 비충돌적 외기권으로 취급하여 태양풍을 해석적으로 모델링하는 가장 단순한 접근이다. 핵심 개념:

Exospheric models treat the corona as a collisionless exosphere, the simplest approach to analytically model the solar wind. Key concepts:

- **Exobase**: 입자 평균 자유 경로가 기압 스케일 높이를 초과하는 고도. 최근 모델에서는 $r_0 \sim 1.1$-5 $R_\odot$로 설정. / Altitude where particle mean free path exceeds barometric scale height. Set at $r_0 \sim 1.1$-5 $R_\odot$ in recent models.

- **유효 포텐셜(Effective potential)** (Eq 5): 자기 거울력 + 중력 + 정전 포텐셜의 합:

- **Effective potential** (Eq 5): sum of magnetic mirror force + gravity + electrostatic potential:

$$\Psi(r) = \mu B(r) + m\Phi_g(r) + q\Phi_e(r) \tag{5}$$

- **핵심 메커니즘**: 초열적 전자 꼬리가 설정하는 강한 전기장이 양성자를 가속. Lamy et al. (2003), Zouganelis et al. (2003, 2004)의 현대 exospheric 모델은 exobase의 전자 VDF에 kappa 분포(Eq 6)를 가정하여 비단조적(non-monotonic) 포텐셜 에너지를 구현.

- **Key mechanism**: A strong electric field set up by suprathermal electron tails accelerates protons. Modern exospheric models by Lamy et al. (2003), Zouganelis et al. (2003, 2004) assume kappa distributions (Eq 6) for the electron VDF at the exobase, implementing non-monotonic potential energy.

Figure 6은 exospheric 모델의 정전 포텐셜(좌)과 전기력 대 중력 비율(우)을 보여준다. 전기력이 중력을 초과하는 영역에서 양성자가 가속된다.

Figure 6 shows the electrostatic potential (left) and ratio of electric force to gravity (right) in the exospheric model. Protons are accelerated where electric force exceeds gravity.

**한계**: Exospheric 모델은 이온과 전자의 온도 비등방성이 관측보다 너무 크게 나오며, 파동과 충돌에 의한 산란을 포함하지 않으면 *in situ* 관측과 불일치한다. 또한 초열적 전자의 코로나 기원이 불명확하다.

**Limitation**: Exospheric models produce temperature anisotropies of ions and electrons that are far too large compared to observations, and disagree with *in situ* measurements unless scattering by waves and collisions is included. Also, the coronal origin of suprathermal electrons is unclear.

### 3.4 충돌에 의한 코로나 가열의 실패 / Failure to Heat Corona by Collisions

고전적 소산률(점성, 열전도, Ohm 저항)은 코로나 조건에서 복사 냉각보다 **6자릿수(orders of magnitude)**나 작다. 예를 들어:

Classical dissipation rates (viscosity, thermal conduction, Ohmic resistance) are **6 orders of magnitude** smaller than radiative cooling under coronal conditions. For example:

- 복사 손실: $Q_R = n^2\Lambda(T) = 10^{-1}$ erg cm$^{-3}$ s$^{-1}$
- 점성 가열: $Q_V \sim 2 \times 10^{-8}$ (채층) / Viscous heating: $Q_V \sim 2 \times 10^{-8}$ (chromosphere)
- 열전도: $Q_c \sim 3 \times 10^{-7}$ / Thermal conduction
- Ohmic: $Q_J \sim 7 \times 10^{-7}$

이러한 불일치는 고전적 수송 계수($\eta, \kappa, \sigma$)가 유도되는 기본 가정(Eq 27) -- 즉, 충돌 시간이 유체 변화 시간보다 훨씬 짧고, 평균 자유 경로가 기울기 스케일보다 훨씬 짧아야 한다는 조건:

This discrepancy arises because the basic assumptions (Eq 27) under which classical transport coefficients ($\eta, \kappa, \sigma$) are derived -- namely, collision time much shorter than fluid variation time, and mean free path much shorter than gradient scale:

$$\left(\frac{d}{dt}\right)^{-1} \gg \tau_c, \quad L \gg \lambda_c \tag{27}$$

-- 이 코로나에서 심각하게 위반되기 때문이다. 따라서 고전적 수송 패러다임을 수정하고 운동론적 수송 체계를 개발하는 것이 필수적이다.

-- are severely violated in the corona. Therefore, revising the classical transport paradigm and developing a kinetic transport scheme is essential.

### 3.5-3.6 Vlasov-Boltzmann 이론의 기초 / Basics of Vlasov-Boltzmann Theory

코로나 팽창과 태양풍 가속은 입자 VDF의 상세한 평가가 필요한 복잡한 과정이다. Vlasov-Boltzmann 방정식(Eq 9)은 위상 공간에서 종(species) $j$의 VDF 시간 진화를 기술하는 핵심 방정식이다:

Coronal expansion and solar wind acceleration are complex processes requiring detailed evaluation of particle VDFs. The Vlasov-Boltzmann equation (Eq 9) is the fundamental equation describing the time evolution of species $j$ VDF in phase space:

$$\left[\frac{\partial}{\partial t} + \mathbf{v} \cdot \frac{\partial}{\partial \mathbf{x}} + \left(\mathbf{g} + \frac{e_j}{m_j}\left(\mathbf{E} + \frac{1}{c}\mathbf{v} \times \mathbf{B}\right)\right) \cdot \frac{\partial}{\partial \mathbf{v}}\right] f_j = \left[\frac{d}{dt}f_j\right]_{c,w} \tag{9}$$

- **좌변**: 위상 공간에서의 VDF 수송 -- 자유 이동($\mathbf{v} \cdot \partial/\partial\mathbf{x}$), 중력($\mathbf{g}$), 전자기 Lorentz 힘 / **LHS**: VDF transport in phase space -- free streaming, gravity, electromagnetic Lorentz force
- **우변**: 충돌 + 파동-입자 상호작용 항. Fokker-Planck 충돌 연산자 또는 준선형 확산 연산자로 기술 / **RHS**: collision + wave-particle interaction terms. Described by Fokker-Planck collision operator or quasilinear diffusion operator

### 3.7 Vlasov-Boltzmann 방정식과 유체 이론 / V-B Equation and Fluid Theory

이동 좌표계에서 Eq 9를 다시 쓰면(Eq 10), 충돌 연산자의 구체적 형태가 드러난다. 충돌 연산자(Eq 13)는 속도 공간에서의 마찰 + 확산으로 기술된다:

Rewriting Eq 9 in a moving frame (Eq 10) reveals the specific form of the collision operator. The collision operator (Eq 13) is described as friction + diffusion in velocity space:

$$\mathcal{C}f = -\frac{\partial}{\partial \mathbf{v}} \cdot \left(\mathbf{A} - \frac{1}{2}\mathcal{D} \cdot \frac{\partial}{\partial v}\right) f \tag{13}$$

Coulomb 충돌의 경우 Rosenbluth 포텐셜(Eq 14)을 이용하면 마찰 가속도 $\mathbf{A}_i$와 확산 텐서 $\mathcal{D}_i$를 구체적으로 계산할 수 있다(Eq 15):

For Coulomb collisions, the Rosenbluth potentials (Eq 14) allow explicit calculation of frictional acceleration $\mathbf{A}_i$ and diffusion tensor $\mathcal{D}_i$ (Eq 15):

$$H_j(\mathbf{v}) = \int d^3v' \frac{f_j(\mathbf{v}')}{|\mathbf{v} - \mathbf{v}'|}, \quad G_j(\mathbf{v}) = \int d^3v' f_j(\mathbf{v}') |\mathbf{v} - \mathbf{v}'| \tag{14}$$

$$\mathbf{A}_i(\mathbf{v}) = \Gamma_{ij}\left(1+\frac{m_i}{m_j}\right)\frac{\partial}{\partial \mathbf{v}} H_j(\mathbf{v}), \quad \mathcal{D}_i(\mathbf{v}) = \Gamma_{ij}\frac{\partial^2}{\partial \mathbf{v}\partial \mathbf{v}} G_j(\mathbf{v}) \tag{15}$$

여기서 $\Gamma_{ij} = 4\pi e_i^2 e_j^2/m_i^2 \ln\Lambda$이며, $\ln\Lambda$는 Coulomb 대수이다.

where $\Gamma_{ij} = 4\pi e_i^2 e_j^2/m_i^2 \ln\Lambda$ and $\ln\Lambda$ is the Coulomb logarithm.

VDF의 속도 적률(velocity moments)을 취하면 유체 방정식이 유도된다:

Taking velocity moments of the VDF yields the fluid equations:

- **연속 방정식 / Continuity** (Eq 17): $\frac{dn}{dt} = -n\frac{\partial}{\partial \mathbf{x}} \cdot \mathbf{u}$
- **운동량 방정식 / Momentum** (Eq 18): $nm\frac{d}{dt}\mathbf{u} = -\frac{\partial}{\partial \mathbf{x}} \cdot \mathcal{P} + nq[\mathbf{E} + \frac{1}{c}\mathbf{u} \times \mathbf{B}] + \mathbf{R}$
- **내부 에너지 방정식 / Internal energy** (Eq 21): $k_B(\frac{3}{2}n\frac{dT}{dt} - T\frac{dn}{dt}) = -\mathbf{\Pi}:\frac{\partial \mathbf{u}}{\partial \mathbf{x}} - \frac{\partial}{\partial \mathbf{x}} \cdot \mathbf{q} + Q$

여기서 $\mathbf{R} = m\langle \mathbf{w}\mathcal{C}f \rangle$는 충돌/파동 운동량 전달률, $Q = \frac{m}{2}\langle w^2 \mathcal{C}f \rangle$는 체적 가열률이다. 이 moment 체계는 무한 계열이므로 **닫힘(closure)** 문제가 발생하며, 이는 수송 이론이 해결해야 할 핵심 과제이다.

Here $\mathbf{R} = m\langle \mathbf{w}\mathcal{C}f \rangle$ is the collisional/wave momentum transfer rate and $Q = \frac{m}{2}\langle w^2 \mathcal{C}f \rangle$ is the volumetric heating rate. This moment hierarchy is infinite, giving rise to the **closure problem**, the key challenge that transport theory must solve.

---

## 4. 수송 / Transport

### 4.1 충돌성 플라즈마의 수송 이론 / Transport Theory in Collisional Plasma

고전적 수송 이론의 기본 가정: 충돌이 강하여 VDF가 국소 Maxwellian에서 약하게만 벗어남. VDF를 $F_0 = F_M(n, \mathbf{u}, T, \mathbf{w})$의 다항식 전개(Eq 23)로 표현:

The basic assumption of classical transport theory: collisions are strong so the VDF deviates only weakly from a local Maxwellian. The VDF is expressed as a polynomial expansion about $F_0$ (Eq 23):

$$f(\mathbf{w}) = F_0(\mathbf{w}) + \mathbf{w} \cdot \mathbf{F}_1(\mathbf{w}) + \mathbf{ww}:\mathcal{F}_2(\mathbf{w}) + \ldots \tag{23}$$

1차 보정은 온도 기울기 $\mathcal{T}$, 밀도 기울기 $\mathcal{N}$, 운동량 전달 $\mathbf{R}$에 비례하고, 2차 보정은 속도 전단 $\mathcal{U}$에 비례한다. 이로부터 열전도, 점성 등의 수송 관계가 얻어진다: $\mathbf{\Pi} \sim \mathcal{U}$, $\mathbf{q} \sim \mathcal{T}$.

First-order corrections are proportional to temperature gradient $\mathcal{T}$, density gradient $\mathcal{N}$, and momentum transfer $\mathbf{R}$, while second-order corrections are proportional to velocity shear $\mathcal{U}$. Transport relations are obtained: $\mathbf{\Pi} \sim \mathcal{U}$, $\mathbf{q} \sim \mathcal{T}$.

그러나 빠른 태양풍에서는 충돌 횟수 $N = (\tau_c \frac{d}{dt})^{-1}$이 시간의 90%에서 $N < 5$이고, 느린 풍에서도 30-40%에서 $N > 1$이다. 따라서 Chapman-Enskog 형 전개는 수렴하지 않으며, **비섭동적 운동론적 처리(non-perturbative kinetic treatment)**가 필요하다.

However, in the fast solar wind the number of collisions $N = (\tau_c \frac{d}{dt})^{-1}$ is $N < 5$ for 90% of the time, and in slow wind $N > 1$ for only 30-40%. Therefore Chapman-Enskog type expansions do not converge, and a **non-perturbative kinetic treatment** is required.

### 4.2 천이영역에서의 Spitzer-Harm 전자 열유속 / Spitzer-Harm Heat Flux in Transition Region

고전적 전자 열전도 법칙(Spitzer-Harm, Eq 29):

Classical electron heat conduction law (Spitzer-Harm, Eq 29):

$$\mathbf{q}_e = -\kappa_e T_e^{5/2} \nabla T_e \tag{29}$$

Lie-Svendsen et al. (1999)은 시험 입자 접근법으로 천이영역(TR)에서 이 법칙이 유효함을 보였다. 그러나 Shoub (1983)과 Landi & Pantellini (2001)은 Landau-Fokker-Planck 방정식의 수치 풀이에서, 매우 낮은 Knudsen 수($\epsilon = 10^{-3}$)에서도 상당한 초열적 꼬리가 발달함을 발견하여, 이 결론에 의문을 제기했다.

Lie-Svendsen et al. (1999) showed using a test-particle approach that this law is valid in the transition region (TR). However, Shoub (1983) and Landi & Pantellini (2001) found from numerical solutions of the Landau-Fokker-Planck equation that sizable suprathermal tails develop even at very low Knudsen number ($\epsilon = 10^{-3}$), questioning this conclusion.

### 4.3 코로나에서의 고전적 전자 수송 붕괴 / Breakdown of Classical Transport in Corona

Dorelli & Scudder (2003)는 초열적 전자의 감속이 코로나에서 열을 온도 기울기 **반대 방향**(반태양 방향)으로 흐르게 할 수 있음을 보였다. Figure 9는 kappa VDF에 대해 계산된 정규화 열유속 $\theta = q_e/q_{sat}$ vs. $\kappa$를 보여준다: $\kappa < 10$이면 열이 온도가 증가하는 방향(반태양 방향)으로 흐르고, $\kappa \approx 10$에서 방향이 역전된다.

Dorelli & Scudder (2003) showed that deceleration of suprathermal electrons in the corona can allow electron heat to flow radially outward **against** the local temperature gradient (anti-Sunward). Figure 9 shows normalized heat flux $\theta = q_e/q_{sat}$ vs. $\kappa$ calculated for kappa VDFs: for $\kappa < 10$, heat flows in the direction of increasing temperature (anti-Sunward), and the direction reverses at $\kappa \approx 10$.

Landi & Pantellini (2001)은 $\kappa > 5$일 때만 고전적 열전도 법칙이 적용 가능하며, 극도로 강한 초열적 꼬리($\kappa < 4$)가 코로나 밑면에 없는 한, 코로나 밑면과 온도 극대 사이의 가파른 온도 기울기를 유지하기 위해 **파동에 의한 국소 가열**이 필요하다고 결론지었다.

Landi & Pantellini (2001) concluded that the classical heat conduction law is applicable only for $\kappa > 5$, and unless extremely strong suprathermal tails ($\kappa < 4$) exist at the coronal base, **local wave heating** is needed to sustain the steep temperature gradient between the coronal base and temperature maximum.

### 4.4 고차 자이로트로픽 다중유체 방정식 / Higher-Order Gyrotropic Multi-Fluid Equations (Eq 32-37)

자기장 방향의 이방성을 포함하는 16-moment 수송 방정식 체계(Demars & Schunk 1979):

The 16-moment transport equation set (Demars & Schunk 1979) including anisotropy along the magnetic field:

$$\frac{\partial n_s}{\partial t} + \frac{1}{A}\frac{\partial}{\partial r}(n_s u_s A) = \frac{\delta n_s}{\delta t} \tag{32}$$

$$\frac{\partial u_s}{\partial t} + u_s\frac{\partial u_s}{\partial r} = -\frac{k_B}{m_s}\left[\frac{1}{n_s}\frac{\partial(n_s T_{s\parallel})}{\partial r} + \frac{1}{A}\frac{\partial A}{\partial r}(T_{s\parallel} - T_{s\perp})\right] + \frac{e_s}{m_s}E - \frac{G_\odot M_\odot}{r^2} + \frac{1}{n_s m_s}\frac{\delta M_s}{\delta t} \tag{33}$$

$$\frac{\partial T_{s\parallel}}{\partial t} + u_s\frac{\partial T_{s\parallel}}{\partial r} = -2T_{s\parallel}\frac{\partial u_s}{\partial r} - \frac{1}{n_s k_B}\left[\frac{\partial q_{s\parallel}}{\partial r} + \frac{1}{A}\frac{\partial A}{\partial r}(q_{s\parallel} - 2q_{s\perp})\right] + \frac{1}{n_s k_B}\frac{\delta E_{s\parallel}}{\delta t} \tag{34}$$

$$\frac{\partial T_{s\perp}}{\partial t} + u_s\frac{\partial T_{s\perp}}{\partial r} = -u_s T_{s\perp}\frac{1}{A}\frac{\partial A}{\partial r} - \frac{1}{n_s k_B}\left[\frac{\partial q_{s\perp}}{\partial r} + \frac{1}{A}\frac{\partial A}{\partial r}2q_{s\perp}\right] + \frac{1}{n_s k_B}\frac{\delta E_{s\perp}}{\delta t} \tag{35}$$

이 방정식들에서 $A(r)$는 flux tube 단면적($A \propto 1/B$), $\delta/\delta t$ 항들은 종간 충돌, 파동 에너지 교환, Alfven 파압을 포함하는 소스/싱크 항이다. 모델 VDF(Eq 38)를 이용하여 교환률을 계산한다.

In these equations, $A(r)$ is the flux tube cross-sectional area ($A \propto 1/B$), and the $\delta/\delta t$ terms are source/sink terms including inter-species collisions, wave energy exchange, and Alfven wave pressure. Exchange rates are calculated using model VDFs (Eq 38).

### 4.5 Moment 전개로부터의 모델 VDF / Model VDFs from Moment Expansions

표준 모델 VDF(Eq 38)는 bi-Maxwellian에 다항식 보정함수 $\Phi_s$를 곱한 형태:

The standard model VDF (Eq 38) is a bi-Maxwellian multiplied by a polynomial correction function $\Phi_s$:

$$f_s(w_\parallel, w_\perp) = \frac{n_s}{\pi^{3/2}V_{s\parallel}V_{s\perp}^2}\exp\left[-\left(\frac{w_\parallel}{V_{s\parallel}}\right)^2 - \left(\frac{w_\perp}{V_{s\perp}}\right)^2\right](1 + \Phi_s) \tag{38}$$

Figure 10은 Li (1999)의 양성자 모델 VDF가 1 $R_\odot$에서 215 $R_\odot$까지 어떻게 진화하는지를 보여준다. 저고도에서 conic-like VDF가 형성되고, $T_{p\parallel} > T_{p\perp}$로의 진화가 나타난다. 그러나 관측된 beam과 강한 비등방성을 재현하는 데 실패하며, 이는 moment 전개의 근본적 한계를 보여준다.

Figure 10 shows how Li (1999)'s proton model VDFs evolve from 1 $R_\odot$ to 215 $R_\odot$. Conic-like VDFs form at low altitudes, and evolution toward $T_{p\parallel} > T_{p\perp}$ appears. However, it fails to reproduce observed beams and strong anisotropies, demonstrating the fundamental limitations of moment expansions.

Figure 11은 Leblanc & Hubert (1997)의 skewed weight function을 사용한 모델이 beam 재현에 더 나음을 보여주나, 수렴이 여전히 느리다.

Figure 11 shows that models using Leblanc & Hubert (1997)'s skewed weight function better reproduce beams, but convergence is still slow.

---

## 5. 플라즈마 파동과 미시불안정성 / Plasma Waves and Microinstabilities

### 5.1 코로나와 태양풍의 플라즈마 파동 / Plasma Waves in Corona and Solar Wind

태양풍에서 네 가지 주요 파동 모드와 자유 에너지원:

Four salient wave modes and free energy sources in the solar wind:

| 파동 모드 / Wave Mode | 자유 에너지원 / Free Energy Source |
|---|---|
| Ion acoustic wave | 양성자 beam, 전자 열유속 / Proton beam, electron heat flux |
| EM Alfven-cyclotron wave | 양성자 beam, core 온도 비등방성 / Proton beam, core temperature anisotropy |
| Magnetosonic wave | 양성자 beam, 이온 차등 스트리밍 / Proton beam, ion differential streaming |
| Whistler-mode / lower-hybrid wave | Core-halo 드리프트, 전자 열유속 / Core-halo drift, electron heat flux |

고주파 플라즈마 파동(Alfven/ion-cyclotron 모드)은 코로나를 태양 반지름의 일부 내에서 빠르게 가열한다는 제안이 있다(Axford & McKenzie 1992). 이 파동은 재결합 영역에서 기원하며, 빠르게 감쇠하는 자기장에서의 **frequency sweeping** 메커니즘을 통해 태양 근처에서 강한 가열을 제공한다.

High-frequency plasma waves (Alfven/ion-cyclotron mode) have been suggested to heat the corona rapidly within a fraction of a solar radius (Axford & McKenzie 1992). These waves may originate from reconnection events and provide strong heating close to the Sun through the **frequency sweeping** mechanism in the rapidly declining magnetic field.

### 5.2 분산 관계와 Landau/Cyclotron 공명 / Dispersion Relations and Resonance

평행 전파하는 좌/우 원편광 전자기파의 분산 방정식(Eq 41):

Dispersion equation for parallel-propagating left/right circularly polarized EM waves (Eq 41):

$$k_\parallel^2 = \left(\frac{\tilde{\omega}}{c}\right)^2 + \sum_j \hat{\rho}_j \left(\frac{\Omega_j}{V_A}\right)^2 \hat{\varepsilon}_j^\pm(k_\parallel, \tilde{\omega}) \tag{41}$$

유전 상수는 VDF에 대한 공명 적분(Eq 42)을 포함한다:

The dielectric constant includes a resonance integral over the VDF (Eq 42):

$$\hat{\varepsilon}_j^\pm(k_\parallel, \tilde{\omega}) = 2\pi \int_0^\infty dw_\perp w_\perp \int_{-\infty}^\infty dw_\parallel \frac{w_\perp/2}{w_\parallel - w_j^\pm} \left[\left(w_\parallel - \frac{\tilde{\omega}'(k_\parallel)}{k_\parallel}\right)\frac{\partial}{\partial w_\perp} - w_\perp\frac{\partial}{\partial w_\parallel}\right] f_j(w_\perp, w_\parallel) \tag{42}$$

**Cyclotron 공명 조건**: 입자 속도가 $w_\parallel = w_j^\pm = (\tilde{\omega}'(k_\parallel) \pm \Omega_j)/k_\parallel$을 만족할 때 발생. Doppler-shifted 파동 주파수가 입자 자이로주파수와 일치. / **Cyclotron resonance condition**: occurs when particle velocity satisfies $w_\parallel = w_j^\pm$. Doppler-shifted wave frequency matches particle gyrofrequency.

**Landau 공명**: $\omega(\mathbf{k}) - \mathbf{k} \cdot \mathbf{v} = 0$. 파동의 위상 속도와 같은 속도의 입자가 에너지를 교환. / **Landau resonance**: $\omega(\mathbf{k}) - \mathbf{k} \cdot \mathbf{v} = 0$. Particles traveling at the wave phase speed exchange energy.

Table 2는 세 가지 감쇠 영역을 정리한다:

Table 2 summarizes three distinct damping regimes:

| 공명 유형 / Resonance Type | 파수 범위 / Wave Number Range | $\beta$ 범위 / $\beta$ Range | 전파 방향 / Propagation |
|---|---|---|---|
| Proton cyclotron | $k_d < k_\parallel$ | All $\beta_p$ | Quasi-parallel |
| Electron Landau | $k_\parallel < k_d$ | All $\beta_e$ | Oblique |
| Proton Landau | $k_\parallel < k_d$ | $0.1 < \beta_p$ | Oblique |

### 5.3-5.4 Kinetic Alfven Waves (KAW)

파동벡터 성분 $k_\parallel$이 감소하면 이온-파동 공명 상호작용이 약해지고, $k_\perp$가 증가하면 Landau 공명이 중요해진다. 자기장에 대해 강하게 비스듬히(obliquely) 전파하는 파동은 **kinetic Alfven waves (KAW)**라 불린다.

As $k_\parallel$ decreases, resonant ion-wave interactions weaken, and as $k_\perp$ increases, Landau resonance becomes important. Waves propagating strongly obliquely to the field are called **kinetic Alfven waves (KAW)**.

KAW의 분산 관계(Eq 44):

KAW dispersion relation (Eq 44):

$$\frac{\omega^2}{(k_\parallel V_A)^2} = \frac{1 + (k_\perp C_s/\Omega_p)^2}{1 + (k_\perp c/\omega_e)^2} \tag{44}$$

여기서 유효 음속 $C_s = \sqrt{(\gamma_e k_B T_e + \gamma_p k_B T_p)/m_p}$. KAW는 이온 관성 길이보다 짧은 수직 파장을 가지며, 저beta 코로나에서는 전자 Landau 감쇠가 가장 가능성 높은 소산 메커니즘이다.

where effective sound speed $C_s = \sqrt{(\gamma_e k_B T_e + \gamma_p k_B T_p)/m_p}$. KAW have perpendicular wavelengths shorter than the ion inertial length, and in the low-beta corona, electron Landau damping is the most likely dissipation mechanism.

### 5.5 비선형 파동 결합과 붕괴 / Non-Linear Wave Couplings and Decays

KAW의 비선형 여기 메커니즘: 펌프 Alfven 파동의 공명적 붕괴 AW $\to$ KAW1 + KAW2. 최대 비선형 성장률(Eq 45):

Non-linear excitation mechanism of KAW: resonant decay of pump Alfven wave AW $\to$ KAW1 + KAW2. Maximum non-linear growth rate (Eq 45):

$$\gamma_{NL} = \sqrt{\Omega_p \Omega_e}\,\Gamma(\beta)\frac{\delta B}{B_0} \tag{45}$$

여기서 $\Gamma(\beta)$는 0과 1 사이의 무차원 함수. 이 붕괴는 MHD 기준으로도 매우 빠를 수 있다. 이 교차 스케일(cross-scale) 결합은 소산 영역에서의 에너지 보충 메커니즘을 제공하며, 코로나와 빠른 태양풍 가열에 기여할 수 있다.

Here $\Gamma(\beta)$ is a dimensionless function between 0 and 1. This decay can be very fast by MHD standards. This cross-scale coupling provides a mechanism for replenishing energy in the dissipation domain, and may contribute to heating of the corona and fast solar wind.

MHD 난류 시뮬레이션은 일반적으로 요동 에너지를 자기장에 수직 방향의 짧은 파장으로 전달한다. 결과적 요동이 KAW라면, 저beta 코로나에서 전자 Landau 감쇠가 지배적이어서 이온을 직접 가열/가속하기 어렵다. 반면, 평행 전파 Alfven-cyclotron 파동은 frequency sweeping과 함께 양성자와 중이온의 강한 수직 가열을 제공할 수 있다.

MHD turbulence simulations generally transfer fluctuation energy to short wavelengths perpendicular to the field. If the resulting fluctuations are KAWs, electron Landau damping dominates in the low-beta corona, making it difficult to directly heat/accelerate ions. In contrast, parallel-propagating Alfven-cyclotron waves can provide strong perpendicular heating of protons and heavy ions via frequency sweeping.

---

## 6. 파동-입자 상호작용 / Wave-Particle Interactions

이 장은 논문의 가장 중요한 기여 중 하나로, 코로나 가열 문제에 대한 운동론적 해답의 핵심을 제시한다.

This section is among the paper's most important contributions, presenting the core of kinetic answers to the coronal heating problem.

### 6.1 비탄성 Pitch-angle 확산 / Inelastic Pitch-angle Diffusion (QLT)

준선형 이론(QLT)의 기본 확산 방정식(Eq 50):

The fundamental diffusion equation of quasilinear theory (QLT) (Eq 50):

$$\frac{\partial}{\partial t} f_j(v_\parallel, v_\perp, t) = \int_{-\infty}^{+\infty} \frac{d^3k}{(2\pi)^3} \sum_M \hat{\mathcal{B}}_M(\mathbf{k}) \frac{1}{v_\perp}\frac{\partial}{\partial \alpha}\left(v_\perp \nu_{j,M}(\mathbf{k}; v_\parallel, v_\perp)\frac{\partial}{\partial \alpha} f_j\right) \tag{50}$$

여기서 pitch-angle 미분(Eq 51):

where the pitch-angle derivative (Eq 51):

$$\frac{\partial}{\partial \alpha} = v_\perp \frac{\partial}{\partial v_\parallel} - (v_\parallel - v_M(\mathbf{k}))\frac{\partial}{\partial v_\perp} \tag{51}$$

물리적 해석: QLT는 VDF와 전자기장 요동 사이의 결합에서 **2차 비선형**이지만, 요동의 성장률이 작아 VDF와 파동 PSD가 느리게 공진화한다는 의미에서 "선형"이다. 핵심은 **pitch-angle 확산**: 입자가 파동 위상 속도 $v_M$에서 정의되는 특성 곡선(원호)을 따라 확산되며, 이 과정에서 총 운동 에너지를 보존한다(wave frame에서). 파동의 에너지는 입자에 의해 흡수(감쇠)되거나 방출(성장)될 수 있다.

Physical interpretation: QLT is **quadratically nonlinear** in coupling between VDF and EM field fluctuations, but "linear" in the sense that both evolve slowly with small growth rates. The key is **pitch-angle diffusion**: particles diffuse along characteristic curves (arcs) defined by wave phase speed $v_M$, conserving total kinetic energy in the wave frame. Wave energy can be absorbed (damped) or emitted (grown) by particles.

이온-파동 완화률(Eq 53):

Ion-wave relaxation rate (Eq 53):

$$\nu_{j,M}(\mathbf{k}; v_\parallel, v_\perp) = \pi\frac{\Omega_j^2}{|k_\parallel|}\sum_{s=-\infty}^{+\infty}\delta(V_j(\mathbf{k},s) - v_\parallel)\left|\frac{1}{2}(J_{s-1}e_M^+ + J_{s+1}e_M^-) + \frac{v_\parallel}{v_\perp}J_s e_{Mz}\right|^2 \tag{53}$$

여기서 $V_j(\mathbf{k},s) = (\omega_M(\mathbf{k}) - s\Omega_j)/k_\parallel$는 $s$-차 공명 속도이며, $J_s$는 Bessel 함수이다.

where $V_j(\mathbf{k},s) = (\omega_M(\mathbf{k}) - s\Omega_j)/k_\parallel$ is the $s$-th order resonance speed and $J_s$ are Bessel functions.

### 6.2 양성자에 대한 파동 산란 효과의 증거 / Evidence for Wave Scattering Effects on Protons

QLT는 이온-cyclotron 파동과 공명하는 이온이 pitch-angle만 확산되면서 총 에너지를 보존하며, 충분한 파동 에너지가 있으면 시간적으로 점근적 상태(준선형 고원, quasilinear plateau)에 도달함을 예측한다(Eq 56):

QLT predicts that ions in resonance with ion-cyclotron waves undergo only pitch-angle diffusion while conserving total energy, and with sufficient wave power reach a time-asymptotic state (quasilinear plateau) (Eq 56):

$$f_j(v_\perp, v_\parallel) = f_j\left(v_\perp^2 - v_{\perp 0}^2 + v_\parallel^2 - 2\int_{v_{\parallel 0}}^{v_\parallel} dv_\parallel' \frac{\omega(k_\parallel)}{k_\parallel}(v_\parallel')\right) \tag{56}$$

Figure 14는 Helios에서 측정된 양성자 VDF 등고선과 준선형 고원의 비교를 보여준다. 이론적 고원(원호로 표시)이 관측된 VDF 등고선과 잘 일치하며, 이는 cyclotron 공명에 의한 pitch-angle 확산이 실제로 작동함을 시사한다. 이 관측적 증거는 파동-입자 상호작용의 가장 직접적인 확인 중 하나이다.

Figure 14 shows comparison of measured proton VDF contours with the quasilinear plateau. The theoretical plateaus (shown as arcs) agree well with observed VDF contours, suggesting that pitch-angle diffusion by cyclotron resonance actually operates. This observational evidence is one of the most direct confirmations of wave-particle interactions.

Figure 15 (Gary & Saito 2003)의 PIC 시뮬레이션은 확산이 $v_\parallel = 0$ 선을 넘어 이온을 수송할 수 있음을 보여주며, 이는 Isenberg (2004)의 kinetic shell 모델에서의 가정(cyclotron 파동이 이 선을 넘어 산란할 수 없다)에 반한다.

Figure 15 PIC simulations by Gary & Saito (2003) show that diffusion can transport ions across the $v_\parallel = 0$ line, contradicting the assumption in Isenberg (2004)'s kinetic shell model that cyclotron waves cannot scatter across this line.

### 6.3 Kinetic Shell 모델 / The Kinetic Shell Model

Isenberg (2001, 2004)의 kinetic shell (bi-shell) 모델: cyclotron 공명이 다른 모든 과정보다 훨씬 빠르다고 가정하여, 공명 입자들이 속도 공간의 중첩된 shell 위에서 한계 안정성(marginal stability)을 유지한다고 가정. 이 shell들은 비공명적 시간 스케일(팽창, 중력, 전하 분리 전기장, 자기 거울)에 따라 진화.

Isenberg (2001, 2004)'s kinetic shell (bi-shell) model: assumes cyclotron resonance proceeds much faster than any other process, so resonant particles maintain marginal stability on nested shells of constant density in velocity space. These shells evolve on the non-resonant timescale (expansion, gravity, charge-separation electric field, magnetic mirror).

이 모델의 부정적 결론: 코로나 홀에서의 양성자 가열과 가속이 평행 전파 이온-cyclotron 파동의 소산만으로는 설명되지 않음. 그러나 이는 $v_\parallel = 0$을 넘는 산란 불가능이라는 **잘못된 가정**에 기반한 것일 수 있다.

The model's negative conclusion: proton heating and acceleration in coronal holes cannot be explained by dissipation of parallel-propagating ion-cyclotron waves alone. However, this may be based on the **invalid assumption** that scattering cannot cross $v_\parallel = 0$.

### 6.4 양성자 Core 온도 비등방성의 조절 / Regulation of Proton Core Temperature Anisotropy

이론으로부터 추정된 비등방성 임계값(Eq 57):

Anisotropy threshold estimated from theory (Eq 57):

$$\frac{T_{\perp p}}{T_{\parallel p}} - 1 = \frac{S_p}{\beta_{\parallel p}^{\alpha_p}} \tag{57}$$

여기서 $S_p \approx 1$, $\alpha_p \approx 0.4$. $\beta_{\parallel p} = 8\pi n_p k_B T_{\parallel p}/B_0^2$.

where $S_p \approx 1$, $\alpha_p \approx 0.4$, $\beta_{\parallel p} = 8\pi n_p k_B T_{\parallel p}/B_0^2$.

이 관계는 태양풍이 **한계 안정성(marginal stability) 근처에서 유지됨**을 의미한다. 비등방성이 이 임계값을 초과하면 cyclotron 불안정성이 성장하여 요동을 증가시키고, 이 요동이 다시 입자를 산란시켜 비등방성을 감소시키는 자기 조절(self-regulation) 메커니즘이 작동한다.

This relation implies that the solar wind is maintained **near marginal stability**. When the anisotropy exceeds this threshold, cyclotron instability grows, enhancing fluctuations that scatter particles back to reduce the anisotropy -- a self-regulation mechanism.

Figure 17은 Helios 데이터(25,439개 데이터 포인트)에 대한 $A + 1 = T_{\perp c}/T_{\parallel c}$ vs. $\beta$ 관계를 보여준다. 최소제곱 피팅: $A = e^a \beta^b - 1$, $a = 1.505 \times 10^{-1}$, $b = -5.533 \times 10^{-1}$. 상관계수 0.78로 이론적 예측과 잘 일치한다. 빠른 태양풍에서 양성자 core 비등방성과 플라즈마 beta 사이의 강한 상관은 파동에 의한 VDF 형태 조절의 직접적 증거이다.

Figure 17 shows the $A + 1 = T_{\perp c}/T_{\parallel c}$ vs. $\beta$ relation for Helios data (25,439 data points). Least-squares fit: $A = e^a \beta^b - 1$, $a = 1.505 \times 10^{-1}$, $b = -5.533 \times 10^{-1}$. The correlation coefficient of 0.78 agrees well with theoretical predictions. The strong correlation between proton core anisotropy and plasma beta in fast solar wind is direct evidence for wave regulation of VDF shapes.

최대 양성자 산란률(Eq 58):

Maximum proton scattering rate (Eq 58):

$$\frac{\tilde{\nu}_p}{\Omega_p} = 0.15 \exp(-5.5/x_p^2) \tag{58}$$

여기서 $x_p = \beta_{\parallel p}^{0.4}(T_{\perp p}/T_{\parallel p} - 1)$. 등방성에 도달하면 산란이 효과적으로 소멸한다.

where $x_p = \beta_{\parallel p}^{0.4}(T_{\perp p}/T_{\parallel p} - 1)$. Scattering effectively ceases when isotropy is reached.

SOHO/UVCS의 원격 관측에서 O$^{5+}$ 이온의 $T_{\perp o}/T_{\parallel o}$가 100을 초과할 수 있음이 추론되었다(Kohl et al. 1998). 중이온에 대한 비등방성 임계값(Eq 59): $T_{\perp i}/T_{\parallel i} - 1 = S_i/[(m_p/m_i)\beta_{\parallel i}]^{\alpha_i}$, $\alpha_i \approx 0.4$.

Remote-sensing observations from SOHO/UVCS inferred that $T_{\perp o}/T_{\parallel o}$ of O$^{5+}$ ions may exceed 100 (Kohl et al. 1998). Anisotropy threshold for heavy ions (Eq 59): $T_{\perp i}/T_{\parallel i} - 1 = S_i/[(m_p/m_i)\beta_{\parallel i}]^{\alpha_i}$, $\alpha_i \approx 0.4$.

### 6.5 양성자 Beam의 기원과 조절 / Origin and Regulation of Proton Beams

양성자 beam은 빠른 태양풍 VDF의 두드러진 특징이다(Figure 3, 18). Beam 드리프트 속도 $v_d/V_A$와 core 플라즈마 beta $\beta_{\parallel c}$의 경험적 관계(Eq 60):

Proton beams are a salient feature of fast solar wind VDFs (Figure 3, 18). Empirical relation between beam drift speed $v_d/V_A$ and core plasma beta $\beta_{\parallel c}$ (Eq 60):

$$v_d/V_A = (2.16 \pm 0.03)\,\beta_{\parallel c}^{(0.281 \pm 0.008)} \tag{60}$$

상관계수 0.82로 놀랍도록 강한 상관. Alfven I 불안정성 임계값(Eq 61):

Correlation coefficient 0.82, a surprisingly strong correlation. Alfven I instability threshold (Eq 61):

$$v_d/V_A = \Delta_1 + \Delta_2(0.5 - n_b/n_e)^3 \tag{61}$$

Figure 19는 Helios(좌)와 Ulysses(우) 데이터를 이론적 불안정성 임계값과 비교한다. 대부분의 데이터 포인트가 불안정성 영역 아래에 분포하여, 양성자 beam이 실질적으로 안정하며 선형 불안정성에 의해 **조절(regulated)**됨을 보여준다.

Figure 19 compares Helios (left) and Ulysses (right) data with theoretical instability thresholds. Most data points lie below the instability region, showing that proton beams are practically stable and **regulated** by linear instabilities.

### 6.6 파동 결합이 선형 Beam 불안정성에 미치는 효과 / Effects of Wave Couplings on Beam Instabilities

Figure 20 (Araneda & Gomberoff 2004)은 핵심 결과를 보여준다: 큰 진폭의 Alfven-cyclotron 파동이 존재하면 beam 불안정성이 **완전히 안정화**될 수 있다. 이는 빠른 태양풍에 편재하는 유한 진폭 Alfvenic 난류가 beam 안정성에 근본적 영향을 미침을 의미하며, 표준 선형 안정성 분석에서 이를 무시한 것이 부적절함을 시사한다.

Figure 20 (Araneda & Gomberoff 2004) shows a key result: large-amplitude Alfven-cyclotron waves can **completely stabilize** beam instabilities. This means the finite-amplitude Alfvenic turbulence ubiquitous in fast solar wind has a fundamental impact on beam stability, and neglecting it in standard linear stability analyses is inadequate.

### 6.7-6.8 이온 차등 운동과 전자 열유속의 조절 / Ion Differential Motion and Electron Heat Flux Regulation

이온 차등 스트리밍 $\Delta\mathbf{V}_{\alpha,p} = \mathbf{V}_\alpha - \mathbf{V}_p$은 Alfven-cyclotron 공명과 양성자-cyclotron 감쇠에 민감하게 의존한다. 전자 열유속은 주로 halo 전자에 의해 운반되며, $q_{\parallel e} = 12.7\,R^{-3.1}\,\mu$ W/m$^2$의 스케일링을 따른다. 열유속 조절 메커니즘(Eq 62):

Ion differential streaming $\Delta\mathbf{V}_{\alpha,p} = \mathbf{V}_\alpha - \mathbf{V}_p$ depends sensitively on Alfven-cyclotron resonance and proton-cyclotron damping. Electron heat flux is primarily carried by halo electrons and follows the scaling $q_{\parallel e} = 12.7\,R^{-3.1}\,\mu$ W/m$^2$. Heat flux regulation mechanism (Eq 62):

$$q_{\parallel e} = \frac{1}{2}n_H \triangle V_H k_B \left[3(T_{\parallel H} - T_{\parallel C}) + 2(T_{\perp H} - T_{\perp C})\right] \tag{62}$$

Halo 드리프트 속도 $V_H \sim V_A$로 Alfven 속도에 밀접하게 연결되어 있으며(Figure 22), 이는 whistler-mode 파동에 의한 열유속 조절을 지지한다. Figure 24에서 측정된 열유속은 whistler 불안정성 임계값보다 항상 아래에 있어, 상한(upper bound)을 제공한다.

Halo drift speed $V_H \sim V_A$ is closely tied to the Alfven speed (Figure 22), supporting regulation of heat flux by whistler-mode waves. In Figure 24, the measured heat flux is always below the whistler instability threshold, providing an upper bound.

### 6.9 파동 흡수(방출)에 의한 플라즈마 가열(냉각) / Plasma Heating by Wave Absorption (Eq 63)

QLT에 따른 파동 가열/가속률(Eq 63):

Wave heating/acceleration rates according to QLT (Eq 63):

$$\begin{pmatrix} R_j \\ Q_{j\parallel} \\ Q_{j\perp} \end{pmatrix} = \rho_j \int_{-\infty}^{+\infty} \frac{d^3k}{(2\pi)^3}\left(\frac{\Omega_j}{k_\parallel}\right)^2 \sum_M \hat{\mathcal{B}}_M \sum_{s=-\infty}^{+\infty} \mathcal{R}_j(\mathbf{k},s) \begin{pmatrix} k_\parallel \\ 2k_\parallel V_j(\mathbf{k},s) \\ s\Omega_j \end{pmatrix} \tag{63}$$

여기서 $\mathcal{R}_j(\mathbf{k},s)$는 공명 함수(wave absorption coefficient, Eq 64)이며, 방사 전달 이론의 "파동 불투명도(wave opacity)" 역할을 한다:

where $\mathcal{R}_j(\mathbf{k},s)$ is the resonance function (wave absorption coefficient, Eq 64), playing the role of "wave opacity" in radiation transfer theory:

$$\mathcal{R}_j(\mathbf{k},s) = \text{sign}(k_\parallel)\,2\pi^2 \int_0^\infty dv_\perp |v_\perp \frac{1}{2}(J_{s-1}e_M^+ + J_{s+1}e_M^-) + V_j(\mathbf{k},s)J_s e_{Mz}|^2 \left(-\frac{\partial f_j}{\partial \alpha}\right)_{v_\parallel = V_j(\mathbf{k},s)} \tag{64}$$

**핵심**: 파동 흡수는 pitch-angle 기울기가 0이 되면(고원 형성) 소멸한다. 따라서 $\partial f_j/\partial\alpha = 0$인 고원 상태는 파동과 입자 사이의 에너지 교환이 더 이상 일어나지 않는 동적 평형이다. 실제 가열률 계산에는 VDF와 파동 PSD 모두가 알려져야 하며, 코로나에서의 파동 PSD는 경험적으로 미지이다.

**Key insight**: Wave absorption vanishes when the pitch-angle gradient becomes zero (plateau formation). Thus the plateau state with $\partial f_j/\partial\alpha = 0$ is a dynamic equilibrium where no more energy exchange between waves and particles occurs. Actual heating rate calculations require knowledge of both VDF and wave PSD, and the wave PSD in the corona is empirically unknown.

---

## 7. 운동론적 모델링 / Kinetic Modelling

### 7.1 태양풍 전자 + 양성자의 운동론적 모델 / Kinetic Models of Solar Wind Electrons and Protons

완전한 Vlasov-Boltzmann 방정식(Eq 9)을 수치적으로 풀기 위해 자이로트로픽 VDF를 가정하여 속도 좌표를 $v_\parallel$, $v_\perp$의 2차원으로 축소한다. 이는 모든 특성 시간이 이온(및 전자) 자이로주기보다 길다는 합리적 가정에 기반한다.

To numerically solve the full Vlasov-Boltzmann equation (Eq 9), a gyrotropic VDF is assumed, reducing velocity coordinates to 2D ($v_\parallel$, $v_\perp$). This is based on the reasonable assumption that all characteristic times are longer than the ion (and electron) gyroperiod.

Lie-Svendsen et al. (1997)의 전자 모델은 Fokker-Planck 방정식만으로(파동 제외) core-halo 구조와 strahl을 부분적으로 재현할 수 있음을 보였다(Figure 26). 그러나 halo는 재현하지 못하고, 전자 pitch-angle 분포는 strahl만 현실적이다. Core-halo 전자 VDF는 Coulomb 충돌과 대규모 전기장 + 중력장에 의해 형성될 수 있다.

Lie-Svendsen et al. (1997)'s electron model showed that the Fokker-Planck equation alone (without waves) can partially reproduce core-halo structure and strahl (Figure 26). However, it fails to reproduce the halo, and only the strahl electron pitch-angle distribution is realistic. The core-halo electron VDF can be produced by Coulomb collisions and large-scale electric + gravitational fields.

Tam & Chang (1999, 2001)의 전역 hybrid 모델은 파동장 확산, ambipolar 전기장, Coulomb 충돌을 포함하여 양성자 가속, 중이온 우선 가열, 이중 beam 형성을 정성적으로 설명할 수 있었으나, VDF의 정량적 세부사항은 관측과 잘 맞지 않았다.

Tam & Chang (1999, 2001)'s global hybrid model, including wave-field diffusion, ambipolar electric field, and Coulomb collisions, could qualitatively explain proton acceleration, preferential heavy-ion heating, and double-beam formation, but quantitative details of VDFs poorly matched observations.

### 7.2 코로나 이온의 운동론적 모델 / Kinetic Model of Coronal Ions

Vocks & Marsch (2001, 2002)는 **reduced VDF** 기법(Eq 65-67)을 이용한 반운동론적(semi-kinetic) 모델을 개발했다:

Vocks & Marsch (2001, 2002) developed a **semi-kinetic** model using the reduced VDF technique (Eqs 65-67):

$$\binom{F_{j\parallel}(v_\parallel)}{F_{j\perp}(v_\parallel)} = 2\pi \int_0^\infty dv_\perp v_\perp \binom{1}{v_\perp^2/2} f_j(v_\perp, v_\parallel) \tag{65}$$

Reduced Boltzmann 방정식 쌍(Eq 68-69):

Pair of reduced Boltzmann equations (Eqs 68-69):

$$\frac{\partial F_\parallel}{\partial t} + v_\parallel\frac{\partial F_\parallel}{\partial r} + \left(\frac{qE_\parallel}{m} - g(r)\right)\frac{\partial F_\parallel}{\partial v_\parallel} + \frac{1}{A}\frac{\partial A}{\partial r}\left(\frac{\partial F_\perp}{\partial v_\parallel} + v_\parallel F_\parallel\right) = \left(\frac{\delta F_\parallel}{\delta t}\right)_w + \left(\frac{\delta F_\parallel}{\delta t}\right)_c \tag{68}$$

이 모델은 coronal funnel에서의 중이온(O$^{5+}$) VDF 진화를 재현한다(Figure 27). 핵심 결과: (1) 중이온이 양성자보다 우선적으로 가열되고, (2) 큰 수직 온도 비등방성이 형성되며, (3) 이온 beam과 열유속이 형성되고, (4) VDF가 한계 안정성에 도달하여 파동 흡수/방출이 소멸하는 준선형 고원을 형성한다.

This model reproduces the VDF evolution of heavy ions (O$^{5+}$) in coronal funnels (Figure 27). Key results: (1) heavy ions are preferentially heated over protons, (2) large perpendicular temperature anisotropy develops, (3) ion beams and heat fluxes form, and (4) VDFs reach marginal stability forming quasilinear plateaus where wave absorption/emission vanishes.

Hellinger et al. (2005)의 expanding box 모델은 Alfven 파동이 radial 팽창에 의해 frequency sweeping을 통해 이온을 순차적으로 가열함을 보였다: O$^{5+}$ $\to$ alpha $\to$ proton 순서. 산소 이온은 수직 방향으로 효율적으로 가열되지만, 가속은 미미하다.

Hellinger et al. (2005)'s expanding box model showed that Alfven waves heat ions sequentially through frequency sweeping via radial expansion: O$^{5+}$ $\to$ alpha $\to$ proton order. Oxygen ions are efficiently heated in the perpendicular direction, but acceleration is minor.

### 7.3 코로나 전자의 운동론적 모델 / Kinetic Model of Coronal Electrons

Vocks & Mann (2003)은 코로나 전자에 대한 3차원(속도 공간) 운동론적 모델을 개발했다(Eq 71):

Vocks & Mann (2003) developed a 3D (velocity space) kinetic model for coronal electrons (Eq 71):

$$\frac{\partial f}{\partial t} + v_\parallel\frac{\partial f}{\partial s} + \left(g_\parallel - \frac{qE_\parallel}{m}\right)\frac{\partial f}{\partial v_\parallel} + \frac{v_\perp}{2A}\frac{\partial A}{\partial s}\left(v_\perp\frac{\partial f}{\partial v_\parallel} - v_\parallel\frac{\partial f}{\partial v_\perp}\right) = \left(\frac{\partial f}{\partial t}\right)_w + \left(\frac{\partial f}{\partial t}\right)_c \tag{71}$$

핵심 결과(Figure 28): (1) 코로나 밑면(1.014 $R_\odot$)에서 수직 온도 비등방성과 공명 고원(resonant plateaus)이 형성, (2) 6.5 $R_\odot$에서 자기장 방향의 뚜렷한 skewness(비고전적 열유속)가 발달, (3) whistler 파동에 의한 pitch-angle 산란이 전자를 태양 쪽 작은 $v_\parallel$에서 자기장에 수직한 큰 $v_\perp$로 가속, (4) 거울력(mirror force)이 전자를 좁은 field-aligned strahl로 집속.

Key results (Figure 28): (1) perpendicular temperature anisotropy and resonant plateaus form at the coronal base (1.014 $R_\odot$), (2) distinct skewness (non-classical heat flux) along the field develops at 6.5 $R_\odot$, (3) pitch-angle scattering by whistler waves accelerates electrons from small Sunward $v_\parallel$ to large $v_\perp$ perpendicular to the field, (4) mirror force focuses electrons into a narrow field-aligned strahl.

이 모델은 **초열적 코로나 전자가 코로나 플라즈마 과정에서 직접 생성될 수 있음**을 보여주는 중요한 결과이다.

This model demonstrates an important result: **suprathermal coronal electrons can be directly generated from coronal plasma processes**.

---

## 8. 요약 및 결론 / Summary and Conclusions

### 핵심 결론 / Key Conclusions

1. 약하게 충돌적인 코로나에서 파동과 입자는 플라즈마 불안정성과 파동-입자 상호작용을 통해 **긴밀하게 연결**되어 있다. 이 과정은 MHD 파동보다 훨씬 높은 주파수(MHz 영역까지)에서 작동한다. / In the weakly collisional corona, waves and particles are **intimately linked** through plasma instabilities and wave-particle interactions. These processes operate at frequencies much higher than MHD waves (up to MHz range).

2. 코로나와 태양풍의 포괄적 이론은 이방적이고 다종 유체(anisotropic, multi-species fluid) 이론에 기반해야 하며, 많은 경우 완전한 운동론적 이론이 필요하다. / A comprehensive theory of the corona and solar wind must rely on anisotropic, multi-species fluid theory, and in many cases full kinetic theory.

3. 향후 연구 방향: (a) 새로운 multi-fluid/kinetic 모델, (b) 파동 스펙트럼과 VDF의 자기 일관적(self-consistent) 진화 기술, (c) 운동론적 영역에서의 소규모 소산 과정 연구, (d) Solar Probe를 통한 0.3 AU 이내의 *in situ* 관측 필요. / Future research directions: (a) new multi-fluid/kinetic models, (b) self-consistent description of wave spectrum and VDF co-evolution, (c) study of small-scale dissipation in the kinetic domain, (d) need for *in situ* observations within 0.3 AU via Solar Probe.

---

## 핵심 시사점 / Key Takeaways

1. **유체 모델만으로는 충분하지 않다**: 태양 코로나와 태양풍의 열역학과 가속을 이해하려면 반드시 운동론적 관점이 필요하며, MHD를 넘어서야 한다. / **Fluid models alone are insufficient**: understanding the thermodynamics and acceleration of the solar corona and solar wind necessarily requires a kinetic perspective, going beyond MHD.

2. **관측된 VDF는 강하게 비Maxwell적이다**: 전자의 core-halo-strahl 구조, 양성자의 온도 비등방성과 beam, 중이온의 kappa 분포 꼬리는 비충돌적 과정과 파동-입자 상호작용의 직접적 증거이다. / **Observed VDFs are strongly non-Maxwellian**: electron core-halo-strahl structure, proton temperature anisotropy and beams, and heavy-ion kappa distribution tails are direct evidence of collisionless processes and wave-particle interactions.

3. **고전적 수송 이론이 코로나에서 붕괴된다**: 충돌 자유 경로가 온도 기울기 스케일보다 훨씬 크므로 Chapman-Enskog/Spitzer-Harm 이론의 기본 가정이 위반된다. / **Classical transport theory breaks down in the corona**: collision free paths far exceed temperature gradient scales, violating the basic assumptions of Chapman-Enskog/Spitzer-Harm theory.

4. **Cyclotron 공명이 이온 가열의 핵심 메커니즘이다**: Alfven-cyclotron 파동과의 공명에 의한 pitch-angle 확산이 양성자와 중이온의 수직 가열을 설명하며, Helios 관측(준선형 고원)에 의해 직접 확인된다. / **Cyclotron resonance is the key mechanism for ion heating**: pitch-angle diffusion from resonance with Alfven-cyclotron waves explains perpendicular heating of protons and heavy ions, directly confirmed by Helios observations (quasilinear plateaus).

5. **온도 비등방성은 불안정성에 의해 자기 조절된다**: $T_\perp/T_\parallel - 1 = S/\beta_\parallel^{\alpha}$ 형태의 임계 관계가 이론과 관측 모두에서 확인되며, 태양풍이 한계 안정성 근처에서 유지됨을 보여준다. / **Temperature anisotropy is self-regulated by instabilities**: threshold relations of the form $T_\perp/T_\parallel - 1 = S/\beta_\parallel^{\alpha}$ are confirmed by both theory and observations, showing the solar wind is maintained near marginal stability.

6. **Exospheric 모델은 한계가 있다**: 초열적 전자에 의한 전기장 구동 메커니즘이 원리적으로 작동하지만, 관측과 일치하는 결과를 얻으려면 비현실적으로 큰 전자 꼬리가 필요하다. 파동 가열이 필수적이다. / **Exospheric models have limitations**: the electric field driving mechanism via suprathermal electrons works in principle, but unrealistically large electron tails are needed to match observations. Wave heating is essential.

7. **Kinetic Alfven Waves(KAW)가 전자 가열에 중요할 수 있다**: MHD 난류 cascade가 수직 방향의 짧은 파장으로 에너지를 전달하면 KAW가 형성되고, 이는 전자 Landau 감쇠를 통해 전자를 가열한다. 그러나 이온 가열에는 평행 전파 cyclotron 파동이 여전히 필요하다. / **KAW may be important for electron heating**: if MHD turbulence cascades transfer energy to short perpendicular wavelengths forming KAW, electron Landau damping heats electrons. However, parallel-propagating cyclotron waves are still needed for ion heating.

8. **Parker Solar Probe가 핵심 관측 공백을 메울 것이다**: 이 리뷰 시점에서 *in situ* 관측은 0.3 AU(Helios perihelion) 이상으로 제한되어 있으며, 코로나에서 태양풍이 형성되는 영역의 직접 관측이 결여되어 있다. Solar Probe가 이 공백을 메울 것이다. / **Parker Solar Probe will fill the key observational gap**: at the time of this review, *in situ* observations were limited to beyond 0.3 AU (Helios perihelion), lacking direct observations of the region where the solar wind forms. Solar Probe will fill this gap.

---

## 수학적 요약 / Mathematical Summary

### Bernoulli 에너지 보존 / Bernoulli Energy Conservation (Eq 2)

$$\frac{1}{2}V^2 = \frac{\gamma}{\gamma-1}\frac{2k_B T_C}{m_p} - \frac{GM_\odot}{R_\odot}$$

코로나 팽창의 에너지 수지. 종단 속도 $V$는 코로나 온도 $T_C$와 polytropic 지수 $\gamma$에 의해 결정된다. / Energy budget of coronal expansion. Terminal speed $V$ is determined by coronal temperature $T_C$ and polytropic index $\gamma$.

### Kappa 분포 / Kappa Distribution (Eq 6)

$$f(v) = \frac{n}{(\pi\kappa v_\kappa^2)^{3/2}}\frac{\Gamma(\kappa+1)}{\Gamma(\kappa-1/2)}\left[1+\frac{v^2}{\kappa v_\kappa^2}\right]^{-(\kappa+1)}, \quad v_\kappa = \left(\frac{2\kappa-3}{\kappa}\frac{k_B T_\kappa}{m}\right)^{1/2}$$

$\kappa \to \infty$이면 Maxwell 분포, $\kappa \sim 2$-4이면 강한 초열적 꼬리. $v \gg v_\kappa$에서 $f \sim v^{-2(\kappa+1)}$ 멱법칙. / Maxwellian for $\kappa \to \infty$; strong suprathermal tails for $\kappa \sim 2$-4. Power-law tail $f \sim v^{-2(\kappa+1)}$ for $v \gg v_\kappa$.

### Vlasov-Boltzmann 방정식 / Vlasov-Boltzmann Equation (Eq 9)

$$\left[\frac{\partial}{\partial t} + \mathbf{v} \cdot \frac{\partial}{\partial \mathbf{x}} + \left(\mathbf{g} + \frac{e_j}{m_j}\left(\mathbf{E} + \frac{1}{c}\mathbf{v} \times \mathbf{B}\right)\right) \cdot \frac{\partial}{\partial \mathbf{v}}\right] f_j = \left[\frac{d}{dt}f_j\right]_{c,w}$$

운동론적 물리학의 마스터 방정식. 좌변은 위상 공간 수송, 우변은 충돌 + 파동 상호작용. / Master equation of kinetic physics. LHS is phase-space transport, RHS is collision + wave interaction.

### 충돌 연산자 (Fokker-Planck) / Collision Operator (Eq 13-15)

$$\mathcal{C}f = -\frac{\partial}{\partial \mathbf{v}} \cdot \left(\mathbf{A} - \frac{1}{2}\mathcal{D} \cdot \frac{\partial}{\partial v}\right) f$$

$$H_j(\mathbf{v}) = \int d^3v'\frac{f_j(\mathbf{v}')}{|\mathbf{v}-\mathbf{v}'|}, \quad G_j(\mathbf{v}) = \int d^3v' f_j(\mathbf{v}')|\mathbf{v}-\mathbf{v}'|$$

$$\mathbf{A}_i = \Gamma_{ij}(1+m_i/m_j)\frac{\partial}{\partial \mathbf{v}}H_j, \quad \mathcal{D}_i = \Gamma_{ij}\frac{\partial^2}{\partial \mathbf{v}\partial\mathbf{v}}G_j$$

Rosenbluth 포텐셜 $H_j$, $G_j$에 기반한 Coulomb 충돌의 표현. $\Gamma_{ij} = 4\pi e_i^2 e_j^2/m_i^2\ln\Lambda$. / Expression of Coulomb collisions based on Rosenbluth potentials. $\Gamma_{ij} = 4\pi e_i^2 e_j^2/m_i^2\ln\Lambda$.

### 연속/운동량/에너지 방정식 / Continuity/Momentum/Energy Equations (Eq 17-21)

$$\frac{dn}{dt} = -n\frac{\partial}{\partial \mathbf{x}} \cdot \mathbf{u} \tag{17}$$

$$nm\frac{d\mathbf{u}}{dt} = -\frac{\partial}{\partial \mathbf{x}} \cdot \mathcal{P} + nq\left[\mathbf{E}+\frac{1}{c}\mathbf{u}\times\mathbf{B}\right] + \mathbf{R} \tag{18}$$

$$k_B\left(\frac{3}{2}n\frac{dT}{dt} - T\frac{dn}{dt}\right) = -\mathbf{\Pi}:\frac{\partial\mathbf{u}}{\partial\mathbf{x}} - \frac{\partial}{\partial\mathbf{x}} \cdot \mathbf{q} + Q \tag{21}$$

VDF의 적률로부터 유도된 유체 방정식. 닫힘 문제가 핵심 과제. / Fluid equations derived from VDF moments. The closure problem is the key challenge.

### 고전적 수송 조건 / Classical Transport Conditions (Eq 27)

$$\left(\frac{d}{dt}\right)^{-1} \gg \tau_c, \quad L \gg \lambda_c$$

유체 변화 시간이 충돌 시간보다, 기울기 스케일이 평균 자유 경로보다 훨씬 커야 함. 코로나에서 위반됨. / Fluid variation time must be much longer than collision time, gradient scale much larger than mean free path. Violated in corona.

### Spitzer-Harm 열유속 / Spitzer-Harm Heat Flux (Eq 29)

$$\mathbf{q}_e = -\kappa_e T_e^{5/2}\nabla T_e$$

고전적 전자 열전도. 충돌 지배적 환경에서만 유효. / Classical electron heat conduction. Valid only in collision-dominated environments.

### 다중-moment 유체 방정식 / Multi-Moment Fluid Equations (Eq 32-37)

$$\frac{\partial n_s}{\partial t} + \frac{1}{A}\frac{\partial}{\partial r}(n_s u_s A) = \frac{\delta n_s}{\delta t} \tag{32}$$

$$\frac{\partial u_s}{\partial t} + u_s\frac{\partial u_s}{\partial r} = -\frac{k_B}{m_s}\left[\ldots\right] + \frac{e_s}{m_s}E - \frac{G_\odot M_\odot}{r^2} + \frac{1}{n_s m_s}\frac{\delta M_s}{\delta t} \tag{33}$$

(Eq 34-37은 $T_{s\parallel}$, $T_{s\perp}$, $q_{s\parallel}$, $q_{s\perp}$에 대한 진화 방정식) / (Eqs 34-37 are evolution equations for $T_{s\parallel}$, $T_{s\perp}$, $q_{s\parallel}$, $q_{s\perp}$)

16-moment 자이로트로픽 수송 방정식 체계. $A(r) \propto 1/B(r)$는 flux tube 면적. / 16-moment gyrotropic transport equation set. $A(r) \propto 1/B(r)$ is flux tube area.

### 분산 관계 / Dispersion Relation (Eq 41)

$$k_\parallel^2 = \left(\frac{\tilde{\omega}}{c}\right)^2 + \sum_j \hat{\rho}_j\left(\frac{\Omega_j}{V_A}\right)^2\hat{\varepsilon}_j^\pm(k_\parallel, \tilde{\omega})$$

평행 전파 좌/우 원편광 전자기파의 분산 관계. $\hat{\varepsilon}_j^\pm$는 VDF에 의존하는 유전 상수. / Dispersion relation for parallel left/right circularly polarized EM waves. $\hat{\varepsilon}_j^\pm$ is VDF-dependent dielectric constant.

### Cyclotron 공명 적분 / Cyclotron Resonance Integral (Eq 42)

$$\hat{\varepsilon}_j^\pm = 2\pi\int_0^\infty dw_\perp w_\perp \int_{-\infty}^\infty dw_\parallel \frac{w_\perp/2}{w_\parallel - w_j^\pm}\left[\left(w_\parallel - \frac{\tilde{\omega}'}{k_\parallel}\right)\frac{\partial}{\partial w_\perp} - w_\perp\frac{\partial}{\partial w_\parallel}\right]f_j$$

VDF의 pitch-angle 기울기에 대한 공명 적분. 자유 에너지(비등방성, beam, 비대칭)가 파동 성장/감쇠를 결정. / Resonance integral over pitch-angle gradient of VDF. Free energy (anisotropy, beams, skewness) determines wave growth/damping.

### 준선형 확산 / Quasilinear Diffusion (Eq 50)

$$\frac{\partial}{\partial t}f_j = \int\frac{d^3k}{(2\pi)^3}\sum_M \hat{\mathcal{B}}_M(\mathbf{k})\frac{1}{v_\perp}\frac{\partial}{\partial\alpha}\left(v_\perp\nu_{j,M}\frac{\partial}{\partial\alpha}f_j\right)$$

파동장에 의한 VDF의 pitch-angle 확산. $\hat{\mathcal{B}}_M$은 정규화 자기장 요동 스펙트럼, $\nu_{j,M}$은 이온-파동 완화율. / Pitch-angle diffusion of VDF by wave fields. $\hat{\mathcal{B}}_M$ is normalized magnetic fluctuation spectrum, $\nu_{j,M}$ is ion-wave relaxation rate.

### 비등방성 임계값 / Anisotropy Threshold (Eq 57)

$$\frac{T_{\perp p}}{T_{\parallel p}} - 1 = \frac{S_p}{\beta_{\parallel p}^{\alpha_p}}$$

$S_p \sim 1$, $\alpha_p \sim 0.4$. Cyclotron 불안정성에 의한 온도 비등방성 자기 조절. Helios 관측과 상관계수 0.78. / Self-regulation of temperature anisotropy by cyclotron instability. Correlation coefficient 0.78 with Helios observations.

### 파동 가열률 / Wave Heating Rates (Eq 63)

$$\begin{pmatrix} R_j \\ Q_{j\parallel} \\ Q_{j\perp} \end{pmatrix} = \rho_j \int\frac{d^3k}{(2\pi)^3}\left(\frac{\Omega_j}{k_\parallel}\right)^2\sum_M \hat{\mathcal{B}}_M \sum_s \mathcal{R}_j(\mathbf{k},s)\begin{pmatrix} k_\parallel \\ 2k_\parallel V_j(\mathbf{k},s) \\ s\Omega_j \end{pmatrix}$$

QLT에 의한 파동 가열/가속률. $\mathcal{R}_j$는 공명 함수("wave opacity"). VDF와 파동 PSD 모두 필요. / Wave heating/acceleration rates from QLT. $\mathcal{R}_j$ is the resonance function ("wave opacity"). Both VDF and wave PSD are required.

---

## 역사 속의 논문 / Paper in the Arc of History

```
1958  Parker               — 태양풍의 유체역학적 모델 (초음속 팽창)
                              Hydrodynamic model of solar wind (supersonic expansion)
      |
1963  Parker               — 행성간 역학 및 자기장 (Parker spiral)
                              Interplanetary dynamics and magnetic field
      |
1970s Helios 1 & 2         — 0.3 AU까지 in situ 측정, 비Maxwell VDF 최초 발견
                              In situ to 0.3 AU, first non-Maxwellian VDF discovery
      |
1975  Feldman et al.       — 전자 core-halo 구조 최초 관측
                              First electron core-halo structure observation
      |
1979  Scudder & Olbert     — exospheric 모델 초기 형태
                              Early exospheric model
      |
1982  Marsch et al.        — 양성자 VDF의 beam + 비등방성 발견
                              Proton VDF beam + anisotropy discovery
      |
1988  Tu                   — Alfven 파동 감쇠에 의한 양성자 가열 모델
                              Proton heating by Alfven wave damping
      |
1990  Schwenn              — Helios 이후 행성간 매질 종합 리뷰
                              Post-Helios review of interplanetary medium
      |
1990s Ulysses, SOHO        — 고위도 태양풍, 코로나 원격 관측
                              High-latitude wind, coronal remote sensing
      |
1992  Axford & McKenzie    — 고주파 이온-cyclotron 파동에 의한 코로나 가열 제안
                              Proposed coronal heating by high-freq ion-cyclotron waves
      |
1997  Maksimovic et al.    — kappa 분포 기반 exospheric 모델
                              Kappa-based exospheric model
      |
2001  Vocks & Marsch       — 코로나 이온의 semi-kinetic 모델
                              Semi-kinetic model of coronal ions
      |
2003  Lamy, Zouganelis     — 현대 exospheric 모델 (비단조 포텐셜)
                              Modern exospheric models (non-monotonic potential)
      |
>>>   2006  Marsch          — 이 리뷰: 운동론적 물리학의 포괄적 종합    <<<
>>>                           This review: comprehensive kinetic synthesis <<<
      |
2007  Cranmer et al.       — 코로나 가열 + 태양풍 가속의 통합 모델
                              Unified coronal heating + wind acceleration model
      |
2012  Maruca et al.        — WIND 데이터: 비등방성 임계값 확인
                              WIND data: anisotropy threshold confirmation
      |
2018+ Parker Solar Probe   — 0.05 AU까지 직접 탐사, 코로나 운동론의 직접 관측 시작
                              Direct exploration to 0.05 AU, direct coronal kinetics
```

---

## 다른 논문과의 연결 / Connections to Other Papers

### LRSP 시리즈 내 연결 / Within LRSP Series

| 논문 / Paper | 연결 / Connection |
|---|---|
| LRSP #1 Wood (2004) — Astrospheres | 항성풍의 운동론적 측면을 이 논문이 태양풍에 구체적으로 적용. Astrosphere 관측은 항성풍 속도의 제약을 제공하며, 이는 운동론적 가열 모델의 검증에 활용 가능 / Stellar wind kinetic aspects applied specifically to solar wind. Astrosphere observations constrain stellar wind speeds, usable for kinetic heating model validation |
| LRSP #2 Miesch (2005) — Large-Scale Dynamics | 태양 내부의 유체역학적 기술(MHD) vs. 이 논문의 코로나/태양풍 운동론적 기술. 내부 역학이 코로나 자기장 구조를 결정하여 운동론적 과정의 경계 조건을 설정 / Hydrodynamic (MHD) interior dynamics vs. kinetic corona/wind description. Interior dynamics determine coronal field structure, setting boundary conditions for kinetic processes |
| LRSP #3 Nakariakov & Verwichte (2005) — Coronal Oscillations | MHD 파동의 코로나 진단 → 이 논문은 MHD를 넘어 운동론적 파동(ion-cyclotron, KAW, whistler)으로 확장. 코로나 파동의 관측적 증거가 운동론적 소산 이론의 동기 / MHD coronal diagnostics → this paper extends to kinetic waves. Observational evidence of coronal waves motivates kinetic dissipation theory |
| LRSP #4 Sheeley (2005) — Magnetic Flux Transport | 광구 자기장 수송 → 코로나 funnel 구조를 결정하며, 이것이 태양풍의 기원과 운동론적 과정의 무대 / Photospheric flux transport → determines coronal funnel structure, the stage for solar wind origin and kinetic processes |
| LRSP #5 Gizon & Birch (2005) — Helioseismology | 일진학의 파동 이론이 운동론적 파동 이론의 배경 지식 제공. 내부 회전과 자기장이 코로나 가열의 에너지원과 연결 / Helioseismic wave theory provides background for kinetic wave theory. Internal rotation and fields connect to coronal heating energy source |
| LRSP #6 Longcope (2005) — Reconnection | 자기 재결합이 에너지 방출 → 운동론적 소산의 미시적 과정에서 재결합 산물(beam, 가열)이 중요. 재결합이 고주파 cyclotron 파동의 생성원일 가능성 / Magnetic reconnection as energy release → reconnection products (beams, heating) important in microscopic kinetic dissipation. Reconnection may generate high-frequency cyclotron waves |
| LRSP #7 Berdyugina (2005) — Starspots | 항성 자기 활동의 표면 현상 vs. 이 논문의 코로나/태양풍 입자 수준 물리. 항성풍의 운동론적 성질은 astrosphere를 통해 간접적으로 추론 가능 / Surface manifestation of stellar magnetic activity vs. particle-level corona/wind physics. Kinetic properties of stellar winds can be inferred indirectly through astrospheres |

### 다른 학습 논문과의 연결 / Connections to Other Studied Papers

| 논문 / Paper | 연결 / Connection |
|---|---|
| Parker (1958) — Solar Wind | Marsch의 출발점. Parker의 유체 모델을 운동론적으로 확장하고, 유체 모델의 한계를 규명 / Starting point for Marsch. Extends Parker's fluid model kinetically, identifying fluid model limitations |
| Cranmer & van Ballegooijen (2003, 2005) — Alfvenic Turbulence | MHD Alfven 난류의 코로나 소산 모델. Marsch는 이를 운동론적 영역(cyclotron 공명, KAW)으로 확장 / MHD Alfven turbulence dissipation in corona. Marsch extends this to kinetic domain (cyclotron resonance, KAW) |
| Gary (1993) — Space Plasma Microinstabilities | 분산 관계와 미시불안정성의 교과서적 참고문헌. Marsch가 광범위하게 인용하는 핵심 이론적 기반 / Textbook reference for dispersion relations and microinstabilities. Key theoretical foundation widely cited by Marsch |

---

## 참고문헌 / References

- Marsch, E., "Kinetic Physics of the Solar Corona and Solar Wind", *Living Rev. Solar Phys.*, **3**, 1, 2006. [DOI: 10.12942/lrsp-2006-1](http://www.livingreviews.org/lrsp-2006-1)
- Parker, E.N., "Dynamics of the Interplanetary Gas and Magnetic Fields", *Astrophys. J.*, **128**, 664, 1958.
- Feldman, W.C. et al., "Solar wind electrons", *J. Geophys. Res.*, **80**, 4181, 1975.
- Marsch, E. et al., "Solar wind protons: Three-dimensional velocity distributions and derived plasma parameters", *J. Geophys. Res.*, **87**, 52-72, 1982c.
- Scudder, J.D., Olbert, S., "A theory of local and global processes which affect solar wind electrons", *J. Geophys. Res.*, **84**, 2755, 1979a,b.
- Gary, S.P., *Theory of Space Plasma Microinstabilities*, Cambridge Univ. Press, 1993.
- Hollweg, J.V., Isenberg, P.A., "Generation of the fast solar wind: A review with emphasis on the resonant cyclotron interaction", *J. Geophys. Res.*, **107**, 1147, 2002.
- Cranmer, S.R., van Ballegooijen, A.A., "Alfvenic turbulence in the extended solar corona", *Astrophys. J.*, **594**, 573, 2003.
- Tu, C.-Y., Marsch, E., "MHD structures, waves and turbulence in the solar wind", *Space Sci. Rev.*, **73**, 1, 1995.
- Vocks, C., Marsch, E., "A semi-kinetic model of wave-ion interaction in the solar corona", *Geophys. Res. Lett.*, **28**, 1917, 2001.
- Maksimovic, M. et al., "Radial evolution of the electron distribution functions in the fast solar wind between 0.3 and 1.5 AU", *J. Geophys. Res.*, **110**, A09104, 2005.
- Spitzer, L., Harm, R., "Transport phenomena in a completely ionized gas", *Phys. Rev.*, **89**, 977, 1953.
