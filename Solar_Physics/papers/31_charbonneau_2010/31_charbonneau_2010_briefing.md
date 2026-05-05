---
title: "Dynamo Models of the Solar Cycle — Briefing"
date: 2026-04-27
topic: Solar_Physics
tags: [solar dynamo, mean-field, Babcock-Leighton, flux transport, solar cycle, MHD]
paper: Charbonneau (2010), Living Reviews in Solar Physics 7, 3
doi: 10.12942/lrsp-2010-3
---

# Pre-Reading Briefing — Charbonneau (2010) "Dynamo Models of the Solar Cycle"
# 사전 읽기 브리핑 — Charbonneau (2010) "태양 주기의 다이나모 모델"

## 1. Why This Paper Matters / 이 논문이 중요한 이유

**English.** This is the most cited modern review of solar dynamo theory. Charbonneau gives a critical, comparative tour of every major class of solar cycle model — from classical $\alpha\Omega$ mean-field dynamos through interface dynamos, mean-field models with meridional circulation, models based on shear/buoyancy/flux-tube instabilities, to Babcock–Leighton flux-transport dynamos and large-scale MHD simulations. It also treats amplitude fluctuations, parity modulation, intermittency (Maunder-type Grand Minima), and dynamo-based cycle prediction. For anyone moving from "the Sun has a cycle" to "here is the equation set used to model it," this is the indispensable bridge.

**한국어.** 이 논문은 현대 태양 다이나모 이론의 가장 많이 인용되는 리뷰입니다. Charbonneau는 고전적 $\alpha\Omega$ 평균장 다이나모, interface 다이나모, 자오선 순환을 포함한 평균장 모델, 전단/부력/자속관 불안정에 기반한 모델, Babcock–Leighton 플럭스 수송 다이나모, 그리고 대규모 MHD 시뮬레이션까지 모든 주요 태양 주기 모델을 비판적·비교적으로 정리합니다. 진폭 변동, 패리티 변조, 간헐성(Maunder형 Grand Minima), 다이나모 기반 주기 예측까지 다룹니다. "태양에는 주기가 있다"에서 "그 주기를 모델링하는 방정식 집합은 이것이다"로 넘어가는 데 필수적인 다리 역할을 합니다.

## 2. Historical Context / 역사적 맥락

**English.** Hale (1908–1924) established the magnetic nature of sunspots and Hale's polarity laws. Larmor (1919) proposed inductive fluid motions as the source of solar magnetism. Cowling's antidynamo theorem (1933) showed purely axisymmetric flows cannot sustain an axisymmetric field — a crisis. Parker (1955) resolved it with cyclonic Coriolis-twisted convection ($\alpha$-effect), founding mean-field electrodynamics (Steenbeck, Krause, Rädler, 1960s). Babcock (1961) and Leighton (1969) proposed surface flux-transport via tilted active regions. Helioseismology (1980s+) revealed differential rotation profiles incompatible with classical mean-field dynamos, triggering modern flux-transport / Babcock–Leighton models that this 2010 review consolidates.

**한국어.** Hale (1908–1924)이 흑점의 자기적 성질과 Hale 극성 법칙을 확립했습니다. Larmor (1919)는 유체의 유도 운동을 태양 자기장의 원천으로 제안했습니다. Cowling 반다이나모 정리(1933)는 순수한 축대칭 흐름만으로는 축대칭 자기장을 유지할 수 없음을 보였습니다(위기). Parker (1955)는 Coriolis 비틀림 cyclonic convection($\alpha$ 효과)으로 이를 해결했고, 이는 평균장 전기역학(Steenbeck, Krause, Rädler, 1960년대)의 기초가 되었습니다. Babcock (1961)·Leighton (1969)은 기울어진 활동 영역에 의한 표면 자속 수송을 제안했습니다. 1980년대 이후 일진동학(helioseismology)은 고전 평균장 다이나모와 호환되지 않는 미분 회전 프로파일을 밝혀냈고, 이 2010 리뷰가 통합한 현대 플럭스 수송 / Babcock–Leighton 모델이 등장했습니다.

## 3. Prerequisites / 선수 지식

**English.** (i) MHD induction equation $\partial_t \mathbf{B} = \nabla\times(\mathbf{u}\times\mathbf{B}) - \nabla\times(\eta\nabla\times\mathbf{B})$. (ii) Poloidal–toroidal decomposition in axisymmetry: $\mathbf{B} = \nabla\times(A\hat{e}_\phi) + B\hat{e}_\phi$. (iii) Mean-field electrodynamics: $\mathcal{E} = \alpha\langle\mathbf{B}\rangle - \eta_T\nabla\times\langle\mathbf{B}\rangle$. (iv) Helioseismic differential rotation $\Omega(r,\theta)$ and the tachocline. (v) Parker's dynamo wave dispersion. Recommended prior papers: #9 Parker (1955), #10 Babcock (1961), #17 Steenbeck–Krause–Rädler.

**한국어.** (i) MHD 유도 방정식. (ii) 축대칭에서의 poloidal–toroidal 분해. (iii) 평균장 전기역학과 $\alpha$-효과·난류 확산도. (iv) 일진동학적 미분 회전 $\Omega(r,\theta)$과 tachocline. (v) Parker의 다이나모 파동 분산 관계. 추천 선행 논문: #9 Parker (1955), #10 Babcock (1961), #17 Steenbeck–Krause–Rädler.

## 4. Key Vocabulary / 핵심 용어

| Term / 용어 | Definition / 정의 |
|---|---|
| Poloidal field / 폴로이달장 | Meridional-plane component $B_r, B_\theta$ — represented by vector potential $A$. / 자오면 성분, 벡터 퍼텐셜 $A$로 표현. |
| Toroidal field / 토로이달장 | Azimuthal $B_\phi$ component — produces sunspots after buoyant rise. / 방위각 성분 $B_\phi$, 부력 상승 후 흑점 생성. |
| $\Omega$-effect / 오메가 효과 | Differential rotation shears poloidal into toroidal. / 미분 회전이 폴로이달을 토로이달로 전단. |
| $\alpha$-effect / 알파 효과 | Helical turbulence regenerates poloidal from toroidal. / 나선형 난류가 토로이달에서 폴로이달을 재생. |
| Babcock–Leighton mechanism / Babcock–Leighton 메커니즘 | Surface decay of tilted bipolar regions produces poloidal field. / 기울어진 쌍극 영역의 표면 붕괴로 폴로이달 생성. |
| Tachocline / 타코클라인 | Thin shear layer at base of convection zone where strong $\Omega$-effect operates. / 대류층 하부의 얇은 전단층. |
| Meridional circulation / 자오선 순환 | Slow flow ($\sim 20$ m/s surface poleward), conveyor belt for flux. / 느린 자오면 흐름, 자속 컨베이어 벨트. |
| Dynamo number $D$ / 다이나모 수 | $C_\alpha C_\Omega = (\alpha_0 R/\eta_T)(\Omega_0 R^2/\eta_T)$, criticality. / 임계성 결정 무차원 수. |
| Parity / 패리티 | Equatorial symmetry: dipolar (A0, antisymm.) vs quadrupolar (S0, symm.). / 적도 대칭성. |
| Butterfly diagram / 버터플라이 다이어그램 | Latitude–time map of sunspot occurrence. / 흑점 위도–시간 분포도. |

## 5. Reading Strategy / 읽기 전략

**English.** Sections 1–3 give the conceptual foundation; read carefully. Section 4 is the heart — focus on §4.2 ($\alpha\Omega$), §4.4 (meridional circulation), §4.8 (Babcock–Leighton) on first pass. Section 5 (fluctuations, intermittency) on second pass. Section 6 (open questions) gives the critical perspective. Equations to derive yourself: (i) the linearized $\alpha\Omega$ system, (ii) the dynamo-wave dispersion relation, (iii) the Babcock–Leighton surface source term.

**한국어.** 1–3장은 개념적 기초이므로 꼼꼼히 읽으세요. 4장이 핵심이며, 첫 독해에서는 §4.2 ($\alpha\Omega$), §4.4 (자오선 순환), §4.8 (Babcock–Leighton)에 집중하세요. 5장(변동·간헐성)은 두 번째 읽기에서 다루세요. 6장(미해결 문제)은 비판적 시각을 제공합니다. 직접 유도해 볼 방정식: (i) 선형화된 $\alpha\Omega$ 시스템, (ii) 다이나모 파동 분산 관계, (iii) Babcock–Leighton 표면 원천항.

## 6. Core Q&A / 핵심 질의응답

**Q1. Why does Cowling's theorem matter? / Cowling 정리가 왜 중요한가?**
- EN: It forbids axisymmetric dynamos with axisymmetric flow, forcing inclusion of either non-axisymmetric flow or parametrized $\alpha$-effect.
- KR: 축대칭 흐름만으로는 축대칭 다이나모가 불가능하므로, 비축대칭 흐름 또는 매개변수화된 $\alpha$-효과 도입이 필요합니다.

**Q2. What is the dynamo number $D$? / 다이나모 수란? **
- EN: $D = C_\alpha C_\Omega$. Above a critical value $|D| > D_c$ field amplifies exponentially; below it decays.
- KR: $D = C_\alpha C_\Omega$. 임계값을 초과하면 자기장이 지수적으로 증폭, 미달이면 감쇠합니다.

**Q3. What problem does Babcock–Leighton solve? / Babcock–Leighton이 해결하는 문제? **
- EN: It bypasses turbulent $\alpha$-quenching by sourcing poloidal field at the surface, where observations directly constrain it.
- KR: 표면에서 폴로이달장을 생성함으로써 난류 $\alpha$-quenching을 우회하고 관측으로 직접 제약 가능하게 합니다.

**Q4. Why meridional circulation? / 자오선 순환이 필요한 이유? **
- EN: It transports surface poloidal field down to the tachocline where $\Omega$-effect operates, setting cycle period $\sim L/u_0$.
- KR: 표면 폴로이달장을 $\Omega$-효과가 작용하는 tachocline까지 운반하며, 주기를 $\sim L/u_0$로 결정합니다.

**Q5. What is intermittency in the solar cycle? / 태양 주기의 간헐성? **
- EN: Episodes (e.g., Maunder Minimum 1645–1715) when cyclic activity nearly ceases — possibly from stochastic, nonlinear, threshold, or time-delay mechanisms.
- KR: 주기 활동이 거의 멈추는 시기(예: Maunder Minimum 1645–1715). 확률적·비선형·임계·시간지연 메커니즘으로 설명 시도됩니다.
