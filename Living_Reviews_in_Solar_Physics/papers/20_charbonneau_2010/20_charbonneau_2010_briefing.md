---
title: "Pre-Reading Briefing: Dynamo Models of the Solar Cycle"
paper_id: "20_charbonneau_2010"
topic: Living_Reviews_in_Solar_Physics
date: 2026-04-19
type: briefing
---

# Dynamo Models of the Solar Cycle: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Charbonneau, P. (2010). *Dynamo Models of the Solar Cycle*. Living Reviews in Solar Physics, 7, 3.
**Author(s)**: Paul Charbonneau (Université de Montréal)
**Year**: 2010
**DOI**: 10.12942/lrsp-2010-3

---

## 1. 핵심 기여 / Core Contribution

**한국어:**
이 논문은 태양 다이나모(solar dynamo) 이론의 **결정판 리뷰**로, 태양이 어떻게 약 11년 주기의 자기장을 스스로 생성·유지하는지를 설명하는 거의 모든 주요 모델들을 한 자리에 모아 정리한다. Charbonneau는 자기유체역학(MHD) 유도 방정식에서 출발하여, 평균장(mean-field) 이론의 α-효과와 Ω-효과, Babcock-Leighton 메커니즘, 그리고 자오선 순환(meridional circulation)에 의한 플럭스 수송 다이나모(flux-transport dynamo)까지 각 모델의 **물리적 가정, 지배 방정식, 성공과 한계**를 비교한다. 이 논문은 "태양 주기 모델링의 현재 수준"을 정의하는 교과서이자, 이후 10여 년간 쓰인 거의 모든 다이나모 논문의 기준점 역할을 했다.

**English:**
This paper is the **definitive review** of solar dynamo theory, bringing together in one place nearly every major model that explains how the Sun self-generates and sustains its ~11-year magnetic cycle. Starting from the magnetohydrodynamic (MHD) induction equation, Charbonneau systematically develops mean-field theory with its α- and Ω-effects, the Babcock-Leighton mechanism driven by sunspot decay, and flux-transport dynamos powered by meridional circulation. For each class of model he compares **physical assumptions, governing equations, successes, and failures**. The review has defined the "state of the art" in cycle modeling and served as the reference point for essentially every dynamo paper written in the decade that followed.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

**한국어:**
태양의 11년 흑점 주기(Schwabe, 1844), 흑점의 자기장(Hale, 1908), 그리고 주기마다 자극이 뒤바뀌는 Hale 법칙(1919)이 발견된 후, "태양은 어떻게 자기장을 만드는가?"라는 질문은 20세기 천체물리학의 중심 과제가 되었다. Larmor(1919)가 유도 다이나모 아이디어를 제시했지만, Cowling의 반(反)다이나모 정리(1933)가 축대칭 다이나모는 불가능함을 증명하여 장벽이 되었다. Parker(1955)의 cyclonic convection 개념, Steenbeck-Krause-Rädler의 평균장 전기역학(1966) 체계화, 그리고 1960–70년대 αΩ 다이나모 모델이 첫 세대 이론을 구축했다. 1980년대 이후 태양의 내부 차등 회전이 helioseismology로 관측되자(tachocline 발견, Howe 등), α-효과가 발생하는 위치와 부호에 대한 심각한 의문이 제기되었고, Babcock(1961)과 Leighton(1969)의 표면 현상 기반 모델이 재조명되었다. Charbonneau가 이 리뷰를 쓴 2010년은 **flux-transport dynamo**가 관측된 자오선 순환 속도와 Maunder Minimum 같은 grand minima를 설명할 유력 후보로 떠오르던 시기였다.

**English:**
After the discovery of the 11-year sunspot cycle (Schwabe, 1844), the magnetic nature of sunspots (Hale, 1908), and the polarity-reversal Hale's law (1919), the question "how does the Sun generate its magnetic field?" became a central problem of 20th-century astrophysics. Larmor (1919) proposed the inductive-dynamo idea, but Cowling's antidynamo theorem (1933) proved axisymmetric dynamos impossible — a decisive obstacle. Parker's (1955) cyclonic-convection concept, the Steenbeck-Krause-Rädler mean-field formalism (1966), and the αΩ models of the 1960s–70s built the first generation of theory. After helioseismology revealed the Sun's internal differential rotation and the tachocline (Howe and others, 1980s–90s), serious doubts arose about where the α-effect operates and with what sign, and the surface-based ideas of Babcock (1961) and Leighton (1969) regained prominence. By 2010, when Charbonneau wrote this review, **flux-transport dynamos** — driven by the observed meridional flow — had emerged as the leading candidate for reproducing both the cycle and grand minima such as the Maunder Minimum.

### 타임라인 / Timeline

```
1844  Schwabe — 11년 흑점 주기 / 11-yr sunspot cycle
1908  Hale — 흑점 자기장 / sunspot magnetism
1919  Larmor — 자기유체 다이나모 아이디어 / MHD dynamo idea
1933  Cowling — 반다이나모 정리 / antidynamo theorem
1955  Parker — cyclonic convection, α-effect 씨앗 / α-effect seed
1961  Babcock — 표면 플럭스 기반 주기 모델 / surface-flux cycle model
1966  Steenbeck-Krause-Rädler — 평균장 MHD / mean-field electrodynamics
1969  Leighton — 확산 전달 다이나모 / diffusive transport dynamo
1980s Helioseismology — 내부 회전 프로파일 / internal rotation profile
1995  Choudhuri, Schüssler, Dikpati — flux-transport dynamo 모형
2005  예측 논쟁 / Cycle-24 prediction controversy (Dikpati vs Choudhuri)
2010  **이 논문 / This paper** — 10년간의 종합 정리 / synthesis of a decade
```

---

## 3. 필요한 배경 지식 / Prerequisites

**한국어:**
- **자기유체역학(MHD)**: 유도 방정식 $\partial_t \mathbf{B} = \nabla\times(\mathbf{u}\times\mathbf{B}) + \eta\nabla^2\mathbf{B}$. 자기 Reynolds 수, 자기장 얼음 효과(frozen-in), 자기 확산 개념
- **벡터해석**: 축대칭 분해 $\mathbf{B} = B_\phi\hat{\phi} + \nabla\times(A\hat{\phi})$, 극벡터/축대칭 성분
- **태양 내부 구조**: 복사층·대류층, tachocline (~0.7 $R_\odot$), 차등 회전 프로파일
- **이전 논문 연결**: LRSP #2 (Hathaway — Solar Cycle), #15 (Fan — flux emergence)
- **수학 도구**: 편미분방정식 수치 해법 기초(음해 차분, 경계 조건), 고유치 문제(dynamo number)

**English:**
- **MHD basics**: the induction equation $\partial_t \mathbf{B} = \nabla\times(\mathbf{u}\times\mathbf{B}) + \eta\nabla^2\mathbf{B}$, magnetic Reynolds number, flux-freezing, and magnetic diffusion
- **Vector calculus**: axisymmetric decomposition $\mathbf{B} = B_\phi\hat{\phi} + \nabla\times(A\hat{\phi})$ into toroidal and poloidal parts
- **Solar interior**: radiative/convective zones, tachocline (~0.7 $R_\odot$), differential-rotation profile
- **Prior papers**: LRSP #2 (Hathaway — Solar Cycle), #15 (Fan — flux emergence)
- **Math tools**: basic numerical PDE methods (implicit schemes, boundary conditions), eigenvalue problems (dynamo number)

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **Toroidal / poloidal field** | 축대칭 자기장의 두 성분: $B_\phi$ (경도 방향) / $B_r, B_\theta$ (자오면). 태양 다이나모의 두 "저장고" / The two components of an axisymmetric field: $B_\phi$ (azimuthal) vs. $B_r, B_\theta$ (meridional). The two "reservoirs" of the solar dynamo |
| **Ω-effect** | 차등 회전이 poloidal 장을 늘려 toroidal 장을 만드는 과정 / Differential rotation shears poloidal field into toroidal field |
| **α-effect** | 난류 사이클론 운동(또는 Babcock-Leighton)이 toroidal → poloidal로 되돌리는 과정. Cowling 정리 회피의 핵심 / Cyclonic turbulence (or Babcock-Leighton) regenerates poloidal from toroidal. Key to bypassing Cowling's theorem |
| **Mean-field MHD** | 자기장과 속도장을 평균 + 요동으로 분해해 난류 효과를 α, β 텐서로 표현 / Decomposing fields into mean + fluctuation; turbulent effects parametrized by α, β tensors |
| **Dynamo number $D = C_\alpha C_\Omega$** | 다이나모 작동의 임계 수. 일정 값 이상에서 지수적 자기장 증폭 / Critical number for dynamo operation; exponential growth above threshold |
| **Tachocline** | 복사층(강체 회전)과 대류층(차등 회전)의 경계 얇은 전단층. ~0.7 $R_\odot$ / Thin shear layer between rigidly rotating radiative zone and differentially rotating convection zone |
| **Babcock-Leighton mechanism** | 기울어진 BMR(bipolar magnetic region)의 붕괴로 표면에 poloidal 장이 만들어지는 비-전통적 α-효과 / Non-classical α-effect where tilted BMRs decay to produce surface poloidal field |
| **Meridional circulation** | 적도→극 표면 흐름과 극→적도 심층 반류로 이루어진 1-cell 자오면 순환 / Equator-to-pole surface flow + pole-to-equator deep return flow forming a single-cell meridional circulation |
| **Flux-transport dynamo** | 자오선 순환이 생성물(toroidal/poloidal flux)을 운반하여 주기를 결정하는 모델 / Dynamo where meridional circulation transports flux and sets the cycle period |
| **Butterfly diagram** | 흑점 위도의 시간 진화 도표. 관측된 적도 방향 표류를 모형이 재현해야 함 / Time-latitude diagram of sunspot emergence; equatorward drift must be reproduced |
| **Grand minimum** | Maunder Minimum처럼 주기가 거의 정지하는 수십 년 기간. 다이나모의 확률적/간헐적 거동 / Decades-long near-shutdown (e.g., Maunder Minimum); intermittent/stochastic behavior |
| **Parker-Yoshimura rule** | 다이나모 파동의 전파 방향 법칙: $\alpha \cdot \partial_r\Omega$ 부호가 결정 / Dynamo-wave propagation rule: sign of $\alpha \cdot \partial_r\Omega$ decides direction |

---

## 5. 수식 미리보기 / Equations Preview

### (1) MHD 유도 방정식 / The MHD induction equation

$$
\frac{\partial \mathbf{B}}{\partial t} = \nabla \times (\mathbf{u}\times \mathbf{B}) - \nabla\times(\eta \nabla\times\mathbf{B})
$$

**한국어:** 모든 다이나모 이론의 출발점. 첫 항은 유도(장 생성), 둘째 항은 저항 확산(장 소멸). 이 둘의 경쟁이 다이나모의 존재 조건을 결정.
**English:** The starting point of all dynamo theory. First term: induction (field generation); second term: Ohmic diffusion (field decay). Their competition decides whether a dynamo exists.

### (2) 축대칭 분해 / Axisymmetric decomposition

$$
\mathbf{B}(r,\theta,t) = \nabla\times[A(r,\theta,t)\,\hat{\boldsymbol{\phi}}] + B(r,\theta,t)\,\hat{\boldsymbol{\phi}}
$$

**한국어:** $A$는 poloidal 벡터 퍼텐셜, $B$는 toroidal 장 크기. 두 스칼라만 풀면 3차원 자기장이 결정된다.
**English:** $A$ is the poloidal vector potential, $B$ the toroidal field strength. Solving these two scalars determines the full 3D field.

### (3) 평균장 다이나모 방정식 / Mean-field dynamo equations

$$
\begin{aligned}
\frac{\partial A}{\partial t} &= \eta_T\left(\nabla^2 - \frac{1}{\varpi^2}\right)A + \alpha B \\
\frac{\partial B}{\partial t} &= \eta_T\left(\nabla^2 - \frac{1}{\varpi^2}\right)B + \varpi(\nabla\times A\hat{\phi})\cdot\nabla\Omega - \frac{1}{\varpi}\nabla\eta_T\times\nabla(\varpi B)
$$

**한국어:** $\alpha B$ 항: α-효과가 poloidal 장 재생. $(\nabla\times A)\cdot\nabla\Omega$: Ω-효과가 toroidal 장 생성. 이 한 쌍의 PDE가 본 리뷰의 주인공이다.
**English:** The $\alpha B$ term: α-effect regenerates the poloidal field. The $(\nabla\times A)\cdot\nabla\Omega$ term: Ω-effect creates toroidal field. This pair of PDEs is the central object of the review.

### (4) 다이나모 수 / Dynamo number

$$
D = C_\alpha C_\Omega = \frac{\alpha_0 R_\odot}{\eta_T}\cdot\frac{(\Delta\Omega) R_\odot^2}{\eta_T}
$$

**한국어:** $|D|$가 임계값을 넘으면 장이 지수적으로 성장. 태양은 $D \sim 10^3$–$10^4$ 규모로 어림.
**English:** If $|D|$ exceeds a critical value, the field grows exponentially. For the Sun $D \sim 10^3$–$10^4$.

### (5) Parker-Yoshimura 부호 규칙 / Parker-Yoshimura sign rule

$$
s_P = \alpha\,\frac{\partial \Omega}{\partial r}
$$

**한국어:** $s_P > 0$이면 다이나모 파가 극방향, $s_P < 0$이면 적도방향으로 전파. 태양의 "butterfly"(적도 이동)를 맞추는 제약 조건.
**English:** $s_P > 0$ means poleward propagation; $s_P < 0$ means equatorward. This constrains models to match the observed equatorward butterfly drift.

---

## 6. 읽기 가이드 / Reading Guide

**한국어:**
이 리뷰는 매우 방대하므로(100+ 페이지) 다음 **4-pass 전략**을 권장한다.

1. **1-pass (개념 스캔, ~1시간)**: §1 서론, §4.1~§4.5 (모델 유형 표), §9 결론만 읽고 "어떤 모델들이 있는지" 지도만 그린다.
2. **2-pass (평균장 MHD, ~2-3시간)**: §2(관측 제약), §3(평균장 이론 유도), §4.2(αΩ 모델). 유도 방정식에서 식 (3)까지 손으로 따라간다.
3. **3-pass (태양에 구체화, ~3-4시간)**: §4.3~§4.5 (interface, flux-transport, Babcock-Leighton). 각 모델이 "태양의 어떤 관측 사실을 설명하려고 만든 것인지" 표로 정리.
4. **4-pass (심화/확장, 시간 허락시)**: §5(비선형 포화), §6(grand minima), §7(예측 가능성), §8(3D 글로벌 시뮬레이션). 현재 연구 전선.

**읽을 때 주의할 점:**
- Charbonneau는 MHD 유도 방정식에서 거의 모든 모델을 **체계적으로 파생**한다. 중간 대수 전개는 건너뛰어도 되지만, **각 모델을 정의하는 가정**(어떤 항을 어떻게 근사했는지)은 반드시 표로 정리할 것.
- 관측 제약(§2)을 먼저 잘 읽어두면, 이후 각 모델의 "성공/실패" 판정 기준이 분명해진다.
- Figure 4(차등 회전), Figure 22(butterfly), Figure 27(grand minima)는 이 리뷰의 "상징" 그림. 반드시 주의 깊게 볼 것.

**English:**
Because the review is very long (100+ pages), I recommend a **4-pass strategy**:

1. **Pass 1 (concept scan, ~1 hour)**: §1 Introduction, §4.1–§4.5 (the model-type tables), and §9 Conclusion. Build a mental map of "which families of models exist."
2. **Pass 2 (mean-field MHD, ~2–3 hours)**: §2 (observational constraints), §3 (mean-field derivation), §4.2 (αΩ models). Work through the derivation up to Eq. (3) by hand.
3. **Pass 3 (applied to the Sun, ~3–4 hours)**: §4.3–§4.5 (interface, flux-transport, Babcock-Leighton dynamos). For each model, tabulate which observation it is designed to explain.
4. **Pass 4 (advanced, as time permits)**: §5 (nonlinear saturation), §6 (grand minima), §7 (predictability), §8 (3D global simulations). The current research frontier.

**While reading:**
- Charbonneau **systematically derives** nearly every model from the induction equation. You can skip the intermediate algebra, but always tabulate **the defining assumptions** of each model (which terms were approximated, and how).
- Read §2 (observations) carefully first — the later "successes/failures" verdicts only make sense with those constraints in hand.
- Figure 4 (differential rotation), Figure 22 (butterfly diagram), and Figure 27 (grand minima) are the iconic plots of the review — study them closely.

---

## 7. 현대적 의의 / Modern Significance

**한국어:**
이 리뷰는 2010년 당시의 "솔라 다이나모 현재 수준"을 고정하는 기준점이 되었고, 이후 연구의 **세 방향**을 명확히 제시했다.

1. **Flux-transport dynamo의 정착**: 이후 태양 주기 예측(SC24, SC25) 연구의 기반이 됨. Dikpati-Choudhuri 예측 논쟁(2006)의 이론적 배경.
2. **Grand minima / 확률적 요소**: 단순 결정론적 PDE를 넘어서, 플럭스 수송 다이나모에 BMR emergence noise를 주입하는 현대적 연구(Cameron & Schüssler 2017 등)의 출발점.
3. **3D 글로벌 MHD 시뮬레이션과의 연결**: ASH, EULAG-MHD, Pencil Code로 2010년 이후 폭발적으로 성장. Charbonneau 본인이 2014년 후속 리뷰에서 이 분야를 확장.

실용적으로, 이 논문은 **우주 기상 예보**(태양 활동 예측), **항성 자기 주기** 연구(Sun-as-a-star), 그리고 **지구 자기장 역전**(Earth's geodynamo) 이론과도 공통 수학 구조를 공유한다. 이 논문을 읽고 나면 태양 활동 주기를 보는 눈이 질적으로 달라진다.

**English:**
This review has become the fixed reference point for the 2010 "state of the art" of solar dynamo theory and explicitly charted **three directions** for subsequent research:

1. **Consolidation of flux-transport dynamos**, which underpins later cycle-prediction work (SC24, SC25) and the famous Dikpati–Choudhuri prediction controversy (2006).
2. **Grand minima and stochasticity** — moving beyond deterministic PDEs toward flux-transport dynamos with stochastic BMR-emergence noise (e.g., Cameron & Schüssler 2017).
3. **Connection to 3D global MHD simulations** — an explosion of ASH, EULAG-MHD, Pencil Code work after 2010; Charbonneau himself extended this frontier in a 2014 follow-up review.

Practically, the paper shares mathematical structure with **space-weather forecasting** (solar activity prediction), **stellar magnetic cycles** (Sun-as-a-star), and **Earth's geodynamo**. After reading it you will see the solar cycle in a qualitatively different way.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
