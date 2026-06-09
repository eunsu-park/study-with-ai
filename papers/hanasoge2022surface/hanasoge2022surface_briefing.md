---
title: "Pre-Reading Briefing: Surface and Interior Meridional Circulation in the Sun"
paper_id: "79"
topic: Living_Reviews_in_Solar_Physics
date: 2026-04-23
type: briefing
---

# Surface and Interior Meridional Circulation in the Sun: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Hanasoge, S. M., "Surface and interior meridional circulation in the Sun", Living Reviews in Solar Physics, 19:3, 2022. DOI: 10.1007/s41116-022-00034-7
**Author(s)**: Shravan M. Hanasoge (Tata Institute of Fundamental Research, Mumbai)
**Year**: 2022

---

## 1. 핵심 기여 / Core Contribution

**한국어:**
이 리뷰 논문은 태양의 자오면 순환(meridional circulation, MC) — 적도에서 극으로 향하는 표면 흐름 (약 20 m/s)과 대류층(convection zone) 하부 어딘가에서 적도로 되돌아오는 return flow를 포함하는 축대칭(axisymmetric) 순환 시스템 — 에 관한 지난 50여 년의 관측적, 이론적 진보를 정리한다. MC는 자전(rotation, ~300 m/s, 평균 회전율의 약 7%)에 비해 매우 약하고 (~20 m/s, 1%) 다양한 체계적 오차(systematical errors)가 그 크기에 필적하기 때문에 측정이 극도로 어렵다. 저자는 흑점 추적, 자기요소 추적, 도플러 추정, supergranule 파동 추적, 링 다이어그램 분석(RDA), 시간-거리 헬리오지스몰로지(time-distance helioseismology, TD), normal-mode coupling 등 다양한 기법의 장단점을 개관하고, MC가 태양 diffeomorphism, 자기선속 수송(flux-transport) 다이나모, 대규모 각운동량 수송에서 차지하는 중심적 역할을 강조한다. 특히 return flow의 깊이와 단일/다중 셀 구조는 여전히 미해결 쟁점으로 남아 있음을 밝힌다.

**English:**
This review synthesizes roughly fifty years of observational and theoretical progress on solar meridional circulation (MC) — an axisymmetric flow system with a ~20 m/s poleward surface branch and an equatorward return flow somewhere in the convection zone. Because MC is weak (~1% of the mean rotation rate) compared with differential rotation (~300 m/s, ~7%) and is comparable in amplitude to numerous systematical biases, accurate measurements are exceptionally difficult. The paper surveys the strengths and weaknesses of techniques (sunspot tracking, magnetic-element tracking, Doppler estimation, supergranular wave tracking, ring-diagram analysis, time-distance helioseismology, normal-mode coupling) and highlights MC's central role in global angular-momentum balance and the flux-transport dynamo model. The depth of the return flow and whether MC is single- or multi-cell remain among the most important open problems.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

**한국어:**
자오면 순환의 이론적 예상은 Eddington-Sweet(1925)의 복사 내부 순환에서 출발했으나, 이는 태양 연대 스케일에서 무의미하다. 대류층 내 MC에 대한 논의는 Lebedinsky(1941)와 Kippenhahn(1959, 1963)의 "이방성 점성(anisotropic viscosity)" 모델로 이어졌고, Kippenhahn은 이를 통해 자전의 등방성 파괴가 차등자전(differential rotation)과 MC를 동시에 낳는다고 보였다. 흑점 추적에 의한 첫 MC 측정은 1913(Dyson & Maunder), 1942(Tuominen)로 거슬러 올라가나 일관된 결과를 내지 못했다. Duvall(1978, 1979)이 도플러 관측으로 ~20 m/s 극방향 흐름을 확립했고, 1993년 Duvall et al.의 time-distance helioseismology 등장, 1995년 MDI/SOHO 및 GONG 운영으로 현대적 내부 inferences가 가능해졌다. Babcock(1961)과 flux-transport dynamo 모델(Wang et al. 1991; Choudhuri et al. 1995)의 발전은 MC를 11년 주기의 페이스메이커로 자리매김시켰다.

**English:**
The theoretical story begins with Eddington-Sweet (1925) radiative circulation, irrelevant at solar timescales. Kippenhahn (1959, 1963), building on Lebedinsky (1941), showed that anisotropic turbulent viscosity in the stratified convection zone produces both differential rotation and MC. Early sunspot-tracking measurements (Dyson & Maunder 1913; Tuominen 1942) were inconclusive; Duvall (1978, 1979) used Doppler observations to establish a ~20 m/s poleward surface flow. The formal birth of time-distance helioseismology (Duvall et al. 1993), combined with the launch of GONG (1995) and MDI/SOHO (1996–2011), opened the era of interior inferences that continues with HMI/SDO (2010–). Parallel to this, the flux-transport dynamo model (Babcock 1961; Wang et al. 1991; Choudhuri et al. 1995) elevated MC to a pacemaker of the ~11-yr cycle.

### 타임라인 / Timeline

```
1925 ─── Eddington-Sweet radiative circulation
1941 ─── Lebedinsky: turbulence ↔ rotation
1959 ─── Kippenhahn anisotropic viscosity model
1961 ─── Babcock flux-transport concept
1963 ─── Kippenhahn differential rotation + MC
1978 ─── Duvall first modern Doppler MC (~20 m/s)
1988 ─── Hill ring-diagram analysis
1991 ─── Wang, Nash & Sheeley flux-transport dynamo
1993 ─── Duvall et al. time-distance helioseismology
1995 ─── GONG network begins
1996 ─── MDI/SOHO operational
1997 ─── Giles et al. first TD interior MC inversion
2010 ─── HMI/SDO launches
2010─── Hathaway & Rightmire magnetic-element tracking to 75°
2012 ─── Zhao et al. center-to-limb systematic
2013 ─── Zhao et al. two-cell in radius result
2015 ─── Rajaguru & Antia one-cell HMI inversion
2020 ─── Gizon et al. MDI+GONG+HMI single-cell inference
2022 ─── Hanasoge review (this paper)
```

---

## 3. 필요한 배경 지식 / Prerequisites

**한국어:**
- **유체역학/MHD**: 연속 방정식, 운동량 방정식, 축대칭 기하학, 소리속도 근사
- **헬리오지스몰로지 기초**: 정상모드(normal modes) $f, p$-모드, 음향파 분산, ray theory vs. Born 근사
- **Time-distance 기법**: 상관함수(cross-correlation), travel-time perturbation $\delta\tau$, sensitivity kernel
- **회전 및 자기장 물리**: 차등자전 프로파일 $\Omega(r,\theta)$, Reynolds stress, Taylor-Proudman 제약, thermal-wind balance
- **Flux-transport dynamo**: Babcock-Leighton 메커니즘, 극성 뒤집힘 ~11년 주기
- **통계/역문제(inverse problem)**: regularization, error propagation, Bayesian 접근

**English:**
- **Fluid/MHD**: mass conservation, momentum equation, axisymmetric geometry, anelastic approximation
- **Helioseismology basics**: normal modes ($f, p$), acoustic dispersion, ray theory vs. Born approximation
- **Time-distance**: cross-correlation, travel-time perturbation $\delta\tau$, sensitivity kernels
- **Rotation & MHD**: differential rotation $\Omega(r,\theta)$, Reynolds stresses, Taylor-Proudman, thermal-wind
- **Flux-transport dynamo**: Babcock-Leighton mechanism, ~11-yr polarity reversal
- **Inverse problems**: regularization, error propagation, Bayesian methods

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| Meridional Circulation (MC) | 적도→극 표면 흐름 + 대류층 내부 귀환 흐름으로 이루어진 축대칭 순환. Axisymmetric flow: equator→pole at surface plus return flow in the interior. |
| Time-distance helioseismology (TD) | 두 표면점 사이 파동의 왕복시간을 이용한 국소(local) 헬리오지스몰로지. Uses two-point cross-correlations of surface wavefield to infer subsurface flows. |
| Ring-diagram analysis (RDA) | 15°×15° 패치에서 고차 모드의 frequency shift를 이용해 국소 평균류를 추정. Uses frequency splitting of high-degree modes over finite patches. |
| Travel-time perturbation $\delta\tau$ | 배경 대비 왕복시간 차이; 흐름에 의해 유발됨. Travel-time difference due to flow, used as seismic datum. |
| Sensitivity kernel $K_i(\mathbf{x})$ | 흐름 분포와 측정값을 연결하는 선형 민감도 함수. Linear kernel linking flow distribution to seismic datum. |
| Centre-to-limb systematic | 디스크 중심→림으로 변하는 체계적 travel-time 편차; MC 신호에 필적. Systematic travel-time bias varying with heliocentric angle, comparable to MC signal. |
| Flux-transport dynamo | 자기속(flux)의 MC에 의한 수송을 dynamo 주기의 페이스메이커로 삼는 모델. Model where MC advection of magnetic flux sets the cycle period. |
| Gyroscopic pumping | Reynolds stress 발산이 MC를 구동하는 원리. Reynolds-stress divergence drives MC along/across $\mathcal{L}=r^2\sin^2\theta\,\Omega$ surfaces. |
| Thermal-wind balance | baroclinicity와 자전의 균형; 깊은 대류층 rotation 구조를 결정. Balance between baroclinicity and rotation-axis-aligned rotation deviation. |
| Babcock-Leighton mechanism | 활동영역(bipolar regions)의 선도/후행 극성 비대칭이 뒤따르는 극성 뒤집힘. Leading/following polarity asymmetry drives polar field reversal. |
| Supergranular wave | Supergranule 패턴의 파동 분산 관계; 흐름에 의해 Doppler shift. Wave-like dispersion of supergranulation used to infer MC. |
| Stream function $\psi$ | 질량 보존을 내장한 MC 표현, $\mathbf{v}=\nabla\times(\psi\mathbf{e}_\phi)$. Stream function formulation enforcing $\nabla\cdot(\rho\mathbf{v})=0$. |

---

## 5. 수식 미리보기 / Equations Preview

### 1. 각운동량 수송 방정식 / Angular momentum transport (Eq. 1)
$$
\nabla\cdot\left(\rho r\sin\theta\langle u'_\phi \mathbf{u}'_m\rangle + \rho\mathbf{u}_m r^2\sin^2\theta\,\Omega + \rho\nu r^2\sin^2\theta\,\nabla\Omega - r\sin\theta B_\phi \mathbf{B}_m - r\sin\theta\langle B'_\phi \mathbf{B}'_m\rangle\right)=0
$$

**해석 / Interpretation**: 축대칭 각운동량 수송은 Reynolds stress, MC advection, 점성, 평균 및 요동 Lorentz 응력의 균형이다. Under anelastic approximation, MC advection balances Reynolds-stress convergence/divergence: $\nabla\cdot(\rho\mathbf{u}_m\mathcal{L})=-\nabla\cdot(\rho r\sin\theta\langle u'_\phi\mathbf{u}'_m\rangle)$, with $\mathcal{L}=r^2\sin^2\theta\,\Omega$.

### 2. Thermal-wind balance (Eq. 3)
$$
\Omega_0\frac{\partial\Omega}{\partial z}=\frac{g}{2C_p r}\frac{\partial\langle S\rangle}{\partial\theta}+\mathcal{F}
$$

**해석 / Interpretation**: 깊은 대류층에서 자전 축에서의 $\Omega$ 편차 ($\partial_z\Omega$)가 위도 엔트로피 구배(baroclinicity)에 의해 유지된다. The $z$-dependence of rotation balances latitudinal entropy gradient (baroclinicity) plus stress forcing $\mathcal{F}$.

### 3. 역문제(Inverse Problem) 모델 (Eq. 5)
$$
d_i=\int_\odot d\mathbf{x}\, K_i(\mathbf{x})\,\psi(\mathbf{x})+\epsilon_i
$$

**해석 / Interpretation**: travel-time 측정값 $d_i$는 스트림 함수 $\psi$에 sensitivity kernel $K_i$를 적분한 것과 realization noise $\epsilon_i$의 합. Linear forward model relating seismic datum to flow via sensitivity kernel plus noise.

### 4. 질량 보존 / Mass conservation for MC
$$
\nabla\cdot(\rho\mathbf{v}_m)=0 \;\Rightarrow\; \mathbf{v}=\nabla\times(\psi\mathbf{e}_\phi)
$$

**해석 / Interpretation**: 축대칭 질량보존을 자동으로 만족시키는 stream function $\psi$. 표면 poleward 20 m/s 흐름이 밀도 증가($\rho\propto r^{-n}$) 때문에 깊이에서 훨씬 느린 return flow (2-5 m/s)로 변환. Encodes mass conservation: a thin dense deep layer returns at a much slower speed than the lighter surface layer.

### 5. Gyroscopic pumping principle
$$
\nabla\cdot(\rho r\sin\theta\langle u'_\phi\mathbf{u}'_m\rangle)=-\rho\mathbf{u}_m\cdot\nabla\mathcal{L}
$$

**해석 / Interpretation**: Reynolds stress 발산이 음(양)이면 MC가 회전축에서 멀어지는(가까워지는) 방향. Sign of Reynolds-stress divergence decides MC direction — negative drives MC away from rotation axis (poleward near surface).

---

## 6. 읽기 가이드 / Reading Guide

**한국어:**
1. **Section 1-2 (Intro + Surface obs)**: 관측 기법 6종 (sunspot, magnetic element, Doppler, supergranule, granule, RDA)의 특성 비교 — Fig. 1, 2는 반드시 이해할 것.
2. **Section 3 (Global angular momentum)**: Eq. (1)-(4) 유도 — Reynolds stress, gyroscopic pumping, thermal-wind balance의 상호작용. Fig. 3, 4 참고.
3. **Section 4 (Flux-transport dynamo)**: Babcock-Leighton 메커니즘과 MC의 역할.
4. **Section 5 (Simulations)**: Fig. 5 — Rossby number에 따른 단일/다중 셀 구조.
5. **Section 6-7 (Interior + Systematics)**: TD 기법 + centre-to-limb bias 4가지 주요 체계 오차. 가장 기술적으로 어려운 부분.
6. **Section 8-9 (Active regions + Inversions)**: Fig. 14-16 — 다양한 그룹 간 inversion 차이가 핵심 논쟁.
7. **Section 10 (Conclusions)**: 단일 셀 vs. 다중 셀 논쟁에 대한 저자의 잠정 입장.

**English:**
1. Start with Sections 1-2: surface observation techniques (6 methods); Figs. 1, 2 are essential.
2. Section 3: work through Eqs. (1)-(4); understand gyroscopic pumping and thermal-wind balance (Figs. 3, 4).
3. Section 4: flux-transport dynamo and Babcock-Leighton.
4. Section 5: Fig. 5 shows Rossby-number dependence of MC cell structure.
5. Sections 6-7: time-distance and systematic errors (centre-to-limb is the biggest villain).
6. Sections 8-9: active region inflows + inversion disagreements (Figs. 14-16 encapsulate the controversy).
7. Section 10: author's tentative conclusion — single-cell down to ~0.9 $R_\odot$ is most supported, but deep return flow remains uncertain.

---

## 7. 현대적 의의 / Modern Significance

**한국어:**
MC는 태양의 11년 자기 주기를 결정하는 핵심 메커니즘으로 간주되며, 항성 자기활동 예측(space weather) 및 Sun-as-a-star 모델에 직접 연결된다. 현재 HMI, GONG 데이터를 이용한 inversion 결과들이 여전히 서로 상충하며, return flow가 $0.7 R_\odot$까지 내려가는지 (single-cell) 아니면 0.9 근처에서 역전하는지 (two-cell) 해결되지 않았다. 이 질문의 답은 dynamo 주기의 물리적 기반, convection zone의 각운동량 수송 메커니즘, 그리고 항성 자전-자기장 결합 이론의 보편성에 결정적이다. 2020년대에는 normal-mode coupling 기법과 장기간 관측 축적으로 deep return flow 결정이 현실적 목표가 되고 있다.

**English:**
MC is believed to set the pacemaker of the 11-yr magnetic cycle, directly connecting to space-weather prediction and Sun-as-a-star modeling. Current HMI/GONG inversions still disagree on whether a single poleward-to-equatorward cell spans the entire convection zone or whether a two-cell structure exists with a reversal near 0.9 $R_\odot$. Resolving this determines the physical basis for the dynamo period, angular-momentum balance in the convection zone, and the universality of rotation-magnetism coupling in stars. Normal-mode coupling and continuing long-baseline datasets (HMI+MDI+GONG) make deep-return-flow determination a realistic goal for the 2020s.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
