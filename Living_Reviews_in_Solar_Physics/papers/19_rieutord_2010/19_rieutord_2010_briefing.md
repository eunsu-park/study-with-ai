---
title: "Pre-Reading Briefing: The Sun's Supergranulation"
paper_id: "19_rieutord_2010"
topic: Living_Reviews_in_Solar_Physics
date: 2026-04-17
type: briefing
---

# The Sun's Supergranulation: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Rieutord, M., & Rincon, F. (2010). *The Sun's Supergranulation*. Living Reviews in Solar Physics, 7, 2. DOI: 10.12942/lrsp-2010-2
**Author(s)**: Michel Rieutord, François Rincon
**Year**: 2010

---

## 1. 핵심 기여 / Core Contribution

### 한국어
이 리뷰 논문은 태양 표면의 속도장을 지배하는 **초과립(supergranulation)** 패턴에 대한 당시까지의 관측, 이론, 수치 시뮬레이션 결과를 종합 정리합니다. 초과립은 특성 길이 약 **30–35 Mm**, 수평 속도 ~300–400 m/s, 수명 ~1–2일을 가지는 대류 세포로서, 태양 자기장의 네트워크 패턴을 형성하는 데 결정적 역할을 합니다. 그러나 **입상(granulation, ~1 Mm)** 및 거대 세포(giant cells, >100 Mm)와 달리 초과립의 **기원은 여전히 미해결 문제**로 남아 있습니다 — 단순 Rayleigh-Bénard 대류 이론으로는 설명되지 않는 "선호 스케일(preferred scale)"로서, 표준 혼합 길이 이론(mixed-length theory)과 전면적으로 충돌합니다. 저자들은 관측적 특성, 제안된 이론(재결합 이온화, 대규모 불안정성, 자기 대류 결합 등), 그리고 고해상도 3D 자기유체역학(MHD) 시뮬레이션의 가능성을 체계적으로 검토합니다.

### English
This review synthesizes observations, theory, and numerical simulations on **supergranulation** — the dominant velocity pattern on the Sun's surface with characteristic length **30–35 Mm**, horizontal velocity ~300–400 m/s, and lifetime ~1–2 days. Supergranules organize the magnetic network but, unlike granulation (~1 Mm) and giant cells (>100 Mm), their **origin remains unresolved**. They represent a "preferred scale" that standard Rayleigh-Bénard convection and mixing-length theory fail to predict. The authors systematically review observational properties, proposed theoretical mechanisms (recombination ionization, large-scale instabilities, magnetoconvective coupling), and prospects offered by high-resolution 3D MHD simulations.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

#### 한국어
초과립은 1954년 Hart가 Mt. Wilson 자기장 관측에서 태양 표면 도플러 속도장의 대규모 패턴을 처음 감지하면서 발견되었고, 1962년 Leighton, Noyes & Simon이 체계적으로 기록하며 "supergranulation"이라는 이름을 붙였습니다. 이후 40여 년 동안 SOHO/MDI (1996), Hinode (2006), 지상 태양망원경이 관측 해상도를 극적으로 향상시켰고, 헬리오사이즈몰로지(helioseismology)로 지하 유동까지 탐침할 수 있게 되었습니다. 동시에 태양 대류 시뮬레이션은 Nordlund, Stein 등의 선구적 연구를 통해 입상 스케일에서는 관측과 일치했지만, 초과립 스케일에서는 여전히 재현하지 못했습니다. 이 2010년 리뷰는 **관측 · 이론 · 시뮬레이션의 세 축이 교차하는 시점**에서 문제의 현 상태를 정리한 것입니다.

#### English
Supergranulation was first detected by Hart (1954) in Mt. Wilson magnetograph observations and named by Leighton, Noyes & Simon (1962). Over 40+ years, observational capability advanced dramatically through SOHO/MDI (1996), Hinode (2006), and modern ground-based instruments, with helioseismology probing subsurface flows. Meanwhile, convection simulations pioneered by Nordlund, Stein, and others matched granulation well but failed to reproduce supergranulation. This 2010 review consolidates the state of a problem at the intersection of observation, theory, and simulation.

### 타임라인 / Timeline

```
1954  Hart detects large-scale photospheric flows (Mt. Wilson)
1962  Leighton, Noyes & Simon — coin "supergranulation"
1964  Simon & Leighton — magnetic network ↔ supergranule boundaries
1975  November et al. — mesogranulation discovered (~5-10 Mm)
1981  Nordlund — first realistic 3D granulation simulations
1989  Duvall — time-distance helioseismology
1996  SOHO/MDI launched; subsurface supergranule flows probed
2002  Gizon, Duvall, Schou — "traveling waves" in supergranulation
2003  Hathaway — Doppler spectrum analysis reveals clear SG peak
2006  Hinode launched; sub-arcsec photospheric imaging
2010  ← This review (Rieutord & Rincon)
```

---

## 3. 필요한 배경 지식 / Prerequisites

### 한국어
1. **유체역학 기초**: Navier-Stokes 방정식, 연속방정식, 비압축성 / 압축성 근사
2. **대류 이론**:
   - Rayleigh-Bénard 대류 — 임계 Rayleigh 수 $Ra_c$, 세포 스케일 선정
   - Boussinesq / anelastic 근사
   - 혼합 길이 이론(mixing-length theory, MLT) — Böhm-Vitense
3. **태양 구조**:
   - 태양 내부 구조 (복사층, 대류층, 광구)
   - 태양 대기의 척도 높이 $H_p \sim 150$ km (광구)
   - 이온화 영역 (H, He I, He II)
4. **헬리오사이즈몰로지**: f-mode / p-mode, 시간-거리 기법, ring-diagram 분석
5. **자기유체역학(MHD)** 기초: 유도방정식, 자기력 플럭스 동결, 자기 장력
6. **관측 기법**: 도플러그램, 자기장, correlation tracking, local helioseismology
7. **사전 논문**: 논문 #16 (Nordlund et al. 2009, "Solar Surface Convection")

### English
1. **Fluid dynamics basics**: Navier-Stokes, continuity, incompressible/compressible approximations
2. **Convection theory**:
   - Rayleigh-Bénard — critical $Ra_c$, cell-scale selection
   - Boussinesq / anelastic approximations
   - Mixing-length theory (MLT) — Böhm-Vitense
3. **Solar structure**: radiative/convective zones, photospheric scale height $H_p \sim 150$ km, ionization zones (H, He I, He II)
4. **Helioseismology**: f/p-modes, time-distance, ring-diagrams
5. **MHD fundamentals**: induction equation, flux freezing, magnetic tension
6. **Observational techniques**: Dopplergrams, magnetograms, correlation tracking, local helioseismology
7. **Prior paper**: Paper #16 (Nordlund et al. 2009, "Solar Surface Convection")

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **Supergranulation** | 태양 표면 속도장 패턴, 특성 스케일 ~30–35 Mm, 수평 속도 ~300–400 m/s, 수명 ~1–2일 / Velocity pattern at Sun's surface with characteristic scale ~30-35 Mm, horizontal velocity ~300-400 m/s, lifetime ~1-2 days |
| **Granulation** | 광구에서 보이는 ~1 Mm 스케일의 대류 세포, 수명 ~5–10분 / ~1 Mm convective cells visible in photosphere, lifetime ~5-10 min |
| **Mesogranulation** | ~5–10 Mm 중간 스케일 패턴, 실체 논쟁 중 / ~5-10 Mm intermediate pattern, existence debated |
| **Giant cells** | 가설적 >100 Mm 대류 세포, 회전에 의해 조직화 / Hypothetical >100 Mm cells organized by rotation |
| **Magnetic network** | 초과립 경계에 쌓인 자기장 네트워크 / Magnetic field pattern accumulated at SG boundaries |
| **Rayleigh number** $Ra$ | 부력 대 점성·확산 효과의 비, $Ra = g\alpha\Delta T L^3/(\nu\kappa)$ / Buoyancy vs viscous-diffusive effects |
| **Reynolds decomposition** | 유동을 평균과 요동으로 분해 $u = \bar{u} + u'$ / Decomposition of flow into mean and fluctuations |
| **$k$-$\omega$ diagram** | 파수-주파수 공간 속도장 스펙트럼, 각 모드의 성질 식별 / Wavenumber-frequency power spectrum |
| **Correlation tracking** | 두 이미지에서 구조 이동 추적으로 속도장 도출 / Deriving velocity by tracking structures between images |
| **Anelastic approximation** | 음파 제거한 낮은 마하 수 대류 방정식 / Low-Mach equations filtering sound waves |
| **Local helioseismology** | 국소 영역 지하 유동 진단 기법 / Subsurface flow diagnostic over local patches |
| **Recombination ionization** | H, He 재결합으로 인한 부력 강화 / Buoyancy enhanced by H, He recombination in subsurface layers |

---

## 5. 수식 미리보기 / Equations Preview

### (1) Anelastic 대류 방정식 / Anelastic convection equations

$$
\nabla \cdot (\bar{\rho}\,\mathbf{u}) = 0
$$

$$
\bar{\rho}\left(\frac{\partial \mathbf{u}}{\partial t} + \mathbf{u}\cdot\nabla\mathbf{u}\right) = -\nabla p' + \rho'\,\mathbf{g} + \nabla\cdot\boldsymbol{\tau}
$$

- $\bar{\rho}$: 배경 밀도(background density) / background density
- $\rho', p'$: 섭동(perturbations) / perturbations
- $\boldsymbol{\tau}$: 점성 응력 텐서(viscous stress tensor) / viscous stress tensor
- 음파를 필터링하여 깊은 층 대류에 적합 / Filters sound waves — suitable for deep convection

### (2) 혼합 길이 이론 속도 척도 / Mixing-length velocity scale

$$
v_{\rm MLT} \sim \left(\frac{\ell \, F_{\rm conv}}{\rho}\right)^{1/3}
$$

- $\ell$: 혼합 길이(mixing length), 보통 $\ell \sim H_p$ / mixing length, typically $\sim H_p$
- $F_{\rm conv}$: 대류 에너지 플럭스(convective energy flux) / convective energy flux
- 이 이론은 **선호 스케일을 예측하지 못함** → 초과립 기원 문제의 핵심 / Does NOT predict a preferred scale → core puzzle of supergranulation

### (3) 파워 스펙트럼 분해 / Power spectrum analysis

$$
E(k) = \frac{1}{2}\int |\hat{\mathbf{u}}(\mathbf{k})|^2 \, \delta(|\mathbf{k}|-k) \, d^2\mathbf{k}
$$

- 태양 표면 속도장을 구면 조화 함수로 분해: $\hat{\mathbf{u}}(\ell, m)$ / Decompose surface velocity in spherical harmonics
- 초과립 피크: 구면 조화 수 $\ell \approx 120\text{–}130$ (~35 Mm) / Supergranulation peak at spherical harmonic $\ell \approx 120\text{-}130$

### (4) 자기 대류 결합 / Magnetoconvective coupling

$$
\frac{\partial \mathbf{B}}{\partial t} = \nabla \times (\mathbf{u}\times\mathbf{B}) + \eta\nabla^2\mathbf{B}
$$

- 초과립 유동이 자기장을 경계로 쓸어 모음 → 자기 네트워크 형성 / SG flow sweeps magnetic field to boundaries → magnetic network

### (5) Roudier-Rieutord SG 속도-크기 관계 / Velocity-size relation

$$
v_h \propto \sqrt{D}
$$

- $v_h$: 수평 속도, $D$: 세포 직경 / horizontal velocity, cell diameter
- 자기 유사성(self-similarity)을 시사 / Suggests self-similar scaling

---

## 6. 읽기 가이드 / Reading Guide

### 한국어
이 리뷰는 매우 광범위하므로 **다음 순서로 읽기를 권장**합니다:

1. **서론 (Sec. 1)** — 문제 설정과 왜 중요한지 파악. 여기서 "초과립은 수수께끼"라는 핵심 메시지를 놓치지 말 것.
2. **관측적 특성 (Sec. 2)** — 길이/속도/수명 수치를 **정확히 외우기**. 특히 표로 정리된 값들(Table 1 류).
3. **$k$-$\omega$ 스펙트럼 및 파형 (Sec. 3–4)** — 왜 "traveling wave"처럼 보이는가? 회전 효과와 비회전 효과를 구분.
4. **이론적 제안 (Sec. 5–6)** — 네 가지 주요 시나리오를 비교:
   - (a) 열역학적: H/He 재결합 이온화로 부력 강화
   - (b) 선형 불안정성: 태양 대류층 전체 구조에서 선호 모드
   - (c) 자기 피드백: 자기장이 세포 크기 조절
   - (d) 난류 캐스케이드: 입상 → 초과립 역 cascade
5. **시뮬레이션 현황 (Sec. 7)** — 왜 현재 시뮬레이션이 실패하는가? 해상도 vs. 영역 크기 trade-off.
6. **결론 및 전망 (Sec. 8)** — 미해결 질문 리스트.

**읽으면서 계속 물어볼 것**: "이 주장의 증거는 관측인가, 이론인가, 시뮬레이션인가?"

### English
Recommended reading order:
1. **Introduction** — problem framing; key message = supergranulation is a puzzle
2. **Observational characteristics** — memorize numbers (length/velocity/lifetime)
3. **$k$-$\omega$ spectrum and waves** — why traveling-wave appearance? separate rotation vs. non-rotation effects
4. **Theoretical proposals** — compare four scenarios: (a) thermodynamic recombination, (b) linear instability, (c) magnetic feedback, (d) turbulent cascade
5. **Simulation status** — why they fail; resolution vs. domain trade-off
6. **Conclusions** — list of open questions

**Always ask while reading**: "Is this claim based on observation, theory, or simulation?"

---

## 7. 현대적 의의 / Modern Significance

### 한국어
초과립의 미해결 문제는 2010년 이후에도 여전히 뜨거운 주제입니다:

- **DKIST(Daniel K. Inouye Solar Telescope, 2020–)**: 10 cm급 해상도로 초과립 경계의 미세 역학을 직접 관측
- **Solar Orbiter(2020–)**: 고위도 대류 구조 최초 관측, 자오면 유동과의 연결
- **PLATO, WISPR**: 다른 항성의 대류 패턴과 비교
- **DNS 수준 시뮬레이션**: Stein-Nordlund, MURaM 등이 ~100 Mm 영역에서 ~km 해상도 달성 중이지만 여전히 명확한 초과립 피크는 재현 못 함
- **연결된 응용**: 자기 네트워크 → 태양풍 기원 지점, 코로나 가열, 태양 다이나모(dynamo) 이해에 필수

이 리뷰는 향후 10–20년간의 연구 프로그램을 정의한 참고 문헌입니다. 2010년 이후의 논문들은 거의 모두 이 리뷰를 인용합니다.

### English
The SG puzzle remains hot post-2010:
- **DKIST (2020–)**: resolves fine dynamics at SG boundaries
- **Solar Orbiter (2020–)**: first views of high-latitude convection, links to meridional flow
- **Other stars (PLATO, WISPR)**: stellar convection comparisons
- **DNS-level simulations**: Stein-Nordlund, MURaM approaching large domains at km resolution but still no clear SG peak
- **Applications**: magnetic network → solar wind sources, coronal heating, solar dynamo

This review defined the research program for 10-20 years — nearly all post-2010 papers cite it.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
