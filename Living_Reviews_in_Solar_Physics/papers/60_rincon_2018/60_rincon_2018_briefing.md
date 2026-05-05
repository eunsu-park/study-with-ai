---
title: "Pre-Reading Briefing: The Sun's Supergranulation (2018 Update)"
paper_id: "60"
topic: Living_Reviews_in_Solar_Physics
date: 2026-04-23
type: briefing
---

# The Sun's Supergranulation (2018 Update): Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Rincon, F., & Rieutord, M., "The Sun's supergranulation", Living Reviews in Solar Physics, 15:6 (2018). DOI: 10.1007/s41116-018-0013-5
**Author(s)**: François Rincon, Michel Rieutord (IRAP, Université de Toulouse / CNRS)
**Year**: 2018

> This paper is an **update** to the 2010 Living Reviews article by Rieutord & Rincon (Paper #19 in our series). About 40 new references, major revision, switched author order. It consolidates a decade (2008-2018) of SDO/HMI, Hinode, and high-resolution numerical simulation results on the long-standing supergranulation puzzle.
>
> 본 논문은 2010년 Rieutord & Rincon의 Living Reviews 리뷰(본 시리즈의 19번 논문)에 대한 **업데이트**입니다. 약 40개의 새 참고문헌이 추가되었고 대규모 개정 및 저자 순서 변경이 있었습니다. SDO/HMI, Hinode, 고해상도 수치 시뮬레이션을 통해 지난 10년간(2008-2018) 축적된 초립상(supergranulation) 미해결 문제에 대한 연구 결과를 종합합니다.

---

## 1. 핵심 기여 / Core Contribution

**EN**: Supergranulation is a quiet-Sun photospheric cellular flow pattern with horizontal scale ~30-35 Mm, lifetime ~24-48 h, horizontal rms velocity 300-400 m/s, and much weaker vertical velocity 20-30 m/s. Despite 60+ years since Hart (1954) and Leighton et al. (1962), a predictive theory explaining *why this scale is special* remains elusive. The 2018 update argues — based on new SDO/HMI observations and large-scale numerical simulations — that supergranulation is most likely a **large-scale, buoyancy-driven, near-critical nonlinear convection** phenomenon living on the large-scale side of the photospheric convection injection spectrum, rather than a separate, independent process. The review systematically covers observations (flow measurement methods, scales, structure, rotation, magnetic effects), classical fluid theory (Rayleigh-Bénard, rotating MHD convection), and state-of-the-art numerical simulations.

**KR**: 초립상(supergranulation)은 조용한 태양 광구에 나타나는 셀 형태의 유동 패턴으로, 수평 규모 약 30-35 Mm, 수명 약 24-48시간, 수평 rms 속도 300-400 m/s, 그에 비해 훨씬 약한 수직 속도 20-30 m/s를 특징으로 합니다. Hart(1954)와 Leighton et al.(1962) 이후 60년이 넘었지만 *왜 이 규모가 특별한가*를 설명하는 예측 가능한 이론은 여전히 미해결 상태입니다. 2018년 업데이트 리뷰는 — 새로운 SDO/HMI 관측과 대규모 수치 시뮬레이션을 바탕으로 — 초립상이 **대규모 부력 구동 임계 근접(near-critical) 비선형 대류** 현상이며 광구 대류 주입 스펙트럼의 대규모 측에 위치할 가능성이 가장 높다고 주장합니다. 관측(유동 측정법, 규모와 구조, 자전, 자기장 효과), 고전 유체 이론(Rayleigh-Bénard, 회전 MHD 대류), 최신 수치 시뮬레이션을 체계적으로 다룹니다.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

**EN**: The supergranulation story began in Oxford in 1953 when Avril B. Hart reported a "noisy" fluctuating velocity field atop the mean solar rotation. Hart (1956) gave an accurate 26 Mm estimate. Leighton et al. (1962) established supergranulation as a persistent quiet-Sun feature via the first Doppler images; Simon & Leighton (1964) linked it to the magnetic network. From 1980s-2000s, SOHO/MDI brought continuous full-disk coverage. From 2010 onwards, SDO/HMI (launched 2010) and Hinode/SOT delivered the high-resolution, long-baseline datasets that drive the 2018 update's new conclusions. At the same time, petascale computing has finally made realistic large-scale stratified convection simulations feasible.

**KR**: 초립상 연구는 1953년 옥스퍼드에서 Avril B. Hart가 태양 평균 자전 속도 위의 "잡음"처럼 요동하는 속도장을 보고하면서 시작되었습니다. Hart(1956)는 26 Mm 정도의 정확한 추정값을 제시했고, Leighton et al.(1962)은 최초의 Doppler 이미지를 통해 초립상을 조용한 태양의 지속적 특징으로 확립했습니다. Simon & Leighton(1964)은 이를 자기 네트워크(magnetic network)와 연결시켰습니다. 1980-2000년대에는 SOHO/MDI가 연속 전면 디스크 관측을 제공했습니다. 2010년 이후 SDO/HMI(2010년 발사)와 Hinode/SOT가 고해상도 장기 데이터셋을 공급하면서 2018 업데이트의 새로운 결론을 이끌었습니다. 동시에 페타스케일 컴퓨팅 덕분에 현실적인 대규모 성층 대류 시뮬레이션이 가능해졌습니다.

### 타임라인 / Timeline

```
1915 ─── Plaskett — possible first detection in noise
  │
1953-54 ─ Hart — discovery of supergranulation noise
  │
1956 ──── Hart — 26 Mm scale estimate
  │
1962 ──── Leighton, Noyes, Simon — Doppler imaging confirms pattern
  │
1964 ──── Simon & Leighton — link to magnetic network
  │
1981 ──── November et al. — claim of mesogranulation (8 Mm)
  │
1989 ──── Wang & Zirin — lifetime depends on tracer
  │
1996-2010  SOHO/MDI era — continuous Dopplergrams
  │
2003-2007  Gizon/Duvall, local helioseismology — depth, superrotation, anticyclones
  │
2009-2010  Hinode/SOT — detailed surface flow maps
  │
2010 ──── Rieutord & Rincon LRSP review (Paper #19) ─── previous version
  │
2010 ──── SDO/HMI launch — continuous high-res, long-baseline data
  │
2011-2017  Williams & Pensell, Hathaway, Rincon et al. — full-disk spectra
  │
2014-2016  Langfellner et al., Greer et al. — subsurface dynamics, vorticity
  │
★ 2018 ── Rincon & Rieutord — UPDATED review (this paper)
  │
2020+ ─── (future: DKIST, improved MHD simulations)
```

---

## 3. 필요한 배경 지식 / Prerequisites

**EN**:
- **Fluid dynamics**: Navier-Stokes equations, anelastic approximation, scale heights, Reynolds number
- **Thermal convection**: Rayleigh-Bénard setup, Schwarzschild stability criterion, Boussinesq vs. stratified convection
- **Solar structure**: solar convection zone (SCZ, outer 30% radius), granulation (1 Mm, ~5-10 min lifetime), mesogranulation range, supergranulation
- **Observational techniques**: Doppler imaging, local correlation tracking (LCT), coherent structure tracking (CST), ball tracking (BT), local helioseismology (f-modes, ring diagrams, time-distance)
- **Spectral analysis**: power spectrum on a sphere (spherical harmonic degree ℓ), horizontal wavenumber k, kinetic energy spectrum E(k)
- **MHD and turbulence**: magnetic Prandtl number, Kolmogorov vs. Bolgiano-Obukhov phenomenology, large-scale dynamo concepts
- **Rotation**: Rossby number, Taylor-Proudman constraint, Coriolis force
- **Previous paper**: Paper #19 (Rieutord & Rincon 2010 LRSP) — this is its update

**KR**:
- **유체역학**: Navier-Stokes 방정식, 비탄성(anelastic) 근사, 스케일 높이, Reynolds 수
- **열대류**: Rayleigh-Bénard 셋업, Schwarzschild 안정도 기준, Boussinesq 대 성층 대류
- **태양 구조**: 태양 대류층(SCZ, 외곽 30% 반지름), 입상(granulation, 1 Mm, 5-10분 수명), 중간립상 범위, 초립상
- **관측 기법**: Doppler 이미징, 국소 상관 추적(LCT), 결맞음 구조 추적(CST), 볼 추적(BT), 국소 일식학(f-모드, 링 다이어그램, 시간-거리)
- **스펙트럼 분석**: 구면 위 파워 스펙트럼(구면 조화 차수 ℓ), 수평 파수 k, 운동 에너지 스펙트럼 E(k)
- **MHD와 난류**: 자기 Prandtl 수, Kolmogorov 대 Bolgiano-Obukhov 현상학, 대규모 발전기(dynamo) 개념
- **자전**: Rossby 수, Taylor-Proudman 제약, Coriolis 힘
- **선행 논문**: Paper #19 (Rieutord & Rincon 2010 LRSP) — 본 논문은 그 업데이트

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **Supergranulation / 초립상** | Quiet-Sun photospheric cellular flow, scale ~30-35 Mm, lifetime ~24-48 h, horizontal velocity 300-400 m/s / 조용한 태양 광구 셀 유동, 규모 30-35 Mm, 수명 24-48시간 |
| **Granulation / 입상** | Smaller-scale convection cells, ~1 Mm, lifetime 5-10 min, velocity 1-2 km/s / 소규모 대류 셀, 1 Mm, 수명 5-10분 |
| **Mesogranulation / 중간립상** | Contested intermediate scale (~8 Mm); 2018 review treats it as "mesoscale range" without special status / 논란이 있는 중간 규모, 본 리뷰는 특별한 물리적 지위 없이 "중간 규모 대역"으로 취급 |
| **Solar convection zone (SCZ) / 태양 대류층** | Outer ~30% of solar radius where energy is transported by convection / 에너지가 대류로 전달되는 외곽 30% |
| **Schwarzschild criterion / Schwarzschild 기준** | Stability: $(dT/dr)_{\text{actual}} > (dT/dr)_{\text{ad}}$ (remember signs are negative) → unstable to convection / 대류 불안정 조건 |
| **Bolgiano scale $L_B$ / Bolgiano 규모** | Scale at which buoyancy becomes as important as inertia in stratified turbulence / 성층 난류에서 부력이 관성과 동등해지는 규모 |
| **Rossby number $Ro$ / Rossby 수** | $Ro = V/(2\Omega L)$; measures rotational influence on dynamics / 회전의 동역학적 영향 측정 |
| **Rayleigh-Bénard convection** | Classical setup: fluid between hot lower and cold upper plate / 아래 뜨거운 판과 위 차가운 판 사이 유체의 고전적 대류 |
| **Magnetic network / 자기 네트워크** | Strong magnetic flux concentration at supergranule boundaries / 초립상 경계의 강한 자속 집중 |
| **Doppler imaging / Doppler 이미징** | Line-of-sight velocity via Doppler shift; primary supergranulation detection / 시선 방향 속도 측정, 주요 초립상 탐지법 |
| **Local correlation tracking (LCT)** | Tracks small features to infer large-scale flows / 작은 구조를 추적해 대규모 유동 추론 |
| **f-mode helioseismology** | Uses surface-gravity wave modes to probe near-surface flows / 표면-중력 모드로 표층 유동 탐사 |
| **Anelastic approximation / 비탄성 근사** | Filters sound waves while keeping density stratification / 음파 제거, 밀도 성층 유지 |
| **Superrotation / 초자전** | Supergranulation pattern rotates ~4% faster than the plasma / 초립상 패턴이 플라즈마보다 4% 빠르게 자전 |

---

## 5. 수식 미리보기 / Equations Preview

### (a) Rossby number / Rossby 수
$$Ro = \frac{V}{2\Omega L} = (2\Omega \tau)^{-1}$$

**EN**: $\tau_{SG} \sim 1.7$ days, rotation period 27 days → $Ro_{SG} \sim 2$-3. Rotation moderately affects supergranules.
**KR**: $\tau_{SG} \sim 1.7$일, 자전 주기 27일 → $Ro_{SG} \sim 2$-3. 자전이 초립상에 중간 정도의 영향.

### (b) Anelastic mass conservation / 비탄성 질량 보존
$$\partial_z v_z = -v_z \partial_z \ln\rho - \nabla_h \cdot \mathbf{v}_h$$

**EN**: Links horizontal/vertical flow and density stratification → used to infer vertical structure.
**KR**: 수평/수직 유동과 밀도 성층을 연결 → 수직 구조 추론에 사용.

### (c) Viscous dissipation scale / 점성 소산 규모 (Kolmogorov)
$$\ell_\nu \sim Re^{-3/4} L$$

**EN**: With $Re \sim 10^{10}$-$10^{13}$ in the SCZ, $\ell_\nu \sim 10^{-3}$ m near the surface — far below any observable/simulatable scale.
**KR**: SCZ에서 $Re \sim 10^{10}$-$10^{13}$일 때 표면 근처 $\ell_\nu \sim 10^{-3}$ m — 관측/시뮬레이션 불가능한 작은 규모.

### (d) Velocity from kinetic energy spectrum / 운동 에너지 스펙트럼에서 속도
$$V_\lambda = \sqrt{k \, E_h(k)}, \quad k = 2\pi/\lambda$$

**EN**: At $\lambda = 36$ Mm, $E_h \sim 500$ km$^3$/s$^2$ → $V_\lambda \simeq 300$ m/s, consistent with Doppler.
**KR**: $\lambda = 36$ Mm에서 $E_h \sim 500$ km$^3$/s$^2$ → $V_\lambda \simeq 300$ m/s, Doppler 측정과 일치.

### (e) Scale ordering / 규모 순서
$$\ell_\nu \ll \ell_\eta \ll \ell_\kappa \sim H_p \sim H_\rho \sim L_B \sim L_G \ll L_{SG} \ll R_\odot$$

**EN**: Dissipation $\ll$ granulation ($L_G \sim 1$ Mm) $\ll$ supergranulation ($L_{SG} \sim 30$ Mm) $\ll$ solar radius.
**KR**: 소산 $\ll$ 입상 $\ll$ 초립상 $\ll$ 태양 반지름.

---

## 6. 읽기 가이드 / Reading Guide

**EN**:
- **§1-2 (pp. 3-9)**: Introduction and the supergranulation puzzle. Read carefully — sets terminology and scale hierarchy.
- **§3 (pp. 9-28)**: Observational characterization. *The heart of the update.* Focus on §3.2 (scales), §3.4 (rotation), §3.5 (magnetic fields).
- **§4 (pp. 28-38)**: Classical fluid theory. Skim if you know Rayleigh-Bénard; focus on §4.3 (laminar theories of supergranulation) and §4.4 (large-scale instabilities).
- **§5 (pp. 38-54)**: Numerical modelling. This is the other major update. §5.3 (large-scale simulations, esp. 5.3.3 and 5.3.4) is the newest material.
- **§6 (pp. 54-58)**: Discussion and outlook. *Read this first to get the big picture,* then dive back into details.
- **Compare to Paper #19 (2010)**: Note what has changed — especially inclusion of SDO/HMI analyses, new global spherical simulations, and the "near-critical nonlinear convection" framing.

**KR**:
- **§1-2 (pp. 3-9)**: 서론과 초립상 퍼즐. 용어와 규모 계층이 설정되므로 주의 깊게 읽을 것.
- **§3 (pp. 9-28)**: 관측 특성화. *업데이트의 핵심.* §3.2(규모), §3.4(자전), §3.5(자기장)에 집중.
- **§4 (pp. 28-38)**: 고전 유체 이론. Rayleigh-Bénard를 안다면 빠르게 훑고 §4.3(초립상 층류 이론), §4.4(대규모 불안정성)에 집중.
- **§5 (pp. 38-54)**: 수치 모델링. 또 다른 주요 업데이트. §5.3(대규모 시뮬레이션, 특히 5.3.3, 5.3.4)이 가장 새로운 내용.
- **§6 (pp. 54-58)**: 논의와 전망. *전체 그림 파악을 위해 먼저 읽고* 세부 사항으로 돌아올 것.
- **Paper #19(2010)와 비교**: 무엇이 바뀌었는지 주목 — 특히 SDO/HMI 분석 포함, 새로운 전역 구면 시뮬레이션, "임계 근접 비선형 대류" 프레이밍.

---

## 7. 현대적 의의 / Modern Significance

**EN**: Supergranulation sits at the intersection of solar surface convection, the solar dynamo, and stellar convection in general. Understanding it matters for: (1) the small-scale solar dynamo and the formation of the magnetic network; (2) the near-surface shear layer (NSSL) which is dynamically tied to supergranular timescales; (3) stellar convection models that use solar observations for calibration; (4) interpretation of Doppler and intensity signatures in exoplanet host stars. The 2018 update solidifies the buoyancy-driven convective origin view, but the **exact physics that sets the 30-35 Mm scale remains the open question**. DKIST (first light 2020), advanced PSP/Solar Orbiter data, and exascale MHD simulations will be the tools of the next decade. The review's outlook highlights: better helioseismic characterization at supergranulation scales, improved realistic MHD simulations at low magnetic Prandtl number, and convergence between observational techniques.

**KR**: 초립상은 태양 표면 대류, 태양 발전기, 항성 대류의 교차점에 위치합니다. 이를 이해하는 것은 다음에 중요합니다: (1) 소규모 태양 발전기와 자기 네트워크 형성; (2) 초립상 시간 규모와 연결된 근표면 전단층(NSSL); (3) 태양 관측으로 보정하는 항성 대류 모델; (4) 외계행성 모성에서의 Doppler 및 복사 강도 신호 해석. 2018 업데이트는 부력 구동 대류 기원 관점을 공고히 하지만, **30-35 Mm 규모를 결정하는 정확한 물리는 여전히 열린 문제**입니다. DKIST(2020 초광), 고도화된 PSP/Solar Orbiter 데이터, 엑사스케일 MHD 시뮬레이션이 향후 10년의 도구가 될 것입니다. 전망 섹션은 다음을 강조합니다: 초립상 규모에서의 더 나은 일식학적 특성화, 낮은 자기 Prandtl 수에서의 현실적 MHD 시뮬레이션, 관측 기법 간 수렴.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
