---
title: "Pre-Reading Briefing: Solar Stereoscopy and Tomography"
paper_id: "26"
topic: Living_Reviews_in_Solar_Physics
date: 2026-04-23
type: briefing
---

# Solar Stereoscopy and Tomography: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Aschwanden, M. J. (2011), "Solar Stereoscopy and Tomography", *Living Reviews in Solar Physics*, **8**, 5. DOI: 10.12942/lrsp-2011-5
**Author(s)**: Markus J. Aschwanden (Lockheed Martin Solar and Astrophysics Laboratory)
**Year**: 2011

---

## 1. 핵심 기여 / Core Contribution

**English.** This *Living Reviews* article is the first comprehensive synthesis of 3D reconstruction methods for the solar corona. Aschwanden consolidates three decades of disparate techniques — ground-based solar-rotation stereoscopy, space-based single-spacecraft rotation tomography, radio-frequency stereoscopy, and true multi-spacecraft stereoscopy with the STEREO twin satellites (launched 2006) — into a single conceptual and mathematical framework. The review covers both the "geometry side" (epipolar tie-point triangulation, dynamic stereoscopy, magnetic stereoscopy) and the "tomography side" (back-projection, regularized inversion, Kalman filter tomography, instant stereoscopic tomography ISTAR). Twelve categories of coronal phenomena (streamers, active regions, loops, oscillations, filaments, CMEs, flares, global waves, etc.) are surveyed with the 3D quantities they constrain — scale heights, heating rates, magnetic field misalignments, oscillation polarizations, CME masses.

**한국어.** 이 *Living Reviews* 논문은 태양 코로나의 3D 재구성 방법론을 처음으로 포괄적으로 정리한 종합 리뷰이다. Aschwanden은 30년에 걸친 상이한 기법들 — 지상 기반의 태양 자전(solar-rotation) 입체법, 단일 위성 자전 tomography, 전파 주파수 tomography, 그리고 2006년 발사된 쌍둥이 STEREO 위성을 사용한 진정한 다시점 stereoscopy — 을 하나의 개념적·수학적 틀로 통합한다. "기하학 쪽"(에피폴라 타이-포인트 삼각측량, dynamic stereoscopy, magnetic stereoscopy)과 "tomography 쪽"(역투영, 정규화 반전, 칼만 필터 tomography, 활동영역 순간 입체 토모그래피 ISTAR)을 모두 다룬다. 스트리머, 활동영역, 코로나 루프, 진동, 필라멘트, CME, 플레어, 전역 파동 등 12개 부류의 코로나 현상과, 이로부터 얻어진 3D 물리량 — 스케일 높이, 가열률, 자기장 misalignment, 진동 편광, CME 질량 — 을 체계적으로 정리한다.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

**English.** Until STEREO, all solar-dedicated space missions orbited near Earth (Skylab, SMM, Yohkoh, SOHO, Hinode), so true stereoscopic parallax at 1 AU baseline was impossible. For two decades (1985–2006) astronomers used three workarounds: (1) *solar-rotation stereoscopy* — wait ~1 day for the Sun to rotate 13.2° and treat the two views as a stereo pair; (2) *dynamic stereoscopy* — use the magnetic skeleton's stability while plasma flushes through; (3) *multi-frequency radio stereoscopy* at the VLA — use gyroresonance layers at ν = 5, 8, 11 GHz as iso-Gauss altitude indicators. Medical-style tomography was also tried on Skylab white-light and Yohkoh X-ray data, but always under-constrained. The October 2006 launch of STEREO-A and STEREO-B with ~22.5° per year relative drift finally provided true, simultaneous two-viewpoint EUV/white-light imaging, triggering an order-of-magnitude jump in stereoscopic publications.

**한국어.** STEREO 이전에는 모든 태양 전용 우주 임무(Skylab, SMM, Yohkoh, SOHO, Hinode)가 지구 근방 궤도에 있었기 때문에 1 AU 기선(baseline)에서의 진정한 시차(parallax) 관측이 불가능했다. 20년간(1985–2006) 연구자들은 세 가지 우회책을 사용했다: (1) *solar-rotation stereoscopy* — 하루(태양 자전 13.2°) 기다려 같은 구조를 두 번 찍어 스테레오 쌍으로 취급; (2) *dynamic stereoscopy* — 플라즈마는 빠르게 흘러도 자기 골격은 안정하다는 점을 이용; (3) *다주파 전파 stereoscopy* — VLA의 5, 8, 11 GHz에서 자이로공명(gyroresonance) 층을 iso-Gauss 고도 지시자로 사용. Skylab 백색광과 Yohkoh X-선 자료에는 의료용 tomography가 시도됐으나 언제나 under-constrained 문제를 겪었다. 2006년 10월 STEREO-A, STEREO-B가 ~22.5°/년의 상대 표류 속도로 발사되면서 마침내 진짜 동시 2시점 EUV/백색광 영상이 가능해졌고, 관련 논문 수가 한 자릿수 뛰었다.

### 타임라인 / Timeline

```
1979 ─ Altschuler: first 3D density reconstruction from Skylab coronagraph images
      │
1985 ─ Berton & Sakurai: first solar-rotation stereoscopy of XUV loops (Skylab)
      │
1991 ─ Koutchmy & Molodenskij: eclipse-based white-light stereoscopy (Δα ≈ 1.6°)
      │
1992 ─ Davila & Thompson: medical back-projection method demonstrated for Sun
      │
1994 ─ Aschwanden & Bastian: VLA radio stereoscopy (multi-frequency altimetry)
      │
1999 ─ Aschwanden et al.: dynamic stereoscopy using SOHO/EIT magnetic stability
      │
2002 ─ Frazin & Janzen: regularized tomography for LASCO-C2 pB images
      │
2006 ─ STEREO-A & STEREO-B launch (Oct 26) — true twin-view stereoscopy begins
      │
2008 ─ Aschwanden et al.: FIRST 3D stereoscopic reconstruction of 30 EUV loops
      │    (α_sep = 7.3°, active region NOAA 10955)
2009 ─ DeRosa et al.: NLFFF–STEREO misalignment ≈ 20–40° — magnetic modeling crisis
      │
2010 ─ Aschwanden & Sandman: unipolar PFU reduces α_mis to 11–17°
      │
2011 ─ THIS PAPER — Aschwanden's unified review (LRSP 8, 5)
```

---

## 3. 필요한 배경 지식 / Prerequisites

**English.**
- **Geometry / linear algebra.** Epipolar geometry, triangulation from two cameras, rotation matrices, coordinate frame transformations (heliographic ↔ image plane).
- **Solar physics basics.** Differential rotation profile ω(b) = A + B sin²(b) + C sin⁴(b); coronal structures (active regions, streamers, loops, prominences); hydrostatic pressure scale height λ_p = 2k_B T / (μ m_H g).
- **Radiative transfer.** Optically thin free-free (EUV, soft X-ray): F_λ ∝ ∫ n_e² R_λ(T) dz. Thompson scattering in white light. Gyroresonance in microwaves.
- **Differential emission measure (DEM).** dEM(T)/dT and its inversion from multi-filter images.
- **Magnetic field modeling.** Potential field (∇²Φ = 0), linear force-free (∇×B = αB), nonlinear force-free (NLFFF).
- **Tomography fundamentals.** Radon transform, back-projection, ill-posed inverse problems, regularization.

**한국어.**
- **기하학·선형대수.** 에피폴라 기하, 두 카메라 삼각측량, 회전행렬, 좌표계 변환 (helio ↔ 이미지 평면).
- **태양 기본 물리.** 차등 자전 프로파일 ω(b) = A + B sin²(b) + C sin⁴(b); 코로나 구조(활동영역, 스트리머, 루프, 프로미넌스); 정유체(hydrostatic) 스케일 높이 λ_p = 2k_B T / (μ m_H g).
- **복사 전달.** 광학적으로 얇은 free-free (EUV, 연X-ray): F_λ ∝ ∫ n_e² R_λ(T) dz. 백색광의 Thompson 산란. 마이크로파의 gyroresonance 방출.
- **DEM.** dEM(T)/dT와 다중 필터 역산.
- **자기장 모형.** Potential field, LFF (∇×B = αB), NLFFF.
- **Tomography 기초.** Radon 변환, 역투영(back-projection), ill-posed inverse problem, 정규화.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **Stereoscopy / 입체법** | 두 시점에서 같은 점을 찍어 삼각측량으로 3D 좌표를 얻는 방법. Two-viewpoint geometric triangulation giving (x, y, z). |
| **Tomography / 단층촬영** | 많은 시점에서 얻은 2D 투영들을 역변환하여 3D 밀도장을 복원. Many-projection inversion for the full volumetric n_e(x, y, z). |
| **Epipolar plane / 에피폴라 평면** | 두 관측자와 소스 점을 포함하는 평면. 이 평면을 따라 1D 탐색으로 대응점을 찾을 수 있음. Plane containing two observers and source — reduces correspondence search to 1D. |
| **Tie-point method / 타이-포인트 법** | 이미지 B의 점 [X_B, Y_B]를 이미지 A의 점 [X_A, Y_A]에 "묶어" 대응시키고 삼각측량하는 STEREO 표준 기법. The "tying" of corresponding features across images for triangulation. |
| **Solar-rotation stereoscopy** | 태양 자전(~13.2°/day)을 시점 변화로 활용하는 단일 관측소 3D 복원법. Uses ~13.2°/day solar rotation to synthesize stereo from one observatory. |
| **Dynamic stereoscopy** | 자기장의 수일 단위 안정성을 가정하고, 빠르게 흐르는 플라즈마가 교체되어도 자기 골격은 동일하다는 전제로 수행. Exploits magnetic-field stability while plasma is flushed through. |
| **Magnetic stereoscopy** | 이론적 자기장 모델(potential, LFF, NLFFF)을 자료와 비교·피팅하여 3D 구조를 제약. Uses magnetic-field models to resolve stereoscopic correspondence ambiguities. |
| **Misalignment angle α_mis** | 관측된(삼각측량된) 루프 방향과 이론적 자기장선 방향의 3D 각 차이. 0° = 완벽. 실제 NLFFF는 20–40°. 3D angular difference between observed and theoretical field lines. |
| **Thompson scattering / 톰슨 산란** | 백색광 코로나그래프 밝기의 원인. σ_T = 6.65×10⁻²⁵ cm². 밀도에 선형 의존. Cause of white-light corona brightness — linear in n_e. |
| **DEM — Differential Emission Measure** | 온도별 방출측도 분포. dEM/dT = n_e²(dz/dT). EUV/X-ray 반전 변수. Volume emission measure per unit temperature — the EUV/X-ray inversion target. |
| **Pressure scale height λ_p** | 정유체 평형에서의 e-folding 높이. λ_p ≈ 4.7×10⁹ (T_e/MK) cm ≈ 47 (T/MK) Mm. Hydrostatic e-folding height, ≈ 47(T/MK) Mm. |
| **ISTAR** | Instant Stereoscopic Tomography of Active Region — 70개 삼각측량 루프를 뼈대로 8000개 flux tube로 활동영역 전체 DEM을 재구성. Active region tomography built from ~70 triangulated skeleton loops + ~8000 interpolated flux tubes. |

---

## 5. 수식 미리보기 / Equations Preview

**Eq. 1 — Solar-rotation parallax displacement**:
$$\Delta x_{12} = (R_\odot + h) \cos(b_1)[\sin(l_1 + \omega_{syn}(t_2-t_1)) - \sin(l_1)]$$
- One-day rotation (ω_syn = 2π/26.24 day) maps altitude h to east-west displacement Δx on the image.
- 하루의 자전이 고도 h를 이미지 동서 변위로 매핑. 단일 관측소에서 h를 삼각측량할 수 있게 하는 핵심 식.

**Eq. 19–20 — Tie-point triangulation for STEREO A+B**:
$$x = \frac{x_B \tan\gamma_B - x_A \tan\gamma_A}{\tan\gamma_B - \tan\gamma_A}, \qquad z = (x_A - x)\tan\gamma_A$$
- Given observed pixel angles (α_A, δ_A, α_B, δ_B) from two spacecraft with separation α_sep, solve for 3D (x, y, z).
- 두 위성에서 관측한 픽셀 각도로부터 3D 좌표 계산. STEREO 트라이앵귤레이션의 핵심.

**Eq. 9 — Thompson cross-section**:
$$\sigma_T = \frac{8\pi}{3} r_e^2 = 6.65 \times 10^{-25}\,\text{cm}^2$$
- Governs white-light coronagraph brightness — linear in n_e (contrast with n_e² for EUV).
- 백색광 코로나그래프의 밝기 결정. 밀도에 선형(EUV의 n_e²와 대조).

**Eq. 27 — Hydrostatic pressure scale height**:
$$\lambda_p(T_e) = \frac{2 k_B T_e}{\mu m_H g_\odot} \approx 4.7 \times 10^9 \left(\frac{T_e}{1\,\text{MK}}\right)\,\text{cm}$$
- Diagnostic observable when 3D geometry is known. Observed λ_p^obs = λ_p / cos(θ) for inclined loops.
- 3D 기하를 알면 직접 관측 가능. 기울어진 루프는 cos(θ)만큼 확대되어 보임.

**Eq. 39 — Misalignment angle**:
$$\alpha_{mis} = \arccos\left(\frac{\mathbf{r}_{obs}\cdot\mathbf{r}_{pot}}{|\mathbf{r}_{obs}|\,|\mathbf{r}_{pot}|}\right)$$
- Benchmarks magnetic-field models against stereo-triangulated loops. PFSS: 25°±8°; NLFFF: 24–44°; optimized PFU: 11–17°.
- 자기장 모델의 3D 검증 척도. PFSS는 25°±8°, NLFFF는 24–44°, 최적화 PFU는 11–17°.

---

## 6. 읽기 가이드 / Reading Guide

**English.**
1. **Skim the abstract + Section 1.** Note the taxonomy: stereoscopy vs. tomography, and the helioseismic/interplanetary exclusions.
2. **Section 2 (History)** — Map the 5 historical eras: eclipses → Skylab → radio → dynamic → STEREO.
3. **Section 3 (Methods)** is the mathematical core. Read 3.1 → 3.3 → 3.4 → 3.2 → 3.5. **Memorize Figures 8 & 9 (epipolar geometry) and Eqs. 15–23.** The tie-point equations alone are sufficient to write a STEREO triangulation code.
4. **Section 4 (Observations)** — Use the subsection headers as a reference menu. For a first read, prioritize: 4.1 (large-scale corona), 4.4 (loops — hydrostatics, hydrodynamics, B-fields), 4.5 (MHD oscillations with the June 27 2007 case study).
5. **Section 5 (Summary)** is an 11-item executive digest — read twice, first to set expectations before Section 4, again after reading.

**한국어.**
1. **초록 + Section 1 훑어보기.** Stereoscopy vs. tomography 분류, helioseismic/행성간 CME은 제외 범위.
2. **Section 2 (역사)** — 5개 시대 구분: 개기일식 → Skylab → 전파 → dynamic → STEREO.
3. **Section 3 (방법)**이 수학의 핵심. 3.1 → 3.3 → 3.4 → 3.2 → 3.5 순서 추천. **Figures 8, 9 (에피폴라 기하)와 Eq. 15–23을 반드시 외울 것.** 타이-포인트 식들만으로 STEREO 삼각측량 코드를 짤 수 있음.
4. **Section 4 (관측)** — 소절 헤더를 메뉴처럼 활용. 첫 독서는 4.1(대규모 코로나), 4.4(루프), 4.5(MHD 진동, 2007 Jun 27 사례)를 우선.
5. **Section 5 (요약)**는 11개 항목의 executive digest — Section 4 들어가기 전과 읽은 후 두 번 읽으면 이해 배가.

---

## 7. 현대적 의의 / Modern Significance

**English.** This review has become *the* standard reference for 3D coronal reconstruction. It directly enables:
- **SDO+STEREO synergy.** AIA's high-cadence, high-resolution EUV combined with STEREO's off-axis views defines the "triple-viewpoint era" Aschwanden anticipates in Section 2.3.
- **Space-weather operations.** Quasi-real-time CME 3D reconstruction from SECCHI COR-1/2 and HI-1/2 uses tie-point triangulation operationally at NOAA/SWPC.
- **ML / computer-vision bridge.** The loop segmentation (OCCULT code) and tie-point matching problem described here maps directly onto modern deep-learning image-correspondence tasks (e.g., cross-modal feature matching, stereo CNNs).
- **Magnetic-field model validation.** The α_mis metric is now the gold standard for benchmarking NLFFF, MHD-equilibrium, and data-driven coronal B-field codes.
- **Next-decade missions.** Solar Orbiter (launched 2020), PUNCH (2025), and the proposed L5 Observatory all rely on the geometric framework codified here.

**한국어.** 이 리뷰는 3D 코로나 재구성의 표준 참고문헌이 되었다:
- **SDO+STEREO 시너지.** AIA의 고해상도·고시간분해능 EUV + STEREO의 측면 관측이 Aschwanden이 Section 2.3에서 예견한 "세 시점 시대"를 현실화.
- **우주 기상 운영.** SECCHI COR-1/2·HI-1/2의 CME 3D 재구성이 NOAA/SWPC에서 운영적으로 사용 중. 타이-포인트 법이 실시간 예보 파이프라인에 포함됨.
- **ML / 컴퓨터 비전 연결.** 루프 분할(OCCULT)과 타이-포인트 대응 문제는 현대 딥러닝의 이미지 대응(stereo CNN, cross-modal matching) 과제로 직접 매핑.
- **자기장 모델 검증.** α_mis 척도가 NLFFF, MHD, 데이터 기반 코로나 B 모형을 비교하는 사실상의 표준.
- **차세대 임무.** Solar Orbiter(2020 발사), PUNCH(2025), 제안 중인 L5 관측소 — 모두 본 리뷰의 기하학적 틀에 기반.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
