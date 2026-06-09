---
title: "Solar Stereoscopy and Tomography"
authors: Markus J. Aschwanden
year: 2011
journal: "Living Reviews in Solar Physics"
doi: "10.12942/lrsp-2011-5"
topic: Living_Reviews_in_Solar_Physics
tags: [stereoscopy, tomography, STEREO, 3D reconstruction, coronal loops, tie-point method, magnetic stereoscopy, ISTAR, EUV, white-light]
status: completed
date_started: 2026-04-23
date_completed: 2026-04-23
---

# 26. Solar Stereoscopy and Tomography / 태양 입체법과 단층촬영

---

## 1. Core Contribution / 핵심 기여

**English.** Aschwanden's 2011 *Living Reviews* article is the first unified, book-length synthesis of three-dimensional reconstruction methods for the solar corona. Before the STEREO mission (launched October 2006) every solar-dedicated imager orbited near Earth, so the astronomical concept of stereoscopic parallax could not be applied directly. Workers invented surrogate strategies: solar-rotation stereoscopy (Berton & Sakurai 1985), multi-frequency radio stereoscopy (Aschwanden & Bastian 1994), tomographic inversion of single-spacecraft coronagraph sequences (Altschuler 1979; Frazin 2000), and dynamic stereoscopy of magnetically-stable skeletons (Aschwanden et al. 1999). This review consolidates all of these, introduces the full epipolar/tie-point triangulation mathematics that became possible with true twin-view STEREO A+B imaging, and surveys every class of coronal phenomenon to which these geometries have been applied. It spans ~80 pages, ~40 figures, and catalogues the quantitative 3D results — hydrostatic scale heights, over-pressure factors, magnetic misalignment angles, oscillation polarizations, CME masses — that stereoscopy has delivered.

**한국어.** Aschwanden의 2011년 *Living Reviews* 논문은 태양 코로나 3차원 재구성 방법론을 처음으로 책 길이 규모로 통일된 시각에서 종합한 리뷰이다. 2006년 10월 STEREO 발사 이전까지 모든 태양 전용 관측 위성이 지구 근방에 있었기 때문에 천문학의 고전적 시차 개념을 직접 적용할 수 없었다. 연구자들은 우회적 전략 — 태양 자전 입체법(Berton & Sakurai 1985), 다주파 전파 입체법(Aschwanden & Bastian 1994), 단일 위성 코로나그래프 시계열의 토모그래피 반전(Altschuler 1979; Frazin 2000), 자기 골격이 안정하다는 전제의 dynamic stereoscopy(Aschwanden et al. 1999) — 을 개발해왔다. 본 리뷰는 이러한 방법들을 통합하며, STEREO A+B의 진정한 쌍 시점이 가능하게 한 에피폴라/타이-포인트 삼각측량 수학을 완전히 제시하고, 이 기하학이 적용된 모든 코로나 현상 부류를 검토한다. 약 80쪽, 40여 개의 그림, 그리고 입체법이 제공한 정량적 3D 결과들 — 정유체 스케일 높이, 과잉 압력 인자, 자기 misalignment 각, 진동 편광, CME 질량 — 이 집약되어 있다.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction and History / 서론과 역사 (pp. 5–8)

**English.** Section 1 distinguishes two related 3D reconstruction paradigms. *Stereoscopy* (Greek *stereos* = solid body, *skopein* = to see) uses two (or a few) aspect angles, producing triangulated 3D coordinates of point-like or curvi-linear features. *Tomography* (Greek *tomos* = slice, *graphein* = to write) uses many aspect angles to invert a full volumetric density field n_e(x,y,z). The key constraint for the Sun is aspect angle: solar rotation provides ≈ 13.2° per day, but most coronal plasma changes on much shorter timescales. STEREO provided the first "true" stereoscopy by sending twin spacecraft on 1-AU heliospheric orbits, drifting ~22.5° per year apart. The review explicitly excludes helioseismic interior tomography and interplanetary CME reconstruction (beyond ~45 R☉).

Section 2 sketches a five-era history: (i) ground-based eclipse stereoscopy (Koutchmy & Molodenskij 1992 — parallax ≈ 1.6°); (ii) single-spacecraft near-Earth missions (Skylab 1973, SMM, Yohkoh, SOHO) using solar rotation; (iii) multi-frequency VLA radio stereoscopy where iso-Gauss gyroresonance layers at ν = 5, 8, 11 GHz behave as altitude indicators; (iv) dynamic stereoscopy with SOHO/EIT exploiting magnetic-field stability; (v) the STEREO era (2006–). Figure 1 shows the twin orbits; Figure 2 illustrates rotation-based VLA stereoscopy. The STEREO spacecraft carry four instrument packages: SECCHI (EUVI 171/195/284/304 Å, COR-1 for 1.4–4 R☉, COR-2 for 2–15 R☉, HI-1/2 for 8–200 R☉), IMPACT, PLASTIC, and SWAVES.

**한국어.** Section 1은 두 가지 3D 재구성 패러다임을 구분한다. *Stereoscopy*(그리스어 stereos = 입체, skopein = 보다)는 두(또는 몇) 개의 관측 각도를 사용해 점 또는 곡선 구조의 3D 좌표를 삼각측량한다. *Tomography*(tomos = 자르다, graphein = 쓰다)는 많은 각도의 투영을 반전시켜 전체 3D 밀도장 n_e(x,y,z)를 복원한다. 태양의 핵심 제약은 관측 각도: 태양 자전은 약 13.2°/일을 제공하지만, 대부분의 코로나 플라즈마는 훨씬 짧은 시간척도에서 변한다. STEREO는 쌍둥이 위성을 1 AU 궤도에 배치하여 연간 ~22.5°씩 멀어지게 함으로써 최초의 "진짜" 입체법을 가능케 했다. 본 리뷰는 헬리오사이즈믹 내부 토모그래피와 ~45 R☉ 너머의 행성간 CME 재구성은 다루지 않는다.

Section 2는 5개 시대로 역사를 요약한다: (i) 지상 개기일식 입체법 (Koutchmy & Molodenskij 1992, parallax ≈ 1.6°); (ii) 지구 근방 단일 위성 임무 (Skylab 1973, SMM, Yohkoh, SOHO) — 태양 자전 활용; (iii) VLA 다주파 전파 입체법 — 5, 8, 11 GHz의 iso-Gauss 자이로공명 층이 고도 지시자 역할; (iv) SOHO/EIT dynamic stereoscopy — 자기장 안정성 활용; (v) STEREO 시대(2006–). Figure 1은 쌍둥이 궤도, Figure 2는 VLA 자전 입체법의 원리. STEREO 위성은 4개 장비 패키지 탑재: SECCHI (EUVI 171/195/284/304 Å, COR-1 for 1.4–4 R☉, COR-2 for 2–15 R☉, HI-1/2 for 8–200 R☉), IMPACT, PLASTIC, SWAVES.

### Part II: Methods — Solar-Rotation and Tie-Point / 방법론: 자전과 타이-포인트 (pp. 9–22)

**English.** Section 3.1 — Static solar-rotation stereoscopy: For a source at heliographic longitude l₁ and latitude b₁, one-day rotation produces an east-west displacement

Δx₁₂ = (R☉ + h) cos(b₁)[sin(l₁ + ω_syn·Δt) − sin(l₁)]

with ω_syn = 2π/T_syn = 2π/(26.24 d). Measuring Δx at two times with known (l₁, b₁) yields a unique altitude h. The differential rotation rate has latitudinal dependence ω_sid(b) = A + B sin²(b) + C sin⁴(b) with A = 14.71, B = −2.4, C = −1.8 deg/day. A practical lower-bound on observing interval (Eq. 3) is t ≈ (T_syn/2π) · (Δx/Δh), so for 1" spatial resolution and Δh = 3000 km accuracy one needs ~1 day — longer than most coronal dynamics. Berton & Sakurai (1985) applied this to Skylab XUV, measuring a loop at h ≈ 0.15 R☉ = 100,000 km with θ ≈ 25° inclination.

Section 3.1.2 — Dynamic stereoscopy (Aschwanden et al. 1999, 2000): Although plasma churns over hours, the magnetic field is quasi-stationary over days. Assuming loop planarity, only a single free parameter — the inclination angle θ of the loop plane — needs to be fitted, reducing the ill-posed problem to a 1D optimization. Figure 3 illustrates: observed EIT loop tracings at three consecutive days are matched by varying θ = 10°, 20°, 30°, 90°.

Section 3.2 — Solar-rotation tomography: Section 3.2.1 treats optically-thin EUV/X-ray emission: EM ∝ ∫ n_e²(z, T) dz, and F_λ = ∫ (dEM/dT) R_λ(T) dT. The tomographic inverse problem (Eq. 6) becomes F_λ(X_i, Y_j, θ_k) = Σ_l A_λ n_e²(X_i, Y_j, Z_l) + σ_λ, which is under-constrained (N²·N_k observations vs N³ voxels). A rotation (Eq. 7) converts image coordinates (X_i, Y_j) to heliographic (x_i, y_j, z_k) via angle θ_k. Practical inversion uses back-projection, regularized positive-estimation (Frazin 2000), or Kalman filtering (Frazin et al. 2005a). Figure 4 shows the classic medical back-projection principle (Davila & Thompson 1992).

Section 3.2.2 — White light: Thompson scattering cross-section σ_T = (8π/3) r_e² = 6.65×10⁻²⁵ cm² with differential cross-section dσ/dω = ½ r_e² (1 + cos²χ), where χ is the scattering angle between line-of-sight and radial direction (Eq. 8). The polarization ratio p = (I_T − I_R)/(I_T + I_R) = I_P/I_tot (Eq. 10). Modern coronagraphs (LASCO, STEREO/COR) measure linear polarization in three orientations → separate total brightness B and polarized brightness pB.

Section 3.2.3 — Radio tomography uses the free-free absorption coefficient α_ff ∝ (n_e n_i / ν²T^{3/2}) ln Λ (Eq. 11) and opacity τ_ff = ∫ α_ff dz' (Eq. 12). Altitudes where radiation becomes optically thick (τ_ff ~1) serve as iso-T layers; combining ν = 5, 8, 11 GHz gives altitude sampling h ≈ 3000–12,000 km.

Section 3.3 — Tie-point triangulation is *the* workhorse for STEREO. Figure 8 shows the epipolar geometry: the two spacecraft and Sun center define a family of epipolar planes, and any 3D source projects to corresponding lines in each image — reducing correspondence search from 2D to 1D. The process has three steps: (1) rectify image pair into an epipolar coordinate system [X, Y]; (2) identify corresponding features [X_A(s), Y_A(s)] and [X_B(s), Y_B(s)] along a curvi-linear feature; (3) triangulate. In an (X, Y, Z) frame centered at the Sun with Z-axis along STEREO-A line-of-sight, the triangles O-A-x_A and O-B-x_B (Figure 9) yield:

γ_A = π/2 − α_A,  γ_B = π/2 − α_B − α_sep
x_A = d_A tan(α_A),  x_B = d_B sin(α_B)/sin(γ_B)
x = (x_B tan γ_B − x_A tan γ_A) / (tan γ_B − tan γ_A)
z = (x_A − x) tan γ_A
y = (d_A − z) tan δ_A

This is called the tie-point method because every point [X_B(s_i), Y_B(s_i)] is "tied" to the corresponding [X_A(s_i), Y_A(s_i)]. First STEREO application: Aschwanden et al. (2008c) triangulated 30 coronal loops in AR NOAA 10955 at α_sep = 7.3° on 2007 May 9.

Section 3.4 — Magnetic stereoscopy: Uses an external 3D magnetic model B(r) to resolve the correspondence ambiguity that arises when many loops overlap. Wiegelmann & Neukirch (2002) varied the α-parameter in linear force-free extrapolation to optimize the projected match. DeRosa et al. (2009) compared 11 NLFFF models + 1 potential field with STEREO-triangulated loops and found misalignment α_mis ≈ 24°–44°. Aschwanden & Sandman (2010) introduced *unipolar magnetic charges* B_j(r) = Σ B_j (z_j/r_j)² r̂_j (Eq. 40), improving α_mis to 11°–17°. Dipole variants (Sandman & Aschwanden 2011) perform similarly (Eq. 41).

**한국어.** Section 3.1 — 정적 태양 자전 입체법: 태양중심 경위도 (l₁, b₁)의 광원에 대해 하루의 자전은 동서 방향 변위

Δx₁₂ = (R☉ + h) cos(b₁)[sin(l₁ + ω_syn·Δt) − sin(l₁)]

을 만들며, ω_syn = 2π/T_syn = 2π/(26.24 일). 두 시각의 Δx로부터 고도 h를 유일하게 결정. 차등 자전은 ω_sid(b) = A + B sin²(b) + C sin⁴(b), A = 14.71, B = −2.4, C = −1.8 deg/day. 관측 간격 하한(Eq. 3) t ≈ (T_syn/2π)(Δx/Δh)이라, 1″ 분해능·Δh = 3000 km 정확도에서는 ~1일 필요 — 대부분 코로나 동역학보다 길다. Berton & Sakurai (1985)는 Skylab XUV에 적용해 h ≈ 0.15 R☉ = 100,000 km, θ ≈ 25°의 루프를 측정.

Section 3.1.2 — Dynamic stereoscopy (Aschwanden et al. 1999, 2000): 플라즈마는 수 시간 내에 교체되지만 자기장은 며칠간 quasi-stationary. 루프가 평면(planar)이라고 가정하면 자유 매개변수는 오직 루프 평면의 기울기 각 θ 하나만 남아 ill-posed 문제가 1D 최적화로 축소. Figure 3: 3일 연속 EIT 루프 자취를 θ = 10°, 20°, 30°, 90°로 맞춰 최적 θ를 찾음.

Section 3.2 — Solar-rotation tomography: 3.2.1에서 광학적으로 얇은 EUV/X-ray 방출 처리. EM ∝ ∫ n_e²(z, T) dz, F_λ = ∫ (dEM/dT) R_λ(T) dT. Tomographic 역문제(Eq. 6): F_λ(X_i, Y_j, θ_k) = Σ_l A_λ n_e²(X_i, Y_j, Z_l) + σ_λ 는 under-constrained(N²·N_k 관측 vs N³ voxel). 회전(Eq. 7)이 이미지 좌표를 helio 좌표로 매핑. 실제 반전은 back-projection, 정규화 positive-estimation (Frazin 2000), 칼만 필터링 (Frazin et al. 2005a). Figure 4는 의료용 back-projection의 원리.

Section 3.2.2 — 백색광: Thompson 단면적 σ_T = (8π/3) r_e² = 6.65×10⁻²⁵ cm², 미분 단면적 dσ/dω = ½ r_e² (1 + cos²χ), χ는 시선과 반경방향 사이 산란각(Eq. 8). 편광도 p = (I_T − I_R)/(I_T + I_R) = I_P/I_tot (Eq. 10). 현대 코로나그래프(LASCO, STEREO/COR)는 3방향 편광 측정 → 총밝기 B와 편광밝기 pB 분리.

Section 3.2.3 — 전파 tomography: free-free 흡수계수 α_ff ∝ (n_e n_i / ν²T^{3/2}) ln Λ (Eq. 11), opacity τ_ff = ∫ α_ff dz' (Eq. 12). 광학적 두께(τ_ff ~1) 고도가 iso-T 층으로 기능. ν = 5, 8, 11 GHz 조합 시 h ≈ 3000–12,000 km 샘플링.

Section 3.3 — 타이-포인트 삼각측량은 STEREO의 주력 방법. Figure 8은 에피폴라 기하: 두 위성과 태양 중심이 에피폴라 평면 가족을 정의하고, 모든 3D 광원은 각 이미지의 대응 에피폴라 선으로 투영되어 대응점 탐색이 2D → 1D로 축소. 세 단계: (1) 이미지 쌍을 에피폴라 좌표계 [X, Y]로 정렬; (2) 곡선 구조의 대응점 [X_A(s), Y_A(s)], [X_B(s), Y_B(s)] 식별; (3) 삼각측량. 태양 중심 (X, Y, Z) 좌표계에서 Z축이 STEREO-A 시선 방향일 때, 삼각형 O-A-x_A와 O-B-x_B(Figure 9)로부터:

γ_A = π/2 − α_A,  γ_B = π/2 − α_B − α_sep
x_A = d_A tan(α_A),  x_B = d_B sin(α_B)/sin(γ_B)
x = (x_B tan γ_B − x_A tan γ_A) / (tan γ_B − tan γ_A)
z = (x_A − x) tan γ_A
y = (d_A − z) tan δ_A

이미지 B의 각 점 [X_B(s_i), Y_B(s_i)]가 A의 대응점에 "묶인다(tied)"는 의미에서 tie-point method. 첫 STEREO 적용: Aschwanden et al. (2008c), 2007 May 9, NOAA 10955, α_sep = 7.3°, 30개 루프 삼각측량.

Section 3.4 — Magnetic stereoscopy: 외부 3D 자기 모델 B(r)를 사용해 중첩 루프의 대응점 모호성을 해결. Wiegelmann & Neukirch (2002)는 linear force-free의 α를 최적화해 투영 루프와 맞춤. DeRosa et al. (2009)는 11개 NLFFF + 1개 potential field를 STEREO 삼각측량 루프와 비교, α_mis ≈ 24°–44° 발견. Aschwanden & Sandman (2010)은 *unipolar magnetic charges* B_j(r) = Σ B_j (z_j/r_j)² r̂_j (Eq. 40)로 α_mis를 11°–17°로 개선. Dipole 변형 (Sandman & Aschwanden 2011)도 유사 성능 (Eq. 41).

### Part III: Section 3.5 ISTAR and Section 4 Observations / ISTAR와 관측 응용 (pp. 25–73)

**English.** Section 3.5 — Stereoscopic tomography / 3D forward-fitting: Since only 2–3 STEREO-era spacecraft exist, classical tomography is severely under-constrained. *Instant Stereoscopic Tomography of Active Region (ISTAR)* solves this by using the magnetic skeleton as a prior. ~70 stereoscopically-triangulated loops are extended via magnetic-field extrapolation to populate ~8000 flux tubes. Each flux tube is assigned hydrodynamic density n_e(s) and temperature T_e(s) solutions, integrated along the line-of-sight with the filter response R_λ(T), and forward-fitted to three EUVI temperature filters. This yields a space-filling 3D density/temperature model of the active region. Figure 14 shows the rendered flux-tube volume.

Section 4.1 (Large-scale corona): Tomographic inversion from STEREO COR-1 and LASCO gives n_e(l, b, r) at r ≈ 1.0–2.5 R☉. Synoptic maps at r = 2.55 R☉ (LASCO, 87 pB images, Frazin et al. 2007) serve as reference levels for PFSS extrapolations. Frazin et al. (2009b) combined DEM with rotation tomography on STEREO data, producing T_e(l,b,h) at r = 1.075 R☉ for T ≈ 0.5–2.5 MK (Figure 15). Butala et al. (2010) demonstrated dynamic (Kalman-filter) tomography at r = 1.3–4 R☉ over 4 weeks (Figure 16). Vásquez et al. (2010, 2011) added iso-Gauss magnetic overlays (Figures 17–18).

Section 4.2 (Streamers): Coronal streamers show up as the longest-lived tomographic features. Saez et al. (2005) discovered *double plasma sheets* and *triple current sheets* in LASCO C-2 tomography not reproduced by standard PFSS (Figure 19). Streamer-blob tracking (Sheeley Jr. et al. 2009) and streamer-blowout CMEs (Lynch et al. 2010) were tracked with stereoscopic triangulation.

Section 4.3 (Active regions): Stereoscopic triangulation of 66 VLA radio sources in 22 active regions gave h = 25 ± 15 Mm and a center-limb darkening T_B(α) = T_B(0)[0.4 + 0.6 cos²(α)] (Eq. 24) interpreted as opacity from dense cool coronal plasma near the limb. AR NOAA 7986 (SOHO/EIT 171, 195, 284 Å, Aschwanden et al. 2000) — first evidence that EUV loops are dominated by radiative cooling, violating steady-state RTV. AR NOAA 10955 (2007 May 9, STEREO) — 70 loops triangulated across 3 filters, ISTAR reconstructed 8000 flux tubes. DEM extends to log T = 5.0–7.0 (Figure 21).

Section 4.4 (Coronal loops):
- 4.4.1 Hydrostatic scale height: Loop parameterized by center (x_c, y_c, z_c), curvature radius r_c, azimuth α, inclination θ. For semi-circular loop h(s) = (2L/π) sin(πs/2L) (Eq. 25). Pressure p(s) ≈ p₀ exp[−(h(s)−h₀)/λ_p(T_e)] (Eq. 26). **Inclined loops show an *observed* scale height λ_p^obs = λ_p/cos(θ)** (Eq. 28) — "communicating water tubes" analogy (Figure 24). Aschwanden et al. (2009c): pressure scale heights λ_n vs. T_m show **super-hydrostatic heights for T_m ≲ 3 MK, consistent with hydrostatic for hotter soft X-ray loops** (Figure 22).
- 4.4.2 Hydrodynamics: Column depth w_z[x_i, y_i, z_i] = w / cos(ψ[x_i, y_i, z_i]) (Eq. 30), cos(ψ) from 3D loop coordinates (Eq. 31). STEREO A-B self-consistency: T_B/T_A = 1.05 ± 0.09, n_B/n_A = 0.94 ± 0.12, w_B/w_A = 0.96 ± 0.05 (Figure 27). Over-pressure q_p = p_obs/p_RTV = 7.57×10⁻⁷ n_e L / T_e² (Eq. 32) yields **q ≈ 3–15 for EUV loops** (Figure 28) — dynamic non-equilibrium heating, not RTV.
- 4.4.3 Magnetic fields: Potential B = ∇Φ with ∇²Φ = 0 (Eqs. 33–34). LFF (∇×B = αB, Eq. 35) and NLFFF (∇×B = α(r)B, Eq. 36). Fit criterion min[Δ₂(α)] = (1/s²_max) ∫ √|r_obs − r_ff(s, α)|² ds (Eq. 37). Feng et al. (2007a) found |α| ≈ (2–8)×10⁻³ Mm⁻¹ for 5 STEREO loops → twist Φ = 2πn < 0.5, below kink instability threshold Φ ≤ 3.5π. DeRosa et al. (2009) showed NLFFF misalignment α_mis = 24°–44° vs. STEREO loops — magnetic modeling is *not* better than potential field in some cases. Aschwanden & Sandman (2010) unipolar PFU: α_mis = 14.3° ± 11.5°, 13.3° ± 9.3°, 20.3° ± 16.5°, 15.2° ± 12.3° for 4 active regions (Table 1). Non-potentiality α_NP = √(α²_PFU − Δα²_SE) = 11°–17°, correlated with soft X-ray flux (Figure 29).

Section 4.5 (MHD oscillations): 2007 Jun 27, 17:30 UT flare with loop oscillation observed by both STEREO EUVI/A and EUVI/B at α_sep = 8.26°. **Amplitude a(t) = a₀ + a₁ cos(2π(t−t₀)/P + Φ) exp(−(t−t₀)/τ_d)** (Eq. 42). **Fitted values**: a₁ = 2.5 EUVI pixels = 2900 km, P = 565 s ≈ 9 min, τ_d = 1600 s ≈ 27 min, cadence 150 s. 3D triangulation of 12 image pairs revealed an **S-shaped, non-planar, asymmetric loop** (not semi-circular, contrary to standard assumptions). Circular polarization hints at torsional/helical kink mode.

Section 4.6 (MHD waves): Slow-mode MHD waves propagate at sound speed c_s proportional to T_e; stereoscopic triangulation of loops containing waves yields absolute phase speeds, correcting projected values (de-projection).

Sections 4.7–4.12 (Filaments, prominences, jets, plumes, flares, CME source regions, global waves): Each application class is cataloged with characteristic 3D quantities obtained. Highlights: (4.8) erupting filament 3D trajectories reveal asymmetric destabilization timing; (4.10) flares show low-lying confined vs. high-lying eruptive post-flare loops; (4.11) CME 4D modeling of EUV dimming constrains expansion geometry (bubble, ice-cone, flux rope); (4.12) global coronal waves propagate as magneto-acoustic fronts with height and speed constrained by stereoscopic geometry.

**한국어.** Section 3.5 — Stereoscopic tomography / 3D forward-fitting: STEREO 시대에 2–3기 위성뿐이므로 고전적 tomography는 극도로 under-constrained. *Instant Stereoscopic Tomography of Active Region (ISTAR)*는 자기 골격을 prior로 사용해 이를 해결. 삼각측량된 ~70개 루프를 자기장 외삽으로 확장하여 ~8000개 flux tube 채움. 각 flux tube에 유체역학적 n_e(s), T_e(s) 해를 부여, 시선 적분하고 필터 응답 R_λ(T)를 곱해 3개 EUVI 온도 필터에 forward-fit. 결과는 활동영역 공간 채움 3D 밀도/온도 모형. Figure 14는 flux tube 볼륨 렌더링.

Section 4.1 (대규모 코로나): STEREO COR-1·LASCO tomography가 r ≈ 1.0–2.5 R☉의 n_e(l, b, r)을 제공. r = 2.55 R☉의 시놉틱 맵(LASCO 87장 pB, Frazin et al. 2007)이 PFSS 외삽의 참조. Frazin et al. (2009b)은 DEM+자전 tomography로 r = 1.075 R☉, T ≈ 0.5–2.5 MK 온도 맵 생성 (Figure 15). Butala et al. (2010)은 4주에 걸친 r = 1.3–4 R☉ dynamic(Kalman) tomography 시연 (Figure 16). Vásquez et al. (2010, 2011)은 iso-Gauss 자기장 오버레이 추가 (Figures 17–18).

Section 4.2 (스트리머): 코로나 스트리머는 가장 오래 지속되는 토모그래피 구조. Saez et al. (2005)은 PFSS로 재현되지 않는 *이중 플라즈마 시트*, *삼중 전류 시트*를 LASCO C-2로 발견 (Figure 19). Sheeley Jr. et al. (2009)의 스트리머 블롭 추적, Lynch et al. (2010)의 스트리머 blowout CME 추적이 삼각측량 기반.

Section 4.3 (활동영역): 22개 활동영역의 66개 VLA 전파 성분을 삼각측량해 h = 25 ± 15 Mm, center-limb darkening T_B(α) = T_B(0)[0.4 + 0.6 cos²(α)] (Eq. 24) — 림 근방 조밀 냉 플라즈마의 opacity 효과로 해석. AR NOAA 7986 (SOHO/EIT, Aschwanden et al. 2000) — EUV 루프가 복사 냉각 지배라는 최초 증거, 정상상태 RTV 위배. AR NOAA 10955 (2007 May 9, STEREO) — 3필터에서 70 루프 삼각측량, ISTAR로 8000 flux tube 재구성. DEM은 log T = 5.0–7.0 확장 (Figure 21).

Section 4.4 (코로나 루프):
- 4.4.1 정유체 스케일 높이: 루프 매개변수화 — 중심 (x_c, y_c, z_c), 곡률반경 r_c, 방위각 α, 경사 θ. 반원형 루프 h(s) = (2L/π) sin(πs/2L) (Eq. 25). 압력 p(s) ≈ p₀ exp[−(h(s)−h₀)/λ_p(T_e)] (Eq. 26). **기울어진 루프는 *관측* 스케일 높이 λ_p^obs = λ_p/cos(θ)** (Eq. 28) — "연결된 물관(communicating water tubes)" 비유 (Figure 24). Aschwanden et al. (2009c)는 T_m ≲ 3 MK 루프가 super-hydrostatic, soft X-ray의 더 뜨거운 루프는 hydrostatic 근사 (Figure 22).
- 4.4.2 유체역학: 열 깊이 w_z = w / cos(ψ) (Eq. 30), cos(ψ)는 3D 좌표로부터 (Eq. 31). STEREO A-B 자기일관성: T_B/T_A = 1.05 ± 0.09, n_B/n_A = 0.94 ± 0.12, w_B/w_A = 0.96 ± 0.05 (Figure 27). 과잉 압력 q_p = p_obs/p_RTV = 7.57×10⁻⁷ n_e L / T_e² (Eq. 32) → **EUV 루프는 q ≈ 3–15** (Figure 28) — 동적 비평형 가열, RTV 불성립.
- 4.4.3 자기장: Potential B = ∇Φ, ∇²Φ = 0 (Eqs. 33–34). LFF (∇×B = αB, Eq. 35), NLFFF (∇×B = α(r)B, Eq. 36). 피팅 판정 min[Δ₂(α)] = (1/s²_max) ∫ √|r_obs − r_ff(s, α)|² ds (Eq. 37). Feng et al. (2007a): 5개 STEREO 루프에서 |α| ≈ (2–8)×10⁻³ Mm⁻¹ → twist Φ = 2πn < 0.5, kink 불안정 한계 Φ ≤ 3.5π 이하. DeRosa et al. (2009): NLFFF misalignment α_mis = 24°–44° — 일부 경우 potential field보다 *못한* 경우. Aschwanden & Sandman (2010) unipolar PFU: 4개 활동영역에 대해 α_mis = 14.3° ± 11.5°, 13.3° ± 9.3°, 20.3° ± 16.5°, 15.2° ± 12.3° (Table 1). 비-potentiality α_NP = √(α²_PFU − Δα²_SE) = 11°–17°, soft X-ray flux와 상관 (Figure 29).

Section 4.5 (MHD 진동): 2007 Jun 27 17:30 UT 플레어의 루프 진동을 STEREO EUVI/A, B가 α_sep = 8.26°에서 동시 관측. **진폭 a(t) = a₀ + a₁ cos(2π(t−t₀)/P + Φ) exp(−(t−t₀)/τ_d)** (Eq. 42). **피팅값**: a₁ = 2.5 EUVI 픽셀 = 2900 km, P = 565 s ≈ 9 분, τ_d = 1600 s ≈ 27 분, 케이던스 150 s. 12개 이미지 쌍의 3D 삼각측량으로 **S자 비평면 비대칭 루프** 드러남 — 기존 반원형 가정과 불일치. 수평·수직 진폭이 거의 같고 t₀/P = 0.19 vs 0.26 위상차 → 비틀림/나선형 kink 모드 시사.

Section 4.6 (MHD 파동): slow-mode MHD 파는 음속 c_s ∝ √T_e로 전파. 파를 포함하는 루프의 삼각측량은 절대 위상 속도 제공, 투영값 보정(de-projection) 가능.

Sections 4.7–4.12 (필라멘트, 프로미넌스, 제트, 플룸, 플레어, CME 영역, 전역 파동): 각 응용 범주별 특징적 3D 정량. 하이라이트: (4.8) 분출 필라멘트 3D 궤적이 비대칭 자기 불안정 타이밍 밝힘; (4.10) 플레어 — 저고도 confined vs 고고도 분출 post-flare 루프 구분; (4.11) CME 4D 모델링 — EUV dimming이 팽창 기하(bubble, ice-cone, flux rope) 제약; (4.12) 전역 코로나 파동 — 자기음향 파면, 높이·속도가 입체법으로 제약.

### Part IV: Summary Section 5 / 요약 Section 5 (pp. 74–76)

**English.** The closing summary lists **11 phenomenon categories** with their stereoscopically-derivable quantities. I paraphrase them:
1. **Large-scale corona:** n_e(l, b, r) at r = 1.0–2.5 R☉, synoptic maps, radial scale heights.
2. **Streamers:** 3D streamer belt, double plasma sheets, triple current sheets, PFSS source surface.
3. **Active regions:** 3D n_e(l, b, h), T_e(l, b, h) from triple-filter DEM + triangulation + magnetic interpolation (ISTAR).
4. **Coronal loops:** [x(s), y(s), z(s)] triangulated → hydrostatic λ_p, hydrodynamics, magnetic model tests.
5. **MHD kink oscillations:** 3D polarization plane, coplanarity, circularity, helicity.
6. **MHD slow waves:** absolute phase speeds via de-projection.
7. **Erupting filaments:** 3D trajectory, asymmetric destabilization timing.
8. **Bright points/jets/plumes:** altitude, reconnection topology (dipolar/tripolar/quadrupolar/null-point/fan-separatrix), nano-flare frequency distribution.
9. **Solar flares:** low-lying confined vs. high-lying eruptive post-flare loops; flare volume and total energy.
10. **CME source regions and EUV dimming:** 4D CME geometry (bubble/ice-cone/flux rope); tether-cutting, shearing, break-out, kink, torus instability diagnostics.
11. **Global coronal waves:** geometric height, magneto-acoustic phase speed.

The author anticipates that future work will blur "stereoscopy" and "tomography" as full 3D/4D forward-fitting of physical models to multi-spacecraft data becomes standard, and especially highlights the triple-viewpoint STEREO + SDO/AIA synergy.

**한국어.** 마감 요약은 **11개 현상 범주**와 각각에서 입체법으로 얻을 수 있는 정량을 나열:
1. **대규모 코로나:** r = 1.0–2.5 R☉의 n_e(l, b, r), 시놉틱 맵, 반경 스케일 높이.
2. **스트리머:** 3D 스트리머 벨트, 이중 플라즈마 시트, 삼중 전류 시트, PFSS source surface.
3. **활동영역:** 3필터 DEM + 삼각측량 + 자기장 보간(ISTAR)의 3D n_e(l, b, h), T_e(l, b, h).
4. **코로나 루프:** 삼각측량된 [x(s), y(s), z(s)] → 정유체 λ_p, 유체역학, 자기 모델 검증.
5. **MHD kink 진동:** 3D 편광 평면, 공면성, 원형성, 나선성.
6. **MHD slow 파:** de-projection으로 절대 위상속도.
7. **분출 필라멘트:** 3D 궤적, 비대칭 destabilization 타이밍.
8. **밝은점·제트·플룸:** 고도, 재결합 위상(쌍극/삼극/사극/null-point/fan-separatrix), 나노플레어 빈도 분포.
9. **태양 플레어:** 저고도 confined vs 고고도 분출 post-flare 루프; 플레어 부피·에너지.
10. **CME 영역 / EUV dimming:** 4D CME 기하(bubble/ice-cone/flux rope); tether-cutting, shearing, break-out, kink, torus 불안정.
11. **전역 코로나 파:** 기하 높이, 자기음향 위상 속도.

저자는 미래에 다중 위성 자료로의 완전한 3D/4D 물리 모델 forward-fitting이 표준화되면서 "stereoscopy"와 "tomography"의 경계가 흐려질 것이라 예견하며, 세 시점 STEREO + SDO/AIA 시너지에 특히 주목.

---

## 3. Key Takeaways / 핵심 시사점

1. **STEREO changed the game by providing baseline.** / **STEREO는 baseline을 제공해 판을 바꿨다.**
   - Before 2006, all solar space missions sat near Earth. Solar rotation gave ~13.2°/day aspect change, but only for quasi-static structures. STEREO provided ~22.5°/yr drift and, crucially, *simultaneous* two-viewpoint imaging at α_sep = 1°–180° — enabling true triangulation of dynamic features.
   - 2006년 이전 모든 태양 위성이 지구 근방. 태양 자전은 하루 13.2°를 주나 quasi-static 구조에만 유효. STEREO는 연 22.5° 표류·동시 2시점 관측(α_sep = 1°–180°) 제공 → 동적 구조의 진짜 삼각측량이 가능해짐.

2. **The tie-point/epipolar framework is elegantly reducible to six equations.** / **타이-포인트·에피폴라 틀은 6개 식으로 우아하게 요약된다.**
   - Given (α_A, δ_A, α_B, δ_B, α_sep, d_A, d_B), one solves γ_A, γ_B, x_A, x_B → (x, y, z). No iteration, no nonlinear solver. This is why STEREO reconstruction code (e.g., SolarSoft `scc_measure`) fits on a single page.
   - (α_A, δ_A, α_B, δ_B, α_sep, d_A, d_B)만 있으면 γ_A, γ_B, x_A, x_B → (x, y, z)로 해석해. 반복 없음, 비선형 풀이 없음. SolarSoft `scc_measure`가 한 페이지에 들어가는 이유.

3. **All tomography on the Sun is under-constrained.** / **태양 tomography는 언제나 under-constrained.**
   - N²·N_k observations vs N³ voxels; 2–3 viewpoints vs the N_k = N theoretically needed. Geometric priors (spherical symmetry, axis symmetry, magnetic neutral surface alignment) or magnetic-field priors (PFSS, NLFFF) are *required*. This fundamentally distinguishes solar tomography from medical CT.
   - N²·N_k 관측 vs N³ voxel; 이론상 N_k = N 필요하지만 실제는 2–3 시점. 기하 prior(구면·축·자기 중립면 정렬) 또는 자기장 prior(PFSS, NLFFF)가 *필수*. 의료 CT와 본질적으로 다른 점.

4. **Magnetic models fail the stereo test.** / **자기장 모델은 입체 검증을 통과하지 못한다.**
   - NLFFF extrapolations misalign with stereo-triangulated loops by α_mis = 24°–44° — worse than hoped. The problem traces to non-force-free photospheric boundary and limited Hinode field-of-view. Optimized unipolar charge models (Aschwanden & Sandman 2010) reduce α_mis to 11°–17°, but 10° residual remains. This is a *direct diagnostic of non-potentiality* and correlates with soft X-ray flux.
   - NLFFF 외삽이 입체 삼각측량 루프와 α_mis = 24°–44°로 misalign — 기대보다 나쁨. 원인은 비-force-free photosphere와 Hinode의 좁은 FOV. 최적화된 unipolar charge 모형(Aschwanden & Sandman 2010)이 11°–17°로 개선하나 10° 잔차 남음. 이는 *비-potentiality의 직접 진단*이며 soft X-ray flux와 상관.

5. **EUV coronal loops are NOT in RTV steady state.** / **EUV 코로나 루프는 RTV 정상상태가 *아니다*.**
   - Stereoscopic L + n_e + T_e gives the over-pressure q_p = p_obs/p_RTV directly. EUV (T ≈ 1 MK) loops show q = 3–15 — far from thermal equilibrium. They are in a radiative cooling phase, post-impulsive heating. Soft X-ray (T = 3–6 MK) loops approach RTV. This established impulsive, non-steady-state heating for the cool corona — a paradigm shift from the 1980s Rosner-Tucker-Vaiana picture.
   - 입체법으로 L + n_e + T_e → 과잉 압력 q_p = p_obs/p_RTV 직접 관측. EUV(T ≈ 1 MK) 루프 q = 3–15 — 열평형에서 멀리 벗어남. 급작스러운 가열 후 복사 냉각 단계. Soft X-ray(T = 3–6 MK) 루프는 RTV 근처. 저온 코로나의 충격적·비정상 가열을 확정 — 1980s RTV 패러다임 전환.

6. **Loop inclination multiplies the observed scale height.** / **루프 기울기는 관측 스케일 높이를 증폭한다.**
   - The "communicating water tubes" geometry (Figure 24) means a loop inclined at θ = 60° shows twice the vertical scale height: λ_p^obs = λ_p / cos(60°) = 2λ_p. Without 3D stereoscopy this inclination is invisible, and hydrostatic diagnoses are biased low.
   - "연결된 물관" 기하(Figure 24): θ = 60° 기울기 루프는 관측 스케일 높이가 2배, λ_p^obs = λ_p / cos(60°) = 2λ_p. 3D 입체법 없이는 기울기가 보이지 않아 정유체 진단이 과소평가됨.

7. **Oscillating loops are NOT semi-circular or coplanar.** / **진동 루프는 반원형·공면이 *아니다*.**
   - The 2007 Jun 27 flare loop, when triangulated, revealed an S-shape with asymmetry (Δr/r up to 1.45) and non-planarity (up to 0.21 in loop radius units). The semi-circular, planar assumption of coronal seismology introduces significant systematic errors. Circular polarization with similar x- and z-amplitudes is consistent with a *helical kink mode*, not the classical linear kink.
   - 2007 Jun 27 플레어 루프의 삼각측량: S자 비대칭(Δr/r 최대 1.45)·비평면(루프 반경 기준 최대 0.21). 코로나 세이스몰로지의 반원·공면 가정은 계통 오차. 수평·수직 진폭이 유사하며 위상차가 있는 것은 고전 linear kink가 아닌 *나선형 kink 모드*와 일치.

8. **The ISTAR method is forward-fitting, not back-projection.** / **ISTAR는 역투영이 아니라 forward-fitting이다.**
   - Instead of inverting 2–3 projections (impossible), ISTAR uses ~70 triangulated skeleton loops + a magnetic extrapolation to populate ~8000 flux tubes, each with a parameterized hydrodynamic n_e(s), T_e(s). The synthesized emission is forward-projected to 3 EUVI filters and fitted. This scales: it produces a well-defined DEM spanning log T = 5.0–7.0 for an entire active region — something no pure tomographic inversion can achieve.
   - ISTAR는 2–3 투영 역산(불가능) 대신 삼각측량된 ~70개 뼈대 루프 + 자기 외삽으로 ~8000 flux tube 구축, 각각에 매개화된 n_e(s), T_e(s) 부여. 합성 방출을 3개 EUVI 필터로 forward-project하고 피팅. 활동영역 전체에 대해 log T = 5.0–7.0의 DEM을 재구성 — 순수 tomographic 반전으로는 달성 불가.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Solar-Rotation Stereoscopy / 태양 자전 입체법

**Parallax displacement for a source at (l₁, b₁) with altitude h** (Eq. 1):
$$\Delta x_{12} = (R_\odot + h)\cos(b_1)\bigl[\sin(l_1 + \omega_{syn}(t_2 - t_1)) - \sin(l_1)\bigr]$$

- R☉ = 696,000 km (solar radius / 태양 반경)
- ω_syn = 2π/T_syn with T_syn = 26.24 d (synodic rotation period / 동기 자전 주기)
- (l₁, b₁) = heliographic longitude, latitude (태양 경위도)
- h = altitude above surface (표면 위 고도) — *the unknown*
- All other quantities measurable; inversion for h is algebraic.

**Differential rotation** (Eq. 2):
$$\omega_{sid}(b) = A + B\sin^2(b) + C\sin^4(b)$$

with A = 14.71, B = −2.4, C = −1.8 deg/day. T_syn/T_sid = 26.24/24.47 = 1.0723.

**Observing interval lower bound** (Eq. 3):
$$t \approx \frac{T_{syn}}{2\pi}\frac{\Delta x}{\Delta h}$$

For Δx = 1″ = 725 km and Δh = 3000 km → t ≈ 1 day.

### 4.2 Optically-Thin Emission Integrals / 광학적 얇은 방출 적분

**Emission measure** (Eq. 4):
$$EM \propto \int n_e^2(z, T)\,dz$$

**Flux in filter λ** (Eq. 5):
$$F_\lambda = \int \frac{dEM(T)}{dT} R_\lambda(T)\,dT$$

- n_e² dependence is the Bremsstrahlung/free-free signature of optically thin EUV/X-ray emission.
- R_λ(T) is the instrument-temperature response function (e.g., EUVI 171/195/284 Å).

**Under-constrained tomographic inversion** (Eq. 6):
$$F_\lambda(X_i, Y_j, \theta_k) = \sum_l A_\lambda\, n_e^2(X_i, Y_j, Z_l) + \sigma_\lambda(X_i, Y_j, \theta_k)$$

**Image-to-helio rotation** (Eq. 7):
$$\begin{pmatrix} X_i \\ Z_l \end{pmatrix} = \begin{pmatrix} \cos\theta_k & -\sin\theta_k \\ \sin\theta_k & \cos\theta_k \end{pmatrix} \begin{pmatrix} x_i \\ z_l \end{pmatrix}$$

### 4.3 Thompson Scattering (white light) / 톰슨 산란

**Differential cross-section** (Eq. 8):
$$\frac{d\sigma}{d\omega} = \frac{1}{2} r_e^2(1 + \cos^2\chi)$$

- r_e = e²/(m_e c²) = 2.82×10⁻¹³ cm (classical electron radius / 고전 전자반경)
- χ = angle between line-of-sight and radial direction (시선 vs. 반경방향 각)

**Total cross-section** (Eq. 9):
$$\sigma_T = \frac{8\pi}{3} r_e^2 = 6.65 \times 10^{-25}\,\text{cm}^2$$

**Polarization ratio** (Eq. 10):
$$p = \frac{I_T - I_R}{I_T + I_R} = \frac{I_P}{I_{tot}}$$

- White light is n_e-linear (unlike EUV's n_e²), and polarization lets separate the scattering density from the projection.

### 4.4 Tie-Point Triangulation (core STEREO formulae) / 타이-포인트 삼각측량

Given spacecraft A, B at distances d_A, d_B with separation α_sep, observing a source at angles (α_A, δ_A) from A and (α_B, δ_B) from B:

**Triangle-interior angles** (Eqs. 15–16):
$$\gamma_A = \frac{\pi}{2} - \alpha_A, \qquad \gamma_B = \frac{\pi}{2} - \alpha_B - \alpha_{sep}$$

**Image plane X-coordinates** (Eqs. 17–18):
$$x_A = d_A \tan\alpha_A, \qquad x_B = d_B \frac{\sin\alpha_B}{\sin\gamma_B}$$

**Source (x, z) in epipolar XZ plane** (Eqs. 19–20):
$$\boxed{\;x = \frac{x_B \tan\gamma_B - x_A \tan\gamma_A}{\tan\gamma_B - \tan\gamma_A}\;}$$
$$\boxed{\;z = (x_A - x)\tan\gamma_A\;}$$

**Source Y-coordinate from YZ plane** (Eq. 21):
$$y = (d_A - z)\tan\delta_A$$

**Distance and altitude** (Eqs. 22–23):
$$r = \sqrt{x^2 + y^2 + z^2}, \qquad h = r - R_\odot$$

### 4.5 Magnetic-Field Model Metrics / 자기장 모델 지표

**Potential field** (Eqs. 33–34):
$$\mathbf{B}(\mathbf{r}) = \nabla\Phi(\mathbf{r}), \qquad \nabla\cdot\mathbf{B} = \nabla^2\Phi = 0$$

**LFF / NLFFF** (Eqs. 35–36):
$$(\nabla\times\mathbf{B}) = 4\pi\mathbf{j} = \alpha\mathbf{B} \quad \text{(LFF)}; \qquad \alpha = \alpha(\mathbf{r}) \quad \text{(NLFFF)}$$

**Fit criterion** (Eq. 37):
$$\min[\Delta_2(\alpha)] = \frac{1}{s_{max}^2} \int_0^{s_{max}} \sqrt{\bigl[\mathbf{r}_{obs}(s) - \mathbf{r}_{ff}(s, \alpha)\bigr]^2}\,ds$$

**Misalignment angle** (Eq. 39):
$$\alpha_{mis} = \arccos\left(\frac{\mathbf{r}_{obs}\cdot\mathbf{r}_{pot}}{|\mathbf{r}_{obs}|\,|\mathbf{r}_{pot}|}\right)$$

**Unipolar charge field** (Eq. 40):
$$\mathbf{B}(\mathbf{r}) = \sum_{j=1}^{N} B_j\left(\frac{z_j}{r_j}\right)^2 \frac{\mathbf{r}_j}{r_j}$$

**Dipole field** (Eq. 41):
$$\mathbf{B}(\mathbf{r}) = \frac{\mu_0}{4\pi}\sum_{j=1}^{N} \frac{3\hat{\mathbf{r}}_j(\hat{\mathbf{r}}_j\cdot\mathbf{m}_j) - \mathbf{m}_j}{|\mathbf{r} - \mathbf{r}_j|^3}$$

### 4.6 Hydrostatic Loop Equations / 정유체 루프 방정식

**Semi-circular loop altitude profile** (Eq. 25):
$$h(s) = r(s) - R_\odot = \frac{2L}{\pi}\sin\left(\frac{\pi s}{2L}\right)$$

**Barometric pressure** (Eq. 26):
$$p(s) \approx p_0 \exp\left[-\frac{h(s) - h_0}{\lambda_p(T_e)}\right]$$

**Pressure scale height** (Eq. 27):
$$\lambda_p(T_e) = \frac{2 k_B T_e}{\mu m_H g_\odot} \approx 4.7 \times 10^9 \left(\frac{T_e}{1\,\text{MK}}\right)\,\text{cm}$$

**Inclined-loop correction** (Eq. 28):
$$\lambda_p^{obs} = \frac{\lambda_p}{\cos\theta}$$

**Over-pressure ratio relative to RTV scaling** (Eq. 32):
$$q_p = \frac{p_{obs}}{p_{RTV}} = \frac{2 n_e k_B T_e}{(1/L)(T_e/1400)^3} = 7.57 \times 10^{-7} \frac{n_e L}{T_e^2}\,\,(\text{cgs})$$

### 4.7 Damped Oscillation Fit / 감쇠 진동 피팅

**Kink-mode amplitude** (Eq. 42):
$$a(t) = a_0 + a_1\cos\left(\frac{2\pi(t - t_0)}{P} + \Phi\right)\exp\left(-\frac{t - t_0}{\tau_d}\right)$$

- a₁ = initial amplitude (초기 진폭)
- P = oscillation period (진동 주기)
- τ_d = exponential damping time (지수 감쇠 시간)
- Φ = initial phase (초기 위상)

**Fitted for 2007 Jun 27 event**: a₁ = 2.5 EUVI pixel ≈ 2900 km, P = 565 s ≈ 9 min, τ_d = 1600 s ≈ 27 min.

### 4.8 Worked Numerical Example: Triangulating a Point / 점의 삼각측량 예제

Let STEREO-A and -B both be at d_A = d_B = 1 AU = 1.496×10⁸ km from the Sun, with separation α_sep = 30°. Suppose both spacecraft observe a bright point at angles (from Sun center):
- α_A = 0.25° (point appears 0.25° east of Sun center as seen from A)
- α_B = −0.15° (point appears 0.15° *west* of Sun center as seen from B)
- δ_A = 0.10° (northward displacement)

**Step 1** — Interior angles:
- γ_A = 90° − 0.25° = 89.75°
- γ_B = 90° − (−0.15°) − 30° = 60.15°

**Step 2** — Image-plane X-coordinates (small-angle approximation; tan α ≈ α for α ≪ 1°):
- x_A = d_A tan(0.25°) ≈ 1.496×10⁸ × 4.363×10⁻³ = 6.53×10⁵ km
- x_B = d_B sin(−0.15°)/sin(60.15°) ≈ 1.496×10⁸ × (−2.618×10⁻³)/0.868 ≈ −4.51×10⁵ km

**Step 3** — Source x and z:
- x = (x_B tan γ_B − x_A tan γ_A)/(tan γ_B − tan γ_A)
- tan 89.75° ≈ 229.2; tan 60.15° ≈ 1.743
- x = ((−4.51×10⁵)(1.743) − (6.53×10⁵)(229.2)) / (1.743 − 229.2)
- x = (−7.86×10⁵ − 1.497×10⁸) / (−227.5) ≈ 6.60×10⁵ km
- z = (x_A − x) tan γ_A = (6.53×10⁵ − 6.60×10⁵)(229.2) ≈ −1.6×10⁴ km

**Step 4** — Source y:
- y = (d_A − z) tan δ_A = (1.496×10⁸ − (−1.6×10⁴))(1.745×10⁻³) ≈ 2.61×10⁵ km

**Step 5** — Distance and altitude:
- r = √(x² + y² + z²) = √((6.60×10⁵)² + (2.61×10⁵)² + (1.6×10⁴)²) ≈ 7.10×10⁵ km
- h = r − R☉ = 7.10×10⁵ − 6.96×10⁵ = 1.4×10⁴ km ≈ 14,000 km

**Interpretation** — Small stereoscopic parallax of order 10⁻¹ deg at 1 AU resolves an altitude of ~14 Mm above the solar surface. This matches the low-corona EUV loop altitudes actually measured by Aschwanden et al. (2008c).

한국어 해석: 1 AU에서 10⁻¹ 도 수준의 입체 시차로 태양 표면 위 ~14 Mm 고도를 분해. Aschwanden et al. (2008c)이 측정한 저 코로나 EUV 루프 고도와 일치.

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1972 ─ SKYLAB ATM: First systematic XUV & coronagraph imaging (one-viewpoint)
  │
1979 ─ ALTSCHULER: First 3D density reconstruction of solar corona
  │    "Reconstruction of the global-scale three-dimensional solar corona"
  │    → Applies medical CT concept to Skylab data
  │
1985 ─ BERTON & SAKURAI: Solar-rotation stereoscopy of Skylab XUV loops
  │    First stereoscopic 3D coordinates: h ≈ 100 Mm, θ ≈ 25°
  │
1992 ─ DAVILA & THOMPSON: Backprojection tomography demonstrated for Sun
  │ ─ KOUTCHMY & MOLODENSKIJ: Ground eclipse stereoscopy (α ≈ 1.6°)
  │
1994 ─ ASCHWANDEN & BASTIAN: VLA radio stereoscopy
  │    Multi-frequency gyroresonance layers as altitude indicators
  │    h ≈ 3000–12,000 km for sunspot magnetic fields
  │
1995 ─ SOHO/EIT launches; coronagraphs LASCO-C1/C2/C3 begin 10+yr archive
  │
1999 ─ ASCHWANDEN ET AL.: Dynamic stereoscopy with SOHO/EIT
  │    Exploit magnetic quasi-stationarity despite plasma turnover
  │
2000 ─ FRAZIN: Regularized positive-estimation tomography
  │ ─ ASCHWANDEN ET AL.: 65 loops triangulated, L = 300–800 Mm
  │
2002 ─ WIEGELMANN & NEUKIRCH: Magnetic stereoscopy in pre-STEREO era
  │ ─ FRAZIN & JANZEN: LASCO-C2 regularized 3D tomography
  │
2006 ─ ★ STEREO-A & STEREO-B LAUNCH (Oct 26) — baseline at last
  │
2007 ─ FENG ET AL.: First true STEREO loop triangulation (AR, May/June)
  │ ─ ASCHWANDEN ET AL. (2008c): 30 loops in NOAA 10955
  │
2009 ─ DEROSA ET AL.: NLFFF vs stereo → misalignment 24°–44°
  │    Crisis in coronal magnetic modeling
  │ ─ ASCHWANDEN ET AL.: ISTAR — 8000 flux tubes for AR 10955
  │
2010 ─ ASCHWANDEN & SANDMAN: Unipolar PFU → α_mis 11°–17°
  │ ─ BUTALA ET AL.: Kalman filter tomography at r = 1.3–4 R☉
  │ ─ SDO/AIA launches (Feb 2010) — high-cadence partner for STEREO
  │
2011 ─ ★★ THIS REVIEW (Aschwanden, LRSP 8, 5) ★★
  │    Consolidates all 30+ years of methods into one unified framework
  │
... ─ Solar Orbiter (2020), PSP (2018), PUNCH (2025) extend the reach
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Altschuler (1979), "Reconstruction of the global-scale 3D solar corona" | First solar tomographic inversion from Skylab coronagraph sequences | Historical antecedent; Section 2.1 cites it as the pioneering tomography paper for the corona |
| Berton & Sakurai (1985), Solar Phys. 96 | First solar-rotation stereoscopy of XUV loops; inspired Aschwanden's 1999 dynamic stereoscopy | Section 3.1 cites it as the first rotation-based 3D coordinate measurement |
| Aschwanden et al. (1999), ApJ 515 & (2000), ApJ 531 | Dynamic stereoscopy method formalized; showed EUV loops dominated by radiative cooling | Section 3.1.2 and 4.4 build directly on these results; Figure 3 reproduces the method |
| Frazin (2000); Frazin & Janzen (2002) | Regularized tomographic inversion for LASCO pB images; Kalman filter (Frazin et al. 2005a) | Section 3.2.1 and 4.1; backbone for all modern coronal tomography of large-scale corona |
| Wiegelmann & Neukirch (2002) | Proposed magnetic stereoscopy by α-fitting force-free fields to loop projections | Section 3.4; pre-STEREO precursor of modern hybrid methods |
| Feng et al. (2007a) | First STEREO loop triangulation, α_sep = 12°, 5 loops, |α| ≈ 2–8×10⁻³ Mm⁻¹ | Section 3.3 and 4.4.3; sibling paper to Aschwanden et al. (2008c) |
| DeRosa et al. (2009), ApJ 696 | Comprehensive comparison of 11 NLFFF + 1 potential model vs stereo-triangulated loops | Section 3.4 and 4.4.3; α_mis = 24°–44° conclusion forms the "magnetic modeling crisis" narrative |
| Aschwanden & Sandman (2010), AJ 140 | Unipolar PFU model that reduces α_mis to 11°–17° | Section 3.4 and 4.4.3; Eqs. 40–41 and Figures 29–30 come from this work |
| Aschwanden (2005), *Physics of the Solar Corona* (textbook) | Comprehensive background on coronal heating, scale heights, hydrostatics | Many equations (loop parameterization, RTV scaling, scale heights) traced back to this textbook |
| Howard & Tappin (2009a), "Interplanetary coronal mass ejections observed in the heliosphere" | Complementary review of CME reconstruction far from Sun (HI-1/2 regime) | Aschwanden explicitly excludes heliospheric CMEs and refers to this work for that domain |
| Inhester (2006), "Stereoscopy basics for the STEREO mission" (tutorial) | Foundational tutorial on epipolar geometry for solar imaging | Figure 8 adopted from Inhester; Section 3.3 credits this tutorial for epipolar formalism |

---

## 7. References / 참고문헌

- Aschwanden, M. J., "Solar Stereoscopy and Tomography", *Living Reviews in Solar Physics*, **8**, 5 (2011). DOI: 10.12942/lrsp-2011-5
- Aschwanden, M. J., *Physics of the Solar Corona: An Introduction with Problems and Solutions*, 2nd ed., Springer-Praxis (2005). ISBN 978-3540307655
- Aschwanden, M. J., Newmark, J. S., Delaboudinière, J.-P., et al., "Three-dimensional stereoscopic analysis of solar active region loops. I. SOHO/EIT observations", *Astrophys. J.*, **515**, 842–867 (1999).
- Aschwanden, M. J., Alexander, D., Hurlburt, N., et al., "Three-dimensional stereoscopic analysis of solar active region loops. II.", *Astrophys. J.*, **531**, 1129–1149 (2000).
- Aschwanden, M. J., & Bastian, T. S., "VLA stereoscopy of solar active regions. I. Method and tests", *Astrophys. J.*, **426**, 425–433 (1994a).
- Aschwanden, M. J., Nitta, N. V., Wülser, J.-P., & Lemen, J. R., "First 3D reconstructions of coronal loops with the STEREO A+B spacecraft. II. Electron density and temperature measurements", *Astrophys. J.*, **680**, 1477–1495 (2008b).
- Aschwanden, M. J., Wülser, J.-P., Nitta, N. V., & Lemen, J. R., "First three-dimensional reconstructions of coronal loops with the STEREO A and B spacecraft. I. Geometry", *Astrophys. J.*, **679**, 827–842 (2008c).
- Aschwanden, M. J., Wülser, J.-P., Nitta, N. V., Lemen, J. R., & Sandman, A., "First Three-Dimensional Reconstructions of Coronal Loops with the STEREO A+B Spacecraft. III. Instant Stereoscopic Tomography of Active Regions", *Astrophys. J.*, **695**, 12–29 (2009c).
- Aschwanden, M. J., & Sandman, A. W., "Bootstrapping the coronal magnetic field with STEREO: Unipolar potential field modeling", *Astronomical J.*, **140**, 723–734 (2010).
- Altschuler, M. D., "Reconstruction of the global-scale three-dimensional solar corona", in *Image Reconstruction from Projections: Implementation and Applications*, Topics in Applied Physics **32**, pp. 105–145, Springer (1979).
- Berton, R., & Sakurai, T., "Stereoscopic determination of the three-dimensional geometry of coronal magnetic loops", *Solar Phys.*, **96**, 93–111 (1985).
- Davila, J. M., & Thompson, B. J., "Reconstruction of the solar corona by applying the algebraic reconstruction technique (ART) to Skylab images", *Bull. Amer. Astron. Soc.*, **24**, 805 (1992).
- DeRosa, M. L., Schrijver, C. J., Barnes, G., et al., "A critical assessment of nonlinear force-free field modeling of the solar corona for active region 10953", *Astrophys. J.*, **696**, 1780–1791 (2009).
- Feng, L., Inhester, B., Solanki, S. K., et al., "First stereoscopic coronal loop reconstructions from STEREO SECCHI images", *Astrophys. J.*, **671**, L205–L208 (2007a).
- Frazin, R. A., "Tomography of the solar corona. I. A robust, regularized, positive estimation method", *Astrophys. J.*, **530**, 1026–1035 (2000).
- Frazin, R. A., & Janzen, P., "Tomography of the solar corona. II. Robust, regularized, positive estimation of the three-dimensional electron density distribution from LASCO-C2 polarized white-light images", *Astrophys. J.*, **570**, 408–422 (2002).
- Inhester, B., "Stereoscopy basics for the STEREO mission", preprint arXiv:astro-ph/0612649 (2006).
- Koutchmy, S., & Molodenskij, M. M., "Three-dimensional image of the solar corona from white-light observations of the 1991 eclipse", *Nature*, **360**, 717–719 (1992).
- Kramer, M., et al., "3D Tomographic reconstruction of the inner corona with STEREO/COR-1", *Solar Phys.*, **259**, 109–121 (2009).
- Liewer, P. C., de Jong, E. M., Hall, J. R., et al., "Stereoscopic analysis of the 31 August 2007 erupting filament", *Solar Phys.*, **256**, 57–72 (2009).
- Sandman, A. W., & Aschwanden, M. J., "Constraints from STEREO triangulation of coronal loops on MHS extrapolations by magnetic dipoles", *Solar Phys.*, **270**, 503–519 (2011).
- Verwichte, E., Aschwanden, M. J., Van Doorsselaere, T., et al., "Seismology of a large solar coronal loop from EUVI/STEREO observations of its transverse oscillation", *Astrophys. J.*, **698**, 397–404 (2009).
- Wiegelmann, T., & Neukirch, T., "Computing nonlinear force-free coronal magnetic fields", *Solar Phys.*, **208**, 233–247 (2002).
