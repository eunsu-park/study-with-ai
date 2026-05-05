# Plasma Spectroscopy & Diagnostics Paper Reading List / 플라즈마 분광학·진단 논문 읽기 목록

A methodology-focused track on inferring physical conditions of optically thin astrophysical plasmas from spectroscopic observations — emission lines, line ratios, broadband filter response, atomic databases, DEM inversion algorithms, and abundance determinations. Phase B migration completed 2026-05-01: 19 entries from `Solar_Observation/` Phase 7–8 + 1 from `Solar_Physics/`. LRSP entries (#41 Laming, #59 Del Zanna & Mason) cross-referenced.

광학적으로 얇은 천체 플라즈마의 물리 조건을 분광 관측으로부터 추론하는 방법론 트랙 — 방출선, line ratio, 광대역 필터 응답, 원자 데이터베이스, DEM 역산 알고리즘, abundance. 2026-05-01 Phase B 이관 — `Solar_Observation/` Phase 7–8 에서 19편 + `Solar_Physics/` 에서 1편. LRSP entries (#41, #59) 는 cross-reference.

---

## Phase 1: Foundations — Optically Thin Radiative Transfer / 광학적으로 얇은 복사 전달 기초 (1963–1976)

### 1. The Lower Solar Corona: Interpretation of the Ultraviolet Spectrum
- **Authors**: Stuart R. Pottasch
- **Year**: 1963
- **Journal**: *The Astrophysical Journal*, Vol. 137, p. 945
- **DOI**: 10.1086/147569
- **Why it matters**: **Emission Measure(EM) 분포 분석의 효시 논문**. 9개 원소 27개 이온의 공명선을 사용해 코로나·전이영역의 EM(T) ≡ ∫n_e² dh 분포를 정의하고, 온도-밀도 구조의 상세한 모델 없이도 코로나 화학 조성을 결정하는 방법을 확립. **FIP(First Ionization Potential) 효과**도 이 논문에서 처음 보고. 이후 60년간 모든 DEM 분석의 출발점. / **Foundational paper of emission-measure (EM) distribution analysis**. Defines EM(T) ≡ ∫n_e² dh from 27 ions of 9 elements; determines coronal abundances without detailed knowledge of T–n_e structure. Also reported the FIP effect for the first time. Starting point for all subsequent DEM analyses.
- **Prerequisites**: Papers #5, #6 (atomic data); optically thin emission, ionization equilibrium, contribution function G(T,n_e) / 광학적으로 얇은 방출, 이온화 평형, contribution function
- **Status**: [ ] (migrated from Solar_Observation #66)

### 2. The Interpretation of Total Line Intensities from Optically Thin Gases. I: A General Method
- **Authors**: J. T. Jefferies, F. Q. Orrall, J. B. Zirker
- **Year**: 1972
- **Journal**: *Solar Physics*, Vol. 22, p. 307
- **DOI**: 10.1007/BF00148698
- **Why it matters**: 광학적으로 얇은 가스의 분광선 강도 해석을 위한 **일반적 형식론**을 제시 — 시선 방향의 물질 분포를 (T, n_e) 평면 위의 **이변량 분포 함수**(현대 의미의 generalized DEM)로 정의. DEM(T)는 이를 n_e에 대해 적분한 특수형. 이후 Craig & Brown(1976)이 지적할 ill-posed 성질의 형식론적 토대를 마련. / Establishes the **general formalism** for interpreting line intensities from optically thin gases — defines a **bivariate distribution function** in (T, n_e) (the generalized DEM in modern terms). Sets the formal foundation later shown ill-posed by Craig & Brown (1976).
- **Prerequisites**: Paper #1; integral equations of the first kind, bivariate emission measure / 제1종 적분 방정식, 이변량 EM
- **Status**: [ ] (migrated from Solar_Observation #67)

### 3. The Analysis of XUV Emission Lines
- **Authors**: George L. Withbroe
- **Year**: 1975
- **Journal**: *Solar Physics*, Vol. 45, pp. 301–317
- **DOI**: 10.1007/BF00158452
- **Why it matters**: 현대적 의미의 **DEM 형식론을 정립**하고, OSO-4·OSO-6 XUV 데이터에 적용하여 typical active region 평균 코로나 온도 2.1×10⁶ K, quiet region 1.5–2.1×10⁶ K를 도출. **Withbroe-Sylwester 반복법**의 원형. / **Establishes the modern DEM formalism** and applies it to OSO-4/OSO-6 XUV data, deriving mean coronal temperatures (2.1×10⁶ K active, 1.5–2.1×10⁶ K quiet). Prototype of the Withbroe-Sylwester iterative method.
- **Prerequisites**: Papers #1, #2; iterative inversion, atomic line emissivity / 반복적 inversion, 원자 방출률
- **Status**: [ ] (migrated from Solar_Observation #68)

### 4. Fundamental Limitations of X-ray Spectra as Diagnostics of Plasma Temperature Structure
- **Authors**: Ian J. D. Craig, John C. Brown
- **Year**: 1976
- **Journal**: *Astronomy & Astrophysics*, Vol. 49, pp. 239–250
- **DOI**: (ADS: 1976A&A....49..239C)
- **Why it matters**: **DEM 역문제의 ill-posed 성질을 수학적으로 엄밀하게 증명**한 결정적 논문. 연속 bremsstrahlung 스펙트럼의 kernel(Laplace 변환)이 작은 섭동에 대해 매우 불안정함을 보이고, 분광선 데이터에 대해서도 작은 관측 오차가 거대한 해의 오차로 증폭됨을 증명. **모든 정규화/제약 기반 알고리즘의 존재 이유**를 제공. / **Definitive paper establishing the ill-posed nature of the DEM inverse problem mathematically**. Shows the Laplace-transform kernel is highly unstable, justifying the need for all regularization/constraint-based algorithms.
- **Prerequisites**: Paper #2; Laplace transform stability, ill-posed inverse problems, condition number / Laplace 변환 안정성, ill-posed 역문제, 조건수
- **Status**: [ ] (migrated from Solar_Observation #69)

---

## Phase 2: Atomic Physics & Reference Databases / 원자 물리학·참조 데이터베이스 (1994–2023)

### 5. Spectroscopic Diagnostics in the VUV for Solar and Stellar Plasmas
- **Authors**: H. E. Mason, B. C. Monsignori Fossi
- **Year**: 1994
- **Why it matters**: VUV (100–3000 Å) 분광선을 사용한 광학적으로 얇은 태양·항성 플라즈마 진단의 *고전 종설*. 원자 과정(bound-bound/free/free-free), 전자 밀도/온도 진단, DEM, 원소 abundance, OSO·Skylab·SMM·Spacelab·HST·SOHO 데이터 활용을 한 권에 종합. LRSP #59 Del Zanna & Mason 2018 의 24년 전 선행판. / Classic review of VUV (100–3000 Å) spectroscopic diagnostics. Covers atomic processes, density/temperature diagnostics, DEM, abundances, and data from OSO/Skylab/SMM/Spacelab/HST/SOHO. 24-year precursor to LRSP #59 Del Zanna & Mason 2018.
- **Prerequisites**: Atomic spectroscopy basics, Boltzmann/Saha equations, optically thin emission / 원자 분광학 기초, Boltzmann/Saha, 광학적으로 얇은 방출
- **Status**: [x] (migrated from Solar_Observation #65)

### 6. CHIANTI — An Atomic Database for Emission Lines. I. Wavelengths Greater than 50 Å
- **Authors**: K. P. Dere, E. Landi, H. E. Mason, B. C. Monsignori Fossi, P. R. Young
- **Year**: 1997
- **Why it matters**: 광학적으로 얇은 천체 플라즈마(특히 태양 코로나) EUV/UV/X선 분광선을 합성·해석하는 **사실상 표준 원자 데이터베이스의 원조 논문**. 이온 에너지 준위, 파장, 복사 데이터, 전자 충돌여기율을 일관되게 제공. SOHO/CDS·EIS·SDO/AIA·Solar Orbiter/EUI 응답함수 R(T_e) 계산의 토대. / **Foundational paper of the de-facto standard atomic database** for synthesizing/interpreting optically-thin EUV/UV/X-ray spectra. Provides ion energy levels, wavelengths, radiative data, collision rates. Used to compute R(T_e) for every EUV imager/spectrometer.
- **Prerequisites**: Atomic spectroscopy basics, ionization equilibrium, contribution function G(T,n) / 원자 분광학, 이온화 평형, contribution function
- **Status**: [x] (migrated from Solar_Observation #63)

### 7. CHIANTI — An Atomic Database for Emission Lines. XVII. Version 10.1
- **Authors**: K. P. Dere, G. Del Zanna, P. R. Young, E. Landi
- **Year**: 2023
- **Why it matters**: CHIANTI v10.1 갱신판 — 인 등전자열 새 재결합률, N/O 등전자열 8개 이온 새 충돌·복사 데이터, 6개 이온 갱신 에너지·파장. CHIANTI v1 (Dere+ 1997) 의 26년 후 진화상태. / CHIANTI v10.1 update — new recombination rates for the P sequence, new e-collision/radiative data for 8 N/O ions, updated levels/wavelengths for 6 other ions. 26-year update of CHIANTI v1.
- **Prerequisites**: Paper #6; ionization/recombination cross sections, isoelectronic sequences / 이온화·재결합, 등전자열
- **Status**: [x] (migrated from Solar_Observation #64)

> _Cross-reference — LRSP #59_: Del Zanna & Mason 2018, "Solar UV and X-ray Spectral Diagnostics" — comprehensive modern review (retained in LRSP per source-track policy).

---

## Phase 3: Differential Emission Measure (DEM) — Algorithms / DEM 알고리즘 (1980–2023)

### 8. Multi-Temperature Analysis of Solar X-Ray Line Emission (Withbroe–Sylwester Method)
- **Authors**: Janusz Sylwester, J. Schrijver, Rolf Mewe
- **Year**: 1980
- **Journal**: *Solar Physics*, Vol. 67, pp. 285–309
- **DOI**: 10.1007/BF00149808
- **Why it matters**: Withbroe(1975) 형식론을 더 강건한 **반복적 정규화 알고리즘(Withbroe-Sylwester method)**으로 발전시킨 논문. 다중 분광선 EM-loci 곡선 평균을 초기 추정으로 삼고 잔차를 최소화하는 반복 절차. RESIK·SMM/XRP X-ray 분광 표준 분석법. / Develops the Withbroe (1975) formalism into a more robust **iteratively-regularized algorithm**. Standard for X-ray spectral analysis (RESIK, SMM/XRP).
- **Prerequisites**: Papers #3, #4; EM-loci curves, Tikhonov regularization / EM-loci, Tikhonov 정규화
- **Status**: [ ] (migrated from Solar_Observation #70)

### 9. Models for Inner Corona Parameters (Iterative DEM via Arcetri Code)
- **Authors**: B. C. Monsignori-Fossi, M. Landini
- **Year**: 1991
- **Journal**: *Advances in Space Research*, Vol. 11, No. 1, pp. 281–284
- **DOI**: 10.1016/0273-1177(91)90121-Y
- **Why it matters**: **Arcetri spectral code** 기반 반복적 DEM inversion 정형화. 시뮬레이션 데이터 robust 검증. 이후 CHIANTI(#6) 와 통합되어 SOHO/CDS·SUMER 표준. / Formalizes iterative DEM inversion based on the **Arcetri spectral code**; later merged into CHIANTI (#6).
- **Prerequisites**: Papers #3, #6; Arcetri/CHIANTI emissivity tables / Arcetri/CHIANTI 방출률
- **Status**: [ ] (migrated from Solar_Observation #71)

### 10. Fundamental Limitations of Emission-Line Spectra as Diagnostics of Plasma Temperature and Density Structure
- **Authors**: Philip G. Judge, Veronika Hubeny, John C. Brown
- **Year**: 1997
- **Journal**: *The Astrophysical Journal*, Vol. 475, p. 275
- **DOI**: 10.1086/303511
- **Why it matters**: Craig & Brown(1976) 의 **정보론적 분광선 확장**. emission-line 데이터로부터 DEM(T) 에 담을 수 있는 정보량 상한(linear independence·Fisher information) 정량화. 결론: 통상 분광선 세트로 ~3–5개 자유 파라미터만 신뢰성 있게 결정 가능. **현대 DEM 분석의 신중함 기준**. / **Information-theoretic extension of Craig & Brown (1976) to line spectra**. Typical line sets reliably constrain only ~3–5 free DEM parameters.
- **Prerequisites**: Paper #4; Fisher information, SVD, regularization theory / Fisher 정보, SVD, 정규화 이론
- **Status**: [ ] (migrated from Solar_Observation #72)

### 11. Markov-Chain Monte Carlo Reconstruction of Emission Measure Distributions
- **Authors**: Vinay L. Kashyap, Jeremy J. Drake
- **Year**: 1998
- **Journal**: *The Astrophysical Journal*, Vol. 503, pp. 450–466
- **DOI**: 10.1086/305964
- **Why it matters**: **베이지안 MCMC 를 DEM 복원에 처음 도입**. Metropolis sampling 으로 비물리적 smoothness 제약 완화, 신뢰구간·upper-limit 정보 통합. PINTofALE 의 `MCMC_DEM()` 함수로 코드화. / **First introduction of Bayesian MCMC for DEM reconstruction**. Codified as `MCMC_DEM()` in PINTofALE.
- **Prerequisites**: Papers #9, #10; Bayesian inference, Metropolis-Hastings / 베이지안 추론, Metropolis-Hastings
- **Status**: [ ] (migrated from Solar_Observation #73)

### 12. Temperature Diagnostics with Multichannel Imaging Telescopes (xrt_dem_iterative2)
- **Authors**: Mark A. Weber, Edward E. DeLuca, Leon Golub, Aimee L. Sette
- **Year**: 2004
- **Journal**: in *IAU Symposium 223*, ed. A. V. Stepanov et al. (Cambridge Univ. Press), p. 321
- **DOI**: 10.1017/S1743921304006088
- **Why it matters**: **EM-loci 필터비 기법**을 Hinode/XRT 용으로 정형화하고 SSW 코드 `xrt_dem_iterative2.pro` 로 구현. SDO/AIA 시대 이전 broadband DEM 표준 도구. / Formalizes the **EM-loci filter-ratio technique** for Hinode/XRT (`xrt_dem_iterative2.pro`). Standard broadband DEM tool of pre-AIA era.
- **Prerequisites**: Broadband filter response, EM-loci geometric interpretation / 광대역 필터 응답, EM-loci 기하
- **Status**: [ ] (migrated from Solar_Observation #74)

### 13. Differential Emission Measures from the Regularized Inversion of Hinode and SDO Data (demreg)
- **Authors**: Iain G. Hannah, Eduard P. Kontar
- **Year**: 2012
- **Journal**: *Astronomy & Astrophysics*, Vol. 539, A146
- **DOI**: 10.1051/0004-6361/201117576
- **Why it matters**: RHESSI X-ray 분광 분석에 쓰이던 **GSVD 기반 정규화 inversion** 을 DEM 에 도입. 빠르고(1024² 픽셀 ≲수 분), DEM 신뢰영역(uncertainty) 자연 산출. SSW/Python 코드 `demreg` 로 공개되어 SDO/AIA 코로나 연구 최다 사용 알고리즘 중 하나. / Adapts **GSVD-based regularized inversion** to DEM. Released as `demreg` (SSW/Python).
- **Prerequisites**: Papers #4, #10; generalized SVD, Tikhonov regularization, L-curve / GSVD, Tikhonov, L-curve
- **Status**: [ ] (migrated from Solar_Observation #75)

### 14. Automated Temperature and Emission Measure Analysis of Coronal Loops with SDO/AIA
- **Authors**: Markus J. Aschwanden, Paul Boerner, Carolus J. Schrijver, Anna Malanushenko
- **Year**: 2013
- **Journal**: *Solar Physics*, Vol. 283, pp. 5–30
- **DOI**: 10.1007/s11207-011-9876-5
- **Why it matters**: **AIA 6채널 자동화 단일-Gaussian DEM** (T_peak, σ_T, EM_peak). 5억 코로나 루프 자동 검출과 결합해 NOAA 11158 에서 570개 loop segment 분석, RTV 검증. 단순·빠르고 직관적, AIA full-disk 분석 워크호스. / **Automated single-Gaussian DEM for AIA's six channels**. Workhorse for AIA full-disk DEM analysis.
- **Prerequisites**: Lognormal/Gaussian DEM parameterization, AIA channel response, RTV scaling / Gaussian DEM, AIA 채널 응답, RTV 스케일링
- **Status**: [ ] (migrated from Solar_Observation #76)

### 15. Fast Differential Emission Measure Inversion of Solar Coronal Data
- **Authors**: Joseph Plowman, Charles Kankelborg, Piet Martens
- **Year**: 2013
- **Journal**: *The Astrophysical Journal*, Vol. 771, No. 1, 2
- **DOI**: 10.1088/0004-637X/771/1/2
- **Why it matters**: **희소 표현(basis pursuit) + 음수 EM 제거 반복** 으로 AIA 6채널 DEM 을 **초당 1000+ 픽셀** 처리. AIA full-disk 시간 분석 사실상 표준 속도. / Combines **basis-pursuit sparse representation with iterative removal of negative EM**, processing **>1000 pixels/sec** for AIA.
- **Prerequisites**: Paper #13; basis pursuit, non-negativity / basis pursuit, 비음수 제약
- **Status**: [ ] (migrated from Solar_Observation #77)

### 16. Thermal Diagnostics with the AIA on SDO: A Validated Method for DEM Inversions (Sparse DEM)
- **Authors**: Mark C. M. Cheung, Paul Boerner, Carolus J. Schrijver, Pascal Testa, Fan Chen, Hardi Peter, Anna Malanushenko
- **Year**: 2015
- **Journal**: *The Astrophysical Journal*, Vol. 807, No. 2, 143
- **DOI**: 10.1088/0004-637X/807/2/143
- **Why it matters**: AIA 응답 함수와 DEM 분석에서의 활용에 대한 철저한 분석. **희소 표현(sparse_em_init.pro)** 알고리즘 — basis pursuit denoise (BPDN) 의 광자 통계 적용. AIA 데이터를 정량적으로 사용하는 모든 사람에게 필수적. / Sparse DEM via basis pursuit denoise; thorough validation of AIA response functions for DEM analysis. Essential for quantitative use of AIA data.
- **Prerequisites**: AIA channel responses, sparse coding / AIA 응답함수, 희소 코딩
- **Status**: [x] (migrated from Solar_Observation #30)

### 17. Benchmark Test of Differential Emission Measure Codes
- **Authors**: Markus J. Aschwanden, Paul Boerner, Amir Caspi, James M. McTiernan, Daniel Ryan, Harry Warren
- **Year**: 2015
- **Journal**: *Solar Physics*, Vol. 290, pp. 2733–2763
- **DOI**: 10.1007/s11207-015-0790-0
- **Why it matters**: **11종 DEM 코드 동일 합성 데이터에 적용한 정량 비교** — 단일·이중·고정 Gaussian, spline, spatial synthesis, MCMC, regularized, XRT, EVE+GOES, EVE+RHESSI 평가. 평균 정확도: T_EM-weighted 0.9±0.1, EM_peak 0.6±0.2. **알고리즘 선택의 결정적 reference**. / **Only systematic benchmark of 11 DEM codes**. Decisive reference for algorithm choice.
- **Prerequisites**: Papers #11, #13, #14, #15, #16; synthetic-data validation methodology / 합성 데이터 검증
- **Status**: [ ] (migrated from Solar_Observation #78)

### 18. SITES: Solar Iterative Temperature Emission Solver for DEM Inversion of EUV Observations
- **Authors**: Huw Morgan, James Pickering
- **Year**: 2019
- **Journal**: *Solar Physics*, Vol. 294, No. 9, 135
- **DOI**: 10.1007/s11207-019-1525-4
- **Why it matters**: **응답함수에 따라 관측 intensity 를 온도 축에 직접 재분배**해 초기 DEM 을 만들고 intensity residual 로 반복 개선. positivity·smoothness 만 강제. **표준 하드웨어에서 초당 ~1000 DEM** — Plowman 2013 과 비슷한 속도이지만 코드와 수학이 훨씬 간단. 동반 GRID-SITES (Pickering & Morgan 2019, DOI: 10.1007/s11207-019-1526-3). / Redistributes observed intensity onto T-axis by response function; iteratively refines via residuals. ~1000 DEM/sec, simple math.
- **Prerequisites**: Papers #13, #15; iterative residual minimization, AIA T response / 반복 잔차 최소화, AIA T 응답
- **Status**: [ ] (migrated from Solar_Observation #79)

### 19. Robust Construction of Differential Emission Measure Profiles using a Regularized Maximum Likelihood Method
- **Authors**: Paolo Massa, A. Gordon Emslie, Iain G. Hannah, Eduard P. Kontar
- **Year**: 2023
- **Journal**: *Astronomy & Astrophysics*, Vol. 672, A120
- **DOI**: 10.1051/0004-6361/202245883
- **Why it matters**: **Regularized Maximum Likelihood (RML)** 으로 DEM 역문제 재공식화 — 광자 noise Poisson 통계를 likelihood 로 명시 모델링. 존재성·정확도·강건성·비음수성 동시 만족. ML 친화적, 머신러닝 결합 적합. Hannah-Kontar(2012) 정규화의 통계학적 후속. / **Reformulates DEM inversion as Regularized Maximum Likelihood**, explicitly modeling photon Poisson statistics. ML-friendly successor to Hannah-Kontar (2012).
- **Prerequisites**: Papers #13, #17; maximum likelihood, Poisson noise, EM algorithm / 최대우도, Poisson 잡음, EM 알고리즘
- **Status**: [ ] (migrated from Solar_Observation #80)

---

## Phase 4: Density / Temperature / Flow Line-Ratio Diagnostics / 밀도·온도·유동 line-ratio 진단

> _New entries to curate (not yet added):_
> - Doschek 1972 / 1990s — early line-ratio density diagnostics (e.g., O IV, Si III)
> - Young+ 2007 — Hinode/EIS density diagnostics
> - Brooks+ 2011 — Doppler shift / blue-wing asymmetry diagnostics
> - Polito+ 2018 — non-thermal line broadening
> - Tian+ 2014 — IRIS Mg II diagnostics

(Empty — to be curated. The methodology overlaps with Phases 2–3 but focuses on direct density/T inferences without full DEM inversion.)

---

## Phase 5: Element Abundances & FIP Effect / 원소 abundance·FIP 효과 (1963–Present)

> _Pottasch (1963, Phase 1 #1) reported the FIP effect first._

### 20. The Chemical Composition of the Sun
- **Authors**: Martin Asplund, Nicolas Grevesse, A. Jacques Sauval, Pat Scott
- **Year**: 2009
- **DOI**: 10.1146/annurev.astro.46.060407.145222
- **Why it matters**: 3D 유체역학 대기 모델과 NLTE 선 형성을 사용해 태양 원소 조성을 재도출, 금속성을 ~30% 낮춤(특히 C, N, O). 일진학적 Z 결정과 불일치하는 **"태양 조성 문제"** 도입 — 불투명도와 항성 내부 물리에 대한 10년간 논쟁 촉발. / Re-derived solar elemental abundances using 3D RHD atmospheres and NLTE line formation, lowering metallicity by ~30%. Introduced the **"solar abundance problem"** — disagreement with helioseismic Z-determinations.
- **Prerequisites**: 3D RHD atmospheres, NLTE radiative transfer, helioseismic abundance constraints / 3D RHD 대기, NLTE 복사 전달, 일진학적 조성 제약
- **Status**: [ ] (migrated from Solar_Physics #54)

### Cross-references from Living Reviews in Solar Physics / LRSP 교차 참조
> - LRSP #41 Laming 2015 — The FIP and Inverse FIP Effects in Solar and Stellar Coronae (review)
> - LRSP #48 Prieto 2016 — Solar and Stellar Photospheric Abundances
> - LRSP #59 Del Zanna & Mason 2018 — Solar UV and X-ray Spectral Diagnostics

> _New entries to curate:_
> - Schmelz & Bruner 1991 — early FIP measurements
> - Laming 2004 — ponderomotive FIP theory
> - Brooks & Warren 2011 — FIP in active regions
> - Baker+ 2018 — FIP-bias mapping with Hinode/EIS

---

## Legend / 범례
- `[ ]` not started / 시작 전
- `[~]` in progress / 진행 중
- `[x]` completed / 완료

## Migration Log / 이관 로그
**2026-05-01 Phase B**: Migrated 19 entries from `Solar_Observation/papers/reading_list.md` (#30 sparse DEM, #63–65 atomic data, #66–80 Phase 8 DEM 12편) and 1 from `Solar_Physics/papers/reading_list.md` (#54 Asplund 2009). Solar_Observation Phase 8 is now empty (cross-references retained). LRSP #41, #48, #59 cross-referenced (entries retained in LRSP per source-track policy).

`Solar_Observation/` Phase 7 (#30) 과 Phase 8 (#66–80) 전체, Phase 7의 #63–65 (atomic data), `Solar_Physics/` #54 (Asplund 2009) 이관. Solar_Observation Phase 8 은 cross-reference 만 유지.
