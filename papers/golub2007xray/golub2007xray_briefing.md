---
title: "Pre-Reading Briefing: The X-Ray Telescope (XRT) for the Hinode Mission"
paper_id: "48_golub_2007"
topic: Solar_Observation
date: 2026-04-25
type: briefing
---

# The X-Ray Telescope (XRT) for the Hinode Mission: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Golub, L., DeLuca, E., Austin, G., Bookbinder, J., Caldwell, D., Cheimets, P., et al. (2007). "The X-Ray Telescope (XRT) for the Hinode Mission." *Solar Physics*, **243**, 63–86. DOI: 10.1007/s11207-007-0182-1
**Author(s)**: L. Golub *et al.* (Harvard-Smithsonian CfA, JAXA/ISAS, NAOJ, MSFC, Palermo)
**Year**: 2007

---

## 1. 핵심 기여 / Core Contribution

XRT는 Hinode (Solar-B) 위성에 탑재된 grazing-incidence (GI) X-ray telescope로, Yohkoh/SXT 이후 가장 폭넓은 온도 감응도와 가장 높은 공간 해상도를 모두 갖춘 태양 코로나 X선 이미지 망원경이다. 이 논문은 XRT의 광학 설계 (Wolter-I 형 단일 거울쌍, generalized asphere), 9개 focal-plane analysis filter, 그리고 entrance prefilter에서 CCD에 이르는 전체 throughput을 자세히 기술하며, 지상 X선 보정 시설(XRCF, XACT)에서의 측정 결과로 설계 사양을 검증한다. 핵심 결과는 (1) 0.92 arcsec FWHM의 PSF, (2) 1 keV 부근 1.9 cm² 의 effective area, (3) log T = 6.1 – 7.5 의 광범위한 온도 진단 능력이다.

The XRT is a grazing-incidence X-ray telescope on the Hinode (Solar-B) mission that combines, for the first time on a solar X-ray imager, both the broadest coronal temperature coverage to date and the highest spatial resolution. This paper details the optical design (a single Wolter-I generalized-asphere mirror pair), the nine focal-plane analysis filters, and the end-to-end throughput from entrance prefilter through CCD. Ground calibration at NASA's X-Ray Calibration Facility (XRCF) and at the XACT facility (Palermo) verifies the as-built performance: a measured PRF FWHM of 0.92 arcsec, an on-axis effective area of 1.9 cm² near 1 keV, and a temperature diagnostic range covering 6.1 < log T < 7.5 — a regime that spans the quiet Sun, active regions, and X-class flares.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

Yohkoh의 SXT (1991–2001)는 태양 코로나의 X선 영상 천문학을 정착시켰지만 공간 해상도(2.45 arcsec/pixel)와 온도 감응 폭이 제한적이었다. 한편 EUV imager (TRACE, EIT)는 ≈1 arcsec 해상도를 달성했으나 좁은 EUV 대역 때문에 "log T = 6.0 ± 0.2" 정도의 온도 진단에 그쳤고, flare나 active-region core (T > 5 MK)는 거의 보이지 않았다. Hinode (2006년 9월 발사)의 과학 목표 — 코로나 가열, CME, 광구–코로나 결합 — 를 위해서는 EUV 의 좁은 온도 영역을 넘어 X선의 광범위 온도 영역을 ≈1 arcsec급 해상도로 영상화하는 도구가 필요했다.

By the mid-2000s, Yohkoh/SXT had pioneered solar X-ray imaging but its 2.45-arcsec pixels and limited filter set narrowed both spatial and temperature diagnostics. EUV imagers (TRACE, SOHO/EIT) reached ≈1 arcsec resolution but were confined to log T ≈ 6.0 ± 0.2 — blind to the hotter plasmas of active-region cores and flares. The Hinode mission (launched September 2006) was conceived to bridge photosphere, chromosphere, transition region, and corona; its X-ray channel — XRT — was designed to combine TRACE-class resolution with X-ray broadband temperature coverage that SXT lacked.

### 타임라인 / Timeline

```
1973  Skylab S-054   — first GI X-ray imager (filter sequencing established)
1991  Yohkoh/SXT     — Wolter-Schwarzschild, 2.45"/pixel, full-disk
1998  TRACE          — 1" EUV (narrow T)
2001  Smith APEC     — coronal emission model used by XRT
2003  Golub RSI      — XRT science requirements paper
2005  DeLuca ASR     — XRT science capabilities
2006  Hinode launch (Sep 23), XRT first light
2007  THIS PAPER     — XRT instrument paper (mirror, filters, throughput)
2010+ SDO/AIA, IRIS  — multi-band EUV; XRT continues as the X-ray companion
```

---

## 3. 필요한 배경 지식 / Prerequisites

- **Grazing-incidence (GI) X-ray optics**: At soft-X-ray wavelengths, only photons striking a polished surface at glancing angles (typically θ < 1°) reflect efficiently. Two reflections are required (Abbé sine condition); Wolter-I uses a paraboloid–hyperboloid pair. / 연 X선 영역에서는 거의 평행에 가까운 입사각(~1° 미만)에서만 반사가 일어나며, 결상을 위해 paraboloid–hyperboloid 두 면이 필요하다.
- **Coronal emission and DEM**: In an optically thin plasma, observed flux F_λ = ∫ G_λ(T) · ξ(T) dT, where ξ(T) is the differential emission measure. / 광학적으로 얇은 코로나에서 관측 플럭스는 contribution function G(λ,T)와 DEM ξ(T)의 적분으로 표현된다.
- **Spectral models (CHIANTI/APEC)**: Atomic codes that produce ε(λ,T) — the emissivity per unit emission measure used to predict instrument response. / CHIANTI 와 APEC은 emissivity 데이터베이스이며, 기기 응답함수를 예측하는 데 사용된다.
- **Effective area & temperature response**: A_eff(λ) = A_geom · R²(λ) · T_pre(λ) · T_filter(λ) · QE(λ); the temperature response is its convolution with the spectrum at temperature T. / Effective area는 거울 반사도, prefilter/analysis-filter 투과도, CCD 양자효율의 곱이며, 온도 응답은 이를 emission spectrum과 합성한 결과이다.
- **PSF / encircled energy / PRF**: Standard image-quality measures — the diameter that contains 50% (or 68%) of source energy. / PSF, encircled-energy, PRF 의 정의와 사용 방법.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| Grazing incidence | 거의 평행한 입사각(~1° 미만)에서 반사하는 X선 광학 / X-ray reflection regime where the surface is grazed at ≲1° |
| Wolter-I | Paraboloid–hyperboloid 두 반사면 한 쌍으로 구성된 X선 결상계 / classical two-reflection X-ray imaging design |
| Generalized asphere | 표면 형상이 high-order polynomial 로 자유롭게 정의된 비구면 / non-conic mirror profile defined by a polynomial expansion |
| PRF / PSF | Point Response Function / Point Spread Function — 점 광원에 대한 영상 응답 / instrument's response to an idealized point source |
| Encircled Energy (EE) | 직경 D 안에 들어오는 총 에너지 비율 / fraction of source energy within a circle of diameter D |
| Prefilter | 입사구 앞의 얇은 필터 (Al/polyimide), 가시광 차단 + 열부하 저감 / entrance filter for visible-light blocking and heat reduction |
| Analysis filter | 초점면 9종 X선 필터 휠 (Al-mesh, Al-poly, C-poly, Ti-poly, Be 시리즈 등) / nine focal-plane filters spanning ~10⁴ in thickness |
| Temperature response | T 에 대한 단위 EM당 신호 (erg/pix/s) / signal per unit emission measure as a function of T |
| Effective area A_eff(λ) | 기하학적 면적 × 모든 효율 인자 / geometric area times all efficiency factors |
| DEM | Differential Emission Measure ξ(T) = n_e² dV/dT / quantitative measure of plasma at each T |
| XRCF | NASA Marshall X-Ray Calibration Facility (518 m vacuum pipe) / NASA의 X선 보정 시설 |
| XACT | INAF Palermo의 X-Ray Astronomy Calibration and Test 시설 / INAF Palermo's X-ray calibration facility |

---

## 5. 수식 미리보기 / Equations Preview

### (1) Optically thin coronal emission / 광학적으로 얇은 코로나 방출

$$
F(\lambda) \;=\; \int G(\lambda, T)\, \xi(T)\, dT
\quad\text{with}\quad \xi(T) = n_e^2 \,\frac{dV}{dT}
$$

여기서 G(λ,T)는 contribution function, ξ(T) 는 DEM. XRT 분석은 본질적으로 이 적분의 역문제이다. / The XRT analysis problem is the inversion of this integral for ξ(T) given multi-filter observations.

### (2) Total instrument effective area / 총 effective area

$$
A_{\text{eff}}(\lambda) \;=\; A_{\text{geom}} \cdot R^{2}(\lambda) \cdot T_{\text{pre}}(\lambda) \cdot T_{\text{filt}}(\lambda) \cdot \text{QE}(\lambda)
$$

R(λ)은 단일 반사 반사도, R² 는 두 번 반사. T_pre, T_filt 는 entrance/analysis filter 투과율. / R is single-bounce reflectance (squared because the GI design has two reflections); the four transmission/efficiency factors multiply.

### (3) Temperature response per filter / 필터별 온도 응답

$$
\mathcal{R}_{f}(T) \;=\; \int A_{\text{eff},f}(\lambda)\,\varepsilon(\lambda,T)\,d\lambda
\quad [\text{erg cm}^{-2}\,\text{s}^{-1}\,\text{pix}^{-1}\,\text{per } 10^{30}\,\text{cm}^{-5}]
$$

ε(λ,T) 는 단위 EM당 emissivity (CHIANTI/APEC). XRT 9개 채널 각각에 대해 R_f(T) 를 만들어 T 진단 사용. / Convolved with a coronal emission model, this yields the curves in Figure 7.

### (4) Filter-ratio temperature diagnostic / 필터 비 기반 온도 진단 (isothermal toy)

$$
\frac{F_{f_1}}{F_{f_2}} \;=\; \frac{\mathcal{R}_{f_1}(T)}{\mathcal{R}_{f_2}(T)}
$$

광학적으로 얇고 등온 가정 하에서 두 필터의 신호비는 EM 의존성이 사라지고 T 의 함수만 된다 — XRT 의 가장 단순한 thermometry. / Under isothermal assumption, the EM cancels and the ratio is a monotonic (over a useful range) function of T.

### (5) Grazing-incidence reflectance (Fresnel limit) / Fresnel 극한 반사도

$$
R(\theta, E) \;\approx\; 1\quad\text{when}\quad \theta < \theta_c(E),\qquad
\theta_c(E) \;\propto\; \frac{1}{E}\sqrt{\rho Z/A}
$$

ρ, Z/A 는 거울 재료의 밀도와 평균 전자/핵자 비. 임계각 이상에서는 반사도가 급격히 떨어지므로 GI 망원경은 본질적으로 high-pass 가 아니라 low-pass: 광자 에너지가 높을수록 반사 효율이 낮아진다. / The critical angle θ_c sets a high-energy cutoff; XRT's effective-area drop above ~1.5 keV (Fig. 15) is a direct consequence.

---

## 6. 읽기 가이드 / Reading Guide

| 섹션 / Section | 어디에 집중할지 / What to focus on |
|---|---|
| §1 Introduction | XRT 가 풀고자 하는 코로나 관측 문제 (T = 6.1–7.5 + 1″ 해상도) / Defining the corona's spatial/temperature span |
| §2 Brief Science Overview, Tables 1, 2 | 5가지 과학 목표와 12가지 flowdown 요구사항 / map "science → engineering numbers" |
| §3.1 Mirror | Werner (1977) 의 generalized asphere, focal-plane curvature, Fig. 3–4 / mirror choice and field-curvature trade |
| §3.2–3.4 Filters | 9 analysis filter 의 재료/두께 (Table 4), Fig. 7 의 R_f(T) / where the broad-T capability comes from |
| §3.5–3.8 Shutter/VLI | TRACE-유산 셔터 + 가시광 망원경 동축 정렬 / coalignment with VLI ≈17″ |
| §4 XRCF Calibration | PRF FWHM = 0.92″, EE@27 μm = 52% (Fig. 10–13) / mirror image-quality validation |
| §5 Throughput | A_eff(E) (Fig. 15), 측정 vs 예측 필터 투과 (Table 7), 9-channel A_eff(λ) (Fig. 17) | 
| §5.4 DEM Analysis | Why ≥6 channels needed, Fig. 18 4-vs-7 채널 reconstruction / motivates multi-filter design |
| §6 Conclusions | XRT 의 최종 사양 요약 / capability statement |

읽기 순서 / Suggested order: §1 → §2 + Tables 1,2 → §3.1 → §3.4 + Fig. 7 → §4 → §5.3 + Fig. 17 → §5.4. (engineering details in §3.5–3.8 can be skimmed.)

---

## 7. 현대적 의의 / Modern Significance

XRT는 2026년 현재까지 19년 이상 작동 중이며, Hinode의 가장 오랜 운영 X선 영상 자산이다. SDO/AIA (2010–) 의 EUV 채널과 함께 사용하면 EUV+X선 multi-band DEM 재구성이 가능하여, active region core, flare loop 의 hot plasma (>5 MK) 분석에 필수적이다. XRT의 9-필터 설계와 broad-T response는 추후 임무 (NASA MUSE 후보, ESA Smile) 의 multi-channel 코로나 imager 설계에 직접적인 영향을 주었다.

As of 2026, XRT remains operational on Hinode — nearly two decades on orbit — and it is still the workhorse soft-X-ray imager for routine solar observations. Its filter-ratio and DEM analyses are paired daily with SDO/AIA and (since 2020) Solar Orbiter / EUI to study active-region heating, jets, and flare onset. The XRT design philosophy — "few-channel filter spectroscopy with broad temperature coverage" — has become the template for upcoming solar X-ray imagers (e.g., ASO-S/HXI, MUSE concepts).

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
