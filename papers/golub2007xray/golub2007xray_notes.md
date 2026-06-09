---
title: "The X-Ray Telescope (XRT) for the Hinode Mission"
authors: [Golub, L., DeLuca, E., Austin, G., Bookbinder, J., Caldwell, D., Cheimets, P., Kano, R., Tsuneta, S., et al.]
year: 2007
journal: "Solar Physics, 243, 63–86"
doi: "10.1007/s11207-007-0182-1"
topic: Solar_Observation
tags: [hinode, xrt, x-ray-telescope, grazing-incidence, wolter, coronal-imaging, dem, filter-ratio]
status: completed
date_started: 2026-04-25
date_completed: 2026-04-25
---

# 48. The X-Ray Telescope (XRT) for the Hinode Mission / Hinode 위성의 X선 망원경 (XRT)

---

## 1. Core Contribution / 핵심 기여

이 논문은 2006년 9월에 발사된 Hinode (Solar-B) 위성의 X-Ray Telescope (XRT) 설계와 지상 보정 결과를 종합적으로 보고하는 instrument paper이다. XRT는 grazing-incidence (GI) Wolter-I 형 X선 망원경으로, 단일 거울쌍(generalized asphere)을 통해 0.92 arcsec FWHM의 PRF, 1 keV 부근 1.9 cm² 의 effective area, 그리고 9개 focal-plane analysis filter 의 조합으로 log T = 6.1–7.5 의 광범위한 코로나 온도 진단 능력을 달성한다. 이전 X선 영상기인 Yohkoh/SXT (2.45″/pixel, 좁은 T 범위)의 한계를 넘어, EUV imager (TRACE) 수준의 공간 해상도와 X선 broadband T 진단을 동시에 제공하는 첫 임무이다.

The paper is the comprehensive instrument description of the X-Ray Telescope (XRT) on the Hinode (Solar-B) mission, covering optical design, focal-plane filters, end-to-end ground calibration at NASA/MSFC's XRCF, INAF/Palermo's XACT facility, and the predicted/measured throughput of all nine filter channels. XRT achieves a measured PRF FWHM of 0.92 arcsec (sub-pixel image quality), an on-axis effective area of 1.9 cm² near the Cu-L line, and a temperature diagnostic span of 6.1 < log T < 7.5 — the broadest of any solar X-ray imager flown to that date. The authors demonstrate, through Monte-Carlo DEM reconstruction tests (Fig. 18), that at least six independent filter channels are required to reliably retrieve coronal differential emission measures, motivating the nine-channel design.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction & Science Overview / 서론과 과학 목표 (§1–§2, pp. 64–65)

**문제 설정 / Problem statement.** 태양 외기는 광구의 5800 K 부터 코로나의 10⁷ K 이상까지 온도 범위를 가지며, 이에 따라 가시광–X선의 광범위한 파장과 sub-arcsec 부터 10⁵ km 까지의 공간 스케일을 동시에 포착해야 한다. XRT의 과학 목표 (Table 1) 는 다음 다섯 가지로 정리된다.

The solar outer atmosphere covers 5800 K to >10⁷ K, demanding visible-to-X-ray wavelengths and sub-arcsec to 10⁵ km spatial scales. XRT's five science objectives (Table 1) are:

1. **CMEs** — triggers, relation to **B**, link with large-scale instabilities and fine structure / CME 발생 원인과 자기장 구조와의 관계
2. **Coronal heating** — brightening dynamics, wave correlation, loop-loop interaction / 코로나 가열 메커니즘
3. **Reconnection and jets** — where/how reconnection occurs, link to **B** / 재연결과 jet 의 발생 위치
4. **Flare energetics** / 플레어 에너지학
5. **Photosphere–corona coupling** — direct connection of coronal events to photospheric magnetism / 광구–코로나 결합

**XRT의 "5가지 firsts" / The five XRT firsts** (p. 64):
- spatial resolution + FOV + cadence 결합 / unprecedented combination of spatial resolution, FOV, image cadence
- 가장 넓은 T 적용 범위 / broadest T coverage of any coronal imager to date
- 빠른 topology/T 변화 추적을 위한 high data rate / high data rate for rapid topology and temperature tracking
- 코로나 hole 부터 X-flare 까지의 dynamic range / extremely large dynamic range
- flare buffer + onboard storage + downlink rate / flare buffer with onboard storage

**Flowdown 요구사항 / Engineering flowdown** (Table 2): 12개 항목. 핵심은 (1) τ_exp = 4 ms – 10 s, (2) cadence = 2 s (reduced FOV), (3) 6.1 < log T < 7.5, (4) Δ log T = 0.2 분해능, (5) 50% encircled-energy 직경 = 2 arcsec, (6) FOV > 30 arcmin, (7) 가시광 거부도 > 10¹¹.

The 12 flowdown requirements include exposure 4 ms–10 s, cadence 2 s in reduced FOV, log T = 6.1–7.5, Δlog T = 0.2 resolution, 2″ encircled energy, FOV > 30 arcmin, and visible-light rejection > 10¹¹.

### Part II: Mirror Design / 거울 설계 (§3.1, pp. 65–69)

**기본 광학 / Basic optical design.** GI 결상에는 Abbé 사인 조건 (배율 일정) 을 만족시키기 위해 최소 두 면이 필요하다. Wolter (1952) 는 paraboloid–hyperboloid (Wolter-I), Wolter–Schwarzschild는 정확히 사인조건을 만족시키는 더 복잡한 설계. Werner (1977) 은 wide-field instrument 에서는 field-averaged PSF 가 더 나은 figure of merit이며, 현대 polishing 기술이 conic section 에서 벗어난 high-order polynomial surface (generalized asphere) 를 가능하게 함을 지적했다. XRT 는 이 generalized-asphere 접근법을 채택하여 on-axis 성능을 약간 희생하는 대신 off-axis 성능을 크게 개선한다.

GI imaging requires at least two reflections (Abbé sine condition). Werner (1977) recognised that, for wide-field instruments, field-averaged PSF is a better figure of merit than on-axis PSF. With modern polishing, a generalized asphere — a non-conic profile defined by a high-order polynomial — outperforms the classical Wolter-I across the FOV. XRT uses this approach, plus an in-flight focus mechanism (±1 mm) to fine-tune on-axis vs. off-axis trade.

**As-built parameters / 측정 사양** (Table 3, p. 68):

| 항목 / Parameter | 요구 / Required | 측정 / As-built |
|---|---|---|
| Optical design | Single mirror pair | Generalized asphere |
| Wavelength range | 6–60 Å | bare zerodur |
| Entrance diameter | 341.7 ± 0.1 mm | 341.7 mm |
| Focal length | 2708 ± 2 mm | 2707.5 mm |
| Focus knowledge | ±0.050 mm | ±1.4 mm |
| Field of view | 35 arcmin | optimized over 15 arcmin |
| 68% EE diameter | 1.57″ | 2.3″ (at 0.56 keV) |
| Effective area | 1.0 cm² | 1.9 cm² |

**Plate scale / 화각 환산.** 0–15 arcmin off-axis 영역에서 spot centroid 가 ≈0.78553 mm/arcmin 로 이동 → 10 μm/arcsec 수준의 plate scale, CCD 13.5 μm pixel 에서 ≈1.03″/pixel 의 sampling 에 해당.

The off-axis spot moves at ≈0.78553 mm/arcmin from optical axis, corresponding to ≈10 μm/arcsec at the focal plane (≈1″/pixel for the 13.5 μm CCD).

### Part III: Filters — Prefilters and Analysis Filters / 필터 (§3.2–§3.4, pp. 70–73)

**가시광 차단 요구 / VL blocking budget.** 전체 XRT 의 가시광 거부도는 10⁻¹². 이는 prefilter 와 analysis filter 가 각각 10⁻⁶ 의 광 차단을 분담함을 의미한다. 이 사양이 모든 필터의 최소 두께를 결정한다.

The total visible-light rejection of 10⁻¹² is split into 10⁻⁶ each for prefilter and analysis filter, setting their minimum thicknesses.

**Prefilter / 입사구 필터.** 6 개의 환형 분할로 구성. 재료는 Al(1200 Å) + 폴리이미드(2500 Å) + 자연 산화막 Al₂O₃(100 Å). Al 은 시간 경과 + 습도 + 진공 노출로 산화하므로 dry-N₂ 보관과 thermal conduction (필터 → 프레임) 에 신경써야 한다. Luxel Corp. 의 폴리이미드 mesh 사용.

The XRT has six annular prefilter segments — 1200 Å Al + 2500 Å polyimide + ≈100 Å Al₂O₃. Aluminium oxidises continuously from manufacture until launch, requiring dry-nitrogen handling and thermal-conduction control (the filter cools through its frame).

**9 가지 analysis filter / Focal-plane analysis filters** (Table 4):

| Filter ID | 재료 / Material | 두께 / Thickness (Å) | Support | Total oxide (Å) |
|---|---|---|---|---|
| Al-mesh | Al | 1600 | 82% mesh | 150 (Al₂O₃) |
| Al-poly | Al | 1250 | 2500 polyimide | 100 |
| C-poly | C | 6000 | 2500 polyimide | N/A |
| Ti-poly | Ti | 3000 | 2300 polyimide | 100 (TiO₂) |
| Be-thin | Be | 9 × 10⁴ | — | 150 (BeO) |
| Al-med | Al | 1.25 × 10⁵ | — | 150 |
| Be-med | Be | 3.0 × 10⁵ | — | 150 |
| Al-thick | Al | 2.5 × 10⁵ | — | 150 |
| Be-thick | Be | 3.0 × 10⁶ | — | 150 |

가장 얇은 필터 (Al-mesh, 1600 Å) 와 가장 두꺼운 필터 (Be-thick, 3.0 × 10⁶ Å) 의 두께비는 약 10⁴, 이는 dynamic range 확장을 위한 설계이다. faint quiet-Sun 은 thin filter, X-flare 는 thick filter 사용.

The thickness span across the nine filters reaches ≈10⁴ — vital for dynamic range from quiet Sun (thinnest filters) to X-class flares (thickest filters).

**Temperature response / 온도 응답** (Fig. 7). XRT 는 Smith *et al.* (2001) 의 APEC/ATOMDB 코로나 방출 모델과 컨볼루션하여 각 필터의 R_f(T) [erg s⁻¹ pix⁻¹] 곡선을 구한다. 단위 EM = 10³⁰ cm⁻⁵ 가정. Fig. 7 은 9 곡선이 log T = 6.0 부근 (Al-mesh) 에서 log T ≈ 7.0–7.5 부근 (Be-thick) 까지 차례로 피크를 갖는 형태로, 각 필터가 서로 다른 T 영역을 강조함을 보여준다. 이 형태가 XRT의 broad-T 진단 능력을 정량화한다.

The temperature response per filter (Fig. 7) is computed by convolving the predicted A_eff(λ) with the APEC coronal spectrum at unit EM = 10³⁰ cm⁻⁵. The curves peak from log T ≈ 6.0 (thinnest filter A) up to log T ≈ 7.0–7.5 (thickest, Be-thick "I"), giving a quantitative basis for filter selection.

**Filter test / 필터 시험.** Palermo XACT 에서 7 개 필터를 측정 (9 개 중 2개는 운송 중 손상 → 교체). 공간 균일도: 금속/폴리이미드 ≈ 2%, 단일 금속 ≤ 3.3%. 투과율은 예측치의 5–20% 이내. XRCF end-to-end 시험에서도 측정-예측 일치 (Table 7).

XACT testing showed spatial uniformity within 2% (metal-on-polyimide) or 3.3% (single-metal), and transmission within 5–20% of predicted. End-to-end XRCF tests confirmed the measured vs. predicted transmissions agree within counting statistics (Table 7).

### Part IV: Shutter, VLI, and Coalignment / 셔터, 가시광 imager, 정렬 (§3.5–§3.8, pp. 73–76)

**Shutter / 셔터.** TRACE 유산. 두 개의 좁은 슬릿 (1 ms, 8 ms) + ≥44 ms 용 large opening 의 회전 블레이드. start-stop 또는 multi-pass 모드로 36 단계의 노출시간 (1 ms 부터 64 s) 지원 (Table 5). 

The shutter is a TRACE heritage rotating blade with two narrow slits (1 ms, 8 ms) plus a large opening for ≥44 ms. Start–stop or multi-pass modes give 36 selectable exposures from 1 ms to 64 s.

**WL Telescope (VLI) / 가시광 망원경.** XRT와 동축·confocal 한 G-band imager (430.5 ± 18.9 nm). 초점거리 2705 mm, 50 mm aperture (f/54), 2 arcsec 해상도. focal plane filter 휠에 G-band 필터 + ND=1.3 추가. 노출시간 0.01 s 정도. Williams College 의 0.6 m 망원경에서 조정.

The VLI is a coaxial achromat (430.5 nm G-band, 18.9 nm FWHM, f/54, 50 mm aperture, 2705 mm focal length, 2″ resolution) sharing the CCD with the XRT. ND filter 1.3 yields ≈1/100 s exposures; calibrated at Williams College's 0.6 m telescope.

**Coalignment / 정렬.** XRCF 에서 Cu-L X선원 + 430 nm VL 동시 측정. VLI 와 XRT centroid의 net offset = 17.0 ± 5.0 arcsec, ≈1/4″ 의 가시광원 위치 오차에 의해 dominated. 1 arcmin 요구치 안에 안전. 정렬 정밀도(knowledge)는 발사 후 commissioning 에서 결정.

In XRCF tests with adjacent X-ray and VL sources (530.6 m distance, 14.1 cm lateral offset), the measured XRT/VLI offset was 17.0 ± 5.0 arcsec — well within the 1-arcmin requirement.

### Part V: Mirror Imaging Performance / 거울 영상 성능 (§4, pp. 76–80)

**Test plan / XRCF 시험 계획.** XRCF 는 518 m 진공 파이프 + 큰 진공 챔버. 전자빔 충돌 X선원 (Cu-L 0.93 keV 등). 5 개의 X선 lines (Table 6): C-K (0.277 keV), O-K (0.525 keV), Cu-L (0.933 keV), Al-K (1.49 keV), Mo-L (2.29 keV). 다음 7 개 시험 수행:

The XRCF — a 518 m vacuum pipe — was used with five characteristic X-ray lines (C-K to Mo-L, 0.277–2.29 keV, Table 6) for seven test categories: focus determination, on-axis PSF and EE, on-axis effective area, off-axis PSF and EE, off-axis effective area, PSF wings, and thermal response.

**On-axis PSF / 온축 PSF** (Fig. 10–11). 7 μm subpixel 이동 + 보간으로 PSF 재구성 → FWHM = 0.92 arcsec (Fig. 10). 유한 광원 거리 보정 + 중력 변형 보정 후 in-flight FWHM ≈ 0.8 arcsec.

The on-axis PRF was reconstructed via subpixel-shifted images (7 μm ≈ 1/2 pixel in z̃ and ỹ). Raw FWHM = 0.92 arcsec; corrected for finite-source distance and gravity sag, the in-flight FWHM is ≈ 0.8 arcsec — sub-pixel.

**Encircled energy / 둘러싼 에너지** (Fig. 12). 27 μm (= 2 arcsec) 직경에서 EE = 52 ± 0.7%, NASA 의 50% 요구치 충족. Goodrich 예측 (1.0 keV) 과 측정 (CCD-corrected, pinhole-corrected) 이 잘 일치.

At a diameter of 27 μm (≈2 arcsec), the measured EE is 52 ± 0.7% — meeting the 50% requirement. Both CCD-corrected and pinhole-corrected data match the Goodrich prediction at 1.0 keV.

**Off-axis PSF / 비축 PSF** (Fig. 13). RMS spot diameter (BF, no 1G/finite-source corrections) 측정값과 SAO ray-trace 예측이 −15 to +15 arcmin 범위에서 매우 잘 일치 (residual ≲ 0.5″).

Off-axis RMS diameter at the best-focus position matches the SAO ray-trace prediction across ±15 arcmin to within ≈0.5 arcsec.

**PSF wings & scattering / PSF 날개와 산란** (Fig. 14). 100 μm 와 300 μm pinhole + 22 μm 격자 위치에 FPC 측정 → 2D Lorentzian fit. core (10 μm pinhole 7×7 grid) 는 2D Gaussian. 두 모형은 r = 13 μm 에서 매끄럽게 연결. 결과: 1 arcmin off-axis 에서 산란 비율 < 10⁻⁵ at 0.93 keV. → wing requirement (60 arcsec ring 1 arcsec wide) 만족.

Combining a 2-D Gaussian core (sampled by a 10 μm pinhole, 7×7 grid within 22 μm) and a 2-D Lorentzian wing (100 μm and 300 μm pinholes, 0–1000 μm range), normalised to match at 13 μm, the wing response is < 10⁻⁵ at 1 arcmin off-axis (E = 0.93 keV). The wing requirement — that off-axis flux in a 1 arcsec ring at 60 arcsec is suppressed — is met.

### Part VI: Throughput / 총처리량 (§5, pp. 80–83)

**Mirror effective area / 거울 effective area** (Fig. 15). 두 번 반사하므로 A_eff ∝ A_geom · R²(λ). 측정 (asterisks/diamonds) 과 예측 (대시) 이 0.3–2.5 keV 범위에서 잘 일치. on-axis ≈ 1.9 cm² 부근 (1 keV 이하 평탄), 1.5 keV 위에서는 critical angle 효과로 급감. 15.6 arcmin off-axis 에서는 vignetting 으로 A_eff 가 줄어든다.

The mirror effective area depends on geometric area times R²(λ) (two reflections). On-axis effective area is ≈1.9 cm² up to ≈1 keV, then drops sharply above 1.5 keV due to the grazing-incidence critical angle. Off-axis (15.6 arcmin) vignetting lowers the area further.

**Filter transmission test / 필터 투과 시험** (Table 7). XACT 측정 vs. 예측: 대부분 measurement-uncertainty 범위 안. 두꺼운 필터 (Al-thick, Be-thick) 는 투과율이 매우 낮아 주로 통계 한계에 의해 제한.

Predicted vs. measured transmissions (Table 7) agree within counting statistics across the five XRCF lines. Thicker filters have larger fractional uncertainties because few photons get through.

**CCD QE / CCD 양자효율** (Fig. 16). E2V CCD, 1 ≲ λ ≲ 100 Å 영역 QE 곡선. 6.4 Å (Si-K edge) 에서 깊은 dip. 자세한 보정은 companion paper (Kano *et al.*, 2007).

The XRT CCD's QE peaks above ≈90% from 10–60 Å with the characteristic Si-K edge dip at 6.4 Å. Detailed CCD calibration is in Kano et al. (2007).

**Total telescope throughput / 총 처리량** (Fig. 17). 9 개 채널 (A–I) 의 A_eff(λ) [cm²] = (prefilter T) × (mirror A_eff) × (analysis-filter T) × QE. 채널 A–C (Al-mesh, Al-poly, C-poly) 는 ≈ 1 cm² 의 peak 면적, D–F (Ti-poly, Be-thin, Be-med) 는 ≈ 0.3 cm² peak, G–I (Al-med, Al-thick, Be-thick) 는 0.1–0.001 cm² peak. peak 파장은 각 필터의 K/L edge 와 polyimide cutoff 가 결정.

Figure 17 plots A_eff(λ) for all nine channels A–I. Effective-area peaks span from ≈1 cm² (thin filters) down to ≈10⁻³ cm² (Be-thick), distributed across 5–60 Å. Peak wavelengths are set by each filter's K/L edge and polyimide cutoff, giving the well-spaced T responses of Fig. 7.

### Part VII: DEM Analysis & Channel Count / DEM 분석과 채널 수 (§5.4, pp. 84–85)

**DEM 정의 / DEM definition.** 등온 가정은 XRT 픽셀 (≈1″ 코로나 plasma) 에서 일반적으로 부적절. DEM ξ(T) 의 spline 표현 + 균일한 log T 매듭 + 반복 최소제곱법으로 ξ(T) 추정 (Weber *et al.*, 2005).

The isothermal approximation is often inadequate for an XRT-pixel-sized parcel of corona. The DEM ξ(T) is represented on a spline with evenly-spaced knots in log T, and fit iteratively by least-squares (Weber et al. 2005).

**Monte-Carlo 채널 수 시험 / Monte Carlo channel-count test** (Fig. 18). CHIANTI 의 active-region 모델 DEM (두 개의 hump) 을 대상으로 4 채널 vs 7 채널 reconstruction. 4 채널 에서는 hot peak 만 분리, cool component 누락. 7 채널 에서는 두 hump 모두 재현. → "최소 6 채널" 결론, XRT 의 9-필터 설계 근거.

Figure 18 reconstructs a model AR DEM (two-hump) from synthetic XRT data: 4-channel fits resolve only the hot peak, missing the cool component; 7-channel fits recover both humps. Conclusion: ≥6 independent channels are required for reliable DEM inversion → motivates the nine-channel XRT design.

**물리적 제약 / Fundamental limit.** Craig & Brown (1976) 의 한계 — Boltzmann width 에 의해 결정되는 T 분해능의 본질적 한계 — 는 XRT 에서도 유효하다. 어떤 DEM 모델이든 inversion 의 분해능에는 fundamental ceiling 이 존재한다.

The Craig & Brown (1976) limit — DEM-inversion resolution is fundamentally bounded by the Boltzmann thermal width of the emitting lines — applies. No amount of channels can resolve δlog T below this floor.

### Part VIII: Worked Numerical Examples / 정량 예제

이 절에서는 본문의 핵심 수치 정보 몇 가지를 한 번씩 직접 트레이스하여 설계 전체의 일관성을 보인다.

This subsection traces a few key numerical chains end-to-end to demonstrate the internal consistency of the XRT design.

**(a) Plate scale at the focal plane / 초점면 plate scale.** Off-axis spot 이동률 ≈ 0.78553 mm/arcmin = 13.09 μm/arcsec. CCD pixel = 13.5 μm → 1.03 arcsec/pixel. 따라서 XRT는 ≈1″ pixel sampling 을 가진다. PRF FWHM = 0.92″ < 1 pixel 이므로 pixel-limited (under-sampled in the Nyquist sense). / The off-axis displacement rate is 13.09 μm/arcsec, giving 1.03 arcsec/pixel for the 13.5 μm CCD; with a 0.92″ FWHM, XRT is pixel-limited.

**(b) Encircled energy requirement / 50% EE 요구치 점검.** EE_50% diameter ≤ 27 μm = 27 × (1/13.09) ≈ 2.06 arcsec. 측정값 EE(27 μm) = 52% > 50% → 요구 만족. / EE-50 diameter ≤ 27 μm = 2.06 arcsec; the measured 52% exceeds the 50% requirement.

**(c) Geometric area of the annular aperture / 환형 입구 면적.** Entrance diameter D_outer = 341.7 mm; assuming an inner obstruction diameter D_inner ≈ 270 mm (typical for a Wolter-I shell with f/8 at 2.7 m focal length), the geometric annulus has area A_geom = π/4 (D_o² − D_i²) ≈ π/4 (1.168 − 0.729) × 10⁵ mm² ≈ 34 × 10² mm² ≈ 34 cm². With double-bounce reflectivity R² ≈ 0.06 at 1 keV (Zerodur grazing reflectance squared), this gives mirror-only A_eff ≈ 2 cm² — consistent with the measured ≈1.9 cm² in Fig. 15. / 입구 면적 ≈ 34 cm² × R²(1 keV) ≈ 0.06 → 거울 단일 A_eff ≈ 2 cm², 측정 1.9 cm² 와 일치.

**(d) DEM channel-count ROC / DEM 채널 수 ROC.** Fig. 18 의 정성적 결과 — 4 채널 → cool peak miss, 7 채널 → both peaks recovered — 는 다음과 같이 정량화된다. AR DEM 의 cool hump 가 log T = 6.0 부근, hot hump 는 log T = 6.7 부근. XRT 채널 가운데 cool hump 를 strongly weighted 하는 채널은 Al-mesh, Al-poly, C-poly (peaks near log T 6.0–6.4) 이고, hot hump 를 강조하는 채널은 Be-thin, Be-med, Al-med (peaks near log T 6.7–7.0). 둘 모두에서 ≥3 채널씩 (≥6 합계) 필요하다는 결론은 단순히 매개변수 (DEM 의 두 hump 위치/너비/높이) 의 차원수 (≈6) 와 일치한다. / The 6-channel rule reflects the dimensionality of the two-hump DEM model (3 parameters × 2 humps).

### Part IX: Conclusions / 결론 (§6, p. 85)

XRT는 (TRACE 의 EUV 와 함께) 비행한 가장 고해상 태양 X선 망원경이며, 광학·거울 품질 모두 FOV 전반에 걸쳐 우수하다. SOT, EIS 의 관측과 결합하여 broad T 응답, 큰 dynamic range, 높은 throughput 으로 CMEs (개시, 자기 fine structure), 코로나 가열 (loop dynamics, waves, loop–loop interactions), 플레어, reconnection·jet, 광구–코로나 결합 분야에서 돌파구적 과학을 가능하게 한다.

XRT is the highest-resolution solar X-ray telescope ever flown (matching TRACE in EUV). Combined with SOT and EIS, its broad-T response, large dynamic range, and high throughput enable breakthrough science across CME onset, coronal heating, flares, reconnection/jets, and photosphere–corona coupling.

---

## 3. Key Takeaways / 핵심 시사점

1. **Generalized-asphere mirror outperforms classical Wolter-I across the FOV / 일반화된 비구면이 FOV 전반에서 고전적 Wolter-I 를 능가** — Werner (1977) 가 제시한 field-averaged PSF 를 figure of merit 으로 채택하면, on-axis 성능을 약간 양보하더라도 off-axis 가 크게 좋아진다. XRT의 ±15 arcmin 영역 RMS spot 이 그 결과를 증명한다 (Fig. 13). / By using field-averaged PSF as the figure of merit, the generalized asphere accepts a small on-axis penalty in exchange for substantially better off-axis quality — quantified by the RMS spot diameter staying < 2.5 arcsec across ±15 arcmin (Fig. 13).
2. **Sub-pixel image quality validated by ground calibration / 지상 보정으로 sub-pixel 영상 품질 검증** — XRCF에서 측정한 PRF FWHM = 0.92″ 에 finite-source distance + 1G 변형 보정을 적용한 in-flight FWHM ≈ 0.8″ 로, CCD 1 pixel (≈1″) 보다 작아 픽셀-한정 (pixel-limited) imager 로 작동한다. / XRCF measurement gives FWHM = 0.92″ → in-flight ≈ 0.8″ — sub-pixel, so the XRT is pixel-limited rather than optics-limited.
3. **Nine filters span ≈10⁴ in thickness for dynamic-range coverage / 9 개 필터의 두께비 ≈10⁴ — dynamic range 확장** — Al-mesh (1600 Å) 부터 Be-thick (3.0 × 10⁶ Å) 까지의 두께 변화가 quiet Sun 에서 X-flare 까지 saturation 없이 모두 영상화하게 한다. / The 10⁴ thickness span from Al-mesh to Be-thick lets the XRT image quiet Sun to X-class flares without saturation.
4. **Temperature response is the convolution of A_eff(λ) and the coronal spectrum / 온도 응답은 A_eff(λ) 와 코로나 스펙트럼의 컨볼루션** — R_f(T) = ∫A_eff,f(λ) ε(λ,T) dλ (APEC/ATOMDB 사용). 9 곡선이 log T = 6.0–7.5 에 차례로 피크를 가지므로 multi-channel 에서 broad-T DEM 진단이 가능. / R_f(T) computed with APEC/ATOMDB peaks across log T ≈ 6.0–7.5 across the 9 channels (Fig. 7), giving the broad-T diagnostic.
5. **At least six independent channels are required for reliable DEM / 신뢰할 만한 DEM 재구성에는 최소 6 채널이 필요** — Monte-Carlo (Fig. 18): 4 채널은 cool component 누락, 7 채널은 both humps 복원. 9-필터 설계의 정량적 근거. / Four channels miss the cool component of an active-region DEM; seven recover both peaks. This Monte-Carlo experiment quantitatively motivates the nine-channel design.
6. **Grazing-incidence critical angle sets the high-energy cutoff / Grazing-incidence 임계각이 고에너지 차단을 결정** — Fig. 15 의 effective area 가 1.5 keV 위에서 급감하는 것은 R(θ,E) ≈ 1 (θ < θ_c) → R ≪ 1 (θ > θ_c) 의 임계각 효과의 직접적 결과. XRT 가 hard X-ray (>10 keV) 에 비감응한 이유. / The sharp drop in A_eff above 1.5 keV reflects the critical-angle behaviour of grazing reflectance, a fundamental constraint of the GI design.
7. **End-to-end calibration confirmed within counting statistics / End-to-end 보정이 통계 한계 내에서 일치** — Table 7 의 5 가지 X선 emission line × 9 가지 필터에서 측정-예측 일치. on-orbit DEM analysis 가 ground calibration 위에서 정량적으로 정당화됨. / Across 5 X-ray lines × 9 filters, measured and predicted transmissions agree within counting statistics, justifying quantitative on-orbit DEM analysis.
8. **Sub-orbit photometric stability via on-board flare buffer / Onboard flare buffer 로 in-orbit photometric 안정성** — telemetry 한계 (2.4 Mb/s, 60 MB/orbit) 안에서 X-flare 의 ms-scale brightening 을 잡기 위한 flare buffer 와 36 단계 노출 시간 (1 ms – 64 s, Table 5) 의 조합이 핵심. / The combination of flare buffer, 36-step exposure table, and 2-arcsec cadence enables ms-scale flare brightening to be captured within the telemetry budget.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Optically thin coronal flux / 광학적으로 얇은 코로나 플럭스

$$
F(\lambda) \;=\; \int_{T} G(\lambda, T)\, \xi(T)\, dT
$$

- $G(\lambda, T)$ : contribution function (line/continuum 합) [erg cm³ s⁻¹ Å⁻¹]
- $\xi(T) = n_e^2 \, dV/dT$ : differential emission measure [cm⁻⁵ K⁻¹]
- $F(\lambda)$ : observed flux [erg cm⁻² s⁻¹ Å⁻¹]

이 식이 XRT analysis 의 출발점. ξ(T) 의 inversion 은 fundamental ill-posed inverse problem. / This is the starting point for XRT data analysis; recovering ξ(T) from a finite set of filter signals is the central inverse problem.

### 4.2 Filter-integrated signal / 필터 적분 신호

$$
S_f \;=\; \int A_{\text{eff},f}(\lambda)\, F(\lambda)\, d\lambda
\;=\; \int \mathcal{R}_f(T)\, \xi(T)\, dT
$$

with the temperature response

$$
\mathcal{R}_f(T) \;\equiv\; \int A_{\text{eff},f}(\lambda)\, \varepsilon(\lambda,T)\, d\lambda,
$$

여기서 ε(λ,T) 는 단위 EM 당 emissivity (CHIANTI/APEC). R_f(T) 는 필터 f 의 "온도 응답" — Fig. 7. / $\varepsilon(\lambda, T)$ is the emissivity per unit EM (from CHIANTI/APEC); $\mathcal{R}_f(T)$ is the per-filter temperature response (Fig. 7).

### 4.3 Effective area decomposition / Effective area 분해

$$
A_{\text{eff},f}(\lambda) \;=\; A_{\text{geom}}\;\cdot\; R^{2}(\lambda)\;\cdot\; T_{\text{pre}}(\lambda)\;\cdot\; T_f(\lambda)\;\cdot\; \text{QE}(\lambda)
$$

- $A_{\text{geom}}$ ≈ π (D_outer² − D_inner²)/4 : 환형 entrance aperture 의 기하학적 면적 / annular geometric area
- $R^{2}(\lambda)$ : Wolter-I 두 번 반사 / two reflections in Wolter-I
- $T_{\text{pre}}(\lambda)$ : 1200 Å Al + 2500 Å polyimide + Al₂O₃ 의 투과 / prefilter transmission
- $T_f(\lambda)$ : 9 가지 analysis filter 중 하나 / chosen analysis-filter transmission
- $\text{QE}(\lambda)$ : E2V CCD 양자효율 / E2V CCD quantum efficiency

### 4.4 Grazing-incidence reflectance (Fresnel limit) / Fresnel 극한 반사도

광학상수 n = 1 − δ + iβ 에 대해, 작은 grazing angle θ 에서

For X-ray optical constants $n = 1 - \delta + i\beta$ (with $\delta \sim 10^{-3}$–$10^{-5}$ and $\beta \sim 10^{-4}$–$10^{-6}$), at small grazing angle θ,

$$
R(\theta, E) \;\approx\;
\begin{cases}
\;\sim 1 & \text{if } \theta < \theta_c(E) \\
\;\ll 1 & \text{if } \theta > \theta_c(E)
\end{cases}
\qquad
\theta_c(E)\;\approx\;\sqrt{2\delta}\;\propto\;\frac{1}{E}\sqrt{\rho\,Z/A}
$$

여기서 ρ, Z/A 는 거울 재료 (XRT의 경우 bare Zerodur) 의 밀도와 평균 전자/핵자 비. Wolter-I 두 번 반사이므로 throughput ∝ R². / $\rho$ and $Z/A$ are the mirror material's density and mean electrons-per-nucleon (here bare Zerodur). The Wolter-I two-bounce design suppresses transmission as $R^{2}$.

### 4.5 Filter-ratio thermometry (isothermal toy) / 필터 비 온도계 (등온 모형)

코로나 픽셀이 단일 온도 T로 등온이라 가정하면,

If the corona within a pixel is isothermal at temperature T, then $\xi(T') = \mathrm{EM}\,\delta(T'-T)$ and

$$
S_f \;=\; \mathcal{R}_f(T)\cdot\mathrm{EM}\quad\Rightarrow\quad
\frac{S_{f_1}}{S_{f_2}} \;=\; \frac{\mathcal{R}_{f_1}(T)}{\mathcal{R}_{f_2}(T)}
$$

EM 은 비에서 소거되어 ratio 는 T 만의 함수. R_{f₁}/R_{f₂}(T) 가 단조 (monotonic) 인 T 영역에서만 unique inversion 가능. XRT 에서 가장 자주 사용되는 사실상 "filter-ratio" thermometry 의 기본식. / The EM cancels, leaving a monotonic-in-T ratio (over its useful range). This is the everyday XRT "filter-ratio" thermometry.

### 4.6 Encircled energy and PSF metrics / EE 와 PSF 지표

For a circularly symmetric PSF $p(r)$ normalised so $\int_0^\infty 2\pi r\, p(r)\,dr = 1$,

$$
\mathrm{EE}(D) \;=\; \int_0^{D/2} 2\pi r\, p(r)\, dr
$$

XRT requirement: EE(2″) ≥ 50%. 측정값 = 52 ± 0.7% at D = 27 μm = 2 arcsec (Fig. 12). / EE(2″) ≥ 50% (NASA spec). XRT measurement: 52 ± 0.7% at D = 27 μm.

### 4.7 PSF wing model / PSF 날개 모형

XRT 의 PSF 는 core 와 wing 두 영역으로 나누어 모형화한다.

The XRT PSF is modelled as two pieces, joined at $r_0 = 13$ μm:

$$
p(r) \;=\;
\begin{cases}
A_g \exp\!\big[-r^2/(2\sigma_g^2)\big] & \text{(Gaussian core, } r < r_0\text{)}\\[4pt]
A_l \big/\big[1 + (r/r_l)^2\big] & \text{(Lorentzian wing, } r \geq r_0\text{)}
\end{cases}
$$

with $A_g, A_l, \sigma_g, r_l$ chosen so the two pieces match continuously. Result: scattering at 1 arcmin off-axis < 10⁻⁵ at 0.93 keV. / Continuity at $r_0$ fixes the amplitudes; the result reproduces a wing response < 10⁻⁵ at 1 arcmin (E = 0.93 keV).

### 4.8 DEM inversion (linear least-squares with knots) / DEM inversion 선형 최소제곱

ξ(T) 를 N 개 매듭 (knot) 위의 spline 으로 표현, 각 매듭의 값을 c_i 라 하면

If ξ(T) is represented as a spline with N evenly-spaced knots c_i in log T,

$$
S_f \;=\; \sum_{i=1}^{N}\;\Bigl[\int \mathcal{R}_f(T)\,\phi_i(T)\,dT\Bigr]\, c_i \;\equiv\; \sum_i K_{f i}\,c_i,
$$

행렬 K (N_filters × N_knots) 의 최소제곱 / 정칙화 inversion 으로 c_i 결정. N_filters ≥ 6 일 때만 안정한 해 (Fig. 18). / The kernel matrix $K_{fi} = \int \mathcal{R}_f(T)\,\phi_i(T)\,dT$ depends on the filter set. Stable inversion requires $N_\text{filters} \gtrsim 6$ (Fig. 18).

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1973  Skylab S-054 X-ray Spectrographic Telescope (Vaiana et al.)
            │  first solar GI X-ray imager; filter sequencing
            ▼
1977  Werner — generalized asphere as figure of merit (Appl. Opt. 392)
            │  field-averaged PSF supersedes on-axis ideal
            ▼
1991  Yohkoh / SXT (Tsuneta et al., Solar Phys. 136, 37)
            │  Wolter-Schwarzschild, 2.45"/pixel, full-Sun, T-coverage 6.0–6.7
            ▼
1998  TRACE EUV imager (Handy et al.)
            │  ≈1" resolution, narrow EUV temperature
            ▼
2001  Smith et al. — APEC/ATOMDB coronal emission code (ApJ 556, L91)
            │  basis of XRT temperature-response calculation
            ▼
2003  Golub — XRT science requirements (RSI 74, 4583)
2005  DeLuca et al. — XRT science capabilities (Adv. Sp. Res. 36, 1489)
2005  Weber et al. — DEM inversion procedure (IAU 223, 321)
            │
            ▼
2006  Hinode launch (Sep 23); first light
2007  ★ THIS PAPER — XRT instrument (Solar Phys. 243, 63–86)
            │  Wolter-I generalized asphere, 9 filters, 0.92" PRF, log T 6.1–7.5
            ▼
2007+ Kano et al. — XRT camera companion paper
2010  SDO / AIA — multi-band EUV; routinely paired with XRT for DEM
2020+ Solar Orbiter / EUI; ASO-S / HXI — XRT design philosophy continues
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Tsuneta *et al.* 1991 (Yohkoh/SXT) | Direct predecessor — Wolter-Schwarzschild GI design with 2.45″/pixel | XRT inherits the GI X-ray imaging approach but improves resolution and T coverage significantly / Yohkoh/SXT 의 직접 후속, 해상도와 T 진단 능력 모두 개선 |
| Werner 1977 | Theoretical foundation for the generalized-asphere choice | 설계 철학의 출발점 / "Why not classical Wolter?" 의 답 |
| Smith *et al.* 2001 (APEC) | Coronal emission spectrum used to compute R_f(T) | XRT의 9개 필터 온도 응답 곡선 (Fig. 7) 을 만드는 핵심 입력 / supplies $\varepsilon(\lambda,T)$ for Fig. 7 |
| Weber *et al.* 2005 | DEM inversion algorithm (spline-knot least-squares) | XRT의 DEM 분석 절차의 정의 / Defines the iterative least-squares DEM procedure |
| Craig & Brown 1976 | Fundamental DEM-inversion resolution limit | 채널을 더 추가해도 넘을 수 없는 분해능 한계 / Theoretical floor on δlog T resolution |
| Kano *et al.* 2007 | Companion paper on the XRT CCD camera | QE curve (Fig. 16) 의 정량적 근거 / Provides camera calibration that completes the throughput chain |
| Golub 2003; DeLuca *et al.* 2005 | Earlier statements of XRT's science requirements & capabilities | 이 논문 (instrument paper) 의 모(母) 논문 / Parent papers of the requirement set |
| Nariai 1987, 1988 | Off-axis defocus trade in GI imaging (Yohkoh era) | XRT의 focus-mechanism 설계 근거 / Underlies XRT's in-flight focus capability |

---

## 7. References / 참고문헌

### Primary / 본 논문
- Golub, L., DeLuca, E., Austin, G., Bookbinder, J., Caldwell, D., Cheimets, P., *et al.* (2007). "The X-Ray Telescope (XRT) for the Hinode Mission." *Solar Physics* **243**, 63–86. DOI: 10.1007/s11207-007-0182-1

### Cited / 인용
- Barbera, M., *et al.*: 2004, In: Hasinger, G., Turner, M.J.L. (eds.), *UV and Gamma-Ray Space Telescope Systems*, Proc. SPIE **5488**, 423.
- Craig, I.J.D., Brown, J.C.: 1976, *Astron. Astrophys.* **49**, 239.
- DeLuca, E., *et al.*: 2005, *Adv. Space Res.* **36**, 1489.
- Golub, L.: 2003, *Rev. Sci. Instrum.* **74**, 4583.
- Kano, R., *et al.*: 2007, *Solar Phys.* (companion paper).
- Nariai, K.: 1987, *Appl. Opt.* **26**, 4428.
- Nariai, K.: 1988, *Appl. Opt.* **27**, 345.
- Smith, R.K., Brickhouse, N.S., Liedahl, D.A., Raymond, J.C.: 2001, *Astrophys. J.* **556**, L91 (APEC/ATOMDB).
- Tsuneta, S., *et al.*: 1991, *Solar Phys.* **136**, 37 (Yohkoh/SXT).
- Weber, M.A., DeLuca, E.E., Golub, L., Sette, A.L.: 2005, In: *Multi-Wavelength Investigations of Solar Activity*, IAU Symp. **223**, 321.
- Werner, W.: 1977, *Appl. Opt.* **392**, 760.

### Related (further reading) / 관련 (심화)
- Handy, B.N., *et al.*: 1999, *Solar Phys.* **187**, 229 (TRACE).
- Lemen, J.R., *et al.*: 2012, *Solar Phys.* **275**, 17 (SDO/AIA).
- Aschwanden, M.J.: 2005, *Physics of the Solar Corona*, Springer–Praxis.
- Wolter, H.: 1952, *Ann. Phys.* **10**, 94 (foundational Wolter-I/II/III X-ray optics paper).
- Vaiana, G.S., *et al.*: 1973, *Astrophys. J.* **185**, L47 (Skylab S-054, first solar GI X-ray imaging).
- Reale, F.: 2014, *Living Rev. Solar Phys.* **11**, 4 (coronal-loop diagnostics; modern context for filter-ratio thermometry).
- Del Zanna, G., Mason, H.E.: 2018, *Living Rev. Solar Phys.* **15**, 5 (CHIANTI atomic database; modern successor to APEC for emissivity calculations).
- Narukage, N., *et al.*: 2011, *Solar Phys.* **269**, 169 (XRT in-flight calibration update — important companion for any quantitative reanalysis).
- Narukage, N., *et al.*: 2014, *Solar Phys.* **289**, 1029 (post-launch contamination model affecting analysis-filter transmission over time).
- Reale, F., *et al.*: 2007, *Astrophys. J.* **666**, 1245 (early scientific use of XRT temperature diagnostics on hot active-region loops).
- Golub, L., Pasachoff, J.M.: 2009, *The Solar Corona*, 2nd ed., Cambridge Univ. Press (textbook treatment of GI optics and coronal observation).
