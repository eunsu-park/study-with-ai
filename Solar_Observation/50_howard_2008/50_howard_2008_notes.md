---
title: "Sun Earth Connection Coronal and Heliospheric Investigation (SECCHI)"
authors: ["R. A. Howard", "J. D. Moses", "A. Vourlidas", "J. S. Newmark", "D. G. Socker", "S. P. Plunkett", "C. M. Korendyke", "J. W. Cook", "A. Hurley", "J. M. Davila", "W. T. Thompson", "O. C. St Cyr", "E. Mentzell", "K. Mehalick", "J. R. Lemen", "J. P. Wuelser", "D. W. Duncan", "T. D. Tarbell", "C. J. Wolfson", "A. Moore", "R. A. Harrison", "N. R. Waltham", "J. Lang", "C. J. Davis", "C. J. Eyles", "H. Mapson-Menard", "G. M. Simnett", "J. P. Halain", "J. M. Defise", "E. Mazy", "P. Rochus", "R. Mercier", "M. F. Ravet", "F. Delmotte", "F. Auchere", "J. P. Delaboudiniere", "V. Bothmer", "W. Deutsch", "D. Wang", "N. Rich", "S. Cooper", "V. Stephens", "G. Maahs", "R. Baugh", "D. McMullin", "T. Carter"]
year: 2008
journal: "Space Science Reviews"
doi: "10.1007/s11214-008-9341-4"
topic: Solar_Observation
tags: [STEREO, SECCHI, coronagraph, EUV, heliospheric_imager, CME, instrumentation, space_weather]
status: completed
date_started: 2026-04-25
date_completed: 2026-04-25
---

# 50. Sun Earth Connection Coronal and Heliospheric Investigation (SECCHI) / SECCHI: 태양–지구 연결 코로나·헬리오스피어 관측 패키지

---

## 1. Core Contribution / 핵심 기여

This paper is the canonical instrumentation reference for the SECCHI suite — five co-aligned and chained telescopes flown identically on the two STEREO spacecraft (Ahead and Behind, launched 2006 October 25). EUVI (1.0–1.7 R☉, four EUV emission lines), COR1 (1.4–4 R☉, internally-occulted refractive Lyot coronagraph), COR2 (2.5–15 R☉, externally-occulted Lyot), and the two Heliospheric Imagers HI-1 (15–84 R☉) and HI-2 (66–318 R☉) jointly cover, for the first time, the entire radial path from the chromosphere to beyond Earth orbit (1 AU = 215 R☉). All five share a common 2048×2048 back-illuminated e2v CCD42-40 detector, a common SECCHI Electronics Box (SEB) running on a RAD750 PowerPC, and a common command/telemetry framework. Stereoscopy emerges naturally because the two spacecraft drift away from Earth at ~22°/year, providing a continuously growing parallax baseline.

이 논문은 STEREO 임무에 탑재된 SECCHI 패키지의 표준 instrumentation reference이다. 두 STEREO 위성(Ahead/Behind, 2006년 10월 25일 발사)에 동일하게 실린 다섯 개의 망원경 — EUVI(1.0–1.7 R☉, 네 EUV 방출선), COR1(1.4–4 R☉, 내부차폐 refractive Lyot), COR2(2.5–15 R☉, 외부차폐 Lyot), 그리고 두 헬리오스피어 영상기 HI-1(15–84 R☉)·HI-2(66–318 R☉) — 이 함께 채층(chromosphere)에서 지구 궤도 너머(1 AU = 215 R☉)까지 전 영역을 인류 최초로 끊김 없이 영상화한다. 다섯 망원경 모두 동일한 2048×2048 e2v CCD42-40 검출기, 공통 SECCHI Electronics Box(SEB, RAD750 PowerPC), 공통 명령/원격측정 프레임워크를 공유한다. 두 위성이 매년 ~22°씩 지구로부터 멀어지면서 자연스럽게 입체 기저선이 늘어난다.

The paper's significance is twofold. First, it documents engineering choices that became standards for later missions: 4-quadrant Mo/Si multilayer mirrors for multi-wavelength EUV imaging, externally-occulted Lyot designs scaled to wide fields, and (most novel) shutterless wide-field imaging through a 5-vane Fresnel knife-edge cascade baffle that achieves ~10⁻¹³ B☉ stray-light rejection. Second, it formalizes the operational concept that supports modern space-weather forecasting: a 504 bps real-time "beacon mode" for continuous space weather, ICER lossy wavelet compression up to 200×, and an open data policy that releases Level-0.5 FITS files to the community within 30 minutes of routine processing.

이 논문의 의의는 두 가지이다. 첫째, 이후 임무들의 표준이 된 공학적 선택을 문서화: 다파장 EUV 영상을 위한 4-사분면 Mo/Si 다층막 거울, 광시야로 확장된 외부차폐 Lyot 설계, 그리고 가장 새로운 — 5단 Fresnel 칼날(knife-edge) 카스케이드 배플로 ~10⁻¹³ B☉ 미광 거부를 달성하는 무셔터(shutterless) 광시야 영상기. 둘째, 현대 우주기상 예보를 떠받치는 운용 컨셉을 정형화: 504 bps의 실시간 "beacon" 채널, 최대 200×의 ICER 손실 wavelet 압축, 그리고 정상 처리 후 30분 내 Level-0.5 FITS 파일을 공개하는 개방 데이터 정책.

---

## 2. Reading Notes / 읽기 노트

### Part I: Mission Overview & Science Drivers (§1) / 임무 개관과 과학 동기

The introduction (p. 67–70) frames SECCHI within the long history of CME observations: discovered in 1971 (OSO-7) and observed by five subsequent missions, but always from a single vantage point. The paper articulates four primary science questions: (i) timing of physical properties in CME initiation, (ii) 3-D structure and kinematics of CMEs, (iii) 3-D structure of active regions/loops/streamers, (iv) critical forces controlling CME propagation through corona and interplanetary medium. Five "first-time" capabilities are highlighted: stereoscopic CMEs; CME observations matched to in-situ measurements; simultaneous optical and radio CME/shock observations; observations of geo-effective CMEs along the Sun-Earth line; CMEs detected in a field of view that includes Earth.

서론(p. 67–70)은 SECCHI를 CME 관측사의 긴 흐름 속에 위치시킨다 — 1971년 OSO-7에서 처음 발견된 이래 다섯 임무가 관측해 왔지만 모두 단일 시점이었다. 논문이 제시하는 네 가지 일차 과학 목표는: (i) CME 초기 물리량의 timing, (ii) CME의 3D 구조와 운동학, (iii) 활동영역·loop·streamer의 3D 구조, (iv) 코로나·행성간 매질에서 CME 전파를 지배하는 결정적 힘. 다섯 가지 "최초의" 능력이 강조된다: 입체 CME 관측, in-situ 측정과 결합된 CME 관측, 광·전파 동시 관측, 지구 방향(geo-effective) CME의 sun-Earth line 추적, 지구가 포함된 시야 내 CME 검출.

The physical packaging is divided into three units per spacecraft: the **SCIP** (Sun-Centered Instrument Package — EUVI + COR1 + COR2 + Guide Telescope, on a 6.3 cm-thick aluminum honeycomb optical bench, 112×70 cm), the **HI** package (the Heliospheric Imagers in a 720×420×240 mm box mounted on the Earth-facing side), and the **SEB** electronics. The SCIP is co-aligned across its four telescopes via titanium flexure mounts (5 arcsec shimming resolution) and thermally enclosed in a multi-layer-insulation tent.

물리적으로 패키지는 우주선당 셋으로 나뉜다: **SCIP**(Sun-Centered Instrument Package — EUVI + COR1 + COR2 + Guide Telescope, 6.3 cm 두께 알루미늄 honeycomb 광학 벤치 112×70 cm 위에 탑재), **HI** 패키지(헬리오스피어 영상기 두 대를 720×420×240 mm 상자에 담아 지구 방향 면에 장착), **SEB** 전자 장치. SCIP의 네 망원경은 티타늄 flexure 마운트로 5″ shimming 정밀도까지 공통 정렬되며 다층 절연 텐트로 열적으로 둘러싸여 있다.

### Part II: EUVI — Extreme Ultraviolet Imager (§2) / EUVI: 극자외선 영상기

EUVI (§2, p. 70–76) is a Ritchey-Chrétien (RC) telescope with secondary mirror magnification of 2.42, primary aperture 98 mm, effective focal length 1750 mm, providing pixel-limited resolution across a circular ±1.7 R☉ field of view at 1.6″/pixel on the 2048² detector. Critically, the RC's primary and secondary mirrors are each divided into four optical quadrants, *each coated with a different Mo/Si multilayer* targeting one of four EUV emission lines:

EUVI(§2, p. 70–76)는 Ritchey-Chrétien 망원경으로, 2차거울 배율 2.42, 1차 거울 구경 98 mm, 유효 초점거리 1750 mm, 2048² 검출기 위에 1.6″/픽셀의 화소 한계 분해능을 ±1.7 R☉ 원형 시야 전체에 제공한다. 결정적인 특징은 1차/2차 거울이 각각 네 사분면으로 나뉘어 *각 사분면이 서로 다른 Mo/Si 다층막을 코팅* — 네 EUV 방출선에 맞추어진 협대역 반사기로 작동한다.

| Channel / 채널 | Center λ (nm) | FWHM (nm) | Peak refl. | Coating / 코팅 | Solar plasma diagnostic |
|---|---|---|---|---|---|
| He II | 30.7 | 3.0 | 23% | MoSi | chromosphere/transition region (~50,000 K) |
| Fe IX | 17.3 | 1.4 | 39% | MoSi | quiet corona (~1.0 MK) |
| Fe XII | 19.6 | 1.6 | 35% | MoSi | active region (~1.5 MK) |
| Fe XV | 28.5 | 1.9 | 15% | MoSi, variable spacing | flaring corona (~2.0 MK) |

The 28.4 nm coating uses *variable* Mo/Si layer spacing for optimum suppression of the much stronger nearby 30.4 nm He II line. Dual entrance filters (150 nm Al on a coarse Ni grid for 17.1/19.5; 150 nm Al-on-polyimide on fine mesh for 28.4/30.4) reject visible and IR light by >10¹³. A four-position filter wheel near the focal plane provides redundant rejection. EUVI is the only EUV instrument in SECCHI; the other four telescopes are all visible-light. CCD count rates predicted via CHIANTI are tabulated for quiet Sun (40–98 phot/s/pix), active region (118–976), and M-class flare (5540–101200), giving multi-decade dynamic range. **Active secondary mirror tip/tilt** — driven by the Guide Telescope at 250 Hz — provides image stabilization with ±7″ range and factor-3 jitter attenuation at 10 Hz.

28.4 nm 코팅은 인접한 강한 30.4 nm He II 선의 최적 억제를 위해 *가변* Mo/Si 두께를 사용한다. 입구 필터 두 종(17.1/19.5용은 거친 Ni grid 위 150 nm Al; 28.4/30.4용은 미세 mesh 위 폴리이미드 backed Al)은 가시광·IR을 10¹³배 이상 거부한다. 초점면 근처의 4-위치 필터 휠이 추가 거부를 제공. SECCHI 다섯 중 EUVI만 EUV이고 나머지 넷은 모두 가시광. CHIANTI로 예측한 CCD 계수율은 quiet Sun에서 40–98 phot/s/pix, 활동영역 118–976, M급 플레어 5540–101200 — 다중 자릿수 동적 영역. **능동 2차 거울 tip/tilt**는 Guide Telescope의 250 Hz 신호로 구동되어 ±7″ 범위에서 10 Hz에 factor-3의 jitter 감쇠를 제공.

### Part III: COR1 — Inner Coronagraph (§3) / COR1: 내부 코로나그래프

COR1 (§3, p. 76–83) is the **first space-borne internally-occulted *refractive* Lyot coronagraph**. This contrasts with LASCO/C1, which was internally-occulted but reflective. The optical train (Fig. 8): single radiation-hardened BK7-G18 objective lens (singlet, to keep scattering minimal), a cone-shaped occulter at the field-lens position (1.1 R☉ at perigee, 1.3 R☉ in observed image), a Polarcor rotating polarizer between two doublets (positive achromat near the Lyot stop, negative achromat downstream), a 22.5 nm-wide bandpass at 656 nm Hα. FOV 1.4–4 R☉ (vignetted to 1.64 R☉ inner edge by the focal-plane mask). Pixel size 3.75″ full-resolution, 7.5″ in 2×2 binning. Photometric response 7.1×10⁻¹¹ B☉/DN (Ahead). Three polarization images (-60°, 0°, +60°) are taken in 11 s; cadence 8 minutes.

COR1(§3, p. 76–83)은 **세계 최초의 우주용 내부차폐 *굴절(refractive)* Lyot 코로나그래프**이다. 내부차폐였지만 반사식이었던 LASCO/C1과 대비된다. 광학 구성(Fig. 8): 방사선 강화 BK7-G18 단일 대물렌즈(scattering 최소화를 위해 singlet), field lens 위치에 원뿔 형태의 차폐기(궤도 근일점 1.1 R☉, 영상에서 1.3 R☉), 두 doublet 사이의 Polarcor 회전 편광기, 656 nm Hα에 22.5 nm 대역폭의 협대역 필터. FOV 1.4–4 R☉ (focal-plane mask에 의해 1.64 R☉ 내측까지 vignetting). 픽셀 크기 3.75″ 전해상도, 2×2 비닝 시 7.5″. 측광 반응 7.1×10⁻¹¹ B☉/DN(Ahead). 세 편광각(-60°, 0°, +60°) 영상을 11 s 안에 촬영, 카덴스 8 분.

The signal will be dominated by instrumentally scattered light (the scattered photospheric continuum cannot be fully eliminated by Lyot principles since it is largely *unpolarized*). Therefore COR1 *requires* polarization observations to extract the polarized brightness pB. The Gibson 1973 K-corona model used to predict signal-to-noise:

신호는 기기적 산란광에 의해 지배되며 — Lyot 원리로는 완전 제거가 불가하다(산란된 광구 연속광은 대부분 *비편광*). 따라서 COR1은 K-corona의 편광 휘도 pB를 추출하기 위해 *반드시* 편광 관측을 필요로 한다. 신호대잡음 예측에 쓰이는 Gibson 1973 K-corona 모델:

$$
\log_{10}(pB) = -2.65682 - 3.55169\,(R/R_\odot) + 0.459870\,(R/R_\odot)^2
$$

Predicted SNR for 1 s exposure with 2×2 binning is ~50 at 1.5 R☉, dropping to ~1 at 4 R☉ (Fig. 12). Worst-case stray light measured in vacuum tank at NRL is below 10⁻⁶ B☉ on average, with discrete ring features up to 1.4×10⁻⁶ B☉ traced to features on the front lens surface.

1 s 노출, 2×2 비닝 기준 예측 SNR은 1.5 R☉에서 ~50, 4 R☉에서 ~1로 떨어진다(Fig. 12). NRL 진공 탱크에서 측정된 worst-case 미광은 평균 10⁻⁶ B☉ 이하, 일부 ring 모양 특징은 1.4×10⁻⁶ B☉까지 — 이는 대물렌즈 전면의 결함으로 추적됨.

### Part IV: COR2 — Outer Coronagraph (§4) / COR2: 외부 코로나그래프

COR2 (§4, p. 83–88) is an externally-occulted Lyot coronagraph in the LASCO/C2-C3 lineage but redesigned for higher throughput. The external occulter is a three-disk system that creates a deep diffraction shadow at the objective (A1 aperture). The internal occulter blocks the brightly illuminated edge of the third disk, while the Lyot stop blocks the bright A1 edge. Three lens groups (O1: objective, O2: collimator, O3: imaging), spectral filter (650–750 nm), Polarcor polarizer. Pupil at A1 has 34 mm aperture (vs. 20 mm on LASCO/C2 and 9 mm on C3) and f/5.6 — providing both higher light-gathering power and tighter spatial resolution. FOV 2.5–15 R☉ (4° half-angle). Pixel size 14.7″. Photometric response 1.35×10⁻¹² B☉/DN (Ahead). Polarization sequence in <15 s; nominal cadence 15 min.

COR2(§4, p. 83–88)는 LASCO/C2-C3 계보의 외부차폐 Lyot 코로나그래프지만 더 높은 throughput으로 재설계되었다. 외부차폐는 3-디스크 시스템으로 대물(A1) 구경에서 깊은 회절 그림자를 만든다. 내부 차폐기는 세 번째 디스크의 밝게 조명된 가장자리 영상을 가리고, Lyot stop은 A1의 밝은 가장자리를 가린다. 세 렌즈군(O1: 대물, O2: 시준, O3: 결상), 분광 필터(650–750 nm), Polarcor 편광기. A1 동공 구경 34 mm(LASCO/C2의 20 mm, C3의 9 mm 대비)이고 f/5.6 — 더 높은 집광력과 더 정밀한 공간 분해능을 동시 제공. FOV 2.5–15 R☉(반각 4°). 픽셀 14.7″. 측광 반응 1.35×10⁻¹² B☉/DN(Ahead). 편광 시퀀스 <15 s; 카덴스 15 분.

The "polarization sequence in <15 s" is critical: a moderately fast CME at 750 km/s would traverse one COR2 pixel (15″) in ~15 s, so the three polarization frames must be acquired faster than the CME's pixel-crossing time to avoid smearing the K-corona signal. Stray-light measurements show COR2 vacuum SL is ~1 order of magnitude *below* the K-corona model — exceeding requirements. A "double-exposure" mode (two images at 0° and 90° read out without intermediate clear) sums to total brightness in a single downlinked image, used for telemetry-economy and for the space-weather beacon channel.

"15 s 이내 편광 시퀀스"는 결정적이다: 평균 속도 750 km/s의 CME는 COR2 한 픽셀(15″)을 ~15 s에 횡단하므로, 세 편광 프레임은 K-corona 신호의 번짐을 막기 위해 그 시간보다 빠르게 획득해야 한다. 미광 측정 결과 COR2의 진공 SL은 K-corona 모델 대비 약 1자리수 *낮음* — 요구사항을 초과 달성. "double-exposure" 모드(0°와 90° 두 영상을 중간 readout 없이 촬영)는 한 다운링크 영상으로 total brightness를 합산하여 텔레메트리 절약 및 우주기상 beacon 채널에 사용.

### Part V: HI — Heliospheric Imagers (§5) / HI: 헬리오스피어 영상기

HI (§5, p. 88–93) is the *most novel* element of SECCHI: shutterless, baffle-only, wide-field cameras that make K-corona imaging possible 14°–88° from the Sun without an occulter. Two telescopes per spacecraft are mounted on the Earth-facing side:

HI(§5, p. 88–93)는 SECCHI의 *가장 새로운* 요소이다: 셔터 없이, 차폐기 없이, 배플만으로 14°–88° 신연각(elongation)에서 K-corona 영상화를 가능케 한 광시야 카메라. 우주선당 두 망원경이 지구 방향 면에 장착:

| Property / 항목 | HI-1 | HI-2 |
|---|---|---|
| FOV centre | 13.98° | 53.68° |
| Angular FOV | 20° | 70° |
| Elongation range | 3.98–23.98° (15–84 R☉) | 18.68–88.68° (66–318 R☉) |
| Image scale | 70″/pix (2×2 binned 1024²) | 4 arcmin/pix |
| Bandpass | 630–730 nm | 400–1000 nm |
| Exposure | 12–20 s | 60–90 s |
| # exposures summed | ~150 | ~100 |
| Cadence | 60 min | 120 min |
| Brightness sensitivity | 3×10⁻¹⁵ B☉ | 3×10⁻¹⁶ B☉ |

The CME signal is ~3×10⁻¹⁵ B☉ — buried 13 orders of magnitude below the solar disk and 2 orders below the F-corona (zodiacal) and K-corona backgrounds. Three tiers of baffles handle the rejection:

CME 신호는 ~3×10⁻¹⁵ B☉ 수준 — 태양 원반보다 13자리수 어둡고 F-corona(황도광)·K-corona 배경보다도 2자리수 어둡다. 세 단계 배플이 거부를 담당:

1. **Forward baffle (knife-edge cascade)**: Five vanes optimized so that the (n+1)th vane lies in the shadow of the (n-1)th vane. Each vane behaves like a knife-edge with Fresnel-Kirchhoff diffraction profile. The cascade composes successive rejections to yield ~10⁻¹³ B☉ rejection at HI-1's limb-side edge. Measured rejection in vacuum and ambient agrees with theoretical prediction over 12 orders of magnitude (Fig. 18).
2. **Perimeter baffle**: side and rear vanes that block stray sunlight reflected from the spacecraft body (high-gain antenna, monopole antennae, etc.).
3. **Internal baffle**: layers within the optical box that suppress multiple reflections, mainly from stars, planets, Earth, and the SWAVES monopole antenna.

1. **전방 배플(knife-edge cascade)**: (n+1)번째 vane이 (n-1)번째 vane의 그림자에 놓이도록 최적화된 5장 vane. 각 vane은 Fresnel-Kirchhoff 회절 프로파일을 가진 칼날(knife-edge) 역할. 카스케이드는 연속된 거부를 곱셈적으로 합성하여 HI-1 limb 쪽 가장자리에서 ~10⁻¹³ B☉ 거부 달성. 진공·상온 측정값이 이론 예측과 12자리수에 걸쳐 일치(Fig. 18).
2. **둘레 배플(perimeter baffle)**: 우주선 본체(고이득 안테나, monopole 안테나 등)에서 반사된 미광을 차단하는 측면·후면 vane.
3. **내부 배플(internal baffle)**: 별·행성·지구·SWAVES monopole에서의 다중 반사를 억제.

The shutterless operation means each frame is a smear of the static scene plus the (much shorter) image readout transient. Multiple short exposures (up to ~1 minute each) are summed *onboard* — typically 50 frames — after **two-stage cosmic-ray scrubbing** (first a photon-noise compatibility test against the previous frame; second a 3-frame median filter) and 2×2 binning to 1024². The combination yields ~14× SNR over a single exposure. The 32-bit image buffer receives each 1024² frame; the final summed buffer is downlinked using Rice lossless compression (factor 2.5×).

무셔터 운용은 각 프레임이 정적 장면의 번짐 + (훨씬 짧은) readout transient의 합임을 의미한다. 짧은 노출(최대 1분 각각)을 *온보드*에서 합산 — 보통 50장 — 단, **2단계 우주선(cosmic-ray) scrubbing** (1단계: 이전 프레임 대비 photon-noise 호환성 검사; 2단계: 3-프레임 median filter) 후 2×2 비닝으로 1024². 단일 노출 대비 ~14× SNR. 32비트 영상 버퍼가 각 1024² 프레임을 받고, 최종 합산 버퍼는 Rice 무손실 압축(2.5×)으로 다운링크.

### Part VI: Guide Telescope, SCIP, & Mechanisms (§6–§8) / 가이드 망원경·SCIP·기구

The Guide Telescope (§6, p. 93–94) is an achromatic refractor with 27 mm aperture, 1454 mm focal length (Ahead) / 1562 mm (Behind), 50 nm-FWHM bandpass at 570 nm. It images the Sun onto an occulter sized to block most of the disk and pass only the limb. Four photodiodes (90° apart, 4 redundant) sense limb intensity in four sectors. The signals — sampled at 4 ms by a 12-bit ADC in the SEB — generate four sun-presence flags and pitch/yaw error signals. These error signals drive the EUVI active secondary mirror (SECCHI's *Fine Pointing System*) at 250 Hz. The GT is also the spacecraft's fine sun sensor: when all four diodes are illuminated the four flags are set; otherwise the spacecraft Attitude Control System uses the flags to acquire the Sun.

가이드 망원경(§6, p. 93–94)은 27 mm 구경, 초점거리 1454 mm(Ahead)/1562 mm(Behind)의 무색수차 굴절기, 570 nm에서 50 nm-FWHM 대역. 태양상을 occulter(원반 대부분을 가리고 limb만 통과시킴) 위에 결상한다. 4개 photodiode(90° 간격, 4개 중복)가 네 sector의 limb 밝기를 감지. SEB의 12비트 ADC가 4 ms마다 sampling하여 네 sun-presence flag와 pitch/yaw 오류 신호 생성. 이 오류 신호가 EUVI 능동 2차 거울(SECCHI의 *Fine Pointing System*)을 250 Hz로 구동. GT는 우주선의 fine sun sensor 역할도 한다.

The SCIP bench (§7) is a 6.3 cm-thick aluminum honeycomb panel (112×70 cm) with high-modulus graphite/cyanate-ester face sheets. Telescopes are mounted via three titanium flexure mounts each; first natural frequency measured at 54 Hz (requirement: 50 Hz). Three flexures give a "near-kinematic" mount: stiff in the constrained directions, much less so in the released DOFs, isolating the optics from thermal stresses. Mechanisms (§8) total 10 of 6 distinct designs: SESAMEs (re-closable doors, three per SCIP), shutters (heritage from SOHO/MDI, TRACE, GOES-N/SXI; 40 ms–67 s exposures repeatable to 15 μs), EUVI quadrant selector (90° rotation in 45 ms), EUVI filter wheel (300 ms), and the COR polarizer "hollow-core motor" (HCM) (144 steps, 2.5° increments, repeatability 30″, 120° rotation in 400 ms). All meet 3× design lifetime in qualification.

SCIP 벤치(§7)는 6.3 cm 두께 알루미늄 honeycomb 패널(112×70 cm), 고탄성률 흑연/시아네이트 에스테르 face sheet. 각 망원경은 티타늄 flexure 3개로 마운트; 1차 고유진동수 측정값 54 Hz(요구 50 Hz). 3-flexure는 "준-운동학적(near-kinematic)" 마운트로 — 구속 방향엔 단단하고 해제된 DOF엔 부드러워 광학을 열응력에서 분리. 기구(§8)는 6 가지 설계의 10개: SESAMEs(재폐쇄 가능 도어, SCIP당 셋), shutter(SOHO/MDI·TRACE·GOES-N/SXI 유산; 40 ms–67 s, 15 μs 정밀도), EUVI 사분면 selector(45 ms 안에 90° 회전), EUVI 필터 휠(300 ms), COR polarizer "hollow-core motor"(HCM)(144 스텝, 2.5° 증분, 30″ 반복도, 400 ms 안에 120° 회전). 모두 인증에서 3× 설계 수명 충족.

### Part VII: Electronics, CCDs, & Flight Software (§9–§11) / 전자장치·CCD·비행 소프트웨어

The SEB (§9, p. 99–101) is built around a custom 5-slot 6U cPCI backplane and houses six cards: PSIB (power), PIB (power conversion/distribution), HKP (housekeeping), SpaceWire SWIC (image data), 1553 (spacecraft bus), and the **RAD750** processor card (radiation-hardened PowerPC at 116 MHz, 256 kB EEPROM, 128 MB SDRAM with EDAC scrubbing, ~120 MIPS). The SWIC has two 100 Mbit/s SpaceWire links — one to SCIP cameras, one to HI cameras — with 256 MB SDRAM buffer. The Mechanism Electronics Box (MEB) is a separate small box (210×56×171 mm, 1.4 kg) on the underside of the SCIP bench, controlling all SCIP mechanisms via two FPGAs. Camera electronics (CEBs) are remote: one CEB per package (one for SCIP's three CCDs, one for HI's two CCDs), each containing 14-bit ADCs and CDS implemented as RAL CDS/ADC ASICs.

SEB(§9, p. 99–101)는 맞춤형 5-슬롯 6U cPCI 백플레인 위에 여섯 카드를 담는다: PSIB(전원), PIB(전력 변환·분배), HKP(housekeeping), SpaceWire SWIC(영상 데이터), 1553(우주선 버스), **RAD750** 프로세서 카드(방사선 강화 PowerPC 116 MHz, 256 kB EEPROM, EDAC scrubbing 128 MB SDRAM, ~120 MIPS). SWIC는 두 100 Mbit/s SpaceWire 링크(하나는 SCIP, 하나는 HI 카메라)에 256 MB SDRAM 버퍼. 기구 전자상자(MEB)는 SCIP 벤치 하부의 별도 작은 상자(210×56×171 mm, 1.4 kg)로 SCIP 모든 기구를 두 FPGA로 제어. 카메라 전자장치(CEB)는 원격: 패키지당 1대(SCIP 3 CCD용, HI 2 CCD용), 각각 14비트 ADC와 CDS를 RAL CDS/ADC ASIC으로 구현.

The CCDs (§10) are the e2v CCD42-40 — back-illuminated, non-inverted, three-phase, 2048×2052 image area + 100 reference columns — operated at –65°C via passive radiator + cold finger (FPA in §10.2). Quantum efficiency: 80% at 500 nm, 88% at 650 nm, 64% at 800 nm, 34% at 900 nm for the visible-light CCDs (COR1/2, HI-1/2, all anti-reflection coated); the EUVI CCD has *no AR coating* (uncoated backside) yielding QE 74% at 17.1 nm, 70% at 30.3 nm, 46% at 58.4 nm. Charge transfer efficiency >99.999%. Dark current <2 nA/cm²/s at 20°C (24,000 e⁻/pix/s — extreme cooling required).

CCD(§10)는 e2v CCD42-40 — backside illuminated, non-inverted, 3-phase, 2048×2052 imaging + 100열 reference — passive radiator + cold finger 통한 –65°C 운용(§10.2 FPA). 양자 효율: 가시광 CCD(COR1/2, HI-1/2, AR 코팅 모두)에서 500 nm 80%, 650 nm 88%, 800 nm 64%, 900 nm 34%; EUVI CCD는 *AR 코팅 없음*(코팅 없는 후면)으로 17.1 nm에서 74%, 30.3 nm에서 70%, 58.4 nm에서 46%. CTE >99.999%. 암전류 20°C에서 <2 nA/cm²/s(24,000 e⁻/pix/s — 극저온 냉각 필수).

The flight software (§11) runs under VxWorks RTOS on the RAD750. It contains ~250,000 lines of code, derived from SMEX (Small Explorer) heritage and SOHO/LASCO. It performs telemetry, command handling, image scheduling, image processing (120 functions in 100 rows of an *image processing table* — modifiable in flight), and the four compression modes:

비행 소프트웨어(§11)는 RAD750에서 VxWorks RTOS로 구동. ~250,000 lines of code, SMEX 시리즈와 SOHO/LASCO 코드 유산. 텔레메트리·명령·영상 스케줄링·영상 처리(120 함수, 100행의 *image processing table* — 비행 중 수정 가능) 수행, 그리고 네 가지 압축 모드:

| Compression | Type | Factor | Used by |
|---|---|---|---|
| **None** | none | 1× | calibration |
| **Rice** | lossless | ~2.2× | LASCO, EIT heritage; HI summed images |
| **H-Compress** | lossy wavelet | variable | LASCO, EIT heritage |
| **ICER** | lossy wavelet | up to 200× | new — used on Mars Exploration Rovers; user can specify output size |

For HI, the long required exposure (~30 min) is achieved by summing many shorter exposures (≤1 min) onboard to suppress dark current. For each summed image, the cosmic-ray-scrubbed and 2×2-binned 1024² image is added to a 32-bit accumulator buffer; the final buffer is downlinked. A full uncompressed SECCHI image is 8 MB (2048×2048, 14 bit packed); at 20× compression a full image downlinks in ~24 s.

HI는 ~30 분에 달하는 요구 노출을 ≤1 분 짧은 노출 다수의 온보드 합산으로 달성(암전류 억제). 합산 영상마다 cosmic-ray scrubbed + 2×2 비닝된 1024² 영상을 32비트 누산 버퍼에 더하고, 최종 버퍼를 다운링크. 무압축 풀 SECCHI 영상은 8 MB(2048², 14비트 packed); 20× 압축 시 한 영상 다운링크에 ~24 s.

### Part VIIb: Detailed Worked Example — How a CME is captured / 사례: CME가 어떻게 포착되는가

Consider an Earth-directed CME launching from the western limb at 800 km/s on day D. The full SECCHI observation chain unfolds as follows (numbers from the paper's tables):

지구 방향 CME가 D일에 서쪽 limb에서 800 km/s로 발사된다고 하자. 표의 수치를 사용한 SECCHI 관측 체인은 다음과 같다:

1. **t = 0 to +5 min (EUVI)**: The pre-eruption sigmoid and post-eruption flare ribbon are imaged in the four EUV channels at 4-min cadence (synoptic schedule, Table 10) — quiet-Sun rates 40–98 phot/s/pix jump to flare rates 5540–101200 phot/s/pix in the 19.5 nm Fe XII channel. The Guide Telescope 250 Hz signal stabilizes the active secondary so a 1 s exposure yields stable imagery. The EUVI 2k×2k 19.5 nm image at full resolution downlinks at 20 min cadence (1691 Mbits/day budget).
2. **t = +5 to +30 min (COR1)**: The CME front clears 1.4 R☉ and enters COR1's FOV. Three polarization images (-60°, 0°, +60°) acquired in 11 s, repeated every 8 min. pB extracted on the ground; SNR ~50 at 1.5 R☉ falls to ~10 at 3 R☉. CME visible as a bright loop expanding outward.
3. **t = +30 min to +6 hr (COR2)**: At 800 km/s the CME reaches 4 R☉ in ~1 hr and 15 R☉ in ~4–6 hr. COR2 takes polarization triplets every 15 min (pB requires <15 s for the 3 frames so motion < 1 pixel). At 800 km/s a 15 s gap would yield 0.8-pixel motion at 15″/pixel — comfortably within the design.
4. **t = +6 to +20 hr (HI-1)**: CME enters HI-1 FOV at ~15 R☉ (4° elongation). 50 short exposures of 12–20 s summed onboard over ~60 min cadence yield SNR ~14× single-exposure. The CME appears as a faint front against the F-corona zodiacal background, visible at ~3×10⁻¹⁵ B☉.
5. **t = +1 to +4 days (HI-2)**: CME crosses the 24° HI-1/HI-2 boundary at ~84 R☉ and enters HI-2's 19–89° FOV. Cadence drops to 120 min, exposures 60–90 s. CME sweeps past Earth at ~80° elongation. From STEREO-A's vantage with B (a year later, ~22° offset), the parallax allows GCS-model 3-D fit yielding true direction and speed.
6. **At each step**: Beacon channel (504 bps) sends low-resolution snapshots in real-time; primary 24-hr downlink replays full-fidelity ICER-compressed FITS via SSR1; campaign cadences may be running on SSR2.

1. **t = 0 ~ +5 분 (EUVI)**: 분출 전 sigmoid와 분출 후 flare ribbon이 4 EUV 채널에서 4 분 카덴스로 영상화. Quiet Sun 40–98 phot/s/pix가 19.5 nm Fe XII 채널에서 플레어 5540–101200 phot/s/pix로 점프. GT 250 Hz 신호가 능동 2차 거울을 안정화하여 1 s 노출 영상 안정. 19.5 nm 풀 해상도 영상은 20 분 카덴스로 다운링크(1691 Mbits/일).
2. **t = +5 ~ +30 분 (COR1)**: CME 전면이 1.4 R☉를 지나 COR1 시야 진입. 11 s에 세 편광 영상 획득, 8 분마다 반복. 지상에서 pB 추출 — 1.5 R☉에서 SNR ~50, 3 R☉에서 ~10.
3. **t = +30 분 ~ +6 시간 (COR2)**: 800 km/s에서 4 R☉ 약 1 시간, 15 R☉ 약 4–6 시간. COR2가 15 분마다 편광 triplet 촬영(3 프레임 < 15 s 요구 — 800 km/s 시 15 s에 0.8 픽셀, 15″/픽셀로 설계 한계 내).
4. **t = +6 ~ +20 시간 (HI-1)**: CME가 ~15 R☉(4° 신연각)에서 HI-1 진입. 50회 12–20 s 노출의 온보드 합산으로 60 분 카덴스, SNR ~14× single-exposure. F-corona 배경 위에서 ~3×10⁻¹⁵ B☉ 신호.
5. **t = +1 ~ +4 일 (HI-2)**: CME가 24° HI-1/HI-2 경계(~84 R☉)를 넘어 HI-2의 19–89° 시야 진입. 카덴스 120 분, 노출 60–90 s. CME가 ~80° 신연각에서 지구 통과. STEREO-B의 시점과 결합하여 GCS 모델 3D fit으로 진짜 방향·속도 산출.
6. **각 단계에서**: Beacon 채널(504 bps)이 실시간 저해상도 스냅샷 송출; 주 24시간 다운링크가 SSR1을 통해 전체 fidelity ICER 압축 FITS 재생; campaign 카덴스가 SSR2에서 가동 중일 수 있음.

This walkthrough captures *why* SECCHI needed five telescopes: signal levels span 13 orders of magnitude from disk to HI-2, dynamic ranges span 3 orders within EUVI alone, and the time between pre-eruption signature and Earth arrival spans 4 days — covered seamlessly by the chained FOVs.

이 walkthrough는 SECCHI가 *왜* 다섯 망원경을 필요로 했는지를 포착한다: 신호 수준이 원반에서 HI-2까지 13자리, EUVI 한 채널 안에서도 3자리의 동적 영역, 분출 전 신호에서 지구 도달까지 4일 — 모두 체인된 FOV로 끊김없이 커버.

### Part VIII: Concept of Operations & Data Policy (§12) / 운용 컨셉과 데이터 정책

CONOPS (§12) describes a flexible, two-program model: a "synoptic" program (~80% of telemetry, identical on both spacecraft, time-tagged identically — for stereoscopy) and "campaign" programs (higher-cadence, possibly asymmetric, written to the SSR2 overwriting buffer). Sample synoptic daily volume (Table 10): EUVI 4-min cadence + 20-min full-resolution = 4386 Mbits/day total at compression factors 10× (COR/EUVI) to 2.5× (HI). This was sustained against allocations 54 → 50 → 45 kbps (declining with mission age as the spacecraft drift increases distance and decreases downlink bit rate).

CONOPS(§12)는 유연한 2-프로그램 모델을 설명: "synoptic" 프로그램(텔레메트리 ~80%, 양 위성 동일, 동일 time-tag — 입체영상용)과 "campaign" 프로그램(고카덴스, 비대칭 가능, SSR2 overwriting 버퍼 사용). 표 10의 일일 synoptic 부피: EUVI 4 분 카덴스 + 20 분 전해상도 = 압축률 10×(COR/EUVI)–2.5×(HI)로 총 4386 Mbits/day. 할당량 54→50→45 kbps(우주선이 멀어지며 다운링크 비트레이트 감소)에 맞춰 유지.

**Beacon mode** broadcasts 504 bps continuously — heavily compressed and binned subset images for real-time space-weather monitoring. **Campaigns** add a second daily downlink track (12 hr offset), doubling SECCHI's daily bandwidth for 2 × 2-week intervals during the mission. Data policy is fully open: Level-0.5 FITS files are publicly archived within 30 minutes of all-image-data receipt; calibration data and procedures are also public; querying is by FITS header keyword via a searchable database. Higher-level products (calibrated brightness, polarized brightness, Carrington maps, CME catalog, comet/star tables) are generated routinely at the Data Processing Facility (DPF) at NRL and at the SECCHI Science Center (SSC).

**Beacon 모드**는 504 bps로 연속 송출 — 강하게 압축·비닝된 부분집합 영상으로 실시간 우주기상 모니터링. **Campaign**은 12시간 offset의 두 번째 일일 다운링크 트랙으로 SECCHI 일일 대역폭을 임무 중 2 × 2주간 두 배로 확장. 데이터 정책은 완전 개방: Level-0.5 FITS 파일은 모든 영상 데이터 수신 후 30 분 내 공개 아카이브; 교정 데이터·절차도 공개; FITS 헤더 키워드 기반 검색 가능한 DB로 쿼리. 상위 산출물(calibrated brightness, pB, Carrington 맵, CME 카탈로그, 혜성/별 표)은 NRL의 Data Processing Facility(DPF)와 SECCHI Science Center(SSC)에서 정상 생성.

---

## 3. Key Takeaways / 핵심 시사점

1. **Five telescopes, one continuous radial coverage.** SECCHI partitions the entire Sun-to-Earth radial range (1–318 R☉) across five telescopes whose fields of view *deliberately overlap* by 1–2 R☉ at each junction (EUVI/COR1 at 1.4–1.7, COR1/COR2 at 2.5–4, COR2/HI-1 at 15, HI-1/HI-2 at ~24°). This guarantees that no CME is "lost" in transit and that calibration crosschecks are always available between adjacent telescopes. / **다섯 망원경, 하나의 연속 반경 커버.** SECCHI는 Sun–Earth 전 반경(1–318 R☉)을 다섯 망원경에 분할하고 각 경계에서 시야가 1–2 R☉씩 *의도적으로 겹치게* 설계(EUVI/COR1 1.4–1.7, COR1/COR2 2.5–4, COR2/HI-1 15, HI-1/HI-2 ~24°). 어떤 CME도 전파 중 "잃어버려지지" 않으며 인접 망원경 간 교정 cross-check이 항상 가능.

2. **Stereoscopy emerges from spacecraft kinematics, not from instrument design.** SECCHI itself is *not* a stereo instrument — the two SECCHI packages on the two STEREO spacecraft are *identical*. The stereo information comes from the slowly diverging baseline (~22°/yr away from Earth in opposite directions), and from the fact that both spacecraft *sample the same time* with synchronized synoptic schedules. This decouples engineering from science: the same hardware design supports stereoscopy by simple duplication. / **입체영상은 기기 설계가 아니라 우주선 궤도에서 나온다.** SECCHI 자체는 스테레오 기기가 *아니다* — 두 SECCHI 패키지는 *동일*하다. 입체 정보는 양 위성이 매년 ~22°씩 반대 방향으로 지구로부터 멀어지면서 생기는 기저선과 동기화된 synoptic 스케줄로 같은 시각을 sampling한다는 사실에서 나온다. 즉 동일한 하드웨어를 단순 복제하여 스테레오를 지원.

3. **Polarization is the key to extracting CME signal from background.** All three coronagraph telescopes (COR1, COR2, optionally HI) take polarization triplets. Since instrumental scattered light is largely *unpolarized*, the polarized brightness pB cleanly extracts the K-corona electron-Thomson scattering signal. This is why the polarizer (Polarcor in HCM) is one of SECCHI's most exotic mechanisms — 144 steps, 30″ repeatability, qualified to 3.5 million operations. / **편광은 배경에서 CME 신호를 뽑아내는 열쇠.** 세 코로나그래프(COR1, COR2)는 모두 편광 triplet을 촬영. 기기적 산란광이 대부분 *비편광*이기 때문에 편광 휘도 pB는 K-corona의 전자 Thomson 산란 신호만 깔끔히 추출. 그래서 Polarcor in HCM은 SECCHI에서 가장 이국적인 기구 중 하나 — 144 스텝, 30″ 반복도, 350만 회 자격.

4. **Shutterless wide-field imaging — the HI breakthrough.** The HI cameras have *no shutter*, only a one-shot launch door. Each frame inherently smears the static scene during readout, but onboard summation of 50 short exposures + 2×2 binning + 2-stage cosmic-ray scrubbing yields a CME-detection-grade image without any moving parts in the optical path. The 5-vane Fresnel knife-edge cascade alone provides ~10⁻¹³ B☉ rejection — a factor of 10¹³ between the solar disk and the HI signal. / **무셔터 광시야 영상 — HI의 돌파구.** HI 카메라는 *셔터 없이* 발사 도어만 있다. 각 프레임은 readout 동안 정적 장면을 본질적으로 번지게 하지만, 50회 짧은 노출의 온보드 합산 + 2×2 비닝 + 2단계 cosmic-ray scrubbing으로 광학 경로에 가동부 없이도 CME 검출 수준의 영상을 얻는다. 5-vane Fresnel knife-edge cascade만으로 ~10⁻¹³ B☉ 거부 — 태양 원반과 HI 신호 사이의 10¹³ factor.

5. **Onboard image processing is treated as flight software, not pipeline.** The 120 image-processing functions (cosmic-ray scrubbing, summing, binning, ROI/occulter masks, four compression algorithms including ICER's user-specifiable lossy wavelet) all run on the RAD750 in SECCHI's 250,000-line flight code. The image-processing *table* — 100 rows × functions — is uplinkable in flight, allowing the science team to redefine processing chains years into the mission. This pattern (compute-in-orbit) is now standard for high-data-rate space science. / **온보드 영상 처리를 비행 소프트웨어로 취급(파이프라인이 아니라).** 120 영상 처리 함수(cosmic-ray scrubbing, 합산, binning, ROI/occulter 마스크, ICER 등 4가지 압축)가 SECCHI의 250,000-line 비행 코드 안에서 RAD750 위에 모두 실행. 100행 × 함수의 *영상 처리 테이블*은 비행 중 업링크 가능 — 과학팀이 임무 수년 후에도 처리 체인을 재정의 가능. 이 "궤도-내 연산" 패턴은 현재 고데이터율 우주과학의 표준.

6. **Real-time space-weather forecasting was designed in from day one.** The 504 bps "beacon" channel runs continuously, delivering a heavily compressed/binned image subset to ground stations for real-time monitoring — independent of the daily 24-hour synoptic downlink. This is the architectural origin of operational space-weather products that NOAA SWPC now consumes, and motivates the dual SSR partitioning (SSR1 80% science, SSR2 20% campaigns/space weather). / **실시간 우주기상 예보를 임무 설계 시점부터 통합.** 504 bps "beacon" 채널은 계속 가동되어 강한 압축·비닝된 영상 부분집합을 지상국으로 송출 — 24시간 synoptic 다운링크와 독립적. NOAA SWPC가 현재 운용 중인 우주기상 제품의 아키텍처적 기원이며, 이중 SSR 분할(SSR1 80% 과학, SSR2 20% campaign·우주기상)의 동기.

7. **EUV multilayer mirrors with quadrant-tuned coatings push EUV imaging to four wavelengths simultaneously.** EUVI's RC primary and secondary are each divided into 4 optical quadrants, each with a *different* Mo/Si multilayer optimized for one EUV emission line (17.1, 19.5, 28.4, 30.4 nm). Switching channels is by rotating a 4-position quadrant selector (90° in 45 ms). This single-instrument, four-channel design is far cheaper and lighter than four separate EUV telescopes — a design strategy that influenced SDO/AIA's later 7-channel architecture (though AIA used 7 separate telescopes due to even tighter requirements). / **사분면 코팅 EUV 다층막 거울로 4 파장 동시 영상.** EUVI의 RC 1차·2차 거울은 각각 4 광학 사분면으로 나뉘고 각 사분면이 *다른* Mo/Si 다층막(17.1, 19.5, 28.4, 30.4 nm 최적)을 가진다. 채널 전환은 4-위치 사분면 selector 회전(45 ms 안에 90°)으로. 이 단일 기기 4채널 설계는 4개의 별도 EUV 망원경보다 훨씬 가볍고 저렴 — SDO/AIA의 후속 7채널 아키텍처에 영향(다만 AIA는 더 엄격한 요구로 7개 별도 망원경 채택).

8. **Co-alignment via near-kinematic Ti flexures preserves stereoscopy through thermal cycles.** SCIP houses four telescopes (EUVI, COR1, COR2, GT) on a single 6.3 cm composite bench but each telescope sits on three titanium flexures that are stiff axially but compliant transversely. This "near-kinematic" mounting allows shimming co-alignment to 5″ (the angular precision required for stereoscopic feature matching at 1 AU) while isolating each telescope from thermal-induced bench distortions. The choice of flexure material (Ti) and geometry (thin) is a direct response to the tight stereoscopy budget. / **준-운동학적 Ti flexure 공정 정렬로 열주기 동안 입체 정렬 유지.** SCIP는 6.3 cm 복합재 벤치 한 장에 네 망원경(EUVI, COR1, COR2, GT)을 탑재하되 각각 축 방향엔 단단하고 횡방향엔 유연한 티타늄 flexure 3개로 마운트. 이 "준-운동학적" 방식은 5″의 shimming 정렬도(1 AU 거리 입체 매칭에 필요)를 유지하면서 각 망원경을 벤치의 열변형에서 분리. flexure의 재료(Ti)와 형상(얇음)은 입체 예산에 대한 직접 대응.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 K-corona polarized brightness model (Gibson 1973)

$$
\log_{10}(pB) = -2.65682 - 3.55169\,(R/R_\odot) + 0.459870\,(R/R_\odot)^2
$$

- $pB$: polarized brightness in units of mean solar brightness (B☉)
- $R/R_\odot$: heliocentric distance in solar radii
- Validity: 1.4 ≤ R/R☉ ≤ 4 (the COR1 field)
- Used to predict COR1 SNR for 1 s exposures with 2×2 binning (Fig. 12)

태양 평균 휘도 단위의 K-corona 편광 휘도를 1.4–4 R☉에서 거리의 다항식으로 근사. COR1의 1 s 노출 + 2×2 binning SNR 예측에 사용. / Polynomial in heliocentric distance over 1.4–4 R☉, used for COR1 SNR forecasting.

### 4.2 Polarized brightness extraction from triplet

For three exposures at polarizer angles $\theta = -60°, 0°, +60°$:

$$
pB \;=\; \tfrac{2}{3}\sqrt{(I_{0}-I_{60})^{2}+(I_{0}-I_{-60})^{2}+(I_{60}-I_{-60})^{2}}
$$

Total brightness:
$$
B_{\rm tot} = \tfrac{2}{3}(I_{-60}+I_0+I_{60})
$$

Unpolarized component (F-corona + scattered light):
$$
B_{\rm unpol} = B_{\rm tot} - pB
$$

세 편광각 영상의 차이로 K-corona의 편광 신호를 추출. F-corona는 비편광이라 상쇄. / Differences between polarization images extract K-corona; the unpolarized F-corona cancels.

### 4.3 EUVI count rate prediction

$$
N(\lambda) = \int A_{\rm eff}(\lambda)\, F_\odot(\lambda; T)\, d\lambda
$$

where
$$
A_{\rm eff}(\lambda) = A_{\rm aperture}\times R_1(\lambda)\,R_2(\lambda)\,T_{\rm front}(\lambda)\,T_{\rm rear}(\lambda)\,\eta_{\rm CCD}(\lambda)
$$

- $A_{\rm aperture}$: clear primary aperture area
- $R_{1,2}$: reflectivity of primary and secondary (Mo/Si pair)
- $T_{\rm front, rear}$: filter transmissions
- $\eta_{\rm CCD}$: CCD quantum efficiency
- $F_\odot$: solar photon flux from CHIANTI given DEM and temperature T

Predicted rates for quiet Sun at 19.5 nm: 40–41 phot/s/pix. M-class flare at 19.5 nm: 92,200–101,200 phot/s/pix — a dynamic range of >3 decades within a single channel. / 19.5 nm에서 quiet Sun 40–41 phot/s/pix, M 플레어 92,200–101,200 — 단일 채널 내 3자리 이상의 동적 영역.

### 4.4 SNR for summed shutterless exposures (HI)

For $N$ summed individual exposures with single-frame SNR ${\rm SNR}_1$ and 2×2 binning factor (signal $\times 4$, noise $\times 2$):

$$
{\rm SNR}_{\rm sum} = {\rm SNR}_1 \cdot \sqrt{N} \cdot 2
$$

For $N = 50$: ${\rm SNR}_{\rm sum} \approx 14\,{\rm SNR}_1$. Cosmic-ray contamination is removed before summation by the 2-stage scrubbing (photon-noise compatibility test against previous frame; 3-frame median filter). / 50회 합산 시 SNR ≈ 14×.

### 4.5 Fresnel knife-edge cascade rejection (HI forward baffle)

For a single knife-edge, far-field rejection at angular distance $\alpha$ below the edge (in arc units):

$$
\frac{B(\alpha)}{B_\odot} \;\approx\; \frac{1}{4\pi^2 \alpha^2}
$$

(Fresnel-Kirchhoff diffraction, semi-infinite half-screen, large-$\alpha$ asymptote.) For an n-vane cascade where each vane lies in the previous shadow, the rejection composes approximately multiplicatively. The HI-1 5-vane cascade achieves 10⁻¹³ B☉ at the limb-side FOV edge, agreement with theory across 12 orders of magnitude (Fig. 18, p. 90). / 단일 knife-edge의 회절 후방 잔광 근사. 5-vane 카스케이드가 ~10⁻¹³ B☉ 거부 달성.

### 4.6 Compression / image-volume budget

A full SECCHI image is:
$$
V_{\rm raw} = 2048 \times 2048 \times 14\;{\rm bit} = 57.5\;{\rm Mbit} \;\approx\; 8\;{\rm MB\ packed}
$$

At ICER lossy wavelet compression up to 200×, downlink time at 100 kbps becomes ~3 s; at the standard 20× the budget is ~24 s. Daily synoptic volume (Table 10): 4386 Mbits/day at compression factors 10× (COR/EUVI) to 2.5× (HI). / 압축률 20× 시 한 영상 다운링크 ~24 s.

### 4.7 J-map (time–elongation) construction

For each HI image taken at time $t_i$, extract a 1-D strip along the ecliptic latitude $\beta = 0$:

$$
J(\epsilon, t_i) = \langle I(\epsilon, \beta, t_i)\rangle_{|\beta|<\Delta}
$$

Stack strips in time → 2-D J-map. CME tracks appear as bright slanted features whose slope $d\epsilon/dt$ relates to radial speed via Sun-spacecraft geometry. The Fixed-Phi (or Harmonic Mean) approximation gives:

$$
v = \frac{c\,d\epsilon/dt}{\sin(\phi - \epsilon) + \sin\epsilon}
$$

where $\phi$ is the propagation angle (relative to Sun-spacecraft line) and $c = 1$ AU. This is the standard tool in modern HI analysis (post-2008). / HI의 표준 분석 도구로 SECCHI 이후 CME 추적의 표준이 됨.

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1939 ┃ Lyot — "Étude de la couronne solaire en dehors des éclipses"
     ┃   the first coronagraph (ground-based)
1971 ┃ Tousey — first detection of a CME, OSO-7
1973 ┃ Skylab/ATM — first space-based imaging coronagraph (1973–1974)
1979 ┃ Solwind P78-1 — early space coronagraph
1980 ┃ SMM/Coronagraph-Polarimeter — improved capability
1991 ┃ Yohkoh — soft X-ray imaging of the corona
1995 ┃ SOHO/LASCO (C1, C2, C3) — ★ benchmark for CME climatology
1996 ┃ SOHO/EIT — EUV full-disk imaging, antecedent of EUVI
1998 ┃ TRACE — high-resolution EUV imaging, heritage for EUVI mechanisms
2003 ┃ SMEI on Coriolis — first all-sky heliospheric imaging concept
     ┃   (Eyles et al. — direct heritage for HI baffle design)
═══━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2006 ┃ STEREO-A & STEREO-B launch (Oct 25)
     ┃ ★ THIS PAPER ★ Howard et al. SECCHI overview (2008)
═══━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2009 ┃ Thernisien et al. — GCS 3-D CME model fit to SECCHI A/B
2010 ┃ SDO launch — AIA inherits 4-quadrant→7-channel EUV approach
2011 ┃ STEREO-A and -B in quadrature with Earth (90° apart)
2014 ┃ STEREO-B contact lost (recovered 2015 briefly)
2018 ┃ Parker Solar Probe launch — uses SECCHI imaging context
2020 ┃ Solar Orbiter launch — Metis/EUI inherit SECCHI lineage
2025 ┃ NASA PUNCH launch — direct successor to SECCHI HI concept
```

The paper sits at a hinge: it documents the closing of an instrument-development era (1995–2008) when LASCO defined CME imaging, and it inaugurates the era of *systems-of-systems* heliophysics — multi-spacecraft, multi-wavelength, multi-FOV imaging chains that became the rule from 2010 onward.

이 논문은 1995–2008년의 LASCO 기반 코로나그래프 시대를 닫는 동시에, *시스템들의 시스템* 지구물리(다중 우주선·다파장·다FOV 영상 체인) 시대를 여는 경첩에 위치한다. 이 패턴은 2010년 이후 표준이 됨.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **Lyot (1939)** "The study of the solar corona and prominences without eclipses" | The foundational coronagraph principle; cited for COR1/COR2 stray-light suppression strategy. / 코로나그래프의 원리적 토대; COR1/COR2 미광 억제 전략의 직접 인용 출처. | Foundational — without Lyot, no SECCHI coronagraphs / 모든 코로나그래프 설계의 시조. |
| **Brueckner et al. (1995)** SOHO/LASCO | Direct heritage: COR2 design lineage (C2/C3), Rice/H-Compress compression, planning tool, software architecture. / 직접 계보: COR2 설계(C2/C3 파생), Rice/H-Compress 압축, 계획 도구, SW 아키텍처. | Highest — COR2 is essentially "LASCO done better" / COR2는 본질적으로 "더 잘 만든 LASCO". |
| **Delaboudiniere et al. (1995)** SOHO/EIT | Antecedent of EUVI; both use thin-film Al filters, Mo/Si multilayers. / EUVI의 직접 선행; 같은 박막 Al 필터, Mo/Si 다층막 사용. | High — EUVI is "EIT plus quadrant tuning" / EUVI는 "EIT + 사분면 튜닝". |
| **Strong et al. (1994); Handy et al. (1999)** TRACE | EUVI inherits TRACE's quadrant-selector mechanism design. / EUVI는 TRACE의 사분면 selector 기구 설계 계승. | Medium-high — mechanism heritage / 기구 계보. |
| **Eyles et al. (2003)** SMEI on Coriolis | The only prior demonstration that a baffled, shutterless wide-field instrument could detect CMEs against the F-corona — direct technological heritage for HI. / 차폐된 무셔터 광시야 기기로 F-corona 배경 위에서 CME를 검출한 유일 선행 증명 — HI의 직접 기술 계보. | Very high — HI's existence depends on SMEI's success / HI 존재의 전제 조건. |
| **Buffington et al. (1996)** Lab measurements of scattering rejection | Original quantification of vane-cascade rejection vs. occulter geometry — provided the design equations for HI baffles. / vane-cascade 거부의 원초적 정량화 — HI 배플 설계 식의 출처. | High — HI baffle theory / HI 배플 이론. |
| **Tappin et al. (2003)** SMEI flight performance | Demonstrated CME detection in SMEI data; gave HI the empirical confidence to fly. / SMEI 데이터에서 실제 CME 검출 입증; HI 비행 결정에 결정적 경험 근거. | High — empirical validation / 경험적 검증. |
| **Pesnell et al. (2012)** SDO/AIA | Successor: AIA's 7-channel EUV imaging extends EUVI's 4-quadrant strategy to 7 separate telescopes for higher cadence. / 후속: AIA의 7채널 EUV 영상은 EUVI의 4-사분면 전략을 7개 독립 망원경으로 확장. | Successor — generational evolution / 세대적 진화. |
| **Thernisien et al. (2009)** GCS model | First 3-D CME morphology fit using SECCHI A/B simultaneous images — the canonical post-SECCHI analysis tool. / SECCHI A/B 동시 영상으로 한 첫 3D CME 형태 fit — SECCHI 이후 표준 분석 도구. | Foundational consumer / 핵심 후속 분석. |
| **Müller et al. (2020)** Solar Orbiter | Solar Orbiter/Metis (coronagraph) and EUI inherit SECCHI's instrument-chain philosophy and many design conventions. / Solar Orbiter의 Metis·EUI는 SECCHI의 기기 체인 철학과 다수의 설계 관례를 계승. | Successor — design lineage / 설계 계보. |

---

## 7. References / 참고문헌

- R. A. Howard, J. D. Moses, A. Vourlidas, et al., "Sun Earth Connection Coronal and Heliospheric Investigation (SECCHI)", *Space Science Reviews* **136**, 67–115 (2008). DOI: [10.1007/s11214-008-9341-4](https://doi.org/10.1007/s11214-008-9341-4)
- B. Lyot, "The study of the solar corona and prominences without eclipses (George Darwin Lecture, 1939)", *Monthly Notices of the Royal Astronomical Society* **99**, 580–594 (1939).
- G. E. Brueckner, R. A. Howard, M. J. Koomen, et al., "The Large Angle Spectroscopic Coronagraph (LASCO)", *Solar Physics* **162**, 357–402 (1995).
- J.-P. Delaboudinière et al., "EIT: extreme-ultraviolet imaging telescope for the SOHO mission", *Solar Physics* **162**, 291–312 (1995).
- C. J. Eyles, G. M. Simnett, M. P. Cooke, et al., "The Solar Mass Ejection Imager (SMEI)", *Solar Physics* **217**, 319–347 (2003).
- S. J. Tappin, A. Buffington, M. P. Cooke, et al., "Tracking a major interplanetary disturbance with SMEI", *Geophysical Research Letters* **31**, L02802 (2003).
- J. P. Wülser et al., "EUVI: the STEREO-SECCHI extreme ultraviolet imager", *Proc. SPIE* **5171** (2003).
- W. T. Thompson, J. M. Davila, R. R. Fisher, et al., "COR1 inner coronagraph for STEREO-SECCHI", *Proc. SPIE* **4853** (2003).
- J.-M. Defise, J.-P. Halain, E. Mazy, et al., "Design and tests for the heliospheric imager of the STEREO mission", *Proc. SPIE* **4853** (2003).
- A. Buffington, B. V. Jackson, C. M. Korendyke, "Wavelength dependence of scattered light in coronagraph occulters", *Applied Optics* **35**, 6669–6673 (1996).
- E. Gibson, *The Quiet Sun*, NASA SP-303 (1973).
- K. Saito, A. I. Poland, R. H. Munro, "A study of the background corona near solar minimum", *Solar Physics* **55**, 121–134 (1977).
- K. P. Dere, E. Landi, H. E. Mason, B. C. Monsignori-Fossi, P. R. Young, "CHIANTI — an atomic database for emission lines", *A&A Suppl. Ser.* **125**, 149 (1997).
- A. Thernisien, A. Vourlidas, R. A. Howard, "Forward modeling of coronal mass ejections using STEREO/SECCHI data", *Solar Physics* **256**, 111–130 (2009).
