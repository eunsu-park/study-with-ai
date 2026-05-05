---
title: "The Spectrometer/Telescope for Imaging X-rays (STIX)"
authors: ["Säm Krucker", "Gordon J. Hurford", "Oliver Grimm", "STIX Team"]
year: 2020
journal: "Astronomy & Astrophysics"
doi: "10.1051/0004-6361/201937362"
topic: Solar_Observation
tags: [STIX, Solar_Orbiter, hard_X-ray, Fourier_imaging, CdTe, Moire, RHESSI_legacy, visibility, bigrid]
status: completed
date_started: 2026-04-25
date_completed: 2026-04-25
---

# 59. The Spectrometer/Telescope for Imaging X-rays (STIX) / Solar Orbiter 탑재 X선 영상 분광기

---

## 1. Core Contribution / 핵심 기여

**English** — Krucker et al. (2020) document the complete design of STIX, the dedicated 4–150 keV hard X-ray (HXR) imaging spectrometer aboard ESA's Solar Orbiter. STIX's central engineering achievement is to deliver RHESSI-class flare HXR diagnostics (~7 arcsec angular resolution, ~1 keV FWHM at 6 keV, full-disk imaging, 0.1–1 s time resolution) inside an extremely austere resource envelope — 6.58 kg, ~8 W, 700 bit s⁻¹ telemetry — appropriate for a deep-space mission. The instrument is an indirect Fourier-transform imager: 32 tungsten bigrid subcollimators (front + rear grids separated by 55 cm, pitches 38 μm to 1 mm) modulate incident X-rays into Moiré beat patterns; 32 coarsely pixelated CdTe Caliste-SO detectors record those patterns; four "phased" detector pixels per subcollimator measure the real and imaginary parts of one complex Fourier component (visibility) of the source. Thirty visibilities (logarithmically spaced 7.1″–179″) are downlinked, and ground software reconstructs images by back-projection, CLEAN, MEM, or Bayesian inversion. The paper also reports CdTe physics (Schottky-bias polarization, ¹³³Ba calibration, proton-radiation damage), the autonomous Rate Control Regime that handles 10⁴ dynamic range via attenuator + pixel disabling + pixel cycling, and the FPGA/LEON3 onboard architecture that bins 800 000 photons s⁻¹ into 32 energies × 32 detectors × 12 pixel accumulators.

**한국어** — Krucker 외(2020)는 ESA Solar Orbiter에 탑재된 4–150 keV 전용 경X선(HXR) 영상 분광기 STIX의 전체 설계를 문서화한 기기 논문이다. STIX의 핵심 공학적 성취는 RHESSI급 플레어 HXR 진단 능력(분해능 ~7 arcsec, 6 keV에서 FWHM ~1 keV, 전 태양면 영상, 0.1–1 s 시간 분해능)을 6.58 kg, ~8 W, 700 bit s⁻¹ 텔레메트리라는 극도로 빠듯한 심우주 임무 자원 한계 안에서 구현했다는 점이다. 기기는 간접 Fourier 변환 영상기이다: 텅스텐 양면격자 부시준기 32개(전·후 격자 55 cm 간격, 피치 38 μm–1 mm)가 입사 X선을 Moiré 비트 패턴으로 변조하고, 32개의 거친 화소 CdTe Caliste-SO 검출기가 이 패턴을 기록한다. 부시준기당 4개의 "위상" 화소가 광원의 한 복소 Fourier 성분(가시도)의 실수·허수부를 측정한다. 30개 가시도(7.1″–179″, 로그 간격)가 다운링크되며, 지상 소프트웨어가 back-projection, CLEAN, MEM, Bayes 역산으로 영상을 재구성한다. 논문은 또한 CdTe 물리(Schottky 분극, ¹³³Ba 보정, 양성자 방사선 손상), 감쇠기 + 화소 차단 + 화소 순환으로 10⁴ 동적 범위를 다루는 자율 Rate Control Regime, 그리고 800 000 photons s⁻¹를 32 에너지 × 32 검출기 × 12 화소 누산기로 비닝하는 FPGA/LEON3 온보드 아키텍처를 상세히 보고한다.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction & Science Objectives (Sect. 1–2) / 서론·과학 목표

**English** — STIX is one of ten Solar Orbiter instruments and the only HXR imager. Hard X-ray bremsstrahlung from accelerated electrons gives unique diagnostics of (i) the hottest (≳10 MK) flare plasma seen at low energies and (ii) the nonthermal electron tail at higher energies. Brown (1971) provides the inversion from photon spectrum to electron spectrum. Two science objectives: (1) understand electron acceleration at the Sun and their transport into interplanetary space; (2) determine the magnetic connection of Solar Orbiter back to the Sun. STIX is therefore the bridge between Solar Orbiter's remote and in-situ instruments. Figure 1 shows expected count spectra for an A6 (no attenuator) and an M7.7 (with attenuator) flare in default 10 s accumulations, with thermal (red), nonthermal (blue) and background (green) components separated.

**한국어** — STIX는 Solar Orbiter의 10개 기기 중 유일한 HXR 영상기이다. 가속 전자에 의한 경X선 제동복사는 (i) 저에너지에서 가장 뜨거운(≳10 MK) 플레어 플라즈마와 (ii) 고에너지에서 비열적 전자 꼬리를 동시에 진단한다. Brown(1971)이 광자 스펙트럼에서 전자 스펙트럼으로의 역산을 제공한다. 두 과학 목표: (1) 태양에서의 전자 가속과 행성간 공간 수송 이해, (2) Solar Orbiter와 태양 사이의 자기 연결 결정. 따라서 STIX는 원격–현장 기기를 잇는 다리이다. Figure 1은 A6(감쇠기 없음)·M7.7(감쇠기 있음) 플레어의 10 s 적분 예상 계수 스펙트럼을 열적(빨강)·비열적(파랑)·배경(초록) 성분으로 분리해 보여준다.

**STIX specification summary (Table 1):**
- **Energy range**: 4–150 keV
- **Energy resolution (FWHM)**: 1 keV at 6 keV
- **Effective area**: 6 cm²
- **Finest angular resolution**: 7 arcsec
- **FOV for imaging**: 2°×2° (centered on Sun)
- **Image placement accuracy**: 4 arcsec
- **Time resolution**: 0.1–1 s (during flares)
- **Nominal power**: ~8 W; **Mass**: 6.58 kg (excluding window); **Telemetry**: 700 bit s⁻¹

### Part II: Instrument Overview (Sect. 3) / 기기 개요

**English** — Three subsystems (Fig. 2): (a) **Window**: pair of beryllium-coated thermal/X-ray windows (front 2 mm + rear 1 mm) on the spacecraft heat shield, with 5 mm and 25 mm aspect apertures; (b) **Imager**: front grid + rear grid (separation L = 0.55 m), each split into 32 subareas (windows of 22×20 mm front, 13×13 mm rear); (c) **DEM** (Detector/Electronics Module): 32 CdTe Caliste-SO hybrids, ¹³³Ba calibration sources, IDPU, attenuator, cold unit. Figure 3 shows the instrument's three limiting factors as a function of energy: window transmission (low-E rolloff at ~4 keV, attenuator-on rolls off at ~10 keV), grid efficiency (~1 except for the K-edge dip at 60 keV — tungsten K-edge — and gradual decrease above 100 keV due to grid penetration), and detection probability (1 mm CdTe gives ~100 % at 30 keV, ~65 % at 100 keV, ~30 % at 150 keV). The product gives ~6 cm² effective area peaking around 25 keV.

**한국어** — 세 하부 시스템(Fig. 2): (a) **Window**: 우주선 열차폐막에 부착된 베릴륨 코팅 열·X선 창 한 쌍(전 2 mm + 후 1 mm), 5 mm·25 mm 자세 측정용 개구; (b) **Imager**: 전·후 격자(간격 L = 0.55 m), 각각 32개 부면(전 22×20 mm, 후 13×13 mm); (c) **DEM**: 32개 CdTe Caliste-SO 하이브리드, ¹³³Ba 보정원, IDPU, 감쇠기, 냉각 유닛. Figure 3는 에너지에 따른 세 제한 인자를 보여준다: 창 투과도(~4 keV에서 저에너지 컷오프, 감쇠기 ON 시 ~10 keV로 컷오프 이동), 격자 효율(~1; 단 60 keV의 텅스텐 K-edge 흡수 딥, 100 keV 이상에서 격자 침투로 점진적 감소), 검출 확률(1 mm CdTe에서 30 keV ~100 %, 100 keV ~65 %, 150 keV ~30 %). 곱하면 ~25 keV 부근에서 정점을 이루는 ~6 cm² 유효 면적이 된다.

### Part III: Imaging (Sect. 4) — The Heart of the Paper / 영상 — 논문의 심장

**English** — STIX uses Fourier-transform bigrid imaging (Hurford 2013) because focused HXR optics (Wolter-I) are too heavy for Solar Orbiter. Inside each subcollimator, front and rear grids contain large numbers of equispaced parallel slits with slightly different pitch and/or orientation. For parallel incident X-rays, the combined transmission of the grid pair forms a large-scale Moiré pattern on the detector with period equal to the detector width (8.8 mm) and orientation parallel to the s/c Y axis. Amplitude and phase of the Moiré pattern measure the visibility of one Fourier component of the source angular distribution.

**한국어** — STIX는 집속 HXR 광학(Wolter-I)이 Solar Orbiter에 너무 무겁기 때문에 Fourier 변환 양면격자 영상 기법(Hurford 2013)을 사용한다. 각 부시준기 내부에서 전·후 격자는 살짝 다른 피치·방향의 평행 슬릿을 다수 포함한다. 평행 입사 X선에 대해 격자쌍의 결합 투과도는 검출기 위에 대규모 Moiré 패턴(주기 = 검출기 폭 8.8 mm, 방향 = 우주선 Y축에 평행)을 만들고, 이 패턴의 진폭과 위상이 광원의 한 Fourier 성분의 가시도를 측정한다.

#### 4.1 Grids / 격자

**English** — Grids fabricated by Mikro Systems Inc using the same etched-foil-stack technique as RHESSI: 4–12 layers of 35–100 μm tungsten foils, total effective grid thickness 400 μm. The six finest grids (54 μm and 38 μm pitch) use "phased" two-subset stacks to achieve adequate slit etching (so effective absorption is half, but pitch is preserved). Figure 4 photographs the flight-spare front (left) and rear (center) grids; the right panel shows a back-illuminated prototype where Moiré patterns are directly visible. Figure 5 shows a closeup of grid 26 (474 μm pitch) and grid 1 (78 μm pitch) under UV illumination.

**한국어** — 격자는 Mikro Systems사가 RHESSI와 동일한 적층-에칭 기법(35–100 μm 텅스텐 박판 4–12 층, 총 유효 두께 400 μm)으로 제작했다. 가장 미세한 6개 격자(54·38 μm 피치)는 "phased" 2-부분 적층 기법으로 슬릿 에칭 어려움을 극복했다(유효 흡수는 절반이나 피치는 유지). Figure 4는 비행 예비품의 전(좌)·후(중앙) 격자, 우측은 Moiré가 직접 보이는 시제품. Figure 5는 grid 26(474 μm 피치)·grid 1(78 μm 피치) UV 조명 사진.

#### 4.2 Visibilities / 가시도

**English** — Each subcollimator measures a Fourier component at angular frequency $(u,v)$ set by the average pitch and orientation of the grids. The Fourier period equals the ratio of average pitch to grid separation (the angular resolution is half the period):
$$ \theta = \frac{p_{\mathrm{eff}}}{2L} = \frac{38\,\mu\mathrm{m}}{2 \times 0.55\,\mathrm{m}} = 3.45\times 10^{-5}\,\mathrm{rad} = 7.1\,\mathrm{arcsec.} $$
The 30 imaging subcollimators sample 10 different angular resolutions (logarithmically spaced 7.1″–179″, step factor 1.43) at 3 orientations each. Figure 6 illustrates a single subcollimator: front grid + rear grid with slightly different pitch/orientation make a Moiré pattern (bottom left); a simulated off-axis source (top right) shows photons sorted into four large pixels A, B, C, D phased across one Moiré period; the cosine fit (center right) gives Re V = C − A, Im V = D − B, total flux = A+B+C+D, with cross-check A+C = B+D (independent of visibility, flux, and background). Figure 7 shows the (u,v) coverage of the 30 STIX components: the synthesized point response function has FWHM 19″ with natural weighting, 11″ with uniform weighting, over a 360″×360″ FOV. Table 2 lists all 10 grid-resolution groups (3 subcollimators each, identical pitch but different orientation 30°/90°/150° or 50°/110°/170°, etc.).

**한국어** — 각 부시준기는 격자 평균 피치와 방향이 결정하는 각주파수 $(u,v)$의 Fourier 성분을 측정한다. Fourier 주기는 평균 피치/격자 간격이며 각분해능은 그 절반이다:
$$ \theta = \frac{p_{\mathrm{eff}}}{2L} = \frac{38\,\mu\mathrm{m}}{2 \times 0.55\,\mathrm{m}} = 7.1\,\mathrm{arcsec.} $$
30개 영상 부시준기는 10개의 분해능(로그 간격 7.1″–179″, 단계비 1.43)을 각 방향 3개씩 샘플링한다. Figure 6은 한 부시준기를 설명한다: 전·후 격자가 Moiré 패턴 형성, 시뮬레이션된 비축 광원의 광자가 한 주기에 걸쳐 위상 분배된 4개 화소 A,B,C,D에 분류되고, 코사인 피팅이 Re V = C−A, Im V = D−B, 총 광속 = A+B+C+D를 준다. 교차 점검 A+C = B+D는 가시도·광속·배경과 독립이다. Figure 7의 (u,v) 커버리지: 자연 가중 19″, 균일 가중 11″ FWHM의 합성 점퍼짐함수. Table 2는 10개 격자 분해능 그룹(각 3 부시준기, 동일 피치·다른 방향)을 나열.

#### 4.3 Image Reconstruction / 영상 재구성

**English** — Image and visibility are connected by
$$ V(u,v) = \iint I(x,y)\, e^{i 2\pi(ux+vy)}\, dx\, dy, \qquad I(x,y) = \iint V(u,v)\, e^{-i 2\pi(ux+vy)}\, du\, dv. $$
With only 30 sampled $(u,v)$ pairs (vs. hundreds–thousands in radio), STIX cannot fully invert: it produces a "dirty map" that must be deconvolved. Available methods:
1. **Back-projection** = direct sum $\sum_k V_k\, e^{-i2\pi(u_k x + v_k y)}$ — simple, gives the dirty map.
2. **CLEAN** — RHESSI heritage; iteratively subtracts point-source sidelobes.
3. **Maximum Entropy Method (MEM)** — pixon-style image priors.
4. **Count-based imaging methods** (Massa et al. 2019).
5. **Bayesian / regularized methods** — well-suited to Poisson statistics.

For STIX, dynamic range goal is 20:1 for a strong isolated source. Imaging FOV is selectable post-facto anywhere on the Sun.

**한국어** — 영상과 가시도는 다음으로 연결된다:
$$ V(u,v) = \iint I(x,y)\, e^{i 2\pi(ux+vy)}\, dx\, dy, \qquad I(x,y) = \iint V(u,v)\, e^{-i 2\pi(ux+vy)}\, du\, dv. $$
30 표본만으로는 완전 역산이 불가능하므로 STIX는 "dirty map"을 만들고 디컨볼브해야 한다. 사용 가능 방법: (1) **Back-projection** — 직접 합산, dirty map; (2) **CLEAN** — RHESSI 유산, 점원 사이드로브 반복 제거; (3) **MEM** — 픽손 사전; (4) **계수 기반 영상 기법**(Massa 외 2019); (5) **Bayes/정규화** — Poisson 통계에 적합. 강한 단일 광원 동적 범위 목표는 20:1. 영상 FOV는 사후에 태양면 어디든 선택 가능.

#### 4.4–4.6 BKG, CFL, Aspect / 배경·거친 위치·자세

**English** — Two of 32 subcollimators are non-imaging:
- **BKG (subcollimator 10)** — open front, rear has 6 small apertures (two each of 0.01, 0.1, 1 mm²). Never covered by attenuator. Provides live background and unattenuated low-energy flux during big flares. Figure 8 shows the pattern.
- **CFL (subcollimator 9)** — open rear except for "H-shaped" pattern. Provides 2-arcmin real-time flare location across 2°×2° FOV by exploiting the slope of the H pattern across the detector face. Figure 9 shows the geometry.
- **Aspect**: 28 mm focal-length lens in front grid focuses the 550–700 nm solar image onto the rear grid, where 90–300 μm circular apertures in 4 orthogonal radial rows let the integrated light onto 4 photodiodes. As the Sun's apparent diameter changes from orbital motion, limbs cross apertures every few days even with stable pointing — providing absolute aspect to <4 arcsec. Cosmic-ray imaging (Crab pulsar within 1° of Sun) gives absolute roll. Figure 10 shows the aspect schematic.

**한국어** — 32 부시준기 중 2개는 비영상 용도:
- **BKG (부시준기 10)** — 전 격자 개방, 후 격자에 6개 소형 개구(0.01, 0.1, 1 mm² 각 2개). 감쇠기 영향 없이 실시간 배경·저에너지 광속 측정.
- **CFL (부시준기 9)** — 후 격자 개방(H자 패턴 제외). 검출기 면을 가로지르는 H자 기울기로 2°×2° 시야에서 2 arcmin 실시간 플레어 위치 결정.
- **Aspect**: 전 격자의 28 mm 렌즈가 550–700 nm 태양상을 후 격자로 결상, 4 직교 방사 행의 90–300 μm 원형 개구가 4개 광다이오드로 빛을 보낸다. 궤도 운동으로 태양 시직경이 변하므로 안정 지향에서도 며칠마다 림이 개구를 가로질러 <4 arcsec 절대 자세 결정. 태양 1° 내 통과하는 Crab 펄서 관측이 절대 롤 보정.

### Part IV: Detector Electronics Module (Sect. 5) / 검출기 전자 모듈

#### 5.1–5.2 Enclosure & Attenuator / 외피·감쇠기

**English** — The DEM has two boxes: the **Detector Box** (attenuator + cold unit + CdTe + front-end) and the **IDPU Box** (digital electronics + power). Mechanical attenuator: 0.6 mm Al blades inserted on a ~2 s timescale by two redundant Maxon brushless motors. With attenuator inserted, transmission at 6 keV drops to 10⁻⁸; at 10 keV to 2 %; at 20 keV to 60 %. Power profile: linear ramp to first plateau, then second plateau; ≤2 s timeout.

**한국어** — DEM은 두 박스로 구성: **Detector Box**(감쇠기 + 냉각 유닛 + CdTe + 전단 전자장치)와 **IDPU Box**(디지털 전자장치 + 전원). 기계식 감쇠기는 0.6 mm Al 블레이드, 이중화된 Maxon 무브러시 모터로 ~2 s에 삽입. 삽입 시 6 keV 투과율 10⁻⁸, 10 keV 2 %, 20 keV 60 %. 전력 프로파일은 첫 평탄부 → 둘째 평탄부 선형 램프, 2 s 타임아웃.

#### 5.3 X-ray Detection: CdTe / X선 검출 — CdTe

**English** — CdTe properties: density 5.85 g cm⁻³, average atomic number 50, photon-pair energy W = 4.43 eV, so 870 pairs per 4 keV photon. 1 mm thick crystal: ~65 % absorption at 100 keV, ~30 % at 150 keV. Charge-carrier lifetimes ~3 μs (electrons) and ~2 μs (holes). To collect charges swiftly, the maximum bias is required: Schottky electrode (Pt cathode, Au-Ti-Al multilayer anode) tolerates 200–500 V reverse bias (much higher than the depletion voltage). At 1 mm thickness electrons are nearly fully collected; holes only partially, depending on interaction depth — above ~50 keV, where photons can interact deep in the crystal, the spectrum shows a pronounced low-energy tail.

**한국어** — CdTe 특성: 밀도 5.85 g cm⁻³, 평균 원자번호 50, 광자-전하쌍 에너지 W = 4.43 eV이므로 4 keV 광자당 870쌍. 1 mm 두께 결정: 100 keV에서 ~65 %, 150 keV에서 ~30 % 흡수. 전하 운반자 수명 ~3 μs(전자), ~2 μs(정공). 빠른 전하 수집을 위해 최대 편향이 필요 — Schottky 전극(Pt 음극, Au-Ti-Al 다층 양극)이 공핍 전압보다 훨씬 높은 200–500 V 역편향을 견딘다. 1 mm 두께에서 전자는 거의 완전히 수집되지만, 정공은 상호작용 깊이에 따라 부분 수집 — 광자가 깊이 들어가는 ~50 keV 이상에서는 저에너지 꼬리(low-energy tail)가 현저하다.

**Fabrication**: 14×14×1 mm³ CdTe crystals from Acrorad (Japan) with Pt/Au-Ti-Al electrodes. Patterning at PSI by photolithography + plasma etching (Grimm et al. 2015). Final dicing to 10×10 mm². Pixel pattern (Fig. 12): four stripes sampling Moiré pattern, each subdivided into 2 large (9.6 mm² each) + 1 small (1.0 mm²) pixel — so 12 pixels per detector. Small pixels handle high count rates without pile-up; their lower capacitance also gives better energy resolution. Guard ring around crystal border protects from edge leakage.

**제작**: 14×14×1 mm³ CdTe 결정(Acrorad, 일본)에 Pt/Au-Ti-Al 전극. PSI에서 광식각 + 플라즈마 에칭(Grimm 외 2015). 최종 10×10 mm²로 절단. 화소 패턴(Fig. 12): Moiré 샘플링용 4 줄, 각 줄은 대형 화소 2개(각 9.6 mm²) + 소형 화소 1개(1.0 mm²)로 분할 — 검출기당 12 화소. 소형 화소는 고계수율 시 파일업 방지, 낮은 정전용량으로 에너지 분해능 우수. 결정 가장자리의 가드 링이 누설 차단.

#### 5.3.3 Caliste-SO Hybrids / Caliste-SO 하이브리드

**English** — Each CdTe is bonded to a Caliste-SO hybrid (Meuris et al. 2012) hosting an IDeF-X HD ASIC (Michalowska et al. 2010): 13 of 32 input channels are routed (12 pixels + guard ring at same potential). Adjustable per-channel trigger threshold; ~5 μs shaping for lowest noise. Total processing time ~12.5 μs per trigger (= dead time). Multi-pixel-hit events are discarded (ambiguous), but a single readout still occurs to keep dead time uniform. Figure 13 shows a Caliste-SO hybrid (~11×12×15 mm, by L. Godart/CEA).

**한국어** — 각 CdTe는 IDeF-X HD ASIC(Michalowska 외 2010)을 탑재한 Caliste-SO 하이브리드(Meuris 외 2012)에 결합 — 32개 입력 채널 중 13개 사용(12 화소 + 동전위 가드 링). 채널별 가변 트리거 임계, ~5 μs 성형 시간으로 최저 잡음. 트리거당 처리 시간 ~12.5 μs(= 데드 타임). 다중 화소 동시 사건은 모호하므로 폐기되지만 데드 타임 균일성을 위해 단일 판독은 수행. Figure 13에 Caliste-SO(~11×12×15 mm, L. Godart/CEA) 사진.

#### 5.4 Energy Calibration: ¹³³Ba / 에너지 보정

**English** — Imaging requires consistent energy bins across all pixels (goal: 100 eV rms). Achieved by continuously illuminating each pixel with a ¹³³Ba radioactive source: two strong lines at 31 keV and 81 keV. Total source activity 4.5 kBq distributed over 128 dots. Half-life 10.5 yr is enough for the mission. Ground tests during STIX final thermal-vacuum test campaign verified spectroscopic calibration capability (Fig. 15: spectrum from subcollimator 5, 87 s⁻¹ count rate, with FWHM 31 keV histogram for all 384 pixels showing better resolution for small pixels).

**한국어** — 영상에는 모든 화소의 에너지 빈이 일관되어야 한다(목표 100 eV rms). 각 화소를 ¹³³Ba 방사능원으로 지속 조명: 31 keV·81 keV 강한 라인 두 개. 총 활동도 4.5 kBq, 128개 도트에 분산. 반감기 10.5 년으로 임무 충분. STIX 최종 열진공 시험에서 분광 보정 능력 검증(Fig. 15: 부시준기 5의 87 s⁻¹ 계수율 스펙트럼, 384 화소 31 keV FWHM 히스토그램은 소형 화소가 더 좋은 분해능을 보임).

#### 5.5 Sensor Degradation / 센서 열화

**English** — Two slow degradation mechanisms:
1. **Schottky polarization** — at +4 °C and 200 V, FWHM doubles in ~half a day; at the nominal −20 °C, polarization timescale extends to ~1 month. 0 V reset for 2 minutes fully restores. STIX schedules daily short bias resets — negligible live-time impact.
2. **Radiation damage** — displacement damage from non-ionizing energy loss of solar protons (galactic CR is much weaker, displacement is solar-cycle dependent). Crystals were proton-irradiated at PSI with 50 MeV at fluence 1.6×10¹¹ cm⁻² (= mission-end equivalent); FWHM at 31 keV evolves shown in Fig. 16: 100 % fluence (mission end at solar max) gives FWHM rising to ~5–7 keV at 31 keV for some pixels. Calibration & resolution monitored on board with ¹³³Ba so FSW parameters can be updated. Annealing trials post-irradiation showed no useful recovery.

**한국어** — 두 가지 느린 열화 기제:
1. **Schottky 분극** — +4 °C·200 V에서 반나절에 FWHM 두 배; −20 °C 정상 운용 시 분극 시간 척도 ~1 개월. 2분간 0 V 리셋으로 완전 복원. STIX는 매일 짧은 바이어스 리셋 — 라이브타임 영향 미미.
2. **방사선 손상** — 태양 양성자의 비전리 에너지 손실에 의한 격자 변위 손상(은하 우주선보다 훨씬 강함, 태양 주기 의존적). 결정을 PSI에서 50 MeV 양성자 1.6×10¹¹ cm⁻² 조사(임무 종료 등가); Fig. 16의 31 keV FWHM 진화: 100 % 조사량(태양 극대 임무 종료)에서 일부 화소는 FWHM ~5–7 keV로 상승. ¹³³Ba 온보드 보정으로 FSW 파라미터 갱신 가능. 조사 후 어닐링은 유효 회복 미관측.

#### 5.6–5.7 IDPU & PSU / 데이터 처리부·전원 공급부

**English** — Direct digital control via single 100 MHz FPGA in the IDPU. LEON3 SPARC-type processor at 20 MHz handles ASW. Memory: 1 MiB EEPROM (SuSW × 3 copies majority voting), 2 MB SRAM working memory, 2× 64 MB SDRAM (one for working, one for rotating buffer), 16 GB flash for archive (ECC protected). Two redundant boards. Power Supply: two +28 V lines (main + redundant), high-power pulse command (HPC), DC-DC converters supply CdTe HV.

**한국어** — IDPU 내 단일 100 MHz FPGA로 직접 디지털 제어. 20 MHz LEON3 SPARC가 ASW 실행. 메모리: 1 MiB EEPROM (SuSW 3-copy 다수결), 2 MB SRAM 작업 메모리, 2× 64 MB SDRAM(작업·rotating buffer), 16 GB 플래시(ECC). 이중 보드. 전원: 2× +28 V 라인, HPC, CdTe HV용 DC-DC 변환기.

### Part V: Onboard Data Handling (Sect. 6) / 온보드 데이터 처리

**English** — Driver: 800 000 photons s⁻¹ input, 700 bit s⁻¹ output. FPGA prompt processing (Fig. 17) sorts each 24-bit photon word (5-bit detector ID + 4-bit pixel ID + 3 spare + 12-bit ADC energy) into 12 288 accumulators (32 energies × 32 detectors × 12 pixels). LEON3 then runs three parallel paths:
- **Primary path**: full imaging+spectroscopy data, downlinked only for flares.
- **Quick-look (QL) path**: 4 s cadence light curves, 8 s flare flag/location, detector spectra, variance, calibration spectra.
- **Calibration path**: ¹³³Ba long integrations (~20 h) for pixel-by-pixel energy calibration.

12-bit ADC is rebinned via a programmable look-up table to 32 "science energy channels" (30 keV-bins between 4 and 150 keV plus 2 integral bins above/below). Bin widths optimized for typical flare spectra: 1 keV at low E, broadening at high E.

**한국어** — 구동 요인: 800 000 photons s⁻¹ 입력, 700 bit s⁻¹ 출력. FPGA 즉시 처리(Fig. 17)가 각 24-bit 광자어(5-bit 검출기 ID + 4-bit 화소 ID + 3-bit 예비 + 12-bit ADC 에너지)를 12 288개 누산기(32 에너지 × 32 검출기 × 12 화소)에 분류. LEON3는 3 병렬 경로 실행: **Primary**(전체 영상·분광, 플레어시 다운링크), **QL**(4 s 광도곡선·8 s 플레어 플래그/위치·검출기 스펙트럼·분산·보정 스펙트럼), **Calibration**(¹³³Ba ~20 h 적분). 12-bit ADC는 프로그래머블 LUT로 32 "과학 에너지 채널"(4–150 keV 사이 30 빈 + 상하 2개 적분 빈)로 재비닝.

#### 6.5 High Rate Handling / 고계수율 처리

**English** — 8 autonomous Rate Control Regimes (RCRs) handle 10⁴ dynamic range: (1) attenuator insertion (~10² reduction at low E); (2) pixel disabling (top/bottom rows × 2 steps for ×20 reduction); (3) pixel cycling (only 1–2 pixels active at a time for ×2 or ×4). Combined factor up to 10⁴. Latency 4–12 s.

**한국어** — 8 자율 RCR가 10⁴ 동적 범위 처리: (1) 감쇠기 삽입(저에너지 ~10² 감소), (2) 화소 차단(상하 행 × 2 단계 → ×20 감소), (3) 화소 순환(한 번에 1–2 화소만 활성화 → ×2 or ×4). 결합 인자 최대 10⁴. 지연 시간 4–12 s.

### Part VI: Operations & Data Products (Sect. 7) / 운영·데이터 산출물

**English** — Six operating modes: OFF, BOOT, SAFE, CONFIGURATION, MAINTENANCE, NOMINAL. Calibration: energy (¹³³Ba), alignment (visibility redundancy among the 3 same-pitch subcollimators flags grid-grid twist), self-calibration (90 sets of 4-pixel sums {A+B+C+D} should be consistent with the spatially-integrating spectrometer per subcollimator). Data products:
- **Level 1** (raw FITS, units applied)
- **Level 2** QL-based (light curves 4 s × 5 bands, background 16 s, calibration spectra ~1 day ~0.4 keV resolution, aspect ~days <3 arcsec, flare list)
- **Level 2** flare data (spectrograms, spectra, light curves, **30 visibilities**, images up to 360″×360″ with 7″ resolution)
- **Level 3** spatial movies, spectral movies, nonthermal electron spectra (flare-selected only).

**한국어** — 6 운용 모드: OFF, BOOT, SAFE, CONFIGURATION, MAINTENANCE, NOMINAL. 보정: 에너지(¹³³Ba), 정렬(동일 피치 부시준기 3개의 가시도 잉여로 격자 트위스트 검출), 자기보정(부시준기당 90 세트의 4-화소 합 {A+B+C+D}는 공간 적분 분광계와 일관). 데이터 산출물: **Level 1**(원시 FITS), **Level 2 QL**(광도곡선 4 s × 5 밴드, 배경 16 s, 보정 스펙트럼 ~1 일 ~0.4 keV 분해능, 자세 ~며칠 <3 arcsec, 플레어 목록), **Level 2 플레어**(스펙트로그램, 스펙트럼, 광도곡선, **30 가시도**, 영상 최대 360″×360″ FOV·7″ 분해능), **Level 3** 공간 영화·분광 영화·비열적 전자 스펙트럼(플레어 선택분만).

---

## 3. Key Takeaways / 핵심 시사점

1. **Indirect Fourier imaging is the only HXR option for deep-space, low-resource missions.** — STIX shows that ~7-arcsec hard X-ray imaging spectroscopy is feasible with **6.58 kg, 8 W, 700 bit/s** by abandoning focusing optics, abandoning RHESSI's spacecraft rotation, and using stationary tungsten bigrid subcollimators that record visibilities (Fourier components) on coarsely pixelated CdTe. **간접 Fourier 영상은 심우주 저자원 임무의 유일한 HXR 옵션이다.** — 6.58 kg, 8 W, 700 bit/s에서 ~7 arcsec HXR 영상 분광이 가능함을 STIX가 보였다. 집속 광학과 우주선 회전 모두 포기하고, 고정형 텅스텐 양면격자 부시준기가 거친 화소 CdTe 위에 가시도(Fourier 성분)를 기록하는 방식으로 달성.

2. **The 4-phased-pixel readout converts a Moiré pattern into a complex visibility with two subtractions.** — Pixels A,B,C,D phased 0°/90°/180°/270° across one Moiré period yield Re V = C−A, Im V = D−B, total flux = A+B+C+D, with A+C = B+D as a background-independent check. This elegant arithmetic is why STIX can downlink 30 visibilities × 32 energy channels × time bins instead of full pixel-resolved images. **4-위상 화소 판독이 Moiré 패턴을 두 번의 차분으로 복소 가시도로 변환한다.** — 한 Moiré 주기에 걸쳐 0°/90°/180°/270°로 위상 분배된 A,B,C,D는 Re V = C−A, Im V = D−B, 총 광속 = A+B+C+D를 주고, A+C = B+D는 배경 독립 점검이다. STIX가 화소별 영상 대신 30 가시도 × 32 에너지 × 시간 빈만 다운링크하는 이유.

3. **The grid-pitch-to-resolution rule $\theta = p/(2L)$ pins the entire optical design.** — With 55 cm separation, pitch 38 μm gives 7.1 arcsec, pitch 1 mm gives 179 arcsec, and 10 logarithmically-spaced pitches sample sources from a few arcsec (compact footpoints) to active-region scale (~3 arcmin loops). **격자 피치-분해능 규칙 $\theta=p/(2L)$이 광학 설계 전체를 고정한다.** — 55 cm 간격에서 피치 38 μm는 7.1″, 1 mm는 179″, 10개 로그 간격 피치가 수 arcsec(컴팩트 발판)부터 활동영역 규모(~3 arcmin 루프)까지 광원을 샘플링.

4. **CdTe at 1 mm + Schottky bias is the spectroscopy enabler.** — High-Z CdTe (Z̄=50) gives ~65 % photopeak at 100 keV; Schottky-bias 200–500 V drives carriers fast enough that 1 keV FWHM at 6 keV is achievable, but daily polarization resets are mandatory and proton damage will roughly double FWHM by mission end. **CdTe 1 mm + Schottky 편향이 분광 성능의 핵심이다.** — 고원자번호 CdTe(Z̄=50)가 100 keV에서 ~65 % 광피크; Schottky 200–500 V 편향이 전하를 충분히 빠르게 수집해 6 keV에서 1 keV FWHM 달성. 다만 일일 분극 리셋 필수, 양성자 손상으로 임무 종료 시 FWHM ~2배.

5. **Three nested rate-control mechanisms give 10⁴ dynamic range.** — Attenuator (×10²), pixel disabling (×20), pixel cycling (×2–4) are autonomously combined by an onboard Rate Control Regime to keep STIX sensitive from quiet-Sun (~10² ph/s) to GOES X-class (~10¹⁰ ph/s) without saturating or losing imaging. **세 단계 계수율 제어가 10⁴ 동적 범위를 제공한다.** — 감쇠기(×10²), 화소 차단(×20), 화소 순환(×2–4)을 RCR이 자동 결합해 정온 태양(~10² ph/s)부터 GOES X급(~10¹⁰ ph/s)까지 영상 손실 없이 운용.

6. **Sparse (u,v) sampling makes ground-side image reconstruction algorithm-dependent — not a free lunch.** — Only 30 (u,v) points (vs. radio's hundreds-thousands) yield strong sidelobes; STIX therefore relies on CLEAN, MEM, and Bayesian count-based imaging (Massa et al. 2019). Dynamic range is capped at ~20:1 even for strong isolated sources. **희박한 (u,v) 샘플링은 지상 영상 재구성을 알고리즘 의존적으로 만든다.** — 30점만으로는 강한 사이드로브가 생기므로 CLEAN, MEM, Bayes 계수 기반 영상(Massa 외 2019)이 필수. 강한 단일 광원의 동적 범위도 ~20:1.

7. **The CFL subcollimator is the unsung hero for joint observations.** — A single H-shaped open-grid pattern projected onto detector pixels gives a real-time 2-arcmin flare location across 2°×2° FOV in 8 s — distributed on board to other Solar Orbiter instruments (EUI, SPICE, Metis, PHI) and to ground for triggering joint multi-instrument campaigns. **CFL 부시준기는 합동 관측의 숨은 영웅이다.** — H자 개방 격자 패턴이 검출기 화소에 투영되어 2°×2° 시야에서 8 s 안에 2 arcmin 실시간 플레어 위치를 결정, 다른 Solar Orbiter 기기와 지상에 즉시 전달해 합동 관측 트리거.

8. **¹³³Ba onboard calibration solves the in-flight drift problem.** — 4.5 kBq of ¹³³Ba (31 keV + 81 keV) continuously illuminates every pixel; pixel-by-pixel gain & offset are updated to FPGA LUTs every few days, removing both polarization-induced and radiation-damage-induced drifts from the science energy bins. **¹³³Ba 온보드 보정이 비행 중 드리프트 문제를 해결한다.** — 4.5 kBq ¹³³Ba(31 keV + 81 keV)가 모든 화소를 지속 조명, 화소별 이득·오프셋을 며칠마다 FPGA LUT로 갱신하여 분극·방사선 손상 유발 드리프트를 과학 에너지 빈에서 제거.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Core Imaging Equations / 핵심 영상 방정식

**van Cittert–Zernike / visibility definition:**
$$ V(u,v) = \iint I(x,y)\, e^{i\,2\pi(ux+vy)}\, dx\, dy \qquad \text{[Eq. 1 of paper]} $$
$I(x,y)$: source brightness in arcsec, $(u,v)$: angular frequency in arcsec⁻¹. Each STIX subcollimator measures one $V(u_k,v_k)$.

**Inverse (image reconstruction):**
$$ I(x,y) = \iint V(u,v)\, e^{-i\,2\pi(ux+vy)}\, du\, dv \approx \sum_{k=1}^{30} V_k\, e^{-i\,2\pi(u_k x + v_k y)} \qquad \text{[Eq. 2]} $$
With only 30 samples, the sum is the **back-projected dirty map**; CLEAN/MEM/Bayes deconvolution follows.

**Phased-pixel visibility extraction:**
$$ \boxed{\;\mathrm{Re}\,V = C - A,\quad \mathrm{Im}\,V = D - B,\quad F = A+B+C+D,\quad A + C = B + D\;} $$
The first two equations isolate the visibility from background; the third gives total flux; the fourth is a redundancy check independent of source morphology, flux, and background.

### 4.2 Grid Geometry / 격자 기하

**Angular resolution from pitch:**
$$ \theta_k = \frac{p_k^{\mathrm{eff}}}{2L}, \qquad L = 0.55\,\mathrm{m} $$
| Group | Pitch (mm) | $\theta$ (arcsec) |
|---|---|---|
| 1 | 0.0380 | 7.1 |
| 2 | 0.0543 | 10.2 |
| 3 | 0.0777 | 14.6 |
| 4 | 0.1112 | 20.9 |
| 5 | 0.1590 | 29.8 |
| 6 | 0.2275 | 42.7 |
| 7 | 0.3254 | 61.0 |
| 8 | 0.4655 | 87.3 |
| 9 | 0.6659 | 124.9 |
| 10 | 0.9526 | 178.6 |

Logarithmic step factor 1.43; 30 imaging subcollimators (3 orientations per pitch) plus CFL & BKG = 32.

**Moiré-pattern spatial frequency from front (1) and rear (2) grids:**
$$ \vec{f}_\mathrm{Moire} = \vec{f}_1 - \vec{f}_2,\qquad |\vec{f}_i| = 1/p_i $$
Choose $|\vec{f}_\mathrm{Moire}|^{-1} = 8.8$ mm (detector width) and orientation parallel to s/c Y so each detector sees exactly one Moiré period across its active area.

### 4.3 CdTe Spectral Response / CdTe 분광 응답

**Pair generation:** $N_\mathrm{pairs} = E_\gamma / W$ with $W=4.43$ eV (CdTe). Statistical FWHM (Fano-limited):
$$ \mathrm{FWHM}_\mathrm{stat}(E) = 2.355\sqrt{F\,W\,E}, \quad F\approx 0.10 $$
**Total FWHM** including electronic noise $\sigma_\mathrm{el}$ (rms electrons-equivalent):
$$ \mathrm{FWHM}_\mathrm{tot}(E) = 2.355\sqrt{F\,W\,E + (\sigma_\mathrm{el}\, W)^2} $$
**Worked example at E = 6 keV:**
- Pairs $N = 6000/4.43 \approx 1354$
- $\sigma_\mathrm{stat}^2 = F\,W\,E = 0.10 \cdot 4.43 \cdot 6000 = 2658\,\mathrm{eV}^2 \Rightarrow \sigma_\mathrm{stat}=51.5$ eV
- For 1 keV total FWHM: $\sigma_\mathrm{tot} = 1000/2.355 = 425$ eV → $\sigma_\mathrm{el}\cdot W = \sqrt{425^2 - 51.5^2}\approx 422$ eV → $\sigma_\mathrm{el}\approx 95$ rms-electrons (very low).

**Photoelectric absorption probability** (Beer–Lambert) in 1 mm CdTe:
$$ P_\mathrm{abs}(E) = 1 - e^{-\mu(E)\rho\,t}, \quad \rho = 5.85\,\mathrm{g\,cm^{-3}},\, t = 0.1\,\mathrm{cm} $$
Numerical values from paper (Fig. 3, third panel): ~100 % at 30 keV, ~65 % at 100 keV, ~30 % at 150 keV. Tungsten K-edge (~70 keV) is the primary spectral feature in grid efficiency, not detector.

### 4.4 Combined Imaging Efficiency / 결합 영상 효율

$$ \eta_\mathrm{img}(E) = T_\mathrm{window}(E)\, \times\, \eta_\mathrm{grid}(E)\, \times\, P_\mathrm{abs,CdTe}(E) $$
Effective area = $A_\mathrm{geom}\times \eta_\mathrm{img}\sim 6$ cm² peaking near 25 keV.

### 4.5 Onboard Data Architecture / 온보드 데이터 구조

**Accumulator dimensionality:**
$$ N_\mathrm{acc} = N_\mathrm{energy}\times N_\mathrm{detector}\times N_\mathrm{pixel} = 32\times 32\times 12 = 12\,288 $$
**Photon-word format**: 24 bits = 5 bits detector ID + 4 bits pixel ID + 3 spare + 12 bits ADC energy.
**Throughput**: 800 000 photons s⁻¹ → 700 bit s⁻¹ telemetry, compression factor ~3×10³ achieved by accumulator binning + integer compression to 8 bits.

### 4.6 Rate Control Regime Combinatorics / 계수율 제어 조합

$$ R_\mathrm{eff} = R_\mathrm{raw}\times f_\mathrm{att}(E)\times f_\mathrm{disable}\times f_\mathrm{cycle} $$
Worst-case suppression factor:
$$ f_\mathrm{att}\sim 10^{-2}\;(\text{low E}) \times f_\mathrm{disable}\sim 1/20 \times f_\mathrm{cycle}\sim 1/4 = 1.25\times 10^{-4} $$
i.e. ~10⁴ dynamic range, transitioning between 8 RCR levels with 4–12 s latency.

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1948 ─── Fritz/de Jager & Burgess: first solar X-ray detection (rockets)
1969 ─── OSO-3,5,7: first orbital HXR spectrometers
1971 ─── Brown: HXR thick-target inversion → electron-spectrum-from-photons
1980 ─── SMM/HXRBS: 30–500 keV collimated spectroscopy (no imaging)
1991 ─── Yohkoh/HXT (Kosugi+): 4-band Fourier-synthesis imaging (15–93 keV)
1999 ─── Hessian/HESI selected by NASA
2002 ─── RHESSI (Lin+): 9 RMC subcollimators, 3 keV–17 MeV, 2.3″–180″ at LEO
2007 ─── Piana+: visibility-based regularized imaging (Pixon)
2008 ─── Hannah+: microflare nonthermal statistics
2011 ─── Fletcher/Holman/Kontar reviews: flare standard model, electron acceleration
2013 ─── Hurford: Fourier imaging review (Observing Photons in Space)
2014 ─── FOXSI sounding rocket (Krucker+): focusing-optics HXR direct imaging demo
2015 ─── Grimm+: CdTe Schottky detector qualification at PSI
2018 ─── RHESSI decommissioned (Aug)
2019 ─── ASO-S/HXI design papers (Zhang+; Gan+)
2020 ── ► KRUCKER ET AL. (THIS PAPER): STIX instrument paper
2020 ─── Solar Orbiter launch (10 Feb)
2021 ─── First STIX flare detections; cruise commissioning
2022 ─── ASO-S launches (China), HXI begins co-observation with STIX
2024 ─── Solar Orbiter perihelia ≤0.3 AU; major joint campaigns w/ Parker Solar Probe
2026 ── ◀ (today) STIX is the only operating dedicated solar HXR imager
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Brown 1971, Sol. Phys. 18, 489 | HXR thick-target inversion theory | STIX's 4–150 keV photon spectra are inverted to electron spectra via this formalism. STIX 광자 스펙트럼이 이 식으로 전자 스펙트럼으로 역산. |
| Kosugi et al. 1991, Sol. Phys. 136, 17 | Yohkoh/HXT — first solar Fourier-imaging HXR | Direct ancestor of bigrid imaging. STIX inherits 4-pixel-per-subcollimator readout from HXT lineage. 양면격자 영상의 직접 조상. |
| Lin et al. 2002, Sol. Phys. 210, 3 | RHESSI instrument paper | The RHESSI paper STIX explicitly emulates (and slimsdown). RHESSI's RMC visibility paradigm is replaced by static bigrids; CdTe replaces cooled Ge. STIX가 명시적으로 모방·축소한 RHESSI 논문. |
| Hurford 2013, Observing Photons in Space | Fourier imaging review | Theoretical backbone for STIX's $V(u,v) = \int I e^{i2\pi(ux+vy)}$ design. $V(u,v)=\int I e^{i 2\pi(ux+vy)}$ 설계의 이론 기반. |
| Krucker et al. 2014 (FOXSI) | Focusing HXR demo | Cited as the alternative (focusing) approach STIX could not afford on Solar Orbiter. Solar Orbiter에 실을 수 없었던 대안 집속 광학. |
| Grimm et al. 2015, JINST 10, C02011 | CdTe Schottky detector qualification at PSI | The detector-physics paper specifying 1 mm CdTe, 200–500 V Schottky bias, 4.43 eV pair energy used in STIX. STIX의 1 mm CdTe·Schottky 편향·4.43 eV 전하쌍 에너지 사양 논문. |
| Meuris et al. 2012, Nucl. Instr. Meth. A695, 288 | Caliste-SO hybrid design | The detector hybrid (CdTe + IDeF-X HD ASIC) used in STIX's 32 detectors. STIX의 32개 검출기에 사용된 하이브리드. |
| Massa et al. 2019, A&A 624, A130 | Count-based imaging methods | One of the four ground-side reconstruction methods STIX relies on. STIX 지상 재구성 4대 방법 중 하나. |
| Müller et al. 2020, A&A 642, A1 | Solar Orbiter mission paper | Describes the spacecraft & mission STIX flies on; STIX paper is part of the same A&A Solar Orbiter Special Issue. STIX가 탑재된 우주선·임무 논문, 동일 특별호 수록. |
| Zhang et al. 2019, RAA 19, 157; Gan et al. 2019, RAA 19, 157 | ASO-S/HXI | Chinese parallel HXR imager (also bigrid) with which STIX co-observes. STIX와 공동 관측하는 중국의 병렬 HXR 영상기. |
| Piana et al. 2007, ApJ 665, 864 | Regularized visibility-based imaging | One of STIX's image reconstruction algorithms. STIX의 영상 재구성 알고리즘 중 하나. |

---

## 7. References / 참고문헌

- Brown, J. C., "The Deduction of Energy Spectra of Non-Thermal Electrons in Flares from the Observed Dynamic Spectra of Hard X-Ray Bursts", *Solar Physics*, 18, 489 (1971).
- Cola, A. & Farella, I., "The polarization mechanism in CdTe Schottky detectors", *Applied Physics Letters*, 94, 102 (2009).
- Eisen, Y., Evans, L. G., Floyd, S., et al., "CdTe semiconductor detectors for radiation environment in space", *Nucl. Instr. and Meth. A*, 491, 176 (2002).
- Fletcher, L., Dennis, B. R., Hudson, H. S., et al., "An Observational Overview of Solar Flares", *Space Sci. Rev.*, 159, 19 (2011).
- Gan, W., Zhu, C., Deng, Y., et al., "Advanced Space-based Solar Observatory (ASO-S)", *Research in Astronomy and Astrophysics*, 19, 157 (2019).
- Grimm, O., Bednarzik, M., Birrer, G., et al., "Patterning of pixelized CdTe detectors for STIX", *JINST*, 10, C02011 (2015).
- Hannah, I. G., Christe, S., Krucker, S., et al., "RHESSI microflare statistics", *ApJ*, 677, 704 (2008).
- Holman, G. D., Aschwanden, M. J., Aurass, H., et al., "Implications of X-Ray Observations for Electron Acceleration and Propagation in Solar Flares", *Space Sci. Rev.*, 159, 107 (2011).
- Hurford, G. J., "X-Ray Imaging with Collimators, Masks and Grids", in *Observing Photons in Space*, Springer, 2013.
- Kontar, E. P., Brown, J. C., Emslie, A. G., et al., "Deducing Electron Properties from Hard X-Ray Observations", *Space Sci. Rev.*, 159, 301 (2011).
- Kosugi, T., Makishima, K., Murakami, T., et al., "The Hard X-ray Telescope (HXT) for the Solar-A Mission", *Solar Physics*, 136, 17 (1991).
- **Krucker, S., Hurford, G. J., Grimm, O., et al., "The Spectrometer/Telescope for Imaging X-rays (STIX)", *A&A*, 642, A15 (2020). DOI: 10.1051/0004-6361/201937362** [this paper]
- Krucker, S., Christe, S., Glesener, L., et al., "First Images from the Focusing Optics X-Ray Solar Imager", *ApJ*, 793, L32 (2014).
- Lin, R. P., Dennis, B. R., Hurford, G. J., et al., "The Reuven Ramaty High-Energy Solar Spectroscopic Imager (RHESSI)", *Solar Physics*, 210, 3 (2002).
- Massa, P., Piana, M., Massone, A. M., et al., "Count-based imaging for STIX", *A&A*, 624, A130 (2019).
- Meuris, A., Limousin, O., & Blondel, C., "Front-end electronics for IDeF-X HD ASIC", *Nucl. Instr. and Meth. A*, 654, 293 (2011).
- Meuris, A., Limousin, O., Gevin, O., et al., "Caliste-SO design", *Nucl. Instr. and Meth. A*, 695, 288 (2012).
- Michalowska, A., Gevin, O., Lemaire, O., et al., "IDeF-X HD: a low-power multi-channel ASIC for high-resolution X-ray spectroscopy", *Nucl. Sci. Symp. Conf. Rec.*, 1556 (2010).
- Müller, D., St. Cyr, O. C., Zouganelis, I., et al., "The Solar Orbiter mission. Science overview", *A&A*, 642, A1 (2020).
- Piana, M., Massone, A. M., Hurford, G. J., et al., "Regularized image reconstruction for RHESSI", *ApJ*, 665, 846 (2007).
- Warmuth, A., Önel, H., & Mann, G., "The STIX aspect system", *Solar Physics*, 295, 90 (2020).
- Zhang, Z., Chen, G., Wu, Z., et al., "ASO-S/HXI design", *Research in Astronomy and Astrophysics*, 19, 157 (2019).
