---
title: "Pre-Reading Briefing: The Spectrometer/Telescope for Imaging X-rays (STIX)"
paper_id: "59_krucker_2020"
topic: Solar_Observation
date: 2026-04-25
type: briefing
---

# The Spectrometer/Telescope for Imaging X-rays (STIX): Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Krucker, S., Hurford, G. J., Grimm, O., et al., "The Spectrometer/Telescope for Imaging X-rays (STIX)", Astronomy & Astrophysics, 642, A15 (2020). DOI: 10.1051/0004-6361/201937362
**Author(s)**: Säm Krucker, Gordon J. Hurford, Oliver Grimm, and the STIX Team (~80 co-authors across CH/USA/PL/FR/DE/AT/IE/IT/ES)
**Year**: 2020

---

## 1. 핵심 기여 / Core Contribution

**English** — STIX is the dedicated hard X-ray (HXR) imaging spectrometer aboard ESA's Solar Orbiter, designed to diagnose >10 MK thermal flare plasmas and the nonthermal electrons accelerated during solar flares over the 4–150 keV band at ~1 keV FWHM resolution. The instrument paper presents the complete design of an indirect Fourier-transform imager built around 32 tungsten bigrid subcollimators that encode angular information (7–180 arcsec) as Moiré patterns on 32 coarsely pixelated CdTe Caliste-SO detectors. Because Solar Orbiter cannot afford the mass, power, and telemetry of a focusing optic or RHESSI-class rotating modulation collimator, STIX records visibilities (Fourier components of the source) and ground-reconstructs images — a strategy that fits within ~700 bit/s telemetry, ~8 W power, and 6.58 kg mass. STIX provides the only HXR view of the Sun co-orbiting with in-situ particle/field instruments at perihelia as close as 0.28 AU, enabling unique remote–in-situ linkage science.

**한국어** — STIX는 ESA Solar Orbiter 위성에 탑재된 전용 경X선(HXR) 영상 분광기로, 4–150 keV 대역에서 약 1 keV FWHM 분해능으로 태양 플레어의 >10 MK 고온 열 플라즈마와 비열적 가속 전자를 진단하도록 설계되었다. 본 기기 논문은 32개의 텅스텐 양면격자(bigrid) 부시준기(subcollimator)와 32개의 거친 화소(coarsely pixelated) CdTe Caliste-SO 검출기로 구성된 간접 Fourier 변환 영상기의 완전한 설계를 제시한다. 격자쌍이 만드는 Moiré 패턴이 7–180 arcsec 각 정보를 부호화하며, 검출기에서 각 부시준기는 한 개의 (u,v) Fourier 성분(가시도, visibility)을 측정한다. Solar Orbiter는 집속 광학이나 RHESSI식 회전변조 시준기를 감당할 질량·전력·텔레메트리 자원이 없기 때문에, STIX는 가시도만 다운링크하여 지상에서 영상을 재구성하는 전략을 취한다 — 약 700 bit/s 텔레메트리, ~8 W 전력, 6.58 kg 질량 안에서 동작한다. 0.28 AU 근일점까지 접근하는 궤도에서 입자·자기장 in-situ 기기와 함께 유일한 HXR 원격 시야를 제공함으로써 원격–현장 연계 과학을 가능하게 한다.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

**English** — Hard X-ray imaging of solar flares began with collimated detectors on OSO and SMM, matured with Yohkoh/HXT (Kosugi et al. 1991), and reached its modern peak with RHESSI (Lin et al. 2002), a low Earth orbit (LEO) HXR spectroscopic imager that used nine pairs of rotating tungsten grids to encode visibilities at 2.3″–180″ over 3 keV–17 MeV. RHESSI established the visibility-based Fourier imaging paradigm in solar HXR but required spacecraft rotation and a cooled-Ge spectrometer mass/power budget that Solar Orbiter (a deep-space, thermally hostile, telemetry-starved mission) cannot host. By 2014–2019, focused Wolter-I HXR optics (FOXSI sounding rockets) demonstrated direct imaging up to ~25 keV but remained too heavy for Solar Orbiter. STIX was conceived to deliver RHESSI-class HXR imaging spectroscopy with a tenth of the mass, a fixed (non-rotating) instrument, and ground-only image reconstruction, while the spacecraft itself spends most of its time orbiting between Mercury and Venus and observing the Sun off the ecliptic.

**한국어** — 태양 플레어의 경X선 영상은 OSO·SMM의 시준 검출기에서 출발하여 Yohkoh/HXT(Kosugi 외 1991)에서 성숙했고, RHESSI(Lin 외 2002)에 이르러 현대적 정점을 찍었다. RHESSI는 저궤도(LEO)에서 회전하는 텅스텐 격자 9쌍으로 3 keV–17 MeV, 2.3″–180″ 가시도를 부호화한 HXR 분광 영상기였다. RHESSI는 가시도 기반 Fourier 영상 패러다임을 태양 HXR에 정착시켰으나, 우주선(spacecraft) 자체 회전과 냉각 Ge 분광기의 질량·전력을 요구하여 Solar Orbiter(심우주·열적 가혹·텔레메트리 제약)에는 부적합했다. 2014–2019년에는 FOXSI 사운딩 로켓이 Wolter-I 집속 광학으로 ~25 keV 직접 영상을 입증했지만 Solar Orbiter에 싣기에는 여전히 무거웠다. STIX는 RHESSI급 HXR 영상 분광 능력을 1/10 질량, 회전 없는 고정 기기, 지상 영상 재구성만으로 구현하도록 설계되었다. 한편 Solar Orbiter는 수성–금성 사이를 돌며 황도면 밖에서 태양을 관측한다.

### 타임라인 / Timeline

```
1971 ─── Brown: thick-target HXR bremsstrahlung theory
1980 ─── SMM/HXRBS launches (collimated HXR spectrometer)
1991 ─── Yohkoh/HXT: 4-band Fourier-synthesis imaging
2002 ─── RHESSI launch (Lin et al.) — 9 RMC visibilities, 3 keV–17 MeV
2008 ─── Hannah et al.: microflare nonthermal statistics
2011 ─── Fletcher/Holman/Kontar reviews: flare standard model
2013 ─── Hurford: Fourier imaging review (Observing Photons in Space)
2014 ─── Krucker et al.: FOXSI focusing-optics HXR imaging demo
2015 ─── Grimm et al.: CdTe detector qualification for STIX
2020 ── ► KRUCKER ET AL. (THIS PAPER): STIX instrument paper
2020 ─── Solar Orbiter launch (10 Feb 2020)
2021+ ── First STIX flares, joint with Parker Solar Probe / EUI / SPICE
2025+ ── ASO-S/HXI (Zhang et al. 2019; Gan et al. 2019) operational HXR imaging
```

---

## 3. 필요한 배경 지식 / Prerequisites

**English** — To follow this paper comfortably, you should be familiar with:
1. **Bremsstrahlung emission**: HXR continuum from electrons decelerating on ambient ions; Brown (1971) thick-target inversion.
2. **Fourier-transform imaging / aperture synthesis**: van Cittert–Zernike theorem, visibility V(u,v) = ∫ I(x,y) exp[i 2π(ux+vy)] dx dy, and how a sparse (u,v) sample yields a "dirty map".
3. **Moiré patterns from grid pairs**: spatial-frequency arithmetic; how two slightly mismatched periodic transmission functions multiply to produce a low-frequency beat.
4. **Bigrid (RMC ancestry) imaging**: RHESSI subcollimators and how each pair encodes one Fourier component; Hurford (2013) review.
5. **Semiconductor X-ray detectors**: photoelectric absorption, charge-pair generation (W = 4.43 eV in CdTe), Ramo's theorem for induced charge, Schottky-bias polarization, and energy-resolution noise sources.
6. **Solar flare basics**: Neupert effect, looptops vs. footpoints, thermal/nonthermal partition, GOES classes (A/B/C/M/X).

**한국어** — 본 논문을 편안히 따라가려면 다음 배경이 필요하다:
1. **제동복사 (Bremsstrahlung)**: 전자가 주변 이온에서 감속될 때 방출되는 HXR 연속체; Brown(1971) 두꺼운-표적 역산.
2. **Fourier 변환 영상 / 어퍼처 합성**: van Cittert–Zernike 정리, 가시도 V(u,v) = ∫ I(x,y) exp[i 2π(ux+vy)] dx dy, 희박한 (u,v) 샘플로부터 "dirty map" 형성.
3. **격자쌍이 만드는 Moiré 패턴**: 공간 주파수 산술; 살짝 다른 두 주기 투과 함수의 곱이 어떻게 저주파 비트(beat)를 생성하는가.
4. **양면격자(Bigrid) 영상 (RMC 계보)**: RHESSI 부시준기 및 각 격자쌍이 하나의 Fourier 성분을 부호화하는 원리; Hurford(2013) 리뷰.
5. **반도체 X선 검출기**: 광전 흡수, 전하쌍 생성(CdTe에서 W = 4.43 eV), Ramo 정리에 의한 유도 전하, Schottky 편향 분극(polarization) 효과, 에너지 분해능 잡음원.
6. **태양 플레어 기초**: Neupert 효과, 루프 정점 vs. 발판(footpoint), 열적/비열적 분할, GOES 등급(A/B/C/M/X).

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **Bigrid imager / 양면격자 영상기** | A pair of X-ray-opaque grids (front + rear) separated by ~55 cm; each subcollimator encodes one (u,v) component as a Moiré pattern. 전후 두 텅스텐 격자가 55 cm 간격으로 놓인 영상기; 각 부시준기는 하나의 (u,v) 성분을 Moiré로 부호화. |
| **Subcollimator / 부시준기** | A {front grid window + rear grid window + detector pixel} triplet. STIX has 32 (30 imaging + 1 BKG + 1 CFL). {전방 격자창 + 후방 격자창 + 검출기 화소} 단위; STIX는 30 영상 + 1 BKG + 1 CFL = 32개. |
| **Moiré pattern / 무아레 패턴** | The slow spatial beat seen on the detector when front and rear grid transmissions multiply with slightly different pitch/orientation. 전·후 격자의 피치/방향 차이로 생기는 검출기 상의 저주파 비트 패턴. |
| **Visibility / 가시도** | Complex Fourier component V(u,v) of the source angular distribution; STIX measures Re/Im as differences of 4-pixel counts: Re=C−A, Im=D−B. 광원 각분포의 복소 Fourier 성분; STIX는 4화소 차이 C−A, D−B로 측정. |
| **CdTe (cadmium telluride) / 카드뮴 텔루라이드** | High-Z (Cd 48, Te 52) semiconductor with 5.85 g/cm³, ~65 % absorption at 100 keV in 1 mm thick Schottky-biased crystal. 고원자번호 반도체로 1 mm 두께 100 keV에서 ~65 % 흡수. |
| **Caliste-SO / 칼리스트-SO** | The detector hybrid: CdTe crystal bonded to IDeF-X HD ASIC, 12 pixels (8 large 9.6 mm² + 4 small 1.0 mm²) per detector. CdTe + IDeF-X HD ASIC 하이브리드, 8 대형 + 4 소형 화소. |
| **Caliste pixel layout (A,B,C,D) / 화소 배치** | Four "phased" large pixels per detector that sample one full Moiré period at 0°, 90°, 180°, 270°. 검출기당 0°,90°,180°,270° 위상 샘플링용 대형 화소 4개. |
| **DEM / 검출기 전자모듈** | Detector/Electronics Module — encloses CdTe array, attenuator, cold unit, IDPU. CdTe 어레이·감쇠기·냉각 유닛·IDPU 포함. |
| **Attenuator / 감쇠기** | 0.6 mm Al blade movable into beam; reduces 6 keV flux by 10⁻⁸ at 6 keV → 60 % at 20 keV. 6 keV 통과율 10⁻⁸, 20 keV 60 %로 만드는 0.6 mm Al 블레이드. |
| **CFL (Coarse Flare Locator) / 거친 플레어 위치추정기** | 32nd subcollimator with H-shaped open rear grid that gives a 2-arcmin real-time flare location across a 2°×2° FOV. H자 모양 후방 창으로 2°×2° 시야에서 2 arcmin 실시간 위치 결정. |
| **BKG (Background) subcollimator / 배경 부시준기** | Detector with 6 small apertures (0.01/0.1/1 mm²) for live, attenuator-free background. 0.01/0.1/1 mm² 6개 구멍으로 감쇠기 없이 배경 측정. |
| **Polarization (Schottky) / Schottky 분극** | Time-dependent change of the internal CdTe field under bias, partly cured by daily 0 V resets. 편향 하 CdTe 내부 전기장이 시간에 따라 변하는 효과; 일일 0 V 리셋으로 완화. |
| **¹³³Ba calibration / ¹³³Ba 보정원** | Onboard 4.5 kBq source with 31 keV and 81 keV lines for in-flight gain/offset calibration of every pixel. 31 keV·81 keV 라인으로 화소별 이득/오프셋 보정. |
| **IDPU / 기기 데이터 처리부** | Instrument Data Processing Unit: LEON3 SPARC + 100 MHz FPGA; sorts 24-bit photon words into accumulators. LEON3 SPARC + 100 MHz FPGA로 24-bit 광자어를 정렬·집계. |
| **Rate Control Regime (RCR) / 계수율 제어 영역** | One of 8 autonomous configurations (attenuator + pixel disabling + pixel cycling) that handles ~10⁴× dynamic range. 감쇠기 삽입·화소 차단·화소 순환을 결합한 8단계 자동 모드. |

---

## 5. 수식 미리보기 / Equations Preview

**English** — Five core equations anchor the paper.

(1) **Fourier-transform imaging (visibility)**: Each subcollimator measures one complex visibility,
$$ V(u,v) = \iint I(x,y)\, e^{\,i\,2\pi(ux+vy)}\,dx\,dy $$
where $I(x,y)$ is the X-ray source brightness and $(u,v)$ is the angular frequency vector set by grid pitch and orientation.

(2) **Visibility from 4-pixel counts**: With pixels A, B, C, D phased at 0°, 90°, 180°, 270° across one Moiré period,
$$ \mathrm{Re}\,V = C - A, \qquad \mathrm{Im}\,V = D - B, \qquad \mathrm{Flux} = A+B+C+D $$
The differences kill the (background + uniform-source) pedestal; the sum yields total flux. A consistency check $A + C = B + D$ flags errors.

(3) **Angular resolution from grid geometry**: Each subcollimator's resolution is
$$ \theta = \frac{p_{\mathrm{eff}}}{2 L}\,, \qquad p_{\mathrm{eff}} \in [38\,\mu\mathrm{m},\,1\,\mathrm{mm}],\ L = 0.55\,\mathrm{m} $$
giving the finest 38 μm grid → 1/(2·38 μm/0.55 m) = 7.1 arcsec.

(4) **Image reconstruction (inverse Fourier transform)**:
$$ I(x,y) = \iint V(u,v)\, e^{-i\,2\pi(ux+vy)}\, du\,dv \approx \sum_{k=1}^{30} V_k\, e^{-i\,2\pi(u_k x + v_k y)} $$
with N = 30 sampled (u,v) pairs (logarithmically spaced 1/179″ to 1/7.1″). Sparse sampling makes a "dirty map" that needs CLEAN, MEM, or Bayesian inversion (Massa et al. 2019).

(5) **CdTe charge-pair statistics & FWHM**: Number of pairs per photon $N = E/W$ with $W = 4.43$ eV; energy resolution adds Fano-limited and electronic noise in quadrature,
$$ \mathrm{FWHM}(E) = 2.355\sqrt{F\,W\,E + \sigma_{\mathrm{el}}^2\,W^2}\, \approx 1\,\mathrm{keV\ at\ 6\ keV\ for\ STIX} $$
with $F\sim0.10$ for CdTe.

**한국어** — 다섯 핵심 수식이 논문을 관통한다.

(1) **Fourier 변환 영상(가시도)**: 각 부시준기는 하나의 복소 가시도를 측정한다.
$$ V(u,v) = \iint I(x,y)\, e^{\,i\,2\pi(ux+vy)}\,dx\,dy $$
$I(x,y)$는 X선 광원 밝기 분포, $(u,v)$는 격자 피치와 방향이 결정하는 각주파수 벡터.

(2) **4-화소 계수로부터의 가시도**: 화소 A,B,C,D가 한 Moiré 주기에 걸쳐 0°,90°,180°,270°로 위상 분배되면
$$ \mathrm{Re}\,V = C - A, \qquad \mathrm{Im}\,V = D - B, \qquad \mathrm{Flux} = A+B+C+D $$
차분은 (배경+균일성분) 받침을 제거하고, 합은 총 광속을 준다. $A+C=B+D$는 일관성 점검.

(3) **격자 기하로부터의 각분해능**: 각 부시준기의 분해능은
$$ \theta = \frac{p_{\mathrm{eff}}}{2 L} $$
$p_{\mathrm{eff}}\in[38\,\mu\mathrm{m},\,1\,\mathrm{mm}]$, $L=0.55\,\mathrm{m}$. 가장 미세한 38 μm 격자가 7.1 arcsec를 제공.

(4) **영상 재구성(역 Fourier 변환)**: 30개의 (u,v) 표본으로 근사된다.
$$ I(x,y) \approx \sum_{k=1}^{30} V_k\, e^{-i\,2\pi(u_k x + v_k y)} $$
희박 샘플링은 "dirty map"을 만들어 CLEAN/MEM/Bayes 역산이 필요(Massa 외 2019).

(5) **CdTe 전하쌍 통계와 FWHM**: 광자당 전하쌍 $N=E/W$, $W=4.43$ eV. 에너지 분해능은 Fano 한계와 전자 잡음의 제곱합 합산이며 STIX는 6 keV에서 1 keV FWHM 목표.
$$ \mathrm{FWHM}(E) = 2.355\sqrt{F\,W\,E + \sigma_{\mathrm{el}}^2\,W^2} $$

---

## 6. 읽기 가이드 / Reading Guide

**English** — Suggested order:
1. **Sections 1–2** (intro, science objectives) — pin down what flare physics needs HXR diagnostics.
2. **Section 3** (instrument overview) — memorize the three-block architecture: window → imager (front/rear grids) → DEM (CdTe + electronics + attenuator).
3. **Section 4 (Imaging)** — **the heart of the paper**. Read 4.1 (grids), 4.2 (visibility math), 4.3 (image reconstruction) twice. Trace Fig. 6 carefully: simulation panel shows photons sorted into 4 phased pixels; cosine fit gives Re/Im V.
4. **Section 5 (DEM)** — focus on 5.3 CdTe physics, 5.4 ¹³³Ba calibration, 5.5 polarization & radiation damage.
5. **Section 6 (onboard data)** — skim the FPGA/IDPU description; key takeaway is the 32 energy × 32 detector × 12 pixel = 12 288 accumulator architecture.
6. **Section 7 (operations)** — Level 0–3 data products, especially the visibility (Level 2) and image (Level 3) products you will eventually use.

**한국어** — 권장 순서:
1. **1–2장(서론·과학 목표)** — 어떤 플레어 물리가 HXR 진단을 요구하는지 못박기.
2. **3장(기기 개요)** — 창 → 영상기(전·후 격자) → DEM(CdTe + 전자장치 + 감쇠기) 3블록 아키텍처를 외우기.
3. **4장(영상)** — **논문의 심장**. 4.1(격자), 4.2(가시도 수학), 4.3(영상 재구성)을 두 번 읽기. Fig. 6: 4 위상 화소로 분류된 광자, 코사인 피팅으로 Re/Im V 추출 — 이 그림을 정밀히 따라가기.
4. **5장(DEM)** — 5.3 CdTe 물리, 5.4 ¹³³Ba 보정, 5.5 분극 & 방사선 손상에 집중.
5. **6장(온보드 데이터)** — FPGA/IDPU 서술은 빠르게 훑되, 32 에너지 × 32 검출기 × 12 화소 = 12 288 누산기 구조를 핵심으로 남기기.
6. **7장(운영)** — Level 0–3 데이터 산출물, 특히 Level 2 가시도와 Level 3 영상 산출물을 숙지.

---

## 7. 현대적 의의 / Modern Significance

**English** — STIX is the only currently operating dedicated solar HXR imager (RHESSI was decommissioned in 2018) and the first flying outside ~1 AU LEO since the Yohkoh era. Its 4–150 keV imaging spectroscopy, combined with Solar Orbiter's in-situ instruments (EPD, MAG, SWA) and remote-sensing imagers (EUI, SPICE, Metis, PHI), enables — for the first time — simultaneous determination of (i) where electrons are accelerated at the Sun (HXR footpoints/looptops), (ii) the spectrum of those electrons (HXR spectral inversion), and (iii) the in-situ population that arrived at Solar Orbiter through interplanetary space. This closes the long-standing "remote vs. in-situ electron" loop. STIX's ground-only image-reconstruction pipeline (back-projection, CLEAN, MEM, Bayesian — Massa et al. 2019) is also the technological ancestor of the Chinese ASO-S/HXI imager (Zhang et al. 2019, Gan et al. 2019) launched in 2022, and informs proposed missions such as MUSE and the Solar-C/EUVST joint architecture. Visibility-based bigrid imaging is now the dominant low-resource HXR imaging paradigm.

**한국어** — STIX는 현재 가동 중인 유일한 전용 태양 HXR 영상기이며(RHESSI는 2018년 폐기), Yohkoh 이래 ~1 AU LEO 밖에서 비행하는 최초 사례이다. 4–150 keV 영상 분광 능력과 Solar Orbiter의 in-situ 기기(EPD, MAG, SWA), 원격 영상기(EUI, SPICE, Metis, PHI)의 결합은, 사상 처음으로 (i) 태양에서 전자가 가속되는 위치(HXR 발판/루프 정점), (ii) 그 전자 스펙트럼(HXR 분광 역산), (iii) 행성간 공간을 거쳐 위성에 도달한 in-situ 전자 집단을 동시에 결정할 수 있게 한다. 이는 오래된 "원격 vs. 현장 전자" 연결 고리를 닫는다. STIX의 지상 영상 재구성 파이프라인(back-projection, CLEAN, MEM, Bayes — Massa 외 2019)은 2022년 발사된 중국 ASO-S/HXI(Zhang 외 2019; Gan 외 2019)의 기술적 선조이며, MUSE·Solar-C/EUVST 등 차세대 임무 설계에 영향을 준다. 가시도 기반 양면격자 영상은 저자원 HXR 영상의 표준 패러다임으로 자리잡았다.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
