---
title: "The Visible Spectro-Polarimeter of the Daniel K. Inouye Solar Telescope"
authors: Alfred G. de Wijn, Roberto Casini, Amanda Carlile, A. R. Lecinski, S. Sewell, P. Zmarzly, A. D. Eigenbrot, C. Beck, F. Wöger, M. Knölker
year: 2022
journal: "Solar Physics, Vol. 297, Article 22"
doi: "10.1007/s11207-022-01954-1"
topic: Solar_Observation
tags: [DKIST, ViSP, spectro-polarimetry, echelle grating, polarimetric modulation, dual-beam polarimetry, first-light instrument]
status: completed
date_started: 2026-04-19
date_completed: 2026-04-19
---

# 24. The Visible Spectro-Polarimeter of the Daniel K. Inouye Solar Telescope / DKIST의 가시광 분광편광측정기

---

## 1. Core Contribution / 핵심 기여

### 한국어
이 논문은 **DKIST의 5대 first-light 기기 중 하나인 Visible Spectro-Polarimeter (ViSP)**의 설계, 광학, 변조, 제어 소프트웨어, 그리고 Science Verification 단계의 초기 과학 결과를 종합적으로 기술한다. ViSP는 **slit-scanning echelle spectrograph** 기반의 분광편광측정기로, 세 가지 독창적 특징을 갖는다:

1. **Wavelength versatility**: 380–900 nm 전체 범위에서 **어떤 파장이든 자동 재구성** 관측 가능. 기존 Fabry-Pérot 기반 기기(CRISP, TESOS)가 전용 pre-filter 때문에 가진 제한을 echelle grating + order-sorting filter set으로 극복.
2. **3개 arm 동시 관측**: 3개의 독립 camera arm이 임의의 세 파장 영역을 동시 관측 (예: Fe I 630.2 + Ca II 396.8 + Ca II 854.2). 이로써 **광구→채층→하부 코로나에 이르는 수직 tomography** 가능.
3. **고사양**: $R \gtrsim 180{,}000$, DKIST 회절 한계의 **2배 공간 분해능**(450 nm에서 0.028″까지), polarimetric **accuracy $5\times 10^{-4} I_{\text{cont}}$**, **sensitivity $10^{-4} I_{\text{cont}}$** in 10 s (630 nm에서는 **2 s** 만에 $10^{-3}$ 달성).

이 사양은 다음 기술 요소의 조합으로 달성된다 — (i) Casini & Nelson (2014)의 grating finesse 이론에 기반한 316 ℓ/mm, blaze 63.4°, order 6–14의 COTS echelle, (ii) Tomczyk et al. (2010)의 **polychromatic polarization modulator** (넓은 파장 대역에서 고효율 유지), (iii) **dual-beam polarimetry** (20:1 contrast PBS로 seeing-induced cross-talk 제거), (iv) **자동화된 3 arm reconfiguration** (curved rail + Aerotech stages로 δ = −3.3° ~ −35.3° 범위 이동), (v) **5-slit library** (0.028″–0.2″). 2021년 5월 NOAA AR 12822에 대한 Science Verification campaign에서 Fe I 630.2 nm, Ca II 396.8 nm, Ca II 854.2 nm 삼중 동시 관측으로 **sunspot의 광구–채층 3D 자기 구조**를 얻어 설계 목표 달성을 입증했다. ViSP는 DKIST의 **유일한 wavelength-versatile 분광편광측정기**로, 확정된 진단뿐 아니라 **새로운 편광 진단법 탐색을 위한 discovery instrument**로 기능한다.

### English
This paper describes ViSP — one of DKIST's five first-light instruments — covering its design, optics, polarimetric modulation, control software, and first Science Verification (SV) results. ViSP is a **slit-scanning echelle spectrograph-based spectro-polarimeter** with three distinguishing features:

1. **Wavelength versatility**: automatically reconfigurable to any wavelength across 380–900 nm, overcoming the pre-filter limitation of Fabry-Pérot instruments (CRISP, TESOS) via an echelle grating paired with a set of order-sorting filters.
2. **Three-arm simultaneous observing**: three independent, automatically-positioned camera arms observe up to three arbitrary wavelength regions at once (e.g., Fe I 630.2 + Ca II 396.8 + Ca II 854.2), enabling **vertical tomography from photosphere through chromosphere to lower corona**.
3. **High performance**: $R \gtrsim 180{,}000$, spatial resolution at 2× the DKIST diffraction limit (down to 0.028″ at 450 nm), polarimetric **accuracy $5\times 10^{-4} I_{\text{cont}}$**, and **sensitivity $10^{-4} I_{\text{cont}}$** in 10 s (achieving $10^{-3}$ in just 2 s at 630 nm).

These specs are enabled by (i) a COTS 316 ℓ/mm echelle (blaze 63.4°, orders 6–14) designed via Casini & Nelson (2014) finesse theory; (ii) a Tomczyk et al. (2010) polychromatic polarization modulator that maintains high modulation efficiency across the full band; (iii) dual-beam polarimetry with a 20:1 contrast polarizing beam splitter that removes seeing-induced cross-talk; (iv) automated three-arm reconfiguration (curved rail + Aerotech stages spanning δ = −3.3° to −35.3°); and (v) a five-slit library (0.028″–0.2″). The May 2021 SV campaign on NOAA AR 12822, simultaneously observing Fe I 630.2, Ca II 396.8, and Ca II 854.2 nm, captured the full 3D sunspot magnetic structure from photosphere to chromosphere, demonstrating on-specification performance. ViSP is the **only wavelength-versatile spectro-polarimeter at DKIST**, functioning both as a workhorse for established diagnostics and as a **discovery instrument** for exploring new polarization signatures.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction & Science Objectives (§1–§2) / 도입 및 과학 목표

**한국어**
§1은 ViSP가 DKIST 5대 기기 (VBI, ViSP, DL-NIRSP, CRYO-NIRSP, VTF) 중 유일한 **slit-based, wavelength-versatile** 기기임을 강조한다. 비교 기기들:
- **ASP** (Elmore et al. 1992): DST에서의 4-Stokes 측정 원조. ViSP의 계보 시작.
- **TESOS** (Kentischer et al. 1998): VTT, Fabry-Pérot 기반.
- **CRISP** (Scharmer et al. 2008): SST, dual Fabry-Pérot — imaging spectro-polarimetry의 golden standard. **주요 비교 대상**.
- **SPINOR** (Socas-Navarro et al. 2006): DST, echelle-based multi-line. **ViSP의 직접 선조** — ASP를 대체.
- **GRIS** (Collados et al. 2012): GREGOR, IR 대응.
- **Hinode/SP** (Lites et al. 2013): 우주 관측이지만 slit-based spectro-polarimeter.

**Fabry-Pérot vs Echelle trade-off**: CRISP/TESOS 같은 Fabry-Pérot은 2D 순간 이미지를 얻지만 wavelength마다 전용 pre-filter가 필요. ViSP는 slit-scan으로 시간 희생이 있지만 **하나의 격자로 전 파장 커버**.

§2는 과학 동기를 상세화한다:
- **Evolution of small-scale magnetism**: 조용한 태양, plage, coronal hole의 사소한 자기 구조 진화
- **Active region dynamics**: 출현, 진화, 소멸
- **Flare/CME precursors**: filament, prominence 관찰
- **Mass & energy cycle**: 광구에서 코로나로의 flow
- **Oscillations & waves**: 광구·채층 진동 모드

편광의 물리적 기원 3가지:
1. **Zeeman 효과**: 자기장이 sufficient splitting을 만들 때 (광구). Fe I 630.2 nm doublet, Fe I 524.7 & 525.0 nm가 workhorse.
2. **Hanle 효과**: 약한 자기장이 scattering polarization을 수정. $B < 1$ G 수준 검출 가능. **Zeeman으로 cancellation되는 경우**(e.g., turbulent local dynamo; Pietarila Graham, Danilovic, Schüssler 2009)에도 Hanle은 민감.
3. **Anisotropic excitation** (scattering polarization): limb 근처에서 anisotropic radiation이 atomic sub-level 불균형 유발 → "second solar spectrum" (Stenflo 1997). Sr I 460.7 nm는 광구에서조차 강한 scattering polarization.
4. **Alignment-to-orientation (A-O)**: 자기장이 유도하는 atomic level interference. Stokes V에 강도같은 신호 나타남 (Landi Degl'Innocenti & Landolfi 2004).

**English**
§1 positions ViSP among DKIST's five first-light instruments as the only slit-based, wavelength-versatile spectro-polarimeter. Peer instruments: ASP (DST, 1992) — lineage origin; TESOS (VTT, 1998) — Fabry-Pérot; CRISP (SST, 2008) — dual Fabry-Pérot imaging gold standard; SPINOR (DST, 2006) — direct echelle predecessor that replaced ASP; GRIS (GREGOR, 2012) — IR counterpart; Hinode/SP (2013) — space-based slit SP.

The core Fabry-Pérot vs. echelle trade-off: FP gives instantaneous 2D frames but requires a dedicated pre-filter per wavelength; ViSP trades instant 2D imaging for slit-scan in exchange for full-band coverage with a single grating.

§2 details the science drivers (small-scale quiet-sun magnetism, active-region evolution, flare/CME precursors, mass & energy cycle, oscillations) and the physical origins of polarization: Zeeman (photosphere), Hanle (weak fields, robust to cancellation in turbulent fields), anisotropic scattering (limb, "second solar spectrum"), and alignment-to-orientation (A-O) coupling that creates intensity-like signatures in Stokes V.

### Part II: Requirements (§3) / 요구 사양

Table 1의 핵심 사양 / Key specs from Table 1:

| Parameter | Requirement | Measured | Notes |
|---|---|---|---|
| Wavelength range | 380–900 nm | ✓ | |
| Simultaneous wavelengths | 3 | ✓ | |
| Spectral resolving power $R$ | 180,000 | ✓ | |
| Spatial FOV | 120″ × 120″ | 120″ × 78″ | limited by Arm 1 camera |
| Spatial resolution | 2× DKIST diffraction limit at all wavelengths | met for $\lambda > 450$ nm | |
| Slit scan repeatability | ± (slit width)/2 | < 1% slit width | |
| Slit scan accuracy | 0.1″ | 0.03″ | |
| Polarimetric sensitivity | $10^{-4} I_{\text{cont}}$ | $10^{-3}$ in 2 s at 630 nm | |
| Polarimetric accuracy | $5 \times 10^{-4} I_{\text{cont}}$ | | |
| Temporal resolution | 10 s to reach $10^{-3}$ for $\lambda > 500$ nm | ✓ | |
| Spectral bandpass | 1.1 nm at 630 nm | | |
| Setup time | 1 channel in 10 min | | |
| Slit move time | 200 ms between positions | | |
| Slit slew velocity | 2 arcmin in 30 s | | |

**한국어 해설**:
- 공간 분해능 "2× diffraction limit" 선택 이유: Nyquist sampling 만족 + polarimetric throughput 확보. **450 nm 미만에서는 가장 좁은 slit(17.6 μm)의 폭에 의해 제한**됨.
- Polarimetric sensitivity는 **photon-statistics limited** ($\sigma = 1/\sqrt{N}$). 630 nm 2 s에서 $10^{-3}$ → 다른 파장·조건은 scaled.
- Setup time 10 min: 기기 운영 속도를 결정.

**English**: Table 1 summarizes all specifications. The 2× diffraction-limit spatial resolution satisfies Nyquist sampling while preserving polarimetric throughput; below 450 nm the spatial resolution is instead set by the narrowest slit (17.6 μm). Polarimetric sensitivity is photon-statistics limited.

### Part III: Instrument Design (§4) / 기기 설계

#### 4.1 Feed Optics & Slit Focal Plane / 공급 광학 및 슬릿 초점면

**한국어**
- FIDO에서 들어오는 DKIST 빔을 받는 3-mirror **Schiefspiegler** (off-axis reflector) 설계. $f/\text{\#}_\text{tel} \approx 32$.
- 380 nm 회절 한계 분해능(0.024″)을 위해 17.6 μm 슬릿 필요 — 5000:1 aspect ratio (10–20 μm × 수 cm)
- 슬릿: **photo-lithography로 glass substrate에 알루미늄 반사 코팅**한 5개 aperture의 단일 기판 (±1 μm 공차)
- 5개 슬릿 폭: 17.6 / 25.7 / 33.3 μm + wider 2개 (~0.1″, 0.2″ sampling) — 공간·분광 해상도를 throughput과 trade-off
- **Fiducial hairlines 45.2″ 간격** — 3 arm 간 정렬 기준
- 슬릿 스캔: **Aerotech ANT180-260-L** translation stage (0.03″ accuracy 측정)
- Field of view image 기울기 5.4° → 반사된 슬릿 mask로 가는 빛을 **beam dump**로 보내 stray light 감소
- Plate scale: 1.6″/mm at f/32

#### 4.2 Spectrograph — Grating / 분광기 — 격자

**한국어**
격자 설계의 핵심 수식 (Casini & Nelson 2014):

**Finesse profile (Eq. 1)**:
$$
\mathcal{F}(\alpha, \beta) = \operatorname{sinc}^2\!\left[\pi \frac{L}{\lambda}(\sin\beta - \sin\alpha)\right]
$$

**Spectral dispersion (Eq. 2)**: 목표 분해능 $R$에 대한 $\beta$ 범위:
$$
\delta\beta_R = \frac{1}{R}\frac{\sin\beta - \sin\alpha}{\cos\beta}
$$

**Finesse FWHM (Eq. 3)**:
$$
\delta\beta_g = \frac{\lambda}{L\cos\beta}
$$

**Sampling 조건**: $\delta\beta_R \approx 2\delta\beta_g$

**Littrow-비슷한 조건** ($\beta = -\alpha = \varphi$):
$$
R \approx \frac{L}{\lambda}\sin\varphi = \frac{w_C}{\lambda}\tan\varphi \quad (\text{Eq. 4})
$$

**ViSP 선택**:
- Aluminum-coated echelle, **316 ℓ/mm**, blaze $\varphi = 63.4°$, **90×340 mm²**
- $\alpha = -68°$, orders $m = 6$–$14$, deviation angles $\delta = -3.3°$ to $-35.3°$
- $w_C \approx 10$ cm (collimator 제약) → $R \approx 180{,}000$ at 500 nm

**Grating polarization** (§8.1 Potential Upgrades): 낮은 차수 격자는 $\lambda \gtrsim 0.1 d$에서 **TE/TM 효율 dephasing**으로 partial polarizer 역할. 모델:
$$
S^\pm = \frac{1}{2}(1 \pm p)(I + Q)
$$
$p$는 편광 contrast. 큰 $p$는 두 빔 intensity 불균형 유발 → dual-beam cancellation 효율 저하. Mitigation:
1. **Shadow-cast grating coating** (Keller & Meltzer 1966)
2. **Quarter-wave plates** 격자 전후:
$$
S^\pm = \frac{1}{2}(I - pV \mp \sqrt{1-p^2}\,Q)
$$
→ $Q = 0$이면 두 빔 완벽 balance
3. **Dual retarder stacks** (Harrington 2021): ViSP 전 파장에서 보편적 mitigation

**English**
The grating is designed via Casini & Nelson (2014) finesse theory. For a Littrow-like configuration ($\beta = -\alpha = \varphi$):
$$
R \approx \frac{L}{\lambda}\sin\varphi = \frac{w_C}{\lambda}\tan\varphi
$$
ViSP uses an aluminum-coated 316 ℓ/mm echelle, blaze 63.4°, illuminated over $w_C \approx 10$ cm, yielding $R \approx 180{,}000$ at 500 nm. It operates in orders 6–14 with $\alpha = -68°$ and deviation angles $\delta = -3.3°$ to $-35.3°$. A known weakness — "grating polarization" from TE/TM efficiency dephasing — acts as a partial polarizer, reducing dual-beam cancellation efficiency. Three mitigations are being studied: shadow-cast coating, $\lambda/4$ plates flanking the grating, and broadband dual retarder stacks (Harrington 2021).

#### 4.3 Spectrograph — Collimator & Camera Arms / 콜리메이터 및 카메라 암

**한국어**
**Collimator**:
- Achromatic doublet, $f_{\text{coll}} = 2.37$ m, CA 90 mm, 수동 spherical + flat exit
- **$w_C \gtrsim 9$ cm 필요** (Eq. 4에서 유도, $\varphi = 63.4°$, $\lambda = 900$ nm)
- S-FPL53 + S-BSL7 (Ohara) with air gap — 열팽창 계수 차이로 cemented 불가
- 사각 단면 (격자 빔 형상 매칭)
- **Folded optical path** — slit-optic 번역 시 슬릿과 콜리메이터 거리 보존을 위해 **retroreflector**가 slit 이동 속도의 절반으로 동기 이동 (Aerotech ANT180-160-L)

**Camera arms (3 arms)**:
- Camera lens: achromatic doublet (S-LAL12 + S-BSL7), cemented, rectangular
- **$(f/\text{\#})_{\text{cam}} \approx 8$** 모든 arm 공통 (spectral dimension)
- Focal length는 arm마다 다름 (anamorphic magnification factor $r^{-1} = \cos\alpha/\cos\beta$에 맞춰): Arm 1 ~1.16, Arm 2 ~1.44, Arm 3 ~1.75 (Table 2)
- Spectrograph magnification $f_{\text{cam}}/f_{\text{coll}} \approx 0.35$ (Arm 1의 경우) — 6.5 μm Andor Zyla pixel에 매칭
- **Pixel matching** at 450 nm: 17.6 μm slit width → 1 pixel (Nyquist $\delta\beta_R \approx 2\delta\beta_g$)
- **Andor Zyla 5.5**: 2560 × 2160 pixels, 6.5 μm pitch
- FOV height along slit: Arm 1 ≤ 78″, Arm 2 ≤ 62″, Arm 3 ≤ 52″
- **Curved rail (THK HCR 65A, radius 3 m)** on which arms move via Aerotech BMS60 motors + Nexen HGP17 roller pinion (backlash-free)
- Arms 사이 최소 각도 분리: 3.55° (Arms 1-2), 4.35° (Arms 2-3)

#### 4.4 Order-Sorting Filters / 차수 분리 필터

**한국어**
차수 분리의 물리:
- 격자가 order $m$에서 파장 $\lambda$ 회절할 때, **인접 차수의 "conjugate wavelengths"**:
$$
\lambda_\pm = \frac{m}{m \pm 1}\lambda \quad (\text{Eq. 7})
$$
같은 공간 위치에 겹침.
- 필터 대역폭 조건 (0.1% 억제):
$$
|\lambda - \lambda_\pm| < \frac{\lambda}{m+1} \quad (\text{Eq. 8})
$$

ViSP는 **21개의 COTS Semrock bandpass filter** (Table 4) + **3개의 edge-blocking filter** (Table 3)로 blue/red leak 차단. 18 custom filter 원안은 비용 문제로 descoped (§8.2 Filter Jukebox에서 향후 업그레이드 항목으로 언급).

#### 4.5 Polarization Analyzer / 편광 분석기

**한국어**
- **Polarizing beam splitter (PBS)**: 20:1 contrast, analyzer efficiency ≳ 90%. **각 arm에 독립 PBS**.
- **이유 왜 arm별**: 모든 arm이 전 파장 커버해야 하므로 PBS를 spectrograph 공통 경로에 배치 불가
- **각도 수용 범위**: f/8 → ±3.8° (PBS에 과도) → relay lens로 ±2.9°로 축소 (Fig. 5)
- **PBS 재질**: Ohara S-LAH65V (n = 1.8, 380 nm 투과)
- Reflected contrast > 160:1 over most band, drops to 20:1 below 400 nm
- PBS는 slit FOV의 2 arcmin을 spectral 방향으로 split → **두 빔이 detector의 left/right half에 side-by-side**
- **Beam combiner wedge** at 67.5° angle → 두 빔 재결합하여 single detector에 기록

### Part IV: Modes of Operation (§5) / 운영 모드

**한국어**

**Polarimetric Mode**:
- **Polychromatic modulator** (Tomczyk et al. 2010)이 연속 회전 (max **5 Hz = 300 rpm**)
- 슬릿은 폴 회전이 완료되는 동안 정지 (step-and-stare)
- **Modulation cycle** = modulator의 half rotation
- **Typical 10 states per cycle** (5 states까지 가능하지만 efficiency 저하)
- Theoretical min integration = **0.1 s per modulation cycle** (5 Hz × 0.5 rotation)
- Typical integration: **~1 s (on-disk)** ~ **1 min (off-limb prominence)**
- Many modulation cycles co-added per integration
- Duty cycle 일반적으로 <30% (fast polarimetric 제외) — **0.2 s slit move time**이 제약

**Intensity Mode**:
- Slit **연속 scan**, modulator 정지
- 편광 측정 불가 — modulation states가 다른 공간 지점에 해당하므로 inconsistent
- 두 직교 편광 빔 합산으로 **unpolarized intensity** 복원
- 매우 빠른 rastering: 1.5″/s at 650 nm → 2 arcmin FOV in **80 s**
- 광시야 map cadence **100 s** vs. polarimetric mode 15 min → **9× 빠름**

### Part V: Software & Performance Calculator (§6) / 소프트웨어 및 성능 계산기

**한국어**
- **ICS (Instrument Control System)**: DKIST CSF(Common Services Framework) 기반
- **OCS (Observatory Control System)**: telescope operator가 ViSP를 telescope와 통합 제어 (Johansson & Goodrich 2012)
- 3개 카메라 + modulator → **Camera System Software + Polarization Modulator Controller**
- 엄격한 sync (slit motion, modulator rotation, camera exposure)는 **TRADS** (Time Reference And Distribution System; Ferayorni et al. 2014)가 담당

**Detailed Display plugin**: polarimetric mode에서 실시간 modulation states / 역변조된 데이터 / Stokes 맵 디스플레이

**Ancillary processing plugins**:
- **Focus plugin**: Sobel operator로 image gradient 최대 카메라 위치 결정
- **Alignment plugin**: GOS pinhole의 center-of-mass, line-grid의 Hough transform으로 방향·축척 결정
- **FITS WCS plugin**: pixel 좌표→물리 좌표 메타데이터 생성

**Instrument Performance Calculator (IPC)** (Fig. 6):
- IDL 기반 GUI
- 사용자 입력: mode, slit, grating tilt, mapping, μ-angle, exposure, arm 위치, diffraction order, binning
- **ViSP Configuration Optimizer (ICO)**: spectroscopy 라인 리스트 기반으로 **spectrograph parameter space 자동 탐색**, 최적 설정 제안. ICO는 Casini & Nelson (2014) 분석 모델 사용.
- Output: continuum SNR, spatial/spectral resolution & bandwidth, map size/duration, data volume, rate — 관측 계획 핵심 도구

### Part VI: Example Data (§7) / 예시 데이터

**한국어**
2021년 5월 8일 **Science Verification** 캠페인, NOAA **AR 12822** 관측.

**Configuration**:
- Arm 1: Fe I 630.2 nm
- Arm 2: Ca II 396.8 nm (H line)
- Arm 3: Ca II 854.2 nm

**WFC locked**, 적당한 seeing. DKIST Data Center pipeline으로 처리.

**Figure 7**: Arm 1 full bandwidth Stokes-I raw data (1.28 nm bandwidth). O₂ telluric lines, Fe I pair at 630.2 nm, temperature-sensitive Ti I, forbidden [O I] 관측.

**Figure 8**: Fe I 630.2 nm 영역, 전 4-Stokes 맵. Dark/quiet region은 Q, U가 noise-dominated, V는 매우 약함. Sunspot에서는 **선 프로파일의 성분 분리**, Q, U, V 강한 신호.

**Figure 9**: 2개 위치의 full Stokes profile:
- Sunspot (left): Q/I ~ 0.3, V/I ~ 0.4 수준의 강한 신호
- Quiet region (right): all < 0.01

**정량 결과**:
- Continuum noise: **$0.8 \times 10^{-3} I_c$** with **1.5 s integration** (per step)
- Acquisition time per step: **7.5 s** (낮은 duty cycle, Andor Zyla hardware 제약)
- Scan: **300 steps, 12.3″ coverage** with 0.041″ slit width (matching slit-step distance)

**Figure 10**: Total polarization $P_{\text{tot}}$, net linear $Q_{\text{tot}}$, preferred azimuth $\phi_r$ maps for Fe I 630.2 nm — sunspot magnetic azimuth 구조 명확.

**Figure 11**: Intensity mode로 36 scans (**34 s cadence!**) — Fe I 630.25 nm 라인 코어 vs. continuum, Ca II 396.8 nm core, Ca II 854.2 nm core. Sunspot의 광구→채층 구조 비교 가능.

### Part VII: Potential Upgrades (§8) / 잠재 업그레이드

**한국어**
1. **Grating polarization mitigation**: shadow-cast coating or $\lambda/4$ retarders (§4.2.1 논의)
2. **Filter jukebox**: 자동 order-sorting filter 교체 (원안에서 cost로 descoped)
3. **Cameras**: Andor Balor (larger pixel) 대체 어려움 (optics 재설계 비용). 향후 4k×4k @ 6.5 μm 가능 시 Andor Zyla 교체

### Part VIII: Conclusions (§9) / 결론

**한국어**
- ViSP는 DKIST의 **유일한 wavelength-versatile 분광편광측정기** (380–900 nm)
- 3 arm 자동 포지셔닝 + broad-dispersion grating + fringe-free polychromatic modulator → **사실상 무한한 스펙트럼 조합** 접근
- Polarimetric SNR 1000 달성 시간 on-disk target: **5 s 미만** (630 nm peak QE)
- 최대 공간 분해능: **0.028″** (가장 narrow slit)
- 5-slit library로 과학 목표 맞춤화
- SV(2021.05) 완료, 2021.11 DKIST 첫해 commissioning science 시작 가능 상태

---

## 3. Key Takeaways / 핵심 시사점

1. **Echelle + order-sorting filters가 Fabry-Pérot의 "pre-filter per line" 제약을 깨뜨린다 / Echelle with order-sorting filters breaks the Fabry-Pérot "pre-filter per line" constraint** — CRISP는 관측 라인을 바꾸려면 전용 interference filter 설치가 필요하다. ViSP는 **grating만 기울이고 arm만 움직이면** 380–900 nm 어떤 조합이든 자동 설정 가능. 이는 "설계된 관측"에서 "탐색적 관측"으로 패러다임 전환을 가능케 한다. COTS 격자 + COTS Semrock 필터 library라는 **"engineering 실용주의"**가 핵심.

2. **3-arm 동시 관측이 수직 tomography를 현실화한다 / Three-arm simultaneous observing realizes vertical tomography** — Fe I 630.2 (photosphere) + Ca II 396.8 (mid-chromosphere) + Ca II 854.2 (chromosphere upper) 동시 관측으로 **같은 공간 지점의 서로 다른 높이**의 자기·열역학 정보를 **시간 동기화** 상태에서 획득. 이는 기존 sequential 관측으로는 spicule·wave 전파 분석 시 심각한 artifact를 만들었던 부분을 근본 해결.

3. **Grating finesse 이론 (Casini & Nelson 2014)이 설계 전반을 구조화한다 / Grating finesse theory structures the design** — $R \approx (w_C/\lambda)\tan\varphi$ 공식이 $w_C \gtrsim 9$ cm → collimator 크기 → camera lens 크기 → optical table 크기 → 전체 기기 크기의 연쇄적 제약을 만든다. 광학 기기 설계에서 **단일 방정식이 전체를 결정**하는 드문 사례.

4. **Dual-beam polarimetry + polychromatic modulator 조합이 seeing-induced cross-talk을 제거한다 / Dual-beam + polychromatic modulator eliminates seeing-induced cross-talk** — Casini, de Wijn & Judge (2012)의 분석에 따르면 가장 문제가 되는 오류원은 **변조 주기보다 짧은 시간 스케일**의 atmospheric seeing 변동으로 intensity→Q/U/V 누출. 두 빔 차분은 이 신호를 cancellation하고, polychromatic modulator (Tomczyk 2010)는 **파장 대역 전체에서 balanced modulation** 유지해 calibration error 최소화. **하드웨어·소프트웨어 공동 설계**의 성공 예.

5. **Grating polarization은 ViSP의 아킬레스건 / Grating polarization is ViSP's Achilles heel** — 낮은 차수 grating은 TE/TM 효율 차이 + phase shift로 **부분 polarizer처럼 작동**. $S^\pm = (1 \pm p)(I + Q)/2$ 모델에서 $p$가 크면 dual-beam cancellation 효율 저하. Shadow-cast coating, $\lambda/4$ plates, dual retarder stack 세 가지 mitigation 연구 중 — 이는 **현재 설계의 한계를 솔직하게 인정**하는 논문의 미덕.

6. **Intensity mode가 polarimetric mode보다 9배 빠르다는 것은 운영 계획의 본질이다 / Intensity mode being 9× faster than polarimetric mode is central to operational planning** — 같은 FOV (2 arcmin) polarimetric에서 15 min vs. intensity에서 100 s. 이는 동역학적 이벤트(flare precursor, wave propagation)는 intensity mode로, 정적 자기장 측정은 polarimetric mode로 분리해야 함을 시사. Proposal 작성 시 **mode trade-off가 곧 science trade-off**.

7. **Science Verification 결과가 specs를 증명한다 / Science Verification results validate the specs** — SV 캠페인의 단일 measurement가 **0.8 × 10⁻³ Ic at 1.5 s integration** in 630 nm → sensitivity spec ($10^{-4}$ in 10 s)과 duty cycle을 고려하면 일관됨. 즉 ViSP는 design → fabrication → commissioning → verified performance의 전 과정을 **2022년 시점에서 이미 증명**. 이는 DKIST Critical Science Plan(Rast et al. 2021)의 과학 목표가 **현실화 가능**함을 의미.

8. **ViSP는 DKIST 시대의 "open-endedness" 철학의 상징 / ViSP symbolizes DKIST's "open-endedness" philosophy** — 5대 first-light 기기 중 VBI, VTF, DL-NIRSP, CRYO-NIRSP는 특정 과학 목표에 최적화. ViSP만 **research instrument** 성격으로, 예측되지 않은 과학을 위해 설계. 44년 운영 동안 태양 물리학의 **예상 못한 새로운 진단**이 등장할 것을 전제한 설계 — 장수명 시설의 지혜.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Grating equation / 격자 방정식

$$
m\lambda = d(\sin\alpha + \sin\beta)
$$

- $m$: diffraction order / 회절 차수
- $d$: groove spacing / 격자 간격 (ViSP: $d = 1/316$ mm $\approx 3.16$ μm)
- $\alpha, \beta$: 입사각 / 회절각 (격자 법선 기준)
- ViSP: $\alpha = -68°$, $m = 6$–$14$

### 4.2 Grating finesse profile (Eq. 1)

$$
\mathcal{F}(\alpha, \beta) = \operatorname{sinc}^2\!\left[\pi \frac{L}{\lambda}(\sin\beta - \sin\alpha)\right]
$$

$L$: 조명된 격자의 너비 / illuminated grating width.
FWHM이 spectral resolution 결정. $L$ 클수록 샤프.

### 4.3 Dispersion interval for target $R$ (Eq. 2)

$$
\delta\beta_R = \frac{1}{R}\frac{\sin\beta - \sin\alpha}{\cos\beta}
$$

### 4.4 Finesse FWHM in $\beta$ (Eq. 3)

$$
\delta\beta_g = \frac{\lambda}{L\cos\beta}
$$

### 4.5 Resolving power (Eq. 4, Littrow-like)

$\beta = -\alpha = \varphi$에서:
$$
R \approx \frac{L}{\lambda}\sin\varphi = \frac{w_C}{\lambda}\tan\varphi
$$

$w_C$: 격자 투영 너비.

**수치 대입**: $w_C = 10$ cm, $\varphi = 63.4°$, $\lambda = 500$ nm:
$$
R = \frac{0.10}{5 \times 10^{-7}}\tan(63.4°) = 2 \times 10^5 \cdot 2.0 = 4 \times 10^5
$$
(slit width, anamorphic factor 고려 후 실제 $R \approx 180{,}000$)

### 4.6 Grating efficiency in scalar diffraction (Eq. 5)

$$
I(\alpha, \beta) = \operatorname{sinc}^2\!\left[\pi \frac{d}{\lambda}\cos\alpha \,\frac{\sin(\beta - \varphi) - \sin(\alpha + \varphi)}{\cos(\alpha + \varphi)}\right]
$$

**Usable $\beta$ range** (Eq. 6):
$$
\Delta\beta_g = \frac{\lambda}{d\cos\alpha}\frac{\cos(\alpha + \varphi)}{\cos(\beta - \varphi)}
$$

ViSP: $\Delta\beta_g \approx 20°$ → groove density $> 300$ ℓ/mm 필요 for $\lambda = 500$ nm, $\varphi \approx 60°$.

### 4.7 Order overlap wavelengths (Eq. 7)

$$
\lambda_\pm = \frac{m}{m \pm 1}\lambda
$$

인접 차수가 같은 공간 위치에 겹치는 파장.

### 4.8 Order-sorting filter bandwidth (Eq. 8)

$$
|\lambda - \lambda_\pm| < \frac{\lambda}{m+1}
$$

필터 half-bandwidth가 이보다 작아야 $10^{-3}$ 수준 차수 억제.

### 4.9 Polarization measurement / 편광 측정

시간 순차 intensity 측정:
$$
\vec{I}_{\text{meas}}(t) = \mathbf{O}(t) \cdot \vec{S} + \vec{n}
$$

Demodulation:
$$
\vec{S} = \mathbf{O}^{-1} \cdot \vec{I}_{\text{meas}}
$$

**Ideal balanced modulator**: $\epsilon_Q = \epsilon_U = \epsilon_V = 1/\sqrt{3} \approx 0.577$.

### 4.10 Grating polarization model / 격자 편광 모델

Partial $Q$-polarizer (contrast $p \in [0,1]$):
$$
S^\pm = \frac{1}{2}(1 \pm p)(I + Q)
$$

**Mitigation with $\lambda/4$ plates**:
$$
S^\pm = \frac{1}{2}\left(I - pV \mp \sqrt{1-p^2}\,Q\right)
$$

### 4.11 Anamorphic magnification / 비대칭 배율

격자 후의 스펙트럼 dimension 축소 factor:
$$
r = \frac{\cos\alpha}{\cos\beta}
$$

ViSP arms scale (Table 2): $r^{-1} \approx 1.16, 1.44, 1.75$.

Camera focal lengths scale inversely to preserve pixel matching.

### 4.12 Photon-limited polarimetric accuracy / 광자 한계 편광 정확도

$$
\sigma_{\text{pol}} = \frac{1}{\sqrt{N}}
$$

$10^{-4}$ in 10 s ⇒ $N \gtrsim 10^8$ photons/resolution element/modulation state — achievable by 4 m DKIST + ~50% throughput.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
태양 분광편광측정(spectro-polarimetry)의 발전사
──────────────────────────────────────────────────────────────────────────

1897  Zeeman        Zeeman 효과 발견 (자기장 → 스펙트럼선 분열)
1924  Hanle         Hanle 효과 발견 (약한 자기장 → 산란 편광 변화)
1940s-60s           Stokes 파라미터 개념 (Chandrasekhar)
1973  Stenflo       조용한 태양 자기장 1st detections
1982  Stenflo       Hanle 관측의 실용화
1992  Elmore+       ASP at DST — modern 4-Stokes의 기원 ◆
1997  Stenflo       "Second solar spectrum" (limb polarization atlas)
1998  Kentischer    TESOS (VTT) — Fabry-Pérot 시작
2000  del Toro/Coll Optimum modulation matrix 이론
2003  Scharmer      SST 1 m 첫 빛
2004  Landi/Landolfi 종합 참고서 "Polarization in Spectral Lines"
2005  Rimmele/SWG   ATST Science Requirements Document
2006  Socas-Navarro SPINOR at DST — echelle multi-line (ViSP 직접 선조) ◆
2008  Scharmer+     CRISP at SST — dual Fabry-Pérot golden standard ◆
2010  Tomczyk+      Polychromatic modulator 이론
2011  Rimmele/Marino Solar AO review
2012  Collados+     GRIS at GREGOR (IR)
2013  Lites+        Hinode/SP 10-year review
2014  Casini/Nelson Grating finesse 이론 (ViSP 설계 기반)
2017  Harrington    DKIST polarization modeling part 1
2018  Casini+       Shadow-cast grating 실측
2019.12           DKIST first light
2020  Rimmele+    DKIST Overview (논문 #23)
2020  Harrington+ DKIST polarization modeling parts 5-7
 ▼
═════════════════════════════════════════════════════════════════════════
2022: ViSP paper (de Wijn et al., 이 논문)  ◆ DKIST 시대 분광편광측정의 기준
      + May 2021 SV campaign (AR 12822)
═════════════════════════════════════════════════════════════════════════
 ▼
2022  Jaeggli+   DL-NIRSP paper (논문 #25, 코로나 자기장 IR SP)
2022+ OCP        Operations Commissioning Phase 시작
2024  Science    과학 관측 본격 가동
2029  EST        유럽 4 m 태양 망원경 — ViSP-like 기기 예정
2030+ MCAO       DKIST MCAO 업그레이드

◆ = ViSP의 직접 선조 / 핵심 비교 대상
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **#23 Rimmele et al. 2020** (DKIST Overview) | ViSP는 이 논문에서 간략 소개된 기기의 상세 기술. FIDO, Polarimetry calibration 섹션이 이 ViSP 논문에 연결 | ViSP의 overall context 제공; Mueller matrix calibration의 전체 구조가 Rimmele 2020에 있음 / Provides the overall observatory context and polarimetry framework |
| **#22 Scharmer et al. 2008** (CRISP) | Fabry-Pérot imaging spectro-polarimeter의 표준 — ViSP는 이와 **상보적** (echelle vs. FP, slit vs. imaging). CRISP는 2D 영상 but FP pre-filter 필요; ViSP는 slit-scan but wavelength-versatile | ViSP의 **설계 선택 근거**를 이해하려면 CRISP의 한계 이해 필수 / Essential for understanding ViSP's design choices — CRISP shows the complementary strength (2D imaging) and weakness (per-line pre-filter) |
| **#20 Rimmele & Marino 2011** (Solar AO) | DKIST AO가 ViSP에 AO-corrected beam 제공. Section 1에서 언급된 "adaptive optics can effectively correct for seeing aberrations (approximately $\lambda \geq 500$ nm)"가 ViSP spatial resolution spec을 결정 | ViSP의 450 nm 이하 spatial resolution 한계의 **근본 원인** / Root cause of ViSP's spatial-resolution limit below 450 nm |
| **#21 Wöger et al. 2008** (Speckle interferometry) | ViSP의 **intensity mode**는 speckle reconstruction과 결합해 dynamic event 관측 가능. Intensity mode의 1.5″/s scan speed가 speckle burst와 compatible | Intensity mode의 post-processing 파이프라인 / Post-processing pipeline for intensity mode |
| **#25 Jaeggli et al. 2022** (DL-NIRSP) | 같은 DKIST Topical Collection의 기기 논문. ViSP(visible) + DL-NIRSP(IR) 조합이 **전 파장 분광편광 커버리지** 완성 | DKIST 분광편광 기기 세트의 상호 보완 / Complementary coverage within DKIST spectro-polarimetry suite |
| **Harrington et al. 2020** (DKIST polarization parts 5–7) | ViSP의 polarimetric accuracy $5\times 10^{-4}$는 Harrington의 전체 시스템 Mueller matrix modeling 작업 위에 성립 | ViSP 사양의 **실현 가능성의 근거** / Basis for ViSP's polarimetric-accuracy achievability |
| **Tomczyk et al. 2010** (Polychromatic modulator) | ViSP modulator의 설계 이론 | ViSP의 핵심 하드웨어 구성 요소의 이론적 근거 / Theoretical basis for a key hardware component |
| **Casini & Nelson 2014** (Grating finesse) | ViSP 격자·콜리메이터·카메라 arm 설계의 **수학적 뼈대**. Eq. 1–6 전부 이 논문에서 유래 | ViSP 설계의 수학적 기원 / Mathematical origin of the ViSP design |
| **Socas-Navarro et al. 2006** (SPINOR) | ViSP의 **직접 선조** — DST의 ASP를 대체한 echelle-based multi-line SP | 설계 계보의 직접 연결 / Direct design lineage |

---

## 7. References / 참고문헌

### Primary paper / 본 논문
- de Wijn, A. G., Casini, R., Carlile, A., Lecinski, A. R., Sewell, S., Zmarzly, P., Eigenbrot, A. D., Beck, C., Wöger, F., Knölker, M., "The Visible Spectro-Polarimeter of the Daniel K. Inouye Solar Telescope", *Solar Physics*, Vol. 297, Article 22 (2022). [DOI: 10.1007/s11207-022-01954-1]

### Critical ViSP design references
- Casini, R. & Nelson, P. G., "On the intensity distribution function of blazed reflective diffraction gratings", *J. Opt. Soc. Am. A* 31, 2179 (2014). — ViSP 격자 설계 수학의 근본.
- Casini, R. & de Wijn, A. G., "On the instrument profile of slit spectrographs", *J. Opt. Soc. Am. A* 31, 2002 (2014).
- Casini, R., de Wijn, A. G., Judge, P. G., "Analysis of seeing-induced polarization cross-talk and modulation scheme performance", *ApJ* 757, 45 (2012).
- Casini, R., Gallagher, D., Cordova, J. V. M., Morgan, M., "Measured performance of shadow-cast coated gratings for spectro-polarimetric applications", *Proc. SPIE* 2018.
- Tomczyk, S., et al., "Polychromatic polarization modulator" (2010). — ViSP modulator의 이론적 근거.

### DKIST-level references
- Rimmele, T. R., et al., "The Daniel K. Inouye Solar Telescope – Observatory Overview", *Solar Physics* 295, 172 (2020). — 이 프로젝트의 flagship 논문.
- Rast, M. P., et al., "Critical Science Plan for the DKIST", 2021.
- Harrington, D. M., Sueoka, S. R., "Polarization modeling and predictions for DKIST part 1", *J. Astron. Telesc. Instrum. Syst.* 3, 018002 (2017).
- Harrington, D. M., Jaeggli, S. A., Schad, T. A., White, A. J., Sueoka, S. R., "Polarization modeling and predictions for DKIST part 6", *J. Astron. Telesc. Instrum. Syst.* 6, 038001 (2020).
- Ferayorni, A., et al., "DKIST controls model for synchronization of instrument cameras, polarization modulators, and mechanisms", *Proc. SPIE* CS-9152 (2014).

### Peer/precursor instruments
- Elmore, D. F., et al., "The advanced Stokes polarimeter — A new instrument for solar magnetic field research", *Proc. SPIE* 1746, 22 (1992). — ASP at DST.
- Socas-Navarro, H., et al., "SPINOR at DST", 2006. — **ViSP 직접 선조**.
- Scharmer, G. B., et al., "CRISP at SST", 2008. — FP imaging SP golden standard.
- Kentischer, T. J., et al., "TESOS: double Fabry-Pérot instrument for solar spectroscopy", *A&A* 340, 569 (1998).
- Collados, M., et al., "GRIS: The GREGOR infrared spectrograph", *Astron. Nachr.* 333, 872 (2012).
- Lites, B. W., et al., "Hinode Spectro-Polarimeter", 2013.

### Solar polarization theory
- Zeeman, P. (1897). — Zeeman 효과의 원조.
- Hanle, W., "Über magnetische Beeinflussung der Polarisation der Resonanzfluoreszenz", *Z. Phys.* 30, 93 (1924).
- Stenflo, J. O., "Second solar spectrum" (1997).
- Landi Degl'Innocenti, E. & Landolfi, M., *Polarization in Spectral Lines*, Kluwer, 2004. — 표준 참고서.
- del Toro Iniesta, J. C., *Introduction to Spectropolarimetry*, Cambridge Univ. Press, 2003.
- del Toro Iniesta, J. C., Collados, M., "Optimum modulation and demodulation matrices for solar polarimetry", *Appl. Opt.* 39, 1637 (2000).

### Supporting
- Leenaarts, J., "Non-LTE radiative transfer in the solar atmosphere", review, 2020.
- Pietarila Graham, J., Danilovic, S., Schüssler, M., "Turbulent small-scale dynamo field strength", *ApJ* 2009.
- Bianda, M., Berdyugina, S., et al., "Spatial variations of Sr I 460.7 nm scattering polarization peak", *A&A* 2018.
- Lites, B. W., et al., "Probable identification of the on-disk counterpart of spicules in Hinode Ca II H filter imaging data", *ApJL* 757, L17 (2012).
