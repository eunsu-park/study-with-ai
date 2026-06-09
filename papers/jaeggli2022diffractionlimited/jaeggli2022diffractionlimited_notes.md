---
title: "The Diffraction-Limited Near-Infrared Spectropolarimeter (DL-NIRSP) of the Daniel K. Inouye Solar Telescope (DKIST)"
authors: ["Sarah A. Jaeggli", "Haosheng Lin", "Peter Onaka", "Hubert Yamada", "Tetsu Anan", "Morgan Bonnet", "Gregory Ching", "Xiao-Pei Huang", "Maxim Kramar", "Helen McGregor", "Garry Nitta", "Craig Rae", "Louis Robertson", "Thomas A. Schad", "David M. Harrington", "Mary Liang", "Myles Puentes", "Predrag Sekulic", "Brett Smith", "Stacey R. Sueoka", "Paul Toyama", "Jessica Young", "Chris Berst"]
year: 2022
journal: "Solar Physics"
doi: "10.1007/s11207-022-02062-w"
topic: Solar_Observation
tags: [DKIST, DL-NIRSP, spectropolarimetry, integral-field, near-infrared, BiFOIS, coronal-magnetometry, instrument-paper]
status: completed
date_started: 2026-04-19
date_completed: 2026-04-19
---

# 25. The Diffraction-Limited Near-Infrared Spectropolarimeter (DL-NIRSP) of DKIST / DKIST의 회절한계 근적외선 분광편광기 DL-NIRSP

---

## 1. Core Contribution / 핵심 기여

**English**
DL-NIRSP is the first fiber-fed, integral-field spectropolarimeter installed as a facility instrument on a 4 m-class solar telescope. It is one of the five first-light instruments at the Daniel K. Inouye Solar Telescope (DKIST), designed specifically to capture *simultaneous* 2D polarized spectra of magnetically sensitive lines spanning the photosphere, chromosphere, and corona. The instrument combines three innovations on a single optical train: (1) two novel *Birefringent Fiber-Optic Image Slicers* (BiFOIS-36 and BiFOIS-72), which reformat a rectangular 2D field into four vertical slits while preserving linear polarization; (2) a reconfigurable, all-reflecting near-Littrow spectrograph with three spectral arms (500–900, 900–1350, and 1350–1800 nm) that achieves resolving power $R > 10^{5}$ and allows simultaneous multi-wavelength observations; and (3) a dual-beam (Wollaston-prism) polarimetric analysis chain with a rotating polycarbonate modulator targeting continuum-referenced polarimetric accuracy of $5\times10^{-4}$. The feed optics provide three spatial-sampling modes — 0.03″ (High-Res, f/62), 0.077″ (Mid-Res, f/24), and 0.464″ (Wide-Field, f/8) — with field-scanning mosaic coverage up to $2'\times 2'$. First-light observations in 2021 August verified that DL-NIRSP can (a) map a photospheric pore simultaneously in Fe I 630.2, Si I 1082.7, and Fe I 1564.9 nm in agreement with SDO/HMI and (b) detect forbidden coronal lines [Fe XI] 789.2, [Fe XIII] 1074.7, and (notably) [Si X] 1430.0 — a line historically hard to observe from the ground — at 1.2 $R_\odot$ above an active region.

**한국어**
DL-NIRSP는 4 m 급 태양망원경에 정규 관측기기로 설치된 최초의 광섬유 기반 적분영상분광편광기(integral-field spectropolarimeter)이다. DKIST의 5개 첫세대(first-light) 기기 중 하나로, **광구·채층·코로나의 자기장 진단선을 2차원으로 동시 관측**할 수 있도록 전용 설계되었다. 본 논문은 세 가지 기술 혁신을 하나의 광학계에 통합한다: (1) 두 가지 BiFOIS(**B**irefringent **Fi**ber-**O**ptic **I**mage **S**licer) IFU — BiFOIS-36(36 μm 화소, 고분해/중분해)과 BiFOIS-72(72 μm 화소, 중분해/광역) — 가 직사각형 2D 시야를 네 개의 수직 슬릿으로 재구성하면서 선형편광을 보존한다. (2) 재구성 가능한 전반사 near-Littrow 분광기가 500–900 nm, 900–1350 nm, 1350–1800 nm 세 암(arm)으로 $R > 10^{5}$의 분해능과 다파장 동시 관측을 제공한다. (3) Wollaston 프리즘 기반 dual-beam 편광 분석 체계와 5층 폴리카보네이트 회전 변조기가 연속체 대비 $5\times10^{-4}$ 편광 정확도를 목표한다. 피드광학은 **세 가지 공간 샘플링 모드**(0.03″/0.077″/0.464″)를 제공하며 최대 $2'\times 2'$ 시야를 모자이크로 커버한다. 2021년 8월 첫세대 관측은 (a) 광구 기공(pore)을 Fe I 630.2, Si I 1082.7, Fe I 1564.9 nm 세 파장에서 동시 매핑하고 SDO/HMI와 일치함을 확인했으며, (b) 1.2 $R_\odot$ 활동영역 상공에서 [Fe XI] 789.2, [Fe XIII] 1074.7, 지상관측이 극히 어려웠던 [Si X] 1430.0 nm 금지선을 검출하는 데 성공했다.

---

## 2. Reading Notes / 읽기 노트

### Part I: §1 Introduction — DL-NIRSP within DKIST / DKIST 속 DL-NIRSP의 위치

**English**
DKIST (Rimmele et al. 2020, paper #23 in this series) is a 4 m off-axis Gregorian with a high-order adaptive optics (HOAO) system. Its coudé lab houses five first-light instruments fed by the *Facility Instrument Distribution Optics* (FIDO) — a tree of dichroic beam-splitters that routes light between VBI (Visible Broadband Imager; Wöger et al. 2021), ViSP (Visible Spectropolarimeter; de Wijn et al. 2022, paper #24), CryoNIRSP (Cryogenic NIR Spectropolarimeter; Fehlmann 2022), VTF (Visible Tunable Filter; von der Lühe et al. 2022), and DL-NIRSP. What distinguishes DL-NIRSP is **integral-field spectroscopy**: it is the only DKIST instrument that records the full $(x,y,\lambda,\text{Stokes})$ datacube at every instant, without scanning a slit across the scene. The scientific argument (§2) rests on five drivers: high spatial resolution (to resolve kG flux tubes <100 km), polarization sensitivity (to see $B$-induced signals at $10^{-4}$ of continuum in the chromosphere), high cadence (to capture 3-minute chromospheric and dynamic-event timescales), multi-wavelength simultaneity (to stitch 3D atmospheric structure), and coronal spectropolarimetry (to constrain the B-vector off-limb).

**한국어**
DKIST(#23)은 HOAO를 갖춘 4 m 비축 그레고리안 망원경이다. 코우데 랩의 FIDO 다이크로익 트리가 5개 첫세대 기기(VBI, ViSP, CryoNIRSP, VTF, DL-NIRSP)에 빛을 분배하며, 그 중 DL-NIRSP만이 **적분영상분광(Integral-Field Spectroscopy)** — 슬릿 스캔 없이 $(x,y,\lambda,\text{Stokes})$ 를 순간에 기록 — 을 수행한다. §2의 과학 동인 다섯 가지: 고공간분해(<100 km 자기튜브 해상), 편광 감도(채층 $10^{-4}$ 수준 신호), 고시간분해(3분 진동·동적 이벤트), 다파장 동시성(3D 대기 스티칭), 코로나 편광측정(림 외곽 B-벡터 제약).

### Part II: §2 Scientific Objectives & §3 Elements — 왜, 무엇을, 어떻게 / Why, What, How

**English**
Table 1 of the paper compares *goals vs. as-built*:

| Parameter | Goal | As-Built |
|---|---|---|
| Spectral coverage | 900–2500 nm | **500–1800 nm** |
| Resolving power | 50,000–200,000 | **>105,000** |
| Spatial sampling | 1.22 λ/2D at 900 nm | TBD (verified at diffraction limit) |
| FOV | 2.8′ round, 2′ square | Same |
| Polarimetric accuracy | $5\times10^{-4}$ | TBD |
| Modulation cadence | 0.1 s | 0.3 s |
| Modulation efficiency | >0.1 | **>0.4** |
| Multi-wavelength | 3 channels | Same |
| Simultaneous AO-corrected operation with VBI/ViSP/VTF | Yes | Yes; also CryoNIRSP after FIDO upgrade |

The wavelength coverage was shifted blueward of the design to include the 500–900 nm visible range, a major change. The eleven spectral lines currently targeted (Table 2) sample from $\log T_\text{max} = 6.3$ coronal forbidden lines ([Fe XIV] 530.3, [Fe XI] 789.2, [Fe XIII] 1074.7/1079.8, [Si X] 1430.0) through chromospheric lines (He I D3 587.6, Ca II 854.2, He I 1083) down to photospheric Fe I 630.2 and Fe I 1565.0.

The optical layout of DL-NIRSP has two big halves: **feed optics** and **spectrograph**.

*Feed optics* (§3.1, Figure 3): All-reflecting silver-coated mirrors reimage the DKIST coudé beam at one of three plate scales. The feed is organized around a Field Scanning Mirror (FSM) — a 220 mm spherical piezo-tip/tilt mirror steered with 0.3 μrad (0.006″) precision. Downstream of the FSM two paths exist:
- **Mid-Res / Wide-Field path** (f/24, shown in red trace in Fig. 3): through an f/24 off-axis prolate ellipsoid, reflecting off Fold Mirror 4 to the final focal plane. For coronal work an f/8 coronal triplet lens is inserted in front of the IFU, reducing the effective focal ratio.
- **High-Res path** (f/62, blue trace): the f/62 off-axis mirror is slid *into* the beam on a linear stage, and an on-axis oblate ellipsoid (f/62 Convex Mirror) gently lowers the F/#. Together MF62-1 and MF62-2 form a Dall–Kirkham variant.
Feed optics parameters (Table 3): seven powered or folding mirrors with radii of curvature 2.8 m to 9.6 m and conic constants spanning –1.376 to +2.832 (prolate ellipsoids to oblate ellipsoids).

*Spectrograph* (§3.2, Figures 4–5): An all-reflecting off-axis **Littrow** design. Light from the IFU exit slits is telecentric at f/12 and Gaussian. It bounces off Spectrograph Fold Mirror 1, strikes the **Spectrograph Off-Axis Mirror (OAM)** — an off-axis hyperboloid acting as the collimator — then diffracts off a **23.2 lines/mm, 63° blaze** grating at ≈1250 mm focal length. The dispersed beam returns to the OAM for refocusing to an intermediate focal plane at the front of each spectral arm. The grating angle is nominally 5° off Littrow.
After the OAM, two dichroic **beam-splitters** route light to three spectral arms:
- BS-1 (dichroic 950 nm long-pass): reflects <900 nm to Arm 1.
- BS-2 (dichroic 1350 nm long-pass): reflects 900–1350 nm to Arm 2; transmits 1350–1800 nm to Arm 3.
Each arm then has an intermediate focal plane, collimator triplet, narrow-band filter ($\lambda/\Delta\lambda \approx 750$), spectral mask (a field stop with four slits matching the IFU exit slits — creates the "dark gaps" between IFU slit spectra visible in Figure 6), **Wollaston prism** (34° internal wedge angle, quartz crystal, splits into orthogonal linear polarizations), camera triplet lens, and finally the detector. The spectral mask+narrow-band filter combination yields an effective spectral bandwidth $\lambda/\Delta\lambda \approx 1250$.

Critically, the spectrograph is **reconfigurable**: the grating rotates to change Littrow angle, and each spectral arm's optics are mounted on a translation stage to center the chosen bandpass. A custom *Spectrograph Calculator* (§3.6.4, Figure 9) assists in choosing valid three-wavelength combinations; Table 6 enumerates 20 tested/candidate combinations.

**한국어**
**설계 목표 vs. 실측치**: 목표가 900–2500 nm이었으나 실제로는 500–1800 nm로 파장대가 청색편이되어 가시광 영역을 포함하게 되었다(큰 변경). 분해능은 목표를 초과하여 R > 105,000, 변조 효율도 목표 0.1을 크게 넘어 0.4 이상을 달성했다. 변조 주기는 0.1 s 목표가 0.3 s로 다소 느려진 상태.

**11개 관측 파장**(Table 2): 코로나 금지선(log T_max=6.3) [Fe XIV] 530.3, [Fe XI] 789.2, [Fe XIII] 1074.7/1079.8, [Si X] 1430.0부터 채층선 (He I D3 587.6, Ca II 854.2, He I 1083)을 거쳐 광구 Fe I 630.2, Fe I 1565.0까지.

**피드광학**(§3.1): 전반사 은코팅 거울들. 중심에 있는 것이 **Field Scanning Mirror (FSM)** — 220 mm 구면 거울을 피에조 tip/tilt로 0.3 μrad (= 0.006″) 정밀도로 조종. FSM 이후 두 광학 경로:
- **Mid-Res/Wide-Field 경로** (f/24): f/24 비축 프롤레이트 타원체 거울 사용. 코로나 관측 시 IFU 앞에 f/8 코로나 triplet 렌즈를 삽입.
- **High-Res 경로** (f/62): f/62 거울을 리니어 스테이지로 광로에 삽입, on-axis 오블레이트 타원체가 f/# 를 완만하게 낮춤 (Dall–Kirkham 변형).

**분광기**(§3.2): 전반사 비축 near-Littrow 설계. IFU 출사슬릿의 텔레센트릭 f/12 빔이 Fold Mirror 1 → **비축 쌍곡면 OAM** (collimator 역할, 초점거리 ≈1250 mm) → **회절격자 (23.2 선/mm, blaze 63°)** → OAM → 중간초점 → 각 암 광학계로 진행. 두 다이크로익 빔스플리터가 <900 nm / 900–1350 nm / 1350–1800 nm 로 분배. 각 암은 narrow-band 필터(λ/Δλ≈750)와 **spectral mask**(IFU 출사슬릿과 대응되는 네 슬릿 field stop; 검출기상에서 슬릿 간 "빈 공간"을 만들어 Wollaston 출력을 분리), Wollaston 프리즘(34° quartz), camera triplet, 검출기로 구성된다. 분광기는 **재구성 가능**하다 — 격자 회전 + 각 암의 translation stage. *Spectrograph Calculator* 가 Table 6의 20가지 조합을 관리.

### Part III: §3.3 BiFOIS — 광섬유 이미지 슬라이서의 심장 / Heart of the Fiber-Optic IFU

**English**
The Birefringent Fiber-Optic Image Slicer (BiFOIS; Lin & Versteegh 2006; Schad et al. 2014) is the scientific heart of DL-NIRSP. It is an assembly of fiber-optic **ribbons** — flat fiber bundles with rectangular cores — that physically transport light from a densely packed 2D "imaging array" (at the feed focal plane) to four vertical "slit arrays" (at the spectrograph entrance). Key properties (Table 5):

| Property | BiFOIS-36 | BiFOIS-72 |
|---|---|---|
| Core (μm) | 29 × 5 | 29 × 5 |
| Cladding thickness (μm) | 3.5 | 3.5 |
| Ribbon format | 1 × 90 cores | 2 × 180 cores |
| Input format | 64 × 2 ribbons | 32 × 2 ribbons |
| Exit slit height (mm) | 36 | 36 |
| Exit slit avg spacing (mm) | 10.9 | 10.9 |
| Imaging elements (x, y) | 64 × 60 | 32 × 60 |
| Sky angular size of imaging element | High-Res 0.030″, Mid-Res 0.077″ | Wide-Field 0.464″ |
| FOV on sky | 1.92″×1.80″ (Hi-Res); 4.93″×4.62″ (Mid-Res) | 14.8″×27.8″ (Wide-Field) |

Fiber glass choice (§3.3): **LF5 core, KG-12 cladding**. Rationale: (i) matching melting temperatures enabled fabrication; (ii) refractive indices diverge *less* in the IR than many alternatives, reducing intensity cross-talk at long wavelengths; (iii) both glasses are somewhat absorptive in the IR → ribbons were kept physically short to preserve throughput. Cladding was increased to 3.5 μm (from the 1 μm of earlier Schad et al. 2014 ribbons) to further reduce cross-talk — the design target was ≤3 fibers of cross-talk at 1565 nm.

Polarization behavior: ordinary round fibers scramble polarization, but **rectangular cores** with the right aspect ratio behave as *multimode* along the long axis and *single-mode* along the short axis (Schad et al. 2014). This, combined with stress-induced birefringence, preserves Stokes-$Q$ (linear polarization oriented along the fiber's rectangular axis) through the IFU. DL-NIRSP therefore places the **modulator upstream** of the IFU (mixing the desired state into local Stokes-Q) and the **Wollaston analyzer downstream** in the spectrograph, all arranged in-plane to minimize U/V-to-Q crosstalk.

**한국어**
BiFOIS는 DL-NIRSP의 과학적 핵심이다. 평평한 광섬유 리본으로 구성된 번들 — **직사각형 코어**의 섬유들이 2D "이미징 어레이"(피드 초점면)에서 네 개의 수직 "슬릿 어레이"(분광기 입사면)로 빛을 물리적으로 전송한다.

**리본 구조**: BiFOIS-36은 1×90 코어 리본 64×2개, BiFOIS-72는 2×180 코어 리본 32×2개. 하늘에서 보면 하나의 imaging pixel = 3개 fiber core (BiFOIS-36) 또는 6×2 (BiFOIS-72).

**유리 선택**: 코어 LF5 + 클래딩 KG-12. 이유: (1) 용융온도가 비슷해 제작 가능, (2) IR 에서 굴절률 차이가 크지 않아 장파장 강도 cross-talk 감소, (3) 두 유리 모두 IR 에서 약간의 흡수 → 리본을 물리적으로 짧게 유지해야 처리량 보존. 클래딩을 기존 1 μm 에서 3.5 μm 로 늘려 cross-talk 목표 (1565 nm 에서 3 fiber 이하) 달성.

**편광 거동**: 원형 섬유는 편광을 섞지만 **직사각형 코어**는 종횡비가 적절하면 장축 방향으로 multimode, 단축 방향으로 single-mode 처럼 작동하며(Schad et al. 2014), 응력유도 복굴절과 합쳐져 Stokes-Q(리본 축 방향 선형편광)를 보존한다. 따라서 DL-NIRSP는 **변조기를 IFU 앞**에 두어 원하는 상태를 local Stokes-Q 로 믹싱하고, **Wollaston 해석기를 IFU 뒤(분광기)**에 둔다. 모든 반사는 in-plane 으로 배치해 U/V → Q 누설을 최소화한다.

### Part IV: §3.4 Detectors — sCMOS and H2RG / 검출기

**English**
- **Arm 1 (visible)**: Andor *Balor* sCMOS, 4128×4104 px, 12 μm square, ~30 Hz frame rate (matched to the IR cameras, Wöger et al. 2021). Only ~60% of the detector area is read out because the spectrograph magnification is matched to the IR detectors. Supports *global reset* and *rolling shutter* modes — this choice interacts with polarimetric modulation.
- **Arms 2 & 3 (IR)**: Teledyne *Hawaii-2RG* (H2RG), 2048×2048 HgCdTe, 18 μm pitch, **cryogenic cooling** via closed-cycle Joule–Thomson chiller. Intended 2.5 μm cutoff; observed extended sensitivity (Urbach tail; Terrien et al. 2016) necessitates new 5 μm cold-blocking filters (identified as an upgrade). Read via *STARGRASP* controllers at 29.39 Hz, 16-bit ADC.
Two readout modes are used:
  - *Fast up-the-ramp*: a reset produces a zero-integration bias frame, then the detector integrates for one or more frame times; successive non-destructive reads (NDRs) sample the charge at each row at the end of its frame time — effective integration time grows with more NDRs, enabling correlated double sampling.
  - *Sub-frame integration*: rows are reset and read over a fixed number of integer frame times → 0.1 s minimum, 34 ms (1023 sub-frame rows) maximum, adjustable in 0.06 s (2 frame) steps.

A key subtlety: because H2RG rows are reset/read at *slightly different times*, each pixel samples a slightly different modulation cycle. The demodulation matrix must therefore be built **pixel-by-pixel**, not row- or column-wise.

**한국어**
- **Arm 1**: Andor Balor sCMOS, 4128×4104, 12 μm, ~30 Hz. 실제로 검출기 면적 60% 만 읽어낸다. Global reset / rolling shutter 모드 선택이 편광 변조와 상호작용.
- **Arms 2,3**: Teledyne H2RG, 2048×2048 HgCdTe, 18 μm, 저온 냉각(JT 칠러). 검출기 컷오프 2.5 μm 목표였으나 Urbach tail 로 인해 감도가 3 μm 이상까지 이어짐 → 5 μm cold filter 업그레이드 예정. 29.39 Hz, 16-bit ADC.
두 가지 읽기 모드: **Fast up-the-ramp**(여러 비파괴 리드로 적분시간 증가) / **Sub-frame integration**(0.1 s 최소, 34 ms 최대, 0.06 s 단위).

**핵심 주의점**: H2RG 행마다 리셋/읽기 시각이 조금씩 다르므로 각 픽셀이 약간 다른 변조 사이클을 샘플링한다. 따라서 **복조 행렬은 픽셀 단위(pixel-by-pixel)** 로 결정해야 한다.

### Part V: §3.5 Polarization System — 듀얼빔 + 회전 변조기 / Dual-Beam + Rotating Modulator

**English**
DL-NIRSP follows the rotating-waveplate, dual-beam scheme of Lites (1987):

1. **Modulator** (upstream of IFU): a spare NSO **elliptical retarder** built as a five-layer polycarbonate stack sandwiched between BK-7 windows (Harrington et al. 2020). Not a simple waveplate — it has spatially and wavelength-dependent retardance, engineered to give balanced efficiency over visible + near-IR. During rotation it mixes Stokes $Q, U, V$ into the Stokes-$Q$ direction (local frame).
2. **IFU (BiFOIS)**: preserves Stokes-$Q$ linear polarization along the fiber-ribbon axis.
3. **Analyzer** (downstream, one per arm): **Wollaston prism** — a bonded pair of quartz right-angle prisms with crystal axes orthogonal. Unpolarized light entering splits into two beams with a divergence angle matched to *half* the IFU slit separation, so the two orthogonal-polarization spectra land in the dark gaps produced by the spectral mask.

**Modulation efficiency** (Figure 16, measured on-sky using the GOS wire-grid + retarder at 630/1083/1565 nm on 2021-08-05):
- Average efficiency for Stokes-$I$: >0.75 (target 1.0).
- Average combined efficiency for $(Q,U,V)$: >0.65 ($\xi_Q^2+\xi_U^2+\xi_V^2$ target 1.0).
- With the wavelength-dependent modulator response + 8-state modulation cycle, the authors estimate >0.85 for I and >0.90 combined for Q,U,V.

**Pixel-by-pixel demodulation matrix** is mandated by the rolling detector readout behavior, and this is encoded directly in DL-NIRSP's data-processing pipeline.

**한국어**
DL-NIRSP는 Lites (1987)의 rotating-waveplate + dual-beam 방식을 따른다:

1. **변조기** (IFU 앞): NSO 코우데 분광기의 스페어 **elliptical retarder** — 5층 폴리카보네이트 스택을 BK-7 창 사이에 끼워 index-matching 접착제로 붙인 것 (Harrington et al. 2020). 단순 파장판이 아니라 공간·파장 의존 retardance 를 가지며, 가시광+근적외선에서 균형잡힌 효율을 목표로 설계되었다. 회전하며 Stokes $Q, U, V$ 성분을 **local Stokes-$Q$** 방향으로 믹싱한다.
2. **IFU (BiFOIS)**: Stokes-$Q$ 선형편광을 섬유 축방향으로 보존.
3. **Analyzer (암마다)**: **Wollaston 프리즘** — quartz 결정축이 직교하는 right-angle prism 쌍이 본딩된 형태. 발산각(34° 웨지)을 IFU 슬릿 간격의 **절반**에 맞춰, 두 직교편광 스펙트럼이 spectral mask가 만든 빈 영역으로 떨어지게 한다.

**변조 효율** (2021-08-05 측정): Stokes-I 평균 >0.75, $(Q,U,V)$ 결합 >0.65 (이상값 1.0). 파장의존 모델 + 8-state 변조 적용 시 각각 >0.85, >0.90 추정.

**픽셀단위 복조 행렬**이 필수적 (H2RG 행별 비동기 읽기 때문).

### Part VI: §4 Optical Alignment — 뭐가 어렵나 / What Made Alignment Hard

**English**
Alignment required four metrology instruments working together:
- **Laser tracker** (FARO Vantage / API Omnitrac2): ~20 μm positional, ~0.01° angular accuracy.
- **Coordinate-measuring machine (CMM) arm** (FARO FaroArm / Edge): for precise distances.
- **Theodolites** (Wild-Leica T3000A): autocollimating — measure angles between flat optical surfaces at ≈30″ in reflection.
- **Optical interferometer** (4D Technology PhaseCam 6000, 633 nm HeNe): measures wavefront error of the assembled optical system.
These were tied to DKIST's global coordinate system via laser-tracker reads of the six spherical reflectors on the structural pillars, and the instruments were aligned to DKIST M9 through theodolite measurements. Components were placed with ~0.1 mm horizontal, ≤2 mm vertical accuracy, then powered mirrors were tilted to minimize astigmatism and defocus in the interferometer.

**Residual wavefront error (Figure 10)** after feed-optics alignment:
- Mid-Res (f/24): RMS 69 nm, PV 434 nm.
- High-Res (f/62): RMS 47 nm, PV 324 nm.
- Spectrograph Arm 1 (Figure 11): RMS 68 nm, PV 704 nm.
The interferometer only works at 633 nm, so IR spectrograph optics were aligned with a **"star-field"** of single-mode HeNe fibers at 633/1152/1523 nm injected at the spectrograph input (Figure 12). Fitted PSF FWHM: Arm 1 (633 nm) ~22×20 μm vs. model 16×12 μm; Arm 2 (1152 nm) ~29×20 μm vs. 19×18 μm; Arm 3 (1523 nm) ~29×31 μm vs. 22×23 μm. Measured spots are ~1.5× broader than the ideal PSF — acceptable but leaves room for improvement.

**한국어**
정렬에는 네 가지 계측 장비가 동시에 사용되었다: **laser tracker**(FARO, 20 μm/0.01° 정확도), **CMM arm**(정밀거리), **autocollimating theodolites**(30″ 정확도의 각도 측정), **interferometer**(4D PhaseCam, 633 nm HeNe, 파면 측정). 이들이 DKIST 전역 좌표계 — 구조 기둥의 구면 반사체 6개 — 에 연결된다. 부품은 수평 0.1 mm, 수직 ≤2 mm 정확도로 배치 후 파워드 거울의 틸트로 비점수차/defocus를 최소화.

**잔류 파면오차** (Fig. 10): Mid-Res f/24 RMS 69 nm (PV 434 nm); High-Res f/62 RMS 47 nm (PV 324 nm); Spectrograph Arm 1 RMS 68 nm. Interferometer가 633 nm 에서만 동작하므로 IR 암은 **"star-field"** 방법 — 단일모드 HeNe 섬유(633/1152/1523 nm)를 분광기 입사면에 주입해 PSF 측정 — 을 사용. 실측 스폿이 모델 PSF 대비 약 1.5배 넓음 (사용 가능한 수준).

### Part VII: §5 As-Built Performance — 실측 성능 / Measured Performance

**English**

**§5.1 Spatial Resolution**: Not yet fully characterized (knife-edge scan + VBI co-observation proposed). Wavefront measurements (Figures 10–12) show spectrograph is within budget on Arms 1 & 3; Arm 2 has higher wavefront error attributable to its Wollaston prism — a replacement prism is being fabricated.

**§5.2 Spectral Resolution (Figure 13)**: HeNe-laser profiles were fit with Gaussians. Measured FWHM → equivalent resolving power:
- Arm 1 (633 nm): FWHM = 0.004 nm → **R = 158,000** (vs. theoretical 184,000).
- Arm 2 (1152 nm): FWHM = 0.011 nm → **R = 105,000** (vs. 132,000).
- Arm 3 (1523 nm): FWHM = 0.014 nm → **R = 109,000** (vs. 120,000).
Measured R is slightly below theoretical because the ~29 μm fiber core broadens the slit beyond the Nyquist assumption.

**§5.3 Throughput (Figures 14–15)**: End-to-end efficiency was *estimated*, not directly measured (would require bright calibration standard). Combines DKIST FIDO + feed + spectrograph coatings/gratings/detectors + BiFOIS + narrow-band filters. Peak combined efficiencies: Arm 1 ~5% (green–red), Arm 2 ~4%, Arm 3 ~5% in High-Res mode. Narrow-band filter in-band throughput is typically >80%. Figure 15 shows each filter's measured profile overlaid on the NSO FTS atlas — filters are centered on their target diagnostic lines.

**§5.4 Polarization (Figure 16)**: Modulation-efficiency maps show $\xi_Q,\xi_U,\xi_V \gtrsim 0.4$ and $\xi_I > 0.75$ over most of the detector at 630/1083/1565 nm, with spatial structure reflecting the modulator's spatial retardance variation. Combined with full wavelength-dependent modulator modeling and the 8-state modulation sequence, the authors project $\xi_I > 0.85$, $\sqrt{\xi_Q^2+\xi_U^2+\xi_V^2} > 0.90$ — meeting the "within a factor of two of ideal" requirement.

**§5.5 Stability (Figures 17–18)**:
- **Intra-spectrograph drift** (telescope on quiet Sun, 8 min): Arms 1, 3 show 2–3 μm RMS shifts (well under a 12 μm / 18 μm pixel). Arm 2 is worse — again, Wollaston issue.
- **Feed → IFU drift** (pinhole at GOS, 8 min, HOAO locked): ±10 μm envelope, High-Res a bit scatterier than Mid-Res (2.6× magnification factor). Drift is ~linear in time → mechanical rather than thermal.

**§5.6 Spectrograph Configurations (Table 6)**: 20 valid three-wavelength combinations enumerated; 9 "tested" with hardware. Configurations straying too far from the nominal 5° Littrow angle suffer vignetting at the OAM and dichroic beam-splitters, and reduced diffraction efficiency. A future second grating (e.g. 31.6 lines/mm at 63° blaze) would unlock more configurations.

**한국어**

**§5.1 공간분해능**: 아직 완전 특성화 안 됨 (knife-edge + VBI 동시관측 제안). 분광기는 Arm 1, 3 예산 내이고, Arm 2 는 Wollaston 프리즘 불량으로 인한 파면 오차 (교체 제작 중).

**§5.2 스펙트럼 분해능** (Fig. 13, HeNe 가우시안 피팅):
- Arm 1 (633 nm): FWHM 0.004 nm → **R = 158,000**
- Arm 2 (1152 nm): FWHM 0.011 nm → **R = 105,000**
- Arm 3 (1523 nm): FWHM 0.014 nm → **R = 109,000**
Nyquist 이론치 대비 약간 낮은 이유는 ~29 μm 섬유 코어가 슬릿을 넓히기 때문.

**§5.3 처리량** (Fig. 14–15): 직접측정이 아닌 **추정** (벤더 데이터 + 실측 + 물성치 조합). 각 암 피크 효율: Arm 1 ~5%, Arm 2 ~4%, Arm 3 ~5% (High-Res 모드). Narrow-band 필터 in-band 처리량은 >80%.

**§5.4 편광** (Fig. 16): 각 암에서 $\xi_Q,\xi_U,\xi_V \gtrsim 0.4$, $\xi_I > 0.75$. 변조기 공간 retardance 변화가 그대로 나타남. 파장의존 모델 + 8-state 적용 시 $\xi_I > 0.85$, $\sqrt{\xi_Q^2+\xi_U^2+\xi_V^2} > 0.90$ 로 이상값(1.0) 대비 2배 이내 조건 충족.

**§5.5 안정성** (Fig. 17–18):
- 분광기 내 드리프트(8분, quiet Sun): Arms 1,3 RMS 2–3 μm (12/18 μm 픽셀 대비 작음). Arm 2 는 불량.
- Feed → IFU 드리프트 (핀홀, HOAO 락): ±10 μm 이내. 시간에 선형 → 기계적 원인 (rotator 영향 가능).

**§5.6 분광기 구성** (Table 6): 20개 세 파장 조합; 9개가 실제 하드웨어로 테스트됨. Littrow 각에서 너무 벗어나면 OAM·다이크로익에서 비네팅 + 회절효율 감소. 향후 두 번째 격자(예: 31.6 선/mm) 추가 시 조합 확대 가능.

### Part VIII: §6 First Results — 첫 빛 / First Light

**English**

**§6.1 Photospheric Pore (NOAA 12851, 2021-08-05, Figures 19–20)**:
Target: the largest pore in the leading polarity of AR 12851. Mode: Mid-Res, 630/1083/1565-nm configuration. A $7\times 7$ mosaic with 4.69″ FSM steps covered $30''\times 30''$, repeated four times in ~2 minutes. Full 8-state modulation. Eleven quantities derived per arm (continuum intensity, Doppler velocity, Stokes-V amplitude). Compared to SDO/HMI (Figure 20) — continuum, Dopplergram, and LOS magnetogram are in excellent agreement across all three DL-NIRSP channels, *despite moderate-to-poor seeing*. The 1565-nm IR channel shows **the highest spatial resolution** in this particular dataset because it is least seeing-limited. Mosaic artifacts appear at tile edges especially in IR.

**§6.2 Active-Region Corona (NOAA 12853, 2021-08-07, Figures 21–22)**:
Target: off-limb corona at $1.2R_\odot$ above AR 12853, on the east limb. Mode: Mid-Res, 789/1075/1430-nm configuration. Duration: ~2 min with 8-state discrete modulation. Separate measurement at the north pole provided a scattered-light + disk-leakage subtraction reference. DKIST Lyot stop was engaged; no wavefront correction because off-disk (no AO lock). Gaussian fits to the emission lines (Figure 22):
- [Fe XI] 789.2 nm: FWHM = 42 km/s, peak intensity $\sim 0.8 \times 10^{-4} I_\odot$
- [Fe XIII] 1074.7 nm: FWHM = 46 km/s, peak $\sim 1.1 \times 10^{-4} I_\odot$
- [Si X] 1430.0 nm: FWHM = 56 km/s, peak $\sim 0.18 \times 10^{-4} I_\odot$
The [Si X] 1430-nm detection is particularly significant — Penn & Kuhn (1994) first identified this line's potential, and Dima, Kuhn & Schad (2019) reported early SOLARC observations. DL-NIRSP's detection in ~2 min demonstrates routine ground-based [Si X] capability.

**한국어**

**§6.1 광구 기공** (Fig. 19–20):
목표: AR 12851의 가장 큰 기공. Mid-Res 630/1083/1565 nm 구성, $7\times7$ 모자이크 (4.69″ 스텝, 30″×30″ 총 시야, 4회 반복, 약 2분). 연속체 강도, Doppler 속도, Stokes-V 세 양을 각 암에서 복원. SDO/HMI와 일치. 시상이 나쁜 조건에서도 **1565 nm 채널이 가장 높은 공간분해능** 을 보여줌 — IR 파장이 시상의 영향을 덜 받기 때문.

**§6.2 활동영역 코로나** (Fig. 21–22):
목표: AR 12853 동쪽 림의 1.2 $R_\odot$ 상공. Mid-Res 789/1075/1430 nm 구성, ~2분, 8-state 이산 변조. 북극 관측으로 산란광·원반 누설 배경을 빼서 순수 코로나 방출선 추출.
- [Fe XI] 789.2: FWHM 42 km/s, 피크 $\sim 0.8\times 10^{-4} I_\odot$
- [Fe XIII] 1074.7: FWHM 46 km/s, 피크 $\sim 1.1\times 10^{-4} I_\odot$
- [Si X] 1430.0: FWHM 56 km/s, 피크 $\sim 0.18\times 10^{-4} I_\odot$
[Si X] 1430 nm 는 지상에서 관측이 극히 어려웠던 코로나 금지선이며, DL-NIRSP 의 2분 관측으로 일상화 가능성을 시연했다.

### Part IX: §7 Conclusions & Outlook — 업그레이드 로드맵 / Upgrade Roadmap

**English**
Announced upgrades:
1. **MISI-36 (Machined Image Slicer)** — polished-glass IFU to replace BiFOIS-36, with ~3× higher throughput and better intensity uniformity/polarization fidelity. Functionally similar to the GRIS slicer (Domínguez-Tagle et al. 2022).
2. **Image slicer with coarser sampling** for coronal Mid-Res observations.
3. **Arm 2 Wollaston prism** — replacement in fabrication, to resolve poor image quality on that arm.
4. **IR cold-blocking filters** to 5 μm, addressing H2RG Urbach-tail sensitivity.
5. **New polycarbonate modulator** specifically designed to suppress fringe frequencies in the IR arms (modulator was a spare from a lab system, not fringe-optimized).
6. **Second grating** (e.g. 31.6 lines/mm at 63° blaze) to broaden wavelength-combination options.
7. **FIDO M9a beam-splitter upgrade** → simultaneous DL-NIRSP + CryoNIRSP operation.

**한국어**
주요 업그레이드:
1. **MISI-36** (연마유리 IFU) → BiFOIS-36 대체, 처리량 3배, 강도 균일성·편광 충실도 개선 (GRIS slicer 유사).
2. 코로나 Mid-Res 를 위한 거친 샘플링의 image slicer.
3. Arm 2 Wollaston 프리즘 교체.
4. IR 냉각차단 필터 5 μm 까지 확장.
5. 신규 폴리카보네이트 변조기 (IR fringe 억제 설계).
6. 두 번째 격자 (예: 31.6 선/mm) → 파장 조합 확대.
7. FIDO M9a 빔스플리터 업그레이드 → CryoNIRSP 와 동시관측.

---

## 3. Key Takeaways / 핵심 시사점

1. **Integral-field is the defining capability, not an incremental upgrade** — DL-NIRSP is the only DKIST instrument that captures $(x,y,\lambda,\text{Stokes})$ per exposure. All other DKIST spectropolarimeters (ViSP, CryoNIRSP) scan a slit across the scene. For dynamic events — flares, reconnection, wave propagation — this removes temporal aliasing in the spatial dimension, a systematic error no amount of averaging in a long-slit system can remove.
   / **IFU는 점진적 개선이 아닌 결정적 차별점** — 슬릿을 스캔하는 ViSP/CryoNIRSP와 달리 DL-NIRSP는 매 노출에서 $(x,y,\lambda,\text{Stokes})$ 전체를 얻는다. 플레어·재결합·파동 같은 동적 현상에서 공간축 시간 앨리어싱(aliasing) 을 제거한다.

2. **BiFOIS is the innovation that made facility-class solar IFU spectropolarimetry possible** — previous IFUs either scrambled polarization (round fibers) or required impractically large spectrographs (glass slicers for the visible). BiFOIS exploits *rectangular fiber cores with controlled aspect ratio* + *stress-induced birefringence* to preserve linear polarization along a specific axis, allowing modulator + analyzer to sandwich the IFU.
   / **BiFOIS는 시설급 태양 IFU 편광분광을 가능케 한 핵심 혁신** — 기존 IFU는 편광을 섞거나(원형 섬유) 비현실적인 크기의 분광기가 필요했다(유리 슬라이서). BiFOIS는 직사각형 코어의 종횡비와 응력유도 복굴절을 조합해 선형편광을 특정 축으로 보존한다.

3. **Near-IR is not just convenient — it is the only way to meet all five design drivers simultaneously** — Zeeman splitting scales as $\lambda^2$ (Fe I 1565 > 630 by 6×), chromospheric diagnostics (He I 1083) live in the IR, seeing degrades *slowly* with wavelength (hence the surprising result that 1565 nm had the highest effective resolution in §6.1), and coronal forbidden lines (Fe XIII, Si X) are dense in the IR. The visible was added (500–900) because the blue-dichroic FIDO path was already available.
   / **근적외선은 편리한 선택이 아니라 유일한 선택** — Zeeman 분리가 $\lambda^2$, He I 1083 등 채층 진단, 시상의 파장의존 감소(1565 nm 가 실제로 가장 높은 유효분해능을 보임), 코로나 금지선 풍부. 가시광(500–900 nm) 추가는 FIDO 청색 경로가 이미 존재했기에 가능.

4. **Dual-beam polarimetry is mandatory at $10^{-4}$ sensitivity in the presence of seeing** — separating $\pm Q$ into the dark gaps of the spectral mask using a Wollaston prism cancels common-mode (seeing-induced) intensity variations. Combined with 8-state modulation and pixel-by-pixel demodulation (forced by H2RG rolling readout), DL-NIRSP can reach the required polarimetric accuracy.
   / **시상이 있는 조건에서 $10^{-4}$ 감도를 위해 dual-beam 은 필수** — Wollaston 으로 ±Q를 spectral mask 빈 공간에 분리해 공통모드 잡음 상쇄. 8-state 변조 + 픽셀별 복조(H2RG rolling readout 때문) 로 목표 달성.

5. **Performance is constrained less by optics than by mechanics, electronics, and integration** — wavefront error (Figs. 10–12) is within budget; the real pain points are (a) Arm 2 Wollaston defects, (b) H2RG Urbach-tail filter inadequacy, (c) modulator fringes not designed for, (d) coudé rotator drift. Instrument papers are largely a catalog of "what didn't meet spec and what we'll fix." The paper is honest about these ("TBD", "fabrication underway") and provides a concrete upgrade roadmap.
   / **성능 제약은 광학이 아니라 역학·전자·통합에서** — 파면오차는 예산 내, 실제 문제는 Arm 2 Wollaston, H2RG Urbach tail, 변조기 fringe, 코우데 rotator drift 이다. 논문은 이들을 "TBD" 로 정직하게 기술하고 구체 업그레이드 계획을 제공.

6. **Science-driven flexibility has an engineering cost** — the reconfigurable spectrograph (rotatable grating + translating arms) unlocks 20+ wavelength combinations but couples Littrow-angle choice to image quality via vignetting on the OAM/dichroics. A second grating with different ruling would decouple these. Lesson: flexibility is not free; the *Spectrograph Calculator* is a necessary engineering response.
   / **과학 중심의 유연성은 공학적 비용을 수반** — 회전격자 + 이동암 덕에 20+ 조합 가능하나 Littrow 각 선택이 OAM/다이크로익 비네팅과 결합. 두 번째 격자 도입 시 분리 가능. *Spectrograph Calculator* 는 유연성에 대응하는 공학적 필요.

7. **First-light science validates the concept — but also exposes how seeing interacts with diffraction limit** — the 2021-08 pore observation showed HMI-agreement across all three arms, *and* showed that when seeing is moderate, longer wavelengths dominate in effective spatial resolution despite their lower diffraction limit. This inverts the usual "shorter wavelength = higher resolution" intuition and argues strongly for multi-wavelength design when seeing is uncontrollable.
   / **첫 빛 관측이 개념을 검증하면서 시상-회절한계 상호작용도 드러남** — 2021-08 기공 관측에서 세 파장 모두 HMI와 일치. 시상이 보통인 경우 **1565 nm 가 가장 높은 유효 공간분해능**을 보임 — "단파장 = 고분해능" 통념을 뒤집는다. 시상을 제어할 수 없을 때 다파장 설계의 정당성.

8. **Ground-based coronal spectropolarimetry transitions from specialist feat to routine capability** — detecting [Si X] 1430 nm in ~2 minutes of integration, off-limb, without AO lock, is a milestone. Lin, Penn & Tomczyk (2000) spent a dedicated campaign for the first [Fe XIII] circular-polarization measurement; DL-NIRSP makes such measurements routine, enabling vector tomography (Kramar et al. 2013) at a production scale.
   / **지상 코로나 편광관측이 특수 실험에서 상시 관측 능력으로 전환** — AO락 없이 2분 관측으로 [Si X] 1430 nm 검출. Lin et al. (2000) 의 첫 [Fe XIII] 원형편광 캠페인 수준의 측정이 DL-NIRSP 로 routine 화되어, Kramar et al. (2013) 식 vector tomography 를 대규모로 가능하게 한다.

---

## 4. Mathematical Summary / 수학적 요약

### (1) Diffraction-limited angular resolution / 회절한계
$$
\theta_{\text{diff}} = 1.22\,\frac{\lambda}{D}
$$
For $D=4$ m: $\theta=0.0315''$ at 500 nm, $0.0394''$ at 630 nm, $0.0680''$ at 1083 nm, $0.0984''$ at 1565 nm. DL-NIRSP's 0.030″ High-Res sampling is Nyquist-matched to 500 nm.
/ $D=4$ m 에서 파장별: 500 nm 0.0315″, 630 nm 0.0394″, 1083 nm 0.068″, 1565 nm 0.0984″. 0.030″ 샘플링은 500 nm Nyquist 매칭.

### (2) Resolving power from grating equation / 격자 방정식에서 분해능
Littrow condition: incidence angle = diffraction angle $= \alpha = \beta = \theta_L$.
$$
m\lambda = 2 d \sin\theta_L \qquad\Rightarrow\qquad R = \frac{\lambda}{\Delta\lambda} = m N
$$
where $d=1/(23.2\,\text{mm}^{-1})=43.1\,\mu\text{m}$, $m$ is diffraction order, and $N$ is the number of illuminated grating lines over the 300 mm grating width. With $N\approx 23.2\times 300 = 6960$ lines, $m=68–75$ (per Fig. 9) → $R = mN \approx 5\times10^{5}$ theoretical. Measured R is lower (see Eq. 5 below) due to finite slit (fiber core) width.
/ Littrow 조건 하에 $R=mN$. 격자폭 300 mm × 23.2 선/mm = 6960 선, $m\approx 68{-}75$ → 이론 $R\approx 5\times10^5$. 실측은 섬유 코어 유한폭 때문에 낮아짐.

### (3) Zeeman splitting scaling / 제만 분리 파장의존성
$$
\Delta\lambda_B = \frac{e}{4\pi m_e c}\,g_{\text{eff}}\,\lambda^2 B \propto \lambda^2 B
$$
For the same magnetic field $B$, Fe I 1565 nm ($g_\text{eff}=3$) splits $\sim (1565/630)^2 \cdot 3/2.5 \approx 7.4\times$ more than Fe I 630.25 nm ($g_\text{eff}=2.5$). This is the fundamental reason DL-NIRSP emphasizes the NIR.
/ 같은 $B$ 에서 Fe I 1565 nm 는 Fe I 630.25 nm 대비 약 7.4배 분리된다. DL-NIRSP가 NIR 을 강조하는 근본 이유.

### (4) Dual-beam polarimetric modulation / Dual-beam 변조
Measured intensity at detector for one of two Wollaston beams during modulation state $j$:
$$
I_{\pm,j} = \tfrac{1}{2}\bigl[I \pm \bigl(m_{Q,j}Q + m_{U,j}U + m_{V,j}V\bigr)\bigr]
$$
Summing ($I_{+,j}+I_{-,j}=I$) cancels polarization; differencing ($I_{+,j}-I_{-,j}=m_{Q,j}Q+m_{U,j}U+m_{V,j}V$) cancels seeing-induced intensity fluctuations that are common between the two beams. Stacking $N$ modulation states:
$$
\begin{pmatrix}I_{+,1}-I_{-,1}\\I_{+,2}-I_{-,2}\\ \vdots \\ I_{+,N}-I_{-,N}\end{pmatrix}
= \mathbf{M}\,\begin{pmatrix}Q\\U\\V\end{pmatrix},\qquad
\begin{pmatrix}\hat Q\\\hat U\\\hat V\end{pmatrix}=\mathbf{D}\,(I_{+,j}-I_{-,j})
$$
where $\mathbf{D}$ is the optimal demodulation matrix (pseudo-inverse of $\mathbf{M}$).
/ Wollaston 두 빔 합은 편광을 소거, 차는 시상 공통모드를 소거. $N$개 변조상태를 쌓아 선형시스템 $\mathbf M$, 복조행렬 $\mathbf D = \mathbf M^+$.

### (5) Modulation efficiency / 변조 효율 (del Toro Iniesta 2003 Eq. 5.29)
$$
\xi_i = \left(N\sum_{j=1}^{N} D_{ij}^2\right)^{-1/2},\quad i\in\{I,Q,U,V\}
$$
Ideal: $\xi_I = 1$ and $\sqrt{\xi_Q^2+\xi_U^2+\xi_V^2}=1$. DL-NIRSP measures $\xi_I>0.75$, $\sqrt{\xi_Q^2+\xi_U^2+\xi_V^2}>0.65$ at 630/1083/1565 nm; modeled $>0.85$ and $>0.90$ with full 8-state cycle.
/ 이상값 $\xi_I=1$, $\sqrt{\xi_Q^2+\xi_U^2+\xi_V^2}=1$. 실측 $>0.75$, $>0.65$; 모델 $>0.85$, $>0.90$.

### (6) Polarimetric SNR requirement / 편광 SNR
Photon-noise-limited Stokes-parameter uncertainty:
$$
\sigma_{X} = \frac{1}{\xi_X \sqrt{N_\gamma}} \quad\text{for}\;X\in\{Q,U,V\}
$$
Goal $\sigma=5\times10^{-4}$ with $\xi=0.4$ requires $N_\gamma \geq 2.5\times 10^{7}$ photons per resolved spectral/spatial element per Stokes parameter. DL-NIRSP's throughput (~5% peak) and integration flexibility (0.1 s to several seconds) meet this for photospheric and chromospheric lines; coronal lines are typically $\sim 10^{-4}\,I_\odot$ and demand many-minute integration.
/ 광자잡음 한계에서 $\sigma_X = 1/(\xi_X \sqrt{N_\gamma})$. $\sigma=5\times10^{-4}$, $\xi=0.4$ 에 $N_\gamma \geq 2.5\times 10^{7}$ 필요. DL-NIRSP 처리량 5%로 광구/채층 달성, 코로나선은 수 분 적분 필요.

### (7) Effective slit width and R / 유효 슬릿폭 기여
$$
R_\text{meas}^{-1} \approx \sqrt{R_\text{diff}^{-2} + R_\text{slit}^{-2}}
$$
with $R_\text{slit}=\lambda f_\text{coll}/(w d_\text{grating})$ where $w$ is the slit (fiber core) width. For $w=29\,\mu$m, $f_\text{coll}=1250$ mm, $d_\text{grating}=1/23.2\text{ lines/mm}$: $R_\text{slit}$ dominates and explains the measured 15–20% shortfall below the theoretical Nyquist values.
/ 측정 $R^{-2}$ 은 회절폭과 슬릿폭의 RSS. 29 μm 코어가 이론값 대비 15–20% 감소 설명.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1870s   Lockyer: solar spectroscopy begins — single line, single slit
1908    Hale: Zeeman effect detected in sunspots — magnetic Sun
1952    Babcock: Stokes polarimeter — systematic solar magnetography
1960s   Kitt Peak SP, HAO ASP: long-slit spectropolarimetry establishes
1970s   (Sacramento Peak, BBSO) long-slit standard
1974    Martin, Ramsey, Carroll, Martin: multi-slit spectrograph concept
1987    Lites: rotating-waveplate + dual-beam polarimetry formalized
1994    Penn & Kuhn: [Si X] 1431 nm as potential coronal diagnostic
1996    SoHO launch (MDI, EIT, LASCO) — space-based synergy
2000    Lin, Penn, Tomczyk: first circular-polarization detection in corona
2003    Henault+: MUSE — nighttime IFS precedent (massive multi-arm spectrograph)
2004    Lin, Kuhn, Coulter: first solar IFU concept (intellectual ancestor of DL-NIRSP)
2006    ★ Lin & Versteegh: BiFOIS — fiber-optic image slicer patent
2010    Jaeggli+: FIRS — facility IR spectropolarimeter at DST (precursor)
2012    Collados+: GREGOR Infrared Spectrograph (GRIS) operational
2012    SDO/HMI full-disk magnetograms (Schou+, Pesnell+) — space reference
2014    ★ Schad+: BiFOIS laboratory prototype with 38×8 μm cores
2019    Anan+: DL-NIRSP IFU science demonstration at Dunn Solar Telescope
2020    Rimmele+: DKIST overview (paper #23 here)
2021 Nov DL-NIRSP coudé components fully installed
2022    de Wijn+: ViSP instrument paper (paper #24 here)
2022    ★ THIS PAPER — Jaeggli+: DL-NIRSP formal instrument description
2022    Domínguez-Tagle+: GRIS polished-glass IFU first light
2022    Fehlmann: CryoNIRSP instrument paper
2022    von der Lühe+: VTF instrument paper
 ↓
2023+   MISI-36 (machined image slicer) upgrade planned
2025+   Coronal-field tomography surveys using DL-NIRSP as data engine
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Rimmele et al. (2020), *DKIST* (#23) | Facility overview — DKIST's 4 m aperture, HOAO, coudé lab, FIDO; the shared optical foundation on which DL-NIRSP is mounted | High — DL-NIRSP 의 물리적 환경 전체 |
| de Wijn et al. (2022), *ViSP* (#24) | Companion instrument paper — ViSP is the long-slit sibling; contrast clarifies why IFU capability was needed on DKIST | High — 기능·파장 중복과 차별 |
| Lites (1987), *Rotating waveplates* | Foundational polarimetric modulation scheme used directly by DL-NIRSP (Sect. 3.5) | High — 편광 변조 원리 |
| Lin & Versteegh (2006), *BiFOIS patent* | Intellectual origin of the fiber-optic image slicer; DL-NIRSP is the first facility deployment | High — 핵심 기술 원천 |
| Schad et al. (2014), *BiFOIS laboratory* | Laboratory demonstration of linear-polarization preservation in rectangular-core ribbons; design parameters carried forward to DL-NIRSP | High — IFU 편광 거동 |
| Harrington et al. (2020, 2021), *DKIST polarization modeling* | Full Mueller-matrix model of DKIST optics + modulator retardance used to compute DL-NIRSP's wavelength-dependent efficiency | High — 편광 성능 해석 |
| Lin, Penn, Tomczyk (2000), *First coronal V* | First ground-based coronal circular polarization; DL-NIRSP makes this class of measurement routine | Medium — 코로나 magnetometry 맥락 |
| Wöger et al. (2021), *VBI* | Co-observation partner — VBI data are what DL-NIRSP's end-to-end spatial resolution will ultimately be tested against | Medium — AO-동시관측 검증 |
| Rempel (2014), *Quiet-Sun dynamo simulation* | Predicts 50% of magnetic energy at <100 km — motivates DL-NIRSP's 0.03″ High-Res mode | Medium — 과학 동기 |
| del Toro Iniesta (2003), *Spectropolarimetry textbook* | Source of modulation efficiency formulas (Eqs. 5.28–5.29) used in §5.4 | Medium — 이론 배경 |

---

## 7. References / 참고문헌

### Primary paper / 원 논문
- Jaeggli, S. A., Lin, H., Onaka, P., Yamada, H., Anan, T., Bonnet, M., Ching, G., Huang, X.-P., Kramar, M., McGregor, H., Nitta, G., Rae, C., Robertson, L., Schad, T. A., Harrington, D. M., Liang, M., Puentes, M., Sekulic, P., Smith, B., Sueoka, S. R., Toyama, P., Young, J., Berst, C. (2022). "The Diffraction-Limited Near-Infrared Spectropolarimeter (DL-NIRSP) of the Daniel K. Inouye Solar Telescope (DKIST)." *Solar Physics*, **297**, 137. https://doi.org/10.1007/s11207-022-02062-w

### Key citations within the paper / 내부 주요 인용
- Lin, H., Versteegh, A. (2006). "VisIRIS: a visible/IR imaging spectropolarimeter based on a birefringent fiber-optic image slicer." *Proc. SPIE* **6269**, 62690K.
- Schad, T. A., Lin, H., Ichimoto, K., Katsukawa, Y. (2014). "Polarization properties of a birefringent fiber optic image slicer for a duald-beam spectropolarimeter." *Proc. SPIE* **9147**, 91476E.
- Lites, B. W. (1987). "Rotating waveplates as polarization modulators for Stokes polarimetry of the sun." *Applied Optics*, **26**, 3838.
- Rimmele, T. R., et al. (2020). "The Daniel K. Inouye Solar Telescope — Observatory overview." *Solar Physics*, **295**, 172.
- de Wijn, A. G., Casini, R., Carlile, A., Lecinski, A. R., Sewell, S., Zmarzly, P., Eigenbrot, A., Beck, C., Wöger, F., Knölker, M. (2022). "The visible spectro-polarimeter of the Daniel K. Inouye Solar Telescope." *Solar Physics*, **297**, 22.
- Harrington, D. M., Sueoka, S. R., White, A. J. (2019). "Polarization modeling and predictions for DKIST part 5." *J. Astron. Telesc. Instrum. Syst.*, **5**, 1.
- Harrington, D. M., Jaeggli, S. A., Schad, T. A., White, A. J., Sueoka, S. R. (2020). "Polarization modeling and predictions for DKIST part 6." *J. Astron. Telesc. Instrum. Syst.*, **6**, 1.
- del Toro Iniesta, J. C. (2003). *Introduction to Spectropolarimetry*. Cambridge University Press.
- Lin, H., Penn, M. J., Tomczyk, S. (2000). "A new precise measurement of the coronal magnetic-field strength." *Astrophysical Journal Letters*, **541**, L83.
- Lin, H., Kuhn, J. R., Coulter, R. (2004). "Coronal magnetic field measurements." *Astrophysical Journal Letters*, **613**, L177.
- Penn, M. J., Kuhn, J. R. (1994). "How bright is the [Si X] 1431 nm coronal emission line?" *Solar Physics*, **151**, 51.
- Rempel, M. (2014). "Numerical simulations of quiet-Sun magnetism." *Astrophysical Journal*, **789**, 132.
- Wöger, F., et al. (2021). "The Daniel K. Inouye Solar Telescope (DKIST)/Visible Broadband Imager (VBI)." *Solar Physics*, **296**, 145.

### Related prior papers in this series / 본 프로젝트 선행 논문
- #23 Rimmele et al. (2020) — DKIST overview.
- #24 de Wijn et al. (2022) — ViSP instrument paper.
