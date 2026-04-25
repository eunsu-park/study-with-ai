---
title: "Far Ultraviolet Imaging from the IMAGE Spacecraft. 1. System Design"
authors: ["S. B. Mende", "H. Heetderks", "H. U. Frey", "M. Lampton", "S. P. Geller", "S. Habraken", "E. Renotte", "C. Jamar", "P. Rochus", "J. Spann", "S. A. Fuselier", "J.-C. Gerard", "R. Gladstone", "S. Murphree", "L. Cogger"]
year: 2000
journal: "Space Science Reviews"
doi: "10.1023/A:1005271728567"
topic: Space_Weather
tags: [FUV, IMAGE, aurora, magnetosphere, TDI, geocorona, Lyman-alpha, LBH, OI_135.6, instrument]
status: completed
date_started: 2026-04-25
date_completed: 2026-04-25
---

# 51. Far Ultraviolet Imaging from the IMAGE Spacecraft. 1. System Design / IMAGE 우주선의 원자외선 영상. 1. 시스템 설계

---

## 1. Core Contribution / 핵심 기여

This paper is the system-level design overview of the Far Ultraviolet (FUV) instrument complement on NASA's IMAGE (Imager for Magnetopause-to-Aurora Global Exploration) mission. IMAGE was the first space-borne program to observe the magnetosphere by remote sensing only, and the FUV package — three instruments sharing one Main Electronics Package (MEP) — provides the indispensable global auroral context and geocoronal hydrogen background that the magnetospheric neutral-atom and EUV imagers require. The complement is composed of a Wideband Imaging Camera (WIC) for broadband 140–190 nm LBH morphology, a Spectrographic Imager (SI) with two channels (SI-1216 for Doppler-shifted proton-aurora Ly-α and SI-1356 for OI 135.6 nm electron-aurora), and a 3-tube Geocoronal photometer (GEO) for 360° Ly-α scans. The central engineering insight is that all three imagers must function on a 2-rpm spinning platform whose Earth-staring window is only ~10 s out of every 120 s spin, and all three must reject the strong polar dayglow at 121.5667 (cold geocoronal Ly-α), 130.4 (resonance OI), and other lines.

이 논문은 NASA IMAGE 임무의 원자외선(FUV) 장비 복합체에 대한 시스템 수준 설계 개요입니다. IMAGE는 자기권을 오직 원격탐사로 관측한 최초의 우주임무이며, 하나의 주전자 패키지(MEP)를 공유하는 세 장비로 구성된 FUV 패키지는 자기권 중성 원자/EUV 영상기에 필수적인 전 지구 오로라 맥락과 지구코로나 수소 배경을 제공합니다. 복합체는 광대역 140–190 nm LBH 형태 영상을 담당하는 광대역 영상 카메라(WIC), 양성자 오로라의 도플러 편이 Ly-α를 담당하는 SI-1216 및 전자 오로라 OI 135.6 nm를 담당하는 SI-1356의 두 채널을 갖는 분광 영상기(SI), 360° Ly-α 스캔을 담당하는 3채널 지구코로나 광도계(GEO)로 구성됩니다. 핵심 공학적 통찰은 세 영상기 모두 2 rpm 자전 플랫폼에서 한 회전(120 s) 중 약 10 s만 지구를 바라보고 작동해야 하며, 121.5667 nm(차가운 지구코로나 Ly-α), 130.4 nm(공명 OI) 등의 강한 극지 데이글로를 모두 억제해야 한다는 점입니다.

The paper's algorithmic core is the description of Time Delay Integration (TDI) combined with on-board polynomial distortion correction implemented as a 12-bit look-up table (LUT). Hundreds of frames per spin are co-added with pixel offsets matching the 3°/s spin, after each frame is geometrically de-warped by polynomials whose coefficients (6 for SI, 10 for WIC) are derived from pre-flight calibration via least-squares fitting against a known angular grid. A key sub-feature is the "randomizer" that prevents fixed-pattern noise from the discrete LUT mapping and limits image non-uniformity to better than 1/16. End-to-end engineering tests with a deliberately distorting mirror on a rotating platform demonstrated that the resulting image is sharp to better than 0.5 pixel and that real-time FUV TDI hardware works as designed. Finally, the paper summarizes the as-built equivalent apertures (WIC: 0.04 cm², SI: 0.01 cm², GEO: 0.019 cm²) and shows that all FUV channels meet or exceed the science requirements summarized in Table I.

논문의 알고리즘 핵심은 시간 지연 적분(TDI)과 12비트 룩업 테이블(LUT)로 구현된 다항식 왜곡 보정의 결합 설명입니다. 한 회전당 수백 프레임이 3°/s 자전에 맞춘 픽셀 오프셋으로 누적되고, 각 프레임은 사전 발사 보정에서 최소제곱법으로 결정된 다항식(SI: 계수 6개, WIC: 계수 10개)으로 기하 보정됩니다. 핵심 부속 기법인 "randomizer"는 LUT 이산 매핑에서 발생하는 고정 패턴 잡음을 방지해 영상 비균일성을 1/16 이하로 제한합니다. 회전 플랫폼 위에 의도적 왜곡 거울을 둔 종단 공학 시험은 결과 영상이 0.5 픽셀 이내로 선명하며 실시간 FUV TDI 하드웨어가 설계대로 작동함을 입증했습니다. 마지막으로 표 III에 제작 후 등가 면적(WIC 0.04 cm², SI 0.01 cm², GEO 0.019 cm²)이 정리되어 있고, 모든 FUV 채널이 표 I의 과학 요구를 충족하거나 초과함을 보입니다.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction and Imaging Philosophy (Section 1, pp. 243–246) / 서론 및 영상 철학

The introduction frames magnetospheric science as the study of how charged particles, magnetic fields, and electric fields interact under solar-wind influence, and notes that for forty years this science was conducted from single-point in-situ probes. IMAGE, by contrast, is the first satellite to attempt magnetospheric remote sensing only — He II 30.4 nm imaging of the cold plasma, neutral-atom imaging of energetic ions via charge exchange, RPI radio sounding, and FUV imaging of the auroral footprint. The auroral oval is critical because it is the ionospheric projection of magnetospheric boundaries and processes; without simultaneous global FUV observations, in-flight magnetospheric measurements lose context.

서론은 자기권 과학을 태양풍 영향 아래 하전 입자·자기장·전기장 상호작용 연구로 정의하고, 40년간 이를 단일 지점 in-situ 탐사로 수행해 왔음을 언급합니다. 반면 IMAGE는 자기권을 원격탐사만으로 관측하는 최초의 위성으로, He II 30.4 nm로 차가운 플라즈마, 전하 교환을 통한 중성 원자 영상기로 고에너지 이온, RPI 전파 음파 측정, 그리고 FUV로 오로라 발자국을 영상화합니다. 오로라 오발은 자기권 경계와 과정의 전리권 투영이므로 동시 전 지구 FUV 관측 없이는 자기권 측정의 맥락이 소실됩니다.

Mende et al. then list four general considerations for any imager: (1) defining the mean direction of light entering each pixel, (2) the spatial size associated with that pixel, (3) the photon-counting efficiency, and (4) wavelength acceptance/rejection. The Rayleigh photometric unit is introduced and worked out as $1\ \mathrm{R} = 10^{6}/4\pi \approx 80\,000$ photons s⁻¹ cm⁻² sr⁻¹, and the per-pixel response is multiplied by FOV solid angle in steradians, exposure in seconds, and equivalent collecting area $A_e$ to obtain counts per Rayleigh. UV FUV efficiencies are typically 1–2%, dominated by photocathode QE and optical reflectivities, so $A_e$ — the product of geometric aperture and total efficiency — is the figure of merit for any FUV instrument.

Mende 등은 영상기 일반 고려사항 4가지를 제시합니다: (1) 픽셀로 들어오는 광선의 평균 방향 정의, (2) 픽셀의 공간 크기, (3) 광자 계수 효율, (4) 파장 수용/거부. 레일리 단위는 $1\ \mathrm{R} = 10^{6}/4\pi \approx 80\,000$ photons s⁻¹ cm⁻² sr⁻¹로 유도되며, 픽셀당 응답은 FOV 입체각, 노출 시간, 등가 집광 면적 $A_e$의 곱으로 카운트/레일리를 산출합니다. UV FUV 효율은 광음극 QE와 광학 반사율에 의해 1–2%로 제한되므로, 기하 면적 × 전체 효율인 $A_e$가 모든 FUV 장비의 성능 지표가 됩니다.

### Part II: IMAGE FUV Requirements (Section 2, pp. 246–253) / IMAGE FUV 요구사항

The IMAGE program asks three top-level questions: (1) what mechanisms inject plasma into the magnetosphere on substorm/storm timescales? (2) what is the directly driven response of the magnetosphere to solar-wind changes? (3) how are magnetospheric plasmas energized, transported, and lost? FUV addresses chiefly question 3 by mapping the auroral footprint of substorm injections, ring-current pitch-angle scattering losses, and plasmaspheric/dayside coupling. From IMAGE apogee at 7 R_E (~44 000 km), the Earth subtends only 16°, the auroral oval ~8°. To capture the entire oval from apogee, FOVs of >8° (chosen as 17° for both SI and WIC) are required. To resolve the ~100 km auroral structures, $\geq 128 \times 128$ pixels per memory matrix are needed (WIC has 256², SI has 128²).

IMAGE 프로그램은 세 가지 최상위 질문을 던집니다: (1) 부폭풍/폭풍 시간 규모에서 자기권으로 플라즈마를 주입하는 메커니즘은 무엇인가? (2) 태양풍 변화에 대한 자기권의 직접 구동 반응은 무엇인가? (3) 자기권 플라즈마는 어떻게 에너지화, 수송, 손실되는가? FUV는 주로 (3)을 다루며, 부폭풍 주입의 오로라 발자국, 링 전류의 피치각 산란 손실, 플라즈마권/주간측 결합을 매핑합니다. 7 R_E 원지점에서 지구는 16°, 오로라 오발은 약 8°를 차지하므로, 원지점에서 오발 전체를 포착하려면 >8°(SI·WIC 모두 17° 채택)의 FOV가 필요합니다. ~100 km 오로라 구조 분해를 위해 메모리당 ≥128×128 픽셀이 필요(WIC 256², SI 128²)합니다.

The physical basis for the three-channel design is the auroral spectrum (Figure 1, p. 249), modeled at 10 kR of OI 130.4 nm: (a) 121.6 nm Lyman-α has both cold geocoronal (121.5667) and Doppler-shifted hot proton-aurora components; (b) 130.4 nm OI is bright but multiply-scattered, so it is a poor choice for resolved auroral imaging; (c) 135.6 nm OI is scattered only weakly (Strickland & Anderson 1983) and traces electron precipitation cleanly; (d) the LBH 140–190 nm band is excited primarily by N₂ + electrons, with the long-wavelength end avoiding O₂ Schumann–Runge absorption and giving altitude/energy information. To suppress 130.4 contamination of the 135.6 channel below 1%, narrowband filters are insufficient — a spectrometer is mandatory. Similarly, separating Doppler-shifted proton Ly-α from the orders-of-magnitude brighter cold geocoronal Ly-α requires anti-coincidence Doppler grills in the SI-1216 channel.

3채널 설계의 물리적 근거는 10 kR OI 130.4 nm 모델링 오로라 스펙트럼(그림 1)에 있습니다: (a) 121.6 nm Ly-α는 차가운 지구코로나(121.5667 nm)와 도플러 편이 양성자 오로라 성분을 모두 포함; (b) 130.4 nm OI는 밝지만 다중 산란되어 분해 영상에 부적절; (c) 135.6 nm OI는 약하게만 산란되어(Strickland & Anderson 1983) 전자 강하를 깨끗이 추적; (d) LBH 140–190 nm 밴드는 주로 N₂+전자로 여기되며 장파장 끝은 O₂ 슈만-룽게 흡수를 피해 고도/에너지 정보를 제공합니다. 135.6 채널의 130.4 오염을 1% 이하로 억제하려면 협대역 필터로는 불충분하고 분광계가 필수입니다. 마찬가지로 도플러 편이 양성자 Ly-α를 자릿수가 큰 차가운 지구코로나 Ly-α에서 분리하려면 SI-1216 채널의 도플러 격자 동시 부합이 필요합니다.

Table I (p. 252) is the FUV requirements summary. It lists for each science target (LBH morphology, OI 135.6 morphology, hydrogen aurora, geocorona, dayglow): wavelength, FOV, pixel size at apogee, angular resolution, photon collection efficiency × exposure (in R⁻¹ pix⁻¹ cm⁻²), and minimum aperture for 1 count per pixel from a 100 R source. Examples: LBH morphology 140–90 nm, FOV >10°, 100 km pixels, 0.13°, $4.1$ R⁻¹ pix⁻¹ cm⁻² × 10 s, $A_e \geq 0.0024$ cm²; geocorona 121.6 nm, 360° coverage, 1° resolution, $5$ R⁻¹ × 0.33 s, $A_e \geq 0.002$ cm². With FUV efficiencies of only 1–2%, these "small" apertures actually demand large geometric collecting areas.

표 I는 FUV 요구사항 요약입니다. 각 과학 대상(LBH 형태, OI 135.6 형태, 수소 오로라, 지구코로나, 데이글로)에 대해 파장, FOV, 원지점 픽셀 크기, 각분해능, 광자 수집 효율×노출, 100 R 신호에서 픽셀당 1 카운트를 위한 최소 면적이 제시됩니다. 예: LBH 140–90 nm, FOV >10°, 100 km 픽셀, 0.13°, 4.1 R⁻¹ pix⁻¹ cm⁻²×10 s, $A_e \geq 0.0024$ cm². FUV 효율이 1–2%에 불과하므로 이 "작은" 면적은 실제로 큰 기하학적 면적을 의미합니다.

The paper also discusses radiation environment: at 7 R_E the IMAGE spacecraft crosses the radiation belts on every orbit, accumulating ~50 kRad in 2 years inside 0.2-inch aluminum walls. The WIC CCD is shielded with a tantalum cup (~500 g) to reject protons below 50 MeV that would otherwise generate charge-trap defects.

방사선 환경: 7 R_E에서 IMAGE는 매 궤도 방사선대를 통과하며 0.2-inch 알루미늄 벽 내부에 2년간 약 50 kRad 누적. WIC CCD는 50 MeV 이하 양성자(전하 트랩 결함 유발)를 차단하기 위해 약 500 g 탄탈룸 컵으로 차폐됩니다.

### Part III: The IMAGE FUV System (Section 3, pp. 253–257) / IMAGE FUV 시스템

The greatest engineering challenge is taking high-resolution, high-sensitivity images during the brief Earth-pointing window of each spin. The full memory FOV (the inertial-space rectangle that holds the integrated image) is wider than the instantaneous detector FOV; as the spacecraft rotates at 3°/s, the instantaneous FOV scans across the memory FOV, and individual frames are co-added at offset addresses. This is the TDI principle (Figure 2, p. 254). For WIC, with a 30°×17° instantaneous FOV and 17°×17° memory FOV, a point on Earth spends $30 \times 120/360 = 10$ s in the FOV per spin; with 30 video frames per second this is ~300 frames co-added per spin. SI has a 15°×15° FOV and only 5 s of exposure per spin per Earth point.

가장 큰 공학적 난제는 한 회전 중 짧은 지구 지향 창 동안 고해상도·고감도 영상을 얻는 것입니다. 전체 메모리 FOV(누적 영상을 보관하는 관성 좌표계 사각형)는 순간 검출기 FOV보다 넓고, 우주선이 3°/s로 회전함에 따라 순간 FOV가 메모리 FOV를 가로질러 스캔하며, 각 프레임은 오프셋 주소로 누적됩니다. 이것이 TDI 원리(그림 2)입니다. WIC는 30°×17° 순간 FOV와 17°×17° 메모리 FOV로, 한 회전당 지구 한 점이 FOV에 머무는 시간은 $30 \times 120/360 = 10$ s, 30 fps에서 회전당 약 300 프레임이 누적됩니다. SI는 15°×15° FOV에 회전당 5 s 노출입니다.

The system block diagram (Figure 3, p. 256) shows four packages: WIC, SI, GEO, and MEP. WIC is a wide-field reflective system (concentric mirror) feeding a single-stage MCP image intensifier; the intensifier phosphor is fiber-optically coupled to a CCD read out at 30 fps. The CCD output is digitized and pipe-lined to the WIC TDI board in the MEP. SI is a 2-D imaging monochromator with two channels — SI-1218 for Doppler-Ly-α (119–126 nm pass-band, with anti-coincidence grills suppressing 121.5667 and 120.0 nm) and SI-1356 for OI 135.6 (135.6 ± 4.0 nm pass-band) — each with its own MgF₂ entrance window, triple MCP stack, and crossed-delay-line (XDL) anode read out via Time Delay Conversion (TDC) electronics that encode each photoelectron's (x, y) position. SI uses photon-counting; WIC uses analog frame integration. GEO contains 3 separate Ly-α detector tubes with MgF₂ lenses, sharing a high-voltage supply, with 4 sun sensors that automatically reduce HV to ~200 V when the Sun enters the FOV.

시스템 블록 다이어그램(그림 3)은 네 패키지를 보여줍니다: WIC, SI, GEO, MEP. WIC는 광시야 반사형 광학계(공심 거울)가 단단 MCP 영상 증폭기로 빛을 전달하고, 증폭기의 인광체가 광섬유 다발로 CCD에 결합되어 30 fps로 읽힙니다. CCD 출력은 디지털화되어 MEP의 WIC TDI 보드로 파이프라인됩니다. SI는 2D 영상 분광기로 두 채널 — SI-1218(119–126 nm 통과대, 121.5667 및 120.0 nm 동시 부합 억제) 도플러 Ly-α 채널과 SI-1356(135.6 ± 4.0 nm) OI 135.6 채널 — 각각 MgF₂ 입사창, 3단 MCP 적층, 교차 지연선(XDL) 양극으로 구성되며 각 광전자의 (x, y) 위치를 암호화하는 시간 지연 변환(TDC) 전자장치로 읽힙니다. SI는 광자 계수, WIC는 아날로그 프레임 적분 방식입니다. GEO는 MgF₂ 렌즈를 갖는 3개의 별도 Ly-α 검출기관으로 구성되며 공통 HV 전원을 공유하고, 4개 태양 센서가 태양이 FOV에 들어오면 HV를 자동으로 ~200 V로 낮춥니다.

The MEP houses the delay lines, two SI TDI boards (TDI12, TDI13), two TDC boards (TDC12, TDC13), the WIC TDI board, the Data Processing Unit (DPU), and power control. The DPU receives a "nadir pulse" sync signal each rotation (FUV imagers are 315° from spacecraft nadir), instantaneous spin rate from the IMAGE star tracker via the Central Instrument Data Processor (CIDP), and commands. Average FUV power is <13 W (excluding heaters); much of that is saved by powering down between Earth-pointing windows.

MEP는 지연선, 두 SI TDI 보드(TDI12, TDI13), 두 TDC 보드(TDC12, TDC13), WIC TDI 보드, 데이터 처리 장치(DPU), 전원 제어 장치를 수용합니다. DPU는 회전당 "nadir pulse" 동기 신호(FUV 영상기는 우주선 nadir로부터 315° 위치), 중앙 장비 데이터 처리기(CIDP)를 통해 IMAGE 스타 트래커에서 즉시 자전율, 명령을 받습니다. 평균 FUV 전력은 <13 W(히터 제외); 대부분은 지구 지향 창 사이에 전원을 낮춰 절약합니다.

### Part IV: TDI Operation and Distortion Correction (Section 4, pp. 257–264) / TDI 동작 및 왜곡 보정

This is the algorithmic heart of the paper. The optical distortion of any wide-field imager maps inertial coordinates $(x_0, y_0)$ to detector coordinates $(x_1, y_1)$ via $(x_1, y_1) = (f_1, f_2)(x_0, y_0)$. The on-board distortion correction inverts this: $(x_2, y_2) = (F_1, F_2)(x_1, y_1) \approx (x_0, y_0)$. Since the corrections must run at CCD readout rates (~100 ns per pixel for WIC), they are implemented as a 12-bit address LUT, with content pre-computed on the ground from a polynomial least-squares fit. The polynomials are:

이 절은 논문의 알고리즘 핵심입니다. 광시야 영상기의 광학 왜곡은 관성 좌표 $(x_0, y_0)$를 검출기 좌표 $(x_1, y_1)$로 매핑합니다: $(x_1, y_1) = (f_1, f_2)(x_0, y_0)$. 온보드 왜곡 보정은 이를 반전합니다: $(x_2, y_2) = (F_1, F_2)(x_1, y_1) \approx (x_0, y_0)$. 이 보정은 CCD 읽기 속도(WIC 픽셀당 ~100 ns)로 실행되어야 하므로, 12비트 주소 LUT로 구현되며 그 내용은 사전에 지상에서 다항식 최소제곱 적합으로 계산됩니다. 다항식은 다음과 같습니다:

For SI (quadratic, 6 coefficients):

SI(2차, 계수 6개):

$$
x_2 = A_0 + A_1 x_1 + A_2 y_1 + A_3 x_1 y_1 + A_4 x_1^2 + A_5 y_1^2 \tag{1}
$$

$$
y_2 = B_0 + B_1 x_1 + B_2 y_1 + B_3 x_1 y_1 + B_4 x_1^2 + B_5 y_1^2 \tag{2}
$$

For WIC (cubic, 10 coefficients):

WIC(3차, 계수 10개):

$$
x_2 = A_0 + A_1 x_1 + A_2 y_1 + A_3 x_1^2 + A_4 y_1^2 + A_5 x_1 y_1 + A_6 x_1^3 + A_7 x_1^2 y_1 + A_8 x_1 y_1^2 + A_9 y_1^3 \tag{3}
$$

$$
y_2 = B_0 + B_1 x_1 + B_2 y_1 + B_3 x_1^2 + B_4 y_1^2 + B_5 x_1 y_1 + B_6 x_1^3 + B_7 x_1^2 y_1 + B_8 x_1 y_1^2 + B_9 y_1^3 \tag{4}
$$

The rotation correction is then applied as a simple time-dependent offset: at time $t$ within the spin (rotation rate $K = 3°/\mathrm{s}$), the memory address $(x_n, y_n)$ where the photon is co-added is

회전 보정은 단순한 시간 의존 오프셋으로 적용됩니다: 회전 내 시각 $t$(자전율 $K = 3°/\mathrm{s}$)에서 광자가 누적되는 메모리 주소 $(x_n, y_n)$는

$$
x_n = x_2 - K t, \qquad y_n = y_2.
$$

The distortion correction creates a subtle problem: when a regular distorted grid is unwarped, source pixels must be redistributed across multiple destination pixels (Figure 4, p. 260). Ideally one would split charge by overlap area, but at 100 ns per pixel this is impossible. The chosen solution is **probabilistic single-pixel routing**: the LUT routes the entire pixel into one of the candidate destinations selected by overlap-area-weighted probability. To prevent fixed-pattern noise, a "randomizer" adds a value 0–15 to the 12-bit destination address before truncation to 8 bits, with the value cycling each new CCD frame. This limits non-uniformity to less than 1 part in 16 (~6%), which the TDI process further smooths along the rotation direction.

왜곡 보정은 미묘한 문제를 일으킵니다: 정규 왜곡 격자를 펼치면 원본 픽셀이 다수의 목적 픽셀에 재분배되어야 합니다(그림 4). 이상적으로는 면적 비율로 전하를 분할해야 하지만 픽셀당 100 ns에서는 불가능합니다. 채택된 해법은 **확률적 단일 픽셀 라우팅**으로, LUT가 픽셀 전체를 면적 가중 확률로 선택된 후보 목적 픽셀 하나로 라우팅합니다. 고정 패턴 잡음 방지를 위해 "randomizer"가 12비트 목적 주소에 0–15 값을 더한 뒤 8비트로 절단하며, 이 값은 매 CCD 프레임마다 순환합니다. 이는 비균일성을 16분의 1(~6%) 이하로 제한하며, TDI 과정은 이를 회전 방향으로 더 평활화합니다.

The polynomial coefficients $A_i, B_i$ are determined by ground calibration: a collimated UV beam is directed at the imager at known angles $(x_0, y_0)$, the corresponding detector coordinates $(x_1, y_1)$ are recorded, and a least-squares fit yields the coefficients. These are uplinked to the MEP, the DPU computes the LUT contents, and the instrument is then ready for real-time correction. The technique was validated by mounting a WIC Engineering Test Unit (ETU) — flight-identical electronics with a visible-light front end — behind a deliberately distorting convex mirror on a 3°/s rotating platform. Figures 5–7 show the ETU results: Figure 5(a) is the raw distorted grid as imaged through the mirror, Figure 5(b) shows the same grid after applying the correction LUT — much closer to rectilinear, with worst-case error <0.5 pixel. Figure 6 demonstrates the full TDI mode: 6(a) is a still-platform target, 6(b) is the same scene with the platform rotating but no distortion correction (greatly blurred), 6(c) is still-platform with correction (sharpest), and 6(d) is rotating with correction (much improved over 6(b), with some residual blur from the TDI process). Figure 7 shows that blanking the outermost pixels (where distortion is largest) further sharpens the image, providing an on-orbit sharpening backup option.

다항식 계수 $A_i, B_i$는 지상 보정으로 결정됩니다: 알려진 각도 $(x_0, y_0)$로 평행광 UV 빔을 영상기에 입사시켜 대응 검출기 좌표 $(x_1, y_1)$를 기록하고, 최소제곱 적합으로 계수를 얻습니다. 이를 MEP에 업링크하면 DPU가 LUT 내용을 계산하고 영상기는 실시간 보정 준비 완료. 이 기법은 비행과 동일한 전자장치에 가시광 전단을 갖는 WIC 공학 시험 단위(ETU)를 의도적 왜곡 볼록 거울 뒤에 배치한 3°/s 회전 플랫폼에서 검증되었습니다. 그림 5(a)는 거울로 촬영한 원본 왜곡 격자, 5(b)는 보정 LUT 적용 후로 직선에 훨씬 가까우며 최악 오차 <0.5 픽셀. 그림 6은 전체 TDI 모드를 보여줍니다: 6(a)는 정지 플랫폼 표적, 6(b)는 회전하지만 보정 없음(크게 흐림), 6(c)는 정지 플랫폼+보정(가장 선명), 6(d)는 회전+보정(6b 대비 크게 개선, TDI 과정에서 약간의 잔여 흐림). 그림 7은 가장자리 픽셀(왜곡이 큰 영역)을 블랭킹하면 영상이 더욱 선명해짐을 보이며, 이는 궤도상 영상 선명화 백업 옵션을 제공합니다.

A subtle but important point: the SI uses the same TDI/distortion logic, but its detector intrinsic resolution is ~1024×1024 (vs. 128² output), so the distortion correction can be processed at sub-pixel resolution with full 12-bit spatial accuracy. WIC's 256² output approaches the detector resolution limit, so accuracy is barely adequate. The TDI process requires synchronization to the spin rate to better than 0.5% to remain in phase across 128 pixels of integration.

미묘하지만 중요한 점: SI는 동일한 TDI/왜곡 논리를 사용하지만, 검출기 내재 분해능이 ~1024×1024(출력 128² 대비)이므로 왜곡 보정을 12비트 공간 정확도로 부분 픽셀 분해능에서 처리할 수 있습니다. WIC의 256² 출력은 검출기 분해능 한계에 접근해 정확도가 간신히 충분합니다. TDI 과정은 128 픽셀 적분 동안 위상 유지를 위해 자전율 동기화가 0.5% 이상 정확해야 합니다.

### Part V: System Performance Validation (Section 5, pp. 264–266) / 시스템 성능 검증

The IMAGE FUV contract was awarded in June 1996 and the flight complement delivered in January 1999 — a remarkably short 31 months for a system of this complexity. The design and build was distributed: SI was designed jointly by the Centre Spatiale of the University of Liège and UC Berkeley/Lockheed-Martin, built and calibrated entirely at Liège; WIC optics came from Canada and were integrated and built at UC Berkeley, with NASA Marshall Space Flight Center calibrating; GEO tubes came from the Max-Planck-Institut für Aeronomie in Lindau, Germany, integrated at Berkeley. The WIC and GEO integration and the MEP were all done at Berkeley. Each instrument was calibrated as a stationary imager (no rotating platform was available in the calibration vacuum chambers); the rotating-platform behavior was inferred from ETU measurements.

IMAGE FUV 계약은 1996년 6월 체결, 비행 복합체는 1999년 1월 인도 — 이 정도 복잡도 시스템으로는 놀랍도록 짧은 31개월. 설계·제작은 분산됨: SI는 Liège 대학의 Centre Spatiale와 UC Berkeley/Lockheed-Martin이 공동 설계, Liège에서 전적으로 제작·보정. WIC 광학계는 캐나다에서 도입, UC Berkeley에서 통합·제작, NASA Marshall에서 보정. GEO 검출기관은 독일 Lindau의 Max-Planck-Institut für Aeronomie에서 공급, Berkeley에서 통합. WIC·GEO 통합과 MEP는 모두 Berkeley에서 수행. 각 장비는 정지 영상기로 보정(보정 진공 챔버에 회전 플랫폼이 없었음); 회전 플랫폼 거동은 ETU 측정에서 추정.

Table III (p. 266) summarizes as-built performance: GEO has a 1° pixel/cell, photon collection 6.3 cm⁻² R⁻¹ cell⁻¹, $A_e = 0.019$ cm², 12 counts/cell per 100 R exposure (5 × 0.33 s); WIC has 0.09° pixel, 0.18° resolution cell, photon collection 5.7, $A_e = 0.04$ cm², 23 counts/cell per 100 R exposure; SI-1218 (Ly-α channel) has 0.11° pixel, 0.11° cell, 1.6, $A_e = 0.010$ cm², 1.8 counts/cell; SI-1356 (OI 135.6) has 0.11° pixel/cell, 1.6, $A_e = 0.008$ cm², 1.3 counts/cell. Compared to Table I requirements (LBH ≥0.0024 cm², OI ≥0.006 cm², H aurora ≥0.0015 cm², geocorona ≥0.002 cm²), all four channels exceed requirements by factors of 1.3 (SI-1356 vs. OI 135.6 OI requirement) to ~17 (WIC vs. LBH).

표 III는 제작 후 성능 요약: GEO는 1° 픽셀/셀, 광자 수집 6.3 cm⁻² R⁻¹ cell⁻¹, $A_e = 0.019$ cm², 100 R에서 셀당 12 카운트(5 × 0.33 s); WIC는 0.09° 픽셀, 0.18° 분해능 셀, 광자 수집 5.7, $A_e = 0.04$ cm², 100 R에서 셀당 23 카운트; SI-1218(Ly-α 채널)는 0.11° 픽셀, 0.11° 셀, 1.6, $A_e = 0.010$ cm², 1.8 카운트; SI-1356(OI 135.6)은 0.11° 픽셀/셀, 1.6, $A_e = 0.008$ cm², 1.3 카운트. 표 I 요구사항(LBH ≥0.0024 cm², OI ≥0.006 cm², H 오로라 ≥0.0015 cm², 지구코로나 ≥0.002 cm²) 대비, 네 채널 모두 1.3배(SI-1356 대 OI 135.6) ~ 약 17배(WIC 대 LBH) 초과.

### Part VI: Data Products (Section 6, pp. 266–268) / 데이터 산출물

FUV data packages contain: SI-1218 image, SI-1356 image, WIC image, three GEO data sets, periodic message data forwarded from the CIDP, and housekeeping. The periodic message provides the inertial direction of the rotation axis, instantaneous spin rate (used directly by the DPU to set TDI rates), star tracker quaternions, and timing — all necessary for ground processing to project the 17°×17° memory FOV into geographic/magnetic coordinates. Star observations during each spin (offset views from nadir) are used to refine the absolute pointing, since the launch environment can disturb pre-flight star-tracker–imager alignment. Browse products use POLAR UVI archived images to mock up what the IMAGE FUV view will look like (Figure 8, p. 268), with the IMAGE FUV FOV being about twice the POLAR UVI FOV. Pulse-height distributions for SI MCPs are continuously down-linked so MCP gain changes can be compensated by HV adjustment; WIC pulse-height distributions are recovered occasionally by stopping the TDI process and reading single frames. GEO data are reported as model parameters (no image is formed); a parametric H exospheric profile (e.g., $N_0 = 1.2$, $a = 3.2$, $b = 0.27$, $c = 0.051$ for Lyman-α at Rev #319, geocentric distance 44 000 km in the Figure 8 example) is fit to the three Ly-α tubes' spin-resolved measurements.

FUV 데이터 패키지는 다음을 포함합니다: SI-1218 영상, SI-1356 영상, WIC 영상, GEO 3채널 데이터, CIDP에서 전달받은 주기 메시지, 하우스키핑. 주기 메시지는 자전축의 관성 방향, 즉시 자전율(DPU가 TDI 속도 설정에 직접 사용), 스타 트래커 쿼터니언, 시각을 제공 — 17°×17° 메모리 FOV를 지리/자기 좌표로 투영하기 위한 지상 처리에 필수. 매 회전 동안의 별 관측(nadir에서 오프셋된 시야)은 절대 지향을 보정하는 데 사용되며, 이는 발사 환경이 사전 비행 스타 트래커-영상기 정렬을 흐트릴 수 있기 때문입니다. 브라우즈 산출물은 POLAR UVI 아카이브 영상을 사용해 IMAGE FUV 시야를 모형화(그림 8)하며, IMAGE FUV FOV는 POLAR UVI FOV의 약 2배입니다. SI MCP의 펄스 높이 분포는 지속 다운링크되어 MCP 이득 변화를 HV 조정으로 보상 가능. WIC 펄스 높이 분포는 TDI를 정지하고 단일 프레임을 읽어 가끔 회복. GEO 데이터는 모델 파라미터로 보고(영상 없음); 매개변수화된 H 외기 프로파일(예: 그림 8의 Rev #319, 지심거리 44 000 km에서 Ly-α에 대해 $N_0 = 1.2$, $a = 3.2$, $b = 0.27$, $c = 0.051$)을 3 Ly-α 튜브의 회전-분해 측정값에 적합합니다.

---

## 3. Key Takeaways / 핵심 시사점

1. **Three complementary channels = full FUV physics** — WIC (LBH morphology), SI-1356 (OI 135.6 electron precipitation), SI-1216 (Doppler-Ly-α proton aurora), and GEO (geocoronal H column) together cover the four scientific quantities needed to interpret IMAGE's magnetospheric remote sensing. No single broadband or narrowband filter could substitute for the spectrograph because in-band rejection of 130.4 (vs. 135.6) and 121.5667 (vs. hot Ly-α) below 1% is required. 세 상호보완 채널이 LBH 형태, OI 135.6 전자 강하, 도플러 Ly-α 양성자 오로라, 지구코로나 H를 모두 담당합니다. 130.4 nm 대비 1% 이하 거부와 121.5667 nm 대비 동시 부합 거부는 협대역 필터로 불가능하므로 분광기가 필수입니다.

2. **Spinning platform → TDI is essential** — IMAGE rotates at 3°/s with a 2-min spin period; each FUV imager has only 5 s (SI) or 10 s (WIC) of Earth-pointing per spin. Storing each frame and offsetting in memory before co-adding (TDI) is the only way to recover usable signal without ballooning the telemetry budget. 자전 플랫폼에서 한 회전당 5–10 s만 지구를 보므로, 메모리 오프셋 누적(TDI)이 통신량을 감내하면서 신호를 회복할 유일한 방법입니다.

3. **On-board polynomial distortion correction via 12-bit LUT** — Wide-field reflective optics introduce significant pincushion/barrel distortion. Real-time correction at 100 ns/pixel rules out per-pixel splitting; the chosen LUT scheme uses a 6-coefficient quadratic (SI) or 10-coefficient cubic (WIC) polynomial determined by ground least-squares fit, with a randomizer reducing fixed-pattern non-uniformity to <1/16. ETU tests on a rotating distorting mirror demonstrated <0.5 pixel error. 광시야 반사 광학의 핀쿠션/배럴 왜곡을 100 ns/픽셀로 보정하기 위해, SI 6계수 2차/WIC 10계수 3차 다항식 LUT와 randomizer로 비균일성 <1/16, 오차 <0.5 픽셀을 달성했습니다.

4. **Doppler grills make proton-aurora imaging possible** — The cold geocoronal Ly-α at 121.5667 nm is orders of magnitude brighter than the Doppler-shifted hot proton-aurora Ly-α; SI-1218's anti-coincidence Doppler grills selectively block 121.5667 (and 120.0) while passing the broader 119–126 nm hot component. This is the world-first capability that distinguishes IMAGE FUV from all earlier missions. 차가운 지구코로나 Ly-α보다 자릿수가 작은 양성자 오로라 도플러 편이 Ly-α를 선택적으로 통과시키는 SI-1218의 도플러 격자 동시 부합 기법은 IMAGE FUV를 이전 모든 임무와 구별하는 세계 최초 능력입니다.

5. **Rayleigh-based radiometric budgeting** — For a 100 R source, a WIC pixel collects $0.10 \times A_e \times 10\ \mathrm{s}$ photons. With $A_e = 0.04$ cm² this is 40 photons → ~23 counts after detector efficiency. The radiometric budgeting is straightforward Poisson statistics ($\mathrm{SNR}^2 = Q$), but the FUV efficiency of 1–2% means that geometric apertures must be a factor 50–100 larger than the formal $A_e$. 100 R 신호에서 WIC 픽셀당 약 40 광자(검출 후 23 카운트)는 $A_e = 0.04$ cm²로부터 곧바로 계산되지만, 1–2% FUV 효율 때문에 기하 면적은 형식적 $A_e$의 50–100배가 필요합니다.

6. **Radiation-hardened CCD shielding** — IMAGE crosses the radiation belts each orbit, accumulating 50 kRad in 2 years. Charge-trap defects in CCDs degrade charge transfer efficiency under proton flux below 50 MeV; WIC uses a ~500 g tantalum cup specifically designed to reject these protons while allowing the optical path. Without this shielding, WIC would lose effective resolution within months. 매 궤도 방사선대 통과로 2년간 50 kRad 누적; 50 MeV 이하 양성자에 의한 전하 트랩 결함을 방지하기 위해 약 500 g 탄탈룸 컵 차폐가 필수이며, 이것 없이는 WIC 분해능이 수개월 내 저하됩니다.

7. **SI uses photon counting, WIC uses analog frame integration** — Different detector architectures suit different rates. SI's photon-counting XDL anode + TDC encodes each photoelectron's (x, y) position into a memory address, ideal for the relatively faint 135.6 nm and Doppler-Ly-α; WIC's MCP+phosphor+CCD captures full frames at 30 fps, suited to the brighter LBH band. The TDI implementation differs accordingly — SI modifies recorded photoelectron addresses, WIC processes whole CCD frames. SI는 광자 계수, WIC는 아날로그 프레임 적분: SI XDL 양극+TDC가 광전자별 (x,y) 주소 인코딩으로 어두운 135.6/도플러 Ly-α에 적합하고, WIC MCP+인광체+CCD가 30 fps 전체 프레임으로 더 밝은 LBH 밴드에 적합합니다.

8. **System performance margin = factor 1.3–17 over requirements** — Table III vs. Table I shows all four channels meeting or exceeding the science-driven equivalent-aperture requirements, with WIC ~17× over and SI-1356 just 1.3× over. The smallest margin (SI-1356) is the most consequential because OI 135.6 mapping is the most quantitative auroral diagnostic. 모든 채널이 표 I 요구사항을 1.3–17배 초과 충족; WIC 마진이 가장 크고(17×) SI-1356이 가장 작음(1.3×)이며, OI 135.6 매핑이 가장 정량적인 오로라 진단이므로 이 마진이 가장 중요합니다.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Photometric units and signal model / 광도 단위와 신호 모델

The fundamental Rayleigh unit is

$$
1\ \mathrm{R} = \frac{10^{6}}{4\pi}\ \mathrm{photons\ s^{-1}\ cm^{-2}\ sr^{-1}} \approx 80\,000\ \mathrm{photons\ s^{-1}\ cm^{-2}\ sr^{-1}}, \tag{R}
$$

representing a 1 cm² column emitting 10⁶ photons per second isotropically into 4π sr (Hunten, Roach & Chamberlain 1956). The expected count rate per pixel for source brightness $I_R$ (in Rayleighs) is

$$
\dot N_{\mathrm{pix}} = I_R \cdot \Omega_{\mathrm{pix}} \cdot 80\,000 \cdot A_e\ \mathrm{[counts\ s^{-1}\ pix^{-1}]}, \tag{S}
$$

where $\Omega_{\mathrm{pix}}$ is pixel solid angle (sr) and $A_e$ is equivalent aperture (cm²). The total counts per spin exposure $t_{\mathrm{exp}}$ is $N_{\mathrm{pix}} = \dot N_{\mathrm{pix}}\, t_{\mathrm{exp}}$, with Poisson SNR$^2$ = $N_{\mathrm{pix}}$ in the source-limited regime.

기본 레일리 단위는 1 cm² 칼럼이 4π sr로 등방으로 10⁶ 광자/초를 방출하는 강도. 픽셀당 카운트율은 (밝기) × (입체각) × 80 000 × ($A_e$). Poisson 한계에서 SNR² = 카운트수.

### 4.2 Pixel solid angle (small-angle) / 픽셀 입체각

For a square pixel of angular side $\theta$ (radians):

$$
\Omega_{\mathrm{pix}} \approx \theta^2.
$$

WIC: $\theta = 0.09° = 1.57\times10^{-3}$ rad → $\Omega = 2.5\times10^{-6}$ sr (paper quotes $1.3\times10^{-6}$, indicating effective pixel size after distortion correction). SI: $\theta = 0.11° = 1.92\times10^{-3}$ rad → $\Omega = 3.7\times10^{-6}$ sr (paper quotes $4.2\times10^{-6}$).

각 변 $\theta$의 정사각 픽셀에서 $\Omega \approx \theta^2$. WIC 0.09° → 약 $2.5\times10^{-6}$ sr, SI 0.11° → 약 $3.7\times10^{-6}$ sr.

### 4.3 Equivalent aperture / 등가 면적

Total equivalent aperture is

$$
A_e = A_{\mathrm{geom}} \cdot T_{\mathrm{opt}} \cdot \mathrm{QE},
$$

where $A_{\mathrm{geom}}$ is the geometric aperture (cm²), $T_{\mathrm{opt}}$ the integrated optical transmission (mirrors × filters × grating efficiency), and QE the photocathode quantum efficiency. With typical $T_{\mathrm{opt}}\,\mathrm{QE} \sim 1$–$2\%$, achieving $A_e = 0.04$ cm² (WIC) requires $A_{\mathrm{geom}} \sim 2$–$4$ cm² of physical aperture.

등가 면적 = (기하 면적) × (광학 투과율) × (양자 효율). 일반 $T_{\mathrm{opt}}\,\mathrm{QE}\approx 1$–2%에서 $A_e=0.04$ cm² (WIC)를 위해 기하 면적 ~2–4 cm² 필요.

### 4.4 Distortion correction polynomials / 왜곡 보정 다항식

For SI (quadratic, 6 coefficients per coordinate):

$$
x_2 = A_0 + A_1 x_1 + A_2 y_1 + A_3 x_1 y_1 + A_4 x_1^2 + A_5 y_1^2, \tag{1}
$$

$$
y_2 = B_0 + B_1 x_1 + B_2 y_1 + B_3 x_1 y_1 + B_4 x_1^2 + B_5 y_1^2. \tag{2}
$$

For WIC (cubic, 10 coefficients per coordinate):

$$
x_2 = A_0 + A_1 x_1 + A_2 y_1 + A_3 x_1^2 + A_4 y_1^2 + A_5 x_1 y_1 + A_6 x_1^3 + A_7 x_1^2 y_1 + A_8 x_1 y_1^2 + A_9 y_1^3, \tag{3}
$$

$$
y_2 = B_0 + B_1 x_1 + B_2 y_1 + B_3 x_1^2 + B_4 y_1^2 + B_5 x_1 y_1 + B_6 x_1^3 + B_7 x_1^2 y_1 + B_8 x_1 y_1^2 + B_9 y_1^3. \tag{4}
$$

Coefficients $\{A_i, B_i\}$ are obtained by least-squares fitting to a calibration grid where $(x_0, y_0)$ are known illumination angles and $(x_1, y_1)$ are measured detector positions. The least-squares normal equations are $\mathbf{X}^T\mathbf{X}\boldsymbol{\beta} = \mathbf{X}^T\mathbf{y}$ where each row of $\mathbf{X}$ is the polynomial-feature vector for one calibration point.

계수 $\{A_i, B_i\}$는 알려진 조명각 $(x_0,y_0)$과 측정 검출 위치 $(x_1,y_1)$를 갖는 보정 격자에 대해 최소제곱 적합으로 결정되며, 정규방정식 $\mathbf{X}^T\mathbf{X}\boldsymbol{\beta} = \mathbf{X}^T\mathbf{y}$로 풀립니다.

### 4.5 TDI rotation offset / TDI 회전 오프셋

After distortion correction, the rotation offset is added in the direction of spacecraft rotation:

$$
x_n(t) = x_2 - K t, \qquad y_n(t) = y_2, \tag{5}
$$

where $K = 3°/\mathrm{s}$ and $t$ runs from 0 to the exposure duration (10 s for WIC, 5 s for SI). The TDI process accumulates intensity at $(x_n, y_n)$ across all frames within a spin. The synchronization tolerance is

$$
\frac{\Delta K}{K} < \frac{1}{2 N_{\mathrm{pix,TDI}}} = \frac{1}{2 \times 128} \approx 0.4\%,
$$

(i.e., 0.5% in the paper) to keep the cumulative phase error <0.5 pixel over 128 pixels of integration.

왜곡 보정 후 회전 오프셋이 자전 방향으로 더해집니다. 동기화 허용 오차는 128 픽셀 누적 동안 누적 위상 오차 <0.5 픽셀 유지를 위해 약 0.4% (논문은 0.5%)입니다.

### 4.6 Randomizer for fixed-pattern suppression / 고정 패턴 억제용 randomizer

Let $a_{12}$ be the 12-bit corrected address. The 8-bit memory address is

$$
a_8 = \lfloor (a_{12} + r_n)/16 \rfloor, \qquad r_n \in \{0, 1, \ldots, 15\},
$$

with $r_n$ cycling through all 16 values across consecutive frames. The maximum non-uniformity is $\leq 1/16 \approx 6\%$, further smoothed by TDI averaging in the rotation direction.

12비트 주소 $a_{12}$, 8비트 메모리 주소 $a_8$. $r_n$이 16 값을 순환하여 비균일성 ≤1/16 ≈6%를 보장하며 회전 방향 TDI 평균으로 추가 평활화.

### 4.7 Three-channel emission separation / 3채널 방출 분리

The auroral FUV spectrum can be decomposed as

$$
I(\lambda) = I_{\mathrm{LBH}}(\lambda) + I_{\mathrm{OI\,135.6}} \,\delta(\lambda - 135.6) + I_{\mathrm{OI\,130.4}}\,\delta(\lambda - 130.4) + I_{\mathrm{Ly}\alpha,\mathrm{cold}}\,\delta(\lambda - 121.5667) + I_{\mathrm{Ly}\alpha,\mathrm{hot}}(\lambda),
$$

where the LBH band is broadband 140–190 nm and $I_{\mathrm{Ly}\alpha,\mathrm{hot}}$ is Doppler-broadened by several Å. The instrument transfer functions $T_k(\lambda)$ for each channel ($k$ = WIC, SI-1356, SI-1218) yield observed counts

$$
C_k = A_{e,k}\,t_{\mathrm{exp},k}\,\Omega_{\mathrm{pix},k}\int T_k(\lambda)\,I(\lambda)\,d\lambda,
$$

with the design constraints $T_{\mathrm{SI-1356}}(130.4)/T_{\mathrm{SI-1356}}(135.6) < 1\%$ and $T_{\mathrm{SI-1218}}(121.5667) \to 0$ (anti-coincidence grill). These transfer-function design constraints are what drive the choice of a spectrograph over filters.

오로라 FUV 스펙트럼은 LBH 광대역과 여러 단일선의 합. 채널별 응답 $T_k(\lambda)$로 카운트 산출. 설계 제약 $T_{\mathrm{SI-1356}}(130.4)/T_{\mathrm{SI-1356}}(135.6) < 1\%$ 및 $T_{\mathrm{SI-1218}}(121.5667) \to 0$이 분광기 채택 이유입니다.

### 4.8 Numerical worked example / 수치 예제

For a typical 1 kR auroral feature observed by WIC (140–190 nm LBH) at apogee:

- Pixel solid angle $\Omega = 1.3\times10^{-6}$ sr (paper Table I value)
- Photon collection efficiency $\eta = 80\,000 \times 1.3\times10^{-6} = 0.104$ photons s⁻¹ R⁻¹ cm⁻² (per pixel)
- Equivalent aperture $A_e = 0.04$ cm² (Table III)
- Exposure $t_{\mathrm{exp}} = 10$ s
- Counts: $N = 1000\ \mathrm{R} \times 0.104 \times 0.04 \times 10 \approx 42$ counts/pixel
- Poisson SNR: $\sqrt{42} \approx 6.5$

For 100 R faint auroral arc: $N \approx 4.2$ counts, SNR ≈ 2.0 — consistent with the "1 count per pixel for 100 R minimum detection" requirement of Table I when binned over a few pixels.

원지점에서 WIC가 1 kR LBH 특징을 관측할 때: 픽셀 입체각 $1.3\times10^{-6}$ sr, 효율 0.104 photons s⁻¹ R⁻¹ cm⁻², $A_e = 0.04$ cm², 10 s 노출 → 픽셀당 약 42 카운트, SNR ~6.5. 100 R 약한 오로라 호: 약 4.2 카운트, SNR ~2.0.

### 4.9 SI-1218 Doppler grill rejection model / SI-1218 도플러 격자 거부 모델

The cold geocoronal Lyman-α line is centered at $\lambda_0 = 121.5667$ nm with thermal width corresponding to $T \sim 1000$ K (Doppler width $\Delta\lambda_{\mathrm{cold}} \sim 5\times10^{-3}$ nm). Proton-aurora Ly-α produced by precipitating ~1 keV protons (after charge exchange to fast hydrogen) is Doppler-shifted away from the observer by several Å (the precipitating beam moves toward the atmosphere, downward), giving a hot component centered roughly at 121.5 nm but smeared over $\Delta\lambda_{\mathrm{hot}} \sim 0.5$–$1.0$ nm. The anti-coincidence grill is designed so its blocking transmission $T_{\mathrm{grill}}(\lambda)$ has a deep notch at $\lambda_0$ but is open elsewhere in the 119–126 nm band:

$$
T_{\mathrm{SI-1218}}(\lambda) = T_{\mathrm{filter}}(\lambda) \cdot [1 - T_{\mathrm{grill}}(\lambda)],
$$

where $T_{\mathrm{filter}}$ is the broadband 119–126 nm pass-band and $T_{\mathrm{grill}}(\lambda_0) \to 1$ (full block at cold Ly-α) but $T_{\mathrm{grill}}(\lambda) \to 0$ for $|\lambda - \lambda_0| \gtrsim 0.1$ nm. The result is a high transmission to the broad hot component while suppressing the line-center cold component by a factor of $\gtrsim 100$.

차가운 지구코로나 Ly-α는 $\lambda_0 = 121.5667$ nm 중심, T~1000 K 도플러 폭 약 $5\times10^{-3}$ nm. 양성자 오로라 Ly-α는 약 1 keV 양성자(전하 교환 후 빠른 수소)에 의한 도플러 편이로 121.5 nm 부근에 0.5–1.0 nm 폭으로 퍼짐. 동시 부합 격자는 $\lambda_0$에 깊은 노치를 갖되 119–126 nm에서는 열려 있어, 광대역 뜨거운 성분은 통과시키고 선 중심 차가운 성분은 100배 이상 억제합니다.

### 4.10 Geocoronal H profile parameterization / 지구코로나 H 프로파일 매개변수화

The Hodges (1994) / Bishop (1999) framework parameterizes the H number density as a function of geocentric distance $r$ (in Earth radii) by

$$
n_H(r) = N_0 \left( \frac{1}{r^a} + b\,e^{-c r} \right),
$$

(or similar separable forms). Figure 8 reports $N_0 = 1.2$, $a = 3.2$, $b = 0.27$, $c = 0.051$ for IMAGE Rev #319 at 44 000 km. The GEO photometers measure the line-of-sight integral

$$
B_{\mathrm{Ly}\alpha}(\hat l) = g(\lambda_0)\,\int_{\mathrm{LOS}} n_H(r(s))\,ds\ \mathrm{[Rayleighs]},
$$

where $g(\lambda_0)$ is the resonance-scattering g-factor for solar Ly-α illumination. By scanning the spin axis through 360° during one rotation, three view-angle pairs yield enough data to fit the four parameters $\{N_0, a, b, c\}$ each spin.

Hodges/Bishop 모델은 H 수밀도를 $r$의 함수로 매개변수화: $n_H(r) = N_0 (1/r^a + b\,e^{-c r})$. 그림 8 예시 값은 $N_0=1.2$, $a=3.2$, $b=0.27$, $c=0.051$. GEO는 시선 적분 $B_{\mathrm{Ly}\alpha} = g(\lambda_0) \int n_H(r)\,ds$를 측정하며, 한 회전당 360° 스캔으로 4개 매개변수를 적합합니다.

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1956 ─ Hunten, Roach, Chamberlain: "A photometric unit for the airglow and aurora"
       Defines the Rayleigh; foundation for all FUV radiometry to follow.
       레일리 단위 정의; 이후 모든 FUV 광도 측정의 기초.

1972 ─ Coroniti & Kennel: substorm growth-phase tail flaring (J. Geophys. Res.)
       Sets the timescale physics that IMAGE substorm imaging targets.

1981 ─ Frank, Craven et al.: DE-1 global auroral imaging instrumentation
       First successful global FUV auroral imager; demonstrates dayglow contrast.
       최초 성공적 전 지구 FUV 오로라 영상기.

1983 ─ Strickland, Anderson: OI 1356 radiation transport (J. Geophys. Res.)
       Justifies 135.6 nm as a clean tracer (limited multiple scattering).
       135.6 nm가 산란이 제한된 깨끗한 추적자임을 입증.

1986 ─ Rairden, Frank, Craven: DE-1 geocoronal Lyman-α imaging
       Direct precedent for IMAGE/GEO geocoronal photometry.

1987 ─ Anger et al.: Viking UV imager (Geophys. Res. Lett.)
       Proves polar-orbit FUV imaging in dayglow.

1991 ─ Meier: "Ultraviolet spectroscopy and remote sensing of upper atmosphere" review
       Definitive reference for FUV emission physics; the spectrum of Figure 1
       is built on this.

1994 ─ Hodges: Monte Carlo H exosphere model (J. Geophys. Res.)
1994 ─ Murphree et al.: Freja UV imager (Space Sci. Rev.)
       Both feed directly into IMAGE FUV/GEO design.

1995 ─ Torr et al.: POLAR UVI for ISTP (Space Sci. Rev.)
       The mock-up source of IMAGE FUV browse products (Figure 8).

1996 ─ IMAGE FUV contract awarded (June)
1999 ─ FUV flight complement delivered (January)

2000 ──── THIS PAPER (Mende et al. 2000, SSR 91) — and companion papers:
        Paper 2: WIC details (Mende et al. 2000b)
        Paper 3: SI and GEO details (Mende et al. 2000c)
        IMAGE launch (March 25)

2001+ ─ Frey, Mende et al.: SI-1218 proton-aurora science (theta aurora,
        cusp imaging, substorm onset latitude); IMAGE FUV becomes the
        standard auroral observatory.

2005 ─ IMAGE communications loss (December)
       FUV archive remains the benchmark dataset for global auroral imaging.

2018+ ─ ICON/MIGHTI, GOLD FUV — successor missions inheriting the
        scanning + spectrograph + LUT-corrected approach.
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Frank et al. 1981 (DE-1 imager) | First global FUV auroral imager; informed IMAGE FOV and dayglow rejection requirements | High — direct heritage |
| Anger et al. 1987 (Viking UV) | Demonstrated FUV imaging from a polar-orbit spinner | High — operational precedent |
| Murphree et al. 1994 (Freja UV) | Provided detector heritage and small-FOV high-resolution lessons | Medium — design heritage |
| Torr et al. 1995 (POLAR UVI) | Used as mock-up source for IMAGE FUV browse products; competing high-resolution UV imager | High — direct comparison/calibration cross-reference |
| Strickland & Anderson 1983 (OI 1356 radiation transport) | Justifies 135.6 nm as quantitative tracer; underlies SI-1356 channel choice | High — drives the spectrograph requirement |
| Rairden, Frank, Craven 1986 (DE-1 Ly-α geocoronal imaging) | Direct heritage for the GEO photometer concept | High — GEO heritage |
| Meier 1991 (FUV review) | Theoretical foundation of the spectrum in Figure 1 | High — physics context |
| Hodges 1994; Bishop 1999 (H exosphere models) | Models that GEO data are fit to provide the geocoronal density used by neutral-atom imagers | High — data-product physics |
| Williams 1990 ("Why we need global observations") | Strategic argument for the IMAGE mission | High — mission rationale |
| Coroniti & Kennel 1972 (substorm growth phase) | Provides the physics targeted by simultaneous IMAGE FUV + RPI observations | Medium — science context |
| Mende et al. 2000b (companion Paper 2: WIC details) | Detailed WIC instrument description; this paper summarizes its requirements | Highest — companion |
| Mende et al. 2000c (companion Paper 3: SI/GEO) | Detailed SI and GEO description and calibration | Highest — companion |

---

## 7. References / 참고문헌

- Mende, S. B., Heetderks, H., Frey, H. U., Lampton, M., Geller, S. P., Habraken, S., Renotte, E., Jamar, C., Rochus, P., Spann, J., Fuselier, S. A., Gerard, J.-C., Gladstone, R., Murphree, S. and Cogger, L. (2000), "Far Ultraviolet Imaging from the IMAGE Spacecraft. 1. System Design", *Space Science Reviews*, 91, 243–270. DOI: 10.1023/A:1005271728567
- Hunten, D. M., Roach, F. E. and Chamberlain, J. W. (1956), "A photometric unit for the airglow and aurora", *J. Atmos. Terr. Phys.* **8**, 345–346.
- Frank, L. A., Craven, J. D., Ackerson, K. L., English, M. R., Eather, R. H. and Crovillano, R. L. (1981), "Global auroral imaging instrumentation for the Dynamics Explorer Mission", *Space Sci. Instrum.* **5**, 369–393.
- Frank, L. A. and Craven, J. D. (1988), "Imaging Results from Dynamics Explorer 1", *Rev. Geophys.* **2**, 249.
- Anger, C. D. et al. (1987), "An Ultraviolet Auroral Imager for the Viking Spacecraft", *Geophys. Res. Lett.* **14**, 387.
- Murphree, J. S. et al. (1994), "The Freja Ultraviolet Imager", *Space Sci. Rev.* **70**, 421–446.
- Torr, M. R. et al. (1995), "A Far Ultraviolet Imager for the International Solar-Terrestrial Physics Mission", *Space Sci. Rev.* **71**, 329.
- Strickland, D. J. and Anderson, D. E., Jr. (1983), "Radiation Transport Effects on the OI 1356-AA Limb Intensity Profile in the Dayglow", *J. Geophys. Res.* **88**, 9260.
- Strickland, D. J., Jasperse, J. P. and Whalen, J. A. (1983), "Dependence of Auroral FUV Emissions on the Incident Electron Spectrum and Neutral Atmosphere", *J. Geophys. Res.* **88**, 8051–8062.
- Strickland, D. J., Daniell, R. E., Jr., Jasperse, J. R. and Basu, B. (1993), "Transport-Theoretic Model for the Electron-Proton-Hydrogen Atom Aurora", *J. Geophys. Res.* **98**, 21533.
- Rairden, R. L., Frank, L. A. and Craven, J. D. (1986), "Geocoronal Imaging with Dynamics Explorer", *J. Geophys. Res.* **91**, 13613.
- Meier, R. R. (1991), "Ultraviolet Spectroscopy and Remote Sensing of the Upper Atmosphere", *Space Sci. Rev.* **58**, 1.
- Hodges, R. R. (1994), "Monte Carlo Simulation of the Terrestrial Hydrogen Exosphere", *J. Geophys. Res.* **99**, 23229.
- Bishop, J. (1999), "Transport of Resonant Atomic Hydrogen Emissions in the Thermosphere and Geocorona", *J. Quant. Spectrosc. Radiat. Transfer* **61**, 473.
- Coroniti, F. V. and Kennel, C. F. (1972), "Changes in Magnetospheric Configuration During the Substorm Growth Phase", *J. Geophys. Res.* **19**, 3361.
- Williams, D. J. (1990), "Why We Need Global Observations", in B. Hultquist and C. G. Fälthammer (eds.), *Magnetospheric Physics*, Plenum, 83–101.
- Roelof, E. C. (1987), "Energetic Neutral Atom Image of the Storm Time Ring Current", *Geophys. Res. Lett.* **14**, 652.
- Lui, A. T. Y., Williams, D. J., Roelof, E. C., McEntire, R. W. and Mitchell, D. G. (1996), "First Composition Measurements of Energetic Neutral Atoms", *Geophys. Res. Lett.* **23**, 2641–2644.
- Gladstone, G. R. (1994), "Simulations of DE-1 UV Airglow Images", *J. Geophys. Res.* **99**, 11441.
- Drob, D. P. et al. (1999), "Atomic Oxygen in the Thermosphere During the July 13, 1982, Solar Proton Event Deduced from Far Ultraviolet Images", *J. Geophys. Res.* **104**, 4267.
