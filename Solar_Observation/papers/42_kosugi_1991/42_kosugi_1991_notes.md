---
title: "The Hard X-ray Telescope (HXT) for the SOLAR-A Mission"
authors: T. Kosugi, K. Makishima, T. Murakami, T. Sakao, T. Dotani, M. Inda, K. Kai, S. Masuda, H. Nakajima, Y. Ogawara, M. Sawa, K. Shibasaki
year: 1991
journal: "Solar Physics 136, 17–36"
doi: "10.1007/BF00151693"
topic: Solar_Observation
tags: [hard_x_ray, fourier_synthesis, modulation_collimator, YOHKOH, SOLAR-A, HXT, MEM, CLEAN, NaI_scintillator, flare_imaging]
status: completed
date_started: 2026-04-25
date_completed: 2026-04-25
---

# 42. The Hard X-ray Telescope (HXT) for the SOLAR-A Mission / 솔라-A 미션을 위한 경 X-선 망원경 (HXT)

---

## 1. Core Contribution / 핵심 기여

Kosugi et al. (1991)는 일본 SOLAR-A (= YOHKOH) 위성에 탑재된 Hard X-ray Telescope (HXT)의 설계, 영상 합성 원리, 그리고 비행 모델 성능을 종합 보고한다. HXT는 **세계 최초의 Fourier-synthesis 형태 X-ray 영상 망원경**으로, 64개의 독립 부시준기 (subcollimator)가 각각 태양 X-ray 휘도 분포의 하나의 일반화된 복소 푸리에 성분을 측정한다. 측정된 64개 점 (16개 fanbeam + 48개 Fourier element = 24쌍의 cos/sin)을 **MEM (Maximum Entropy Method)** 또는 **modified CLEAN** 알고리즘으로 합성하여 ~5″ 각도 분해능, 0.5 s 시간 분해능의 영상을 4개 에너지 대역 (15(19)–24, 24–35, 35–57, 57–100 keV) 에서 동시에 얻는다. 유효 면적 ~70 cm²는 HINOTORI 대비 ~1 자릿수 큰 수치이다.

Kosugi et al. (1991) present the design, image-synthesis principle, and pre-flight performance of the Hard X-ray Telescope (HXT) onboard the Japanese SOLAR-A (= YOHKOH) satellite. HXT is **the world's first Fourier-synthesis X-ray imager**: 64 independent subcollimators each measure one generalised complex Fourier component of the solar X-ray brightness distribution. The 64 measurements (16 fanbeam + 48 Fourier elements = 24 cosine/sine pairs) are synthesised into images via **MEM (Maximum Entropy Method)** or a **modified CLEAN** algorithm, giving ~5″ angular resolution, 0.5 s time resolution, simultaneously in four energy bands (15(19)–24, 24–35, 35–57, 57–100 keV). The total effective area of ~70 cm² is about an order of magnitude larger than HINOTORI's. HXT thereby provides, for the first time, hard X-ray flare images above ~30 keV, directly probing the acceleration sites of nonthermal electrons.

이 논문은 또한 NaI(Tl) 25 mm × 5 mm 신틸레이터 + 광전증배관 (PMT) 검출기, 텅스텐 격자 0.5 mm 두께, CFRP 미터링 튜브 (열팽창 < 10⁻⁶ /K), HXA 가시광 자세 시스템 (1–2″ 정확도) 등 정밀 공학 디테일을 모두 명세하여, 푸리에 영상 망원경의 캘리브레이션과 조립 방법론을 정립한다. 본 기기는 1994년 **Masuda flare** 발견 (above-the-loop-top hard X-ray source)으로 이어졌으며, 2002년의 RHESSI와 2020년의 Solar Orbiter/STIX에까지 영향을 미친 hard X-ray 영상 천문학의 기초가 되었다.

The paper additionally specifies engineering details — NaI(Tl) 25 mm × 5 mm scintillators with photomultiplier tubes, 0.5 mm-thick tungsten grids, a CFRP metering tube (thermal expansion <10⁻⁶/K), and a visible-light HXA aspect system with 1–2″ accuracy — establishing a methodology for calibration and assembly of Fourier-imaging X-ray telescopes. HXT enabled the 1994 **Masuda flare** discovery (above-the-loop-top hard X-ray source) and laid the foundation for hard X-ray imaging astronomy that propagates to RHESSI (2002) and Solar Orbiter/STIX (2020).

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction (Section 1) / 서론

논문은 SOLAR-A를 ISAS의 두 번째 태양 플레어 전용 위성 (HINOTORI 1981 이후) 으로 위치시킨다. SMM (NASA, 1980)과 HINOTORI는 모두 hard X-ray 영상의 가능성을 보여줬으나 결정적 한계가 있었다. (i) 둘 다 30 keV 이하의 1–2개 에너지 대역만; (ii) 각도 분해능이 5–10″ 이상 부족; (iii) 시간 분해능 1.5–10 s. 이로 인해 1980년대 말 비열적 전자의 가속 위치, 메커니즘, 자기장 위상학 등 핵심 질문에 답할 수 없었다 (Dennis 1988). HXT는 이러한 한계를 해결하기 위해 설계되었다.

The paper positions SOLAR-A as ISAS's second flare-dedicated satellite (after HINOTORI 1981). SMM (NASA 1980) and HINOTORI demonstrated hard X-ray imaging feasibility but with critical limitations: (i) only 1–2 energy bands below 30 keV; (ii) insufficient angular resolution (5–10″+); (iii) limited time resolution (1.5–10 s). These prevented answering fundamental questions about nonthermal electron acceleration, locations, and magnetic topology (Dennis 1988). HXT addresses all three.

Hard X-ray가 핵심인 이유: bremsstrahlung은 고에너지 전자–이온 충돌의 직접 결과이며, 광자가 태양 대기를 거의 흡수 없이 통과하므로 가속 영역의 직접적 진단을 제공한다. 영상 (이미징)이 1차원 시간 프로파일보다 결정적으로 우월한 이유다.

Hard X-rays matter because bremsstrahlung is the direct collision product of high-energy electrons with ions, and the photons traverse the solar atmosphere essentially unaffected. This makes spatially resolved imaging — not just time profiles — the decisive diagnostic of the acceleration region.

### Part II: HXT Overview (Section 2) / HXT 개요

기기는 세 모듈로 구성: **HXT-C (collimator)** + **HXT-S (sensors)** + **HXT-E (electronics)**. (Fig. 1 참조)

The instrument has three modules: **HXT-C (collimator)** + **HXT-S (sensors)** + **HXT-E (electronics)** (cf. Fig. 1).

- **HXT-C**: 1400 mm 길이의 미터링 튜브 (417 × 376 × 1400 mm), 양쪽에 텅스텐 0.5 mm 격자판, 무게 ~13.5 kg. 중앙축에 HXA 가시광 자세 광학 (lens + filter + CCD)을 통합.
- **HXT-S**: 64개 검출기 모듈 (각 NaI(Tl) 25 mm² × 5 mm 두께 + PMT), 8개의 고압 전원, 2개의 1D CCD (HXA용), 465 × 392 × 223 mm, 17.1 kg.
- **HXT-E**: pulse-height analysis 후 4개 에너지 채널로 분기 카운팅, 374 × 246 × 220 mm, 10.8 kg. 매 0.5 s마다 데이터 처리기 (DP)로 전송.

HXT-C is a 1400-mm-long metering tube (417×376×1400 mm) carrying tungsten 0.5-mm grid plates at both ends, weighing ~13.5 kg, with the HXA optics integrated along its central axis. HXT-S is an assembly of 64 detector modules (each NaI(Tl) 25-mm square × 5 mm thick + PMT), 8 high-voltage supplies, and 2 1D CCDs for HXA (465×392×223 mm, 17.1 kg). HXT-E performs pulse-height analysis and bins counts into four energy channels, transmitting data to the spacecraft data processor (DP) every 0.5 s (374×246×220 mm, 10.8 kg).

**Table I 비교 / Table I (comparison)**:

| 항목 / Item | HXT (SOLAR-A) | HINOTORI imager | HXIS (SMM) |
|---|---|---|---|
| 시준기 종류 / Collimator | Multi-el. bigrid MC | Rotating bigrid MC | Multi-el. IC |
| 부시준기 수 / # elements | **64** | 2 (orthogonal) | (F)304; (C)128 |
| 영상 획득 / Image acquisition | **2D Fourier synthesis** | 1D scans → 2D | 1 el./pixel |
| 각도 분해능 / Angular res. | **~5″** | ~10″ | 8″ (32″) |
| FOV | 전 태양 / whole Sun | 전 태양 / whole Sun | 2'40″ (6'24″) |
| Synthesis aperture | 2'06″ | 2'12″ | – |
| 시간 분해능 / Time res. | **0.5 s** | ~10 s | 1.5–9 s |
| 에너지 채널 / Energy bands | **4** (~15–100 keV) | 1 (5(17)–40) | 6 (3.5–30) |
| 검출기 / Detector | NaI(Tl) 25 mm² × 64 | NaI(Tl) 120 mmØ × 2 | Gas prop. counter |
| 유효 면적 / Effective area | **~70 cm²** | ~8 cm² × 2 | 0.07 cm²/pixel |

HXT의 핵심 우위는 이 표에서 자명하다 — 더 높은 에너지 (~100 keV), 더 큰 감도, 개선된 각도/시간 분해능, 전 태양 시야, 그리고 같은 위성의 SXT (Tsuneta et al. 1991)와의 동시 관측.

HXT's advantages are clear from this table: higher energy (~100 keV), much greater sensitivity, improved angular/temporal resolution, full-Sun FOV, and simultaneous observation with SXT (Tsuneta et al. 1991) on the same satellite.

**과학 목표 / Science objectives** (논문이 8개로 명시):
1. 전자 가속 위치 — single-loop vs. multiple-loop interaction site? / acceleration location
2. 가속 메커니즘 — 자기장 평행/수직, 무작위, 조건? / acceleration mechanism
3. Type B (impulsive) flare의 double-source 구조 원인? / double-source origin
4. Type B vs. Type C (gradual) flare의 가속 차이? / impulsive vs. gradual
5. 플레어 종류를 결정하는 변수? (밀도, β, 위상학) / what sets flare type
6. 전자와 이온 가속의 동시성/동일 메커니즘? (GRS 동시 관측) / electron–ion correlation
7. 전자의 자기 루프 내 confinement, 재가속? / confinement and reacceleration
8. 플레어 전체에서 가속의 역할 — 부수적인가 핵심인가? / role of acceleration in flares

These eight questions structure the paper's scientific motivation and define the instrument requirements.

### Part III: Design and Image Synthesis Principles (Section 3) / 설계와 영상 합성 원리

#### 3.1 Fourier-synthesis principle / 푸리에 합성 원리

HXT는 라디오 간섭계의 aperture-synthesis 사상을 X-ray로 가져온 첫 사례다. **각 부시준기 = (k,θ) 평면의 한 점을 측정**.

HXT brings the aperture-synthesis idea of radio interferometry into X-rays. **Each subcollimator measures one point in the (k,θ) plane**.

기본 변조 시준기는 **bigrid** 구조: 같은 슬릿/피치를 가진 두 격자가 거리 L 떨어져 있고, 슬릿 너비 = 피치/2. 이 구조의 투과율은 광원 각도에 따라 **삼각 패턴 (triangular)** 이 된다. 삼각 패턴은 DC 성분을 빼고 보면 cosine과 매우 유사 (실제로 푸리에 급수의 기본 항이 cos이다).

The basic modulation collimator is a **bigrid**: two identical grids of slit pitch p and slit width p/2 separated by distance L. The transmission as a function of source angle is a **triangular** pattern. Subtracting the DC component, this is essentially cosine-like (its fundamental Fourier term is a cosine).

**Eq. 1**: 한 부시준기의 투과 함수
$$F_c(k\rho), \quad \rho = X\cos\theta + Y\sin\theta$$
- $k$ = 1, 2, ... : 파수 (wave number) — 기본 주기 (synthesis aperture = 2'06″ = $L_0$) 의 정수배
- $\theta$ : 격자 슬릿의 자세각
- $(X,Y)$ : 천구상 좌표를 기본 주기로 정규화한 값

**Eq. 2**: 사인 짝 (sine partner)
$$F_s(k\rho) = F_c(k\rho - \pi/2)$$
같은 $(k,\theta)$의 두 번째 부시준기는 슬릿이 1/4 피치만큼 시프트되어 cos vs. sin의 90° 위상 차이를 가진다.

**Eq. 3, 4**: 측정값 = 휘도 분포의 푸리에 성분
$$b_c(k,\theta) = A\!\!\int\! B(X,Y)\,F_c(k\rho)\,dX\,dY$$
$$b_s(k,\theta) = A\!\!\int\! B(X,Y)\,F_s(k\rho)\,dX\,dY$$
- $b_c, b_s$ : 두 부시준기에서 0.5 s 간 측정한 광자 수
- $A$ : 부시준기의 유효 면적
- $B(X,Y)$ : 복원 대상 X-ray 휘도 분포 (target image)

따라서 cos/sin 짝 = $(k,\theta)$ 점에서의 **하나의 일반화된 복소 푸리에 성분** $b_c + i b_s$.

So each cos/sin pair yields **one generalised complex Fourier component** $b_c + i b_s$ at (k,θ).

#### 3.2 Why "generalised" Fourier? / 왜 "일반화된" 푸리에인가?

엄밀한 푸리에 변환에서 $B(X,Y)$를 복원하려면 (i) 측정된 함수가 정확한 cos/sin이어야 하고, (ii) $(k,\theta)$ 샘플링이 충분해야 한다. HXT는 **두 조건 모두 완전히 만족하지 못한다** — 투과 함수가 삼각형이지 정확한 cos/sin이 아니고, 64개 점만 샘플링한다. 이 때문에 직접 inverse FT를 쓸 수 없다.

A strict Fourier inversion requires (i) exact cos/sin transmission functions and (ii) sufficient (k,θ) sampling. HXT satisfies **neither condition completely** — the transmission is triangular (not exact sinusoid) and only 64 points are sampled. Direct inverse FT therefore fails.

해결책: 라디오 천문학에서 발달한 **CLEAN** (Högbom 1974) 과 **MEM** (Frieden 1972, Gull & Daniell 1978, Willingale 1981) 두 알고리즘을 병행. 두 방법 모두 함수계의 직교성/완전성을 요구하지 않으며, MEM은 양수 해를 정규화로 강제하고, CLEAN은 점원으로 잔차를 분해한다. 원래 CLEAN은 dirty map의 FT를 사용하지만 HXT에서는 그게 불가능하므로 **modified CLEAN** 으로 대체 (정확한 절차는 본 논문에서 명시되지 않음, 후속 논문 Sakao 1994 참고).

The solution: deploy two algorithms developed in radio astronomy in parallel — **CLEAN** (Högbom 1974) and **MEM** (Frieden 1972, Gull & Daniell 1978, Willingale 1981). Neither requires orthogonality/completeness; MEM regularises with a positivity prior, CLEAN decomposes residuals as point sources. The standard CLEAN uses an FT of the dirty map, which HXT cannot rely on; HXT therefore uses a **modified CLEAN** (full procedure deferred to follow-up papers e.g., Sakao 1994).

#### 3.3 (u,v)-plane sampling / (u,v) 평면 샘플링

기본 시야 단위는 $L_0 = 2'06''$로, 가장 낮은 wave number $k=1$에 해당. 가장 높은 wave number $k_\text{max} = 8$로 선택 (그리드 제작의 한계와 절충). HXT는 다음 a priori 가정을 두었다.

The fundamental FOV unit is $L_0 = 2'06''$, corresponding to $k=1$. The highest wave number is $k_\text{max} = 8$ (compromise with fine-grid fabrication). Two a priori assumptions:

1. Hard X-ray flare의 각도 크기 < ~2 arcmin → fundamental period 2'06″ 채택 / typical flare extent <2′
2. (k, θ) 격자가 직사각형이면 grating response가 생겨 위치 모호성 발생 → **대칭 polar diagram**으로 배치 / use polar (k,θ) layout to avoid grating ambiguity

**최종 64 SC 배열** (Fig. 3, Fig. 5a):
- 16개 **fanbeam elements**: $k = 1, 2$, 4 position angle (0°, 45°, 90°, 135°), 각 angle마다 4개 위상 (90° step). 1D 부채살 방향으로만 위치 정보 제공. → 4 × 4 = 16
- 48개 **Fourier elements**: $k = 3, 4, 5, 6, 7, 8$, 6 position angle (0°, 30°, 60°, 90°, 120°, 150°), 각 (k, θ)마다 cos/sin 짝. → 8 × 6 = 48

Total = 16 + 48 = 64.

Fanbeam이 낮은 k에서 cos/sin pair 대신 사용된 이유: 시뮬레이션 결과 광원이 약간 확장된 경우 high-k Fourier 성분 (cos/sin pair) 만으로는 영상 품질이 나빠짐 → 낮은 k에서 fanbeam 으로 보완 (Fig. 3 (b)). 반대로 high-k에서 fanbeam을 쓰면 분해능이 떨어진다.

Why fanbeam instead of cos/sin at low k: simulations show that for slightly extended sources, an all-Fourier set degrades image quality; replacing low-k Fourier pairs with fanbeam elements rescues extended-source fidelity (Fig. 3b). At high k, fanbeam would degrade angular resolution.

#### 3.4 시뮬레이션과 영상 품질 / Simulations and image quality (Fig. 4)

세 가지 광원 모델로 MEM 시뮬레이션:
Three source models simulated with MEM:
- **Compact scattered sources**: 6″, 8″, 11″, 15″ separation
- **Double sources**: ~7-8″ separation
- **Diffuse sources**: 확장된 영역

각 케이스에 X-ray fluence ~2 × 2 × 1000 c/cm² 와 phase error 1.0' rms, gain error 5%를 추가. 결과:
- 각도 분해능 ~5″ 확정 / angular resolution ~5″ confirmed
- $\chi^2 \approx 1.0–1.8$, $\delta \approx 6.0–13.3\%$ — 잘 수렴 / good convergence

관측 오차 분석:
1. effective area / phase error는 정확히 평가되면 영상 합성 단계에서 제거 가능 / removable if accurately calibrated
2. 단, **effective area rms < 5%** 와 **peak position rms < 1″** 두 기준을 어기면 영상이 매우 불안정 / image becomes unstable if either threshold violated
3. effective area는 검출기 게인에도 의존 → flare의 power-law spectrum 지수 γ ≈ 4 가정 시, **gain 정확도 ~1%** 가 필요 / γ ≈ 4 spectrum implies 1% gain accuracy needed

### Part IV: The Instrument (Section 4) / 기기

#### 4.1 Collimator (HXT-C) / 시준기

**4.1.1 미터링 튜브 / Metering tube**:
- 박스 모양 1400 mm 길이, 격자판 두 개 (각 402 × 362 mm, ~3 kg) 받침
- **CFRP** (carbon-fiber reinforced plastic, ρ ~1.7 g/cm³), 12-층 quasi-isotropic stratification
- 열팽창 계수 α < **1 × 10⁻⁶/K**
- 두께: 측면 1.3 mm, 양 끝면 2.1 mm; 옆면에 stiffener 3개 (위성 본체 부착점도 겸함); 튜브 자체 ~8.0 kg

The metering tube (1400 mm long) is a CFRP box with α < 1 × 10⁻⁶/K thanks to a 12-ply quasi-isotropic carbon-fibre stack. Walls: 1.3 mm side, 2.1 mm faces, with three stiffeners doubling as spacecraft attachments. Tube mass: 8.0 kg.

**Eq. 6** (비틀림 강성 / Twist stiffness):
$$\theta(\text{arcmin}) = 0.025\,T\,(\text{kg m})$$
T = 적용 토크 (kg·m), θ = 결과 비틀림. CFRP 튜브는 매우 강성 (위성 중앙 패널보다 강성) 이지만, 패널의 열변형/기계적 처짐이 튜브를 휘게 할 가능성 있어 특수 부착 방식 채택.

The CFRP tube is stiffer than the spacecraft centre panel; even so, panel deformation could flex it, hence a special compliant attachment.

**4.1.2 격자 어셈블리 / Grid assemblies** (Table II 요약):

| 파라미터 / Parameter | Fanbeam elements | Fourier elements |
|---|---|---|
| 요소 수 / Number of elements | 16 | 48 |
| Mosaic 구조 / Mosaic structure | 4 el. × 4 | 8 el. × 6 |
| Position angles | 0, 45, 90, 135° | 0, 30, 60, 90, 120, 150° |
| 위상 수 / Number of phases | 4 (90° step) | 2 (cos & sin) |
| Wave numbers $k$ | 1, 2 | 3, 4, 5, 6, 7, 8 |
| 피치 / Pitch (arc sec) | 126 | 42.0, 31.5, 25.2, 21.0, 18.0, 15.8 |
| 슬릿 너비 / Slit width (μm) | 210 | 140, 105, 84, 70, 60, 60 |
| 와이어 너비 / Wire width (μm) | 630 | 140, 105, 84, 70, 60, 45 |
| 재료 / Material | 0.5 mm 텅스텐 / 0.5 mm tungsten | 50 μm 텅스텐 호일 × 10 / 50 μm × 10 |
| 가공 / Process | electric discharge | photo-etching |

Layout 규칙:
1. 같은 position angle 끼리는 같은 unit에 → 위상 오차 최소화 / same angle in same unit
2. cos/sin 짝은 이웃하게 → 정확한 90° 위상 차 유지 / cos–sin pairs adjacent
3. 높은 k는 중심부에, 슬릿 방향은 가능한 한 tangential → tube twist의 영향 최소화 / high-k near centre, tangential slits minimise twist sensitivity

Knockpin holes로 ~5 μm 정밀도 조립; 격자판에 추가 정렬 패턴 6개 (가시광 통과)로 front/rear 정렬 확인 → coalignment 정밀도 ~1″ 가능.

Knockpin holes ensure ~5 μm assembly accuracy; 6 visible-light alignment patterns enable ~1″ front–rear coalignment verification.

#### 4.2 Detector Assembly (HXT-S) / 검출기 어셈블리

**4.2.1 검출기 모듈 / Detector module** (Fig. 6):
- NaI(Tl) 신틸레이터: 25 mm 정사각, 5 mm 두께 (격자 구경 23 mm 보다 약간 큼). 5 mm 두께는 고에너지 검출 효율을 결정.
- Al 케이스 0.8 mm로 NaI를 둘러쌈; 전면 Al은 X-ray 필터 역할로 flare soft X-ray의 pile-up 방지.
- 광전증배관 (PMT): Hamamatsu HPK 2497, 진동 방지형 (anti-vibration), μ-metal로 자기 차폐.
- Bleeder string + pre-amp 일체형 하우징.
- 8 모듈 → 1 detector unit (마그네슘 프레임), 8 unit → HXT-S 전체.
- 검출기 가속 정렬 정확도 < 1″, 두께 5 mm 결정.

The NaI(Tl) crystal is 25 mm square × 5 mm thick (slightly larger than the 23-mm grid aperture); 5 mm thickness sets high-energy detection efficiency. A 0.8-mm Al case acts as an X-ray filter to prevent pulse pile-up from flare soft X-rays. The PMT is HPK 2497 (anti-vibration) magnetically shielded by μ-metal. Eight modules form a detector unit (Mg frame); eight units form HXT-S.

**Eq. 5 (Energy resolution)**:
$$\Delta E/E \sim 1.3\,E^{-1/2}\quad (E\text{ in keV})$$

50 keV에서 $\Delta E/E \approx 18.4\%$ (FWHM). 이는 NaI(Tl)의 광자 통계 한계로 표준적이다. 50 keV → ΔE ~9.2 keV.
At 50 keV, ΔE/E ≈ 18.4% (FWHM), i.e., ΔE ~9.2 keV — typical photon-statistics-limited resolution.

**캘리브레이션 광원**: ²⁴¹Am (59.5 keV emission line, 4 mm² ≈ 0.6% aperture)을 각 모듈 전면 중앙에 부착. flare 관측 중에는 수 cps 수준이라 방해 없음.

A ²⁴¹Am calibration source (59.5 keV emission line, 4 mm² ≈ 0.6% of aperture, a few cps) is mounted on each module — negligible during flare observation.

**Pulse pile-up**: 전하 증폭기 시간 상수 ~10 μs. 수 × 10⁴ cps까지 처리 가능 — 가장 큰 플레어 (~10⁴ cps)에서도 안전.
Time constant of charge amplifier ~10 μs allows up to a few × 10⁴ cps without pile-up bias — safe even for the largest expected flares (~10⁴ cps).

**Spectral response (Fig. 7)**:
- 5 mm NaI(Tl): 저에너지에서 ~100% 효율 → 100 keV 부근에서 떨어지기 시작.
- Al 0.8 mm + CFRP 4.2 mm 흡수: 15 keV 이하에서 급격히 차단 → 그래서 lower threshold가 15(19) keV.
- 0.5 mm 텅스텐 격자 자체의 흡수 효율 (stopping power): 100 keV에서도 ~10%로 오류 항.

The 5 mm NaI(Tl) has ~100% efficiency at low energies, falling at ~100 keV. Al 0.8 mm + CFRP 4.2 mm absorption defines the lower threshold at 15(19) keV. The 0.5 mm tungsten grids themselves have ~10% absorption at 100 keV, an important correction term.

**4.2.2 고압 전원 / High-voltage supplies**:
8개의 DC-DC 컨버터, 각각 8개 모듈 담당. 800–1050 V 사이 8 단계 (35 V step), 코맨드로 선택 → 거친 게인 조정 ~20%. radiation belt 통과 시 PMT 보호로 OFF, 위성 night에는 절전 OFF.

Eight DC-DC converters, each driving 8 modules. Eight 35-V steps from 800–1050 V give ~20% coarse gain trim. Powered off during radiation-belt passage (to protect PMTs) and satellite night (power saving).

#### 4.3 Hard X-ray signal processing (HXT-E) / 신호 처리

블록 다이어그램 (Fig. 8):
1. 각 SC의 아날로그 신호 → 64-step gain 조정 (~1% step) → fine gain.
2. peak detect → 6-bit flash ADC (RCA CA3306D), digital discriminator (15 또는 19 keV 선택).
3. **Eq.**: energy (keV) = 1.35 × value + 13.6 (6-bit value 0..63 → 13.6 ~99.7 keV)
4. observation 모드: 256 (=4×64) counter, 12-bit (max 4095)으로 0.5 s마다 DP에 송신.

Block diagram (Fig. 8): the analogue signal is fine-gain adjusted (1% step in 64 steps), peak detected, A/D converted by a 6-bit flash ADC (CA3306D), then digitally discriminated (15 or 19 keV). The 6-bit value $v$ relates to energy: $E (\text{keV}) = 1.35 v + 13.6$. In observation mode, 256 (= 4 × 64) 12-bit counters dump every 0.5 s.

**4.3.1 4 PC 채널 / Four PC channels (Table III)**:

| Channel | Digital value | Energy range (nominal) |
|---|---|---|
| L | 1(4)–7 | 15.0(19.0)–24.4 keV |
| M1 | 8–15 | 24.4–35.2 keV |
| M2 | 16–31 | 35.2–56.8 keV |
| H | 32–63 | 56.8–100.0 keV |

각 SC당 4 채널 → 64 SC → 256 동시 영상 합성 시계열.

Each SC has 4 channels; 256 total channels per 0.5 s frame.

**4.3.2 캘리브레이션 모드 / Calibration mode**:
Pulse-height mode — 각 디지털 값 + 모듈 번호를 64 채널 메모리에 binning. 8초마다 출력. 위성 night 1000 s 적분으로 ²⁴¹Am 59.5 keV peak 위치를 수 % 정확도로 측정 → 64 모듈 게인을 ~1% 동일성으로 맞춤. 모듈/회로의 안정성 덕분에 1개월에 1회 미만 재조정 필요.

In pulse-height mode the digital value plus module number is binned into 64 channels per module, dumped every 8 s. With 1000-s integration during satellite night, the ²⁴¹Am 59.5 keV line can be located to a few %, allowing 64-module gain matching to ~1%. Stability allows recalibration <1×/month.

#### 4.4 Aspect System (HXA) / 자세 시스템

X-ray 광축의 위치 정확도 향상을 위한 가시광 시스템. 이유: 위성 attitude 시스템 + sun/star sensor 조합도 1″ 정확도와 충분한 시간 분해능을 동시에 달성하지 못함.
A visible-light system improves X-ray pointing knowledge: the spacecraft attitude system alone cannot deliver 1″ accuracy at the required cadence.

구성: HXT-C 중심축에 두 개의 동일한 시스템 (achromat doublet 10 mm, front grid에 필터, rear grid에 fiducial mark, HXT-S 위에 1D CCD). 두 CCD는 직교. lens 중심 + fiducial mark = X-ray "axis"를 정의 (metrology로 결정).

Two identical optical systems at HXT-C's central axis — achromatic doublet (10 mm), filter on front grid, fiducial marks on rear grid, 1D CCD on top of HXT-S. The two CCDs are mutually orthogonal. Lens centres + fiducial marks define the X-ray "axis" by metrology.

HXT-E에서 CCD 출력 처리 두 가지: (i) discrimination level 통과한 픽셀 주소만 매초 송신 (low data rate), (ii) 64 s마다 전체 brightness 분포 (high-data rate).
HXT-E processes CCD output two ways: (i) addresses of pixels above threshold sent every 1 s (low data rate); (ii) full brightness distribution every 64 s (high data rate).

#### 4.5 Onboard data handling (DP) / 온보드 데이터 처리

ISAS-DP는 HXT 데이터를 다른 데이터와 합쳐 telemetry frame 생성 (관측 모드: quiet/flare/night/BCS-out; bit rate: 32, 4, 1 kbps).

**4.5.1 4초 사전 저장 / Pre-storage**: quiet mode에서는 channel L 외 telemetry slot 없음. flare flag가 자동 전환되기 전 몇 초가 손실되므로 **4초 buffer**로 보관 → 플레어 시작 부분 손실 방지.

In quiet mode only channel L has a telemetry allocation; a few seconds elapse before the auto flare-flag triggers. A 4 s pre-buffer prevents loss of early flare data.

**4.5.2 시간 분해능 / Time resolution**: flare mode high-bit-rate에서 4 채널 × 0.5 s; medium에서는 4 s 적분. channel L은 별도로 high-bit-rate 2 s 또는 medium 16 s — 위성 night가 아니면 항상 backup으로 송신. 이는 pre-/post-flare 영상에도 활용.

In flare mode high-bit-rate: 4-channel 0.5 s. Medium: 4-s integration. Channel L is also sent separately at 2 s (high) or 16 s (medium) as a backup, useful for pre-/post-flare imaging.

**4.5.3 데이터 압축 / Data compression**: 12-bit count → 8-bit으로 다음과 같이 매핑:
$$m = n \quad (n=0..15)$$
$$m = \text{int}(4\sqrt{n}) \quad (n=16..4080)$$
$$m = 255 \quad (n=4081..4095)$$
이 매핑 → digital error $\le \sqrt{n}/2$, 즉 Poisson noise의 절반. 따라서 8-bit 압축본의 표준편차는 원래 $n$ 값의 표준편차와 거의 동일.

The 12-bit count $n$ is reduced to 8-bit $m$ by piecewise rule above. Digital error stays ≤ Poisson noise/2: $\Delta n' = [(m+1)^2 - m^2]/16 \sim m/8 \sim \sqrt{n}/2$. Hence 8-bit compressed values have nearly the same standard deviation as raw $n$.

### Part V: Final Remarks (Section 5) / 결론

HXT가 HINOTORI imager 보다 우월한 이유 4가지:
1. 푸리에 합성 → 높은 감도와 넓은 시야 동시 확보 / sensitivity + wide FOV
2. 100 keV까지 영상 가능 (텅스텐 격자 정밀 가공 + photo-etching) / imaging up to 100 keV
3. 미터링 튜브 길이 1.4 m 증가 → 더 작은 피치에서 동일 각도 / longer baseline → finer angular sampling
4. 64 SC 독립 운용 → 다중 푸리에 성분 동시 측정 / parallel multi-component measurement

영상 품질을 결정하는 4가지 캘리브레이션 항목:
1. **격자의 정밀성** — slit pitch 평균 1 μm 이내, position 10 μm 이내 (광학 현미경 검증)
2. **64 SC 정렬** — coalignment 1″ 이내 (광학 + X-ray 방법, 격자 분리 거리 줄여 측정)
3. **변조 패턴 평가** — 40 keV 이하에서 X-ray beam 직접 측정. 더 높은 에너지에서는 beam이 없어 추정만 가능 (이는 운용 후 한계로 작용)
4. **각 모듈의 PH gain** ~1% 정확도 측정 / gain 1%

또한: HXA 광축과 X-ray 광축의 정렬은 prism effect 등을 수 arcsec 정확도로 측정. 발사 전까지 캘리브레이션 지속.

Calibration was checked by (1) microscope verification (slit pitch within 1 μm of design, slit position within 10 μm); (2) optical + X-ray coalignment of all 64 SCs at reduced grid separation; (3) X-ray beam measurement of modulation patterns below 40 keV (no beams above this energy — an operational limitation); (4) PMT gain matching to ~1%; (5) HXA optical alignment measurement to a few arcsec.

### Acknowledgements & Tribute / 감사와 추모

논문은 SOLAR-A 프로젝트의 토대를 마련한 Minoru Oda, Yasuo Tanaka, 故 Katsuo Tanaka 교수에게 감사하고, **HXT의 수석 연구자였던 Keizo Kai 박사가 1991년 3월 11일 (논문 제출 다음 날) 별세** 했음을 추모하며, 남은 저자들이 그의 뜻을 이어 프로젝트를 성공시킬 의지를 표한다. 이는 이 논문에 깊은 인간적 차원을 더한다.

The paper acknowledges Minoru Oda, Yasuo Tanaka, and the late Katsuo Tanaka, and pays tribute to **Dr. Keizo Kai, Principal Investigator of HXT, who passed away on 11 March 1991 — one day after the paper was submitted to Solar Physics**. The remaining authors expressed their resolve to complete the project in his memory, adding a poignant human dimension to this engineering work.

---

## 3. Key Takeaways / 핵심 시사점

1. **Each subcollimator measures one Fourier component, not one pixel** — HXT가 라디오 간섭계의 aperture-synthesis 사상을 X-ray로 옮긴 첫 번째 사례임. 64개 SC = 64개 (k,θ) 점이며, 영상은 inverse FT가 아니라 MEM/CLEAN으로 합성된다. 이는 PSD-기반 직접 영상 (HXIS)와 근본적으로 다른 방식이다.

   HXT brought radio aperture-synthesis into X-ray. Each of the 64 SCs gives one (k,θ) point; the image is reconstructed by MEM/CLEAN, not direct inverse FT — fundamentally different from PSD imaging like HXIS.

2. **64 = 16 fanbeam + 48 (=24 cos/sin) Fourier elements is the result of optimisation** — 단순한 Fourier 짝만 64개로 구성하면 확장 광원에서 영상이 나빠지므로, 낮은 k에서는 fanbeam (1D 위치)으로 대체. 또한 (k,θ)를 직사각 격자 대신 polar로 두어 grating ambiguity 제거.

   The 64 = 16 fanbeam + 48 Fourier element design is the result of trade-off: pure cos/sin pairs degrade extended-source quality at low k, so fanbeam replaces them; polar (k,θ) layout (vs. rectangular) avoids grating ambiguity.

3. **5″ angular resolution is set by $k_\text{max} = 8$, not by physics** — 텅스텐 격자 fabrication 한계가 $k_\text{max}$를 결정. $L_0 = 2'06''$, $k_\text{max} = 8 \Rightarrow$ 가장 작은 피치 ~15.8″ → 분해능 ~5″. 더 정밀한 그리드가 가능했다면 더 좋은 분해능도 가능했을 것이다.

   The 5″ angular resolution is set by $k_\text{max}=8$, which is in turn limited by tungsten-grid fabrication, not by physics. With $L_0 = 2'06''$, the finest pitch is ~15.8″, giving ~5″ resolution.

4. **Calibration tolerances are tight: 5% effective area + 1″ peak position rms** — 영상 품질이 안정하려면 64 SC effective area의 rms가 5% 이하, peak position의 rms가 1″ 이하로 측정되어야 함. effective area는 detector gain에 강하게 의존 → flare power-law spectrum (γ ≈ 4)에서 5% 면적 = 1% gain 정확도 요구.

   Image stability requires <5% rms in the 64 effective areas and <1″ rms in peak positions. Effective area depends strongly on detector gain — for a γ ≈ 4 flare spectrum, 5% area accuracy corresponds to 1% gain accuracy.

5. **NaI(Tl) energy resolution $\Delta E/E \sim 1.3 E^{-1/2}$ is the bottleneck** — 50 keV에서 18% (FWHM ~9 keV). 이는 4채널 binning 폭과 잘 맞아 넓은 채널 (35–57, 57–100 keV)을 정당화한다. 게다가 ²⁴¹Am 59.5 keV 라인을 캘리브레이션 표준으로 채택한 것은 NaI 응답이 100 keV 부근에서 가장 잘 알려져 있기 때문.

   NaI(Tl)'s $\Delta E/E \sim 1.3 E^{-1/2}$ (~18% at 50 keV, FWHM ~9 keV) is the spectroscopic bottleneck; it justifies the broad H-channel (57–100 keV). Choosing ²⁴¹Am 59.5 keV as the calibration line is natural because NaI response is best known there.

6. **Mechanical engineering is as critical as optical** — CFRP 미터링 튜브 (α < 10⁻⁶/K, twist 0.025 arcmin/kg-m)와 ~5 μm 정밀 knockpin 조립이 1″ alignment를 가능케 함. 격자 위치 오차 10 μm가 1.4 m baseline에서 ~1.5″ 각오차에 해당 — 한계선에 매우 근접.

   Mechanical engineering matches the optical: a CFRP tube (α < 10⁻⁶/K, twist 0.025 arcmin/kg·m) plus 5-μm knockpin assembly enables ~1″ coalignment. A 10-μm grid offset at 1.4 m baseline corresponds to ~1.5″ error, very close to the tolerance.

7. **Data compression by $m = \text{int}(4\sqrt{n})$ preserves Poisson statistics** — 12-bit→8-bit 변환에서 압축 오차가 √n/2 < 광자 통계 noise. 이런 "noise-aware" 압축은 photon-counting 모드 telemetry-제한 위성의 핵심 기법.

   The $m = \text{int}(4\sqrt{n})$ data-compression preserves Poisson statistics: the digitisation noise is √n/2 < photon-counting noise. Such "noise-aware" compression is essential for telemetry-limited photon-counting satellites.

8. **HXT enabled the Masuda flare discovery (1994)** — HXT가 SXT와 동시에 자기 루프 위 (above-the-loop-top)에서 비열적 hard X-ray source를 검출함으로써, 기존의 "thick-target footpoint only" 모델로는 설명할 수 없는 새로운 가속 영역 (loop-top 근처 reconnection outflow site로 해석)을 발견. 이는 HXT의 가장 중요한 과학적 결과로, RHESSI/STIX의 후속 영상 missions를 추진한 동력이 되었다.

   HXT directly enabled the 1994 Masuda flare discovery — an above-the-loop-top hard X-ray source coincident with SXT loop-top — which could not be explained by the classical thick-target footpoint model and pointed to acceleration near the reconnection outflow region. This drove the case for follow-on imaging missions (RHESSI, STIX).

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Modulation collimator transmission / 변조 시준기 투과 함수

A bigrid modulation collimator with grid separation $L$ and slit pitch $p$ produces a triangular transmission as a function of incident angle. After removing the DC component and Fourier-decomposing:
$$T(\rho) \approx \frac{1}{2} + \sum_{k=1}^{\infty} a_k \cos(2\pi k \rho / p)$$
where $\rho = X\cos\theta + Y\sin\theta$ is the projected angle along the grid normal direction. For HXT, only the $k=1$ component is used per SC; the $a_k$ for $k \ge 2$ Fourier-element grids come from designing the grid geometry (slit width vs. pitch) so that the desired wave number dominates.

In the paper's notation:
$$F_c(k\rho) = \text{cosine-like transmission of one SC at wave number } k$$
$$F_s(k\rho) = F_c(k\rho - \pi/2) = \text{sine-like partner shifted by quarter pitch}$$

### 4.2 Photon count = Fourier component / 광자 카운트와 푸리에 성분의 관계

For a SC pair with effective area $A$ over integration time $\tau$ (here 0.5 s):
$$b_c(k,\theta) = A \tau \int B(X,Y) F_c(k\rho)\,dX\,dY$$
$$b_s(k,\theta) = A \tau \int B(X,Y) F_s(k\rho)\,dX\,dY$$

Defining the complex Fourier coefficient
$$\tilde{B}(k,\theta) \equiv b_c + i b_s = A\tau \int B(X,Y) e^{-i 2\pi k \rho/L_0} dX\,dY \quad (\text{ignoring DC and harmonics})$$

This is the X-ray analogue of the radio interferometric visibility:
$$V(u,v) = \int B(X,Y) e^{-i 2\pi(uX + vY)} dX\,dY$$
with $u = (k/L_0)\cos\theta$, $v = (k/L_0)\sin\theta$.

### 4.3 (k, θ) Sampling / 샘플링 구조

64 SC = 16 fanbeam + 48 Fourier elements:
- **Fanbeam** (low k = 1, 2): 4 angles × 4 phases = 16, single phase per (k,θ), no cos/sin pair
- **Fourier elements** (high k = 3..8): 6 angles × 8 phases = 48, but arranged as 24 cos/sin pairs (k, θ) → 24 complex measurements

Fundamental period $L_0 = 2'06'' = 126''$.
At wave number $k$, pitch $p_k$:
$$p_k = L_0/k$$
- $k=1$: $p_1 = 126''$ (fanbeam)
- $k=2$: $p_2 = 63''$ (fanbeam)
- $k=3$: $p_3 = 42.0''$ (Fourier)
- $k=4$: $p_4 = 31.5''$
- $k=5$: $p_5 = 25.2''$
- $k=6$: $p_6 = 21.0''$
- $k=7$: $p_7 = 18.0''$
- $k=8$: $p_8 = 15.8''$

For the highest k=8 grid in a 1400 mm metering tube, the wire-to-slit physical pitch is: $p_8/(206265'') \times 1400\,\text{mm} \approx 107\,\mu$m.

### 4.4 Energy resolution / 에너지 분해능

NaI(Tl) photon-statistics-limited resolution:
$$\Delta E / E \approx 1.3 E^{-1/2} \quad (E\text{ in keV, FWHM})$$

Tabulated values:
| E (keV) | ΔE/E | ΔE (FWHM) |
|---|---|---|
| 20 | 29% | 5.8 |
| 50 | 18% | 9.2 |
| 100 | 13% | 13.0 |

This sets the four PC channel widths (Table III) and explains the choice of broad H-channel.

### 4.5 Tube twist stiffness / 튜브 비틀림

$$\theta(\text{arcmin}) = 0.025\,T\,(\text{kg m})$$

For example, $T = 1$ kg·m → $\theta = 0.025'$ = 1.5″, which is right at the alignment tolerance. The metering tube is designed so that operational torques are << 1 kg·m.

### 4.6 Pulse-height-to-energy conversion / PH 변환

$$E (\text{keV}) = 1.35 v + 13.6 \quad (v \in \{0..63\})$$

so $v=0 \Rightarrow E=13.6$ keV (pre-threshold), $v=63 \Rightarrow E=98.6$ keV. The four channels (Table III):
- L: $v \in [4,7]$ → 19.0–24.4 keV
- M1: $v \in [8,15]$ → 24.4–35.2 keV
- M2: $v \in [16,31]$ → 35.2–56.8 keV
- H: $v \in [32,63]$ → 56.8–100.0 keV

### 4.7 Data compression rule / 데이터 압축

$$m = \begin{cases} n & 0 \le n \le 15 \\ \text{int}(4\sqrt{n}) & 16 \le n \le 4080 \\ 255 & 4081 \le n \le 4095\end{cases}$$

Inverse:
$$n' = (m/4)^2 \quad (m \ge 16)$$

The induced rounding standard deviation:
$$\sigma_{\text{round}} \approx \frac{\partial n}{\partial m} \cdot \frac{1}{\sqrt{12}} = \frac{m}{8\sqrt{12}} \approx \frac{\sqrt{n}}{4\sqrt{3}}$$

vs. Poisson noise $\sigma_P = \sqrt{n}$; so $\sigma_\text{round} / \sigma_P \approx 1/7$. The paper's stated $\sqrt{n}/2$ is a conservative envelope.

### 4.8 Worked numerical example / 수치 예시

**Compact double source flare**: 두 점원 분리 11″, 각 약 $10^4$ counts/s in M1 channel (24–35 keV), 0.5 s → 5000 counts each.

For wave number $k$ and a pair separated by $\Delta = 11''$:
- $k=4$ ($p=31.5''$): $\Delta/p = 0.349$, modulation amplitude is large → strong (cos, sin) signature
- $k=8$ ($p=15.8''$): $\Delta/p = 0.696$, near maximum modulation → strongest source localisation
- $k=2$ ($p=63''$): $\Delta/p = 0.175$, modulation small but non-zero → contributes mainly to total flux

Expected complex Fourier component magnitude at (k, θ) where θ aligns with source separation axis:
$$|\tilde{B}(k,\theta)| \approx 2 F_0 |\cos(\pi k \Delta/L_0)|$$
where $F_0 = 5000$ counts per source. At $k=8$, $\theta = 0$: $|\tilde{B}| \approx 2 \times 5000 \times |\cos(0.696\pi)| \approx 5860$ counts. Combined with Poisson noise $\sqrt{2 \times 5000} \approx 100$ counts, SNR ≈ 60. This is comfortably above the 5″ resolution criterion.

For an extended source with FWHM 20″, $|\tilde{B}|$ falls off as the source's transform $\sim \exp(-(k \cdot 20''/L_0)^2/4)$; at $k=8$, the modulation is suppressed by $\exp(-1.94) = 0.14$ — explaining why fanbeam elements at low k are needed for extended sources.

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1972 ── Frieden                   MEM 통계적 영상 재구성 / MEM image reconstruction
1974 ── Högbom                    CLEAN deconvolution algorithm / CLEAN 디컨볼루션
1978 ── Makishima et al.          MPMC (Multi-Pitch Modulation Collimator) 제안 / MPMC concept
1980 ── van Beek et al. (HXIS)    SMM의 첫 hard X-ray 영상 / first hard X-ray imaging on SMM
1981 ── HINOTORI                  회전 bigrid 변조 시준기 / rotating bigrid MC, 1D→2D
1983 ── Kosugi & Tsuneta          HINOTORI imaging analysis (Solar Phys. 86)
1984 ── Tsuneta                   HINOTORI image-synthesis methodology refined
1988 ── Prince et al.             PSD-based Fourier-transform telescope concept / PSD 기반 푸리에 망원경
1988 ── Dennis                    SMM era 한계 정리 (Solar Phys. 118)
1991 ──────────── *Kosugi et al. — HXT design paper (this paper)*
1991 (Aug) ── YOHKOH 발사 / YOHKOH launch
1992 ── Tsuneta et al.            SXT first images (companion to HXT)
1994 ── Masuda et al.             above-the-loop-top hard X-ray source (HXT result)
2002 ── Lin et al. (RHESSI)       9 RMC, 3 keV–17 MeV, Fourier imaging continued
2008 ── Hurford et al.            RHESSI image-reconstruction techniques summarised
2020 ── Krucker et al. (STIX)     Solar Orbiter STIX, 32 grid pairs, 4–150 keV
2024 ── ASO-S / HXI               Chinese hard X-ray imager, modulation collimator 계승
```

이 타임라인이 보여주는 것: HXT는 1970년대 라디오 영상 (CLEAN, MEM)의 수학적 도구, 1980년대 변조 시준기 공학 (HXIS, HINOTORI), 그리고 푸리에 영상 망원경 개념 (Makishima 1978; Prince 1988)을 종합한 결과물이며, 이후 30년 hard X-ray 영상 천문학의 출발점이 되었다.

The timeline shows that HXT synthesised: 1970s mathematical tools from radio imaging (CLEAN, MEM), 1980s modulation-collimator engineering (HXIS, HINOTORI), and the Fourier-imaging telescope concept (Makishima 1978; Prince 1988). It became the launching point for 30 years of hard X-ray imaging astronomy.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Makishima (1978) — MPMC concept | HXT의 multi-pitch modulation collimator의 직접적 선례 / direct ancestor of HXT's MPMC design | High — HXT가 이 개념의 첫 우주 구현 / first space implementation |
| Hogbom (1974) — CLEAN | HXT가 modified CLEAN으로 영상 재구성 / HXT uses modified CLEAN | High — radio interferometry → X-ray로 이전 / radio→X-ray transfer |
| Frieden (1972) / Gull & Daniell (1978) — MEM | HXT의 두 영상 합성 알고리즘 중 하나 / one of HXT's two reconstruction algorithms | High — 부족한 (k,θ) 샘플링에서 양수 해 / positivity prior with sparse sampling |
| Prince et al. (1988) — Fourier-transform telescope | PSD 기반 동일 개념의 다른 구현 / alternative implementation using position-sensitive detectors | High — HXT는 non-PSD detector 채택 / HXT chose non-PSD for simplicity |
| Tsuneta et al. (1991) — SXT (companion paper) | YOHKOH 동승 soft X-ray imager / co-mission soft X-ray imager | High — HXT/SXT 동시 관측이 핵심 과학 / joint observation key |
| Dennis (1988) — SMM era review | HXT 설계의 동기 (SMM/HINOTORI 한계) / motivation summary | Medium — context-setting / 배경 |
| Ogawara et al. (1991) — SOLAR-A spacecraft | HXT가 부착되는 위성 시스템 / host spacecraft | Medium — DP, telemetry interfaces / DP·텔레메트리 인터페이스 |
| Yoshimori et al. (1991) — GRS | HXT와 동시 관측되는 감마선 분광계 / co-observation gamma-ray spectrometer | Medium — 전자/이온 가속 비교 / electron–ion comparison |
| Masuda et al. (1994) — above-the-loop-top source | HXT의 가장 유명한 과학 결과 / HXT's most famous discovery | High (downstream) — HXT가 가능케 한 발견 / discovery enabled by HXT |
| Lin et al. (2002) — RHESSI mission | HXT 푸리에 영상의 직접 계승자 / direct descendant of HXT's Fourier imaging | High (downstream) — RMC + Fourier synthesis 사용 / RMC + Fourier synthesis |

---

## 7. References / 참고문헌

- Kosugi, T., Makishima, K., Murakami, T., Sakao, T., Dotani, T., Inda, M., Kai, K., Masuda, S., Nakajima, H., Ogawara, Y., Sawa, M., and Shibasaki, K., "The Hard X-ray Telescope (HXT) for the SOLAR-A Mission", *Solar Physics* **136**, 17–36, 1991. DOI: 10.1007/BF00151693
- Dennis, B. R., "Solar Hard X-Ray Bursts", *Solar Physics* **118**, 49, 1988.
- Frieden, B. R., "Restoring with Maximum Likelihood and Maximum Entropy", *J. Opt. Soc. Am.* **62**, 511, 1972.
- Gull, S. F. and Daniell, G. J., "Image reconstruction from incomplete and noisy data", *Nature* **272**, 686, 1978.
- Högbom, J. A., "Aperture synthesis with a non-regular distribution of interferometer baselines", *Astron. Astrophys. Suppl.* **15**, 417, 1974.
- Kosugi, T. and Tsuneta, S., "Imaging Analysis of HINOTORI", *Solar Physics* **86**, 333, 1983.
- Makishima, K., "Hard X-Ray Imaging Observations with HINOTORI", in Y. Tanaka et al. (eds.), *Proc. HINOTORI Symp. on Solar Flares*, ISAS, p. 120, 1982.
- Makishima, K., Miyamoto, S., Murakami, T., Nishimura, J., Oda, M., Ogawara, Y., and Tawara, Y., in K. van der Hucht and G. S. Vaiana (eds.), *New Instrumentation for Space Astronomy*, Pergamon Press, p. 277, 1978. [MPMC concept]
- Ogawara, Y., Takano, T., Kato, T., Kosugi, T., Tsuneta, S., Watanabe, T., Kondo, I., and Uchida, Y., "The SOLAR-A Mission", *Solar Physics* **136**, 1, 1991.
- Prince, T. A., Hurford, G. J., Hudson, H. S., and Crannell, C. J., "Gamma-Ray and Hard X-Ray Imaging of Solar Flares", *Solar Physics* **118**, 269, 1988.
- Tsuneta, S., Acton, L., Bruner, M., Lemen, J., Brown, W., Caravalho, R., Catura, R., Freeland, S., Jurchevich, B., Morrison, M., Ogawara, Y., Hirayama, T., and Owens, J., "The Soft X-ray Telescope for the SOLAR-A Mission", *Solar Physics* **136**, 37, 1991.
- Tsuneta, S., "HINOTORI Hard X-Ray Imaging Observations", *Ann. Tokyo Astron. Obs., 2nd Series* **20**, 1, 1984.
- Van Beek, H. F., Hoyng, P., Lafleur, B., and Simnett, G. M., "The Hard X-Ray Imaging Spectrometer (HXIS)", *Solar Physics* **65**, 39, 1980.
- Willingale, R., "Use of the Maximum Entropy Method in X-ray Astronomy", *Mon. Not. R. Astron. Soc.* **194**, 359, 1981.
- Yoshimori, M. et al., "The Gamma-Ray Spectrometer (GRS) on SOLAR-A", *Solar Physics* **136**, 61, 1991.
- Masuda, S., Kosugi, T., Hara, H., Tsuneta, S., and Ogawara, Y., "A loop-top hard X-ray source in a compact solar flare", *Nature* **371**, 495, 1994. [downstream HXT result]
- Lin, R. P. et al., "The Reuven Ramaty High-Energy Solar Spectroscopic Imager (RHESSI)", *Solar Physics* **210**, 3, 2002. [HXT successor]
- Krucker, S. et al., "The Spectrometer/Telescope for Imaging X-rays (STIX)", *Astron. Astrophys.* **642**, A15, 2020. [HXT/RHESSI successor]
