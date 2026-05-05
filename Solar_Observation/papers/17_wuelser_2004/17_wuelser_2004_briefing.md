---
title: "Pre-Reading Briefing: EUVI: The STEREO-SECCHI Extreme Ultraviolet Imager"
paper_id: "17_wuelser_2004"
topic: Solar_Observation
date: 2026-04-17
type: briefing
---

# EUVI: The STEREO-SECCHI Extreme Ultraviolet Imager: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Wuelser, J.-P., Lemen, J.R. et al., "EUVI: The STEREO-SECCHI Extreme Ultraviolet Imager," *Proc. SPIE 5171*, Telescopes and Instrumentation for Solar Astrophysics, pp. 111–122, 2004.
**Author(s)**: Jean-Pierre Wuelser, James R. Lemen et al.
**Year**: 2004
**DOI**: 10.1117/12.506877

---

## 1. 핵심 기여 / Core Contribution

EUVI(Extreme Ultraviolet Imager)는 STEREO(Solar TErrestrial RElations Observatory) 미션의 SECCHI(Sun Earth Connection Coronal and Heliospheric Investigation) 장비 패키지에 탑재된 EUV 망원경입니다. STEREO는 두 대의 동일한 우주선으로 구성되며, 각각 태양 주위 궤도에서 지구보다 앞(Ahead)과 뒤(Behind)에 위치합니다. EUVI는 EIT(Paper #9)의 설계를 계승하면서 크게 개선하여, 4개 EUV 파장 밴드(171, 195, 284, 304 Å)에서 full-disk 태양 영상을 촬영하되, **1.6 arcsec 공간 분해능**과 높은 촬영 주기를 달성했습니다. 두 시점에서 동시에 관측함으로써 **사상 최초의 태양 코로나 stereoscopic 관측**을 가능하게 했습니다.

EUVI (Extreme Ultraviolet Imager) is an EUV telescope aboard the SECCHI (Sun Earth Connection Coronal and Heliospheric Investigation) instrument suite of the STEREO (Solar TErrestrial RElations Observatory) mission. STEREO comprises two identical spacecraft in heliocentric orbit, one leading (Ahead) and one trailing (Behind) Earth. EUVI inherits and substantially improves upon the EIT (Paper #9) design, imaging the full solar disk in four EUV bandpasses (171, 195, 284, and 304 Å) with **1.6 arcsec spatial resolution** and high cadence. By observing simultaneously from two vantage points, it enabled **the first-ever stereoscopic observations of the solar corona**.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

2004년 이 논문이 발표될 당시, SOHO/EIT(1995, Paper #9)가 거의 10년간 운용되면서 EUV full-disk 영상의 가치가 입증되었습니다. EIT는 multilayer normal-incidence 광학을 이용한 4밴드 영상 촬영의 개념을 확립했지만, 공간 분해능(~5.2 arcsec/pixel)과 촬영 주기에 한계가 있었습니다. TRACE(1998)가 높은 분해능(0.5 arcsec)을 보여주었지만 시야가 작았고(8.5 × 8.5 arcmin), EUV 관측은 여전히 단일 시점(single viewpoint)에 머물러 있었습니다. 코로나 구조의 3차원 형태를 직접 관측할 수 없다는 점이 근본적인 한계였습니다.

At the time of this 2004 paper, SOHO/EIT (1995, Paper #9) had demonstrated the scientific value of full-disk EUV imaging over nearly a decade of operations. EIT established the concept of four-band imaging with multilayer normal-incidence optics, but had limitations in spatial resolution (~5.2 arcsec/pixel) and cadence. TRACE (1998) showed high-resolution (0.5 arcsec) capability but had a small field of view (8.5 × 8.5 arcmin), and all EUV observations remained from a single viewpoint. The inability to directly observe the 3D structure of coronal features was a fundamental limitation.

STEREO 미션은 이 문제를 해결하기 위해 기획되었습니다. 두 우주선을 태양 주위 서로 다른 위치에 배치하면, 동일한 코로나 구조를 두 각도에서 관측하여 stereoscopic 3D 재구성이 가능해집니다. EUVI는 이 미션의 핵심 장비로서, EIT의 성공적인 설계를 기반으로 하되 분해능과 감도를 크게 높여 stereoscopy에 적합한 영상을 제공하도록 설계되었습니다.

The STEREO mission was conceived to address this limitation. By placing two spacecraft at different locations around the Sun, the same coronal structures could be observed from two angles, enabling stereoscopic 3D reconstruction. EUVI was the core instrument for this mission, built upon EIT's proven design but with significantly improved resolution and sensitivity to provide images suitable for stereoscopy.

### 타임라인 / Timeline

```
1995 ─── SOHO/EIT 발사: EUV 4밴드 full-disk 영상의 시작
          SOHO/EIT launch: Beginning of 4-band full-disk EUV imaging
1998 ─── TRACE 발사: 고분해능 EUV 영상 (좁은 시야)
          TRACE launch: High-resolution EUV imaging (narrow FOV)
2004 ─── ★ 본 논문: EUVI 설계 및 성능 기술
          ★ This paper: EUVI design and performance description
2006 ─── STEREO 발사 (10월 25일): EUVI 운용 시작
          STEREO launch (Oct 25): EUVI operations begin
2010 ─── SDO/AIA 발사: 10채널, 12초 주기, 0.6 arcsec
          SDO/AIA launch: 10 channels, 12-sec cadence, 0.6 arcsec
2011 ─── STEREO-B와 지구 사이 각도 180° 도달
          STEREO-B reaches 180° separation from Earth
2014 ─── STEREO-B 통신 두절
          STEREO-B communication lost
```

---

## 3. 필요한 배경 지식 / Prerequisites

### EUV 다층 코팅 광학 / EUV Multilayer Optics (Paper #9에서 학습)

EUVI는 EIT와 동일하게 Mo/Si (molybdenum/silicon) multilayer coating을 사용한 normal-incidence 반사경으로 작동합니다. 각 파장 밴드는 다층 코팅의 주기(d-spacing)를 조절하여 선택합니다. Paper #9에서 이 원리를 학습하셨으므로 기본 개념은 숙지하고 계실 것입니다.

EUVI uses the same normal-incidence mirrors with Mo/Si multilayer coatings as EIT. Each wavelength band is selected by tuning the d-spacing of the multilayer coating. You learned this principle in Paper #9.

### Ritchey-Chrétien 망원경 설계 / Ritchey-Chrétien Telescope Design

EUVI는 Ritchey-Chrétien 광학 설계를 채택합니다. 이는 포물면이 아닌 **쌍곡면(hyperboloidal)** 주경과 부경으로 구성되어, 넓은 시야에서 coma와 spherical aberration을 동시에 보정하는 설계입니다. 허블 우주 망원경도 이 설계를 사용합니다.

EUVI adopts a Ritchey-Chrétien optical design. This uses **hyperboloidal** primary and secondary mirrors (not parabolic) to correct both coma and spherical aberration over a wide field of view. The Hubble Space Telescope also uses this design.

### Stereoscopic 재구성 / Stereoscopic Reconstruction

두 시점에서 촬영한 영상 쌍으로부터 3D 구조를 복원하는 기법입니다. **Epipolar geometry** (에피폴라 기하학)가 핵심 개념으로, 한 영상의 한 점에 대응하는 다른 영상의 점은 epipolar line 위에 존재합니다. 두 우주선의 위치와 자세를 알면 대응점을 찾아 삼각측량으로 3D 좌표를 결정할 수 있습니다.

Stereoscopic reconstruction recovers 3D structure from image pairs taken from two viewpoints. **Epipolar geometry** is the key concept: a point in one image corresponds to points along an epipolar line in the other image. Knowing the positions and orientations of both spacecraft, one can find corresponding points and determine 3D coordinates via triangulation.

### 온도 응답 함수 / Temperature Response Functions (Paper #12에서 학습)

각 EUV 채널은 특정 온도 범위의 플라즈마에 가장 민감합니다. 171 Å (~1 MK, Fe IX/X), 195 Å (~1.5 MK, Fe XII), 284 Å (~2 MK, Fe XV), 304 Å (~0.08 MK, He II). Paper #12 (AIA)에서 이 개념을 더 자세히 다루셨습니다.

Each EUV channel is most sensitive to plasma at specific temperatures: 171 Å (~1 MK, Fe IX/X), 195 Å (~1.5 MK, Fe XII), 284 Å (~2 MK, Fe XV), 304 Å (~0.08 MK, He II). You covered this in more detail in Paper #12 (AIA).

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| STEREO | Solar TErrestrial RElations Observatory. 두 대의 우주선으로 태양을 입체 관측하는 NASA 미션 / Two-spacecraft NASA mission for stereoscopic solar observation |
| SECCHI | Sun Earth Connection Coronal and Heliospheric Investigation. STEREO의 원격 관측 장비 패키지 / Remote-sensing instrument suite on STEREO |
| EUVI | Extreme Ultraviolet Imager. SECCHI의 EUV 망원경 / EUV telescope within SECCHI |
| Ritchey-Chrétien | 쌍곡면 주경/부경을 사용하여 coma-free 광학을 구현하는 두 거울 설계 / Two-mirror design using hyperboloidal mirrors for coma-free optics |
| Normal incidence | 거울 표면에 거의 수직으로 입사하는 방식; grazing incidence의 반대 / Light hitting mirror nearly perpendicular to surface; opposite of grazing incidence |
| Multilayer coating | Mo/Si 등의 교대 박막층으로 특정 EUV 파장을 선택적으로 반사 / Alternating thin-film layers (e.g., Mo/Si) that selectively reflect specific EUV wavelengths |
| Quadrant detector | 4개 사분면으로 나뉜 검출기; 각 사분면이 다른 파장 밴드에 대응 / Detector divided into four quadrants, each corresponding to a different wavelength band |
| Epipolar geometry | 두 시점 영상에서 대응점을 찾기 위한 기하학적 관계 / Geometric relationship for finding corresponding points between two-viewpoint images |
| Back-thinned CCD | 뒷면을 얇게 깎아 단파장(EUV) 감도를 높인 CCD / CCD with thinned backside for enhanced short-wavelength (EUV) sensitivity |
| Stray light | 의도하지 않은 경로로 검출기에 도달하는 빛; 특히 가시광 억제가 중요 / Unwanted light reaching detector; visible-light rejection is critical |
| FOV (Field of View) | 망원경이 한 번에 촬영할 수 있는 하늘의 영역 / Angular extent of sky captured in a single exposure |

---

## 5. 수식 미리보기 / Equations Preview

### 5.1 공간 분해능과 회절 한계 / Spatial Resolution and Diffraction Limit

$$\theta_{\text{diff}} = 1.22 \frac{\lambda}{D}$$

- $\theta_{\text{diff}}$: 회절 한계 각분해능 (radian) / diffraction-limited angular resolution
- $\lambda$: 관측 파장 (e.g., 171 Å = 17.1 nm) / observing wavelength
- $D$: 주경 유효 구경 / primary mirror effective aperture

EUVI의 경우 $D \approx 98$ mm이므로, 171 Å에서 회절 한계는 약 0.044 arcsec으로 광학 설계보다 훨씬 작습니다. 실제 분해능은 CCD 픽셀 크기와 미러 품질에 의해 결정됩니다.

For EUVI with $D \approx 98$ mm, the diffraction limit at 171 Å is ~0.044 arcsec, far below the optical design. Actual resolution is determined by CCD pixel size and mirror quality.

### 5.2 판 스케일 / Plate Scale

$$\text{Plate Scale} = \frac{206265}{f} \quad [\text{arcsec/mm}]$$

- $f$: 유효 초점거리 (mm) / effective focal length

EUVI의 초점거리 ~1750 mm, CCD 픽셀 크기 ~21 μm이면 pixel scale ≈ 1.6 arcsec/pixel입니다.

With EUVI's focal length ~1750 mm and CCD pixel size ~21 μm, pixel scale ≈ 1.6 arcsec/pixel.

### 5.3 Multilayer 반사 조건 / Multilayer Reflection Condition (Bragg's Law)

$$m\lambda = 2d\sin\theta$$

- $m$: 회절 차수 (보통 1) / diffraction order (usually 1)
- $d$: 다층 주기 / multilayer period (d-spacing)
- $\theta$: 입사각 (표면 법선으로부터) / angle of incidence

Normal incidence ($\theta \approx 90°$)에서 $\lambda \approx 2d$이므로, 원하는 파장의 절반에 해당하는 주기를 선택합니다.

At normal incidence ($\theta \approx 90°$), $\lambda \approx 2d$, so the period is chosen to be half the desired wavelength.

---

## 6. 읽기 가이드 / Reading Guide

### 읽기 순서 제안 / Suggested Reading Order

1. **Introduction / 서론** — STEREO 미션의 과학 목표와 EUVI의 역할을 파악하세요. EIT에서 어떤 개선이 이루어졌는지에 주목하세요.
   Understand the STEREO mission science goals and EUVI's role. Note what improvements were made over EIT.

2. **Optical Design / 광학 설계** — Ritchey-Chrétien 설계의 구체적인 사양(구경, 초점거리, FOV)을 파악하세요. EIT의 광학과 비교해 보세요.
   Note the specific specs (aperture, focal length, FOV) of the Ritchey-Chrétien design. Compare with EIT's optics.

3. **Multilayer Coatings / 다층 코팅** — 4개 파장 밴드의 코팅 사양과 반사율을 확인하세요. EIT와 동일한 파장을 선택한 이유를 생각해 보세요.
   Check coating specs and reflectivities for the four bands. Consider why the same wavelengths as EIT were chosen.

4. **Detector / 검출기** — CCD 사양(픽셀 크기, 배열, back-thinning)과 quadrant 구조를 이해하세요.
   Understand CCD specs (pixel size, array, back-thinning) and quadrant structure.

5. **Stray Light / 산란광** — 가시광 억제 방법과 성능을 확인하세요. EUV 관측에서 이것이 왜 중요한지 생각해 보세요.
   Check visible-light rejection methods and performance. Consider why this is critical for EUV observations.

6. **Performance / 성능** — 예상 감도, 촬영 주기, 영상 품질을 EIT 및 AIA와 비교하며 읽으세요.
   Compare expected sensitivity, cadence, and image quality with EIT and AIA.

### 핵심 질문 / Key Questions to Keep in Mind

- EUVI가 EIT 대비 어떤 광학적 개선을 이루었는가? / What optical improvements does EUVI have over EIT?
- Quadrant detector 방식의 장단점은 무엇인가? / What are the pros and cons of the quadrant detector approach?
- 두 우주선에서의 동시 관측이 가능하려면 어떤 기술적 과제가 있는가? / What technical challenges exist for simultaneous observation from two spacecraft?
- EUVI와 후속 AIA(Paper #12) 사이의 설계 철학 차이는 무엇인가? / How does the design philosophy differ between EUVI and the later AIA (Paper #12)?

---

## 7. 현대적 의의 / Modern Significance

EUVI는 여러 면에서 태양물리학에 지속적인 영향을 미치고 있습니다:

EUVI has had lasting impact on solar physics in several ways:

- **Stereoscopic 관측의 개척**: EUVI는 태양 코로나의 3D 구조를 직접 관측할 수 있게 한 최초의 장비입니다. 이를 통해 coronal loop의 실제 3차원 형태, CME의 전파 방향과 속도를 정량적으로 측정할 수 있게 되었습니다.
  **Pioneering stereoscopic observation**: EUVI was the first instrument to enable direct observation of the 3D structure of the solar corona, allowing quantitative measurement of coronal loop geometry and CME propagation direction and speed.

- **우주 기상 예보에의 기여**: 두 시점 관측은 지구를 향한 CME를 조기에 탐지하고 그 3D 궤적을 추적하는 데 결정적인 역할을 했습니다. 현대 우주 기상 예보 모델의 검증에 STEREO/EUVI 데이터가 광범위하게 사용됩니다.
  **Contribution to space weather forecasting**: Two-viewpoint observations were crucial for early detection of Earth-directed CMEs and tracking their 3D trajectories. STEREO/EUVI data is extensively used for validating modern space weather forecast models.

- **EIT → EUVI → AIA 계보**: EUVI는 EIT에서 AIA로 이어지는 EUV imager 발전 계보의 중요한 중간 단계입니다. Ritchey-Chrétien 설계의 채택, 개선된 CCD, 향상된 stray-light 억제 등 EUVI의 기술적 혁신이 AIA 설계에 반영되었습니다.
  **EIT → EUVI → AIA lineage**: EUVI is a crucial intermediate step in the EUV imager evolution from EIT to AIA. Technical innovations such as the Ritchey-Chrétien design, improved CCDs, and enhanced stray-light rejection in EUVI informed the AIA design.

- **현재까지 운용 중**: STEREO-A는 2006년 발사 이후 현재까지 운용 중이며, EUVI는 계속 데이터를 생산하고 있습니다. SDO/AIA와 결합하면 여전히 다중 시점 관측이 가능합니다.
  **Still operational**: STEREO-A has been operating since its 2006 launch, and EUVI continues to produce data. Combined with SDO/AIA, multi-viewpoint observation is still possible.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
