---
title: "Pre-reading Briefing: The McMath Solar Telescope of Kitt Peak National Observatory"
paper_id: "01_pierce_1964"
topic: Solar Observation
date: 2026-04-10
type: briefing
---

# 사전 브리핑: McMath Solar Telescope / Pre-reading Briefing: The McMath Solar Telescope

**논문 / Paper**: "The McMath Solar Telescope of Kitt Peak National Observatory"
**저자 / Authors**: A. Keith Pierce
**연도 / Year**: 1964
**저널 / Journal**: *Applied Optics*, Vol. 3, No. 12, pp. 1337–1346

---

## 1. 핵심 기여 / Core Contribution

McMath Solar Telescope는 80인치(203cm) 헬리오스탯과 60인치(152cm) 오목 주경을 사용하며, 초점 거리 300 feet(약 90m)를 가진 세계 최대의 태양 망원경입니다. 이 논문은 광학 경로의 열 제어, fused quartz 및 metal-based 거울의 선택과 성능, vacuum double-pass spectrometer와 single-pass spectrograph의 광학 배치 및 성능을 기술합니다. Resolution 600,000, 산란광(scattered light) 3%의 성능을 달성하여 Fraunhofer 선의 고정밀 측광 프로파일과 Doppler shift 매핑을 가능케 했습니다.

The McMath Solar Telescope uses an 80-inch (203 cm) heliostat and a 60-inch (152 cm) concave primary mirror with a 300-foot (~90 m) focal length, making it the world's largest solar telescope. This paper describes the thermal control of the optical path, the selection and performance of fused quartz and metal-based mirrors, and the optical layout and performance of both the vacuum double-pass spectrometer and single-pass spectrograph. Achieving a resolution of 600,000 with 3% total scattered light, it enabled high-precision photometric profiles of Fraunhofer lines and Doppler shift mapping of the solar atmosphere.

---

## 2. 역사적 맥락 / Historical Context

```
태양 관측 기기 발전 타임라인 / Solar Observation Instrument Timeline:

1868 ── Janssen & Lockyer: 분광기로 태양 홍염 관측 (spectroscopic prominences)
  │
1890s ─ Hale: Spectroheliograph 발명 → Mt. Wilson 태양탑 건설
  │
1930s ─ Lyot: Coronagraph 발명 (인공 일식)
  │
1940s ─ McMath-Hulbert Observatory 설립 (Michigan)
  │
1958 ── Kitt Peak National Observatory (KPNO) 설립
  │
★ 1960 ── McMath Solar Telescope 설계 논문 ← 이 논문 / THIS PAPER
  │
1962 ── McMath Solar Telescope 완공 및 첫 관측
  │
1964 ── Pierce: 망원경 성능 보고 (Applied Optics)
  │
1969 ── Dunn: Vacuum Tower Telescope (Sacramento Peak) ← 다음 논문 #2
  │
1985 ── McMath-Pierce로 개명 (Keith Pierce 공헌 기념)
  │
2002 ── NSO Synoptic Optical Long-term Investigation of the Sun (SOLIS) 프로젝트
  │
2020 ── Daniel K. Inouye Solar Telescope (DKIST) — 4m 구경, 현대 최대 태양 망원경
```

이 논문이 쓰인 1960년은 태양 물리학이 단순 관측에서 **정밀 분광학**으로 전환되던 시기입니다. 대형 망원경이 필요한 이유는 더 많은 빛을 모아 **고분산(high-dispersion) 스펙트럼**을 얻어야 했기 때문입니다. McMath 망원경은 이 요구를 충족시킨 최초의 시설이었습니다.

This paper was written in 1960, when solar physics was transitioning from simple observation to **precision spectroscopy**. The need for a large telescope arose from the requirement to collect enough light for **high-dispersion spectra**. The McMath telescope was the first facility to meet this demand.

---

## 3. 필요한 배경 지식 / Prerequisites

### 3.1 기초 광학 / Basic Optics

- **초점 거리 (focal length)**: 거울이나 렌즈가 평행광을 모으는 거리. 길수록 이미지가 크다 (image scale ↑).
  Focal length: distance at which a mirror/lens focuses parallel light. Longer = larger image scale.

- **이미지 스케일 (image scale)**: 초점면에서 1 arcsec에 해당하는 물리적 크기. $s = f \times \tan(1'') \approx f / 206265$ (mm/arcsec).
  Image scale: physical size per arcsecond at the focal plane.

- **F-number ($f/\#$)**: 초점 거리 / 구경. $f/\# = f/D$. 작을수록 밝고, 클수록 이미지가 느리지만 수차가 적다.
  F-number: focal length / aperture diameter. Lower = brighter, higher = less aberration.

### 3.2 태양 망원경 특유의 문제 / Solar Telescope Specific Issues

- **열 부하 (thermal load)**: 태양은 매우 밝아서 망원경 내부가 가열됨 → 공기 대류 → 이미지 흐림 (internal seeing).
  The Sun's brightness heats the telescope interior → air convection → image degradation.

- **시상 (seeing)**: 대기 난류로 인한 이미지 떨림. 지상 태양 관측의 최대 적.
  Atmospheric turbulence causing image distortion — the primary enemy of ground-based solar observation.

- **헬리오스탯 (heliostat)**: 태양을 추적하며 빛을 고정된 방향으로 반사하는 거울 시스템. 망원경 본체를 고정할 수 있어 대형화에 유리.
  A mirror system that tracks the Sun and reflects light in a fixed direction, allowing the telescope body to remain stationary — advantageous for large instruments.

- **쿠데 (coudé) 배치**: 빛을 여러 거울을 통해 고정된 관측실로 보내는 방식. 대형 분광기를 설치할 수 있음.
  Coudé configuration: routing light through multiple mirrors to a fixed observing room, enabling large spectrographs.

### 3.3 분광학 기초 / Spectroscopy Basics

- **회절 격자 (diffraction grating)**: 빛을 파장별로 분리. 격자 상수(groove spacing)와 차수(order)가 분해능을 결정.
  Separates light by wavelength. Groove spacing and diffraction order determine resolving power.

- **분광 분해능 (spectral resolving power)**: $R = \lambda / \Delta\lambda$. 높을수록 좁은 파장 차이를 구별할 수 있음.
  $R = \lambda / \Delta\lambda$. Higher R means finer wavelength discrimination.

- **Fraunhofer 흡수선**: 태양 대기의 원소들이 특정 파장에서 빛을 흡수하여 생기는 어두운 선. 이 선들의 모양(profile)에서 온도, 밀도, 속도, 자기장 정보를 추출.
  Dark lines in the solar spectrum caused by element absorption. Line profiles encode temperature, density, velocity, and magnetic field information.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **Heliostat** | 태양을 추적하며 빛을 고정 방향으로 반사하는 거울. 시계 장치(clock drive)로 구동 / A Sun-tracking mirror that reflects light in a fixed direction, driven by a clock mechanism |
| **Image scale** | 초점면에서의 각도당 물리적 크기 (mm/arcsec). McMath는 약 2.7 mm/arcsec / Physical size per angular unit at the focal plane |
| **Spectrograph** | 빛을 스펙트럼으로 분산시켜 기록하는 장치 / Instrument that disperses light into a spectrum for recording |
| **Grating** | 회절 격자. 미세한 홈(groove)이 새겨진 광학 소자로 빛을 파장별로 분리 / Diffraction grating — optical element with fine grooves that separates light by wavelength |
| **Resolving power** | 분광기가 인접한 두 파장을 구분하는 능력, $R = \lambda/\Delta\lambda$ / Ability of a spectrograph to distinguish two adjacent wavelengths |
| **Seeing** | 대기 난류로 인한 이미지 품질 저하. 보통 arcsec 단위 / Image quality degradation due to atmospheric turbulence, typically measured in arcseconds |
| **Internal seeing** | 망원경 내부의 열 대류로 인한 추가적 이미지 흐림 / Additional image blurring from thermal convection inside the telescope |
| **Light cone** | 거울이 모은 빛이 초점을 향해 수렴하는 원뿔 형태의 광로 / The converging cone of light from the mirror to the focal point |
| **Focal plane** | 상이 맺히는 평면. 여기에 분광기 슬릿이나 카메라를 배치 / The plane where the image forms; where slit or camera is placed |
| **Infrared (IR)** | 가시광보다 긴 파장 ($\lambda > 700$ nm). McMath는 IR 태양 분광학의 선구적 시설 / Wavelengths longer than visible light; McMath pioneered solar IR spectroscopy |

---

## 5. 수식 미리보기 / Equations Preview

### 5.1 이미지 스케일 / Image Scale

$$s = \frac{f}{206265} \quad \text{(mm/arcsec)}$$

여기서 $f$는 초점 거리(mm). McMath의 주경 초점 거리가 약 90m (300 feet)이므로:

Where $f$ is the focal length (mm). With McMath's ~90m (300 ft) focal length:

$$s = \frac{90000}{206265} \approx 0.436 \text{ mm/arcsec}$$

태양 전면(~32 arcmin)의 이미지 크기: $32 \times 60 \times 0.436 \approx 837$ mm ≈ 84 cm. 논문에서 "approximately a yard across"라고 표현한 거대한 태양상입니다.

The full solar disk (~32 arcmin) image size: ~84 cm diameter — "approximately a yard across" as stated in the paper.

### 5.2 분광 분해능 / Spectral Resolving Power

$$R = \frac{\lambda}{\Delta\lambda} = mN$$

여기서 $m$은 회절 차수(diffraction order), $N$은 조사되는 격자 홈(groove)의 수.

Where $m$ is the diffraction order and $N$ is the number of illuminated grooves.

대형 격자(600 grooves/mm, 폭 30 cm)를 5차에서 사용하면:

For a large grating (600 grooves/mm, 30 cm wide) used in 5th order:

$$R = 5 \times (600 \times 300) = 900{,}000$$

이는 태양 흡수선의 미세한 비대칭이나 자기 분리(Zeeman splitting)를 관측하기에 충분합니다.

This is sufficient to observe subtle asymmetries or Zeeman splitting in solar absorption lines.

### 5.3 격자 방정식 / Grating Equation

$$m\lambda = d(\sin\alpha + \sin\beta)$$

여기서 $d$는 격자 간격(groove spacing), $\alpha$는 입사각, $\beta$는 회절각.

Where $d$ is groove spacing, $\alpha$ is incidence angle, $\beta$ is diffraction angle.

### 5.4 집광력 / Light-Gathering Power

$$\text{Light-gathering} \propto D^2$$

구경 $D$가 클수록 더 많은 빛을 모음. 1.6m 거울은 10cm 망원경 대비 $(160/10)^2 = 256$배 더 많은 빛을 수집 → 고분산 분광에 필수.

Larger aperture $D$ collects more light. A 1.6m mirror collects $(160/10)^2 = 256\times$ more light than a 10 cm telescope — essential for high-dispersion spectroscopy.

---

## 6. 읽기 가이드 / Reading Guide

논문을 읽을 때 다음에 주목하세요 / Pay attention to these while reading:

1. **설계 제약 (design constraints)**: 왜 헬리오스탯 방식을 선택했는가? 대안(equatorial mount)의 문제점은?
   Why was the heliostat design chosen? What are the problems with alternatives?

2. **열 관리 (thermal management)**: 태양광의 열 부하를 어떻게 처리하는가? 이 문제가 #2 논문(Vacuum Tower)의 동기가 됩니다.
   How is the thermal load from sunlight handled? This problem motivates Paper #2.

3. **광학 배치 (optical layout)**: 빛이 헬리오스탯에서 주경, 관측실까지 어떤 경로로 이동하는가?
   What path does light follow from heliostat to primary mirror to observing room?

4. **분광기 사양 (spectrograph specifications)**: 어떤 분해능을 달성하며, 이것이 태양 물리학 연구에 어떤 새로운 가능성을 열었는가?
   What resolving power is achieved, and what new solar physics does this enable?

5. **관측 조건 (observing conditions)**: Kitt Peak이 왜 선정되었는가? 고도, 기후, seeing 조건의 장점은?
   Why was Kitt Peak selected? Advantages of altitude, climate, seeing conditions?

---

## Q&A

*(읽기 중 질문이 추가됩니다 / Questions during reading will be appended here)*
