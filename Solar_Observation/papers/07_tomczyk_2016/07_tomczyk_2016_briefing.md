---
title: "Pre-Reading Briefing: Scientific Objectives and Capabilities of the Coronal Solar Magnetism Observatory"
paper_id: "07_tomczyk_2016"
topic: Solar Observation
date: 2026-04-16
type: briefing
---

# Scientific Objectives and Capabilities of the Coronal Solar Magnetism Observatory: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Tomczyk, S. et al. (2016), "Scientific Objectives and Capabilities of the Coronal Solar Magnetism Observatory," *Journal of Geophysical Research: Space Physics*, 121, 7470.
**Author(s)**: Steven Tomczyk, Enrico Landi, Joan T. Burkepile, Roberto Casini, Edward E. DeLuca, Yuhong Fan, Sarah E. Gibson, Haosheng Lin, Scott W. McIntosh, Stanley C. Solomon, Giuliana de Toma, Alfred G. de Wijn, Jie Zhang
**Year**: 2016
**DOI**: 10.1002/2016JA022871

---

## 1. 핵심 기여 / Core Contribution

이 논문은 **COSMO(Coronal Solar Magnetism Observatory)**의 과학적 목표와 기기 역량을 종합적으로 기술합니다. COSMO는 HAO/NCAR에서 개발 중인 차세대 지상 태양 관측 시설로, 코로나와 채층(chromosphere)의 자기장, 밀도, 온도, 속도를 일상적(synoptic)으로 측정하는 것을 목표로 합니다. 논문은 세 가지 핵심 기기—**K-Coronagraph (K-Cor)**, **Chromosphere and Prominence Magnetometer (ChroMag)**, **Large Coronagraph (LC)**—의 설계와 과학적 근거를 제시하며, 코로나 자기장 직접 측정이 우주 날씨 예보와 태양 물리학 발전에 왜 필수적인지를 논증합니다.

This paper comprehensively describes the scientific objectives and instrument capabilities of **COSMO (Coronal Solar Magnetism Observatory)**, a next-generation ground-based solar observing facility under development at HAO/NCAR. COSMO aims to provide routine (synoptic) measurements of the magnetic field, density, temperature, and velocity of the corona and chromosphere. The paper presents the design and scientific rationale for three core instruments—**K-Coronagraph (K-Cor)**, **Chromosphere and Prominence Magnetometer (ChroMag)**, and **Large Coronagraph (LC)**—and argues why direct coronal magnetic field measurement is essential for space weather forecasting and advancing solar physics.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

2016년까지 코로나 자기장은 태양 물리학에서 가장 중요하면서도 가장 측정하기 어려운 물리량으로 남아 있었습니다. 광구 자기장은 Zeeman 효과를 이용하여 일상적으로 측정되었지만 (GONG, HMI 등), 코로나에서는 자기장이 약하고 (~1–10 Gauss) 열적 선폭이 넓어 Zeeman 분리가 사실상 불가능했습니다. 대부분의 코로나 자기장 정보는 광구 측정에서 외삽(extrapolation)하여 얻었으나, 비선형 역학이 지배하는 코로나에서 외삽은 본질적으로 불확실했습니다.

By 2016, the coronal magnetic field remained the most important yet most difficult-to-measure quantity in solar physics. While photospheric magnetic fields were routinely measured using the Zeeman effect (GONG, HMI, etc.), the weak fields (~1–10 Gauss) and broad thermal line widths in the corona made Zeeman splitting virtually undetectable. Most coronal magnetic field information came from extrapolation of photospheric measurements, but such extrapolation is inherently uncertain in the nonlinear, dynamic corona.

한편 Tomczyk 본인이 2004년에 개발한 **CoMP(Coronal Multi-channel Polarimeter)**가 금지선(forbidden line) 편광 관측을 통해 코로나 자기장 방향과 Alfvén 파 전파를 최초로 성공적으로 감지했습니다. 이 성과가 COSMO Large Coronagraph의 직접적 동기가 되었습니다.

Meanwhile, Tomczyk himself had developed the **CoMP (Coronal Multi-channel Polarimeter)** in 2004, which successfully detected coronal magnetic field direction and Alfvénic wave propagation for the first time through forbidden-line polarization observations. This achievement directly motivated the COSMO Large Coronagraph.

### 타임라인 / Timeline

```
1930  ── Lyot, 최초의 코로나그래프 발명
         Lyot invents the first coronagraph

1940s ── 코로나 금지선 (Fe X, Fe XIV 등) 발견, 코로나 100만 K 확인
         Coronal forbidden lines discovered; corona confirmed at ~1 MK

1964  ── Pierce: McMath Solar Telescope (이 시리즈 #1)
         Pierce: McMath Solar Telescope (this series #1)

1980  ── HAO Mk-III K-coronameter, Mauna Loa 설치
         HAO Mk-III K-coronameter installed at Mauna Loa

1996  ── SOHO 발사 — LASCO 코로나그래프로 우주 CME 관측 시대 개막
         SOHO launched — LASCO opens the era of space CME observation

2004  ── Tomczyk: CoMP 개발, 코로나 편광 관측 시작
         Tomczyk: CoMP developed, coronal polarization observations begin

2007  ── CoMP로 코로나 Alfvén 파 최초 관측 (Tomczyk et al., Science)
         First detection of coronal Alfvén waves with CoMP

2013  ── K-Cor, Mauna Loa에 배치 (Mk4 대체)
         K-Cor deployed to Mauna Loa (replacing Mk4)

2016  ── ★ 이 논문: COSMO 과학 목표와 전체 기기 역량 기술 ★
         ★ This paper: COSMO science objectives and full instrument capabilities ★

2021  ── UCoMP (Upgraded CoMP) 가동 시작
         UCoMP (Upgraded CoMP) begins operation
```

---

## 3. 필요한 배경 지식 / Prerequisites

### 코로나그래프 원리 / Coronagraph Principles

- **Lyot 코로나그래프**: 태양 원반의 직사광을 차폐판(occulting disk)으로 가려 희미한 코로나를 관측하는 기기입니다. 내부 차폐(internally occulted)와 외부 차폐(externally occulted) 방식이 있습니다.
  A **Lyot coronagraph** uses an occulting disk to block the bright solar disk, enabling observation of the faint corona. Internal and external occulting designs exist.

- **산란광 (Stray light)**: 코로나 밝기는 태양 원반의 ~10⁻⁶ 수준이므로, 광학 소자에 의한 산란광 억제가 핵심 과제입니다. Lyot stop이 회절광을 차단합니다.
  Coronal brightness is ~10⁻⁶ of the solar disk, so suppressing stray light from optical elements is the central engineering challenge. Lyot stops block diffracted light.

### Thomson 산란과 편광 밝기 (pB) / Thomson Scattering and Polarization Brightness

- 코로나 자유 전자가 광구 광자를 Thomson 산란하면 선편광이 생깁니다. 편광 밝기(**pB**)를 측정하면 시선 방향 전자 밀도 적분값을 얻을 수 있습니다.
  Free electrons in the corona Thomson-scatter photospheric photons, producing linearly polarized light. Measuring the **polarization brightness (pB)** yields the line-of-sight integrated electron density.

### 코로나 금지선 / Coronal Forbidden Lines

- Fe XIII 1074.7 nm, 1079.8 nm 등의 금지선은 코로나 조건에서만 방출됩니다. 이 선의 **선형 편광** (Stokes Q, U)은 자기장 방향(POS)을, **원형 편광** (Stokes V, 세로 Zeeman/Hanle 효과)은 시선 방향 자기장 세기를 나타냅니다.
  Forbidden lines like Fe XIII 1074.7 nm and 1079.8 nm are emitted only under coronal conditions. Their **linear polarization** (Stokes Q, U) reveals POS magnetic field direction, while **circular polarization** (Stokes V, longitudinal Zeeman/Hanle effect) indicates LOS magnetic field strength.

### 이전 논문과의 연결 / Connection to Previous Papers

- **#5 (Harvey 1996, GONG)**: 전 지구 네트워크를 통한 연속 관측의 중요성 — COSMO도 Mauna Loa에서 시놉틱(synoptic) 연속 관측을 목표로 합니다.
- **#6 (Chaplin 1996, BiSON)**: 공명 산란 분광 기법의 원리 — COSMO ChroMag도 유사한 분광 편광 기법을 사용합니다.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **COSMO** | Coronal Solar Magnetism Observatory — HAO/NCAR의 차세대 지상 코로나 관측 시설 / Next-generation ground-based coronal observing facility at HAO/NCAR |
| **K-Cor (K-Coronagraph)** | 백색광 코로나그래프. Thomson 산란 편광(pB)을 측정하여 코로나 전자 밀도와 CME 동역학을 관측 / White-light coronagraph measuring pB to observe coronal electron density and CME dynamics |
| **ChroMag** | Chromosphere and Prominence Magnetometer — 채층·프로미넌스의 자기장과 플라즈마를 전일면(full-disk) 분광편광으로 관측 / Full-disk spectropolarimetric observations of chromosphere and prominence magnetism and plasma |
| **Large Coronagraph (LC)** | 1.5 m 구경의 대형 코로나그래프. 코로나 금지선 편광을 측정하여 코로나 자기장, 온도, 밀도, 속도를 동시 관측 / 1.5 m aperture coronagraph measuring forbidden-line polarization to simultaneously observe coronal magnetic field, temperature, density, and velocity |
| **pB (polarization brightness)** | 편광 밝기 — Thomson 산란에 의한 코로나의 선편광 성분, 전자 밀도에 비례 / Linearly polarized component of coronal light from Thomson scattering, proportional to electron density |
| **Forbidden lines** | 코로나 조건(저밀도, 고온)에서만 관측되는 금지 천이선 (예: Fe XIII 1074.7/1079.8 nm) / Emission lines from forbidden transitions observable only under coronal conditions (low density, high temperature) |
| **Stokes parameters (I, Q, U, V)** | 빛의 편광 상태를 완전히 기술하는 네 매개변수. I=총강도, Q/U=선편광, V=원편광 / Four parameters fully describing light polarization state: I=total intensity, Q/U=linear polarization, V=circular polarization |
| **Coronal seismology** | 코로나 MHD 파동의 특성으로부터 자기장 등 물리량을 추론하는 기법 / Technique to infer physical quantities like magnetic field from properties of coronal MHD waves |
| **Alfvén wave** | 자기장을 따라 전파하는 MHD 파. 위상 속도 $v_A = B/\sqrt{\mu_0 \rho}$로 자기장 세기를 추정 가능 / MHD wave propagating along magnetic field lines; phase speed $v_A = B/\sqrt{\mu_0 \rho}$ enables magnetic field estimation |
| **Synoptic observation** | 태양 전체를 장기간 반복적으로 관측하여 시간 변화를 추적하는 관측 방식 / Long-term, repetitive observation of the full Sun to track temporal evolution |
| **CME (Coronal Mass Ejection)** | 코로나 질량 방출 — 코로나에서 대량의 플라즈마와 자기장이 행성간 공간으로 분출되는 현상 / Eruption of large amounts of plasma and magnetic field from the corona into interplanetary space |

---

## 5. 수식 미리보기 / Equations Preview

### 5.1 Thomson 산란 편광 밝기 / Thomson Scattering Polarization Brightness

$$pB = \int n_e \cdot \sigma_T \cdot G(r) \, dl$$

여기서 $n_e$는 전자 밀도, $\sigma_T$는 Thomson 산란 단면적, $G(r)$은 기하학적 감쇠 함수, $l$은 시선 방향 적분 경로입니다.
Where $n_e$ is electron density, $\sigma_T$ is the Thomson scattering cross-section, $G(r)$ is a geometric dilution function, and $l$ is the line-of-sight integration path.

### 5.2 Alfvén 파 위상 속도와 코로나 자기장 / Alfvén Wave Phase Speed and Coronal Magnetic Field

$$v_A = \frac{B}{\sqrt{\mu_0 \rho}}$$

코로나 seismology에서 Alfvén 파의 위상 속도 $v_A$를 Doppler 관측으로 측정하고, 밀도 $\rho$를 금지선 비율로 추정하면 자기장 세기 $B$를 간접적으로 계산할 수 있습니다.
In coronal seismology, measuring the Alfvén wave phase speed $v_A$ via Doppler observations and estimating density $\rho$ from forbidden-line ratios allows indirect calculation of the magnetic field strength $B$.

$$B = v_A \sqrt{\mu_0 \rho}$$

### 5.3 금지선 비율과 전자 밀도 / Forbidden Line Ratio and Electron Density

$$R = \frac{I(1079.8\,\text{nm})}{I(1074.7\,\text{nm})} \propto f(n_e)$$

Fe XIII 두 금지선의 강도 비율은 전자 밀도에 민감하여 코로나 밀도 진단 도구로 사용됩니다.
The intensity ratio of the two Fe XIII forbidden lines is sensitive to electron density and serves as a coronal density diagnostic.

### 5.4 Zeeman 분리 / Zeeman Splitting

$$\Delta \lambda_Z = \frac{e \lambda^2 B}{4 \pi m_e c}$$

코로나에서 $B \sim 1\text{–}10$ G일 때 Zeeman 분리는 열적 선폭보다 훨씬 작아 직접 감지가 어렵습니다. 이것이 금지선 편광(선편광: Van Vleck 효과, 원편광: 세로 Zeeman)을 이용하는 이유입니다.
For coronal $B \sim 1\text{–}10$ G, Zeeman splitting is much smaller than thermal line widths, making direct detection nearly impossible. This is why forbidden-line polarization (linear: Van Vleck effect; circular: longitudinal Zeeman) is used instead.

---

## 6. 읽기 가이드 / Reading Guide

### 권장 읽기 순서 / Recommended Reading Order

1. **Introduction & Science Objectives (§1–2)**: 코로나 자기장 측정의 중요성과 현재의 한계를 파악하세요. 왜 외삽만으로는 부족한지 이해하는 것이 핵심입니다.
   Understand why coronal magnetic field measurement matters and what the current limitations are. The key insight is why extrapolation alone is insufficient.

2. **K-Cor (§3 또는 해당 섹션)**: 이미 Mauna Loa에 배치된 기기입니다. 백색광 pB 측정의 원리와 CME 조기 감지 능력에 주목하세요. FOV가 1.05 $R_\odot$까지 내려가는 점이 중요합니다.
   Already deployed at Mauna Loa. Note the white-light pB measurement principles and CME early detection capability. The FOV extending down to 1.05 $R_\odot$ is significant.

3. **ChroMag (§해당 섹션)**: 채층-코로나 경계의 자기장 측정입니다. 다중 파장(He I, Hα, Ca II, Fe I)을 통해 서로 다른 대기층을 동시에 관측하는 설계에 주목하세요.
   Focuses on chromosphere-corona boundary magnetic fields. Note the multi-wavelength design (He I, Hα, Ca II, Fe I) for simultaneously observing different atmospheric layers.

4. **Large Coronagraph (§해당 섹션)**: 논문의 핵심입니다. 1.5 m 구경, FOV 1°, 공간 분해능 2″, 분광 분해능 λ/Δλ > 8000의 사양을 기억하세요. 특히 **코로나 금지선 편광 측정**이 어떻게 자기장, 온도, 밀도, 속도를 동시에 제공하는지에 집중하세요.
   The core of the paper. Remember the specs: 1.5 m aperture, 1° FOV, 2″ spatial resolution, spectral resolution λ/Δλ > 8000. Focus on how **forbidden-line polarization measurement** simultaneously provides magnetic field, temperature, density, and velocity.

5. **Space Weather Applications**: CME 예측과 코로나 자기장 측정이 우주 날씨 예보에 어떻게 기여하는지 확인하세요.
   See how CME prediction and coronal magnetic field measurement contribute to space weather forecasting.

### 주의할 점 / Points to Watch

- **세 기기의 상호보완성**: K-Cor(밀도/동역학), ChroMag(채층 자기장), LC(코로나 자기장)가 어떻게 통합적 그림을 제공하는지 파악하세요.
  How K-Cor (density/dynamics), ChroMag (chromospheric magnetism), and LC (coronal magnetism) provide a unified picture.

- **CoMP와의 관계**: COSMO LC는 CoMP의 과학적 성과를 바탕으로 설계된 "다음 세대" 기기입니다. CoMP의 한계(작은 구경 → 낮은 신호 대 잡음비)를 어떻게 극복하는지 주목하세요.
  COSMO LC is designed as the "next generation" of CoMP. Note how it overcomes CoMP's limitations (small aperture → low SNR).

- **지상 vs 우주 관측의 상보성**: 논문이 COSMO를 우주 기반 관측(SOHO, SDO 등)의 대체가 아닌 보완으로 위치시키는 방식에 주목하세요.
  Note how the paper positions COSMO as complementary to (not replacing) space-based observatories like SOHO and SDO.

---

## 7. 현대적 의의 / Modern Significance

- **UCoMP (2021–)**: COSMO LC의 프로토타입 역할을 하는 Upgraded CoMP가 Mauna Loa에서 가동 중이며, 코로나 자기장 관측의 실현 가능성을 지속적으로 검증하고 있습니다.
  The Upgraded CoMP, serving as a prototype for the COSMO LC, is operational at Mauna Loa, continuing to validate the feasibility of coronal magnetic field observations.

- **DKIST와의 시너지**: 2020년 가동을 시작한 4 m DKIST(이 시리즈 #23, Rimmele 2020)는 고분해능 코로나 자기장 관측이 가능하지만 시놉틱 관측에는 적합하지 않습니다. COSMO는 넓은 시야각으로 시놉틱 관측을 제공하여 DKIST를 보완합니다.
  The 4 m DKIST (this series #23), operational since 2020, can observe coronal magnetic fields at high resolution but is not suited for synoptic observations. COSMO provides wide-FOV synoptic coverage complementing DKIST.

- **우주 날씨 운용 예보**: 미국 NOAA/SWPC가 CME 도착 시간과 지자기 폭풍 세기를 예보하는 데 코로나 자기장 데이터가 핵심 누락 입력으로 인식되고 있습니다. COSMO는 이 gap을 메우는 것을 목표로 합니다.
  NOAA/SWPC recognizes coronal magnetic field data as a critical missing input for CME arrival time and geomagnetic storm intensity forecasting. COSMO aims to fill this gap.

- **코로나 seismology의 발전**: CoMP/UCoMP에서 발견된 편재하는 Alfvénic 파동은 코로나 가열 문제와 직결되며, COSMO LC의 높은 감도로 보다 정밀한 연구가 기대됩니다.
  The ubiquitous Alfvénic waves discovered by CoMP/UCoMP are directly linked to the coronal heating problem, and the higher sensitivity of COSMO LC promises more precise studies.

---

## Q&A

### Q1. Thomson 산란의 상세 원리 / Thomson Scattering in Detail

#### 기본 원리 / Basic Principle

Thomson 산란은 **자유 전자에 의한 전자기파의 탄성 산란**입니다. 광자의 에너지(파장)가 변하지 않고, 전자의 진동에 의해 방향만 바뀝니다.

Thomson scattering is the **elastic scattering of electromagnetic waves by free electrons**. The photon energy (wavelength) remains unchanged; only the direction changes due to the electron's oscillation.

광구에서 방출된 광자가 코로나의 자유 전자를 만나면:

When a photon emitted from the photosphere encounters a free electron in the corona:

```
광구 광자 (비편광)  ──→  코로나 자유 전자  ──→  산란된 광자 (부분 편광)
Photospheric photon     Coronal free         Scattered photon
(unpolarized)           electron             (partially polarized)
```

핵심은 전자가 **입사 전기장의 진동 방향으로 가속**되어 쌍극자(dipole) 복사를 한다는 점입니다. 이 때문에 산란각에 따라 편광 특성이 달라집니다.

The key point is that the electron is **accelerated in the oscillation direction of the incident electric field**, producing dipole radiation. This causes the polarization properties to vary with scattering angle.

#### 산란 단면적 / Scattering Cross-Section

Thomson 산란 단면적은 전자의 고전 반지름 $r_e$로 결정됩니다:

The Thomson scattering cross-section is determined by the classical electron radius $r_e$:

$$\sigma_T = \frac{8\pi}{3} r_e^2 = 6.65 \times 10^{-29} \,\text{m}^2$$

여기서 / where $r_e = e^2 / (4\pi \epsilon_0 m_e c^2) = 2.82 \times 10^{-15}$ m (고전 전자 반지름 / classical electron radius)

이 값은 **매우 작습니다**. 코로나의 전자 밀도가 $n_e \sim 10^8\text{–}10^9$ cm⁻³ 정도이므로, 코로나 밝기가 태양 원반의 ~$10^{-6}$ 수준인 이유가 바로 이것입니다.

This value is **extremely small**. With coronal electron density of $n_e \sim 10^8\text{–}10^9$ cm⁻³, this explains why coronal brightness is only ~$10^{-6}$ of the solar disk.

#### 산란각과 편광 / Scattering Angle and Polarization

미분 산란 단면적은 / The differential scattering cross-section is:

$$\frac{d\sigma}{d\Omega} = \frac{r_e^2}{2}(1 + \cos^2 \chi)$$

편광도(degree of polarization)는 / The degree of polarization is:

$$p = \frac{1 - \cos^2 \chi}{1 + \cos^2 \chi} = \frac{\sin^2 \chi}{1 + \cos^2 \chi}$$

- **산란각 χ = 90°** (태양 림 바로 위 / directly above solar limb): 편광도가 **최대** / polarization is **maximum**
- **산란각 χ = 0° 또는 180°** (시선 방향 / along LOS): 편광도가 **0** / polarization is **zero**

#### K-corona, F-corona, E-corona

태양 코로나의 백색광은 세 성분으로 나뉩니다:

The white-light corona is divided into three components:

| 성분 / Component | 원인 / Origin | 특성 / Characteristics |
|---|---|---|
| **K-corona** (Kontinuierlich) | 자유 전자의 Thomson 산란 / Thomson scattering by free electrons | 연속 스펙트럼, **편광됨**, 전자 밀도에 비례 / Continuous spectrum, **polarized**, proportional to $n_e$ |
| **F-corona** (Fraunhofer) | 행성간 먼지의 산란 / Scattering by interplanetary dust | Fraunhofer 흡수선 보존, 비편광 / Fraunhofer lines preserved, unpolarized |
| **E-corona** (Emission) | 코로나 이온의 방출선 / Emission lines from coronal ions | 금지선 (Fe X, Fe XIII 등) / Forbidden lines (Fe X, Fe XIII, etc.) |

K-Cor가 측정하는 것이 바로 **K-corona**입니다. 전자의 열속도($v_{th} \sim 0.03c$)에 의한 Doppler broadening으로 Fraunhofer 흡수선이 **완전히 사라져** 매끄러운 연속 스펙트럼이 됩니다.

K-Cor measures the **K-corona**. The thermal velocity of electrons ($v_{th} \sim 0.03c$) causes Doppler broadening that **completely smears out** Fraunhofer absorption lines, producing a smooth continuum.

#### 편광 밝기 (pB)와 전자 밀도 / Polarization Brightness and Electron Density

$$tB = B_T + B_R \quad,\quad pB = B_T - B_R$$

여기서 / where:
- $B_T$: 태양 림에 **접선(tangential)** 방향의 편광 성분 / tangential polarization component
- $B_R$: **방사(radial)** 방향의 편광 성분 / radial polarization component

시선 적분 / Line-of-sight integration:

$$pB(r) = \frac{\pi \sigma_T \bar{B}_\odot}{2} \int_{-\infty}^{\infty} n_e(l) \cdot \mathcal{G}(r, l) \, dl$$

**핵심**: pB는 전자 밀도 $n_e$에 **선형 비례**하므로, pB를 측정하면 코로나 전자 밀도 분포를 직접 구할 수 있습니다.

**Key point**: pB is **linearly proportional** to $n_e$, so measuring pB directly yields the coronal electron density distribution.

#### 지상 관측에서 편광 측정이 필수적인 이유 / Why Polarization Measurement Is Essential for Ground-Based Observation

```
관측 신호 = K-corona (편광) + F-corona (비편광) + 하늘 산란광 (비편광) + 기기 산란광
Signal    = K-corona (pol.) + F-corona (unpol.) + sky scatter (unpol.) + instrument scatter
```

pB 측정 시 비편광 성분(F-corona, 하늘 배경)이 자동으로 제거됩니다.

When measuring pB, unpolarized components (F-corona, sky background) are automatically removed.

수치 예시 ($r = 1.5 R_\odot$) / Numerical example:

| 물리량 / Quantity | 값 / Value |
|---|---|
| 전자 밀도 / Electron density $n_e$ | ~$10^8$ cm⁻³ |
| K-corona 밝기 / brightness | ~$10^{-6} B_\odot$ |
| 편광도 / Degree of polarization | ~40–60% |
| 맑은 하늘 밝기 / Clear sky brightness (Mauna Loa) | ~$10^{-5} B_\odot$ |

하늘 밝기가 코로나보다 **10배 이상 밝기** 때문에, 편광 측정 없이는 지상에서 코로나를 관측하는 것이 사실상 불가능합니다.

Since sky brightness is **>10× brighter** than the corona, ground-based coronal observation is virtually impossible without polarization measurement.

---

### Q2. 광구 광자가 비편광인 이유 / Why Photospheric Photons Are Unpolarized

광구의 빛은 **열복사(thermal radiation)**입니다. 광구 플라즈마에서 수많은 원자·이온·전자가 **무작위 방향으로 독립적으로** 복사를 방출합니다.

Photospheric light is **thermal radiation**. Countless atoms, ions, and electrons in the photospheric plasma emit radiation **independently in random directions**.

주요 복사 과정 / Major radiation processes:

| 과정 / Process | 설명 / Description |
|---|---|
| **자유-속박 복사** (free-bound) | 자유 전자가 이온에 포획되며 광자 방출 / Free electron captured by ion, emitting photon |
| **자유-자유 복사** (free-free, bremsstrahlung) | 자유 전자가 이온 근처에서 감속되며 광자 방출 / Free electron decelerated near ion, emitting photon |
| **속박-속박 천이** (bound-bound) | 원자/이온의 에너지 준위 천이 → Fraunhofer 흡수선 / Atomic energy level transitions → Fraunhofer absorption lines |
| **H⁻ 연속 흡수/방출** | 광구에서 가장 중요한 연속 opacity 원천 / Dominant continuous opacity source in photosphere |

개별 광자는 특정 편광 상태를 가지지만, **통계적으로 모든 방향이 동등**(등방적 열평형)하므로 합산하면 비편광이 됩니다.

Individual photons may have specific polarization states, but **statistically all directions are equal** (isotropic thermal equilibrium), so the sum is unpolarized.

---

### Q3. Thomson 산란에서 편광이 발생하는 기하학적 원리 / Geometric Origin of Polarization in Thomson Scattering

#### 핵심 의문 / Core Question

코로나도 밀도가 있으므로 광구와 마찬가지로 모든 방향으로 산란이 이루어져 편광이 상쇄되어야 하는 것 아닌가?

Since the corona has some density, shouldn't scattering occur in all directions just like in the photosphere, cancelling out polarization?

#### 답변: 광원의 비등방성 / Answer: Anisotropy of the Light Source

핵심은 **광구가 점이 아니라 원반(disk)**이고, 코로나 전자가 보는 광구의 **기하학적 분포가 비등방적**이라는 것입니다.

The key is that **the photosphere is not a point but a disk**, and the **geometric distribution of the photosphere as seen by a coronal electron is anisotropic**.

코로나 전자 하나의 시점에서 / From the perspective of a single coronal electron:

```
              관측자 (지구) / Observer (Earth)
                👁
                |
                |  시선 방향 (z축) / LOS direction (z-axis)
                |
            ◦ ← 코로나 전자 (림 위) / Coronal electron (above limb)
           /|\
          / | \
         /  |  \
        ■■■■■■■■■  ← 태양 원반 (아래쪽에만 있음) / Solar disk (below only)
```

이 전자는 **아래쪽(태양 방향)에서만** 광자를 받습니다. 위, 옆에서는 광자가 오지 않습니다.

This electron receives photons **only from below (solar direction)**. No photons come from above or the sides.

#### 단계별 편광 발생 과정 / Step-by-Step Polarization Generation

**Step 1**: 아래에서 올라오는 광자(방사 방향)의 전기장은 방사 방향에 수직인 평면 내에서 무작위 방향으로 진동합니다.

The electric field of upward-traveling photons (radial direction) oscillates randomly within the plane perpendicular to the radial direction.

**Step 2**: Thomson 산란에서 전자는 입사 전기장 방향으로 진동하고, 그 진동 방향에 수직으로 복사합니다.

In Thomson scattering, the electron oscillates along the incident E-field direction and radiates perpendicular to that oscillation.

**Step 3**: 관측자 방향으로 나오는 산란광 분석 / Analysis of scattered light toward observer:

```
        👁 관측자 / Observer (z 방향)
        |
        |
    ◦ 전자 / Electron
    
  입사 전기장의 두 성분 / Two components of incident E-field:
  
  (a) ↕ 접선 성분 (시선에 수직) / Tangential component (⊥ to LOS)
      → 전자가 ↕ 방향으로 진동 / Electron oscillates ↕
      → 관측자 방향(z)으로 복사 가능 ✅ / Can radiate toward observer ✅
      
  (b) ↔ 시선 방향 성분 (z 방향) / LOS component (z direction)
      → 전자가 z 방향으로 진동 / Electron oscillates along z
      → z 방향으로는 복사 불가 ❌ (쌍극자 복사 특성) / Cannot radiate along z ❌ (dipole radiation)
```

**접선 성분만 관측자에게 도달**하고 시선 방향 성분은 차단되므로, 비편광이었던 입사광에서 한 성분이 선택적으로 제거 → **편광 발생**

Only the **tangential component reaches the observer**; the LOS component is blocked. One component is selectively removed from the originally unpolarized light → **polarization arises**.

#### 광구와의 핵심 차이 / Key Difference from Photosphere

```
광구 원자의 상황:              코로나 전자의 상황:
Photospheric atom:           Coronal electron:
                         
  ☀☀☀☀☀                        
  ☀ 원자 ☀  ← 사방에서 광자       ◦ ← 한쪽(아래)에서만 광자
  ☀☀☀☀☀    photons from         ■■■■■■■ (태양 / Sun)
            all sides            photons from one side only
                         
  → 대칭 → 비편광                → 비대칭 → 편광!
     Symmetric → unpol.            Asymmetric → polarized!
```

---

### Q4. Thomson 산란과 3D 위치 모호성 / Thomson Scattering and 3D Position Ambiguity

#### 핵심 문제 / Core Problem

Thomson 산란으로 코로나 밀도를 분석하려면, 관측 중인 코로나가 태양–지구 기준으로 **어느 방향(3D 위치)에 있는지**를 알아야 합니다. 시선 방향(LOS)과 하늘 평면(POS)에 있는 코로나는 산란각이 달라 편광 특성과 산란 효율이 다릅니다.

To analyze coronal density via Thomson scattering, one must know the **3D position** of the coronal structure relative to the Sun–Earth line. Corona on the LOS vs. POS has different scattering angles, hence different polarization properties and scattering efficiency.

```
  전자가 POS에 있을 때:           전자가 LOS에 있을 때:
  Electron on POS:              Electron on LOS:
  
       👁                            👁
       |                             |
       |                             ◦ ← 전자 (태양 앞/뒤)
       ◦ (림 위)                          electron (in front/behind Sun)
       |                             |
       ● 태양 / Sun                  ● 태양 / Sun
       
  산란각 χ ≈ 90°                  산란각 χ ≈ 0° 또는 180°
  → 편광 최대, 산란 효율 높음       → 편광 제로, 산란 효율 낮음
  Max polarization, high eff.    Zero polarization, low eff.
```

#### 투영 모호성 문제 / Projection Ambiguity

코로나그래프 영상은 **2D 투영(projection)**이므로, 시선 위의 여러 전자들의 기여가 합산됩니다:

Coronagraph images are **2D projections**, so contributions from multiple electrons along the LOS are summed:

```
  👁 ──────◦────◦────◦────◦──── 
           A    B    C    D     ← 시선 위의 여러 전자들 / electrons along LOS
           
  전자 A: 관측자 가까이 → χ 작음 → pB 기여 작음 / near observer, small χ, low pB
  전자 B: POS 근처     → χ ≈ 90° → pB 기여 최대 / near POS, max pB
  전자 C: POS 근처     → χ ≈ 90° → pB 기여 최대 / near POS, max pB
  전자 D: 태양 뒤쪽    → χ 작음 → pB 기여 작음 / behind Sun, small χ, low pB
```

**POS 근처의 전자가 pB에 가장 크게 기여**합니다 — 이것이 pB 측정의 중요한 특성입니다.

**Electrons near POS contribute most to pB** — this is an important property of pB measurements.

#### 3D 밀도 복원 방법들 / Methods for 3D Density Reconstruction

**방법 1: 구대칭(spherical symmetry) 가정 + Abel 역변환**

코로나 밀도가 $n_e = n_e(r)$이라 가정하면 Abel inversion으로 pB(투영 거리)에서 $n_e(r)$을 복원할 수 있습니다. 조용한 코로나에서 합리적이나 활동 영역 근처에서는 부정확합니다.

Assuming $n_e = n_e(r)$, Abel inversion recovers $n_e(r)$ from pB(projected distance). Reasonable for quiet corona but inaccurate near active regions.

**방법 2: 다시점 관측 (STEREO)**

```
        지구 👁          STEREO-A 👁
        Earth             
              \          /
               \        /
                ◦ 코로나 전자 / coronal electron
                |
                ● 태양 / Sun
                
  → 삼각측량으로 3D 위치 결정 / Triangulation for 3D position
```

**방법 3: 태양 자전 이용 tomography**

~14일간 태양 자전에 따른 다각도 pB로 3D 밀도 복원. 단, 코로나 구조가 자전 기간 동안 변하지 않아야 합니다.

~14-day solar rotation provides multi-angle pB for 3D density reconstruction. Requires coronal structures to remain static during rotation.

**방법 4: pB/tB 비율 활용**

pB는 산란각에 강하게, tB는 약하게 의존하므로, pB/tB 비율로 유효 산란각을 추정하여 구조의 LOS 깊이를 제한할 수 있습니다.

Since pB depends strongly on scattering angle while tB depends weakly, the pB/tB ratio can estimate the effective scattering angle and constrain LOS depth.

#### COSMO와의 연결 / Connection to COSMO

이 투영 모호성이 COSMO가 **여러 물리량을 동시 측정**하려는 이유입니다:

This projection ambiguity is why COSMO aims to **measure multiple physical quantities simultaneously**:

| K-Cor (pB) | LC (금지선 / Forbidden lines) | 조합 효과 / Combined benefit |
|---|---|---|
| 시선 적분 전자 밀도 ($\propto n_e$) / LOS-integrated $n_e$ | 금지선 비율 → 국소 밀도 / Line ratio → local density | 3D 밀도 구조 제약 가능 / Constrains 3D density structure |
| POS 근처에 가중 / Weighted near POS | $n_e^2$ 의존 → 더 국소적 / $n_e^2$ dependence → more localized | 상호보완적 밀도 진단 / Complementary density diagnostics |

금지선 방출은 **$n_e^2$에 비례**하므로 Thomson 산란($n_e$ 비례)보다 시선 방향 국소화(localization)가 우수합니다.

Forbidden-line emission is **proportional to $n_e^2$**, providing better LOS localization than Thomson scattering (proportional to $n_e$).
