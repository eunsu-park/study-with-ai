---
title: "Pre-Reading Briefing: The Visible Spectro-Polarimeter of the Daniel K. Inouye Solar Telescope"
paper_id: "24_de_wijn_2022"
topic: Solar_Observation
date: 2026-04-19
type: briefing
---

# The Visible Spectro-Polarimeter (ViSP) of DKIST: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Alfred G. de Wijn, Roberto Casini, Amanda Carlile, et al., "The Visible Spectro-Polarimeter of the Daniel K. Inouye Solar Telescope", *Solar Physics*, Vol. 297, Article 22 (2022). [DOI: 10.1007/s11207-022-01954-1]
**Author(s)**: A. G. de Wijn (PI, HAO/NCAR), R. Casini (HAO), A. Carlile, A. R. Lecinski, S. Sewell, P. Zmarzly (HAO), A. D. Eigenbrot (NSO), C. Beck (NSO), F. Wöger (NSO), M. Knölker (HAO)
**Year**: 2022

---

## 1. 핵심 기여 / Core Contribution

### 한국어
이 논문은 DKIST의 **첫 세대 first-light 기기 중 하나인 Visible Spectro-Polarimeter (ViSP)**를 상세히 기술한다. ViSP는 **slit-scanning echelle spectrograph** 기반의 분광편광 측정기로, 다음 세 가지 독보적 특징을 갖는다: **(i) Wavelength versatility** — 380–900 nm 가시광/근적외선 범위의 **어떤 파장이든** 자동 재구성 가능 (기존 CRISP, TESOS 같은 Fabry-Pérot 기기는 전용 pre-filter 때문에 불가능), **(ii) 3-arm 동시 관측** — 3개의 독립된 camera arm이 임의의 파장 조합을 동시에 관측 (예: Ca II H+K + Mg I b + Na I D + Ca II IR triplet), **(iii) 높은 사양** — $R \gtrsim 180,000$ spectral resolving power, DKIST 회절 한계의 **2배 공간 분해능**, **$5 \times 10^{-4} I_{\text{cont}}$ polarimetric accuracy**, **$10^{-4} I_{\text{cont}}$ sensitivity** in 10 s. 이러한 사양은 polychromatic polarization modulator, dual-beam polarimetry, 자동 회전 camera arm (grating tilt 자동 조정), 5개의 slit aperture (0.028″–0.2″) 조합으로 달성된다. **Hanle effect로 약한 자기장 측정**, **다중 라인 관측을 통한 광구→코로나 "tomography"**, **Ca II 라인으로 채층 자기장 진단** 등이 주요 과학 사용 케이스이다. DKIST의 5대 first-light 기기 중 **유일한 "wavelength-versatile" 분광편광측정기**이며, discovery instrument로서 **새로운 편광 진단법을 탐색**할 수 있는 유연성을 제공한다.

### English
This paper describes ViSP, a **slit-scanning echelle spectrograph-based spectro-polarimeter** that is one of DKIST's first-light instruments. ViSP is distinguished by three capabilities: **(i) Wavelength versatility** — it can be automatically reconfigured to observe *any* wavelength across 380–900 nm (unlike Fabry-Pérot instruments such as CRISP or TESOS, which are limited by dedicated pre-filters); **(ii) Three-arm simultaneous observing** — three independent, automatically positioned camera arms observe up to three arbitrary wavelength regions at once (e.g., Ca II H+K, Mg I b, Na I D, Ca II IR triplet); **(iii) High performance** — spectral resolving power $R \gtrsim 180{,}000$, spatial resolution at **2× the DKIST diffraction limit**, polarimetric accuracy of $5 \times 10^{-4} I_{\text{cont}}$, and $10^{-4} I_{\text{cont}}$ sensitivity in 10 s. These are achieved via a polychromatic polarization modulator, dual-beam polarimetry, automatically positioned camera arms (with automatic grating-tilt adjustment), and five selectable slits (0.028″–0.2″). ViSP's science drivers include **Hanle-effect measurement of weak magnetic fields**, **photosphere-to-corona "tomography" via multi-line observations**, and **chromospheric magnetometry using Ca II lines**. It is the **only wavelength-versatile spectro-polarimeter** among DKIST's first-light suite, serving as a discovery instrument for exploring new polarization diagnostics.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

**한국어**
ViSP는 **지난 30년간 태양 분광편광측정(spectro-polarimetry) 기술의 집대성**이다. 역사적 흐름:

- **1970–80년대**: 최초의 태양 편광측정기. Stokes I 모드만 일반적, Stokes Q, U, V 측정은 난해.
- **1992년**: Elmore et al.의 **Advanced Stokes Polarimeter (ASP)** at DST — Stokes 벡터의 full 4-component 측정의 이정표.
- **1998년**: Kentischer et al.의 **Triple Etalon Solar Spectrometer (TESOS)** at VTT — Fabry-Pérot 기반 imaging spectro-polarimeter의 시작.
- **2003년**: Scharmer의 **SST** (Swedish Solar Telescope, 1 m) 가동 — 고분해능 ground-based imaging의 르네상스.
- **2006년**: Socas-Navarro et al.의 **SPINOR** at DST — ASP의 후속으로 ViSP와 가장 유사한 기기 (multi-line, echelle 기반). **ViSP의 직접적 선조**.
- **2008년**: Scharmer et al.의 **CRISP** at SST — dual Fabry-Pérot, imaging spectro-polarimeter의 golden standard. **ViSP 논문은 본문에서 CRISP와 빈번 비교**.
- **2012년**: Collados et al.의 **GRIS** (GREGOR Infrared Spectrograph) — IR 대응.
- **2013년**: Lites et al.의 **Hinode/SP** (우주 관측).
- **2022년 (이 논문)**: ViSP — SPINOR + CRISP의 장점을 결합 + **DKIST 4 m 구경 + 자동 reconfiguration** 차별화.

**핵심 질문**: 왜 Fabry-Pérot (CRISP, TESOS) 대신 **echelle grating**을 선택했는가? **답**: Fabry-Pérot는 좁은 free spectral range 때문에 **pre-filter가 파장마다 필요**하여 wavelength versatility가 제한적. Echelle은 **회절 격자 한 개로 380–900 nm 전체를 커버** 가능.

**English**
ViSP culminates 30 years of solar spectro-polarimetry development:
- 1970–80s: first solar polarimeters, mostly Stokes I only
- 1992: ASP (Elmore et al.) at DST — milestone for full 4-Stokes measurement
- 1998: TESOS (Kentischer et al.) at VTT — pioneering Fabry-Pérot spectro-polarimeter
- 2006: SPINOR (Socas-Navarro et al.) at DST — multi-line echelle-based, **direct ViSP predecessor**
- 2008: CRISP (Scharmer et al.) at SST — dual Fabry-Pérot, the imaging spectro-polarimeter gold standard
- 2022: **ViSP** — combines SPINOR's multi-line echelle approach with CRISP-era precision, on a 4 m aperture, with automated reconfiguration

Why echelle instead of Fabry-Pérot (CRISP-like)? Fabry-Pérot requires dedicated pre-filters for each wavelength due to narrow free spectral range, limiting wavelength versatility. An echelle grating covers 380–900 nm with a single dispersive element.

### 타임라인 / Timeline

```
1897  Zeeman       Zeeman effect 발견
1924  Hanle        Hanle effect 발견
1992  Elmore       ASP at DST — modern 4-Stokes 측정의 원조
1997  Stenflo      Second solar spectrum 개념 정립
1998  Kentischer   TESOS (Fabry-Pérot) 시작
2003  Scharmer     SST 가동 (1 m aperture)
2005  Rimmele/SWG  ATST Science Requirements Document
2006  Socas-Navarro SPINOR — ViSP의 직접 선조
2008  Scharmer     CRISP (dual Fabry-Pérot 완성형)
2010  Nelson       ViSP 설계 시작 (HAO led)
2012  Collados     GRIS at GREGOR (IR counterpart)
2014  Casini/Nelson Grating finesse theory (Eq 1)
2019.12 DKIST first light
2020  Rimmele      DKIST Observatory Overview paper
 ▼
═══════════════════════════════════════════════════════
2022: ViSP paper (de Wijn et al., 이 논문) + first-light data
═══════════════════════════════════════════════════════
 ▼
2022+ Science verification → Operations Commissioning Phase (OCP)
2024  ViSP 본격 과학 운영
```

---

## 3. 필요한 배경 지식 / Prerequisites

### 선수 논문 / Prerequisite Papers
- **#23 Rimmele et al. 2020** (DKIST Overview) — 필수. ViSP가 DKIST 시스템 내에서 어떤 위치·역할을 갖는지 이해 전제.
- **#22 Scharmer et al. 2008** (CRISP) — ViSP가 비교 대상으로 반복 언급. Fabry-Pérot 기반 imaging spectro-polarimeter의 한계 이해에 도움.
- **#20 Rimmele & Marino 2011** (Solar AO) — AO-corrected beam을 ViSP가 받음.

### 필수 개념 / Essential Concepts

**한국어**

1. **Stokes vector $(I, Q, U, V)$ / 스토크스 벡터**
   - 빛의 편광 상태를 완전히 기술하는 4-component 벡터.
   - $I$: 총 intensity
   - $Q$: 수평 대 수직 선편광 차이
   - $U$: ±45° 선편광 차이
   - $V$: 원편광 (left-right circular의 차이)
   - 단위: 모두 intensity 단위, 정규화는 $I_{\text{cont}}$(continuum) 기준 사용.

2. **Zeeman effect (자기장 → 스펙트럼선 분열) / 제이만 효과**
   - $\Delta\lambda_B \propto g_{\text{eff}} \lambda^2 B$
   - **Longitudinal** (LOS 방향 자기장): $\sigma^\pm$ 성분이 **원편광** → **Stokes V**로 측정
   - **Transverse** (시선 수직 자기장): $\pi$ 성분이 **선편광** → **Stokes Q, U**로 측정
   - Fe I 630.2 nm doublet, Fe I 524.7 & 525.0 nm — 광구 자기장 진단의 workhorse 라인.

3. **Hanle effect (약한 자기장에 의한 선편광 변화) / 한레 효과**
   - 약한 자기장에서 **scattering polarization의 감소/회전**
   - Zeeman 효과로는 검출 불가한 $< 1$ G 수준 field 감지 가능
   - 필요 조건: **atomic sub-level alignment**. 이는 **anisotropic radiation** (태양 limb 근처에서 발생)으로 생성
   - "Second solar spectrum" (Stenflo 1997): 태양 limb 근처에서 산란 편광 스펙트럼의 풍부한 선들.

4. **Slit-scanning echelle spectrograph / 슬릿 스캔 에셸 분광기**
   - **Slit**: 1-dimensional "aperture" (ViSP: 10–20 μm 너비, 수 cm 길이)
   - **Echelle grating**: **높은 차수(high-order)** 회절 격자로 고분해능 달성. ViSP: $316\,\ell/\text{mm}$, blaze 63.4°, $m \le 15$.
   - **Scanning**: slit을 이동시켜 2D 영역 커버. ViSP: **Aerotech ANT180-260-L translation stage**로 slit scan.

5. **Polarimetry modulation / 편광 변조**
   - **Modulator** (retarder): 입사광의 편광 상태를 시간 순차적으로 회전.
   - ViSP: **polychromatic modulator** — 380–900 nm 전체에 대해 거의 일정한 modulation efficiency 유지.
   - **Demodulation matrix** $\mathbf{O}$: 측정된 intensity sequence를 Stokes 벡터로 변환.

6. **Dual-beam polarimetry / 듀얼 빔 편광 측정**
   - **Polarization analyzer** (beam splitter)가 직교 편광을 두 빔으로 분리 → 두 detector로 동시 기록.
   - Seeing-induced cross-talk (편광 → intensity 방향 오염) 대부분 **두 빔의 차분**으로 제거.

7. **Free spectral range (FSR) / 자유 분광 영역**
   - Fabry-Pérot interferometer의 인접 차수 사이 분리 파장
   - $\text{FSR} = \lambda^2 / (2nd)$ — d는 간격, n은 매질 index
   - Fabry-Pérot는 FSR이 좁아 pre-filter로 차수 고립 필요 → 파장마다 전용 필터
   - Echelle은 FSR이 훨씬 넓고, order-sorting filter 몇 개로 380–900 nm 커버 가능.

**English**

1. **Stokes vector $(I, Q, U, V)$** — complete description of polarization state.
2. **Zeeman effect** — longitudinal $B$ → circular (Stokes V); transverse $B$ → linear (Q, U). $\Delta\lambda_B \propto g \lambda^2 B$.
3. **Hanle effect** — scattering polarization modification by weak fields; requires atomic sub-level alignment from anisotropic radiation.
4. **Slit-scanning echelle** — narrow slit + high-order grating → high $R$ over wide wavelength range.
5. **Polarimetric modulation** — temporal rotation of polarization followed by intensity measurement and demodulation matrix $\mathbf{O}$.
6. **Dual-beam polarimetry** — orthogonal-polarization split removes seeing-induced cross-talk.
7. **Free spectral range** — width of unambiguous order; Fabry-Pérot FSR is narrow (pre-filter required per wavelength), echelle FSR is wide (order-sorting filters cover 380–900 nm).

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **Spectro-polarimeter** | 스펙트럼과 편광을 동시 측정하는 기기 / Instrument measuring both spectrum and polarization |
| **Slit-scanning** | 슬릿을 공간적으로 이동해 2D map 구성 / Spatial scanning of narrow slit to build 2D maps |
| **Echelle grating** | 고차수·고분산 회절 격자 (blaze 60°+) / High-order, high-dispersion diffraction grating |
| **Blaze angle** $\varphi$ | 격자 면의 기울기 각도, 회절 효율 최대화 / Facet tilt angle maximizing diffraction efficiency |
| **Order** $m$ | 회절 차수, $d\sin\theta = m\lambda$ / Diffraction order number |
| **Free Spectral Range** | 인접 차수 간 파장 구분 범위 / Wavelength range between adjacent orders |
| **Finesse profile** | Grating의 spectral response 함수 / Grating's spectral-response function (sinc² form) |
| **Polychromatic modulator** | 넓은 파장 범위에서 일정한 변조 효율의 retarder / Retarder with nearly constant modulation efficiency over broad wavelength range |
| **Dual-beam polarimetry** | 직교 편광을 두 빔으로 분리해 동시 기록 / Orthogonal-polarization split into two beams, recorded simultaneously |
| **Demodulation matrix** | Intensity 측정값을 Stokes 벡터로 변환하는 행렬 / Matrix that converts intensity measurements to Stokes vector |
| **Schiefspiegler** | 비축 off-axis 반사 망원경 설계 (Schief = 기울어진) / Off-axis reflective telescope design ("tilted" in German) |
| **Littrow configuration** | 입사각 = 회절각인 격자 배치 ($\alpha = -\beta$) / Grating configuration where incident = diffracted angle |
| **Order-sorting filter** | 원치 않는 차수를 제거하는 필터 / Filter blocking unwanted diffraction orders |
| **Context imager** | 분광 관측의 공간 맥락을 제공하는 영상 채널 / Imaging channel providing spatial context for spectroscopy |
| **Fiducial hairlines** | slit에 에칭된 정렬용 기준선 / Reference lines etched onto slit for alignment |

---

## 5. 수식 미리보기 / Equations Preview

### 5.1 Grating 방정식 / Grating equation

$$
m\lambda = d(\sin\alpha + \sin\beta)
$$

- $\alpha$: 입사각 (입사광과 격자 수직 사이 각도) / incidence angle
- $\beta$: 회절각 / diffraction angle
- $d$: groove spacing / 격자 간격
- $m$: diffraction order / 회절 차수

**한국어**: ViSP의 $d = 1/316\,\text{mm} \approx 3.16\,\mu\text{m}$, $\alpha = -68°$, blaze $\varphi = 63.4°$, $m = 6$–$14$. 차수 6 ($\lambda \approx 800$ nm)에서 차수 14 ($\lambda \approx 380$ nm)까지 커버.

### 5.2 Grating finesse profile (Eq. 1 논문 본문) / 격자 finesse 프로파일

$$
\mathcal{F}(\alpha, \beta) = \operatorname{sinc}^2 \left[ \pi \frac{L}{\lambda}(\sin\beta - \sin\alpha) \right]
$$

- $L$: 조명된 격자의 너비 / illuminated grating width
- **한국어**: 이 함수의 FWHM이 spectral resolution을 결정. $L$이 클수록 샤프한 프로파일 → 높은 resolving power.

### 5.3 Spectral resolving power (Eq. 4 논문) / 분광 분해능

Littrow 구성 ($\beta = -\alpha = \varphi$)에서:
$$
R \approx \frac{L}{\lambda} \sin\varphi = \frac{w_C}{\lambda}\tan\varphi
$$

- $w_C$: 격자 면의 collimator-정렬 투영 너비 / projected grating width
- **한국어**: ViSP는 $w_C \approx 10$ cm, $\varphi \approx 60°$ → $R \approx L/\lambda \cdot \sin 60° \approx 180{,}000$ at 500 nm.
- **English**: $w_C \approx 10$ cm, $\varphi \approx 60°$ yields $R \approx 180{,}000$ at 500 nm, consistent with $R \gtrsim 180{,}000$ spec.

### 5.4 Zeeman splitting / 제이만 분열

$$
\Delta\lambda_B = \frac{e}{4\pi m_e c}\, g_{\text{eff}}\, \lambda^2 B
$$

**한국어**: ViSP 파장 범위(380–900 nm)에서 가장 자주 사용되는 라인:
- Fe I 630.2 nm doublet, $g_{\text{eff}} = 2.5$
- Fe I 524.7 & 525.0 nm, $g = 3$ & 1.5
- Ca II 393.4 & 396.8 nm (H & K, chromosphere)
- Mg I b1–b3 (~ 517 nm, chromosphere)
- Na I D doublet 589 nm
- Ca II IR triplet 849.8, 854.2, 866.2 nm (chromosphere)

### 5.5 편광 측정 / Polarization measurement

기본 모델:
$$
\vec{I}_{\text{meas}}(t) = \mathbf{O}(t) \cdot \vec{S} + \vec{n}
$$

- $\vec{I}_{\text{meas}}(t)$: time-sequential intensity measurements / 시간 순차 intensity 측정값
- $\mathbf{O}(t)$: modulation matrix at time $t$ / $t$ 시점 모듈레이션 행렬
- $\vec{S} = (I, Q, U, V)^T$: Stokes 벡터
- $\vec{n}$: noise

**Demodulation**:
$$
\vec{S} = \mathbf{O}^{-1} \cdot \vec{I}_{\text{meas}}
$$

**Polarimetric efficiency**: modulator 설계의 핵심 지표. 이상적인 balanced modulator는 $\epsilon_Q = \epsilon_U = \epsilon_V = 1/\sqrt{3} \approx 0.577$. ViSP의 polychromatic modulator는 380–900 nm 전체에서 이에 근접하는 효율 유지.

### 5.6 Light-level requirement / 광량 요건

Polarimetric sensitivity $\sigma_{\text{pol}} = 10^{-4}$ (in 10 s):
$$
N \gtrsim \frac{1}{\sigma_{\text{pol}}^2} = 10^8 \text{ photons per resolution element per modulation state}
$$

이는 4 m DKIST collecting area, $\sim 50\%$ throughput 등을 합쳐 실현 가능.

---

## 6. 읽기 가이드 / Reading Guide

### 한국어

ViSP 논문은 **30 페이지** 규모로 overview 논문보다 깊이 있는 기기 세부를 다룬다. 3단계 독해 전략:

**1단계 — 과학 동기와 요구사항 (§1–§3, 약 30분)**
- §1 Introduction: ViSP의 positioning (wavelength-versatile, 3-arm)
- §2 Science Objectives: Zeeman + Hanle + A-O 효과 기반 과학 사용 케이스
- §3 Requirements: **Table 1** 필수 숙지 — 모든 주요 사양이 여기 정리 (wavelength range, $R$, FOV, polarimetric sensitivity/accuracy)
- **목표**: "ViSP가 왜, 무엇을 하려는지" 이해

**2단계 — 광학 설계 (§4, 약 1시간)**
- §4.1 Feed Optics: Schiefspiegler 3-mirror, 5000:1 aspect slit, 5 slit widths
- §4.2 Spectrograph: **여기가 핵심 수학 부분**
  - §4.2.1 Grating: **Eq. 1–6 차근차근 따라가기**. ViSP 설계는 $d$, $\varphi$, $\alpha$ 선택 로직이 Eq. 1–4에서 유도됨.
  - §4.2.2 Collimator
  - §4.2.3 Camera arms (3-arm 자동 재구성)
- §4.3 Polarimetry: polychromatic modulator + polarization analyzer + slit-optic 위치
- §4.4 Calibration optics
- **수식을 꼼꼼히**: finesse profile, resolving power 공식은 **직접 대입해 숫자 얻어볼 것**
- **Figure 확인**: Fig. 1 (전체 레이아웃), Fig. 2 (grating 기하), Fig. 3 (grating efficiency)

**3단계 — 성능·캘리브레이션·초기 결과 (§5–§8, 약 30분)**
- §5 Mechanical structure
- §6 Control software & data handling
- §7 Calibration approach (DKIST 전체 polarimetric calibration 구조와 연결)
- §8 First-light & commissioning data — **실제 스펙트럼 확인**

### 추천 독해 순서 요약

1. Abstract + Table 1 (15분): 핵심 사양 암기
2. §1 + §2 (20분): 과학 동기
3. §4.2.1 Grating (30분) + Eq 1–6 유도 (수식 노트 필수)
4. §4.1 + §4.3 (30분): feed optics + polarimetry
5. §4.2.2–§4.2.3 (20분): collimator + camera arms
6. §8 first-light (15분): 실제 데이터

**주의할 점** / **Watch out for**:
- Grating geometry의 angle convention ($\alpha, \beta, \varphi, \delta$)이 Fig. 2에 정의됨 — **부호(sign)에 주의**.
- $\alpha$ 는 본문에서 **negative** (논문의 convention: $\alpha = -68°$). 다른 spectroscopy 교재와 부호 다를 수 있음.
- "Littrow configuration"은 $\beta = -\alpha$ 특수 경우이지, ViSP는 실제 Littrow는 아님.

### English

The ViSP paper (~30 pages) goes deeper than an overview. Three-pass strategy:

1. **Science motivation & requirements (§1–§3)** — especially **Table 1** for all key specs.
2. **Optical design (§4)** — core math in §4.2.1 (grating, Eq. 1–6); work through numerically.
3. **Performance & first-light (§5–§8)** — verify specs against actual commissioning data.

Watch the sign conventions in grating geometry (Fig. 2) — $\alpha$ is negative in ViSP's convention. Littrow ($\beta = -\alpha$) is used for analysis, not as actual operating point.

---

## 7. 현대적 의의 / Modern Significance

### 한국어

ViSP는 현대 태양 분광편광측정의 **최첨단 기준점**으로 여러 의의가 있다.

1. **Wavelength versatility의 패러다임 전환** — CRISP 시대의 "전용 pre-filter per line" 모델에서, **자동화된 reconfigurable echelle** 시대로 이동. Research instrument로 새로운 진단법 탐색 가능.

2. **3-line simultaneous 관측의 과학적 힘** — 예: Ca II H+K (photosphere-chromosphere upper), Mg I b (mid-chromosphere), Na I D (mid-chromosphere upper)를 **동시에 기록**하면 하나의 atmospheric column에 대한 **높이 분해(height-resolved) 자기장 + 동역학 측정**이 가능. 기존에는 시간차 관측으로 synchronization 불가능.

3. **Hanle diagnostic의 real-world deployment** — Hanle은 40년간 이론적 알려졌으나 실관측은 희귀. ViSP의 $5 \times 10^{-4}$ accuracy + DKIST photon budget으로 **일상적 Hanle 관측**이 가능.

4. **DKIST Critical Science Plan의 핵심 기기** — Rast et al. 2021의 CSP에서 ViSP는 plage/active region, flare precursors, filament/prominence 동역학 연구의 주요 데이터원.

5. **Space-ground 협업 플랫폼** — Parker Solar Probe, Solar Orbiter의 재연결(reconnection) 이벤트 관측 시 ViSP가 ground-based 동시 관측 제공.

6. **EST의 기술적 기반** — 유럽의 4 m 태양 망원경 EST가 ViSP와 유사한 wavelength-versatile 기기를 계획 중이며, ViSP는 그 설계의 benchmark.

### English

1. **Paradigm shift to wavelength versatility** — moving from CRISP-era "dedicated pre-filter per line" to automated reconfigurable echelle, enabling exploration of new diagnostics.
2. **Three-line simultaneous science** — e.g., Ca II H+K + Mg I b + Na I D gives height-resolved magnetic and dynamical information from a single atmospheric column.
3. **Routine Hanle diagnostics** — with $5\times 10^{-4}$ accuracy and DKIST photon budget, Hanle moves from theory to everyday observing.
4. **Central to DKIST CSP** — primary data source for plage, active regions, flare precursors, and filament studies.
5. **Space-ground coordination** — ViSP provides ground-based simultaneous observations for Parker Solar Probe and Solar Orbiter.
6. **Technical benchmark for EST** — Europe's 4 m solar telescope is planning similar wavelength-versatile instrumentation.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
