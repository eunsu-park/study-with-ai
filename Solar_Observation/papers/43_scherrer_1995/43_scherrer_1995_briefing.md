---
title: "Pre-Reading Briefing: The Solar Oscillations Investigation - Michelson Doppler Imager"
paper_id: "43_scherrer_1995"
topic: Solar_Observation
date: 2026-04-25
type: briefing
---

# The Solar Oscillations Investigation - Michelson Doppler Imager: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Scherrer, P. H., Bogart, R. S., Bush, R. I., Hoeksema, J. T., Kosovichev, A. G., Schou, J., et al. (1995). The Solar Oscillations Investigation - Michelson Doppler Imager. *Solar Physics*, 162, 129-188. DOI: 10.1007/BF00733429
**Authors**: P. H. Scherrer (PI), R. S. Bogart, R. I. Bush, J. T. Hoeksema, A. G. Kosovichev, J. Schou, and the MDI Engineering Team (Stanford / LMSAL)
**Year**: 1995

---

## 1. 핵심 기여 / Core Contribution

이 논문은 SOHO (Solar and Heliospheric Observatory) 위성에 탑재된 MDI (Michelson Doppler Imager) 기기를 이용한 SOI (Solar Oscillations Investigation) 프로그램의 종합적인 설계 문서이다. MDI는 Ni I 6768 Å 흡수선 부근에서 5개의 협대역 (94 mÅ FWHM) 필터그램을 1024×1024 CCD로 1분 간격으로 촬영하여, 전체 태양 원반의 시선 속도 (Doppler velocity), 연속 강도, 종방향 자기장을 4″ 분해능 (full disk) 또는 1.25″ 분해능 (high resolution)으로 산출한다. 본 논문은 과학 목표, 광학 설계 (이중 가변 Michelson 간섭계 + Lyot 필터), 관측 프로그램 (Dynamics 60일 연속, Structure 5 kbps 상시, 8시간 일일 Campaign), 데이터 처리 파이프라인을 포괄적으로 기술한다.

This paper is the comprehensive design document for the Solar Oscillations Investigation (SOI) using the Michelson Doppler Imager (MDI) instrument aboard the SOHO spacecraft. MDI obtains five narrow-band (94 mÅ FWHM) filtergrams near the Ni I 6768 Å absorption line on a 1024×1024 CCD each minute, computing full-disk line-of-sight Doppler velocity, continuum intensity, and longitudinal magnetic field at 4″ resolution (full disk) or 1.25″ (high resolution). The paper covers science objectives, optical design (two tunable Michelson interferometers plus a Lyot filter), observing programs (60-day continuous Dynamics, always-on 5 kbps Structure, 8-hour daily Campaigns), and the data reduction/calibration pipeline.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

1960년 Leighton, Noyes & Simon이 5분 진동을 발견한 이후, Ulrich (1970)와 Leibacher & Stein (1971)은 이들이 광구 아래에 갇힌 음향중력파임을 제안했다. Deubner (1975)는 분산 관계 ($k$-$\omega$ diagram)를 관측으로 확인하였고, 일진동학 (helioseismology)이라는 용어는 1983년부터 널리 쓰이기 시작했다 (Gough). 지상 관측 (BiSON, IRIS, GONG, South Pole 캠페인)은 대기 난류와 주야 데이터 갭의 한계가 있어, NASA의 Bohlin이 1978년 우주 관측 SWG를 조직, Noyes & Rhodes (1984) 보고서가 "L1 헤일로 궤도의 2-D 영상 도플러 측정 임무"를 권고했다. 1987년 ESA/NASA SOHO AO 발표 → 1988년 SOI-MDI 선정 → 1990년 본격 개발 → 1994년 4월 비행기기 인도 → 1995년 12월 발사.

After the discovery of solar 5-minute oscillations (Leighton, Noyes & Simon 1962) and their identification as trapped acoustic-gravity waves (Ulrich 1970; Leibacher & Stein 1971), Deubner (1975) confirmed the dispersion relation observationally. The term "helioseismology" became standard around 1983. Ground-based campaigns (BiSON, IRIS, GONG, South Pole) suffered from atmospheric turbulence and day-night gaps, prompting NASA in 1978 to convene a Science Working Group; the Noyes & Rhodes (1984) report recommended a space-qualified two-dimensional imaging Doppler instrument in a fully-sunlit orbit such as L1. Following the 1987 ESA/NASA SOHO AO, the SOI-MDI proposal (PI: Scherrer) was selected in March 1988, full-scale development began October 1990, and the flight instrument was delivered April 1994.

### 타임라인 / Timeline

```
1960 ─ Leighton 5-minute oscillations discovered / 5분 진동 발견
1962 ─ Leighton, Noyes & Simon publish observations
1970 ─ Ulrich: trapped acoustic mode hypothesis
1971 ─ Leibacher & Stein: independent same hypothesis
1975 ─ Deubner: k-ω dispersion relation confirmed
1978 ─ NASA SWG (Bohlin) for space helioseismology / 우주 일진동학 SWG 발족
1980 ─ Brown: Fourier Tachometer technique / 푸리에 타코미터 기법
1983 ─ "Helioseismology" coined (Gough) / 용어 정착
1984 ─ Noyes & Rhodes report endorses L1 imaging Doppler mission
1987 ─ ESA/NASA SOHO AO released
1988 ─ SOI-MDI proposal selected (Mar) / SOI-MDI 선정
1990 ─ MDI full-scale development begins (Oct)
1994 ─ MDI flight instrument delivery (Apr); SOHO integration
1995 ─ This paper published; SOHO launched (Dec 2) / 본 논문 발표 및 SOHO 발사
1996 ─ MDI begins routine science (May)
2010 ─ HMI (MDI's successor on SDO) launches / HMI 발사
```

---

## 3. 필요한 배경 지식 / Prerequisites

- **Helioseismology basics / 일진동학 기초**: $p$-mode, $g$-mode, $f$-mode; spherical harmonic degree $\ell$, azimuthal order $m$, radial order $n$; mode trapping by sound-speed gradient and density jump; $\ell$-$\nu$ (or $k$-$\omega$) diagnostic diagram
- **Optical interferometry / 광간섭법**: Michelson interferometer with sinusoidal channel spectrum; free spectral range $\Delta\lambda = \lambda^2/(2 n d)$; Lyot birefringent filter (cascade of polarizer-retarder elements)
- **Doppler effect / 도플러 효과**: $\Delta\lambda/\lambda = v/c$ for a 6768 Å line, 1 m/s ≈ 22.6 µÅ shift
- **Stokes / Zeeman effect**: longitudinal Zeeman splitting of Ni I 6768 (Landé $g \approx 1.4$); RCP–LCP differencing for magnetograms
- **CCD physics / CCD 물리**: shot noise, read noise, full well, MPP technology, CTE
- **Spherical harmonic decomposition / 구면조화 분해**: projection of Doppler images onto $Y_\ell^m$ basis
- **Power spectra / 파워 스펙트럼**: $\ell$-$\nu$ diagram, mode ridges, line widths, asymmetry, $a$-coefficients

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **MDI** | Michelson Doppler Imager — SOHO에 탑재된 MDI 기기. Two tunable Michelsons + Lyot filter; 1024² CCD; Ni I 6768 Å. |
| **SOI** | Solar Oscillations Investigation — MDI를 사용하는 과학 프로그램 / scientific program using MDI |
| **Filtergram / 필터그램** | Narrow-band image (94 mÅ FWHM) at one wavelength tuning. 5개 필터그램 ($F_0$–$F_4$)이 75 mÅ 간격으로 촬영됨. |
| **OBSMODE vs CALMODE** | Normal observing (각 픽셀이 태양의 한 점을 봄) vs calibration mode (각 픽셀이 동공 이미지를 봄, 적분된 햇빛). |
| **Lyot filter / Lyot 필터** | Cascade of birefringent retarders giving 465 mÅ FWHM passband near 6767.8 Å. |
| **Free Spectral Range / 자유 스펙트럼 범위** | Period of Michelson sinusoidal transmission: $\Delta\lambda = \lambda^2/(2 n d)$. M1 = 377 mÅ, M2 = 189 mÅ. |
| **$\alpha$ (Doppler ratio)** | Velocity proxy: $\alpha = (F_1+F_2-F_3-F_4)/(F_1-F_3)$ if numerator>0, else $/(F_4-F_2)$. |
| **$\ell$-$\nu$ diagram** | Power spectrum vs spherical harmonic degree and frequency; mode ridges visible. |
| **Dynamics Program / 다이내믹스 프로그램** | 60+ days continuous 160 kbps HRT each year, full-disk and high-res velocity. |
| **Structure Program / 스트럭처 프로그램** | Always-on 5 kbps channel; 20,000 spatial-averaged velocity bins ($\ell\le 250$). |
| **Campaign / 캠페인** | Daily 8-hour HRT intervals for special observations (high-cadence, high-res, etc.). |
| **HRT / LRT** | High-Rate Telemetry (160 kbps) / Low-Rate Telemetry (5 kbps). |
| **ISS** | Image Stabilization System — limb-sensor PZT-driven tilt mirror, ±19″ range, jitter <0.03″. |
| **MTF** | Modulation Transfer Function — 결상 광학계의 공간주파수 응답; FD path defocused 2 steps to suppress aliasing. |

---

## 5. 수식 미리보기 / Equations Preview

**(1) Doppler shift / 도플러 천이**

$$\frac{\Delta\lambda}{\lambda} = \frac{v}{c}, \quad \text{at } 6768\,\text{Å}: \; 1\,\text{m/s} \leftrightarrow 22.6\,\mu\text{Å}$$

**(2) Michelson sinusoidal transmission / 마이켈슨 정현파 투과**

$$T(\lambda) = \frac{1}{2}\left[1 + \cos\!\left(\frac{2\pi(\lambda-\lambda_0)}{\Delta\lambda_{\text{FSR}}}\right)\right]$$

여기서 $\Delta\lambda_{\text{FSR}} = \lambda^2/(2nd)$는 자유 스펙트럼 범위. M1: 377 mÅ, M2: 189 mÅ. / where $\Delta\lambda_{\text{FSR}}$ is the free spectral range.

**(3) MDI velocity proxy / MDI 속도 프록시**

$$\alpha = \begin{cases} (F_1+F_2-F_3-F_4)/(F_1-F_3), & \text{numerator}>0 \\ (F_1+F_2-F_3-F_4)/(F_4-F_2), & \text{numerator}\le 0 \end{cases}$$

This blue-wing minus red-wing ratio is insensitive to wing-slope variations and to linear gain/offset.

**(4) Line depth from filtergrams / 필터그램으로부터의 선 깊이**

$$I_{\text{depth}} = \sqrt{2\,\big[(F_1-F_3)^2 + (F_2-F_4)^2\big]}$$

(discrete Fourier transform interpretation: $I = I_c - I_d \cos(2\pi(\lambda-\lambda_0)/P)$)

**(5) Continuum intensity / 연속 강도**

$$I_c = 2 F_0 + I_{\text{depth}}/2 + I_{\text{ave}}$$

여기서 $I_{\text{ave}}$는 $F_1$–$F_4$의 평균. 도플러 누화가 0.2% 수준에서 상쇄됨. / cancels Doppler crosstalk to 0.2%.

---

## 6. 읽기 가이드 / Reading Guide

- **Sec. 1 (Introduction)**: SOHO mission context와 SOI 채택 과정 (1978 SWG → 1995 launch). 빠르게 통독.
- **Sec. 2 (SOI Science Program)**: 11개 과학 목표 (대류대 동력학, 평균 방사 구조, 내부 회전, 코어, 자기 구조, 여기/감쇠, 대규모 흐름, 자기장, 자기 확산, 림 형상, 방사 플럭스). 각 목표가 어떤 모드를 어떻게 사용하는지 확인.
- **Sec. 3 (Instrument)**: **본 논문의 핵심**. 광학 (3.1), ISS (3.2), 필터 (3.3), CCD (3.4), 컴퓨터 (3.5), 메커니즘 (3.6). Table I (Key Parameters) 와 Fig. 4 (광학 레이아웃)을 외울 것. Lyot 465 mÅ → blocker 8 Å → Michelson 188 mÅ → Michelson 94 mÅ 캐스케이드 이해가 핵심.
- **Sec. 4 (Observables)**: $\alpha$ 공식과 lookup table 보정, 노이즈 표 (Table III). Fig. 12의 속도-α 관계 곡선 주목.
- **Sec. 5 (Observing Programs)**: Dynamics (60-day, 160 kbps), Structure (5 kbps 상시), Campaigns (8-hour HRT), Magnetic. Table IV의 telemetry budget.
- **Sec. 6–7 (Investigations & Data Analysis)**: SSSC 인프라, Level 0–3 처리, 분석 모듈 (구면조화 시계열, 링 다이어그램, 시간-거리, 상관 추적). 빠르게 훑되 이후 paper의 모태가 됨을 인지.

---

## 7. 현대적 의의 / Modern Significance

MDI는 1996–2011년 동안 안정적으로 운영되어 일진동학을 정밀과학으로 끌어올렸다. 주요 성과: (1) 대류대 차등회전과 큰 격동층 (tachocline) 정밀 매핑, (2) 시간-거리 일진동학 (Duvall et al. 1993)으로 흑점 아래 흐름 영상화, (3) 활동 영역 매일 자기장 카탈로그, (4) 진동 주파수의 태양 활동 주기 의존성. MDI의 직접 후속작 SDO/HMI (Schou et al. 2012, 4096² CCD, Fe I 6173 Å)는 본 논문의 광학 개념을 그대로 계승하면서 분해능 (1″ vs 4″), 카덴스 (45 s vs 60 s), 감도를 개선했다. 또한 MDI의 데이터 파이프라인 (SSSC, Level 0–3)은 JSOC (Joint Science Operations Center)와 SDO 데이터 시스템의 청사진이 되었다.

MDI operated reliably from 1996 to 2011, transforming helioseismology into a precision science. Major achievements include: (1) precise mapping of convection-zone differential rotation and the tachocline; (2) time-distance helioseismology (Duvall et al. 1993) imaging subsurface flows beneath sunspots; (3) daily synoptic magnetic field catalogs; and (4) detection of solar-cycle dependence of mode frequencies. MDI's direct successor SDO/HMI (Schou et al. 2012; 4096² CCD, Fe I 6173 Å) inherits the optical concept of this paper while improving resolution (1″ vs 4″), cadence (45 s vs 60 s), and sensitivity. The MDI data pipeline (SSSC, Level 0-3) became the blueprint for JSOC and the SDO data system.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
