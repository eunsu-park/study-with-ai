---
title: "Pre-Reading Briefing: The Soft X-ray Telescope for the Solar-A Mission"
paper_id: "38_tsuneta_1991"
topic: Solar_Observation
date: 2026-04-25
type: briefing
---

# The Soft X-ray Telescope for the Solar-A Mission: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Tsuneta, S., Acton, L., Bruner, M., Lemen, J., Brown, W., Caravalho, R., Catura, R., Freeland, S., Jurcevich, B., Morrison, M., Ogawara, Y., Hirayama, T., and Owens, J., "The Soft X-ray Telescope for the Solar-A Mission", *Solar Physics* **136**, 37–67, 1991. DOI: 10.1007/BF00151694
**Authors**: Tsuneta et al. (13 authors, U. Tokyo / Lockheed Palo Alto / ISAS / NAOJ / MSFC)
**Year**: 1991

---

## 1. 핵심 기여 / Core Contribution

이 논문은 일본 SOLAR-A(발사 후 Yohkoh) 위성에 탑재된 Soft X-ray Telescope (SXT) 기기의 설계, 광학 성능, 검출기, 필터, 운영 모드 전반을 종합적으로 기술하는 instrument paper이다. SXT는 grazing-incidence Wolter-I 변형 광학(이중 hyperboloid)에 1024×1024 virtual-phase CCD를 결합하여, 0.25–4.0 keV 영역에서 고시간(2초)·고공간(≤3 arcsec)·전(全) 디스크 X-선 영상을 처음으로 장기 연속 제공한 기기이며, Skylab 이래 최초의 본격적인 태양 X-선 영상 망원경이다.
This paper is the instrument description for the Soft X-ray Telescope (SXT) on the Japanese SOLAR-A (renamed Yohkoh) mission. SXT couples a grazing-incidence Wolter-I variant optic (twin hyperboloids) with a 1024×1024 virtual-phase CCD to deliver, for the first time, sustained 2 s cadence, ≤3 arcsec resolution, full-disk soft X-ray imaging (0.25–4.0 keV). It is the first major solar X-ray imaging telescope since Skylab and defines the technical and scientific framework adopted by all subsequent missions (TRACE, Hinode/XRT, SDO).

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

1970년대 Skylab의 S-054 X-ray telescope가 최초의 본격적 X-선 태양 영상을 제공한 이후, 1980년대는 Solar Maximum Mission (SMM, 1980)의 hard X-ray 분야가 주를 이루었으나 imaging 기능이 결여되어 있었다. 1991년 SOLAR-A는 hard X-ray (HXT), soft X-ray (SXT), Bragg crystal (BCS), wide-band spectrometer (WBS)를 탑재하여 'flare 전 영역을 동시에 영상화'하는 야심찬 멀티-기기 미션으로 발사되었고, SXT는 그 핵심 영상 기기였다.
After Skylab's S-054 X-ray telescope (1973–74) gave the first true solar X-ray images, the 1980s were dominated by the Solar Maximum Mission (SMM, 1980) with strong hard X-ray spectroscopy but essentially no X-ray imaging. SOLAR-A (launched 30 Aug 1991) was an ambitious multi-instrument flare mission carrying a Hard X-Ray Telescope (HXT), Soft X-Ray Telescope (SXT), Bragg Crystal Spectrometer (BCS) and Wide-Band Spectrometer (WBS). SXT was its imaging workhorse and was designed to push beyond Skylab in cadence (2 s vs. tens of seconds), dynamic range (>10⁷), and digital data quality (12-bit CCD vs. film).

### 타임라인 / Timeline

```
1965 ──── Giacconi & Rossi: Wolter X-ray optics theory
1973–74 ── Skylab S-054 (Vaiana et al.): film-based grazing-incidence
1980 ──── SMM HXIS / BCS: hard X-ray, no full-disk imaging
1987 ──── Nariai: hyperboloid–hyperboloid wide-field design
1989 ──── Bruner et al.: SXT engineering paper
1991 ★── Tsuneta et al.: SXT instrument paper (this paper)
1991 ──── SOLAR-A launch → Yohkoh
1998 ──── TRACE (EUV cousin)
2006 ──── Hinode/XRT (Golub et al.) — direct SXT successor
2010 ──── SDO/AIA (EUV continuation of cadence philosophy)
```

---

## 3. 필요한 배경 지식 / Prerequisites

- **Grazing-incidence optics**: X-선이 임계각 이하의 grazing 입사에서만 전반사된다는 사실, Wolter-I (paraboloid + hyperboloid) 및 그 변형들 / Total external reflection of X-rays only at grazing angles below θ_c; Wolter-I and its variants.
- **Plasma X-ray emission**: Optically thin thermal bremsstrahlung + line emission (Mewe, Gronenschild, van den Oord 1985), emission measure $EM = \int n_e^2 \, dV$ / Optically thin thermal continuum + line emission with emission measure.
- **CCD detectors**: Quantum efficiency, dark current, read noise, ADC, full-well capacity, virtual-phase CCD (Hynecek 1979) / Basic CCD physics including virtual-phase architecture.
- **Differential Emission Measure (DEM)**: $\xi(T) = n_e^2 \, dV/d\log T$, ill-posed inversion / DEM as ill-posed inverse problem.
- **Filter ratio temperature diagnostics**: 두 필터의 신호비가 등온 플라즈마의 온도에 단조 의존하는 원리 / Two-filter signal ratio as monotonic temperature proxy for isothermal plasma.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| Grazing incidence | 임계각 이하의 매우 얕은 입사각에서 X-선이 전반사되는 현상 / Total external reflection of X-rays at incidence below the critical angle. |
| Wolter-I optic | Paraboloid + hyperboloid 2회 반사로 좁은 시야 결상을 얻는 X-선 광학 / Two-reflection grazing-incidence design (paraboloid + hyperboloid). |
| Hyperboloid–hyperboloid | Nariai (1987, 1988)의 wide-field SXT 변형 광학 / Wide-field variant used in SXT (twin hyperboloids of revolution). |
| Effective area $A_{\text{eff}}(\lambda)$ | 파장 함수로서의 거울 면적×반사율×필터 투과율×CCD QE 곱 / Wavelength-dependent product of geometric area, reflectivity, filter transmission, CCD QE. |
| PSF $D_{50}$ | 점 광원 에너지의 50%를 둘러싸는 원의 지름 / Diameter of circle enclosing 50% of imaged energy. |
| Filter ratio | 두 필터의 신호비 → 온도 진단 / Ratio of signals through two filters → temperature diagnostic. |
| DEM | $\xi(T)$, plasma의 단위 온도당 emission measure 분포 / Differential emission measure distribution. |
| Virtual-phase CCD (VPCCD) | Texas Instruments의 박막 산화막 구조 CCD, soft-X 응답이 우수 / TI virtual-phase CCD with thin oxide for good soft-X response. |
| Patrol image / FFI / PFI | 자동 ROI 선택용 / 전체 / 부분 프레임 영상 / Patrol (ARS), Full Frame, Partial Frame Images. |
| AEC / ARS / ART | Automatic Exposure Control / OR Selection / OR Tracking — 온보드 자동화 / On-board autonomy systems. |

---

## 5. 수식 미리보기 / Equations Preview

1. **Effective area / 유효 면적**

$$A_{\text{eff}}(\lambda) = A_{\text{geom}} \cdot R^2(\lambda, \theta) \cdot T_{\text{filt}}(\lambda) \cdot Q_{\text{CCD}}(\lambda)$$

기하학적 면적, 두 번 반사이므로 $R^2$, 필터 투과율, CCD 양자 효율의 곱 / Product of geometric area, two-bounce reflectivity, filter transmission, CCD QE.

2. **PSF $D_{50}$ scaling** (Eq. 1):

$$D_{50} = 7.0 - 2.4 \log_{10}\lambda \quad [\text{arcsec}, \, \lambda \text{ in Å}]$$

파장이 길수록 산란 윙이 작아 D_50이 작아지는 경향을 보정 후 상수로 흡수한 경험식 / Empirical fit; longer wavelengths give tighter cores (less roughness scattering).

3. **Modified Moffat PSF** (Eq. 2):

$$N(r) = \frac{C}{[1 + (r/a)^2]^b}$$

중심 첨두와 비-Gaussian wing을 동시에 표현 / Captures sharp central spike and non-Gaussian wings.

4. **DN per pixel from photon flux** (Eq. 5):

$$N = \frac{n h \nu}{3.65 c} + 11.5$$

n=검출 광자 수, $h\nu$=eV 단위 평균 광자 에너지, c=100 e⁻/DN gain, 11.5는 디지털 오프셋 / n photons, $h\nu$ in eV, c=100 e⁻/DN, 11.5 DN digital offset.

5. **8→12 bit decompression** (Eqs. 3, 4):
square-root encoding으로 12-bit 동적 범위를 8 bit로 보존, 이로 인한 압축 오차 (Eq. 8):

$$\varepsilon = \sqrt{\lambda/34} \cdot (N - M)/\sqrt{N - 11.5}$$

square-root LUT으로 photon shot noise보다 작게 압축오차 유지 / Square-root LUT keeps compression error below photon shot noise.

---

## 6. 읽기 가이드 / Reading Guide

- **Section 1 (Intro & Sci. Obj.)**: SXT의 과학 목표 7가지를 빠르게 훑고 넘어가도 됨. flare 외 코로나 가열·X-ray bright point·daily morphology 등 'imaging movie' 패러다임이 핵심 / Skim science objectives; the key idea is the "X-ray movie" paradigm enabling flare and non-flare studies.
- **Section 2 (Optics)**: 광학·필터·CCD·SXT response가 가장 정량적인 부분이다. Table I (parameters), Fig. 2 (effective area), Fig. 4 (PSF), Fig. 8 (CCD QE), Fig. 9 (T-response)을 반드시 정독 / This is the quantitative core. Study Table I, Fig. 2, Fig. 4, Fig. 8, Fig. 9 carefully.
- **Section 3 (Image Data)**: data compression (Eqs. 3–8), 시퀀스 테이블, ARS/ART/AEC 자동화는 미션 운영 측면에서 중요하지만 빠르게 읽어도 무방 / Read for operational concept; details (sequence tables, telemetry rates) can be skimmed.
- 필터 비율(Fig. 10)과 simulated DEM(Fig. 11)는 SXT 과학 분석의 기초이므로 implementation 노트북에서 재현 / Filter ratio plot and simulated DEM are the basis for SXT science — reproduced in the implementation notebook.

---

## 7. 현대적 의의 / Modern Significance

SXT가 정착시킨 "grazing-incidence + soft-X CCD + 자동 노출 + 필터 비율 온도 진단"의 조합은 이후 모든 태양 X-선 영상 기기의 표준이 되었다. Hinode/XRT(2006)는 SXT 광학을 그대로 계승·확장하였고, TRACE/AIA의 EUV 영상도 cadence·자동화 운영 철학을 SXT에서 차용하였다. 본 논문은 instrument paper의 모범 사례로서, 광학, 검출기, 필터, 데이터 시스템, 자동화 운영을 한 권의 서술에 모두 담아내는 방식 자체가 후속 미션의 기준이 되었다.
The four-pillar combination SXT established — grazing-incidence optic, soft-X-sensitive CCD, on-board automatic exposure control, and filter-ratio temperature diagnostics — became the template for every subsequent solar X-ray imager. Hinode/XRT (2006) is the direct technological descendant; TRACE and SDO/AIA inherited SXT's cadence-and-autonomy philosophy in the EUV. The paper itself is a model instrument paper whose structure (optics → detector → filters → response → data system → operations) is now a de-facto standard.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
