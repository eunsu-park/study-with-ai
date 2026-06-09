---
title: "Pre-Reading Briefing: Metis — the Solar Orbiter visible light and ultraviolet coronal imager"
paper_id: "57_antonucci_2020"
topic: Solar_Observation
date: 2026-04-25
type: briefing
---

# Metis: the Solar Orbiter VL/UV Coronal Imager — Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Antonucci, E. et al., "Metis: the Solar Orbiter visible light and ultraviolet coronal imager", *A&A*, 642, A10 (2020). DOI: 10.1051/0004-6361/201935338
**Author(s)**: E. Antonucci, M. Romoli, V. Andretta, S. Fineschi, P. Heinzel, J. D. Moses, et al. (Solar Orbiter Special Issue, 80+ co-authors)
**Year**: 2020

---

## 1. 핵심 기여 / Core Contribution

**한국어**: Metis는 ESA-NASA Solar Orbiter 임무에 탑재된 최초의 가시광(Visible Light, VL) 및 자외선(Ultraviolet, UV) 동시 관측 코로나그래프(coronagraph)이다. 580–640 nm의 광대역 편광 가시광과 121.6 ± 10 nm의 H I Lyman-α 협대역 UV를 단일 광학계에서 동시에 영상화하여, 근일점(0.28 AU) 시 1.7–3.1 R☉, 원일점 시 9 R☉까지의 외부 코로나(off-limb corona)를 전례 없는 시·공간 분해능과 황도면 외(out-of-ecliptic) 시점에서 진단한다. 핵심 혁신은 "역전된 외부 차폐(Inverted External Occulter, IEO)" 설계로, 작은 입구 구경(40 mm)이 외부 차폐 역할을 하면서 태양광 열부하를 두 자릿수 줄여 근일점 가혹환경에서의 동작을 가능케 한다. Metis는 K-corona의 pB(편광 밝기)로부터 톰슨 산란 역해(van de Hulst 1950 inversion)에 의해 전자 밀도를 도출하고, H I Lyman-α의 도플러 디밍(Doppler dimming)으로부터 중성수소(즉 양성자) 유출 속도장을 구해, 태양풍의 가속·CME 발생·에너지 축적 영역을 글로벌하게 매핑한다.

**English**: Metis is the first space-borne coronagraph designed to perform *simultaneous* visible-light (broadband 580–640 nm, polarised) and ultraviolet (narrowband 121.6 ± 10 nm H I Lyman-α) imaging of the off-limb solar corona. Carried on Solar Orbiter (perihelion 0.28 AU), it observes the solar atmosphere from 1.7 R☉ at perihelion out to 9 R☉ near aphelion, with unprecedented temporal coverage, ≤20 arcsec angular resolution, and unique close-in / out-of-ecliptic vantage points. The key innovation is an *inverted external occulter (IEO)* — a small (40 mm diameter) circular entrance aperture acting as the external occulter, which reduces the thermal load by two orders of magnitude relative to a classical externally occulted Lyot design. Electron density is retrieved from polarised brightness (pB) using van de Hulst's Thomson-scattering inversion; neutral-hydrogen (proxy for proton) outflow speed is retrieved by Doppler-dimming of the resonantly scattered Lyman-α line. Together these diagnostics map the corona where the solar wind is accelerated and where CMEs initiate.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

**한국어**: 1995년 발사된 SOHO의 두 코로나그래프(LASCO 가시광, UVCS 자외선 분광계)는 지난 두 태양 주기(Cycle 23–24) 동안 코로나 관측의 표준이 되었다. LASCO는 30 R☉까지의 가시광 영상화를 가능케 했고, UVCS는 H I Lyman-α 도플러 디밍으로 외부 코로나 양성자/수소의 유출 속도를 처음으로 진단했다. 그러나 두 기기는 별개의 광학계로 동시 영상화가 불가능했고, UVCS는 슬릿 분광계로서 시야가 제한적이었다. 동시에, SOHO는 황도면(L1)에 머물러 극지 코로나의 시점이 부족했다. Solar Orbiter(2020년 2월 발사)는 0.28 AU 근일점과 30°+ 황도 경사각의 궤도를 통해 이 두 한계를 동시에 해결하도록 설계되었으며, Metis는 그 임무 중 코로나의 시·공간 통합 영상 진단을 담당한다.

**English**: For the past two solar cycles (23 and 24), SOHO's coronagraphs LASCO (white-light imaging out to 30 R☉) and UVCS (ultraviolet spectroscopy) defined the state of the art. LASCO mapped CME morphology and dynamics in white light; UVCS pioneered Doppler-dimming diagnostics of H I Lyman-α to measure proton/hydrogen outflow velocities. However, the two instruments could not image the same field simultaneously, UVCS was a slit spectrograph with restricted instantaneous FoV, and SOHO remained near the ecliptic at L1, never seeing the corona from out-of-ecliptic vantage. Solar Orbiter (launched February 2020) was conceived to overcome both limitations: a perihelion of 0.28 AU and a final orbit inclination > 30° give close-up and high-latitude views. Metis is its dedicated coronagraph, designed from inception to combine LASCO-style polarised VL imaging with UVCS-style UV Lyman-α imaging in a single, simultaneous, externally occulted instrument.

### 타임라인 / Timeline

```
1950 ─ van de Hulst pB inversion formalism (electron density from Thomson scattering)
1971 ─ Gabriel: chromospheric Lyman-α resonantly scattered by coronal H I
1980s ─ Withbroe et al., Noci et al.: Doppler-dimming theory for outflow velocity
1995 ─ SOHO launch — LASCO (VL) + UVCS (UV spectroscopy) revolutionise coronal physics
1997 ─ SCORE sounding rocket flights (HERSCHEL): first VL+UV simultaneous test
2013 ─ Fineschi et al.: Metis optical design paper (inverted-occultation concept)
2017 ─ Romoli et al.: detailed inverted external occulter description
2018 ─ Dolei et al.: 2-D H I outflow velocity maps from UVCS data (precursor to Metis)
2020 ─ Solar Orbiter launch (Feb); Metis paper published; first-light June 2020
2025+ ─ out-of-ecliptic phase begins (heliographic latitudes ≥ 30°)
```

---

## 3. 필요한 배경 지식 / Prerequisites

**한국어**:
- **Thomson 산란**: 자유전자에 의한 가시광 산란. 코로나 K-corona의 기원이며, 산란된 빛은 LoS(시선)에 수직인 면으로 편광되므로 pB(polarised brightness)로 전자 밀도를 분리할 수 있다.
- **van de Hulst inversion (1950)**: 구대칭 가정하에 pB(ρ) 측정으로부터 nₑ(r)를 푸는 적분 역해. 다항식 적합으로 변환되는 표준 기법.
- **공명 산란과 도플러 디밍**: 코로나의 중성 H가 색구의 Lyman-α 광자를 공명 산란할 때, 코로나 가스의 유출 속도가 색구 광원과의 상대 도플러 이동을 만들어 흡수 효율이 떨어지는 효과. 80–500 km/s 속도 범위에서 민감.
- **외부 차폐(External occultation)**: 태양 디스크 직접광을 차단하기 위해 광학계 앞에 디스크를 두는 방식. Metis는 이를 "역으로" 작은 구멍으로 전환했다.
- **Lyot stop**: Lyot 코로나그래프의 핵심. 외부 차폐 가장자리에서 회절된 빛을 차단하는 내부 stop.
- **편광 측정**: PMP(편광 변조 패키지)와 LCVR(액정 가변 위상지연기)로 4개 편광각(0°, 45°, 90°, 135°)을 cycling.
- **태양 단위계**: 1 R☉ = 696,000 km. 0.28 AU에서 태양은 시각 직경 ≈ 3.6° (vs 1 AU에서 0.53°).

**English**:
- **Thomson scattering**: free-electron scattering of photospheric light produces the K-corona; scattered light is linearly polarised tangentially (perpendicular to LoS-Sun-centre plane), so pB isolates the electron contribution from F-corona/dust.
- **van de Hulst inversion (1950)**: under spherical symmetry, an Abel-like inversion of pB(ρ) yields nₑ(r); usually fitted with a power-law/polynomial expansion.
- **Resonant scattering & Doppler dimming**: coronal neutral H resonantly scatters chromospheric Lyman-α photons; outflow velocity Doppler-shifts the absorbing profile away from the source spectrum, *dimming* the line. Sensitive to ~80–500 km/s.
- **External occultation**: a disk in front of the optical train blocks direct disk light. Metis inverts this with a *small aperture* (the IEO) acting as the occulter.
- **Lyot stop**: blocks diffraction off the entrance pupil edge inside the telescope; canonical to all classical coronagraphs.
- **Polarimetry**: a polarisation modulation package (PMP) with two LCVRs cycles four polarisation angles to extract Stokes parameters and pB.
- **Solar units**: 1 R☉ = 696,000 km; at 0.28 AU the Sun subtends ~3.6° (vs 0.53° at 1 AU).

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **IEO (Inverted External Occulter) / 역전 외부 차폐** | 40 mm 직경의 입구 원형 구경으로, 외부 차폐 역할을 한다. Sun-pointing 시 태양 디스크 광은 입구를 통과하지만 그 직후 M0 거울에서 후방 반사된다. Heat load를 100배 줄임. |
| **M0 sun-light rejection mirror / 태양광 반사 거울** | 71 mm 구면 거울로 입구로 들어온 태양 디스크 광을 IEO를 통해 우주로 되돌려 보낸다. |
| **IO (Internal Occulter) / 내부 차폐** | M1의 공역면(conjugate plane)에 위치한 5 mm 디스크 stop. IEO 내부 가장자리의 회절상을 가린다. |
| **LS (Lyot Stop) / Lyot 차폐** | M0 가장자리에서 회절된 빛을 차단한다. 고전 Lyot 코로나그래프의 핵심 요소. |
| **Polarised brightness (pB) / 편광 밝기** | tangential 편광과 radial 편광의 차이. K-corona의 톰슨 산란 신호를 분리. |
| **Total brightness (tB) / 전체 밝기** | Stokes I (전체 밝기). pB + F-corona/stray light 포함. |
| **Doppler dimming / 도플러 디밍** | 흡수 분자(coronal H)가 광원(chromosphere)에 대해 운동할 때 공명 산란이 약화되는 현상. 유출 속도 진단 도구. |
| **Lyman-α (121.6 nm) / 라이먼-α** | 수소의 가장 강한 UV 공명선. Metis UV 채널이 ±10 nm 대역으로 영상화. |
| **VLDA / UVDA** | Visible Light Detector Assembly (CMOS APS 2k×2k, 10 µm) / Ultraviolet Detector Assembly (intensified CMOS, MCP+KBr photocathode). |
| **PMP / Polarisation Modulation Package** | LCVR 두 개 + QWP + LP로 구성된 편광 변조 모듈. 전압 4단계로 4개 retardance 상태(λ/4, λ/2, 3λ/4, λ). |
| **MOU / MPPU / CPC / HVU** | Metis Optical Unit / Processing & Power Unit / Camera Power Converter / High Voltage Unit. |
| **van de Hulst inversion** | pB(ρ) → nₑ(r) 역해 알고리즘 (1950). 다항식 적합 후 해석적 변환. |

---

## 5. 수식 미리보기 / Equations Preview

**(1) Polarised brightness from Thomson scattering / 톰슨 산란에 의한 편광 밝기**:
$$pB(\rho) = 2 \, \frac{B_\odot}{1 - u/3} \, \frac{3\sigma_T}{16\pi} \int_\rho^\infty n_e(r) \left[(1-u)A(r) + u B(r)\right] \frac{\rho^2}{r^2} \frac{r \, dr}{\sqrt{r^2 - \rho^2}}$$
- ρ: impact parameter (LoS와 태양 중심의 수직거리, "apparent height").
- B☉: mean solar disk brightness; u = 0.63 (limb-darkening); σ_T = Thomson cross-section.
- A(r), B(r): geometric factors (Minnaert 1930) — depend only on γ where sin γ = R☉/r.

**(2) van de Hulst inversion / van de Hulst 역해**:
$$pB(\rho) = \sum_i c_i \left(\frac{\rho}{R_\odot}\right)^{-d_i}, \qquad n_e(r) = \frac{\sum_i a_i (r/R_\odot)^{-b_i}}{(1-u)A(r) + u B(r)}$$
계수 cᵢ, dᵢ는 χ² 최소화로 측정 pB에 ±5%로 적합. 이후 aᵢ, bᵢ로 변환하여 nₑ를 닫힌 형태로 도출.

**(3) Resonantly scattered Lyman-α intensity / 공명 산란 Lyman-α 세기**:
$$I_r = \frac{1}{4\pi} b h \lambda_0 B_{12} \int_{LoS} \int_\Omega p(\varphi) \, d\omega \, \Phi(\delta\lambda) \, n_i \, dl$$
- b: branching ratio (=1 for Lyman-α); B₁₂: Einstein absorption coefficient.
- p(φ): scattering geometry; Ω: chromospheric source solid angle.
- Φ(δλ): Doppler-dimming integral (depends on outflow speed w).

**(4) Doppler-dimming integral**:
$$\Phi(\delta\lambda) = \int_0^\infty I_{ex}(\lambda - \delta\lambda) \, \Psi(\lambda - \lambda_0) \, d\lambda, \qquad \delta\lambda = \frac{\lambda_0}{c}\, \mathbf{w}\cdot\mathbf{n}$$
- I_ex: chromospheric exciting profile.
- Ψ: coronal absorption profile, Gaussian with σ_λ = (λ₀/c)√(k_B T_{k,n}/m_p).

**(5) Field of view scaling with heliodistance / 일심거리에 따른 시야 스케일링**:
- 1.6° inner FoV: 1.7 R☉ at 0.28 AU; 4.2 R☉ at 0.7 AU.
- 2.9° outer FoV: 3.1 R☉ at 0.28 AU; 7.6 R☉ at 0.7 AU.
- 3.4° corner: 3.6 R☉ at 0.28 AU; 9.0 R☉ at 0.7 AU.

---

## 6. 읽기 가이드 / Reading Guide

**한국어**: 41쪽의 긴 instrument paper이므로 다음 순서로 읽는 것을 권장한다.
1. **§1–2 (목표)**: Solar Orbiter의 4대 과학 질문(태양풍 기원, CME, SEP, 다이나모)과 Metis 기여를 큰 그림으로 파악.
2. **§3–4 (광학)**: Inverted External Occulter 개념과 광로(boom→IEO→M0→M1→M2→IF→VL/UV). Fig. 5의 광학 ray trace가 핵심.
3. **§4.8 (편광계)**: PMP의 LCVR 4단계 변조와 Senarmont 구성. pB가 어떻게 측정되는지 이해.
4. **§7–8 (열·교정)**: 빠르게 훑되 Table 3(stray light 1e-9), Table 12(coating)는 확인.
5. **§9 (데이터)**: VL-pB / VL-tB / UV-PC 모드 및 압축 알고리즘. CCSDS-123 + radialisation 트릭이 흥미로움.
6. **§11 (진단)**: ★최고 우선순위. (i) van de Hulst pB 역해 → nₑ, (ii) 도플러 디밍 → 유출 속도, (iii) 3D CME 재구성, (iv) 요동/난류 진단. 본 노트북은 (i),(ii)에 집중.
7. **부록·참고문헌**: Doppler-dimming 이론은 Noci, Kohl & Withbroe (1987) 원논문 참조 권장.

**English**: This is a 41-page instrument paper; suggested reading order:
1. **§1–2 (objectives)**: Solar Orbiter's four top-level questions and Metis's contributions — get the big picture.
2. **§3–4 (optics)**: the IEO concept and full light path (boom → IEO → M0 → M1 → M2 → interference filter → VL/UV detectors). Figure 5 (ray trace) is the key visual.
3. **§4.8 (polarimeter)**: PMP with two LCVRs in Senarmont configuration; how pB is measured.
4. **§7–8 (thermal/calibration)**: skim, but check Table 3 (stray light < 10⁻⁹) and Table 12 (mirror coating).
5. **§9 (data handling)**: VL-pB / VL-tB / UV-PC schemes; CCSDS-123 + radialisation compression is clever.
6. **§11 (diagnostics)**: ★top priority. (i) van de Hulst pB inversion → n_e, (ii) Doppler dimming → outflow velocity, (iii) CME 3-D reconstruction, (iv) fluctuation/turbulence. The implementation notebook focuses on (i)+(ii).
7. **References**: for Doppler-dimming theory, read Noci, Kohl & Withbroe (1987); for inversion, van de Hulst (1950).

---

## 7. 현대적 의의 / Modern Significance

**한국어**: Metis는 SOHO/UVCS와 LASCO의 통합 후예로서, 도플러 디밍과 톰슨 산란 진단을 한 시야에서 동시에 적용한 최초의 우주 코로나그래프이다. 0.28 AU 근일점과 30°+ 궤도 경사라는 Solar Orbiter의 고유 능력 덕분에, 적도면에 머물던 SOHO 시대의 데이터를 보완하는 코로나 극지(coronal hole) 측면 영상과 황도면 외 시점이 처음으로 가능해졌다. 동일 임무의 EUI(Lyman-α 디스크), STIX(X-선), MAG/SWA/EPD/RPW(in-situ)와의 결합, 그리고 Parker Solar Probe와의 상호 관측은 코로나 가속·CME·SEP을 root에서 1 AU까지 끊김 없이 추적한다. ASO-S(중국, 2022), Aditya-L1(인도, 2024), Proba-3(ESA, 2021–) 등 차세대 코로나그래프 네트워크의 기준점 역할을 하며, 도플러 디밍과 pB 진단 코드는 PUNCH(NASA, 2025)와 같은 다중 시점 코로나 임무로 직접 이어진다.

**English**: Metis is the integrated successor to SOHO/UVCS and LASCO — the first space coronagraph to apply Thomson-scattering *and* Doppler-dimming diagnostics simultaneously over the same FoV. Combined with Solar Orbiter's unique 0.28 AU close-in and out-of-ecliptic vantage (orbit inclination eventually > 30°), Metis enables side-views of polar coronal holes and views of the heliospheric current sheet from above the ecliptic, both impossible from SOHO. Combined with EUI (Lyman-α disk), STIX (X-rays), and the in-situ suite (MAG, SWA, EPD, RPW), and with joint operations with Parker Solar Probe, Metis closes the observational gap between the photospheric source and 1 AU. It also serves as the reference design and analysis benchmark for the next-generation coronagraph network (ASO-S 2022, Aditya-L1 2024, Proba-3 2021–) and informs missions such as PUNCH (NASA 2025) for polarimetric, multi-vantage coronal imaging.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
