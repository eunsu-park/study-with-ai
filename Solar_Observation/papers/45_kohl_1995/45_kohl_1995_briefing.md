---
title: "Pre-Reading Briefing: The Ultraviolet Coronagraph Spectrometer for SOHO"
paper_id: "45_kohl_1995"
topic: Solar_Observation
date: 2026-04-25
type: briefing
---

# The Ultraviolet Coronagraph Spectrometer (UVCS/SOHO): Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Kohl, J. L., Esser, R., Gardner, L. D., Habbal, S., Daigneau, P. S., Dennis, E. F., Nystrom, G. U., Panasyuk, A., Raymond, J. C., Smith, P. L., Strachan, L., Van Ballegooijen, A. A., et al., "The Ultraviolet Coronagraph Spectrometer for the Solar and Heliospheric Observatory", *Solar Physics* **162**, 313–356 (1995). DOI: 10.1007/BF00733433
**Author(s)**: J. L. Kohl et al. (대규모 국제 협력 / large international collaboration: SAO, Florence, ESA, Torino, Padova, Catania, Arcetri, MPI, ESO, ISSI, NB, Maryland, Alenia, OG, Brusag, Oerlikon, Berkeley, Ball, GSFC)
**Year**: 1995

---

## 1. 핵심 기여 / Core Contribution

UVCS는 SOHO 미션에 탑재된 자외선 코로나그래프 분광기로, 외부/내부 차폐(occultation)와 두 개의 토릭(toric) 회절 격자 분광기 그리고 가시광 편광계로 구성되어 있다. 본 논문은 이 장비의 과학 목표(태양풍 가속·코로나 가열 메커니즘 진단), 광학 설계, 분광 진단법(공명 산란/Thomson 산란/Doppler dimming), 검출기, 산란광 억제, 보정 및 운용 모드를 종합적으로 기술하는 미션 정의 논문(mission paper)이다.

UVCS is the ultraviolet coronagraph spectrometer aboard SOHO, comprising three reflecting telescopes with external and internal occultation, two toric grating spectrometer channels (HI Lyα 1216 Å and OVI 1032/1037 Å), plus a visible-light polarimeter (4500–6000 Å). This paper is the comprehensive mission/instrument paper describing the scientific objectives (solar-wind acceleration and coronal heating diagnostics), the spectroscopic techniques (resonant scattering, Thomson scattering, Doppler dimming, line-profile thermometry), the optical/mechanical/electronic design, stray-light suppression, on-ground characterization and on-orbit operations.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

1970년대부터 Skylab/HCO/Spartan 201/UV 로켓 코로나그래프(Kohl et al. 1980, 1984)가 자외선 코로나 분광 관측을 시작했고, Doppler dimming 기법(Hyder & Lites 1970; Withbroe et al. 1982; Kohl & Withbroe 1982; Noci, Kohl & Withbroe 1987)이 확장 코로나의 풍속 측정 가능성을 보여주었다. SOHO(1995년 발사)는 라그랑주 L1 지점에서 연속적 태양 관측을 제공하는 최초의 미션이며, UVCS는 이러한 분광 진단을 12 R⊙까지 확장하여 태양풍 가속·코로나 가열의 결정적 데이터를 제공하기 위해 설계되었다.

Since the 1970s, Skylab, HCO sounding-rocket, and Spartan 201 UV coronagraphs (Kohl et al. 1980, 1984) had pioneered ultraviolet coronal spectroscopy, while the Doppler-dimming technique (Hyder & Lites 1970; Withbroe et al. 1982; Kohl & Withbroe 1982; Noci, Kohl & Withbroe 1987) demonstrated that outflow velocities could be measured in the extended corona. SOHO (launched 1995) is the first mission providing continuous solar viewing from the Sun-Earth L1 Lagrange point, and UVCS was designed to extend these spectroscopic diagnostics out to 12 R⊙, providing critical data on solar-wind acceleration and coronal heating.

### 타임라인 / Timeline

```
1950 ── van de Hulst: K/F corona theory / K/F 코로나 이론
1958 ── Parker: solar wind theory / Parker 태양풍
1965 ── Hughes: e-scattered Lyα electron T_e diagnostic / 전자 산란 Lyα 진단
1970 ── Hyder & Lites: Doppler dimming concept
1970s ─ Skylab + HAO MK3/MK4 / Skylab + HAO 코로나미터
1980 ── Kohl et al. UV rocket coronagraph; first coronal Lyα profile
1982 ── Withbroe et al. coronal hole Doppler dimming
1987 ── Noci, Kohl & Withbroe — OVI doublet diagnostic with C II pumping
1995 ── ★ UVCS instrument paper (this paper) / 본 논문 ★
1995 ── SOHO launch (Dec 2)
1996+── coronal hole proton/ion T anisotropy, fast-wind acceleration
2003+── Cranmer reviews; ion-cyclotron heating debate
```

---

## 3. 필요한 배경 지식 / Prerequisites

- 코로나 물리(K, F, E component)와 Thomson 산란 / Solar-corona physics (K/F/E components) and Thomson scattering
- 공명 산란(resonant scattering)과 광학적으로 얇은(optically thin) 라인 형성 / Resonant scattering and optically thin coronal line formation
- Doppler dimming의 원리: 흐름 속도가 흡수 프로파일을 적색 이동시켜 흡수율을 감소시킴 / Doppler dimming: outflow shifts incoming chromospheric line out of the resonant absorption profile
- 격자 분광기와 Rowland circle 기하학, toric grating의 stigmatic imaging / Grating spectrometer and Rowland-circle geometry; stigmatic imaging with toric gratings
- 차폐 코로나그래프 광학(외부/내부 차폐, sunlight trap, baffles) / Occulted-coronagraph optics (external/internal occulters, sunlight trap, baffles)
- Microchannel plate (MCP) + cross delay line (XDL) 광자 계수 검출기 / Photon-counting MCP+XDL detectors
- 분광선 프로파일과 입자 속도 분포(Maxwellian, kinetic temperature, non-thermal motions)
- 등이온 평형(ionization balance)과 분광 진단 — 전자/양성자 온도, 밀도, 흐름 속도, 화학적 풍부도

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| External / Internal occulter | 외부 차폐(태양 디스크 차단)와 내부 차폐(거울 모서리에서 회절된 빛 차단) / Knife-edge external occulter blocks the solar disk; internal occulter blocks light specularly reflected from diffraction at the mirror edge. |
| Sunlight trap | 망원경 거울에 의해 반사된 태양 디스크 광을 다중 반사로 흡수하는 캐비티 / Cavity with multilayer black coating that absorbs disk light via specular bouncing. |
| Doppler dimming | 코로나 이온이 흐를 때 입사 색구권 광이 도플러 이동되어 흡수가 감소, 결국 공명 산란 강도가 흐름 속도와 함께 감쇠하는 현상 / Reduction of resonantly scattered intensity caused by the Doppler shift of the incoming chromospheric line out of the absorption profile. |
| Doppler pumping (C II → O VI 1037 Å) | C II 1037.018 Å 라인이 90 km/s 이상의 흐름에서 O VI 1037.613 Å 흡수 프로파일로 적색 이동되어 강도를 다시 증가시키는 현상 / At outflow speeds ≥ 90 km/s, C II 1037.018 Å is Doppler-shifted onto the O VI 1037.613 Å absorber, re-pumping the line. |
| Toric grating | 분산 방향과 공간 방향 곡률 반경이 다른(R_v < R_h) 격자로 stigmatic imaging 가능 / Grating with two different curvature radii (R_v ≠ R_h) producing stigmatic spectroscopic imaging at ±β₀. |
| Rowland circle | 직경이 R_h인 원으로, 분광 초점이 형성되는 자취 / Circle of diameter R_h on which the spectral (horizontal) focus lies. |
| XDL detector | 마이크로채널 플레이트 + 크로스 지연선 양극의 광자 계수 2D 검출기(1024×360 픽셀, 25 µm) / KBr-coated MCP Z-stack + cross delay-line readout, 2D photon-counting (1024×360 px, ~25 µm). |
| K, F coronae | K 코로나(Thomson 산란), F 코로나(먼지 산란, 거의 비편광) / K-corona = Thomson scattered electron component (polarized); F-corona = dust-scattered (~unpolarized). |
| HI Lyα profile | 코로나 양성자/수소 운동 분포의 직접 진단(thermal+nonthermal+bulk) / Direct diagnostic of coronal proton/hydrogen kinetic distribution (thermal + nonthermal + bulk). |
| O VI doublet (1032/1037 Å) | 흐름 속도와 이온 운동 진단(특히 1037 Å는 C II 펌핑으로 90–250 km/s에 민감) / Diagnostic of ion outflow and kinetic distribution; 1037 Å sensitive to outflow 90–250 km/s via C II pumping. |
| Stigmatic point ±β₀ | toric grating의 분산 방향 ±β₀에서 spatial+spectral 초점이 동시에 일치하는 점 / Diffraction angles where vertical and horizontal foci coincide. |

---

## 5. 수식 미리보기 / Equations Preview

**Eq. (1) — Electron-scattered HI Lyα profile (Hughes 1965 / Thomson on chromospheric Lyα)**:

$$ I_e(\lambda) = \mathrm{const} \int_{-\infty}^{\infty} N_e \exp\!\left[-\frac{(\lambda-\lambda_0)^2}{\Delta\lambda_e^2}\right] dx $$

전자 온도(T_e)에서 ~7000 km/s의 열운동에 의해 ~50 Å 폭의 광폭 라인이 형성됨; 공명 산란 라인(~1 Å)을 빼서 측정. / Electron-scattered Lyα has FWHM ~50 Å (electron speed ~7000 km/s at 1.5 MK) — much wider than resonant Lyα (~1 Å).

**Eq. (2) — Doppler-dimming intensity ratio (resonant vs electron-scattered visible)**:

$$ \frac{I_r}{I_{WL}} = \mathrm{const} \times A_\mathrm{el} \langle R_i \rangle \langle D_i(V_W) \rangle $$

A_el = 원소 풍부도, ⟨R_i⟩ = 이온 분율(ionization balance), ⟨D_i(V_W)⟩ = Doppler dimming 인자(흐름 속도의 함수). / Ratio of resonant UV line intensity to white-light (e-scattered visible), proportional to abundance × ionization fraction × dimming factor.

**Eq. (3) — Unvignetted telescope mirror area**:

$$ A = h\,D\,\tan[16/60\,(r-1.2)] - b $$

h = 거울 높이, D = 외부 차폐~거울 거리, r = R⊙ 단위 시선 높이, b = over-occulting 폭. / Geometry for the unvignetted area as the line of sight scans 1.2–10 R⊙.

**Eq. (4) — Toric grating stigmatic condition**:

$$ \frac{R_v}{R_h} = \cos\alpha \cdot \cos|\beta_o| $$

α = 입사각, ±β_o = stigmatic 회절각 / α = incidence angle, ±β_o = stigmatic diffraction angles for which spatial+spectral foci coincide.

**Eq. (5) — Linearly polarized radiance from three retarder positions**:

$$ pI = \frac{4}{3}\sqrt{I_o^2+I_+^2+I_-^2-I_oI_+-I_oI_--I_+I_-} $$

I_o, I_+, I_- = 30° 간격 retarder 위치에서의 측정값 / Measurements at three half-wave retarder positions separated by 30°; combines into linearly polarized radiance.

---

## 6. 읽기 가이드 / Reading Guide

1. **Section 1 (Primary Scientific Objectives)**: 가속·가열·태양풍 근원·플라스마 특성 4대 목표를 먼저 머리에 새기기. / Identify the four science pillars (acceleration, heating, sources, plasma properties).
2. **Section 2 (Spectroscopic Techniques)**: HI Lyα 프로파일, OVI doublet, Mg X / Si XII, Doppler dimming, Thomson scattered electron T_e. Figure 2의 dimming 곡선을 시각화하라. / Focus on Fig. 2 dimming curves for HI/OVI/SiXII and the C II pumping bump.
3. **Section 3 (Instrument Overview)**: 세 채널 구조와 FOV(Fig. 4) 이해, Table I 규격(0.23 Å, 12"–24" 분해능) 기억. / Skim the three channels, FOV (Fig. 4), and Table I.
4. **Section 4 (Occulted Telescope)**: 외부+내부 차폐+sunlight trap의 3중 산란광 억제. / Triple stray-light suppression strategy.
5. **Section 5 (Spectrometer Assembly)**: Rowland-Onaka geometry, ±β_o stigmatic, Table II/III 파라미터, XDL detector(1024×360, 25 µm), Table IV QE. / Toric Rowland geometry, parameters, XDL specs.
6. **Section 6 (Stray Light Suppression)**: 산란광 budget(Fig. 14)과 1×10⁻⁸ 수준 — 가장 중요. / Budget down to 10⁻⁸ disk-relative.
7. **Section 7 (Performance) – Section 11 (Operations)**: Table VII으로 마무리된 측정 성능 + 운용 / Wrap up with measured performance (Table VII) and ops.

읽으면서 "왜 두 개의 비슷한 채널이 필요한가?", "왜 toric grating인가?", "Doppler dimming이 왜 OVI 1037 Å에서 비단조적으로 변하는가?" 자문하라. / Ask yourself why two similar channels, why toric, and why OVI 1037 Å is non-monotonic.

---

## 7. 현대적 의의 / Modern Significance

UVCS는 SOHO 미션 기간(1996–2013) 동안 코로나 양성자와 OVI 이온이 약 2 R⊙ 이상에서 강하게 비등방적인 운동학적 온도(T_⊥ ≫ T_∥)를 보임을 발견하여 ion-cyclotron 가열 가설을 직접 지지했다. 이는 Parker의 열압력 구동 모델만으로는 빠른 태양풍을 설명하기 부족함을 시사하며, 현대의 Solar Probe(PSP)·Solar Orbiter·Metis(SO/Metis), ASPIICS/PROBA-3 등의 코로나그래프 분광 미션의 직접적 계승자이다. UVCS의 toric grating, KBr MCP+XDL, 다중 차폐(occulter+trap) 설계 패러다임은 후속 UV/EUV 코로나그래프 표준 설계가 되었다.

UVCS, throughout the SOHO mission (1996–2013), discovered strongly anisotropic kinetic temperatures (T_⊥ ≫ T_∥) of protons and especially O⁵⁺ ions above ~2 R⊙, providing direct support for ion-cyclotron-resonance heating in the fast wind. These results showed that thermal-pressure-driven Parker wind alone is insufficient and motivated all subsequent UV/EUV coronagraph spectrometer concepts (Metis on Solar Orbiter, ASPIICS/PROBA-3, LEMUR, in-situ comparisons with PSP and Solar Orbiter SWA). The instrument-design paradigm — toric Rowland gratings, KBr-coated MCP+XDL detectors, triple stray-light suppression — has become a standard for UV coronagraph spectrometers.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
