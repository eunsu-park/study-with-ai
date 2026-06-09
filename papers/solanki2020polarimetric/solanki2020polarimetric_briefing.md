---
title: "Pre-Reading Briefing: The Polarimetric and Helioseismic Imager on Solar Orbiter (SO/PHI)"
paper_id: "56_solanki_2020"
topic: Solar_Observation
date: 2026-04-25
type: briefing
---

# The Polarimetric and Helioseismic Imager on Solar Orbiter (SO/PHI): Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Solanki, S. K., del Toro Iniesta, J. C., Woch, J., Gandorfer, A., Hirzberger, J., Alvarez-Herrero, A., et al., "The Polarimetric and Helioseismic Imager on Solar Orbiter", A&A 642, A11 (2020). DOI: 10.1051/0004-6361/201935325
**Author(s)**: S. K. Solanki, J. C. del Toro Iniesta, J. Woch, A. Gandorfer, J. Hirzberger, A. Alvarez-Herrero, et al. (large international consortium)
**Year**: 2020

---

## 1. 핵심 기여 / Core Contribution

SO/PHI는 ESA의 Solar Orbiter (SO) 미션에 탑재된 **첫 번째 우주 자기장계 (magnetograph) 겸 헬리오사이즈몰로지 영상기 (helioseismic imager)** 로, 지구-태양선 (Sun-Earth line) 바깥에서 태양 광구 자기장과 LOS 속도를 측정하도록 설계된 사상 최초의 기기이다. SO/PHI는 Fe I 617.3 nm 흡수선의 Zeeman 효과와 Doppler 천이를 동시에 관측하기 위해, **조정 가능한 LiNbO3 Fabry-Pérot 에탈론** (협대역 영상 분광계)과 **액정 가변 위상지연자 (LCVR)** 기반의 편광 변조 패키지를 결합하였다. 두 개의 망원경 — Full Disc Telescope (FDT, 17.5 mm 구경, 2° FOV) 와 High Resolution Telescope (HRT, 140 mm 구경, 0°.28 FOV) — 을 통해 동일한 광학 경로 위에서 풀-디스크와 200 km 분해능의 고해상도 관측을 모두 제공한다. 또한 텔레메트리 한계를 극복하기 위해 **Milne-Eddington 가정 하에 RTE 인버전을 우주에서 (on-board) 직접 수행** 하는 사상 최초의 우주 spectropolarimeter이다.

SO/PHI is the first **space-based magnetograph and helioseismic imager** on the ESA Solar Orbiter (SO) mission, the first ever instrument designed to measure the solar photospheric magnetic field and LOS velocity from outside the Sun-Earth line. It uses Zeeman-effect spectropolarimetry and Doppler imaging on the Fe I 617.3 nm line, combining a **tunable LiNbO3 Fabry-Pérot etalon** narrow-band filter with **liquid-crystal variable retarder (LCVR)** polarisation modulators. Two co-aligned telescopes — the Full Disc Telescope (FDT, 17.5 mm aperture, 2° FOV) and the High Resolution Telescope (HRT, 140 mm aperture, 0°.28 FOV) — share a common filtergraph path, providing full-disc context and ~200 km resolution at perihelion. To overcome severe telemetry limits on the elliptical SO orbit (down to 0.28 AU), SO/PHI is the **first space spectropolarimeter to perform Milne-Eddington Radiative Transfer Equation (RTE) inversions on-board** in dedicated FPGAs, returning compressed maps of $I_c, B, \gamma, \phi, v_{LOS}$ rather than raw Stokes data.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

2020년대에 도달한 태양 자기장 우주 관측의 정점은 **SOHO/MDI (1995)**, **Hinode/SP (2006)**, **SDO/HMI (2010)** 의 계보를 잇는다. 그러나 이들 모두는 황도면 (ecliptic) 안에 위치하여 항상 태양의 적도 영역 정면에서만 관측해 왔다. 태양 다이나모와 자기 플럭스 수송에서 결정적으로 중요한 **고위도와 극 영역**은 시야 효과 (foreshortening) 때문에 신뢰성 있게 관측된 적이 없었다. ESA의 Solar Orbiter (Müller et al. 2020)는 황도면을 벗어나 최대 ~33° 헬리오그래픽 위도까지 도달하고, 근일점 0.28 AU에서 태양 표면 200 km를 분해함으로써 이 공백을 메우려는 첫 시도였다.

By 2020, space-based solar magnetography had matured through **SOHO/MDI (1995)**, **Hinode/SP (2006)**, and **SDO/HMI (2010)**, but every magnetograph flown so far had observed exclusively from the ecliptic plane, looking at the Sun's equator. The **polar regions and high latitudes** — critical for the solar dynamo, meridional flow, and flux transport — had never been imaged with magnetic-field sensitivity comparable to disc-centre observations. ESA's Solar Orbiter mission (Müller et al. 2020) was conceived to break this 60-year ecliptic-plane monopoly by reaching heliographic latitudes up to ~33° and observing at perihelion distances down to 0.28 AU, where 200 km features become resolvable.

### 타임라인 / Timeline

```
1908 ─── Hale: First sunspot Zeeman magnetic field detection
1953 ─── Babcock: First photographic magnetograph
1995 ─── SOHO/MDI launched (Michelson interferometer, Ni I 6768 Å)
2006 ─── Hinode/SP launched (Fe I 6301/6302 Å, slit spectrograph)
2010 ─── SDO/HMI launched (Michelson, Fe I 6173 Å)
2009 ─── Sunrise-1 balloon flight (IMaX: prototype LiNbO3 etalon imager)
2013 ─── Sunrise-2 balloon flight (IMaX validates LiNbO3 + LCVR concept)
2020 ─── Solar Orbiter launched (10 Feb 2020); SO/PHI is fifth space magnetograph
2022 ─── First SO/PHI close perihelion approach (~0.32 AU)
2025 ─── First high-latitude orbit phases (>17° heliographic)
2029+ ── Mission extended phases reach >30° heliographic latitude
```

---

## 3. 필요한 배경 지식 / Prerequisites

본 논문을 이해하려면 다음의 배경 지식이 필요하다.

To follow this paper, readers should be comfortable with:

1. **Stokes vector and Zeeman effect / 스토크스 벡터와 제이만 효과**: 빛의 편광 상태 $(I,Q,U,V)$의 정의와, 자기장에서 흡수선이 분리되는 정상 (normal) Zeeman 패턴 ($\sigma$, $\pi$ 성분).
   The four-component Stokes formalism $(I,Q,U,V)$ describing the polarisation state of light, and the normal Zeeman triplet (σ and π components) split by a magnetic field.

2. **Milne-Eddington atmosphere / MIlne-Eddington 대기**: 라디에이티브 전송 방정식 (RTE)의 분석 가능한 단순화로, 광학 깊이에 대해 선형으로 변하는 source function과 일정한 자기장·속도장을 가정한 모델 (Unno-Rachkovsky 해).
   The simplified RTE solution (Unno-Rachkovsky equations) assuming a linear source function in optical depth and depth-independent magnetic and velocity fields.

3. **Fabry-Pérot etalons / 패브리-페로 에탈론**: 두 평행 반사면 사이 다중 간섭으로 협대역 투과를 만드는 분광 소자. Free Spectral Range (FSR), finesse, 그리고 collimated vs telecentric 구성의 차이.
   Multi-beam interference filters; free spectral range, finesse, and the trade-offs between collimated and telecentric mounting.

4. **Liquid Crystal Variable Retarders (LCVRs) / 액정 가변 위상지연자**: 인가 전압에 따라 복굴절이 바뀌는 네매틱 액정 셀로, 기계적 요소 없이 편광 변조를 수행.
   Nematic liquid crystal cells whose birefringence is voltage-tunable, allowing polarisation modulation without moving parts.

5. **Helioseismology basics / 헬리오사이즈몰로지 기초**: 태양 진동 (p-modes)을 도플러 영상으로 측정하여 내부 구조와 흐름을 추론하는 기법; global vs local helioseismology.
   The use of Doppler imaging of solar p-mode oscillations to probe the solar interior; distinction between global and local helioseismology.

6. **Solar Orbiter mission profile / 솔라 오비터 미션 개요**: 0.28-1.0 AU 타원 궤도, 금성 중력 지원에 의한 황도면 이탈, 헬리오그래픽 위도 ~33°까지의 도달.
   The 0.28-1.0 AU elliptical orbit with Venus gravity assists progressively raising the heliographic latitude up to ~33°.

7. **CCSDS data compression / 우주 데이터 압축 표준**: lossy/lossless 영상 압축 표준 (CCSDS 122.0-B-1).
   The CCSDS 122.0-B-1 image compression standard used to fit data within tight telemetry budgets.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **SO/PHI** | Solar Orbiter Polarimetric and Helioseismic Imager: Solar Orbiter 탑재 자기장 및 헬리오사이즈몰로지 측정 기기. The magnetograph and helioseismograph aboard Solar Orbiter. |
| **HRT** | High Resolution Telescope: 140 mm 구경, 0°.28 FOV, 0.28 AU 근일점에서 ~200 km 분해능. 140 mm aperture telescope, 0°.28 FOV, ~200 km resolution at 0.28 AU perihelion. |
| **FDT** | Full Disc Telescope: 17.5 mm 구경, 2° FOV — 근일점에서도 태양 전체 디스크 관측. 17.5 mm aperture, 2° FOV — sees the full solar disc even at perihelion. |
| **Fe I 617.3 nm** | SO/PHI가 관측하는 Landé factor $g_{eff}=2.5$의 광구 자기장 진단 흡수선. The photospheric magnetic-field-diagnostic absorption line ($g_{eff}=2.5$) sampled by SO/PHI. |
| **PMP** | Polarisation Modulation Package: 두 LCVR + 선형 편광자로 구성된 편광 변조기. Polarisation modulator: two LCVRs plus a linear polariser. |
| **LCVR** | Liquid Crystal Variable Retarder: 전압으로 위상 지연을 조정하는 액정 셀. Voltage-tunable nematic liquid crystal retarder. |
| **LiNbO3 etalon** | 리튬 나이오베이트 결정 패브리-페로: 전기광학 효과로 ~$351\,\mathrm{m\AA/kV}$ 파장 조정. Lithium niobate Fabry-Pérot whose pass-band tunes electro-optically at ~351 mÅ/kV. |
| **HREW** | Heat Rejecting Entrance Window: 다층 코팅 입사창으로, 입사 태양 에너지의 ~3.2%만 통과. Multi-layer entrance window passing only ~3.2% of incoming solar energy. |
| **Milne-Eddington (ME) inversion** | Source function이 광학 깊이에 선형, B와 $v$가 깊이 독립인 RTE 해. RTE solution assuming linear source function in $\tau$ and depth-independent atmospheric parameters. |
| **Stokes I, Q, U, V** | 빛의 강도, 두 선형 편광 차이, 45° 선형 편광 차이, 원편광 — Zeeman 진단. Total intensity, two linear polarisation differences, and circular polarisation — Zeeman diagnostics. |
| **FSR / Finesse** | 패브리-페로의 자유 분광 범위 (FSR=0.301 nm) 와 finesse=30 (passband 폭 ~106 mÅ). Etalon free spectral range (0.301 nm) and finesse (30) defining the 106 mÅ FWHM passband. |
| **RTE on-board inversion** | RTE 인버전을 비행 중 FPGA에서 수행하는 SO/PHI의 첫 시도. SO/PHI's first-ever in-flight FPGA-based RTE inversion. |

---

## 5. 수식 미리보기 / Equations Preview

### (1) Number of accumulations needed for noise floor / 누적 노출 수

$$
N_{acc} = \frac{1}{4}\left(\frac{S/N}{\bar{\epsilon}\, S/N_{single}}\right)^2 \approx 16
$$

각 픽셀에서 $S/N=10^3$ (Stokes Q,U,V 노이즈 ~$10^{-3}I_c$) 를 달성하기 위해 필요한 누적 프레임 수. $S/N_{single}=255$ (단일 노출 신호대잡음)와 평균 편광 효율 $\bar\epsilon \approx 0.57$이 주어지면 $N_{acc} \approx 16$이 산출된다.
The number of frames that must be accumulated per polarisation/wavelength state to reach $S/N=10^3$ in Stokes Q,U,V (noise level ~$10^{-3}I_c$). Given single-exposure $S/N_{single}=255$ and mean polarimetric efficiency $\bar\epsilon\approx 0.57$, this gives $N_{acc}\approx 16$.

### (2) Crosstalk correction / 교차오염 보정

$$
\begin{aligned}
Q_{measured} &= Q_{corr} + a V_{measured},\\
U_{measured} &= U_{corr} + b V_{measured}.
\end{aligned}
$$

연속체 (continuum) 영역에서 Q,U의 평균 오프셋이 V와 어떻게 선형 상관되는지를 적합 (fit)하여 잔여 V→Q,U 교차오염 계수 $a, b$를 산출한 뒤, 보정된 $Q_{corr}, U_{corr}$를 얻는다.
By linearly fitting the mean continuum offsets of measured Q,U against measured V, the residual V→Q,U crosstalk coefficients $a,b$ are derived and the corrected Stokes signals recovered.

### (3) Masked-gradient focus metric (FDT) / FDT 초점 평가량

$$
\delta I = \frac{1}{\langle I\rangle \sum_{i,j} M_{i,j}}\sum_{i,j}\left[\left(\frac{\partial I(x,y)}{\partial x}\right)^2 + \left(\frac{\partial I(x,y)}{\partial y}\right)^2 \right] M_{i,j}
$$

FDT의 자동 재초점에 사용되는 평가량: 태양 림 (limb) 부근의 환형 마스크 $M_{i,j}$ 안에서 영상 그래디언트 제곱의 평균을 계산하고, 이 값이 최대가 되는 초점 위치를 찾는다.
The metric used by FDT autofocus: the mean squared image gradient inside an annular mask $M_{i,j}$ around the defocused solar limb is computed, and the focus position maximising this metric is selected.

### (4) Stokes intensity profile (Unno-Rachkovsky / Milne-Eddington) — implicit / 묵시적 사용

$$
\frac{d\mathbf{I}}{d\tau} = \mathbf{K}\,(\mathbf{I}-\mathbf{S}),\quad \mathbf{S} = (B_0+B_1\tau)(1,0,0,0)^T
$$

ME 가정에서 $\mathbf{K}$ (4x4 흡수 행렬)가 광학 깊이에 무관하면 분석 해 (Unno-Rachkovsky equations)가 존재하며, SO/PHI는 이 해를 인버전 알고리즘에 사용한다.
Under ME assumptions, the 4-Stokes RTE has the analytic Unno-Rachkovsky solution which SO/PHI's on-board inverter exploits.

---

## 6. 읽기 가이드 / Reading Guide

이 논문은 35페이지의 instrument paper이며, 다음 흐름으로 읽기를 권장한다.

This 35-page instrument paper is best read in this order:

- **Sect. 1 (Introduction)**: SO/PHI가 이전 magnetograph (MDI/HMI/Hinode-SP)와 어떻게 다른지 빠르게 파악. Quickly compare SO/PHI with MDI/HMI/Hinode-SP.
- **Sect. 2 (Science objectives)**: 8개 주요 과학 질문 — 다이나모, 자기 플럭스 수송, 태양풍 기원, 헬리오사이즈몰로지, 이바리언스 변동성. Eight overarching science questions: dynamo, flux transport, solar wind, helioseismology, irradiance.
- **Sect. 3 (Instrument overview)**: 동작 원리 도해 (Fig. 3) — 한 페이지로 전체 개념 파악. The functional schematic (Fig. 3) gives the whole concept on one page.
- **Sect. 4 (Optical unit)**: HRT vs FDT 광학 차이 (4.2.1, 4.2.4), PMP/LCVR (4.2.7), Fabry-Pérot (4.2.8). Optical differences between HRT and FDT, PMP and LCVR, Fabry-Pérot.
- **Sect. 5 (Electronics)**: 처음에는 5.2 (DPU)와 5.4 (HVPS)만 보고, 나머지는 참조. First read only 5.2 (DPU) and 5.4 (HVPS); skim the rest.
- **Sect. 6 (Calibration)**: 편광 효율 결과 (0.57 for Q,U,V)와 spectral 평가 (FWHM 106 mÅ). Polarimetric efficiencies and spectral characterisation.
- **Sect. 7 (Operations)**: 7.1 (12 system states), 7.3 (data acquisition), 7.4 (on-board processing pipeline & RTE inversion) 가 핵심. Sect. 7.1, 7.3, and 7.4 (especially the on-board RTE inversion) are key.
- **Sect. 8 (Conclusions)**: 핵심 혁신 5가지 요약. The five key innovations.

특히 다음 그림들은 반드시 분석할 것: Fig. 3 (동작 원리), Fig. 5 (시스템 도면), Fig. 6/7 (HRT 광학), Fig. 9/10 (FDT 광학), Fig. 16 (PMP), Fig. 18 (filtergraph), Fig. 31 (DPU 블록도), Fig. 40 (전처리 파이프라인).

The figures to study carefully: Fig. 3 (functional principle), Fig. 5 (block diagram), Fig. 6/7 (HRT optics), Fig. 9/10 (FDT optics), Fig. 16 (PMP scheme), Fig. 18 (filtergraph), Fig. 31 (DPU block), Fig. 40 (preprocessing pipeline).

---

## 7. 현대적 의의 / Modern Significance

SO/PHI는 단순히 "또 하나의 magnetograph"가 아니라, **세 가지 패러다임 변화**의 출발점이다.

SO/PHI is not just "another magnetograph" — it inaugurates **three paradigm shifts**:

1. **첫 황도면 이탈 자기장 관측 / First out-of-ecliptic magnetograms**: 2025-2029년 고위도 단계에서, 태양 극역 자기장과 자오선 흐름 (meridional flow) 측정에 새로운 기준을 제공할 것이다. Will provide unprecedented measurements of polar magnetic fields and meridional flow during 2025-2029 high-latitude phases.

2. **첫 우주 on-board RTE 인버전 / First on-board RTE inversion in space**: FPGA에서 ME 인버전을 수행하여 raw 데이터 대비 텔레메트리를 ~5배 절약. 이는 향후 모든 deep-space 분광편광 미션 (예: Vigil, MUSE의 후속) 의 표준이 될 것이다. Sets the standard for all future deep-space spectropolarimeters (e.g., ESA Vigil, MUSE follow-ons) by demonstrating on-board ME inversion.

3. **스테레오스코픽 헬리오사이즈몰로지 / Stereoscopic helioseismology**: SO/PHI를 SDO/HMI와 결합하면 태양 내부 음파의 "skip distance" 가 한 둘레의 절반에 도달, **타코클라인 (tachocline) 영역의 직접 조사**가 가능해진다. Combined with SDO/HMI, SO/PHI enables stereoscopic helioseismology with skip distances up to half a circumference, allowing for the first time direct probing of the tachocline.

또한, SO/PHI의 LiNbO3 etalon, LCVR PMP, on-board RTE 인버전 모두 차세대 기기 (예: 일본의 Solar-C, 미국의 MUSE)에 직접 영향을 미치고 있으며, **liquid crystal optics를 우주에서 검증한 첫 미션**이라는 기술 유산도 크다.

The LiNbO3 etalon, LCVR-based PMP, and on-board RTE inversion are all directly inherited by next-generation instruments (Solar-C, MUSE), and SO/PHI represents the **first space-qualification of liquid crystal polarisation optics**.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
