---
title: "Pre-Reading Briefing: Modeling Multiphase Plasma in the Corona: Prominences and Rain"
paper_id: "87"
topic: Living_Reviews_in_Solar_Physics
date: 2026-04-23
type: briefing
---

# Modeling Multiphase Plasma in the Corona: Prominences and Rain — Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Keppens, R., Zhou, Y., Xia, C. "Modeling multiphase plasma in the corona: prominences and rain", *Living Reviews in Solar Physics*, 22:4 (2025). DOI: 10.1007/s41116-025-00043-2

**Authors**: Rony Keppens (KU Leuven), Yuhao Zhou (Nanjing University), Chun Xia (Yunnan University)

**Year**: 2025

---

## 1. 핵심 기여 / Core Contribution

**English**: This Living Review synthesizes a decade of multi-dimensional MHD modeling that unifies two seemingly distinct phenomena — solar prominences (long-lived cool dense structures suspended in the hot corona) and coronal rain (short-lived cool blobs condensing and falling along coronal loops) — under a single physical origin: thermal instability (TI) driven by the imbalance between optically thin radiative losses and coronal heating. The review traces the modeling pipeline from 1D hydrodynamic loop experiments, through 2D/2.5D arcade and flux-rope MHD simulations, up to full 3D chromosphere-to-corona models, highlighting how the open-source MPI-AMRVAC code has made it possible to self-consistently form condensations that reproduce observed morphologies, rain drop sizes (~100 km), fall speeds (~100 km/s), density contrasts (100–1000×), and temperature contrasts (10⁴ K cool phase embedded in 10⁶ K corona).

**Korean / 한국어**: 본 Living Review는 겉보기에 서로 다른 두 현상 — 태양 프로미넌스(뜨거운 코로나 속에 떠 있는 오래 지속되는 저온 고밀도 구조)와 코로나 비(코로나 루프를 따라 응축되어 떨어지는 단명 저온 덩어리) — 을 하나의 물리적 기원, 즉 광학적으로 얇은 복사 손실과 코로나 가열 사이의 불균형이 일으키는 열 불안정성(TI, Thermal Instability)으로 통합하는 지난 10년간의 다차원 MHD 모델링 성과를 종합한다. 1D 수력학적 루프 실험에서 2D/2.5D 아케이드·플럭스로프 MHD 시뮬레이션을 거쳐 전 3D 채층-코로나 모델에 이르기까지의 모델링 파이프라인을 추적하며, 오픈소스 MPI-AMRVAC 코드를 통해 관측된 형태학, 비 방울 크기(~100 km), 낙하 속도(~100 km/s), 밀도 대비(100–1000배), 온도 대비(10⁶ K 코로나에 묻힌 10⁴ K 저온상)를 자기일관적으로 재현할 수 있음을 강조한다.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

**English**: The modeling of coronal condensations sits at the intersection of three classical threads. (i) **Prominence magnetostatics** (Kippenhahn & Schlüter 1957; Kuperus & Raadu 1974) answered "how can cool dense matter be suspended against gravity?" with the dipped-field-line and flux-rope topologies. (ii) **Thermal instability theory** (Parker 1953; Field 1965) answered "why does cool matter appear in the first place?" with the radiative cooling catastrophe. (iii) **Coronal loop hydrodynamics** (Antiochos & Klimchuk 1991; Antiochos et al. 1999, 2000) discovered *thermal non-equilibrium* (TNE) — a loop with footpoint-concentrated heating cannot reach a stationary state and cycles through condensation-drainage episodes. For decades these threads ran in parallel. Only in the 2010s, with adaptive-mesh-refined MHD codes like MPI-AMRVAC and self-consistent chromosphere-to-corona simulations, could in-situ condensation be modeled without ad-hoc mass injection, unifying prominence and rain physics.

**Korean / 한국어**: 코로나 응축 모델링은 고전적 세 흐름의 교차점에 놓여 있다. (i) **프로미넌스 자기정역학** (Kippenhahn & Schlüter 1957; Kuperus & Raadu 1974)은 "어떻게 저온 고밀도 물질이 중력에 버틸 수 있는가?"를 dipped 자력선과 플럭스로프 위상으로 답했다. (ii) **열 불안정성 이론** (Parker 1953; Field 1965)은 "애초에 왜 저온 물질이 나타나는가?"를 복사 냉각 폭주로 답했다. (iii) **코로나 루프 수력학** (Antiochos & Klimchuk 1991; Antiochos et al. 1999, 2000)은 *열 비평형(TNE)* — 풋포인트 집중 가열을 받는 루프는 정상상태에 도달하지 못하고 응축-배수 순환을 겪음 — 을 발견했다. 수십 년간 이 흐름들은 병렬로 달렸다. 2010년대에 이르러서야 MPI-AMRVAC 같은 적응격자 MHD 코드와 자기일관적 채층-코로나 시뮬레이션이 등장하면서, 임시 질량 주입 없이 *in-situ* 응축을 모델링하고 프로미넌스와 비 물리를 통합할 수 있게 되었다.

### 타임라인 / Timeline

```
1953 Parker ───── Thermal instability (pure hydro)
 |
1957 Kippenhahn-Schlüter ── Dipped field prominence support
 |
1965 Field ───── Full TI dispersion relation with B, conduction
 |
1974 Kuperus-Raadu ── Flux-rope prominence topology
 |
1991 Antiochos-Klimchuk ── 1D evaporation-condensation model
 |
1999-2000 Antiochos+ ── Thermal Non-Equilibrium (TNE) cycles
 |
2011-2012 Xia+ ──── First 2.5D chromosphere-to-corona MHD prominence
 |
2014-2016 Xia & Keppens ── First full 3D MHD prominence formation
 |
2018-2022 Froment/Pelouze ── 1000+ loop parameter surveys
 |
2024-2025 Lu+, Donné & Keppens ── 3D rain & reconnection-condensation
```

---

## 3. 필요한 배경 지식 / Prerequisites

**English**:
1. **MHD equations** — continuity, momentum (with Lorentz force), energy equation with anisotropic (field-aligned) thermal conduction, induction equation with resistivity.
2. **Optically thin radiative cooling** — the cooling function Λ(T) for coronal abundances, which rises sharply around T ~ 10⁵ K and falls for T > 10⁶ K.
3. **Thermal instability (TI)** — Field (1965) criterion: isobaric mode unstable when (∂L/∂T)_p < 0; isochoric mode unstable when (∂L/∂T)_ρ < 0.
4. **Coronal loop structure** — transition region, chromospheric evaporation, conductive-radiative balance.
5. **Prominence magnetic topologies** — sheared arcades (Kippenhahn-Schlüter) vs. flux ropes (Kuperus-Raadu).
6. **Numerical MHD** — adaptive mesh refinement, shock capturing, handling of the sharp prominence-corona transition region (PCTR).
7. **Familiarity with Paper #36 (Reale loops), #39 (Mackay prominence magnetic), #61 (Priest/reconnection)** from our previous reading.

**Korean / 한국어**:
1. **MHD 방정식** — 연속성, 운동량(Lorentz 힘 포함), 비등방성(자기장 정렬) 열전도가 포함된 에너지 방정식, 저항이 있는 유도 방정식.
2. **광학적으로 얇은 복사 냉각** — 코로나 원소 조성에 대한 냉각 함수 Λ(T). T ~ 10⁵ K 부근에서 급상승, T > 10⁶ K에서 감소.
3. **열 불안정성 (TI)** — Field (1965) 기준: 등압 모드는 (∂L/∂T)_p < 0일 때 불안정; 등적 모드는 (∂L/∂T)_ρ < 0일 때 불안정.
4. **코로나 루프 구조** — 전이역, 채층 증발, 전도-복사 균형.
5. **프로미넌스 자기 위상** — 전단 아케이드(Kippenhahn-Schlüter) 대 플럭스로프(Kuperus-Raadu).
6. **수치 MHD** — 적응 격자 미세화, 충격파 포착, 날카로운 프로미넌스-코로나 전이역(PCTR) 처리.
7. **이전 읽기 **논문 #36 (Reale 루프), #39 (Mackay 프로미넌스 자기장), #61 (Priest/재결합)**과의 친숙도.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| Thermal Instability (TI) / 열 불안정성 | Field (1965) linear instability where a cool perturbation radiates more efficiently, cools further, and runs away. / 저온 섭동이 더 효율적으로 복사하여 더 냉각되는 폭주 선형 불안정성. |
| Thermal Non-Equilibrium (TNE) / 열 비평형 | Inability of a footpoint-heated loop to reach steady state; manifests as cyclic condensation formation. / 풋포인트 가열 루프가 정상상태에 도달하지 못하고 주기적 응축을 형성하는 현상. |
| Thermal Continuum (TC) / 열 연속체 | Continuous spectrum of unstable thermal modes in stratified/magnetized media (van der Linden & Goossens 1991). / 층화·자화 매질에서 나타나는 불안정 열모드의 연속 스펙트럼. |
| Cooling function Λ(T) / 냉각 함수 | Radiative loss rate per unit emission measure; optically-thin radiative loss is n_e n_H Λ(T). / 방출측도당 복사 손실률. |
| PCTR (Prominence-Corona Transition Region) / 프로미넌스-코로나 전이역 | Thin shell around a prominence with steep T, ρ gradients. / 프로미넌스 주위의 T, ρ 급경사 박층. |
| Evaporation-Condensation / 증발-응축 | Scenario where footpoint heating drives chromospheric evaporation; excess mass in the corona then condenses. / 풋포인트 가열로 채층 증발이 일어나고, 코로나에 쌓인 질량이 응축되는 시나리오. |
| Levitation-Condensation / 부양-응축 | Scenario where reconnection below an arcade forms a flux rope that lifts and traps cool material. / 아케이드 하부 재결합이 플럭스로프를 형성하여 저온 물질을 들어올리고 가두는 시나리오. |
| Kippenhahn-Schlüter (KS) / KS 모델 | Sheared arcade prominence with magnetic tension supporting matter in field-line dips. / 자력선 dip에서 자기장력이 물질을 지지하는 전단 아케이드 프로미넌스. |
| Kuperus-Raadu (KR) / KR 모델 | Flux-rope prominence with inverse polarity, supported by Lorentz force from helical field. / 역극성을 갖고 나선 자기장의 Lorentz 힘으로 지지되는 플럭스로프 프로미넌스. |
| MPI-AMRVAC | Open-source parallel adaptive-mesh-refined MHD code developed at KU Leuven (Keppens et al. 2023). / KU Leuven이 개발한 오픈소스 병렬 적응 격자 MHD 코드. |
| Rayleigh-Taylor Instability (RTI) / 레일리-테일러 불안정성 | Buoyancy instability when dense fluid sits on light fluid; drives prominence plume dynamics. / 무거운 유체가 가벼운 유체 위에 놓일 때의 부력 불안정성. |
| Convective Continuum Instability (CCI) / 대류 연속체 불안정성 | Continuum instability arising when projected Brunt-Väisälä frequency N² < 0 along flux surfaces. / 자속면을 따른 투영 Brunt-Väisälä 진동수의 음수화로 발생하는 연속체 불안정성. |

---

## 5. 수식 미리보기 / Equations Preview

**English**: Five equations form the backbone of the review.

1. **MHD momentum equation** (Eq. 1):
$$\rho(\partial_t \mathbf{v} + \mathbf{v}\cdot\nabla\mathbf{v}) = -\nabla p + \rho\mathbf{g} + \frac{1}{\mu_0}(\nabla\times\mathbf{B})\times\mathbf{B} + \mathbf{F}_{\text{visc}}$$

2. **MHD energy equation with radiative losses & conduction** (Eq. 4):
$$\mathcal{R}\rho\partial_t T + \mathcal{R}\rho\mathbf{v}\cdot\nabla T + (\gamma-1)p\nabla\cdot\mathbf{v} = (\gamma-1)\left[\rho\mathcal{L} + \nabla\cdot(\kappa(\tilde T^{5/2})(\hat{\mathbf{b}}\cdot\nabla T)\hat{\mathbf{b}})\right]$$

3. **Optically thin radiative heating-cooling balance** (Eq. 6):
$$\rho h = n_e n_H \Lambda(T) - \nabla\cdot(\kappa(\tilde T^{5/2})(\hat{\mathbf{b}}\cdot\nabla T)\hat{\mathbf{b}})$$

4. **Field (1965) TI dispersion relation** (Eq. 8, uniform hydro):
$$\omega^3 - i\omega^2\frac{(\gamma-1)\mathcal{L}_T}{\mathcal{R}} - \omega c_0^2 k^2 - i(\gamma-1)(\rho_0\mathcal{L}_\rho - T_0\mathcal{L}_T)k^2 = 0$$

5. **Klimchuk-Luna (2019) TNE onset condition** (Eq. 11):
$$1 + \frac{A_{\text{tr}}R_{\text{tr}}}{A_c R_c} < \frac{H_{\text{foot}}}{H_{\text{apex}}}$$

**Korean / 한국어**: 리뷰의 핵심 다섯 식. 각각 (1) Lorentz 힘을 포함한 MHD 운동량, (2) 자기장 정렬 열전도와 복사 손실을 포함한 MHD 에너지 방정식, (3) 정상상태 가열-복사 균형 — 알려지지 않은 가열 h가 Λ(T)와 전도를 정확히 상쇄해야 함, (4) Field (1965)의 균일 매질 열 불안정성 3차 분산 관계, (5) Klimchuk-Luna 의 TNE 발생 부등식 — 풋포인트 가열 H_foot 이 정점 가열 H_apex 대비 임계값을 초과해야 순환 응축이 일어남.

---

## 6. 읽기 가이드 / Reading Guide

**English**: The paper is 69 pages with 9 sections. Recommended reading order and emphasis:

1. **§1-2 (skim)** — Motivation and pointers to related reviews. Note the key observational number: rain blobs with widths of a few hundred km (down to 100 km or less, see Fig. 17).
2. **§3 (careful)** — Force balance and energy balance; this sets up the static equilibria that all multi-D models perturb. Equations (1)-(7) are foundational.
3. **§4 (the theoretical heart)** — Linear MHD spectroscopy, Field's TI, thermal continuum. Read slowly; equation (8) and the discussion of isobaric/isochoric/entropy modes are critical.
4. **§5 (foundational)** — 1D loop evolution, evaporation-condensation, TNE. The Klimchuk-Luna criterion (Eq. 11) is a litmus test for rain formation.
5. **§6 (the main event)** — Multi-D MHD. Focus on Table 1 (paper classification by magnetic topology and formation pathway). Subsections §6.1-6.3 walk through 2D arcades → 2.5D flux ropes → 3D.
6. **§7 (postflare rain)** — A specialized but important topic; skim first time.
7. **§8 (beyond solar)** — Galactic prominences, stellar slingshot prominences; skim.
8. **§9 (open problems)** — Re-read after the rest; this is where to anchor your own future research questions.

**Korean / 한국어**: 논문은 69페이지 9절. 권장 읽기 순서:

1. **§1-2 (속독)** — 동기와 관련 리뷰로의 포인터. 핵심 관측 숫자: 비 덩어리 폭 수백 km, 최소 100 km 이하 (Fig. 17).
2. **§3 (정독)** — 힘 균형과 에너지 균형. 모든 다차원 모델이 섭동하는 정적 평형 세팅. 식 (1)-(7)이 토대.
3. **§4 (이론적 핵심)** — 선형 MHD 분광학, Field의 TI, 열 연속체. 천천히 읽을 것; 식 (8)과 등압/등적/엔트로피 모드 논의가 결정적.
4. **§5 (토대)** — 1D 루프 진화, 증발-응축, TNE. Klimchuk-Luna 기준 (식 11)은 비 형성의 리트머스 시험.
5. **§6 (본론)** — 다차원 MHD. Table 1 (자기 위상과 형성 경로별 논문 분류)에 집중. §6.1-6.3은 2D 아케이드 → 2.5D 플럭스로프 → 3D 순.
6. **§7 (포스트플레어 비)** — 특화 주제이나 중요; 첫 번째 읽기는 속독.
7. **§8 (태양 외)** — 은하 프로미넌스, 항성 슬링샷 프로미넌스; 속독.
8. **§9 (미해결 문제)** — 나머지 읽은 후 재독; 본인의 향후 연구 질문을 정박시키는 지점.

---

## 7. 현대적 의의 / Modern Significance

**English**: This review is timely because (i) the DKIST at NSF's Inouye Solar Telescope and the Goode Solar Telescope are now resolving prominence fine structure and rain strand widths down to 10-21 km, pushing below the resolution of current state-of-the-art 3D simulations. (ii) Solar Orbiter's Metis coronagraph images erupting prominences out to 10 R_⊙ in polarized H-α. (iii) Space weather forecasting frameworks (e.g., Baratashvili et al. 2025) currently ignore prominence mass in CME ejecta, yet this mass can be >50% of the ejected material — incorporating multiphase physics could improve geo-effectiveness predictions. (iv) Stellar counterparts: slingshot prominences on rapidly-rotating stars influence stellar angular momentum loss; coronal rain on young Suns (Daley-Yates et al. 2023) appears in Hα asymmetries. The MHD machinery is scale-invariant, so the same TI + heating/cooling framework applies from the solar chromosphere to the intergalactic medium (Sharma et al. 2012).

**Korean / 한국어**: 본 리뷰는 다음 이유로 시의적절하다. (i) NSF Inouye의 DKIST와 Goode 태양망원경이 프로미넌스 미세구조와 비 strand 폭을 10-21 km까지 분해하며, 현 최첨단 3D 시뮬레이션의 해상도를 넘어서고 있다. (ii) Solar Orbiter의 Metis 코로나그래프가 10 R_⊙까지 편광 Hα로 분출 프로미넌스를 영상화한다. (iii) 우주기상 예보 체계 (Baratashvili et al. 2025 등)는 현재 CME 분출물의 프로미넌스 질량을 무시하지만, 이 질량이 분출물의 50% 이상일 수 있어 — 다상 물리 통합이 지자기 효과 예측을 개선할 수 있다. (iv) 항성 대응: 고속 회전 별의 슬링샷 프로미넌스가 별 각운동량 손실에 영향; 젊은 태양의 코로나 비 (Daley-Yates et al. 2023)가 Hα 비대칭에 나타난다. MHD 기법은 스케일 불변이므로, 태양 채층에서 은하간 매질 (Sharma et al. 2012)까지 동일한 TI + 가열/냉각 틀이 적용된다.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
