---
title: "Pre-Reading Briefing: The Electron Density of the Solar Corona"
paper_id: "62_van_de_hulst_1950"
topic: Solar_Observation
date: 2026-04-28
type: briefing
---

# The Electron Density of the Solar Corona: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: H. C. van de Hulst, "The electron density of the solar corona", *Bulletin of the Astronomical Institutes of the Netherlands*, Vol. XI, No. 410, pp. 135-149 (1950).
**Author(s)**: H. C. van de Hulst (Leiden Observatory)
**Year**: 1950
**Bibcode**: 1950BAN....11..135V

---

## 1. 핵심 기여 / Core Contribution

### 한국어
이 논문은 **K-코로나의 백색광 밝기와 편광 관측으로부터 코로나 전자 밀도 N(r)을 자기무모순적으로(self-consistently) 도출하는 표준 절차**를 확립한 고전 논문이다. van de Hulst는 (a) F-corona(황도광 성분)와 K-corona(전자 톰슨 산란 성분)를 폐합 방정식으로 분리하고, (b) Schuster-Minnaert 형식의 적분 방정식 K_t(x), K_r(x)를 풀어 전자 밀도 N(r)을 구하며, (c) 결과를 적도/극지 영역, 최소/최대 활동기에 대해 1% 정밀도의 "model corona" 표로 제시한다. 적도 영역과 극지 영역을 분리해서 처리한 것, 그리고 β=70°(중위도) 부근에서 전자 밀도가 최소가 되는 흥미로운 결과를 발견한 것이 새로운 기여이며, 도출된 적분 공식은 70년이 지난 오늘날까지 pB 역변환의 표준 출발점으로 사용되고 있다.

### English
This paper establishes the **standard procedure for self-consistently deriving the coronal electron density N(r) from white-light brightness and polarization observations of the K-corona**. van de Hulst (a) separates the F-corona (zodiacal light component) from the K-corona (Thomson-scattered electron component) via closure equations, (b) solves the Schuster-Minnaert integral equations for K_t(x) and K_r(x) to obtain N(r), and (c) presents the result as a "model corona" table accurate to 1%, separately for equatorial and polar regions in both minimum and maximum phases. The novel contributions are the separate treatment of equatorial and polar regions, and the discovery of an electron-density minimum near β = 70° (mid-latitude). The integral formulae derived here remain the canonical starting point for pB inversion 70+ years later.

### 논문 #61과의 직접 연결 / Direct Connection to Paper #61
**Abbo et al. (2025)의 Eq. (1)** — Solar Orbiter Metis pB로부터 N_e(r)를 구하는 역변환 공식 — 은 본 논문의 **Eq. (17), (18), (19), (20)** 의 직계 후손이다. 특히 K_t - K_r (편광 밝기 pB)에 대한 식 (20)이 Abbo+25 Eq. (1)의 원형이다. #61을 깊이 이해하려면 §4(편광·전자밀도 계산)에 집중하라. / Abbo et al. (2025) Eq. (1) — the pB → N_e(r) inversion in Solar Orbiter Metis — is a direct descendant of **Eqs. (17)-(20) of this paper**. In particular, Eq. (20) for K_t − K_r (the polarized brightness pB) is the prototype of Abbo+25 Eq. (1). To deeply understand #61, focus on §4 (Polarization and Electron Densities).

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

#### 한국어
1950년 시점에 코로나 백색광 분광·측광은 이미 Schuster (1879), Minnaert (1930), Baumbach (1937), Grotrian (1934), Bergstrand (1919) 등의 작업으로 상당히 축적되어 있었다. 그러나 (i) F-corona 보정이 부정확했고, (ii) Baumbach의 전자 밀도 도출은 18%까지 오차가 있는 근사법에 의존했으며, (iii) 극지방·태양활동주기 변화는 거의 다루어지지 않았다. 라디오 천문학(파장 1-10 m)이 막 성장하면서 코로나 전자 밀도의 정밀한 값이 옵서버서리(opacity) 계산에 필수적으로 요구되기 시작했고, 이 수요가 본 논문의 직접적 동기가 되었다. van de Hulst는 1948년 3월 Lick Observatory를 방문해 1893-1932년 사이의 일식 사진건판들을 마이크로포토미터로 측정한 새 자료를 기반으로 작업했다.

#### English
By 1950, photometry of the white-light corona had accumulated substantial work by Schuster (1879), Minnaert (1930), Baumbach (1937), Grotrian (1934), and Bergstrand (1919). However: (i) F-corona corrections were inaccurate, (ii) Baumbach's electron-density derivation relied on an approximation good only to ~18%, and (iii) polar regions and the solar cycle were largely neglected. The growing field of radio astronomy (1-10 m wavelengths) urgently needed precise coronal electron densities for opacity calculations — this requirement directly motivated the present work. van de Hulst's analysis builds on a March 1948 visit to Lick Observatory, where he micro-photometered eclipse plates from 1893-1932.

### 타임라인 / Timeline

```
1879  Schuster      ── Theoretical polarization of K-corona by free electrons
1919  Bergstrand    ── Brightness distribution at 1914 eclipse (equator + polar)
1930  Minnaert      ── Comprehensive polarization theory
1934  Grotrian      ── F/K separation using Fraunhofer-line dilution
1937  Baumbach      ── Compilation of N_e(r); coronagraph era begins (Lyot 1930)
1939  Baumbach      ── Iterative correction for anisotropy
1947  Allen / van de Hulst  ── Improved photometric values
1948  van de Hulst @ Lick   ── Micro-photometry of 1893-1932 eclipse plates
═════════════════════════════════════════════════════════════════════════════
1950  ★ van de Hulst (this paper) ── Self-consistent "model corona"
                                      with 1% accuracy; pole vs. equator;
                                      pB-inversion equations (17)-(20)
═════════════════════════════════════════════════════════════════════════════
1961  van de Hulst ── "Light Scattering by Small Particles" (textbook)
1971  Saito         ── Refinement of equatorial/polar density model
1995  SOHO/LASCO    ── Spaceborne pB observations begin
2020  Solar Orbiter ── Metis coronagraph (pB + UV Lyα)
2025  Abbo+ (#61)   ── Metis pB inversion using van de Hulst Eq. (20) form
```

---

## 3. 필요한 배경 지식 / Prerequisites

### 한국어
- **Thomson 산란**: 자유 전자에 의한 비공명 탄성 산란. 단면적 σ_T = 6.65 × 10⁻²⁵ cm² (논문에서 σ로 표기). 산란된 빛은 산란각에 의존하는 편광을 가진다.
- **편광 천문학 기초**: 자연광이 비등방 전자 진동(접선 성분 ⊥ 시선, 시선 방향 성분)으로 산란될 때 t-성분과 r-성분의 강도 차이가 곧 선편광이 된다.
- **사선광 적분 (line-of-sight integration)**: 코로나의 면 밝기 K(x)는 시선 y를 따라 N(r)·산란커널을 적분한 양. 투영 거리 x와 3차원 거리 r = √(x² + y²)의 관계가 핵심.
- **Abel-type 적분 방정식**: K(x) → N(r) 역변환은 본질적으로 Abel 변환이며, 안정적인 수치 풀이를 위해 ∑ C_n r⁻ⁿ 같은 멱급수 표현을 사용.
- **limb darkening (변연감광)**: 태양 원반의 휘도 분포를 q (논문에서는 q = 0.75)로 매개화. 산란 함수 A(r), B(r)이 q에 의존.
- **F-corona vs. K-corona**: F는 행성간 먼지에 의한 회절광(Fraunhofer 선이 약화되지 않음), K는 자유전자 산란(Doppler 폭이 너무 커서 Fraunhofer 선이 완전 소멸).
- **헬리오그래픽 좌표**: β = 위도. β = 0°는 적도, β = 90°는 극.

### English
- **Thomson scattering**: Non-resonant elastic scattering by free electrons. Cross-section σ_T = 6.65 × 10⁻²⁵ cm² (denoted σ in the paper). The scattered radiation is polarized depending on scattering angle.
- **Polarimetry basics**: Natural light scattered by anisotropic electron oscillations (transverse vs. radial components) produces linearly polarized light proportional to the t/r intensity difference.
- **Line-of-sight integration**: The surface brightness K(x) is an integral of N(r) × scattering kernel along the line of sight y, with x = projected radius, r = √(x² + y²).
- **Abel-type integral equation**: The K(x) → N(r) inversion is essentially an Abel transform; stable numerical solution typically uses power-series expansions ∑ C_n r⁻ⁿ.
- **Limb darkening**: Solar-disk brightness distribution parameterized by q (van de Hulst uses q = 0.75 for ~4700 Å). The scattering functions A(r), B(r) depend on q.
- **F-corona vs. K-corona**: F = diffraction by interplanetary dust (Fraunhofer lines retained); K = scattering by free electrons (Fraunhofer lines obliterated by huge thermal Doppler broadening).
- **Heliographic coordinates**: β = latitude. β = 0° is the equator, β = 90° is the pole.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **K-corona** | 자유 전자에 의한 톰슨 산란이 만든 진짜 코로나 광. Fraunhofer 선이 보이지 않음. / The real corona, scattered by free electrons; Fraunhofer lines are absent. |
| **F-corona** | 행성간 먼지에 의한 회절광 (외부 황도광). Fraunhofer 선이 보존됨. / Diffraction-scattered light by interplanetary dust; Fraunhofer lines preserved. |
| **f = K/(F+K)** | 관측된 총 밝기 중 K-corona가 차지하는 비율. F/K 분리의 핵심 양. / The fraction of total brightness due to the K-corona; central quantity for F/K separation. |
| **K_t, K_r** | 산란광의 접선(transverse, ⊥ 시선·태양 평면) 및 시선방향(radial) 강도 성분. 두 성분의 차이가 곧 편광 밝기 pB. / Tangential and radial intensity components of scattered light; their difference is the polarized brightness pB. |
| **Polarized Brightness (pB)** | pB ≡ K_t − K_r. 편광 밝기. F-corona의 영향이 거의 없으므로 N_e 도출에 가장 깨끗한 양. / pB ≡ K_t − K_r. The polarized brightness; nearly free of F-corona contamination, hence the cleanest observable for N_e. |
| **A(r), B(r)** | 진동 타원체의 축. A는 접선 두 축, B는 시선방향(태양-점) 축에 대응. r→∞에서 B→1/2 r⁻². / Axes of the vibration ellipsoid: A for the two transverse axes, B for the radial axis. B → ½ r⁻² as r → ∞. |
| **Source function J(r)** | 단위 부피의 산란전 mean source function. K(x) = 2 ∫ J(v)·R/√(v²−x²) dv (Abel form). / Mean source function per unit volume; K(x) given by an Abel-type integral over J(v). |
| **Limb darkening coefficient q** | 태양원반 휘도 분포 매개변수. q=0.75가 가시광 (~4700 Å) 표준값. / Limb-darkening parameter; q = 0.75 is standard at ~4700 Å. |
| **Vibration ellipsoid** | 자유전자가 비등방 입사광에 대해 진동하는 모양. 축 비율이 산란광의 편광을 결정. / The shape of free-electron oscillations under anisotropic illumination; its axis ratio sets the scattered polarization. |
| **Model corona** | 본 논문이 산출한, 밝기·편광·전자밀도가 서로 1% 이내로 일치하는 적도/극, 최소/최대상의 표준 표 (Tables 2 & 5). / The self-consistent table of brightness, polarization, and electron density (Tables 2 & 5) for equatorial/polar regions in min/max phases. |
| **β (heliographic latitude)** | 헬리오그래픽 위도. β=0 적도, β=90° 극. / Heliographic latitude (0° equator, 90° pole). |

---

## 5. 수식 미리보기 / Equations Preview

### Eq. (1) — F/K 분리비 / F-K Separation Fraction

$$
f = \frac{K}{F + K}
$$

#### 한국어
관측된 총 표면 밝기 (F + K)에 곱해져 순수 K-corona 밝기를 주는 인자. f의 결정은 §2에서 (a) Fraunhofer 선 잔류 강도, (b) F-corona가 비편광이라는 가정, (c) F를 가정한 직접 차감의 세 가지 방법으로 다룬다. f는 시야 위치 (r, latitude)와 태양활동위상의 함수이다.

#### English
The factor by which the observed total brightness (F + K) must be multiplied to recover the pure K-corona brightness. f is determined in §2 by three methods: (a) residual intensity in Fraunhofer lines, (b) assumption that the F-corona is unpolarized, and (c) direct subtraction with an assumed F-model. f is a function of position (r, latitude) and solar-cycle phase.

---

### Eq. (17) — Total K-corona Surface Brightness Integral / 총 K-corona 표면 밝기 적분

$$
K(x) = C \int_x^{\infty} N(r)\,\Big\{(2 - \tfrac{x^2}{r^2})\,A(r) + \tfrac{x^2}{r^2}\,B(r)\Big\}\,\frac{r\,dr}{\sqrt{r^2 - x^2}}
$$

with $C = \tfrac{3}{4} \cdot 10^8 R \sigma = 3.44 \times 10^{-6}\,\text{cm}^3$ (Eq. 21).

#### 한국어
이 식이 본 논문의 핵심 적분 방정식이다. 좌변은 투영 반경 x에서의 K-corona 표면 밝기, 우변은 시선을 따른 N(r)의 가중 적분. 가중 함수 (2 − x²/r²)A + (x²/r²)B는 Thomson 산란의 비등방 위상 함수에서 유도된다. C에는 태양 반경 R = 6.97 × 10¹⁰ cm와 톰슨 단면적 σ가 포함되어 있다.

#### English
This is the central integral equation. The LHS is the K-corona surface brightness at projected radius x; the RHS is a line-of-sight weighted integral of N(r). The kernel (2 − x²/r²)A + (x²/r²)B comes from the anisotropic Thomson phase function. C absorbs the solar radius R = 6.97 × 10¹⁰ cm and the Thomson cross-section σ.

---

### Eq. (18) & (19) — Tangential and Radial Components / 접선·시선 방향 성분

$$
K_t(x) = C \int_x^{\infty} N(r)\,A(r)\,\frac{r\,dr}{\sqrt{r^2 - x^2}}\quad\text{(18)}
$$

$$
K_r(x) = C \int_x^{\infty} N(r)\,\Big\{(1 - \tfrac{x^2}{r^2})\,A(r) + \tfrac{x^2}{r^2}\,B(r)\Big\}\,\frac{r\,dr}{\sqrt{r^2 - x^2}}\quad\text{(19)}
$$

#### 한국어
표면 밝기를 산란광의 두 편광 성분 — 접선 K_t와 시선방향 K_r — 으로 분해한 식. 합 K_t + K_r = K (Eq. 17), 차 K_t − K_r ≡ pB (편광 밝기, Eq. 20). 관측은 통상 K_t와 K_r을 독립적으로 측정한다.

#### English
Decomposition of the surface brightness into two polarization components: tangential K_t and radial K_r. Their sum equals K (Eq. 17); their difference is the polarized brightness pB (Eq. 20). Observations typically measure K_t and K_r independently.

---

### Eq. (20) — Polarized Brightness pB / 편광 밝기 ★

$$
K_t(x) - K_r(x) = C \int_x^{\infty} N(r)\,\big[A(r) - B(r)\big]\,\frac{x^2\,dr}{r\,\sqrt{r^2 - x^2}}
$$

#### 한국어
**★ 본 논문의 가장 중요한 식 — 그리고 #61의 Eq. (1)이 직접 유래하는 식.** F-corona는 비편광으로 가정되므로 좌변은 F 보정 없이 관측 가능한 가장 깨끗한 양이다. 우변에는 N(r)이 직접 들어 있고, 적분 핵 [A(r) − B(r)]·x²/r 은 r → x에서 (1/√(r²−x²))로 발산하는 Abel-form. 이를 풀면 N(r)이 결정된다.

#### English
**★ The single most important equation of the paper — and the direct ancestor of Paper #61's Eq. (1).** Since the F-corona is assumed unpolarized, the LHS is the cleanest observable (no F-correction needed). The RHS contains N(r) directly, with the Abel-form kernel [A(r) − B(r)]·x²/r that diverges as 1/√(r²−x²) at r → x. Inverting this Abel-type equation determines N(r).

#### Modern Form Used in Abbo et al. (2025), Eq. (1)
The same physics is expressed in modern coronagraph literature as:

$$
pB(\rho) = \frac{3 \sigma_T}{16}\,I_\odot \int_\rho^{\infty} N_e(r)\,\frac{(1 - u + u\sqrt{1 - 1/r^2})\,(1 - 1/r^2)\,r\,dr}{\sqrt{r^2 - \rho^2}}
$$

where ρ = projected radius (van de Hulst's x), u = limb-darkening coefficient (≈ 1 − q in older notation). Comparing terms:
- (3 σ_T / 16)·I_⊙ ↔ van de Hulst's prefactor C
- The (1 − 1/r²) factor ↔ van de Hulst's [A(r) − B(r)]·x²/r² with appropriate substitutions
- u ↔ limb-darkening parameter (van de Hulst's q)

#### 한국어
같은 물리를 현대 코로나그래프 문헌에서는 위 식으로 표현한다. ρ = 투영 반경 (van de Hulst의 x), u = limb darkening coefficient (구 표기에서 ≈ 1 − q). 항별 대응: (3σ_T/16)I_⊙ ↔ van de Hulst의 C, (1 − 1/r²) ↔ [A − B]·x²/r² 의 등가 변형, u ↔ q.

---

### Eq. (29)-(30) — Polar Region Reduction / 극지방 환산 (Section 5)

$$
K(x) = 2 \int_x^{\infty} J(v)\,\frac{v\,dv}{\sqrt{v^2 - x^2}}\quad\text{(29)}
$$

$$
P(x) = -\frac{1}{x}\,\frac{dK(x)}{dx}; \qquad J(v) = \frac{1}{\pi} \int_v^{\infty} P(x)\,\frac{x\,dx}{\sqrt{x^2 - v^2}}\quad\text{(30)}
$$

#### 한국어
극지방은 구 대칭이 성립하지 않으므로 적도용 식 (17)-(20)을 그대로 쓸 수 없다. van de Hulst는 등방 산란 가정 하에 단순화된 적분 방정식 (29)를 풀고, Abel 역변환 (30)으로 source function J(v)를 얻는 우회 절차를 사용한다. 이후 Eq. (33)-(36)으로 비등방 보정을 가해 N(r)을 30%까지 증가시킨다.

#### English
Spherical symmetry fails over the poles, so Eqs. (17)-(20) cannot be used directly. van de Hulst takes a detour: assume isotropic scattering, solve the simplified Eq. (29), invert via the Abel pair (30) to obtain the source function J(v). Anisotropy is then restored via Eqs. (33)-(36), which raises N(r) by ~30% near r = 1.1.

---

## 6. 읽기 가이드 / Reading Guide

### 한국어 — 추천 읽기 순서

논문은 6개의 절로 구성되어 있다. #61 (Abbo+25) 이해가 주목적이라면 **§4를 가장 깊이 읽는다**.

1. **§1 Introduction (pp. 135-136)** — 코로나 빛의 세 성분(방출선 0.5%, K-corona 연속체, F-corona 외부 헤이즈), 라디오 천문학 동기, model corona 개념 소개. 가벼운 마음으로 통독.

2. **§2 Separation of F- and K- (pp. 136-137)** — F/K 분리의 세 가지 방법(Eq. 1-3). pB가 왜 N_e 측정에 가장 좋은 양인지 (방법 2 = 식 3 참고) 이해하라.

3. **§3 Brightness Distribution (pp. 137-141)** — Lick 1893-1932 일식 자료 분석. Bergstrand 1914 일식과 Lick 1900 사진건판이 주요 자료. Eq. (5)-(9)의 ∑ C_n r⁻ⁿ 멱급수 표현이 §4 적분 풀이의 입력이다. **Table 1, Table 2를 외워둘 필요는 없으나 멱급수 형식은 기억하라.**

4. **§4 Polarization and Electron Densities (pp. 141-144)** — ★ **가장 중요한 절. #61 Eq. (1)의 발원지.**
   - p. 141: 진동 타원체와 A(r), B(r) 함수 정의 → Eq. (11), (12)
   - p. 142: 산란 적분 유도 → Eq. (13)-(15)
   - **Eq. (17)-(20): 핵심 적분 방정식. #61과 직접 비교하라.**
   - p. 143-144: 멱급수 + 연속 근사로 N(r)을 푸는 절차. **수식 (22)-(28)은 한 번 읽고 의도만 파악하면 충분.**
   - **Table 5A, 5B: 적도/극지 N(r) 값 — 70년간 표준 참고 데이터.**

5. **§5 Polar Density vs. Latitude (pp. 145-147)** — Lick 1900 자료의 마이크로포토미터링. Eq. (29)-(36)의 우회 풀이. **Figure 7B의 β = 70° 부근 minimum이 발견(highlight).** 흥미로운 천체물리적 결과.

6. **§6 Comparison with Observed Polarization (pp. 147-149)** — 관측 편광 데이터와 모델 비교 (Figure 8). 외부 코로나에서 모델이 관측보다 낮게 나오는 불일치 논의 → sky light 과대보정, F-corona 자체 편광, F 과대평가의 세 설명.

### 한국어 — 시간이 부족할 때
- §1 첫 페이지 + §4 (특히 Eq. 17-20) + Table 5A 행 몇 개만 봐도 #61 이해에 충분.

### English — Recommended Reading Order

The paper has six sections. If your goal is understanding #61 (Abbo+25), **focus deepest on §4**.

1. **§1 Introduction (pp. 135-136)** — Three components of coronal light (0.5% emission lines; K-corona continuum; F-corona haze); motivation from radio astronomy; introduction of the "model corona" concept. Read once, lightly.

2. **§2 Separation of F- and K- (pp. 136-137)** — Three methods for F/K separation (Eqs. 1-3). Understand *why* pB is the cleanest probe of N_e (see method 2 = Eq. 3).

3. **§3 Brightness Distribution (pp. 137-141)** — Analysis of Lick 1893-1932 eclipse data, with Bergstrand 1914 and Lick 1900 plates as key inputs. The ∑ C_n r⁻ⁿ power-series fits in Eqs. (5)-(9) become the input to §4's integral inversion. **You don't need to memorize Tables 1-2, but remember the power-series form.**

4. **§4 Polarization and Electron Densities (pp. 141-144)** — ★ **The most important section. The origin of #61's Eq. (1).**
   - p. 141: vibration ellipsoid and the A(r), B(r) functions → Eqs. (11), (12)
   - p. 142: scattering integral derivation → Eqs. (13)-(15)
   - **Eqs. (17)-(20): the core integral equations. Directly compare with #61.**
   - pp. 143-144: solving for N(r) via power-series + successive approximation. **Eqs. (22)-(28) can be skimmed for intent only.**
   - **Tables 5A, 5B: equatorial/polar N(r) — the reference values for 70+ years.**

5. **§5 Polar Density vs. Latitude (pp. 145-147)** — Micro-photometry of Lick 1900 plates. The detour via Eqs. (29)-(36). **Highlight: the discovery of the N_e minimum near β = 70° (Figure 7B).** An interesting astrophysical finding.

6. **§6 Comparison with Observed Polarization (pp. 147-149)** — Model vs. observed polarization (Figure 8). Discussion of the discrepancy in the outer corona, with three candidate explanations: sky-light over-correction, F-corona self-polarization, F overestimation.

### English — Quick-Read Path
- §1 first page + §4 (especially Eqs. 17-20) + a few rows of Table 5A is enough for understanding #61.

---

## 7. 현대적 의의 / Modern Significance

### 한국어
이 논문이 제시한 식 (17)-(20), 특히 편광 밝기 pB = K_t − K_r 에 대한 식 (20)은 **현재까지도 코로나그래프 pB 관측으로부터 N_e(r)를 도출하는 표준 역변환 (pB inversion)의 출발점**이다. 직접 응용된 사례:

- **Saito (1971)**: van de Hulst의 적도/극지 모델을 정밀화한 분석적 fit.
- **SOHO/LASCO (1995-)**: K-Cor 관측 → pB 합성 → N_e(r) 산출의 모든 파이프라인이 본 논문 Eq. (20) 형식을 사용.
- **STEREO/COR1, COR2**: 동일 원리로 3D 코로나 N_e 재구성.
- **Solar Orbiter Metis (2020-)**: pB와 UV Lyα 동시 관측. **#61 (Abbo+ 2025) Eq. (1)이 본 논문 Eq. (20)의 현대 표기이다.** Metis는 streamer 내부 N_e(r)을 도출하기 위해 본 논문이 개발한 인버전을 그대로 적용한다.
- **PUNCH (2025)**: NASA의 새 가시광 코로나/태양풍 미션 — 동일한 pB 인버전 형식 사용.

또한 본 논문의 **β = 70° 부근 N_e 최소** 발견은 코로나 streamer belt와 polar coronal hole 사이의 전이 영역(streamer 갭)에 대한 최초의 광학적 증거이며, 이후 Skylab/SMM/SOHO의 EUV 관측으로 재확인되었다.

van de Hulst는 1년 후인 1951년 코로나의 21 cm 라디오 방출 예측으로 또 한 번 천문학사를 바꾸지만(중성수소 21cm선 예측은 1944년), 본 1950 논문은 가시광·편광 측광에서의 그의 표준화 업적으로 남는다.

### English
Equations (17)-(20) of this paper — particularly Eq. (20) for pB = K_t − K_r — remain **the standard starting point for inverting coronagraph pB observations into electron density N_e(r) today**. Direct applications:

- **Saito (1971)**: refined analytical fits to van de Hulst's equatorial/polar models.
- **SOHO/LASCO (1995-)**: every pipeline turning K-Cor observations into pB and then N_e(r) uses the form of Eq. (20).
- **STEREO/COR1, COR2**: same principle for 3D coronal N_e reconstruction.
- **Solar Orbiter Metis (2020-)**: simultaneous pB and UV Lyα. **Eq. (1) of #61 (Abbo+ 2025) is the modern notation of this paper's Eq. (20).** Metis applies van de Hulst's inversion directly to derive N_e(r) inside streamers.
- **PUNCH (2025)**: NASA's new visible-light corona/heliosphere mission — same pB-inversion formalism.

Additionally, the discovery of the **N_e minimum near β = 70°** is the earliest optical evidence for the transition zone (streamer gap) between the equatorial streamer belt and the polar coronal holes, later confirmed by EUV observations from Skylab/SMM/SOHO.

van de Hulst would change astronomy again with the 21-cm hydrogen line prediction (1944) and his 1957 monograph "Light Scattering by Small Particles", but among solar-physics observers, this 1950 paper remains his standardizing legacy in white-light coronal photometry.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
