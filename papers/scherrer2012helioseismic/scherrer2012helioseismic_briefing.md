---
title: "Pre-Reading Briefing: The Helioseismic and Magnetic Imager (HMI) Investigation for SDO"
paper_id: "13_scherrer_2012"
topic: Solar Observation
date: 2026-04-16
type: briefing
---

# The Helioseismic and Magnetic Imager (HMI) on SDO — Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Scherrer, P.H., Schou, J., Bush, R.I., et al. (2012). "The Helioseismic and Magnetic Imager (HMI) Investigation for the Solar Dynamics Observatory (SDO)." *Solar Physics*, Vol. 275, pp. 207–227.
**Author(s)**: Philip H. Scherrer (PI, Stanford) + 30+ co-authors
**Year**: 2012

---

## 1. 핵심 기여 / Core Contribution

이 논문은 SDO에 탑재된 **HMI(Helioseismic and Magnetic Imager)**의 설계, 과학 목표, 관측 방법을 기술합니다. HMI는 **Fe I 6173 Å** 흡수선의 편광 관측을 통해 태양 전면의 **도플러 속도, 시선 자기장(LOS), 벡터 자기장**을 측정합니다. SOHO/MDI의 직접적 후계자로, MDI의 1024² CCD/4″ 분해능을 4096² CCD/1″ 분해능으로 업그레이드하고, MDI에 없었던 **벡터 자기장 측정(full Stokes)**을 추가했습니다. HMI의 핵심 광학 설계는 **Lyot 필터 + Michelson 간섭계** 조합으로, 76 mÅ 대역폭을 달성합니다.

This paper describes the design, science goals, and observational methods of **HMI (Helioseismic and Magnetic Imager)** on SDO. HMI measures **Doppler velocity, line-of-sight magnetic field, and vector magnetic field** across the full solar disk through polarimetric observations of the **Fe I 6173 Å** absorption line. As the direct successor to SOHO/MDI, it upgrades from 1024²/4″ to 4096²/1″ and adds **full Stokes vector magnetography**. The optical design combines a **Lyot filter + Michelson interferometer** to achieve 76 mÅ bandpass.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

태양 자기장 측정은 태양 물리학의 근본적 관측입니다. 광구 자기장의 시간적 변화가 플레어, CME, 태양풍을 구동하므로, 전일면 자기장의 연속 관측이 핵심입니다.

HMI 이전의 주요 자기장 측정 기기:
- **GONG (#5)**: 지상 6개소 네트워크, 도플러/자기장, 2.5″/pixel, ~80% duty cycle
- **SOHO/MDI**: 우주, 1024², 4″(전면)/1.25″(고분해능 650″ FOV), Ni I 6768 Å, LOS 자기장만
- **Hinode/SOT (2006)**: 우주, 0.3″ 분해능, 벡터 자기장, 하지만 218″×109″ 소시야

HMI는 "전일면 + 고분해능 + 벡터 자기장 + 연속 관측"을 모두 달성하는 최초의 기기입니다.

### SOHO/MDI → SDO/HMI 비교

| 항목 | MDI (1995) | HMI (2010) |
|------|-----------|------------|
| 파장 | Ni I 6768 Å | Fe I 6173 Å |
| CCD | 1024² | 4096² |
| 전면 분해능 | 4″/pixel | 1″/pixel (0.505″) |
| 자기장 | LOS only | LOS + Vector (full Stokes) |
| 도플러 케이던스 | 60초 | 45초 |
| 벡터 자기장 | 불가 | 135초 (720초 for noise) |
| 필터 | Michelson × 2 | Lyot + Michelson × 2 |
| 텔레메트리 | 5 kbit/s (+160 고속) | ~55 Mbit/s (SDO 내) |

### 타임라인 / Timeline

```
1995  ── SOHO/MDI 발사 [이 시리즈 향후] — Ni I 6768 Å, LOS 자기장
         │
1996  ── GONG [#5] — 지상 일진학 네트워크
         │
2006  ── Hinode/SOT — 0.3" 벡터 자기장 (소시야)
         │
2010  ── ★ SDO/HMI 발사 (Feb 11) ★
      │  Fe I 6173 Å, 4096², 벡터 자기장
         │
2012  ── ★ Scherrer et al.: HMI 기기 논문 출판 ★
         │
2025  ── HMI 15년째 운용 중
```

---

## 3. 필요한 배경 지식 / Prerequisites

### 3.1 Zeeman 효과와 자기장 측정 / Zeeman Effect and Magnetic Field Measurement

외부 자기장에 놓인 원자의 스펙트럼 선이 분리됩니다:

$$\Delta\lambda_Z = \frac{e\lambda^2 g B}{4\pi m_e c} = 4.67 \times 10^{-13} \lambda^2 g B \quad \text{(nm)}$$

HMI의 Fe I 6173 Å ($g_{\text{eff}} = 2.5$, 높은 Landé 인자)에서:
- $B = 1000$ G → $\Delta\lambda_Z \approx 44$ mÅ (HMI 대역폭 76 mÅ와 비교 가능)
- $B = 100$ G → $\Delta\lambda_Z \approx 4.4$ mÅ (직접 분리 불가 → Stokes V 측정)

### 3.2 Stokes 편광 / Stokes Polarimetry

자기장의 완전한 정보는 **Stokes 벡터 (I, Q, U, V)**에 담겨 있습니다:
- **I**: 총 강도
- **Q, U**: 선편광 (횡방향 자기장 $B_\perp$에 민감)
- **V**: 원편광 (시선 방향 자기장 $B_\parallel$에 민감)

LOS 자기장은 V/I만으로 측정 가능하지만, 벡터 자기장은 I, Q, U, V 모두 필요합니다.

### 3.3 Lyot 필터 + Michelson 간섭계 / Lyot Filter + Michelson Interferometer

HMI는 두 단계 필터링을 사용합니다:
1. **Lyot 필터**: 복굴절 결정으로 ~4 Å 예비 필터링 (넓은 대역 제거)
2. **Michelson 간섭계 2개**: 추가 좁힘 → 최종 76 mÅ 대역폭

Fe I 6173 Å 선 프로파일을 6개 파장 위치에서 샘플링하여 도플러 이동과 자기장을 추출합니다.

### 3.4 이전 논문과의 연결

- **#5 GONG**: HMI의 지상 보완 기기. GONG은 원거리 측(far-side) 일진학 제공.
- **#8 SOHO**: MDI가 SOHO의 일진학 기기. HMI는 MDI의 직접 후계.
- **#12 AIA**: HMI와 AIA가 SDO의 핵심 쌍 — 자기장(HMI) + 코로나 응답(AIA).
- **#35 SDO**: SDO 미션 개요. HMI는 SDO 3개 기기 중 하나.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **Dopplergram** | 태양 표면의 시선 방향 속도 지도. p-모드 진동 관측의 기본 데이터. / Map of line-of-sight velocity. Basic data for p-mode observation. |
| **Magnetogram** | 태양 표면의 자기장 지도. LOS (시선 성분) 또는 Vector (3성분). / Map of magnetic field. LOS (line-of-sight) or Vector (3 components). |
| **Stokes I, Q, U, V** | 빛의 편광 상태를 기술하는 4개 파라미터. V=원편광(LOS B), Q/U=선편광(횡 B). / 4 parameters describing light polarization state. |
| **Lyot filter** | 복굴절 결정과 편광기를 교대 배치한 대역 필터. / Birefringent filter with alternating crystals and polarizers. |
| **Michelson interferometer** | 빔 분할기로 두 경로를 만들어 간섭시키는 장치. 파장 선택에 사용. / Beam splitter creating two paths for interference. Used for wavelength selection. |
| **Fe I 6173 Å** | HMI가 관측하는 광구 흡수선. Landé g=2.5로 자기장 민감도가 높음. / Photospheric absorption line observed by HMI. High magnetic sensitivity (g=2.5). |
| **Vector magnetogram** | 자기장의 3성분(Bx, By, Bz)을 모두 측정한 자기도. Full Stokes 관측 필요. / Magnetic field map with all 3 components. Requires full Stokes observation. |
| **Observables** | HMI가 생산하는 기본 물리량: 도플러그램, 자기도, 강도, 선폭 등. / Basic physical quantities produced by HMI. |
| **Disambiguation** | 180° 방위각 모호성 해소. 벡터 자기도에서 횡방향 자기장의 방향을 결정하는 과정. / Resolving the 180° azimuthal ambiguity in transverse magnetic field direction. |

---

## 5. 수식 미리보기 / Equations Preview

### 5.1 Zeeman 분리

$$\Delta\lambda_Z = 4.67 \times 10^{-13} \lambda^2 g_{\text{eff}} B$$

Fe I 6173 Å, $g_{\text{eff}} = 2.5$: $\Delta\lambda_Z = 4.44 \times 10^{-2} B$ [mÅ, B in Gauss]

### 5.2 Stokes 벡터에서 자기장 추출

약한 자기장 근사 (weak-field approximation):
$$V(\lambda) \approx -C g_{\text{eff}} \lambda^2 B_\parallel \frac{dI}{d\lambda}$$

$$B_\parallel \propto \frac{V}{dI/d\lambda}$$

### 5.3 HMI 분광 분해능

$$\Delta\lambda_{\text{HMI}} = 76 \text{ mÅ} \quad \text{(FWHM of combined Lyot + Michelson)}$$

6개 파장 위치 간격: 69 mÅ → 선 프로파일의 414 mÅ 범위를 샘플링

---

## 6. 읽기 가이드 / Reading Guide

약 21페이지의 기기+과학 논문:

1. **§1 Introduction**: 과학 목표 — 일진학, 자기장, 우주 날씨
2. **§2 Science Goals**: 상세 과학 질문
3. **§3 Instrument**: 광학 설계 (Lyot + Michelson), CCD, 편광계
4. **§4 Observables**: 도플러그램, 자기도 생산 과정
5. **§5 Data Products**: Level 0→1→1.5→2, 파이프라인

**읽기 전략**: §3(기기)과 §4(관측량 추출)가 핵심. Lyot+Michelson 조합의 작동 원리와 6-point 파장 샘플링에서 도플러/자기장을 추출하는 방법에 집중.

---

## 7. 현대적 의의 / Modern Significance

HMI는 현재 태양 자기장 연구의 "표준 데이터"입니다:

1. **벡터 자기장 지도**: 전일면 벡터 자기도를 45초 케이던스로 연속 제공 — 플레어/CME 예보의 핵심 입력.
2. **일진학**: MDI보다 4배 높은 분해능으로 태양 내부 구조를 더 정밀하게 탐사.
3. **우주 날씨**: NOAA SWPC가 HMI 자기도를 실시간으로 사용하여 자기장 활동 예보.
4. **AIA와의 시너지**: HMI 자기장 + AIA 코로나 영상 = 자기장 구조와 코로나 응답의 직접적 연결.

---

## Q&A

### Q1: 벡터 자기장 케이던스가 논문에서 135초인데 실제 데이터 제품은 720초인 이유

**135초** = 단일 완전 Stokes (I,Q,U,V) 사이클 취득 시간 (6 wavelengths × 6 polarizations = 36 filtergrams × 3.75s). **720초** = 과학 데이터 제품 케이던스 — ~5개 Stokes 사이클을 시간 평균하여 SNR 개선 후 Milne-Eddington 반전(VFISV) + 180° disambiguation을 수행한 결과.

핵심 이유: Stokes Q/U (횡방향 자기장)가 $B_\perp^2$에 비례하여 약한 자기장에서 SNR이 급격히 떨어지므로, 시간 평균이 필수. LOS 자기도(Stokes V)는 B에 선형 비례하여 45초로도 충분한 SNR 달성.

135s cadence = single complete Stokes cycle acquisition time. 720s cadence = science data product after time-averaging ~5 Stokes cycles to improve SNR + running VFISV Milne-Eddington inversion + 180° disambiguation. Key reason: Stokes Q/U ∝ B⊥² has much lower SNR than Stokes V ∝ B∥ for weak fields.

### Q2: Stokes 파라미터 상세, 측정 방법, 벡터 자기장 도출

#### 2-1. Stokes 파라미터란? / What are Stokes Parameters?

빛의 편광 상태를 강도 측정만으로 완전히 기술하는 4개의 양:

| Stokes | 의미 / Meaning | 측정 개념 / Measurement Concept |
|--------|---------------|-------------------------------|
| **I** | 총 강도 (Total intensity) | 편광 필터 없이 측정 |
| **Q** | 0°/90° 선편광 차이 | $Q = I_{0°} - I_{90°}$ |
| **U** | 45°/135° 선편광 차이 | $U = I_{45°} - I_{135°}$ |
| **V** | 우/좌 원편광 차이 | $V = I_{\text{RCP}} - I_{\text{LCP}}$ |

- **I** = "빛이 얼마나 밝은가" / How bright is the light
- **Q** = "수평-수직 중 어느 방향으로 더 편광되었는가" / Horizontal vs vertical polarization
- **U** = "대각선 중 어느 방향으로 더 편광되었는가" / Diagonal polarization
- **V** = "오른쪽-왼쪽 중 어느 방향으로 더 원편광되었는가" / Right vs left circular polarization
- 완전 비편광(unpolarized): $Q = U = V = 0$, $I > 0$

#### 2-2. 자기장이 Stokes에 미치는 영향 / How Magnetic Fields Affect Stokes

태양 흡수선(Fe I 6173 Å)이 자기장 속에서 Zeeman 효과로 분리될 때, 세 성분($\sigma^+$, $\pi$, $\sigma^-$)이 나타남:

**시선 방향 자기장 ($B_\parallel$, longitudinal) → Stokes V:**

$\sigma^+$ → 우원편광(RCP), $\sigma^-$ → 좌원편광(LCP), $\pi$ → 관측되지 않음

$$V(\lambda) \propto -g_{\text{eff}} \lambda^2 B_\parallel \frac{dI}{d\lambda}$$

V 프로파일은 반대칭(antisymmetric) 형태. 진폭이 $B_\parallel$에 **선형** 비례.

V profile is antisymmetric around line center. Amplitude is **linearly** proportional to $B_\parallel$.

**횡방향 자기장 ($B_\perp$, transverse) → Stokes Q, U:**

$\sigma^{\pm}$ → 자기장에 수직 방향 선편광, $\pi$ → 자기장에 평행 방향 선편광

$$Q(\lambda) \propto g_{\text{eff}}^2 \lambda^4 B_\perp^2 \sin^2\gamma \cos 2\phi \cdot \frac{d^2I}{d\lambda^2}$$

$$U(\lambda) \propto g_{\text{eff}}^2 \lambda^4 B_\perp^2 \sin^2\gamma \sin 2\phi \cdot \frac{d^2I}{d\lambda^2}$$

$\gamma$ = 경사각(inclination, 시선과 B 사이 각도), $\phi$ = 방위각(azimuth).

**핵심 차이 / Key difference**: V ∝ $B$ (선형/linear), Q/U ∝ $B^2$ (이차/quadratic) → Q/U 신호가 약한 자기장에서 훨씬 약함 → 720초 평균이 필요한 근본 원인.

#### 2-3. HMI의 Stokes 측정 방법 / How HMI Measures Stokes

**편광 변조 시스템**: 회전 파장판(rotating waveplate)으로 입사광의 편광 상태를 순차적으로 변환:

태양빛 → [회전 파장판 (6개 각도)] → [편광 빔 분할기] → CCD

6개의 다른 파장판 각도에서 측정한 강도 $I_1, I_2, ..., I_6$의 선형 조합으로 Stokes 벡터 추출:

$$\begin{pmatrix} I \\ Q \\ U \\ V \end{pmatrix} = \mathbf{D}^{-1} \begin{pmatrix} I_1 \\ I_2 \\ I_3 \\ I_4 \\ I_5 \\ I_6 \end{pmatrix}$$

$\mathbf{D}$ = 복조 행렬(demodulation matrix), 각 파장판 각도에서의 편광 응답을 기술.

$\mathbf{D}$ = demodulation matrix describing polarization response at each waveplate angle.

**HMI 관측 시퀀스 (1 Stokes 사이클 = 135초)**:

- 6개 파장 위치 × 6개 편광 상태 = **36 filtergrams**
- 각 filtergram ~3.75초
- 결과: 각 파장 위치에서 **I(λ), Q(λ), U(λ), V(λ) 프로파일**

#### 2-4. Stokes → 벡터 자기장 도출 / Deriving Vector Magnetic Field from Stokes

**방법 1: 약한 자기장 근사 (Weak-Field Approximation) — LOS 자기도 (45초)**

Zeeman 분리 ≪ 열적 선폭일 때:

$$B_\parallel = -\frac{1}{C \cdot g_{\text{eff}} \cdot \lambda^2} \cdot \frac{V}{dI/d\lambda}$$

V/I만 필요 → 빠르고 SNR 좋음. 45초 케이던스 LOS 자기도의 기본 알고리즘.

Only V/I needed → fast, good SNR. Basic algorithm for 45s cadence LOS magnetograms.

**방법 2: Milne-Eddington 반전 (ME Inversion) — 벡터 자기도 (720초)**

HMI는 **VFISV (Very Fast Inversion of the Stokes Vector)** 코드를 사용. Milne-Eddington 대기 가정 하에 Stokes 프로파일을 해석적으로 계산하는 순방향 모델:

$$\begin{pmatrix} I \\ Q \\ U \\ V \end{pmatrix}(\lambda) = f\left(B, \gamma, \phi, v_{\text{LOS}}, \eta_0, \Delta\lambda_D, a, S_0, S_1\right)$$

9개 자유 파라미터 / 9 free parameters:

| 파라미터 / Parameter | 의미 / Meaning |
|---------------------|---------------|
| $B$ | 자기장 세기 / Magnetic field strength (Gauss) |
| $\gamma$ | 경사각 / Inclination (angle between LOS and B, 0°=LOS) |
| $\phi$ | 방위각 / Azimuth (direction of B in sky plane) |
| $v_{\text{LOS}}$ | 시선 속도 / Line-of-sight velocity (Doppler shift) |
| $\eta_0$ | 선/연속체 흡수 비 / Line-to-continuum absorption ratio |
| $\Delta\lambda_D$ | 도플러 폭 / Doppler width |
| $a$ | 감쇠 파라미터 / Damping parameter (Voigt profile) |
| $S_0, S_1$ | 원천 함수 / Source function (linear approximation) |

역문제 — 관측된 Stokes 프로파일에 최적 맞춤 / Inverse problem — best fit to observed Stokes profiles:

$$\chi^2 = \sum_{i=\text{I,Q,U,V}} \sum_{j=1}^{6} \frac{\left[S_i^{\text{obs}}(\lambda_j) - S_i^{\text{model}}(\lambda_j; B, \gamma, \phi, ...)\right]^2}{\sigma_i^2}$$

$\chi^2$를 최소화하는 $(B, \gamma, \phi)$가 벡터 자기장.

The $(B, \gamma, \phi)$ that minimizes $\chi^2$ gives the vector magnetic field.

**180° 모호성 해소 (Disambiguation)**

$Q \propto B_\perp^2 \cos 2\phi$, $U \propto B_\perp^2 \sin 2\phi$ → $\cos 2\phi = \cos 2(\phi + 180°)$이므로 방위각에 본질적 180° 모호성 존재.

Since $\cos 2\phi = \cos 2(\phi + 180°)$, there is an intrinsic 180° ambiguity in azimuth.

HMI 해소 알고리즘: "minimum energy" — 인접 픽셀 간 자기장이 매끄럽게 변하도록 $\phi$ 선택, 전류 밀도 $|\nabla \times \mathbf{B}|$와 자기장 발산 $|\nabla \cdot \mathbf{B}|$을 최소화.

HMI disambiguation: "minimum energy" algorithm — choose $\phi$ so that field varies smoothly between adjacent pixels, minimizing $|\nabla \times \mathbf{B}|$ and $|\nabla \cdot \mathbf{B}|$.

**최종 벡터 자기장 성분 / Final vector field components:**

- $B_x = B \sin\gamma \cos\phi$ (동-서 / East-West)
- $B_y = B \sin\gamma \sin\phi$ (남-북 / North-South)
- $B_z = B \cos\gamma$ (시선 / Line-of-sight, = LOS magnetogram)

#### 2-5. 전체 파이프라인 요약 / Full Pipeline Summary

태양빛 (Fe I 6173 Å) → HMI 편광 변조 + 파장 스캔 (36 filtergrams / 135초) → Stokes 복조 [I(λ), Q(λ), U(λ), V(λ) at 6 wavelengths]

Solar light (Fe I 6173 A) → HMI polarization modulation + wavelength scan (36 filtergrams / 135s) → Stokes demodulation [I(λ), Q(λ), U(λ), V(λ) at 6 wavelengths]

경로 1 (LOS, 45초): 약한 자기장 근사 $B_\parallel \propto V/(dI/d\lambda)$ → LOS 자기도 (Level 1.5)

Path 1 (LOS, 45s): Weak-field approx $B_\parallel \propto V/(dI/d\lambda)$ → LOS magnetogram (Level 1.5)

경로 2 (벡터, 720초): 720초 평균 → VFISV ME 반전 → $(B, \gamma, \phi)$ → 180° Disambiguation → 벡터 자기도 $(B_x, B_y, B_z)$ (Level 2)

Path 2 (Vector, 720s): 720s average → VFISV ME inversion → $(B, \gamma, \phi)$ → 180° Disambiguation → Vector magnetogram $(B_x, B_y, B_z)$ (Level 2)
