# Pre-reading Briefing: Coronal Waves and Oscillations
# 사전 읽기 브리핑: 코로나 파동과 진동

**Paper**: Nakariakov, V. M. & Verwichte, E. (2005)
**Journal**: *Living Reviews in Solar Physics*, **2**, 3
**DOI**: 10.12942/lrsp-2005-3

---

## 핵심 기여 / Core Contribution

이 리뷰 논문은 태양 코로나에서 관측되는 다양한 MHD(자기유체역학) 파동과 진동 현상을 체계적으로 정리한 최초의 포괄적 리뷰입니다. SOHO와 TRACE 우주선의 혁신적 관측으로 발견된 코로나 루프의 kink 진동, sausage 진동, 음향 진동, 전파하는 slow/fast 파동 등을 MHD 이론의 틀에서 해석하고, 이러한 파동을 이용하여 직접 측정이 어려운 코로나 물리량(자기장, 수송 계수 등)을 진단하는 **MHD coronal seismology(MHD 코로나 지진학)** 기법의 기초를 확립합니다.

This review is the first comprehensive survey of MHD wave and oscillatory phenomena observed in the solar corona. It systematically classifies coronal loop kink oscillations, sausage oscillations, acoustic oscillations, and propagating slow/fast waves discovered by SOHO and TRACE, interprets them within MHD theory, and establishes the foundations of **MHD coronal seismology** — a technique to diagnose otherwise unmeasurable coronal parameters (magnetic field, transport coefficients) from wave properties.

---

## 역사적 맥락 / Historical Context

```
1970  Uchida — MHD coronal seismology 개념 최초 제안
          First proposal of MHD coronal seismology concept
1975  Zaitsev & Stepanov — leaky mode 이론
          Leaky mode theory
1981–84  Roberts et al. — 자기 실린더의 MHD 모드 분산 관계 확립
          Established MHD mode dispersion relations for magnetic cylinder
1983  Edwin & Roberts — 완전한 분산 다이어그램 도출
          Complete dispersion diagram derived
1983  Heyvaerts & Priest — Alfvén 파동 위상 혼합 제안
          Alfvén wave phase mixing proposed
1995  SOHO 발사 / SOHO launched
1998  TRACE 발사 → 코로나 루프 kink 진동 최초 관측 (Aschwanden, Nakariakov)
          TRACE launched → First observation of coronal loop kink oscillations
1999  Nakariakov et al. — kink 진동의 정량적 분석, 감쇠 특성 확인
          Quantitative analysis of kink oscillations, damping identified
2001  Nakariakov & Ofman — kink 진동으로 코로나 자기장 최초 추정 (~13 G)
          First coronal magnetic field estimate from kink oscillations (~13 G)
2002  Kliem, Wang et al. — SUMER으로 종방향 음향 진동 발견
          Longitudinal acoustic oscillations discovered with SUMER
2003  Nakariakov et al. — sausage 모드로 마이크로파 맥동 해석
          Microwave pulsations interpreted via sausage mode
>>>  2005  Nakariakov & Verwichte — 이 리뷰 논문 <<<
          This review paper
```

이 논문이 출판된 2005년은 SOHO(1995)와 TRACE(1998) 발사 이후 약 7~10년이 지난 시점으로, 코로나 파동 관측이 폭발적으로 증가하고 이론적 해석이 성숙해진 시기입니다. 이전의 관측 불가능했던 현상들이 EUV, X선, 라디오 밴드에서 확실하게 검출되면서, MHD 이론과 관측의 결합을 통한 코로나 진단이라는 새로운 분야가 열린 것입니다.

Published in 2005, about 7–10 years after SOHO (1995) and TRACE (1998) were launched, this review appeared at a time when coronal wave observations had exploded in number and theoretical interpretation had matured. Phenomena previously unobservable were now confidently detected in EUV, X-ray, and radio bands, opening the new field of coronal diagnostics through the marriage of MHD theory and observation.

---

## 필요한 배경 지식 / Prerequisites

### 물리학 / Physics
- **MHD 기초 / Basic MHD**: 이상 MHD 방정식, frozen-in 자기장 조건, 자기 압력과 장력
  Ideal MHD equations, frozen-in magnetic field condition, magnetic pressure and tension
- **파동 물리학 / Wave physics**: 분산 관계, 위상 속도 vs 군속도, 정상파와 전파파
  Dispersion relations, phase vs group velocity, standing vs propagating waves
- **3가지 기본 MHD 파동 / Three basic MHD waves**:
  - **Alfvén wave**: 비압축성, 자기 장력이 복원력, 속도 $C_A = B_0/\sqrt{\mu_0 \rho_0}$
    Incompressible, magnetic tension as restoring force
  - **Fast magnetoacoustic wave**: 압축성, 자기 압력 + 가스 압력, 등방적 전파
    Compressible, magnetic + gas pressure, isotropic propagation
  - **Slow magnetoacoustic wave**: 압축성, 자기장 방향으로 가이드됨
    Compressible, guided along magnetic field

### 수학 / Mathematics
- **Bessel 함수 / Bessel functions**: $I_m(x)$, $K_m(x)$ — 원통 좌표계에서의 파동 해
  Solutions to wave equations in cylindrical coordinates
- **고유값 문제 / Eigenvalue problems**: 경계 조건에 의한 이산 모드
  Discrete modes from boundary conditions
- **섭동 이론 / Perturbation theory**: 평형 상태 주위의 선형화
  Linearization around equilibrium state

### 선행 논문 / Prior Papers
- LRSP #1 Wood (2004) — 태양풍 기초 / Solar wind basics
- LRSP #2 Miesch (2005) — 태양 내부 역학 / Solar interior dynamics (특히 tachocline)

---

## 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **Coronal loop** / 코로나 루프 | 광구의 두 footpoint를 연결하는 아치형 자기 플럭스 튜브. 코로나에서 가장 기본적인 구조 단위. / An arch-shaped magnetic flux tube connecting two footpoints in the photosphere. The fundamental structural unit of the corona. |
| **Kink mode** ($m=1$) / 킹크 모드 | 루프가 통째로 횡방향으로 흔들리는 모드. 루프 축이 좌우로 변위됨. / The loop oscillates transversely as a whole; the loop axis displaces sideways. |
| **Sausage mode** ($m=0$) / 소시지 모드 | 루프 단면이 주기적으로 팽창·수축하는 모드. 축대칭 변형. / The loop cross-section periodically expands and contracts; axisymmetric deformation. |
| **Kink speed** $C_K$ | Kink 모드의 장파장 한계 위상 속도. 밀도 가중 평균 Alfvén 속도. / Long-wavelength phase speed of kink modes; density-weighted average Alfvén speed: $C_K = \sqrt{(\rho_0 C_{A0}^2 + \rho_e C_{Ae}^2)/(\rho_0 + \rho_e)}$ |
| **Resonant absorption** / 공진 흡수 | 전체적(global) 파동 에너지가 Alfvén 연속체의 국소 모드로 전환되어 감쇠하는 메커니즘. / Mechanism where global wave energy transfers to local Alfvén continuum modes, causing damping. |
| **Phase mixing** / 위상 혼합 | 서로 다른 자기면에서 Alfvén 파동이 각기 다른 속도로 전파하여 횡방향 경사가 급격히 증가하고 산일이 강화되는 현상. / Alfvén waves on adjacent magnetic surfaces propagate at different speeds, steepening transverse gradients and enhancing dissipation. |
| **MHD coronal seismology** / MHD 코로나 지진학 | 코로나 파동/진동의 관측 특성을 MHD 이론과 결합하여 자기장, 밀도, 수송 계수 등 미지의 코로나 물리량을 추정하는 방법. / Method of combining observed wave properties with MHD theory to estimate unknown coronal parameters. |
| **Plasma-$\beta$** / 플라즈마 베타 | 가스 압력과 자기 압력의 비율. 코로나에서는 $\beta \ll 1$ (자기 지배적). / Ratio of gas to magnetic pressure. In the corona, $\beta \ll 1$ (magnetically dominated). |
| **Dispersion relation** / 분산 관계 | 주파수 $\omega$와 파수 $k$ 사이의 관계식. 파동의 존재 조건과 특성을 결정. / Relation between frequency $\omega$ and wavenumber $k$; determines wave existence and properties. |
| **Leaky mode** / 누설 모드 | 에너지가 구조 외부로 방사되는 파동 모드. 복소 고유진동수를 가짐. / Wave mode that radiates energy into the external medium; has complex eigenfrequency. |
| **EIT wave** / EIT 파동 | SOHO/EIT로 관측되는 코로나의 대규모 전파 교란. "coronal Moreton wave"라고도 함. / Large-scale propagating disturbance observed by SOHO/EIT; also called "coronal Moreton wave." |
| **Tadpole** / 올챙이 구조 | Supra-arcade에서 아래로 이동하는 밀도 감소 구조. 꼬리 부분에서 kink 파동이 관측됨. / Density depletion moving sunward in supra-arcades; kink waves observed in the tail region. |

---

## 수식 미리보기 / Equations Preview

### 1. 총 압력 평형 / Total Pressure Balance
자기 플럭스 튜브의 내부와 외부가 평형을 이루려면:
For a magnetic flux tube to be in equilibrium between inside and outside:

$$p_0 + \frac{B_0^2}{2\mu_0} = p_e + \frac{B_e^2}{2\mu_0} \tag{1}$$

가스 압력 + 자기 압력이 경계에서 연속이어야 합니다.
Gas pressure + magnetic pressure must be continuous at the boundary.

### 2. 특성 속도 / Characteristic Speeds

| 속도 / Speed | 수식 / Formula | 의미 / Meaning |
|---|---|---|
| Sound speed $C_s$ | $\sqrt{\gamma p_0 / \rho_0}$ | 가스 압력에 의한 파동 속도 / Wave speed from gas pressure |
| Alfvén speed $C_A$ | $B_0 / \sqrt{\mu_0 \rho_0}$ | 자기 장력에 의한 파동 속도 / Wave speed from magnetic tension |
| Tube (cusp) speed $C_T$ | $C_s C_A / \sqrt{C_A^2 + C_s^2}$ | 느린 파동의 가이드 속도 / Guided speed for slow waves |
| Kink speed $C_K$ | $\sqrt{(\rho_0 C_{A0}^2 + \rho_e C_{Ae}^2)/(\rho_0+\rho_e)}$ | Kink 모드의 장파장 한계 속도 / Long-wavelength limit of kink mode |

### 3. 분산 관계 / Dispersion Relation (Edwin & Roberts 1983)
자기 실린더에서 자기음향 파동의 분산 관계:
Dispersion relation for magnetoacoustic waves in a magnetic cylinder:

$$\rho_e(\omega^2 - k_z^2 C_{Ae}^2)\kappa_0 \frac{I_m'(\kappa_0 a)}{I_m(\kappa_0 a)} + \rho_0(k_z^2 C_{A0}^2 - \omega^2)\kappa_e \frac{K_m'(\kappa_e a)}{K_m(\kappa_e a)} = 0 \tag{7}$$

- $m=0$: sausage 모드 / sausage modes
- $m=1$: kink 모드 / kink modes
- $m \geq 2$: flute (ballooning) 모드 / flute (ballooning) modes

### 4. Kink 진동으로 자기장 추정 / Magnetic Field from Kink Oscillations
실용적 공식 — 관측 가능한 양으로 자기장 $B_0$을 추정:
Practical formula — estimate magnetic field $B_0$ from observables:

$$B_0 \approx 1.02 \times 10^{-12} \frac{d\sqrt{\mu n_0}\sqrt{1 + n_e/n_0}}{P} \tag{32}$$

$d$: footpoint 간 거리(m), $n_0$: 루프 내 수밀도(m$^{-3}$), $P$: 진동 주기(s).
$d$: footpoint distance (m), $n_0$: loop number density (m$^{-3}$), $P$: oscillation period (s).

### 5. 공진 흡수에 의한 감쇠 / Resonant Absorption Damping
Kink 진동의 e-folding 감쇠 시간:
E-folding decay time for kink oscillations:

$$\frac{\tau}{P} = \frac{2}{\pi}\left(\frac{\ell}{a}\right)^{-1}\left(\frac{\rho_0 + \rho_e}{\rho_0 - \rho_e}\right) \tag{34}$$

$\ell$: 경계층 두께, $a$: 루프 반지름. 감쇠는 점성과 무관하며 순전히 모드 변환에 의함.
$\ell$: boundary layer width, $a$: loop radius. Damping is independent of viscosity — purely from mode conversion.

### 6. 전파하는 느린 파동의 진화 방정식 / Evolutionary Equation for Propagating Slow Waves

$$\frac{\partial A}{\partial s} - a_1 A - a_2 \frac{\partial^2 A}{\partial \xi^2} + a_3 A \frac{\partial A}{\partial \xi} = 0 \tag{52}$$

$a_1$: 성층화, $a_2$: 열전도/점성에 의한 산일, $a_3$: 비선형성. 확장된 Burgers 방정식.
$a_1$: stratification, $a_2$: dissipation by thermal conductivity/viscosity, $a_3$: nonlinearity. Extended Burgers equation.

---

## 논문 구조 안내 / Paper Structure Guide

| 섹션 / Section | 내용 / Content | 난이도 / Difficulty |
|---|---|---|
| §1 Introduction | MHD 코로나 지진학 개념 소개 | 쉬움 / Easy |
| §2 MHD Modes of Plasma Structures | 이론: 분산 관계, 공진 흡수, 위상 혼합, Epstein 프로파일 | 어려움 / Hard |
| §3 Kink Oscillations | TRACE 관측, 자기장 추정, 감쇠 메커니즘 | 보통 / Medium |
| §4 Sausage Oscillations | 마이크로파 맥동과의 연관, 존재 조건 | 보통 / Medium |
| §5 Acoustic Oscillations | SUMER Doppler 진동, 정상 음향 모드 | 보통 / Medium |
| §6 Propagating Acoustic Waves | EUV 전파 교란, Burgers 방정식 모델링 | 보통–어려움 / Medium-Hard |
| §7 Propagating Fast Waves | 개기일식 관측, tadpole 구조, 분산 파열 | 보통 / Medium |
| §8 Torsional Modes | 비열 선폭, Doppler 변동 | 쉬움 / Easy |
| §9 Conclusions | 요약 및 전망 | 쉬움 / Easy |

---

## 읽기 전략 / Reading Strategy

1. **§1 → §2.1**: 먼저 분산 관계와 모드 분류(kink, sausage, Alfvén)를 확실히 이해하세요. Figure 3의 분산 다이어그램이 전체 논문의 핵심입니다.
   First understand the dispersion relation and mode classification. Figure 3's dispersion diagram is the backbone of the entire paper.

2. **§3**: Kink 진동이 가장 중요한 관측 결과입니다. Figure 10의 감쇠 진동 곡선과 Eq. (32)의 자기장 추정 공식에 집중하세요.
   Kink oscillations are the most important observational result. Focus on Figure 10's damped oscillation curve and Eq. (32) for magnetic field estimation.

3. **§2.2**: 공진 흡수와 위상 혼합은 감쇠 메커니즘을 이해하는 데 필수적이지만, 수학적 세부사항보다 물리적 그림에 초점을 맞추세요.
   Resonant absorption and phase mixing are essential for understanding damping, but focus on the physical picture rather than mathematical details.

4. **§4–§8**: 나머지 모드들은 kink 모드와의 비교 관점에서 읽으면 효과적입니다.
   Read the remaining modes in comparison with kink modes.
