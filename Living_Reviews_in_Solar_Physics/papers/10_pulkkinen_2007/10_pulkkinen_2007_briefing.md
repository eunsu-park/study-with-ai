---
title: "Space Weather: Terrestrial Perspective — Pre-reading Briefing"
paper: "Pulkkinen, T. (2007), Space Weather: Terrestrial Perspective, Living Rev. Solar Phys., 4, 1"
date: 2026-04-09
type: briefing
---

# 사전 읽기 브리핑: Space Weather: Terrestrial Perspective
# Pre-reading Briefing: Space Weather: Terrestrial Perspective

**저자 / Author**: Tuija Pulkkinen (Finnish Meteorological Institute, Helsinki)
**출판 / Published**: Living Reviews in Solar Physics, 4, 1 (2007)
**DOI**: 10.12942/lrsp-2007-1
**분량 / Length**: ~57 pages (review article)

---

## 핵심 기여 / Core Contribution

이 리뷰는 Schwenn (2006)의 "태양 관점" 우주 기상 리뷰와 쌍을 이루며, **지구 자기권-전리권 시스템의 관점에서 우주 기상을 포괄적으로 다룬다**. 태양풍 에너지가 자기권에 진입하는 메커니즘(Dungey cycle 재결합), 자기 꼬리에서의 재결합과 서브스톰, 내부 자기권의 고리 전류와 Van Allen 벨트 상대론적 전자 역학, 그리고 이러한 과정들이 위성, 전리권, 대기, 지상 기반 시설(전력망, GPS)에 미치는 실제적 영향을 체계적으로 정리한다. 특히 전역 MHD 시뮬레이션(GUMICS-4, LFM)을 관측과 비교하여 자기권 역학의 정량적 이해를 추구하며, Burton 공식의 개선된 형태를 포함한 우주 기상 예보 방법론을 논의한다.

This review complements Schwenn's (2006) "solar perspective" by comprehensively covering space weather **from the terrestrial magnetosphere-ionosphere system viewpoint**. It systematically describes how solar wind energy enters the magnetosphere (Dungey cycle reconnection), magnetotail reconnection and substorms, inner magnetosphere ring current and Van Allen belt relativistic electron dynamics, and the practical effects on satellites, ionosphere, atmosphere, and ground-based infrastructure (power grids, GPS). Using global MHD simulations (GUMICS-4, LFM) compared with observations, it pursues quantitative understanding of magnetospheric dynamics and discusses space weather prediction methodologies including an improved Burton formula.

---

## 역사적 맥락 / Historical Context

```
1716  Halley — 오로라와 지구 자기장선의 연결 제안
       Halley — Suggested aurora-magnetic field line connection
              |
1859  Carrington — 태양 플레어-지자기 폭풍 연관 발견
       Carrington — Flare-geomagnetic storm connection
              |
1961  Dungey — 자기 재결합에 의한 자기권 대류 모델
       Dungey — Magnetospheric convection via reconnection
              |
1961  Axford & Hines — 점성 상호작용 모델
       Axford & Hines — Viscous interaction model
              |
1964  Akasofu — 오로라 서브스톰의 체계적 기술
       Akasofu — Systematic description of auroral substorms
              |
1975  Burton et al. — Dst 예측 경험식
       Burton et al. — Empirical Dst prediction formula
              |
1995  SOHO/WIND 발사 → L1 태양풍 모니터링 시작
       SOHO/WIND launch → L1 solar wind monitoring begins
              |
2000  IMAGE 미션 — 자기권 중성원자 영상화
       IMAGE mission — Magnetospheric neutral atom imaging
              |
>>>  2007  Pulkkinen — 이 리뷰: 지구 관점의 우주 기상 종합 <<<
>>>  2007  Pulkkinen — this review: terrestrial perspective space weather synthesis <<<
```

---

## 필요한 배경 지식 / Prerequisites

### 자기권 물리학 / Magnetospheric Physics
- **Dungey cycle**: 주간 재결합 → 자기 꼬리 축적 → 야간 재결합 → 대류 귀환 / Dayside reconnection → tail loading → nightside reconnection → convection return
- **서브스톰 / Substorms**: growth-expansion-recovery 3단계 (Akasofu 1964) / Three-phase cycle
- **고리 전류 / Ring Current**: 내부 자기권 (2-7 $R_E$)의 포획 입자에 의한 서향 전류 / Westward current by trapped particles

### 수학적 도구 / Mathematical Tools
- **MHD 방정식**: 이상 MHD의 기본 이해 / Basic ideal MHD understanding
- **입자 표류 운동 / Particle drift motion**: $\mathbf{E} \times \mathbf{B}$ 표류, gradient/curvature 표류, 자기 모멘트 보존 / E×B drift, gradient/curvature drift, magnetic moment conservation
- **지자기 지수 / Geomagnetic indices**: $D_{st}$, $K_p$, AE, AL, AU의 물리적 의미 / Physical meaning of indices

### 이전 논문 / Prior Papers
- **LRSP #9 Schwenn (2006)**: "태양 관점" — 이 논문은 태양풍이 지구에 도착한 이후를 다룸 / "Solar perspective" — this paper covers after solar wind arrives at Earth
- **Space Weather #6 Dungey (1961)**: 자기 재결합 → Section 5-6의 이론적 기반 / Reconnection theory
- **Space Weather #7 Axford & Hines (1961)**: 점성 상호작용 → 에너지 진입의 ~10% / Viscous interaction
- **Space Weather #8 Akasofu (1964)**: 서브스톰 → Section 3.3, 6의 핵심 내용 / Substorm framework

---

## 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **Magnetopause** | 자기권 경계면. 태양풍 동압과 자기권 자기압이 균형하는 곳. 평균 ~10 $R_E$ (주간 측) / Magnetosphere boundary. Pressure balance at ~10 $R_E$ subsolar |
| **Magnetotail** | 태양 반대쪽으로 길게 늘어진 자기 꼬리. 수백 $R_E$ 이상 확장. 플라즈마 시트와 tail lobe로 구성 / Elongated magnetic tail extending hundreds of $R_E$. Plasma sheet + tail lobes |
| **Substorm** | 자기 꼬리에서의 에너지 축적-방출 주기 (2-3시간). Growth → expansion → recovery / Magnetotail energy loading-release cycle |
| **Ring current** | 2-7 $R_E$에서 지구를 감싸는 서향 전류. keV-수백 keV 이온 (주로 양성자 + O$^+$). $D_{st}$ 지수로 측정 / Westward current encircling Earth at 2-7 $R_E$. Measured by $D_{st}$ |
| **Plasmasphere** | 이온권에서 기원한 차가운 (~1 eV), 조밀한 (10-1000 cm$^{-3}$) 플라즈마 토러스. 공회전 전기장에 의해 유지 / Cold, dense plasma torus maintained by corotation electric field |
| **Van Allen belts** | 내부 (양성자, ~1.5 $R_E$) + 외부 (상대론적 전자, 3-7 $R_E$) 복사 벨트 / Inner (protons) + outer (relativistic electrons) radiation belts |
| **$\epsilon$ parameter** | Akasofu의 에너지 결합 함수: $\epsilon = 10^7 V_{sw} B^2 (7R_E)^2 \sin^4(\theta/2)$. 태양풍→자기권 에너지 전달률의 근사 / Akasofu's energy coupling function approximating solar wind → magnetosphere energy transfer |
| **$D_{st}$ index** | 중위도 4개 관측소의 수평 자기장 변화 평균. 고리 전류 강도의 프록시. 음의 값 = 폭풍 / Average horizontal field disturbance from 4 mid-latitude stations. Ring current proxy |
| **Burton formula** | $dD_{st}^*/dt = Q(t) - D_{st}^*/\tau$ — 태양풍 입력으로 $D_{st}$ 시간 진화를 예측하는 경험식 / Empirical formula predicting $D_{st}$ evolution from solar wind input |
| **Harris current sheet** | $B_x = B_0 \tanh(Z/\lambda)$ — 자기 꼬리 플라즈마 시트의 1D 모델 / 1D model of magnetotail plasma sheet |
| **Plasmoid** | 서브스톰 시 재결합으로 형성되어 꼬리쪽으로 방출되는 자기장-플라즈마 구조 / Magnetic field-plasma structure formed by reconnection and ejected tailward during substorms |
| **GIC (Geomagnetically Induced Current)** | 급격한 지자기장 변화가 지상 도체(전력망, 파이프라인)에 유도하는 전류. 변압기 손상 가능 / Currents induced in ground conductors by rapid geomagnetic field changes |
| **Poynting flux** | $\mathbf{K} = (1/\mu_0) \mathbf{E} \times \mathbf{B}$ — 자기권 경계를 통한 전자기 에너지 전달의 정량적 측정 / Quantitative measure of electromagnetic energy transfer through magnetopause |

---

## 수식 미리보기 / Equations Preview

### 1. Shue 자기권계면 모델 / Shue Magnetopause Model (Eq 1-2)

자기권계면의 형태를 태양풍 동압과 IMF $B_z$의 함수로 기술.
Describes magnetopause shape as function of solar wind dynamic pressure and IMF $B_z$.

$$R(\phi) = R_0 \left(\frac{2}{1 + \cos\phi}\right)^\alpha$$

$$R_0 = (10.22 + 1.29 \tanh[0.184(B_z + 8.14)]) \cdot P_{sw}^{-1/6.6}$$

- $R_0$: 주간 기립 거리 ($R_E$ 단위) / subsolar standoff distance
- $\alpha$: 꼬리 플레어링 파라미터 / tail flaring parameter
- 강한 폭풍 시 $R_0$가 정지궤도(6.6 $R_E$) 이내로 축소 가능 / Can compress inside geostationary orbit during strong storms

### 2. Akasofu $\epsilon$ 파라미터 / Akasofu Epsilon Parameter (Eq 3)

태양풍에서 자기권으로의 에너지 전달률을 근사하는 경험적 파라미터.
Empirical parameter approximating energy transfer rate from solar wind to magnetosphere.

$$\epsilon = 10^7 V_{sw} B^2 (7\,R_E)^2 \sin^4(\theta/2)$$

- $V_{sw}$: 태양풍 속도, $B$: IMF 크기, $\theta$: IMF clock angle / Solar wind speed, IMF magnitude, clock angle
- $\theta = 180°$ (순수 남향) → $\sin^4(\theta/2) = 1$ → 최대 에너지 전달 / Maximum energy transfer for pure southward IMF

### 3. 개선된 Burton 공식 / Improved Burton Formula (Eq 6-7)

압력 보정된 $D_{st}^*$의 시간 진화. 우주 기상 예보의 핵심 도구.
Time evolution of pressure-corrected $D_{st}^*$. Key space weather forecasting tool.

$$D_{st}^* = D_{st} - 7.26\sqrt{P_{sw}} + 11.0$$

$$\frac{dD_{st}^*}{dt} = Q(t) - \frac{D_{st}^*}{\tau}$$

$$Q(t) = -4.4(V_{sw}B_s - E_c), \quad \tau = 2.40 \exp\left(\frac{9.74}{4.69 + V_{sw}B_s}\right)$$

- $B_s$: IMF 남향 성분, $E_c = 0.49$ mV/m: 임계값 / Southward IMF, critical threshold
- $\tau$: 고리 전류 감쇠 시간 (태양풍 구동에 의존) / Ring current decay time (depends on solar wind driving)

### 4. 총 에너지 플럭스 / Total Energy Flux (Eq 8)

전역 MHD 시뮬레이션에서의 보존량. 자기권계면을 통한 에너지 전달 추적에 사용.
Conserved quantity in global MHD. Used to trace energy transfer through magnetopause.

$$\mathbf{K} = \left(U + P - \frac{B^2}{2\mu_0}\right)\mathbf{v} + \frac{1}{\mu_0}\mathbf{E} \times \mathbf{B}$$

- 첫째 항: 운동 + 열 + 자기 에너지 플럭스 / Kinetic + thermal + magnetic energy flux
- 둘째 항: Poynting flux / Poynting flux

### 5. 입자 표류 속도 / Particle Drift Velocity (Eq 9-10)

내부 자기권에서의 입자 수송과 에너지화를 기술하는 안내 중심 표류.
Guiding center drifts describing particle transport and energization in inner magnetosphere.

$$\mathbf{V} = \frac{\mathbf{B}}{eB} \times \left[2W_\parallel (\mathbf{B} \cdot \nabla)\mathbf{B} + \mu \nabla B\right] + \frac{\mathbf{E} \times \mathbf{B}}{B^2}$$

$$\mu = \frac{W_\perp}{B} \quad \text{(first adiabatic invariant / 제1 단열 불변량)}$$

### 6. Volland-Stern 대류 전기장 / Volland-Stern Convection Electric Field (Eq 11-12)

내부 자기 꼬리의 대류 전기 퍼텐셜. 고리 전류 입자 수송 모델의 기본 입력.
Convection electric potential in inner tail. Basic input for ring current transport models.

$$\Phi_{conv} = A L^\gamma \sin(\phi - \phi_0)$$

$$A = \frac{0.045}{(1 - 0.159K_p + 0.0093K_p^2)^3} \quad \text{kV}/R_E^2$$

---

## 논문 구조 개요 / Paper Structure Overview

| 장 / Ch. | 제목 / Title | 핵심 내용 / Key Content |
|---|---|---|
| 1 | Introduction | 우주 기상 정의, 시간 규모 (태양 주기 ~ 초), 예보 리드 타임 / Definition, timescales, forecasting lead times |
| 2 | Solar Influence on Geospace | 태양 활동-지자기 활동 상관, CME/ICME, CIR, Russell-McPherron 효과 / Solar-geomagnetic correlation, CME/CIR drivers, seasonal modulation |
| 3 | The Magnetosphere | 구조 (자기권계면, 꼬리, 고리 전류), 플라즈마 (전리권, 플라즈마구), 역학 (서브스톰, 폭풍, SMC) / Structure, plasmas, dynamics |
| 4 | Monitoring | 관측 (Shue 모델, $\epsilon$ 파라미터, 지자기 지수, Burton 공식), 전역 MHD 시뮬레이션 (GUMICS-4, LFM) / Observations and global MHD simulations |
| 5 | Solar Wind Energy Entry | Poynting flux 추적, 자기권계면 에너지 전달 위치, 전리권 소산 (Joule 가열, 입자 강수) / Energy transfer tracing through magnetopause |
| 6 | Reconnection in Magnetotail | Harris 전류 시트, 박화/강화, 서브스톰 재결합, plasmoid 방출, 전역 MHD 결과 / Current sheet thinning, substorm reconnection, MHD results |
| 7 | Space Weather in Inner Magnetosphere | 전자기장 변동, 고리 전류 (표류 모델), 플라즈마구 변화, 상대론적 전자 가속/손실 / EM fields, ring current, plasmasphere, relativistic electrons |
| 8 | Space Weather Effects | 자기권 (위성 대전/항력), 전리권 (GPS 오차), 대기 (NO₂, 오존), 지상 (GIC, 전력망) / Effects on satellites, ionosphere, atmosphere, ground |
| 9 | Space Weather Predictions | 경험적/물리 기반 예보, 리드 타임 한계 (최대 80시간, 실질 ~1시간) / Empirical/physics-based forecasting, lead time limits |
| 10 | Concluding Remarks | 주요 도전과제, 미래 방향 / Key challenges and future directions |

---

## 읽기 전략 / Reading Strategy

1. **Section 3 (자기권 구조/역학)이 가장 중요**: 나머지 모든 내용의 물리적 기반. Figure 2-5를 충분히 이해할 것 / Most important — physical basis for everything else

2. **Section 5의 에너지 전달 분석에 주목**: GUMICS-4 시뮬레이션으로 Poynting flux 추적 — 정량적 이해의 핵심 / Energy transfer analysis with GUMICS-4 — key for quantitative understanding

3. **Eq 7 (개선된 Burton 공식)을 완전히 이해할 것**: 우주 기상 예보의 핵심 도구 / Must fully understand the improved Burton formula

4. **Section 8 (실제 효과)은 실용적 관점에서 중요**: 왜 우주 기상을 연구하는지의 동기 / Important for practical motivation

---

## Schwenn (2006) vs Pulkkinen (2007) 비교 / Comparison

| 측면 / Aspect | Schwenn (LRSP #9) | Pulkkinen (LRSP #10) |
|---|---|---|
| 관점 / Perspective | 태양 / Solar | 지구 / Terrestrial |
| 초점 / Focus | 태양풍, 플레어, SEP, CME | 자기권, 전리권, 고리 전류, Van Allen 벨트 |
| 핵심 도구 / Key tool | 코로나그래프 (LASCO) | MHD 시뮬레이션 (GUMICS-4, LFM) |
| 시간 범위 / Timescale | 태양 표면 → 1 AU | 자기권계면 → 지상 |
| 예보 도구 / Forecasting | CME 도착 시간 경험식 | Burton/Dst 예측, AI/경험적 모델 |
