---
title: "Pre-reading Briefing: An Empirical Relationship between Interplanetary Conditions and Dst"
authors: Rande K. Burton, Robert L. McPherron, Christopher T. Russell
year: 1975
journal: "Journal of Geophysical Research, Vol. 80, No. 31, pp. 4204–4214"
doi: "10.1029/JA080i031p04204"
topic: Space Weather / Geomagnetic Storms
type: briefing
date: 2026-04-11
---

# Pre-reading Briefing / 사전 읽기 브리핑

## Burton, McPherron & Russell (1975)
### "An Empirical Relationship between Interplanetary Conditions and Dst"

---

## 1. 핵심 기여 / Core Contribution

이 논문은 태양풍 매개변수(특히 행성간 전기장)와 지자기 폭풍 강도의 척도인 Dst 지수 사이의 최초의 정량적 경험식을 도출했습니다. "Burton equation"으로 알려진 이 관계식은 태양풍의 남향 행성간 자기장(southward IMF)이 ring current에 에너지를 주입하여 Dst를 감소시키는 과정을 1차 미분방정식으로 모델링합니다. 이 방정식은 이후 수십 년간 모든 경험적 지자기 폭풍 예보 모델의 기초가 되었으며, 오늘날에도 운용 우주기상 예보에 사용됩니다.

This paper derived the first quantitative empirical formula relating solar wind parameters (specifically the interplanetary electric field) to the Dst index, a measure of geomagnetic storm intensity. Known as the "Burton equation," this relationship models how the southward interplanetary magnetic field (IMF) injects energy into the ring current, decreasing Dst, as a first-order ordinary differential equation. This equation became the foundation for all subsequent empirical geomagnetic storm forecasting models and remains in operational use today.

---

## 2. 역사적 맥락 / Historical Context

### 이 논문이 등장하기까지 / Road to This Paper

1940년대부터 Chapman & Bartels가 지자기 지수(Dst, Kp)를 체계화했지만, 태양풍 조건과 지자기 활동 사이의 **정량적** 관계는 알려지지 않았습니다. 1961년 Dungey가 southward IMF에 의한 magnetic reconnection 이론을 제시하고, 1960-70년대에 위성 관측이 축적되면서 태양풍-자기권 결합의 물리적 기초가 마련되었습니다.

From the 1940s, Chapman & Bartels had systematized geomagnetic indices (Dst, Kp), but no **quantitative** relationship between solar wind conditions and geomagnetic activity was known. After Dungey proposed the magnetic reconnection mechanism driven by southward IMF in 1961, and satellite observations accumulated through the 1960s-70s, the physical basis for solar wind-magnetosphere coupling was established.

### 타임라인 / Timeline

```
1940  Chapman & Bartels — Dst/Kp 지수 체계화
       |
1958  Parker — 태양풍 예측 / Solar wind prediction
       |
1961  Dungey — Southward IMF reconnection 이론
       |
1964  Akasofu — Substorm 형태학 정의
       |
1966  Dessler-Parker-Sckopke — Dst와 ring current 에너지 관계
       |
1969  Davis & Sugiura — Hourly Dst index 공식 발표
       |
1973  McPherron, Russell & Aubry — 다중위성 substorm 분석
       |
>>>>  1975  Burton, McPherron & Russell — Dst 경험식 (이 논문)  <<<<
       |
1982  Cowley — 통합 대류 모델
       |
1988  Richmond & Kamide — AMIE 기법
       |
1994  Gonzalez et al. — 지자기 폭풍 정의와 분류
```

### 왜 이 시점에 가능했는가 / Why This Timing

- Explorer 33/35, IMP 위성들이 L1 근처에서 태양풍 데이터를 제공하기 시작
- Hourly Dst 지수가 1957년부터 체계적으로 편찬됨
- 충분한 양의 동시 태양풍-Dst 데이터 축적 (1967-1972)

- Explorer 33/35 and IMP satellites began providing solar wind data near L1
- Hourly Dst index had been systematically compiled since 1957
- Sufficient simultaneous solar wind-Dst data had accumulated (1967-1972)

---

## 3. 필요한 배경 지식 / Prerequisites

### 3.1 Dst 지수란? / What Is the Dst Index?

Dst (Disturbance Storm Time) 지수는 지구 적도 부근의 지자기 관측소 네트워크에서 측정한 수평 자기장 성분의 교란을 시간 평균한 값입니다.

The Dst (Disturbance Storm Time) index is the hourly average of the horizontal magnetic field disturbance measured by a network of near-equatorial geomagnetic observatories.

- **Dst < 0**: Ring current가 강화되어 지표면 자기장을 약화시킴 (폭풍 상태)
- **Dst > 0**: 태양풍 동압에 의한 자기권 압축 (sudden commencement)
- **단위**: nT (nanotesla)
- **폭풍 분류**: Moderate (-50 ~ -100 nT), Intense (-100 ~ -250 nT), Super-storm (< -250 nT)

- **Dst < 0**: Enhanced ring current weakens surface magnetic field (storm conditions)
- **Dst > 0**: Magnetospheric compression by solar wind dynamic pressure (sudden commencement)
- **Units**: nT (nanotesla)
- **Storm classification**: Moderate (-50 to -100 nT), Intense (-100 to -250 nT), Super-storm (< -250 nT)

### 3.2 Ring Current / 환전류

Ring current는 지구 자기권 내부 (약 3~8 $R_E$)에서 에너지 이온(주로 H⁺, O⁺, 수 keV ~ 수백 keV)이 자기장에 갇혀 지구 주위를 drift하면서 형성하는 전류입니다.

The ring current is a toroidal current in the inner magnetosphere (about 3–8 $R_E$) formed by energetic ions (mainly H⁺, O⁺, a few keV to hundreds of keV) that are trapped in the magnetic field and drift around Earth.

**Dessler-Parker-Sckopke (DPS) 관계식**:

$$Dst \propto -E_{\text{RC}}$$

여기서 $E_{\text{RC}}$는 ring current 입자의 총 운동 에너지입니다. Dst가 음의 방향으로 커질수록 ring current 에너지가 크다는 의미입니다.

Where $E_{\text{RC}}$ is the total kinetic energy of ring current particles. A more negative Dst means greater ring current energy.

### 3.3 Southward IMF와 Energy Injection / Southward IMF and Energy Injection

Dungey (1961)에 따르면, 행성간 자기장(IMF)의 $B_z$ 성분이 남향(음)일 때 지구 자기장과 reconnection이 일어나 태양풍 에너지가 자기권에 유입됩니다.

According to Dungey (1961), when the $B_z$ component of the interplanetary magnetic field (IMF) is southward (negative), reconnection with Earth's field occurs, allowing solar wind energy to enter the magnetosphere.

**Dawn-to-dusk 행성간 전기장**:

$$E_y = -V_{sw} \times B_z$$

여기서 $V_{sw}$는 태양풍 속도, $B_z$는 IMF의 남북 성분입니다. $B_z < 0$ (southward)일 때 $E_y > 0$이 되어 에너지 주입이 일어납니다.

Where $V_{sw}$ is solar wind speed and $B_z$ is the north-south IMF component. When $B_z < 0$ (southward), $E_y > 0$ and energy injection occurs.

### 3.4 기초 수학 / Mathematical Prerequisites

이 논문의 핵심은 1차 선형 상미분방정식(ODE)입니다:

The core of this paper is a first-order linear ordinary differential equation (ODE):

$$\frac{dx}{dt} + \frac{x}{\tau} = F(t)$$

이 형태의 방정식은 "입력(injection)과 감쇠(decay)가 동시에 작용하는 시스템"을 기술합니다. 이전 논문에서 다뤘던 RC 회로의 충방전과 동일한 수학적 구조입니다.

This equation describes a system where "injection (input) and decay act simultaneously." It has the same mathematical structure as an RC circuit's charge/discharge, which may be familiar from earlier studies.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Description |
|---|---|
| **Dst index** | 적도 부근 지자기 교란의 시간 평균 지수. Ring current 강도의 대리변수 / Hourly average of equatorial geomagnetic disturbance. Proxy for ring current intensity |
| **Dst*** | 태양풍 동압 효과를 보정한 Dst. "Pressure-corrected Dst" / Dst corrected for solar wind dynamic pressure effect |
| **Ring current** | 내부 자기권의 환상 전류. 에너지 이온의 gradient-curvature drift에 의해 형성 / Toroidal current in inner magnetosphere formed by gradient-curvature drift of energetic ions |
| **IMF $B_z$** | 행성간 자기장의 남북 성분. 남향(음)이면 reconnection 촉진 / North-south component of interplanetary magnetic field. Southward (negative) promotes reconnection |
| **Dawn-dusk electric field ($E_y$)** | $-V_{sw} B_z$. 태양풍이 자기권에 전달하는 전기장 / Electric field conveyed by solar wind to magnetosphere |
| **Solar wind dynamic pressure ($P_{dyn}$)** | $\frac{1}{2}\rho V_{sw}^2$. 자기권을 압축하는 태양풍의 운동 압력 / Ram pressure of solar wind compressing the magnetosphere |
| **Injection function $Q$** | 태양풍 조건에 따른 ring current 에너지 주입률 / Rate of ring current energy injection as a function of solar wind conditions |
| **Decay time ($\tau$)** | Ring current 에너지가 e-folding으로 감소하는 시간 (~7.7시간) / Time for ring current energy to decrease by factor of $e$ (~7.7 hours) |
| **Magnetopause** | 태양풍과 자기권의 경계면 / Boundary between solar wind and magnetosphere |
| **Sudden commencement (SC)** | 충격파 도달 시 Dst의 급격한 양의 변화 / Sudden positive jump in Dst when a shock arrives |
| **Half-wave rectifier** | 남향 $B_z$만 선택하는 함수 ($B_z < 0$일 때만 작동) / Function that selects only southward $B_z$ (active only when $B_z < 0$) |

---

## 5. 수식 미리보기 / Equations Preview

### 5.1 Burton Equation (핵심 방정식)

이 논문의 가장 중요한 결과는 다음 미분방정식입니다:

The most important result of this paper is the following differential equation:

$$\frac{dDst^*}{dt} = Q(E_y) - \frac{Dst^*}{\tau}$$

여기서:

Where:

- $Dst^* = Dst - b\sqrt{P_{dyn}} + c$ : 동압 보정된 Dst
- $Q(E_y)$: 에너지 주입 함수 (injection function)
- $\tau$: ring current 감쇠 시간 상수 (decay time constant)

- $Dst^* = Dst - b\sqrt{P_{dyn}} + c$ : pressure-corrected Dst
- $Q(E_y)$: energy injection function
- $\tau$: ring current decay time constant

### 5.2 동압 보정 / Dynamic Pressure Correction

태양풍 동압은 자기권을 압축하여 Dst를 양의 방향으로 변화시킵니다. 이 효과를 제거해야 ring current의 순수 기여만 볼 수 있습니다:

Solar wind dynamic pressure compresses the magnetosphere, causing a positive shift in Dst. This effect must be removed to see the pure ring current contribution:

$$Dst^* = Dst - b\sqrt{P_{dyn}} + c$$

여기서 $P_{dyn} = \frac{1}{2} m_p n V_{sw}^2$ (양성자 질량 $m_p$, 밀도 $n$, 속도 $V_{sw}$).

Where $P_{dyn} = \frac{1}{2} m_p n V_{sw}^2$ (proton mass $m_p$, density $n$, speed $V_{sw}$).

Burton et al.은 $b \approx 15.8$ nT/nPa$^{1/2}$, $c \approx 20$ nT로 결정했습니다.

Burton et al. determined $b \approx 15.8$ nT/nPa$^{1/2}$, $c \approx 20$ nT.

### 5.3 에너지 주입 함수 / Energy Injection Function

$$Q = \begin{cases} d(E_y - E_c) & \text{if } E_y > E_c \text{ (southward IMF)} \\ 0 & \text{if } E_y \leq E_c \end{cases}$$

여기서:

Where:

- $E_y = -V_{sw} B_z$ (dawn-dusk 전기장, mV/m)
- $E_c \approx 0.5$ mV/m (임계 전기장 — 이 이하에서는 주입 없음)
- $d \approx -1.5 \times 10^{-3}$ nT/(s·mV/m)

이 "half-wave rectifier" 형태는 남향 IMF만이 ring current에 에너지를 주입한다는 Dungey의 reconnection 이론과 일치합니다.

This "half-wave rectifier" form is consistent with Dungey's reconnection theory that only southward IMF injects energy into the ring current.

### 5.4 감쇠항 / Decay Term

$$\frac{Dst^*}{\tau}, \quad \tau \approx 7.7 \text{ hours}$$

Ring current 이온은 charge exchange, Coulomb scattering, wave-particle interaction 등으로 에너지를 잃습니다. Burton et al.은 이를 단일 지수 감쇠($\tau \approx 7.7$시간)로 근사했습니다.

Ring current ions lose energy through charge exchange, Coulomb scattering, and wave-particle interactions. Burton et al. approximated this as a single exponential decay ($\tau \approx 7.7$ hours).

### 5.5 물리적 해석 / Physical Interpretation

Burton equation은 **에너지 수지 방정식**입니다:

The Burton equation is an **energy balance equation**:

$$\underbrace{\frac{dDst^*}{dt}}_{\text{Dst 변화율}} = \underbrace{Q(E_y)}_{\text{태양풍 에너지 주입}} - \underbrace{\frac{Dst^*}{\tau}}_{\text{ring current 에너지 손실}}$$

- $Q > |Dst^*/\tau|$: Dst가 음의 방향으로 감소 → **폭풍 main phase**
- $Q = 0$: 에너지 주입 중단, 자연 감쇠만 → **recovery phase**
- 정상 상태: $Q = Dst^*/\tau$ → Dst가 일정

- $Q > |Dst^*/\tau|$: Dst decreases (becomes more negative) → **storm main phase**
- $Q = 0$: injection stops, only natural decay → **recovery phase**
- Steady state: $Q = Dst^*/\tau$ → Dst is constant

---

## 6. 읽기 전 생각해볼 질문 / Questions to Consider While Reading

1. **왜 half-wave rectifier인가?** 왜 북향 IMF는 ring current에 에너지를 주입하지 못하는가? Dungey의 reconnection 모델과 어떻게 연결되는가?
   **Why a half-wave rectifier?** Why can't northward IMF inject energy into the ring current? How does this connect to Dungey's reconnection model?

2. **단일 감쇠 시간 상수의 한계는?** 실제 ring current 손실 메커니즘은 에너지에 따라 다른 시간 스케일을 가질 수 있다. 이 단순화가 어떤 결과를 초래하는가?
   **Limitations of a single decay constant?** Real ring current loss mechanisms may have different timescales depending on energy. What consequences does this simplification have?

3. **동압 보정은 왜 필요한가?** $\sqrt{P_{dyn}}$에 비례하는 이유는 무엇인가? (힌트: Chapman-Ferraro magnetopause 거리)
   **Why is the pressure correction needed?** Why proportional to $\sqrt{P_{dyn}}$? (Hint: Chapman-Ferraro magnetopause distance)

4. **임계 전기장 $E_c$의 물리적 의미는?** 왜 약간의 southward IMF로는 충분하지 않은가?
   **Physical meaning of the threshold $E_c$?** Why isn't a slight southward IMF sufficient?

5. **이 경험식이 극한 폭풍에서도 유효한가?** Dst < -300 nT 수준의 폭풍에서 이 선형 모델이 여전히 작동하는가?
   **Is this empirical formula valid for extreme storms?** Does this linear model still work for storms with Dst < -300 nT?

---

## 7. 다음 논문과의 연결 / Connection to Next Papers

| 다음 논문 / Next Paper | 연결점 / Connection |
|---|---|
| #12 Cowley (1982) | Burton equation의 물리적 기반이 되는 convection 이론을 통합 |
| #13 Richmond & Kamide (1988) | Dst만이 아닌 전리층 전기역학의 전역적 관측 기법 |
| #15 Gonzalez et al. (1994) | Burton equation 기반으로 폭풍 강도 분류 기준 확립 |
| #17 Allen et al. (1989) | Burton equation으로 예측 가능한 실제 극한 폭풍의 사회적 영향 |
| #33 Camporeale et al. (2019) | Burton equation을 ML 모델로 확장/대체하는 현대적 접근 |

---

*이 브리핑은 논문을 읽기 전에 참고하기 위한 자료입니다.*
*This briefing is a reference document to consult before reading the paper.*
*VSCode에서 Cmd+Shift+V로 미리보기하면 수식이 렌더링됩니다.*
*Preview with Cmd+Shift+V in VSCode to render equations.*
