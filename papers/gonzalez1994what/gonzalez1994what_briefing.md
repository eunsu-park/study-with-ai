---
title: "Pre-Reading Briefing: What Is a Geomagnetic Storm?"
paper_id: "15_gonzalez_1994"
topic: Space_Weather
date: 2026-04-15
type: briefing
---

# What Is a Geomagnetic Storm?: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Gonzalez, W. D., Joselyn, J. A., Kamide, Y., Kroehl, H. W., Rostoker, G., Tsurutani, B. T., & Vasyliunas, V. M. (1994). What Is a Geomagnetic Storm? *Journal of Geophysical Research*, 99(A4), 5771–5792. doi:10.1029/93JA02867
**Author(s)**: Walter D. Gonzalez, Joselyn A. Joselyn, Yohsuke Kamide, Herbert W. Kroehl, Gordon Rostoker, Bruce T. Tsurutani, Vytenis M. Vasyliunas
**Year**: 1994

---

## 1. 핵심 기여 / Core Contribution

이 논문은 "지자기 폭풍(geomagnetic storm)이란 무엇인가?"라는 근본적 질문에 대해 당대 최고 전문가 7인이 합의한 답을 제시합니다. 1990년대 초반까지 지자기 폭풍의 정의는 연구자마다 달랐고, substorm과의 관계도 불명확했습니다. 이 논문은 폭풍의 정량적 분류 기준(Dst 임계값), 폭풍의 3단계(initial phase, main phase, recovery phase), 그리고 폭풍과 substorm의 근본적 차이를 체계적으로 정리했습니다. 이후 우주기상 커뮤니티의 표준 용어와 분류 체계의 기반이 되었습니다.

This paper provides a definitive, community-consensus answer to the fundamental question "What is a geomagnetic storm?" Written by seven leading experts, it resolves longstanding ambiguities in storm definition, establishes quantitative classification criteria (Dst thresholds for weak/moderate/intense/super-intense storms), defines the three canonical storm phases (initial, main, recovery), and clarifies the distinction between storms and substorms. It became the standard reference for space weather terminology and storm classification.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

1990년대 초반은 우주기상(space weather)이 독립적인 학문 분야로 자리잡기 시작한 시기입니다. 1989년 3월의 강력한 지자기 폭풍이 캐나다 Quebec 전력망을 마비시킨 사건 이후, 지자기 폭풍의 실용적 예보와 체계적 분류에 대한 수요가 급증했습니다. 그러나 당시에는:

- **정의의 혼란**: "storm"이라는 용어가 연구자마다 다르게 사용됨
- **storm vs substorm 논쟁**: 폭풍이 substorm의 단순 집합인지, 별개의 현상인지 논란
- **분류 기준 부재**: 폭풍 강도를 구분하는 공식적 기준이 없었음

The early 1990s marked the emergence of "space weather" as a distinct discipline. The devastating March 1989 geomagnetic storm that collapsed Quebec's power grid created urgent demand for systematic storm classification and forecasting. Yet the community lacked consensus on basic definitions: what exactly constitutes a "storm" versus a "substorm," and how should storm intensity be categorized? Burton et al. (1975, Paper #11) had established the empirical Dst–solar wind relationship, but the community still needed a unified conceptual framework.

### 타임라인 / Timeline

```
1964  Akasofu — substorm 개념 확립 (Paper #8)
  │     오로라 substorm의 형태학적 정의
1973  McPherron et al. — substorm의 위성 관측 확인 (Paper #10)
  │     NENL 모델, 성장-팽창-회복 시퀀스
1975  Burton et al. — Dst와 태양풍의 경험적 관계 (Paper #11)
  │     Dst 예측을 위한 Burton equation
1989  Quebec blackout — 우주기상의 실용적 중요성 부각
  │     지자기 폭풍으로 인한 대규모 정전
1991  ISTP (International Solar-Terrestrial Physics) 프로그램 시작
  │
1994  ★ Gonzalez et al. — 지자기 폭풍의 정의와 분류 ★
  │     storm/substorm 구별, Dst 기반 분류
1997  Kamide et al. — 자기 폭풍 예보 능력 재검토
  │     예보 정확도 평가
```

---

## 3. 필요한 배경 지식 / Prerequisites

### 3.1 Ring Current / 환전류

지자기 폭풍의 핵심 메커니즘입니다. 지구 자기장에 갇힌 이온(주로 H⁺, O⁺)과 전자가 적도면 근처(약 3–8 $R_E$)에서 자기장 구배와 곡률 drift에 의해 지구를 둘러싸며 흐르는 전류입니다. 이 전류는 지표면에서 자기장을 약화시키며, 이 약화 정도가 Dst 지수로 측정됩니다.

The ring current is a toroidal electric current flowing westward around Earth at ~3–8 $R_E$ in the equatorial plane. It is carried by trapped energetic ions (H⁺, O⁺) and electrons drifting due to magnetic gradient and curvature forces. The ring current weakens the surface magnetic field, and this depression is measured by the Dst index.

### 3.2 Dst Index / Dst 지수

Disturbance storm time index의 약자. 적도 근처 4개 관측소의 수평 자기장 성분(H) 변화량을 평균하여 시간당 산출합니다. 음의 값이 클수록 강한 폭풍을 의미합니다.

- **Quiet**: Dst > −20 nT
- **Weak storm**: −50 nT < Dst ≤ −30 nT
- **Moderate storm**: −100 nT < Dst ≤ −50 nT
- **Intense storm**: Dst ≤ −100 nT

### 3.3 Storm vs Substorm / 폭풍 vs 서브스톰

이 논문을 읽기 전에 반드시 이해해야 할 핵심 구별입니다:

- **Substorm**: 자기권 꼬리에서의 에너지 저장과 방출 과정 (시간 규모 ~1–3시간)
- **Storm**: ring current의 강화로 인한 전지구적 자기장 교란 (시간 규모 ~수시간–수일)

The key distinction this paper clarifies: a substorm is a magnetotail energy loading/unloading process (~1–3 hours); a storm is a global magnetic disturbance driven by ring current enhancement (~hours to days). The paper argues these are fundamentally different phenomena, not just different scales of the same process.

### 3.4 IMF Bz와 Reconnection / IMF Bz and Reconnection

행성간 자기장(IMF)의 남북 성분 $B_z$가 음수(남향)일 때, 지구 자기장과 magnetic reconnection이 발생하여 태양풍 에너지가 자기권으로 유입됩니다. 장시간(~3시간 이상) 강한 남향 $B_z$가 지속될 때 지자기 폭풍이 발생합니다.

When the interplanetary magnetic field (IMF) $B_z$ component is negative (southward), magnetic reconnection at the dayside magnetopause transfers solar wind energy into the magnetosphere. Sustained southward $B_z$ (≳3 hours) is the primary driver of geomagnetic storms.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **Dst (Disturbance Storm Time)** | 적도 관측소에서 측정한 수평 자기장 교란의 시간별 평균; ring current 강도의 대리 지표 / Hourly average of horizontal magnetic field disturbance at equatorial stations; proxy for ring current intensity |
| **Ring current (환전류)** | 적도면 3–8 $R_E$에서 서쪽으로 흐르는 포획 입자의 환형 전류 / Toroidal current of trapped energetic particles at 3–8 $R_E$ flowing westward |
| **Storm sudden commencement (SSC)** | 행성간 충격파(interplanetary shock)가 자기권계면에 도달할 때 발생하는 갑작스런 자기장 증가 / Sudden increase in H caused by an interplanetary shock compressing the magnetopause |
| **Main phase (주상)** | Dst가 급격히 감소하는 폭풍의 주요 단계; ring current가 강화되는 시기 / Period of rapid Dst decrease as ring current intensifies |
| **Recovery phase (회복상)** | ring current가 소산되며 Dst가 서서히 정상으로 복귀하는 단계 / Gradual return of Dst to quiet levels as ring current decays |
| **Initial phase (초기상)** | SSC 후 main phase 시작 전의 기간; Dst가 양의 값을 보일 수 있음 / Period between SSC and main phase onset; Dst may be positive |
| **Substorm (서브스톰)** | 자기권 꼬리에서의 에너지 축적과 폭발적 방출 과정; ~1–3시간 지속 / Magnetotail energy loading/unloading cycle lasting ~1–3 hours |
| **IMF $B_z$ (행성간 자기장 남북 성분)** | 행성간 자기장의 남북 방향 성분; 남향(음)일 때 reconnection 유발 / North-south component of interplanetary magnetic field; southward (negative) enables reconnection |
| **$V B_s$ (solar wind electric field)** | 태양풍 속도 $V$와 남향 IMF $B_s$ ($= |B_z|$ when $B_z < 0$)의 곱; 에너지 입력의 핵심 매개변수 / Product of solar wind speed and southward IMF; key energy coupling parameter |
| **Interplanetary shock (행성간 충격파)** | CME 또는 고속 태양풍에 의해 형성된 충격파; SSC의 원인 / Shock wave driven by CMEs or high-speed streams; triggers SSC |
| **ICME (Interplanetary Coronal Mass Ejection)** | 태양에서 방출된 코로나 물질이 행성간 공간으로 전파된 구조 / Coronal mass ejection propagated into interplanetary space |
| **CIR (Corotating Interaction Region)** | 고속·저속 태양풍 경계에서 형성되는 상호작용 영역; 반복 폭풍의 원인 / Interaction region at fast/slow solar wind boundary; causes recurrent storms |

---

## 5. 수식 미리보기 / Equations Preview

### 5.1 Burton Equation (Paper #11 복습)

$$\frac{dDst^*}{dt} = Q(t) - \frac{Dst^*}{\tau}$$

- $Dst^*$: 압력 보정된 Dst (pressure-corrected Dst)
- $Q(t)$: 에너지 주입률 (energy injection rate), $VB_s$에 비례
- $\tau$: ring current 소산 시간상수 (~7.7 시간)

이 방정식은 ring current의 에너지 균형을 나타냅니다. 논문에서 폭풍 main phase를 정량적으로 설명하는 핵심 프레임워크입니다.

This equation describes the energy balance of the ring current. It is the quantitative framework the paper uses to explain the main phase of storms.

### 5.2 Dessler-Parker-Sckopke (DPS) Relation

$$\frac{\Delta B}{B_0} = -\frac{2}{3} \frac{E_k}{E_m}$$

- $\Delta B$: 지표 자기장 변화 (≈ Dst)
- $B_0$: 정상 상태 적도 자기장 (~31,000 nT)
- $E_k$: ring current 입자의 총 운동에너지
- $E_m$: 지구 쌍극자 자기장의 총 에너지

이 관계식은 Dst 변화와 ring current 에너지 사이의 직접적 연결을 제공합니다.

This relation directly connects the Dst depression to the total kinetic energy of ring current particles, providing the physical basis for using Dst as a storm intensity measure.

### 5.3 Storm Intensity Classification / 폭풍 강도 분류

| 분류 / Category | Dst 범위 / Dst Range | 빈도 / Frequency |
|---|---|---|
| Weak (약) | −30 to −50 nT | 매우 빈번 / Very frequent |
| Moderate (중) | −50 to −100 nT | 빈번 / Frequent |
| Intense (강) | −100 to −250 nT | ~1회/월 / ~1/month |
| Super-intense (초강) | < −250 nT (−500 이하) | 드묾 / Rare |

### 5.4 Energy Coupling Function / 에너지 결합 함수

$$\epsilon = V B^2 \sin^4\left(\frac{\theta}{2}\right) l_0^2$$

- $V$: 태양풍 속도
- $B$: IMF 크기
- $\theta$: IMF clock angle (시계 각도)
- $l_0$: 유효 자기권 길이 (~7 $R_E$)

Perreault-Akasofu epsilon 함수로, 태양풍에서 자기권으로의 총 에너지 전달률을 추정합니다.

The Perreault-Akasofu epsilon function estimates the total energy transfer rate from solar wind to magnetosphere.

---

## 6. 읽기 가이드 / Reading Guide

### 추천 읽기 순서 / Recommended Reading Order

1. **Section 1 (Introduction)**: 폭풍 정의의 역사적 발전과 현안을 파악하세요. 왜 이 논문이 필요했는지 맥락을 설정합니다.
   Read the introduction to understand the historical development of storm definitions and why this consensus paper was needed.

2. **Section 2 (Storm morphology/phases)**: 핵심 섹션입니다. SSC, initial phase, main phase, recovery phase의 특성을 주의 깊게 읽으세요. 각 단계에서 Dst, AE, 오로라 등이 어떻게 변하는지 파악하세요.
   Core section. Carefully study the characteristics of each storm phase and how various indices (Dst, AE, aurora) evolve.

3. **Section 3 (Storm classification)**: Dst 기반 분류 기준과 통계적 특성을 파악하세요. 이 분류가 현대 우주기상 예보의 표준이 되었습니다.
   Learn the Dst-based classification criteria — these became the community standard.

4. **Section 4 (Solar wind drivers)**: 어떤 태양풍 조건이 폭풍을 유발하는지, IMF $B_z$의 역할, ICME vs CIR의 차이를 이해하세요.
   Understand which solar wind conditions drive storms, the role of IMF $B_z$, and the ICME vs CIR distinction.

5. **Section 5 (Storm vs substorm)**: 가장 논쟁적인 섹션입니다. 폭풍이 substorm의 단순 집합이 아니라는 논거를 주의 깊게 따라가세요.
   The most debated section. Follow the argument that storms are NOT merely superpositions of substorms.

6. **Section 6 (Summary/Conclusions)**: 전체 논문의 핵심 합의 사항을 정리합니다.

### 주의 깊게 볼 그림 / Key Figures to Watch

- **Figure 1–2**: 전형적 폭풍의 Dst 프로파일 — 각 단계 식별 연습
- **Storm statistics plots**: 다양한 강도 폭풍의 발생 빈도와 지속 시간
- **Solar wind parameter plots**: 폭풍 유발 태양풍 조건의 예시

### 핵심 질문 / Key Questions to Keep in Mind

1. Dst < −100 nT를 intense storm의 임계값으로 정한 물리적/통계적 근거는?
   What is the physical/statistical basis for the −100 nT threshold for intense storms?

2. 폭풍과 substorm은 어떤 근본적 차이가 있는가? 왜 "big substorm = small storm"이 아닌가?
   What fundamentally distinguishes a storm from a substorm? Why isn't a "big substorm" just a "small storm"?

3. ICME 구동 폭풍과 CIR 구동 폭풍의 특성 차이는?
   How do ICME-driven storms differ from CIR-driven storms in character?

---

## 7. 현대적 의의 / Modern Significance

### 지속적 영향 / Lasting Impact

이 논문의 분류 체계(Dst 기반 weak/moderate/intense/super-intense)는 30년이 지난 현재까지 우주기상 커뮤니티의 표준으로 사용되고 있습니다. NOAA Space Weather Prediction Center의 G-scale (G1–G5)은 이 분류를 기반으로 발전했습니다.

The Dst-based storm classification established in this paper remains the community standard after 30 years. NOAA's operational G-scale (G1–G5) evolved from this framework.

### 현대적 발전 / Modern Developments

- **SYM-H index**: Dst의 1분 해상도 버전으로, 현대 연구에서 더 자주 사용됨
- **Real-time Dst**: 실시간 Dst 산출로 우주기상 예보에 활용
- **Storm-substorm debate**: 이 논문 이후로도 논쟁이 계속되었으나, 폭풍이 substorm과 독립적인 현상이라는 관점이 주류가 됨
- **CME/CIR 이분법의 정교화**: 현대 연구는 ICME의 sheath vs magnetic cloud 구분, CIR의 stream interface 역할 등을 더 세분화함

- **SYM-H index**: 1-min resolution version of Dst, now preferred in modern research
- **Real-time Dst**: Operational real-time Dst computation enables space weather forecasting
- **Storm-substorm debate**: Continued after this paper, but the view that storms are independent phenomena became mainstream
- **Refined CME/CIR taxonomy**: Modern work distinguishes sheath vs magnetic cloud effects in ICMEs, and stream interface roles in CIRs

### 실용적 응용 / Practical Applications

- 전력망 보호를 위한 GIC(지자기유도전류) 예측
- 위성 운용자를 위한 방사선 벨트 환경 예보
- GPS/GNSS 정확도에 영향을 미치는 전리층 폭풍 예측
- 우주비행사 방사선 피폭 경보

- GIC (geomagnetically induced current) prediction for power grid protection
- Radiation belt environment forecasting for satellite operators
- Ionospheric storm prediction affecting GPS/GNSS accuracy
- Astronaut radiation exposure warnings

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
