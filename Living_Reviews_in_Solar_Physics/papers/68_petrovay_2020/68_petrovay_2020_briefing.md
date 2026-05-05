---
title: "Pre-Reading Briefing: Solar Cycle Prediction"
paper_id: "68"
topic: Living_Reviews_in_Solar_Physics
date: 2026-04-23
type: briefing
---

# Solar Cycle Prediction: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Petrovay, K., "Solar cycle prediction", Living Reviews in Solar Physics, 17:2 (2020). DOI: 10.1007/s41116-020-0022-z
**Author(s)**: Kristóf Petrovay (Eötvös University, Budapest)
**Year**: 2020

---

## 1. 핵심 기여 / Core Contribution

### 한국어
본 논문은 태양 주기(Solar Cycle)의 진폭을 예측하는 방법론 전체를 망라한 Living Review의 2판이다. Petrovay는 예측 방법을 세 가지 큰 범주로 분류한다: (1) **선행 지표(Precursor) 방법** — 태양 활동의 특정 시점 값이나 자기장 강도를 이용해 다음 극대기 진폭을 예측, (2) **모델 기반(Model-based) 방법** — 표면 자속 수송(SFT) 모델이나 다이나모 모델에 기반, (3) **외삽(Extrapolation) 방법** — 시계열 분석 기법을 흑점 수 기록에 직접 적용. 저자는 지난 몇 주기 동안 **극자기장 선행지표(polar field precursor) 방법이 가장 신뢰할 만한 성과를 보였다**고 결론지으며, Cycle 25는 Cycle 24와 유사한 약한 주기(피크 SSN ~110-160)가 될 것이라는 컨센서스를 제시한다.

### English
This paper is the second edition of a Living Review comprehensively surveying methods for predicting solar cycle amplitude. Petrovay classifies prediction methods into three broad groups: (1) **Precursor methods** — using a measure of solar activity or magnetism at a specified time to predict the next maximum, (2) **Model-based methods** — using Surface Flux Transport (SFT) models or full dynamo models, and (3) **Extrapolation methods** — time-series analysis directly on the sunspot number record. The author concludes that over the past few cycles the **polar field precursor has consistently yielded the most reliable predictions**, and presents a broad consensus that Cycle 25 will be similar in amplitude to Cycle 24 (peak SSN ~110-160), marking a continuation of the end of the "Modern Maximum."

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

### 한국어
태양 주기 예측의 역사는 19세기 중반 Rudolf Wolf의 흑점 수 정의로 거슬러 올라간다. 1963년 Vitinsky가 첫 전공 서적을 출간한 이래, 수많은 예측 방법이 제안되어 왔다. 그러나 Cycle 24(2008-2019)는 예측의 한계를 극명히 드러낸 시점이었다. SWPC 패널(2009)은 전례 없는 의견 불일치를 보였고, NASA MSFC 팀의 예측치는 급격히 변동하여 이 분야에 대한 심각한 재평가를 촉발했다. 특히 Cycle 23-24 전이기의 활동도 급락은 'Modern Maximum'(Cycle 17-23)의 종말을 시사했고, 2015년 흑점 수 시계열의 대규모 수정(v2.0)이 이루어졌다. 이러한 배경 속에서 Cycle 25 예측(2019년 패널 공지)을 앞두고 방법론의 총정리가 필요했다.

### English
Solar cycle prediction has a long history dating back to Wolf's definition of the sunspot number in the 19th century. Since Vitinsky's 1963 monograph, numerous methods have been proposed. However, Cycle 24 (2008-2019) exposed the limits of prediction: the 2009 SWPC panel showed unprecedented disagreement, and NASA MSFC team forecasts swung wildly, prompting a serious re-evaluation of the field. The steep drop in activity from Cycle 23 to 24 suggested the end of the "Modern Maximum" (Cycles 17-23), and the 2015 major revision of the sunspot number series (version 2.0) added further complication. Against this backdrop, with the Cycle 25 Prediction Panel being convened in 2019, a comprehensive update of the methodology was urgently needed.

### 타임라인 / Timeline

```
1850 ──── Wolf introduces relative sunspot number R_Z
1913 ──── Kimura: first spectral prediction attempt
1952 ──── Gleissberg: "each cycle as a closed whole"
1953 ──── Bracewell: alternating sunspot series R_±
1963 ──── Vitinsky: first prediction monograph
1966 ──── Ohl: geomagnetic aa-index precursor
1978 ──── Schatten et al.: polar field precursor proposed
1982 ──── Feynman: interplanetary/geomagnetic separation
2006 ──── Dikpati-Gilman: flux-transport dynamo prediction
2008 ──── Choudhuri et al.: Surya code predicts weak Cycle 24
2009 ──── SWPC Panel chaos; no consensus on Cycle 24
2010 ──── Petrovay Living Review 1st edition
2015 ──── Sunspot Number revision (v2.0 released)
2017-2019 ── Surge in SFT-based Cycle 25 forecasts
2020 ──── Petrovay Living Review 2nd edition (this paper)
```

---

## 3. 필요한 배경 지식 / Prerequisites

### 한국어
- **흑점 수(Sunspot Number, SSN)**: Wolf 공식 $R_Z = k(10g + f)$와 13개월 이동평균
- **태양 다이나모 이론**: $\alpha\Omega$ 다이나모, Babcock-Leighton 메커니즘, poloidal/toroidal 장의 상호 변환
- **Hale 법칙**: 자기 극성이 주기마다 반전 → 22년 주기
- **Spörer 법칙과 butterfly diagram**: 주기 진행에 따른 흑점 위도 이동
- **지자기 지수(aa index)**: 행성간 자기장의 영향을 지구에서 측정
- **통계학**: 선형회귀, 자기회귀 (AR), ARMA, 스펙트럴 분석 (Fourier, MEM, wavelet)
- **기계학습 개념**: 신경망, 위상공간 재구성

### English
- **Sunspot Number (SSN)**: Wolf's formula $R_Z = k(10g + f)$ and 13-month running mean
- **Solar dynamo theory**: $\alpha\Omega$ dynamo, Babcock-Leighton mechanism, poloidal/toroidal field interconversion
- **Hale's law**: magnetic polarity reverses every cycle → 22-year magnetic cycle
- **Spörer's law and butterfly diagram**: equatorward drift of active latitudes over a cycle
- **Geomagnetic aa index**: Earth-based proxy for interplanetary field strength
- **Statistics**: linear regression, autoregression (AR), ARMA, spectral analysis (Fourier, MEM, wavelet)
- **Machine learning concepts**: neural networks, phase-space reconstruction

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| Precursor method / 선행 지표 방법 | 특정 시점의 태양 활동 지표로 다음 극대기 진폭 예측 (e.g., polar field at minimum) / Predicts next maximum using a specific activity measure (e.g., polar field at cycle minimum) |
| Polar field precursor / 극자기장 선행지표 | 극소기의 극자기장 강도가 다음 주기 진폭에 비례 (Schatten et al. 1978) / Polar field at minimum correlates with next cycle amplitude |
| Waldmeier effect / 발트마이어 효과 | 강한 주기는 상승시간이 짧음 (상승-진폭 역상관) / Stronger cycles have shorter rise times (rise-amplitude anti-correlation) |
| Babcock-Leighton mechanism / Babcock-Leighton 메커니즘 | 기울어진 활동 영역의 양극성 자속이 극지로 운반되어 poloidal 장 생성 / Tilted AR bipolar flux is transported poleward to generate poloidal field |
| SFT model / 표면 자속 수송 모델 | 태양 표면의 자속을 확산+이류로 모델링 (Eq. 14) / Models photospheric flux transport as diffusion + advection |
| Flux transport dynamo / 자속 수송 다이나모 | 자오면 순환이 toroidal 장을 수송하는 다이나모 / Dynamo where meridional flow transports toroidal field |
| Rogue active region / 이상 활동 영역 | Joy's law를 크게 벗어나 극자기장에 과도한 영향을 주는 AR / AR with unusual tilt/flux significantly affecting polar field |
| Tilt quenching (TQ) / 기울기 켄칭 | 강한 주기에서 AR tilt가 감소하는 비선형 피드백 / Nonlinear feedback reducing AR tilts in stronger cycles |
| Axial dipole moment / 축 쌍극자 모멘트 | $D(t)=\frac{3}{2}\int_0^\pi \bar{B}(\theta,t)\cos\theta\sin\theta d\theta$ / The $Y_1^0$ spherical harmonic coefficient of photospheric field |
| Gnevyshev-Ohl rule / Gnevyshev-Ohl 규칙 | 연속 홀/짝 주기 쌍에서 짝수 주기가 더 약한 경향 / Even-numbered cycles tend to be weaker than the following odd cycle |
| Gleissberg cycle / Gleissberg 주기 | 약 80-100년의 장주기 변조 / ~80-100 year long-period modulation of solar activity |
| LSTM / 장단기 기억 신경망 | 시계열 패턴을 학습하는 순환 신경망 구조 / Recurrent neural network architecture for learning time-series patterns |

---

## 5. 수식 미리보기 / Equations Preview

### (1) Smoothed sunspot number / 평활화 흑점 수
$$R = \frac{1}{24}\left(R_{m,-6} + 2\sum_{i=-5}^{i=5} R_{m,i} + R_{m,6}\right) \tag{2}$$

한국어: 13개월 이동평균으로 계절적 변동 제거. 월별 값 $R_{m,i}$의 가중 평균.
English: 13-month running mean removing seasonal fluctuations.

### (2) Minimum-based precursor (Brown 1976) / 최소값 기반 선행지표
$$R_{\max} = 114.3 + 6.1 R_{\min} \tag{10}$$

한국어: 이전 극소기의 흑점 수를 이용한 다음 극대기 진폭 예측. 상관계수 r=0.676.
English: Predicting next maximum from previous minimum. Correlation r=0.676.

### (3) Minimax3 (Cameron & Schüssler 2007) / 최소 3년 전 선행지표
$$R_{\max} = 79 + 1.52\, R(t_{\min} - 3\text{ years}) \tag{11}$$

한국어: 극소 3년 전의 활동 수준이 더 좋은 예측 (r=0.800). 이는 주기 중첩 효과 때문.
English: Activity 3 years before minimum is a better predictor (r=0.800), due to cycle overlap effect.

### (4) Axial dipole coefficient / 축 쌍극자 계수
$$D(t) = \frac{3}{2}\int_0^\pi \bar{B}(\theta,t)\cos\theta\sin\theta\, d\theta \tag{12}$$

한국어: 태양 광구 자기장의 $Y_1^0$ 구면조화함수 계수. 극자기장 선행지표의 핵심 물리량.
English: The $Y_1^0$ spherical harmonic coefficient of the photospheric radial field — the core physical quantity for polar field precursor.

### (5) SFT equation / 표면 자속 수송 방정식
$$\frac{\partial B}{\partial t} = -\Omega(\theta)\frac{\partial B}{\partial \phi} - \frac{1}{R_\odot \sin\theta}\frac{\partial}{\partial \theta}[v(\theta)B\sin\theta] + \frac{\eta}{R_\odot^2\sin\theta}\left[\frac{\partial}{\partial\theta}\left(\sin\theta\frac{\partial B}{\partial\theta}\right) + \frac{1}{\sin\theta}\frac{\partial^2 B}{\partial\phi^2}\right] - B/\tau + S(\theta,\phi,t) \tag{14}$$

한국어: 광구 자기장의 수송: 미분회전, 자오면 순환, 확산, 감쇠, 소스항(자속 출현).
English: Photospheric flux transport: differential rotation + meridional flow + diffusion + decay + source (flux emergence).

---

## 6. 읽기 가이드 / Reading Guide

### 한국어
이 논문은 93페이지의 방대한 리뷰이므로 다음 순서로 읽기를 권장한다:

1. **§1 (pp.3-22) 서론과 SSN**: 흑점 수의 정의, 2015년 수정, Waldmeier 효과 (섹션 1.4.3)에 집중. 식 (2)-(9)를 이해하라.
2. **§2 (pp.24-44) Precursor methods**: 핵심 섹션! §2.3 Polar precursor와 §2.4 Geomagnetic precursor가 가장 중요. Table 1 (pp.31)은 Cycle 25 예측의 근거.
3. **§3 (pp.45-59) Model-based predictions**: §3.1 SFT models와 §3.4 Dynamo models를 중심으로. 수식 (14)-(15)와 Fig. 13의 앙상블 예측.
4. **§4 (pp.60-73) Extrapolation methods**: 관심에 따라 선택적으로 읽되, §4.1 AR/ARMA와 §4.3.4 신경망은 반드시.
5. **§5-6 (pp.73-75) Summary & Cycle 25 forecasts**: Table 2는 최종 컨센서스. 이 섹션을 먼저 읽고 돌아가서 세부 방법론을 읽는 것도 좋다.

**주의**: 수식 (10)-(13)는 precursor의 핵심이므로 반드시 이해할 것. 그림 10, 11, 12는 시각적 이해의 열쇠이다.

### English
Given the review's 93 pages, the recommended reading order is:

1. **§1 (pp.3-22) Introduction and SSN**: Focus on SSN definition, 2015 revision, and Waldmeier effect (§1.4.3). Understand Eqs. (2)-(9).
2. **§2 (pp.24-44) Precursor methods**: The core! §2.3 Polar precursor and §2.4 Geomagnetic precursor are most important. Table 1 (p.31) grounds the Cycle 25 forecast.
3. **§3 (pp.45-59) Model-based predictions**: Focus on §3.1 SFT models and §3.4 Dynamo models. Equations (14)-(15) and Fig. 13 ensemble prediction.
4. **§4 (pp.60-73) Extrapolation methods**: Read selectively — §4.1 AR/ARMA and §4.3.4 Neural networks are must-reads.
5. **§5-6 (pp.73-75) Summary & Cycle 25 forecasts**: Table 2 is the final consensus. You may read this first and return to methodology.

**Note**: Equations (10)-(13) are the heart of precursor methods — understand them thoroughly. Figs. 10, 11, 12 are keys to visual understanding.

---

## 7. 현대적 의의 / Modern Significance

### 한국어
이 리뷰는 단순한 학술적 정리 이상의 실용적 가치를 지닌다. 태양 활동은 우주 날씨와 우주 기후의 주된 결정 인자로서, **우주 탐사, 민간 항공, 인공위성 통신, 전력망 안정성에 직접적 영향**을 미친다. 정확한 주기 예측은:

- **위성 운용자**: 상층 대기 팽창으로 인한 궤도 감쇠 예측
- **우주 탐사 미션**: Cycle 25 피크 시기의 방사선 노출 대비 (Artemis, Mars 2030s)
- **항공사**: 극지 항로 HF 통신 장애 예측
- **전력망 운영자**: 지자기 유도 전류 (GIC) 위험 관리

에 필수적이다. 더욱이, Cycle 24-25 전이기는 **Modern Maximum 이후 Sun이 Grand Minimum에 접근하는가?**라는 근본적 물리 질문에 대한 단서를 제공한다. 신경망과 데이터 동화(data assimilation) 기법의 부상은 다음 판에서 ML 기반 예측이 주류로 자리 잡을 것임을 예고한다.

### English
This review has practical value beyond academic synthesis. Solar activity is the prime determinant of space weather and space climate, directly affecting **space exploration, civil aviation, satellite communications, and power-grid stability**. Accurate cycle prediction is essential for:

- **Satellite operators**: predicting orbital decay from upper atmosphere expansion
- **Space exploration missions**: radiation exposure preparation at Cycle 25 peak (Artemis, Mars 2030s)
- **Airlines**: polar-route HF communication disruption forecasting
- **Power grid operators**: managing Geomagnetically Induced Current (GIC) risks

Moreover, the Cycle 24-25 transition provides clues to the fundamental physics question: **Is the Sun approaching a Grand Minimum after the Modern Maximum?** The rise of neural-network and data-assimilation techniques foreshadows that ML-based prediction will become mainstream in the next edition.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
