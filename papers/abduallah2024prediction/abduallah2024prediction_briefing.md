---
title: "Pre-Reading Briefing: Prediction of the SYM-H Index Using a Bayesian Deep Learning Method With Uncertainty Quantification"
paper_id: "38_abduallah_2024"
topic: Space_Weather
date: 2026-04-15
type: briefing
---

# Prediction of the SYM-H Index Using a Bayesian Deep Learning Method With Uncertainty Quantification: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Abduallah, Y., Alobaid, K. A., Wang, J. T. L., Wang, H., Jordanova, V. K., Yurchyshyn, V., Cavus, H., & Jing, J. (2024). *Space Weather*, 10.1029/2023SW003824.
**Author(s)**: Yasser Abduallah, Khalid A. Alobaid, Jason T. L. Wang, Haimin Wang, Vania K. Jordanova, Vasyl Yurchyshyn, Husein Cavus, Ju Jing
**Year**: 2024

---

## 1. 핵심 기여 / Core Contribution

이 논문은 SYMHnet이라는 새로운 딥러닝 프레임워크를 제안하여, 그래프 신경망(GNN)과 양방향 장단기 메모리(BiLSTM) 네트워크를 결합하고 Bayesian inference를 통합하여 SYM-H 지자기 지수를 1-2시간 전에 예측합니다. 핵심 혁신은 세 가지입니다: (1) 태양풍/IMF 파라미터 간의 관계를 완전 연결 그래프(FCG)로 모델링하는 GNN, (2) 시계열의 양방향 시간 패턴을 포착하는 BiLSTM, (3) Monte Carlo dropout을 사용한 aleatoric(데이터) 및 epistemic(모델) 불확실성의 정량화. 1분 해상도 SYM-H 예측에 딥러닝을 최초로 적용했으며, 기존 방법(LCNN, GBM, LSTM, CNN, Burton equation) 대비 우수한 성능을 보입니다.

This paper proposes SYMHnet, a novel deep learning framework that combines Graph Neural Networks (GNN) with Bidirectional Long Short-Term Memory (BiLSTM) networks, integrated with Bayesian inference, to predict the SYM-H geomagnetic index 1-2 hours in advance. Three key innovations: (1) GNN modeling inter-parameter relationships via a Fully Connected Graph (FCG), (2) BiLSTM capturing bidirectional temporal patterns, (3) Monte Carlo dropout for quantifying aleatoric (data) and epistemic (model) uncertainty. This is the first deep learning application to 1-min resolution SYM-H prediction, outperforming existing methods (LCNN, GBM, LSTM, CNN, Burton equation).

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

SYM-H 지수는 ring current의 강도를 나타내는 지자기 지수로, Dst 지수의 1분 해상도 버전입니다. 지자기 폭풍의 강도를 실시간으로 추적하는 데 핵심적이며, 위성 운용, 전력망 보호 등의 의사결정에 활용됩니다. 기존 SYM-H 예측 연구는 대부분 5분 해상도 데이터를 사용했으나, 1분 해상도는 SYM-H의 높은 진동 특성 때문에 훨씬 어려운 과제입니다.

The SYM-H index reflects ring current intensity and is essentially a 1-min resolution version of the Dst index. It's critical for real-time tracking of geomagnetic storm intensity, used in satellite operations and power grid protection. Previous SYM-H prediction studies mostly used 5-min resolution data; 1-min resolution is significantly more challenging due to SYM-H's highly oscillating nature.

### 타임라인 / Timeline

```
1975  Burton et al.          최초의 Dst 경험적 관계식 / First empirical Dst relationship
1996  Gleisner et al.        최초의 ANN 기반 Dst/Kp 예측 / First ANN-based Dst/Kp prediction
2010  Cai et al.             NARX 신경망으로 SYM-H 5분 평균 예측 / SYM-H 5-min prediction with NARX
2017  Chandorkar et al.      Gaussian Process로 Dst 예측 / Dst prediction with Gaussian Processes
2018  Gruet et al.           LSTM으로 Dst 예측 / Dst prediction with LSTM
2021  Siciliano et al.       LSTM+CNN으로 SYM-H 5분 예측 비교 / LSTM+CNN SYM-H 5-min comparison
2021  Collado-Villaverde     LCNN (LSTM+CNN) 5분 SYM-H 예측 / LCNN 5-min SYM-H prediction
2022  long et al.            GBM으로 SYM-H 5분 예측 / GBM for SYM-H 5-min prediction
2024  ★ Abduallah et al.     본 논문: SYMHnet — GNN+BiLSTM+Bayesian, 최초 1분 해상도
                             This paper: SYMHnet — first 1-min resolution with uncertainty
```

---

## 3. 필요한 배경 지식 / Prerequisites

### 지자기 지수 / Geomagnetic Indices
- **SYM-H**: ring current 강도의 1분 해상도 지표. 양수 = SSC(storm sudden commencement), 음수 = main phase 강도. 단위: nT.
  **SYM-H**: 1-min resolution measure of ring current intensity. Positive = SSC, negative = main phase intensity. Unit: nT.
- **Dst vs SYM-H**: Dst는 1시간 해상도, SYM-H는 1분 해상도. SYM-H가 더 세밀한 폭풍 구조를 포착.
  **Dst vs SYM-H**: Dst is 1-hr resolution; SYM-H is 1-min. SYM-H captures finer storm structure.

### 그래프 신경망 / Graph Neural Networks
- **GNN**: 그래프 구조 데이터에서 노드 간 관계를 학습하는 신경망. 각 노드가 파라미터를 나타내고 엣지가 관계를 표현.
  **GNN**: Neural network learning inter-node relationships on graph-structured data. Nodes represent parameters, edges represent relationships.
- **GCN (Graph Convolutional Network)**: 이웃 노드의 정보를 집약하여 노드 표현을 업데이트.
  **GCN**: Updates node representations by aggregating neighbor information.

### BiLSTM
- **LSTM**: 장기 의존성을 학습할 수 있는 RNN 변형. forget gate, input gate, output gate로 구성.
  **LSTM**: RNN variant capable of learning long-term dependencies. Composed of forget, input, output gates.
- **BiLSTM**: 정방향과 역방향 LSTM을 결합하여 과거와 미래 양쪽 컨텍스트를 포착.
  **BiLSTM**: Combines forward and backward LSTM to capture both past and future context.

### Bayesian Deep Learning
- **MC Dropout**: 테스트 시에도 dropout을 유지하고 K번 반복 추론하여 예측 분포를 생성. 불확실성 정량화의 실용적 방법.
  **MC Dropout**: Keeping dropout active during testing and running K inference passes to generate prediction distributions. Practical method for uncertainty quantification.
- **Aleatoric uncertainty**: 데이터 자체의 노이즈로 인한 불확실성. 더 많은 데이터로도 줄일 수 없음.
  Uncertainty from inherent data noise. Cannot be reduced with more data.
- **Epistemic uncertainty**: 모델의 지식 부족으로 인한 불확실성. 더 많은 데이터로 줄일 수 있음.
  Uncertainty from model's lack of knowledge. Can be reduced with more data.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **SYM-H** | Symmetric H-component index. Ring current의 1분 해상도 지표. Dst의 고해상도 버전. / 1-min resolution ring current measure. High-resolution version of Dst. |
| **FCG (Fully Connected Graph)** | 모든 노드 쌍이 엣지로 연결된 그래프. 7개 태양풍/IMF 파라미터 + SYM-H = 8노드, 56 엣지. / Graph where every node pair is connected. 7 solar wind/IMF params + SYM-H = 8 nodes, 56 edges. |
| **GCN Layer** | 그래프 합성곱 레이어. 이웃 노드 정보를 집약하여 노드 표현을 업데이트. / Graph convolutional layer aggregating neighbor info to update node representations. |
| **BiLSTM** | 양방향 LSTM. 시퀀스를 정방향/역방향 모두 처리하여 양쪽 시간 컨텍스트 포착. / Bidirectional LSTM processing sequences in both forward and backward directions. |
| **MC Dropout** | Monte Carlo dropout. 테스트 시 dropout을 유지하고 K=100번 추론하여 예측 분포 생성. / Keeping dropout active during testing, running K=100 inferences to generate prediction distribution. |
| **Aleatoric uncertainty** | 데이터 노이즈에서 비롯된 불확실성 (줄일 수 없음). / Data noise uncertainty (irreducible). |
| **Epistemic uncertainty** | 모델 지식 부족에서 비롯된 불확실성 (더 많은 데이터로 감소 가능). / Model knowledge uncertainty (reducible with more data). |
| **FSS (Forecast Skill Score)** | Burton equation 대비 예측 스킬. 0-1 사이에서 높을수록 좋음. 음수 = Burton보다 나쁨. / Skill relative to Burton equation. Higher (0-1) is better. Negative = worse than Burton. |
| **Burton equation** | 태양풍 파라미터에서 Dst를 추정하는 경험적 공식 (1975). 베이스라인 모델로 사용. / Empirical formula estimating Dst from solar wind parameters (1975). Used as baseline. |
| **Parameter graph $G_t$** | 시점 $t$에서 7개 태양풍/IMF 파라미터 + SYM-H 값으로 구성된 완전연결 그래프. / FCG at time $t$ composed of 7 solar wind/IMF parameters + SYM-H values. |

---

## 5. 수식 미리보기 / Equations Preview

### 수식 1: RMSE

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

- $y_i$: 관측된 SYM-H 값 / observed SYM-H value
- $\hat{y}_i$: 예측된 SYM-H 값 / predicted SYM-H value
- $n$: 테스트 샘플 수 / number of test samples

### 수식 2: Forecast Skill Score (FSS)

$$\text{FSS} = 1 - \frac{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\frac{1}{n}\sum_{i=1}^{n}(y_i - y_i^b)^2}$$

- $y_i^b$: Burton equation의 예측값 / Burton equation prediction
- FSS = 1: 완벽한 예측 / perfect prediction
- FSS = 0: Burton equation과 동일 / same as Burton
- FSS < 0: Burton보다 나쁨 / worse than Burton

### 수식 3: R² (결정계수 / Coefficient of Determination)

$$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$$

### GNN 메시지 패싱 (개념적) / GNN Message Passing (conceptual)

$$h_v^{(l+1)} = \sigma\left(\sum_{u \in \mathcal{N}(v)} W^{(l)} h_u^{(l)}\right)$$

- $h_v^{(l)}$: 레이어 $l$에서 노드 $v$의 표현 / node $v$ representation at layer $l$
- $\mathcal{N}(v)$: 노드 $v$의 이웃 (FCG이므로 모든 다른 노드) / neighbors of node $v$ (all other nodes in FCG)
- $W^{(l)}$: 학습 가능한 가중치 행렬 / learnable weight matrix

---

## 6. 읽기 가이드 / Reading Guide

### 읽기 순서 권장 / Recommended Reading Order

1. **Abstract + Plain Language Summary** (p.1): "GNN + BiLSTM + Bayesian"이라는 세 가지 키워드를 먼저 파악하세요.
   Grasp the three keywords: GNN + BiLSTM + Bayesian.

2. **Figure 1 (p.4)**: 파라미터 그래프의 구조를 이해하세요. 7개 태양풍/IMF 파라미터 + SYM-H가 어떻게 시계열 그래프 시퀀스로 구성되는지.
   Understand the parameter graph structure. How 7 solar wind/IMF parameters + SYM-H form a time-series graph sequence.

3. **Figure 2 (p.5)**: 전체 SYMHnet 아키텍처. (a) 전체 구조, (b) GNN 컴포넌트, (c) BiLSTM 컴포넌트. 이것이 논문의 핵심 도표입니다.
   The full SYMHnet architecture. (a) overall, (b) GNN, (c) BiLSTM. This is the paper's key diagram.

4. **Section 3 (Methodology)** (pp.4-7): 3.1 (Parameter Graph) → 3.2 (SYMHnet Framework) → 3.3 (Uncertainty Quantification) 순서로 읽으세요.
   Read 3.1 → 3.2 → 3.3 sequentially.

5. **Table 6 (p.7)**: Ablation study 결과. SYMHnet vs SYMHnet-B vs -G vs -BG를 비교하여 각 컴포넌트의 기여를 이해하세요.
   Ablation study results. Compare variants to understand each component's contribution.

6. **Figures 3-4 (pp.8-9)**: 1분 해상도 예측 결과와 불확실성 정량화. 폭풍 #36 (moderate)과 #37 (very large)의 차이에 주목하세요.
   1-min prediction results and uncertainty quantification. Note differences between storm #36 (moderate) and #37 (very large).

7. **Tables 8-12 (pp.11-13)**: 기존 방법과의 비교. SYMHnet이 거의 모든 폭풍에서 최고 성능임을 확인하세요.
   Comparison with existing methods. Verify SYMHnet achieves best performance on nearly all storms.

### 주의 깊게 볼 포인트 / Points to Watch

- **GNN의 역할**: 왜 단순히 7개 파라미터를 벡터로 입력하지 않고 그래프로 구성하는가? GNN이 파라미터 간 상호작용을 명시적으로 학습.
  **Role of GNN**: Why structure parameters as a graph instead of a simple vector? GNN explicitly learns inter-parameter interactions.
- **1분 vs 5분 해상도**: 1분 데이터의 높은 진동성이 예측을 어렵게 하므로, 더 큰 모델(더 많은 뉴런, 높은 dropout, 더 많은 epoch)이 필요.
  **1-min vs 5-min resolution**: High oscillation of 1-min data makes prediction harder, requiring larger models.
- **Epistemic vs Aleatoric**: Figure 4에서 파란색(epistemic)이 회색(aleatoric)보다 좁은 이유를 생각해보세요.
  **Epistemic vs Aleatoric**: In Figure 4, consider why blue (epistemic) is narrower than gray (aleatoric).

---

## 7. 현대적 의의 / Modern Significance

1. **그래프 신경망의 우주기상 적용**: 태양풍/IMF 파라미터를 독립 변수가 아닌 상호작용하는 그래프로 모델링하는 패러다임. 이는 파라미터 간의 물리적 관계(예: B_z와 V의 결합 효과)를 명시적으로 학습할 수 있게 합니다.
   **GNN application to space weather**: Paradigm of modeling solar wind/IMF parameters as interacting graphs rather than independent variables, explicitly learning physical relationships.

2. **불확실성 정량화의 중요성**: 예측값뿐만 아니라 신뢰도를 함께 제공하여 의사결정자가 예보를 얼마나 신뢰할 수 있는지 판단 가능. 폭풍이 강할수록 불확실성이 커지는 직관적 결과를 보여줍니다.
   **Importance of uncertainty quantification**: Providing confidence alongside predictions so decision-makers can judge forecast reliability.

3. **1분 해상도의 의미**: SYM-H의 급격한 변화(SSC, main phase onset)를 포착할 수 있어, 실시간 운용 환경에서의 활용 가능성이 높습니다.
   **Significance of 1-min resolution**: Captures rapid SYM-H changes (SSC, main phase onset), highly applicable in real-time operational environments.

4. **SW #37과의 연결**: Billcliff et al. (2026)의 로지스틱 회귀보다 훨씬 복잡한 딥러닝 접근이지만, 목적이 다릅니다 — #37은 폭풍 발생 확률 예보(분류), #38은 SYM-H 값 자체의 연속 예측(회귀). 두 접근법은 상호보완적입니다.
   **Connection to SW #37**: Much more complex deep learning than Billcliff's logistic regression, but different purpose — #37 is storm probability (classification), #38 is continuous SYM-H value prediction (regression). The two approaches are complementary.

---

## Q&A

(읽기 세션 중 추가됨 / Populated during reading session)
