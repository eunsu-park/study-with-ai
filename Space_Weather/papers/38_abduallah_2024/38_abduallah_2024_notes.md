---
title: "Prediction of the SYM-H Index Using a Bayesian Deep Learning Method With Uncertainty Quantification"
authors: Yasser Abduallah, Khalid A. Alobaid, Jason T. L. Wang, Haimin Wang, Vania K. Jordanova, Vasyl Yurchyshyn, Husein Cavus, Ju Jing
year: 2024
journal: "Space Weather, 22, e2023SW003824"
doi: "10.1029/2023SW003824"
topic: Space Weather / Geomagnetic Index Prediction
tags: [SYM-H, GNN, BiLSTM, Bayesian deep learning, MC dropout, uncertainty quantification, geomagnetic storm, solar wind, ring current, graph neural network]
status: completed
date_started: 2026-04-15
date_completed: 2026-04-15
---

# 38. Prediction of the SYM-H Index Using a Bayesian Deep Learning Method With Uncertainty Quantification / Bayesian 딥러닝과 불확실성 정량화를 이용한 SYM-H 지수 예측

---

## 1. Core Contribution / 핵심 기여

이 논문은 SYMHnet이라는 새로운 딥러닝 프레임워크를 제안하여 SYM-H 지자기 지수를 1시간 및 2시간 전에 예측합니다. SYMHnet은 세 가지 핵심 구성요소를 결합합니다: (1) 7개의 태양풍/IMF 파라미터와 SYM-H 값을 8개 노드, 56개 엣지를 가진 완전 연결 그래프(FCG)로 구성하고 2층 그래프 합성곱 네트워크(GCN)로 파라미터 간 상호작용을 학습하는 GNN, (2) 시계열의 양방향 시간 의존성을 포착하는 BiLSTM(각 방향 400 유닛), (3) Monte Carlo dropout(K=100 샘플)을 사용한 Bayesian inference로 aleatoric(데이터) 및 epistemic(모델) 불확실성을 분리·정량화하는 메커니즘. 특히 SYM-H의 1분 해상도 예측에 딥러닝을 최초로 적용한 논문이며, 1분 데이터의 높은 진동 특성을 다루기 위해 더 큰 모델(더 많은 뉴런, 높은 dropout, 더 많은 epoch)이 필요함을 보였습니다. Solar cycle 23, 24의 42개 폭풍(1998–2018)에 대한 실험에서 SYMHnet은 1분 해상도 1시간 예측에서 RMSE=3.002 nT, FSS=0.668, R²=0.993을 달성하여 기존의 LCNN, GBM, LSTM, CNN, Burton equation 대비 우수한 성능을 보입니다. Ablation study를 통해 GNN과 BiLSTM 각각의 기여를 정량적으로 확인하였으며, epistemic 불확실성이 aleatoric 불확실성보다 작다는 것은 모델 자체보다 입력 데이터의 노이즈가 주요 불확실성 원인임을 시사합니다.

This paper proposes SYMHnet, a novel deep learning framework for predicting the SYM-H geomagnetic index 1 and 2 hours in advance. SYMHnet combines three key components: (1) a GNN that structures 7 solar wind/IMF parameters plus SYM-H as a fully connected graph (FCG) with 8 nodes and 56 edges, processed through 2 graph convolutional layers to learn inter-parameter interactions; (2) a BiLSTM (400 units per direction) that captures bidirectional temporal dependencies in time series; and (3) Bayesian inference via Monte Carlo dropout (K=100 samples) that decomposes prediction uncertainty into aleatoric (data) and epistemic (model) components. Notably, this is the first application of deep learning to 1-min resolution SYM-H prediction, demonstrating that higher-oscillation 1-min data requires larger models (more neurons, higher dropout, more epochs). Evaluated on 42 storms from solar cycles 23 and 24 (1998–2018), SYMHnet achieves RMSE=3.002 nT, FSS=0.668, and R²=0.993 for 1-min resolution 1-hr ahead prediction, outperforming LCNN, GBM, LSTM, CNN, and the Burton equation baseline. The ablation study quantitatively confirms the contribution of both GNN and BiLSTM, while the finding that epistemic uncertainty is smaller than aleatoric uncertainty indicates that input data noise, rather than model limitations, is the dominant source of prediction uncertainty.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction / 서론 (Section 1, pp.1–3)

지자기 폭풍은 태양풍-자기권 상호작용에 의해 발생하는 대규모 자기권 교란으로, 위성 운용, 전력망, GPS, 통신 시스템에 심각한 영향을 미칩니다. SYM-H(Symmetric H-component) 지수는 ring current의 강도를 1분 해상도로 측정하는 지표로, Dst 지수의 고해상도 버전입니다. SYM-H < −30 nT이면 지자기 폭풍으로 간주됩니다.

Geomagnetic storms are large-scale magnetospheric disturbances driven by solar wind–magnetosphere interaction, impacting satellites, power grids, GPS, and communications. The SYM-H (Symmetric H-component) index measures ring current intensity at 1-min resolution, serving as a high-resolution counterpart to the Dst index. A SYM-H value below −30 nT indicates a geomagnetic storm.

저자들은 기존 SYM-H 예측 연구의 한계를 세 가지로 지적합니다: (1) 대부분 5분 해상도만 다루었음 — 1분 해상도는 데이터의 높은 진동성(highly oscillating nature) 때문에 훨씬 어려움, (2) 단일 모델 아키텍처(LSTM 또는 CNN)에 의존 — 파라미터 간 상호작용을 명시적으로 학습하지 못함, (3) 예측의 불확실성을 정량화하지 않음 — 운용 환경에서 예보 신뢰도 평가가 불가능. SYMHnet은 이 세 가지 문제를 동시에 해결합니다.

The authors identify three limitations of prior SYM-H prediction research: (1) most studies only addressed 5-min resolution — 1-min is far more challenging due to SYM-H's highly oscillating nature; (2) reliance on single architectures (LSTM or CNN alone) — unable to explicitly learn inter-parameter interactions; (3) no uncertainty quantification — impossible to assess forecast reliability in operational settings. SYMHnet addresses all three simultaneously.

### Part II: Data / 데이터 (Section 2, pp.3–4)

#### 2.1 입력 파라미터 / Input Parameters

SYMHnet의 입력은 8개 파라미터로 구성됩니다:

SYMHnet uses 8 input parameters:

| 파라미터 / Parameter | 설명 / Description | 단위 / Unit |
|---|---|---|
| $B$ | IMF magnitude / IMF 크기 | nT |
| $B_y$ | IMF $y$-component / IMF $y$ 성분 | nT |
| $B_z$ | IMF $z$-component / IMF $z$ 성분 | nT |
| $V$ | Solar wind speed / 태양풍 속도 | km/s |
| $N_p$ | Proton number density / 양성자 수밀도 | cm⁻³ |
| $P_{\text{dyn}}$ | Dynamic pressure / 동압 | nPa |
| $E_F$ | Electric field / 전기장 | mV/m |
| SYM-H | Current SYM-H value / 현재 SYM-H 값 | nT |

데이터는 NASA SPDF의 OMNIWeb (https://omniweb.gsfc.nasa.gov)에서 수집하였으며, solar cycle 23과 24에 걸친 1998–2018년의 42개 지자기 폭풍을 사용합니다.

Data were collected from NASA SPDF's OMNIWeb (https://omniweb.gsfc.nasa.gov), covering 42 geomagnetic storms from 1998–2018 across solar cycles 23 and 24.

#### 2.2 데이터 분할 / Data Split

42개 폭풍은 다음과 같이 분할됩니다:

The 42 storms are split as follows:

- **Training / 훈련**: 20개 폭풍 — 모델 학습에 사용 / used for model training
- **Validation / 검증**: 5개 폭풍 — 하이퍼파라미터 튜닝에 사용 / used for hyperparameter tuning
- **Testing / 테스트**: 17개 폭풍 — 최종 성능 평가에 사용 / used for final performance evaluation

각 폭풍 이벤트는 시작 전 24시간(quiet time)부터 종료 후 24시간(recovery phase)까지 포함합니다. 결과의 일관성을 확인하기 위해 14-fold cross validation도 수행하였습니다.

Each storm event includes 24 hours before onset (quiet time) through 24 hours after end (recovery phase). 14-fold cross validation was also performed to confirm result consistency.

#### 2.3 데이터 해상도 / Data Resolution

논문은 두 가지 해상도를 모두 다룹니다:

The paper addresses both resolutions:

- **1분 해상도 / 1-min resolution**: SYM-H의 원래 해상도. 높은 진동성으로 예측이 어려움. 더 큰 모델 필요.
  The native resolution of SYM-H. Highly oscillating, making prediction difficult. Requires larger models.
- **5분 해상도 / 5-min resolution**: 기존 연구와의 비교를 위해 5분 평균 데이터도 사용.
  5-min averaged data also used for comparison with prior studies.

### Part III: Methodology — Parameter Graph / 파라미터 그래프 (Section 3.1, pp.4–5)

SYMHnet의 첫 번째 핵심 아이디어는 시점 $t$에서의 8개 파라미터를 완전 연결 그래프(FCG) $G_t$로 구성하는 것입니다. 각 노드는 하나의 파라미터를, 각 엣지는 두 파라미터 간의 관계를 나타냅니다.

SYMHnet's first key idea is to represent the 8 parameters at time $t$ as a fully connected graph (FCG) $G_t$. Each node represents one parameter, each edge represents the relationship between two parameters.

- **노드 수 / Number of nodes**: 8 (7 solar wind/IMF + SYM-H)
- **엣지 수 / Number of edges**: $8 \times 7 = 56$ (directed edges in FCG)
- **입력 시퀀스 길이 / Input sequence length**: $m = 10$ records

따라서 모델의 입력은 $m$개의 연속 그래프 $\{G_{t-m+1}, G_{t-m+2}, \ldots, G_t\}$의 시퀀스입니다. 이 그래프 시퀀스는 GNN에 의해 처리됩니다.

Thus the model input is a sequence of $m$ consecutive graphs $\{G_{t-m+1}, G_{t-m+2}, \ldots, G_t\}$, processed by the GNN.

이 접근의 핵심 동기: 태양풍 파라미터들은 독립적이지 않습니다. 예를 들어, $B_z$와 $V$의 결합 효과(southward IMF + fast solar wind)가 지자기 폭풍의 강도를 결정합니다. GNN은 이러한 파라미터 간 상호작용을 명시적으로 학습할 수 있습니다.

Key motivation: solar wind parameters are not independent. For example, the combined effect of $B_z$ and $V$ (southward IMF + fast solar wind) determines geomagnetic storm intensity. GNN can explicitly learn such inter-parameter interactions.

### Part IV: Methodology — SYMHnet Architecture / SYMHnet 아키텍처 (Section 3.2, pp.5–6)

SYMHnet은 세 가지 주요 컴포넌트로 구성됩니다 (Figure 2):

SYMHnet consists of three main components (Figure 2):

#### (a) GNN Component / GNN 컴포넌트

2층 Graph Convolutional Network(GCN)를 사용합니다. 각 GCN 레이어의 메시지 패싱 규칙:

Uses a 2-layer Graph Convolutional Network (GCN). Message passing rule for each GCN layer:

$$h_v^{(l+1)} = \sigma\left(\sum_{u \in \mathcal{N}(v)} W^{(l)} h_u^{(l)}\right)$$

여기서 $h_v^{(l)}$은 레이어 $l$에서 노드 $v$의 hidden representation, $\mathcal{N}(v)$는 노드 $v$의 이웃 집합(FCG이므로 나머지 7개 노드 전부), $W^{(l)}$은 학습 가능한 가중치 행렬, $\sigma$는 ReLU 활성화 함수입니다.

Here $h_v^{(l)}$ is node $v$'s hidden representation at layer $l$, $\mathcal{N}(v)$ is the neighbor set of node $v$ (all other 7 nodes in FCG), $W^{(l)}$ is a learnable weight matrix, and $\sigma$ is the ReLU activation function.

GNN은 각 시점 $t$에서 그래프 $G_t$를 처리하여 파라미터 간 상호작용이 반영된 노드 임베딩을 출력합니다. 이 임베딩 시퀀스가 BiLSTM의 입력이 됩니다.

The GNN processes graph $G_t$ at each time step $t$ to produce node embeddings that encode inter-parameter interactions. This embedding sequence becomes the BiLSTM input.

#### (b) BiLSTM Component / BiLSTM 컴포넌트

LSTM의 기본 게이트 방정식:

Basic LSTM gate equations:

**Forget gate / 망각 게이트:**
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

**Input gate / 입력 게이트:**
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

**Candidate cell state / 후보 셀 상태:**
$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

**Cell state update / 셀 상태 업데이트:**
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

**Output gate / 출력 게이트:**
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

**Hidden state / 은닉 상태:**
$$h_t = o_t \odot \tanh(C_t)$$

여기서 $\sigma$는 sigmoid 함수, $\odot$는 element-wise 곱, $x_t$는 시점 $t$의 입력(GNN 출력), $h_{t-1}$는 이전 hidden state입니다.

Here $\sigma$ is the sigmoid function, $\odot$ is element-wise multiplication, $x_t$ is the input at time $t$ (GNN output), and $h_{t-1}$ is the previous hidden state.

BiLSTM은 정방향(forward) LSTM $\overrightarrow{h}_t$와 역방향(backward) LSTM $\overleftarrow{h}_t$를 결합합니다:

BiLSTM concatenates forward LSTM $\overrightarrow{h}_t$ and backward LSTM $\overleftarrow{h}_t$:

$$h_t^{\text{BiLSTM}} = [\overrightarrow{h}_t \| \overleftarrow{h}_t]$$

각 방향의 LSTM은 400 유닛으로 구성되어 총 800차원의 출력을 생성합니다.

Each direction has 400 units, producing an 800-dimensional output.

#### (c) Dense Layer / 완전 연결 레이어

BiLSTM의 출력은 200 뉴런의 Dense 레이어를 거쳐 최종 SYM-H 예측값 $\hat{y}_{t+\Delta t}$을 출력합니다. $\Delta t$는 예측 리드 타임(1시간 또는 2시간)입니다.

The BiLSTM output passes through a 200-neuron Dense layer to produce the final SYM-H prediction $\hat{y}_{t+\Delta t}$, where $\Delta t$ is the forecast lead time (1 or 2 hours).

#### Architecture Summary Table / 아키텍처 요약 표 (Table 4)

| Component / 구성요소 | Specification / 사양 |
|---|---|
| GNN | 8 nodes, 56 edges, 2 GCN layers, ReLU activation |
| Forward LSTM | 400 units |
| Backward LSTM | 400 units |
| Dense layer | 200 neurons |
| Output | 1 neuron (predicted SYM-H) |

#### Hyperparameters / 하이퍼파라미터 (Table 5)

| Parameter / 파라미터 | 1-min resolution | 5-min resolution |
|---|---|---|
| Dropout rate | 0.5 | 0.2 |
| Batch size | 1024 | 512 |
| Epochs | 50 | 30 |
| Optimizer | RMSProp | RMSProp |
| Learning rate | 0.0002 | 0.0002 |
| Loss function | MSE | MSE |

1분 해상도 데이터는 5분 해상도보다 높은 진동성을 가지므로, 더 높은 dropout(0.5 vs 0.2), 더 큰 batch size(1024 vs 512), 더 많은 학습 epoch(50 vs 30)이 필요합니다. 이는 모델이 1분 데이터의 노이즈를 과적합하지 않도록 정규화를 강화해야 하기 때문입니다.

The 1-min resolution data has higher oscillation than 5-min, requiring higher dropout (0.5 vs 0.2), larger batch size (1024 vs 512), and more training epochs (50 vs 30). This strengthens regularization to prevent overfitting to 1-min data noise.

### Part V: Methodology — Uncertainty Quantification / 불확실성 정량화 (Section 3.3, pp.6–7)

#### Bayesian Inference via MC Dropout / MC Dropout을 통한 Bayesian Inference

전통적인 Bayesian neural network은 가중치에 대한 사후분포 $P(W|X,Y)$를 직접 계산하지만, 이는 계산적으로 불가능합니다(intractable). 대신, variational inference를 사용하여 사후분포를 근사합니다:

Traditional Bayesian neural networks compute the posterior distribution $P(W|X,Y)$ over weights directly, but this is computationally intractable. Instead, variational inference approximates the posterior:

$$\text{minimize} \quad \text{KL}\left(q(W) \| P(W|X,Y)\right)$$

여기서 $q(W)$는 dropout을 적용한 가중치 분포로, tractable한 variational distribution입니다. Gal & Ghahramani (2016)는 테스트 시에도 dropout을 유지하는 것이 variational Bayesian inference의 근사와 수학적으로 동치임을 증명했습니다.

Here $q(W)$ is the weight distribution with dropout applied, serving as a tractable variational distribution. Gal & Ghahramani (2016) proved that keeping dropout active during testing is mathematically equivalent to approximate variational Bayesian inference.

**MC Dropout 절차 / MC Dropout Procedure:**

1. 테스트 시 dropout을 유지한 채 동일한 입력 $x^*$에 대해 $K=100$번 추론을 반복합니다.
   Run $K=100$ inference passes on the same input $x^*$ with dropout active during testing.

2. 예측 평균(최종 예측) / Predictive mean (final prediction):
$$\hat{y}(x^*) = \frac{1}{K}\sum_{k=1}^{K} f^{W_k}(x^*)$$

3. 전체 불확실성 / Total uncertainty:
$$\text{Var}(y^*) = \frac{1}{K}\sum_{k=1}^{K}\left(f^{W_k}(x^*) - \hat{y}(x^*)\right)^2$$

#### 불확실성 분리 / Uncertainty Decomposition

전체 불확실성은 두 가지 성분으로 분리됩니다:

Total uncertainty decomposes into two components:

**Aleatoric uncertainty (데이터 불확실성):**
$$\sigma_{\text{aleatoric}}^2 = \frac{1}{K}\sum_{k=1}^{K}\sigma_k^2$$

- 데이터 자체의 노이즈에서 기인 / Originates from inherent data noise
- 더 많은 데이터로도 줄일 수 없음 / Cannot be reduced with more data
- 태양풍 데이터의 측정 오차, 결측값 보간 등 / Solar wind measurement errors, data gap interpolation, etc.

**Epistemic uncertainty (모델 불확실성):**
$$\sigma_{\text{epistemic}}^2 = \frac{1}{K}\sum_{k=1}^{K}\left(f^{W_k}(x^*) - \hat{y}(x^*)\right)^2 - \sigma_{\text{aleatoric}}^2$$

- 모델의 지식 부족에서 기인 / Originates from model's lack of knowledge
- 더 많은 데이터/더 좋은 모델로 줄일 수 있음 / Can be reduced with more data or better models

논문의 핵심 발견: **epistemic 불확실성 < aleatoric 불확실성**. 이는 모델의 학습이 충분하며, 예측 불확실성의 주된 원인이 입력 데이터의 노이즈임을 의미합니다.

Key finding: **epistemic uncertainty < aleatoric uncertainty**. This means the model has learned sufficiently and the dominant source of prediction uncertainty is input data noise.

### Part VI: Evaluation Metrics / 평가 지표 (Section 4, pp.7)

#### Metric 1: RMSE (Eq. 1)

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

- $y_i$: 관측된 SYM-H / observed SYM-H
- $\hat{y}_i$: 예측된 SYM-H / predicted SYM-H
- 낮을수록 좋음. 단위: nT / Lower is better. Unit: nT

#### Metric 2: Forecast Skill Score — FSS (Eq. 2)

$$\text{FSS} = 1 - \frac{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\frac{1}{n}\sum_{i=1}^{n}(y_i - y_i^b)^2}$$

- $y_i^b$: Burton equation의 예측값 / Burton equation prediction
- FSS = 1: 완벽한 예측 / perfect prediction
- FSS = 0: Burton equation과 동일 / same as Burton equation
- FSS < 0: Burton equation보다 나쁨 / worse than Burton equation
- 높을수록 좋음 / Higher is better

Burton equation (Burton et al., 1975)은 태양풍 파라미터로부터 Dst(≈SYM-H)를 경험적으로 추정하는 공식으로, 물리 기반 베이스라인으로 사용됩니다.

The Burton equation (Burton et al., 1975) empirically estimates Dst (≈SYM-H) from solar wind parameters, serving as a physics-based baseline.

#### Metric 3: R² (Eq. 3)

$$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$$

- $\bar{y}$: 관측값의 평균 / mean of observed values
- R² = 1: 완벽한 예측 / perfect prediction
- R² = 0: 평균만으로 예측하는 것과 동일 / same as predicting the mean

### Part VII: Results — Ablation Study / 절제 연구 (Section 4.1, pp.7–8)

Ablation study는 SYMHnet의 각 컴포넌트 기여도를 정량적으로 평가합니다. 네 가지 변형 모델을 비교합니다 (Table 6):

The ablation study quantitatively evaluates each component's contribution via four model variants (Table 6):

| Model / 모델 | GNN | BiLSTM | Description / 설명 |
|---|---|---|---|
| **SYMHnet** | ✓ | ✓ | Full model / 전체 모델 |
| **SYMHnet-B** | ✓ | ✗ | No BiLSTM (GNN + unidirectional LSTM) / BiLSTM 제거 |
| **SYMHnet-G** | ✗ | ✓ | No GNN (BiLSTM only) / GNN 제거 |
| **SYMHnet-BG** | ✗ | ✗ | No GNN, no BiLSTM (basic LSTM) / 둘 다 제거 |

**핵심 결과 (1-min, 1-hr ahead) / Key results:**

| Model | RMSE (nT) | FSS | R² |
|---|---|---|---|
| SYMHnet | **3.002** | **0.668** | **0.993** |
| SYMHnet-B | 3.132 | 0.639 | 0.992 |
| SYMHnet-G | 4.508 | 0.252 | 0.985 |
| SYMHnet-BG | 4.615 | 0.215 | 0.984 |

핵심 관찰:
Key observations:

1. **SYMHnet > SYMHnet-B**: BiLSTM이 단방향 LSTM보다 시간 의존성을 더 잘 포착합니다.
   BiLSTM captures temporal dependencies better than unidirectional LSTM.

2. **SYMHnet-B ≫ SYMHnet-G**: GNN이 있는 모델(SYMHnet-B)이 GNN이 없는 모델(SYMHnet-G)보다 훨씬 우수합니다. 이는 **GNN이 시계열 회귀에도 효과적**임을 보여줍니다.
   Models with GNN (SYMHnet-B) significantly outperform those without (SYMHnet-G), demonstrating **GNN's effectiveness for time series regression**.

3. **SYMHnet-G ≈ SYMHnet-BG**: GNN이 없으면 BiLSTM의 추가 효과가 미미합니다. 이전 연구(Siciliano et al., 2021)의 발견과 일치합니다.
   Without GNN, BiLSTM's additional benefit is marginal, consistent with prior findings (Siciliano et al., 2021).

### Part VIII: Results — Main Performance / 주요 성능 (Section 4.2–4.3, pp.8–12)

#### 1-min Resolution Results / 1분 해상도 결과

| Setting / 설정 | RMSE (nT) | FSS | R² |
|---|---|---|---|
| 1-min, 1-hr ahead | 3.002 | 0.668 | 0.993 |
| 1-min, 2-hr ahead | 3.171 | 0.760 | 0.993 |

2시간 예측의 FSS(0.760)가 1시간 예측(0.668)보다 높은 이유: Burton equation은 리드 타임이 길수록 성능이 급격히 저하되지만, SYMHnet은 상대적으로 안정적이므로 Burton 대비 스킬이 오히려 증가합니다.

The 2-hr FSS (0.760) exceeds 1-hr FSS (0.668) because the Burton equation degrades more rapidly at longer lead times while SYMHnet remains relatively stable, yielding higher relative skill.

#### 5-min Resolution Results / 5분 해상도 결과

| Setting / 설정 | RMSE (nT) | FSS | R² |
|---|---|---|---|
| 5-min, 1-hr ahead | 5.914 | 0.484 | 0.993 |
| 5-min, 2-hr ahead | 7.397 | 0.620 | 0.990 |

5분 해상도의 RMSE가 1분보다 높은 이유: 5분 평균화 과정에서 진동이 평활화되지만, 급격한 변화의 타이밍 오차가 더 큰 RMSE로 나타납니다.

The higher RMSE at 5-min resolution occurs because while averaging smooths oscillations, timing errors in rapid changes produce larger RMSE.

#### Comparison with Existing Methods / 기존 방법과의 비교 (Tables 8–12)

17개 테스트 폭풍에 대해 SYMHnet은 LCNN (Collado-Villaverde et al., 2021), GBM (long et al., 2022), LSTM, CNN, Burton equation과 비교되었습니다. SYMHnet은 거의 모든 폭풍에서 최고 성능을 달성했습니다.

SYMHnet was compared against LCNN, GBM, LSTM, CNN, and Burton equation across 17 test storms, achieving best performance on nearly all storms.

5분 해상도 비교 (representative):

| Method / 방법 | RMSE (nT) 1-hr | FSS 1-hr |
|---|---|---|
| **SYMHnet** | **5.914** | **0.484** |
| LCNN | 7.229 | 0.229 |
| GBM | 8.186 | 0.013 |
| LSTM | 8.363 | −0.031 |
| CNN | 8.500 | −0.065 |
| Burton | 9.056 | — (baseline) |

LSTM과 CNN은 FSS가 음수로, Burton equation보다도 성능이 낮습니다. 이는 복잡한 모델이 항상 더 좋은 것은 아니며, 파라미터 간 상호작용 학습(GNN)이 중요함을 보여줍니다.

LSTM and CNN have negative FSS, performing worse than even the Burton equation. This shows complex models are not always better — learning inter-parameter interactions (GNN) is crucial.

### Part IX: Results — Case Studies / 사례 연구 (Section 4.4, pp.8–10)

두 개의 대표 폭풍에 대한 상세 분석:

Detailed analysis of two representative storms:

#### Storm #36 — Moderate Storm (2018-08-25)

- **SYM-H 최솟값 / SYM-H minimum**: −137 nT
- **예측 오차 범위 / Prediction error range** (1-min, 1-hr): −15 ~ +23 nT
- 수치적 예시 / Numerical example: SYM-H 최솟값 구간에서 예측값은 약 −120 nT로, 관측값(−137 nT)보다 약 17 nT 높음. 이는 main phase의 급격한 하강을 약간 과소추정하는 경향을 보여줍니다.
  Near the SYM-H minimum, the predicted value was approximately −120 nT compared to the observed −137 nT, showing a ~17 nT overestimation. This reflects a tendency to slightly underestimate rapid main phase descent.

#### Storm #37 — Very Large Storm (2015-03-17, St. Patrick's Day)

- **SYM-H 최솟값 / SYM-H minimum**: −393 nT (본 데이터셋에서 가장 강한 폭풍 / strongest storm in dataset)
- **예측 오차 범위 / Prediction error range** (1-min, 1-hr): −50 ~ +34 nT
- 불확실성이 moderate storm보다 훨씬 큰 것은 직관적으로 타당합니다 — 극단적인 폭풍은 태양풍 조건의 급격한 변화를 수반하므로 예측이 더 어렵습니다.
  The larger uncertainty compared to the moderate storm is intuitive — extreme storms involve rapid solar wind changes, making prediction more difficult.

### Part X: Results — Uncertainty Analysis / 불확실성 분석 (Section 4.5, pp.9–11)

Figure 4에서 불확실성의 구조가 시각화됩니다:

Figure 4 visualizes the uncertainty structure:

- **회색 밴드 / Gray band**: Aleatoric uncertainty (±2σ) — 데이터 노이즈 / data noise
- **파란색 밴드 / Blue band**: Epistemic uncertainty (±2σ) — 모델 불확실성 / model uncertainty

핵심 관찰:
Key observations:

1. **Epistemic < Aleatoric**: 모든 폭풍에서 epistemic 불확실성이 aleatoric보다 작습니다. 이는 모델이 충분히 학습되었으며, 주된 불확실성 원인이 입력 데이터의 노이즈임을 의미합니다.
   Epistemic uncertainty is smaller than aleatoric across all storms, indicating sufficient model training and that input data noise is the primary uncertainty source.

2. **폭풍 강도와 불확실성의 상관 / Storm intensity–uncertainty correlation**: 강한 폭풍(#37, SYM-H min=−393 nT)에서 불확실성이 더 큽니다. 이는 극단적 이벤트의 예측이 본질적으로 더 불확실함을 반영합니다.
   Stronger storms (#37, SYM-H min=−393 nT) show larger uncertainty, reflecting the inherent difficulty of predicting extreme events.

3. **Main phase에서 불확실성 최대 / Maximum uncertainty during main phase**: SYM-H가 급격히 감소하는 main phase에서 두 불확실성 모두 최대가 됩니다. Recovery phase에서는 불확실성이 감소합니다.
   Both uncertainties peak during the rapid SYM-H decrease in the main phase and decrease during recovery.

### Part XI: Cross-Validation / 교차 검증 (Section 4.6, p.13)

14-fold cross validation을 수행하여 결과의 일관성을 확인했습니다. 42개 폭풍을 3개씩 그룹으로 나누어, 매번 3개 폭풍을 테스트 세트로 사용하고 나머지를 학습/검증 세트로 사용합니다. 결과는 기본 설정과 일관되게 SYMHnet이 최고 성능을 유지했습니다.

14-fold cross validation confirmed result consistency. The 42 storms were divided into groups of 3, each serving as the test set while the remainder was used for training/validation. Results consistently showed SYMHnet maintaining best performance.

### Part XII: Discussion / 토의 (Section 5, pp.14–16)

저자들은 SYM-H < −30 nT 임계값을 사용하여 폭풍 발생 감지가 가능함을 보여줍니다. SYMHnet의 연속 예측값이 이 임계값을 교차하는 시점을 폭풍 시작/종료로 판단할 수 있습니다.

The authors demonstrate that a SYM-H threshold of −30 nT can be used to detect storm occurrence. The time when SYMHnet's continuous predictions cross this threshold marks storm onset/end.

논문 코드는 Zenodo(https://doi.org/10.5281/zenodo.10602518)에 공개되어 있어 재현 가능합니다.

The paper's code is publicly available on Zenodo (https://doi.org/10.5281/zenodo.10602518) for reproducibility.

---

## 3. Key Takeaways / 핵심 시사점

1. **GNN은 시계열 회귀에 효과적이다** — 태양풍/IMF 파라미터를 독립 변수가 아닌 상호작용하는 그래프로 모델링하는 것이 단순히 벡터로 입력하는 것보다 훨씬 우수한 성능을 보입니다. Ablation study에서 GNN 제거 시(SYMHnet-G) RMSE가 3.002→4.508 nT로 50% 증가합니다.
   **GNN is effective for time series regression** — modeling solar wind/IMF parameters as interacting graph nodes significantly outperforms simple vector input. Removing GNN (SYMHnet-G) increases RMSE from 3.002 to 4.508 nT (50% increase).

2. **1분 해상도 SYM-H 예측이 실현 가능하다** — 이전 연구들이 5분 해상도에 머물렀던 것과 달리, SYMHnet은 1분 해상도에서도 RMSE=3.002 nT, R²=0.993이라는 높은 정확도를 달성합니다. 단, 더 큰 모델과 강한 정규화가 필요합니다.
   **1-min resolution SYM-H prediction is feasible** — unlike prior studies limited to 5-min resolution, SYMHnet achieves RMSE=3.002 nT and R²=0.993 at 1-min. However, larger models and stronger regularization are required.

3. **불확실성 정량화가 운용 예보에 필수적이다** — MC dropout으로 aleatoric/epistemic 불확실성을 분리하면 의사결정자가 예보 신뢰도를 판단할 수 있습니다. 강한 폭풍일수록 불확실성이 커지므로, 예보의 한계를 정직하게 전달하는 것이 중요합니다.
   **Uncertainty quantification is essential for operational forecasting** — separating aleatoric/epistemic uncertainty via MC dropout enables decision-makers to assess forecast reliability. Stronger storms show larger uncertainty, making honest communication of forecast limits critical.

4. **Epistemic 불확실성이 aleatoric보다 작다** — 이는 모델이 충분히 학습되었으며, 성능 개선의 병목이 모델 아키텍처가 아닌 입력 데이터 품질임을 시사합니다. 태양풍 데이터의 측정 정확도 향상이 예측 개선의 핵심입니다.
   **Epistemic uncertainty is smaller than aleatoric** — the model has learned sufficiently and the performance bottleneck is input data quality, not model architecture. Improving solar wind measurement accuracy is key to better predictions.

5. **BiLSTM의 기여는 GNN 위에 쌓일 때 의미 있다** — Ablation study에서 GNN 없이 BiLSTM만 사용(SYMHnet-G)하면 기본 LSTM(SYMHnet-BG)과 큰 차이가 없지만, GNN과 결합하면 RMSE가 3.132→3.002 nT로 개선됩니다.
   **BiLSTM's contribution is meaningful on top of GNN** — without GNN, BiLSTM alone (SYMHnet-G) barely improves over basic LSTM (SYMHnet-BG), but combined with GNN, RMSE improves from 3.132 to 3.002 nT.

6. **장기 예측에서 딥러닝의 상대적 우위가 커진다** — FSS가 1시간(0.668)보다 2시간(0.760) 예측에서 더 높은 것은, Burton equation이 먼 리드 타임에서 급격히 성능 저하하는 반면 SYMHnet은 안정적임을 보여줍니다.
   **Deep learning's relative advantage increases for longer forecasts** — higher FSS at 2-hr (0.760) vs 1-hr (0.668) shows that while the Burton equation degrades rapidly at longer lead times, SYMHnet remains stable.

7. **극단적 폭풍에서의 과소예측 경향** — St. Patrick's Day storm(SYM-H min=−393 nT)에서 최대 −50 nT의 오차가 발생했습니다. 극단적 이벤트가 학습 데이터에 희소하기 때문이며, 이는 space weather 머신러닝의 일반적 한계입니다.
   **Underprediction tendency for extreme storms** — the St. Patrick's Day storm (SYM-H min=−393 nT) showed errors up to −50 nT. Extreme events are rare in training data, a common limitation in space weather ML.

8. **재현성이 보장된다** — 데이터(NASA SPDF)와 코드(Zenodo)가 모두 공개되어 있어, 결과의 독립적 검증과 확장이 가능합니다. 이는 오픈 사이언스의 모범 사례입니다.
   **Reproducibility is ensured** — both data (NASA SPDF) and code (Zenodo) are publicly available, enabling independent verification and extension. This exemplifies open science best practices.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Parameter Graph Construction / 파라미터 그래프 구성

시점 $t$에서의 8개 파라미터 값을 완전 연결 그래프로 구성:

Construct a fully connected graph from 8 parameter values at time $t$:

$$G_t = (V, E), \quad |V| = 8, \quad |E| = 56$$

입력 시퀀스: $\{G_{t-m+1}, \ldots, G_t\}$ where $m=10$.

### 4.2 GCN Message Passing / GCN 메시지 패싱

$$h_v^{(l+1)} = \sigma\left(\sum_{u \in \mathcal{N}(v)} W^{(l)} h_u^{(l)}\right)$$

- 2 layers ($l = 0, 1$), $\sigma = \text{ReLU}$
- FCG이므로 $|\mathcal{N}(v)| = 7$ for all $v$ / In FCG, every node has 7 neighbors

### 4.3 LSTM Gates / LSTM 게이트

$$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f), \quad i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$$

$$\tilde{C}_t = \tanh(W_C [h_{t-1}, x_t] + b_C), \quad C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

$$o_t = \sigma(W_o [h_{t-1}, x_t] + b_o), \quad h_t = o_t \odot \tanh(C_t)$$

### 4.4 BiLSTM Concatenation / BiLSTM 결합

$$h_t^{\text{BiLSTM}} = [\overrightarrow{h}_t \| \overleftarrow{h}_t] \in \mathbb{R}^{800}$$

- $\overrightarrow{h}_t \in \mathbb{R}^{400}$: forward LSTM, $\overleftarrow{h}_t \in \mathbb{R}^{400}$: backward LSTM

### 4.5 MC Dropout Inference / MC Dropout 추론

$$\hat{y}(x^*) = \frac{1}{K}\sum_{k=1}^{K} f^{W_k}(x^*), \quad K = 100$$

### 4.6 Uncertainty Decomposition / 불확실성 분리

$$\sigma_{\text{total}}^2 = \underbrace{\frac{1}{K}\sum_{k=1}^{K}\sigma_k^2}_{\text{aleatoric}} + \underbrace{\frac{1}{K}\sum_{k=1}^{K}\left(f^{W_k}(x^*) - \hat{y}(x^*)\right)^2 - \frac{1}{K}\sum_{k=1}^{K}\sigma_k^2}_{\text{epistemic}}$$

### 4.7 Variational Inference Objective / Variational Inference 목적함수

$$\hat{W} = \arg\min_W \text{KL}\left(q(W) \| P(W|X,Y)\right)$$

Dropout을 적용한 $q(W)$가 진정한 사후분포 $P(W|X,Y)$를 근사합니다.

$q(W)$ with dropout approximates the true posterior $P(W|X,Y)$.

### 4.8 Evaluation Metrics / 평가 지표

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

$$\text{FSS} = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - y_i^b)^2}$$

$$R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

### 4.9 Worked Example: FSS Computation / FSS 계산 예시

SYMHnet 1-min, 1-hr ahead 결과를 사용한 수치 예시:

Numerical example using SYMHnet 1-min, 1-hr ahead results:

- RMSE(SYMHnet) = 3.002 nT → MSE = $3.002^2 = 9.012$ nT²
- RMSE(Burton) ≈ 5.21 nT (FSS=0.668에서 역산) → MSE = $5.21^2 ≈ 27.14$ nT²
  (Derived from FSS=0.668)
- $\text{FSS} = 1 - \frac{9.012}{27.14} = 1 - 0.332 = 0.668$ ✓

이는 SYMHnet이 Burton equation 대비 MSE를 66.8% 줄였음을 의미합니다.

This means SYMHnet reduced MSE by 66.8% relative to the Burton equation.

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1975  Burton et al.              Dst 경험적 관계식 / Empirical Dst equation
      │                          → 물리 기반 베이스라인 / physics-based baseline
      │
1996  Gleisner et al.            최초 ANN 기반 Dst/Kp 예측 / First ANN Dst/Kp prediction
      │                          → ML의 우주기상 진입 / ML enters space weather
      │
2010  Cai et al.                 NARX로 SYM-H 5분 예측 / SYM-H 5-min with NARX
      │                          → SYM-H 예측의 시작 / SYM-H prediction begins
      │
2016  Gal & Ghahramani           MC dropout ≈ Bayesian inference 증명
      │                          → 실용적 불확실성 정량화 / practical UQ
      │
2018  Gruet et al.               LSTM으로 Dst 예측 / Dst prediction with LSTM
      │                          → 딥러닝의 지자기 지수 예측 적용 / DL for geomagnetic indices
      │
2021  Siciliano et al.           LSTM+CNN으로 SYM-H 5분 비교 / SYM-H 5-min with LSTM+CNN
      │
2021  Collado-Villaverde et al.  LCNN (LSTM+CNN) SYM-H 5분 / LCNN for SYM-H 5-min
      │                          → 복합 아키텍처 시대 / era of hybrid architectures
      │
2022  long et al.                GBM으로 SYM-H 5분 / GBM for SYM-H 5-min
      │                          → 앙상블 기법 적용 / ensemble methods applied
      │
2024  ★ Abduallah et al.         SYMHnet: GNN + BiLSTM + Bayesian
      │                          → 최초 1분 해상도, 불확실성 정량화 / first 1-min, UQ
      │
2026  Billcliff et al.           태양풍 앙상블 + 로지스틱 회귀 / ensemble + logistic regression
                                 → 분류적 접근 (상호보완) / classification approach (complementary)
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **SW #11 — Burton et al. (1975)** | FSS 계산의 베이스라인으로 사용. 태양풍 파라미터로부터 Dst를 경험적으로 추정하는 물리 기반 모델. / Used as FSS baseline. Physics-based model estimating Dst from solar wind parameters. | Burton equation은 긴 리드 타임에서 성능이 급격히 저하되어, SYMHnet의 상대적 우위가 커짐. / Burton degrades at longer lead times, making SYMHnet's relative advantage larger. |
| **SW #37 — Billcliff et al. (2026)** | 상호보완적 접근: Billcliff는 폭풍 발생 확률 예보(분류), SYMHnet은 SYM-H 값 예측(회귀). / Complementary approaches: Billcliff predicts storm probability (classification), SYMHnet predicts SYM-H values (regression). | 두 모델을 결합하면 "폭풍이 올 확률"과 "폭풍의 강도"를 모두 예보 가능. / Combining both enables forecasting both storm probability and intensity. |
| **SW #33 — Camporeale et al. (2019)** | 우주기상 분야의 ML 응용 리뷰. SYMHnet이 이 리뷰에서 식별된 주요 과제(불확실성 정량화, 해석 가능성)를 직접 다룸. / ML in space weather review. SYMHnet directly addresses key challenges identified (UQ, interpretability). | 리뷰의 권고사항 — 특히 불확실성 정량화와 모델 간 비교 — 을 SYMHnet이 구현. / SYMHnet implements the review's recommendations, especially UQ and model comparison. |
| **Collado-Villaverde et al. (2021)** | LCNN(LSTM+CNN) 모델. SYMHnet의 직접 비교 대상. 5분 해상도에서 SYMHnet이 RMSE 기준 18% 우수. / LCNN (LSTM+CNN) model. Direct comparison target. SYMHnet outperforms by ~18% RMSE at 5-min. | LCNN은 파라미터를 독립적으로 처리하지만, SYMHnet의 GNN은 상호작용을 학습하여 성능 차이 발생. / LCNN processes parameters independently, while SYMHnet's GNN learns interactions. |
| **long et al. (2022)** | GBM(Gradient Boosting Machine) 모델. SYMHnet의 또 다른 비교 대상. 앙상블 트리 기반 접근. / GBM model. Another comparison target. Ensemble tree-based approach. | GBM은 FSS=0.013으로 Burton과 거의 동일한 성능. 딥러닝이 전통 ML보다 우수함을 입증. / GBM achieves FSS=0.013, nearly equivalent to Burton. Demonstrates deep learning superiority over traditional ML. |
| **Siciliano et al. (2021)** | LSTM+CNN 결합 모델로 SYM-H 5분 예측 비교 연구. GNN 없이 LSTM만으로는 충분하지 않다는 발견과 일치. / LSTM+CNN comparison for SYM-H 5-min. Consistent with finding that LSTM alone is insufficient without GNN. | SYMHnet-G(GNN 없음)의 낮은 성능이 이 논문의 결론을 지지. / SYMHnet-G's poor performance supports their conclusion. |

---

## 7. References / 참고문헌

- Abduallah, Y., Alobaid, K. A., Wang, J. T. L., Wang, H., Jordanova, V. K., Yurchyshyn, V., Cavus, H., & Jing, J. (2024). "Prediction of the SYM-H Index Using a Bayesian Deep Learning Method With Uncertainty Quantification," *Space Weather*, 22, e2023SW003824. [DOI: 10.1029/2023SW003824]
- Burton, R. K., McPherron, R. L., & Russell, C. T. (1975). "An empirical relationship between interplanetary conditions and Dst," *Journal of Geophysical Research*, 80(31), 4204–4214.
- Gleisner, H., Lundstedt, H., & Wintoft, P. (1996). "Predicting geomagnetic storms from solar wind data using time-delay neural networks," *Annales Geophysicae*, 14, 679–686.
- Gruet, M. A., Chandorkar, M., Sicard, A., & Mottez, F. (2018). "Multiple-hour-ahead forecast of the Dst index using a combination of long short-term memory neural network and Gaussian process," *Space Weather*, 16, 1882–1896.
- Siciliano, F., Consolini, G., Tozzi, R., Gentili, M., Giannattasio, F., & De Michelis, P. (2021). "Forecasting SYM-H index: A comparison between long short-term memory and convolutional neural networks," *Space Weather*, 19, e2020SW002589.
- Collado-Villaverde, A., Muñoz, P., & Cid, C. (2021). "Deep neural networks with convolutional and LSTM layers for SYM-H and ASY-H forecasting," *Space Weather*, 19, e2021SW002748.
- long, Y., Yang, Y., & Feng, X. (2022). "SYM-H forecasting with a gradient boosting machine," *Space Weather*, 20, e2022SW003131.
- Gal, Y., & Ghahramani, Z. (2016). "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning," *Proceedings of the 33rd ICML*.
- Camporeale, E. (2019). "The Challenge of Machine Learning in Space Weather: Nowcasting and Forecasting," *Space Weather*, 17, 1166–1207.
- Billcliff, M., Smith, A. W., Owens, M., Woo, W. L., Barnard, L., Edward-Inatimi, N., & Rae, I. J. (2026). "Extended Lead-Time Geomagnetic Storm Forecasting With Solar Wind Ensembles and Machine Learning," *Space Weather*, 24, e2025SW004823.
- NASA SPDF OMNIWeb: https://omniweb.gsfc.nasa.gov
- Code repository: https://doi.org/10.5281/zenodo.10602518
