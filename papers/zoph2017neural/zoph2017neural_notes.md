---
title: "Neural Architecture Search with Reinforcement Learning"
authors: [Barret Zoph, Quoc V. Le]
year: 2017
journal: "ICLR 2017 (arXiv:1611.01578)"
doi: "10.48550/arXiv.1611.01578"
topic: Artificial_Intelligence
tags: [neural-architecture-search, reinforcement-learning, automl, REINFORCE, controller-RNN, CIFAR-10, Penn-Treebank]
status: completed
date_started: 2026-04-28
date_completed: 2026-04-28
---

# 27. Neural Architecture Search with Reinforcement Learning / 강화학습을 이용한 신경망 구조 탐색

---

## 1. Core Contribution / 핵심 기여

**한국어:** 본 논문은 신경망의 구조 자체를 학습하는 문제를 강화학습으로 정식화하고, 이를 대규모 자원으로 실증한 NAS(Neural Architecture Search) 분야의 출발점이다. 핵심 아이디어는 두 단계로 분해된다. (1) RNN controller가 자식 네트워크의 hyperparameter를 가변 길이 토큰 시퀀스(예: filter height, filter width, stride, #filters, …)로 자동회귀적(auto-regressive)으로 샘플링한다. (2) 샘플링된 architecture로 만든 child network를 실제로 학습시킨 뒤 검증 세트 정확도를 reward $R$로 받아, REINFORCE 정책 경사법으로 controller의 파라미터 $\theta_c$를 업데이트한다. 보상이 미분 불가하므로 표준 backprop이 불가능하다는 점이 RL을 도입하는 핵심 동기다. baseline(이전 보상의 EMA)으로 분산을 줄이고, parameter server 위에서 K=20–100개의 controller replica × m=1–8개의 child가 800 GPU(또는 400 CPU)에서 비동기로 학습된다. CIFAR-10에서 12,800개 architecture를 평가해 39-layer CNN을 찾았으며 test error 3.65%(DenseNet 3.46%에 근접하면서 1.05x 빠름), Penn Treebank에서는 $6\times10^{16}$가지의 새로운 RNN cell 공간에서 perplexity 62.4(이전 SOTA 대비 -3.6)의 cell을 자동으로 발견했다. 이 cell은 TensorFlow에 NASCell로 통합되었고 GNMT에 단순 drop-in시 BLEU +0.5의 개선을 보였다.

**English:** This paper is the foundational work that frames neural network architecture design as a reinforcement learning problem and empirically demonstrates the framework at scale, founding the Neural Architecture Search (NAS) field. The method has two parts. (1) An RNN controller auto-regressively samples a variable-length sequence of architecture tokens (filter height, filter width, stride, #filters, etc.) describing a child network. (2) The child network is actually trained on real data and its held-out validation accuracy serves as a non-differentiable reward $R$ used by the REINFORCE policy gradient to update the controller's parameters $\theta_c$. The non-differentiability of accuracy is the precise reason RL is invoked. A baseline (exponential moving average of past rewards) reduces gradient variance, and a parameter-server scheme parallelizes training across K=20–100 controller replicas × m=1–8 child replicas on 800 GPUs (or 400 CPUs for the PTB experiment). Evaluating 12,800 CIFAR-10 architectures yields a 39-layer CNN at 3.65% test error (vs DenseNet's 3.46% but 1.05x faster), and searching a $6\times10^{16}$-cell space on Penn Treebank discovers a novel recurrent cell that achieves test perplexity 62.4 (3.6 perplexity below the prior SOTA). The discovered cell was integrated into TensorFlow as `NASCell`, transferred to PTB character-level (1.214 bpc — SOTA), and gave +0.5 BLEU when dropped into GNMT for English→German.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction (§1, p.1–2) / 도입부

**한국어:** 저자는 SIFT/HOG 시대의 "feature engineering"이 AlexNet 이후 "architecture engineering"으로 옮겨갔지만, 여전히 사람이 수많은 hyperparameter를 손으로 정해야 한다는 문제를 지적한다. 핵심 관찰: "신경망 구조는 가변 길이 문자열로 표현 가능하다." 따라서 RNN — 즉 controller — 이 그 문자열을 만들 수 있고, child network의 검증 정확도를 강화학습 보상으로 쓰면 controller의 분포는 점차 좋은 구조 쪽으로 이동한다. Figure 1의 sample → train → reward → policy gradient의 닫힌 루프가 전체 그림이다.

**English:** The introduction frames the field's evolution: feature engineering (SIFT, HOG) → architecture engineering (AlexNet through ResNet). The key observation is that a neural network's structure can be encoded as a variable-length string. Therefore an RNN — the *controller* — can generate that string, and the validation accuracy of the resulting *child network* can drive a reinforcement learning loop. Figure 1 shows the full cycle: sample architecture A with probability p → train child network → obtain accuracy R → compute gradient of log p scaled by R → update controller.

### Part II: Related Work (§2, p.2) / 관련 연구

**한국어:** 세 갈래의 선행 연구를 위치시킨다.
- **Hyperparameter optimization:** Bergstra & Bengio (2012), Snoek et al. (2012, 2015) 등의 Bayesian/grid/random search는 고정 길이 실수 벡터에는 강하지만, 가변 길이 구조를 다룰 수 없다.
- **Neuroevolution:** Wierstra (2005), Stanley NEAT 계열은 가변 길이 구조 진화가 가능하지만 search-based여서 느리고 휴리스틱이 많다.
- **Program synthesis & 학습으로 학습하기 (learning-to-learn):** Andrychowicz (2016) "Learning to learn by gradient descent" 및 Li & Malik (2016)와 정신적으로 가까움. 또한 본 방법의 auto-regressive controller는 seq2seq의 디코더(Sutskever 2014)와, 미분 불가 BLEU 최적화(Ranzato 2015, Shen 2016)는 본 방법의 REINFORCE와 동치이다.

**English:** Three lines of prior work:
- **Hyperparameter optimization** (Bergstra, Snoek): handles fixed-length real-valued configurations, but cannot search topology.
- **Neuroevolution** (NEAT, HyperNEAT): can evolve variable-length topologies, but is slow and heuristic-heavy.
- **Program synthesis & meta-learning**: Andrychowicz (2016) "learning to learn by gradient descent" and Li & Malik (2016) on learning optimizers via RL. The auto-regressive controller mirrors the seq2seq decoder (Sutskever 2014); REINFORCE on non-differentiable BLEU (Ranzato 2015, Shen 2016) is the closest methodological precedent.

### Part III: Methods §3.1 — Generate Model Descriptions with a Controller RNN (p.3) / 컨트롤러 RNN으로 모델 설명 생성

**한국어:** controller는 LSTM이며 매 시점마다 token을 한 번에 하나씩 sample한다. CIFAR 실험의 토큰 종류는 한 layer당 5개:
1. filter height (∈ {1,3,5,7})
2. filter width (∈ {1,3,5,7})
3. stride height (∈ {1,2,3} 또는 고정 1)
4. stride width
5. number of filters (∈ {24,36,48,64})

각 token은 softmax로 sampled된 후 다음 시점의 입력 embedding으로 들어간다(auto-regressive). layer 수는 학습 진행에 따라 schedule되어 점점 늘어난다(예: 6 layers부터 시작해 매 1,600 sample마다 +2). controller는 layer 수가 정해진 한도에 도달하면 architecture 생성을 멈춘다.

**English:** The controller is an LSTM that samples one token at a time. For CIFAR experiments, each layer is described by 5 tokens: filter height ∈ {1,3,5,7}, filter width ∈ {1,3,5,7}, stride height ∈ {1,2,3} (or fixed 1), stride width, and number of filters ∈ {24,36,48,64}. Each sampled token is fed back as the next-step input embedding (auto-regressive). The total layer count follows a schedule that grows during training — e.g., start at 6 layers and add 2 every 1,600 samples. Generation halts when the layer count reaches the current cap. After the controller finishes, the child network is built and trained, and its validation accuracy is recorded.

### Part IV: §3.2 — Training with REINFORCE (p.3–4) / REINFORCE로 학습

**한국어:** controller가 만든 토큰 시퀀스 $a_{1:T}$를 action으로 보고, child network의 수렴 후 검증 정확도 $R$을 reward로 본다. 목적함수는

$$ J(\theta_c) = \mathbb{E}_{P(a_{1:T};\theta_c)}[R]. $$

$R$이 미분 불가이므로 REINFORCE(Williams 1992):

$$ \nabla_{\theta_c} J(\theta_c) = \sum_{t=1}^{T} \mathbb{E}\big[\nabla_{\theta_c} \log P(a_t \mid a_{(t-1):1};\theta_c)\, R\big]. $$

$m$개 architecture sample로 Monte-Carlo 추정:

$$ \hat g = \frac{1}{m}\sum_{k=1}^{m}\sum_{t=1}^{T} \nabla_{\theta_c}\log P(a_t \mid a_{(t-1):1};\theta_c)(R_k - b). $$

baseline $b$는 이전 보상들의 exponential moving average. $b$가 현재 action에 의존하지 않는 한 estimator는 unbiased이며 분산만 감소.

**한국어 (Parallelism):** 1 child training이 수 시간이 걸리므로 distributed: parameter server $S$ shards가 controller 가중치를 저장, $K$ controller replica가 각각 $m$개 child를 병렬 학습 → 그래디언트를 모아 server로 비동기 전송. CIFAR: $S=20, K=100, m=8$ → 800 GPU 동시 학습.

**English:** Treating the token sequence $a_{1:T}$ as actions and the child's converged validation accuracy as reward $R$, the goal is $J(\theta_c)=\mathbb{E}[R]$. Since $R$ is non-differentiable, REINFORCE (Williams 1992) supplies the gradient $\sum_t \mathbb{E}[\nabla \log P(a_t|\cdot) R]$, estimated by Monte-Carlo over $m$ sampled architectures. A baseline $b$ — exponential moving average of past rewards — is subtracted to reduce variance without introducing bias (because $b$ is independent of the current action). Distributed training uses $S$ parameter-server shards, $K$ controller replicas, and $m$ child replicas per controller; CIFAR-10 uses $S=20, K=100, m=8$, training 800 child networks concurrently.

### Part V: §3.3 — Skip Connections and Other Layer Types (p.4–5) / 스킵 연결과 다른 레이어 타입

**한국어:** 단순한 chain CNN은 ResNet/DenseNet/GoogLeNet 같은 분기 구조를 표현하지 못한다. 저자는 layer N에 N-1개의 sigmoid (anchor point)를 추가해 set-selection attention(Vinyals 2015 pointer)과 유사한 방식으로 어떤 이전 layer를 입력으로 쓸지 확률적으로 결정한다:

$$ P(\text{Layer } j \text{ is input to layer } i) = \mathrm{sigmoid}\big(v^\top \tanh(W_{prev} h_j + W_{curr} h_i)\big). $$

여러 layer가 입력으로 선택되면 depth 차원으로 concatenate. compilation failure(어떤 layer가 입력이 없거나, 출력이 어디에도 안 가거나, 입력 크기가 다른 경우)는 세 휴리스틱으로 해결: (1) 입력 없으면 image를 입력으로, (2) 마지막 layer에서 unconnected output을 모두 concat, (3) 크기가 다른 입력은 zero-padding.

또한 본 논문에서는 학습률·풀링·LN·BN을 controller가 직접 예측하지 않고 고정. 확장 시에는 controller가 layer type을 먼저 예측하고 type-specific token을 이어서 예측하는 식으로 일반화 가능.

**English:** Pure chain CNNs cannot express branching architectures like ResNet/DenseNet/GoogLeNet. The authors add $N-1$ anchor sigmoids at each layer $N$ that act as set-selection attention (Vinyals 2015), each predicting the probability that earlier layer $j$ is connected to current layer $i$:

$$ P(\text{Layer } j\to i) = \mathrm{sigmoid}\big(v^\top \tanh(W_{prev} h_j + W_{curr} h_i)\big). $$

Multiple selected inputs are concatenated along depth. Three simple heuristics resolve compilation failures: (1) layers without inputs default to the image, (2) unconnected outputs at the final layer are concatenated before the classifier, (3) mismatched-size inputs are zero-padded. Learning rate, pooling, batch-norm are *not* predicted by the controller in this paper but could be added as additional tokens.

### Part VI: §3.4 — Generate Recurrent Cell Architectures (p.5–6) / 재귀 셀 구조 생성

**한국어:** RNN cell의 search space는 tree 구조로 표현. base number $B$ leaf node + 내부 node로 구성된 tree (CIFAR식 chain과 다름). 각 node에서 controller는 (1) combination method ∈ {add, elem_mult}, (2) activation ∈ {identity, tanh, sigmoid, ReLU}를 예측. 마지막에는 cell state $c_t$/$c_{t-1}$를 어떤 node에 inject/연결할지 마지막 두 block으로 예측.

**Figure 5 base-2 example (논문 그대로 따라가보기):**
- tree index 0: controller가 (Add, Tanh) 예측 → $a_0 = \tanh(W_1 x_t + W_2 h_{t-1})$
- tree index 1: (ElemMult, ReLU) → $a_1 = \mathrm{ReLU}((W_3 x_t) \odot (W_4 h_{t-1}))$
- "Cell Inject" 두 번째 element 0, (Add, ReLU) → $a_0^{\text{new}} = \mathrm{ReLU}(a_0 + c_{t-1})$
- tree index 2: (ElemMult, Sigmoid) → $a_2 = \sigma(a_0^{\text{new}} \odot a_1)$ → max index가 2이므로 $h_t = a_2$.
- "Cell Index" 첫 번째 element 1 → $c_t = (W_3 x_t)\odot(W_4 h_{t-1})$ (즉 tree index 1의 활성화 직전 값).

실험에서는 base=8 → 약 $6\times10^{16}$ cell 후보. controller는 이 거대한 공간에서 perplexity가 좋은 cell을 찾는다.

**English:** The recurrent cell search space is tree-structured. Each tree has $B$ leaf nodes (the *base number*) and internal nodes that combine pairs. For each node the controller predicts (1) combination method ∈ {add, elem_mult} and (2) activation ∈ {identity, tanh, sigmoid, ReLU}. Two extra "Cell Inject"/"Cell Index" blocks at the end specify how memory state $c_{t-1}$ is fed in and how $c_t$ is read out.

For Figure 5's base-2 example, the prediction sequence yields exactly: $a_0 = \tanh(W_1 x_t + W_2 h_{t-1})$, $a_1 = \mathrm{ReLU}((W_3 x_t)\odot(W_4 h_{t-1}))$, $a_0^{\text{new}} = \mathrm{ReLU}(a_0 + c_{t-1})$, $a_2 = \sigma(a_0^{\text{new}} \odot a_1) = h_t$, and $c_t = (W_3 x_t)\odot(W_4 h_{t-1})$. With base=8, the search space has approximately $6\times10^{16}$ candidate cells.

### Part VII: §4.1 — CIFAR-10 Experiments (p.6–8) / CIFAR-10 실험

**한국어:**
- **Dataset/preprocessing:** 50,000 train (45,000 train + 5,000 validation) + 10,000 test, whitening, upsample → random 32×32 crop, horizontal flip.
- **Search space:** filter height ∈ {1,3,5,7}, filter width ∈ {1,3,5,7}, #filters ∈ {24,36,48,64}. ReLU, BN, skip connections 포함. stride 실험 두 가지: (i) stride=1 고정, (ii) stride ∈ {1,2,3} 예측.
- **Controller training:** 2-layer LSTM, 35 hidden units, ADAM lr=0.0006, weights init [-0.08, 0.08]. distributed: $S=20, K=100, m=8$ → 동시에 800 GPU.
- **Child training:** Momentum optimizer, lr=0.1, weight decay 1e-4, momentum 0.9, Nesterov. 50 epoch 학습. depth schedule: 6 layers부터 시작 → 매 1,600 sample마다 +2.
- **Reward:** 마지막 5 epoch의 max validation accuracy의 cube ($R = \text{acc}^3$). cube는 좋은 모델 사이의 차이를 증폭.
- **결과 (Table 1):**
  - NAS v1 (no stride/pool, 15 layers, 4.2M params): 5.50% test error.
  - NAS v2 (predicting strides, 20 layers, 2.5M params): 6.01%.
  - NAS v3 (max pooling at layer 13/24, 39 layers, 7.1M): **4.47%**.
  - NAS v3 + 40 more filters (39 layers, 37.4M): **3.65%** (DenseNet-BC 3.46%의 1.05x 빠름).
  - 12,800 architecture를 평가 후 best에 grid search(lr, weight decay, BN ε, decay epoch).
- **검증:** v1 architecture에서 모든 layer를 dense skip하면 5.56%로 약간 악화, skip 다 제거하면 7.97%로 크게 악화 → controller가 찾은 패턴이 local optimum.

**English:**
- **Data:** CIFAR-10, whitened, upsampled then 32×32 random crop, horizontal flip. 45k train, 5k validation, 10k test.
- **Search space:** filter height/width ∈ {1,3,5,7}, #filters ∈ {24,36,48,64}, ReLU, BN, optional skip connections. Two strides settings (fixed 1 vs predicted ∈ {1,2,3}).
- **Controller:** 2-layer LSTM, 35 hidden, ADAM lr=0.0006, weights uniformly in [-0.08, 0.08]. Parallelism $S=20, K=100, m=8$ — 800 GPUs concurrently.
- **Child training:** Nesterov SGD lr=0.1, weight decay 1e-4, momentum 0.9, 50 epochs. Layer count schedule: start 6, +2 every 1,600 samples.
- **Reward:** $R = (\text{max val acc over last 5 epochs})^3$ — cubing amplifies gaps among good models.
- **Headline numbers (Table 1):** v1 5.50% (no stride/pool, 15 layers), v2 6.01% (predicted strides, 20 layers), v3 4.47% (39 layers + max pool), v3+40 filters 3.65% (37.4M params, 1.05x faster than DenseNet 3.46%).
- **Robustness checks:** densely connecting *all* layers → 5.56% (slightly worse), removing *all* skips → 7.97% (much worse). The controller-found pattern is a local optimum.

### Part VIII: §4.2 — Penn Treebank RNN Cell (p.8–10) / Penn Treebank RNN 셀 실험

**한국어:**
- **Search space:** base=8 → $\approx 6\times10^{16}$ cell. combination ∈ {add, elem_mult}, activation ∈ {identity, tanh, sigmoid, ReLU}.
- **Controller:** lr=0.0005 (CIFAR보다 약간 작음). distributed $S=20, K=400, m=1$ → 400 CPU, 10 그래디언트 누적 후 업데이트.
- **Child training:** 2-layer RNN, hidden size를 medium baseline(Zaremba 2014, Gal 2015)에 맞춤. embedding dropout, recurrent dropout 사용. 35 epoch.
- **Reward:** $R = c/\text{(val perplexity)}^2$, $c=80$. perplexity는 작을수록 좋으므로 역수 제곱.
- **결과 (Table 2):**
  - NAS base 8 (32M params): 67.9 perplexity.
  - NAS base 8 + shared embeddings (25M): **64.0**.
  - NAS base 8 + shared embeddings (54M): **62.4** ← 새 SOTA. 이전 best Zilly RHN(24M) 66.0 대비 -3.6 perplexity, 그리고 cell당 step이 1번이므로 RHN의 10번 대비 2배 이상 빠름.
- **Transfer to PTB character LM:** 같은 cell로 dropout 0.2/0.5, ADAM lr=0.001, embedding 128, hidden 800, batch 32, BPTT 100 → **1.214 bpc** (이전 SOTA Ha et al. 1.219 능가).
- **Transfer to GNMT (English→German):** GNMT의 LSTM cell을 NAS cell로 단순 교체 → BLEU +0.5 (튜닝 없이).

**English:**
- **Space:** base=8 cells, ~$6\times10^{16}$ candidates.
- **Controller:** lr=0.0005, $S=20, K=400, m=1$ — 400 CPUs, asynchronous updates after 10 accumulated gradients.
- **Child:** 2-layer RNN with hidden width matched to Zaremba "medium" baseline; embedding & recurrent dropout (Gal 2015); 35 epochs.
- **Reward:** $R = c / \text{val\_pp}^2$ with $c=80$.
- **PTB results (Table 2):** 67.9 pp at 32M params, 64.0 pp at 25M with shared embeddings, **62.4 pp at 54M with shared embeddings — new SOTA**, 3.6 perplexity below Zilly's RHN (66.0) and 2x faster per step (1 cell call vs RHN's 10).
- **Transfer to character-level PTB:** with dropouts 0.2/0.5, ADAM lr=0.001, hidden=800, BPTT=100, batch=32 → **1.214 bpc — SOTA** (vs Ha 2-layer Norm HyperLSTM 1.219).
- **Transfer to GNMT En→De:** drop NASCell into GNMT with no retuning → +0.5 BLEU on WMT'14 test.

### Part IX: Control Experiments (p.10) / 제어 실험

**한국어:**
- **Control 1 — bigger search space:** combination에 max, activation에 sin 추가. 결과: comparable. 흥미롭게도 최종 best cell은 sin을 채택하지 않았다.
- **Control 2 — random search baseline:** policy gradient 대신 controller를 random initialization 그대로 사용 → architecture 균일 sample. Figure 6은 매 400 model마다 top-1, top-5, top-15 model의 평균 성능을 random search 대비 표시. RL이 random search보다 항상 우위이며 차이가 시간이 지날수록 벌어진다 → controller가 실제로 학습되고 있다는 결정적 증거.

**English:**
- **Control 1 — larger search space:** add `max` to combinations and `sin` to activations. Performance is comparable; notably the chosen best cell does *not* use `sin`.
- **Control 2 — random search:** Figure 6 plots the perplexity gap between top-{1,5,15} models found by RL vs random search every 400 models. RL strictly outperforms random search, with the gap widening over training — the controller is genuinely learning, not just sampling lucky models.

### Part X: §5 Conclusion + Appendix (p.10–16) / 결론 + 부록

**한국어:** 요약은 명료하다 — RNN을 controller로 쓰면 가변 길이 architecture 공간을 RL로 탐색할 수 있고, 인간 SOTA에 필적하는 모델을 발견할 수 있다. 코드와 NASCell은 TensorFlow에 공개. Appendix A의 Figure 7은 v1 CNN(15-layer)의 발견된 구조 — 직사각형 필터, 위쪽 layer일수록 큰 필터, ResNet-style one-step skip이 많고 dense 연결은 아님. Figure 8은 발견된 RNN cell — 첫 몇 step이 LSTM과 매우 유사 ($W_1 h_{t-1} + W_2 x_t$를 여러 번 반복해서 다른 component로 보내는 패턴) 하지만 더 깊고 풍부한 비선형성.

**English:** The conclusion is concise — using an RNN controller, RL can search variable-length architecture spaces and discover models on par with hand-designed SOTA. Code and `NASCell` were released in TensorFlow. Appendix A's Figure 7 visualizes the v1 15-layer CNN: rectangular filters, larger filters at top layers, many one-step skip connections (ResNet-like) but not fully dense. Figure 8 shows the discovered RNN cell — the bottom few steps look very LSTM-like (repeated $W_1 h_{t-1} + W_2 x_t$ branches feeding different components) but with deeper and richer nonlinearities than vanilla LSTM.

### Part XI: Discovered architecture details (Appendix A Figure 7) / 발견된 구조 세부

**한국어:** v1 CNN(stride/pool 없음, 15 layer, 4.2M params)의 layer별 hyperparameter는 다음 패턴이 관찰된다(논문 Figure 7 그래프 기반):
- **하단 (image 가까이):** FH/FW 작음 (3×3, 5×5), N=36~48 — local feature 추출.
- **중단:** filter가 점차 직사각형으로 변형 (FH≠FW, 예: 7×1, 7×3, 1×7) — 비대칭 receptive field가 흥미로운 발견.
- **상단 (softmax 가까이):** FH/FW 큼 (7×5, 7×7), N=48 — 고수준 표현.
- **Skip connections:** 서로 인접한 layer 사이의 one-step skip이 가장 많지만, 일부 long-range skip도 나타남 → ResNet과 DenseNet의 중간 형태.

이 패턴은 "사람이 직관적으로 설계하지 않을" 비대칭 필터를 controller가 자동으로 학습했음을 시사. 실제로 ablation에서 모든 layer를 dense skip하면 5.50% → 5.56%로 약간 악화되어, NAS가 찾은 패턴이 단순한 "더 많이 연결"보다 정밀한 local optimum임을 보여준다.

**English:** Layer-by-layer patterns in the v1 CNN (15 layers, 4.2M params, no stride/pool) from Figure 7:
- **Bottom (near image):** small FH/FW (3×3, 5×5), N=36–48 — local features.
- **Middle:** progressively rectangular filters (FH ≠ FW, e.g., 7×1, 7×3, 1×7) — asymmetric receptive fields, an unusual choice no human had committed to in 2016.
- **Top (near softmax):** large FH/FW (7×5, 7×7), N=48 — high-level features.
- **Skip pattern:** most are one-step skips like ResNet, with a few long-range skips — an intermediate between ResNet and DenseNet.

The fact that random densely-connecting all layers worsens accuracy (5.50% → 5.56%) confirms the controller-discovered pattern is a precise local optimum, not just "more connections is better."

### Part XII: Discovered RNN cell details (Appendix A Figure 8) / 발견된 RNN 셀 세부

**한국어:** Figure 8 (top right)의 base-8 cell을 LSTM과 비교하면 흥미로운 패턴이 드러난다:
- **LSTM과 유사:** 입력 부분에서 $W_1 h_{t-1} + W_2 x_t$ 형태의 add를 여러 번 반복하고 sigmoid/tanh로 게이팅 — 즉 "input gate, forget gate, candidate" 같은 LSTM의 구조적 모티프가 controller가 자체적으로 발견한 cell에서도 나타남.
- **LSTM과 차이:** (i) 더 많은 분기와 더 깊은 합성 함수, (ii) elem_mult가 LSTM보다 깊은 위치에 등장, (iii) cell state $c_t$의 inject 위치가 비전형적, (iv) ReLU activation이 cell 내부에서 사용 — RNN cell에서 흔치 않은 선택.
- **결과적 함의:** controller는 LSTM의 "성공적 구조 motif"는 보존하면서, 인간이 시도하지 않았던 깊이/비선형성을 추가했다.

**English:** Comparing the base-8 cell (Figure 8 top right) with LSTM:
- **LSTM-like:** the bottom of the cell repeats $W_1 h_{t-1} + W_2 x_t$ adds gated by sigmoid/tanh — the same input-gate / forget-gate / candidate motif emerged independently in the controller-found cell.
- **Different:** (i) more branches and deeper compositions; (ii) elem_mult appears deeper than LSTM places it; (iii) atypical $c_t$ inject location; (iv) ReLU is used inside the cell — rare for recurrent cells in 2016.
- **Implication:** the controller preserved LSTM's most successful structural motifs while adding depth and nonlinearity that humans had not explored.

### Part XIII: Compute cost analysis / 연산 비용 분석

**한국어:** 본 논문의 비용을 명시적으로 계산해보자.
- CIFAR-10: 12,800 child architecture × 50 epoch × ~0.5 GPU-hour/epoch ≈ 320,000 GPU-hours = ~13,300 GPU-days. 800 GPU 동시 실행이면 wall-clock으로 ~17일.
- 후속 연구(NASNet 2017)는 dataset을 CIFAR-10에 한정하고 cell-based search로 비용을 줄였으나 여전히 ~2,000 GPU-days.
- ENAS(2018)는 weight sharing으로 child를 처음부터 학습하지 않고 supernet의 subgraph만 활성화 → CIFAR-10 NAS를 0.45 GPU-day로 단축 (본 논문 대비 ~30,000배 감소).
- DARTS(2019)는 architecture를 continuous distribution으로 relaxation해 미분 가능 — bilevel optimization으로 1 GPU-day.
- 본 논문이 자원 면에서 "산업계만 가능한 연구"임을 명확히 보여줬고, 이것이 곧 학계 후속 연구의 핵심 모티프가 됐다.

**English:** Explicit cost arithmetic for this paper:
- CIFAR-10: 12,800 children × 50 epochs × ~0.5 GPU-hour/epoch ≈ ~320,000 GPU-hours ≈ ~13,300 GPU-days; with 800 concurrent GPUs, ~17 wall-clock days.
- NASNet (2017) reduced this somewhat by cell-based search but still cost ~2,000 GPU-days.
- ENAS (2018) made child networks share supernet weights, dropping CIFAR-10 NAS to 0.45 GPU-day — a ~30,000× reduction from this paper.
- DARTS (2019) made the search differentiable via continuous relaxation — ~1 GPU-day.
- This paper made clear that NAS-as-presented was an "industry-only" research program; closing that compute gap became the dominant follow-up theme.

---

## 3. Key Takeaways / 핵심 시사점

1. **Architecture design as RL on variable-length sequences / 구조 설계를 가변 길이 시퀀스 RL로 정식화** — 핵심 통찰은 "신경망 구조 = 가변 길이 문자열"이라는 점이다. 이로써 hyperparameter optimization(고정 길이) 한계를 돌파하고, seq2seq 디코더와 동일한 도구(softmax + auto-regressive feedback)로 architecture를 생성할 수 있게 됐다. The pivotal insight is that *architectures are variable-length strings*, which transcends fixed-length hyperparameter optimization and enables seq2seq-style auto-regressive generation.

2. **REINFORCE handles non-differentiable accuracy / REINFORCE로 미분 불가 보상을 처리** — 검증 정확도/perplexity는 backprop을 통과하지 않기 때문에 표준 학습이 불가능하다. Williams 1992의 log-derivative trick으로 그래디언트를 estimator로 변환하고, baseline EMA로 분산을 줄였다. NLP의 BLEU 최적화(Ranzato 2015)와 같은 기법의 NAS 버전. Validation accuracy is non-differentiable; the log-derivative trick (REINFORCE) plus an EMA baseline converts it into a usable training signal — the same idea Ranzato used for BLEU in NMT.

3. **Skip connections via set-selection attention / set-selection attention으로 스킵 연결** — anchor point에서 sigmoid-attention으로 N-1개의 이진 결정을 내리는 트릭은 ResNet/DenseNet 모두를 표현할 수 있게 했고, 이후 NAS 후속 연구의 표준이 됐다. The anchor-point attention trick uniformly expresses ResNet/DenseNet/GoogLeNet topologies and became standard in later NAS work.

4. **Massive parallelism is essential / 막대한 병렬성 필요** — 800 GPU × 4주(약 22,400 GPU-days)는 당시에도 극단적이었다. parameter server + asynchronous update이 학습 속도를 결정. 이는 곧 후속 연구(ENAS의 weight sharing, DARTS의 differentiable relaxation)가 효율 개선에 집중한 직접적 이유다. The 800-GPU × ~28-day cost (≈22,400 GPU-days) was extreme even in 2016 and directly motivated subsequent efficiency-focused work like ENAS and DARTS.

5. **Tree-structured search space for RNN cells / RNN 셀의 트리 기반 탐색 공간** — base=8 트리는 약 $6\times10^{16}$ 가지의 cell을 담는다. 이 표현이 LSTM도 표현 가능하면서 새로운 cell도 자연스레 발견하게 만든 핵심. The base-8 tree representation contains $6\times10^{16}$ cells, is rich enough to contain LSTM as a special case, and is the structural reason novel cells emerge.

6. **Reward shaping matters / 보상 설계가 중요** — CIFAR에서는 $\text{acc}^3$, PTB에서는 $80/\text{pp}^2$. 단순한 정확도/perplexity 그대로보다 우수 모델 사이의 차이를 증폭하는 변환이 controller 학습에 크게 도움. Cubing accuracy and inverse-squared perplexity are simple but consequential — they amplify reward gaps among already-good architectures.

7. **Random search is a strong but beatable baseline / 랜덤 서치는 강력하나 RL이 우위** — Figure 6의 control experiment는 RL이 random search 대비 일관되게 우위임을 보였다. 이는 이후 NAS 평가 표준이 되었고, 5년 뒤 Li & Talwalkar (2019), Yu et al. (2020)이 일부 NAS 방법은 random search에 미치지 못함을 보여 평가 엄밀성 논쟁의 발단이 됐다. Figure 6's RL-vs-random-search comparison set the evaluation standard; later "is random search competitive?" debates (Li & Talwalkar 2019) trace back to this exact graph.

8. **Transfer & generalization of discovered cells / 발견된 셀의 전이성** — PTB에서 word-level로 찾은 cell이 character-level과 GNMT En→De에 그대로 transfer되며 모두 SOTA 또는 +0.5 BLEU. 이는 NAS가 단순 overfit이 아니라 진짜 architectural innovation을 만들어냄을 입증. The PTB-discovered cell transfers to character-level (1.214 bpc SOTA) and to GNMT (+0.5 BLEU), evidence that NAS finds genuine architectural innovations, not dataset-specific overfits.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Controller objective / 컨트롤러 목적함수

$$
J(\theta_c) = \mathbb{E}_{P(a_{1:T};\theta_c)}[R]
$$

- $\theta_c$: controller(LSTM) 파라미터.
- $a_{1:T}$: controller가 생성한 architecture 토큰 시퀀스(action sequence).
- $T$: token 개수 = (CIFAR) layer 수 × layer당 token 수.
- $R$: child network의 검증 정확도(또는 그 변형 — CIFAR에선 $\text{acc}^3$, PTB에선 $80/\text{pp}^2$).
- 의미: controller policy가 만드는 architecture distribution 하에서 기대 보상 최대화.
- **Meaning:** maximize the expected reward over architectures sampled from the controller's policy.

### 4.2 REINFORCE policy gradient / REINFORCE 정책 경사

$$
\nabla_{\theta_c} J(\theta_c) = \sum_{t=1}^{T} \mathbb{E}_{P(a_{1:T};\theta_c)} \big[ \nabla_{\theta_c} \log P(a_t \mid a_{(t-1):1};\theta_c)\, R \big]
$$

- $P(a_t \mid a_{(t-1):1};\theta_c)$: 시점 $t$에서 LSTM softmax가 출력하는 토큰 $a_t$의 확률(이전 토큰 conditioning).
- log-derivative trick: $\nabla_\theta \mathbb{E}_\pi[f] = \mathbb{E}_\pi[\nabla_\theta \log \pi \cdot f]$.
- 기대값 안에 그래디언트가 있으므로 sampling으로 추정 가능 → 미분 불가 $R$도 학습 신호로 사용.
- **Meaning:** the log-derivative trick converts the gradient of an expectation into an expectation of a gradient, enabling Monte-Carlo estimation even when $R$ is non-differentiable.

### 4.3 Empirical Monte-Carlo estimator with baseline / 베이스라인 적용 경험적 추정량

$$
\hat g(\theta_c) = \frac{1}{m}\sum_{k=1}^{m}\sum_{t=1}^{T} \nabla_{\theta_c}\log P\!\big(a_t^{(k)} \mid a_{(t-1):1}^{(k)};\theta_c\big)\,(R_k - b)
$$

- $m$: 한 update에 sample하는 architecture 수 (CIFAR $m=8$, PTB $m=1$).
- $R_k$: $k$번째 child의 보상.
- $b$: baseline. 본 논문에서는 이전 보상들의 exponential moving average. action에 의존하지 않으므로 unbiased.
- **Variance reduction:** $\mathrm{Var}[\nabla \log \pi \cdot (R - b)] < \mathrm{Var}[\nabla \log \pi \cdot R]$ when $b$ is well-chosen (correlated with $R$).
- **Unbiasedness sketch:** $\mathbb{E}[\nabla\log\pi \cdot b] = b\cdot\mathbb{E}[\nabla\log\pi] = b\cdot 0 = 0$, so subtracting $b$ does not bias the gradient.

### 4.4 Skip connection probability / 스킵 연결 확률

$$
P(\text{Layer } j \to i) = \mathrm{sigmoid}\big(v^\top \tanh(W_{prev} h_j + W_{curr} h_i)\big)
$$

- $h_j$: layer $j$의 anchor에서 controller LSTM hidden state.
- $h_i$: layer $i$의 현재 hidden state.
- $W_{prev}, W_{curr}, v$: 학습 가능 attention 파라미터.
- 각 가능 입력 layer마다 독립적으로 Bernoulli sample → set-selection.
- **Meaning:** content-based attention compares the controller's hidden states at layer $i$ and an earlier layer $j$ to decide independently whether $j$ feeds $i$.

### 4.5 Reward shaping / 보상 변환

CIFAR-10:
$$ R_{\text{CIFAR}} = \big(\max_{e \in \text{last 5}} \text{val\_acc}_e\big)^3 $$

PTB:
$$ R_{\text{PTB}} = \frac{c}{\text{val\_pp}^2}, \quad c = 80 $$

- **Why cube:** 두 model이 0.92 vs 0.93 차이일 때 raw 차이는 0.01이지만 cube는 $0.93^3 - 0.92^3 \approx 0.026$로 약 2.6배 증폭 → controller가 좋은 model 사이의 "더 좋은" 방향을 학습하기 쉬워진다.
- **Why inverse-squared perplexity:** perplexity는 작을수록 좋으므로 역수, 그리고 squared로 우수 cell의 차이를 증폭하면서 양수 reward 보장.

### 4.6 Worked numerical example: REINFORCE update on toy controller / 장난감 컨트롤러의 REINFORCE 업데이트 수치 예제

**한국어:** 2-layer architecture에 token이 layer당 1개(단순화)라 가정. 두 token a_1, a_2 ∈ {0,1}이고 controller가 각각 0/1을 동일 확률 0.5로 출력한다(초기). $m=4$ sample을 뽑았다고 하자 — 보상은 임의:

| k | $a_1, a_2$ | $R_k$ |
|---|---|---|
| 1 | 0, 0 | 0.60 |
| 2 | 0, 1 | 0.85 |
| 3 | 1, 0 | 0.70 |
| 4 | 1, 1 | 0.92 |

baseline $b = (0.60 + 0.85 + 0.70 + 0.92)/4 = 0.7675$.

$\nabla_\theta \log P(a_t | \cdot)$를 단순 logit 모델로 두면 (token=1일 때 $\partial = 0.5$, token=0일 때 $\partial = -0.5$ 가정):

- k=1: $(R_1 - b) = -0.1675$, gradient = $-0.5 + -0.5 = -1$ → contribution $0.16$.
- k=2: $(R_2 - b) = 0.0825$, gradient = $-0.5 + 0.5 = 0$ → contribution $0$.
- k=3: $(R_3 - b) = -0.0675$, gradient = $0.5 + -0.5 = 0$ → contribution $0$.
- k=4: $(R_4 - b) = 0.1525$, gradient = $0.5 + 0.5 = 1$ → contribution $0.15$.

평균 그래디언트는 $\approx (0.16 + 0 + 0 + 0.15)/4 = 0.078$ → token=1을 선호하는 방향으로 logit 증가. 4개 sample만으로도 best architecture (1,1)이 가장 큰 양의 기여를 함을 알 수 있다.

**English:** Assume a toy 2-token controller with each token ∈ {0,1} and uniform initial policy. After 4 sampled architectures with rewards as above and baseline $b = 0.7675$:
- the (1,1) architecture has positive advantage $+0.1525$ and pushes both tokens toward 1;
- the (0,0) architecture has negative advantage $-0.1675$ and pushes them away from 0.

Average gradient $\approx 0.078$ in the "logit for 1" direction — exactly the right signal even from just 4 samples. The real paper uses $m=8$ and runs 12,800 such updates.

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1992 ── REINFORCE (Williams) ─── policy gradient theorem
2002 ── NEAT (Stanley) ──────── neuroevolution baseline
2011 ── Bergstra Bayes opt. ── fixed-length HPO
2012 ── AlexNet ──────────────── architecture engineering era begins
2014 ── Sutskever seq2seq ───── auto-regressive token generation
2015 ── ResNet, BatchNorm ──── building blocks NAS will use
2015 ── Ranzato MIXER ───────── REINFORCE on BLEU (precedent)
2016 ── DenseNet ─────────────── strongest hand-crafted CIFAR baseline
2016 ── Zoph & Le NAS  ◄ THIS PAPER ► launches AutoML/NAS field
2017 ── NASNet ──────────────── cell-based NAS, transfers to ImageNet
2018 ── ENAS (Pham) ──────────── weight sharing → 0.45 GPU-days
2019 ── DARTS (Liu) ──────────── differentiable NAS
2019 ── Random search debate ── Li & Talwalkar
2019+── EfficientNet, AutoAugment, RLHF — same controller-reward loop reapplied
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Williams (1992), *Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning* | 본 논문이 사용하는 REINFORCE의 원전. log-derivative trick 그대로 차용. / The exact REINFORCE rule used here. | **Foundational** — 식 (2) 그대로 인용. |
| Hochreiter & Schmidhuber (1997), *Long Short-Term Memory* | controller는 LSTM이고 발견된 RNN cell은 LSTM의 일반화. / Controller is LSTM; discovered cell generalizes LSTM. | **Direct** — paper 26 (LSTM)와 §3.4 cell 비교가 핵심. |
| Sutskever, Vinyals, Le (2014), *Sequence to Sequence Learning* | controller의 auto-regressive token 생성 = seq2seq 디코더. / Auto-regressive controller mirrors seq2seq decoder. | **Methodological** — Quoc Le 자신의 이전 논문. |
| Bergstra & Bengio (2012), *Random Search for Hyperparameter Optimization* | random search가 본 논문의 baseline (Fig 6). 고정 길이 HPO의 한계가 NAS의 motivation. / Random search is the explicit baseline in Fig 6. | **Comparative** — control experiment 2의 직접적 대상. |
| Ranzato et al. (2015), *Sequence Level Training with Recurrent Neural Networks (MIXER)* | 미분 불가 BLEU를 REINFORCE로 최적화 — 본 논문의 NAS 버전이 정확히 같은 트릭 사용. / Same REINFORCE-on-non-differentiable-metric trick. | **Methodological** — sequence training의 NAS 일반화. |
| He et al. (2016), *Deep Residual Learning (ResNet)* | skip connection의 영감. NAS의 anchor-point가 ResNet/DenseNet topology를 표현 가능하게 함. / Inspires the skip-connection mechanism. | **Architectural** — NAS의 search space가 ResNet을 포함. |
| Andrychowicz et al. (2016), *Learning to learn by gradient descent by gradient descent* | meta-learning (network가 다른 network의 update를 학습)의 정신적 동족. / Spirit-cousin: a network learning another network's behavior. | **Conceptual** — "learning to learn"의 NAS 버전. |
| Pham et al. (2018), *Efficient Neural Architecture Search via Parameter Sharing (ENAS)* | 본 논문의 22,400 GPU-day를 0.45 GPU-day로 줄인 직접 후속작. weight sharing 도입. / Direct follow-up: weight sharing reduces 22,400 → 0.45 GPU-days. | **Successor** — 효율성 한계를 직격. |

---

## 7. References / 참고문헌

- Zoph, B., & Le, Q. V., "Neural Architecture Search with Reinforcement Learning", ICLR 2017. arXiv:1611.01578. https://arxiv.org/abs/1611.01578
- Williams, R. J., "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning", *Machine Learning*, 1992.
- Hochreiter, S., & Schmidhuber, J., "Long Short-Term Memory", *Neural Computation*, 1997.
- Sutskever, I., Vinyals, O., & Le, Q. V., "Sequence to Sequence Learning with Neural Networks", NIPS 2014.
- Bergstra, J., & Bengio, Y., "Random Search for Hyperparameter Optimization", *JMLR*, 2012.
- Ranzato, M., Chopra, S., Auli, M., & Zaremba, W., "Sequence Level Training with Recurrent Neural Networks", arXiv:1511.06732, 2015.
- He, K., Zhang, X., Ren, S., & Sun, J., "Deep Residual Learning for Image Recognition", CVPR 2016.
- Huang, G., Liu, Z., & Weinberger, K. Q., "Densely Connected Convolutional Networks", arXiv:1608.06993, 2016.
- Zaremba, W., Sutskever, I., & Vinyals, O., "Recurrent Neural Network Regularization", arXiv:1409.2329, 2014.
- Gal, Y., "A Theoretically Grounded Application of Dropout in Recurrent Neural Networks", arXiv:1512.05287, 2015.
- Andrychowicz, M., et al., "Learning to learn by gradient descent by gradient descent", arXiv:1606.04474, 2016.
- Vinyals, O., Fortunato, M., & Jaitly, N., "Pointer Networks", NIPS 2015.
- Dean, J., et al., "Large Scale Distributed Deep Networks", NIPS 2012.
- Pham, H., Guan, M., Zoph, B., Le, Q. V., & Dean, J., "Efficient Neural Architecture Search via Parameter Sharing (ENAS)", ICML 2018. arXiv:1802.03268.
- Liu, H., Simonyan, K., & Yang, Y., "DARTS: Differentiable Architecture Search", ICLR 2019. arXiv:1806.09055.
- Li, L., & Talwalkar, A., "Random Search and Reproducibility for Neural Architecture Search", UAI 2019.
- Yu, K., Sciuto, C., Jaggi, M., Musat, C., & Salzmann, M., "Evaluating the Search Phase of Neural Architecture Search", ICLR 2020.
- Krizhevsky, A., Sutskever, I., & Hinton, G. E., "ImageNet Classification with Deep Convolutional Neural Networks (AlexNet)", NIPS 2012.
- Zilly, J. G., Srivastava, R. K., Koutník, J., & Schmidhuber, J., "Recurrent Highway Networks", arXiv:1607.03474, 2016.
- Wu, Y., Schuster, M., Chen, Z., Le, Q. V., Norouzi, M., et al., "Google's Neural Machine Translation System (GNMT)", arXiv:1609.08144, 2016.

---

## Appendix: Glossary of NAS terms used since 2017 / NAS 분야 용어집 (참고)

| 용어 / Term | 본 논문 / This paper | 후속 연구 / Later usage |
|---|---|---|
| Macro search | layer마다 토큰 자유 선택 (CIFAR §3.1) / per-layer free token choice | 본 논문 + NAS-Bench-101 |
| Cell / micro search | RNN cell 트리 (§3.4) / RNN-cell tree | NASNet, ENAS, DARTS 표준 |
| One-shot NAS | 본 논문 ✗ / not used here | ENAS, SMASH, DARTS |
| Differentiable NAS | 본 논문 ✗ — 어디까지나 RL / RL only | DARTS, GDAS |
| Predictor-based NAS | 본 논문 ✗ / not used | NAS-Bench, BANANAS |
