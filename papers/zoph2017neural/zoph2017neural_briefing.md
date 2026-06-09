---
title: "Pre-Reading Briefing: Neural Architecture Search with Reinforcement Learning"
paper_id: "27_zoph_2017"
topic: Artificial_Intelligence
date: 2026-04-28
type: briefing
---

# Neural Architecture Search with Reinforcement Learning: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Zoph, B. & Le, Q. V. (2017). Neural Architecture Search with Reinforcement Learning. ICLR 2017. arXiv:1611.01578.
**Author(s)**: Barret Zoph, Quoc V. Le (Google Brain)
**Year**: 2017

---

## 1. 핵심 기여 / Core Contribution

**한국어:** 이 논문은 신경망의 구조(architecture) 자체를 자동으로 설계하는 문제를 강화학습 문제로 정식화한 최초의 본격적인 시도이다. 저자들은 RNN 기반의 "controller"가 네트워크 구조를 가변 길이 문자열로 토큰 단위로 샘플링하게 하고, 이 구조로 만든 child network를 실제로 학습시킨 뒤 검증 정확도를 보상(reward)으로 사용해 REINFORCE 정책 경사로 controller를 업데이트한다. 결과적으로 사람이 설계한 SOTA 모델에 필적하거나 능가하는 CNN(CIFAR-10 test error 3.65%)과 RNN cell(Penn Treebank perplexity 62.4)을 처음부터 자동 발견했다.

**English:** This paper is the first large-scale demonstration that neural network architecture design itself can be cast as a reinforcement learning problem. The authors use an RNN "controller" to autoregressively sample variable-length architecture descriptions as sequences of tokens; each sampled architecture (child network) is fully trained on real data, and its validation accuracy is used as a non-differentiable reward to update the controller via the REINFORCE policy gradient. With ~800 GPUs running ~12,800 child networks, the method discovers from scratch a CNN that achieves 3.65% test error on CIFAR-10 (matching DenseNet) and a novel recurrent cell that achieves 62.4 perplexity on Penn Treebank — both rivaling or surpassing hand-designed state-of-the-art.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

**한국어:** 2012년 AlexNet 이후 5년 동안 CNN 설계의 트렌드는 사람의 직관과 경험에 의한 수공예적 발전이었다 — VGG(2014), GoogLeNet(2014), ResNet(2015), DenseNet(2016)으로 이어지는 흐름. 각 구조는 박사급 연구자들이 수개월에 걸쳐 ablation을 통해 도출했다. 한편 hyperparameter optimization 연구(Bergstra, Snoek 등)는 학습률·배치 크기 같은 고정 길이 벡터에는 잘 동작했지만 "구조 자체"를 탐색하지는 못했다. Neuroevolution(Stanley NEAT 등)은 가변 길이 구조 탐색이 가능했으나 휴리스틱이 많고 대규모 확장이 어려웠다. 본 논문은 sequence-to-sequence(Sutskever 2014)의 auto-regressive generation 아이디어와 BLEU 같은 미분 불가 지표를 RL로 최적화한 경험(Ranzato 2015, Shen 2016)을 결합해 "구조 생성도 sequence generation으로 다룰 수 있다"는 통찰을 실현했다.

**English:** Between 2012 (AlexNet) and 2016 the dominant pattern for CNN design was hand-crafted innovation by expert researchers — VGG (2014), GoogLeNet (2014), ResNet (2015), DenseNet (2016) — each requiring months of ablation. Hyperparameter optimization (Bergstra, Snoek) handled fixed-length real-valued vectors (learning rate, batch size) but could not search the topology itself. Neuroevolution (Stanley's NEAT, HyperNEAT) handled variable-length topologies but was heuristic-heavy and hard to scale. This paper bridges the gap by reusing two existing ideas: (1) auto-regressive sequence generation from seq2seq (Sutskever 2014) to sample architectures token-by-token, and (2) policy gradients on non-differentiable rewards (BLEU optimization in Ranzato 2015) to optimize validation accuracy.

### 타임라인 / Timeline

```
1992 ── REINFORCE policy gradient (Williams)
2002 ── NEAT neuroevolution (Stanley & Miikkulainen)
2011 ── Bergstra et al.: Bayesian hyperparameter optimization
2012 ── AlexNet wins ImageNet (Krizhevsky)
2014 ── Sequence-to-sequence learning (Sutskever, Vinyals, Le)
2015 ── ResNet (He et al.) — skip connections normalize deep nets
2015 ── Ranzato et al.: REINFORCE for non-differentiable BLEU
2016 ── Andrychowicz et al.: "Learning to learn by gradient descent"
2016 ── DenseNet (Huang et al.) — dense skip connections
2016 ── Zoph & Le: Neural Architecture Search ← THIS PAPER
2017 ── NASNet (Zoph et al.): cell-based search transferred to ImageNet
2018 ── ENAS (Pham et al.): weight sharing reduces 22,400 → 0.45 GPU-days
2019 ── DARTS (Liu et al.): differentiable architecture search
```

---

## 3. 필요한 배경 지식 / Prerequisites

**한국어:**
- **RNN/LSTM 기본:** Hochreiter & Schmidhuber 1997 LSTM cell. controller가 2-layer LSTM이며, 발견된 cell 자체도 LSTM의 일반화이다.
- **Policy gradient / REINFORCE:** $\nabla_\theta J = \mathbb{E}_\pi[\nabla_\theta \log \pi(a|s) R]$의 의미와 분산 감소를 위한 baseline 트릭.
- **CNN 기본 구성요소:** filter height/width, stride, number of filters, batch normalization, skip connection.
- **분산 학습:** parameter server 아키텍처(Dean et al. 2012)와 비동기 SGD.
- **Penn Treebank language modeling:** perplexity 정의와 LSTM-based 언어모델 학습.

**English:**
- **RNN/LSTM basics:** Hochreiter & Schmidhuber 1997. The controller is a 2-layer LSTM and the discovered RNN cell is a generalization of LSTM.
- **Policy gradient / REINFORCE:** the gradient $\nabla_\theta J = \mathbb{E}_\pi[\nabla_\theta \log \pi(a|s) R]$ and the baseline trick for variance reduction.
- **CNN building blocks:** filter height/width, stride, number of filters, batch normalization, skip connections (ResNet/DenseNet style).
- **Distributed training:** parameter server architecture (Dean et al. 2012) and asynchronous SGD.
- **Penn Treebank language modeling:** definition of perplexity and LSTM-based language model training.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **Controller (RNN)** | 자식 네트워크의 구조를 토큰 단위로 샘플링하는 RNN. 본 논문에서는 2-layer LSTM (35 hidden units). / The RNN that samples child architectures token by token. Here a 2-layer LSTM with 35 hidden units. |
| **Child network** | controller가 샘플링한 구조로 실제 빌드되어 학습되는 신경망. CIFAR-10 실험에서 50 epoch 학습. / The neural network instantiated from the controller's sample and actually trained on real data. In CIFAR-10, trained for 50 epochs. |
| **REINFORCE** | Williams (1992)의 policy gradient 알고리즘. 미분 불가 보상으로부터 policy의 그래디언트를 추정. / Williams's (1992) Monte-Carlo policy gradient that estimates $\nabla_\theta J$ from non-differentiable rewards. |
| **Reward $R$** | 자식 네트워크의 holdout 검증 정확도 (CIFAR-10) 또는 $c/\text{perplexity}^2$ (PTB). 미분 불가하므로 RL이 필요. / Validation accuracy on the child network (CIFAR-10) or $c/\text{perplexity}^2$ (PTB). Non-differentiable, hence RL. |
| **Baseline $b$** | variance 감소를 위해 reward에서 빼주는 값. exponential moving average of previous accuracies 사용. / Subtracted from reward to reduce gradient variance. Implemented as exponential moving average of past rewards. |
| **Skip connection / Anchor point** | layer N에서 N-1개의 sigmoid를 사용해 이전 layer 중 어느 것을 입력으로 쓸지 set-selection attention으로 결정. / At layer $N$, $N-1$ content-based sigmoids decide which earlier layers to feed in via set-selection attention. |
| **Search space** | controller가 탐색하는 token vocabulary. CIFAR: filter height ∈ {1,3,5,7}, width ∈ {1,3,5,7}, #filters ∈ {24,36,48,64}. / Vocabulary the controller chooses from. CIFAR: heights {1,3,5,7}, widths {1,3,5,7}, filter counts {24,36,48,64}. |
| **Base number (RNN cell)** | RNN cell tree의 leaf node 수. 본 논문에서 base=8 → 약 $6 \times 10^{16}$ 가지 cell 구조. / Number of leaf nodes in the recurrent cell's computation tree. base=8 → about $6 \times 10^{16}$ candidate cells. |
| **Parameter server** | controller 파라미터 $\theta_c$를 저장하고 K=20~100개의 controller replica에 비동기 분배. / Stores controller parameters $\theta_c$ and distributes them asynchronously to K=20–100 controller replicas. |
| **NAS** | Neural Architecture Search의 약어. 본 논문 이후 자동 구조 탐색 분야 전체를 가리키는 용어가 됨. / Acronym for Neural Architecture Search; after this paper it became the name of the entire field. |
| **NASCell** | 발견된 RNN cell이 TensorFlow에 추가된 공식 이름. / The official name of the discovered RNN cell, integrated into TensorFlow. |
| **Validation accuracy cubed** | 보상으로 검증 정확도를 그대로가 아닌 $\text{acc}^3$로 사용해 우수 모델 차이를 강조. / Reward in CIFAR experiments is the cube of validation accuracy to amplify gaps between top architectures. |

---

## 5. 수식 미리보기 / Equations Preview

**(1) 목적함수 / Objective**
$$ J(\theta_c) = \mathbb{E}_{P(a_{1:T};\theta_c)}[R] $$
controller 파라미터 $\theta_c$에 대해 기대 보상을 최대화. action 시퀀스 $a_{1:T}$는 controller가 만든 architecture description.
Maximize the expected reward $R$ over architectures $a_{1:T}$ sampled from the controller policy $P(\cdot;\theta_c)$.

**(2) REINFORCE policy gradient**
$$ \nabla_{\theta_c} J(\theta_c) = \sum_{t=1}^{T} \mathbb{E}_{P(a_{1:T};\theta_c)} \big[ \nabla_{\theta_c} \log P(a_t \mid a_{(t-1):1}; \theta_c)\, R \big] $$
보상이 미분 불가이므로 log-derivative trick으로 그래디언트를 표현. Williams (1992).
The log-derivative trick converts a non-differentiable reward into a usable gradient.

**(3) Empirical estimator with baseline**
$$ \frac{1}{m} \sum_{k=1}^{m} \sum_{t=1}^{T} \nabla_{\theta_c} \log P(a_t \mid a_{(t-1):1}; \theta_c)\, (R_k - b) $$
$m$개 architecture 샘플로 Monte-Carlo 근사. baseline $b$ (이전 보상의 EMA)는 분산을 줄여주지만 편향은 도입하지 않는다.
Monte-Carlo estimator over $m$ sampled architectures; baseline $b$ (EMA of past rewards) reduces variance without introducing bias.

**(4) Skip connection (anchor) probability**
$$ P(\text{Layer } j \text{ is input to layer } i) = \mathrm{sigmoid}(v^\top \tanh(W_{prev} h_j + W_{curr} h_i)) $$
content-based attention으로 이전 layer 연결 여부를 확률적으로 결정.
Content-based attention to decide whether earlier layer $j$ feeds into layer $i$.

**(5) PTB reward**
$$ R = \frac{c}{\text{(validation perplexity)}^2}, \qquad c = 80 $$
perplexity는 작을수록 좋으므로 역수의 제곱을 사용 — 좋은 모델의 차이를 증폭.
Inverse-squared perplexity ensures lower perplexity yields higher reward and amplifies differences among top cells.

---

## 6. 읽기 가이드 / Reading Guide

**한국어:**
- **§1–2 (Intro/Related):** "변수 길이 구조를 sample할 수 있어야 한다"는 motivation에 집중. hyperparameter optimization과의 차별점 명확히.
- **§3.1–3.2 (Methods):** 가장 중요. controller가 토큰을 어떻게 만들어내는지(softmax → input feedback) 시각화하면서 읽고, REINFORCE 식 (1)-(3)을 손으로 한 번 derive.
- **§3.3 (Skip connections):** anchor point + sigmoid attention이 핵심. 어떻게 "compilation failure"를 단순한 휴리스틱으로 처리하는지 메모.
- **§3.4 (RNN cell):** tree-structured cell — base=8이면 너무 추상적이니 Figure 5의 base=2 예제로 직접 계산해보기.
- **§4 (Experiments):** Table 1 (CIFAR), Table 2 (PTB) 숫자 외울 필요는 없지만 NAS v3 39-layer 4.47%, +filters 3.65%, base-8 PTB 62.4 정도는 기억.
- **§4 끝 control experiments:** random search baseline과의 비교 그래프(Fig 6)가 RL이 정말 도움됐다는 결정적 증거.
- **Appendix:** 발견된 CNN(Fig 7)과 RNN cell(Fig 8) 구조도 시각화 — LSTM과의 유사성/차이를 본다.

**English:**
- **§1–2 (Intro/Related):** focus on the "variable-length string" motivation; distinguish from hyperparameter optimization.
- **§3.1–3.2 (Methods):** the heart of the paper. Trace how the controller emits tokens (softmax → fed back as input). Derive equations (1)–(3) by hand.
- **§3.3 (Skip connections):** anchor points + sigmoid attention. Note how compilation failures are handled by simple heuristics.
- **§3.4 (RNN cell):** tree-based cell space; use Figure 5's base-2 example to walk through manually since base=8 is too abstract.
- **§4 (Experiments):** Table 1 (CIFAR), Table 2 (PTB). Memorize a few key numbers — NAS v3 39-layer 4.47%, +40 filters 3.65%, base-8 PTB 62.4.
- **End of §4 control experiments:** Figure 6 random-search baseline is the decisive evidence that RL helps over random sampling.
- **Appendix:** the discovered CNN (Fig 7) and RNN cell (Fig 8). Compare the cell to LSTM.

---

## 7. 현대적 의의 / Modern Significance

**한국어:** 이 논문은 AutoML과 NAS라는 분야 전체를 사실상 출범시켰다. 그러나 800 GPU × 28일 ≈ 22,400 GPU-days라는 비용 때문에 1년 안에 후속 연구들이 비용 절감에 집중했다: NASNet(2017)은 동일 cell을 반복 사용해 ImageNet으로 transfer, ENAS(2018)는 weight sharing으로 1000배 가속, DARTS(2019)는 search space를 연속 완화시켜 미분 가능 구조 탐색을 가능케 했다. 또한 본 논문이 도입한 "controller가 시퀀스로 구조를 만들고 보상으로 학습한다"는 패러다임은 RLHF(reward로 LM tuning), AutoAugment(데이터 증강 정책 탐색), Quoc V. Le 본인의 EfficientNet 계열로 이어졌다. 발견된 NASCell은 TensorFlow에 정식 RNN cell로 통합되었고, GNMT에 dropping해 BLEU +0.5 개선까지 보였다. 한편 이 논문이 대중화시킨 "자동으로 SOTA를 능가한다"는 마케팅은 이후 'NAS는 random search와 비슷한 성능'이라는 비판(Li & Talwalkar 2019, Yu et al. 2020)으로 이어져 평가 방법론 논쟁의 출발점이 되기도 했다.

**English:** This paper effectively launched the AutoML/NAS field. Its 800-GPU × 28-day (~22,400 GPU-days) compute cost triggered a wave of follow-ups focused on efficiency: NASNet (2017) made cells transferable from CIFAR to ImageNet, ENAS (2018) reduced cost ~1000x via weight sharing, and DARTS (2019) made the search differentiable by continuous relaxation. The "controller emits structures, reward signal trains it" paradigm later inspired RLHF (reward-based LM tuning), AutoAugment (learned augmentation policies), and Quoc Le's own EfficientNet line. The discovered NASCell was integrated into TensorFlow as a first-class RNN cell and gave +0.5 BLEU when dropped into GNMT. The paper's "auto-discovered SOTA" framing also spurred a methodological debate culminating in Li & Talwalkar (2019) and Yu et al. (2020), who argued that random search can be competitive — which itself became a healthy turning point for the field's evaluation rigor.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
