---
title: "Pre-Reading Briefing: Mastering the game of Go with deep neural networks and tree search (AlphaGo)"
paper_id: "23_silver_2016"
topic: Artificial_Intelligence
date: 2026-04-19
type: briefing
---

# AlphaGo: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Silver, D., Huang, A., Maddison, C. J., Guez, A., Sifre, L., van den Driessche, G., … Hassabis, D. (2016). *Mastering the game of Go with deep neural networks and tree search.* **Nature, 529**(7587), 484–489. DOI: [10.1038/nature16961](https://doi.org/10.1038/nature16961)
**Author(s)**: David Silver, Aja Huang, Chris J. Maddison, Arthur Guez, Laurent Sifre, George van den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, Sander Dieleman, Dominik Grewe, John Nham, Nal Kalchbrenner, Ilya Sutskever, Timothy Lillicrap, Madeleine Leach, Koray Kavukcuoglu, Thore Graepel, Demis Hassabis (Google DeepMind)
**Year**: 2016 (Jan 28, published online)

---

## 1. 핵심 기여 / Core Contribution

### 한국어
AlphaGo는 **딥 컨볼루션 신경망(CNN)** 과 **몬테카를로 트리 탐색(MCTS)** 을 결합하여 바둑에서 인간 프로 기사를 처음으로 이긴 시스템입니다. 핵심 아이디어는 두 종류의 신경망을 학습하는 것입니다. (1) **정책망(policy network)** 은 다음 수의 확률 분포 $p(a \mid s)$ 를 예측하여 탐색의 **너비(breadth)** 를 줄이고, (2) **가치망(value network)** 은 현재 국면의 승률 $v(s)$ 를 예측하여 탐색의 **깊이(depth)** 를 줄입니다. 학습은 3단계 파이프라인으로 진행됩니다. ① 인간 전문가 기보로부터의 지도학습(SL) 정책망 $p_\sigma$, ② 자가 대국(self-play)을 통한 강화학습(RL) 정책망 $p_\rho$, ③ RL 정책망이 둔 자가 대국에서 추출한 3천만 개의 독립 국면으로 학습한 가치망 $v_\theta$. 실제 대국에서는 이 두 네트워크가 **비동기 MCTS** 에 결합되어, 정책망은 유망한 수를 제안하고 가치망은 리프 평가를 담당합니다. AlphaGo는 다른 바둑 프로그램 대비 99.8% 승률을 기록했고, 유럽 챔피언 Fan Hui에게 5대 0으로 완승했습니다. 이는 "적어도 10년은 걸릴 것"이라 여겨졌던 풀 사이즈(19×19) 바둑 정복을 성취한 사건입니다.

### English
AlphaGo is the first computer program to defeat a human professional Go player on a full 19×19 board, achieved by combining **deep convolutional neural networks** with **Monte Carlo Tree Search (MCTS)**. Two networks carry the signal. (1) A **policy network** outputs a probability distribution $p(a \mid s)$ over moves and reduces the **breadth** of the search tree by sampling promising actions. (2) A **value network** predicts the expected outcome $v(s)$ from a position and reduces the **depth** of the search by truncating rollouts with a learned evaluator. Training follows a three-stage pipeline: ① a supervised-learning (SL) policy network $p_\sigma$ trained on 30M human expert moves (57.0% top-1 move-prediction accuracy); ② a reinforcement-learning (RL) policy network $p_\rho$ improved by policy-gradient self-play; ③ a value network $v_\theta$ regressed on 30M self-play positions using the RL network. At test time an **asynchronous MCTS** combines the networks: the policy prior guides action selection via a PUCT-style bonus, a fast rollout policy $p_\pi$ provides one leaf estimate, and the value network provides another, mixed with weight $\lambda$. AlphaGo won 99.8% of games against other Go programs and defeated European champion Fan Hui 5–0—a landmark long thought to be a decade away.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

#### 한국어
바둑은 경우의 수 $b^d$ 에서 $b \approx 250,\ d \approx 150$ 로, 체스 ($b\approx 35, d\approx 80$) 보다 조합적으로 훨씬 큽니다. 체스는 1997년 Deep Blue가 세계 챔피언 Kasparov를 이겼지만, 바둑은 (1) **탐색 공간이 너무 넓고**, (2) **국면 평가가 극히 어렵다**는 이중의 장벽이 있었습니다. 2006년 이후 **몬테카를로 트리 탐색(MCTS)** 이 Crazy Stone, Fuego, Zen, Pachi 같은 프로그램에 적용되어 아마추어 단 수준까지 올라왔지만, 프로 수준은 여전히 요원했습니다. 한편 2012년 AlexNet 이래 CNN이 이미지 인식을 혁신했고, 2014년 Clark & Storkey (정책망), 2015년 Maddison et al.의 연구가 **CNN으로 프로의 수를 예측**할 수 있음을 보였습니다. 강화학습 쪽에서는 2013–15년의 DQN이 Atari 게임을 end-to-end 학습으로 풀어 "범용 RL + 딥러닝"의 가능성을 열었습니다. AlphaGo는 바로 이 흐름—CNN, 정책 경사 RL, MCTS—이 수렴한 결과입니다.

#### English
Go is combinatorially far larger than chess: $b \approx 250$, $d \approx 150$ versus chess's $b\approx 35$, $d\approx 80$. Deep Blue solved chess in 1997 by brute search plus hand-crafted evaluation, but Go presented a double barrier: an enormous search space **and** a board whose positional value resists hand-coded heuristics. From 2006 onward, **Monte Carlo Tree Search (MCTS)** —exemplified by Crazy Stone, Fuego, Zen, and Pachi—pushed computer Go to strong amateur dan level but not to the professional ceiling. Meanwhile, since AlexNet (2012), CNNs had transformed visual recognition; Clark & Storkey (2014) and Maddison et al. (2015) showed that CNNs could **predict expert Go moves** with surprising accuracy. On the RL side, DQN (2013–15) demonstrated that deep networks could be trained end-to-end to play Atari. AlphaGo is the convergence of these three currents—CNNs, policy-gradient RL, and MCTS—into a single system.

### 타임라인 / Timeline

```
1950 ─ Shannon: chess programming essay / "Programming a computer for chess"
1997 ─ Deep Blue beats Kasparov (chess solved at top level)
2006 ─ Modern MCTS (UCT) introduced by Kocsis & Szepesvári
2006–12 Crazy Stone / Fuego / Zen / Pachi reach strong amateur Go
2012 ─ AlexNet: CNNs revolutionise image recognition
2013 ─ DeepMind DQN plays Atari from pixels
2014 ─ Clark & Storkey: CNN policy for Go move prediction
2015 ─ Maddison et al.: move prediction without search reaches strong amateur
2015 Oct AlphaGo 5–0 vs Fan Hui (European champion, 2-dan pro)  ← this paper
2016 Jan Paper published in Nature
2016 Mar AlphaGo 4–1 vs Lee Sedol (9-dan; global media event)
2017 ─ AlphaGo Master, AlphaGo Zero, AlphaZero (no human data)
```

---

## 3. 필요한 배경 지식 / Prerequisites

### 한국어
- **게임트리 탐색 기초**: Minimax, α-β 가지치기, 평가 함수의 역할
- **몬테카를로 트리 탐색(MCTS)**: Selection / Expansion / Simulation (Rollout) / Backup 의 4단계 사이클, UCT 공식 $Q + c\sqrt{\ln N / n}$
- **CNN**: 컨볼루션 필터, stride, ReLU, 분류를 위한 softmax. 19×19 바둑판이 "이미지"로 들어갑니다
- **지도학습**: cross-entropy 손실, SGD, 과적합
- **정책 경사(Policy Gradient) RL**: REINFORCE 추정치 $\nabla_\rho \mathbb{E}[R] = \mathbb{E}[\nabla_\rho \log p_\rho(a|s) \cdot R]$, baseline 사용
- **가치 함수 회귀**: TD(0)와 Monte Carlo 회귀의 차이, 이 논문은 MC 회귀를 사용
- **바둑 규칙 기본**: 흑백 번갈아 두기, 자충수 금지, 집(territory) 계산 — 세부 규칙은 몰라도 됩니다. 논문은 Tromp–Taylor 규칙을 사용합니다
- **직전 논문(#22 DQN)** 의 "딥러닝 + RL" 사고방식

### English
- **Game-tree search basics**: minimax, α-β pruning, why evaluation functions matter.
- **Monte Carlo Tree Search (MCTS)**: the four-step loop (Selection / Expansion / Simulation / Backup) and the UCT formula $Q + c\sqrt{\ln N / n}$.
- **CNNs**: convolutional filters, strides, ReLU, softmax classification. The 19×19 board enters as an "image" with multiple feature planes.
- **Supervised learning**: cross-entropy loss, stochastic gradient descent, overfitting.
- **Policy-gradient RL**: the REINFORCE estimator $\nabla_\rho \mathbb{E}[R] = \mathbb{E}[\nabla_\rho \log p_\rho(a|s) \cdot R]$, and the role of a baseline for variance reduction.
- **Value regression**: the distinction between Monte Carlo and TD targets; this paper uses MC regression with the outcome $z \in \{-1,+1\}$.
- **Basic Go rules**: alternating moves, ko, territory scoring. Tromp–Taylor rules are used here, but deep Go knowledge is not required.
- **Predecessor paper (#22 DQN)**: the "deep learning + RL" mindset.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **SL policy network $p_\sigma$** | 인간 기보 3천만 수로 지도학습된 13-layer CNN 정책망. Top-1 이동 예측 정확도 57.0% (이전 최고 44.4%). / A 13-layer CNN policy network trained via supervised learning on 30M human expert moves; achieves 57.0% top-1 move-prediction accuracy (prior SOTA 44.4%). |
| **Rollout policy $p_\pi$** | 선형 softmax에 수작업 패턴 특징을 더한 "빠른" 정책망. 착수당 ≈2μs (CNN은 3ms)로 시뮬레이션에 사용. 정확도는 24.2%로 낮지만 속도가 중요. / A fast linear softmax policy with handcrafted pattern features, ≈2 μs per move versus 3 ms for the CNN; 24.2% accuracy but used for rollouts where speed dominates. |
| **RL policy network $p_\rho$** | SL 정책망을 초기값으로, 과거 버전들과의 self-play로 policy gradient로 개선. SL 대비 80% 승률, Pachi 대비 85% 승률. / Initialised from $p_\sigma$ and improved by policy-gradient self-play against a pool of prior networks; wins 80% against $p_\sigma$ and 85% against Pachi. |
| **Value network $v_\theta$** | 국면 $s$ 에서 최종 승률 $v^{p_\rho}(s)$ 를 예측하는 CNN. RL 정책의 3천만 자가 대국에서 각 대국당 한 국면만 샘플링해 과적합 방지. / A CNN predicting the expected outcome $v^{p_\rho}(s)$ of a position; trained on 30M self-play games with one position sampled per game to prevent overfitting. |
| **Feature planes (입력 특징)** | 19×19×48 텐서. 돌의 색, 자유도(liberties), 축(ladder), 마지막 수 이후 경과 등 48개 이진·정수 평면. / A 19×19×48 tensor with 48 feature planes: stone colour, liberties, ladder status, turns since move, and more. |
| **MCTS with PUCT bonus** | 선택 단계에서 $a_t = \arg\max_a \big[ Q(s,a) + u(s,a) \big]$, $u \propto p_\sigma(a\mid s) / (1+N(s,a))$. 정책망이 prior로 작동. / Each MCTS step selects $a_t = \arg\max_a \big[ Q(s,a) + u(s,a) \big]$ with $u \propto p_\sigma(a\mid s) / (1+N(s,a))$; the policy net is a prior. |
| **Mixed evaluation $V(s_L)$** | 리프 평가는 $V(s_L) = (1-\lambda) v_\theta(s_L) + \lambda z_L$, 여기서 $z_L$ 은 rollout 승패. $\lambda = 0.5$ 최적. / Leaf value mixes value net and rollout: $V(s_L) = (1-\lambda) v_\theta(s_L) + \lambda z_L$. Optimal $\lambda = 0.5$. |
| **Asynchronous MCTS** | 여러 CPU 스레드가 in-flight 시뮬레이션을 공유. "Virtual loss"로 다양성을 유도. 분산 버전은 40 search threads, 1,202 CPU, 176 GPU. / CPU threads share an in-flight tree; virtual loss diversifies paths. The distributed version uses 40 search threads, 1,202 CPUs, and 176 GPUs. |
| **Elo rating** | 체스식 상대평가 점수. Fan Hui 버전 ≈ 3,144 Elo, 분산 버전 ≈ 3,140+ Elo (아마추어 단 2,100 → 프로 9단 3,600+). / Chess-style relative rating. Single-machine AlphaGo ≈ 3,144 Elo; distributed version higher. (Amateur dan ≈ 2,100, pro 9-dan ≈ 3,600+.) |
| **Tromp–Taylor rules** | 게임의 수학적으로 엄밀한 규칙 세트. 자충수 / 기권 / 2연속 패스로 종료 등. 학습·시뮬레이션에 일관된 룰 제공. / A mathematically unambiguous Go rule set used for consistent training and simulation. |

---

## 5. 수식 미리보기 / Equations Preview

### (1) MCTS 선택 규칙 / Action selection in MCTS

$$
a_t = \arg\max_{a}\ \big[\, Q(s_t, a) + u(s_t, a) \,\big], \qquad u(s, a) = c_{\text{puct}} \cdot \frac{p_\sigma(a \mid s)}{1 + N(s, a)}
$$

- $Q(s,a)$: 지금까지 이 엣지를 지나간 시뮬레이션들의 평균 평가 / mean value of simulations through edge $(s,a)$
- $N(s,a)$: 방문 횟수 / visit count
- $p_\sigma(a\mid s)$: **SL 정책망** 의 prior (RL 정책망이 아니라 **SL** 을 쓰는 것이 놀라운 발견) / prior from the **SL** policy network — surprisingly better than the RL net as a prior
- $u$ 는 방문할수록 감소하여 exploration 감소, $Q$ 로 점점 의존 이동 / $u$ shrinks with $N$, driving exploitation over exploration

### (2) 정책 경사 업데이트 / Policy gradient (REINFORCE with outcome $z$)

$$
\Delta \rho \ \propto\ \frac{\partial \log p_\rho(a_t \mid s_t)}{\partial \rho}\, z_t
$$

- $z_t \in \{-1, +1\}$: 현재 플레이어 관점의 최종 승패 / terminal reward from current player's perspective
- baseline은 나중에 value network 도입 / a baseline is introduced later via the value network
- $\rho$ 는 SL 네트워크 $\sigma$ 로 초기화 / $\rho$ is initialised from $\sigma$

### (3) 가치망 회귀 / Value network regression

$$
\Delta \theta \ \propto\ \frac{\partial v_\theta(s)}{\partial \theta}\, \big( z - v_\theta(s) \big)
$$

- MSE 손실의 그래디언트 / gradient of mean-squared error
- 학습 데이터: 각 self-play 대국에서 **한 국면만** 추출한 3천만 (state, outcome) 쌍 (연속 국면 상관 관계로 인한 과적합 회피) / training set: 30M (state, outcome) pairs with **one position per game** to avoid overfitting from correlated consecutive positions

### (4) 리프 평가 혼합 / Leaf evaluation

$$
V(s_L) = (1 - \lambda)\, v_\theta(s_L) + \lambda\, z_L
$$

- $z_L$: rollout policy $p_\pi$ 로 게임 끝까지 진행한 뒤의 승패 / outcome of a fast rollout played out with $p_\pi$
- $\lambda = 0$: value net만 사용, $\lambda = 1$: rollout만 사용. 실험적으로 $\lambda = 0.5$ 가 최적 / $\lambda = 0$ uses only the value net, $\lambda = 1$ only rollouts; $\lambda = 0.5$ is optimal

### (5) 백업 / Backup of statistics

$$
N(s,a) = \sum_{i=1}^{n} \mathbf{1}(s,a,i), \qquad Q(s,a) = \frac{1}{N(s,a)} \sum_{i=1}^{n} \mathbf{1}(s,a,i)\, V(s_L^i)
$$

- $i$: 시뮬레이션 인덱스 / simulation index
- $\mathbf{1}(s,a,i)$: 시뮬레이션 $i$ 가 엣지 $(s,a)$ 를 지나갔는지의 표시 / indicator that simulation $i$ traversed edge $(s,a)$
- 최종 선택: 루트에서 가장 많이 방문된 수 $\arg\max_a N(s_0, a)$ (최대 Q 가 아니라!) / final move = most-visited root action, **not** the max-Q child, for robustness to noise

---

## 6. 읽기 가이드 / Reading Guide

### 한국어
논문은 Nature 형식으로 짧지만(≈6페이지) 밀도가 매우 높습니다. 다음 순서를 권합니다.

1. **Abstract + Fig 1 먼저**: 3단계 학습 파이프라인을 한눈에 파악 (SL → RL → Value).
2. **"Supervised learning of policy networks"**: 입력 특징, 네트워크 크기(13-layer, 192 filters), 57% 정확도.
3. **"Reinforcement learning of policy networks"**: self-play와 Pachi 대비 85% 승률.
4. **"Reinforcement learning of value networks"**: "한 대국당 한 국면" 데이터 구성 이유가 중요—Table 4와 overfitting 논의.
5. **"Searching with policy and value networks"**: **수식 (1)–(5)** 가 모두 여기에 있음. Fig 3의 MCTS 4단계 그림을 곁에 두고 읽을 것.
6. **"Evaluating the playing strength of AlphaGo"**: Fig 4의 Elo 톱니를 음미. 특히 value-only / rollout-only / mixed 의 비교(Fig 4b)가 핵심 ablation.
7. **Fan Hui 대국 분석 (Fig 5)**: 프로도 이해하기 어려운 수들—"move 37"의 전조.
8. **Methods 섹션**: 시간 있으면—48개 feature plane 리스트(Extended Data Table 2), 분산 시스템 구조, 가중치 세부사항.

### English
The Nature paper is short (~6 pages) but dense. Recommended reading order:

1. **Abstract + Figure 1** to internalise the three-stage pipeline (SL → RL → Value).
2. **"Supervised learning of policy networks"**: input features, 13-layer, 192-filter CNN, 57% accuracy.
3. **"Reinforcement learning of policy networks"**: self-play with a pool of previous networks; 85% win rate vs. Pachi.
4. **"Reinforcement learning of value networks"**: the "one position per game" trick — read carefully; Table 4's overfitting argument is the key methodological insight.
5. **"Searching with policy and value networks"**: **equations (1)–(5)** live here. Keep Figure 3 in sight.
6. **"Evaluating the playing strength of AlphaGo"**: Figure 4's Elo ladder, especially the value-only vs. rollout-only vs. mixed ablation (Fig 4b).
7. **Fan Hui match analysis (Fig 5)**: moves even pros found hard to parse — foreshadowing "move 37" against Lee Sedol.
8. **Methods**: if time permits — the 48 input features (Extended Data Table 2), the distributed system, and hyperparameters.

---

## 7. 현대적 의의 / Modern Significance

### 한국어
AlphaGo는 세 가지 차원에서 현대 AI의 방향을 바꿨습니다.

1. **"딥러닝 + 탐색"의 청사진**: 학습된 정책/가치와 전통적 트리 탐색의 결합은 이후 **AlphaGo Zero (2017)**, **AlphaZero** (체스·쇼기·바둑 통합), **MuZero (2019)** (환경 모델도 학습), **AlphaFold 2** (단백질 구조), **AlphaCode**, 심지어 LLM의 **tree-of-thoughts**, **RLHF + best-of-N** 같은 기법으로 이어졌습니다.
2. **Self-play의 증명**: 인간 데이터 없이도 자가 대국만으로 점점 강해질 수 있다는 아이디어를 공고히 했습니다. AlphaGo Zero는 인간 기보를 **전혀** 쓰지 않았고, 그것이 AlphaGo보다 더 강했습니다.
3. **대중·사회적 충격**: 2016년 3월 Lee Sedol 전은 전 세계 2억 명 이상이 시청한 사건이었고 ("move 37"), AI에 대한 일반 대중의 인식을 바꾸는 분기점이었습니다. "AI winter"의 완전한 종결을 상징합니다.

현재 LLM 시대에도 "신경망(직관) + 탐색(숙고)"의 이중 프로세스 아이디어는 여전히 유효합니다—OpenAI o1 / Anthropic Sonnet의 "thinking" 모드, verifier + search 패러다임은 모두 AlphaGo의 후손입니다.

### English
AlphaGo bent the trajectory of modern AI along three axes.

1. **The "deep learning + search" blueprint.** Coupling a learned policy/value with a classical search algorithm became a template: **AlphaGo Zero (2017)**, **AlphaZero** (one algorithm for chess, shogi, and Go), **MuZero (2019)** (learning the world model too), **AlphaFold 2** (protein structure), **AlphaCode**, and eventually LLM-era ideas like **tree-of-thoughts**, verifier-guided search, and best-of-N with RLHF.
2. **A proof of self-play.** That a system can improve purely through self-play — with no human examples — was not merely conjectured after AlphaGo but **demonstrated** by AlphaGo Zero (2017), which surpassed AlphaGo without any human data.
3. **Cultural and societal impact.** The Lee Sedol match (March 2016, watched by >200M people; "move 37") marked the end of the final "AI winter" in public perception and catalysed the AI investment and policy waves of the late 2010s.

In the LLM era, the "intuition (neural net) + deliberation (search)" dual process is still alive—OpenAI o1 / Anthropic's thinking modes, verifier + search paradigms, and scaling inference-time compute are all descendants of AlphaGo's core bet.

---

## Q&A

### Q1. MCTS에 대한 자세한 설명 / Detailed explanation of MCTS

#### 1. 왜 MCTS인가? / Why MCTS?

**한국어**: 바둑처럼 $b^d \approx 250^{150}$ 규모인 게임에서 전체 트리 minimax는 불가능하다. 따라서 (1) **breadth** 를 줄이고(유망한 수만), (2) **depth** 를 줄여야(중간에 평가) 한다. 체스는 수작업 평가함수가 통하지만 바둑은 그렇지 않다. MCTS의 아이디어는 *"평가함수를 짤 수 없으면 끝까지 무작위로 둬서 이긴 비율로 대신 쓰자"* 이다. 그리고 그 롤아웃 결과를 트리에 누적해 "많이 이긴 가지"로 탐색을 집중시킨다.

**English**: In games like Go with $b^d \approx 250^{150}$, full minimax is hopeless. We must reduce **breadth** (only promising moves) and **depth** (evaluate mid-game). Chess got away with hand-crafted evaluators; Go did not. MCTS's insight: *if you can't evaluate, play random games to the end and use the win rate as a surrogate.* Those rollout outcomes are accumulated in a tree, biasing search toward branches that win more often.

#### 2. 네 단계 루프 / The four-step loop

```
Root s₀
  │
  │ ① Selection — UCT/PUCT로 리프까지 내려감
  ▼
Leaf s_L
  │ ② Expansion — 새 자식 노드 추가, prior = p_σ(a|s)
  │ ③ Simulation — rollout policy로 게임 끝까지, z ∈ {−1, +1}
  │ ④ Backup — 경로상 모든 엣지의 N, Q 업데이트
  ▲ (z 를 루트로 역전파)
```

#### 3. 선택 공식 / Selection formulae

**Classic UCT**:
$$ a^* = \arg\max_a\ Q(s,a) + c\sqrt{\tfrac{\ln N(s)}{N(s,a)}} $$

**AlphaGo PUCT**:
$$ a^* = \arg\max_a\ Q(s,a) + c_{\text{puct}} \cdot p_\sigma(a \mid s) \cdot \frac{\sqrt{N(s)}}{1 + N(s,a)} $$

- **한국어**: UCT는 $\ln N / N(s,a)$ 로만 exploration을 유도하지만, PUCT는 **정책망의 prior $p_\sigma$** 로 "유망한 수를 더 자주" 탐색한다. $N(s,a)=0$ 일 때 분모가 1이라 수치적으로 안정.
- **English**: UCT uses pure log-visit exploration, while PUCT adds the **policy prior $p_\sigma$**, biasing search toward moves the SL network deems likely. The $+1$ in the denominator keeps $N(s,a)=0$ finite.

#### 4. 리프 평가 / Leaf evaluation

$$ V(s_L) = (1 - \lambda)\, v_\theta(s_L) + \lambda\, z_L, \qquad \lambda = 0.5 $$

- **한국어**: 가치망은 **positional pattern**, rollout은 **tactical reading** 에 강해 상관이 낮다. 둘을 섞으면 개별보다 강함 (Fig 4b의 핵심 ablation).
- **English**: The value net captures **positional patterns** while rollouts capture **tactical reading**; the two signals are only weakly correlated, so the mixture outperforms either alone (Fig. 4b's key ablation).

#### 5. 백업 / Backup

$$ N(s,a) \leftarrow N(s,a) + 1, \qquad W(s,a) \leftarrow W(s,a) + V(s_L), \qquad Q(s,a) = \frac{W(s,a)}{N(s,a)} $$

- **한국어**: 2인 제로섬이므로 각 층에서 **부호 반전**. 평균은 온라인으로 계산.
- **English**: Because Go is zero-sum, the sign flips between plies. The mean is maintained online.

#### 6. 최종 수 선택 / Final move

$$ a_{\text{play}} = \arg\max_a\ N(s_0, a) $$

- **한국어**: $Q$ 가 아니라 **방문 횟수 $N$ 최대** 를 택한다. 이유: $N$ 은 분산이 낮고 노이즈에 강함. 수렴 시 $N_{\max}$ 와 $Q_{\max}$ 는 일치하지만 유한 시뮬레이션에서는 $N$ 쪽이 더 안정.
- **English**: The most-visited move, **not** max-Q. The visit count is a lower-variance statistic, robust to fluke high-Q nodes. In the limit, the two agree, but with finite simulations $N$ is safer.

#### 7. Toy 예시 / Worked example

루트 $s_0$, 세 수 $a_1, a_2, a_3$, prior $(0.5, 0.3, 0.2)$, $c_{\text{puct}}=1$.

| Iter | 선택 / Pick | 이유 / Reason | 결과 / Outcome | After |
|---|---|---|---|---|
| 1 | $a_1$ | 모두 0이라 prior 최대 | $z=+1$ | $Q_1{=}1.0, N_1{=}1$ |
| 2 | $a_1$ | $1.0 + 0.5 \sqrt{1}/2 = 1.25$ 최대 | $z=-1$ | $Q_1{=}0.0, N_1{=}2$ |
| 3 | $a_2$ | $0 + 0.3\sqrt{2}/1 \approx 0.42$ > $a_1$의 $0.24$ | — | exploration 발동 |

**한국어**: 승률 0이 된 $a_1$ 를 계속 파지 않고, 한 번도 안 본 $a_2$ 로 옮겨감 — exploration의 본질.
**English**: Instead of drilling into the losing $a_1$, search shifts to the untried $a_2$ — exploration in action.

#### 8. AlphaGo만의 6가지 혁신 / Six innovations AlphaGo adds on top of classic MCTS

1. **정책망 prior로 breadth 축소** / Policy prior shrinks breadth from ~250 to effective tens.
2. **가치망으로 depth 축소** / Value net evaluates leaves without full rollouts.
3. **혼합 평가** / Mixed leaf value combines value net + rollout ($\lambda=0.5$).
4. **Asynchronous + lock-free 트리** / Virtual-loss diversifies concurrent paths.
5. **GPU batching** / Network calls are batched; 40 threads, 8 GPUs (single), 176 GPUs (distributed).
6. **Prior는 SL, value 학습은 RL** / Surprisingly, **SL** policy is the better prior (more diverse); **RL** policy is better for generating value-training data.

#### 9. 세대별 비교 / Generational comparison

| Generation | Search | Policy | Evaluator |
|---|---|---|---|
| α-β (Deep Blue, 1997) | α-β pruning | hand-crafted ordering | hand-crafted eval fn |
| Classic MCTS (2006) | UCT | random / weak heuristics | rollouts only |
| Pattern-MCTS (Crazy Stone, Zen) | UCT | pattern-based | rollouts + patterns |
| **AlphaGo (2016)** | **PUCT** | **CNN (SL + RL)** | **CNN value + rollout** |
| AlphaGo Zero (2017) | PUCT | CNN (self-play only) | **CNN only**, rollouts removed |
| MuZero (2019) | PUCT | CNN | CNN + **learned dynamics** |

**한국어**: 핵심 흐름은 **점점 더 많은 구성요소가 "학습" 되는 것**. AlphaGo는 정책·가치를 학습하고 탐색은 수동, AlphaGo Zero는 rollout까지 없애고, MuZero는 환경 모델도 학습한다.
**English**: The through-line is that **more and more components become learned**. AlphaGo learns policy and value; AlphaGo Zero drops rollouts entirely; MuZero additionally learns the dynamics model.

#### 10. 요약 / Takeaway

- **한국어**: MCTS는 "트리 확장 + 롤아웃 + 통계 누적"의 반복. AlphaGo는 그 **prior** 자리에 정책망, **evaluator** 자리에 가치망을 끼워넣어 바둑을 정복했다. 정책망은 breadth, 가치망은 depth를 줄이는 이중 역할을 수행한다.
- **English**: MCTS is an iterated loop of "expand tree + roll out + accumulate statistics." AlphaGo slots a policy network into the **prior** role and a value network into the **evaluator** role. The policy network shrinks breadth, the value network shrinks depth—together they crack Go.

---

