---
title: "Pre-Reading Briefing: Playing Atari with Deep Reinforcement Learning (DQN)"
paper_id: "22_mnih_2013"
topic: Artificial_Intelligence
date: 2026-04-19
type: briefing
---

# Playing Atari with Deep Reinforcement Learning (DQN): Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., & Riedmiller, M. (2013). *Playing Atari with Deep Reinforcement Learning*. arXiv:1312.5602 (NeurIPS 2013 Deep Learning Workshop).
**Author(s)**: Volodymyr Mnih, Koray Kavukcuoglu, David Silver et al. (DeepMind)
**Year**: 2013 (NeurIPS workshop; Nature publication 2015)

---

## 1. 핵심 기여 / Core Contribution

### 한국어
DQN은 **강화학습(RL)과 딥러닝을 최초로 성공적으로 결합**한 연구로, 게임 화면 픽셀만 보고(자체 특징 공학 없음) 7개 Atari 2600 게임에서 인간 수준 혹은 그 이상의 플레이를 학습했습니다. 이전 RL은 수작업으로 설계한 저차원 state feature에 의존했는데, DQN은 원시 이미지 → CNN으로 바로 Q-값 근사. 그러나 순진하게 neural network로 Q-learning을 하면 **심각한 불안정성**이 나타납니다 (연속 샘플의 상관관계, target의 non-stationarity, Q 값의 작은 변화가 policy를 크게 흔듦). 논문은 두 가지 핵심 기법으로 이를 해결합니다: (1) **Experience Replay** — 과거 transition을 버퍼에 저장하고 랜덤 샘플링하여 i.i.d. 근사를 회복, (2) **CNN-based Q-function** — 여러 프레임을 쌓아 시간적 정보를 포착. 같은 네트워크 구조와 하이퍼파라미터로 7개 게임 중 6개에서 기존 방법을 능가, 3개에서 인간 전문가를 능가했습니다. 이 연구는 **"범용 에이전트(general agent)"** 개념의 씨앗이 되었고, 2015년 Nature 후속 논문(Target Network 추가)으로 이어져 AlphaGo, AlphaZero, MuZero, GPT-RLHF, Sora 등 모든 현대 RL/심층 강화학습의 기반이 됩니다.

### English
DQN is the first work to **successfully combine reinforcement learning with deep learning**, learning to play seven Atari 2600 games at human-level or above from raw pixels — with no hand-crafted features. Prior RL relied on engineered low-dimensional state representations; DQN uses a CNN to approximate Q-values directly from screen pixels. However, naively combining neural networks with Q-learning produces **severe instability** (correlated sequential samples, non-stationary targets, small Q-value changes causing large policy shifts). The paper's key fixes: (1) **Experience Replay** — store transitions in a buffer and sample randomly to recover the i.i.d. assumption; (2) **CNN Q-function** — stack recent frames to capture temporal information. With a single architecture and hyperparameter set, DQN surpasses prior methods in 6 of 7 games and outperforms human experts in 3. This work seeded the concept of a **"general agent"**, and via the 2015 Nature follow-up (adding Target Network) became the foundation for AlphaGo, AlphaZero, MuZero, RLHF in LLMs, and modern deep RL as a whole.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

**한국어**
1989년 Watkins의 Q-learning 이후, 테이블 기반 RL은 이론적으로 아름답지만 고차원 state에서는 폭발적인 표 크기 때문에 실용성이 없었습니다. Function approximation으로 Q-값을 표현하려는 시도는 여러 번 있었지만(Tesauro의 TD-Gammon 1995가 backgammon에서 성공한 예외), 대체로 divergence(발산)하거나 불안정했습니다. Tsitsiklis & Van Roy (1997)는 **function approximation + bootstrapping + off-policy = 발산 가능** 이라는 "deadly triad"를 이론적으로 정식화했고, 이것이 RL과 neural network 결합의 큰 걸림돌이었습니다. 2010년대 초 딥러닝이 CNN(#13 AlexNet 2012)과 RNN 등에서 폭발적 성공을 거두자, DeepMind의 Mnih 등은 "딥러닝으로 RL을 하면 어떻게 될까"라는 질문에 **experience replay + raw pixel input + CNN + Q-learning** 의 조합으로 답했고, 이것이 DQN입니다.

**English**
After Watkins' Q-learning (1989), tabular RL was theoretically elegant but impractical in high-dimensional state spaces due to exponential table size. Attempts at function approximation were often unstable (with the notable exception of Tesauro's TD-Gammon, 1995, on backgammon). Tsitsiklis & Van Roy (1997) formalized the "deadly triad": **function approximation + bootstrapping + off-policy = divergence possible**. This was the major obstacle to combining neural networks with RL. In the early 2010s, deep learning exploded via CNNs (paper #13 AlexNet 2012) and RNNs. DeepMind's Mnih et al. asked "what if we do RL with deep learning?" and answered with **experience replay + raw pixel input + CNN + Q-learning** — this is DQN.

### 타임라인 / Timeline

```
1951  Robbins-Monro — stochastic approximation
1988  Sutton        — TD(λ) learning
1989  Watkins       — Q-learning
1992  Lin           — Experience replay (for neural net RL)
1995  Tesauro       — TD-Gammon (neural net + TD learning for backgammon)
1997  Tsitsiklis    — The "deadly triad" impossibility
2005  Riedmiller    — Neural Fitted Q (NFQ)
2012  Krizhevsky    — AlexNet (deep learning revolution)           [#13]
2013  Mnih (DQN)    — Playing Atari with Deep RL                   ★ THIS PAPER
2015  Mnih (Nature) — Human-level control (DQN + Target Network, 49 games)
2015  Silver (AlphaGo Fan Hui) — RL + MCTS on Go
2016  AlphaGo Lee    — beats world champion
2016  van Hasselt    — Double DQN
2016  Wang           — Dueling DQN
2017  Silver (AlphaZero) — self-play master
2017  Schulman       — PPO                                          [#24]
2019  OpenAI Five   — Dota 2
2022  ChatGPT / RLHF — RL for language models
2024+ 모든 RL-based LLM/로봇/게임 에이전트 — DQN의 사상을 계승
```

---

## 3. 필요한 배경 지식 / Prerequisites

### 한국어
- **Markov Decision Process (MDP)**: state $s$, action $a$, reward $r$, transition $P(s'|s,a)$, discount $\gamma$.
- **Q-learning** (Watkins 1989): $Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$. Off-policy, bootstrapped.
- **Bellman equation**: $Q^*(s,a) = \mathbb{E}[r + \gamma \max_{a'} Q^*(s',a')]$.
- **Function approximation**: Q를 테이블 대신 parameterized function $Q_\theta$로.
- **Exploration vs exploitation**: ε-greedy 정책.
- **CNN** (#13 AlexNet): 이 논문에서 Q 함수 근사에 사용.
- **SGD/RMSProp**: 훈련 최적화.
- **Atari 2600 emulator (ALE)**: Bellemare 2013의 표준 벤치마크.

### English
- **Markov Decision Process (MDP)**: state $s$, action $a$, reward $r$, transition $P(s'|s,a)$, discount $\gamma$.
- **Q-learning** (Watkins 1989): $Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$. Off-policy, bootstrapped.
- **Bellman equation**: $Q^*(s,a) = \mathbb{E}[r + \gamma \max_{a'} Q^*(s',a')]$.
- **Function approximation**: replace Q-table with a parameterized $Q_\theta$.
- **Exploration vs exploitation**: ε-greedy policy.
- **CNN (paper #13 AlexNet)**: used here to approximate Q.
- **SGD/RMSProp**: training optimizer.
- **Atari 2600 emulator (ALE, Bellemare 2013)**: standard RL benchmark.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **Q-function** $Q(s,a)$ | 상태 $s$에서 행동 $a$를 취한 뒤 최적 정책을 따랐을 때의 기대 누적 보상. Expected cumulative reward for taking action $a$ in state $s$ then following the optimal policy. |
| **Bellman equation** | $Q^*(s,a) = \mathbb{E}[r + \gamma \max_{a'} Q^*(s',a')]$. Q 함수가 만족해야 할 재귀 방정식. Recursive consistency equation for $Q^*$. |
| **Experience Replay** | 과거 transition $(s, a, r, s')$을 버퍼 $\mathcal{D}$에 저장 후 무작위 배치로 학습. Store past transitions in a buffer $\mathcal{D}$, sample random mini-batches. |
| **ε-greedy** | 확률 $\varepsilon$으로 무작위 행동, $1-\varepsilon$로 argmax Q. Explore with prob ε, exploit with 1−ε. |
| **TD error** | $\delta = r + \gamma \max_{a'} Q_\theta(s', a') - Q_\theta(s, a)$. Q 업데이트의 "오차". Temporal-difference error driving Q updates. |
| **Off-policy** | 데이터 수집 정책(behavior)과 학습 대상 정책(target)이 다를 수 있음. Behavior and target policies may differ; Q-learning is off-policy. |
| **Bootstrapping** | 추정값을 또 다른 추정값으로 업데이트. Q(s,a)를 $\max Q(s',a')$로 갱신. Updating an estimate using another estimate. |
| **Deadly triad** | Function approx + bootstrap + off-policy의 결합이 발산을 일으킬 수 있다는 고전적 경고. Combination known to cause divergence in theory. |
| **Frame stacking** | 최근 $k=4$ 프레임을 쌓아 시간적 정보(속도) 제공. Stack last 4 frames to capture temporal info. |
| **Atari Learning Environment (ALE)** | Atari 2600 emulator 기반 표준 RL 벤치마크 (Bellemare 2013). Standard RL benchmark over Atari 2600 games. |
| **Reward clipping** | 모든 보상을 {−1, 0, +1}로 제한하여 게임 간 scale 차이 해소. Clamp rewards to {−1, 0, +1} to normalize across games. |
| **Target Q-network** | 이 2013 논문에는 **없음**. 2015 Nature 후속에서 도입되어 target의 non-stationarity를 완화. Introduced in the 2015 Nature follow-up (not in this paper). |

---

## 5. 수식 미리보기 / Equations Preview

### (1) Optimal Q-function / 최적 Q 함수 (Bellman optimality)
$$
Q^*(s, a) = \mathbb{E}_{s' \sim \mathcal{E}}\!\left[ r + \gamma \max_{a'} Q^*(s', a') \;\big|\; s, a \right]
$$
$\mathcal{E}$는 환경의 dynamics. $\gamma \in [0, 1)$는 discount factor.
$\mathcal{E}$ is the environment dynamics; $\gamma$ is the discount.

### (2) Q-learning iteration / Q-learning 갱신
$$
Q_{i+1}(s,a) = \mathbb{E}\!\left[ r + \gamma \max_{a'} Q_i(s', a') \;\big|\; s, a \right]
$$
$i \to \infty$일 때 $Q_i \to Q^*$ (테이블 경우).
Converges to $Q^*$ as $i \to \infty$ (tabular case).

### (3) DQN loss (핵심 식) / DQN 손실
$$
L_i(\theta_i) = \mathbb{E}_{(s,a,r,s') \sim U(\mathcal{D})}\!\left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta_{i-1}) - Q(s, a; \theta_i) \right)^2 \right]
$$

- $U(\mathcal{D})$: 경험 버퍼에서 **균등 랜덤 샘플링**.
- $\theta_{i-1}$: 이전 iteration의 파라미터를 target 계산에 사용 (하지만 2013 논문에서는 매 step마다 업데이트되므로 target network는 없음. 2015 Nature에서는 target network를 별도로 고정).
- $\theta_i$: 현재 학습 중인 파라미터.

Key loss; $U(\mathcal{D})$ is uniform sampling from the experience replay buffer. Note: in this 2013 paper, $\theta$ is updated every step — no separate target network yet (added in 2015 Nature).

### (4) Gradient of DQN loss / DQN 손실의 기울기
$$
\nabla_{\theta_i} L_i(\theta_i) = \mathbb{E}\!\left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta_{i-1}) - Q(s, a; \theta_i) \right) \nabla_{\theta_i} Q(s, a; \theta_i) \right]
$$
표준 MSE loss 기울기 형태. SGD/RMSProp으로 업데이트.

Standard MSE gradient; trained with SGD/RMSProp.

### (5) ε-greedy action selection / 행동 선택
$$
a_t = \begin{cases} \text{random action} & \text{with probability } \varepsilon_t \\ \arg\max_a Q(\phi(s_t), a; \theta) & \text{otherwise} \end{cases}
$$
$\varepsilon_t$는 훈련 초기 1.0 → 0.1로 선형 감소 (처음 1M steps).

$\varepsilon_t$ decays linearly from 1.0 to 0.1 over the first 1M steps.

---

## 6. 읽기 가이드 / Reading Guide

### 한국어
이 논문은 8페이지로 짧지만 밀도가 높습니다. 다음 순서로 읽으시면 효과적:

- **§1 Introduction**: "RL + 딥러닝이 왜 어려운가"의 세 가지 문제를 명확히 파악 (correlated samples, non-stationary targets, policy oscillation).
- **§2 Background**: MDP, Q-learning의 표준 수식. Q-learning을 아신다면 빠르게 훑기.
- **§3 Related Work**: TD-Gammon, NFQ 등 선행 연구와의 비교. DQN의 차별점(on-line, CNN, pixel input)에 주목.
- **§4 Deep Reinforcement Learning**: **가장 중요한 섹션**. Experience replay (Algorithm 1) 정독. 식 (2)의 loss 함수와 업데이트 규칙 이해.
- **§5 Experiments**: 전처리(§5.1: grayscale, 84×84 crop, 4-frame stacking), 아키텍처(§5.2: 2개 conv + 2개 FC), 훈련 (§5.3), 결과 (§5.4: 표 1과 그림 2). **reward clipping**에 주의.
- **§6 Conclusion**: 한 페이지.

**Algorithm 1 (DQN with Experience Replay)**은 이 논문의 심장입니다. 8줄짜리 pseudocode를 줄 단위로 이해하시면 논문의 90%를 파악한 것입니다.

### English
Short but dense (8 pages). Suggested reading order:
- **§1 Introduction**: three obstacles to RL+DL — correlated samples, non-stationary targets, policy oscillation.
- **§2 Background**: MDP, standard Q-learning. Skim if familiar.
- **§3 Related Work**: comparison to TD-Gammon, NFQ. Note what makes DQN different: on-line, CNN, raw pixels.
- **§4 Deep Reinforcement Learning**: **most important**. Read Algorithm 1 line-by-line. Understand the loss in Eq. (2).
- **§5 Experiments**: preprocessing (§5.1: grayscale, 84×84 crop, 4-frame stacking), architecture (§5.2: 2 conv + 2 FC), training (§5.3), results (§5.4: Table 1, Figure 2). Note **reward clipping**.
- **§6 Conclusion**: one page.

**Algorithm 1 (DQN with Experience Replay)** is the heart of the paper. Understanding its 8 lines of pseudocode is 90% of the paper.

---

## 7. 현대적 의의 / Modern Significance

### 한국어
DQN은 현대 심층 강화학습 전체의 **출발점**입니다. 이 논문에서 확립된 3가지 원리는 지금도 유지됩니다:

**1. Experience Replay** — 오늘날 거의 모든 off-policy RL(DQN, DDPG, SAC, 그리고 로봇 학습)에서 쓰입니다. Prioritized Replay (Schaul 2015), Hindsight Replay (Andrychowicz 2017) 등의 변형으로 발전.

**2. CNN-based value/policy network** — 픽셀에서 직접 학습하는 패러다임은 로봇 시각 제어, 자율주행, 게임 AI로 이어졌습니다. AlphaGo Zero도 바둑판 이미지에서 CNN으로 value+policy를 학습.

**3. 범용 에이전트(general agent)의 씨앗** — "같은 아키텍처/하이퍼파라미터로 여러 태스크" 개념이 Multi-task RL, Meta-RL, 현대 foundation model RL의 기초.

**후속 발전**:
- **2015 Nature DQN**: Target Network 추가로 49개 Atari 게임 정복
- **Double DQN** (2016): Q 값 overestimation 완화
- **Dueling DQN** (2016): V(s)와 A(s,a)로 분해
- **Rainbow** (2017): 6가지 개선 결합
- **AlphaGo/AlphaZero**: DQN과 MCTS 결합
- **PPO** (2017, #24): policy gradient 계열의 현대 표준
- **RLHF (GPT/Claude)**: LLM 정렬 — PPO 기반
- **MuZero**: 환경 모델까지 학습
- **Decision Transformer / Gato**: RL을 sequence modeling으로

오늘날 ChatGPT의 RLHF, 자율 로봇, 자동 게임 플레이, 생성 모델의 강화 정렬 — 이 모든 것의 **직계 조상**이 DQN입니다. 이 논문이 없었다면 현재의 "AI가 상호작용으로 학습하는" 시대는 존재하지 않았을 것입니다.

### English
DQN is the **starting point** of all modern deep RL. The three principles it established still hold:

**1. Experience Replay** — used in nearly every off-policy RL today (DQN, DDPG, SAC, and robotic learning). Evolved into Prioritized Replay (Schaul 2015), Hindsight Replay (Andrychowicz 2017).

**2. CNN-based value/policy network** — learning directly from pixels spread to robot vision, autonomous driving, and game AI. AlphaGo Zero also learns value+policy CNN from board images.

**3. Seed of the general agent** — "same architecture/hyperparameters across tasks" became the basis of multi-task RL, meta-RL, and modern foundation-model RL.

**Descendants**:
- **DQN Nature 2015**: Target Network, 49 Atari games at human level
- **Double DQN** (2016): mitigates Q overestimation
- **Dueling DQN** (2016): V(s) + A(s,a) decomposition
- **Rainbow** (2017): 6 improvements combined
- **AlphaGo/AlphaZero**: RL + MCTS
- **PPO** (2017, paper #24): modern policy-gradient standard
- **RLHF (GPT/Claude)**: LLM alignment via PPO
- **MuZero**: learns the environment model
- **Decision Transformer / Gato**: RL as sequence modeling

Today's ChatGPT RLHF, autonomous robots, game-playing agents, and RL-aligned generative models all trace their lineage to DQN. Without this paper, the current era of "AI that learns by interaction" would not exist.

---

## Q&A

### Q1. Q-learning에 대한 설명 / What is Q-learning?

DQN을 이해하려면 Q-learning이 먼저 명확해야 합니다. 단계별로 설명하겠습니다.

To understand DQN, Q-learning must be clear first. Step by step below.

---

#### ① MDP부터 다시 — 문제 설정 / The MDP setup

**한국어**
강화학습이 풀고자 하는 문제는 **Markov Decision Process (MDP)** 로 정식화됩니다:
- **State** $s$: 지금 환경이 어떤 상태에 있는가 (예: Atari 화면 픽셀, 바둑판, 로봇 관절 각도).
- **Action** $a$: 에이전트가 할 수 있는 행동 (예: 레버 좌우, 돌 놓기, 관절 토크).
- **Reward** $r$: 행동 직후 받는 즉시 보상 (Atari 점수 변화, 게임 승리 +1).
- **Transition** $P(s' \mid s, a)$: $s$에서 $a$를 하면 다음 상태 $s'$가 어떻게 되는가 (보통 stochastic).
- **Discount** $\gamma \in [0, 1)$: 미래 보상을 얼마나 할인할 것인가. 보통 0.99.

에이전트의 목표는 **할인된 누적 보상을 최대화하는 정책(policy) $\pi(a|s)$** 를 찾는 것입니다:
$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$$

**English**
The RL problem is formalized as an MDP — state, action, reward, transition, discount. The agent maximizes cumulative discounted reward $G_t = \sum_k \gamma^k r_{t+k}$.

---

#### ② Q 함수 — "상태-행동 쌍의 가치" / The Q-function

**한국어**
상태 $s$에서 행동 $a$를 취한 뒤, **최적의 정책**을 따른다면 얻게 될 기대 누적 보상을 $Q^*(s, a)$라 정의합니다:
$$Q^*(s, a) = \mathbb{E}\!\left[ \sum_{k=0}^{\infty} \gamma^k r_{t+k} \;\Big|\; s_t = s, a_t = a, \pi^* \right]$$

직관: "이 상태에서 이 행동을 하면 (앞으로 잘만 하면) 총 얼마나 좋은가?"

만약 $Q^*$를 안다면 최적 행동은 단순합니다:
$$a^* = \arg\max_a Q^*(s, a)$$

즉 **Q 함수만 알면 최적 정책도 얻는다**. 이것이 Q-learning의 핵심 아이디어입니다 — 정책을 직접 배우지 말고 Q를 배우자.

**English**
$Q^*(s, a)$ = expected cumulative discounted reward from taking action $a$ in state $s$, then following the optimal policy. If you know $Q^*$, the optimal action is simply $a^* = \arg\max_a Q^*(s, a)$. So Q-learning's insight: **learn Q, not the policy directly**.

---

#### ③ Bellman 방정식 — Q의 재귀 구조 / The Bellman equation

**한국어**
$Q^*$가 만족해야 하는 자기 모순 없는 방정식:
$$Q^*(s, a) = \mathbb{E}_{s'}\!\left[ r + \gamma \max_{a'} Q^*(s', a') \;\Big|\; s, a \right]$$

**어떻게 유도되나?**
1. 지금 상태 $s$에서 $a$를 하면 즉시 보상 $r$을 받고, 다음 상태 $s'$로 감.
2. 그 다음부터는 "상태 $s'$에서 최적 행동 $a' = \arg\max Q^*(s', a')$를 할 때의 가치"를 얻음. 이는 $\max_{a'} Q^*(s', a')$ 자체.
3. 미래 보상은 $\gamma$로 할인.

즉 "지금 가치 = 즉시 보상 + $\gamma \times$ 다음 상태의 최고 가치"라는 **자기 참조적 재귀**. 이것이 Bellman optimality equation입니다.

**English**
$Q^*$ satisfies the self-consistent recursion: "value now = immediate reward + $\gamma$ × best value at next state." This **self-referential structure** is the engine of all value-based RL.

---

#### ④ Q-learning 업데이트 규칙 / The Q-learning update rule

**한국어**
Watkins (1989)의 통찰: Bellman 방정식을 **목표(target)** 로 삼아 현재 추정치 $Q$를 그쪽으로 조금씩 당기자.

$$Q(s, a) \leftarrow Q(s, a) + \alpha \underbrace{\Big[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \Big]}_{\text{TD error } \delta}$$

- $\alpha$: learning rate (예: 0.1)
- $r + \gamma \max_{a'} Q(s', a')$: **TD target** (Bellman 우변의 샘플 추정).
- $Q(s, a)$: 현재 추정.
- $\delta$: target과 현재의 차이 — **temporal-difference (TD) error**.

**의미**: "현재 추정 $Q(s,a)$가 실제 경험한 ($r$ 받고 $s'$로 감) + 다음 상태의 최고 추정 가치보다 작으면, 그쪽으로 올리고, 크면 내린다." 무수히 많은 경험 위에서 반복하면 $Q \to Q^*$ (tabular, under mild conditions; Watkins & Dayan 1992의 수렴 증명).

**English**
Watkins' insight: treat Bellman's RHS as a target and nudge $Q(s,a)$ toward it. The update pulls current estimate toward $r + \gamma \max_{a'} Q(s', a')$ by a small step $\alpha$. Under tabular representation and mild conditions, $Q \to Q^*$ (Watkins & Dayan 1992).

---

#### ⑤ 구체적 예시 — 3-상태 그리드 / Concrete example: 3-state grid

**한국어**
다음 간단한 환경을 보자:
```
  [S1] ──a_right──▶ [S2] ──a_right──▶ [GOAL, r=+1]
   │                 │
  a_left            a_left
   ▼                 ▼
  stay              [S1]
```
$\gamma = 0.9$, Q 테이블을 0으로 초기화.

**Step 1**: S1에서 $a_{\text{right}}$ → S2, $r=0$. 그 다음 **S2**에서 무엇을 하든 아직 Q는 0. 업데이트:
$$Q(S1, \text{right}) \leftarrow 0 + 0.1[0 + 0.9 \cdot 0 - 0] = 0$$
아무 변화 없음.

**Step 2**: S2에서 $a_{\text{right}}$ → GOAL, $r=+1$. 업데이트:
$$Q(S2, \text{right}) \leftarrow 0 + 0.1[1 + 0.9 \cdot 0 - 0] = 0.1$$

**Step 3**: 다시 S1→S2. 이번에는:
$$Q(S1, \text{right}) \leftarrow 0 + 0.1[0 + 0.9 \cdot \max(0, 0.1) - 0] = 0.009$$
**보상이 뒤에서 앞으로 "역전파"** 되는 것을 볼 수 있다. 이 과정을 수천 번 반복하면:
- $Q(S2, \text{right}) \to 1.0$
- $Q(S1, \text{right}) \to 0.9$

이것이 Q-learning이 작동하는 방식입니다 — **가치가 목표에서 시작해서 Bellman 재귀를 타고 거꾸로 흐른다**.

**English**
In a small grid, rewards "propagate backward" through repeated Q updates via the Bellman recursion. After many episodes, $Q(S_2, \text{right}) \to 1.0$ and $Q(S_1, \text{right}) \to 0.9$, revealing the correct long-term values.

---

#### ⑥ Q-learning의 세 가지 특성 / Three key properties

**한국어**

1. **Off-policy**: 데이터를 수집한 정책(behavior policy, 예: ε-greedy, 심지어 랜덤)과 학습하는 정책(target policy, argmax Q)이 **달라도 된다**. 이유: target이 $\max_{a'} Q(s', a')$로 정의되어 behavior policy와 무관.
   → 장점: 과거 데이터(experience replay 버퍼)를 재활용 가능.

2. **Bootstrapping**: 업데이트 target에 **또 다른 Q 추정치** $\max Q(s', a')$가 들어감. Monte Carlo 방법(실제 에피소드 끝까지 가서 누적 보상으로 업데이트)과 대조적.
   → 장점: 에피소드가 끝날 때까지 기다릴 필요 없음, 빠른 학습.
   → 단점: target이 "정확하지 않은 추정"이라 불안정 가능.

3. **Temporal Difference (TD)**: 한 step의 경험 $(s, a, r, s')$으로 즉시 업데이트. 이름의 "temporal difference"는 $Q(s,a)$와 $r + \gamma Q(s',a')$의 **시간적 차이**를 의미.

**English**
1. **Off-policy**: behavior ≠ target policy OK → replay buffer usable.
2. **Bootstrapping**: target uses another Q estimate → fast but can be unstable.
3. **Temporal-difference**: updates from one-step transitions, no waiting for episode end.

---

#### ⑦ Deadly Triad — 왜 DQN이 필요했나 / Why DQN needs its tricks

**한국어**
Q-learning에 **neural network 같은 function approximation**을 결합하면 다음 세 가지가 합쳐져 **발산(divergence)** 이 가능:

1. **Function approximation**: 테이블이 아닌 $Q_\theta(s, a)$ 함수로 근사.
2. **Bootstrapping**: target에 $Q_\theta$ 자체가 들어감.
3. **Off-policy**: 학습 데이터 분포와 policy가 다름.

이 **deadly triad** (Tsitsiklis & Van Roy 1997, Sutton 1988)는 이론적으로 수렴을 보장하지 못합니다. 실제로 naive하게 neural Q-learning을 시도하면 Q 값이 폭발하거나 oscillate하는 일이 흔합니다.

**DQN의 트릭**이 바로 이 triad의 문제를 완화합니다:
- **Experience replay** → 샘플 간 상관관계 깨고 i.i.d. 근사 회복 (off-policy의 부작용 완화).
- **Target network** (2015 Nature 버전) → target 계산에 별도의 고정 파라미터 사용 → bootstrapping의 non-stationarity 완화.
- **Reward clipping + RMSProp** → 기울기 스케일 안정화.

이 세 가지가 바로 DQN이 Q-learning + neural network를 처음으로 **실제로 작동시킨** 핵심입니다.

**English**
Combining Q-learning with neural nets creates the **deadly triad**: function approximation + bootstrapping + off-policy → possible divergence. DQN's tricks — experience replay, (2015) target network, reward clipping, RMSProp — are precisely what tames this triad. This is why DQN is not just "Q-learning with a neural net" but a carefully engineered solution to a 20-year-old problem.

---

#### ⑧ 한 줄 요약 / One-line summary

**한국어**
**Q-learning = "Bellman 방정식을 target으로 삼아, 경험 $(s,a,r,s')$마다 $Q(s,a)$를 target 방향으로 조금씩 당기는 알고리즘"** 입니다. DQN은 여기서 Q를 테이블 대신 CNN으로 근사하고, 발산하지 않도록 experience replay를 추가한 것입니다.

**English**
**Q-learning = "pull $Q(s,a)$ toward the Bellman target $r + \gamma \max_{a'} Q(s', a')$ every transition."** DQN replaces the Q-table with a CNN and adds experience replay to tame the deadly triad.

---

### Q2. Q-learning이 어떻게 Deep Learning과 연결되나? / How does Q-learning connect to deep learning?

핵심은 **"테이블 대신 함수, 그것도 학습 가능한 깊은 함수로"** 입니다. 이 연결을 제대로 이해하려면 먼저 deep learning이 무엇인지부터 명확히 해야 합니다.

The core connection is: **"replace the table with a function — specifically a deep, learnable one."** To see this clearly, we first need to define deep learning precisely.

---

#### ① Deep Learning이란 무엇인가 / What is Deep Learning?

**한국어**
**Deep Learning의 정의 (세 가지 요소)**:

1. **다층(multi-layer) 파라미터 함수**: 입력 $x$를 일련의 선형 변환과 비선형 함수를 거쳐 출력으로 매핑.
   $$y = f_\theta(x) = \sigma_L \!\big(W_L \cdots \sigma_2(W_2 \sigma_1(W_1 x + b_1) + b_2) \cdots + b_L\big)$$
   여기서 $\theta = \{W_1, b_1, \ldots, W_L, b_L\}$는 학습 가능 파라미터, $\sigma$는 비선형성(ReLU 등).

2. **End-to-end 학습 (raw input → target)**: 수작업 feature engineering 없이, 원시 데이터(픽셀, 오디오 파형, 토큰)에서 목표(레이블, Q 값, 행동 확률)까지 **한 번에** 학습. "사람이 feature를 설계"하는 대신 **네트워크가 계층적 feature를 자동으로 학습**.

3. **Backpropagation으로 최적화**: 모든 파라미터에 대한 손실함수의 기울기 $\nabla_\theta L$을 체인 룰로 계산 → SGD류 알고리즘으로 업데이트.

**핵심 성질**: 충분히 깊고 넓은 신경망은 **임의의 연속 함수를 근사** 가능 (Universal Approximation Theorem, Cybenko 1989; Hornik 1991). 즉 **신경망 = 강력한 일반 함수 근사기(function approximator)**.

**English**
Deep learning is defined by three components:
1. **Multi-layer parameterized function** $f_\theta(x)$ — stacked linear transforms interleaved with nonlinearities.
2. **End-to-end learning from raw input to target** — no hand-engineered features; the network learns hierarchical representations automatically.
3. **Optimized by backpropagation** — gradient of the loss computed via chain rule, applied with SGD-family updates.

Key property: sufficiently deep/wide networks are **universal function approximators** (Cybenko 1989, Hornik 1991).

---

#### ② Q-learning의 벽 — 테이블의 한계 / The wall Q-learning hits

**한국어**
④에서 본 Q-learning 업데이트는 깔끔합니다:
$$Q(s, a) \leftarrow Q(s, a) + \alpha\!\left[r + \gamma \max_{a'} Q(s', a') - Q(s, a)\right]$$

그러나 이것이 작동하려면 $Q(s,a)$를 저장할 **테이블**이 있어야 합니다. state가 $|S|$개, action이 $|A|$개면 $|S| \times |A|$ 크기의 테이블.

**Atari의 경우**:
- 화면: 210 × 160 픽셀 × 3 채널 (RGB) = 100,800차원
- 각 픽셀 값 0-255 → 이론적 state 수 = $256^{100{,}800} \approx 10^{242{,}000}$
- 우주의 원자 수보다 훨씬 많음. 테이블로 불가능.

또한 **generalization이 없음**: 거의 비슷한 두 화면(공이 1픽셀 옆으로 이동)도 완전히 다른 state로 취급. 한 화면에서 배운 게 다른 화면으로 전혀 전달되지 않음.

**결론**: 테이블 Q-learning은 **저차원 이산 state**에서만 실용적. 고차원 또는 연속 state에서는 다른 방법이 필요.

**English**
Tabular Q-learning needs a table of size $|S| \times |A|$. Atari has $\sim 10^{242{,}000}$ possible states (210×160×3 pixels, 256 values each) — impossible to store. Also, nearly identical screens (ball shifted 1 pixel) are treated as totally different states — zero generalization. Tabular Q-learning is practical only for low-dimensional discrete state spaces.

---

#### ③ Function Approximation — 테이블을 함수로 / Function approximation: replace table with function

**한국어**
해결책: Q를 **테이블이 아닌 파라미터화된 함수**로 표현.
$$Q(s, a) \approx Q_\theta(s, a)$$

여기서 $\theta$는 학습 가능한 파라미터 (예: 몇 개의 weight vector). 이제 state $s$를 **직접 저장**하지 않고, 함수가 받은 $s$를 처리해서 Q를 출력.

**Function approximation의 두 단계 역사**:

**(a) 선형 function approximation (1990s~2000s)**
$$Q_\theta(s, a) = \theta^\top \phi(s, a)$$
- $\phi(s, a)$: **사람이 설계한** feature vector (예: "플레이어 x좌표", "벽까지 거리", "적 개수").
- TD-Gammon (Tesauro 1995)가 이 방식으로 backgammon 세계 챔피언 수준 달성.
- **한계**: 좋은 feature $\phi$를 설계하려면 도메인 전문가가 필요. 게임마다 다시 설계. Atari처럼 다양한 게임을 **하나의 방식**으로 풀 수 없음.

**(b) 비선형 function approximation = Deep Learning (2010s~)**
$$Q_\theta(s, a) = \text{NeuralNet}_\theta(s, a)$$
- **원시 입력(픽셀)에서 feature를 자동 학습**.
- 같은 아키텍처로 여러 게임/태스크 처리 가능.
- 단점: Universal approximation은 가능하지만 **최적화가 어렵다**(deadly triad).

**English**
(a) Linear function approximation $Q_\theta = \theta^\top \phi(s,a)$ with hand-crafted features $\phi$ — e.g., TD-Gammon (1995). Limited by feature design.
(b) **Deep function approximation** — neural net learns features from raw input. Universal but hard to optimize (deadly triad).

---

#### ④ Q-learning + Deep Learning의 구체적 결합 / The concrete fusion

**한국어**
DQN의 아이디어를 수식으로 쓰면:

**(1) Q를 CNN으로 대체**:
$$Q(s, a; \theta) = \text{CNN}_\theta(s)[a]$$
- 입력: 게임 화면 이미지 $s$ (84×84×4, 4-frame stacking).
- 출력: **모든 action에 대한 Q 값 벡터** — $|A|$개 출력 (Atari는 보통 4~18개).
- 즉 한 번의 forward pass로 모든 action의 Q를 동시에 계산.

**(2) Bellman update → 미분 가능한 손실함수**:
테이블 업데이트 $Q \leftarrow Q + \alpha \delta$는 파라미터 공간에서 작동하지 않으므로, 이를 **MSE 손실의 기울기 하강**으로 재해석:
$$L(\theta) = \mathbb{E}\!\left[\left(\underbrace{r + \gamma \max_{a'} Q(s', a'; \theta)}_{\text{TD target (고정으로 취급)}} - Q(s, a; \theta)\right)^2\right]$$

$\theta$를 따라 경사하강:
$$\theta \leftarrow \theta - \alpha \nabla_\theta L(\theta)$$

**핵심 관찰**: 이 기울기 $\nabla_\theta L$이 정확히 TD error $\delta$와 동일한 역할을 한다. 테이블 업데이트 $Q(s,a) \leftarrow Q(s,a) + \alpha\delta$는 "특정 $(s,a)$ 항만 $\delta$만큼 올린다"는 뜻인데, 함수 근사에서는 **$\theta$ 전체를 $\delta \cdot \nabla_\theta Q(s,a;\theta)$ 만큼 움직여** 그 $(s,a)$에서의 Q가 $\delta$만큼 변하도록 한다. 그리고 덤으로 **비슷한 상태들의 Q도 함께 움직인다** — 이것이 generalization.

**(3) Backprop**: $\nabla_\theta Q$는 신경망의 체인 룰로 자동 계산. PyTorch의 `loss.backward()`가 이 일을 한다.

**(4) End-to-end**: 픽셀 → Conv1 → Conv2 → FC → Q values. 수작업 feature 없음. 네트워크가 스스로 "공의 위치", "적의 움직임" 같은 feature를 학습.

**English**
DQN's fusion in one picture:
1. Replace $Q(s,a)$ table with $Q(s,a;\theta)$ = CNN output indexed by action.
2. Replace tabular update with gradient descent on MSE loss $L(\theta) = \mathbb{E}[(r + \gamma \max_{a'} Q(s',a';\theta) - Q(s,a;\theta))^2]$.
3. Parameters updated via backprop.
4. No hand-crafted features — pixels → Q end-to-end.

Key insight: the gradient update touches not just $(s,a)$ but **all similar states simultaneously** — this is **generalization**, something a table cannot do.

---

#### ⑤ DQN 아키텍처 구체 / DQN's concrete architecture

**한국어**
논문 §5.2의 설계:
```
Input: 84 × 84 × 4 (4개의 grayscale 프레임 스택)
   │
   ▼
Conv1: 16 filters, 8×8 kernel, stride 4, ReLU
   │  출력: 20 × 20 × 16
   ▼
Conv2: 32 filters, 4×4 kernel, stride 2, ReLU
   │  출력: 9 × 9 × 32 = 2592
   ▼
Flatten → FC: 256 units, ReLU
   │
   ▼
Output FC: |A| units (action마다 하나의 Q 값)
```

총 파라미터는 약 1.5M. AlexNet(#13)의 ~60M에 비하면 작지만, Atari state의 통계 구조가 훨씬 단순해서 충분.

**왜 CNN인가**: state가 이미지이므로 (1) 공간적 locality (근처 픽셀이 의미적으로 연결), (2) translation invariance (공이 어디 있든 "공"), (3) 계층적 feature (edge → shape → object) — 이 세 가지 성질을 CNN이 자연스럽게 포착.

**English**
Architecture (§5.2): 84×84×4 input → 2 conv layers (16 @ 8×8 s4, 32 @ 4×4 s2) → FC 256 → output of $|A|$ Q-values. ~1.5M parameters. CNN is chosen because states are images with spatial locality, translation invariance, and hierarchical structure.

---

#### ⑥ "Deep"이 무엇을 더해주는가 / What "deep" adds

**한국어**
선형 function approximation과 비교해 "deep" function approximation이 제공하는 세 가지:

| 측면 / Aspect | Linear FA ($\theta^\top \phi$) | Deep FA (Neural Net) |
|---|---|---|
| **Feature 설계** | 사람이 수작업 | **네트워크가 학습** |
| **표현력** | $\phi$ 공간 내 선형 함수만 | **임의의 연속 함수** (Universal approx) |
| **Generalization** | $\phi$ 유사도 기반 | **픽셀 수준 유사도 → 의미 수준 유사도로 변환** |
| **태스크 전이** | 게임마다 $\phi$ 재설계 | 같은 구조로 여러 태스크 |
| **확장성** | 차원 저주에 민감 | 고차원에서도 작동 |

DQN이 7개 게임에서 **"같은 아키텍처, 같은 하이퍼파라미터"** 로 성공한 것이 바로 이 deep FA의 힘입니다 — **게임별 튜닝 없이 일반화**.

**English**
Deep FA over linear FA: (1) learns features instead of hand-designing, (2) universal approximation vs. linear-in-$\phi$, (3) semantic generalization (not just input similarity), (4) task transfer, (5) scales to high-dim. DQN's single-architecture, single-hyperparameter success on 7 games directly demonstrates this.

---

#### ⑦ 같은 논리가 오늘날 모든 RL에 적용된다 / The same template governs all modern RL

**한국어**
DQN이 세운 공식은 이후 모든 심층 RL에 적용됩니다:
- **AlphaGo/AlphaZero**: 바둑판 이미지 → CNN → value + policy. Q-learning 대신 policy gradient + MCTS, 하지만 "raw board → deep net → value" 패턴은 동일.
- **PPO (#24)**: state → MLP/CNN → action 확률. Policy gradient지만 같은 end-to-end 철학.
- **RLHF (GPT/Claude)**: token sequence → Transformer → reward + policy. State가 언어 시퀀스이고 function approximator가 Transformer인 것 외에는 DQN 구조와 동일.
- **Robot learning**: camera image → CNN/Transformer → motor commands.

즉 DQN은 "**고차원 raw input에서 RL 가치/정책을 deep net으로 근사한다**"는 공식의 **최초 성공 사례이자 템플릿**입니다.

**English**
Every modern deep RL system follows the DQN template — AlphaGo/AlphaZero, PPO, RLHF (GPT/Claude), robot learning — all replace the tabular value/policy function with a deep net trained end-to-end from raw input.

---

#### ⑧ 한 줄 정리 / One-line summary

**한국어**
- **Deep Learning**: 원시 입력에서 목표까지 **end-to-end로 학습되는 미분 가능한 다층 함수 근사기**.
- **Q-learning**: Bellman 방정식을 target으로 삼는 **tabular value iteration 알고리즘**.
- **DQN = Q-learning의 테이블 자리에 Deep Learning을 끼워넣은 것** + deadly triad를 달래는 experience replay.

이 한 줄이 DQN이 왜 "단순히 Q-learning에 NN 얹은 것"이 아니라 **새로운 학문 분야(deep RL)의 시작점**인 이유입니다.

**English**
- **Deep learning** = differentiable multi-layer function approximator trained end-to-end from raw input.
- **Q-learning** = tabular value-iteration with Bellman-target updates.
- **DQN = Q-learning with the table replaced by a deep net** + experience replay to tame the deadly triad.

This is why DQN is not "just Q-learning with a neural net" but the founding moment of a new field — **deep reinforcement learning**.

---

### Q3. AlphaGo는 바둑판을 "영상"으로 인식하는가? / Does AlphaGo treat the Go board as an image?

**본질적으로는 네, 맞습니다.** 더 정확히 말하면, 바둑판을 **19×19 해상도의 다채널 "이미지 텐서"** 로 보고 그것을 CNN으로 처리합니다. 다만 "영상"이라는 말이 주는 인상과는 몇 가지 차이가 있어 단계별로 풀어 설명합니다.

Essentially yes. More precisely, AlphaGo treats the board as a **multi-channel 19×19 "image tensor"** processed by a CNN. But there are some nuances worth unpacking.

---

#### ① 입력 텐서의 형태 / The input tensor shape

**한국어**
바둑판의 "픽셀"은 19×19 = 361개의 격자 교차점입니다. 각 교차점에 "검은 돌/흰 돌/빈 칸"이 있는데, 이걸 **여러 개의 binary 평면(plane)** 으로 쪼개서 **채널 축**에 배치합니다.

- AlphaGo (2016, Nature): 입력 shape = **19 × 19 × 48 channels**
- AlphaGo Zero (2017, Nature): **19 × 19 × 17 channels**

즉 AlphaGo의 입력은 `(height=19, width=19, channels=C)`인 **3차원 텐서** — RGB 이미지가 `(H, W, 3)`인 것과 **정확히 같은 구조**입니다. CNN의 입장에서는 바둑판이나 RGB 사진이나 똑같이 "다채널 2D 격자"일 뿐입니다.

**English**
A Go board has 361 intersections arranged in a 19×19 grid. Each intersection's content (black/white/empty) is split across **multiple binary planes** stacked along a channel axis:
- AlphaGo (2016 Nature): 19×19×**48 channels**
- AlphaGo Zero (2017 Nature): 19×19×**17 channels**

Structurally identical to an RGB image `(H, W, C)`. From the CNN's perspective, a Go board and a photograph are both "multi-channel 2D grids."

---

#### ② 채널에 무엇이 들어가는가 / What goes into the channels

**한국어**

**AlphaGo (2016) — 48개 채널** (Nature 논문 Table 2):
- 현재 흑돌 위치 (1 channel, binary 19×19)
- 현재 백돌 위치 (1 channel)
- 빈 칸 위치 (1 channel)
- 각 돌의 **liberty 수** (8 channels — one-hot)
- 돌이 놓인 지 얼마나 됐는지(turns since played, 8 channels)
- 바로 이전 몇 수에 놓인 돌들 위치
- 착수 금지(ko) 지점
- ... 총 48개

여기엔 사람이 만든 **약간의 도메인 feature**가 섞여 있습니다 (liberty 수 등). 완전한 "raw board"가 아님.

**AlphaGo Zero (2017) — 17개 채널** (Nature 논문, 더 깔끔한 버전):
- 현재 자기 돌 위치 + 최근 7수 전까지의 자기 돌 위치 (8 channels)
- 현재 상대 돌 위치 + 최근 7수 전까지의 상대 돌 위치 (8 channels)
- 현재 둘 차례 (1 channel, 전체가 0 또는 1)

**완전히 raw**. 사람이 설계한 feature 없이 **순수 반상 상태만**. 그리고 Zero는 인간 기보도 전혀 안 쓰고 자기 대국만으로 학습 — 이름의 "Zero"가 뜻하는 바.

**English**
- **AlphaGo (2016)**: 48 channels — stone positions + *some hand-engineered features* (liberty count, turns-since-played, ko, etc.).
- **AlphaGo Zero (2017)**: 17 channels — only raw stone positions (current + 7 history moves for each color, + whose turn). **No hand-engineered features, no human games**, hence "Zero."

---

#### ③ CNN이 바둑에 자연스러운 이유 / Why CNNs are natural for Go

**한국어**
사진과 바둑판이 구조적으로 같아도, "왜 하필 CNN이 적합한가?"는 별개의 질문입니다. 바둑에는 **이미지와 공유하는 세 가지 성질**이 있습니다:

1. **Spatial locality (공간적 국소성)**: 바둑의 전술적 모양(벽, 호구, 마늘모, 3-3 침입 등)은 대부분 **국소적 패턴**입니다. 멀리 떨어진 돌들이 조합되는 경우보다 인접한 돌들의 배치가 훨씬 중요. → CNN의 receptive field 구조와 잘 맞음.

2. **Translation equivariance (병진 등변성)**: 한 모서리에 만들어진 정석(joseki)은 다른 모서리에서도 비슷하게 유효합니다. 바둑판 **어디에 있든 같은 패턴은 같은 의미**. → CNN의 weight sharing이 이걸 그대로 인코딩. (다만 가장자리 효과 때문에 완벽한 불변은 아님 — 후속 연구에서 edge handling 개선.)

3. **Hierarchical structure (계층적 구조)**: 개별 돌 → 모양(shape) → 무리(group) → 집(territory)의 **계층**이 있습니다. CNN의 깊은 구조(edge → shape → object)가 이 계층과 자연스럽게 매핑.

바로 이 세 가지 성질 덕분에, **"바둑 프로기사의 직관적 패턴 인식"이 이미지 분류기와 수학적으로 유사한 문제**로 환원되는 것입니다. CNN이 AlphaGo에서 폭발적으로 잘 작동한 이유입니다.

**English**
Three properties shared by images and Go:
1. **Spatial locality** — tactical shapes (walls, hane, 3-3 invasion) are mostly local patterns → matches CNN receptive fields.
2. **Translation equivariance** — a joseki in one corner works similarly in another → CNN weight-sharing encodes this for free.
3. **Hierarchical structure** — stones → shapes → groups → territory → matches CNN depth (edges → shapes → objects).

These three properties make "professional Go intuition" mathematically similar to image recognition.

---

#### ④ 구체적 아키텍처 / Concrete architecture

**한국어**

**AlphaGo (2016)**:
- 13-layer CNN (policy network).
- Conv 3×3, 192 filters, padding 1. 끝에 softmax로 361+1 (pass 포함) move 확률.
- Value network는 비슷하지만 끝이 scalar (승률).

**AlphaGo Zero (2017)**:
- **ResNet 기반** (이 프로젝트의 논문 #20을 직접 인용). 19 또는 39 residual blocks, 각 block 2개 conv 3×3, 256 filters, BatchNorm + ReLU.
- **Two heads (공유 body, 분리된 머리)**:
  - **Policy head**: Conv → softmax → 362-dim 확률 벡터 (19×19 + pass)
  - **Value head**: Conv → FC → tanh → scalar 승률 $v \in [-1, 1]$

ResNet + 두 개의 head라는 설계는 **오늘날 바둑/체스/쇼기 엔진의 표준 템플릿**이 되었고, 최근의 KataGo, LeelaChessZero 등 오픈소스 엔진도 이 구조를 따릅니다.

**English**
- **AlphaGo (2016)**: 13-layer plain CNN with 192 3×3 filters.
- **AlphaGo Zero (2017)**: ResNet (20 or 40 blocks, 256 filters, BN + ReLU) with two heads sharing a body — **policy head** (softmax over 362 moves) and **value head** (tanh scalar). This "ResNet + two heads" design is now the standard template in KataGo, LeelaChessZero, etc.

---

#### ⑤ DQN과 무엇이 같고 무엇이 다른가 / What's the same as DQN, what's different

**한국어**

**같은 점 — "DQN의 템플릿"**:
- **Raw state → CNN → RL target**. 바둑판 텐서를 **수작업 feature 없이** (Zero의 경우) 직접 CNN에 입력.
- **End-to-end 학습**: 픽셀 수준에서 승률/정책까지 한 번에.
- **CNN의 세 성질** (locality, translation, hierarchy)을 활용.

**다른 점**:

| 측면 | DQN (Atari) | AlphaGo |
|---|---|---|
| State 공간 | 이미지 (84×84×4) | 바둑판 (19×19×C) |
| 시간축 | Frame stacking 4프레임 | 수순 history (Zero: 8수) |
| RL 알고리즘 | Q-learning (value-based) | Policy gradient + MCTS (actor-critic-like) |
| 출력 | Q 값 벡터 (action마다) | Policy + Value (dual head) |
| 학습 신호 | 환경 보상 (Atari 점수) | 승/패 결과 |
| Exploration | ε-greedy | MCTS 탐색 |
| 아키텍처 | 2-layer CNN | ResNet 20+ blocks |

즉 AlphaGo는 **"DQN의 설계 원리 + ResNet 깊이 + MCTS 탐색 + Policy/Value dual head"** 의 결합입니다. "raw state를 CNN으로" 라는 공통된 토대 위에 더 정교한 RL 방법(MCTS)과 더 깊은 아키텍처(ResNet)를 얹은 것.

**English**
Same as DQN: raw state → CNN → RL target, trained end-to-end, exploiting locality/translation/hierarchy. Different: Go board instead of frames, history instead of frame stack, MCTS + policy gradient instead of Q-learning, dual-head (policy + value), ResNet depth. AlphaGo = DQN's template + MCTS + ResNet + dual head.

---

#### ⑥ "영상"과의 중요한 차이 / Important differences from "images"

**한국어**
CNN이 처리한다는 점에서는 같지만, 바둑판은 **자연 영상과 구조적으로 다른 점**이 있습니다:

1. **채널의 의미**: RGB는 색 정보의 세 성분이지만, AlphaGo의 채널은 **"현재 흑돌", "현재 백돌", "7수 전 흑돌"** 등 **의미적으로 다른 평면**. 채널 간 관계가 색 공간과 다름.

2. **Discrete state**: 각 교차점은 3개 상태만 (흑/백/빈). Binary planes로 표현. 자연 영상의 연속값 픽셀과 대조.

3. **Sparse signal**: 게임 초반에는 대부분 빈 칸. 자연 영상은 모든 픽셀이 "정보"를 가짐.

4. **엄격한 경계 효과**: 19×19 격자의 가장자리는 "벽"으로 작용. CNN의 padding 처리가 중요.

5. **시간 방향이 본질적**: 자연 영상 분류는 한 장이면 충분. 바둑은 **수순 history**가 규칙(ko, superko 등)에 영향 → AlphaGo Zero가 8수 history 채널을 넣은 이유.

이런 차이에도 불구하고 CNN의 **귀납적 편향(inductive bias)** — 국소성, 병진 등변성, 계층성 — 이 바둑의 구조와 잘 맞기 때문에 "바둑판 = 영상"이라는 개념적 매핑이 성립하는 것입니다.

**English**
Structural differences from natural images: (1) channels carry *semantic* meaning (current black, current white, 7-moves-ago black) rather than color; (2) discrete state (3 values per intersection); (3) sparse signal (early game is mostly empty); (4) strict boundary effect on a 19×19 grid; (5) move history matters for rules (ko / superko) → AlphaGo Zero's 8-move history channels. Despite these, CNN's inductive bias (locality, translation equivariance, hierarchy) aligns well with Go's structure.

---

#### ⑦ 이 관점이 왜 중요한가 / Why this perspective matters

**한국어**
"바둑판 = 다채널 이미지" 관점이 중요한 이유:

- **일반화 가능성**: 체스판, 쇼기판, 연결-4, 틱택토, 오목 — 모두 같은 틀로 접근 가능. **AlphaZero (2017)** 가 이걸 정확히 증명 — **같은 아키텍처 (ResNet + dual head) 로 바둑·체스·쇼기 모두 세계 최강**.
- **비전 분야의 모든 기법 이식 가능**: Batch Norm, ResNet skip, 데이터 증강(대칭 회전), 어텐션 등이 전부 바둑 엔진에도 적용됨.
- **로봇/자율주행으로의 연결**: 로봇 주변 환경도 "카메라 이미지 + 추가 센서 채널"로 표현 → **같은 설계 원리로 물리 세계 문제에도 deep RL 적용 가능**.

DQN이 "raw image → deep net → RL"의 첫 사례였다면, AlphaGo는 그것이 **특정 도메인(Atari 게임)에 국한된 것이 아니라 구조적으로 격자/공간 상태를 갖는 모든 문제에 보편적으로 작동함**을 증명한 결정적 후속입니다.

**English**
The "board = multi-channel image" view generalizes to chess, shogi, connect-4, tic-tac-toe, gomoku — AlphaZero (2017) proved exactly this, using **identical architecture** (ResNet + dual head) to master Go, chess, and shogi simultaneously. It also opens the door to transplanting all computer-vision techniques (BN, ResNet, attention, augmentation via symmetries) into game engines, and eventually to robotics where cameras + additional sensor planes become the "image."

DQN was the first "raw image → deep net → RL" success; AlphaGo proved the template works beyond Atari, on any grid-structured domain.

---

#### ⑧ 한 줄 정리 / One-line summary

**한국어**
✅ **"AlphaGo는 바둑판을 19×19 해상도의 다채널 이미지 텐서로 보고, CNN(후기에는 ResNet)으로 처리한다."** 자연 영상과 채널의 의미는 다르지만, CNN의 귀납적 편향이 바둑의 공간적 구조와 잘 맞아서 이 매핑이 성립합니다. 이것이 DQN에서 AlphaGo로, 그리고 모든 현대 deep RL로 이어지는 **공통 설계 원리**입니다.

**English**
✅ **"AlphaGo treats the board as a multi-channel 19×19 image tensor and processes it with a CNN (later ResNet)."** Channels carry different semantic meaning from RGB, but CNN's inductive bias aligns with Go's spatial structure — making the image analogy exact at the architectural level. This is the shared design principle that connects DQN, AlphaGo, and all modern deep RL.

---

### Q4. 보상이 1/0이고 모델이 보상 받도록 업데이트되는 것인가? 보상이 손실함수로 작동하는가? 미분 가능한가? / Is reward just 1/0 and the model updates to receive it? Does reward act as the loss? Is it differentiable?

첫 질문은 **부분적으로 맞고**, 두 번째 질문은 명확히 **아니오** — 그리고 이 "아니오"가 RL이 supervised learning과 본질적으로 다른 이유입니다.

Q1 is **partially correct**; Q2 is a firm **no** — and this "no" is exactly what makes RL structurally different from supervised learning.

---

#### ① Q1 답 — 보상은 1/0이 아니라 일반 스칼라 / Reward is a general scalar, not 1/0

**한국어**
**맞는 부분**: "모델이 보상을 최대화하도록 업데이트된다" — 정확합니다.

**수정할 부분**: "보상 = 1/0" 은 특수 경우일 뿐입니다. 일반적으로 보상은 **실수값 스칼라 $r \in \mathbb{R}$**:

| 상황 | 보상 형태 |
|---|---|
| **AlphaGo (바둑)** | 에피소드 끝에만: 승 +1, 패 −1, 무 0 — 이진에 가까움 |
| **Atari (DQN)** | 게임 점수 변화 — +10, +200, −200 등 정수 |
| **Robot reaching** | $r = -\|\mathbf{x} - \mathbf{x}_{\text{goal}}\|$ — 연속값 |
| **자율주행** | 전진 +1, 충돌 −100, 연료 −0.01 — 혼합 |
| **RLHF (Claude 훈련)** | 인간 선호도 reward model 출력 — 연속 스칼라 |

**중요한 개념적 정정**: 모델은 단일 보상이 아니라 **누적 할인 보상**을 최대화:
$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$$

"지금부터 영원까지의 할인 합"을 최대화 → **장기 전략**이 학습됨.

**DQN의 reward clipping 주의**: 논문이 보상을 $\{-1, 0, +1\}$로 clipping한 것은 "보상이 이진"이라는 뜻이 아니라, 게임 간 점수 스케일 차이를 통일하기 위한 **공학적 트릭**입니다.

**English**
Correct: "the model updates to maximize reward." Wrong: "reward is 1/0." Generally reward is a **real scalar** $r \in \mathbb{R}$. And the agent maximizes **cumulative discounted reward** $G_t = \sum_k \gamma^k r_{t+k}$, not a single reward — that's how long-term strategy emerges. DQN's reward clipping to $\{-1, 0, +1\}$ is an engineering trick for cross-game scale normalization, not a statement that reward is binary.

---

#### ② Q2 답 — 보상은 loss가 아니고, 미분 불가능하다 / Reward is NOT the loss and is NOT differentiable

**한국어**
이것이 RL의 **가장 중요한 개념적 지점**입니다. 정답:

**보상은 손실함수가 아니며, $\theta$에 대해 미분 불가능합니다.**

**왜?**
보상은 **환경에서 나오는 관측값**:
$$r_t = \text{Env}(s_t, a_t)$$

환경은:
- **블랙박스**: Atari 에뮬레이터, 물리 시뮬레이터, 실세계 로봇
- **일반적으로 미분 불가능**: 게임 규칙, 충돌 감지, 인간 피드백은 불연속적/이산적
- **$\theta$의 함수가 아님**: 보상은 θ → π → a → r로 **간접 영향**만 받음
- **$\partial r / \partial \theta$를 backprop으로 계산할 방법 없음**

따라서 "reward가 높아지는 방향으로 θ를 미분" 은 **불가능**합니다. Supervised learning의 $L = \|y - \hat{y}\|^2$과 결정적으로 다른 점.

**English**
This is the most important conceptual point in RL. **Reward is NOT a loss, and is NOT differentiable with respect to $\theta$.** Rewards come from the environment — a generally non-differentiable black box (emulator, simulator, real world). You cannot compute $\partial r / \partial \theta$ via backprop. Unlike supervised learning, "differentiate the reward to maximize it" is impossible.

---

#### ③ 그럼 DQN의 loss는 무엇이며 기울기는 어떻게 흐르나 / Then what IS the loss, and how does the gradient flow?

**한국어**
DQN loss:
$$L(\theta) = \mathbb{E}\!\left[\Big(\underbrace{r + \gamma \max_{a'} Q(s', a'; \theta)}_{\text{TD target (상수로 취급)}} - Q(s, a; \theta)\Big)^2\right]$$

**핵심 관찰**: 이 loss에서 $r$은 **관측된 스칼라**로 **상수(constant)** 취급됩니다. 미분되지 않습니다. 미분되는 것은 오직 **신경망 $Q(s, a; \theta)$**.

**기울기 흐름**:
$$\nabla_\theta L = -2 \cdot \delta \cdot \nabla_\theta Q(s, a; \theta)$$
- $\delta = r + \gamma \max Q(s') - Q(s, a)$ : **스칼라 값**, 기울기의 **가중치**로만 작용
- $\nabla_\theta Q(s, a; \theta)$ : 일반적인 신경망 backprop
- 기울기는 **Q 네트워크를 타고 흐르지, 보상을 타고 흐르지 않음**

**시각화**:
```
Env → r (scalar, 상수)              ┐
                                      ├─ 합쳐서 TD target (스칼라, 상수 취급)
Q_θ(s', a') → max → (상수 취급)     ┘          │
                                                 ▼
Q_θ(s, a) ─────────────────────────► [target − Q(s,a)]² = L
   │                                             │
   └────── 기울기 ◄── ∂L/∂θ ── backprop ─────────┘
```

**English**
DQN loss = TD error squared, where $r$ is treated as a **constant scalar** (an observation). Only $Q(s,a;\theta)$ is differentiated. Gradient flows **through the Q network, not through the reward**. The reward is a *material* used to build the loss — not the target of differentiation.

---

#### ④ 그럼 어떻게 "보상을 받도록" 학습되는가 / So how does the agent actually learn to get rewards?

**한국어**
직접 reward를 미분할 수 없지만, **Bellman 방정식이 간접 경로를 제공**합니다:

1. **Bellman 방정식**: $Q^*$는 $r + \gamma \max Q^*(s')$를 만족해야 함 (자기 일관성).
2. **DQN loss는 Q가 이 방정식을 만족하도록 강제**: MSE를 최소화하면 $Q \to Q^*$.
3. **$Q^*$를 알면 최적 정책은** $a^* = \arg\max_a Q^*(s, a)$.
4. **최적 정책 = 기대 누적 보상 최대화**.

즉 전략은:
> **"reward를 직접 미분"이 아니라, "Q가 Bellman target과 일치하도록 훈련 → 그 결과 최적 정책이 얻어진다"** 는 **간접 최적화**.

보상은 loss를 **구성하는 재료**일 뿐, 미분의 대상이 아닙니다. 이것이 Watkins의 Q-learning이 천재적인 이유입니다 — 환경의 미분 가능성 없이도 최적 정책을 간접적으로 찾을 수 있게 해 줍니다.

**English**
We cannot differentiate reward, but the **Bellman equation gives an indirect route**:
1. $Q^*$ must satisfy $r + \gamma \max Q^*(s')$ (self-consistency).
2. DQN loss forces $Q$ toward this equation.
3. Knowing $Q^*$ gives optimal policy $a^* = \arg\max_a Q^*(s,a)$.
4. Optimal policy maximizes expected cumulative reward.

So the strategy is **indirect optimization**: not "differentiate the reward" but "make $Q$ satisfy Bellman's equation, and optimal behavior follows."

---

#### ⑤ Policy Gradient에서도 같은 구조 / Same structure in Policy Gradient

**한국어**
REINFORCE/PPO의 loss:
$$L(\theta) = -\mathbb{E}\!\left[ R \cdot \log \pi_\theta(a \mid s) \right]$$

여기서도 $R$(누적 보상)은 **상수 가중치**. 기울기는 오직 $\log \pi_\theta$를 타고 흐름:
$$\nabla_\theta L = -\mathbb{E}[R \cdot \nabla_\theta \log \pi_\theta(a \mid s)]$$

Score function trick(log-derivative trick): "**환경을 미분하지 않고도** 기대 보상의 기울기를 추정"하는 policy gradient의 핵심 아이디어. Q-learning과 다른 접근이지만 **"보상은 상수"라는 원리는 동일**.

**English**
Policy gradient (REINFORCE/PPO) loss: $L = -\mathbb{E}[R \log \pi_\theta]$. Here too, $R$ is a **constant weight**; gradient flows only through $\log \pi_\theta$. The score-function trick estimates $\nabla_\theta \mathbb{E}[R]$ without differentiating the environment — different route than Q-learning, same "reward is constant" principle.

---

#### ⑥ Supervised Learning vs RL 비교표 / SL vs RL comparison

| | Supervised Learning | Reinforcement Learning |
|---|---|---|
| **Target 출처** | 데이터셋의 정답 label | **환경의 보상 신호** (블랙박스) |
| **Loss** | $\|y - \hat{y}\|^2$ (직접) | $\text{(TD target} - Q)^2$ (**간접 재구성**) |
| **Target 안정성** | 고정 (ground truth) | **매 step 움직임** (bootstrapping의 non-stationarity) |
| **Target 미분 가능?** | 상수 취급 ✗ | 상수 취급 ✗ |
| **모델 미분 가능?** | ✓ | ✓ |
| **환경 미분 가능?** | 필요 없음 | **일반적으로 ✗** (문제의 근원) |
| **Exploration** | 필요 없음 (모든 샘플 주어짐) | **필수** (어떤 경험이 가치 있는지 모름) |
| **Sample efficiency** | 보통 높음 | 보통 낮음 (환경과 상호작용 비싸다) |

RL이 어려운 근본 이유: **환경이 미분 불가능한 블랙박스** → 보상 신호를 직접 미분해 최적화 불가 → Bellman/score function 같은 **우회 전략** 필요. 그리고 Bellman의 self-referential 구조가 **deadly triad**(보상 자체와는 무관한 새 난제)를 만들어냄.

**English**
RL's fundamental difficulty: the environment is a non-differentiable black box → cannot directly differentiate reward → need detours like Bellman equation or score function trick → Bellman's self-referential structure creates the deadly triad (a new problem unrelated to reward per se).

---

#### ⑦ 흥미로운 예외: differentiable simulator / Interesting exception

**한국어**
최근 연구에서는 **환경 자체가 미분 가능한** 특수 경우를 탐구합니다:
- **Differentiable physics simulator** (Brax, MuJoCo with gradient, DiffTaichi): 물리 시뮬레이터를 PyTorch/JAX로 구현 → $\partial r / \partial a$ 직접 계산 가능.
- 이 경우에는 supervised learning처럼 보상을 직접 미분해 최적화 가능 → 훨씬 sample-efficient.
- 한계: 실세계 물리, 게임 규칙, 인간 피드백은 일반적으로 미분 불가. 따라서 대부분의 실용 RL은 여전히 전통적 우회 전략을 씀.

**English**
Recent work explores **differentiable simulators** (Brax, DiffTaichi) where the environment *is* differentiable — then reward can be directly differentiated and optimized like supervised learning. Much more sample-efficient, but limited to engineered simulators; real-world physics, game rules, and human feedback remain non-differentiable.

---

#### ⑧ 한 줄 정리 / One-line summary

**한국어**
- ✅ 모델이 **누적 할인 보상**을 최대화하도록 업데이트됨. (Q1의 "최대화" 부분 정확)
- ❌ 보상은 1/0이 아니라 일반 실수 스칼라. 이진은 특수 경우.
- ❌ 보상 자체는 **loss가 아니며 θ에 대해 미분 불가능**. 환경은 블랙박스.
- ✅ Loss는 **보상을 재료로 구성된** 별도의 미분 가능 함수 (TD error², policy score 등).
- ✅ **기울기는 Q/π 네트워크로만** 흐르고, 보상은 **상수 가중치**로 들어감.

이 한 문단을 이해하면 RL이 왜 supervised learning과 구조적으로 다른지, 왜 DQN에 experience replay 같은 트릭이 필요한지가 명확해집니다.

**English**
- ✅ Model updates to maximize **cumulative discounted** reward.
- ❌ Reward is a real scalar, not 1/0 (binary is a special case).
- ❌ Reward itself is **not the loss, not differentiable** in $\theta$; environment is a black box.
- ✅ Loss is a separate differentiable function **built from reward** (TD error², policy score, etc.).
- ✅ **Gradient flows only through the Q or π network**; reward enters as a constant weight.

Grasping this paragraph makes it clear why RL differs structurally from supervised learning, and why DQN needs tricks like experience replay.

---

### Q5. Bellman 방정식과 "target"에 대한 자세한 설명 / Bellman equation and "target" in depth

RL의 모든 것이 여기서 나옵니다. 천천히, 수학적 정의부터 실용 구현까지 단계별로 봅시다.

Everything in RL flows from here. Step by step from mathematical definition to practical implementation.

---

#### ① Value function 복습 — $V$와 $Q$ 두 가지 / Recap: two value functions

**한국어**
두 종류의 가치 함수를 구분해야 합니다:

**State value function** $V^\pi(s)$: "정책 $\pi$를 따를 때, 상태 $s$에서 시작한 기대 누적 할인 보상"
$$V^\pi(s) = \mathbb{E}_\pi\!\left[ G_t \;\Big|\; s_t = s \right] = \mathbb{E}_\pi\!\left[ \sum_{k=0}^{\infty} \gamma^k r_{t+k} \;\Big|\; s_t = s \right]$$

**Action value function** $Q^\pi(s, a)$: "정책 $\pi$를 따를 때, 상태 $s$에서 **먼저 행동 $a$를 취한** 뒤의 기대 누적 할인 보상"
$$Q^\pi(s, a) = \mathbb{E}_\pi\!\left[ G_t \;\Big|\; s_t = s, a_t = a \right]$$

두 관계:
$$V^\pi(s) = \mathbb{E}_{a \sim \pi}[Q^\pi(s, a)] = \sum_a \pi(a|s) Q^\pi(s, a)$$

**최적 버전**:
$$V^*(s) = \max_\pi V^\pi(s), \qquad Q^*(s, a) = \max_\pi Q^\pi(s, a)$$

Q-learning은 **$Q^*$를 직접 추정**하는 것이 목표. 왜? $Q^*$를 알면 $\pi^*$는 공짜로 얻어짐: $\pi^*(s) = \arg\max_a Q^*(s, a)$.

**English**
Two value functions:
- **$V^\pi(s)$**: expected cumulative discounted reward starting from state $s$ under policy $\pi$.
- **$Q^\pi(s, a)$**: same, but starting with a specific action $a$, then following $\pi$.
- Relation: $V^\pi(s) = \mathbb{E}_{a \sim \pi}[Q^\pi(s, a)]$.
- Optimal: $V^*$, $Q^*$.
- Knowing $Q^*$ → optimal policy $\pi^*(s) = \arg\max_a Q^*(s,a)$ for free.

---

#### ② Bellman Expectation Equation — 정책 $\pi$ 고정 / Bellman expectation (fixed policy)

**한국어**
$V^\pi$의 정의를 보면 $G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots$ 의 기대값입니다. 첫 항과 나머지를 분리:
$$G_t = r_t + \gamma \big( r_{t+1} + \gamma r_{t+2} + \dots \big) = r_t + \gamma G_{t+1}$$

이 **재귀적 분해**를 기대값 안에 넣으면:
$$V^\pi(s) = \mathbb{E}_\pi[G_t | s_t = s] = \mathbb{E}_\pi[r_t + \gamma G_{t+1} | s_t = s]$$

환경의 전이와 정책에 대해 전개:
$$\boxed{\;V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)\!\left[ r(s,a,s') + \gamma V^\pi(s') \right]\;}$$

이것이 **Bellman expectation equation** ($V$ 버전). $Q$ 버전도 마찬가지:
$$\boxed{\;Q^\pi(s, a) = \sum_{s'} P(s'|s,a)\!\left[ r(s,a,s') + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s', a') \right]\;}$$

**의미**: "어느 상태의 가치 = (즉시 보상) + $\gamma \times$ (다음 상태의 가치의 기대)". 미래 전체를 한 발짝 기대값으로 **자기 참조적으로 압축**.

**English**
Starting from the definition, use $G_t = r_t + \gamma G_{t+1}$ to unroll one step inside the expectation. This gives the **Bellman expectation equation** — a self-referential recursion "value here = immediate reward + $\gamma$ × expected value at next state." Two forms: for $V^\pi$ and for $Q^\pi$.

---

#### ③ Bellman Optimality Equation — 최적 정책 / Bellman optimality

**한국어**
최적 정책 $\pi^*$는 각 상태에서 가장 높은 Q를 주는 행동을 deterministic하게 선택:
$$\pi^*(s) = \arg\max_a Q^*(s, a)$$

그러면 $\pi^*$에 대한 expectation은 $\max$로 바뀌고:

$$\boxed{\;V^*(s) = \max_a \sum_{s'} P(s'|s,a)\!\left[ r + \gamma V^*(s') \right]\;}$$

$$\boxed{\;Q^*(s, a) = \sum_{s'} P(s'|s,a)\!\left[ r + \gamma \max_{a'} Q^*(s', a') \right]\;}$$

이것이 **Bellman optimality equation**. Q-learning과 DQN이 사용하는 방정식.

**구조적 핵심**: 두 개의 self-consistent 방정식 — expectation과 optimality — 이 RL의 두 가지 주 패러다임을 결정합니다:
- **Policy iteration / Actor-Critic**: expectation 쓰기 — 현재 정책 평가 후 개선
- **Q-learning / Value iteration**: optimality 쓰기 — 바로 $Q^*$를 추적

**English**
For the optimal policy $\pi^*(s) = \arg\max_a Q^*(s,a)$, the expectation over $a$ becomes a $\max$. Two equations: $V^*(s) = \max_a \sum_{s'} P(\cdot)[r + \gamma V^*(s')]$ and $Q^*(s,a) = \sum_{s'} P(\cdot)[r + \gamma \max_{a'} Q^*(s',a')]$. These two self-consistency equations split RL into two paradigms: expectation-based (policy iteration, actor-critic) and optimality-based (Q-learning, value iteration).

---

#### ④ 왜 Bellman이 "유일한 해"를 갖는가 / Why Bellman has a unique solution

**한국어**
Bellman optimality equation을 연산자 $\mathcal{T}$로 쓰면:
$$(\mathcal{T} Q)(s, a) = \sum_{s'} P(s'|s,a)\!\left[ r + \gamma \max_{a'} Q(s', a') \right]$$

$Q^*$는 **고정점**: $\mathcal{T} Q^* = Q^*$.

**핵심 정리 (Contraction Mapping / Banach fixed point theorem)**: $\gamma \in [0, 1)$일 때 $\mathcal{T}$는 sup-norm에서 **$\gamma$-contraction**:
$$\|\mathcal{T} Q_1 - \mathcal{T} Q_2\|_\infty \le \gamma \|Q_1 - Q_2\|_\infty$$

이 성질 덕분에:
1. **유일한 고정점 존재** — $Q^*$가 유일.
2. **어떤 초깃값에서 시작해도 반복하면 $Q^*$로 수렴** — $Q_{i+1} = \mathcal{T} Q_i \to Q^*$. 이것이 **Value iteration**.
3. **수렴 속도는 $\gamma^k$로 지수적**.

즉 Bellman 방정식은 단순한 "정의"가 아니라 **수학적으로 풀 수 있는 구조**이고, 이것이 RL 알고리즘 수렴 이론의 토대입니다.

**English**
Define the Bellman operator $\mathcal{T}$; then $Q^*$ is its fixed point: $\mathcal{T} Q^* = Q^*$. Key theorem: $\mathcal{T}$ is a **$\gamma$-contraction** in sup-norm, so by Banach fixed-point theorem, $Q^*$ is unique and iterating $Q_{i+1} = \mathcal{T} Q_i$ converges exponentially (rate $\gamma^k$) from any initialization — this is **value iteration**. Bellman isn't just a definition; it's a mathematically solvable structure.

---

#### ⑤ "Target" 이란 무엇인가 — 세 종류 / What is a "target"? Three flavors

**한국어**
"Target" = **"$Q(s, a)$가 그쪽으로 이동해야 할 목표 값"**. 실제 알고리즘에서는 $Q$를 매번 업데이트할 때 "어떤 값을 target으로 쓸 것인가" 선택해야 합니다. 세 가지 주요 선택지:

**(A) Monte Carlo target (에피소드 끝까지)**
에피소드가 끝날 때까지 **실제로 받은 보상의 합**:
$$y^{\text{MC}} = G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots + \gamma^{T-t} r_T$$

- 장점: **편향 없음** (실제 환경에서 샘플된 정답). $\mathbb{E}[G_t] = V^\pi(s_t)$.
- 단점: **분산 큼** (에피소드마다 들쭉날쭉). 에피소드 끝까지 기다려야 함.

**(B) TD(0) target — Bellman target — Bootstrapping**
**딱 한 step만** 실제 보상을 보고, 그 다음은 현재 $Q$ 추정값으로 대체:
$$y^{\text{TD}} = r_t + \gamma \max_{a'} Q(s_{t+1}, a')$$

- 장점: **분산 작음**, 매 step 즉시 업데이트 가능.
- 단점: $Q$ 추정 자체가 부정확하면 target도 부정확 → **편향 있음** ($Q$의 오차가 target으로 전파).

**(C) n-step target — 중간 지점**
$n$ step은 실제 보상, 그 이후는 bootstrap:
$$y^{(n)} = r_t + \gamma r_{t+1} + \dots + \gamma^{n-1} r_{t+n-1} + \gamma^n \max_{a'} Q(s_{t+n}, a')$$

- $n=1$: TD(0), $n=\infty$: Monte Carlo. 중간값 $n$이 많은 실용 알고리즘(A3C, Rainbow, MuZero)에서 쓰임.

**DQN은 TD(0) target 사용**. 이유: 매 step 업데이트, Atari의 긴 에피소드, replay buffer 설계.

**English**
Three target choices:
- **MC target**: $y^{MC} = G_t$ (actual return). Unbiased but high variance.
- **TD(0) target** (Bellman target): $y^{TD} = r + \gamma \max Q(s')$. Low variance but biased by current $Q$ estimate (bootstrapping).
- **n-step target**: interpolates. $n=1$ is TD, $n=\infty$ is MC.
- DQN uses TD(0) for per-step updates and replay-buffer compatibility.

---

#### ⑥ Bias-Variance Trade-off / 편향-분산 상충

**한국어**

| | Bias | Variance | Wait time |
|---|---|---|---|
| **MC** | 0 (unbiased) | High | Episode end |
| **TD(0)** | Possibly large (bootstrap error) | Low | Per step |
| **n-step** | Middle | Middle | n steps |

이것이 RL 알고리즘 설계에서 가장 고전적인 trade-off입니다. 환경이 **episodic & short**이면 MC가 좋고, **long-horizon & online**이면 TD가 낫습니다. Atari는 긴 에피소드 + 매 step 학습이 필요 → TD가 선택됨.

**English**
Classic trade-off: MC is unbiased but high-variance and requires episode end; TD is biased but low-variance and supports online updates; n-step interpolates. DQN (long Atari episodes, online learning) chose TD.

---

#### ⑦ DQN에서 target의 구체적 용법 / Concrete use of target in DQN

**한국어**
DQN loss:
$$L(\theta) = \mathbb{E}\!\left[\Big(\underbrace{r + \gamma \max_{a'} Q(s', a'; \theta)}_{y = \text{TD target}} - Q(s, a; \theta)\Big)^2\right]$$

**결정적 구현 세부**: target $y$를 계산할 때는 "**stop gradient**" — θ에 대해 미분하지 **않음**. Python/PyTorch 코드로:
```python
with torch.no_grad():
    y = r + gamma * Q_net(s_next).max(dim=-1).values
loss = F.mse_loss(Q_net(s)[a], y)
```

**왜 stop gradient?**
- Bellman optimality는 "$Q$가 target과 같아야 한다"는 **equation**이지, $Q$와 target이 **같은 값** 이라는 뜻이 아닙니다.
- 만약 target도 미분하면: $Q$가 target을 따라오고, target도 $Q$를 따라감 → **circular, 발산 가능**.
- Target을 "현재 알고 있는 최선의 추정"으로 **고정**해서 $Q$만 그쪽으로 당기는 것이 올바른 최적화.

이 stop-gradient는 RL 구현의 가장 흔한 실수 지점입니다. `torch.no_grad()`, `.detach()`, `tf.stop_gradient()` 등으로 명시적으로 처리해야 합니다.

**English**
Critical implementation detail: the target $y$ must be **stop-gradient** — do not differentiate through it. Bellman optimality is an *equation*, not a statement that $Q$ and target share parameters — differentiating both would be circular and could diverge. Fix target as "current best estimate," pull $Q$ toward it.

---

#### ⑧ Target Network (2015 Nature DQN 개선) / Target network

**한국어**
2013 논문 (본 논문)에서는 $L$의 target에 **현재 네트워크 θ**를 그대로 씁니다. 문제: $Q$가 한 step 업데이트되면 target도 즉시 바뀜 → target이 **non-stationary** → 불안정.

**2015 Nature DQN의 해결책**: 별도의 target network $Q_{\theta^-}$를 유지하고, 매 $C$ step(보통 10000)마다 $\theta^- \leftarrow \theta$로 복사:
$$L(\theta) = \mathbb{E}\!\left[\Big(r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta)\Big)^2\right]$$

효과:
- Target을 **$C$ step 동안 고정** → supervised learning처럼 "정답"이 안정적.
- $Q$가 target에 수렴한 다음 target을 업데이트 → 더 안정적 진행.
- 이 단순한 개선만으로 Atari 성능이 극적으로 향상 (Nature 논문).

Soft-update 변형(DDPG, SAC): $\theta^- \leftarrow \tau \theta + (1-\tau)\theta^-$ with $\tau \ll 1$ — 연속적 업데이트.

**English**
2013 paper uses current $\theta$ in the target — target shifts every step, causing instability. 2015 Nature fix: maintain a **target network $Q_{\theta^-}$**, copied from $\theta$ every $C$ steps (typically 10000). Target stays fixed for $C$ steps — much more stable. Soft-update variants (DDPG, SAC) use $\theta^- \leftarrow \tau\theta + (1-\tau)\theta^-$ for smooth tracking.

---

#### ⑨ 구체적 수치 예시 / Concrete numerical example

**한국어**
단순한 1D 환경: 상태 S0 → S1 → S2 → Goal. 각 step reward = 0, Goal에서 $r=+1$. $\gamma = 0.9$.

초기 $Q$ = 0 (모든 상태·행동).

**Step 1** (S2 → Goal, $r=1$):
- TD target: $y = 1 + 0.9 \cdot \max Q(\text{Goal}, \cdot) = 1 + 0 = 1$
- $Q(S_2, \text{right})$를 1 방향으로 당김. α=0.1이라면 $Q(S_2, \text{right}) \leftarrow 0 + 0.1(1 - 0) = 0.1$.

**Step 2** (S1 → S2, $r=0$):
- TD target: $y = 0 + 0.9 \cdot \max Q(S_2, \cdot) = 0.9 \cdot 0.1 = 0.09$
- $Q(S_1, \text{right}) \leftarrow 0 + 0.1(0.09 - 0) = 0.009$.

**Step 3** (S0 → S1, $r=0$):
- $y = 0 + 0.9 \cdot 0.009 = 0.0081$
- $Q(S_0, \text{right}) \leftarrow 0.00081$.

**관찰**: Goal의 reward가 S2 → S1 → S0로 **거꾸로 전파**됩니다. 한 epoch이 끝나면 $Q(S_2) > Q(S_1) > Q(S_0)$ — 정확한 계층을 학습. 이걸 수천 번 반복하면 수렴값:
- $Q(S_2, \text{right}) \to 1.0$
- $Q(S_1, \text{right}) \to 0.9$
- $Q(S_0, \text{right}) \to 0.81$

**이것이 Bellman target이 작동하는 방식입니다** — 실제 보상이 **Bellman 재귀를 타고** 공간(상태)을 따라 흘러 최종 가치 landscape를 만듭니다.

**English**
Walk through a 4-state chain S0→S1→S2→Goal with $r=+1$ at Goal, $\gamma=0.9$. After TD updates propagate backward, $Q(S_2, \text{right}) \to 1.0$, $Q(S_1) \to 0.9$, $Q(S_0) \to 0.81$. **Reward flows backward through the state space via the Bellman recursion.** This is what Bellman targets do in practice.

---

#### ⑩ 한 줄 요약 / One-line summary

**한국어**
- **Bellman 방정식** = "$V$ 또는 $Q$가 만족해야 할 자기 일관성 재귀" — optimality 버전은 $\max$로 정의됨.
- **Target** = "현재 $Q$가 그쪽으로 이동해야 할 값". MC(실제 reward 합), TD(Bellman 한 step), n-step(사이값) 세 가지.
- **DQN의 Bellman target**: $y = r + \gamma \max Q(s'; \theta)$. 반드시 **stop-gradient**로 처리.
- **Target network** (2015): target을 느리게 업데이트해 안정화.
- **Bellman contraction 성질** ($\gamma < 1$): 유일 고정점 존재 + 지수 수렴 → RL 알고리즘의 수학적 토대.

핵심 통찰: **Bellman은 RL을 "fixed-point 찾기" 문제로 바꾸고, target은 그 fixed-point를 향한 "나침반"** 입니다. DQN은 이 나침반을 deep net으로 대체 가능하게 만들었고, deadly triad를 달래는 experience replay와 target network가 이 대체를 안정화했습니다.

**English**
- **Bellman equation** = the self-consistency recursion $V$ or $Q$ must satisfy; optimality form uses $\max$.
- **Target** = the value we pull $Q$ toward. Three flavors: MC, TD, n-step.
- **DQN's Bellman target**: $y = r + \gamma \max Q(s'; \theta)$, must be **stop-gradient**.
- **Target network** (2015): slow target updates stabilize training.
- **Bellman contraction** ($\gamma < 1$): unique fixed point + exponential convergence — the mathematical foundation of RL.

Key insight: **Bellman turns RL into a fixed-point problem; targets are compasses pointing toward that fixed point.** DQN made the compass representable by a deep net; experience replay and target network stabilized the substitution.
