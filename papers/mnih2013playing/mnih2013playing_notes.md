---
title: "Playing Atari with Deep Reinforcement Learning"
authors: Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, Martin Riedmiller
year: 2013
journal: "arXiv preprint (NeurIPS 2013 Deep Learning Workshop); Nature follow-up in 2015"
doi: "arXiv:1312.5602"
topic: Artificial Intelligence / Deep Reinforcement Learning
tags: [dqn, deep-reinforcement-learning, q-learning, experience-replay, atari, cnn, deadly-triad, td-learning, value-based-rl]
status: completed
date_started: 2026-04-19
date_completed: 2026-04-19
---

# 22. Playing Atari with Deep Reinforcement Learning (DQN) / 딥 강화학습을 이용한 Atari 게임 플레이

---

## 1. Core Contribution / 핵심 기여

DQN은 **강화학습(RL)과 딥러닝을 최초로 성공적으로 결합**한 연구로, 게임 화면 픽셀만 입력으로 받아(수작업 feature 없음) 7개 Atari 2600 게임에서 인간 수준 혹은 그 이상의 플레이를 학습했다. 핵심 문제는 **"deadly triad"** — function approximation + bootstrapping + off-policy의 조합이 이론적으로 발산을 일으킬 수 있다는 고전적 난제였다. 본 논문은 두 가지 공학적 기법으로 이 문제를 달랜다: (1) **Experience Replay** — 과거 transition $(s, a, r, s')$을 버퍼에 저장하고 무작위로 mini-batch sampling하여 연속된 샘플의 상관관계를 깨고 i.i.d. 근사를 회복, (2) **CNN-based Q-function** — 84×84×4 (4-frame stacked grayscale) 이미지를 입력받아 2-layer conv + 2-layer FC로 모든 action의 Q 값을 한 번에 출력. 같은 아키텍처와 하이퍼파라미터로 7개 Atari 게임 중 6개에서 기존 방법을 능가, 3개(Breakout, Enduro, Pong)에서 인간 전문가를 능가했다. 이 결과는 **"범용 에이전트(general agent)"** 개념의 첫 실증이며, 2015년 Nature 후속 논문(Target Network 추가로 49개 게임 인간 수준)으로 이어져 AlphaGo, AlphaZero, MuZero, RLHF(ChatGPT/Claude), 현대 로봇 학습 등 **모든 심층 강화학습의 출발점**이 되었다.

DQN is the first work to **successfully combine reinforcement learning with deep learning**, achieving human-level or superhuman play on seven Atari 2600 games directly from raw pixels — with no hand-crafted features. The core problem was the classic **deadly triad** — the theoretical instability arising from function approximation + bootstrapping + off-policy learning. The paper introduces two engineering fixes: (1) **Experience Replay** — store past transitions $(s, a, r, s')$ in a buffer and sample random mini-batches, breaking sample correlations and restoring the i.i.d. approximation; (2) **CNN Q-function** — a network taking 84×84×4 (4-frame stacked grayscale) input through 2 conv + 2 FC layers to output Q-values for all actions simultaneously. With a single architecture and hyperparameter configuration, DQN surpasses prior methods in 6 of 7 games and matches or exceeds human experts in 3 (Breakout, Enduro, Pong). The paper provides the first demonstration of a **"general agent"**, extended in the 2015 Nature follow-up (Target Network, 49 Atari games at human level) — becoming the foundation of AlphaGo, AlphaZero, MuZero, RLHF in LLMs (ChatGPT/Claude), and modern robot learning. DQN is the birthplace of **deep reinforcement learning**.

---

## 2. Reading Notes / 읽기 노트

### Section 1: Introduction / 도입

**세 가지 장애물**
딥러닝을 RL에 적용하는 데 있어 세 가지 근본적 문제:

1. **데이터 특성 차이**: 딥러닝은 대량의 *레이블된 훈련 데이터*를 요구하지만, RL 알고리즘은 *sparse, noisy, delayed* 보상 신호만 받는다. 행동과 그로 인한 보상 사이의 지연 — credit assignment 문제 — 이 수천 step 수준으로 발생 가능.

2. **i.i.d. 가정 위반**: 딥러닝은 샘플이 서로 독립 동일분포라고 가정하지만, RL에서는 연속된 state들이 강하게 상관관계.

3. **Non-stationary distribution**: RL 에이전트의 정책이 바뀌면 만나는 데이터 분포도 바뀜. 딥러닝의 fixed-distribution 가정과 충돌.

**DQN의 해결책**:
- **CNN + Q-learning variant + SGD + experience replay**의 결합.
- 같은 아키텍처/하이퍼파라미터로 7개 게임.

**Three obstacles** in applying deep learning to RL:
1. Data nature mismatch — DL wants labeled data, RL gets sparse/noisy/delayed rewards.
2. i.i.d. assumption violated — sequential states are strongly correlated.
3. Non-stationary distribution — policy change alters the data distribution.

**DQN's answer**: CNN + a Q-learning variant + SGD + experience replay. Single architecture, single hyperparameter set across games.

### Section 2: Background / 배경

MDP의 표준 정식화:
- Agent가 환경과 상호작용, 매 step에서 action $a_t$를 선택.
- 환경이 reward $r_t$와 다음 관측 $x_{t+1}$을 반환.
- **Sequence** $s_t = x_1, a_1, x_2, \dots, a_{t-1}, x_t$를 state로 간주 (Atari는 부분 관측이므로 history 자체가 state).

**최적 action-value function**:
$$Q^*(s, a) = \max_\pi \mathbb{E}\!\left[R_t \mid s_t = s, a_t = a, \pi\right]$$

**Bellman optimality**:
$$Q^*(s, a) = \mathbb{E}_{s'}\!\left[r + \gamma \max_{a'} Q^*(s', a') \mid s, a\right]$$

Value iteration: $Q_{i+1}(s,a) = \mathbb{E}[r + \gamma \max_{a'} Q_i(s', a') | s, a]$는 $i \to \infty$에서 $Q^*$로 수렴 — 하지만 테이블 기반이라 실용적이지 않음.

**Function approximation**: $Q(s, a; \theta) \approx Q^*(s, a)$. 선형 함수가 일반적이었지만, 본 논문은 **neural network = Q-network**.

**Q-network training**: sequence of loss functions $L_i(\theta_i)$를 최소화:
$$L_i(\theta_i) = \mathbb{E}_{s, a \sim \rho(\cdot)}\!\left[\left(y_i - Q(s, a; \theta_i)\right)^2\right]$$
여기서 target $y_i = \mathbb{E}_{s'}[r + \gamma \max_{a'} Q(s', a'; \theta_{i-1}) | s, a]$.

**Gradient**:
$$\nabla_{\theta_i} L_i = \mathbb{E}\!\left[\left(r + \gamma \max_{a'} Q(s', a'; \theta_{i-1}) - Q(s, a; \theta_i)\right) \nabla_{\theta_i} Q(s, a; \theta_i)\right]$$

실제로는 전체 기대값 계산이 비싸므로 **single sample에 대한 stochastic gradient descent** 사용. 또한 Q-learning은 **off-policy** — 행동은 ε-greedy로 수집하지만 target은 greedy 정책 기준.

Standard MDP framework. Optimal Q satisfies Bellman optimality. Value iteration converges but is tabular. Using $Q(s,a;\theta) \approx Q^*$, loss $L_i(\theta_i) = \mathbb{E}[(y_i - Q(s,a;\theta_i))^2]$ with target $y_i$ using previous parameters. SGD + off-policy (ε-greedy behavior, greedy target).

### Section 3: Related Work / 관련 연구

**TD-Gammon** (Tesauro 1992, 1995): backgammon에 neural net을 사용한 고전. On-policy, self-play, hand-crafted features. 20년간 이 성공이 *다른* 게임으로 일반화되지 못했음 — chess, Go, checkers 모두 실패.

**이 실패의 이유(당시 추측)**:
1. Backgammon의 주사위 요소가 **stochasticity를 보존**하여 state space를 탐색시켜 줌.
2. 다른 게임은 deterministic → 국소 최적 정책에 갇힘.

**NFQ (Neural Fitted Q-iteration)** (Riedmiller 2005): experience replay를 사용한 선행 연구. 하지만 **batch gradient descent** 사용 → 큰 데이터셋에서는 너무 비쌈. DQN은 **stochastic gradient**로 이를 우회.

**본 논문의 차별점**:
- Raw pixel에서 end-to-end 학습.
- Model-free, off-policy Q-learning.
- Stochastic mini-batch 경사 하강.

Prior: TD-Gammon (on-policy, hand-crafted features) — didn't generalize beyond backgammon. NFQ — batch updates, too expensive. DQN's novelty: raw pixels, model-free off-policy Q-learning, SGD on mini-batches.

### Section 4: Deep Reinforcement Learning / 심층 강화학습 (가장 중요한 섹션)

#### 4.1 Experience Replay / 경험 재생

**아이디어**: 매 time step의 transition $e_t = (s_t, a_t, r_t, s_{t+1})$을 **replay memory** $\mathcal{D} = \{e_1, \dots, e_N\}$에 저장. 학습 시 $\mathcal{D}$에서 **무작위 mini-batch** 샘플링하여 Q-learning 업데이트 적용.

**장점 4가지**:
1. **각 experience가 여러 번 사용** → data efficiency 상승.
2. **연속 샘플의 상관관계 제거** → 분산 감소, 기울기 추정 개선.
3. **On-policy와 달리 과거 정책의 경험도 사용** → behavior 분포가 평균화되어 학습 진동/발산 방지.
4. **중요**: Q-learning이 off-policy이므로 이런 replay가 이론적으로 허용됨.

**실제 구현**:
- 버퍼 크기 $N = 10^6$ (최근 1M transitions).
- 새 transition이 오면 가장 오래된 것 제거(FIFO).
- 매 step 하나의 mini-batch (크기 32)를 샘플링해 gradient step 수행.

#### 4.2 Algorithm 1 / 알고리즘 1

**Deep Q-Learning with Experience Replay** (논문의 8줄 pseudocode):

```
Initialize replay memory D to capacity N
Initialize action-value function Q with random weights θ
for episode = 1 to M do
    Initialise sequence s_1 = {x_1}, φ_1 = φ(s_1)
    for t = 1 to T do
        With probability ε select a random action a_t
        otherwise select a_t = argmax_a Q(φ(s_t), a; θ)
        Execute action a_t, observe reward r_t and image x_{t+1}
        Set s_{t+1} = s_t, a_t, x_{t+1}, φ_{t+1} = φ(s_{t+1})
        Store transition (φ_t, a_t, r_t, φ_{t+1}) in D
        Sample random minibatch of transitions (φ_j, a_j, r_j, φ_{j+1}) from D
        Set y_j = r_j                     if φ_{j+1} is terminal
                 r_j + γ max_{a'} Q(φ_{j+1}, a'; θ)  otherwise
        Perform gradient descent step on (y_j - Q(φ_j, a_j; θ))² w.r.t. θ
    end for
end for
```

**핵심 포인트**:
- $\phi(s)$: 전처리 함수 (§4.3).
- Terminal state에서는 $y_j = r_j$ (bootstrapping 없음).
- **매 step마다** 행동, 저장, 그리고 학습이 모두 일어남.

#### 4.3 Preprocessing and Model Architecture / 전처리 및 아키텍처

**전처리 $\phi$**:
- RGB 210×160 → grayscale (1 channel).
- Downsample: 210×160 → 110×84.
- Crop: 중앙 84×84 영역 (GPU의 2D conv가 정사각형 입력을 요구).
- **Frame stacking**: 최근 **4프레임**을 채널 축으로 쌓음 → 입력 shape = (84, 84, 4).

**왜 4-frame stacking?** Atari는 단일 프레임으로는 **공의 속도와 방향**을 알 수 없음 (Markov state가 아님). 4프레임이면 속도·가속도까지 파악 가능.

**아키텍처 (논문 §4.3)**:
```
Input:     84 × 84 × 4  (4 stacked grayscale frames)
    │
    ▼
Conv1: 16 filters of 8×8, stride 4, ReLU
    │  Output: 20 × 20 × 16
    ▼
Conv2: 32 filters of 4×4, stride 2, ReLU
    │  Output: 9 × 9 × 32
    ▼
Flatten → FC: 256 units, ReLU
    │
    ▼
Output FC: |A| units (one Q value per action)
```

**핵심 설계 선택**: 입력에 state $s$만 넣고 **출력이 모든 action의 Q 값 벡터**. 한 번의 forward pass로 모든 Q를 동시에 계산 → argmax가 간단.

### Section 5: Experiments / 실험

**데이터셋**: 7개 Atari 2600 게임 — Beam Rider, Breakout, Enduro, Pong, Q*bert, Seaquest, Space Invaders. 각 게임 **동일한 네트워크, 동일한 하이퍼파라미터**.

**Reward clipping** (중요): 모든 게임의 보상을 $\{-1, 0, +1\}$로 clipping. 이유: 게임마다 점수 스케일이 다름 (Pong은 1점, Space Invaders는 수천점). 같은 learning rate를 쓰려면 스케일 통일 필요. **단점**: 에이전트가 점수의 "크기"를 구분 못 함.

**Training details**:
- RMSProp optimizer, mini-batch 32.
- ε = 1.0 → 0.1 선형 감소 (첫 1M frames), 이후 0.1 고정.
- 총 10M frames 훈련.
- **Frame skipping**: k=4 (k=3 for Space Invaders due to laser flickering). 에이전트가 k번째 프레임마다만 행동 선택, 사이 프레임은 같은 행동 반복 → ~4배 경험 가능.

**결과 (Table 1, Table 2)**:

| Game | Sarsa | Contingency | DQN | Best (Bellemare 2013) | Human | DQN > Human? |
|---|---|---|---|---|---|---|
| B. Rider | 996 | 1743 | **4092** | 6846 | 7456 | ✗ |
| Breakout | 5.2 | 6 | **168** | 42 | 31 | **✓** |
| Enduro | 129 | 159 | **470** | 287 | 368 | **✓** |
| Pong | -19 | -17 | **20** | 19 | -3 | **✓** |
| Q*bert | 614 | 960 | **1952** | 4500 | 18900 | ✗ |
| Seaquest | 665 | 723 | **1705** | 4070 | 28010 | ✗ |
| S. Invaders | 271 | 268 | **581** | 669 | 3690 | ✗ |

**핵심 관찰**:
- **6/7에서 기존 RL 방법(Sarsa, Contingency Aware 등) 능가**.
- **3/7 (Breakout, Enduro, Pong)에서 인간 전문가 능가**.
- Q*bert, Seaquest, Space Invaders는 인간을 못 따라감 — 이들은 **더 긴 시간적 전략** 필요.
- Space Invaders에서 Bellemare best(669)에 못 미침 — reward clipping이 "점수 크기"를 잃게 만든 영향 추정.

#### 5.1 Training and Stability / 훈련 및 안정성

**Average reward 곡선 (Figure 2)**은 매우 **noisy**. 이는 ε-greedy 탐색과 단일 에피소드 결과의 변동성 때문. 그러나 평균화하면 일관된 향상.

**Average Q-value 곡선 (Figure 2)**은 훨씬 **smooth**하게 상승. Q 함수 자체가 정책보다 직접적이고 안정적인 진전 지표.

**Bellman divergence 없음**: Deadly triad의 이론적 발산 경고와 달리, DQN은 발산하지 않음. Experience replay + SGD + reward clipping의 공학적 조합이 이론을 이긴 예.

#### 5.2 Visualizing the Value Function / 가치 함수 시각화

Seaquest 게임에서 적이 등장하는 순간 Q 값이 **급격히 상승**하고, 적을 쏘면 정점을 찍은 뒤 **보상을 받고 하락**. 즉 DQN이 **게임의 의미적 순간(적 등장, 공격 기회, 보상 수령)을 학습했음**을 시각적으로 확인.

#### 5.3 Main Evaluation / 주요 평가

그리고 §5.3에서 기존 방법과의 비교 (위 표), §5.4 개괄적 해석.

### Section 6: Conclusion / 결론

이 논문은 **deep learning 모델을 이용한 강화학습의 새 접근**을 제시. Stochastic gradient + experience replay가 이전의 불안정 문제를 극복. 픽셀 입력만으로 7개 게임을 같은 구조로 학습. 이 결과는 **범용 에이전트**의 가능성을 제시.

Paper introduces a new RL approach using deep models. SGD + experience replay overcome prior instability. Same architecture across 7 games from raw pixels. Results suggest the path to general agents.

---

## 3. Key Takeaways / 핵심 시사점

1. **Experience Replay는 RL의 i.i.d. 문제를 우회하는 엔지니어링적 정답** — 이론적으로 완벽하지 않지만(off-policy correction 없이도 작동), 실제로 deadly triad의 발산을 막는 가장 중요한 트릭. 오늘날 DDPG, SAC, Rainbow 등 모든 off-policy deep RL의 표준 구성 요소.
   **Experience replay is the engineering answer to RL's i.i.d. violation** — imperfect in theory but crucial in practice. Now a standard component in DDPG, SAC, Rainbow, and every off-policy deep RL.

2. **"Same architecture, same hyperparameters" 실험 설계가 범용 에이전트의 첫 증거** — 7개 게임에 모두 같은 CNN을 쓴다는 사실이 단일 게임의 성능보다 큰 기여. AlphaZero(바둑·체스·쇼기), Gato(604 태스크), 그리고 foundation model RL(GPT, Claude)이 모두 이 원리를 계승.
   **The "single architecture, single hyperparameters" paradigm is the first evidence of a general agent** — arguably more important than the per-game scores. AlphaZero, Gato, and foundation-model RL all inherit this design philosophy.

3. **Reward clipping은 간단하지만 의미 있는 손실** — $\{-1,0,+1\}$로 clipping하면 점수의 크기 정보를 잃어 Space Invaders 같은 게임에서 열등. 2015 Nature에서도 유지되지만, 이후 정교한 normalization(PopArt, reward-rescaling)으로 대체됨.
   **Reward clipping is a simple but costly normalization** — loses score magnitude (bad for Space Invaders), later replaced by PopArt or learned reward rescaling.

4. **Deadly triad는 이론과 실무가 극적으로 다를 수 있음을 보여준다** — Tsitsiklis & Van Roy (1997)는 function approx + bootstrap + off-policy의 발산을 수학적으로 증명했지만, 실제로는 experience replay + target network + careful tuning으로 충분히 안정화 가능. "이론적 문제 = 실용적 불가능"이 아니라는 중요한 교훈.
   **The deadly triad shows theory and practice can diverge** — the 1997 divergence theorem is real, but replay + target network + tuning suffice empirically. "Theoretical problem ≠ practical impossibility" is a key lesson.

5. **Frame stacking이 Markov 성질을 복원하는 방식** — Atari 단일 프레임은 공의 속도·방향 정보가 없어 non-Markov. 4프레임 스택으로 **수작업 feature engineering 없이** 시간적 정보를 입력에 포함. 같은 아이디어가 오늘날 비디오 모델, 로봇 센서 fusion의 기본 패턴.
   **Frame stacking restores Markov property** — a single frame lacks velocity/direction info; stacking 4 provides temporal context without hand engineering. This pattern lives in video models and robotic sensor fusion today.

6. **"State = action-value 벡터 출력"이라는 아키텍처 선택** — 입력 $s$만 받아 모든 action의 Q 값을 한 번에 출력. Argmax가 단순해지고, action 간 **파라미터 공유**로 sample efficiency 향상. Dueling DQN(2016)은 이를 $V(s) + A(s,a)$로 확장.
   **Architecture: state in, Q-vector out** — one forward pass gives Q for all actions. Parameter sharing across actions improves sample efficiency. Later extended to $V(s) + A(s,a)$ (Dueling DQN 2016).

7. **Stability tricks는 여러 개가 중첩되어야 함** — DQN의 성공은 단일 마법 하나가 아니라 (a) experience replay, (b) reward clipping, (c) 4-frame skipping, (d) RMSProp, (e) ε-annealing의 **조합**. 이후 Rainbow(2017)가 6개 추가 기법(Double DQN, Dueling, Prioritized Replay, Multi-step, Distributional, Noisy Nets)을 결합해 추가 개선을 입증.
   **Multiple stability tricks are required** — success comes from the combination of experience replay, reward clipping, frame skipping, RMSProp, ε-annealing. Rainbow (2017) added six more for further gains.

8. **Target의 non-stationarity는 본 논문에서는 미해결** — 매 step $\theta$가 변하면 target $y = r + \gamma \max Q(s'; \theta)$도 변함. 2015 Nature가 **target network** 도입으로 해결. 이 단순한 추가가 49개 Atari 게임에서 인간 수준을 가능하게 함. 본 논문의 한계이자 후속 연구의 열쇠.
   **Non-stationary targets are unsolved here** — every $\theta$ update shifts the target. The 2015 Nature follow-up's target network fix enabled human-level play on 49 games. A clear limitation of this paper that became the key next step.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Markov Decision Process / 마르코프 결정 과정

- State $s \in \mathcal{S}$, action $a \in \mathcal{A}$, reward $r \in \mathbb{R}$
- Transition $P(s' \mid s, a)$
- Discount $\gamma \in [0, 1)$
- Return $G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$

### 4.2 Q-function and Bellman Optimality / Q 함수와 Bellman 최적성

$$Q^*(s, a) = \max_\pi \mathbb{E}\!\left[G_t \mid s_t = s, a_t = a, \pi\right]$$

$$Q^*(s, a) = \mathbb{E}_{s' \sim \mathcal{E}}\!\left[r + \gamma \max_{a'} Q^*(s', a') \mid s, a\right]$$

최적 정책: $\pi^*(s) = \arg\max_a Q^*(s, a)$.

### 4.3 DQN Loss / DQN 손실 함수

$$L_i(\theta_i) = \mathbb{E}_{(s, a, r, s') \sim U(\mathcal{D})}\!\left[\left(y_i - Q(s, a; \theta_i)\right)^2\right]$$

여기서
$$y_i = \begin{cases} r & \text{if terminal} \\ r + \gamma \max_{a'} Q(s', a'; \theta_{i-1}) & \text{otherwise} \end{cases}$$

$U(\mathcal{D})$: replay buffer에서의 uniform sampling.
$\theta_{i-1}$: 본 논문에서는 이전 iteration의 θ (실제 구현에서는 매 step update).

### 4.4 Gradient / 기울기

$$\nabla_{\theta_i} L_i = \mathbb{E}\!\left[(r + \gamma \max_{a'} Q(s', a'; \theta_{i-1}) - Q(s, a; \theta_i)) \cdot \nabla_{\theta_i} Q(s, a; \theta_i)\right]$$

**중요**: $y_i$ 내부의 $\theta_{i-1}$은 **stop-gradient**로 처리. 실제 구현에서 `torch.no_grad()` 또는 `.detach()` 필수.

### 4.5 Algorithm 1 (Deep Q-Learning with Experience Replay) — 8-line pseudocode

```python
# Initialize
D = ReplayBuffer(capacity=1_000_000)
Q = CNN(input_shape=(4, 84, 84), output_dim=|A|)  # random init

for episode in range(M):
    s = env.reset()
    phi = preprocess(s)  # 4-frame stacked grayscale 84x84
    for t in range(T):
        # ε-greedy action selection
        a = random_action() if random() < eps else argmax Q(phi)
        # Execute
        x_next, r, done = env.step(a)
        phi_next = stack_frames(phi, x_next)
        # Store transition
        D.append((phi, a, r, phi_next, done))
        phi = phi_next
        # Sample mini-batch and train
        batch = D.sample(32)
        for (phi_j, a_j, r_j, phi_{j+1}, done_j) in batch:
            with torch.no_grad():
                y_j = r_j if done_j else r_j + gamma * Q(phi_{j+1}).max()
            loss = (y_j - Q(phi_j)[a_j])**2
            loss.backward(); optimizer.step()
```

### 4.6 Network Architecture / 네트워크 구조 (Worked Example)

입력: $(84, 84, 4)$
- Conv1: 16 filters, 8×8 kernel, stride 4, ReLU → $(20, 20, 16)$
  - FLOPs: $8^2 \times 4 \times 16 \times 20^2 = 1{,}638{,}400$
  - Params: $8^2 \times 4 \times 16 + 16 = 4{,}112$
- Conv2: 32 filters, 4×4 kernel, stride 2, ReLU → $(9, 9, 32)$
  - FLOPs: $4^2 \times 16 \times 32 \times 9^2 = 663{,}552$
  - Params: $4^2 \times 16 \times 32 + 32 = 8{,}224$
- FC1: $(9 \cdot 9 \cdot 32 = 2592) \to 256$, ReLU
  - Params: $2592 \times 256 + 256 = 663{,}808$
- FC2: $256 \to |A|$ (예: $|A| = 18$)
  - Params: $256 \times 18 + 18 = 4{,}626$

**총 파라미터**: ~681K (AlexNet의 60M에 비하면 매우 작음).
**게임당 총 FLOPs/forward**: ~2.3M.

### 4.7 Hyperparameters (from paper)

| Hyperparameter | Value |
|---|---|
| Replay memory size $N$ | $10^6$ |
| Mini-batch size | 32 |
| Frame skip $k$ | 4 (3 for Space Invaders) |
| History length | 4 frames |
| Target update | Every step (no target network in 2013) |
| $\varepsilon$ schedule | 1.0 → 0.1 linearly over 1M frames |
| Replay start size | 50,000 |
| Optimizer | RMSProp |
| Learning rate | 0.00025 |
| Discount $\gamma$ | 0.99 |
| Total frames | 10M |

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1951  Robbins-Monro      — Stochastic approximation
1988  Sutton             — TD(λ) learning
1989  Watkins            — Q-learning (tabular)
1992  Lin                — Experience replay (for neural net RL)
1995  Tesauro            — TD-Gammon (backgammon, neural net + TD)
1997  Tsitsiklis-VanRoy  — "Deadly triad" divergence theorem
2005  Riedmiller         — Neural Fitted Q-iteration (NFQ)
2012  Krizhevsky         — AlexNet (deep learning revolution)      [#13]
2013  Bellemare          — Arcade Learning Environment
2013  Mnih               — DQN                                     ★ THIS PAPER
2015  Mnih (Nature)      — DQN + Target Network, 49 Atari games
2015  van Hasselt        — Double DQN (overestimation fix)
2015  Schaul             — Prioritized Experience Replay
2015  Wang               — Dueling DQN (V + A decomposition)
2016  AlphaGo (Nature)   — beats Lee Sedol
2016  Lillicrap          — DDPG (continuous control)
2017  Schulman           — PPO                                     [#24]
2017  Hessel             — Rainbow DQN (6 improvements combined)
2017  Silver             — AlphaZero (Go, chess, shogi)
2019  OpenAI Five        — Dota 2
2019  Kalashnikov        — QT-Opt (robot grasping)
2022  Schrittwieser      — MuZero + stochastic environments
2022  ChatGPT / RLHF     — PPO for language alignment
2023+ Claude, GPT-4      — all use RLHF, descendant of this work
```

DQN 이전과 이후 심층 강화학습의 풍경은 완전히 다릅니다. DQN 이전: 도메인별 수작업 feature + 얕은 모델 + 부서지기 쉬운 튜닝. DQN 이후: 원시 입력 + deep net + 상대적으로 범용적인 레시피. 이 논문은 "neural net + RL"이 **작동 가능한 조합**임을 처음으로 증명했습니다.

The deep-RL landscape before and after DQN is completely different. Before: per-domain hand-crafted features, shallow models, fragile tuning. After: raw inputs, deep nets, a relatively universal recipe. DQN first proved "neural net + RL" is a workable combination.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **#13 AlexNet (Krizhevsky 2012)** | DQN의 CNN 아키텍처가 AlexNet 계보. "pixel → CNN → high-level decision" 패러다임을 RL로 이식. | AlexNet의 성공이 DQN의 raw-pixel 접근을 가능하게 함. ImageNet 혁명이 RL로 확장된 첫 사례. |
| **#10 LeCun 1998 (LeNet-5)** | CNN의 원조. DQN의 conv+FC 구조와 직접 계보. | Spatial locality, translation invariance 같은 CNN inductive bias를 RL에 적용. |
| **Watkins 1989 (Q-learning)** | DQN의 RL 알고리즘적 뿌리. Tabular Q-learning을 neural net로 확장. | Bellman equation, TD update, off-policy는 모두 Watkins의 유산. |
| **Tsitsiklis-Van Roy 1997 (Deadly Triad)** | DQN이 달래야 했던 이론적 난제. FA + bootstrap + off-policy의 이론적 발산. | Experience replay가 이 triad를 실용적으로 다루는 핵심 증거. |
| **Tesauro 1995 (TD-Gammon)** | Neural net RL의 첫 성공 사례. Backgammon 세계 챔피언 수준. | DQN은 TD-Gammon의 한계(단일 게임, hand-crafted feature)를 돌파. 범용성과 raw input이 차별점. |
| **Riedmiller 2005 (NFQ)** | Experience replay의 선행 구현. Batch fitting. | DQN은 NFQ의 batch를 stochastic mini-batch로 바꾸고 CNN을 도입. |
| **Bellemare 2013 (ALE)** | Atari 벤치마크. DQN 평가의 기반. | 동일 벤치마크에서 다른 방법들과 직접 비교 가능. |
| **#20 ResNet (He 2015)** | AlphaGo Zero의 Q/policy network가 ResNet 기반. DQN의 2-layer CNN에서 진화한 모습. | DQN → AlphaGo의 아키텍처 연결: 얕은 CNN → ResNet. |
| **#24 PPO (Schulman 2017)** | DQN이 value-based의 대표라면 PPO는 policy-based의 대표. 현대 RLHF의 기반. | 같은 deep RL 시대를 이루는 두 기둥. RLHF는 DQN의 간접 후손. |
| **#25 Transformer (Vaswani 2017)** | RLHF(GPT, Claude)에서 Transformer + PPO는 DQN의 정신적 후손. | Decision Transformer, Gato 등이 "RL을 sequence modeling으로" 재해석. |

---

## 7. References / 참고문헌

- **This paper**: Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., & Riedmiller, M. (2013). *Playing Atari with Deep Reinforcement Learning*. arXiv:1312.5602 (NeurIPS 2013 Deep Learning Workshop).
- Mnih, V., et al. (2015). *Human-level control through deep reinforcement learning*. Nature 518, 529-533. (DQN with Target Network, 49 games)
- Watkins, C. J. C. H. (1989). *Learning from delayed rewards*. PhD thesis, Cambridge.
- Watkins, C. J. C. H., & Dayan, P. (1992). *Q-learning*. Machine Learning 8(3-4), 279-292. (Convergence proof)
- Tesauro, G. (1995). *Temporal Difference Learning and TD-Gammon*. Communications of the ACM.
- Tsitsiklis, J. N., & Van Roy, B. (1997). *An analysis of temporal-difference learning with function approximation*. IEEE TAC. (Deadly triad)
- Riedmiller, M. (2005). *Neural Fitted Q Iteration*. ECML 2005.
- Lin, L.-J. (1992). *Self-improving reactive agents based on reinforcement learning, planning and teaching*. Machine Learning. (Experience replay origin)
- Bellemare, M. G., Naddaf, Y., Veness, J., & Bowling, M. (2013). *The Arcade Learning Environment*. JAIR.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*, 2nd ed. MIT Press. (Canonical textbook)
- van Hasselt, H., Guez, A., & Silver, D. (2016). *Deep Reinforcement Learning with Double Q-learning*. AAAI 2016.
- Wang, Z., et al. (2016). *Dueling Network Architectures for Deep Reinforcement Learning*. ICML 2016.
- Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2016). *Prioritized Experience Replay*. ICLR 2016.
- Hessel, M., et al. (2018). *Rainbow: Combining Improvements in Deep Reinforcement Learning*. AAAI 2018.
- Silver, D., et al. (2017). *Mastering the game of Go without human knowledge*. Nature 550, 354-359. (AlphaGo Zero)
- Silver, D., et al. (2018). *A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play*. Science. (AlphaZero)
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347. (PPO — paper #24)
