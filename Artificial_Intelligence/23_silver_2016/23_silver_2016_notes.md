---
title: "Mastering the game of Go with deep neural networks and tree search"
authors: [David Silver, Aja Huang, Chris J. Maddison, Arthur Guez, Laurent Sifre, George van den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, Sander Dieleman, Dominik Grewe, John Nham, Nal Kalchbrenner, Ilya Sutskever, Timothy Lillicrap, Madeleine Leach, Koray Kavukcuoglu, Thore Graepel, Demis Hassabis]
year: 2016
journal: "Nature"
doi: "10.1038/nature16961"
topic: Artificial_Intelligence
tags: [reinforcement-learning, monte-carlo-tree-search, deep-learning, self-play, game-ai, cnn, policy-gradient]
status: completed
date_started: 2026-04-19
date_completed: 2026-04-19
---

# 23. Mastering the game of Go with deep neural networks and tree search / 딥 뉴럴넷과 트리 탐색으로 바둑 정복하기

---

## 1. Core Contribution / 핵심 기여

### English
AlphaGo is the first computer Go program to defeat a human professional on a standard 19×19 board, achieving this by marrying three previously separate traditions: deep convolutional networks (for pattern recognition), reinforcement learning (for self-improvement), and Monte Carlo Tree Search (for lookahead). Two specialised networks do the heavy lifting. A **policy network** $p(a\mid s)$ outputs a distribution over moves and shrinks the **breadth** of search by letting MCTS focus on plausible actions. A **value network** $v(s)$ outputs a scalar win probability from a board position and shrinks the **depth** of search by replacing expensive rollouts at leaves. These networks are trained in a three-stage pipeline: (1) a supervised-learning policy $p_\sigma$ trained to imitate 30 million expert moves from the KGS server (57.0% top-1 accuracy, versus 44.4% prior state of the art); (2) a reinforcement-learning policy $p_\rho$ initialised from $p_\sigma$ and improved by policy-gradient self-play against a pool of previous snapshots (80% win rate against $p_\sigma$, 85% against Pachi); (3) a value network $v_\theta$ regressed on 30 million self-play positions (sampling one position per game to avoid correlated-state overfitting). At play time an asynchronous MCTS uses $p_\sigma$ as a PUCT prior and combines the value network with a fast rollout policy $p_\pi$ via a mixing weight $\lambda$, with $\lambda = 0.5$ optimal. The single-machine AlphaGo won 494/495 games (99.8%) against Crazy Stone, Zen, Pachi, Fuego, and GnuGo. The distributed version (40 search threads, 1,202 CPUs, 176 GPUs) won 77% of games against single-machine AlphaGo and defeated Fan Hui—European champion, 2-dan professional, winner of the 2013–2015 European championships—**5 games to 0** in a formal match held 5–9 October 2015. This was the first defeat of a human professional at full-size Go without handicap, a milestone considered a decade away.

### 한국어
AlphaGo는 표준 19×19 바둑판에서 인간 프로 기사를 처음으로 이긴 프로그램으로, 이전까지 분리되어 있던 세 흐름—**딥 컨볼루션 신경망**(패턴 인식), **강화학습**(자기개선), **몬테카를로 트리 탐색**(수읽기)—을 하나로 결합하여 달성했다. 두 전문 네트워크가 핵심이다. **정책망** $p(a\mid s)$ 은 가능한 수에 대한 확률 분포를 출력해 MCTS가 유망한 수에 집중하도록 **탐색의 너비(breadth)** 를 줄이고, **가치망** $v(s)$ 은 국면의 승률을 스칼라로 출력해 비싼 롤아웃을 대체함으로써 **탐색의 깊이(depth)** 를 줄인다. 학습은 3단계 파이프라인으로 진행된다. (1) KGS 서버의 3천만 개 전문가 수를 모방 학습한 지도학습 정책망 $p_\sigma$ (Top-1 정확도 57.0%, 이전 SOTA 44.4%); (2) $p_\sigma$ 로 초기화되어 과거 스냅샷 풀과 self-play하며 policy gradient로 강화된 RL 정책망 $p_\rho$ ($p_\sigma$ 상대 80% 승률, Pachi 상대 85% 승률); (3) RL 정책 기반 3천만 self-play 국면(각 대국에서 단 한 국면만 추출해 상관 과적합 방지)으로 회귀 학습한 가치망 $v_\theta$. 플레이 시에는 비동기 MCTS가 $p_\sigma$ 를 PUCT prior로 쓰고, 가치망과 빠른 롤아웃 정책 $p_\pi$ 를 혼합 파라미터 $\lambda$ 로 결합하며, $\lambda = 0.5$ 에서 최적이다. 단일 머신 AlphaGo는 Crazy Stone, Zen, Pachi, Fuego, GnuGo를 상대로 495전 494승(99.8%)을 거두었고, 분산 버전(40 search threads, 1,202 CPU, 176 GPU)은 단일 머신 대비 77% 승률을 기록하며 유럽 챔피언 2단 프로 Fan Hui를 **5대 0** 으로 완파했다 (2015년 10월 5–9일 정식 대국). 접바둑 없이 풀 사이즈 바둑에서 인간 프로를 이긴 첫 사례로, "적어도 10년 후의 일"이라 여겨졌던 기념비적 사건이다.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction and Motivation / 서론과 동기

**English**: The paper frames the problem in terms of the canonical game-tree equation: every perfect-information game has an optimal value function $v^*(s)$, recoverable by recursive minimax over $\sim b^d$ positions, where $b$ is breadth and $d$ is depth. Chess sits at $(b\approx 35, d\approx 80)$ and was solved at the top level by Deep Blue. Go sits at $(b\approx 250, d\approx 150)$ — a space so vast that the authors note exhaustive search is "infeasible." The escape hatches are the same two ideas repeated throughout the paper: *depth reduction* via position evaluation (truncating the tree and replacing subtrees with $v(s)$) and *breadth reduction* via sampling from a policy $p(a\mid s)$. Prior Go programs used MCTS with shallow linear policies/value functions on hand-crafted features; they reached strong amateur dan level but could not beat professionals.

**한국어**: 논문은 문제를 고전적인 게임 트리 방정식으로 제시한다: 모든 완전정보 게임에는 최적 가치 함수 $v^*(s)$ 가 존재하고, 이는 $\sim b^d$ 개 국면에 대한 재귀적 minimax로 구할 수 있다 ($b$ = 너비, $d$ = 깊이). 체스는 $(b\approx 35, d\approx 80)$ 으로 Deep Blue가 풀었다. 바둑은 $(b\approx 250, d\approx 150)$ 로 저자들조차 "전수 탐색은 불가능"이라고 단언한다. 탈출구는 논문 전체를 관통하는 두 아이디어다: *가치 함수 $v(s)$* 로 서브트리를 대체하는 **깊이 축소** 와, *정책 $p(a\mid s)$* 에서 샘플링하는 **너비 축소**. 이전 바둑 프로그램들은 수작업 특징 위에 얕은 선형 정책·가치를 올린 MCTS를 썼고, 강한 아마추어 단 수준까지 올랐으나 프로는 넘지 못했다.

### Part II: Supervised Learning of the Policy Network / 정책망의 지도학습

**English**: The SL policy $p_\sigma(a\mid s)$ is a **13-layer CNN** operating on a 19×19 input with 48 feature planes (stone colour, liberties, turns since last move, capture size, ladder status, etc.; Extended Data Table 2). Training is simple cross-entropy classification on 30 million state–action pairs from the KGS Go Server. The update is stochastic gradient **ascent** on log-likelihood:
$$
\Delta\sigma \propto \frac{\partial \log p_\sigma(a\mid s)}{\partial \sigma}
$$
Accuracy scaled with width: 128 → 192 → 256 → 384 filters all helped training accuracy, but more filters slowed evaluation. The 192-filter model hit **57.0%** top-1 accuracy using all features, **55.7%** using only raw board + move history; the prior SOTA at submission was 44.4%. Importantly, "small improvements in accuracy led to large improvements in playing strength" (Fig. 2a) — accuracy is a proxy for strength but not a linear one. Alongside $p_\sigma$, a much faster **rollout policy** $p_\pi$ is trained as a linear softmax on small hand-crafted pattern features, reaching 24.2% accuracy but evaluating in **2 μs** per move versus **3 ms** for the CNN (a ~1,500× speedup) — critical because MCTS rollouts run millions of moves.

**한국어**: SL 정책망 $p_\sigma(a\mid s)$ 는 **13층 CNN** 으로, 19×19 입력에 48개 feature plane(돌 색, 자유도, 수 이후 경과, 따낸 돌 수, 축 상태 등; Extended Data Table 2)을 받는다. 학습은 KGS Go Server의 3천만 (상태, 행동) 쌍에 대한 단순 cross-entropy 분류이다. 업데이트는 로그-가능도 방향의 SGD **상승**:
$$
\Delta\sigma \propto \frac{\partial \log p_\sigma(a\mid s)}{\partial \sigma}
$$
정확도는 너비에 따라 스케일링되었다: 128 → 192 → 256 → 384 필터 모두 training 정확도를 높였지만 필터가 많을수록 평가가 느려졌다. 192-filter 모델은 모든 특징을 사용할 때 **57.0%** , 원본 보드 + 수 순서만 사용할 때 **55.7%** Top-1 정확도를 기록했다 (제출 당시 SOTA 44.4%). 핵심 관찰: "작은 정확도 향상이 큰 기력 향상으로 이어진다"(Fig 2a). 즉 정확도는 기력의 프록시이지만 선형 관계는 아니다. $p_\sigma$ 와 함께 훨씬 빠른 **롤아웃 정책** $p_\pi$ 도 학습했다. 이는 수작업 패턴 특징 위의 선형 softmax로, 정확도는 24.2%에 불과하지만 착수당 **2 μs** (CNN의 **3 ms** 대비 ~1,500배 빠름)로 MCTS의 수백만 롤아웃에 결정적이다.

### Part III: Reinforcement Learning of the Policy Network / 정책망의 강화학습

**English**: Stage 2 takes $p_\sigma$ and keeps improving it with policy-gradient RL to *optimise for winning*, not for imitating humans. The RL policy $p_\rho$ has identical architecture and is initialised $\rho \leftarrow \sigma$. Games are played between the current $p_\rho$ and a **randomly sampled previous snapshot**, not just the latest version — this opponent pool prevents policy-gradient collapse into narrow self-exploits (overfitting to the current policy). The reward is zero for all non-terminal steps, and $z_t = \pm r(s_T) \in \{+1, -1\}$ at the end. REINFORCE update:
$$
\Delta \rho \propto \frac{\partial \log p_\rho(a_t\mid s_t)}{\partial \rho}\, z_t
$$
Results: head-to-head, $p_\rho$ beats $p_\sigma$ **>80%** of the time. Against **Pachi** — a sophisticated open-source MCTS program at roughly 2-dan amateur strength, executing 100,000 simulations per move — **$p_\rho$ wins 85% of games with no search at all**. Prior SL-only CNN approaches won just 11% against Pachi and 12% against Fuego. This is the first evidence in the paper that deep RL alone, with no lookahead, could already exceed pre-AlphaGo Go software.

**한국어**: 2단계는 $p_\sigma$ 를 policy gradient RL로 "승리 최적화"한다 (인간 모방이 아니라 **이기는 것** 이 목적). RL 정책 $p_\rho$ 는 구조가 동일하며 $\rho \leftarrow \sigma$ 로 초기화된다. 대국은 현재 $p_\rho$ 와 **무작위로 샘플링된 과거 스냅샷** 사이에서 진행 — 최신 버전끼리만 두면 좁은 self-exploit에 빠지는(현재 정책에 과적합) policy gradient collapse가 발생하므로 상대 풀을 유지하는 것이 중요하다. 보상은 비종단 스텝에서 0이고 $z_t = \pm r(s_T) \in \{+1, -1\}$ 이다. REINFORCE 업데이트:
$$
\Delta \rho \propto \frac{\partial \log p_\rho(a_t\mid s_t)}{\partial \rho}\, z_t
$$
결과: $p_\rho$ 가 $p_\sigma$ 를 **>80%** 로 이긴다. **Pachi** 상대 — 착수당 10만 시뮬레이션을 돌리는 아마추어 2단 수준 오픈소스 MCTS 프로그램 — **탐색 없이** $p_\rho$ 가 **85%** 승률을 기록한다. 이전 SL 전용 CNN 접근은 Pachi 상대 11%, Fuego 상대 12%에 그쳤다. 이는 논문에서 "딥 RL만으로도 수읽기 없이 기존 바둑 소프트웨어를 넘을 수 있다"는 첫 번째 증거다.

### Part IV: Reinforcement Learning of the Value Network / 가치망의 강화학습

**English**: The value network estimates
$$
v^p(s) = \mathbb{E}[z_t \mid s_t = s, a_{t\ldots T} \sim p]
$$
— the expected game outcome from $s$ under policy $p$. The authors approximate the outcome under their strongest policy $p_\rho$:
$$
v_\theta(s) \approx v^{p_\rho}(s) \approx v^*(s).
$$
The architecture mirrors the policy network but outputs a **single scalar** instead of a distribution. Training minimises MSE with the terminal outcome:
$$
\Delta \theta \propto \frac{\partial v_\theta(s)}{\partial \theta} \cdot (z - v_\theta(s))
$$
**The overfitting trap**: naive training on complete games (using all positions from a game as training data, all sharing the same target $z$) memorises outcomes rather than generalising. Training set MSE = **0.19**, test MSE = **0.37** — massive overfitting. **The fix**: generate a **new self-play dataset** of **30 million distinct positions, each from a separate game**. Each game is played between $p_\rho$ and itself, and only one position is kept. This decorrelates consecutive states, and the resulting MSEs are 0.226 (train) and 0.234 (test) — almost no gap, minimal overfitting. **Shocking efficiency**: a single forward pass of $v_\theta$ approaches the accuracy of Monte Carlo rollouts using $p_\rho$ but uses **15,000× less computation**. This is the single biggest compute argument in the paper — the value network replaces an army of rollouts.

**한국어**: 가치망은 다음을 추정한다:
$$
v^p(s) = \mathbb{E}[z_t \mid s_t = s, a_{t\ldots T} \sim p]
$$
즉 정책 $p$ 로 플레이할 때 $s$ 로부터의 기대 결과. 저자들은 가장 강한 정책 $p_\rho$ 로 근사한다:
$$
v_\theta(s) \approx v^{p_\rho}(s) \approx v^*(s).
$$
구조는 정책망과 유사하나 분포가 아닌 **스칼라 하나** 를 출력한다. 학습은 종단 결과에 대한 MSE 최소화:
$$
\Delta \theta \propto \frac{\partial v_\theta(s)}{\partial \theta} \cdot (z - v_\theta(s))
$$
**과적합의 함정**: 완전한 대국 데이터로 순진하게 학습하면(한 대국의 모든 국면이 같은 타깃 $z$ 를 공유), 일반화가 아니라 결과 암기가 일어난다. 훈련 MSE = **0.19**, 테스트 MSE = **0.37** — 심각한 과적합. **해결책**: **각각 다른 대국에서 뽑은 3천만 개의 독립 국면** 으로 self-play 데이터셋을 재생성. 각 대국은 $p_\rho$ 끼리 두고 딱 한 국면만 보관한다. 이로써 연속 상태의 상관이 사라지고, MSE는 0.226(훈련)/0.234(테스트)로 거의 차이 없음 — 과적합 최소화. **충격적인 효율성**: $v_\theta$ 의 단 한 번의 forward pass가 $p_\rho$ 로 하는 MC 롤아웃의 정확도에 근접하면서 연산은 **15,000배** 적다. 논문의 가장 강력한 compute 논거 — 가치망이 롤아웃 군단을 대체한다.

### Part V: Searching with Policy and Value Networks / 정책·가치망을 이용한 탐색

**English**: At play time, AlphaGo runs an MCTS that stores at each edge $(s,a)$: an action value $Q(s,a)$, a visit count $N(s,a)$, and a prior $P(s,a)$. One simulation proceeds in four phases.

**(a) Selection**: At each internal node, choose
$$
a_t = \arg\max_a\ \big[Q(s_t, a) + u(s_t, a)\big], \qquad u(s, a) \propto \frac{P(s, a)}{1 + N(s, a)}
$$
The bonus $u$ decays with visits, pushing exploration early and exploitation later.

**(b) Expansion**: When the traversal reaches a leaf $s_L$, the SL policy network evaluates it once to produce priors: $P(s, a) = p_\sigma(a\mid s)$.

**(c) Evaluation**: The leaf is evaluated in **two complementary ways** — by the value network $v_\theta(s_L)$, and by playing a fast rollout with $p_\pi$ to the end of the game for outcome $z_L$. These are mixed:
$$
V(s_L) = (1 - \lambda) v_\theta(s_L) + \lambda z_L
$$

**(d) Backup**: Statistics along the traversed path are updated:
$$
N(s, a) = \sum_{i=1}^{n} \mathbf{1}(s, a, i), \qquad Q(s, a) = \frac{1}{N(s, a)} \sum_{i=1}^{n} \mathbf{1}(s, a, i)\, V(s_L^i)
$$
After all simulations, AlphaGo plays the **most-visited** root move, **not the highest-$Q$** move — visit count is a lower-variance statistic robust to fluke high-$Q$ leaves.

**Two surprising asymmetries**:
1. **SL** policy is used as the prior $P(s,a)$, not RL, because humans play a **diverse beam** of good moves whereas RL converges to a single peaky best move. Diversity helps search.
2. The **value network** derived from the **RL** policy is used (because it's stronger), *not* the value network from the SL policy. Strength helps evaluation.

So prior ← SL (diversity), value ← RL (strength) — role-matched rather than uniform.

**Compute**: Single-machine AlphaGo uses 40 search threads, 48 CPUs, 8 GPUs. Distributed AlphaGo uses 40 search threads, 1,202 CPUs, 176 GPUs. Asynchronous lock-free search with virtual loss diversifies concurrent threads.

**한국어**: 대국 시 AlphaGo는 각 엣지 $(s,a)$ 에 action value $Q(s,a)$, 방문 횟수 $N(s,a)$, prior $P(s,a)$ 를 저장하는 MCTS를 실행한다. 한 시뮬레이션은 4단계로 진행된다.

**(a) 선택 / Selection**: 각 내부 노드에서
$$
a_t = \arg\max_a\ \big[Q(s_t, a) + u(s_t, a)\big], \qquad u(s, a) \propto \frac{P(s, a)}{1 + N(s, a)}
$$
보너스 $u$ 는 방문이 늘수록 감소 → 초기엔 탐험, 후기엔 활용.

**(b) 확장 / Expansion**: 리프 $s_L$ 에 도달하면 SL 정책망이 한 번 평가하여 prior 생성: $P(s, a) = p_\sigma(a\mid s)$.

**(c) 평가 / Evaluation**: 리프를 **두 가지 상호보완적 방법** 으로 평가 — 가치망 $v_\theta(s_L)$ 와 $p_\pi$ 를 이용한 빠른 롤아웃의 게임 결과 $z_L$. 혼합:
$$
V(s_L) = (1 - \lambda) v_\theta(s_L) + \lambda z_L
$$

**(d) 역전파 / Backup**: 경로상 모든 엣지 통계를 업데이트:
$$
N(s, a) = \sum_{i=1}^{n} \mathbf{1}(s, a, i), \qquad Q(s, a) = \frac{1}{N(s, a)} \sum_{i=1}^{n} \mathbf{1}(s, a, i)\, V(s_L^i)
$$
모든 시뮬레이션 후 AlphaGo는 **가장 많이 방문된** 루트 수를 선택, **$Q$ 최대가 아님** — 방문 횟수는 분산이 낮아 운 좋은 고-$Q$ 리프에 속지 않는다.

**두 가지 놀라운 비대칭성**:
1. prior $P(s,a)$ 에는 **SL** 정책이 사용된다 (RL이 아님). 인간은 여러 좋은 수를 **다양하게** 두는 반면, RL은 단일 최선수로 수렴(peaky). 탐색엔 다양성이 유리하다.
2. 가치망은 **RL** 정책에서 파생된 버전을 사용한다 (SL 버전 아님, RL이 더 강하므로). 평가엔 강도가 유리하다.

즉 prior ← SL (다양성), value ← RL (강도) — 획일적이 아니라 역할에 맞춘 선택.

**연산 자원**: 단일 머신 AlphaGo = 40 search threads, 48 CPUs, 8 GPUs. 분산 AlphaGo = 40 search threads, 1,202 CPUs, 176 GPUs. 비동기 lock-free 탐색과 virtual loss로 동시 스레드의 경로 다양성 확보.

### Part VI: Evaluating Playing Strength / 기력 평가

**English**: An internal tournament pitted AlphaGo against Crazy Stone, Zen, Pachi, Fuego, and GnuGo with 5 s per move.

- **Single-machine AlphaGo**: **494/495** games won (99.8%) — the lone loss illustrates MCTS robustness.
- **Four-stone handicap** (opponents get 4 free moves): AlphaGo still wins 77% vs Crazy Stone, 86% vs Zen, 99% vs Pachi.
- **Distributed AlphaGo**: 77% of games vs single-machine AlphaGo, 100% of games vs other programs.

**Elo ratings** (95% CI shown in Fig. 4a): Crazy Stone/Zen ~1,950, Pachi ~1,750, Fuego ~1,600, GnuGo ~450, Fan Hui ~2,900, single-machine AlphaGo ~2,900, distributed AlphaGo ~3,140. A 230-Elo gap corresponds to 79% win probability.

**Leaf-evaluation ablation (Fig. 4b)** — the key methodological result:
| $\lambda$ | Method | Behaviour |
|---|---|---|
| $\lambda = 0$ | value-only | Beats every other Go program |
| $\lambda = 1$ | rollout-only | Strong but weaker than value-only |
| $\lambda = 0.5$ | mixed | **Wins ≥95% of games against all variants** |

This is the empirical vindication of the "two networks are complementary" design: the value net approximates outcomes under the **strong but slow** $p_\rho$, while rollouts actually play out games using the **weak but fast** $p_\pi$. Their errors are de-correlated.

**Scalability (Fig. 4c)**: Elo continues rising with more search threads and GPUs, up to the distributed 40-thread, 1,202-CPU, 176-GPU configuration — no plateau observed at these scales.

**한국어**: 5초/수 제한으로 Crazy Stone, Zen, Pachi, Fuego, GnuGo와 내부 토너먼트를 개최.

- **단일 머신 AlphaGo**: **494/495승** (99.8%) — 유일한 1패는 MCTS의 강건성을 보여준다.
- **4점 접바둑** (상대에게 4수 선착권): AlphaGo 승률 Crazy Stone 77%, Zen 86%, Pachi 99%.
- **분산 AlphaGo**: 단일 머신 상대 77%, 다른 프로그램 상대 **100%**.

**Elo 등급** (95% CI, Fig 4a): Crazy Stone/Zen ~1,950, Pachi ~1,750, Fuego ~1,600, GnuGo ~450, Fan Hui ~2,900, 단일 AlphaGo ~2,900, 분산 AlphaGo ~3,140. 230 Elo 차이 = 79% 승률.

**리프 평가 ablation (Fig 4b)** — 핵심 방법론적 결과:
| $\lambda$ | 방법 | 동작 |
|---|---|---|
| $\lambda = 0$ | value만 | 다른 모든 바둑 프로그램을 이김 |
| $\lambda = 1$ | rollout만 | 강하지만 value-only보다 약함 |
| $\lambda = 0.5$ | 혼합 | **모든 변형 상대 ≥95% 승률** |

이는 "두 네트워크는 상호보완"이라는 설계의 실증적 뒷받침이다. 가치망은 **강하지만 느린** $p_\rho$ 의 결과를 근사하고, 롤아웃은 **약하지만 빠른** $p_\pi$ 로 실제 끝까지 둔다. 오차가 탈-상관되어 있다.

**확장성 (Fig 4c)**: 탐색 스레드와 GPU를 늘릴수록 Elo가 계속 상승 — 이 규모에서는 아직 plateau가 나타나지 않았다.

### Part VII: Match against Fan Hui / Fan Hui와의 대국

**English**: Fan Hui — European champion 2013/2014/2015, 2-dan professional — played a **formal 5-game match** with distributed AlphaGo from 5–9 October 2015 (no handicap, professional time controls). **AlphaGo 5–0 Fan Hui** (Game 1 by 2.5 points; Games 2–5 by resignation). This was the first time a computer had defeated a human professional on a full-size board without handicap. The accompanying "position visualisation" (Fig. 5) shows six views of a single mid-game position: value-only and rollout-only tree evaluations, policy network priors, simulation frequencies, and the principal variation. Notably, in the visualised position Fan Hui's post-game commentary agreed that the move AlphaGo preferred (labelled 1) was better than what he actually played — an early signal that the machine's positional judgment was not merely "tactical" but strategic.

**한국어**: Fan Hui — 2013/2014/2015 유럽 챔피언, 2단 프로 — 는 2015년 10월 5–9일에 분산 AlphaGo와 **정식 5국 대국** 을 치렀다 (접바둑 없음, 프로 제한시간). **AlphaGo 5–0 Fan Hui** (1국 2.5집 승, 2–5국 불계승). 접바둑 없이 풀 보드에서 컴퓨터가 인간 프로를 이긴 첫 사례. 함께 실린 "국면 시각화"(Fig 5)는 중반 한 국면에 대한 6가지 뷰 — value-only 및 rollout-only 트리 평가, 정책망 prior, 시뮬레이션 빈도, 최선 변화(principal variation)—를 보여준다. 주목할 점: 시각화된 국면에서 Fan Hui는 사후 분석에서 "AlphaGo가 선호한 수(label 1)가 자신이 실제로 둔 수보다 좋았다"고 인정했다. 기계의 포지셔널 판단이 단순 전술(tactical)이 아니라 전략적(strategic)이라는 초기 신호.

### Part VIII: Discussion / 논의

**English**: The authors contextualise three points. (1) During the Fan Hui match, AlphaGo evaluated **thousands of times fewer positions** than Deep Blue did against Kasparov — compensated by more intelligent selection (policy network) and more precise evaluation (value network). This hints at a path "perhaps closer to how humans play." (2) Where Deep Blue used a hand-crafted evaluator, AlphaGo's networks are trained **purely through general-purpose supervised and reinforcement learning methods** — no Go-specific human knowledge beyond the input features and board rules. (3) Go had been a grand challenge of AI; the paper frames this as a methodological template for "seemingly intractable AI domains" requiring challenging decision-making, intractable search, and complex-to-evaluate optimal solutions. The subtext: the recipe (deep nets + RL + MCTS) generalises.

**한국어**: 저자들은 세 가지를 맥락화한다. (1) Fan Hui 대국 동안 AlphaGo는 Deep Blue가 Kasparov 상대로 평가한 것보다 **수천 배 적은 국면** 을 평가했다 — 더 똑똑한 선택(정책망)과 더 정확한 평가(가치망)가 이를 보완한다. "인간이 두는 방식에 더 가까운 길"을 시사한다. (2) Deep Blue는 수작업 평가함수였지만, AlphaGo의 네트워크들은 **범용 지도학습·강화학습만으로** 학습된다 — 입력 특징과 규칙 외에 바둑 특화 인간 지식이 없다. (3) 바둑은 AI의 "grand challenge"였고, 이 논문은 난해한 결정·비가역적 탐색·복잡한 평가가 얽힌 AI 도메인의 방법론적 템플릿을 제시한다. 숨은 주장: 이 레시피(딥넷 + RL + MCTS)는 일반화된다.

---

## 3. Key Takeaways / 핵심 시사점

1. **Policy and value networks attack breadth and depth independently / 정책망과 가치망은 너비와 깊이를 독립적으로 공략한다** — The intellectual core of the paper is a clean decomposition: $p(a\mid s)$ tells MCTS "where to look" (breadth reduction), $v(s)$ tells MCTS "how to evaluate where you end up" (depth reduction). Each is a neural network; each is trained separately; each plugs into a well-understood slot of classical MCTS. / 논문의 지적 핵심은 깔끔한 분해다: $p(a\mid s)$ 가 MCTS에게 "어디를 볼지"(너비 축소), $v(s)$ 가 "도달한 국면을 어떻게 평가할지"(깊이 축소). 각각은 신경망이고, 각각 따로 학습되며, 고전 MCTS의 명확한 자리에 들어간다.

2. **Three-stage training pipeline matters, and each stage has a distinct role / 3단계 학습 파이프라인이 중요하며, 각 단계는 서로 다른 역할을 한다** — SL produces a **diverse imitator**, RL produces a **strong specialist**, value regression produces a **cheap evaluator**. Skipping or merging stages would not work: the RL step needs a good initialisation (SL), and the value step needs a strong target policy (RL). / SL은 **다양한 모방자** 를, RL은 **강한 전문가** 를, value 회귀는 **값싼 평가자** 를 만든다. 단계를 생략하거나 합칠 수 없다: RL 단계는 좋은 초기화(SL)를, value 단계는 강한 타깃 정책(RL)을 필요로 한다.

3. **"One position per game" is a methodological gem / "대국당 한 국면"은 방법론적 보석** — The gap between train MSE 0.19 and test MSE 0.37 when training the value network on whole games is the hidden villain of the paper. The fix — generating 30M games and keeping a single state from each — is simple, surprising, and quantitatively decisive (MSE gap collapses to 0.226 vs 0.234). A lesson about correlated data in RL that extends far beyond Go. / 가치망을 완전 대국으로 학습할 때의 훈련 MSE 0.19 vs 테스트 MSE 0.37 격차는 논문의 숨은 악당이다. 해법—3천만 대국을 만들고 각 대국에서 단 하나의 상태만 보관—은 단순하고 놀랍고 정량적으로 결정적이다 (격차가 0.226 vs 0.234로 붕괴). 바둑을 넘어서는 RL에서 상관된 데이터에 관한 교훈.

4. **The SL-as-prior / RL-as-value asymmetry is a subtle engineering call / SL이 prior, RL이 value에 쓰이는 비대칭은 미묘한 엔지니어링 선택** — RL policies become peaky (low entropy); SL policies retain breadth. MCTS priors benefit from breadth, so SL wins there. Evaluation targets benefit from strength, so the stronger RL policy wins as the target for the value network. The paper almost buries this lede — it's a warning that "stronger is always better" is wrong in search systems. / RL 정책은 뾰족해진다(낮은 엔트로피); SL 정책은 너비를 유지한다. MCTS prior는 너비가 이득이므로 SL이 이긴다. 평가 타깃은 강도가 이득이므로 더 강한 RL 정책이 가치망의 타깃으로 이긴다. 논문은 이 요점을 거의 묻어두었지만—탐색 시스템에서 "더 강한 것이 늘 옳다"는 통념에 대한 경고다.

5. **Mixed leaf evaluation beats either pure evaluator / 혼합 리프 평가가 어느 한쪽 단독보다 낫다** — Fig. 4b is the crown jewel of the ablations. $\lambda=0$ (value only) beats every other Go program on Earth; $\lambda=1$ (rollout only) also strong; $\lambda=0.5$ (mix) wins ≥95% against all other variants. The two signals are uncorrelated enough that combining them is pure gain. This idea — *ensemble different kinds of estimators* — echoes through modern RL. / Fig 4b는 ablation의 보석이다. $\lambda=0$ (가치만)이 지상의 모든 바둑 프로그램을 이긴다; $\lambda=1$ (롤아웃만)도 강함; $\lambda=0.5$ (혼합)은 모든 변형 상대 ≥95% 승률. 두 신호의 상관이 충분히 낮아서 결합이 순이득이다. *서로 다른 종류의 추정기를 앙상블하라* 는 현대 RL의 반향.

6. **Self-play with a pool of past versions, not just the latest / 최신 대신 과거 스냅샷 풀과 self-play** — "Randomising from a pool of opponents stabilises training by preventing overfitting to the current policy." This is the origin of many modern self-play recipes (PSRO, league training in StarCraft II, etc.). Had DeepMind let $p_\rho$ play only itself, cycles and exploitable blind spots would emerge. / "과거 상대 풀에서 무작위 샘플링하여 현재 정책에 대한 과적합을 방지하고 학습을 안정화." 이는 많은 현대 self-play 레시피(PSRO, StarCraft II의 league training 등)의 기원이다. $p_\rho$ 가 자기 자신과만 두었다면 사이클과 착취 가능한 사각지대가 생겼을 것.

7. **Compute numbers reveal why this worked in 2016, not 2010 / 연산 수치가 이 연구가 2010이 아닌 2016에 성공한 이유를 드러낸다** — 30M KGS positions is tractable because of modern GPUs; training three networks is tractable because of batched CNN kernels; distributed AlphaGo used 176 GPUs because DeepMind had them. The single-machine → distributed Elo gap is ~200 points (~76% win rate), meaning compute still mattered at the frontier. / 3천만 KGS 국면은 현대 GPU가 있어야 가능하다; 세 네트워크 학습은 배치 CNN 커널이 있어야 가능하다; 분산 AlphaGo는 DeepMind에 176 GPU가 있었기에 가능했다. 단일→분산 Elo 격차가 ~200점(~76% 승률)이라는 점은, 최전선에서 연산이 여전히 중요함을 의미한다.

8. **Playing the most-visited, not the most-valued, root move / $Q$ 최대가 아니라 방문 횟수 최대의 루트 수를 둔다** — A small but crucial design choice. Visit count is accumulated across many simulations and averages out noise; a single lucky rollout might inflate $Q$ for an otherwise bad move. This same choice propagates into AlphaGo Zero and AlphaZero. / 작지만 결정적인 설계. 방문 횟수는 여러 시뮬레이션에 걸쳐 누적되므로 노이즈가 평균화된다; 운 좋은 롤아웃 하나가 나쁜 수의 $Q$ 를 부풀릴 수 있다. 이 선택은 AlphaGo Zero와 AlphaZero에도 이어진다.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Game-tree preliminaries / 게임 트리 예비지식

$$ v^*(s) = \max_a \min_{a'} \max_{a''} \cdots r(s_T) $$
- Exhaustive minimax over $\sim b^d$ positions; infeasible for Go ($b^d \approx 250^{150}$).
- Depth reduction: $v(s) \approx v^*(s)$ replaces subtrees with an evaluator.
- Breadth reduction: sample from $p(a\mid s)$ instead of enumerating all actions.
- 바둑에서는 $b^d$ 전수 탐색이 불가능하므로 **깊이** 는 $v(s)$ 로, **너비** 는 $p(a\mid s)$ 로 축소한다.

### 4.2 SL policy update / SL 정책 업데이트

$$ \Delta\sigma \propto \frac{\partial \log p_\sigma(a\mid s)}{\partial \sigma} $$
- Stochastic gradient **ascent** on cross-entropy of expert moves (maximum likelihood).
- 13-layer CNN; 192 filters; 48-plane 19×19 input.
- Top-1 accuracy 57.0% (all features), 55.7% (board + move history only). Prior SOTA 44.4%.
- **정확도 스케일링**: 필터 수 ↑ → 정확도 ↑, 평가 시간 ↑.
- 전문가 수에 대한 cross-entropy의 SGD 상승 (최대가능도 추정).

### 4.3 Rollout policy / 롤아웃 정책

- Linear softmax on small pattern features. 24.2% top-1 accuracy.
- **2 μs** per move versus **3 ms** for the CNN (≈1,500× faster).
- Used exclusively inside rollouts where speed is decisive.
- 수작업 패턴 특징 위의 선형 softmax. 착수당 2μs로 CNN보다 ~1,500배 빠름. 롤아웃 전용.

### 4.4 RL policy gradient update / RL 정책 경사 업데이트

$$ \Delta\rho \propto \frac{\partial \log p_\rho(a_t\mid s_t)}{\partial \rho}\, z_t $$
- REINFORCE with terminal reward $z_t \in \{-1, +1\}$, zero reward elsewhere.
- Opponent sampled from **pool of previous $\rho$ snapshots**, not only the latest.
- Outcome: $p_\rho$ beats $p_\sigma$ >80%, beats Pachi 85% (no search).
- 종단 보상 $z_t$ 를 사용하는 REINFORCE; 상대는 과거 스냅샷 풀에서 샘플링하여 현재 정책 과적합 방지.

### 4.5 Value network regression / 가치망 회귀

$$ v^p(s) = \mathbb{E}[z_t \mid s_t = s,\ a_{t\ldots T} \sim p] $$
$$ v_\theta(s) \approx v^{p_\rho}(s) \approx v^*(s) $$
$$ \Delta\theta \propto \frac{\partial v_\theta(s)}{\partial \theta} (z - v_\theta(s)) $$
- MSE regression on 30M positions, **one per self-play game**.
- MSE: 0.226 (train) vs 0.234 (test). Overfitting minimised.
- Compare to whole-game training: 0.19 / 0.37 — large gap → memorisation.
- Single $v_\theta$ eval ≈ MC-rollout accuracy under $p_\rho$, but **15,000× less compute**.
- $p_\rho$ self-play 3천만 국면(대국당 하나)으로 MSE 회귀 학습. 단 한 번의 평가가 MC 롤아웃과 동급 정확도이면서 연산은 15,000배 적다.

### 4.6 PUCT selection in MCTS / MCTS의 PUCT 선택

$$ a_t = \arg\max_a [Q(s_t, a) + u(s_t, a)] $$
$$ u(s, a) \propto \frac{P(s, a)}{1 + N(s, a)}, \qquad P(s, a) = p_\sigma(a\mid s) $$
- Prior $P$ biases search toward moves the **SL** policy (not RL) considers likely — chosen for **diversity**, not peak strength.
- Bonus decays with visits → exploration shifts to exploitation over time.
- prior로 **SL** 정책을 쓰는 이유: 다양성 확보 (RL은 peaky해서 탐색 효율 떨어짐).

### 4.7 Mixed leaf evaluation / 혼합 리프 평가

$$ V(s_L) = (1 - \lambda) v_\theta(s_L) + \lambda z_L $$
- $z_L$ obtained by rolling out from $s_L$ with $p_\pi$ to the end of the game.
- Ablation: $\lambda=0$ beats all programs; $\lambda=1$ strong; $\lambda=0.5$ dominates ≥95%.
- 롤아웃 승패 $z_L$ 과 가치망 예측을 $\lambda$ 로 혼합. $\lambda=0.5$ 가 최적.

### 4.8 Backup statistics / 역전파 통계

$$ N(s, a) = \sum_{i=1}^n \mathbf{1}(s, a, i), \qquad Q(s, a) = \frac{1}{N(s, a)} \sum_{i=1}^n \mathbf{1}(s, a, i) V(s_L^i) $$
- Visit count and mean leaf evaluation along edges traversed by the $i$-th simulation.
- Final play: $a_{\text{play}} = \arg\max_a N(s_0, a)$ (not max-$Q$).
- 시뮬레이션 $i$ 가 엣지를 지났는지를 indicator로 표현하여 방문 횟수와 평균 평가 업데이트. 최종 수는 $N$ 최대값 (로버스트니스).

### 4.9 Worked numerical example / 수치 예시

Suppose after $n = 1000$ simulations at the root, three candidate moves have:

| Move | $N$ | $Q$ | $P$ |
|---|---|---|---|
| $a_1$ | 700 | +0.22 | 0.55 |
| $a_2$ | 220 | +0.30 | 0.25 |
| $a_3$ | 80  | +0.08 | 0.10 |

- By $Q$: $a_2$ wins (+0.30), but it's had fewer visits — its $Q$ is less reliable.
- By $N$: $a_1$ wins (700 visits) — $Q$ is also respectable (+0.22).
- AlphaGo plays **$a_1$** because 700 simulations is strong evidence that the subtree is sound, whereas $a_2$'s +0.30 could dip as more simulations arrive.

**한국어**: 루트에서 1000회 시뮬레이션 후 세 후보수가 위와 같다면, $Q$ 최대는 $a_2$ (+0.30)이지만 방문이 적어 신뢰도 낮다. AlphaGo는 $N$ 최대인 $a_1$ (700회 방문, $Q$=+0.22)을 선택한다 — 많은 방문이 서브트리 건전성의 강한 증거이며, $a_2$ 는 더 돌리면 $Q$ 가 떨어질 수 있다.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1950 ─ Shannon: "Programming a computer for playing chess"
1959 ─ Samuel: self-play checkers learner (first RL-like TD in a game)
1992 ─ TD-Gammon (Tesauro): neural net + TD learning reaches world-class backgammon
1997 ─ Deep Blue beats Kasparov — brute-force + hand-crafted eval in chess
2006 ─ Kocsis & Szepesvári: UCT / modern MCTS formulation
2006–12 Crazy Stone / Fuego / Zen / Pachi: MCTS brings Go to strong amateur dan
2012 ─ AlexNet (Krizhevsky): CNNs dominate ImageNet
2013 ─ DQN (Mnih): deep RL plays Atari from pixels (#22 in our reading list)
2014 ─ Clark & Storkey: CNN policy for Go move prediction (~44% accuracy)
2015 ─ Maddison et al.: CNN move prediction reaches strong amateur without search
2015 Oct AlphaGo 5–0 Fan Hui (European champion)                 ← THIS PAPER
2016 Jan Paper published in Nature, Vol. 529
2016 Mar AlphaGo 4–1 Lee Sedol (9-dan); "move 37" (Game 2, move 37)
2017 May AlphaGo 3–0 Ke Jie (world #1); project retired
2017 Oct AlphaGo Zero: no human data; surpasses AlphaGo with self-play alone
2018    AlphaZero: one algorithm masters chess, shogi, Go
2019    MuZero: also learns the environment model
2020+   AlphaFold 2 (protein), AlphaCode, AlphaTensor, AlphaDev
2024+   LLM "thinking" (o1, Claude thinking) — inference-time search echoes MCTS
```

**한국어 요약**: 이 논문은 (1) 1950년대 Shannon/Samuel의 게임 AI 아이디어, (2) 1992년 TD-Gammon의 신경망+RL 프로토타입, (3) 2006년 UCT/MCTS, (4) 2012년 AlexNet 이후의 CNN 혁명, (5) 2013년 DQN의 딥 RL을 한 점에서 통합한다. 이후 AlphaGo Zero/AlphaZero/MuZero로 "인간 데이터 제거 → 일반화 → 환경 모델 학습"의 축이 진행되고, AlphaFold 등 과학 도메인으로 확장된다. 현재 LLM의 "thinking" 모드도 이 계보의 후손이다.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **#22 Mnih et al. 2015 — Human-level control through deep reinforcement learning (DQN)** | Immediate predecessor — first paper to show deep networks could be trained end-to-end with RL on pixel inputs. / 바로 이전 논문 — 딥 네트워크를 픽셀 입력 위에서 end-to-end RL로 학습시킨 첫 논문. | AlphaGo shares the "deep RL" thesis but replaces Q-learning with policy gradient and adds MCTS. / AlphaGo는 "딥 RL" 명제를 공유하되 Q-learning 대신 policy gradient를 쓰고 MCTS를 추가. |
| **Krizhevsky, Sutskever & Hinton 2012 — AlexNet (ImageNet CNN)** | The CNN architecture at the heart of both the policy and value networks. / 정책·가치망 양쪽의 CNN 구조적 토대. | Without AlexNet-era CNN advances, 57% move-prediction accuracy would be unreachable. / AlexNet급 CNN 진전 없이는 57% 이동 예측 정확도 불가능. |
| **Maddison, Huang, Sutskever & Silver 2015 — Move Evaluation in Go Using Deep CNNs (ICLR)** | Direct methodological precursor from the same team — CNN move prediction at strong amateur level *without search*. / 같은 팀의 직접적 방법론 선행 — 탐색 없이 CNN만으로 강한 아마추어 수준. | Sets the baseline (44.4%) that AlphaGo's $p_\sigma$ pushes to 57.0%. / AlphaGo의 $p_\sigma$ 가 57%로 끌어올린 baseline(44.4%)을 제시. |
| **Kocsis & Szepesvári 2006 — Bandit-based Monte-Carlo Planning (UCT)** | Provides the MCTS + UCT framework that AlphaGo extends with neural priors. / AlphaGo가 신경망 prior로 확장하는 MCTS+UCT 프레임워크 제공. | The PUCT formula replaces the $\ln N(s)/N(s,a)$ term with a neural prior, marrying classic and learned components. / PUCT 공식은 $\ln N/N(s,a)$ 항을 신경망 prior로 치환하여 고전과 학습을 결합. |
| **Williams 1992 — REINFORCE (Simple statistical gradient-following algorithms)** | The policy-gradient identity driving the RL policy stage. / RL 정책 단계를 이끄는 policy gradient 항등식. | $\Delta\rho \propto \nabla \log p_\rho \cdot z$ is pure REINFORCE, with outcome $z$ as the (unbiased, high-variance) return. / $\Delta\rho \propto \nabla \log p_\rho \cdot z$ 는 순수 REINFORCE. |
| **Tesauro 1994 — TD-Gammon** | Proof-of-concept that neural nets + self-play can reach world-class play in a large game. / 신경망+self-play가 대규모 게임에서 세계 정상급에 오를 수 있다는 PoC. | AlphaGo is TD-Gammon's grandchild: deeper nets, full MCTS, Go's vastly larger state space. / AlphaGo는 TD-Gammon의 손자 격 — 더 깊은 망, 전면 MCTS, 훨씬 큰 상태 공간. |
| **Silver et al. 2017 — AlphaGo Zero** | Direct successor — removes human data entirely, uses the same policy + value idea but initialised from scratch via self-play. / 직접 후속작 — 인간 데이터 완전 제거, 같은 정책+가치 아이디어를 self-play만으로 밑바닥부터. | Retrospectively reframes this paper: the human-data stage is not essential, only helpful as an accelerator. / 인간 데이터 단계는 필수가 아닌 가속기였음을 회고적으로 재해석. |
| **Silver et al. 2018 — AlphaZero** | Generalises the recipe to chess and shogi with **one algorithm, no game-specific knowledge**. / 체스/쇼기로 **단일 알고리즘, 게임 특화 지식 없이** 일반화. | Validates the "deep RL + MCTS" pipeline as a general method, not Go-specific. / "딥 RL + MCTS" 파이프라인이 바둑 특화가 아닌 일반 방법임을 검증. |

---

## 7. References / 참고문헌

### Primary paper / 본 논문
- Silver, D., Huang, A., Maddison, C. J., Guez, A., Sifre, L., van den Driessche, G., Schrittwieser, J., Antonoglou, I., Panneershelvam, V., Lanctot, M., Dieleman, S., Grewe, D., Nham, J., Kalchbrenner, N., Sutskever, I., Lillicrap, T., Leach, M., Kavukcuoglu, K., Graepel, T., & Hassabis, D. (2016). *Mastering the game of Go with deep neural networks and tree search.* **Nature, 529**(7587), 484–489. DOI: [10.1038/nature16961](https://doi.org/10.1038/nature16961)

### Key references cited by the paper / 본 논문이 인용한 주요 참고문헌
- Kocsis, L. & Szepesvári, C. (2006). Bandit-based Monte-Carlo planning. *15th European Conference on Machine Learning*, 282–293. (UCT / MCTS 기초)
- Coulom, R. (2007). Efficient selectivity and backup operators in Monte-Carlo tree search. *5th International Conference on Computers and Games*, 72–83.
- Maddison, C. J., Huang, A., Sutskever, I. & Silver, D. (2015). Move evaluation in Go using deep convolutional neural networks. *ICLR 2015*. (직접 선행 연구)
- Clark, C. & Storkey, A. (2015). Training deep convolutional neural networks to play Go. *ICML 2015*, 1766–1774.
- Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. **Nature, 518**(7540), 529–533. (DQN — Paper #22)
- Krizhevsky, A., Sutskever, I. & Hinton, G. (2012). ImageNet classification with deep convolutional neural networks. *NeurIPS 2012*, 1097–1105. (AlexNet)
- Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. **Machine Learning, 8**, 229–256. (REINFORCE)
- Sutton, R. S., McAllester, D., Singh, S. & Mansour, Y. (2000). Policy gradient methods for reinforcement learning with function approximation. *NeurIPS 2000*, 1057–1063.
- LeCun, Y., Bengio, Y. & Hinton, G. (2015). Deep learning. **Nature, 521**, 436–444. (딥러닝 개관)
- Campbell, M., Hoane, A. J. & Hsu, F. (2002). Deep Blue. **Artificial Intelligence, 134**, 57–83. (체스의 대선배)

### Historical successors / 역사적 후속작 (본 논문 이후)
- Silver, D., et al. (2017). Mastering the game of Go without human knowledge. **Nature, 550**, 354–359. (AlphaGo Zero)
- Silver, D., et al. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play. **Science, 362**(6419), 1140–1144. (AlphaZero)
- Schrittwieser, J., et al. (2020). Mastering Atari, Go, chess and shogi by planning with a learned model. **Nature, 588**, 604–609. (MuZero)
