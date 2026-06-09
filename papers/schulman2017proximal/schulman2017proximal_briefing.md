---
title: "Pre-Reading Briefing: Proximal Policy Optimization Algorithms (PPO)"
paper_id: "24_schulman_2017"
topic: Artificial_Intelligence
date: 2026-04-19
type: briefing
---

# PPO: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347.
**Author(s)**: John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov (OpenAI)
**Year**: 2017 (v1: 19 Jul 2017; v2: 28 Aug 2017)

---

## 1. 핵심 기여 / Core Contribution

### 한국어
PPO(Proximal Policy Optimization)는 **정책 경사(policy gradient)** 방법의 실용적 표준을 확립한 알고리즘이다. 문제의식은 명확하다: (1) **Vanilla policy gradient**(REINFORCE, A2C)는 한 번의 과도한 업데이트가 정책을 망가뜨려 복구가 어렵고, (2) **TRPO** (Trust Region Policy Optimization)는 성능이 안정적이지만 **conjugate gradient + 자연 경사 + KL 제약 + line search** 라는 2차 최적화 파이프라인으로 구현이 복잡하고, dropout이나 파라미터 공유 같은 현대적 아키텍처와 호환되지 않는다. PPO의 해법은 놀랍도록 단순하다. **Importance sampling ratio** $r_t(\theta) = \pi_\theta(a_t\mid s_t) / \pi_{\theta_\text{old}}(a_t\mid s_t)$ 를 $[1-\epsilon, 1+\epsilon]$ 구간 밖으로 나가지 못하게 **clip** 하고, clipped surrogate와 원 surrogate의 **최솟값** 을 취하는 목적 함수:

$$ L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t\left[\min\left(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right] $$

이 식 하나가 TRPO의 복잡한 2차 기구(KL divergence constraint + 자연 경사)를 대체한다. 핵심 장점은 (a) **SGD + Adam** 만 사용하는 1차 최적화, (b) **동일 데이터 배치로 여러 epoch** 반복 가능(샘플 효율), (c) 구현 복잡도가 TRPO 대비 1/10 수준, (d) actor-critic, RNN, parameter sharing 모두와 호환. 실험: **OpenAI Gym MuJoCo**(연속 제어 로봇 locomotion 7개 환경), **Atari 49 게임** 에서 PPO가 A2C, ACER, TRPO, vanilla PG를 **샘플 효율과 최종 성능 모두** 에서 능가하거나 필적하면서 **훨씬 단순** 함을 입증. PPO는 이후 OpenAI Five (Dota 2), OpenAI 로봇 손, **InstructGPT/ChatGPT의 RLHF** 에 이르기까지 가장 널리 쓰이는 RL 알고리즘이 되었다.

### English
PPO (Proximal Policy Optimization) established the de facto standard for **policy-gradient** RL. The diagnosis is sharp: (1) **Vanilla policy gradient** (REINFORCE, A2C) suffers from destructively large updates that can permanently degrade the policy, and (2) **TRPO** (Trust Region Policy Optimization) is stable but implements **conjugate gradient + natural gradient + KL constraint + line search** — a second-order pipeline that is complex to code and incompatible with modern tricks like dropout or weight sharing between policy and value networks. PPO's remedy is remarkably simple. Define the **importance-sampling ratio** $r_t(\theta) = \pi_\theta(a_t\mid s_t) / \pi_{\theta_\text{old}}(a_t\mid s_t)$; clip it to $[1-\epsilon, 1+\epsilon]$; and take the minimum of the clipped and un-clipped surrogates:

$$ L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t\left[\min\left(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right] $$

A single scalar objective supplants TRPO's second-order machinery. The wins are (a) pure first-order optimisation with **SGD/Adam**, (b) multiple **epochs of minibatch updates on the same rollout** (sample efficiency), (c) ~10× less implementation complexity than TRPO, and (d) compatibility with actor-critic, RNNs, and parameter sharing. Experimental results: on **OpenAI Gym MuJoCo** (7 continuous-control robotic-locomotion environments) and **Atari** (49 games), PPO **matches or beats A2C, ACER, TRPO, and vanilla PG in both sample efficiency and final return** while being far simpler. PPO became the workhorse of subsequent achievements — OpenAI Five (Dota 2), the OpenAI robotic hand, and most significantly **the RL step of InstructGPT/ChatGPT (RLHF)**.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

#### 한국어
2017년 당시 deep RL은 세 계열로 나뉘어 있었다. (1) **Deep Q-learning** 계열 (DQN, Double/Duelling DQN, Rainbow): 값 함수 기반, 이산 행동에 강하지만 연속 제어는 곤란. (2) **Vanilla policy gradient** (REINFORCE/A2C/A3C): 간단하지만 데이터 효율과 안정성이 낮다. (3) **TRPO + Natural policy gradient** (Schulman 2015): 이론적으로 monotonic improvement를 보장하지만 구현이 복잡하고, 2차 방법이라 parameter-sharing 아키텍처와 충돌. 같은 저자 Schulman의 **TRPO(2015)** 논문이 이 흐름의 직전 성취였다. 한편 2016–17년 **GAE** (Generalized Advantage Estimation, Schulman et al. 2016)로 advantage 추정의 분산/편향 트레이드오프가 깔끔하게 정리되었고, OpenAI의 **baselines** 라이브러리 같은 인프라가 보편화되면서 **"안정적이면서도 단순한 1차 policy gradient"** 에 대한 수요가 커졌다. PPO는 바로 이 공백을 메웠다.

#### English
By 2017, deep RL had three lanes. (1) **Deep Q-learning** (DQN, Double/Duelling DQN, Rainbow): value-based, strong on discrete control but awkward on continuous actions. (2) **Vanilla policy gradient** (REINFORCE, A2C/A3C): simple, but sample-inefficient and unstable. (3) **TRPO + natural policy gradient** (Schulman 2015): theoretically sound with monotonic-improvement guarantees, but hairy to implement and second-order methods fight with parameter-sharing architectures. Schulman's own **TRPO (2015)** was the immediate predecessor. Around the same time, **GAE** (Generalized Advantage Estimation, Schulman et al. 2016) cleanly resolved the bias–variance trade-off in advantage estimation, and OpenAI's **baselines** library made reproducing RL results more tractable. The demand was for a **"stable but simple first-order policy gradient."** PPO filled exactly that gap.

### 타임라인 / Timeline

```
1992 ─ Williams: REINFORCE (the canonical policy-gradient estimator)
1999 ─ Sutton, McAllester, Singh, Mansour: Policy Gradient Theorem
2002 ─ Kakade: Natural Policy Gradient (Fisher-information-aware updates)
2013 ─ DQN (Mnih et al., Atari)                                ← #22 in our list
2015 ─ TRPO (Schulman et al., ICML)   — theoretical monotonic-improvement guarantee
2016 ─ A3C (Mnih et al., async advantage actor-critic)
2016 ─ GAE (Schulman et al., ICLR) — advantage estimation bias-variance knob λ
2016 ─ ACER (Wang et al.) — off-policy actor-critic with experience replay
2017 Jul PPO (Schulman et al.)                                  ← THIS PAPER
2017 ─ IMPALA (Espeholt et al.) — scalable actor-critic with V-trace
2018 ─ OpenAI Five (Dota 2) trained with PPO
2019 ─ OpenAI robotic hand manipulation (Rubik's cube) — PPO
2022 ─ InstructGPT / ChatGPT RLHF pipeline uses PPO
2023–  GRPO (Shao et al., DeepSeek) simplifies PPO for LLM alignment
```

---

## 3. 필요한 배경 지식 / Prerequisites

### 한국어
- **Markov Decision Process (MDP)** 의 기본: 상태 $s$, 행동 $a$, 보상 $r$, 정책 $\pi(a|s)$, 할인율 $\gamma$, 리턴 $G_t = \sum \gamma^k r_{t+k}$
- **Policy gradient theorem**: $\nabla_\theta J(\theta) = \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) \cdot A(s,a)]$
- **Actor-critic**: 정책(actor)과 값 함수(critic)를 동시에 학습; advantage $A_t = Q(s,a) - V(s)$
- **Advantage estimation**: TD(0), Monte Carlo, **GAE** $\hat{A}^{\text{GAE}(\gamma,\lambda)}_t = \sum_{l=0}^\infty (\gamma\lambda)^l \delta_{t+l}$, where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$
- **KL divergence**: $D_{KL}(p \| q) = \sum p(x) \log \frac{p(x)}{q(x)}$; 분포간 차이 측정
- **Importance sampling**: 다른 분포 $q$ 에서 수집한 데이터로 $p$ 의 기대값 추정: $\mathbb{E}_p[f] = \mathbb{E}_q[(p/q) f]$
- **TRPO의 아이디어**: $\max_\theta L^{PG}(\theta)$ subject to $D_{KL}(\pi_{old} \| \pi_\theta) \le \delta$
- **직전 논문 #22 (DQN)** 의 "경험 수집 → 업데이트" 사고방식
- **선형대수/확률 기초**: 기대값 추정, 분산, 몬테카를로 샘플링

### English
- **Markov Decision Process basics**: state $s$, action $a$, reward $r$, policy $\pi(a|s)$, discount $\gamma$, return $G_t = \sum \gamma^k r_{t+k}$.
- **Policy gradient theorem**: $\nabla_\theta J(\theta) = \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) \cdot A(s,a)]$.
- **Actor-critic**: joint learning of policy (actor) and value (critic); advantage $A_t = Q(s,a) - V(s)$.
- **Advantage estimation**: TD(0), Monte Carlo, or **GAE** $\hat{A}^{\text{GAE}(\gamma,\lambda)}_t = \sum_{l=0}^\infty (\gamma\lambda)^l \delta_{t+l}$ with $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$.
- **KL divergence**: $D_{KL}(p \| q)$ — distance between distributions.
- **Importance sampling**: estimate $\mathbb{E}_p[f]$ from samples of a different distribution $q$: $\mathbb{E}_p[f] = \mathbb{E}_q[(p/q) f]$.
- **TRPO**: $\max_\theta L^{PG}(\theta)$ s.t. $D_{KL}(\pi_{old} \| \pi_\theta) \le \delta$.
- **Previous paper #22 (DQN)**: the "collect data → update" RL loop.
- **Elementary linear algebra/probability**: expectation estimates, variance, Monte Carlo sampling.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **Surrogate objective** | 원래의 기대 리턴 $J(\theta)$ 를 직접 최적화하지 않고 대체로 최적화하는 **대리 목적함수**. PG에서는 $\log \pi \cdot A$, TRPO/PPO에서는 $(\pi/\pi_{old}) \cdot A$. / A proxy objective optimised in place of the true expected return; vanilla PG uses $\log \pi \cdot A$, TRPO/PPO use $(\pi/\pi_\text{old}) \cdot A$. |
| **Importance ratio $r_t(\theta)$** | 같은 행동 $a_t$ 에 대한 **새 정책 / 이전 정책** 확률 비: $r_t(\theta) = \pi_\theta(a_t\mid s_t) / \pi_{\theta_\text{old}}(a_t\mid s_t)$. 과거 데이터 재활용의 수학적 근거. / Ratio of new-to-old policy probabilities for the same action; the mathematical basis for reusing old rollouts. |
| **Clipping** | $r_t(\theta)$ 를 $[1-\epsilon, 1+\epsilon]$ 로 **자르기**. $\epsilon=0.2$ 가 표준. 정책이 한 번에 20% 이상 변하지 못하게 막는 **first-order 신뢰 영역**. / Hard truncation of $r_t$ into $[1-\epsilon, 1+\epsilon]$, $\epsilon=0.2$ typical. A first-order trust region preventing >20% policy drift in one step. |
| **PPO-Clip** | 본 논문의 주력 변형. $L^{\text{CLIP}} = \mathbb{E}[\min(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t)]$. 단순하고 구현 쉬움. / The paper's flagship variant; simple and effective. |
| **PPO-KL-Penalty** | 대안 변형. 라그랑지안 형태: $L^{\text{KL}} = \mathbb{E}[r_t A_t - \beta D_{KL}(\pi_{old} \| \pi_\theta)]$. $\beta$ 를 목표 KL에 맞춰 adaptive하게 조정. 최종적으로 clip 변형이 더 잘 작동함. / Lagrangian alternative with an adaptive $\beta$ tracking a target KL; found to underperform the clip version. |
| **GAE (Generalized Advantage Estimation)** | $\hat{A}_t^{\text{GAE}(\gamma,\lambda)} = \sum_{l=0}^\infty (\gamma\lambda)^l \delta_{t+l}$. $\lambda$ 로 TD(0)와 MC 사이 조절. $\lambda=0.95$ 가 표준. / Weighted geometric sum of TD residuals; $\lambda$ interpolates between TD(0) bias and MC variance. |
| **Clipped value loss** | 구현 트릭: $L^{VF} = \max((V_\theta - V_t^{targ})^2, (V^{clipped} - V_t^{targ})^2)$ 로 가치 함수 업데이트도 clip. 실험적으로 도움. / Implementation trick: clip the value-function update symmetrically; empirically stabilises. |
| **Multiple epochs per rollout** | 한 번의 데이터 수집($T$ 스텝)으로 $K$ epoch 학습. TRPO는 1 epoch만 가능했음. PPO의 sample efficiency 비밀. / $K$ training epochs per rollout batch; TRPO could only afford 1. The source of PPO's sample efficiency. |
| **Actor-critic with shared network** | 정책과 가치 함수가 backbone을 공유. Total loss: $L^{CLIP} - c_1 L^{VF} + c_2 S[\pi]$ where $S$ is entropy bonus. / Shared backbone with combined loss including entropy bonus. |
| **Fixed-length trajectory segments** | 에피소드 전체가 아닌 $T$-step segment로 데이터 수집. $N$ actors × $T$ steps = $NT$ transitions per iteration. / Fixed $T$-step segments collected by $N$ parallel actors per iteration. |

---

## 5. 수식 미리보기 / Equations Preview

### (1) 일반 정책 경사 / Vanilla Policy Gradient

$$
\hat{g} = \hat{\mathbb{E}}_t\left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t)\, \hat{A}_t \right], \qquad L^{PG}(\theta) = \hat{\mathbb{E}}_t\left[ \log \pi_\theta(a_t\mid s_t)\, \hat{A}_t \right]
$$

- 업데이트는 advantage 가중 로그확률의 그래디언트
- 여러 epoch 돌리면 분포 이동으로 발산 → PPO가 해결하려는 문제
- Gradient of advantage-weighted log-probabilities; multi-epoch training causes distributional drift — the problem PPO fixes.

### (2) TRPO의 대리 목적 / TRPO surrogate (importance-sampled)

$$
L^{CPI}(\theta) = \hat{\mathbb{E}}_t\left[\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_\text{old}}(a_t\mid s_t)}\hat{A}_t\right] = \hat{\mathbb{E}}_t[r_t(\theta)\hat{A}_t]
$$

subject to $\hat{\mathbb{E}}_t[D_{KL}(\pi_{\theta_\text{old}} \| \pi_\theta)] \le \delta$.

- CPI = "Conservative Policy Iteration" — 탐색은 old 정책 샘플로 하되 $\theta$ 의 비를 통해 업데이트
- KL 제약이 trust region을 정의; 2차 최적화 필요
- Samples are collected under $\pi_\text{old}$; the ratio $r_t$ reweights. The KL constraint defines a trust region, needing second-order methods.

### (3) PPO의 Clipped Objective — 논문의 핵심 / PPO's clipped objective — the heart of the paper

$$
L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t\left[\min\left(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]
$$

**한국어 해설**:
- $\hat{A}_t > 0$ (좋은 행동): $r_t$ 가 $1+\epsilon$ 을 넘지 못하게 clip. 즉 좋은 행동의 확률을 무한정 올리지 못함
- $\hat{A}_t < 0$ (나쁜 행동): $r_t$ 가 $1-\epsilon$ 이하로 내려가지 못하게 clip. 즉 나쁜 행동 확률을 무한정 내리지 못함
- **min** 은 pessimistic bound: 원 objective와 clip된 objective 중 작은 쪽. 업데이트가 과도하게 낙관적이지 않게 함
- $\epsilon = 0.2$ (연속 제어에서 표준)

**English**:
- For $\hat{A}_t > 0$: $r_t$ is clipped at $1+\epsilon$ — no unbounded upside on good actions.
- For $\hat{A}_t < 0$: $r_t$ is clipped at $1-\epsilon$ — no unbounded downside on bad actions.
- **min** gives a pessimistic (lower) bound on improvement, preventing over-optimistic updates.
- $\epsilon = 0.2$ is the canonical value.

### (4) 적응적 KL 페널티 변형 / Adaptive-KL variant (alternative)

$$
L^{\text{KLPEN}}(\theta) = \hat{\mathbb{E}}_t\left[r_t(\theta)\hat{A}_t - \beta\, D_{KL}(\pi_{\theta_\text{old}} \| \pi_\theta)\right]
$$

- $\beta$ 를 매 업데이트 후 rule-based로 조정: 실제 KL이 목표의 1.5배 초과 → $\beta \leftarrow 2\beta$; 목표의 1/1.5 미만 → $\beta \leftarrow \beta/2$
- 결과적으로 clipping 변형이 더 잘 동작 (구현 단순, 성능 우위)
- Adaptive $\beta$ tracks a target KL; found empirically weaker than the clip variant.

### (5) 전체 손실 / Total loss for actor-critic sharing

$$
L_t^{CLIP+VF+S}(\theta) = \hat{\mathbb{E}}_t\left[L_t^{CLIP}(\theta) - c_1 L_t^{VF}(\theta) + c_2 S[\pi_\theta](s_t)\right]
$$

- $L^{VF} = (V_\theta(s_t) - V_t^{targ})^2$: 가치 함수 MSE
- $S[\pi_\theta]$: 엔트로피 보너스 (exploration 유도)
- 논문 권장: $c_1 \approx 1$, $c_2 \approx 0.01$, $\epsilon = 0.2$, GAE $\lambda = 0.95$, $\gamma = 0.99$
- $L^{VF}$ = value MSE; $S$ = entropy bonus encouraging exploration.

---

## 6. 읽기 가이드 / Reading Guide

### 한국어
논문은 짧다(10페이지). 다음 순서로 읽으면 좋다.

1. **Abstract + §1 Introduction**: "TRPO는 복잡, vanilla PG는 불안정, 그 사이가 필요" 문제의식 파악.
2. **§2 Background: Policy Optimization** — §2.1(PG 리뷰), §2.2(TRPO 리뷰)까지 확실히 이해. PPO를 읽기 위한 기반.
3. **§3 Clipped Surrogate Objective**: **핵심**. 식 (6)-(7) — 이 두 줄이 논문의 전부. Fig 1의 clipping behaviour 그림 주의.
4. **§4 Adaptive KL Penalty**: 대안 변형. 결론은 "clip이 이김".
5. **§5 Algorithm**: PPO의 full algorithm box. Algorithm 1을 손으로 따라 쓰면서 GAE, minibatch update loop 이해.
6. **§6 Experiments**: §6.1(surrogate objective 비교, Table 1), §6.2(연속 제어 MuJoCo, Fig 3), §6.3(TRPO·CEM·A2C 비교, Fig 4), §6.4(Humanoid 데모), §6.5(Atari, Table 4). Fig 3의 learning curve가 논문의 상징.
7. **§7 Conclusion**: 요약.

### English
The paper is short (10 pages). Recommended reading order:

1. **Abstract + §1 Introduction**: internalise "TRPO is complex, vanilla PG is unstable — there must be a middle path."
2. **§2 Background: Policy Optimization** — §2.1 (PG review), §2.2 (TRPO review). Solid groundwork for reading PPO.
3. **§3 Clipped Surrogate Objective**: **the core**. Equations (6)–(7) — the whole paper in two lines. Linger on Figure 1's clipping behaviour.
4. **§4 Adaptive KL Penalty**: the alternative variant. The punchline: clip wins.
5. **§5 Algorithm**: the full algorithm box. Trace Algorithm 1 by hand; understand GAE and the minibatch update loop.
6. **§6 Experiments**: §6.1 (surrogate comparison, Table 1), §6.2 (MuJoCo, Figure 3), §6.3 (vs TRPO/CEM/A2C, Figure 4), §6.4 (Humanoid demo), §6.5 (Atari, Table 4). Figure 3's learning curves are the paper's visual signature.
7. **§7 Conclusion**: wrap-up.

---

## 7. 현대적 의의 / Modern Significance

### 한국어
PPO는 딥 RL 역사에서 **"단순함이 이긴다"** 의 대표적 사례다. 구체적 영향:

1. **사실상의 표준 RL 알고리즘**: OpenAI baselines, Stable-Baselines3, Ray RLlib 모두 PPO를 1순위로 제공. "RL을 어느 알고리즘으로 시작할까?"의 답은 2017년 이후 PPO.
2. **OpenAI Five (Dota 2, 2018)** 및 **OpenAI robotic hand (Rubik's cube, 2019)**: 10,000 GPU 수준의 대규모 self-play RL이 PPO 기반으로 동작함을 입증.
3. **RLHF와 LLM 정렬**: **InstructGPT (2022)** 와 **ChatGPT (2022)** 가 보상 모델 위에서 PPO로 정책을 최적화. 거의 모든 초기 LLM RLHF 파이프라인(Anthropic의 초기 Claude, Meta LLaMA-2 Chat 등)이 PPO 기반.
4. **GRPO, DPO, RLOO, IPO 등 후속 알고리즘의 기준점**: 이들 대부분은 PPO의 단점(reward model 별도 학습, 샘플 비용, 하이퍼파라미터 민감도)을 해결하려는 방향. PPO가 없었다면 이들 연구도 없었다.
5. **구현 문화**: "first-order + clip"이라는 PPO의 접근은 이후 **BERT의 warmup**, **GPT의 gradient clipping**, **robust RL의 constraint softening** 등 여러 맥락에 녹아들었다.

흥미로운 점: 논문은 10페이지밖에 안 되지만, Schulman이 훗날 "여러 구현 디테일(value clipping, reward normalisation, advantage normalisation, orthogonal init 등)이 실제로는 식 (7)만큼 중요하다"고 회고. Engstrom et al. (2020) "Implementation Matters in Deep RL" 논문이 이를 실증했다.

### English
PPO is the canonical "simple beats complex" moment in deep RL history.

1. **De facto standard RL algorithm**: OpenAI baselines, Stable-Baselines3, and Ray RLlib all feature PPO prominently. Post-2017, the default answer to "which RL algorithm should I start with?" is PPO.
2. **OpenAI Five (Dota 2, 2018)** and **the OpenAI robotic hand (Rubik's cube, 2019)**: both demonstrated PPO scaling to 10,000-GPU self-play.
3. **RLHF and LLM alignment**: **InstructGPT (2022)** and **ChatGPT (2022)** optimise policies with PPO against a learned reward model. Nearly every first-generation LLM RLHF pipeline (early Claude, LLaMA-2 Chat, etc.) used PPO as the inner loop.
4. **Baseline for successors — GRPO, DPO, RLOO, IPO**: most are responses to PPO's weaknesses (separate reward model, sample cost, hyperparameter sensitivity). They exist because PPO existed first.
5. **Implementation culture**: the "first-order + clip" idiom propagated — BERT-style warmup, GPT-era gradient clipping, robust-RL constraint softening all echo PPO's stance.

A candid footnote: the paper is only 10 pages, but Schulman later acknowledged that **implementation details** — value clipping, reward/advantage normalisation, orthogonal init, learning-rate annealing — matter as much as Equation (7). Engstrom et al. (2020), *Implementation Matters in Deep Policy Gradients*, documents this empirically.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
