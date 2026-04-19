---
title: "Proximal Policy Optimization Algorithms"
authors: [John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov]
year: 2017
journal: "arXiv:1707.06347"
doi: "arXiv:1707.06347"
topic: Artificial_Intelligence
tags: [reinforcement-learning, policy-gradient, trust-region, actor-critic, rlhf, deep-rl]
status: completed
date_started: 2026-04-19
date_completed: 2026-04-19
---

# 24. Proximal Policy Optimization Algorithms / 근접 정책 최적화 알고리즘

---

## 1. Core Contribution / 핵심 기여

### English
PPO (Proximal Policy Optimization) introduces a family of policy-gradient algorithms that capture the stability and reliability of **Trust Region Policy Optimization (TRPO)** using only **first-order** updates. The authors observe that vanilla policy gradient is destructively unstable when multiple gradient steps are taken on the same rollout (the policy drifts far from the data-collection distribution, and the importance-sampling-based surrogate becomes meaningless), while TRPO solves this via a KL-constrained second-order step that is complex, brittle, and incompatible with modern tricks like dropout or actor-critic parameter sharing. PPO's central contribution is the **clipped surrogate objective**
$$ L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t\left[\min\!\left(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right] $$
where $r_t(\theta) = \pi_\theta(a_t\mid s_t)/\pi_{\theta_\text{old}}(a_t\mid s_t)$ and $\epsilon$ is typically $0.2$. The clip prevents the ratio from ever helpfully exceeding $1\pm\epsilon$; the outer $\min$ ensures the objective is a pessimistic **lower bound** on the unclipped one. The full algorithm alternates between $N$ parallel actors each collecting $T=2048$ timesteps of data and $K=10$ epochs of minibatch SGD/Adam over the combined $NT$ transitions, with GAE for advantage estimation and an optional entropy bonus. An alternative adaptive-KL-penalty variant is included for completeness but underperforms. On **7 MuJoCo continuous-control benchmarks** (1M timesteps), PPO with $\epsilon=0.2$ achieves the best normalized score (**0.82** vs. 0.74 for the best KL variant, and $-0.39$ for the un-clipped baseline), and on **49 Atari games** PPO wins the **fast-learning metric** on 30 games (vs. 18 for ACER, 1 for A2C). The paper is 10 pages long but reshaped policy-gradient deep RL: PPO became the default algorithm powering OpenAI Five, the OpenAI robotic hand, and most consequentially the **RL-from-Human-Feedback loop of InstructGPT and ChatGPT**.

### 한국어
PPO(Proximal Policy Optimization)는 **Trust Region Policy Optimization(TRPO)** 의 안정성과 신뢰성을 **1차(first-order) 최적화** 만으로 구현하는 정책 경사 알고리즘 계열이다. 저자들의 관찰은 명확하다: vanilla policy gradient는 한 번의 rollout 데이터로 여러 gradient step을 밟으면 policy가 수집 분포에서 멀어져 파괴적으로 불안정해지며(importance-sampling 기반 대리 목적이 무의미해짐), TRPO는 이를 KL 제약 2차 스텝으로 해결하지만 구현이 복잡하고 dropout이나 actor-critic 파라미터 공유 같은 현대적 기법과 충돌한다. PPO의 핵심 기여는 **clipped surrogate objective**:
$$ L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t\left[\min\!\left(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right] $$
여기서 $r_t(\theta) = \pi_\theta(a_t\mid s_t)/\pi_{\theta_\text{old}}(a_t\mid s_t)$ 이고 $\epsilon=0.2$ 가 표준이다. clip은 비율이 $1\pm\epsilon$ 을 유리하게 벗어나지 못하게 하고, 바깥의 $\min$ 은 원 목적 함수에 대한 pessimistic **하한(lower bound)** 을 보장한다. 전체 알고리즘은 $N$ 개의 병렬 actor가 각각 $T=2048$ timestep 데이터를 수집한 뒤 그 $NT$ transition들에 대해 $K=10$ epoch의 minibatch SGD/Adam을 교대 실행하며, GAE로 advantage를 추정하고 선택적 entropy 보너스를 더한다. 대안으로 adaptive KL penalty 변형을 제시하지만 성능이 떨어진다. **7개 MuJoCo 연속 제어 벤치마크** (100만 timestep)에서 $\epsilon=0.2$ PPO가 최고 정규화 점수(**0.82** vs 최상의 KL 변형 0.74, clip 없음 baseline $-0.39$)를 달성하고, **49개 Atari 게임** 에서 fast-learning 지표 기준 **30게임** 승리(ACER 18, A2C 1)를 기록했다. 논문은 10페이지에 불과하지만 policy-gradient 딥 RL의 지형을 재편했다: PPO는 OpenAI Five, OpenAI 로봇 손, 가장 중요하게는 **InstructGPT와 ChatGPT의 RLHF 루프** 의 기본 알고리즘이 되었다.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction — The Gap Between Vanilla PG and TRPO / 서론 — 일반 PG와 TRPO 사이의 공백

**English**: The paper opens by taxonomizing deep-RL approaches into three leading contenders: deep Q-learning (Mnih 2015), vanilla policy gradient (Mnih 2016), and trust-region / natural policy gradient (Schulman 2015). Each has a weakness: Q-learning fails on continuous control and is poorly understood, vanilla PG has poor data efficiency and robustness, and TRPO is "relatively complicated" and incompatible with architectures that include noise (dropout) or parameter sharing between policy and value. The paper's explicit goal is to retain TRPO's data efficiency and reliability while using only first-order optimisation. The core invention is an objective with **clipped probability ratios**, which forms a pessimistic lower bound on the policy's performance. The training loop alternates between sampling and performing several epochs of minibatch optimisation on the sampled data.

**한국어**: 논문은 딥 RL의 세 주력 접근—deep Q-learning (Mnih 2015), vanilla policy gradient (Mnih 2016), trust-region / natural policy gradient (Schulman 2015)—을 분류하며 시작한다. 각각의 약점은 분명하다: Q-learning은 연속 제어에 실패하고 이론적 이해가 부족하며, vanilla PG는 데이터 효율과 안정성이 낮고, TRPO는 "상대적으로 복잡"하고 dropout이나 정책-가치 공유 아키텍처와 호환되지 않는다. 논문의 명시적 목표는 TRPO의 데이터 효율과 신뢰성을 유지하면서 1차 최적화만 사용하는 것이다. 핵심 발명은 **clipped probability ratio** 기반 목적 함수로, 이는 정책 성능의 pessimistic 하한을 형성한다. 학습 루프는 샘플링과 샘플 데이터에 대한 여러 epoch의 minibatch 최적화를 교대한다.

### Part II: §2 Background — Policy Gradients and Trust-Region Methods / 배경 — 정책 경사와 신뢰 영역 방법

**English**: §2.1 reviews **policy gradients**:
$$ \hat{g} = \hat{\mathbb{E}}_t[\nabla_\theta \log \pi_\theta(a_t\mid s_t)\, \hat{A}_t], \qquad L^{PG}(\theta) = \hat{\mathbb{E}}_t[\log \pi_\theta(a_t\mid s_t)\, \hat{A}_t] $$
Autodiff systems differentiate $L^{PG}$ to recover $\hat{g}$. **The critical warning**: while tempting, optimising $L^{PG}$ for multiple steps on the same trajectory is "not well-justified and empirically often leads to destructively large policy updates." This single paragraph motivates the entire paper — the objective needs to *degrade gracefully* when re-used, and $L^{PG}$ does not.

§2.2 reviews **TRPO**:
$$ \max_\theta\ \hat{\mathbb{E}}_t\!\left[\tfrac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_\text{old}}(a_t\mid s_t)}\hat{A}_t\right]\quad\text{s.t.}\quad \hat{\mathbb{E}}_t[\text{KL}[\pi_{\theta_\text{old}}(\cdot\mid s_t), \pi_\theta(\cdot\mid s_t)]] \le \delta. $$
Solved with conjugate gradient + linear approximation to the objective + quadratic approximation to the constraint. The paper notes that TRPO theory actually suggests the penalty form (equation 5)
$$ \max_\theta\ \hat{\mathbb{E}}_t[r_t(\theta)\hat{A}_t - \beta\,\text{KL}[\pi_{\theta_\text{old}}, \pi_\theta]] $$
but empirically picking a fixed $\beta$ is hard — hence the hard-constraint form in practice. PPO's adaptive-KL variant in §4 is, in a sense, "doing TRPO's penalty form right."

**한국어**: §2.1은 **정책 경사** 를 리뷰한다:
$$ \hat{g} = \hat{\mathbb{E}}_t[\nabla_\theta \log \pi_\theta(a_t\mid s_t)\, \hat{A}_t], \qquad L^{PG}(\theta) = \hat{\mathbb{E}}_t[\log \pi_\theta(a_t\mid s_t)\, \hat{A}_t] $$
자동미분 시스템은 $L^{PG}$ 를 미분해 $\hat{g}$ 를 얻는다. **결정적 경고**: 같은 trajectory로 $L^{PG}$ 를 여러 step 최적화하는 것은 "이론적으로 정당화되지 않으며 실험적으로 파괴적인 대형 정책 업데이트를 일으킨다." 이 한 문단이 논문 전체의 동기다 — 목적 함수가 *재사용될 때 우아하게 퇴화* 해야 하는데 $L^{PG}$ 는 그렇지 않다.

§2.2는 **TRPO** 를 리뷰한다:
$$ \max_\theta\ \hat{\mathbb{E}}_t\!\left[\tfrac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_\text{old}}(a_t\mid s_t)}\hat{A}_t\right]\quad\text{s.t.}\quad \hat{\mathbb{E}}_t[\text{KL}[\pi_{\theta_\text{old}}, \pi_\theta]] \le \delta. $$
Conjugate gradient + 목적의 선형 근사 + 제약의 2차 근사로 해결. TRPO 이론은 실제로 penalty 형식(식 5)
$$ \max_\theta\ \hat{\mathbb{E}}_t[r_t(\theta)\hat{A}_t - \beta\,\text{KL}] $$
을 제시하지만 고정 $\beta$ 를 고르기가 어려워 실제로는 hard constraint를 쓴다. PPO의 §4 adaptive-KL 변형은 어떤 의미에서 "TRPO의 penalty 형식을 제대로 하는 것"이다.

### Part III: §3 Clipped Surrogate Objective — The Heart of the Paper / 클립된 대리 목적 — 논문의 심장

**English**: Let $r_t(\theta) = \pi_\theta(a_t\mid s_t)/\pi_{\theta_\text{old}}(a_t\mid s_t)$, so $r_t(\theta_\text{old}) = 1$. TRPO's "CPI" (Conservative Policy Iteration) objective is
$$ L^{CPI}(\theta) = \hat{\mathbb{E}}_t[r_t(\theta)\hat{A}_t]. $$
Without any constraint, $L^{CPI}$ can be driven arbitrarily positive, causing destructive updates. The paper's proposal:
$$ L^{CLIP}(\theta) = \hat{\mathbb{E}}_t\!\left[\min\!\left(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]. $$

The authors spell out the geometry (Figure 1 of the paper):
- **When $\hat{A}_t > 0$** (action was above-average): the objective $r_t \hat{A}_t$ increases with $r_t$. Clipping caps it at $r_t = 1+\epsilon$, so further gradient vanishes beyond that — no incentive to push probability ratios above $1+\epsilon$.
- **When $\hat{A}_t < 0$** (action was below-average): the objective decreases with $r_t$. Clipping caps it at $r_t = 1-\epsilon$, so further gradient vanishes below that — no incentive to push probabilities below $1-\epsilon$.
- The outer $\min$ means: if clipping would make the objective **better**, ignore it (keep the unclipped term as the bound); if clipping would make it **worse**, use the clipped term. Result: $L^{CLIP} \le L^{CPI}$ everywhere — $L^{CLIP}$ is a **pessimistic lower bound**.

**Figure 2 interpretation**: as $\theta$ interpolates from $\theta_\text{old}$ (factor 0) to the post-update parameters (factor 1), $L^{CPI}$ grows monotonically (unbounded optimism), the raw clip term flattens, and $L^{CLIP}$ **rises then decays** — peaking near the actual update that produces a KL of about $0.02$. This peak coincides with a sensible update size.

**한국어**: $r_t(\theta) = \pi_\theta(a_t\mid s_t)/\pi_{\theta_\text{old}}(a_t\mid s_t)$ 이므로 $r_t(\theta_\text{old}) = 1$. TRPO의 "CPI"(Conservative Policy Iteration) 목적은
$$ L^{CPI}(\theta) = \hat{\mathbb{E}}_t[r_t(\theta)\hat{A}_t] $$
이며, 제약 없이는 무한정 커져 파괴적 업데이트를 야기한다. 논문의 제안:
$$ L^{CLIP}(\theta) = \hat{\mathbb{E}}_t\!\left[\min\!\left(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]. $$

저자들은 기하학적 해석(Figure 1)을 명확히 제시한다:
- **$\hat{A}_t > 0$** 일 때(평균 이상 행동): 목적은 $r_t$ 증가 시 증가. $r_t = 1+\epsilon$ 에서 clip되어 그 이상에서 gradient가 0 — 확률 비를 $1+\epsilon$ 위로 밀어올릴 유인 없음.
- **$\hat{A}_t < 0$** 일 때(평균 이하 행동): 목적은 $r_t$ 증가 시 감소. $r_t = 1-\epsilon$ 에서 clip되어 그 아래에서 gradient가 0 — 확률을 $1-\epsilon$ 아래로 내릴 유인 없음.
- 바깥 $\min$ 의 의미: clipping이 목적을 **더 좋게** 만들면 무시(unclipped항 유지); **더 나쁘게** 만들면 clipped항 사용. 결과: $L^{CLIP} \le L^{CPI}$ 가 모든 곳에서 성립 — $L^{CLIP}$ 은 **pessimistic 하한**.

**Figure 2 해석**: $\theta$ 가 $\theta_\text{old}$(factor 0)에서 update 후 파라미터(factor 1)로 보간됨에 따라 $L^{CPI}$ 는 단조 증가(무한정 낙관), 순 clip 항은 평평해지고, $L^{CLIP}$ 은 **올랐다가 감소** — KL이 약 $0.02$ 인 실제 업데이트 지점에서 최대. 이 peak이 합리적인 업데이트 크기와 일치.

### Part IV: §4 Adaptive KL Penalty — The Alternative That Loses / 적응적 KL 페널티 — 지는 대안

**English**: A second variant computes gradients on
$$ L^{KLPEN}(\theta) = \hat{\mathbb{E}}_t[r_t(\theta)\hat{A}_t - \beta\,\text{KL}[\pi_{\theta_\text{old}}, \pi_\theta]], $$
and adjusts $\beta$ after each update based on the realized KL divergence:
- Compute $d = \hat{\mathbb{E}}_t[\text{KL}[\pi_{\theta_\text{old}}, \pi_\theta]]$.
- If $d < d_\text{targ}/1.5$, halve $\beta$.
- If $d > d_\text{targ}\times 1.5$, double $\beta$.

The constants $1.5$ and $2$ are heuristic but "not very sensitive." The punchline: KL penalty **underperforms the clipped objective** empirically (Table 1: best adaptive KL = 0.74 vs. clip $\epsilon=0.2$ = **0.82**). The paper includes KL-penalty for completeness and as a baseline — history remembers the clip variant.

**한국어**: 두 번째 변형은
$$ L^{KLPEN}(\theta) = \hat{\mathbb{E}}_t[r_t(\theta)\hat{A}_t - \beta\,\text{KL}[\pi_{\theta_\text{old}}, \pi_\theta]] $$
의 gradient를 쓰며, 업데이트 후 realized KL에 따라 $\beta$ 를 조정한다:
- $d = \hat{\mathbb{E}}_t[\text{KL}]$ 계산.
- $d < d_\text{targ}/1.5$ 이면 $\beta$ 를 절반으로.
- $d > d_\text{targ}\times 1.5$ 이면 $\beta$ 를 2배로.

상수 $1.5, 2$ 는 휴리스틱이지만 민감하지 않다. 결론: KL penalty는 **clip 목적에 지속적으로 패배** (Table 1: 최상의 adaptive KL = 0.74 vs clip $\epsilon=0.2$ = **0.82**). 논문은 완전성과 baseline으로 KL penalty를 포함시켰지만, 역사는 clip 변형을 기억한다.

### Part V: §5 Full Algorithm — PPO Actor-Critic / 전체 알고리즘 — PPO Actor-Critic

**English**: For the actor-critic style with shared policy/value parameters, the objective combines three terms:
$$ L_t^{CLIP+VF+S}(\theta) = \hat{\mathbb{E}}_t\!\left[L_t^{CLIP}(\theta) - c_1\, L_t^{VF}(\theta) + c_2\, S[\pi_\theta](s_t)\right] $$
where $L_t^{VF}(\theta) = (V_\theta(s_t) - V_t^\text{targ})^2$ and $S[\pi_\theta]$ is the policy entropy. Standard coefficients: $c_1 = 1$, $c_2 = 0.01$.

**GAE advantage estimator** (truncated to horizon $T$):
$$ \hat{A}_t = \delta_t + (\gamma\lambda)\delta_{t+1} + \cdots + (\gamma\lambda)^{T-t+1}\delta_{T-1}, \qquad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t). $$
Reducing to $\lambda=1$ recovers the $T$-step Monte-Carlo-minus-baseline estimator $\hat{A}_t = -V(s_t) + r_t + \gamma r_{t+1} + \cdots + \gamma^{T-t+1} V(s_T)$.

**Algorithm 1**:
```
for iteration = 1, 2, ... do
    for actor = 1, 2, ..., N do
        Run policy π_{θ_old} for T timesteps
        Compute advantage estimates Â_1, ..., Â_T
    end for
    Optimize L w.r.t. θ with K epochs and minibatch size M ≤ NT
    θ_old ← θ
end for
```
The loop's structure is unusual: data is on-policy at collection time but off-policy during the $K$-epoch training pass — the clip keeps the drift bounded.

**한국어**: 정책/가치 공유 파라미터의 actor-critic 스타일에서 목적 함수는 세 항의 결합:
$$ L_t^{CLIP+VF+S}(\theta) = \hat{\mathbb{E}}_t\!\left[L_t^{CLIP}(\theta) - c_1\, L_t^{VF}(\theta) + c_2\, S[\pi_\theta](s_t)\right] $$
$L_t^{VF}(\theta) = (V_\theta(s_t) - V_t^\text{targ})^2$, $S[\pi_\theta]$ 는 정책 엔트로피. 표준 계수: $c_1 = 1$, $c_2 = 0.01$.

**GAE advantage 추정량** (horizon $T$ 로 절단):
$$ \hat{A}_t = \delta_t + (\gamma\lambda)\delta_{t+1} + \cdots + (\gamma\lambda)^{T-t+1}\delta_{T-1}, \qquad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t). $$
$\lambda=1$ 이면 $T$-step MC-minus-baseline 추정량으로 환원.

**Algorithm 1**: 반복마다 $N$ actor가 각각 $T$ timestep 수집 → advantage 계산 → minibatch size $M \le NT$ 로 $K$ epoch SGD → $\theta_\text{old} \leftarrow \theta$. 루프 구조의 특이점: 수집 시점엔 on-policy이지만 $K$-epoch 학습 중엔 off-policy가 된다 — clip이 drift를 제한한다.

### Part VI: §6 Experiments / 실험

**English**: Four experimental subsections, each designed to isolate one question.

**§6.1 Surrogate Objective Ablation (Table 1)** — on 7 MuJoCo tasks, 3 seeds each, 21 runs, scores normalized so random = 0, best = 1. Results:

| Setting | Avg. Normalized Score |
|---|---|
| No clipping or penalty ($L^{CPI}$) | $-0.39$ |
| Clipping $\epsilon=0.1$ | 0.76 |
| **Clipping $\epsilon=0.2$** | **0.82** (best) |
| Clipping $\epsilon=0.3$ | 0.70 |
| Adaptive KL $d_\text{targ}=0.003$ | 0.68 |
| Adaptive KL $d_\text{targ}=0.01$ | 0.74 |
| Adaptive KL $d_\text{targ}=0.03$ | 0.71 |
| Fixed KL $\beta=0.3$ | 0.62 |
| Fixed KL $\beta=1$ | 0.71 |
| Fixed KL $\beta=3$ | 0.72 |
| Fixed KL $\beta=10$ | 0.69 |

The un-clipped baseline is **below random** — proof that multi-epoch optimisation on $L^{CPI}$ really is destructive. Log-space clipping was tried and did no better. This table is the empirical core of the paper.

**§6.2 Continuous Control (Figure 3)** — on the same 7 MuJoCo environments, compared against A2C, A2C+TrustRegion, CEM, TRPO, vanilla-PG-with-adaptive-stepsize, all run 1M timesteps. PPO is best or tied-for-best on 6 of 7 (HalfCheetah, Hopper, InvertedDoublePendulum, Reacher, Swimmer, Walker2d) and competitive on InvertedPendulum. Policy: 2-hidden-layer MLP, 64 units, tanh nonlinearities, Gaussian output.

**§6.3 3D Humanoid (Figure 4, 5)** — Roboschool Humanoid, HumanoidFlagrun, HumanoidFlagrunHarder. PPO trains stable locomotion policies to ~2,500–4,000 reward over 50–100M timesteps. Figure 5 shows still frames of the learned policy running toward a target and then re-orienting when the target is randomly moved. Large-scale continuous control demonstration.

**§6.4 Atari (Table 2)** — 49 games, PPO vs. well-tuned A2C and ACER. Two metrics:
| Metric | A2C | ACER | PPO | Tie |
|---|---|---|---|---|
| Avg episode reward **over all training** (fast learning) | 1 | 18 | **30** | 0 |
| Avg episode reward **over last 100 episodes** (final performance) | 1 | **28** | 19 | 1 |

PPO is best on sample efficiency (games won under the "all-training" metric), ACER edges out on final performance. The paper frames this as a favourable sample-complexity + simplicity trade-off; PPO requires no replay buffer, no off-policy correction machinery.

**한국어**: 실험은 네 소섹션으로, 각각 한 가지 질문만 분리한다.

**§6.1 대리 목적 Ablation (Table 1)** — 7개 MuJoCo task, 각 3 seed, 21 runs, 점수 정규화(random=0, best=1). 결과(위 표). un-clipped baseline은 **random 이하** — 다중 epoch $L^{CPI}$ 최적화가 실제로 파괴적임의 증거. 로그 공간 clipping도 시도했지만 나아지지 않았다. 이 표가 논문의 실증적 중심.

**§6.2 연속 제어 (Figure 3)** — 같은 7개 MuJoCo, A2C, A2C+Trust Region, CEM, TRPO, vanilla-PG-with-adaptive-stepsize 와 비교 (1M timestep). PPO가 7개 중 6개(HalfCheetah, Hopper, InvertedDoublePendulum, Reacher, Swimmer, Walker2d)에서 최고/최고-동률, InvertedPendulum에서 경쟁력. 정책: 2-layer MLP, 64 unit, tanh, Gaussian 출력.

**§6.3 3D Humanoid (Figure 4, 5)** — Roboschool Humanoid, HumanoidFlagrun, HumanoidFlagrunHarder. PPO가 50–100M timestep으로 reward ~2,500–4,000의 안정적 locomotion 학습. Figure 5는 목표를 향해 달리다 목표가 무작위 이동하면 재지향하는 정책의 still frame.

**§6.4 Atari (Table 2)** — 49 게임, PPO vs A2C vs ACER. 두 지표: **전체 학습 평균 보상** 기준 PPO **30승** / ACER 18 / A2C 1; **마지막 100 episode 보상** 기준 ACER **28승** / PPO 19 / A2C 1. PPO는 샘플 효율(빠른 학습)에서 우위, ACER가 최종 성능에서 소폭 우위. PPO는 replay buffer도 off-policy 보정 장치도 필요 없다는 단순성 + 샘플 효율 trade-off를 보여줌.

### Part VII: Hyperparameter Tables (Appendix A) / 하이퍼파라미터 표 (부록 A)

**English**: The appendix quietly provides the recipe that practitioners actually memorize.

**MuJoCo (Table 3)**: horizon $T=2048$, Adam step $3\times 10^{-4}$, epochs $K=10$, minibatch $M=64$, $\gamma=0.99$, GAE $\lambda=0.95$.

**Roboschool (Table 4)**: $T=512$, adaptive Adam step, $K=15$, minibatch $M=4096$, $\gamma=0.99$, $\lambda=0.95$, $N=32$ actors for locomotion / $N=128$ for flagrun, log-stdev linearly annealed from $-0.7$ to $-1.6$.

**Atari (Table 5)**: $T=128$, Adam step $2.5\times 10^{-4}\times \alpha$, $K=3$, minibatch $32\times 8$, $N=8$ actors, $\epsilon=0.1\times\alpha$ (annealed), $c_1=1$, $c_2=0.01$. Here $\alpha$ linearly anneals from $1$ to $0$ over training.

These specific numbers (especially $T, K, M, \epsilon$) became the unofficial defaults that hundreds of later papers inherit.

**한국어**: 부록이 실무자들이 실제로 외우는 레시피를 조용히 제공한다.

**MuJoCo (Table 3)**: horizon $T=2048$, Adam step $3\times 10^{-4}$, epochs $K=10$, minibatch $M=64$, $\gamma=0.99$, GAE $\lambda=0.95$.

**Roboschool (Table 4)**: $T=512$, adaptive Adam step, $K=15$, minibatch $M=4096$, locomotion $N=32$ / flagrun $N=128$, log-stdev $-0.7$ → $-1.6$ 선형 감소.

**Atari (Table 5)**: $T=128$, Adam step $2.5\times 10^{-4}\times\alpha$, $K=3$, minibatch $32\times 8$, actor $N=8$, $\epsilon=0.1\times\alpha$ (감소), $c_1=1$, $c_2=0.01$. $\alpha$ 는 학습 동안 1 → 0 선형 감소.

이 구체적 숫자($T, K, M, \epsilon$)들이 수백편의 후속 논문이 물려받은 비공식 기본값이 된다.

### Part VIII: §7 Conclusion — A Minimalist Claim / 결론 — 미니멀리스트의 주장

**English**: The conclusion is unusually short and specific: PPO is a family of policy-optimization methods that (a) use multiple epochs of stochastic gradient ascent per update, (b) match trust-region methods' stability and reliability, (c) require only a few lines of code change from vanilla PG, (d) work in joint policy-value architectures, and (e) have better overall performance. No grandiose claims about AGI, no hand-wavy future work. The paper lets its experiments do the arguing.

**한국어**: 결론은 이례적으로 짧고 구체적이다: PPO는 (a) 업데이트당 SGD 여러 epoch을 사용하고, (b) trust-region 방법의 안정성과 신뢰성에 필적하며, (c) vanilla PG에서 몇 줄의 코드 변경만 필요하고, (d) 정책-가치 공유 아키텍처에서 동작하며, (e) 전반적으로 더 나은 성능을 갖는 정책 최적화 방법 계열이다. AGI에 대한 거창한 주장도, 모호한 future work도 없다. 실험이 대신 논증한다.

---

## 3. Key Takeaways / 핵심 시사점

1. **Clipped surrogate = cheap first-order trust region / Clipped surrogate는 값싼 1차 trust region이다** — The whole value proposition is collapsing TRPO's KL-constrained quadratic problem into a clip operation inside the surrogate. The clip is a differentiable proxy for "don't move too far." No conjugate gradient, no Fisher matrix, no line search — just `torch.clamp(ratio, 1-eps, 1+eps)`. / 전체 가치 제안은 TRPO의 KL 제약 2차 문제를 대리 목적 내부의 clip 연산으로 축소하는 것. clip은 "너무 멀리 가지 마라"의 미분 가능한 프록시. conjugate gradient 없음, Fisher 행렬 없음, line search 없음 — 그저 `torch.clamp(ratio, 1-eps, 1+eps)`.

2. **The outer $\min$ is what makes the clip correct / 바깥쪽 $\min$ 이 clip을 올바르게 만든다** — Merely clipping $r_t$ is not enough; you also need the $\min$ so the objective becomes a **lower bound** rather than a two-sided truncation that could introduce new optima. When clipping would improve the objective, we ignore it; when it would hurt, we use it. The asymmetry is the whole point: no incentive to over-shoot, always an incentive to not under-shoot. / $r_t$ 를 단순히 clip하는 것만으론 부족; $\min$ 이 있어야 목적이 새 최적을 도입하는 양방향 절단이 아닌 **하한(lower bound)** 이 된다. clip이 목적을 개선하면 무시, 해치면 사용. 비대칭성이 핵심: over-shoot 유인 없음, under-shoot 방지 유인은 항상 있음.

3. **Multiple epochs on the same rollout is where PPO wins / 같은 rollout에 대한 여러 epoch이 PPO의 승부처** — Vanilla PG gets one gradient step per trajectory before the data is stale (the policy has moved). TRPO gets one CG-approximated step. PPO gets **$K=3$ to $K=15$ SGD epochs**. This is not a marginal efficiency gain — it is a qualitative change in data utilisation, and it is mathematically safe only because the clip prevents $r_t$ from exploding. / Vanilla PG는 trajectory당 한 번의 gradient step 이후 데이터가 stale; TRPO는 CG 근사 step 하나; PPO는 **$K=3\sim15$ SGD epoch**. 한계 효율 향상이 아니라 데이터 활용의 질적 변화, 그리고 clip이 $r_t$ 폭발을 막기에 수학적으로 안전.

4. **The "no clipping" baseline scores $-0.39$ — worse than random / "clip 없음" baseline은 $-0.39$ — 랜덤보다 나쁨** — Buried in Table 1 is a shocking data point: vanilla $L^{CPI}$ with multi-epoch optimisation is not just suboptimal, it is actively destructive — it scores below the initial random policy. In one environment (HalfCheetah) it diverges. This is the quantitative smoking gun that motivates PPO's entire existence. / Table 1에 숨어있는 충격적 데이터: multi-epoch $L^{CPI}$ 는 suboptimal이 아니라 **능동적으로 파괴적** 이며 초기 랜덤 정책보다 점수가 낮다. HalfCheetah에서는 발산. PPO의 존재 이유를 정량적으로 확증.

5. **$\epsilon = 0.2$ is a remarkably robust default / $\epsilon=0.2$ 는 놀랍도록 강건한 기본값** — Table 1 shows $\epsilon=0.1$ (0.76), $\epsilon=0.2$ (0.82), $\epsilon=0.3$ (0.70). The sweet spot is narrow but not knife-edge; $\epsilon=0.2$ is better than $\pm 0.1$ either way by only a few percentage points. This robustness is why the hyperparameter gets copied without thought into RLHF pipelines and LLM-alignment codebases. / Table 1은 $\epsilon=0.1$ (0.76), $\epsilon=0.2$ (0.82), $\epsilon=0.3$ (0.70). sweet spot은 좁지만 극단적이지 않음; $\epsilon=0.2$ 는 $\pm 0.1$ 어느 쪽보다 몇 퍼센트포인트 우위. 이 강건성이 RLHF 파이프라인과 LLM 정렬 코드베이스에 이 하이퍼파라미터가 무작정 복사되는 이유.

6. **Adaptive KL is included as a principled loser / Adaptive KL은 원칙에 충실하지만 지는 대안으로 포함** — The paper is transparent: KL penalty is the theoretically cleaner route (it is literally what TRPO theory suggests), but it loses empirically. The authors publish it anyway. The lesson is cultural: when two methods tie, take the simpler one; when one is more principled but loses by a margin, don't paper over the loss. / 논문의 투명성: KL penalty가 이론적으로 더 깔끔하지만(TRPO 이론이 문자 그대로 제시하는 것) 실험적으로 진다. 저자들은 그래도 발표한다. 문화적 교훈: 두 방법이 비등하면 단순한 쪽, 한쪽이 더 원칙적이지만 명확히 지면 패배를 호도하지 말 것.

7. **PPO is compatible with every modern trick; TRPO is not / PPO는 모든 현대적 트릭과 호환, TRPO는 아님** — A throwaway sentence in §1 turns out to be massively consequential: TRPO breaks with dropout (because the KL computation assumes deterministic forward passes) and with policy/value parameter sharing (because second-order constraint applied to a shared backbone corrupts the value head). PPO has no such sensitivities. This compatibility is why PPO scales to LLMs (where dropout and shared transformer backbones are universal). / §1의 지나가는 한 문장이 엄청난 결과를 낳는다: TRPO는 dropout과 충돌(KL 계산이 결정적 forward pass 가정) 및 정책/가치 파라미터 공유와 충돌(공유 backbone에 2차 제약이 value head를 오염). PPO는 이런 민감성이 없음. 이 호환성이 LLM으로의 확장(dropout과 공유 transformer backbone이 보편적)을 가능케 함.

8. **The single line of PyTorch that changed RL / RL을 바꾼 PyTorch 한 줄** — In implementation, the entire PPO trick is `loss = -torch.min(ratio * adv, torch.clamp(ratio, 1-eps, 1+eps) * adv).mean()`. That single expression, slotted into a vanilla actor-critic loop, is why PPO proliferated. The core idea fits in a tweet; the experiments fit in 10 pages; the impact fits in the history of AI. / 구현상 PPO의 전체 트릭은 `loss = -torch.min(ratio * adv, torch.clamp(ratio, 1-eps, 1+eps) * adv).mean()` 한 줄. vanilla actor-critic 루프에 끼워넣는 이 한 표현이 PPO 확산의 이유. 핵심 아이디어는 트윗 한 줄, 실험은 10페이지, 영향은 AI 역사 전체.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Policy gradient estimator / 정책 경사 추정량

$$ \hat{g} = \hat{\mathbb{E}}_t[\nabla_\theta \log \pi_\theta(a_t\mid s_t)\, \hat{A}_t] $$

- Obtained by differentiating $L^{PG}(\theta) = \hat{\mathbb{E}}_t[\log \pi_\theta(a_t\mid s_t) \hat{A}_t]$.
- Unsafe to multi-epoch: the policy drifts and $L^{PG}$ becomes non-meaningful.
- $L^{PG}$ 를 여러 epoch 돌리면 정책이 분포 이동되어 의미를 잃음.

### 4.2 TRPO constrained surrogate / TRPO 제약 대리 목적

$$ \max_\theta\ \hat{\mathbb{E}}_t[r_t(\theta)\hat{A}_t] \quad\text{s.t.}\quad \hat{\mathbb{E}}_t[\text{KL}[\pi_{\theta_\text{old}}, \pi_\theta]] \le \delta $$

- $r_t(\theta) = \pi_\theta(a_t\mid s_t) / \pi_{\theta_\text{old}}(a_t\mid s_t)$.
- Solved with conjugate gradient + linear/quadratic approximations.
- Theoretical equivalent: penalty form $\max_\theta[r_t \hat{A}_t - \beta\text{KL}]$.
- CG + 선형/2차 근사로 해결; 이론상은 $\beta$ penalty 형태와 등가.

### 4.3 PPO clipped objective (equation 7 of the paper) / PPO 클립 목적 (논문 식 7)

$$ L^{CLIP}(\theta) = \hat{\mathbb{E}}_t\!\left[\min\!\left(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right] $$

- Default $\epsilon = 0.2$.
- $L^{CLIP} \le L^{CPI}$ pointwise → pessimistic lower bound.
- First-order equivalent to $L^{CPI}$ near $\theta_\text{old}$ (identical derivatives).
- Clip 과 unclip 의 $\min$ 으로 hard trust region 대체.

### 4.4 PPO adaptive-KL objective / 적응적 KL 변형

$$ L^{KLPEN}(\theta) = \hat{\mathbb{E}}_t[r_t(\theta)\hat{A}_t - \beta \text{KL}[\pi_{\theta_\text{old}}, \pi_\theta]] $$

$$ \beta \leftarrow
\begin{cases}
\beta / 2 & \text{if } d < d_\text{targ}/1.5 \\
2\beta & \text{if } d > 1.5\, d_\text{targ} \\
\beta & \text{otherwise}
\end{cases} $$

- Empirically weaker than clipping.
- $d_\text{targ}$ 는 목표 KL; $\beta$ 는 매 업데이트 후 rule-based 조정.

### 4.5 GAE advantage estimator / GAE advantage 추정량

$$ \hat{A}_t = \sum_{l=0}^{T-t-1}(\gamma\lambda)^l \delta_{t+l}, \qquad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t) $$

- $\lambda = 0$: one-step TD (low variance, high bias).
- $\lambda = 1$: Monte-Carlo minus baseline (zero bias, high variance).
- Paper uses $\lambda = 0.95$, $\gamma = 0.99$.
- $\lambda$ 로 TD(0)와 MC 사이 bias-variance 조절.

### 4.6 Full actor-critic objective / 전체 actor-critic 목적

$$ L^{CLIP+VF+S}(\theta) = \hat{\mathbb{E}}_t[L^{CLIP}_t(\theta) - c_1 L^{VF}_t(\theta) + c_2 S[\pi_\theta](s_t)] $$

$$ L^{VF}_t = (V_\theta(s_t) - V_t^\text{targ})^2 $$

- $c_1 = 1$, $c_2 = 0.01$ (Atari defaults).
- Entropy bonus $S$ encourages exploration.
- $S$ 는 탐험 장려.

### 4.7 Worked numerical example / 수치 예시

Suppose at some $(s_t, a_t)$ we have:
- Old policy probability $\pi_\text{old}(a_t\mid s_t) = 0.30$
- Current policy probability $\pi_\theta(a_t\mid s_t) = 0.42$
- Advantage $\hat{A}_t = +1.5$ (good action)
- $\epsilon = 0.2$

Then $r_t = 0.42 / 0.30 = 1.40$.

- Unclipped term: $r_t \hat{A}_t = 1.40 \times 1.5 = 2.10$
- Clipped ratio: $\text{clip}(1.40, 0.8, 1.2) = 1.20$
- Clipped term: $1.20 \times 1.5 = 1.80$
- $L^{CLIP} = \min(2.10, 1.80) = 1.80$ ← clip is binding

Because $\hat{A}_t > 0$ and $r_t > 1+\epsilon$, the clip is active and the gradient w.r.t. $\theta$ through this term is zero — no more incentive to push the probability of this good action further up.

Now suppose $\pi_\theta(a_t\mid s_t) = 0.20$, so $r_t = 0.20/0.30 = 0.67$, same $\hat{A}_t = +1.5$.

- Unclipped: $0.67 \times 1.5 = 1.00$
- Clip: $\text{clip}(0.67, 0.8, 1.2) \cdot 1.5 = 0.80 \times 1.5 = 1.20$
- $L^{CLIP} = \min(1.00, 1.20) = 1.00$ ← unclipped is binding (smaller)

Here clipping would make the objective better (1.20 > 1.00), so the $\min$ picks the smaller unclipped value — we still have a gradient encouraging the policy to raise this action's probability.

**한국어**: 같은 국면에서 old 확률 0.30, 새 확률 0.42, advantage $+1.5$, $\epsilon=0.2$ 라면 $r_t = 1.40$, clip은 1.20에서 활성, $L^{CLIP} = 1.80$ — clip 적용되어 gradient 0. 반대로 새 확률이 0.20으로 떨어지면 $r_t=0.67$, $L^{CLIP} = \min(1.00, 1.20) = 1.00$ — unclipped이 작으므로 선택되고 gradient는 여전히 확률을 올리도록 유도. 비대칭성의 작동 예시.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1992 ─ REINFORCE (Williams): the original policy-gradient estimator
1999 ─ Sutton et al.: Policy Gradient Theorem
2002 ─ Kakade: Natural Policy Gradient (Fisher-information update)
2002 ─ Kakade & Langford: Conservative Policy Iteration (CPI) — origin of L^CPI
2013 ─ DQN (Mnih et al., Atari)                                  ← #22 our list
2015 ─ TRPO (Schulman et al.) — theoretical monotonic improvement
2016 ─ GAE (Schulman et al.) — λ-weighted advantage estimator
2016 ─ A3C / A2C (Mnih et al.) — async advantage actor-critic
2016 ─ ACER (Wang et al.) — off-policy actor-critic with replay
2017 Jul PPO (Schulman et al.)                                   ← THIS PAPER
2017 ─ IMPALA (Espeholt) — scalable actor-critic with V-trace
2018 ─ OpenAI Five (Dota 2) trained with PPO
2019 ─ OpenAI robotic hand / Rubik's cube — PPO at scale
2020 ─ Engstrom et al.: "Implementation Matters in Deep Policy Gradients"
2022 ─ InstructGPT / ChatGPT — PPO for RLHF
2023 ─ DPO (Rafailov et al.) — reformulates RLHF without explicit PPO
2024 ─ GRPO (DeepSeek), RLOO — PPO simplifications tailored for LLMs
```

**한국어 요약**: PPO는 (1) 1992 REINFORCE, (2) 2002 CPI/NPG의 이론 토대, (3) 2015 TRPO의 trust region 사고, (4) 2016 GAE의 advantage 추정을 단 하나의 clip 연산으로 통합한다. 이후 OpenAI Five, 로봇 손, **ChatGPT의 RLHF**를 거치며 딥 RL의 표준이 되고, 2023–24년의 DPO/GRPO/RLOO 같은 LLM 특화 변형의 기준점이 된다.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **#22 Mnih et al. 2015 — Human-level control through deep reinforcement learning (DQN)** | DQN is the value-based counterpart; PPO is the policy-based one. Both are the canonical deep-RL starting points. / DQN은 값 기반, PPO는 정책 기반 — 딥 RL의 두 정통 출발점. | PPO's Atari experiments (§6.4) directly benchmark against the same environments DQN established. / PPO의 Atari 실험은 DQN이 확립한 동일 환경에서 벤치마크. |
| **Schulman et al. 2015 — Trust Region Policy Optimization (TRPO)** | Direct predecessor by the same first author; PPO is its first-order simplification. / 같은 제1저자의 직접 선행작; PPO는 그 1차 최적화 단순화. | The surrogate $L^{CPI}$ comes from here; PPO replaces the hard KL constraint with clip. / 대리 목적 $L^{CPI}$ 는 여기서 유래; PPO는 hard KL 제약을 clip으로 대체. |
| **Schulman et al. 2016 — High-Dimensional Continuous Control Using GAE** | Provides the advantage estimator PPO uses internally. / PPO가 내부적으로 쓰는 advantage 추정량 제공. | GAE $\lambda=0.95$ appears in every PPO hyperparameter table in this paper. / $\lambda=0.95$ GAE는 이 논문의 모든 하이퍼파라미터 표에 등장. |
| **Mnih et al. 2016 — Asynchronous Methods for Deep Reinforcement Learning (A3C/A2C)** | Established the $N$-actor parallel rollout pattern PPO adopts. / $N$-actor 병렬 rollout 패턴 확립. | Algorithm 1's outer loop structure ($N$ actors × $T$ steps per iteration) is inherited from A2C. / Algorithm 1의 외부 루프 구조는 A2C에서 상속. |
| **Kakade & Langford 2002 — Approximately Optimal Approximate RL (CPI)** | Origin of the Conservative Policy Iteration surrogate, TRPO and PPO's mathematical root. / CPI 대리 목적의 기원, TRPO와 PPO의 수학적 뿌리. | The "CPI" superscript in $L^{CPI}$ is a direct citation. / $L^{CPI}$ 의 "CPI" 위첨자는 이 논문에 대한 직접 인용. |
| **Wang et al. 2016 — ACER (Sample-Efficient Actor-Critic with Experience Replay)** | Off-policy actor-critic competitor on Atari; bests PPO on final performance but loses on fast learning. / 오프폴리시 actor-critic 경쟁자; 최종 성능은 우위, 빠른 학습은 열세. | Table 2 Atari comparison is directly against ACER. / Table 2 Atari 비교는 직접 ACER 대상. |
| **Williams 1992 — Simple Statistical Gradient-Following (REINFORCE)** | The ancestral policy-gradient update $\nabla\log\pi \cdot R$. / 정책 경사 업데이트의 시조 $\nabla\log\pi \cdot R$. | Vanilla PG in PPO's §2.1 is exactly REINFORCE with a baseline. / §2.1의 vanilla PG는 baseline이 있는 REINFORCE 그 자체. |
| **Engstrom et al. 2020 — Implementation Matters in Deep Policy Gradients** | Ex-post analysis showing PPO's success depends heavily on implementation tricks (value clipping, advantage normalization, orthogonal init, LR annealing). / 구현 트릭에 대한 사후 분석. | A cautionary companion — reproducible PPO requires more than equation (7). / 재현 가능한 PPO는 식 (7)만으로는 부족. |
| **Ouyang et al. 2022 — InstructGPT / RLHF** | PPO's most consequential application: aligning LLMs with human preferences via a reward model. / PPO의 가장 중요한 응용: 보상 모델을 통한 LLM 인간 선호 정렬. | The inner RL loop of InstructGPT/ChatGPT is literally PPO on a per-token reward. / InstructGPT/ChatGPT의 내부 RL 루프는 토큰별 보상 위의 PPO 그 자체. |

---

## 7. References / 참고문헌

### Primary paper / 본 논문
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms.* arXiv preprint [arXiv:1707.06347](https://arxiv.org/abs/1707.06347).

### Key references cited by the paper / 본 논문이 인용한 주요 참고문헌
- Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. **Machine Learning, 8**(3–4), 229–256. (REINFORCE)
- Kakade, S. & Langford, J. (2002). Approximately optimal approximate reinforcement learning. *ICML*, 2, 267–274. (Conservative Policy Iteration — source of $L^{CPI}$)
- Schulman, J., Levine, S., Moritz, P., Jordan, M. I. & Abbeel, P. (2015). Trust region policy optimization. *CoRR*, abs/1502.05477. (TRPO — direct predecessor)
- Schulman, J., Moritz, P., Levine, S., Jordan, M. & Abbeel, P. (2015/2016). High-dimensional continuous control using generalized advantage estimation. arXiv:1506.02438. (GAE)
- Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. **Nature, 518**(7540), 529–533. (DQN — Paper #22)
- Mnih, V., et al. (2016). Asynchronous methods for deep reinforcement learning. arXiv:1602.01783. (A3C / A2C)
- Wang, Z., et al. (2016). Sample-efficient actor-critic with experience replay (ACER). arXiv:1611.01224.
- Brockman, G., et al. (2016). OpenAI Gym. arXiv:1606.01540.
- Todorov, E., Erez, T. & Tassa, Y. (2012). MuJoCo: A physics engine for model-based control. *IROS 2012*, 5026–5033.
- Duan, Y., Chen, X., Houthooft, R., Schulman, J. & Abbeel, P. (2016). Benchmarking deep reinforcement learning for continuous control. arXiv:1604.06778.
- Kingma, D. & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv:1412.6980.
- Bellemare, M., Naddaf, Y., Veness, J. & Bowling, M. (2015). The arcade learning environment. *IJCAI 2015*.

### Historical successors / 역사적 후속작 (본 논문 이후)
- Heess, N., et al. (2017). Emergence of locomotion behaviours in rich environments. arXiv:1707.02286. (Concurrent adaptive-KL PPO variant)
- Espeholt, L., et al. (2018). IMPALA: Scalable distributed deep-RL with importance weighted actor-learner architectures. *ICML 2018*.
- OpenAI (2019). Dota 2 with large scale deep reinforcement learning (OpenAI Five). arXiv:1912.06680.
- Engstrom, L., et al. (2020). Implementation matters in deep policy gradients: A case study on PPO and TRPO. *ICLR 2020*.
- Ouyang, L., et al. (2022). Training language models to follow instructions with human feedback (InstructGPT). *NeurIPS 2022*.
- Rafailov, R., et al. (2023). Direct Preference Optimization (DPO): Your language model is secretly a reward model. *NeurIPS 2023*.
- Shao, Z., et al. (2024). DeepSeekMath (GRPO). arXiv:2402.03300.
