# 09. Actor-Critic 방법론

**난이도: ⭐⭐⭐⭐ (고급)**

## 학습 목표
- Actor-Critic 아키텍처 이해
- Advantage 함수와 GAE 학습
- A2C와 A3C 알고리즘 비교
- PyTorch로 Actor-Critic 구현

---

## 1. Actor-Critic 개요

### 1.1 핵심 아이디어

**Actor**: 정책 π(a|s;θ)를 학습
**Critic**: 가치 함수 V(s;w)를 학습

```
Actor-Critic = Policy Gradient + TD Learning
```

### 1.2 REINFORCE vs Actor-Critic

| REINFORCE | Actor-Critic |
|-----------|--------------|
| 에피소드 종료 후 업데이트 | 매 스텝 업데이트 |
| 실제 리턴 G 사용 | TD Target 사용 |
| 높은 분산 | 낮은 분산, 약간의 편향 |

---

## 2. Advantage 함수

### 2.1 정의

$$A(s, a) = Q(s, a) - V(s)$$

**의미:** 평균보다 얼마나 좋은 행동인가

### 2.2 TD Error를 Advantage로 사용

```
δ_t = r_t + γV(s_{t+1}) - V(s_t)

E[δ_t | s_t, a_t] = Q(s_t, a_t) - V(s_t) = A(s_t, a_t)
```

TD Error는 Advantage의 불편 추정량입니다.

```python
def compute_advantage(rewards, values, next_values, dones, gamma=0.99):
    """1-step Advantage 계산"""
    advantages = []
    for r, v, nv, d in zip(rewards, values, next_values, dones):
        if d:
            advantage = r - v
        else:
            advantage = r + gamma * nv - v
        advantages.append(advantage)
    return advantages
```

---

## 3. A2C (Advantage Actor-Critic)

### 3.1 네트워크 구조

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()

        # 공유 특징 추출
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor (정책)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Critic (가치)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        features = self.shared(state)
        policy = F.softmax(self.actor(features), dim=-1)
        value = self.critic(features)
        return policy, value

    def get_action(self, state):
        policy, value = self.forward(state)
        dist = torch.distributions.Categorical(policy)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value
```

### 3.2 A2C 에이전트

```python
class A2CAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99,
                 value_coef=0.5, entropy_coef=0.01):
        self.network = ActorCritic(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

        self.gamma = gamma
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        # 에피소드 버퍼
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.entropies = []

    def choose_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        policy, value = self.network(state_tensor)

        dist = torch.distributions.Categorical(policy)
        action = dist.sample()

        self.log_probs.append(dist.log_prob(action))
        self.values.append(value)
        self.entropies.append(dist.entropy())

        return action.item()

    def store(self, reward, done):
        self.rewards.append(reward)
        self.dones.append(done)

    def compute_returns(self, next_value):
        """n-step returns 계산"""
        returns = []
        R = next_value

        for r, d in zip(reversed(self.rewards), reversed(self.dones)):
            if d:
                R = 0
            R = r + self.gamma * R
            returns.insert(0, R)

        return torch.tensor(returns)

    def update(self, next_state):
        # 다음 상태의 가치 (부트스트래핑)
        with torch.no_grad():
            _, next_value = self.network(
                torch.FloatTensor(next_state).unsqueeze(0)
            )
            next_value = next_value.item()

        returns = self.compute_returns(next_value)
        values = torch.cat(self.values).squeeze()
        log_probs = torch.stack(self.log_probs)
        entropies = torch.stack(self.entropies)

        # Advantage
        advantages = returns - values.detach()

        # 손실 계산
        actor_loss = -(log_probs * advantages).mean()
        critic_loss = F.mse_loss(values, returns)
        entropy_loss = -entropies.mean()

        total_loss = (actor_loss +
                     self.value_coef * critic_loss +
                     self.entropy_coef * entropy_loss)

        # 업데이트
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()

        # 버퍼 초기화
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.entropies = []

        return actor_loss.item(), critic_loss.item()
```

### 3.3 A2C 학습

```python
import gymnasium as gym
import numpy as np

def train_a2c(env_name='CartPole-v1', n_episodes=1000, n_steps=5):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = A2CAgent(state_dim, action_dim)
    scores = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        step_count = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store(reward, done)
            state = next_state
            total_reward += reward
            step_count += 1

            # n-step 업데이트
            if step_count % n_steps == 0 or done:
                agent.update(next_state)

        scores.append(total_reward)

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}, Avg: {np.mean(scores[-100:]):.2f}")

    return agent, scores
```

---

## 4. GAE (Generalized Advantage Estimation)

### 4.1 n-step Returns 트레이드오프

| n | 편향 | 분산 |
|---|------|------|
| 1 (TD) | 높음 | 낮음 |
| ∞ (MC) | 낮음 | 높음 |

### 4.2 GAE 공식

모든 n-step advantage를 기하급수적으로 가중 평균:

$$A^{GAE}\_t = \sum\_{l=0}^{\infty} (\gamma \lambda)^l \delta\_{t+l}$$

여기서 δ_t = r_t + γV(s_{t+1}) - V(s_t)

```python
def compute_gae(rewards, values, next_values, dones, gamma=0.99, lam=0.95):
    """Generalized Advantage Estimation"""
    advantages = []
    gae = 0

    for t in reversed(range(len(rewards))):
        if dones[t]:
            delta = rewards[t] - values[t]
            gae = delta
        else:
            delta = rewards[t] + gamma * next_values[t] - values[t]
            gae = delta + gamma * lam * gae

        advantages.insert(0, gae)

    return torch.tensor(advantages)
```

### 4.3 GAE 적용 A2C

```python
class A2CWithGAE(A2CAgent):
    def __init__(self, *args, gae_lambda=0.95, **kwargs):
        super().__init__(*args, **kwargs)
        self.gae_lambda = gae_lambda
        self.next_values = []

    def compute_gae_returns(self):
        """GAE 기반 advantage와 returns"""
        values = torch.cat(self.values).squeeze().tolist()
        next_vals = self.next_values

        advantages = []
        gae = 0

        for t in reversed(range(len(self.rewards))):
            if self.dones[t]:
                delta = self.rewards[t] - values[t]
                gae = delta
            else:
                delta = self.rewards[t] + self.gamma * next_vals[t] - values[t]
                gae = delta + self.gamma * self.gae_lambda * gae

            advantages.insert(0, gae)

        advantages = torch.tensor(advantages)
        returns = advantages + torch.tensor(values)

        return advantages, returns
```

---

## 5. A3C (Asynchronous Advantage Actor-Critic)

### 5.1 핵심 아이디어

여러 워커가 병렬로 환경과 상호작용하며 비동기적으로 그래디언트를 업데이트합니다.

```
┌─────────────────────────────────────┐
│          Global Network             │
│         (Shared Parameters)          │
└──────────┬──────────┬───────────────┘
           │          │
     ┌─────┴────┐  ┌──┴─────┐
     │ Worker 1 │  │ Worker 2│  ...
     │   Env 1  │  │  Env 2  │
     └──────────┘  └─────────┘
```

### 5.2 의사 코드

```python
# 각 워커의 동작
def worker(global_network, optimizer, env):
    local_network = copy(global_network)

    while True:
        # 로컬 네트워크로 경험 수집
        trajectory = collect_trajectory(local_network, env)

        # 그래디언트 계산
        loss = compute_loss(trajectory)
        gradients = compute_gradients(loss, local_network)

        # 비동기 업데이트
        apply_gradients(optimizer, global_network, gradients)

        # 로컬 네트워크 동기화
        local_network.load_state_dict(global_network.state_dict())
```

### 5.3 A2C vs A3C

| A2C | A3C |
|-----|-----|
| 동기 업데이트 | 비동기 업데이트 |
| 배치 처리 | 스트림 처리 |
| 더 안정적 | 더 빠름 (병렬) |
| GPU 효율적 | CPU 효율적 |

**현재 권장:** A2C가 GPU에서 더 효율적이므로 많이 사용됨

---

## 6. 연속 행동 공간 Actor-Critic

```python
class ContinuousActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        # 공유 네트워크
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor: 평균과 표준편차
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        features = self.shared(state)
        mean = self.actor_mean(features)
        std = self.actor_log_std.exp()
        value = self.critic(features)
        return mean, std, value

    def get_action(self, state, deterministic=False):
        mean, std, value = self.forward(state)

        if deterministic:
            action = mean
        else:
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()

        return action, value

    def evaluate(self, state, action):
        mean, std, value = self.forward(state)
        dist = torch.distributions.Normal(mean, std)

        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        entropy = dist.entropy().sum(-1, keepdim=True)

        return value, log_prob, entropy
```

---

## 7. 학습 안정화 기법

### 7.1 그래디언트 클리핑

```python
# 그래디언트 노름 클리핑
torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=0.5)

# 그래디언트 값 클리핑
torch.nn.utils.clip_grad_value_(network.parameters(), clip_value=1.0)
```

### 7.2 학습률 스케줄링

```python
from torch.optim.lr_scheduler import LinearLR

scheduler = LinearLR(
    optimizer,
    start_factor=1.0,
    end_factor=0.1,
    total_iters=total_timesteps
)
```

### 7.3 보상 정규화

```python
class RunningMeanStd:
    def __init__(self):
        self.mean = 0
        self.var = 1
        self.count = 1e-4

    def update(self, x):
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = len(x)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean += delta * batch_count / total_count
        self.var = (self.var * self.count + batch_var * batch_count +
                   delta**2 * self.count * batch_count / total_count) / total_count
        self.count = total_count

    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)
```

---

## 8. 실습: LunarLander

```python
def train_lunarlander():
    env = gym.make('LunarLander-v2')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = A2CAgent(
        state_dim, action_dim,
        lr=7e-4, gamma=0.99,
        value_coef=0.5, entropy_coef=0.01
    )

    scores = []
    n_steps = 5

    for episode in range(2000):
        state, _ = env.reset()
        total_reward = 0
        steps = 0

        while True:
            action = agent.choose_action(state)
            next_state, reward, done, truncated, _ = env.step(action)

            agent.store(reward, done or truncated)
            state = next_state
            total_reward += reward
            steps += 1

            if steps % n_steps == 0 or done or truncated:
                agent.update(next_state)

            if done or truncated:
                break

        scores.append(total_reward)

        if (episode + 1) % 100 == 0:
            avg = np.mean(scores[-100:])
            print(f"Episode {episode + 1}, Avg: {avg:.2f}")

            if avg >= 200:
                print("Solved!")
                break

    return agent, scores
```

---

## 요약

| 구성요소 | 역할 | 학습 대상 |
|---------|------|----------|
| Actor | 정책 | θ (정책 파라미터) |
| Critic | 가치 평가 | w (가치 파라미터) |
| Advantage | 행동 품질 측정 | A = Q - V ≈ δ |

**손실 함수:**
```
L = L_actor + c1 * L_critic + c2 * L_entropy
L_actor = -log π(a|s) * A
L_critic = (V - target)²
L_entropy = -Σ π log π
```

---

## 다음 단계

- [10_PPO_TRPO.md](./10_PPO_TRPO.md) - 신뢰 영역 정책 최적화
