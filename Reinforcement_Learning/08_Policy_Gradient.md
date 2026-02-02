# 08. 정책 경사 (Policy Gradient)

**난이도: ⭐⭐⭐⭐ (고급)**

## 학습 목표
- 정책 기반 방법의 장단점 이해
- 정책 경사 정리 (Policy Gradient Theorem) 유도
- REINFORCE 알고리즘 구현
- Baseline을 통한 분산 감소 기법
- Actor-Critic으로의 연결

---

## 1. 가치 기반 vs 정책 기반

### 1.1 비교

| 특성 | 가치 기반 (DQN) | 정책 기반 |
|------|----------------|----------|
| 학습 대상 | Q(s, a) | π(a\|s) |
| 정책 도출 | Q에서 간접 유도 | 직접 학습 |
| 행동 공간 | 이산 (주로) | 이산 + 연속 |
| 확률적 정책 | 어려움 | 자연스러움 |
| 수렴 | 불안정 가능 | 지역 최적 |

### 1.2 정책 기반의 장점

```
1. 연속 행동 공간 처리 가능 (로봇 제어)
2. 확률적 정책 학습 가능 (가위바위보)
3. 정책 공간이 더 단순할 수 있음
4. 더 나은 수렴 보장 (일부 경우)
```

---

## 2. 정책의 파라미터화

### 2.1 소프트맥스 정책 (이산 행동)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscretePolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        logits = self.network(state)
        return F.softmax(logits, dim=-1)

    def get_action(self, state):
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob
```

### 2.2 가우시안 정책 (연속 행동)

```python
class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        features = self.shared(state)
        mean = self.mean_layer(features)
        std = self.log_std.exp()
        return mean, std

    def get_action(self, state):
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob
```

---

## 3. 정책 경사 정리

### 3.1 목표 함수

정책 π_θ의 성능을 최대화:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)]$$

여기서 τ = (s₀, a₀, r₀, s₁, a₁, r₁, ...) 는 궤적(trajectory)

### 3.2 정책 경사 정리

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t \right]$$

**직관적 해석:**
- 좋은 결과(높은 G_t)를 가져온 행동의 확률을 높임
- 나쁜 결과를 가져온 행동의 확률을 낮춤

### 3.3 유도 (Log-derivative trick)

```
∇_θ π(a|s;θ) = π(a|s;θ) · ∇_θ log π(a|s;θ)

따라서:
∇_θ J(θ) = E[R · ∇_θ log π(a|s;θ)]
         = E[∇_θ log π(a|s;θ) · R]
```

---

## 4. REINFORCE 알고리즘

### 4.1 기본 REINFORCE

몬테카를로 정책 경사 방법입니다.

```python
class REINFORCE:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.policy = DiscretePolicy(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma

        # 에피소드 저장
        self.log_probs = []
        self.rewards = []

    def choose_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action, log_prob = self.policy.get_action(state_tensor)
        self.log_probs.append(log_prob)
        return action

    def store_reward(self, reward):
        self.rewards.append(reward)

    def compute_returns(self):
        """할인된 리턴 계산"""
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns)
        # 정규화 (선택적이지만 권장)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def update(self):
        returns = self.compute_returns()

        # 정책 손실 계산
        policy_loss = []
        for log_prob, G in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * G)  # 음수 (경사 상승)

        loss = torch.stack(policy_loss).sum()

        # 업데이트
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 에피소드 데이터 초기화
        self.log_probs = []
        self.rewards = []

        return loss.item()
```

### 4.2 학습 루프

```python
import gymnasium as gym
import numpy as np

def train_reinforce(env_name='CartPole-v1', n_episodes=1000):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = REINFORCE(state_dim, action_dim, lr=1e-3)

    scores = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_reward(reward)
            state = next_state
            total_reward += reward

        # 에피소드 종료 후 업데이트
        loss = agent.update()
        scores.append(total_reward)

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}, Avg Score: {np.mean(scores[-100:]):.2f}")

    return agent, scores
```

---

## 5. Baseline을 통한 분산 감소

### 5.1 분산 문제

REINFORCE의 그래디언트는 높은 분산을 가집니다.

```
Var(∇_θ J) ∝ E[(G - b)²]
```

### 5.2 Baseline 도입

상수 b를 빼도 기대값은 변하지 않지만 분산은 감소합니다.

$$\nabla_\theta J(\theta) = \mathbb{E} \left[ \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot (G_t - b) \right]$$

가장 좋은 baseline: b = V(s)

```python
class REINFORCEWithBaseline:
    def __init__(self, state_dim, action_dim, lr_policy=1e-3, lr_value=1e-3, gamma=0.99):
        self.policy = DiscretePolicy(state_dim, action_dim)
        self.value = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr_policy)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=lr_value)
        self.gamma = gamma

        self.log_probs = []
        self.values = []
        self.rewards = []
        self.states = []

    def choose_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        # 정책에서 행동 샘플링
        action, log_prob = self.policy.get_action(state_tensor)

        # 가치 예측
        value = self.value(state_tensor)

        self.log_probs.append(log_prob)
        self.values.append(value)
        self.states.append(state_tensor)

        return action

    def update(self):
        returns = self.compute_returns()

        values = torch.cat(self.values).squeeze()
        log_probs = torch.stack(self.log_probs)

        # Advantage = Return - Baseline (Value)
        advantages = returns - values.detach()

        # 정책 손실
        policy_loss = -(log_probs * advantages).mean()

        # 가치 손실
        value_loss = F.mse_loss(values, returns)

        # 정책 업데이트
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # 가치 업데이트
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # 초기화
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.states = []

        return policy_loss.item(), value_loss.item()
```

---

## 6. 연속 행동 공간 예제

### 6.1 연속 행동 REINFORCE

```python
class ContinuousREINFORCE:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.policy = GaussianPolicy(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma

        self.log_probs = []
        self.rewards = []

    def choose_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action, log_prob = self.policy.get_action(state_tensor)

        self.log_probs.append(log_prob)
        return action.detach().numpy().squeeze()

    def update(self):
        returns = self.compute_returns()

        policy_loss = []
        for log_prob, G in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * G)

        loss = torch.stack(policy_loss).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.log_probs = []
        self.rewards = []
```

### 6.2 MountainCarContinuous 예제

```python
def train_continuous():
    env = gym.make('MountainCarContinuous-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = ContinuousREINFORCE(state_dim, action_dim, lr=1e-3)

    for episode in range(500):
        state, _ = env.reset()
        total_reward = 0

        while True:
            action = agent.choose_action(state)
            action = np.clip(action, env.action_space.low, env.action_space.high)

            next_state, reward, done, truncated, _ = env.step(action)
            agent.rewards.append(reward)

            state = next_state
            total_reward += reward

            if done or truncated:
                break

        agent.update()
        print(f"Episode {episode + 1}, Reward: {total_reward:.2f}")
```

---

## 7. 고급 기법

### 7.1 엔트로피 정규화

탐색을 장려하기 위해 정책의 엔트로피를 손실에 추가합니다.

```python
def compute_entropy(probs):
    """정책의 엔트로피 계산"""
    return -(probs * probs.log()).sum(dim=-1).mean()

# 손실 함수
total_loss = policy_loss - entropy_coef * entropy
```

### 7.2 Reward Shaping

희소 보상 문제를 해결하기 위한 보상 변환:

```python
def shape_reward(reward, state, next_state, done):
    """보상 형성 예시"""
    # 원래 보상에 추가적인 시그널
    position_reward = abs(next_state[0] - state[0])  # 움직임 장려

    if done and reward > 0:
        bonus = 100  # 목표 달성 보너스
    else:
        bonus = 0

    return reward + 0.1 * position_reward + bonus
```

---

## 8. REINFORCE의 한계

### 8.1 문제점

1. **높은 분산**: 에피소드 전체를 사용하므로 분산이 큼
2. **샘플 비효율**: 에피소드 종료까지 기다려야 함
3. **크레딧 할당**: 어떤 행동이 좋은 결과를 가져왔는지 파악 어려움

### 8.2 해결책 → Actor-Critic

- TD 학습과 정책 경사의 결합
- 부트스트래핑으로 분산 감소
- 스텝마다 업데이트 가능

---

## 요약

| 알고리즘 | 업데이트 시점 | Baseline | 특징 |
|---------|-------------|----------|------|
| REINFORCE | 에피소드 종료 | 없음 | 단순, 높은 분산 |
| REINFORCE + Baseline | 에피소드 종료 | V(s) | 낮은 분산 |
| Actor-Critic | 매 스텝 | V(s) 또는 Q(s,a) | 효율적 |

**핵심 공식:**
```
∇_θ J(θ) = E[∇_θ log π_θ(a|s) · (G - b)]
```

---

## 다음 단계

- [09_Actor_Critic.md](./09_Actor_Critic.md) - Actor-Critic 방법론
