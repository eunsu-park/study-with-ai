# 10. PPO와 TRPO

**난이도: ⭐⭐⭐⭐ (고급)**

## 학습 목표
- 정책 업데이트의 안정성 문제 이해
- TRPO의 신뢰 영역 개념 학습
- PPO의 클리핑 메커니즘 이해
- PyTorch로 PPO 구현

---

## 1. 정책 최적화의 문제

### 1.1 큰 업데이트의 위험성

정책 경사에서 너무 큰 업데이트는 성능을 급격히 저하시킬 수 있습니다.

```
θ_new = θ_old + α∇J(θ)

문제: α가 크면 정책이 급격히 변해 학습 불안정
해결: 정책 변화를 제한
```

### 1.2 해결 방향

- **TRPO**: KL divergence로 신뢰 영역 제한 (복잡)
- **PPO**: Clipping으로 간단하게 제한

---

## 2. TRPO (Trust Region Policy Optimization)

### 2.1 목표 함수

새 정책과 이전 정책의 비율을 사용:

$$L^{CPI}(\theta) = \mathbb{E}\left[\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)} A^{\pi_{old}}(s, a)\right]$$

### 2.2 KL Divergence 제약

$$\text{maximize}_\theta \quad L^{CPI}(\theta)$$
$$\text{subject to} \quad \mathbb{E}[D_{KL}(\pi_{\theta_{old}} || \pi_\theta)] \leq \delta$$

### 2.3 TRPO의 문제점

- 2차 미분(Hessian) 계산 필요
- Conjugate gradient 알고리즘 필요
- 구현이 복잡하고 계산 비용이 높음

---

## 3. PPO (Proximal Policy Optimization)

### 3.1 핵심 아이디어

Clipping을 사용하여 정책 비율을 제한합니다.

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$

### 3.2 Clipped 목표 함수

$$L^{CLIP}(\theta) = \mathbb{E}\left[\min(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t)\right]$$

```python
def compute_ppo_loss(ratio, advantage, clip_epsilon=0.2):
    """PPO Clipped 손실"""
    # 클리핑된 비율
    clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)

    # 두 항 중 작은 값 선택
    loss1 = ratio * advantage
    loss2 = clipped_ratio * advantage

    return -torch.min(loss1, loss2).mean()
```

### 3.3 Clipping 직관

```
Advantage > 0 (좋은 행동):
- ratio 증가 → 확률 증가
- 단, ratio > 1+ε 이상은 무시 (급격한 증가 방지)

Advantage < 0 (나쁜 행동):
- ratio 감소 → 확률 감소
- 단, ratio < 1-ε 이하는 무시 (급격한 감소 방지)
```

---

## 4. PPO 전체 구현

### 4.1 PPO 에이전트

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PPONetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()

        # Actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.actor(state), self.critic(state)

    def get_action(self, state, action=None):
        probs, value = self.forward(state)
        dist = torch.distributions.Categorical(probs)

        if action is None:
            action = dist.sample()

        return action, dist.log_prob(action), dist.entropy(), value


class PPOAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        update_epochs=10,
        batch_size=64
    ):
        self.network = PPONetwork(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size

    def collect_rollouts(self, env, n_steps):
        """경험 수집"""
        states, actions, rewards, dones = [], [], [], []
        values, log_probs = [], []

        state, _ = env.reset()

        for _ in range(n_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            with torch.no_grad():
                action, log_prob, _, value = self.network.get_action(state_tensor)

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            states.append(state)
            actions.append(action.item())
            rewards.append(reward)
            dones.append(done)
            values.append(value.item())
            log_probs.append(log_prob.item())

            state = next_state if not done else env.reset()[0]

        # 마지막 상태의 가치
        with torch.no_grad():
            _, _, _, last_value = self.network.get_action(
                torch.FloatTensor(state).unsqueeze(0)
            )

        return {
            'states': np.array(states),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'dones': np.array(dones),
            'values': np.array(values),
            'log_probs': np.array(log_probs),
            'last_value': last_value.item()
        }

    def compute_gae(self, rollout):
        """GAE 계산"""
        rewards = rollout['rewards']
        values = rollout['values']
        dones = rollout['dones']
        last_value = rollout['last_value']

        advantages = np.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]

            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae

        returns = advantages + values
        return advantages, returns

    def update(self, rollout):
        """PPO 업데이트"""
        advantages, returns = self.compute_gae(rollout)

        # 정규화
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 텐서 변환
        states = torch.FloatTensor(rollout['states'])
        actions = torch.LongTensor(rollout['actions'])
        old_log_probs = torch.FloatTensor(rollout['log_probs'])
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)

        # 여러 에폭 업데이트
        for _ in range(self.update_epochs):
            # 미니배치 생성
            indices = np.random.permutation(len(states))

            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                # 현재 정책으로 평가
                _, new_log_probs, entropy, values = self.network.get_action(
                    batch_states, batch_actions
                )

                # 비율 계산
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # Clipped 손실
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # 가치 손실
                critic_loss = F.mse_loss(values.squeeze(), batch_returns)

                # 엔트로피 보너스
                entropy_loss = -entropy.mean()

                # 총 손실
                loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss

                # 업데이트
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

        return actor_loss.item(), critic_loss.item()
```

### 4.2 PPO 학습 루프

```python
import gymnasium as gym

def train_ppo(env_name='CartPole-v1', total_timesteps=100000, n_steps=2048):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PPOAgent(state_dim, action_dim)

    timesteps = 0
    episode_rewards = []
    current_episode_reward = 0

    while timesteps < total_timesteps:
        # 롤아웃 수집
        rollout = agent.collect_rollouts(env, n_steps)
        timesteps += n_steps

        # 에피소드 보상 추적
        for r, d in zip(rollout['rewards'], rollout['dones']):
            current_episode_reward += r
            if d:
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0

        # PPO 업데이트
        actor_loss, critic_loss = agent.update(rollout)

        # 로깅
        if len(episode_rewards) > 0 and timesteps % 10000 < n_steps:
            avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
            print(f"Timesteps: {timesteps}, Avg Reward: {avg_reward:.2f}")

    return agent, episode_rewards
```

---

## 5. PPO 변형들

### 5.1 PPO-Clip (기본)

위에서 구현한 방식입니다.

### 5.2 PPO-Penalty

KL divergence를 페널티로 추가:

```python
def ppo_penalty_loss(ratio, advantage, old_probs, new_probs, beta=0.01):
    policy_loss = (ratio * advantage).mean()

    kl_div = F.kl_div(new_probs.log(), old_probs, reduction='batchmean')

    return -policy_loss + beta * kl_div
```

### 5.3 Clipped Value Loss

가치 함수에도 클리핑 적용:

```python
def clipped_value_loss(values, old_values, returns, clip_epsilon=0.2):
    # 클리핑된 가치
    clipped_values = old_values + torch.clamp(
        values - old_values, -clip_epsilon, clip_epsilon
    )

    # 두 손실 중 큰 값
    loss1 = (values - returns) ** 2
    loss2 = (clipped_values - returns) ** 2

    return 0.5 * torch.max(loss1, loss2).mean()
```

---

## 6. 연속 행동 공간 PPO

```python
class ContinuousPPONetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()

        # Actor
        self.actor_mean = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        mean = self.actor_mean(state)
        std = self.actor_log_std.exp()
        value = self.critic(state)
        return mean, std, value

    def get_action(self, state, action=None):
        mean, std, value = self.forward(state)
        dist = torch.distributions.Normal(mean, std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)

        return action, log_prob, entropy, value
```

---

## 7. 하이퍼파라미터 가이드

### 7.1 일반적인 설정

```python
config = {
    # 학습
    'lr': 3e-4,                  # 학습률
    'gamma': 0.99,               # 할인율
    'gae_lambda': 0.95,          # GAE lambda

    # PPO 특정
    'clip_epsilon': 0.2,         # 클리핑 범위
    'update_epochs': 10,         # 업데이트 반복
    'batch_size': 64,            # 미니배치 크기

    # 손실 계수
    'value_coef': 0.5,           # 가치 손실 계수
    'entropy_coef': 0.01,        # 엔트로피 계수

    # 롤아웃
    'n_steps': 2048,             # 롤아웃 길이
    'n_envs': 8,                 # 병렬 환경 수

    # 안정화
    'max_grad_norm': 0.5,        # 그래디언트 클리핑
}
```

### 7.2 환경별 튜닝

| 환경 | lr | n_steps | clip_epsilon |
|------|-----|---------|--------------|
| CartPole | 3e-4 | 128 | 0.2 |
| LunarLander | 3e-4 | 2048 | 0.2 |
| Atari | 2.5e-4 | 128 | 0.1 |
| MuJoCo | 3e-4 | 2048 | 0.2 |

---

## 8. PPO vs 다른 알고리즘

| 알고리즘 | 복잡도 | 샘플 효율 | 안정성 |
|---------|-------|----------|--------|
| REINFORCE | 낮음 | 낮음 | 낮음 |
| A2C | 중간 | 중간 | 중간 |
| TRPO | 높음 | 높음 | 높음 |
| **PPO** | **중간** | **높음** | **높음** |
| SAC | 중간 | 높음 | 높음 |

**PPO의 장점:**
- TRPO 수준의 성능, 구현은 간단
- 다양한 환경에서 안정적
- 하이퍼파라미터 민감도 낮음

---

## 요약

**PPO 핵심:**
```
L^{CLIP} = E[min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)]

r(θ) = π_θ(a|s) / π_θ_old(a|s)  # 정책 비율
```

**클리핑 효과:**
- 정책 변화를 [1-ε, 1+ε] 범위로 제한
- 급격한 업데이트 방지
- 학습 안정성 확보

---

## 다음 단계

- [11_Multi_Agent_RL.md](./11_Multi_Agent_RL.md) - 다중 에이전트 강화학습
