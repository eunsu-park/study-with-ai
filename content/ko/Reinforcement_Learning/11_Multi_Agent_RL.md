# 11. 다중 에이전트 강화학습 (Multi-Agent RL)

**난이도: ⭐⭐⭐⭐ (고급)**

## 학습 목표
- 다중 에이전트 환경의 특성 이해
- 협력, 경쟁, 혼합 시나리오 구분
- 중앙집중/분산 학습 패러다임
- MARL 알고리즘: IQL, QMIX, MAPPO

---

## 1. 다중 에이전트 환경 개요

### 1.1 단일 vs 다중 에이전트

| 특성 | 단일 에이전트 | 다중 에이전트 |
|------|-------------|--------------|
| 환경 | 정적 (에이전트 관점) | 동적 (다른 에이전트) |
| 보상 | 개인 보상 | 개인/팀/글로벌 |
| 최적성 | 최적 정책 존재 | 내쉬 균형 추구 |
| 학습 | 정상성 가정 | 비정상성 (이동 타겟) |

### 1.2 환경 유형

```
┌─────────────────────────────────────────────────────┐
│                    MARL 환경 유형                    │
├──────────────┬──────────────┬──────────────────────┤
│    협력       │     경쟁      │        혼합          │
│ (Cooperative) │(Competitive) │      (Mixed)         │
├──────────────┼──────────────┼──────────────────────┤
│ 팀 스포츠    │ 제로섬 게임   │ 일반 섬 게임         │
│ 로봇 협동    │ 1v1 대전     │ 협력적 경쟁          │
│ 스웜 로봇    │ 가위바위보    │ 사회적 딜레마         │
└──────────────┴──────────────┴──────────────────────┘
```

---

## 2. MARL의 도전 과제

### 2.1 비정상성 (Non-stationarity)

다른 에이전트도 학습하므로 환경이 계속 변합니다.

```python
# 에이전트 i의 관점에서
# 환경 전이: P(s'|s, a_i, a_{-i})
# 다른 에이전트의 정책이 변하면 전이 확률도 변함

class NonStationaryEnv:
    def step(self, actions):
        # actions: 모든 에이전트의 행동
        joint_action = tuple(actions)
        next_state = self.transition(self.state, joint_action)
        rewards = self.reward_function(self.state, joint_action, next_state)
        return next_state, rewards
```

### 2.2 신용 할당 (Credit Assignment)

팀 보상에서 개인 기여도를 파악하기 어렵습니다.

### 2.3 확장성 (Scalability)

에이전트 수가 늘면 상태-행동 공간이 기하급수적으로 증가합니다.

---

## 3. 학습 패러다임

### 3.1 중앙집중 학습, 분산 실행 (CTDE)

**Centralized Training, Decentralized Execution**

```
훈련 시: 글로벌 정보 접근 가능
실행 시: 로컬 관측만 사용

┌─────────────────────────────────┐
│       Central Critic            │  (훈련 시)
│   (글로벌 상태, 모든 행동 접근) │
└─────────────┬───────────────────┘
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
┌───────┐ ┌───────┐ ┌───────┐
│Actor 1│ │Actor 2│ │Actor 3│  (실행 시)
│(로컬) │ │(로컬) │ │(로컬) │
└───────┘ └───────┘ └───────┘
```

### 3.2 완전 분산 (Independent Learning)

각 에이전트가 독립적으로 학습합니다.

```python
class IndependentQLearning:
    """각 에이전트가 독립적으로 Q-learning"""
    def __init__(self, n_agents, state_dim, action_dim):
        self.agents = [
            QLearningAgent(state_dim, action_dim)
            for _ in range(n_agents)
        ]

    def choose_actions(self, observations):
        return [
            agent.choose_action(obs)
            for agent, obs in zip(self.agents, observations)
        ]

    def update(self, observations, actions, rewards, next_observations, dones):
        for i, agent in enumerate(self.agents):
            agent.update(
                observations[i], actions[i],
                rewards[i], next_observations[i], dones[i]
            )
```

---

## 4. IQL (Independent Q-Learning)

### 4.1 개념

각 에이전트가 다른 에이전트를 환경의 일부로 취급합니다.

```python
import torch
import torch.nn as nn
import numpy as np

class IQLAgent:
    def __init__(self, obs_dim, action_dim, lr=1e-3, gamma=0.99, epsilon=0.1):
        self.q_network = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_dim = action_dim

    def choose_action(self, obs):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        with torch.no_grad():
            q_values = self.q_network(torch.FloatTensor(obs))
            return q_values.argmax().item()

    def update(self, obs, action, reward, next_obs, done):
        obs_tensor = torch.FloatTensor(obs)
        next_obs_tensor = torch.FloatTensor(next_obs)

        current_q = self.q_network(obs_tensor)[action]

        with torch.no_grad():
            if done:
                target_q = reward
            else:
                target_q = reward + self.gamma * self.q_network(next_obs_tensor).max()

        loss = (current_q - target_q) ** 2

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class IQLSystem:
    def __init__(self, n_agents, obs_dims, action_dims):
        self.agents = [
            IQLAgent(obs_dims[i], action_dims[i])
            for i in range(n_agents)
        ]

    def step(self, env):
        observations = env.get_observations()
        actions = [
            agent.choose_action(obs)
            for agent, obs in zip(self.agents, observations)
        ]

        next_obs, rewards, dones, _ = env.step(actions)

        for i, agent in enumerate(self.agents):
            agent.update(
                observations[i], actions[i],
                rewards[i], next_obs[i], dones[i]
            )

        return rewards, dones
```

### 4.2 IQL의 한계

- 다른 에이전트 정책 변화로 환경이 비정상적
- 협력 학습에서 수렴이 어려울 수 있음

---

## 5. VDN과 QMIX (가치 분해)

### 5.1 VDN (Value Decomposition Networks)

팀 Q값을 개인 Q값의 합으로 분해:

$$Q_{tot}(s, \mathbf{a}) = \sum_{i=1}^{n} Q_i(o_i, a_i)$$

```python
class VDN:
    def __init__(self, n_agents, obs_dim, action_dim):
        self.agents = nn.ModuleList([
            nn.Sequential(
                nn.Linear(obs_dim, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim)
            )
            for _ in range(n_agents)
        ])

    def get_q_values(self, observations):
        """각 에이전트의 Q값"""
        return [
            agent(obs)
            for agent, obs in zip(self.agents, observations)
        ]

    def get_total_q(self, observations, actions):
        """팀 Q값 = 개인 Q값의 합"""
        q_values = self.get_q_values(observations)
        individual_q = [
            q[a] for q, a in zip(q_values, actions)
        ]
        return sum(individual_q)
```

### 5.2 QMIX

더 일반적인 분해를 허용합니다. 단조성 조건만 만족:

$$\frac{\partial Q_{tot}}{\partial Q_i} \geq 0$$

```python
class QMIXMixer(nn.Module):
    """QMIX 믹싱 네트워크"""
    def __init__(self, n_agents, state_dim, embed_dim=32):
        super().__init__()
        self.n_agents = n_agents

        # 하이퍼네트워크 (가중치 생성)
        self.hyper_w1 = nn.Linear(state_dim, n_agents * embed_dim)
        self.hyper_w2 = nn.Linear(state_dim, embed_dim)
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

        self.embed_dim = embed_dim

    def forward(self, agent_qs, state):
        """
        agent_qs: [batch, n_agents] - 각 에이전트의 Q값
        state: [batch, state_dim] - 글로벌 상태
        """
        batch_size = agent_qs.size(0)
        agent_qs = agent_qs.view(batch_size, 1, self.n_agents)

        # 첫 번째 레이어 가중치 (양수로 제한)
        w1 = torch.abs(self.hyper_w1(state))
        w1 = w1.view(batch_size, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(state).view(batch_size, 1, self.embed_dim)

        # 두 번째 레이어 가중치
        w2 = torch.abs(self.hyper_w2(state))
        w2 = w2.view(batch_size, self.embed_dim, 1)
        b2 = self.hyper_b2(state).view(batch_size, 1, 1)

        # 믹싱
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        q_tot = torch.bmm(hidden, w2) + b2

        return q_tot.squeeze(-1).squeeze(-1)
```

---

## 6. MADDPG (Multi-Agent DDPG)

### 6.1 개념

CTDE 패러다임 + Actor-Critic

- **Actor**: 로컬 관측만 사용
- **Critic**: 모든 에이전트의 관측과 행동 사용

```python
class MADDPGAgent:
    def __init__(self, agent_id, obs_dims, action_dims, n_agents):
        self.agent_id = agent_id
        self.n_agents = n_agents

        # Actor (로컬)
        self.actor = nn.Sequential(
            nn.Linear(obs_dims[agent_id], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dims[agent_id]),
            nn.Tanh()
        )

        # Critic (중앙집중)
        total_obs_dim = sum(obs_dims)
        total_action_dim = sum(action_dims)
        self.critic = nn.Sequential(
            nn.Linear(total_obs_dim + total_action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def act(self, obs, noise_scale=0.1):
        """로컬 관측으로 행동 결정"""
        action = self.actor(torch.FloatTensor(obs))
        noise = torch.randn_like(action) * noise_scale
        return (action + noise).clamp(-1, 1)

    def get_q_value(self, all_obs, all_actions):
        """글로벌 정보로 Q값 계산"""
        x = torch.cat([*all_obs, *all_actions], dim=-1)
        return self.critic(x)
```

---

## 7. MAPPO (Multi-Agent PPO)

### 7.1 구조

PPO를 다중 에이전트로 확장:

```python
class MAPPOAgent:
    def __init__(self, obs_dim, action_dim, state_dim):
        # Actor (로컬 관측)
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic (글로벌 상태)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def get_action(self, obs):
        probs = self.actor(torch.FloatTensor(obs))
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def get_value(self, state):
        return self.critic(torch.FloatTensor(state))


class MAPPO:
    def __init__(self, n_agents, obs_dims, action_dims, state_dim):
        self.agents = [
            MAPPOAgent(obs_dims[i], action_dims[i], state_dim)
            for i in range(n_agents)
        ]
        self.n_agents = n_agents

    def collect_rollout(self, env, n_steps):
        """모든 에이전트의 경험 수집"""
        rollouts = [{
            'obs': [], 'actions': [], 'rewards': [],
            'values': [], 'log_probs': [], 'dones': []
        } for _ in range(self.n_agents)]

        obs = env.reset()
        state = env.get_state()

        for _ in range(n_steps):
            actions = []
            for i, agent in enumerate(self.agents):
                action, log_prob = agent.get_action(obs[i])
                value = agent.get_value(state)

                actions.append(action)
                rollouts[i]['obs'].append(obs[i])
                rollouts[i]['actions'].append(action)
                rollouts[i]['values'].append(value.item())
                rollouts[i]['log_probs'].append(log_prob)

            next_obs, rewards, dones, _ = env.step(actions)
            next_state = env.get_state()

            for i in range(self.n_agents):
                rollouts[i]['rewards'].append(rewards[i])
                rollouts[i]['dones'].append(dones[i])

            obs = next_obs
            state = next_state

        return rollouts
```

---

## 8. Self-Play

### 8.1 개념

에이전트가 자기 자신의 복사본과 대전하며 학습합니다.

```python
class SelfPlayTrainer:
    def __init__(self, agent_class, env):
        self.current_agent = agent_class()
        self.opponent_pool = []
        self.env = env

    def train_episode(self):
        # 상대 선택 (과거 버전 중 무작위)
        if len(self.opponent_pool) > 0 and np.random.random() < 0.8:
            opponent = np.random.choice(self.opponent_pool)
        else:
            opponent = self.current_agent  # 자기 자신

        # 대전
        state = self.env.reset()
        done = False

        while not done:
            # 현재 에이전트 행동
            action1 = self.current_agent.choose_action(state[0])
            # 상대 행동
            action2 = opponent.choose_action(state[1])

            next_state, rewards, done, _ = self.env.step([action1, action2])

            # 학습 (현재 에이전트만)
            self.current_agent.update(
                state[0], action1, rewards[0], next_state[0], done
            )

            state = next_state

    def save_snapshot(self):
        """현재 에이전트를 상대 풀에 추가"""
        snapshot = copy.deepcopy(self.current_agent)
        self.opponent_pool.append(snapshot)

        # 풀 크기 제한
        if len(self.opponent_pool) > 10:
            self.opponent_pool.pop(0)
```

---

## 9. MARL 환경 예시

### 9.1 PettingZoo

```python
from pettingzoo.mpe import simple_spread_v2

def run_pettingzoo():
    env = simple_spread_v2.parallel_env()
    observations = env.reset()

    while env.agents:
        actions = {
            agent: env.action_space(agent).sample()
            for agent in env.agents
        }
        observations, rewards, terminations, truncations, infos = env.step(actions)

    env.close()
```

---

## 요약

| 알고리즘 | 패러다임 | 협력/경쟁 | 특징 |
|---------|---------|----------|------|
| IQL | 분산 | 둘 다 | 간단, 비정상성 문제 |
| VDN | CTDE | 협력 | 합 분해 |
| QMIX | CTDE | 협력 | 단조적 분해 |
| MADDPG | CTDE | 둘 다 | 연속 행동 |
| MAPPO | CTDE | 둘 다 | PPO 확장 |

---

## 다음 단계

- [12_Practical_RL_Project.md](./12_Practical_RL_Project.md) - 실전 프로젝트
