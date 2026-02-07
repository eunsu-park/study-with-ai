# 11. Multi-Agent Reinforcement Learning (MARL)

**Difficulty: ⭐⭐⭐⭐ (Advanced)**

## Learning Objectives
- Understand characteristics of multi-agent environments
- Distinguish cooperative, competitive, and mixed scenarios
- Learn centralized/decentralized learning paradigms
- Study MARL algorithms: IQL, QMIX, MAPPO

---

## 1. Multi-Agent Environment Overview

### 1.1 Single vs Multi-Agent

| Feature | Single Agent | Multi-Agent |
|---------|-------------|-------------|
| Environment | Static (from agent's view) | Dynamic (other agents) |
| Rewards | Individual reward | Individual/Team/Global |
| Optimality | Optimal policy exists | Nash equilibrium |
| Learning | Stationarity assumption | Non-stationarity (moving target) |

### 1.2 Environment Types

```
┌─────────────────────────────────────────────────────┐
│                MARL Environment Types                │
├──────────────┬──────────────┬──────────────────────┤
│  Cooperative │  Competitive │        Mixed         │
│              │              │                      │
├──────────────┼──────────────┼──────────────────────┤
│ Team sports  │ Zero-sum games│ General-sum games   │
│ Robot teams  │ 1v1 battles  │ Competitive cooperation│
│ Swarm robots │ Rock-paper-scissors│ Social dilemmas│
└──────────────┴──────────────┴──────────────────────┘
```

---

## 2. MARL Challenges

### 2.1 Non-stationarity

Other agents are also learning, so the environment constantly changes.

```python
# From agent i's perspective
# Transition: P(s'|s, a_i, a_{-i})
# When other agents' policies change, transition probabilities also change

class NonStationaryEnv:
    def step(self, actions):
        # actions: all agents' actions
        joint_action = tuple(actions)
        next_state = self.transition(self.state, joint_action)
        rewards = self.reward_function(self.state, joint_action, next_state)
        return next_state, rewards
```

### 2.2 Credit Assignment

Difficult to determine individual contributions from team rewards.

### 2.3 Scalability

State-action space grows exponentially with number of agents.

---

## 3. Learning Paradigms

### 3.1 Centralized Training, Decentralized Execution (CTDE)

**Centralized Training, Decentralized Execution**

```
Training: Global information access
Execution: Local observations only

┌─────────────────────────────────┐
│       Central Critic            │  (Training)
│ (Access to global state, all actions)│
└─────────────┬───────────────────┘
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
┌───────┐ ┌───────┐ ┌───────┐
│Actor 1│ │Actor 2│ │Actor 3│  (Execution)
│(Local)│ │(Local)│ │(Local)│
└───────┘ └───────┘ └───────┘
```

### 3.2 Fully Decentralized (Independent Learning)

Each agent learns independently.

```python
class IndependentQLearning:
    """Each agent independently performs Q-learning"""
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

### 4.1 Concept

Each agent treats other agents as part of the environment.

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

### 4.2 IQL Limitations

- Environment becomes non-stationary due to changing other agent policies
- Convergence can be difficult in cooperative learning

---

## 5. VDN and QMIX (Value Decomposition)

### 5.1 VDN (Value Decomposition Networks)

Decompose team Q-value as sum of individual Q-values:

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
        """Q-values for each agent"""
        return [
            agent(obs)
            for agent, obs in zip(self.agents, observations)
        ]

    def get_total_q(self, observations, actions):
        """Team Q-value = sum of individual Q-values"""
        q_values = self.get_q_values(observations)
        individual_q = [
            q[a] for q, a in zip(q_values, actions)
        ]
        return sum(individual_q)
```

### 5.2 QMIX

Allows more general decomposition. Only requires monotonicity condition:

$$\frac{\partial Q_{tot}}{\partial Q_i} \geq 0$$

```python
class QMIXMixer(nn.Module):
    """QMIX Mixing Network"""
    def __init__(self, n_agents, state_dim, embed_dim=32):
        super().__init__()
        self.n_agents = n_agents

        # Hypernetworks (generate weights)
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
        agent_qs: [batch, n_agents] - Each agent's Q-value
        state: [batch, state_dim] - Global state
        """
        batch_size = agent_qs.size(0)
        agent_qs = agent_qs.view(batch_size, 1, self.n_agents)

        # First layer weights (constrain to positive)
        w1 = torch.abs(self.hyper_w1(state))
        w1 = w1.view(batch_size, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(state).view(batch_size, 1, self.embed_dim)

        # Second layer weights
        w2 = torch.abs(self.hyper_w2(state))
        w2 = w2.view(batch_size, self.embed_dim, 1)
        b2 = self.hyper_b2(state).view(batch_size, 1, 1)

        # Mixing
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        q_tot = torch.bmm(hidden, w2) + b2

        return q_tot.squeeze(-1).squeeze(-1)
```

---

## 6. MADDPG (Multi-Agent DDPG)

### 6.1 Concept

CTDE paradigm + Actor-Critic

- **Actor**: Uses only local observations
- **Critic**: Uses all agents' observations and actions

```python
class MADDPGAgent:
    def __init__(self, agent_id, obs_dims, action_dims, n_agents):
        self.agent_id = agent_id
        self.n_agents = n_agents

        # Actor (local)
        self.actor = nn.Sequential(
            nn.Linear(obs_dims[agent_id], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dims[agent_id]),
            nn.Tanh()
        )

        # Critic (centralized)
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
        """Decide action with local observation"""
        action = self.actor(torch.FloatTensor(obs))
        noise = torch.randn_like(action) * noise_scale
        return (action + noise).clamp(-1, 1)

    def get_q_value(self, all_obs, all_actions):
        """Compute Q-value with global information"""
        x = torch.cat([*all_obs, *all_actions], dim=-1)
        return self.critic(x)
```

---

## 7. MAPPO (Multi-Agent PPO)

### 7.1 Architecture

Extend PPO to multi-agent:

```python
class MAPPOAgent:
    def __init__(self, obs_dim, action_dim, state_dim):
        # Actor (local observation)
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic (global state)
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
        """Collect experience from all agents"""
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

### 8.1 Concept

Agents learn by competing against copies of themselves.

```python
class SelfPlayTrainer:
    def __init__(self, agent_class, env):
        self.current_agent = agent_class()
        self.opponent_pool = []
        self.env = env

    def train_episode(self):
        # Choose opponent (random from past versions)
        if len(self.opponent_pool) > 0 and np.random.random() < 0.8:
            opponent = np.random.choice(self.opponent_pool)
        else:
            opponent = self.current_agent  # Self

        # Play match
        state = self.env.reset()
        done = False

        while not done:
            # Current agent action
            action1 = self.current_agent.choose_action(state[0])
            # Opponent action
            action2 = opponent.choose_action(state[1])

            next_state, rewards, done, _ = self.env.step([action1, action2])

            # Learn (current agent only)
            self.current_agent.update(
                state[0], action1, rewards[0], next_state[0], done
            )

            state = next_state

    def save_snapshot(self):
        """Add current agent to opponent pool"""
        snapshot = copy.deepcopy(self.current_agent)
        self.opponent_pool.append(snapshot)

        # Limit pool size
        if len(self.opponent_pool) > 10:
            self.opponent_pool.pop(0)
```

---

## 9. MARL Environment Example

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

## Summary

| Algorithm | Paradigm | Cooperative/Competitive | Characteristics |
|-----------|----------|------------------------|-----------------|
| IQL | Decentralized | Both | Simple, non-stationarity issue |
| VDN | CTDE | Cooperative | Sum decomposition |
| QMIX | CTDE | Cooperative | Monotonic decomposition |
| MADDPG | CTDE | Both | Continuous actions |
| MAPPO | CTDE | Both | PPO extension |

---

## Next Steps

- [12_Practical_RL_Project.md](./12_Practical_RL_Project.md) - Practical Projects
