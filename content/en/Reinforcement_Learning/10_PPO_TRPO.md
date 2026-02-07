# 10. PPO and TRPO

**Difficulty: ⭐⭐⭐⭐ (Advanced)**

## Learning Objectives
- Understand stability issues in policy updates
- Learn TRPO's trust region concept
- Understand PPO's clipping mechanism
- Implement PPO with PyTorch

---

## 1. Problems in Policy Optimization

### 1.1 Dangers of Large Updates

In policy gradients, overly large updates can drastically degrade performance.

```
θ_new = θ_old + α∇J(θ)

Problem: Large α causes drastic policy changes, making learning unstable
Solution: Constrain policy changes
```

### 1.2 Solution Approaches

- **TRPO**: Constrain with KL divergence trust region (complex)
- **PPO**: Simple constraint using clipping

---

## 2. TRPO (Trust Region Policy Optimization)

### 2.1 Objective Function

Use ratio of new and old policies:

$$L^{CPI}(\theta) = \mathbb{E}\left[\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)} A^{\pi_{old}}(s, a)\right]$$

### 2.2 KL Divergence Constraint

$$\text{maximize}_\theta \quad L^{CPI}(\theta)$$
$$\text{subject to} \quad \mathbb{E}[D_{KL}(\pi_{\theta_{old}} || \pi_\theta)] \leq \delta$$

### 2.3 Problems with TRPO

- Requires second-order derivatives (Hessian)
- Needs conjugate gradient algorithm
- Complex implementation and high computational cost

---

## 3. PPO (Proximal Policy Optimization)

### 3.1 Core Idea

Use clipping to constrain policy ratio.

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$

### 3.2 Clipped Objective Function

$$L^{CLIP}(\theta) = \mathbb{E}\left[\min(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t)\right]$$

```python
def compute_ppo_loss(ratio, advantage, clip_epsilon=0.2):
    """PPO Clipped Loss"""
    # Clipped ratio
    clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)

    # Take minimum of two terms
    loss1 = ratio * advantage
    loss2 = clipped_ratio * advantage

    return -torch.min(loss1, loss2).mean()
```

### 3.3 Clipping Intuition

```
Advantage > 0 (good action):
- ratio increases → probability increases
- But ignore if ratio > 1+ε (prevent drastic increase)

Advantage < 0 (bad action):
- ratio decreases → probability decreases
- But ignore if ratio < 1-ε (prevent drastic decrease)
```

---

## 4. Complete PPO Implementation

### 4.1 PPO Agent

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
        """Collect experience"""
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

        # Last state value
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
        """Compute GAE"""
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
        """PPO Update"""
        advantages, returns = self.compute_gae(rollout)

        # Normalize
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        states = torch.FloatTensor(rollout['states'])
        actions = torch.LongTensor(rollout['actions'])
        old_log_probs = torch.FloatTensor(rollout['log_probs'])
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)

        # Multiple epoch updates
        for _ in range(self.update_epochs):
            # Generate minibatches
            indices = np.random.permutation(len(states))

            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                # Evaluate with current policy
                _, new_log_probs, entropy, values = self.network.get_action(
                    batch_states, batch_actions
                )

                # Compute ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # Clipped loss
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                critic_loss = F.mse_loss(values.squeeze(), batch_returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss

                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

        return actor_loss.item(), critic_loss.item()
```

### 4.2 PPO Training Loop

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
        # Collect rollouts
        rollout = agent.collect_rollouts(env, n_steps)
        timesteps += n_steps

        # Track episode rewards
        for r, d in zip(rollout['rewards'], rollout['dones']):
            current_episode_reward += r
            if d:
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0

        # PPO update
        actor_loss, critic_loss = agent.update(rollout)

        # Logging
        if len(episode_rewards) > 0 and timesteps % 10000 < n_steps:
            avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
            print(f"Timesteps: {timesteps}, Avg Reward: {avg_reward:.2f}")

    return agent, episode_rewards
```

---

## 5. PPO Variants

### 5.1 PPO-Clip (Basic)

The method implemented above.

### 5.2 PPO-Penalty

Add KL divergence as a penalty:

```python
def ppo_penalty_loss(ratio, advantage, old_probs, new_probs, beta=0.01):
    policy_loss = (ratio * advantage).mean()

    kl_div = F.kl_div(new_probs.log(), old_probs, reduction='batchmean')

    return -policy_loss + beta * kl_div
```

### 5.3 Clipped Value Loss

Apply clipping to value function as well:

```python
def clipped_value_loss(values, old_values, returns, clip_epsilon=0.2):
    # Clipped values
    clipped_values = old_values + torch.clamp(
        values - old_values, -clip_epsilon, clip_epsilon
    )

    # Maximum of two losses
    loss1 = (values - returns) ** 2
    loss2 = (clipped_values - returns) ** 2

    return 0.5 * torch.max(loss1, loss2).mean()
```

---

## 6. Continuous Action Space PPO

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

## 7. Hyperparameter Guide

### 7.1 General Settings

```python
config = {
    # Learning
    'lr': 3e-4,                  # Learning rate
    'gamma': 0.99,               # Discount factor
    'gae_lambda': 0.95,          # GAE lambda

    # PPO specific
    'clip_epsilon': 0.2,         # Clipping range
    'update_epochs': 10,         # Update iterations
    'batch_size': 64,            # Minibatch size

    # Loss coefficients
    'value_coef': 0.5,           # Value loss coefficient
    'entropy_coef': 0.01,        # Entropy coefficient

    # Rollout
    'n_steps': 2048,             # Rollout length
    'n_envs': 8,                 # Parallel environments

    # Stabilization
    'max_grad_norm': 0.5,        # Gradient clipping
}
```

### 7.2 Environment-Specific Tuning

| Environment | lr | n_steps | clip_epsilon |
|-------------|-----|---------|--------------|
| CartPole | 3e-4 | 128 | 0.2 |
| LunarLander | 3e-4 | 2048 | 0.2 |
| Atari | 2.5e-4 | 128 | 0.1 |
| MuJoCo | 3e-4 | 2048 | 0.2 |

---

## 8. PPO vs Other Algorithms

| Algorithm | Complexity | Sample Efficiency | Stability |
|-----------|-----------|-------------------|-----------|
| REINFORCE | Low | Low | Low |
| A2C | Medium | Medium | Medium |
| TRPO | High | High | High |
| **PPO** | **Medium** | **High** | **High** |
| SAC | Medium | High | High |

**PPO Advantages:**
- TRPO-level performance, simple implementation
- Stable across various environments
- Low sensitivity to hyperparameters

---

## Summary

**PPO Core:**
```
L^{CLIP} = E[min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)]

r(θ) = π_θ(a|s) / π_θ_old(a|s)  # Policy ratio
```

**Clipping Effect:**
- Constrains policy changes to [1-ε, 1+ε] range
- Prevents drastic updates
- Ensures learning stability

---

## Next Steps

- [11_Multi_Agent_RL.md](./11_Multi_Agent_RL.md) - Multi-Agent Reinforcement Learning
