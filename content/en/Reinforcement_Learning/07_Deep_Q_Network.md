# 07. Deep Q-Network (DQN)

**Difficulty: ⭐⭐⭐ (Intermediate)**

## Learning Objectives
- Understand DQN's core ideas and structure
- Grasp Experience Replay principles and implementation
- Understand the need for Target Network and its operation
- Learn improvement techniques like Double DQN, Dueling DQN
- Implement DQN with PyTorch

---

## 1. Limitations of Q-Learning and DQN

### 1.1 Limitations of Tabular Q-Learning

```
Problems:
1. Cannot store table for large state spaces (Atari: 256^(84*84*4) states)
2. Cannot handle continuous state spaces
3. No generalization between similar states
```

### 1.2 Function Approximation

```python
# Approximate Q function with neural network instead of table
# Q(s, a) ≈ Q(s, a; θ)

import torch
import torch.nn as nn

class QNetwork(nn.Module):
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
        return self.network(state)  # Output Q values for all actions
```

---

## 2. Core Techniques of DQN

### 2.1 Experience Replay

Store experiences in a buffer and sample randomly for learning.

**Advantages:**
- Improved data efficiency (reuse experiences)
- Remove correlation between consecutive samples
- Stabilize learning

```python
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones)
        )

    def __len__(self):
        return len(self.buffer)
```

### 2.2 Target Network

Use a separate target network to stabilize learning.

**Problem:** When updating Q(s,a;θ), target y = r + γ max Q(s',a';θ) also changes
**Solution:** Fix target network θ⁻ and update periodically

```python
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-4):
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)

        # Initialize target network (same weights)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.gamma = 0.99

    def update_target_network(self):
        """Hard update target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def soft_update_target(self, tau=0.005):
        """Soft update target network"""
        for target_param, param in zip(
            self.target_network.parameters(),
            self.q_network.parameters()
        ):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data
            )
```

---

## 3. Complete DQN Implementation

### 3.1 Agent Class

```python
import numpy as np

class DQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=1e-4,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        buffer_size=100000,
        batch_size=64,
        target_update_freq=1000
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.learn_step = 0

        # Networks
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)

    def choose_action(self, state, training=True):
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Target Q values (using target network)
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # Calculate loss and update
        loss = nn.MSELoss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping (stability)
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)
        self.optimizer.step()

        # Update target network
        self.learn_step += 1
        if self.learn_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss.item()
```

### 3.2 Training Loop

```python
import gymnasium as gym

def train_dqn(env_name='CartPole-v1', n_episodes=500):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)

    rewards_history = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.learn()

            state = next_state
            total_reward += reward

        rewards_history.append(total_reward)

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}")

    return agent, rewards_history
```

---

## 4. DQN Improvements

### 4.1 Double DQN

Solves Q-value overestimation problem in vanilla DQN.

```python
# Vanilla DQN: y = r + γ max_a' Q(s', a'; θ⁻)
# Double DQN: y = r + γ Q(s', argmax_a' Q(s', a'; θ); θ⁻)

def compute_double_dqn_target(self, rewards, next_states, dones):
    with torch.no_grad():
        # Select action with Q network
        next_actions = self.q_network(next_states).argmax(1, keepdim=True)

        # Evaluate Q value with target network
        next_q = self.target_network(next_states).gather(1, next_actions).squeeze()

        target_q = rewards + self.gamma * next_q * (1 - dones)

    return target_q
```

### 4.2 Dueling DQN

Decompose Q function into V (state value) and A (advantage).

```
Q(s, a) = V(s) + A(s, a) - mean(A(s, ·))
```

```python
class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()

        # Shared feature extraction
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )

        # Value stream (V)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Advantage stream (A)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        features = self.feature(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Q = V + A - mean(A)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values
```

### 4.3 Prioritized Experience Replay (PER)

Sample experiences with higher TD error more frequently.

```python
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling exponent
        self.buffer = []
        self.priorities = np.zeros(capacity)
        self.position = 0

    def push(self, *experience, priority=1.0):
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience

        self.priorities[self.position] = priority ** self.alpha
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        total_priority = self.priorities[:len(self.buffer)].sum()
        probs = self.priorities[:len(self.buffer)] / total_priority

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)

        # Importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        batch = [self.buffer[i] for i in indices]
        return batch, indices, torch.FloatTensor(weights)

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = (abs(td_error) + 1e-6) ** self.alpha
```

---

## 5. CNN-based DQN (Atari)

### 5.1 Image Input Network

```python
class AtariDQN(nn.Module):
    def __init__(self, n_actions):
        super().__init__()

        # Input: 84x84x4 (4 frame stack)
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        # x shape: (batch, 4, 84, 84)
        x = x / 255.0  # Normalize
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
```

### 5.2 Frame Preprocessing

```python
import cv2

class AtariPreprocessor:
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess_frame(self, frame):
        """Convert to 84x84 grayscale"""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, frame):
        processed = self.preprocess_frame(frame)
        for _ in range(self.frame_stack):
            self.frames.append(processed)
        return np.array(self.frames)

    def step(self, frame):
        processed = self.preprocess_frame(frame)
        self.frames.append(processed)
        return np.array(self.frames)
```

---

## 6. Practice: CartPole-v1

```python
def main():
    # Environment setup
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # DQN agent
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        batch_size=64,
        target_update_freq=100
    )

    # Training
    n_episodes = 300
    scores = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        score = 0

        for t in range(500):
            action = agent.choose_action(state)
            next_state, reward, done, truncated, _ = env.step(action)

            agent.store_transition(state, action, reward, next_state, done or truncated)
            agent.learn()

            state = next_state
            score += reward

            if done or truncated:
                break

        scores.append(score)

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}, Score: {np.mean(scores[-10:]):.2f}")

        # Solved condition
        if np.mean(scores[-100:]) >= 475:
            print(f"Solved in {episode + 1} episodes!")
            break

    env.close()
    return agent, scores

if __name__ == "__main__":
    agent, scores = main()
```

---

## Summary

| Technique | Purpose | Key Idea |
|-----------|---------|----------|
| Experience Replay | Data efficiency, remove correlation | Random sampling from buffer |
| Target Network | Stabilize learning | Fix target, periodic updates |
| Double DQN | Prevent overestimation | Separate action selection/evaluation |
| Dueling DQN | Efficient learning | Separate V and A |
| PER | Efficient sampling | Priority based on TD error |

---

## Next Steps

- [08_Policy_Gradient.md](./08_Policy_Gradient.md) - Policy-based methods

---

## References

- Mnih et al., "Playing Atari with Deep Reinforcement Learning" (2013)
- Mnih et al., "Human-level control through deep reinforcement learning" (Nature 2015)
- van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning" (2015)
- Wang et al., "Dueling Network Architectures for Deep Reinforcement Learning" (2016)
- Schaul et al., "Prioritized Experience Replay" (2015)
