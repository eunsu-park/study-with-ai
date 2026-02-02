# 07. Deep Q-Network (DQN)

**난이도: ⭐⭐⭐ (중급)**

## 학습 목표
- DQN의 핵심 아이디어와 구조 이해
- Experience Replay의 원리와 구현
- Target Network의 필요성과 동작 방식
- Double DQN, Dueling DQN 등 개선 기법
- PyTorch로 DQN 구현

---

## 1. Q-Learning의 한계와 DQN

### 1.1 테이블 기반 Q-Learning의 한계

```
문제점:
1. 상태 공간이 크면 테이블 저장 불가 (Atari: 256^(84*84*4) 상태)
2. 연속 상태 공간 처리 불가
3. 비슷한 상태 간 일반화 불가
```

### 1.2 함수 근사 (Function Approximation)

```python
# 테이블 대신 신경망으로 Q 함수 근사
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
        return self.network(state)  # 모든 행동의 Q값 출력
```

---

## 2. DQN의 핵심 기법

### 2.1 Experience Replay

경험을 버퍼에 저장하고 무작위로 샘플링하여 학습합니다.

**장점:**
- 데이터 효율성 향상 (경험 재사용)
- 연속 샘플의 상관관계 제거
- 학습 안정화

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

별도의 타겟 네트워크를 사용하여 학습 안정화합니다.

**문제:** Q(s,a;θ) 업데이트 시 타겟 y = r + γ max Q(s',a';θ)도 변함
**해결:** 타겟 네트워크 θ⁻를 고정하고 주기적으로 업데이트

```python
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-4):
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)

        # 타겟 네트워크 초기화 (동일한 가중치)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.gamma = 0.99

    def update_target_network(self):
        """타겟 네트워크 하드 업데이트"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def soft_update_target(self, tau=0.005):
        """타겟 네트워크 소프트 업데이트"""
        for target_param, param in zip(
            self.target_network.parameters(),
            self.q_network.parameters()
        ):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data
            )
```

---

## 3. DQN 전체 구현

### 3.1 에이전트 클래스

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

        # 네트워크
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

        # 배치 샘플링
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # 현재 Q값
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()

        # 타겟 Q값 (타겟 네트워크 사용)
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # 손실 계산 및 업데이트
        loss = nn.MSELoss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # 그래디언트 클리핑 (안정성)
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)
        self.optimizer.step()

        # 타겟 네트워크 업데이트
        self.learn_step += 1
        if self.learn_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Epsilon 감소
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss.item()
```

### 3.2 학습 루프

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

## 4. DQN 개선 기법

### 4.1 Double DQN

일반 DQN의 Q값 과대추정 문제를 해결합니다.

```python
# 일반 DQN: y = r + γ max_a' Q(s', a'; θ⁻)
# Double DQN: y = r + γ Q(s', argmax_a' Q(s', a'; θ); θ⁻)

def compute_double_dqn_target(self, rewards, next_states, dones):
    with torch.no_grad():
        # Q 네트워크로 행동 선택
        next_actions = self.q_network(next_states).argmax(1, keepdim=True)

        # 타겟 네트워크로 Q값 평가
        next_q = self.target_network(next_states).gather(1, next_actions).squeeze()

        target_q = rewards + self.gamma * next_q * (1 - dones)

    return target_q
```

### 4.2 Dueling DQN

Q 함수를 V(상태 가치)와 A(어드밴티지)로 분해합니다.

```
Q(s, a) = V(s) + A(s, a) - mean(A(s, ·))
```

```python
class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()

        # 공유 특징 추출
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )

        # 가치 스트림 (V)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # 어드밴티지 스트림 (A)
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

TD 오류가 큰 경험을 더 자주 샘플링합니다.

```python
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha  # 우선순위 지수
        self.beta = beta    # 중요도 샘플링 지수
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

        # 중요도 샘플링 가중치
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        batch = [self.buffer[i] for i in indices]
        return batch, indices, torch.FloatTensor(weights)

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = (abs(td_error) + 1e-6) ** self.alpha
```

---

## 5. CNN 기반 DQN (Atari)

### 5.1 이미지 입력 네트워크

```python
class AtariDQN(nn.Module):
    def __init__(self, n_actions):
        super().__init__()

        # 입력: 84x84x4 (4 프레임 스택)
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
        x = x / 255.0  # 정규화
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
```

### 5.2 프레임 전처리

```python
import cv2

class AtariPreprocessor:
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess_frame(self, frame):
        """84x84 그레이스케일로 변환"""
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

## 6. 실습: CartPole-v1

```python
def main():
    # 환경 설정
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # DQN 에이전트
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

    # 학습
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

        # 해결 조건
        if np.mean(scores[-100:]) >= 475:
            print(f"Solved in {episode + 1} episodes!")
            break

    env.close()
    return agent, scores

if __name__ == "__main__":
    agent, scores = main()
```

---

## 요약

| 기법 | 목적 | 핵심 아이디어 |
|------|------|--------------|
| Experience Replay | 데이터 효율성, 상관관계 제거 | 버퍼에서 무작위 샘플링 |
| Target Network | 학습 안정화 | 타겟 고정, 주기적 업데이트 |
| Double DQN | 과대추정 방지 | 행동 선택/평가 분리 |
| Dueling DQN | 효율적 학습 | V와 A 분리 |
| PER | 효율적 샘플링 | TD 오류 기반 우선순위 |

---

## 다음 단계

- [08_Policy_Gradient.md](./08_Policy_Gradient.md) - 정책 기반 방법론
