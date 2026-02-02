# 12. 실전 RL 프로젝트

**난이도: ⭐⭐⭐⭐ (고급)**

## 학습 목표
- Gymnasium 환경 사용법 숙달
- 완전한 RL 프로젝트 구조 이해
- 학습 모니터링과 디버깅 기법
- Atari 게임 에이전트 구현
- 학습된 모델 저장과 평가

---

## 1. 프로젝트 구조

### 1.1 권장 디렉토리 구조

```
rl_project/
├── config/
│   ├── default.yaml
│   └── atari.yaml
├── agents/
│   ├── __init__.py
│   ├── base.py
│   ├── dqn.py
│   └── ppo.py
├── networks/
│   ├── __init__.py
│   ├── mlp.py
│   └── cnn.py
├── utils/
│   ├── __init__.py
│   ├── buffer.py
│   ├── logger.py
│   └── wrappers.py
├── envs/
│   └── custom_env.py
├── train.py
├── evaluate.py
└── requirements.txt
```

### 1.2 설정 파일

```yaml
# config/default.yaml
env:
  name: "CartPole-v1"
  n_envs: 4

agent:
  type: "PPO"
  lr: 3e-4
  gamma: 0.99
  gae_lambda: 0.95
  clip_epsilon: 0.2
  epochs: 10
  batch_size: 64

training:
  total_timesteps: 100000
  eval_freq: 10000
  save_freq: 50000
  log_freq: 1000

logging:
  use_wandb: true
  project_name: "rl-project"
```

---

## 2. Gymnasium 환경

### 2.1 기본 사용법

```python
import gymnasium as gym
import numpy as np

def basic_usage():
    # 환경 생성
    env = gym.make("CartPole-v1", render_mode="human")

    # 환경 정보
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # 에피소드 실행
    observation, info = env.reset(seed=42)

    for _ in range(1000):
        action = env.action_space.sample()  # 무작위 행동
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()
```

### 2.2 벡터화 환경 (병렬 처리)

```python
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

def make_env(env_name, seed):
    def _init():
        env = gym.make(env_name)
        env.reset(seed=seed)
        return env
    return _init

def vectorized_envs():
    n_envs = 4
    env_name = "CartPole-v1"

    # 비동기 환경 (각 환경이 별도 프로세스)
    envs = AsyncVectorEnv([
        make_env(env_name, seed=i) for i in range(n_envs)
    ])

    # 모든 환경 동시 리셋
    observations, infos = envs.reset()
    print(f"Observations shape: {observations.shape}")

    # 모든 환경 동시 스텝
    actions = envs.action_space.sample()
    observations, rewards, terminateds, truncateds, infos = envs.step(actions)

    envs.close()
```

### 2.3 환경 래퍼

```python
import gymnasium as gym
from gymnasium import spaces
from collections import deque

class FrameStack(gym.Wrapper):
    """연속 프레임을 스택"""
    def __init__(self, env, n_frames=4):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = deque(maxlen=n_frames)

        # 관측 공간 수정
        obs_shape = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(n_frames, *obs_shape),
            dtype=np.uint8
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return np.array(self.frames), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return np.array(self.frames), reward, terminated, truncated, info


class RewardWrapper(gym.RewardWrapper):
    """보상 스케일링/클리핑"""
    def reward(self, reward):
        return np.clip(reward, -1, 1)


class NormalizeObservation(gym.ObservationWrapper):
    """관측값 정규화"""
    def __init__(self, env):
        super().__init__(env)
        self.mean = 0
        self.var = 1
        self.count = 1e-4

    def observation(self, obs):
        self.update_stats(obs)
        return (obs - self.mean) / np.sqrt(self.var + 1e-8)

    def update_stats(self, obs):
        batch_mean = np.mean(obs)
        batch_var = np.var(obs)
        batch_count = obs.size

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean += delta * batch_count / total_count
        self.var = (self.var * self.count + batch_var * batch_count) / total_count
        self.count = total_count
```

---

## 3. 완전한 PPO 프로젝트

### 3.1 네트워크 정의

```python
# networks/mlp.py
import torch
import torch.nn as nn
from torch.distributions import Categorical

class ActorCriticMLP(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_sizes=(64, 64)):
        super().__init__()

        # 공유 레이어
        layers = []
        prev_size = obs_dim
        for size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, size),
                nn.Tanh()
            ])
            prev_size = size

        self.shared = nn.Sequential(*layers)

        # Actor와 Critic 헤드
        self.actor = nn.Linear(prev_size, action_dim)
        self.critic = nn.Linear(prev_size, 1)

        # 가중치 초기화
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)

    def forward(self, obs):
        features = self.shared(obs)
        return self.actor(features), self.critic(features)

    def get_action_and_value(self, obs, action=None):
        logits, value = self.forward(obs)
        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), value.squeeze(-1)
```

### 3.2 PPO 에이전트

```python
# agents/ppo.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PPO:
    def __init__(
        self,
        env,
        network,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        n_epochs=10,
        batch_size=64,
        device="cpu"
    ):
        self.env = env
        self.network = network.to(device)
        self.optimizer = optim.Adam(network.parameters(), lr=lr)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device

    def collect_rollout(self, n_steps):
        """경험 수집"""
        obs_buf = []
        act_buf = []
        rew_buf = []
        done_buf = []
        val_buf = []
        logp_buf = []

        obs, _ = self.env.reset()
        obs = torch.FloatTensor(obs).to(self.device)

        for _ in range(n_steps):
            with torch.no_grad():
                action, logp, _, value = self.network.get_action_and_value(obs)

            next_obs, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy())
            done = terminated or truncated

            obs_buf.append(obs.cpu().numpy())
            act_buf.append(action.cpu().numpy())
            rew_buf.append(reward)
            done_buf.append(done)
            val_buf.append(value.cpu().numpy())
            logp_buf.append(logp.cpu().numpy())

            obs = torch.FloatTensor(next_obs).to(self.device)
            if done:
                obs, _ = self.env.reset()
                obs = torch.FloatTensor(obs).to(self.device)

        # 마지막 가치 추정
        with torch.no_grad():
            _, _, _, last_value = self.network.get_action_and_value(obs)

        return {
            'obs': np.array(obs_buf),
            'actions': np.array(act_buf),
            'rewards': np.array(rew_buf),
            'dones': np.array(done_buf),
            'values': np.array(val_buf),
            'log_probs': np.array(logp_buf),
            'last_value': last_value.cpu().numpy()
        }

    def compute_gae(self, rollout):
        """GAE 계산"""
        rewards = rollout['rewards']
        values = rollout['values']
        dones = rollout['dones']
        last_value = rollout['last_value']

        n_steps = len(rewards)
        advantages = np.zeros(n_steps)
        last_gae = 0

        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
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
        obs = torch.FloatTensor(rollout['obs']).to(self.device)
        actions = torch.LongTensor(rollout['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(rollout['log_probs']).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        # 여러 에폭
        n_samples = len(obs)
        indices = np.arange(n_samples)

        total_loss = 0
        for _ in range(self.n_epochs):
            np.random.shuffle(indices)

            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                _, new_log_probs, entropy, values = self.network.get_action_and_value(
                    obs[batch_idx], actions[batch_idx]
                )

                # 비율
                ratio = torch.exp(new_log_probs - old_log_probs[batch_idx])

                # Clipped loss
                surr1 = ratio * advantages[batch_idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages[batch_idx]
                actor_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(values, returns[batch_idx])

                # Entropy loss
                entropy_loss = -entropy.mean()

                # Total loss
                loss = actor_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()

        return total_loss / (self.n_epochs * (n_samples // self.batch_size))

    def save(self, path):
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
```

---

## 4. 학습 스크립트

```python
# train.py
import gymnasium as gym
import numpy as np
import torch
from agents.ppo import PPO
from networks.mlp import ActorCriticMLP
from utils.logger import Logger

def train(config):
    # 환경 생성
    env = gym.make(config['env']['name'])

    # 네트워크 생성
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    network = ActorCriticMLP(obs_dim, action_dim)

    # 에이전트 생성
    agent = PPO(
        env=env,
        network=network,
        **config['agent']
    )

    # 로거
    logger = Logger(config['logging'])

    # 학습 루프
    total_timesteps = config['training']['total_timesteps']
    n_steps = config['training']['n_steps']
    timesteps = 0
    episode_rewards = []
    current_episode_reward = 0

    while timesteps < total_timesteps:
        # 롤아웃 수집
        rollout = agent.collect_rollout(n_steps)
        timesteps += n_steps

        # 에피소드 보상 추적
        for r, d in zip(rollout['rewards'], rollout['dones']):
            current_episode_reward += r
            if d:
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0

        # 업데이트
        loss = agent.update(rollout)

        # 로깅
        if len(episode_rewards) > 0:
            logger.log({
                'timesteps': timesteps,
                'loss': loss,
                'mean_reward': np.mean(episode_rewards[-10:]),
                'episodes': len(episode_rewards)
            })

        # 체크포인트 저장
        if timesteps % config['training']['save_freq'] == 0:
            agent.save(f"checkpoints/ppo_{timesteps}.pt")

    env.close()
    return agent

if __name__ == "__main__":
    import yaml
    with open("config/default.yaml") as f:
        config = yaml.safe_load(f)

    train(config)
```

---

## 5. 평가 스크립트

```python
# evaluate.py
import gymnasium as gym
import torch
import numpy as np

def evaluate(agent, env_name, n_episodes=10, render=False):
    """학습된 에이전트 평가"""
    render_mode = "human" if render else None
    env = gym.make(env_name, render_mode=render_mode)

    episode_rewards = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                action, _, _, _ = agent.network.get_action_and_value(obs_tensor)

            obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            total_reward += reward

        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1}: {total_reward}")

    env.close()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    print(f"\nMean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    return episode_rewards
```

---

## 6. 로깅 및 시각화

### 6.1 Weights & Biases 통합

```python
# utils/logger.py
import wandb
import matplotlib.pyplot as plt
from collections import deque

class Logger:
    def __init__(self, config):
        self.use_wandb = config.get('use_wandb', False)
        self.rewards_buffer = deque(maxlen=100)

        if self.use_wandb:
            wandb.init(
                project=config.get('project_name', 'rl-project'),
                config=config
            )

    def log(self, metrics):
        if 'mean_reward' in metrics:
            self.rewards_buffer.append(metrics['mean_reward'])

        if self.use_wandb:
            wandb.log(metrics)
        else:
            print(f"Step {metrics.get('timesteps', 0)}: "
                  f"Reward={metrics.get('mean_reward', 0):.2f}")

    def plot_rewards(self, rewards, save_path=None):
        plt.figure(figsize=(10, 5))
        plt.plot(rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Progress')

        if save_path:
            plt.savefig(save_path)
        plt.show()

    def close(self):
        if self.use_wandb:
            wandb.finish()
```

---

## 7. Atari 프로젝트

### 7.1 CNN 네트워크

```python
# networks/cnn.py
class AtariNetwork(nn.Module):
    def __init__(self, action_dim):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.actor = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

        self.critic = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = x / 255.0  # 정규화
        features = self.conv(x)
        return self.actor(features), self.critic(features)
```

### 7.2 Atari 래퍼

```python
from gymnasium.wrappers import AtariPreprocessing, FrameStack

def make_atari_env(env_name):
    env = gym.make(env_name)
    env = AtariPreprocessing(
        env,
        frame_skip=4,
        grayscale_obs=True,
        scale_obs=False
    )
    env = FrameStack(env, 4)
    return env
```

---

## 8. 디버깅 팁

### 8.1 일반적인 문제

| 문제 | 원인 | 해결책 |
|------|------|--------|
| 보상이 증가하지 않음 | 학습률 너무 높음/낮음 | 학습률 그리드 서치 |
| 학습 불안정 | 그래디언트 폭발 | 그래디언트 클리핑 |
| 갑작스러운 성능 저하 | 정책 급변 | clip_epsilon 감소 |
| 메모리 부족 | 버퍼 크기 | 배치 크기 조정 |

### 8.2 디버깅 코드

```python
def debug_training(agent):
    """학습 디버깅"""
    # 그래디언트 확인
    for name, param in agent.network.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"{name}: grad_norm={grad_norm:.6f}")

    # 정책 엔트로피 확인
    obs = torch.randn(1, obs_dim)
    logits, _ = agent.network(obs)
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * probs.log()).sum()
    print(f"Policy entropy: {entropy.item():.4f}")
```

---

## 요약

**프로젝트 체크리스트:**
- [ ] 환경 설정 및 테스트
- [ ] 네트워크 아키텍처 정의
- [ ] 에이전트 구현
- [ ] 학습 루프 작성
- [ ] 로깅 설정
- [ ] 하이퍼파라미터 튜닝
- [ ] 모델 저장/로드
- [ ] 평가 및 시각화

**핵심 도구:**
- Gymnasium: 환경
- PyTorch: 신경망
- Weights & Biases: 실험 추적
- NumPy: 수치 연산

---

## 추가 학습 자료

- [Spinning Up in Deep RL](https://spinningup.openai.com/)
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)
- [CleanRL](https://github.com/vwxyzjn/cleanrl)
- [RLlib](https://docs.ray.io/en/latest/rllib/)
