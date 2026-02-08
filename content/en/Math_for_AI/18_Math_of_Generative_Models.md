# 18. Mathematics of Generative Models

## Learning Objectives

- Understand the goals of generative models and the differences between explicit and implicit density models
- Fully derive the ELBO of VAE and explain the meaning of reconstruction and KL terms
- Understand the minimax game theory of GANs and the relationship to JS divergence
- Understand the basics of Wasserstein distance and optimal transport theory
- Understand the mathematics of the forward/reverse process of diffusion models and score matching
- Understand recent advances in Flow Matching and Continuous Normalizing Flows (CNF)

---

## 1. Goals of Generative Models

### 1.1 Learning Data Distributions

**Goal**: Learn the true data distribution $p_{\text{data}}$ from observed data $\{\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(n)}\} \sim p_{\text{data}}(\mathbf{x})$

**Applications**:
- **Sampling**: Generate new data
- **Density estimation**: Compute $p(\mathbf{x})$ (anomaly detection)
- **Conditional generation**: $p(\mathbf{y}|\mathbf{x})$ (image-text, style transfer)

### 1.2 Explicit vs Implicit Density

**Explicit density models**:
- Directly define probability density $p_\theta(\mathbf{x})$
- Maximize likelihood $\log p_\theta(\mathbf{x})$
- Examples: VAE, autoregressive models, normalizing flows

**Implicit density models**:
- Define only sampling process without specifying density
- $\mathbf{x} = G_\theta(\mathbf{z})$, $\mathbf{z} \sim p(\mathbf{z})$
- Example: GAN

### 1.3 Training Paradigms

| Method | Objective | Sampling | Density Evaluation |
|------|----------|--------|----------|
| **VAE** | Maximize ELBO | Fast | Approximate |
| **GAN** | Adversarial game | Fast | Not possible |
| **Normalizing Flow** | Exact likelihood | Fast | Exact |
| **Diffusion Models** | Noise prediction | Slow (iterative) | Implicit |

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 간단한 1D 데이터 분포 (혼합 가우시안)
def generate_data(n_samples=1000):
    """진짜 데이터 분포: 두 개의 가우시안 혼합"""
    modes = np.random.choice([0, 1], size=n_samples, p=[0.3, 0.7])
    data = np.where(modes == 0,
                    np.random.randn(n_samples) * 0.5 - 2,
                    np.random.randn(n_samples) * 0.7 + 2)
    return data

data = generate_data(n_samples=5000)

plt.figure(figsize=(10, 4))
plt.hist(data, bins=50, density=True, alpha=0.7, label='True data distribution')
plt.xlabel('x')
plt.ylabel('Density')
plt.title('Example data distribution (mixture of Gaussians)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('data_distribution.png', dpi=150, bbox_inches='tight')
print("데이터 분포 시각화 저장 완료")
```

## 2. Mathematics of VAE: Complete ELBO Derivation

### 2.1 Latent Variable Model

**Generative process**:
1. Sample latent variable: $\mathbf{z} \sim p(\mathbf{z})$ (usually $\mathcal{N}(\mathbf{0}, I)$)
2. Generate data: $\mathbf{x} \sim p_\theta(\mathbf{x}|\mathbf{z})$

**Goal**: Maximize marginal likelihood

$$\log p_\theta(\mathbf{x}) = \log \int p_\theta(\mathbf{x}|\mathbf{z}) p(\mathbf{z}) d\mathbf{z}$$

**Problem**: Integral is intractable

### 2.2 ELBO Derivation (Method 1: Jensen's Inequality)

Using concavity of log and Jensen's inequality:

$$\log p_\theta(\mathbf{x}) = \log \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} \left[ \frac{p_\theta(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})} \right]$$

$$\geq \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} \left[ \log \frac{p_\theta(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})} \right]$$

$$= \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} \left[ \log p_\theta(\mathbf{x}|\mathbf{z}) + \log p(\mathbf{z}) - \log q_\phi(\mathbf{z}|\mathbf{x}) \right]$$

$$= \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} \left[ \log p_\theta(\mathbf{x}|\mathbf{z}) \right] - D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$$

This is the **ELBO** (Evidence Lower BOund):

$$\mathcal{L}(\theta, \phi; \mathbf{x}) = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} \left[ \log p_\theta(\mathbf{x}|\mathbf{z}) \right] - D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$$

### 2.3 ELBO Derivation (Method 2: KL Decomposition)

KL divergence between posterior $p_\theta(\mathbf{z}|\mathbf{x})$ and approximation $q_\phi(\mathbf{z}|\mathbf{x})$:

$$D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p_\theta(\mathbf{z}|\mathbf{x})) = \mathbb{E}_{q_\phi} \left[ \log \frac{q_\phi(\mathbf{z}|\mathbf{x})}{p_\theta(\mathbf{z}|\mathbf{x})} \right]$$

Bayes' theorem: $p_\theta(\mathbf{z}|\mathbf{x}) = \frac{p_\theta(\mathbf{x}|\mathbf{z})p(\mathbf{z})}{p_\theta(\mathbf{x})}$

$$D_{\text{KL}}(q_\phi \| p_\theta) = \mathbb{E}_{q_\phi} \left[ \log q_\phi(\mathbf{z}|\mathbf{x}) - \log p_\theta(\mathbf{x}|\mathbf{z}) - \log p(\mathbf{z}) + \log p_\theta(\mathbf{x}) \right]$$

Rearranging:

$$\log p_\theta(\mathbf{x}) = D_{\text{KL}}(q_\phi \| p_\theta) + \mathcal{L}(\theta, \phi; \mathbf{x})$$

Since $D_{\text{KL}} \geq 0$, $\mathcal{L}$ is a lower bound for $\log p_\theta(\mathbf{x})$.

### 2.4 Interpretation of the Two ELBO Terms

$$\mathcal{L} = \underbrace{\mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} \left[ \log p_\theta(\mathbf{x}|\mathbf{z}) \right]}_{\text{Reconstruction term}} - \underbrace{D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))}_{\text{Regularization term}}$$

**Reconstruction term**:
- How well does it reconstruct $\mathbf{x}$ from latent representation $\mathbf{z}$
- Quality of decoder $p_\theta(\mathbf{x}|\mathbf{z})$

**Regularization term**:
- Keeps encoder distribution close to prior
- Structures the latent space
- Prevents overfitting

### 2.5 Reparameterization Trick

**Problem**: How to compute $\nabla_\phi \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} [f(\mathbf{z})]$?

**Solution**: Express $\mathbf{z}$ as deterministic function + noise

For Gaussian case: $q_\phi(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}_\phi(\mathbf{x}), \boldsymbol{\sigma}_\phi^2(\mathbf{x}))$

$$\mathbf{z} = \boldsymbol{\mu}_\phi(\mathbf{x}) + \boldsymbol{\sigma}_\phi(\mathbf{x}) \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, I)$$

Now $\mathbf{z}$ is differentiable with respect to $\phi$.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim=784, latent_dim=20):
        super().__init__()

        # 인코더
        self.fc1 = nn.Linear(input_dim, 400)
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)

        # 디코더
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, input_dim)

    def encode(self, x):
        """인코더: x -> (mu, logvar)"""
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """재매개변수화: z = mu + sigma * epsilon"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z

    def decode(self, z):
        """디코더: z -> x_recon"""
        h = F.relu(self.fc3(z))
        x_recon = torch.sigmoid(self.fc4(h))
        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

def vae_loss(x, x_recon, mu, logvar):
    """
    VAE 손실 함수

    Parameters:
    -----------
    x : Tensor
        원본 입력
    x_recon : Tensor
        재구성된 출력
    mu, logvar : Tensor
        잠재 분포의 평균과 로그 분산

    Returns:
    --------
    loss : Tensor
        총 손실 (재구성 + KL)
    """
    # 재구성 손실 (Bernoulli 분포 가정)
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')

    # KL 발산 (가우시안 간의 닫힌 형태)
    # KL(N(mu, sigma^2) || N(0, 1)) = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kl_loss, recon_loss, kl_loss

# 예제: 간단한 2D 잠재 공간
vae_model = VAE(input_dim=2, latent_dim=2)
x_sample = torch.randn(10, 2)

x_recon, mu, logvar = vae_model(x_sample)
total_loss, recon, kl = vae_loss(x_sample, x_recon, mu, logvar)

print(f"총 손실: {total_loss.item():.4f}")
print(f"재구성 손실: {recon.item():.4f}")
print(f"KL 손실: {kl.item():.4f}")
```

### 2.6 Closed-Form KL Divergence for Gaussians

When $q = \mathcal{N}(\boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma}^2))$, $p = \mathcal{N}(\mathbf{0}, I)$:

$$D_{\text{KL}}(q \| p) = \frac{1}{2} \sum_{i=1}^{d} \left( \sigma_i^2 + \mu_i^2 - 1 - \log \sigma_i^2 \right)$$

**Derivation**: Using definition of KL divergence and entropy formula for Gaussians

## 3. Mathematics of GANs

### 3.1 Minimax Game

**Generator** (G) $G$: $\mathbf{z} \sim p(\mathbf{z}) \mapsto G(\mathbf{z}) \approx p_{\text{data}}$

**Discriminator** (D) $D$: $\mathbf{x} \mapsto D(\mathbf{x}) \in [0, 1]$ (probability of being real)

**Objective function**:

$$\min_G \max_D \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}} [\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p(\mathbf{z})} [\log(1 - D(G(\mathbf{z})))]$$

**Interpretation**:
- $D$ tries to assign high probability to real data and low to fake
- $G$ tries to fool $D$ (make fake look real)

### 3.2 Optimal Discriminator

When $G$ is fixed, the optimal discriminator is:

$$D^*(x) = \frac{p_{\text{data}}(\mathbf{x})}{p_{\text{data}}(\mathbf{x}) + p_g(\mathbf{x})}$$

where $p_g$ is the distribution induced by the generator.

**Proof**: Differentiate objective with respect to $D(\mathbf{x})$ and set to zero

$$\frac{\partial}{\partial D(\mathbf{x})} \left[ p_{\text{data}}(\mathbf{x}) \log D(\mathbf{x}) + p_g(\mathbf{x}) \log(1 - D(\mathbf{x})) \right] = 0$$

$$\frac{p_{\text{data}}(\mathbf{x})}{D(\mathbf{x})} - \frac{p_g(\mathbf{x})}{1 - D(\mathbf{x})} = 0$$

### 3.3 Relationship to JS Divergence

Substituting optimal discriminator $D^*$, the generator's objective becomes:

$$C(G) = -\log 4 + 2 \cdot \text{JS}(p_{\text{data}} \| p_g)$$

where **Jensen-Shannon divergence**:

$$\text{JS}(p \| q) = \frac{1}{2} D_{\text{KL}}\left(p \middle\| \frac{p+q}{2}\right) + \frac{1}{2} D_{\text{KL}}\left(q \middle\| \frac{p+q}{2}\right)$$

**Conclusion**: Training a GAN is equivalent to minimizing JS divergence.

### 3.4 Mathematical Causes of Training Instability

**Gradient vanishing**:
- When $p_g$ and $p_{\text{data}}$ don't overlap, $D$ becomes perfect
- $\nabla_\theta \log(1 - D(G(\mathbf{z}))) \approx 0$

**Mode collapse**:
- Generator produces only some modes ($p_g$ covers only part of $p_{\text{data}}$)
- JS divergence can still be small

```python
# 간단한 GAN 예제 (1D)
class Generator(nn.Module):
    def __init__(self, latent_dim=10, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 50),
            nn.ReLU(),
            nn.Linear(50, output_dim)
        )

    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 50),
            nn.LeakyReLU(0.2),
            nn.Linear(50, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# 모델 초기화
latent_dim = 10
G = Generator(latent_dim=latent_dim, output_dim=1)
D = Discriminator(input_dim=1)

# 손실 함수
criterion = nn.BCELoss()

# 옵티마이저
optimizer_G = torch.optim.Adam(G.parameters(), lr=2e-4)
optimizer_D = torch.optim.Adam(D.parameters(), lr=2e-4)

# 간단한 학습 루프
def train_gan_step(real_data, G, D, optimizer_G, optimizer_D, latent_dim):
    """GAN 1 스텝 학습"""
    batch_size = real_data.size(0)

    # 레이블
    real_labels = torch.ones(batch_size, 1)
    fake_labels = torch.zeros(batch_size, 1)

    # 판별자 학습
    optimizer_D.zero_grad()

    # 진짜 데이터
    D_real = D(real_data)
    loss_D_real = criterion(D_real, real_labels)

    # 가짜 데이터
    z = torch.randn(batch_size, latent_dim)
    fake_data = G(z)
    D_fake = D(fake_data.detach())
    loss_D_fake = criterion(D_fake, fake_labels)

    loss_D = loss_D_real + loss_D_fake
    loss_D.backward()
    optimizer_D.step()

    # 생성자 학습
    optimizer_G.zero_grad()

    z = torch.randn(batch_size, latent_dim)
    fake_data = G(z)
    D_fake = D(fake_data)
    loss_G = criterion(D_fake, real_labels)  # 진짜로 분류되길 원함

    loss_G.backward()
    optimizer_G.step()

    return loss_D.item(), loss_G.item()

# 예제 데이터
real_samples = torch.FloatTensor(generate_data(100)).unsqueeze(1)
loss_d, loss_g = train_gan_step(real_samples, G, D, optimizer_G, optimizer_D, latent_dim)

print(f"판별자 손실: {loss_d:.4f}")
print(f"생성자 손실: {loss_g:.4f}")
```

## 4. Wasserstein Distance and WGAN

### 4.1 Optimal Transport

**Wasserstein-1 distance** (Earth Mover's Distance) between two probability distributions $P, Q$:

$$W_1(P, Q) = \inf_{\gamma \in \Pi(P, Q)} \mathbb{E}_{(\mathbf{x}, \mathbf{y}) \sim \gamma} \left[ \|\mathbf{x} - \mathbf{y}\| \right]$$

where $\Pi(P, Q)$ is the set of all joint distributions with marginals $P$ and $Q$.

**Intuition**: Minimum cost to "transport" $P$ to $Q$

### 4.2 Kantorovich-Rubinstein Duality

**Dual problem**:

$$W_1(P, Q) = \sup_{f: \|f\|_L \leq 1} \mathbb{E}_{\mathbf{x} \sim P}[f(\mathbf{x})] - \mathbb{E}_{\mathbf{y} \sim Q}[f(\mathbf{y})]$$

where $\|f\|_L \leq 1$ is the **1-Lipschitz function** constraint:

$$|f(\mathbf{x}_1) - f(\mathbf{x}_2)| \leq \|\mathbf{x}_1 - \mathbf{x}_2\|$$

### 4.3 WGAN Objective Function

**Replace discriminator with critic $f_w$** (output is not a probability):

$$\min_G \max_{w: f_w \text{ is 1-Lipschitz}} \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[f_w(\mathbf{x})] - \mathbb{E}_{\mathbf{z} \sim p(\mathbf{z})}[f_w(G(\mathbf{z}))]$$

**Advantages**:
- Improved training stability
- Meaningful loss values (approximate distance)
- Reduced mode collapse

### 4.4 Lipschitz Constraint: Gradient Penalty

**Original WGAN**: Weight clipping → limited

**WGAN-GP**: Gradient penalty

$$\mathcal{L} = \mathbb{E}_{\tilde{\mathbf{x}}} [f(\tilde{\mathbf{x}})] - \mathbb{E}_{\mathbf{x}} [f(\mathbf{x})] + \lambda \mathbb{E}_{\hat{\mathbf{x}}} \left[ (\|\nabla_{\hat{\mathbf{x}}} f(\hat{\mathbf{x}})\| - 1)^2 \right]$$

where $\hat{\mathbf{x}} = \epsilon \mathbf{x} + (1 - \epsilon) \tilde{\mathbf{x}}$ is interpolation between real and fake.

```python
def gradient_penalty(D, real_data, fake_data, device='cpu'):
    """
    그래디언트 패널티 계산 (WGAN-GP)

    Parameters:
    -----------
    D : nn.Module
        판별자 (critic)
    real_data : Tensor
        진짜 데이터
    fake_data : Tensor
        가짜 데이터
    device : str
        디바이스

    Returns:
    --------
    gp : Tensor
        그래디언트 패널티
    """
    batch_size = real_data.size(0)

    # 랜덤 보간
    epsilon = torch.rand(batch_size, 1).to(device)
    interpolates = epsilon * real_data + (1 - epsilon) * fake_data
    interpolates.requires_grad_(True)

    # 판별자 출력
    D_interpolates = D(interpolates)

    # 그래디언트 계산
    gradients = torch.autograd.grad(
        outputs=D_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(D_interpolates),
        create_graph=True,
        retain_graph=True
    )[0]

    # 그래디언트 norm
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)

    # 패널티
    gp = ((gradient_norm - 1) ** 2).mean()

    return gp

# 예제
real = torch.randn(32, 1, requires_grad=False)
fake = torch.randn(32, 1, requires_grad=False)

D_critic = Discriminator(input_dim=1)
gp = gradient_penalty(D_critic, real, fake)

print(f"그래디언트 패널티: {gp.item():.4f}")
```

## 5. Mathematics of Diffusion Models

### 5.1 Forward Process

Progressively add Gaussian noise to data $\mathbf{x}_0 \sim q(\mathbf{x}_0)$:

$$q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t I)$$

**Noise schedule**: $\beta_1, \ldots, \beta_T$ (typically $10^{-4} \to 0.02$)

**Closed form**: $\alpha_t = 1 - \beta_t$, $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$

$$q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t) I)$$

That is, in one step $\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}$, $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, I)$

### 5.2 Reverse Process

**Goal**: Recover $\mathbf{x}_0$ from $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, I)$

$$p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \Sigma_\theta(\mathbf{x}_t, t))$$

**Theory**: If $\beta_t$ is sufficiently small, the reverse is also Gaussian.

### 5.3 DDPM Loss Derivation

Deriving the variational lower bound (ELBO, similar to VAE):

$$\mathcal{L} = \mathbb{E}_q \left[ D_{\text{KL}}(q(\mathbf{x}_T|\mathbf{x}_0) \| p(\mathbf{x}_T)) + \sum_{t=2}^T D_{\text{KL}}(q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) \| p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)) - \log p_\theta(\mathbf{x}_0|\mathbf{x}_1) \right]$$

**Key**: Closed-form exists for posterior $q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)$

$$q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0), \tilde{\beta}_t I)$$

where:

$$\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} \mathbf{x}_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t$$

**Reparameterization with noise prediction**:

Substituting $\mathbf{x}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon})$,

**Simplified loss**:

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[ \|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2 \right]$$

That is, it reduces to a **noise prediction** problem!

```python
def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    """선형 노이즈 스케줄"""
    return np.linspace(beta_start, beta_end, timesteps)

def get_diffusion_parameters(timesteps=1000):
    """확산 모델 파라미터 계산"""
    betas = linear_beta_schedule(timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas)
    alphas_cumprod_prev = np.concatenate([[1.0], alphas_cumprod[:-1]])

    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'alphas_cumprod_prev': alphas_cumprod_prev,
        'sqrt_alphas_cumprod': np.sqrt(alphas_cumprod),
        'sqrt_one_minus_alphas_cumprod': np.sqrt(1.0 - alphas_cumprod)
    }

params = get_diffusion_parameters(timesteps=1000)

# 노이즈 추가 예제
def q_sample(x_0, t, params, noise=None):
    """
    정방향 과정: x_0에서 x_t 샘플링

    Parameters:
    -----------
    x_0 : Tensor
        원본 데이터
    t : int
        타임스텝
    params : dict
        확산 파라미터
    noise : Tensor, optional
        노이즈 (None이면 샘플링)

    Returns:
    --------
    x_t : Tensor
        노이즈가 추가된 데이터
    """
    if noise is None:
        noise = torch.randn_like(x_0)

    sqrt_alpha_cumprod_t = params['sqrt_alphas_cumprod'][t]
    sqrt_one_minus_alpha_cumprod_t = params['sqrt_one_minus_alphas_cumprod'][t]

    x_t = sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise

    return x_t

# 예제: 1D 데이터에 노이즈 추가
x_0 = torch.FloatTensor([[1.0], [2.0], [3.0]])
timesteps_to_visualize = [0, 250, 500, 750, 999]

fig, axes = plt.subplots(1, len(timesteps_to_visualize), figsize=(15, 3))

for idx, t in enumerate(timesteps_to_visualize):
    x_t = q_sample(x_0, t, params)
    axes[idx].hist(x_t.numpy(), bins=20, alpha=0.7)
    axes[idx].set_title(f't={t}')
    axes[idx].set_xlabel('Value')
    axes[idx].set_xlim(-4, 4)

plt.tight_layout()
plt.savefig('diffusion_forward_process.png', dpi=150, bbox_inches='tight')
print("확산 정방향 과정 시각화 저장 완료")
```

### 5.4 Relationship to Score Matching

**Score function**: $\nabla_{\mathbf{x}} \log p(\mathbf{x})$

**Score matching**: Approximate score function with neural network

$$\mathcal{L}_{\text{score}} = \mathbb{E}_{p(\mathbf{x})} \left[ \left\| \nabla_{\mathbf{x}} \log p(\mathbf{x}) - s_\theta(\mathbf{x}) \right\|^2 \right]$$

**Connection**: Relationship between noise prediction $\boldsymbol{\epsilon}_\theta$ and score estimation

$$\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t|\mathbf{x}_0) = -\frac{\boldsymbol{\epsilon}}{\sqrt{1 - \bar{\alpha}_t}}$$

Therefore, learning $\boldsymbol{\epsilon}_\theta$ is equivalent to learning the score function.

### 5.5 Langevin Dynamics

**Sampling method**: Markov Chain Monte Carlo using score function

$$\mathbf{x}_{t+1} = \mathbf{x}_t + \epsilon \nabla_{\mathbf{x}} \log p(\mathbf{x}_t) + \sqrt{2\epsilon} \mathbf{z}_t$$

where $\mathbf{z}_t \sim \mathcal{N}(\mathbf{0}, I)$

The reverse process of diffusion models can be viewed as **discretized Langevin dynamics**.

## 6. Recent Advances: Flow Matching and CNF

### 6.1 Continuous Normalizing Flow (CNF)

**Idea**: Transform data distribution via ODE

$$\frac{d\mathbf{x}(t)}{dt} = f_\theta(\mathbf{x}(t), t)$$

**Time $t=0$**: $\mathbf{x}(0) \sim p_0$ (noise)
**Time $t=1$**: $\mathbf{x}(1) \sim p_1$ (data)

**Neural ODE**: Parameterize $f_\theta$ with neural network

### 6.2 Flow Matching

**Goal**: Directly learn vector field $u_t(\mathbf{x})$

**Conditional Flow Matching** (CFM):

$$\mathcal{L}_{\text{CFM}} = \mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}_1} \left[ \left\| u_t(\mathbf{x}_t) - \frac{d\mathbf{x}_t}{dt} \right\|^2 \right]$$

where $\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1$ is linear interpolation.

**Advantages**:
- Simulation-free training
- Faster sampling than diffusion models
- Stable training

### 6.3 Consistency Models

**Idea**: Map all points on ODE trajectory to the same point

$$f_\theta(\mathbf{x}_t, t) = f_\theta(\mathbf{x}_{t'}, t') \quad \forall t, t'$$

**Consistency loss**:

$$\mathcal{L} = \mathbb{E} \left[ d(f_\theta(\mathbf{x}_{t_{n+1}}, t_{n+1}), f_{\theta^-}(\mathbf{x}_{t_n}, t_n)) \right]$$

where $\theta^-$ is EMA (exponential moving average) parameter.

**Advantage**: 1-step generation possible (diffusion models need hundreds of steps)

```python
# 간단한 Flow Matching 예제
class FlowMatchingModel(nn.Module):
    def __init__(self, dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, 64),  # x + t
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, dim)
        )

    def forward(self, x, t):
        """벡터 필드 예측"""
        t_expanded = t.view(-1, 1).expand(-1, x.size(1))
        xt = torch.cat([x, t_expanded[:, :1]], dim=1)
        return self.net(xt)

def flow_matching_loss(model, x0, x1):
    """
    Flow Matching 손실

    Parameters:
    -----------
    model : nn.Module
        벡터 필드 모델
    x0 : Tensor
        시작 분포 샘플 (노이즈)
    x1 : Tensor
        목표 분포 샘플 (데이터)

    Returns:
    --------
    loss : Tensor
        Flow Matching 손실
    """
    batch_size = x0.size(0)

    # 랜덤 시간
    t = torch.rand(batch_size, 1)

    # 선형 보간
    x_t = (1 - t) * x0 + t * x1

    # 진짜 벡터 필드 (선형 보간의 도함수)
    true_velocity = x1 - x0

    # 예측 벡터 필드
    pred_velocity = model(x_t, t.squeeze())

    # 손실
    loss = F.mse_loss(pred_velocity, true_velocity)

    return loss

# 예제
fm_model = FlowMatchingModel(dim=2)
x0_samples = torch.randn(32, 2)
x1_samples = torch.randn(32, 2) + torch.tensor([2.0, 0.0])

loss = flow_matching_loss(fm_model, x0_samples, x1_samples)
print(f"Flow Matching 손실: {loss.item():.4f}")
```

## Practice Problems

### Problem 1: Two Methods of ELBO Derivation
1. Write step-by-step derivation of ELBO using Jensen's inequality.
2. Write step-by-step derivation using KL decomposition.
3. Explain why both methods give the same result.

### Problem 2: VAE Implementation and Experiments
For 2D Gaussian mixture data:
1. Implement a VAE with 2D latent space.
2. Plot training curves of reconstruction loss and KL loss.
3. Visualize latent space and decoder output via grid sampling.
4. Implement $\beta$-VAE (weight $\beta$ on KL term) and analyze the effect of $\beta$.

### Problem 3: Proof of Optimal Discriminator for GAN
When $G$ is fixed, prove that the discriminator's objective:

$$\max_D \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}} [\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{x} \sim p_g} [\log(1 - D(\mathbf{x}))]$$

is maximized at $D^*(\mathbf{x}) = \frac{p_{\text{data}}(\mathbf{x})}{p_{\text{data}}(\mathbf{x}) + p_g(\mathbf{x})}$.

### Problem 4: Forward Process of Diffusion Models
1. Derive $q(\mathbf{x}_t|\mathbf{x}_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1-\bar{\alpha}_t)I)$ recursively.
2. Visualize the distribution of $\mathbf{x}_t$ for 1D data at various $t$.
3. Show that $\mathbf{x}_T \approx \mathcal{N}(\mathbf{0}, I)$ as $T \to \infty$.

### Problem 5: Flow Matching vs Diffusion Models
1. Implement both a simple Flow Matching model and DDPM.
2. Train both models on the same 2D data.
3. Compare sampling speed (number of steps).
4. Compare generation quality (FID, Inception Score, etc.).

## References

### Papers
- **VAE**: Kingma, D. P., & Welling, M. (2014). "Auto-Encoding Variational Bayes." *ICLR*.
- **GAN**: Goodfellow, I., et al. (2014). "Generative Adversarial Nets." *NeurIPS*.
- **WGAN**: Arjovsky, M., et al. (2017). "Wasserstein Generative Adversarial Networks." *ICML*.
- **WGAN-GP**: Gulrajani, I., et al. (2017). "Improved Training of Wasserstein GANs." *NeurIPS*.
- **DDPM**: Ho, J., et al. (2020). "Denoising Diffusion Probabilistic Models." *NeurIPS*.
- **Score-Based Models**: Song, Y., & Ermon, S. (2019). "Generative Modeling by Estimating Gradients of the Data Distribution." *NeurIPS*.
- **Flow Matching**: Lipman, Y., et al. (2023). "Flow Matching for Generative Modeling." *ICLR*.
- **Consistency Models**: Song, Y., et al. (2023). "Consistency Models." *ICML*.

### Online Resources
- [Lil'Log: From Autoencoder to Beta-VAE (Lilian Weng)](https://lilianweng.github.io/posts/2018-08-12-vae/)
- [Lil'Log: What are Diffusion Models? (Lilian Weng)](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- [Understanding Diffusion Models (Calvin Luo)](https://arxiv.org/abs/2208.11970)
- [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion)

### Libraries
- `torch`: PyTorch implementation
- `diffusers` (Hugging Face): Diffusion model library
- `stable-diffusion`: Stable Diffusion implementation
