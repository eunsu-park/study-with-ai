---
title: "Pre-Reading Briefing: Denoising Diffusion Probabilistic Models"
paper_id: "35_ho_2020"
topic: Artificial_Intelligence
date: 2026-04-28
type: briefing
---

# Denoising Diffusion Probabilistic Models (DDPM): Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. *Advances in Neural Information Processing Systems (NeurIPS) 33*. arXiv:2006.11239.
**Author(s)**: Jonathan Ho, Ajay Jain, Pieter Abbeel (UC Berkeley)
**Year**: 2020

---

## 1. 핵심 기여 / Core Contribution

**한국어**
이 논문은 **확산 확률 모델(Diffusion Probabilistic Models)**을 GAN과 경쟁할 수 있는 수준의 고품질 이미지 생성기로 끌어올린 작품입니다. 핵심 기여는 두 가지입니다. 첫째, 저자들은 평균 $\mu_\theta$를 직접 예측하는 대신 **노이즈 $\epsilon$을 예측하는 재매개변수화(reparameterization)**를 제안합니다 — 이는 결국 변분 하한(ELBO)을 매우 단순한 가중 MSE 손실 $L_\text{simple}$로 환원시킵니다. 둘째, 이 단순한 손실이 **다중 노이즈 스케일에서의 denoising score matching** 및 Langevin dynamics와 정확히 일치함을 보입니다. 결과적으로 CIFAR10에서 unconditional FID 3.17 (당시 SOTA), Inception Score 9.46을 달성하며 LSUN 256×256과 CelebA-HQ 256×256에서 ProgressiveGAN에 필적하는 샘플 품질을 보입니다.

**English**
This paper elevates **diffusion probabilistic models** from theoretical curiosities into a practical, GAN-competitive image generator. The contribution is twofold. First, the authors introduce a **noise-prediction reparameterization** — instead of predicting the reverse-process mean $\mu_\theta$, the network predicts the noise $\epsilon$ — which reduces the variational bound to a remarkably simple weighted MSE loss $L_\text{simple}$. Second, they show this simplified objective is precisely equivalent to **denoising score matching across multiple noise scales** combined with annealed Langevin sampling. The result: on unconditional CIFAR10 they reach FID 3.17 (SOTA at publication) and Inception Score 9.46, with LSUN-256 and CelebA-HQ-256 samples rivaling ProgressiveGAN.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

**한국어**
2014~2019년은 GAN의 황금기였지만, 다음과 같은 한계가 점점 드러났습니다: 학습 불안정성(mode collapse), 명시적 가능도(likelihood) 부재, 다양성 부족. 한편 **확산 모델**은 Sohl-Dickstein et al. (2015) "Deep Unsupervised Learning using Nonequilibrium Thermodynamics" 논문에서 비평형 열역학 영감을 받아 제시되었으나, 5년간 샘플 품질이 GAN을 따라가지 못해 잊혀진 분야였습니다. 동시기에 Yang Song & Stefano Ermon (2019, NCSN)은 **score-based generative model**을 제안 — Langevin dynamics로 데이터 분포의 score $\nabla_x \log p(x)$를 따라가며 샘플링하는 접근. DDPM은 이 두 흐름이 본질적으로 같음을 명시적으로 보이고, 단순한 노이즈 예측 손실로 통합한 결정적 작품입니다.

**English**
The years 2014–2019 were the golden age of GANs, but cracks were showing: training instability, mode collapse, lack of explicit likelihoods, and limited diversity. **Diffusion models** had been proposed by Sohl-Dickstein et al. (2015) ("Deep Unsupervised Learning using Nonequilibrium Thermodynamics") inspired by non-equilibrium thermodynamics, but for five years their sample quality lagged GANs and the field was largely dormant. In parallel, Song & Ermon (2019, NCSN) proposed **score-based generative models** — sampling via Langevin dynamics that follow the data score $\nabla_x \log p(x)$. DDPM is the decisive paper that shows these two threads are essentially the same and unifies them under a single, simple noise-prediction loss.

### 타임라인 / Timeline

```
1907 ─ Langevin: stochastic differential equations (Langevin dynamics)
2005 ─ Hyvärinen: Score Matching
2011 ─ Vincent: Denoising Score Matching ↔ DAE connection
2013 ─ Kingma & Welling: VAE (paper #15) — variational inference framework
2014 ─ Goodfellow et al.: GAN (paper #16)
2015 ─ Sohl-Dickstein et al.: Diffusion probabilistic models (founder paper)
2018 ─ Karras et al.: ProgressiveGAN (256² benchmark)
2019 ─ Song & Ermon: NCSN (Noise Conditional Score Network)
2019 ─ Song & Ermon: NCSNv2 (improved score matching)
2020 ─ ★★★ Ho, Jain, Abbeel: DDPM (this paper) ★★★
2021 ─ Nichol & Dhariwal: Improved DDPM
2021 ─ Song et al.: Score SDE (continuous-time unification)
2021 ─ Dhariwal & Nichol: Diffusion beats GANs on image synthesis
2022 ─ Rombach et al.: Latent Diffusion (Stable Diffusion)
2022 ─ Saharia et al.: Imagen
2022 ─ Ramesh et al.: DALL·E 2
```

---

## 3. 필요한 배경 지식 / Prerequisites

**한국어**
1. **변분 추론과 ELBO** — VAE(논문 #15)에서 익힌 $\log p_\theta(x) \geq \mathbb{E}_q[\log p_\theta(x,z)/q(z|x)]$. DDPM의 학습 손실은 이 ELBO의 일반화입니다.
2. **Markov chain과 Gaussian transition** — 가우시안 분포의 합성, KL divergence 사이의 닫힌 형태 식.
3. **Reparameterization trick** — $\mathcal{N}(\mu, \sigma^2 I)$에서의 샘플링을 $\mu + \sigma \epsilon$, $\epsilon \sim \mathcal{N}(0, I)$로 표현.
4. **Score matching** — 데이터 분포의 score $\nabla_x \log p(x)$를 추정하는 학습 방법. denoising score matching: $\mathbb{E}[\|s_\theta(\tilde x) - \nabla_{\tilde x} \log q(\tilde x|x)\|^2]$.
5. **Langevin dynamics** — $x_{t+1} = x_t + \frac{\delta}{2}\nabla \log p(x_t) + \sqrt{\delta}\, z$, $z \sim \mathcal{N}(0,I)$. 충분한 시간 후 $p(x)$로 수렴.
6. **U-Net 구조** — Ronneberger et al. (2015), encoder-decoder + skip connection. 본 논문에서 reverse 함수 $\epsilon_\theta$의 백본.
7. **Sinusoidal positional embedding** — Transformer (논문 #25)의 시간 t 인코딩.
8. **FID, Inception Score** — 생성 모델 평가지표.

**English**
1. **Variational inference & ELBO** — from VAE (paper #15): $\log p_\theta(x) \geq \mathbb{E}_q[\log p_\theta(x,z)/q(z|x)]$. DDPM's training loss is a generalized ELBO.
2. **Markov chains with Gaussian transitions** — Gaussian convolution closure, closed-form KL between Gaussians.
3. **Reparameterization trick** — sampling from $\mathcal{N}(\mu,\sigma^2 I)$ as $\mu + \sigma\epsilon$, $\epsilon\sim\mathcal{N}(0,I)$.
4. **Score matching** — estimating $\nabla_x \log p(x)$. Denoising score matching: $\mathbb{E}[\|s_\theta(\tilde x) - \nabla_{\tilde x}\log q(\tilde x | x)\|^2]$.
5. **Langevin dynamics** — $x_{t+1} = x_t + (\delta/2)\nabla\log p(x_t) + \sqrt{\delta}\, z$ converges to $p(x)$.
6. **U-Net** — Ronneberger et al. (2015), encoder-decoder with skip connections; used as the backbone for $\epsilon_\theta$.
7. **Sinusoidal positional embedding** — from Transformer (paper #25), used here to embed timestep $t$.
8. **FID and Inception Score** — generative model evaluation metrics.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **Forward (diffusion) process** $q(x_t|x_{t-1})$ | 데이터 $x_0$에 점진적으로 가우시안 노이즈를 더하는 고정된 Markov chain. $x_T$는 거의 순수 노이즈. / Fixed Markov chain that gradually adds Gaussian noise to data $x_0$ until $x_T$ is nearly pure noise. |
| **Reverse process** $p_\theta(x_{t-1}|x_t)$ | 노이즈 $x_T$로부터 데이터 $x_0$로 가는 학습된 Markov chain. 가우시안 conditional로 매개변수화. / Learned Markov chain mapping noise $x_T$ back to data $x_0$, parameterized by Gaussian conditionals. |
| **$\beta_t$ (variance schedule)** | $t$번째 단계의 노이즈 분산. 본 논문은 $\beta_1=10^{-4}$에서 $\beta_T=0.02$까지 선형. / Variance at step $t$. Linear schedule from $10^{-4}$ to $0.02$ over $T=1000$ steps. |
| **$\alpha_t = 1-\beta_t$, $\bar\alpha_t = \prod_{s=1}^t \alpha_s$** | 누적 시그널 보존률. $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon$. / Cumulative signal preservation: $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$. |
| **ELBO / Variational bound $L$** | 음의 로그가능도의 상한. KL divergence들의 합으로 분해. / Upper bound on negative log-likelihood; decomposes into KL divergences. |
| **$L_\text{simple}$** | DDPM의 단순화된 손실: 진짜 노이즈 $\epsilon$과 예측 노이즈 $\epsilon_\theta$의 MSE. / Simplified DDPM loss: MSE between true and predicted noise. |
| **$\epsilon$-prediction parameterization** | 평균 $\mu_\theta$ 대신 노이즈 $\epsilon_\theta$를 예측. score matching과 동치. / Predict noise $\epsilon_\theta$ instead of mean $\mu_\theta$; equivalent to score matching. |
| **U-Net** | encoder-decoder 구조 + skip connection. PixelCNN++/Wide ResNet 기반. / Encoder-decoder with skip connections, here based on PixelCNN++/Wide ResNet. |
| **Sinusoidal time embedding** | Transformer 식 시간 $t$ 인코딩. residual block마다 더해짐. / Transformer-style embedding of $t$, added in each residual block. |
| **Langevin dynamics** | score를 따르며 노이즈를 더하는 샘플링. DDPM의 reverse step과 등가. / Sampling that follows the score plus noise; equivalent to DDPM's reverse step. |
| **Denoising score matching** | 노이즈가 추가된 데이터로부터 score를 추정하는 학습법. $\epsilon$-prediction과 등가. / Estimates the score from noised data; equivalent to $\epsilon$-prediction. |
| **FID (Fréchet Inception Distance)** | 생성 샘플과 실제 데이터의 Inception 특징 분포 거리. 낮을수록 좋음. / Distance between generated and real Inception-feature distributions; lower is better. |

---

## 5. 수식 미리보기 / Equations Preview

**한국어**

**(1) Forward process (식 2):** 데이터에 가우시안 노이즈를 더합니다.
$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}\, x_{t-1},\ \beta_t I)$$
$\sqrt{1-\beta_t}$로 신호를 살짝 줄이면서 $\beta_t$ 분산의 노이즈를 더합니다 — 이렇게 하면 분산이 폭발하지 않고 안정적입니다.

**(2) Closed-form forward (식 4):** 임의의 $t$에서 한 번에 샘플링.
$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar\alpha_t}\, x_0,\ (1-\bar\alpha_t) I), \quad \bar\alpha_t = \prod_{s=1}^t (1-\beta_s)$$
**핵심**: 학습 중에는 임의의 $t$를 뽑아 한 단계로 $x_t$를 만들 수 있습니다 — Markov chain을 시뮬레이션할 필요 없음.

**(3) Reverse process (식 1):**
$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t),\ \Sigma_\theta(x_t, t))$$
DDPM은 $\Sigma_\theta = \sigma_t^2 I$로 고정 (학습하지 않음).

**(4) Simplified loss (식 14):**
$$L_\text{simple}(\theta) = \mathbb{E}_{t, x_0, \epsilon}\!\left[\big\|\epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\ t)\big\|^2\right]$$
**해석**: 무작위 $t$, 무작위 $\epsilon$을 뽑아 $x_t$를 만들고, 네트워크가 그 $\epsilon$을 다시 맞추도록 학습.

**(5) Sampling step (Algorithm 2, 식 11):**
$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\!\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t, t)\right) + \sigma_t z, \quad z \sim \mathcal{N}(0, I)$$
이 한 줄이 Langevin 한 스텝과 같습니다.

**English**

**(1) Forward process (Eq. 2):** add Gaussian noise gradually:
$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}\, x_{t-1},\ \beta_t I)$$
The signal is slightly shrunk by $\sqrt{1-\beta_t}$ while $\beta_t$-variance noise is added — keeping variance stable.

**(2) Closed-form (Eq. 4):**
$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar\alpha_t}\, x_0,\ (1-\bar\alpha_t)I)$$
**Key**: training samples any $t$ in one shot — no Markov-chain simulation.

**(3) Reverse process (Eq. 1):**
$$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t,t),\ \Sigma_\theta(x_t,t))$$
DDPM fixes $\Sigma_\theta = \sigma_t^2 I$.

**(4) Simplified loss (Eq. 14):**
$$L_\text{simple} = \mathbb{E}_{t,x_0,\epsilon}\!\left[\|\epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)\|^2\right]$$
Pick random $(t,\epsilon)$, form $x_t$, ask the net to recover $\epsilon$.

**(5) Sampling step:**
$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\!\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right) + \sigma_t z$$
One line — equivalent to one Langevin step.

---

## 6. 읽기 가이드 / Reading Guide

**한국어**
- **§1 Introduction**: 제목과 그림 1만 훑고 넘어가도 됩니다. 핵심은 "diffusion이 GAN과 경쟁한다"는 결과 문장.
- **§2 Background**: 가장 중요. 식 (1)–(7)을 직접 종이에 적어가며 따라가세요. 특히 식 (4)의 closed-form은 학습 알고리즘을 한 줄로 만드는 마법.
- **§3 Diffusion models and denoising autoencoders**: 논문의 진짜 기여. §3.2의 $\epsilon$-prediction 재매개변수화 → 식 (12) → §3.4의 $L_\text{simple}$ 흐름을 정확히 이해하면 80% 이해한 것.
- **§4 Experiments**: §4.2 ablation 표(Table 2)에서 $\epsilon$-prediction이 $\mu$-prediction보다 훨씬 안정적임을 확인.
- **Algorithm 1 & 2**: 학습 6줄, 샘플링 5줄. 이 11줄을 외우세요 — 그것이 DDPM의 본질입니다.
- **§4.3 Progressive coding**: rate-distortion 관점은 후속 연구(Improved DDPM, classifier-free guidance)의 토대.
- **Appendix A**: ELBO 분해의 자세한 유도. 처음에는 건너뛰고 두 번째 읽기에서.

**English**
- **§1 Introduction**: skim. Just note the headline result that diffusion now rivals GANs.
- **§2 Background**: most important. Re-derive Eqs. (1)–(7) on paper. The closed-form (4) is the magic that makes training a one-liner.
- **§3 Diffusion models and denoising autoencoders**: the real contribution. Trace §3.2 ($\epsilon$-prediction reparameterization) → Eq. (12) → §3.4 ($L_\text{simple}$); understanding this trio is 80% of the paper.
- **§4 Experiments**: in Table 2 confirm that $\epsilon$-prediction is far more stable than $\mu$-prediction.
- **Algorithms 1 & 2**: training is 6 lines, sampling is 5 lines. Memorize these 11 lines — they are DDPM's essence.
- **§4.3 Progressive coding**: rate-distortion view foreshadows Improved DDPM and classifier-free guidance.
- **Appendix A**: detailed ELBO derivation; skip on first pass.

---

## 7. 현대적 의의 / Modern Significance

**한국어**
DDPM 이후 5년 만에 **확산 모델은 생성형 AI의 표준 백본**이 되었습니다. Stable Diffusion, DALL·E 2, Imagen, Sora, Veo 등 거의 모든 텍스트→이미지/비디오 시스템이 DDPM의 직접적 후예입니다. 본 논문이 정립한 **세 가지 설계 원칙** — (1) $\epsilon$-prediction 재매개변수화, (2) $L_\text{simple}$ 가중 MSE, (3) 시간 $t$를 sinusoidal embedding으로 conditioning — 은 거의 그대로 유지되어 현재 파운데이션 모델의 핵심을 이룹니다.

또한 본 논문은 **GAN 시대의 종식과 likelihood-based 모델의 부활**을 알린 신호탄입니다. 2021년 Dhariwal & Nichol "Diffusion beats GANs"가 ImageNet 256에서 결정타를 날렸고, 2022년 latent diffusion (Stable Diffusion)이 등장하며 diffusion이 사실상 표준이 되었습니다. score matching, energy-based model, autoregressive decoding과의 통합 시각도 본 논문이 제시했습니다.

**English**
Five years after publication, **diffusion models have become the standard backbone of generative AI**. Stable Diffusion, DALL·E 2, Imagen, Sora, Veo — nearly every text-to-image or text-to-video system is a direct descendant of DDPM. The three design principles established here — (1) $\epsilon$-prediction, (2) the weighted-MSE $L_\text{simple}$ objective, (3) sinusoidal time conditioning — survive nearly unchanged in today's foundation models.

DDPM also signaled **the end of the GAN era and the resurgence of likelihood-based generative modeling**. Dhariwal & Nichol's 2021 "Diffusion beats GANs" delivered the knockout on ImageNet-256; latent diffusion (Stable Diffusion, 2022) made diffusion the practical default. The unifying view connecting score matching, energy-based models, and autoregressive decoding — also articulated here — became the conceptual frame for the entire field.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
