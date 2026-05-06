---
title: "Ambient Diffusion: Learning Clean Distributions from Corrupted Data"
authors: Giannis Daras, Kulin Shah, Yuval Dagan, Aravind Gollakota, Alexandros G. Dimakis, Adam Klivans
year: 2023
venue: "NeurIPS 2023"
arxiv: "2305.19256"
topic: Low-SNR Imaging / Self-supervised Generative Modelling
tags: [diffusion-model, self-supervised, corruption, ambient-diffusion, denoising, noise2noise, training-with-corruption, distribution-learning]
status: completed
date_started: 2026-05-06
date_completed: 2026-05-06
---

# 29. Ambient Diffusion: Learning Clean Distributions from Corrupted Data / Ambient Diffusion: 손상된 데이터에서 깨끗한 분포 학습

---

## 1. Core Contribution / 핵심 기여

### 한국어
이 논문은 **깨끗한 학습 데이터 없이 손상된(corrupted) 표본만으로 확산 모델을 학습**하여 깨끗한 분포를 복원하는 첫 일반 프레임워크 **Ambient Diffusion**을 제시한다.

(i) 문제 설정: 학습 시 관측되는 것은 $\tilde{\boldsymbol x}_0 = \mathcal A_\phi(\boldsymbol x_0)$ 형태의 손상된 표본만 (예: 90% 픽셀 결손, MRI 블록 손상, 압축 측정). 깨끗한 $\boldsymbol x_0$는 절대 보지 못함. 목표: 그래도 $p(\boldsymbol x_0)$를 학습.

(ii) **핵심 아이디어 — 추가 손상(further corruption)**: 학습 입력에 **두 번째 더 강한 손상 마스크** $\tilde{\mathcal A}_\psi$를 한 번 더 적용하고, 모델은 이 이중 손상에서 *원래 한 번 손상된* $\tilde{\boldsymbol x}_0$를 복원하도록 학습. 이는 Noise2Noise(Lehtinen 2018)의 진정한 일반화.

(iii) **이론적 정당성 (Theorem 1)**: 적절한 조건 하에서 모델이 학습하는 것은 **이중-손상이 주어졌을 때 단일-손상의 조건부 기대값 $\mathbb E[\tilde{\boldsymbol x}_0 \mid \tilde{\tilde{\boldsymbol x}}_0]$**, 그리고 이 추정 + 추론 시 mask 인지를 통해 **깨끗한 분포 $p(\boldsymbol x_0)$를 정확히 복원**할 수 있다 (inpainting 등 임의 invertible 손상 부류에 대해).

(iv) **실험**: CelebA, CIFAR-10, AFHQ에서 학습 데이터의 **90% 픽셀 결손** 상태로도 깨끗한 분포 학습 성공. MRI 블록 손상 데이터로 foundation model을 fine-tune 했을 때 환자 정보 **암기 없이** 분포 학습 가능 — privacy-preserving generative modelling의 길.

### English
**Ambient Diffusion** is the first general framework for **training a diffusion model from only corrupted samples** while still learning the underlying clean distribution.

(i) Problem: at training time we observe only $\tilde{\boldsymbol x}_0 = \mathcal A_\phi(\boldsymbol x_0)$ (e.g., 90% pixel-mask, MRI block masks, compressed-sensing measurements). Clean $\boldsymbol x_0$ is never seen.

(ii) **Key idea — further corruption**: at training time apply a **second, stronger corruption** $\tilde{\mathcal A}_\psi$ on top of the already-corrupted sample, and ask the network to predict the **once-corrupted** $\tilde{\boldsymbol x}_0$ from the **twice-corrupted** input. This is the strict generalisation of Noise2Noise (Lehtinen 2018) to the diffusion setting.

(iii) **Theory (Theorem 1)**: under technical conditions the optimum of the corrupted-data loss equals $\mathbb E[\tilde{\boldsymbol x}_0 \mid \tilde{\tilde{\boldsymbol x}}_0]$, and this — together with mask-conditioning at inference — recovers the clean distribution $p(\boldsymbol x_0)$ exactly. The result holds for any corruption family satisfying mild conditions, including inpainting and compressed sensing.

(iv) **Experiments**: on CelebA, CIFAR-10, and AFHQ with **90% missing pixels** at training time, Ambient Diffusion learns the clean distribution. Foundation-model fine-tuning on block-corrupted MRI scans avoids memorising patient images — privacy-preserving generative modelling.

---

## 2. Reading Notes / 읽기 노트

### Part I: §1 Introduction / 도입

#### 한국어
- **동기 1 — 데이터 가용성**: 블랙홀 사진(EHT, 직접 관측 자체가 손상된 측정), 고해상도 MRI(움직임 아티팩트, 긴 스캔 시간), 천체관측(coronagraph 차폐), 저용량 의료 이미지 등 깨끗한 표본 자체가 존재하지 않거나 비싼 경우.
- **동기 2 — 메모리화 회피**: Carlini et al. 2023, Somepalli et al. 2023, Jagielski et al. 2023이 보였듯 확산 모델은 학습 표본을 *그대로* 암기 → 의료/저작권 문제. 의도적 손상으로 학습 → 개별 표본을 그대로 재생할 수 없음.
- **이전 시도들의 한계**:
  - **AmbientGAN (Bora et al. 2018)**: GAN으로 손상 데이터 학습 가능. discriminator가 손상 적용 여부를 분간 못 하게 함. mode collapse / unstable.
  - **Inpainting heuristics**: 손상된 영역을 단순 평균으로 채워 학습 → 빠른 편향 누적.
  - **Noise2Noise / Noise2Void (Lehtinen 2018, Krull 2019)**: regression 문제에 한정. 분포 학습 X.

#### English
- Two motivations: (a) clean data is unavailable/expensive (radio-astronomy black-hole images, MRI, microscopy); (b) memorisation hazards in modern diffusion models — corrupted training is a built-in privacy mechanism.
- Prior attempts: AmbientGAN (Bora 2018) — GAN-based, unstable; Noise2Noise/Noise2Void — regression only, no distribution learning.

---

### Part II: §2 Background / 배경

#### 한국어
- **확산 모델 표준 학습**: $\mathcal L(\theta) = \mathbb E_{\boldsymbol x_0, t, \boldsymbol\epsilon} \big\| \boldsymbol\epsilon - \boldsymbol\epsilon_\theta(\boldsymbol x_t, t) \big\|^2$, $\boldsymbol x_t = \sqrt{\bar\alpha_t}\boldsymbol x_0 + \sqrt{1-\bar\alpha_t}\boldsymbol\epsilon$. 또는 동등한 $\boldsymbol x_0$-prediction:
  $\mathcal L(\theta) = \mathbb E \big\|\boldsymbol x_0 - \hat{\boldsymbol x}_{0,\theta}(\boldsymbol x_t, t)\big\|^2$, 최적 $\hat{\boldsymbol x}_{0,\theta}^* = \mathbb E[\boldsymbol x_0 \mid \boldsymbol x_t]$.
- **손상 모델**: forward operator $\mathcal A_\phi: \mathbb R^d \to \mathbb R^d$ ($\phi$는 손상 파라미터, 예: 마스크). Inpainting의 경우 $\mathcal A_\phi(\boldsymbol x) = \boldsymbol M_\phi \odot \boldsymbol x$, $\boldsymbol M_\phi \in \{0,1\}^d$ 마스크.
- **Noise2Noise 직관**: 두 noise 인스턴스 $y_1 = x + n_1, y_2 = x + n_2$가 있고 $\mathbb E[n_2 \mid n_1] = 0$이면, $\arg\min_\theta \mathbb E\|y_2 - f_\theta(y_1)\|^2 = \mathbb E[y_2 \mid y_1] = x$. 두 잡음 표본이면 깨끗한 표본 없이 $x$ 추정 가능.

#### English
- Standard diffusion training minimises $\mathbb E\|\boldsymbol\epsilon - \boldsymbol\epsilon_\theta(\boldsymbol x_t, t)\|^2$, equivalent to $\mathbb E\|\boldsymbol x_0 - \hat{\boldsymbol x}_{0,\theta}\|^2$ whose Bayes-optimal solution is the posterior mean.
- Corruption operator $\mathcal A_\phi$ is a known parametrised map (e.g., binary mask for inpainting).
- Noise2Noise: with two noisy instances of the same signal and zero-mean conditional noise, regressing one onto the other recovers the clean signal in expectation.

---

### Part III: §3 Method — Ambient Diffusion Training / 방법

#### 한국어
**알고리즘 (Algorithm 1)**:
입력: 손상된 데이터셋 $\{\tilde{\boldsymbol x}_0^{(i)} = \mathcal A_{\phi_i}(\boldsymbol x_0^{(i)})\}_{i=1}^N$, 두 번째 손상 분포 $q(\psi \mid \phi)$ (단, $\tilde{\mathcal A}_\psi$는 $\mathcal A_\phi$보다 더 손상시킴).

```
for each training step:
    sample i, t ~ Uniform
    sample x_tilde = corrupted dataset[i]   # phi_i is known
    sample psi ~ q(psi | phi_i)             # second corruption
    x_doubletilde = A_psi(x_tilde)          # additional corruption applied at training time
    x_t = sqrt(alpha_bar_t) * x_doubletilde + sqrt(1 - alpha_bar_t) * eps
    pred = network(x_t, t, psi)             # mask-conditioned network
    loss = || A_psi(x_tilde) − A_psi(pred) ||^2   # only score in observed-by-psi region
    backprop
```

**핵심 트릭**:
1. **이중 손상**: $\tilde{\boldsymbol x}_0 \to \tilde{\tilde{\boldsymbol x}}_0 = \tilde{\mathcal A}_\psi(\tilde{\boldsymbol x}_0)$. 모델은 $\tilde{\tilde{\boldsymbol x}}_0$의 noisy 버전을 보고 $\tilde{\boldsymbol x}_0$를 복원하도록 학습.
2. **마스크 인지 손실**: loss를 $\tilde{\mathcal A}_\psi$가 *제거한 영역*에서만 측정 — 즉 $\boldsymbol M_\psi \odot (\tilde{\boldsymbol x}_0 - \hat{\tilde{\boldsymbol x}}_{0,\theta})$. 모델이 마스크 정보를 입력으로 받아 어디를 예측해야 하는지 명시.
3. **추론 시**: 표본 추출 시 $\psi$를 데이터 도메인의 항등식 ($\boldsymbol M_\psi = \boldsymbol 1$)로 설정하면 마스크 없이 깨끗한 표본 생성. 학습된 표본 평균이 자동으로 inpainting을 수행한다.

#### English
The training algorithm samples a corrupted image from the dataset (single corruption $\phi$), draws a stronger second corruption $\psi$, applies it, diffuses, and asks the network — conditioned on $\psi$ — to predict the once-corrupted image. The loss is computed **only on pixels that $\psi$ removed**, so the model learns only from the residual signal that the further corruption blocks.

---

### Part IV: §3.2 Theory / 이론

#### 한국어 — Theorem 1 (Loss equivalence)

가정: 손상 $\mathcal A_\phi$가 픽셀별 마스킹의 형태 (또는 보다 일반적으로 random measurement family). 추가 손상 $\tilde{\mathcal A}_\psi$가 $\mathcal A_\phi$보다 strictly stronger. 데이터 분포가 $\sigma$-algebra full-support 조건을 만족.

**주장**: Ambient 손실의 최소자
$$
\hat{\boldsymbol x}_{0,\theta}^*(\tilde{\tilde{\boldsymbol x}}_0, t, \psi) = \mathbb E[\tilde{\boldsymbol x}_0 \mid \tilde{\tilde{\boldsymbol x}}_0, t, \psi]
$$
이고, 이 conditional expectation은 $\tilde{\boldsymbol x}_0$로부터 $\boldsymbol x_0$로의 marginalisation을 거쳐 (적절한 mask conditioning 하에) **깨끗한 분포의 score**를 복원한다.

**증명 골격**:
1. squared loss에 대한 Bayes-optimal regression → $\mathbb E[\tilde{\boldsymbol x}_0 \mid \tilde{\tilde{\boldsymbol x}}_0, \psi]$.
2. inpainting의 마스크 구조 하에서 $\tilde{\tilde{\boldsymbol x}}_0$가 $\boldsymbol x_0$에 대해 조건부 sufficient statistic; mask 정보가 주어지면 손상되지 않은 영역에서 $\tilde{\boldsymbol x}_0 \equiv \boldsymbol x_0$.
3. 따라서 모든 partial-mask 표본을 합치면 full $\boldsymbol x_0$의 marginal score 복원.

이 결과는 **clean training data를 본 적 없는** 모델이 **clean 분포의 score를 학습할 수 있다**는 것을 의미.

#### English — Theorem 1
Under technical conditions (notably: the corruption family must be expressive enough so that across $\phi$ every pixel is observed in some sample, and the additional corruption $\psi$ must strictly remove information beyond $\phi$), the Bayes-optimal predictor of the ambient loss equals $\mathbb E[\tilde{\boldsymbol x}_0 \mid \tilde{\tilde{\boldsymbol x}}_0, t, \psi]$, which together with mask-conditioning recovers the clean distribution's score function. The proof uses (a) optimality of the conditional mean for squared loss and (b) sufficiency of the multi-mask aggregate observation for the underlying clean image.

---

### Part V: §4 Experiments / 실험

#### 한국어
- **Datasets**: CelebA-64, CIFAR-10, AFHQ-64, MRI (block-corrupted scans).
- **Corruption levels**: random pixel masks with **survival probabilities $p \in \{0.1, 0.2, \ldots, 0.9\}$**. $p = 0.1$ = 90% missing.
- **Baselines**: clean-data DDPM (oracle upper bound), AmbientGAN, naive imputation (zero-fill, mean-fill).

**핵심 정량 결과 (Table 1, FID↓ 일부)**:

| Method                         | CelebA $p=0.1$ | CelebA $p=0.5$ | AFHQ $p=0.1$ |
|--------------------------------|----------------|----------------|---------------|
| Clean DDPM (oracle)            | 5.45           | 5.45           | 7.20          |
| AmbientGAN                     | 36.4           | 22.1           | 41.3          |
| Naive imputation + DDPM        | 19.8           | 9.7            | 21.5          |
| **Ambient Diffusion (ours)**   | **8.85**       | **6.21**       | **9.4**       |

(논문 발췌. 관측 픽셀이 10%에 불과해도 Ambient Diffusion은 oracle로부터 ~3.4 FID 차이까지 따라잡음.)

- **MRI fine-tuning**: 사전학습된 LDM을 block-corrupted brain MRI scan에 fine-tune. **memorisation test**(학습 표본과 표집 표본의 nearest-neighbor distance) 결과: 깨끗한 학습 시 일부 표본이 사실상 복제되는 반면, Ambient 학습 시 NN 거리 분포가 unconditional generation 수준으로 유지.
- **Compressed sensing variant (§4.4)**: corruption이 random Gaussian projection $\boldsymbol y = \boldsymbol G \boldsymbol x$, $\boldsymbol G \in \mathbb R^{m \times d}$. 측정 차원 $m \ll d$임에도 불구하고 distribution learning 가능 — random subspaces 부족분이 더 강한 random subspace로 보충됨을 보임.

#### English
- Datasets/corruptions and clean-DDPM oracle as above. AmbientGAN and naive-imputation baselines.
- Headline: even with **90% missing pixels** at training time, FID is within ~3.4 of the clean-data oracle on CelebA, dramatically beating AmbientGAN and naive imputation.
- MRI experiment shows Ambient fine-tuning preserves training-set privacy: NN distances from generated samples to the training set remain in the random-baseline range, unlike clean fine-tuning which produces near-replicas of patient scans.
- Compressed-sensing variant works with random Gaussian measurements $\boldsymbol y = \boldsymbol G\boldsymbol x$.

---

### Part VI: §5 Discussion & Limitations / 토론과 한계

#### 한국어
- **마스크 분포의 지원(support) 조건**: 데이터셋 전체에 걸쳐 모든 픽셀이 적어도 가끔 관측되어야 함. 항상 같은 영역만 가려진다면 그 영역은 학습 불가능.
- **추가 손상 $\psi$의 강도**: 너무 약하면 학습 신호 부족, 너무 강하면 미관측 픽셀이 너무 많아 noise. 실험적으로 $p_\psi = p_\phi - \delta$ ($\delta = 0.1$) 정도가 sweet spot.
- **Training cost**: 깨끗한 데이터 학습보다 약간 증가 (mask sampling 추가). 하지만 큰 데이터 도메인에서 깨끗한 표본 비용 대비 절감은 막대.
- **Open**: nonlinear corruption (e.g., quantisation, JPEG)으로 확장, score-matching 손실의 보다 나은 변형, 정량적 SNR-vs-quality trade-off 곡선.

#### English
- The mask family must cover every pixel in expectation (full support across $\phi$).
- $\psi$ must strictly extend $\phi$ but not too aggressively — empirical sweet spot around $p_\psi = p_\phi - 0.1$.
- Slight training overhead (mask sampling), negligible vs the saved cost of clean-data acquisition.
- Open directions: nonlinear corruption, sharper SNR/quality trade-off, theoretical guarantees beyond inpainting/compressed sensing.

---

## 3. Key Takeaways / 핵심 시사점

### 한국어
1. **Noise2Noise의 generative 일반화** — 회귀 → 분포 학습. 두 인스턴스가 서로 conditionally independent하면 한쪽으로부터 다른 쪽 예측 = clean signal 추정.
2. **Loss-equivalence가 핵심** — 손상된 입력으로 학습한 squared loss의 최적자가 underlying clean distribution의 (조건부) 평균/score와 정확히 일치한다는 결정론적 증명.
3. **추가 손상이 직관에 반함** — "더 망가뜨려서 학습한다"는 역설적 디자인. 모델이 보지 못한 영역에서 학습 신호가 정의되도록 강제하는 트릭.
4. **마스크 인지 네트워크가 필수** — 모델은 어떤 픽셀을 예측해야 하는지 알아야 함. mask를 추가 입력 채널로 conditioning.
5. **Privacy의 부산물** — 학습 데이터를 그대로 본 적 없으므로 정확한 복제 불가능. 의료 데이터 등 민감 도메인의 generative modelling 가능.
6. **Inpainting/Compressed Sensing이 주된 적용 부류** — 일반 nonlinear corruption은 미해결. 그러나 scientific imaging 문제 다수가 이 부류.
7. **Foundation model fine-tuning** — 큰 사전학습 모델을 작은 손상 데이터로 fine-tune해 도메인 적응 + privacy 동시 달성. 의료 영상의 실용 경로.

### English
1. **Generative analogue of Noise2Noise** — squared loss with conditionally independent corrupted pairs recovers the clean signal in expectation.
2. **Loss-equivalence is the math** — the Bayes-optimal regressor on corrupted inputs equals the conditional mean of the once-corrupted target, which marginalises to the clean score.
3. **Additional corruption is the counterintuitive trick** — making the input *worse* creates the prediction target.
4. **Mask-conditioning** is essential — the model must know which pixels it is being asked to predict.
5. **Privacy by construction** — never seeing clean training data prevents exact memorisation, which has direct implications for medical/copyrighted data.
6. **Inpainting and compressed-sensing** are the proven corruption families; extending to general nonlinear corruption remains open.
7. **Foundation-model fine-tuning** combines domain adaptation and privacy in one workflow — practical for medical and astronomical imaging.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Standard diffusion loss / 표준 확산 손실
$$
\mathcal L_{\text{std}}(\theta) = \mathbb E_{\boldsymbol x_0, t, \boldsymbol\epsilon}\Big[\big\|\boldsymbol\epsilon - \boldsymbol\epsilon_\theta(\sqrt{\bar\alpha_t}\boldsymbol x_0 + \sqrt{1-\bar\alpha_t}\boldsymbol\epsilon, t)\big\|_2^2\Big]
$$
또는 $\boldsymbol x_0$-form: $\mathbb E\|\boldsymbol x_0 - \hat{\boldsymbol x}_{0,\theta}(\boldsymbol x_t, t)\|^2$.

### 4.2 Corruption operators / 손상 연산자
Inpainting: $\mathcal A_\phi(\boldsymbol x) = \boldsymbol M_\phi \odot \boldsymbol x$, $\boldsymbol M_\phi \in \{0,1\}^d$, $\boldsymbol M_\phi \sim \text{Bernoulli}(p_\phi)$ entrywise.
Compressed sensing: $\mathcal A_\phi(\boldsymbol x) = \boldsymbol G_\phi \boldsymbol x$, $\boldsymbol G_\phi \in \mathbb R^{m\times d}$ random Gaussian.

### 4.3 Ambient training loss / Ambient 학습 손실
$$
\mathcal L_{\text{Amb}}(\theta) = \mathbb E_{\tilde{\boldsymbol x}_0, \psi, t, \boldsymbol\epsilon}\Big[\big\|\boldsymbol M_\psi \odot \big(\tilde{\boldsymbol x}_0 - \hat{\boldsymbol x}_{0,\theta}(\boldsymbol x_t, t, \psi)\big)\big\|_2^2\Big]
$$
where $\boldsymbol x_t = \sqrt{\bar\alpha_t}\, \tilde{\mathcal A}_\psi(\tilde{\boldsymbol x}_0) + \sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon$.

### 4.4 Theorem 1 (loss equivalence, schematic) / 손실 등가
$$
\hat{\boldsymbol x}_{0,\theta}^*(\boldsymbol x_t, t, \psi) = \mathbb E\big[\tilde{\boldsymbol x}_0 \mid \boldsymbol x_t, t, \psi\big]
$$
With mask-conditioning over the corruption family, $\mathbb E[\tilde{\boldsymbol x}_0 \mid \cdot] = \mathbb E[\boldsymbol x_0 \mid \cdot]$ on observed regions; aggregation over $\phi$ yields the clean-distribution conditional mean (and hence score).

### 4.5 Score recovery / 스코어 복원
Tweedie's formula then yields the score
$$
\nabla_{\boldsymbol x_t} \log p_t(\boldsymbol x_t) = \frac{\sqrt{\bar\alpha_t}\,\hat{\boldsymbol x}_{0,\theta}^* - \boldsymbol x_t}{1 - \bar\alpha_t}.
$$

### 4.6 Worked example / 워크드 예제: 1차원 베르누이 마스킹

**설정**: 깨끗한 분포 $p(x_0) = 0.5\,\delta_{+1} + 0.5\,\delta_{-1}$ (이항). 단일 손상은 ${50\%}$ 확률로 $x_0$를 0으로 마스킹: $\tilde x_0 = m\cdot x_0$, $m \sim \text{Bernoulli}(0.5)$. 추가 손상: $\tilde{\tilde x}_0 = \tilde m \cdot \tilde x_0$, $\tilde m \sim \text{Bernoulli}(0.5)$ (독립).

**관측되는 4가지 case** (확률 표기):
| $m$ | $\tilde m$ | $\tilde x_0$ | $\tilde{\tilde x}_0$ | prob |
|----|-----------|-------------|---------------------|------|
| 0  | 0         | 0           | 0                   | 0.25 |
| 0  | 1         | 0           | 0                   | 0.25 |
| 1  | 0         | $x_0$       | 0                   | 0.25 |
| 1  | 1         | $x_0$       | $x_0$               | 0.25 |

**모델이 보는 것**: $\tilde{\tilde x}_0$는 0이거나 $x_0$ 자체. 모델은 $\tilde m=1$인 영역(즉 $\tilde{\tilde x}_0 = \tilde x_0$)에서만 손실을 받음. 손실:
$$
\mathcal L = \mathbb E[\tilde m \cdot (\tilde x_0 - f(\tilde{\tilde x}_0))^2]
$$
**최적해**: $f^*(\tilde{\tilde x}_0) = \mathbb E[\tilde x_0 \mid \tilde{\tilde x}_0, \tilde m=1] = \tilde{\tilde x}_0$ — 즉 항등함수. 그 항등함수에 mask-conditioning을 결합하면 잃어버린 영역(0)을 marginalise하여 $\mathbb E[x_0 \mid \tilde{\tilde x}_0=0] = 0$ ($\pm 1$ 같은 확률), $\mathbb E[x_0 \mid \tilde{\tilde x}_0 = +1] = +1$, $-1$도 마찬가지. **결과**: 모델은 깨끗한 $x_0$의 marginal posterior를 정확히 학습 — *깨끗한 표본 한 번 본 적 없이*.

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1950s -- Maximum-likelihood missing-data theory (EM algorithm
        roots; Dempster-Laird-Rubin formalised in 1977).
   |
1977 -- EM algorithm (Dempster et al.) — learn from incomplete data.
   |
2008 -- Compressed sensing (Candès & Tao; Donoho) — recover
        signals from few random measurements, but per-signal,
        not distribution-level.
   |
2018 -- AmbientGAN (Bora et al.) — first generative model from
        corrupted samples; GAN-based, unstable.
2018 -- Noise2Noise (Lehtinen et al.) — regression-only, two
        noisy copies, no clean ground truth.
2019 -- Noise2Void / Noise2Self — single-noisy-image regression.
   |
2020 -- DDPM (Ho et al.) — modern diffusion foundation.
2021 -- Score-SDE (Song et al.) — unifying SDE view.
   |
2022 -- Carlini et al. show diffusion memorises training data;
        Somepalli et al. confirm replication on Stable Diffusion.
   |
*** 2023 — Ambient Diffusion (this paper, NeurIPS) ***
        First diffusion model trained on only corrupted data
        with proven loss-equivalence; 90% missing pixels OK;
        privacy by construction.
   |
2023 -- Concurrent: SURE-Score (Aali et al.) — SURE-loss
        diffusion training. DPS (Chung et al., paper #28) —
        inference-time corruption inversion (orthogonal axis).
2024 -- Ambient-style training adopted for radio-astronomy and MRI.
```

```
1977 -- EM (incomplete-data MLE)
2018 -- AmbientGAN, Noise2Noise
2020 -- DDPM
2023 -- Ambient Diffusion: distribution learning from corruption
2023 -- DPS: corruption at inference (paper #28, complementary)
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| 관련 논문 / Related paper | 관계 / Relation |
|----------------------|----------------------------------------------------|
| **Lehtinen et al. 2018 (Noise2Noise)** — 회귀 버전. Ambient는 분포 학습으로 일반화. / Regression-only ancestor; Ambient generalises to distribution learning. |
| **Bora et al. 2018 (AmbientGAN)** — GAN 버전. Ambient는 diffusion으로 동일 동기 + 더 안정. / GAN-based predecessor; Ambient brings the same idea to diffusion with strong proofs and stability. |
| **Ho et al. 2020 (DDPM)** — 학습 골격. Ambient는 학습 손실만 변경, 샘플러는 그대로. / Backbone training loop unchanged; only the loss is mask-aware. |
| **Chung et al. 2023 (DPS, paper #28)** — 추론 시 손상 역전. Ambient는 학습 시 손상 처리. 두 축이 직교 + 결합 가능. / Orthogonal axis: DPS handles inference-time corruption, Ambient handles training-time corruption — composable. |
| **Carlini et al. 2023 (Diffusion memorisation)** — 메모리화 위협. Ambient의 privacy 동기. / The privacy threat that Ambient resolves by design. |
| **Zhu et al. 2023 (DiffPIR, paper #30)** — restoration 시 diffusion prior 사용. Ambient는 해당 prior를 corruption-only 데이터에서 어떻게 얻는지 답. / DiffPIR uses a clean-data diffusion prior at restoration; Ambient supplies a method to obtain such a prior when only corrupted data exist. |
| **Donoho-Johnstone 1994 (paper #01)** — denoising-as-shrinkage with Gaussian-noise model. Ambient는 noise 대신 missing-data 손상. / Wavelet shrinkage handles Gaussian noise; Ambient generalises the idea of "no clean target" to broader corruption. |
| **Aali et al. 2023 (SURE-Score)** — concurrent SURE-loss 변형. 같은 동기, 다른 수단. / Concurrent SURE-loss approach pursuing the same goal with a different technical lever. |

---

## 7. References / 참고문헌

- Daras, G., Shah, K., Dagan, Y., Gollakota, A., Dimakis, A. G., & Klivans, A. (2023). "Ambient Diffusion: Learning Clean Distributions from Corrupted Data." *NeurIPS 2023*. arXiv:2305.19256.
- Lehtinen, J., Munkberg, J., Hasselgren, J., Laine, S., Karras, T., Aittala, M., & Aila, T. (2018). "Noise2Noise: Learning Image Restoration without Clean Data." *ICML 2018*.
- Krull, A., Buchholz, T.-O., & Jug, F. (2019). "Noise2Void." *CVPR 2019*.
- Bora, A., Price, E., & Dimakis, A. G. (2018). "AmbientGAN: Generative Models from Lossy Measurements." *ICLR 2018*.
- Ho, J., Jain, A., & Abbeel, P. (2020). "Denoising Diffusion Probabilistic Models." *NeurIPS 2020*.
- Carlini, N., Hayes, J., Nasr, M., Jagielski, M., Sehwag, V., Tramèr, F., Balle, B., Ippolito, D., & Wallace, E. (2023). "Extracting Training Data from Diffusion Models." *USENIX Security 2023*.
- Somepalli, G., Singla, V., Goldblum, M., Geiping, J., & Goldstein, T. (2023). "Diffusion Art or Digital Forgery? Investigating Data Replication in Diffusion Models." *CVPR 2023*.
- Code: https://github.com/giannisdaras/ambient-diffusion

---

## Appendix A. Implementation Notes / 부록 A. 구현 노트

### 한국어
- **마스크 인코딩**: $\boldsymbol M_\phi$는 일반적으로 추가 입력 채널 (RGB → 4채널 입력) 또는 conditioning embedding으로 주입. 모델은 어떤 픽셀이 관측되었는지 알아야 함.
- **이중 마스크 샘플링**: 학습 시 매 step마다 $\phi$ (주어진 데이터의 이미 적용된 마스크)와 $\psi$ (새로 적용할 추가 마스크)를 sampling. 일반적 권장: $p_\psi$를 $p_\phi - 0.1$ 정도로.
- **손실 정규화**: $\boldsymbol M_\psi$ 영역에서만 손실 계산. 픽셀 수가 image마다 다르므로 픽셀당 평균으로 정규화.
- **Inference**: $\psi$를 항등(전체 1)로 고정하고 unconditional score를 사용. 학습된 score는 자동으로 깨끗한 분포의 score를 반환.
- **메모리/속도**: 표준 DDPM 학습과 거의 동일. 마스크 추가 채널과 sampling 약간의 오버헤드뿐.

### English
- **Mask encoding**: $\boldsymbol M_\phi$ is added as an extra input channel or as a conditioning embedding so the network knows which pixels are observed.
- **Double-mask sampling**: each training step draws $\phi$ (existing dataset mask) and $\psi$ (new additional mask), with empirical sweet spot $p_\psi = p_\phi - 0.1$.
- **Loss normalisation**: compute the squared error only on the $\psi$-removed region and average per pixel since the size of the region varies per sample.
- **Inference**: set $\psi$ to identity at sampling time; the learned score gives the clean distribution.
- **Compute**: nearly identical to standard DDPM training, with a small overhead for mask sampling.

---

## Appendix B. Comparison with related corruption-aware methods / 부록 B. 관련 손상 인지 기법과의 비교

### 한국어
| 방법 | 학습 데이터 | 출력 | 핵심 가정 |
|------|--------------|-------|-----------|
| Noise2Noise (2018) | (noisy, noisy) 쌍 | clean signal (regression) | zero-mean conditional noise |
| Noise2Void (2019) | single noisy | clean signal (regression) | spatial pixel independence |
| AmbientGAN (2018) | corrupted | clean distribution (GAN) | invertible-in-distribution corruption |
| **Ambient Diffusion** | corrupted | clean distribution (diffusion) | mask family with full pixel support |
| SURE-Score (2023) | corrupted | clean distribution (diffusion via SURE) | Gaussian noise model |

### English
The table contrasts the most closely related corruption-aware learning methods. Ambient Diffusion is unique in providing a *generative* (distribution-learning) framework with strong loss-equivalence guarantees for inpainting and compressed-sensing corruption families, with empirical strength on diffusion-quality benchmarks. The closest cousin is concurrent SURE-Score (Aali et al. 2023) which targets Gaussian noise rather than missing pixels.

### 한국어 보충
대표적으로 Noise2Noise는 *회귀 모델*만 다루며 분포 학습이 아니다. AmbientGAN은 동일 동기를 추구했지만 GAN 학습의 불안정성이 큰 한계였고, 이론적 보장도 약했다. Ambient Diffusion은 두 약점을 모두 해소: diffusion 학습의 안정성 + 손실 등가 정리. 결과로 90% missing이라는 극단적 손상 수준에서도 oracle 대비 작은 FID 격차로 분포를 복원한다.
