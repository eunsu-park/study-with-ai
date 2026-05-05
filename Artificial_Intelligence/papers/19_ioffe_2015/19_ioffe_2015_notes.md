---
title: "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"
authors: Sergey Ioffe, Christian Szegedy
year: 2015
journal: "Proceedings of the 32nd International Conference on Machine Learning (ICML 2015)"
doi: "arXiv:1502.03167"
topic: Artificial Intelligence / Deep Learning
tags: [batch-normalization, internal-covariate-shift, regularization, training-acceleration, normalization, deep-networks, inception]
status: completed
date_started: 2026-04-17
date_completed: 2026-04-17
---

# 19. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift / 배치 정규화: 내부 공변량 이동 감소를 통한 심층 네트워크 훈련 가속

---

## 1. Core Contribution / 핵심 기여

Batch Normalization(BN)은 딥 네트워크 훈련에서 각 레이어의 입력 분포가 이전 레이어의 파라미터 변화로 인해 지속적으로 변하는 현상 — **Internal Covariate Shift(ICS)** — 을 해결하기 위해 제안된 기법이다. 핵심 아이디어는 각 레이어의 입력을 미니배치 단위로 정규화(평균 0, 분산 1)한 뒤, 학습 가능한 affine 파라미터($\gamma$, $\beta$)로 네트워크의 표현력을 보존하는 것이다. 이 간단하지만 강력한 변환을 통해 (1) 훈련 속도를 14배 가속하고, (2) 훨씬 높은 learning rate를 안전하게 사용할 수 있으며, (3) 초기화에 대한 민감성을 줄이고, (4) regularization 효과로 Dropout의 필요성을 감소시키며, (5) sigmoid 같은 saturating nonlinearity의 사용을 가능하게 한다. BN-Inception 앙상블은 ImageNet에서 top-5 error 4.9%를 달성하며 당시 최고 기록을 경신했다.

Batch Normalization (BN) is a technique proposed to address **Internal Covariate Shift (ICS)** — the phenomenon where the distribution of each layer's inputs continuously changes during training due to parameter updates in preceding layers. The core idea is to normalize each layer's inputs over a mini-batch to zero mean and unit variance, then restore the network's representational power via learnable affine parameters ($\gamma$, $\beta$). This simple yet powerful transform (1) accelerates training by 14x, (2) enables safely using much higher learning rates, (3) reduces sensitivity to initialization, (4) provides regularization that reduces the need for Dropout, and (5) enables the use of saturating nonlinearities like sigmoid. A BN-Inception ensemble achieved 4.9% top-5 error on ImageNet, setting a new state of the art.

---

## 2. Reading Notes / 읽기 노트

### Section 1: Introduction / 도입

딥러닝은 vision, speech 등 많은 분야에서 최첨단 성능을 달성했으며, SGD와 그 변형들(momentum, Adagrad)이 핵심 최적화 도구이다. 그러나 SGD 훈련은 학습률, 초기화 등의 hyperparameter에 매우 민감하다. 특히 각 레이어의 입력이 이전 레이어의 파라미터에 의존하기 때문에, 작은 파라미터 변화가 깊은 네트워크에서 증폭되는 문제가 있다.

Deep learning has achieved state-of-the-art in many fields, with SGD and its variants (momentum, Adagrad) as core optimization tools. However, SGD training is highly sensitive to hyperparameters. Since each layer's inputs depend on preceding layer parameters, small parameter changes amplify through deep networks.

논문은 네트워크를 sub-network으로 분해하여 설명한다: $\ell = F_2(F_1(\mathbf{u}, \Theta_1), \Theta_2)$에서 $\mathbf{x} = F_1(\mathbf{u}, \Theta_1)$이 sub-network의 입력이라면, $\Theta_2$의 학습은 $\mathbf{x}$의 분포가 고정될 때 가장 효율적이다. 이는 covariate shift (Shimodaira, 2000)의 내부 확장이다.

The paper decomposes the network into sub-networks: in $\ell = F_2(F_1(\mathbf{u}, \Theta_1), \Theta_2)$, if $\mathbf{x} = F_1(\mathbf{u}, \Theta_1)$ is the sub-network's input, learning $\Theta_2$ is most efficient when the distribution of $\mathbf{x}$ is fixed. This is an internal extension of covariate shift (Shimodaira, 2000).

Sigmoid 활성화 함수 $g(x) = \frac{1}{1+\exp(-x)}$에서 $|x|$가 커지면 $g'(x) \to 0$이 되어 gradient vanishing이 발생한다. $\mathbf{x} = W\mathbf{u} + \mathbf{b}$에서 $W$와 하위 레이어의 파라미터가 변하면 $\mathbf{x}$의 많은 차원이 saturated regime으로 이동하여 수렴이 느려진다. ReLU, 신중한 초기화, 낮은 learning rate가 이 문제의 대처법이었지만, 근본적 해결책은 아니었다.

With sigmoid $g(x) = \frac{1}{1+\exp(-x)}$, $g'(x) \to 0$ as $|x|$ grows, causing gradient vanishing. As $W$ and lower-layer parameters change, many dimensions of $\mathbf{x} = W\mathbf{u} + \mathbf{b}$ drift into the saturated regime, slowing convergence. ReLU, careful initialization, and small learning rates were workarounds, not fundamental solutions.

---

### Section 2: Towards Reducing Internal Covariate Shift / 내부 공변량 이동 감소를 향하여

**Internal Covariate Shift의 정의**: 훈련 중 네트워크 파라미터의 변화로 인한 네트워크 activation 분포의 변화.

**Definition of ICS**: The change in the distribution of network activations due to changes in network parameters during training.

입력 whitening(평균 0, 단위 분산, 무상관)이 훈련을 가속한다는 것은 알려져 있었다 (LeCun et al., 1998b; Wiesler & Ney, 2011). 각 레이어의 입력에도 동일하게 적용하면 ICS를 줄일 수 있다.

Input whitening (zero mean, unit variance, decorrelated) is known to speed up training (LeCun et al., 1998b; Wiesler & Ney, 2011). Applying the same to each layer's inputs could reduce ICS.

**왜 단순 정규화가 실패하는가 — 핵심 예시**: activation $x = u + b$에 bias $b$를 학습하고, $\hat{x} = x - \text{E}[x]$로 정규화한다고 하자. Gradient descent가 $\text{E}[x]$의 $b$에 대한 의존성을 무시하면, $\Delta b \propto -\partial\ell/\partial\hat{x}$로 업데이트하지만, 정규화 후 $\hat{x} = u + (b + \Delta b) - \text{E}[u + (b + \Delta b)] = u + b - \text{E}[u + b]$가 되어 **출력은 변하지 않는다.** 결과적으로 $b$가 무한히 커지면서 loss는 그대로 — 최적화가 정규화의 효과를 상쇄하는 것이다. 정규화가 단순히 center와 scale만 하면 더 심각해진다.

**Why naive normalization fails — key example**: Consider activation $x = u + b$ with learnable bias $b$, normalized as $\hat{x} = x - \text{E}[x]$. If gradient descent ignores E$[x]$'s dependence on $b$, it updates $\Delta b \propto -\partial\ell/\partial\hat{x}$, but after normalization $\hat{x} = u + b - \text{E}[u + b]$ — **the output doesn't change.** So $b$ grows indefinitely while loss stays fixed — optimization cancels out normalization. This worsens when normalization also scales activations.

**해결 방향**: gradient descent가 정규화를 "인식"해야 한다. 즉, 정규화를 네트워크 아키텍처의 일부로 포함시켜, 어떤 파라미터 값에서든 항상 원하는 분포의 activation을 생성하도록 해야 한다.

**Solution direction**: Gradient descent must "account for" the normalization. The normalization must be incorporated into the network architecture itself, so that for any parameter values, the network always produces activations with the desired distribution.

---

### Section 3: Normalization via Mini-Batch Statistics / 미니배치 통계를 통한 정규화

전체 데이터셋에 대한 whitening은 비용이 크고(공분산 행렬 계산 필요), 어디서나 미분 가능하지도 않다. 논문은 두 가지 핵심 단순화를 도입한다:

Full-dataset whitening is expensive (requires covariance matrix computation) and not everywhere differentiable. The paper introduces two key simplifications:

**단순화 1**: Feature 간 decorrelation 대신, 각 scalar feature를 **독립적으로** 정규화한다.

**Simplification 1**: Instead of decorrelating features, normalize each scalar feature **independently**.

$$\hat{x}^{(k)} = \frac{x^{(k)} - \text{E}[x^{(k)}]}{\sqrt{\text{Var}[x^{(k)}]}}$$

단, 이렇게만 하면 sigmoid의 linear regime에 activation이 갇혀 레이어의 표현력이 제한된다.

However, this alone constrains activations to the linear regime of sigmoid, limiting the layer's representational power.

**단순화 2의 보완 — learnable parameters**: 각 activation $x^{(k)}$에 대해 학습 가능한 $\gamma^{(k)}$, $\beta^{(k)}$를 도입한다:

**Complement to simplification 2 — learnable parameters**: Introduce learnable $\gamma^{(k)}$, $\beta^{(k)}$ for each activation:

$$y^{(k)} = \gamma^{(k)} \hat{x}^{(k)} + \beta^{(k)}$$

핵심: $\gamma^{(k)} = \sqrt{\text{Var}[x^{(k)}]}$, $\beta^{(k)} = \text{E}[x^{(k)}]$로 설정하면 **원래 activation을 완벽히 복원**할 수 있다. 따라서 BN은 네트워크의 표현력을 절대 제한하지 않는다.

Key insight: Setting $\gamma^{(k)} = \sqrt{\text{Var}[x^{(k)}]}$, $\beta^{(k)} = \text{E}[x^{(k)}]$ **perfectly recovers the original activations**. Thus BN never limits the network's representational power.

**단순화 2**: 전체 데이터셋 대신 **미니배치 통계**를 사용한다. SGD의 미니배치가 gradient의 추정치를 제공하듯, 미니배치의 평균과 분산이 전체 데이터셋의 통계 추정치를 제공한다.

**Simplification 2**: Use **mini-batch statistics** instead of full-dataset statistics. Just as mini-batches provide gradient estimates in SGD, mini-batch mean and variance provide estimates of full-dataset statistics.

#### Algorithm 1: Batch Normalizing Transform

미니배치 $\mathcal{B} = \{x_{1...m}\}$에 대해:

For a mini-batch $\mathcal{B} = \{x_{1...m}\}$:

$$\mu_{\mathcal{B}} \leftarrow \frac{1}{m}\sum_{i=1}^{m} x_i$$

$$\sigma_{\mathcal{B}}^2 \leftarrow \frac{1}{m}\sum_{i=1}^{m}(x_i - \mu_{\mathcal{B}})^2$$

$$\hat{x}_i \leftarrow \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}}$$

$$y_i \leftarrow \gamma \hat{x}_i + \beta \equiv \text{BN}_{\gamma,\beta}(x_i)$$

$\epsilon$은 수치 안정성을 위한 작은 상수이다. BN은 training example을 독립적으로 처리하지 않는다 — $\text{BN}_{\gamma,\beta}(x)$는 해당 training example과 미니배치 내 다른 모든 example에 모두 의존한다.

$\epsilon$ is a small constant for numerical stability. BN does not process training examples independently — $\text{BN}_{\gamma,\beta}(x)$ depends on both the training example and all other examples in the mini-batch.

#### Backpropagation through BN / BN을 통한 역전파

BN은 미분 가능한 변환이므로 chain rule로 역전파가 가능하다:

BN is a differentiable transformation, so backpropagation via chain rule is straightforward:

$$\frac{\partial \ell}{\partial \hat{x}_i} = \frac{\partial \ell}{\partial y_i} \cdot \gamma$$

$$\frac{\partial \ell}{\partial \sigma_{\mathcal{B}}^2} = \sum_{i=1}^{m} \frac{\partial \ell}{\partial \hat{x}_i} \cdot (x_i - \mu_{\mathcal{B}}) \cdot \frac{-1}{2}(\sigma_{\mathcal{B}}^2 + \epsilon)^{-3/2}$$

$$\frac{\partial \ell}{\partial \mu_{\mathcal{B}}} = \sum_{i=1}^{m} \frac{\partial \ell}{\partial \hat{x}_i} \cdot \frac{-1}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}} + \frac{\partial \ell}{\partial \sigma_{\mathcal{B}}^2} \cdot \frac{\sum_{i=1}^{m} -2(x_i - \mu_{\mathcal{B}})}{m}$$

$$\frac{\partial \ell}{\partial x_i} = \frac{\partial \ell}{\partial \hat{x}_i} \cdot \frac{1}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}} + \frac{\partial \ell}{\partial \sigma_{\mathcal{B}}^2} \cdot \frac{2(x_i - \mu_{\mathcal{B}})}{m} + \frac{\partial \ell}{\partial \mu_{\mathcal{B}}} \cdot \frac{1}{m}$$

$$\frac{\partial \ell}{\partial \gamma} = \sum_{i=1}^{m} \frac{\partial \ell}{\partial y_i} \cdot \hat{x}_i, \quad \frac{\partial \ell}{\partial \beta} = \sum_{i=1}^{m} \frac{\partial \ell}{\partial y_i}$$

이 도함수들을 통해 정규화 통계($\mu_{\mathcal{B}}$, $\sigma_{\mathcal{B}}^2$)를 통한 gradient flow가 보장된다. 이것이 Section 2의 "naive normalization"과의 핵심 차이다 — gradient descent가 정규화를 인식한다.

These derivatives ensure gradient flow through the normalization statistics ($\mu_{\mathcal{B}}$, $\sigma_{\mathcal{B}}^2$). This is the key difference from the "naive normalization" in Section 2 — gradient descent accounts for the normalization.

---

### Section 3.1: Training and Inference with BN / BN으로 훈련과 추론

**훈련 시**: 미니배치 통계($\mu_{\mathcal{B}}$, $\sigma_{\mathcal{B}}^2$)를 사용한다.

**During training**: Mini-batch statistics ($\mu_{\mathcal{B}}$, $\sigma_{\mathcal{B}}^2$) are used.

**추론 시**: 출력이 입력에만 결정적으로 의존해야 하므로, 전체 훈련 데이터의 **population statistics**를 사용한다:

**During inference**: Output must depend deterministically on input only, so **population statistics** of the entire training set are used:

$$\hat{x} = \frac{x - \text{E}[x]}{\sqrt{\text{Var}[x] + \epsilon}}$$

Population statistics는 훈련 중 미니배치 통계의 **moving average**로 추정한다. 추론 시 BN은 단일 affine transformation으로 축소된다:

Population statistics are estimated via **moving averages** of mini-batch statistics during training. At inference, BN collapses to a single affine transformation:

$$y = \frac{\gamma}{\sqrt{\text{Var}[x] + \epsilon}} \cdot x + \left(\beta - \frac{\gamma \cdot \text{E}[x]}{\sqrt{\text{Var}[x] + \epsilon}}\right)$$

이는 추가 계산 비용 없이 기존 레이어에 합쳐질 수 있다. 분산 추정에는 unbiased estimate $\text{Var}[x] = \frac{m}{m-1} \cdot \text{E}_{\mathcal{B}}[\sigma_{\mathcal{B}}^2]$를 사용한다.

This can be folded into the existing layer with no additional computational cost. Variance estimation uses the unbiased estimate $\text{Var}[x] = \frac{m}{m-1} \cdot \text{E}_{\mathcal{B}}[\sigma_{\mathcal{B}}^2]$.

#### Algorithm 2: Training a Batch-Normalized Network

1. 네트워크의 activation 부분집합 $\{x^{(k)}\}_{k=1}^{K}$에 BN Transform을 삽입
2. $\Theta \cup \{\gamma^{(k)}, \beta^{(k)}\}_{k=1}^{K}$를 함께 최적화
3. 추론 네트워크: training 중 추정한 population statistics로 BN을 고정 affine transform으로 대체

1. Insert BN Transform for a subset of activations $\{x^{(k)}\}_{k=1}^{K}$
2. Optimize $\Theta \cup \{\gamma^{(k)}, \beta^{(k)}\}_{k=1}^{K}$ jointly
3. Inference network: replace BN with fixed affine transform using population statistics estimated during training

---

### Section 3.2: Batch-Normalized Convolutional Networks / BN 적용 CNN

CNN에서 BN은 affine transformation 직후, nonlinearity 직전에 적용된다: $z = g(\text{BN}(Wu))$. Bias $b$는 BN의 $\beta$에 흡수되므로 제거된다.

In CNNs, BN is applied immediately after the affine transformation, before the nonlinearity: $z = g(\text{BN}(Wu))$. Bias $b$ is removed since it is absorbed by BN's $\beta$.

**Convolutional 특성 유지**: 같은 feature map의 다른 위치들은 동일한 방식으로 정규화되어야 한다. 따라서 배치 크기 $m$, feature map 크기 $p \times q$이면, effective mini-batch 크기는 $m' = m \cdot p \cdot q$이 되며, $\gamma^{(k)}$, $\beta^{(k)}$는 **feature map 단위**로 학습된다 (activation 단위가 아님).

**Preserving convolutional property**: Different elements of the same feature map at different locations must be normalized the same way. For batch size $m$ and feature maps of size $p \times q$, the effective mini-batch size is $m' = m \cdot p \cdot q$, and $\gamma^{(k)}$, $\beta^{(k)}$ are learned **per feature map** (not per activation).

---

### Section 3.3: BN Enables Higher Learning Rates / BN이 높은 학습률을 가능하게 함

전통적으로 너무 높은 learning rate는 gradient 폭발/소실 또는 나쁜 local minima에 갇히는 문제를 일으켰다. BN은 이를 해결한다:

Traditionally, too-high learning rates caused gradient explosion/vanishing or getting stuck in poor local minima. BN addresses this:

**Scale invariance 성질**:

$$\text{BN}(Wu) = \text{BN}((aW)u)$$

$$\frac{\partial \text{BN}((aW)u)}{\partial u} = \frac{\partial \text{BN}(Wu)}{\partial u}$$

$$\frac{\partial \text{BN}((aW)u)}{\partial (aW)} = \frac{1}{a} \cdot \frac{\partial \text{BN}(Wu)}{\partial W}$$

**의미**: weight의 scale이 레이어 Jacobian이나 gradient에 영향을 주지 않는다. 큰 weight → 작은 gradient → 파라미터 성장 안정화. 이는 학습률이 파라미터의 scale에 의해 증폭되지 않으므로, 높은 학습률을 안전하게 사용할 수 있게 한다.

**Implication**: Weight scale does not affect the layer Jacobian or gradients. Larger weights → smaller gradients → stabilized parameter growth. Since the learning rate is not amplified by the parameter scale, higher learning rates can be used safely.

또한 Jacobian의 singular value가 1에 가까워져 gradient flow가 개선된다고 추측한다 (완전한 증명은 아니지만, 정규화된 입력이 Gaussian이고 무상관이라면 $JJ^T = I$가 됨).

The paper also conjectures that Jacobian singular values approach 1, improving gradient flow (not a complete proof, but if normalized inputs are Gaussian and uncorrelated, $JJ^T = I$).

---

### Section 3.4: BN Regularizes the Model / BN의 정규화 효과

BN에서 각 training example은 미니배치 내 다른 example들과 함께 정규화되므로, 네트워크가 더 이상 하나의 training example에 대해 결정적(deterministic)인 값을 생성하지 않는다. 이 확률적(stochastic) 효과가 **regularization** 역할을 하여, 실험에서 Dropout을 제거하거나 약화해도 과적합이 증가하지 않았다.

In BN, each training example is normalized in conjunction with other examples in the mini-batch, so the network no longer produces deterministic values for a given training example. This stochastic effect acts as **regularization**, and experiments showed that Dropout can be removed or reduced without increasing overfitting.

---

### Section 4: Experiments / 실험

#### 4.1 MNIST에서의 Activation 분포 (Figure 1)

3개의 fully-connected hidden layer (각 100개 activation), sigmoid nonlinearity, SGD로 50000 스텝 훈련.

3 fully-connected hidden layers (100 activations each), sigmoid nonlinearity, trained with SGD for 50000 steps.

**결과**: BN 없는 네트워크의 마지막 hidden layer activation 분포는 훈련이 진행되면서 평균과 분산 모두 크게 변동한다. BN 네트워크의 activation 분포는 훈련 내내 훨씬 안정적이며, 더 빠르게 더 높은 test accuracy를 달성한다 (Figure 1a).

**Result**: Without BN, the last hidden layer's activation distribution shifts significantly in both mean and variance during training. BN's activation distributions remain much more stable throughout, reaching higher test accuracy faster (Figure 1a).

#### 4.2 ImageNet Classification

Inception (GoogLeNet 변형, 13.6M 파라미터)을 기반으로 실험. SGD with momentum, 배치 크기 32, distributed training.

Based on Inception (GoogLeNet variant, 13.6M parameters). SGD with momentum, batch size 32, distributed training.

**BN-Inception 변형들**:

| 모델 / Model | 설명 / Description | 72.2% 도달 스텝 | 최대 정확도 |
|---|---|---|---|
| **Inception** | 기본 모델, lr=0.0015 | $31.0 \times 10^6$ | 72.2% |
| **BN-Baseline** | BN만 추가 | $13.3 \times 10^6$ (2.3x 빠름) | 72.7% |
| **BN-x5** | BN + lr 5x 증가(0.0075) + 수정사항 | $2.1 \times 10^6$ (14x 빠름) | 73.0% |
| **BN-x30** | BN + lr 30x 증가(0.045) | $2.7 \times 10^6$ | **74.8%** |
| **BN-x5-Sigmoid** | BN-x5 + sigmoid (ReLU 대신) | — | 69.8% |

**BN-x5의 수정 사항** (Section 4.2.1):
1. Learning rate를 5배 증가
2. Dropout 제거 (BN이 regularization 역할)
3. $L_2$ weight regularization 5배 감소
4. Learning rate decay 6배 가속
5. Local Response Normalization 제거
6. Within-shard shuffling 활성화 (~1% 향상)
7. Photometric distortion 감소 (빠른 훈련 → 더 "실제" 이미지에 집중)

**핵심 발견들**:
- **BN만 추가해도** 훈련 스텝이 절반 이하로 감소 (BN-Baseline: 2.3배 빠름)
- **수정 사항 결합 시** 14배 빠른 수렴 (BN-x5)
- Learning rate를 30배로 올리면 초기 수렴은 느리지만 **최종 정확도가 더 높음** (BN-x30: 74.8%)
- **Sigmoid로도 훈련 가능**: BN 없이는 sigmoid Inception이 chance level (0.1%)에 머물지만, BN-x5-Sigmoid는 69.8% 달성

**Key findings**:
- Adding BN alone cuts training steps by more than half (BN-Baseline: 2.3x faster)
- Combined modifications yield 14x faster convergence (BN-x5)
- 30x learning rate is initially slower but reaches higher final accuracy (BN-x30: 74.8%)
- **Sigmoid becomes trainable**: Without BN, sigmoid Inception stays at chance level (0.1%); BN-x5-Sigmoid reaches 69.8%

#### 4.2.3 Ensemble Classification

6개의 BN-x30 기반 모델 앙상블 (각 모델 약 $6 \times 10^6$ 스텝 훈련, 다양한 수정 적용):

Ensemble of 6 BN-x30-based models (each trained ~$6 \times 10^6$ steps with various modifications):

| 모델 / Model | Top-5 error |
|---|---|
| GoogLeNet ensemble (2014 우승) | 6.67% |
| Deep Image ensemble | 5.98% |
| **BN-Inception single crop** | 5.82% |
| **BN-Inception multicrop** | 5.82% |
| **BN-Inception ensemble** | **4.9%** * |

*4.82% test error (ILSVRC server 기준), 인간 평가자 정확도(~5.1%)를 초과.

*4.82% test error (per ILSVRC server), exceeding estimated accuracy of human raters (~5.1%).

---

## 3. Key Takeaways / 핵심 시사점

1. **Internal Covariate Shift는 실질적 훈련 장애물** — 이전 레이어 파라미터 변화가 후속 레이어 입력 분포를 지속적으로 변화시켜, 각 레이어가 끊임없이 새 분포에 적응해야 한다. BN은 이 문제를 각 레이어 입력의 분포를 안정화시켜 해결한다.
   ICS is a real training obstacle — preceding layer parameter changes continuously shift subsequent layer input distributions. BN solves this by stabilizing input distributions at each layer.

2. **단순 정규화는 실패한다 — "gradient가 정규화를 인식해야"** — 정규화를 네트워크 외부에서 수행하면 gradient descent가 이를 무시하고 파라미터를 발산시킨다 (bias $b$ 발산 예시). BN을 네트워크 아키텍처 내부에 포함시킴으로써 이 문제를 해결한다.
   Naive normalization fails — gradient descent ignores external normalization and diverges parameters. Embedding BN within the architecture ensures gradients account for normalization.

3. **$\gamma$, $\beta$가 표현력을 보존하는 핵심** — 정규화만으로는 레이어의 표현력을 제한한다 (sigmoid의 linear regime). $\gamma$, $\beta$를 통해 identity transform을 포함한 모든 affine transformation을 학습할 수 있어, BN이 네트워크의 능력을 절대 제한하지 않는다.
   $\gamma$, $\beta$ are crucial for preserving representational power — normalization alone constrains layers. Learnable affine parameters ensure BN never limits the network's capacity.

4. **훈련과 추론의 분리는 필수적 설계** — 훈련 시 미니배치 통계, 추론 시 population statistics (moving average 추정). 이 분리가 추론 시 결정적(deterministic) 출력을 보장하며, BN을 단일 affine transform으로 축소하여 추가 비용이 없다.
   The training/inference split is essential — mini-batch stats during training, population stats (moving average estimates) during inference. This ensures deterministic inference and collapses BN to a single cost-free affine transform.

5. **Scale invariance가 높은 learning rate를 가능하게** — $\text{BN}(Wu) = \text{BN}((aW)u)$이므로 weight의 scale이 출력에 영향을 주지 않으며, 큰 weight는 오히려 작은 gradient를 받아 파라미터 성장이 자동 안정화된다. 이는 learning rate를 5~30배 높여도 안전하게 훈련할 수 있는 이론적 근거이다.
   $\text{BN}(Wu) = \text{BN}((aW)u)$ means weight scale doesn't affect output, and larger weights receive smaller gradients, auto-stabilizing parameter growth. This theoretically justifies safely using 5-30x higher learning rates.

6. **BN은 암시적 정규화기(regularizer)** — 미니배치 내 다른 샘플들에 의존하는 stochastic normalization이 Dropout과 유사한 정규화 효과를 제공한다. 실험에서 Dropout을 완전히 제거해도 과적합이 증가하지 않았다.
   BN acts as an implicit regularizer — stochastic normalization depending on other mini-batch samples provides Dropout-like regularization. Experiments showed removing Dropout entirely didn't increase overfitting.

7. **Saturating nonlinearity를 부활시킴** — BN 이전에는 sigmoid/tanh가 깊은 네트워크에서 사실상 사용 불가능했다. BN이 activation을 non-saturated regime에 유지시켜, sigmoid로도 69.8% ImageNet accuracy를 달성했다. (BN 없이는 0.1%.)
   BN revived saturating nonlinearities — before BN, sigmoid/tanh were practically unusable in deep networks. BN keeps activations in the non-saturated regime, achieving 69.8% on ImageNet with sigmoid (vs. 0.1% without BN).

8. **CNN에서 BN은 feature map 단위로 적용** — 각 activation이 아니라 채널(feature map)별로 하나의 $\gamma$, $\beta$를 학습하며, effective batch size는 $m \cdot p \cdot q$로 spatial dimension을 포함한다. 이는 convolutional 특성(같은 feature map의 모든 위치가 동일 변환)을 보존한다.
   In CNNs, BN is applied per feature map — one $\gamma$, $\beta$ per channel, with effective batch size $m \cdot p \cdot q$ including spatial dimensions. This preserves the convolutional property (all locations in the same feature map share the same transform).

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Batch Normalizing Transform (Algorithm 1)

미니배치 $\mathcal{B} = \{x_1, ..., x_m\}$, 학습 파라미터 $\gamma$, $\beta$:

Mini-batch $\mathcal{B} = \{x_1, ..., x_m\}$, learnable parameters $\gamma$, $\beta$:

| 단계 / Step | 수식 / Equation | 설명 / Description |
|---|---|---|
| 미니배치 평균 | $\mu_{\mathcal{B}} = \frac{1}{m}\sum_{i=1}^{m} x_i$ | 배치 내 평균 계산 / Compute batch mean |
| 미니배치 분산 | $\sigma_{\mathcal{B}}^2 = \frac{1}{m}\sum_{i=1}^{m}(x_i - \mu_{\mathcal{B}})^2$ | 배치 내 분산 계산 / Compute batch variance |
| 정규화 | $\hat{x}_i = \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}}$ | 평균 0, 분산 1로 변환 / Transform to zero mean, unit variance |
| Scale & shift | $y_i = \gamma \hat{x}_i + \beta$ | 표현력 복원 / Restore representational power |

### 4.2 Backpropagation Gradients

$$\frac{\partial \ell}{\partial \gamma} = \sum_{i=1}^{m} \frac{\partial \ell}{\partial y_i} \cdot \hat{x}_i, \quad \frac{\partial \ell}{\partial \beta} = \sum_{i=1}^{m} \frac{\partial \ell}{\partial y_i}$$

$$\frac{\partial \ell}{\partial x_i} = \frac{\partial \ell}{\partial \hat{x}_i} \cdot \frac{1}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}} + \frac{\partial \ell}{\partial \sigma_{\mathcal{B}}^2} \cdot \frac{2(x_i - \mu_{\mathcal{B}})}{m} + \frac{\partial \ell}{\partial \mu_{\mathcal{B}}} \cdot \frac{1}{m}$$

핵심: $\frac{\partial \ell}{\partial x_i}$는 $\mu_{\mathcal{B}}$와 $\sigma_{\mathcal{B}}^2$를 통한 gradient를 포함하므로, gradient descent가 정규화를 완전히 인식한다.

Key: $\frac{\partial \ell}{\partial x_i}$ includes gradients through $\mu_{\mathcal{B}}$ and $\sigma_{\mathcal{B}}^2$, so gradient descent fully accounts for normalization.

### 4.3 Inference-Time Transformation

$$y = \underbrace{\frac{\gamma}{\sqrt{\text{Var}[x] + \epsilon}}}_{W_{\text{BN}}} \cdot x + \underbrace{\left(\beta - \frac{\gamma \cdot \text{E}[x]}{\sqrt{\text{Var}[x] + \epsilon}}\right)}_{b_{\text{BN}}}$$

단일 affine transformation으로 축소 — 추론 시 추가 계산 비용 없음.

Collapses to a single affine transformation — no additional inference cost.

### 4.4 Scale Invariance Property

$$\text{BN}(Wu) = \text{BN}((aW)u) \quad \forall a > 0$$

$$\frac{\partial \text{BN}((aW)u)}{\partial u} = \frac{\partial \text{BN}(Wu)}{\partial u}, \quad \frac{\partial \text{BN}((aW)u)}{\partial (aW)} = \frac{1}{a} \cdot \frac{\partial \text{BN}(Wu)}{\partial W}$$

큰 weight ($a > 1$) → gradient가 $1/a$로 축소 → 자동 안정화.

Larger weights ($a > 1$) → gradients scaled by $1/a$ → automatic stabilization.

### 4.5 CNN에서의 Effective Batch Size / Effective Batch Size in CNNs

배치 크기 $m$, feature map 크기 $p \times q$:

Batch size $m$, feature map size $p \times q$:

$$m' = |\mathcal{B}| = m \cdot p \cdot q$$

$\gamma^{(k)}$, $\beta^{(k)}$는 feature map당 하나 (per-activation이 아님).

One $\gamma^{(k)}$, $\beta^{(k)}$ per feature map (not per activation).

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1986  Rumelhart et al. — Backpropagation (gradient 기반 학습의 기초)
 |
1998  LeCun et al. — Efficient Backprop (입력 정규화가 수렴을 가속)
 |
2010  Glorot & Bengio — Xavier initialization (깊은 네트워크의 초기화 전략)
 |     Nair & Hinton — ReLU (saturating nonlinearity 문제의 우회)
 |
2011  Duchi et al. — AdaGrad (adaptive learning rate)
 |
2012  Krizhevsky et al. — AlexNet (깊은 CNN의 부상)
 |
2013  Sutskever et al. — Momentum SGD + 초기화 (Paper #13)
 |
2014  Szegedy et al. — GoogLeNet/Inception (Paper #18)
 |     Srivastava et al. — Dropout (정규화 기법)
 |     Kingma & Ba — Adam optimizer (Paper #18)
 |
2015  ★ Ioffe & Szegedy — Batch Normalization ← 이 논문
 |     He et al. — ResNet (BN을 핵심 구성 요소로 사용)
 |
2016  Ba et al. — Layer Normalization (RNN/Transformer용)
 |     Ulyanov et al. — Instance Normalization (style transfer용)
 |
2018  Wu & He — Group Normalization (작은 batch에서의 대안)
 |     Santurkar et al. — "BN은 ICS가 아니라 loss landscape smoothing" 논쟁
 |
2020+ Normalization은 거의 모든 딥러닝 아키텍처의 표준 구성 요소
      Normalization becomes a standard component in nearly all architectures
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| #8 Rumelhart et al. (1986) — Backpropagation | BN의 역전파 도함수는 chain rule에 기반. BN이 gradient flow를 개선하여 backpropagation을 더 효과적으로 만듦 / BN's backprop derivatives are based on chain rule. BN improves gradient flow, making backpropagation more effective | 높음 / High |
| #13 Sutskever et al. (2013) — Momentum SGD | BN 실험의 기본 optimizer. BN이 momentum SGD의 learning rate 제약을 대폭 완화 / Base optimizer for BN experiments. BN greatly relaxes momentum SGD's learning rate constraints | 높음 / High |
| #18 Szegedy et al. (2014) — Inception/GoogLeNet | BN이 적용된 기본 아키텍처. 같은 저자(Szegedy)가 참여. BN-Inception이 원래 Inception을 14배 빠르게 수렴 / Base architecture for BN application. Same author (Szegedy). BN-Inception converges 14x faster | 매우 높음 / Very High |
| Srivastava et al. (2014) — Dropout | BN의 stochastic normalization이 Dropout과 유사한 정규화 효과를 제공하여 Dropout을 대체하거나 약화 가능 / BN's stochastic normalization provides Dropout-like regularization, replacing or reducing Dropout | 높음 / High |
| LeCun et al. (1998b) — Efficient Backprop | 입력 정규화가 수렴을 가속한다는 선행 연구. BN은 이를 모든 레이어에 확장 적용 / Predecessor showing input normalization accelerates convergence. BN extends this to all layers | 높음 / High |
| He et al. (2015) — ResNet | BN을 핵심 구성 요소로 채택한 후속 아키텍처. BN 없이는 100+ 레이어 네트워크 훈련이 사실상 불가능 / Adopted BN as a core component. Training 100+ layer networks would be virtually impossible without BN | 매우 높음 / Very High |
| Ba et al. (2016) — Layer Normalization | BN의 한계(작은 batch, RNN)를 극복하기 위한 후속 연구. 단일 sample의 모든 feature에 대해 정규화 / Follow-up addressing BN's limitations (small batches, RNNs). Normalizes across all features of a single sample | 높음 / High |

---

## 7. References / 참고문헌

- Ioffe, S. & Szegedy, C., "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift", ICML, 2015. [arXiv:1502.03167]
- LeCun, Y., Bottou, L., Orr, G. & Muller, K., "Efficient Backprop", in Neural Networks: Tricks of the Trade, Springer, 1998b.
- Shimodaira, H., "Improving predictive inference under covariate shift by weighting the log-likelihood function", JSPI, 2000.
- Szegedy, C. et al., "Going deeper with convolutions", CoRR, abs/1409.4842, 2014.
- Srivastava, N. et al., "Dropout: A simple way to prevent neural networks from overfitting", JMLR, 2014.
- He, K. et al., "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification", arXiv, 2015.
- Sutskever, I. et al., "On the importance of initialization and momentum in deep learning", ICML, 2013.
- Glorot, X. & Bengio, Y., "Understanding the difficulty of training deep feedforward neural networks", AISTATS, 2010.
- Ba, J., Kiros, R. & Hinton, G., "Layer Normalization", arXiv:1607.06450, 2016.
- Wu, Y. & He, K., "Group Normalization", ECCV, 2018.
- Santurkar, S. et al., "How Does Batch Normalization Help Optimization?", NeurIPS, 2018.
- Nair, V. & Hinton, G., "Rectified Linear Units Improve Restricted Boltzmann Machines", ICML, 2010.
- Duchi, J. et al., "Adaptive subgradient methods for online learning and stochastic optimization", JMLR, 2011.
