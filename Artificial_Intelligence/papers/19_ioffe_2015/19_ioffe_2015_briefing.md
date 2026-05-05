---
title: "Pre-Reading Briefing: Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"
paper_id: "19_ioffe_2015"
topic: Artificial Intelligence
date: 2026-04-17
type: briefing
---

# Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Ioffe, S. & Szegedy, C. (2015). "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift." *Proceedings of the 32nd International Conference on Machine Learning (ICML 2015)*.
**Author(s)**: Sergey Ioffe, Christian Szegedy
**Year**: 2015

---

## 1. 핵심 기여 / Core Contribution

이 논문은 딥 네트워크 훈련에서 가장 널리 사용되는 기법 중 하나인 **Batch Normalization (BN)**을 제안합니다. 핵심 아이디어는 간단합니다: 네트워크의 각 레이어 입력을 미니배치 단위로 정규화(평균 0, 분산 1)한 뒤, 학습 가능한 scale($\gamma$)과 shift($\beta$) 파라미터로 복원하는 것입니다. 이를 통해 **Internal Covariate Shift** — 훈련 중 이전 레이어의 파라미터 변화로 인해 각 레이어 입력 분포가 계속 바뀌는 현상 — 을 줄여 훈련을 극적으로 가속합니다. BN을 적용한 Inception 모델은 원래 모델 대비 **14배 적은 훈련 스텝**으로 동일 정확도에 도달하고, 앙상블로 ImageNet top-5 error **4.9%**를 달성하여 당시 최고 기록을 경신했습니다.

This paper introduces **Batch Normalization (BN)**, one of the most widely adopted techniques in deep network training. The core idea is simple: normalize each layer's inputs over a mini-batch to have zero mean and unit variance, then restore representational power with learnable scale ($\gamma$) and shift ($\beta$) parameters. This reduces **Internal Covariate Shift** — the phenomenon where the distribution of each layer's inputs changes during training as the parameters of preceding layers change — dramatically accelerating training. A BN-augmented Inception model matches the original's accuracy in **14x fewer training steps**, and an ensemble achieves **4.9% top-5 error** on ImageNet, setting a new state of the art at the time.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

2014-2015년은 딥러닝의 황금기였습니다. GoogLeNet/Inception (Szegedy et al., 2014)과 VGGNet이 ImageNet에서 경쟁하고 있었고, 네트워크는 점점 깊어지고 있었습니다. 그러나 깊은 네트워크를 훈련시키는 것은 여전히 어려웠습니다:

2014-2015 was the golden age of deep learning. GoogLeNet/Inception (Szegedy et al., 2014) and VGGNet were competing on ImageNet, and networks were getting deeper. But training deep networks remained challenging:

- **느린 훈련 / Slow training**: 낮은 learning rate를 사용해야 했고, 수렴에 수백만 스텝이 필요했습니다.
  Low learning rates were required, and convergence took millions of steps.
- **초기화 민감성 / Initialization sensitivity**: Xavier/He initialization이 필요했고, 잘못된 초기화는 훈련 실패로 이어졌습니다.
  Xavier/He initialization was needed; bad initialization caused training failure.
- **Saturating nonlinearities 문제**: Sigmoid/tanh 활성화 함수는 깊은 네트워크에서 gradient vanishing을 야기했습니다.
  Sigmoid/tanh activations caused gradient vanishing in deep networks.
- **Dropout 의존성**: 과적합 방지를 위해 Dropout (Srivastava et al., 2014)이 거의 필수적이었습니다.
  Dropout was nearly mandatory for preventing overfitting.

### 타임라인 / Timeline

```
1998  LeCun et al. — input normalization speeds convergence (Efficient Backprop)
 |
2010  Glorot & Bengio — Xavier initialization for deep feedforward networks
 |
2012  Raiko et al. — deep learning with linear transformations in perceptrons
 |
2013  Sutskever et al. — momentum SGD, careful initialization (Paper #13)
 |     Gülçehre & Bengio — knowledge matters for optimization
 |
2014  Szegedy et al. — GoogLeNet/Inception (Paper #18)
 |     Srivastava et al. — Dropout regularization
 |     Wiesler et al. — mean-normalized SGD
 |
2015  ★ Ioffe & Szegedy — Batch Normalization ← 이 논문
 |     He et al. — ResNet (BN을 핵심 구성 요소로 사용)
 |
2016  Ba et al. — Layer Normalization (BN의 한계를 극복하려는 후속 연구)
```

---

## 3. 필요한 배경 지식 / Prerequisites

### 필수 개념 / Essential Concepts

| 개념 / Concept | 설명 / Explanation |
|---|---|
| **SGD와 mini-batch** / SGD and mini-batches | 전체 데이터셋 대신 작은 배치로 gradient를 근사하여 파라미터를 업데이트하는 방식. The method of approximating gradients with small batches instead of the full dataset. (Paper #13) |
| **Backpropagation** / 역전파 | Chain rule을 이용해 손실 함수의 gradient를 네트워크의 각 파라미터에 대해 계산하는 알고리즘. Algorithm for computing loss gradients w.r.t. each parameter using the chain rule. (Paper #8) |
| **Covariate Shift** | 훈련 데이터와 테스트 데이터의 입력 분포가 다른 현상 (Shimodaira, 2000). 이 논문은 이를 네트워크 내부 레이어에 확장 적용. When training and test input distributions differ. This paper extends the idea to internal layers. |
| **Inception/GoogLeNet 아키텍처** | 여러 크기의 convolution을 병렬로 적용하는 모듈 기반 CNN (Paper #18). A modular CNN applying convolutions of multiple sizes in parallel. |
| **Gradient flow** / 기울기 흐름 | 깊은 네트워크에서 gradient가 여러 레이어를 통과하며 소실(vanishing)되거나 폭발(exploding)하는 현상. How gradients vanish or explode as they pass through many layers in deep networks. |

### 수학적 배경 / Mathematical Background

- **평균과 분산** / Mean and variance: $\mu = \frac{1}{m}\sum x_i$, $\sigma^2 = \frac{1}{m}\sum(x_i - \mu)^2$
- **정규화** / Normalization: $\hat{x} = \frac{x - \mu}{\sigma}$ (z-score 변환)
- **Chain rule**: $\frac{\partial \ell}{\partial x} = \frac{\partial \ell}{\partial y} \cdot \frac{\partial y}{\partial x}$ (backpropagation의 기초)
- **Affine transformation**: $y = \gamma x + \beta$ (선형 변환)

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **Internal Covariate Shift** | 훈련 중 이전 레이어 파라미터 변화로 인해 현재 레이어의 입력 분포가 바뀌는 현상. Change in the distribution of layer inputs during training due to parameter updates in preceding layers. |
| **Batch Normalizing Transform** | 미니배치 통계량(평균, 분산)을 이용해 각 activation을 정규화하는 변환. A transform that normalizes each activation using mini-batch statistics (mean, variance). |
| **Learnable parameters ($\gamma$, $\beta$)** | BN 후 정규화된 값을 scale하고 shift하는 학습 가능한 파라미터. 네트워크의 표현력을 보존. Learnable scale and shift parameters that preserve the network's representational power after normalization. |
| **Population statistics** | 추론(inference) 시 미니배치 통계 대신 사용하는 전체 훈련 데이터의 평균과 분산 (moving average로 추정). Mean and variance of the entire training set, estimated via moving averages, used at inference time instead of mini-batch statistics. |
| **Whitening** | 데이터를 평균 0, 단위 분산, 무상관으로 변환하는 과정. BN은 이의 간소화된 버전. Transforming data to zero mean, unit variance, and decorrelated. BN is a simplified version. |
| **Saturating nonlinearity** | Sigmoid, tanh 같이 큰 입력에서 gradient가 0에 가까워지는 활성화 함수. Activation functions like sigmoid/tanh where gradients approach zero for large inputs. |
| **BN-Inception** | Batch Normalization을 적용한 Inception 모델의 변형. A variant of the Inception model augmented with Batch Normalization. |
| **$\epsilon$ (epsilon)** | 수치 안정성을 위해 분산에 더하는 작은 상수 ($\sqrt{\sigma^2 + \epsilon}$). A small constant added to variance for numerical stability. |
| **Unbiased variance estimate** | $\text{Var}[x] = \frac{m}{m-1} \cdot \mathbb{E}_{\mathcal{B}}[\sigma_{\mathcal{B}}^2]$, 추론 시 사용하는 편향 보정된 분산 추정치. Bias-corrected variance estimate using Bessel's correction, used for inference. |
| **BN-x5, BN-x30** | Learning rate를 각각 5배, 30배 높인 BN-Inception 변형. BN-Inception variants with 5x and 30x higher learning rates. |

---

## 5. 수식 미리보기 / Equations Preview

### 수식 1: Batch Normalizing Transform (Algorithm 1) / 배치 정규화 변환

$$\mu_{\mathcal{B}} \leftarrow \frac{1}{m}\sum_{i=1}^{m} x_i \quad \text{(mini-batch mean / 미니배치 평균)}$$

$$\sigma_{\mathcal{B}}^2 \leftarrow \frac{1}{m}\sum_{i=1}^{m}(x_i - \mu_{\mathcal{B}})^2 \quad \text{(mini-batch variance / 미니배치 분산)}$$

$$\hat{x}_i \leftarrow \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}} \quad \text{(normalize / 정규화)}$$

$$y_i \leftarrow \gamma \hat{x}_i + \beta \equiv \text{BN}_{\gamma,\beta}(x_i) \quad \text{(scale and shift / 스케일 및 시프트)}$$

핵심 포인트: $\gamma$와 $\beta$가 없으면 정규화가 레이어의 표현력을 제한합니다. $\gamma = \sqrt{\text{Var}[x]}$, $\beta = \text{E}[x]$로 설정하면 원래 activation을 복원할 수 있습니다.

Key point: Without $\gamma$ and $\beta$, normalization would constrain the layer's representational power. Setting $\gamma = \sqrt{\text{Var}[x]}$, $\beta = \text{E}[x]$ recovers the original activations.

### 수식 2: Backpropagation through BN / BN을 통한 역전파

$$\frac{\partial \ell}{\partial \hat{x}_i} = \frac{\partial \ell}{\partial y_i} \cdot \gamma$$

$$\frac{\partial \ell}{\partial \sigma_{\mathcal{B}}^2} = \sum_{i=1}^{m} \frac{\partial \ell}{\partial \hat{x}_i} \cdot (x_i - \mu_{\mathcal{B}}) \cdot \frac{-1}{2}(\sigma_{\mathcal{B}}^2 + \epsilon)^{-3/2}$$

$$\frac{\partial \ell}{\partial \mu_{\mathcal{B}}} = \sum_{i=1}^{m} \frac{\partial \ell}{\partial \hat{x}_i} \cdot \frac{-1}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}} + \frac{\partial \ell}{\partial \sigma_{\mathcal{B}}^2} \cdot \frac{-2\sum_{i=1}^{m}(x_i - \mu_{\mathcal{B}})}{m}$$

$$\frac{\partial \ell}{\partial x_i} = \frac{\partial \ell}{\partial \hat{x}_i} \cdot \frac{1}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}} + \frac{\partial \ell}{\partial \sigma_{\mathcal{B}}^2} \cdot \frac{2(x_i - \mu_{\mathcal{B}})}{m} + \frac{\partial \ell}{\partial \mu_{\mathcal{B}}} \cdot \frac{1}{m}$$

$$\frac{\partial \ell}{\partial \gamma} = \sum_{i=1}^{m} \frac{\partial \ell}{\partial y_i} \cdot \hat{x}_i, \quad \frac{\partial \ell}{\partial \beta} = \sum_{i=1}^{m} \frac{\partial \ell}{\partial y_i}$$

BN이 미분 가능한 변환이므로 표준 backpropagation으로 $\gamma$, $\beta$를 학습할 수 있습니다.

Since BN is a differentiable transform, standard backpropagation can learn $\gamma$ and $\beta$.

### 수식 3: Inference-time normalization / 추론 시 정규화

$$y = \frac{\gamma}{\sqrt{\text{Var}[x] + \epsilon}} \cdot x + \left(\beta - \frac{\gamma \text{E}[x]}{\sqrt{\text{Var}[x] + \epsilon}}\right)$$

추론 시에는 미니배치 통계 대신 훈련 중 moving average로 추정한 population statistics ($\text{E}[x]$, $\text{Var}[x]$)를 사용합니다. 이는 단일 affine transformation으로 축소되어 추가 계산 비용이 없습니다.

At inference time, population statistics estimated via moving averages during training replace mini-batch statistics. This collapses to a single affine transformation with no additional computational cost.

### 수식 4: Scale invariance property / 스케일 불변 성질

$$\text{BN}(Wu) = \text{BN}((aW)u)$$

$$\frac{\partial \text{BN}((aW)u)}{\partial u} = \frac{\partial \text{BN}(Wu)}{\partial u}, \quad \frac{\partial \text{BN}((aW)u)}{\partial (aW)} = \frac{1}{a} \cdot \frac{\partial \text{BN}(Wu)}{\partial W}$$

Weight의 scale이 BN 출력에 영향을 주지 않으며, 큰 weight는 오히려 작은 gradient를 받아 파라미터 성장이 안정화됩니다.

The scale of weights does not affect BN output, and larger weights receive smaller gradients, stabilizing parameter growth.

---

## 6. 읽기 가이드 / Reading Guide

### 읽기 순서 / Recommended Reading Order

1. **Abstract + Section 1 (Introduction)** — Internal Covariate Shift 문제 정의와 동기를 파악하세요.
   Understand the problem definition and motivation for Internal Covariate Shift.

2. **Section 2 (Towards Reducing ICS)** — 왜 단순 정규화가 실패하는지 (bias $b$가 무한히 커지는 예시) 주의 깊게 읽으세요. 이것이 BN 설계의 핵심 동기입니다.
   Read carefully why naive normalization fails (the example where bias $b$ grows indefinitely). This is the key motivation for BN's design.

3. **Section 3 (Normalization via Mini-Batch Statistics)** — 핵심 알고리즘입니다. Algorithm 1 (BN Transform)과 Algorithm 2 (Training procedure)를 완전히 이해하세요.
   The core algorithm. Fully understand Algorithm 1 (BN Transform) and Algorithm 2 (Training procedure).

4. **Section 3.1-3.4** — BN의 네 가지 장점 (training/inference, higher learning rates, gradient propagation, regularization)을 하나씩 확인하세요.
   Verify BN's four advantages one by one.

5. **Section 4 (Experiments)** — Figure 1 (MNIST), Figure 2-3 (ImageNet), Figure 4 (SOTA 비교)에 집중하세요.
   Focus on the experimental figures and tables.

### 주의할 점 / Points to Watch

- **Section 2의 "naive normalization" 실패 예시**: $\hat{x} = x - \text{E}[x]$만 하면 왜 $b$가 발산하는지 — gradient descent가 normalization을 무시하기 때문입니다.
  Why subtracting the mean alone causes $b$ to diverge — because gradient descent ignores the normalization.

- **$\gamma$, $\beta$의 역할**: 왜 정규화만으로 부족한지 (sigmoid의 linear regime에 갇히는 문제), identity transform을 표현할 수 있어야 하는 이유를 이해하세요.
  Why normalization alone is insufficient (trapped in sigmoid's linear regime) and why the ability to represent the identity transform is essential.

- **Training vs. Inference**: 훈련과 추론에서 BN이 다르게 동작한다는 점 (mini-batch stats vs. population stats)이 실전에서 매우 중요합니다.
  BN behaves differently during training (mini-batch stats) vs. inference (population stats) — this is crucial in practice.

---

## 7. 현대적 의의 / Modern Significance

### 영향력 / Impact

Batch Normalization은 현대 딥러닝의 **필수 구성 요소**가 되었습니다:

Batch Normalization became a **fundamental building block** of modern deep learning:

- **거의 모든 CNN 아키텍처에 포함**: ResNet (2015), DenseNet (2017), EfficientNet (2019) 등 거의 모든 후속 아키텍처가 BN을 기본으로 사용합니다.
  Nearly all CNN architectures (ResNet, DenseNet, EfficientNet) use BN as a default component.

- **훈련 레시피의 혁명**: 높은 learning rate, Dropout 제거, 느슨한 초기화 — BN이 가능하게 한 변화들이 현대 훈련 관행의 기초가 되었습니다.
  Higher learning rates, removal of Dropout, relaxed initialization — changes enabled by BN became foundations of modern training practice.

- **후속 정규화 기법들의 영감**: Layer Normalization (Ba et al., 2016), Instance Normalization (Ulyanov et al., 2016), Group Normalization (Wu & He, 2018) 등 BN의 한계를 극복하려는 다양한 변형들이 등장했습니다.
  Inspired many normalization variants (LayerNorm, InstanceNorm, GroupNorm) that address BN's limitations.

### 현대적 한계 / Modern Limitations

- **작은 batch size에서 성능 저하**: batch 통계가 불안정해져 Group Normalization 등이 대안으로 사용됩니다.
  Poor performance with small batch sizes; GroupNorm is used as an alternative.

- **RNN/Transformer에 부적합**: 가변 길이 시퀀스에서 batch 통계 계산이 어려워 Layer Normalization이 표준입니다.
  Not suitable for RNNs/Transformers; LayerNorm became standard for sequential models.

- **"Internal Covariate Shift" 가설 논쟁**: Santurkar et al. (2018)의 "How Does Batch Normalization Help Optimization?"은 BN의 효과가 ICS 감소보다는 loss landscape smoothing에 있다고 주장하여 논쟁을 불러일으켰습니다.
  The ICS hypothesis is debated; Santurkar et al. (2018) argued BN's benefit comes from smoothing the loss landscape rather than reducing ICS.

---

## Q&A

### Q1. Batch 단위 normalization은 배치 내 샘플들이 비슷한 정보를 갖고 있다고 가정하는 건가요?

**아닙니다.** "비슷한 정보"가 아니라 **"같은 분포에서 샘플링되었다(i.i.d.)"는 가정**입니다.

**No.** The assumption is not "similar information" but **"sampled from the same distribution (i.i.d.)"**.

배치 내 샘플들의 **내용(content)**이 비슷할 필요는 없습니다. 고양이 이미지와 자동차 이미지가 같은 배치에 있어도 괜찮습니다. 중요한 것은 그 activation 값들이 **같은 통계적 분포**를 따른다는 것이고, 미니배치의 평균과 분산이 전체 데이터셋의 통계량에 대한 합리적인 **추정치(estimate)**가 될 수 있다는 점입니다.

The **content** of samples within a batch need not be similar — a cat image and a car image in the same batch is fine. What matters is that their activation values follow the **same statistical distribution**, and the mini-batch mean and variance serve as reasonable **estimates** of the full dataset statistics.

실전에서의 주의점 / Practical considerations:
- **배치 크기가 너무 작으면** 통계 추정이 불안정 (batch size 1이면 분산이 0)
  Very small batch sizes make statistics unstable (batch size 1 → variance is 0)
- 배치 내 다른 샘플들과 함께 정규화되면서 **stochastic한 정규화 노이즈**가 발생하고, 이것이 **regularization 효과** (Section 3.4)를 줌
  Normalizing with other samples introduces stochastic noise, which provides a **regularization effect** (Section 3.4)
- **Shuffling이 중요**: 같은 클래스만 모인 배치는 통계가 편향됨 (논문 p.6: within-shard shuffling으로 ~1% 정확도 향상)
  **Shuffling matters**: batches of the same class bias statistics (paper p.6: within-shard shuffling improved accuracy by ~1%)

---

### Q2. U-Net의 1×1 bottleneck처럼 spatial dimension이 극단적으로 작은 경우에는?

논문 Section 3.2에서 CNN용 BN을 설명할 때, feature map 크기 $p \times q$, 배치 크기 $m$이면 effective mini-batch 크기는 $m' = m \cdot p \cdot q$입니다.

Per Section 3.2, for CNN-based BN with feature maps of size $p \times q$ and batch size $m$, the effective mini-batch size is $m' = m \cdot p \cdot q$.

**1×1 feature map**이면 $m' = m \cdot 1 \cdot 1 = m$으로, 배치 크기 자체가 통계 추정의 유일한 원천이 됩니다. 배치 크기가 충분히 크면 (16~32) 동작은 하지만, 실전에서는 대안들이 선호됩니다:

For **1×1 feature maps**, $m' = m$, so the batch size is the sole source of statistics. It works if $m$ is large enough (16–32), but in practice alternatives are preferred:

| 방법 / Method | 설명 / Description |
|---|---|
| **Group Normalization** | 채널을 그룹으로 나눠 정규화. spatial 크기와 무관. U-Net에서 자주 사용 / Normalizes across channel groups; spatial-size-independent. Common in U-Net |
| **Instance Normalization** | 각 샘플, 각 채널별 독립 정규화. spatial이 충분히 클 때만 유효 / Per-sample, per-channel normalization; effective only with sufficient spatial size |
| **Layer Normalization** | 한 샘플의 모든 채널+spatial을 정규화. batch 크기와 무관 / Normalizes all channels+spatial for one sample; batch-size-independent |
| **BN 그대로 사용** | $m$이 충분히 크면 동작하나, 분산 추정이 noisy / Works if $m$ is large enough, but variance estimates are noisy |

**실무적 권장 / Practical recommendation**: U-Net bottleneck에서는 **GroupNorm**이 가장 안정적입니다. 의료 영상 등에서 배치 크기가 작을 수밖에 없는 경우가 많아, 최근 U-Net 구현들은 BN 대신 GroupNorm을 기본으로 사용합니다.

**GroupNorm** is the most stable choice for U-Net bottlenecks. Medical imaging often requires small batch sizes, so modern U-Net implementations default to GroupNorm over BN.

---

### Q3. 평균이 이미 0인 레이어에서는 어떻게 동작하나요?

평균이 0이면 BN의 mean subtraction 단계가 사실상 **no-op**이 됩니다:

When the mean is already 0, BN's mean subtraction step becomes a **no-op**:

$$\mu_{\mathcal{B}} \approx 0 \implies \hat{x}_i = \frac{x_i - 0}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}} = \frac{x_i}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}}$$

즉, **분산 정규화(variance normalization)만 수행**됩니다. 이것 자체는 전혀 문제가 없습니다.

Only **variance normalization** is performed. This is perfectly fine.

$\gamma$, $\beta$의 역할이 여기서 중요합니다 / The role of $\gamma$, $\beta$ is crucial here:
- 네트워크가 평균 0이 최적이라면: $\beta \approx 0$으로 학습됨 / If zero mean is optimal: $\beta \approx 0$ is learned
- 평균 0이 최적이 아니라면: $\beta$가 적절한 값으로 학습되어 **최적의 평균을 자동으로 찾음** / If not: $\beta$ learns the optimal mean automatically

극단적인 경우들 / Edge cases:
- **평균 0 + 분산 1**: $\hat{x}_i \approx x_i$ (identity). 네트워크는 $\gamma \approx 1$, $\beta \approx 0$으로 학습하여 BN을 사실상 "통과"시킴
  Mean 0 + variance 1: normalization becomes near-identity; network learns $\gamma \approx 1$, $\beta \approx 0$, effectively "passing through" BN
- **모든 값이 동일** (분산 = 0): $\epsilon$이 0으로 나누는 것을 방지. $\hat{x}_i = 0$이 되고, 출력은 $y_i = \beta$로 상수
  All values identical (variance = 0): $\epsilon$ prevents division by zero; $\hat{x}_i = 0$, output $y_i = \beta$ (constant)

핵심 통찰 (논문 p.3) / Key insight (paper p.3):
> "$\gamma^{(k)} = \sqrt{\text{Var}[x^{(k)}]}$, $\beta^{(k)} = \text{E}[x^{(k)}]$로 설정하면 원래 activation을 복원 가능"
> "by setting $\gamma^{(k)} = \sqrt{\text{Var}[x^{(k)}]}$ and $\beta^{(k)} = \text{E}[x^{(k)}]$, we could recover the original activations"

**BN은 "항상 정규화를 강제"하는 게 아니라, 네트워크가 정규화할지 말지를 학습하게 해주는 메커니즘입니다.**

**BN does not "force normalization" — it lets the network learn whether or not to normalize.**
