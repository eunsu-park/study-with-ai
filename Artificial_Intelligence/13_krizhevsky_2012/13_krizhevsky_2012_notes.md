---
title: "ImageNet Classification with Deep Convolutional Neural Networks"
authors: Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton
year: 2012
journal: "Advances in Neural Information Processing Systems 25 (NIPS 2012), pp. 1097–1105"
topic: Artificial Intelligence / Deep Learning
tags: [AlexNet, CNN, convolutional neural network, ImageNet, ILSVRC, ReLU, dropout, data augmentation, GPU training, local response normalization, overlapping pooling, deep learning, computer vision, large-scale image classification]
status: completed
date_started: 2026-04-13
date_completed: 2026-04-13
---

# ImageNet Classification with Deep Convolutional Neural Networks
**Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton (2012)**

---

## 핵심 기여 / Core Contribution

이 논문은 **딥러닝이 대규모 시각 인식에서 기존 방법을 압도한다**는 것을 실증적으로 증명한 기념비적 논문입니다. 2012년 ImageNet Large Scale Visual Recognition Challenge (ILSVRC)에서 top-5 error **15.3%**를 달성하여, 2위의 hand-crafted feature 방식(26.2%)을 **10.8%p** 차이로 압도했습니다. 핵심 기술 기여는 다섯 가지입니다. 첫째, **ReLU** 활성화 함수를 도입하여 tanh/sigmoid 대비 **6배 빠른 학습**을 달성했습니다 (§3.1). 둘째, GTX 580 GPU 2개에 네트워크를 분할하는 **2-GPU 병렬 학습 아키텍처**를 설계하여 메모리 제약을 극복했고, 이 과정에서 두 GPU가 자연스럽게 색상-agnostic/색상-specific 필터로 분업하는 emergent specialization을 관찰했습니다 (§3.2, §6.1). 셋째, **Dropout** 정규화(p=0.5)를 fully-connected 층에 적용하여 6천만 파라미터의 overfitting을 효과적으로 방지했습니다 (§4.2). 넷째, 두 가지 **Data Augmentation** — random 224×224 crop + horizontal flip (2048배 증강)과 PCA 기반 색상 변화 (top-1 error 1%+ 감소) — 을 적용했습니다 (§4.1). 다섯째, **Local Response Normalization**과 **Overlapping Pooling** 등 세부 기법으로 추가적 성능 향상을 달성했습니다. 이 논문의 가장 중요한 메시지는 "깊이가 중요하다"는 것입니다 — 어떤 합성곱 층을 하나라도 제거하면 top-1 error가 약 2% 상승합니다. 특히 주목할 점은, Paper #12 (Hinton 2006)의 비지도 사전학습(unsupervised pre-training) 없이도 순수한 지도 학습만으로 대규모 deep network를 성공적으로 학습할 수 있음을 보여주었다는 것입니다.

This paper is the landmark that **empirically proved deep learning's dominance in large-scale visual recognition**. At ILSVRC 2012, it achieved a top-5 error of **15.3%**, crushing the runner-up's hand-crafted feature approach (26.2%) by **10.8 percentage points**. Five key technical contributions: First, the **ReLU** activation function achieves **6x faster training** compared to tanh/sigmoid (§3.1). Second, a **2-GPU parallel training architecture** splits the network across two GTX 580 GPUs, overcoming memory constraints, and produces emergent specialization where the two GPUs naturally divide into color-agnostic and color-specific filters (§3.2, §6.1). Third, **Dropout** regularization (p=0.5) on fully-connected layers effectively prevents overfitting in a 60-million-parameter network (§4.2). Fourth, two forms of **Data Augmentation** — random 224×224 crops + horizontal flips (2048x augmentation) and PCA-based color perturbation (>1% top-1 error reduction) — are applied (§4.1). Fifth, **Local Response Normalization** and **Overlapping Pooling** provide additional performance gains. The paper's most important message is "depth matters" — removing any single convolutional layer increases top-1 error by ~2%. Notably, this paper demonstrates that large-scale deep networks can be successfully trained with **purely supervised learning**, without the unsupervised pre-training of Paper #12 (Hinton 2006).

---

## 읽기 노트 / Reading Notes

### §1: Introduction — 왜 대규모 CNN인가 / Why Large-Scale CNN

#### 기존 접근법의 한계 / Limitations of Previous Approaches

논문은 물체 인식(object recognition)의 가변성(variability) 문제로 시작합니다. 현실 세계의 물체는 시점, 조명, 크기, 형태가 매우 다양하여, 이를 인식하려면 **대규모 학습 데이터**가 필요합니다. 기존 데이터셋(NORB, Caltech-101/256, CIFAR-10/100)은 수만 장 수준으로 부족했으나, ImageNet(1,500만+ 이미지, 22,000+ 카테고리)의 등장으로 충분한 데이터가 확보되었습니다.

The paper begins with the variability problem in object recognition. Real-world objects vary enormously in viewpoint, illumination, scale, and shape, requiring **large-scale training data**. Previous datasets (NORB, Caltech-101/256, CIFAR-10/100) had only tens of thousands of images, but ImageNet (15M+ images, 22K+ categories) provided sufficient data.

#### CNN의 장점 / Advantages of CNNs

CNN은 이미지에 대한 **두 가지 강력한 사전 지식(prior)**을 내장하고 있습니다:

CNNs embed **two powerful priors** about images:

1. **통계의 정상성(stationarity of statistics)**: 이미지의 한 부분에서 유용한 패턴은 다른 부분에서도 유용합니다 → 가중치 공유(weight sharing)가 가능. 예: 고양이 귀의 edge는 이미지 어디에 있든 같은 필터로 감지 가능

   **Stationarity of statistics**: Useful patterns in one part of an image are useful elsewhere → weight sharing is possible. E.g., the edge of a cat's ear can be detected by the same filter regardless of position

2. **픽셀 의존성의 지역성(locality of pixel dependencies)**: 가까운 픽셀끼리의 관계가 멀리 떨어진 픽셀보다 중요합니다 → 지역적 연결(local connectivity)이 효과적

   **Locality of pixel dependencies**: Nearby pixels are more related than distant ones → local connectivity is effective

이 두 가정 덕분에 CNN은 같은 크기의 fully-connected 네트워크보다 **훨씬 적은 연결과 파라미터**를 가지면서도 이론적으로 거의 같은 성능을 달성할 수 있습니다.

Thanks to these two assumptions, CNNs have **far fewer connections and parameters** than same-sized fully-connected networks while theoretically achieving nearly equivalent performance.

#### GPU의 역할 / Role of GPUs

저자들은 GPU가 CNN의 대규모 적용을 가능하게 한 핵심 기술이라고 강조합니다. 논문의 핵심 메시지: "데이터(ImageNet) + 모델 용량(deep CNN) + 하드웨어(GPU) + 정규화(dropout, augmentation)의 조합"이 이 결과를 만들었습니다.

The authors emphasize GPUs as the key technology enabling large-scale CNN application. Core message: "the combination of data (ImageNet) + model capacity (deep CNN) + hardware (GPU) + regularization (dropout, augmentation)" produced these results.

---

### §2: The Dataset — ImageNet과 ILSVRC

#### ImageNet의 규모 / Scale of ImageNet

- **전체**: 1,500만+ 고해상도 라벨 이미지, 22,000+ 카테고리 / Full: 15M+ labeled high-resolution images, 22K+ categories
- **ILSVRC subset**: 1,000 카테고리, 각 ~1,000 이미지 / ILSVRC subset: 1,000 categories, ~1,000 images each
  - 학습: ~120만 / Training: ~1.2M
  - 검증: 50,000 / Validation: 50,000
  - 테스트: 150,000 / Test: 150,000

Amazon Mechanical Turk 크라우드소싱으로 라벨링되었습니다.

Labeled via Amazon Mechanical Turk crowdsourcing.

#### 전처리 / Preprocessing

가변 해상도 이미지를 **고정 크기 256×256**으로 변환하는 과정:

Converting variable-resolution images to **fixed 256×256**:

1. 짧은 변을 256으로 rescale / Rescale shorter side to 256
2. 중앙 256×256 패치를 crop / Crop central 256×256 patch
3. 각 픽셀에서 학습 세트의 평균값을 뺌 (mean subtraction) / Subtract training set mean from each pixel

**추가 전처리 없음** — 원시 RGB 픽셀 값에서 직접 학습합니다. Hand-crafted feature 추출이 전혀 없다는 점이 기존 방법과의 핵심 차이입니다.

**No additional preprocessing** — training directly on raw RGB pixel values. The complete absence of hand-crafted feature extraction is the key difference from previous methods.

#### 평가 지표 / Evaluation Metrics

- **Top-1 error**: 모델의 최상위 1개 예측이 정답이 아닌 비율 / Fraction where the model's single top prediction is incorrect
- **Top-5 error**: 모델의 상위 5개 예측에 정답이 없는 비율 / Fraction where correct label is not in the model's top 5 predictions

1,000개 클래스에서 Top-1은 매우 가혹하므로(시각적으로 유사한 하위 카테고리가 많음), Top-5가 주요 지표로 사용됩니다.

With 1,000 classes, Top-1 is very strict (many visually similar subcategories), so Top-5 is the primary metric.

---

### §3: The Architecture — AlexNet의 설계

저자들은 §3.1–3.4의 기법들을 **중요도 순서대로** 정렬하여 서술합니다. 이것 자체가 중요한 정보입니다.

The authors sort §3.1–3.4 techniques **by estimated importance**. This ordering itself is important information.

#### §3.1: ReLU Nonlinearity — 가장 중요한 기여

$$f(x) = \max(0, x)$$

**왜 중요한가 / Why it matters:**

기존 활성화 함수(tanh, sigmoid)는 **saturating nonlinearity**입니다 — 입력이 크거나 작으면 출력이 거의 변하지 않고(기울기 ≈ 0), 이것이 deep network에서 gradient를 소멸시킵니다. ReLU는 양수 영역에서 기울기가 항상 1이므로 이 문제를 해결합니다.

Previous activation functions (tanh, sigmoid) are **saturating nonlinearities** — for large or small inputs, output barely changes (gradient ≈ 0), causing gradient vanishing in deep networks. ReLU has gradient always equal to 1 in the positive region, solving this problem.

**실험 결과 (Figure 1)**: CIFAR-10에서 4층 CNN 학습 시, ReLU는 tanh보다 **6배 빠르게** 25% 학습 오류에 도달합니다. 논문은 이 속도 차이가 없었다면 "이렇게 큰 네트워크를 이 데이터셋에서 실험하는 것 자체가 불가능했을 것"이라고 명시합니다.

**Experimental result (Figure 1)**: On CIFAR-10 with a 4-layer CNN, ReLU reaches 25% training error **6x faster** than tanh. The paper states this speed difference was necessary — without it, "we would not have been able to experiment with such large neural networks."

**Nair and Hinton (2010)의 선행 연구**: ReLU 이름은 이 논문에서 처음 사용된 것이 아닙니다. Nair와 Hinton이 RBM에서 ReLU의 효과를 먼저 연구했으며, AlexNet은 이를 대규모 CNN에 적용한 것입니다.

**Prior work by Nair and Hinton (2010)**: The ReLU name was not coined in this paper. Nair and Hinton first studied ReLU's effect in RBMs; AlexNet applied it to large-scale CNNs.

#### §3.2: Training on Multiple GPUs — 메모리 제약의 극복

GTX 580의 VRAM은 **3GB**에 불과합니다. 6천만 파라미터의 네트워크를 하나의 GPU에 올릴 수 없었기 때문에, 네트워크를 **반으로 나눠 2개의 GPU**에 배치합니다.

GTX 580 has only **3GB** VRAM. A 60-million-parameter network couldn't fit on a single GPU, so the network is **split in half across 2 GPUs**.

**GPU 간 통신 규칙 / Inter-GPU communication rules:**
- Conv2, Conv4, Conv5: 같은 GPU의 이전 층에서만 입력 받음 (교차 통신 없음) / Input only from the same GPU's previous layer (no cross-communication)
- **Conv3**: 양쪽 GPU의 모든 Conv2 feature map에서 입력 받음 (전체 교차) / Input from **all** Conv2 feature maps on both GPUs (full cross-communication)
- FC 층: 모든 뉴런이 연결됨 / All neurons connected

이 제한된 교차 연결 패턴은 cross-validation으로 최적화했으며, top-1/top-5 error를 **1.7%/1.2%** 감소시켰습니다 (1-GPU 대비).

This limited cross-connection pattern was optimized via cross-validation and reduced top-1/top-5 error by **1.7%/1.2%** (vs. 1-GPU).

**Emergent specialization (§6.1에서 확인)**: GPU 1은 색상-agnostic 필터(edge, frequency), GPU 2는 색상-specific 필터(color blob)를 학습합니다. 이것은 의도적 설계가 아닌 **자발적으로 나타난 현상**이며, 랜덤 초기화에 무관하게 매번 재현됩니다.

**Emergent specialization (confirmed in §6.1)**: GPU 1 learns color-agnostic filters (edges, frequencies), GPU 2 learns color-specific filters (color blobs). This is **emergent behavior**, not intentional design, and reproduces regardless of random weight initialization.

#### §3.3: Local Response Normalization (LRN) — 측면 억제

$$b^i_{x,y} = \frac{a^i_{x,y}}{\left(k + \alpha \sum_{j=\max(0,i-n/2)}^{\min(N-1,i+n/2)} (a^j_{x,y})^2\right)^\beta}$$

하이퍼파라미터: $k=2, n=5, \alpha=10^{-4}, \beta=0.75$ (validation set에서 결정)

Hyperparameters: $k=2, n=5, \alpha=10^{-4}, \beta=0.75$ (determined on validation set)

**메커니즘**: 같은 공간 위치 $(x,y)$에서 인접한 $n$개의 커널(feature map) 간 **경쟁**을 유발합니다. 강하게 활성화된 커널이 인접 커널의 반응을 억제합니다 — 생물학적 뉴런의 **lateral inhibition**에서 영감.

**Mechanism**: Creates **competition** among $n$ adjacent kernels (feature maps) at the same spatial position $(x,y)$. Strongly activated kernels suppress neighboring kernel responses — inspired by biological **lateral inhibition**.

**ReLU와의 관계**: ReLU는 saturation이 없으므로 입력 정규화가 이론적으로 불필요하지만, 이 지역적 정규화가 일반화에 도움이 됩니다. top-1/top-5 error **1.4%/1.2%** 감소.

**Relation to ReLU**: ReLU doesn't require input normalization to prevent saturation, but this local normalization aids generalization. Reduces top-1/top-5 error by **1.4%/1.2%**.

> **현대적 관점**: LRN은 이후 VGGNet (2014)에서 효과가 미미한 것으로 밝혀졌고, Batch Normalization (Ioffe & Szegedy, 2015)으로 완전히 대체되었습니다.
>
> **Modern perspective**: LRN was later found to have negligible effect by VGGNet (2014) and was completely replaced by Batch Normalization (Ioffe & Szegedy, 2015).

#### §3.4: Overlapping Pooling — 겹치는 풀링

전통적 pooling: 풀링 영역이 겹치지 않음 ($s = z$, stride = pool size). AlexNet은 **겹치는 pooling**을 사용합니다: $s = 2, z = 3$ (stride 2, pool size 3×3).

Traditional pooling: non-overlapping ($s = z$). AlexNet uses **overlapping pooling**: $s = 2, z = 3$ (stride 2, pool size 3×3).

효과: top-1/top-5 error **0.4%/0.3%** 감소, overfitting도 약간 감소.

Effect: reduces top-1/top-5 error by **0.4%/0.3%**, slightly reduces overfitting.

#### §3.5: Overall Architecture — 전체 아키텍처

```
입력 / Input: 224 × 224 × 3 (RGB)

Conv1: 96 kernels, 11×11×3, stride 4    → 55×55×96
  + ReLU → LRN → MaxPool(3×3, stride 2) → 27×27×96

Conv2: 256 kernels, 5×5×48, pad 2       → 27×27×256
  + ReLU → LRN → MaxPool(3×3, stride 2) → 13×13×256

Conv3: 384 kernels, 3×3×256, pad 1      → 13×13×384
  + ReLU                                  (Conv2의 양쪽 GPU 모두에서 입력)

Conv4: 384 kernels, 3×3×192, pad 1      → 13×13×384
  + ReLU                                  (같은 GPU에서만 입력)

Conv5: 256 kernels, 3×3×192, pad 1      → 13×13×256
  + ReLU → MaxPool(3×3, stride 2)        → 6×6×256

Flatten: 6×6×256 = 9,216

FC6: 4096 units + ReLU + Dropout(0.5)
FC7: 4096 units + ReLU + Dropout(0.5)
FC8: 1000 units + Softmax
```

**파라미터 분포 / Parameter distribution:**

| 층 / Layer | 파라미터 수 / Parameters | 비율 / Ratio |
|---|---|---|
| Conv1 | 11×11×3×96 = 34,848 | 0.06% |
| Conv2 | 5×5×48×256 = 307,200 | 0.51% |
| Conv3 | 3×3×256×384 = 884,736 | 1.47% |
| Conv4 | 3×3×192×384 = 663,552 | 1.11% |
| Conv5 | 3×3×192×256 = 442,368 | 0.74% |
| **FC6** | **9,216×4,096 = 37,748,736** | **62.9%** |
| FC7 | 4,096×4,096 = 16,777,216 | 28.0% |
| FC8 | 4,096×1,000 = 4,096,000 | 6.8% |
| **합계 / Total** | **~60M** | **100%** |

파라미터의 **97.7%가 FC 층**에 집중되어 있습니다. Conv 층은 모델 파라미터의 단 2.3%이지만, 어떤 Conv 층을 제거해도 성능이 ~2% 하락합니다 — 깊이의 중요성을 보여줍니다.

**97.7% of parameters are in FC layers**. Conv layers are only 2.3% of parameters, yet removing any Conv layer degrades performance by ~2% — demonstrating the importance of depth.

**목적 함수 / Objective**: Multinomial logistic regression = softmax + cross-entropy loss:

$$L = -\frac{1}{N}\sum_{i=1}^{N} \log P(y_i = \text{correct} \mid \mathbf{x}_i) = -\frac{1}{N}\sum_{i=1}^{N} \log \frac{e^{z_{y_i}}}{\sum_{j=1}^{1000} e^{z_j}}$$

---

### §4: Reducing Overfitting — 과적합 방지

6천만 파라미터 + 120만 학습 이미지 = 심각한 overfitting 위험. 각 학습 이미지가 1,000 클래스 분류에 부과하는 제약은 약 **10비트** ($\log_2 1000 \approx 10$)이므로, 전체 제약은 $120만 \times 10 = 1,200만$ 비트. 이것은 6천만 파라미터(각 32비트 float)를 충분히 제약하지 못합니다.

60M parameters + 1.2M training images = serious overfitting risk. Each training image imposes ~**10 bits** of constraint ($\log_2 1000 \approx 10$), so total constraint is $1.2M \times 10 = 12M$ bits — insufficient to constrain 60M parameters (each 32-bit float).

#### §4.1: Data Augmentation — 데이터 증강

**구현의 핵심**: CPU에서 Python으로 변환된 이미지를 생성하는 동안 GPU는 이전 배치를 학습합니다. 따라서 이 증강은 "사실상 계산 비용이 없습니다(computationally free)."

**Key implementation detail**: Transformed images are generated on CPU in Python while the GPU trains on the previous batch. Thus augmentation is "computationally free."

**기법 1: Random Crop + Horizontal Flip**

- 학습 시: 256×256 이미지에서 **무작위** 224×224 패치 추출 + 50% 확률로 좌우 반전 / Training: extract **random** 224×224 patches from 256×256 images + 50% horizontal flip
- 가능한 패치: $(256-224+1)^2 = 33^2 = 1,089$ 위치 × 2 (반전) = **2,178** (논문은 약 2,048배라 기술) / Possible patches: 1,089 positions × 2 (flip) ≈ **2,048x** augmentation
- 테스트 시: 4 코너 + 중앙 = 5 패치 × 2 (반전) = **10 패치**의 softmax 예측을 평균 / Test: 4 corners + center = 5 patches × 2 (flips) = average softmax predictions over **10 patches**
- "이 기법 없이는 상당한 overfitting으로 훨씬 작은 네트워크를 사용해야 했을 것" / "Without this scheme, our network suffers from substantial overfitting"

**기법 2: PCA Color Augmentation**

$$\Delta \mathbf{c} = [\mathbf{p}_1, \mathbf{p}_2, \mathbf{p}_3][\alpha_1 \lambda_1, \alpha_2 \lambda_2, \alpha_3 \lambda_3]^T, \quad \alpha_i \sim \mathcal{N}(0, 0.1)$$

- ImageNet 전체 학습 이미지의 RGB 픽셀에서 3×3 공분산 행렬의 고유분해 수행 / Eigendecomposition of 3×3 covariance matrix of RGB pixels across all training images
- 각 학습 이미지에 $\Delta \mathbf{c}$를 모든 픽셀에 동일하게 추가 ($\alpha$는 이미지마다 한 번 샘플링) / Add $\Delta \mathbf{c}$ to all pixels of each training image ($\alpha$ sampled once per image)
- 자연 이미지의 조명 변화가 RGB 주성분 방향을 따른다는 관찰에 기반 / Based on the observation that illumination changes in natural images follow RGB principal component directions
- top-1 error **1% 이상** 감소 / Top-1 error reduced by **over 1%**

#### §4.2: Dropout — 드롭아웃

학습 시 각 hidden neuron의 출력을 확률 0.5로 0으로 설정합니다:

During training, set each hidden neuron's output to 0 with probability 0.5:

$$h_i^{\text{dropped}} = \begin{cases} 0 & \text{with probability } 0.5 \\ h_i & \text{with probability } 0.5 \end{cases}$$

테스트 시에는 모든 뉴런을 사용하되 출력에 0.5를 곱합니다 — 지수적으로 많은 "thinned" 네트워크의 예측을 기하 평균하는 것의 근사입니다.

At test time, use all neurons but multiply outputs by 0.5 — an approximation to the geometric mean of exponentially many "thinned" networks' predictions.

**적용 위치**: FC6, FC7 (처음 두 fully-connected 층에서만) / **Where applied**: FC6, FC7 (first two fully-connected layers only)

**효과 / Effects:**
- 뉴런 간 "co-adaptation" 방지: 특정 다른 뉴런의 존재에 의존할 수 없으므로, 각 뉴런이 독립적으로 유용한 특징을 학습 / Prevents "co-adaptation": each neuron can't rely on specific other neurons, so learns independently useful features
- 수렴에 필요한 iteration 수가 약 **2배** 증가 / Approximately **doubles** the number of iterations to converge
- "Dropout 없이는 상당한 overfitting을 보임" / "Without dropout, our network exhibits substantial overfitting"

---

### §5: Details of Learning — 학습 세부사항

#### SGD with Momentum

$$v_{i+1} = 0.9 \cdot v_i - 0.0005 \cdot \epsilon \cdot w_i - \epsilon \cdot \left\langle \frac{\partial L}{\partial w}\bigg|_{w_i} \right\rangle_{D_i}$$

$$w_{i+1} = w_i + v_{i+1}$$

| 하이퍼파라미터 / Hyperparameter | 값 / Value | 비고 / Note |
|---|---|---|
| Batch size | 128 | |
| Momentum | 0.9 | |
| Weight decay | 0.0005 | 정규화 + 학습 자체에 도움 / Regularization + helps training itself |
| Initial learning rate | 0.01 | |
| LR schedule | validation error 정체 시 1/10로 감소 / Divide by 10 when val error plateaus | 총 3번 감소 / Reduced 3 times |
| Total epochs | ~90 | |
| Training time | 5–6일 (GTX 580 × 2) / 5–6 days on 2× GTX 580 | |

**Weight decay에 대한 중요한 관찰**: 저자들은 weight decay가 "단순한 정규화가 아니라 학습 자체에 중요하다"고 강조합니다. 이것은 L2 정규화가 가중치를 작게 유지하여 gradient flow를 개선하고, 네트워크가 보다 분산된 표현을 학습하도록 유도하기 때문으로 해석됩니다.

**Important observation on weight decay**: The authors emphasize it's "not merely a regularizer: it reduces the model's training error." This can be interpreted as L2 regularization maintaining small weights to improve gradient flow and encouraging more distributed representations.

#### 초기화 / Initialization

- **가중치**: 모든 층에서 $\mathcal{N}(0, 0.01)$ / All layers: $\mathcal{N}(0, 0.01)$
- **Bias**:
  - Conv2, Conv4, Conv5, FC6, FC7: **1** → ReLU에 양수 입력을 제공하여 초기 학습을 가속 / Provides positive inputs to ReLU, accelerating early learning
  - Conv1, Conv3, FC8: **0**

Bias를 1로 초기화하는 것은 ReLU의 "dead neuron" 문제를 방지하기 위한 것입니다. ReLU는 음수 입력에서 gradient가 0이므로, 초기에 양수 입력을 제공하면 더 많은 뉴런이 활성화되어 학습이 시작됩니다.

Initializing bias to 1 prevents ReLU's "dead neuron" problem. Since ReLU has zero gradient for negative inputs, providing positive inputs initially activates more neurons to start learning.

---

### §6: Results — 결과

#### ILSVRC-2010 결과 (Table 1)

| 방법 / Method | Top-1 | Top-5 |
|---|---|---|
| Sparse coding [2] | 47.1% | 28.2% |
| SIFT + Fisher Vectors [24] | 45.7% | 25.7% |
| **CNN (AlexNet)** | **37.5%** | **17.0%** |

기존 최고 성능 대비 top-1에서 **8.2%**, top-5에서 **8.7%** 향상 — 단일 방법으로 이런 규모의 개선은 전례가 없었습니다.

Improvement of **8.2%** in top-1 and **8.7%** in top-5 over previous best — an unprecedented magnitude of improvement from a single method.

#### ILSVRC-2012 결과 (Table 2)

| 방법 / Method | Top-1 (val) | Top-5 (val) | Top-5 (test) |
|---|---|---|---|
| SIFT + FVs [7] | — | — | 26.2% |
| 1 CNN | 40.7% | 18.2% | — |
| 5 CNNs (앙상블) | 38.1% | 16.4% | **16.4%** |
| 1 CNN* (ImageNet Fall 2011 사전학습) | 39.0% | 16.6% | — |
| 7 CNNs* (앙상블) | 36.7% | 15.4% | **15.3%** |

\* ImageNet Fall 2011 전체 릴리스(15M 이미지, 22K 카테고리)로 사전학습 후 ILSVRC-2012에 fine-tuning

\* Pre-trained on full ImageNet Fall 2011 release (15M images, 22K categories), then fine-tuned on ILSVRC-2012

**2위와의 격차**: 15.3% vs. 26.2% = **10.9%p 차이**. 이 격차는 이전 대회에서의 연간 개선폭(1-2%p)을 훨씬 초과하며, 컴퓨터 비전 커뮤니티 전체에 충격을 주었습니다.

**Gap from runner-up**: 15.3% vs. 26.2% = **10.9pp difference**. This gap far exceeded previous annual improvements (1-2pp), shocking the entire computer vision community.

#### §6.1: Qualitative Evaluations — 정성적 분석

**학습된 필터 (Figure 3)**: Conv1의 96개 11×11×3 필터를 시각화. GPU 1의 48개 필터는 **색상과 무관한**(color-agnostic) 패턴 — Gabor-like edge와 frequency 패턴. GPU 2의 48개 필터는 **색상에 특화된**(color-specific) 패턴 — color blob과 color edge.

**Learned filters (Figure 3)**: Visualization of Conv1's 96 11×11×3 filters. GPU 1's 48 filters show **color-agnostic** patterns — Gabor-like edges and frequency patterns. GPU 2's 48 filters show **color-specific** patterns — color blobs and color edges.

**Feature vector 유사도 (Figure 4 오른쪽)**: 마지막 hidden layer(4096차원)에서의 feature vector 간 Euclidean distance로 이미지 유사도를 측정합니다. 놀라운 결과: **픽셀 수준에서는 전혀 다르지만 의미적으로 유사한 이미지**가 가까운 벡터 표현을 갖습니다. 예: 다양한 포즈의 코끼리들, 다양한 포즈의 개들이 서로 가까운 벡터를 가집니다. 이것은 네트워크가 "semantic representation"을 학습했음을 시사합니다.

**Feature vector similarity (Figure 4 right)**: Measuring image similarity via Euclidean distance in the last hidden layer (4096-dim). Remarkable result: **images that are completely different at pixel level but semantically similar** have close vector representations. E.g., elephants in various poses, dogs in various poses cluster together. This suggests the network learned "semantic representation."

저자들은 이 4096차원 벡터를 autoencoder로 짧은 이진 코드로 압축하면, raw pixel에 autoencoder를 적용하는 것보다 훨씬 나은 이미지 검색이 가능할 것이라고 제안합니다 — 이 아이디어는 이후 **transfer learning**의 핵심이 됩니다.

The authors suggest compressing these 4096-dim vectors to short binary codes via autoencoder for image retrieval — this idea later becomes central to **transfer learning**.

---

### §7: Discussion — 토론

#### 깊이의 중요성 / Importance of Depth

"어떤 중간 합성곱 층을 제거해도 top-1 성능이 약 2% 하락합니다. 따라서 깊이가 우리의 결과를 달성하는 데 정말로 중요합니다."

"Removing any of the middle layers results in a loss of about 2% for the top-1 performance. So the depth really is important for achieving our results."

각 합성곱 층은 모델 파라미터의 1% 미만을 차지하지만, 성능에는 불균형적으로 큰 영향을 미칩니다. 이것은 deep network의 각 층이 **점점 더 추상적인 특징 계층(feature hierarchy)**을 구축한다는 것을 의미합니다.

Each Conv layer contains less than 1% of model parameters but has a disproportionately large impact on performance. This means each layer of a deep network builds **progressively more abstract feature hierarchies**.

#### 비지도 사전학습에 대한 언급 / On Unsupervised Pre-training

"실험을 단순화하기 위해 비지도 사전학습을 사용하지 않았지만, 도움이 될 것으로 기대합니다."

"We did not use any unsupervised pre-training even though we expect that it will help."

이것은 Paper #12 (Hinton 2006)에 대한 직접적 언급입니다. AlexNet은 ReLU + Dropout + Data Augmentation의 조합으로 사전학습 없이도 deep network를 성공적으로 학습할 수 있음을 보여주었고, 이것은 역설적으로 Hinton 본인의 이전 연구(RBM 사전학습)의 역사적 역할이 완료되었음을 시사합니다.

This directly references Paper #12 (Hinton 2006). AlexNet showed that the combination of ReLU + Dropout + Data Augmentation enables successful deep network training without pre-training — paradoxically suggesting that Hinton's own previous work (RBM pre-training) had fulfilled its historical role.

#### 미래 방향 / Future Directions

저자들은 "더 빠른 GPU와 더 큰 데이터셋만으로 결과가 개선될 수 있다"고 예측하며, 궁극적으로 **비디오 시퀀스**에 대규모 deep CNN을 적용하고 싶다고 말합니다. 이 예측은 정확히 실현되었습니다 — 이후 VGGNet (2014), GoogLeNet (2014), ResNet (2015) 등이 더 깊고 큰 네트워크로 성능을 계속 향상시켰습니다.

The authors predict "results can be improved simply by waiting for faster GPUs and bigger datasets," and ultimately want to apply large deep CNNs to **video sequences**. This prediction was exactly realized — VGGNet (2014), GoogLeNet (2014), ResNet (2015) continued improving with deeper and larger networks.

---

## 핵심 시사점 / Key Takeaways

1. **"대규모 데이터 + 대규모 모델 + GPU"의 승리 공식**: AlexNet은 단일 알고리즘의 혁신이 아닌, **세 가지 요소의 시너지**를 증명했습니다. ImageNet(대규모 데이터) + 깊은 CNN(대규모 모델) + GPU(대규모 연산)의 조합이 hand-crafted feature의 수십 년 연구를 단번에 넘어섰습니다. 이 "스케일링 레시피"는 이후 GPT-3, Foundation Model 등 현대 AI의 핵심 원칙이 됩니다.

   **The winning formula of "large data + large model + GPU"**: AlexNet proved the **synergy of three factors**, not a single algorithmic innovation. ImageNet (large data) + deep CNN (large model) + GPU (large compute) surpassed decades of hand-crafted feature research. This "scaling recipe" becomes a core principle of modern AI (GPT-3, Foundation Models).

2. **ReLU — 과소평가된 혁명**: Sigmoid/tanh에서 ReLU로의 전환은 단순한 활성화 함수 변경을 넘어서, **deep network 학습의 실용성**을 결정짓는 변환이었습니다. 6배 빠른 학습이 없었다면 AlexNet의 실험 자체가 불가능했을 것입니다. 이후 ReLU는 사실상 모든 deep network의 기본 활성화 함수가 되었고, Leaky ReLU, PReLU, GELU 등 변종이 등장했습니다.

   **ReLU — the underestimated revolution**: The transition from sigmoid/tanh to ReLU was more than an activation function change — it determined the **practical feasibility of deep network training**. Without 6x faster training, AlexNet's experiments would have been impossible. ReLU became the default activation for virtually all deep networks, spawning variants (Leaky ReLU, PReLU, GELU).

3. **Dropout — 앙상블의 민주화**: 여러 모델을 독립적으로 학습하는 앙상블은 비용이 막대하지만, Dropout은 하나의 네트워크 학습 과정에서 **지수적으로 많은 서브네트워크의 앙상블을 근사**합니다. 학습 비용이 2배 증가하는 것은 모든 가능한 서브네트워크를 개별 학습하는 것에 비하면 극히 저렴합니다. 이 아이디어는 이후 DropConnect, DropPath, Stochastic Depth 등으로 확장됩니다.

   **Dropout — democratization of ensembles**: While independent model ensembles are prohibitively expensive, Dropout **approximates an ensemble of exponentially many subnetworks** during single network training. The 2x training cost increase is negligible compared to training all possible subnetworks individually. Extended to DropConnect, DropPath, Stochastic Depth.

4. **Hand-crafted vs. Learned Features 논쟁의 종결**: SIFT, HOG, Fisher Vector 등 수십 년간 발전시킨 hand-crafted feature가 AlexNet의 end-to-end 학습에 10.9%p 차이로 패배했습니다. 이것은 "도메인 전문가의 특징 설계"에서 "데이터에서 자동으로 특징을 학습"으로의 **패러다임 전환**을 불가역적으로 만들었습니다. 이후 컴퓨터 비전의 거의 모든 연구가 deep learning 기반으로 전환되었습니다.

   **End of the hand-crafted vs. learned features debate**: Decades of hand-crafted features (SIFT, HOG, Fisher Vectors) were defeated by AlexNet's end-to-end learning by 10.9pp. This irreversibly established the **paradigm shift** from "expert-designed features" to "automatically learned features from data." Virtually all computer vision research subsequently shifted to deep learning.

5. **비지도 사전학습의 역사적 역할 완료**: Paper #12 (Hinton 2006)의 RBM 사전학습은 "deep network가 학습 가능하다"는 것을 증명하는 역사적 역할을 했지만, AlexNet은 ReLU + Dropout + Data Augmentation으로 사전학습 없이도 대규모 deep network를 학습할 수 있음을 보여주었습니다. 흥미롭게도, 이 논문의 지도교수인 Hinton 자신이 자기 이전 연구의 필요성을 줄인 셈입니다. 그러나 "사전학습 → fine-tuning" 패러다임 자체는 BERT, GPT 등에서 다른 형태로 부활합니다.

   **Completion of unsupervised pre-training's historical role**: Paper #12's RBM pre-training served the historical role of proving "deep networks can be trained," but AlexNet showed large-scale deep networks can be trained without pre-training via ReLU + Dropout + Data Augmentation. Ironically, Hinton (the advisor) reduced the need for his own previous work. However, the "pre-training → fine-tuning" paradigm resurfaces in different forms (BERT, GPT).

6. **Emergent specialization — 설계하지 않은 지능**: GPU 메모리 제약이라는 공학적 제약에서 색상/형태 분업이라는 의미 있는 구조가 자발적으로 나타났습니다. 이것은 **충분한 용량 + 적절한 제약 + 충분한 데이터** 조건에서 네트워크가 인간이 설계하지 않은 유의미한 내부 구조를 발전시킬 수 있음을 보여주는 초기 사례입니다.

   **Emergent specialization — undesigned intelligence**: A meaningful structure (color/form division) emerged spontaneously from an engineering constraint (GPU memory limitation). An early example showing that under **sufficient capacity + appropriate constraints + sufficient data**, networks can develop meaningful internal structures not designed by humans.

7. **Feature vector로서의 hidden layer — Transfer Learning의 전조**: 마지막 hidden layer의 4096차원 벡터가 의미적 유사도를 포착한다는 발견은, 이 벡터를 다른 task의 입력으로 사용할 수 있음을 시사합니다. 이 아이디어는 이후 **transfer learning** — ImageNet으로 학습한 CNN의 feature를 다른 시각 task에 전이하는 것 — 의 핵심이 되었으며, "pretrained backbone + task-specific head" 패턴은 현대 컴퓨터 비전의 표준 관행이 되었습니다.

   **Hidden layer as feature vector — precursor to transfer learning**: The discovery that the last hidden layer's 4096-dim vector captures semantic similarity suggests using it as input for other tasks. This becomes central to **transfer learning** — transferring ImageNet CNN features to other vision tasks — and the "pretrained backbone + task-specific head" pattern became standard practice in modern computer vision.

8. **"더 크고 더 깊게"의 시작**: 논문의 마지막 문장 "더 빠른 GPU와 더 큰 데이터셋만으로 결과가 개선될 수 있다"는 이후 10년간의 딥러닝 연구 방향을 정확히 예견합니다. VGGNet → GoogLeNet → ResNet → GPT-3 → GPT-4로 이어지는 "스케일링"의 역사가 이 한 문장에서 시작됩니다.

   **The beginning of "bigger and deeper"**: The paper's final statement "results can be improved simply by waiting for faster GPUs and bigger datasets" exactly predicts the next decade of deep learning research. The scaling history from VGGNet → GoogLeNet → ResNet → GPT-3 → GPT-4 begins with this single sentence.

---

## 수학적 요약 / Mathematical Summary

### AlexNet 순전파 / AlexNet Forward Pass

```
=== Input Preprocessing ===
  x_raw: H×W×3 variable-resolution image
  1. Rescale shorter side to 256
  2. Center crop to 256×256×3
  3. Subtract per-pixel mean: x = x_raw - μ_train
  4. (Training) Random crop 224×224 + random horizontal flip
     (Test) 5 crops (4 corners + center) × 2 (flips) = 10 patches

=== Convolutional Feature Extraction ===
  For each layer l in [Conv1, Conv2, ..., Conv5]:
    z_l = W_l * x_{l-1} + b_l        (convolution + bias)
    a_l = max(0, z_l)                 (ReLU activation)
    
    If l in {1, 2}:                   (LRN after Conv1, Conv2 only)
      b^i_{x,y} = a^i_{x,y} / (2 + 10^{-4} Σ_{j} (a^j_{x,y})^2)^{0.75}
                                      (sum over 5 adjacent kernels)
    
    If l in {1, 2, 5}:               (MaxPool after Conv1, Conv2, Conv5)
      p_l = MaxPool(a_l, size=3, stride=2)  (overlapping pooling)

=== Fully-Connected Classification ===
  Flatten: 6×6×256 → 9,216-dim vector
  
  For each layer l in [FC6, FC7]:
    z_l = W_l · x_{l-1} + b_l
    a_l = max(0, z_l)                 (ReLU)
    (Training) h_l = a_l ⊙ mask_l    (Dropout: mask_l ~ Bernoulli(0.5))
    (Test)     h_l = 0.5 · a_l       (Scale by keep probability)
  
  FC8 (output):
    z_8 = W_8 · h_7 + b_8            (1000-dim logits)
    P(y=k|x) = exp(z_k) / Σ_j exp(z_j)  (softmax)

=== Loss ===
  L = -log P(y = correct | x) + 0.0005 · Σ ||W||²  (cross-entropy + L2)

=== SGD with Momentum ===
  v_{t+1} = 0.9·v_t - 0.0005·ε·w_t - ε·∇L
  w_{t+1} = w_t + v_{t+1}
  
  ε: 0.01 → 0.001 → 0.0001 → 0.00001 (when val error plateaus)
```

### 각 기법의 기여도 정리 / Contribution of Each Technique

| 기법 / Technique | Top-1 감소 / Reduction | Top-5 감소 / Reduction | 적용 위치 / Applied to |
|---|---|---|---|
| ReLU (vs. tanh) | 6× faster convergence | — | 모든 층 / All layers |
| 2-GPU (vs. 1-GPU) | 1.7% | 1.2% | 전체 아키텍처 / Architecture |
| LRN | 1.4% | 1.2% | Conv1, Conv2 이후 / After Conv1, Conv2 |
| Overlapping Pooling | 0.4% | 0.3% | Conv1, Conv2, Conv5 이후 / After Conv1, Conv2, Conv5 |
| Data Aug (crop+flip) | 매우 큼 (필수) / Very large (essential) | — | 학습 시 / Training |
| PCA Color Aug | >1% | — | 학습 시 / Training |
| Dropout | 매우 큼 (필수) / Very large (essential) | — | FC6, FC7 |
| 10-patch test avg | 1.5% | 1.3% | 추론 시 / Inference |

---

## 역사 속의 논문 / Paper in the Arc of History

```
1989  LeCun et al. — CNN for Zip Codes (#7)
       │  최초의 성공적 CNN, 하지만 소규모 데이터
       ▼
1998  LeCun et al. — LeNet-5 (#10)
       │  CNN 아키텍처의 성숙, MNIST에서 검증
       ▼
2001  Breiman — Random Forests (#11)
       │  "얕은" ML이 실용적으로 우세한 시기
       ▼
2006  Hinton et al. — Deep Belief Nets (#12)
       │  딥러닝 부활의 신호탄, RBM 사전학습
       ▼
2009  Deng et al. — ImageNet
       │  1,400만+ 이미지의 대규모 데이터셋 공개
       ▼
2010  Nair & Hinton — ReLU in RBMs
       │  ReLU 개념의 소개 (restricted Boltzmann machines에서)
       ▼
★ 2012  KRIZHEVSKY, SUTSKEVER & HINTON — ALEXNET ★
       │  ReLU + Dropout + GPU + Data Augmentation
       │  ILSVRC 2012: top-5 error 15.3% (2위 26.2%)
       │  사전학습 없이 순수 지도 학습으로 대규모 deep CNN 성공
       │  → hand-crafted feature 시대의 종말
       ▼
2013  Zeiler & Fergus — ZFNet
       │  AlexNet 분석 & 개선 (ILSVRC 2013 우승)
       ▼
2014  Simonyan & Zisserman — VGGNet
       │  "더 깊으면 더 좋다" (16–19층, 3×3 필터만)
       │  LRN이 불필요함을 발견
       ▼
2014  Szegedy et al. — GoogLeNet/Inception
       │  Inception module로 효율적 깊은 네트워크 (22층)
       ▼
2015  Ioffe & Szegedy — Batch Normalization
       │  LRN을 완전히 대체, 학습 안정화
       ▼
2015  He et al. — ResNet (#19)
       │  152층! Skip connection으로 초심층 학습
       │  → "깊이의 한계"를 또 한 번 돌파
       ▼
2017+ Transformer, Vision Transformer (ViT)
       │  CNN 아키텍처 자체에 대한 도전
```

---

## 다른 논문과의 연결 / Connections to Other Papers

| 논문 / Paper | 관계 / Relationship |
|---|---|
| #7 LeCun et al. (1989) — CNN | AlexNet의 직접적 선조. 같은 CNN 패러다임이지만 32×32 → 224×224으로 입력 크기가 49배, 파라미터가 수천 배 증가. GPU의 병렬 연산이 이 스케일 업을 가능하게 함 / Direct ancestor. Same CNN paradigm but 49x larger input (32×32 → 224×224), thousands of times more parameters. GPU parallelism enabled this scale-up |
| #10 LeCun et al. (1998) — LeNet-5 | LeNet-5의 구조(Conv → Pool → FC)를 대폭 확장. Average pooling → Max pooling, sigmoid → ReLU로 교체. "깊이 + 크기의 중요성"을 실증 / Massive expansion of LeNet-5's structure. Replaced average → max pooling, sigmoid → ReLU. Demonstrated "importance of depth + scale" |
| #11 Breiman (2001) — Random Forest | AlexNet 이전까지 실용적으로 우세했던 "얕은" ML의 대표. ILSVRC에서 SIFT + Fisher Vector(shallow pipeline)가 AlexNet에 압도당하며, deep vs. shallow 논쟁이 사실상 종결 / Representative of "shallow" ML dominant before AlexNet. SIFT + Fisher Vectors crushed by AlexNet in ILSVRC, effectively ending deep vs. shallow debate |
| #12 Hinton et al. (2006) — DBN | Hinton이 지도교수. DBN의 사전학습이 deep network 학습 가능성을 열었지만, AlexNet은 사전학습 없이 ReLU + Dropout으로 같은 목표를 달성. 학생(Krizhevsky)이 스승의 이전 연구를 초월 / Hinton as advisor. DBN pre-training opened deep network possibility, but AlexNet achieved the same goal without pre-training via ReLU + Dropout. Student surpassed mentor's previous work |
| #15 Kingma & Welling (2013) — VAE | AlexNet이 열어준 GPU 딥러닝 생태계 위에서 생성 모델의 새로운 패러다임 등장. AlexNet의 encoder 구조(Conv → FC)는 VAE encoder의 전형이 됨 / New generative model paradigm emerged on the GPU deep learning ecosystem AlexNet created. AlexNet's encoder structure became the VAE encoder archetype |
| #16 Goodfellow et al. (2014) — GAN | AlexNet 스타일의 discriminator/generator 네트워크 사용. AlexNet이 없었다면 GAN의 실용적 구현이 어려웠을 것 / Uses AlexNet-style discriminator/generator networks. Without AlexNet, practical GAN implementation would have been difficult |
| #17 Bahdanau et al. (2014) — Attention | AlexNet이 시각에서 보여준 "end-to-end 학습"의 성공이 NLP에서도 같은 접근을 촉진. 두 분야 모두 hand-crafted feature에서 learned representation으로 전환 / AlexNet's "end-to-end learning" success in vision catalyzed the same approach in NLP. Both fields transitioned from hand-crafted to learned representations |
| #19 He et al. (2015) — ResNet | AlexNet이 "깊이가 중요하다"를 증명 → VGGNet(19층) → ResNet(152층)으로 이어지는 "더 깊은 네트워크" 경쟁의 기원. ResNet의 skip connection은 AlexNet이 제기한 "깊이의 한계"를 돌파 / AlexNet proved "depth matters" → VGGNet (19 layers) → ResNet (152 layers). Origin of the "deeper network" race. ResNet's skip connections broke through the "depth limits" AlexNet raised |

---

## 참고문헌 / References

- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). "ImageNet Classification with Deep Convolutional Neural Networks." *Advances in Neural Information Processing Systems 25 (NIPS 2012)*, pp. 1097–1105.
- Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., & Fei-Fei, L. (2009). "ImageNet: A Large-Scale Hierarchical Image Database." *CVPR 2009*.
- Nair, V. & Hinton, G. E. (2010). "Rectified linear units improve restricted Boltzmann machines." *Proc. 27th ICML*.
- Hinton, G. E., Srivastava, N., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. R. (2012). "Improving neural networks by preventing co-adaptation of feature detectors." *arXiv:1207.0580*.
- Sánchez, J. & Perronnin, F. (2011). "High-dimensional signature compression for large-scale image classification." *CVPR 2011*, pp. 1665–1672.
- Simard, P. Y., Steinkraus, D., & Platt, J. C. (2003). "Best practices for convolutional neural networks applied to visual document analysis." *ICDAR 2003*, Vol. 2, pp. 958–962.
- Simonyan, K. & Zisserman, A. (2014). "Very deep convolutional networks for large-scale image recognition." *arXiv:1409.1556*.
- He, K., Zhang, X., Ren, S., & Sun, J. (2015). "Deep residual learning for image recognition." *arXiv:1512.03385*.
- Ioffe, S. & Szegedy, S. (2015). "Batch Normalization: Accelerating deep network training by reducing internal covariate shift." *ICML 2015*.
