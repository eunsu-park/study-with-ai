# Pre-reading Briefing: ImageNet Classification with Deep Convolutional Neural Networks (Krizhevsky, Sutskever & Hinton, 2012)
# 사전 읽기 브리핑: 심층 합성곱 신경망을 이용한 ImageNet 분류 (Krizhevsky, Sutskever & Hinton, 2012)

---

## 핵심 기여 / Core Contribution

이 논문은 **딥러닝이 대규모 시각 인식 문제에서 압도적으로 우수하다**는 것을 실증적으로 증명한 기념비적 논문입니다. 2012년 ImageNet Large Scale Visual Recognition Challenge (ILSVRC)에서 top-5 error율 **15.3%**를 달성하여, 2위(26.2%)를 **10.8%p** 차이로 압도했습니다. 이전까지 컴퓨터 비전은 hand-crafted feature(SIFT, HOG 등)에 의존했지만, AlexNet은 **raw pixel에서 직접 특징을 학습**하는 end-to-end 방식이 가능하다는 것을 보여주었습니다. 핵심 기술적 기여는: ① **ReLU** 활성화 함수로 학습 속도 대폭 향상, ② **2-GPU 병렬 학습** 아키텍처, ③ **Dropout** 정규화로 overfitting 방지, ④ **Data augmentation**과 **Local Response Normalization** 적용입니다. 이 논문 이후 딥러닝은 컴퓨터 비전의 지배적 패러다임이 되었고, AI 연구 전체에 "더 크고 더 깊은 네트워크"를 향한 대전환이 시작되었습니다.

This paper is the landmark that **empirically proved deep learning's overwhelming superiority in large-scale visual recognition**. At the 2012 ImageNet Large Scale Visual Recognition Challenge (ILSVRC), it achieved a top-5 error rate of **15.3%**, crushing the runner-up (26.2%) by **10.8 percentage points**. Until then, computer vision relied on hand-crafted features (SIFT, HOG, etc.), but AlexNet demonstrated that **learning features directly from raw pixels** via an end-to-end approach was not just viable but dramatically superior. Key technical contributions include: ① **ReLU** activation for massively faster training, ② **2-GPU parallel training** architecture, ③ **Dropout** regularization to combat overfitting, ④ **Data augmentation** and **Local Response Normalization**. After this paper, deep learning became the dominant paradigm in computer vision, and AI research pivoted toward "bigger and deeper networks."

---

## 역사적 맥락 / Historical Context

| 연도 / Year | 업적 / Milestone | 관련성 / Relevance |
|---|---|---|
| 1989 | LeCun et al. — CNN for Zip Codes (#7) | 최초의 성공적 CNN, 하지만 소규모 데이터에서만 / First successful CNN, but only on small-scale data |
| 1998 | LeCun et al. — LeNet-5 (#10) | CNN 아키텍처의 성숙, MNIST에서 검증 / Mature CNN architecture, validated on MNIST |
| 2001 | Breiman — Random Forests (#11) | 비전 포함 "얕은" ML이 주류 / "Shallow" ML (including vision) was mainstream |
| 2006 | Hinton et al. — Deep Belief Nets (#12) | 딥러닝 부활의 신호탄, 하지만 비전에서 실용적 우위 미증명 / Deep learning revival signal, but practical vision superiority unproven |
| 2009 | Deng et al. — ImageNet | 1,400만+ 이미지, 22,000+ 클래스의 대규모 데이터셋 공개 / 14M+ images, 22K+ classes large-scale dataset released |
| 2010 | ILSVRC 시작 | ImageNet subset (1,000 클래스)으로 연례 벤치마크 시작 / Annual benchmark started with 1,000-class ImageNet subset |
| 2010–2011 | 기존 우승자들 (hand-crafted features) | SIFT + Fisher Vector 등으로 top-5 error ~26–28% / Using SIFT + Fisher Vector etc., top-5 error ~26–28% |
| **2012** | **Krizhevsky, Sutskever & Hinton — AlexNet** | **이 논문: top-5 error 15.3%로 대회 역사를 바꿈** / **This paper: changed competition history with 15.3% top-5 error** |
| 2013 | Zeiler & Fergus — ZFNet | AlexNet 개선 (ILSVRC 2013 우승) / AlexNet refinement (won ILSVRC 2013) |
| 2014 | Simonyan & Zisserman — VGGNet | "더 깊으면 더 좋다"를 증명 (16-19층) / Proved "deeper is better" (16-19 layers) |
| 2014 | Szegedy et al. — GoogLeNet/Inception | Inception module로 효율적 깊은 네트워크 / Efficient deep network with Inception modules |
| 2015 | He et al. — ResNet (#19) | 152층! Skip connection으로 초심층 학습 가능 / 152 layers! Skip connections enable ultra-deep training |

**2012년은 AI 역사의 분수령**: AlexNet의 압도적 승리는 학계와 산업계 모두에 충격을 주었습니다. "hand-crafted features vs. learned features" 논쟁이 사실상 종결되었고, NVIDIA GPU 수요가 폭발적으로 증가했으며, Google이 Hinton의 연구실을 인수하는 등 AI 산업 생태계 자체가 재편되었습니다.

**2012 was a watershed moment in AI history**: AlexNet's overwhelming victory shocked both academia and industry. The "hand-crafted vs. learned features" debate was effectively settled, NVIDIA GPU demand exploded, Google acquired Hinton's lab, and the entire AI industry ecosystem was reshaped.

---

## 필요한 배경 지식 / Prerequisites

### 1. Convolutional Neural Networks (CNN) — 합성곱 신경망

**Papers #7 (LeCun 1989)과 #10 (LeCun 1998)에서 학습한 CNN의 핵심 개념을 복습합니다.**

**Review the core CNN concepts learned from Papers #7 (LeCun 1989) and #10 (LeCun 1998).**

**합성곱 연산 / Convolution Operation:**

입력 이미지 $\mathbf{I}$에 필터(커널) $\mathbf{K}$를 적용합니다:

Apply a filter (kernel) $\mathbf{K}$ to input image $\mathbf{I}$:

$$(\mathbf{I} * \mathbf{K})(i,j) = \sum_m \sum_n \mathbf{I}(i+m, j+n) \cdot \mathbf{K}(m,n)$$

- **Feature map**: 하나의 필터를 이미지 전체에 슬라이딩하여 만든 출력 / Output from sliding one filter across the entire image
- **다중 필터**: 각 층에서 여러 필터를 사용하여 다양한 특징을 추출 / Multiple filters per layer extract diverse features
- **Stride**: 필터가 이동하는 간격 (AlexNet은 첫 층에서 stride 4 사용) / Step size of filter movement (AlexNet uses stride 4 in first layer)

**Pooling (서브샘플링):**

$$\text{MaxPool}(x_{i,j}) = \max_{(m,n) \in \mathcal{R}} x_{i+m, j+n}$$

- AlexNet은 **overlapping max pooling**을 사용합니다 (pool size 3×3, stride 2) — LeNet-5의 average pooling과 다릅니다.
- AlexNet uses **overlapping max pooling** (pool size 3×3, stride 2) — different from LeNet-5's average pooling.

### 2. ReLU (Rectified Linear Unit) — 정류 선형 유닛

AlexNet의 가장 중요한 기술적 기여 중 하나입니다:

One of AlexNet's most important technical contributions:

$$f(x) = \max(0, x)$$

**왜 중요한가 / Why it matters:**

| 활성화 함수 / Activation | 수식 / Formula | 문제점 / Problem |
|---|---|---|
| Sigmoid | $\sigma(x) = \frac{1}{1+e^{-x}}$ | Saturation → vanishing gradient (큰 $|x|$에서 기울기 ≈ 0) |
| Tanh | $\tanh(x)$ | 같은 saturation 문제 / Same saturation problem |
| **ReLU** | $\max(0, x)$ | **양수 영역에서 기울기 항상 1** → gradient가 사라지지 않음 / **Gradient always 1 in positive region** → no vanishing gradient |

Krizhevsky et al.은 ReLU가 sigmoid/tanh보다 **6배 빠르게** 동일한 정확도에 도달한다고 보고합니다. 이것은 대규모 데이터셋에서의 학습을 현실적으로 가능하게 했습니다.

Krizhevsky et al. report that ReLU reaches the same accuracy **6x faster** than sigmoid/tanh. This made training on large-scale datasets practically feasible.

### 3. GPU Computing / GPU 컴퓨팅

2012년 당시 GPU는 아직 딥러닝의 주류 도구가 아니었습니다. AlexNet은 **GTX 580 GPU 2개**에서 학습되었습니다.

In 2012, GPUs were not yet mainstream tools for deep learning. AlexNet was trained on **2 GTX 580 GPUs**.

- **GPU의 장점**: 행렬 연산의 대규모 병렬 처리 (수천 개의 CUDA 코어) / **GPU advantage**: massive parallelism for matrix operations (thousands of CUDA cores)
- **메모리 제약**: GTX 580은 VRAM 3GB — 이 제약이 2-GPU 분할 아키텍처를 강제함 / **Memory constraint**: GTX 580 has 3GB VRAM — this constraint forced the 2-GPU split architecture
- **학습 시간**: 5–6일 소요 (120만 이미지, 90 epochs) / **Training time**: 5–6 days (1.2M images, 90 epochs)

### 4. Overfitting과 Regularization — 과적합과 정규화

6,000만 개의 파라미터를 가진 AlexNet은 심각한 overfitting 위험이 있었습니다. 논문에서 사용한 기법들:

With 60 million parameters, AlexNet faced serious overfitting risk. Techniques used in the paper:

**Dropout** (Hinton et al., 2012):

학습 시 각 뉴런을 확률 $p$로 무작위 비활성화합니다:

During training, randomly deactivate each neuron with probability $p$:

$$\tilde{h}_i = \begin{cases} 0 & \text{with probability } p \\ h_i / (1-p) & \text{with probability } 1-p \end{cases}$$

- 효과: 뉴런들이 서로에 대한 "co-adaptation"을 방지 → 각 뉴런이 독립적으로 유용한 특징을 학습
- Effect: prevents "co-adaptation" between neurons → each neuron learns independently useful features
- AlexNet은 fully-connected 층에서 $p = 0.5$ 사용 / AlexNet uses $p = 0.5$ in fully-connected layers

**Data Augmentation / 데이터 증강:**

- 원본 256×256 이미지에서 224×224 패치를 무작위 추출 + 좌우 반전 → 학습 데이터 2048배 증가
- Random 224×224 crops from 256×256 images + horizontal flips → 2048x training data increase
- RGB 채널에 PCA 기반 색상 변화 추가
- PCA-based color augmentation on RGB channels

### 5. Softmax와 Cross-Entropy Loss

**1,000개 클래스 분류를 위한 출력층:**

**Output layer for 1,000-class classification:**

$$P(y = k | \mathbf{x}) = \frac{e^{z_k}}{\sum_{j=1}^{1000} e^{z_j}}$$

여기서 $z_k$는 클래스 $k$에 대한 네트워크 출력(logit)입니다. Cross-entropy loss:

Where $z_k$ is the network output (logit) for class $k$. Cross-entropy loss:

$$L = -\sum_{k=1}^{1000} y_k \log P(y = k | \mathbf{x})$$

---

## 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **ImageNet** | 1,400만+ 이미지, 22,000+ 카테고리의 대규모 시각 데이터셋. ILSVRC는 이 중 1,000 클래스 subset 사용 / 14M+ image, 22K+ category large-scale visual dataset. ILSVRC uses 1,000-class subset |
| **ILSVRC** | ImageNet Large Scale Visual Recognition Challenge — 연례 이미지 인식 대회 (2010–2017) / Annual image recognition competition (2010–2017) |
| **Top-5 Error** | 모델이 예측한 상위 5개 클래스에 정답이 없는 비율. 1,000개 클래스에서 Top-1은 너무 가혹하므로 Top-5를 주요 지표로 사용 / Fraction where ground truth is not in the model's top 5 predictions. With 1,000 classes, Top-1 is too strict, so Top-5 is the primary metric |
| **ReLU** | Rectified Linear Unit: $\max(0,x)$. Saturating 비선형 함수를 대체하여 학습 속도를 극적으로 향상 / Replaces saturating nonlinearities to dramatically improve training speed |
| **Dropout** | 학습 시 무작위로 뉴런을 비활성화하는 정규화 기법. "매번 다른 네트워크를 학습"하는 효과 / Regularization by randomly deactivating neurons during training. Effect of "training a different network each time" |
| **Local Response Normalization (LRN)** | 인접 feature map 간 경쟁을 모사하는 정규화. 생물학적 "lateral inhibition"에서 영감 / Normalization that simulates competition between adjacent feature maps. Inspired by biological "lateral inhibition" |
| **Overlapping Pooling** | Pool 영역이 겹치는 pooling (size > stride). 전통적 non-overlapping pooling보다 overfitting 감소 / Pooling where pool regions overlap (size > stride). Reduces overfitting compared to traditional non-overlapping pooling |
| **Data Augmentation** | 원본 이미지를 변형(크롭, 반전, 색상 변화)하여 학습 데이터를 인위적으로 증가시키는 기법 / Artificially increasing training data by transforming originals (crop, flip, color shift) |
| **Feature Map** | 합성곱 층의 출력. 하나의 필터가 입력에서 특정 패턴을 감지한 결과 / Output of a convolution layer. Result of one filter detecting a specific pattern in the input |
| **Stride** | 합성곱 또는 pooling 시 필터가 이동하는 간격. 큰 stride → 출력 크기 감소 / Step size for convolution or pooling. Larger stride → smaller output |
| **SGD with Momentum** | 미니배치 경사하강법에 이전 업데이트 방향의 관성을 추가. 학습 안정성과 수렴 속도 향상 / Mini-batch gradient descent with inertia from previous update direction. Improves training stability and convergence |

---

## 수식 미리보기 / Equations Preview

### 1. AlexNet 아키텍처 / Architecture

```
입력 / Input: 224 × 224 × 3 (RGB 이미지 / RGB image)

Conv1: 96 filters, 11×11, stride 4  → 55 × 55 × 96
  + ReLU → LRN → MaxPool (3×3, stride 2) → 27 × 27 × 96

Conv2: 256 filters, 5×5, pad 2      → 27 × 27 × 256
  + ReLU → LRN → MaxPool (3×3, stride 2) → 13 × 13 × 256

Conv3: 384 filters, 3×3, pad 1      → 13 × 13 × 384
  + ReLU

Conv4: 384 filters, 3×3, pad 1      → 13 × 13 × 384
  + ReLU

Conv5: 256 filters, 3×3, pad 1      → 13 × 13 × 256
  + ReLU → MaxPool (3×3, stride 2) → 6 × 6 × 256

FC6: 4096 units + ReLU + Dropout(0.5)
FC7: 4096 units + ReLU + Dropout(0.5)
FC8: 1000 units + Softmax
```

총 파라미터 / Total parameters: **~60 million** (대부분 FC 층에 집중 / mostly in FC layers)

### 2. Local Response Normalization (LRN)

$$b^i_{x,y} = \frac{a^i_{x,y}}{\left(k + \alpha \sum_{j=\max(0,i-n/2)}^{\min(N-1,i+n/2)} (a^j_{x,y})^2\right)^\beta}$$

- $a^i_{x,y}$: 위치 $(x,y)$에서 $i$번째 커널의 활성화 / Activation of $i$-th kernel at position $(x,y)$
- $N$: 해당 층의 총 커널 수 / Total number of kernels in the layer
- $n$: 인접 커널의 범위 (논문에서 $n=5$) / Range of adjacent kernels (paper uses $n=5$)
- $k=2, \alpha=10^{-4}, \beta=0.75$: 하이퍼파라미터 (validation set으로 결정) / Hyperparameters (determined on validation set)

**직관**: 같은 위치에서 인접한 feature map들의 활성화가 클수록, 현재 feature map의 활성화가 억제됩니다. 생물학의 lateral inhibition과 유사 — 강한 반응이 주변의 약한 반응을 억제합니다.

**Intuition**: The larger the activations in adjacent feature maps at the same position, the more the current feature map's activation is suppressed. Similar to biological lateral inhibition — strong responses suppress weaker neighboring responses.

> **참고 / Note**: LRN은 이후 연구에서 효과가 미미한 것으로 밝혀져 현대 네트워크에서는 사용되지 않습니다 (Batch Normalization으로 대체). 하지만 2012년 시점에서는 중요한 기법이었습니다.
>
> LRN was later found to have minimal effect and is not used in modern networks (replaced by Batch Normalization). However, it was an important technique in 2012.

### 3. PCA Color Augmentation

각 RGB 이미지 픽셀에 다음을 추가합니다:

Add the following to each RGB pixel:

$$[\mathbf{p}_1, \mathbf{p}_2, \mathbf{p}_3][\alpha_1 \lambda_1, \alpha_2 \lambda_2, \alpha_3 \lambda_3]^T$$

- $\mathbf{p}_i, \lambda_i$: 전체 학습 이미지 RGB 픽셀 값의 3×3 공분산 행렬의 고유벡터, 고유값 / Eigenvectors and eigenvalues of the 3×3 covariance matrix of RGB pixel values across the training set
- $\alpha_i \sim \mathcal{N}(0, 0.1)$: 이미지마다 랜덤 샘플 / Randomly sampled per image

**직관**: 자연 이미지의 조명 변화는 RGB 공간에서 주성분 방향을 따릅니다. 이 방향으로 약간의 무작위 변화를 주면, 네트워크가 조명 변화에 강건해집니다.

**Intuition**: Illumination changes in natural images follow principal component directions in RGB space. Adding small random perturbations along these directions makes the network robust to illumination changes.

### 4. 학습 규칙 / Training Rule

**SGD with Momentum:**

$$v_{t+1} = 0.9 \cdot v_t - 0.0005 \cdot \epsilon \cdot w_t - \epsilon \cdot \left\langle \frac{\partial L}{\partial w} \bigg|_{w_t} \right\rangle_{D_i}$$

$$w_{t+1} = w_t + v_{t+1}$$

- $\epsilon$: learning rate (초기 0.01, 3번 1/10로 감소) / learning rate (initial 0.01, reduced by 1/10 three times)
- $0.9$: momentum 계수 / momentum coefficient
- $0.0005$: weight decay (L2 정규화) — 저자들은 이것이 정규화뿐 아니라 학습 자체에도 도움이 된다고 강조 / weight decay (L2 regularization) — authors emphasize it helps training itself, not just regularization
- $D_i$: 128개 이미지의 미니배치 / mini-batch of 128 images

**초기화 / Initialization:**
- 모든 층의 가중치: $\mathcal{N}(0, 0.01)$ / All layer weights: $\mathcal{N}(0, 0.01)$
- Conv2, Conv4, Conv5 및 FC 층의 bias: 1 (ReLU에 양수 입력을 제공하여 초기 학습 가속) / Biases in Conv2, Conv4, Conv5 and FC layers: 1 (provides positive inputs to ReLU, accelerating early learning)
- 나머지 층의 bias: 0 / Remaining layer biases: 0

---

## 논문을 읽을 때 주목할 점 / What to Watch For While Reading

1. **Figure 3 (학습된 필터 시각화)**: 첫 번째 합성곱 층이 학습한 96개 필터를 살펴보세요. GPU 1은 색상에 무관한 패턴(edge, frequency)을, GPU 2는 색상에 특화된 패턴을 학습합니다. 이 "자연스러운 분업"이 2-GPU 아키텍처의 흥미로운 부산물입니다.

   **Figure 3 (learned filter visualization)**: Look at the 96 filters learned by the first convolutional layer. GPU 1 learns color-agnostic patterns (edges, frequencies), GPU 2 learns color-specific patterns. This "natural division of labor" is a fascinating byproduct of the 2-GPU architecture.

2. **Table 1과 Table 2 (ablation study)**: 각 기법의 개별적 기여를 분석한 결과입니다. 어떤 기법이 가장 큰 차이를 만드는지 주목하세요.

   **Tables 1 & 2 (ablation study)**: Results analyzing each technique's individual contribution. Note which techniques make the biggest difference.

3. **Section 6.1 (정성적 분석)**: 네트워크의 마지막 hidden layer(4096차원)에서 이미지 간 유사도를 계산한 결과입니다. 의미적으로 유사한 이미지가 가까운 벡터 표현을 갖는다는 것은, 네트워크가 진정한 "이해"에 가까운 표현을 학습했음을 시사합니다.

   **Section 6.1 (qualitative analysis)**: Similarity computed between images in the last hidden layer (4096-dim). Semantically similar images having close vector representations suggests the network learned representations approaching genuine "understanding."

4. **깊이(depth)의 중요성**: 저자들은 어떤 합성곱 층을 하나라도 제거하면 성능이 약 2% 하락한다고 보고합니다. 이것은 "깊이가 중요하다"는 메시지를 강력히 전달합니다.

   **Importance of depth**: The authors report that removing any single convolutional layer degrades performance by about 2%. This powerfully conveys the message that "depth matters."

---

## 읽기 전 확인 질문 / Pre-reading Check Questions

이 논문을 읽기 전에 다음 질문에 답할 수 있는지 확인해 보세요:

Before reading this paper, check if you can answer the following:

1. 합성곱 연산에서 stride가 4라면, 224×224 입력에 11×11 필터를 적용했을 때 출력 크기는? / If stride is 4, what is the output size when applying an 11×11 filter to a 224×224 input?
   - 힌트: $\lfloor (224 - 11)/4 \rfloor + 1 = 54 + 1 = 55$ → 55×55

2. ReLU가 sigmoid보다 학습에 유리한 이유를 gradient 관점에서 설명할 수 있는가? / Can you explain why ReLU is better for training than sigmoid from a gradient perspective?
   - 힌트: sigmoid의 gradient는 최대 0.25이고 양 끝에서 0에 수렴 / sigmoid's gradient is at most 0.25 and approaches 0 at both ends

3. Dropout이 앙상블 학습과 어떤 관계가 있는지 설명할 수 있는가? / Can you explain how Dropout relates to ensemble learning?
   - 힌트: 학습 시 매번 다른 "thinned" 네트워크를 사용 → 지수적으로 많은 네트워크의 앙상블 근사 / Each training step uses a different "thinned" network → approximate ensemble of exponentially many networks

---

## References / 참고문헌

- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). "ImageNet Classification with Deep Convolutional Neural Networks." *Advances in Neural Information Processing Systems 25 (NIPS 2012)*, pp. 1097–1105.
- Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., & Fei-Fei, L. (2009). "ImageNet: A Large-Scale Hierarchical Image Database." *CVPR 2009*.
- LeCun, Y., Boser, B., Denker, J. S., et al. (1989). "Backpropagation Applied to Handwritten Zip Code Recognition." *Neural Computation*, 1(4), 541–551.
- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). "Gradient-Based Learning Applied to Document Recognition." *Proceedings of the IEEE*, 86(11), 2278–2324.
- Hinton, G. E., Srivastava, N., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. R. (2012). "Improving neural networks by preventing co-adaptation of feature detectors." *arXiv:1207.0580*.

---

## Q&A

### Q1: GPU 분업은 의도한 것인가, 자동으로 된 것인가? / Was the GPU specialization intentional or emergent?

**자동으로 나타난 현상(emergent behavior)입니다.** 저자들은 GPU 메모리 부족(GTX 580의 3GB VRAM) 때문에 어쩔 수 없이 네트워크를 반으로 나눠 GPU 2개에 배치했을 뿐입니다. 각 GPU가 "어떤 종류의 필터를 학습해라"고 지시한 적이 없습니다.

**This is emergent behavior.** The authors split the network across 2 GPUs only because of memory constraints (GTX 580's 3GB VRAM). They never instructed each GPU to learn specific types of filters.

그런데 학습이 끝나고 보니:
- **GPU 1**: 색상과 무관한 패턴 (edge, frequency, orientation)
- **GPU 2**: 색상에 특화된 패턴 (color blob, color edge)

After training:
- **GPU 1**: color-agnostic patterns (edges, frequencies, orientations)
- **GPU 2**: color-specific patterns (color blobs, color edges)

저자들은 이것이 **랜덤 초기화에 무관하게** 매번 재현된다고 보고합니다 ("independent of the particular random weight initialization").

The authors report this reproduces consistently "independent of the particular random weight initialization."

**왜 이런 일이 일어나는가**: 두 GPU는 Conv1에서 완전히 독립적으로 학습하고, Conv3에서만 서로의 feature map을 모두 받습니다. 이 제한된 교차 연결 구조에서, gradient descent가 자연스럽게 "서로 다른 것을 학습하는 게 loss를 더 줄인다"는 방향으로 수렴합니다. 마치 두 사람에게 제한된 소통만 허용하면, 자연스럽게 역할을 나누게 되는 것과 비슷합니다.

**Why this happens**: The two GPUs learn completely independently in Conv1, and only receive each other's feature maps in Conv3. With this limited cross-connection structure, gradient descent naturally converges toward "learning different things reduces loss more." Similar to how two people with limited communication naturally divide roles.

---

### Q2: Softmax와 Cross-Entropy 상세 설명 / Detailed explanation of Softmax and Cross-Entropy

#### Softmax — "확률로 변환하기" / "Converting to probabilities"

네트워크의 마지막 층(FC8)은 1,000개의 raw 숫자(logit)를 출력합니다:

The last layer (FC8) outputs 1,000 raw numbers (logits):

$$z = [2.1, \; 0.3, \; -1.5, \; 5.2, \; \ldots] \quad \text{(1,000개 / 1,000 values)}$$

이것을 **확률 분포**로 바꾸는 함수가 softmax입니다:

Softmax converts these into a **probability distribution**:

$$P(y = k \mid \mathbf{x}) = \frac{e^{z_k}}{\sum_{j=1}^{1000} e^{z_j}}$$

**단계별 예시 (3개 클래스로 단순화) / Step-by-step example (simplified to 3 classes):**

| 단계 / Step | 연산 / Operation | 예시 / Example |
|---|---|---|
| ① Logit | 네트워크 출력 / Network output | $z = [2.0, \; 1.0, \; 0.1]$ |
| ② Exponentiate | $e^{z_k}$ 계산 / Compute $e^{z_k}$ | $[7.39, \; 2.72, \; 1.11]$ |
| ③ Normalize | 합으로 나눔 / Divide by sum | 합=11.22 → $[0.659, \; 0.242, \; 0.099]$ |

**핵심 성질 / Key properties:**
- 모든 출력이 $(0, 1)$ 범위 / All outputs in $(0, 1)$
- 합이 정확히 1 / Sum is exactly 1
- **가장 큰 logit이 지수적으로 강조됨** → "winner-take-most" 효과 / **Largest logit is exponentially amplified** → "winner-take-most" effect
- 미분 가능 → gradient descent 학습 가능 / Differentiable → trainable with gradient descent

#### Cross-Entropy Loss — "정답과 얼마나 다른가" / "How different from the correct answer"

정답 레이블은 one-hot vector입니다. 예: 클래스 2가 정답이면:

Ground truth label is a one-hot vector. E.g., if class 2 is correct:

$$\mathbf{y} = [0, \; 0, \; 1, \; 0, \; \ldots, \; 0]$$

$$L = -\sum_{k=1}^{1000} y_k \log P(y=k \mid \mathbf{x})$$

$y_k = 0$인 항은 모두 사라지므로 / Since all terms where $y_k = 0$ vanish:

$$L = -\log P(y = \text{correct class} \mid \mathbf{x})$$

| 정답 확률 / Correct class prob. | $-\log(P)$ = Loss | 의미 / Meaning |
|---|---|---|
| 0.99 | 0.01 | 거의 완벽 → loss 아주 작음 / Nearly perfect → very small loss |
| 0.5 | 0.69 | 반반 → loss 중간 / 50-50 → moderate loss |
| 0.01 | 4.61 | 거의 틀림 → loss 매우 큼 / Nearly wrong → very large loss |
| → 0 | → ∞ | 완전히 틀림 → loss 폭발 / Completely wrong → loss explodes |

**왜 MSE가 아니라 cross-entropy인가? / Why cross-entropy instead of MSE?**

Softmax + cross-entropy의 조합은 매우 깔끔한 gradient를 만듭니다:

The softmax + cross-entropy combination produces a very clean gradient:

$$\frac{\partial L}{\partial z_k} = P(k) - y_k$$

예측이 틀리면 gradient가 크고, 맞으면 작습니다. MSE + softmax를 쓰면 sigmoid의 saturation과 비슷한 문제로 gradient가 작아져 학습이 느려집니다.

Wrong predictions → large gradient, correct predictions → small gradient. MSE + softmax suffers from saturation similar to sigmoid, causing small gradients and slow learning.

**정보 이론적 해석 / Information-theoretic interpretation:**

$$H(p, q) = -\sum_k p(k) \log q(k) = H(p) + D_{KL}(p \| q)$$

$H(p) = 0$ (one-hot)이므로, cross-entropy 최소화 = KL divergence 최소화 = "모델의 예측 분포를 정답 분포에 가깝게 만들어라."

Since $H(p) = 0$ (one-hot), minimizing cross-entropy = minimizing KL divergence = "make the model's predicted distribution close to the true distribution."

---

### Q3: PCA Color Augmentation 상세 설명 / Detailed explanation of PCA Color Augmentation

#### 문제 의식 / Motivation

같은 물체라도 조명에 따라 색이 달라집니다. 햇빛 아래 빨간 사과와 형광등 아래 빨간 사과는 RGB 값이 다르지만, 네트워크는 둘 다 "사과"로 인식해야 합니다.

The same object has different colors under different lighting. A red apple in sunlight and under fluorescent light have different RGB values, but the network must recognize both as "apple."

#### 핵심 관찰 / Key Observation

자연 이미지의 색상 변화는 RGB 공간에서 **특정 방향**을 따릅니다. ImageNet 전체 학습 이미지의 모든 픽셀 RGB 값(3차원 벡터)으로부터 **공분산 행렬**을 구합니다:

Color changes in natural images follow **specific directions** in RGB space. Compute the **covariance matrix** from all pixel RGB values across the entire ImageNet training set:

$$\mathbf{C} = \frac{1}{N} \sum_{i=1}^{N} (\mathbf{x}_i - \bar{\mathbf{x}})(\mathbf{x}_i - \bar{\mathbf{x}})^T \quad \text{(3×3 행렬 / 3×3 matrix)}$$

PCA를 적용하면 3개의 고유벡터와 고유값:

Applying PCA yields 3 eigenvectors and eigenvalues:

- $\mathbf{p}_1$ (제1 주성분 / 1st principal component): 색상 변화가 **가장 큰** 방향 → 대략 전체 밝기 / Direction of **largest** color variation → roughly overall brightness
- $\mathbf{p}_2$: 두 번째로 큰 변화 → 대략 blue-yellow 축 / 2nd largest → roughly blue-yellow axis
- $\mathbf{p}_3$: 가장 작은 변화 → 대략 red-green 축 / smallest → roughly red-green axis
- $\lambda_1 > \lambda_2 > \lambda_3$: 각 방향의 분산 크기 / variance along each direction

#### 증강 방법 / Augmentation Method

각 학습 이미지의 **모든 픽셀**에 다음 벡터를 더합니다:

Add the following vector to **every pixel** of each training image:

$$\Delta \mathbf{c} = [\mathbf{p}_1, \mathbf{p}_2, \mathbf{p}_3] \begin{bmatrix} \alpha_1 \lambda_1 \\ \alpha_2 \lambda_2 \\ \alpha_3 \lambda_3 \end{bmatrix}, \quad \alpha_i \sim \mathcal{N}(0, 0.1)$$

$\alpha$는 이미지마다 한 번씩 랜덤 샘플 (픽셀별이 아님).

$\alpha$ is sampled once per image (not per pixel).

#### 왜 효과적인가 / Why It Works

1. **주성분 방향 = 자연스러운 조명 변화 방향**: 밝기가 변하면 RGB가 동시에 비슷한 비율로 변합니다. PCA의 제1 주성분이 정확히 이 방향입니다. / **Principal component direction = natural illumination change direction**
2. **고유값에 비례하는 변화**: $\alpha_i \lambda_i$이므로, 자연적으로 큰 변화 방향으로는 많이, 작은 방향으로는 적게 변화 → 자연스러운 변화 / **Variation proportional to eigenvalue**: more variation along naturally variable directions
3. **이미지 전체에 동일한 Δc**: 같은 조명 변화는 모든 픽셀에 비슷하게 영향 / **Same Δc for entire image**: illumination changes affect all pixels similarly
4. **효과**: top-1 error **1% 이상** 감소 / **Effect**: top-1 error reduced by **over 1%**

---

### Q4: 아키텍처는 그대로 두고 단일 GPU에서 학습하면 분업이 일어나는가? / Does specialization occur if the same architecture runs on a single GPU?

**동일한 분업이 발생합니다.** 분업의 원인은 GPU라는 *물리적 장치*가 아니라, **교차 연결이 차단된 아키텍처 구조** 자체입니다. Conv2, Conv4, Conv5에서 채널을 두 그룹으로 나누고 교차 연결을 차단하는 구조(`groups=2`)를 유지하는 한, 단일 GPU에서도 같은 분업이 나타납니다.

**The same specialization occurs.** The cause is not the physical GPU device but the **architecture's restricted cross-connection structure** itself. As long as the grouped convolution structure (`groups=2`) blocking cross-connections in Conv2, Conv4, Conv5 is maintained, the same specialization appears on a single GPU.

```python
# 교차 연결 차단 = 그룹 합성곱 / Blocked cross-connection = grouped convolution
nn.Conv2d(96, 256, kernel_size=5, groups=2)   # Conv2: 그룹 분할 / grouped
nn.Conv2d(256, 384, kernel_size=3, groups=1)   # Conv3: 전체 교차 / full cross
nn.Conv2d(384, 384, kernel_size=3, groups=2)   # Conv4: 그룹 분할 / grouped
```

반대로, 2개 GPU를 사용하더라도 **모든 층에서 교차 연결을 허용**하면 (`groups=1`), 뚜렷한 이분법적 분업은 사라집니다. **핵심은 `groups=2`라는 연결 제약이지, 물리적 GPU 분리가 아닙니다.**

Conversely, even with 2 GPUs, if **cross-connections are allowed at every layer** (`groups=1`), the clear dichotomous specialization disappears. **The key is the `groups=2` connectivity constraint, not physical GPU separation.**

---

### Q5: GPU 간 연산량 불균형으로 병목이 발생할 가능성은? / Could workload imbalance between GPUs cause bottlenecks?

**연산량 차이는 발생하지 않습니다.** 합성곱의 연산량은 "어떤 값을 학습했느냐"가 아니라 "텐서의 크기"에 의해 결정되기 때문입니다. 한쪽이 edge를 학습하든 color blob을 학습하든, 11×11×3 필터 48개의 합성곱 비용은 동일합니다.

**No workload imbalance occurs.** Convolution cost is determined by "tensor dimensions," not "what values were learned." Whether one side learns edges or color blobs, the convolution cost for 48 filters of 11×11×3 is identical.

| 요소 / Factor | 연산량 차이? / Imbalance? | 이유 / Reason |
|---|---|---|
| 합성곱 / Convolution | 동일 / Equal | 필터 크기, 수, 입출력 크기가 동일 / Same filter size, count, I/O size |
| ReLU | 동일 / Equal | `max(0,x)` 비용은 값에 무관 / `max(0,x)` cost is value-independent |
| Backprop | 동일 / Equal | 같은 수의 파라미터 / Same number of parameters |

**다만 실제 병목이 발생하는 지점은 Conv3의 교차 통신입니다.** Conv3는 양쪽 GPU의 Conv2 출력을 모두 받으므로 GPU 간 데이터 전송이 필요합니다. 저자들이 "통신량이 계산량의 수용 가능한 비율이 되도록 정밀하게 조절했다"고 말한 것은 이 지점을 언급한 것입니다. 전송량 자체는 ~172KB로 PCIe 대역폭(~6GB/s)에서 무시할 수준이지만, 동기화 지연(synchronization latency)이 존재합니다.

**However, the actual bottleneck is the cross-communication at Conv3.** Conv3 receives all Conv2 outputs from both GPUs, requiring inter-GPU data transfer. The authors' statement about "precisely tuning the amount of communication" refers to this. Transfer volume (~172KB) is negligible on PCIe (~6GB/s), but synchronization latency exists.

---

### Q6: Softmax 출력은 진정한 확률인가? / Is softmax output true probability?

**엄밀하게는 "확률의 옷을 입은 점수(score dressed as probability)"입니다.**

**Strictly speaking, it's a "score dressed as probability."**

#### 만족하는 것 / What it satisfies:

Softmax 출력은 확률의 **Kolmogorov 공리**를 형식적으로 만족합니다:

Softmax output formally satisfies the **Kolmogorov axioms** of probability:

$$0 < P(k) < 1, \quad \sum_k P(k) = 1$$

#### 만족하지 않는 것: Calibration / What it doesn't satisfy: Calibration

"네트워크가 0.9를 출력하면, 실제로 100번 중 90번 맞는가?" — **아닙니다.**

"If the network outputs 0.9, is it correct 90 out of 100 times?" — **No.**

| 네트워크 출력 / Output | 실제 정답 비율 / Actual accuracy | 상태 / State |
|---|---|---|
| 0.9 | 0.9 | Well-calibrated (이상적 / ideal) |
| 0.9 | 0.7 | **Over-confident** (과신 — 대부분의 deep network) |

**AlexNet을 포함한 대부분의 deep network는 over-confident합니다.** 이유:

**Most deep networks including AlexNet are over-confident.** Reasons:

1. **Cross-entropy loss의 특성**: Loss를 최소화하려면 정답 logit을 **한없이 크게** 만들면 됩니다. $z_{\text{correct}} \to \infty$이면 $L \to 0$. 네트워크는 "불확실성"이 아닌 "loss 감소"를 최적화합니다. / Cross-entropy optimization drives correct logits toward infinity, not toward calibrated uncertainty.

2. **Softmax의 지수 함수**: Logit 간의 작은 차이도 확률에서는 극단적 차이로 변환됩니다: / Exponential function amplifies small logit differences into extreme probability differences:

$$z = [10, 8, 7] \Rightarrow P = [0.88, 0.12, 0.004]$$
$$z = [50, 8, 7] \Rightarrow P \approx [1.00, 0.00, 0.00]$$

#### 해결 방법: Temperature Scaling / Solution: Temperature Scaling

Logit을 온도 $T$로 나눈 후 softmax를 적용합니다:

Divide logits by temperature $T$ before applying softmax:

$$P(k) = \frac{e^{z_k / T}}{\sum_j e^{z_j / T}}$$

- $T > 1$: 분포가 부드러워짐 (confidence 감소, calibration 개선) / Distribution softens (reduced confidence, better calibration)
- $T < 1$: 분포가 날카로워짐 / Distribution sharpens
- $T$는 validation set에서 NLL 최소화로 결정 / $T$ determined by minimizing NLL on validation set

#### 정리 / Summary

| 관점 / Perspective | 확률인가? / Is it probability? |
|---|---|
| 수학적 형식 / Formal math | ✅ 공리 만족 / Axioms satisfied |
| 빈도론적 해석 / Frequentist | ❌ Calibration 보장 없음 / No calibration guarantee |
| 실용적 / Practical | ⚠️ 순서(ranking)는 유용, 절대값은 비신뢰 / Ranking useful, absolute values unreliable |

---

### Q7: PCA Augmentation을 특정 도메인(태양, 의료)이나 다채널/단채널 영상에 적용하는 방법 / Applying PCA Augmentation to domain-specific, multi-channel, or single-channel images

#### 핵심 원리 / Core Principle

AlexNet의 PCA augmentation은 "RGB 색상"에 특화된 기법이 아닙니다. 본질은:

AlexNet's PCA augmentation is not RGB-specific. The essence is:

> **데이터의 채널 간 공분산 구조를 분석하고, 자연스러운 변동 방향으로 perturbation을 가한다.**
>
> **Analyze the inter-channel covariance structure and apply perturbation along natural variation directions.**

이 원리는 **채널 수, 파장, 도메인에 무관하게** 적용 가능합니다.

This principle applies **regardless of channel count, wavelength, or domain**.

#### 일반화된 알고리즘 / Generalized Algorithm

채널 수 $C$인 영상에 대해: / For images with $C$ channels:

1. 학습 데이터 전체에서 모든 픽셀의 $C$차원 벡터를 수집 → $C \times C$ 공분산 행렬 / Collect all pixel $C$-dim vectors across training data → $C \times C$ covariance matrix
2. 고유분해 → $C$개의 고유벡터 $\mathbf{p}_1, \ldots, \mathbf{p}_C$와 고유값 $\lambda_1 \geq \ldots \geq \lambda_C$ / Eigendecomposition → $C$ eigenvectors and eigenvalues
3. 각 이미지에 perturbation: $\Delta = \sum_{i=1}^{k} \alpha_i \lambda_i \mathbf{p}_i, \; \alpha_i \sim \mathcal{N}(0, \sigma)$ / Per-image perturbation

#### 도메인별 적용 / Domain-specific Application

##### 1. 태양 관측 영상 (SDO/AIA 등) / Solar Observation (SDO/AIA etc.)

SDO/AIA: 10개 EUV/UV 파장 채널 (94Å, 131Å, 171Å, 193Å, 211Å, 304Å, 335Å, 1600Å, 1700Å, 4500Å)

SDO/AIA: 10 EUV/UV wavelength channels

```
입력 / Input: H × W × 10
공분산 행렬 / Covariance: 10 × 10
고유벡터 / Eigenvectors: 10개, 각 10차원
```

**PCA 주성분의 예상 물리적 의미 / Expected physical meaning of principal components:**

| 주성분 / PC | 예상 의미 / Expected Meaning |
|---|---|
| PC1 | 전체 밝기 변동 — 태양 활동 주기에 의한 전반적 방출량 변화 / Overall brightness — solar cycle emission variation |
| PC2 | 코로나 vs 천이영역 대비 — 고온(94Å, 131Å) vs 저온(304Å) 채널 간 반상관 / Corona vs transition region — hot vs cool channel anti-correlation |
| PC3 | 플레어/활동영역 vs 코로나 홀 — 국소적 가열의 채널별 차등 반응 / Flare/AR vs coronal hole — differential heating response |

**주의사항 / Caveats:**

- **$\sigma$를 작게**: 태양 영상은 물리적 제약이 강함. $\sigma = 0.01 \sim 0.05$ 권장 (ImageNet의 0.1보다 훨씬 작게) / Use small $\sigma$: solar images have strong physical constraints. Recommend $\sigma = 0.01$–$0.05$
- **상위 $k$개 주성분만 사용**: 분산의 90-95%를 설명하는 상위 $k$개만 사용 (하위 주성분은 노이즈에 대응할 수 있음) / Use only top $k$ PCs explaining 90-95% of variance
- **음수 clamp**: 방출 강도는 음수가 될 수 없음 → `np.maximum(augmented, 0)` / Emission intensity can't be negative

```python
def solar_pca_augmentation(image, eigvecs, eigvals, sigma=0.03, top_k=5):
    """PCA augmentation for multi-channel solar images.

    Args:
        image: Shape (H, W, C), e.g., C=10 for SDO/AIA.
        eigvecs: Shape (C, C) from training set PCA.
        eigvals: Shape (C,) sorted descending.
        sigma: Std of alpha (smaller than ImageNet's 0.1).
        top_k: Number of principal components to use.

    Returns:
        Augmented image, same shape.
    """
    alphas = np.random.normal(0, sigma, size=top_k)
    delta = eigvecs[:, :top_k] @ (alphas * eigvals[:top_k])
    augmented = image + delta[np.newaxis, np.newaxis, :]
    return np.maximum(augmented, 0)  # Emission can't be negative
```

##### 2. 의료 다채널 영상 (MRI multi-sequence) / Medical Multi-channel (MRI)

MRI: T1, T2, FLAIR, DWI, ADC 등 → $C = 3 \sim 6$

| 주성분 / PC | 예상 의미 / Expected Meaning |
|---|---|
| PC1 | 전체 신호 강도 — 코일 감도, 환자 위치 등 / Overall signal — coil sensitivity, patient position |
| PC2 | T1 vs T2 대비 — 조직 특성(지방 vs 수분) / T1 vs T2 contrast — tissue properties |
| PC3 | FLAIR 억제 정도 — 병변 신호 특성 / FLAIR suppression — lesion signal characteristics |

**핵심 이점**: 스캐너 간, 환자 간 변동을 자연스럽게 모델링 → **domain shift** 문제 완화에 특히 유용

**Key benefit**: Naturally models inter-scanner, inter-patient variation → especially useful for **domain shift** mitigation

##### 3. 단채널(단파장) 영상의 우회 방법 / Workarounds for Single-channel Images

단채널 ($C=1$)에서는 공분산 행렬이 $1 \times 1$ 스칼라(= 분산)가 되어 "방향"이 없으므로 직접 적용 불가.

With $C=1$, the covariance matrix is a $1 \times 1$ scalar (= variance), so there's no "direction" — direct application impossible.

**방법 A: 시계열 프레임을 채널로 사용** / Use temporal frames as channels:

$$\mathbf{x} = [I(t), \; I(t-\Delta t), \; I(t-2\Delta t), \; \ldots]$$

PCA로 시간적 변동의 주요 모드를 찾음. PC1 = 전체 밝기, PC2 = 급격한 변화(플레어 등)

PCA finds temporal variation modes. PC1 = overall brightness, PC2 = rapid changes (flares etc.)

**방법 B: 파생 특징을 채널로 사용** / Use derived features as channels:

```
원본 / Original:       I
Sobel x gradient:     ∂I/∂x
Sobel y gradient:     ∂I/∂y
Laplacian:            ∇²I
로그 강도 / Log:       log(I + 1)
```

5개를 채널로 쌓아 PCA → 강도-구조 간 공변 방향을 찾음

Stack 5 as channels → PCA finds intensity-structure covariation directions

**방법 C: 인접 픽셀로 의사 채널 생성** / Create pseudo-channels from neighboring pixels:

$$\mathbf{x}_{i,j} = [I(i,j), \; I(i-1,j), \; I(i+1,j), \; I(i,j-1), \; I(i,j+1)]$$

공간적 질감(texture)의 자연스러운 변동 방향을 찾음

Finds natural variation directions of spatial texture

#### 도메인별 적용 가이드 요약 / Domain Application Guide Summary

| 도메인 / Domain | 채널 수 / Channels | $\sigma$ 권장 / Recommended | 주의사항 / Notes |
|---|---|---|---|
| 자연 이미지 / Natural (ImageNet) | 3 (RGB) | 0.1 | 원래 AlexNet 설정 / Original AlexNet setting |
| 태양 EUV / Solar (SDO/AIA) | 7–10 | 0.01–0.05 | 물리적 온도 구조 제약, 음수 clamp, top-k / Physical temperature constraints, clamp negatives |
| 의료 MRI / Medical MRI | 3–6 | 0.02–0.05 | 스캐너 간 변동 모델링 / Inter-scanner variation modeling |
| 천문 다파장 / Astronomy | 4–20+ | 0.01–0.03 | SED 물리적 제약, 적색이동 / SED physical constraints, redshift |
| 단채널 / Single-channel (CT) | 1 | — | 직접 적용 불가 → 방법 A/B/C로 우회 / Direct application impossible → workarounds A/B/C |
| 초분광 / Hyperspectral | 100+ | 0.005–0.02 | 차원 축소 후 적용, 인접 밴드 상관 높음 / Apply after dimensionality reduction |

**핵심 통찰**: PCA augmentation의 본질은 "데이터가 자연스럽게 변동하는 방향으로만 perturbation을 가한다"는 것. 이것은 임의의 노이즈보다 효과적 — **물리적으로 의미 있는 변동**만 만들어내기 때문. 태양 데이터에서는 "태양 주기에 따른 전반적 방출 변화"나 "플레어 전후의 채널별 강도 변화 패턴"과 같은 실제 관측 조건 변동을 모사할 수 있음.

**Key insight**: PCA augmentation's essence is "perturb only along directions where data naturally varies." More effective than random noise because it creates only **physically meaningful variations**. For solar data, this can simulate real observational condition variations like "overall emission changes across the solar cycle" or "channel-specific intensity change patterns around flares."
