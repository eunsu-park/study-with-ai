---
title: "Deep Residual Learning for Image Recognition"
authors: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
year: 2015
journal: "IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2016) — Best Paper Award"
doi: "arXiv:1512.03385"
topic: Artificial Intelligence / Deep Learning
tags: [resnet, residual-learning, skip-connection, identity-mapping, deep-networks, cnn, imagenet, degradation-problem, bottleneck-block]
status: completed
date_started: 2026-04-19
date_completed: 2026-04-19
---

# 20. Deep Residual Learning for Image Recognition / 이미지 인식을 위한 심층 잔차 학습

---

## 1. Core Contribution / 핵심 기여

ResNet은 딥러닝에서 "더 깊게 쌓으면 더 좋아질 것"이라는 직관이 실제로는 무너지는 현상 — **degradation problem** (깊이가 증가할수록 *훈련* 오차가 오히려 증가) — 을 진단하고, 이를 해결하는 단순하지만 혁명적인 아키텍처 원리를 제시했다. 핵심 아이디어는 각 블록이 직접 원하는 매핑 $H(x)$를 학습하는 대신, **잔차 $F(x) = H(x) - x$** 를 학습하고 skip connection으로 입력 $x$를 더해주는 것이다: $y = F(x) + x$. 이 재정식화는 (1) **identity mapping**이 $F = 0$으로 자명하게 달성되어 불필요한 층이 네트워크 성능을 해치지 않게 하고, (2) 역전파 시 기울기가 "+1" 항을 통해 어떤 깊이에서도 손실 없이 전파되며, (3) 파라미터 수나 계산량을 거의 늘리지 않으면서 깊이의 이점만 취할 수 있게 한다. 그 결과 ImageNet에서 **152층 ResNet이 top-5 error 3.57%** 로 인간 수준을 최초로 넘어섰고, ILSVRC 2015 분류/검출/위치추정과 COCO 2015 검출/분할 5개 부문을 석권했다. Skip connection은 이후 DenseNet, Transformer, Diffusion U-Net 등 거의 모든 현대 아키텍처의 **기본 문법(grammar)** 으로 자리잡았다.

ResNet diagnoses the **degradation problem** — a phenomenon where simply stacking more layers causes *training* accuracy to saturate and then degrade, contradicting the intuition that deeper networks should perform at least as well as shallower ones. The key idea reformulates each block: instead of directly fitting the desired mapping $H(x)$, the layers fit a **residual function** $F(x) = H(x) - x$, with a skip connection adding the input back: $y = F(x) + x$. This reformulation (1) makes the **identity mapping trivial** to learn ($F = 0$), so adding layers cannot hurt training error; (2) creates a "+1" gradient highway that propagates gradients losslessly at any depth; (3) adds almost no parameters or FLOPs. The result: **152-layer ResNet achieves 3.57% top-5 error on ImageNet**, the first model to surpass human-level performance on this benchmark, sweeping all five ILSVRC 2015 and COCO 2015 competitions. Skip connections have since become the default primitive in nearly every modern deep architecture — DenseNet, ResNeXt, Transformers, diffusion U-Nets, and beyond.

---

## 2. Reading Notes / 읽기 노트

### Section 1: Introduction / 도입

**딥러닝의 깊이 경쟁과 한계**
AlexNet(2012), VGGNet(2014), GoogLeNet(2014) 이후 "깊이가 곧 성능"이라는 공식이 확립되었다. 그러나 단순히 층을 쌓는 것은 **두 가지 장애물**에 부딪혔다: (1) **vanishing/exploding gradients** — 해결책: Xavier/He 초기화, Batch Normalization; (2) **degradation problem** — ResNet이 다루는 바로 그 문제.

After AlexNet (2012), VGG (2014), and GoogLeNet (2014), "depth = performance" became conventional wisdom. But naively stacking layers faces two obstacles: (1) **vanishing/exploding gradients** — addressed by Xavier/He initialization and Batch Normalization; (2) the **degradation problem** — what this paper tackles.

**Degradation problem 정의** (Figure 1, p. 1)
CIFAR-10에서 56층 plain network가 20층보다 **훈련 오차**가 더 높고, 테스트 오차도 더 높다. 이는 과적합이 아니다 (훈련 오차조차 나쁘므로). 기울기 소실도 아니다 (BN 사용 시 순방향/역방향 활성값이 건강함). 단순히 "깊은 최적화가 어렵다"는 새로운 문제이다.

In Figure 1, a 56-layer plain network on CIFAR-10 has **higher training error** (and test error) than a 20-layer plain network. This is not overfitting (training error is worse), nor vanishing gradients (BN keeps activations healthy). It is a distinct "optimization-is-hard-when-deep" problem.

**핵심 통찰 (Construction argument)**
얕은 네트워크 + 추가 identity 층으로 임의의 깊은 네트워크를 **구성**할 수 있다. 따라서 깊은 네트워크가 얕은 네트워크보다 **훈련 오차가 절대 높을 수 없어야 한다**. 그러나 실제로는 SGD가 그 해를 찾지 못한다. 저자들은 "**학습기(learner)가 잔차를 모델링하게 하면 identity를 찾기 쉬워진다**"는 가설을 세운다.

Construction argument: a deep model built from a shallower model plus additional identity layers has the same training error as the shallow model. So a deep model should *never* have higher training error — yet SGD fails to find such a solution. The authors hypothesize: "**if learners fit the residual, the identity solution becomes easy to reach.**"

**Residual learning 제안** (식 1–2, p. 2)
블록이 $H(x)$ 대신 $F(x) := H(x) - x$를 학습하고, 원하는 매핑을 $F(x) + x$로 재구성한다. Identity가 최적일 경우 $F(x) \to 0$으로 푸시하기만 하면 된다 — 이는 모든 가중치를 0 근처로 미는 것과 같아서 최적화가 훨씬 쉽다.

Blocks fit $F(x) := H(x) - x$ instead of $H(x)$, reconstructing the target as $F(x) + x$. If identity is optimal, pushing $F \to 0$ (weights → 0) achieves it — far easier to optimize than fitting an identity through stacked nonlinear layers.

### Section 2: Related Work / 관련 연구

**Residual representations**: VLAD, Fisher Vector (이미지 검색), multigrid (PDE 풀이), wavelet 전처리 — 모두 **잔차를 표현하는 것이 원래 값을 표현하는 것보다 쉽다**는 경험적 증거. Pre-conditioning도 잔차 관점에서 수렴을 가속시킨다.

Residual representations: VLAD, Fisher Vector (image retrieval), multigrid methods (PDEs), wavelet preconditioning — all provide empirical evidence that **reformulating as residuals simplifies optimization**.

**Shortcut connections**: LeNet에도 "auxiliary classifiers", "inception branches", **Highway Networks** (Srivastava 2015) 같은 선행 연구가 있다. 특히 Highway Network는 $y = F(x)\cdot T(x) + x\cdot(1-T(x))$로 **gated** skip을 사용하는데, $T$가 학습된 sigmoid 게이트이다. ResNet은 이 게이트를 **항상 1로 고정** (parameter-free identity)하여 더 단순하고, 모든 정보가 항상 흐르도록 한다. Highway는 매우 깊어져도 성능 향상이 보고되지 않은 반면, ResNet은 100층 이상에서 **큰 이득**을 보여준다.

Shortcut connections: LeNet's "auxiliary classifiers," Inception branches, and **Highway Networks** (Srivastava 2015) are precedents. Highway uses $y = F(x)\cdot T(x) + x\cdot(1-T(x))$ with a learned sigmoid gate $T$. ResNet sets $T \equiv 1$ — a **parameter-free identity** that always lets information through. Crucially, Highway Networks have *not* reported accuracy gains at extreme depths, whereas ResNet shows **substantial gains beyond 100 layers**.

### Section 3: Deep Residual Learning / 심층 잔차 학습

#### 3.1 Residual Learning / 잔차 학습

$\mathcal{H}(x)$를 몇 개의 쌓인 층이 근사해야 할 매핑이라 하자. 다층 비선형 네트워크가 복잡한 함수를 근사할 수 있다는 것이 **Universal Approximation Theorem**으로 보장되지만, 이는 $\mathcal{H}(x) - x$ (잔차) 역시 근사할 수 있음을 의미한다. 두 형식은 **동일한 표현력**을 갖지만 **학습 난이도가 다르다**.

Let $\mathcal{H}(x)$ be a mapping to be fit by a few stacked layers. By the universal approximation theorem, these layers can fit both $\mathcal{H}(x)$ and the residual $\mathcal{H}(x) - x$. Both forms have equal expressive power, but **differ in ease of learning**.

만약 최적 함수가 identity에 가깝다면, 잔차 형식은 "0으로 가기"만 하면 된다. 실험 결과(§4.1)는 실제로 학습된 잔차 함수들이 대체로 **작은 응답**을 가짐을 보여 — 이 가설을 뒷받침한다 (Figure 7).

If the optimal function is close to an identity, the residual formulation just needs to push toward zero. Section 4.1 experiments show that learned residual functions have **small responses** in general, supporting the hypothesis (Figure 7).

#### 3.2 Identity Mapping by Shortcuts / Shortcut을 통한 Identity 매핑

**기본 블록** (식 1):
$$\mathbf{y} = F(\mathbf{x}, \{W_i\}) + \mathbf{x}$$

**차원이 다를 때** (식 2):
$$\mathbf{y} = F(\mathbf{x}, \{W_i\}) + W_s \mathbf{x}$$

여기서 $W_s$는 1×1 conv로 채널 수/해상도를 맞춰주는 **linear projection**이다. Shortcut이 identity일 때는 **파라미터도, 계산량도 추가되지 않는다**는 점이 중요하다. $F$는 최소 2개 층을 포함해야 하며(1개면 선형이라 이득이 없음), 실험에서는 2~3개 층이 주로 쓰인다.

Where $W_s$ is a 1×1 conv **linear projection** aligning dimensions. Identity shortcuts add **no parameters and no computation**. $F$ must have at least 2 layers (1 would be linear, providing no benefit); experiments use 2 or 3 layers per block.

**Element-wise addition**: 채널별, 공간별로 동일한 위치끼리 더한다. 따라서 $F(x)$의 출력 텐서와 $x$의 텐서는 같은 shape이어야 한다. ReLU는 덧셈 **후**에 적용된다 (post-activation, v1 스타일).

Element-wise addition happens channel-wise and spatially. $F(x)$ and $x$ must share the same shape. ReLU is applied **after** the addition (post-activation, v1 style).

#### 3.3 Network Architectures / 네트워크 아키텍처

**Plain baseline (34층)** — VGG 스타일 영감:
- 3×3 conv만 사용
- 같은 출력 feature map 크기에서는 같은 필터 수
- Feature map 크기가 절반이 되면 필터 수를 두 배로 (시간 복잡도 유지)
- Downsampling은 stride 2 conv로
- 끝에는 GAP + 1000-way FC + softmax
- 총 34개 가중치 층, 3.6 **billion FLOPs** (ILSVRC VGG-19의 19.6G의 18%)

**Plain baseline (34 layers)**, VGG-inspired:
- Only 3×3 convs
- Same filter count for same feature map size
- Filters double when feature map halves (constant time complexity)
- Downsampling via stride-2 convs
- Ends with GAP + 1000-way FC + softmax
- 34 weight layers, **3.6 billion FLOPs** (18% of VGG-19's 19.6G)

**Residual network (34층)**:
- Plain에 skip connection 추가
- Identity shortcut은 같은 차원일 때
- 차원 증가 시 옵션: (A) zero-padding identity (파라미터 추가 없음), (B) projection shortcut만 사용, (C) 모든 shortcut을 projection

**Residual network (34 layers)**: same as plain plus skip connections. At dimension changes, options:
- (A) identity + zero-padding extra channels (no new parameters)
- (B) projection only where dimensions change, identity elsewhere
- (C) projection for every shortcut

#### 3.4 Implementation / 구현

- 이미지: [256, 480] 범위의 short side로 scale, 224×224 random crop, mean subtracted, standard color augmentation
- 각 conv 뒤, activation 전에 **BN 적용**
- He initialization (2015)
- SGD, mini-batch 256
- Learning rate: 0.1에서 시작, 오차 안정화 시 10분의 1로 감소
- $60 \times 10^4$ iterations, weight decay 0.0001, momentum 0.9
- **Dropout 사용하지 않음** (BN의 regularization 효과로 충분)
- 테스트: 10-crop, 다양한 스케일의 fully-convolutional 평가 후 평균

- Input: resize short side to [256, 480], 224×224 random crop, mean subtract, standard color aug
- **BN** after each conv, before activation
- He initialization (2015)
- SGD, mini-batch 256; lr 0.1 with 10× decay on plateau
- $60 \times 10^4$ iterations, weight decay 1e-4, momentum 0.9
- **No dropout** (BN provides sufficient regularization)
- Test: 10-crop + fully-conv evaluation at multiple scales, averaged

### Section 4: Experiments / 실험

#### 4.1 ImageNet Classification

**Dataset**: ImageNet 2012 (1.28M training, 50K validation, 100K test images, 1000 classes). 평가 지표: top-1 error, top-5 error.

**Plain Networks (Table 2)**:
| Model | top-1 error |
|---|---|
| Plain-18 | 27.94% |
| Plain-34 | **28.54%** |

34층이 18층보다 오차가 **더 높다**. 이는 기울기 소실이 아니다 — BN 덕분에 forward activation이 0이 아닌 분산을 가지며, backward gradient도 건강한 norm을 보인다 (실험적으로 확인). 저자들은 "plain 깊은 네트워크는 **지수적으로 낮은 수렴률(exponentially low convergence rate)** 을 가질 가능성이 크다"고 추측한다.

Plain-34 (28.54%) is worse than Plain-18 (27.94%). Not vanishing gradients — BN ensures healthy forward/backward signals (verified empirically). Authors speculate deep plain nets have **exponentially low convergence rates**.

**Residual Networks (Table 2)**:
| Model | top-1 error |
|---|---|
| ResNet-18 | 27.88% |
| ResNet-34 (A) | 25.03% |
| ResNet-34 (B) | 24.52% |
| ResNet-34 (C) | 24.19% |

**세 가지 핵심 관찰**:
1. **Degradation이 해결되었다**: ResNet-34가 ResNet-18보다 2.8% 낮은 오차 → 깊이의 이점을 제대로 누림.
2. **Plain vs Residual (동일 깊이)**: ResNet-34 (25.03%) vs Plain-34 (28.54%) → 3.5% 개선. 훈련 오차도 크게 낮아 최적화가 잘 됨.
3. **18층에서는 효과 미미**: ResNet-18과 Plain-18이 비슷. Residual learning의 이점은 **최적화가 어려운 깊은 네트워크**에서 두드러진다. 단, ResNet-18은 **더 빨리 수렴**한다.

Three key findings:
1. **Degradation solved**: ResNet-34 is 2.8% better than ResNet-18 → depth now helps.
2. **Plain vs Residual (same depth)**: ResNet-34 beats Plain-34 by 3.5%; training error also drops.
3. **Minimal effect at 18 layers**: ResNet and Plain are close, but ResNet **converges faster**. The benefit is pronounced only when optimization is hard.

**Identity vs Projection shortcuts**:
- (A) zero-padding identity: 25.03%
- (B) projection for dimension change only: 24.52%
- (C) projection for all shortcuts: 24.19%

(A), (B), (C)의 차이는 작다. 즉 **projection shortcut은 필수가 아니며**, identity로도 충분하다. 이후 모든 실험은 (B)를 기본으로 사용 (계산 효율). (C)는 더 많은 파라미터를 쓰지만 성능 차이는 미미.

Differences are small — **projection shortcuts are not essential**; identity suffices. Default uses (B) for efficiency. (C) adds parameters for marginal gain.

#### Deeper Bottleneck Architectures / 더 깊은 bottleneck 아키텍처

50층 이상에서는 계산 효율을 위해 **bottleneck block** 도입 (Figure 5 right):
- 3개 층: 1×1 (채널 축소) → 3×3 (주 연산) → 1×1 (채널 복원)
- 예: 256채널 입력 → 1×1로 64로 축소 → 3×3 → 1×1로 256 복원
- Basic block(2×3×3)과 **유사한 계산량**이지만 더 깊은 표현 가능

For 50+ layers, introduce **bottleneck blocks** (Figure 5 right): 1×1 (reduce) → 3×3 (main) → 1×1 (restore). Example: 256 → 64 via 1×1 → 3×3 on 64 → 256 via 1×1. Comparable FLOPs to basic block (two 3×3) but enables much deeper models.

**결과 (Table 3, Table 4)**:

| Model | top-1 err | top-5 err (single model, 10-crop) |
|---|---|---|
| ResNet-18 | 27.88% | - |
| ResNet-34 | 24.19% | - |
| ResNet-50 | 22.85% | 6.71% |
| ResNet-101 | 21.75% | 6.05% |
| ResNet-152 | **21.43%** | **5.71%** |

**ResNet-152**는 VGG-16 (15.3B FLOPs)보다도 적은 **11.3B FLOPs**를 쓴다. 깊지만 가볍다.

ResNet-152 uses only **11.3B FLOPs**, less than VGG-16 (15.3B). Deep yet efficient.

**Ensemble (Table 5)**: ResNet-{50, 101×2, 152×2, different depths} 6 model 앙상블 → **top-5 error 3.57%**. 인간 성능(Russakovsky 5.1%)을 넘어선 최초 기록.

Ensemble of 6 models (different depths): top-5 error **3.57%** — first to surpass human performance (~5.1%).

#### 4.2 CIFAR-10 and Analysis

**작은 데이터셋**에서 깊이의 영향 조사. 32×32 이미지, 50K train/10K test, 10 classes.

**아키텍처**: 첫 3×3 conv → 3개 스테이지의 n개 블록 (feature map 32, 16, 8) → GAP → 10-way FC. 총 $6n+2$ 층.
- n=3,5,7,9: 20, 32, 44, 56층
- n=18: 110층
- n=200: 1202층 (특수 실험)

Architecture: initial 3×3 conv → 3 stages of $n$ blocks each (feature maps 32/16/8) → GAP → 10-way FC. Depth = $6n+2$.
- n=3,5,7,9: 20/32/44/56 layers
- n=18: 110 layers
- n=200: 1202 layers

**Plain vs ResNet (Figure 6)**:
| Model | Plain error | ResNet error |
|---|---|---|
| 20 layers | 9.03% | 8.75% |
| 32 layers | 9.74% | 7.51% |
| 44 layers | 10.30% | 7.17% |
| 56 layers | 11.32% | 6.97% |
| 110 layers | - | **6.43%** |

**Plain은 깊이가 증가할수록 오차가 증가**; ResNet은 감소. 110층 ResNet이 6.43%로 최고 성능.

Plain nets degrade with depth; ResNets improve. **110-layer ResNet: 6.43%** — best.

**1202층 실험**: 6.93% 오차. 110층(6.43%)보다 **오히려 나빠짐**. 저자들은 과적합으로 해석 (CIFAR-10 크기 대비 모델이 너무 큼). 이 문제는 ResNet v2(2016) pre-activation과 정규화 기법(Dropout 등)으로 해결된다.

1202-layer ResNet: 6.93% — **worse** than 110-layer (6.43%). Interpreted as overfitting (model too large for CIFAR-10). Later resolved by ResNet v2 (2016) with pre-activation and stronger regularization.

**잔차 함수의 응답 분석 (Figure 7)**:
각 층의 $F(x)$의 표준편차(scale)를 측정 → ResNet의 잔차는 plain 네트워크의 대응 층 응답보다 **훨씬 작다**. 이는 "실제 최적 함수가 identity에 가깝고, 잔차 형식이 적절하다"는 가설을 지지한다.

Response analysis (Figure 7): $F(x)$ standard deviations in ResNets are **much smaller** than corresponding plain-net responses — supporting the hypothesis that optimal functions are close to identity, and the residual form is appropriate.

#### 4.3 Object Detection on PASCAL and MS COCO

Backbone을 VGG-16에서 ResNet-101로 교체한 Faster R-CNN이 주요 설정.

**PASCAL VOC (Table 7)**: mAP가 VGG-16의 73.2%에서 ResNet-101의 **76.4%** 로 (VOC 2007).

**MS COCO (Table 8)**: mAP@[.5:.95]가 VGG의 21.2%에서 ResNet-101의 **27.2%** 로 → **28% 상대 개선**. 이는 ILSVRC/COCO 2015 모든 부문(분류, 검출, 위치추정, 인스턴스 분할) 석권으로 이어졌다.

Backbone swap from VGG-16 to ResNet-101 in Faster R-CNN:
- PASCAL VOC 2007: 73.2% → **76.4% mAP**
- MS COCO: 21.2% → **27.2% mAP@[.5:.95]** (+28% relative)

This led to sweeping all tracks of ILSVRC/COCO 2015 (classification, detection, localization, instance segmentation).

---

## 3. Key Takeaways / 핵심 시사점

1. **Degradation problem은 기울기 문제가 아닌 최적화 문제이다** — BN으로 forward/backward signal이 건강한데도 깊은 plain network가 훈련 오차조차 높아지는 현상. 이는 깊은 비선형 네트워크가 identity를 근사하기 어려운 근본적인 한계를 드러낸다.
   **The degradation problem is an optimization problem, not a gradient problem** — plain deep networks fail to converge to good training error even with BN; identity is surprisingly hard to approximate through stacked nonlinear layers.

2. **재정식화(reformulation)는 표현력이 아닌 학습 난이도를 바꾼다** — $F(x) + x$와 $H(x)$는 동일한 함수 공간을 표현하지만, SGD가 **어느 점에서 출발하느냐**와 **identity 근처에서 얼마나 효율적이냐**가 극적으로 다르다. "같은 표현력"이 "같은 학습 가능성"을 의미하지 않는다는 중요한 교훈.
   **Reformulation changes trainability, not expressiveness** — $F(x)+x$ and $H(x)$ span the same function space, but SGD's starting point and efficiency near the identity differ dramatically. Equal representational power ≠ equal trainability.

3. **Identity shortcut은 파라미터-프리 최고의 선택** — Projection shortcut(옵션 C)은 약간 더 좋지만 파라미터와 계산량을 늘린다. Identity가 "가장 깔끔한 기울기 경로"를 제공하며, 후속 연구(ResNet v2)에서는 **어떤 비선형성도 skip 경로에 넣으면 안 된다**는 원리로 발전.
   **Identity shortcuts are the best parameter-free choice** — projection shortcuts give marginal gains but add cost. Identity provides the cleanest gradient path, later formalized in ResNet v2 as "no nonlinearity on the skip path."

4. **Bottleneck block은 깊이-계산량 trade-off의 핵심** — 1×1 conv로 차원을 축소/복원하여 3×3 conv의 계산량을 관리 가능한 수준으로 유지. ResNet-152가 VGG-16보다 **적은 FLOPs**를 쓰면서도 훨씬 깊고 정확한 비결.
   **The bottleneck block is the key depth-compute trade-off** — 1×1 convs shrink/restore channels around the 3×3, keeping cost manageable. This is why ResNet-152 uses **fewer FLOPs** than VGG-16.

5. **잔차는 실제로 작다** — Figure 7의 잔차 응답 분석은 대부분의 블록에서 $F(x)$의 크기가 작음을 보여준다. 즉 각 블록은 입력에 **작은 보정(small correction)** 을 더하는 방식으로 작동한다. 이는 ResNet을 **반복적 표현 개선(iterative refinement)** 으로 해석하는 관점(Greff et al. 2017)의 경험적 근거.
   **Residuals are empirically small** — Figure 7 shows small $F(x)$ magnitudes; each block adds a small correction. Provides empirical grounding for the "iterative refinement" interpretation (Greff et al. 2017).

6. **Ensemble은 여전히 중요하다** — 3.57% top-5는 **6개 모델 앙상블**의 결과. 단일 모델 최고는 4.49%. 이는 ResNet이 개별 모델로도 강력하지만, 실무에서는 앙상블이 필수였음을 보여준다. 이후 self-distillation, EMA 등이 앙상블의 효과를 단일 모델에 녹이는 방향으로 발전.
   **Ensembling still matters** — the 3.57% figure is a **6-model ensemble**; best single model is 4.49%. ResNet alone is strong, but practical SOTA needed ensembles. Later work (self-distillation, EMA) internalized ensemble effects into single models.

7. **1202층 실험은 새로운 문제(과적합)의 시작** — 깊이의 장벽을 넘자, 그 다음 장벽(**표현 용량 > 데이터**)이 드러났다. 이는 후속의 다양한 정규화 기법(Stochastic Depth, MixUp, CutMix)과 더 큰 데이터셋(ImageNet-21k, JFT-300M), 그리고 Pre-activation ResNet v2 설계의 동기가 되었다.
   **The 1202-layer experiment reveals the next barrier** — overfitting. Crossing the depth barrier exposed the capacity-vs-data barrier, motivating later regularizers (Stochastic Depth, MixUp) and scale-up efforts (ImageNet-21k, JFT-300M).

8. **Skip connection은 범용 설계 원리가 되었다** — DenseNet(concat), Highway(gated), Transformer(pre/post-norm residual), Diffusion U-Net, GNN의 residual — 모두 "입력을 보존하면서 점진적으로 수정"이라는 ResNet의 철학을 계승. GPT·Claude 같은 현대 LLM도 이 원리 없이는 학습 불가능.
   **Skip connections became a universal design primitive** — DenseNet (concat), Highway (gated), Transformer residual (pre/post-norm), diffusion U-Nets, GNN residuals — all inherit ResNet's philosophy of "preserve input, refine incrementally." Modern LLMs like GPT/Claude would not train without it.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Residual Block / 잔차 블록

**기본 형식 (identity shortcut)**
$$\mathbf{y} = F(\mathbf{x}, \{W_i\}) + \mathbf{x}$$
- $\mathbf{x}, \mathbf{y}$: 블록의 입력/출력 (같은 shape).
- $F$: 학습할 잔차 함수 (2개 이상의 층).
- 덧셈 후 ReLU (post-activation, v1 스타일).

**차원이 바뀔 때 (projection shortcut)**
$$\mathbf{y} = F(\mathbf{x}, \{W_i\}) + W_s \mathbf{x}$$
- $W_s$: 1×1 conv, stride 조정 포함 — 채널 수와 공간 해상도를 맞춤.

### 4.2 Basic Block (2-layer) / 기본 2층 블록

$$F(\mathbf{x}) = W_2 \cdot \sigma(\text{BN}(W_1 \mathbf{x}))$$

블록 전체 흐름:
```
x → [Conv 3×3, BN, ReLU] → [Conv 3×3, BN] → (+x) → ReLU → y
```
- 두 conv 모두 같은 필터 수 (예: 64)
- 첫 conv 뒤에만 ReLU, 두 번째 conv 뒤에는 덧셈 후 ReLU

### 4.3 Bottleneck Block (3-layer) / 병목 3층 블록

$$F(\mathbf{x}) = W_3 \cdot \sigma(\text{BN}(W_2 \cdot \sigma(\text{BN}(W_1 \mathbf{x}))))$$

- $W_1$: 1×1 conv, 256 → 64 (채널 4배 축소)
- $W_2$: 3×3 conv, 64 → 64 (주 연산)
- $W_3$: 1×1 conv, 64 → 256 (채널 복원)

**계산량 비교** (같은 입출력 채널 256 기준):
- Basic 2-layer: $2 \times (3^2 \times 256 \times 256) = 2 \times 589{,}824 \approx 1.18\text{M FLOPs/pixel}$
- Bottleneck 3-layer: $(1 \times 256 \times 64) + (9 \times 64 \times 64) + (1 \times 64 \times 256) = 16{,}384 + 36{,}864 + 16{,}384 \approx 69.6\text{K FLOPs/pixel}$

Bottleneck이 **~17배 적은 계산**으로 더 깊은 표현을 가능하게 한다.

Bottleneck is **~17× cheaper** per block, enabling much deeper networks at similar compute budget.

### 4.4 Gradient Flow / 기울기 전파

순방향 재귀 (pre-activation ResNet v2 형식에서 가장 깔끔):
$$\mathbf{x}_{l+1} = \mathbf{x}_l + F(\mathbf{x}_l, W_l)$$
$$\mathbf{x}_L = \mathbf{x}_l + \sum_{i=l}^{L-1} F(\mathbf{x}_i, W_i)$$

역방향:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}_l} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}_L} \cdot \left(1 + \frac{\partial}{\partial \mathbf{x}_l} \sum_{i=l}^{L-1} F(\mathbf{x}_i, W_i)\right)$$

- "**+1**" 항: 어떤 깊이에서도 기울기의 최소 보장. Vanishing 방지.
- 합 항: 잔차들의 기여. $F_i$가 작을수록 기울기는 입력 층 기울기에 **수렴**.

The "+1" term is an unconditional lower bound on the gradient — gradients cannot vanish regardless of depth. The sum term captures contributions from residual branches.

### 4.5 Network Depth Formula (CIFAR) / CIFAR 네트워크 깊이 공식

$$\text{depth} = 6n + 2$$
- $n$: 각 스테이지의 블록 수 (3 스테이지: feature map 32, 16, 8)
- 각 블록 = 2 conv → $6n$ conv + 첫 conv + 최종 FC = $6n + 2$

Examples:
- $n = 3$ → 20 layers
- $n = 18$ → 110 layers
- $n = 200$ → 1202 layers

### 4.6 Worked Example: 34-layer ResNet on ImageNet / 구체적 예시

입력: 224×224×3 이미지.

| Stage | Output | Layer | Blocks |
|---|---|---|---|
| conv1 | 112×112×64 | 7×7 conv, stride 2 | — |
| maxpool | 56×56×64 | 3×3 max, stride 2 | — |
| conv2_x | 56×56×64 | [3×3, 64] × 2 | 3 blocks |
| conv3_x | 28×28×128 | [3×3, 128] × 2 | 4 blocks |
| conv4_x | 14×14×256 | [3×3, 256] × 2 | 6 blocks |
| conv5_x | 7×7×512 | [3×3, 512] × 2 | 3 blocks |
| GAP+FC | 1000 | avg pool, 1000-d FC | — |

총 가중치 층: $1 + (3+4+6+3) \times 2 + 1 = 34$층. FLOPs ≈ 3.6B.

첫 블록은 identity shortcut (같은 차원). 스테이지 전환에서는 projection shortcut (stride 2 + 채널 증가).

Total weight layers: $1 + (3+4+6+3)\times 2 + 1 = 34$. FLOPs ≈ 3.6B. Identity shortcuts within a stage; projection shortcuts at stage transitions (stride 2 + channel doubling).

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1943  McCulloch-Pitts     — binary neuron                        [#01]
1958  Rosenblatt          — Perceptron                           [#03]
1969  Minsky-Papert       — Perceptron limits                    [#04]
1986  Rumelhart           — Backpropagation                      [#06]
1989  LeCun               — LeNet, first CNN                     [#07]
1998  LeCun               — LeNet-5, gradient-based learning     [#10]
2006  Hinton              — DBN, layer-wise pretraining          [#12]
2012  Krizhevsky          — AlexNet, deep learning revolution    [#13]
2014  Simonyan-Zisserman  — VGGNet (19 layers)
2014  Szegedy             — GoogLeNet / Inception (22 layers)    [#19]
2015  Ioffe-Szegedy       — Batch Normalization                  [#18]
2015  Srivastava          — Highway Networks (gated skip)
2015  He-Zhang-Ren-Sun    — ResNet (152 layers, 3.57% top-5)    ★ THIS PAPER
2016  He                  — ResNet v2 (pre-activation, 1001 L)
2016  Huang               — DenseNet (dense concat skip)
2017  Xie                 — ResNeXt (grouped residuals)
2017  Vaswani             — Transformer (residual in attention)  [#25]
2020  Dosovitskiy         — ViT (Transformer for vision)
2022  Liu                 — ConvNeXt (ResNet modernized)
```

ResNet은 "깊이의 장벽"을 처음으로 체계적으로 넘은 아키텍처이다. 이전의 모든 노력(BN, Xavier/He 초기화, Highway)은 **증상을 완화**했지만, ResNet은 **근본 원인(최적화 어려움)** 을 재정식화로 해결했다. 이 아이디어는 10년 뒤에도 모든 대형 모델의 기반으로 남아 있다.

ResNet was the first architecture to systematically break the "depth barrier." Prior efforts (BN, He init, Highway) mitigated symptoms; ResNet addressed the root cause — optimization difficulty — via reformulation. A decade later, the idea remains foundational across all large models.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **#13 AlexNet (Krizhevsky 2012)** | Deep learning의 시작점. 8층으로 ImageNet 우승 → ResNet은 이를 152층까지 확장. | CNN + ReLU + GPU 훈련 패러다임을 확립한 AlexNet이 없었다면 ResNet의 깊이 실험도 의미가 없었음. |
| **#18 Batch Normalization (Ioffe-Szegedy 2015)** | ResNet의 모든 conv 뒤에 BN 적용. BN이 기울기 소실을 해결하자 degradation 문제가 **남는 진짜 장벽**으로 드러남. | BN은 ResNet의 **필수 전제 조건**. BN 없는 ResNet은 훨씬 학습이 어려움. |
| **#19 Inception / GoogLeNet (Szegedy 2014)** | Bottleneck 1×1 conv 아이디어의 기원. ResNet-50/101/152의 bottleneck 블록은 Inception에서 차용. | "깊이와 너비를 효율적으로 확장하는" Inception 철학이 ResNet의 bottleneck 설계로 이어짐. |
| **Highway Networks (Srivastava 2015)** | Gated skip connection의 선행 연구. ResNet은 게이트를 1로 고정하여 단순화 + 성능 향상. | ResNet은 "**parameter-free identity가 learned gate보다 낫다**"는 중요한 교훈을 남김. |
| **ResNet v2 (He 2016)** | 같은 저자의 직접적 후속. Pre-activation 설계로 1001층 네트워크 성공. | 본 논문의 1202층 실험 한계(과적합으로 해석)의 상당 부분이 **사실은 최적화 문제**였음을 보이고 해결. |
| **DenseNet (Huang 2016)** | Skip을 concat으로 구현한 대안. 파라미터 효율성은 더 좋지만 메모리 비용 큼. | ResNet의 "addition" 선택이 **최적화 용이성**에 최적화된 설계였음을 대비로 보여줌. |
| **#25 Transformer (Vaswani 2017)** | 각 attention/FFN 서브층이 $x + \text{Sublayer}(x)$ 구조 — 정확히 ResNet의 skip. | GPT, BERT, LLaMA, Claude 등 **모든 현대 LLM이 skip connection 없이는 학습 불가능**. ResNet의 유산이 NLP/멀티모달 전체를 관통. |
| **Stochastic Depth (Huang 2016)** | ResNet 블록을 훈련 중 확률적으로 건너뛰는 정규화. | ResNet의 "블록은 선택적 기여"라는 관점을 극단화. 1000층+ 모델 훈련의 필수 기법. |

---

## 7. References / 참고문헌

- **This paper**: He, K., Zhang, X., Ren, S., & Sun, J. (2015). *Deep Residual Learning for Image Recognition*. arXiv:1512.03385 (CVPR 2016, Best Paper Award).
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Identity Mappings in Deep Residual Networks*. ECCV 2016. (Pre-activation ResNet v2)
- Ioffe, S., & Szegedy, C. (2015). *Batch Normalization*. ICML 2015.
- Szegedy, C., et al. (2014). *Going Deeper with Convolutions*. CVPR 2015. (GoogLeNet / Inception)
- Simonyan, K., & Zisserman, A. (2014). *Very Deep Convolutional Networks for Large-Scale Image Recognition*. ICLR 2015. (VGGNet)
- Krizhevsky, A., Sutskever, I., & Hinton, G. (2012). *ImageNet Classification with Deep CNNs*. NeurIPS 2012. (AlexNet)
- Srivastava, R. K., Greff, K., & Schmidhuber, J. (2015). *Highway Networks*. ICML 2015 DL Workshop.
- Huang, G., Liu, Z., van der Maaten, L., & Weinberger, K. Q. (2016). *Densely Connected Convolutional Networks*. CVPR 2017. (DenseNet)
- Huang, G., Sun, Y., Liu, Z., Sedra, D., & Weinberger, K. Q. (2016). *Deep Networks with Stochastic Depth*. ECCV 2016.
- Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS 2017. (Transformer)
- Greff, K., Srivastava, R. K., & Schmidhuber, J. (2017). *Highway and Residual Networks Learn Unrolled Iterative Estimation*. ICLR 2017.
- Xiong, R., et al. (2020). *On Layer Normalization in the Transformer Architecture*. ICML 2020. (Pre-norm vs post-norm)
- He, K., Zhang, X., Ren, S., & Sun, J. (2015). *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification*. ICCV 2015. (He initialization)
- Russakovsky, O., et al. (2015). *ImageNet Large Scale Visual Recognition Challenge*. IJCV.
