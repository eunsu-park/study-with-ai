---
title: "Gradient-Based Learning Applied to Document Recognition"
authors: Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick Haffner
year: 1998
journal: "Proceedings of the IEEE, Vol. 86(11), pp. 2278–2324"
topic: Artificial Intelligence / Convolutional Neural Networks
tags: [CNN, LeNet-5, MNIST, convolutional network, weight sharing, sub-sampling, feature map, gradient-based learning, graph transformer network, document recognition]
status: completed
date_started: 2026-04-09
date_completed: 2026-04-09
---

# Gradient-Based Learning Applied to Document Recognition
**Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick Haffner (1998)**

---

## 핵심 기여 / Core Contribution

이 논문은 **Convolutional Neural Network (CNN)**의 결정판으로, 세 가지 측면에서 역사적 의미를 가집니다. 첫째, **LeNet-5** 아키텍처를 7층 구조(C1-S2-C3-S4-C5-F6-Output)로 상세히 제시하며, 국소 수용 영역(local receptive fields), 가중치 공유(shared weights), 서브샘플링(sub-sampling)이라는 세 가지 핵심 원리가 어떻게 이미지의 2D 구조적 지식을 네트워크에 내장하는지 설명합니다. 340,908개의 연결에도 불구하고 가중치 공유로 인해 60,000개의 파라미터만 학습되므로, 과적합 없이 높은 성능을 달성합니다. 둘째, **MNIST 데이터셋**을 만들고 선형 분류기, kNN, SVM, 다층 FC 네트워크, CNN 등 14가지 이상의 방법을 동일한 조건에서 체계적으로 비교하여, CNN이 0.95%의 최저 오류율을 달성함을 보입니다. 셋째, 여러 미분 가능한 모듈을 그래프로 연결하여 전체를 gradient-based로 학습하는 **Graph Transformer Network (GTN)** 개념을 도입합니다 — 이것은 현대 딥러닝 프레임워크(PyTorch, TensorFlow)의 computational graph의 직접적 선구입니다. LeNet-5는 NCR의 은행 수표 인식 시스템에 상용 배포되어 미국 내 여러 은행에서 매달 수백만 장의 수표를 읽었습니다.

This paper is the definitive work on **Convolutional Neural Networks (CNNs)**, with historical significance in three aspects. First, it presents the **LeNet-5** architecture in detail as a 7-layer structure (C1-S2-C3-S4-C5-F6-Output), explaining how three core principles — local receptive fields, shared weights, and sub-sampling — embed knowledge of 2D image structure into the network. Despite 340,908 connections, weight sharing yields only 60,000 trainable parameters, achieving high performance without overfitting. Second, it creates the **MNIST dataset** and systematically compares 14+ methods (linear classifiers, kNN, SVM, multi-layer FC networks, CNNs) under identical conditions, demonstrating CNN's best error rate of 0.95%. Third, it introduces **Graph Transformer Networks (GTN)** — connecting differentiable modules as a graph trained end-to-end — a direct precursor to modern deep learning frameworks' computational graphs. LeNet-5 was commercially deployed in NCR's bank check recognition system, reading millions of checks monthly across US banks.

---

## 읽기 노트 / Reading Notes

### Section I: Introduction — 패턴 인식의 패러다임 전환 / Paradigm Shift in Pattern Recognition

#### 전통적 접근: 수작업 특징 추출 + 학습 가능한 분류기 / Traditional: Hand-crafted Features + Trainable Classifier

논문은 전통적 패턴 인식 시스템이 두 모듈로 구성된다고 설명합니다 (Figure 1):

The paper explains that traditional pattern recognition systems consist of two modules (Figure 1):

1. **Feature Extractor**: 입력을 저차원 벡터로 변환. 대부분 **수작업(hand-crafted)**으로 설계. 도메인 지식이 집중되는 곳 / Transforms input to low-dimensional vectors. Mostly **hand-crafted**. Where domain knowledge concentrates
2. **Trainable Classifier**: 특징 벡터를 분류. 범용적이고 학습 가능 / Classifies feature vectors. General-purpose and trainable

이 접근의 근본적 문제: 인식 정확도가 **설계자의 특징 추출 능력**에 의존합니다. 새 문제마다 특징을 다시 설계해야 합니다.

The fundamental problem: recognition accuracy depends on the **designer's feature extraction ability**. Features must be redesigned for each new problem.

#### 논문의 핵심 메시지 / The Paper's Core Message

> "The main message of this paper is that better pattern recognition systems can be built by relying more on automatic learning, and less on hand-designed heuristics."

특징 추출기 자체를 학습 가능하게 만들면, 원시 픽셀에서 직접 패턴을 학습할 수 있습니다. CNN은 이 비전의 구현체입니다: **특징 추출과 분류를 하나의 학습 가능한 시스템으로 통합**합니다.

If the feature extractor itself is made learnable, patterns can be learned directly from raw pixels. CNN is the realization of this vision: **unifying feature extraction and classification into one trainable system**.

#### 세 가지 촉진 요인 / Three Enabling Factors

논문은 이 패러다임 전환을 가능하게 한 세 가지 요인을 지적합니다:

The paper identifies three factors enabling this paradigm shift:

1. **저렴한 컴퓨팅**: 브루트포스 수치적 방법에 의존 가능 / Cheap computing: Can rely on brute-force numerical methods
2. **대규모 데이터베이스**: 실제 데이터에 의존하여 시스템 구축 / Large databases: Build systems relying on real data
3. **강력한 기계학습 기법**: 고차원 입력을 처리하고 복잡한 결정 함수를 생성 / Powerful ML techniques: Handle high-dimensional inputs, generate complex decision functions

이 세 요인은 2012년 딥러닝 혁명에서 다시 한번 결정적 역할을 합니다 (GPU, ImageNet, CNN).

These three factors played a decisive role again in the 2012 deep learning revolution (GPUs, ImageNet, CNNs).

#### Structural Risk Minimization과 일반화 / SRM and Generalization

일반화 오류에 대한 핵심 공식:

The key formula for generalization error:

$$E_{test} - E_{train} = k\left(\frac{h}{P}\right)^\alpha$$

$h$는 모델의 "유효 용량(effective capacity)", $P$는 학습 샘플 수, $\alpha \in [0.5, 1.0]$. **용량 $h$를 줄이면서 $E_{train}$을 낮게 유지하는 것이 일반화의 핵심**입니다. CNN의 가중치 공유는 자유 파라미터를 줄여 $h$를 낮추면서도, 충분한 표현력을 유지합니다.

$h$ is the model's "effective capacity," $P$ is training samples, $\alpha \in [0.5, 1.0]$. **Reducing capacity $h$ while keeping $E_{train}$ low is the key to generalization.** CNN's weight sharing reduces free parameters (lowering $h$) while maintaining sufficient representational power.

논문은 흥미롭게도 큰 네트워크의 gradient descent가 **자체 정규화 효과**를 가진다고 추측합니다: 가중치 공간의 원점이 거의 모든 방향에서 매력적인 안장점이므로, 학습 초기에 가중치가 작아지면서 낮은 용량 모델처럼 행동하다가 점진적으로 용량을 높입니다 — Vapnik의 SRM 원리의 자연스러운 구현.

The paper interestingly conjectures that gradient descent in large networks has a **self-regularization effect**: the origin in weight space is an attractive saddle point in nearly every direction, so early training shrinks weights (behaving like a low-capacity model) then gradually increases capacity — a natural implementation of Vapnik's SRM principle.

#### Stochastic Gradient Descent의 우월성 / Superiority of SGD

논문은 SGD가 대규모 데이터에서 배치 gradient descent보다 우월함을 주장합니다:

The paper argues SGD is superior to batch gradient descent on large datasets:

$$W_k = W_{k-1} - \epsilon \frac{\partial E^{p_k}(W)}{\partial W}$$

이유: 대규모 학습 세트에서 많은 샘플이 중복적(redundant)이므로, 모든 샘플의 기울기를 평균내는 것보다 하나의 샘플로 빠르게 업데이트하는 것이 수렴이 빠릅니다. 논문은 부록 B에서 이를 상세히 분석합니다.

Reason: In large training sets, many samples are redundant, so updating quickly with one sample converges faster than averaging gradients over all samples. The paper analyzes this in detail in Appendix B.

---

### Section II: Convolutional Neural Networks — CNN의 완전한 이론 / Complete CNN Theory

#### 완전 연결 네트워크의 한계 / Limitations of Fully-Connected Networks

이미지를 완전 연결(FC) 네트워크에 입력하면 두 가지 근본적 문제가 있습니다:

Feeding images to fully-connected (FC) networks has two fundamental problems:

**1. 파라미터 폭발**: $28 \times 28 = 784$ 픽셀 입력에 100개 hidden unit이면 78,400개 가중치. 이 많은 파라미터는 과적합을 유발하고 더 많은 학습 데이터를 요구합니다.

**1. Parameter explosion**: 784 pixel input with 100 hidden units = 78,400 weights. Too many parameters cause overfitting and demand more training data.

**2. 이동 불변성(translation invariance) 부재**: FC 네트워크는 입력 변수의 순서에 무관합니다 — 픽셀을 임의로 섞어도 결과가 같습니다. 그러나 이미지는 강한 **2D 국소 구조(local structure)**를 가집니다: 인접 픽셀은 높은 상관관계를 가지며, 국소 패턴(에지, 코너)의 조합으로 물체를 인식합니다.

**2. No translation invariance**: FC networks are agnostic to input variable ordering — shuffling pixels gives the same result. But images have strong **2D local structure**: neighboring pixels are highly correlated, and objects are recognized by combinations of local patterns (edges, corners).

#### CNN의 세 가지 핵심 원리 / Three Core Principles of CNN

CNN은 세 가지 아키텍처적 아이디어로 이 문제를 해결합니다:

CNN solves these with three architectural ideas:

##### (1) Local Receptive Fields — 국소 수용 영역

각 뉴런은 이미지의 **작은 영역(예: 5×5)**에만 연결됩니다. 이렇게 하면:

Each neuron connects only to a **small region (e.g., 5×5)** of the image. This way:

- 에지, 코너 같은 **국소 특징(local features)**을 추출 / Extracts **local features** like edges, corners
- 이 특징들은 상위 층에서 결합되어 점점 더 복잡한 패턴이 됨 / These features are combined in higher layers to form increasingly complex patterns
- Hubel & Wiesel (1962)의 시각 피질 발견에서 영감 / Inspired by Hubel & Wiesel's (1962) visual cortex discovery

##### (2) Shared Weights (Weight Replication) — 가중치 공유

같은 feature map 내의 모든 유닛이 **동일한 가중치 벡터(필터)**를 공유합니다:

All units within a feature map share the **same weight vector (filter)**:

- 하나의 특징 검출기가 이미지의 **모든 위치**에서 동일한 패턴을 찾음 / One feature detector searches for the same pattern at **all locations**
- 이것이 **이동 불변성(translation invariance)**을 제공: 숫자 "7"이 왼쪽에 있든 오른쪽에 있든 동일한 필터가 검출 / Provides **translation invariance**: the filter detects "7" whether it's on the left or right
- 파라미터 수를 극적으로 줄임: 28×28 feature map에서 각 유닛이 독립적이면 784×25 = 19,600 파라미터, 공유하면 **25+1 = 26 파라미터** / Dramatically reduces parameters: independent would be 19,600, shared is **26 parameters**

가중치 공유의 부가적 이점: backpropagation을 수정하여 공유 가중치에 대한 기울기를 계산할 수 있습니다 — 각 연결에 대한 편미분을 계산한 후, 같은 파라미터를 공유하는 모든 연결의 편미분을 **합산**합니다.

Additional benefit of weight sharing: backpropagation is modified to compute gradients for shared weights — compute partial derivatives for each connection, then **sum** partials of all connections sharing the same parameter.

##### (3) Sub-sampling (Pooling) — 서브샘플링

특징이 검출되면 그 **정확한 위치**는 덜 중요해집니다. 중요한 것은 다른 특징과의 **상대적 위치**입니다:

Once a feature is detected, its **exact location** becomes less important. What matters is its **relative position** to other features:

- Sub-sampling 층은 해상도를 줄임 (LeNet-5: 2×2 영역의 평균) / Sub-sampling layers reduce resolution (LeNet-5: average of 2×2 regions)
- 이동과 변형에 대한 **강건성** 제공 / Provides **robustness** to shifts and distortions
- Feature map 수가 증가하면서 공간 해상도가 감소 → "bi-pyramid" 구조 / Feature maps increase while spatial resolution decreases → "bi-pyramid" structure

이 계층적 구조를 논문은 Fukushima의 Neocognitron (1980)에서 영감받은 Hubel & Wiesel의 "simple cell"과 "complex cell"의 교대에 비유합니다.

The paper compares this hierarchical structure to the alternation of Hubel & Wiesel's "simple cells" and "complex cells," inspired by Fukushima's Neocognitron (1980).

---

### Section II.B: LeNet-5 — 완전한 아키텍처 / Complete Architecture

LeNet-5는 7개의 학습 가능한 층으로 구성됩니다 (입력 제외):

LeNet-5 consists of 7 trainable layers (excluding input):

#### 입력 / Input: 32×32

MNIST 이미지는 28×28이지만, 32×32로 확장 (zero-padding). 이유: 가장 높은 레벨의 feature detector(C3의 수용 영역)의 중심이 입력 이미지의 **중앙 20×20 영역**에 놓이도록. 픽셀 값은 배경=-0.1, 전경=1.175로 정규화하여 평균≈0, 분산≈1이 되게 합니다 — 학습을 가속화합니다.

MNIST images are 28×28, extended to 32×32 (zero-padding). Reason: centers of highest-level feature detectors (C3's receptive fields) fall in the **central 20×20 area** of input. Pixel values normalized to background=-0.1, foreground=1.175 for mean≈0, variance≈1 — accelerates learning.

#### Layer C1: Convolutional, 6 feature maps @ 28×28

- 5×5 수용 영역, stride 1 / 5×5 receptive field, stride 1
- 파라미터: $(5 \times 5 + 1) \times 6 = 156$ / Parameters: 156
- 연결: $28 \times 28 \times 6 \times (5 \times 5 + 1) = 122,304$ / Connections: 122,304
- 각 feature map은 다른 종류의 국소 특징을 검출 / Each feature map detects a different type of local feature

#### Layer S2: Sub-sampling, 6 feature maps @ 14×14

- 2×2 비겹침 영역의 평균 × 학습 가능한 계수 + bias → 시그모이드 / Average of 2×2 non-overlapping region × trainable coefficient + bias → sigmoid
- 파라미터: $(1 + 1) \times 6 = 12$ / Parameters: 12
- 해상도를 절반으로 줄여 이동 불변성 강화 / Halves resolution, strengthening translation invariance

#### Layer C3: Convolutional, 16 feature maps @ 10×10

- 5×5 수용 영역, S2의 **부분적 연결(partial connections)** / 5×5 receptive field, **partial connections** from S2
- **Table I의 연결 패턴이 핵심**: 모든 S2 feature map에 연결하지 않는 이유:
  1. 연결 수를 적절하게 유지 / Keep connections manageable
  2. **대칭성을 깨서** 다른 feature map이 다른 특징을 추출하도록 강제 / **Break symmetry** to force different feature maps to extract different features
- 파라미터: 1,516 / Parameters: 1,516

**Table I의 연결 패턴 / Table I Connection Pattern:**

| C3 map | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S2 maps connected | 012 | 123 | 234 | 345 | 450 | 501 | 0123 | 1234 | 2345 | 3450 | 4501 | 0135 | 0124 | 0345 | 1245 | all |

처음 6개는 3개의 연속 S2 map, 다음 6개는 4개의 연속 map, 다음 3개는 4개의 불연속 map, 마지막 1개는 모든 S2 map에 연결됩니다.

First 6 connect to 3 contiguous S2 maps, next 6 to 4 contiguous, next 3 to 4 discontinuous, last 1 to all.

#### Layer S4: Sub-sampling, 16 feature maps @ 5×5

- S2와 동일한 방식 / Same mechanism as S2
- 파라미터: 32 / Parameters: 32

#### Layer C5: Convolutional, 120 feature maps @ 1×1

- 5×5 수용 영역으로 S4 전체(5×5)를 커버 → 사실상 FC / 5×5 receptive field covers all of S4 (5×5) → effectively FC
- S4의 16개 모든 feature map에 연결 / Connected to all 16 S4 feature maps
- "convolutional"로 명명한 이유: 입력이 32×32보다 커지면 feature map도 1×1보다 커지므로 / Named "convolutional" because: if input were larger than 32×32, feature maps would be larger than 1×1
- 파라미터: 48,120 / Parameters: 48,120

#### Layer F6: Fully connected, 84 units

- C5에서 F6로 완전 연결 / Fully connected from C5
- 활성화: $f(a) = A \tanh(Sa)$, $A = 1.7159$ / Activation: scaled tanh
- 왜 84? 출력 코드가 7×12 비트맵(ASCII 문자의 양식화된 이미지)이므로 / Why 84? Output codes are 7×12 bitmaps (stylized images of ASCII characters)
- 파라미터: 10,164 / Parameters: 10,164

#### Output: 10 Euclidean RBF units

- 각 유닛이 F6 출력과 해당 클래스의 "이상적 코드" 사이의 유클리드 거리를 계산 / Each unit computes Euclidean distance between F6 output and class's "ideal code"

$$y_i = \sum_j (x_j - w_{ij})^2$$

- 값이 작을수록 해당 클래스일 가능성이 높음 / Smaller value = more likely that class
- 이상적 코드는 7×12 비트맵으로 수작업 설계 (Figure 3) / Ideal codes are hand-designed 7×12 bitmaps (Figure 3)
- 유사한 문자(O/0, 1/l)가 유사한 코드를 가져 후처리에서 혼동 수정 가능 / Similar characters (O/0, 1/l) have similar codes, enabling confusion correction in post-processing

#### 활성화 함수 / Activation Function

$$f(a) = A \tanh(Sa), \quad A = 1.7159$$

$f(\pm 1) = \pm 1$이 되도록 설계. 이유: 시그모이드의 최대 곡률 점이 $\pm 1$이며, 이 점에서 가장 비선형적으로 작동. 포화(saturation)를 방지하여 학습을 가속화합니다.

Designed so $f(\pm 1) = \pm 1$. Reason: maximum curvature of sigmoid is at $\pm 1$, operating most non-linearly there. Prevents saturation, accelerating learning.

#### 전체 아키텍처 요약 / Architecture Summary

| 층 / Layer | 유형 / Type | Feature Maps | 크기 / Size | 파라미터 / Params | 연결 / Connections |
|---|---|---|---|---|---|
| Input | — | 1 | 32×32 | — | — |
| C1 | Conv 5×5 | 6 | 28×28 | 156 | 122,304 |
| S2 | AvgPool 2×2 | 6 | 14×14 | 12 | 5,880 |
| C3 | Conv 5×5 (partial) | 16 | 10×10 | 1,516 | 151,600 |
| S4 | AvgPool 2×2 | 16 | 5×5 | 32 | 2,000 |
| C5 | Conv 5×5 (=FC) | 120 | 1×1 | 48,120 | 48,120 |
| F6 | FC | — | 84 | 10,164 | 10,164 |
| Output | RBF | — | 10 | 840 | 840 |
| **Total** | | | | **~60,000** | **340,908** |

핵심: **340,908개 연결이지만 가중치 공유로 60,000개 파라미터만 학습**. 이것이 CNN의 정규화 효과입니다.

Key: **340,908 connections but only 60,000 trainable parameters due to weight sharing.** This is CNN's regularization effect.

---

### Section III: Results and Comparison — MNIST 벤치마크 / The MNIST Benchmark

#### MNIST 데이터셋 / The MNIST Dataset

논문이 만든 데이터셋으로, 딥러닝의 "Hello World"가 됩니다:

The dataset created by this paper, becoming deep learning's "Hello World":

- NIST Special Database 3 (Census Bureau 직원) + Special Database 1 (고등학생) / NIST SD-3 (Census Bureau employees) + SD-1 (high school students)
- 60,000 학습 + 10,000 테스트 이미지 / 60,000 training + 10,000 test images
- 28×28 grayscale, center of mass 정렬 / 28×28 grayscale, centered by center of mass
- 세 버전: regular (28×28), deslanted (20×20), 16×16 / Three versions

#### 핵심 결과 — Figure 9의 체계적 비교 / Key Results — Figure 9 Systematic Comparison

논문의 가장 중요한 그림인 Figure 9는 14가지 이상의 방법을 비교합니다:

The paper's most important figure, Figure 9, compares 14+ methods:

**전통적 방법 / Traditional methods:**

| 방법 / Method | 오류율 / Error |
|---|---|
| Linear classifier | 12.0% |
| Pairwise linear | 7.6% |
| K-NN Euclidean | 5.0% |
| K-NN Euclidean (deslanted) | 2.4% |
| PCA + quadratic | 3.3% |
| RBF (1000 units) | 3.6% |
| Tangent Distance (16×16) | 1.1% |
| **SVM poly 4** | **1.1%** |

**신경망 / Neural networks:**

| 방법 / Method | 오류율 / Error |
|---|---|
| 1-hidden FC (28×28-300-10) | 4.7% |
| 1-hidden FC (28×28-1000-10) | 4.5% |
| 2-hidden FC (28×28-300-100-10) | 3.05% |
| 2-hidden FC + distortion | 2.50% |
| LeNet-1 (16×16) | 1.7% |
| LeNet-4 | 1.1% |
| **LeNet-5** | **0.95%** |
| **LeNet-5 + distortion** | **0.8%** |
| Boosted LeNet-4 | **0.7%** |

**핵심 관찰 / Key observations:**

1. **CNN이 최고 성능**: LeNet-5(0.95%) > SVM(1.1%) > FC 네트워크(3.05%) > kNN(2.4%) > 선형(12.0%)
2. **가중치 공유의 효과**: FC 네트워크(28×28-300-100-10)는 파라미터가 더 많지만(~270,000) LeNet-5(~60,000)보다 3배 높은 오류율
3. **데이터 증강(distortion)의 효과**: LeNet-5의 0.95% → 0.8%, 학습 데이터 540,000장으로 확장
4. **SVM의 인상적 성능**: 도메인 지식 없이 1.1% — 논문 #8에서 확인한 결과와 일치
5. **부스팅의 효과**: LeNet-4 3개를 부스팅하면 0.7% — 최저 오류율

#### 82개 오분류 패턴 분석 / Analysis of 82 Misclassified Patterns (Figure 8)

LeNet-5가 오분류한 82개 패턴을 분석하면, 대부분이 **인간도 판별하기 어려운** 모호한 패턴이거나, **학습 세트에 과소 대표된 스타일**로 작성된 숫자입니다. 이것은 더 많은 학습 데이터로 개선될 수 있음을 시사합니다.

Analyzing LeNet-5's 82 misclassified patterns reveals most are either **genuinely ambiguous** (difficult even for humans) or digits written in **under-represented styles**. This suggests improvement with more training data.

#### 계산 비용과 메모리 / Computational Cost and Memory (Figures 11-12)

논문은 정확도뿐 아니라 **실용적 측면**도 비교합니다:

The paper compares not just accuracy but **practical aspects**:

- **LeNet-5**: ~401K multiply-adds, ~60K 파라미터 / ~401K multiply-adds, ~60K parameters
- **SVM poly 4**: ~14M multiply-adds (LeNet-5의 35배!) / ~14M multiply-adds (35× LeNet-5!)
- **kNN**: 24M multiply-adds 이상 / 24M+ multiply-adds
- CNN은 메모리와 계산 모두에서 가장 효율적 / CNN is most efficient in both memory and computation

**결론**: CNN은 정확도, 계산 비용, 메모리 모두에서 최적의 균형을 달성합니다.

**Conclusion**: CNN achieves the optimal balance in accuracy, computation, and memory.

---

### Section IV: Multi-Module Systems and GTN — 현대 딥러닝의 청사진 / Blueprint for Modern Deep Learning

#### Gradient-Based Learning의 일반화 / Generalizing Gradient-Based Learning

논문은 CNN을 넘어서, **미분 가능한 모듈의 임의적 조합**을 gradient로 학습할 수 있다고 주장합니다. 시스템을 모듈의 캐스케이드 $X_n = F_n(W_n, X_{n-1})$로 보면:

Beyond CNN, the paper argues that **arbitrary compositions of differentiable modules** can be trained with gradients. Viewing the system as a cascade of modules $X_n = F_n(W_n, X_{n-1})$:

$$\frac{\partial E^p}{\partial W_n} = \frac{\partial F}{\partial W}(W_n, X_{n-1}) \frac{\partial E^p}{\partial X_n}$$

$$\frac{\partial E^p}{\partial X_{n-1}} = \frac{\partial F}{\partial X}(W_n, X_{n-1}) \frac{\partial E^p}{\partial X_n}$$

이것은 **현대 딥러닝 프레임워크의 자동 미분(autograd)**과 정확히 같은 개념입니다: 각 모듈은 `forward()`와 `backward()` 메서드를 가지며, 체인 룰로 기울기를 역전파합니다.

This is exactly the same concept as **modern deep learning frameworks' autograd**: each module has `forward()` and `backward()` methods, and gradients are backpropagated via chain rule.

#### Graph Transformer Network (GTN) / 그래프 변환 네트워크

GTN은 모듈 간에 **그래프(directed acyclic graph)** 형태의 정보를 주고받는 시스템입니다. 이것은 고정 크기 벡터만 전달하는 전통적 신경망을 넘어:

GTN is a system where modules exchange information in **graph (directed acyclic graph)** form. Going beyond traditional neural networks that only pass fixed-size vectors:

- 가변 길이 입력(단어, 문장) 처리 가능 / Can handle variable-length inputs (words, sentences)
- 여러 해석 가능성을 동시에 유지 / Maintain multiple interpretations simultaneously
- Viterbi 알고리즘으로 최적 경로 선택 / Select optimal path with Viterbi algorithm

이 개념은 이후 HMM과 신경망의 결합, end-to-end 음성 인식, 그리고 현대의 encoder-decoder 구조의 선구가 됩니다.

This concept is a precursor to HMM-neural network combinations, end-to-end speech recognition, and modern encoder-decoder architectures.

---

### Section II.E: Invariance and Noise Resistance — 불변성과 노이즈 강건성

LeNet-5는 크기, 위치, 방향이 다양한 형태를 인식하는 데 특히 뛰어납니다:

LeNet-5 excels at recognizing shapes with varying size, position, and orientation:

- 스케일 변화 최대 2배까지 정확한 인식 / Accurate recognition up to 2× scale variation
- 수직 이동 약 문자 높이의 절반까지 / Vertical shift up to about half character height
- 회전 ±30도까지 / Rotation up to ±30 degrees
- Salt-and-pepper 노이즈(각 픽셀 10% 확률 반전)에서도 강건 / Robust to salt-and-pepper noise (10% pixel inversion)

Figure 13은 극단적으로 노이즈가 있거나 왜곡된 문자를 LeNet-5가 정확히 인식하는 인상적인 예시를 보여줍니다.

Figure 13 shows impressive examples of LeNet-5 correctly recognizing extremely noisy or distorted characters.

---

## 핵심 시사점 / Key Takeaways

1. **구조적 지식의 내장이 학습의 핵심이다**: LeNet-5의 세 원리(국소 수용 영역, 가중치 공유, 서브샘플링)는 "이미지가 2D 국소 구조를 가진다"는 사전 지식을 아키텍처에 내장합니다. 이것은 SVM이나 FC 네트워크가 할 수 없는 것이며, 같은 파라미터 수 대비 훨씬 뛰어난 성능의 원천입니다. 올바른 **귀납적 편향(inductive bias)**을 선택하는 것이 모든 머신러닝의 핵심입니다.

   **Embedding structural knowledge is the key to learning**: LeNet-5's three principles embed the prior knowledge that "images have 2D local structure" into the architecture. This is something SVMs or FC networks cannot do, and is the source of superior performance per parameter count. Choosing the right **inductive bias** is central to all machine learning.

2. **가중치 공유는 정규화와 효율성을 동시에 달성한다**: 340,908개 연결에서 60,000개 파라미터로의 축소는 SRM 관점에서 모델 용량을 제한하여 일반화를 개선합니다. 동시에 계산과 메모리도 줄입니다. 이 "제약이 곧 강점"이라는 통찰은 현대 딥러닝 설계 철학의 핵심입니다.

   **Weight sharing simultaneously achieves regularization and efficiency**: Reducing from 340,908 connections to 60,000 parameters limits model capacity (SRM perspective), improving generalization while also reducing computation and memory. This insight that "constraints are strengths" is core to modern deep learning design philosophy.

3. **MNIST는 단순한 데이터셋이 아니라 방법론적 기여다**: 500명의 필기자로부터 수집하고, 학습/테스트 세트를 공정하게 분할하고, 14가지 이상의 방법을 동일 조건에서 비교한 것은 당시 매우 이례적이었습니다. 이것은 기계학습 벤치마크 문화의 시작이며, 이후 CIFAR-10, ImageNet 등의 벤치마크로 이어집니다.

   **MNIST is not just a dataset but a methodological contribution**: Collecting from 500 writers, fairly splitting train/test, and comparing 14+ methods under identical conditions was exceptional at the time. This marked the beginning of ML benchmarking culture, leading to CIFAR-10, ImageNet, etc.

4. **CNN은 특징 추출과 분류를 통합한 최초의 실용적 시스템이다**: 전통적 "수작업 특징 + 분류기" 패러다임에서 "end-to-end 학습" 패러다임으로의 전환을 실증합니다. 원시 픽셀에서 직접 학습하면서도 수작업 시스템을 능가합니다.

   **CNN is the first practical system unifying feature extraction and classification**: It demonstrates the transition from "hand-crafted features + classifier" to "end-to-end learning." It learns directly from raw pixels yet surpasses hand-crafted systems.

5. **GTN은 현대 딥러닝 프레임워크의 청사진이다**: 미분 가능한 모듈의 그래프, `forward()`/`backward()` 패턴, 자동 미분 — 이 모든 것이 1998년에 이미 개념화되었습니다. PyTorch의 `nn.Module`은 GTN의 직접적 후계입니다.

   **GTN is the blueprint for modern deep learning frameworks**: Graphs of differentiable modules, `forward()`/`backward()` patterns, automatic differentiation — all conceptualized in 1998. PyTorch's `nn.Module` is a direct descendant of GTN.

6. **상용 배포는 학술적 성공을 넘어선 증명이다**: LeNet-5가 NCR의 은행 수표 인식 시스템에 배포되어 매달 수백만 장의 수표를 읽은 것은, CNN이 단순한 학술적 실험이 아니라 실제 산업적 가치를 가진 기술임을 증명합니다. 2012년 AlexNet의 ImageNet 혁명보다 14년이나 앞선 상용화입니다.

   **Commercial deployment is proof beyond academic success**: LeNet-5's deployment in NCR's bank check system reading millions of checks monthly proves CNN is not just an academic experiment but a technology with real industrial value — commercialization 14 years before AlexNet's 2012 ImageNet revolution.

7. **데이터 증강(distortion training)은 강력한 정규화 기법이다**: 60,000개 원본에 540,000개의 왜곡 이미지를 추가하여 오류를 0.95%에서 0.8%로 줄인 것은, 데이터 증강이 모델 복잡도 제어 못지않게 중요함을 보여줍니다. 이 기법은 현재도 거의 모든 컴퓨터 비전 시스템에서 사용됩니다.

   **Data augmentation (distortion training) is a powerful regularization technique**: Adding 540,000 distorted images to 60,000 originals reduced error from 0.95% to 0.8%, showing data augmentation is as important as model complexity control. This technique is still used in virtually all computer vision systems today.

---

## 수학적 요약 / Mathematical Summary

### CNN Forward Pass — LeNet-5 스타일 / LeNet-5 Style

**Convolutional Layer:**
$$\text{output}_{ij}^k = f\left(\sum_{m} \sum_{p,q} w_{pq}^{km} \cdot \text{input}_{(i+p)(j+q)}^{m} + b^k\right)$$

$k$: 출력 feature map 인덱스, $m$: 입력 feature map 인덱스, $(p,q)$: 커널 내 위치

**Sub-sampling Layer:**
$$\text{output}_{ij}^k = f\left(c^k \cdot \frac{1}{4}\sum_{p=0}^{1}\sum_{q=0}^{1} \text{input}_{(2i+p)(2j+q)}^{k} + b^k\right)$$

$c^k$: 학습 가능한 계수, $b^k$: 학습 가능한 bias

**Weight Sharing의 기울기 계산 / Gradient with Weight Sharing:**
$$\frac{\partial E}{\partial w_{\text{shared}}} = \sum_{\text{all connections sharing } w} \frac{\partial E}{\partial w_{\text{connection}}}$$

---

## 역사적 맥락의 타임라인 / Paper in the Arc of History

```
1962 ─── Hubel & Wiesel ─── 시각 피질의 수용 영역 발견
            │
1980 ─── Fukushima ─── Neocognitron (비지도 CNN 유사 구조)
            │
1986 ─── Rumelhart et al. ─── Backpropagation
            │
1989 ─── LeCun et al. ─── CNN + Backprop (우편번호 인식)
            │
1995 ─── Cortes & Vapnik ─── SVM (CNN의 경쟁자)
            │
     ╔═══════════════════════════════════════════╗
     ║  ★ 1998 ─── LeCun, Bottou, Bengio, Haffner║
     ║       LeNet-5 + MNIST + GTN               ║
     ║       CNN의 결정판                          ║
     ╚═══════════════════════════════════════════╝
            │
2006 ─── Hinton et al. ─── Deep Belief Nets
            │
2012 ─── Krizhevsky et al. ─── AlexNet (ImageNet 혁명)
            │         LeNet의 정신적 후계: ReLU, Dropout, GPU
            │
2014 ─── Simonyan & Zisserman ─── VGGNet (더 깊은 CNN)
            │
2015 ─── He et al. ─── ResNet (152층!)
            │
현재 ── CNN은 컴퓨터 비전의 표준 백본
```

---

## 다른 논문과의 연결 / Connections to Other Papers

| 논문 / Paper | 연결 / Connection |
|---|---|
| #3 Rosenblatt (1958) | Perceptron의 후계. Rosenblatt의 선형 분류기 → CNN의 비선형 계층적 특징 추출 / Perceptron's successor. Rosenblatt's linear classifier → CNN's non-linear hierarchical feature extraction |
| #6 Rumelhart et al. (1986) | Backpropagation이 CNN 학습의 기반. 가중치 공유를 위한 수정이 필요 / Backpropagation is CNN's training foundation, modified for weight sharing |
| #7 LeCun et al. (1989) | LeNet-1의 직접적 확장. 1989년의 우편번호 인식 → 1998년의 완전한 LeNet-5 / Direct extension of LeNet-1. 1989 zip code → 1998 complete LeNet-5 |
| #8 Cortes & Vapnik (1995) | 직접적 경쟁자. MNIST에서 SVM(1.1%) vs LeNet-5(0.95%). SVM은 도메인 지식 불필요, CNN은 더 효율적 / Direct competitor. SVM needs no domain knowledge, CNN is more efficient |
| #9 Hochreiter & Schmidhuber (1997) | LSTM은 시간적 구조, CNN은 공간적 구조를 학습. 나중에 CNN+LSTM이 비디오/음성에서 결합 / LSTM for temporal structure, CNN for spatial. Later combined for video/speech |
| #12 Hinton et al. (2006) | Deep Belief Nets가 딥러닝을 부활시켰지만, AlexNet은 LeNet의 아키텍처를 확장한 것 / DBN revived deep learning, but AlexNet extended LeNet's architecture |
| #13 Krizhevsky et al. (2012) | AlexNet = LeNet의 정신적 후계: 더 깊고, ReLU, Dropout, GPU. LeNet 없이 AlexNet은 없었음 / AlexNet = LeNet's spiritual successor: deeper, ReLU, Dropout, GPU. No LeNet, no AlexNet |
| #19 He et al. (2015) | ResNet은 CNN을 100+층으로 확장. LeNet의 "합성곱+풀링" 기본 구조를 유지 / ResNet extends CNN to 100+ layers, maintaining LeNet's basic conv+pool structure |

---

## 참고문헌 / References

- LeCun, Y., Bottou, L., Bengio, Y. & Haffner, P., "Gradient-Based Learning Applied to Document Recognition", *Proceedings of the IEEE*, 86(11), pp. 2278–2324, 1998.
- LeCun, Y. et al., "Backpropagation Applied to Handwritten Zip Code Recognition", *Neural Computation*, 1(4), pp. 541–551, 1989.
- Hubel, D.H. & Wiesel, T.N., "Receptive Fields, Binocular Interaction and Functional Architecture in the Cat's Visual Cortex", *Journal of Physiology*, 160, pp. 106–154, 1962.
- Fukushima, K., "Neocognitron: A Self-organizing Neural Network Model for a Mechanism of Pattern Recognition Unaffected by Shift in Position", *Biological Cybernetics*, 36, pp. 193–202, 1980.
- Cortes, C. & Vapnik, V., "Support-Vector Networks", *Machine Learning*, 20, pp. 273–297, 1995.
- Vapnik, V., *The Nature of Statistical Learning Theory*, Springer, 1995.
