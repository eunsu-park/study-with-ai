---
title: "Gradient-Based Learning Applied to Document Recognition — Pre-reading Briefing"
paper: "Gradient-Based Learning Applied to Document Recognition"
authors: Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick Haffner
year: 1998
journal: "Proceedings of the IEEE, Vol. 86(11), pp. 2278–2324"
type: briefing
date: 2026-04-09
---

# Gradient-Based Learning Applied to Document Recognition — 사전 읽기 브리핑 / Pre-reading Briefing

## 핵심 기여 / Core Contribution

이 논문은 **Convolutional Neural Network (CNN)**의 결정판으로, **LeNet-5** 아키텍처를 상세히 제시하고 MNIST 벤치마크에서 다른 모든 방법과 체계적으로 비교합니다. LeNet-5는 세 가지 핵심 아이디어를 결합합니다: (1) **국소 수용 영역(local receptive fields)** — 각 뉴런이 이미지의 작은 영역만 관찰하여 에지, 코너 등 국소 특징을 추출; (2) **가중치 공유(shared weights/weight replication)** — 같은 필터가 이미지 전체에 적용되어 위치 불변성을 제공하고 파라미터 수를 극적으로 줄임; (3) **서브샘플링(sub-sampling)** — 해상도를 점진적으로 줄여 이동/변형에 대한 강건성을 확보. 340,908개의 연결이 있지만 가중치 공유로 인해 실제 학습 가능한 파라미터는 단 60,000개입니다. MNIST에서 0.95% 오류율을 달성하며, distortion으로 학습 시 0.8%까지 낮아집니다. 또한 이 논문은 여러 모듈을 그래프로 연결하여 전체를 gradient-based로 학습하는 **Graph Transformer Network (GTN)** 개념을 도입하며, 실제 은행 수표 인식 시스템에 LeNet-5를 상용 배포한 사례를 제시합니다.

This paper is the definitive work on **Convolutional Neural Networks (CNNs)**, presenting the **LeNet-5** architecture in full detail and systematically comparing it against all other methods on the MNIST benchmark. LeNet-5 combines three key ideas: (1) **local receptive fields** — each neuron observes only a small region of the image, extracting local features like edges and corners; (2) **shared weights (weight replication)** — the same filter is applied across the entire image, providing translation invariance and dramatically reducing parameters; (3) **sub-sampling** — progressively reducing resolution for robustness to shifts and distortions. Despite 340,908 connections, weight sharing yields only 60,000 trainable parameters. On MNIST, it achieves 0.95% error, dropping to 0.8% with distortion training. The paper also introduces **Graph Transformer Networks (GTN)** — connecting multiple modules as a graph trained end-to-end with gradient-based methods — and presents the commercial deployment of LeNet-5 in a bank check recognition system reading millions of checks per month.

---

## 역사적 맥락 / Historical Context

```
1962 ─── Hubel & Wiesel ─── 시각 피질의 국소 수용 영역 발견
            │         CNN의 생물학적 영감
            │
1980 ─── Fukushima ─── Neocognitron
            │         최초의 CNN 유사 아키텍처 (비지도)
            │
1986 ─── Rumelhart et al. ─── Backpropagation
            │         다층 신경망 학습의 시작
            │
1989 ─── LeCun et al. ─── CNN으로 우편번호 인식 (논문 #7)
            │         최초의 backprop+CNN 성공 사례
            │
1995 ─── Cortes & Vapnik ─── SVM (논문 #8)
            │         CNN의 주요 경쟁자
            │
1997 ─── Hochreiter & Schmidhuber ─── LSTM (논문 #9)
            │         시퀀스를 위한 해결. CNN은 공간을 위한 해결
            │
     ╔═══════════════════════════════════════════╗
     ║  ★ 1998 ─── LeCun, Bottou, Bengio, Haffner║
     ║       LeNet-5 + MNIST + GTN               ║
     ║       CNN의 결정판, 현대 딥러닝의 청사진      ║
     ╚═══════════════════════════════════════════╝
            │
2006 ─── Hinton ─── Deep Belief Nets → 딥러닝 부활
            │
2012 ─── Krizhevsky et al. ─── AlexNet
            │         LeNet의 정신적 후계자, ImageNet 혁명
            │
현재 ── CNN은 컴퓨터 비전의 표준, LeNet은 모든 CNN의 시조
```

1998년은 CNN과 SVM이 MNIST에서 경쟁하던 시기입니다. LeNet-5(0.95%)는 SVM(1.1%)보다 우수했지만, SVM은 도메인 지식 없이도 비슷한 성능을 달성했습니다. 이 논문은 CNN의 우월성을 실험적으로 입증하면서도, 두 방법의 장단점을 공정하게 비교합니다.

1998 was when CNNs and SVMs competed on MNIST. LeNet-5 (0.95%) outperformed SVM (1.1%), but SVM achieved similar performance without domain knowledge. This paper experimentally demonstrates CNN's superiority while fairly comparing both methods' strengths and weaknesses.

---

## 필요한 배경 지식 / Prerequisites

### 1. 합성곱 연산 (Convolution) / Convolution Operation
- 커널(필터)을 이미지 위에서 슬라이딩하며 원소별 곱의 합을 계산
- 출력 = $\sum_{i,j} \text{kernel}(i,j) \cdot \text{input}(x+i, y+j) + \text{bias}$
- 에지 검출, 블러링 등 이미지 처리의 기본 연산

### 2. 이전 논문의 개념 / Concepts from Prior Papers
- **Backpropagation (논문 #6)**: CNN의 학습 알고리즘. 가중치 공유를 위한 수정 필요
- **CNN 기초 (논문 #7)**: LeNet-1의 개념. LeNet-5는 이것의 확장/완성
- **SVM (논문 #8)**: 주요 비교 대상. 커널 + 마진 최대화 vs 합성곱 + 가중치 공유
- **LSTM (논문 #9)**: 시간적 구조 학습. CNN은 공간적 구조 학습

### 3. Feature Map과 채널 / Feature Maps and Channels
- 하나의 합성곱 층은 여러 개의 feature map(채널)을 출력
- 각 feature map은 하나의 필터가 이미지 전체에 적용된 결과
- 여러 feature map = 여러 종류의 특징(에지, 코너 등)을 동시 추출

### 4. Structural Risk Minimization / 구조적 위험 최소화
- Vapnik의 이론: $E_{test} \leq E_{train} + \text{complexity penalty}$
- CNN의 가중치 공유는 자유 파라미터를 줄여 complexity를 낮춤
- 이것이 CNN이 과적합에 강한 이유

---

## 핵심 용어 / Key Vocabulary

| 용어 / Term | 직관적 설명 / Intuitive Explanation |
|---|---|
| **Feature Map** | 하나의 필터가 이미지 전체를 스캔한 결과물. "이 위치에 이 패턴이 얼마나 있는가"를 나타내는 2D 맵 / Result of one filter scanning the entire image. A 2D map showing "how much of this pattern exists at each location" |
| **Receptive Field** | 하나의 뉴런이 보는 입력 영역. LeNet-5에서는 5×5 / The input region one neuron observes. 5×5 in LeNet-5 |
| **Weight Sharing** | 같은 feature map 내의 모든 뉴런이 동일한 가중치(필터)를 공유. 파라미터 수를 극적으로 줄이고 이동 불변성 제공 / All neurons in a feature map share identical weights (filter). Dramatically reduces parameters and provides translation invariance |
| **Sub-sampling (Pooling)** | 해상도를 줄이는 연산. LeNet-5에서는 2×2 영역의 평균 → 학습 가능한 계수 × 결과 + bias / Resolution reduction. In LeNet-5: average of 2×2 region × trainable coefficient + bias |
| **MNIST** | 28×28 필기 숫자 이미지 데이터셋. 60,000 학습 + 10,000 테스트. 이 논문에서 만든 딥러닝의 표준 벤치마크 / 28×28 handwritten digit dataset. 60K train + 10K test. The standard deep learning benchmark created by this paper |
| **LeNet-5** | 7층 CNN: C1→S2→C3→S4→C5→F6→Output. 60,000 파라미터 / 7-layer CNN with 60,000 parameters |
| **RBF Output** | 출력층이 Euclidean RBF로 구성. 각 클래스의 "이상적 패턴"과의 거리를 계산 / Output layer composed of Euclidean RBFs, computing distance to each class's "ideal pattern" |
| **Graph Transformer Network (GTN)** | 여러 미분 가능한 모듈을 그래프로 연결하여 end-to-end 학습하는 패러다임. 현대 딥러닝 프레임워크의 선구 / Paradigm of connecting differentiable modules as a graph for end-to-end training. Precursor to modern deep learning frameworks |
| **Stochastic Gradient Descent (SGD)** | 미니배치가 아닌 단일 샘플로 기울기를 근사하여 업데이트. 논문에서는 이것이 대규모 데이터에서 더 빠르다고 주장 / Approximating gradients with single samples instead of mini-batches. Paper argues this is faster for large datasets |

---

## 수식 미리보기 / Equations Preview

### 1. 일반화 오류 상계 / Generalization Error Bound

$$E_{test} - E_{train} = k\left(\frac{h}{P}\right)^\alpha$$

$P$: 학습 샘플 수, $h$: 모델의 "유효 용량(effective capacity)", $\alpha \in [0.5, 1.0]$, $k$: 상수. **가중치 공유로 $h$를 줄이면 일반화 오류가 감소합니다.**

$P$: training samples, $h$: "effective capacity," $\alpha \in [0.5, 1.0]$, $k$: constant. **Reducing $h$ through weight sharing decreases generalization error.**

### 2. SGD 업데이트 규칙 / SGD Update Rule

$$W_k = W_{k-1} - \epsilon \frac{\partial E^{p_k}(W)}{\partial W}$$

단일 샘플 $p_k$에 대한 기울기로 업데이트. 대규모 데이터셋에서 정규 gradient descent보다 훨씬 빠릅니다.

Update using gradient from single sample $p_k$. Much faster than regular gradient descent on large datasets.

### 3. LeNet-5의 활성화 함수 / LeNet-5's Activation Function

$$f(a) = A \tanh(Sa)$$

$A = 1.7159$, $S$는 기울기. $f(\pm 1) = \pm 1$로, 시그모이드의 포화 문제를 줄이기 위해 설계되었습니다.

$A = 1.7159$, $S$ determines slope. Designed so $f(\pm 1) = \pm 1$, reducing sigmoid saturation.

### 4. RBF 출력 / RBF Output

$$y_i = \sum_j (x_j - w_{ij})^2$$

각 출력 RBF 유닛은 F6 출력 벡터와 해당 클래스의 "이상적 코드" 사이의 **유클리드 거리의 제곱**을 계산합니다. 작을수록 해당 클래스일 가능성이 높습니다.

Each output RBF unit computes the **squared Euclidean distance** between F6's output vector and that class's "ideal code." Smaller = more likely to be that class.

### 5. MAP Loss Function / MAP 손실 함수

$$E(W) = \frac{1}{P}\sum_{p=1}^{P}\left(y_{D_p}(Z^p, W) + \log\left(e^{-j} + \sum_i e^{-y_i(Z^p, W)}\right)\right)$$

MSE 대신 이 discriminative 손실을 사용하면, 정답 클래스의 penalty를 줄이면서 오답 클래스의 penalty를 **능동적으로 올립니다**.

Using this discriminative loss instead of MSE actively **pulls up** penalties for incorrect classes while pushing down the correct class.

---

## LeNet-5 아키텍처 요약 / LeNet-5 Architecture Summary

```
Input: 32×32 (1 channel, grayscale)
  │
  ▼
C1: 6 feature maps @ 28×28 (5×5 conv, 156 params)
  │
  ▼
S2: 6 feature maps @ 14×14 (2×2 avg pool, 12 params)
  │
  ▼
C3: 16 feature maps @ 10×10 (5×5 conv, partial connections, 1,516 params)
  │
  ▼
S4: 16 feature maps @ 5×5 (2×2 avg pool, 32 params)
  │
  ▼
C5: 120 feature maps @ 1×1 (5×5 conv = FC, 48,120 params)
  │
  ▼
F6: 84 units (fully connected, 10,164 params)
  │
  ▼
Output: 10 RBF units (84 inputs each)

Total: 340,908 connections, ~60,000 trainable parameters
```

---

## 읽기 팁 / Reading Tips

1. **Section II가 CNN의 본질**: local receptive fields, weight sharing, sub-sampling의 동기와 수학. Figure 2의 LeNet-5 다이어그램을 완전히 이해하는 것이 핵심입니다.
2. **Section III.C (Figure 9)가 핵심 결과**: 모든 방법의 MNIST 오류율 비교 막대 그래프. 이 한 장의 그림이 논문의 가치를 증명합니다.
3. **Table I (p.8)의 C3 연결 패턴**: C3가 S2의 모든 feature map에 연결되지 않는 이유 — 대칭성을 깨서 다양한 특징을 추출하도록 강제합니다.
4. **Section IV의 GTN 개념**: 현대 딥러닝 프레임워크(PyTorch, TensorFlow)의 "computational graph"와 직접 연결됩니다.

1. **Section II is the essence of CNNs**: Motivation and math of local receptive fields, weight sharing, sub-sampling. Fully understanding Figure 2's LeNet-5 diagram is the key.
2. **Section III.C (Figure 9) is the key result**: Bar chart comparing MNIST error rates across all methods. This single figure proves the paper's value.
3. **Table I (p.8) C3 connection pattern**: Why C3 doesn't connect to all S2 feature maps — forces diversity by breaking symmetry.
4. **Section IV's GTN concept**: Directly connects to modern deep learning frameworks' (PyTorch, TensorFlow) "computational graphs."
