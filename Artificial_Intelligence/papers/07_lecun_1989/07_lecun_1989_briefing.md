# Pre-Reading Briefing: Backpropagation Applied to Handwritten Zip Code Recognition (1989)
# 사전 읽기 브리핑: 손글씨 우편번호 인식에 적용된 역전파 (1989)

**Authors / 저자**: Y. LeCun, B. Boser, J. S. Denker, D. Henderson, R. E. Howard, W. Hubbard, L. D. Jackel
**Journal / 저널**: *Neural Computation*, Vol. 1, pp. 541–551, 1989
**Institution / 소속**: AT&T Bell Laboratories, Holmdel, NJ

---

## 핵심 기여 / Core Contribution

LeCun et al.은 Rumelhart et al.(1986)의 backpropagation을 **최초로 대규모 실제 문제** — 미국 우편 서비스의 손글씨 우편번호(zip code) 숫자 인식 — 에 성공적으로 적용했습니다. 핵심 혁신은 네트워크 아키텍처에 **과제 영역의 사전 지식(domain knowledge)** 을 제약 조건(constraint)으로 통합한 것입니다: (1) **국소 수용 영역(local receptive fields)** — 각 뉴런이 이미지의 작은 영역만 봄, (2) **가중치 공유(weight sharing)** — 같은 특징 감지기를 이미지 전체에서 재사용, (3) **부분표본화(subsampling)** — 해상도를 점진적으로 줄여 위치 불변성 확보. 이 세 가지가 결합된 구조가 바로 **Convolutional Neural Network(CNN)** 이며, 이 논문은 CNN을 실용적으로 적용한 최초의 논문입니다. 64,660개 연결이지만 자유 파라미터는 9,760개에 불과 — "제약된 backpropagation(constrained backpropagation)"이 핵심입니다.

LeCun et al. were the first to successfully apply Rumelhart et al.'s (1986) backpropagation to a **large-scale real-world problem** — handwritten zip code digit recognition from the U.S. Postal Service. The key innovation is integrating **domain knowledge** into the network architecture as constraints: (1) **local receptive fields** — each neuron sees only a small image region, (2) **weight sharing** — reusing the same feature detector across the entire image, (3) **subsampling** — progressively reducing resolution for position invariance. This combined structure is the **Convolutional Neural Network (CNN)**, and this paper is the first practical application of CNNs. 64,660 connections but only 9,760 free parameters — "constrained backpropagation" is the key.

---

## 역사적 맥락 / Historical Context

```
1962  Hubel & Wiesel ────── 고양이 시각 피질의 수용 영역 발견
  │                         Receptive fields in cat's visual cortex
  │
1980  Fukushima ─────────── Neocognitron: 학습 없는 CNN의 선조
  │                         CNN ancestor without learning
  │
1986  Rumelhart et al. ──── Backpropagation 대중화 (Nature)
  │                         Backpropagation popularized
  │
  ╔════════════════════════════════════════════════════════════╗
  ║ ★ 1989  LeCun et al. ★                                    ║
  ║  Backprop + CNN → 최초의 실용적 딥러닝 시스템               ║
  ║  First practical deep learning system                      ║
  ║  5% error on real zip codes, 10+ digits/sec                ║
  ╚════════════════════════════════════════════════════════════╝
  │
1998  LeCun et al. ──────── LeNet-5: 이 논문의 완성판
  │                         Definitive version of this paper
  │
2012  Krizhevsky et al. ──── AlexNet: GPU + 더 깊은 CNN → 빅뱅
                              Deeper CNN on GPU → big bang
```

**왜 이 논문이 특별한가 / Why this paper is special:**

1. **이론에서 실용으로의 도약**: Backpropagation은 1986년에 소개되었지만 작은 장난감 문제(XOR, 대칭 감지)에만 적용되었습니다. 이 논문은 9,298개의 **실제** 손글씨 숫자 이미지에 적용하여, 딥러닝이 현실 문제를 풀 수 있음을 증명했습니다.

   Backpropagation was introduced in 1986 but only applied to toy problems. This paper applied it to 9,298 **real** handwritten digit images, proving deep learning can solve real problems.

2. **"제약이 곧 지식이다"**: 자유 파라미터를 줄이면(64,660 연결 → 9,760 파라미터) 일반화가 향상된다는 핵심 설계 원리. 이것은 현대 정규화(regularization) 이론의 실용적 선조입니다.

   "Constraints are knowledge": reducing free parameters (64,660 connections → 9,760 parameters) improves generalization. A practical ancestor of modern regularization theory.

3. **생물학적 영감의 공학적 구현**: Hubel & Wiesel(1962)의 시각 피질 수용 영역 → feature map + weight sharing으로 공학적 구현.

   Engineering implementation of biological inspiration: Hubel & Wiesel's visual cortex receptive fields → feature maps + weight sharing.

---

## 필요한 배경 지식 / Prerequisites

### 1. 이전 논문에서 알아야 할 것 / From previous papers

| 논문 / Paper | 필요한 개념 / Needed concept |
|---|---|
| #6 Rumelhart et al. (1986) | Backpropagation 알고리즘 전체 — 이 논문은 backprop을 **적용**하는 논문이므로 알고리즘 자체는 알고 있어야 함 / The full backprop algorithm — this paper **applies** it |
| #4 Minsky & Papert (1969) | 단층의 한계 → 다층이 필요하다는 동기 / Single-layer limits → motivation for multi-layer |

### 2. 합성곱 연산 / Convolution operation

- **2D 합성곱 (2D Convolution)**: 작은 필터(커널)를 이미지 위에서 슬라이딩하며 가중합을 계산합니다. 이 논문에서는 5×5 커널을 사용합니다.

  A small filter (kernel) slides over the image computing weighted sums. This paper uses 5×5 kernels.

  ```
  Input image:        Kernel (5×5):       Output:
  ┌─────────────┐    ┌─────┐             ┌─────────┐
  │ . . . . . . │    │ w w │             │ o o o . │
  │ . [x x x] . │ *  │ w w │     →      │ o o o . │
  │ . [x x x] . │    │ w w │             │ . . . . │
  │ . . . . . . │    └─────┘             └─────────┘
  └─────────────┘
  ```

- **Stride**: 커널이 한 번에 이동하는 픽셀 수. 이 논문에서는 stride 2 — 출력 크기가 입력의 절반.

  Number of pixels the kernel moves per step. This paper uses stride 2 — output size is half of input.

### 3. 일반화와 과적합 / Generalization and overfitting

- **과적합(Overfitting)**: 학습 데이터에는 잘 맞지만 새 데이터에는 성능이 떨어지는 현상. 파라미터가 많을수록 위험 증가.

  Model fits training data well but performs poorly on new data. More parameters = higher risk.

- **VC 차원(Vapnik-Chervonenkis dimension)**: 모델의 "복잡도"를 측정하는 이론적 지표. 자유 파라미터가 적을수록 VC 차원이 낮고, 일반화가 좋음.

  Theoretical measure of model "complexity." Fewer free parameters = lower VC dimension = better generalization.

---

## 핵심 용어 / Key Vocabulary

| 용어 / Term | 직관적 설명 / Intuitive explanation |
|---|---|
| **Feature map** | 하나의 특징 감지기(커널)가 이미지 전체를 스캔한 결과. 예: "세로선 감지기"가 만든 feature map은 세로선이 있는 위치에서 밝음 / Result of scanning the entire image with one feature detector (kernel). E.g., a "vertical line detector" produces a map bright where vertical lines exist |
| **Weight sharing** | 같은 커널(가중치 세트)을 이미지의 모든 위치에서 공유. 64개 위치 × 25개 가중치 = 1,600 연결이지만 자유 파라미터는 25개뿐 / Same kernel (weight set) shared across all image positions. 64 positions × 25 weights = 1,600 connections but only 25 free parameters |
| **Local receptive field** | 각 뉴런이 이미지의 **전체가 아닌 작은 영역(5×5)** 만 봄. Hubel & Wiesel의 시각 피질 수용 영역에서 영감 / Each neuron sees only a **small region (5×5)**, not the whole image. Inspired by Hubel & Wiesel's visual cortex receptive fields |
| **Subsampling** | 해상도를 줄여 위치의 정확한 정보를 버리고 대략적 위치만 유지. "특징이 있는가"가 중요하지 "정확히 어디에"는 덜 중요 / Reducing resolution, discarding exact position info and keeping approximate position. "Is the feature present?" matters more than "exactly where?" |
| **Constrained backprop** | Backprop을 그대로 사용하되, 가중치 공유 등의 **제약**을 부여하여 자유 파라미터를 줄임. 이것이 CNN이 일반화를 잘 하는 비결 / Standard backprop with **constraints** like weight sharing to reduce free parameters. The secret to CNN's generalization |
| **Place coding** | 출력 10개 유닛 중 하나만 활성화되는 인코딩. 숫자 "3" → 4번째 유닛만 활성 / Encoding where only one of 10 output units is active. Digit "3" → only 4th unit active |
| **Scaled hyperbolic tangent** | 이 논문의 활성화 함수: $f(x) = A \tanh(Sx)$. Sigmoid 대신 사용 — 대칭이라 수렴이 빠름 / This paper's activation: $f(x) = A\tanh(Sx)$. Used instead of sigmoid — symmetric, faster convergence |

---

## 네트워크 아키텍처 미리보기 / Network Architecture Preview

이 논문의 핵심은 수식보다 **아키텍처**입니다. Fig. 3의 구조:

The core of this paper is the **architecture** rather than equations. Structure from Fig. 3:

```
입력 / Input:     16 × 16 = 256 유닛 (정규화된 숫자 이미지)
                   (normalized digit image, grayscale -1 to 1)
        │
        ▼
H1:     12 feature maps × 8 × 8 = 768 유닛
        커널: 5×5, stride 2 / Kernel: 5×5, stride 2
        연결: ~20,000 / 자유 파라미터: 1,068
        (Connections: ~20,000 / Free params: 1,068)
        │
        ▼
H2:     12 feature maps × 4 × 4 = 192 유닛
        커널: 5×5×8 (H1의 8개 map에서 입력) / stride 2
        연결: ~40,000 / 자유 파라미터: 2,592
        │
        ▼
H3:     30 유닛 (fully connected to H2)
        연결: ~6,000 / 자유 파라미터: 5,790
        │
        ▼
출력 / Output: 10 유닛 (place coding, 0–9)
        연결: ~300 / 자유 파라미터: 310
```

**핵심 수치 / Key numbers:**
- 총 유닛: 1,256
- 총 연결: **64,660**
- 자유 파라미터: **9,760** (연결 수의 15%!)
- 학습 데이터: 7,291개 / 테스트: 2,007개
- 결과: 테스트 오류율 **5.0%** (102개 오분류)
- 처리 속도: 30+ 분류/초 (DSP 칩)

---

## 핵심 설계 원리 미리보기 / Key Design Principles Preview

### 원리 1: "자유 파라미터를 줄여 일반화를 높여라"

$$\text{일반화 성능} \propto \frac{1}{\text{자유 파라미터 수}}$$

fully connected 네트워크(10,690 연결, 40 hidden units)는 테스트 오류 8.1%. CNN(64,660 연결, 9,760 파라미터)은 5.0%. **연결이 6배 많지만 파라미터가 적어서 일반화가 훨씬 좋음.**

Fully connected (10,690 connections, 40 hidden): 8.1% test error. CNN (64,660 connections, 9,760 params): 5.0%. **6× more connections but fewer parameters → much better generalization.**

### 원리 2: Weight sharing = 합성곱

같은 feature map 내의 모든 유닛이 **동일한 25개 가중치**(5×5 커널)를 공유합니다. 이것은 수학적으로 **합성곱(convolution)** 연산과 동일합니다. 이미지의 어디에 있든 같은 특징을 감지합니다.

All units in the same feature map share the **same 25 weights** (5×5 kernel). Mathematically identical to **convolution**. Detects the same feature regardless of position.

### 원리 3: Stochastic gradient > Batch gradient

논문은 **stochastic gradient(패턴 하나마다 갱신)** 이 batch gradient(전체 데이터 후 갱신)보다 "훨씬 빠르게 수렴하고, 더 강건한 해를 찾는다"고 보고합니다. 이것은 현대 딥러닝의 SGD/mini-batch의 직접적 선조입니다.

The paper reports that **stochastic gradient (update after each pattern)** converges "much faster than true gradient and finds more robust solutions." Direct ancestor of modern SGD/mini-batch.

---

## 읽기 가이드 / Reading Guide

논문은 11페이지로, 수식보다 **아키텍처 설계의 동기와 근거**가 핵심입니다.

The paper is 11 pages; the core is the **motivation and rationale for architecture design** rather than equations.

1. **§1 Introduction** — 설계 철학: 파라미터를 줄여 일반화 향상 / Design philosophy
2. **§2 Zip Codes** — 데이터: 9,298개 실제 우편번호 숫자, 16×16 정규화 / Real data
3. **§3.2 Feature Maps and Weight Sharing** — 가장 중요! CNN의 핵심 아이디어 3가지 / Most important! Three core CNN ideas
4. **§3.3 Network Architecture** — Fig. 3의 구조 상세 / Detailed structure
5. **§4 Experimental Environment** — Stochastic gradient, tanh 활성화, LeCun 초기화 / Training details
6. **§5 Results** — 5.0% 오류, fully connected 대비 성능 비교 / Results and comparison
7. **§6 Conclusion** — "제약된 backpropagation"의 의의 / Significance of constrained backprop

**특히 주의할 점 / Pay special attention to:**
- §3.2의 weight sharing 개념 — 왜 같은 특징 감지기를 여러 위치에서 공유하는 것이 합리적인가
- Fig. 3의 파라미터 수 계산 — 64,660 연결 vs 9,760 파라미터의 차이가 어디서 오는가
- §5의 fully connected vs CNN 비교 — 연결 수가 아니라 **자유 파라미터 수**가 일반화를 결정
- 학습된 커널이 Hubel & Wiesel의 생물학적 특징 감지기와 유사하다는 관찰

- Weight sharing concept in §3.2 — why sharing the same detector across positions is rational
- Parameter count in Fig. 3 — where the gap between 64,660 connections and 9,760 parameters comes from
- FC vs CNN comparison in §5 — **free parameter count**, not connection count, determines generalization
- Observation that learned kernels resemble Hubel & Wiesel's biological feature detectors
