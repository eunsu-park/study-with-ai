---
title: "Backpropagation Applied to Handwritten Zip Code Recognition"
authors: Y. LeCun, B. Boser, J. S. Denker, D. Henderson, R. E. Howard, W. Hubbard, L. D. Jackel
year: 1989
journal: "Neural Computation, Vol. 1, pp. 541–551"
topic: Artificial Intelligence / Convolutional Neural Networks
tags: [CNN, convolution, weight sharing, feature map, local receptive field, subsampling, handwriting recognition, constrained backpropagation, stochastic gradient descent]
status: completed
date_started: 2026-04-09
date_completed: 2026-04-09
---

# Backpropagation Applied to Handwritten Zip Code Recognition (1989)
# 손글씨 우편번호 인식에 적용된 역전파 (1989)

**Y. LeCun, B. Boser, J. S. Denker, D. Henderson, R. E. Howard, W. Hubbard, L. D. Jackel**
AT&T Bell Laboratories, Holmdel, NJ

---

## Core Contribution / 핵심 기여

LeCun et al.은 backpropagation(Rumelhart et al., 1986)을 **최초로 대규모 실제 문제** — 미국 우편 서비스의 손글씨 우편번호 숫자 인식(9,298개 이미지) — 에 성공적으로 적용했습니다. 핵심 혁신은 네트워크 아키텍처에 과제 영역의 **사전 지식(domain knowledge)** 을 제약 조건으로 통합한 것입니다: (1) **국소 수용 영역(local receptive fields)** — Hubel & Wiesel(1962)의 시각 피질에서 영감받아 각 뉴런이 5×5 영역만 봄, (2) **가중치 공유(weight sharing)** — 동일한 특징 감지기를 이미지 전체에서 재사용하여 합성곱(convolution)을 구현, (3) **부분표본화(subsampling)** — stride 2로 해상도를 점진적으로 줄여 위치 불변성 확보. 이 결과 64,660개 연결이지만 자유 파라미터는 **9,760개**(15%)에 불과하여, 10,690개 연결의 fully connected 네트워크(8.1% 오류)보다 **훨씬 좋은 일반화**(5.0% 오류)를 달성했습니다. 이것이 **Convolutional Neural Network(CNN)** 의 최초 실용적 적용이며, 현대 컴퓨터 비전의 직접적 출발점입니다.

LeCun et al. were the first to successfully apply backpropagation (Rumelhart et al., 1986) to a **large-scale real-world problem** — handwritten zip code digit recognition (9,298 images) from the U.S. Postal Service. The key innovation was integrating **domain knowledge** into the network architecture as constraints: (1) **local receptive fields** — inspired by Hubel & Wiesel's (1962) visual cortex, each neuron sees only a 5×5 region, (2) **weight sharing** — reusing the same feature detector across the image, implementing convolution, (3) **subsampling** — stride 2 progressively reduces resolution for position invariance. This resulted in 64,660 connections but only **9,760 free parameters** (15%), achieving **far better generalization** (5.0% error) than a fully connected network with 10,690 connections (8.1% error). This is the first practical application of the **Convolutional Neural Network (CNN)** and the direct starting point of modern computer vision.

---

## Reading Notes / 읽기 노트

### §1 Introduction (p.541) — 설계 철학

**핵심 원리: 제약을 통한 일반화 향상 / Core principle: better generalization through constraints:**

논문의 첫 문장이 전체를 요약합니다: "학습 네트워크의 일반화 능력은 과제 영역의 제약 조건을 제공함으로써 크게 향상될 수 있다(The ability of learning networks to generalize can be greatly enhanced by providing constraints from the task domain)." LeCun의 이전 연구(LeCun 1989)에서 이미 입증된 원리입니다: **자유 파라미터 수를 줄이되, 계산 능력은 과도하게 줄이지 않는 것**이 핵심입니다.

The opening sentence summarizes everything: "The ability of learning networks to generalize can be greatly enhanced by providing constraints from the task domain." Already demonstrated in LeCun's prior work: the key is **reducing free parameters without overly reducing computational power**.

**이론적 정당화 / Theoretical justification:**

이 원리의 이론적 근거가 명시적으로 제시됩니다: (1) 제약된 네트워크는 **엔트로피가 줄어든(reduced entropy)** 아키텍처를 가짐, (2) **Vapnik-Chervonenkis 차원**이 낮아져 더 적은 데이터로도 일반화 가능, (3) Baum & Haussler(1989)의 이론에 부합. 이것은 "왜 더 적은 파라미터가 더 좋은가?"에 대한 체계적 답변이며, 현대의 정규화(regularization) 이론의 실용적 선구자입니다.

Theoretical justification is explicitly provided: (1) constrained networks have **reduced entropy** architecture, (2) lower **Vapnik-Chervonenkis dimension** enables generalization with less data, (3) consistent with Baum & Haussler (1989). This is a systematic answer to "why are fewer parameters better?" and a practical precursor to modern regularization theory.

**이전 연구와의 차별점 / Distinction from prior work:**

이전 Bell Labs 연구(Denker et al., 1989)에서는 합성곱의 커널 계수가 **수동으로 설계**되었습니다. 이 논문에서는 처음 두 합성곱 층의 커널이 **backpropagation으로 자동 학습**됩니다. 또한 이전에는 특징 벡터(feature vectors)를 입력으로 사용했지만, 여기서는 **원시 이미지를 직접** 입력합니다 — "대량의 저수준 정보를 다루는 backpropagation의 능력을 입증"합니다.

Previous Bell Labs work (Denker et al., 1989) used **hand-designed** convolution kernels. This paper **automatically learns** kernels via backprop. Also, prior work used feature vectors as input, but here **raw images are directly input** — "demonstrating the ability of backpropagation networks to deal with large amounts of low-level information."

---

### §2 Zip Codes (pp.541–542) — 데이터와 전처리

**데이터베이스 / Database:**

Buffalo, NY 우체국을 통과하는 미국 우편물의 손글씨 우편번호에서 추출한 **9,298개 분리된 숫자** 이미지. 다양한 사람, 크기, 필기 스타일, 필기 도구로 작성. 학습: 7,291개, 테스트: 2,007개. **중요한 특징**: 학습셋과 테스트셋 모두에 모호하거나, 분류 불가능하거나, 잘못 분류된 예시가 다수 포함. 이것은 실제 데이터의 "지저분함(messiness)"을 의도적으로 반영합니다.

9,298 segmented digit images from handwritten zip codes on U.S. mail through Buffalo, NY. Written by many different people, sizes, styles, instruments. Train: 7,291, Test: 2,007. **Important feature**: both sets contain numerous ambiguous, unclassifiable, or misclassified examples — intentionally reflecting real-world data "messiness."

**전처리 / Preprocessing:**

우편번호에서 개별 숫자를 분리하는 것(segmentation)은 Postal Service 계약업체가 수행. 원래 ~40×60 픽셀 크기의 숫자를 **선형 변환으로 16×16으로 정규화** — 종횡비 보존, 불필요한 표시 제거 후 수행. 선형 변환이기에 결과 이미지는 이진이 아닌 **다단계 회색 톤(multiple gray levels)** 을 가짐. 회색 레벨은 $[-1, 1]$ 범위로 스케일링.

Segmentation done by Postal Service contractors. Original ~40×60 pixel digits **linearly normalized to 16×16** — preserving aspect ratio, after removing extraneous marks. Due to linear transformation, resulting images have **multiple gray levels** (not binary). Gray levels scaled to $[-1, 1]$.

---

### §3 Network Design (pp.542–546) — 네트워크 설계 (논문의 핵심)

이 섹션이 논문의 **가장 중요한 부분**입니다 — CNN의 세 가지 핵심 아이디어가 도입됩니다.

This section is the **most important part** — three core CNN ideas are introduced.

#### §3.1 Input and Output — 입출력

입력: 16×16 정규화 이미지 = 256 유닛. 출력: 10 유닛(place coding — 숫자당 하나). 모든 연결은 적응적(adaptive)이고 backpropagation으로 학습되지만 "heavily constrained"됩니다.

Input: 16×16 normalized image = 256 units. Output: 10 units (place coding — one per digit). All connections adaptive and trained by backprop but "heavily constrained."

#### §3.2 Feature Maps and Weight Sharing (p.542–544) — Feature Map과 가중치 공유

**핵심 아이디어 1: 국소 수용 영역 (Local receptive fields)**

시각적 패턴 인식의 고전적 지식: **국소 특징(local features)을 추출하고 결합하여 상위 특징을 형성**합니다. 이것을 네트워크에 구현하는 방법: hidden units가 **국소 정보 소스만** 결합하도록 강제합니다. 각 hidden unit은 전체 이미지가 아닌 5×5 = 25 픽셀만 봅니다.

Classical knowledge in visual pattern recognition: **extract local features and combine them to form higher-order features**. Implementation in the network: force hidden units to combine **only local information sources**. Each hidden unit sees only 5×5 = 25 pixels, not the whole image.

**핵심 아이디어 2: 가중치 공유 = 합성곱 (Weight sharing = Convolution)**

"특정 특징의 정확한 위치는 분류와 무관하므로, 약간의 위치 정보를 잃을 수 있다. 그러나 **대략적 위치 정보(approximate position)** 는 보존되어야 한다." 이 통찰에서 핵심 아이디어가 나옵니다: 어떤 위치에서든 같은 특징을 감지해야 하므로, **같은 가중치 세트를 이미지의 모든 위치에서 공유**합니다.

"Since the precise location of a feature is not relevant to the classification, we can afford to lose some position information. Nevertheless, approximate position information must be preserved." From this insight comes the key idea: since the same feature must be detected at any position, **share the same weight set across all positions**.

Rumelhart et al.(1986)에서 "T-C problem"을 위해 기술한 weight sharing을 적용합니다. 여러 연결이 **하나의 파라미터(가중치)로 제어**되며, 이는 연결 강도 사이에 **등식 제약(equality constraints)** 을 부과하는 것과 같습니다. 계산 오버헤드가 거의 없습니다.

Weight sharing from Rumelhart et al. (1986) for the "T-C problem" is applied. Multiple connections controlled by **a single parameter (weight)** — equivalent to imposing **equality constraints** among connection strengths. Very little computational overhead.

**Feature map의 정의 / Definition of feature map:**

첫 번째 hidden layer는 여러 **평면(planes)** — **feature maps** — 으로 구성됩니다. 같은 feature map 내의 모든 유닛이 **동일한 가중치 세트를 공유**합니다. "각 유닛은 이미지의 대응하는 부분에서 동일한 연산을 수행합니다. feature map이 수행하는 함수는 **5×5 커널을 가진 비선형 부분표본화 합성곱(nonlinear subsampled convolution)** 으로 해석할 수 있습니다."

First hidden layer consists of multiple **planes** — **feature maps**. All units in the same map share the **same weight set**. "Each unit performs the same operation on corresponding parts of the image. The function performed by a feature map can thus be interpreted as a **nonlinear subsampled convolution with a 5 by 5 kernel**."

**핵심 아이디어 3: 부분표본화 (Subsampling)**

H1에서 입력 이미지의 인접 유닛은 2 픽셀 간격(stride 2)으로 배치됩니다. 따라서 16×16 입력 → 8×8 feature map. 동기: "특징의 존재를 감지하는 데는 높은 해상도가 필요할 수 있지만, 정확한 위치를 결정할 필요는 그만큼 높지 않다."

In H1, adjacent units are spaced 2 pixels apart (stride 2) on the input. So 16×16 input → 8×8 feature map. Motivation: "high resolution may be needed to detect the presence of a feature, while its exact position need not be determined with equally high precision."

#### §3.3 Network Architecture (pp.544–546) — 아키텍처 상세

**Layer H1 — 12 feature maps, 8×8:**

- 12개의 독립적인 8×8 feature maps (H1.1, ..., H1.12)
- 각 유닛은 입력 이미지의 5×5 영역에서 입력 받음 (stride 2)
- 총 유닛: 12 × 64 = 768
- 총 연결: 768 × 26 = 19,968 (25 가중치 + 1 bias per unit)
- **자유 파라미터: 12 × 26 = 1,068** (각 map은 26개 파라미터만!)
- 경계 밖 연결은 상수 배경값 -1로 처리

12 independent 8×8 feature maps. Each unit receives 5×5 input (stride 2). Total units: 768. Total connections: 19,968. **Free parameters: only 1,068** (26 per map!). Connections past boundaries padded with constant -1.

**Layer H2 — 12 feature maps, 4×4:**

- H1과 유사하지만 H1이 다중 2차원 map이므로 더 복잡
- 각 유닛은 H1의 **8개** feature map에서 동일 위치의 5×5 영역을 입력 받음
- 입력 수: 8 × 25 = 200 가중치 + 1 bias = 201 per unit
- H1의 **어떤 8개** map을 사용할지는 "주어진 map이 훈련되지 않을 scheme"에 따라 선택
- 총 유닛: 12 × 16 = 192
- 총 연결: 192 × 201 = 38,592
- **자유 파라미터: 12 × 201 = 2,592** (H2도 feature map당 201개만!)

Similar to H1 but more complex since H1 has multiple 2D maps. Each unit takes 5×5 from **8** H1 maps at identical positions. Which 8 maps are chosen by a scheme "that will not be described here." Total units: 192. Connections: 38,592. **Free parameters: only 2,592.**

**Layer H3 — 30 units (fully connected):**

- H2에 완전 연결 (fully connected)
- 총 연결: 30 × (192 + 1) = 5,790
- 자유 파라미터: 5,790 (여기서는 제약 없음)

Fully connected to H2. Connections: 5,790. Free params: 5,790 (no constraints here).

**Output — 10 units (place coding):**

- H3에 완전 연결
- 총 연결: 10 × (30 + 1) = 310
- 자유 파라미터: 310

**총계 요약 / Summary:**

| Layer | Units | Connections | Free Parameters | Ratio |
|-------|-------|-------------|-----------------|-------|
| H1 | 768 | 19,968 | 1,068 | 5.3% |
| H2 | 192 | 38,592 | 2,592 | 6.7% |
| H3 | 30 | 5,790 | 5,790 | 100% |
| Output | 10 | 310 | 310 | 100% |
| **Total** | **1,256** | **64,660** | **9,760** | **15.1%** |

이 표가 논문의 핵심 메시지를 수치로 보여줍니다: weight sharing으로 인해 합성곱 층(H1, H2)의 자유 파라미터는 연결 수의 5~7%에 불과합니다. 이것이 "제약된 backpropagation"의 효과입니다.

This table numerically shows the paper's core message: due to weight sharing, free parameters in convolutional layers (H1, H2) are only 5-7% of connections. This is the power of "constrained backpropagation."

---

### §4 Experimental Environment (pp.546–547) — 실험 환경

**활성화 함수: Scaled hyperbolic tangent**

Sigmoid($1/(1+e^{-x})$) 대신 **scaled tanh**: $f(x) = A \tanh(Sx)$. 대칭 함수(원점 대칭)가 수렴이 더 빠르다고 알려져 있습니다(LeCun 1987). 출력 유닛의 목표값은 sigmoid의 준선형(quasilinear) 범위 내에서 선택하여, 가중치가 무한히 커지는 것과 "flat spot"에서의 정체를 방지합니다.

Instead of sigmoid, **scaled tanh**: $f(x) = A\tanh(Sx)$. Symmetric functions (symmetric about origin) are believed to yield faster convergence (LeCun 1987). Target values for output units chosen within quasilinear range of sigmoid to prevent weights growing indefinitely and stalling in "flat spots."

**가중치 초기화: LeCun 초기화 / LeCun initialization:**

가중치를 $[-2.4/F, \, 2.4/F]$ 균일 분포에서 랜덤 초기화. $F$는 해당 유닛의 **입력 수(fan-in)**. 이 기법은 "총 입력을 sigmoid의 작동 범위 내로 유지하는 경향이 있다." 이것은 후에 **Xavier 초기화(Glorot & Bengio, 2010)** 와 **He 초기화(He et al., 2015)** 로 발전하는 가중치 초기화 이론의 실용적 시초입니다.

Weights initialized from uniform $[-2.4/F, 2.4/F]$ where $F$ is the unit's **fan-in** (number of inputs). This "tends to keep the total inputs within the operating range of the sigmoid." Practical precursor to **Xavier initialization** (Glorot & Bengio, 2010) and **He initialization** (He et al., 2015).

**Stochastic gradient descent:**

논문의 중요한 실험적 발견: **패턴 하나를 제시할 때마다 가중치를 갱신**하는 "stochastic gradient"가, 전체 학습셋에 대해 기울기를 평균한 후 갱신하는 "true gradient"보다 **훨씬 빠르게 수렴하고 더 강건한 해를 찾는다**고 보고합니다. "특히 대규모, 중복적(redundant) 데이터베이스에서" 효과적입니다.

Important experimental finding: **updating weights after each pattern** ("stochastic gradient") converges **much faster than true gradient** (averaging over entire training set) and "finds solutions that are more robust," "especially on large, redundant data bases."

이것은 현대 딥러닝의 **SGD(Stochastic Gradient Descent)** 와 **mini-batch** 학습의 직접적 선조입니다. 논문에서 "on-line procedure"라고 부르는 이 방법은 각 패턴이 가중치에 "노이즈가 있는 기울기 추정(noisy gradient estimate)"를 제공하며, 이 노이즈가 역설적으로 local minima 탈출에 도움이 됩니다.

Direct ancestor of modern **SGD** and **mini-batch** training. The "on-line procedure" provides a "noisy gradient estimate" per pattern, and paradoxically this noise helps escape local minima.

**2차 방법 사용 / Second-order method:**

모든 실험은 Hessian 행렬의 양의 대각 근사를 사용하는 Newton's algorithm의 특수 버전으로 수행(LeCun 1987; Becker and LeCun 1988). "학습 속도를 엄청나게 향상시킨다고 믿어지지만, 파라미터의 광범위한 조정 없이 확실히 수렴한다."

All experiments used a special version of Newton's algorithm with positive diagonal approximation of the Hessian. "Not believed to bring a tremendous increase in learning speed but it converges reliably without requiring extensive adjustments of the parameters."

---

### §5 Results (pp.547–549) — 실험 결과

**학습 과정 / Training process:**

23번의 학습셋 통과(pass) = 167,693 패턴 제시. Fig. 2(상단)에서 MSE가 학습셋과 테스트셋 모두에서 급격히 감소한 후 안정화. "수렴이 매우 빠르다(convergence is extremely quick)" — 실제 데이터의 높은 중복성(redundancy) 덕분입니다.

23 passes through training set = 167,693 pattern presentations. MSE drops rapidly then stabilizes on both training and test sets. "Convergence is extremely quick" — thanks to high redundancy in real data.

**핵심 결과 / Key results:**

| 측정 / Metric | 학습셋 / Training | 테스트셋 / Test |
|---|---|---|
| MSE (평균) | $2.5 \times 10^{-3}$ | $1.8 \times 10^{-2}$ |
| 오분류율 / Error rate | 0.14% (10개) | **5.0%** (102개) |

**거부(rejection) 메커니즘**: 실제 응용에서는 raw error rate보다 "일정 정확도를 위해 몇 %를 거부해야 하는가"가 중요합니다. 거부 기준: 가장 활성화된 두 출력 유닛의 활동 차이가 임계값 이상이어야 함. **1% 오류율**을 위해 테스트 패턴의 **12.1%를 거부**해야 했습니다.

Rejection mechanism: In practice, "what percentage must be rejected for X% accuracy" matters more than raw error. Rejection criterion: difference between two most active outputs must exceed threshold. For **1% error rate**: **12.1% rejection** of test patterns needed.

**Fully connected 네트워크와의 비교 — 가장 중요한 결과 / Comparison with fully connected — the most important result:**

| 네트워크 / Network | 연결 / Connections | 파라미터 / Parameters | 학습 오류 / Train | 테스트 오류 / Test | 거부율 (1%용) |
|---|---|---|---|---|---|
| Fully connected (40 hidden) | 10,690 | 10,690 | 1.6% | **8.1%** | 19.4% |
| **CNN (이 논문)** | **64,660** | **9,760** | 0.14% | **5.0%** | 12.1% |

**이 비교가 논문의 핵심 메시지입니다**: CNN은 연결이 **6배 많지만** 자유 파라미터는 **더 적고**, 일반화는 **훨씬 좋습니다**. "연결 수가 아니라 **자유 파라미터 수**가 일반화를 결정한다." 더 나아가, "일반화 성능이 상당히 좋은데, 이는 문제에 대한 훨씬 적은 사전 정보가 네트워크에 내장되었다는 점을 고려하면 놀랍다."

**This comparison is the paper's core message**: CNN has **6× more connections** but **fewer** free parameters, and **much better** generalization. "Connection count does not determine generalization — **free parameter count** does." Furthermore, "the present system performs slightly better than the previous system. This is remarkable considering that much less specific information about the problem was built into the network."

**학습된 커널의 생물학적 유사성 / Biological similarity of learned kernels:**

"네트워크가 합성한 일부 커널은 생물학적 시각 시스템(Hubel and Wiesel, 1962)에서 존재하는 것으로 알려진 특징 감지기 및/또는 이전 인공 문자 인식기에서 설계된 것과 놀랍도록 유사하게 해석될 수 있다 — 공간 미분 추정기(spatial derivative estimators)나 중심부 흥분/주변부 억제(off-center/on-surround) 유형 특징 감지기처럼."

"Some kernels synthesized by the network can be interpreted as feature detectors remarkably similar to those found to exist in biological vision systems (Hubel and Wiesel 1962) and/or designed into previous artificial character recognizers, such as spatial derivative estimators or off-center/on-surround type feature detectors."

**주요 오류 원인 / Main error sources:**

대부분의 오분류는 이미지 분할(segmentation)의 오류에 기인. 특히 문자가 겹치는 경우. 그 외: 모호한 패턴, 저해상도, 학습셋에 없는 필기 스타일.

Most misclassifications due to image segmentation errors, especially overlapping characters. Others: ambiguous patterns, low resolution, writing styles absent from training set.

---

### §5.2 DSP Implementation (p.549) — 하드웨어 구현

AT&T DSP-32C(25 MFLOPS) 칩을 사용하여 **30+ 분류/초** 달성. 이미지 획득부터 포함하면 10-12 분류/초. 인식 과정의 계산 시간은 거의 전부 multiply-accumulate 연산으로, DSP에 최적. 이것은 신경망이 **상업적 하드웨어에서 실시간 처리 가능**함을 최초로 입증한 것입니다.

Achieved **30+ classifications/sec** on AT&T DSP-32C (25 MFLOPS). Including image acquisition: 10-12/sec. Computation time almost entirely multiply-accumulate operations, ideal for DSP. First demonstration that neural networks are **real-time capable on commercial hardware**.

---

### §6 Conclusion (pp.549–550) — 결론

**세 가지 핵심 주장 / Three key claims:**

1. **Backpropagation이 대규모 실제 과제에 성공적으로 적용됨**: "우리의 결과는 숫자 인식에서 state of the art에 해당하는 것으로 보인다."

   "Our results appear to be at the state of the art in digit recognition."

2. **최소 전처리로 저수준 데이터에서 직접 학습**: "우리의 네트워크는 정교한 특징 추출이 아닌 최소 전처리의 저수준 표현에서 학습되었다." 이것은 현대 "end-to-end learning"의 선구자적 입장입니다.

   "Our network was trained on a low-level representation of data that had minimal preprocessing." Precursor to modern "end-to-end learning."

3. **제약이 핵심이다**: "네트워크 아키텍처와 가중치에 대한 제약은 과제에 대한 기하학적 지식을 시스템에 통합하도록 설계되었다. 데이터의 중복적 특성과 네트워크에 부과된 제약 덕분에, 학습 시간은 학습셋 크기를 고려하면 비교적 짧았다."

   "The network architecture and the constraints on the weights were designed to incorporate geometric knowledge about the task. Because of the redundant nature of the data and constraints, learning time was relatively short considering training set size."

**스케일링 속성 / Scaling properties:**

"더 작은 인공 문제에서의 backpropagation 결과를 외삽하여 예상할 수 있는 것보다 훨씬 좋은 스케일링 속성을 보였다." 이것은 데이터가 실제적이고 중복적일수록 CNN이 더 효율적으로 학습한다는 발견으로, 빅데이터 시대 딥러닝 성공의 예언적 관찰입니다.

"Scaling properties were far better than one would expect just from extrapolating results of backpropagation on smaller, artificial problems." A prophetic observation about deep learning success in the big data era.

---

## Key Takeaways / 핵심 시사점

1. **"제약이 곧 일반화다"**: 64,660 연결 / 9,760 파라미터 — weight sharing이라는 제약이 자유 파라미터를 85% 줄이면서도 계산 능력은 유지합니다. Fully connected(10,690 연결/파라미터)보다 연결이 6배 많지만 파라미터가 적어 일반화가 훨씬 좋습니다(5.0% vs 8.1%).

   **"Constraints are generalization"**: Weight sharing reduces free parameters by 85% while maintaining computational power. 6× more connections but fewer parameters → much better generalization.

2. **CNN의 세 기둥**: (1) 국소 수용 영역 — 5×5만 봄, (2) 가중치 공유 — 같은 커널 재사용 = 합성곱, (3) 부분표본화 — stride 2로 해상도 감소. 이 세 가지는 현대 CNN(AlexNet, VGG, ResNet)의 동일한 기본 구성 요소입니다.

   **Three pillars of CNN**: (1) Local receptive fields, (2) Weight sharing = convolution, (3) Subsampling. These remain the identical building blocks of modern CNNs.

3. **Stochastic gradient가 batch gradient를 이긴다**: 패턴마다 갱신하는 "on-line" 방법이 전체 데이터 후 갱신하는 "true gradient"보다 빠르고 강건합니다. 현대 SGD/mini-batch의 직접적 선조.

   **Stochastic gradient beats batch gradient**: Per-pattern "on-line" updates faster and more robust than "true gradient." Direct ancestor of modern SGD/mini-batch.

4. **LeCun 초기화**: $w \sim U[-2.4/F, 2.4/F]$ ($F$ = fan-in). 가중치를 sigmoid/tanh의 작동 범위 내로 유지하는 실용적 규칙. Xavier/He 초기화의 직접적 선구자.

   **LeCun initialization**: $w \sim U[-2.4/F, 2.4/F]$. Practical rule keeping weights in sigmoid/tanh operating range. Direct precursor to Xavier/He initialization.

5. **End-to-end learning의 시작**: 원시 이미지 → 최종 분류까지 **하나의 네트워크**가 전체를 학습. 수동 특징 추출이 불필요. 이것이 현대 딥러닝의 핵심 패러다임.

   **Start of end-to-end learning**: One network learns the entire pipeline from raw image → final classification. No manual feature extraction needed. Core paradigm of modern deep learning.

6. **학습된 커널 ≈ 생물학적 특징 감지기**: 네트워크가 자동으로 학습한 커널이 Hubel & Wiesel의 시각 피질 수용 영역과 유사. 이것은 CNN의 생물학적 타당성에 대한 강력한 증거.

   **Learned kernels ≈ biological feature detectors**: Automatically learned kernels resemble Hubel & Wiesel's visual cortex receptive fields. Strong evidence for CNN's biological plausibility.

---

## Mathematical Summary / 수학적 요약

### CNN Forward Pass — 합성곱 신경망 순전파

**입력 / Input**: 16×16 grayscale image, pixel values in $[-1, 1]$

**Layer H1 (Convolution + Subsampling):**
For feature map $k = 1, \ldots, 12$, position $(i, j)$ in the 8×8 output:
$$h1_k(i,j) = f\!\left(\sum_{m=0}^{4}\sum_{n=0}^{4} w_k(m,n) \cdot \text{input}(2i+m, \, 2j+n) + b_k \right)$$
where $f(x) = A\tanh(Sx)$ and stride = 2.

**Layer H2 (Convolution over feature maps):**
For feature map $k = 1, \ldots, 12$, position $(i, j)$ in the 4×4 output:
$$h2_k(i,j) = f\!\left(\sum_{l \in S_k}\sum_{m=0}^{4}\sum_{n=0}^{4} w_{k,l}(m,n) \cdot h1_l(2i+m, \, 2j+n) + b_k \right)$$
where $S_k$ is the set of 8 H1 maps connected to H2 map $k$.

**Layer H3 (Fully connected):** $h3_j = f\!\left(\sum_i w_{ji} \cdot \text{flatten}(h2)_i + b_j\right)$

**Output (Fully connected):** $\text{out}_j = f\!\left(\sum_i w_{ji} \cdot h3_i + b_j\right)$, $j = 0, \ldots, 9$

**Backprop with weight sharing constraint:**
Standard backprop (Rumelhart et al., 1986) but shared weights receive **sum of gradients** from all positions:
$$\frac{\partial E}{\partial w_k(m,n)} = \sum_{i,j} \frac{\partial E}{\partial h1_k(i,j)} \cdot \text{input}(2i+m, \, 2j+n) \cdot f'(\cdot)$$

---

## Paper in the Arc of History / 역사적 맥락의 타임라인

```
1962    Hubel & Wiesel ──────── 시각 피질: 수용 영역, 단순/복잡 세포
          │                       Visual cortex: receptive fields, simple/complex cells
1980    Fukushima ───────────── Neocognitron: 학습 없는 CNN 선조
          │                       CNN ancestor without backprop learning
1986    Rumelhart et al. ────── Backpropagation: 다층 학습 가능
          │                       Multi-layer learning enabled
1987    Denker et al. ─────── 수동 설계 커널 + 신경망 칩
          │                       Hand-designed kernels + neural network chip
          │
  ╔════════════════════════════════════════════════════════════╗
  ║ ★ 1989  LeCun et al. ★                                    ║
  ║  Backprop + Weight Sharing + Local Fields = CNN            ║
  ║  최초의 실용적 딥러닝: 5% error, 30 digits/sec             ║
  ╚════════════════════════════════════════════════════════════╝
          │
1998    LeCun et al. ────────── LeNet-5: 이 논문의 완성판
          │                       Definitive CNN paper
2006    Hinton et al. ─────── Deep Belief Nets
          │
2012    Krizhevsky et al. ──── AlexNet: GPU + 더 깊은 CNN
          │                       60M params, ImageNet 정복
2015    He et al. ─────────── ResNet: 152 layers, skip connections
          │
2020    Dosovitskiy et al. ─── ViT: CNN에서 Transformer로
                                  From CNN to Transformer
```

---

## Connections to Other Papers / 다른 논문과의 연결

| 논문 / Paper | 관계 / Relationship |
|---|---|
| **#1 McCulloch & Pitts (1943)** | 뉴런 모델의 기원. CNN의 각 유닛은 여전히 가중합 + 비선형 활성화라는 M-P 뉴런 구조 / Origin of neuron model. Each CNN unit still follows M-P weighted-sum + nonlinear-activation |
| **#4 Minsky & Papert (1969)** | 단층 한계 → 다층 필요 → backprop → CNN. Minsky의 도전이 이 논문까지 연쇄적으로 이어짐 / Single-layer limits → multi-layer → backprop → CNN. Minsky's challenge cascades to this paper |
| **#5 Hopfield (1982)** | 논문 참고문헌에 Denker et al. (1987)의 공저자로 Hopfield 포함. 에너지 기반 관점이 Bell Labs의 신경망 연구 배경 / Hopfield co-authored Denker et al. (1987). Energy-based perspective as Bell Labs NN research background |
| **#6 Rumelhart et al. (1986)** | **직접적 기반**: 이 논문은 backprop을 "적용"하는 논문. Rumelhart의 weight sharing 개념도 직접 차용 / **Direct foundation**: this paper "applies" backprop. Weight sharing concept also borrowed from Rumelhart |
| **#10 LeCun et al. (1998)** | 이 논문의 **완성판** — LeNet-5. 더 크고 정교한 아키텍처 + MNIST + 종합 비교 / **Definitive version** — LeNet-5. Larger architecture + MNIST + comprehensive comparison |
| **#13 Krizhevsky et al. (2012)** | AlexNet은 이 논문의 직계 후손. 같은 3원칙(local fields, weight sharing, subsampling) + GPU + ReLU + Dropout / AlexNet is a direct descendant. Same 3 principles + GPU + ReLU + Dropout |
| **Hubel & Wiesel (1962)** | 시각 피질 수용 영역 → local receptive fields의 생물학적 영감 / Visual cortex receptive fields → biological inspiration for local receptive fields |
| **Fukushima (1980)** | Neocognitron = 학습 없는 CNN의 선조. LeCun이 여기에 backprop을 결합 / Neocognitron = CNN ancestor without learning. LeCun combined it with backprop |

---

## References / 참고문헌

- LeCun, Y., Boser, B., Denker, J. S., Henderson, D., Howard, R. E., Hubbard, W., & Jackel, L. D., "Backpropagation applied to handwritten zip code recognition," *Neural Computation*, 1, pp. 541–551, 1989.
- Rumelhart, D. E., Hinton, G. E., & Williams, R. J., "Learning internal representations by error propagation," in *Parallel Distributed Processing*, Vol. 1, MIT Press, pp. 318–362, 1986.
- Hubel, D. H. & Wiesel, T. N., "Receptive fields, binocular interaction and functional architecture in the cat's visual cortex," *J. of Physiol.*, 160, pp. 106–154, 1962.
- Fukushima, K., "Neocognitron: A self-organizing neural network model for a mechanism of pattern recognition unaffected by shift in position," *Biol. Cybern.*, 36, pp. 193–202, 1980.
- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P., "Gradient-based learning applied to document recognition," *Proc. IEEE*, 86(11), pp. 2278–2324, 1998.
- Denker, J. S. et al., "Neural network recognizer for hand-written zip code digits," in *NIPS*, pp. 323–331, 1989.
- Baum, E. B. & Haussler, D., "What size net gives valid generalization?" *Neural Comp.*, 1, pp. 151–160, 1989.
