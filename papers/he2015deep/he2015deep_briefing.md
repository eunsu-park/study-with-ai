---
title: "Pre-Reading Briefing: Deep Residual Learning for Image Recognition"
paper_id: "20_he_2015"
topic: Artificial_Intelligence
date: 2026-04-19
type: briefing
---

# Deep Residual Learning for Image Recognition (ResNet): Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: He, K., Zhang, X., Ren, S., & Sun, J. (2015). *Deep Residual Learning for Image Recognition*. arXiv:1512.03385 (CVPR 2016).
**Author(s)**: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (Microsoft Research Asia)
**Year**: 2015 (arXiv Dec 2015 / CVPR 2016 Best Paper)

---

## 1. 핵심 기여 / Core Contribution

### 한국어
ResNet은 "네트워크를 더 깊이 쌓을수록 성능이 좋아질 것"이라는 직관이 실제로는 무너지는 현상(**degradation problem**)을 진단하고, 이를 해결하는 단순하면서도 혁명적인 설계 — **잔차 연결(residual connection, skip connection)** — 을 제시했습니다. 층이 학습해야 할 함수를 $H(x)$가 아니라 $F(x) = H(x) - x$로 **재정식화(reformulate)** 함으로써, 매우 깊은 네트워크(152층, 심지어 1202층)도 안정적으로 학습되도록 만들었습니다. 그 결과 ImageNet에서 3.57% top-5 error로 인간 수준을 최초로 넘어섰고, ILSVRC 2015 분류/검출/위치추정 및 COCO 검출/분할 5개 대회를 휩쓸었습니다. 오늘날 거의 모든 딥러닝 아키텍처(Transformer 포함)의 기본 구성 요소가 된 설계입니다.

### English
ResNet diagnoses the **degradation problem** — a phenomenon where simply stacking more layers causes *training* accuracy to degrade, contradicting the intuition that deeper networks should perform at least as well as shallower ones. It introduces **residual connections (skip connections)** as a simple yet revolutionary remedy: each block learns a *residual function* $F(x) = H(x) - x$ rather than $H(x)$ directly. This reformulation makes very deep networks (152 layers, even 1202 layers) optimizable. The result: 3.57% top-5 error on ImageNet — the first architecture to surpass human-level performance — and a sweep of five ILSVRC & COCO 2015 competitions. Skip connections are now a foundational primitive in nearly every modern deep network, including Transformers.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

**한국어**
2012년 AlexNet(#13)이 ImageNet에서 우승하며 딥러닝 혁명이 시작된 후, 3년간 커뮤니티는 "더 깊게(deeper)"를 외쳤습니다. VGGNet(2014, 19층)과 GoogLeNet/Inception(#19, 2014, 22층)은 깊이가 성능의 핵심임을 보였고, Batch Normalization(#18, 2015)은 기울기 소실/폭주(vanishing/exploding gradient) 문제를 상당 부분 완화했습니다. 그러나 2015년 초반까지도 30층을 넘는 일반적 평범 CNN(plain CNN)은 **정상적으로 수렴하지 않는** 이상 현상이 관찰되었습니다 — 기울기 문제는 아닌데도 훈련 오차가 오히려 더 나빠지는 것이었습니다. 이것이 He 등이 진단한 degradation problem이며, ResNet은 이 문제를 정면으로 해결했습니다.

**English**
After AlexNet (paper #13) kicked off the deep learning revolution in 2012, the mantra for three years was "go deeper." VGGNet (2014, 19 layers) and GoogLeNet/Inception (paper #19, 2014, 22 layers) established depth as a key to accuracy, while Batch Normalization (paper #18, 2015) largely tamed vanishing/exploding gradients. Yet by early 2015, "plain" CNNs beyond ~30 layers failed to converge properly — **training** error grew worse with depth, even though gradients were well-behaved. This puzzle is the *degradation problem* that He et al. diagnosed, and ResNet solved it head-on.

### 타임라인 / Timeline

```
1989  LeCun — LeNet, first CNN
1998  LeCun — LeNet-5
2012  Krizhevsky — AlexNet (8 layers, ILSVRC winner)        [#13]
2014  Simonyan & Zisserman — VGG (19 layers)
2014  Szegedy — GoogLeNet / Inception v1 (22 layers)         [#19]
2015  Ioffe & Szegedy — Batch Normalization                 [#18]
2015  Srivastava — Highway Networks (gated skip connections)
2015  He — ResNet (152 layers, 3.57% top-5, human-level)    ★ THIS PAPER
2016  He — Identity Mappings in Deep Residual Networks (pre-activation ResNet)
2016  Huang — DenseNet (densely connected skip)
2017  Vaswani — Transformer (skip connections inside attention blocks)
2020+ Nearly every modern architecture uses residual connections
```

---

## 3. 필요한 배경 지식 / Prerequisites

### 한국어
- **CNN 기초**: 합성곱, 풀링, ReLU (AlexNet #13 학습 완료)
- **Batch Normalization** (#18): ResNet의 각 conv 뒤에 BN이 들어갑니다.
- **Inception/GoogLeNet** (#19): 비교 대상이자 ResNet의 기반이 되는 디자인 철학(bottleneck 1×1 conv).
- **역전파와 체인 룰**: skip connection이 기울기 흐름에 미치는 영향을 이해하는 데 필요.
- **SGD + momentum**: 실험 설정 이해.
- **PyTorch/TensorFlow에서의 모듈(block) 개념**: `Residual Block`을 구현/시각화할 때 도움.

### English
- **CNN basics**: convolution, pooling, ReLU (covered in AlexNet #13).
- **Batch Normalization (#18)**: BN is applied after every conv in ResNet.
- **Inception/GoogLeNet (#19)**: a comparison point and the source of the bottleneck 1×1 conv design.
- **Backpropagation & chain rule**: needed to understand why skip connections help gradient flow.
- **SGD + momentum**: for reading the training setup.
- **Module (block) abstraction in PyTorch/TensorFlow**: helps when implementing the Residual Block.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **Residual function** $F(x)$ | 층이 학습해야 할 "차이" — 원하는 매핑 $H(x)$에서 입력 $x$를 뺀 것: $F(x) = H(x) - x$. Learned residual mapping; the "delta" a block adds to its input. |
| **Identity shortcut / skip connection** | 입력을 변환 없이 출력에 더하는 연결 ($y = F(x) + x$). Direct path that adds the input to the output. |
| **Projection shortcut** | 채널 수 또는 해상도가 바뀔 때 사용하는 1×1 conv 기반 shortcut ($y = F(x) + W_s x$). Used when dimensions mismatch. |
| **Degradation problem** | 네트워크가 깊어질수록 **훈련** 오차가 오히려 증가하는 현상 — 과적합도 기울기 문제도 아님. Training error grows with depth, distinct from overfitting or gradient vanishing. |
| **Bottleneck block** | 1×1 → 3×3 → 1×1 conv 구조로 계산량을 줄인 깊은 ResNet(50/101/152) 기본 블록. 3-conv block that shrinks and restores channel count. |
| **Pre-activation** | BN→ReLU→Conv 순서로 재배열된 변형(후속 논문). Variant reordering BN/ReLU before Conv (follow-up paper). |
| **Ensemble** | 서로 다른 모델의 예측을 평균내는 기법. 최종 3.57%는 6개 모델 앙상블. Averaging predictions from multiple models. |
| **Top-1 / Top-5 error** | ImageNet 평가 지표: 1개/5개 예측 내에 정답이 없을 확률. Classification metrics: accuracy within 1 or 5 predictions. |
| **Plain network** | Skip connection이 없는 일반 CNN (ResNet의 대조군). A vanilla CNN without residual connections. |
| **VGG-style design** | 3×3 conv만 반복하고, 피쳐맵이 절반으로 줄면 필터 수를 두 배로 늘리는 설계 원칙. Design using 3×3 convs with channel doubling on downsampling. |
| **Global Average Pooling (GAP)** | Fully-connected 대신 각 채널의 평균을 쓰는 pooling. Replaces FC for final feature aggregation. |
| **FLOPs** | Floating-point operations, 계산 복잡도 지표. Measure of computational cost. |

---

## 5. 수식 미리보기 / Equations Preview

### (1) Residual mapping / 잔차 매핑
$$
\mathbf{y} = F(\mathbf{x}, \{W_i\}) + \mathbf{x}
$$
여기서 $F$는 학습할 잔차 함수, $+\mathbf{x}$는 identity shortcut.
Here $F$ is the learned residual; $+\mathbf{x}$ is the identity shortcut.

### (2) Projection shortcut (dimension change) / 투영 shortcut
$$
\mathbf{y} = F(\mathbf{x}, \{W_i\}) + W_s \mathbf{x}
$$
채널 수나 해상도가 달라질 때 $W_s$는 1×1 conv로 맞춰줍니다.
$W_s$ is a 1×1 conv that aligns dimensions when $F$ changes them.

### (3) Two-layer residual block (basic) / 2층 기본 블록
$$
F(\mathbf{x}) = W_2\,\sigma(W_1 \mathbf{x}), \quad \sigma = \text{ReLU}
$$
실제로는 각 conv 뒤에 BN이 들어가며 addition 후 최종 ReLU가 붙습니다.
In practice: Conv–BN–ReLU–Conv–BN, then add, then ReLU.

### (4) Bottleneck block (3-layer) / 병목 블록
$$
F(\mathbf{x}) = W_3\,\sigma\!\big(W_2\,\sigma(W_1 \mathbf{x})\big)
$$
여기서 $W_1$은 1×1 (채널 축소), $W_2$는 3×3, $W_3$은 1×1 (채널 복원).
$W_1$ is 1×1 reduction, $W_2$ is 3×3, $W_3$ is 1×1 restoration.

### (5) Gradient flow through skip connection / skip의 기울기 전달
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \cdot \left(1 + \frac{\partial F}{\partial \mathbf{x}}\right)
$$
"$+1$" 항 덕분에 매우 깊은 네트워크에서도 기울기가 최소한 입력까지 전달됩니다.
The "+1" term guarantees gradient flow to the input even through many layers — the key theoretical insight.

---

## 6. 읽기 가이드 / Reading Guide

### 한국어
- **§1 Introduction**: Figure 1의 degradation plot에 주목. 20층 vs 56층 plain network가 왜 이상한지 확인.
- **§3 Deep Residual Learning**: 핵심 섹션. §3.1의 residual 재정식화 동기, §3.2의 identity 수식, §3.3의 네트워크 아키텍처(Figure 3과 Table 1을 반드시 같이 볼 것)를 정독.
- **§4.1 ImageNet Classification**: Table 2와 Figure 4가 핵심 실험 결과. 18층/34층 plain vs ResNet 비교가 degradation 해결의 증거.
  - §4.1 "Identity vs Projection Shortcuts": 옵션 (A)(B)(C) 비교.
  - §4.1 "Deeper Bottleneck Architectures": 50/101/152층 설계.
- **§4.2 CIFAR-10**: 1202층 실험과 과적합 논의.
- **§4.3 PASCAL/COCO Detection**: 시간 없으면 결과표만 확인.
- **수식 위주로 읽기 팁**: 식 (1)~(2) 외에는 복잡한 수식이 거의 없습니다. 대신 블록 구조 다이어그램(Figure 2, 5)을 손으로 그려보세요.

### English
- **§1 Introduction**: pay attention to Figure 1 (degradation plot) — 20 vs 56 layers on CIFAR-10.
- **§3 Deep Residual Learning**: the core. §3.1 motivates the reformulation, §3.2 gives the identity equation, §3.3 shows architectures (always read Figure 3 + Table 1 together).
- **§4.1 ImageNet Classification**: Table 2 + Figure 4 are the key empirical results. 18-layer / 34-layer plain-vs-ResNet is the smoking gun for degradation.
  - "Identity vs Projection Shortcuts": compares options (A)(B)(C).
  - "Deeper Bottleneck Architectures": 50/101/152-layer design.
- **§4.2 CIFAR-10**: 1202-layer experiment and overfitting discussion.
- **§4.3 PASCAL/COCO Detection**: just skim the result tables if short on time.
- **Math is light**: only Eqs. (1)–(2). Instead, hand-draw the block diagrams (Figures 2 and 5) — that's where insight lives.

---

## 7. 현대적 의의 / Modern Significance

### 한국어
잔차 연결은 **딥러닝 아키텍처 설계의 기본 문법**이 되었습니다:

- **컴퓨터 비전**: ResNet-50은 사실상 모든 downstream task(검출, 분할, 포즈, 비디오)의 표준 backbone입니다. DenseNet, ResNeXt, Wide ResNet, RegNet은 모두 ResNet의 직계 후손입니다.
- **Transformer**: 2017년 Vaswani의 *Attention Is All You Need*에서 각 attention/FFN 서브층은 $\text{LayerNorm}(x + \text{Sublayer}(x))$ 형태 — 이는 정확히 ResNet의 skip connection입니다. GPT, BERT, LLaMA, Claude — 모두 skip connection 없이는 학습 불가능합니다.
- **생성 모델**: Diffusion model의 U-Net, GAN의 generator/discriminator 모두 residual block을 내장.
- **이론**: "Loss landscape" 연구(Li et al. 2018)에서 skip connection이 손실 지형을 부드럽게 만든다는 것이 시각화로 증명되었습니다.

ResNet 이전과 이후는 **"50층 이상의 네트워크를 자유롭게 쌓을 수 있는가"** 라는 능력의 차이로 나뉩니다. 오늘날 수백~수천 층(또는 수백 개 Transformer 블록)을 쌓을 수 있는 것은 전적으로 He 등의 이 아이디어 덕분입니다.

### English
Residual connections became the **default grammar of deep architectures**:

- **Computer Vision**: ResNet-50 is the de facto backbone for nearly every downstream task (detection, segmentation, pose, video). DenseNet, ResNeXt, Wide ResNet, RegNet are all direct descendants.
- **Transformers**: In Vaswani et al. (2017), each attention/FFN sub-layer is $\text{LayerNorm}(x + \text{Sublayer}(x))$ — exactly the ResNet skip. GPT, BERT, LLaMA, Claude — none of them train without residual connections.
- **Generative models**: Diffusion U-Nets, GAN generators/discriminators all embed residual blocks.
- **Theory**: Loss-landscape visualization work (Li et al. 2018) demonstrated that skip connections dramatically smooth the loss surface.

The line between "pre-ResNet" and "post-ResNet" deep learning is whether you can freely stack networks beyond ~50 layers. Everything from 1000-layer vision models to hundred-block Transformers exists because of this one idea.

---

## Q&A

### Q1. 왜 덧셈(addition)인가? Concatenation이 정보 보존에는 더 좋지 않은가? / Why addition, not concatenation?

**직관적 의심은 합리적입니다.** Concat은 "모든 정보를 그대로 이어붙이므로" 정보 손실이 0이고, addition은 "값을 섞어버리니까" 정보가 희석될 것 같습니다. 그러나 He 등이 덧셈을 선택한 이유는 **정보 전달력보다는 최적화(optimization) 관점**에서 훨씬 우월하기 때문입니다. 하나씩 봅시다.

The intuition is fair, but addition wins on **optimization grounds**, not information grounds. Here's why:

---

#### ① 차원 유지 / Dimension preservation

**한국어**
Addition: $\mathbf{y} = F(\mathbf{x}) + \mathbf{x}$ → 출력 채널 수 = 입력 채널 수 (불변).
Concat: $\mathbf{y} = [F(\mathbf{x}),\, \mathbf{x}]$ → 블록마다 채널이 **누적 증가**.

ResNet-152는 잔차 블록이 약 50개입니다. 매 블록마다 채널이 64 → 128 → 256 → ... 이렇게 concat으로 쌓이면, 150층 지점에서 채널 수가 기하급수적으로 폭발해서 **네트워크 자체를 깊게 만드는 게 불가능**해집니다. ResNet의 목표가 "매우 깊은 네트워크를 만들기"였으므로, 차원이 불변인 연산이 필수였습니다.

**English**
Addition keeps channel count constant; concat accumulates channels at every block. ResNet-152 has ~50 residual blocks — concatenating would blow up channels exponentially, defeating the very goal of going deep. Addition lets you stack indefinitely.

---

#### ② Identity mapping의 수학적 단순성 / The identity mapping is trivial with addition

**한국어**
ResNet의 핵심 동기는 "**깊은 네트워크가 최소한 얕은 네트워크만큼은 나와야 하는데 안 나온다**"는 관찰이었습니다. 이 문제를 해결하려면, 불필요한 층은 **identity 함수**($y = x$)가 되도록 **학습이 쉬워야** 합니다.

- **Addition**에서 identity가 되려면: $F(x) = 0$ → 모든 가중치를 0으로 밀기만 하면 됨 → 최적화가 매우 쉬움. BN + ReLU 초기화만으로도 자연스럽게 출력이 작게 시작함.
- **Concat**에서 identity가 되려면: 다음 층이 $[F(x), x]$에서 $x$만 선택적으로 골라 쓰도록 **가중치를 학습해야 함** → 단순 제로화로는 안 됨 → 여전히 "학습해야 할 것"이 남음.

즉, addition은 **"하지 않는 것(do nothing)"을 0으로 학습하는 것이 최적화의 기저(baseline)** 이고, concat은 그 baseline이 존재하지 않습니다.

**English**
ResNet's core motivation was that a deep net should be *at least as good as* a shallow one. To make this true, unnecessary layers should collapse to an **identity** easily.
- With addition: identity = $F(x) = 0$. Just push weights toward zero. Extremely easy — BN+ReLU initialization already makes $F$ start near zero.
- With concat: identity requires the *next* layer to learn to ignore $F(x)$ and select only $x$. Requires specific learned weights — no "free lunch."

Addition makes "do nothing" the default baseline; concatenation doesn't.

---

#### ③ 기울기 전달의 "+1" 항 / The "+1" in gradient flow

**한국어**
$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \cdot \left(1 + \frac{\partial F}{\partial \mathbf{x}}\right)$$

이 **"+1"** 덕분에, 아무리 깊어져도 기울기가 최소 1배로 전달됩니다(vanishing 방지). Concat도 기울기 전달 자체는 가능하지만, 다음 층의 projection이 섞여 있어 "+1이 보장된다"는 깔끔한 성질이 사라집니다. 2016년 He 등의 후속 논문 *Identity Mappings in Deep Residual Networks*에서 이 "+1" 성질이 매우 깊은 네트워크의 학습 성공의 핵심이라고 이론적으로 강조합니다.

**English**
The "+1" term guarantees a gradient highway that cannot vanish. With concat, the next layer's projection entangles the gradient path — you lose the clean "identity gradient" guarantee that He's 2016 follow-up paper shows is essential for 1000-layer training.

---

#### ④ 파라미터 / 연산량 / Parameters and FLOPs

**한국어**
질문하신 연산량도 실제 요인입니다. Concat을 쓰면 다음 블록의 입력 채널이 두 배가 되므로 conv의 파라미터와 FLOPs가 **4배**가 됩니다(conv는 입력 × 출력 채널에 비례). ResNet-50을 concat-ResNet으로 바꾸면 파라미터가 수억 개로 부풀어 학습 자체가 어려워집니다.

**English**
Yes, compute matters. Concat doubles input channels of the next conv, so its parameters and FLOPs grow **4×** (conv cost ∝ in × out channels). Concat-ResNet-50 would balloon into hundreds of millions of parameters.

---

#### ⑤ 반론: DenseNet은 concat을 선택했다 / The counter-example: DenseNet

**한국어**
흥미롭게도 2016년 Huang 등의 **DenseNet**은 concat 방식을 실제로 채택했습니다. 이들은 채널 폭발을 막기 위해 **growth rate**($k=12\sim32$)라는 매우 작은 증가량을 쓰고, 중간에 **transition layer**(1×1 conv + pooling)로 채널 수를 강제로 줄였습니다. DenseNet은 파라미터 효율 면에서는 ResNet보다 나았지만, **메모리 사용량(모든 이전 feature map을 저장)** 과 **구현 복잡성**이 커서 실제 산업 현장에서는 ResNet이 압도적으로 많이 쓰입니다.

**English**
DenseNet (Huang et al., 2016) actually chose concatenation, but only by adding tight **growth-rate** constraints ($k=12\sim32$) and **transition layers** (1×1 conv + pooling) to prevent channel explosion. It's more parameter-efficient than ResNet, but pays heavily in memory (stores all previous feature maps) and implementation complexity — which is why ResNet, not DenseNet, became the industry default.

---

#### ⑥ "정보 손실" 직관을 다시 보기 / Revisiting the "information loss" worry

**한국어**
Addition은 "정보가 섞여서 손실된다"처럼 느껴지지만, 실제로 **$F(x)$와 $x$는 서로 다른 feature들**이고, 둘의 합은 "원본 + 잔차(보정)"라는 **의미 있는 해석**을 가집니다. 이는 신경망의 각 층을 **"점진적 개선(iterative refinement)"** 으로 보는 관점입니다. 실제로 후속 연구(Greff et al. 2017, "Highway and Residual Networks Learn Unrolled Iterative Estimation")는 ResNet의 동작을 "반복적인 추정 업데이트"로 해석합니다.

또한 concat이 "정보 보존"에는 유리해도, 다음 층이 그 정보를 **활용**하려면 결국 선형 결합(linear combination)을 학습해야 하는데, 이는 본질적으로 **가중합(weighted addition)** 입니다. 즉 concat 뒤에 conv가 붙으면 수학적으로는 "학습된 가중합"이고, addition은 "가중치를 1로 고정한 버전"일 뿐입니다. ResNet은 이 "단순한 고정 가중치"가 최적화에 더 유리함을 실험으로 증명한 것입니다.

**English**
Addition can look "lossy," but $F(x)$ and $x$ carry complementary features; their sum has a meaningful reading as *original + correction*. Later work (Greff et al., 2017) interprets ResNets as performing **iterative refinement** of a hidden state. Also, concat only preserves information if the next conv *learns* to combine it — and that combination is ultimately a weighted sum, i.e., **addition with learned weights**. ResNet's design fixes those weights at 1 and shows empirically that this simpler choice optimizes better.

---

#### 결론 / Bottom line

**한국어**
- Concat은 "정보 이론적으로" 손실이 없지만, **학습 가능성(optimizability)** 측면에서 addition이 훨씬 우수합니다.
- Addition의 핵심 미덕: ① 차원 유지 → 깊이 확장 가능, ② identity가 $F=0$으로 자연스럽게 수렴, ③ 깨끗한 기울기 경로, ④ 낮은 파라미터/FLOPs.
- "정보 전달에 충분한가?" → 네, 충분합니다. ImageNet, COCO, 그리고 오늘날의 Transformer 전부가 이를 실증합니다.

**English**
- Concat loses zero information in a Shannon sense, but addition wins on **optimizability**.
- Addition's virtues: (1) constant dimensionality → depth is cheap, (2) identity is trivially $F=0$, (3) clean gradient highway, (4) low parameters/FLOPs.
- Is addition "enough" for information transfer? Empirically yes — ImageNet, COCO, and every modern Transformer prove it.

---

### Q2. 왜 어떤 residual 모델은 덧셈 후 activation을 생략하는가? / Why do some residual models skip the activation after addition?

**정확한 관찰입니다.** 원래 ResNet(2015, 이 논문)은 "덧셈 → ReLU" 순서(**post-activation**)였지만, 1년 뒤 같은 저자들이 발표한 ***Identity Mappings in Deep Residual Networks* (He et al., ECCV 2016)** 은 덧셈 후 activation을 없앤 **pre-activation ResNet (ResNet v2)** 이 더 잘 학습된다는 것을 이론적·실험적으로 증명했습니다. 오늘날 Transformer의 주류 설계(pre-norm)도 같은 원리를 따릅니다.

You've noticed correctly. The original ResNet (2015, this paper) does **add → ReLU** (*post-activation*), but one year later the same authors' follow-up, ***Identity Mappings in Deep Residual Networks*** (He et al., ECCV 2016), showed both theoretically and empirically that removing the ReLU after the addition — the **pre-activation** design (ResNet v2) — trains even deeper networks *better*. The mainstream Transformer design today (pre-norm) follows the same principle.

---

#### ① 두 설계의 비교 / The two designs side-by-side

**Post-activation (원본 ResNet v1, 2015)**
```
x ──► Conv ─► BN ─► ReLU ─► Conv ─► BN ─┐
 │                                       ▼
 └──────────────────────────────────────► (+) ─► ReLU ─► y
                                                   ↑
                                          activation AFTER add
```
$$y = \sigma\big(F(x) + x\big), \quad \sigma = \text{ReLU}$$

**Pre-activation (ResNet v2, 2016)**
```
x ──► BN ─► ReLU ─► Conv ─► BN ─► ReLU ─► Conv ─┐
 │                                                ▼
 └──────────────────────────────────────────────► (+) ─► y
                                          no activation after add
```
$$y = F(x) + x$$

**핵심 차이**: v2는 activation과 BN이 **잔차 함수 $F$ 내부**에만 존재하고, skip을 통과하는 identity 경로에는 **어떤 비선형성도 끼지 않습니다**.
The key difference: in v2 all activations/BN live *inside* $F$, and the identity path through the skip is **purely linear — nothing touches it**.

---

#### ② 수학적 이유: "깨끗한 identity 경로" / Mathematical reason: clean identity path

**한국어**
Pre-activation에서 여러 블록을 쌓으면 다음과 같이 **재귀적으로 깔끔하게** 풀립니다:
$$x_{l+1} = x_l + F(x_l, W_l)$$
$$x_{L} = x_l + \sum_{i=l}^{L-1} F(x_i, W_i)$$

즉, 층 $L$의 출력은 층 $l$의 입력에 그 사이의 모든 잔차를 **단순히 더한 것**입니다. 역전파도 마찬가지로:
$$\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L} \cdot \left(1 + \frac{\partial}{\partial x_l}\sum_{i=l}^{L-1} F(x_i, W_i)\right)$$

이 "+1" 항은 **어떤 가중치나 activation에도 무관하게 항상 존재**합니다. 즉 깊이가 1000층이어도 기울기가 입력까지 **손실 없이** 도달합니다.

반면 post-activation(v1)에서는 각 덧셈 뒤에 ReLU가 끼어서, 역전파할 때 ReLU의 기울기(0 또는 1)가 곱해집니다:
$$\frac{\partial x_{l+1}}{\partial x_l} = \sigma'(F + x) \cdot (1 + F'_x)$$

이 $\sigma'$ 때문에 기울기가 **차단**되거나 **왜곡**될 수 있습니다. 층이 수백 개 쌓이면 이 효과가 누적되어 학습이 어려워집니다. 이것이 He 등이 v1으로 1202층을 학습했을 때 과적합이 아니라 **최적화 자체**가 잘 안 되던 원인이었고, v2로 바꾸자 CIFAR-10에서 1001층 ResNet이 성공적으로 학습된 이유입니다.

**English**
Stacking pre-activation blocks gives a clean recursion:
$$x_L = x_l + \sum_{i=l}^{L-1} F(x_i, W_i)$$
So the output of layer $L$ is just the input of layer $l$ plus the sum of all residuals in between. The backward pass inherits the same purity:
$$\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L}\left(1 + \frac{\partial}{\partial x_l}\sum F\right)$$
The "+1" term is **unconditional** — independent of weights or activations. Gradients flow to the input losslessly at any depth.

With post-activation, each addition is followed by ReLU, so $\sigma'(F+x) \in \{0,1\}$ multiplies into the gradient. Over hundreds of layers this accumulates and breaks optimization. This is exactly why ResNet v1's 1202-layer model underperformed (optimization, not overfitting), and why v2 successfully trained a 1001-layer CIFAR-10 network.

---

#### ③ "비선형성이 사라지는 것 아닌가?"라는 걱정 / "But doesn't nonlinearity disappear?"

**한국어**
아주 자연스러운 걱정입니다. 답은: **$F(x)$ 내부에 이미 두 번(basic block) 또는 세 번(bottleneck)의 ReLU가 있습니다.** 따라서 블록 전체는 여전히 충분히 비선형적입니다.

네트워크의 비선형성은 "덧셈 후 ReLU가 있느냐"가 아니라 "**잔차 함수 내부에 ReLU가 몇 개 있느냐**"가 결정합니다. Pre-activation은 BN→ReLU→Conv→BN→ReLU→Conv 구조로, 비선형성은 내부에 **완전히 충분**하고, identity 경로는 **순수하게 선형**으로 유지하여 최적화 이점을 얻는 설계입니다.

**English**
A fair worry. But each $F(x)$ already contains **two (basic block) or three (bottleneck) ReLUs inside**. The block as a whole is just as nonlinear as before. Nonlinearity comes from "how many ReLUs are *inside* $F$," not from "whether there's a ReLU *after* the addition." Pre-activation simply **moves** the activations inside $F$ so the skip path can stay cleanly linear.

---

#### ④ 현대로의 연결: Transformer의 pre-norm / Modern connection: pre-norm Transformers

**한국어**
Vaswani의 원본 Transformer(2017)는 **post-norm**(post-activation과 유사):
$$y = \text{LayerNorm}(x + \text{Sublayer}(x))$$
덧셈 **후** LayerNorm을 적용합니다. 이는 ResNet v1과 같은 설계 철학입니다.

그러나 GPT-2(2019) 이후 거의 모든 대형 언어 모델(GPT-3/4, LLaMA, Claude 등)은 **pre-norm**으로 전환했습니다:
$$y = x + \text{Sublayer}(\text{LayerNorm}(x))$$
LayerNorm은 sublayer **내부**로 들어가고, **덧셈 이후에는 어떤 연산도 없이 그대로 다음 블록으로 전달**됩니다. 이유는 정확히 ResNet v2와 동일합니다: 수십~수백 개의 Transformer 블록을 안정적으로 학습하려면 **residual stream(identity 경로)** 이 깔끔해야 하기 때문입니다. Xiong et al. (2020) *"On Layer Normalization in the Transformer Architecture"* 는 post-norm이 warm-up 없이는 학습이 발산하는 반면 pre-norm은 안정적임을 증명했습니다.

**Mechanistic interpretability** 관점에서도 현대 LLM은 잔차 경로를 **"residual stream"** 이라 부르며, 각 층이 이 스트림에 정보를 **"읽고/쓰는"** 구조로 해석합니다 — 이 관점은 pre-norm처럼 identity 경로가 깔끔할 때만 성립합니다.

**English**
The original Transformer (Vaswani 2017) is **post-norm**:
$$y = \text{LayerNorm}(x + \text{Sublayer}(x))$$
— LayerNorm *after* the addition, analogous to ResNet v1.

Since GPT-2 (2019), virtually all large language models (GPT-3/4, LLaMA, Claude) switched to **pre-norm**:
$$y = x + \text{Sublayer}(\text{LayerNorm}(x))$$
LayerNorm moves *inside* the sublayer, and **nothing happens after the addition**. The reason is identical to ResNet v2: stacking dozens or hundreds of blocks requires a clean **residual stream**. Xiong et al. (2020) proved post-norm diverges without warm-up, while pre-norm trains stably from scratch.

Mechanistic interpretability takes this further: modern LLMs are analyzed as layers that **read from / write to** the residual stream — an interpretation that only makes sense because pre-norm keeps that stream unperturbed.

---

#### ⑤ 그런데 왜 원래 ResNet(v1)에서는 ReLU를 뒀는가? / So why did the original ResNet put a ReLU there?

**한국어**
첫 발견 당시 He 등도 단순히 "합 후 ReLU가 자연스럽다"는 관례를 따랐습니다. 이 선택이 왜 차선인지는 실제로 1202층 ResNet을 훈련해 보고 나서야 (과적합이 아닌 **최적화 실패**로) 드러났습니다. 1년 후 v2 논문에서 다음을 실험적으로 확인합니다:
- v1 CIFAR-10 1202층: 7.93% 오차(과적합)
- v2 CIFAR-10 1001층: **4.62% 오차** (훨씬 깊은데도 더 좋음)

**English**
At the time, He et al. simply followed the convention "activation follows addition." Only after training a 1202-layer v1 (which hit optimization, not overfitting, problems) did they realize the cost. The v2 paper shows:
- v1 on CIFAR-10 @ 1202 layers: 7.93% error (worse than 110-layer v1)
- v2 on CIFAR-10 @ 1001 layers: **4.62% error** (deeper *and* much better)

---

#### 결론 / Bottom line

**한국어**
- 덧셈 후 activation을 생략하는 설계는 **실수가 아니라 개선**입니다. ResNet v2(2016)에서 처음 정식화되었고, 현대 Transformer의 pre-norm까지 이어지는 원리입니다.
- 가능한 이유: ① 비선형성은 잔차 함수 $F$ **내부**에 충분히 존재, ② 식 $x_L = x_l + \sum F$의 깨끗한 덧셈 구조로 기울기 "+1"이 무조건 보장, ③ 매우 깊은(수백~수천 층) 네트워크 학습에 결정적.
- 실무적으로: 50~100층 수준에서는 v1/v2 차이가 작지만, 200층 이상 또는 수십 개의 Transformer 블록에서는 v2/pre-norm이 사실상 필수입니다.

**English**
- Skipping activation after addition isn't a mistake — it's an improvement, first formalized in ResNet v2 (2016) and now the default in modern pre-norm Transformers.
- Why it works: (1) nonlinearity already lives inside $F$, (2) the clean additive recursion $x_L = x_l + \sum F$ gives an unconditional "+1" gradient highway, (3) this is decisive for hundreds-to-thousands of layers.
- Practical rule: below ~100 layers v1 and v2 differ little; beyond 200 layers or for any deep Transformer stack, pre-activation / pre-norm is effectively mandatory.

