---
title: "Pre-Reading Briefing: The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks"
paper_id: "30_frankle_2019"
topic: Artificial_Intelligence
date: 2026-04-28
type: briefing
---

# The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Frankle, J., & Carbin, M. (2019). The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks. *International Conference on Learning Representations (ICLR 2019)*. arXiv:1803.03635.
**Author(s)**: Jonathan Frankle, Michael Carbin (MIT CSAIL)
**Year**: 2019

---

## 1. 핵심 기여 / Core Contribution

**한국어**
이 논문은 **Lottery Ticket Hypothesis (복권 가설)**라는 새로운 관점을 제시합니다. 가설의 내용: "**무작위 초기화된 dense feed-forward 네트워크는, 단독으로 학습했을 때 원본 네트워크와 동등한 정확도를 동등하거나 더 적은 반복(iteration) 안에 도달할 수 있는 subnetwork(즉 winning ticket)를 포함한다.**" 저자들은 표준 magnitude pruning이 이러한 winning ticket을 자연스럽게 발견함을 실험적으로 증명합니다. 핵심 절차는 (1) 무작위 초기화 → (2) 학습 → (3) 작은 magnitude의 weight 가지치기 → (4) **남은 weight를 원래 초기화 값으로 RESET** → (5) 재학습. 이 단순한 발견이 dense network의 압도적 over-parameterization을 설명하는 새로운 이론적 관점을 열고, 신경망 압축 연구의 패러다임을 바꿉니다.

**English**
This paper introduces the **Lottery Ticket Hypothesis**: "A randomly-initialized, dense feed-forward network contains a subnetwork that is initialized such that—when trained in isolation—it can match the test accuracy of the original network after training for at most the same number of iterations." Such subnetworks are called **winning tickets**. The authors empirically demonstrate that standard magnitude pruning naturally uncovers these winning tickets through a procedure of (1) random initialization, (2) training, (3) pruning smallest-magnitude weights, (4) **resetting** remaining weights to their original initialization, and (5) retraining. This simple finding offers a new perspective on neural network over-parameterization and reshapes the paradigm of network compression research.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

**한국어**
2010년대 중후반, 신경망 압축(network compression)은 활발한 연구 분야였습니다. LeCun et al. (1990)의 Optimal Brain Damage, Hassibi & Stork (1993)의 Optimal Brain Surgeon이 시초이고, Han et al. (2015)이 magnitude-based pruning으로 AlexNet/VGG의 매개변수를 90% 이상 줄일 수 있음을 보여 큰 영향을 끼쳤습니다. 그러나 통념(conventional wisdom)은 "**가지치기로 얻은 sparse architecture는 처음부터 학습하기 어렵다**"였습니다 — Li et al. (2016), Han et al. (2015) 모두 이를 관찰했습니다. Frankle & Carbin은 이 통념을 정면으로 반박합니다: 적절한 초기화만 있으면 sparse network도 처음부터 학습 가능하다는 것입니다.

**English**
In the mid-to-late 2010s, neural network compression was an active field. Pioneered by LeCun et al. (1990, Optimal Brain Damage) and Hassibi & Stork (1993, Optimal Brain Surgeon), the area saw a major boost from Han et al. (2015), who showed magnitude-based pruning could reduce AlexNet/VGG parameters by 90%+. However, the conventional wisdom held that "**sparse architectures produced by pruning are difficult to train from scratch**" — observed by Li et al. (2016) and Han et al. (2015) alike. Frankle & Carbin directly challenge this: with the right initialization, sparse networks *can* be trained from scratch.

### 타임라인 / Timeline

```
1990 ─ LeCun et al.: Optimal Brain Damage (second-derivative pruning)
1993 ─ Hassibi & Stork: Optimal Brain Surgeon
2014 ─ Ba & Caruana: "Do deep nets really need to be deep?" (distillation)
2015 ─ Hinton et al.: Knowledge distillation
2015 ─ Han et al.: Deep Compression (magnitude pruning, 90%+ reduction)
2016 ─ Li et al.: Pruning filters for efficient ConvNets — observes pruned networks
        train poorly from scratch
2016 ─ Iandola et al.: SqueezeNet (engineered small networks)
2017 ─ Howard et al.: MobileNets
2018 ─ Bellec et al.: Deep rewiring (training sparse networks directly)
2019 ─ ★★★ Frankle & Carbin: Lottery Ticket Hypothesis ★★★
2019 ─ Liu et al.: "Rethinking the Value of Network Pruning"
2020 ─ Frankle et al.: Linear Mode Connectivity (rewinding extension)
2021 ─ Chen et al.: LTH on BERT, Vision Transformers
```

---

## 3. 필요한 배경 지식 / Prerequisites

**한국어**
- **Neural network pruning**: magnitude-based pruning (Han et al. 2015), structured vs unstructured pruning
- **MNIST, CIFAR-10**: 표준 vision benchmark
- **LeNet-300-100, Conv-2/4/6, VGG-19, ResNet-18**: 본 논문에서 다루는 architecture
- **SGD, momentum, Adam**: optimization 기법
- **Glorot 초기화 (Xavier)**: $\mathcal{N}(0, 2/(n_{in}+n_{out}))$
- **Early stopping, validation loss**: 학습 속도 측정 proxy
- **Dropout, batch normalization, weight decay**: regularization 기법
- **Learning rate warmup**: 큰 모델/큰 lr에서 초기 학습 안정화

**English**
- **Neural network pruning**: magnitude-based pruning (Han et al. 2015), structured vs unstructured pruning
- **MNIST and CIFAR-10**: standard vision benchmarks
- **LeNet-300-100, Conv-2/4/6, VGG-19, ResNet-18**: architectures studied
- **SGD, momentum, Adam**: optimization methods
- **Glorot (Xavier) initialization**: $\mathcal{N}(0, 2/(n_{in}+n_{out}))$
- **Early stopping and validation loss**: proxy for learning speed
- **Dropout, batch normalization, weight decay**: regularization
- **Learning-rate warmup**: stabilizes early training for large lr / large models

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| Lottery Ticket Hypothesis | 무작위 초기화 dense network는 단독 학습으로 원본과 동등한 성능을 내는 subnetwork(winning ticket)를 포함한다는 가설 / The conjecture that a dense, randomly-initialized network contains a subnetwork that, when trained alone, matches the original |
| Winning Ticket | 적절한 초기화를 "복권에 당첨"한 sparse subnetwork $f(x; m \odot \theta_0)$ / A sparse subnetwork whose initialization "won the lottery" |
| Mask $m$ | weight를 살릴지 결정하는 binary 마스크 $m \in \{0,1\}^{|\theta|}$ / Binary mask deciding which weights survive |
| Sparsity $P_m$ | 마스크 후 남은 weight 비율 $P_m = \|m\|_0 / \|\theta\|$ / Fraction of weights remaining |
| Magnitude pruning | 절댓값이 작은 weight를 제거하는 전략 / Removing weights with smallest absolute value |
| One-shot pruning | 한 번 학습 후 한 번에 $p\%$ 가지치기 / Train once, prune $p\%$ once |
| Iterative Magnitude Pruning (IMP) | $n$ 라운드에 걸쳐 매번 $p^{1/n}\%$씩 점진적 가지치기 / Prune $p^{1/n}\%$ per round over $n$ rounds |
| Reset / Rewinding | 살아남은 weight를 원래 초기화 $\theta_0$로 되돌림 / Restore surviving weights to $\theta_0$ |
| Random reinitialization | 마스크 구조는 유지하되 새 무작위 초기화 사용 (대조군) / Same mask, fresh random init (control) |
| Random sparsity | 임의 패턴으로 가지치기 (구조 통제군) / Random mask pattern (structure control) |
| Commensurate accuracy/training | "동등한" 정확도와 학습 시간 (가설 정의) / "comparable" accuracy and iteration count (hypothesis definition) |
| Global pruning | 전체 layer를 통합해 한꺼번에 가지치기 (deep network용) / Prune across all layers jointly |
| Learning rate warmup | $k$ iteration 동안 lr을 0에서 목표값까지 선형 증가 / Linear ramp lr 0 → target over $k$ iterations |

---

## 5. 수식 미리보기 / Equations Preview

**한국어**

**(1) Lottery Ticket Hypothesis 형식화**:
$$\exists\, m \in \{0,1\}^{|\theta|} : f(x; m \odot \theta_0) \text{ achieves } a' \geq a \text{ at iteration } j' \leq j, \text{ with } \|m\|_0 \ll |\theta|$$

원본 네트워크 $f(x;\theta_0)$가 iteration $j$에서 정확도 $a$에 도달한다면, 어떤 마스크 $m$이 존재하여 sparse network $f(x; m \odot \theta_0)$가 동등 이하의 iteration $j'$에 동등 이상의 정확도 $a'$를 달성합니다.

**(2) Sparsity (남은 weight 비율)**:
$$P_m = \frac{\|m\|_0}{|\theta|}$$
예: $P_m = 25\%$이면 75% weight가 가지치기됨.

**(3) Iterative Magnitude Pruning per-round rate**:
$$\text{round-rate} = p^{1/n}, \quad P_m^{(t+1)} = P_m^{(t)} \cdot (1 - p^{1/n})^{-?}$$
$n$ 라운드 후 총 $p\%$ 가지치기 달성. 본 논문에서는 round당 보통 20%씩 (예: $1.0 \to 0.8 \to 0.64 \to \ldots$).

**(4) IMP procedure**:
```
1. θ₀ ~ D_θ (random init), m = 1
2. Train f(x; m ⊙ θ₀) for j iters → θⱼ
3. Prune p^(1/n)% smallest of m⊙θⱼ → m'
4. Reset: θ ← θ₀
5. m ← m', repeat 2-4
```

**(5) Cross-entropy loss (실험에서 사용)**:
$$\mathcal{L}(\theta) = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c} y_{ic} \log f_c(x_i; m \odot \theta)$$

**English**
The same five equations describe the formal hypothesis, sparsity definition, iterative pruning rate, the IMP procedure pseudocode, and the supervised loss function used in experiments. See section 4 of the notes for full term-by-term explanations.

---

## 6. 읽기 가이드 / Reading Guide

**한국어**
1. **§1 Introduction**: 가설의 직관과 원본 발견 (Figure 1) — sparse network는 학습이 어렵다는 통념을 뒤집음
2. **§2 Winning Tickets in Fully-Connected Networks**: LeNet-300-100 on MNIST. IMP, one-shot, random reinit 비교 (Figure 3, 4)
3. **§3 Winning Tickets in Convolutional Networks**: Conv-2/4/6 on CIFAR-10 (Figure 5, 6)
4. **§4 VGG and Resnet for CIFAR10**: 깊은 network에서는 learning rate warmup이 필수 (Figure 7, 8)
5. **§5 Discussion**: 초기화의 중요성, generalization 개선 ("Occam's Hill"), Liu et al. (2019)와의 관계
6. **§7 Related Work**: pre-/during/after-training 압축 분류
7. **Appendix B**: 두 IMP 전략 — Strategy 1 (resetting, 본문) vs Strategy 2 (continued training)
8. **Appendix F**: winning ticket 초기화의 분포 (bimodal!)

**English**
1. **§1**: Hypothesis intuition + original discovery (Figure 1) overturns "sparse hard to train"
2. **§2**: LeNet-300-100 on MNIST — IMP vs one-shot vs random reinit (Figs 3-4)
3. **§3**: Conv-2/4/6 on CIFAR-10 (Figs 5-6)
4. **§4**: VGG-19, ResNet-18 — warmup essential (Figs 7-8)
5. **§5**: Discussion — initialization matters, generalization ("Occam's Hill")
6. **§7**: Related work taxonomy (pre/during/after training)
7. **App. B**: Strategy 1 (resetting, used) vs Strategy 2 (continued training)
8. **App. F**: Bimodal distribution of winning-ticket initializations

---

## 7. 현대적 의의 / Modern Significance

**한국어**
이 논문은 ICLR 2019 Best Paper Award를 수상했고, 신경망 압축의 새 시대를 열었습니다. 핵심 후속 연구:
- **Frankle et al. (2020) Linear Mode Connectivity / Rewinding**: 큰 모델에서는 $\theta_0$ 대신 학습 초기 $\theta_k$로 rewind하면 더 효과적
- **Liu et al. (2019) Rethinking the Value of Network Pruning**: 어떤 sparse architecture는 random init으로도 학습 가능 → LTH와 부분적 충돌. Frankle은 fine-grained vs structured pruning 차이로 설명
- **Chen et al. (2020-2021)**: LTH on BERT, GANs, Vision Transformers, RL — winning ticket이 여러 도메인에서 발견됨
- **GraSP, SNIP, SynFlow**: 학습 없이 (at initialization) winning ticket 찾기
- **Strong Lottery Ticket**: 무작위 초기화에서 학습 없이도 좋은 성능을 내는 subnetwork가 존재 (Ramanujan et al. 2020)

이 논문은 over-parameterization 이론, NTK 분석(Du et al., Arora et al.), pruning-aware training 등 다양한 흐름과 연결됩니다. 실용적으로는 efficient inference, edge deployment, foundation model compression의 기초입니다.

**English**
This paper won ICLR 2019 Best Paper and ushered in a new era for compression research. Major follow-ups:
- **Frankle et al. (2020) Linear Mode Connectivity / Rewinding**: for larger models, rewinding to early-iter $\theta_k$ rather than $\theta_0$ works better
- **Liu et al. (2019) Rethinking the Value of Network Pruning**: certain sparse architectures train fine from random init — partial conflict with LTH, attributed by Frankle to fine-grained vs structured pruning
- **Chen et al. (2020–2021)**: LTH demonstrated on BERT, GANs, ViTs, RL — winning tickets across domains
- **GraSP, SNIP, SynFlow**: find winning tickets at initialization (no training)
- **Strong Lottery Ticket** (Ramanujan et al. 2020): subnetworks of random nets that perform well *without any training*

The paper connects to over-parameterization theory, NTK analysis (Du, Arora), and pruning-aware training. Practically, it underpins efficient inference, edge deployment, and foundation-model compression.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
