---
title: "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks"
authors: [Jonathan Frankle, Michael Carbin]
year: 2019
journal: "International Conference on Learning Representations (ICLR)"
doi: "arXiv:1803.03635"
topic: Artificial_Intelligence
tags: [pruning, network-compression, lottery-ticket-hypothesis, sparse-networks, initialization, magnitude-pruning, iterative-pruning, ICLR-best-paper]
status: completed
date_started: 2026-04-28
date_completed: 2026-04-28
---

# 30. The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks / 복권 가설: 희소하고 학습 가능한 신경망 찾기

---

## 1. Core Contribution / 핵심 기여

**한국어**
Frankle과 Carbin은 ICLR 2019 Best Paper로 선정된 이 논문에서 신경망 압축에 대한 통념을 정면으로 뒤집습니다. 통념: "**가지치기로 얻은 sparse architecture는 처음부터 학습하기 어렵다**" (Han et al., 2015; Li et al., 2016). 이 논문의 발견: 그것은 architecture 자체의 문제가 아니라 **초기화의 문제**다. 저자들은 **Lottery Ticket Hypothesis**를 다음과 같이 정식화합니다 — "**무작위 초기화된 dense feed-forward 네트워크는, 단독으로 학습했을 때(in isolation) 원본 네트워크와 동등한 정확도를 동등하거나 더 적은 반복(iteration) 안에 도달할 수 있는 subnetwork를 포함한다.**" 이러한 운 좋은 subnetwork를 **winning ticket**이라 부릅니다 — 이들은 "초기화 복권에 당첨"된 것입니다.

핵심은 단순하지만 결정적인 5단계 절차입니다: (1) 무작위 초기화 $\theta_0$, (2) $j$ iteration 학습 → $\theta_j$, (3) 가장 작은 magnitude의 weight $p\%$ 가지치기로 마스크 $m$ 생성, (4) **남은 weight를 원래 초기화 $\theta_0$로 RESET** ($f(x; m \odot \theta_0)$), (5) 재학습. 이 단순한 변형 — "**unique to our work**"라고 강조되는 step 4 — 이 모든 것을 바꿉니다. 무작위로 재초기화하면($f(x; m \odot \theta_0')$) winning ticket의 성능이 무너집니다. 즉 winning ticket은 sparse architecture와 그 architecture에 맞는 특정 초기화의 **결합**입니다. 실험적으로 LeNet (MNIST)는 21.1% sparsity, 7%까지 (iterative) 원본 정확도 유지; Conv-2/4/6 (CIFAR-10)은 8.8–15.1% sparsity에서 더 빠른 학습과 더 높은 정확도; VGG-19와 ResNet-18은 learning rate warmup 도입 시 winning ticket 발견 가능 (1.5%–11.8% 수준). 이는 over-parameterization 이론, generalization, optimization landscape에 새로운 관점을 제시하며, ICLR 2019 Best Paper로 선정되었습니다.

**English**
Frankle and Carbin's ICLR 2019 Best Paper directly overturns conventional wisdom about network compression. The wisdom: "**sparse architectures produced by pruning are difficult to train from scratch**" (Han et al., 2015; Li et al., 2016). The paper's finding: this is not an architectural limitation but an **initialization** problem. The authors formalize the **Lottery Ticket Hypothesis**: "A randomly-initialized, dense feed-forward network contains a subnetwork that—when trained in isolation—can match the test accuracy of the original network after training for at most the same number of iterations." Such fortunate subnetworks are dubbed **winning tickets** — they have "won the initialization lottery."

The mechanism is a simple but pivotal 5-step procedure: (1) randomly initialize $\theta_0$, (2) train for $j$ iterations to obtain $\theta_j$, (3) prune $p\%$ of smallest-magnitude weights creating mask $m$, (4) **RESET surviving weights to original initialization $\theta_0$**, yielding $f(x; m \odot \theta_0)$, (5) retrain. Step 4 — emphasized as "unique to our work" — changes everything. If instead the surviving weights are *randomly reinitialized* ($f(x; m \odot \theta_0')$), winning-ticket performance collapses. A winning ticket is therefore the *combination* of a sparse architecture **and** its specific original initialization. Experimentally, LeNet on MNIST retains accuracy down to 21.1% sparsity and 7% with iterative pruning; Conv-2/4/6 on CIFAR-10 hit 8.8–15.1% sparsity with faster learning and higher accuracy; VGG-19 and ResNet-18 require **learning-rate warmup** to find winning tickets (1.5%–11.8% sparsity). The paper offers a fresh perspective on over-parameterization, generalization, and optimization landscapes, and won the ICLR 2019 Best Paper Award.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction (§1) / 서론

**한국어**
저자들은 LeCun et al. (1990) 이래의 신경망 가지치기 문헌으로 시작합니다. Han et al. (2015)는 magnitude-based pruning으로 매개변수 수를 90% 이상 줄여도 정확도가 유지됨을 보였습니다. 이는 inference cost와 storage를 크게 줄여 모바일 배포에 유리합니다 (Han et al., 2015; Hinton et al., 2015). 그러나 의문이 남습니다: **"네트워크를 줄일 수 있다면, 처음부터 그 작은 네트워크를 학습하지 못할 이유는 무엇인가?"** 통념적 답: sparse architecture는 처음부터 학습하기 어렵다 (Li et al., 2016).

저자들은 Figure 1에서 이 통념을 보여줍니다. LeNet (MNIST)와 Conv-2/4/6 (CIFAR-10)에서 무작위 sparse subnetwork (dashed lines)는 sparsity가 증가할수록 학습이 느려지고 최종 정확도도 낮아집니다. 그러나 동일 architecture를 가지치기로 발견한 winning ticket (solid lines)은 sparsity가 증가해도 (어느 수준까지는) 학습이 더 빨라지고 정확도가 더 높아집니다. 이 대비가 논문의 출발점입니다.

**Lottery Ticket Hypothesis 형식적 정의** (paper §1):
- $f(x;\theta_0)$: dense 네트워크, $\theta_0 \sim \mathcal{D}_\theta$
- $f$가 SGD로 iteration $j$에서 minimum validation loss와 test accuracy $a$ 달성
- 마스크 $m \in \{0,1\}^{|\theta|}$로 학습한 $f(x; m \odot \theta)$가 iteration $j' \leq j$에서 accuracy $a' \geq a$ 달성, $\|m\|_0 \ll |\theta|$
- 이러한 $m$이 존재한다는 것이 가설

세 가지 조건을 동시에 만족: **commensurate training time** ($j' \leq j$), **commensurate accuracy** ($a' \geq a$), **fewer parameters** ($\|m\|_0 \ll |\theta|$).

**English**
The authors open with the pruning literature since LeCun et al. (1990). Han et al. (2015) showed magnitude-based pruning can reduce parameter counts by over 90% while preserving accuracy — important for inference cost and storage on mobile devices. The natural question: **"if a network can be reduced in size, why not just train that smaller architecture in the first place?"** Conventional answer: sparse architectures uncovered by pruning are harder to train from scratch (Li et al., 2016).

Figure 1 illustrates this. For LeNet (MNIST) and Conv-2/4/6 (CIFAR-10), random sparse subnetworks (dashed) train slower and reach lower accuracy as sparsity increases. But subnetworks found *via pruning* (solid; the winning tickets) train faster and reach higher accuracy as sparsity increases — at least until heavily pruned. This contrast launches the paper.

**Formal hypothesis**: there exists a mask $m \in \{0,1\}^{|\theta|}$ such that $f(x; m \odot \theta_0)$ trained from $\theta_0$ for $j' \leq j$ iterations reaches accuracy $a' \geq a$, with $\|m\|_0 \ll |\theta|$. Three simultaneous conditions: commensurate training time, commensurate accuracy, fewer parameters.

### Part II: The Identification Procedure / 식별 절차

**한국어**
**중심 실험 (paper §1, p. 2)**:
1. 무작위 초기화: $f(x;\theta_0)$, $\theta_0 \sim \mathcal{D}_\theta$
2. $j$ iteration 학습 → $\theta_j$
3. $\theta_j$에서 가장 작은 magnitude의 weight $p\%$ 가지치기 → 마스크 $m$ 생성
4. **남은 weight를 $\theta_0$로 리셋** → winning ticket $f(x; m \odot \theta_0)$
5. (필요시) 다시 학습/가지치기 반복

이 5단계가 "**central experiment**"입니다. Step 4가 핵심: "**Unique to our work, each unpruned connection's value is then reset to its initialization from the original network before it was trained.**" 즉, 살아남은 weight가 학습된 값($\theta_j$)이 아니라 원래 무작위 값($\theta_0$)으로 돌아갑니다.

**One-shot vs Iterative**:
- **One-shot pruning**: 한 번만 학습 → $p\%$ 한꺼번에 가지치기. 빠르지만 작은 winning ticket을 찾기 어려움.
- **Iterative pruning (IMP)**: $n$ 라운드, 각 라운드마다 $p^{1/n}\%$씩 가지치기. 비용은 더 들지만 더 작은 winning ticket을 찾음.

본 논문은 거의 모든 실험에서 iterative pruning을 사용 ("iterative pruning finds winning tickets that match the accuracy of the original network at smaller sizes than does one-shot pruning"). 라운드당 일반적으로 20% (LeNet은 fc20%, Conv는 conv10-20% + fc20%).

**Pruning details**:
- **Layer-wise pruning**: LeNet, Conv-2/4/6에서 각 layer별로 일정 비율 가지치기
- **Global pruning**: ResNet-18, VGG-19에서 모든 layer를 통합해 한꺼번에 가지치기 (작은 layer가 bottleneck이 되는 것 방지)
- **Output layer**: 다른 layer의 절반 비율로 가지치기 (LeNet에서 fc20% → output 10%)

**English**
The **central experiment** (§1, p. 2):
1. Randomly initialize $f(x;\theta_0)$, $\theta_0 \sim \mathcal{D}_\theta$
2. Train for $j$ iterations $\to \theta_j$
3. Prune $p\%$ of smallest-magnitude weights in $\theta_j$ to form mask $m$
4. **Reset surviving weights to $\theta_0$**, yielding winning ticket $f(x; m \odot \theta_0)$
5. (Optionally) iterate

Step 4 is *the* novelty: surviving weights revert to their *original* random values $\theta_0$, not the trained values $\theta_j$. The authors stress: "Unique to our work, each unpruned connection's value is then reset to its initialization from the original network before it was trained."

**One-shot vs iterative**:
- **One-shot**: prune all $p\%$ at once after a single training run. Fast but finds larger winning tickets.
- **Iterative magnitude pruning (IMP)**: $n$ rounds, prune $p^{1/n}\%$ each round. More expensive but finds smaller winning tickets — used throughout the paper.

Per-round pruning rates from Figure 2: LeNet fc20%, Conv-2 conv10%+fc20%, Conv-4/6 conv10-15%+fc20%, ResNet-18 conv20%+fc0%, VGG-19 conv20%+fc0%. Output-layer pruning is at half the rate of other layers.

**Layer-wise vs global pruning**:
- LeNet, Conv-2/4/6: layer-wise (each layer pruned at its own rate)
- ResNet-18, VGG-19: **global** (rank smallest weights across all conv layers and remove jointly) — necessary because layer sizes differ by 1000× (1728 vs 2.35M parameters), so per-layer rates would let small layers become bottlenecks

### Part III: Winning Tickets in Fully-Connected Networks (§2) / 완전 연결망에서의 승리 티켓

**한국어**
**Architecture (Figure 2)**: LeNet-300-100 (LeCun et al., 1998), 266K 매개변수. fc layer 3개 (300, 100, 10 units).
**Training**: Adam lr 1.2e-3, 50K iterations, batch 60. Pruning rate fc20% per round (output 10%).

**Iterative pruning 결과 (Figure 3)**:
- $P_m = 100\%$ (원본): test accuracy ~98.2%, early-stopping iter ~14K
- $P_m = 51.3\%$: 학습 빨라짐, 정확도 ~98.5%
- $P_m = 21.1\%$: 학습 38% 더 빠름 (early-stopping at ~9K vs 14K). 정확도가 13.5%p보다 더 향상.
- $P_m = 7.0\%$: 여전히 winning ticket, 학습 살짝 느림
- $P_m = 3.6\%$: 학습 느려짐, 정확도 원본과 비슷한 수준
- $P_m = 1.9\%$: winning ticket 사라짐, 정확도 원본 미만

**핵심 수치**:
- "winning tickets learn faster as $P_m$ decreases from 100% to 21%, at which point early-stopping occurs **38% earlier** than for the original network"
- "test accuracy increases with pruning, improving by more than **0.3 percentage points** when $P_m = 13.5\%$"
- "after this point, accuracy decreases, returning to the level of the original network when $P_m = 3.6\%$"

**Random reinitialization 대조 실험 (Figure 4a, orange)**:
같은 구조 $m$이지만 새 무작위 초기화 $\theta_0' \sim \mathcal{D}_\theta$. 결과: "the average reinitialized iterative winning ticket's test accuracy drops off from the original accuracy when $P_m = 21.1\%$, compared to **2.9% for the winning ticket**." 즉 winning ticket은 7배 더 작아질 수 있을 때까지 정확도 유지. "When $P_m = 21\%$, the winning ticket reaches minimum validation loss **2.51× faster** than when reinitialized and is half a percentage point more accurate."

**One-shot pruning 결과 (Figure 4c)**:
"When $67.5\% \geq P_m \geq 17.6\%$, the average winning tickets reach minimum validation accuracy earlier than the original network. When $95.0\% \geq P_m \geq 5.17\%$, test accuracy is higher than the original network." 즉 one-shot도 winning ticket을 찾지만, 더 작은 winning ticket은 iterative만 찾음.

**English**
**Architecture (Figure 2)**: LeNet-300-100 (LeCun et al., 1998), 266K parameters, 3 FC layers (300, 100, 10).
**Training**: Adam lr 1.2e-3, 50K iters, batch 60. Per-round pruning rate fc20% (output 10%).

**Iterative pruning results (Figure 3)**:
- $P_m=100\%$ (full): test acc ~98.2%, early-stop ~14K iters
- $P_m=51.3\%$: faster, ~98.5%
- $P_m=21.1\%$: 38% faster early-stopping, accuracy gain >0.3pp by $P_m=13.5\%$
- $P_m=7.0\%$: still a winning ticket, slightly slower
- $P_m=3.6\%$: returns to original-network level
- $P_m=1.9\%$: lottery ticket effect lost

Quoted: winning tickets reach early-stopping **38% earlier** at $P_m=21\%$; accuracy improves >0.3pp by 13.5%; effect dies near 3.6%.

**Random reinitialization (Figure 4a, orange)**: same mask $m$, fresh init $\theta_0' \sim \mathcal{D}_\theta$. Reinitialized winning tickets fall off accuracy when $P_m=21.1\%$, vs **2.9%** for the original-init winning ticket. At $P_m=21\%$: winning ticket reaches min val loss **2.51× faster** and is +0.5pp more accurate.

**One-shot pruning (Figure 4c)**: winning tickets exist for $67.5\%\geq P_m \geq 17.6\%$ (faster) and $95.0\%\geq P_m \geq 5.17\%$ (higher accuracy). One-shot does find winning tickets but iterative reaches smaller sizes.

### Part IV: Winning Tickets in Convolutional Networks (§3) / 합성곱 신경망에서의 승리 티켓

**한국어**
**Architectures (Figure 2)**: VGG-style scaled-down for CIFAR-10:
- **Conv-2**: 2 conv layers (64,64) + pool + 3 fc (256,256,10). 4.3M params.
- **Conv-4**: 4 conv (64,64,128,128) + pools + fc. 2.4M params.
- **Conv-6**: 6 conv (64,64,128,128,256,256) + pools + fc. 1.7M params.

**Training**: Adam, 20K/25K/30K iterations, batch 60. Pruning conv10-15%, fc20% per round.

**결과 (Figure 5)**:
| Network | Best $P_m$ (faster early-stop) | Speedup | Best $P_m$ (higher accuracy) | Acc gain |
|---|---|---|---|---|
| Conv-2 | 8.8% | 3.5× | 4.6% | +3.4pp |
| Conv-4 | 9.2% | 3.5× | 11.1% | +3.5pp |
| Conv-6 | 15.1% | 2.5× | 26.4% | +3.3pp |

"All three networks remain above their original average test accuracy when $P_m \geq 2\%$." 즉 50배 압축까지 정확도 유지.

**Generalization 향상**: training accuracy도 비슷하게 상승하지만, 학습 종료 시점(20K/25K/30K iter)의 training accuracy는 거의 모든 네트워크에서 100%에 도달 ($P_m \geq 2\%$). 그러나 winning ticket의 test accuracy는 더 높음 → **train-test gap이 더 작음** = 일반화가 더 좋음.

**Random reinitialization (Figure 5 dashed)**: Conv-2/4에서는 reinitialized network의 early-stopping 시점 정확도가 winning ticket보다 빨리 무너짐. 그러나 final test accuracy 차이는 Lenet만큼 크지 않음 — Conv는 inductive bias가 강하기 때문 (Cohen & Shashua, 2016).

**Dropout 효과 (Figure 6)**: dropout 0.5로 학습한 Conv-2/4/6에서 IMP 적용. dropout이 정확도를 +2.1, +3.0, +2.4pp 향상; iterative pruning이 추가로 +2.3, +4.6, +4.7pp 향상. 두 기법이 **상보적(complementary)**임을 시사. Dropout은 sparsity-inducing dropout (Srivastava et al., 2014) 관점에서 winning ticket과 자연스럽게 상호작용.

**English**
**Architectures (Fig 2)**: VGG-style for CIFAR-10:
- Conv-2: 2 conv (64,64) + 3 fc. 4.3M params.
- Conv-4: 4 conv (64,64,128,128) + fc. 2.4M params.
- Conv-6: 6 conv (64,64,128,128,256,256) + fc. 1.7M params.

**Results (Fig 5)**: best winning tickets for early-stop speedup:
- Conv-2: $P_m=8.8\%$, **3.5×** faster, +3.4pp at $P_m=4.6\%$
- Conv-4: $P_m=9.2\%$, **3.5×** faster, +3.5pp at $P_m=11.1\%$
- Conv-6: $P_m=15.1\%$, **2.5×** faster, +3.3pp at $P_m=26.4\%$

All three networks stay above original test accuracy at $P_m \geq 2\%$ — ~50× compression with accuracy preserved.

**Train-test gap**: at end of training, training acc ~100% for all $P_m \geq 2\%$, but winning-ticket test acc is higher — generalization improves.

**Dropout (Fig 6)**: Dropout 0.5 adds +2.1, +3.0, +2.4pp; iterative pruning then adds +2.3, +4.6, +4.7pp on top — complementary.

### Part V: VGG and ResNet for CIFAR-10 (§4) / 깊은 네트워크에서

**한국어**
이 섹션이 가장 미묘합니다. 깊은 network에서는 단순 IMP가 항상 작동하지 않습니다.

**VGG-19 (20M params)**:
- Liu et al. (2019)의 setup: 160 epochs (112,480 iters), SGD momentum 0.9, lr 0.1 → 0.01 → 0.001 (factor of 10 at epochs 80, 120)
- Global pruning, conv 20%, fc 0% per round
- **lr 0.01 (낮은 lr)**: IMP가 winning ticket 발견 (Figure 7 가운데). $P_m \geq 3.5\%$까지 원본 정확도와 1pp 이내.
- **lr 0.1 (높은 lr, 표준 훈련)**: IMP가 winning ticket을 못 찾음. random reinit과 동일한 성능.
- **해법: learning rate warmup**. $k=10000$ iter 동안 lr을 0에서 0.1로 선형 증가. 결과: warmup으로 lr 0.1 + winning ticket 가능. $P_m \geq 1.5\%$까지 가능.

**ResNet-18 (271K params)**:
- 30K iterations, SGD momentum 0.9, lr 0.1 → 0.01 → 0.001 (factor of 10 at 20K, 25K)
- Global pruning, conv 20%, fc 0%
- **lr 0.01**: best winning ticket at $P_m=21.9\%$ → 89.5% (원본 90.5% 대비 -1pp). winning ticket이 학습 빠르나 후반에 따라잡힘.
- **lr 0.1**: winning ticket 안 보임.
- **lr 0.03 + warmup ($k=20000$)**: 90.5% test acc at $P_m=27.1\%$. winning tickets at $P_m \geq 11.8\%$.
- 그러나 lr 0.1로 winning ticket을 찾는 것은 실패.

**핵심 통찰**: 깊은 network에서 IMP는 **learning rate에 민감**. 큰 lr이 좋은 정확도와 일반화에 필요하지만 winning ticket 발견에는 방해. Warmup이 둘을 화해시킴.

**English**
This is the most nuanced section. For deeper networks, naive IMP doesn't always work.

**VGG-19 (20M params)**: 160 epochs, SGD-momentum, lr schedule 0.1→0.01→0.001 at epochs 80/120. Global pruning, conv20%/fc0%.
- lr=0.01: IMP finds winning tickets (Fig 7 center) — within 1pp of original down to $P_m \geq 3.5\%$.
- lr=0.1: IMP fails — performs same as random reinit.
- lr=0.1 + linear warmup over $k=10000$ iters: winning tickets recovered, $P_m \geq 1.5\%$.

**ResNet-18 (271K params)**: 30K iters, SGD-momentum, lr 0.1→0.01→0.001 at 20K/25K.
- lr=0.01: best ticket 89.5% at $P_m=21.9\%$ (vs original 90.5% at lr=0.1, so -1pp).
- lr=0.1: no winning tickets.
- lr=0.03 + warmup ($k=20000$): matches original 90.5% at $P_m=27.1\%$, winning tickets $P_m \geq 11.8\%$.
- lr=0.1 + warmup: still couldn't recover winning tickets.

**Insight**: IMP is **lr-sensitive** in deep nets. Large lr improves final accuracy but disrupts winning-ticket discovery; warmup partially reconciles them.

### Part VI: Discussion (§5) / 논의

**한국어**

**(a) Importance of winning ticket initialization**:
- random reinit하면 winning ticket이 더 느리게 학습하고 더 낮은 정확도 도달.
- "These initial weights are close to their final values after training" — Liu et al. (2019)의 가설을 검증: Appendix F는 반대를 보임. winning ticket weight는 다른 weight보다 **더 많이** 움직임. 즉 "이미 학습된" 상태가 아님.
- **잠재적 설명**: 초기화가 optimization 알고리즘과 데이터에 따라 정해진 loss landscape의 특정 region에 위치하여 그 알고리즘으로 잘 최적화됨.

**(b) Liu et al. (2019)와의 관계**:
- Liu et al.은 pruned network가 random reinit으로도 학습 가능하다고 주장.
- 부분적으로 일치: VGG-19에서 80% 가지치기까지는 random reinit이 원본 성능 매칭. 하지만 그 이상으로 가지치기하면 (논문에서 다루지 않은 영역), winning ticket만 정확도 유지.
- 결론: "**up to a certain level of sparsity—highly overparameterized networks can be pruned, reinitialized, and retrained successfully; however, beyond this point, extremely pruned, less severely overparameterized networks only maintain accuracy with fortuitous initialization.**"

**(c) Importance of structure**:
- winning ticket의 sparse architecture는 학습 과제에 맞춤화된 inductive bias 인코딩 (Cohen & Shashua, 2016 인용).
- 학습 데이터로 발견되었기 때문에 random sparse mask보다 더 좋음.

**(d) Improved generalization (Occam's Hill)**:
- "test accuracy increases and then decreases as we prune, forming an Occam's Hill" (Rasmussen & Ghahramani, 2001)
- 원본은 too much complexity (overfitting); extremely pruned는 too little. 적절한 sparsity가 sweet spot.
- Zhou et al. (2018)과 Arora et al. (2018)이 compression-based generalization bound 제시. Lottery ticket은 이 관점에서: **larger networks might explicitly contain simpler representations**.

**(e) Implications for optimization (Du et al. 2019)**:
- Du et al. (2019): 충분히 over-parameterized 2-layer ReLU 네트워크는 SGD로 global optimum에 수렴.
- Lottery ticket 관점에서의 conjecture: "**SGD seeks out and trains a well-initialized subnetwork**". over-parameterization은 더 많은 winning ticket 후보를 제공하기 때문에 학습이 쉬워짐.

**English**

**(a) Initialization importance**: random reinit collapses winning tickets. App. F shows winning-ticket weights move *more* than other weights — they are *not* "already trained" but land in regions amenable to the chosen optimizer.

**(b) Reconciliation with Liu et al. (2019)**: agreement up to a sparsity threshold (~80% prune for VGG-19) where random reinit matches; beyond that, only winning-ticket initialization preserves accuracy.

**(c) Structure**: sparse architecture itself encodes task-specific inductive bias (Cohen & Shashua, 2016).

**(d) Occam's Hill generalization**: test acc rises then falls with pruning — original overfits, heavily pruned underfits, middle is the sweet spot. Connects to Zhou et al. (2018), Arora et al. (2018) compression-based bounds: large networks may *explicitly contain* simpler representations.

**(e) Optimization (Du et al. 2019)**: conjecture — SGD finds and trains well-initialized subnetworks; over-parameterization helps because more winning-ticket candidates exist.

### Part VII: Limitations and Related Work (§6, §7) / 한계 및 관련 연구

**한국어**

**한계**:
1. **Vision-only**: MNIST, CIFAR-10만. ImageNet 미실험 (iterative pruning이 너무 비쌈 — 15+ 연속 학습).
2. **Unstructured sparse pruning**: 현대 hardware에서 효율적 가속 어려움. structured pruning (filter/channel) 미연구.
3. **Magnitude만 사용**: 다른 pruning heuristic (second-order, importance) 미연구.
4. **Deep network는 warmup 필요**: 왜 필요한지 미해명.

**Related work 분류 (§7)**:
- **Prior to training**: SqueezeNet (Iandola et al., 2016), MobileNets (Howard et al., 2017) — 직접 작은 architecture 설계
- **After training**: distillation (Ba & Caruana, 2014; Hinton et al., 2015), pruning (LeCun et al., 1990; Han et al., 2015; Guo et al., 2016; Hu et al., 2016)
- **During training**: Bellec et al. (2018) deep rewiring, Srinivas et al. (2017), Louizos et al. (2018) $\ell_0$ regularization
- **Filter Lottery (Cohen et al., 2016)**: filter는 초기화에 민감. throughout training random reinit unimportant filters → 초기화 중요성 시사 (LTH의 선조 격)

**English**
**Limitations**: vision-only (MNIST, CIFAR-10); unstructured sparse pruning (not hardware-friendly); only magnitude-based heuristic; warmup needed for deep nets without explanation.

**Related work taxonomy (§7)**:
- **Prior to training**: SqueezeNet, MobileNets — directly engineer small models.
- **After training**: distillation (Ba & Caruana, Hinton et al.), pruning (LeCun et al. 1990, Han et al. 2015, Guo, Hu).
- **During training**: deep rewiring (Bellec et al. 2018), Srinivas et al., Louizos $\ell_0$.
- **Filter Lottery (Cohen et al. 2016)**: precursor — observes filter initialization sensitivity.

### Part VIII: Appendix highlights / 부록 핵심

**한국어**

**Appendix B (두 IMP 전략)**:
- **Strategy 1 (resetting, 본문에서 사용)**: 매 라운드 후 weight를 $\theta_0$로 reset 후 재학습.
- **Strategy 2 (continued training)**: reset 없이 학습된 weight에서 계속 학습. 마지막에만 reset.
- Figure 9, 10: Strategy 1이 일관되게 더 높은 정확도 + 더 빠른 early-stopping at 작은 sparsity. 따라서 본문은 Strategy 1 사용.

**Appendix C (Early stopping criterion)**:
- "iteration of minimum validation loss"를 학습 속도 proxy로 사용. validation loss가 감소→최소→증가 패턴을 따른다는 가정.
- Figure 11: validation loss 패턴이 명확함을 확인.

**Appendix F (Winning ticket initialization 분포)** — 매우 흥미로움:
- Figure 15: $P_m$이 작아짐에 따라 second hidden layer와 output layer의 winning ticket 초기화가 **bimodal**이 됨 — 0의 양쪽으로 peak 형성. 즉 winning ticket weight는 magnitude가 큰 weight들로 구성.
- Asymmetric: second hidden layer는 positive peak가 더 큼, output layer는 negative peak가 더 큼.
- First hidden layer는 분포 유지 (Glorot 그대로).
- F.3: 이 bimodal 분포 $\mathcal{D}_m$에서 새로 sampling해도 winning ticket 성능 회복 안 됨 → **특정 weight 값 자체가 중요**, 분포 모양만으로는 부족.
- F.4 (iteration 0에서 가지치기): 처음부터 작은 weight를 가지치기 시도 → winning ticket보다 더 나쁨. 학습 과정이 "어느 weight가 중요한지" 식별하는 데 필수임을 시사.

**English**
**App. B**: Strategy 1 (reset every round) > Strategy 2 (continued training). Figs 9-10 confirm.

**App. C**: early stopping = iteration of min validation loss; Fig 11 shows the typical decrease-min-increase shape.

**App. F (fascinating)**:
- Fig 15: as $P_m$ shrinks, hidden-2 and output-layer winning-ticket initializations become **bimodal** — peaks on either side of 0. Winning-ticket weights are large-magnitude.
- Peaks are asymmetric (hidden-2 favors positive; output favors negative); hidden-1 keeps Glorot shape.
- F.3: sampling from $\mathcal{D}_m$ doesn't recover winning tickets — specific values matter, not just distributional shape.
- F.4: pruning at iteration 0 (smallest at init) is worse than IMP — training is necessary to identify which weights matter.

---

## 3. Key Takeaways / 핵심 시사점

1. **The hypothesis itself reframes neural network training / 가설 자체가 신경망 학습을 재조명**
   "**Dense, randomly-initialized networks contain trainable sparse subnetworks**". Over-parameterization은 단순 능력 증가가 아니라 **더 많은 winning ticket 후보를 포함하기 위함**일 수 있다는 새로운 관점. / Over-parameterization may not be about capacity per se but about having more lottery-ticket candidates — a paradigm shift in thinking about why large networks train well.

2. **Reset to $\theta_0$ is the load-bearing innovation / $\theta_0$로의 reset이 핵심 혁신**
   기존 magnitude pruning은 거의 동일하지만 step 4 (reset)만 추가. 이것 없이는 (random reinit) winning ticket 효과 사라짐. Han et al. (2015)와 본 논문의 차이가 이 한 줄임. / The technical novelty is a single line of code — yet without it (e.g., random reinit), winning-ticket gains evaporate. The whole paper hinges on this.

3. **Iterative pruning finds smaller winning tickets than one-shot / IMP가 더 작은 winning ticket 발견**
   LeNet에서 one-shot은 17%까지, iterative는 7%까지. Conv-6는 iterative로 26.4%까지 정확도 향상. 비용 trade-off는 분명하지만 가장 sparse한 winning ticket을 찾으려면 IMP 필수. / IMP costs more compute (15+ training cycles) but yields significantly smaller winning tickets — necessary for the smallest viable subnetworks.

4. **Architecture matters but isn't sufficient — initialization matters too / Architecture만으로는 부족, 초기화가 함께 중요**
   random reinit (같은 mask, 다른 init): 정확도 빠르게 무너짐. random sparsity (다른 mask, random init): 더 빨리 무너짐. winning ticket = 두 요소의 결합. / Random reinit collapses winning tickets faster than random masks — the *combination* of structure and initialization is what wins.

5. **Learning rate warmup unlocks deep networks / Warmup이 깊은 네트워크를 연다**
   VGG-19, ResNet-18에서 표준 lr (0.1)로는 IMP 실패. Linear warmup (k=10K-20K iter)으로 lr 0.03-0.1에서도 winning ticket 발견. 이것이 LTH의 robustness 한계와 follow-up 연구 (Frankle 2020 rewinding)의 출발점. / Without warmup, IMP can't find winning tickets at standard learning rates for deeper nets — a key practical caveat that motivated the rewinding extension.

6. **Generalization improves as Occam's Hill / 일반화가 Occam's Hill 형태로 개선**
   pruning은 자동 regularization. test accuracy ↑→max→↓; train-test gap이 winning ticket에서 더 작음. Zhou (2018), Arora (2018) compression-based generalization bound와 연결. / Pruning acts as automatic regularization; the train-test gap shrinks, connecting LTH to compression-based generalization theory.

7. **Bimodal winning-ticket initialization distribution / Winning ticket의 이중 모드 분포**
   App F: 살아남은 weight는 magnitude가 큰 weight들. 그러나 이 분포에서 새로 sampling해도 효과 없음 → 특정 weight 값 자체가 중요. 분포만으로는 winning ticket을 만들 수 없음. / Surviving weights cluster at large magnitudes but resampling from that distribution doesn't reproduce winning-ticket performance — specific values matter.

8. **Implications for optimization theory / 최적화 이론에 대한 시사점**
   "SGD seeks out and trains a well-initialized subnetwork" conjecture. Du et al. (2019) over-parameterization → global optimum 결과와 맥락 일치. NTK 이론, mean-field analysis, lazy regime 연구에 새 prior 제공. / The paper's conjecture — SGD effectively trains lottery winners — links empirical findings to over-parameterization theory (Du, Arora, Zhou).

---

## 4. Mathematical Summary / 수학적 요약

### Formal hypothesis statement / 가설의 형식적 진술

**한국어**
$f(x;\theta)$를 dense feed-forward 네트워크라 하자. SGD로 학습 시 $f(x;\theta_0)$, $\theta_0 \sim \mathcal{D}_\theta$가 iteration $j$에서 minimum validation loss와 test accuracy $a$ 달성. Lottery Ticket Hypothesis는:

$$\exists\, m \in \{0,1\}^{|\theta|}: \text{ training } f(x; m \odot \theta_0) \text{ from } \theta_0 \text{ yields} \begin{cases} \text{validation loss minimum at } j' \leq j & \text{(commensurate training)} \\ \text{test accuracy } a' \geq a & \text{(commensurate accuracy)} \\ \|m\|_0 \ll |\theta| & \text{(fewer parameters)} \end{cases}$$

**English**
For dense network $f(x;\theta)$ trained with SGD, where $f(x;\theta_0)$ with $\theta_0 \sim \mathcal{D}_\theta$ reaches min val loss at iteration $j$ with test acc $a$, the LTH asserts: there exists $m \in \{0,1\}^{|\theta|}$ such that training $f(x; m \odot \theta_0)$ from $\theta_0$ reaches min val loss at $j' \leq j$ with test acc $a' \geq a$ and $\|m\|_0 \ll |\theta|$.

### Sparsity definition / 희소도 정의

$$P_m = \frac{\|m\|_0}{|\theta|} = \frac{\#\{i : m_i = 1\}}{|\theta|}$$

- $P_m \in [0, 1]$: 남은 weight 비율 / fraction remaining
- $1 - P_m$: 가지치기된 weight 비율 / fraction pruned
- 예 / e.g., $P_m = 0.211$ means 78.9% pruned

### Iterative Magnitude Pruning (IMP) algorithm

```
Algorithm: Iterative Magnitude Pruning (Strategy 1, Resetting)
Input: 
    f(x; θ): network with parameters θ
    j: training iterations per round
    p: total prune fraction
    n: number of rounds
    s_round: per-round prune rate (typically 0.2)
Output: Mask m identifying winning ticket f(x; m ⊙ θ₀)

1. θ₀ ~ D_θ                           // random init (e.g., Glorot)
2. m ← 1 ∈ {0,1}^|θ|                  // start with full mask
3. for t = 1 to n:                    // n rounds
4.     θ_j ← Train(f(x; m ⊙ θ₀), j iterations)
5.     // Identify smallest s_round fraction of currently-active weights
6.     τ_t ← quantile(|m ⊙ θ_j|, s_round, ignore zeros)
7.     for i in indices where m_i = 1:
8.         if |θ_{j,i}| ≤ τ_t:
9.             m_i ← 0               // prune
10.    // Reset: surviving weights revert to θ₀
11. return m, f(x; m ⊙ θ₀)
```

**Per-round rate relationship / 라운드당 비율 관계**:
- LeNet에서 fc20% per round (s_round = 0.2)
- $n$ 라운드 후 sparsity: $P_m^{(n)} = (1 - 0.2)^n \cdot 100\% = 0.8^n$
- 예 / Examples: $0.8^1 = 80.0\%, \ 0.8^5 = 32.8\%, \ 0.8^{10} = 10.7\%, \ 0.8^{15} = 3.5\%, \ 0.8^{20} = 1.2\%$

### Pruning mask formalization / 가지치기 마스크 형식화

**Forward pass with mask** / 마스크가 있는 순전파:
$$h_l = \sigma\!\left( (m_l \odot W_l) h_{l-1} + b_l \right)$$
where $m_l \in \{0,1\}^{n_l \times n_{l-1}}$ is the layer-$l$ mask and $\odot$ is Hadamard (elementwise) product.

**Backward pass with mask** / 마스크가 있는 역전파 (gradient through frozen mask):
$$\frac{\partial \mathcal{L}}{\partial W_l} = m_l \odot \frac{\partial \mathcal{L}}{\partial \tilde{W}_l}, \qquad \tilde{W}_l = m_l \odot W_l$$

마스크는 "frozen"이므로 backprop이 마스킹된 weight에는 gradient가 흐르지 않습니다. / Mask is frozen so gradients don't update pruned weights.

### Loss with mask / 마스크가 있는 손실 함수

$$\mathcal{L}(\theta; m) = \frac{1}{N} \sum_{i=1}^{N} \ell\bigl(f(x_i; m \odot \theta), y_i\bigr) + \lambda \|m \odot \theta\|^2_2$$

$\ell$은 cross-entropy, $\lambda$는 weight decay (VGG/ResNet 실험에 사용).

### Worked example: LeNet-300-100 IMP trace / 워크스루 예제

**한국어**
LeNet-300-100, fc20% per round.
- Round 0 (full): 266K weights, 100%, accuracy 98.2%, early-stop ~14K iter
- Round 1: $0.8 \times 100\% = 80.0\%$, accuracy ~98.3%, early-stop ~13K
- Round 5: $0.8^5 \approx 32.8\%$, accuracy ~98.4%, early-stop ~10K
- Round 7: $0.8^7 \approx 21.0\%$, accuracy ~98.5%, early-stop ~8.7K (38% faster)
- Round 12: $0.8^{12} \approx 6.9\%$, accuracy ~98.2% (commensurate), early-stop ~9K
- Round 16: $0.8^{16} \approx 2.8\%$, accuracy ~98.0% (slight drop)
- Round 20: $0.8^{20} \approx 1.2\%$, accuracy <97.5% (winning ticket lost)

각 라운드에서 step 4 (reset to $\theta_0$)를 거치며 매번 같은 초기 weight 값으로 돌아갑니다. 학습된 magnitudes만 가지치기 결정에 사용.

**English**
LeNet trace at 20% per round: starts at $P_m=100\%$, reaches $P_m=21.0\%$ in 7 rounds (commensurate accuracy + 38% faster), $P_m=6.9\%$ in 12 rounds (still matches), $P_m=1.2\%$ in 20 rounds (winning ticket dies).

### Comparison: number of weights at $P_m$ levels / 비교: 각 sparsity의 weight 수

| Network | Total | $P_m=21\%$ | $P_m=10\%$ | $P_m=2\%$ |
|---|---|---|---|---|
| LeNet-300-100 | 266K | 56K | 27K | 5.3K |
| Conv-2 | 4.3M | 903K | 430K | 86K |
| Conv-6 | 1.7M | 357K | 170K | 34K |
| ResNet-18 | 271K | 57K | 27K | 5.4K |
| VGG-19 | 20M | 4.2M | 2.0M | 400K |

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1990 ─ LeCun, Denker, Solla: Optimal Brain Damage
        │       (second-derivative based pruning)
        │
1993 ─ Hassibi & Stork: Optimal Brain Surgeon
        │
1998 ─ LeCun et al.: LeNet-5 (the "LeNet" architecture used here)
        │
2010 ─ Glorot & Bengio: Glorot init (used throughout)
        │
2014 ─ Ba & Caruana: "Do deep nets really need to be deep?"
        │       (knowledge distillation precursor)
        │
2014 ─ Simonyan & Zisserman: VGG (basis for VGG-19)
        │
2015 ─ Hinton et al.: Distilling knowledge in NN
        │
2015 ─ Han et al.: Deep Compression
        │       ★ Direct predecessor — magnitude pruning to 90%+
        │
2016 ─ Li et al.: Pruning filters for efficient ConvNets
        │       (observes pruned ≠ trainable from scratch)
        │
2016 ─ He et al.: ResNet (basis for ResNet-18 used here)
        │
2016 ─ Cohen et al.: Filter Lottery (initialization sensitivity)
        │       ★ Conceptual precursor
        │
2018 ─ Bellec et al.: Deep rewiring
        │
2019 ─ ★★★ Frankle & Carbin: Lottery Ticket Hypothesis ★★★
        │       (ICLR 2019 Best Paper)
        │
2019 ─ Liu et al.: "Rethinking the Value of Network Pruning"
        │       (partial conflict — random reinit can work for some sparse archs)
        │
2019 ─ Du et al.: SGD provably optimizes over-parameterized nets
        │       (theoretical context for LTH conjecture)
        │
2020 ─ Frankle et al.: Linear Mode Connectivity & Rewinding
        │       (extends LTH — rewind to early θ_k instead of θ_0)
        │
2020 ─ Ramanujan et al.: Strong Lottery Tickets
        │       (subnetworks of random nets work without training!)
        │
2020 ─ Lee et al. SNIP, Wang et al. GraSP, Tanaka et al. SynFlow:
        │       (find lottery tickets at initialization, no training)
        │
2020 ─ Chen et al.: LTH on BERT
2021 ─ Chen et al.: LTH on Vision Transformers
        │
2022+─ LTH on foundation models, Diffusion, RLHF
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **Han et al. (2015) — Deep Compression** | 직접 선조 — magnitude pruning. LTH는 한 줄 추가 (step 4 reset)로 차별화 / Direct predecessor — magnitude pruning. LTH adds the one-line reset step | High — entire paper structure mirrors Han's pipeline / 전체 파이프라인이 Han에서 출발 |
| **#13 Krizhevsky et al. (2012) — AlexNet** | over-parameterization 시대를 연 모델. LTH는 그 이유를 "더 많은 winning ticket 후보"로 재해석 / Opened the over-parameterization era; LTH reframes it as more lottery candidates | High — context for why over-parameterization works / Over-parameterization 작동 이유의 맥락 |
| **#19 Ioffe & Szegedy (2015) — Batch Normalization** | VGG-19, ResNet-18 실험에 사용된 batchnorm은 IMP의 lr-sensitivity와 상호작용 / Batchnorm in VGG-19/ResNet-18 experiments interacts with IMP's lr sensitivity | Medium — practical training detail / 실험적 학습 디테일 |
| **#20 He et al. (2015) — ResNet** | ResNet-18을 직접 실험 architecture로 사용. residual connection이 winning ticket 발견에 영향 / ResNet-18 used as test architecture; residuals affect ticket discovery | High — architecture under direct study / 직접 연구된 architecture |
| **#18 Kingma & Ba (2014) — Adam** | LeNet, Conv-2/4/6 실험에 Adam 사용. winning ticket이 optimizer-specific일 가능성 시사 / Adam used for LeNet/Conv experiments — winning tickets may be optimizer-specific | Medium — experimental method / 실험적 방법 |
| **Hinton et al. (2015) — Knowledge Distillation** | "after training" 압축 카테고리 (§7); LTH는 같은 sparse network를 처음부터 학습 가능함을 보여 distillation의 필요성 재조명 / "After training" compression category; LTH shows sparse network trainable from scratch | Medium — alternative compression paradigm / 대안적 압축 패러다임 |
| **#28 Zhang et al. (2017) — Understanding deep learning requires rethinking generalization** | over-parameterization과 generalization의 미스터리 — LTH의 Occam's Hill가 부분적 답 제시 / Over-parameterization and generalization mystery — LTH's Occam's Hill offers a partial answer | High — generalization theory connection / 일반화 이론 연결 |
| **Liu et al. (2019) — Rethinking Network Pruning** | LTH와 부분 충돌. random reinit가 어떤 sparse network에 작동. Frankle은 fine-grained vs structured pruning 차이로 설명 / Partial conflict — random reinit works for some sparse archs; Frankle attributes to fine-grained vs structured | High — direct dialogue / 직접 논쟁 |
| **Du et al. (2019) — SGD on over-parameterized nets** | LTH conjecture의 이론적 토대: over-parameterization → global optimum. LTH는 그 메커니즘으로 winning ticket을 제안 / Theoretical context for LTH conjecture; LTH proposes winning tickets as the mechanism | High — theoretical underpinning / 이론적 기초 |
| **Frankle et al. (2020) — Linear Mode Connectivity** | Frankle의 follow-up. $\theta_0$ 대신 학습 초기 $\theta_k$로 rewind = "rewinding" — large model에서 더 robust / Author's own follow-up. Rewinding to early-iter $\theta_k$ instead of $\theta_0$; more robust at scale | High — direct extension / 직접적 확장 |
| **Ramanujan et al. (2020) — Strong Lottery Ticket** | LTH의 강한 버전: random initialization에서 학습 없이 좋은 subnet 발견 가능 / Stronger form: subnetworks of random init networks work without any training | High — theoretical extension / 이론적 확장 |

---

## 7. References / 참고문헌

### Primary paper / 본 논문
- Frankle, J., & Carbin, M. (2019). The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks. *International Conference on Learning Representations (ICLR 2019)*. arXiv:1803.03635.
- Code: https://github.com/google-research/lottery-ticket-hypothesis

### Direct predecessors / 직접 선조
- Han, S., Pool, J., Tran, J., & Dally, W. (2015). Learning both weights and connections for efficient neural network. *NeurIPS 2015*.
- Han, S., Mao, H., & Dally, W. (2016). Deep Compression. *ICLR 2016*.
- LeCun, Y., Denker, J. S., & Solla, S. A. (1990). Optimal Brain Damage. *NeurIPS 1990*.
- Hassibi, B., & Stork, D. G. (1993). Second order derivatives for network pruning: Optimal Brain Surgeon. *NeurIPS 1993*.
- Li, H., Kadav, A., Durdanovic, I., Samet, H., & Graf, H. P. (2017). Pruning Filters for Efficient ConvNets. *ICLR 2017*.

### Architectures used / 사용된 architecture
- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proc. IEEE*, 86(11), 2278–2324.
- Simonyan, K., & Zisserman, A. (2015). Very deep convolutional networks for large-scale image recognition. *ICLR 2015* (VGG).
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR 2016* (ResNet).
- Liu, Z., Sun, M., et al. (2019). Rethinking the Value of Network Pruning. *ICLR 2019* (VGG-19 setup).

### Optimization & initialization / 최적화 및 초기화
- Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR 2015*.
- Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. *AISTATS 2010*.
- Du, S. S., Zhai, X., Poczos, B., & Singh, A. (2019). Gradient descent provably optimizes over-parameterized neural networks. *ICLR 2019*.

### Generalization theory / 일반화 이론
- Arora, S., Ge, R., Neyshabur, B., & Zhang, Y. (2018). Stronger generalization bounds for deep nets via a compression approach. *ICML 2018*.
- Zhou, W., Veitch, V., Austern, M., Adams, R. P., & Orbanz, P. (2018). Compressibility and generalization in large-scale deep learning. *arXiv:1804.05862*.
- Zhang, C., Bengio, S., Hardt, M., Recht, B., & Vinyals, O. (2017). Understanding deep learning requires rethinking generalization. *ICLR 2017*.
- Rasmussen, C. E., & Ghahramani, Z. (2001). Occam's Razor. *NeurIPS 13*.

### Related compression literature / 관련 압축 문헌
- Ba, J., & Caruana, R. (2014). Do deep nets really need to be deep? *NeurIPS 2014*.
- Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. *arXiv:1503.02531*.
- Iandola, F. N., Han, S., et al. (2016). SqueezeNet. *arXiv:1602.07360*.
- Howard, A. G., et al. (2017). MobileNets. *arXiv:1704.04861*.
- Bellec, G., Kappel, D., Maass, W., & Legenstein, R. (2018). Deep Rewiring: Training Very Sparse Deep Networks. *ICLR 2018*.
- Cohen, J. P., Lo, H. Z., & Ding, W. (2016). Randomout: Using a convolutional gradient norm to win the filter lottery. *ICLR Workshop 2016*.
- Cohen, N., & Shashua, A. (2016). Inductive bias of deep convolutional networks through pooling geometry. *arXiv:1605.06743*.

### Follow-up / 후속 연구
- Frankle, J., Dziugaite, G. K., Roy, D. M., & Carbin, M. (2020). Linear mode connectivity and the lottery ticket hypothesis. *ICML 2020*.
- Liu, Z., Sun, M., Zhou, T., Huang, G., & Darrell, T. (2019). Rethinking the Value of Network Pruning. *ICLR 2019*.
- Ramanujan, V., Wortsman, M., Kembhavi, A., Farhadi, A., & Rastegari, M. (2020). What's hidden in a randomly weighted neural network? *CVPR 2020*.
- Lee, N., Ajanthan, T., & Torr, P. (2019). SNIP: Single-shot Network Pruning based on Connection Sensitivity. *ICLR 2019*.
- Chen, T., Frankle, J., et al. (2020). The Lottery Ticket Hypothesis for Pre-trained BERT Networks. *NeurIPS 2020*.
