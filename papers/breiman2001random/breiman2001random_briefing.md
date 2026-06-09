# Pre-reading Briefing: Random Forests (Breiman, 2001)
# 사전 읽기 브리핑: Random Forests (Breiman, 2001)

---

## 핵심 기여 / Core Contribution

Leo Breiman은 **Random Forest**를 형식적으로 정의하고 이론적 기반을 마련한 논문을 발표했습니다. Random Forest는 여러 개의 decision tree를 bootstrap 샘플링(bagging)과 랜덤 feature 선택을 결합하여 앙상블로 구성하는 방법입니다. 핵심 통찰은 일반화 오류의 상한이 **개별 트리의 강도(strength)**와 **트리 간 상관관계(correlation)**라는 두 가지 양에 의해 결정된다는 것이며, 상관관계를 낮추면서 강도를 유지하는 것이 최적의 성능을 달성하는 열쇠입니다. 대수의 법칙(Strong Law of Large Numbers)에 의해 트리를 추가해도 overfitting이 발생하지 않음을 증명하였고, out-of-bag(OOB) 추정을 통해 별도의 test set 없이도 일반화 오류를 추정할 수 있음을 보였습니다.

Leo Breiman formally defined **Random Forests** and established their theoretical foundation. A Random Forest is an ensemble of decision trees, each grown on a bootstrap sample (bagging) with random feature selection at each node. The key insight is that the upper bound on generalization error depends on two quantities — the **strength** of individual trees and the **correlation** between them — and minimizing correlation while maintaining strength is the key to optimal performance. Using the Strong Law of Large Numbers, Breiman proved that adding more trees does not cause overfitting, and demonstrated that out-of-bag (OOB) estimation can estimate generalization error without a separate test set.

---

## 역사적 맥락 / Historical Context

이 논문은 2001년 발표 당시 ensemble methods의 황금기에 등장했습니다. 이전 논문들과의 관계는 다음과 같습니다:

This paper appeared in 2001 during the golden era of ensemble methods. Its relation to prior work:

| 연도 / Year | 업적 / Milestone | 관련성 / Relevance |
|---|---|---|
| 1984 | Breiman et al. — CART | Decision tree의 기초 / Foundation of decision trees |
| 1990 | Schapire — Boosting 이론 | 약한 학습기를 결합하는 개념 / Combining weak learners |
| 1994 | Ho — Random subspace method | 랜덤 feature 부분집합 아이디어의 선구자 / Precursor to random feature subsets |
| 1996 | Breiman — Bagging | Bootstrap aggregating — Random Forest의 직접적 전신 / Direct predecessor |
| 1996 | Freund & Schapire — AdaBoost | 당시 최고 성능의 앙상블 방법, 논문의 주요 비교 대상 / State-of-the-art ensemble, main comparison target |
| 1997 | Amit & Geman | 랜덤 feature 선택 + 트리의 결합, Breiman에게 직접 영감 / Random feature selection + trees, direct inspiration |
| **2001** | **Breiman — Random Forests** | **이 논문: 이론적 기반 + 실용적 알고리즘 통합** / **This paper: unified theory + practical algorithm** |

Breiman은 자신의 bagging (1996) 위에, Amit & Geman (1997)의 랜덤 feature 선택 아이디어를 결합하고, 이를 뒷받침하는 수학적 이론(margin, strength, correlation)을 제공했습니다. AdaBoost와의 비교를 통해 Random Forest가 동등하거나 더 나은 성능을 달성하면서도 noise에 더 강건하고 병렬화가 용이함을 보였습니다.

Breiman built upon his own bagging (1996), combined it with the random feature selection idea of Amit & Geman (1997), and provided a mathematical framework (margin, strength, correlation) to support it. Through comparison with AdaBoost, he showed Random Forests achieve comparable or better accuracy while being more robust to noise and easily parallelizable.

---

## 필요한 배경 지식 / Prerequisites

### 수학적 배경 / Mathematical Background

1. **Decision Tree (의사결정 트리)**: 데이터를 재귀적으로 분할하여 예측하는 트리 구조의 모델. CART (Classification and Regression Trees)가 대표적입니다.
   A tree-structured model that recursively partitions data for prediction. CART is the representative method.

2. **Bootstrap Sampling (부트스트랩 샘플링)**: 원본 데이터에서 복원 추출(with replacement)로 새로운 훈련 세트를 생성하는 방법. $N$개의 데이터에서 $N$번 복원 추출하면 약 63.2%의 고유 샘플이 포함됩니다.
   Creating new training sets by sampling with replacement from the original data. Drawing $N$ samples with replacement from $N$ data points includes about 63.2% unique samples.

3. **Bagging (Bootstrap Aggregating)**: 여러 bootstrap 샘플에서 각각 모델을 훈련하고 예측을 투표(분류) 또는 평균(회귀)으로 합치는 방법.
   Training models on multiple bootstrap samples and combining predictions by voting (classification) or averaging (regression).

4. **Margin (마진)**: 분류기에서 올바른 클래스에 대한 투표 비율과 가장 높은 오답 클래스 투표 비율의 차이. margin이 클수록 분류가 확실합니다.
   The difference between the vote fraction for the correct class and the highest vote fraction for any wrong class. Larger margin means more confident classification.

5. **강대수의 법칙 / Strong Law of Large Numbers**: $X_1, X_2, \ldots$가 i.i.d.이면 $\frac{1}{N}\sum_{i=1}^{N} X_i \to E[X]$ (거의 확실하게, almost surely). Random Forest에서 트리 수가 증가하면 오류가 수렴하는 이론적 근거.
   If $X_1, X_2, \ldots$ are i.i.d., then $\frac{1}{N}\sum_{i=1}^{N} X_i \to E[X]$ almost surely. This is the theoretical basis for why adding trees causes the error to converge.

6. **Chebyshev 부등식 / Chebyshev's Inequality**: $P(|X - \mu| \geq k\sigma) \leq 1/k^2$. 일반화 오류 상한을 유도하는 데 사용됩니다.
   Used to derive the upper bound on generalization error.

### 선행 논문 / Prior Papers

- **#8 Cortes & Vapnik (1995) — SVM**: margin의 개념을 이해하는 데 도움 / Helpful for understanding the margin concept
- **#7 LeCun et al. (1989)**: 학습 모델의 일반화에 대한 감각 / Intuition about model generalization
- Breiman의 bagging (1996) 논문 (reading list에는 없지만 핵심 배경) / Breiman's bagging paper (not on reading list but essential background)

---

## 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Description |
|---|---|
| **Random Forest** | 독립적인 랜덤 벡터에 의해 생성된 tree classifier들의 앙상블. 각 트리가 투표하여 다수결로 예측합니다. / An ensemble of tree classifiers, each generated by independent random vectors. Each tree casts a vote, and the majority vote is the prediction. |
| **Bagging** | Bootstrap Aggregating의 줄임말. 복원 추출된 데이터로 각 트리를 훈련합니다. / Short for Bootstrap Aggregating. Each tree is trained on data sampled with replacement. |
| **Out-of-Bag (OOB)** | 각 bootstrap 샘플에 포함되지 않은 약 1/3의 데이터. 이를 내부 test set으로 사용하여 별도 test set 없이 오류를 추정합니다. / The ~1/3 of data not included in each bootstrap sample. Used as an internal test set to estimate error without a separate test set. |
| **Strength ($s$)** | 개별 트리들이 올바르게 예측하는 능력의 기댓값. $s = E_{\mathbf{X},Y}[mr(\mathbf{X}, Y)]$. / The expected ability of individual trees to predict correctly. |
| **Correlation ($\bar{\rho}$)** | 트리 간 raw margin function의 가중 평균 상관계수. 낮을수록 앙상블 효과가 큽니다. / The weighted mean correlation of raw margin functions between trees. Lower correlation means greater ensemble effect. |
| **c/s2 ratio** | $\bar{\rho}/s^2$ — 상관관계를 강도의 제곱으로 나눈 비율. 이 값이 작을수록 Random Forest의 성능이 좋습니다. / Correlation divided by strength squared. Smaller values indicate better Random Forest performance. |
| **Forest-RI** | Random Input selection — 각 노드에서 $F$개의 랜덤 input 변수를 선택하여 분할합니다. / Random Input selection — selects $F$ random input variables at each node for splitting. |
| **Forest-RC** | Random Combination — 각 노드에서 input 변수들의 랜덤 선형 결합을 feature로 사용합니다. / Random Combination — uses random linear combinations of input variables as features at each node. |
| **Variable Importance** | OOB 데이터에서 특정 변수를 permute(셔플)했을 때 오류 증가율로 측정하는 변수 중요도. / Variable importance measured by the increase in error rate when a specific variable is permuted (shuffled) in OOB data. |

---

## 수식 미리보기 / Equations Preview

### 1. Margin Function / 마진 함수

분류기 앙상블 $h_1(\mathbf{x}), h_2(\mathbf{x}), \ldots, h_K(\mathbf{x})$에 대해 margin을 다음과 같이 정의합니다:

For an ensemble of classifiers $h_1(\mathbf{x}), h_2(\mathbf{x}), \ldots, h_K(\mathbf{x})$, the margin is defined as:

$$mg(\mathbf{X}, Y) = av_k I(h_k(\mathbf{X}) = Y) - \max_{j \neq Y} av_k I(h_k(\mathbf{X}) = j)$$

- $I(\cdot)$: indicator function (조건이 참이면 1, 거짓이면 0)
- $av_k$: $k$에 대한 평균
- **직관 / Intuition**: margin이 양수이면 올바른 클래스가 가장 많은 투표를 받음 → 정확한 예측. margin이 클수록 확신이 높습니다.
  If margin is positive, the correct class received the most votes → accurate prediction. Larger margin means higher confidence.

### 2. Generalization Error / 일반화 오류

$$PE^* = P_{\mathbf{X},Y}(mg(\mathbf{X}, Y) < 0)$$

margin이 음수일 확률 = 오분류 확률입니다.
The probability that margin is negative = misclassification probability.

### 3. Random Forest의 Margin / Random Forest Margin

트리 수가 무한히 증가하면, 대수의 법칙에 의해:

As the number of trees increases to infinity, by the Law of Large Numbers:

$$mr(\mathbf{X}, Y) = P_\Theta(h(\mathbf{X}, \Theta) = Y) - \max_{j \neq Y} P_\Theta(h(\mathbf{X}, \Theta) = j)$$

여기서 $\Theta$는 각 트리 생성에 사용되는 랜덤 벡터입니다.
Where $\Theta$ is the random vector used to generate each tree.

### 4. Strength와 일반화 오류 상한 / Strength and Error Upper Bound

**강도 / Strength**:
$$s = E_{\mathbf{X},Y} \, mr(\mathbf{X}, Y)$$

**Chebyshev 부등식에 의한 상한 / Upper bound by Chebyshev's inequality**:
$$PE^* \leq \text{var}(mr) / s^2$$

### 5. 핵심 정리: 일반화 오류의 상한 / Key Theorem: Upper Bound on Generalization Error (Theorem 2.3)

$$PE^* \leq \bar{\rho}(1 - s^2) / s^2$$

- $\bar{\rho}$: 트리 간 평균 상관계수 / mean correlation between trees
- $s$: 강도 / strength
- **직관 / Intuition**: 오류를 줄이려면 ① 상관관계 $\bar{\rho}$를 낮추고 (트리들이 서로 다르게 만듦) ② 강도 $s$를 높여야 (개별 트리가 정확해야) 합니다. Random feature 선택은 ①을 달성하는 주요 메커니즘입니다.
  To reduce error: ① lower correlation $\bar{\rho}$ (make trees different from each other) and ② increase strength $s$ (individual trees must be accurate). Random feature selection is the primary mechanism for achieving ①.

### 6. c/s2 비율 / c/s2 Ratio

$$c/s2 = \bar{\rho} / s^2$$

이 비율이 작을수록 Random Forest의 성능이 좋습니다. Feature 수 $F$를 조절하여 이 비율을 최소화합니다.
Smaller values of this ratio mean better Random Forest performance. The number of features $F$ is tuned to minimize this ratio.

### 7. 회귀에서의 일반화 오류 / Generalization Error in Regression (Theorem 11.2)

$$PE^*(\text{forest}) \leq \bar{\rho} \cdot PE^*(\text{tree})$$

Random Forest의 오류는 개별 트리 오류에 상관계수를 곱한 것 이하입니다. 상관관계가 낮으면 앙상블이 개별 트리보다 훨씬 우수합니다.
The forest error is at most the individual tree error multiplied by the correlation. Low correlation makes the ensemble much better than individual trees.

---

## 논문의 구조 / Paper Structure

| 섹션 / Section | 내용 / Content |
|---|---|
| §1 | Random Forest 정의 및 논문 개요 / Definition and paper outline |
| §2 | 이론: 수렴성, strength, correlation / Theory: convergence, strength, correlation |
| §3 | OOB 추정, Forest-RI 소개 / OOB estimation, Forest-RI introduction |
| §4 | Forest-RI 실험 결과 (20개 데이터셋) / Forest-RI experimental results (20 datasets) |
| §5 | Forest-RC (랜덤 선형 결합) / Forest-RC (random linear combinations) |
| §6 | Strength-correlation 실증 분석 / Empirical analysis of strength-correlation |
| §7 | "AdaBoost는 Random Forest다" 추측 / "AdaBoost is a Random Forest" conjecture |
| §8 | Noise에 대한 강건성 / Robustness to noise |
| §9 | 약한 입력이 많은 데이터 / Data with many weak inputs |
| §10 | Variable importance / Variable importance |
| §11–12 | 회귀 Random Forest / Regression Random Forests |
| §13 | 결론 / Conclusions |

---

## 읽기 팁 / Reading Tips

1. **§2의 수학적 유도**가 논문의 핵심입니다. Theorem 2.3의 상한 $PE^* \leq \bar{\rho}(1-s^2)/s^2$를 이해하면 나머지가 자연스럽게 따라옵니다.
   The mathematical derivation in §2 is the heart of the paper. Understanding the bound in Theorem 2.3 makes everything else follow naturally.

2. **Table 2, 3**의 실험 결과를 AdaBoost와 비교하며 읽으세요. Random Forest가 대등하거나 더 우수한 경우를 확인합니다.
   Read the experimental results in Tables 2, 3 comparing with AdaBoost. Notice where Random Forests match or surpass AdaBoost.

3. **Figure 1–3**의 strength/correlation 대 F 그래프를 주의 깊게 보세요. F(랜덤 feature 수)가 증가하면 strength와 correlation이 동시에 증가하는 trade-off가 핵심입니다.
   Pay close attention to the strength/correlation vs. F graphs in Figures 1–3. The key is the trade-off: as F (number of random features) increases, both strength and correlation increase simultaneously.

4. §7의 **"AdaBoost는 Random Forest"** 추측은 대담하고 독창적입니다. 증명되지는 않았지만 두 방법 사이의 깊은 연결을 시사합니다.
   The "AdaBoost is a Random Forest" conjecture in §7 is bold and original. Though unproven, it suggests deep connections between the two methods.

---

## Q&A

### Q1: Decision Tree, Bagging, AdaBoost 상세 설명 / Detailed explanation of Decision Tree, Bagging, AdaBoost

#### Decision Tree (의사결정 트리)

##### 기본 아이디어 / Basic Idea

Decision tree는 **"예/아니오" 질문을 반복**하여 데이터를 분류하거나 예측하는 모델입니다. 마치 스무고개 게임과 같습니다.

A decision tree is a model that classifies or predicts data by **repeating "yes/no" questions**. It's like the game of 20 questions.

예를 들어, "이메일이 스팸인가?"를 판단한다면:

For example, to determine "Is this email spam?":

```
              [제목에 "무료" 포함? / Subject contains "free"?]
               /            \
             예/Yes          아니오/No
             /                \
    [발신자가 연락처에?]     [첨부파일 있음?]
    [Sender in contacts?]   [Has attachment?]
       /       \              /       \
     예        아니오        예        아니오
      |          |           |          |
   정상/Normal 스팸/Spam   스팸/Spam  정상/Normal
```

##### 트리 구축 과정 (CART 알고리즘) / Tree Construction (CART Algorithm)

CART (Classification and Regression Trees)는 Breiman 자신이 1984년에 만든 알고리즘입니다.

CART is an algorithm Breiman himself created in 1984.

**Step 1**: 전체 데이터에서 시작합니다. / Start with all data.

**Step 2**: 모든 변수와 모든 분할점(split point)을 시도하여 **가장 순수한(pure) 두 그룹**을 만드는 분할을 찾습니다. "순수함"의 척도로 **Gini impurity**를 사용합니다.

Try all variables and all split points to find the split that creates the **purest two groups**. **Gini impurity** is used as the purity measure.

$$\text{Gini}(t) = 1 - \sum_{k=1}^{C} p_k^2$$

여기서 $p_k$는 노드 $t$에서 클래스 $k$의 비율입니다. / Where $p_k$ is the proportion of class $k$ at node $t$.

- **완벽히 순수 / Perfectly pure** (한 클래스만 / one class only): $\text{Gini} = 0$
- **완벽히 섞임 / Perfectly mixed** (2클래스 반반 / 2 classes 50-50): $\text{Gini} = 0.5$

**Step 3**: 자식 노드들에 대해 같은 과정을 재귀적으로 반복합니다. / Recursively repeat for child nodes.

**Step 4**: 종료 조건 (노드의 샘플이 너무 적거나, 이미 순수하거나)에서 멈춥니다. / Stop at termination conditions (too few samples or already pure).

##### 숫자 예시 / Numerical Example

8명의 환자 데이터로 당뇨병을 예측한다고 가정합시다:

Suppose we predict diabetes from 8 patients' data:

| 환자/Patient | 나이/Age | BMI | 운동/Exercise | 당뇨?/Diabetes? |
|---|---|---|---|---|
| 1 | 25 | 22 | 많음/High | X |
| 2 | 45 | 30 | 적음/Low | O |
| 3 | 35 | 28 | 많음/High | X |
| 4 | 50 | 32 | 적음/Low | O |
| 5 | 30 | 25 | 적음/Low | X |
| 6 | 55 | 35 | 적음/Low | O |
| 7 | 40 | 27 | 많음/High | X |
| 8 | 60 | 33 | 적음/Low | O |

트리가 "BMI > 29?"로 첫 분할을 하면: / If the tree splits on "BMI > 29?":
- **왼쪽/Left** (BMI ≤ 29): 환자 1,3,5,7 → 모두 X → Gini = 0 (완벽!/Perfect!)
- **오른쪽/Right** (BMI > 29): 환자 2,4,6,8 → 모두 O → Gini = 0 (완벽!/Perfect!)

##### Decision Tree의 문제점 / Problems with Decision Trees

- **고분산(High Variance)**: 훈련 데이터가 조금만 바뀌어도 완전히 다른 트리가 만들어질 수 있습니다. / Even small changes in training data can produce completely different trees.
- **Overfitting**: 깊은 트리는 훈련 데이터의 noise까지 학습합니다. / Deep trees learn the noise in training data.
- **불안정성**: 개별 트리 하나는 신뢰하기 어렵습니다. / A single tree is unreliable.

→ 이 문제들을 해결하기 위해 **Bagging**이 등장합니다. / **Bagging** was introduced to solve these problems.

---

#### Bagging (Bootstrap Aggregating)

##### Bootstrap Sampling 먼저 / Bootstrap Sampling First

원본 데이터에서 **복원 추출(with replacement)**로 같은 크기의 새 데이터를 만드는 것입니다.

Creating new datasets of the same size by **sampling with replacement** from the original data.

원본 데이터가 {A, B, C, D, E} 5개라면: / If original data is {A, B, C, D, E}:
- Bootstrap 샘플 1: {A, A, C, D, D} — A와 D가 두 번 뽑힘, B와 E는 빠짐 / A and D drawn twice, B and E left out
- Bootstrap 샘플 2: {B, C, C, E, E} — C와 E가 두 번 뽑힘 / C and E drawn twice
- Bootstrap 샘플 3: {A, B, D, D, E} — D가 두 번 뽑힘 / D drawn twice

수학적으로, $N$개에서 $N$번 복원 추출하면 특정 샘플이 한 번도 안 뽑힐 확률은:

Mathematically, drawing $N$ samples with replacement from $N$ items, the probability a specific sample is never drawn:

$$P(\text{안 뽑힘/not drawn}) = \left(1 - \frac{1}{N}\right)^N \approx e^{-1} \approx 0.368$$

따라서 **약 63.2%의 데이터가 각 bootstrap 샘플에 포함**되고, **약 36.8%는 빠집니다**. 이 빠진 데이터가 바로 **Out-of-Bag (OOB)** 데이터입니다.

Therefore **~63.2% of data is included** in each bootstrap sample, and **~36.8% is left out**. This left-out data is the **Out-of-Bag (OOB)** data.

##### Bagging의 핵심 아이디어 / Core Idea of Bagging

1. 원본 데이터에서 $B$개의 bootstrap 샘플을 만듭니다 / Create $B$ bootstrap samples from original data
2. 각 bootstrap 샘플로 하나의 decision tree를 (pruning 없이 끝까지) 키웁니다 / Grow a decision tree (to maximum depth, no pruning) on each sample
3. 새 데이터가 들어오면 **모든 트리에 통과시키고 다수결 투표**합니다 / For new data, pass through all trees and take **majority vote**

```
원본 데이터 / Original Data
    |
    ├── Bootstrap 1 → Tree 1 → 예측/Prediction: 클래스/Class A
    ├── Bootstrap 2 → Tree 2 → 예측/Prediction: 클래스/Class B
    ├── Bootstrap 3 → Tree 3 → 예측/Prediction: 클래스/Class A
    ├── Bootstrap 4 → Tree 4 → 예측/Prediction: 클래스/Class A
    └── Bootstrap 5 → Tree 5 → 예측/Prediction: 클래스/Class B

    다수결/Majority: A가 3표, B가 2표 → 최종 예측/Final = 클래스/Class A
```

##### 왜 Bagging이 효과적인가? / Why Does Bagging Work?

**분산 감소 원리 / Variance Reduction Principle**: $n$개의 i.i.d. 확률 변수의 평균의 분산은 $\sigma^2/n$입니다. 여러 트리의 예측을 평균(투표)하면 분산이 줄어듭니다.

The variance of the mean of $n$ i.i.d. random variables is $\sigma^2/n$. Averaging (voting) over multiple trees reduces variance.

하지만 문제가 있습니다 — bootstrap 샘플들은 **같은 원본 데이터에서 나왔기 때문에 완전히 독립이 아닙니다**. 특히 매우 강한 예측 변수가 있으면 모든 트리가 그 변수를 루트에 선택하여 **트리들이 서로 비슷해집니다(높은 상관관계)**.

But there's a problem — bootstrap samples are **not fully independent since they come from the same original data**. If there's a very strong predictor, all trees select it at the root, making **trees similar to each other (high correlation)**.

상관관계가 $\rho$인 경우: / When correlation is $\rho$:

$$\text{Var}(\text{평균/mean}) = \rho \sigma^2 + \frac{1-\rho}{n}\sigma^2$$

$n$이 아무리 커져도 $\rho\sigma^2$ 이하로는 줄지 않습니다!

No matter how large $n$ gets, variance cannot decrease below $\rho\sigma^2$!

→ 이 상관관계를 줄이기 위해 **Random Forest**가 각 노드에서 랜덤으로 feature를 선택하는 것입니다.

→ This is why **Random Forest** randomly selects features at each node — to reduce this correlation.

---

#### AdaBoost (Adaptive Boosting)

##### 핵심 아이디어 / Core Idea

Bagging은 트리들을 **독립적으로 병렬** 학습하지만, AdaBoost는 **순차적으로** 학습하면서 **이전 트리가 틀린 샘플에 더 집중**합니다.

Bagging trains trees **independently in parallel**, but AdaBoost trains **sequentially**, **focusing more on samples the previous trees got wrong**.

비유하자면: / As an analogy:
- **Bagging**: 여러 학생이 **각자 독립적으로** 같은 시험 문제를 풀고 답을 합침 / Multiple students **independently** solve the same exam and combine answers
- **AdaBoost**: 첫 학생이 풀고, 두 번째 학생은 **첫 학생이 틀린 문제를 중점적으로** 풀고, 세 번째 학생은 **앞의 두 학생이 아직 틀리는 문제에 집중**... / First student solves, second student **focuses on problems the first got wrong**, third student **focuses on remaining errors**...

##### 알고리즘 단계 / Algorithm Steps

**초기화 / Initialization**: 모든 $N$개 훈련 샘플에 동일한 가중치 $w_i = 1/N$을 부여합니다. / Assign equal weights $w_i = 1/N$ to all $N$ training samples.

**반복 / Iterate** ($t = 1, 2, \ldots, T$):

**Step 1**: 가중치 $w_i$에 따라 약한 학습기(보통 깊이 1인 "stump" 트리)를 훈련합니다. / Train a weak learner (usually a depth-1 "stump" tree) according to weights $w_i$.

**Step 2**: 가중 오류율을 계산합니다: / Compute weighted error rate:
$$\epsilon_t = \sum_{i: h_t(x_i) \neq y_i} w_i$$

**Step 3**: 학습기의 중요도를 계산합니다: / Compute learner importance:
$$\alpha_t = \frac{1}{2} \ln\left(\frac{1 - \epsilon_t}{\epsilon_t}\right)$$

- 오류가 적으면 ($\epsilon_t$ 작음) → $\alpha_t$가 큼 → 이 학습기의 발언권이 큼 / Low error → large $\alpha_t$ → more influence
- 오류가 50%에 가까우면 → $\alpha_t \approx 0$ → 랜덤 수준이므로 무시 / Error near 50% → $\alpha_t \approx 0$ → random level, ignored

**Step 4**: 가중치를 업데이트합니다: / Update weights:
$$w_i \leftarrow w_i \times \begin{cases} e^{\alpha_t} & \text{if } h_t(x_i) \neq y_i \quad \text{(틀린 것 → 가중치 증가 / wrong → weight up)} \\ e^{-\alpha_t} & \text{if } h_t(x_i) = y_i \quad \text{(맞은 것 → 가중치 감소 / correct → weight down)} \end{cases}$$

그 후 가중치를 정규화하여 합이 1이 되게 합니다. / Then normalize weights to sum to 1.

**최종 예측 / Final prediction**: 모든 학습기의 **가중 투표**: / **Weighted vote** of all learners:
$$H(x) = \text{sign}\left(\sum_{t=1}^{T} \alpha_t \, h_t(x)\right)$$

##### 숫자 예시 / Numerical Example

10개 샘플, 초기 가중치 $w_i = 0.1$: / 10 samples, initial weights $w_i = 0.1$:

| 샘플/Sample | 정답/Label | Stump 1 예측/Prediction | 맞음?/Correct? |
|---|---|---|---|
| 1 | + | + | O |
| 2 | + | + | O |
| 3 | + | - | X ← 틀림/Wrong |
| 4 | - | - | O |
| 5 | - | + | X ← 틀림/Wrong |

가중 오류 $\epsilon_1 = 0.2$ (2개 틀림)이라면: / If weighted error $\epsilon_1 = 0.2$ (2 wrong):

$$\alpha_1 = \frac{1}{2}\ln\frac{0.8}{0.2} = \frac{1}{2}\ln 4 \approx 0.693$$

틀린 샘플 3, 5의 가중치: $0.1 \times e^{0.693} = 0.2$ (2배로 증가!) / Wrong samples 3, 5 weights: doubled!

맞은 샘플들의 가중치: $0.1 \times e^{-0.693} = 0.05$ (절반으로 감소!) / Correct samples' weights: halved!

→ 다음 라운드에서 학습기는 샘플 3, 5를 더 중요하게 취급합니다. / Next round, the learner treats samples 3, 5 as more important.

##### AdaBoost의 장단점 / AdaBoost Pros and Cons

**장점 / Pros**:
- 매우 정확한 성능 (2001년 당시 최고 수준) / Very accurate performance (state-of-the-art in 2001)
- 이론적으로 잘 정립됨 (margin theory) / Well-established theory (margin theory)

**단점 / Cons** (Random Forest와의 비교에서 드러남 / revealed in comparison with Random Forest):
- **Noise에 취약**: 잘못된 라벨(noise)이 있는 샘플에 계속 가중치를 높여서 그 noise를 학습하게 됨 / **Sensitive to noise**: keeps increasing weights on mislabeled samples, learning the noise
- **순차적**: 각 단계가 이전에 의존하므로 병렬화 불가 / **Sequential**: each step depends on previous, cannot parallelize
- **느림**: 모든 변수를 사용하여 트리를 만들어야 함 / **Slow**: must use all variables to build trees

---

#### 세 방법 비교 / Comparison of Three Methods

| | Decision Tree | Bagging | AdaBoost | **Random Forest** |
|---|---|---|---|---|
| 트리 수/# Trees | 1개/1 | 다수/Many | 다수/Many | 다수/Many |
| 샘플링/Sampling | 전체/All | Bootstrap | 가중치 재조정/Reweighting | Bootstrap |
| Feature 선택/Selection | 전체/All | 전체/All | 전체/All | **랜덤 $F$개/Random $F$** |
| 학습 방식/Learning | 독립/Independent | 독립·병렬/Independent·Parallel | 순차적/Sequential | 독립·병렬/Independent·Parallel |
| 투표 방식/Voting | 없음/None | 동일 가중치/Equal weight | 성능 기반 가중치/Performance-weighted | 동일 가중치/Equal weight |
| Overfitting | 높음/High | 낮음/Low | 낮음/Low | **가장 낮음/Lowest** |
| Noise 강건성/Robustness | 낮음/Low | 중간/Medium | **낮음/Low** | **높음/High** |
| 속도/Speed | 빠름/Fast | 중간/Medium | 느림/Slow | **빠름/Fast** |

Random Forest는 Bagging에 랜덤 feature 선택을 추가한 것이고, 이것이 트리 간 상관관계를 낮추어 Bagging보다 더 나은 성능을 달성합니다. AdaBoost와 정확도는 비슷하지만 noise에 훨씬 강건하고 병렬화가 가능합니다.

Random Forest adds random feature selection to Bagging, lowering inter-tree correlation to achieve better performance than Bagging. It matches AdaBoost in accuracy but is much more robust to noise and can be parallelized.

---

### Q2: Gini Impurity 상세 설명 / Detailed explanation of Gini Impurity

#### 직관적 이해 / Intuitive Understanding

Gini impurity는 **"이 그룹에서 랜덤으로 하나를 뽑아 랜덤으로 라벨을 붙이면 틀릴 확률"**입니다.

Gini impurity is **"the probability of incorrectly labeling a randomly drawn sample if you label it randomly according to the class distribution in the group."**

- 바구니에 **빨간 공만** 있다면 → 뽑아서 "빨강"이라 하면 무조건 맞음 → **Gini = 0** (순수/pure)
- 바구니에 **빨강 50%, 파랑 50%** → 맞출 확률이 가장 낮음 → **Gini = 0.5** (최대 불순도/maximum impurity)

If a basket has **only red balls** → always correct → **Gini = 0** (pure).
If **50% red, 50% blue** → lowest chance of being correct → **Gini = 0.5** (maximum impurity).

#### 공식 / Formula

노드에 $C$개 클래스가 있고, 클래스 $k$의 비율이 $p_k$일 때:

For a node with $C$ classes, where $p_k$ is the proportion of class $k$:

$$\text{Gini}(t) = 1 - \sum_{k=1}^{C} p_k^2$$

$p_k^2$은 "랜덤으로 뽑은 샘플이 클래스 $k$이고, 랜덤으로 붙인 라벨도 $k$일 확률"입니다. 이것의 합 $\sum p_k^2$은 **맞출 확률**이고, $1$에서 빼면 **틀릴 확률** = 불순도가 됩니다.

$p_k^2$ is the probability that a randomly drawn sample is class $k$ AND a randomly assigned label is also $k$. The sum $\sum p_k^2$ is the **probability of correct labeling**; subtracting from 1 gives the **probability of error** = impurity.

#### 숫자로 이해하기 / Understanding with Numbers

**예시 1 / Example 1**: 노드에 10개 샘플 — 고양이 10, 개 0 / 10 samples — cat 10, dog 0

$$\text{Gini} = 1 - \left(\frac{10}{10}\right)^2 - \left(\frac{0}{10}\right)^2 = 1 - 1 - 0 = 0$$

→ 완벽히 순수! / Perfectly pure!

**예시 2 / Example 2**: 노드에 10개 샘플 — 고양이 5, 개 5 / 10 samples — cat 5, dog 5

$$\text{Gini} = 1 - \left(\frac{5}{10}\right)^2 - \left(\frac{5}{10}\right)^2 = 1 - 0.25 - 0.25 = 0.5$$

→ 최대 불순도. / Maximum impurity.

**예시 3 / Example 3**: 노드에 10개 샘플 — 고양이 8, 개 2 / 10 samples — cat 8, dog 2

$$\text{Gini} = 1 - \left(\frac{8}{10}\right)^2 - \left(\frac{2}{10}\right)^2 = 1 - 0.64 - 0.04 = 0.32$$

→ 꽤 순수하지만 완벽하지는 않음. / Fairly pure but not perfect.

**예시 4 / Example 4**: 3클래스 — A 7개, B 2개, C 1개 (총 10개) / 3 classes — A 7, B 2, C 1 (total 10)

$$\text{Gini} = 1 - \left(\frac{7}{10}\right)^2 - \left(\frac{2}{10}\right)^2 - \left(\frac{1}{10}\right)^2 = 1 - 0.49 - 0.04 - 0.01 = 0.46$$

#### Gini로 최적 분할 찾기 / Finding the Optimal Split with Gini

Decision tree는 **분할 후 Gini가 가장 많이 줄어드는 분할**을 선택합니다.

The decision tree selects the split that **reduces Gini the most**.

구체적인 예를 봅시다. 12명의 환자 데이터: / A concrete example with 12 patients:

| 환자/Patient | BMI | 당뇨/Diabetes |
|---|---|---|
| 1–4 | 20, 22, 24, 25 | X, X, X, X |
| 5–8 | 26, 27, 28, 29 | X, X, O, O |
| 9–12 | 30, 32, 34, 36 | O, O, O, O |

분할 전 전체 노드: 당뇨 O 6개, X 6개 / Before split: Diabetes O 6, X 6

$$\text{Gini}_{\text{before}} = 1 - (6/12)^2 - (6/12)^2 = 0.5$$

**분할 후보 A / Candidate A**: BMI > 29?

- 왼쪽/Left (BMI ≤ 29): 8명 중 X 6개, O 2개 / 8 samples, X 6, O 2 → $\text{Gini}_L = 1 - (6/8)^2 - (2/8)^2 = 0.375$
- 오른쪽/Right (BMI > 29): 4명 중 O 4개 / 4 samples, O 4 → $\text{Gini}_R = 0$
- 가중 평균/Weighted average: $\text{Gini}_A = \frac{8}{12} \times 0.375 + \frac{4}{12} \times 0 = 0.25$

**분할 후보 B / Candidate B**: BMI > 25?

- 왼쪽/Left (BMI ≤ 25): 4명 중 X 4개 / 4 samples, X 4 → $\text{Gini}_L = 0$
- 오른쪽/Right (BMI > 25): 8명 중 X 2개, O 6개 / 8 samples, X 2, O 6 → $\text{Gini}_R = 0.375$
- 가중 평균/Weighted average: $\text{Gini}_B = \frac{4}{12} \times 0 + \frac{8}{12} \times 0.375 = 0.25$

**분할 후보 C / Candidate C**: BMI > 27?

- 왼쪽/Left (BMI ≤ 27): 6명 중 X 6개 / 6 samples, X 6 → $\text{Gini}_L = 0$
- 오른쪽/Right (BMI > 27): 6명 중 O 6개 / 6 samples, O 6 → $\text{Gini}_R = 0$
- 가중 평균/Weighted average: $\text{Gini}_C = 0$

**Gini 감소량 (Information Gain) / Gini Reduction**:

| 분할/Split | $\text{Gini}_{\text{after}}$ | $\Delta\text{Gini} = 0.5 - \text{Gini}_{\text{after}}$ |
|---|---|---|
| A: BMI > 29 | 0.25 | 0.25 |
| B: BMI > 25 | 0.25 | 0.25 |
| **C: BMI > 27** | **0** | **0.5** ← 최대/Maximum |

→ **BMI > 27이 최적 분할**입니다. Gini가 0.5에서 0으로 완전히 감소했습니다.

→ **BMI > 27 is the optimal split**. Gini decreased completely from 0.5 to 0.

#### Gini vs Entropy

Decision tree에서 불순도를 측정하는 또 다른 방법으로 **Entropy**가 있습니다:

Another impurity measure in decision trees is **Entropy**:

$$\text{Entropy}(t) = -\sum_{k=1}^{C} p_k \log_2 p_k$$

| 비율/Ratio (2클래스/2-class) | Gini | Entropy |
|---|---|---|
| 100%:0% | 0 | 0 |
| 90%:10% | 0.18 | 0.47 |
| 70%:30% | 0.42 | 0.88 |
| 50%:50% | 0.50 | 1.00 |

둘의 형태는 매우 비슷하고, 실제 성능 차이는 거의 없습니다. Breiman의 CART는 **Gini**를 사용하고, ID3/C4.5 (Quinlan)는 **Entropy**를 사용합니다. Random Forest 논문에서는 CART 방식의 Gini를 사용합니다.

Both have very similar shapes and produce nearly identical results in practice. Breiman's CART uses **Gini**, while ID3/C4.5 (Quinlan) uses **Entropy**. The Random Forest paper uses CART-style Gini.

#### Random Forest에서의 역할 / Role in Random Forest

Random Forest에서 Gini의 역할은 동일하지만, 핵심 차이가 있습니다:

Gini's role is the same in Random Forest, but with a key difference:

- **일반 Decision Tree / Standard Decision Tree**: 모든 $M$개 변수 중 Gini를 최대로 줄이는 변수와 분할점을 선택 / Select the variable and split point that maximally reduce Gini from all $M$ variables
- **Random Forest (Forest-RI)**: **랜덤으로 $F$개 변수만 후보로 선정**한 후, 그 중에서 Gini를 최대로 줄이는 것을 선택 / **Randomly select only $F$ candidate variables**, then choose the one that maximally reduces Gini among them

이 $F$개 제한이 트리 간 상관관계를 낮추는 핵심 메커니즘입니다. 논문에서는 $F = 1$ (변수 하나만!) 또는 $F = \lfloor\log_2 M + 1\rfloor$ 을 실험합니다.

This restriction to $F$ variables is the key mechanism for reducing inter-tree correlation. The paper experiments with $F = 1$ (just one variable!) or $F = \lfloor\log_2 M + 1\rfloor$.
