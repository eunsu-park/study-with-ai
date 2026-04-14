---
title: "Random Forests"
authors: Leo Breiman
year: 2001
journal: "Machine Learning, Vol. 45, pp. 5–32"
topic: Artificial Intelligence / Ensemble Methods
tags: [random forest, bagging, bootstrap, ensemble, decision tree, out-of-bag, variable importance, correlation, strength, classification, regression]
status: completed
date_started: 2026-04-11
date_completed: 2026-04-11
---

# Random Forests
**Leo Breiman (2001)**

---

## 핵심 기여 / Core Contribution

Leo Breiman은 **Random Forest**를 형식적으로 정의하고 이론적 기반을 확립한 논문을 발표했습니다. Random Forest는 여러 개의 decision tree를 **bagging (bootstrap aggregating)**과 **랜덤 feature 선택**을 결합하여 앙상블로 구성하는 방법입니다. 논문의 핵심 기여는 세 가지입니다. 첫째, 대수의 법칙(Strong Law of Large Numbers)을 통해 트리를 아무리 많이 추가해도 **overfitting이 발생하지 않으며** 일반화 오류가 특정 한계값으로 수렴함을 증명했습니다 (Theorem 1.2). 둘째, 일반화 오류의 상한이 개별 트리의 **강도(strength, $s$)**와 트리 간 **상관관계(correlation, $\bar{\rho}$)**라는 두 양에 의해 $PE^* \leq \bar{\rho}(1-s^2)/s^2$로 결정됨을 보였으며 (Theorem 2.3), 이를 통해 "상관관계를 낮추면서 강도를 유지하라"는 명확한 설계 지침을 제공했습니다. 셋째, **out-of-bag (OOB) 추정**을 통해 별도의 test set 없이도 일반화 오류, strength, correlation을 내부적으로 추정할 수 있음을 보였습니다. 20개 데이터셋에서의 실험을 통해 Random Forest가 당시 최고 성능의 AdaBoost와 대등하거나 더 나은 정확도를 달성하면서도 noise에 훨씬 강건하고, 병렬화가 가능하며, 변수 중요도(variable importance)를 제공한다는 실용적 장점을 입증했습니다.

Leo Breiman formally defined **Random Forests** and established their theoretical foundation. A Random Forest constructs an ensemble of decision trees by combining **bagging (bootstrap aggregating)** with **random feature selection**. The paper makes three key contributions. First, using the Strong Law of Large Numbers, it proves that adding more trees **never causes overfitting** — the generalization error converges to a limiting value (Theorem 1.2). Second, it shows that the upper bound on generalization error is determined by two quantities — the **strength ($s$)** of individual trees and the **correlation ($\bar{\rho}$)** between them — giving $PE^* \leq \bar{\rho}(1-s^2)/s^2$ (Theorem 2.3), which provides a clear design guideline: "minimize correlation while maintaining strength." Third, it demonstrates that **out-of-bag (OOB) estimation** can internally estimate generalization error, strength, and correlation without a separate test set. Experiments on 20 datasets show Random Forests achieve accuracy comparable to or better than the state-of-the-art AdaBoost, while being far more robust to noise, easily parallelizable, and capable of providing variable importance measures.

---

## 읽기 노트 / Reading Notes

### §1: Random Forests — 정의와 개요 / Definition and Overview

#### Random Forest의 형식적 정의 / Formal Definition

논문은 Random Forest를 다음과 같이 정의합니다 (Definition 1.1):

The paper defines Random Forest as follows (Definition 1.1):

> **Random Forest**는 tree-structured classifier $\{h(\mathbf{x}, \Theta_k), k = 1, \ldots\}$의 모음(collection)으로, $\{\Theta_k\}$는 독립 동일 분포(i.i.d.)의 랜덤 벡터이고, 각 트리는 입력 $\mathbf{x}$에 대해 가장 많은 표를 받은 클래스에 단위 투표(unit vote)를 합니다.
>
> A **Random Forest** is a collection of tree-structured classifiers $\{h(\mathbf{x}, \Theta_k), k = 1, \ldots\}$ where $\{\Theta_k\}$ are independent identically distributed random vectors, and each tree casts a unit vote for the most popular class at input $\mathbf{x}$.

여기서 $\Theta_k$는 $k$번째 트리의 생성을 지배하는 랜덤 벡터입니다. Bagging에서는 $\Theta$가 bootstrap 샘플에 해당하고 (각 데이터 포인트가 포함될 횟수를 결정하는 $N$개의 카운트), random split selection에서는 분할에 사용할 feature의 인덱스 등이 됩니다.

Here $\Theta_k$ is the random vector governing the generation of the $k$-th tree. In bagging, $\Theta$ corresponds to the bootstrap sample (counts of how many times each data point is included), and in random split selection, it includes the indices of features to use for splitting.

이 정의가 중요한 이유는 $\Theta_k$들이 **i.i.d.**라는 조건 덕분에 대수의 법칙을 적용할 수 있고, 이것이 overfitting 방지를 증명하는 핵심이 되기 때문입니다.

This definition is important because the **i.i.d.** condition on $\Theta_k$ enables the application of the Law of Large Numbers, which is the key to proving overfitting prevention.

#### 선행 연구와의 관계 / Relation to Prior Work

Breiman은 자신의 Random Forest가 여러 선행 연구의 공통 요소를 통합한 것임을 명시합니다:

Breiman acknowledges that his Random Forest unifies common elements from several prior works:

- **Bagging** (Breiman, 1996): 복원 추출로 여러 트리를 만들어 투표. 하지만 모든 변수를 사용하므로 트리 간 상관관계가 높음 / Bootstrap sampling to create multiple trees for voting. But uses all variables, leading to high inter-tree correlation
- **Random split selection** (Dietterich, 1998): 각 노드에서 $K$개 최적 분할 중 랜덤으로 선택 / Randomly selects among $K$ best splits at each node
- **Random subspace method** (Ho, 1998): 각 트리에 feature의 부분집합만 사용 / Uses only a subset of features for each tree
- **Amit & Geman** (1997): 대량의 기하학적 feature에서 랜덤 선택하여 분할. **Breiman에게 가장 큰 영감을 준 논문** / Random selection from a large set of geometric features for splitting. **The most influential paper for Breiman**

이 모든 방법의 공통점은: $k$번째 트리에 대해 랜덤 벡터 $\Theta_k$가 생성되고, 이전 $\Theta_1, \ldots, \Theta_{k-1}$과 독립이지만 같은 분포를 따르며, 훈련 세트와 $\Theta_k$를 사용하여 트리를 키운다는 것입니다.

The common element across all these methods: for the $k$-th tree, a random vector $\Theta_k$ is generated, independent of previous $\Theta_1, \ldots, \Theta_{k-1}$ but with the same distribution, and a tree is grown using the training set and $\Theta_k$.

---

### §2: Characterizing the Accuracy — 이론적 핵심 / Theoretical Core

이 섹션은 논문의 수학적 심장부로, Random Forest가 왜 작동하는지에 대한 이론적 근거를 세 단계로 제시합니다.

This section is the mathematical heart of the paper, presenting the theoretical basis for why Random Forests work in three stages.

#### 2.1: Random Forest는 수렴한다 / Random Forests Converge

분류기 앙상블 $h_1(\mathbf{x}), h_2(\mathbf{x}), \ldots, h_K(\mathbf{x})$에 대해, **margin function**을 다음과 같이 정의합니다:

For an ensemble of classifiers $h_1(\mathbf{x}), h_2(\mathbf{x}), \ldots, h_K(\mathbf{x})$, the **margin function** is defined as:

$$mg(\mathbf{X}, Y) = av_k I(h_k(\mathbf{X}) = Y) - \max_{j \neq Y} av_k I(h_k(\mathbf{X}) = j)$$

여기서 $I(\cdot)$은 indicator function, $av_k$는 $k$에 대한 평균입니다. margin이 양수이면 올바른 클래스가 가장 많은 투표를 받았다는 뜻이고, 그 크기가 클수록 분류의 확신이 높습니다. 일반화 오류는 margin이 음수일 확률입니다:

Where $I(\cdot)$ is the indicator function and $av_k$ is the average over $k$. Positive margin means the correct class received the most votes, and larger margin means higher confidence. The generalization error is the probability of negative margin:

$$PE^* = P_{\mathbf{X},Y}(mg(\mathbf{X}, Y) < 0)$$

**Theorem 1.2 (수렴 정리 / Convergence Theorem)**: 트리의 수가 증가하면, 거의 확실하게(almost surely) $PE^*$가 다음으로 수렴합니다:

**Theorem 1.2 (Convergence)**: As the number of trees increases, $PE^*$ converges almost surely to:

$$P_{\mathbf{X},Y}\left(P_\Theta(h(\mathbf{X}, \Theta) = Y) - \max_{j \neq Y} P_\Theta(h(\mathbf{X}, \Theta) = j) < 0\right) \tag{1}$$

**증명의 핵심 아이디어 / Key proof idea** (Appendix I): 고정된 $\Theta$와 훈련 세트에 대해, $h(\Theta, \mathbf{x}) = j$인 $\mathbf{x}$의 집합은 hyper-rectangle들의 합집합입니다. 이런 합집합의 종류는 유한 개($K$개)이므로, $\varphi(\Theta) = k$로 이 합집합을 인덱싱할 수 있습니다. 대수의 법칙에 의해 $\frac{1}{N}\sum I(\varphi(\Theta_n) = k) \to P_\Theta(\varphi(\Theta) = k)$이 거의 확실하게 성립하고, 유한 개의 $k$에 대한 합집합을 취하면 수렴이 실패하는 집합의 확률이 0이 됩니다.

**Key proof idea** (Appendix I): For fixed $\Theta$ and training set, the set of $\mathbf{x}$ where $h(\Theta, \mathbf{x}) = j$ is a union of hyper-rectangles. There are finitely many ($K$) such unions, indexable by $\varphi(\Theta) = k$. By the Law of Large Numbers, $\frac{1}{N}\sum I(\varphi(\Theta_n) = k) \to P_\Theta(\varphi(\Theta) = k)$ almost surely, and taking unions over finitely many $k$ gives a zero-probability set where convergence fails.

**이 정리의 의미 / Significance of this theorem**: Random Forest에 트리를 계속 추가해도 일반화 오류가 한없이 커지지 않고, 특정 한계값으로 **수렴**합니다. 즉, **overfitting이 발생하지 않습니다**. 이것은 개별 decision tree (트리가 깊어질수록 overfitting)와 근본적으로 다른 성질입니다. 실용적으로는 "트리를 많이 넣을수록 좋거나 최소한 나빠지지 않는다"는 것을 의미합니다.

**Significance**: Adding more trees never degrades performance — the generalization error **converges** to a limit. This means **no overfitting occurs**. This is fundamentally different from individual decision trees (deeper = more overfitting). Practically, it means "more trees is always better, or at least never worse."

#### 2.2: Strength과 Correlation — 성능의 두 축 / The Two Axes of Performance

Breiman은 일반화 오류의 상한을 두 가지 양으로 표현합니다. 먼저 Random Forest의 margin function을 정의합니다 (Definition 2.1):

Breiman expresses the upper bound on generalization error in terms of two quantities. First, define the Random Forest margin function (Definition 2.1):

$$mr(\mathbf{X}, Y) = P_\Theta(h(\mathbf{X}, \Theta) = Y) - \max_{j \neq Y} P_\Theta(h(\mathbf{X}, \Theta) = j) \tag{2}$$

이것은 Theorem 1.2의 수렴 한계에서의 margin입니다 — 무한히 많은 트리의 투표 비율 차이입니다. 그리고 **강도(strength)**를 이 margin의 기댓값으로 정의합니다:

This is the margin at the convergence limit of Theorem 1.2 — the vote proportion difference with infinitely many trees. The **strength** is defined as the expectation of this margin:

$$s = E_{\mathbf{X},Y} \, mr(\mathbf{X}, Y) \tag{3}$$

$s > 0$이면 평균적으로 올바른 클래스에 더 많은 투표가 가고, $s$가 클수록 개별 트리들이 더 정확합니다.

If $s > 0$, on average the correct class receives more votes; larger $s$ means more accurate individual trees.

Chebyshev 부등식 $P(X \leq 0) \leq \text{var}(X)/E[X]^2$ (단, $E[X] > 0$)을 적용하면:

Applying Chebyshev's inequality $P(X \leq 0) \leq \text{var}(X)/E[X]^2$ (when $E[X] > 0$):

$$PE^* \leq \text{var}(mr) / s^2 \tag{4}$$

이제 $\text{var}(mr)$을 상관관계로 표현하는 것이 핵심입니다. **Raw margin function** (Definition 2.2)을 도입합니다:

The key is expressing $\text{var}(mr)$ in terms of correlation. The **raw margin function** (Definition 2.2) is introduced:

$$rmg(\Theta, \mathbf{X}, Y) = I(h(\mathbf{X}, \Theta) = Y) - I(h(\mathbf{X}, \Theta) = \hat{j}(\mathbf{X}, Y))$$

여기서 $\hat{j}(\mathbf{X}, Y) = \arg\max_{j \neq Y} P_\Theta(h(\mathbf{X}, \Theta) = j)$는 "가장 강한 오답 클래스"입니다. $rmg$는 개별 트리 $\Theta$의 raw한 기여분으로, $mr(\mathbf{X}, Y) = E_\Theta[rmg(\Theta, \mathbf{X}, Y)]$입니다.

Where $\hat{j}(\mathbf{X}, Y) = \arg\max_{j \neq Y} P_\Theta(h(\mathbf{X}, \Theta) = j)$ is the "strongest wrong class." $rmg$ is the raw contribution of individual tree $\Theta$, and $mr(\mathbf{X}, Y) = E_\Theta[rmg(\Theta, \mathbf{X}, Y)]$.

독립인 $\Theta, \Theta'$에 대해 $[E_\Theta f(\Theta)]^2 = E_{\Theta,\Theta'} f(\Theta)f(\Theta')$라는 항등식을 사용하면:

Using the identity $[E_\Theta f(\Theta)]^2 = E_{\Theta,\Theta'} f(\Theta)f(\Theta')$ for independent $\Theta, \Theta'$:

$$mr(\mathbf{X}, Y)^2 = E_{\Theta,\Theta'} rmg(\Theta, \mathbf{X}, Y) \cdot rmg(\Theta', \mathbf{X}, Y) \tag{5}$$

그리고 $\text{var}(mr)$을 전개하면:

Expanding $\text{var}(mr)$:

$$\text{var}(mr) = E_{\Theta,\Theta'}(\rho(\Theta, \Theta') \cdot sd(\Theta) \cdot sd(\Theta')) \tag{6}$$

여기서 $\rho(\Theta, \Theta')$는 $rmg(\Theta, \mathbf{X}, Y)$과 $rmg(\Theta', \mathbf{X}, Y)$ 사이의 상관관계이고 ($\Theta, \Theta'$ 고정, $\mathbf{X}, Y$에 대한 평균), $sd(\Theta) = \sqrt{\text{var}_{\mathbf{X},Y}(rmg(\Theta, \mathbf{X}, Y))}$입니다.

Where $\rho(\Theta, \Theta')$ is the correlation between $rmg(\Theta, \mathbf{X}, Y)$ and $rmg(\Theta', \mathbf{X}, Y)$ (with $\Theta, \Theta'$ fixed, averaged over $\mathbf{X}, Y$), and $sd(\Theta) = \sqrt{\text{var}_{\mathbf{X},Y}(rmg(\Theta, \mathbf{X}, Y))}$.

**가중 평균 상관관계 / Weighted mean correlation** $\bar{\rho}$를 다음과 같이 정의하면:

$$\bar{\rho} = E_{\Theta,\Theta'}(\rho(\Theta, \Theta') \cdot sd(\Theta) \cdot sd(\Theta')) / E_{\Theta,\Theta'}(sd(\Theta) \cdot sd(\Theta'))$$

Jensen 부등식을 사용하여 $\text{var}(mr) \leq \bar{\rho} \cdot E_\Theta \text{var}(\Theta)$를 얻고 (Eq. 7), $E_\Theta \text{var}(\Theta) \leq 1 - s^2$ (Eq. 8)을 결합하면:

Using Jensen's inequality to get $\text{var}(mr) \leq \bar{\rho} \cdot E_\Theta \text{var}(\Theta)$ (Eq. 7), and combining with $E_\Theta \text{var}(\Theta) \leq 1 - s^2$ (Eq. 8):

#### Theorem 2.3: 핵심 정리 / The Key Theorem

$$\boxed{PE^* \leq \frac{\bar{\rho}(1 - s^2)}{s^2}}$$

**이 부등식의 의미를 직관적으로 이해하기 / Intuitive understanding**:

- **$\bar{\rho}$ (상관관계)가 작을수록**: 트리들이 서로 다른 실수를 하므로, 투표로 합치면 개별 실수가 상쇄됩니다. 극단적으로 $\bar{\rho} = 0$이면 $PE^* = 0$ (완벽한 분류). / When **$\bar{\rho}$ (correlation) is small**: trees make different mistakes, so voting cancels individual errors. In the extreme $\bar{\rho} = 0$ gives $PE^* = 0$ (perfect classification).
- **$s$ (강도)가 클수록**: 분모 $s^2$이 커져서 상한이 줄어들고, 분자의 $(1 - s^2)$도 줄어듭니다. 개별 트리가 정확할수록 앙상블도 정확합니다. / When **$s$ (strength) is large**: denominator $s^2$ increases, reducing the bound, and numerator $(1 - s^2)$ also decreases. More accurate individual trees mean more accurate ensemble.
- **Trade-off**: feature 수 $F$를 늘리면 개별 트리가 더 강해지지만(strength ↑), 트리들이 비슷해지기도 합니다(correlation ↑). 최적의 $F$는 이 둘의 균형점에 있습니다. / **Trade-off**: increasing $F$ makes individual trees stronger (strength ↑), but also makes trees more similar (correlation ↑). Optimal $F$ balances the two.

#### c/s2 비율 — 실용적 지표 / The c/s2 Ratio — A Practical Metric

Breiman은 이 trade-off를 하나의 비율로 요약합니다 (Definition 2.4):

Breiman summarizes this trade-off in a single ratio (Definition 2.4):

$$c/s2 = \bar{\rho} / s^2$$

이 비율이 작을수록 Random Forest의 성능이 좋습니다. OOB 추정으로 이 비율을 모니터링하면서 $F$를 조정할 수 있습니다.

Smaller values mean better performance. This ratio can be monitored via OOB estimation while tuning $F$.

---

### §3: Using Random Features — OOB 추정과 알고리즘 / OOB Estimation and Algorithm

#### Out-of-Bag (OOB) 추정의 원리 / Principle of OOB Estimation

Breiman의 실험에서는 bagging과 random feature selection을 함께 사용합니다. 각 트리는 원본 훈련 세트에서 복원 추출(bootstrap)된 데이터로 키워지며, **pruning하지 않습니다**. Bagging을 사용하는 두 가지 이유가 있습니다:

In Breiman's experiments, bagging and random feature selection are used together. Each tree is grown on a bootstrap sample from the original training set, and **trees are not pruned**. There are two reasons for using bagging:

1. **정확도 향상**: random feature를 사용할 때 bagging이 정확도를 높입니다 / **Accuracy improvement**: bagging enhances accuracy when random features are used
2. **내부 추정**: bagging을 통해 일반화 오류($PE^*$), strength, correlation의 내부 추정이 가능합니다 / **Internal estimation**: bagging enables internal estimates of generalization error ($PE^*$), strength, and correlation

OOB 추정의 과정: / OOB estimation process:

1. 특정 훈련 세트 $T$에서 bootstrap 훈련 세트 $T_k$를 만들어 분류기 $h(\mathbf{x}, T_k)$를 구축합니다 / Build classifier $h(\mathbf{x}, T_k)$ from bootstrap training set $T_k$ drawn from $T$
2. 각 훈련 샘플 $(y, \mathbf{x})$에 대해, **그 샘플을 포함하지 않는 bootstrap 세트로 만든 분류기들만** 사용하여 투표합니다 / For each training sample $(y, \mathbf{x})$, vote using **only classifiers built from bootstrap sets that don't contain that sample**
3. 이것이 OOB 분류기이고, 그 오류율이 OOB 오류 추정입니다 / This is the OOB classifier, and its error rate is the OOB error estimate

각 bootstrap 샘플에서 약 1/3의 데이터가 빠지므로, OOB 추정은 전체 앙상블의 약 1/3 크기의 분류기들의 투표에 기반합니다. Breiman (1996b)의 연구에 따르면 OOB 추정은 **동일 크기의 test set을 사용한 것만큼 정확**하여, 별도의 test set이 불필요합니다.

About 1/3 of data is left out in each bootstrap sample, so OOB estimates are based on votes from about 1/3 as many classifiers as the full ensemble. Breiman (1996b) showed OOB estimates are **as accurate as using a test set of the same size**, eliminating the need for a separate test set.

중요한 차이: cross-validation에서는 bias가 존재하지만 그 크기를 알 수 없는 반면, OOB 추정은 **unbiased**입니다 (충분히 많은 트리를 사용할 경우). 단, OOB 추정은 현재 앙상블보다 작은 앙상블에 기반하므로 현재 오류율을 약간 과대추정하는 경향이 있어, 수렴하는 시점까지 충분히 많은 트리를 만들어야 합니다.

Key difference: cross-validation has bias of unknown magnitude, while OOB estimates are **unbiased** (with enough trees). However, OOB estimates are based on smaller ensembles than the current one, so they slightly overestimate the current error rate — enough trees must be grown to reach convergence.

---

### §4: Forest-RI — 랜덤 입력 선택 / Random Input Selection

#### Forest-RI 알고리즘 / Forest-RI Algorithm

가장 단순한 형태의 Random Forest는 **Forest-RI (Random Input)**입니다:

The simplest form of Random Forest is **Forest-RI (Random Input)**:

1. Bootstrap 샘플을 추출합니다 / Draw a bootstrap sample
2. 각 노드에서 $F$개의 입력 변수를 **랜덤으로** 선택합니다 / At each node, **randomly** select $F$ input variables
3. 그 $F$개 중에서 CART 방식으로 최적 분할을 찾습니다 / Find the best split among those $F$ using CART methodology
4. 트리를 최대 크기까지 키우고 **pruning하지 않습니다** / Grow the tree to maximum size, **do not prune**
5. 이 과정을 반복하여 많은 트리를 만들고 투표합니다 / Repeat to create many trees and vote

$F$의 값으로 두 가지를 실험합니다: / Two values of $F$ are tested:
- $F = 1$: 랜덤으로 **딱 하나**의 변수만 선택하여 분할 / Randomly select **just one** variable for splitting
- $F = \lfloor\log_2 M + 1\rfloor$: $M$이 입력 변수의 수 / Where $M$ is the number of input variables

놀랍게도 $F = 1$이라는 극단적으로 랜덤한 선택도 꽤 좋은 성능을 보입니다. 두 값 사이의 절대적 오류 차이는 대부분의 데이터셋에서 1% 미만이었습니다.

Surprisingly, even the extremely random choice of $F = 1$ performs quite well. The absolute error difference between the two values was less than 1% on most datasets.

#### 실험 결과 (Table 2) / Experimental Results

13개의 작은 UCI 데이터셋과 3개의 큰 데이터셋, 4개의 합성 데이터셋에서 실험합니다. 작은 데이터셋에서는 10%를 test set으로 분리하고 100번 반복하여 평균을 냅니다. Random Forest는 100개의 트리, AdaBoost는 50개의 트리를 사용합니다.

Experiments on 13 small UCI datasets, 3 large datasets, and 4 synthetic datasets. For small datasets, 10% is held out as test set and repeated 100 times. Random Forest uses 100 trees, AdaBoost uses 50 trees.

핵심 결과: / Key results:
- Random Forest의 오류율은 **AdaBoost와 비교할 만하며**, 때로는 더 나음 / Random Forest error rates are **comparable to AdaBoost**, sometimes better
- 단일 트리 대비 대폭적인 성능 향상 (예: Glass 데이터 — 단일 트리 36.9% → RF 20.6%) / Dramatic improvement over single trees (e.g., Glass — single tree 36.9% → RF 20.6%)
- $F = 1$과 $F = \lfloor\log_2 M + 1\rfloor$ 사이의 차이가 작으며, **OOB 추정으로 더 나은 쪽을 자동 선택** 가능 / Difference between $F = 1$ and $F = \lfloor\log_2 M + 1\rfloor$ is small, and **OOB estimation can automatically select the better one**

속도 비교: Forest-RI의 계산 시간은 일반 트리 대비 $F \cdot \log_2(N) / M$ 비율입니다. Zip-code 데이터에서 $F = 1$일 때 이 비율은 0.025로, **Forest-RI가 40배 빠릅니다**. 100개 트리를 만드는 데 4분 vs. AdaBoost의 50개 트리에 약 3시간.

Speed comparison: Forest-RI computation time ratio to regular trees is $F \cdot \log_2(N) / M$. On zip-code data with $F = 1$, this ratio is 0.025, meaning **Forest-RI is 40× faster**. 100 trees in 4 minutes vs. ~3 hours for 50 AdaBoost trees.

---

### §5: Forest-RC — 랜덤 선형 결합 / Random Linear Combinations

입력 변수가 적은 경우($M$이 작을 때), $F$를 $M$의 상당 부분으로 설정하면 strength는 올라가지만 correlation도 올라갑니다. 이때의 대안이 **Forest-RC (Random Combination)**입니다:

When there are few input variables (small $M$), setting $F$ to a large fraction of $M$ increases strength but also correlation. The alternative is **Forest-RC (Random Combination)**:

1. 각 노드에서 $L$개의 입력 변수를 랜덤으로 선택합니다 / At each node, randomly select $L$ input variables
2. $[-1, 1]$에서 균일 분포의 랜덤 계수를 생성합니다 / Generate random coefficients from uniform distribution on $[-1, 1]$
3. $F$개의 랜덤 선형 결합 $\sum a_l x_l$을 만들어 이들 중 최적 분할을 찾습니다 / Create $F$ random linear combinations $\sum a_l x_l$ and find the best split among them

$L = 3$, $F = 2$ 또는 $8$을 사용합니다. $L = 3$인 이유는 $O(M^3)$개의 서로 다른 조합이 가능하여 correlation을 크게 높이지 않기 때문입니다.

Uses $L = 3$, $F = 2$ or $8$. $L = 3$ because $O(M^3)$ different combinations are possible, avoiding large increases in correlation.

Forest-RC는 전반적으로 Forest-RI보다 AdaBoost에 유리하게 비교됩니다. 특히 합성 데이터에서 우수한 성능을 보입니다.

Forest-RC generally compares more favorably to AdaBoost than Forest-RI. It performs especially well on synthetic data.

#### 범주형 변수 처리 / Handling Categorical Variables

$I$개 값을 가진 범주형 변수는 $I - 1$개의 dummy 변수로 코딩됩니다. 이 경우 낮은 $F$는 낮은 correlation을 주지만 strength도 낮아지므로, $F$를 $\text{int}(\log_2 M + 1)$의 2-3배로 늘려야 합니다.

A categorical variable with $I$ values is coded as $I - 1$ dummy variables. Low $F$ gives low correlation but also low strength, so $F$ should be increased to 2-3 times $\text{int}(\log_2 M + 1)$.

---

### §6: Empirical Results on Strength and Correlation — 이론의 실증 / Empirical Validation of Theory

이 섹션은 §2의 이론을 OOB 추정을 사용하여 실증적으로 검증합니다. Breiman이 특히 알고 싶었던 것은: **왜 일반화 오류가 $F$에 둔감한가?**

This section empirically validates the theory of §2 using OOB estimates. What Breiman particularly wanted to understand: **why is generalization error insensitive to $F$?**

#### Sonar 데이터 실험 (Figure 1) / Sonar Data Experiment

60개 입력, 208개 샘플. $F$를 1에서 50까지 변화시키면서 strength와 correlation을 측정합니다.

60 inputs, 208 samples. Varies $F$ from 1 to 50 while measuring strength and correlation.

- **상단 그래프**: $F \approx 4$ 이후 strength가 거의 일정합니다. 하지만 correlation은 계속 증가합니다 / **Top graph**: Strength plateaus around $F \approx 4$. But correlation keeps increasing
- **하단 그래프**: test set 오류와 OOB 오류 모두 $F = 1$에서 약간 높지만, $F = 4$에서 8 사이에서 최소가 되고 이후 서서히 증가합니다 / **Bottom graph**: Both test set error and OOB error are slightly higher at $F = 1$, minimize around $F = 4$-8, then gradually increase

**핵심 통찰 / Key insight**: strength가 일정한 상태에서 correlation만 증가하므로, $c/s2 = \bar{\rho}/s^2$가 증가하고, 이에 따라 오류도 서서히 증가합니다. 하지만 그 증가 폭이 작아서 결과가 $F$에 **둔감(insensitive)**합니다.

Strength is constant while only correlation increases, so $c/s2 = \bar{\rho}/s^2$ increases and error gradually rises. But the increase is small, making results **insensitive** to $F$.

#### Breast Cancer 데이터 (Figure 2) / Breast Cancer Data

9개 입력. 랜덤 3개 입력의 선형 결합을 feature로 사용합니다. strength가 거의 완벽(~0.9)하고 correlation이 매우 낮아(~0.2), 오류가 3% 정도로 일정합니다. **놀라운 점: strength가 일정하다는 것 — feature 수를 늘려도 개별 트리의 정확도가 향상되지 않습니다.**

9 inputs. Uses linear combinations of 3 random inputs as features. Strength is nearly perfect (~0.9) and correlation is very low (~0.2), keeping error constant at ~3%. **The surprise: strength is constant — increasing features doesn't improve individual tree accuracy.**

#### Satellite 데이터 (Figure 3) / Satellite Data

큰 데이터셋에서는 다른 양상입니다. correlation과 strength 모두 서서히 증가하며, 오류율은 약간 감소합니다. $F$를 더 키우면(100까지) 성능이 계속 향상되어, 이 세 데이터셋에서 역대 최저 오류를 달성합니다.

Larger datasets show different behavior. Both correlation and strength slowly increase, and error rates slightly decrease. Increasing $F$ further (up to 100) continues to improve performance, achieving the lowest errors ever on these three datasets.

---

### §7: Conjecture — AdaBoost는 Random Forest다 / AdaBoost is a Random Forest

이 섹션은 논문에서 가장 대담한 추측입니다. Breiman은 AdaBoost를 Random Forest의 프레임워크로 재해석합니다.

This section contains the paper's boldest conjecture. Breiman reinterprets AdaBoost within the Random Forest framework.

**논증 구조 / Argument structure**:

1. 분류기를 훈련 세트와 가중치의 함수 $h(\mathbf{x}, \mathbf{w})$로 일반화합니다 / Generalize classifiers as functions of training set and weights $h(\mathbf{x}, \mathbf{w})$
2. $K$개의 가중치 집합 $\mathbf{w}(1), \ldots, \mathbf{w}(K)$에 확률 $p(1), \ldots, p(K)$를 부여하면 Random Forest가 됩니다 / Assign probabilities $p(1), \ldots, p(K)$ to $K$ weight sets $\mathbf{w}(1), \ldots, \mathbf{w}(K)$ to get a Random Forest
3. AdaBoost에서 가중치 $\mathbf{w}(k+1) = \phi(\mathbf{w}(k))$이므로, 연산자 $\mathbf{T}f(\mathbf{w}) = f(\phi(\mathbf{w}))$를 정의합니다 / In AdaBoost, $\mathbf{w}(k+1) = \phi(\mathbf{w}(k))$, so define operator $\mathbf{T}f(\mathbf{w}) = f(\phi(\mathbf{w}))$
4. **추측**: $\mathbf{T}$가 에르고딕(ergodic)이면, 불변 측도 $\pi$를 가지고, AdaBoost의 가중 투표 (Eq. 10)가 가중치가 $Q\pi$ 분포에서 랜덤으로 선택된 Random Forest와 등가가 됩니다 / **Conjecture**: if $\mathbf{T}$ is ergodic with invariant measure $\pi$, AdaBoost's weighted vote (Eq. 10) becomes equivalent to a Random Forest where weights are randomly selected from the $Q\pi$ distribution

**실증적 증거 / Empirical evidence**: AdaBoost를 75회 반복하여 50개의 가중치 집합을 만들고(처음 25개는 버림), Random Forest를 구성합니다. 결과: AdaBoost의 평균 오류 2.91% vs. Random Forest 2.94%로 거의 동일합니다.

AdaBoost run 75 times producing 50 weight sets (first 25 discarded), forming a Random Forest. Result: AdaBoost average error 2.91% vs. Random Forest 2.94% — nearly identical.

이 추측이 사실이라면 **AdaBoost가 트리를 추가해도 overfitting하지 않는 이유**도 설명됩니다 — Random Forest의 수렴 정리 (Theorem 1.2)가 적용되기 때문입니다.

If true, this would also explain **why AdaBoost doesn't overfit as trees are added** — because Theorem 1.2's convergence applies.

---

### §8: Effects of Output Noise — Noise에 대한 강건성 / Robustness to Noise

Dietterich (1998)의 방법론을 따라 5% noise (20개 중 1개의 라벨을 랜덤으로 변경)를 주입합니다.

Following Dietterich (1998)'s methodology, 5% noise is injected (randomly changing 1 in 20 labels).

**결과 (Table 4)**: AdaBoost는 noise에 **극적으로 취약**합니다. Breast Cancer에서 43.2%, Votes에서 48.9%의 오류 증가가 발생합니다. 반면 Forest-RI는 대부분 1-8% 증가에 그치며, 일부에서는 오히려 감소합니다(Sonar: -6.6%, Liver: -0.2%).

**Results (Table 4)**: AdaBoost is **dramatically vulnerable** to noise. Error increases of 43.2% on Breast Cancer, 48.9% on Votes. Forest-RI mostly shows 1-8% increase, and in some cases even decreases (Sonar: -6.6%, Liver: -0.2%).

**원인 분석 / Root cause analysis**: AdaBoost는 반복적으로 최근에 오분류된 샘플의 가중치를 높입니다. 잘못된 라벨의 샘플은 계속 오분류되므로 가중치가 계속 증가하여, AdaBoost가 이 noisy 샘플들에 **과도하게 집중**하게 됩니다. Random Forest는 각 트리가 독립적으로 랜덤 샘플에서 학습하므로, 특정 샘플에 가중치를 집중하지 않아 noise의 영향이 작습니다.

AdaBoost iteratively increases weights on recently misclassified samples. Mislabeled samples are continuously misclassified, so their weights keep increasing, causing AdaBoost to **overfocus** on noisy samples. Random Forest trains each tree independently on random samples without concentrating weight on any subset, minimizing noise impact.

---

### §9: Data with Many Weak Inputs — 약한 입력이 많은 데이터 / Many Weak Inputs

현대 응용(의료 진단, 문서 검색)에서는 수백~수천 개의 입력 변수가 있고, 각각은 약한 예측력만 가집니다. Breiman은 1,000개의 이진 입력, 10개 클래스의 합성 데이터를 생성합니다.

Modern applications (medical diagnosis, document retrieval) have hundreds to thousands of input variables, each with weak predictive power. Breiman generates synthetic data with 1,000 binary inputs and 10 classes.

결과: / Results:
- Bayes 오류율 1.0%, Naive Bayes 6.2% / Bayes error rate 1.0%, Naive Bayes 6.2%
- $F = 1$: 수렴이 매우 느리고 2,500 반복 후에도 10.7%. Strength 0.069, correlation 0.012, c/s2 = 2.5 / Very slow convergence, 10.7% after 2,500 iterations
- $F = 10$: 2,000 반복 후 3.0%. Strength 0.22, correlation 0.045, c/s2 = 0.91 / 3.0% after 2,000 iterations
- $F = 25$: 2,000 반복 후 **2.8%** (Bayes 1.0%에 근접!) / **2.8%** after 2,000 iterations (close to Bayes 1.0%!)

**주목할 점 / Notable**: 개별 트리의 오류율이 $F = 1$에서 80%, $F = 25$에서 60%로 매우 높음에도 불구하고, **correlation이 낮으면 약한 분류기들의 앙상블이 Bayes 오류에 가까운 성능을 달성**할 수 있습니다. AdaBoost는 개별 분류기가 너무 약해서 이 데이터에서 실행조차 되지 못합니다.

Even though individual tree error rates are very high (80% for $F = 1$, 60% for $F = 25$), **with low correlation, an ensemble of weak classifiers can achieve near-Bayes performance**. AdaBoost cannot even run on this data because individual classifiers are too weak.

---

### §10: Variable Importance — 변수 중요도 / Variable Importance

Random Forest의 "블랙박스" 문제를 해결하기 위한 접근입니다.

An approach to address the "black box" problem of Random Forests.

#### Permutation-based Variable Importance / 순열 기반 변수 중요도

알고리즘: / Algorithm:

1. 각 트리 구축 후, OOB 데이터로 원래의 오류율을 계산합니다 / After building each tree, compute the original error rate on OOB data
2. $m$번째 변수의 값을 OOB 샘플에서 **랜덤으로 셔플(permute)**합니다 / **Randomly shuffle (permute)** the values of the $m$-th variable in OOB samples
3. 셔플된 데이터로 다시 오류율을 계산합니다 / Compute error rate again with shuffled data
4. 오류율 증가분이 $m$번째 변수의 중요도입니다 / The increase in error rate is the importance of the $m$-th variable
5. 모든 변수 $m = 1, 2, \ldots, M$에 대해 반복합니다 / Repeat for all variables $m = 1, 2, \ldots, M$

**직관 / Intuition**: 변수 $m$이 정말 중요하다면, 그 값을 셔플하면 변수와 label 사이의 관계가 파괴되어 오류가 크게 증가합니다. 중요하지 않은 변수를 셔플하면 오류가 거의 변하지 않습니다.

If variable $m$ is truly important, shuffling its values destroys its relationship with the label, causing a large error increase. Shuffling an unimportant variable barely changes the error.

#### 실험 결과 / Experimental Results

**Diabetes 데이터 (Figure 4, 5)**: 변수 2가 압도적으로 중요(오류 35% 증가). 변수 8과 6도 중요하지만, 변수 2가 이미 포함된 후에는 변수 8의 추가 기여가 없습니다 — **변수 8은 변수 2와 유사한 정보를 담고 있기 때문**입니다.

**Diabetes data (Figures 4, 5)**: Variable 2 is overwhelmingly important (35% error increase). Variables 8 and 6 are also important, but after variable 2 is included, variable 8 adds no contribution — **because variable 8 carries similar information as variable 2**.

**Votes 데이터 (Figure 6)**: 변수 4(특정 의안에 대한 투표)가 압도적으로 중요(오류 300% 증가!). 변수 4 하나만으로도 전체 변수 사용과 거의 같은 정확도(4.3%)를 달성합니다.

**Votes data (Figure 6)**: Variable 4 (vote on a specific issue) is overwhelmingly important (300% error increase!). Using only variable 4 achieves nearly the same accuracy (4.3%) as using all variables.

이 결과는 **중복 정보를 가진 변수들의 상호작용**을 보여줍니다: 동일한 정보를 가진 $x_1, x_2$가 있으면, 둘 다 개별적으로 높은 중요도를 보이지만, 하나가 모델에 포함되면 다른 하나의 추가 기여는 없습니다.

This demonstrates **interactions among variables with redundant information**: if $x_1, x_2$ carry the same information, both show high individual importance, but once one is included, the other adds nothing.

---

### §11–12: Random Forests for Regression — 회귀 / Regression

Random Forest는 회귀에도 적용됩니다. 분류에서는 투표(voting)를 하지만, 회귀에서는 **평균(averaging)**을 합니다:

Random Forest also applies to regression. Classification uses voting, regression uses **averaging**:

$$\hat{Y}(\mathbf{x}) = \frac{1}{K}\sum_{k=1}^{K} h(\mathbf{x}, \Theta_k)$$

#### Theorem 11.1 (수렴) / Convergence

분류와 마찬가지로, 트리 수가 무한으로 가면 MSE가 수렴합니다:

As with classification, MSE converges as tree count goes to infinity:

$$E_{\mathbf{X},Y}(Y - av_k h(\mathbf{X}, \Theta_k))^2 \to E_{\mathbf{X},Y}(Y - E_\Theta h(\mathbf{X}, \Theta))^2 \tag{12}$$

#### Theorem 11.2 (핵심 결과) / Key Result

$E_\Theta h(\mathbf{X}, \Theta)$가 편향되지 않은(unbiased) 예측자라면 ($EY = E_{\mathbf{X}} h(\mathbf{X}, \Theta)$):

If $E_\Theta h(\mathbf{X}, \Theta)$ is an unbiased predictor ($EY = E_{\mathbf{X}} h(\mathbf{X}, \Theta)$):

$$PE^*(\text{forest}) \leq \bar{\rho} \cdot PE^*(\text{tree})$$

**의미 / Meaning**: Random Forest의 MSE는 **개별 트리의 MSE에 상관관계 $\bar{\rho}$를 곱한 것 이하**입니다. 분류에서의 상한($\bar{\rho}(1-s^2)/s^2$)보다 더 깔끔하고 직관적입니다. $\bar{\rho} = 0.5$이면 forest는 개별 트리보다 최소 2배 나은 성능을 보장합니다.

Random Forest MSE is at most **individual tree MSE multiplied by correlation $\bar{\rho}$**. Cleaner and more intuitive than the classification bound ($\bar{\rho}(1-s^2)/s^2$). If $\bar{\rho} = 0.5$, the forest guarantees at least 2× better performance than individual trees.

분류와의 흥미로운 차이: 회귀에서는 feature 수를 늘릴 때 correlation 증가가 느려서 주요 효과가 $PE^*(\text{tree})$의 감소입니다. 따라서 분류보다 더 많은 feature가 필요합니다.

An interesting difference from classification: in regression, correlation increases slowly with more features, so the main effect is the decrease in $PE^*(\text{tree})$. Therefore, more features are needed than in classification.

#### 실험 결과 (Table 6–8) / Experimental Results

8개 데이터셋에서 bagging, adaptive bagging, Random Forest를 비교합니다. Random Forest가 항상 bagging보다 좋지만, adaptive bagging이 큰 오류 감소를 보이는 데이터에서는 그 효과가 더 두드러집니다.

Compares bagging, adaptive bagging, and Random Forest on 8 datasets. Random Forest always beats bagging, but the effect is more pronounced on datasets where adaptive bagging shows large error reductions.

---

### §13: Conclusions — 결론

Breiman은 Random Forest의 핵심 성질을 요약합니다:

Breiman summarizes the key properties of Random Forests:

1. **Overfitting 없음**: 대수의 법칙에 의해 트리를 추가해도 일반화 오류가 수렴 / **No overfitting**: generalization error converges as trees are added, by the Law of Large Numbers
2. **적절한 랜덤성**: correlation을 낮추면서 strength를 유지하는 "적절한 종류의 랜덤성"이 핵심 / **Right kind of randomness**: the key is randomness that reduces correlation while maintaining strength
3. **Boosting과의 경쟁력**: AdaBoost와 동등하거나 더 나은 성능, 하지만 noise에 훨씬 강건 / **Competitive with boosting**: comparable or better than AdaBoost, but far more robust to noise
4. **편향 감소**: Random Forest가 어떻게 bias를 줄이는지는 아직 미스터리 — Bayesian 관점에서의 해석 가능성을 언급 / **Bias reduction**: how Random Forests reduce bias remains a mystery — mentions possible Bayesian interpretation
5. **랜덤성의 결합**: 다른 종류의 랜덤성(bagging, random features, random outputs)을 결합하면 더 나은 결과를 얻을 수 있음 / **Combining randomness**: combining different types (bagging, random features, random outputs) can yield better results

---

## 핵심 시사점 / Key Takeaways

1. **"더 많은 트리는 절대 해롭지 않다"**: 대수의 법칙에 의한 수렴 보장 — neural network나 개별 decision tree에서의 overfitting 문제를 근본적으로 해결한 최초의 앙상블 방법입니다. 실무적으로 "트리 수를 늘릴 여유가 있으면 늘려라"라는 간단한 지침을 줍니다.
   **"More trees never hurts"**: convergence guaranteed by the Law of Large Numbers — the first ensemble method to fundamentally solve the overfitting problem seen in neural networks and individual decision trees. Practically gives the simple guideline "if you can afford more trees, add them."

2. **일반화 오류 = f(strength, correlation)**: Theorem 2.3의 $PE^* \leq \bar{\rho}(1-s^2)/s^2$는 앙상블 방법의 성능을 **두 개의 측정 가능한 양**으로 분해하여, "왜 작동하는가"와 "어떻게 개선하는가"에 대한 명확한 프레임워크를 제공합니다.
   **Generalization error = f(strength, correlation)**: Theorem 2.3 decomposes ensemble performance into **two measurable quantities**, providing a clear framework for "why it works" and "how to improve."

3. **랜덤 feature 선택의 위력**: 각 노드에서 전체 변수 대신 소수($F$개)만 후보로 사용한다는 단순한 수정이, bagging의 핵심 한계(트리 간 상관관계)를 해결합니다. 놀랍게도 $F = 1$이라는 극단적 설정에서도 좋은 성능을 보여, "랜덤성이 정확도에 해롭다"는 직관이 틀렸음을 보여줍니다.
   **Power of random feature selection**: the simple modification of using only a few ($F$) candidate variables at each node, instead of all, resolves bagging's core limitation (inter-tree correlation). Remarkably, even the extreme $F = 1$ performs well, disproving the intuition that "randomness hurts accuracy."

4. **OOB = 공짜 cross-validation**: bootstrap의 부산물인 OOB 데이터를 활용하면, 별도의 test set이나 cross-validation 없이도 unbiased한 오류 추정, strength/correlation 모니터링, 변수 중요도 측정이 가능합니다. 이것은 Random Forest의 가장 실용적인 장점 중 하나입니다.
   **OOB = free cross-validation**: leveraging OOB data (a byproduct of bootstrap) enables unbiased error estimation, strength/correlation monitoring, and variable importance measurement — all without a separate test set or cross-validation. One of Random Forest's most practical advantages.

5. **Noise에 대한 강건성은 구조적**: AdaBoost가 noise에 취약한 이유(오분류 샘플에 가중치 집중)와 Random Forest가 강건한 이유(독립 랜덤 샘플링으로 어떤 샘플에도 가중치를 집중하지 않음)를 대비시킴으로써, 앙상블 방법의 noise 강건성이 알고리즘의 구조적 특성임을 보여줍니다.
   **Noise robustness is structural**: by contrasting why AdaBoost is vulnerable (concentrating weight on misclassified samples) with why Random Forest is robust (independent random sampling, no weight concentration on any sample), the paper shows that noise robustness is a structural property of the algorithm design.

6. **"AdaBoost는 Random Forest"라는 대담한 추측**: 겉보기에 완전히 다른 두 방법(순차적 가중치 vs. 독립 랜덤)이 수학적으로 등가일 수 있다는 추측은, 앙상블 학습의 깊은 구조적 통일성을 시사합니다. 비록 증명되지는 않았지만, 후속 연구의 방향을 제시합니다.
   **"AdaBoost is a Random Forest" conjecture**: the bold conjecture that two seemingly different methods (sequential weighting vs. independent random) might be mathematically equivalent suggests deep structural unity in ensemble learning. Though unproven, it sets direction for future research.

7. **약한 분류기의 힘**: §9의 실험에서 개별 트리 오류율이 60-80%인데도 앙상블이 Bayes 오류(1%)에 가까운 2.8%를 달성합니다. 이것은 "개별 분류기가 약해도 상관관계가 낮으면 앙상블이 강하다"는 Random Forest의 핵심 원리를 극적으로 보여줍니다.
   **Power of weak classifiers**: in §9's experiment, individual tree error rates of 60-80% yield an ensemble error of 2.8%, close to Bayes error (1%). This dramatically demonstrates Random Forest's core principle: "even weak classifiers, if uncorrelated, make a strong ensemble."

---

## 수학적 요약 / Mathematical Summary

### Random Forest 알고리즘 (Forest-RI) / Random Forest Algorithm (Forest-RI)

```
Input:  Training set T = {(x₁,y₁), ..., (xₙ,yₙ)}, number of trees B, features per node F
Output: Ensemble classifier H(x)

For b = 1 to B:
    1. Draw bootstrap sample T_b from T (N samples with replacement)
    2. Grow tree h_b on T_b:
       At each node:
         a. Randomly select F variables from M total
         b. Find best split among those F variables (using Gini impurity)
         c. Split the node into two children
       Until: node is pure or minimum node size reached
       Do NOT prune
    3. Store tree h_b

Predict H(x) = argmax_j Σ_b I(h_b(x) = j)    [majority vote]
```

### 핵심 정리 모음 / Key Theorems

| 정리 / Theorem | 내용 / Statement | 의미 / Significance |
|---|---|---|
| Theorem 1.2 | $PE^* \to P_{\mathbf{X},Y}(P_\Theta(h = Y) - \max_{j \neq Y} P_\Theta(h = j) < 0)$ | 트리 추가 시 수렴, overfitting 없음 / Convergence with more trees, no overfitting |
| Theorem 2.3 | $PE^* \leq \bar{\rho}(1-s^2)/s^2$ | 오류 상한 = f(correlation, strength) / Error bound = f(correlation, strength) |
| Theorem 11.1 | $MSE \to E(Y - E_\Theta h)^2$ | 회귀에서도 수렴 / Convergence in regression too |
| Theorem 11.2 | $PE^*(\text{forest}) \leq \bar{\rho} \cdot PE^*(\text{tree})$ | 회귀 오류 ≤ 상관관계 × 트리 오류 / Regression error ≤ correlation × tree error |

### 핵심 수식 / Key Equations

| 수식 / Equation | 역할 / Role |
|---|---|
| $mg = av_k I(h_k = Y) - \max_{j \neq Y} av_k I(h_k = j)$ | 유한 앙상블의 margin / Margin of finite ensemble |
| $mr = P_\Theta(h = Y) - \max_{j \neq Y} P_\Theta(h = j)$ | 무한 앙상블의 margin / Margin of infinite ensemble |
| $s = E_{\mathbf{X},Y} mr$ | 강도: margin의 기댓값 / Strength: expected margin |
| $c/s2 = \bar{\rho}/s^2$ | 성능 비율: 작을수록 좋음 / Performance ratio: smaller is better |

---

## 역사 속의 논문 / Paper in the Arc of History

```
1984  Breiman et al. — CART (Classification and Regression Trees)
       │  Decision tree의 기초 정립
       ▼
1990  Schapire — Boosting 이론적 증명
       │  약한 학습기를 결합하여 강한 학습기 가능
       ▼
1994  Ho — Random subspace method
       │  Feature 부분집합으로 트리 앙상블
       ▼
1996  Breiman — Bagging
       │  Bootstrap으로 분산 감소
       │
1996  Freund & Schapire — AdaBoost
       │  순차적 가중치 기반 앙상블, 최고 성능
       ▼
1997  Amit & Geman — Random feature selection + trees
       │  Breiman에게 직접 영감
       ▼
★ 2001  BREIMAN — RANDOM FORESTS ★
       │  Bagging + 랜덤 feature 선택의 이론적 통합
       │  Strength/correlation 프레임워크
       │  OOB 추정, 변수 중요도
       ▼
2001  Breiman — "Statistical Modeling: The Two Cultures"
       │  알고리즘적 모델링 vs. 통계적 모델링 논쟁
       ▼
2006  Ishwaran et al. — Random Survival Forests
       │  생존 분석으로 확장
       ▼
2017  현재까지 — Kaggle 등에서 tabular 데이터의 go-to 알고리즘
       │  XGBoost, LightGBM 등 gradient boosting과 양대 산맥
```

---

## 다른 논문과의 연결 / Connections to Other Papers

| 논문 / Paper | 관계 / Relationship |
|---|---|
| #8 Cortes & Vapnik (1995) — SVM | Margin 개념을 공유하지만, SVM은 단일 결정 경계를, RF는 앙상블 투표를 사용. SVM은 고차원에서 RF와 경쟁적 / Shares margin concept, but SVM uses single decision boundary while RF uses ensemble voting |
| #9 Hochreiter & Schmidhuber (1997) — LSTM | 시퀀스 데이터에서 LSTM이 우위, 정형 데이터에서 RF가 우위. 둘 다 각자의 도메인에서 오랜 기간 표준으로 군림 / LSTM dominates sequence data, RF dominates tabular data |
| #10 LeCun et al. (1998) — LeNet-5 | CNN은 이미지의 공간 구조를 활용, RF는 feature 간 관계를 랜덤 탐색. 이미지 분류에서 CNN이 압도적, 정형 데이터에서 RF가 우위 / CNN exploits spatial structure, RF randomly explores feature relationships |
| #12 Hinton et al. (2006) — Deep Belief Nets | 딥러닝 혁명의 시작. 이후 대부분의 비정형 데이터에서 신경망이 우위를 가져가지만, 정형 데이터에서 RF의 지위는 유지 / Start of deep learning revolution. Neural nets dominate unstructured data, but RF maintains dominance on tabular |
| Breiman (1996) — Bagging | RF의 직접적 선조. RF = Bagging + 랜덤 feature 선택. Bagging의 "트리 간 상관관계" 한계를 해결 / Direct ancestor of RF. RF = Bagging + random feature selection. Solves bagging's "inter-tree correlation" limitation |
| Freund & Schapire (1996) — AdaBoost | 논문의 주요 비교 대상. 정확도는 비슷하지만 RF가 noise에 강건하고 병렬화 가능. "AdaBoost는 RF" 추측으로 연결 / Main comparison target. Similar accuracy but RF is more robust and parallelizable. Connected by "AdaBoost is RF" conjecture |

---

## 참고문헌 / References

- Breiman, L., "Random Forests", *Machine Learning*, Vol. 45, pp. 5–32, 2001. [DOI: 10.1023/A:1010933404324]
- Breiman, L., "Bagging predictors", *Machine Learning*, Vol. 26(2), pp. 123–140, 1996.
- Freund, Y. & Schapire, R., "Experiments with a new boosting algorithm", *Proceedings of the 13th International Conference on Machine Learning*, pp. 148–156, 1996.
- Amit, Y. & Geman, D., "Shape quantization and recognition with randomized trees", *Neural Computation*, Vol. 9, pp. 1545–1588, 1997.
- Ho, T. K., "The random subspace method for constructing decision forests", *IEEE Trans. PAMI*, Vol. 20(8), pp. 832–844, 1998.
- Dietterich, T., "An experimental comparison of three methods for constructing ensembles of decision trees: Bagging, boosting and randomization", *Machine Learning*, pp. 1–22, 1998.
- Breiman, L., Friedman, J., Olshen, R. & Stone, C., *Classification and Regression Trees*, Wadsworth, 1984.
