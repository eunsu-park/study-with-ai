---
title: "Support-Vector Networks"
authors: Corinna Cortes, Vladimir Vapnik
year: 1995
journal: "Machine Learning, Vol. 20, pp. 273–297"
topic: Artificial Intelligence / Statistical Learning Theory
tags: [SVM, kernel trick, soft margin, optimal hyperplane, quadratic programming, VC dimension, structural risk minimization, classification]
status: completed
date_started: 2026-04-09
date_completed: 2026-04-09
---

# Support-Vector Networks
**Corinna Cortes & Vladimir Vapnik (1995)**

---

## 핵심 기여 / Core Contribution

이 논문은 **Support Vector Machine (SVM)**을 완전한 형태로 제시합니다. SVM은 세 가지 근본적 아이디어를 결합한 분류기입니다: (1) **최적 초평면(optimal hyperplane)** — 학습 데이터를 최대 마진으로 분리하는 유일한 결정 경계로, support vector라는 소수의 핵심 데이터 점만으로 결정됩니다; (2) **커널 트릭(kernel trick)** — 입력 벡터를 고차원 feature space로 명시적으로 변환하지 않고도, 커널 함수 $K(\mathbf{u}, \mathbf{v})$를 통해 그 공간에서의 내적을 효율적으로 계산하여 비선형 결정 경계를 구성합니다; (3) **소프트 마진(soft margin)** — slack variable $\xi_i$와 정규화 파라미터 $C$를 도입하여, 완벽히 분리 불가능한 현실 데이터에도 적용할 수 있게 합니다. NIST 숫자 인식 벤치마크에서 4차 다항식 커널 SVM이 1.1% 오류율을 달성하여, 도메인 지식을 활용한 신경망(LeNet)과 동등한 성능을 보입니다. 이 결과는 SVM이 이후 10년 이상 기계학습을 지배하는 출발점이 되었습니다.

This paper presents the **Support Vector Machine (SVM)** in its complete form. SVM is a classifier combining three fundamental ideas: (1) **optimal hyperplanes** — the unique decision boundary separating training data with maximum margin, determined by only a few critical data points called support vectors; (2) the **kernel trick** — constructing non-linear decision boundaries by efficiently computing dot products in high-dimensional feature space through kernel functions $K(\mathbf{u}, \mathbf{v})$, without explicitly performing the transformation; (3) **soft margins** — introducing slack variables $\xi_i$ and regularization parameter $C$ to handle real-world data that cannot be perfectly separated. On the NIST digit recognition benchmark, a 4th-degree polynomial kernel SVM achieves 1.1% error rate, matching neural networks (LeNet) that leverage domain-specific knowledge. This result marked the beginning of SVM's dominance in machine learning for over a decade.

---

## 읽기 노트 / Reading Notes

### Section 1: Introduction — 패턴 인식의 역사와 SVM의 위치 / History of Pattern Recognition and SVM's Position

#### Fisher에서 Perceptron까지 / From Fisher to Perceptron

논문은 패턴 인식의 역사를 Fisher (1936)의 선형 판별 분석(Linear Discriminant Analysis)에서 시작합니다. Fisher는 두 정규분포 집단 $N(\mathbf{m}_1, \Sigma_1)$과 $N(\mathbf{m}_2, \Sigma_2)$를 분리하는 최적 결정 함수가 **이차(quadratic)** 형태임을 보였습니다:

The paper begins with the history of pattern recognition, starting from Fisher's (1936) Linear Discriminant Analysis. Fisher showed that the optimal decision function for separating two normally distributed populations $N(\mathbf{m}_1, \Sigma_1)$ and $N(\mathbf{m}_2, \Sigma_2)$ is **quadratic**:

$$F_{\text{sq}}(\mathbf{x}) = \text{sign}\left[\frac{1}{2}(\mathbf{x} - \mathbf{m}_1)^T \Sigma_1^{-1}(\mathbf{x} - \mathbf{m}_1) - \frac{1}{2}(\mathbf{x} - \mathbf{m}_2)^T \Sigma_2^{-1}(\mathbf{x} - \mathbf{m}_2) + \ln\frac{|\Sigma_2|}{|\Sigma_1|}\right]$$

공분산이 동일한 경우($\Sigma_1 = \Sigma_2 = \Sigma$), 이것은 **선형** 판별 함수로 축소됩니다. 그러나 이차 함수는 $\frac{n(n+3)}{2}$개의 자유 파라미터를 추정해야 하므로, 데이터가 적을 때(예: $10n^2$ 미만) 이차 추정은 신뢰할 수 없습니다. Fisher는 따라서 $\Sigma_1 \neq \Sigma_2$인 경우에도 선형 판별 함수를 권장했습니다.

When covariances are equal ($\Sigma_1 = \Sigma_2 = \Sigma$), this reduces to a **linear** discriminant. However, the quadratic function requires estimating $\frac{n(n+3)}{2}$ free parameters, making quadratic estimation unreliable when data is scarce (fewer than $10n^2$ observations). Fisher therefore recommended linear discriminants even when $\Sigma_1 \neq \Sigma_2$.

이 관찰이 중요한 이유는 **편향-분산 트레이드오프(bias-variance tradeoff)**의 초기 사례이기 때문입니다. 더 복잡한 모델(이차)이 이론적으로는 최적이지만, 제한된 데이터에서는 더 단순한 모델(선형)이 더 나은 일반화를 보입니다. SVM은 바로 이 딜레마에 대한 우아한 해결책을 제공합니다.

This observation matters because it is an early instance of the **bias-variance tradeoff**. A more complex model (quadratic) is theoretically optimal, but with limited data, a simpler model (linear) generalizes better. SVM provides an elegant solution to precisely this dilemma.

#### Rosenblatt의 접근과 신경망의 등장 / Rosenblatt's Approach and Neural Networks

Rosenblatt (1962)의 perceptron은 입력을 비선형 변환한 feature space에서 선형 결정 함수를 구성하는 방식이었습니다:

Rosenblatt's (1962) perceptron constructs a linear decision function in a feature space obtained by non-linear transformation of the input:

$$I(\mathbf{x}) = \text{sign}\left(\sum_i \alpha_i z_i(\mathbf{x})\right)$$

여기서 $z_i(\mathbf{x})$는 hidden unit의 출력입니다. 그러나 Rosenblatt의 시대에는 모든 가중치를 동시에 최적화하는 알고리즘이 없었고, 출력층의 가중치 $\alpha_i$만 학습 가능했습니다. 1986년 backpropagation이 등장하면서 비로소 모든 가중치를 학습할 수 있게 되었고, 신경망은 "조각별 선형(piecewise linear)" 결정 함수를 구현합니다.

Here $z_i(\mathbf{x})$ are hidden unit outputs. In Rosenblatt's era, no algorithm existed to optimize all weights simultaneously — only output weights $\alpha_i$ were trainable. With backpropagation (1986), all weights became learnable, and neural networks implement "piecewise linear" decision functions.

#### SVM의 핵심 아이디어 / The Core Idea of SVM

SVM은 다른 접근을 취합니다. 입력 벡터를 비선형 매핑을 통해 고차원 feature space $Z$로 변환한 후, 그 공간에서 **특별한 속성(최대 마진)**을 가진 선형 결정면을 구성합니다. 결정적으로, 1992년에 Boser, Guyon & Vapnik이 보인 것처럼, 연산의 순서를 바꿀 수 있습니다:

SVM takes a different approach. Input vectors are mapped into a high-dimensional feature space $Z$ via a non-linear mapping, then a linear decision surface with **special properties (maximum margin)** is constructed in that space. Crucially, as Boser, Guyon & Vapnik (1992) showed, the order of operations can be interchanged:

- **기존 방식**: 먼저 변환 → feature space에서 내적 → 결정 / **Traditional**: transform first → dot products in feature space → decide
- **커널 방식**: 먼저 입력 공간에서 비교(커널) → 비선형 변환 → 결정 / **Kernel way**: compare in input space (kernel) first → non-linear transform → decide

이 순서 교환이 kernel trick의 본질이며, 임의 차수의 다항식 결정면을 효율적으로 구성할 수 있게 합니다.

This interchange of operations is the essence of the kernel trick, enabling efficient construction of polynomial decision surfaces of arbitrary degree.

---

### Section 2: Optimal Hyperplanes — 분리 가능한 경우의 최적 분류기 / Optimal Classifier for Separable Data

#### 문제 정의 / Problem Definition

레이블이 부여된 학습 데이터 $(y_1, \mathbf{x}_1), \ldots, (y_\ell, \mathbf{x}_\ell)$에서 $y_i \in \{-1, 1\}$이 주어질 때, 데이터가 **선형 분리 가능(linearly separable)**하다 함은 벡터 $\mathbf{w}$와 스칼라 $b$가 존재하여 다음이 성립하는 것입니다:

Given labeled training data $(y_1, \mathbf{x}_1), \ldots, (y_\ell, \mathbf{x}_\ell)$ with $y_i \in \{-1, 1\}$, the data is **linearly separable** if there exists a vector $\mathbf{w}$ and scalar $b$ such that:

$$y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1, \quad i = 1, \ldots, \ell$$

이 부등식은 모든 양성 클래스($y_i = 1$)의 점이 초평면 $\mathbf{w} \cdot \mathbf{x} + b = 0$의 한쪽에, 음성 클래스($y_i = -1$)의 점이 반대쪽에 있으며, 초평면에서의 함수값 절대값이 최소 1임을 의미합니다. 여기서 1이라는 값은 $\mathbf{w}$와 $b$의 스케일링으로 항상 달성 가능하므로 일반성을 잃지 않습니다.

This inequality means all positive class ($y_i = 1$) points are on one side of hyperplane $\mathbf{w} \cdot \mathbf{x} + b = 0$ and negative class ($y_i = -1$) points on the other, with functional margin at least 1. The value 1 is achievable without loss of generality through scaling of $\mathbf{w}$ and $b$.

#### 마진과 최적 초평면 / Margin and Optimal Hyperplane

**최적 초평면** $\mathbf{w}_0 \cdot \mathbf{x} + b_0 = 0$은 두 클래스 사이의 **마진(margin)**을 최대화하는 유일한 초평면입니다. 마진은 두 클래스의 가장 가까운 점들 사이의 거리로 정의됩니다:

The **optimal hyperplane** $\mathbf{w}_0 \cdot \mathbf{x} + b_0 = 0$ is the unique hyperplane that maximizes the **margin** — the distance between the closest points of the two classes:

$$\rho(\mathbf{w}, b) = \min_{\{x:y=1\}} \frac{\mathbf{x} \cdot \mathbf{w}}{|\mathbf{w}|} - \max_{\{x:y=-1\}} \frac{\mathbf{x} \cdot \mathbf{w}}{|\mathbf{w}|}$$

제약 조건 $y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1$과 결합하면, 최적 마진은 정확히 $\frac{2}{|\mathbf{w}_0|}$입니다. 따라서 **마진 최대화는 $\mathbf{w} \cdot \mathbf{w}$의 최소화와 동치**입니다.

Combined with constraint $y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1$, the optimal margin is exactly $\frac{2}{|\mathbf{w}_0|}$. Thus **maximizing margin is equivalent to minimizing $\mathbf{w} \cdot \mathbf{w}$**.

이것이 왜 좋은 전략인지 직관적으로 이해하면: 마진이 넓을수록, 새로운 테스트 데이터가 약간의 노이즈를 가지더라도 올바르게 분류될 가능성이 높습니다. 좁은 마진의 분류기는 학습 데이터에 "과적합"된 것이고, 넓은 마진의 분류기는 더 "안전한" 결정 경계를 가집니다.

Intuitively, why this is a good strategy: the wider the margin, the more likely new test data with slight noise will be correctly classified. A narrow-margin classifier is "overfitted" to training data, while a wide-margin classifier has a "safer" decision boundary.

#### Primal 문제에서 Dual 문제로 / From Primal to Dual Problem

**Primal 문제**는 직접적입니다:

The **Primal problem** is straightforward:

$$\min_{\mathbf{w}, b} \frac{1}{2}\mathbf{w} \cdot \mathbf{w} \quad \text{s.t.} \quad y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1$$

이를 Lagrange 승수법으로 변환합니다. Lagrangian은:

Transforming via Lagrange multipliers. The Lagrangian is:

$$L(\mathbf{w}, b, \mathbf{\Lambda}) = \frac{1}{2}\mathbf{w} \cdot \mathbf{w} - \sum_{i=1}^{\ell} \alpha_i [y_i(\mathbf{x}_i \cdot \mathbf{w} + b) - 1]$$

$\mathbf{w}$와 $b$에 대한 편미분을 0으로 놓으면 다음을 얻습니다:

Setting partial derivatives with respect to $\mathbf{w}$ and $b$ to zero yields:

$$\frac{\partial L}{\partial \mathbf{w}} = 0 \implies \mathbf{w}_0 = \sum_{i=1}^{\ell} \alpha_i y_i \mathbf{x}_i$$

$$\frac{\partial L}{\partial b} = 0 \implies \sum_{i=1}^{\ell} y_i \alpha_i = 0$$

첫 번째 조건이 핵심입니다: **최적 가중치 벡터 $\mathbf{w}_0$는 학습 데이터의 선형 결합**입니다. 그리고 $\alpha_i > 0$인 점들만 기여하므로, 이 점들이 **support vector**입니다. KKT 조건에 의해, $\alpha_i > 0$은 $y_i(\mathbf{w}_0 \cdot \mathbf{x}_i + b_0) = 1$인 경우에만 발생합니다 — 즉, support vector는 마진 경계 위에 정확히 놓인 점들입니다.

The first condition is key: **the optimal weight vector $\mathbf{w}_0$ is a linear combination of training data**. Since only points with $\alpha_i > 0$ contribute, these points are **support vectors**. By the KKT conditions, $\alpha_i > 0$ occurs only when $y_i(\mathbf{w}_0 \cdot \mathbf{x}_i + b_0) = 1$ — i.e., support vectors lie exactly on the margin boundary.

이를 Lagrangian에 대입하면 **Dual 문제**를 얻습니다:

Substituting back into the Lagrangian yields the **Dual problem**:

$$\max_{\mathbf{\Lambda}} W(\mathbf{\Lambda}) = \sum_{i=1}^{\ell} \alpha_i - \frac{1}{2} \sum_{i=1}^{\ell} \sum_{j=1}^{\ell} \alpha_i \alpha_j y_i y_j (\mathbf{x}_i \cdot \mathbf{x}_j)$$

제약: $\alpha_i \geq 0$, $\sum \alpha_i y_i = 0$.

Constraints: $\alpha_i \geq 0$, $\sum \alpha_i y_i = 0$.

이것은 **이차 프로그래밍(QP)** 문제로, 행렬 형태로 $W(\mathbf{\Lambda}) = \mathbf{\Lambda}^T \mathbf{1} - \frac{1}{2}\mathbf{\Lambda}^T \mathbf{D} \mathbf{\Lambda}$이며, $D_{ij} = y_i y_j \mathbf{x}_i \cdot \mathbf{x}_j$입니다. Dual 문제의 결정적 특성은 **데이터가 오직 내적 $\mathbf{x}_i \cdot \mathbf{x}_j$를 통해서만 등장한다는 것**입니다. 이것이 곧 kernel trick의 문을 엽니다.

This is a **Quadratic Programming (QP)** problem, in matrix form $W(\mathbf{\Lambda}) = \mathbf{\Lambda}^T \mathbf{1} - \frac{1}{2}\mathbf{\Lambda}^T \mathbf{D} \mathbf{\Lambda}$ with $D_{ij} = y_i y_j \mathbf{x}_i \cdot \mathbf{x}_j$. The crucial property of the dual is that **data appears only through dot products $\mathbf{x}_i \cdot \mathbf{x}_j$**. This opens the door to the kernel trick.

#### 일반화 오류 상계 / Generalization Error Bound

논문은 놀라운 결과를 제시합니다:

The paper presents a remarkable result:

$$E[\Pr(\text{error})] \leq \frac{E[\text{number of support vectors}]}{\text{number of training vectors}}$$

이 상계는 **feature space의 차원에 무관합니다**. 10억 차원의 feature space에서도, support vector의 수가 적으면 일반화 오류가 낮습니다. 논문의 실험에서 이 비율은 0.03(3%) 이하였으며, 실제 테스트 오류율은 이보다 훨씬 낮았습니다.

This bound is **independent of feature space dimensionality**. Even in a billion-dimensional feature space, if the number of support vectors is small, generalization error is low. In the paper's experiments, this ratio was below 0.03 (3%), and actual test error was much lower.

---

### Section 3: Soft Margin Hyperplane — 비분리 가능한 데이터의 처리 / Handling Non-Separable Data

#### 현실 문제로의 확장 / Extension to Real Problems

현실 데이터는 대부분 완벽히 선형 분리가 불가능합니다. 노이즈, 아웃라이어, 클래스 간 겹침이 존재합니다. Soft margin은 **slack variable** $\xi_i \geq 0$을 도입하여 일부 데이터 점이 마진 안쪽이나 반대편에 위치하는 것을 허용합니다:

Real data is mostly not perfectly linearly separable. Noise, outliers, and class overlap exist. Soft margin introduces **slack variables** $\xi_i \geq 0$ allowing some data points to be within or on the wrong side of the margin:

$$y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

- $\xi_i = 0$: 올바르게 분류되고 마진 밖에 위치 / Correctly classified, outside margin
- $0 < \xi_i < 1$: 올바르게 분류되었지만 마진 안에 위치 / Correctly classified but inside margin
- $\xi_i = 1$: 결정 경계 위에 위치 / On the decision boundary
- $\xi_i > 1$: 오분류 / Misclassified

#### Soft Margin 최적화 문제 / Soft Margin Optimization

목적함수는 마진 최대화(= $\|\mathbf{w}\|^2$ 최소화)와 오류 최소화(= $\sum \xi_i$ 최소화)를 동시에 달성해야 합니다:

The objective must simultaneously maximize margin (= minimize $\|\mathbf{w}\|^2$) and minimize errors (= minimize $\sum \xi_i$):

$$\min_{\mathbf{w}, b, \xi} \frac{1}{2}\mathbf{w} \cdot \mathbf{w} + C \cdot F\left(\sum_{i=1}^{\ell} \xi_i^\sigma\right)$$

논문은 **$\sigma = 1$과 $F(u) = u^2$**의 경우를 다룹니다:

The paper treats the case **$\sigma = 1$ and $F(u) = u^2$**:

$$\min_{\mathbf{w}, b, \xi} \frac{1}{2}\mathbf{w} \cdot \mathbf{w} + C \sum_{i=1}^{\ell} \xi_i^2$$

여기서 **$C$는 정규화 파라미터**로, 두 목적 사이의 트레이드오프를 조절합니다:
- $C \to \infty$: 어떤 오류도 허용하지 않음 (hard margin). 노이즈에 민감
- $C \to 0$: 오류에 관대하지만 마진이 넓음. 과소적합 위험
- 적절한 $C$: 일반화 성능을 최적화

Here **$C$ is the regularization parameter**, controlling the trade-off:
- $C \to \infty$: No errors tolerated (hard margin). Sensitive to noise
- $C \to 0$: Lenient on errors but wider margin. Risk of underfitting
- Appropriate $C$: Optimizes generalization

#### Soft Margin의 Dual 형태 / Dual Form of Soft Margin

Dual 문제는 hard margin과 거의 동일하지만, 한 가지 차이가 있습니다:

The dual is nearly identical to hard margin, with one difference:

$$\max_{\mathbf{\Lambda}} W(\mathbf{\Lambda}, \delta) = \sum \alpha_i - \frac{1}{2}\left[\sum \sum \alpha_i \alpha_j y_i y_j \mathbf{x}_i \cdot \mathbf{x}_j + \frac{\delta^2}{C}\right]$$

제약: $0 \leq \alpha_i \leq \delta$, $\sum \alpha_i y_i = 0$. 여기서 $\delta = \max(\alpha_1, \ldots, \alpha_\ell)$입니다.

Constraints: $0 \leq \alpha_i \leq \delta$, $\sum \alpha_i y_i = 0$, where $\delta = \max(\alpha_1, \ldots, \alpha_\ell)$.

핵심적 차이는 $\alpha_{\max}^2 / C$ 항의 추가입니다. 이것은 정확히 **정규화 항**으로, Lagrange 승수의 크기를 제한하여 개별 support vector가 결정 경계에 과도한 영향을 미치는 것을 방지합니다. $C$가 클수록 이 제한이 약해지고, 작을수록 강해집니다.

The key difference is the added $\alpha_{\max}^2 / C$ term. This is precisely a **regularization term**, limiting the magnitude of Lagrange multipliers to prevent individual support vectors from having excessive influence on the decision boundary. Larger $C$ weakens this limit; smaller $C$ strengthens it.

---

### Section 4: Kernel Method — 비선형 SVM의 핵심 / The Heart of Non-Linear SVM

#### Feature Space 매핑 / Feature Space Mapping

SVM의 힘은 선형 분류기를 비선형 문제에 적용하는 데서 나옵니다. 아이디어는 간단합니다: $n$차원 입력 $\mathbf{x}$를 비선형 함수 $\phi$를 통해 $N$차원 feature space로 변환합니다 ($N \gg n$):

SVM's power comes from applying linear classifiers to non-linear problems. The idea is simple: map $n$-dimensional input $\mathbf{x}$ into $N$-dimensional feature space via non-linear function $\phi$ ($N \gg n$):

$$\phi: \mathbb{R}^n \to \mathbb{R}^N$$

**예시**: 2차원 입력 $(x_1, x_2)$에 대한 2차 다항식 변환:

**Example**: 2nd-degree polynomial mapping for 2D input $(x_1, x_2)$:

$$\phi(x_1, x_2) = (x_1, x_2, x_1^2, x_2^2, x_1 x_2) \in \mathbb{R}^5$$

일반적으로 $n$차원 입력에 대한 $d$차 다항식 변환은 $N = \binom{n+d}{d}$ 차원의 feature space를 생성합니다. $n = 256$ (16×16 이미지), $d = 7$이면 feature space는 약 $10^{16}$ 차원입니다!

Generally, a $d$-degree polynomial mapping on $n$-dimensional input creates an $N = \binom{n+d}{d}$-dimensional feature space. For $n = 256$ (16×16 image), $d = 7$, the feature space has approximately $10^{16}$ dimensions!

#### Kernel Trick의 핵심 / The Kernel Trick's Essence

Dual 문제에서 데이터는 오직 내적 $\phi(\mathbf{x}_i) \cdot \phi(\mathbf{x}_j)$를 통해서만 등장합니다. **커널 함수**는 이 내적을 입력 공간에서 직접 계산합니다:

In the dual, data appears only through dot products $\phi(\mathbf{x}_i) \cdot \phi(\mathbf{x}_j)$. **Kernel functions** compute these dot products directly in input space:

$$\phi(\mathbf{u}) \cdot \phi(\mathbf{v}) \equiv K(\mathbf{u}, \mathbf{v})$$

이것이 왜 혁명적인가: $10^{16}$ 차원의 벡터를 명시적으로 구성하고 내적을 계산하는 것은 계산적으로 불가능합니다. 그러나 커널 함수는 원래 $n$차원 공간에서의 간단한 연산으로 동일한 결과를 줍니다.

Why this is revolutionary: explicitly constructing $10^{16}$-dimensional vectors and computing their dot products is computationally impossible. But kernel functions give identical results via simple operations in the original $n$-dimensional space.

#### 주요 커널 함수 / Key Kernel Functions

**다항식 커널 (Polynomial kernel)**:

$$K(\mathbf{u}, \mathbf{v}) = (\mathbf{u} \cdot \mathbf{v} + 1)^d$$

$d$차 다항식 결정 경계를 생성합니다. $d = 1$이면 선형, $d = 2$이면 이차곡선(타원, 쌍곡선 등), $d$가 높을수록 더 복잡한 경계를 만듭니다.

Produces polynomial decision boundaries of degree $d$. $d = 1$ gives linear, $d = 2$ gives conics (ellipses, hyperbolas, etc.), higher $d$ gives more complex boundaries.

**RBF(가우시안) 커널 (RBF/Gaussian kernel)**:

$$K(\mathbf{u}, \mathbf{v}) = \exp\left(-\frac{|\mathbf{u} - \mathbf{v}|^2}{\sigma^2}\right)$$

이 커널은 **무한 차원** feature space에 대응합니다 (가우시안의 Taylor 전개가 무한 항을 가지므로). $\sigma$는 각 support vector의 "영향 반경"을 제어합니다.

This kernel corresponds to an **infinite-dimensional** feature space (since the Gaussian Taylor expansion has infinite terms). $\sigma$ controls the "influence radius" of each support vector.

#### Mercer's Theorem — 유효한 커널의 조건 / Conditions for Valid Kernels

모든 함수가 커널이 될 수 있는 것은 아닙니다. **Mercer's Theorem**은 함수 $K(\mathbf{u}, \mathbf{v})$가 어떤 feature space에서의 내적에 대응하기 위한 필요충분 조건을 제공합니다:

Not every function can be a kernel. **Mercer's Theorem** provides the necessary and sufficient condition for $K(\mathbf{u}, \mathbf{v})$ to correspond to a dot product in some feature space:

$$\iint K(\mathbf{u}, \mathbf{v}) g(\mathbf{u}) g(\mathbf{v}) \, d\mathbf{u} \, d\mathbf{v} > 0$$

모든 $g$에 대해 ($\int g^2(\mathbf{u}) \, d\mathbf{u} < \infty$). 이것은 커널 행렬이 **양의 준정부호(positive semi-definite)**여야 한다는 조건과 동치입니다. Hilbert-Schmidt 이론에 의해, 이 조건을 만족하는 $K$는 고유함수 전개를 가집니다:

For all $g$ (with $\int g^2(\mathbf{u}) \, d\mathbf{u} < \infty$). This is equivalent to requiring the kernel matrix to be **positive semi-definite**. By Hilbert-Schmidt theory, $K$ satisfying this has an eigenfunction expansion:

$$K(\mathbf{u}, \mathbf{v}) = \sum_{i=1}^{\infty} \lambda_i \phi_i(\mathbf{u}) \cdot \phi_i(\mathbf{v})$$

여기서 $\lambda_i \geq 0$은 고유값, $\phi_i$는 고유함수입니다. 이것이 바로 $K$가 (무한 차원일 수 있는) feature space에서의 내적임을 보장합니다.

Where $\lambda_i \geq 0$ are eigenvalues and $\phi_i$ are eigenfunctions. This guarantees that $K$ is a dot product in a (possibly infinite-dimensional) feature space.

#### 커널화된 결정 함수 / Kernelized Decision Function

최종 결정 함수는 support vector와의 커널 값의 가중합입니다:

The final decision function is a weighted sum of kernel values with support vectors:

$$f(\mathbf{x}) = \text{sign}\left(\sum_{\text{support vectors}} y_i \alpha_i K(\mathbf{x}, \mathbf{x}_i) + b\right)$$

새로운 점 $\mathbf{x}$를 분류하려면, support vector들과의 커널 값만 계산하면 됩니다. 학습 데이터 전체가 아닌 support vector만 저장하면 되므로, 메모리 효율적이기도 합니다.

To classify a new point $\mathbf{x}$, only kernel values with support vectors need to be computed. Since only support vectors (not all training data) need to be stored, this is also memory-efficient.

---

### Section 5: General Features — SVM의 이론적 강점 / Theoretical Strengths of SVM

#### 효율성 / Efficiency

SVM 학습의 핵심은 QP 문제를 푸는 것입니다. 논문은 학습 데이터를 작은 부분들로 나누어 점진적으로 QP를 풀어가는 **청킹(chunking)** 전략을 설명합니다. 각 단계에서 현재 support vector와 아직 올바르게 분류되지 않은 새로운 데이터를 QP에 추가합니다. 이 과정은 최종적으로 전체 데이터에 대한 최적해로 수렴합니다.

SVM training's core is solving a QP problem. The paper describes a **chunking** strategy that incrementally solves QP by dividing training data into small portions. At each step, current support vectors and new incorrectly classified data are added to the QP. This process converges to the optimal solution for the full dataset.

학습 시간은 다항식의 차수가 아닌 **support vector의 수에만 의존**합니다. 이것은 7차 다항식 SVM이 1차 선형 SVM과 비슷한 시간에 학습될 수 있음을 의미합니다 (support vector 수가 비슷하다면).

Training time depends only on **the number of support vectors**, not polynomial degree. This means a 7th-degree polynomial SVM can train in similar time to a 1st-degree linear SVM (if support vector counts are similar).

#### 범용성 (Universal Machine) / Universality

커널 함수 $K(\mathbf{u}, \mathbf{v})$를 바꾸는 것만으로 완전히 다른 유형의 분류기를 구현할 수 있습니다:
- 다항식 커널 → 다항식 분류기 / Polynomial kernel → Polynomial classifier
- RBF 커널 → Radial Basis Function 네트워크 / RBF kernel → RBF network
- 시그모이드 커널 → 2층 신경망과 유사 / Sigmoid kernel → Similar to 2-layer neural network

이러한 범용성은 SVM을 **universal machine**으로 만들며, 동일한 학습 알고리즘(QP)으로 다양한 결정면을 생성할 수 있습니다.

This universality makes SVM a **universal machine** — the same training algorithm (QP) can generate diverse decision surfaces simply by changing the kernel.

#### 일반화 능력과 Structural Risk Minimization / Generalization and SRM

SVM은 Vapnik의 **Structural Risk Minimization (SRM)** 원리를 직접 구현합니다. SRM은 학습 기계의 일반화 능력이 다음에 의해 상계됨을 말합니다:

SVM directly implements Vapnik's **Structural Risk Minimization (SRM)** principle. SRM states that the generalization ability of a learning machine is bounded by:

$$\Pr(\text{test error}) \leq \text{Frequency}(\text{training error}) + \text{Confidence Interval}$$

Confidence Interval은 모델의 **VC-dimension**에 의존합니다. SVM의 최적 마진 방법은:
1. 첫 번째 항(training error)을 0으로 유지 (분리 가능한 경우)
2. 두 번째 항(confidence interval)을 $\mathbf{w} \cdot \mathbf{w}$ 최소화로 줄임

The Confidence Interval depends on the model's **VC-dimension**. SVM's optimal margin method:
1. Keeps the first term (training error) at zero (separable case)
2. Reduces the second term (confidence interval) by minimizing $\mathbf{w} \cdot \mathbf{w}$

핵심 통찰: 고정된 임계값 $b$를 가진 선형 함수 $I(\mathbf{x}) = \text{sign}(\mathbf{w} \cdot \mathbf{x} + b)$의 VC-dimension은 입력 차원과 같지만, $|\mathbf{w}| \leq C_\mathbf{w}$로 가중치의 크기를 제한하면 VC-dimension이 입력 차원보다 **작아질 수 있습니다**. 이것이 SVM이 고차원 feature space에서도 과적합하지 않는 이유입니다: 마진 최대화가 자동으로 모델 복잡도를 제한합니다.

Key insight: the VC-dimension of linear functions $I(\mathbf{x}) = \text{sign}(\mathbf{w} \cdot \mathbf{x} + b)$ with fixed threshold equals the input dimension, but constraining weight magnitude $|\mathbf{w}| \leq C_\mathbf{w}$ can make VC-dimension **smaller than** input dimension. This is why SVM doesn't overfit in high-dimensional feature spaces: margin maximization automatically constrains model complexity.

---

### Section 6: Experimental Analysis — 실험적 검증 / Experimental Validation

#### 2D 평면 실험 / 2D Plane Experiments

논문은 먼저 2D 데이터에서 2차 다항식 커널($d = 2$)로 SVM의 동작을 시각화합니다 (Figure 5). 이 실험은 SVM의 핵심 특성을 직관적으로 보여줍니다:
- Support vector (이중 원으로 표시)는 결정 경계 근처에 위치
- 오분류 점 (X로 표시)은 매우 적음
- 결정 경계는 매끄러운 비선형 곡선

The paper first visualizes SVM behavior on 2D data with quadratic polynomial kernel ($d = 2$) (Figure 5). This demonstrates SVM's key characteristics intuitively:
- Support vectors (marked with double circles) are near the decision boundary
- Misclassified points (marked with X) are very few
- The decision boundary is a smooth non-linear curve

#### US Postal Service Database / 미국 우체국 데이터베이스

16×16 필셀의 필기 숫자 이미지. 7,300개 학습, 2,000개 테스트. 다양한 차수의 다항식 커널로 실험:

16×16 pixel handwritten digit images. 7,300 training, 2,000 test. Experiments with polynomial kernels of various degrees:

| 분류기 / Classifier | 오류율 / Error Rate |
|---|---|
| Human performance | 2.5% |
| Decision tree (CART) | 17% |
| Decision tree (C4.5) | 16% |
| Best 2-layer neural network | 6.6% |
| Special 5-layer network (LeNet1) | 5.1% |
| **SVM (polynomial $d \geq 2$)** | **~4.3%** |

SVM은 2차 다항식부터 이미 특수 설계된 5층 신경망(LeNet1)보다 우수한 성능을 보입니다.

SVM with polynomial degree $\geq 2$ already outperforms the specially designed 5-layer neural network (LeNet1).

#### NIST Database — 핵심 벤치마크 / The Key Benchmark

28×28 픽셀, 60,000개 학습, 10,000개 테스트. 4차 다항식 커널, 전처리 없음. 10개의 one-vs-all 분류기를 구성:

28×28 pixels, 60,000 training, 10,000 test. 4th-degree polynomial kernel, no preprocessing. 10 one-vs-all classifiers:

| 분류기 / Classifier | 테스트 오류율 / Test Error |
|---|---|
| Linear classifier | 8.4% |
| $k$=3 nearest neighbor | 2.4% |
| LeNet1 (5-layer CNN) | 1.7% |
| LeNet4 (advanced CNN) | 1.1% |
| **SVM (polynomial $d = 4$)** | **1.1%** |

**Table 2의 놀라운 결과** — 다항식 차수별 성능 (US Postal):

**Table 2's remarkable results** — Performance by polynomial degree (US Postal):

| 차수 $d$ / Degree | 오류율 / Error | Support Vectors | Feature Space 차원 / Dimensionality |
|---|---|---|---|
| 1 | 12.0% | 200 | 256 |
| 2 | 4.7% | 127 | ~33,000 |
| 3 | 4.4% | 148 | ~$10^6$ |
| 4 | 4.3% | 165 | ~$10^9$ |
| 5 | 4.3% | 175 | ~$10^{12}$ |
| 6 | 4.2% | 185 | ~$10^{14}$ |
| 7 | 4.3% | 190 | ~$10^{16}$ |

이 표가 보여주는 것:
1. Feature space 차원이 $10^{16}$으로 폭발해도 오류율은 거의 동일 — **과적합이 발생하지 않음**
2. Support vector 수는 느리게 증가 — 일반화 상계 $E[\text{SV}] / \ell$이 항상 낮게 유지됨
3. 성능은 $d = 3$ 이후 포화 — 더 복잡한 모델이 항상 더 좋지는 않음

What this table shows:
1. Feature space dimension explodes to $10^{16}$ yet error stays nearly constant — **no overfitting occurs**
2. Support vector count grows slowly — generalization bound $E[\text{SV}] / \ell$ stays low
3. Performance saturates after $d = 3$ — more complex models aren't always better

#### SVM의 가장 인상적인 특성 / SVM's Most Impressive Characteristic

논문의 결론에서 인용된 벤치마크 연구 결과: "The support-vector network has excellent accuracy, which is most remarkable, because unlike the other high performance classifiers, it does not include knowledge about the geometry of the problem."

SVM은 이미지의 기하학적 구조(회전, 이동, 기울기 불변성)에 대한 도메인 지식을 전혀 사용하지 않고도 LeNet4와 동일한 1.1% 오류율을 달성했습니다. 데이터의 원시 픽셀 값만 사용합니다.

SVM achieves the same 1.1% error rate as LeNet4 without using any domain knowledge about the geometry of images (rotation, translation, skew invariance). It uses only raw pixel values.

---

### Appendix A: Mathematical Derivation — Lagrangian에서 Dual까지 / From Lagrangian to Dual

부록은 Section 2-3의 수학적 유도를 상세히 제공합니다.

The appendix provides detailed mathematical derivations from Sections 2-3.

#### Optimal Hyperplane 유도 / Derivation

1. Lagrangian 구성: $L(\mathbf{w}, b, \mathbf{\Lambda}) = \frac{1}{2}\mathbf{w} \cdot \mathbf{w} - \sum \alpha_i [y_i(\mathbf{x}_i \cdot \mathbf{w} + b) - 1]$
2. $\partial L / \partial \mathbf{w} = 0 \implies \mathbf{w}_0 = \sum \alpha_i y_i \mathbf{x}_i$
3. $\partial L / \partial b = 0 \implies \sum y_i \alpha_i = 0$
4. 대입하면 Dual: $W(\mathbf{\Lambda}) = \sum \alpha_i - \frac{1}{2} \sum \sum \alpha_i \alpha_j y_i y_j \mathbf{x}_i \cdot \mathbf{x}_j$

KKT 조건에서: $\alpha_i [y_i(\mathbf{x}_i \cdot \mathbf{w}_0 + b_0) - 1] = 0$. 이것은 $\alpha_i \neq 0$이면 반드시 $y_i(\mathbf{x}_i \cdot \mathbf{w}_0 + b_0) = 1$임을 의미합니다. 즉, **support vector는 마진 경계 위에 정확히 놓인 점들**입니다.

From the KKT conditions: $\alpha_i [y_i(\mathbf{x}_i \cdot \mathbf{w}_0 + b_0) - 1] = 0$. This means if $\alpha_i \neq 0$ then necessarily $y_i(\mathbf{x}_i \cdot \mathbf{w}_0 + b_0) = 1$. That is, **support vectors lie exactly on the margin boundary**.

---

## 핵심 시사점 / Key Takeaways

1. **마진 최대화는 일반화의 핵심이다**: SVM은 단순히 학습 데이터를 분류하는 것이 아니라, 두 클래스 사이의 "빈 공간"을 최대화함으로써 새로운 데이터에 대한 강건한 예측을 보장합니다. 이것은 Occam's Razor의 수학적 구현이며, structural risk minimization 원리의 직접적 실현입니다.

   **Margin maximization is the key to generalization**: SVM doesn't merely classify training data — it maximizes the "gap" between classes, ensuring robust predictions on new data. This is a mathematical implementation of Occam's Razor and a direct realization of structural risk minimization.

2. **Support vector의 희소성이 SVM의 강점을 결정한다**: 결정 경계는 전체 데이터가 아닌 소수의 support vector만으로 결정됩니다. US Postal 실험에서 7,300개 학습 데이터 중 127~200개만 support vector였습니다. 이 희소성이 메모리 효율성, 빠른 예측, 그리고 낮은 일반화 오류 상계를 동시에 보장합니다.

   **Sparsity of support vectors determines SVM's strength**: The decision boundary is determined by only a few support vectors, not all data. In the US Postal experiment, only 127–200 out of 7,300 training points were support vectors. This sparsity simultaneously guarantees memory efficiency, fast prediction, and low generalization error bounds.

3. **Kernel trick은 차원의 저주를 극복한다**: 내적만을 통해 고차원 feature space에서 작업할 수 있게 함으로써, $10^{16}$ 차원의 공간에서도 과적합 없이 분류가 가능합니다. 이것은 "데이터가 내적을 통해서만 알고리즘에 진입한다"는 구조적 특성에서 비롯됩니다.

   **The kernel trick overcomes the curse of dimensionality**: By working in high-dimensional feature space through dot products alone, classification without overfitting is possible even in $10^{16}$ dimensions. This stems from the structural property that "data enters the algorithm only through dot products."

4. **Soft margin은 이론적 우아함을 실용성으로 변환한다**: Hard margin SVM은 수학적으로 아름답지만 현실 데이터에 적용 불가능합니다. Slack variable $\xi_i$와 파라미터 $C$의 도입은 노이즈와 아웃라이어를 우아하게 처리하면서도 이론적 보장을 유지합니다.

   **Soft margin converts theoretical elegance into practicality**: Hard margin SVM is mathematically beautiful but inapplicable to real data. Introducing slack variables $\xi_i$ and parameter $C$ gracefully handles noise and outliers while maintaining theoretical guarantees.

5. **Dual 문제의 구조가 kernel trick을 가능하게 한다**: Primal에서 Dual로의 변환은 단순한 수학적 기교가 아닙니다. Dual에서 데이터가 내적으로만 등장한다는 사실이 kernel trick의 전제 조건이며, 이것이 SVM을 선형 분류기에서 임의의 비선형 분류기로 확장합니다.

   **The dual problem's structure enables the kernel trick**: The primal-to-dual transformation is not mere mathematical elegance. The fact that data appears only as dot products in the dual is the prerequisite for the kernel trick, extending SVM from linear to arbitrary non-linear classifiers.

6. **SVM은 도메인 지식 없이도 최고 성능을 달성한다**: NIST 벤치마크에서 SVM은 이미지의 기하학적 불변성을 전혀 활용하지 않고도 LeNet4와 동일한 1.1% 오류율을 달성했습니다. 이것은 SVM의 학습 알고리즘 자체가 데이터의 본질적 구조를 자동으로 포착함을 시사합니다.

   **SVM achieves top performance without domain knowledge**: On the NIST benchmark, SVM achieved the same 1.1% error as LeNet4 without leveraging any geometric invariances of images. This suggests SVM's learning algorithm automatically captures the essential structure of data.

7. **SVM은 통계적 학습 이론의 실용적 증명이다**: Vapnik의 VC 이론, structural risk minimization, 일반화 상계 등의 이론적 결과가 SVM에서 직접 실현됩니다. 이것은 "이론이 실무를 이끄는" 드문 사례로, 이론적 원리에서 출발하여 최고 성능의 알고리즘이 탄생했습니다.

   **SVM is a practical proof of statistical learning theory**: Vapnik's VC theory, structural risk minimization, and generalization bounds are directly realized in SVM. This is a rare case of "theory driving practice" — a top-performing algorithm born from theoretical principles.

8. **SVM은 universal machine이다**: 커널 함수를 바꾸는 것만으로 다항식 분류기, RBF 네트워크, 심지어 신경망과 유사한 분류기까지 구현할 수 있습니다. 동일한 QP 학습 알고리즘이 이 모든 것을 통합합니다.

   **SVM is a universal machine**: Simply changing the kernel function implements polynomial classifiers, RBF networks, and even neural network-like classifiers. The same QP training algorithm unifies all of these.

---

## 수학적 요약 / Mathematical Summary

### SVM 알고리즘 전체 흐름 / Complete SVM Algorithm Flow

**입력 / Input**: 학습 데이터 $\{(\mathbf{x}_i, y_i)\}_{i=1}^{\ell}$, $y_i \in \{-1, +1\}$, 커널 함수 $K$, 파라미터 $C$

**Step 1 — 커널 행렬 계산 / Compute Kernel Matrix**:
$$D_{ij} = y_i y_j K(\mathbf{x}_i, \mathbf{x}_j), \quad i, j = 1, \ldots, \ell$$

**Step 2 — QP 문제 풀기 / Solve QP Problem**:
$$\max_{\mathbf{\Lambda}} \sum_{i=1}^{\ell} \alpha_i - \frac{1}{2} \mathbf{\Lambda}^T \mathbf{D} \mathbf{\Lambda} \quad \text{s.t.} \quad 0 \leq \alpha_i \leq C, \quad \sum \alpha_i y_i = 0$$

(Note: soft margin의 경우 상한 $C$가 추가됨. Hard margin은 $C = \infty$. 논문의 soft margin formulation은 $F(u) = u^2$를 사용하지만, 현대적 표준 SVM은 $F(u) = u$ (hinge loss)를 사용하며 이 경우 $0 \leq \alpha_i \leq C$가 됩니다.)

**Step 3 — Support Vector 식별 / Identify Support Vectors**:
$$\text{SV} = \{i : \alpha_i > 0\}$$

**Step 4 — 편향 계산 / Compute Bias**:
$b$는 임의의 support vector $\mathbf{x}_s$ ($0 < \alpha_s < C$)를 사용하여:
$$b = y_s - \sum_{i \in \text{SV}} \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}_s)$$

**Step 5 — 예측 / Prediction**:
새로운 점 $\mathbf{x}$에 대해:
$$f(\mathbf{x}) = \text{sign}\left(\sum_{i \in \text{SV}} \alpha_i y_i K(\mathbf{x}, \mathbf{x}_i) + b\right)$$

### 주요 커널 함수 요약 / Key Kernel Functions Summary

| 커널 / Kernel | $K(\mathbf{u}, \mathbf{v})$ | Feature Space 차원 | 특성 / Characteristics |
|---|---|---|---|
| Linear | $\mathbf{u} \cdot \mathbf{v}$ | $n$ | 선형 경계 / Linear boundary |
| Polynomial | $(\mathbf{u} \cdot \mathbf{v} + 1)^d$ | $\binom{n+d}{d}$ | $d$차 다항식 경계 / Degree-$d$ polynomial boundary |
| RBF (Gaussian) | $\exp(-\|\mathbf{u}-\mathbf{v}\|^2/\sigma^2)$ | $\infty$ | 지역적 결정 경계 / Local decision boundary |

---

## 역사적 맥락의 타임라인 / Paper in the Arc of History

```
1936 ─── Fisher ─── Linear Discriminant Analysis
            │         선형 판별 분석의 시작
            │
1958 ─── Rosenblatt ─── Perceptron
            │         학습 가능한 선형 분류기
            │
1963 ─── Vapnik & Chervonenkis ─── VC Theory
            │         모델 복잡도의 수학적 정의
            │
1965 ─── Vapnik ─── Optimal Hyperplane (separable case)
            │         분리 가능한 경우의 최적 마진
            │
1969 ─── Minsky & Papert ─── Perceptron Limits
            │         선형 분류의 한계 → AI 겨울
            │
1982 ─── Vapnik ─── Structural Risk Minimization 원리
            │         일반화 능력의 이론적 프레임워크
            │
1986 ─── Rumelhart et al. ─── Backpropagation
            │         다층 신경망 학습 → 신경망 부활
            │
1992 ─── Boser, Guyon & Vapnik ─── Kernel Trick
            │         비선형 SVM의 탄생
            │
     ╔═══════════════════════════════════════════╗
     ║  ★ 1995 ─── Cortes & Vapnik              ║
     ║       Support-Vector Networks              ║
     ║       Soft margin + Kernel = 완전한 SVM     ║
     ╚═══════════════════════════════════════════╝
            │
1998 ─── Platt ─── Sequential Minimal Optimization (SMO)
            │         SVM 학습의 효율적 알고리즘
            │
2001 ─── Schölkopf & Smola ─── "Learning with Kernels"
            │         Kernel method의 체계화
            │
2006+ ── Deep Learning Revolution
            │         신경망이 다시 SVM을 추월
            │
현재 ── SVM은 여전히 소규모/표형 데이터에서 강력한 선택
```

---

## 다른 논문과의 연결 / Connections to Other Papers

| 논문 / Paper | 연결 / Connection |
|---|---|
| #1 McCulloch & Pitts (1943) | SVM의 결정 함수 $\text{sign}(\mathbf{w} \cdot \mathbf{x} + b)$는 M-P 뉴런의 직접적 일반화. 임계값 활성화 함수를 공유 / SVM's decision function is a direct generalization of the M-P neuron, sharing the threshold activation |
| #3 Rosenblatt (1958) | Perceptron은 SVM의 전신. 둘 다 선형 결정면을 구성하지만, perceptron은 "아무" 분리면, SVM은 "최적의" 분리면을 찾음 / Perceptron is SVM's predecessor. Both construct linear decision surfaces, but perceptron finds "any" separator while SVM finds the "optimal" one |
| #4 Minsky & Papert (1969) | XOR 문제 = 선형 비분리. SVM의 kernel trick이 이 한계를 우회: 고차원으로 올리면 XOR도 선형 분리 가능 / XOR = linearly inseparable. SVM's kernel trick bypasses this: lifting to higher dimensions makes XOR linearly separable |
| #6 Rumelhart et al. (1986) | Backpropagation은 신경망의 학습 알고리즘, SVM은 QP를 사용. 두 접근의 근본적 차이: 경사 하강법 vs 볼록 최적화. SVM은 전역 최적해 보장 / Backprop is the NN training algorithm; SVM uses QP. Fundamental difference: gradient descent vs convex optimization. SVM guarantees global optimum |
| #7 LeCun et al. (1989) | CNN은 SVM의 직접적 경쟁자. CNN은 도메인 지식(지역 수용장, 가중치 공유)을 활용, SVM은 순수 데이터 기반. NIST에서 동등한 성능 / CNN is SVM's direct competitor. CNN uses domain knowledge; SVM is purely data-driven. Equal performance on NIST |
| #5 Hopfield (1982) | 에너지 함수 최소화와 QP 풀기 사이의 유사성. 둘 다 물리학/최적화 관점에서 학습을 바라봄 / Similarity between energy minimization and QP solving. Both view learning from physics/optimization perspective |
| #9 Hochreiter & Schmidhuber (1997) | LSTM은 시퀀스 데이터에, SVM은 고정 길이 벡터에 최적화. 다른 도메인을 지배 / LSTM for sequences, SVM for fixed-length vectors. Dominate different domains |
| #11 Breiman (2001) | Random Forest는 SVM의 주요 경쟁자. 표형 데이터에서 종종 SVM보다 실용적 (튜닝이 덜 필요) / Random Forest is SVM's main competitor, often more practical for tabular data (less tuning needed) |

---

## 참고문헌 / References

- Cortes, C. & Vapnik, V., "Support-Vector Networks", *Machine Learning*, 20, pp. 273–297, 1995.
- Vapnik, V., "Estimation of Dependences Based on Empirical Data", Springer-Verlag, 1982.
- Boser, B., Guyon, I. & Vapnik, V., "A Training Algorithm for Optimal Margin Classifiers", *COLT*, 1992.
- Fisher, R.A., "The Use of Multiple Measurements in Taxonomic Problems", *Annals of Eugenics*, 1936.
- Mercer, J., "Functions of Positive and Negative Type and Their Connection with the Theory of Integral Equations", *Phil. Trans. Royal Soc.*, 1909.
- LeCun, Y. et al., "Backpropagation Applied to Handwritten Zip Code Recognition", *Neural Computation*, 1989.
- Bottou, L. et al., "Comparison of Classifier Methods: A Case Study in Handwritten Digit Recognition", *ICPR*, 1994.
