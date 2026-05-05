---
title: "Support-Vector Networks — Pre-reading Briefing"
paper: "Support-Vector Networks"
authors: Corinna Cortes, Vladimir Vapnik
year: 1995
journal: "Machine Learning, Vol. 20, pp. 273–297"
type: briefing
date: 2026-04-09
---

# Support-Vector Networks — 사전 읽기 브리핑 / Pre-reading Briefing

## 핵심 기여 / Core Contribution

이 논문은 **Support Vector Machine (SVM)**을 소개합니다. SVM은 입력 벡터를 비선형 매핑을 통해 매우 고차원의 feature space로 변환한 뒤, 그 공간에서 **최대 마진(maximal margin)**을 가지는 최적 분리 초평면(optimal separating hyperplane)을 구성하는 분류기입니다. 핵심적인 세 가지 아이디어를 결합합니다: (1) 최적 초평면 — support vector만으로 결정 경계를 표현, (2) **kernel trick** — 고차원 feature space에서의 내적을 입력 공간의 커널 함수로 대체하여 계산 효율성 확보, (3) **soft margin** — 학습 데이터에 오류가 있는 비분리 가능한 경우까지 확장. NIST 숫자 인식 벤치마크에서 신경망(LeNet)과 동등한 1.1% 오류율을 달성하며, 도메인 지식 없이도 최고 수준의 성능을 보여줍니다.

This paper introduces **Support Vector Machines (SVMs)** — a classifier that maps input vectors via a non-linear mapping into a very high-dimensional feature space, then constructs an **optimal separating hyperplane** with maximal margin in that space. It combines three key ideas: (1) optimal hyperplanes — the decision boundary is expressed solely in terms of support vectors, (2) the **kernel trick** — dot products in feature space are replaced by kernel functions in input space for computational efficiency, (3) **soft margins** — extending the approach to non-separable cases where training errors exist. On the NIST digit recognition benchmark, SVMs achieve 1.1% error rate, matching neural networks (LeNet) without any domain-specific knowledge.

---

## 역사적 맥락 / Historical Context

```
1936  Fisher의 선형 판별 분석 (Linear Discriminant Analysis)
  │
1958  Rosenblatt의 Perceptron — 학습 가능한 선형 분류기
  │
1969  Minsky & Papert — 단층 perceptron의 한계 증명 → AI 겨울
  │
1982  Vapnik — optimal hyperplane 이론 (분리 가능한 경우)
  │
1986  Rumelhart et al. — Backpropagation → 신경망 부활
  │
1989  LeCun — CNN으로 우편번호 인식
  │
1992  Boser, Guyon & Vapnik — kernel trick으로 비선형 SVM 확장
  │
★1995  Cortes & Vapnik — Soft margin SVM + 커널 조합 = 이 논문 ★
  │
1998  LeCun et al. — LeNet-5 (SVM의 경쟁자)
  │
2001  Breiman — Random Forests (또 다른 강력한 경쟁자)
```

1990년대 중반은 신경망과 통계적 학습 이론이 경쟁하던 시기입니다. 신경망은 backpropagation으로 강력했지만 이론적 보장이 부족했습니다. Vapnik의 통계적 학습 이론(VC dimension, structural risk minimization)은 일반화 능력에 대한 엄밀한 수학적 보장을 제공했고, SVM은 그 이론을 실용적 알고리즘으로 구현한 것입니다. SVM은 이후 10년 이상 기계학습의 지배적 알고리즘이 됩니다.

The mid-1990s saw competition between neural networks and statistical learning theory. Neural networks were powerful via backpropagation but lacked theoretical guarantees. Vapnik's statistical learning theory (VC dimension, structural risk minimization) provided rigorous mathematical guarantees on generalization, and SVMs were the practical realization of that theory. SVMs would dominate machine learning for over a decade.

---

## 필요한 배경 지식 / Prerequisites

### 1. 선형대수학 / Linear Algebra
- **벡터 내적 (dot product)**: $\mathbf{u} \cdot \mathbf{v} = \sum_i u_i v_i$ — SVM의 모든 계산이 내적에 기반
- **초평면 (hyperplane)**: $\mathbf{w} \cdot \mathbf{x} + b = 0$ — $n$차원 공간을 두 부분으로 나누는 $(n-1)$차원 평면
- **벡터 노름 (norm)**: $|\mathbf{w}| = \sqrt{\mathbf{w} \cdot \mathbf{w}}$ — 마진 계산에 사용

### 2. 최적화 이론 / Optimization Theory
- **Lagrange 승수법 (Lagrange multipliers)**: 제약 조건이 있는 최적화 문제를 풀기 위한 방법. 제약 $g(\mathbf{x}) = 0$ 하에서 $f(\mathbf{x})$를 최적화할 때 $L = f - \alpha g$ 구성
- **이차 프로그래밍 (Quadratic Programming, QP)**: 이차 목적함수 + 선형 제약 조건의 최적화 문제. SVM 학습은 QP 문제로 귀결
- **KKT 조건 (Karush-Kuhn-Tucker conditions)**: 부등식 제약이 있는 최적화의 필요충분 조건. $\alpha_i [y_i(\mathbf{w} \cdot \mathbf{x}_i + b) - 1] = 0$ — support vector를 정의하는 핵심

### 3. 이전 논문의 개념 / Concepts from Prior Papers
- **Perceptron (논문 #3)**: 선형 분류기의 기본 개념. SVM은 "최적의" 선형 분류기
- **Backpropagation (논문 #6)**: 신경망 학습 방법. SVM과 대비되는 접근법
- **CNN (논문 #7)**: SVM의 벤치마크 경쟁 상대

### 4. 통계적 학습 이론 기초 / Statistical Learning Theory Basics
- **VC dimension**: 모델이 완벽히 분류할 수 있는 점의 최대 개수. 모델 복잡도의 척도
- **Structural Risk Minimization (SRM)**: 학습 오류 + 모델 복잡도를 동시에 최소화하는 원리. Occam's Razor의 수학적 형식화

---

## 핵심 용어 / Key Vocabulary

| 용어 / Term | 직관적 설명 / Intuitive Explanation |
|---|---|
| **Support Vector** | 결정 경계(초평면)에 가장 가까운 학습 데이터 점들. 이 점들만으로 초평면이 결정됨. 나머지 데이터는 제거해도 결과 동일 / Training points closest to the decision boundary. Only these determine the hyperplane — all other data can be removed without changing the result |
| **Margin** | 두 클래스 사이의 "빈 공간"의 너비. SVM은 이 마진을 최대화 / The width of the "gap" between two classes. SVM maximizes this margin |
| **Optimal Hyperplane** | 마진을 최대화하는 유일한 분리 초평면 / The unique separating hyperplane that maximizes the margin |
| **Kernel Trick** | 고차원 feature space에서의 내적을 직접 계산하지 않고, 입력 공간에서 커널 함수 $K(\mathbf{u}, \mathbf{v})$로 대체하는 기법. 무한 차원도 가능 / Computing dot products in high-dimensional feature space without explicitly transforming, by using a kernel function $K(\mathbf{u}, \mathbf{v})$ in input space. Even infinite dimensions become tractable |
| **Soft Margin** | 학습 데이터가 완벽히 분리 불가능할 때, 일부 오분류를 허용하면서 마진을 최대화하는 방법. slack variable $\xi_i$ 도입 / When data is not perfectly separable, allowing some misclassifications while maximizing margin, via slack variables $\xi_i$ |
| **Slack Variable ($\xi_i$)** | 각 데이터 점의 마진 위반 정도를 나타내는 변수. $\xi_i = 0$이면 올바르게 분류, $\xi_i > 1$이면 오분류 / A variable measuring how much each point violates the margin. $\xi_i = 0$ means correctly classified, $\xi_i > 1$ means misclassified |
| **Regularization Parameter ($C$)** | 마진 최대화와 오분류 허용 사이의 트레이드오프를 조절. $C$가 크면 오분류에 엄격, 작으면 관대 / Controls the trade-off between maximizing margin and allowing misclassifications. Large $C$ = strict, small $C$ = lenient |
| **Polynomial Kernel** | $K(\mathbf{u}, \mathbf{v}) = (\mathbf{u} \cdot \mathbf{v} + 1)^d$ — 차수 $d$의 다항식 결정 경계를 생성 / Produces polynomial decision boundaries of degree $d$ |
| **RBF Kernel** | $K(\mathbf{u}, \mathbf{v}) = \exp(-\|\mathbf{u} - \mathbf{v}\|^2 / \sigma^2)$ — 가우시안 형태의 비선형 결정 경계 / Gaussian-shaped non-linear decision boundaries |
| **Dual Problem** | 원래 최적화 문제(primal)를 Lagrange 승수에 대한 문제로 변환한 것. kernel trick 적용이 가능해짐 / The original optimization rewritten in terms of Lagrange multipliers, enabling the kernel trick |
| **Mercer's Theorem** | 함수가 유효한 커널(내적에 대응)이 되기 위한 조건. 양의 정부호 조건 / The condition for a function to be a valid kernel (corresponding to a dot product). Positive semi-definiteness |

---

## 수식 미리보기 / Equations Preview

### 1. 선형 분류기의 결정 함수 / Linear Classifier Decision Function

$$f(\mathbf{x}) = \text{sign}(\mathbf{w} \cdot \mathbf{x} + b)$$

가장 기본적인 형태. 가중치 벡터 $\mathbf{w}$와 데이터 $\mathbf{x}$의 내적 + 편향 $b$의 부호로 분류합니다.

The most basic form. Classification is determined by the sign of the dot product between weight vector $\mathbf{w}$ and data $\mathbf{x}$, plus bias $b$.

### 2. 마진의 정의 / Margin Definition

$$\rho(\mathbf{w}, b) = \frac{2}{|\mathbf{w}|}$$

마진은 $\mathbf{w}$의 노름에 반비례합니다. 따라서 **마진 최대화 = $|\mathbf{w}|^2$ 최소화**입니다.

The margin is inversely proportional to the norm of $\mathbf{w}$. Thus **maximizing margin = minimizing $|\mathbf{w}|^2$**.

### 3. 최적 초평면의 Primal 문제 / Optimal Hyperplane Primal Problem

$$\min_{\mathbf{w}, b} \frac{1}{2}\mathbf{w} \cdot \mathbf{w} \quad \text{s.t.} \quad y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1, \quad i = 1, \ldots, \ell$$

$\mathbf{w}$의 크기를 최소화(마진 최대화)하되, 모든 학습 데이터가 올바르게 분류되는 제약 조건을 만족해야 합니다.

Minimize the magnitude of $\mathbf{w}$ (maximize margin) subject to all training data being correctly classified.

### 4. Dual 문제 (Lagrange 승수) / Dual Problem

$$W(\mathbf{\Lambda}) = \sum_{i=1}^{\ell} \alpha_i - \frac{1}{2} \sum_{i=1}^{\ell} \sum_{j=1}^{\ell} \alpha_i \alpha_j y_i y_j \mathbf{x}_i \cdot \mathbf{x}_j$$

이것을 **최대화**합니다. 제약: $\alpha_i \geq 0$, $\sum \alpha_i y_i = 0$. 내적 $\mathbf{x}_i \cdot \mathbf{x}_j$만 등장하는 것이 kernel trick의 열쇠입니다.

**Maximize** this. Constraints: $\alpha_i \geq 0$, $\sum \alpha_i y_i = 0$. The fact that only dot products $\mathbf{x}_i \cdot \mathbf{x}_j$ appear is the key to the kernel trick.

### 5. Soft Margin 목적함수 / Soft Margin Objective

$$\min_{\mathbf{w}, b, \xi} \frac{1}{2}\mathbf{w} \cdot \mathbf{w} + C \sum_{i=1}^{\ell} \xi_i^2$$

제약: $y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1 - \xi_i$, $\xi_i \geq 0$

$C$가 오분류 페널티를 조절합니다. $C \to \infty$이면 hard margin과 동일해집니다.

$C$ controls the misclassification penalty. As $C \to \infty$, this becomes equivalent to the hard margin case.

### 6. Kernel Trick — 핵심 수식 / The Key Equation

$$f(\mathbf{x}) = \text{sign}\left(\sum_{i=1}^{\ell} y_i \alpha_i K(\mathbf{x}, \mathbf{x}_i) + b\right)$$

여기서 $K(\mathbf{u}, \mathbf{v}) = \phi(\mathbf{u}) \cdot \phi(\mathbf{v})$. 고차원 변환 $\phi$를 명시적으로 계산하지 않고, 커널 함수 $K$만으로 분류가 가능합니다.

Where $K(\mathbf{u}, \mathbf{v}) = \phi(\mathbf{u}) \cdot \phi(\mathbf{v})$. Classification is possible using only the kernel function $K$, without explicitly computing the high-dimensional mapping $\phi$.

### 7. 일반화 오류 상계 / Generalization Error Bound

$$E[\Pr(\text{error})] \leq \frac{E[\text{number of support vectors}]}{\text{number of training vectors}}$$

놀랍도록 간결한 결과: 일반화 오류는 support vector의 비율로 상계됩니다. Feature space의 차원에 무관합니다!

A remarkably simple result: generalization error is bounded by the ratio of support vectors. Independent of feature space dimensionality!

---

## 논문 구조 미리보기 / Paper Structure Preview

| 섹션 / Section | 내용 / Content |
|---|---|
| 1. Introduction | Fisher의 판별 분석부터 perceptron, 신경망까지의 역사. SVM의 핵심 아이디어 소개 |
| 2. Optimal Hyperplanes | 분리 가능한 경우의 최적 초평면 이론. QP 문제로의 정식화 |
| 3. Soft Margin Hyperplane | 비분리 가능한 경우로의 확장. Slack variable과 $C$ 파라미터 도입 |
| 4. Kernel Method | Kernel trick의 이론. Mercer's theorem. 다항식/RBF 커널 |
| 5. General Features | SVM의 효율성, 범용성, 일반화 능력에 대한 이론적 분석 |
| 6. Experimental Analysis | 2D 실험 + US Postal Service/NIST 숫자 인식 벤치마크 |
| 7. Conclusion | 세 가지 핵심 아이디어의 조합으로서의 SVM |
| Appendix A | 최적 초평면과 soft margin의 수학적 유도 상세 |

---

## 읽기 팁 / Reading Tips

1. **Section 2-3이 수학적 핵심**: 최적화 문제의 primal → dual 변환을 천천히 따라가세요. KKT 조건에서 support vector가 자연스럽게 나타나는 것이 아름다운 부분입니다.
2. **Section 4의 kernel trick**: $\phi(\mathbf{u}) \cdot \phi(\mathbf{v}) = K(\mathbf{u}, \mathbf{v})$라는 한 줄이 전체 방법론의 열쇠입니다. 이것이 왜 작동하는지 이해하면 SVM의 본질을 파악한 것입니다.
3. **Table 2 (p.288)**: 다항식 차수를 올려도 성능이 거의 변하지 않는 것이 SVM의 강력한 일반화 능력을 보여줍니다. Feature space 차원이 $10^{16}$이 되어도 과적합하지 않습니다!
4. **Figure 9 (p.290)**: SVM이 신경망과 동등한 성능을 달성하는 벤치마크 결과. 도메인 지식 없이 이 성능을 달성한 것이 핵심입니다.

1. **Sections 2-3 are the mathematical core**: Follow the primal → dual transformation slowly. The natural emergence of support vectors from KKT conditions is the elegant part.
2. **Section 4's kernel trick**: The single line $\phi(\mathbf{u}) \cdot \phi(\mathbf{v}) = K(\mathbf{u}, \mathbf{v})$ is the key to the entire methodology. Understanding why this works means grasping the essence of SVMs.
3. **Table 2 (p.288)**: Performance barely changes as polynomial degree increases — demonstrating SVM's powerful generalization. No overfitting even with $10^{16}$-dimensional feature space!
4. **Figure 9 (p.290)**: SVM matches neural networks on benchmarks, achieving this without any domain knowledge.
