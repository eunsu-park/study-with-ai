# 08. Probability for Machine Learning

## Learning Objectives

- Understand and apply basic probability axioms, conditional probability, and Bayes' theorem
- Learn the concept of random variables and the differences between discrete and continuous distributions
- Calculate and interpret key statistics of random variables such as expectation, variance, and covariance
- Learn the characteristics and applications of probability distributions commonly used in machine learning
- Implement probabilistic inference and Bayesian updates using Bayes' theorem
- Understand the difference between generative and discriminative models from a probabilistic perspective

---

## 1. Foundations of Probability

### 1.1 Axioms of Probability

**Sample Space** $\Omega$: set of all possible outcomes

**Event** $A$: subset of the sample space

**Probability Measure** $P$ satisfies the following axioms:

1. **Non-negativity**: $P(A) \geq 0$ for all $A$
2. **Normalization**: $P(\Omega) = 1$
3. **Countable Additivity**: For mutually exclusive events $A_1, A_2, \ldots$
   $$P\left(\bigcup_{i=1}^\infty A_i\right) = \sum_{i=1}^\infty P(A_i)$$

### 1.2 Conditional Probability

Probability of event $A$ occurring given that event $B$ has occurred:

$$
P(A|B) = \frac{P(A \cap B)}{P(B)}, \quad \text{if } P(B) > 0
$$

**Intuition**: When we know $B$ has occurred, the sample space shrinks from $\Omega$ to $B$.

### 1.3 Independence

Events $A$ and $B$ are independent if:

$$
P(A \cap B) = P(A) \cdot P(B)
$$

or equivalently:
$$
P(A|B) = P(A)
$$

### 1.4 Law of Total Probability

If $B_1, \ldots, B_n$ form a partition of the sample space:

$$
P(A) = \sum_{i=1}^n P(A|B_i)P(B_i)
$$

### 1.5 Bayes' Theorem

$$
P(A|B) = \frac{P(B|A)P(A)}{P(B)}
$$

or using the law of total probability:

$$
P(A|B) = \frac{P(B|A)P(A)}{\sum_{i} P(B|A_i)P(A_i)}
$$

**Terminology:**
- $P(A)$: **prior probability**
- $P(B|A)$: **likelihood**
- $P(A|B)$: **posterior probability**
- $P(B)$: **marginal probability** or **evidence**

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 베이즈 정리 예제: 의료 진단
# 질병 유병률: 1%
P_disease = 0.01
P_no_disease = 1 - P_disease

# 검사 정확도
# 민감도 (sensitivity): 병이 있을 때 양성 확률
P_positive_given_disease = 0.95
# 특이도 (specificity): 병이 없을 때 음성 확률
P_negative_given_no_disease = 0.95
P_positive_given_no_disease = 1 - P_negative_given_no_disease

# 전확률: 양성 검사 확률
P_positive = (P_positive_given_disease * P_disease +
              P_positive_given_no_disease * P_no_disease)

# 베이즈 정리: 양성일 때 실제 병이 있을 확률
P_disease_given_positive = (P_positive_given_disease * P_disease) / P_positive

print("의료 진단 예제 (베이즈 정리)")
print(f"질병 유병률 (사전 확률): {P_disease:.1%}")
print(f"검사 민감도: {P_positive_given_disease:.1%}")
print(f"검사 특이도: {P_negative_given_no_disease:.1%}")
print(f"\n양성 검사 확률 (전확률): {P_positive:.4f}")
print(f"양성일 때 실제 병이 있을 확률 (사후 확률): {P_disease_given_positive:.1%}")
print(f"\n해석: 검사가 양성이어도 실제 병이 있을 확률은 {P_disease_given_positive:.1%}에 불과")
print("       (낮은 유병률로 인해 위양성이 많음)")

# 시각화: 베이즈 정리
fig, ax = plt.subplots(figsize=(12, 6))

categories = ['사전 확률\n(병 있음)', '우도\n(양성|병)', '사후 확률\n(병|양성)']
probabilities = [P_disease, P_positive_given_disease, P_disease_given_positive]
colors = ['skyblue', 'lightgreen', 'salmon']

bars = ax.bar(categories, probabilities, color=colors, edgecolor='black', linewidth=2)

# 값 표시
for bar, prob in zip(bars, probabilities):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{prob:.1%}', ha='center', va='bottom', fontsize=14, fontweight='bold')

ax.set_ylabel('확률', fontsize=13)
ax.set_title('베이즈 정리: 의료 진단 예제', fontsize=15)
ax.set_ylim(0, 1.0)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('bayes_theorem_medical.png', dpi=150)
plt.show()
```

## 2. Random Variables

### 2.1 Definition of Random Variables

**Random Variable**: a function from the sample space to real numbers
$$X: \Omega \to \mathbb{R}$$

**Discrete random variable**: takes countable values (e.g., dice, coins)
**Continuous random variable**: takes continuous values (e.g., height, temperature)

### 2.2 Probability Mass Function (PMF)

For a discrete random variable $X$:

$$
p_X(x) = P(X = x)
$$

**Properties:**
- $p_X(x) \geq 0$ for all $x$
- $\sum_{x} p_X(x) = 1$

### 2.3 Probability Density Function (PDF)

For a continuous random variable $X$:

$$
P(a \leq X \leq b) = \int_a^b f_X(x) dx
$$

**Properties:**
- $f_X(x) \geq 0$ for all $x$
- $\int_{-\infty}^{\infty} f_X(x) dx = 1$
- $P(X = x) = 0$ (probability at a single point is 0)

### 2.4 Cumulative Distribution Function (CDF)

$$
F_X(x) = P(X \leq x)
$$

**Properties:**
- Non-decreasing function
- $\lim_{x \to -\infty} F_X(x) = 0$, $\lim_{x \to \infty} F_X(x) = 1$
- For continuous random variables: $f_X(x) = \frac{d}{dx}F_X(x)$

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. 이산: 이항 분포
n, p = 10, 0.5
x_binom = np.arange(0, n+1)
pmf_binom = stats.binom.pmf(x_binom, n, p)
cdf_binom = stats.binom.cdf(x_binom, n, p)

axes[0, 0].bar(x_binom, pmf_binom, color='skyblue', edgecolor='black')
axes[0, 0].set_title('이항 분포 PMF\n$n=10, p=0.5$', fontsize=12)
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('P(X=x)')
axes[0, 0].grid(True, alpha=0.3)

axes[1, 0].step(x_binom, cdf_binom, where='post', linewidth=2, color='blue')
axes[1, 0].set_title('이항 분포 CDF', fontsize=12)
axes[1, 0].set_xlabel('x')
axes[1, 0].set_ylabel('P(X≤x)')
axes[1, 0].grid(True, alpha=0.3)

# 2. 연속: 정규 분포
mu, sigma = 0, 1
x_norm = np.linspace(-4, 4, 1000)
pdf_norm = stats.norm.pdf(x_norm, mu, sigma)
cdf_norm = stats.norm.cdf(x_norm, mu, sigma)

axes[0, 1].plot(x_norm, pdf_norm, linewidth=2, color='red')
axes[0, 1].fill_between(x_norm, pdf_norm, alpha=0.3, color='red')
axes[0, 1].set_title('정규 분포 PDF\n$\mu=0, \sigma=1$', fontsize=12)
axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel('f(x)')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 1].plot(x_norm, cdf_norm, linewidth=2, color='darkred')
axes[1, 1].set_title('정규 분포 CDF', fontsize=12)
axes[1, 1].set_xlabel('x')
axes[1, 1].set_ylabel('F(x)')
axes[1, 1].grid(True, alpha=0.3)

# 3. 연속: 지수 분포
lam = 1.0
x_exp = np.linspace(0, 5, 1000)
pdf_exp = stats.expon.pdf(x_exp, scale=1/lam)
cdf_exp = stats.expon.cdf(x_exp, scale=1/lam)

axes[0, 2].plot(x_exp, pdf_exp, linewidth=2, color='green')
axes[0, 2].fill_between(x_exp, pdf_exp, alpha=0.3, color='green')
axes[0, 2].set_title('지수 분포 PDF\n$\lambda=1$', fontsize=12)
axes[0, 2].set_xlabel('x')
axes[0, 2].set_ylabel('f(x)')
axes[0, 2].grid(True, alpha=0.3)

axes[1, 2].plot(x_exp, cdf_exp, linewidth=2, color='darkgreen')
axes[1, 2].set_title('지수 분포 CDF', fontsize=12)
axes[1, 2].set_xlabel('x')
axes[1, 2].set_ylabel('F(x)')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pmf_pdf_cdf.png', dpi=150)
plt.show()

print("PMF vs PDF:")
print("  PMF (이산): 특정 값의 확률 P(X=x)")
print("  PDF (연속): 확률 밀도, 구간 확률은 적분으로 계산")
print("  CDF: 누적 확률 P(X≤x), 이산/연속 모두 정의")
```

### 2.5 Joint, Marginal, and Conditional Distributions

**Joint Distribution:**
$$P(X = x, Y = y)$$ or $$f_{X,Y}(x, y)$$

**Marginal Distribution:**
$$p_X(x) = \sum_y p_{X,Y}(x, y)$$ or $$f_X(x) = \int f_{X,Y}(x, y) dy$$

**Conditional Distribution:**
$$p_{X|Y}(x|y) = \frac{p_{X,Y}(x, y)}{p_Y(y)}$$

```python
# 결합 분포 예제: 이변량 정규분포
from scipy.stats import multivariate_normal

# 파라미터
mu = np.array([0, 0])
cov = np.array([[1, 0.7],
                [0.7, 1]])

# 그리드
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

# 결합 PDF
rv = multivariate_normal(mu, cov)
Z = rv.pdf(pos)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 등고선
ax = axes[0]
contour = ax.contourf(X, Y, Z, levels=15, cmap='viridis')
plt.colorbar(contour, ax=ax)
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_title('결합 분포 $f_{X,Y}(x,y)$ (이변량 정규)', fontsize=14)
ax.grid(True, alpha=0.3)

# 주변 분포
ax = axes[1]
marginal_X = stats.norm.pdf(x, mu[0], np.sqrt(cov[0, 0]))
marginal_Y = stats.norm.pdf(y, mu[1], np.sqrt(cov[1, 1]))
ax.plot(x, marginal_X, linewidth=3, label='주변 분포 $f_X(x)$', color='blue')
ax.plot(y, marginal_Y, linewidth=3, label='주변 분포 $f_Y(y)$', color='red')
ax.set_xlabel('값', fontsize=12)
ax.set_ylabel('밀도', fontsize=12)
ax.set_title('주변 분포', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('joint_marginal_distributions.png', dpi=150)
plt.show()

print(f"공분산 행렬:\n{cov}")
print(f"상관계수: {cov[0,1] / np.sqrt(cov[0,0] * cov[1,1]):.2f}")
```

## 3. Expectation and Variance

### 3.1 Expectation

**Discrete:**
$$\mathbb{E}[X] = \sum_x x \cdot p_X(x)$$

**Continuous:**
$$\mathbb{E}[X] = \int_{-\infty}^{\infty} x \cdot f_X(x) dx$$

**Expectation of a function (LOTUS - Law of the Unconscious Statistician):**
$$\mathbb{E}[g(X)] = \sum_x g(x) \cdot p_X(x) \quad \text{or} \quad \int g(x) \cdot f_X(x) dx$$

### 3.2 Properties of Expectation

1. **Linearity:**
   $$\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]$$

2. **Product of independent variables:**
   If $X, Y$ independent then $\mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y]$

### 3.3 Variance

$$
\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2
$$

**Standard Deviation:**
$$\sigma_X = \sqrt{\text{Var}(X)}$$

**Properties of variance:**
- $\text{Var}(aX + b) = a^2 \text{Var}(X)$
- If $X, Y$ independent then $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$

### 3.4 Covariance

$$
\text{Cov}(X, Y) = \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])] = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]
$$

**Correlation Coefficient:**
$$
\rho_{X,Y} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y} \in [-1, 1]
$$

```python
import numpy as np
import matplotlib.pyplot as plt

# 몬테카를로로 기댓값과 분산 추정
np.random.seed(42)

# 정규 분포 샘플링
mu, sigma = 2, 1.5
samples = np.random.normal(mu, sigma, 10000)

# 기댓값과 분산 추정
estimated_mean = np.mean(samples)
estimated_var = np.var(samples, ddof=0)
estimated_std = np.std(samples, ddof=0)

print("몬테카를로 추정")
print(f"이론적 평균: {mu}, 추정 평균: {estimated_mean:.4f}")
print(f"이론적 분산: {sigma**2}, 추정 분산: {estimated_var:.4f}")
print(f"이론적 표준편차: {sigma}, 추정 표준편차: {estimated_std:.4f}")

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 히스토그램 + 이론적 PDF
ax = axes[0]
ax.hist(samples, bins=50, density=True, alpha=0.7, color='skyblue',
        edgecolor='black', label='샘플 히스토그램')
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
pdf = stats.norm.pdf(x, mu, sigma)
ax.plot(x, pdf, linewidth=3, color='red', label='이론적 PDF')
ax.axvline(estimated_mean, color='green', linestyle='--', linewidth=2,
           label=f'추정 평균 = {estimated_mean:.2f}')
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('밀도', fontsize=12)
ax.set_title(f'정규 분포 샘플링 (μ={mu}, σ={sigma})', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# 샘플 크기에 따른 수렴
ax = axes[1]
sample_sizes = np.arange(10, 10001, 10)
running_means = [np.mean(samples[:n]) for n in sample_sizes]

ax.plot(sample_sizes, running_means, linewidth=2, color='blue',
        label='누적 평균')
ax.axhline(mu, color='red', linestyle='--', linewidth=2, label=f'이론적 평균 = {mu}')
ax.set_xlabel('샘플 크기', fontsize=12)
ax.set_ylabel('누적 평균', fontsize=12)
ax.set_title('대수의 법칙 (Law of Large Numbers)', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('expectation_variance_estimation.png', dpi=150)
plt.show()

# 공분산 예제
np.random.seed(42)
n = 1000

# 양의 상관관계
X1 = np.random.randn(n)
Y1 = 0.8 * X1 + 0.3 * np.random.randn(n)

# 음의 상관관계
X2 = np.random.randn(n)
Y2 = -0.8 * X2 + 0.3 * np.random.randn(n)

# 독립
X3 = np.random.randn(n)
Y3 = np.random.randn(n)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

datasets = [(X1, Y1, '양의 상관'), (X2, Y2, '음의 상관'), (X3, Y3, '독립 (상관 없음)')]
for idx, (X, Y, title) in enumerate(datasets):
    ax = axes[idx]
    ax.scatter(X, Y, alpha=0.5, s=20, edgecolors='k', linewidths=0.5)

    # 통계량 계산
    cov = np.cov(X, Y)[0, 1]
    corr = np.corrcoef(X, Y)[0, 1]

    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title(f'{title}\nCov={cov:.3f}, ρ={corr:.3f}', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('covariance_correlation.png', dpi=150)
plt.show()

print("\n공분산과 상관계수:")
print("  Cov > 0: 양의 관계 (X 증가 → Y 증가)")
print("  Cov < 0: 음의 관계 (X 증가 → Y 감소)")
print("  Cov = 0: 선형 관계 없음 (독립이면 Cov=0, 역은 성립 안 함)")
print("  ρ ∈ [-1, 1]: 정규화된 공분산 (단위 무관)")
```

## 4. Common Probability Distributions

### 4.1 Discrete Distributions

**Bernoulli Distribution:**
$$X \sim \text{Ber}(p), \quad P(X=1) = p, \; P(X=0) = 1-p$$
- Mean: $p$, Variance: $p(1-p)$

**Binomial Distribution:**
$$X \sim \text{Bin}(n, p), \quad P(X=k) = \binom{n}{k}p^k(1-p)^{n-k}$$
- Number of successes in $n$ independent Bernoulli trials
- Mean: $np$, Variance: $np(1-p)$

**Poisson Distribution:**
$$X \sim \text{Pois}(\lambda), \quad P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}$$
- Number of events in unit time/space
- Mean: $\lambda$, Variance: $\lambda$

**Categorical Distribution:**
$$X \sim \text{Cat}(p_1, \ldots, p_K), \quad P(X=k) = p_k, \; \sum p_k = 1$$
- Basic distribution for multiclass classification

### 4.2 Continuous Distributions

**Uniform Distribution:**
$$X \sim \text{Unif}(a, b), \quad f(x) = \frac{1}{b-a} \text{ for } x \in [a, b]$$
- Mean: $\frac{a+b}{2}$, Variance: $\frac{(b-a)^2}{12}$

**Normal/Gaussian Distribution:**
$$X \sim \mathcal{N}(\mu, \sigma^2), \quad f(x) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$
- Mean: $\mu$, Variance: $\sigma^2$
- Arises naturally via Central Limit Theorem

**Exponential Distribution:**
$$X \sim \text{Exp}(\lambda), \quad f(x) = \lambda e^{-\lambda x} \text{ for } x \geq 0$$
- Waiting time in Poisson process
- Mean: $1/\lambda$, Variance: $1/\lambda^2$

**Beta Distribution:**
$$X \sim \text{Beta}(\alpha, \beta), \quad f(x) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)} \text{ for } x \in [0, 1]$$
- Distribution of probabilities (prior in Bayesian inference)

```python
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 3, figsize=(18, 15))

# 1. 베르누이
ax = axes[0, 0]
p = 0.7
x = [0, 1]
pmf = [1-p, p]
ax.bar(x, pmf, color='skyblue', edgecolor='black', width=0.4)
ax.set_title(f'베르누이 (p={p})', fontsize=12)
ax.set_xticks([0, 1])
ax.set_ylabel('P(X=x)')

# 2. 이항
ax = axes[0, 1]
n, p = 20, 0.5
x = np.arange(0, n+1)
pmf = stats.binom.pmf(x, n, p)
ax.bar(x, pmf, color='lightgreen', edgecolor='black')
ax.set_title(f'이항 (n={n}, p={p})', fontsize=12)
ax.set_xlabel('x')

# 3. 포아송
ax = axes[0, 2]
lam = 5
x = np.arange(0, 20)
pmf = stats.poisson.pmf(x, lam)
ax.bar(x, pmf, color='salmon', edgecolor='black')
ax.set_title(f'포아송 (λ={lam})', fontsize=12)
ax.set_xlabel('x')

# 4. 균등
ax = axes[1, 0]
a, b = 0, 1
x = np.linspace(-0.5, 1.5, 1000)
pdf = stats.uniform.pdf(x, a, b-a)
ax.plot(x, pdf, linewidth=3, color='blue')
ax.fill_between(x, pdf, alpha=0.3, color='blue')
ax.set_title(f'균등 (a={a}, b={b})', fontsize=12)
ax.set_ylabel('f(x)')

# 5. 정규 (여러 파라미터)
ax = axes[1, 1]
x = np.linspace(-5, 5, 1000)
params = [(0, 1), (0, 0.5), (1, 1)]
for mu, sigma in params:
    pdf = stats.norm.pdf(x, mu, sigma)
    ax.plot(x, pdf, linewidth=2, label=f'μ={mu}, σ={sigma}')
ax.set_title('정규 분포', fontsize=12)
ax.legend(fontsize=9)

# 6. 지수
ax = axes[1, 2]
x = np.linspace(0, 5, 1000)
lambdas = [0.5, 1, 2]
for lam in lambdas:
    pdf = stats.expon.pdf(x, scale=1/lam)
    ax.plot(x, pdf, linewidth=2, label=f'λ={lam}')
ax.set_title('지수 분포', fontsize=12)
ax.legend(fontsize=9)

# 7. 감마
ax = axes[2, 0]
x = np.linspace(0, 20, 1000)
params = [(1, 1), (2, 2), (5, 1)]
for k, theta in params:
    pdf = stats.gamma.pdf(x, k, scale=theta)
    ax.plot(x, pdf, linewidth=2, label=f'k={k}, θ={theta}')
ax.set_title('감마 분포', fontsize=12)
ax.set_ylabel('f(x)')
ax.legend(fontsize=9)

# 8. 베타
ax = axes[2, 1]
x = np.linspace(0, 1, 1000)
params = [(0.5, 0.5), (2, 2), (5, 2)]
for alpha, beta in params:
    pdf = stats.beta.pdf(x, alpha, beta)
    ax.plot(x, pdf, linewidth=2, label=f'α={alpha}, β={beta}')
ax.set_title('베타 분포', fontsize=12)
ax.legend(fontsize=9)

# 9. 카이제곱
ax = axes[2, 2]
x = np.linspace(0, 15, 1000)
dfs = [2, 4, 6]
for df in dfs:
    pdf = stats.chi2.pdf(x, df)
    ax.plot(x, pdf, linewidth=2, label=f'df={df}')
ax.set_title('카이제곱 분포', fontsize=12)
ax.legend(fontsize=9)

for ax in axes.flat:
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('common_distributions.png', dpi=150)
plt.show()

print("머신러닝에서의 활용:")
print("  베르누이/이항: 이진 분류")
print("  카테고리컬: 다항 분류")
print("  정규: 연속 데이터, 오차 모델, VAE 잠재 공간")
print("  포아송: 카운트 데이터 (추천 시스템, 웹 트래픽)")
print("  베타: 베이지안 추론의 사전분포")
print("  지수/감마: 대기 시간, 생존 분석")
```

### 4.3 Multivariate Normal Distribution

$$
\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma}), \quad f(\mathbf{x}) = \frac{1}{\sqrt{(2\pi)^d |\boldsymbol{\Sigma}|}}\exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)
$$

- $\boldsymbol{\mu} \in \mathbb{R}^d$: mean vector
- $\boldsymbol{\Sigma} \in \mathbb{R}^{d \times d}$: covariance matrix (positive definite)

```python
from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt

# 다변량 정규분포 시각화
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

mu = np.array([0, 0])
covs = [
    np.array([[1, 0], [0, 1]]),      # 독립
    np.array([[1, 0.8], [0.8, 1]]),  # 양의 상관
    np.array([[1, -0.8], [-0.8, 1]]) # 음의 상관
]
titles = ['독립 (ρ=0)', '양의 상관 (ρ=0.8)', '음의 상관 (ρ=-0.8)']

x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

for ax, cov, title in zip(axes, covs, titles):
    rv = multivariate_normal(mu, cov)
    Z = rv.pdf(pos)

    contour = ax.contourf(X, Y, Z, levels=15, cmap='viridis')
    ax.contour(X, Y, Z, levels=15, colors='white', alpha=0.3, linewidths=0.5)
    ax.set_xlabel('$X_1$', fontsize=12)
    ax.set_ylabel('$X_2$', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('multivariate_normal.png', dpi=150)
plt.show()

print("다변량 정규분포:")
print("  - 고차원 데이터 모델링의 기본")
print("  - 가우시안 프로세스, GMM, VAE 등에서 핵심")
print("  - 공분산 행렬로 변수 간 의존성 표현")
```

## 5. Advanced Bayes' Theorem

### 5.1 Bayesian Update

**Prior** → **Data** → **Posterior**

$$
P(\theta | D) = \frac{P(D | \theta) P(\theta)}{P(D)} \propto P(D | \theta) P(\theta)
$$

- $\theta$: parameter (treated as random variable)
- $D$: observed data
- $P(\theta)$: prior probability (belief before data)
- $P(D | \theta)$: likelihood (plausibility of data given parameter)
- $P(\theta | D)$: posterior probability (updated belief after data)

### 5.2 Example: Coin Flip (Beta-Binomial Model)

```python
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

# 베타-이항 모델: 동전의 앞면 확률 추정
# 사전분포: Beta(α, β)
# 우도: Binomial
# 사후분포: Beta(α + n_heads, β + n_tails)

np.random.seed(42)

# 진짜 동전 확률 (알 수 없다고 가정)
true_p = 0.7

# 사전분포 (균등 사전: Beta(1, 1))
alpha_prior, beta_prior = 1, 1

# 동전 던지기 시뮬레이션
n_flips_list = [0, 1, 5, 20, 100]
data = np.random.binomial(1, true_p, 100)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

p_vals = np.linspace(0, 1, 1000)

for idx, n_flips in enumerate(n_flips_list):
    ax = axes[idx]

    if n_flips == 0:
        # 사전분포만
        prior_pdf = stats.beta.pdf(p_vals, alpha_prior, beta_prior)
        ax.plot(p_vals, prior_pdf, linewidth=3, color='blue', label='사전분포')
    else:
        # 데이터
        observed_data = data[:n_flips]
        n_heads = np.sum(observed_data)
        n_tails = n_flips - n_heads

        # 사후분포
        alpha_post = alpha_prior + n_heads
        beta_post = beta_prior + n_tails
        posterior_pdf = stats.beta.pdf(p_vals, alpha_post, beta_post)

        # 사전분포
        prior_pdf = stats.beta.pdf(p_vals, alpha_prior, beta_prior)

        ax.plot(p_vals, prior_pdf, linewidth=2, color='blue', linestyle='--',
                label='사전분포', alpha=0.7)
        ax.plot(p_vals, posterior_pdf, linewidth=3, color='red', label='사후분포')

        # MAP 추정 (최대 사후 확률)
        map_estimate = (alpha_post - 1) / (alpha_post + beta_post - 2)
        ax.axvline(map_estimate, color='red', linestyle=':', linewidth=2,
                   label=f'MAP = {map_estimate:.3f}')

    # 진짜 확률
    ax.axvline(true_p, color='green', linestyle='--', linewidth=2,
               label=f'진짜 p = {true_p}')

    ax.set_xlabel('p (앞면 확률)', fontsize=11)
    ax.set_ylabel('밀도', fontsize=11)
    ax.set_title(f'동전 {n_flips}번 던진 후' if n_flips > 0 else '사전분포', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

# 수렴 곡선
ax = axes[-1]
n_range = np.arange(1, 101)
map_estimates = []
for n in n_range:
    n_heads = np.sum(data[:n])
    n_tails = n - n_heads
    alpha_post = alpha_prior + n_heads
    beta_post = beta_prior + n_tails
    map_est = (alpha_post - 1) / (alpha_post + beta_post - 2)
    map_estimates.append(map_est)

ax.plot(n_range, map_estimates, linewidth=2, color='red', label='MAP 추정')
ax.axhline(true_p, color='green', linestyle='--', linewidth=2, label=f'진짜 p = {true_p}')
ax.set_xlabel('동전 던진 횟수', fontsize=11)
ax.set_ylabel('추정된 p', fontsize=11)
ax.set_title('베이지안 학습 수렴', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('bayesian_update_coin.png', dpi=150)
plt.show()

print("베이지안 업데이트:")
print("  - 데이터가 늘어날수록 사후분포가 진짜 값 주변에 집중")
print("  - 사전분포의 영향은 데이터가 많아지면 감소")
print("  - 불확실성을 분포로 표현 (점 추정이 아님)")
```

## 6. Probability in Machine Learning

### 6.1 Generative vs Discriminative Models

**Generative Model:**
- Models $P(X, Y) = P(Y)P(X|Y)$
- Learns data distribution per class
- Prediction: $P(Y|X) = \frac{P(X|Y)P(Y)}{P(X)}$ via Bayes' theorem
- Examples: Naive Bayes, GMM, VAE, GAN

**Discriminative Model:**
- Directly models $P(Y|X)$
- Learns only decision boundary
- Examples: Logistic regression, SVM, neural networks

```python
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# 데이터 생성
X, y = make_classification(n_samples=300, n_features=2, n_informative=2,
                           n_redundant=0, n_clusters_per_class=1,
                           random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 생성 모델: 나이브 베이즈
generative_model = GaussianNB()
generative_model.fit(X_train, y_train)

# 판별 모델: 로지스틱 회귀
discriminative_model = LogisticRegression()
discriminative_model.fit(X_train, y_train)

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 그리드
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

models = [
    (generative_model, '생성 모델 (나이브 베이즈)', axes[0]),
    (discriminative_model, '판별 모델 (로지스틱 회귀)', axes[1])
]

for model, title, ax in models:
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    ax.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1],
               c='blue', marker='o', s=50, edgecolors='k', label='Class 0')
    ax.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1],
               c='red', marker='s', s=50, edgecolors='k', label='Class 1')

    score = model.score(X_test, y_test)
    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)
    ax.set_title(f'{title}\nTest Accuracy: {score:.3f}', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('generative_vs_discriminative.png', dpi=150)
plt.show()

print("생성 vs 판별:")
print("  생성: P(X,Y) 전체 분포 모델링 → 샘플 생성 가능")
print("  판별: P(Y|X) 조건부만 → 예측만 가능, 보통 더 높은 성능")
```

### 6.2 Naive Bayes Classifier

**Assumption**: features are conditionally independent given the class

$$
P(X_1, \ldots, X_d | Y) = \prod_{i=1}^d P(X_i | Y)
$$

**Prediction:**
$$
\hat{y} = \arg\max_y P(Y=y) \prod_{i=1}^d P(X_i | Y=y)
$$

```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# 텍스트 분류 예제
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
train = fetch_20newsgroups(subset='train', categories=categories, random_state=42)
test = fetch_20newsgroups(subset='test', categories=categories, random_state=42)

# 특징 추출 (Bag-of-Words)
vectorizer = CountVectorizer(max_features=1000)
X_train = vectorizer.fit_transform(train.data)
X_test = vectorizer.transform(test.data)

# 나이브 베이즈 학습
nb_model = MultinomialNB()
nb_model.fit(X_train, train.target)

# 예측
y_pred = nb_model.predict(X_test)

print("나이브 베이즈 텍스트 분류:")
print(classification_report(test.target, y_pred, target_names=test.target_names))

print("\n나이브 베이즈의 특징:")
print("  - 조건부 독립 가정 (naive) → 계산 효율적")
print("  - 고차원 데이터에서도 잘 작동 (텍스트 분류)")
print("  - 확률적 해석 가능")
print("  - 작은 데이터셋에서도 합리적 성능")
```

### 6.3 Probabilistic Graphical Models

- **Bayesian Network**: represents conditional independence with directed acyclic graph (DAG)
- **Markov Random Field**: undirected graph
- **Hidden Markov Model (HMM)**: inference of hidden states in time series data
- **Applications**: speech recognition, natural language processing, computer vision

```python
# 간단한 베이지안 네트워크 예제 (개념적)
import networkx as nx
import matplotlib.pyplot as plt

# 베이지안 네트워크 구조
# Rain → Sprinkler, Rain → Grass Wet, Sprinkler → Grass Wet
G = nx.DiGraph()
G.add_edges_from([('Rain', 'Sprinkler'), ('Rain', 'Grass Wet'),
                  ('Sprinkler', 'Grass Wet')])

plt.figure(figsize=(10, 6))
pos = {'Rain': (0.5, 1), 'Sprinkler': (0, 0), 'Grass Wet': (1, 0)}
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue',
        font_size=12, font_weight='bold', arrowsize=20, arrows=True)
plt.title('베이지안 네트워크: 비 → 스프링클러, 잔디 젖음', fontsize=14)
plt.tight_layout()
plt.savefig('bayesian_network_example.png', dpi=150)
plt.show()

print("확률적 그래프 모델:")
print("  - 변수 간 의존성을 그래프로 표현")
print("  - 조건부 독립성으로 계산 효율화")
print("  - 추론: 관측된 변수로 숨겨진 변수 추정")
print("  - 학습: 데이터로부터 그래프 구조와 확률 파라미터 학습")
```

## Practice Problems

1. **Bayes' Theorem Application**: Design a spam filter using Bayes' theorem. Derive the formula for calculating spam probability given the presence of specific words, and implement with simple example data.

2. **Distribution Fitting**: Use `scipy.stats` to fit a normal distribution to real data (e.g., height, test scores) and verify goodness-of-fit with Q-Q plot. If normal distribution is inadequate, try other distributions.

3. **Monte Carlo Integration**: For $X \sim \mathcal{N}(0, 1)$, compute $\mathbb{E}[e^X]$ (1) analytically and (2) estimate via Monte Carlo sampling. Verify convergence as sample size increases.

4. **Bayesian Linear Regression**: Implement linear regression from a Bayesian perspective. Assign normal prior to weights and update posterior with each data observation. Visualize posterior mean and uncertainty.

5. **Naive Bayes vs Logistic Regression**: Compare performance of Naive Bayes (generative) and logistic regression (discriminative) on Iris dataset. Plot learning curves as training data size varies. Analyze which model is advantageous in which situations.

## References

- Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press
  - The bible of ML from a probabilistic viewpoint
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer
  - Chapter 1-2: Probability foundations and distributions
- Wasserman, L. (2004). *All of Statistics*. Springer
  - Concise summary of statistics and probability
- Koller, D., & Friedman, N. (2009). *Probabilistic Graphical Models*. MIT Press
  - Comprehensive textbook on probabilistic graphical models
- SciPy Stats Documentation: https://docs.scipy.org/doc/scipy/reference/stats.html
- Seeing Theory (probability/statistics visualization): https://seeing-theory.brown.edu/
- Bayesian Methods for Hackers (online book): https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers
