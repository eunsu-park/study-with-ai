# Advanced Statistics - Overview

## Introduction

This folder is a collection of materials for systematically learning **advanced statistics**. It is designed for learners who have completed basic statistics (descriptive statistics, basic hypothesis testing) and provides step-by-step advanced learning from the theoretical foundations of probability theory to generalized linear models.

All concepts are provided with Python code examples and contain practical content that can be directly applied in real-world settings.

---

## File List

| Number | Filename | Topic | Key Content |
|--------|----------|-------|-------------|
| 00 | [00_Overview.md](./00_Overview.md) | Learning Roadmap | Overall structure, required libraries, learning sequence |
| 01 | [01_Probability_Review.md](./01_Probability_Review.md) | Probability Review | Probability axioms, random variables, distributions, expected values, Central Limit Theorem |
| 02 | [02_Sampling_and_Estimation.md](./02_Sampling_and_Estimation.md) | Sampling and Estimation | Sampling distributions, standard error, point estimation, MLE, method of moments |
| 03 | [03_Confidence_Intervals.md](./03_Confidence_Intervals.md) | Confidence Intervals | Interval estimation, t-distribution, proportion/variance confidence intervals, bootstrap |
| 04 | [04_Hypothesis_Testing_Advanced.md](./04_Hypothesis_Testing_Advanced.md) | Advanced Hypothesis Testing | Power, effect size, sample size determination, multiple testing correction |
| 05 | [05_ANOVA.md](./05_ANOVA.md) | Analysis of Variance | One-way/two-way ANOVA, F-distribution, post-hoc tests |
| 06 | [06_Regression_Analysis_Advanced.md](./06_Regression_Analysis_Advanced.md) | Advanced Regression Analysis | Multiple regression, diagnostics, multicollinearity, variable selection |
| 07 | [07_Generalized_Linear_Models.md](./07_Generalized_Linear_Models.md) | GLM | Logistic regression, Poisson regression, link functions |
| 08 | [08_Bayesian_Statistics_Basics.md](./08_Bayesian_Statistics_Basics.md) | Bayesian Basics | Bayes' theorem, prior/posterior distributions, conjugate priors |
| 09 | [09_Bayesian_Inference.md](./09_Bayesian_Inference.md) | Bayesian Inference | PyMC, MCMC, posterior estimation, model comparison |
| 10 | [10_Time_Series_Basics.md](./10_Time_Series_Basics.md) | Time Series Basics | Stationarity, ACF/PACF, time series decomposition |
| 11 | [11_Time_Series_Models.md](./11_Time_Series_Models.md) | Time Series Models | ARIMA, seasonality, forecasting, Prophet |
| 12 | [12_Multivariate_Analysis.md](./12_Multivariate_Analysis.md) | Multivariate Analysis | PCA, factor analysis, cluster analysis, discriminant analysis |
| 13 | [13_Nonparametric_Statistics.md](./13_Nonparametric_Statistics.md) | Nonparametric Statistics | Rank tests, Mann-Whitney, Kruskal-Wallis |
| 14 | [14_Experimental_Design.md](./14_Experimental_Design.md) | Experimental Design | A/B testing, factorial design, repeated measures |

---

## Required Libraries

### Core Libraries

```bash
# Basic installation
pip install numpy pandas scipy statsmodels

# Visualization
pip install matplotlib seaborn

# Advanced statistical analysis
pip install pingouin  # ANOVA, effect sizes, various tests

# Bayesian statistics (optional)
pip install pymc arviz  # Bayesian inference (Python 3.9+)
```

### Library Usage

| Library | Main Purpose | Example Use Cases |
|---------|-------------|-------------------|
| `scipy.stats` | Probability distributions, basic tests | Normal distribution, t-test, chi-square test |
| `statsmodels` | Regression analysis, GLM, time series | OLS, logistic regression, ANOVA |
| `pingouin` | Advanced tests, effect sizes | ANOVA, post-hoc tests, Cohen's d |
| `pymc` | Bayesian inference | MCMC, posterior distribution estimation |

### Environment Setup Example

```python
# Basic import setup
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

# Font settings for Korean (macOS)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# Font settings for Korean (Windows)
# plt.rcParams['font.family'] = 'Malgun Gothic'

# Visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# Suppress warnings (if needed)
import warnings
warnings.filterwarnings('ignore')
```

---

## Recommended Learning Sequence

### Phase 1: Theoretical Foundations (1-2 weeks)

```
01_Probability_Review → 02_Sampling_and_Estimation → 03_Confidence_Intervals
```

- Review core concepts of probability theory
- Understand the mathematical foundations of estimation theory
- Learn the principles and interpretation of interval estimation

### Phase 2: Advanced Inferential Statistics (1-2 weeks)

```
04_Hypothesis_Testing_Advanced → 05_ANOVA
```

- Power analysis and sample size design
- Multiple testing problems and solutions
- ANOVA for group comparisons

### Phase 3: Regression Models (2-3 weeks)

```
06_Regression_Analysis_Advanced → 07_Generalized_Linear_Models
```

- Assumptions and diagnostics of regression models
- GLM for categorical/count data

### Phase 4: Bayesian Statistics (2 weeks)

```
08_Bayesian_Statistics_Basics → 09_Bayesian_Inference
```

- Understanding the Bayesian paradigm
- Bayesian inference using PyMC

### Phase 5: Time Series Analysis (2 weeks)

```
10_Time_Series_Basics → 11_Time_Series_Models
```

- Characteristics and decomposition of time series data
- ARIMA models and forecasting

### Phase 6: Advanced Topics (2-3 weeks)

```
12_Multivariate_Analysis → 13_Nonparametric_Statistics → 14_Experimental_Design
```

- Multivariate analysis techniques (PCA, factor analysis)
- Nonparametric testing methods
- Experimental design and A/B testing

### Learning Tips

1. **Learn sequentially**: Concepts from earlier lessons form the foundation for later lessons
2. **Execute code directly**: Type example code yourself rather than copying
3. **Modify data**: Transform example data to observe changes in results
4. **Use visualization**: Understand statistical concepts intuitively through graphs

---

## Prerequisites

### Required

- **Basic Statistics**
  - Descriptive statistics (mean, variance, standard deviation)
  - Basic hypothesis testing (t-test, chi-square test)
  - Correlation analysis, simple regression

- **Python Programming**
  - Basic syntax (variables, functions, conditionals, loops)
  - NumPy array operations
  - Pandas DataFrame manipulation
  - Basic Matplotlib visualization

### Recommended

- **Linear Algebra Basics**
  - Matrix operations, matrix inversion
  - Vector dot product

- **Calculus Basics**
  - Concept of differentiation (needed for optimization)
  - Concept of integration (needed for probability distributions)

---

## Learning Objectives

Upon completing this course, you will be able to:

1. **Understand probability foundations**: Explain the mathematical foundations of statistical methods
2. **Apply estimation theory**: Apply MLE and interval estimation to real data
3. **Design tests**: Determine appropriate sample sizes through power analysis
4. **Perform ANOVA**: Conduct analysis of variance for multiple group comparisons
5. **Diagnose regression**: Review and resolve regression model assumptions
6. **Apply GLM**: Select appropriate models for different types of dependent variables
7. **Bayesian inference**: Implement Bayesian models using PyMC
8. **Time series analysis**: Perform time series forecasting using ARIMA models
9. **Multivariate analysis**: Apply dimensionality reduction techniques like PCA and factor analysis
10. **Experimental design**: Design and analyze experiments including A/B tests

---

## References

### Textbooks
- Casella & Berger, *Statistical Inference* (theoretical depth)
- Agresti, *Foundations of Linear and Generalized Linear Models*
- James et al., *An Introduction to Statistical Learning* (practical)

### Online Resources
- [scipy.stats Documentation](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [statsmodels Documentation](https://www.statsmodels.org/stable/index.html)
- [pingouin Documentation](https://pingouin-stats.org/)

---

## Version Information

- **Initial Release**: 2026-01-29
- **Python Version**: 3.9+
- **Main Library Versions**: scipy >= 1.9, statsmodels >= 0.14, pingouin >= 0.5
