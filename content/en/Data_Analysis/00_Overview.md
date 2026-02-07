# Data Analysis Study Guide

## Overview

Data analysis is the process of extracting meaningful information and deriving insights from data. This learning material systematically covers Python-based data analysis tools and techniques.

---

## Learning Roadmap

```
NumPy Basics → NumPy Advanced → Pandas Basics → Pandas Data Manipulation → Pandas Advanced
                                              ↓
Practical Projects ← Statistical Analysis ← Advanced Visualization ← Basic Visualization ← Descriptive Stats/EDA ← Data Preprocessing
```

---

## File List

| File | Topic | Key Content |
|------|------|----------|
| [01_NumPy_Basics.md](./01_NumPy_Basics.md) | NumPy Basics | Array creation, indexing/slicing, broadcasting, basic operations |
| [02_NumPy_Advanced.md](./02_NumPy_Advanced.md) | NumPy Advanced | Linear algebra, statistical functions, random number generation, performance optimization |
| [03_Pandas_Basics.md](./03_Pandas_Basics.md) | Pandas Basics | Series, DataFrame, data loading (CSV/Excel/JSON) |
| [04_Pandas_Data_Manipulation.md](./04_Pandas_Data_Manipulation.md) | Pandas Data Manipulation | Filtering, sorting, grouping, merging (merge/join/concat) |
| [05_Pandas_Advanced.md](./05_Pandas_Advanced.md) | Pandas Advanced | Pivot tables, multi-index, time series data processing |
| [06_Data_Preprocessing.md](./06_Data_Preprocessing.md) | Data Preprocessing | Missing value handling, outlier detection, normalization, encoding |
| [07_Descriptive_Stats_EDA.md](./07_Descriptive_Stats_EDA.md) | Descriptive Statistics and EDA | Descriptive statistics, distribution analysis, exploratory data analysis |
| [08_Data_Visualization_Basics.md](./08_Data_Visualization_Basics.md) | Data Visualization Basics | Matplotlib basics, various chart types |
| [09_Data_Visualization_Advanced.md](./09_Data_Visualization_Advanced.md) | Data Visualization Advanced | Seaborn, advanced visualization techniques, dashboards |
| [10_Statistical_Analysis_Basics.md](./10_Statistical_Analysis_Basics.md) | Statistical Analysis Basics | Probability distributions, hypothesis testing, confidence intervals, correlation analysis |
| [11_Practical_Projects.md](./11_Practical_Projects.md) | Practical Projects | Kaggle dataset EDA, comprehensive exercises |

---

## Environment Setup

### Installing Required Libraries

```bash
# Using pip
pip install numpy pandas matplotlib seaborn scipy

# Using conda
conda install numpy pandas matplotlib seaborn scipy

# Jupyter Notebook (recommended)
pip install jupyter
jupyter notebook
```

### Version Check

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print(f"NumPy: {np.__version__}")
print(f"Pandas: {pd.__version__}")
```

### Recommended Versions
- Python: 3.9+
- NumPy: 1.21+
- Pandas: 1.5+
- Matplotlib: 3.5+
- Seaborn: 0.12+

---

## Recommended Learning Path

### Stage 1: NumPy Basics (01-02)
- Fundamentals of array operations
- Understanding vectorized operations

### Stage 2: Pandas Basics (03-05)
- Understanding data structures
- Data manipulation capabilities

### Stage 3: Data Preprocessing and EDA (06-07)
- Most commonly used in practice
- Data quality management

### Stage 4: Visualization (08-09)
- Visual representation of data
- Communicating insights

### Stage 5: Statistics and Practice (10-11)
- Statistical analysis
- Comprehensive projects

---

## References

### Official Documentation
- [NumPy Documentation](https://numpy.org/doc/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Seaborn Documentation](https://seaborn.pydata.org/)

### Recommended Datasets
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/)
- [Korea Public Data Portal](https://www.data.go.kr/)
