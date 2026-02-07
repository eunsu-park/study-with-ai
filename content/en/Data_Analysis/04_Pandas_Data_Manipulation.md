# Pandas Data Manipulation

## Overview

This covers core Pandas data manipulation techniques including filtering, sorting, grouping, and merging. These techniques are most commonly used in practical data analysis.

---

## 1. Data Filtering

### 1.1 Conditional Filtering

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'age': [25, 30, 35, 28, 32],
    'department': ['Sales', 'IT', 'IT', 'HR', 'Sales'],
    'salary': [50000, 60000, 70000, 55000, 65000]
})

# Single condition
print(df[df['age'] > 30])
print(df[df['department'] == 'IT'])

# Compound conditions (AND: &, OR: |) (AND: &, OR: |)
print(df[(df['age'] > 25) & (df['salary'] >= 60000)])
print(df[(df['department'] == 'IT') | (df['department'] == 'Sales')])

# NOT condition
print(df[~(df['department'] == 'HR')])

# Range condition
print(df[df['age'].between(25, 30)])  # 25 <= age <= 30
```

### 1.2 Filtering with isin

```python
# Match any of multiple values
departments = ['IT', 'Sales']
print(df[df['department'].isin(departments)])

# Not matching
print(df[~df['department'].isin(departments)])
```

### 1.3 String Conditions

```python
# Contains string
print(df[df['name'].str.contains('a', case=False)])

# Start/end strings
print(df[df['name'].str.startswith('A')])
print(df[df['name'].str.endswith('e')])

# Regular expression
print(df[df['name'].str.match(r'^[A-C]')])  # A, B, C로 시작
```

### 1.4 query Method

```python
# SQL-style query
print(df.query('age > 30'))
print(df.query('department == "IT"'))
print(df.query('age > 25 and salary >= 60000'))

# Use variables
min_age = 30
print(df.query('age >= @min_age'))

# Index reference
df_indexed = df.set_index('name')
print(df_indexed.query('index == "Alice"'))
```

### 1.5 Filtering Missing Values

```python
df_with_na = pd.DataFrame({
    'A': [1, 2, None, 4],
    'B': [None, 2, 3, 4],
    'C': [1, 2, 3, None]
})

# Rows with missing values
print(df_with_na[df_with_na['A'].isna()])

# Rows without missing values
print(df_with_na[df_with_na['A'].notna()])

# Rows with no missing in any column
print(df_with_na.dropna())

# Rows with no missing in specific columns
print(df_with_na.dropna(subset=['A', 'B']))
```

---

## 2. Sorting

### 2.1 Sort by Values

```python
df = pd.DataFrame({
    'name': ['Charlie', 'Alice', 'Bob', 'Diana'],
    'age': [35, 25, 30, 25],
    'score': [85, 95, 75, 90]
})

# Single column sort
print(df.sort_values('age'))
print(df.sort_values('age', ascending=False))

# Multiple column sort
print(df.sort_values(['age', 'score']))
print(df.sort_values(['age', 'score'], ascending=[True, False]))

# Missing value position
df_na = df.copy()
df_na.loc[0, 'age'] = None
print(df_na.sort_values('age', na_position='first'))  # 결측값 맨 앞
print(df_na.sort_values('age', na_position='last'))   # 결측값 맨 뒤

# In-place sorting
df.sort_values('age', inplace=True)
```

### 2.2 Sort by Index

```python
df = pd.DataFrame({
    'value': [10, 20, 30, 40]
}, index=['d', 'b', 'c', 'a'])

# Index ascending
print(df.sort_index())

# Index descending
print(df.sort_index(ascending=False))

# Sort column index
df_wide = pd.DataFrame({
    'C': [1, 2],
    'A': [3, 4],
    'B': [5, 6]
})
print(df_wide.sort_index(axis=1))
```

### 2.3 Ranking

```python
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'score': [85, 95, 85, 90]
})

# Default ranking (average for ties)
df['rank'] = df['score'].rank(ascending=False)
print(df)

# Ranking methods
# method='average': 평균 순위 (기본값)
# method='min': 최소 순위
# method='max': 최대 순위
# method='first': 먼저 나온 순서대로
# method='dense': 밀집 순위 (간격 없음)

df['rank_min'] = df['score'].rank(ascending=False, method='min')
df['rank_dense'] = df['score'].rank(ascending=False, method='dense')
print(df)
```

---

## 3. Grouping (GroupBy)

### 3.1 Basic Grouping

```python
df = pd.DataFrame({
    'department': ['Sales', 'IT', 'IT', 'HR', 'Sales', 'HR'],
    'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank'],
    'salary': [50000, 60000, 70000, 55000, 65000, 52000],
    'bonus': [5000, 8000, 10000, 6000, 7000, 5500]
})

# Grouping
grouped = df.groupby('department')

# Check groups
print(grouped.groups)
print(grouped.ngroups)  # 그룹 수

# Get specific group
print(grouped.get_group('IT'))
```

### 3.2 Aggregation Functions

```python
# Single aggregation
print(df.groupby('department')['salary'].mean())
print(df.groupby('department')['salary'].sum())

# Multiple column aggregation
print(df.groupby('department')[['salary', 'bonus']].mean())

# Multiple aggregation functions
print(df.groupby('department')['salary'].agg(['mean', 'sum', 'count']))

# Custom function
print(df.groupby('department')['salary'].agg(lambda x: x.max() - x.min()))
```

### 3.3 agg Method

```python
# Different aggregations per column
agg_result = df.groupby('department').agg({
    'salary': ['mean', 'max'],
    'bonus': 'sum',
    'name': 'count'
})
print(agg_result)

# Named aggregations
agg_result = df.groupby('department').agg(
    avg_salary=('salary', 'mean'),
    max_salary=('salary', 'max'),
    total_bonus=('bonus', 'sum'),
    employee_count=('name', 'count')
)
print(agg_result)
```

### 3.4 transform and apply

```python
# transform: result with same size as original
df['dept_avg_salary'] = df.groupby('department')['salary'].transform('mean')
print(df)

# Normalize within group
df['salary_normalized'] = df.groupby('department')['salary'].transform(
    lambda x: (x - x.mean()) / x.std()
)

# apply: flexible group operations
def top_n(group, n=2, column='salary'):
    return group.nlargest(n, column)

print(df.groupby('department').apply(top_n, n=1))
```

### 3.5 Group by Multiple Columns

```python
df = pd.DataFrame({
    'year': [2022, 2022, 2023, 2023, 2022, 2023],
    'quarter': ['Q1', 'Q2', 'Q1', 'Q2', 'Q1', 'Q1'],
    'sales': [100, 150, 120, 180, 110, 130]
})

# Multiple column grouping
print(df.groupby(['year', 'quarter'])['sales'].sum())

# Result as DataFrame
print(df.groupby(['year', 'quarter'])['sales'].sum().reset_index())

# Unstack
print(df.groupby(['year', 'quarter'])['sales'].sum().unstack())
```

### 3.6 Filtering

```python
df = pd.DataFrame({
    'department': ['Sales', 'IT', 'IT', 'HR', 'Sales'],
    'salary': [50000, 60000, 70000, 55000, 65000]
})

# Filter groups satisfying condition
# 평균 급여가 55000 이상인 부서
result = df.groupby('department').filter(lambda x: x['salary'].mean() >= 55000)
print(result)
```

---

## 4. Data Merging

### 4.1 merge (SQL-style Join)

```python
# Example data
employees = pd.DataFrame({
    'emp_id': [1, 2, 3, 4],
    'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'dept_id': [10, 20, 10, 30]
})

departments = pd.DataFrame({
    'dept_id': [10, 20, 40],
    'dept_name': ['Sales', 'IT', 'Marketing']
})

# Inner join (default)
result = pd.merge(employees, departments, on='dept_id')
print(result)

# Left join
result = pd.merge(employees, departments, on='dept_id', how='left')
print(result)

# Right join
result = pd.merge(employees, departments, on='dept_id', how='right')
print(result)

# Outer join
result = pd.merge(employees, departments, on='dept_id', how='outer')
print(result)
```

### 4.2 Join with Different Column Names

```python
employees = pd.DataFrame({
    'emp_id': [1, 2, 3],
    'name': ['Alice', 'Bob', 'Charlie'],
    'department': [10, 20, 10]
})

departments = pd.DataFrame({
    'id': [10, 20],
    'dept_name': ['Sales', 'IT']
})

# Join with different column names
result = pd.merge(employees, departments,
                  left_on='department', right_on='id')
print(result)
```

### 4.3 Index-based Join

```python
employees = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'salary': [50000, 60000, 70000]
}, index=[1, 2, 3])

bonuses = pd.DataFrame({
    'bonus': [5000, 8000, 10000]
}, index=[1, 2, 4])

# Join by index
result = pd.merge(employees, bonuses, left_index=True, right_index=True, how='outer')
print(result)

# join 메서드 (인덱스 기반)
result = employees.join(bonuses, how='outer')
print(result)
```

### 4.4 concat (Concatenation)

```python
df1 = pd.DataFrame({
    'A': [1, 2],
    'B': [3, 4]
})

df2 = pd.DataFrame({
    'A': [5, 6],
    'B': [7, 8]
})

# Vertical concatenation (row-wise)
result = pd.concat([df1, df2])
print(result)

# Reset index
result = pd.concat([df1, df2], ignore_index=True)
print(result)

# Horizontal concatenation (column-wise)
result = pd.concat([df1, df2], axis=1)
print(result)

# Add keys
result = pd.concat([df1, df2], keys=['first', 'second'])
print(result)
```

### 4.5 append (행 추가) - deprecated in pandas 2.0

```python
# concat 사용 권장
new_row = pd.DataFrame({'A': [9], 'B': [10]})
result = pd.concat([df1, new_row], ignore_index=True)
```

---

## 5. Pivot and Melt

### 5.1 pivot

```python
df = pd.DataFrame({
    'date': ['2023-01', '2023-01', '2023-02', '2023-02'],
    'city': ['Seoul', 'Busan', 'Seoul', 'Busan'],
    'sales': [100, 80, 120, 90]
})

# Pivot table
pivot = df.pivot(index='date', columns='city', values='sales')
print(pivot)
#          Busan  Seoul
# date
# 2023-01     80    100
# 2023-02     90    120
```

### 5.2 pivot_table

```python
df = pd.DataFrame({
    'date': ['2023-01', '2023-01', '2023-01', '2023-02'],
    'city': ['Seoul', 'Seoul', 'Busan', 'Seoul'],
    'category': ['A', 'B', 'A', 'A'],
    'sales': [100, 150, 80, 120]
})

# Apply aggregation function
pivot = pd.pivot_table(df, values='sales', index='date',
                       columns='city', aggfunc='sum')
print(pivot)

# Multiple aggregation functions
pivot = pd.pivot_table(df, values='sales', index='date',
                       columns='city', aggfunc=['sum', 'mean'])
print(pivot)

# Multiple indices
pivot = pd.pivot_table(df, values='sales',
                       index=['date', 'category'],
                       columns='city',
                       aggfunc='sum',
                       fill_value=0)
print(pivot)

# Add margins
pivot = pd.pivot_table(df, values='sales', index='date',
                       columns='city', aggfunc='sum', margins=True)
print(pivot)
```

### 5.3 melt (Unpivot)

```python
df_wide = pd.DataFrame({
    'name': ['Alice', 'Bob'],
    'math': [90, 85],
    'english': [80, 95],
    'science': [85, 90]
})

# Wide → Long 변환
df_long = pd.melt(df_wide,
                  id_vars=['name'],
                  value_vars=['math', 'english', 'science'],
                  var_name='subject',
                  value_name='score')
print(df_long)
#     name  subject  score
# 0  Alice     math     90
# 1    Bob     math     85
# 2  Alice  english     80
# 3    Bob  english     95
# 4  Alice  science     85
# 5    Bob  science     90
```

### 5.4 stack and unstack

```python
df = pd.DataFrame({
    'A': [1, 2],
    'B': [3, 4]
}, index=['x', 'y'])

# stack: columns to rows
stacked = df.stack()
print(stacked)
# x  A    1
#    B    3
# y  A    2
#    B    4

# unstack: rows to columns
unstacked = stacked.unstack()
print(unstacked)
```

---

## 6. Handling Duplicates

```python
df = pd.DataFrame({
    'A': [1, 1, 2, 2, 3],
    'B': ['a', 'a', 'b', 'c', 'c'],
    'C': [10, 10, 20, 30, 40]
})

# Check duplicates
print(df.duplicated())
print(df.duplicated(subset=['A', 'B']))
print(df[df.duplicated(keep=False)])  # 모든 Duplicates 행

# Count duplicates
print(df.duplicated().sum())

# Remove duplicates
print(df.drop_duplicates())
print(df.drop_duplicates(subset=['A']))
print(df.drop_duplicates(subset=['A'], keep='last'))  # 마지막 유지
print(df.drop_duplicates(subset=['A'], keep=False))   # 모든 Duplicates 제거
```

---

## 7. Cross Tabulation

```python
df = pd.DataFrame({
    'gender': ['M', 'F', 'M', 'F', 'M', 'F'],
    'department': ['Sales', 'IT', 'IT', 'Sales', 'HR', 'IT'],
    'salary': [50000, 60000, 55000, 58000, 52000, 62000]
})

# Frequency cross table
ct = pd.crosstab(df['gender'], df['department'])
print(ct)

# Add margins
ct = pd.crosstab(df['gender'], df['department'], margins=True)
print(ct)

# Normalize
ct = pd.crosstab(df['gender'], df['department'], normalize=True)
print(ct)

# Apply aggregation function
ct = pd.crosstab(df['gender'], df['department'],
                 values=df['salary'], aggfunc='mean')
print(ct)
```

---

## Practice Problems

### Problem 1: Group Statistics
Calculate average salary and employee count by department.

```python
df = pd.DataFrame({
    'department': ['Sales', 'IT', 'IT', 'HR', 'Sales'],
    'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'salary': [50000, 60000, 70000, 55000, 65000]
})

# Solution
result = df.groupby('department').agg(
    avg_salary=('salary', 'mean'),
    count=('name', 'count')
)
print(result)
```

### Problem 2: Data Merging
Join two DataFrames to include department names for employees.

```python
employees = pd.DataFrame({
    'emp_id': [1, 2, 3],
    'name': ['Alice', 'Bob', 'Charlie'],
    'dept_id': [10, 20, 10]
})

departments = pd.DataFrame({
    'dept_id': [10, 20],
    'dept_name': ['Sales', 'IT']
})

# Solution
result = pd.merge(employees, departments, on='dept_id')
print(result)
```

### Problem 3: Pivot Table
Create a pivot table with sales totals by month and category.

```python
sales = pd.DataFrame({
    'month': ['Jan', 'Jan', 'Feb', 'Feb', 'Jan', 'Feb'],
    'category': ['A', 'B', 'A', 'B', 'A', 'A'],
    'amount': [100, 150, 120, 180, 110, 130]
})

# Solution
pivot = pd.pivot_table(sales, values='amount',
                       index='month', columns='category',
                       aggfunc='sum')
print(pivot)
```

---

## Summary

| Feature | Functions/Methods |
|------|------------|
| Filtering | `df[condition]`, `query()`, `isin()` |
| Sorting | `sort_values()`, `sort_index()`, `rank()` |
| Grouping | `groupby()`, `agg()`, `transform()`, `apply()` |
| Merging | `merge()`, `join()`, `concat()` |
| Pivot | `pivot()`, `pivot_table()`, `melt()`, `stack()`, `unstack()` |
| Duplicates | `duplicated()`, `drop_duplicates()` |
| Cross Tab | `crosstab()` |
