# PySpark DataFrame

## Overview

Spark DataFrame is a high-level API that represents distributed data in table format. It provides SQL-like operations and is automatically optimized through the Catalyst optimizer.

---

## 1. SparkSession and DataFrame Creation

### 1.1 SparkSession Initialization

```python
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, DateType

# Create SparkSession
spark = SparkSession.builder \
    .appName("PySpark DataFrame Tutorial") \
    .config("spark.sql.shuffle.partitions", 100) \
    .config("spark.sql.adaptive.enabled", True) \
    .getOrCreate()

# Check Spark version
print(f"Spark Version: {spark.version}")
```

### 1.2 DataFrame Creation Methods

```python
# Method 1: From Python list
data = [
    ("Alice", 30, "Engineering"),
    ("Bob", 25, "Marketing"),
    ("Charlie", 35, "Engineering"),
]
df1 = spark.createDataFrame(data, ["name", "age", "department"])

# Method 2: With explicit schema
schema = StructType([
    StructField("name", StringType(), nullable=False),
    StructField("age", IntegerType(), nullable=True),
    StructField("department", StringType(), nullable=True),
])
df2 = spark.createDataFrame(data, schema)

# Method 3: From list of dictionaries
dict_data = [
    {"name": "Alice", "age": 30, "department": "Engineering"},
    {"name": "Bob", "age": 25, "department": "Marketing"},
]
df3 = spark.createDataFrame(dict_data)

# Method 4: From Pandas DataFrame
import pandas as pd
pdf = pd.DataFrame(data, columns=["name", "age", "department"])
df4 = spark.createDataFrame(pdf)

# Method 5: From RDD
rdd = spark.sparkContext.parallelize(data)
df5 = rdd.toDF(["name", "age", "department"])
```

### 1.3 Reading DataFrames from Files

```python
# CSV file
df_csv = spark.read.csv(
    "data.csv",
    header=True,           # First row as header
    inferSchema=True,      # Auto infer schema
    sep=",",               # Delimiter
    nullValue="NA",        # NULL representation
    dateFormat="yyyy-MM-dd"
)

# Explicit schema (recommended - better performance)
schema = StructType([
    StructField("id", IntegerType()),
    StructField("name", StringType()),
    StructField("amount", DoubleType()),
    StructField("date", DateType()),
])
df_csv = spark.read.csv("data.csv", header=True, schema=schema)

# Parquet file (recommended - columnar format)
df_parquet = spark.read.parquet("data.parquet")

# JSON file
df_json = spark.read.json("data.json")

# ORC file
df_orc = spark.read.orc("data.orc")

# JDBC (database)
df_jdbc = spark.read.format("jdbc") \
    .option("url", "jdbc:postgresql://localhost:5432/mydb") \
    .option("dbtable", "public.users") \
    .option("user", "user") \
    .option("password", "password") \
    .option("driver", "org.postgresql.Driver") \
    .load()

# Delta Lake
df_delta = spark.read.format("delta").load("path/to/delta")
```

---

## 2. Basic DataFrame Operations

### 2.1 Data Inspection

```python
# Preview data
df.show()           # Top 20 rows
df.show(5)          # Top 5 rows
df.show(truncate=False)  # No column truncation

# Check schema
df.printSchema()
df.dtypes           # [(column_name, type), ...]
df.columns          # Column list

# Statistics
df.describe().show()        # Descriptive statistics
df.summary().show()         # Extended statistics

# Count records
df.count()

# Count unique values
df.select("department").distinct().count()

# First row
df.first()
df.head(5)

# Convert to Pandas (small datasets only)
pdf = df.toPandas()
```

### 2.2 Column Selection

```python
from pyspark.sql.functions import col, lit

# Single column
df.select("name")
df.select(col("name"))
df.select(df.name)
df.select(df["name"])

# Multiple columns
df.select("name", "age")
df.select(["name", "age"])
df.select(col("name"), col("age"))

# All columns + additional column
df.select("*", lit(1).alias("constant"))

# Drop column
df.drop("department")

# Rename column
df.withColumnRenamed("name", "full_name")

# Rename multiple columns
df.toDF("name_new", "age_new", "dept_new")

# Using alias
df.select(col("name").alias("employee_name"))
```

### 2.3 Filtering

```python
from pyspark.sql.functions import col

# Basic filter
df.filter(col("age") > 30)
df.filter(df.age > 30)
df.filter("age > 30")           # SQL expression
df.where(col("age") > 30)       # Same as filter

# Compound conditions
df.filter((col("age") > 25) & (col("department") == "Engineering"))
df.filter((col("age") < 25) | (col("department") == "Marketing"))
df.filter(~(col("age") > 30))   # NOT

# String filters
df.filter(col("name").startswith("A"))
df.filter(col("name").endswith("e"))
df.filter(col("name").contains("li"))
df.filter(col("name").like("%li%"))
df.filter(col("name").rlike("^[A-C].*"))  # Regex

# IN condition
df.filter(col("department").isin(["Engineering", "Marketing"]))

# NULL handling
df.filter(col("age").isNull())
df.filter(col("age").isNotNull())

# BETWEEN
df.filter(col("age").between(25, 35))
```

---

## 3. Transformations

### 3.1 Adding/Modifying Columns

```python
from pyspark.sql.functions import col, lit, when, concat, upper, lower, length

# Add new column
df.withColumn("bonus", col("salary") * 0.1)

# Constant column
df.withColumn("country", lit("USA"))

# Modify existing column
df.withColumn("name", upper(col("name")))

# Conditional column (CASE WHEN)
df.withColumn("age_group",
    when(col("age") < 30, "Young")
    .when(col("age") < 50, "Middle")
    .otherwise("Senior")
)

# Multiple columns at once
df.withColumns({
    "name_upper": upper(col("name")),
    "age_plus_10": col("age") + 10,
})

# String concatenation
df.withColumn("full_info", concat(col("name"), lit(" - "), col("department")))

# Type casting
df.withColumn("age_double", col("age").cast("double"))
df.withColumn("age_string", col("age").cast(StringType()))
```

### 3.2 Aggregation Operations

```python
from pyspark.sql.functions import (
    count, sum as _sum, avg, min as _min, max as _max,
    countDistinct, collect_list, collect_set,
    first, last, stddev, variance
)

# Overall aggregation
df.agg(
    count("*").alias("total_count"),
    _sum("salary").alias("total_salary"),
    avg("salary").alias("avg_salary"),
    _min("salary").alias("min_salary"),
    _max("salary").alias("max_salary"),
).show()

# Group aggregation
df.groupBy("department").agg(
    count("*").alias("employee_count"),
    avg("salary").alias("avg_salary"),
    _sum("salary").alias("total_salary"),
    countDistinct("name").alias("unique_names"),
)

# Multiple column grouping
df.groupBy("department", "age_group").count()

# List/set aggregation
df.groupBy("department").agg(
    collect_list("name").alias("employee_names"),
    collect_set("age").alias("unique_ages"),
)

# Pivot table
df.groupBy("department") \
    .pivot("age_group", ["Young", "Middle", "Senior"]) \
    .agg(count("*"))
```

### 3.3 Sorting

```python
from pyspark.sql.functions import col, asc, desc

# Single column sort
df.orderBy("age")                    # Ascending (default)
df.orderBy(col("age").desc())        # Descending
df.orderBy(desc("age"))

# Multiple column sort
df.orderBy(["department", "age"])
df.orderBy(col("department").asc(), col("age").desc())

# NULL handling
df.orderBy(col("age").asc_nulls_first())
df.orderBy(col("age").desc_nulls_last())

# sort is same as orderBy
df.sort("age")
```

### 3.4 Joins

```python
# Test data
employees = spark.createDataFrame([
    (1, "Alice", 101),
    (2, "Bob", 102),
    (3, "Charlie", 101),
], ["id", "name", "dept_id"])

departments = spark.createDataFrame([
    (101, "Engineering"),
    (102, "Marketing"),
    (103, "Finance"),
], ["dept_id", "dept_name"])

# Inner Join (default)
employees.join(departments, employees.dept_id == departments.dept_id)
employees.join(departments, "dept_id")  # Same column name

# Left Join
employees.join(departments, "dept_id", "left")

# Right Join
employees.join(departments, "dept_id", "right")

# Full Outer Join
employees.join(departments, "dept_id", "full")

# Cross Join (Cartesian)
employees.crossJoin(departments)

# Semi Join (left table only, condition met)
employees.join(departments, "dept_id", "left_semi")

# Anti Join (left table only, condition not met)
employees.join(departments, "dept_id", "left_anti")

# Join with compound conditions
employees.join(
    departments,
    (employees.dept_id == departments.dept_id) & (employees.id > 1),
    "inner"
)
```

---

## 4. Actions

### 4.1 Data Collection

```python
# Collect data to Driver
result = df.collect()           # All data (caution: memory)
result = df.take(10)            # Top 10 rows
result = df.first()             # First row
result = df.head(5)             # Top 5 rows

# Convert to list
ages = df.select("age").rdd.flatMap(lambda x: x).collect()

# To Pandas DataFrame
pdf = df.toPandas()             # Small data only

# Iterator (large data)
for row in df.toLocalIterator():
    print(row)
```

### 4.2 File Writing

```python
# Parquet (recommended)
df.write.parquet("output/data.parquet")

# Specify mode
df.write.mode("overwrite").parquet("output/data.parquet")
# overwrite: Overwrite existing
# append: Append to existing
# ignore: Ignore if exists
# error: Error if exists (default)

# Partitioned save
df.write.partitionBy("date", "department").parquet("output/partitioned")

# CSV
df.write.csv("output/data.csv", header=True)

# JSON
df.write.json("output/data.json")

# Save as single file
df.coalesce(1).write.csv("output/single_file.csv", header=True)

# JDBC (database)
df.write.format("jdbc") \
    .option("url", "jdbc:postgresql://localhost:5432/mydb") \
    .option("dbtable", "public.output_table") \
    .option("user", "user") \
    .option("password", "password") \
    .mode("overwrite") \
    .save()
```

---

## 5. UDF (User Defined Functions)

### 5.1 Basic UDF

```python
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType, IntegerType

# Define Python function
def categorize_age(age):
    if age is None:
        return "Unknown"
    elif age < 30:
        return "Young"
    elif age < 50:
        return "Middle"
    else:
        return "Senior"

# Register UDF (decorator style)
@udf(returnType=StringType())
def categorize_age_udf(age):
    if age is None:
        return "Unknown"
    elif age < 30:
        return "Young"
    elif age < 50:
        return "Middle"
    else:
        return "Senior"

# Register UDF (function style)
categorize_udf = udf(categorize_age, StringType())

# Use
df.withColumn("age_category", categorize_udf(col("age")))
df.withColumn("age_category", categorize_age_udf(col("age")))
```

### 5.2 Pandas UDF (Performance Improvement)

```python
from pyspark.sql.functions import pandas_udf
import pandas as pd

# Scalar Pandas UDF (1:1 mapping)
@pandas_udf(StringType())
def categorize_pandas_udf(age_series: pd.Series) -> pd.Series:
    return age_series.apply(
        lambda x: "Unknown" if x is None
        else "Young" if x < 30
        else "Middle" if x < 50
        else "Senior"
    )

# Use
df.withColumn("age_category", categorize_pandas_udf(col("age")))

# Grouped Pandas UDF (group processing)
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

result_schema = StructType([
    StructField("department", StringType()),
    StructField("avg_salary", DoubleType()),
    StructField("employee_count", IntegerType()),
])

@pandas_udf(result_schema, PandasUDFType.GROUPED_MAP)
def analyze_department(pdf: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "department": [pdf["department"].iloc[0]],
        "avg_salary": [pdf["salary"].mean()],
        "employee_count": [len(pdf)],
    })

# Use
df.groupby("department").apply(analyze_department)
```

### 5.3 Using UDF in SQL

```python
# Register UDF for SQL
spark.udf.register("categorize_age", categorize_age, StringType())

# Use in SQL
df.createOrReplaceTempView("employees")
spark.sql("""
    SELECT name, age, categorize_age(age) as age_category
    FROM employees
""").show()
```

---

## 6. Window Functions

```python
from pyspark.sql.window import Window
from pyspark.sql.functions import (
    row_number, rank, dense_rank,
    lead, lag, sum as _sum, avg,
    first, last, ntile
)

# Define windows
window_dept = Window.partitionBy("department").orderBy("salary")
window_all = Window.orderBy("salary")

# Ranking functions
df.withColumn("row_num", row_number().over(window_dept))
df.withColumn("rank", rank().over(window_dept))
df.withColumn("dense_rank", dense_rank().over(window_dept))
df.withColumn("ntile_4", ntile(4).over(window_dept))

# Previous/next values
df.withColumn("prev_salary", lag("salary", 1).over(window_dept))
df.withColumn("next_salary", lead("salary", 1).over(window_dept))

# Cumulative sum
window_cumsum = Window.partitionBy("department") \
    .orderBy("date") \
    .rowsBetween(Window.unboundedPreceding, Window.currentRow)

df.withColumn("cumsum_salary", _sum("salary").over(window_cumsum))

# Moving average
window_moving = Window.partitionBy("department") \
    .orderBy("date") \
    .rowsBetween(-2, 0)  # Current + previous 2

df.withColumn("moving_avg", avg("salary").over(window_moving))

# First/last value in group
df.withColumn("first_name", first("name").over(window_dept))
df.withColumn("last_name", last("name").over(window_dept))
```

---

## Practice Problems

### Problem 1: Data Transformation
Calculate total sales and average sales by month and category from sales data.

### Problem 2: Window Functions
Rank employees by salary within each department and extract the top 3 highest paid employees per department.

### Problem 3: UDF Writing
Write a UDF that extracts the domain from an email address and apply it.

---

## Summary

| Operation | Description | Example |
|-----------|-------------|---------|
| **select** | Column selection | `df.select("name", "age")` |
| **filter** | Row filtering | `df.filter(col("age") > 30)` |
| **groupBy** | Grouping | `df.groupBy("dept").agg(...)` |
| **join** | Table join | `df1.join(df2, "key")` |
| **orderBy** | Sorting | `df.orderBy(desc("salary"))` |
| **withColumn** | Add/modify column | `df.withColumn("new", ...)` |

---

## References

- [PySpark DataFrame Guide](https://spark.apache.org/docs/latest/sql-getting-started.html)
- [PySpark Functions](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/functions.html)
