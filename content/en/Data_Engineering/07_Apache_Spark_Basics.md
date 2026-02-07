# Apache Spark Basics

## Overview

Apache Spark is a unified analytics engine for large-scale data processing. It provides faster performance than Hadoop MapReduce through in-memory processing and supports both batch processing and streaming.

---

## 1. Spark Overview

### 1.1 Spark Features

```
┌────────────────────────────────────────────────────────────────┐
│                    Apache Spark Features                        │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   1. Speed                                                     │
│      - 100x faster than Hadoop with in-memory processing       │
│      - 10x faster than disk-based processing                   │
│                                                                │
│   2. Ease of Use                                               │
│      - Supports Python, Scala, Java, R                         │
│      - Provides SQL interface                                  │
│                                                                │
│   3. Generality                                                │
│      - SQL, streaming, ML, graph processing                    │
│      - Diverse workloads with one engine                       │
│                                                                │
│   4. Compatibility                                             │
│      - Various data sources: HDFS, S3, Cassandra, etc.         │
│      - YARN, Kubernetes, Standalone clusters                   │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 1.2 Spark Ecosystem

```
┌─────────────────────────────────────────────────────────────────┐
│                     Spark Ecosystem                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐  │
│   │  Spark SQL │ │ Streaming  │ │   MLlib    │ │  GraphX    │  │
│   │    + DF    │ │ (Structured)│ │(Machine   │ │  (Graph)   │  │
│   │            │ │             │ │ Learning) │ │            │  │
│   └────────────┘ └────────────┘ └────────────┘ └────────────┘  │
│   ─────────────────────────────────────────────────────────────│
│   │                     Spark Core                           │  │
│   │                 (RDD, Task Scheduling)                   │  │
│   ─────────────────────────────────────────────────────────────│
│   ─────────────────────────────────────────────────────────────│
│   │    Standalone    │    YARN    │    Kubernetes    │ Mesos │  │
│   ─────────────────────────────────────────────────────────────│
│   ─────────────────────────────────────────────────────────────│
│   │  HDFS  │   S3   │   GCS   │  Cassandra  │  JDBC  │ etc │  │
│   ─────────────────────────────────────────────────────────────│
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Spark Architecture

### 2.1 Cluster Configuration

```
┌─────────────────────────────────────────────────────────────────┐
│                    Spark Cluster Architecture                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌───────────────────────────────────────────────────────┐    │
│   │                    Driver Program                      │    │
│   │   ┌─────────────────────────────────────────────────┐ │    │
│   │   │              SparkContext                        │ │    │
│   │   │   - Application entry point                      │ │    │
│   │   │   - Connects to cluster                          │ │    │
│   │   │   - Job creation and scheduling                  │ │    │
│   │   └─────────────────────────────────────────────────┘ │    │
│   └───────────────────────────────────────────────────────┘    │
│                              ↓                                  │
│   ┌───────────────────────────────────────────────────────┐    │
│   │                  Cluster Manager                       │    │
│   │       (Standalone, YARN, Kubernetes, Mesos)            │    │
│   └───────────────────────────────────────────────────────┘    │
│                              ↓                                  │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │
│   │   Worker    │  │   Worker    │  │   Worker    │           │
│   │  ┌───────┐  │  │  ┌───────┐  │  │  ┌───────┐  │           │
│   │  │Executor│ │  │  │Executor│ │  │  │Executor│ │           │
│   │  │ Task  │  │  │  │ Task  │  │  │  │ Task  │  │           │
│   │  │ Task  │  │  │  │ Task  │  │  │  │ Task  │  │           │
│   │  │ Cache │  │  │  │ Cache │  │  │  │ Cache │  │           │
│   │  └───────┘  │  │  └───────┘  │  │  └───────┘  │           │
│   └─────────────┘  └─────────────┘  └─────────────┘           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Core Concepts

| Concept | Description |
|---------|-------------|
| **Driver** | Executes main program, creates SparkContext |
| **Executor** | Executes tasks on worker nodes |
| **Task** | Basic unit of execution |
| **Job** | Parallel computation triggered by an action |
| **Stage** | Group of tasks within a job (shuffle boundary) |
| **Partition** | Logical division unit of data |

### 2.3 Execution Flow

```python
"""
Spark execution flow:
1. Create SparkContext in Driver
2. Parse application code
3. Transformations → Create DAG (Directed Acyclic Graph)
4. Create job when action is called
5. Decompose job → Stages → Tasks
6. Cluster Manager assigns tasks to Executors
7. Executors execute tasks
8. Return results to Driver
"""

# Example code flow
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Example").getOrCreate()

# Transformations (Lazy - not executed)
df = spark.read.csv("data.csv", header=True)  # Read plan
df2 = df.filter(df.age > 20)                  # Filter plan
df3 = df2.groupBy("city").count()             # Aggregation plan

# Action (triggers actual execution)
result = df3.collect()  # Create job → Stages → Tasks → Execute
```

---

## 3. RDD (Resilient Distributed Dataset)

### 3.1 RDD Concept

RDD is Spark's fundamental data structure, an immutable distributed collection of data.

```python
from pyspark import SparkContext

sc = SparkContext("local[*]", "RDD Example")

# Ways to create RDD
# 1. From collection
rdd1 = sc.parallelize([1, 2, 3, 4, 5])

# 2. From external data
rdd2 = sc.textFile("data.txt")

# 3. From existing RDD transformation
rdd3 = rdd1.map(lambda x: x * 2)

# RDD properties
"""
R - Resilient: Fault-recoverable (recompute via lineage)
D - Distributed: Distributed across cluster
D - Dataset: Data collection
"""
```

### 3.2 RDD Operations

```python
# Transformations (Lazy)
# - Return new RDD
# - Only create execution plan

rdd = sc.parallelize([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# map: Apply function to each element
mapped = rdd.map(lambda x: x * 2)  # [2, 4, 6, ...]

# filter: Select elements matching condition
filtered = rdd.filter(lambda x: x % 2 == 0)  # [2, 4, 6, 8, 10]

# flatMap: map then flatten
flat = rdd.flatMap(lambda x: [x, x*2])  # [1, 2, 2, 4, 3, 6, ...]

# distinct: Remove duplicates
distinct = rdd.distinct()

# union: Merge two RDDs
union = rdd.union(sc.parallelize([11, 12]))

# groupByKey: Group by key
pairs = sc.parallelize([("a", 1), ("b", 2), ("a", 3)])
grouped = pairs.groupByKey()  # [("a", [1, 3]), ("b", [2])]

# reduceByKey: Reduce by key
reduced = pairs.reduceByKey(lambda a, b: a + b)  # [("a", 4), ("b", 2)]


# Actions (Eager)
# - Return results or save
# - Trigger actual execution

# collect: Return all elements to Driver
result = rdd.collect()  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# count: Count elements
count = rdd.count()  # 10

# first / take: First element / n elements
first = rdd.first()  # 1
take3 = rdd.take(3)  # [1, 2, 3]

# reduce: Reduce all
total = rdd.reduce(lambda a, b: a + b)  # 55

# foreach: Apply function to each element (side effect)
rdd.foreach(lambda x: print(x))

# saveAsTextFile: Save to file
rdd.saveAsTextFile("output/")
```

### 3.3 Pair RDD Operations

```python
# Key-Value pair RDD operations
sales = sc.parallelize([
    ("Electronics", 100),
    ("Clothing", 50),
    ("Electronics", 200),
    ("Clothing", 75),
    ("Food", 30),
])

# Sum by key
total_by_category = sales.reduceByKey(lambda a, b: a + b)
# [("Electronics", 300), ("Clothing", 125), ("Food", 30)]

# Average by key
count_sum = sales.combineByKey(
    lambda v: (v, 1),                      # createCombiner
    lambda acc, v: (acc[0] + v, acc[1] + 1),  # mergeValue
    lambda acc1, acc2: (acc1[0] + acc2[0], acc1[1] + acc2[1])  # mergeCombiner
)
avg_by_category = count_sum.mapValues(lambda x: x[0] / x[1])

# Sort
sorted_rdd = sales.sortByKey()

# Join
inventory = sc.parallelize([
    ("Electronics", 50),
    ("Clothing", 100),
])

joined = sales.join(inventory)
# [("Electronics", (100, 50)), ("Electronics", (200, 50)), ...]
```

---

## 4. Installation and Execution

### 4.1 Local Installation (PySpark)

```bash
# pip installation
pip install pyspark

# Check version
pyspark --version

# Start PySpark shell
pyspark

# Execute script with spark-submit
spark-submit my_script.py
```

### 4.2 Docker Installation

```yaml
# docker-compose.yaml
version: '3'

services:
  spark-master:
    image: bitnami/spark:3.4
    environment:
      - SPARK_MODE=master
    ports:
      - "8080:8080"
      - "7077:7077"

  spark-worker:
    image: bitnami/spark:3.4
    environment:
      - SPARK_MODE=worker
      - SPARK_MASTER_URL=spark://spark-master:7077
    depends_on:
      - spark-master
```

```bash
# Run
docker-compose up -d

# Submit job to cluster
spark-submit --master spark://localhost:7077 my_script.py
```

### 4.3 Cluster Mode

```bash
# Standalone cluster
spark-submit \
    --master spark://master:7077 \
    --deploy-mode cluster \
    --executor-memory 4G \
    --executor-cores 2 \
    --num-executors 10 \
    my_script.py

# YARN cluster
spark-submit \
    --master yarn \
    --deploy-mode cluster \
    --executor-memory 4G \
    my_script.py

# Kubernetes cluster
spark-submit \
    --master k8s://https://k8s-master:6443 \
    --deploy-mode cluster \
    --conf spark.kubernetes.container.image=my-spark-image \
    my_script.py
```

---

## 5. SparkSession

### 5.1 Creating SparkSession

```python
from pyspark.sql import SparkSession

# Basic SparkSession
spark = SparkSession.builder \
    .appName("My Application") \
    .getOrCreate()

# With configuration
spark = SparkSession.builder \
    .appName("My Application") \
    .master("local[*]") \
    .config("spark.sql.shuffle.partitions", 200) \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "2g") \
    .config("spark.sql.adaptive.enabled", "true") \
    .enableHiveSupport() \
    .getOrCreate()

# Access SparkContext
sc = spark.sparkContext

# Check configuration
print(spark.conf.get("spark.sql.shuffle.partitions"))

# Stop session
spark.stop()
```

### 5.2 Common Configurations

```python
# Frequently used configurations
common_configs = {
    # Memory settings
    "spark.executor.memory": "4g",
    "spark.driver.memory": "2g",
    "spark.executor.memoryOverhead": "512m",

    # Parallelism settings
    "spark.executor.cores": "4",
    "spark.default.parallelism": "100",
    "spark.sql.shuffle.partitions": "200",

    # Serialization settings
    "spark.serializer": "org.apache.spark.serializer.KryoSerializer",

    # Adaptive Query Execution (Spark 3.0+)
    "spark.sql.adaptive.enabled": "true",
    "spark.sql.adaptive.coalescePartitions.enabled": "true",
    "spark.sql.adaptive.skewJoin.enabled": "true",

    # Cache settings
    "spark.storage.memoryFraction": "0.6",

    # Shuffle settings
    "spark.shuffle.compress": "true",
}

# Apply configuration example
spark = SparkSession.builder \
    .config("spark.sql.shuffle.partitions", 100) \
    .config("spark.sql.adaptive.enabled", True) \
    .getOrCreate()
```

---

## 6. Basic Examples

### 6.1 Word Count

```python
from pyspark.sql import SparkSession

# Create SparkSession
spark = SparkSession.builder \
    .appName("Word Count") \
    .getOrCreate()

sc = spark.sparkContext

# Read text file
text_rdd = sc.textFile("input.txt")

# Word count logic
word_counts = text_rdd \
    .flatMap(lambda line: line.split()) \
    .map(lambda word: (word.lower(), 1)) \
    .reduceByKey(lambda a, b: a + b) \
    .sortBy(lambda x: x[1], ascending=False)

# Print results
for word, count in word_counts.take(10):
    print(f"{word}: {count}")

# Save to file
word_counts.saveAsTextFile("output/word_counts")

spark.stop()
```

### 6.2 DataFrame Basics

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as _sum, avg

spark = SparkSession.builder.appName("DataFrame Example").getOrCreate()

# Create DataFrame
data = [
    ("Alice", "Engineering", 50000),
    ("Bob", "Engineering", 60000),
    ("Charlie", "Marketing", 45000),
    ("Diana", "Marketing", 55000),
]

df = spark.createDataFrame(data, ["name", "department", "salary"])

# Basic operations
df.show()
df.printSchema()

# Filtering
df.filter(col("salary") > 50000).show()

# Aggregation
df.groupBy("department") \
    .agg(
        _sum("salary").alias("total_salary"),
        avg("salary").alias("avg_salary")
    ) \
    .show()

# Using SQL
df.createOrReplaceTempView("employees")
spark.sql("""
    SELECT department, AVG(salary) as avg_salary
    FROM employees
    GROUP BY department
""").show()

spark.stop()
```

---

## Practice Problems

### Problem 1: Basic RDD Operations
Find the sum of squares of even numbers from 1 to 100.

```python
# Solution
sc = spark.sparkContext
result = sc.parallelize(range(1, 101)) \
    .filter(lambda x: x % 2 == 0) \
    .map(lambda x: x ** 2) \
    .reduce(lambda a, b: a + b)
print(result)  # 171700
```

### Problem 2: Pair RDD
Aggregate log counts by error level from a log file.

```python
# Input: "2024-01-01 ERROR: Connection failed"
logs = sc.textFile("logs.txt")
error_counts = logs \
    .map(lambda line: line.split()[1].replace(":", "")) \
    .map(lambda level: (level, 1)) \
    .reduceByKey(lambda a, b: a + b) \
    .collect()
```

---

## Summary

| Concept | Description |
|---------|-------------|
| **Spark** | Unified engine for large-scale data processing |
| **RDD** | Basic distributed data structure |
| **Transformation** | Creates new RDD (Lazy) |
| **Action** | Returns result (Eager) |
| **Driver** | Main program execution node |
| **Executor** | Task execution worker |

---

## References

- [Apache Spark Documentation](https://spark.apache.org/docs/latest/)
- [PySpark API Reference](https://spark.apache.org/docs/latest/api/python/)
- [Learning Spark (O'Reilly)](https://www.oreilly.com/library/view/learning-spark-2nd/9781492050032/)
