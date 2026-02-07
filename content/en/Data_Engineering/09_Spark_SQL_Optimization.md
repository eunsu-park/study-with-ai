# Spark SQL Optimization

## Overview

To optimize Spark SQL performance, you need to understand how the Catalyst optimizer works and properly utilize partitioning, caching, join strategies, and other techniques.

---

## 1. Catalyst Optimizer

### 1.1 Understanding Execution Plans

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder.appName("Optimization").getOrCreate()

df = spark.read.parquet("sales.parquet")

# Check execution plan
query = df.filter(col("amount") > 100) \
          .groupBy("category") \
          .sum("amount")

# Logical plan
query.explain(mode="simple")

# Full plan (logical + physical)
query.explain(mode="extended")

# Cost-based plan
query.explain(mode="cost")

# Formatted output
query.explain(mode="formatted")
```

### 1.2 Catalyst Optimization Phases

```
┌─────────────────────────────────────────────────────────────────┐
│                   Catalyst Optimizer Phases                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   1. Analysis                                                   │
│      - Verify column/table names                               │
│      - Type validation                                         │
│      ↓                                                          │
│   2. Logical Optimization                                       │
│      - Predicate Pushdown                                      │
│      - Column Pruning                                          │
│      - Constant Folding                                        │
│      ↓                                                          │
│   3. Physical Planning                                          │
│      - Select join strategy                                    │
│      - Select aggregation strategy                             │
│      ↓                                                          │
│   4. Code Generation                                            │
│      - Whole-Stage Code Generation                             │
│      - JIT compilation                                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 Key Optimization Techniques

```python
# 1. Predicate Pushdown
# Push filter to data source level
df = spark.read.parquet("data.parquet")
filtered = df.filter(col("date") == "2024-01-01")  # Filter directly in Parquet

# 2. Column Pruning
# Read only needed columns
df.select("name", "amount")  # Other columns not read

# 3. Projection Pushdown
# Push SELECT to data source
df = spark.read.format("jdbc") \
    .option("pushDownPredicate", "true") \
    .load()

# 4. Constant Folding
# Pre-compute constant expressions
df.filter(col("value") > 1 + 2)  # Transformed to > 3
```

---

## 2. Partitioning

### 2.1 Partition Concepts

```python
# Check number of partitions
df.rdd.getNumPartitions()

# Repartition
df.repartition(100)                      # Into 100 partitions
df.repartition("date")                   # Partition by column
df.repartition(100, "date", "category")  # Column + number specified

# Reduce partitions (without shuffle)
df.coalesce(10)  # Reduce partitions without shuffle

# Check partition information
def print_partition_info(df):
    print(f"Partitions: {df.rdd.getNumPartitions()}")
    for idx, partition in enumerate(df.rdd.glom().collect()):
        print(f"Partition {idx}: {len(partition)} rows")
```

### 2.2 Partitioning Strategies

```python
# Calculate appropriate number of partitions
"""
Recommended formula:
- Number of partitions = Data size (MB) / 128MB
- Or: Cluster cores * 2~4

Examples:
- 10GB data → 10,000MB / 128MB ≈ 80 partitions
- 100 core cluster → 200~400 partitions
"""

# Set number of partitions
spark.conf.set("spark.sql.shuffle.partitions", 200)

# Range partitioning (sorted partitions)
df.repartitionByRange(100, "date")

# Hash partitioning
df.repartition(100, "user_id")  # Hash based on user_id
```

### 2.3 Partition Storage

```python
# Save by partition
df.write \
    .partitionBy("year", "month") \
    .parquet("output/partitioned_data")

# Resulting directory structure:
# output/partitioned_data/
#   year=2024/
#     month=01/
#       part-00000.parquet
#     month=02/
#       part-00000.parquet

# Read partitioned data (pruning)
df = spark.read.parquet("output/partitioned_data")
# Only reads year=2024, month=01 partition
df.filter((col("year") == 2024) & (col("month") == 1))

# Bucketing (join optimization)
df.write \
    .bucketBy(100, "user_id") \
    .sortBy("timestamp") \
    .saveAsTable("bucketed_table")
```

---

## 3. Caching

### 3.1 Cache Basics

```python
# Cache DataFrame
df.cache()           # Default MEMORY_AND_DISK
df.persist()         # Same

# Specify cache level
from pyspark import StorageLevel

df.persist(StorageLevel.MEMORY_ONLY)           # Memory only
df.persist(StorageLevel.MEMORY_AND_DISK)       # Memory + disk
df.persist(StorageLevel.MEMORY_ONLY_SER)       # Serialized (memory saving)
df.persist(StorageLevel.DISK_ONLY)             # Disk only
df.persist(StorageLevel.MEMORY_AND_DISK_SER)   # Serialized + disk

# Unpersist cache
df.unpersist()

# Check cache status
spark.catalog.isCached("table_name")
```

### 3.2 Caching Strategies

```python
# Caching is effective when:
# 1. Same DataFrame used multiple times
# 2. Reuse after expensive transformations
# 3. Iterative algorithms

# Example: Reuse in multiple aggregations
expensive_df = spark.read.parquet("large_data.parquet") \
    .filter(col("status") == "active") \
    .join(other_df, "key")

expensive_df.cache()

# Reuse in multiple operations
result1 = expensive_df.groupBy("category").count()
result2 = expensive_df.groupBy("region").sum("amount")
result3 = expensive_df.filter(col("amount") > 1000).count()

# Release after completion
expensive_df.unpersist()
```

### 3.3 Cache Monitoring

```python
# Check in Spark UI (http://localhost:4040/storage)

# Programmatic checking
sc = spark.sparkContext

# List cached RDDs
for rdd_id, rdd_info in sc._jsc.sc().getRDDStorageInfo():
    print(f"RDD {rdd_id}: {rdd_info}")

# Clear all caches
spark.catalog.clearCache()
```

---

## 4. Join Strategies

### 4.1 Join Type Characteristics

```python
# Spark join strategies:
join_strategies = {
    "Broadcast Hash Join": {
        "condition": "Small table (< 10MB default)",
        "performance": "Fastest",
        "shuffle": "None (broadcast small table)"
    },
    "Sort Merge Join": {
        "condition": "Join between large tables",
        "performance": "Stable",
        "shuffle": "Shuffle + sort both tables"
    },
    "Shuffle Hash Join": {
        "condition": "When one side is smaller",
        "performance": "Medium",
        "shuffle": "Shuffle both sides"
    },
    "Broadcast Nested Loop Join": {
        "condition": "No join condition (Cross)",
        "performance": "Slow",
        "shuffle": "None (broadcast)"
    }
}
```

### 4.2 Force Broadcast Join

```python
from pyspark.sql.functions import broadcast

# Broadcast hint for small table
large_df.join(broadcast(small_df), "key")

# Adjust threshold via configuration
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", 100 * 1024 * 1024)  # 100MB

# Disable broadcast
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)

# SQL hint
spark.sql("""
    SELECT /*+ BROADCAST(small_table) */
        large_table.*, small_table.name
    FROM large_table
    JOIN small_table ON large_table.id = small_table.id
""")
```

### 4.3 Join Optimization Tips

```python
# 1. Filter before join
# Bad
df1.join(df2, "key").filter(col("status") == "active")

# Good
df1.filter(col("status") == "active").join(df2, "key")


# 2. Match join key data types
# Bad (type mismatch causes implicit casting)
df1.join(df2, df1.id == df2.id)  # id is string vs int

# Good
df1 = df1.withColumn("id", col("id").cast("int"))
df1.join(df2, "id")


# 3. Handle skewed data (Skew Join)
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", True)
spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionFactor", 5)
spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes", "256MB")


# 4. Optimize joins with bucketing
# Bucket tables on creation
df.write.bucketBy(100, "user_id").saveAsTable("users_bucketed")
other_df.write.bucketBy(100, "user_id").saveAsTable("orders_bucketed")

# Join bucketed tables (no shuffle)
spark.table("users_bucketed").join(spark.table("orders_bucketed"), "user_id")
```

---

## 5. Performance Tuning

### 5.1 Configuration Optimization

```python
# Memory settings
spark = SparkSession.builder \
    .config("spark.executor.memory", "8g") \
    .config("spark.executor.memoryOverhead", "2g") \
    .config("spark.driver.memory", "4g") \
    .config("spark.memory.fraction", "0.8") \
    .config("spark.memory.storageFraction", "0.3") \
    .getOrCreate()

# Parallelism settings
spark.conf.set("spark.default.parallelism", 200)
spark.conf.set("spark.sql.shuffle.partitions", 200)

# Adaptive Query Execution (AQE) - Spark 3.0+
spark.conf.set("spark.sql.adaptive.enabled", True)
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", True)
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", True)
spark.conf.set("spark.sql.adaptive.localShuffleReader.enabled", True)

# Serialization
spark.conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

# Dynamic allocation
spark.conf.set("spark.dynamicAllocation.enabled", True)
spark.conf.set("spark.dynamicAllocation.minExecutors", 2)
spark.conf.set("spark.dynamicAllocation.maxExecutors", 100)
```

### 5.2 Data Format Optimization

```python
# Parquet settings
spark.conf.set("spark.sql.parquet.compression.codec", "snappy")  # or zstd
spark.conf.set("spark.sql.parquet.filterPushdown", True)

# File size optimization
spark.conf.set("spark.sql.files.maxPartitionBytes", "128MB")
spark.conf.set("spark.sql.files.openCostInBytes", "4MB")

# Merge small files
spark.conf.set("spark.sql.adaptive.coalescePartitions.parallelismFirst", False)
spark.conf.set("spark.sql.adaptive.advisoryPartitionSizeInBytes", "128MB")

# Verify column pruning
df.select("needed_column1", "needed_column2").explain()
```

### 5.3 Shuffle Optimization

```python
# Optimize shuffle partition count
# Recommended: use AQE for automatic tuning
spark.conf.set("spark.sql.adaptive.enabled", True)

# Manual setting
data_size_gb = 10
partition_size_mb = 128
optimal_partitions = (data_size_gb * 1024) // partition_size_mb
spark.conf.set("spark.sql.shuffle.partitions", optimal_partitions)

# Shuffle compression
spark.conf.set("spark.shuffle.compress", True)

# Minimize shuffle spill
spark.conf.set("spark.shuffle.spill.compress", True)

# External shuffle service
spark.conf.set("spark.shuffle.service.enabled", True)
```

---

## 6. Performance Monitoring

### 6.1 Using Spark UI

```python
# Access Spark UI: http://<driver-host>:4040

# Information by UI tab:
"""
Jobs: Job execution status, time
Stages: Stage details (shuffle, data size)
Storage: Cached RDD/DataFrame
Environment: Configuration values
Executors: Executor status, memory
SQL: SQL query plans
"""

# History server (for completed jobs)
# spark.eventLog.enabled=true
# spark.history.fs.logDirectory=hdfs:///spark-history
```

### 6.2 Programmatic Monitoring

```python
# Measure execution time
import time

start = time.time()
result = df.groupBy("category").count().collect()
end = time.time()
print(f"Execution time: {end - start:.2f} seconds")

# Check shuffle in execution plan
df.explain(mode="formatted")

# Check join strategy in physical plan
# Exchange = shuffle occurs
# BroadcastHashJoin = broadcast join
# SortMergeJoin = sort merge join
```

### 6.3 Metrics Collection

```python
# Estimate DataFrame size
def estimate_size(df):
    """Estimate DataFrame size (bytes)"""
    return df._jdf.queryExecution().optimizedPlan().stats().sizeInBytes()

# Record count per partition
partition_counts = df.rdd.mapPartitions(
    lambda it: [sum(1 for _ in it)]
).collect()

print(f"Min: {min(partition_counts)}, Max: {max(partition_counts)}")
print(f"Skew ratio: {max(partition_counts) / (sum(partition_counts) / len(partition_counts)):.2f}")
```

---

## 7. Common Performance Issues and Solutions

### 7.1 Data Skew

```python
# Problem: Data concentrated in specific keys
# Symptom: Some tasks take much longer

# Solution 1: AQE skew join
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", True)

# Solution 2: Add salt key
from pyspark.sql.functions import rand, floor

num_salts = 10
df_salted = df.withColumn("salt", floor(rand() * num_salts))

# Salted join
result = df_salted.join(
    other_df.crossJoin(
        spark.range(num_salts).withColumnRenamed("id", "salt")
    ),
    ["key", "salt"]
).drop("salt")

# Solution 3: Broadcast (if possible)
result = df.join(broadcast(small_df), "key")
```

### 7.2 OOM (Out of Memory)

```python
# Problem: Memory shortage
# Symptom: OutOfMemoryError

# Solution 1: Increase executor memory
spark.conf.set("spark.executor.memory", "8g")
spark.conf.set("spark.executor.memoryOverhead", "2g")

# Solution 2: Increase partition count (distribute data)
df.repartition(500)

# Solution 3: Release unnecessary caches
spark.catalog.clearCache()

# Solution 4: Reduce broadcast threshold
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "10MB")
```

### 7.3 Excessive Shuffling

```python
# Problem: Network/disk I/O due to shuffle
# Symptom: Increased wait time between stages

# Solution 1: Filter before shuffle
df.filter(col("status") == "active").groupBy("key").count()

# Solution 2: Change partitioning strategy
# Data partitioned by same key can join without shuffle
df1.repartition(100, "key").join(df2.repartition(100, "key"), "key")

# Solution 3: Use bucketing
df.write.bucketBy(100, "key").saveAsTable("bucketed_table")
```

---

## Practice Problems

### Problem 1: Execution Plan Analysis
Analyze the execution plan of a given query and find optimization points.

### Problem 2: Join Optimization
Design the optimal method to join a transaction table with 100 million records and a customer table with 1 million records.

### Problem 3: Skew Handling
Improve aggregation performance when data is concentrated in specific categories.

---

## Summary

| Optimization Area | Techniques |
|-------------------|------------|
| **Catalyst** | Predicate Pushdown, Column Pruning |
| **Partitioning** | repartition, coalesce, partitionBy |
| **Caching** | cache, persist, StorageLevel |
| **Join** | Broadcast, Sort Merge, Bucketing |
| **AQE** | Automatic partition coalescing, skew handling |

---

## References

- [Spark SQL Tuning](https://spark.apache.org/docs/latest/sql-performance-tuning.html)
- [Spark Configuration](https://spark.apache.org/docs/latest/configuration.html)
- [Adaptive Query Execution](https://spark.apache.org/docs/latest/sql-performance-tuning.html#adaptive-query-execution)
