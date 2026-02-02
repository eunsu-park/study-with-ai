# Spark SQL 최적화

## 개요

Spark SQL의 성능을 최적화하기 위해서는 Catalyst 옵티마이저의 동작 원리를 이해하고, 파티셔닝, 캐싱, 조인 전략 등을 적절히 활용해야 합니다.

---

## 1. Catalyst Optimizer

### 1.1 실행 계획 이해

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder.appName("Optimization").getOrCreate()

df = spark.read.parquet("sales.parquet")

# 실행 계획 확인
query = df.filter(col("amount") > 100) \
          .groupBy("category") \
          .sum("amount")

# 논리적 계획
query.explain(mode="simple")

# 전체 계획 (논리적 + 물리적)
query.explain(mode="extended")

# 비용 기반 계획
query.explain(mode="cost")

# 형식화된 출력
query.explain(mode="formatted")
```

### 1.2 Catalyst 최적화 단계

```
┌─────────────────────────────────────────────────────────────────┐
│                   Catalyst Optimizer 단계                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   1. Analysis (분석)                                            │
│      - 컬럼/테이블 이름 확인                                     │
│      - 타입 검증                                                │
│      ↓                                                          │
│   2. Logical Optimization (논리적 최적화)                        │
│      - Predicate Pushdown (조건절 푸시다운)                      │
│      - Column Pruning (컬럼 가지치기)                            │
│      - Constant Folding (상수 폴딩)                             │
│      ↓                                                          │
│   3. Physical Planning (물리적 계획)                             │
│      - 조인 전략 선택                                           │
│      - 집계 전략 선택                                           │
│      ↓                                                          │
│   4. Code Generation (코드 생성)                                │
│      - Whole-Stage Code Generation                              │
│      - JIT 컴파일                                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 주요 최적화 기법

```python
# 1. Predicate Pushdown (조건절 푸시다운)
# 필터를 데이터 소스 레벨로 푸시
df = spark.read.parquet("data.parquet")
filtered = df.filter(col("date") == "2024-01-01")  # Parquet에서 직접 필터링

# 2. Column Pruning (컬럼 가지치기)
# 필요한 컬럼만 읽기
df.select("name", "amount")  # 다른 컬럼은 읽지 않음

# 3. Projection Pushdown
# SELECT를 데이터 소스로 푸시
df = spark.read.format("jdbc") \
    .option("pushDownPredicate", "true") \
    .load()

# 4. Constant Folding
# 상수 표현식 미리 계산
df.filter(col("value") > 1 + 2)  # > 3으로 변환
```

---

## 2. 파티셔닝

### 2.1 파티션 개념

```python
# 파티션 수 확인
df.rdd.getNumPartitions()

# 파티션 재분배
df.repartition(100)                      # 100개 파티션으로
df.repartition("date")                   # 컬럼 기준 파티셔닝
df.repartition(100, "date", "category")  # 컬럼 + 수 지정

# 파티션 줄이기 (셔플 없이)
df.coalesce(10)  # 셔플 없이 파티션 축소

# 파티션 정보 확인
def print_partition_info(df):
    print(f"Partitions: {df.rdd.getNumPartitions()}")
    for idx, partition in enumerate(df.rdd.glom().collect()):
        print(f"Partition {idx}: {len(partition)} rows")
```

### 2.2 파티션 전략

```python
# 적절한 파티션 수 계산
"""
권장 공식:
- 파티션 수 = 데이터 크기(MB) / 128MB
- 또는: 클러스터 코어 수 * 2~4

예시:
- 10GB 데이터 → 10,000MB / 128MB ≈ 80 파티션
- 100 코어 클러스터 → 200~400 파티션
"""

# 파티션 수 설정
spark.conf.set("spark.sql.shuffle.partitions", 200)

# 범위 파티셔닝 (정렬된 파티션)
df.repartitionByRange(100, "date")

# 해시 파티셔닝
df.repartition(100, "user_id")  # user_id 기준 해시
```

### 2.3 파티션 저장

```python
# 파티션별 저장
df.write \
    .partitionBy("year", "month") \
    .parquet("output/partitioned_data")

# 결과 디렉토리 구조:
# output/partitioned_data/
#   year=2024/
#     month=01/
#       part-00000.parquet
#     month=02/
#       part-00000.parquet

# 파티션 데이터 읽기 (프루닝)
df = spark.read.parquet("output/partitioned_data")
# year=2024, month=01 파티션만 읽음
df.filter((col("year") == 2024) & (col("month") == 1))

# 버킷팅 (조인 최적화)
df.write \
    .bucketBy(100, "user_id") \
    .sortBy("timestamp") \
    .saveAsTable("bucketed_table")
```

---

## 3. 캐싱

### 3.1 캐시 기본

```python
# DataFrame 캐시
df.cache()           # MEMORY_AND_DISK 기본
df.persist()         # 동일

# 캐시 레벨 지정
from pyspark import StorageLevel

df.persist(StorageLevel.MEMORY_ONLY)           # 메모리만
df.persist(StorageLevel.MEMORY_AND_DISK)       # 메모리 + 디스크
df.persist(StorageLevel.MEMORY_ONLY_SER)       # 직렬화 (메모리 절약)
df.persist(StorageLevel.DISK_ONLY)             # 디스크만
df.persist(StorageLevel.MEMORY_AND_DISK_SER)   # 직렬화 + 디스크

# 캐시 해제
df.unpersist()

# 캐시 상태 확인
spark.catalog.isCached("table_name")
```

### 3.2 캐시 전략

```python
# 캐시가 효과적인 경우:
# 1. 동일 DataFrame을 여러 번 사용
# 2. 비싼 변환 후 재사용
# 3. 반복 알고리즘

# 예시: 여러 집계에서 재사용
expensive_df = spark.read.parquet("large_data.parquet") \
    .filter(col("status") == "active") \
    .join(other_df, "key")

expensive_df.cache()

# 여러 작업에서 재사용
result1 = expensive_df.groupBy("category").count()
result2 = expensive_df.groupBy("region").sum("amount")
result3 = expensive_df.filter(col("amount") > 1000).count()

# 작업 완료 후 해제
expensive_df.unpersist()
```

### 3.3 캐시 모니터링

```python
# Spark UI에서 확인 (http://localhost:4040/storage)

# 프로그래밍 방식 확인
sc = spark.sparkContext

# 캐시된 RDD 목록
for rdd_id, rdd_info in sc._jsc.sc().getRDDStorageInfo():
    print(f"RDD {rdd_id}: {rdd_info}")

# 전체 캐시 클리어
spark.catalog.clearCache()
```

---

## 4. 조인 전략

### 4.1 조인 유형별 특성

```python
# Spark 조인 전략:
join_strategies = {
    "Broadcast Hash Join": {
        "condition": "작은 테이블 (< 10MB 기본)",
        "performance": "가장 빠름",
        "shuffle": "없음 (작은 테이블 브로드캐스트)"
    },
    "Sort Merge Join": {
        "condition": "큰 테이블 간 조인",
        "performance": "안정적",
        "shuffle": "양쪽 테이블 셔플 + 정렬"
    },
    "Shuffle Hash Join": {
        "condition": "한쪽이 작을 때",
        "performance": "중간",
        "shuffle": "양쪽 셔플"
    },
    "Broadcast Nested Loop Join": {
        "condition": "조인 조건 없음 (Cross)",
        "performance": "느림",
        "shuffle": "없음 (브로드캐스트)"
    }
}
```

### 4.2 Broadcast Join 강제

```python
from pyspark.sql.functions import broadcast

# 작은 테이블 브로드캐스트 힌트
large_df.join(broadcast(small_df), "key")

# 설정으로 임계값 조정
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", 100 * 1024 * 1024)  # 100MB

# 브로드캐스트 비활성화
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)

# SQL 힌트
spark.sql("""
    SELECT /*+ BROADCAST(small_table) */
        large_table.*, small_table.name
    FROM large_table
    JOIN small_table ON large_table.id = small_table.id
""")
```

### 4.3 조인 최적화 팁

```python
# 1. 조인 전 필터링
# 나쁜 예
df1.join(df2, "key").filter(col("status") == "active")

# 좋은 예
df1.filter(col("status") == "active").join(df2, "key")


# 2. 조인 키 데이터 타입 일치
# 나쁜 예 (타입 불일치로 암시적 캐스팅)
df1.join(df2, df1.id == df2.id)  # id가 string vs int

# 좋은 예
df1 = df1.withColumn("id", col("id").cast("int"))
df1.join(df2, "id")


# 3. 스큐 데이터 처리 (Skew Join)
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", True)
spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionFactor", 5)
spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes", "256MB")


# 4. 버킷팅으로 조인 최적화
# 테이블 생성 시 버킷팅
df.write.bucketBy(100, "user_id").saveAsTable("users_bucketed")
other_df.write.bucketBy(100, "user_id").saveAsTable("orders_bucketed")

# 버킷팅된 테이블 조인 (셔플 없음)
spark.table("users_bucketed").join(spark.table("orders_bucketed"), "user_id")
```

---

## 5. 성능 튜닝

### 5.1 설정 최적화

```python
# 메모리 설정
spark = SparkSession.builder \
    .config("spark.executor.memory", "8g") \
    .config("spark.executor.memoryOverhead", "2g") \
    .config("spark.driver.memory", "4g") \
    .config("spark.memory.fraction", "0.8") \
    .config("spark.memory.storageFraction", "0.3") \
    .getOrCreate()

# 병렬성 설정
spark.conf.set("spark.default.parallelism", 200)
spark.conf.set("spark.sql.shuffle.partitions", 200)

# Adaptive Query Execution (AQE) - Spark 3.0+
spark.conf.set("spark.sql.adaptive.enabled", True)
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", True)
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", True)
spark.conf.set("spark.sql.adaptive.localShuffleReader.enabled", True)

# 직렬화
spark.conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

# 동적 할당
spark.conf.set("spark.dynamicAllocation.enabled", True)
spark.conf.set("spark.dynamicAllocation.minExecutors", 2)
spark.conf.set("spark.dynamicAllocation.maxExecutors", 100)
```

### 5.2 데이터 형식 최적화

```python
# Parquet 설정
spark.conf.set("spark.sql.parquet.compression.codec", "snappy")  # 또는 zstd
spark.conf.set("spark.sql.parquet.filterPushdown", True)

# 파일 크기 최적화
spark.conf.set("spark.sql.files.maxPartitionBytes", "128MB")
spark.conf.set("spark.sql.files.openCostInBytes", "4MB")

# 작은 파일 병합
spark.conf.set("spark.sql.adaptive.coalescePartitions.parallelismFirst", False)
spark.conf.set("spark.sql.adaptive.advisoryPartitionSizeInBytes", "128MB")

# 컬럼 프루닝 확인
df.select("needed_column1", "needed_column2").explain()
```

### 5.3 셔플 최적화

```python
# 셔플 파티션 수 최적화
# AQE로 자동 조정 권장
spark.conf.set("spark.sql.adaptive.enabled", True)

# 수동 설정 시
data_size_gb = 10
partition_size_mb = 128
optimal_partitions = (data_size_gb * 1024) // partition_size_mb
spark.conf.set("spark.sql.shuffle.partitions", optimal_partitions)

# 셔플 압축
spark.conf.set("spark.shuffle.compress", True)

# 셔플 스필 최소화
spark.conf.set("spark.shuffle.spill.compress", True)

# 셔플 서비스 (외부)
spark.conf.set("spark.shuffle.service.enabled", True)
```

---

## 6. 성능 모니터링

### 6.1 Spark UI 활용

```python
# Spark UI 접근: http://<driver-host>:4040

# UI 탭별 정보:
"""
Jobs: Job 실행 현황, 시간
Stages: Stage별 상세 (셔플, 데이터 크기)
Storage: 캐시된 RDD/DataFrame
Environment: 설정 값
Executors: Executor 상태, 메모리
SQL: SQL 쿼리 계획
"""

# 이력 서버 (완료된 작업)
# spark.eventLog.enabled=true
# spark.history.fs.logDirectory=hdfs:///spark-history
```

### 6.2 프로그래밍 방식 모니터링

```python
# 실행 시간 측정
import time

start = time.time()
result = df.groupBy("category").count().collect()
end = time.time()
print(f"Execution time: {end - start:.2f} seconds")

# 실행 계획에서 셔플 확인
df.explain(mode="formatted")

# 물리적 계획에서 조인 전략 확인
# Exchange = 셔플 발생
# BroadcastHashJoin = 브로드캐스트 조인
# SortMergeJoin = 소트 머지 조인
```

### 6.3 메트릭 수집

```python
# DataFrame 크기 추정
def estimate_size(df):
    """DataFrame 크기 추정 (바이트)"""
    return df._jdf.queryExecution().optimizedPlan().stats().sizeInBytes()

# 파티션별 레코드 수
partition_counts = df.rdd.mapPartitions(
    lambda it: [sum(1 for _ in it)]
).collect()

print(f"Min: {min(partition_counts)}, Max: {max(partition_counts)}")
print(f"Skew ratio: {max(partition_counts) / (sum(partition_counts) / len(partition_counts)):.2f}")
```

---

## 7. 일반적인 성능 문제와 해결

### 7.1 데이터 스큐 (Skew)

```python
# 문제: 특정 키에 데이터 집중
# 증상: 일부 Task만 오래 걸림

# 해결 1: AQE 스큐 조인
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", True)

# 해결 2: 솔트 키 추가
from pyspark.sql.functions import rand, floor

num_salts = 10
df_salted = df.withColumn("salt", floor(rand() * num_salts))

# 솔트 조인
result = df_salted.join(
    other_df.crossJoin(
        spark.range(num_salts).withColumnRenamed("id", "salt")
    ),
    ["key", "salt"]
).drop("salt")

# 해결 3: 브로드캐스트 (가능한 경우)
result = df.join(broadcast(small_df), "key")
```

### 7.2 OOM (Out of Memory)

```python
# 문제: 메모리 부족
# 증상: OutOfMemoryError

# 해결 1: Executor 메모리 증가
spark.conf.set("spark.executor.memory", "8g")
spark.conf.set("spark.executor.memoryOverhead", "2g")

# 해결 2: 파티션 수 증가 (데이터 분산)
df.repartition(500)

# 해결 3: 불필요한 캐시 해제
spark.catalog.clearCache()

# 해결 4: 브로드캐스트 임계값 감소
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "10MB")
```

### 7.3 셔플 과다

```python
# 문제: 셔플로 인한 네트워크/디스크 I/O
# 증상: Stage 간 대기 시간 증가

# 해결 1: 셔플 전 필터링
df.filter(col("status") == "active").groupBy("key").count()

# 해결 2: 파티셔닝 전략 변경
# 같은 키로 파티셔닝된 데이터는 셔플 없이 조인
df1.repartition(100, "key").join(df2.repartition(100, "key"), "key")

# 해결 3: 버킷팅 사용
df.write.bucketBy(100, "key").saveAsTable("bucketed_table")
```

---

## 연습 문제

### 문제 1: 실행 계획 분석
주어진 쿼리의 실행 계획을 분석하고 최적화 포인트를 찾으세요.

### 문제 2: 조인 최적화
1억 건의 트랜잭션 테이블과 100만 건의 고객 테이블을 조인하는 최적의 방법을 설계하세요.

### 문제 3: 스큐 처리
특정 카테고리에 데이터가 집중된 상황에서 집계 성능을 개선하세요.

---

## 요약

| 최적화 영역 | 기법 |
|-------------|------|
| **Catalyst** | Predicate Pushdown, Column Pruning |
| **파티셔닝** | repartition, coalesce, partitionBy |
| **캐싱** | cache, persist, StorageLevel |
| **조인** | Broadcast, Sort Merge, 버킷팅 |
| **AQE** | 자동 파티션 병합, 스큐 처리 |

---

## 참고 자료

- [Spark SQL Tuning](https://spark.apache.org/docs/latest/sql-performance-tuning.html)
- [Spark Configuration](https://spark.apache.org/docs/latest/configuration.html)
- [Adaptive Query Execution](https://spark.apache.org/docs/latest/sql-performance-tuning.html#adaptive-query-execution)
